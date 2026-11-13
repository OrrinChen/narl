"""Measure CPU GRPO time to held-out accuracy on a synthetic token-class task."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from random import Random
import statistics
import subprocess
import sys

import torch
import transformers

from experiments.cpu_baseline import save_fixture
from experiments.kv_online import (FIRST_CONTENT_ID, GROUP_SIZE, MODES, PROMPT_TOKENS,
                                   QUEUE_CAPACITY, _run, _threads)
from tasks import LLMTask, SMOKE_CHOICES, SMOKE_QUERIES
from train import Config


LABELS = ("0", "1")
TRAIN_SHIFTS = (0, 1)
TRAIN_EXAMPLES = 320
HELDOUT_EXAMPLES = 64
LENGTHS = (8, 12, 16)
EVAL_EVERY = 24
PILOT_SEEDS = (100, 101)
MAIN_SEEDS = (0, 1, 2, 3, 4)
MAIN_GROUPS = 384
TARGET_ACCURACY = 0.75
LEARNING_RATE = 0.002


def _examples(n, shifts, filler_start, distractors):
    combinations = [(query, shift) for query in range(len(SMOKE_QUERIES)) for shift in shifts]
    fillers = Random(99 + filler_start).sample(range(filler_start, filler_start + n), n)
    for index, filler_id in enumerate(fillers):
        query, shift = combinations[index % len(combinations)]
        distractor = (query + shift) % len(distractors)
        filler = " ".join(format(filler_id, "010b"))
        operands = (distractors[distractor], SMOKE_QUERIES[query])
        yield {
            "prompt": f"{filler} {operands[0]} {operands[1]}",
            "operands": operands,
            "filler": filler,
            "answer": LABELS[query % 2],
        }


class QueryClassTask(LLMTask):
    """Classify a known query token amid distractors held out from training."""

    heldout_distractors = SMOKE_CHOICES[3:]
    heldout_filler_start = 768

    def __init__(self, n_examples, n_eval, seed):
        heldout = list(_examples(n_eval, (0,), self.heldout_filler_start, self.heldout_distractors))
        train = list(_examples(n_examples, TRAIN_SHIFTS, 0, SMOKE_CHOICES[:2]))
        self._split(heldout + train, n_eval, seed)

    def build_prompt(self, problem):
        return problem["prompt"]


class DevelopmentQueryClassTask(QueryClassTask):
    heldout_distractors = SMOKE_CHOICES[2:3]
    heldout_filler_start = 640


def target_crossing(evaluations, target):
    previous = None
    for row in evaluations:
        if previous is not None and row["update"] != previous["update"]:
            if row["reward"] >= target and previous["reward"] >= target:
                return {"wall_s": row["time_s"],
                        "received_generated_tokens": row["generated_tokens"],
                        "actor_completed_tokens": row["actor_completed_tokens"],
                        "update": row["update"], "accuracy": row["reward"]}
        previous = row
    return None


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _measure(config, task_type, seed, mode, groups, target, out):
    shared = mode == "share_prompt"
    result = _run(config, seed, mode, groups, out, 0, shared, shared, task_type=task_type,
                  lengths=LENGTHS, reward_index=0, eval_length=1, eval_every=EVAL_EVERY,
                  credit_tokens=1, action_count=len(LABELS))
    trace = _read_jsonl(out / "groups.jsonl")
    evaluations = _read_jsonl(out / "eval.jsonl")
    task = task_type(TRAIN_EXAMPLES, HELDOUT_EXAMPLES, seed)
    expected_tokens = GROUP_SIZE * sum(LENGTHS[index % len(LENGTHS)] for index in range(groups))
    if len(trace) != groups or result["generated_tokens"] != expected_tokens:
        raise RuntimeError("incomplete generated-token budget")
    for index, group in enumerate(trace):
        problem, = task.sample(1)
        length = LENGTHS[index % len(LENGTHS)]
        if group["group_id"] != index or group["prompt"] != problem["prompt"]:
            raise RuntimeError("rollout differs from the seeded training prompt stream")
        if (group["completion_length"] != length or len(group["tokens"]) != GROUP_SIZE
                or any(len(row) != PROMPT_TOKENS + length for row in group["tokens"])):
            raise RuntimeError("rollout token shape differs from the frozen schedule")
        output_labels = [row[PROMPT_TOKENS:] for row in group["tokens"]]
        if any(token not in range(FIRST_CONTENT_ID, FIRST_CONTENT_ID + len(LABELS))
               for row in output_labels for token in row):
            raise RuntimeError("rollout token outside the fixed action support")
        answer = FIRST_CONTENT_ID + LABELS.index(problem["answer"])
        rewards = [float(row[0] == answer) for row in output_labels]
        if group["rewards"] != rewards:
            raise RuntimeError("logged reward does not match the first completion token")
        lag = group["learner_version"] - min(group["version_by_token"])
        if lag < 0 or (group["status"] == "consumed") != (lag <= config.max_staleness):
            raise RuntimeError("learner admission disagrees with the recorded behavior versions")
        credit = GROUP_SIZE if group["status"] == "consumed" and len(set(rewards)) > 1 else 0
        if group["loss_tokens"] != credit:
            raise RuntimeError("logged credit does not match the single-token GRPO mask")
    if evaluations[-1]["generated_tokens"] != expected_tokens:
        raise RuntimeError("evaluation trace ended before the generated-token budget")
    final_update = result["consumed_groups"]
    expected_updates = [*range(0, final_update + 1, EVAL_EVERY), final_update]
    if [row["update"] for row in evaluations] != expected_updates:
        raise RuntimeError("evaluation trace differs from the frozen update schedule")
    for row in evaluations:
        if not 0 <= row["reward"] <= 1 or not (row["reward"] * HELDOUT_EXAMPLES).is_integer():
            raise RuntimeError("invalid held-out accuracy")
        row["actor_completed_tokens"] = sum(GROUP_SIZE * group["completion_length"] for group in trace
                                             if group["generated_at_s"] <= row["time_s"])
        if not row["generated_tokens"] <= row["actor_completed_tokens"] <= expected_tokens:
            raise RuntimeError("invalid actor progress at evaluation")
    (out / "eval-audit.json").write_text(json.dumps(evaluations, indent=2) + "\n")
    return {
        **result,
        "target_crossing": target_crossing(evaluations, target),
        "best_heldout_accuracy": max(row["reward"] for row in evaluations),
        "training_prompt_sha256": hashlib.sha256("\n".join(g["prompt"] for g in trace).encode()).hexdigest(),
    }


def _paired(runs):
    pairs = []
    for index in range(0, len(runs) - 1, 2):
        pair = {row["mode"]: row for row in runs[index:index + 2]}
        baseline = pair["recompute"]
        shared = pair["share_prompt"]
        for key in ("seed", "initial_reward", "generated_tokens", "training_prompt_sha256"):
            if baseline[key] != shared[key]:
                raise RuntimeError(f"paired runs disagree on {key}")
        first, second = baseline["target_crossing"], shared["target_crossing"]
        pairs.append({
            "seed": baseline["seed"],
            "baseline_target": first,
            "shared_target": second,
            "target_wall_speedup": first["wall_s"] / second["wall_s"]
            if first and second and first["wall_s"] and second["wall_s"] else None,
            "target_actor_completed_token_delta": second["actor_completed_tokens"] - first["actor_completed_tokens"]
            if first and second else None,
            "budget_wall_speedup": baseline["wall_s"] / shared["wall_s"],
            "useful_token_throughput_change": shared["loss_tokens_s"] / baseline["loss_tokens_s"] - 1,
            "final_accuracy_delta": shared["final_reward"] - baseline["final_reward"],
        })
    return pairs


def _summarize(pairs):
    complete = [pair for pair in pairs if pair["target_wall_speedup"] is not None]
    return {
        "paired_target_reaches": len(complete),
        "target_reaches_by_arm": {
            arm: sum(pair[f"{arm}_target"] is not None for pair in pairs)
            for arm in ("baseline", "shared")
        },
        "target_wall_speedup_median": statistics.median(p["target_wall_speedup"] for p in complete)
        if complete else None,
        "target_actor_completed_token_delta_median": statistics.median(
            p["target_actor_completed_token_delta"] for p in complete)
        if complete else None,
        "budget_wall_speedup_median": statistics.median(p["budget_wall_speedup"] for p in pairs),
        "useful_token_throughput_change_median": statistics.median(p["useful_token_throughput_change"] for p in pairs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true", help="use development seeds and a separate held-out split")
    parser.add_argument("--pilot-groups", type=int)
    parser.add_argument("--pilot-lr", type=float)
    args = parser.parse_args()
    if not args.pilot and (args.pilot_groups is not None or args.pilot_lr is not None):
        parser.error("pilot overrides require --pilot")
    groups = args.pilot_groups if args.pilot_groups is not None else MAIN_GROUPS
    lr = args.pilot_lr if args.pilot_lr is not None else LEARNING_RATE
    if groups < EVAL_EVERY or lr <= 0:
        parser.error("the group budget must cover one evaluation and the learning rate must be positive")
    task_type = DevelopmentQueryClassTask if args.pilot else QueryClassTask
    seeds = PILOT_SEEDS if args.pilot else MAIN_SEEDS
    _threads()
    out = args.out.resolve()
    out.mkdir(parents=True)
    fixture = out / "model"
    parameters = save_fixture(fixture)
    config = Config(task="query-class", algo="grpo", model=str(fixture), dtype="float32",
                    group_size=GROUP_SIZE, n_prompts=1, micro_batch=GROUP_SIZE, lr=lr,
                    n_examples=TRAIN_EXAMPLES, eval_n=HELDOUT_EXAMPLES, device="cpu", max_staleness=1)
    task = task_type(TRAIN_EXAMPLES, HELDOUT_EXAMPLES, seeds[0])
    (out / "heldout.json").write_text(json.dumps(task.eval_data, indent=2) + "\n")
    protocol = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "pilot" if args.pilot else "main",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path("experiments/kv_learning.py"), Path("experiments/kv_online.py"),
                                       Path("experiments/shared_prefix.py"))},
        "seeds": seeds, "groups_per_arm": groups, "group_size": GROUP_SIZE,
        "generated_tokens_per_arm": GROUP_SIZE * sum(LENGTHS[index % len(LENGTHS)] for index in range(groups)),
        "prompt_tokens": PROMPT_TOKENS, "completion_lengths": LENGTHS, "eval_completion_length": 1,
        "reward": "first generated token equals the class of the final input query token",
        "training_credit": "first generated token; remaining tokens exercise rollout and in-flight swaps",
        "task": "synthetic query classification with disjoint held-out distractors and filler strings; "
                "all four query tokens and both labels appear in training; the 64 held-out prompts "
                "are repeated context variations of four unique distractor-query pairs",
        "train_operand_shifts": TRAIN_SHIFTS,
        "training_distractors": SMOKE_CHOICES[:2],
        "heldout_distractors": task_type.heldout_distractors,
        "heldout_size": HELDOUT_EXAMPLES, "training_size": TRAIN_EXAMPLES,
        "heldout_label_counts": {label: sum(row["answer"] == label for row in task.eval_data) for label in LABELS},
        "heldout_unique_operand_pairs": len({row["operands"] for row in task.eval_data}),
        "heldout_sha256": hashlib.sha256((out / "heldout.json").read_bytes()).hexdigest(),
        "initial_model_sha256": hashlib.sha256((fixture / "model.safetensors").read_bytes()).hexdigest(),
        "target_accuracy": TARGET_ACCURACY,
        "target_crossing": "the second of two consecutive greedy evaluations at distinct update counts "
                           "both meeting target; missing means a sustained crossing was not observed",
        "target_token_count": "complete actor-generated groups by the end of evaluation; excludes the "
                              "partially decoded in-flight group; received count recorded separately",
        "action_labels": LABELS,
        "eval_every_admitted_groups": EVAL_EVERY,
        "arm_order": "alternate by seed index",
        "arms": {"recompute": "per-completion prefill, rebuild, and learner scoring",
                 "share_prompt": "share prompt prefill at start and swap, and during learner scoring"},
        "timing": "includes actor startup, online training, periodic/final evaluation and logging; "
                  "initial model load and first evaluation precede the timer",
        "queue_capacity": QUEUE_CAPACITY, "max_lag": config.max_staleness,
        "threads_per_process": 1, "snapshot_slots": 1, "parameters": parameters,
        "platform": platform.platform(), "logical_cpus": os.cpu_count(),
        "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if sys.platform == "darwin" else platform.processor(),
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "train_config": vars(config),
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    results = []
    for index, seed in enumerate(seeds):
        modes = MODES if index % 2 == 0 else MODES[::-1]
        for mode in modes:
            run_dir = out / f"{seed}-{mode}"
            run_dir.mkdir()
            result = _measure(config, task_type, seed, mode, groups, TARGET_ACCURACY, run_dir)
            results.append(result)
            pairs = _paired(results)
            report = {"protocol": protocol, "runs": results, "pairs": pairs,
                      "summary": _summarize(pairs) if pairs else {}}
            (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
