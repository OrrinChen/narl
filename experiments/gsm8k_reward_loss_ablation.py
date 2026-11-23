"""Replay the selected GRPO traces with one reward or loss change on seen Q1."""

import argparse
from pathlib import Path

from safetensors.torch import load_file
import torch

from experiments.audit_gsm8k_regroup import NUMBER
from experiments.audit_kv_online import read_json, read_rows, require
from experiments.gsm8k_budget import evaluate
from experiments.gsm8k_grpo_diagnosis import output_metrics
from experiments.gsm8k_matched import save_summary
from experiments.gsm8k_mlx_development import adapter_snapshot, load_mlx_model, set_adapter
from experiments.gsm8k_mlx_grpo import MlxGrpoLearner
from experiments.gsm8k_online_lag import advantages
from experiments.gsm8k_regroup import adapter_digest
from experiments.gsm8k_signal import TRAIN_SHA256, read_split
from experiments.gsm8k_stable import digest

RECIPE = "lr-0.0001-groups-16"


def fallback_reward(sample, gold):
    if sample["truncated"] or sample["has_answer_tag"]:
        return sample["reward"]
    numbers = NUMBER.findall(sample["completion"].replace(",", ""))
    return int(bool(numbers) and abs(float(numbers[-1]) - gold) < 1e-4)


def replay(model, start, train, rows, updates, arm):
    set_adapter(model, start)
    learner = MlxGrpoLearner(model, 1e-4)
    changes, batches = [], []
    active = active_tokens = switched_on = switched_off = sign_flips = 0
    for update in updates:
        batch = [rows[i] for i in update["group_indices"]]
        lengths = [len(s["token_ids"]) for row in batch for s in row["samples"]]
        mean_length = sum(lengths) / len(lengths)
        groups = []
        for row in batch:
            samples, values = row["samples"], row["advantages"]
            if arm == "reward":
                gold = float(train[row["prompt_id"]]["value"])
                samples = [{**s, "reward": fallback_reward(s, gold)} for s in samples]
                changes.extend(s["id"] for s, old in zip(samples, row["samples"], strict=True)
                               if s["reward"] != old["reward"])
                values, _ = advantages(row["prompt_ids"], samples)
            elif arm == "loss":
                # Scaling each A by L / mean(L) turns the existing per-sequence
                # mean into a sum of token surrogates divided by all batch tokens.
                values = [a * len(s["token_ids"]) / mean_length
                          for a, s in zip(values, samples, strict=True)]
            switched_on += int(any(values) and not any(row["advantages"]))
            switched_off += int(not any(values) and any(row["advantages"]))
            sign_flips += int(any(v * a < 0 for a, v in zip(row["advantages"], values, strict=True)))
            active += int(any(values))
            active_tokens += sum(len(s["token_ids"]) for v, s in zip(values, samples, strict=True) if v)
            groups.append((row["prompt_ids"], samples, values))
        diagnostics = learner.step_groups(groups)
        checkpoint = adapter_digest(adapter_snapshot(model))
        if arm == "baseline":
            require(checkpoint == update["new_sha256"], "baseline batch hash did not reproduce")
        batches.append({"batch": update["batch_index"], "sha256": checkpoint, "diagnostics": diagnostics})
    return {"changed_reward_sample_ids": changes, "active_groups": active, "active_tokens": active_tokens,
            "switched_on_groups": switched_on, "switched_off_groups": switched_off, "reversed_groups": sign_flips,
            "batches": batches, "final_sha256": batches[-1]["sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("runs/gsm8k-stable-v1"))
    parser.add_argument("--source", type=Path, default=Path("runs/gsm8k-budget-5seed"))
    parser.add_argument("--train", type=Path, default=Path("runs/gsm8k-data/train.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    parser.add_argument("--arms", nargs="+", choices=["baseline", "reward", "loss"], default=["reward", "loss"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = read_json(args.base / "plan.json")
    protocol = plan["source_protocol"]
    selection = read_json(args.base / "round-2/selection.json")
    require(selection["selected"] == RECIPE and selection["qualification_index"] == 0, "unexpected selected recipe")
    train = read_split(args.train, TRAIN_SHA256)
    q1 = args.base / "round-2/qualification"
    ids = plan["qualification_prompt_ids"][0]
    original_initial = read_json(q1 / "initial.json")
    require([r["prompt_id"] for r in original_initial["rows"]] == ids and len(ids) == 128, "wrong Q1 questions")
    start = load_file(args.source / "start-adapter.safetensors")
    require(adapter_digest(start) == protocol["start_sha256"], "start adapter differs")
    model, tokenizer = load_mlx_model(None, len(protocol["qualification_protocol"]["lora_layers"]), 92333)
    for seed in args.seeds:
        source = args.base / "models" / RECIPE / f"seed-{seed}"
        training = read_json(source / "training.json")
        rows = read_rows(source / "trace.jsonl")
        updates = read_rows(source / "updates.jsonl")
        original = read_json(q1 / f"seed-{seed}.json")
        require([r["prompt_id"] for r in original["rows"]] == ids, "original endpoint used different questions")
        require(training["trace_sha256"] == digest(source / "trace.jsonl")
                and training["updates_sha256"] == digest(source / "updates.jsonl")
                and sum(row["generated_tokens"] for row in rows) == 32768, "trace provenance mismatch")
        folder = args.out / f"seed-{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        for arm in args.arms:
            result = replay(model, start, train, rows, updates, arm)
            if arm == "baseline":
                require(result["final_sha256"] == training["final_sha256"], "baseline endpoint differs")
                evaluation = original
            else:
                evaluation = evaluate(model, tokenizer, train, ids, protocol)
            diffs = [after["reward"] - before["reward"]
                     for before, after in zip(original["rows"], evaluation["rows"], strict=True)]
            report = {"arm": arm, "seed": seed, "lr": training["lr"], "accumulation_groups": training["accumulation_groups"],
                      "mode": "fixed original trajectories and behavior logprobs; exploratory on previously scored Q1",
                      "reward_rule": "strict unless reward arm: normally ended untagged last numeric token heuristic",
                      "loss_rule": "unweighted per-sequence mean except loss arm: mean over all complete batch tokens",
                      "original_trace_tokens": sum(r["generated_tokens"] for r in rows), "new_rollout_tokens": 0,
                      "trace_sha256": training["trace_sha256"], "updates_sha256": training["updates_sha256"],
                      "plan_sha256": digest(args.base / "plan.json"), "start_sha256": protocol["start_sha256"],
                      "original_endpoint_sha256": training["final_sha256"], "original_correct": original["correct"],
                      "starting_correct": original_initial["correct"], "wins_vs_original": diffs.count(1),
                      "losses_vs_original": diffs.count(-1), "metrics": output_metrics(evaluation),
                      "evaluation": evaluation, **result}
            save_summary(folder / f"{arm}.json", report)
            print(seed, arm, "Q1", evaluation["correct"], "vs original", original["correct"],
                  "active groups", result["active_groups"], "changed rewards", len(result["changed_reward_sample_ids"]),
                  flush=True)


if __name__ == "__main__":
    main()
