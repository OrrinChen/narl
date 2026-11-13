"""Compare two-group CPU GRPO updates under FIFO and version-aware dispatch."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import queue
import statistics
import subprocess
import sys
from time import perf_counter

import torch

from algos import grpo
from core import Batch
from experiments.cpu_baseline import save_fixture
from experiments.kv_batch_dispatch import BatchDispatch, GROUPS_PER_UPDATE
from experiments.kv_online import (
    FIRST_CONTENT_ID, GROUP_SIZE, PROMPT_TOKENS, _PublishedWeights, _generate, _scoring, _threads,
    _trajectories, _write_line,
)
from experiments.kv_sequence_task import (
    HELDOUT_EXAMPLES, LABELS, TRAIN_EXAMPLES, DevelopmentSequenceTask, SequenceTask, evaluate,
)
from train import Config, make_policy, optimize

ACTORS = 3
MAX_LAG = 1
BUCKETS = ("short", "long")


def _worker(config, seed, actor_id, task_type, problems, dispatch, results):
    _threads()
    policy = make_policy(config, None)
    task = task_type(config.n_examples, config.eval_n, seed)
    while True:
        start = perf_counter()
        reservation = dispatch.acquire()
        dispatched_at = perf_counter()
        if reservation is None:
            results.put({"finished_actor": actor_id})
            break
        group_id, slot, reserved_version = reservation
        problem = problems[group_id]
        generator = torch.Generator().manual_seed(seed * 100_000 + group_id)
        payload = _generate(policy, task, problem, problem["completion_length"], dispatch.weights,
                            "share_prompt", generator, share_initial=True, action_count=len(LABELS))
        payload.update(group_id=group_id, actor_id=actor_id, slot=slot, reserved_version=reserved_version,
                       bucket=problem["bucket"], dispatched_at=dispatched_at, generated_at=perf_counter(),
                       dispatch_wait_s=dispatched_at - start)
        results.put(payload)
    results.close()
    results.join_thread()


def _consume(parked, batch_groups, dispatch, policy, optimizer, config, started, trace):
    for row in parked:
        slot = row["group_id"], row["slot"], row["reserved_version"]
        version = dispatch.weights.version.value
        versions = row["version_by_token"]
        if not row["reserved_version"] <= min(versions) <= max(versions) <= version:
            raise RuntimeError(f"invalid behavior version for group {row['group_id']}")
        lag = version - min(versions)
        if lag > MAX_LAG:
            if dispatch.version_aware:
                raise RuntimeError(f"a reserved group expired: {row['group_id']}")
            dispatch.retire(*slot)
            row.update(status="stale", learner_version=version, lag=lag, loss_tokens=0)
            _write_line(trace, row)
        elif dispatch.may_admit(*slot):
            pending = dispatch.admit(*slot)
            row.update(status="pending", learner_version=version, lag=lag)
            batch_groups.append(row)
            if pending != len(batch_groups):
                raise RuntimeError("learner and dispatcher disagree on the pending batch")
            if pending == GROUPS_PER_UPDATE:
                batch = Batch.from_groups([_trajectories(group) for group in batch_groups])
                advantage = grpo(batch, policy)
                update_start = perf_counter()
                diagnostics = optimize(policy, optimizer, batch, advantage, config)
                dispatch.complete_update(policy.model)
                update_s = perf_counter() - update_start
                for index, group in enumerate(batch_groups):
                    signal = int(advantage[index * GROUP_SIZE:(index + 1) * GROUP_SIZE].count_nonzero())
                    group.update(status="consumed", update=version + 1, loss_tokens=signal)
                    _write_line(trace, group)
                _write_line(trace, {"update": version + 1, "groups": [g["group_id"] for g in batch_groups],
                                    "finished_s": perf_counter() - started, "update_s": update_s,
                                    "diagnostics": diagnostics})
                batch_groups.clear()
        else:
            continue
        parked.remove(row)
        return True
    return False


def _evaluate_at(policy, task, received, received_tokens, weights, started):
    eval_started_s = 0.0 if started is None else perf_counter() - started
    scores = evaluate(policy, task)
    return {"received_groups": received, "received_tokens": received_tokens,
            "update": weights.version.value, "time_s": 0.0 if started is None else perf_counter() - started,
            "eval_started_s": eval_started_s, "accuracy": scores}


def _run(config, task_type, seed, total, window, queue_size, version_aware, eval_every, output):
    ctx = mp.get_context("spawn")
    task = task_type(config.n_examples, config.eval_n, seed)
    problems = task.sample(total)
    policy = make_policy(config, None)
    _scoring(policy, share_learner=True, action_count=len(LABELS))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr)
    weights = _PublishedWeights(ctx, policy.model)
    dispatch = BatchDispatch(ctx, weights, total, window, MAX_LAG, version_aware)
    results = ctx.Queue(maxsize=queue_size)
    actors = [ctx.Process(target=_worker, args=(config, seed, actor_id, task_type, problems, dispatch, results))
              for actor_id in range(ACTORS)]
    seen, finished = set(), set()
    received_tokens, parked, pending = 0, [], []
    initial = _evaluate_at(policy, task, 0, 0, weights, None)
    started = perf_counter()
    try:
        for actor in actors:
            actor.start()
        with (output / "trace.jsonl").open("w") as trace, (output / "eval.jsonl").open("w") as evals:
            _write_line(evals, initial)
            while len(finished) < ACTORS:
                try:
                    row = results.get(timeout=45)
                except queue.Empty:
                    raise RuntimeError(f"rollout queue stalled, actor exits={[p.exitcode for p in actors]}") from None
                if "finished_actor" in row:
                    actor_id = row["finished_actor"]
                    if actor_id in finished or not 0 <= actor_id < ACTORS:
                        raise RuntimeError("invalid actor completion marker")
                    finished.add(actor_id)
                    continue
                group_id = row["group_id"]
                if group_id in seen or not 0 <= group_id < total:
                    raise RuntimeError(f"invalid or duplicate group ID {group_id}")
                problem = problems[group_id]
                for key in ("prompt", "bucket", "completion_length"):
                    if row[key] != problem[key]:
                        raise RuntimeError(f"group {group_id} changed {key}")
                seen.add(group_id)
                received_tokens += GROUP_SIZE * row["completion_length"]
                row.update(receive_order=len(seen), generated_s=row.pop("generated_at") - started,
                           received_s=perf_counter() - started)
                parked.append(row)
                while _consume(parked, pending, dispatch, policy, optimizer, config, started, trace):
                    pass
                if len(seen) % eval_every == 0 and len(seen) < total:
                    _write_line(evals, _evaluate_at(policy, task, len(seen), received_tokens, weights, started))
            if parked or len(seen) != total or (version_aware and pending):
                raise RuntimeError(f"incomplete drain: {len(seen)}/{total}, {len(parked)=}, {len(pending)=}")
            for row in pending:
                row.update(status="unbatched", loss_tokens=0)
                _write_line(trace, row)
            for actor in actors:
                actor.join(timeout=5)
                if actor.exitcode != 0:
                    raise RuntimeError(f"actor exit {actor.exitcode}")
            _write_line(evals, _evaluate_at(policy, task, len(seen), received_tokens, weights, started))
    finally:
        for actor in actors:
            if actor.is_alive():
                actor.terminate()
                actor.join(timeout=5)
                if actor.is_alive():
                    actor.kill()
                    actor.join()
            actor.close()
        results.close()


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _target(evals, bucket, threshold):
    previous = None
    for row in evals[1:]:
        if previous is not None and previous["update"] != row["update"]:
            if min(previous["accuracy"][bucket], row["accuracy"][bucket]) >= threshold:
                return row
        previous = row
    return None


def _measure(output, task, seed, total, window, version_aware, threshold):
    trace = _jsonl(output / "trace.jsonl")
    groups = sorted((row for row in trace if "group_id" in row), key=lambda row: row["group_id"])
    updates = [row for row in trace if "groups" in row]
    if [row["group_id"] for row in groups] != list(range(total)):
        raise RuntimeError("trace has missing or duplicate group IDs")
    consumed_ids = {row["group_id"] for row in groups if row["status"] == "consumed"}
    update_ids = [gid for row in updates for gid in row["groups"]]
    if len(update_ids) != len(consumed_ids) or set(update_ids) != consumed_ids:
        raise RuntimeError("trace has missing or duplicate consumed groups")
    for version, update in enumerate(updates):
        if update["update"] != version + 1 or len(update["groups"]) != GROUPS_PER_UPDATE:
            raise RuntimeError("optimizer version did not advance by exactly one full batch")
        for gid in update["groups"]:
            row = groups[gid]
            if row["learner_version"] != version or row["update"] != version + 1 or not 0 <= row["lag"] <= MAX_LAG:
                raise RuntimeError(f"group {gid} violated the optimizer/version contract")
    for problem, row in zip(task.sample(total), groups):
        length = problem["completion_length"]
        answer = [LABELS.index(token) + FIRST_CONTENT_ID for token in problem["answer_tokens"]]
        if any(row[k] != problem[k] for k in ("prompt", "completion_length", "bucket")):
            raise RuntimeError("trace disagrees with the seeded prompt schedule")
        for sequence, reward, logp in zip(row["tokens"], row["rewards"], row["behavior_logprobs"], strict=True):
            completion = sequence[PROMPT_TOKENS:]
            expected = sum(a == b for a, b in zip(completion, answer, strict=True)) / length
            if len(logp) != length or abs(reward - expected) > 1e-7:
                raise RuntimeError("trace reward or behavior logprob disagrees with the full answer")
    evals = _jsonl(output / "eval.jsonl")
    generated_tokens = sum(row["completion_length"] for row in groups) * GROUP_SIZE
    if evals[-1]["received_tokens"] != generated_tokens or evals[-1]["update"] != len(updates):
        raise RuntimeError("evaluation log and trace disagree on final budget")
    for row in evals:
        row["actor_completed_tokens"] = sum(GROUP_SIZE * group["completion_length"] for group in groups
                                            if group["generated_s"] <= row["time_s"])
    targets = {bucket: _target(evals, bucket, threshold) for bucket in BUCKETS}
    by_bucket = {}
    for bucket in BUCKETS:
        rows = [row for row in groups if row["bucket"] == bucket]
        consumed = [row for row in rows if row["status"] == "consumed"]
        post_swap = [row for row in consumed if row["switches"] and row["loss_tokens"]]
        by_bucket[bucket] = {
            "statuses": dict(Counter(row["status"] for row in rows)), "groups": len(rows),
            "generated_tokens": sum(row["completion_length"] for row in rows) * GROUP_SIZE,
            "post_swap_nonzero_advantage_groups": len(post_swap),
            "exact_post_swap_nonzero_advantage_tokens": sum(
                row["loss_tokens"] // row["completion_length"]
                * (row["completion_length"] - row["switches"][0]["after_tokens"]) for row in post_swap),
        }
    return {"seed": seed, "arm": "version_aware" if version_aware else f"fifo-{window}", "window": window,
            "initial_accuracy": evals[0]["accuracy"], "final_accuracy": evals[-1]["accuracy"],
            "best_accuracy": {bucket: max(row["accuracy"][bucket] for row in evals) for bucket in BUCKETS},
            "target": targets, "wall_s": evals[-1]["time_s"], "generated_tokens": generated_tokens,
            "updates": len(updates), "by_bucket": by_bucket,
            "loss_tokens_s": sum(row["loss_tokens"] for row in groups) / evals[-1]["time_s"]}


def _pairs(runs, windows):
    keyed = {(row["seed"], row["arm"]): row for row in runs}
    comparisons = {}
    for window in windows:
        pairs = []
        for seed, arm in keyed:
            if arm != "version_aware" or (seed, f"fifo-{window}") not in keyed:
                continue
            base, candidate = keyed[seed, f"fifo-{window}"], keyed[seed, arm]
            if base["generated_tokens"] != candidate["generated_tokens"]:
                raise RuntimeError("paired arms generated different token budgets")
            a, b = base["target"]["long"], candidate["target"]["long"]
            pairs.append({"seed": seed, "fifo": a, "version_aware": b,
                          "time_saved_s": a["time_s"] - b["time_s"] if a and b else None,
                          "time_saved_fraction": 1 - b["time_s"] / a["time_s"] if a and b else None,
                          "actor_completed_tokens_saved": a["actor_completed_tokens"] - b["actor_completed_tokens"]
                          if a and b else None})
        complete = [row for row in pairs if row["time_saved_s"] is not None]
        comparisons[f"fifo-{window}"] = {
            "pairs": pairs, "paired_hits": len(complete),
            "fifo_hits": sum(row["fifo"] is not None for row in pairs),
            "version_aware_hits": sum(row["version_aware"] is not None for row in pairs),
            "version_aware_faster": sum(row["time_saved_s"] > 0 for row in complete),
            "median_time_saved_s": statistics.median(row["time_saved_s"] for row in complete) if complete else None,
            "median_time_saved_fraction": statistics.median(row["time_saved_fraction"] for row in complete)
            if complete else None,
            "median_actor_completed_tokens_saved": statistics.median(
                row["actor_completed_tokens_saved"] for row in complete) if complete else None,
        }
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--eval-every", type=int, default=48)
    parser.add_argument("--fifo-windows", type=int, nargs="+", required=True)
    parser.add_argument("--version-window", type=int, required=True)
    args = parser.parse_args()
    if (args.groups < 2 * args.eval_every or args.groups % GROUPS_PER_UPDATE or args.groups >= 100_000
            or args.eval_every < GROUPS_PER_UPDATE or not 0.5 < args.threshold <= 1 or args.lr <= 0
            or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds)
            or min(*args.fifo_windows, args.version_window) < 1
            or len(set(args.fifo_windows)) != len(args.fifo_windows)):
        parser.error("require distinct nonnegative seeds, even group budget, valid threshold, and positive unique windows")
    _threads()
    output = args.out.resolve()
    output.mkdir(parents=True)
    fixture = output / "model"
    parameters = save_fixture(fixture)
    task_type = DevelopmentSequenceTask if args.development else SequenceTask
    config = Config(task="sequence", algo="grpo", model=str(fixture), dtype="float32", group_size=GROUP_SIZE,
                    n_prompts=GROUPS_PER_UPDATE, micro_batch=GROUP_SIZE, lr=args.lr,
                    n_examples=TRAIN_EXAMPLES, eval_n=HELDOUT_EXAMPLES, device="cpu", max_staleness=MAX_LAG)
    queue_size = max(*args.fifo_windows, args.version_window)
    sources = ("experiments/kv_sequence.py", "experiments/kv_batch_dispatch.py",
               "experiments/kv_sequence_task.py", "experiments/kv_online.py", "experiments/shared_prefix.py")
    protocol = {"stage": "development" if args.development else "main", "args": {**vars(args), "out": str(output)},
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "source_sha256": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sources},
                "initial_model_sha256": hashlib.sha256((fixture / "model.safetensors").read_bytes()).hexdigest(),
                "optimizer_groups_per_update": GROUPS_PER_UPDATE, "max_lag": MAX_LAG,
                "actor_count": ACTORS, "torch_threads_per_process": 1, "queue_capacity": queue_size,
                "max_pending_learner_groups": GROUPS_PER_UPDATE,
                "group_size": GROUP_SIZE, "prompt_tokens": PROMPT_TOKENS, "model_parameters": parameters,
                "platform": platform.platform(), "python": sys.version, "torch": torch.__version__,
                "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
                if sys.platform == "darwin" else platform.processor(),
                "train_config": vars(config),
                "train_reward": "fraction of correct answer positions across the entire completion",
                "eval": "greedy exact match by bucket; evaluated every fixed number of received groups, "
                        "timed at the second consecutive post-initial hit at distinct update counts",
                "wall_clock": "actor startup, training, periodic and final eval, trace writes; initial eval untimed",
                "arms": "all use shared prefix and at most one in-flight switch; FIFO uses arrival order"}
    heldout = task_type(config.n_examples, config.eval_n, args.seeds[0]).eval_data
    (output / "heldout.json").write_text(json.dumps(heldout, indent=2) + "\n")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    arms = [(window, False) for window in args.fifo_windows] + [(args.version_window, True)]
    runs = []
    for index, seed in enumerate(args.seeds):
        for window, aware in (arms if index % 2 == 0 else arms[::-1]):
            arm = "version_aware" if aware else f"fifo-{window}"
            path = output / f"{seed}-{arm}"
            path.mkdir()
            _run(config, task_type, seed, args.groups, window, queue_size, aware, args.eval_every, path)
            row = _measure(path, task_type(config.n_examples, config.eval_n, seed), seed, args.groups,
                           window, aware, args.threshold)
            runs.append(row)
            report = {"protocol": protocol, "runs": runs, "comparisons": _pairs(runs, args.fifo_windows)}
            (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
