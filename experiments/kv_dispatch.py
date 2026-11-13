"""Compare FIFO and version-aware rollout dispatch on the same CPU budget."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
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
from experiments.cpu_baseline import save_fixture
from experiments.kv_online import (
    GROUP_SIZE, PROMPT_TOKENS, _PublishedWeights, _batch, _evaluate, _generate,
    _scoring, _threads, _write_line,
)
from tasks import TokenChoiceTask
from train import Config, make_policy, optimize

ACTORS = 3
QUEUE_CAPACITY = 2
MAX_LAG = 1
LENGTHS = (8, 24, 64)
VERSION_WINDOW = MAX_LAG + 1


class _Dispatch:
    """A reservation covers generation, queueing, and the learner step."""

    def __init__(self, ctx, weights, total, window, max_lag, version_aware):
        self.condition = ctx.Condition()
        self.next_id = ctx.RawValue("i", 0)
        self.reserved = ctx.RawArray("i", [-1] * window)
        self.weights = weights
        self.total = total
        self.max_lag = max_lag
        self.version_aware = version_aware

    def acquire(self):
        with self.condition:
            while self.next_id.value < self.total:
                active = [version for version in self.reserved if version >= 0]
                version = self.weights.version.value
                # If the oldest reservation finishes last, every other active group,
                # plus the candidate, can publish before it and use one unit of lag.
                would_expire = active and version - min(active) + len(active) > self.max_lag
                if len(active) < len(self.reserved) and not (self.version_aware and would_expire):
                    slot = list(self.reserved).index(-1)
                    self.reserved[slot] = version
                    group_id = self.next_id.value
                    self.next_id.value += 1
                    return group_id, slot, version
                self.condition.wait()
            return None

    def finish(self, slot, reserved_version, model):
        with self.condition:
            if self.reserved[slot] != reserved_version:
                raise RuntimeError("rollout returned a reservation it does not own")
            if model is not None:
                self.weights.publish(model)
            self.reserved[slot] = -1
            self.condition.notify_all()


def _worker(config, seed, actor_id, problems, dispatch, results):
    _threads()
    policy = make_policy(config, None)
    task = TokenChoiceTask(n_examples=config.n_examples, n_eval=config.eval_n, seed=seed)
    while True:
        idle_start = perf_counter()
        reservation = dispatch.acquire()
        dispatched_at = perf_counter()
        if reservation is None:
            results.put({"finished_actor": actor_id, "terminal_idle_s": dispatched_at - idle_start})
            break
        group_id, slot, version = reservation
        generator = torch.Generator().manual_seed(seed * 100_000 + group_id)
        payload = _generate(policy, task, problems[group_id], LENGTHS[group_id % len(LENGTHS)],
                            dispatch.weights, "share_prompt", generator, share_initial=True)
        payload.update(group_id=group_id, actor_id=actor_id, slot=slot, reserved_version=version,
                       idle_s=dispatched_at - idle_start, dispatched_at=dispatched_at,
                       generated_at=perf_counter())
        results.put(payload)
    results.close()
    results.join_thread()


def _summary(records, wall_s):
    buckets = {}
    for length in LENGTHS:
        rows = [row for row in records if row["completion_length"] == length]
        stale = sum(row["status"] == "stale" for row in rows)
        buckets[str(length)] = {"groups": len(rows), "stale_groups": stale,
                                "stale_tokens": stale * length * GROUP_SIZE}
    stale_tokens = sum(bucket["stale_tokens"] for bucket in buckets.values())
    generated_tokens = sum(row["completion_length"] * GROUP_SIZE for row in records)
    consumed = [row for row in records if row["status"] == "consumed"]
    latencies = sorted(row["dispatch_to_finish_s"] for row in consumed)
    signal_tokens = sum(row["loss_tokens"] for row in consumed)
    return {
        "generated_tokens": generated_tokens, "generated_tokens_s": generated_tokens / wall_s,
        "consumed_groups": len(consumed), "consumed_groups_s": len(consumed) / wall_s,
        "accepted_tokens_s": (generated_tokens - stale_tokens) / wall_s,
        "loss_tokens": signal_tokens, "loss_tokens_s": signal_tokens / wall_s,
        "stale_groups": len(records) - len(consumed), "stale_tokens": stale_tokens,
        "p95_consumed_group_s": latencies[math.ceil(0.95 * len(latencies)) - 1],
        "lag_histogram_consumed": dict(sorted(Counter(row["lag"] for row in consumed).items())),
        "buckets": buckets,
    }


def _run(config, seed, groups, window, version_aware, output):
    ctx = mp.get_context("spawn")
    policy = make_policy(config, None)
    _scoring(policy, share_learner=True)
    task = TokenChoiceTask(n_examples=config.n_examples, n_eval=config.eval_n, seed=seed)
    problems = [task.sample(1)[0] for _ in range(groups)]
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr)
    weights = _PublishedWeights(ctx, policy.model)
    dispatch = _Dispatch(ctx, weights, groups, window, config.max_staleness, version_aware)
    results = ctx.Queue(maxsize=QUEUE_CAPACITY)
    actors = [ctx.Process(target=_worker, args=(config, seed, actor_id, problems, dispatch, results))
              for actor_id in range(ACTORS)]
    records, seen, terminal_idle = [], set(), {}
    initial_reward = _evaluate(policy, task)
    started = perf_counter()
    try:
        for actor in actors:
            actor.start()
        with (output / "groups.jsonl").open("w") as trace:
            while len(terminal_idle) < ACTORS:
                try:
                    payload = results.get(timeout=45)
                except queue.Empty:
                    raise RuntimeError(f"rollout queue stalled, actor exits={[p.exitcode for p in actors]}") from None
                if "finished_actor" in payload:
                    actor_id = payload["finished_actor"]
                    if actor_id in terminal_idle:
                        raise RuntimeError("duplicate actor completion")
                    terminal_idle[actor_id] = payload["terminal_idle_s"]
                    continue
                group_id = payload["group_id"]
                if not 0 <= group_id < groups or group_id in seen:
                    raise RuntimeError(f"invalid or duplicate group ID {group_id}")
                seen.add(group_id)
                if payload["prompt"] != problems[group_id]["prompt"] or (
                    payload["completion_length"] != LENGTHS[group_id % len(LENGTHS)]
                ):
                    raise RuntimeError(f"group {group_id} does not match its input")
                version = weights.version.value
                behavior = payload["version_by_token"]
                if not payload["reserved_version"] <= min(behavior) <= max(behavior) <= version:
                    raise RuntimeError(f"group {group_id} has invalid behavior versions")
                lag = version - min(behavior)
                accepted = lag <= config.max_staleness
                if version_aware and not accepted:
                    raise RuntimeError(f"version-aware group {group_id} exceeded its lag budget")
                admitted_at = perf_counter()
                diagnostics, signal_tokens = None, 0
                if accepted:
                    batch = _batch(payload)
                    advantage = grpo(batch, policy)
                    signal_tokens = int((advantage != 0).sum())
                    diagnostics = optimize(policy, optimizer, batch, advantage, config)
                dispatch.finish(payload["slot"], payload["reserved_version"], policy.model if accepted else None)
                row = {
                    **payload, "learner_version": version, "lag": lag,
                    "status": "consumed" if accepted else "stale", "loss_tokens": signal_tokens,
                    "generation_s": payload["generated_at"] - payload["dispatched_at"],
                    "queue_s": admitted_at - payload["generated_at"],
                    "dispatch_to_finish_s": perf_counter() - payload["dispatched_at"],
                    "diagnostics": diagnostics,
                }
                records.append(row)
                _write_line(trace, row)
        completed = perf_counter()
        for actor in actors:
            actor.join(timeout=5)
        if len(seen) != groups or any(actor.exitcode != 0 for actor in actors):
            raise RuntimeError(f"incomplete run: {len(seen)}/{groups}, exits={[p.exitcode for p in actors]}")
        final_reward = _evaluate(policy, task)
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
    wall_s = completed - started
    return {
        "seed": seed, "arm": "version_aware" if version_aware else f"fifo-{window}", "window": window,
        "wall_s": wall_s, "initial_reward": initial_reward, "final_reward": final_reward,
        "actor_idle_s": sum(row["idle_s"] for row in records) + sum(terminal_idle.values()),
        "terminal_idle_s_by_actor": terminal_idle,
        **_summary(records, wall_s),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=384)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--windows", type=int, nargs="+", required=True,
                        help="FIFO reservation windows to compare or tune")
    args = parser.parse_args()
    if (args.groups < 24 or args.groups % len(LENGTHS) or len(set(args.seeds)) != len(args.seeds)
        or min(args.seeds) < 0 or len(set(args.windows)) != len(args.windows)
        or min(args.windows) < 1 or args.groups >= 100_000):
        parser.error("use a multiple of three in [24, 100000), distinct nonnegative seeds and distinct positive windows")
    _threads()
    output = args.out.resolve()
    output.mkdir(parents=True)
    fixture = output / "model"
    parameters = save_fixture(fixture)
    config = Config(task="token-choice", algo="grpo", model=str(fixture), dtype="float32",
                    group_size=GROUP_SIZE, n_prompts=1, micro_batch=GROUP_SIZE, lr=0.001,
                    n_examples=80, eval_n=16, device="cpu", max_staleness=MAX_LAG)
    arms = [(window, False) for window in args.windows] + [(VERSION_WINDOW, True)]
    protocol = {
        "seeds": args.seeds, "groups": args.groups, "actors": ACTORS, "threads_per_process": 1,
        "queue_capacity": QUEUE_CAPACITY, "max_lag": MAX_LAG, "fifo_windows": args.windows,
        "version_aware_window": VERSION_WINDOW, "group_size": GROUP_SIZE,
        "prompt_tokens": PROMPT_TOKENS, "completion_lengths": LENGTHS, "parameters": parameters,
        "wall_time": "actor startup, rollout, training and trace writes; eval outside timer",
        "cache_mode": "shared at start, in-flight rebuild and learner scoring in all arms",
        "policy": "reserve only if all outstanding groups can tolerate worst-case publication order",
        "publication": "once per admitted group, including groups with zero advantage",
        "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
        "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if sys.platform == "darwin" else platform.processor(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                          for path in ("experiments/kv_dispatch.py", "experiments/kv_online.py",
                                       "experiments/shared_prefix.py")},
        "train_config": vars(config),
    }
    report = {"protocol": protocol, "runs": []}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for index, seed in enumerate(args.seeds):
        for window, version_aware in (arms if index % 2 == 0 else arms[::-1]):
            name = "version_aware" if version_aware else f"fifo-{window}"
            run_dir = output / f"{seed}-{name}"
            run_dir.mkdir()
            result = _run(config, seed, args.groups, window, version_aware, run_dir)
            report["runs"].append(result)
            (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
