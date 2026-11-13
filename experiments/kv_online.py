"""Paired CPU GRPO with actual in-flight weight changes and group-wise KV rebuild."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import queue
import statistics
import subprocess
import sys
from time import perf_counter, sleep

from experiments.cpu_baseline import save_fixture
from experiments.kv_replay import _forward, rebuild_cache

import torch
import torch.nn.functional as F
import transformers

from algos import grpo
from core import Batch, Trajectory
from tasks import TokenChoiceTask
from train import Config, make_policy, optimize

GROUP_SIZE = 8
PROMPT_TOKENS = 128
QUEUE_CAPACITY = 2
TAU = 1
LENGTHS = (16, 24, 32)
EVAL_LENGTH = 16
FIRST_CONTENT_ID = 3
MODES = ("recompute", "share_prompt")


def _threads():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _prompt(policy, task, problem):
    ids = policy.tok.encode(task.build_prompt(problem), add_special_tokens=False)
    return torch.tensor([[3, 4] * ((PROMPT_TOKENS - len(ids) + 1) // 2) + ids])[:, -PROMPT_TOKENS:]


class _PublishedWeights:
    """One shared snapshot. Reading the version and replacing weights share a lock."""
    def __init__(self, ctx, model):
        self.vector = torch.nn.utils.parameters_to_vector(model.parameters()).detach().share_memory_()
        self.version = ctx.Value("i", 0)

    @torch.no_grad()
    def publish(self, model):
        with self.version.get_lock():
            self.vector.copy_(torch.nn.utils.parameters_to_vector(model.parameters()))
            self.version.value += 1

    @torch.no_grad()
    def pull(self, model, have):
        with self.version.get_lock():
            version = self.version.value
            if version == have:
                return have
            snapshot = self.vector.clone()
        torch.nn.utils.vector_to_parameters(snapshot, model.parameters())
        return version


@torch.no_grad()
def _generate(policy, task, problem, steps, weights, mode, generator):
    model = policy.model
    prompt = _prompt(policy, task, problem)
    prompt_mask = torch.ones_like(prompt)
    start_sync = perf_counter()
    version = weights.pull(model, -1)
    start_version = version
    weight_sync_ms = (perf_counter() - start_sync) * 1000
    # Keep identical initial generation in both arms. The treatment starts at a swap.
    history = prompt.expand(GROUP_SIZE, -1)
    out = _forward(model, history, prompt_mask.expand(GROUP_SIZE, -1))
    cache = out.past_key_values
    logits = out.logits[:, -1]
    samples, logprobs, versions, switches = [], [], [], []
    for position in range(steps):
        distribution = logits[:, FIRST_CONTENT_ID:].log_softmax(dim=-1)
        action = torch.multinomial(distribution.exp(), 1, generator=generator)
        logprobs.append(distribution.gather(1, action))
        token = action + FIRST_CONTENT_ID
        samples.append(token)
        versions.append(version)
        if position + 1 == steps:
            break

        # Inspect the published version every four tokens, with at most one switch per group.
        mask = torch.ones((GROUP_SIZE, PROMPT_TOKENS + position + 1), dtype=torch.long)
        if position % 4 == 3 and not switches:
            sync_start = perf_counter()
            latest = weights.pull(model, version)
            sync_ms = (perf_counter() - sync_start) * 1000
            weight_sync_ms += sync_ms
            if latest != version:
                # The pending token was sampled with the old weights. Rebuild only tokens
                # that were already in cache, then consume the pending token under new weights.
                suffix = torch.cat(samples[:-1], dim=1)
                restore_start = perf_counter()
                cache, compute_ms, copy_ms = rebuild_cache(
                    model, mode, prompt, suffix, prompt_mask, mask[:, :-1],
                )
                out = _forward(model, token, mask, cache)
                switches.append({
                    "after_tokens": position + 1, "from": version, "to": latest,
                    "recompute_ms": compute_ms, "cache_copy_ms": copy_ms,
                    "recovery_ms": (perf_counter() - restore_start) * 1000,
                    "prompt_tokens_recomputed": PROMPT_TOKENS * (GROUP_SIZE if mode == "recompute" else 1),
                    "suffix_tokens_recomputed": suffix.numel(),
                })
                version = latest
            else:
                out = _forward(model, token, mask, cache)
        else:
            out = _forward(model, token, mask, cache)
        cache, logits = out.past_key_values, out.logits[:, -1]

    completion = torch.cat(samples, dim=1)
    states = torch.cat((history, completion), dim=1)
    answer = policy.tok.convert_tokens_to_ids(problem["answer"])
    rewards = completion[:, -1].eq(answer).float()
    return {
        "prompt": problem["prompt"], "tokens": states.tolist(),
        "behavior_logprobs": torch.cat(logprobs, dim=1).tolist(), "rewards": rewards.tolist(),
        "version_by_token": versions, "initial_version": start_version, "switches": switches,
        "completion_length": steps, "weight_sync_ms": weight_sync_ms,
    }


def _actor(config, seed, mode, weights, results, total_groups):
    _threads()
    policy = make_policy(config, None)
    task = TokenChoiceTask(n_examples=config.n_examples, n_eval=config.eval_n, seed=seed)
    # Each group receives its own independent reproducible random stream.
    put_ms = []
    for group_id in range(total_groups):
        generator = torch.Generator().manual_seed(seed * 100_000 + group_id)
        problem, = task.sample(1)
        start = perf_counter()
        payload = _generate(policy, task, problem, LENGTHS[group_id % len(LENGTHS)],
                            weights, mode, generator)
        payload.update(group_id=group_id, generation_s=perf_counter() - start,
                       generated_at=perf_counter())
        results.put(payload)
        put_ms.append((perf_counter() - payload["generated_at"]) * 1000)
    results.put({"queue_put_ms": put_ms})
    results.close()
    results.join_thread()


def _batch(payload):
    trajectories = []
    for ids, behavior, reward in zip(payload["tokens"], payload["behavior_logprobs"],
                                     payload["rewards"]):
        states = torch.tensor(ids)
        old_logp = F.pad(torch.tensor(behavior), (PROMPT_TOKENS, 0))
        mask = torch.zeros_like(old_logp)
        mask[PROMPT_TOKENS:] = 1
        rewards = torch.zeros_like(old_logp)
        rewards[-1] = reward
        trajectories.append(Trajectory(states, states.clone(), old_logp, rewards, mask))
    return Batch.from_groups([trajectories])


def _scoring(policy):
    """Use the same content-only categorical support as the actor."""
    def logprobs(batch):
        logits = policy.model(batch.states, use_cache=False).logits[:, :-1, FIRST_CONTENT_ID:]
        labels = (batch.states[:, 1:] - FIRST_CONTENT_ID).clamp_min(0)
        result = -F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
        return F.pad(result, (1, 0)) * batch.mask
    policy.logprobs = logprobs


@torch.no_grad()
def _evaluate(policy, task):
    prompts = torch.cat([_prompt(policy, task, row) for row in task.eval_data])
    generated = policy.model.generate(
        input_ids=prompts, attention_mask=torch.ones_like(prompts), do_sample=False,
        suppress_tokens=list(range(FIRST_CONTENT_ID)), max_new_tokens=EVAL_LENGTH,
    )
    predicted = generated[:, -1]
    answers = torch.tensor([policy.tok.convert_tokens_to_ids(row["answer"]) for row in task.eval_data])
    return float(predicted.eq(answers).float().mean())


def _write_line(handle, row):
    handle.write(json.dumps(row, allow_nan=False) + "\n")


def _run(config, seed, mode, total_groups, output, stress_delay_s):
    ctx = mp.get_context("spawn")
    policy = make_policy(config, None)
    _scoring(policy)
    task = TokenChoiceTask(n_examples=config.n_examples, n_eval=config.eval_n, seed=seed)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr)
    weights = _PublishedWeights(ctx, policy.model)
    results = ctx.Queue(maxsize=QUEUE_CAPACITY)
    actor = ctx.Process(target=_actor, args=(config, seed, mode, weights, results, total_groups))
    seen = updates = generated_tokens = loss_tokens = accepted_tokens = 0
    dropped = switched = zero_groups = 0
    lags = {}
    first_eval = _evaluate(policy, task)
    started = perf_counter()
    actor.start()
    if stress_delay_s:
        # Give the actor time to fill the bounded queue before the learner drains it.
        # This acceptance fixture is separate from the timed comparison.
        sleep(stress_delay_s)
    try:
        with (output / "groups.jsonl").open("w") as groups, (output / "eval.jsonl").open("w") as evaluations:
            _write_line(evaluations, {"update": 0, "time_s": 0, "loss_tokens": 0, "reward": first_eval})
            while True:
                try:
                    payload = results.get(timeout=30)
                except queue.Empty:
                    raise RuntimeError(f"actor stopped delivering, exit code={actor.exitcode}") from None
                if "queue_put_ms" in payload:
                    put_ms = payload["queue_put_ms"]
                    break
                if payload["group_id"] != seen:
                    raise RuntimeError("actor group sequence is incomplete")
                seen += 1
                tokens = GROUP_SIZE * payload["completion_length"]
                generated_tokens += tokens
                switched += len(payload["switches"])
                learner_version = weights.version.value
                lag = learner_version - min(payload["version_by_token"])
                if lag < 0:
                    raise RuntimeError("rollout references an unpublished version")
                accepted = lag <= config.max_staleness
                wait_s = perf_counter() - payload.pop("generated_at")
                publish_ms = 0.0
                update_ms = 0.0
                update = None
                diagnostics = None
                signal_tokens = 0
                if accepted:
                    start_update = perf_counter()
                    batch = _batch(payload)
                    advantage = grpo(batch, policy)
                    signal_tokens = int((advantage != 0).sum())
                    zero_groups += signal_tokens == 0
                    accepted_tokens += tokens
                    loss_tokens += signal_tokens  # one-group batches have all or zero signal here
                    diagnostics = optimize(policy, optimizer, batch, advantage, config)
                    update_ms = (perf_counter() - start_update) * 1000
                    start_publish = perf_counter()
                    weights.publish(policy.model)
                    publish_ms = (perf_counter() - start_publish) * 1000
                    updates += 1
                    update = updates
                    lags[str(lag)] = lags.get(str(lag), 0) + 1
                else:
                    dropped += 1
                _write_line(groups, {
                    **payload, "learner_version": learner_version, "lag": lag,
                    "status": "consumed" if accepted else "stale", "update": update,
                    "queue_s_including_backpressure": wait_s, "publish_ms": publish_ms,
                    "update_ms": update_ms,
                    "loss_tokens": signal_tokens, "diagnostics": diagnostics,
                })
                if accepted and updates % 12 == 0:
                    reward = _evaluate(policy, task)
                    _write_line(evaluations, {"update": updates, "time_s": perf_counter() - started,
                                              "loss_tokens": loss_tokens, "reward": reward})
            actor.join(timeout=5)
            if actor.exitcode != 0 or seen != total_groups:
                raise RuntimeError(f"actor exit={actor.exitcode}, groups={seen}/{total_groups}")
            final_reward = _evaluate(policy, task)
            elapsed = perf_counter() - started
            _write_line(evaluations, {"update": updates, "time_s": elapsed,
                                      "loss_tokens": loss_tokens, "reward": final_reward})
            (output / "actor_queue.json").write_text(json.dumps(put_ms) + "\n")
    finally:
        if actor.is_alive():
            actor.terminate()
            actor.join(timeout=5)
        results.close()
    if not switched or not loss_tokens:
        raise RuntimeError(f"online arm failed to exercise swap and gradient: {switched=}, {loss_tokens=}")
    return {
        "seed": seed, "mode": mode, "wall_s": elapsed, "generated_tokens": generated_tokens,
        "generated_tokens_s": generated_tokens / elapsed, "accepted_tokens": accepted_tokens,
        "loss_tokens": loss_tokens, "loss_tokens_s": loss_tokens / elapsed,
        "consumed_groups": updates, "consumed_groups_s": updates / elapsed,
        "dropped_stale_groups": dropped, "dropped_overflow_groups": 0,
        "zero_signal_groups": zero_groups, "lag_histogram_consumed": lags,
        "weight_switches_in_flight": switched, "initial_reward": first_eval, "final_reward": final_reward,
        "queue_put_median_ms": statistics.median(put_ms),
        "queue_put_max_ms": max(put_ms),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=96)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-lag", type=int, default=TAU, choices=(1, 2))
    parser.add_argument("--stress-delay-s", type=float, default=0)
    args = parser.parse_args()
    if args.groups < 12 or len(set(args.seeds)) != len(args.seeds):
        parser.error("use at least 12 groups and distinct seeds")
    if args.stress_delay_s < 0:
        parser.error("--stress-delay-s must be nonnegative")
    _threads()
    output = args.out.resolve()
    output.mkdir(parents=True)
    fixture = output / "model"
    parameters = save_fixture(fixture)
    config = Config(task="token-choice", algo="grpo", model=str(fixture), dtype="float32",
                    group_size=GROUP_SIZE, n_prompts=1, micro_batch=GROUP_SIZE, lr=0.001,
                    n_examples=80, eval_n=16, device="cpu", max_staleness=args.max_lag)
    report = {
        "protocol": {
            "comparison": list(MODES), "selection": "both cross-version recompute modes passed replay parity",
            "seeds": args.seeds, "groups": args.groups, "prompt_tokens": PROMPT_TOKENS,
            "completion_lengths": LENGTHS, "fixed_horizon": True, "first_content_token": FIRST_CONTENT_ID,
            "reward_position": "last", "eval_completion_length": EVAL_LENGTH,
            "queue_capacity": QUEUE_CAPACITY, "max_lag": args.max_lag, "snapshot_slots": 1,
            "publish_every_accepted_groups": 1, "max_swaps_per_group": 1,
            "weight_check_every_tokens": 4, "threads_per_process": 1,
            "initial_learner_stress_delay_s": args.stress_delay_s,
            "train_config": vars(config), "platform": platform.platform(),
            "python": platform.python_version(), "torch": torch.__version__, "parameters": parameters,
            "transformers": transformers.__version__, "logical_cpus": os.cpu_count(),
            "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                           text=True).strip() if sys.platform == "darwin"
            else platform.processor(),
            "source_commit": os.environ.get("NARL_SOURCE_COMMIT") or subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True,
            ).strip(),
            "wall_time_includes": "actor startup, training, periodic and final evaluation, output logging",
        },
        "runs": [],
    }
    (output / "protocol.json").write_text(json.dumps(report["protocol"], indent=2) + "\n")
    for index, seed in enumerate(args.seeds):
        modes = MODES if index % 2 == 0 else MODES[::-1]
        for mode in modes:
            run_dir = output / f"{seed}-{mode}"
            run_dir.mkdir()
            result = _run(config, seed, mode, args.groups, run_dir, args.stress_delay_s)
            report["runs"].append(result)
            (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
