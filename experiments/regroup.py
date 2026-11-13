"""Compare GRPO gradients on the same completions with two group assignments."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
import transformers
import yaml

from experiments.cpu_baseline import CONFIG, ROOT, save_fixture
from algos import make_advfn
from core import Batch, Trajectory
from tasks import TokenChoiceTask
from train import Config, SyncSource, loss_denoms, make_policy, optimize, rl_loss, rollout_kwargs
from utils import set_seed

SNAPSHOT_STEPS = (4, 12, 24, 31, 32)
SAMPLING_SEEDS = (101, 202, 303)
REGROUPINGS = 8
NEAR_ZERO = 1e-10
SAMPLER_RESCORE_ATOL = 2e-6


@dataclass(frozen=True)
class Completion:
    id: str
    prompt_index: int
    optimizer_step: int
    sampling_seed: int
    trajectory: Trajectory

    def record(self):
        return {
            "id": self.id, "prompt_index": self.prompt_index,
            "optimizer_step": self.optimizer_step, "sampling_seed": self.sampling_seed,
            "token_ids": self.trajectory.states.tolist(), "mask": self.trajectory.mask.tolist(),
            "behavior_logprobs": self.trajectory.old_logp.tolist(),
            "reward": float(self.trajectory.rewards.sum()),
            "truncated": self.trajectory.truncated,
        }


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_record(handle, data):
    handle.write(json.dumps(data, allow_nan=False) + "\n")


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def collect_snapshots(policy, task, cfg, output):
    set_seed(cfg.seed)
    source = SyncSource(task, cfg, rollout_kwargs(cfg), cfg.n_prompts)
    advantage = make_advfn("grpo", cfg)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    snapshots = {}
    nonzero_steps = 0
    with (output / "snapshot_training.jsonl").open("w") as trace:
        for step in range(1, cfg.steps + 1):
            batch, _ = source.next_batch(policy)
            metrics = optimize(policy, optimizer, batch, advantage(batch, policy), cfg)
            nonzero_steps += int(metrics["gnorm"] > 0)
            write_record(trace, {"step": step, "mean_reward": batch.reward_mean(), **metrics})
            if step in SNAPSHOT_STEPS:
                path = output / f"snapshot_{step:03}.pt"
                state = policy.state_dict()
                torch.save(state, path)
                snapshots[step] = {"path": path.name, "tensor_sha256": tensor_hash(state),
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if len({snap["tensor_sha256"] for snap in snapshots.values()}) != len(SNAPSHOT_STEPS):
        raise RuntimeError("Snapshot training did not produce distinct weights at every checkpoint")
    return snapshots, nonzero_steps


@torch.inference_mode()
def sample_completions(policy, task, prompts, cfg, step, seed, stream):
    """Save the selected token's logprob from generate's actual sampling scores."""
    records = []
    policy.eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for prompt_index, problem in enumerate(prompts):
            prompt_text = task.prompt_text(policy, problem)
            prompt_ids = policy.tok(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids
            inputs = prompt_ids.expand(cfg.group_size, -1).contiguous()
            result = policy.model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs),
                do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                max_new_tokens=cfg.max_new_tokens, use_cache=True,
                pad_token_id=policy.pad_id, return_dict_in_generate=True, output_scores=True,
            )
            prefix_length = inputs.shape[1]
            for sample_index, row in enumerate(result.sequences):
                completion = row[prefix_length:]
                stop = next((i for i, token in enumerate(completion.tolist())
                             if token in policy.stop_ids), None)
                if stop is not None:
                    completion = completion[: stop + 1]
                ids = torch.cat((row[:prefix_length], completion))
                logprobs = torch.zeros(len(ids))
                for i, token in enumerate(completion):
                    logprobs[prefix_length + i] = result.scores[i][sample_index].log_softmax(-1)[token]
                mask = torch.zeros_like(logprobs)
                mask[prefix_length:] = 1
                reward = torch.zeros_like(logprobs)
                text = policy.tok.decode(completion, skip_special_tokens=True)
                reward[-1] = sum(weight * fn(prompt_text, text, problem) for fn, weight in task.reward_fns)
                record_id = f"s{seed}:{stream}:v{step}:p{prompt_index}:r{sample_index}"
                records.append(Completion(record_id, prompt_index, step, seed,
                                          Trajectory(ids, ids, logprobs, reward, mask, stop is None)))
    batch = contiguous_batch(records, cfg.group_size)
    active = batch.mask.bool()
    rescore_error = float((policy.logprobs(batch) - batch.old_logp).abs()[active].max())
    if not rescore_error <= SAMPLER_RESCORE_ATOL:
        raise RuntimeError(f"Sampling and forward scoring differ by {rescore_error:.3g} for {stream} v{step}")
    return records, rescore_error


def contiguous_batch(records, size):
    return Batch.from_groups([[record.trajectory for record in records[start : start + size]]
                              for start in range(0, len(records), size)])


def group_orders(n_prompts, size, seed):
    """Index the canonical sequence [older G, current G] for each prompt."""
    rng = random.Random(seed)
    groups = []
    for prompt_index in range(n_prompts):
        start = 2 * size * prompt_index
        older = list(range(start, start + size))
        current = list(range(start + size, start + 2 * size))
        rng.shuffle(older)
        rng.shuffle(current)
        half = size // 2
        groups.extend((older[:half] + current[:half], older[half:] + current[half:]))
    return torch.tensor(groups)


def regrouped_advantage(records, canonical, order, advantage, policy, *, mixed):
    flat_order = order.flatten()
    inverse = torch.argsort(flat_order)
    if not torch.equal(flat_order[inverse], torch.arange(len(records))):
        raise AssertionError("Group assignment duplicated or lost a completion")
    assignments = [[records[index] for index in group] for group in order.tolist()]
    for group in assignments:
        if len({record.prompt_index for record in group}) != 1:
            raise AssertionError("Group combined completions from different prompts")
        first_stream = group[0].sampling_seed
        expected_count = len(group) // 2 if mixed else len(group)
        if sum(record.sampling_seed == first_stream for record in group) != expected_count:
            raise AssertionError("Group does not have the requested source-cohort composition")
    grouped = Batch.from_groups([[record.trajectory for record in group] for group in assignments])
    for field in ("states", "actions", "rewards", "mask", "old_logp", "truncated"):
        if not torch.equal(getattr(grouped, field)[inverse], getattr(canonical, field)):
            raise AssertionError(f"Regrouping changed {field}")
    grouped_adv = advantage(grouped, policy)
    zero_groups = ~grouped_adv.reshape(*order.shape, -1).ne(0).any(dim=(1, 2))
    return grouped_adv[inverse], float(zero_groups.double().mean())


def gradient(policy, batch, advantage, cfg, denominator):
    loss, _ = rl_loss(policy, batch, advantage, cfg, denoms=denominator)
    gradients = torch.autograd.grad(loss, tuple(policy.parameters()))
    vector = torch.cat([grad.detach().flatten().double() for grad in gradients])
    if not torch.isfinite(vector).all():
        raise RuntimeError("Non-finite GRPO gradient")
    return float(loss.detach()), vector


@torch.inference_mode()
def history_distributions(policy, batch):
    """Next-token distribution on common histories before each decision token."""
    logits = policy.model(batch.states, use_cache=False).logits[:, :-1]
    return logits[batch.mask[:, 1:].bool()].double().log_softmax(-1)


def compare_pair(policy, older, current, cfg, sample_seed, pair_name, trace):
    size = cfg.group_size
    n_prompts = len(current) // size
    records = []
    for start in range(0, len(current), size):
        records.extend(older[start : start + size] + current[start : start + size])
    if len({record.id for record in records}) != len(records):
        raise AssertionError("Completion IDs must be unique within a pair")
    homogeneous = torch.arange(len(records)).reshape(-1, size)
    batch = contiguous_batch(records, size)
    denominator = loss_denoms(batch, cfg)
    advantage = make_advfn("grpo", cfg)
    pure_adv, pure_zero = regrouped_advantage(records, batch, homogeneous, advantage, policy, mixed=False)
    pure_loss, pure_gradient = gradient(policy, batch, pure_adv, cfg, denominator)
    pure_norm = float(pure_gradient.norm())
    active_length = batch.mask.sum(-1)
    pure_by_id = pure_adv.sum(-1) / active_length

    observations = []
    for repetition in range(REGROUPINGS):
        grouping_seed = 10_000 + 16 * sample_seed + repetition
        mixed = group_orders(n_prompts, size, grouping_seed)
        mixed_adv, mixed_zero = regrouped_advantage(records, batch, mixed, advantage, policy, mixed=True)
        mixed_loss, mixed_gradient = gradient(policy, batch, mixed_adv, cfg, denominator)
        mixed_norm = float(mixed_gradient.norm())
        mixed_by_id = mixed_adv.sum(-1) / active_length
        difference = (mixed_by_id - pure_by_id).abs()
        row = {
            "pair": pair_name, "sample_seed": sample_seed, "grouping_seed": grouping_seed,
            "sample_count": len(records), "active_tokens": int(active_length.sum()),
            "mean_abs_advantage_change": float(difference.double().mean()),
            "max_abs_advantage_change": float(difference.max()),
            "zero_signal_group_fraction_homogeneous": pure_zero,
            "zero_signal_group_fraction_mixed": mixed_zero,
            "advantage_became_zero": int((pure_by_id.ne(0) & mixed_by_id.eq(0)).sum()),
            "advantage_left_zero": int((pure_by_id.eq(0) & mixed_by_id.ne(0)).sum()),
            "loss_homogeneous": pure_loss, "loss_mixed": mixed_loss,
            "gradient_norm_homogeneous": pure_norm, "gradient_norm_mixed": mixed_norm,
            "gradient_norm_abs_change": abs(mixed_norm - pure_norm),
            "gradient_difference_norm": float((mixed_gradient - pure_gradient).norm()),
            "gradient_cosine": float(torch.dot(pure_gradient, mixed_gradient) / (pure_norm * mixed_norm))
            if min(pure_norm, mixed_norm) > NEAR_ZERO else None,
            "gradient_near_zero_homogeneous": pure_norm <= NEAR_ZERO,
            "gradient_near_zero_mixed": mixed_norm <= NEAR_ZERO,
        }
        observations.append(row)
        write_record(trace, {**row, "completion_ids": [record.id for record in records],
                             "homogeneous_group_indices": homogeneous.tolist(), "mixed_group_indices": mixed.tolist(),
                             "homogeneous_advantages": pure_by_id.tolist(), "mixed_advantages": mixed_by_id.tolist()})
    return observations


def summarize(rows):
    metrics = ("mean_abs_advantage_change", "zero_signal_group_fraction_homogeneous",
               "zero_signal_group_fraction_mixed", "gradient_cosine", "gradient_difference_norm",
               "gradient_norm_abs_change")
    result = {}
    for key in metrics:
        values = [row[key] for row in rows if row[key] is not None]
        result[key] = {"median": statistics.median(values), "min": min(values), "max": max(values)} if values else None
    result["near_zero_gradient_pairs"] = sum(row["gradient_cosine"] is None for row in rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new directory for experiment records")
    parser.add_argument("--seed", type=int, default=0, help="snapshot training seed")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = args.out.resolve()
    output.mkdir(parents=True)
    fixture = output / "model"
    parameter_count = save_fixture(fixture)
    base = Config(**yaml.safe_load(CONFIG.read_text()), model=str(fixture), seed=args.seed)
    cfg = replace(base, steps=SNAPSHOT_STEPS[-1], eval_every=0)
    task = TokenChoiceTask(cfg.n_examples, seed=cfg.seed, n_eval=cfg.eval_n)
    policy = make_policy(cfg, task).eval()
    snapshots, nonzero_steps = collect_snapshots(policy, task, cfg, output)
    prompts = task.eval_data
    write_json(output / "prompts.json", prompts)
    write_json(output / "snapshot_config.json", asdict(cfg))
    states = {step: torch.load(output / snap["path"], map_location="cpu", weights_only=True)
              for step, snap in snapshots.items()}
    current_step = SNAPSHOT_STEPS[-1]
    pairs = {}
    max_rescore_error = 0.0

    with (output / "completions.jsonl").open("w") as completion_trace, (output / "pairs.jsonl").open("w") as trace:
        for seed in SAMPLING_SEEDS:
            policy.load_state_dict(states[current_step])
            current, error = sample_completions(policy, task, prompts, cfg, current_step, seed + 1_000, "current")
            max_rescore_error = max(max_rescore_error, error)
            for record in current:
                write_record(completion_trace, record.record())
            current_batch = contiguous_batch(current, cfg.group_size)
            new_logp = history_distributions(policy, current_batch)
            for older_step in SNAPSHOT_STEPS:
                policy.load_state_dict(states[older_step])
                older, error = sample_completions(policy, task, prompts, cfg, older_step, seed, "older")
                max_rescore_error = max(max_rescore_error, error)
                for record in older:
                    write_record(completion_trace, record.record())
                old_logp = history_distributions(policy, current_batch)
                kl = (old_logp.exp() * (old_logp - new_logp)).sum(-1).clamp_min(0)
                policy.load_state_dict(states[current_step])
                name = f"v{older_step}-v{current_step}"
                pair = pairs.setdefault(name, {"older_step": older_step, "current_step": current_step,
                                               "lag_optimizer_steps": current_step - older_step,
                                               "same_weights": older_step == current_step,
                                               "drift_by_seed": [], "observations": []})
                pair["drift_by_seed"].append({"sample_seed": seed, "mean_kl_old_to_current_nats": float(kl.mean()),
                                              "max_kl_old_to_current_nats": float(kl.max()), "prefixes": len(kl)})
                pair["observations"].extend(compare_pair(policy, older, current, cfg, seed, name, trace))

    null_name = f"v{current_step}-v{current_step}"
    null_rows = {(row["sample_seed"], row["grouping_seed"]): row
                 for row in pairs[null_name]["observations"]}
    for name, pair in pairs.items():
        rows = pair.pop("observations")
        pair["summary_over_regroupings"] = summarize(rows)
        pair["summary_by_sampling_seed"] = {seed: summarize([row for row in rows if row["sample_seed"] == seed])
                                            for seed in SAMPLING_SEEDS}
        if name != null_name:
            pair["mean_abs_advantage_change_minus_null_by_seed"] = {
                seed: statistics.median(
                    row["mean_abs_advantage_change"] - null_rows[seed, row["grouping_seed"]]["mean_abs_advantage_change"]
                    for row in rows if row["sample_seed"] == seed
                ) for seed in SAMPLING_SEEDS
            }
    write_json(output / "summary.json", {
        "machine": platform.platform(), "cpu": platform.processor(),
        "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "repo_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "experiment_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parameters": parameter_count, "snapshot_training_seed": cfg.seed, "sampling_seeds": SAMPLING_SEEDS,
        "snapshot_training_nonzero_updates": nonzero_steps,
        "current_stream_seed_offset": 1_000, "regroupings_per_sampling_seed": REGROUPINGS,
        "prompts": len(prompts), "group_size": cfg.group_size, "reward": "binary first-token exact match",
        "max_new_tokens": cfg.max_new_tokens, "snapshots": snapshots, "gradient_near_zero_threshold": NEAR_ZERO,
        "sampler_rescore_atol": SAMPLER_RESCORE_ATOL, "max_sampler_rescore_error": max_rescore_error,
        "kl_definition": "Exact conditional KL(old || current) on shared teacher-forced histories from current completions.",
        "loss_definition": "nanoRL clipped GRPO, using captured sampler logprobs, a fixed learner and loss denominator.",
        "pairs": pairs,
    })
    print(f"Saved {len(pairs)} snapshot pairs × {len(SAMPLING_SEEDS)} sampling seeds × {REGROUPINGS} groupings to {output}")


if __name__ == "__main__":
    main()
