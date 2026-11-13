"""Check sequence-task learnability on development seeds using synchronous CPU GRPO."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from algos import grpo
from core import Batch
from experiments.cpu_baseline import save_fixture
from experiments.kv_online import GROUP_SIZE, _generate, _scoring, _threads, _trajectories
from experiments.kv_sequence_task import (
    HELDOUT_EXAMPLES, LABELS, PILOT_SEEDS, TRAIN_EXAMPLES, DevelopmentSequenceTask, evaluate,
)
from train import Config, make_policy, optimize


class _SameWeights:
    def pull(self, _model, _have):
        return 0


def _run(config, seed, groups, eval_every):
    task = DevelopmentSequenceTask(config.n_examples, config.eval_n, seed)
    policy = make_policy(config, None)
    _scoring(policy, share_learner=True, action_count=len(LABELS))
    opt = torch.optim.AdamW(policy.parameters(), lr=config.lr)
    weights = _SameWeights()
    useful_groups = {"short": 0, "long": 0}
    curve = [{"groups": 0, **evaluate(policy, task)}]
    for offset in range(0, groups, config.n_prompts):
        rows = task.sample(config.n_prompts)
        samples = []
        for index, problem in enumerate(rows, start=offset):
            generator = torch.Generator().manual_seed(seed * 100_000 + index)
            payload = _generate(policy, task, problem, problem["completion_length"], weights,
                                "share_prompt", generator, share_initial=True, action_count=len(LABELS))
            samples.append(_trajectories(payload))
            useful_groups[problem["bucket"]] += len(set(payload["rewards"])) > 1
        batch = Batch.from_groups(samples)
        optimize(policy, opt, batch, grpo(batch, policy), config)
        seen = offset + len(rows)
        if seen % eval_every == 0 or seen == groups:
            score = {"groups": seen, **evaluate(policy, task)}
            print(json.dumps({"seed": seed, **score}), flush=True)
            curve.append(score)
    return {"seed": seed, "evaluations": curve, "groups_with_reward_variance": useful_groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--groups", default=512, type=int)
    parser.add_argument("--eval-every", default=64, type=int)
    parser.add_argument("--lr", default=0.002, type=float)
    parser.add_argument("--updates-groups", default=2, type=int)
    args = parser.parse_args()
    if args.updates_groups < 1 or args.groups < 1 or args.eval_every < 1 or args.lr <= 0:
        parser.error("budgets and learning rate must be positive")
    if args.groups % args.updates_groups or args.eval_every % args.updates_groups:
        parser.error("group and evaluation budgets must be multiples of groups per update")
    _threads()
    args.out.mkdir(parents=True)
    fixture = args.out / "model"
    save_fixture(fixture)
    config = Config(task="sequence", algo="grpo", model=str(fixture), dtype="float32",
                    group_size=GROUP_SIZE, n_prompts=args.updates_groups, micro_batch=GROUP_SIZE,
                    lr=args.lr, n_examples=TRAIN_EXAMPLES, eval_n=HELDOUT_EXAMPLES, device="cpu")
    report = {
        "stage": "synchronous development pilot", "seeds": PILOT_SEEDS,
        "groups_per_run": args.groups, "evaluate_every_groups": args.eval_every,
        "reward": "fraction of correct completion positions", "evaluation": "greedy exact match by bucket",
        "task_sha256": hashlib.sha256(Path("experiments/kv_sequence_task.py").read_bytes()).hexdigest(),
        "config": vars(config), "runs": [],
    }
    for seed in PILOT_SEEDS:
        report["runs"].append(_run(config, seed, args.groups, args.eval_every))
        (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
