"""Measure one verified GRPO step's change in preference for sampled correct answers."""

import argparse
from pathlib import Path
from statistics import mean, median

import numpy as np
from safetensors.torch import load_file
import torch

from experiments.audit_kv_online import read_json, read_rows, require
from experiments.gsm8k_grpo_diagnosis import selected_logps
from experiments.gsm8k_matched import save_summary
from experiments.gsm8k_mlx_development import adapter_snapshot, load_mlx_model, set_adapter
from experiments.gsm8k_mlx_grpo import MlxGrpoLearner
from experiments.gsm8k_regroup import adapter_digest


def margin(model, row):
    positive, negative = sorted(row["samples"], key=lambda sample: -sample["reward"])
    require((positive["reward"], negative["reward"]) == (1, 0), "expected a discordant group")
    values = selected_logps(model, row["prompt_ids"], [positive, negative])
    split = len(positive["token_ids"])
    return float(values[:split].mean(dtype=np.float64) - values[split:].mean(dtype=np.float64))


def summarize(deltas):
    return {"count": len(deltas), "positive": sum(x > 0 for x in deltas), "mean": mean(deltas),
            "median": median(deltas)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("runs/gsm8k-stable-v1"))
    parser.add_argument("--source", type=Path, default=Path("runs/gsm8k-budget-5seed"))
    parser.add_argument("--out", type=Path, default=Path("runs/gsm8k-reward-loss-probe/step.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = read_json(args.base / "plan.json")
    recipe = read_json(args.base / "round-2/selection.json")["selected"]
    require(recipe == "lr-0.0001-groups-16", "unexpected selected recipe")
    seeds = plan["seeds"]
    start = load_file(args.source / "start-adapter.safetensors")
    initial_hash = adapter_digest(start)
    require(initial_hash == plan["source_protocol"]["start_sha256"], "initial adapter mismatch")
    cases, active = {}, {}
    for seed in seeds:
        folder = args.base / "models" / recipe / f"seed-{seed}"
        update = read_rows(folder / "updates.jsonl")[0]
        trace = read_rows(folder / "trace.jsonl")
        rows = [trace[i] for i in update["group_indices"]]
        require(len(rows) == 16 and update["old_sha256"] == initial_hash and update["batch_index"] == 0,
                "first update was not a full frozen batch at the common initialization")
        for row in rows:
            require(row["sha256"] == initial_hash and all(s["behavior_sha256"] == initial_hash for s in row["samples"]),
                    "sample policy differs from baseline")
            if row["nonzero_advantage_tokens"]:
                active[seed, row["group_index"]] = row
        cases[seed] = rows, update

    layers = len(plan["source_protocol"]["qualification_protocol"]["lora_layers"])
    model, _ = load_mlx_model(None, layers, plan["test_seed"])
    set_adapter(model, start)
    baseline = {key: margin(model, row) for key, row in active.items()}
    report = {"recipe": recipe, "common_start_sha256": initial_hash,
              "measurement": "mean token log probability of sampled correct completion minus that of paired incorrect completion, nats per token",
              "comparison": "first accumulation batch only, with other-seed comparison excluding every prompt in the training batch; "
                            "other-seed pairs can recur across updated models", "per_seed": {}}
    for seed, (rows, update) in cases.items():
        set_adapter(model, start)
        learner = MlxGrpoLearner(model, 1e-4)
        learner.step_groups([(r["prompt_ids"], r["samples"], r["advantages"]) for r in rows])
        actual_hash = adapter_digest(adapter_snapshot(model))
        require(actual_hash == update["new_sha256"], f"seed {seed}: first-step weights differ")
        trained_ids = {r["prompt_id"] for r in rows}
        pairs = []
        for key, row in active.items():
            source_seed, group = key
            if source_seed != seed and row["prompt_id"] in trained_ids:
                continue
            after = margin(model, row)
            pairs.append({"source_seed": source_seed, "group_index": group, "prompt_id": row["prompt_id"],
                          "own_batch": source_seed == seed, "before": baseline[key], "after": after,
                          "delta": after - baseline[key]})
        summaries = {label: summarize([p["delta"] for p in pairs if p["own_batch"] == own])
                     for label, own in (("own", True), ("other", False))}
        report["per_seed"][str(seed)] = {"adapter_sha256": actual_hash, "pairs": pairs, **summaries}
        print(seed, summaries, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_summary(args.out, report)


if __name__ == "__main__":
    main()
