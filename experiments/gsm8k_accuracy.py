"""Screen worked-solution training after GRPO, then confirm free GSM8K accuracy."""

import argparse
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from random import Random
from statistics import median

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW
from safetensors.torch import load_file, save_file
import torch

from experiments.audit_kv_online import read_json, require
from experiments.gsm8k_loss import encoded_rows, load_source, score_loss
from experiments.gsm8k_matched import evaluate_test, save_summary
from experiments.gsm8k_mlx_development import (
    adapter_snapshot, load_mlx_model, set_adapter, supervised_loss, train_step,
)
from experiments.gsm8k_mlx_generate import generate_batch
from experiments.gsm8k_regroup import adapter_digest
from experiments.gsm8k_signal import MODEL, REVISION, TEST_SHA256, TRAIN_SHA256, read_split


DEV_SEED = 64403
TEST_SEED = 65509
EXTRA_OFFSET = 68903
SEEDS = list(range(5))
SCREEN_SEEDS = [0, 1]
CANDIDATES = [48, 128, 256]
SFT_LR = 0.0005
QUALIFICATION = {"positive_seeds": 4, "median_gain": 0.05, "bootstrap_lower": 0}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def evaluate(model, tokenizer, examples, ids, protocol):
    return evaluate_test(model, tokenizer, examples, {**protocol, "heldout_prompt_ids": ids}, generate_batch)


def extend_sft(model, tokenizer, train, schedule, lr, milestones, evaluate_checkpoint):
    loss_and_grad = nn.value_and_grad(model, supervised_loss)
    optimizer = AdamW(learning_rate=lr, weight_decay=0.01, bias_correction=True)
    updates = []
    for step, prompt_id in enumerate(schedule, 1):
        metrics = train_step(model, tokenizer, train[prompt_id], loss_and_grad, optimizer)
        updates.append({"prompt_id": prompt_id, **metrics})
        if step in milestones:
            mx.clear_cache()
            evaluate_checkpoint(step, updates)
    return updates


def freeze(args, train, test):
    meta, start, records, source = load_source(args.source)
    baseline_meta = read_json(args.baseline / "protocol.json")
    require(baseline_meta["source"] == source and baseline_meta["seeds"] == SEEDS,
            "baseline checkpoints come from different GRPO training")
    require(read_json(args.baseline / "audit.json")["qualified"], "baseline GRPO weight replay audit is missing")
    traces = {seed: [row["prompt_id"] for row in rows] for seed, rows in records.items()}
    paths = {str(seed): args.baseline / f"seed-{seed}" / "grpo" for seed in SEEDS}
    summaries = {seed: read_json(path / "summary.json") for seed, path in paths.items()}
    for seed, summary in summaries.items():
        require(summary["start_sha256"] == adapter_digest(start) and
                [row["prompt_id"] for row in summary["training"]] == traces[seed] and
                sum(row["token_budget"] for row in summary["training"]) == meta["token_budget"],
                f"seed {seed}: baseline GRPO budget or start checkpoint differs")
    excluded_test = sorted(set(meta["excluded_test_ids"] + meta["heldout_prompt_ids"] + baseline_meta["heldout_prompt_ids"]))
    test_ids = Random(args.test_seed).sample(sorted(set(range(len(test))) - set(excluded_test)), args.heldout)
    excluded_train = sorted(set(meta["excluded_train_ids"] + read_json(args.baseline / "development.json")["prompt_ids"] +
                                [i for ids in traces.values() for i in ids]))
    train_pool = sorted(set(range(len(train))) - set(excluded_train))
    dev_ids = Random(DEV_SEED).sample(train_pool, args.dev)
    extra_pool = sorted(set(train_pool) - set(dev_ids))
    extra = {str(seed): Random(EXTRA_OFFSET + seed).sample(extra_pool, max(args.candidates)) for seed in SEEDS}
    dev_questions = {train[i]["question"] for i in dev_ids}
    train_questions = {train[i]["question"] for i in set(excluded_train) | {i for ids in extra.values() for i in ids}}
    require(not dev_questions & train_questions and
            not {test[i]["question"] for i in test_ids} & (train_questions | dev_questions), "evaluation text overlaps training")
    protocol = {"model": MODEL, "revision": REVISION, "train_sha256": TRAIN_SHA256, "test_sha256": TEST_SHA256,
                "source": source, "baseline_run": args.baseline.name,
                "baseline_summary_sha256": {seed: digest(path / "summary.json") for seed, path in paths.items()},
                "seeds": SEEDS, "train_prompt_ids": traces, "extra_prompt_ids": extra,
                "excluded_train_ids": excluded_train, "excluded_test_ids": excluded_test, "test_seed": args.test_seed,
                "heldout_prompt_ids": test_ids, "dev_prompt_ids": dev_ids, "dev_seed": DEV_SEED,
                "extra_seed_offset": EXTRA_OFFSET, "screen_seeds": SCREEN_SEEDS, "candidates": args.candidates,
                "sft_lr": SFT_LR, "baseline_lr": baseline_meta["lr"], "rollout_tokens": meta["token_budget"],
                "eval_max_new_tokens": meta["eval_max_new_tokens"], "eval_batch": 12,
                "qualification_protocol": meta["qualification_protocol"], "qualification": QUALIFICATION,
                "mlx": version("mlx"), "mlx_lm": version("mlx-lm")}
    return protocol, paths, summaries


def load_baseline(model, path, summary):
    snapshot = load_file(path / "final-adapter.safetensors")
    require(adapter_digest(snapshot) == summary["final_sha256"], "baseline weight hash mismatch")
    set_adapter(model, snapshot)


def screen(model, tokenizer, train, protocol, paths, baselines, out):
    result = {"seeds": {}}
    for seed in map(str, protocol["screen_seeds"]):
        load_baseline(model, paths[seed], baselines[seed])
        baseline = evaluate(model, tokenizer, train, protocol["dev_prompt_ids"], protocol)
        entry = {"grpo": {"evaluation": baseline}, "candidates": {}}
        result["seeds"][seed] = entry
        print(f"dev seed {seed} GRPO: {baseline['correct']}/{len(protocol['dev_prompt_ids'])}", flush=True)

        def checkpoint(step, updates):
            eval_ = evaluate(model, tokenizer, train, protocol["dev_prompt_ids"], protocol)
            snapshot = adapter_snapshot(model)
            entry["candidates"][str(step)] = {"evaluation": eval_, "final_sha256": adapter_digest(snapshot),
                                              "supervised_tokens": sum(row["supervised_tokens"] for row in updates)}
            save_file(snapshot, out / f"development-{seed}-{step}.safetensors")
            save_summary(out / "development.json", result)
            print(f"dev seed {seed} SFT {step}: {eval_['correct']}/{len(protocol['dev_prompt_ids'])}", flush=True)

        schedule = protocol["extra_prompt_ids"][seed]
        entry["training"] = extend_sft(model, tokenizer, train, schedule, protocol["sft_lr"], protocol["candidates"], checkpoint)
        save_summary(out / "development.json", result)
    deltas = {str(step): [entry["candidates"][str(step)]["evaluation"]["correct"] - entry["grpo"]["evaluation"]["correct"]
                          for entry in result["seeds"].values()] for step in protocol["candidates"]}
    result["deltas"] = deltas
    passing = [step for step in protocol["candidates"] if min(deltas[str(step)]) > 0 and
               median(deltas[str(step)]) / len(protocol["dev_prompt_ids"]) >= QUALIFICATION["median_gain"]]
    result["selected_steps"] = max(passing, key=lambda step: (median(deltas[str(step)]), -step)) if passing else None
    save_summary(out / "development.json", result)
    print(f"selected steps: {result['selected_steps']}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("train", "test", "source", "baseline", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--dev", type=int, default=48)
    parser.add_argument("--heldout", type=int, default=96)
    parser.add_argument("--test-seed", type=int, default=TEST_SEED)
    parser.add_argument("--candidates", type=int, nargs="+", default=CANDIDATES)
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--score-nll", action="store_true", help="score the already archived formal checkpoints")
    args = parser.parse_args()
    args.candidates = sorted(set(args.candidates))
    require(min(args.dev, args.heldout, *args.candidates) > 0, "invalid evaluation or training budgets")
    torch.set_num_threads(1)
    train, test = read_split(args.train, TRAIN_SHA256), read_split(args.test, TEST_SHA256)
    protocol, paths, baselines = freeze(args, train, test)
    args.out.mkdir(parents=True, exist_ok=True)
    plan_path = args.out / "plan.json"
    if plan_path.exists():
        require(read_json(plan_path) == protocol, "settings differ from the frozen split and training plan")
    else:
        save_summary(plan_path, protocol)
    model, tokenizer = load_mlx_model(None, len(protocol["qualification_protocol"]["lora_layers"]), DEV_SEED)
    if args.score_nll:
        formal = read_json(args.out / "protocol.json")
        require(formal["heldout_prompt_ids"] == protocol["heldout_prompt_ids"], "NLL split differs from confirmation")
        rows = encoded_rows(tokenizer, test, formal["heldout_prompt_ids"], formal["qualification_protocol"]["prompt_mode"])
        results = {}
        for seed in map(str, SEEDS):
            load_baseline(model, paths[seed], baselines[seed])
            baseline = score_loss(model, rows)
            folder = args.out / f"seed-{seed}" / "aux"
            load_baseline(model, folder, read_json(folder / "summary.json"))
            candidate = score_loss(model, rows)
            results[seed] = {"grpo": baseline, "aux": candidate}
            print(f"seed {seed} NLL: {baseline['token_nll']:.6f} -> {candidate['token_nll']:.6f}", flush=True)
        drops = [1 - pair["aux"]["token_nll"] / pair["grpo"]["token_nll"] for pair in results.values()]
        save_summary(args.out / "nll.json", {"prompt_ids": formal["heldout_prompt_ids"], "seeds": results,
                                             "median_relative_drop": median(drops), "lower_seeds": sum(d > 0 for d in drops)})
        return
    dev_path = args.out / "development.json"
    dev = read_json(dev_path) if dev_path.exists() else {}
    if "selected_steps" not in dev:
        dev = screen(model, tokenizer, train, protocol, paths, baselines, args.out)
    selected = dev["selected_steps"]
    if args.screen_only or selected is None:
        return
    protocol.update(selected_steps=selected, screen_sha256=digest(dev_path))
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists():
        require(read_json(protocol_path) == protocol, "confirmation protocol changed")
    else:
        save_summary(protocol_path, protocol)
    for seed in map(str, SEEDS):
        baseline = baselines[seed]
        load_baseline(model, paths[seed], baseline)
        info = {"start_sha256": baseline["start_sha256"], "final_sha256": baseline["final_sha256"],
                "rollout_tokens": protocol["rollout_tokens"]}
        folder = args.out / f"seed-{seed}"
        for arm in ("grpo", "aux"):
            (folder / arm).mkdir(parents=True, exist_ok=True)
        if not (folder / "grpo/summary.json").exists():
            eval_ = evaluate(model, tokenizer, test, protocol["heldout_prompt_ids"], protocol)
            save_summary(folder / "grpo/summary.json", {"evaluation": eval_, **info})
            print(f"test seed {seed} GRPO: {eval_['correct']}/{args.heldout}", flush=True)
        if (folder / "aux/summary.json").exists():
            continue
        if seed in dev["seeds"]:
            entry = dev["seeds"][seed]
            update_rows = entry["training"][:selected]
            snapshot = load_file(args.out / f"development-{seed}-{selected}.safetensors")
            require(adapter_digest(snapshot) == entry["candidates"][str(selected)]["final_sha256"], "development hash mismatch")
            set_adapter(model, snapshot)
        else:
            schedule = protocol["extra_prompt_ids"][seed][:selected]
            update_rows = extend_sft(model, tokenizer, train, schedule, protocol["sft_lr"], (), None)
            snapshot = adapter_snapshot(model)
        mx.clear_cache()
        eval_ = evaluate(model, tokenizer, test, protocol["heldout_prompt_ids"], protocol)
        save_file(snapshot, folder / "aux/final-adapter.safetensors")
        save_summary(folder / "aux/summary.json", {**info, "evaluation": eval_, "final_sha256": adapter_digest(snapshot),
                     "sft_start_sha256": info["final_sha256"], "training": update_rows,
                     "supervised_tokens": sum(row["supervised_tokens"] for row in update_rows)})
        print(f"test seed {seed} auxiliary: {eval_['correct']}/{args.heldout}", flush=True)


if __name__ == "__main__":
    main()
