"""Run rolling lag-one GSM8K GRPO with fresh, stale, and mixed groups on MLX."""

import argparse
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from random import Random
from time import perf_counter

import mlx.core as mx
from safetensors.torch import load_file, save_file
import torch

from algos import grpo
from core import Batch
from experiments.audit_kv_online import read_json, read_rows, require
from experiments.gsm8k_lag_stats import ARMS, summarize_results
from experiments.gsm8k_matched import evaluate_test, qualify, save_summary
from experiments.gsm8k_mlx_development import adapter_snapshot, load_mlx_model, set_adapter
from experiments.gsm8k_mlx_generate import generate_batch
from experiments.gsm8k_mlx_grpo import MlxGrpoLearner
from experiments.gsm8k_regroup import adapter_digest, trajectory
from experiments.gsm8k_signal import MODEL, REVISION, TEST_SHA256, TRAIN_SHA256, read_split
from experiments.regroup import write_json, write_record


SEEDS = tuple(range(5))
PROMPT_SEED_OFFSET = 81271
HOLDOUT_SEED = 53167
WARMUP_PROMPT = 4492
ANCHOR_PROMPTS = (4492, 4661, 1435, 557, 5687)
LR = 0.0001


def advantages(prompt_ids, samples):
    batch = Batch.from_groups([[trajectory(prompt_ids, sample) for sample in samples]])
    advantage = grpo(batch, None)
    return advantage[:, len(prompt_ids)].tolist(), int(advantage.ne(0).sum())


def bootstrap(model, tokenizer, train, qualified, record, source_path, protocol, out):
    qualified_hash = adapter_digest(qualified)
    samples = record["cohorts"]["fresh"]
    values, signal = advantages(record["prompt_ids"], samples)
    require(signal > 0, "warmup group must have reward variation")
    set_adapter(model, qualified)
    learner = MlxGrpoLearner(model, LR)
    diagnostics = learner.step(record["prompt_ids"], samples, values)
    old = adapter_snapshot(model)
    old_hash = adapter_digest(old)
    require(diagnostics["gnorm"] > 0 and old_hash != qualified_hash, "first GRPO step did not change weights")
    preparation = {"source_sample_sha256": sha256(source_path.read_bytes()).hexdigest(), "prompt_id": record["prompt_id"],
                   "sample_ids": [sample["id"] for sample in samples], "from_version": 0, "to_version": 1,
                   "old_adapter_sha256": qualified_hash, "current_adapter_sha256": old_hash, "diagnostics": diagnostics}
    mode = protocol["qualification_protocol"]["prompt_mode"]
    attempts = []
    for index, prompt_id in enumerate(ANCHOR_PROMPTS):
        mx.random.seed(22417 + index)
        outputs = generate_batch(model, tokenizer, [train[prompt_id]] * 4, protocol["max_new_tokens"],
                                 sample=True, capture=True, prompt_mode=mode)
        prompt_ids = outputs[0][0]
        samples = [{**row, "id": f"anchor:{prompt_id}:{slot}", "behavior_version": 1, "behavior_sha256": old_hash}
                   for slot, (_, row) in enumerate(outputs)]
        attempts.append({"prompt_id": prompt_id, "prompt_ids": prompt_ids, "samples": samples})
        values, signal = advantages(prompt_ids, samples)
        print("anchor", prompt_id, [sample["reward"] for sample in samples], flush=True)
        if signal:
            break
    require(signal > 0, "anchor generation had no nonzero advantages")
    diagnostics = learner.step(prompt_ids, samples, values)
    current = adapter_snapshot(model)
    current_hash = adapter_digest(current)
    require(diagnostics["gnorm"] > 0 and current_hash != old_hash, "second GRPO step did not change weights")
    save_file(old, out / "old-adapter.safetensors")
    save_file(current, out / "current-adapter.safetensors")
    samples_path = out / "warmup-samples.json"
    write_json(samples_path, attempts)
    report = {"source_sample_sha256": sha256(samples_path.read_bytes()).hexdigest(), "prompt_id": prompt_id,
              "sample_ids": [sample["id"] for sample in samples], "from_version": 1, "to_version": 2,
              "old_adapter_sha256": old_hash, "current_adapter_sha256": current_hash, "diagnostics": diagnostics,
              "preparation": preparation}
    return report, old, current


def rollouts(model, tokenizer, example, protocol, seed, arm, index, remaining, previous, current, current_version):
    mode = protocol["qualification_protocol"]["prompt_mode"]
    normal_cap = protocol["max_new_tokens"]
    ages = {"fresh": (0, 0), "stale": (1, 1), "mixed": (1, 0) if index % 2 == 0 else (0, 1)}[arm]
    versions = (current, previous)
    seed_key = (seed + 9107) * 100_000 + index * 2

    def sample(age, cap, size, seed_offset):
        set_adapter(model, versions[age][0])
        mx.random.seed(seed_key + seed_offset)
        return generate_batch(model, tokenizer, [example] * size, cap, sample=True, capture=True, prompt_mode=mode)

    if ages[0] == ages[1] and remaining >= 2 * normal_cap:
        outputs = sample(ages[0], normal_cap, 2, 0)
        caps = [normal_cap, normal_cap]
    else:
        outputs, caps = [], []
        for slot, age in enumerate(ages):
            if not remaining:
                break
            cap = min(normal_cap, max(1, remaining - (1 - slot)))
            output, = sample(age, cap, 1, slot)
            outputs.append(output)
            caps.append(cap)
            remaining -= len(output[1]["token_ids"])
    prompt_ids = outputs[0][0]
    samples = []
    for slot, (cap, (ids, row)) in enumerate(zip(caps, outputs, strict=True)):
        require(ids == prompt_ids, "weight version changed prompt tokenization")
        age = ages[slot]
        samples.append({**row, "id": f"{seed}:{arm}:{index}:{slot}", "behavior_version": current_version - age,
                        "behavior_sha256": versions[age][1], "token_cap": cap})
    return prompt_ids, samples


def train_arm(model, tokenizer, train, test, protocol, seed, arm, old, initial, out):
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    if summary_path.exists():
        return read_json(summary_path)
    warmup = protocol["warmup"]
    previous = (old, warmup["old_adapter_sha256"])
    current = (initial, warmup["current_adapter_sha256"])
    current_version = warmup["to_version"]
    set_adapter(model, initial)
    learner = MlxGrpoLearner(model, protocol["lr"])
    budget = protocol["token_budget"]
    generated = consumed = 0
    allowed = sorted(set(range(len(train))) - set(protocol["excluded_train_ids"]))
    schedule = Random(seed + protocol["prompt_seed_offset"]).sample(allowed, min(budget, len(allowed)))
    with (out / "trace.jsonl").open("w") as handle:
        for index, prompt_id in enumerate(schedule):
            prompt_ids, samples = rollouts(model, tokenizer, train[prompt_id], protocol, seed, arm, index,
                                           budget - generated, previous, current, current_version)
            tokens = sum(len(sample["token_ids"]) for sample in samples)
            accepted = tokens if len(samples) == protocol["group_size"] else 0
            diagnostics, signal = None, 0
            next_version, next_hash = current_version, current[1]
            if accepted:
                values, signal = advantages(prompt_ids, samples)
                set_adapter(model, current[0])
                diagnostics = learner.step(prompt_ids, samples, values)
                if signal:
                    snapshot = adapter_snapshot(model)
                    next_hash = adapter_digest(snapshot)
                    next_version += int(next_hash != current[1])
            row = {"group_index": index, "prompt_id": prompt_id, "prompt_ids": prompt_ids, "samples": samples,
                   "current_version": current_version, "previous_sha256": previous[1], "current_sha256": current[1],
                   "next_version": next_version, "next_sha256": next_hash, "generated_tokens": tokens,
                   "consumed_tokens": accepted, "diagnostics": diagnostics, "nonzero_advantage_tokens": signal}
            write_record(handle, row)
            handle.flush()
            if next_version != current_version:
                previous, current = current, (snapshot, next_hash)
                current_version = next_version
            generated += tokens
            consumed += accepted
            print(seed, arm, index, f"{generated}/{budget}", current_version,
                  [sample["reward"] for sample in samples], flush=True)
            if generated == budget:
                break
    require(generated == budget, "prompt schedule ran out before the token budget")
    set_adapter(model, current[0])
    save_file(current[0], out / "final-adapter.safetensors")
    mx.clear_cache()
    evaluation = evaluate_test(model, tokenizer, test, protocol, generate_batch)
    report = {"evaluation": evaluation, "generated_tokens": generated, "consumed_tokens": consumed,
              "final_version": current_version, "final_sha256": current[1]}
    save_summary(summary_path, report)
    print(seed, arm, "heldout", evaluation["correct"], generated, consumed, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "test", "qualification", "warmup-run", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=6144)
    parser.add_argument("--heldout-prompts", type=int, default=32)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    require(min(args.token_budget, args.heldout_prompts, args.eval_batch, args.max_new_tokens) > 0, "invalid budget")
    torch.set_num_threads(1)
    train, test = read_split(args.train, TRAIN_SHA256), read_split(args.test, TEST_SHA256)
    dev, qualified, snapshots = qualify(args.qualification, train)
    warmup_protocol = read_json(args.warmup_run / "protocol.json")
    require(warmup_protocol["qualification"] == qualified and warmup_protocol["qualification_protocol"] == dev,
            "warmup samples must be generated from the qualified adapter")
    source = args.warmup_run / "samples.jsonl"
    require(read_json(args.warmup_run / "summary.json")["sample_sha256"] == sha256(source.read_bytes()).hexdigest(),
            "warmup samples differ from the audited fixed-rollout pool")
    warmup_record, = [row for row in read_rows(source) if row["prompt_id"] == WARMUP_PROMPT]
    excluded_train = sorted(set(dev["training_prompt_ids"] + dev["development_prompt_ids"] + warmup_protocol["replay_prompt_ids"]))
    excluded_test = sorted(set(warmup_protocol["excluded_test_ids"] + warmup_protocol["heldout_prompt_ids"]))
    heldout = Random(HOLDOUT_SEED).sample(sorted(set(range(len(test))) - set(excluded_test)), args.heldout_prompts)
    require(not {row["question"] for row in train} & {test[i]["question"] for i in heldout}, "train and held-out overlap")
    protocol = {"model": MODEL, "revision": REVISION, "train_sha256": TRAIN_SHA256, "test_sha256": TEST_SHA256,
                "qualification_protocol": dev, "qualification": qualified, "seeds": list(SEEDS),
                "excluded_train_ids": excluded_train, "excluded_test_ids": excluded_test, "heldout_prompt_ids": heldout,
                "holdout_seed": HOLDOUT_SEED, "prompt_seed_offset": PROMPT_SEED_OFFSET, "token_budget": args.token_budget,
                "group_size": 2, "max_new_tokens": args.max_new_tokens, "eval_max_new_tokens": dev["max_new_tokens"],
                "eval_batch": args.eval_batch, "warmup_lr": LR, "lr": LR, "mlx": version("mlx"), "mlx_lm": version("mlx-lm"),
                "stop_token_ids": [151643, 151645], "budget_unit": "emitted completion tokens including EOS"}
    args.out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_mlx_model(None, len(dev["lora_layers"]), HOLDOUT_SEED)
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists():
        saved = read_json(protocol_path)
        warmup = saved.pop("warmup")
        require(saved == protocol, "settings differ from the frozen experiment")
        old, initial = (load_file(args.out / f"{name}-adapter.safetensors") for name in ("old", "current"))
        require((adapter_digest(old), adapter_digest(initial)) ==
                (warmup["old_adapter_sha256"], warmup["current_adapter_sha256"]), "anchor adapter hash changed")
    else:
        warmup, old, initial = bootstrap(model, tokenizer, train, snapshots["fresh"], warmup_record, source, protocol, args.out)
    protocol["warmup"] = warmup
    if not protocol_path.exists():
        write_json(protocol_path, protocol)
    eval_path = args.out / "initial-evaluation.json"
    if eval_path.exists():
        initial_eval = read_json(eval_path)
    else:
        set_adapter(model, initial)
        initial_eval = evaluate_test(model, tokenizer, test, protocol, generate_batch)
        save_summary(eval_path, initial_eval)
        print("initial", initial_eval["correct"], flush=True)
    started = perf_counter()
    arms = {seed: {arm: train_arm(model, tokenizer, train, test, protocol, seed, arm, old, initial,
                                 args.out / f"seed-{seed}" / arm) for arm in ARMS} for seed in SEEDS}
    report = summarize_results(initial_eval, arms)
    report["elapsed_s"] = perf_counter() - started
    save_summary(args.out / "summary.json", report)


if __name__ == "__main__":
    main()
