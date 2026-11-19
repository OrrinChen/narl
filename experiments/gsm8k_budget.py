"""Compare SFT, online GRPO, and GRPO then SFT at equal response-token budgets."""

import argparse
from hashlib import sha256
from importlib.metadata import version
from math import isfinite
from pathlib import Path
from random import Random
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW, clip_grad_norm
from safetensors.torch import load_file, save_file
import torch

from experiments.audit_gsm8k_accuracy import sft_target
from experiments.audit_kv_online import read_json, require
from experiments.gsm8k_matched import evaluate_test, save_summary
from experiments.gsm8k_mlx_development import adapter_snapshot, load_mlx_model, set_adapter, supervised_loss
from experiments.gsm8k_mlx_generate import generate_batch
from experiments.gsm8k_mlx_grpo import MlxGrpoLearner
from experiments.gsm8k_online_lag import advantages, rollouts
from experiments.gsm8k_regroup import adapter_digest
from experiments.gsm8k_signal import MODEL, REVISION, TEST_SHA256, TRAIN_SHA256, prompt_text, read_split
from experiments.regroup import write_record


ARMS = ("sft", "grpo", "grpo_sft")
SEEDS = list(range(5))
DEV_SEED, TEST_SEED = 70183, 70189
GRPO_OFFSET, SFT_OFFSET = 70207, 70229


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def freeze(args, train, test):
    previous = read_json(args.previous / "protocol.json")
    original = read_json(args.qualification / "protocol.json")
    qualified = read_json(args.qualification / "dev-qualified.json")
    start = load_file(args.qualification / "fresh-adapter.safetensors")
    require(qualified["qualified"] and adapter_digest(start) == qualified["fresh_adapter_sha256"],
            "common SFT-only initialization failed qualification or changed weights")
    require((original["model"], original["revision"], original["train_sha256"], qualified["selected_step"]) ==
            (MODEL, REVISION, TRAIN_SHA256, 80), "expected the pinned 80-step SFT initialization")
    # Keep every previously inspected test question out of this confirmation.
    excluded_test = sorted(set(previous["excluded_test_ids"] + previous["heldout_prompt_ids"]))
    excluded_train = sorted(set(previous["excluded_train_ids"] + previous["dev_prompt_ids"] +
                                [i for ids in previous["extra_prompt_ids"].values() for i in ids]))
    pool = sorted(set(range(len(train))) - set(excluded_train))
    dev = Random(DEV_SEED).sample(pool, args.dev)
    pool = sorted(set(pool) - set(dev))
    heldout = Random(TEST_SEED).sample(sorted(set(range(len(test))) - set(excluded_test)), args.heldout)
    require(not {train[i]["question"] for i in dev} &
            {train[i]["question"] for i in excluded_train + pool}, "development text overlaps training")
    require(not {test[i]["question"] for i in heldout} & {row["question"] for row in train},
            "held-out question text overlaps the official training split")
    protocol = {
        "model": MODEL, "revision": REVISION, "train_sha256": TRAIN_SHA256, "test_sha256": TEST_SHA256,
        "qualification_run": args.qualification.name, "qualification_protocol": original,
        "qualification": qualified, "start_sha256": adapter_digest(start),
        "previous_run": args.previous.name, "previous_protocol_sha256": digest(args.previous / "protocol.json"),
        "seeds": SEEDS, "arms": list(ARMS), "screen_seeds": [0, 1],
        "excluded_train_ids": excluded_train, "excluded_test_ids": excluded_test,
        "dev_seed": DEV_SEED, "dev_prompt_ids": dev, "test_seed": TEST_SEED, "heldout_prompt_ids": heldout,
        "grpo_prompt_seed_offset": GRPO_OFFSET, "sft_prompt_seed_offset": SFT_OFFSET,
        "grpo_prompt_ids": {str(s): Random(GRPO_OFFSET + s).sample(pool, len(pool)) for s in SEEDS},
        "sft_prompt_ids": {str(s): Random(SFT_OFFSET + s).sample(pool, len(pool)) for s in SEEDS},
        "total_tokens": args.total_tokens, "combo_rollout_tokens": args.combo_rollout_tokens,
        "budget_unit": "emitted GRPO completion tokens including EOS plus SFT CE target tokens including EOS",
        "common_initialization": "80 worked-answer SFT examples; no project GRPO updates",
        "budget_exclusions": "common initialization and development/test inference; tracked separately",
        "grpo_lr": 0.0006, "sft_lr": 0.0005, "group_size": 2, "max_new_tokens": 256,
        "eval_max_new_tokens": 256, "eval_batch": 12, "stop_token_ids": [151643, 151645],
        "grpo_optimizer": "AdamW wd=0.01; continuous across the 6144-token boundary",
        "sft_optimizer": "new AdamW wd=0.01 in each SFT arm, including after GRPO",
        "selection": "fixed prior-development recipes; new development is diagnostic, no winner-dependent gate",
        "primary_comparison": "grpo_sft versus sft; two other pairwise contrasts are secondary",
        "mlx": version("mlx"), "mlx_lm": version("mlx-lm"),
    }
    return protocol, start


def evaluate(model, tokenizer, examples, ids, protocol):
    return evaluate_test(model, tokenizer, examples, {**protocol, "heldout_prompt_ids": ids}, generate_batch)


def grpo_accounting(rows):
    return {"generated_tokens": sum(row["generated_tokens"] for row in rows), "supervised_tokens": 0,
            "generation_prompt_tokens": sum(len(row["prompt_ids"]) * len(row["samples"]) for row in rows),
            "learner_input_tokens": sum(row["learner_input_tokens"] for row in rows),
            "nonzero_advantage_tokens": sum(row["nonzero_advantage_tokens"] for row in rows),
            "consumed_tokens": sum(row["consumed_tokens"] for row in rows),
            "updates": sum(row["next_sha256"] != row["current_sha256"] for row in rows),
            "groups": len(rows)}


def train_grpo(model, tokenizer, train, protocol, seed, initial, folder):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "training.json").exists():
        return read_json(folder / "training.json")
    set_adapter(model, initial)
    learner = MlxGrpoLearner(model, protocol["grpo_lr"])
    current = (initial, protocol["start_sha256"])
    version_, generated, rows = 0, 0, []
    boundary, budget = protocol["combo_rollout_tokens"], protocol["total_tokens"]
    started = perf_counter()
    with (folder / "trace.jsonl").open("w") as handle:
        for index, prompt_id in enumerate(protocol["grpo_prompt_ids"][str(seed)]):
            phase = "prefix" if generated < boundary else "continuation"
            limit = boundary if phase == "prefix" else budget
            prompt_ids, samples = rollouts(model, tokenizer, train[prompt_id], protocol, seed, "fresh", index,
                                           limit - generated, current, current, version_)
            tokens = sum(len(sample["token_ids"]) for sample in samples)
            accepted = tokens if len(samples) == protocol["group_size"] else 0
            values, signal = advantages(prompt_ids, samples) if accepted else ([0.0] * len(samples), 0)
            diagnostics = None
            next_hash = current[1]
            replay_input = 0
            if accepted:
                set_adapter(model, current[0])
                diagnostics = learner.step(prompt_ids, samples, values)
                replay_input = sum(len(prompt_ids) + len(sample["token_ids"]) - 1
                                   for sample, value in zip(samples, values, strict=True) if value != 0)
                if signal:
                    snapshot = adapter_snapshot(model)
                    next_hash = adapter_digest(snapshot)
            next_version = version_ + int(next_hash != current[1])
            row = {"phase": phase, "group_index": index, "prompt_id": prompt_id, "prompt_ids": prompt_ids,
                   "samples": samples, "current_version": version_, "current_sha256": current[1],
                   "next_version": next_version, "next_sha256": next_hash, "generated_tokens": tokens,
                   "consumed_tokens": accepted, "diagnostics": diagnostics, "nonzero_advantage_tokens": signal,
                   "learner_input_tokens": replay_input}
            write_record(handle, row)
            handle.flush()
            rows.append(row)
            if next_hash != current[1]:
                current = (snapshot, next_hash)
            version_ = next_version
            generated += tokens
            print(f"seed {seed} GRPO {index}: {generated}/{budget}, v{version_}, "
                  f"rewards {[s['reward'] for s in samples]}", flush=True)
            if generated == boundary:
                save_file(current[0], folder / "prefix-adapter.safetensors")
                save_summary(folder / "prefix.json", {"accounting": grpo_accounting(rows),
                             "training_elapsed_s": perf_counter() - started, "start_sha256": protocol["start_sha256"],
                             "final_sha256": current[1], "groups": len(rows)})
            if generated == budget:
                break
    require(generated == budget, "GRPO prompt stream exhausted before the budget")
    elapsed = perf_counter() - started
    save_file(current[0], folder / "final-adapter.safetensors")
    report = {"start_sha256": protocol["start_sha256"], "final_sha256": current[1],
              "accounting": grpo_accounting(rows), "training_elapsed_s": elapsed,
              "trace_sha256": digest(folder / "trace.jsonl")}
    save_summary(folder / "training.json", report)
    return report


def train_sft(model, tokenizer, train, protocol, seed, arm, initial, folder, prefix=None):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "training.json").exists():
        return read_json(folder / "training.json")
    set_adapter(model, initial)
    optimizer = AdamW(learning_rate=protocol["sft_lr"], weight_decay=0.01, bias_correction=True)
    loss_and_grad = nn.value_and_grad(model, supervised_loss)
    budget = protocol["total_tokens"] - (protocol["combo_rollout_tokens"] if prefix else 0)
    used, rows = 0, []
    started = perf_counter()
    with (folder / "sft-trace.jsonl").open("w") as handle:
        for index, prompt_id in enumerate(protocol["sft_prompt_ids"][str(seed)]):
            example = train[prompt_id]
            prompt = tokenizer.encode(prompt_text(tokenizer, example, "reasoning"), add_special_tokens=False)
            target = sft_target(tokenizer, example)
            labels = target[:budget - used]
            loss, grads = loss_and_grad(model, mx.array([prompt + labels]), len(labels))
            grads, norm = clip_grad_norm(grads, 1.0)
            mx.eval(loss, norm)
            require(isfinite(loss.item()) and isfinite(norm.item()), "nonfinite supervised update")
            optimizer.update(model, grads)
            mx.eval(model.trainable_parameters(), optimizer.state)
            row = {"prompt_id": prompt_id, "prompt_tokens": len(prompt), "full_target_tokens": len(target),
                   "supervised_tokens": len(labels), "loss": loss.item(), "gradient_norm": norm.item()}
            rows.append(row)
            write_record(handle, row)
            handle.flush()
            used += len(labels)
            if index % 16 == 0 or used == budget:
                print(f"seed {seed} {arm} {index}: {used}/{budget} labels", flush=True)
            if used == budget:
                break
    require(used == budget, "SFT prompt stream exhausted before the budget")
    elapsed = perf_counter() - started
    snapshot = adapter_snapshot(model)
    save_file(snapshot, folder / "final-adapter.safetensors")
    accounting = dict(prefix["accounting"]) if prefix else dict.fromkeys(
        ("generated_tokens", "supervised_tokens", "generation_prompt_tokens", "learner_input_tokens",
         "nonzero_advantage_tokens", "consumed_tokens", "updates", "groups"), 0)
    accounting["supervised_tokens"] += used
    accounting["learner_input_tokens"] += sum(row["prompt_tokens"] + row["supervised_tokens"] - 1 for row in rows)
    accounting["updates"] += len(rows)
    report = {"start_sha256": protocol["start_sha256"], "sft_start_sha256": adapter_digest(initial),
              "final_sha256": adapter_digest(snapshot), "accounting": accounting, "training": rows,
              "training_elapsed_s": elapsed + (prefix["training_elapsed_s"] if prefix else 0),
              "sft_elapsed_s": elapsed, "sft_trace_sha256": digest(folder / "sft-trace.jsonl")}
    save_summary(folder / "training.json", report)
    return report


def train_seed(model, tokenizer, train, protocol, seed, start, out):
    folder = out / f"seed-{seed}"
    train_grpo(model, tokenizer, train, protocol, seed, start, folder / "grpo")
    prefix = read_json(folder / "grpo/prefix.json")
    snapshot = load_file(folder / "grpo/prefix-adapter.safetensors")
    require(adapter_digest(snapshot) == prefix["final_sha256"], "GRPO prefix checkpoint changed")
    train_sft(model, tokenizer, train, protocol, seed, "grpo_sft", snapshot, folder / "grpo_sft", prefix)
    train_sft(model, tokenizer, train, protocol, seed, "sft", start, folder / "sft")


def evaluate_arm(model, tokenizer, examples, ids, protocol, folder):
    report = read_json(folder / "training.json")
    snapshot = load_file(folder / "final-adapter.safetensors")
    require(adapter_digest(snapshot) == report["final_sha256"], "training checkpoint hash changed")
    set_adapter(model, snapshot)
    mx.clear_cache()
    return evaluate(model, tokenizer, examples, ids, protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("train", "test", "qualification", "previous", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--total-tokens", type=int, default=32768)
    parser.add_argument("--combo-rollout-tokens", type=int, default=6144)
    parser.add_argument("--dev", type=int, default=48)
    parser.add_argument("--heldout", type=int, default=96)
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    require(0 < args.combo_rollout_tokens < args.total_tokens and min(args.dev, args.heldout) > 0, "invalid budget")
    torch.set_num_threads(1)
    train, test = read_split(args.train, TRAIN_SHA256), read_split(args.test, TEST_SHA256)
    protocol, start = freeze(args, train, test)
    args.out.mkdir(parents=True, exist_ok=True)
    plan_path = args.out / "plan.json"
    if plan_path.exists():
        require(read_json(plan_path) == protocol, "settings changed from the frozen plan")
    else:
        save_summary(plan_path, protocol)
        save_file(start, args.out / "start-adapter.safetensors")
    model, tokenizer = load_mlx_model(None, len(protocol["qualification_protocol"]["lora_layers"]), DEV_SEED)
    dev_path = args.out / "development.json"
    if not dev_path.exists():
        set_adapter(model, start)
        initial_eval = evaluate(model, tokenizer, train, protocol["dev_prompt_ids"], protocol)
        save_summary(dev_path, {"initial_evaluation": initial_eval, "seeds": {}})
        print(f"development start: {initial_eval['correct']}/{args.dev}", flush=True)
    dev = read_json(dev_path)
    for seed in protocol["screen_seeds"]:
        key = str(seed)
        train_seed(model, tokenizer, train, protocol, seed, start, args.out)
        dev["seeds"].setdefault(key, {})
        for arm in ARMS:
            if arm not in dev["seeds"][key]:
                evaluation = evaluate_arm(model, tokenizer, train, protocol["dev_prompt_ids"], protocol,
                                          args.out / f"seed-{seed}" / arm)
                dev["seeds"][key][arm] = evaluation
                save_summary(dev_path, dev)
                print(f"development {seed} {arm}: {evaluation['correct']}/{args.dev}", flush=True)
    if args.development_only:
        return
    protocol["development_sha256"] = digest(dev_path)
    formal_path = args.out / "protocol.json"
    if formal_path.exists():
        require(read_json(formal_path) == protocol, "formal protocol changed")
    else:
        save_summary(formal_path, protocol)
    initial_path = args.out / "initial-evaluation.json"
    if not initial_path.exists():
        set_adapter(model, start)
        save_summary(initial_path, evaluate(model, tokenizer, test, protocol["heldout_prompt_ids"], protocol))
    for seed in SEEDS:
        train_seed(model, tokenizer, train, protocol, seed, start, args.out)
        for arm in ARMS:
            folder = args.out / f"seed-{seed}" / arm
            if not (folder / "summary.json").exists():
                report = read_json(folder / "training.json")
                evaluation = evaluate_arm(model, tokenizer, test, protocol["heldout_prompt_ids"], protocol, folder)
                save_summary(folder / "summary.json", {**report, "evaluation": evaluation})
                print(f"test {seed} {arm}: {evaluation['correct']}/{args.heldout}", flush=True)


if __name__ == "__main__":
    main()
