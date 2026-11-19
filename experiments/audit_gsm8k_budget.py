"""Independently audit SFT/GRPO ablations with equal response-token budgets."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path
from random import Random
import re

from experiments.audit_gsm8k_accuracy import paired_accuracy, replay_eval_batches
from experiments.audit_gsm8k_online_lag import (
    EOS, MODEL, REVISION, TEST_SHA256, TRAIN_SHA256, adapter_digest,
    check_diagnostics, check_evaluation, normalized_advantages, validate_sample,
)
from experiments.audit_gsm8k_regroup import check_prompt_ids
from experiments.audit_kv_online import check_reported, read_json, read_rows, require


ARMS = ("sft", "grpo", "grpo_sft")
SEEDS = tuple(range(5))
COMPARISONS = (("sft", "grpo"), ("sft", "grpo_sft"), ("grpo", "grpo_sft"))


def file_digest(path):
    return sha256(path.read_bytes()).hexdigest()


def target_tokens(tokenizer, example):
    """Reconstruct the official worked-answer target without runner helpers."""
    rationale = re.sub(r"<<[^>]+>>", "", example["rationale"])
    text = f"{rationale}\n<answer>{example['value']}</answer>"
    return tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]


def prompt_tokens(tokenizer, example):
    # Keep the audit prompt literal independent of the experimental runner.
    message = (
        "Solve the math problem. Show the calculation step by step before giving the answer. "
        "On the final line, write the number as <answer>NUMBER</answer>.\n\n"
        "Example: Maya buys three packs of four pens and gives away two. How many pens remain?\n"
        "Three packs contain 3 * 4 = 12 pens. After giving away two, 12 - 2 = 10 remain.\n"
        "<answer>10</answer>\n\nProblem: " + example["question"]
    )
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def paired_results(reports, heldout):
    """Preserve pairing across seeds and shared questions for all three contrasts."""
    contrasts = {}
    for baseline, candidate in COMPARISONS:
        pairs = {}
        for seed in SEEDS:
            base = reports[str(seed)][baseline]["evaluation"]["rows"]
            new = reports[str(seed)][candidate]["evaluation"]["rows"]
            require([row["prompt_id"] for row in base] == heldout == [row["prompt_id"] for row in new],
                    f"seed {seed}: paired evaluation order differs")
            pairs[str(seed)] = ([row["reward"] for row in base], [row["reward"] for row in new],
                                [row["token_ids"] for row in base], [row["token_ids"] for row in new])
        result = paired_accuracy(pairs, len(heldout))
        # Fair attribution, rather than any particular method winning, is the acceptance condition.
        result.pop("qualified")
        contrasts[f"{candidate}_vs_{baseline}"] = result
    return contrasts


def check_supervision(report, schedule, budget, train, tokenizer, label):
    updates = report["training"]
    require(updates and [row["prompt_id"] for row in updates] == schedule[:len(updates)],
            f"{label}: changed supervised question schedule")
    totals = Counter()
    for index, row in enumerate(updates):
        example = train[row["prompt_id"]]
        prompt = prompt_tokens(tokenizer, example)
        full = target_tokens(tokenizer, example)
        used = min(len(full), budget - totals["supervised_tokens"])
        require(used > 0, f"{label}: supervised update after budget exhaustion")
        check_reported(row, {"prompt_tokens": len(prompt), "full_target_tokens": len(full),
                             "supervised_tokens": used}, f"{label}/step {index}")
        require(all(isfinite(row[key]) and row[key] >= 0 for key in ("loss", "gradient_norm")),
                f"{label}/step {index}: invalid supervised diagnostics")
        totals.update(supervised_tokens=used, learner_input_tokens=len(prompt) + used - 1,
                      supervised_prompt_tokens=len(prompt), supervised_steps=1,
                      shortened_targets=int(used < len(full)))
    require(totals["supervised_tokens"] == budget, f"{label}: supervised budget is not exhausted exactly")
    return totals


def check_protocol(run, qualification_path, previous_path, train, test):
    protocol = read_json(run / "protocol.json")
    require(read_json(run / "plan.json") ==
            {key: value for key, value in protocol.items() if key != "development_sha256"},
            "formal protocol differs from the pre-development plan")
    require((protocol["model"], protocol["revision"], protocol["train_sha256"], protocol["test_sha256"]) ==
            (MODEL, REVISION, TRAIN_SHA256, TEST_SHA256), "model or dataset revision changed")
    require(protocol["arms"] == list(ARMS) and protocol["seeds"] == list(SEEDS)
            and protocol["screen_seeds"] == [0, 1] and protocol["group_size"] == 2,
            "expected three arms, five paired seeds, and size-two GRPO groups")
    require((protocol["total_tokens"], protocol["combo_rollout_tokens"], protocol["grpo_lr"], protocol["sft_lr"],
             protocol["max_new_tokens"], protocol["eval_max_new_tokens"], protocol["eval_batch"]) ==
            (32768, 6144, 0.0006, 0.0005, 256, 256, 12), "frozen budget, optimizer recipe, or decode settings changed")
    require(set(protocol["stop_token_ids"]) == EOS and protocol["dev_seed"] == 70183
            and protocol["test_seed"] == 70189 and protocol["grpo_prompt_seed_offset"] == 70207
            and protocol["sft_prompt_seed_offset"] == 70229, "sampling or split seeds changed")
    qualified = read_json(qualification_path / "dev-qualified.json")
    original = read_json(qualification_path / "protocol.json")
    start_hash = adapter_digest(qualification_path / "fresh-adapter.safetensors")
    require(protocol["qualification_run"] == qualification_path.name
            and protocol["qualification_protocol"] == original and protocol["qualification"] == qualified
            and qualified["qualified"] and qualified["selected_step"] == 80
            and start_hash == qualified["fresh_adapter_sha256"] == protocol["start_sha256"]
            == adapter_digest(run / "start-adapter.safetensors"),
            "common starting weights do not match the 80-example SFT qualification")
    require(original["prompt_mode"] == "reasoning", "common checkpoint used a different prompt")
    previous = read_json(previous_path / "protocol.json")
    require(protocol["previous_run"] == previous_path.name
            and protocol["previous_protocol_sha256"] == file_digest(previous_path / "protocol.json"),
            "previous experiment provenance changed")
    excluded_test = sorted(set(previous["excluded_test_ids"] + previous["heldout_prompt_ids"]))
    excluded_train = sorted(set(previous["excluded_train_ids"] + previous["dev_prompt_ids"] +
                                [i for ids in previous["extra_prompt_ids"].values() for i in ids]))
    require(protocol["excluded_test_ids"] == excluded_test and protocol["excluded_train_ids"] == excluded_train,
            "excluded questions differ from the previously inspected/trained sets")
    for ids, examples, label in ((excluded_test, test, "old test"), (excluded_train, train, "old train"),
                                 (protocol["dev_prompt_ids"], train, "new development"),
                                 (protocol["heldout_prompt_ids"], test, "new holdout")):
        check_prompt_ids(ids, examples, label)
    pool = sorted(set(range(len(train))) - set(excluded_train))
    dev = protocol["dev_prompt_ids"]
    heldout = protocol["heldout_prompt_ids"]
    require(len(dev) == 48 and dev == Random(70183).sample(pool, len(dev)), "new development draw changed")
    test_pool = sorted(set(range(len(test))) - set(excluded_test))
    require(len(heldout) == 96 and heldout == Random(70189).sample(test_pool, len(heldout)),
            "new holdout draw changed or reused previously scored questions")
    pool = sorted(set(pool) - set(dev))
    dev_questions = {train[i]["question"] for i in dev}
    used_questions = {train[i]["question"] for i in excluded_train + pool}
    require(not dev_questions & used_questions and
            not {test[i]["question"] for i in heldout} & {row["question"] for row in train},
            "development or holdout question text overlaps training")
    for field, offset in (("grpo_prompt_ids", 70207), ("sft_prompt_ids", 70229)):
        require(set(protocol[field]) == {str(seed) for seed in SEEDS}, "training schedule omits a paired seed")
        for seed in SEEDS:
            require(protocol[field][str(seed)] == Random(offset + seed).sample(pool, len(pool)),
                    f"seed {seed}: changed {field} schedule")
    require(file_digest(run / "development.json") == protocol["development_sha256"],
            "development records changed after confirmation was frozen")
    return protocol


def check_grpo_trace(run, seed, protocol, train, tokenizer):
    folder = run / f"seed-{seed}" / "grpo"
    rows = read_rows(folder / "trace.jsonl")
    require(rows, f"seed {seed}: missing GRPO trajectory")
    budget, boundary = protocol["total_tokens"], protocol["combo_rollout_tokens"]
    schedule = protocol["grpo_prompt_ids"][str(seed)]
    totals = Counter()
    prefix_totals = None
    current_hash, version = protocol["start_sha256"], 0
    for index, row in enumerate(rows):
        label = f"seed {seed}/GRPO/group {index}"
        generated = totals["generated_tokens"]
        phase = "prefix" if generated < boundary else "continuation"
        remaining = (boundary if phase == "prefix" else budget) - generated
        require(remaining > 0 and row["phase"] == phase and row["group_index"] == index
                and row["prompt_id"] == schedule[index], f"{label}: wrong phase, group, or prompt order")
        expected_prompt = prompt_tokens(tokenizer, train[row["prompt_id"]])
        require(row["prompt_ids"] == expected_prompt and row["current_sha256"] == current_hash
                and row["current_version"] == version, f"{label}: broken prompt or parameter publication chain")
        samples = row["samples"]
        require(len(samples) == min(2, remaining), f"{label}: incomplete or excessive generated slots")
        lengths = []
        for slot, sample in enumerate(samples):
            name = f"{seed}:fresh:{index}:{slot}"
            cap = min(256, remaining - sum(lengths) - (len(samples) - slot - 1))
            require(sample["id"] == name and sample["behavior_version"] == version
                    and sample["behavior_sha256"] == current_hash and sample["token_cap"] == cap,
                    f"{label}/slot {slot}: wrong generation version or budget cap")
            lengths.append(validate_sample(sample, float(train[row["prompt_id"]]["value"]), cap, name, tokenizer))
        emitted = sum(lengths)
        consumed = emitted if len(samples) == 2 else 0
        advantages = normalized_advantages(samples) if consumed else [0.0]
        signal = sum(length for length, advantage in zip(lengths, advantages, strict=True) if advantage)
        learner_inputs = sum(len(expected_prompt) + length - 1
                             for length, advantage in zip(lengths, advantages, strict=True) if advantage)
        check_reported(row, {"generated_tokens": emitted, "consumed_tokens": consumed,
                             "nonzero_advantage_tokens": signal, "learner_input_tokens": learner_inputs}, label)
        changed = row["next_sha256"] != current_hash
        require(re.fullmatch(r"[0-9a-f]{64}", row["next_sha256"]) is not None
                and row["next_version"] == version + int(changed), f"{label}: invalid parameter version increment")
        if consumed:
            check_diagnostics(row["diagnostics"], advantages, label)
            require(not changed or signal > 0, f"{label}: zero-advantage group changed parameters")
        else:
            require(row["diagnostics"] is None and not changed, f"{label}: incomplete group trained the model")
        totals.update(generated_tokens=emitted, supervised_tokens=0,
                      generation_prompt_tokens=len(expected_prompt) * len(samples),
                      learner_input_tokens=learner_inputs, nonzero_advantage_tokens=signal,
                      consumed_tokens=consumed, updates=int(changed), groups=1)
        current_hash, version = row["next_sha256"], row["next_version"]
        if totals["generated_tokens"] == boundary:
            require(prefix_totals is None, f"seed {seed}: duplicate prefix boundary")
            prefix_totals = totals.copy()
            prefix = read_json(folder / "prefix.json")
            check_reported(prefix["accounting"], dict(prefix_totals), f"seed {seed}/prefix")
            require(prefix["start_sha256"] == protocol["start_sha256"] and prefix["final_sha256"] == current_hash
                    == adapter_digest(folder / "prefix-adapter.safetensors") and prefix["groups"] == index + 1,
                    f"seed {seed}: prefix checkpoint disagrees with the shared trajectory")
    require(totals["generated_tokens"] == budget and prefix_totals is not None,
            f"seed {seed}: GRPO did not use exactly the total/prefix budgets")
    report = read_json(folder / "training.json")
    check_reported(report["accounting"], dict(totals), f"seed {seed}/GRPO")
    require(report["start_sha256"] == protocol["start_sha256"] and report["final_sha256"] == current_hash
            == adapter_digest(folder / "final-adapter.safetensors")
            and report["trace_sha256"] == file_digest(folder / "trace.jsonl"),
            f"seed {seed}: full GRPO checkpoint or trace digest changed")
    require(isfinite(report["training_elapsed_s"]) and report["training_elapsed_s"] >= 0
            and 0 <= prefix["training_elapsed_s"] <= report["training_elapsed_s"],
            f"seed {seed}: invalid GRPO/prefix timing")
    return rows, report, prefix


def audit(run, qualification_path, previous_path, train_path, test_path, tokenizer_path,
          replay_training=False, verify_batches=None):
    from transformers import AutoTokenizer

    from experiments.gsm8k_signal import read_split

    train = read_split(train_path, TRAIN_SHA256)
    test = read_split(test_path, TEST_SHA256)
    test_gold = [(row["question"], float(row["value"])) for row in test]
    train_gold = [(row["question"], float(row["value"])) for row in train]
    protocol = check_protocol(run, qualification_path, previous_path, train, test)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    development = read_json(run / "development.json")
    check_evaluation(development["initial_evaluation"], protocol["dev_prompt_ids"], train_gold,
                     "development/initial", 256, tokenizer)
    require(set(development["seeds"]) == {"0", "1"}, "development uses unexpected seeds")
    for seed in ("0", "1"):
        require(set(development["seeds"][seed]) == set(ARMS), "development omits an arm")
        for arm in ARMS:
            check_evaluation(development["seeds"][seed][arm], protocol["dev_prompt_ids"], train_gold,
                             f"development/{seed}/{arm}", 256, tokenizer)
    initial = read_json(run / "initial-evaluation.json")
    check_evaluation(initial, protocol["heldout_prompt_ids"], test_gold, "test/initial", 256, tokenizer)
    reports, traces, training = {}, {}, {}
    for seed in SEEDS:
        key = str(seed)
        trace, grpo, prefix = check_grpo_trace(run, seed, protocol, train, tokenizer)
        traces[key] = trace
        reports[key], training[key] = {}, {}
        for arm in ARMS:
            folder = run / f"seed-{seed}" / arm
            report = read_json(folder / "training.json")
            summary = read_json(folder / "summary.json")
            require({field: value for field, value in summary.items() if field != "evaluation"} == report,
                    f"seed {seed}/{arm}: confirmation changed the trained checkpoint")
            require(report["start_sha256"] == protocol["start_sha256"]
                    and report["final_sha256"] == adapter_digest(folder / "final-adapter.safetensors"),
                    f"seed {seed}/{arm}: checkpoint digest differs from training")
            if arm != "grpo":
                label = f"seed {seed}/{arm}"
                inherited = prefix["accounting"] if arm == "grpo_sft" else dict.fromkeys(grpo["accounting"], 0)
                labels = protocol["total_tokens"] - inherited["generated_tokens"]
                totals = check_supervision(report, protocol["sft_prompt_ids"][key], labels, train, tokenizer, label)
                require(read_rows(folder / "sft-trace.jsonl") == report["training"]
                        and report["sft_trace_sha256"] == file_digest(folder / "sft-trace.jsonl"),
                        f"{label}: SFT trace differs from archived training")
                expected = dict(inherited)
                expected["supervised_tokens"] += labels
                expected["learner_input_tokens"] += totals["learner_input_tokens"]
                expected["updates"] += totals["supervised_steps"]
                check_reported(report["accounting"], expected, label)
                expected_start = prefix["final_sha256"] if arm == "grpo_sft" else protocol["start_sha256"]
                require(report["sft_start_sha256"] == expected_start
                        and isfinite(report["sft_elapsed_s"]) and report["sft_elapsed_s"] >= 0
                        and isclose(report["training_elapsed_s"], report["sft_elapsed_s"] +
                                    (prefix["training_elapsed_s"] if arm == "grpo_sft" else 0), rel_tol=1e-10),
                        f"{label}: wrong SFT starting checkpoint or timing inheritance")
            accounting = report["accounting"]
            require(accounting["generated_tokens"] + accounting["supervised_tokens"] == protocol["total_tokens"],
                    f"seed {seed}/{arm}: unequal total response-token budget")
            scores = check_evaluation(summary["evaluation"], protocol["heldout_prompt_ids"], test_gold,
                                      f"test/{seed}/{arm}", 256, tokenizer)
            reports[key][arm] = summary
            training[key][arm] = {**accounting, "total_tokens": protocol["total_tokens"],
                                  "training_elapsed_s": report["training_elapsed_s"], "correct": sum(scores),
                                  "truncated": summary["evaluation"]["truncated"],
                                  "final_sha256": report["final_sha256"]}
    result = {"audit_passed": True, "heldout_count": len(protocol["heldout_prompt_ids"]),
              "development_count": len(protocol["dev_prompt_ids"]), "initial_correct": initial["correct"],
              "token_budget_per_arm": protocol["total_tokens"], "seeds": training,
              "contrasts": paired_results(reports, protocol["heldout_prompt_ids"]),
              "development_correct": {seed: {arm: evaluation["correct"] for arm, evaluation in arms.items()}
                                      for seed, arms in development["seeds"].items()},
              "decoded_token_text_checked": True}
    if replay_training or verify_batches is not None:
        result["model_replay"] = verify_models(run, protocol, train, test, tokenizer_path, reports, traces,
                                               replay_training, verify_batches)
    return result


def verify_sft(model, tokenizer, train, summary, lr, label):
    """Rebuild target labels, replay all AdamW steps, and compare endpoint bytes."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.optimizers import AdamW, clip_grad_norm

    from experiments.gsm8k_mlx_development import adapter_snapshot, suffix_logits
    from experiments.gsm8k_regroup import adapter_digest as tensor_digest

    def loss_fn(model, inputs, actions):
        logits = suffix_logits(model, inputs, actions.shape[1]).astype(mx.float32)
        return nn.losses.cross_entropy(logits, actions, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = AdamW(learning_rate=lr, weight_decay=0.01, bias_correction=True)
    for index, row in enumerate(summary["training"]):
        example = train[row["prompt_id"]]
        prompt = prompt_tokens(tokenizer, example)
        target = target_tokens(tokenizer, example)[:row["supervised_tokens"]]
        loss, gradients = loss_and_grad(model, mx.array([(prompt + target)[:-1]]), mx.array([target]))
        gradients, norm = clip_grad_norm(gradients, 1.0)
        require(isclose(loss.item(), row["loss"], rel_tol=1e-6, abs_tol=1e-7)
                and isclose(norm.item(), row["gradient_norm"], rel_tol=1e-6, abs_tol=1e-7),
                f"{label}/step {index}: SFT replay diagnostics differ")
        optimizer.update(model, gradients)
        mx.eval(model.trainable_parameters(), optimizer.state)
    require(tensor_digest(adapter_snapshot(model)) == summary["final_sha256"],
            f"{label}: SFT replay endpoint differs from saved weights")
    return len(summary["training"])


def verify_models(run, protocol, train, test, tokenizer_path, reports, traces,
                  replay_training, verify_batches):
    """Optionally replay optimizer work and complete original greedy batches."""
    from importlib.metadata import version

    import mlx.core as mx
    import numpy as np
    from safetensors.torch import load_file
    import torch

    from experiments.gsm8k_mlx_development import adapter_snapshot, load_mlx_model, set_adapter
    from experiments.gsm8k_mlx_grpo import MlxGrpoLearner
    from experiments.gsm8k_regroup import adapter_digest as tensor_digest

    require((version("mlx"), version("mlx-lm")) == (protocol["mlx"], protocol["mlx_lm"]),
            "model replay must use the frozen MLX versions")
    torch.set_num_threads(1)
    model, tokenizer = load_mlx_model(tokenizer_path, len(protocol["qualification_protocol"]["lora_layers"]),
                                      protocol["dev_seed"])
    initial = load_file(run / "start-adapter.safetensors")
    checks = Counter()
    for seed in SEEDS:
        key = str(seed)
        folder = run / f"seed-{seed}"
        if replay_training:
            set_adapter(model, initial)
            learner = MlxGrpoLearner(model, protocol["grpo_lr"])
            for row in traces[key]:
                if row["diagnostics"] is not None:
                    rewards = np.asarray([sample["reward"] for sample in row["samples"]], dtype=np.float32)
                    advantages = ((rewards - rewards.mean()) /
                                  (rewards.std(ddof=1) + np.float32(1e-4))).tolist()
                    actual = learner.step(row["prompt_ids"], row["samples"], advantages)
                    expected = row["diagnostics"]
                    require(actual.keys() == expected.keys()
                            and all(isclose(value, expected[name], rel_tol=1e-5, abs_tol=1e-6)
                                    for name, value in actual.items()),
                            f"seed {seed}/group {row['group_index']}: GRPO replay diagnostics differ")
                    checks["grpo_optimizer_batches"] += 1
                require(tensor_digest(adapter_snapshot(model)) == row["next_sha256"],
                        f"seed {seed}/group {row['group_index']}: GRPO replay parameter bytes differ")
                checks["grpo_groups"] += 1
            require(tensor_digest(adapter_snapshot(model)) == reports[key]["grpo"]["final_sha256"],
                    f"seed {seed}: full GRPO replay endpoint differs")
            for arm in ("sft", "grpo_sft"):
                start = initial if arm == "sft" else load_file(folder / "grpo/prefix-adapter.safetensors")
                set_adapter(model, start)
                checks["sft_optimizer_steps"] += verify_sft(model, tokenizer, train, reports[key][arm],
                                                            protocol["sft_lr"], f"seed {seed}/{arm}")
            checks["replayed_seeds"] += 1
        if verify_batches is not None:
            for arm in ARMS:
                set_adapter(model, load_file(folder / arm / "final-adapter.safetensors"))
                mx.clear_cache()
                count = replay_eval_batches(model, tokenizer, test, protocol["heldout_prompt_ids"],
                                            reports[key][arm]["evaluation"], protocol, verify_batches)
                checks["redecoded_questions"] += count
                checks["redecoded_arms"] += 1
        print(f"model replay seed {seed}: {dict(checks)}", flush=True)
    return dict(checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "train", "test", "tokenizer", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, default=Path("runs/gsm8k-dev-mlx-final"))
    parser.add_argument("--previous", type=Path, default=Path("runs/gsm8k-accuracy-v1"))
    parser.add_argument("--verify-training", action="store_true", help="replay every GRPO and SFT optimizer update")
    parser.add_argument("--verify-batches", type=int, help="redecode this many greedy batches per arm; 0 means all")
    args = parser.parse_args()
    if args.verify_batches is not None and args.verify_batches < 0:
        parser.error("--verify-batches must be nonnegative")
    tokenizer_path = args.tokenizer.parent if args.tokenizer.is_file() else args.tokenizer
    result = audit(args.run, args.qualification, args.previous, args.train, args.test, tokenizer_path,
                   args.verify_training, args.verify_batches)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Audited three arms and five paired seeds at {result['token_budget_per_arm']} response tokens "
          f"per arm on {result['heldout_count']} new held-out questions.")


if __name__ == "__main__":
    main()
