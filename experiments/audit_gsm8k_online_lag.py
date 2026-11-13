"""Independently validate the online GSM8K one-version lag experiment."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import isclose, isfinite, sqrt
from pathlib import Path
from random import Random
import re

from experiments.audit_gsm8k_matched import check_development, score_evaluation
from experiments.audit_gsm8k_regroup import check_prompt_ids, check_score, read_split
from experiments.audit_kv_online import check_reported, read_json, read_rows, require
from experiments.gsm8k_lag_stats import ARMS, SEED_COUNT, summarize_results


SHA256 = re.compile(r"[0-9a-f]{64}")
EOS = {151643, 151645}
ANCHOR_PROMPTS = (4492, 4661, 1435, 557, 5687)
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
TRAIN_SHA256 = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"
TEST_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"


def adapter_digest(path):
    """Match the runner's ordered tensor-name and tensor-byte hash."""
    from safetensors import safe_open

    digest = sha256()
    with safe_open(path, framework="numpy") as tensors:
        for key in sorted(tensors.keys()):
            digest.update(key.encode())
            digest.update(tensors.get_tensor(key).tobytes())
    return digest.hexdigest()


def validate_sample(row, gold, cap, label, tokenizer=None):
    check_score(row, gold, label)
    tokens = row["token_ids"]
    logprobs = row["behavior_logprobs"]
    require(1 <= len(tokens) <= cap and all(type(token) is int and 0 <= token < 151936 for token in tokens),
            f"{label}: invalid completion tokens")
    require(not EOS.intersection(tokens[:-1]) and row["truncated"] == (tokens[-1] not in EOS)
            and (not row["truncated"] or len(tokens) == cap), f"{label}: invalid EOS or generation cap")
    require(len(tokens) == len(logprobs) and all(isfinite(logp) and logp <= 1e-6 for logp in logprobs),
            f"{label}: invalid behavior logprobs")
    if tokenizer is not None:
        require(tokenizer.decode(tokens, skip_special_tokens=True) == row["completion"],
                f"{label}: completion text disagrees with decoded tokens")
    return len(tokens)


def normalized_advantages(samples):
    """Rebuild GRPO's per-group sample-standard-deviation normalization."""
    count = len(samples)
    rewards = [row["reward"] for row in samples]
    mean = sum(rewards) / count
    std = sqrt(sum((reward - mean) ** 2 for reward in rewards) / (count - 1))
    return [(reward - mean) / (std + 1e-4) for reward in rewards]


def check_diagnostics(diagnostics, advantages, label):
    require(diagnostics is not None and all(isfinite(value) for value in diagnostics.values())
            and diagnostics["gnorm"] >= 0, f"{label}: invalid gradient diagnostics")
    skips = sum(value == 0 for value in advantages)
    require(diagnostics["skipped"] == skips and (skips < len(advantages) or diagnostics["gnorm"] == 0),
            f"{label}: optimizer skips disagree with GRPO advantages")


def check_warmup(run, protocol, train, source_path, tokenizer):
    warmup = protocol["warmup"]
    preparation = warmup["preparation"]
    prompt = preparation["prompt_id"]
    require(prompt == ANCHOR_PROMPTS[0] and set(ANCHOR_PROMPTS) <= set(protocol["excluded_train_ids"]),
            "unexpected preparation prompt or reuse in the measured arms")
    require(sha256(source_path.read_bytes()).hexdigest() == preparation["source_sample_sha256"],
            "warmup source digest does not match")
    records = [row for row in read_rows(source_path) if row["prompt_id"] == prompt]
    require(len(records) == 1, "missing or duplicate warmup source prompt")
    cohort = records[0]["cohorts"]["fresh"]
    ids = preparation["sample_ids"]
    require(ids == [f"{prompt}:fresh:{slot}" for slot in range(4)] == [sample["id"] for sample in cohort],
            "warmup did not use unique completions from the qualified snapshot")
    source_protocol = read_json(source_path.with_name("protocol.json"))
    require(source_protocol["qualification"] == protocol["qualification"]
            and source_protocol["qualification_protocol"] == protocol["qualification_protocol"],
            "warmup source used a different qualified checkpoint")
    require(read_json(source_path.with_name("summary.json"))["sample_sha256"] == preparation["source_sample_sha256"],
            "warmup source differs from the fixed-rollout report")
    for sample in cohort:
        validate_sample(sample, train[prompt][1], source_protocol["max_new_tokens"], sample["id"], tokenizer)
    check_prompt(records[0]["prompt_ids"], train[prompt][0], tokenizer, "warmup preparation")
    advantages = normalized_advantages(cohort)
    check_diagnostics(preparation["diagnostics"], advantages, "warmup preparation")
    require(any(advantages) and preparation["diagnostics"]["gnorm"] > 0
            and (preparation["from_version"], preparation["to_version"]) == (0, 1),
            "first GRPO preparation step did not produce an effective update")
    old = adapter_digest(run / "old-adapter.safetensors")
    current = adapter_digest(run / "current-adapter.safetensors")
    qualified = protocol["qualification"]["fresh_adapter_sha256"]
    require(len({old, current, qualified}) == 3
            and (qualified, old) == (preparation["old_adapter_sha256"], preparation["current_adapter_sha256"])
            and (old, current) == (warmup["old_adapter_sha256"], warmup["current_adapter_sha256"]),
            "warmup checkpoint bytes disagree with adjacent versions")

    path = run / "warmup-samples.json"
    require(sha256(path.read_bytes()).hexdigest() == warmup["source_sample_sha256"],
            "second GRPO step sample digest does not match")
    attempts = read_json(path)
    require(attempts and [item["prompt_id"] for item in attempts] == list(ANCHOR_PROMPTS[:len(attempts)]),
            "second GRPO step sample schedule is invalid")
    for index, attempt in enumerate(attempts):
        prompt = attempt["prompt_id"]
        check_prompt(attempt["prompt_ids"], train[prompt][0], tokenizer, "warmup anchor")
        samples = attempt["samples"]
        require(len(samples) == 4, "second GRPO step needs four completed samples")
        for slot, sample in enumerate(samples):
            sample_id = f"anchor:{prompt}:{slot}"
            require(sample["id"] == sample_id and sample["behavior_version"] == 1 and sample["behavior_sha256"] == old,
                    f"{sample_id}: second GRPO step used the wrong behavior checkpoint")
            validate_sample(sample, train[prompt][1], protocol["max_new_tokens"], sample_id, tokenizer)
        advantages = normalized_advantages(samples)
        require(any(advantages) == (index == len(attempts) - 1), "anchor did not stop at the first signal-bearing group")
    require(warmup["prompt_id"] == prompt and warmup["sample_ids"] == [sample["id"] for sample in samples],
            "second GRPO update does not identify its source group")
    check_diagnostics(warmup["diagnostics"], advantages, "warmup anchor")
    require(warmup["diagnostics"]["gnorm"] > 0 and (warmup["from_version"], warmup["to_version"]) == (1, 2),
            "second GRPO preparation step did not produce an effective update")
    return old, current, source_protocol


def check_splits(protocol, train, test, source_protocol):
    train_ids = protocol["excluded_train_ids"]
    excluded_test = protocol["excluded_test_ids"]
    heldout = protocol["heldout_prompt_ids"]
    check_prompt_ids(train_ids, train, "excluded training prompts")
    check_prompt_ids(excluded_test, test, "previously inspected test prompts")
    check_prompt_ids(heldout, test, "held-out prompts")
    dev = protocol["qualification_protocol"]
    require(train_ids == sorted(set(dev["training_prompt_ids"] + dev["development_prompt_ids"]
                                    + source_protocol["replay_prompt_ids"]))
            and excluded_test == sorted(set(source_protocol["excluded_test_ids"] + source_protocol["heldout_prompt_ids"])),
            "exclusions do not match the development and previous experiment splits")
    test_pool = sorted(set(range(len(test))) - set(excluded_test))
    require(heldout == Random(protocol["holdout_seed"]).sample(test_pool, len(heldout)),
            "held-out prompt order does not match the frozen unseen split")
    train_excluded = set(train_ids)
    require(not {test[i][0] for i in heldout} & {row[0] for row in train},
            "held-out question occurs in the train split")
    allowed = [i for i in range(len(train)) if i not in train_excluded]
    return allowed, heldout


def check_prompt(prompt_ids, question, tokenizer, label):
    require(prompt_ids and all(type(token) is int and 0 <= token < 151936 for token in prompt_ids),
            f"{label}: missing or invalid prompt tokens")
    if tokenizer is not None:
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        require(question in prompt, f"{label}: prompt tokens disagree with the training question")


def check_group(row, seed, arm, index, expected_prompt, train, state, remaining, cap, tokenizer):
    version, previous_hash, current_hash = state
    label = f"seed {seed}/{arm}/group {index}"
    require(row["group_index"] == index and row["prompt_id"] == expected_prompt,
            f"{label}: unexpected prompt schedule")
    require((row["current_version"], row["previous_sha256"], row["current_sha256"]) == state,
            f"{label}: current or previous weights break the publication chain")
    question, gold = train[expected_prompt]
    check_prompt(row["prompt_ids"], question, tokenizer, label)
    expected_count = min(2, remaining)
    samples = row["samples"]
    require(len(samples) == expected_count, f"{label}: missing or extra rollout slots")
    stale = arm == "stale"
    lag_pattern = (stale, stale) if arm != "mixed" else (index % 2 == 0, index % 2 != 0)
    lengths = []
    lag_tokens = Counter()
    for slot, sample in enumerate(samples):
        sample_id = f"{seed}:{arm}:{index}:{slot}"
        require(sample["id"] == sample_id, f"{label}: wrong sample ID")
        lag = int(lag_pattern[slot])
        expected_hash = previous_hash if lag else current_hash
        require(sample["behavior_version"] == version - lag and sample["behavior_sha256"] == expected_hash,
                f"{sample_id}: behavior version/hash or arm lag is wrong")
        expected_cap = min(cap, remaining - sum(lengths) - (expected_count - slot - 1))
        require(type(sample["token_cap"]) is int and sample["token_cap"] == expected_cap,
                f"{sample_id}: wrong token cap at the budget boundary")
        lengths.append(validate_sample(sample, gold, expected_cap, sample_id, tokenizer))
        lag_tokens[lag] += lengths[-1]
    generated = sum(lengths)
    consumed = generated if expected_count == 2 else 0
    next_hash = row["next_sha256"]
    require(isinstance(next_hash, str) and SHA256.fullmatch(next_hash) is not None,
            f"{label}: invalid post-update weight hash")
    changed = next_hash != current_hash
    require(row["next_version"] == version + int(changed),
            f"{label}: policy version must advance exactly when weights change")
    diagnostics = row["diagnostics"]
    advantages = normalized_advantages(samples) if consumed else [0]
    signal = sum(length for length, advantage in zip(lengths, advantages, strict=True) if advantage)
    check_reported(row, {"generated_tokens": generated, "consumed_tokens": consumed,
                         "nonzero_advantage_tokens": signal}, label)
    if consumed:
        check_diagnostics(diagnostics, advantages, label)
        require(not changed or any(advantages), f"{label}: an all-skipped update changed weights")
    else:
        require(diagnostics is None and not changed, f"{label}: incomplete tail changed weights")
    totals = Counter(groups=1, completed_groups=int(bool(consumed)), generated_tokens=generated,
                     consumed_tokens=consumed, nonzero_advantage_tokens=signal,
                     effective_updates=int(changed), nonzero_gradient_updates=int(bool(diagnostics and diagnostics["gnorm"])),
                     fresh_tokens=lag_tokens[0], stale_tokens=lag_tokens[1],
                     fresh_completions=sum(not lag for lag in lag_pattern[:expected_count]),
                     stale_completions=sum(lag_pattern[:expected_count]),
                     correct_completions=sum(sample["reward"] for sample in samples),
                     tagged_completions=sum(sample["has_answer_tag"] for sample in samples),
                     truncated_completions=sum(sample["truncated"] for sample in samples),
                     reduced_cap_completions=sum(sample["token_cap"] < cap for sample in samples),
                     unconsumed_tail_tokens=generated - consumed)
    next_state = (row["next_version"], current_hash, next_hash) if changed else state
    return next_state, totals


def audit_arm(run, seed, arm, protocol, train, test, schedule, state, tokenizer):
    folder = run / f"seed-{seed}" / arm
    trace = read_rows(folder / "trace.jsonl")
    require(0 < len(trace) <= len(schedule), f"seed {seed}/{arm}: incomplete or excessive prompt schedule")
    totals = Counter()
    budget = protocol["token_budget"]
    for index, row in enumerate(trace):
        remaining = budget - totals["generated_tokens"]
        require(remaining > 0, f"seed {seed}/{arm}: generated beyond the budget")
        state, counts = check_group(row, seed, arm, index, schedule[index], train, state, remaining,
                                    protocol["max_new_tokens"], tokenizer)
        totals.update(counts)
    version, _, weight_hash = state
    label = f"seed {seed}/{arm}"
    require(totals["generated_tokens"] == budget, f"{label}: did not exhaust the exact generation budget")
    require(adapter_digest(folder / "final-adapter.safetensors") == weight_hash,
            f"{label}: final adapter bytes disagree with the update chain")
    report = read_json(folder / "summary.json")
    check_reported(report, {"generated_tokens": budget, "consumed_tokens": totals["consumed_tokens"],
                            "final_version": version, "final_sha256": weight_hash}, label)
    evaluation = report["evaluation"]
    scores = check_evaluation(evaluation, protocol["heldout_prompt_ids"], test, label,
                              protocol["eval_max_new_tokens"], tokenizer)
    return {**totals, "correct": sum(scores), "accuracy": sum(scores) / len(scores),
            "final_version": version, "final_sha256": weight_hash}, report


def check_evaluation(evaluation, heldout, test, label, cap, tokenizer):
    scores = score_evaluation(evaluation, heldout, test, label, cap)
    for row in evaluation["rows"]:
        ids = row["token_ids"]
        require(all(type(token) is int and 0 <= token < 151936 for token in ids)
                and not EOS.intersection(ids[:-1]) and row["truncated"] == (ids[-1] not in EOS),
                f"{label}: invalid held-out completion token IDs or EOS")
        if tokenizer is not None:
            require(tokenizer.decode(ids, skip_special_tokens=True) == row["completion"],
                    f"{label}: held-out text disagrees with decoded tokens")
    return scores


def verify_training(run, protocol, qualification_path, source_path):
    """Replay optimizer updates from saved rollouts without sampling or evaluation."""
    from importlib.metadata import version

    import numpy as np
    from safetensors.torch import load_file

    from experiments.gsm8k_mlx_development import adapter_snapshot, load_mlx_model, set_adapter
    from experiments.gsm8k_mlx_grpo import MlxGrpoLearner
    from experiments.gsm8k_regroup import adapter_digest as tensor_digest

    require((version("mlx"), version("mlx-lm")) == (protocol["mlx"], protocol["mlx_lm"]),
            "training replay must use the frozen MLX versions")
    model, _ = load_mlx_model(None, len(protocol["qualification_protocol"]["lora_layers"]), protocol["holdout_seed"])

    def step(learner, prompt, samples, expected, expected_hash, label):
        # Narl's advantage math is float32. Float64 reconstruction is used for
        # structural audit above, while bitwise replay must round each operation.
        rewards = np.asarray([sample["reward"] for sample in samples], dtype=np.float32)
        values = ((rewards - rewards.mean()) / (rewards.std(ddof=1) + np.float32(1e-4))).tolist()
        actual = learner.step(prompt, samples, values)
        require(actual.keys() == expected.keys() and actual["skipped"] == expected["skipped"]
                and all(isclose(value, expected[key], rel_tol=1e-5, abs_tol=1e-6) for key, value in actual.items()),
                f"{label}: replayed optimizer diagnostics differ")
        require(tensor_digest(adapter_snapshot(model)) == expected_hash,
                f"{label}: replayed optimizer step differs from saved parameter bytes")

    set_adapter(model, load_file(qualification_path / "fresh-adapter.safetensors"))
    learner = MlxGrpoLearner(model, protocol["warmup_lr"])
    warmup = protocol["warmup"]
    prepare = warmup["preparation"]
    record, = [row for row in read_rows(source_path) if row["prompt_id"] == prepare["prompt_id"]]
    step(learner, record["prompt_ids"], record["cohorts"]["fresh"], prepare["diagnostics"],
         prepare["current_adapter_sha256"], "GRPO preparation")
    anchor = read_json(run / "warmup-samples.json")[-1]
    step(learner, anchor["prompt_ids"], anchor["samples"], warmup["diagnostics"],
         warmup["current_adapter_sha256"], "GRPO anchor")

    initial = load_file(run / "current-adapter.safetensors")
    checks = Counter()
    for seed in protocol["seeds"]:
        for arm in ARMS:
            set_adapter(model, initial)
            learner = MlxGrpoLearner(model, protocol["lr"])
            trace = read_rows(run / f"seed-{seed}" / arm / "trace.jsonl")
            for row in trace:
                if row["diagnostics"] is not None:
                    label = f"seed {seed}/{arm}/group {row['group_index']}"
                    step(learner, row["prompt_ids"], row["samples"], row["diagnostics"], row["next_sha256"], label)
                    checks["optimizer_batches"] += 1
                    checks["changed_weights"] += row["next_version"] != row["current_version"]
            checks["arms"] += 1
    return {"warmup_updates": 2, **checks}


def audit(run, train_path, test_path, source_path, qualification_path, tokenizer_path=None):
    tokenizer = None
    if tokenizer_path is not None:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    protocol = read_json(run / "protocol.json")
    require((protocol["model"], protocol["revision"], protocol["train_sha256"], protocol["test_sha256"]) ==
            (MODEL, REVISION, TRAIN_SHA256, TEST_SHA256), "model or GSM8K split is not the frozen revision")
    train = read_split(train_path, TRAIN_SHA256)
    test = read_split(test_path, TEST_SHA256)
    budget = protocol["token_budget"]
    seeds = protocol["seeds"]
    require(protocol["group_size"] == 2 and seeds == list(range(SEED_COUNT))
            and set(protocol["stop_token_ids"]) == EOS
            and min(budget, protocol["max_new_tokens"], protocol["eval_max_new_tokens"]) > 0,
            "invalid group size, token budget or paired seeds")
    dev, qualification = protocol["qualification_protocol"], protocol["qualification"]
    require(read_json(qualification_path / "protocol.json") == dev
            and read_json(qualification_path / "dev-qualified.json") == qualification,
            "recorded qualification differs from the development artifacts")
    development = check_development(qualification_path, dev, qualification, train)
    require(adapter_digest(qualification_path / "fresh-adapter.safetensors") == qualification["fresh_adapter_sha256"],
            "qualified checkpoint bytes disagree with the recorded hash")
    old_hash, current_hash, source_protocol = check_warmup(run, protocol, train, source_path, tokenizer)
    allowed, heldout = check_splits(protocol, train, test, source_protocol)
    initial = read_json(run / "initial-evaluation.json")
    check_evaluation(initial, heldout, test, "initial", protocol["eval_max_new_tokens"], tokenizer)
    seed_stats = {}
    seed_arms = {}
    for seed in seeds:
        schedule = Random(seed + protocol["prompt_seed_offset"]).sample(allowed, min(budget, len(allowed)))
        seed_stats[str(seed)] = {}
        seed_arms[str(seed)] = {}
        for arm in ARMS:
            state = (protocol["warmup"]["to_version"], old_hash, current_hash)
            stats, report = audit_arm(run, seed, arm, protocol, train, test, schedule, state, tokenizer)
            seed_stats[str(seed)][arm] = stats
            seed_arms[str(seed)][arm] = report
    paired = summarize_results(initial, seed_arms)
    summary_path = run / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        elapsed = summary.pop("elapsed_s")
        require(isfinite(elapsed) and elapsed >= 0 and summary == paired,
                "saved paired summary differs from independent audit")
    total_keys = ("groups", "completed_groups", "generated_tokens", "consumed_tokens", "nonzero_advantage_tokens",
                  "effective_updates", "nonzero_gradient_updates", "fresh_tokens", "stale_tokens", "fresh_completions",
                  "stale_completions", "correct_completions", "tagged_completions", "truncated_completions",
                  "reduced_cap_completions", "unconsumed_tail_tokens")
    totals = {arm: {key: sum(stats[arm][key] for stats in seed_stats.values()) for key in total_keys} for arm in ARMS}
    return {"token_budget_per_arm": budget, "development_gate": development, "arm_totals": totals,
            "initial_versions": {"old": old_hash, "current": current_hash},
            "decoded_token_text_checked": tokenizer is not None, "seeds": seed_stats, **paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "train", "test", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--warmup-source", type=Path, default=Path("runs/gsm8k-matched-mlx-final/samples.jsonl"))
    parser.add_argument("--qualification", type=Path, default=Path("runs/gsm8k-dev-mlx-final"))
    parser.add_argument("--tokenizer", type=Path, help="optional pinned tokenizer.json for text/token verification")
    parser.add_argument("--verify-training", action="store_true", help="replay every MLX optimizer step and weight hash")
    args = parser.parse_args()
    result = audit(args.run, args.train, args.test, args.warmup_source, args.qualification, args.tokenizer)
    if args.verify_training:
        result["training_replay"] = verify_training(args.run, read_json(args.run / "protocol.json"),
                                                   args.qualification, args.warmup_source)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Audited {result['seed_count']} paired seeds, {result['token_budget_per_arm']} tokens per arm, "
          f"and {result['heldout_count']} held-out questions")


if __name__ == "__main__":
    main()
