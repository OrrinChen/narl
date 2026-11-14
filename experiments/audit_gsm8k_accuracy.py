"""Audit paired GSM8K greedy accuracy on a frozen held-out question set."""

import argparse
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path
from random import Random
import re
from statistics import median

from experiments.audit_gsm8k_loss import check_training, read_split, source_trace
from experiments.audit_gsm8k_matched import score_change
from experiments.audit_gsm8k_online_lag import adapter_digest, check_evaluation
from experiments.audit_gsm8k_regroup import check_prompt_ids
from experiments.audit_kv_online import check_reported, read_json, require
from experiments.gsm8k_lag_stats import paired_sign_flip_p


SEEDS = tuple(range(5))
MIN_POSITIVE_SEEDS = 4
MIN_MEDIAN_GAIN = 0.05
BOOTSTRAP_SEED = 80161
BOOTSTRAP_REPLICATES = 10000
DEV_SEED = 64403
EXTRA_SEED_OFFSET = 68903


def check_holdout(protocol, source, prior_loss, train, test):
    ids = protocol["heldout_prompt_ids"]
    excluded = protocol["excluded_test_ids"]
    previous = set(source["excluded_test_ids"] + source["heldout_prompt_ids"] + prior_loss["heldout_prompt_ids"])
    check_prompt_ids(ids, test, "accuracy holdout")
    check_prompt_ids(excluded, test, "excluded test prompts")
    require(excluded == sorted(previous), "test exclusions differ from the previously scored questions")
    allowed = sorted(set(range(len(test))) - set(excluded))
    require(Random(protocol["test_seed"]).sample(allowed, len(ids)) == ids, "holdout differs from its frozen random split")
    questions = {test[index]["question"] for index in ids}
    require(len(questions) == len(ids) and not questions & {row["question"] for row in train}
            and not questions & {test[index]["question"] for index in excluded},
            "holdout repeats a previously scored or training question")
    return ids


def check_decoded(evaluation, ids, examples, label, cap, tokenizer):
    gold = [(row["question"], float(row["value"])) for row in examples]
    scores = check_evaluation(evaluation, ids, gold, label, cap, tokenizer)
    return scores, [row["token_ids"] for row in evaluation["rows"]]


def replay_eval_batches(model, tokenizer, examples, ids, evaluation, protocol, count):
    """Replay complete original decode batches so padding and shape stay matched."""
    from experiments.gsm8k_mlx_generate import generate_batch

    batch_size = protocol["eval_batch"]
    selected = 0
    for offset in range(0, len(ids), batch_size):
        batch = ids[offset:offset + batch_size]
        rows = generate_batch(model, tokenizer, [examples[index] for index in batch], protocol["eval_max_new_tokens"],
                              prompt_mode=protocol["qualification_protocol"]["prompt_mode"])
        expected = evaluation["rows"][offset:offset + batch_size]
        require(all(actual["token_ids"] == saved["token_ids"] for (_, actual), saved in zip(rows, expected, strict=True)),
                f"fresh greedy decode differs from archive for batch at offset {offset}")
        selected += len(batch)
        if count and selected >= count * batch_size:
            break
    return selected


def paired_accuracy(pairs, heldout_count):
    """Each seed keeps its own per-question outcomes on the shared questions."""
    require(heldout_count > 0 and len(pairs) == len(SEEDS), "expected five paired seeds with nonempty scores")
    per_seed = {}
    for seed in SEEDS:
        base_scores, new_scores, base_outputs, new_outputs = pairs[str(seed)]
        require(all(len(rows) == heldout_count for rows in (base_scores, new_scores, base_outputs, new_outputs)),
                f"seed {seed}: evaluation matrix differs from the frozen held-out set")
        result = score_change(base_scores, new_scores, base_outputs, new_outputs)
        per_seed[str(seed)] = {"baseline_correct": sum(base_scores), "candidate_correct": sum(new_scores), **result}
    deltas = [row["wins"] - row["losses"] for row in per_seed.values()]
    positive = sum(delta > 0 for delta in deltas)
    gain = median(deltas) / heldout_count
    intervals = paired_question_bootstrap(pairs, heldout_count)
    return {"per_seed": per_seed, "positive_seeds": positive, "tied_seeds": deltas.count(0),
            "negative_seeds": sum(delta < 0 for delta in deltas), "median_accuracy_gain": gain,
            "min_accuracy_gain": min(deltas) / heldout_count, "max_accuracy_gain": max(deltas) / heldout_count,
            "exact_seed_sign_flip_p": paired_sign_flip_p(deltas), "bootstrap": intervals,
            "qualified": (positive >= MIN_POSITIVE_SEEDS and gain >= MIN_MEDIAN_GAIN
                          and intervals["question_block_median_95"][0] > 0)}


def paired_question_bootstrap(pairs, heldout_count):
    """Share resampled question indices across seeds; also stress-test seed variation."""
    import numpy as np

    pairs_by_seed = [pairs[str(seed)] for seed in SEEDS]
    matrix = np.asarray([np.asarray(candidate, dtype=np.int8) - np.asarray(baseline, dtype=np.int8)
                         for baseline, candidate, _, _ in pairs_by_seed])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ids = rng.integers(0, heldout_count, size=(BOOTSTRAP_REPLICATES, heldout_count))
    question_effects = matrix[:, ids].mean(axis=-1)
    seed_ids = rng.integers(0, len(SEEDS), size=(len(SEEDS), BOOTSTRAP_REPLICATES))
    crossed_effects = question_effects[seed_ids, np.arange(BOOTSTRAP_REPLICATES)]

    def interval(data):
        return np.quantile(data, [0.025, 0.975]).tolist()

    return {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES,
            "question_block_median_95": interval(np.median(question_effects, axis=0)),
            "question_block_mean_95": interval(question_effects.mean(axis=0)),
            "crossed_seed_question_median_95": interval(np.median(crossed_effects, axis=0)),
            "crossed_seed_question_mean_95": interval(crossed_effects.mean(axis=0))}


def sft_target(tokenizer, row):
    rationale = re.sub(r"<<[^>]+>>", "", row["rationale"])
    return tokenizer.encode(f"{rationale}\n<answer>{row['value']}</answer>", add_special_tokens=False) + [tokenizer.eos_token_id]


def sft_tokens(tokenizer, row):
    return len(sft_target(tokenizer, row))


def check_train_schedule(report, extra_ids, train, tokenizer, label):
    training = report["training"]
    require([step["prompt_id"] for step in training] == extra_ids,
            f"{label}: supervised training schedule changed")
    for step in training:
        expected = sft_tokens(tokenizer, train[step["prompt_id"]])
        require(step["supervised_tokens"] == expected, f"{label}: supervised token budget is incorrect")
        require(all(isfinite(step[key]) and step[key] >= 0 for key in ("loss", "gradient_norm")),
                f"{label}: invalid training diagnostics")
    total = sum(step["supervised_tokens"] for step in training)
    check_reported(report, {"supervised_tokens": total}, label)
    return total


def check_prior_grpo(protocol, source_path, baseline_path, source, prior, tokenizer, train):
    """Verify that each baseline consumes the captured rollout budget and matches its archived weights."""
    input_hash = adapter_digest(source_path / "current-adapter.safetensors")
    require(source["warmup"]["current_adapter_sha256"] == input_hash == protocol["source"]["source_adapter_sha256"],
            "original GRPO starting adapter differs from the rollout source")
    require(prior["source"] == protocol["source"] and prior["train_prompt_ids"] == protocol["train_prompt_ids"],
            "archived baseline does not use the frozen paired rollout pool")
    require(protocol["baseline_run"] == baseline_path.name and protocol["baseline_lr"] == prior["lr"],
            "baseline path or learning rate differs from the archived run")
    old_audit = read_json(baseline_path / "audit.json")
    baselines = {}
    for seed in SEEDS:
        key = str(seed)
        trace = source_trace(source_path, seed, protocol["source"]["trace_sha256"][key], train, source)
        require([row["prompt_id"] for row in trace] == protocol["train_prompt_ids"][key],
                f"seed {seed}: original GRPO prompt order changed")
        budget = sum(row["consumed_tokens"] for row in trace)
        require(budget == source["token_budget"] == protocol["rollout_tokens"] == 6144,
                f"seed {seed}: rollout budget is not 6,144 tokens")
        folder = baseline_path / f"seed-{seed}" / "grpo"
        summary_path = folder / "summary.json"
        require(sha256(summary_path.read_bytes()).hexdigest() == protocol["baseline_summary_sha256"][key],
                f"seed {seed}: archived GRPO summary digest changed")
        archived = read_json(summary_path)
        counts = check_training(archived, trace, tokenizer, train, "grpo", f"seed {seed}/archived GRPO")
        require(counts["steps"] == old_audit["seeds"][key]["grpo"]["replayed_steps"],
                f"seed {seed}: archived baseline lacked optimizer replay for every group")
        final = adapter_digest(folder / "final-adapter.safetensors")
        require(archived["start_sha256"] == input_hash and archived["final_sha256"] == final,
                f"seed {seed}: archived GRPO checkpoint hashes changed")
        baselines[key] = {"start_sha256": input_hash, "final_sha256": final, "rollout_tokens": budget}
    return baselines


def check_extra_ids(protocol, source, baseline_path, train):
    used = {index for ids in protocol["train_prompt_ids"].values() for index in ids}
    excluded = protocol["excluded_train_ids"]
    check_prompt_ids(excluded, train, "excluded training prompts")
    earlier_dev = read_json(baseline_path / "development.json")["prompt_ids"]
    require(excluded == sorted(set(source["excluded_train_ids"]) | used | set(earlier_dev)),
            "supervision/development pool differs from the frozen exclusions")
    allowed = sorted(set(range(len(train))) - set(excluded))
    dev = protocol["dev_prompt_ids"]
    check_prompt_ids(dev, train, "accuracy development prompts")
    require(protocol["dev_seed"] == DEV_SEED and Random(DEV_SEED).sample(allowed, len(dev)) == dev,
            "development split differs from its frozen draw")
    extra_pool = sorted(set(allowed) - set(dev))
    desired = max(protocol["candidates"])
    dev_questions = {train[index]["question"] for index in dev}
    used_questions = {train[index]["question"] for index in excluded}
    require(not dev_questions & used_questions, "development repeats an earlier training question")
    for seed in SEEDS:
        extra_ids = protocol["extra_prompt_ids"][str(seed)]
        check_prompt_ids(extra_ids, train, f"seed {seed}/supervised prompts")
        require(protocol["extra_seed_offset"] == EXTRA_SEED_OFFSET
                and extra_ids == Random(EXTRA_SEED_OFFSET + seed).sample(extra_pool, desired),
                f"seed {seed}: supervised prompt order differs from its frozen draw")
        questions = {train[index]["question"] for index in extra_ids}
        require(not questions & (used_questions | dev_questions), f"seed {seed}: supervised question overlaps development")


def check_dev_screen(run, protocol, train, tokenizer):
    path = run / "development.json"
    require(sha256(path.read_bytes()).hexdigest() == protocol["screen_sha256"], "development screen bytes changed")
    report = read_json(path)
    ids = protocol["dev_prompt_ids"]
    cap = protocol["eval_max_new_tokens"]
    seed_rows = report["seeds"]
    require(set(seed_rows) == {str(seed) for seed in protocol["screen_seeds"]}, "development screen used other seeds")
    candidates = protocol["candidates"]
    scores = {steps: [] for steps in candidates}
    for key, data in seed_rows.items():
        base, _ = check_decoded(data["grpo"]["evaluation"], ids, train, f"development/{key}/grpo", cap, tokenizer)
        require(set(data["candidates"]) == {str(steps) for steps in candidates}, "development screen omitted a candidate")
        check_train_schedule({"training": data["training"],
                              "supervised_tokens": sum(row["supervised_tokens"] for row in data["training"])},
                             protocol["extra_prompt_ids"][key], train, tokenizer, f"development/{key}")
        for steps in candidates:
            candidate = data["candidates"][str(steps)]
            label = f"development/{key}/{steps}"
            predicted, _ = check_decoded(candidate["evaluation"], ids, train, label, cap, tokenizer)
            extra = protocol["extra_prompt_ids"][key][:steps]
            require(candidate["supervised_tokens"] == sum(sft_tokens(tokenizer, train[index]) for index in extra),
                    f"{label}: supervised token budget differs from the frozen schedule")
            require(candidate["final_sha256"] == adapter_digest(run / f"development-{key}-{steps}.safetensors"),
                    f"{label}: candidate checkpoint bytes differ from the archived hash")
            scores[steps].append(sum(predicted) - sum(base))
    require({str(steps): values for steps, values in scores.items()} == report["deltas"],
            "recorded development deltas disagree with re-scoring")
    passing = [steps for steps, values in scores.items() if min(values) > 0 and median(values) / len(ids) >= MIN_MEDIAN_GAIN]
    selected = max(passing, key=lambda steps: (median(scores[steps]), -steps)) if passing else None
    require(selected == report["selected_steps"] == protocol["selected_steps"],
            "confirmation checkpoint disagrees with the frozen development selection rule")
    return report, scores


def verify_sft(model, tokenizer, train, candidate, protocol, label):
    """Rebuild supervised targets and replay the optimizer from the original GRPO adapter."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.optimizers import AdamW, clip_grad_norm

    from experiments.gsm8k_mlx_development import adapter_snapshot, suffix_logits
    from experiments.gsm8k_regroup import adapter_digest as weight_digest
    from experiments.gsm8k_signal import prompt_text

    def loss_function(model, ids, targets):
        logits = suffix_logits(model, ids, targets.shape[1]).astype(mx.float32)
        return nn.losses.cross_entropy(logits, targets, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_function)
    optimizer = AdamW(learning_rate=protocol["sft_lr"], weight_decay=0.01, bias_correction=True)
    for step in candidate["training"]:
        example = train[step["prompt_id"]]
        prompt = tokenizer.encode(prompt_text(tokenizer, example, "reasoning"), add_special_tokens=False)
        target = sft_target(tokenizer, example)
        loss, grads = loss_and_grad(model, mx.array([(prompt + target)[:-1]]), mx.array([target]))
        grads, norm = clip_grad_norm(grads, 1.0)
        require(isclose(loss.item(), step["loss"], rel_tol=1e-6, abs_tol=1e-7)
                and isclose(norm.item(), step["gradient_norm"], rel_tol=1e-6, abs_tol=1e-7),
                f"{label}/{step['prompt_id']}: supervised optimizer diagnostics differ from replay")
        optimizer.update(model, grads)
        mx.eval(model.trainable_parameters(), optimizer.state)
    require(weight_digest(adapter_snapshot(model)) == candidate["final_sha256"],
            f"{label}: final adapter differs from SFT replay")
    return len(candidate["training"])


def audit(run, source_path, baseline_path, train_path, test_path, tokenizer_path, verify_batches=None, replay_training=False):
    from transformers import AutoTokenizer

    from experiments.gsm8k_signal import read_split as read_training_split

    protocol = read_json(run / "protocol.json")
    plan = read_json(run / "plan.json")
    require(plan == {key: value for key, value in protocol.items() if key not in ("selected_steps", "screen_sha256")},
            "formal protocol differs from pre-screen plan")
    source = read_json(source_path / "protocol.json")
    prior = read_json(baseline_path / "protocol.json")
    for key in ("model", "revision", "train_sha256", "test_sha256", "eval_max_new_tokens", "mlx", "mlx_lm"):
        require(protocol[key] == source[key] == prior[key], f"{key} changed from the original rollout and learner")
    require(protocol["qualification_protocol"] == source["qualification_protocol"]
            and protocol["qualification_protocol"]["prompt_mode"] == prior["prompt_mode"] == "reasoning"
            and protocol["eval_max_new_tokens"] == 256 and protocol["eval_batch"] == 12,
            "free decoder differs from the frozen model and reasoning prompt")
    require(protocol["seeds"] == source["seeds"] == list(SEEDS) and protocol["screen_seeds"] == [0, 1],
            "five paired training seeds and two development seeds are required")
    require(protocol["qualification"] == {"positive_seeds": MIN_POSITIVE_SEEDS, "median_gain": MIN_MEDIAN_GAIN,
                                           "bootstrap_lower": 0} and protocol["sft_lr"] == 0.0005,
            "accuracy acceptance rule or SFT learning rate changed")
    candidates = protocol["candidates"]
    selected = protocol["selected_steps"]
    require(candidates == sorted(set(candidates)) and all(type(steps) is int and steps > 0 for steps in candidates)
            and selected in candidates, "formal SFT budget must be one of the frozen development checkpoints")
    train = read_training_split(train_path, protocol["train_sha256"])
    test = read_split(test_path, protocol["test_sha256"])
    heldout = check_holdout(protocol, source, prior, train, test)
    require(len(heldout) >= 96 and len(protocol["dev_prompt_ids"]) >= 48, "held-out or development set is too small")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    check_extra_ids(protocol, source, baseline_path, train)
    baseline_info = check_prior_grpo(protocol, source_path, baseline_path, source, prior, tokenizer, train)
    development, dev_scores = check_dev_screen(run, protocol, train, tokenizer)

    real_model = None
    if verify_batches is not None or replay_training:
        from importlib.metadata import version

        from safetensors.torch import load_file

        from experiments.gsm8k_mlx_development import load_mlx_model, set_adapter

        require((version("mlx"), version("mlx-lm")) == (protocol["mlx"], protocol["mlx_lm"]),
                "decode replay requires the frozen MLX dependency versions")
        real_model, tokenizer = load_mlx_model(tokenizer_path, len(protocol["qualification_protocol"]["lora_layers"]), DEV_SEED)

    pairs = {}
    for seed in SEEDS:
        key = str(seed)
        arm_path = run / f"seed-{seed}"
        grpo = read_json(arm_path / "grpo/summary.json")
        candidate = read_json(arm_path / "aux/summary.json")
        info = baseline_info[key]
        label = f"seed {seed}"
        check_reported(grpo, info, f"{label}/GRPO")
        check_reported(candidate, {"start_sha256": info["start_sha256"], "sft_start_sha256": info["final_sha256"],
                                   "rollout_tokens": info["rollout_tokens"]}, f"{label}/GRPO+SFT")
        final_path = arm_path / "aux/final-adapter.safetensors"
        require(candidate["final_sha256"] == adapter_digest(final_path) != info["final_sha256"],
                f"{label}: final SFT weights are missing or unchanged")
        if key in development["seeds"]:
            checkpoint = development["seeds"][key]["candidates"][str(selected)]
            require(candidate["final_sha256"] == checkpoint["final_sha256"]
                    and candidate["training"] == development["seeds"][key]["training"][:selected],
                    f"{label}: formal SFT was changed after development")
        extra_ids = protocol["extra_prompt_ids"][key][:selected]
        info["supplemental_supervised_tokens"] = check_train_schedule(candidate, extra_ids, train, tokenizer, label)
        base_scores, base_outputs = check_decoded(grpo["evaluation"], heldout, test, f"{label}/GRPO", 256, tokenizer)
        new_scores, new_outputs = check_decoded(candidate["evaluation"], heldout, test, f"{label}/GRPO+SFT", 256, tokenizer)
        pairs[key] = (base_scores, new_scores, base_outputs, new_outputs)
        baseline_adapter = baseline_path / f"seed-{seed}" / "grpo/final-adapter.safetensors"
        if replay_training:
            set_adapter(real_model, load_file(baseline_adapter))
            info["replayed_supervised_steps"] = verify_sft(real_model, tokenizer, train, candidate, protocol, label)
        if verify_batches is not None:
            for name, summary, adapter in (("grpo", grpo, baseline_adapter), ("aux", candidate, final_path)):
                set_adapter(real_model, load_file(adapter))
                info[f"{name}_redecoded_questions"] = replay_eval_batches(
                    real_model, tokenizer, test, heldout, summary["evaluation"], protocol, verify_batches)
    return {"heldout_count": len(heldout), "dev_count": len(protocol["dev_prompt_ids"]),
            "selected_sft_steps": selected, "development_deltas": dev_scores,
            "rollout_tokens_per_arm": [baseline_info[str(seed)]["rollout_tokens"] for seed in SEEDS],
            "training": baseline_info, **paired_accuracy(pairs, len(heldout))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "source", "baseline", "train", "test", "tokenizer", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--verify-batches", type=int, help="replay this many greedy batches per arm, or 0 for all")
    parser.add_argument("--verify-training", action="store_true", help="replay SFT for each candidate adapter")
    args = parser.parse_args()
    if args.verify_batches is not None and args.verify_batches < 0:
        parser.error("--verify-batches must be nonnegative")
    tokenizer_path = args.tokenizer.parent if args.tokenizer.is_file() else args.tokenizer
    result = audit(args.run, args.source, args.baseline, args.train, args.test, tokenizer_path, args.verify_batches,
                   args.verify_training)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Audited five paired seeds across {result['heldout_count']} unseen questions. "
          f"Passed accuracy gate: {result['qualified']}")


if __name__ == "__main__":
    main()
