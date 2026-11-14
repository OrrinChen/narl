"""Audit paired GSM8K held-out answer loss against raw rollouts and target scores."""

import argparse
from hashlib import sha256
import json
from math import fsum, isclose, isfinite
from pathlib import Path
from random import Random
from statistics import median

from experiments.audit_gsm8k_matched import score_change
from experiments.audit_gsm8k_online_lag import adapter_digest, check_evaluation, validate_sample
from experiments.audit_gsm8k_regroup import check_prompt_ids
from experiments.audit_kv_online import check_reported, read_json, read_rows, require
from experiments.gsm8k_lag_stats import paired_sign_flip_p


ARMS = ("grpo", "aux")
DEV_SEED = 61043
SCREEN_WEIGHTS = (0.1, 0.4, 1.0)


def read_split(path, checksum):
    data = path.read_bytes()
    require(sha256(data).hexdigest() == checksum, f"{path}: dataset checksum does not match")
    rows = [json.loads(line) for line in data.splitlines()]
    return [{"question": row["question"], "value": row["answer"].rsplit("####", 1)[1].strip().replace(",", "")}
            for row in rows]


def targets(tokenizer, rows, ids, prefix):
    from experiments.gsm8k_signal import prompt_text

    tag_ids = tokenizer.encode(prefix, add_special_tokens=False)
    return [(tokenizer.encode(prompt_text(tokenizer, rows[index], "reasoning"), add_special_tokens=False) + tag_ids,
             tokenizer.encode(rows[index]["value"], add_special_tokens=False)) for index in ids]


def check_loss(evaluation, ids, expected, label):
    rows = evaluation["loss_rows"]
    require([row["prompt_id"] for row in rows] == ids, f"{label}: held-out IDs or order changed")
    for row, (prompt, answer) in zip(rows, expected, strict=True):
        require(row["prompt_ids"] == prompt and row["target_token_ids"] == answer,
                f"{label}/{row['prompt_id']}: wrong prompt prefix or canonical gold answer tokens")
        nlls = row["token_nll"]
        require(len(nlls) == len(answer) and all(isfinite(nll) and nll >= 0 for nll in nlls),
                f"{label}/{row['prompt_id']}: invalid token NLL values")
    count = sum(len(answer) for _, answer in expected)
    value = fsum(nll for row in rows for nll in row["token_nll"]) / count
    check_reported(evaluation, {"token_nll": value}, label)
    return value, count


def verify_loss(model, evaluation, label, count):
    """Recompute archived losses directly from each adapter's logits."""
    import mlx.core as mx
    from experiments.gsm8k_mlx_development import suffix_logits

    rows = evaluation["loss_rows"]
    indexes = range(len(rows)) if count == 0 else sorted({i * len(rows) // count for i in range(count)})
    for index in indexes:
        row = rows[index]
        prompt, target = row["prompt_ids"], row["target_token_ids"]
        logits = suffix_logits(model, mx.array([(prompt + target)[:-1]]), len(target))[0].astype(mx.float32)
        gold = logits[mx.arange(len(target)), mx.array(target)]
        result = (mx.logsumexp(logits, axis=-1) - gold).tolist()
        require(all(isclose(actual, saved, rel_tol=1e-6, abs_tol=3e-5)
                    for actual, saved in zip(result, row["token_nll"], strict=True)),
                f"{label}/{row['prompt_id']}: archived NLL differs from forward scoring")
    return len(indexes)


def verify_training(model, tokenizer, start, report, trace, train, protocol, arm, label):
    """Rebuild auxiliary targets and GRPO advantages, then replay the optimizer."""
    import numpy as np
    from experiments.gsm8k_mlx_development import adapter_snapshot, set_adapter
    from experiments.gsm8k_mlx_grpo import MlxGrpoLearner
    from experiments.gsm8k_regroup import adapter_digest as digest_parameters

    set_adapter(model, start)
    learner = MlxGrpoLearner(model, protocol["lr"])
    aux_targets = targets(tokenizer, train, [row["prompt_id"] for row in trace], protocol["answer_prefix"])
    for row, update, (prompt, answer) in zip(trace, report["training"], aux_targets, strict=True):
        rewards = np.asarray([sample["reward"] for sample in row["samples"]], dtype=np.float32)
        advantages = ((rewards - rewards.mean()) / (rewards.std(ddof=1) + np.float32(1e-4))).tolist()
        auxiliary = (prompt, answer, protocol["aux_weight"]) if arm == "aux" else None
        actual = learner.step(row["prompt_ids"], row["samples"], advantages, auxiliary=auxiliary)
        saved = update["diagnostics"]
        require(actual.keys() == saved.keys() and actual["skipped"] == saved["skipped"]
                and all(isclose(value, saved[key], rel_tol=1e-5, abs_tol=1e-6) for key, value in actual.items()),
                f"{label}/{row['prompt_id']}: optimizer diagnostics differ from replay")
    require(digest_parameters(adapter_snapshot(model)) == report["final_sha256"],
            f"{label}: final adapter differs from optimizer replay")
    return len(trace)


def source_trace(source, seed, checksum, train, protocol):
    path = source / f"seed-{seed}" / "fresh" / "trace.jsonl"
    require(sha256(path.read_bytes()).hexdigest() == checksum, f"seed {seed}: source rollout trace changed")
    rows = read_rows(path)
    total = 0
    selected = []
    for row in rows:
        index = row["prompt_id"]
        require(type(index) is int and 0 <= index < len(train), f"seed {seed}: invalid source training prompt")
        samples = row["samples"]
        gold = float(train[index]["value"])
        n = sum(validate_sample(sample, gold, sample["token_cap"], sample["id"]) for sample in samples)
        require(len(samples) in (1, protocol["group_size"]) and row["generated_tokens"] == n,
                f"seed {seed}/{index}: source sample budget mismatch")
        expected_consumed = n if len(samples) == protocol["group_size"] else 0
        require(row["consumed_tokens"] == expected_consumed, f"seed {seed}/{index}: incomplete group handling changed")
        total += n
        if expected_consumed:
            selected.append(row)
    require(total == protocol["token_budget"], f"seed {seed}: source generated token budget differs from protocol")
    return selected


def check_training(report, trace, tokenizer, train, arm, label):
    training = report["training"]
    require(len(training) == len(trace), f"{label}: number of training attempts changed")
    for update, row in zip(training, trace, strict=True):
        index = row["prompt_id"]
        aux_tokens = len(tokenizer.encode(train[index]["value"], add_special_tokens=False)) if arm == "aux" else 0
        check_reported(update, {"prompt_id": index, "token_budget": row["consumed_tokens"], "aux_tokens": aux_tokens},
                       f"{label}/{index}")
        diagnostics = update["diagnostics"]
        require(diagnostics and all(isfinite(value) for value in diagnostics.values()),
                f"{label}/{index}: missing or invalid update diagnostics")
        rewards = {sample["reward"] for sample in row["samples"]}
        skipped = len(row["samples"]) if len(rewards) == 1 else 0
        require(diagnostics["skipped"] == skipped and diagnostics["gnorm"] >= 0,
                f"{label}/{index}: update diagnostics do not match the GRPO group")
        require(("aux_nll" in diagnostics) == (arm == "aux") and (arm != "aux" or diagnostics["aux_nll"] >= 0),
                f"{label}/{index}: supervised objective does not match the treatment arm")
    return {"steps": len(training), "rollout_tokens": sum(row["token_budget"] for row in training),
            "auxiliary_tokens": sum(row["aux_tokens"] for row in training)}


def paired_loss(baselines, adjusted):
    require(all(value > 0 for value in baselines), "relative loss requires positive baseline NLL")
    drops = [base - aux for base, aux in zip(baselines, adjusted, strict=True)]
    relative = [drop / base for drop, base in zip(drops, baselines, strict=True)]
    return {"lower_seeds": sum(value > 0 for value in drops), "median_nll_drop": median(drops),
            "median_relative_drop": median(relative), "relative_drop_range": [min(relative), max(relative)],
            "exact_seed_sign_flip_p": paired_sign_flip_p(drops)}


def check_development(run, protocol, upstream, train):
    path = run / "development.json"
    require(sha256(path.read_bytes()).hexdigest() == protocol["development_sha256"],
            "development screen does not match the frozen protocol checksum")
    report = read_json(path)
    used = {index for schedule in protocol["train_prompt_ids"].values() for index in schedule}
    excluded = set(upstream["excluded_train_ids"]) | used
    ids = Random(DEV_SEED).sample(sorted(set(range(len(train))) - excluded), 48)
    require(report["prompt_ids"] == ids and report["seed"] == 0,
            "development split differs from the independent frozen screen")
    require(not {train[index]["question"] for index in ids} & {train[index]["question"] for index in excluded},
            "a development question overlaps training or earlier development")
    require(report["source"] == protocol["source"] and report["lr"] == protocol["lr"],
            "development screening changed source rollouts or learning rate")
    losses = report["results"]
    require(set(losses) == {str(weight) for weight in (0.0, *SCREEN_WEIGHTS)}
            and all(isfinite(value) and value > 0 for value in (report["initial_nll"], *losses.values())),
            "development losses or screened candidate weights are invalid")
    selected = min(SCREEN_WEIGHTS, key=lambda weight: losses[str(weight)])
    initial = report["initial_nll"]
    passes = losses[str(selected)] <= (1 - protocol["loss_margin"]) * min(initial, losses["0.0"])
    require(passes and report["passes_screen"] is True and selected == report["selected_weight"] == protocol["aux_weight"],
            "confirmation does not use the best eligible development candidate")
    return {"count": len(ids), "selected_weight": selected, "initial_nll": initial, "results": losses}


def audit(run, source, train_path, test_path, tokenizer_path, verify_count=None, replay_training=False):
    from transformers import AutoTokenizer

    protocol = read_json(run / "protocol.json")
    upstream = read_json(source / "protocol.json")
    source_adapter = source / "current-adapter.safetensors"
    start_hash = adapter_digest(source_adapter)
    meta = protocol["source"]
    require(start_hash == meta["source_adapter_sha256"] == upstream["warmup"]["current_adapter_sha256"],
            "starting adapter does not match the previous online run")
    require(protocol["train_sha256"] == upstream["train_sha256"] and protocol["test_sha256"] == upstream["test_sha256"],
            "dataset revisions differ from source training")
    require((protocol["model"], protocol["revision"], protocol["prompt_mode"]) ==
            (upstream["model"], upstream["revision"], upstream["qualification_protocol"]["prompt_mode"])
            and protocol["prompt_mode"] == "reasoning", "model or answer context differs from the frozen source")
    train, test = read_split(train_path, protocol["train_sha256"]), read_split(test_path, protocol["test_sha256"])
    development = check_development(run, protocol, upstream, train)
    heldout, greedy_ids, seeds = (protocol[key] for key in ("heldout_prompt_ids", "greedy_prompt_ids", "seeds"))
    require(seeds == upstream["seeds"] == list(range(5)), "expected the five original paired seeds")
    require(protocol["answer_prefix"] == "<answer>" and protocol["loss_margin"] == 0.02 and protocol["loss_seed_min"] == 4,
            "answer context or predeclared loss criterion changed")
    require(isfinite(protocol["aux_weight"]) and protocol["aux_weight"] > 0
            and isfinite(protocol["lr"]) and protocol["lr"] > 0, "invalid optimizer protocol")
    check_prompt_ids(heldout, test, "loss holdout")
    excluded = set(protocol["excluded_test_ids"])
    require(set(upstream["excluded_test_ids"] + upstream["heldout_prompt_ids"]) <= excluded,
            "held-out pool did not exclude all questions inspected in earlier runs")
    allowed = [index for index in range(len(test)) if index not in excluded]
    require(Random(protocol["test_seed"]).sample(allowed, len(heldout)) == heldout,
            "loss holdout does not match the frozen unseen split")
    require(greedy_ids and greedy_ids == heldout[:len(greedy_ids)], "greedy subset differs from frozen holdout prefix")
    require(not {test[index]["question"] for index in heldout} & {row["question"] for row in train},
            "test question leaks into the training split")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    expected = targets(tokenizer, test, heldout, protocol["answer_prefix"])
    score_gold = [(row["question"], float(row["value"])) for row in test]
    greedy_cap = protocol.get("eval_max_new_tokens", upstream["eval_max_new_tokens"])
    real_model = None
    if verify_count is not None or replay_training:
        from importlib.metadata import version
        from safetensors.torch import load_file
        from experiments.gsm8k_mlx_development import load_mlx_model, set_adapter

        require((version("mlx"), version("mlx-lm")) == (protocol["mlx"], protocol["mlx_lm"]),
                "model verification requires the frozen MLX dependency versions")
        real_model, _ = load_mlx_model(tokenizer_path, len(upstream["qualification_protocol"]["lora_layers"]), 0)
        start_weights = load_file(source_adapter) if replay_training else None

    def evaluate(item, adapter, label):
        nll, tokens = check_loss(item, heldout, expected, label)
        scores = check_evaluation(item["greedy"], greedy_ids, score_gold, label, greedy_cap, tokenizer)
        verified = 0
        if verify_count is not None:
            set_adapter(real_model, load_file(adapter))
            verified = verify_loss(real_model, item, label, verify_count)
        greedy = scores, [row["token_ids"] for row in item["greedy"]["rows"]]
        question_nll = [fsum(row["token_nll"]) for row in item["loss_rows"]]
        return ({"token_nll": nll, "scored_tokens": tokens, "greedy_correct": sum(scores),
                 "recomputed_questions": verified}, greedy, question_nll)

    start, start_greedy, start_question_nll = evaluate(read_json(run / "evaluation-start.json"), source_adapter, "start")
    by_seed = {}
    for seed in seeds:
        key = str(seed)
        trace = source_trace(source, seed, meta["trace_sha256"][key], train, upstream)
        require([row["prompt_id"] for row in trace] == protocol["train_prompt_ids"][key],
                f"seed {seed}: training prompt schedule differs from source")
        by_seed[key] = {}
        arm_greedy = {}
        arm_question_nll = {}
        for arm in ARMS:
            label = f"seed {seed}/{arm}"
            arm_path = run / f"seed-{seed}" / arm
            report = read_json(arm_path / "summary.json")
            final_path = arm_path / "final-adapter.safetensors"
            require(report["start_sha256"] == start_hash and adapter_digest(final_path) == report["final_sha256"],
                    f"{label}: reported adapter hashes differ from archived weights")
            require(report["final_sha256"] != start_hash or arm != "aux", f"{label}: auxiliary training did not change weights")
            training = check_training(report, trace, tokenizer, train, arm, label)
            if replay_training:
                training["replayed_steps"] = verify_training(real_model, tokenizer, start_weights, report, trace,
                                                              train, protocol, arm, label)
            evaluation, arm_greedy[arm], arm_question_nll[arm] = evaluate(report["evaluation"], final_path, label)
            by_seed[key][arm] = {**evaluation, **training}
        current = arm_greedy["aux"]
        by_seed[key]["greedy_pairs"] = {
            label: score_change(control[0], current[0], control[1], current[1])
            for label, control in (("aux_vs_grpo", arm_greedy["grpo"]), ("aux_vs_start", start_greedy))
        }
        by_seed[key]["questions_with_lower_nll"] = {
            label: sum(new < old for new, old in zip(arm_question_nll["aux"], control, strict=True))
            for label, control in (("aux_vs_grpo", arm_question_nll["grpo"]), ("aux_vs_start", start_question_nll))
        }
    baseline = [by_seed[str(seed)]["grpo"]["token_nll"] for seed in seeds]
    candidate = [by_seed[str(seed)]["aux"]["token_nll"] for seed in seeds]
    comparisons = {"aux_vs_grpo": paired_loss(baseline, candidate),
                   "aux_vs_start": paired_loss([start["token_nll"]] * len(seeds), candidate)}
    qualified = all(value["lower_seeds"] >= protocol["loss_seed_min"]
                    and value["median_relative_drop"] >= protocol["loss_margin"] for value in comparisons.values())
    return {"development": development, "heldout_count": len(heldout), "greedy_count": len(greedy_ids),
            "initial": start, "seeds": by_seed,
            "comparisons": comparisons, "qualified": qualified}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "source", "train", "test", "tokenizer", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--verify-loss", type=int, metavar="QUESTIONS",
                        help="re-score this many questions per adapter with MLX, or 0 for all")
    parser.add_argument("--verify-training", action="store_true", help="replay optimization and verify the final adapter")
    args = parser.parse_args()
    tokenizer_path = args.tokenizer.parent if args.tokenizer.is_file() else args.tokenizer
    if args.verify_loss is not None and args.verify_loss < 0:
        parser.error("--verify-loss must be nonnegative")
    result = audit(args.run, args.source, args.train, args.test, tokenizer_path, args.verify_loss, args.verify_training)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Audited {len(result['seeds'])} paired seeds across {result['heldout_count']} unseen questions. "
          f"Passed loss gate: {result['qualified']}")


if __name__ == "__main__":
    main()
