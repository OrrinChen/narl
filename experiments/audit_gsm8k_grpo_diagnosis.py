"""Audit GRPO drift diagnostics without rescoring any new official test answers."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path
import re
from statistics import mean, median

from experiments.audit_gsm8k_budget import check_grpo_trace
from experiments.audit_gsm8k_online_lag import (
    MODEL, REVISION, TRAIN_SHA256, adapter_digest, check_diagnostics, check_evaluation, normalized_advantages,
)
from experiments.audit_kv_online import check_reported, read_json, read_rows, require


CHECKPOINTS = (0, 6, 10, 11, 12, 13, 15, 16)


def file_digest(path):
    return sha256(path.read_bytes()).hexdigest()


def metrics(evaluation):
    rows = evaluation["rows"]
    return {"correct": sum(row["reward"] for row in rows), "count": len(rows),
            "truncated": sum(row["truncated"] for row in rows),
            "tagged": sum(row["has_answer_tag"] for row in rows),
            "median_tokens": median(len(row["token_ids"]) for row in rows),
            "pure_answer": sum(re.fullmatch(r"\s*<answer>[^<>]+</answer>\s*", row["completion"]) is not None
                               for row in rows)}


def check_probe(probe, label):
    keys = ("first_token_ref_kl_mean", "first_token_ref_kl_median",
            "answer_opening_probability_mean", "answer_opening_probability_median")
    require(set(probe) == set(keys) and all(isfinite(probe[key]) for key in keys), f"{label}: invalid probe fields")
    require(all(probe[key] >= -1e-6 for key in keys[:2]) and
            all(0 <= probe[key] <= 1 for key in keys[2:]), f"{label}: invalid KL or opening probability")


def check_eval(report, ids, gold, tokenizer, label):
    check_evaluation(report["evaluation"], ids, gold, label, 256, tokenizer)
    derived = metrics(report["evaluation"])
    check_reported(report["metrics"], derived, label)
    return derived


def trace_analysis(source):
    result = {}
    for seed in range(5):
        folder = source / f"seed-{seed}" / "grpo"
        rows = read_rows(folder / "trace.jsonl")
        by_version = {}
        for row in rows:
            by_version.setdefault(str(row["current_version"]), []).extend(row["samples"])
        versions = {}
        for version, samples in by_version.items():
            versions[version] = {"count": len(samples), "correct": sum(s["reward"] for s in samples),
                                 "median_tokens": median(len(s["token_ids"]) for s in samples),
                                 "pure_answer": sum(re.fullmatch(r"\s*<answer>[^<>]+</answer>\s*", s["completion"])
                                                    is not None for s in samples)}
        summary = read_json(folder / "summary.json")
        result[str(seed)] = {"accounting": summary["accounting"], "evaluation": metrics(summary["evaluation"]),
                             "reward_patterns": dict(Counter("".join(str(s["reward"]) for s in row["samples"])
                                                              for row in rows)), "versions": versions,
                             "active_gradient_norms": [r["diagnostics"]["gnorm"] for r in rows
                                                       if r["nonzero_advantage_tokens"]]}
    return result


def check_replay(folder, lr, trace, source, protocol, gold, tokenizer):
    report = read_json(folder / "summary.json")
    require(report["lr"] == lr and report["kind"] == "fixed original trajectory replay; low-LR replay is off-policy",
            f"{folder.name}: replay learning rate or off-policy declaration changed")
    require(report["generated_tokens_in_source"] == sum(row["generated_tokens"] for row in trace) == 32768,
            f"{folder.name}: fixed source budget changed")
    active = [row for row in trace if row["nonzero_advantage_tokens"]]
    require(len(report["updates"]) == len(active) == 16, f"{folder.name}: missing effective source groups")
    for row, update in zip(active, report["updates"], strict=True):
        label = f"{folder.name}/update {row['next_version']}"
        check_reported(update, {"source_update": row["next_version"], "group_index": row["group_index"],
                                "prompt_id": row["prompt_id"]}, label)
        check_diagnostics(update["diagnostics"], normalized_advantages(row["samples"]), label)
        require(isfinite(update["own_step_ratio_outside_0.8_1.2"])
                and 0 <= update["own_step_ratio_outside_0.8_1.2"] <= 1
                and isfinite(update["own_step_abs_logratio_mean"]) and update["own_step_abs_logratio_mean"] >= 0,
                f"{label}: invalid own-step logratio diagnostics")
        if lr == source["grpo_lr"]:
            require(update["diagnostics"].keys() == row["diagnostics"].keys()
                    and all(isclose(value, row["diagnostics"][key], rel_tol=1e-5, abs_tol=1e-6)
                            for key, value in update["diagnostics"].items()), f"{label}: original replay changed")
    expected_hashes = {0: source["start_sha256"], **{row["next_version"]: row["next_sha256"] for row in active}}
    require(set(report["checkpoints"]) == {str(version) for version in CHECKPOINTS},
            f"{folder.name}: localization checkpoints changed")
    checkpoints = {}
    for version in CHECKPOINTS:
        label = f"{folder.name}/checkpoint {version}"
        checkpoint = read_json(folder / f"update-{version}.json")
        require(checkpoint == report["checkpoints"][str(version)] and checkpoint["source_update"] == version,
                f"{label}: changed checkpoint archive")
        digest = adapter_digest(folder / f"update-{version}.safetensors")
        require(checkpoint["adapter_sha256"] == digest, f"{label}: saved adapter hash mismatch")
        if version == 0 or lr == source["grpo_lr"]:
            require(digest == expected_hashes[version], f"{label}: original parameter chain mismatch")
        checks = check_eval(checkpoint, protocol["localization_prompt_ids"], gold, tokenizer, label)
        check_probe(checkpoint["probe"], label)
        if version == 0:
            require(abs(checkpoint["probe"]["first_token_ref_kl_mean"]) <= 1e-6 and
                    abs(checkpoint["probe"]["first_token_ref_kl_median"]) <= 1e-6, f"{label}: nonzero initial KL")
        checkpoints[str(version)] = {**checks, **checkpoint["probe"]}
    require(report["final_sha256"] == adapter_digest(folder / "final-adapter.safetensors") ==
            report["checkpoints"]["16"]["adapter_sha256"], f"{folder.name}: wrong replay endpoint")
    endpoint = check_eval(report, protocol["development_prompt_ids"], gold, tokenizer, folder.name)
    return {"lr": lr, "development": endpoint, "checkpoints": checkpoints,
            "mean_own_step_ratio_outside_0.8_1.2": mean(r["own_step_ratio_outside_0.8_1.2"] for r in report["updates"]),
            "max_own_step_ratio_outside_0.8_1.2": max(r["own_step_ratio_outside_0.8_1.2"] for r in report["updates"]),
            "final_sha256": report["final_sha256"]}


def verify_online_training(source_path, rows, report, source, tokenizer_path):
    from importlib.metadata import version

    import numpy as np
    from safetensors.torch import load_file
    import torch

    from experiments.gsm8k_mlx_development import adapter_snapshot, load_mlx_model, set_adapter
    from experiments.gsm8k_mlx_grpo import MlxGrpoLearner
    from experiments.gsm8k_regroup import adapter_digest as tensor_digest

    require((version("mlx"), version("mlx-lm")) == (source["mlx"], source["mlx_lm"]), "MLX replay versions changed")
    torch.set_num_threads(1)
    model, _ = load_mlx_model(tokenizer_path, len(source["qualification_protocol"]["lora_layers"]), source["dev_seed"])
    set_adapter(model, load_file(source_path / "start-adapter.safetensors"))
    learner = MlxGrpoLearner(model, 0.0001)
    batches = 0
    for row in rows:
        if row["diagnostics"] is not None:
            rewards = np.asarray([s["reward"] for s in row["samples"]], dtype=np.float32)
            values = ((rewards - rewards.mean()) / (rewards.std(ddof=1) + np.float32(1e-4))).tolist()
            actual = learner.step(row["prompt_ids"], row["samples"], values)
            expected = row["diagnostics"]
            require(actual.keys() == expected.keys() and
                    all(isclose(value, expected[key], rel_tol=1e-5, abs_tol=1e-6) for key, value in actual.items()),
                    f"online group {row['group_index']}: optimizer diagnostics mismatch")
            batches += 1
        require(tensor_digest(adapter_snapshot(model)) == row["next_sha256"],
                f"online group {row['group_index']}: optimizer parameter bytes mismatch")
    require(tensor_digest(adapter_snapshot(model)) == report["final_sha256"], "online optimizer endpoint mismatch")
    return {"groups": len(rows), "learner_batches": batches, "changed_weights": report["accounting"]["updates"],
            "endpoint_matches": True, "every_group_hash_matches": True}


def audit(source_path, run, train_path, tokenizer_path, verify_online=False):
    from transformers import AutoTokenizer

    from experiments.gsm8k_signal import read_split

    train = read_split(train_path, TRAIN_SHA256)
    gold = [(row["question"], float(row["value"])) for row in train]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    source = read_json(source_path / "protocol.json")
    protocol = read_json(run / "protocol.json")
    require((source["model"], source["revision"], source["train_sha256"]) == (MODEL, REVISION, TRAIN_SHA256)
            and protocol["source_protocol"] == source
            and protocol["source_protocol_sha256"] == file_digest(source_path / "protocol.json")
            and protocol["source_trace_sha256"] == file_digest(source_path / "seed-4/grpo/trace.jsonl"),
            "diagnosis source/model/data provenance changed")
    require(read_json(source_path / "audit.json")["model_replay"]["replayed_seeds"] == 5
            and adapter_digest(source_path / "start-adapter.safetensors") == source["start_sha256"],
            "original experiment or common initialization lacks matching audit")
    require(protocol["seed"] == 4 and protocol["source_high_lr"] == source["grpo_lr"] == 0.0006
            and protocol["low_lr"] == 0.0001 and protocol["checkpoints"] == list(CHECKPOINTS)
            and protocol["development_prompt_ids"] == source["dev_prompt_ids"]
            and protocol["localization_prompt_ids"] == source["dev_prompt_ids"][:24]
            and protocol["probe_prompt_ids"] == source["dev_prompt_ids"][:12]
            and len(source["dev_prompt_ids"]) == 48 and source["total_tokens"] == 32768
            and source["combo_rollout_tokens"] == 6144 and source["group_size"] == 2,
            "diagnostic seed, budgets, recipes, or development split changed")
    require(protocol["evaluation_policy"] ==
            "development only; no new scoring of official test; original results immutable" and
            protocol["online_control"] ==
            "same start, seed, prompt order, G2, reward, optimizer, and 32768-token budget; LR only" and
            protocol["stopping_rule"] ==
            "one fixed high/low replay pair and one prespecified 1e-4 on-policy seed; no search until gain",
            "diagnostic scope or stopping rule changed")
    dev_ids = set(protocol["development_prompt_ids"])
    for field in ("grpo_prompt_ids", "sft_prompt_ids"):
        require(all(not dev_ids.intersection(ids) for ids in source[field].values()), "development reused in training")
    derived_trace = trace_analysis(source_path)
    require(read_json(run / "trace-analysis.json") == derived_trace, "descriptive trace analysis disagrees with source")
    trace, _, _ = check_grpo_trace(source_path, 4, source, train, tokenizer)
    initial = read_json(run / "initial-development.json")
    initial_metrics = check_eval(initial, protocol["development_prompt_ids"], gold, tokenizer, "initial")
    require(isfinite(initial["answer_opening_probability_mean"]) and
            0 <= initial["answer_opening_probability_mean"] <= 1, "invalid initial opening probability")
    prior = read_json(source_path / "development.json")["initial_evaluation"]
    require([r["token_ids"] for r in initial["evaluation"]["rows"]] ==
            [r["token_ids"] for r in prior["rows"]], "common initial development decode changed")
    replays = {}
    for name, lr in (("replay-high", 0.0006), ("replay-low", 0.0001)):
        replays[name] = check_replay(run / name, lr, trace, source, protocol, gold, tokenizer)
        zero = read_json(run / name / "update-0.json")
        require(zero["evaluation"]["rows"] == initial["evaluation"]["rows"][:24], "initial localization decode changed")
    low_protocol = {**source, "grpo_lr": 0.0001}
    online, training, prefix = check_grpo_trace(run, 4, low_protocol, train, tokenizer, folder=run / "online-low")
    online_reports = {}
    for checkpoint, archive in (("prefix", prefix), ("final", training)):
        report = read_json(run / "online-low" / f"{checkpoint}-development.json")
        label = f"online-low/{checkpoint}"
        checked = check_eval(report, protocol["development_prompt_ids"], gold, tokenizer, label)
        check_probe(report["probe"], label)
        require(report["accounting"] == archive["accounting"], f"{label}: budget accounting changed")
        online_reports[checkpoint] = {"development": checked, "probe": report["probe"],
                                      "accounting": report["accounting"], "adapter_sha256": archive["final_sha256"]}
    result = {"audit_passed": True, "initial_development": initial_metrics, "replays": replays,
              "online_low": online_reports, "source_trace_analysis": derived_trace,
              "rescore_scope": "new evaluations use only 48 training-split development questions",
              "probe_values_range_checked": True, "probe_values_model_recomputed": False,
              "offline_low_training_independently_replayed": False,
              "online_training_independently_replayed": verify_online}
    if verify_online:
        result["online_optimizer_replay"] = verify_online_training(source_path, online, training, source, tokenizer_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/gsm8k-budget-5seed"))
    parser.add_argument("--train", type=Path, default=Path("runs/gsm8k-data/train.jsonl"))
    for key in ("run", "tokenizer", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--verify-online-training", action="store_true")
    args = parser.parse_args()
    path = args.tokenizer.parent if args.tokenizer.is_file() else args.tokenizer
    result = audit(args.source, args.run, args.train, path, args.verify_online_training)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Audited source replay checkpoints, development scores, and the 32,768-token low-LR online control.")


if __name__ == "__main__":
    main()
