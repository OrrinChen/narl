"""Localize GRPO seed-4 drift with fixed-trace replay and one on-policy LR control."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import mean, median

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors.torch import load_file, save_file
import torch

from experiments.audit_kv_online import read_json, read_rows, require
from experiments.gsm8k_budget import evaluate, train_grpo
from experiments.gsm8k_mlx_development import adapter_snapshot, load_mlx_model, set_adapter, suffix_logits
from experiments.gsm8k_mlx_grpo import MlxGrpoLearner
from experiments.gsm8k_online_lag import advantages
from experiments.gsm8k_regroup import adapter_digest
from experiments.gsm8k_signal import TRAIN_SHA256, prompt_text, read_split
from experiments.gsm8k_matched import save_summary


CHECKPOINTS = (0, 6, 10, 11, 12, 13, 15, 16)
LOW_LR = 0.0001
PURE_ANSWER = re.compile(r"\s*<answer>[^<>]+</answer>\s*")


def file_digest(path):
    return sha256(path.read_bytes()).hexdigest()


def output_metrics(evaluation):
    rows = evaluation["rows"]
    return {"correct": evaluation["correct"], "count": len(rows), "truncated": evaluation["truncated"],
            "tagged": evaluation["tagged"], "median_tokens": median(len(row["token_ids"]) for row in rows),
            "pure_answer": sum(bool(PURE_ANSWER.fullmatch(row["completion"])) for row in rows)}


def inspect(source):
    result = {}
    for seed in range(5):
        folder = source / f"seed-{seed}" / "grpo"
        trace = read_rows(folder / "trace.jsonl")
        version_rows = {}
        for row in trace:
            version_rows.setdefault(str(row["current_version"]), []).extend(row["samples"])
        versions = {}
        for version, samples in version_rows.items():
            versions[version] = {"count": len(samples), "correct": sum(s["reward"] for s in samples),
                                 "median_tokens": median(len(s["token_ids"]) for s in samples),
                                 "pure_answer": sum(bool(PURE_ANSWER.fullmatch(s["completion"])) for s in samples)}
        active = [row for row in trace if row["nonzero_advantage_tokens"]]
        report = read_json(folder / "summary.json")
        result[str(seed)] = {"accounting": report["accounting"], "evaluation": output_metrics(report["evaluation"]),
                             "reward_patterns": dict(Counter("".join(str(s["reward"]) for s in r["samples"])
                                                              for r in trace)), "versions": versions,
                             "active_gradient_norms": [r["diagnostics"]["gnorm"] for r in active]}
    return result


def fixed_prompt_probe(model, tokenizer, examples, ids):
    """Full-vocabulary distributions at the first generation position on fixed prompts."""
    opening = tokenizer.encode("<answer>", add_special_tokens=False)
    distributions, opening_probabilities = [], []
    for index in ids:
        prompt = tokenizer.encode(prompt_text(tokenizer, examples[index], "reasoning"), add_special_tokens=False)
        logits = suffix_logits(model, mx.array([prompt + opening[:-1]]), len(opening))[0].astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        chosen = mx.take_along_axis(logp, mx.array(opening)[:, None], axis=-1).sum()
        mx.eval(logp, chosen)
        distributions.append(np.array(logp[0]))
        opening_probabilities.append(float(mx.exp(chosen).item()))
    return distributions, opening_probabilities


def probe_change(model, tokenizer, examples, ids, reference):
    current, opening = fixed_prompt_probe(model, tokenizer, examples, ids)
    kl = [float((np.exp(ref) * (ref - now)).sum()) for ref, now in zip(reference, current, strict=True)]
    return {"first_token_ref_kl_mean": mean(kl), "first_token_ref_kl_median": median(kl),
            "answer_opening_probability_mean": mean(opening), "answer_opening_probability_median": median(opening)}


def selected_logps(model, prompt_ids, samples):
    rows = []
    for sample in samples:
        tokens = sample["token_ids"]
        logits = suffix_logits(model, mx.array([prompt_ids + tokens[:-1]]), len(tokens)).astype(mx.float32)
        logp = -nn.losses.cross_entropy(logits, mx.array([tokens]), reduction="none")[0]
        mx.eval(logp)
        rows.append(np.array(logp))
    return np.concatenate(rows)


def snapshot_evaluation(model, tokenizer, train, protocol, reference, folder, update):
    path = folder / f"update-{update}.json"
    if path.exists():
        return read_json(path)
    snapshot = adapter_snapshot(model)
    save_file(snapshot, folder / f"update-{update}.safetensors")
    mx.clear_cache()
    evaluation = evaluate(model, tokenizer, train, protocol["localization_prompt_ids"], protocol["source_protocol"])
    probe = probe_change(model, tokenizer, train, protocol["probe_prompt_ids"], reference)
    report = {"source_update": update, "adapter_sha256": adapter_digest(snapshot), "evaluation": evaluation,
              "metrics": output_metrics(evaluation), "probe": probe}
    save_summary(path, report)
    print(folder.name, "update", update, report["metrics"], probe, flush=True)
    return report


def replay(model, tokenizer, train, protocol, start, trace, lr, folder, reference):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "summary.json").exists():
        return read_json(folder / "summary.json")
    set_adapter(model, start)
    learner = MlxGrpoLearner(model, lr)
    snapshots = {"0": snapshot_evaluation(model, tokenizer, train, protocol, reference, folder, 0)}
    updates = []
    for row in trace:
        if row["diagnostics"] is not None:
            values, signal = advantages(row["prompt_ids"], row["samples"])
            before = selected_logps(model, row["prompt_ids"], row["samples"]) if signal else None
            diagnostics = learner.step(row["prompt_ids"], row["samples"], values)
            if signal:
                after = selected_logps(model, row["prompt_ids"], row["samples"])
                ratio = np.exp(after - before)
                updates.append({"source_update": row["next_version"], "group_index": row["group_index"],
                                "prompt_id": row["prompt_id"], "diagnostics": diagnostics,
                                "own_step_ratio_outside_0.8_1.2": float(np.mean((ratio < 0.8) | (ratio > 1.2))),
                                "own_step_abs_logratio_mean": float(np.mean(np.abs(after - before)))})
        snapshot = adapter_snapshot(model)
        if lr == protocol["source_protocol"]["grpo_lr"]:
            require(adapter_digest(snapshot) == row["next_sha256"], "original replay diverged from saved weights")
        if row["nonzero_advantage_tokens"] and row["next_version"] in CHECKPOINTS:
            version = row["next_version"]
            snapshots[str(version)] = snapshot_evaluation(model, tokenizer, train, protocol, reference, folder, version)
    save_file(adapter_snapshot(model), folder / "final-adapter.safetensors")
    evaluation = evaluate(model, tokenizer, train, protocol["development_prompt_ids"], protocol["source_protocol"])
    report = {"lr": lr, "kind": "fixed original trajectory replay; low-LR replay is off-policy",
              "generated_tokens_in_source": sum(row["generated_tokens"] for row in trace),
              "updates": updates, "checkpoints": snapshots, "evaluation": evaluation,
              "metrics": output_metrics(evaluation), "final_sha256": adapter_digest(adapter_snapshot(model))}
    save_summary(folder / "summary.json", report)
    print(folder.name, "full development", report["metrics"], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/gsm8k-budget-5seed"))
    parser.add_argument("--train", type=Path, default=Path("runs/gsm8k-data/train.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("inspect", "replay", "online", "all"), default="all")
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = read_json(args.source / "protocol.json")
    require(read_json(args.source / "audit.json")["model_replay"]["replayed_seeds"] == 5, "source lacks full replay audit")
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = {"source_protocol": source, "source_protocol_sha256": file_digest(args.source / "protocol.json"),
                "source_trace_sha256": file_digest(args.source / "seed-4/grpo/trace.jsonl"), "seed": 4,
                "source_high_lr": source["grpo_lr"], "low_lr": LOW_LR, "checkpoints": list(CHECKPOINTS),
                "localization_prompt_ids": source["dev_prompt_ids"][:24],
                "probe_prompt_ids": source["dev_prompt_ids"][:12], "development_prompt_ids": source["dev_prompt_ids"],
                "evaluation_policy": "development only; no new scoring of official test; original results immutable",
                "online_control": "same start, seed, prompt order, G2, reward, optimizer, and 32768-token budget; LR only",
                "stopping_rule": "one fixed high/low replay pair and one prespecified 1e-4 on-policy seed; no search until gain"}
    path = args.out / "protocol.json"
    if path.exists():
        require(read_json(path) == protocol, "diagnostic settings changed")
    else:
        save_summary(path, protocol)
    facts = inspect(args.source)
    save_summary(args.out / "trace-analysis.json", facts)
    print({seed: row["evaluation"] for seed, row in facts.items()}, flush=True)
    if args.stage == "inspect":
        return
    train = read_split(args.train, TRAIN_SHA256)
    start = load_file(args.source / "start-adapter.safetensors")
    require(adapter_digest(start) == source["start_sha256"], "diagnostic starting weights changed")
    trace = read_rows(args.source / "seed-4/grpo/trace.jsonl")
    model, tokenizer = load_mlx_model(None, len(source["qualification_protocol"]["lora_layers"]), source["dev_seed"])
    set_adapter(model, start)
    reference, opening = fixed_prompt_probe(model, tokenizer, train, protocol["probe_prompt_ids"])
    initial_path = args.out / "initial-development.json"
    if not initial_path.exists():
        evaluation = evaluate(model, tokenizer, train, protocol["development_prompt_ids"], source)
        save_summary(initial_path, {"evaluation": evaluation, "metrics": output_metrics(evaluation),
                                  "answer_opening_probability_mean": mean(opening)})
        print("initial full development", output_metrics(evaluation), flush=True)
    if args.stage in ("replay", "all"):
        replay(model, tokenizer, train, protocol, start, trace, source["grpo_lr"], args.out / "replay-high", reference)
        replay(model, tokenizer, train, protocol, start, trace, LOW_LR, args.out / "replay-low", reference)
    if args.stage in ("online", "all"):
        low_protocol = {**source, "grpo_lr": LOW_LR}
        folder = args.out / "online-low"
        report = train_grpo(model, tokenizer, train, low_protocol, 4, start, folder)
        for checkpoint, name in (("prefix", "prefix-adapter.safetensors"), ("final", "final-adapter.safetensors")):
            target = folder / f"{checkpoint}-development.json"
            if target.exists():
                continue
            set_adapter(model, load_file(folder / name))
            evaluation = evaluate(model, tokenizer, train, protocol["development_prompt_ids"], source)
            save_summary(target, {"evaluation": evaluation, "metrics": output_metrics(evaluation),
                                  "probe": probe_change(model, tokenizer, train, protocol["probe_prompt_ids"], reference),
                                  "accounting": read_json(folder / "prefix.json")["accounting"] if checkpoint == "prefix"
                                  else report["accounting"]})
            print("on-policy low", checkpoint, output_metrics(evaluation), flush=True)


if __name__ == "__main__":
    main()
