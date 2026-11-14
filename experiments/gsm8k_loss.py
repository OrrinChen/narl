"""Screen and confirm an auxiliary gold-answer loss on paired GSM8K GRPO rollouts."""

import argparse
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from random import Random

import mlx.core as mx
import mlx.nn as nn
from safetensors.torch import load_file, save_file
import torch

from experiments.audit_kv_online import read_json, read_rows, require
from experiments.gsm8k_matched import evaluate_test, save_summary
from experiments.gsm8k_mlx_development import adapter_snapshot, load_mlx_model, set_adapter, suffix_logits
from experiments.gsm8k_mlx_generate import generate_batch
from experiments.gsm8k_mlx_grpo import MlxGrpoLearner
from experiments.gsm8k_online_lag import advantages
from experiments.gsm8k_regroup import adapter_digest
from experiments.gsm8k_signal import MODEL, REVISION, TEST_SHA256, TRAIN_SHA256, prompt_text, read_split


ANSWER_PREFIX = "<answer>"
TEST_SEED = 60381
DEV_SEED = 61043
SEEDS = list(range(5))
MARGIN = 0.02
MIN_SEEDS = 4
SCREEN_WEIGHTS = (0.1, 0.4, 1.0)


def target_tokens(tokenizer, example, mode):
    ids = tokenizer.encode(prompt_text(tokenizer, example, mode), add_special_tokens=False)
    ids += tokenizer.encode(ANSWER_PREFIX, add_special_tokens=False)
    target = tokenizer.encode(example["value"], add_special_tokens=False)
    require(bool(target), "empty answer target")
    return ids, target


def score_loss(model, examples):
    rows = []
    for prompt_id, prompt_ids, target in examples:
        inputs = mx.array([prompt_ids + target[:-1]])
        logits = suffix_logits(model, inputs, len(target)).astype(mx.float32)
        nll = nn.losses.cross_entropy(logits, mx.array([target]), reduction="none")[0].tolist()
        rows.append({"prompt_id": prompt_id, "prompt_ids": prompt_ids, "target_token_ids": target, "token_nll": nll})
    return {"loss_rows": rows, "token_nll": sum(sum(row["token_nll"]) for row in rows) /
            sum(len(row["token_nll"]) for row in rows)}


def load_source(source):
    meta = read_json(source / "protocol.json")
    current = load_file(source / "current-adapter.safetensors")
    current_hash = adapter_digest(current)
    require(current_hash == meta["warmup"]["current_adapter_sha256"], "source start adapter differs from its hash")
    paths = {str(seed): source / f"seed-{seed}" / "fresh" / "trace.jsonl" for seed in SEEDS}
    records = {seed: [row for row in read_rows(path) if row["consumed_tokens"]] for seed, path in paths.items()}
    require(all(sum(row["consumed_tokens"] for row in rows) == meta["token_budget"] for rows in records.values()),
            "source must consume all emitted tokens in every seed")
    source_meta = {"source_adapter_sha256": current_hash,
                   "trace_sha256": {seed: sha256(path.read_bytes()).hexdigest() for seed, path in paths.items()}}
    return meta, current, records, source_meta


def train_arm(model, tokenizer, train, records, start, weight, lr, mode):
    set_adapter(model, start)
    learner = MlxGrpoLearner(model, lr)
    updates = []
    for row in records:
        values, _ = advantages(row["prompt_ids"], row["samples"])
        auxiliary, aux_tokens = None, 0
        if weight:
            prompt, target = target_tokens(tokenizer, train[row["prompt_id"]], mode)
            auxiliary = (prompt, target, weight)
            aux_tokens = len(target)
        diagnostics = learner.step(row["prompt_ids"], row["samples"], values, auxiliary=auxiliary)
        updates.append({"prompt_id": row["prompt_id"], "token_budget": row["consumed_tokens"],
                        "aux_tokens": aux_tokens, "diagnostics": diagnostics})
    snapshot = adapter_snapshot(model)
    mx.clear_cache()
    return updates, snapshot


def evaluate(model, tokenizer, examples, score_rows, greedy_ids, cap, mode):
    report = score_loss(model, score_rows)
    if greedy_ids:
        protocol = {"heldout_prompt_ids": greedy_ids, "eval_batch": 8, "eval_max_new_tokens": cap,
                    "qualification_protocol": {"prompt_mode": mode}}
        report["greedy"] = evaluate_test(model, tokenizer, examples, protocol, generate_batch)
    return report


def encoded_rows(tokenizer, examples, ids, mode):
    return [(i, *target_tokens(tokenizer, examples[i], mode)) for i in ids]


def screen(model, tokenizer, train, meta, records, start, protocol, out):
    mode = meta["qualification_protocol"]["prompt_mode"]
    used = {row["prompt_id"] for rows in records.values() for row in rows}
    excluded = set(meta["excluded_train_ids"]) | used
    ids = Random(DEV_SEED).sample(sorted(set(range(len(train))) - excluded), 48)
    eval_rows = encoded_rows(tokenizer, train, ids, mode)
    set_adapter(model, start)
    initial = score_loss(model, eval_rows)["token_nll"]
    report = {"prompt_ids": ids, "seed": 0, "initial_nll": initial, "results": {}, "source": protocol["source"],
              "lr": protocol["lr"]}
    for weight in (0.0, *SCREEN_WEIGHTS):
        train_arm(model, tokenizer, train, records["0"], start, weight, protocol["lr"], mode)
        loss = score_loss(model, eval_rows)["token_nll"]
        report["results"][str(weight)] = loss
        print(f"development weight {weight}: {loss:.6f} nats/token (initial {initial:.6f})", flush=True)
    # Freeze one candidate based only on the disjoint training-split development set.
    selected = min(SCREEN_WEIGHTS, key=lambda weight: report["results"][str(weight)])
    report["selected_weight"] = selected
    report["passes_screen"] = report["results"][str(selected)] <= (1 - MARGIN) * min(initial, report["results"]["0.0"])
    save_summary(out / "development.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "test", "source", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--heldout", type=int, default=64)
    parser.add_argument("--greedy", type=int, default=24)
    parser.add_argument("--lr", type=float, help="shared optimizer learning rate for both arms")
    parser.add_argument("--screen-only", action="store_true")
    args = parser.parse_args()
    require(0 < args.greedy <= args.heldout and (args.lr is None or args.lr > 0), "invalid evaluation or optimizer budget")
    torch.set_num_threads(1)
    train, test = read_split(args.train, TRAIN_SHA256), read_split(args.test, TEST_SHA256)
    meta, start, records, source = load_source(args.source)
    excluded = sorted(set(meta["excluded_test_ids"] + meta["heldout_prompt_ids"]))
    heldout = Random(TEST_SEED).sample(sorted(set(range(len(test))) - set(excluded)), args.heldout)
    require(not {row["question"] for row in train} & {test[i]["question"] for i in heldout}, "heldout questions overlap training")
    mode = meta["qualification_protocol"]["prompt_mode"]
    protocol = {"model": MODEL, "revision": REVISION, "train_sha256": TRAIN_SHA256, "test_sha256": TEST_SHA256,
                "seeds": SEEDS, "train_prompt_ids": {seed: [row["prompt_id"] for row in rows] for seed, rows in records.items()},
                "excluded_test_ids": excluded, "test_seed": TEST_SEED, "heldout_prompt_ids": heldout,
                "greedy_prompt_ids": heldout[:args.greedy], "answer_prefix": ANSWER_PREFIX, "prompt_mode": mode,
                "loss_metric": "gold numeric token mean NLL in nats after forced opening tag; excludes tags and EOS",
                "loss_margin": MARGIN, "loss_seed_min": MIN_SEEDS, "source": source, "lr": args.lr or meta["lr"],
                "eval_max_new_tokens": meta["eval_max_new_tokens"], "mlx": version("mlx"), "mlx_lm": version("mlx-lm")}
    args.out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_mlx_model(None, len(meta["qualification_protocol"]["lora_layers"]), TEST_SEED)
    dev_path = args.out / "development.json"
    dev = read_json(dev_path) if dev_path.exists() else screen(model, tokenizer, train, meta, records, start, protocol, args.out)
    require(dev["source"] == source and dev.get("lr", meta["lr"]) == protocol["lr"],
            "development screen used different source rollouts or learning rate")
    if args.screen_only or not dev["passes_screen"]:
        return
    protocol["aux_weight"] = dev["selected_weight"]
    protocol["development_sha256"] = sha256(dev_path.read_bytes()).hexdigest()
    path = args.out / "protocol.json"
    if path.exists():
        require(read_json(path) == protocol, "settings differ from the frozen confirmation protocol")
    else:
        save_summary(path, protocol)
    eval_rows = encoded_rows(tokenizer, test, heldout, mode)
    start_path = args.out / "evaluation-start.json"
    if not start_path.exists():
        set_adapter(model, start)
        result = evaluate(model, tokenizer, test, eval_rows, protocol["greedy_prompt_ids"], protocol["eval_max_new_tokens"], mode)
        save_summary(start_path, result)
        print(f"starting NLL {result['token_nll']:.6f}, greedy {result['greedy']['correct']}/{args.greedy}", flush=True)
    for seed in SEEDS:
        for arm, weight in (("grpo", 0.0), ("aux", protocol["aux_weight"])):
            folder = args.out / f"seed-{seed}" / arm
            path = folder / "summary.json"
            if path.exists():
                continue
            folder.mkdir(parents=True, exist_ok=True)
            updates, snapshot = train_arm(model, tokenizer, train, records[str(seed)], start, weight, protocol["lr"], mode)
            result = evaluate(model, tokenizer, test, eval_rows, protocol["greedy_prompt_ids"], protocol["eval_max_new_tokens"], mode)
            save_file(snapshot, folder / "final-adapter.safetensors")
            save_summary(path, {"evaluation": result, "training": updates, "start_sha256": adapter_digest(start),
                                "final_sha256": adapter_digest(snapshot)})
            print(f"seed {seed} {arm}: NLL {result['token_nll']:.6f}, greedy {result['greedy']['correct']}/{args.greedy}", flush=True)


if __name__ == "__main__":
    main()
