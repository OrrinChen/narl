"""Qualify a GSM8K LoRA adapter using the native Apple Silicon MLX backend."""

import argparse
from importlib.metadata import version
import json
from math import isfinite
from pathlib import Path
from random import Random
import re
from statistics import median

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import load_model
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
import torch
from transformers import AutoTokenizer

from experiments.gsm8k_regroup import adapter_digest
from experiments.gsm8k_signal import MODEL, REVISION, TRAIN_SHA256, prompt_text, read_split
from experiments.regroup import write_json


TARGETS = ["q_proj", "v_proj"]
RANK = 4
ALPHA = 8
CACHE_LIMIT_BYTES = 256 * 1024**2


def load_mlx_model(model_path, last_layers, seed):
    path = Path(model_path or snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    mx.set_cache_limit(CACHE_LIMIT_BYTES)
    mx.random.seed(seed)
    model, config = load_model(path)
    if config.get("quantization") or config.get("quantization_config"):
        raise ValueError("This runner requires the unquantized Qwen2.5-0.5B-Instruct checkpoint")
    if model.model.embed_tokens.weight.dtype != mx.bfloat16 or not 0 < last_layers <= len(model.layers):
        raise ValueError("Expected BF16 base weights and a valid number of LoRA layers")
    model.freeze()
    linear_to_lora_layers(model, last_layers, {
        "rank": RANK, "scale": ALPHA / RANK, "dropout": 0.0,
        "keys": [f"self_attn.{target}" for target in TARGETS],
    })
    if len(tree_flatten(model.trainable_parameters())) != 2 * len(TARGETS) * last_layers:
        raise RuntimeError("Unexpected trainable LoRA parameter set")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    eos = json.loads((path / "generation_config.json").read_text())["eos_token_id"]
    tokenizer.gsm8k_stop_ids = set(eos if isinstance(eos, list) else [eos])
    return model, tokenizer


def peft_name(name):
    stem, kind = name.rsplit(".", 1)
    if kind not in ("lora_a", "lora_b"):
        raise ValueError(f"Unexpected adapter tensor {name}")
    return f"base_model.model.{stem}.lora_{kind[-1].upper()}.weight"


def adapter_snapshot(model):
    # MLX stores A as [in, rank] and B as [rank, out]. PEFT uses their transposes.
    return {peft_name(name): torch.tensor(value.T.astype(mx.float32).tolist(), dtype=torch.float32)
            for name, value in tree_flatten(model.trainable_parameters())}


def set_adapter(model, snapshot):
    leaves = tree_flatten(model.trainable_parameters())
    expected = {peft_name(name): (name, value) for name, value in leaves}
    if snapshot.keys() != expected.keys():
        raise ValueError("Adapter checkpoint keys do not match the MLX model")
    weights = []
    for key, (name, current) in expected.items():
        value = snapshot[key]
        if tuple(value.shape[::-1]) != current.shape:
            raise ValueError(f"Adapter tensor shape mismatch for {key}")
        weights.append((name, mx.array(value.detach().cpu().numpy().T, dtype=current.dtype)))
    model.load_weights(weights, strict=False)
    mx.eval(model.trainable_parameters())


def suffix_logits(model, input_ids, length):
    hidden = model.model(input_ids)[:, -length:, :]
    return model.model.embed_tokens.as_linear(hidden) if model.args.tie_word_embeddings else model.lm_head(hidden)


def evaluate(model, tokenizer, train, ids, cap, mode, batch_size):
    from experiments.gsm8k_mlx_generate import generate_batch

    rows = []
    for start in range(0, len(ids), batch_size):
        selected = ids[start:start + batch_size]
        outputs = generate_batch(model, tokenizer, [train[i] for i in selected], cap, prompt_mode=mode)
        rows.extend({"prompt_id": i, **row} for i, (_, row) in zip(selected, outputs, strict=True))
    return {"correct": sum(row["reward"] for row in rows), "tagged": sum(row["has_answer_tag"] for row in rows),
            "truncated": sum(row["truncated"] for row in rows),
            "median_completion_tokens": median(len(row["token_ids"]) for row in rows), "rows": rows}


def supervised_loss(model, ids, answer_length):
    logits = suffix_logits(model, ids[:, :-1], answer_length).astype(mx.float32)
    return nn.losses.cross_entropy(logits, ids[:, -answer_length:], reduction="mean")


def train_step(model, tokenizer, example, loss_and_grad, optimizer):
    rationale = re.sub(r"<<[^>]+>>", "", example["rationale"])
    prompt = tokenizer.encode(prompt_text(tokenizer, example, "reasoning"), add_special_tokens=False)
    answer = tokenizer.encode(f"{rationale}\n<answer>{example['value']}</answer>", add_special_tokens=False)
    answer.append(tokenizer.eos_token_id)
    loss, grads = loss_and_grad(model, mx.array([prompt + answer]), len(answer))
    grads, norm = optim.clip_grad_norm(grads, 1.0)
    mx.eval(loss, norm)
    if not isfinite(loss.item()) or not isfinite(norm.item()):
        raise RuntimeError("Nonfinite supervised loss or gradient")
    optimizer.update(model, grads)
    mx.eval(model.trainable_parameters(), optimizer.state)
    return {"loss": loss.item(), "gradient_norm": norm.item(), "supervised_tokens": len(answer)}


def memory_mib():
    return {name: round(size / 1024**2, 1) for name, size in
            (("active", mx.get_active_memory()), ("cache", mx.get_cache_memory()), ("peak", mx.get_peak_memory()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, help="local copy of the pinned BF16 model")
    parser.add_argument("--seed", type=int, default=41501)
    parser.add_argument("--development-prompts", type=int, default=24)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[16, 40, 80])
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--eval-batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--lora-last-layers", type=int, default=4)
    parser.add_argument("--min-gain", type=int, default=3)
    parser.add_argument("--conditions", choices=("all", "reasoning"), default="all")
    args = parser.parse_args()
    train = read_split(args.train, TRAIN_SHA256)
    checkpoints = sorted(set(args.checkpoints))
    if (min(checkpoints + [args.development_prompts, args.max_new_tokens, args.eval_batch,
                           args.lora_last_layers, args.min_gain, args.lr]) <= 0
            or args.development_prompts + checkpoints[-1] > len(train)):
        parser.error("Invalid development or training budgets")
    ids = Random(args.seed).sample(range(len(train)), args.development_prompts + checkpoints[-1])
    development = ids[:args.development_prompts]
    training = ids[args.development_prompts:]
    if {train[i]["question"] for i in development} & {train[i]["question"] for i in training}:
        raise RuntimeError("Training and development questions overlap")
    args.out.mkdir(parents=True)
    model, tokenizer = load_mlx_model(args.model_path, args.lora_last_layers, args.seed)
    depth = len(model.layers)
    layers = list(range(depth - args.lora_last_layers, depth))
    old = adapter_snapshot(model)
    save_file(old, args.out / "old-adapter.safetensors")
    protocol = {"model": MODEL, "revision": REVISION, "train_sha256": TRAIN_SHA256, "seed": args.seed,
                "training_prompt_ids": training, "development_prompt_ids": development,
                "lora_rank": RANK, "lora_targets": TARGETS, "lora_layers": layers, "lora_alpha": ALPHA,
                "prompt_mode": "reasoning", "max_new_tokens": args.max_new_tokens, "lr": args.lr,
                "dtype": "bfloat16", "device": "mlx", "backend": "mlx", "checkpoints": checkpoints,
                "minimum_correct_gain": args.min_gain, "eval_batch": args.eval_batch,
                "conditions": args.conditions, "mlx": version("mlx"), "mlx_lm": version("mlx-lm"),
                "mlx_cache_limit_bytes": CACHE_LIMIT_BYTES,
                "old_adapter_sha256": adapter_digest(old)}
    write_json(args.out / "protocol.json", protocol)
    report = {"conditions": {}, "training": [], "checkpoints": {}}
    conditions = [("reasoning", args.max_new_tokens)]
    if args.conditions == "all":
        conditions = list(dict.fromkeys([("original", 128), ("original", args.max_new_tokens)] + conditions))
    for mode, cap in conditions:
        key = f"{mode}-{cap}"
        result = evaluate(model, tokenizer, train, development, cap, mode, args.eval_batch)
        result["mlx_memory_mib"] = memory_mib()
        report["conditions"][key] = result
        write_json(args.out / "development.json", report)
        print(key, result["correct"], result["tagged"], result["truncated"], result["mlx_memory_mib"], flush=True)
    mx.clear_cache()
    before = report["conditions"][f"reasoning-{args.max_new_tokens}"]
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01, bias_correction=True)
    loss_and_grad = nn.value_and_grad(model, supervised_loss)
    qualification = {"qualified": False, "development_count": len(development), "before_correct": before["correct"],
                     "old_adapter_sha256": adapter_digest(old)}
    for step, index in enumerate(training, 1):
        metrics = train_step(model, tokenizer, train[index], loss_and_grad, optimizer)
        report["training"].append({"step": step, "prompt_id": index, **metrics})
        if step % 4 == 0:
            print(f"SFT step {step} loss {metrics['loss']:.4f} {memory_mib()}", flush=True)
        if step in checkpoints:
            mx.clear_cache()
            result = evaluate(model, tokenizer, train, development, args.max_new_tokens, "reasoning", args.eval_batch)
            result["mlx_memory_mib"] = memory_mib()
            report["checkpoints"][step] = result
            write_json(args.out / "development.json", report)
            print(f"SFT dev {step}: {result['correct']}/{len(development)}", flush=True)
            mx.clear_cache()
            qualification.update(after_correct=result["correct"], selected_step=step)
            if result["correct"] - before["correct"] >= args.min_gain:
                fresh = adapter_snapshot(model)
                save_file(fresh, args.out / "fresh-adapter.safetensors")
                qualification.update(qualified=True, fresh_adapter_sha256=adapter_digest(fresh))
                break
    write_json(args.out / "dev-qualified.json", qualification)


if __name__ == "__main__":
    main()
