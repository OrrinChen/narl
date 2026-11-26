"""Collect common frozen-policy GSM8K candidates with token-level timing for budget replay."""

import argparse
import fcntl
import json
from importlib.metadata import version
from math import isclose, isfinite
from pathlib import Path
from time import perf_counter

from experiments.audit_kv_online import require
from experiments.gsm8k_reward_study import digest, freeze, read


OUT = Path("runs/gsm8k-decision-v1")
SOURCE = Path("runs/gsm8k-budget-5seed")
TRAIN = Path("runs/gsm8k-data/train.jsonl")
PROMPT_MODE = "reasoning"


def sampling_key(protocol, seed, prompt_id, slot):
    require(slot in (0, 1) and seed in protocol["seeds"] and 0 <= prompt_id < 10_000,
            "invalid candidate coordinates")
    return protocol["rng_base"] + 2 * (seed * 10_000 + prompt_id) + slot


def load_frozen(protocol, source=SOURCE, model_path=None):
    import torch
    from safetensors.torch import load_file
    from experiments.gsm8k_mlx_development import adapter_snapshot, load_mlx_model, set_adapter
    from experiments.gsm8k_regroup import adapter_digest
    from experiments.gsm8k_signal import MODEL, REVISION, TRAIN_SHA256

    source = Path(source)
    prior = read(source / "protocol.json")
    for name, expected in (("model", MODEL), ("revision", REVISION), ("train_sha256", TRAIN_SHA256)):
        require(protocol[name] == prior[name] == expected, f"frozen {name} differs from the common start")
    require(prior["qualification_protocol"]["prompt_mode"] == PROMPT_MODE, "unexpected prompt format")
    require((prior["mlx"], prior["mlx_lm"]) == (version("mlx"), version("mlx-lm")), "MLX versions changed")
    start = load_file(source / "start-adapter.safetensors")
    require(adapter_digest(start) == prior["start_sha256"] == protocol["start_sha256"], "wrong SFT start")
    torch.set_num_threads(1)
    model, tokenizer = load_mlx_model(model_path, len(prior["qualification_protocol"]["lora_layers"]), 92333)
    set_adapter(model, start)
    require(adapter_digest(adapter_snapshot(model)) == protocol["start_sha256"], "loaded weights changed")
    return model, tokenizer


def answer_fields(text, example):
    from tasks import extract_answer, gsm8k_reward

    return {"has_answer_tag": extract_answer(text) is not None,
            "strict_reward": int(gsm8k_reward("", text, example))}


def sample_candidate(model, tokenizer, example, prompt_id, seed, slot, protocol, cap=None):
    """One independent categorical draw; resetting the key also reproduces shorter prefixes."""
    import mlx.core as mx
    from experiments.gsm8k_mlx_generate import FP32SamplingBatch, LastTokenLogits
    from experiments.gsm8k_signal import prompt_text

    cap = protocol["max_new_tokens"] if cap is None else cap
    require(isinstance(cap, int) and 1 <= cap <= protocol["max_new_tokens"], "invalid candidate token cap")
    key = sampling_key(protocol, seed, prompt_id, slot)
    started = perf_counter()
    mx.random.seed(key)
    message = prompt_text(tokenizer, example, PROMPT_MODE)
    prompt_ids = tokenizer.encode(message, add_special_tokens=False)
    generator = FP32SamplingBatch(LastTokenLogits(model), stop_tokens=tokenizer.gsm8k_stop_ids,
                                  sampler=mx.random.categorical, completion_batch_size=1, prefill_batch_size=1)
    tokens, logprobs, elapsed = [], [], []
    finish = None
    try:
        generator.insert([prompt_ids], cap)
        while replies := generator.next():
            reply, = replies
            tokens.append(reply.token)
            logprobs.append(reply.logprobs.item())
            elapsed.append(perf_counter() - started)
            finish = reply.finish_reason
    finally:
        generator.close()
    require(finish in ("stop", "length") and 0 < len(tokens) <= cap, "incomplete sampled candidate")
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    total = perf_counter() - started
    return {"seed": seed, "prompt_id": prompt_id, "slot": slot, "rng_seed": key, "token_cap": cap,
            "checkpoint_sha256": protocol["start_sha256"], "prompt_ids": prompt_ids, "token_ids": tokens,
            "behavior_logprobs": logprobs, "mean_logprob": sum(logprobs) / len(tokens),
            "truncated": finish == "length", "completion": text, "token_elapsed_s": elapsed,
            "generation_s": total, **answer_fields(text, example)}


def truncate_candidate(tokenizer, example, sample, remaining):
    """Replay a capped draw. Partial timing ends at the measured arrival of its last token."""
    require(isinstance(remaining, int) and remaining > 0, "a retry needs at least one token")
    n = len(sample["token_ids"])
    if remaining >= n:
        return sample if remaining >= sample["token_cap"] else {**sample, "token_cap": remaining}
    tokens = sample["token_ids"][:remaining]
    logprobs = sample["behavior_logprobs"][:remaining]
    elapsed = sample["token_elapsed_s"][:remaining]
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return {**sample, "token_ids": tokens, "token_cap": remaining, "behavior_logprobs": logprobs,
            "mean_logprob": sum(logprobs) / remaining, "truncated": True, "completion": text,
            "token_elapsed_s": elapsed, "generation_s": elapsed[-1], **answer_fields(text, example)}


def validate_cached(row, tokenizer, example, prompt_ids, coordinates, protocol, protocol_hash):
    seed, prompt_id, slot = coordinates
    expected = {"protocol_sha256": protocol_hash, "checkpoint_sha256": protocol["start_sha256"],
                "seed": seed, "prompt_id": prompt_id, "slot": slot, "prompt_ids": prompt_ids,
                "rng_seed": sampling_key(protocol, seed, prompt_id, slot), "token_cap": protocol["max_new_tokens"]}
    require(all(row[k] == v for k, v in expected.items()), "cached candidate binding mismatch")
    tokens, logprobs, elapsed = row["token_ids"], row["behavior_logprobs"], row["token_elapsed_s"]
    n = len(tokens)
    require(0 < n <= row["token_cap"] and n == len(logprobs) == len(elapsed), "cached candidate length mismatch")
    stop_ids = tokenizer.gsm8k_stop_ids
    require(not any(t in stop_ids for t in tokens[:-1]) and (tokens[-1] in stop_ids or n == row["token_cap"])
            and row["truncated"] == (tokens[-1] not in stop_ids), "cached candidate termination mismatch")
    require(all(isfinite(x) and x <= 1e-5 for x in logprobs) and
            isclose(row["mean_logprob"], sum(logprobs) / n, rel_tol=1e-12, abs_tol=1e-12),
            "cached candidate logprobs invalid")
    require(all(isfinite(t) for t in elapsed) and elapsed[0] > 0 and
            all(b >= a for a, b in zip(elapsed, elapsed[1:])) and
            isfinite(row["generation_s"]) and row["generation_s"] >= elapsed[-1], "invalid generation timings")
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    require(row["completion"] == text and all(row[k] == v for k, v in answer_fields(text, example).items()),
            "cached candidate decode or strict score mismatch")


def collect_seed(folder, protocol, protocol_hash, panel, seed, examples, model, tokenizer, *, sample=sample_candidate):
    from experiments.gsm8k_signal import prompt_text

    ids = protocol["eval_sets"][panel]
    require(seed in protocol["seeds"] and ids and len(ids) == len(set(ids))
            and all(0 <= i < len(examples) for i in ids), "invalid frozen selection")
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".candidate.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, prompt_id in enumerate(ids):
            example = examples[prompt_id]
            prompt_ids = tokenizer.encode(prompt_text(tokenizer, example, PROMPT_MODE), add_special_tokens=False)
            for slot in (0, 1):
                path = folder / f"{prompt_id}-{slot}.json"
                if path.is_file():
                    row = read(path)
                    validate_cached(row, tokenizer, example, prompt_ids, (seed, prompt_id, slot), protocol, protocol_hash)
                    continue
                row = sample(model, tokenizer, example, prompt_id, seed, slot, protocol)
                row["protocol_sha256"] = protocol_hash
                validate_cached(row, tokenizer, example, prompt_ids, (seed, prompt_id, slot), protocol, protocol_hash)
                freeze(path, (json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n").encode())
            if (index + 1) % 16 == 0 or index + 1 == len(ids):
                print(f"{panel} seed {seed}: {index + 1}/{len(ids)} pairs", flush=True)


def main():
    from experiments.gsm8k_signal import read_split

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument("--panel", choices=("holdout",), default="holdout")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    path = args.out / "protocol.json"
    protocol = read(path)
    protocol_hash = digest(path)
    lock = read(args.out / "head-lock.json")
    require(lock["protocol_sha256"] == protocol_hash and lock["head_sha256"] == digest(args.out / "head.json"),
            "decision head must be frozen for this protocol before holdout generation")
    seeds = protocol["seeds"] if args.seed is None else [args.seed]
    require(all(seed in protocol["seeds"] for seed in seeds), "seed outside frozen study")
    model, tokenizer = load_frozen(protocol, args.source)
    examples = read_split(args.train, protocol["train_sha256"])
    for seed in seeds:
        folder = args.out / "candidates" / args.panel / f"seed-{seed}"
        collect_seed(folder, protocol, protocol_hash, args.panel, seed, examples, model, tokenizer)


if __name__ == "__main__":
    main()
