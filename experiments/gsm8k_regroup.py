"""Fixed-rollout comparison of GSM8K version-homogeneous and mixed GRPO."""

import argparse
import hashlib
import json
from math import isfinite
from pathlib import Path
from random import Random
import re
from time import perf_counter

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from algos import grpo
from core import Batch, Trajectory
from experiments.gsm8k_signal import (
    MODEL, REVISION, TEST_SHA256, TRAIN_SHA256, generate_group, load_model, prompt_text, read_split,
)
from experiments.regroup import group_orders, write_json, write_record
from train import Config, optimize


def adapter_snapshot(model):
    return {key: value.detach().to("cpu", copy=True)
            for key, value in get_peft_model_state_dict(model, save_embedding_layers=False).items()}


def adapter_digest(snapshot):
    digest = hashlib.sha256()
    for key, tensor in sorted(snapshot.items()):
        digest.update(key.encode())
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class Learner:
    def __init__(self, model):
        self.model = model

    def parameters(self):
        return (parameter for parameter in self.model.parameters() if parameter.requires_grad)

    def logprobs(self, batch):
        # Microbatches contain one completion. Keep its full history and omit padding.
        active = batch.mask[0].nonzero().flatten()
        first, end = int(active[0]), int(active[-1]) + 1
        suffix = batch.states[:, first:end]
        logits = self.model(batch.states[:, :end - 1], use_cache=False, logits_to_keep=suffix.size(1)).logits
        scores = -F.cross_entropy(logits[0].float(), suffix[0], reduction="none").unsqueeze(0)
        drift = scores.detach() - batch.old_logp[:, first:end]
        if not torch.isfinite(drift).all() or drift.max() >= 80:
            raise RuntimeError(f"Invalid behavior/current logprob difference: [{float(drift.min())}, {float(drift.max())}]")
        return F.pad(scores, (first, batch.states.size(1) - end))


def train_snapshot(model, tokenizer, train, indices):
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.0005)
    losses = []
    for index in indices:
        example = train[index]
        # The public GSM8K rationale includes calculator annotations.
        rationale = re.sub(r"<<[^>]+>>", "", example["rationale"])
        prompt = tokenizer.encode(prompt_text(tokenizer, example), add_special_tokens=False)
        answer = tokenizer.encode(f"{rationale}\n<answer>{example['value']}</answer>", add_special_tokens=False)
        answer.append(tokenizer.eos_token_id)
        ids = torch.tensor([prompt + answer], device=model.device)
        logits = model(ids[:, :-1], use_cache=False, logits_to_keep=len(answer)).logits
        loss = F.cross_entropy(logits[0].float(), ids[0, -len(answer):])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    optimizer.zero_grad(set_to_none=True)
    return losses


def collect(model, tokenizer, train, ids, group_size, max_tokens, seed, cohort):
    for prompt_id in ids:
        torch.manual_seed(seed * 100_000 + prompt_id)
        prompt, samples = generate_group(model, tokenizer, train[prompt_id], group_size, max_tokens, capture=True)
        for index, sample in enumerate(samples):
            sample["id"] = f"{prompt_id}:{cohort}:{index}"
        yield prompt_id, prompt, samples


def evaluate(model, tokenizer, test, ids, max_tokens):
    evaluated = []
    for prompt_id in ids:
        _, rows = generate_group(model, tokenizer, test[prompt_id], 1, max_tokens, sample=False)
        evaluated.append({"prompt_id": prompt_id, **rows[0]})
    return evaluated


def trajectory(prompt_ids, sample):
    ids = torch.tensor(prompt_ids + sample["token_ids"])
    logp = F.pad(torch.tensor(sample["behavior_logprobs"]), (len(prompt_ids), 0))
    mask = torch.ones_like(logp)
    mask[:len(prompt_ids)] = 0
    rewards = torch.zeros_like(logp)
    rewards[-1] = sample["reward"]
    return Trajectory(ids, ids, logp, rewards, mask, sample["truncated"])


@torch.no_grad()
def move_model(model, device, dtype):
    model.to(device)
    for parameter in model.parameters():
        if not parameter.requires_grad:
            parameter.data = parameter.data.to(dtype=getattr(torch, dtype))
    return model


def replay_arm(learner, tokenizer, snapshot, records, test, test_ids, args, cohorts, mixed):
    model = move_model(learner.model, args.learner_device, args.learner_dtype)
    set_peft_model_state_dict(model, snapshot)
    model.get_base_model().set_attn_implementation("eager")
    optimizer = torch.optim.AdamW(learner.parameters(), lr=args.lr)
    cfg = Config(model=MODEL, task="gsm8k", algo="grpo", group_size=args.group_size,
                 micro_batch=1, lr=args.lr, max_new_tokens=args.max_new_tokens, device=args.learner_device)
    updates, assignments = [], []
    consumed_tokens = 0
    for record in records:
        pool = [sample for name in cohorts for sample in record["cohorts"][name]]
        if mixed:
            ordering = group_orders(1, args.group_size, args.seed + record["prompt_id"]).tolist()
        else:
            ordering = [list(range(args.group_size)), list(range(args.group_size, 2 * args.group_size))]
        groups = [[pool[index] for index in group] for group in ordering]
        assignments.append({"prompt_id": record["prompt_id"], "groups": [[r["id"] for r in g] for g in groups]})
        trajectories = [trajectory(record["prompt_ids"], sample) for sample in pool]
        batch = Batch.from_groups([trajectories[:args.group_size], trajectories[args.group_size:]]).to(args.learner_device)
        grouped = Batch.from_groups([[trajectories[index] for index in group] for group in ordering])
        inverse = torch.argsort(torch.tensor(ordering).flatten())
        advantage = grpo(grouped, learner)[inverse].to(args.learner_device)
        metrics = optimize(learner, optimizer, batch, advantage, cfg)
        if not all(isfinite(value) for value in metrics.values()):
            raise RuntimeError(f"Invalid update at prompt {record['prompt_id']}: {metrics}")
        updates.append({"prompt_id": record["prompt_id"], "nonzero_advantage_tokens": int(advantage.ne(0).sum()),
                        **metrics})
        print(f"  prompt {record['prompt_id']}: gradient norm {metrics['gnorm']:.5f}", flush=True)
        consumed_tokens += int(batch.mask.sum())
    optimizer.zero_grad(set_to_none=True)
    move_model(model, args.device, args.dtype)
    model.get_base_model().set_attn_implementation("sdpa")
    evaluated = evaluate(model, tokenizer, test, test_ids, args.max_new_tokens)
    return {"assignments": assignments, "updates": updates, "consumed_tokens": consumed_tokens,
            "evaluated": evaluated, "heldout_correct": sum(row["reward"] for row in evaluated)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--replay-prompts", type=int, default=8)
    parser.add_argument("--replay-prompt-ids", type=int, nargs="+")
    parser.add_argument("--sft-prompts", type=int, default=12)
    parser.add_argument("--heldout-prompts", type=int, default=12)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=8401)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--learner-device", choices=("cpu", "mps"), help="defaults to --device")
    parser.add_argument("--learner-dtype", choices=("float32", "bfloat16"))
    parser.add_argument("--lora-last-layers", type=int, default=0, help="0 adapts all layers")
    parser.add_argument("--replay-from", type=Path, help="reuse saved rollouts and their exact adapters")
    args = parser.parse_args()
    args.learner_device = args.learner_device or args.device
    args.learner_dtype = args.learner_dtype or ("float32" if args.learner_device == "cpu" else args.dtype)
    train = read_split(args.train, TRAIN_SHA256)
    test = read_split(args.test, TEST_SHA256)
    replay_count = len(args.replay_prompt_ids) if args.replay_prompt_ids else args.replay_prompts
    if (min(replay_count, args.sft_prompts, args.heldout_prompts, args.max_new_tokens, args.threads) < 1
            or args.group_size < 2 or args.group_size % 2 or args.lr <= 0
            or replay_count + args.sft_prompts > len(train) or args.heldout_prompts > len(test)
            or args.lora_last_layers < 0):
        parser.error("invalid budgets or group size (must be positive and even)")
    if "mps" in (args.device, args.learner_device) and not torch.backends.mps.is_available():
        parser.error("PyTorch cannot see an MPS device")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    rng = Random(args.seed)
    if args.replay_prompt_ids:
        replay_ids = args.replay_prompt_ids
        excluded = set(replay_ids)
        if len(excluded) != replay_count or any(not 0 <= index < len(train) for index in replay_ids):
            parser.error("replay prompt IDs must be unique training indices")
        sft_ids = rng.sample([index for index in range(len(train)) if index not in excluded], args.sft_prompts)
    else:
        train_ids = rng.sample(range(len(train)), replay_count + args.sft_prompts)
        replay_ids, sft_ids = train_ids[:replay_count], train_ids[replay_count:]
    test_ids = rng.sample(range(len(test)), args.heldout_prompts)
    if {train[i]["question"] for i in replay_ids + sft_ids} & {test[i]["question"] for i in test_ids}:
        raise RuntimeError("training and test samples overlap")
    args.out.mkdir(parents=True)
    model, tokenizer = load_model(getattr(torch, args.dtype))
    depth = model.config.num_hidden_layers
    if args.lora_last_layers > depth:
        parser.error(f"--lora-last-layers exceeds model depth {depth}")
    layers = list(range(depth - args.lora_last_layers, depth)) if args.lora_last_layers else None
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0,
                                            target_modules=["q_proj", "v_proj"], layers_to_transform=layers,
                                            task_type="CAUSAL_LM")).to(args.device).eval()
    learner = Learner(model)
    if args.replay_from:
        source = json.loads((args.replay_from / "protocol.json").read_text())
        initial = load_file(args.replay_from / "old-adapter.safetensors")
        fresh = load_file(args.replay_from / "fresh-adapter.safetensors")
        losses = source["snapshot_sft_losses"]
    else:
        initial = adapter_snapshot(model)
        print("Training fresh snapshot", flush=True)
        model.get_base_model().set_attn_implementation("eager")
        losses = train_snapshot(model, tokenizer, train, sft_ids)
        model.get_base_model().set_attn_implementation("sdpa")
        fresh = adapter_snapshot(model)
    if adapter_digest(initial) == adapter_digest(fresh):
        raise RuntimeError("snapshot training produced no parameter change")
    protocol = {"train_sha256": TRAIN_SHA256, "test_sha256": TEST_SHA256, "model": MODEL, "revision": REVISION,
                "seed": args.seed, "group_size": args.group_size, "max_new_tokens": args.max_new_tokens,
                "replay_selection": "explicit" if args.replay_prompt_ids else "random",
                "replay_prompt_ids": replay_ids, "snapshot_training_prompt_ids": sft_ids,
                "heldout_prompt_ids": test_ids, "old_adapter_sha256": adapter_digest(initial),
                "fresh_adapter_sha256": adapter_digest(fresh), "snapshot_sft_losses": losses,
                "device": args.device, "dtype": args.dtype, "lora_rank": 4, "lora_targets": ["q_proj", "v_proj"],
                "lora_layers": layers,
                "training_attention": "eager", "generation_attention": "sdpa",
                "lr": args.lr, "threads": args.threads, "torch": torch.__version__,
                "sampling_seeds": {cohort: args.seed + offset for offset, cohort in enumerate(("old", "fresh_a", "fresh_b"))},
                "budget": "one optimizer step per replay prompt; two groups of matched completions"}
    if args.replay_from and any(source.get(key) != value for key, value in protocol.items()):
        raise ValueError("Saved rollout pool and adapter hashes must match the current protocol")
    protocol["learner_device"] = args.learner_device
    protocol["learner_dtype"] = args.learner_dtype
    write_json(args.out / "protocol.json", protocol)
    save_file(initial, args.out / "old-adapter.safetensors")
    save_file(fresh, args.out / "fresh-adapter.safetensors")
    started = perf_counter()
    if args.replay_from:
        records = [json.loads(line) for line in (args.replay_from / "samples.jsonl").read_text().splitlines()]
    else:
        # Load each snapshot once, preserving per-prompt seeds across execution.
        by_id = {prompt_id: {"prompt_id": prompt_id, "cohorts": {}} for prompt_id in replay_ids}
        for cohort, seed in protocol["sampling_seeds"].items():
            set_peft_model_state_dict(model, initial if cohort == "old" else fresh)
            for prompt_id, prompt, samples in collect(model, tokenizer, train, replay_ids, args.group_size,
                                                       args.max_new_tokens, seed, cohort):
                record = by_id[prompt_id]
                if record.setdefault("prompt_ids", prompt) != prompt:
                    raise RuntimeError("prompt encoding changed across cohorts")
                record["cohorts"][cohort] = samples
                print(cohort, prompt_id, [row["reward"] for row in samples], flush=True)
        records = list(by_id.values())
    with (args.out / "samples.jsonl").open("w") as handle:
        for record in records:
            write_record(handle, record)
    summary = {"arms": {}}
    for names, cohorts in ((("homogeneous", "mixed"), ("old", "fresh_a")),
                           (("same_version_homogeneous", "same_version_mixed"), ("fresh_a", "fresh_b"))):
        for name, mixed in zip(names, (False, True)):
            print(f"Replaying {name}", flush=True)
            result = replay_arm(learner, tokenizer, fresh, records, test, test_ids, args, cohorts, mixed)
            summary["arms"][name] = result
            summary["elapsed_s"] = perf_counter() - started
            write_json(args.out / "summary.json", summary)
            print(name, result["heldout_correct"], result["consumed_tokens"], flush=True)
    set_peft_model_state_dict(model, fresh)
    summary["initial_eval"] = evaluate(model, tokenizer, test, test_ids, args.max_new_tokens)
    summary["elapsed_s"] = perf_counter() - started
    write_json(args.out / "summary.json", summary)


if __name__ == "__main__":
    main()
