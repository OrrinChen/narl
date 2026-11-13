"""Two-digit addition and CPU-only initialization for the online GRPO comparison."""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations_with_replacement
import json
from pathlib import Path
from random import Random

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from algos import grpo
from core import Batch
from experiments.kv_online import (
    FIRST_CONTENT_ID, GROUP_SIZE, PROMPT_TOKENS, _generate, _prompt, _scoring, _threads, _trajectories,
)
from tasks import LLMTask
from train import Config, make_policy, optimize


LABELS = tuple("0123456789")
BUCKETS = ("noncarry", "carry")
COMPLETION_LENGTH = 3
TRAIN_EXAMPLES = 3_200
HELDOUT_EXAMPLES = 128
PILOT_SEEDS = (130, 131)
SPLIT_SEED = 2041
VOCABULARY = ("[PAD]", "[EOS]", "[UNK]", *LABELS, "+", "=", "add")


def _bucket(left, right):
    return "carry" if left % 10 + right % 10 >= 10 else "noncarry"


def _pair_splits(n_examples, n_eval):
    if n_examples < 2 or n_eval < 2 or n_examples % 2 or n_eval % 2:
        raise ValueError("split sizes must be positive and even")
    sets = {name: [] for name in ("train", "development", "heldout")}
    pairs = list(combinations_with_replacement(range(10, 100), 2))
    for index, bucket in enumerate(BUCKETS):
        selected = [pair for pair in pairs if _bucket(*pair) == bucket]
        Random(SPLIT_SEED + index).shuffle(selected)
        if len(selected) < n_examples // 2 + n_eval:
            raise ValueError("not enough disjoint operand pairs to balance the requested splits")
        sets["development"].extend(selected[:n_eval // 2])
        sets["heldout"].extend(selected[n_eval // 2:n_eval])
        sets["train"].extend(selected[n_eval:n_eval + n_examples // 2])
    return sets


def _row(left, right):
    return {
        "prompt": f"add {left % 10} {right % 10} + {left // 10} {right // 10} =",
        "answer_tokens": list(reversed(f"{left + right:03d}")),
        "completion_length": COMPLETION_LENGTH,
        "bucket": _bucket(left, right), "left": left, "right": right,
    }


class ArithmeticTask(LLMTask):
    """Zero-padded decimal addition with complete unordered operand pairs held out."""

    eval_split = "heldout"
    reward_kind = "exact_match"

    def __init__(self, n_examples=TRAIN_EXAMPLES, n_eval=HELDOUT_EXAMPLES, seed=0):
        pairs = _pair_splits(n_examples, n_eval)
        self._split([_row(*pair) for pair in pairs[self.eval_split] + pairs["train"]], n_eval, seed)

    def build_prompt(self, problem):
        return problem["prompt"]


class DevelopmentArithmeticTask(ArithmeticTask):
    eval_split = "development"


def save_fixture(path):
    tokenizer = Tokenizer(WordLevel({word: i for i, word in enumerate(VOCABULARY)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]",
        model_input_names=["input_ids", "attention_mask"],
    ).save_pretrained(path)
    config = LlamaConfig(
        vocab_size=len(VOCABULARY), hidden_size=64, intermediate_size=128,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, pad_token_id=0, eos_token_id=1,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    model.save_pretrained(path)
    return sum(parameter.numel() for parameter in model.parameters())


@torch.no_grad()
def evaluate(policy, task):
    """Free greedy decoding and exact-match accuracy on the fixed balanced split."""
    matches = {bucket: [] for bucket in (*BUCKETS, "all")}
    end = FIRST_CONTENT_ID + len(LABELS)
    for offset in range(0, len(task.eval_data), 32):
        rows = task.eval_data[offset:offset + 32]
        prompts = torch.cat([_prompt(policy, task, row) for row in rows])
        generated = policy.model.generate(
            input_ids=prompts, attention_mask=torch.ones_like(prompts),
            do_sample=False, max_new_tokens=COMPLETION_LENGTH,
            suppress_tokens=[*range(FIRST_CONTENT_ID), *range(end, policy.model.config.vocab_size)],
        )[:, -COMPLETION_LENGTH:]
        answers = torch.tensor([policy.tok.convert_tokens_to_ids(row["answer_tokens"]) for row in rows])
        exact = generated.eq(answers).all(dim=1).tolist()
        for row, match in zip(rows, exact):
            matches[row["bucket"]].append(match)
            matches["all"].append(match)
    return {bucket: sum(values) / len(values) for bucket, values in matches.items()}


def _sft(policy, task, out, steps, batch_size, eval_every, lr):
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    curve = [{"step": 0, **evaluate(policy, task)}]
    checkpoints = {}
    for step in range(1, steps + 1):
        rows = task.sample(batch_size)
        prompts = torch.cat([_prompt(policy, task, row) for row in rows])
        answers = torch.tensor([policy.tok.convert_tokens_to_ids(row["answer_tokens"]) for row in rows])
        # Every row has the same leading filler. Keep its graph and sum gradients
        # from the individually scored arithmetic suffixes into that shared graph.
        prompt_body_length = len(policy.tok.encode(rows[0]["prompt"], add_special_tokens=False))
        prefix_length = PROMPT_TOKENS - prompt_body_length
        prefix = policy.model(prompts[:1, :prefix_length], use_cache=True, logits_to_keep=1)
        prefix.past_key_values.batch_repeat_interleave(len(rows))
        suffix = torch.cat((prompts[:, prefix_length:], answers[:, :-1]), dim=1)
        outputs = policy.model(suffix, past_key_values=prefix.past_key_values,
                               use_cache=True, logits_to_keep=COMPLETION_LENGTH)
        logits = outputs.logits[:, :, FIRST_CONTENT_ID:FIRST_CONTENT_ID + len(LABELS)]
        loss = F.cross_entropy(logits.flatten(0, 1), (answers - FIRST_CONTENT_ID).flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % eval_every == 0 or step == steps:
            result = {"step": step, "loss": float(loss.detach()), **evaluate(policy, task)}
            curve.append(result)
            print(json.dumps({"stage": "sft", **result}), flush=True)
            (out / "sft-curve.json").write_text(json.dumps(curve, indent=2) + "\n")
            if 0.15 <= result["all"] < 0.8:
                checkpoint = out / f"sft-{step}"
                policy.model.generation_config.top_k = None
                policy.model.save_pretrained(checkpoint)
                policy.tok.save_pretrained(checkpoint)
                checkpoints[step] = str(checkpoint)
                (out / "sft-checkpoints.json").write_text(json.dumps(checkpoints, indent=2) + "\n")
    return checkpoints


class _SameWeights:
    def pull(self, _model, _have):
        return 0


def _grpo_pilot(config, seed, groups, eval_every, out):
    task = DevelopmentArithmeticTask(config.n_examples, config.eval_n, seed)
    policy = make_policy(config, None)
    _scoring(policy, share_learner=True, action_count=len(LABELS))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr)
    weights = _SameWeights()
    curve = [{"groups": 0, **evaluate(policy, task)}]
    useful = {bucket: 0 for bucket in BUCKETS}
    for offset in range(0, groups, config.n_prompts):
        samples = []
        for index, problem in enumerate(task.sample(config.n_prompts), start=offset):
            generator = torch.Generator().manual_seed(seed * 100_000 + index)
            payload = _generate(policy, task, problem, COMPLETION_LENGTH, weights, "share_prompt",
                                generator, share_initial=True, action_count=len(LABELS))
            samples.append(_trajectories(payload))
            useful[problem["bucket"]] += len(set(payload["rewards"])) > 1
        batch = Batch.from_groups(samples)
        optimize(policy, optimizer, batch, grpo(batch, policy), config)
        seen = offset + config.n_prompts
        if seen % eval_every == 0 or seen == groups:
            result = {"groups": seen, **evaluate(policy, task)}
            curve.append(result)
            print(json.dumps({"stage": "grpo", "seed": seed, **result}), flush=True)
            (out / f"grpo-{seed}.json").write_text(json.dumps({"curve": curve, "useful_groups": useful}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sft-steps", type=int, default=600)
    parser.add_argument("--sft-eval-every", type=int, default=30)
    parser.add_argument("--sft-lr", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--grpo-groups", type=int, default=0)
    parser.add_argument("--grpo-lr", type=float, default=0.0003)
    parser.add_argument("--eval-every", type=int, default=96)
    args = parser.parse_args()
    if args.model is None and (args.sft_steps < 1 or args.sft_eval_every < 1 or args.batch_size < 1):
        parser.error("SFT budgets must be positive")
    if args.grpo_groups and (args.model is None or args.grpo_groups % 2 or args.eval_every < 1):
        parser.error("GRPO pilot requires --model and an even group budget")
    _threads()
    out = args.out.resolve()
    out.mkdir(parents=True)
    fixture = args.model.resolve() if args.model else out / "initial"
    parameters = save_fixture(fixture) if args.model is None else None
    config = Config(task="arithmetic", algo="grpo", model=str(fixture), dtype="float32",
                    group_size=GROUP_SIZE, n_prompts=2, micro_batch=GROUP_SIZE,
                    lr=args.grpo_lr, n_examples=TRAIN_EXAMPLES, eval_n=HELDOUT_EXAMPLES, device="cpu")
    task = DevelopmentArithmeticTask(config.n_examples, config.eval_n, SPLIT_SEED)
    (out / "train.json").write_text(json.dumps(task.data) + "\n")
    (out / "development.json").write_text(json.dumps(task.eval_data) + "\n")
    protocol = {"args": vars(args), "config": vars(config), "parameters": parameters,
                "model_sha256": hashlib.sha256((fixture / "model.safetensors").read_bytes()).hexdigest(),
                "development_only": True, "split_seed": SPLIT_SEED, "grpo_seeds": PILOT_SEEDS,
                "grpo_reward_kind": task.reward_kind}
    (out / "protocol.json").write_text(json.dumps(protocol, default=str, indent=2) + "\n")
    if args.grpo_groups:
        for seed in PILOT_SEEDS:
            _grpo_pilot(config, seed, args.grpo_groups, args.eval_every, out)
    else:
        policy = make_policy(config, None)
        policy.model.generation_config.top_k = None
        checkpoints = _sft(policy, task, out, args.sft_steps, args.batch_size, args.sft_eval_every, args.sft_lr)
        (out / "sft-checkpoints.json").write_text(json.dumps(checkpoints, indent=2) + "\n")


if __name__ == "__main__":
    main()
