"""Measure exact-answer reward variation on a fixed GSM8K training subset."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
from random import Random
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

from model import chat_prompt
from tasks import extract_answer, gsm8k_reward

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
TRAIN_SHA256 = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"
TEST_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"


def read_split(path, expected_hash):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"GSM8K file does not match the pinned source: {path}")
    rows = []
    for line in data.splitlines():
        example = json.loads(line)
        rationale, answer = example["answer"].rsplit("####", 1)
        rows.append({"question": example["question"], "rationale": rationale.strip(),
                     "value": answer.strip().replace(",", "")})
    return rows


def prompt_text(tokenizer, example, prompt_mode="original"):
    if prompt_mode == "reasoning":
        message = ("Solve the math problem. Show the calculation step by step before giving the answer. "
                   "On the final line, write the number as <answer>NUMBER</answer>.\n\n"
                   "Example: Maya buys three packs of four pens and gives away two. How many pens remain?\n"
                   "Three packs contain 3 * 4 = 12 pens. After giving away two, 12 - 2 = 10 remain.\n"
                   "<answer>10</answer>\n\nProblem: " + example["question"])
    else:
        message = (example["question"] + "\nReason briefly, then put ONLY the final number "
                   "inside <answer> </answer> tags.")
    return chat_prompt(tokenizer, message, None, False)


def load_model(dtype):
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                                dtype=dtype, attn_implementation="sdpa").eval()
    return model, tokenizer


@torch.no_grad()
def generate_batch(model, tokenizer, examples, max_new_tokens, *, sample=False, capture=False,
                   prompt_mode="original"):
    prompts = [prompt_text(tokenizer, example, prompt_mode) for example in examples]
    encoded = tokenizer(prompts, add_special_tokens=False, padding=True, padding_side="left", return_tensors="pt")
    input_ids = encoded.input_ids.tolist()
    masks = encoded.attention_mask.tolist()
    encoded = encoded.to(model.device)
    prefix_len = encoded.input_ids.size(1)
    sampling = ({"temperature": 1.0, "top_p": 1.0, "top_k": 0} if sample
                else {"temperature": None, "top_p": None, "top_k": None})
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens, do_sample=sample, repetition_penalty=1.0,
        pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=True, output_scores=capture,
        **sampling,
    )
    stop_ids = model.generation_config.eos_token_id
    stop_ids = set(stop_ids if isinstance(stop_ids, list) else [stop_ids])
    if capture:
        actions = output.sequences[:, prefix_len:]
        selected = torch.stack([
            logits.float().log_softmax(-1).gather(1, actions[:, pos:pos + 1]).squeeze(1)
            for pos, logits in enumerate(output.scores)
        ], dim=1).tolist()
    rows = []
    for index, response in enumerate(output.sequences[:, prefix_len:].tolist()):
        stop = next((pos for pos, token in enumerate(response) if token in stop_ids), None)
        response = response[:stop + 1] if stop is not None else response
        text = tokenizer.decode(response, skip_special_tokens=True)
        row = {"completion": text, "token_ids": response, "truncated": stop is None,
               "has_answer_tag": extract_answer(text) is not None,
               "reward": int(gsm8k_reward(prompts[index], text, examples[index]))}
        if capture:
            row["behavior_logprobs"] = selected[index][:len(response)]
        prompt_ids = input_ids[index][-sum(masks[index]):]
        rows.append((prompt_ids, row))
    return rows


def generate_group(model, tokenizer, example, group_size, max_new_tokens, *, sample=True, capture=False,
                   prompt_mode="original"):
    results = generate_batch(model, tokenizer, [example] * group_size, max_new_tokens,
                             sample=sample, capture=capture, prompt_mode=prompt_mode)
    return results[0][0], [row for _, row in results]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    train = read_split(args.train, TRAIN_SHA256)
    if not 1 <= args.prompts <= len(train) or args.group_size < 2 or min(args.max_new_tokens, args.threads) < 1:
        parser.error("require valid prompt, token and thread counts, and at least two answers per prompt")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.out.mkdir(parents=True)
    model, tokenizer = load_model(getattr(torch, args.dtype))
    indices = Random(args.seed).sample(range(len(train)), args.prompts)
    counts = Counter()
    started = perf_counter()
    with (args.out / "groups.jsonl").open("w") as handle:
        for prompt_id in indices:
            torch.manual_seed(args.seed * 100_000 + prompt_id)
            prompt, rows = generate_group(model, tokenizer, train[prompt_id], args.group_size, args.max_new_tokens)
            rewards = [row["reward"] for row in rows]
            counts["mixed_groups"] += int(0 < sum(rewards) < len(rewards))
            counts["all_correct_groups"] += int(all(rewards))
            counts["all_wrong_groups"] += int(not any(rewards))
            counts["correct"] += sum(rewards)
            counts["truncated"] += sum(row["truncated"] for row in rows)
            counts["tagged"] += sum(row["has_answer_tag"] for row in rows)
            counts["completion_tokens"] += sum(len(row["token_ids"]) for row in rows)
            handle.write(json.dumps({"prompt_id": prompt_id, "prompt_tokens": len(prompt), "samples": rows}) + "\n")
            handle.flush()
            print(prompt_id, rewards, flush=True)
    summary = {"model": MODEL, "revision": REVISION, "data_sha256": TRAIN_SHA256,
               "seed": args.seed, "prompt_ids": indices, "group_size": args.group_size,
               "max_new_tokens": args.max_new_tokens, "device": "cpu", "dtype": args.dtype,
               "platform": platform.platform(), "threads": args.threads, "torch": torch.__version__,
               "transformers": transformers_version, "samples": args.prompts * args.group_size,
               "elapsed_s": perf_counter() - started, **counts}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
