"""Batched MLX GSM8K decoding with sampled-token behavior logprobs."""

import mlx.core as mx
from mlx_lm.generate import BatchGenerator

from experiments.gsm8k_signal import prompt_text
from tasks import extract_answer, gsm8k_reward


class FP32SamplingBatch(BatchGenerator):
    def _step(self, input_tokens, prompt_cache):
        logits = self.model(input_tokens, cache=prompt_cache)[:, -1].astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tokens = self.sampler(logprobs)
        # Keep only the sampled-token logprob for each response.
        selected = mx.take_along_axis(logprobs, tokens[:, None], axis=-1).squeeze(-1)
        return tokens, list(selected)


class LastTokenLogits:
    def __init__(self, model):
        self.model = model
        self.layers = model.layers

    def __call__(self, tokens, cache):
        hidden = self.model.model(tokens, cache=cache)[:, -1:]
        return self.model.model.embed_tokens.as_linear(hidden)


def generate_batch(model, tokenizer, examples, max_new_tokens, *, sample=False, capture=False,
                   prompt_mode="original"):
    if not examples:
        return []
    prompts = [prompt_text(tokenizer, example, prompt_mode) for example in examples]
    prompt_ids = tokenizer(prompts, add_special_tokens=False).input_ids
    stop_ids = getattr(tokenizer, "gsm8k_stop_ids", [tokenizer.eos_token_id])
    sampler = mx.random.categorical if sample else None
    generator = FP32SamplingBatch(LastTokenLogits(model), stop_tokens=set(stop_ids), sampler=sampler,
                                  completion_batch_size=len(examples), prefill_batch_size=len(examples))
    tokens, scores = [[] for _ in examples], [[] for _ in examples]
    truncated = [None] * len(examples)
    try:
        uids = generator.insert(prompt_ids, max_new_tokens)
        positions = {uid: index for index, uid in enumerate(uids)}
        while responses := generator.next():
            if capture:
                mx.eval(*(response.logprobs for response in responses))
            for response in responses:
                index = positions[response.uid]
                tokens[index].append(response.token)
                if capture:
                    scores[index].append(response.logprobs.item())
                if response.finish_reason:
                    truncated[index] = response.finish_reason == "length"
    finally:
        generator.close()
    if any(value is None for value in truncated):
        raise RuntimeError("MLX generation ended before every completion finished")
    result = []
    for index, ids in enumerate(tokens):
        text = tokenizer.decode(ids, skip_special_tokens=True)
        row = {"completion": text, "token_ids": ids, "truncated": truncated[index],
               "has_answer_tag": extract_answer(text) is not None,
               "reward": int(gsm8k_reward(prompts[index], text, examples[index]))}
        if capture:
            row["behavior_logprobs"] = scores[index]
        result.append((prompt_ids[index], row))
    return result
