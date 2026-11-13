"""Offline CPU checks for teacher-forced KV replay on a tiny causal LM."""

import copy
import math
from pathlib import Path
import sys

import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.kv_replay import ATOL, RTOL, _compare, prefill, replay, valid_prediction_mask


def _model():
    config = LlamaConfig(
        vocab_size=24, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
        pad_token_id=0, eos_token_id=1, attn_implementation="eager",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        return LlamaForCausalLM(config).eval()


def _inputs():
    prompt = torch.tensor([[2, 5, 4, 11]])
    suffix = torch.tensor([[7, 9, 10], [12, 13, 14]])
    continuation = torch.tensor([[15, 16, 17], [18, 19, 20]])
    return prompt, suffix, continuation


@torch.inference_mode()
def _full_logits(model, prompt, suffix, continuation, prompt_mask):
    group_size = suffix.shape[0]
    history = torch.cat((prompt.expand(group_size, -1), suffix, continuation), dim=1)
    attention_mask = torch.cat((prompt_mask.expand(group_size, -1),
                                torch.ones_like(suffix), torch.ones_like(continuation)), dim=1)
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp_min(0)
    result = model(input_ids=history, attention_mask=attention_mask,
                   position_ids=position_ids, use_cache=False)
    return result.logits[:, -continuation.shape[1]:]


def _assert_replay(logits, expected, continuation):
    torch.testing.assert_close(logits, expected, atol=ATOL, rtol=RTOL)
    assert torch.equal(logits.argmax(dim=-1), expected.argmax(dim=-1))
    # The first replay logit follows the first forced token, so it scores token 1.
    next_tokens = continuation[:, 1:].unsqueeze(-1)
    actual_logp = logits[:, :-1].log_softmax(dim=-1).gather(-1, next_tokens)
    expected_logp = expected[:, :-1].log_softmax(dim=-1).gather(-1, next_tokens)
    torch.testing.assert_close(actual_logp, expected_logp, atol=ATOL, rtol=RTOL)


def _assert_metrics(metrics, prompt_tokens, suffix_tokens):
    assert metrics["prompt_tokens_recomputed"] == prompt_tokens
    assert metrics["suffix_tokens_recomputed"] == suffix_tokens
    for key in ("cache_copy_ms", "recompute_ms", "first_token_ms"):
        assert math.isfinite(metrics[key]) and metrics[key] >= 0, key


def test_same_weights_retained_cache_matches_full_forward():
    model = _model()
    prompt, suffix, continuation = _inputs()
    history = torch.cat((prompt.expand(suffix.shape[0], -1), suffix), dim=1)
    cache = prefill(model, history).past_key_values
    expected = _full_logits(model, prompt, suffix, continuation, torch.ones_like(prompt))

    actual, metrics = replay(model, "retain", prompt, suffix, continuation, old_cache=cache)

    _assert_replay(actual, expected, continuation)
    _assert_metrics(metrics, prompt_tokens=0, suffix_tokens=0)


def test_fresh_replays_match_full_forward_with_left_padding_and_position_offset():
    old_model = _model()
    new_model = copy.deepcopy(old_model)
    with torch.no_grad():
        new_model.model.layers[0].self_attn.v_proj.weight.neg_()
    prompt, suffix, continuation = _inputs()
    # Nonzero masked IDs catch a missing prompt mask without relying on pad embeddings.
    prompt = torch.cat((torch.tensor([[21, 22]]), prompt), dim=1)
    prompt_mask = torch.tensor([[0, 0, 1, 1, 1, 1]])
    history = torch.cat((prompt.expand(suffix.shape[0], -1), suffix), dim=1)
    history_mask = torch.cat((prompt_mask.expand(suffix.shape[0], -1),
                              torch.ones_like(suffix)), dim=1)
    old_cache = prefill(old_model, history, history_mask).past_key_values
    same_weight_cache = prefill(new_model, history, history_mask).past_key_values
    expected = _full_logits(new_model, prompt, suffix, continuation, prompt_mask)

    independent, independent_metrics = replay(new_model, "recompute", prompt, suffix,
                                               continuation, prompt_mask=prompt_mask)
    shared, shared_metrics = replay(new_model, "share_prompt", prompt, suffix, continuation,
                                    prompt_mask=prompt_mask)
    retained, _ = replay(new_model, "retain", prompt, suffix, continuation, old_cache=old_cache,
                         prompt_mask=prompt_mask)
    same_weight_retained, _ = replay(new_model, "retain", prompt, suffix, continuation,
                                     old_cache=same_weight_cache, prompt_mask=prompt_mask)

    _assert_replay(independent, expected, continuation)
    _assert_replay(shared, expected, continuation)
    _assert_replay(same_weight_retained, expected, continuation)
    assert not torch.allclose(retained, expected, atol=ATOL, rtol=RTOL)
    active_prompt_tokens = int(prompt_mask.sum())
    _assert_metrics(independent_metrics, suffix.shape[0] * active_prompt_tokens, suffix.numel())
    _assert_metrics(shared_metrics, active_prompt_tokens, suffix.numel())


def test_shared_prompt_branches_are_isolated():
    model = _model()
    prompt, suffix, continuation = _inputs()
    original, _ = replay(model, "share_prompt", prompt, suffix, continuation)

    changed_suffix = suffix.clone()
    changed_suffix[0] = torch.tensor([20, 21, 22])
    changed, _ = replay(model, "share_prompt", prompt, changed_suffix, continuation)
    permuted, _ = replay(model, "share_prompt", prompt, suffix.flip(0), continuation.flip(0))

    assert not torch.allclose(original[0], changed[0], atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(original[1], changed[1], atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(original, permuted.flip(0), atol=ATOL, rtol=RTOL)


def test_prediction_mask_excludes_first_eos_and_all_following_tokens():
    continuation = torch.tensor([[1, 7, 1, 9], [7, 8, 1, 9], [7, 8, 9, 10]])
    mask = valid_prediction_mask(continuation, eos_id=1)
    expected = torch.tensor([[False, False, False, False],
                             [True, True, False, False],
                             [True, True, True, True]])
    assert mask.dtype == torch.bool
    assert torch.equal(mask, expected)


def test_prediction_metrics_ignore_logits_after_eos():
    continuation = torch.tensor([[1, 7, 1, 9], [7, 8, 1, 9]])
    valid = valid_prediction_mask(continuation, eos_id=1)
    reference = torch.zeros(2, 4, 12)
    candidate = reference.clone()
    candidate[:, :, 1].masked_fill_(~valid, 20.0)

    metrics = _compare(reference, candidate, continuation, valid)

    assert metrics["predictions"] == metrics["greedy_matches"] == 2
    assert metrics["max_abs_logit_error"] == 0
    assert metrics["max_abs_forced_token_logprob_error"] == 0
    assert metrics["mean_kl_by_decode_step_nats"] == [0.0, 0.0, None, None]


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
