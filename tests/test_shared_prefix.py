"""Forward and backward equivalence for the online shared-prompt learner."""
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.kv_online import FIRST_CONTENT_ID, _scoring
from experiments.shared_prefix import shared_prefix_logprobs


def _check_suffix(suffix, completion_mask, *, action_count=None):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=13, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        ))
    model.eval()
    prompt = torch.tensor([[3, 4, 5, 6, 3, 7]])
    n, prompt_tokens = len(suffix), prompt.size(1)
    batch = SimpleNamespace(
        states=torch.cat((prompt.expand(n, -1), torch.tensor(suffix)), dim=1),
        mask=torch.cat((torch.zeros(n, prompt_tokens), torch.tensor(completion_mask)), dim=1),
    )
    policy = SimpleNamespace(model=model)
    _scoring(policy, action_count=action_count)
    dense = policy.logprobs(batch)
    shared = shared_prefix_logprobs(model, batch, prompt_tokens, FIRST_CONTENT_ID,
                                    action_count=action_count)
    torch.testing.assert_close(shared, dense, atol=2e-6, rtol=1e-5)

    # Different signed token weights exercise accumulation through the shared cache.
    weights = torch.linspace(-1, 1, dense.numel()).reshape_as(dense)
    dense_grads = torch.autograd.grad((dense * weights).sum(), tuple(model.parameters()))
    shared_grads = torch.autograd.grad((shared * weights).sum(), tuple(model.parameters()))
    assert dense_grads[0][3].norm() > 0  # Prefix embeddings contribute to the loss.
    for got, expected in zip(shared_grads, dense_grads):
        torch.testing.assert_close(got, expected, atol=2e-6, rtol=1e-5)


def test_shared_prefix_scores_and_backpropagates_multiple_suffixes():
    _check_suffix([[9, 10, 11], [10, 11, 0], [12, 9, 10]],
                  [[1, 1, 1], [1, 1, 0], [1, 1, 1]])


def test_shared_prefix_scores_one_token_completions():
    _check_suffix([[9], [10], [11]], [[1], [1], [1]])


def test_shared_prefix_matches_binary_answer_scoring():
    _check_suffix([[3], [4], [3]], [[1], [1], [1]], action_count=2)


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_shared_prefix_scores_and_backpropagates_multiple_suffixes()
    test_shared_prefix_scores_one_token_completions()
    test_shared_prefix_matches_binary_answer_scoring()
    print("ok  shared prompt learner logprobs and gradients match dense scoring")
