"""Two prompt groups share one learner update with causal right padding."""
from copy import deepcopy
from dataclasses import replace
from math import log
import multiprocessing as mp
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algos import grpo
from core import Batch, Trajectory
from experiments.kv_batch_dispatch import BatchDispatch
from experiments.kv_online import FIRST_CONTENT_ID, GROUP_SIZE, PROMPT_TOKENS, _scoring
from experiments.kv_sequence import _ChangedWeights
from train import Config, optimize


def _group(prompt, patterns, answer):
    group = []
    for index in range(GROUP_SIZE):
        completion = patterns[index % len(patterns)]
        ids = torch.tensor(prompt + [FIRST_CONTENT_ID + bit for bit in completion])
        mask = torch.zeros(ids.shape, dtype=torch.float32)
        mask[PROMPT_TOKENS:] = 1
        rewards = torch.zeros_like(mask)
        rewards[-1] = completion[-1] == answer
        group.append(Trajectory(ids, ids.clone(), mask * -log(2), rewards, mask))
    return group


def test_two_padded_prompt_groups_match_one_dense_grpo_update():
    short = _group([5, 6] * (PROMPT_TOKENS // 2), [(0, 0), (1, 1), (0, 1)], answer=0)
    long = _group([7, 8] * (PROMPT_TOKENS // 2), [(1, 1, 0, 1), (0, 1, 1, 0)], answer=1)
    batch = Batch.from_groups([short, long])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=13, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=PROMPT_TOKENS + 8,
        )).eval()
    shared_model = deepcopy(model)
    dense = SimpleNamespace(model=model, parameters=model.parameters)
    shared = SimpleNamespace(model=shared_model, parameters=shared_model.parameters)
    _scoring(dense, action_count=2)
    _scoring(shared, share_learner=True, action_count=2)

    with torch.no_grad():
        dense_logp = dense.logprobs(batch)
        _scoring(dense, action_count=2, suffix_logits_only=True)
        torch.testing.assert_close(dense.logprobs(batch), dense_logp, atol=2e-6, rtol=1e-5)
        shared_logp = torch.cat([shared.logprobs(group) for _, group in batch.micro_batches(GROUP_SIZE)])
        torch.testing.assert_close(shared_logp, dense_logp, atol=2e-6, rtol=1e-5)
        unpadded_logp = shared.logprobs(Batch.from_groups([short]))
        torch.testing.assert_close(shared_logp[:GROUP_SIZE, :unpadded_logp.size(1)], unpadded_logp,
                                   atol=2e-6, rtol=1e-5)
        assert not shared_logp[:GROUP_SIZE, unpadded_logp.size(1):].any()

        # Independently score the valid actions over the binary policy support.
        valid = batch.mask[:, 1:].bool()
        logits = model(batch.states, use_cache=False).logits[:, :-1, FIRST_CONTENT_ID:FIRST_CONTENT_ID + 2]
        labels = (batch.actions[:, 1:][valid] - FIRST_CONTENT_ID).unsqueeze(1)
        expected = logits[valid].log_softmax(dim=-1).gather(1, labels).squeeze(1)
        torch.testing.assert_close(shared_logp[:, 1:][valid], expected, atol=2e-6, rtol=1e-5)

    cfg = Config(model="fixture", algo="grpo", n_prompts=2, group_size=GROUP_SIZE,
                 micro_batch=GROUP_SIZE, lr=0.001)
    adv = grpo(batch, dense)
    assert adv[:GROUP_SIZE].any() and adv[GROUP_SIZE:].any()
    dense_opt = torch.optim.AdamW(dense.parameters(), lr=cfg.lr)
    shared_opt = torch.optim.AdamW(shared.parameters(), lr=cfg.lr)
    expected = optimize(dense, dense_opt, batch, adv, replace(cfg, micro_batch=0))
    actual = optimize(shared, shared_opt, batch, adv, cfg)
    assert expected["gnorm"] > 0
    for key in ("loss", "gnorm"):
        torch.testing.assert_close(actual[key], expected[key], atol=2e-6, rtol=1e-5)
    for actual_param, expected_param in zip(shared.parameters(), dense.parameters()):
        torch.testing.assert_close(actual_param.grad, expected_param.grad, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(actual_param, expected_param, atol=2e-6, rtol=1e-5)
    assert all(state["step"] == 1 for state in shared_opt.state.values())


def test_version_tracks_snapshot_bytes_and_empty_publication_releases_batch():
    ctx = mp.get_context("spawn")
    model = torch.nn.Linear(1, 1, bias=False)
    replica = deepcopy(model)
    with torch.no_grad():
        model.weight.fill_(0.0)
    weights = _ChangedWeights(ctx, model)
    dispatch = BatchDispatch(ctx, weights, total=6, window=2, max_lag=1, version_aware=True)
    initial_digest = weights.digest
    assert weights.pull(replica, -1) == 0
    for expected_before, expected_after, replacement in [(0, 0, None), (0, 1, -0.0), (1, 2, 0.0)]:
        assert dispatch.admit(*dispatch.acquire()) == 1
        assert dispatch.admit(*dispatch.acquire()) == 2
        if replacement is not None:
            with torch.no_grad():
                model.weight.fill_(replacement)
        change = dispatch.complete_update(model)
        assert change["version_before"] == expected_before and change["version_after"] == expected_after
        assert change["weights_changed"] == (replacement is not None)
        assert (change["weight_sha256_before"] != change["weight_sha256_after"]) == change["weights_changed"]
        assert weights.pull(replica, expected_before) == expected_after
        assert torch.equal(replica.weight.view(torch.uint8), model.weight.view(torch.uint8))
    assert weights.digest == initial_digest and weights.version.value == 2
    assert dispatch.acquire() is None


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_two_padded_prompt_groups_match_one_dense_grpo_update()
    test_version_tracks_snapshot_bytes_and_empty_publication_releases_batch()
    print("ok  two mixed-length prompt groups match the joint dense GRPO update")
    print("ok  byte-exact policy versions and no-op batch release")
