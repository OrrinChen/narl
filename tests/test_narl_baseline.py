"""Deterministic CPU contract checks for the inherited GRPO training path."""

import copy
from contextlib import contextmanager
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algos import grpo
from core import Batch, Trajectory
from model import MLPPolicy
from serve import RolloutClient
from train import AsyncSource, Config, SyncSource, optimize, score_batch


def _group(rewards):
    trajectories = []
    for index, reward in enumerate(rewards):
        actions = torch.tensor([index, (index + 1) % 3])
        trajectories.append(Trajectory(
            states=torch.tensor([[0.5, -0.2], [0.9, 0.1]]),
            actions=actions,
            old_logp=torch.zeros(2),
            rewards=torch.tensor([0.0, reward]),
            mask=torch.ones(2),
        ))
    return Batch.from_groups([trajectories])


class _FixedRollout:
    def sample(self, n):
        return ["fixture"] * n

    def rollout(self, policy, prompts, group_size, *, score):
        return _group([0.0, 2.0])


@contextmanager
def _loopback(config):
    config.serve_host = "127.0.0.1"
    source = AsyncSource(config)
    try:
        port = source.server.server.server_port
        yield source, RolloutClient(f"http://127.0.0.1:{port}")
    finally:
        source.server.close()


def test_same_group_matches_sync_grpo_update_at_zero_lag():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(37)
        initial_policy = MLPPolicy(2, 3)
        worker_policy = MLPPolicy(2, 3)
    sync_policy = copy.deepcopy(initial_policy)
    async_policy = copy.deepcopy(initial_policy)
    config = Config(model="fixture", group_size=2, micro_batch=2, device="cpu",
                    serve_port=0, max_staleness=0, pop_timeout=5)
    task = _FixedRollout()
    sync_batch, sync_lag = SyncSource(task, config, {}, 1).next_batch(sync_policy)

    with _loopback(config) as (source, client):
        source.publish(async_policy)
        version, weights = client.pull_weights(have=0)
        assert version == 1
        worker_policy.load_state_dict(weights)
        submitted = _group([0.0, 2.0])
        submitted.old_logp = score_batch(worker_policy, submitted, config)
        client.submit(version, submitted)
        async_batch, async_lag = source.next_batch(async_policy)

    assert sync_lag == async_lag == 0
    for field in ("states", "actions", "rewards", "mask", "group_id", "old_logp"):
        assert torch.equal(getattr(sync_batch, field), getattr(async_batch, field)), field
    assert torch.equal(async_batch.sample_logp, submitted.old_logp)

    sync_advantage = grpo(sync_batch, sync_policy)
    async_advantage = grpo(async_batch, async_policy)
    assert torch.equal(sync_advantage, async_advantage)

    sync_optimizer = torch.optim.AdamW(sync_policy.parameters(), lr=0.01, weight_decay=0)
    async_optimizer = torch.optim.AdamW(async_policy.parameters(), lr=0.01, weight_decay=0)
    sync_metrics = optimize(sync_policy, sync_optimizer, sync_batch, sync_advantage, config)
    async_metrics = optimize(async_policy, async_optimizer, async_batch, async_advantage, config)

    assert sync_metrics == async_metrics
    assert all(torch.equal(a, b) for a, b in zip(sync_policy.parameters(), async_policy.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(sync_policy.parameters(), initial_policy.parameters()))


def test_group_expiring_in_queue_is_rejected_at_consumption():
    config = Config(device="cpu", serve_port=0, max_staleness=1, pop_timeout=5)
    policy = MLPPolicy(2, 3)

    with _loopback(config) as (source, client):
        source.publish(policy)
        old_version = source.version
        client.submit(old_version, _group([10.0, 20.0]))
        source.publish(policy)
        source.publish(policy)
        assert source.version - old_version == config.max_staleness + 1

        client.submit(source.version, _group([1.0, 3.0]))
        assert source.stats()["queued"] == 2
        assert source.stats()["dropped"] == 0

        accepted, lag = source.next_batch(policy)
        assert lag == 0
        assert torch.equal(accepted.group_terminal_rewards(), torch.tensor([[1.0, 3.0]]))
        assert source.stats()["dropped"] == 1
        assert source.stats()["dropped_stale"] == 1
        assert source.stats()["dropped_overflow"] == 0
        assert source.stats()["queued"] == 0


if __name__ == "__main__":
    for test in (test_same_group_matches_sync_grpo_update_at_zero_lag,
                 test_group_expiring_in_queue_is_rejected_at_consumption):
        test()
        print(f"ok  {test.__name__}")
