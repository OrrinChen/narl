"""In-flight weight change oracle for the CPU online rollout path."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.cpu_baseline import save_fixture
from experiments.kv_online import (
    FIRST_CONTENT_ID, PROMPT_TOKENS, _batch, _generate, _scoring,
)
from tasks import TokenChoiceTask
from train import Config, make_policy


class _SwitchOnce:
    def __init__(self, old, new):
        self.old = old
        self.new = new

    def pull(self, model, have):
        if have == -1:
            model.load_state_dict(self.old)
            return 0
        assert have == 0
        model.load_state_dict(self.new)
        return 1


def test_fixed_switch_preserves_old_logprobs_and_recompute_equivalence():
    with TemporaryDirectory() as path:
        save_fixture(Path(path))
        config = Config(model=path, device="cpu", dtype="float32")
        policy = make_policy(config, None)
        task = TokenChoiceTask()
        problem, = task.sample(1)
        old = {name: tensor.detach().clone() for name, tensor in policy.model.state_dict().items()}
        new = {name: tensor.clone() for name, tensor in old.items()}
        new["model.layers.0.self_attn.v_proj.weight"].neg_()
        weights = _SwitchOnce(old, new)
        runs = [
            _generate(policy, task, problem, 9, weights, mode, torch.Generator().manual_seed(23))
            for mode in ("recompute", "share_prompt")
        ]

        independent, shared = runs
        assert independent["tokens"] == shared["tokens"]
        assert independent["rewards"] == shared["rewards"]
        assert independent["version_by_token"] == shared["version_by_token"] == [0] * 4 + [1] * 5
        assert independent["switches"][0]["prompt_tokens_recomputed"] == 8 * PROMPT_TOKENS
        assert shared["switches"][0]["prompt_tokens_recomputed"] == PROMPT_TOKENS
        assert shared["switches"][0]["suffix_tokens_recomputed"] == 8 * 3

        _scoring(policy)
        batch = _batch(shared)
        with torch.no_grad():
            policy.model.load_state_dict(old)
            old_scores = policy.logprobs(batch)[:, PROMPT_TOKENS:PROMPT_TOKENS + 4]
            policy.model.load_state_dict(new)
            new_scores = policy.logprobs(batch)[:, PROMPT_TOKENS + 4:]
        assert int(batch.actions[:, PROMPT_TOKENS:].min()) >= FIRST_CONTENT_ID
        expected = torch.cat((old_scores, new_scores), dim=1)
        for run in runs:
            torch.testing.assert_close(torch.tensor(run["behavior_logprobs"]), expected,
                                       atol=2e-6, rtol=1e-5)


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_fixed_switch_preserves_old_logprobs_and_recompute_equivalence()
    print("ok  in-flight switch behavior logprobs and recompute parity")
