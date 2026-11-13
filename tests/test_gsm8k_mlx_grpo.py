"""Optional MLX parity check against Narl's PyTorch GRPO update."""

from importlib.util import find_spec
from math import isclose
from types import SimpleNamespace
from unittest import TestCase, main, skipUnless

import torch

from core import Batch
from train import Config, optimize


@skipUnless(all(find_spec(package) for package in ("mlx", "mlx_lm", "peft")),
            "MLX experiment dependencies are optional")
class MlxGrpoTest(TestCase):
    def test_mixed_lengths_clipping_and_zero_advantage_match_torch(self):
        import mlx.core as mx
        import mlx.nn as nn

        from experiments.gsm8k_mlx_grpo import MlxGrpoLearner
        from experiments.gsm8k_regroup import trajectory

        weights = [[0.2, -0.5, 0.1], [-0.1, 0.3, -0.4], [0.6, 0.7, -0.2]]

        class MlxTiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.args = SimpleNamespace(tie_word_embeddings=False)
                self.lm_head = nn.Linear(3, 3, bias=False)
                self.lm_head.weight = mx.array(weights).T

            def model(self, inputs):
                return mx.eye(3)[inputs]

        class TorchTiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(weights))

            def logprobs(self, batch):
                logits = self.weight[batch.states[:, :-1]]
                logp = logits.log_softmax(-1).gather(-1, batch.actions[:, 1:, None]).squeeze(-1)
                return torch.nn.functional.pad(logp, (1, 0)) * batch.mask

        prompt = [1, 0]
        samples = [
            {"token_ids": [2, 1], "behavior_logprobs": [-0.2, -2.0], "reward": 1, "truncated": False},
            {"token_ids": [0], "behavior_logprobs": [-3.4], "reward": 0, "truncated": False},
            {"token_ids": [1], "behavior_logprobs": [-1.3], "reward": 0, "truncated": False},
        ]
        batch = Batch.from_groups([[trajectory(prompt, sample) for sample in samples]])
        torch_model, mlx_model = TorchTiny(), MlxTiny()
        cfg = Config(model="tiny", micro_batch=1, lr=0.001)
        torch_opt = torch.optim.AdamW(torch_model.parameters(), lr=cfg.lr)
        mlx_learner = MlxGrpoLearner(mlx_model, cfg.lr, micro_batch=2)
        for values in ([1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, 0.0]):
            expected = optimize(torch_model, torch_opt, batch, torch.tensor(values)[:, None] * batch.mask, cfg)
            result = mlx_learner.step(prompt, samples, values)
            for key, value in expected.items():
                self.assertTrue(isclose(result[key], value, rel_tol=1e-6, abs_tol=1e-6), (key, result[key], value))
            torch.testing.assert_close(torch_model.weight, torch.tensor(mlx_model.lm_head.weight.T.tolist()),
                                       atol=1e-7, rtol=1e-6)
        self.assertEqual(mlx_learner.optimizer.step.item(), 2)


if __name__ == "__main__":
    main()
