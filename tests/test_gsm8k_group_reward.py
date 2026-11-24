"""A whole-group reward must be reproducible even for the one-token budget tail."""

from importlib.util import find_spec
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import patch

import torch


@skipUnless(all(find_spec(name) for name in ("mlx", "mlx_lm", "peft")), "optional MLX dependencies")
class GroupRewardTest(TestCase):
    def test_group_context_audit_and_singleton_tail(self):
        from experiments import audit_gsm8k_stable as audit, gsm8k_stable as runner

        model = SimpleNamespace(w=0.)
        snapshot = lambda model: {"weight": torch.tensor([model.w])}
        start = snapshot(model)
        plan = {"train_prompt_ids": {"0": [0, 1]}, "token_budget": 3, "group_size": 2,
                "source_protocol": {"start_sha256": runner.adapter_digest(start)}}
        train = [{"value": "1"}, {"value": "2"}]
        seen = []

        def sample(model, tokenizer, example, protocol, seed, arm, index, remaining, previous, current, version):
            self.assertEqual(model.w, version)
            size = min(2, remaining)
            return [10], [{"id": f"0:fresh:{index}:{i}", "behavior_sha256": current[1], "behavior_version": version,
                           "completion": "", "has_answer_tag": False, "reward": 0, "truncated": False,
                           "token_ids": [151645], "behavior_logprobs": [-0.5], "token_cap": 1 if size == 1 else 2}
                          for i in range(size)]

        def score(samples, example):
            seen.append((len(samples), example["value"]))
            self.assertTrue(all(s["reward"] == s["strict_reward"] == 0 and "reward_reason" not in s for s in samples))
            return [(0.25, "singleton")] if len(samples) == 1 else [(0.5, "preferred"), (0, "other")]

        class Learner:
            def __init__(self, model, lr):
                self.model = model

            def step_groups(self, groups):
                assert groups[0][2][0] > 0 > groups[0][2][1]
                self.model.w += 1
                return {"gnorm": 1.0, "skipped": 0}

        with TemporaryDirectory() as temp, patch.object(runner, "rollouts", sample), \
                patch.object(runner, "set_adapter", lambda m, x: setattr(m, "w", x["weight"].item())), \
                patch.object(runner, "adapter_snapshot", snapshot), patch.object(runner, "MlxGrpoLearner", Learner), \
                patch.object(audit, "prompt_tokens", return_value=[10]):
            folder = Path(temp)
            report = runner.train_case(model, None, train, plan, start, .0001, 1, 0, folder, training_group_reward=score)
            self.assertEqual(report["accounting"]["consumed_tokens"], 2)
            self.assertEqual(report["accounting"]["changed_weights"], 1)

            def check():
                return audit.check_case(folder, plan, train, None, .0001, 1, 0, training_group_reward=score)

            _, rows, _ = check()
            self.assertEqual(seen, [(2, "1"), (1, "2")] * 2)
            self.assertEqual(rows[-1]["advantages"], [0.0])
            for field, wrong, error in (("reward", 0, "group reward"), ("reward_reason", "other", "group reward"),
                                        ("strict_reward", 1, "numeric reward")):
                with self.subTest(field=field):
                    stored = rows[-1]["samples"][0]
                    original = stored[field]
                    stored[field] = wrong
                    (folder / "trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
                    with self.assertRaisesRegex(ValueError, error):
                        check()
                    stored[field] = original
