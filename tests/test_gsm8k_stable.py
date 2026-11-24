"""Exercise frozen-policy batch boundaries without loading model weights."""

from importlib.util import find_spec
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main, skipUnless
from unittest.mock import patch

import torch


@skipUnless(all(find_spec(package) for package in ("mlx", "mlx_lm", "peft")),
            "MLX experiment dependencies are optional")
class StableGrpoBatchTest(TestCase):
    def test_audit_reproduces_fractional_advantage_bits(self):
        from experiments.audit_gsm8k_stable import replay_advantages
        from experiments.gsm8k_online_lag import advantages

        # Matching one or two of three reference operations gives these step rewards.
        # NumPy's float32 std can differ from torch's by one ULP for this G2.
        for scores in ((0.4 / 3, 0.4 * 2 / 3), (1, 0), (0.05, 0.05)):
            samples = [{"reward": score, "token_ids": [7], "behavior_logprobs": [-1.0], "truncated": False}
                       for score in scores]
            expected, _ = advantages([3], samples)
            actual = torch.tensor(replay_advantages(samples), dtype=torch.float32)
            self.assertTrue(torch.equal(actual.view(torch.int32), torch.tensor(expected).view(torch.int32)))

    def test_frozen_batches_zero_signal_and_budget_tails(self):
        from experiments import gsm8k_stable as runner

        cases = (
            # Two groups update together, followed by one complete tail group.
            ([(4, 4), (4, 4), (2, 2)], [(1, 0), (0, 0), (0, 1)],
             [0, 0, 1], [[0, 1], [2]], 20, 2),
            # A charged single-token tail has no corresponding learner batch.
            ([(4, 4), (4, 4), (1,)], [(1, 0), (0, 0), (0,)],
             [0, 0, 1], [[0, 1]], 16, 1),
            # A single-token tail still flushes earlier complete pending groups.
            ([(4, 4), (1,)], [(1, 0), (0,)],
             [0, 0], [[0]], 8, 1),
            # All-zero advantages consume budget and retain the weight version.
            ([(4, 4), (4, 4)], [(0, 0), (1, 1)],
             [0, 0], [[0, 1]], 16, 0),
        )
        for lengths, rewards, versions, expected_batches, consumed, final_version in cases:
            with self.subTest(lengths=lengths, rewards=rewards), TemporaryDirectory() as temporary:
                model = SimpleNamespace(weight=0.0)
                snapshots = lambda model: {"weight": torch.tensor([model.weight])}
                start = snapshots(model)
                start_hash = runner.adapter_digest(start)
                sampled, learned = [], []

                def restore(model, snapshot):
                    model.weight = snapshot["weight"].item()

                def sample(model, tokenizer, example, protocol, seed, arm, index, remaining,
                           previous, current, version):
                    self.assertEqual(model.weight, current[0]["weight"].item())
                    self.assertEqual(current[1], runner.adapter_digest(snapshots(model)))
                    sampled.append((version, model.weight, current[1], remaining))
                    prompt = [index + 1] * (index + 1)
                    outputs = [{"token_ids": [slot + 1] * length, "behavior_logprobs": [-1.0] * length,
                                "reward": reward, "truncated": False, "behavior_version": version,
                                "behavior_sha256": current[1]}
                               for slot, (length, reward) in enumerate(zip(lengths[index], rewards[index], strict=True))]
                    return prompt, outputs

                class Learner:
                    def __init__(self, model, lr):
                        self.model = model

                    def step_groups(self, groups):
                        learned.append(groups)
                        active = any(any(a != 0 for a in values) for _, _, values in groups)
                        if active:
                            self.model.weight += 1.0
                        return {"loss": 0.0, "gnorm": float(active)}

                budget = sum(sum(row) for row in lengths)
                plan = {"source_protocol": {"start_sha256": start_hash}, "token_budget": budget,
                        "group_size": 2, "train_prompt_ids": {"0": list(range(len(lengths)))}}
                folder = Path(temporary)
                with patch.object(runner, "set_adapter", restore), patch.object(runner, "rollouts", sample), \
                        patch.object(runner, "adapter_snapshot", snapshots), patch.object(runner, "MlxGrpoLearner", Learner):
                    report = runner.train_case(model, None, [{}] * len(lengths), plan, start, 0.0001, 2, 0, folder)
                rows = [json.loads(line) for line in (folder / "trace.jsonl").read_text().splitlines()]
                updates = [json.loads(line) for line in (folder / "updates.jsonl").read_text().splitlines()]
                self.assertEqual([row["version"] for row in rows], versions)
                self.assertEqual([row["batch_index"] for row in rows], [i // 2 for i in range(len(rows))])
                self.assertEqual([value[0] for value in sampled], versions)
                self.assertEqual([value[1] for value in sampled], versions)
                self.assertEqual([u["group_indices"] for u in updates], expected_batches)
                self.assertEqual([u["sequence_count"] for u in updates], [len(ids) * 2 for ids in expected_batches])
                self.assertEqual([len(groups) for groups in learned], [len(ids) for ids in expected_batches])
                self.assertEqual(report["accounting"]["generated_tokens"], budget)
                self.assertEqual(report["accounting"]["consumed_tokens"], consumed)
                self.assertEqual(report["accounting"]["optimizer_batches"], len(expected_batches))
                self.assertEqual(report["accounting"]["changed_weights"], final_version)
                previous_hash = start_hash
                for batch_index, update in enumerate(updates):
                    self.assertEqual(update["batch_index"], batch_index)
                    self.assertEqual(update["old_sha256"], previous_hash)
                    for index in update["group_indices"]:
                        self.assertEqual(rows[index]["sha256"], previous_hash)
                    self.assertEqual(update["new_version"] - update["old_version"],
                                     int(update["new_sha256"] != update["old_sha256"]))
                    previous_hash = update["new_sha256"]
                self.assertEqual(report["final_sha256"], previous_hash)
                if len(lengths[-1]) == 1:
                    self.assertEqual(rows[-1]["generated_tokens"], 1)
                    self.assertEqual(rows[-1]["consumed_tokens"], 0)
                    self.assertTrue(all(len(samples) == 2 for groups in learned for _, samples, _ in groups))


if __name__ == "__main__":
    main()
