import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.audit_gsm8k_curriculum import check_binary, check_prefix, training_metrics
from experiments.audit_gsm8k_stable import check_case, digest
from experiments.gsm8k_training_reward_v2 import score_training_reward


class CurriculumAuditTest(unittest.TestCase):
    def test_strict_and_training_rewards_are_verified_separately(self):
        sample = {"completion": "The answer is 10.", "token_ids": [151645], "behavior_logprobs": [-1.],
                  "truncated": False, "has_answer_tag": False, "strict_reward": 0}
        sample["reward"], sample["reward_reason"] = score_training_reward(sample, "10")
        self.assertEqual(sample["reward"], 1)
        self.assertEqual(check_binary(sample, {"value": "10"}, 5, "case", None), 1)
        for field, value in (("reward", 0), ("strict_reward", 1), ("reward_reason", "incorrect")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_binary({**sample, field: value}, {"value": "10"}, 5, "case", None)

    def test_density_counts_all_generated_tokens_including_incomplete_tail(self):
        report = {"start_sha256": "before", "final_sha256": "after",
                  "accounting": {"generated_tokens": 100, "changed_weights": 1}}
        rows = [{"samples": [{"reward": reward} for reward in group]} for group in [[0, 1], [0, 0], [1]]]
        result = training_metrics((report, rows, [{"nonzero_groups": 1}]))
        self.assertEqual((result["complete_groups"], result["effective_groups"],
                          result["effective_groups_per_10000_generated_tokens"]), (2, 1, 100))

    def test_rejects_replaced_short_trace_and_endpoint(self):
        short = ({"final_sha256": "old"}, [{"id": 0}], [{"batch": 0}])
        good = ({}, [{"id": 0}, {"sha256": "old"}], [{"batch": 0}, {"old_sha256": "old"}])
        check_prefix(short, good)
        for bad in (({}, [{"id": 1}, {"sha256": "old"}], good[2]),
                    ({}, good[1], [{"batch": 0}, {"old_sha256": "another"}])):
            with self.assertRaises(ValueError):
                check_prefix(short, bad)

    def test_budget_boundary_caps_generation_and_flushes_partial_optimizer_batch(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            sha = "0" * 64
            rows, updates = [], []
            for i in range(2):
                samples = [{"id": f"0:fresh:{i}:{slot}", "token_cap": cap, "behavior_version": 0, "behavior_sha256": sha,
                            "reward": 0, "token_ids": [151645] * cap} for slot, cap in enumerate([2, 1])]
                rows.append({"group_index": i, "prompt_id": 0, "prompt_ids": [4], "batch_index": i, "version": 0,
                             "sha256": sha, "samples": samples, "advantages": [0., 0.], "generated_tokens": 3,
                             "consumed_tokens": 3, "nonzero_advantage_tokens": 0, "learner_input_tokens": 0})
                updates.append({"batch_index": i, "group_indices": [i], "sequence_count": 2, "nonzero_groups": 0,
                                "old_version": 0, "old_sha256": sha, "new_version": 0, "new_sha256": sha,
                                "cumulative_generated_tokens": (i + 1) * 3, "diagnostics": {"skipped": 2, "gnorm": 0}})
            for name, items in (("trace", rows), ("updates", updates)):
                (folder / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in items))
            report = {"lr": 1e-4, "accumulation_groups": 16, "seed": 0, "start_sha256": sha, "final_sha256": sha,
                      "trace_sha256": digest(folder / "trace.jsonl"), "updates_sha256": digest(folder / "updates.jsonl"),
                      "training_elapsed_s": 1., "accounting": {"groups": 2, "generated_tokens": 6, "supervised_tokens": 0,
                      "consumed_tokens": 6, "nonzero_advantage_tokens": 0, "learner_input_tokens": 0,
                      "generation_prompt_tokens": 4, "optimizer_batches": 2, "changed_weights": 0}}
            (folder / "training.json").write_text(json.dumps(report))
            plan = {"source_protocol": {"start_sha256": sha}, "token_budget": 6, "train_prompt_ids": {"0": [0, 0]}}
            args = (folder, plan, [{"value": 0}], None, 1e-4, 16, 0)
            def validate(sample, *_):
                return len(sample["token_ids"])
            with patch("experiments.audit_gsm8k_stable.prompt_tokens", return_value=[4]), \
                 patch("experiments.audit_gsm8k_stable.adapter_digest", return_value=sha):
                check_case(*args, checkpoints=(3,), training_sample_validator=validate)
                with self.assertRaisesRegex(ValueError, "generation cap"):
                    check_case(*args, training_sample_validator=validate)
