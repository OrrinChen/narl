import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.audit_gsm8k_reward_replication import SEEDS, audit_panel, check_question_blocks
from experiments.audit_gsm8k_stable import comparison


class TestReplicationAudit(unittest.TestCase):
    def test_the_same_question_is_sampled_for_all_five_endpoints(self):
        def make(results):
            rows = [{"reward": value, "token_ids": [value], "prompt_id": i} for i, value in enumerate(results)]
            return {str(s): {"rows": rows} for s in SEEDS}
        start = make([0, 0, 1, 1])
        candidate = make([1, 1, 0, 0])
        stats = comparison(start, candidate, list(range(4)))
        check_question_blocks(start, candidate, stats)
        self.assertEqual(stats["bootstrap"]["question_block_median_95"], [-1, 1])
        stats["bootstrap"]["question_block_median_95"] = [-0.5, 0.5]
        with self.assertRaisesRegex(ValueError, "question-block interval"):
            check_question_blocks(start, candidate, stats)

    def test_reward_is_recomputed_from_original_gold(self):
        with TemporaryDirectory() as temp:
            panel = Path(temp)
            (panel / "gate.json").write_text("{}")
            report = {"protocol_sha256": "freeze", "checkpoint_sha256": "start", "correct": 1,
                      "rows": [{"prompt_id": 0, "reward": 1, "completion": "<answer>9</answer>",
                                "token_ids": [151645], "truncated": False, "has_answer_tag": True}]}
            (panel / "initial.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "wrong numeric reward"):
                audit_panel(panel, [0], {"initial": "start"}, "freeze", [("question", 8)], None)
