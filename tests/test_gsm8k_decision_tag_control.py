import unittest

from experiments.gsm8k_decision_eval import evaluate, retry_order
from experiments.gsm8k_decision_tag_control import compare_control, replay, tag_priority
from tests.test_gsm8k_decision_eval import panel, sample, truncate


class SupplementalTagControlTests(unittest.TestCase):
    def test_missing_tag_first_and_ties_keep_frozen_permutation(self):
        fields = {
            10: {"has_answer_tag": True, "mean_logprob": -4},
            20: {"has_answer_tag": False, "mean_logprob": -0.1},
            30: {"has_answer_tag": True, "mean_logprob": -4},
            40: {"has_answer_tag": False, "mean_logprob": -1.1},
        }
        tied = [q for q in retry_order(fields, "uniform", 123, 0) if q in (10, 30)]
        self.assertEqual(tag_priority(fields, 123, 0), [40, 20, *tied])

    def test_capped_retry_and_two_tagged_ranking_coverage(self):
        rows = panel()
        rows[0]["samples"][0] = sample(2, 0, False, -0.05, 0)
        primary = evaluate(rows, 4, truncate, 41)
        addition = compare_control(rows, primary, 41, truncate)
        self.assertEqual(addition["both_tagged_ranking_pooled"],
                         {"pairs": 3, "rule_correct": 0, "judge_correct": 3, "wins": 3, "losses": 0, "ties": 0})
        details = addition["per_seed"]["0"]["questions"]
        self.assertEqual([(q["prompt_id"], q["tokens"], q["correct"]) for q in details], [(20, 3, 1), (40, 1, 1)])
        self.assertEqual(addition["per_seed"]["0"]["total_tokens"], primary["per_seed"]["0"]["allocation"]["judge"]["total_tokens"])
        with self.assertRaisesRegex(ValueError, "prefix"):
            replay({row["prompt_id"]: row for row in rows[:2]}, 1, 41, 0, lambda r, s, n: s)
