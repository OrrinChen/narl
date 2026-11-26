import copy
import unittest

from experiments.gsm8k_decision_eval import evaluate, retry_order, rule_select


def sample(tokens, reward, tag, logp, p, elapsed=1):
    return {"token_ids": list(range(1, tokens + 1)), "has_answer_tag": tag, "strict_reward": reward,
            "mean_logprob": logp, "decision_score": p, "generation_s": elapsed, "decision_s": 0.2}


def panel():
    a = [sample(2, 0, True, -0.3, 0.1), sample(3, 1, True, -0.8, 0.8)]
    b = [sample(3, 1, True, -0.9, 0.7), sample(3, 0, True, -0.1, 0.2)]
    return [{"seed": s, "prompt_id": q, "samples": copy.deepcopy(values)}
            for s in (0, 1) for q, values in ((20, a), (40, b))]


def truncate(row, retry, n):
    # The answer tag is written last. Its tail is absent from a truncated prefix.
    return {**retry, "token_ids": retry["token_ids"][:n], "strict_reward": 0,
            "has_answer_tag": False, "generation_s": n / len(retry["token_ids"])}


class DecisionEvaluationTests(unittest.TestCase):
    def test_order_only_needs_visible_fields(self):
        signals = {40: {"mean_logprob": -0.2, "decision_score": 0.1},
                   20: {"mean_logprob": -0.9, "decision_score": 0.9}}
        self.assertEqual(retry_order(signals, "judge", 11, 0), [40, 20])
        self.assertEqual(retry_order(signals, "confidence", 11, 0), [20, 40])
        equal = {i: {"mean_logprob": -0.3, "decision_score": 0.5} for i in [21, 22, 23, 24]}
        uniform = retry_order(equal, "uniform", 142, 0)
        for arm in ("judge", "confidence"):
            self.assertEqual(retry_order(dict(reversed(equal.items())), arm, 142, 0), uniform)
        tied = sample(2, 1, False, -0.2, 0.0)
        self.assertEqual(rule_select(tied, tied), 0)

    def test_paired_blocks_and_same_final_rule(self):
        report = evaluate(panel(), 4, truncate, 41)
        fixed = report["fixed_pair_scores"]
        self.assertEqual(fixed["judge"]["accuracy"], 1)
        self.assertEqual(fixed["rule"]["accuracy"], 0)
        self.assertEqual(fixed["random_expected"]["accuracy"], 0.5)
        delta = report["fixed_pair_contrasts"]["judge_vs_rule"]
        self.assertEqual(delta["question_block_median_95"], [1.0, 1.0])
        self.assertEqual((delta["wins"], delta["losses"]), (4, 0))
        for item in report["per_seed"].values():
            self.assertEqual(item["allocation"]["confidence"]["correct"], 0)
            self.assertEqual(item["allocation"]["judge"]["correct"], 1)
            self.assertEqual(item["allocation"]["judge"]["wrong_first_correct_retry"], 1)
            self.assertAlmostEqual(item["allocation"]["judge"]["profile_estimated_s"], 2 + 1 + 1/3 + 0.4)
            for arm in item["allocation"].values():
                self.assertEqual((arm["total_tokens"], arm["retry_tokens"], arm["retried_questions"]), (9, 4, 2))
            self.assertEqual(item["allocation"]["judge"]["partial_retry_question"], 40)
        self.assertEqual(report["allocation_contrasts"]["judge_vs_confidence"]["wins"], 2)

    def test_budget_edges_and_question_alignment(self):
        both = panel()
        zero = evaluate(both, 0, truncate, 41)
        for method in ("uniform", "confidence", "judge"):
            self.assertEqual(zero["allocation_scores"][method], zero["allocation_scores"]["first"])
            self.assertEqual(zero["per_seed"]["0"]["allocation"][method]["judge_scoring_s"], 0)
        with self.assertRaisesRegex(ValueError, "exhausted"):
            evaluate(both, 7, truncate, 41)
        with self.assertRaisesRegex(ValueError, "same panel"):
            evaluate(both[:-1], 1, truncate, 41)
        with self.assertRaisesRegex(ValueError, "invalid truncated"):
            evaluate(both, 1, lambda row, s, n: s, 41)

    def test_conditional_interval_keeps_question_draws_paired_across_seeds(self):
        rows = panel()
        for row in rows:
            first, retry = row["samples"]
            first["mean_logprob"], retry["mean_logprob"] = -0.1, -0.9
            first["decision_score"], retry["decision_score"] = 0.2, 0.8
            first["strict_reward"] = int((row["seed"] == 1) == (row["prompt_id"] == 20))
            retry["strict_reward"] = 1 - first["strict_reward"]
        delta = evaluate(rows, 1, truncate, 41)["fixed_pair_contrasts"]["judge_vs_rule"]
        self.assertEqual((delta["wins"], delta["losses"]), (2, 2))
        self.assertEqual(delta["question_block_mean_95"], [0, 0])
