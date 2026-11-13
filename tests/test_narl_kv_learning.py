import unittest

from collections import Counter

from experiments.kv_learning import DevelopmentQueryClassTask, QueryClassTask, target_crossing
from experiments.kv_online import PROMPT_TOKENS, _batch


class LearnableWorkloadTests(unittest.TestCase):
    def test_disjoint_operand_pairs_and_fillers(self):
        main = QueryClassTask(320, 64, seed=0)
        pilot = DevelopmentQueryClassTask(320, 64, seed=100)
        for key in ("prompt", "filler", "operands"):
            train = {row[key] for row in main.data}
            heldout = {row[key] for row in main.eval_data}
            development = {row[key] for row in pilot.eval_data}
            self.assertFalse(train & heldout)
            self.assertFalse(train & development)
            self.assertFalse(heldout & development)
        self.assertEqual({row["answer"] for row in main.data},
                         {row["answer"] for row in main.eval_data})
        self.assertEqual(Counter(row["answer"] for row in main.eval_data), {"0": 32, "1": 32})
        self.assertEqual(Counter(row["answer"] for row in main.data), {"0": 160, "1": 160})
        train_distractors = {row["operands"][0] for row in main.data}
        self.assertFalse(train_distractors & {row["operands"][0] for row in main.eval_data})
        self.assertFalse(train_distractors & {row["operands"][0] for row in pilot.eval_data})
        self.assertEqual(main.eval_data, QueryClassTask(320, 64, seed=1).eval_data)

    def test_only_answer_token_receives_credit(self):
        payload = {"tokens": [[3] * PROMPT_TOKENS + [5, 6, 7]],
                   "behavior_logprobs": [[-0.3, -0.4, -0.5]], "rewards": [1.0]}
        batch = _batch(payload, credit_tokens=1)
        self.assertEqual(batch.states.size(1), PROMPT_TOKENS + 1)
        self.assertEqual(int(batch.states[0, -1]), 5)
        self.assertEqual(float(batch.mask.sum()), 1.0)
        self.assertEqual(float(batch.rewards[0, -1]), 1.0)
        self.assertAlmostEqual(float(batch.old_logp[0, -1]), -0.3)

    def test_target_crossing_requires_two_distinct_evaluations(self):
        first = {"time_s": 0, "generated_tokens": 0, "actor_completed_tokens": 0,
                 "reward": 0.5, "update": 0}
        cross = {"time_s": 2, "generated_tokens": 1_000, "actor_completed_tokens": 1_064,
                 "reward": 0.75, "update": 24}
        later = {"time_s": 4, "generated_tokens": 2_000, "actor_completed_tokens": 2_064,
                 "reward": 1.0, "update": 48}
        self.assertIsNone(target_crossing([first], 0.75))
        self.assertIsNone(target_crossing([first, cross, cross], 0.75))
        self.assertEqual(target_crossing([first, cross, later], 0.75),
                         {"wall_s": 4, "received_generated_tokens": 2_000, "actor_completed_tokens": 2_064,
                          "accuracy": 1.0, "update": 48})


if __name__ == "__main__":
    unittest.main()
