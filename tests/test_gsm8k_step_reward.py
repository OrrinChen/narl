import unittest

from experiments.gsm8k_step_reward import score_format_reward, score_training_reward


class VerifiedStepRewardTest(unittest.TestCase):
    example = {"rationale": "First <<4*12=48>>48, then <<48-2=46>>46.", "value": "46"}

    def score(self, text, scorer=score_training_reward):
        return scorer({"completion": text, "truncated": False}, self.example)[0]

    def test_partial_credit_requires_a_distinct_correct_reference_equation(self):
        self.assertEqual(self.score("12 × 4 = 48.\n4*12=48\n<answer>0</answer>"), 0.2)
        for text in ("1 + 12 * 4 = 48", "4 * 12 = 49", "2+2=4", "4*12=48+4",
                     "<answer>0</answer>\n4 * 12 = 48"):
            with self.subTest(text=text):
                self.assertEqual(self.score(text), 0)

    def test_final_correct_outranks_partial_and_format_credit(self):
        self.assertEqual(self.score("4*12=48\n48-2=46\n<answer>0</answer>", score_format_reward), 0.45)
        self.assertEqual(self.score("The answer is 46.", score_format_reward), 0.8)
        self.assertEqual(self.score("<answer>46</answer>", score_format_reward), 1.0)
