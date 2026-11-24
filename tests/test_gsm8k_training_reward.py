import unittest

from experiments.gsm8k_training_reward import score_training_reward


def reward(text, gold, truncated=False):
    return score_training_reward({"completion": text, "truncated": truncated}, gold)[0]


class TrainingRewardTest(unittest.TestCase):
    def test_final_answers(self):
        for text, gold in [
            ("The answer is $900.", 900),
            ("3×4 + 6×6 + 2×5 = 58.\nTherefore, the answer is 58.", 58),
            ("Therefore, the answer is 20 cats × 2 = 40 cats.", 40),
            ("Thus, the total time is 30 minutes.", 30),
            ("<solution>\n15\n</solution>", 15),
            ("<number>7</number>", 7),
            ("<answer>$24. 00 + $8. 00 = $32. 00</answer>", "32"),
            ("<answer>2/4</answer>", ".5"),
            ("<answer>-$15</answer>", -15),
            ("<answer>1,250</answer> ounces of milk remain.", 1250),
        ]:
            with self.subTest(text=text):
                self.assertEqual(reward(text, gold), 1)
        self.assertEqual(reward("<answer>42</answer>", 42, truncated=True), 1)

    def test_ambiguous_and_intermediate_numbers(self):
        cases = [
            ("An intermediate calculation follows.\n6 * 7 = 42", 42),
            ("Steps remaining:\n42.", 42),
            ("The answer is not 42.", 42),
            ("The answer is 1 or 42.", 42),
            ("The answer is 1,42.", 42),
            ("The answer is 42 million.", 42),
            ("The answer is 20.\nThe answer is 42.", 42),
            ("The answer is 20.\n<answer>42</answer>", 42),
            ("Maybe <answer>42</answer>", 42),
            ("<answer>3</answer> or <answer>42</answer>", 42),
            ("<answer>42</answer>\nActually the answer is 24.", 42),
            ("<answer>10</answer> 10 was intermediate. The final answer is 20.", 10),
            ("<answer>10</answer> 10 seemed plausible. It should be 20.", 10),
            ("<answer>10</answer> Initially 10, final answer: 20.", 10),
            ("<answer>42</answer>\n42 was an intermediate. The final answer is 24.", 42),
            ("<answer>42</answer>\nAnother problem: what is 42?", 42),
            ("<answer>42</answer>\n<answer>", 42),
            ("<answer>1/42</answer>", 42),
            ("<answer>2/0</answer>", 0),
            ("<answer>-$42</answer>", 42),
            ("<answer>20 + 42</answer>", 42),
            ("<answer>5 hours, 42 minutes</answer>", 42),
        ]
        for text, gold in cases:
            with self.subTest(text=text):
                self.assertEqual(reward(text, gold), 0)
        self.assertEqual(reward("The answer is 42.", 42, truncated=True), 0)
        self.assertEqual(reward("<answer>42</answer>\nExplain the solution", 42, truncated=True), 0)
