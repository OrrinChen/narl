import unittest

from experiments.gsm8k_training_reward_v3 import score_training_reward


def reward(text, gold, truncated=False):
    return score_training_reward({"completion": text, "truncated": truncated}, gold)[0]


class TrainingRewardV3Test(unittest.TestCase):
    def test_terminal_forms_in_archived_answers(self):
        cases = [
            ("The total cost is 100.\n<answer>#100</answer>", 100),
            ("Clare has money left.\n<answer>$47-12=35$</answer>", 35),
            ("Noemi began with $1700. When writing the result, the correct answer is 1700.", 1700),
            (r"So, the total number of tabs is \(\boxed{60}\).", 60),
            ("The final answer is:\n" + r"\[ \boxed{4} \]", 4),
            (r"\[ \boxed{10 \text{ sections}} \]", 10),
            ("<answer>Family left with $56</answer>", 56),
            ("12 inches", 12),
        ]
        for text, gold in cases:
            with self.subTest(text=text):
                self.assertEqual(reward(text, gold), 1)
                self.assertEqual(reward(text, gold, truncated=True), 0)

    def test_abstains_on_ambiguous_or_nonfinal_claims(self):
        cases = [
            (r"The subtotal is \boxed{56}.", 56),
            ("Intermediate result:\n" + r"\boxed{56}.", 56),
            (r"Maybe the final answer is \boxed{56}.", 56),
            (r"The answer is 11. Thus, the answer is \boxed{56}.", 56),
            (r"Thus, the answer is \boxed{56}. Next question?", 56),
            (r"\boxed{12} or \boxed{56}", 56),
            ("<answer>#-100</answer>", 100),
            ("<answer>$1/56$</answer>", 56),
            ("<answer>Family might have $56</answer>", 56),
            ("<answer>Family left with $56 or $40</answer>", 56),
            ("<answer>#100</answer>\nActually the correct answer is 30.", 100),
            ("<answer>30</answer> or <answer>#100</answer>", 100),
            ("Step 2:\n12 inches", 12),
            ("3. 12 inches", 12),
            ("<answer>costs -$100</answer>", 100),
        ]
        for text, gold in cases:
            with self.subTest(text=text):
                self.assertEqual(reward(text, gold), 0)
