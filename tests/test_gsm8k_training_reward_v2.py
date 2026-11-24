import unittest

from experiments.gsm8k_training_reward_v2 import score_training_reward


def score(text, gold=22, truncated=False):
    return score_training_reward({"completion": text, "truncated": truncated}, gold)


class TrainingRewardV2Test(unittest.TestCase):
    def test_number_typography_and_units(self):
        self.assertEqual(score("<answer>6+2*（6/3）+4=24（miles）</answer>", 24)[0], 1)
        self.assertEqual(score("<answer>２４（miles）</answer>", 24)[0], 1)
        self.assertEqual(score("<answer>２４（million）</answer>", 24)[0], 0)
        self.assertEqual(score("<answer>3²</answer>", 32)[0], 0)
        self.assertEqual(score("<answer>2/(2+2)</answer>", 2)[0], 0)
        self.assertEqual(score("<answer>1（mile） or 24（miles）</answer>", 24)[0], 0)

    def test_supported_terminal_summary(self):
        text = ("Remaining doctors and nurses.\n- Total remaining = doctors + nurses = 6 + 16 = 22"
                "\nStep 4: Write the answer as <answer>\n<answer>22</answer>\n")
        suffix = ("This problem uses the concept of subtraction to find the difference between the remaining "
                  "doctors and nurses, and then combining both to find the overall number of remaining people.")
        self.assertEqual(score(text + suffix), (1, "supported_explanation"))
        for candidate in [
            text + "New problem: combine doctors and nurses.",
            text + "Add the nurses and doctors to get another answer.",
            text + "Subtract nurses and doctors to get twenty instead.",
            text + "We added all the ships and sailors.",
            text + "Adding doctors and nurses makes 99.",
            text.replace("6 + 16", "7 + 16") + suffix,
            text.replace("6 + 16", "2 * 6 + 16") + suffix,
        ]:
            with self.subTest(candidate=candidate):
                self.assertEqual(score(candidate)[0], 0)
        self.assertEqual(score(text + suffix, truncated=True)[0], 0)
