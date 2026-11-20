"""Keep answer-only diagnostics separate from outputs containing reasoning."""

from importlib.util import find_spec
from unittest import TestCase, main, skipUnless


@skipUnless(all(find_spec(package) for package in ("mlx", "mlx_lm", "peft")),
            "MLX experiment dependencies are optional")
class GrpoDiagnosisTest(TestCase):
    def test_pure_answer_requires_one_tagged_answer_and_only_surrounding_whitespace(self):
        from experiments.gsm8k_grpo_diagnosis import output_metrics

        cases = (
            (" \n<answer>42</answer>\t", 1),
            ("<answer>7</answer>\n3 + 4 = 7.\n<answer>7</answer>", 0),
            ("<answer>42</answer> extra words", 0),
            ("3 + 4 = 7.\n<answer>7</answer>", 0),
        )
        for completion, expected in cases:
            with self.subTest(completion=completion):
                evaluation = {"correct": 0, "truncated": 0, "tagged": 1,
                              "rows": [{"completion": completion, "token_ids": [1]}]}
                self.assertEqual(output_metrics(evaluation)["pure_answer"], expected)


if __name__ == "__main__":
    main()
