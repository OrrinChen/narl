from collections import Counter
import unittest

from experiments.kv_arithmetic_task import ArithmeticTask, DevelopmentArithmeticTask


class ArithmeticTaskTests(unittest.TestCase):
    def test_unordered_pair_splits_are_disjoint_and_balanced(self):
        final = ArithmeticTask(seed=0)
        development = DevelopmentArithmeticTask(seed=130)

        def pairs(rows):
            return {tuple(sorted((row["left"], row["right"]))) for row in rows}

        self.assertEqual(pairs(final.data), pairs(development.data))
        for first, second in ((final.data, final.eval_data), (final.data, development.eval_data),
                              (final.eval_data, development.eval_data)):
            self.assertFalse(pairs(first) & pairs(second))
        self.assertEqual(final.eval_data, ArithmeticTask(seed=1).eval_data)
        for rows in (final.data, final.eval_data, development.eval_data):
            self.assertEqual(len(pairs(rows)), len(rows))
            self.assertEqual(Counter(row["bucket"] for row in rows),
                             {"carry": len(rows) // 2, "noncarry": len(rows) // 2})
            for row in rows:
                self.assertEqual(len(row["answer_tokens"]), row["completion_length"])
                self.assertEqual(int("".join(reversed(row["answer_tokens"]))), row["left"] + row["right"])


if __name__ == "__main__":
    unittest.main()
