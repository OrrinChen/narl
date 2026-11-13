from collections import Counter
import unittest

from experiments.kv_sequence_task import DevelopmentSequenceTask, SequenceTask


class SequenceTaskTests(unittest.TestCase):
    def test_balanced_phase_and_disjoint_contexts(self):
        task = SequenceTask(seed=0)
        dev = DevelopmentSequenceTask(seed=120)
        for key in ("prompt", "filler", "distractor"):
            training = {row[key] for row in task.data}
            development = {row[key] for row in dev.eval_data}
            heldout = {row[key] for row in task.eval_data}
            self.assertFalse(training & development)
            self.assertFalse(training & heldout)
            self.assertFalse(development & heldout)
        self.assertEqual(task.eval_data, SequenceTask(seed=1).eval_data)

        for split, counts in [(task.data, {3: 160, 12: 80, 16: 80}),
                              (task.eval_data, {3: 32, 12: 16, 16: 16})]:
            self.assertEqual(Counter(row["completion_length"] for row in split), counts)
            for length, count in counts.items():
                rows = [row for row in split if row["completion_length"] == length]
                self.assertEqual(Counter(row["answer_tokens"][0] for row in rows),
                                 {"0": count // 2, "1": count // 2})
                for row in rows:
                    answer = row["answer_tokens"]
                    self.assertEqual(len(answer), length)
                    self.assertTrue(all(left != right for left, right in zip(answer, answer[1:])))


if __name__ == "__main__":
    unittest.main()
