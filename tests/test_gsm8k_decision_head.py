from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from experiments.gsm8k_decision_head import encode_features, fit_head, predict_proba


class ToyEncoder:
    identity = {"checkpoint": "test"}
    readout_ids = [11]

    def last_hidden(self, sequences):
        return np.array([[sum(row), len(row)] for row in sequences])


class DecisionHeadTest(unittest.TestCase):
    def test_cache_binds_order_and_observables_ignores_labels(self):
        records = [
            {"prompt_ids": [2, 1], "token_ids": [3, 2], "has_answer_tag": False, "behavior_logprobs": [-2., -1.]},
            {"prompt_ids": [4], "token_ids": [4], "has_answer_tag": True, "behavior_logprobs": [-1.]},
        ]
        encoder = ToyEncoder()
        with TemporaryDirectory() as folder:
            cache = Path(folder) / "features.npz"
            x, measured = encode_features(encoder, records, 2, cache)
            np.testing.assert_allclose(x, [[19, 5, 0, -1.5], [19, 3, 1, -1]])
            self.assertFalse(measured["cache_hit"])
            self.assertEqual(len(measured["row_encoder_seconds"]), 2)
            y, cached = encode_features(encoder, [dict(r, reward=1) for r in records], 2, cache)
            np.testing.assert_array_equal(x, y)
            self.assertTrue(cached["cache_hit"])
            self.assertEqual(cached["encoder_seconds"], 0)
            self.assertEqual(cached["row_encoder_seconds"], measured["row_encoder_seconds"])
            with self.assertRaisesRegex(ValueError, "different inputs"):
                encode_features(encoder, records[::-1], 2, cache)
            with self.assertRaisesRegex(ValueError, "Malformed"):
                encode_features(encoder, [dict(records[0], behavior_logprobs=[])], 2)

    def test_fit_uses_only_tagged_training_rows_and_masks_impossible_outcomes(self):
        x = np.array([[-3, 77, 1, -1], [-1, 77, 1, -1], [3, 77, 1, -1], [1, 77, 1, -1], [50, -99, 0, -1]])
        labels = [0, 0, 1, 1, 0]
        head = fit_head(x, labels)
        self.assertEqual(head["n_train_tagged"], 4)
        self.assertEqual(head["mean"], [0, 77, -1])
        score = predict_proba(head, x)
        self.assertTrue(np.all(score[2:4] > score[:2]))
        self.assertEqual(score[-1], 0)
        with self.assertRaisesRegex(ValueError, "untagged"):
            fit_head(x, [0, 0, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
