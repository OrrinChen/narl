import unittest

from experiments import gsm8k_composite_reward as reward
from experiments.gsm8k_step_reward import score_training_reward as simple_reward
from experiments.gsm8k_training_reward_v2 import score_training_reward as outcome_reward


def sample(text):
    return {'completion': text, 'truncated': False}


class VerifiedCompositeTest(unittest.TestCase):
    example = {'value': '77', 'rationale': '<<90*3/5=54>> <<3/5*90=54>> <<2+3=5>> <<2+3=5>>'}
    first = r'90 \times \frac{3}{5} = 54'

    def group(self, *texts):
        return reward.score_group_training_reward([sample(text) for text in texts], self.example)

    def test_exact_rational_canonicalization_and_operation_order(self):
        for alternatives in ((self.first, '60% × 90 = 54', '54 = .60 * 90'),
                             ('(2+4)*3=18', '3*(4+2)=18'),
                             ('1+2+3=6', '(3+1)+2=6')):
            expected = reward.relations(alternatives[0])
            self.assertTrue(expected)
            for text in alternatives[1:]:
                self.assertEqual(reward.relations(text), expected)
        self.assertFalse(reward.relations('(1+2)*2=6') & reward.relations('3+3=6'))
        self.assertFalse(reward.relations('3/5*90=54') & reward.relations('90*3/5=54'))

    def test_complete_numeric_equality_survives_chain_and_prose_boundaries(self):
        self.assertIn(next(iter(reward.relations('9+3=12'))), reward.relations('C + 3 = 9 + 3 = 12'))
        self.assertEqual(reward.relations('有10*3.5=35杯'), reward.relations('10*3.5=35'))

    def test_incorrect_and_unsupported_expressions(self):
        for text in ('1+2*3=6', 'z * 2 * 3=6', 'x+2*3=6', '2^2*3=6', '2*3=6^2',
                     '2**3=8', '1/0=0', '9999999999999999999999999999999*1=1', 'x=6'):
            with self.subTest(text=text):
                self.assertFalse(reward.relations(text))

    def test_scalar_scores_distinct_references_and_caps_partial_credit(self):
        score, _ = reward.score_training_reward(sample(self.first + '\n' + self.first), self.example)
        self.assertEqual(score, 0.4 / 3)
        full = sample(self.first + '\n90 * 3 / 5 = 54\n3+2=5\n<answer>0</answer>')
        self.assertEqual(reward.score_training_reward(full, self.example)[0], 0.4)
        tail = sample('<answer>0</answer>\n' + self.first)
        self.assertEqual(reward.score_training_reward(tail, self.example)[0], 0)
        other = {'rationale': '3/5*90=54 (unannotated)', 'value': '77'}
        self.assertEqual(reward.score_training_reward(sample(self.first), other)[0], 0)

    def test_group_breaks_only_tied_wrong_answers_with_strictly_nested_evidence(self):
        self.assertEqual([r for r, _ in self.group(self.first, 'No steps')], [0.4 / 3, 0.0])
        self.assertEqual(self.group('No steps', self.first), list(reversed(self.group(self.first, 'No steps'))))
        texts = (self.first, '2+3=5')
        self.assertEqual(self.group(*texts), [simple_reward(sample(t), self.example) for t in texts])
        unrelated = {'rationale': '<<3/5*90=54>> <<3*(4+2)=18>> <<(1+2)*2=6>>', 'value': '77'}
        two = sample(self.first + '\n3 * (4 + 2) = 18')
        one = sample('(1+2)*2=6')
        self.assertGreater(reward.score_training_reward(two, unrelated)[0],
                           reward.score_training_reward(one, unrelated)[0])
        result = reward.score_group_training_reward([two, one], unrelated)
        self.assertEqual(result, [simple_reward(s, unrelated) for s in (two, one)])

    def test_correct_v2_outcome_and_single_response_tail(self):
        correct = '<answer>７７</answer>'
        self.assertIs(reward.outcome_reward, outcome_reward)
        self.assertEqual(reward.score_training_reward(sample(correct), self.example),
                         outcome_reward(sample(correct), self.example['value']))
        self.assertEqual(self.group(correct, self.first),
                         [simple_reward(sample(t), self.example) for t in (correct, self.first)])
        self.assertEqual(self.group(self.first), [simple_reward(sample(self.first), self.example)])
        for lengths in (0, 3):
            with self.assertRaises(ValueError):
                self.group(*(['none'] * lengths))


if __name__ == '__main__':
    unittest.main()
