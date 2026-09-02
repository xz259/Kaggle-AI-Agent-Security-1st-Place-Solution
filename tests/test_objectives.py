import unittest

import numpy as np

from hybrid_gcg.objectives import SequenceScore, score_logits


class ObjectiveTests(unittest.TestCase):
    def test_score_logits_exact_margin_and_nll(self) -> None:
        score = score_logits(np.array([1.0, 4.0, 2.0]), 1)
        self.assertTrue(score.exact)
        self.assertEqual(score.greedy_id, 1)
        self.assertEqual(score.competitor_id, 2)
        self.assertEqual(score.margin, 2.0)
        self.assertGreater(score.nll, 0.0)

    def test_sequence_uses_weakest_margin(self) -> None:
        first = score_logits([0.0, 3.0, 1.0], 1)
        second = score_logits([2.0, 1.5, 0.0], 1)
        sequence = SequenceScore((first, second))
        self.assertFalse(sequence.exact)
        self.assertEqual(sequence.min_margin, -0.5)

    def test_non_finite_logits_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            score_logits([0.0, np.nan], 0)


if __name__ == "__main__":
    unittest.main()
