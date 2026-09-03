import unittest

import numpy as np

from hybrid_gcg.hotflip import (
    Rankings,
    build_radius1_panel,
    rank_probabilities,
    transformers_dtype_keyword,
)


class HotFlipTests(unittest.TestCase):
    def test_transformers_dtype_keyword_tracks_major_version(self) -> None:
        self.assertEqual(transformers_dtype_keyword("4.57.1"), "torch_dtype")
        self.assertEqual(transformers_dtype_keyword("5.0.0"), "dtype")
        self.assertEqual(transformers_dtype_keyword("unknown"), "torch_dtype")

    def test_power_rank_distribution_is_normalized_and_biased(self) -> None:
        probabilities = rank_probabilities(256, 0.1)
        self.assertTrue(np.isclose(probabilities.sum(), 1.0))
        self.assertGreater(probabilities[0], probabilities[1])
        self.assertGreater(probabilities[1], probabilities[-1])
        self.assertGreater(probabilities[:25].sum(), 0.6)

    def test_radius1_panel_is_unique_and_changes_one_coordinate(self) -> None:
        rankings = Rankings(
            positions=(0, 1),
            token_ids=np.array([[10, 11, 12], [20, 21, 22]]),
            predicted_deltas=np.array(
                [[-3.0, -2.0, -1.0], [-3.0, -2.0, -1.0]]
            ),
            loss=1.0,
            competitor_id=99,
        )
        panel = build_radius1_panel(
            [1, 2],
            rankings,
            budget=6,
            temperature=0.1,
            rng=np.random.default_rng(7),
            visited=set(),
        )
        self.assertEqual(len(panel), 6)
        self.assertEqual(
            len({(row.position, row.replacement_id) for row in panel}), 6
        )


if __name__ == "__main__":
    unittest.main()
