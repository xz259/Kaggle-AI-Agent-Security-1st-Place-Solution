import unittest

import numpy as np

from hybrid_gcg.hotflip import Rankings, prompt_hash
from hybrid_gcg.ridge import build_ridge_panel


class RidgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.current = (1, 2)
        self.rankings = Rankings(
            positions=(0, 1),
            token_ids=np.array([[10, 11, 12], [20, 21, 22]]),
            predicted_deltas=np.array(
                [[-3.0, -2.0, -1.0], [-2.5, -1.5, -0.5]]
            ),
            loss=1.0,
            competitor_id=99,
        )

    def test_falls_back_until_enough_exact_observations_exist(self) -> None:
        panel, diagnostics = build_ridge_panel(
            self.current,
            self.rankings,
            budget=3,
            temperature=0.1,
            rng=np.random.default_rng(7),
            visited=set(),
            observations=[],
            exploration_fraction=0.25,
            regularization=0.01,
            minimum_observations=4,
        )
        self.assertEqual(len(panel), 3)
        self.assertEqual(
            diagnostics["strategy"],
            "gradient_fallback_insufficient_gguf_observations",
        )
        self.assertTrue(all(row.source == "gradient_fallback" for row in panel))

    def test_calibrated_panel_interleaves_loss_and_margin_predictions(self) -> None:
        observations = [
            {
                "position": 0,
                "replacement_id": 10,
                "gradient_rank": 0,
                "predicted_delta": -3.0,
                "loss_delta": -1.0,
                "margin_gain": 0.5,
            },
            {
                "position": 0,
                "replacement_id": 11,
                "gradient_rank": 1,
                "predicted_delta": -2.0,
                "loss_delta": -0.6,
                "margin_gain": 0.3,
            },
            {
                "position": 1,
                "replacement_id": 20,
                "gradient_rank": 0,
                "predicted_delta": -2.5,
                "loss_delta": -0.8,
                "margin_gain": 0.4,
            },
            {
                "position": 1,
                "replacement_id": 21,
                "gradient_rank": 1,
                "predicted_delta": -1.5,
                "loss_delta": -0.4,
                "margin_gain": 0.2,
            },
        ]
        visited: set[str] = set()
        for row in observations:
            candidate = list(self.current)
            candidate[int(row["position"])] = int(row["replacement_id"])
            visited.add(prompt_hash(candidate))

        panel, diagnostics = build_ridge_panel(
            self.current,
            self.rankings,
            budget=2,
            temperature=0.1,
            rng=np.random.default_rng(11),
            visited=visited,
            observations=observations,
            exploration_fraction=0.0,
            regularization=0.01,
            minimum_observations=4,
            cover_coordinates=False,
        )
        self.assertEqual(len(panel), 2)
        self.assertEqual(diagnostics["strategy"], "gguf_ridge_calibrated")
        self.assertEqual(diagnostics["observation_count"], 4)
        self.assertEqual(
            {(row.position, row.replacement_id) for row in panel},
            {(0, 12), (1, 22)},
        )
        self.assertTrue(
            all(row.source in {"ridge_loss", "ridge_margin"} for row in panel)
        )
        self.assertIn("train_rmse", diagnostics["loss_fit"])
        self.assertIn("train_rmse", diagnostics["margin_fit"])


if __name__ == "__main__":
    unittest.main()
