"""Tests for the policy-value network and MCTS."""

import unittest

import numpy as np
import torch

import AI_logic as game
from mcts import (
    MCTS,
    choose_action,
    visit_policy
)


class AlphaZeroTests(unittest.TestCase):

    def setUp(self) -> None:
        # White pieces in the opening position.
        self.own = (
            game.square(3, 3)
            | game.square(4, 4)
        )

        # Black pieces in the opening position.
        self.opponent = (
            game.square(3, 4)
            | game.square(4, 3)
        )

    def test_network_shapes(self) -> None:
        """The network should return 64 logits and one value."""
        with torch.no_grad():
            policy, value = game.net(
                game.to_tensor(
                    self.own,
                    self.opponent
                )
            )

        self.assertEqual(
            tuple(policy.shape),
            (1, 64)
        )

        self.assertEqual(
            tuple(value.shape),
            (1, 1)
        )

        self.assertGreaterEqual(
            float(value.item()),
            -1.0
        )

        self.assertLessEqual(
            float(value.item()),
            1.0
        )

    def test_mcts_returns_legal_move(self) -> None:
        """MCTS should only select an opening legal move."""

        def neutral_evaluator(
            own: int,
            opponent: int
        ) -> tuple[np.ndarray, float]:
            logits = np.zeros(
                64,
                dtype=np.float32
            )

            return logits, 0.0

        searcher = MCTS(
            game.legal_moves,
            game.play_move,
            neutral_evaluator
        )

        root = searcher.search(
            self.own,
            self.opponent,
            simulations=20
        )

        policy = visit_policy(root)
        action = choose_action(root)

        self.assertAlmostEqual(
            float(policy.sum()),
            1.0,
            places=5
        )

        legal_moves = game.legal_moves(
            self.own,
            self.opponent
        )

        self.assertTrue(
            legal_moves & (1 << action)
        )


if __name__ == "__main__":
    unittest.main()