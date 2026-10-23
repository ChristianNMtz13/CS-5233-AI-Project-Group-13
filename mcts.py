"""Single-threaded Monte Carlo Tree Search for Reversi."""

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# The neural network has 64 outputs. Passing is handled inside MCTS.
PASS_ACTION = 64

PolicyValueFunction = Callable[
    [int, int],
    tuple[np.ndarray, float]
]

LegalMovesFunction = Callable[[int, int], int]

PlayMoveFunction = Callable[
    [int, int, int],
    tuple[int, int]
]


@dataclass
class Node:
    """One position in the search tree."""

    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)
    expanded: bool = False
    terminal_value: float | None = None

    @property
    def mean_value(self) -> float:
        """Return this position's average search value."""
        if self.visit_count == 0:
            return 0.0

        return self.value_sum / self.visit_count


class MCTS:
    """AlphaZero-style MCTS using policy and value predictions."""

    def __init__(
        self,
        legal_moves: LegalMovesFunction,
        play_move: PlayMoveFunction,
        policy_value: PolicyValueFunction,
        c_puct: float = 1.5
    ) -> None:
        self.legal_moves = legal_moves
        self.play_move = play_move
        self.policy_value = policy_value
        self.c_puct = c_puct

    @staticmethod
    def actions_from_bits(move_bits: int) -> list[int]:
        """Convert a legal-move bitboard into action numbers."""
        return [
            action
            for action in range(64)
            if move_bits & (1 << action)
        ]

    @staticmethod
    def terminal_score(own: int, opponent: int) -> float:
        """Return the final result from the current player's view."""
        difference = own.bit_count() - opponent.bit_count()

        if difference > 0:
            return 1.0

        if difference < 0:
            return -1.0

        return 0.0

    def expand(self, node: Node, own: int, opponent: int) -> float:
        """Evaluate a new position and create its children."""
        moves = self.legal_moves(own, opponent)

        if moves == 0:
            opponent_moves = self.legal_moves(opponent, own)

            if opponent_moves == 0:
                node.expanded = True
                node.terminal_value = self.terminal_score(
                    own,
                    opponent
                )
                return node.terminal_value

            # The current player must pass.
            node.children[PASS_ACTION] = Node(prior=1.0)
            node.expanded = True

            _, value = self.policy_value(own, opponent)
            return float(value)

        logits, value = self.policy_value(own, opponent)
        actions = self.actions_from_bits(moves)

        legal_logits = np.asarray(
            [logits[action] for action in actions],
            dtype=np.float64
        )

        # Stable softmax over legal moves only.
        legal_logits -= np.max(legal_logits)
        probabilities = np.exp(legal_logits)
        total = float(probabilities.sum())

        if not np.isfinite(total) or total <= 0.0:
            probabilities = np.full(
                len(actions),
                1.0 / len(actions)
            )
        else:
            probabilities /= total

        for action, probability in zip(actions, probabilities):
            node.children[action] = Node(
                prior=float(probability)
            )

        node.expanded = True
        return float(value)

    def add_root_noise(
        self,
        root: Node,
        alpha: float = 0.3,
        fraction: float = 0.25
    ) -> None:
        """Add exploration noise during self-play training."""
        actions = [
            action
            for action in root.children
            if action != PASS_ACTION
        ]

        if not actions:
            return

        noise = np.random.dirichlet(
            [alpha] * len(actions)
        )

        for action, random_prior in zip(actions, noise):
            child = root.children[action]

            child.prior = (
                (1.0 - fraction) * child.prior
                + fraction * float(random_prior)
            )

    def select_child(self, node: Node) -> tuple[int, Node]:
        """Select a child using the PUCT formula."""
        parent_scale = math.sqrt(
            max(1, node.visit_count)
        )

        def score(item: tuple[int, Node]) -> float:
            _, child = item

            # The child belongs to the other player's perspective.
            exploitation = -child.mean_value

            exploration = (
                self.c_puct
                * child.prior
                * parent_scale
                / (1 + child.visit_count)
            )

            return exploitation + exploration

        return max(node.children.items(), key=score)

    def next_position(
        self,
        own: int,
        opponent: int,
        action: int
    ) -> tuple[int, int]:
        """Return the position after one move or pass."""
        if action == PASS_ACTION:
            return opponent, own

        move = 1 << action
        return self.play_move(own, opponent, move)

    @staticmethod
    def backpropagate(
        path: list[Node],
        value: float
    ) -> None:
        """Send the leaf value back through the search path."""
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value

            # Each level switches to the other player's view.
            value = -value

    def search(
        self,
        own: int,
        opponent: int,
        simulations: int = 100,
        deadline: float | None = None,
        add_root_noise: bool = False
    ) -> Node:
        """Build and return a tree for the supplied position."""
        root = Node()
        self.expand(root, own, opponent)

        if add_root_noise:
            self.add_root_noise(root)

        for _ in range(simulations):
            if (
                deadline is not None
                and time.perf_counter() >= deadline
            ):
                break

            if root.terminal_value is not None:
                break

            node = root
            current_own = own
            current_opponent = opponent
            path = [node]

            while node.expanded and node.children:
                action, node = self.select_child(node)

                current_own, current_opponent = (
                    self.next_position(
                        current_own,
                        current_opponent,
                        action
                    )
                )

                path.append(node)

                if not node.expanded:
                    break

            if node.terminal_value is not None:
                leaf_value = node.terminal_value

            elif not node.expanded:
                leaf_value = self.expand(
                    node,
                    current_own,
                    current_opponent
                )

            else:
                leaf_value = self.terminal_score(
                    current_own,
                    current_opponent
                )

            self.backpropagate(path, leaf_value)

        return root


def visit_policy(
    root: Node,
    temperature: float = 1.0
) -> np.ndarray:
    """Convert root visits into a 64-move probability array."""
    policy = np.zeros(64, dtype=np.float32)

    legal_children = {
        action: child
        for action, child in root.children.items()
        if action != PASS_ACTION
    }

    if not legal_children:
        return policy

    actions = list(legal_children)

    visits = np.asarray(
        [
            legal_children[action].visit_count
            for action in actions
        ],
        dtype=np.float64
    )

    if temperature <= 0.0:
        best_index = int(np.argmax(visits))
        policy[actions[best_index]] = 1.0
        return policy

    adjusted = np.power(
        visits,
        1.0 / temperature
    )

    if adjusted.sum() <= 0.0:
        adjusted = np.asarray(
            [
                legal_children[action].prior
                for action in actions
            ],
            dtype=np.float64
        )

    adjusted /= adjusted.sum()

    for action, probability in zip(actions, adjusted):
        policy[action] = float(probability)

    return policy


def choose_action(
    root: Node,
    temperature: float = 0.0
) -> int:
    """Select one action from a completed search."""
    policy = visit_policy(root, temperature)

    if policy.sum() <= 0.0:
        return PASS_ACTION

    if temperature <= 0.0:
        return int(np.argmax(policy))

    return int(np.random.choice(64, p=policy))