"""Self-play, training data, and checkpoint tools for Reversi."""

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import AI_logic as game
from mcts import MCTS, choose_action, visit_policy


@dataclass
class TrainingExample:
    """One board position and its training targets."""

    state: np.ndarray
    policy: np.ndarray
    value: float


class ReplayBuffer:
    """Store examples from recent self-play games."""

    def __init__(self, maximum_size: int = 50000) -> None:
        self.examples = deque(maxlen=maximum_size)

    def add_game(
        self,
        examples: list[TrainingExample]
    ) -> None:
        self.examples.extend(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def sample(
        self,
        amount: int
    ) -> list[TrainingExample]:
        amount = min(amount, len(self.examples))

        return random.sample(
            list(self.examples),
            amount
        )


def state_array(
    own: int,
    opponent: int
) -> np.ndarray:
    """Create the two input layers expected by the network."""
    planes = np.zeros(
        (2, 8, 8),
        dtype=np.float32
    )

    for action in range(64):
        row, column = divmod(action, 8)

        planes[0, row, column] = (
            own >> action
        ) & 1

        planes[1, row, column] = (
            opponent >> action
        ) & 1

    return planes


def initial_position() -> tuple[int, int]:
    """Return the opening position as white and black."""
    white = (
        game.square(3, 3)
        | game.square(4, 4)
    )

    black = (
        game.square(3, 4)
        | game.square(4, 3)
    )

    return white, black


def model_evaluator(model: nn.Module):
    """Create the network-evaluation function used by MCTS."""

    def evaluate(
        own: int,
        opponent: int
    ) -> tuple[np.ndarray, float]:
        model.eval()

        with torch.no_grad():
            policy_logits, value = model(
                game.to_tensor(own, opponent)
            )

        logits = (
            policy_logits[0]
            .detach()
            .cpu()
            .numpy()
        )

        return logits, float(value.item())

    return evaluate


def augment_example(
    example: TrainingExample
) -> list[TrainingExample]:
    """Create eight rotations and reflections."""
    policy_board = example.policy.reshape(8, 8)
    results = []

    for turns in range(4):
        rotated_state = np.rot90(
            example.state,
            turns,
            axes=(1, 2)
        ).copy()

        rotated_policy = np.rot90(
            policy_board,
            turns
        ).copy()

        results.append(
            TrainingExample(
                rotated_state,
                rotated_policy.reshape(64),
                example.value
            )
        )

        reflected_state = np.flip(
            rotated_state,
            axis=2
        ).copy()

        reflected_policy = np.fliplr(
            rotated_policy
        ).copy()

        results.append(
            TrainingExample(
                reflected_state,
                reflected_policy.reshape(64),
                example.value
            )
        )

    return results


def play_self_game(
    model: nn.Module,
    simulations: int = 50,
    exploration_moves: int = 12,
    add_symmetries: bool = True
) -> tuple[list[TrainingExample], int, int]:
    """Play one game and return its training examples."""
    own, opponent = initial_position()

    # White is 1 and moves first for this project.
    side_to_move = 1
    move_number = 0
    history = []

    evaluator = model_evaluator(model)

    searcher = MCTS(
        game.legal_moves,
        game.play_move,
        evaluator
    )

    while True:
        moves = game.legal_moves(own, opponent)

        if moves == 0:
            opponent_moves = game.legal_moves(
                opponent,
                own
            )

            if opponent_moves == 0:
                break

            # Current player passes.
            own, opponent = opponent, own
            side_to_move = -side_to_move
            continue

        root = searcher.search(
            own,
            opponent,
            simulations=simulations,
            add_root_noise=True
        )

        policy = visit_policy(
            root,
            temperature=1.0
        )

        history.append(
            (
                state_array(own, opponent),
                policy,
                side_to_move
            )
        )

        if move_number < exploration_moves:
            temperature = 1.0
        else:
            temperature = 0.0

        action = choose_action(
            root,
            temperature=temperature
        )

        own, opponent = game.play_move(
            own,
            opponent,
            1 << action
        )

        side_to_move = -side_to_move
        move_number += 1

    own_count = own.bit_count()
    opponent_count = opponent.bit_count()

    if own_count > opponent_count:
        winner = side_to_move

    elif opponent_count > own_count:
        winner = -side_to_move

    else:
        winner = 0

    labeled_examples = []

    for state, policy, recorded_side in history:
        if winner == 0:
            value = 0.0

        elif recorded_side == winner:
            value = 1.0

        else:
            value = -1.0

        example = TrainingExample(
            state,
            policy,
            value
        )

        if add_symmetries:
            labeled_examples.extend(
                augment_example(example)
            )
        else:
            labeled_examples.append(example)

    # Convert the counts back into white, black order.
    if side_to_move == 1:
        white_count = own_count
        black_count = opponent_count
    else:
        white_count = opponent_count
        black_count = own_count

    return (
        labeled_examples,
        white_count,
        black_count
    )


def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    examples: list[TrainingExample],
    epochs: int = 2,
    batch_size: int = 128
) -> dict[str, float]:
    """Train the policy and value outputs."""
    if not examples:
        raise ValueError(
            "Cannot train without self-play examples."
        )

    states = torch.tensor(
        np.stack([
            example.state
            for example in examples
        ])
    )

    target_policies = torch.tensor(
        np.stack([
            example.policy
            for example in examples
        ])
    )

    target_values = torch.tensor(
        [
            example.value
            for example in examples
        ],
        dtype=torch.float32
    )

    dataset = TensorDataset(
        states,
        target_policies,
        target_values
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    model.train()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    batches = 0

    for _ in range(epochs):
        for (
            state_batch,
            policy_batch,
            value_batch
        ) in loader:
            state_batch = state_batch.to(game.device)
            policy_batch = policy_batch.to(game.device)
            value_batch = value_batch.to(game.device)

            optimizer.zero_grad()

            logits, predicted_values = model(
                state_batch
            )

            policy_loss = -(
                policy_batch
                * functional.log_softmax(
                    logits,
                    dim=1
                )
            ).sum(dim=1).mean()

            value_loss = functional.mse_loss(
                predicted_values.squeeze(1),
                value_batch
            )

            loss = policy_loss + value_loss

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_policy_loss += float(
                policy_loss.item()
            )
            total_value_loss += float(
                value_loss.item()
            )

            batches += 1

    return {
        "loss": total_loss / batches,
        "policy_loss": (
            total_policy_loss / batches
        ),
        "value_loss": (
            total_value_loss / batches
        )
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_games: int
) -> None:
    """Save the model and optimizer progress."""
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "completed_games": completed_games
        },
        Path(path)
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None
) -> int:
    """Load a saved model and return its game count."""
    checkpoint = torch.load(
        Path(path),
        map_location=game.device
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    if (
        optimizer is not None
        and "optimizer_state" in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )

    return int(
        checkpoint.get("completed_games", 0)
    )