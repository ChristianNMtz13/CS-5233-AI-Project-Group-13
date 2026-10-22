"""Use the trained network and MCTS to choose Reversi moves."""

import time
from pathlib import Path

import numpy as np
import torch

import AI_logic as game
from mcts import MCTS, choose_action


CHECKPOINT = Path(__file__).with_name(
    "reversi_model.pt"
)

SAFETY_MARGIN = 0.5
MAX_SIMULATIONS = 100000


# Keep the agent single-threaded.
torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def load_weights() -> None:
    """Load trained weights if the checkpoint exists."""
    if not CHECKPOINT.exists():
        return

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=game.device
    )

    game.net.load_state_dict(
        checkpoint["model_state"]
    )


load_weights()


def evaluate_position(
    own: int,
    opponent: int
) -> tuple[np.ndarray, float]:
    """Ask the network for policy logits and a value."""
    game.net.eval()

    with torch.no_grad():
        logits, value = game.net(
            game.to_tensor(own, opponent)
        )

    policy_logits = (
        logits[0]
        .detach()
        .cpu()
        .numpy()
    )

    return policy_logits, float(value.item())


def choose_move(
    board,
    turn: int,
    time_limit: float | None
) -> tuple[int, int]:
    """Select a legal move using timed MCTS."""
    own, opponent = game.board_to_bitboards(
        board,
        turn
    )

    moves = game.legal_moves(
        own,
        opponent
    )

    if moves == 0:
        return -1, -1

    # Keep a legal fallback in case time runs out quickly.
    fallback = next(
        action
        for action in range(64)
        if moves & (1 << action)
    )

    if time_limit is None:
        available_time = game.DEFAULT_TIME_LIMIT
    else:
        available_time = time_limit

    search_time = max(
        0.02,
        available_time - SAFETY_MARGIN
    )

    deadline = (
        time.perf_counter()
        + search_time
    )

    searcher = MCTS(
        game.legal_moves,
        game.play_move,
        evaluate_position
    )

    root = searcher.search(
        own,
        opponent,
        simulations=MAX_SIMULATIONS,
        deadline=deadline
    )

    action = choose_action(
        root,
        temperature=0.0
    )

    if (
        action < 0
        or action >= 64
        or not moves & (1 << action)
    ):
        action = fallback

    return divmod(action, 8)