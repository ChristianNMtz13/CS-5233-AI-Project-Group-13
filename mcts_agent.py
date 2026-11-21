import functools
import sys
import time
from pathlib import Path

import torch

import bitboard
import network
from mcts import MCTS, choose_action


# Uses the trained model.
CHECKPOINT = globals().get("CHECKPOINT") or Path(__file__).parent / "checkpoints" / "reversi_model.pt"

# 5 seconds a move, and save .5 for network latency.
DEFAULT_TIME_LIMIT = 5.0
SAFETY_MARGIN = 0.5
# Basically no cap. The clock stops the search.
MAX_SIMULATIONS = 100000

# Leave this as None for the tournament. A number here makes it search a fixed amount instead of using the clock, only good for testing.
SIMULATIONS = None

# Turns on the extra stuff we only use when playing for real, not in training.
PLAY = True


# Single thread for tournament.
network.single_thread()


# Loads the trained model.
def load_model() -> network.PolicyValueNet:
    if not CHECKPOINT.exists():
        print(f"WARNING: {CHECKPOINT} not found; playing with an UNTRAINED network.",file=sys.stderr)
        return network.PolicyValueNet()

    checkpoint = torch.load(CHECKPOINT, map_location="cpu")

    return network.from_state_dict(checkpoint.get("average_state", checkpoint["model_state"]))


model = load_model()

last_simulations = 0

# We save search tree between moves so we don't throw away work we already did.
tree = None
tree_position = (0, 0)

# Unlike the transposition table we used for the traditional bot I decided to just do the functools cache, which is 1000x easier.
# Implementing a transposition table for MCTS was starting to confuse me. Got lost in the sauce and gave up.
@functools.lru_cache(maxsize=100000) 
def evaluate_position(
    own: int,
    opponent: int
) -> tuple:
    return network.evaluate_batch(
        model,
        [(own, opponent)],
        PLAY
    )[0]

# solver=True means once the search can see all the way to the end of the game, it locks in the exact result instead of guessing.
searcher = MCTS(evaluate_position, solver=True, play=PLAY)

def choose_move(
    board,
    turn: int,
    time_limit: float | None
) -> tuple[int, int]:
    global last_simulations, tree, tree_position
    last_simulations = 0

    own, opponent = bitboard.board_to_bitboards(board, turn)

    if tree is not None:
        tree = searcher.find_subtree(tree, *tree_position, own, opponent)

    tree_position = (own, opponent)

    moves = bitboard.legal_moves(own, opponent)

    if moves == 0:
        return -1, -1

    if moves.bit_count() == 1:
        return divmod(moves.bit_length() - 1, 8)

    if SIMULATIONS:
        simulations, deadline = SIMULATIONS, None
    else:
        limit = DEFAULT_TIME_LIMIT if time_limit is None else time_limit
        simulations = MAX_SIMULATIONS
        deadline = time.perf_counter() + max(0.02, limit - SAFETY_MARGIN)

    root = searcher.search(own, opponent, simulations=simulations, deadline=deadline, root=tree)

    tree = root
    last_simulations = root.visit_count

    return divmod(choose_action(root), 8)
