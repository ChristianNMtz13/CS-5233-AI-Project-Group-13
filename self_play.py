# https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md
# https://arxiv.org/pdf/1902.10565

import signal
from dataclasses import dataclass, fields

import numpy as np

import bitboard
from mcts import (
    MCTS,
    PASS_ACTION,
    choose_action,
    run_in_lockstep,
    temperature_at,
    visit_policy
)
from reversi import reversi


# How random the moves are in training games.
MOVE_TEMP_EARLY = 0.75
MOVE_TEMP = 0.3

# On average the first 2.5 moves get played straight from the network with no search, so games start out slightly different.
POLICY_INIT_MOVES = 0.04 * 64
POLICY_INIT_TEMP = 2.0

# The net also tries to guess what the board will look like 2 and 8 moves from now. Good learning signal for long term planning.
FUTURE_TURNS = (2, 8)

# How far ahead the short term score guesses look. Roughly 12, 5 and 2 moves.
TD_FACTORS = tuple(1 / (1 + 64 * c) for c in (0.176, 0.056, 0.016))


# Everything we keep from one position to train on later.
@dataclass
class TrainingExample:
    state: np.ndarray
    policy: np.ndarray
    boards: np.ndarray
    margin: float
    td_margin: np.ndarray
    variance_time: float


# Turns a list of examples into columns and back. Columns are way faster to send around and save.
def stack_examples(examples: list[TrainingExample]) -> dict[str, np.ndarray]:
    return {
        field.name: np.array([getattr(example, field.name) for example in examples])
        for field in fields(TrainingExample)
    }


def unstack_examples(columns) -> list[TrainingExample]:
    return [
        TrainingExample(*row)
        for row in zip(*(columns[field.name] for field in fields(TrainingExample)))
    ]

def initial_position() -> tuple[int, int]:
    return bitboard.board_to_bitboards(reversi().board, 1)


# Sets up a bunch of games to play at once.
def self_games(games: int, **search) -> list:
    searcher = MCTS()
    return [self_game_steps(searcher, **search) for _ in range(games)]


# Runs in a worker process. Plays its share of the games and sends positions back to the main process whenever it needs the network.
def play_as_worker(connection, games: int, seed: int, search: dict) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    np.random.seed(seed)

    def evaluate(positions: list) -> list:
        connection.send_bytes(bitboard.pack_states(positions))
        return bitboard.split_replies(np.frombuffer(connection.recv_bytes(), dtype=np.float32))

    results = run_in_lockstep(evaluate, self_games(games, **search))
    connection.send_bytes(b"")
    connection.send([(stack_examples(examples), white, black)for examples, white, black in results])


# Plays one whole game against itself.
def self_game_steps(
    searcher: MCTS,
    simulations: int,
    full_search_fraction: float = 1.0,
    fast_simulations: int = 0
):
    own, opponent = initial_position()

    side_to_move = 1

    # a few opening moves straight from the raw policy, for game variety
    for _ in range(int(np.random.exponential(POLICY_INIT_MOVES))):
        moves = bitboard.legal_moves(own, opponent)

        if moves == 0:
            break

        logits, _ = yield own, opponent
        actions = bitboard.squares(moves)
        action = actions[np.random.choice(len(actions), p=searcher.priors(logits, actions, POLICY_INIT_TEMP, True))]
        own, opponent = searcher.next_position(own, opponent, action)
        side_to_move = -side_to_move

    turns = []

    tree = None

    while True:
        moves = bitboard.legal_moves(own, opponent)
        policy = margin = None
        full = False

        if moves == 0:
            if bitboard.legal_moves(opponent, own) == 0:
                break
            action = PASS_ACTION
            
        elif moves.bit_count() == 1:
            action = moves.bit_length() - 1
            policy = np.zeros(64, dtype=np.float32)
            policy[action] = 1.0
        else:
            # playout cap randomization
            full = np.random.random() < full_search_fraction

            tree = yield from searcher.search_steps(own, opponent, simulations=simulations if full else fast_simulations, add_root_noise=True, root=tree if full else None)

            policy = visit_policy(tree)
            margin = side_to_move * 64 * tree.mean_value

            action = choose_action(tree, temperature_at(own, opponent, MOVE_TEMP_EARLY, MOVE_TEMP))

        board = (own, opponent) if side_to_move == 1 else (opponent, own)
        turns.append((board, side_to_move, policy, margin, full))
        tree = tree.children.get(action) if tree else None
        own, opponent = searcher.next_position(own, opponent, action)
        side_to_move = -side_to_move

    final = (own, opponent) if side_to_move == 1 else (opponent, own)

    return label_examples(turns, final), final[0].bit_count(), final[1].bit_count()


# Goes back over the finished game and works out what the network should learn from each position.
def label_examples(
    turns: list,
    final: tuple[int, int]
) -> list[TrainingExample]:
    end = len(turns)
    final_margin = final[0].bit_count() - final[1].bit_count()

    bits = bitboard.bit_planes([board for board, *_ in turns] + [final]).astype(np.int8)
    discs = bits[:, 0] - bits[:, 1]

    values = [final_margin]

    for _, _, _, margin, _ in reversed(turns):
        values.append(values[-1] if margin is None else margin)

    values = np.array(values[::-1])

    # short-term targets: exponentially discounted future margins
    def discounted(ahead: np.ndarray, factor: float) -> float:
        weights = factor * (1 - factor) ** np.arange(len(ahead))
        weights[-1] = (1 - factor) ** (len(ahead) - 1)
        return float(weights @ ahead)

    examples = []

    for index, (board, side, policy, _, full) in enumerate(turns):
        # only full searches get trained on, this is for playout cap randomization.
        if not full:
            continue

        ahead = values[index:]
        reply = turns[index + 1][2] if index + 1 < end else None
        own, opponent = board if side == 1 else board[::-1]

        examples.append(TrainingExample(
            state=bitboard.state_array(own, opponent),
            # opponent's reply as a policy target
            policy=np.stack((policy, np.zeros_like(policy) if reply is None else reply)),
            # final board for the ownership target, KataGo found this helps speed up training.
            boards=side * discs[[end] + [min(index + k, end) for k in FUTURE_TURNS]],
            margin=float(side * final_margin),
            td_margin=side * np.array([discounted(ahead, f) for f in TD_FACTORS]),
            variance_time=float(np.arange(1, len(ahead)) @ np.diff(ahead / 64) ** 2)
        ))

    return examples
