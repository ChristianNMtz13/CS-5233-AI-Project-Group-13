"""Readable Reversi AI using minimax and alpha-beta pruning on bitboards."""
import time

from bitboard import board_to_bitboards, legal_moves, play_move, square


# indexed by square, 8 * row + col
POSITION_VALUES = [
    100, -25, 10,  5,  5, 10, -25, 100,
    -25, -40, -5, -5, -5, -5, -40, -25,
     10,  -5,  8,  3,  3,  8,  -5,  10,
      5,  -5,  3,  2,  2,  3,  -5,   5,
      5,  -5,  3,  2,  2,  3,  -5,   5,
     10,  -5,  8,  3,  3,  8,  -5,  10,
    -25, -40, -5, -5, -5, -5, -40, -25,
    100, -25, 10,  5,  5, 10, -25, 100,
]

# every square of the same value in one mask, so a board is scored with a few bit counts
WEIGHT_MASKS = {
    value: sum(1 << i for i, v in enumerate(POSITION_VALUES) if v == value)
    for value in set(POSITION_VALUES)
}

CORNERS = square(0, 0) | square(0, 7) | square(7, 0) | square(7, 7)


class OutOfTime(Exception):
    """Stops the current search when the time limit is almost reached."""


def squares(moves):
    """Turn a move bitboard into a list of square numbers, 8 * row + col."""
    return [i for i in range(64) if moves >> i & 1]


def evaluate(own, opp):
    """Score a position from the point of view of the player owning `own`."""
    my_moves = legal_moves(own, opp).bit_count()
    opponent_moves = legal_moves(opp, own).bit_count()
    piece_difference = own.bit_count() - opp.bit_count()

    if not my_moves and not opponent_moves:
        if piece_difference > 0:
            return 100000 + piece_difference
        if piece_difference < 0:
            return -100000 + piece_difference
        return 0

    positional_score = sum(
        value * ((own & mask).bit_count() - (opp & mask).bit_count())
        for value, mask in WEIGHT_MASKS.items()
    )
    mobility_score = my_moves - opponent_moves
    corner_score = (own & CORNERS).bit_count() - (opp & CORNERS).bit_count()
    empty_squares = 64 - (own | opp).bit_count()
    piece_weight = 1 if empty_squares > 20 else 8

    return (
        positional_score * 3
        + mobility_score * 15
        + corner_score * 100
        + piece_difference * piece_weight
    )


def order_moves(own, opp, moves):
    """Search strong-looking moves first so alpha-beta can prune more."""
    def move_score(move):
        score = POSITION_VALUES[move]
        if (1 << move) & CORNERS:
            score += 1000

        # play_move returns the opponent's side first, so this counts their replies
        score -= 5 * legal_moves(*play_move(own, opp, 1 << move)).bit_count()
        return score

    return sorted(squares(moves), key=move_score, reverse=True)


def minimax(own, opp, depth, alpha, beta, deadline, passed=False):
    """Minimax search with alpha-beta pruning, in negamax form.

    `own` is always the player to move and scores are from their point of view,
    so the opponent's score is just ours negated and one branch covers both sides.
    """
    if time.perf_counter() >= deadline:
        raise OutOfTime

    moves = legal_moves(own, opp)

    if not moves:
        if passed:
            return evaluate(own, opp)
        return -minimax(opp, own, depth, -beta, -alpha, deadline, True)

    if depth == 0:
        return evaluate(own, opp)

    best_score = -float("inf")
    for move in order_moves(own, opp, moves):
        score = -minimax(
            *play_move(own, opp, 1 << move), depth - 1,
            -beta, -alpha, deadline
        )
        best_score = max(best_score, score)
        alpha = max(alpha, best_score)
        if alpha >= beta:
            break
    return best_score


def search_depth(own, opp, depth, deadline, previous_best):
    """Search every root move at one completed depth."""
    moves = order_moves(own, opp, legal_moves(own, opp))

    if previous_best in moves:
        moves.remove(previous_best)
        moves.insert(0, previous_best)

    best_move = moves[0]
    best_score = -float("inf")
    alpha = -float("inf")
    beta = float("inf")

    for move in moves:
        score = -minimax(
            *play_move(own, opp, 1 << move), depth - 1,
            -beta, -alpha, deadline
        )
        if score > best_score:
            best_score = score
            best_move = move
        alpha = max(alpha, best_score)

    return best_move


def choose_move(board, turn, time_limit):
    """Choose a legal move and return it as (row, column)."""
    own, opp = board_to_bitboards(board, turn)
    moves = squares(legal_moves(own, opp))

    if not moves:
        return (-1, -1)
    if len(moves) == 1:
        return divmod(moves[0], 8)

    best_move = max(moves, key=lambda move: POSITION_VALUES[move])

    limit = 4.49 if time_limit is None else min(4.49, max(0.05, time_limit - 0.5))
    deadline = time.perf_counter() + limit
    empty_squares = 64 - (own | opp).bit_count()

    # Search one move farther on every completed iteration. With ten or fewer
    # empty squares, the loop attempts to solve the exact ending.
    maximum_depth = empty_squares

    for depth in range(1, maximum_depth * 2):
        try:
            best_move = search_depth(own, opp, depth, deadline, best_move)
        except OutOfTime:
            break

    return divmod(best_move, 8)
