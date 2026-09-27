"""Readable Reversi AI using minimax and alpha-beta pruning."""

import time
import numpy as np
from reversi import reversi


POSITION_VALUES = np.array([
    [100, -25, 10,  5,  5, 10, -25, 100],
    [-25, -40, -5, -5, -5, -5, -40, -25],
    [ 10,  -5,  8,  3,  3,  8,  -5,  10],
    [  5,  -5,  3,  2,  2,  3,  -5,   5],
    [  5,  -5,  3,  2,  2,  3,  -5,   5],
    [ 10,  -5,  8,  3,  3,  8,  -5,  10],
    [-25, -40, -5, -5, -5, -5, -40, -25],
    [100, -25, 10,  5,  5, 10, -25, 100]
])

CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]


class OutOfTime(Exception):
    """Stops the current search when the time limit is almost reached."""


def make_game(board):
    game = reversi()
    game.board = board.copy()
    return game


def play_move(board, move, player):
    """Return a new board after player makes move."""
    game = make_game(board)
    game.step(move[0], move[1], player)
    return game.board


def game_finished(board):
    game = make_game(board)
    return not game.legal_moves(1) and not game.legal_moves(-1)


def evaluate(board, ai_player):
    """Score a board from the AI's point of view."""
    opponent = -ai_player

    if game_finished(board):
        difference = int(np.count_nonzero(board == ai_player)) - int(
            np.count_nonzero(board == opponent)
        )
        if difference > 0:
            return 100000 + difference
        if difference < 0:
            return -100000 + difference
        return 0

    positional_score = int(np.sum(POSITION_VALUES[board == ai_player]))
    positional_score -= int(np.sum(POSITION_VALUES[board == opponent]))

    game = make_game(board)
    my_moves = len(game.legal_moves(ai_player))
    opponent_moves = len(game.legal_moves(opponent))
    mobility_score = my_moves - opponent_moves

    corner_score = 0
    for row, col in CORNERS:
        if board[row, col] == ai_player:
            corner_score += 1
        elif board[row, col] == opponent:
            corner_score -= 1

    piece_difference = int(np.count_nonzero(board == ai_player))
    piece_difference -= int(np.count_nonzero(board == opponent))
    empty_squares = int(np.count_nonzero(board == 0))
    piece_weight = 1 if empty_squares > 20 else 8

    return (
        positional_score * 3
        + mobility_score * 15
        + corner_score * 100
        + piece_difference * piece_weight
    )


def order_moves(board, moves, player):
    """Search strong-looking moves first so alpha-beta can prune more."""
    def move_score(move):
        row, col = move
        score = int(POSITION_VALUES[row, col])
        if move in CORNERS:
            score += 1000

        next_board = play_move(board, move, player)
        opponent_game = make_game(next_board)
        score -= 5 * len(opponent_game.legal_moves(-player))
        return score

    return sorted(moves, key=move_score, reverse=True)


def minimax(board, player, ai_player, depth, alpha, beta, deadline, passed=False):
    """Minimax search with alpha-beta pruning."""
    if time.perf_counter() >= deadline:
        raise OutOfTime

    game = make_game(board)
    moves = game.legal_moves(player)

    if not moves:
        if passed:
            return evaluate(board, ai_player)
        return minimax(
            board, -player, ai_player, depth, alpha, beta, deadline, True
        )

    if depth == 0:
        return evaluate(board, ai_player)

    moves = order_moves(board, moves, player)

    if player == ai_player:
        best_score = -float("inf")
        for move in moves:
            next_board = play_move(board, move, player)
            score = minimax(
                next_board, -player, ai_player, depth - 1,
                alpha, beta, deadline
            )
            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break
        return best_score

    best_score = float("inf")
    for move in moves:
        next_board = play_move(board, move, player)
        score = minimax(
            next_board, -player, ai_player, depth - 1,
            alpha, beta, deadline
        )
        best_score = min(best_score, score)
        beta = min(beta, best_score)
        if alpha >= beta:
            break
    return best_score


def search_depth(board, player, depth, deadline, previous_best):
    """Search every root move at one completed depth."""
    game = make_game(board)
    moves = game.legal_moves(player)
    moves = order_moves(board, moves, player)

    if previous_best in moves:
        moves.remove(previous_best)
        moves.insert(0, previous_best)

    best_move = moves[0]
    best_score = -float("inf")
    alpha = -float("inf")
    beta = float("inf")

    for move in moves:
        next_board = play_move(board, move, player)
        score = minimax(
            next_board, -player, player, depth - 1,
            alpha, beta, deadline
        )
        if score > best_score:
            best_score = score
            best_move = move
        alpha = max(alpha, best_score)

    return best_move


def choose_move(board, turn, time_limit):
    """Choose a legal move and return it as (row, column)."""
    game = make_game(board)
    legal_moves = game.legal_moves(turn)

    if not legal_moves:
        return (-1, -1)
    if len(legal_moves) == 1:
        return legal_moves[0]

    best_move = max(
        legal_moves,
        key=lambda move: POSITION_VALUES[move[0], move[1]]
    )

    limit = 4.4 if time_limit is None else min(4.4, max(0.05, time_limit - 0.5))
    deadline = time.perf_counter() + limit
    empty_squares = int(np.count_nonzero(board == 0))

    # Search one move farther on every completed iteration. With ten or fewer
    # empty squares, the loop attempts to solve the exact ending.
    maximum_depth = empty_squares if empty_squares <= 10 else 8

    for depth in range(1, maximum_depth + 1):
        try:
            best_move = search_depth(board, turn, depth, deadline, best_move)
        except OutOfTime:
            break

    return best_move
