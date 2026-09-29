#Zijie Zhang, Sep.24/2023
#
#This is the only file you need to change. reversi_client.py calls choose_move() every time it is your turn.

import time
import numpy as np
from reversi import reversi

#Positional weights: corners are great, squares beside an empty corner are a trap
WEIGHTS = np.array([
    [120, -20,  20,   5,   5,  20, -20, 120],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [120, -20,  20,   5,   5,  20, -20, 120],
], dtype=float)

DIRECTIONS = [(1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,1),(-1,0),(-1,-1)]
CORNERS = [(0,0), (0,7), (7,0), (7,7)]

TIME_SAFETY_MARGIN = 0.3
NO_LIMIT_TIME_BUDGET = 4.0
MAX_DEPTH = 64


#---- Board helpers: work on raw numpy arrays for speed during search ----

def get_legal_moves(board, piece):
    return [(x, y) for x in range(8) for y in range(8)
            if board[x, y] == 0 and any_flips(board, x, y, piece)]


def any_flips(board, x, y, piece):
    #True if playing at (x, y) sandwiches at least one opponent line
    for dx, dy in DIRECTIONS:
        cx, cy = x + dx, y + dy
        seen_opponent = False
        while 0 <= cx <= 7 and 0 <= cy <= 7:
            val = board[cx, cy]
            if val == 0:
                break
            if val == piece:
                if seen_opponent:
                    return True
                break
            seen_opponent = True
            cx, cy = cx + dx, cy + dy
    return False


def apply_move(board, x, y, piece):
    #Returns a new board with the move applied; never mutates the input
    new_board = board.copy()
    new_board[x, y] = piece
    for dx, dy in DIRECTIONS:
        cx, cy = x + dx, y + dy
        flip_list = []
        while 0 <= cx <= 7 and 0 <= cy <= 7:
            val = board[cx, cy]
            if val == 0:
                break
            if val == piece:
                for fx, fy in flip_list:
                    new_board[fx, fy] = piece
                break
            flip_list.append((cx, cy))
            cx, cy = cx + dx, cy + dy
    return new_board


#---- Evaluation function: higher is better for `piece` ----

def evaluate(board, piece):
    opponent = -piece

    positional = np.sum(WEIGHTS * board) * piece

    my_moves = len(get_legal_moves(board, piece))
    opp_moves = len(get_legal_moves(board, opponent))
    mobility = 100 * (my_moves - opp_moves) / (my_moves + opp_moves) if my_moves + opp_moves else 0

    my_corners = sum(1 for cx, cy in CORNERS if board[cx, cy] == piece)
    opp_corners = sum(1 for cx, cy in CORNERS if board[cx, cy] == opponent)
    corners = 100 * (my_corners - opp_corners) / (my_corners + opp_corners) if my_corners + opp_corners else 0

    #Piece count only matters once the board is nearly full
    empty_squares = int(np.count_nonzero(board == 0))
    if empty_squares < 12:
        my_count = int(np.count_nonzero(board == piece))
        opp_count = int(np.count_nonzero(board == opponent))
        parity = 100 * (my_count - opp_count) / (my_count + opp_count)
        return 2.0 * positional + 3.0 * mobility + 6.0 * corners + 4.0 * parity

    return 3.0 * positional + 4.0 * mobility + 8.0 * corners


#---- Minimax with alpha-beta pruning ----

def order_moves(moves):
    #Corners first, so pruning kicks in sooner
    return sorted(moves, key=lambda m: 0 if m in CORNERS else 1)


def minimax(board, piece, root_piece, depth, alpha, beta, deadline):
    if time.perf_counter() > deadline:
        raise TimeoutError

    moves = get_legal_moves(board, piece)
    if not moves:
        opponent_moves = get_legal_moves(board, -piece)
        if not opponent_moves:
            my_count = int(np.count_nonzero(board == root_piece))
            opp_count = int(np.count_nonzero(board == -root_piece))
            return 10000 if my_count > opp_count else -10000 if my_count < opp_count else 0
        return minimax(board, -piece, root_piece, depth - 1, alpha, beta, deadline)   #pass the turn

    if depth == 0:
        return evaluate(board, root_piece)

    maximizing = (piece == root_piece)
    best = float('-inf') if maximizing else float('inf')
    for x, y in order_moves(moves):
        child = apply_move(board, x, y, piece)
        value = minimax(child, -piece, root_piece, depth - 1, alpha, beta, deadline)
        if maximizing:
            best = max(best, value)
            alpha = max(alpha, best)
        else:
            best = min(best, value)
            beta = min(beta, best)
        if beta <= alpha:
            break   #alpha-beta cutoff
    return best


def choose_move(board, turn, time_limit):
    """Pick the square to play and return it as (x, y).

    board      : 8x8 numpy array. 1 = white piece, -1 = black piece, 0 = empty.
                 board[x, y] is row x, column y, exactly as print(board) and the game window show it.
    turn       : 1 if you are playing white, -1 if you are playing black.
    time_limit : seconds you have to answer, or None if there is no limit.
                 Answer late, or with an illegal move, and you forfeit this turn.

    You are only asked to move when you have at least one legal move.
    The reversi class helps you think:
        game.legal_moves(turn)                  -> list of (x, y) you may play
        game.step(x, y, turn, commit = False)   -> how many pieces that move flips, board untouched
        game.step(x, y, turn)                   -> actually play it (copy the board first when searching ahead)
    """

    start = time.perf_counter()
    budget = NO_LIMIT_TIME_BUDGET if time_limit is None else max(0.1, time_limit - TIME_SAFETY_MARGIN)
    deadline = start + budget

    moves = get_legal_moves(board, turn)
    if not moves:
        return (-1, -1)

    empty_squares = int(np.count_nonzero(board == 0))

    #Iterative deepening: search depth 1, 2, 3... keeping the last depth that finished in time
    best_move = moves[0]
    depth = 1
    try:
        while depth <= MAX_DEPTH:
            current_best_move = best_move
            current_best_score = float('-inf')
            alpha, beta = float('-inf'), float('inf')
            for x, y in order_moves(moves):
                child = apply_move(board, x, y, turn)
                score = minimax(child, -turn, turn, depth - 1, alpha, beta, deadline)
                if score > current_best_score:
                    current_best_score = score
                    current_best_move = (x, y)
                alpha = max(alpha, current_best_score)

            best_move = current_best_move
            if depth >= empty_squares:
                break   #searched to the end of the game
            depth += 1
    except TimeoutError:
        pass   #keep whatever the last fully-completed depth found

    return best_move