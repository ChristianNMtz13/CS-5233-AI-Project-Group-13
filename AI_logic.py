"""Readable Reversi AI using minimax and alpha-beta pruning."""
import time
from typing import Any

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
import numpy.typing as npt
import torch
from torch import nn

# Time
DEFAULT_TIME_LIMIT = 5          # seconds per move
SAFETY_MARGIN = 0.5             # seconds kept for latency


# Types for readability

# full game board
Board = npt.NDArray[np.integer[Any]]

# both players positions in bitboard
Position = tuple[int, int]


#
# board masks
#

# an empty bitboard with only a single square set to 1.
def square(row: int, col: int) -> int:
    return 1 << (8 * row + col)


FULL = (1 << 64) - 1                    # all 64 squares
NOT_FIRST_COLUMN = 0xFEFEFEFEFEFEFEFE   # columns 1-7
NOT_LAST_COLUMN = 0x7F7F7F7F7F7F7F7F    # columns 0-6: discs that can shift toward col + 1
NOT_EDGE_COLUMNS = NOT_FIRST_COLUMN & NOT_LAST_COLUMN  # columns 1-6


#
# move generation
#

# split the board into two piles of discs, ours and the other player's
def board_to_bitboards(board: Board, player: int) -> Position:
    squares = board.flatten().tolist()
    own = sum(1 << i for i, value in enumerate(squares) if value == player)
    opp = sum(1 << i for i, value in enumerate(squares) if value == -player)
    return own, opp


# every empty square where the player can put a disc and turn some of the other side's discs over.
def legal_moves(own: int, opp: int) -> int:
    sideways = opp & NOT_EDGE_COLUMNS
    moves = 0

    # this is done for every direction. Same as how we flip tiles.
    # col + 1
    line = sideways & (own << 1)
    while line:
        line <<= 1
        moves |= line
        line &= sideways

    # col - 1
    line = sideways & (own >> 1)
    while line:
        line >>= 1
        moves |= line
        line &= sideways

    # row + 1, col - 1
    line = sideways & (own << 7)
    while line:
        line <<= 7
        moves |= line
        line &= sideways

    # row - 1, col + 1
    line = sideways & (own >> 7)
    while line:
        line >>= 7
        moves |= line
        line &= sideways

    # row + 1, col + 1
    line = sideways & (own << 9)
    while line:
        line <<= 9
        moves |= line
        line &= sideways

    # row - 1, col - 1
    line = sideways & (own >> 9)
    while line:
        line >>= 9
        moves |= line
        line &= sideways

    # row + 1
    line = opp & (own << 8)
    while line:
        line <<= 8
        moves |= line
        line &= opp

    # row - 1
    line = opp & (own >> 8)
    while line:
        line >>= 8
        moves |= line
        line &= opp

    return moves & ~(own | opp) & FULL


# put our disc on the square and turn over every disc necessary.
# returns the new position from the other player's side: (their discs, our discs)
def play_move(own: int, opp: int, move: int) -> Position:
    sideways = opp & NOT_EDGE_COLUMNS   # for the same reasons as in legal_moves()
    flips = 0

    # same as legal moves function

    # col + 1
    at = move << 1
    if at & sideways:
        line = at
        at <<= 1
        while at & sideways:
            line |= at
            at <<= 1
        if at & own:
            flips |= line

    # col - 1
    at = move >> 1
    if at & sideways:
        line = at
        at >>= 1
        while at & sideways:
            line |= at
            at >>= 1
        if at & own:
            flips |= line

    # row + 1, col - 1
    at = move << 7
    if at & sideways:
        line = at
        at <<= 7
        while at & sideways:
            line |= at
            at <<= 7
        if at & own:
            flips |= line

    # row - 1, col + 1
    at = move >> 7
    if at & sideways:
        line = at
        at >>= 7
        while at & sideways:
            line |= at
            at >>= 7
        if at & own:
            flips |= line

    # row + 1, col + 1
    at = move << 9
    if at & sideways:
        line = at
        at <<= 9
        while at & sideways:
            line |= at
            at <<= 9
        if at & own:
            flips |= line

    # row - 1, col - 1
    at = move >> 9
    if at & sideways:
        line = at
        at >>= 9
        while at & sideways:
            line |= at
            at >>= 9
        if at & own:
            flips |= line

    # row + 1
    at = move << 8
    if at & opp:
        line = at
        at <<= 8
        while at & opp:
            line |= at
            at <<= 8
        if at & own:
            flips |= line

    # row - 1
    at = move >> 8
    if at & opp:
        line = at
        at >>= 8
        while at & opp:
            line |= at
            at >>= 8
        if at & own:
            flips |= line

    return opp ^ flips, own | move | flips


#
# network
#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# board input: (batch, 2, 8, 8), our discs and their discs
# out: 64 move logits, and a value in [-1, 1] for how good the board is for us
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 2:             2 boards, one for our discs, and one for opponents.
        # 64:            Amount of filters. Width of the network. 
        #                Bigger number = stronger evaluation but searches moves slower
        #                smaller = weaker but searches moves faster
        # kernel_size=3: Each of the 64 filters looks at a 3x3 patch of squares.
        # padding=1:     Adds a ring of 0s around the board so a 3x3 patch can still look at edge squares.
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.policy = nn.Linear(64 * 8 * 8, 64) # (... , 64): One score per square, this is what move the network wants to play 
        self.value = nn.Linear(64 * 8 * 8, 1)   # (... , 1): Single number range of -1 to 1. 
                                                #            Whether the network thinks its winning or losing. 
                                                #            -1 if the network thinks its definitely losing. 0 for tie. 1 for definitely winning.

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.conv1(x))   # (batch, 64, 8, 8)
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = x.flatten(1)                # (batch, 4096)
        policy = self.policy(x)         # (batch, 64)
        value = torch.tanh(self.value(x))  # (batch, 1)
        return policy, value


# bitboards to the net's input, bit 8 * row + col lands on [row][col]
def to_tensor(own: int, opp: int) -> torch.Tensor:
    planes = [[own >> i & 1 for i in range(64)], [opp >> i & 1 for i in range(64)]]
    return torch.tensor(planes, dtype=torch.float32, device=device).view(1, 2, 8, 8)


# random weights until we load from a weights file from self play training
net = Net().to(device)


#
# Move timer
#

# raise this to stop the search when the time for this move is up
class SearchTimeout(Exception):
    pass


deadline = 0.0   # set at the start of every choose_move()

def choose_move(board: Board, turn: int, time_limit: float | None) -> tuple[int, int]:
    global deadline
    deadline = time.perf_counter() + (time_limit or DEFAULT_TIME_LIMIT) - SAFETY_MARGIN
    own, opp = board_to_bitboards(board, turn)
    moves = legal_moves(own, opp)
    with torch.no_grad():
        policy, _ = net(to_tensor(own, opp))
    scores = policy[0].tolist()
    move = max((x for x in range(64) if moves >> x & 1), key=lambda x: scores[x])
    return divmod(move, 8)
