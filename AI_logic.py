import random
import time
from typing import Any

import numpy as np
import numpy.typing as npt

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
# Move timer
#

# stops the search when the time for this move is up
class SearchTimeout(Exception):
    pass


deadline = 0.0   # set at the start of every choose_move()


# a search calls this at every node, and catches SearchTimeout to fall back to its best move so far
def check_time() -> None:
    if time.perf_counter() > deadline:
        raise SearchTimeout


#this is for the reversi client to call.
def choose_move(board: Board, turn: int, time_limit: float | None) -> tuple[int, int]:
    global deadline
    deadline = time.perf_counter() + (time_limit or DEFAULT_TIME_LIMIT) - SAFETY_MARGIN
    own, opp = board_to_bitboards(board, turn)
    moves = legal_moves(own, opp)
    move = random.choice([x for x in range(64) if moves >> x & 1])
    return divmod(move, 8)
