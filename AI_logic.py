import time
from typing import Any

import numpy as np
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
                                                #            Whether the netsssswork thinks its winning or losing. 
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
