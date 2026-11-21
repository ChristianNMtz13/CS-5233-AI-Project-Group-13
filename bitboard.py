# The board is stored as two 64 bit numbers, one for our pieces and one for theirs.
# Each bit is one square. It's a lot faster than looping over a numpy array.

import numpy as np


# (our pieces, their pieces). Always from the side of whoever's turn it is.
Position = tuple[int, int]


# Row and column to the bit for that square.
def square(row: int, col: int) -> int:
    return 1 << (8 * row + col)


# The other way around. Gives back the square number of every bit that's turned on.
def squares(bits: int) -> list[int]:
    found = []
    while bits:
        lowest = bits & -bits
        found.append(lowest.bit_length() - 1)
        bits ^= lowest
    return found


FULL = (1 << 64) - 1
# Everything except the left and right columns. Stops moves from wrapping around the edge of the board.
NOT_EDGE_COLUMNS = 0x7E7E7E7E7E7E7E7E


# Turns the game's numpy board into our two numbers.
def board_to_bitboards(board: np.ndarray, player: int) -> Position:
    cells = board.flatten().tolist()
    own = sum(1 << i for i, value in enumerate(cells) if value == player)
    opp = sum(1 << i for i, value in enumerate(cells) if value == -player)
    return own, opp


# What the network gets to see. Our pieces, their pieces, our legal moves and their legal moves.
INPUT_PLANES = 4

STATE_BYTES = 8 * INPUT_PLANES


def state_array(own: int, opp: int) -> np.ndarray:
    return unpack_states(pack_states([(own, opp)]))[0]


def bit_planes(boards) -> np.ndarray:
    boards = np.asarray(boards, dtype="<u8")
    return np.unpackbits(boards.view(np.uint8), axis=-1, bitorder="little").reshape(*boards.shape, 8, 8)


# Squishes positions down to bytes so they're cheap to send between processes.
def pack_states(positions: list[Position]) -> bytes:
    return b"".join(
        bits.to_bytes(8, "little")
        for own, opp in positions
        for bits in (own, opp, legal_moves(own, opp), legal_moves(opp, own))
    )


# Unsquishes them back into 8x8 grids for the network.
def unpack_states(raw: bytes) -> np.ndarray:
    return bit_planes(np.frombuffer(raw, dtype="<u8").reshape(-1, INPUT_PLANES))


# Counts the pieces on the board. Only used to show how far along the games are.
def packed_discs(packed: bytes) -> int:
    return int.from_bytes(packed[:16], "little").bit_count()


# When playing for real the network sends back two policies, a score and how unsure it is. 64 + 64 + 1 + 1.
PLAY_ROW = 130


# Cuts the network's answers back up into one per position. In training it's just a policy and a score.
def split_replies(rows: np.ndarray, play: bool = False) -> list[tuple]:
    if play:
        return [
            (row[:128].reshape(2, 64), float(row[128]), float(row[129]))
            for row in rows.reshape(-1, PLAY_ROW)
        ]

    rows = rows.reshape(-1, 65)
    return list(zip(rows[:, :64], rows[:, 64].tolist()))


# Finds every legal move at once. Slide our pieces in all 8 directions across the opponent's pieces.
# Wherever that lands on an empty square is a legal move.
def legal_moves(own: int, opp: int) -> int:
    sideways = opp & NOT_EDGE_COLUMNS
    moves = 0

    line = sideways & (own << 1)
    while line:
        line <<= 1
        moves |= line
        line &= sideways

    line = sideways & (own >> 1)
    while line:
        line >>= 1
        moves |= line
        line &= sideways

    line = sideways & (own << 7)
    while line:
        line <<= 7
        moves |= line
        line &= sideways

    line = sideways & (own >> 7)
    while line:
        line >>= 7
        moves |= line
        line &= sideways

    line = sideways & (own << 9)
    while line:
        line <<= 9
        moves |= line
        line &= sideways

    line = sideways & (own >> 9)
    while line:
        line >>= 9
        moves |= line
        line &= sideways

    line = opp & (own << 8)
    while line:
        line <<= 8
        moves |= line
        line &= opp

    line = opp & (own >> 8)
    while line:
        line >>= 8
        moves |= line
        line &= opp

    return moves & ~(own | opp) & FULL


# Places a piece and flips what it captures. Walk out in each direction, and if we hit
# one of our own pieces at the end, everything in between gets flipped.
def play_move(own: int, opp: int, move: int) -> Position:
    sideways = opp & NOT_EDGE_COLUMNS
    flips = 0

    at = move << 1
    if at & sideways:
        line = at
        at <<= 1
        while at & sideways:
            line |= at
            at <<= 1
        if at & own:
            flips |= line

    at = move >> 1
    if at & sideways:
        line = at
        at >>= 1
        while at & sideways:
            line |= at
            at >>= 1
        if at & own:
            flips |= line

    at = move << 7
    if at & sideways:
        line = at
        at <<= 7
        while at & sideways:
            line |= at
            at <<= 7
        if at & own:
            flips |= line

    at = move >> 7
    if at & sideways:
        line = at
        at >>= 7
        while at & sideways:
            line |= at
            at >>= 7
        if at & own:
            flips |= line

    at = move << 9
    if at & sideways:
        line = at
        at <<= 9
        while at & sideways:
            line |= at
            at <<= 9
        if at & own:
            flips |= line

    at = move >> 9
    if at & sideways:
        line = at
        at >>= 9
        while at & sideways:
            line |= at
            at >>= 9
        if at & own:
            flips |= line

    at = move << 8
    if at & opp:
        line = at
        at <<= 8
        while at & opp:
            line |= at
            at <<= 8
        if at & own:
            flips |= line

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
