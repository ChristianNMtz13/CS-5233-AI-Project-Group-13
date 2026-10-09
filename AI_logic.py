import math
import time
from itertools import combinations
from operator import itemgetter
from typing import Any

import numpy as np
import numpy.typing as npt

MOBILITY_RATIO_WEIGHT, MOBILITY_RATIO_WEIGHT_EMPTY_MULT = 743, 232   # 16 * (our legal moves - theirs) / (both counts + 1)
CORNER_WEIGHT, CORNER_WEIGHT_EMPTY_MULT = 1250, 1396                  # corners held
X_SQUARE_WEIGHT, X_SQUARE_WEIGHT_EMPTY_MULT = -539, 1064             # X-squares held next to an empty corner (counted against)
C_SQUARE_WEIGHT, C_SQUARE_WEIGHT_EMPTY_MULT = 587, 491               # C-squares held next to an occupied corner
FRONTIER_WEIGHT, FRONTIER_WEIGHT_EMPTY_MULT = 1088, 135                # discs next to an empty square (counted against)
DISC_WEIGHT, DISC_WEIGHT_EMPTY_MULT = 942, -18                      # discs
QUIET_MOVE_WEIGHT, QUIET_MOVE_WEIGHT_EMPTY_MULT = 268, 438          # legal moves into a hole: an empty square with no empty neighbour

# Time
DEFAULT_TIME_LIMIT = 5          # seconds per move
SAFETY_MARGIN = 0.5             # seconds kept for latency

#Transposition cache table min depth ("cache" table)
TABLE_MIN_DEPTH = 2

# Multi-ProbCut
MPC_MIN_DEPTH = 2
MPC_T = 1.0
MPC_SIGMA, MPC_SIGMA_EMPTY_MULT = 9476, 4
MPC_SIGMA_PER_SHALLOW_PLY, MPC_SIGMA_PER_DEEP_PLY = -2229, 1400

# Endgame solver
SOLVE_EMPTIES = 18                  # when to start perfect solving
SOLVE_RESERVE = 0.5
SOLVE_SHALLOW = 5


# Types for readability

# full game board
Board = npt.NDArray[np.integer[Any]]

# both players positions in bitboard
Position = tuple[int, int]

Child = tuple[int, int, Position, int]

# transposition table entry (depth searched, bound type, score, best move).
Entry = tuple[float, int, float, int | None]
EXACT, LOWER_BOUND, UPPER_BOUND = 0, 1, 2
PROVEN = math.inf               # depth searched for a proven result: it holds at any depth


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
START = square(3, 3) | square(3, 4) | square(4, 3) | square(4, 4)  # the four discs a game starts with

CORNER_REGIONS = (
    (square(0, 0), square(1, 1), square(0, 1) | square(1, 0)),
    (square(0, 7), square(1, 6), square(0, 6) | square(1, 7)),
    (square(7, 0), square(6, 1), square(7, 1) | square(6, 0)),
    (square(7, 7), square(6, 6), square(7, 6) | square(6, 7)),
)
CORNERS = sum(corner for corner, _, _ in CORNER_REGIONS)

# every combination of x squares and c squares, used for evaluate()
def xcsquares(part: int) -> dict[int, int]:
    return {sum(region[0] for region in chosen): sum(region[part] for region in chosen)
            for count in range(len(CORNER_REGIONS) + 1)
            for chosen in combinations(CORNER_REGIONS, count)}

X_SQUARES = xcsquares(1)
C_SQUARES = xcsquares(2)

# squares next to discs
def neighbors(discs: int) -> int:
    right = discs & NOT_LAST_COLUMN   # can move to col + 1
    left = discs & NOT_FIRST_COLUMN   # can move to col - 1
    return (discs << 8 | discs >> 8
            | right << 1 | right << 9 | right >> 7
            | left >> 1 | left >> 9 | left << 7)

LANE = 128  # 128 bits

def repeat_in_lanes(mask: int, lanes: int) -> int:
    return sum(mask << (LANE * i) for i in range(lanes))

LANE_MASKS = [(repeat_in_lanes(NOT_EDGE_COLUMNS, n), repeat_in_lanes(FULL, n)) for n in range(61)]

# for evaluate to use for the "lane"
NOT_EDGE_COLUMNS_2, FULL_2 = LANE_MASKS[2]


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
def legal_moves(own: int, opp: int, not_edge: int = NOT_EDGE_COLUMNS, full: int = FULL) -> int:
    sideways = opp & not_edge
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

    return moves & ~(own | opp) & full


# put our disc on the square and turn over every disc necessary.
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


# search every move and put the one that leaves the other player the fewest replies first. 
def ordered_children(own: int, opp: int, moves: int) -> list[Child]:
    played = []
    packed_to_move = packed_waiting = lane_offset = 0
    while moves:
        move = moves & -moves   # the lowest square left
        moves ^= move
        child = play_move(own, opp, move)
        played.append((move, child))
        packed_to_move |= child[0] << lane_offset   # each child goes in the next lane
        packed_waiting |= child[1] << lane_offset
        lane_offset += LANE

    # All the children's replies are counted in a single call.
    replies = legal_moves(packed_to_move, packed_waiting, *LANE_MASKS[len(played)])
    children = []
    for move, child in played:
        child_replies = replies & FULL   # this child's lane
        children.append((child_replies.bit_count(), move, child, child_replies))
        replies >>= LANE
    children.sort()
    return children


#
# Evaluation
#

# how good the position is for the player to move, this is where we use the heuristic weights
def evaluate(own: int, opp: int) -> int:
    occupied = own | opp
    empty = FULL & ~occupied
    # check both sides moves in one call. Our side in one lane, theirs in the other.
    moves = legal_moves(own | opp << LANE, opp | own << LANE, NOT_EDGE_COLUMNS_2, FULL_2)
    own_moves, opp_moves = moves & FULL, moves >> LANE
    # discs with an empty square next to them
    frontier = neighbors(empty)
    # x squares next to an empty corner
    risky = X_SQUARES[CORNERS & empty]
    # c squares next to a corner we already hold
    safe = C_SQUARES[CORNERS & occupied]

    # bit counting and bitwise operation stuff, too much to comment.
    own_mobility, opp_mobility = own_moves.bit_count(), opp_moves.bit_count()
    mobility_ratio = int(16 * (own_mobility - opp_mobility) / (own_mobility + opp_mobility + 1))
    corners = (own & CORNERS).bit_count() - (opp & CORNERS).bit_count()
    x_squares = (opp & risky).bit_count() - (own & risky).bit_count()
    c_squares = (own & safe).bit_count() - (opp & safe).bit_count()
    frontier_discs = (opp & frontier).bit_count() - (own & frontier).bit_count()
    discs = own.bit_count() - opp.bit_count()
    # A "quiet" moves fill in a hole with no empty squares next to it.
    quiet_moves = (own_moves & ~frontier).bit_count() - (opp_moves & ~frontier).bit_count()

    empties = 64 - occupied.bit_count()
    return ((MOBILITY_RATIO_WEIGHT + MOBILITY_RATIO_WEIGHT_EMPTY_MULT * empties) * mobility_ratio
            + (CORNER_WEIGHT + CORNER_WEIGHT_EMPTY_MULT * empties) * corners
            + (X_SQUARE_WEIGHT + X_SQUARE_WEIGHT_EMPTY_MULT * empties) * x_squares
            + (C_SQUARE_WEIGHT + C_SQUARE_WEIGHT_EMPTY_MULT * empties) * c_squares
            + (FRONTIER_WEIGHT + FRONTIER_WEIGHT_EMPTY_MULT * empties) * frontier_discs
            + (DISC_WEIGHT + DISC_WEIGHT_EMPTY_MULT * empties) * discs
            + (QUIET_MOVE_WEIGHT + QUIET_MOVE_WEIGHT_EMPTY_MULT * empties) * quiet_moves)


# the exact score of a finished game, scaled by n^28 so that it always beats our heuristic.
def final_score(own: int, opp: int) -> int:
    return (own.bit_count() - opp.bit_count()) << 28


# the result of a finished game. 1 for a win, 0 for a draw, -1 for a loss
def outcome(own: int, opp: int) -> int:
    difference = own.bit_count() - opp.bit_count()
    return (difference > 0) - (difference < 0)


#
# Endgame solver tables
#

#
# Buckle up.
# This section gets messy.
#
# I used Egaroucid as the reference to write this.

QUADRANTS = (0x0F0F0F0F, 0xF0F0F0F0, 0x0F0F0F0F << 32, 0xF0F0F0F0 << 32)  # the four 4x4 corners of the board
QUADRANT = [(x >> 2 & 1) | (x >> 4 & 2) for x in range(64)]   # which of QUADRANTS each square is in
NEIGHBOURS = [neighbors(1 << x) & FULL for x in range(64)]   # the squares next to each square


# order to try the empty squares in
# squares in a corner area holding an odd number of empty squares go first, because
# whoever plays there can also take that area's last square
def parity_orders(n: int) -> list[itemgetter]:
    orders = []
    for key in range(4 ** n):
        quadrants = [key >> 2 * i & 3 for i in range(n)]
        orders.append(itemgetter(*sorted(range(n), key=lambda i: quadrants.count(quadrants[i]) % 2 == 0)))
    return orders


PARITY_ORDER_3, PARITY_ORDER_4 = parity_orders(3), parity_orders(4)

# how many discs a move at `position` turns over along a single line of 8 squares that is full
def flips_along(position: int, own: int) -> int:
    flips = 0
    for step in (1, -1):
        end = position + step
        while 0 <= end < 8 and not own >> end & 1:
            end += step
        if 0 <= end < 8:   # the opponent's discs end at one of ours
            flips += abs(end - position) - 1
    return flips


# all the squares on the diagonal line through a square, going one column per row
def diagonal_through(row: int, col: int, col_step: int) -> int:
    return sum(square(r, col + (r - row) * col_step)
               for r in range(8) if 0 <= col + (r - row) * col_step < 8)


LAST_FLIPS = [[flips_along(position, own) for own in range(256)] for position in range(8)]
FIRST_COLUMN = 0x0101010101010101
COLUMN_TO_ROW = 0x0102040810204080     # column 0 times this puts row i at bit 56 + i
DIAGONAL_TO_ROW = 0x0101010101010101   # a diagonal times this puts column i at bit 56 + i
# for each square, what last_flips() needs for the 4 lines through it.
LAST_LINES = [(8 * row, col, diagonal_through(row, col, 1), diagonal_through(row, col, -1), LAST_FLIPS[col], LAST_FLIPS[row])
              for row in range(8) for col in range(8)]


# how many discs the player to move turns over by playing x, the only empty square left.
# Everything else belongs to the other side, so the tables above answer it without playing the move
def last_flips(own: int, x: int) -> int:
    row_shift, col, diagonal, antidiagonal, along_row, along_column = LAST_LINES[x]
    return (along_row[own >> row_shift & 0xFF]
            + along_column[(own >> col & FIRST_COLUMN) * COLUMN_TO_ROW >> 56 & 0xFF]
            + along_row[(own & diagonal) * DIAGONAL_TO_ROW >> 56 & 0xFF]
            + along_row[(own & antidiagonal) * DIAGONAL_TO_ROW >> 56 & 0xFF])


#
# Searching
#

# helper function
# stops the search when the time for this move is up
class SearchTimeout(Exception):
    pass


# the bot's actual brain: it looks for the best move and uses transposition table to "cache" what it learned on earlier moves
class Searcher:

    def __init__(self) -> None:
        self.table: dict[Position, Entry] = {}
        # The solver's answers: (lowest, highest, best move), so the real result is between lowest and highest.
        self.solved: dict[Position, tuple[int, int, int]] = {}
        self.deadline = 0.0
        self.solve_deadline = 0.0
        self.nodes = 0      # positions visited this move
        self.last_occupied = FULL   # squares taken right after our last move (FULL: none yet)

    # play one move: find the best square we can within the time allowed
    def choose_move(self, board: Board, player: int, time_limit: float | None) -> tuple[int, int]:
        started = time.perf_counter()
        own, opp = board_to_bitboards(board, player)
        self.deadline = started + (time_limit or DEFAULT_TIME_LIMIT) - SAFETY_MARGIN
        self.nodes = 0

        occupied = own | opp
        empties = 64 - occupied.bit_count()
        self.prune_tables(occupied, empties)

        # near the end, first try to perfect solve a win or a draw. If the game is lost anyway, or the
        # proof runs out of time, the normal search picks the move.
        proof = None
        if empties <= SOLVE_EMPTIES:
            self.solve_deadline = self.deadline - SOLVE_RESERVE
            proof = self.prove(own, opp, empties)
        if proof is None:
            move, score, depth, partial = self.iterative_deepening(own, opp, empties)
            searched = f'{depth:>2}{"+" if partial else " "}'
        else:
            move, score = proof
            searched = ('draw', 'win')[score]

        self.report(move, searched, score, occupied, empties, time.perf_counter() - started)
        self.last_occupied = occupied | move   # our move adds one disc; flips don't change occupancy
        return divmod(move.bit_length() - 1, 8)

    # print the one line of stats for the move that was just chosen
    def report(self, move: int, searched: str, score: float | None, occupied: int, empties: int, elapsed: float) -> None:
        row, col = divmod(move.bit_length() - 1, 8)
        print(f'Move: {61 - empties:>2} | '
            f'Depth: {searched}/{empties:<2} | '
            f'Opp: {self.opponent_moves(occupied, empties):<5} | '
            f'Pos: ({row},{col}) | '
            f'Eval: {str(score):>2} | '
            f'{self.nodes:>9,} nodes | '
            f'{elapsed:>3.2f}s | '
            f'{self.nodes / elapsed:>7,.0f} nodes/s')

    # what the other player did since our last move, worked out from the discs that have appeared
    # this is purely for stat printing.
    def opponent_moves(self, occupied: int, empties: int) -> str:
        before = self.last_occupied
        new_game = before & ~occupied    # a disc we saw is gone (or we haven't moved yet)
        if new_game:
            # Only the usual start is known; after a swapped opening the moves can't be told apart.
            before = START if empties >= 59 else occupied
        replies = occupied & ~before
        if not replies:
            return '-' if new_game else 'pass'
        return ' '.join(f'({index // 8},{index % 8})'
                        for index in range(64) if replies >> index & 1)

    # forget the remembered positions that can't come up any more by pruning table.
    def prune_tables(self, occupied: int, empties: int) -> None:
        if empties >= 59:
            self.table, self.solved = {}, {}
        else:
            self.table = {key: entry for key, entry in self.table.items()
                          if ((key[0] | key[1]) & occupied) == occupied}
            self.solved = {key: entry for key, entry in self.solved.items()
                           if ((key[0] | key[1]) & occupied) == occupied}

    # search depth 1, then 2, then 3, and so on until time runs out.
    def iterative_deepening(self, own: int, opp: int, empties: int) -> tuple[int, float | None, int, bool]:
        root = ordered_children(own, opp, legal_moves(own, opp))
        best_move, best_score, finished_depth, partial = root[0][1], None, 0, False
        try:
            # a pass uses up a step without filling a square
            # so a game can last up to twice as many steps as there are empty squares.
            for depth in range(1, 2 * empties + 1):
                # the last depth's best move goes first, the others keep their order.
                root.sort(key=lambda child: child[1] != best_move)

                _, first_move, position, replies = root[0]
                score, all_proven = self.search_child(position, replies, depth, -math.inf, math.inf)
                alpha, depth_best = score, first_move
                for _, move, position, replies in root[1:]:
                    score, proven = self.scout_child(position, replies, depth, alpha, math.inf)
                    all_proven = all_proven and proven
                    if score > alpha:
                        alpha, depth_best = score, move
                        # partial means that the depth didn't completely finish searching
                        # but it still found a better move than the previous depth.
                        best_move, best_score, partial = move, score, True

                best_move, best_score, finished_depth, partial = depth_best, alpha, depth, False
                if all_proven:
                    break
        except SearchTimeout:
            pass
        return best_move, best_score, finished_depth, partial

    # the main search function, score the position for the player to move, looking "depth" moves ahead.
    # alpha and beta say which scores still matter. outside them only a bound is needed
    def negamax(self, own: int, opp: int, depth: int, alpha: float, beta: float,
                moves: int | None = None) -> tuple[float, bool]:
        self.nodes += 1
        if time.perf_counter() > self.deadline:
            raise SearchTimeout

        if depth == 0 and (own | opp) != FULL:
            return evaluate(own, opp), False

        # check if position is already in cache table.
        use_table = depth >= TABLE_MIN_DEPTH
        entry = self.table.get((own, opp)) if use_table else None
        best_move = None
        if entry is not None:
            stored_depth, bound, stored_score, best_move = entry
            # a remembered score settles the position if it came from a search at least as deep,
            # and it's exact or a bound that is already past alpha or beta.
            if stored_depth >= depth and (bound == EXACT
                                          or bound == LOWER_BOUND and stored_score >= beta
                                          or bound == UPPER_BOUND and stored_score <= alpha):
                return stored_score, stored_depth == PROVEN

        if depth >= MPC_MIN_DEPTH:
            cut = self.probcut(own, opp, depth, alpha, beta)
            if cut is not None:
                return cut, False

        if moves is None:
            moves = legal_moves(own, opp)
        if moves:
            if depth == 1:
                score, proven = self.search_last_ply(own, opp, moves, alpha, beta)
            else:
                score, best_move, proven = self.search_moves(own, opp, moves, depth, alpha, beta, best_move)
        elif legal_moves(opp, own):
            # pass
            score, proven = self.search_child((opp, own), None, depth, alpha, beta)
        else:
            # neither player can move, game over.
            score, proven = final_score(own, opp), True

        stored_depth = PROVEN if proven else depth
        if use_table and (entry is None or stored_depth >= entry[0]):
            bound = UPPER_BOUND if score <= alpha else LOWER_BOUND if score >= beta else EXACT
            self.table[(own, opp)] = (stored_depth, bound, score, best_move)
        return score, proven
    
    # this one is nuts, but it had a huge improvement on the bot strength.
    # multi-probcut uses the fact that a deep search and a much cheaper shallow 
    # one nearly always agree on the score.
    # When the shallow answer is already clear the deep search is skipped. 
    def probcut(self, own: int, opp: int, depth: int, alpha: float, beta: float) -> float | None:
        empties = 64 - (own | opp).bit_count()
        if depth >= empties:
            return None
        # the shallow search keeps the same odd/even depth as the deep one
        shallow = (depth >> 1 & ~1) | (depth & 1)
        margin = int(MPC_T * (MPC_SIGMA + MPC_SIGMA_EMPTY_MULT * empties
                             + MPC_SIGMA_PER_SHALLOW_PLY * shallow + MPC_SIGMA_PER_DEEP_PLY * depth))
        if beta < math.inf:
            bound = beta + margin
            if self.negamax(own, opp, shallow, bound - 1, bound)[0] >= bound:
                return beta
        if alpha > -math.inf:
            bound = alpha - margin
            if self.negamax(own, opp, shallow, bound, bound + 1)[0] <= bound:
                return alpha
        return None

    def search_moves(self, own: int, opp: int, moves: int, depth: int, alpha: float, beta: float,
                     best_move: int | None) -> tuple[float, int, bool]:
        children: list[Child] | None = None
        if best_move is None:
            children = ordered_children(own, opp, moves)
            _, best_move, position, replies = children.pop(0)
        else:
            position, replies = play_move(own, opp, best_move), None
        score, proven = self.search_child(position, replies, depth, alpha, beta)
        alpha = max(alpha, score)
        if alpha >= beta:
            return alpha, best_move, proven

        # the rest of the moves are expected to be worse, which scout_child() confirms.
        if children is None:
            children = ordered_children(own, opp, moves ^ best_move)
        for _, move, position, replies in children:
            score, child_proven = self.scout_child(position, replies, depth, alpha, beta)
            proven = proven and child_proven
            if score > alpha:
                alpha, best_move = score, move
                if alpha >= beta:
                    break
        return alpha, best_move, proven

    # the move loop for the last step, where every move is scored on the spot instead of being searched further.
    def search_last_ply(self, own: int, opp: int, moves: int, alpha: float, beta: float) -> tuple[float, bool]:
        proven = True
        while moves:
            move = moves & -moves   # the lowest square left
            moves ^= move
            self.nodes += 1
            child = play_move(own, opp, move)
            if (child[0] | child[1]) == FULL:
                score = -final_score(*child)
            else:
                score = -evaluate(*child)
                proven = False
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
        return alpha, proven

    def search_child(self, position: Position, replies: int | None, depth: int,
                     alpha: float, beta: float) -> tuple[float, bool]:
        score, proven = self.negamax(*position, depth - 1, -beta, -alpha, replies)
        return -score, proven

    # check whether a move beats the best one so far.
    def scout_child(self, position: Position, replies: int | None, depth: int,
                    alpha: float, beta: float) -> tuple[float, bool]:
        score, proven = self.search_child(position, replies, depth, alpha, alpha + 1)
        if alpha < score < beta:
            score, rescored_proven = self.search_child(position, replies, depth, alpha, beta)
            proven = proven and rescored_proven
        return score, proven

    # look for a move that wins, or failing that a move that draws.
    def prove(self, own: int, opp: int, empties: int) -> tuple[int, int] | None:
        children = ordered_children(own, opp, legal_moves(own, opp))
        try:
            for target in (1, 0):   # a win, then a draw
                for _, move, position, replies in children:.
                    if -self.solve(*position, empties - 1, -target, replies) >= target:
                        return move, target
        except SearchTimeout:
            pass
        return None

    # the exact result with perfect play for the player
    def solve(self, own: int, opp: int, empties: int, alpha: int, moves: int | None = None) -> int:
        self.nodes += 1
        if time.perf_counter() > self.solve_deadline:
            raise SearchTimeout

        lowest, highest, best_move = self.solved.get((own, opp), (-1, 1, 0))
        if lowest > alpha:
            return lowest
        if highest <= alpha:
            return highest

        if moves is None:
            moves = legal_moves(own, opp)
        if not moves:
            if legal_moves(opp, own):
                return -self.solve(opp, own, empties, -alpha - 1)   # pass
            return outcome(own, opp)

        best = -2   # below any result
        if best_move:
            best = -self.solve_child(play_move(own, opp, best_move), empties, alpha)
            moves ^= best_move
        if best <= alpha and moves:
            children = ordered_children(own, opp, moves)
            children.sort(key=lambda child: child[0] - bool(child[1] & CORNERS))   # corners a reply earlier
            for _, move, position, replies in children:
                score = -self.solve_child(position, empties, alpha, replies)
                if score > best:
                    best, best_move = score, move
                    if best > alpha:
                        break

        if best > alpha:
            lowest = best
        else:
            highest = best
        self.solved[(own, opp)] = (lowest, highest, best_move)
        return best

    # solve one move ahead for the other player, handing the last few squares to the quicker solver
    def solve_child(self, position: Position, empties: int, alpha: int, replies: int | None = None) -> int:
        if empties - 1 > SOLVE_SHALLOW:
            return self.solve(*position, empties - 1, -alpha - 1, replies)
        return self.solve_shallow(*position, FULL & ~(position[0] | position[1]), empties - 1, -alpha - 1)

    #
    # the last 4 moves have their own specific optimizations
    # these functions are referenced from Egaroucid technical explanation page.
    #
    def solve_shallow(self, own: int, opp: int, empty: int, empties: int, alpha: int) -> int:
        if empties <= 4:
            squares = []
            while empty:
                squares.append((empty & -empty).bit_length() - 1)
                empty &= empty - 1
            if empties == 4:
                return self.solve_4(own, opp, *squares, alpha)
            if empties == 3:
                return self.solve_3(own, opp, *squares, alpha)
            if empties == 2:
                return self.solve_2(own, opp, *squares, alpha)
            if empties == 1:
                return self.solve_1(own, opp, *squares)
            return outcome(own, opp)
        self.nodes += 1
        odd = 0
        for quadrant in QUADRANTS:
            if (empty & quadrant).bit_count() & 1:
                odd |= empty & quadrant
        best = -2   # below any result: no legal move yet
        for group in (odd, empty ^ odd):
            while group:
                move = group & -group   # the lowest square left
                group ^= move
                child = play_move(own, opp, move)
                if child[0] != opp:   # something flipped, so the move is legal
                    score = -self.solve_shallow(*child, empty ^ move, empties - 1, -alpha - 1)
                    if score > alpha:
                        return score
                    best = max(best, score)
        if best > -2:
            return best
        if legal_moves(opp, own):
            return -self.solve_shallow(opp, own, empty, empties, -alpha - 1)   # pass
        return outcome(own, opp)

    # solvers for exactly 4, 3, 2 and 1 empty squares

    # solver for 4 empty squares left
    def solve_4(self, own: int, opp: int, x1: int, x2: int, x3: int, x4: int,
                alpha: int, passed: bool = False) -> int:
        self.nodes += 1
        x1, x2, x3, x4 = PARITY_ORDER_4[QUADRANT[x1] | QUADRANT[x2] << 2 | QUADRANT[x3] << 4
                                        | QUADRANT[x4] << 6]((x1, x2, x3, x4))
        best = -2   # below any result: no legal move yet
        for x, a, b, c in ((x1, x2, x3, x4), (x2, x1, x3, x4), (x3, x1, x2, x4), (x4, x1, x2, x3)):
            if NEIGHBOURS[x] & opp:
                child = play_move(own, opp, 1 << x)
                if child[0] != opp:   # something flipped, so the move is legal
                    score = -self.solve_3(*child, a, b, c, -alpha - 1)
                    if score > alpha:
                        return score
                    best = max(best, score)
        if best > -2:
            return best
        if passed:
            return outcome(own, opp)
        return -self.solve_4(opp, own, x1, x2, x3, x4, -alpha - 1, True)   # pass

    # solver for 3 empty squares left
    def solve_3(self, own: int, opp: int, x1: int, x2: int, x3: int,
                alpha: int, passed: bool = False) -> int:
        self.nodes += 1
        x1, x2, x3 = PARITY_ORDER_3[QUADRANT[x1] | QUADRANT[x2] << 2 | QUADRANT[x3] << 4]((x1, x2, x3))
        best = -2
        for x, a, b in ((x1, x2, x3), (x2, x1, x3), (x3, x1, x2)):
            if NEIGHBOURS[x] & opp:
                child = play_move(own, opp, 1 << x)
                if child[0] != opp:
                    score = -self.solve_2(*child, a, b, -alpha - 1)
                    if score > alpha:
                        return score
                    best = max(best, score)
        if best > -2:
            return best
        if passed:
            return outcome(own, opp)
        return -self.solve_3(opp, own, x1, x2, x3, -alpha - 1, True)

    # solver for 2 empty squares left, ordering the moves isn't worth it
    def solve_2(self, own: int, opp: int, x1: int, x2: int,
                alpha: int, passed: bool = False) -> int:
        self.nodes += 1
        best = -2
        if NEIGHBOURS[x1] & opp:
            child = play_move(own, opp, 1 << x1)
            if child[0] != opp:
                best = -self.solve_1(*child, x2)
                if best > alpha:
                    return best
        if NEIGHBOURS[x2] & opp:
            child = play_move(own, opp, 1 << x2)
            if child[0] != opp:
                return max(best, -self.solve_1(*child, x1))
        if best > -2:
            return best
        if passed:
            return outcome(own, opp)
        return -self.solve_2(opp, own, x1, x2, -alpha - 1, True)

    # solver for 1 empty square left.
    def solve_1(self, own: int, opp: int, x: int) -> int:
        self.nodes += 1
        discs = own.bit_count()   # the opponent has the other 63 - discs
        if flips := last_flips(own, x):
            difference = 2 * (discs + flips) - 62    # (discs + 1 + flips) - (63 - discs - flips)
        elif flips := last_flips(opp, x):
            difference = 2 * (discs - flips) - 64    # (discs - flips) - (64 - discs + flips)
        else:
            difference = 2 * discs - 63              # neither side can play it
        return (difference > 0) - (difference < 0)


brain = Searcher()

#this is for the reversi client to call.
def choose_move(board: Board, turn: int, time_limit: float | None) -> tuple[int, int]:
    return brain.choose_move(board, turn, time_limit)
