#Zijie Zhang, Sep.24/2023
#
#This is the only file you need to change. reversi_client.py calls choose_move() every time it is your turn.

import numpy as np
from reversi import reversi


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

    #Local Greedy - Replace with your algorithm
    game = reversi()
    game.board = board.copy()

    best_move = (-1, -1)
    most_flips = 0
    for x, y in game.legal_moves(turn):
        flips = game.step(x, y, turn, commit = False)
        if flips > most_flips:
            most_flips = flips
            best_move = (x, y)
    return best_move
