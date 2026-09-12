#Zijie Zhang, Sep.24/2023
"""Play two AI files against each other on this machine, without the arena. Handy for testing.

    python local_match.py AI_logic.py my_bot.py            # one game, first file plays white
    python local_match.py AI_logic.py my_bot.py --games 10 # colours alternate every game
"""

import argparse
import importlib.util
import time

import numpy as np

from reversi import reversi


def load_player(path):
    spec = importlib.util.spec_from_file_location(f'ai_{abs(hash(path))}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.choose_move


def play(white, black, time_limit, verbose):
    game = reversi()
    bots = {1: white, -1: black}
    while True:
        if not game.legal_moves(game.turn):
            if not game.legal_moves(-game.turn):
                break
            game.turn = -game.turn
            continue
        started = time.perf_counter()
        try:
            x, y = bots[game.turn](game.board.copy(), game.turn, time_limit)
            x, y = int(x), int(y)
        except Exception as e:
            x, y = -1, -1
            if verbose:
                print(f'{"white" if game.turn == 1 else "black"} raised {e!r}, turn forfeited')
        elapsed = time.perf_counter() - started
        if elapsed > time_limit:
            if verbose:
                print(f'{"white" if game.turn == 1 else "black"} took {elapsed:.2f}s, turn forfeited')
        elif game.step(x, y, game.turn) <= 0:
            if verbose:
                print(f'{"white" if game.turn == 1 else "black"} played illegal move ({x}, {y}), turn forfeited')
        game.turn = -game.turn
    return int(np.count_nonzero(game.board == 1)), int(np.count_nonzero(game.board == -1))


def main():
    parser = argparse.ArgumentParser(description = 'Local Reversi match between two AI files.')
    parser.add_argument('first')
    parser.add_argument('second')
    parser.add_argument('--games', type = int, default = 1)
    parser.add_argument('--time-limit', type = float, default = 5.0)
    parser.add_argument('--quiet', action = 'store_true')
    args = parser.parse_args()

    first, second = 'A: ' + args.first, 'B: ' + args.second
    bots = {first: load_player(args.first), second: load_player(args.second)}
    wins = {first: 0, second: 0, 'draw': 0}
    for n in range(args.games):
        white_name, black_name = (first, second) if n % 2 == 0 else (second, first)
        white, black = play(bots[white_name], bots[black_name], args.time_limit, not args.quiet)
        winner = white_name if white > black else black_name if black > white else 'draw'
        wins[winner] += 1
        print(f'game {n + 1}: [{white_name}] (white) {white} : {black} [{black_name}] (black)  ->  {winner}')
    print(f'\n{first}: {wins[first]} wins, {second}: {wins[second]} wins, draws: {wins["draw"]}')


if __name__ == '__main__':
    main()
