#Zijie Zhang, Sep.24/2023

import json
import numpy as np

class reversi:
    def __init__(self) -> None:
        self.board = np.zeros([8,8], dtype=int)

        self.board[3,4] = -1
        self.board[3,3] = 1
        self.board[4,3] = -1
        self.board[4,4] = 1
        self.white_count = 2
        self.black_count = 2
        self.directions = [
            [1,1],
            [1,0],
            [1,-1],
            [0,1],
            [0,-1],
            [-1,1],
            [-1,0],
            [-1,-1]
        ]

        self.turn = 1

    def step(self, x, y, piece = 1, commit = True) -> int:
        """Play `piece` (1 = white, -1 = black) at board[x, y].

        Returns the number of pieces flipped (> 0) for a legal move, or a negative code:
        -1 square already occupied, -2 out of bounds, -3 flips nothing.
        With commit=False the board is left untouched, so this doubles as a legality check.
        """

        #Out of bound (checked before touching the board: numpy would wrap negative indexes)
        if x < 0 or x > 7 or y < 0 or y > 7:
            return -2

        #Piece already exists
        if self.board[x,y] != 0:
            return -1

        flipped = 0
        for dx, dy in self.directions:
            cursor_x, cursor_y = x + dx, y + dy
            flip_list = []
            while 0 <= cursor_x <= 7 and 0 <= cursor_y <= 7:
                if self.board[cursor_x, cursor_y] == 0:
                    break
                elif self.board[cursor_x, cursor_y] == piece:
                    if commit:
                        for cx, cy in flip_list:
                            self.board[cx, cy] = piece
                    flipped += len(flip_list)
                    break
                else:
                    flip_list.append((cursor_x, cursor_y))
                    cursor_x, cursor_y = cursor_x + dx, cursor_y + dy

        #Illegal Move
        if flipped == 0:
            return -3

        if commit:
            self.board[x,y] = piece
            if piece == 1:
                self.white_count += 1
            else:
                self.black_count += 1
            self.white_count += flipped * piece
            self.black_count -= flipped * piece
        return flipped

    def legal_moves(self, piece) -> list:
        """All (x, y) where `piece` can legally play on the current board."""
        return [(x, y) for x in range(8) for y in range(8) if self.step(x, y, piece, commit = False) > 0]


#---- Network protocol shared by the server and the players ----
#Every message is one line of JSON terminated by '\n'.
#Server -> player : {"request_id": n, "turn": 1 | -1, "board": [[8 ints] * 8], "time_limit": seconds}
#                   {"turn": 0, "board": ..., "white": n, "black": n, "winner": "white" | "black" | "draw"}
#Player -> server : {"request_id": n, "x": int, "y": int}   (request_id echoes the request being answered)

MAX_MESSAGE_CHARS = 65536

def send_message(sock, message: dict) -> None:
    sock.sendall((json.dumps(message) + '\n').encode('utf-8'))

def recv_message(reader):
    """`reader` is sock.makefile('r'). Returns the decoded message, or None once the connection is closed."""
    line = reader.readline(MAX_MESSAGE_CHARS)
    if not line:
        return None
    return json.loads(line)
