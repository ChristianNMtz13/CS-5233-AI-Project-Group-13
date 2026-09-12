#Zijie Zhang, Sep.24/2023
#Updated Sep.09/2026 with help of Claude
"""Reversi client. Log in to the arena, pick a chair in the lobby, and play with your AI or by hand.

    python reversi_client.py                       # plays with AI_logic.py
    python reversi_client.py --player my_bot.py    # plays with another AI file
    python reversi_client.py --player human        # you play by clicking squares (no time limit)
"""

import argparse
import importlib.util
import os
import queue
import socket
import threading
import time
import traceback
from sys import exit

import numpy as np
import pygame

from reversi import reversi, send_message, recv_message

#---- Fill these in for your class (command line options override them) ----
SERVER_HOST = 'raplanka.ddns.net'
SERVER_PORT = 33333
PASSWORD = 'CS5233Fall2026'

WHITE = 1
BLACK = -1
FPS = 30
CELL = 100
PANEL_X, PANEL_W, PAD = 800, 400, 24
NAME_CHARS = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-')
TABLE_COUNT = 3

#Colours
PANEL = (24, 26, 32)
CARD = (40, 43, 52)
CARD_ACTIVE = (54, 58, 72)
ACCENT = (255, 196, 61)
TEXT = (238, 238, 242)
MUTED = (140, 146, 162)
DIM = (96, 100, 112)
GOOD = (88, 200, 120)
WARN = (255, 196, 61)
BAD = (232, 84, 84)
GRID = (225, 225, 225)
FLIP = (120, 200, 255)
FELT = (22, 94, 46)


def name_of(piece) -> str:
    return 'White' if piece == WHITE else 'Black'


def load_player(spec):
    """'human' -> None, otherwise the choose_move function of the given Python file."""
    if spec == 'human':
        return None
    path = spec if spec.endswith('.py') else spec + '.py'
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    module_spec = importlib.util.spec_from_file_location('student_ai', path)
    if module_spec is None:
        raise SystemExit(f'cannot load {spec}')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    if not callable(getattr(module, 'choose_move', None)):
        raise SystemExit(f'{path} must define choose_move(board, turn, time_limit)')
    return module.choose_move


class Network:
    """Socket connection to the arena. A thread reads lines into `inbox`; None in the inbox means disconnected."""

    def __init__(self) -> None:
        self.sock = None
        self.inbox = queue.Queue()
        self.connected = False

    def connect(self, host, port) -> None:
        self.sock = socket.create_connection((host, port), timeout = 5)
        self.sock.settimeout(None)
        self.connected = True
        threading.Thread(target = self._read_loop, daemon = True).start()

    def _read_loop(self) -> None:
        reader = self.sock.makefile('r', encoding = 'utf-8', errors = 'replace')
        while True:
            try:
                message = recv_message(reader)
            except (OSError, ValueError):
                message = None
            if message is None:
                self.connected = False
                self.inbox.put(None)
                return
            self.inbox.put(message)

    def send(self, message) -> None:
        try:
            send_message(self.sock, message)
        except OSError:
            self.connected = False

    def close(self) -> None:
        self.connected = False
        if self.sock is not None:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)   #tells the arena right away, even while the reader thread is blocked
            except OSError:
                pass
            try:
                self.sock.close()
            except OSError:
                pass


class drawable_reversi(reversi):
    """The game screen: board on the left, info panel on the right."""

    def __init__(self, _white_pic, _black_pic) -> None:
        super().__init__()
        self.white_pic = pygame.transform.smoothscale(_white_pic, (72, 72))
        self.black_pic = pygame.transform.smoothscale(_black_pic, (72, 72))
        self.white_icon = pygame.transform.smoothscale(_white_pic, (44, 44))
        self.black_icon = pygame.transform.smoothscale(_black_pic, (44, 44))
        self.title_font = self._font(34, True)
        self.big_font = self._font(40, True)
        self.font = self._font(22, True)
        self.small_font = self._font(18, False)
        self.hint = pygame.Surface((24, 24), pygame.SRCALPHA)
        pygame.draw.circle(self.hint, (*ACCENT, 130), (12, 12), 9)

    @staticmethod
    def _font(size, bold):
        return pygame.font.SysFont('segoeui,helvetica,arial,freesansbold', size, bold = bold)

    def text(self, screen, s, x, y, font = None, color = TEXT, anchor = 'center'):
        surface = (font or self.font).render(s, True, color)
        screen.blit(surface, surface.get_rect(**{anchor: (x, y)}))

    @staticmethod
    def fit(s, font, width) -> str:
        if font.size(s)[0] <= width:
            return s
        while s and font.size(s + '...')[0] > width:
            s = s[:-1]
        return s + '...'

    @staticmethod
    def message_color(message) -> tuple:
        if 'forfeit' in message or 'left the game' in message:
            return BAD
        if 'passes' in message or 'Game over' in message:
            return ACCENT
        return TEXT

    def draw_board(self, screen, background, view) -> None:
        screen.blit(background, (0, 0))
        for i in range(1, 8):
            pygame.draw.line(screen, GRID, (CELL*i, 0), (CELL*i, 800), 2)
            pygame.draw.line(screen, GRID, (0, CELL*i), (800, CELL*i), 2)

        #board[row, col] is drawn with col across and row down, so the window matches print(board)
        for piece, pic in ((WHITE, self.white_pic), (BLACK, self.black_pic)):
            rows, cols = np.where(self.board == piece)
            screen.blits([(pic, (c*CELL + 14, r*CELL + 14)) for r, c in zip(rows, cols)])

        for r, c in view['flipped']:
            pygame.draw.circle(screen, FLIP, (c*CELL + 50, r*CELL + 50), 40, 3)
        if view['last_move'] is not None:
            r, c = view['last_move']
            pygame.draw.circle(screen, ACCENT, (c*CELL + 50, r*CELL + 50), 42, 4)
        if view['hints']:
            for r, c in self.legal_moves(self.turn):
                screen.blit(self.hint, (c*CELL + 38, r*CELL + 38))

    def draw_card(self, screen, x, y, w, h, icon, label, count, active) -> None:
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(screen, CARD_ACTIVE if active else CARD, rect, border_radius = 12)
        if active:
            pygame.draw.rect(screen, ACCENT, rect, 3, border_radius = 12)
        screen.blit(icon, (x + 16, y + (h - 44)//2))
        self.text(screen, self.fit(label, self.font, w - 160), x + 76, y + h//2 - (10 if active else 0), self.font, TEXT, 'midleft')
        if active:
            self.text(screen, 'to move', x + 76, y + h//2 + 14, self.small_font, ACCENT, 'midleft')
        self.text(screen, str(count), x + w - 20, y + h//2, self.big_font, TEXT, 'midright')

    def draw_panel(self, screen, view) -> None:
        x0, w, pad = PANEL_X, PANEL_W, PAD
        pygame.draw.rect(screen, PANEL, (x0, 0, w, 800))
        self.text(screen, 'REVERSI', x0 + w//2, 44, self.title_font, ACCENT)

        white = int(np.count_nonzero(self.board == WHITE))
        black = int(np.count_nonzero(self.board == BLACK))
        names = view['names']
        white_label = f"White: {names['white']}" if names['white'] else 'White'
        black_label = f"Black: {names['black']}" if names['black'] else 'Black'
        playing = view['hints']
        self.draw_card(screen, x0 + pad, 84, w - 2*pad, 76, self.white_icon, white_label, white, playing and self.turn == WHITE)
        self.draw_card(screen, x0 + pad, 172, w - 2*pad, 76, self.black_icon, black_label, black, playing and self.turn == BLACK)

        #Timer
        y = 282
        self.text(screen, 'Time to move', x0 + pad, y, self.font, MUTED, 'midleft')
        bar = pygame.Rect(x0 + pad, y + 20, w - 2*pad, 12)
        pygame.draw.rect(screen, CARD, bar, border_radius = 6)
        if view['time_limit'] is None or view['time_left'] is None:
            self.text(screen, 'no limit' if playing else '-', x0 + w - pad, y, self.font, MUTED, 'midright')
        else:
            fraction = view['time_left'] / view['time_limit'] if view['time_limit'] > 0 else 1
            color = GOOD if fraction > 0.5 else WARN if fraction > 0.25 else BAD
            self.text(screen, f"{view['time_left']:.1f} s", x0 + w - pad, y, self.font, color, 'midright')
            if fraction > 0:
                pygame.draw.rect(screen, color, (bar.x, bar.y, max(12, int(bar.w * fraction)), bar.h), border_radius = 6)

        #Event log, newest at the bottom, older entries fade out
        y = 346
        pygame.draw.line(screen, CARD, (x0 + pad, y), (x0 + w - pad, y), 1)
        self.text(screen, 'Log', x0 + pad, y + 22, self.font, MUTED, 'midleft')
        if names.get('observer'):
            self.text(screen, self.fit(f"Observer: {names['observer']}", self.small_font, 240), x0 + w - pad, y + 22, self.small_font, MUTED, 'midright')
        entries = view['messages'][-12:]
        for k, message in enumerate(entries):
            age = len(entries) - 1 - k
            color = self.message_color(message) if age == 0 else MUTED if age < 4 else DIM
            self.text(screen, self.fit(message, self.small_font, w - 2*pad),
                      x0 + pad, y + 56 + k*30, self.small_font, color, 'midleft')

        if view.get('footer'):
            self.text(screen, self.fit(view['footer'], self.small_font, w - 2*pad), x0 + w//2, 776, self.small_font, MUTED)

    def draw_banner(self, screen, title, subtitle) -> None:
        overlay = pygame.Surface((800, 170), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 205))
        screen.blit(overlay, (0, 315))
        self.text(screen, self.fit(title, self.title_font, 760), 400, 380, self.title_font, ACCENT)
        if subtitle:
            self.text(screen, self.fit(subtitle, self.font, 760), 400, 432, self.font, TEXT)

    def render(self, screen, background, view) -> None:
        self.draw_board(screen, background, view)
        self.draw_panel(screen, view)
        if view['banner'] is not None:
            self.draw_banner(screen, *view['banner'])


class Button:
    def __init__(self, rect, label) -> None:
        self.rect = pygame.Rect(rect)
        self.label = label
        self.enabled = True

    def draw(self, screen, font, active = False) -> None:
        color = ACCENT if active else CARD_ACTIVE if self.enabled else CARD
        pygame.draw.rect(screen, color, self.rect, border_radius = 10)
        label = font.render(self.label, True, PANEL if active else TEXT if self.enabled else DIM)
        screen.blit(label, label.get_rect(center = self.rect.center))

    def hit(self, pos) -> bool:
        return self.enabled and self.rect.collidepoint(pos)


class App:
    def __init__(self, args) -> None:
        self.args = args
        self.choose_move = load_player(args.player)
        self.kind = 'human' if self.choose_move is None else 'ai'

        pygame.init()
        self.screen = pygame.display.set_mode((1200, 800))
        pygame.display.set_caption('Reversi')
        self.clock = pygame.time.Clock()
        here = os.path.dirname(os.path.abspath(__file__))
        self.background = pygame.transform.smoothscale(pygame.image.load(os.path.join(here, 'data/background.jpeg')).convert(), (800, 800))
        white_piece = pygame.image.load(os.path.join(here, 'data/white_piece.png')).convert_alpha()
        black_piece = pygame.image.load(os.path.join(here, 'data/black_piece.png')).convert_alpha()
        self.game = drawable_reversi(white_piece, black_piece)

        self.net = Network()
        self.screen_name = 'login'          #'login' | 'lobby' | 'game'
        self.name = None
        self.name_input = args.name or ''
        self.login_error = ''
        self.login_pending = False
        self.lobby = None
        self.you = {'table': None, 'seat': None, 'ready': False}
        self.state = None                   #last game_state from the arena
        self.seen_request = None
        self.deadline = None
        self.bot_results = queue.Queue()
        self.status = ('', 0.0)
        self.chair_rects = []
        self.ready_button = Button((430, 590, 160, 48), 'I am ready')
        self.stand_button = Button((610, 590, 160, 48), 'Stand up')

    #---- helpers ----

    def notify(self, text) -> None:
        self.status = (text, time.monotonic())

    def my_seat(self):
        if self.state is not None and not self.state['game_over'] or self.screen_name == 'game':
            for seat in ('white', 'black', 'observer'):
                if self.state is not None and self.state['names'].get(seat) == self.name:
                    return seat
        return self.you.get('seat')

    def my_piece(self):
        seat = self.my_seat()
        return WHITE if seat == 'white' else BLACK if seat == 'black' else None

    def reset_connection(self, reason) -> None:
        self.net.close()
        self.net = Network()
        self.screen_name = 'login'
        self.login_error = reason
        self.login_pending = False
        self.name, self.lobby, self.state = None, None, None
        self.you = {'table': None, 'seat': None, 'ready': False}

    #---- events ----

    def handle_event(self, event) -> None:
        if event.type == pygame.QUIT:
            self.net.close()
            pygame.quit()
            exit()
        if self.screen_name == 'login':
            self.login_event(event)
        elif self.screen_name == 'lobby':
            self.lobby_event(event)
        else:
            self.game_event(event)

    def login_event(self, event) -> None:
        if event.type != pygame.KEYDOWN or self.login_pending:
            return
        if event.key == pygame.K_RETURN:
            self.submit_login()
        elif event.key == pygame.K_BACKSPACE:
            self.name_input = self.name_input[:-1]
        elif event.unicode in NAME_CHARS and len(self.name_input) < 16:
            self.name_input += event.unicode

    def submit_login(self) -> None:
        if not self.name_input:
            self.login_error = 'the name cannot be empty'
            return
        if not self.net.connected:
            try:
                self.net.connect(self.args.host, self.args.port)
            except OSError as e:
                self.login_error = f'cannot reach {self.args.host}:{self.args.port} ({e})'
                return
        self.net.send({'type': 'login', 'name': self.name_input, 'password': self.args.password, 'player_kind': self.kind})
        self.login_pending = True
        self.login_error = ''

    def lobby_event(self, event) -> None:
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return
        for index, seat, rect in self.chair_rects:
            if rect.collidepoint(event.pos):
                if self.you['table'] == index and self.you['seat'] == seat:
                    self.net.send({'type': 'stand'})
                else:
                    self.net.send({'type': 'sit', 'table': index, 'seat': seat})
                return
        if self.ready_button.hit(event.pos):
            self.net.send({'type': 'ready', 'ready': not self.you['ready']})
        elif self.stand_button.hit(event.pos):
            self.net.send({'type': 'stand'})

    def game_event(self, event) -> None:
        state = self.state
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if state['game_over']:
                self.screen_name = 'lobby'
            elif self.my_seat() == 'observer':
                self.net.send({'type': 'stand'})
                self.screen_name = 'lobby'
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if state['game_over']:
                self.screen_name = 'lobby'
            elif self.choose_move is None and self.my_piece() == state['turn'] and event.pos[0] < 800:
                col, row = event.pos[0] // CELL, event.pos[1] // CELL
                if self.game.step(row, col, state['turn'], commit = False) > 0:
                    self.net.send({'type': 'move', 'request_id': state['request_id'], 'x': row, 'y': col})

    #---- messages from the arena ----

    def poll_network(self) -> None:
        while True:
            try:
                message = self.net.inbox.get_nowait()
            except queue.Empty:
                return
            if message is None:
                self.reset_connection('connection to the server was lost')
                continue
            kind = message.get('type')
            if kind == 'login_ok':
                self.name = message['name']
                self.login_pending = False
                self.screen_name = 'lobby'
            elif kind == 'login_error':
                self.login_error = message.get('reason', 'login refused')
                self.login_pending = False
            elif kind == 'lobby':
                self.lobby = message
                self.you = message.get('you', self.you)
            elif kind == 'game_state':
                self.on_game_state(message)
            elif kind == 'error':
                self.notify(message.get('reason', 'error'))

    def on_game_state(self, state) -> None:
        self.state = state
        self.game.board = np.array(state['board'])
        self.game.turn = state['turn']
        if not state['game_over']:
            self.screen_name = 'game'
        if state['game_over']:
            self.deadline = None
        elif state['request_id'] != self.seen_request:
            self.seen_request = state['request_id']
            self.deadline = None if state['time_limit'] is None else time.monotonic() + state['time_limit']
            if self.choose_move is not None and self.my_piece() == state['turn']:
                threading.Thread(target = self.run_bot, args = (state['request_id'], self.game.board.copy(),
                                 state['turn'], state['time_limit']), daemon = True).start()

    def run_bot(self, request_id, board, turn, time_limit) -> None:
        #Runs in its own thread so a slow choose_move never freezes the window
        try:
            move = self.choose_move(board, turn, time_limit)
            x, y = int(move[0]), int(move[1])
        except Exception:
            print('choose_move raised an exception, forfeiting this turn:')
            traceback.print_exc()
            x = y = -1
        self.bot_results.put((request_id, x, y))

    def poll_bot(self) -> None:
        while True:
            try:
                request_id, x, y = self.bot_results.get_nowait()
            except queue.Empty:
                return
            if self.state is not None and not self.state['game_over'] and request_id == self.state['request_id']:
                self.net.send({'type': 'move', 'request_id': request_id, 'x': x, 'y': y})
            else:
                print(f'choose_move answered too late, the turn was forfeited (request {request_id})')

    #---- drawing ----

    def draw(self) -> None:
        if self.screen_name == 'login':
            self.draw_login()
        elif self.screen_name == 'lobby':
            self.draw_lobby()
        else:
            self.draw_game()
        pygame.display.update()

    def draw_login(self) -> None:
        s, g = self.screen, self.game
        s.fill(PANEL)
        g.text(s, 'REVERSI ARENA', 600, 230, g.title_font, ACCENT)
        g.text(s, f'Server {self.args.host}:{self.args.port}', 600, 280, g.small_font, MUTED)
        box = pygame.Rect(400, 340, 400, 56)
        pygame.draw.rect(s, CARD, box, border_radius = 10)
        pygame.draw.rect(s, ACCENT, box, 2, border_radius = 10)
        cursor = '_' if (time.monotonic() * 2) % 2 < 1 and not self.login_pending else ''
        g.text(s, self.name_input + cursor, box.x + 16, box.centery, g.font, TEXT, 'midleft')
        g.text(s, 'Type your name and press Enter (letters, digits, _ and -)', 600, 430, g.small_font, MUTED)
        if self.login_pending:
            g.text(s, 'Connecting...', 600, 480, g.font, ACCENT)
        elif self.login_error:
            g.text(s, g.fit(self.login_error, g.font, 1100), 600, 480, g.font, BAD)
        player = 'you play by hand' if self.choose_move is None else f'AI from {os.path.basename(self.args.player)}'
        g.text(s, f'Player: {player}', 600, 540, g.small_font, MUTED)

    def draw_chair(self, table, index, seat, rect, wide) -> None:
        s, g = self.screen, self.game
        occupant = table['seats'].get(seat)
        mine = self.you['table'] == index and self.you['seat'] == seat
        pygame.draw.rect(s, CARD_ACTIVE if occupant else PANEL, rect, border_radius = 10)
        border = ACCENT if mine else GOOD if occupant and occupant['ready'] else DIM
        pygame.draw.rect(s, border, rect, 2, border_radius = 10)
        label = seat.capitalize() + (' (first)' if seat == 'white' and wide else '')
        if wide:
            g.text(s, seat.capitalize(), rect.x + 14, rect.centery, g.small_font, MUTED, 'midleft')
            if occupant:
                g.text(s, g.fit(occupant['name'], g.font, 110), rect.centerx + 10, rect.centery, g.font, TEXT)
                tag = 'READY' if occupant['ready'] else ('human' if occupant['kind'] == 'human' else 'AI')
                g.text(s, tag, rect.right - 14, rect.centery, g.small_font, GOOD if occupant['ready'] else MUTED, 'midright')
            else:
                g.text(s, 'empty', rect.centerx + 10, rect.centery, g.small_font, DIM)
        else:
            g.text(s, seat.capitalize(), rect.centerx, rect.y + 16, g.small_font, MUTED)
            if occupant:
                g.text(s, g.fit(occupant['name'], g.small_font, rect.w - 10), rect.centerx, rect.y + 48, g.small_font, TEXT)
                g.text(s, 'human' if occupant['kind'] == 'human' else 'AI', rect.centerx, rect.y + 72, g.small_font, MUTED)
                if occupant['ready']:
                    g.text(s, 'READY', rect.centerx, rect.bottom - 18, g.small_font, GOOD)
            else:
                g.text(s, 'empty', rect.centerx, rect.y + 48, g.small_font, DIM)
        self.chair_rects.append((index, seat, rect))

    def draw_lobby(self) -> None:
        s, g = self.screen, self.game
        s.fill(PANEL)
        g.text(s, 'REVERSI ARENA', 30, 40, g.title_font, ACCENT, 'midleft')
        g.text(s, f"{self.name}  ({'human' if self.kind == 'human' else 'AI'})", 1170, 40, g.font, TEXT, 'midright')

        lobby = self.lobby or {'tables': [{'status': 'waiting', 'countdown': None, 'seats': {}}] * TABLE_COUNT, 'users': []}
        self.chair_rects = []
        for index, table in enumerate(lobby['tables']):
            x0, y0 = 30 + index*385, 90
            area = pygame.Rect(x0, y0, 360, 420)
            pygame.draw.rect(s, CARD, area, border_radius = 16)
            status = table['status']
            g.text(s, f'Table {index + 1}', x0 + 180, y0 + 30, g.font, TEXT)
            if status == 'playing':
                label, color = 'Game in progress', BAD
            elif status == 'countdown':
                label, color = f"Starting in {table['countdown']}...", GOOD
            else:
                label, color = 'Waiting for players', MUTED
            g.text(s, label, x0 + 180, y0 + 58, g.small_font, color)

            #Mini board between the two player chairs
            board = pygame.Rect(x0 + 116, y0 + 96, 128, 128)
            pygame.draw.rect(s, FELT if status != 'playing' else (16, 60, 32), board, border_radius = 6)
            for k in range(1, 8):
                pygame.draw.line(s, (60, 130, 80), (board.x + 16*k, board.y), (board.x + 16*k, board.bottom), 1)
                pygame.draw.line(s, (60, 130, 80), (board.x, board.y + 16*k), (board.right, board.y + 16*k), 1)
            for (r, c), color in (((3, 3), TEXT), ((4, 4), TEXT), ((3, 4), (30, 30, 34)), ((4, 3), (30, 30, 34))):
                pygame.draw.circle(s, color, (board.x + 16*c + 8, board.y + 16*r + 8), 6)
            if status == 'playing':
                g.text(s, 'LIVE', board.centerx, board.centery, g.font, BAD)

            self.draw_chair(table, index, 'white', pygame.Rect(x0 + 10, y0 + 96, 98, 128), False)
            self.draw_chair(table, index, 'black', pygame.Rect(x0 + 252, y0 + 96, 98, 128), False)
            self.draw_chair(table, index, 'observer', pygame.Rect(x0 + 60, y0 + 250, 240, 56), True)
            g.text(s, 'White moves first', x0 + 180, y0 + 340, g.small_font, DIM)
            if self.you['table'] == index:
                g.text(s, 'You are at this table', x0 + 180, y0 + 380, g.small_font, ACCENT)

        seated = self.you['table'] is not None
        table_status = lobby['tables'][self.you['table']]['status'] if seated and self.lobby else 'waiting'
        self.ready_button.enabled = seated and table_status != 'playing'
        self.ready_button.label = 'Cancel ready' if self.you['ready'] else 'I am ready'
        self.stand_button.enabled = seated and not (table_status == 'playing' and self.you['seat'] != 'observer')
        self.ready_button.draw(s, g.font, active = self.you['ready'])
        self.stand_button.draw(s, g.font)

        g.text(s, g.fit('Online: ' + ', '.join(lobby.get('users', [])), g.small_font, 1140), 600, 680, g.small_font, MUTED)
        text, when = self.status
        if text and time.monotonic() - when < 4:
            g.text(s, g.fit(text, g.font, 1140), 600, 725, g.font, BAD)
        g.text(s, 'Click an empty chair to sit, click your chair to stand up. The game starts 3 s after everyone at the table is ready.',
               600, 770, g.small_font, DIM)

    def draw_game(self) -> None:
        state = self.state
        if self.deadline is None or state['time_limit'] is None:
            time_left, limit = None, None
        else:
            limit = state['time_limit']
            time_left = max(0.0, min(limit, self.deadline - time.monotonic()))
        seat = self.my_seat()
        if seat == 'observer':
            footer = 'You are observing  -  press Esc to leave the table'
        elif self.choose_move is None:
            footer = f'You play {seat} by clicking a square' if seat else ''
        else:
            footer = f'You play {seat} with {os.path.basename(self.args.player)}' if seat else ''
        view = {'time_left': time_left, 'time_limit': limit, 'messages': state['log'], 'last_move': state['last_move'],
                'flipped': state['flipped'], 'hints': not state['game_over'], 'banner': None,
                'names': state['names'], 'footer': footer}
        if state['game_over']:
            winner = state['winner']
            title = 'Draw' if winner == 'draw' else f"{state['names'].get(winner) or name_of(WHITE if winner == 'white' else BLACK)} wins"
            view['banner'] = (title, f"White {state['white']} : {state['black']} Black   -   click to return to the lobby")
        self.game.render(self.screen, self.background, view)

    #---- main loop ----

    def run(self) -> None:
        while True:
            for event in pygame.event.get():
                self.handle_event(event)
            self.poll_network()
            self.poll_bot()
            self.draw()
            self.clock.tick(FPS)


def parse_args():
    parser = argparse.ArgumentParser(description = 'Reversi arena client.')
    parser.add_argument('--host', default = SERVER_HOST)
    parser.add_argument('--port', type = int, default = SERVER_PORT)
    parser.add_argument('--password', default = PASSWORD)
    parser.add_argument('--name', default = '', help = 'pre-fill the user name')
    parser.add_argument('--player', default = 'AI_logic.py', help = 'Python file with choose_move(), or "human"')
    return parser.parse_args()


if __name__ == '__main__':
    App(parse_args()).run()
