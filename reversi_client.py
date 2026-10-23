#Zijie Zhang, Sep.24/2023
#Updated Sep.09/2026 with help of Claude
#Nine tables, tournament billboard and automatic reconnect added Oct.07/2026 with help of Claude
"""Reversi client. Log in to the arena, pick a chair in the lobby, and play with your AI or by hand.

    python reversi_client.py                       # plays with AI_logic.py
    python reversi_client.py --player my_bot.py    # plays with another AI file
    python reversi_client.py --player human        # you play by clicking squares (no time limit outside tournaments)
"""

import argparse
import importlib.util
import json
import os
import queue
import socket
import threading
import time
import traceback
from sys import exit

import numpy as np
import pygame

from reversi import reversi

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
MAX_MESSAGE_CHARS = 4 * 1024 * 1024
RECONNECT_WINDOW = 90          #seconds the client keeps trying to get back after losing the connection

#Lobby layout
TABLE_VIEW = pygame.Rect(16, 104, 796, 560)    #scrolling area with the table cards
SIDE = pygame.Rect(828, 104, 356, 680)         #billboard / history / online panel
CARD_W, CARD_H, GAP, COLUMNS = 380, 300, 12, 2
TAB_NAMES = ('Billboard', 'History', 'Online')

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
DARK_PIECE = (30, 30, 34)


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


def load_images():
    """Board background and the two piece pictures. Call after pygame.display.set_mode."""
    here = os.path.dirname(os.path.abspath(__file__))
    background = pygame.transform.smoothscale(pygame.image.load(os.path.join(here, 'data/background.jpeg')).convert(), (800, 800))
    white_piece = pygame.image.load(os.path.join(here, 'data/white_piece.png')).convert_alpha()
    black_piece = pygame.image.load(os.path.join(here, 'data/black_piece.png')).convert_alpha()
    return background, white_piece, black_piece


def enable_keepalive(sock) -> None:
    """Notice a dead connection (lost Wi-Fi, server restart) within about half a minute."""
    options = [(socket.SOL_SOCKET, 'SO_KEEPALIVE', 1), (socket.IPPROTO_TCP, 'TCP_KEEPIDLE', 10),
               (socket.IPPROTO_TCP, 'TCP_KEEPINTVL', 5), (socket.IPPROTO_TCP, 'TCP_KEEPCNT', 3)]
    for level, name, value in options:
        if hasattr(socket, name):
            try:
                sock.setsockopt(level, getattr(socket, name), value)
            except OSError:
                pass


class Network:
    """Connection to the arena, run by a background thread. Everything is reported through `inbox`:
    {'type': '_connected'}, {'type': '_connect_failed', 'reason': ...}, the server's messages, and None once disconnected."""

    def __init__(self) -> None:
        self.sock = None
        self.inbox = queue.Queue()
        self.connected = False
        self.connecting = False
        self.closed = False

    def connect(self, host, port) -> None:
        self.connecting = True
        threading.Thread(target = self._run, args = (host, port), daemon = True).start()

    def _run(self, host, port) -> None:
        try:
            sock = socket.create_connection((host, port), timeout = 5)
            sock.settimeout(None)
            enable_keepalive(sock)
        except OSError as e:
            self.connecting = False
            self.inbox.put({'type': '_connect_failed', 'reason': str(e) or type(e).__name__})
            return
        if self.closed:
            sock.close()
            return
        self.sock = sock
        self.connected, self.connecting = True, False
        self.inbox.put({'type': '_connected'})
        reader = sock.makefile('r', encoding = 'utf-8', errors = 'replace')
        while True:
            try:
                line = reader.readline(MAX_MESSAGE_CHARS)
            except (OSError, ValueError):
                line = ''
            if not line:
                self.connected = False
                self.inbox.put(None)
                return
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if isinstance(message, dict):
                self.inbox.put(message)

    def send(self, message) -> None:
        if not self.connected:
            return
        try:
            self.sock.sendall((json.dumps(message) + '\n').encode('utf-8'))
        except OSError:
            self.connected = False

    def close(self) -> None:
        self.closed = True
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
    """The game screen: board on the left, info panel on the right. Also holds the fonts for every screen."""

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
        rect = surface.get_rect(**{anchor: (x, y)})
        screen.blit(surface, rect)
        return rect

    @staticmethod
    def fit(s, font, width) -> str:
        if font.size(s)[0] <= width:
            return s
        while s and font.size(s + '...')[0] > width:
            s = s[:-1]
        return s + '...'

    @staticmethod
    def message_color(message) -> tuple:
        if 'forfeit' in message or 'left the game' in message or 'cancelled' in message or 'lost connection' in message:
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
        self.text(screen, 'TOURNAMENT GAME' if view.get('tournament') else 'REVERSI', x0 + w//2, 44, self.title_font, ACCENT)

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

    def draw(self, screen, font, active = False, color = None) -> None:
        background = color or (ACCENT if active else CARD_ACTIVE if self.enabled else CARD)
        pygame.draw.rect(screen, background, self.rect, border_radius = 10)
        label = font.render(self.label, True, PANEL if active or color else TEXT if self.enabled else DIM)
        screen.blit(label, label.get_rect(center = self.rect.center))

    def hit(self, pos) -> bool:
        return self.enabled and self.rect.collidepoint(pos)


#---- Side panel pieces, shared with tournament_admin.py ----

def draw_tabs(screen, g, area, names, selected) -> list:
    """Draws a row of tabs at the top of `area`. Returns [(name, rect)] for hit testing."""
    rects = []
    width = (area.w - 24 - 8 * (len(names) - 1)) // len(names)
    for i, name in enumerate(names):
        rect = pygame.Rect(area.x + 12 + i * (width + 8), area.y + 12, width, 36)
        active = name == selected
        pygame.draw.rect(screen, CARD_ACTIVE if active else PANEL, rect, border_radius = 10)
        if active:
            pygame.draw.rect(screen, ACCENT, rect, 2, border_radius = 10)
        g.text(screen, name, rect.centerx, rect.centery, g.small_font, ACCENT if active else MUTED)
        rects.append((name, rect))
    return rects


def tournament_shown(tournament) -> bool:
    return tournament is not None and (tournament['phase'] != 'normal' or tournament.get('final'))


def draw_standings(screen, g, area, tournament, scroll, me = None, large = False) -> int:
    """The billboard. Returns the height of the content, for scrolling."""
    if not tournament_shown(tournament):
        g.text(screen, 'No tournament yet', area.centerx, area.y + 40, g.small_font, MUTED)
        return 0
    phase = tournament['phase']
    rows = tournament['standings']
    row_h = 40 if large else 30
    font = g.font if large else g.small_font
    title = {'pre': 'Registration: everyone online when it starts', 'running': 'Live standings'}.get(phase, 'Final standings')
    rank_x, name_x = area.x + 8, area.x + (60 if large else 46)
    wdl_x, pts_x = area.right - (150 if large else 92), area.right - 10

    previous_clip = screen.get_clip()
    screen.set_clip(area)
    y = area.y - scroll
    g.text(screen, title, area.x + 4, y + 12, g.small_font, ACCENT, 'midleft')
    y += 30
    for label, x, anchor in (('#', rank_x, 'midleft'), ('Group', name_x, 'midleft'), ('W-D-L', wdl_x, 'center'), ('Pts', pts_x, 'midright')):
        g.text(screen, label, x, y + row_h // 2, g.small_font, DIM, anchor)
    y += row_h
    for row in rows:
        if area.y - row_h <= y <= area.bottom:
            if row['name'] == me:
                pygame.draw.rect(screen, CARD_ACTIVE, (area.x, y + 2, area.w, row_h - 4), border_radius = 8)
            podium = phase != 'pre' and row['rank'] <= 3 and row['points'] > 0
            g.text(screen, str(row['rank']), rank_x, y + row_h // 2, font, ACCENT if podium else MUTED, 'midleft')
            if phase != 'normal':
                pygame.draw.circle(screen, GOOD if row['online'] else DIM, (name_x - 12, y + row_h // 2), 4)
            g.text(screen, g.fit(row['name'], font, wdl_x - 36 - name_x), name_x, y + row_h // 2, font, TEXT, 'midleft')
            g.text(screen, f"{row['wins']}-{row['draws']}-{row['losses']}", wdl_x, y + row_h // 2, g.small_font, MUTED)
            g.text(screen, str(row['points']), pts_x, y + row_h // 2, g.font if large else g.font, ACCENT if podium else TEXT, 'midright')
        y += row_h
    if not rows:
        g.text(screen, 'Nobody is online', area.centerx, y + 20, g.small_font, DIM)
        y += 40
    screen.set_clip(previous_clip)
    return y + scroll - area.y


def draw_history(screen, g, area, tournament, scroll, me = None) -> int:
    """Finished tournament games, newest first. Returns the height of the content."""
    history = tournament.get('history') if tournament_shown(tournament) else None
    if not history:
        g.text(screen, 'Tournament games will be listed here', area.centerx, area.y + 40, g.small_font, MUTED)
        return 0
    row_h = 58
    previous_clip = screen.get_clip()
    screen.set_clip(area)
    y = area.y - scroll
    for game in reversed(history):
        if area.y - row_h <= y <= area.bottom:
            box = pygame.Rect(area.x, y, area.w, row_h - 6)
            pygame.draw.rect(screen, PANEL, box, border_radius = 8)
            if me in (game['white'], game['black']):
                pygame.draw.rect(screen, ACCENT, box, 1, border_radius = 8)
            if game['winner']:
                result, color = f"+{game['points']} {game['winner']}", GOOD
            else:
                result, color = 'draw', MUTED
            result = g.fit(result, g.small_font, box.w // 2)
            right = g.text(screen, result, box.right - 10, y + 15, g.small_font, color, 'midright')
            info = f"#{game['n']}  {game['time']}  Table {game['table']}" + (f"  {game['note']}" if game['note'] else '')
            g.text(screen, g.fit(info, g.small_font, right.x - box.x - 20), box.x + 10, y + 15, g.small_font, DIM, 'midleft')
            pygame.draw.circle(screen, TEXT, (box.x + 16, y + 38), 6)
            score = g.fit(f"{game['white']}  {game['ws']} : {game['bs']}  {game['black']}", g.small_font, box.w - 60)
            end = g.text(screen, score, box.x + 28, y + 38, g.small_font, TEXT, 'midleft').right
            pygame.draw.circle(screen, DARK_PIECE, (end + 12, y + 38), 6)
            pygame.draw.circle(screen, MUTED, (end + 12, y + 38), 6, 1)
        y += row_h
    screen.set_clip(previous_clip)
    return len(history) * row_h


def draw_online(screen, g, area, lobby, scroll, me = None) -> int:
    users = (lobby or {}).get('users', [])
    row_h = 30
    previous_clip = screen.get_clip()
    screen.set_clip(area)
    y = area.y - scroll
    g.text(screen, f'{len(users)} online', area.x + 4, y + 12, g.small_font, ACCENT, 'midleft')
    y += 30
    for user in users:
        if area.y - row_h <= y <= area.bottom:
            pygame.draw.circle(screen, GOOD, (area.x + 10, y + row_h // 2), 4)
            g.text(screen, user + ('  (you)' if user == me else ''), area.x + 24, y + row_h // 2, g.small_font,
                   ACCENT if user == me else TEXT, 'midleft')
        y += row_h
    screen.set_clip(previous_clip)
    return y + scroll - area.y


def winners_line(tournament) -> str:
    winners = [row for row in tournament.get('standings', []) if row['rank'] == 1 and row['points'] > 0]
    if not winners:
        return 'No points were scored'
    names = ', '.join(row['name'] for row in winners)
    return f"{'Winners' if len(winners) > 1 else 'Winner'}: {names} with {winners[0]['points']} points"


def draw_final_overlay(screen, g, tournament, me = None) -> None:
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 190))
    screen.blit(overlay, (0, 0))
    card = pygame.Rect(300, 50, 600, 700)
    pygame.draw.rect(screen, CARD, card, border_radius = 18)
    pygame.draw.rect(screen, ACCENT, card, 2, border_radius = 18)
    g.text(screen, 'TOURNAMENT OVER', card.centerx, card.y + 44, g.title_font, ACCENT)
    g.text(screen, g.fit(winners_line(tournament), g.font, card.w - 40), card.centerx, card.y + 88, g.font, TEXT)
    draw_standings(screen, g, pygame.Rect(card.x + 24, card.y + 116, card.w - 48, card.h - 170), tournament, 0, me, large = True)
    g.text(screen, 'Click anywhere to close', card.centerx, card.bottom - 26, g.small_font, MUTED)


def clamp(value, low, high):
    return max(low, min(high, value))


class App:
    def __init__(self, args) -> None:
        self.args = args
        self.choose_move = load_player(args.player)
        self.kind = 'human' if self.choose_move is None else 'ai'

        pygame.init()
        self.screen = pygame.display.set_mode((1200, 800))
        pygame.display.set_caption('Reversi')
        self.clock = pygame.time.Clock()
        self.background, white_piece, black_piece = load_images()
        self.game = drawable_reversi(white_piece, black_piece)

        self.net = Network()
        self.screen_name = 'login'          #'login' | 'lobby' | 'game'
        self.name = None
        self.token = None                   #given by the arena at login, used to take the name back after a drop
        self.name_input = args.name or ''
        self.login_error = ''
        self.login_pending = False
        self.reconnect_until = None         #set while trying to get back after a lost connection
        self.next_attempt = 0.0
        self.lobby = None
        self.phase = 'normal'
        self.tournament = None
        self.show_final = False
        self.you = {'table': None, 'seat': None, 'ready': False}
        self.state = None                   #last game_state from the arena
        self.seen_request = None
        self.deadline = None
        self.bot_results = queue.Queue()
        self.status = ('', 0.0, BAD)
        self.chair_rects = []
        self.tab = 'Online'
        self.tab_rects = []
        self.side_scroll = {name: 0 for name in TAB_NAMES}
        self.table_scroll = 0
        self.max_scroll = 0
        self.thumb = None                   #(track rect, thumb rect) when the table list scrolls
        self.drag = None                    #(mouse y, scroll) while dragging the scrollbar
        self.ready_button = Button((234, 678, 170, 46), 'I am ready')
        self.stand_button = Button((424, 678, 170, 46), 'Stand up')

    #---- helpers ----

    def notify(self, text, color = BAD) -> None:
        self.status = (text, time.monotonic(), color)

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
        self.reconnect_until = None
        self.name, self.token, self.lobby, self.state, self.tournament = None, None, None, None, None
        self.phase, self.show_final = 'normal', False
        self.you = {'table': None, 'seat': None, 'ready': False}

    def send_login(self, name) -> None:
        self.net.send({'type': 'login', 'name': name, 'password': self.args.password,
                       'player_kind': self.kind, 'token': self.token})

    #---- events ----

    def handle_event(self, event) -> None:
        if event.type == pygame.QUIT:
            self.net.close()
            pygame.quit()
            exit()
        if self.show_final:
            if event.type == pygame.KEYDOWN or (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1):
                self.show_final = False
            return
        if self.reconnect_until is not None and event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
            return
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
        self.login_error = ''
        self.login_pending = True
        if self.net.connected:
            self.send_login(self.name_input)
        elif not self.net.connecting:
            self.net.connect(self.args.host, self.args.port)

    def lobby_event(self, event) -> None:
        if event.type == pygame.MOUSEWHEEL:
            if SIDE.collidepoint(pygame.mouse.get_pos()):
                self.side_scroll[self.tab] -= event.y * 40
            else:
                self.table_scroll -= event.y * 60
            return
        if event.type == pygame.MOUSEBUTTONUP:
            self.drag = None
            return
        if event.type == pygame.MOUSEMOTION and self.drag is not None and self.thumb is not None:
            track, thumb = self.thumb
            start_y, start_scroll = self.drag
            travel = max(1, track.h - thumb.h)
            self.table_scroll = start_scroll + (event.pos[1] - start_y) * self.max_scroll / travel
            return
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return
        pos = event.pos
        for name, rect in self.tab_rects:
            if rect.collidepoint(pos):
                self.tab = name
                return
        if self.thumb is not None:
            track, thumb = self.thumb
            if thumb.inflate(10, 0).collidepoint(pos):
                self.drag = (pos[1], self.table_scroll)
                return
            if track.inflate(10, 0).collidepoint(pos):
                self.table_scroll += TABLE_VIEW.h if pos[1] > thumb.centery else -TABLE_VIEW.h
                return
        if TABLE_VIEW.collidepoint(pos):
            for index, seat, rect in self.chair_rects:
                if rect.collidepoint(pos):
                    if self.you['table'] == index and self.you['seat'] == seat:
                        self.net.send({'type': 'stand'})
                    else:
                        self.net.send({'type': 'sit', 'table': index, 'seat': seat})
                    return
        if self.ready_button.hit(pos):
            self.net.send({'type': 'ready', 'ready': not self.you['ready']})
        elif self.stand_button.hit(pos):
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
                self.on_disconnect()
                continue
            kind = message.get('type')
            if kind == '_connected':
                if self.reconnect_until is not None:
                    self.send_login(self.name)
                elif self.login_pending:
                    self.send_login(self.name_input)
            elif kind == '_connect_failed':
                self.net = Network()
                if self.reconnect_until is None:
                    self.login_pending = False
                    self.login_error = f"cannot reach {self.args.host}:{self.args.port} ({message.get('reason')})"
            elif kind == 'login_ok':
                self.on_login_ok(message)
            elif kind == 'login_error':
                reason = message.get('reason', 'login refused')
                if self.reconnect_until is None:
                    self.login_error = reason
                    self.login_pending = False
                elif 'already connected' in reason:
                    #The arena has not noticed the old connection is dead yet: try again shortly
                    self.net.close()
                    self.net = Network()
                    self.next_attempt = time.monotonic() + 3.0
                else:
                    self.reset_connection(reason)
            elif kind == 'lobby':
                self.lobby = message
                self.you = message.get('you', self.you)
                self.phase = message.get('phase', 'normal')
            elif kind == 'tournament':
                self.on_tournament(message)
            elif kind == 'game_state':
                self.on_game_state(message)
            elif kind == 'error':
                self.notify(message.get('reason', 'error'), BAD)
            elif kind == 'notice':
                self.notify(message.get('text', ''), ACCENT)

    def on_login_ok(self, message) -> None:
        self.name = message['name']
        self.token = message.get('token')
        self.login_pending = False
        if self.reconnect_until is not None:
            self.reconnect_until = None
            self.seen_request = None       #ask the AI again if the arena is still waiting for our move
            self.notify('Reconnected', GOOD)
            print('Reconnected')
            if self.screen_name == 'game' and self.state is not None and not self.state['game_over']:
                self.screen_name = 'lobby'  #the arena sends the game again right away if it is still running
        else:
            self.screen_name = 'lobby'

    def on_disconnect(self) -> None:
        self.net.close()
        self.net = Network()
        if self.name is None:
            self.login_pending = False
            if not self.login_error:
                self.login_error = 'connection to the server was lost'
            return
        if self.reconnect_until is None:
            self.reconnect_until = time.monotonic() + RECONNECT_WINDOW
            print('Connection lost, reconnecting...')
        self.next_attempt = time.monotonic() + 1.0

    def tick_reconnect(self) -> None:
        if self.reconnect_until is None or self.net.connecting or self.net.connected:
            return
        now = time.monotonic()
        if now > self.reconnect_until:
            self.reset_connection('lost the connection to the server and could not get back')
        elif now >= self.next_attempt:
            self.next_attempt = now + 3.0
            self.net.connect(self.args.host, self.args.port)

    def on_tournament(self, message) -> None:
        previous = self.tournament
        self.tournament = message
        was_running = previous is not None and previous['phase'] == 'running'
        if message['phase'] != 'normal' and (previous is None or previous['phase'] == 'normal'):
            self.tab = 'Billboard'
        elif previous is None and message.get('final'):
            self.tab = 'Billboard'
        if message.get('final') and was_running:
            self.show_final = True
            self.tab = 'Billboard'

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
        if self.show_final and self.tournament is not None:
            draw_final_overlay(self.screen, self.game, self.tournament, self.name)
        if self.reconnect_until is not None:
            self.draw_reconnecting()
        pygame.display.update()

    def draw_reconnecting(self) -> None:
        s, g = self.screen, self.game
        strip = pygame.Rect(300, 10, 600, 44)
        pygame.draw.rect(s, (90, 30, 30), strip, border_radius = 12)
        pygame.draw.rect(s, BAD, strip, 2, border_radius = 12)
        left = max(0, int(self.reconnect_until - time.monotonic()))
        g.text(s, f'Connection lost, reconnecting...  ({left}s)', strip.centerx, strip.centery, g.font, TEXT)

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
        if occupant:
            if not occupant.get('online', True):
                tag, tag_color = 'offline', BAD
            elif occupant['ready']:
                tag, tag_color = 'READY', GOOD
            else:
                tag, tag_color = 'human' if occupant['kind'] == 'human' else 'AI', MUTED
        if wide:
            g.text(s, seat.capitalize(), rect.x + 14, rect.centery, g.small_font, MUTED, 'midleft')
            if occupant:
                g.text(s, g.fit(occupant['name'], g.font, 110), rect.centerx + 10, rect.centery, g.font, TEXT)
                g.text(s, tag, rect.right - 14, rect.centery, g.small_font, tag_color, 'midright')
            else:
                g.text(s, 'empty', rect.centerx + 10, rect.centery, g.small_font, DIM)
        else:
            g.text(s, seat.capitalize(), rect.centerx, rect.y + 16, g.small_font, MUTED)
            if occupant:
                g.text(s, g.fit(occupant['name'], g.small_font, rect.w - 10), rect.centerx, rect.y + 48, g.small_font, TEXT)
                g.text(s, 'human' if occupant['kind'] == 'human' else 'AI', rect.centerx, rect.y + 72, g.small_font, MUTED)
                if tag in ('READY', 'offline'):
                    g.text(s, tag, rect.centerx, rect.bottom - 18, g.small_font, tag_color)
            else:
                g.text(s, 'empty', rect.centerx, rect.y + 48, g.small_font, DIM)
        if TABLE_VIEW.colliderect(rect):
            self.chair_rects.append((index, seat, rect))

    def draw_table(self, table, index, rect) -> None:
        s, g = self.screen, self.game
        mine = self.you['table'] == index
        status = table['status']
        pygame.draw.rect(s, CARD, rect, border_radius = 16)
        if mine:
            pygame.draw.rect(s, ACCENT, rect, 2, border_radius = 16)
        g.text(s, f'Table {index + 1}', rect.centerx, rect.y + 26, g.font, TEXT)
        if status == 'playing':
            label, color = 'Game in progress', BAD
        elif status == 'countdown':
            label, color = f"Starting in {table['countdown']}...", GOOD
        elif self.phase == 'pre':
            label, color = 'Closed until the tournament starts', DIM
        else:
            label, color = 'Waiting for players', MUTED
        g.text(s, label, rect.centerx, rect.y + 52, g.small_font, color)

        #Mini board between the two player chairs
        board = pygame.Rect(rect.centerx - 64, rect.y + 72, 128, 128)
        pygame.draw.rect(s, FELT if status != 'playing' else (16, 60, 32), board, border_radius = 6)
        for k in range(1, 8):
            pygame.draw.line(s, (60, 130, 80), (board.x + 16*k, board.y), (board.x + 16*k, board.bottom), 1)
            pygame.draw.line(s, (60, 130, 80), (board.x, board.y + 16*k), (board.right, board.y + 16*k), 1)
        for (r, c), color in (((3, 3), TEXT), ((4, 4), TEXT), ((3, 4), DARK_PIECE), ((4, 3), DARK_PIECE)):
            pygame.draw.circle(s, color, (board.x + 16*c + 8, board.y + 16*r + 8), 6)
        if status == 'playing':
            g.text(s, 'LIVE', board.centerx, board.centery, g.font, BAD)

        self.draw_chair(table, index, 'white', pygame.Rect(rect.x + 12, rect.y + 72, 100, 128), False)
        self.draw_chair(table, index, 'black', pygame.Rect(rect.right - 112, rect.y + 72, 100, 128), False)
        self.draw_chair(table, index, 'observer', pygame.Rect(rect.centerx - 120, rect.y + 214, 240, 50), True)
        g.text(s, 'You are at this table' if mine else 'White moves first', rect.centerx, rect.y + 282,
               g.small_font, ACCENT if mine else DIM)

        if self.phase == 'pre':
            shade = pygame.Surface(rect.size, pygame.SRCALPHA)
            shade.fill((24, 26, 32, 150))
            s.blit(shade, rect.topleft)
            g.text(s, 'CLOSED', board.centerx, board.centery, g.font, ACCENT)

    def draw_phase_strip(self) -> None:
        s, g = self.screen, self.game
        strip = pygame.Rect(16, 66, 1168, 30)
        if self.phase == 'pre':
            background, color = (70, 56, 20), ACCENT
            text = 'TOURNAMENT DECLARED  -  Tables are closed. Everyone online when the admin starts it is registered.'
        elif self.phase == 'running':
            background, color = (20, 60, 36), GOOD
            text = ('TOURNAMENT IN PROGRESS  -  Each pairing once per colour. The winner scores the piece difference, '
                    f"a wipe-out scores {(self.tournament or {}).get('wipe_points', 64)}.")
        else:
            background, color = CARD, MUTED
            text = 'Click an empty chair to sit. The left chair plays white and moves first. Click your own chair to stand up.'
        pygame.draw.rect(s, background, strip, border_radius = 8)
        g.text(s, g.fit(text, g.small_font, strip.w - 24), strip.centerx, strip.centery, g.small_font, color)

    def draw_lobby(self) -> None:
        s, g = self.screen, self.game
        s.fill(PANEL)
        g.text(s, 'REVERSI ARENA', 24, 36, g.title_font, ACCENT, 'midleft')
        g.text(s, f"{self.name}  ({'human' if self.kind == 'human' else 'AI'})", 1176, 36, g.font, TEXT, 'midright')
        self.draw_phase_strip()

        lobby = self.lobby or {'tables': [], 'users': []}
        tables = lobby['tables']

        #Table cards in a scrolling grid
        rows = (len(tables) + COLUMNS - 1) // COLUMNS
        content_h = rows * (CARD_H + GAP) - GAP + 8 if rows else 0
        self.max_scroll = max(0, content_h - TABLE_VIEW.h)
        self.table_scroll = clamp(self.table_scroll, 0, self.max_scroll)
        self.chair_rects = []
        s.set_clip(TABLE_VIEW)
        for index, table in enumerate(tables):
            row, col = divmod(index, COLUMNS)
            rect = pygame.Rect(TABLE_VIEW.x + 4 + col * (CARD_W + GAP),
                               TABLE_VIEW.y + 4 + row * (CARD_H + GAP) - int(self.table_scroll), CARD_W, CARD_H)
            if rect.colliderect(TABLE_VIEW):
                self.draw_table(table, index, rect)
        s.set_clip(None)
        if self.max_scroll > 0:
            track = pygame.Rect(TABLE_VIEW.right - 10, TABLE_VIEW.y + 4, 6, TABLE_VIEW.h - 8)
            thumb_h = max(40, int(track.h * TABLE_VIEW.h / content_h))
            thumb = pygame.Rect(track.x, track.y + int((track.h - thumb_h) * self.table_scroll / self.max_scroll), track.w, thumb_h)
            pygame.draw.rect(s, CARD, track, border_radius = 3)
            pygame.draw.rect(s, ACCENT if self.drag else MUTED, thumb, border_radius = 3)
            self.thumb = (track, thumb)
        else:
            self.thumb = None

        #Buttons and status under the tables
        seated = self.you['table'] is not None and self.you['table'] < len(tables)
        table_status = tables[self.you['table']]['status'] if seated else 'waiting'
        self.ready_button.enabled = seated and table_status != 'playing' and self.phase != 'pre'
        self.ready_button.label = 'Cancel ready' if self.you['ready'] else 'I am ready'
        self.stand_button.enabled = seated and not (table_status == 'playing' and self.you['seat'] != 'observer')
        self.ready_button.draw(s, g.font, active = self.you['ready'])
        self.stand_button.draw(s, g.font)
        text, when, color = self.status
        if text and time.monotonic() - when < 6:
            g.text(s, g.fit(text, g.font, TABLE_VIEW.w), TABLE_VIEW.centerx, 750, g.font, color)
        elif seated:
            g.text(s, f"You are at Table {self.you['table'] + 1} ({self.you['seat']})", TABLE_VIEW.centerx, 750, g.font, ACCENT)
        else:
            g.text(s, 'Pick a chair to join a table', TABLE_VIEW.centerx, 750, g.small_font, MUTED)
        g.text(s, f"{len(lobby.get('users', []))} online  -  scroll the tables with the mouse wheel",
               TABLE_VIEW.centerx, 780, g.small_font, DIM)

        self.draw_side_panel()

    def draw_side_panel(self) -> None:
        s, g = self.screen, self.game
        pygame.draw.rect(s, CARD, SIDE, border_radius = 16)
        self.tab_rects = draw_tabs(s, g, SIDE, TAB_NAMES, self.tab)
        content = pygame.Rect(SIDE.x + 12, SIDE.y + 60, SIDE.w - 24, SIDE.h - 72)
        scroll = self.side_scroll[self.tab]
        if self.tab == 'Billboard':
            height = draw_standings(s, g, content, self.tournament, scroll, self.name)
        elif self.tab == 'History':
            height = draw_history(s, g, content, self.tournament, scroll, self.name)
        else:
            height = draw_online(s, g, content, self.lobby, scroll, self.name)
        self.side_scroll[self.tab] = clamp(scroll, 0, max(0, height - content.h))

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
                'names': state['names'], 'footer': footer, 'tournament': state.get('tournament')}
        if state['game_over']:
            winner = state['winner']
            if winner == 'cancelled':
                view['banner'] = ('Game cancelled', 'It does not count  -  click to return to the lobby')
            else:
                title = 'Draw' if winner == 'draw' else f"{state['names'].get(winner) or name_of(WHITE if winner == 'white' else BLACK)} wins"
                view['banner'] = (title, f"White {state['white']} : {state['black']} Black   -   click to return to the lobby")
        self.game.render(self.screen, self.background, view)
        text, when, color = self.status
        if text and time.monotonic() - when < 6:
            pill = pygame.Rect(40, 730, 720, 44)
            pygame.draw.rect(self.screen, PANEL, pill, border_radius = 12)
            self.game.text(self.screen, self.game.fit(text, self.game.font, pill.w - 30), pill.centerx, pill.centery, self.game.font, color)

    #---- main loop ----

    def run(self) -> None:
        while True:
            for event in pygame.event.get():
                self.handle_event(event)
            self.poll_network()
            self.poll_bot()
            self.tick_reconnect()
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
