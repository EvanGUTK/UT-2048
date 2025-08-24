import json
import math
import os
import random
import sys
from pathlib import Path

import pygame

DEFAULT_SIZE   = 4
FPS            = 60
ANIM_MOVE_MS   = 140
ANIM_POP_MS    = 110
SPAWN_4_CHANCE = 0.10

UT_ORANGE        = (247, 127, 0)
UT_ORANGE_DARK   = (222, 114, 0)
UT_SMOKEY        = (102, 102, 102)
UT_SMOKEY_DARK   = (72, 72, 72)
UT_WHITE         = (255, 255, 255)
UT_PAPER         = (250, 248, 239)

THEMES = [
    {
        "name": "UT Orange",
        "BG_APP": UT_PAPER,
        "BG_BOARD": (187, 173, 160),
        "BG_CELL_EMPTY": (205, 193, 180),
        "TEXT_MAIN": (119, 110, 101),
        "TEXT_INV": (249, 246, 242),
        "PILL": (143, 131, 121),
        "BTN": UT_ORANGE,
        "BTN_HOVER": (255, 146, 30),
        "BTN_DISABLED": (196, 186, 178),
        "HEADER_TILE_BG": UT_ORANGE,
        "HEADER_TILE_FG": UT_WHITE,
    },
    {
        "name": "Dark",
        "BG_APP": (28, 29, 33),
        "BG_BOARD": (40, 42, 46),
        "BG_CELL_EMPTY": (55, 58, 64),
        "TEXT_MAIN": (220, 222, 225),
        "TEXT_INV": (250, 250, 250),
        "PILL": (80, 83, 90),
        "BTN": (90, 114, 255),
        "BTN_HOVER": (120, 140, 255),
        "BTN_DISABLED": (90, 90, 95),
        "HEADER_TILE_BG": (255, 122, 53),
        "HEADER_TILE_FG": (20, 20, 22),
    },
    {
        "name": "Neon",
        "BG_APP": (10, 10, 14),
        "BG_BOARD": (20, 20, 28),
        "BG_CELL_EMPTY": (35, 35, 45),
        "TEXT_MAIN": (240, 240, 255),
        "TEXT_INV": (10, 10, 14),
        "PILL": (55, 55, 85),
        "BTN": (57, 255, 20),
        "BTN_HOVER": (140, 255, 140),
        "BTN_DISABLED": (70, 70, 90),
        "HEADER_TILE_BG": (255, 20, 147),
        "HEADER_TILE_FG": (10, 10, 14),
    },
    {
        "name": "Pastel",
        "BG_APP": (248, 247, 252),
        "BG_BOARD": (225, 220, 233),
        "BG_CELL_EMPTY": (235, 232, 240),
        "TEXT_MAIN": (90, 85, 95),
        "TEXT_INV": (255, 255, 255),
        "PILL": (190, 182, 198),
        "BTN": (255, 179, 186),
        "BTN_HOVER": (255, 204, 212),
        "BTN_DISABLED": (210, 205, 215),
        "HEADER_TILE_BG": (255, 223, 186),
        "HEADER_TILE_FG": (90, 85, 95),
    },
]

TILE_SIZE   = 106
TILE_MARGIN = 12

SAVE_PATH = Path(__file__).with_name("2048_save.json")
SAVE_VERSION = 2

def ease_out_cubic(t): return 1 - (1 - t) ** 3
def ease_out_back(t):
    c1 = 1.70158; c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2

def fmt_score(n: int) -> str:
    if n >= 1_000_000: s = f"{n/1_000_000:.1f}m"
    elif n >= 1_000:   s = f"{n/1_000:.1f}k"
    else: return str(n)
    return s.replace(".0", "")

def atomic_write_json(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    os.replace(tmp, path)

def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def lerp_color(a, b, t):
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )

def make_linear_gradient(size, c_top, c_bot):
    w, h = size
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    for y in range(h):
        t = y / max(1, h - 1)
        col = lerp_color(c_top, c_bot, t)
        pygame.draw.line(surf, col, (0, y), (w, y))
    return surf

def fit_font_for_tile(font_name, value, max_w, max_h, bold=True, min_size=16, max_size=64):
    low, high = min_size, max_size
    best = min_size
    while low <= high:
        mid = (low + high) // 2
        f = pygame.font.SysFont(font_name, mid, bold=bold)
        w, h = f.size(value)
        if w <= max_w and h <= max_h:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return pygame.font.SysFont(font_name, best, bold=bold)

def draw_power_t(surface, rect, color_fg, color_bg=None, border_radius=8):
    x, y, w, h = rect
    if color_bg:
        pygame.draw.rect(surface, color_bg, rect, border_radius=border_radius)
    bar_h = int(h * 0.28)
    stem_w = int(w * 0.30)
    stem_h = int(h * 0.60)

    top_rect = pygame.Rect(x + int(w*0.06), y + int(h*0.10), int(w*0.88), bar_h)
    pygame.draw.rect(surface, color_fg, top_rect, border_radius=6)

    stem_x = x + (w - stem_w)//2
    stem_y = top_rect.bottom - int(bar_h*0.15)
    stem_rect = pygame.Rect(stem_x, stem_y, stem_w, stem_h)
    pygame.draw.rect(surface, color_fg, stem_rect, border_radius=6)

def make_app_icon():

    sz = 128
    icon = pygame.Surface((sz, sz), pygame.SRCALPHA)
    draw_power_t(icon, pygame.Rect(0,0,sz,sz), UT_WHITE, UT_ORANGE, border_radius=16)
    return icon

class Game2048:
    def __init__(self, size=DEFAULT_SIZE, rng=None):
        self.size = int(size)
        self.rng = rng or random.Random()
        self.score = 0
        self.best = 0
        self._undo_board = None
        self._undo_score = 0
        self.can_undo = False
        self.board = [[0]*self.size for _ in range(self.size)]
        self.theme_index = 0
        self._load_state_or_reset()

    def _load_state_or_reset(self):
        data = load_json(SAVE_PATH)
        if data and isinstance(data, dict) and data.get("version") == SAVE_VERSION:
            try:
                self.best  = int(data.get("best", 0))
                self.theme_index = int(data.get("theme_index", 0)) % len(THEMES)
                saved_size = int(data.get("size", self.size))
                if saved_size != self.size:
                    self.reset()
                    return
                board = data.get("board")
                score = int(data.get("score", 0))
                if self._valid_board(board):
                    self.board = [row[:] for row in board]
                    self.score = score
                    self.can_undo = False
                    self._undo_board = None
                    self._undo_score = 0
                    if all(v == 0 for row in self.board for v in row):
                        self._spawn(); self._spawn()
                    return
            except Exception:
                pass
        self.best = int(data.get("best", 0)) if isinstance(data, dict) else 0
        self.theme_index = int(data.get("theme_index", 0)) % len(THEMES) if isinstance(data, dict) else 0
        self.reset()

    def _valid_board(self, board):
        if not isinstance(board, list) or len(board) != self.size: return False
        for row in board:
            if not isinstance(row, list) or len(row) != self.size: return False
            for v in row:
                if not isinstance(v, int) or v < 0: return False
        return True

    def _save_state(self):
        try:
            atomic_write_json(SAVE_PATH, {
                "version": SAVE_VERSION,
                "best": self.best,
                "size": self.size,
                "score": self.score,
                "board": self.board,
                "theme_index": self.theme_index,
            })
        except Exception:
            pass

    def _save_best(self):
        self._save_state()

    def reset(self):
        self.board = [[0]*self.size for _ in range(self.size)]
        self.score = 0
        self.can_undo = False
        self._undo_board = None
        self._undo_score = 0
        self._spawn(); self._spawn()
        self._save_state()

    def snapshot_for_undo(self):
        self._undo_board = [row[:] for row in self.board]
        self._undo_score = self.score
        self.can_undo = True

    def use_undo(self):
        if not self.can_undo or self._undo_board is None:
            return False
        self.board = [row[:] for row in self._undo_board]
        self.score = self._undo_score
        self.can_undo = False
        self._save_state()
        return True

    def _spawn(self):
        empty = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r][c] == 0]
        if not empty: return None
        r, c = self.rng.choice(empty)
        self.board[r][c] = 4 if self.rng.random() < SPAWN_4_CHANCE else 2
        return (r, c)

    def can_move(self):
        if any(0 in row for row in self.board): return True
        n = self.size
        for r in range(n):
            for c in range(n):
                v = self.board[r][c]
                if r+1 < n and self.board[r+1][c] == v: return True
                if c+1 < n and self.board[r][c+1] == v: return True
        return False

    def move_and_get_motion(self, direction):
        n = self.size
        moved = False
        motions = []
        merged_target = [[False]*n for _ in range(n)]
        score_gain = 0

        if direction == 'left':
            dr, dc = 0, -1; rows = range(n); cols = range(n)
        elif direction == 'right':
            dr, dc = 0,  1; rows = range(n); cols = range(n-1, -1, -1)
        elif direction == 'up':
            dr, dc = -1, 0; rows = range(n); cols = range(n)
        elif direction == 'down':
            dr, dc =  1, 0; rows = range(n-1, -1, -1); cols = range(n)
        else:
            return False, [], None, [], 0

        for r in rows:
            for c in cols:
                v = self.board[r][c]
                if v == 0: continue
                sr, sc = r, c
                nr, nc = r, c
                while True:
                    tr, tc = nr + dr, nc + dc
                    if not (0 <= tr < n and 0 <= tc < n): break
                    if self.board[tr][tc] == 0:
                        nr, nc = tr, tc; continue
                    if self.board[tr][tc] == v and not merged_target[tr][tc]:
                        nr, nc = tr, tc
                    break
                if (nr, nc) != (sr, sc):
                    moved = True
                    if self.board[nr][nc] == v and not merged_target[nr][nc]:
                        merged_target[nr][nc] = True
                        motions.append({'start': (sr, sc), 'end': (nr, nc), 'value': v, 'merge': True})
                        self.board[sr][sc] = 0
                    else:
                        motions.append({'start': (sr, sc), 'end': (nr, nc), 'value': v, 'merge': False})
                        self.board[nr][nc], self.board[sr][sc] = self.board[sr][sc], 0

        merges_for_pop = []
        for r in range(n):
            for c in range(n):
                if merged_target[r][c]:
                    self.board[r][c] *= 2
                    score_gain += self.board[r][c]
                    merges_for_pop.append((r, c))

        spawn_pos = None
        if moved:
            self.score += score_gain
            if self.score > self.best:
                self.best = self.score
                self._save_best()
            spawn_pos = self._spawn()
            self._save_state()
        return moved, motions, spawn_pos, merges_for_pop, score_gain

class View:
    def __init__(self, game: Game2048):
        pygame.init()
        pygame.display.set_caption("2048 — UTK Edition 1.0.1")


        def sysfont(name, size, bold=False):
            try: return pygame.font.SysFont(name, size, bold=bold)
            except Exception: return pygame.font.Font(None, size)
        self.font_pill_label  = sysfont("arial", 16, True)
        self.font_pill_value  = sysfont("arial", 26, True)
        self.font_btn         = sysfont("arial", 22, True)
        self.font_tag         = sysfont("arial", 22, False)
        self.font_helper      = sysfont("arial", 16, False)
        self.font_header_num  = sysfont("arial", 40, True)  # used when needed


        self.SIDE_PAD = 20
        self.TOP_PAD  = 16
        self.TITLE_W  = 120
        self.TITLE_H  = 120
        label_h = self.font_pill_label.get_height()
        value_h = self.font_pill_value.get_height()
        min_pill_h = (8 + 6) + label_h + 4 + value_h
        self.CTRL_H = max(56, min_pill_h)
        self.CTRL_W = 120
        self.CTRL_GAP = 12
        self.header_h = max(self.TITLE_H, 2*self.CTRL_H + self.CTRL_GAP)


        TAGLINE_OFFSET = 14
        self.TAGLINE_Y = self.TOP_PAD + self.header_h + 6 + TAGLINE_OFFSET

        self.game = game


        self.GRID_PX = self.game.size * TILE_SIZE + (self.game.size + 1) * TILE_MARGIN
        self.BOARD_LEFT = self.SIDE_PAD
        self.BOARD_TOP  = self.TAGLINE_Y + self.font_tag.get_height() + 8
        self.WIN_W = self.GRID_PX + 2*self.SIDE_PAD
        self.WIN_H = self.BOARD_TOP + self.GRID_PX + 28
        self.screen = pygame.display.set_mode((self.WIN_W, self.WIN_H))


        pygame.display.set_icon(make_app_icon())

        self.clock = pygame.time.Clock()

        board_right = self.BOARD_LEFT + self.GRID_PX
        self.COL1_X = board_right - 2*self.CTRL_W - self.CTRL_GAP
        self.COL2_X = board_right - self.CTRL_W
        self.ROW_TOP_Y = self.TOP_PAD
        self.ROW_BOT_Y = self.TOP_PAD + self.CTRL_H + self.CTRL_GAP

        self.rect_title = pygame.Rect(self.BOARD_LEFT, self.ROW_TOP_Y, self.TITLE_W, self.TITLE_H)
        self.rect_score = pygame.Rect(self.COL1_X, self.ROW_TOP_Y, self.CTRL_W, self.CTRL_H)
        self.rect_best  = pygame.Rect(self.COL2_X, self.ROW_TOP_Y, self.CTRL_W, self.CTRL_H)
        self.rect_new   = pygame.Rect(self.COL1_X, self.ROW_BOT_Y, self.CTRL_W, self.CTRL_H)
        self.rect_undo  = pygame.Rect(self.COL2_X, self.ROW_BOT_Y, self.CTRL_W, self.CTRL_H)


        self.animations = []
        self.pop_events = []
        self.input_locked = False
        self.undo_hover = False
        self.new_hover  = False
        self.shake_until_ms = 0
        self.shake_amp_px = 3


    @property
    def theme(self):
        return THEMES[self.game.theme_index % len(THEMES)]

    def cycle_theme(self):
        self.game.theme_index = (self.game.theme_index + 1) % len(THEMES)
        self.game._save_state()

    def grid_to_px(self, r, c):
        x = self.BOARD_LEFT + TILE_MARGIN + c * (TILE_SIZE + TILE_MARGIN)
        y = self.BOARD_TOP  + TILE_MARGIN + r * (TILE_SIZE + TILE_MARGIN)
        return x, y

    def draw_header(self):
        t = self.theme
        self.screen.fill(t["BG_APP"])


        draw_power_t(self.screen, self.rect_title, t["HEADER_TILE_FG"], t["HEADER_TILE_BG"], border_radius=8)


        self._draw_pill(self.rect_score, "SCORE", fmt_score(self.game.score))
        self._draw_pill(self.rect_best,  "BEST",  fmt_score(self.game.best))


        self._draw_button(self.rect_new,  "NEW",  enabled=True,  hover=self.new_hover)
        can_undo = self.game.can_undo and not self.input_locked
        self._draw_button(self.rect_undo, "UNDO", enabled=can_undo, hover=(self.undo_hover and can_undo))


        tag = self.font_tag.render("Join the numbers and get to the 2048 tile!", True, t["TEXT_MAIN"])
        self.screen.blit(tag, tag.get_rect(center=(self.WIN_W//2, self.TAGLINE_Y)))


        helper = self.font_helper.render("Arrows/WASD to move • U to undo • R to restart • C to change theme • ESC to quit", True, t["TEXT_MAIN"])
        self.screen.blit(helper, helper.get_rect(center=(self.WIN_W//2, self.BOARD_TOP - 10)))


        ox = oy = 0
        now = pygame.time.get_ticks()
        if now < self.shake_until_ms:
            phase = (self.shake_until_ms - now) / ANIM_MOVE_MS * math.pi * 6
            ox = int(math.sin(phase) * self.shake_amp_px)

        pygame.draw.rect(self.screen, t["BG_BOARD"],
                         (self.BOARD_LEFT + ox, self.BOARD_TOP + oy, self.GRID_PX, self.GRID_PX),
                         border_radius=8)
        for r in range(self.game.size):
            for c in range(self.game.size):
                x, y = self.grid_to_px(r, c)
                pygame.draw.rect(self.screen, t["BG_CELL_EMPTY"], (x + ox, y + oy, TILE_SIZE, TILE_SIZE), border_radius=6)
        return ox, oy

    def _draw_pill(self, rect, label, value_text):
        t = self.theme
        pygame.draw.rect(self.screen, t["PILL"], rect, border_radius=8)
        lab = self.font_pill_label.render(label, True, t["TEXT_INV"])
        self.screen.blit(lab, lab.get_rect(midtop=(rect.centerx, rect.top + 8)))
        val = self.font_pill_value.render(value_text, True, t["TEXT_INV"])
        self.screen.blit(val, val.get_rect(midbottom=(rect.centerx, rect.bottom - 6)))

    def _draw_button(self, rect, text, enabled=True, hover=False):
        t = self.theme
        base = t["BTN"] if enabled else t["BTN_DISABLED"]
        if enabled and hover: base = t["BTN_HOVER"]
        pygame.draw.rect(self.screen, base, rect, border_radius=8)
        text_col = (255,255,255) if enabled else (240,240,240)
        label = self.font_btn.render(text, True, text_col)
        self.screen.blit(label, label.get_rect(center=rect.center))

    def draw_tiles(self, ox=0, oy=0):

        for r in range(self.game.size):
            for c in range(self.game.size):
                v = self.game.board[r][c]
                if not v: continue
                x, y = self.grid_to_px(r, c)
                self._draw_tile_with_gradient(v, x + ox, y + oy, TILE_SIZE, TILE_SIZE)


        now = pygame.time.get_ticks()
        for anim in self.animations:
            t = max(0, min(1, (now - anim['start_ms']) / ANIM_MOVE_MS))
            t_e = ease_out_cubic(t)
            (sr, sc) = anim['start']; (er, ec) = anim['end']
            sx, sy = self.grid_to_px(sr, sc); ex, ey = self.grid_to_px(er, ec)
            x = sx + (ex - sx) * t_e + ox; y = sy + (ey - sy) * t_e + oy
            self._draw_tile_with_gradient(anim['value'], x, y, TILE_SIZE, TILE_SIZE)


        for pop in self.pop_events:
            t = max(0, min(1, (now - pop['start_ms']) / ANIM_POP_MS))
            t_e = ease_out_back(t)
            r, c = pop['cell']; x, y = self.grid_to_px(r, c)
            v = self.game.board[r][c]
            if v == 0: continue
            scale = 0.6 + 0.4 * t_e; pad = (1 - scale) * TILE_SIZE / 2
            self._draw_tile_with_gradient(v, x + pad + ox, y + pad + oy, TILE_SIZE*scale, TILE_SIZE*scale)


        self.animations[:] = [a for a in self.animations if now - a['start_ms'] < ANIM_MOVE_MS]
        self.pop_events[:] = [p for p in self.pop_events if now - p['start_ms'] < ANIM_POP_MS]
        if not self.animations and not self.pop_events:
            self.input_locked = False

    def _tile_colors_for_value(self, v):

        base = {
            2:   ((238, 228, 218), (232, 220, 210)),
            4:   ((237, 224, 200), (230, 216, 194)),
            8:   ((242, 177, 121), (238, 165, 110)),
            16:  ((245, 149,  99), (240, 138,  90)),
            32:  ((246, 124,  95), (240, 110,  84)),
            64:  ((246,  94,  59), (235,  84,  48)),
            128: ((237, 207, 114), (232, 198, 105)),
            256: ((237, 204,  97), (232, 195,  90)),
            512: ((237, 200,  80), (230, 190,  72)),
            1024:((237, 197,  63), (230, 185,  55)),
            2048:((237, 194,  46), (230, 182,  42)),
        }
        if v in base:
            return base[v]

        t = min(1.0, (math.log2(v) - 11) / 4.0) if v > 2048 else 0.0
        top = lerp_color((60,58,50), (30,30,30), t)
        bot = lerp_color((50,48,42), (20,20,20), t)
        return (top, bot)

    def _draw_tile_with_gradient(self, v, x, y, w, h):
        t = self.theme
        top_col, bot_col = self._tile_colors_for_value(int(v))
        tile = make_linear_gradient((int(w), int(h)), top_col, bot_col)
        rect = pygame.Rect(int(x), int(y), int(w), int(h))
        # round corners by masking
        mask = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        pygame.draw.rect(mask, (255,255,255,255), (0,0,rect.w,rect.h), border_radius=6)
        tile.blit(mask, (0,0), special_flags=pygame.BLEND_RGBA_MULT)
        self.screen.blit(tile, rect.topleft)


        value = str(v)
        pad = 18
        max_w, max_h = rect.w - pad*2, rect.h - pad*2
        font = fit_font_for_tile("arial", value, max_w, max_h, bold=True, min_size=18, max_size=64)
        color = (t["TEXT_MAIN"] if v <= 4 else t["TEXT_INV"])
        text = font.render(value, True, color)
        self.screen.blit(text, text.get_rect(center=rect.center))


    def animate_move(self, motions, spawn_pos, merges):
        self.animations = [{
            'start': m['start'], 'end': m['end'], 'value': m['value'],
            'merge': m['merge'], 'start_ms': pygame.time.get_ticks()
        } for m in motions]
        self.input_locked = True
        base = pygame.time.get_ticks()
        if spawn_pos:
            self.pop_events.append({'cell': spawn_pos, 'type': 'spawn', 'start_ms': base + ANIM_MOVE_MS})
        for cell in merges:
            self.pop_events.append({'cell': cell, 'type': 'merge', 'start_ms': base + ANIM_MOVE_MS})


    def handle_mouse(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.new_hover  = self.rect_new.collidepoint(event.pos)
            self.undo_hover = self.rect_undo.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect_new.collidepoint(event.pos): self._new_game()
            elif self.rect_undo.collidepoint(event.pos) and self.game.can_undo and not self.input_locked:
                self._undo()

    def handle_keys(self, event):
        if event.key == pygame.K_c:
            self.cycle_theme()
            return
        if self.input_locked: return
        key = event.key
        if key == pygame.K_r: self._new_game()
        elif key == pygame.K_u: self._undo()
        elif key in (pygame.K_UP, pygame.K_w):    self._move('up')
        elif key in (pygame.K_RIGHT, pygame.K_d): self._move('right')
        elif key in (pygame.K_DOWN, pygame.K_s):  self._move('down')
        elif key in (pygame.K_LEFT, pygame.K_a):  self._move('left')

    def _move(self, dir_name):
        self.game.snapshot_for_undo()
        moved, motions, spawn_pos, merges, _ = self.game.move_and_get_motion(dir_name)
        if moved:
            self.animate_move(motions, spawn_pos, merges)
        else:
            self.shake_until_ms = pygame.time.get_ticks() + 120
            self.game.can_undo = False
            self.game._undo_board = None

    def _undo(self):
        if self.game.use_undo():
            self.animations.clear(); self.pop_events.clear(); self.input_locked = False

    def _new_game(self):
        self.game.reset()
        self.animations.clear(); self.pop_events.clear(); self.input_locked = False


    def game_loop(self):
        running = True
        while running:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    else:
                        self.handle_keys(event)
                else:
                    self.handle_mouse(event)

            ox, oy = self.draw_header()
            self.draw_tiles(ox, oy)
            pygame.display.flip()

            if not self.input_locked and not self.game.can_move():
                overlay = pygame.Surface((self.GRID_PX, self.GRID_PX), pygame.SRCALPHA)
                overlay.fill((0,0,0,140))
                self.screen.blit(overlay, (self.BOARD_LEFT, self.BOARD_TOP))
                msg = self.font_header_num.render("Game Over — R to restart", True, (255,255,255))
                self.screen.blit(msg, msg.get_rect(center=(self.BOARD_LEFT + self.GRID_PX//2,
                                                           self.BOARD_TOP + self.GRID_PX//2)))
                pygame.display.flip()


        self.game._save_state()
        pygame.quit()
        sys.exit()


def parse_args():

    size = DEFAULT_SIZE
    seed = None
    if len(sys.argv) >= 2:
        try: size = int(sys.argv[1])
        except: pass
    if len(sys.argv) >= 3:
        try: seed = int(sys.argv[2])
        except: seed = sys.argv[2]
    return size, seed

def main():
    size, seed = parse_args()
    rng = random.Random(seed) if seed is not None else random.Random()
    View(Game2048(size, rng)).game_loop()

if __name__ == "__main__":
    main()
