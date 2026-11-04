"""
觀眾用的牌局畫面 (外接螢幕全螢幕顯示)

只顯示公開資訊：
  - 紅黃綠狀態燈 (綠 = 正常進行、黃 = 處理中 / 等待設定、紅 = 需要人處理)
  - 4 家剩餘張數 (牌背 + 數字)，輪到誰會發光，Pass 的人有標記
  - 出牌區已出過的總張數
  - 出牌區鏡頭即時畫面
不顯示任何人的手牌，也不顯示複雜文字。
"""
import math
import time

import cv2
import numpy as np

from config import TABLE_FULLSCREEN, TABLE_SCREEN_X, TABLE_SCREEN_Y, TABLE_SEATS

W, H = 1920, 1080
WINDOW = "BigTwo Table"

# 顏色 (BGR)
FELT = (40, 95, 30)
FELT_EDGE = (25, 60, 18)
RIM = (30, 45, 80)
CARD_BACK = (150, 60, 25)
CARD_BACK_DARK = (110, 40, 15)
WHITE = (245, 245, 245)
GOLD = (40, 200, 255)
GREY = (150, 150, 150)
LIGHTS = {'red': (40, 40, 235), 'yellow': (30, 210, 250), 'green': (60, 220, 60)}
LIGHT_ORDER = ['red', 'yellow', 'green']

SEAT_POS = {'bottom': (960, 905), 'top': (960, 165), 'left': (235, 540), 'right': (1685, 540)}
NAMES = {0: 'AI', 1: 'P2', 2: 'P3', 3: 'P4'}
CARD_W, CARD_H = 72, 104
FONT = cv2.FONT_HERSHEY_DUPLEX


def _rounded_rect(img, p1, p2, color, radius=12, thickness=-1):
    x1, y1 = p1
    x2, y2 = p2
    r = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1, cv2.LINE_AA)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        for (cx, cy), a in (((x1 + r, y1 + r), 180), ((x2 - r, y1 + r), 270),
                            ((x1 + r, y2 - r), 90), ((x2 - r, y2 - r), 0)):
            cv2.ellipse(img, (cx, cy), (r, r), a, 0, 90, color, thickness, cv2.LINE_AA)


_CARD_CACHE = {}


def _make_card(w, h):
    """畫一張牌背 (只畫一次後快取)：回傳 (圖, 遮罩)"""
    card = np.zeros((h, w, 3), np.uint8)
    mask = np.zeros((h, w), np.uint8)
    _rounded_rect(mask, (0, 0), (w - 1, h - 1), 255, 9)
    _rounded_rect(card, (0, 0), (w - 1, h - 1), WHITE, 9)
    inner = np.full((h - 10, w - 10, 3), CARD_BACK, np.uint8)
    for k in range(-h, w + h, 12):     # 菱形格花紋 (畫在內框裡，不會超出牌)
        cv2.line(inner, (k, 0), (k + h, h), CARD_BACK_DARK, 1, cv2.LINE_AA)
        cv2.line(inner, (k, h), (k + h, 0), CARD_BACK_DARK, 1, cv2.LINE_AA)
    inner_mask = np.zeros((h - 10, w - 10), np.uint8)
    _rounded_rect(inner_mask, (0, 0), (w - 11, h - 11), 255, 6)
    region = card[5:h - 5, 5:w - 5]
    region[inner_mask > 0] = inner[inner_mask > 0]
    _rounded_rect(card, (5, 5), (w - 6, h - 6), WHITE, 6, 2)
    return card, mask > 0


def _card_back(img, cx, cy, horizontal=False):
    w, h = (CARD_H, CARD_W) if horizontal else (CARD_W, CARD_H)
    if (w, h) not in _CARD_CACHE:
        _CARD_CACHE[(w, h)] = _make_card(w, h)
    card, mask = _CARD_CACHE[(w, h)]
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    shadow = img[y1 + 4:y1 + 4 + h, x1 + 3:x1 + 3 + w]
    shadow[mask[:shadow.shape[0], :shadow.shape[1]]] //= 2
    roi = img[y1:y1 + h, x1:x1 + w]
    roi[mask] = card[mask]


def _text_center(img, text, center, scale, color, thick):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    org = (int(center[0] - tw / 2), int(center[1] + th / 2))
    cv2.putText(img, text, (org[0] + 3, org[1] + 3), FONT, scale, (10, 20, 10), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _glow(img, center, reach, draw, color, strength=20):
    """柔光效果：只在 center 周圍 reach 範圍內處理 (整張 1080p 模糊在樹莓派上太慢)"""
    cx, cy = center
    x1, y1 = max(0, cx - reach), max(0, cy - reach)
    x2, y2 = min(W, cx + reach), min(H, cy + reach)
    layer = np.zeros((y2 - y1, x2 - x1, 3), np.uint8)
    draw(layer, color, -x1, -y1)          # draw(圖層, 顏色, x 位移, y 位移)
    blur = cv2.GaussianBlur(layer, (0, 0), strength)
    roi = img[y1:y2, x1:x2]
    cv2.add(roi, blur, roi)
    cv2.add(roi, layer // 2, roi)


def _background():
    img = np.full((H, W, 3), (18, 28, 14), np.uint8)
    cv2.ellipse(img, (960, 540), (830, 470), 0, 0, 360, RIM, -1, cv2.LINE_AA)
    cv2.ellipse(img, (960, 540), (800, 440), 0, 0, 360, FELT_EDGE, -1, cv2.LINE_AA)
    felt = np.zeros_like(img)
    cv2.ellipse(felt, (960, 540), (780, 420), 0, 0, 360, FELT, -1, cv2.LINE_AA)
    # 中間亮、邊緣暗
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt(((xx - 960) / 780.0) ** 2 + ((yy - 540) / 420.0) ** 2)
    shade = np.clip(1.25 - 0.45 * d, 0.6, 1.25)[..., None]
    mask = felt.any(axis=2)
    img[mask] = np.clip(felt[mask] * shade[mask], 0, 255).astype(np.uint8)
    return img


_BG = None


def _traffic_light(img, status):
    x1, y1, x2, y2 = 40, 36, 170, 386
    _rounded_rect(img, (x1 + 5, y1 + 6), (x2 + 5, y2 + 6), (5, 8, 5), 28)
    _rounded_rect(img, (x1, y1), (x2, y2), (25, 25, 25), 28)
    _rounded_rect(img, (x1, y1), (x2, y2), (70, 70, 70), 28, 3)
    for i, name in enumerate(LIGHT_ORDER):
        c = (105, 95 + i * 115)
        on = (name == status)
        color = LIGHTS[name]
        if on:
            pulse = 0.85 + 0.15 * math.sin(time.time() * 4) if name == 'red' else 1.0
            lit = tuple(int(v * pulse) for v in color)
            _glow(img, c, 120, lambda L, col, ox, oy: cv2.circle(L, (c[0] + ox, c[1] + oy), 46, col, -1, cv2.LINE_AA), lit, 22)
            cv2.circle(img, c, 46, lit, -1, cv2.LINE_AA)
            cv2.circle(img, (c[0] - 14, c[1] - 14), 12, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(img, c, 46, tuple(v // 6 for v in color), -1, cv2.LINE_AA)
        cv2.circle(img, c, 46, (60, 60, 60), 2, cv2.LINE_AA)


def _seat(img, player, count, is_turn, passed, winner):
    side = TABLE_SEATS.get(player, 'bottom')
    cx, cy = SEAT_POS[side]
    vertical = side in ('left', 'right')
    n = max(0, count)

    # 牌背扇形
    span = 330 if vertical else 560
    step = min(42, span / max(n - 1, 1))
    total = step * (n - 1) if n > 1 else 0
    for i in range(n):
        off = -total / 2 + i * step
        if vertical:
            _card_back(img, cx, int(cy + off), horizontal=True)
        else:
            _card_back(img, int(cx + off), cy)

    # 名字 + 張數徽章
    if side == 'bottom':
        badge = (cx, cy - 105)
    elif side == 'top':
        badge = (cx, cy + 105)
    elif side == 'left':
        badge = (cx + 125, cy)
    else:
        badge = (cx - 125, cy)

    if is_turn:
        r = 70 + 6 * math.sin(time.time() * 5)
        _glow(img, badge, 130, lambda L, col, ox, oy: cv2.circle(L, (badge[0] + ox, badge[1] + oy), int(r), col, 10, cv2.LINE_AA), GOLD, 14)
    cv2.circle(img, badge, 56, (20, 30, 20), -1, cv2.LINE_AA)
    cv2.circle(img, badge, 56, GOLD if is_turn else (200, 200, 200), 4, cv2.LINE_AA)
    _text_center(img, str(n), (badge[0], badge[1] + 6), 1.9, WHITE, 4)

    label_pos = {
        'bottom': (cx, cy + 92), 'top': (cx, cy - 92),
        'left': (cx, cy - 215), 'right': (cx, cy - 215),
    }[side]
    _text_center(img, NAMES[player], label_pos, 1.2, GOLD if is_turn else WHITE, 2)

    if passed and not winner:
        px, py = badge[0], badge[1] + 82 if side != 'bottom' else badge[1] - 82
        _rounded_rect(img, (px - 62, py - 24), (px + 62, py + 24), (70, 70, 70), 14)
        _text_center(img, "PASS", (px, py), 0.95, (220, 220, 220), 2)

    if winner:
        # 皇冠放在徽章旁邊 (上下兩家放右邊，左右兩家放上面)，不會蓋住中間的數字
        _crown(img, (badge[0] + 135, badge[1]) if side in ('top', 'bottom') else (badge[0], badge[1] - 110))


def _crown(img, c):
    x, y = c
    pts = np.array([[x - 60, y + 35], [x - 60, y - 15], [x - 30, y + 10], [x, y - 40],
                    [x + 30, y + 10], [x + 60, y - 15], [x + 60, y + 35]], np.int32)
    _glow(img, (x, y), 130, lambda L, col, ox, oy: cv2.fillPoly(L, [pts + (ox, oy)], col, cv2.LINE_AA), GOLD, 18)
    cv2.fillPoly(img, [pts], GOLD, cv2.LINE_AA)
    for px in (x - 60, x, x + 60):
        cv2.circle(img, (px, y - 15 if px != x else y - 40), 9, (60, 60, 230), -1, cv2.LINE_AA)


def _pile(img, played_total):
    cx, cy = 960, 520
    rng = np.random.default_rng(7)
    shown = min(played_total, 8)
    for i in range(shown):
        dx, dy = rng.integers(-40, 41), rng.integers(-24, 25)
        _card_back(img, cx + int(dx), cy + int(dy), horizontal=bool(i % 2))
    if played_total == 0:
        _rounded_rect(img, (cx - 60, cy - 80), (cx + 60, cy + 80), (60, 120, 50), 12, 3)
    cv2.circle(img, (cx, cy + 135), 50, (20, 30, 20), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy + 135), 50, (200, 200, 200), 3, cv2.LINE_AA)
    _text_center(img, str(played_total), (cx, cy + 141), 1.6, WHITE, 3)


def _camera_inset(img, frame):
    x1, y1, w, h = 1408, 786, 480, 270
    _rounded_rect(img, (x1 - 10, y1 - 10), (x1 + w + 10, y1 + h + 10), (30, 30, 30), 16)
    if frame is not None and frame.size:
        img[y1:y1 + h, x1:x1 + w] = cv2.resize(frame, (w, h))
    else:
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (15, 15, 15), -1)
        _text_center(img, "--", (x1 + w // 2, y1 + h // 2), 2.0, GREY, 3)
    _rounded_rect(img, (x1 - 10, y1 - 10), (x1 + w + 10, y1 + h + 10), (90, 90, 90), 16, 3)
    cv2.circle(img, (x1 + 22, y1 + 22), 9, (40, 40, 235), -1, cv2.LINE_AA)   # 錄影中紅點


def render(state, cam_frame):
    """
    state: {
      'status': 'green' | 'yellow' | 'red',
      'counts': [AI, P2, P3, P4 剩餘張數],
      'played_total': 已出過張數,
      'turn': 輪到誰 (0~3) 或 None,
      'passed': {這一輪已 Pass 的玩家},
      'winner': 0~3 或 None,
    }
    """
    global _BG
    if _BG is None:
        _BG = _background()
    img = _BG.copy()
    _pile(img, state.get('played_total', 0))
    winner = state.get('winner')
    for p in range(4):
        _seat(img, p, state['counts'][p], state.get('turn') == p and winner is None,
              p in state.get('passed', ()), winner == p)
    _traffic_light(img, state.get('status', 'yellow'))
    _camera_inset(img, cam_frame)
    return img


class TableDisplay:
    def __init__(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        if TABLE_SCREEN_X or TABLE_SCREEN_Y:
            cv2.moveWindow(WINDOW, TABLE_SCREEN_X, TABLE_SCREEN_Y)
        if TABLE_FULLSCREEN:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(WINDOW, 1280, 720)

    def show(self, state, cam_frame):
        cv2.imshow(WINDOW, render(state, cam_frame))
