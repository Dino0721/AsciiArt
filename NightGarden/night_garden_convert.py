"""
Night Garden
------------
A night garden, made entirely of text.
Glowing hands, a 3D lotus on water, an eye in a moon portal, and moonlit
waves are drawn with Python. Every frame is then turned into live ASCII
art, and the real code doing it scrolls in the editor panel at the bottom.

Run
  python night_garden.py                      # pop-out window (9:16, TikTok shape)
  python night_garden.py --record garden.mp4  # save a 20 s video for posting
  python night_garden.py --font-size 8        # smaller characters = more detail
  python night_garden_convert.py --detail      # shape-matched characters (ascii_detail.py)

Keys
  F11 fullscreen   resize the window freely (it scales to fit)
  Z zoom in   X zoom out   0 reset zoom (or the mouse wheel)
  SPACE pause   RIGHT next scene   E outline strength   S screenshot   Q / ESC quit
"""
import argparse
import keyword
import math
import os
import re
import sys
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import cv2
import numpy as np
import pygame

# AsciiArt is split into folders - let this file find Common/ and Bouquet/
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ("Common", "Bouquet"):
    if _os.path.join(_ROOT, _d) not in _sys.path:
        _sys.path.insert(0, _os.path.join(_ROOT, _d))

from window_util import ViewZoom, Window, zoom_image
import bouquet_3d as b3d                     # 3D petal + depth-sorted drawing
from ascii_detail import GlyphMatcher         # shape-matched "detail" characters

WIN_W, WIN_H = 540, 960                      # 9:16 like a phone video
ART_H = 610                                  # top: ASCII art, bottom: code
SW, SH = 432, 488                            # scene coordinates (same shape as the art area)
RES = 1 if "--detail" in sys.argv else 2     # draw at 1/RES size (detail mode needs full size)
AW, AH = SW // RES, SH // RES                # actual pixel size of the scene image
SCENE_LEN, FADE = 5.0, 1.0

YY, XX = (np.mgrid[0:AH, 0:AW] * RES).astype(np.float32)
_rng = np.random.default_rng(5)


def P(x, y):
    """Scene coordinates -> pixel of the (smaller) scene image."""
    return (int(round(x / RES)), int(round(y / RES)))


def L(v):
    """Scene length -> pixels."""
    return max(1, int(round(v / RES)))


def _norm01(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-6)


NOISE = (0.45 * _norm01(cv2.GaussianBlur(_rng.random((AH, AW)).astype(np.float32), (0, 0), 1.0))
         + 0.55 * _norm01(cv2.GaussianBlur(_rng.random((AH, AW)).astype(np.float32), (0, 0), 18 / RES)))
STAR_X = _rng.integers(0, AW, 80)
STAR_Y = _rng.integers(0, AH, 80)
STAR_P = _rng.random(80) * 6.28
VIGNETTE = np.clip(1 - 0.9 * ((XX / SW - 0.5) ** 2 + (YY / SH - 0.5) ** 2), 0, 1)[..., None]


def mixc(a, b, t):
    """Blend two colours by t (array or float)."""
    a, b = np.array(a, np.float32), np.array(b, np.float32)
    t = np.asarray(t, np.float32)
    if t.ndim:
        t = t[..., None]
    return a + (b - a) * t


def sky(t, top=(4, 8, 24), bottom=(10, 24, 55)):
    """Night sky: gradient + drifting texture + twinkling stars."""
    img = mixc(top, bottom, YY / SH)
    tex = np.roll(NOISE, int(t * 12 / RES) % AH, axis=0)
    img *= (0.55 + 0.9 * tex)[..., None]
    tw = (0.5 + 0.5 * np.sin(t * 2.0 + STAR_P)) * 190
    img[STAR_Y, STAR_X] += tw[:, None]
    return img


def glow(img, cx, cy, r, color, k):
    d2 = (XX - cx) ** 2 + (YY - cy) ** 2
    img += np.array(color, np.float32) * (k * np.exp(-d2 / (r * r)))[..., None]


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


# ================================================================ scenes ===
def _seg(m, p0, p1, w):
    a, b = P(*p0), P(*p1)
    cv2.line(m, a, b, 1.0, L(w), cv2.LINE_AA)
    cv2.circle(m, b, L(w / 2), 1.0, -1, cv2.LINE_AA)


def hand_mask(dy, curl):
    """Silhouette of a left hand, palm up, fingers reaching to the centre."""
    m = np.zeros((AH, AW), np.float32)
    cx, cy = 148.0, 392.0 + dy
    _seg(m, (60, SH + 70), (cx - 15, cy + 50), 64)                     # forearm
    cv2.ellipse(m, P(cx, cy), (L(42), L(50)), 15, 0, 360, 1.0, -1, cv2.LINE_AA)
    for i, (a, lm) in enumerate(((-128, 0.74), (-108, 0.93), (-88, 1.0), (-68, 0.9))):
        ang = math.radians(a + 12)
        p = (cx + math.cos(ang) * 38, cy + math.sin(ang) * 46)
        d, w = math.radians(-100 + i * 9), 21.0 - i * 0.8
        for seg in (30, 24, 18):                                         # 3 finger joints
            q = (p[0] + math.cos(d) * seg * lm, p[1] + math.sin(d) * seg * lm)
            _seg(m, p, q, w)
            p, w, d = q, w * 0.88, d + math.radians(curl)
    p, d, w = (cx + 34, cy + 10), math.radians(-60), 21.0               # thumb
    for seg in (24, 18):
        q = (p[0] + math.cos(d) * seg, p[1] + math.sin(d) * seg)
        _seg(m, p, q, w)
        p, w, d = q, w * 0.85, d - math.radians(curl * 0.8)
    return m


def scene_hands(t, lt):
    img = sky(t)
    ox, oy = SW / 2, 128 + 6 * math.sin(t * 1.5)
    curl = 11 + 5 * math.sin(t * 1.6)
    m = np.maximum(hand_mask(4 * math.sin(t * 1.1), curl),
                   np.fliplr(hand_mask(4 * math.sin(t * 1.1 + 1.3), curl)))
    # fake 3D: blur the silhouette into a height map, light it from the orb
    b = cv2.GaussianBlur(m, (0, 0), 6 / RES)
    gx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3) * (10 / RES)
    gy = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3) * (10 / RES)
    lx, ly, lz = ox - XX, oy - YY, np.full_like(XX, 140.0)
    ln = np.sqrt(lx * lx + ly * ly + lz * lz)
    nn = np.sqrt(gx * gx + gy * gy + 1)
    diff = np.clip((-gx * lx - gy * ly + lz) / (ln * nn), 0, 1)
    fall = 0.45 + 0.9 * np.exp(-((XX - ox) ** 2 + (YY - oy) ** 2) / 270 ** 2)
    hand = mixc((10, 18, 40), (240, 155, 118), np.clip(diff * fall * 1.1, 0, 1.15))
    img = img * (1 - m[..., None]) + hand * m[..., None]
    # the glowing orb + rising sparks
    glow(img, ox, oy, 120, (60, 110, 230), 0.55)
    glow(img, ox, oy, 34, (170, 210, 255), 1.0)
    glow(img, ox, oy, 10, (255, 255, 255), 2.0)
    sparks = np.zeros_like(img)
    for k in range(30):
        life = (t * 0.3 + k / 30) % 1.0
        x = ox + math.sin(k * 2.1 + t * 0.9) * (20 + 70 * life)
        y = oy + 70 - life * 190
        c = 230 * (1 - life)
        cv2.circle(sparks, P(x, y), 1, (c * 0.8, c * 0.9, c), -1, cv2.LINE_AA)
    return img + sparks


class Cam:
    """Tiny camera with the interface bouquet_3d's drawing helpers expect."""

    def __init__(self, cx, cy, f, dist, yaw, pitch):
        self.cx, self.cy, self.f, self.dist = cx, cy, f, dist
        self.cyw, self.syw = math.cos(yaw), math.sin(yaw)
        self.cp, self.sp = math.cos(pitch), math.sin(pitch)

    def rot(self, p):
        x, y, z = p
        x1 = x * self.cyw + z * self.syw
        z1 = -x * self.syw + z * self.cyw
        return (x1, y * self.cp - z1 * self.sp, y * self.sp + z1 * self.cp)

    def proj(self, p, off=(0, 0, 0)):
        x1, y2, z2 = self.rot((p[0] + off[0], p[1] + off[1], p[2] + off[2]))
        s = self.f / (self.dist + z2)
        return (self.cx + x1 * s, self.cy - y2 * s, z2, s)


LOTUS = [  # count, open, closed, length, width, curl, cup, colour mult
    (6, 0.30, 0.08, 0.70, 0.34, 0.10, 0.9, 0.85),
    (8, 0.70, 0.12, 0.92, 0.38, 0.15, 0.8, 0.95),
    (10, 1.05, 0.18, 1.02, 0.38, 0.20, 0.65, 1.05),
    (12, 1.40, 0.30, 1.00, 0.36, 0.12, 0.5, 1.12),
]


def scene_lotus(t, lt):
    img = sky(t, (3, 8, 22), (6, 22, 45))
    hy = 280.0
    water = YY >= hy
    dz = np.maximum(YY - hy, 0) + 1
    rip = 0.5 + 0.5 * np.sin(dz ** 0.8 * 1.3 - t * 2.4 + 0.5 * np.sin(XX * 0.04 + t))
    col = mixc((3, 20, 38), (40, 130, 150), rip ** 6 * np.clip(dz / 120, 0.2, 1))
    streak = np.exp(-((XX - SW / 2) / (8 + dz * 0.5)) ** 2) * rip ** 3 * 0.8
    col += np.array((120, 200, 230), np.float32) * streak[..., None]
    img = np.where(water[..., None], col * (0.6 + 0.6 * NOISE[..., None]), img)

    surf = pygame.surfarray.make_surface(np.clip(img, 0, 255).astype(np.uint8).swapaxes(0, 1))
    cam = Cam(AW / 2, (hy + 18) / RES, 700 / RES, 560, t * 0.35, -0.45)
    items = []
    A = (0.0, 1.0, 0.0)
    e1, e2 = b3d.basis(A)
    for (px, pz, r) in ((0, 0, 95), (-150, 70, 60), (140, -60, 55), (60, 150, 45)):
        pts = []
        for i in range(22):
            a = 0.35 + (2 * math.pi - 0.5) * i / 21              # leave a notch
            pts.append((px + math.cos(a) * r, -2, pz + math.sin(a) * r))
        pts.append((px, -2, pz))
        b3d.add_poly(items, cam, pts, A, (25, 100, 80), amb=0.45)
    bloom = b3d.ease_out_back((lt + FADE) / 2.6)
    c = (0.0, 6 + 3 * math.sin(t * 1.2), 0.0)
    S = 92
    for li, (cnt, th_o, th_c, ln, wd, cu, cup, cm) in enumerate(LOTUS):
        th = th_c + (th_o - th_c) * bloom
        plen = S * ln * (0.5 + 0.5 * min(bloom, 1.05))
        tip = b3d.shade((150, 228, 232), cm)
        inner = b3d.shade((20, 105, 140), cm)
        for k in range(cnt):
            phi = 2 * math.pi * k / cnt + li * 0.45
            for pts, n, colr in b3d.petal(c, A, e1, e2, phi, th, plen, S * wd, cu * bloom, cup, tip, inner):
                b3d.add_poly(items, cam, pts, n, colr, amb=0.45)
    pts, n, colr = b3d.disc(c, A, e1, e2, S * 0.16, S * 0.18, (230, 210, 90))
    b3d.add_poly(items, cam, pts, n, colr, amb=0.6)
    b3d.draw_items(surf, items)
    img = pygame.surfarray.array3d(surf).swapaxes(0, 1).astype(np.float32)
    glow(img, SW / 2, hy - 30, 140, (40, 120, 160), 0.35)
    return img


def scene_eye(t, lt):
    img = sky(t)
    pcx, pcy, R = SW / 2, 172.0, 118.0
    d = np.sqrt((XX - pcx) ** 2 + (YY - pcy) ** 2)
    a = np.clip(R - d, 0, 2)[..., None] / 2
    portal = mixc((228, 142, 118), (40, 100, 160), np.clip(d / R, 0, 1) ** 1.6)
    img = img * (1 - a) + portal * a
    img += np.array((110, 190, 255), np.float32) * (np.exp(-((d - R) / 3) ** 2) * 0.9)[..., None]
    img += np.array((30, 80, 150), np.float32) * (np.exp(-((d - R) / 28) ** 2) * 0.5)[..., None]

    # eye inside the portal (blinks, looks around)
    ex, ey, ew = pcx, 178.0, 84.0
    cyc = lt % 3.3
    openness = 1 - (math.sin(math.pi * (cyc - 3.0) / 0.3) if cyc > 3.0 else 0)
    eh = 36 * openness + 1
    u = (XX - ex) / ew
    shape = np.clip(1 - u * u, 0, 1)
    top, bot = ey - eh * shape, ey + eh * 0.8 * shape
    m = np.clip(np.minimum(YY - top, bot - YY) / 1.5, 0, 1) * (np.abs(u) < 1)
    sclera = mixc((80, 60, 70), (232, 226, 220), np.clip((YY - top) / (eh * 0.9), 0, 1)) * (1 - 0.45 * u * u)[..., None]
    ix, iy = ex + 22 * math.sin(lt * 0.8), ey + 4 + 5 * math.sin(lt * 1.3)
    di = np.sqrt((XX - ix) ** 2 + (YY - iy) ** 2)
    ang = np.arctan2(YY - iy, XX - ix)
    ir = 27.0
    iris = mixc((70, 180, 200), (20, 60, 115), np.clip(di / ir, 0, 1))
    iris *= (0.75 + 0.25 * np.sin(ang * 31 + 3 * np.sin(ang * 7)))[..., None]
    iris *= (1 - 0.6 * np.clip((di - ir * 0.8) / (ir * 0.2), 0, 1))[..., None]
    pr = 10 + 2 * math.sin(lt)
    iris = np.where((di < pr)[..., None], np.array((4, 4, 10), np.float32), iris)
    im = np.clip(ir - di, 0, 1)[..., None]
    eye = sclera * (1 - im) + iris * im
    hl = np.exp(-((XX - ix + 8) ** 2 + (YY - iy + 9) ** 2) / 16)[..., None]
    eye += 255 * hl
    img = img * (1 - m[..., None]) + eye * m[..., None]
    lid = [P(ex + uu * ew, ey - eh * (1 - uu * uu)) for uu in np.linspace(-1, 1, 24)]
    cv2.polylines(img, [np.array(lid, np.int32)], False, (70, 30, 35), 2, cv2.LINE_AA)
    for uu in np.linspace(-0.8, 0.8, 9):                         # lashes
        x0, y0 = ex + uu * ew, ey - eh * (1 - uu * uu)
        cv2.line(img, P(x0, y0), P(x0 + uu * 10, y0 - 9 * openness - 2), (60, 25, 30), 1, cv2.LINE_AA)
    cv2.ellipse(img, P(ex, ey - 4), (L(ew * 0.95), L(eh * 1.3 + 10)), 0, 200, 340,
                (150, 80, 80), 1, cv2.LINE_AA)

    # thread + falling glowing drop + ripples
    base = pcy + R
    cv2.line(img, P(pcx, base), P(pcx, base + 40), (120, 170, 230), 1, cv2.LINE_AA)
    fall = (lt * 0.55) % 1.0
    dy = base + 45 + fall ** 2 * 170
    glow(img, pcx, dy, 16, (150, 210, 255), 0.9 * (1 - fall * 0.5))
    cv2.circle(img, P(pcx, dy), L(6), (230, 245, 255), -1, cv2.LINE_AA)
    tri = np.array([P(pcx - 6, dy - 1), P(pcx + 6, dy - 1), P(pcx, dy - 15)], np.int32)
    cv2.fillPoly(img, [tri], (230, 245, 255), cv2.LINE_AA)
    since = (lt * 0.55) % 1.0
    for k in range(3):
        rr = (since + k * 0.33) % 1.0
        cv2.ellipse(img, P(pcx, 462), (L(10 + rr * 140), L(3 + rr * 18)), 0, 0, 360,
                    (60 * (1 - rr), 120 * (1 - rr), 180 * (1 - rr)), 1, cv2.LINE_AA)
    return img


WAVES = [(1.0, 0.10, 0.45, 1.4), (0.7, -0.22, 0.7, 2.0), (0.45, 0.35, 1.1, 2.7),
         (0.3, -0.5, 1.8, 3.5), (0.18, 0.8, 2.6, 4.4)]


def scene_waves(t, lt):
    img = sky(t, (3, 6, 20), (14, 32, 72))
    hy = 175
    mx, my = 300.0, 80.0
    glow(img, mx, my, 70, (60, 90, 150), 0.6)
    md = np.sqrt((XX - mx) ** 2 + (YY - my) ** 2)
    img += np.array((235, 235, 215), np.float32) * np.clip(26 - md, 0, 1)[..., None]

    ha = hy // RES
    yy = YY[ha:] - hy + 2
    z = 900.0 / yy
    xw = (XX[ha:] - SW / 2) * z / 300.0
    h = np.zeros_like(z)
    dhdz = np.zeros_like(z)
    for A, kx, kz, w in WAVES:
        ph = kx * xw + kz * z - w * t
        h += A * np.sin(ph)
        dhdz += A * kz * np.cos(ph)
    calm = yy / (yy + 25)
    h, dhdz = h * calm, dhdz * calm
    shade = np.clip(0.4 - 0.45 * dhdz, 0, 1) ** 1.5
    sea = mixc((2, 10, 32), (30, 95, 160), shade)
    col_w = np.exp(-((XX[ha:] - mx) / (5 + yy * 0.16)) ** 2)
    spec = col_w * np.clip(-dhdz * 0.7 + 0.1, 0, 1) ** 3 * 1.6
    sea += np.array((220, 230, 255), np.float32) * spec[..., None]
    foam = np.clip((h - 1.3) / 0.5, 0, 1) * np.clip(yy / 120, 0, 1)
    sea += np.array((170, 215, 240), np.float32) * foam[..., None]
    haze = np.exp(-yy / 14)[..., None]
    sea = sea * (1 - haze) + np.array((25, 45, 90), np.float32) * haze
    img[ha:] = sea
    return img


SCENES = [scene_hands, scene_lotus, scene_eye, scene_waves]


def garden_frame(t):
    """One RGB frame of the night garden (cross-fading between scenes)."""
    i = int(t // SCENE_LEN) % len(SCENES)
    lt = t % SCENE_LEN
    img = SCENES[i](t, lt)
    if lt > SCENE_LEN - FADE:
        a = smooth((lt - (SCENE_LEN - FADE)) / FADE)
        img = img * (1 - a) + SCENES[(i + 1) % len(SCENES)](t, lt - SCENE_LEN) * a
    return np.clip(img * VIGNETTE, 0, 255).astype(np.uint8)


# ======================================================= text renderer ===
class Renderer:
    """Turns every frame into coloured text - this is the code on screen."""
    RAMP = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    EDGE = np.array(["-", "\\", "|", "/"])

    def __init__(self, cols, rows, font, edges=0.55, matcher=None):
        self.cols, self.rows, self.font = cols, rows, font
        self.matcher = matcher
        self.cw, self.ch = font.size("M")
        self.ramp = np.array(list(self.RAMP))
        self.edges = edges
        self.cache = {}

    def convert(self, frame):
        if self.matcher is not None:                      # detail: match character shapes
            hi = cv2.resize(frame, self.matcher.sample_size(self.cols, self.rows),
                            interpolation=cv2.INTER_AREA).astype(np.float32)
            glyph, rgb, peak = self.matcher.match(hi, self.cols, self.rows)
            lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0
            glyph = np.where(peak < 0.04, " ", glyph)
            v = np.maximum(rgb.max(-1, keepdims=True), 1.0)
            fg_rgb = rgb * 0.4 + 0.6 * rgb / v * (60 + 195 * np.sqrt(lum))[..., None]
            return glyph, np.clip(fg_rgb, 0, 255).astype(np.uint8)
        small = cv2.resize(frame, (self.cols, self.rows), interpolation=cv2.INTER_AREA)
        rgb = small.astype(np.float32)
        lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0
        n = len(self.ramp)
        idx = np.clip((lum * (n - 1) + 0.5).astype(np.int32), 0, n - 1)
        idx[lum < 0.03] = 0
        glyph = self.ramp[idx]

        if self.edges > 0:
            gray = lum.astype(np.float32)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.hypot(gx, gy) / 4.0
            ang = (np.degrees(np.arctan2(gy, gx)) + 90.0) % 180.0
            sel = self.EDGE[(((ang + 22.5) % 180.0) // 45).astype(np.int32) % 4]
            mask = (mag > (0.8 - 0.6 * self.edges)) & (lum > 0.12)
            glyph = np.where(mask, sel, glyph)

        v = np.maximum(rgb.max(-1, keepdims=True), 1.0)
        fg_rgb = rgb * 0.4 + 0.6 * rgb / v * (60 + 195 * np.sqrt(lum))[..., None]
        return glyph, np.clip(fg_rgb, 0, 255).astype(np.uint8)

    def draw(self, surf, glyph, fg_rgb, x0=0, y0=0):
        q = (fg_rgb >> 4) << 4
        if len(self.cache) > 40000:
            self.cache.clear()
        blits = []
        ys, xs = np.nonzero(glyph != " ")
        for y, x in zip(ys.tolist(), xs.tolist()):
            c = q[y, x]
            key = (glyph[y, x], int(c[0]), int(c[1]), int(c[2]))
            img = self.cache.get(key)
            if img is None:
                img = self.cache[key] = self.font.render(key[0], True, key[1:])
            blits.append((img, (x0 + x * self.cw, y0 + y * self.ch)))
        surf.blits(blits, doreturn=False)


# =========================================================== code panel ===
TOKEN = re.compile(r"(#.*$)|(\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')|(\b\d+\.?\d*\b)"
                   r"|([A-Za-z_]\w*)|(\s+)|(.)")
C_TEXT, C_COMMENT, C_STRING = (171, 178, 191), (106, 115, 125), (152, 195, 121)
C_NUMBER, C_KEYWORD, C_FUNC = (209, 154, 102), (198, 120, 221), (97, 175, 239)
C_SELF, C_MODULE = (224, 108, 117), (86, 182, 194)


class CodePanel:
    """Editor-style panel that scrolls through this file's real Renderer code."""

    def __init__(self, rect):
        self.rect = rect
        self.font = pygame.font.SysFont("consolas,couriernew,monospace", 12)
        self.title_font = pygame.font.SysFont("segoeui,arial", 13)
        self.lh = self.font.get_linesize() + 1
        with open(os.path.abspath(__file__), encoding="utf-8") as f:
            self.lines = f.read().splitlines()
        start = next(i for i, l in enumerate(self.lines) if l.startswith("class Renderer"))
        end = next((i for i in range(start + 1, len(self.lines)) if self.lines[i].startswith("# ====")),
                   len(self.lines))
        while end > start and not self.lines[end - 1].strip():
            end -= 1
        self.start, self.end = start, end
        self.cache = {}

    def line_surface(self, i):
        s = self.cache.get(i)
        if s is not None:
            return s
        text = self.lines[i]
        s = pygame.Surface((self.rect.w, self.lh), pygame.SRCALPHA)
        x = 0
        for m in TOKEN.finditer(text):
            tok = m.group(0)
            if m.group(1):
                col = C_COMMENT
            elif m.group(2):
                col = C_STRING
            elif m.group(3):
                col = C_NUMBER
            elif m.group(4):
                if tok in keyword.kwlist:
                    col = C_KEYWORD
                elif tok in ("self", "cls"):
                    col = C_SELF
                elif tok in ("np", "cv2", "pygame", "math"):
                    col = C_MODULE
                elif text[m.end():m.end() + 1] == "(":
                    col = C_FUNC
                else:
                    col = C_TEXT
            else:
                col = C_TEXT
            if tok.strip():
                s.blit(self.font.render(tok, True, col), (x, 0))
            x += self.font.size(tok)[0]
        self.cache[i] = s
        return s

    def draw(self, surf, t):
        r = self.rect
        pygame.draw.rect(surf, (22, 24, 30), r)
        bar = pygame.Rect(r.x, r.y, r.w, 28)
        pygame.draw.rect(surf, (36, 38, 46), bar)
        for k, col in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
            pygame.draw.circle(surf, col, (r.x + 16 + k * 18, r.y + 14), 6)
        title = self.title_font.render(os.path.basename(__file__) + "  \u2014  Renderer.convert() + draw()", True, (190, 192, 204))
        surf.blit(title, title.get_rect(midleft=(r.x + 76, r.y + 14)))

        top = r.y + 34
        area = pygame.Rect(r.x, top, r.w, r.bottom - top)
        old_clip = surf.get_clip()
        surf.set_clip(area)
        count = self.end - self.start
        off = (t * 16.0) % (count * self.lh)
        first, frac = int(off // self.lh), off % self.lh
        visible = area.h // self.lh + 2
        hl_y = top + (visible // 2 - 1) * self.lh
        pygame.draw.rect(surf, (40, 46, 62), (r.x, hl_y, r.w, self.lh))
        for k in range(visible):
            li = self.start + (first + k) % count
            y = int(top + k * self.lh - frac)
            num = self.font.render(str(li + 1).rjust(4), True, (90, 96, 110))
            surf.blit(num, (r.x + 4, y))
            surf.blit(self.line_surface(li), (r.x + 46, y))
        surf.set_clip(old_clip)


# ================================================================= main ===
def main():
    ap = argparse.ArgumentParser(description="A night garden, made entirely of text")
    ap.add_argument("--record", metavar="OUT.mp4", help="save a video instead of just playing")
    ap.add_argument("--seconds", type=float, default=SCENE_LEN * len(SCENES), help="length of the recording")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--font-size", type=int, default=9, help="ASCII character size (smaller = more detail)")
    ap.add_argument("--edges", type=float, default=0.55, help="0-1 outline strength (0 = off)")
    ap.add_argument("--detail", action="store_true", help="shape-matched characters for sharper detail")
    args = ap.parse_args()

    pygame.init()
    info = pygame.display.Info()
    scale = min(1.0, (info.current_h - 90) / WIN_H) if info.current_h > 0 else 1.0
    win = Window((int(WIN_W * scale), int(WIN_H * scale)), "Night Garden")
    pygame.display.set_caption("Night Garden - made entirely of text")
    canvas = pygame.Surface((WIN_W, WIN_H))

    art_font = pygame.font.SysFont("consolas,couriernew,monospace", args.font_size, bold=True)
    cw, ch = art_font.size("M")
    cols, rows = WIN_W // cw, ART_H // ch
    x0, y0 = (WIN_W - cols * cw) // 2, (ART_H - rows * ch) // 2
    renderer = Renderer(cols, rows, art_font, args.edges,
                        GlyphMatcher(art_font) if args.detail else None)
    panel = CodePanel(pygame.Rect(0, ART_H, WIN_W, WIN_H - ART_H))
    zoom = ViewZoom()

    writer = None
    if args.record:
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (WIN_W, WIN_H))
        total = int(args.seconds * args.fps)
        print(f"Recording {total} frames to {args.record} ...")

    clock = pygame.time.Clock()
    t, frame_no, paused = 0.0, 0, False
    last = time.perf_counter()
    try:
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return
                if win.handle(e):                  # resize / F11
                    continue
                if zoom.handle(e):                 # Z / X / 0 / mouse wheel
                    pygame.display.set_caption(f"Night Garden  [zoom {zoom.percent}%]")
                    continue
                if e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_ESCAPE, pygame.K_q):
                        return
                    if e.key == pygame.K_SPACE:
                        paused = not paused
                    elif e.key == pygame.K_RIGHT:
                        t += SCENE_LEN - (t % SCENE_LEN) - FADE
                    elif e.key == pygame.K_e:
                        renderer.edges = (round(renderer.edges + 0.25, 2)) % 1.25
                    elif e.key == pygame.K_s:
                        name = time.strftime("night_garden_%H%M%S.png")
                        pygame.image.save(canvas, name)

            now = time.perf_counter()
            if writer:
                t = frame_no / args.fps
            elif not paused:
                t += min(now - last, 0.1)
            last = now

            canvas.fill((0, 0, 0))
            glyph, fg_rgb = renderer.convert(zoom_image(garden_frame(t), zoom.factor))
            renderer.draw(canvas, glyph, fg_rgb, x0, y0)
            panel.draw(canvas, t)

            if writer:
                arr = pygame.surfarray.array3d(canvas).swapaxes(0, 1)
                writer.write(cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_RGB2BGR))
                frame_no += 1
                pygame.display.set_caption(f"Recording {frame_no}/{total}")
                if frame_no >= total:
                    break

            win.present(canvas)                    # scale to fit the (resizable) window
            pygame.display.flip()
            if not writer:
                clock.tick(args.fps)
    finally:
        if writer:
            writer.release()
            print(f"Saved {args.record}")
        pygame.quit()


if __name__ == "__main__":
    main()
