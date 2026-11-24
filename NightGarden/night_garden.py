"""
Night Garden - an animation made entirely of text
-------------------------------------------------
Nothing here is a picture turned into ASCII. Every shape is built directly
from characters, cell by cell:

  hands  -> spelled out with the letters of "touch the light"
  orb    -> rings of rotating digits 0-9
  lotus  -> petals filled with streams of the word "lotus"
  eye    -> iris made of radial  / - \\ |  strokes, portal ring of digits
  waves  -> rows of  ~ ^ - *  rolling toward you
  garden -> stems of  | / \\ , flower heads of letters, fireflies of  * +

New characters "decode" from random symbols when they appear, numbers rain
softly in the background, and the real code scrolls in the editor panel.

Run
  python night_garden.py                      # pop-out window (9:16, TikTok shape)
  python night_garden.py --record garden.mp4  # save the full loop as a video
  python night_garden.py --font-size 7        # even more detail (smaller characters)

Keys
  F11 fullscreen   resize the window freely (it scales to fit)
  Z zoom in   X zoom out   0 reset zoom (or the mouse wheel)
  SPACE pause   RIGHT next scene   S screenshot   Q / ESC quit
"""
import argparse
import keyword
import math
import os
import re
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import cv2                     # only used to save the video when recording
import numpy as np
import pygame

# AsciiArt is split into folders - let this file find Common/ and Bouquet/
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ("Common", "Bouquet"):
    if _os.path.join(_ROOT, _d) not in _sys.path:
        _sys.path.insert(0, _os.path.join(_ROOT, _d))

from window_util import ViewZoom, Window

WIN_W, WIN_H = 540, 960        # 9:16 like a phone video
ART_H = 610                    # top: the text animation, bottom: the code
SCENE_LEN, FADE = 6.0, 1.2

SCRAMBLE = np.array(list("!@#$%&*+=?/\\|<>{}[]0123456789ABCDEFXYZ"))
EDGE = np.array(["-", "\\", "|", "/"])
DIGITS = np.array(list("0123456789"))


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def mixc(a, b, t):
    """Blend colour a -> b by t (float or array)."""
    a, b = np.array(a, np.float32), np.array(b, np.float32)
    t = np.asarray(t, np.float32)
    return a + (b - a) * (t[..., None] if t.ndim else t)


_CHARS = {}


def chars(s):
    a = _CHARS.get(s)
    if a is None:
        a = _CHARS[s] = np.array(list(s))
    return a


def sd_capsule(U, V, ax, ay, bx, by, r):
    """Signed distance to a rounded line segment (negative = inside)."""
    pax, pay, bax, bay = U - ax, V - ay, bx - ax, by - ay
    h = np.clip((pax * bax + pay * bay) / (bax * bax + bay * bay + 1e-9), 0, 1)
    return np.hypot(pax - bax * h, pay - bay * h) - r


# ======================================================== text canvas ====
class TextCanvas:
    """A grid of characters + colours. Scenes paint shapes straight into it."""

    def __init__(self, cols, rows, cw, ch):
        self.cols, self.rows, self.cw, self.ch = cols, rows, cw, ch
        self.half = cols / 2
        self.asp = ch / cw                                  # a cell is taller than wide
        x = (np.arange(cols) + 0.5 - cols / 2) / self.half
        y = (np.arange(rows) + 0.5 - rows / 2) * self.asp / self.half
        self.U, self.V = np.meshgrid(x.astype(np.float32), y.astype(np.float32))
        self.C, self.R = np.meshgrid(np.arange(cols), np.arange(rows))
        rng = np.random.default_rng(3)
        self.rand = rng.random((rows, cols)).astype(np.float32)
        self.rand2 = rng.random((rows, cols)).astype(np.float32)
        self.rain = [(int(rng.integers(0, cols)), rng.uniform(4, 11), rng.uniform(0, rows))
                     for _ in range(cols // 6)]
        self.base_U, self.base_V = self.U, self.V
        self.set_zoom(1.0)
        self.clear()

    def set_zoom(self, z):
        """Zoom the whole scene: z > 1 = closer (shapes bigger), z < 1 = further away."""
        self.zoom = z
        self.scale = self.half * z                          # characters per scene unit
        self.U, self.V = self.base_U / z, self.base_V / z
        self.cell = 2 / self.cols / z                       # one cell width in scene units

    def clear(self):
        self.glyph = np.full((self.rows, self.cols), " ", dtype="<U1")
        self.color = np.zeros((self.rows, self.cols, 3), np.float32)

    def put(self, mask, glyph, color):
        if not mask.any():
            return
        self.glyph[mask] = glyph if isinstance(glyph, str) else glyph[mask]
        col = np.asarray(color, np.float32)
        self.color[mask] = col if col.ndim == 1 else col[mask]

    def dot(self, u, v, glyph, color):
        c = int(u * self.scale + self.cols / 2)
        r = int(v * self.scale / self.asp + self.rows / 2)
        if 0 <= c < self.cols and 0 <= r < self.rows:
            self.glyph[r, c] = glyph
            self.color[r, c] = color

    def text(self, s, idx):
        a = chars(s)
        return a[np.mod(idx, len(a)).astype(np.int32)]

    def edges(self, sd, mask=None):
        """Outline characters that follow the direction of a shape's edge."""
        gy, gx = np.gradient(sd)
        out = np.full(sd.shape, " ", dtype="<U1")
        m = np.ones(sd.shape, bool) if mask is None else mask
        ang = (np.degrees(np.arctan2(gy[m] / self.ch, gx[m] / self.cw)) + 90.0) % 180.0
        out[m] = EDGE[(((ang + 22.5) % 180.0) // 45).astype(np.int32) % 4]
        return out

    def shape(self, sd, fill_glyph, fill_color, edge_color):
        inside = sd < 0
        if not inside.any():
            return
        self.put(inside, fill_glyph, fill_color)
        core = inside.copy()
        core[1:, :] &= inside[:-1, :]
        core[:-1, :] &= inside[1:, :]
        core[:, 1:] &= inside[:, :-1]
        core[:, :-1] &= inside[:, 1:]
        rim = inside & ~core
        self.put(rim, self.edges(sd, rim), edge_color)


def background(cv, t):
    """Twinkling stars + a soft rain of numbers."""
    r = cv.rand
    star = r < 0.035
    tw = 0.5 + 0.5 * np.sin(t * 2.0 + r * 600)
    cv.put(star, cv.text(".'+*.", r * 1000), (40 + 170 * tw)[..., None] * np.array((0.6, 0.7, 1.0)))
    for col, speed, off in cv.rain:
        head = (t * speed + off) % (cv.rows + 10)
        for k in range(7):
            row = int(head) - k
            if 0 <= row < cv.rows:
                cv.glyph[row, col] = DIGITS[(row * 7 + col + int(t * 9)) % 10]
                cv.color[row, col] = np.array((35, 70, 130)) * (1 - k / 7) * (1.6 if k == 0 else 1)


# ============================================================== scenes ===
def hand_sd(U, V, rise, curl):
    """Two hands (one mirrored), palms up, fingers reaching for the orb."""
    Ul, V = -np.abs(U), V - rise
    sd = sd_capsule(Ul, V, -0.82, 1.45, -0.45, 0.66, 0.13)              # forearm
    sd = np.minimum(sd, sd_capsule(Ul, V, -0.43, 0.56, -0.36, 0.40, 0.125))  # palm
    for i, lm in enumerate((0.74, 0.95, 1.0, 0.9)):
        bx, by = -0.505 + i * 0.062, 0.315 - (0.02 if i in (1, 2) else 0)
        ang, r = math.radians(-106 + i * 9), 0.035
        for seg in (0.10, 0.08, 0.06):                                    # 3 finger joints
            ex, ey = bx + math.cos(ang) * seg * lm, by + math.sin(ang) * seg * lm
            sd = np.minimum(sd, sd_capsule(Ul, V, bx, by, ex, ey, r))
            bx, by, r, ang = ex, ey, r * 0.9, ang + math.radians(curl)
    bx, by, ang, r = -0.28, 0.45, math.radians(-55), 0.04                 # thumb
    for seg in (0.08, 0.065):
        ex, ey = bx + math.cos(ang) * seg, by + math.sin(ang) * seg
        sd = np.minimum(sd, sd_capsule(Ul, V, bx, by, ex, ey, r))
        bx, by, r, ang = ex, ey, r * 0.88, ang - math.radians(curl * 0.8)
    return sd


def scene_hands(cv, t, lt):
    U, V = cv.U, cv.V
    ox, oy = 0.0, -0.52 + 0.03 * math.sin(t * 1.5)
    d = np.hypot(U - ox, V - oy)
    ang = np.arctan2(V - oy, U - ox)
    # glow haze, light rays, rings of digits, bright core
    haze = (d < 0.55) & (cv.rand < 0.2 * (1 - d / 0.55))
    cv.put(haze, cv.text(".:'", cv.rand * 99), mixc((30, 50, 120), (90, 130, 230), 1 - d / 0.55))
    rays = (np.abs(np.sin(ang * 6 + t * 0.4)) > 0.985) & (d > 0.14) & (d < 0.62)
    cv.put(rays, ":", mixc((60, 90, 180), (170, 200, 255), np.clip(1 - d / 0.62, 0, 1)))
    rings = ((d * 16 - t * 1.6) % 1.0 < 0.2) & (d > 0.1) & (d < 0.42)
    cv.put(rings, DIGITS[((ang / (2 * np.pi)) * 70 + t * 8).astype(np.int32) % 10],
           mixc((70, 110, 230), (210, 230, 255), np.clip(1 - d / 0.42, 0, 1)))
    cv.put(d < 0.11, cv.text("Oo0", cv.C + cv.R), (190, 215, 255))
    cv.put(d < 0.06, "@", (255, 255, 255))
    # hands rise into view, spelled with letters, lit by the orb
    rise = (1 - smooth(lt / 1.6)) * 0.9
    sd = hand_sd(U, V, rise, 11 + 5 * math.sin(t * 1.6))
    depth = np.clip(-sd / 0.07, 0, 1)
    lit = np.clip(1.3 - np.hypot(U - ox, V - oy) / 1.3, 0.3, 1.0)
    skin = mixc((90, 45, 40), (245, 160, 120), (0.35 + 0.65 * depth) * lit)
    fill = cv.text("touch the light ", cv.C + cv.R * 7 - int(t * 6))
    cv.shape(sd, fill, skin, mixc((150, 80, 60), (255, 205, 160), lit))
    # sparks floating up out of the orb
    for k in range(34):
        life = (t * 0.3 + k / 34) % 1.0
        u = ox + math.sin(k * 2.1 + t * 0.9) * (0.04 + 0.32 * life)
        v = oy + 0.18 - life * 0.75
        b = 1 - life
        cv.dot(u, v, "*+'."[k % 4], (150 * b + 60, 180 * b + 60, 255 * b))


LOTUS_LAYERS = [  # petals, spread (deg), length, width, base colour, tip colour
    (9, 84, 0.50, 0.11, (8, 55, 80), (60, 150, 170)),
    (7, 66, 0.58, 0.13, (12, 75, 100), (95, 195, 205)),
    (5, 44, 0.54, 0.135, (20, 105, 130), (145, 228, 232)),
    (3, 20, 0.45, 0.14, (40, 140, 160), (205, 252, 250)),
]
BASE_Y = 0.2


def lotus_petals(U, V, bloom):
    out = []
    for li, (n, spread, L, w, cin, cout) in enumerate(LOTUS_LAYERS):
        L *= 0.5 + 0.5 * min(bloom, 1.05)
        sp = spread * (0.15 + 0.85 * bloom)
        order = sorted(range(n), key=lambda k: -abs(k - (n - 1) / 2))  # outer first
        for k in order:
            th = math.radians(-sp + 2 * sp * k / (n - 1)) if n > 1 else 0.0
            dx, dy = U, V - BASE_Y
            along = dx * math.sin(th) - dy * math.cos(th)
            across = dx * math.cos(th) + dy * math.sin(th)
            taper = 1 - 0.6 * np.clip(along / L, 0, 1) ** 2
            q = np.hypot((along - L / 2) / (L / 2), across / (w * taper + 1e-3))
            out.append(((q - 1) * min(L / 2, w), along, L, cin, cout, li * 5 + k))
        if li == 2:
            out.append(None)                                          # seed pod goes here
    return out


def draw_lotus(cv, U, V, bloom, t, dim=1.0, mask=None):
    for p in lotus_petals(U, V, bloom):
        if p is None:
            sd = (np.hypot(U / 0.08, (V - BASE_Y + 0.08) / 0.045) - 1) * 0.045
            if mask is not None:
                sd = np.where(mask, sd, 1)
            cv.shape(sd, cv.text("o0O", cv.C), np.array((235, 210, 90)) * dim, np.array((255, 240, 150)) * dim)
            continue
        sd, along, L, cin, cout, key = p
        if mask is not None:
            sd = np.where(mask, sd, 1)
        depth = np.clip(-sd / 0.05, 0, 1)
        col = mixc(cin, cout, np.clip(along / L, 0, 1)) * (0.55 + 0.45 * depth)[..., None] * dim
        fill = cv.text("lotus ", (along * 30 - t * 6).astype(np.int32) + cv.C + key)
        cv.shape(sd, fill, col, np.array(cout) * dim)


def scene_lotus(cv, t, lt):
    U, V = cv.U, cv.V
    bloom = smooth((lt + FADE) / 3.0)
    water = V > BASE_Y + 0.02
    # water: rows of ripples, brighter near the lotus, a moon streak
    wave = np.sin(U * 9 + t * 2.2 + V * 40) * 0.6 + np.sin(U * 17 - t * 1.6 + V * 25) * 0.4
    near = np.clip((V - BASE_Y) * 1.6, 0.25, 1)
    g = np.where(wave > 0.55, "~", np.where(wave > 0.1, "-", np.where(cv.rand < 0.5, ".", " ")))
    cv.put(water & (g != " "), g, mixc((10, 45, 70), (50, 140, 160), (wave * 0.5 + 0.5) * near))
    streak = water & (np.abs(U) < 0.04 + (V - BASE_Y) * 0.25) & (np.sin(V * 60 - t * 4 + U * 20) > 0.2)
    cv.put(streak, "=", (150, 220, 230))
    # reflection of the lotus in the water (wobbly, dim)
    wob = 0.015 * np.sin(V * 50 + t * 3)
    draw_lotus(cv, U + wob, 2 * (BASE_Y + 0.02) - V, bloom, t, dim=0.32, mask=water)
    # lily pads (with a notch) made of leaf letters
    for cx, cy, rx, ph in ((-0.62, 0.55, 0.26, 0.5), (0.63, 0.42, 0.2, 2.6), (0.28, 0.88, 0.22, 4.0),
                           (-0.2, 1.0, 0.15, 1.4)):
        a = np.arctan2((V - cy) / 0.35, U - cx)
        notch = np.abs(np.angle(np.exp(1j * (a - ph)))) < 0.32
        sd = (np.hypot((U - cx) / rx, (V - cy) / (rx * 0.35)) - 1) * rx * 0.35
        sd = np.where(notch, np.abs(sd) + 0.01, sd)
        vein = np.abs(np.sin(a * 5)) < 0.15
        cv.shape(sd, np.where(vein, "|", cv.text("leaf", cv.C + cv.R * 3)), mixc((15, 70, 45), (40, 140, 85), cv.rand * 0.4 + 0.4),
                 (70, 180, 110))
    # the lotus itself, opening and swaying
    sway = 0.02 * math.sin(t * 1.1)
    draw_lotus(cv, U - sway, V, bloom, t)
    # fireflies over the pond
    for k in range(14):
        u = math.sin(t * 0.37 + k * 1.9) * 0.85
        v = -0.65 + 0.5 * math.sin(t * 0.29 + k * 2.7)
        b = 0.5 + 0.5 * math.sin(t * 3 + k)
        cv.dot(u, v, "*+"[k % 2], (170 * b + 40, 240 * b + 15, 120 * b + 20))


def scene_eye(cv, t, lt):
    U, V = cv.U, cv.V
    pcx, pcy, R = 0.0, -0.12, 0.72
    d = np.hypot(U - pcx, V - pcy)
    ang = np.arctan2(V - pcy, U - pcx)
    # portal: warm sparse fill, a spinning ring of digits, outer glow
    inside = d < R - 0.03
    cv.put(inside & (cv.rand < 0.5), cv.text(".,:'`", cv.rand * 50),
           mixc((200, 120, 105), (40, 85, 150), np.clip(d / R, 0, 1) ** 1.4))
    ring = np.abs(d - R) < 0.028
    cv.put(ring, DIGITS[((ang / (2 * np.pi)) * 140 + t * 14).astype(np.int32) % 10], (130, 205, 255))
    halo = (np.abs(d - R) < 0.12) & ~ring & (cv.rand < 0.4 * (1 - np.abs(d - R) / 0.12))
    cv.put(halo, ".", (50, 100, 180))
    ring2 = np.abs(d - R * 0.92) < 0.012
    cv.put(ring2 & (cv.rand < 0.6), DIGITS[((ang / (2 * np.pi)) * 110 - t * 9).astype(np.int32) % 10], (80, 140, 210))
    # eyebrow arc of letters
    bu = U / 0.62
    brow_v = pcy - 0.34 - 0.06 * (1 - bu ** 2)
    brow = (np.abs(bu) < 1) & (np.abs(V - brow_v) < 0.025 * (1.3 - np.abs(bu)))
    cv.put(brow, cv.text("dream", cv.C), (150, 90, 80))
    # the eye: almond shape that blinks, iris follows a slow path
    ex, ey, ew = pcx, pcy + 0.02, 0.56
    cyc = lt % 3.4
    openness = 1 - (math.sin(math.pi * (cyc - 3.1) / 0.3) if cyc > 3.1 else 0.0)
    eh = 0.27 * openness + 0.005
    u = U / ew
    shp = np.clip(1 - u * u, 0, 1)
    top, bot = ey - eh * shp, ey + eh * 0.8 * shp
    eye = (np.abs(u) < 1) & (V > top) & (V < bot)
    shadow = np.clip((V - top) / (eh * 0.9 + 1e-3), 0, 1)
    cv.put(eye, cv.text("=-:", cv.C + cv.R), mixc((110, 90, 100), (235, 230, 225), shadow * (1 - 0.4 * u * u)))
    ix, iy = ex + 0.15 * math.sin(lt * 0.8), ey + 0.03 + 0.03 * math.sin(lt * 1.3)
    di = np.hypot(U - ix, V - iy)
    ia = np.arctan2(V - iy, U - ix)
    iris = eye & (di < 0.19)
    radial = EDGE[(((np.degrees(ia) + 22.5 + 360) % 180) // 45).astype(np.int32) % 4]
    stri = 0.75 + 0.25 * np.sin(ia * 23)
    cv.put(iris, radial, mixc((80, 195, 215), (20, 60, 120), np.clip(di / 0.19, 0, 1)) * stri[..., None])
    cv.put(iris & (di > 0.165), "o", (30, 70, 125))
    pr = 0.07 + 0.01 * math.sin(lt)
    cv.put(eye & (di < pr), "@", (20, 25, 60))
    cv.dot(ix - 0.06, iy - 0.06, "*", (255, 255, 255))
    cv.dot(ix - 0.04, iy - 0.06, "'", (220, 230, 255))
    # lids, crease and lashes
    lid = (np.abs(u) < 1.02) & (np.abs(V - top) < 0.022)
    cv.put(lid, "~", (190, 100, 100))
    cv.put((np.abs(u) < 1.0) & (np.abs(V - bot) < 0.02), "-", (170, 100, 100))
    crease = (np.abs(u) < 0.95) & (np.abs(V - (ey - (eh * 1.25 + 0.06) * shp)) < 0.015)
    cv.put(crease, "_", (140, 75, 80))
    for uu in np.linspace(-0.85, 0.85, 11):
        x0, y0 = ex + uu * ew, ey - eh * (1 - uu * uu)
        cv.dot(x0 + uu * 0.02, y0 - 0.05, "/" if uu > 0.1 else "\\" if uu < -0.1 else "|", (120, 60, 70))
    # glowing teardrop falling from the portal, ripples where it lands
    base = pcy + R
    for k in range(3):
        cv.dot(0, base + 0.03 + k * 0.045, "|", (110, 160, 230))
    fall = (lt * 0.55) % 1.0
    dv = base + 0.17 + fall ** 2 * 0.42
    cv.dot(0, dv - 0.045, "^", (200, 230, 255))
    cv.dot(0, dv, "O", (235, 245, 255))
    cv.dot(-0.02, dv, "(", (150, 200, 255))
    cv.dot(0.02, dv, ")", (150, 200, 255))
    for k in range(3):
        rr = (fall + k * 0.33) % 1.0
        e = np.abs(np.hypot(U / (0.05 + rr * 0.55), (V - 1.02) / (0.015 + rr * 0.07)) - 1) < 0.12
        cv.put(e, "~", np.array((70, 130, 200)) * (1 - rr))


WAVES = [(1.0, 0.10, 0.45, 1.4), (0.7, -0.22, 0.7, 2.0), (0.45, 0.35, 1.1, 2.7),
         (0.3, -0.5, 1.8, 3.5), (0.18, 0.8, 2.6, 4.4)]


def scene_waves(cv, t, lt):
    U, V = cv.U, cv.V
    hy = -0.28
    mx, my = 0.42, -0.74
    md = np.hypot(U - mx, V - my)
    # moon with craters, a halo, drifting clouds of text
    cv.put((md < 0.26) & (cv.rand < 0.5 * (1 - md / 0.26)), ".", (70, 90, 140))
    crater = (np.hypot(U - mx + 0.04, V - my + 0.03) < 0.035) | (np.hypot(U - mx - 0.05, V - my - 0.04) < 0.025)
    cv.put(md < 0.13, np.where(crater, "o", cv.text("MOON", cv.C)), np.where(crater[..., None], np.array((180, 180, 160)), np.array((245, 240, 210))))
    cv.put(np.abs(md - 0.13) < 0.012, "O", (255, 250, 225))
    for cx, cy, w, sp in ((-0.4, -0.78, 0.35, 0.03), (0.1, -0.55, 0.28, 0.05), (-0.75, -0.45, 0.2, 0.04)):
        x = ((cx + t * sp + 1.3) % 2.6) - 1.3
        cloud = np.hypot((U - x) / w, (V - cy) / 0.05) < 1 + 0.25 * np.sin(U * 30)
        cv.put(cloud & (cv.rand < 0.75), cv.text("cloud ", cv.C), (75, 85, 120))
    # distant hills of symbols on the horizon
    hill = hy - 0.05 - 0.06 * np.sin(U * 3.1 + 1) - 0.03 * np.sin(U * 9)
    cv.put((V > hill) & (V < hy), cv.text("^/\\", cv.C + cv.R), (25, 40, 70))
    # the sea: perspective rows of rolling wave characters
    sea = V > hy
    dz = np.maximum(V - hy, 0) + 0.02
    z = 0.4 / dz
    xw = U * z
    h = np.zeros_like(U)
    for A, kx, kz, w in WAVES:
        h += A * np.sin(kx * xw * 6 + kz * z * 3 - w * t)
    near_w = np.clip((dz - 0.3) * 1.5, 0, 1)                    # big close-up swells
    h += near_w * (0.9 * np.sin(U * 7 + V * 22 - t * 1.8) + 0.5 * np.sin(U * 15 - V * 14 - t * 2.6))
    h = h / 1.9 * np.clip(dz * 5, 0.3, 1)
    q1, q2, q3, q4 = np.quantile(h[sea], (0.2, 0.48, 0.74, 0.92))  # foam = top 8 % only
    g = np.where(h > q4, "*", np.where(h > q3, "^", np.where(h > q2, "~", np.where(h > q1, "-", "."))))
    near = np.clip(dz * 1.4, 0.2, 1)
    col = mixc((8, 25, 70), (60, 140, 215), np.clip((h - q1) / (q4 - q1 + 1e-6), 0, 1) * near)
    col = np.where((h > q4)[..., None], np.array((200, 230, 250)) * near[..., None], col)
    cv.put(sea, g, col)
    # moonlight path on the water
    path = sea & (np.abs(U - mx) < 0.02 + dz * 0.22) & (np.sin(V * 70 + np.sin(U * 40) * 2 - t * 5) > 0.25)
    cv.put(path, np.where(h > 0, "|", ":"), (240, 235, 200))
    cv.put(np.abs(V - hy) < 0.012, "_", (90, 110, 160))


GARDEN = [  # x, height, petals, name, colour
    (-0.78, 0.95, 5, "rose", (230, 70, 100)), (-0.5, 1.25, 6, "lily", (245, 245, 255)),
    (-0.22, 0.85, 8, "iris", (160, 110, 240)), (0.05, 1.4, 5, "rose", (255, 120, 160)),
    (0.3, 1.0, 7, "tulip", (255, 170, 60)), (0.55, 1.2, 6, "lily", (255, 220, 120)),
    (0.8, 0.9, 5, "iris", (120, 160, 255)), (-0.62, 0.6, 6, "dahlia", (255, 90, 70)),
    (0.68, 0.62, 8, "daisy", (250, 250, 250)), (-0.05, 0.65, 6, "aster", (200, 140, 255)),
]


def scene_garden(cv, t, lt):
    U, V = cv.U, cv.V
    ground = 0.86
    # crescent moon of letters
    m1 = np.hypot(U + 0.6, V + 0.78) < 0.17
    m2 = np.hypot(U + 0.52, V + 0.83) < 0.155
    cv.put((np.hypot(U + 0.6, V + 0.78) < 0.3) & (cv.rand < 0.12), ".", (60, 70, 110))
    cv.put(m1 & ~m2, cv.text("moon ", cv.C + cv.R), (250, 240, 200))
    # flowers grow one after another, stems sway, heads open
    for i, (x, hgt, n, name, col) in enumerate(GARDEN):
        grow = smooth((lt + FADE - i * 0.25) / 1.8)
        if grow <= 0:
            continue
        top = ground - hgt * grow
        sway = 0.05 * math.sin(t * 1.3 + i)
        rows = np.arange(cv.rows)
        vv = (rows + 0.5 - cv.rows / 2) * cv.asp / cv.scale
        for r, v in zip(rows, vv):
            if top < v < ground:
                f = (ground - v) / (ground - top)
                u = x + sway * f * f
                slope = 2 * sway * f / max(hgt * grow, 0.05)
                cv.dot(u, v, "|" if abs(slope) < 0.08 else ("/" if slope > 0 else "\\"), (50, 140, 70))
                if abs(f - 0.45) < 0.03:
                    cv.dot(u - 0.03, v, "<", (60, 170, 80))
                    cv.dot(u - 0.05, v, "(", (60, 170, 80))
                if abs(f - 0.62) < 0.03:
                    cv.dot(u + 0.03, v, ">", (60, 170, 80))
                    cv.dot(u + 0.05, v, ")", (60, 170, 80))
        hx, hy = x + sway, top
        bloom = smooth((lt + FADE - i * 0.25 - 1.2) / 1.2)
        if bloom > 0:
            R = (0.09 + 0.035 * (i % 3)) * bloom
            dh = np.hypot(U - hx, (V - hy) * 1.0)
            a = np.arctan2(V - hy, U - hx) + t * 0.3
            rr = R * (0.55 + 0.45 * np.abs(np.cos(a * n / 2)))
            head = dh < rr
            shade = np.clip(1.1 - dh / (R + 1e-3), 0.35, 1)
            cv.put(head, cv.text(name, cv.C + cv.R * 2), np.array(col) * shade[..., None])
            cv.put(dh < R * 0.25, "@", (255, 225, 90))
    # grass blades swaying along the ground, soil below
    gcol = np.arange(cv.cols)
    for c in gcol:
        u = (c + 0.5 - cv.cols / 2) / cv.scale
        hgt = 0.05 + 0.07 * cv.rand[0, c]
        s = math.sin(t * 1.6 + c * 0.35)
        for k in range(1 + int(hgt / (cv.cell * cv.asp))):
            v = ground - k * cv.cell * cv.asp
            cv.dot(u, v, "|" if abs(s) < 0.3 else ("/" if s > 0 else "\\"), (35, 110 + 30 * cv.rand[1, c], 55))
    cv.put(V > ground, cv.text(",.'`", cv.C + cv.R), (35, 60, 40))
    # fireflies drifting through the garden
    for k in range(26):
        u = math.sin(t * 0.33 + k * 1.7) * 0.9
        v = 0.1 + 0.55 * math.sin(t * 0.27 + k * 2.3)
        b = 0.5 + 0.5 * math.sin(t * 3.2 + k * 1.3)
        cv.dot(u, v, "*+."[k % 3], (180 * b + 40, 250 * b + 5, 110 * b + 20))


SCENES = [scene_hands, scene_lotus, scene_eye, scene_waves, scene_garden]


def compose(cv, t):
    """One frame: background + current scene, dissolving cell by cell into the next."""
    i = int(t // SCENE_LEN) % len(SCENES)
    lt = t % SCENE_LEN
    cv.clear()
    background(cv, t)
    SCENES[i](cv, t, lt)
    glyph, color = cv.glyph.copy(), cv.color.copy()
    if lt > SCENE_LEN - FADE:
        a = smooth((lt - (SCENE_LEN - FADE)) / FADE)
        cv.clear()
        background(cv, t)
        SCENES[(i + 1) % len(SCENES)](cv, t, lt - SCENE_LEN)
        take = cv.rand2 < a
        glyph = np.where(take, cv.glyph, glyph)
        color = np.where(take[..., None], cv.color, color)
    return glyph, color


# ============================================================= drawing ===
class Decoder:
    """Cells that just appeared flicker through random symbols, then settle."""

    def __init__(self, shape):
        self.last_seen = np.full(shape, -99.0)      # last time each cell had a character
        self.born = np.full(shape, -9.0)

    def apply(self, glyph, color, t):
        filled = glyph != " "
        # only cells that were empty for a while decode (not every moving edge)
        fresh = filled & (t - self.last_seen > 1.0)
        self.born[fresh] = t
        self.last_seen[filled] = t
        young = filled & (t - self.born < 0.35)
        if young.any():
            noise = SCRAMBLE[np.random.randint(0, len(SCRAMBLE), glyph.shape)]
            glyph = np.where(young, noise, glyph)
            color = np.where(young[..., None], color * 0.5 + 120, color)
        return glyph, color


class GlyphPainter:
    def __init__(self, font):
        self.font = font
        self.cw, self.ch = font.size("M")
        self.cache = {}

    def draw(self, surf, glyph, color, x0, y0):
        q = (np.clip(color, 0, 255).astype(np.uint8) >> 3) << 3
        if len(self.cache) > 50000:
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


# ========================================================== code panel ===
TOKEN = re.compile(r"(#.*$)|(\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')|(\b\d+\.?\d*\b)"
                   r"|([A-Za-z_]\w*)|(\s+)|(.)")
C_TEXT, C_COMMENT, C_STRING = (171, 178, 191), (106, 115, 125), (152, 195, 121)
C_NUMBER, C_KEYWORD, C_FUNC = (209, 154, 102), (198, 120, 221), (97, 175, 239)
C_SELF, C_MODULE = (224, 108, 117), (86, 182, 194)


class CodePanel:
    """Editor-style panel scrolling through this file's real scene code."""

    def __init__(self, rect):
        self.rect = rect
        self.font = pygame.font.SysFont("consolas,couriernew,monospace", 12)
        self.title_font = pygame.font.SysFont("segoeui,arial", 13)
        self.lh = self.font.get_linesize() + 1
        with open(os.path.abspath(__file__), encoding="utf-8") as f:
            self.lines = f.read().splitlines()
        self.start = next(i for i, l in enumerate(self.lines) if l.startswith("def hand_sd"))
        self.end = next(i for i, l in enumerate(self.lines) if l.startswith("SCENES ="))
        self.cache = {}

    def line_surface(self, i):
        s = self.cache.get(i)
        if s is not None:
            return s
        text = self.lines[i]
        s = pygame.Surface((self.rect.w * 2, self.lh), pygame.SRCALPHA)
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
                elif tok in ("self", "cv"):
                    col = C_SELF
                elif tok in ("np", "math", "cv2", "pygame"):
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

    def draw(self, surf, t, scene_name):
        r = self.rect
        pygame.draw.rect(surf, (22, 24, 30), r)
        pygame.draw.rect(surf, (36, 38, 46), (r.x, r.y, r.w, 28))
        for k, col in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
            pygame.draw.circle(surf, col, (r.x + 16 + k * 18, r.y + 14), 6)
        title = self.title_font.render(f"night_garden.py  \u2014  {scene_name}()", True, (190, 192, 204))
        surf.blit(title, title.get_rect(midleft=(r.x + 76, r.y + 14)))
        top = r.y + 34
        area = pygame.Rect(r.x, top, r.w, r.bottom - top)
        old = surf.get_clip()
        surf.set_clip(area)
        count = self.end - self.start
        off = (t * 22.0) % (count * self.lh)
        first, frac = int(off // self.lh), off % self.lh
        visible = area.h // self.lh + 2
        pygame.draw.rect(surf, (40, 46, 62), (r.x, top + (visible // 2 - 1) * self.lh, r.w, self.lh))
        for k in range(visible):
            li = self.start + (first + k) % count
            y = int(top + k * self.lh - frac)
            surf.blit(self.font.render(str(li + 1).rjust(4), True, (90, 96, 110)), (r.x + 4, y))
            surf.blit(self.line_surface(li), (r.x + 46, y))
        surf.set_clip(old)


# ================================================================ main ===
def main():
    ap = argparse.ArgumentParser(description="A night garden, made entirely of text")
    ap.add_argument("--record", metavar="OUT.mp4", help="save a video of the full loop")
    ap.add_argument("--seconds", type=float, default=SCENE_LEN * len(SCENES))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--font-size", type=int, default=9, help="character size (smaller = more detail)")
    args = ap.parse_args()

    pygame.init()
    info = pygame.display.Info()
    scale = min(1.0, (info.current_h - 90) / WIN_H) if info.current_h > 0 else 1.0
    win = Window((int(WIN_W * scale), int(WIN_H * scale)), "Night Garden")
    pygame.display.set_caption("Night Garden - made entirely of text")
    canvas = pygame.Surface((WIN_W, WIN_H))

    font = pygame.font.SysFont("consolas,couriernew,monospace", args.font_size, bold=True)
    painter = GlyphPainter(font)
    cols, rows = WIN_W // painter.cw, ART_H // painter.ch
    x0, y0 = (WIN_W - cols * painter.cw) // 2, (ART_H - rows * painter.ch) // 2
    cv = TextCanvas(cols, rows, painter.cw, painter.ch)
    zoom = ViewZoom()
    decoder = Decoder((rows, cols))
    panel = CodePanel(pygame.Rect(0, ART_H, WIN_W, WIN_H - ART_H))

    writer, total = None, 0
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
                    cv.set_zoom(zoom.factor)
                    pygame.display.set_caption(f"Night Garden  [zoom {zoom.percent}%]")
                    continue
                if e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_ESCAPE, pygame.K_q):
                        return
                    if e.key == pygame.K_SPACE:
                        paused = not paused
                    elif e.key == pygame.K_RIGHT:
                        t += SCENE_LEN - (t % SCENE_LEN) - FADE
                    elif e.key == pygame.K_s:
                        pygame.image.save(canvas, time.strftime("night_garden_%H%M%S.png"))

            now = time.perf_counter()
            if writer:
                t = frame_no / args.fps
            elif not paused:
                t += min(now - last, 0.1)
            last = now

            glyph, color = compose(cv, t)
            glyph, color = decoder.apply(glyph, color, t)
            canvas.fill((0, 0, 0))
            painter.draw(canvas, glyph, color, x0, y0)
            panel.draw(canvas, t, SCENES[int(t // SCENE_LEN) % len(SCENES)].__name__)

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
