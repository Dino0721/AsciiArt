"""
3D Animated Bouquet (pygame)
----------------------------
A single flower grows first, then more stems, leaves and flowers appear one
by one, petals unfold in 3D, and finally the bunch gets wrapped into a
bouquet with a ribbon.  Everything is real 3D: petals are curved polygons,
shaded by a light source, depth-sorted, and the camera orbits around.

Controls
  mouse drag   rotate            mouse wheel  zoom (or Z / X, 0 = reset)
  SPACE        pause / resume    R            restart the animation
  S            save screenshot   ESC          quit
  F11          fullscreen        resize the window freely (the picture scales to fit)
  Z / X        zoom in / out     0  reset zoom

Run:  pip install pygame
      python bouquet_3d.py
      python bouquet_3d.py --detail   # 2x supersampled + smoother petals (ascii_detail.py)
"""
import math
import random
import sys

import pygame
import pygame.gfxdraw as gfx

# AsciiArt is split into folders - let this file find Common/ and Bouquet/
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ("Common", "Bouquet"):
    if _os.path.join(_ROOT, _d) not in _sys.path:
        _sys.path.insert(0, _os.path.join(_ROOT, _d))

W, H = 1000, 820
FPS = 60
SEED = 11
SEGS = 6                     # petal smoothness (--detail raises it)


# =============================================================== vectors ===
def add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def mul(a, s): return (a[0] * s, a[1] * s, a[2] * s)
def dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def norm(a):
    l = math.sqrt(dot(a, a)) or 1.0
    return (a[0] / l, a[1] / l, a[2] / l)


def clamp(x, lo=0.0, hi=1.0): return lo if x < lo else hi if x > hi else x
def smooth(t): t = clamp(t); return t * t * (3 - 2 * t)


def ease_out_back(t):
    t = clamp(t)
    c1 = 1.4
    return 1 + (c1 + 1) * (t - 1) ** 3 + c1 * (t - 1) ** 2


def basis(axis):
    """Two unit vectors perpendicular to axis (right-handed: e1 x e2 = axis)."""
    up = (0, 1, 0) if abs(axis[1]) < 0.9 else (1, 0, 0)
    e1 = norm(cross(up, axis))
    e2 = cross(axis, e1)
    return e1, e2


def shade(c, f):
    return (min(255, int(c[0] * f)), min(255, int(c[1] * f)), min(255, int(c[2] * f)))


def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def bezier3(p0, p1, p2, p3, t):
    u = 1 - t
    a, b, c, d = u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t
    return (a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
            a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
            a * p0[2] + b * p1[2] + c * p2[2] + d * p3[2])


# ================================================================ petal ===
def petal(center, A, e1, e2, phi, theta, length, width, curl, cup,
          color, inner_color, segs=6):
    """
    The 3D version of the classic turtle petal function.
    Returns two polygons (inner half / outer half) so each half gets its own
    lighting -> a soft gradient that makes the petal look curved.
    """
    cp, sp = math.cos(phi), math.sin(phi)
    R = (e1[0] * cp + e2[0] * sp, e1[1] * cp + e2[1] * sp, e1[2] * cp + e2[2] * sp)
    T = (-e1[0] * sp + e2[0] * cp, -e1[1] * sp + e2[1] * cp, -e1[2] * sp + e2[2] * cp)
    left, right, normals = [], [], []
    for i in range(segs + 1):
        u = i / segs
        th = theta + curl * u
        ct, st = math.cos(th), math.sin(th)
        d = (A[0] * ct + R[0] * st, A[1] * ct + R[1] * st, A[2] * ct + R[2] * st)
        n = cross(d, T)                                   # inner-side normal
        w = width * math.sin(math.pi * (0.1 + 0.9 * u)) ** 0.7
        c = add(center, mul(d, length * u))
        lift = mul(n, cup * w * 0.6)                      # cup the edges
        left.append(add(add(c, mul(T, w)), lift))
        right.append(add(sub(c, mul(T, w)), lift))
        normals.append(n)
    h = segs // 2
    base = left[:h + 1] + right[h::-1]
    tip = left[h:] + right[:h - 1:-1]
    return [(base, normals[h // 2], inner_color), (tip, normals[(h + segs) // 2], color)]


def disc(center, A, e1, e2, r, lift, color, n=14):
    c = add(center, mul(A, lift))
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        pts.append(add(c, add(mul(e1, math.cos(a) * r), mul(e2, math.sin(a) * r))))
    return (pts, A, color)


# ======================================================= flower recipes ===
# layer: (count, theta_open, theta_closed, length, width, curl, cup, color_mult)
RECIPES = {
    "rose": dict(layers=[(4, 0.25, 0.05, 0.55, 0.55, 0.15, 0.9, 0.70),
                         (5, 0.55, 0.10, 0.75, 0.60, 0.25, 0.8, 0.85),
                         (6, 0.90, 0.15, 0.95, 0.62, 0.35, 0.6, 1.00),
                         (7, 1.25, 0.20, 1.05, 0.65, 0.45, 0.45, 1.10)],
                 disc=None),
    "peony": dict(layers=[(6, 0.45, 0.08, 0.60, 0.45, 0.20, 0.9, 0.80),
                          (8, 0.95, 0.12, 0.85, 0.48, 0.30, 0.7, 0.95),
                          (10, 1.35, 0.18, 1.00, 0.45, 0.35, 0.5, 1.08)],
                  disc=None),
    "daisy": dict(layers=[(14, 1.40, 0.12, 0.92, 0.15, 0.05, 0.25, 0.90),
                          (14, 1.50, 0.15, 1.00, 0.15, -0.05, 0.25, 1.00)],
                  disc=((250, 190, 40), 0.28)),
    "cosmos": dict(layers=[(8, 1.30, 0.15, 1.00, 0.42, 0.10, 0.35, 1.00)],
                   disc=((110, 60, 30), 0.17)),
    "tulip": dict(layers=[(3, 0.30, 0.06, 1.15, 0.58, -0.10, 0.9, 0.85),
                          (3, 0.40, 0.08, 1.20, 0.60, -0.12, 0.9, 1.00)],
                  disc=None),
}

PALETTE = {
    "rose": [(215, 30, 55), (240, 110, 150), (255, 140, 110), (250, 245, 245)],
    "peony": [(250, 150, 185), (255, 190, 210), (235, 120, 170)],
    "daisy": [(255, 255, 255), (255, 225, 90), (255, 200, 225)],
    "cosmos": [(200, 120, 230), (255, 120, 170), (250, 160, 200)],
    "tulip": [(235, 60, 90), (170, 100, 225), (255, 170, 60)],
}

GREEN = (70, 160, 75)
DARK_GREEN = (40, 115, 55)


class Flower:
    def __init__(self, pos, axis, kind, color, size, t0):
        self.pos, self.axis, self.kind = pos, axis, kind
        self.color, self.size, self.t0 = color, size, t0
        self.e1, self.e2 = basis(axis)
        self.phase = random.uniform(0, 6.28)
        self.spin = random.uniform(0, 6.28)
        self.cache = None
        # stem: cubic curve from the hidden base up to the flower head
        x, y, z = pos
        self.stem = ((x * 0.03, -300, z * 0.03),
                     (x * 0.08, -170, z * 0.08),
                     sub(pos, mul(axis, 95)),
                     sub(pos, mul(axis, 4)))
        self.leaves = [(random.uniform(0.45, 0.62), random.uniform(0, 6.28)),
                       (random.uniform(0.66, 0.8), random.uniform(0, 6.28))]

    def head(self, bloom):
        """World-space polygons for the flower head at a bloom value 0..1+."""
        if bloom >= 1.0 and self.cache is not None:
            return self.cache
        polys = []
        A, e1, e2, S, c = self.axis, self.e1, self.e2, self.size, self.pos
        # green sepals behind the petals
        for k in range(5):
            polys += petal(c, A, e1, e2, self.spin + k * 1.2566, 0.6 + 1.2 * min(bloom, 1),
                           S * 0.45, S * 0.12, 0.2, 0.3, GREEN, DARK_GREEN, segs=4)
        recipe = RECIPES[self.kind]
        for li, (cnt, th_o, th_c, ln, wd, curl, cup, cm) in enumerate(recipe["layers"]):
            col = shade(self.color, cm)
            inner = mix(shade(col, 0.72), (120, 30, 60), 0.12)
            th = th_c + (th_o - th_c) * bloom
            L = S * ln * (0.45 + 0.55 * min(bloom, 1.05))
            for k in range(cnt):
                phi = self.spin + 2 * math.pi * k / cnt + li * 0.7
                polys += petal(c, A, e1, e2, phi, th, L, S * wd, curl * bloom, cup,
                               col, inner, segs=SEGS)
        if recipe["disc"] and bloom > 0.3:
            dcol, dr = recipe["disc"]
            r = S * dr * smooth((bloom - 0.3) / 0.5)
            polys.append(disc(c, A, e1, e2, r, S * 0.06, dcol))
            polys.append(disc(c, A, e1, e2, r * 0.6, S * 0.11, shade(dcol, 1.25)))
        if bloom >= 1.0:
            self.cache = polys
        return polys


# ============================================================== scene ====
class Scene:
    def __init__(self):
        random.seed(SEED)
        self.flowers = []
        N = 19
        kinds = ["daisy", "peony", "cosmos", "tulip", "rose"] * 4
        random.shuffle(kinds)
        t = 0.6
        for i in range(N):
            r = 178 * math.sqrt((i + 0.25) / N)
            a = i * 2.39996                             # golden angle spiral
            x, z = r * math.cos(a), r * math.sin(a)
            y = 140 - 0.004 * r * r + random.uniform(-8, 8)
            axis = norm((x * 1.25, y + 140, z * 1.25))
            kind = "rose" if i == 0 else kinds[i]
            color = random.choice(PALETTE[kind])
            size = random.uniform(36, 44) if kind != "daisy" else random.uniform(30, 36)
            self.flowers.append(Flower((x, y, z), axis, kind, color, size, t))
            t += 2.2 if i == 0 else 0.7                 # first flower blooms alone
        self.flowers_done = t + 2.0

        # greenery sprays around the edge
        self.sprays = []
        for k in range(9):
            a = k * 2 * math.pi / 9 + 0.3
            base = (math.cos(a) * 165, -5, math.sin(a) * 165)
            axis = norm((math.cos(a), 0.45, math.sin(a)))
            self.sprays.append((base, axis, t * 0.3 + k * t * 0.06))

        # baby's breath
        self.dots = []
        for _ in range(60):
            a = random.uniform(0, 6.28)
            r = random.uniform(20, 200)
            y = 150 - 0.004 * r * r + random.uniform(5, 30)
            self.dots.append(((math.cos(a) * r, y, math.sin(a) * r),
                              random.uniform(2.2, 3.8),
                              random.uniform(self.flowers_done * 0.55, self.flowers_done)))

        self.wrap_start = self.flowers_done - 0.5
        self.ribbon_start = self.wrap_start + 2.2
        self.text_start = self.ribbon_start + 0.8


# ============================================================= renderer ===
class Camera:
    def __init__(self):
        self.yaw, self.pitch, self.dist, self.f = 0.6, -0.42, 1000.0, 1050.0
        self.ss = 1                  # supersampling factor (draw bigger, scale down later)

    def setup(self):
        self.cy, self.sy = math.cos(self.yaw), math.sin(self.yaw)
        self.cp, self.sp = math.cos(self.pitch), math.sin(self.pitch)

    def rot(self, p):
        x, y, z = p
        x1 = x * self.cy + z * self.sy
        z1 = -x * self.sy + z * self.cy
        return (x1, y * self.cp - z1 * self.sp, y * self.sp + z1 * self.cp)

    def proj(self, p, off=(0, 0, 0)):
        x, y, z = p[0] + off[0], p[1] + off[1] + 70, p[2] + off[2]
        x1 = x * self.cy + z * self.sy
        z1 = -x * self.sy + z * self.cy
        y2 = y * self.cp - z1 * self.sp
        z2 = y * self.sp + z1 * self.cp
        s = self.f / (self.dist + z2) * self.ss
        return (W * 0.5 * self.ss + x1 * s, H * 0.5 * self.ss - y2 * s, z2, s)


LIGHT = norm((-0.45, 0.65, -0.6))


def lit(cam, n, color, amb=0.38):
    n = cam.rot(n)
    back = 1.0
    if n[2] > 0:                       # facing away -> show the back side
        n = (-n[0], -n[1], -n[2])
        back = 0.8
    lam = max(0.0, dot(n, LIGHT))
    spec = lam ** 14 * 70
    k = (amb + (1 - amb) * lam) * back
    return (min(255, int(color[0] * k + spec)),
            min(255, int(color[1] * k + spec)),
            min(255, int(color[2] * k + spec)))


def add_poly(items, cam, pts, n, color, off=(0, 0, 0), amb=0.38):
    p2, zs = [], 0.0
    for p in pts:
        sx, sy, z, _ = cam.proj(p, off)
        p2.append((int(sx), int(sy)))
        zs += z
    items.append((zs / len(pts), 0, p2, lit(cam, n, color, amb)))


def build_wrap(scene, cam, items, t):
    g = smooth((t - scene.wrap_start) / 2.0)
    if g <= 0:
        return
    yt, yr, Rr, segs, rings = -330.0, -45.0, 150.0, 28, 4
    outer, inner = (250, 175, 195), (255, 230, 236)
    top = yt + (yr - yt) * g
    for j in range(rings):
        v0, v1 = j / rings, (j + 1) / rings
        y0, y1 = yt + (top - yt) * v0, yt + (top - yt) * v1
        for i in range(segs):
            a0, a1 = 2 * math.pi * i / segs, 2 * math.pi * (i + 1) / segs
            quad = []
            for (a, y, v) in ((a0, y0, v0), (a1, y0, v0), (a1, y1, v1), (a0, y1, v1)):
                frac = (y - yt) / (yr - yt)
                wave = 1 + 0.07 * math.sin(a * 7) * frac ** 3      # scalloped rim
                r = (8 + (Rr - 8) * frac) * wave
                yy = y + 14 * math.sin(a * 7 + 1) * frac ** 4
                quad.append((math.cos(a) * r, yy, math.sin(a) * r))
            am = (a0 + a1) / 2
            n = norm((math.cos(am), -Rr / (yr - yt), math.sin(am)))
            pleat = 1.0 if i % 2 else 0.9
            nv = cam.rot(n)
            col = shade(outer, pleat) if nv[2] < 0 else shade(inner, pleat)
            add_poly(items, cam, quad, n, col, amb=0.7)


def build_ribbon(scene, cam, items, t):
    k = ease_out_back((t - scene.ribbon_start) / 0.8)
    if k <= 0:
        return
    red, yb = (205, 25, 60), -215.0
    frac = (yb - -330) / (-45 - -330)
    rb = 8 + 142 * frac + 3
    segs = 28
    for i in range(segs):                               # ribbon band
        a0, a1 = 2 * math.pi * i / segs, 2 * math.pi * (i + 1) / segs
        quad = [(math.cos(a0) * rb, yb - 9, math.sin(a0) * rb),
                (math.cos(a1) * rb, yb - 9, math.sin(a1) * rb),
                (math.cos(a1) * rb, yb + 9, math.sin(a1) * rb),
                (math.cos(a0) * rb, yb + 9, math.sin(a0) * rb)]
        am = (a0 + a1) / 2
        add_poly(items, cam, quad, (math.cos(am), 0, math.sin(am)), red, amb=0.6)
    # bow on the front
    C = (0, yb, -rb - 6)
    X, Y, out = (1, 0, 0), (0, 1, 0), (0, 0, -1)
    for sgn in (-1, 1):
        loop = []
        for i in range(14):
            u = 2 * math.pi * i / 14
            lx, ly = 30 * (1 - math.cos(u)) / 2 * 1.0, 13 * math.sin(u) * (0.4 + 0.6 * (1 - math.cos(u)) / 2)
            ang = 0.35 * sgn
            px = (lx * math.cos(ang) - ly * math.sin(ang)) * sgn * k
            py = (lx * math.sin(ang) * sgn + ly * math.cos(ang)) * k
            loop.append(add(C, add(mul(X, px), add(mul(Y, py), mul(out, 3)))))
        add_poly(items, cam, loop, norm((sgn * 0.3, 0.2, -1)), red)
        tail = [C, add(C, (sgn * 10 * k, -8 * k, -2)),
                add(C, (sgn * 26 * k, -55 * k, -2)), add(C, (sgn * 12 * k, -48 * k, -2))]
        add_poly(items, cam, tail, norm((sgn * 0.2, -0.1, -1)), shade(red, 0.9))
    sx, sy, z, s = cam.proj(add(C, (0, 0, -5)))
    items.append((z, 2, (int(sx), int(sy)), max(2, int(8 * s * k)), shade(red, 1.15)))


def build(scene, cam, t):
    items = []
    for idx, fl in enumerate(scene.flowers):
        lt = t - fl.t0
        if lt <= 0:
            continue
        grow = smooth(lt / 1.0)
        bloom = ease_out_back((lt - 0.9) / 1.8) if lt > 0.9 else -1
        sway = (math.sin(t * 1.3 + fl.phase) * 3.0, math.sin(t * 0.9 + fl.phase) * 1.0,
                math.cos(t * 1.1 + fl.phase) * 3.0)

        # stem grows segment by segment
        nseg = 12
        last = None
        for s in range(nseg + 1):
            u = s / nseg * grow
            p = bezier3(*fl.stem, u)
            o = mul(sway, u * u)
            sx, sy, z, sc = cam.proj(p, o)
            if last:
                lx, ly, lz, lsc = last
                w = max(1, int(5 * sc))
                col = lit(cam, (0, 0, -1), shade(GREEN, 0.8 + 0.25 * u), 0.6)
                items.append(((z + lz) / 2, 1, (int(lx), int(ly)), (int(sx), int(sy)), w, col))
            last = (sx, sy, z, sc)

        # leaves pop out once the stem passes them
        for (lu, lphi) in fl.leaves:
            if grow > lu:
                k = smooth((grow - lu) / 0.25)
                p = bezier3(*fl.stem, lu)
                tan = norm(sub(bezier3(*fl.stem, lu + 0.02), p))
                e1, e2 = basis(tan)
                for poly in petal(p, tan, e1, e2, lphi, 0.9, 42 * k, 11 * k, 0.5, 0.35,
                                  GREEN, DARK_GREEN, segs=4):
                    add_poly(items, cam, poly[0], poly[1], poly[2], mul(sway, lu * lu))

        # flower head: a closed bud that unfolds
        if bloom > -1:
            b = max(0.0, bloom)
            for pts, n, col in fl.head(b):
                add_poly(items, cam, pts, n, col, sway)

    # greenery sprays
    for base, axis, ts in scene.sprays:
        k = smooth((t - ts) / 1.2)
        if k <= 0:
            continue
        e1, e2 = basis(axis)
        for j in range(3):
            phi = 1.0 + j * 2.1
            for poly in petal(base, axis, e1, e2, phi, 0.25 + 0.15 * j, 70 * k, 16 * k,
                              0.6, 0.3, shade(GREEN, 0.95 + 0.08 * j), DARK_GREEN, segs=4):
                add_poly(items, cam, poly[0], poly[1], poly[2])

    # baby's breath
    for p, r, ts in scene.dots:
        k = smooth((t - ts) / 0.6)
        if k > 0:
            sx, sy, z, s = cam.proj(p)
            items.append((z, 2, (int(sx), int(sy)), max(1, int(r * s * k)), (255, 255, 250)))

    build_wrap(scene, cam, items, t)
    build_ribbon(scene, cam, items, t)
    return items


def draw_items(surf, items):
    items.sort(key=lambda it: -it[0])          # far to near (painter's algorithm)
    for it in items:
        kind = it[1]
        if kind == 0:
            pts, col = it[2], it[3]
            if len(pts) >= 3:
                gfx.filled_polygon(surf, pts, col)
                gfx.aapolygon(surf, pts, col)
        elif kind == 1:
            _, _, a, b, w, col = it
            pygame.draw.line(surf, col, a, b, w)
            if w > 2:
                pygame.draw.circle(surf, col, b, w // 2)
        else:
            _, _, c, r, col = it
            gfx.filled_circle(surf, c[0], c[1], r, col)
            gfx.aacircle(surf, c[0], c[1], r, col)


# ============================================================ background ===
def make_background():
    bg = pygame.Surface((W, H))
    top, bottom = (14, 18, 40), (58, 28, 70)
    for y in range(H):
        pygame.draw.line(bg, mix(top, bottom, y / H), (0, y), (W, y))
    glow = pygame.Surface((W, H), pygame.SRCALPHA)
    for r in range(380, 0, -6):
        a = int(40 * (1 - r / 380) ** 1.5)
        pygame.draw.circle(glow, (255, 150, 200, a), (W // 2, int(H * 0.42)), r)
    bg.blit(glow, (0, 0))
    return bg


class FallingPetal:
    def __init__(self, start_top=False):
        self.reset(start_top)

    def reset(self, top=True):
        self.x = random.uniform(-50, W + 50)
        self.y = random.uniform(-80, -10) if top else random.uniform(0, H)
        self.vy = random.uniform(25, 55)
        self.vx = random.uniform(-15, 15)
        self.rot = random.uniform(0, 6.28)
        self.vr = random.uniform(-2, 2)
        self.size = random.uniform(5, 10)
        self.col = random.choice([(255, 170, 200), (255, 120, 160), (255, 220, 235), (240, 90, 120)])
        self.front = random.random() < 0.3

    def update(self, dt, t):
        self.y += self.vy * dt
        self.x += (self.vx + math.sin(t * 1.5 + self.rot) * 20) * dt
        self.rot += self.vr * dt
        if self.y > H + 20:
            self.reset()

    def draw(self, surf):
        flip = abs(math.sin(self.rot * 1.7)) * 0.7 + 0.3       # fake 3D tumble
        pts = []
        for i in range(10):
            a = 2 * math.pi * i / 10
            px = math.cos(a) * self.size
            py = math.sin(a) * self.size * 0.55 * flip
            ca, sa = math.cos(self.rot), math.sin(self.rot)
            pts.append((int(self.x + px * ca - py * sa), int(self.y + px * sa + py * ca)))
        col = shade(self.col, 0.7 + 0.3 * flip)
        gfx.filled_polygon(surf, pts, col)
        gfx.aapolygon(surf, pts, col)


# ================================================================= main ===
def main():
    global SEGS
    pygame.init()
    from window_util import Window, Zoom
    win = Window((W, H), "3D Bouquet")
    screen = pygame.Surface((W, H))              # draw here, then scale to the window
    detail = "--detail" in sys.argv
    sampler = None
    if detail:
        from ascii_detail import SuperSampler
        sampler = SuperSampler((W, H), 2)        # draw at 2x, scale down = smooth edges
        SEGS = 10
    pygame.display.set_caption("3D Bouquet" + ("  [detail]" if detail else ""))
    clock = pygame.time.Clock()
    bg = make_background()
    stars = [(random.randint(0, W), random.randint(0, H), random.uniform(0, 6.28),
              random.choice((1, 1, 2))) for _ in range(120)]
    try:
        title_font = pygame.font.SysFont("segoescript,brushscriptmt,georgia", 52, bold=True)
    except Exception:
        title_font = pygame.font.Font(None, 60)
    hint_font = pygame.font.SysFont("segoeui,arial", 16)

    scene = Scene()
    cam = Camera()
    if sampler:
        cam.ss = sampler.factor
    zoom = Zoom(cam, 1150.0)                     # start a little zoomed out
    petals = [FallingPetal() for _ in range(35)]
    t, paused, dragging = 0.0, False, False

    while True:
        dt = clock.tick(FPS) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()
            if win.handle(e):                        # resize / F11
                continue
            if zoom.handle(e):                       # Z / X / 0 / mouse wheel
                pygame.display.set_caption(f"3D Bouquet  [zoom {zoom.percent}%]")
                continue
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_r:
                    scene, t = Scene(), 0.0
                elif e.key == pygame.K_s:
                    pygame.image.save(screen, "bouquet_screenshot.png")
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                dragging = True
            if e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                dragging = False
            if e.type == pygame.MOUSEMOTION and dragging:
                cam.yaw += e.rel[0] * 0.01
                cam.pitch = clamp(cam.pitch - e.rel[1] * 0.006, -1.3, 0.35)

        if not paused:
            t += dt
            if not dragging:
                cam.yaw += dt * 0.28

        # ---- background
        screen.blit(bg, (0, 0))
        for (x, y, ph, s) in stars:
            b = int(140 + 115 * math.sin(t * 2 + ph))
            screen.fill((b, b, min(255, b + 30)), (x, y, s, s))
        n_petals = int(clamp((t - 3) / 15) * len(petals))
        for p in petals[:n_petals]:
            if not paused:
                p.update(dt, t)
            if not p.front:
                p.draw(screen)

        # ---- the bouquet
        cam.setup()
        if sampler:
            draw_items(sampler.begin(screen), build(scene, cam, t))
            sampler.end(screen)
        else:
            draw_items(screen, build(scene, cam, t))

        for p in petals[:n_petals]:
            if p.front:
                p.draw(screen)

        # ---- text
        a = smooth((t - scene.text_start) / 1.5)
        if a > 0:
            txt = title_font.render("For You", True, (255, 215, 235))
            txt.set_alpha(int(255 * a))
            screen.blit(txt, txt.get_rect(center=(W // 2, 70 - int(15 * (1 - a)))))
        hint = hint_font.render("drag: rotate   wheel / Z X: zoom   0: reset   SPACE: pause   R: restart   F11: full",
                                True, (170, 160, 200))
        screen.blit(hint, hint.get_rect(center=(W // 2, H - 18)))

        win.present(screen)
        pygame.display.flip()


if __name__ == "__main__":
    main()
