"""
ASCII Bouquet
-------------
The 3D bouquet from bouquet_3d.py, but every flower, leaf, stem, falling
petal and the "For You" title is turned into coloured text characters,
like the ASCII-art flower videos.

How it works
  1. Render the 3D scene to an off-screen image (pygame).
  2. Shrink the image so each pixel = one character cell.
  3. Brightness  -> a character from a density ramp (" .:-=+*#%@" style).
  4. Sobel edges -> outlines are drawn with  -  \\  |  /  by edge direction.
  5. Pixel colour -> the character's colour.
  New cells "decode" from random symbols before they settle.

Run
  python ascii_bouquet.py              # pop-out window (default, good for recording)
  python ascii_bouquet.py --terminal   # pure ASCII inside the terminal
  python ascii_bouquet.py --mode love  # chars: ramp | mix | binary | digits | letters | love
  python ascii_bouquet.py --mono       # no colour, just characters
  python ascii_bouquet.py --detail     # shape-matched characters (ascii_detail.py) = sharper

Keys (terminal and window)
  SPACE pause   R restart   M change character mode   E toggle edges   D detail on/off
  Z zoom in   X zoom out   0 reset zoom (or the mouse wheel)
  Q / ESC quit
  (window only: drag to rotate, wheel to zoom, resize the window freely, F11 fullscreen)
"""
import argparse
import os
import random
import shutil
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

import bouquet_3d as b3d
from ascii_detail import GlyphMatcher
from window_util import Window, Zoom

W, H = b3d.W, b3d.H

RAMP = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
MIX_TIERS = [".,'`:", ";-~^\"", "+=<>*/|!", "abcdefghijkmnopqrstuvwxyz",
             "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789", "#$%&@8BW"]
SCRAMBLE = "!@#$%^&*01<>?/\\|{}[]ABCDEFXYZ0123456789"
EDGE_CHARS = np.array(["-", "\\", "|", "/"])
MODES = ["ramp", "mix", "binary", "digits", "letters", "love"]


# ======================================================= scene rendering ===
class SceneRenderer:
    """Draws the 3D bouquet (from bouquet_3d.py) on a black off-screen frame."""

    def __init__(self):
        self.frame = pygame.Surface((W, H))
        self.cam = b3d.Camera()
        self.cam.dist = 1000.0                     # start a little zoomed out (Z / X to zoom)
        self.title_font = pygame.font.SysFont("arialblack,impact,arial", 110, bold=True)
        self.restart()

    def restart(self):
        self.scene = b3d.Scene()
        self.petals = [b3d.FallingPetal() for _ in range(28)]
        for p in self.petals:
            p.size *= 2.2                         # bigger, so they survive the shrink
        self.t = 0.0

    def step(self, dt, auto_spin=True):
        self.t += dt
        if auto_spin:
            self.cam.yaw += dt * 0.28
        n = int(b3d.clamp((self.t - 3) / 15) * len(self.petals))
        for p in self.petals[:n]:
            p.update(dt, self.t)

    def draw(self):
        f, t = self.frame, self.t
        f.fill((0, 0, 0))
        n = int(b3d.clamp((t - 3) / 15) * len(self.petals))
        for p in self.petals[:n]:
            p.draw(f)
        self.cam.setup()
        b3d.draw_items(f, b3d.build(self.scene, self.cam, t))
        a = b3d.smooth((t - self.scene.text_start) / 1.5)
        if a > 0:
            txt = self.title_font.render("For You", True, (255, 120, 170))
            txt.set_alpha(int(255 * a))
            f.blit(txt, txt.get_rect(center=(W // 2, 70)))
        return f


# ======================================================= ASCII converter ===
class AsciiConverter:
    def __init__(self, mode="ramp", edges=True, decode=True):
        self.mode, self.edges, self.decode = mode, edges, decode
        self.edge_thresh = 0.5
        self.black = 0.07                      # darker than this -> blank cell
        self.matcher = None                    # GlyphMatcher -> shape-matched detail mode
        self.shape = None

    def _resize(self, cols, rows):
        self.shape = (rows, cols)
        rng = np.random.default_rng(7)
        self.rand = rng.integers(0, 1 << 16, size=(rows, cols))
        self.birth = np.full((rows, cols), -1.0)
        yy, xx = np.mgrid[0:rows, 0:cols]
        self.love_idx = (xx + yy * 3) % 8

    def sample(self, frame, cols, rows, cell_aspect):
        """Shrink the interesting part of the frame to cols x rows pixels."""
        crop_h = 770.0
        crop_w = crop_h * cols / (rows * cell_aspect)
        if crop_w > W:                                  # very wide terminal
            crop_w = W
            crop_h = W * rows * cell_aspect / cols
        rect = pygame.Rect(0, 0, int(crop_w), int(min(crop_h, H)))
        rect.center = (W // 2, 405)
        rect = rect.clamp(frame.get_rect())
        size = self.matcher.sample_size(cols, rows) if self.matcher else (cols, rows)
        small = pygame.transform.smoothscale(frame.subsurface(rect), size)
        rgb = np.frombuffer(pygame.image.tobytes(small, "RGB"), dtype=np.uint8)
        return rgb.reshape(size[1], size[0], 3).astype(np.float32)

    def convert(self, rgb, t):
        shape_glyph = None
        if self.matcher is not None and rgb.shape[0] % self.matcher.sub[1] == 0                 and rgb.shape[1] % self.matcher.sub[0] == 0 and rgb.shape[0] > 0:
            # detail mode: rgb is the sub-sampled image -> match character shapes
            sx, sy = self.matcher.sub
            rows, cols = rgb.shape[0] // sy, rgb.shape[1] // sx
            shape_glyph, rgb, _ = self.matcher.match(rgb, cols, rows)
            mean = self.matcher.last_mean
        rows, cols = rgb.shape[:2]
        if self.shape != (rows, cols):
            self._resize(cols, rows)

        lum = rgb.max(axis=2) / 255.0                  # 0..1 brightness
        filled = lum > self.black
        if shape_glyph is not None:
            filled &= (mean > self.black * 0.5) & (shape_glyph != " ")

        # ---- characters from brightness
        if shape_glyph is not None and self.mode in ("ramp", "mix"):
            glyph = shape_glyph                        # detail: best-fitting character shape
        elif self.mode == "ramp":
            n = len(RAMP)
            idx = np.clip((lum * (n - 1) * 1.15 + 0.5).astype(np.int32), 1, n - 1)
            glyph = np.array(list(RAMP))[idx]
        elif self.mode == "mix":
            tiers = len(MIX_TIERS)
            tier = np.clip((lum * tiers).astype(np.int32), 0, tiers - 1)
            glyph = np.empty((rows, cols), dtype="<U1")
            for k, chars in enumerate(MIX_TIERS):
                m = tier == k
                glyph[m] = np.array(list(chars))[self.rand[m] % len(chars)]
        elif self.mode == "binary":
            glyph = np.where(self.rand % 2 == 0, "0", "1")
        elif self.mode == "digits":
            glyph = np.array(list("0123456789"))[self.rand % 10]
        elif self.mode == "letters":
            glyph = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))[self.rand % 26]
        else:  # love
            glyph = np.array(list("ILOVEYOU"))[self.love_idx]

        # ---- Sobel edges -> direction characters
        if self.edges and not (shape_glyph is not None and self.mode in ("ramp", "mix")):
            g = np.pad(lum, 1, mode="edge")
            gx = (g[:-2, 2:] + 2 * g[1:-1, 2:] + g[2:, 2:]
                  - g[:-2, :-2] - 2 * g[1:-1, :-2] - g[2:, :-2])
            gy = (g[2:, :-2] + 2 * g[2:, 1:-1] + g[2:, 2:]
                  - g[:-2, :-2] - 2 * g[:-2, 1:-1] - g[:-2, 2:])
            mag = np.hypot(gx, gy) / 4.0
            ang = (np.degrees(np.arctan2(gy, gx)) + 90.0) % 180.0
            sel = EDGE_CHARS[(((ang + 22.5) % 180.0) // 45).astype(np.int32) % 4]
            edge = (mag > self.edge_thresh) & filled
            glyph = np.where(edge, sel, glyph)

        # ---- "decode" effect: new cells flicker through random symbols
        self.birth[filled & (self.birth < 0)] = t
        self.birth[~filled] = -1.0
        young = filled & (t - self.birth < 0.35) if self.decode else np.zeros_like(filled)
        if young.any():
            scr = np.array(list(SCRAMBLE))
            glyph = np.where(young, scr[np.random.randint(0, len(scr), size=(rows, cols))], glyph)

        glyph = np.where(filled, glyph, " ")

        # ---- colours: keep the hue, lift dark cells so they stay readable
        mx = np.maximum(rgb.max(axis=2, keepdims=True), 1.0)
        target = (140 + 115 * lum)[..., None]
        color = rgb / mx * target
        grey = color.mean(axis=2, keepdims=True)
        color = grey + (color - grey) * 1.6            # punchier colours
        color = np.where(young[..., None], color * 0.5 + 127, color)
        return glyph, np.clip(color, 0, 255).astype(np.uint8)


# ======================================================== terminal output ===
class Keys:
    """Non-blocking single key reads in the terminal (Windows + POSIX)."""

    def __init__(self):
        self.win = os.name == "nt"
        self.old = None
        if not self.win and sys.stdin.isatty():
            import termios
            import tty
            self.old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

    def get(self):
        if self.win:
            import msvcrt
            if msvcrt.kbhit():
                return msvcrt.getwch().lower()
            return None
        import select
        if self.old is not None and select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1).lower()
        return None

    def close(self):
        if self.old is not None:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old)


def run_terminal(args):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    if os.name == "nt":
        os.system("")                                  # enable ANSI colours on Windows
    renderer = SceneRenderer()
    conv = AsciiConverter(args.mode, not args.no_edges)
    if args.detail:
        conv.matcher = GlyphMatcher()
        b3d.SEGS = 10
    zoom = Zoom(renderer.cam, 1000.0)
    keys = Keys()
    out = sys.stdout
    out.write("\x1b[?25l\x1b[2J")                      # hide cursor, clear
    paused, last = False, time.perf_counter()
    frame_time = 1.0 / args.fps
    try:
        while True:
            now = time.perf_counter()
            dt, last = min(now - last, 0.1), now
            k = keys.get()
            if k in ("q", "\x1b"):
                break
            if k == " ":
                paused = not paused
            elif k == "r":
                renderer.restart()
            elif k == "m":
                conv.mode = MODES[(MODES.index(conv.mode) + 1) % len(MODES)]
            elif k == "e":
                conv.edges = not conv.edges
            elif k == "d":
                conv.matcher = None if conv.matcher else GlyphMatcher()
            elif k == "z":
                zoom.step(1)
            elif k == "x":
                zoom.step(-1)
            elif k == "0":
                zoom.reset()
            if not paused:
                renderer.step(dt)

            size = shutil.get_terminal_size((120, 40))
            cols, rows = max(20, size.columns - 1), max(10, size.lines - 1)
            rgb = conv.sample(renderer.draw(), cols, rows, cell_aspect=2.0)
            glyph, color = conv.convert(rgb, renderer.t)

            parts = ["\x1b[H"]
            q = (color >> 3) << 3                      # fewer colour changes = faster
            for y in range(rows):
                row_g, row_c = glyph[y], q[y]
                prev = None
                for x in range(cols):
                    ch = row_g[x]
                    if ch != " " and not args.mono:
                        c = (row_c[x, 0], row_c[x, 1], row_c[x, 2])
                        if c != prev:
                            parts.append("\x1b[38;2;%d;%d;%dm" % c)
                            prev = c
                    parts.append(ch)
                parts.append("\x1b[0m\n" if y < rows - 1 else "\x1b[0m")
            out.write("".join(parts))
            out.flush()

            spare = frame_time - (time.perf_counter() - now)
            if spare > 0:
                time.sleep(spare)
    except KeyboardInterrupt:
        pass
    finally:
        keys.close()
        out.write("\x1b[0m\x1b[?25h\x1b[2J\x1b[H")
        out.flush()
        pygame.quit()


# ========================================================== window output ===
def run_window(args):
    pygame.init()
    font = pygame.font.SysFont("consolas,couriernew,monospace", args.font_size, bold=True)
    cw, ch = font.size("M")
    cols, rows = 1000 // cw, 860 // ch
    win = Window((cols * cw, rows * ch), "ASCII Bouquet")
    screen = win.surface
    x0 = y0 = 0
    clock = pygame.time.Clock()
    renderer = SceneRenderer()
    conv = AsciiConverter(args.mode, not args.no_edges)
    if args.detail:
        conv.matcher = GlyphMatcher(font)          # match against the exact font on screen
        b3d.SEGS = 10                              # smoother 3D petals too
    zoom = Zoom(renderer.cam, 1000.0)
    cache = {}
    paused, dragging = False, False

    while True:
        dt = min(clock.tick(args.fps) / 1000.0, 0.1)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                return
            if win.handle(e):                          # resize / F11
                continue
            if zoom.handle(e):                         # Z / X / 0 / mouse wheel
                pygame.display.set_caption(f"ASCII Bouquet  [zoom {zoom.percent}%]")
                continue
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    return
                if e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_r:
                    renderer.restart()
                elif e.key == pygame.K_m:
                    conv.mode = MODES[(MODES.index(conv.mode) + 1) % len(MODES)]
                    pygame.display.set_caption(f"ASCII Bouquet  [{conv.mode}]")
                elif e.key == pygame.K_e:
                    conv.edges = not conv.edges
                elif e.key == pygame.K_d:
                    conv.matcher = None if conv.matcher else GlyphMatcher(font)
                    pygame.display.set_caption(f"ASCII Bouquet  [detail {'on' if conv.matcher else 'off'}]")
                elif e.key == pygame.K_s:
                    pygame.image.save(screen, "ascii_bouquet.png")
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                dragging = True
            if e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                dragging = False
            if e.type == pygame.MOUSEMOTION and dragging:
                renderer.cam.yaw += e.rel[0] * 0.01
                renderer.cam.pitch = b3d.clamp(renderer.cam.pitch - e.rel[1] * 0.006, -1.3, 0.35)

        if win.take_resized():                         # re-flow: bigger window = more characters
            screen = win.surface
            ww, wh = win.size
            cols, rows = max(10, ww // cw), max(5, wh // ch)
            x0, y0 = (ww - cols * cw) // 2, (wh - rows * ch) // 2
        if not paused:
            renderer.step(dt, auto_spin=not dragging)
        rgb = conv.sample(renderer.draw(), cols, rows, cell_aspect=ch / cw)
        glyph, color = conv.convert(rgb, renderer.t)

        screen.fill((0, 0, 0))
        q = (color >> 4) << 4
        if args.mono:
            q[:] = 220
        blits = []
        ys, xs = np.nonzero(glyph != " ")
        for y, x in zip(ys.tolist(), xs.tolist()):
            key = (glyph[y, x], int(q[y, x, 0]), int(q[y, x, 1]), int(q[y, x, 2]))
            surf = cache.get(key)
            if surf is None:
                surf = font.render(key[0], True, key[1:])
                cache[key] = surf
            blits.append((surf, (x0 + x * cw, y0 + y * ch)))
        screen.blits(blits, doreturn=False)
        pygame.display.flip()


def main():
    ap = argparse.ArgumentParser(description="3D bouquet drawn with text characters")
    ap.add_argument("--terminal", action="store_true", help="draw inside the terminal instead of a window")
    ap.add_argument("--mode", choices=MODES, default="ramp", help="which characters to draw with")
    ap.add_argument("--no-edges", action="store_true", help="turn off the - \\ | / outline characters")
    ap.add_argument("--mono", action="store_true", help="no colours")
    ap.add_argument("--detail", action="store_true", help="shape-matched characters for sharper detail")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--font-size", type=int, default=11, help="window mode character size")
    args = ap.parse_args()
    if args.terminal:
        run_terminal(args)
    else:
        run_window(args)


if __name__ == "__main__":
    main()
