"""
ASCII Bouquet HD
----------------
The same animation as ascii_bouquet.py (one flower -> a full bouquet, wrapped,
ribbon, "For You"), made with text characters - but sharper and clearer:

  * 2-3x more characters   bigger window + smaller font
  * sharper source         the 3D bouquet is drawn at 1.5x resolution with
                           smoother petals before it is turned into text
  * shape-matched chars    each cell picks the character whose SHAPE fits
                           best (ascii_detail.py) -> crisp outlines, visible shading
  * soft glow              a faint bloom behind the characters so they read
                           clearly on black, finer colour steps

ascii_bouquet.py is untouched; this file reuses its scene and converter.

Run
  python ascii_bouquet_hd.py
  python ascii_bouquet_hd.py --font-size 7          # even more detail
  python ascii_bouquet_hd.py --record bouquet.mp4   # save a video (30 s)
  python ascii_bouquet_hd.py --mode love            # ramp | mix | binary | digits | letters | love

Keys
  SPACE pause   R restart   M character mode   D detail on/off   G glow on/off
  + / - character size   S screenshot   Q / ESC quit   drag = rotate   wheel = zoom
  Resize the window freely (bigger = more characters), F11 = fullscreen
  Z zoom in   X zoom out   0 reset zoom (or the mouse wheel)
"""
import argparse
import os
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
from ascii_bouquet import MODES, AsciiConverter
from ascii_detail import GlyphMatcher
from window_util import Window, Zoom

W, H = b3d.W, b3d.H
SS = 1.5                                   # draw the 3D scene at 1.5x resolution


class HDScene:
    """The bouquet from bouquet_3d.py, rendered at SS x resolution on black."""

    def __init__(self, ss=SS):
        self.ss = ss
        self.frame = pygame.Surface((int(W * ss), int(H * ss)))
        self.cam = b3d.Camera()
        self.cam.dist = 1000.0                     # start a little zoomed out
        self.cam.ss = ss
        self.title_font = pygame.font.SysFont("arialblack,impact,arial", int(110 * ss), bold=True)
        self.restart()

    def restart(self):
        self.scene = b3d.Scene()
        self.petals = [b3d.FallingPetal() for _ in range(28)]
        for p in self.petals:
            p.size *= 1.8
        self.t = 0.0

    def step(self, dt, auto_spin=True):
        self.t += dt
        if auto_spin:
            self.cam.yaw += dt * 0.28
        for p in self.petals[:self._n_petals()]:
            p.update(dt, self.t)

    def _n_petals(self):
        return int(b3d.clamp((self.t - 3) / 15) * len(self.petals))

    def draw(self):
        f, t, ss = self.frame, self.t, self.ss
        f.fill((0, 0, 0))
        for p in self.petals[:self._n_petals()]:          # 2D petals, scaled up
            x, y, size = p.x, p.y, p.size
            p.x, p.y, p.size = x * ss, y * ss, size * ss
            p.draw(f)
            p.x, p.y, p.size = x, y, size
        self.cam.setup()
        b3d.draw_items(f, b3d.build(self.scene, self.cam, t))
        a = b3d.smooth((t - self.scene.text_start) / 1.5)
        if a > 0:
            txt = self.title_font.render("For You", True, (255, 120, 170))
            txt.set_alpha(int(255 * a))
            f.blit(txt, txt.get_rect(center=(int(W * ss / 2), int(70 * ss))))
        return f


def sample(frame, ss, cols, rows, cell_aspect, matcher):
    """Crop around the bouquet and shrink to the character grid (x sub-pixels)."""
    fw, fh = frame.get_size()
    crop_h = 770.0 * ss
    crop_w = crop_h * cols / (rows * cell_aspect)
    if crop_w > fw:
        crop_w = fw
        crop_h = fw * rows * cell_aspect / cols
    rect = pygame.Rect(0, 0, int(crop_w), int(min(crop_h, fh)))
    rect.center = (fw // 2, int(405 * ss))
    rect = rect.clamp(frame.get_rect())
    size = matcher.sample_size(cols, rows) if matcher else (cols, rows)
    small = pygame.transform.smoothscale(frame.subsurface(rect), size)
    rgb = np.frombuffer(pygame.image.tobytes(small, "RGB"), dtype=np.uint8)
    rgb = rgb.reshape(size[1], size[0], 3)
    return rgb if matcher else rgb.astype(np.float32)


class Painter:
    """Draws the character grid; caches each (char, colour) glyph image."""

    def __init__(self, font):
        self.font = font
        self.cw, self.ch = font.size("M")
        self.cache = {}

    def draw(self, surf, glyph, color):
        q = (color >> 3) << 3                    # 32 colour steps per channel
        if len(self.cache) > 60000:
            self.cache.clear()
        cache, font, cw, ch = self.cache, self.font, self.cw, self.ch
        seq = []
        ys, xs = np.nonzero(glyph != " ")
        for y, x in zip(ys.tolist(), xs.tolist()):
            c = q[y, x]
            key = (glyph[y, x], int(c[0]), int(c[1]), int(c[2]))
            img = cache.get(key)
            if img is None:
                img = cache[key] = font.render(key[0], True, key[1:])
            seq.append((img, (x * cw, y * ch)))
        if hasattr(surf, "fblits"):
            surf.fblits(seq)
        else:
            surf.blits(seq, doreturn=False)


def add_glow(surf, glyph, color, strength=0.32):
    """Soft bloom: each character's colour, blurred, added behind/around the text."""
    lit = color.astype(np.float32) * (glyph != " ")[..., None] * strength
    lit = cv2.GaussianBlur(lit, (0, 0), 1.2)
    small = pygame.surfarray.make_surface(np.clip(lit, 0, 255).astype(np.uint8).swapaxes(0, 1))
    surf.blit(pygame.transform.smoothscale(small, surf.get_size()), (0, 0), special_flags=pygame.BLEND_ADD)


def main():
    ap = argparse.ArgumentParser(description="HD version of the ASCII bouquet")
    ap.add_argument("--mode", choices=MODES, default="ramp")
    ap.add_argument("--font-size", type=int, default=9, help="smaller = more characters / detail")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-glow", action="store_true")
    ap.add_argument("--record", metavar="OUT.mp4", help="save a video instead of just playing")
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    pygame.init()
    b3d.SEGS = 9                                  # smoother petal curves (default is 6)
    info = pygame.display.Info()
    max_w = min(1280, info.current_w - 80) if info.current_w > 0 else 1280
    max_h = min(1000, info.current_h - 100) if info.current_h > 0 else 1000

    state = {"font_size": args.font_size}
    win = Window((max_w, max_h), "ASCII Bouquet HD")

    def layout():
        """Fit the character grid to the current window size."""
        font = pygame.font.SysFont("consolas,couriernew,monospace", state["font_size"], bold=True)
        painter = Painter(font)
        ww, wh = win.size
        cols, rows = max(10, ww // painter.cw), max(5, wh // painter.ch)
        art = pygame.Surface((cols * painter.cw, rows * painter.ch))
        return font, painter, cols, rows, art

    font, painter, cols, rows, screen = layout()
    scene = HDScene()
    conv = AsciiConverter(args.mode, True, decode=True)
    conv.black = 0.05
    conv.matcher = GlyphMatcher(font)
    glow = not args.no_glow
    zoom = Zoom(scene.cam, 1000.0)

    writer, total = None, 0
    if args.record:
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, screen.get_size())
        total = int(args.seconds * args.fps)
        print(f"Recording {total} frames ({cols}x{rows} characters) to {args.record} ...")

    clock = pygame.time.Clock()
    paused, dragging, frame_no = False, False, 0
    try:
        while True:
            dt = min(clock.tick(args.fps if not writer else 0) / 1000.0, 0.1)
            if writer:
                dt = 1.0 / args.fps
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return
                if win.handle(e):                      # resize / F11
                    continue
                if zoom.handle(e):                     # Z / X / 0 / mouse wheel
                    pygame.display.set_caption(f"ASCII Bouquet HD  [zoom {zoom.percent}%]")
                    continue
                if e.type == pygame.KEYDOWN:
                    k = e.key
                    if k in (pygame.K_ESCAPE, pygame.K_q):
                        return
                    if k == pygame.K_SPACE:
                        paused = not paused
                    elif k == pygame.K_r:
                        scene.restart()
                    elif k == pygame.K_m:
                        conv.mode = MODES[(MODES.index(conv.mode) + 1) % len(MODES)]
                    elif k == pygame.K_d:
                        conv.matcher = None if conv.matcher else GlyphMatcher(font)
                    elif k == pygame.K_g:
                        glow = not glow
                    elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS, pygame.K_MINUS, pygame.K_KP_MINUS) \
                            and not writer:
                        step = 1 if k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS) else -1
                        state["font_size"] = max(5, min(20, state["font_size"] + step))
                        font, painter, cols, rows, screen = layout()
                        if conv.matcher:
                            conv.matcher = GlyphMatcher(font)
                    elif k == pygame.K_s:
                        pygame.image.save(screen, time.strftime("ascii_bouquet_hd_%H%M%S.png"))
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    dragging = True
                if e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                    dragging = False
                if e.type == pygame.MOUSEMOTION and dragging:
                    scene.cam.yaw += e.rel[0] * 0.01
                    scene.cam.pitch = b3d.clamp(scene.cam.pitch - e.rel[1] * 0.006, -1.3, 0.35)

            if win.take_resized() and not writer:      # re-flow to the new window size
                font, painter, cols, rows, screen = layout()
                if conv.matcher:
                    conv.matcher = GlyphMatcher(font)
            if not paused:
                scene.step(dt, auto_spin=not dragging)
            rgb = sample(scene.draw(), scene.ss, cols, rows, painter.ch / painter.cw, conv.matcher)
            glyph, color = conv.convert(rgb, scene.t)

            screen.fill((0, 0, 0))
            painter.draw(screen, glyph, color)
            if glow:
                add_glow(screen, glyph, color)

            if writer:
                arr = pygame.surfarray.array3d(screen).swapaxes(0, 1)
                writer.write(cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_RGB2BGR))
                frame_no += 1
                pygame.display.set_caption(f"Recording {frame_no}/{total}")
                if frame_no >= total:
                    break
            ww, wh = win.size
            aw, ah = screen.get_size()
            if aw <= ww and ah <= wh:                  # grid fits: draw crisp, centred
                win.surface.fill((0, 0, 0))
                win.surface.blit(screen, ((ww - aw) // 2, (wh - ah) // 2))
            else:
                win.present(screen)
            pygame.display.flip()
    finally:
        if writer:
            writer.release()
            print(f"Saved {args.record}")
        pygame.quit()


if __name__ == "__main__":
    main()
