"""
ASCII Player
------------
Plays any video, image or your webcam as live coloured ASCII art in a
pop-out window, the same way as the "night garden made of text" TikTok:

  every frame -> shrink to a grid -> brightness picks a character
              -> Sobel edges become  -  \\  |  /  -> each char keeps its colour

Run
  python ascii_player.py                         # pick a file in a dialog
  python ascii_player.py flowers.mp4             # play a video (loops)
  python ascii_player.py lotus.jpg               # a still image
  python ascii_player.py 0                       # webcam
  python ascii_player.py clip.mp4 --size 540x960 # TikTok-shaped vertical window
  python ascii_player.py clip.mp4 --record out.mp4   # save the ASCII version as a video
  python ascii_player.py clip.mp4 --detail       # shape-matched characters = sharper (ascii_detail.py)

Keys
  SPACE pause        LEFT/RIGHT seek 5 s     R restart
  M  character mode  E  edges on/off         C  colour / mono
  F  fit / fill      + / -  character size   S  screenshot   D  detail on/off
  Q / ESC quit       drag & drop a new file onto the window to play it
  Resize the window freely (bigger = more characters), F11 = fullscreen
  Z zoom in   X zoom out   0 reset zoom (or the mouse wheel)
"""
import argparse
import os
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

from ascii_bouquet import MODES, AsciiConverter
from ascii_detail import GlyphMatcher
from window_util import ViewZoom, Window, zoom_image

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
MAX_W, MAX_H = 1100, 900


# ================================================================ source ===
class Source:
    """Video file, still image or webcam, with simple looping and seeking."""

    def __init__(self, src):
        self.name = os.path.basename(str(src))
        self.image = None
        self.cap = None
        self.live = False
        ext = os.path.splitext(str(src))[1].lower()
        if str(src).isdigit():
            self.cap = cv2.VideoCapture(int(src))
            self.live = True
        elif ext in IMAGE_EXT:
            self.image = cv2.imread(str(src))
            if self.image is None:
                raise SystemExit(f"Could not open image: {src}")
        else:
            self.cap = cv2.VideoCapture(str(src))
        if self.cap is not None and not self.cap.isOpened():
            raise SystemExit(f"Could not open video: {src}")
        self.fps = (self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 0) or 30.0
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap and not self.live else 0
        self.index = 0
        self.frame = self.image
        self.ended = False
        if self.frame is None:
            self.next_frame()

    @property
    def size(self):
        h, w = self.frame.shape[:2]
        return w, h

    @property
    def seconds(self):
        return self.index / self.fps

    def next_frame(self, loop=True):
        if self.image is not None:
            return self.frame
        ok, frame = self.cap.read()
        if not ok:
            if self.live or not loop:
                self.ended = True
                return self.frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.index = 0
            ok, frame = self.cap.read()
            if not ok:
                self.ended = True
                return self.frame
        self.index += 1
        self.frame = frame
        return frame

    def skip(self, n):
        """Drop n frames quickly (to keep up with real time)."""
        for _ in range(n):
            if self.cap is None or not self.cap.grab():
                break
            self.index += 1

    def seek(self, seconds):
        if self.cap is None or self.live or self.count <= 0:
            return
        target = int(max(0, min(self.count - 1, seconds * self.fps)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        self.index = target
        self.next_frame()


# ========================================================= frame -> grid ===
def frame_to_grid(frame, cols, rows, cw, ch, fill, contrast, brightness, sub=(1, 1)):
    """Resize a BGR frame into a cols x rows RGB grid (fit = letterbox, fill = crop).
    sub = sub-pixels per cell (detail mode samples each cell as a small grid)."""
    cols, rows, cw, ch = cols * sub[0], rows * sub[1], cw / sub[0], ch / sub[1]
    vh, vw = frame.shape[:2]
    out_w, out_h = cols * cw, rows * ch
    if fill:
        target = out_w / out_h
        if vw / vh > target:                       # too wide -> crop sides
            nw = int(vh * target)
            x0 = (vw - nw) // 2
            frame = frame[:, x0:x0 + nw]
        else:                                      # too tall -> crop top/bottom
            nh = int(vw / target)
            y0 = (vh - nh) // 2
            frame = frame[y0:y0 + nh]
        small = cv2.resize(frame, (cols, rows), interpolation=cv2.INTER_AREA)
    else:
        scale = min(out_w / vw, out_h / vh)
        dc = max(1, min(cols, int(round(vw * scale / cw))))
        dr = max(1, min(rows, int(round(vh * scale / ch))))
        small = np.zeros((rows, cols, 3), np.uint8)
        x0, y0 = (cols - dc) // 2, (rows - dr) // 2
        small[y0:y0 + dr, x0:x0 + dc] = cv2.resize(frame, (dc, dr), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
    if contrast != 1.0 or brightness != 0:
        rgb = np.clip((rgb - 128) * contrast + 128 + brightness, 0, 255)
    return rgb


# ================================================================ player ===
class Player:
    def __init__(self, args, source):
        self.args = args
        self.font_size = args.font_size
        self.conv = AsciiConverter(args.mode, not args.no_edges, decode=args.decode)
        self.conv.edge_thresh = args.edge_threshold
        self.conv.black = args.black
        self.mono = args.mono
        self.fill = args.fill
        self.paused = False
        self.zoom = ViewZoom()
        self.detail = args.detail
        self.toast, self.toast_t = "", 0.0
        self.hud_font = pygame.font.SysFont("segoeui,arial", 15)
        self.open(source)

    # ---------------------------------------------------------- layout ----
    def open(self, source):
        self.src = source
        self.clock_t = 0.0
        self.layout()
        pygame.display.set_caption(f"ASCII Player - {source.name}")

    def layout(self):
        self.font = pygame.font.SysFont("consolas,couriernew,monospace", self.font_size, bold=True)
        self.cw, self.ch = self.font.size("M")
        if self.args.size:
            w, h = (int(v) for v in self.args.size.lower().split("x"))
        else:
            vw, vh = self.src.size
            a = vw / vh
            h = MAX_H
            w = int(h * a)
            if w > MAX_W:
                w, h = MAX_W, int(MAX_W / a)
        if getattr(self, "win", None) is None:
            self.win = Window((w, h), "ASCII Player")
        ww, wh = self.win.size                      # the grid always fills the window
        self.cols, self.rows = max(10, ww // self.cw), max(5, wh // self.ch)
        self.screen = pygame.Surface((self.cols * self.cw, self.rows * self.ch))
        self.cache = {}
        self.conv.matcher = GlyphMatcher(self.font) if self.detail else None

    def say(self, text):
        self.toast, self.toast_t = text, time.perf_counter()

    # ---------------------------------------------------------- render ----
    def render(self, t):
        sub = self.conv.matcher.sub if self.conv.matcher else (1, 1)
        rgb = frame_to_grid(zoom_image(self.src.frame, self.zoom.factor), self.cols, self.rows, self.cw, self.ch,
                            self.fill, self.args.contrast, self.args.brightness, sub)
        glyph, color = self.conv.convert(rgb, t)
        scr = self.screen
        scr.fill((0, 0, 0))
        q = (color >> 4) << 4
        if self.mono:
            q[:] = 225
        cache, font, cw, ch = self.cache, self.font, self.cw, self.ch
        blits = []
        ys, xs = np.nonzero(glyph != " ")
        for y, x in zip(ys.tolist(), xs.tolist()):
            c = q[y, x]
            key = (glyph[y, x], int(c[0]), int(c[1]), int(c[2]))
            surf = cache.get(key)
            if surf is None:
                surf = font.render(key[0], True, key[1:])
                cache[key] = surf
            blits.append((surf, (x * cw, y * ch)))
        scr.blits(blits, doreturn=False)

    def draw_hud(self):
        if time.perf_counter() - self.toast_t < 1.6 and self.toast:
            img = self.hud_font.render(self.toast, True, (255, 255, 255))
            pad = img.get_rect().inflate(16, 8)
            pad.topleft = (10, 10)
            pygame.draw.rect(self.screen, (0, 0, 0), pad, border_radius=6)
            self.screen.blit(img, (18, 14))

    # ----------------------------------------------------------- input ----
    def present(self):
        """Draw the character grid in the window (crisp and centred)."""
        ww, wh = self.win.size
        aw, ah = self.screen.get_size()
        if aw <= ww and ah <= wh:
            self.win.surface.fill((0, 0, 0))
            self.win.surface.blit(self.screen, ((ww - aw) // 2, (wh - ah) // 2))
        else:
            self.win.present(self.screen)

    def handle(self, e):
        if self.win.handle(e):                      # resize / F11
            return
        if self.zoom.handle(e):                     # Z / X / 0 / mouse wheel
            self.say(f"Zoom {self.zoom.percent}%")
            return
        if e.type == pygame.DROPFILE:
            try:
                self.open(Source(e.file))
                self.say(f"Opened {os.path.basename(e.file)}")
            except SystemExit as err:
                self.say(str(err))
            return
        if e.type != pygame.KEYDOWN:
            return
        k = e.key
        if k == pygame.K_SPACE:
            self.paused = not self.paused
            self.say("Paused" if self.paused else "Playing")
        elif k == pygame.K_m:
            self.conv.mode = MODES[(MODES.index(self.conv.mode) + 1) % len(MODES)]
            self.say(f"Mode: {self.conv.mode}")
        elif k == pygame.K_e:
            self.conv.edges = not self.conv.edges
            self.say(f"Edges: {'on' if self.conv.edges else 'off'}")
        elif k == pygame.K_c:
            self.mono = not self.mono
            self.say("Mono" if self.mono else "Colour")
        elif k == pygame.K_d:
            self.detail = not self.detail
            self.conv.matcher = GlyphMatcher(self.font) if self.detail else None
            self.say(f"Detail: {'on' if self.detail else 'off'}")
        elif k == pygame.K_f:
            self.fill = not self.fill
            self.say("Fill (crop)" if self.fill else "Fit (letterbox)")
        elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.font_size = min(24, self.font_size + 1)
            self.layout()
            self.say(f"Char size {self.font_size}  ({self.cols}x{self.rows})")
        elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.font_size = max(5, self.font_size - 1)
            self.layout()
            self.say(f"Char size {self.font_size}  ({self.cols}x{self.rows})")
        elif k == pygame.K_RIGHT:
            self.src.seek(self.src.seconds + 5)
        elif k == pygame.K_LEFT:
            self.src.seek(self.src.seconds - 5)
        elif k == pygame.K_r:
            self.src.seek(0)
        elif k == pygame.K_s:
            name = time.strftime("ascii_%Y%m%d_%H%M%S.png")
            pygame.image.save(self.screen, name)
            self.say(f"Saved {name}")


def pick_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Choose a video or image",
            filetypes=[("Video / image", "*.mp4 *.mov *.avi *.mkv *.webm *.gif *.png *.jpg *.jpeg *.bmp *.webp"),
                       ("All files", "*.*")])
        root.destroy()
        return path or None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Play video / images / webcam as coloured ASCII art")
    ap.add_argument("source", nargs="?", help="video, image, or webcam number (0)")
    ap.add_argument("--mode", choices=MODES, default="ramp")
    ap.add_argument("--size", help="window size like 540x960 (default: match the video)")
    ap.add_argument("--font-size", type=int, default=10, help="character size (smaller = more detail)")
    ap.add_argument("--fill", action="store_true", help="crop to fill the window instead of letterboxing")
    ap.add_argument("--no-edges", action="store_true", help="no - \\ | / outline characters")
    ap.add_argument("--edge-threshold", type=float, default=0.35, help="lower = more outline characters")
    ap.add_argument("--black", type=float, default=0.07, help="0-1: pixels darker than this become blank (raise to hide backgrounds)")
    ap.add_argument("--mono", action="store_true", help="no colours")
    ap.add_argument("--decode", action="store_true", help="random-symbol flicker when cells appear")
    ap.add_argument("--detail", action="store_true", help="shape-matched characters for sharper detail")
    ap.add_argument("--contrast", type=float, default=1.15)
    ap.add_argument("--brightness", type=float, default=0.0)
    ap.add_argument("--record", metavar="OUT.mp4", help="save the ASCII video (plays once, every frame)")
    args = ap.parse_args()

    src_path = args.source or pick_file()
    if not src_path:
        ap.print_help()
        return

    pygame.init()
    source = Source(src_path)
    player = Player(args, source)
    writer = None
    if args.record:
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), source.fps,
                                 player.screen.get_size())
        print(f"Recording to {args.record} ...")

    clock = pygame.time.Clock()
    t0 = last = time.perf_counter()
    play_t = 0.0
    rec_size = player.screen.get_size()
    try:
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q)):
                    return
                player.handle(e)

            if player.win.take_resized() and not writer:   # re-flow to the new window size
                player.layout()

            now = time.perf_counter()
            dt, last = now - last, now
            src = player.src

            if writer:
                # recording: use every single frame, stop at the end of the video
                src.next_frame(loop=False)
                if src.ended:
                    break
            elif not player.paused and src.image is None:
                # real-time playback at the video's own fps (drop frames if slow)
                play_t += dt
                target = int(play_t * src.fps)
                if target > src.index:
                    src.skip(target - src.index - 1)
                    src.next_frame()
                if abs(target - src.index) > src.fps:      # looped or seeked -> resync
                    play_t = src.index / src.fps

            player.render(now - t0)

            if writer:
                arr = pygame.surfarray.array3d(player.screen).swapaxes(0, 1)
                if (arr.shape[1], arr.shape[0]) != rec_size:
                    arr = cv2.resize(arr, rec_size, interpolation=cv2.INTER_AREA)
                writer.write(cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_RGB2BGR))
                if src.count:
                    pygame.display.set_caption(f"Recording {src.index}/{src.count}")

            player.draw_hud()
            player.present()
            pygame.display.flip()
            if not writer:
                clock.tick(60)
    finally:
        if writer:
            writer.release()
            print(f"Saved {args.record}")
        pygame.quit()


if __name__ == "__main__":
    main()
