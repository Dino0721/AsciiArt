"""
Window helper shared by every pop-out script in this folder.

  * drag the window edges / maximize  -> the window resizes
  * F11                               -> fullscreen on / off
  * Z / X / mouse wheel / 0           -> zoom in / out / reset (Zoom, ViewZoom)

Text (ASCII) scripts re-flow to the new size (bigger window = more characters).
Fixed-format scripts draw on a canvas and call present(), which scales the
canvas to fit the window without stretching it (black bars where needed).
"""
import pygame


class Window:
    def __init__(self, size, title, min_size=(320, 240)):
        self.min_w, self.min_h = min_size
        self.resized = False
        # never open bigger than the screen (leave room for the taskbar + title bar)
        info = pygame.display.Info()
        if info.current_w > 0 and info.current_h > 0:
            fit = min(1.0, (info.current_w - 60) / size[0], (info.current_h - 110) / size[1])
            if fit < 1.0:
                size = (int(size[0] * fit), int(size[1] * fit))
                self.resized = True                    # text scripts re-flow on the first frame
        self.windowed_size = (max(size[0], self.min_w), max(size[1], self.min_h))
        self.fullscreen = False
        self.surface = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)
        pygame.display.set_caption(title)

    @property
    def size(self):
        return self.surface.get_size()

    def handle(self, e):
        """Call with every event. Returns True if it was a window event (resize / F11)."""
        if e.type == pygame.VIDEORESIZE and not self.fullscreen:
            w, h = max(e.w, self.min_w), max(e.h, self.min_h)
            self.surface = pygame.display.get_surface()
            if self.surface.get_size() != (w, h):
                self.surface = pygame.display.set_mode((w, h), pygame.RESIZABLE)
            self.windowed_size = (w, h)
            self.resized = True
            return True
        if e.type == pygame.KEYDOWN and e.key == pygame.K_F11:
            self.toggle_fullscreen()
            return True
        return False

    def toggle_fullscreen(self):
        if self.fullscreen:
            self.surface = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)
        else:
            self.surface = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.fullscreen = not self.fullscreen
        self.resized = True

    def take_resized(self):
        """True once after the window changed size (then resets)."""
        r, self.resized = self.resized, False
        return r

    def present(self, canvas):
        """Scale `canvas` to fit the window (keeping its shape) and draw it centred."""
        ww, wh = self.surface.get_size()
        cw, ch = canvas.get_size()
        if (ww, wh) == (cw, ch):
            self.surface.blit(canvas, (0, 0))
            return
        s = min(ww / cw, wh / ch)
        size = (max(1, int(cw * s)), max(1, int(ch * s)))
        self.surface.fill((0, 0, 0))
        self.surface.blit(pygame.transform.smoothscale(canvas, size),
                          ((ww - size[0]) // 2, (wh - size[1]) // 2))


class _ZoomKeys:
    """Shared keys:  Z / wheel up = zoom in,  X / wheel down = zoom out,  0 = reset."""

    def handle(self, e):
        """Call with every event. Returns True if it changed the zoom."""
        if e.type == pygame.MOUSEWHEEL and e.y:
            self.step(1 if e.y > 0 else -1)
            return True
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_z:
                self.step(1)
                return True
            if e.key == pygame.K_x:
                self.step(-1)
                return True
            if e.key in (pygame.K_0, pygame.K_KP0):
                self.reset()
                return True
        return False


class Zoom(_ZoomKeys):
    """Zoom a 3D camera by moving it closer / further (bouquet scripts)."""

    def __init__(self, cam, default_dist, lo=450.0, hi=2800.0):
        self.cam, self.default, self.lo, self.hi = cam, default_dist, lo, hi
        cam.dist = default_dist

    def step(self, direction):
        """direction +1 = zoom in, -1 = zoom out (10 % per step)."""
        d = self.cam.dist * (0.9 if direction > 0 else 1 / 0.9)
        self.cam.dist = min(self.hi, max(self.lo, d))

    def reset(self):
        self.cam.dist = self.default

    @property
    def percent(self):
        return int(round(100 * self.default / self.cam.dist))


class ViewZoom(_ZoomKeys):
    """A plain zoom factor (1.0 = normal) for 2D pictures, videos and text scenes."""

    def __init__(self, default=1.0, lo=0.35, hi=5.0):
        self.factor, self.default, self.lo, self.hi = default, default, lo, hi

    def step(self, direction):
        f = self.factor * (1 / 0.9 if direction > 0 else 0.9)
        self.factor = min(self.hi, max(self.lo, f))

    def reset(self):
        self.factor = self.default

    @property
    def percent(self):
        return int(round(100 * self.factor))


def zoom_image(img, factor):
    """Zoom a numpy image around its centre: >1 crops in (bigger), <1 shrinks with a black border."""
    if abs(factor - 1.0) < 1e-3:
        return img
    import cv2
    h, w = img.shape[:2]
    if factor > 1:
        cw, ch = max(1, int(w / factor)), max(1, int(h / factor))
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        return cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
    nw, nh = max(1, int(w * factor)), max(1, int(h * factor))
    out = img.copy()
    out[:] = 0
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out
