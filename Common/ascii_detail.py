"""
ASCII Detail
------------
One shared "more detail" method for every ASCII script in this folder.

1. Shape-matched characters  (GlyphMatcher)
   A normal ASCII converter looks at ONE brightness value per character cell,
   so a cell that is half bright / half dark becomes a grey middle character
   and the edge disappears.  Here every cell is sampled as a small grid of
   sub-pixels (4 x 6 by default) and compared against the real drawn shape
   of every character in the font.  The closest shape wins:

       bright left half  ->  [  |  (       bright bottom  ->  _  .  ,
       diagonal edge     ->  /  \\          bright top     ->  "  '  ^
       solid             ->  @  #  M       speck          ->  .  `  '

   Flat areas still use a brightness ramp (after a local-contrast boost),
   so only cells that really contain an edge switch to a shape character.
   Same number of characters, much sharper outlines and finer detail.

2. Sharpening  (sharpen)
   Unsharp mask before matching so small features survive the shrink.

3. Supersampling  (SuperSampler)
   For pygame 3D drawings: draw at 2x size, smooth-scale down
   -> anti-aliased, finer polygons (used by bouquet_3d.py --detail).

Used by:  ascii_bouquet.py --detail,  bouquet_3d.py --detail,
          Ascii_Picture/ascii_player.py --detail,  night_garden_convert.py --detail
"""
import cv2
import numpy as np
import pygame

PRINTABLE = "".join(chr(c) for c in range(32, 127))
RAMP = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
SUB = (4, 6)                      # sub-pixels per character cell (x, y)


def max_channel(rgb):
    """Brightest of R, G, B per pixel (much faster than rgb.max(axis=2))."""
    return np.maximum(np.maximum(rgb[..., 0], rgb[..., 1]), rgb[..., 2])


def sharpen(img, amount=0.6, sigma=1.0):
    """Unsharp mask: boost fine detail so it survives the shrink to characters."""
    if amount <= 0:
        return img
    img = img.astype(np.float32, copy=False)
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return np.clip(img + amount * (img - blur), 0, 255)


class GlyphMatcher:
    """Picks, for every cell, the character whose drawn shape best fits the image."""

    def __init__(self, font=None, charset=PRINTABLE, sub=SUB, font_name="consolas", size=14):
        if not pygame.font.get_init():
            pygame.font.init()
        if font is None:
            font = pygame.font.SysFont(font_name + ",couriernew,monospace", size, bold=True)
        self.sub = sub
        self.chars = np.array(list(charset))
        cw, ch = font.size("M")
        feats = []
        for c in charset:
            surf = pygame.Surface((cw, ch))
            surf.fill((0, 0, 0))
            surf.blit(font.render(c, True, (255, 255, 255)), (0, 0))
            a = pygame.surfarray.array3d(surf)[..., 0].T.astype(np.float32) / 255.0
            feats.append(cv2.resize(a, sub, interpolation=cv2.INTER_AREA).ravel())
        G = np.array(feats, np.float32)
        G /= max(G.max(), 1e-6)                      # densest glyph pixel = 1.0
        self.G = G
        self.G2 = (G * G).sum(1)
        self._clahe = None
        self._ramp_cache = {}
        self.last_mean = None

    def sample_size(self, cols, rows):
        """Pixel size to resize an image to before calling match()."""
        return cols * self.sub[0], rows * self.sub[1]

    def match(self, rgb_hi, cols, rows, sharp=0.6, ramp=None, structure=0.09, clahe=2.0):
        """
        rgb_hi: image of size (rows*sub_y, cols*sub_x, 3), values 0..255.
        Returns glyph (rows, cols), the colour of each cell's ink, and each cell's peak brightness.

        Hybrid: cells with an edge inside (contrast > `structure`) get the best-matching
        character SHAPE; flat cells get a density-ramp character from the local-contrast-
        boosted brightness, so soft shading becomes visible texture.
        """
        sx, sy = self.sub
        rgb8 = rgb_hi if rgb_hi.dtype == np.uint8 else np.clip(rgb_hi, 0, 255).astype(np.uint8)
        r, g, b = cv2.split(rgb8)
        lum8 = cv2.max(cv2.max(r, g), b)                      # brightest channel per pixel
        if sharp > 0:                                          # unsharp mask (brightness only)
            blur = cv2.GaussianBlur(lum8, (0, 0), 1.0)
            lum8 = cv2.addWeighted(lum8, 1 + sharp, blur, -sharp, 0)
        if clahe > 0:                                          # local contrast boost
            if self._clahe is None:
                self._clahe = cv2.createCLAHE(clipLimit=clahe, tileGridSize=(8, 8))
            lum_e = cv2.addWeighted(lum8, 0.5, self._clahe.apply(lum8), 0.5, 0)
            dark = lum8 < 10
            lum_e[dark] = lum8[dark]                           # don't boost the black background
        else:
            lum_e = lum8
        blocks = lum_e.reshape(rows, sy, cols, sx).transpose(0, 2, 1, 3).reshape(rows * cols, sx * sy)
        blocks = blocks.astype(np.float32) * (1 / 255.0)
        peak = blocks.max(1)
        spread = peak - blocks.min(1)
        mean = cv2.resize(lum_e, (cols, rows), interpolation=cv2.INTER_AREA).astype(np.float32) * (1 / 255.0)
        self.last_mean = mean

        # density ramp for flat cells
        ramp_chars = self._ramp_cache.get(ramp or RAMP)
        if ramp_chars is None:
            ramp_chars = self._ramp_cache[ramp or RAMP] = np.array(list(ramp or RAMP))
        n = len(ramp_chars)
        glyph = ramp_chars[np.clip((mean.ravel() * (n - 1) * 1.1 + 0.5).astype(np.int32), 0, n - 1)]

        # shape match (normalised by the cell's own peak), only for cells that contain an edge
        edge = spread > structure
        if edge.any():
            P = blocks[edge] / np.maximum(peak[edge], 0.2)[:, None]
            err = self.G2[None, :] - 2 * (P @ self.G.T)       # |P|^2 is the same for every glyph
            glyph[edge] = self.chars[err.argmin(1)]
        glyph = glyph.reshape(rows, cols)

        # colour of each cell = area average, scaled up to the cell's brightest pixel
        color = cv2.resize(rgb8, (cols, rows), interpolation=cv2.INTER_AREA).astype(np.float32)
        cmax = np.maximum(color.max(axis=2, keepdims=True), 1.0)
        color = color / cmax * (peak.reshape(rows, cols, 1) * 255.0)
        return glyph, color, peak.reshape(rows, cols)


class SuperSampler:
    """Draw at `factor` x size, then smooth-scale down -> anti-aliased, finer drawing."""

    def __init__(self, size, factor=2):
        self.size, self.factor = size, factor
        self.big = pygame.Surface((size[0] * factor, size[1] * factor))

    def begin(self, under):
        """Start from an enlarged copy of what is already on `under` (no dark seams)."""
        pygame.transform.scale(under, self.big.get_size(), self.big)
        return self.big

    def end(self, target, pos=(0, 0)):
        target.blit(pygame.transform.smoothscale(self.big, self.size), pos)
