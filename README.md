# AsciiArt

Flowers and night gardens drawn with Python and text characters.

```
AsciiArt/
├── Bouquet/          3D flower bouquet (normal, ASCII, ASCII HD)
├── NightGarden/      "A night garden, made entirely of text"
├── Ascii_Picture/    turn any video / image / webcam into ASCII
└── Common/           shared helpers (no need to run these)
```

## What to run

| Folder | File | What it shows |
|---|---|---|
| Bouquet | `bouquet_3d.py` | Animated 3D bouquet: one flower grows into a wrapped bouquet |
| Bouquet | `ascii_bouquet.py` | The same bouquet made of coloured text characters |
| Bouquet | `ascii_bouquet_hd.py` | Same again, sharper and clearer (more characters, glow) |
| NightGarden | `night_garden.py` | Animation built directly from characters: hands, lotus, eye, waves, garden |
| NightGarden | `night_garden_convert.py` | Same scenes drawn as pictures, then converted to ASCII |
| Ascii_Picture | `ascii_player.py` | Play a video / image as ASCII (`python ascii_player.py flowers.mp4`) |
| Ascii_Picture | `ascii_webcam.py` | Your webcam as live ASCII |
| Common | `window_util.py` | Resizable window, fullscreen, zoom (used by everything) |
| Common | `ascii_detail.py` | Shape-matched characters + supersampling for extra detail |

Run a file from anywhere, for example:

```
python Bouquet/ascii_bouquet_hd.py
python NightGarden/night_garden.py
python Ascii_Picture/ascii_webcam.py
```

Needs: `pip install pygame numpy opencv-python`

## Keys (every window)

| Key | Does |
|---|---|
| **Z** / mouse wheel up | zoom in |
| **X** / mouse wheel down | zoom out |
| **0** | reset zoom |
| **F11** | fullscreen on / off |
| drag window edges | resize (text windows show more characters when bigger) |
| **SPACE** | pause |
| **S** | screenshot |
| **Q** / **Esc** | quit |

Each file lists its extra keys and options at the top (e.g. `--record out.mp4`
to save a video, `--detail` for sharper characters, `--mode love` for other characters).
