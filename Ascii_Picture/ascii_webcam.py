"""
ASCII Webcam
------------
Shortcut for `python ascii_player.py 0` - opens your webcam as live
coloured ASCII art in a pop-out window.

Run:  python ascii_webcam.py            (default camera)
      python ascii_webcam.py 1          (a second camera)
      python ascii_webcam.py --mode love --size 540x960
Any ascii_player.py option works here too (see that file for keys).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ascii_player

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        args = ["0"] + args                # camera 0 unless one was given
    sys.argv = [sys.argv[0]] + args
    ascii_player.main()
