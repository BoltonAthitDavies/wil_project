#!/usr/bin/env python3
"""Render a world's floor as the 2-D viewer draws it, to PNG, for the report.

    python3 script/render_floor.py --world <a.world> --out <f.png> [--ppm 20]
    python3 script/render_floor.py --world <a.world> --out <f.png> \\
        --crop 0,0,10   # 10 m square centred on world (0,0)

WHY NOT JUST SHOW THE TEXTURE FILE
    A raw texture tile says nothing about what a camera actually sees. The bay-line
    atlas is a 512 px image of four colour bands; on the floor it is mapped onto
    ~150 thin painted strips, each 0.145 m wide, laid out around the bays. Printing
    the tile makes the floor look uniformly stripey, which is the opposite of the
    truth -- most of the floor is unpainted, and that is precisely why the
    "textured" condition supplies so few features.

    So this renders the real thing: viewer.py's own floor rasteriser, at a stated
    metres-per-pixel, with the world's own ground model resolved through
    ground_assets(). Whatever the viewer shows the operator, this writes to a file.

WHY IT IMPORTS viewer.py RATHER THAN COPYING ITS CODE
    The UV fit that places the concrete tile (u = y/TILE_U + U0, v = x/TILE_V + V0)
    was measured off the DAE and is documented in viewer.py with its refit residual.
    Copying those constants here would create a second place for them to be wrong
    after the next rescale. viewer.py imports Qt at module scope, so the import runs
    under QT_QPA_PLATFORM=offscreen; nothing drawn here touches Qt.
"""

import argparse
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import viewer


def render(world, ppm, flip, markings=True):
    assets = viewer.ground_assets(world)
    if assets is None:
        raise SystemExit("no GroundB model found in %s" % world)
    dae, concrete, atlas, n_marks = assets
    img = viewer.build_floor_array(concrete, ppm, flip)
    if markings and atlas and n_marks:
        viewer.paint_markings(img, ppm, flip, dae_path=dae, atlas_path=atlas)
    return img, (dae, concrete, atlas, n_marks)


def crop_world(img, ppm, cx, cy, size):
    """Cut a `size` metre square centred on world (cx, cy) out of the floor array."""
    h, w = img.shape[:2]
    half = size * 0.5 * ppm
    c = (cx - viewer.FLOOR_X0) * ppm
    r = (viewer.FLOOR_Y1 - cy) * ppm
    c0, c1 = int(round(c - half)), int(round(c + half))
    r0, r1 = int(round(r - half)), int(round(r + half))
    c0, r0 = max(0, c0), max(0, r0)
    c1, r1 = min(w, c1), min(h, r1)
    return img[r0:r1, c0:c1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--world', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ppm', type=float, default=20.0,
                    help='pixels per metre. 20 gives ~840 px across the 42 m floor.')
    ap.add_argument('--flip', default='', choices=['', 'u', 'v', 'uv'],
                    help="matches viewer.py's --floor-flip")
    ap.add_argument('--no-markings', action='store_true')
    ap.add_argument('--crop', default=None,
                    help='cx,cy,size in metres: a square detail cut of the floor')
    a = ap.parse_args()

    img, (dae, concrete, atlas, n_marks) = render(
        a.world, a.ppm, a.flip, markings=not a.no_markings)
    print("world    %s" % os.path.basename(a.world))
    print("ground   %s" % os.path.basename(os.path.dirname(os.path.dirname(dae))))
    print("concrete %s" % (concrete if isinstance(concrete, str) else
                           "flat diffuse %s" % (concrete,)))
    print("markings %s (%d faces)" % (os.path.basename(atlas) if atlas else "none",
                                      n_marks))
    if a.crop:
        cx, cy, size = (float(v) for v in a.crop.split(','))
        img = crop_world(img, a.ppm, cx, cy, size)
        print("crop     %.1f m square at (%.1f, %.1f)" % (size, cx, cy))
    Image.fromarray(img).save(a.out)
    print("wrote    %s  %dx%d px" % (a.out, img.shape[1], img.shape[0]))


if __name__ == '__main__':
    main()
