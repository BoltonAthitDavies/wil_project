#!/usr/bin/env python3
"""Add hazard-striped parcel bay outlines to the warehouse floor.

RUN THIS *AFTER* scale_warehouse.py.  Coordinates below are CURRENT world metres,
not stock ones.  Re-running scale_warehouse.py reverts the meshes to pristine and
wipes these bays, so the order is always:

    git -C aws-robomaker-small-warehouse-world checkout -- models/
    python3 script/scale_warehouse.py 3.0105741651 2.0
    python3 script/add_bays.py

HOW A BAY IS BUILT
    Each bay is a mitred picture frame: 8 vertices (4 outer corners + 4 inset
    corners) and 8 triangles (2 per side).  The stock bays' UV mapping is
    normalised PER STRIP -- every strip spans ~0.043 in u and ~1.778 in v
    regardless of its physical size -- so the stripe pattern is fitted to each
    strip rather than tiled in world space.  That means new bays can REUSE the
    stock normal and UV indices verbatim and only contribute new positions.
    Nothing is appended to the Normal0 or UV0 arrays.

    The u values land in 0.000-0.342, the hazard-stripe band of
    GroundB_02.png.  Do not scale those UVs -- see scale_warehouse.py's UV NOTE.

BAYS ARE PAINT, NOT GEOMETRY
    This touches only the VISUAL mesh.  The collision mesh is untouched, so the
    baked nav2 map does not change and does not need regenerating.
"""
import re, os, sys

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'aws-robomaker-small-warehouse-world', 'models')
DAE = os.path.join(PKG, 'aws_robomaker_warehouse_GroundB_01', 'meshes',
                   'aws_robomaker_warehouse_GroundB_01_visual.DAE')

Z   = 12.45      # cm, floor top face + the same lift the stock bays use
T_X = 0.2176     # m, vertical-strip thickness (x)
T_Y = 0.1446     # m, horizontal-strip thickness (y)
STOCK_VERTS = 100                     # bail out if the mesh already has bays

# (v, n, uv) triples of one stock bay, vertices expressed 0..7 relative to its base
TEMPLATE = [
    [(0, 36, 108), (1, 37, 109), (5, 38, 118)],
    [(0, 39, 108), (5, 40, 118), (4, 41, 119)],
    [(1, 42,  25), (3, 43,  28), (6, 44,  29)],
    [(1, 45,  25), (6, 46,  29), (5, 47,  26)],
    [(3, 48, 120), (2, 49, 122), (7, 50, 123)],
    [(3, 51, 120), (7, 52, 123), (6, 53, 121)],
    [(2, 54,  30), (0, 55,  24), (4, 56,  27)],
    [(2, 57,  30), (4, 58,  27), (7, 59,  31)],
]

# Row y-spans reused from the stock bay column.
ROWS = [(-17.643, -11.643), (-11.199, -5.199), (-4.755, 1.245),
        (1.689, 7.689), (8.133, 14.133)]
WEST_X = (-19.706, -13.706)      # 6.000 m, between the west wall and walkway A
EAST_X = (9.182, 18.212)         # 9.030 m, between walkway B and the east wall
# walkway D crosses the east band at y 3.457..5.250, so east skips row index 3.
BAYS = ([(WEST_X[0], WEST_X[1], y0, y1) for (y0, y1) in ROWS] +
        [(EAST_X[0], EAST_X[1], y0, y1) for i, (y0, y1) in enumerate(ROWS) if i != 3])


def bay_verts(x0, x1, y0, y1):
    """8 corners in the stock order: outer SW,SE,NW,NE then inner SW,SE,NE,NW."""
    ix0, ix1 = x0 + T_X, x1 - T_X
    iy0, iy1 = y0 + T_Y, y1 - T_Y
    return [(x0, y0), (x1, y0), (x0, y1), (x1, y1),
            (ix0, iy0), (ix1, iy0), (ix1, iy1), (ix0, iy1)]


def main():
    force = '--force' in sys.argv
    dry = '--dry-run' in sys.argv
    text = open(DAE).read()

    pat = re.compile(r'(<float_array id="[^"]*_visual-POSITION-array" count=")(\d+)(">\n)(.*?)(</float_array>)', re.S)
    m = pat.search(text)
    if not m:
        raise SystemExit('POSITION array not found')
    vals = m.group(4).split()
    nvert = len(vals) // 3
    if nvert != STOCK_VERTS and not force:
        raise SystemExit('mesh has %d verts, expected %d -- bays already added? '
                         'Re-run scale_warehouse.py from pristine, or pass --force.'
                         % (nvert, STOCK_VERTS))

    new_lines, tris = [], []
    for (x0, x1, y0, y1) in BAYS:
        base = nvert + len(new_lines)
        for (x, y) in bay_verts(x0, x1, y0, y1):
            new_lines.append('%f %f %f' % (x * 100.0, y * 100.0, Z))
        for t in TEMPLATE:
            tris.append(' '.join('%d %d %d' % (base + v, n, uv) for (v, n, uv) in t))

    body = m.group(4).rstrip('\n') + '\n' + '\n'.join(new_lines) + '\n'
    total = (nvert + len(new_lines)) * 3
    text = text[:m.start()] + m.group(1) + str(total) + m.group(3) + body + m.group(5) + text[m.end():]
    text = re.sub(r'(<accessor source="#[^"]*_visual-POSITION-array" count=")\d+(")',
                  r'\g<1>%d\g<2>' % (nvert + len(new_lines)), text)

    tp = re.compile(r'(<triangles count=")(\d+)(" material="Material #946569">.*?<p>)(.*?)(</p>)', re.S)
    tm = tp.search(text)
    if not tm:
        raise SystemExit('atlas <triangles> block not found')
    ntri = int(tm.group(2))
    newp = tm.group(4).rstrip() + ' ' + ' '.join(tris)
    text = (text[:tm.start()] + tm.group(1) + str(ntri + len(tris)) + tm.group(3)
            + newp + tm.group(5) + text[tm.end():])

    print('  %d bays  ->  +%d vertices (%d -> %d), +%d triangles (%d -> %d)'
          % (len(BAYS), len(new_lines), nvert, nvert + len(new_lines),
             len(tris), ntri, ntri + len(tris)))
    for (x0, x1, y0, y1) in BAYS:
        print('     x %8.3f..%8.3f   y %8.3f..%8.3f   %5.3f x %5.3f m'
              % (x0, x1, y0, y1, x1 - x0, y1 - y0))
    if dry:
        print('\n(dry run -- nothing written)')
        return
    open(DAE, 'w').write(text)
    print('\nWrote %s' % os.path.basename(DAE))


if __name__ == '__main__':
    main()
