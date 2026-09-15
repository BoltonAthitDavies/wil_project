#!/usr/bin/env python3
"""Halve every walkway lane and grow the bays to fill the freed floor.

    python3 script/reshape_floor.py [--clearance 0.30] [--dry-run]

WHY
    Only the west zone was tight (0.95 m between bay and walkway). stock and east
    wasted 2.2-2.6 m on each side. Halving the lanes about their own centres frees
    more, and the bays then expand to fill each zone with a fixed clearance.

HOW THE GREEN LINES MOVE
    Every green vertex sits on exactly one lane line, so each is shifted inward by
    lane_width/4 -- x for the N-S lanes (A, B), y for the E-W lanes (C, D, E).
    Translation leaves UVs untouched, which matters: the marking UVs are per-face
    islands and scaling them repaints the lines (that bug cost 72 of 82 faces once).
    The E-W lanes butt against the N-S lanes, so their endpoint x values are lane
    A/B coordinates and move with them -- the joins stay closed for free.

HOW THE BAYS ARE REBUILT
    All hazard triangles are dropped and 14 bays re-emitted from the stock 8-vertex
    /8-triangle template, reusing its normal and UV indices verbatim (the marking
    UV mapping is normalised per strip, so it fits any rectangle). Only POSITION
    grows. Row heights are per zone so each zone fills exactly.
"""
import re, os, sys, math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAE = os.path.join(ROOT, 'aws-robomaker-small-warehouse-world', 'models',
                   'aws_robomaker_warehouse_GroundB_01', 'meshes',
                   'aws_robomaker_warehouse_GroundB_01_visual.DAE')
NS = '{http://www.collada.org/2005/11/COLLADASchema}'
ATLAS = 'Material #946569'
Z = 12.45                      # cm, marking height (floor top)
T_X, T_Y = 0.2176, 0.1446      # bay outline stroke widths
ROW_GAP = 0.444                # gap between bay rows, as authored
WALL = (-20.656, 20.627, -20.642, 20.642)

# lane line centres as authored -> (which axis, inward shift)
# shift = lane width / 4, so centre-to-centre halves about the lane's own centre.
LANES = {
    'A': ('x', (-12.629, -10.045)),
    'B': ('x', (4.121, 6.631)),
    'C': ('y', (-19.7615, -18.077)),
    'D': ('y', (3.457, 5.250)),
    'E': ('y', (17.593, 19.2785)),
}
TEMPLATE = [  # (vertex 0..7, normal idx, uv idx) of one stock bay
    [(0, 36, 108), (1, 37, 109), (5, 38, 118)],
    [(0, 39, 108), (5, 40, 118), (4, 41, 119)],
    [(1, 42,  25), (3, 43,  28), (6, 44,  29)],
    [(1, 45,  25), (6, 46,  29), (5, 47,  26)],
    [(3, 48, 120), (2, 49, 122), (7, 50, 123)],
    [(3, 51, 120), (7, 52, 123), (6, 53, 121)],
    [(2, 54,  30), (0, 55,  24), (4, 56,  27)],
    [(2, 57,  30), (4, 58,  27), (7, 59,  31)],
]
BANDS = [(0.0, 0.342, 'hazard'), (0.342, 0.590, 'green'),
         (0.590, 0.820, 'blue'), (0.820, 1.0, 'yellow')]


def band(u):
    u %= 1.0
    for lo, hi, n in BANDS:
        if lo <= u < hi:
            return n
    return 'yellow'


def shift_for(val, axis):
    """Inward shift for a green vertex coordinate, or 0 if it is not on a lane."""
    for name, (ax, (a, b)) in LANES.items():
        if ax != axis:
            continue
        d = (b - a) / 4.0
        if abs(val - a) < 0.20:
            return +d
        if abs(val - b) < 0.20:
            return -d
    return 0.0


def lane_edges():
    """Outer edges of each lane after halving, in world metres."""
    out = {}
    for name, (ax, (a, b)) in LANES.items():
        d = (b - a) / 4.0
        out[name] = (a + d, b - d)          # new line CENTRES
    return out


def main():
    clear = float(sys.argv[sys.argv.index('--clearance') + 1]) if '--clearance' in sys.argv else 0.30
    dry = '--dry-run' in sys.argv
    import xml.etree.ElementTree as ET

    text = open(DAE).read()
    root = ET.fromstring(text)
    arr = {fa.get('id'): [float(x) for x in fa.text.split()]
           for fa in root.iter(NS + 'float_array')}
    pos = next(v for k, v in arr.items() if k.endswith('POSITION-array'))
    uvs = next(v for k, v in arr.items() if k.endswith('UV0-array'))

    tri_elem = next(t for t in root.iter(NS + 'triangles') if t.get('material') == ATLAS)
    inp = {i.get('semantic'): int(i.get('offset')) for i in tri_elem.findall(NS + 'input')}
    P = [int(x) for x in tri_elem.find(NS + 'p').text.split()]
    st = max(inp.values()) + 1

    green_tris, green_verts = [], set()
    for i in range(0, len(P), st * 3):
        tri, us = [], []
        for k in range(3):
            v = P[i + k * st + inp['VERTEX']]
            n = P[i + k * st + inp['NORMAL']]
            t = P[i + k * st + inp['TEXCOORD']]
            tri.append((v, n, t)); us.append(uvs[t * 2])
        if band(sum(us) / 3) == 'green':
            green_tris.append(tri)
            green_verts.update(v for v, _, _ in tri)

    # --- 1. halve the lanes -----------------------------------------------------
    moved = 0
    for v in green_verts:
        x, y = pos[v * 3] / 100.0, pos[v * 3 + 1] / 100.0
        dx, dy = shift_for(x, 'x'), shift_for(y, 'y')
        if dx or dy:
            moved += 1
        pos[v * 3] = (x + dx) * 100.0
        pos[v * 3 + 1] = (y + dy) * 100.0

    C = lane_edges()
    sx = 0.274 / 2       # N-S stroke half-width
    sy = 0.177 / 2       # E-W stroke half-width
    zones = {
        'west':   (WALL[0] + clear, C['A'][0] - sx - clear),
        'stock':  (C['A'][1] + sx + clear, C['B'][0] - sx - clear),
        'east':   (C['B'][1] + sx + clear, WALL[1] - clear),
    }
    yspan = {
        'west':   (WALL[2] + clear, C['E'][0] - sy - clear),
        'stock':  (C['C'][1] + sy + clear, WALL[3] - clear),
        'east_s': (WALL[2] + clear, C['D'][0] - sy - clear),
        'east_n': (C['D'][1] + sy + clear, WALL[3] - clear),
    }

    def rows(y0, y1, n):
        h = (y1 - y0 - (n - 1) * ROW_GAP) / n
        return [(y0 + i * (h + ROW_GAP), y0 + i * (h + ROW_GAP) + h) for i in range(n)], h

    bays = []
    wr, wh = rows(*yspan['west'], 5)
    sr, sh = rows(*yspan['stock'], 5)
    er, eh = rows(*yspan['east_s'], 3)
    # east_n holds ONE bay in a 15 m zone. Filling it would make bay 14 210 m2
    # against 57-107 m2 elsewhere, so cap it at the east_s row height and centre
    # it; the leftover is unavoidable without adding a 15th bay.
    nh = eh
    n0, n1 = yspan['east_n']
    nc = (n0 + n1) / 2
    nr = [(nc - nh / 2, nc + nh / 2)]
    for a, b in sr:
        bays.append(('stock', zones['stock'][0], zones['stock'][1], a, b))
    for a, b in wr:
        bays.append(('west', zones['west'][0], zones['west'][1], a, b))
    for a, b in er:
        bays.append(('east_s', zones['east'][0], zones['east'][1], a, b))
    for a, b in nr:
        bays.append(('east_n', zones['east'][0], zones['east'][1], a, b))

    # --- 2. rebuild the bay outlines --------------------------------------------
    nvert = len(pos) // 3
    newp = []
    for tri in green_tris:
        for v, n, t in tri:
            newp += [v, n, t]
    for kind, x0, x1, y0, y1 in bays:
        base = len(pos) // 3
        ix0, ix1, iy0, iy1 = x0 + T_X, x1 - T_X, y0 + T_Y, y1 - T_Y
        for (px, py) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1),
                         (ix0, iy0), (ix1, iy0), (ix1, iy1), (ix0, iy1)]:
            pos += [px * 100.0, py * 100.0, Z]
        for t in TEMPLATE:
            for (v, n, uv) in t:
                newp += [base + v, n, uv]

    ntri = len(newp) // (st * 3)
    text = re.sub(r'(<float_array id="[^"]*_visual-POSITION-array" count=")\d+(">\n)(.*?)(</float_array>)',
                  lambda m: m.group(1) + str(len(pos)) + m.group(2)
                            + '\n'.join(' '.join('%f' % pos[i + j] for j in range(3))
                                        for i in range(0, len(pos), 3)) + '\n' + m.group(4),
                  text, count=1, flags=re.S)
    text = re.sub(r'(<accessor source="#[^"]*_visual-POSITION-array" count=")\d+(")',
                  r'\g<1>%d\g<2>' % (len(pos) // 3), text)
    text = re.sub(r'(<triangles count=")\d+(" material="%s">.*?<p>)(.*?)(</p>)' % re.escape(ATLAS),
                  lambda m: m.group(1) + str(ntri) + m.group(2) + ' '.join(str(v) for v in newp) + m.group(4),
                  text, count=1, flags=re.S)

    print('  lanes halved (green vertices moved): %d' % moved)
    for n, (ax, (a, b)) in sorted(LANES.items()):
        print('     lane %s  %s  width %.3f -> %.3f m' % (n, ax, abs(b - a), abs(b - a) / 2))
    print('\n  %-8s %-22s %-8s %-8s %s' % ('zone', 'x span', 'width', 'row h', 'bay area'))
    for z, n_, h in (('west', 5, wh), ('stock', 5, sh), ('east_s', 3, eh), ('east_n', 1, nh)):
        k = 'west' if z == 'west' else ('stock' if z == 'stock' else 'east')
        w = zones[k][1] - zones[k][0]
        print('  %-8s %8.3f..%8.3f  %-8.3f %-8.3f %.1f m2  (%d rows)'
              % (z, zones[k][0], zones[k][1], w, h, w * h, n_))
    print('\n  vertices %d -> %d   atlas triangles -> %d' % (nvert, len(pos) // 3, ntri))
    if dry:
        print('\n(dry run -- nothing written)')
        return
    open(DAE, 'w').write(text)
    print('\nWrote %s' % os.path.basename(DAE))


if __name__ == '__main__':
    main()
