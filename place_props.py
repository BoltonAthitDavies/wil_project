#!/usr/bin/env python3
"""Re-pose the warehouse props: clutter+buckets inside bays, shelves+bin by the walls.

    python3 place_props.py [--seed N] [--dry-run]

Rewrites only the <pose> of live <include> blocks in small_warehouse.world.
Commented-out blocks are skipped (ClutteringC_01_031 exists both commented and
live, so matching by <name> alone would hit the wrong one).

LAYOUT
    clutter + buckets  -> 2 each inside 6 bays chosen at random from the 14
    shelves + trashcan -> against the walls, clear of every bay

    ShelfF is 2.10 x 18.05 m and only fits along the EAST wall: walkways A, B and
    E chop every other near-wall corridor into runs shorter than 18 m.  At yaw 0
    it is already long in +y, so it needs no rotation, and it must sit below
    walkway D (y 3.364..5.338) to clear it.

Run AFTER scale_warehouse.py and add_bays.py -- the coordinates are current world
metres and assume the 14-bay floor.
"""
import re, os, sys, math, random

ROOT = os.path.dirname(os.path.abspath(__file__))
WORLD = os.path.join(ROOT, 'aws-robomaker-small-warehouse-world', 'worlds',
                     'small_warehouse', 'small_warehouse.world')

ROWS = [(-17.643, -11.643), (-11.199, -5.199), (-4.755, 1.245),
        (1.689, 7.689), (8.133, 14.133)]
BAYS = ([('stock', -7.689, 1.342, y0, y1) for (y0, y1) in ROWS] +
        [('west', -19.706, -13.706, y0, y1) for (y0, y1) in ROWS] +
        [('east', 9.182, 18.212, y0, y1) for i, (y0, y1) in enumerate(ROWS) if i != 3])
T_X, T_Y = 0.2176, 0.1446          # bay stroke thickness -> usable interior

# footprint (dx, dy) at yaw 0, measured from the collision meshes
FOOT = {'Bucket_01': (0.941, 1.222), 'ClutteringA_01': (2.161, 2.002),
        'ClutteringC_01': (1.773, 2.060), 'ClutteringD_01': (1.017, 1.522),
        'ShelfD_01': (3.916, 0.879), 'ShelfE_01': (3.916, 0.879),
        'ShelfF_01': (2.102, 18.047), 'TrashCanC_01': (1.479, 0.909)}

IN_BAYS = ['Bucket_01_020', 'Bucket_01_021', 'Bucket_01_022',
           'ClutteringA_01_016', 'ClutteringA_01_017',
           'ClutteringC_01_027', 'ClutteringC_01_028', 'ClutteringC_01_029',
           'ClutteringC_01_030', 'ClutteringC_01_031', 'ClutteringC_01_032',
           'ClutteringD_01_005']

# x, y for the wall group. yaw stays 0 for all of these (already correct).
BY_WALL = {
    'ShelfD_01_001': (-16.5, 20.00), 'ShelfD_01_002': (-5.0, 20.00),
    'ShelfD_01_003': (0.0, 20.00),
    'ShelfE_01_001': (-16.5, -19.50), 'ShelfE_01_002': (9.0, -19.50),
    'ShelfE_01_003': (14.0, -19.50),
    'ShelfF_01_001': (19.40, -11.2765),
    'TrashCanC_01_002': (10.0, 20.00),
}


def live_spans(text):
    """Character ranges that are NOT inside an XML comment."""
    out, prev = [], 0
    for m in re.finditer(r'<!--.*?-->', text, re.S):
        out.append((prev, m.start())); prev = m.end()
    out.append((prev, len(text)))
    return out


PREFIX = 'aws_robomaker_warehouse_'


def set_pose(text, name, x, y, yaw=None):
    name = PREFIX + name
    spans = live_spans(text)
    pat = re.compile(r'(<name>%s</name>\s*\n\s*<pose>)([^<]*)(</pose>)' % re.escape(name))
    for m in pat.finditer(text):
        if not any(a <= m.start() < b for (a, b) in spans):
            continue
        p = m.group(2).split()
        z = p[2] if len(p) > 2 else '0'
        r, pi = (p[3], p[4]) if len(p) > 4 else ('0', '0')
        yw = p[5] if len(p) > 5 and yaw is None else ('%.6f' % (yaw or 0.0))
        new = '%.6f %.6f %s %s %s %s' % (x, y, z, r, pi, yw)
        return text[:m.start()] + m.group(1) + new + m.group(3) + text[m.end():], float(yw)
    raise SystemExit('live <include> for %s not found' % name)


def main():
    seed = int(sys.argv[sys.argv.index('--seed') + 1]) if '--seed' in sys.argv else 7
    dry = '--dry-run' in sys.argv
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(BAYS)), 6))
    items = list(IN_BAYS); rng.shuffle(items)

    text = open(WORLD).read()
    placed = []

    for k, bi in enumerate(chosen):
        kind, x0, x1, y0, y1 = BAYS[bi]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        pair = items[k * 2:k * 2 + 2]
        # side by side along the bay's long axis
        span = (x1 - x0) - 2 * T_X
        off = min(span / 4, 2.0)
        for s, name in zip((-off, off), pair):
            model = re.sub(r'_\d+$', '', name)
            text, yw = set_pose(text, name, cx + s, cy)
            placed.append((name, model, cx + s, cy, yw, 'bay %d (%s)' % (bi + 1, kind)))

    for name, (x, y) in BY_WALL.items():
        model = re.sub(r'_\d+$', '', name)
        text, yw = set_pose(text, name, x, y, yaw=0.0)
        placed.append((name, model, x, y, yw, 'wall'))

    print('  bays chosen (seed %d): %s\n' % (seed, [b + 1 for b in chosen]))
    print('  %-36s %9s %9s %7s  %s' % ('include', 'x', 'y', 'yaw', 'where'))
    for (n, m, x, y, yw, w) in placed:
        print('  %-36s %9.3f %9.3f %7.3f  %s' % (n, x, y, yw, w))

    if dry:
        print('\n(dry run -- nothing written)')
        return
    open(WORLD, 'w').write(text)
    print('\nWrote %s' % os.path.basename(WORLD))


if __name__ == '__main__':
    main()
