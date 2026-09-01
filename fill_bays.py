#!/usr/bin/env python3
"""Fill every floor bay in small_warehouse_static.world with clutter and buckets.

    python3 fill_bays.py [--seed N] [--dry-run]

Targets small_warehouse_static.world ONLY -- small_warehouse.world keeps the
sparser 6-bay layout from place_props.py.  Idempotent: every live Bucket /
ClutteringA / ClutteringC / ClutteringD include is removed and regenerated, so
re-running never accumulates props.

STACKING
    All four prop models declare <static>, so a stack cannot settle or topple --
    placing one on another is purely a matter of z, and no physics step is
    involved.  For B on A:

        pose_z(B) = pose_z(A) + z_max(A) - z_min(B)

    z_min is NOT zero for these meshes (ClutteringD's is 0.333, which is exactly
    why its stock world pose carries a -0.3196 z offset).  Using the mesh extents
    rather than assuming a base at the origin is what keeps stacks flush.

    ClutteringA is the base of choice: it is the flattest and widest (2.16 x 2.00,
    top at 1.116) so a Bucket sits on it without overhang.

LAYOUT
    2 x 2 grid inside each bay's interior, insetting by the marking stroke width.
    One of the four cells is a ClutteringA carrying a Bucket, giving 14 two-layer
    piles.  Cell pitch is at least 2.78 m against a 2.16 m worst-case footprint,
    so ground items never touch.

Run AFTER scale_warehouse.py, add_bays.py and place_props.py.
"""
import re, os, sys, math, random

ROOT = os.path.dirname(os.path.abspath(__file__))
WORLD = os.path.join(ROOT, 'aws-robomaker-small-warehouse-world', 'worlds',
                     'small_warehouse_static', 'small_warehouse_static.world')
PRE = 'aws_robomaker_warehouse_'

ROWS = [(-17.643, -11.643), (-11.199, -5.199), (-4.755, 1.245),
        (1.689, 7.689), (8.133, 14.133)]
BAYS = ([(-7.689, 1.342, y0, y1) for (y0, y1) in ROWS] +
        [(-19.706, -13.706, y0, y1) for (y0, y1) in ROWS] +
        [(9.182, 18.212, y0, y1) for i, (y0, y1) in enumerate(ROWS) if i != 3])
T_X, T_Y = 0.2176, 0.1446

# model -> (footprint dx, dy, collision z_min, z_max, stock pose z)
PROP = {
    'Bucket_01':      (0.941, 1.222, 0.005, 1.409,  0.000000),
    'ClutteringA_01': (2.161, 2.002, 0.030, 1.116, -0.017477),
    'ClutteringC_01': (1.773, 2.060, 0.030, 1.790, -0.015663),
    'ClutteringD_01': (1.017, 1.522, 0.333, 1.825, -0.319559),
}
GROUND_CYCLE = ['ClutteringC_01', 'ClutteringA_01', 'ClutteringD_01', 'ClutteringC_01']
BASE = 'ClutteringA_01'      # cell that carries a stacked Bucket
TOP = 'Bucket_01'

BLOCK = ('    <include>\n'
         '      <uri>model://%s%s</uri>\n'
         '      <name>%s%s_%d</name>\n'
         '      <pose>%.6f %.6f %.6f 0 0 %.6f</pose>\n'
         '    </include>\n')


def blank_comments(text):
    """Same-length copy with comment bodies blanked to spaces.

    Scanning the raw text does not work: a commented block ends '</include> -->',
    so a match starting at its '<include>' cannot close there and instead runs on
    to the NEXT live block's '</include>', swallowing it. Blanking first makes
    every match a real one, and offsets still line up with the original.
    """
    out = list(text)
    for c in re.finditer(r'<!--.*?-->', text, re.S):
        for i in range(c.start(), c.end()):
            if out[i] != '\n':
                out[i] = ' '
    return ''.join(out)


def main():
    seed = int(sys.argv[sys.argv.index('--seed') + 1]) if '--seed' in sys.argv else 3
    dry = '--dry-run' in sys.argv
    rng = random.Random(seed)
    text = open(WORLD).read()
    clean = blank_comments(text)

    spans = []
    for m in re.finditer(r'[ \t]*<include>.*?</include>\n', clean, re.S):
        uri = re.search(r'<uri>model://([^<]+)</uri>', m.group(0))
        if uri and uri.group(1)[len(PRE):] in PROP:
            spans.append((m.start(), m.end()))
    if not spans:
        raise SystemExit('no live prop includes found -- wrong world file?')
    insert_at = spans[0][0]

    blocks, placed = [], []
    n = 100
    for bi, (x0, x1, y0, y1) in enumerate(BAYS):
        ix0, ix1 = x0 + T_X, x1 - T_X
        iy0, iy1 = y0 + T_Y, y1 - T_Y
        cx, cy = (ix0 + ix1) / 2, (iy0 + iy1) / 2
        dx, dy = (ix1 - ix0) / 4, (iy1 - iy0) / 4
        cells = [(cx - dx, cy - dy), (cx + dx, cy - dy),
                 (cx - dx, cy + dy), (cx + dx, cy + dy)]
        stack_cell = rng.randrange(4)
        for ci, (px, py) in enumerate(cells):
            model = BASE if ci == stack_cell else GROUND_CYCLE[(bi + ci) % 4]
            yaw = rng.choice([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
            z = PROP[model][4]
            blocks.append(BLOCK % (PRE, model, PRE, model, n, px, py, z, yaw))
            placed.append((model, n, px, py, z, yaw, bi, False)); n += 1
            if ci == stack_cell:
                tz = z + PROP[model][3] - PROP[TOP][2]
                tyaw = rng.choice([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
                blocks.append(BLOCK % (PRE, TOP, PRE, TOP, n, px, py, tz, tyaw))
                placed.append((TOP, n, px, py, tz, tyaw, bi, True)); n += 1

    for (a, b) in reversed(spans):
        text = text[:a] + text[b:]
        if b <= insert_at:
            insert_at -= (b - a)
    text = text[:insert_at] + ''.join(blocks) + text[insert_at:]

    ground = sum(1 for p in placed if not p[7])
    print('  removed %d existing prop includes' % len(spans))
    print('  added   %d props: %d ground + %d stacked, across %d bays'
          % (len(placed), ground, len(placed) - ground, len(BAYS)))
    print('  stack top z = %.4f m (ClutteringA top %.3f + bucket base %.3f)'
          % (PROP[BASE][4] + PROP[BASE][3] - PROP[TOP][2], PROP[BASE][3], PROP[TOP][2]))
    if dry:
        print('\n(dry run -- nothing written)')
        return
    open(WORLD, 'w').write(text)
    print('\nWrote %s' % os.path.basename(WORLD))


if __name__ == '__main__':
    main()
