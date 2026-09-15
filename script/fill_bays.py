#!/usr/bin/env python3
"""Fill EVERY floor bay with clutter and buckets, at full grid capacity.

    python3 script/fill_bays.py [--world PATH] [--seed N] [--dry-run]

Idempotent: every live Bucket / ClutteringA / ClutteringC / ClutteringD include
is removed and regenerated, so re-running never accumulates props.  Only those
four models are touched -- shelves, desks, the pallet jack and the trash can are
left exactly where they are.

BAY GEOMETRY IS DERIVED, NOT HARD-CODED
    An earlier version carried its own BAYS table (6.00 m bays at y0 = -17.643).
    reshape_floor.py then halved the walkways and grew the bays, and the table
    silently went stale -- it still parsed, still placed 56 props, and put every
    one of them in the wrong place.  Bay rects now come from reshuffle_bays,
    which reads them out of the ground mesh's painted markings at import.

LAYOUT
    One prop per grid cell: big bays 3 x 2 = 6, small bays 2 x 2 = 4.  Worst-case
    footprint is 2.16 m against a cell pitch of at least 3.47 m, so ground items
    never touch, and every cell centre sits far enough inside the bay interior
    that a prop stays contained at any of the four yaws.  Both are asserted, not
    assumed.

ONE LAYER, ALWAYS
    Every prop sits on the floor. An earlier version put one Bucket per bay on top
    of a ClutteringA, giving 14 two-layer piles; that was removed on request.
    Buckets are still placed -- they are simply members of the ground cycle now.

    z comes from sit_z(), which puts each model's collision floor exactly on the
    floor surface. z_min is NOT zero for these meshes (ClutteringD's is 0.333,
    which is why its stock pose carries a -0.3196 z offset), and the stock poses
    bury every prop 20-29 mm.

Run AFTER scale_warehouse.py, add_bays.py and reshape_floor.py.
"""
import re, os, sys, math, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reshuffle_bays as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLD = os.path.join(ROOT, 'aws-robomaker-small-warehouse-world', 'worlds',
                     'small_warehouse_static', 'small_warehouse_static.world')
PRE = R.PRE

# Collision z_max, needed only for the stack base. Everything else -- footprint,
# z_min, capacities, bay rects -- is reshuffle_bays'.
Z_MAX = {'Bucket_01': 1.409, 'ClutteringA_01': 1.116,
         'ClutteringC_01': 1.790, 'ClutteringD_01': 1.825}

GROUND_CYCLE = ['ClutteringC_01', 'ClutteringA_01', 'Bucket_01', 'ClutteringD_01',
                'ClutteringC_01', 'Bucket_01', 'ClutteringD_01', 'ClutteringA_01']
YAWS = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
FIRST_ID = 400               # matches the existing _4xx naming in these worlds

BLOCK = ('    <include>\n'
         '      <uri>model://%s%s</uri>\n'
         '      <name>%s%s_%d</name>\n'
         '      <pose>%.6f %.6f %.6f 0 0 %.6f</pose>\n'
         '      <static>true</static>\n'
         '    </include>\n')


def arg(flag, default, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def obstacles(clean):
    """AABBs of the fixed furniture, so we can report any bay cell that hits one."""
    out = []
    for m in re.finditer(r'[ \t]*<include>.*?</include>\n', clean, re.S):
        b = m.group(0)
        u = re.search(r'<uri>model://([^<]+)</uri>', b)
        if not u:
            continue
        model = u.group(1)[len(PRE):]
        if model not in R.SHELF:
            continue
        p = [float(v) for v in re.search(r'<pose>([^<]*)</pose>', b).group(1).split()]
        dx, dy = R.SHELF[model]
        c, s = abs(math.cos(p[5])), abs(math.sin(p[5]))
        out.append((model, p[0], p[1], (dx * c + dy * s) / 2, (dx * s + dy * c) / 2))
    return out


def main():
    seed = arg('--seed', 3, int)
    world = arg('--world', WORLD)
    dry = '--dry-run' in sys.argv
    rng = random.Random(seed)

    text = open(world).read()
    clean = R.blank_comments(text)

    spans = [(m.start(), m.end())
             for m in re.finditer(r'[ \t]*<include>.*?</include>\n', clean, re.S)
             if (lambda u: u and u.group(1)[len(PRE):] in R.PROP)(
                 re.search(r'<uri>model://([^<]+)</uri>', m.group(0)))]
    if not spans:
        raise SystemExit('no live prop includes found -- wrong world file?')
    insert_at = spans[0][0]

    blocks, placed = [], []
    n = FIRST_ID
    for bi, (kind, x0, x1, y0, y1) in enumerate(R.BAYS):
        pts = R.cells(bi)
        for ci, (px, py) in enumerate(pts):
            model = GROUND_CYCLE[(bi + ci) % len(GROUND_CYCLE)]
            yaw = rng.choice(YAWS)
            z = R.sit_z(model)
            blocks.append(BLOCK % (PRE, model, PRE, model, n, px, py, z, yaw))
            placed.append(dict(model=model, n=n, x=px, y=py, z=z, yaw=yaw,
                               bay=bi, stacked=False)); n += 1

    # --- checks: containment, ground-level overlap, furniture ------------------
    bad = []
    ground = [p for p in placed if not p['stacked']]
    for p in ground:
        hw, hh = R.half(p['model'], p['yaw'])
        kind, x0, x1, y0, y1 = R.BAYS[p['bay']]
        if not (x0 + R.T_X <= p['x'] - hw and p['x'] + hw <= x1 - R.T_X and
                y0 + R.T_Y <= p['y'] - hh and p['y'] + hh <= y1 - R.T_Y):
            bad.append('%s_%d escapes bay %d' % (p['model'], p['n'], p['bay']))
    for i, a in enumerate(ground):
        ahw, ahh = R.half(a['model'], a['yaw'])
        for b in ground[i + 1:]:
            if a['bay'] != b['bay']:
                continue
            bhw, bhh = R.half(b['model'], b['yaw'])
            if abs(a['x'] - b['x']) < ahw + bhw and abs(a['y'] - b['y']) < ahh + bhh:
                bad.append('%s_%d overlaps %s_%d' % (a['model'], a['n'],
                                                     b['model'], b['n']))
    hits = []
    for name, ox, oy, ohw, ohh in obstacles(clean):
        for p in ground:
            hw, hh = R.half(p['model'], p['yaw'])
            if abs(p['x'] - ox) < hw + ohw and abs(p['y'] - oy) < hh + ohh:
                hits.append('%s_%d (bay %d) intersects %s' % (p['model'], p['n'],
                                                              p['bay'], name))
    if bad:
        raise SystemExit('placement invalid:\n  ' + '\n  '.join(bad))

    for (a, b) in reversed(spans):
        text = text[:a] + text[b:]
        if b <= insert_at:
            insert_at -= (b - a)
    text = text[:insert_at] + ''.join(blocks) + text[insert_at:]

    print('  world   %s' % os.path.basename(world))
    print('  removed %d existing prop includes' % len(spans))
    assert not any(p['stacked'] for p in placed), 'a stack survived'
    print('  added   %d props across %d bays (all occupied), every one on the floor'
          % (len(placed), len(R.BAYS)))
    for bi in range(len(R.BAYS)):
        g = [p for p in placed if p['bay'] == bi]
        print('    bay %2d  %-7s %-5s  %d props'
              % (bi, R.ZONE[bi], R.BAYS[bi][0], len(g)))
    print('  floor heights: ' + ', '.join('%s %.6f' % (m, R.sit_z(m))
                                          for m in sorted(set(GROUND_CYCLE))))
    print('  checks: containment OK, no same-bay overlap')
    if hits:
        print('  WARNING: %d prop/furniture intersections:' % len(hits))
        for h in hits:
            print('    ' + h)
    else:
        print('  checks: clear of shelves, trash can (%d obstacles tested)'
              % len(obstacles(clean)))
    if dry:
        print('\n(dry run -- nothing written)')
        return
    open(world, 'w').write(text)
    print('\nWrote %s' % world)


if __name__ == '__main__':
    main()
