#!/usr/bin/env python3
"""Build small_warehouse_dynamic.world from small_warehouse_static.world.

    python3 script/make_dynamic.py [--movers N] [--seed N] [--dry-run]

Takes the static world's 70-prop bay layout verbatim, then re-arms a subset of
props with <static>false> + the Ignition velocity-control plugin so the world is
actually dynamic again. Idempotent: it always regenerates from the static world,
so re-running never doubles up plugins.

WHICH PROPS MAY MOVE
    Two exclusions, both about the 14 two-layer piles:

      * a STACK BASE must not move -- the Bucket resting on it is <static>, so it
        would not follow and would be left hanging in mid-air.
      * a STACKED item must not move -- dropping <static> puts it under gravity
        with nothing beneath once it slides off its base.

    So movers are drawn only from plain ground props, one per bay, spread across
    the floor. The wall-side TrashCan moves too, as it did in the stock world.

WHY THEY DO NOT FALL THROUGH THE FLOOR
    Ground props sit ~0.02 m below the floor surface (pose z -0.017, collision
    base 0.0125, floor top 0.0342) -- that interpenetration is stock. It is
    harmless because VelocityControl writes the FULL linear velocity every step
    with z = 0, which cancels gravity rather than fighting it.
"""
import re, os, sys, math, random, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WD = os.path.join(ROOT, 'aws-robomaker-small-warehouse-world', 'worlds')
SRC = os.path.join(WD, 'small_warehouse_static', 'small_warehouse_static.world')
DST = os.path.join(WD, 'small_warehouse_dynamic', 'small_warehouse_dynamic.world')
PRE = 'aws_robomaker_warehouse_'
SPEED = 1.011                                    # m/s, the stock value
DIRS = [(1, 0), (0, -1), (-1, 0), (0, 1)]

PLUGIN = ('      <static>false</static>\n'
          '      <plugin filename="ignition-gazebo-velocity-control-system"\n'
          '              name="ignition::gazebo::systems::VelocityControl">\n'
          '        <initial_linear>%.3f %.3f 0</initial_linear>\n'
          '      </plugin>\n')


def blank_comments(text):
    out = list(text)
    for c in re.finditer(r'<!--.*?-->', text, re.S):
        for i in range(c.start(), c.end()):
            if out[i] != '\n':
                out[i] = ' '
    return ''.join(out)


def includes(text):
    """[(start, end, name, model, pose)] for live <include> blocks."""
    clean = blank_comments(text)
    out = []
    for m in re.finditer(r'[ \t]*<include>.*?</include>\n', clean, re.S):
        b = m.group(0)
        nm = re.search(r'<name>(\S+?)</name>', b)
        uri = re.search(r'model://(\S+?)</uri>', b)
        po = re.search(r'<pose>([^<]*)</pose>', b)
        if not (nm and uri and po):
            continue
        out.append((m.start(), m.end(), nm.group(1), uri.group(1),
                    [float(v) for v in po.group(1).split()]))
    return out


def main():
    nmov = int(sys.argv[sys.argv.index('--movers') + 1]) if '--movers' in sys.argv else 11
    seed = int(sys.argv[sys.argv.index('--seed') + 1]) if '--seed' in sys.argv else 5
    dry = '--dry-run' in sys.argv
    rng = random.Random(seed)

    text = open(SRC).read()
    inc = includes(text)

    props = [i for i in inc if i[3][len(PRE):].split('_01')[0]
             in ('Bucket', 'ClutteringA', 'ClutteringC', 'ClutteringD')]
    # a stacked item sits well above the floor; its base is whatever shares its x,y
    stacked = {i[2] for i in props if i[4][2] > 0.1}
    bases = {i[2] for i in props if i[2] not in stacked and
             any(s[4][2] > 0.1 and abs(s[4][0] - i[4][0]) < 1e-6
                 and abs(s[4][1] - i[4][1]) < 1e-6 for s in props)}
    free = [i for i in props if i[2] not in stacked and i[2] not in bases]

    # one mover per bay-ish: bucket the candidates by y row then pick spread
    free.sort(key=lambda i: (round(i[4][1], 1), i[4][0]))
    step = max(1, len(free) // max(1, nmov - 1))
    chosen = free[::step][:nmov - 1]
    trash = [i for i in inc if i[2].endswith('TrashCanC_01_002')]
    chosen += trash

    edits = []
    for k, i in enumerate(chosen):
        dx, dy = DIRS[k % 4]
        edits.append((i[1], PLUGIN % (dx * SPEED, dy * SPEED), i[2], dx, dy))

    for (end, block, _, _, _) in sorted(edits, key=lambda e: -e[0]):
        close = text.rindex('    </include>\n', 0, end)
        text = text[:close] + block + text[close:]

    print('  %d props in the static layout: %d stacked, %d stack bases, %d free'
          % (len(props), len(stacked), len(bases), len(free)))
    print('  %d movers armed (never a base or a stacked item):' % len(edits))
    for (_, _, name, dx, dy) in sorted(edits, key=lambda e: e[2]):
        print('     %-38s v = (%+.3f, %+.3f)' % (name.replace(PRE, ''), dx * SPEED, dy * SPEED))
    if dry:
        print('\n(dry run -- nothing written)')
        return
    open(DST, 'w').write(text)
    print('\nWrote %s' % os.path.basename(DST))


if __name__ == '__main__':
    main()
