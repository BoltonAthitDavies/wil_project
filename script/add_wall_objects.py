#!/usr/bin/env python3
"""Stand furniture against the warehouse walls, outside every bay.

    python3 script/add_wall_objects.py --world <a.world> [--world <b.world>] [--dry-run]

Everything it writes goes inside ONE delimited region, split into a group per
wall, so a group can be commented out by hand without hunting through the file.
Re-running removes that whole region first, so it never accumulates.

HOW LITTLE ROOM THERE IS
    reshape_floor.py grew the bays to within 0.30 m of the walls, so most of the
    perimeter has nothing behind it. Measured pockets, and what fits:

      north  x  6.44..20.33, y 16.60..20.64  4.05 m deep  -> 2 shelves + trash can
      south  x -10.25.. 4.31, y -20.64..-18.11 2.53 m     -> 4 shelves
      east   x  6.14..20.63, y  4.89.. 8.94   4.05 m tall -> 2 desks
             plus the NE corner above bay 13              -> 1 trash can
      west   x -20.66..-12.11, y 18.94..20.64 1.70 m tall -> 2 pallet jacks

    The west wall is the cramped one: bays 0-4 run its whole length, and the four
    inter-bay gaps are 0.444 m -- narrower than the smallest model's 0.540 m
    short side, so nothing fits in them. Only the north-west corner pocket is
    usable, and at 1.70 m of run it takes two pallet jacks and nothing larger.

    The east wall cannot take a shelf. The y 4.89..8.94 band between bays 12 and
    13 is 4.049 m and a shelf is 3.918 m long, which leaves 0.065 m a side -- less
    than the bay margin. Desks fit with room to spare, so desks it is.

VERIFIED, NOT ASSUMED
    Every placement is checked against geometry read from the ground mesh at
    runtime: inside the wall, clear of all 14 bay rects, clear of the painted
    walkway strips, and clear of everything else. A candidate that fails is
    reported and skipped rather than written.
"""
import re, os, sys, math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reshuffle_bays as R
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRE = R.PRE

# model -> (dx, dy, collision z_min)
FURNITURE = {'ShelfD_01': (3.918, 0.880, 0.0293),
             'ShelfE_01': (3.918, 0.880, 0.0293),
             'DeskC_01': (0.882, 1.555, 0.0000),
             'PalletJackB_01': (1.161, 0.540, 0.0119),
             'TrashCanC_01': (1.479, 0.909, 0.0330)}

BAY_MARGIN = 0.20               # never encroach on a bay
LANE_MARGIN = 0.15              # do not stand on painted walkway
ENT_MARGIN = 0.40               # elbow room around the robot and the props
RUN_GAP = 0.20                  # between two units standing side by side
WALL_GAP = 0.10                 # back of the unit to the wall

# Units in one run are NOT held to ENT_MARGIN. That margin is about leaving room
# to drive past things; units in a row against a wall should read as one run, so
# they only have to not touch.

WX0, WX1, WY0, WY1 = R.WALL
ROW_GAP = 0.30                  # between rows stacked back from the wall
INSET = 0.02                    # keeps a packed row off the exact margin boundary
Q = math.pi / 2


def extents(model, yaw):
    """(x extent, y extent) of the model at this yaw."""
    dx, dy = FURNITURE[model][:2]
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    return (dx * c + dy * s, dx * s + dy * c)


# Model cycles. Yaw is chosen so the unit is SHALLOW against its wall and long
# along it. Row depth is the deepest member (0.909, the bin), and every unit's
# back aligns to the row's near edge, so a mixed row still lines up.
NS = [('ShelfD_01', 0.0), ('DeskC_01', Q), ('TrashCanC_01', 0.0),
      ('PalletJackB_01', 0.0)]
NS_E = [('ShelfE_01', 0.0), ('DeskC_01', Q), ('TrashCanC_01', 0.0),
        ('PalletJackB_01', 0.0)]
NS_FLAT = [('PalletJackB_01', 0.0)]          # 0.540 deep, the only thing that fits
EW = [('ShelfE_01', Q), ('DeskC_01', 0.0), ('TrashCanC_01', Q),
      ('PalletJackB_01', Q)]
EW_FLAT = [('PalletJackB_01', 0.0)]

# (group, wall, x0, x1, y0, y1, cycle, max rows). Bounds already carry BAY_MARGIN
# off bays, LANE_MARGIN off painted strips and WALL_GAP off the wall, so the
# packer only has to fit inside them -- the verifier re-checks every result anyway.
#
# Row count is capped rather than run to exhaustion. The north-east and east
# pockets are ~53 m each and would take 12 rows, which stops being "objects along
# the wall" and becomes a solid block of furniture. Three rows lines the wall.
REGIONS = [
    # One row, not three. Packed three deep this pocket reads as a block of
    # furniture in the middle of the floor rather than a lined wall.
    ('north', 'N', 6.291, 20.527, 16.795, 20.542, NS, 1, False),
    ('north', 'N', -19.195, -12.259, 19.089, 20.542, NS, 1, False),
    ('south', 'S', -12.220, 4.462, -20.542, -19.579, NS_E, 1, False),
    # No second south band. The gap behind the shelving looks like 1.14 m of free
    # floor, but a SECOND green line runs at y -18.584..-18.412: what is between
    # the two lines IS the walkway, 0.668 m wide, and 0.368 m of that is margin.
    # Nothing fits, and filling it would have blocked the lane.
    #
    # The east pocket is 14.24 x 3.70 m. Packed against the east wall its rows are
    # only 3.70 m long and hold 2 units each; run along its LONG axis instead and a
    # row holds six. Same pocket, three times the furniture.
    # The gap between bays 12 and 13. Filled as a pocket it was 18 units of
    # free-standing furniture anchored to a BAY edge, not to a wall. Anchored to
    # the east wall instead it becomes one row that joins the band segments above
    # and below it, so the east wall reads as continuously lined.
    ('east', 'E', 19.148, 20.527, 5.039, 8.738, EW, 1, False),
    # x1 is the jack's own width plus the inset, not the width alone: at -19.395
    # the 2 cm inset left 1.121 m for a 1.161 m unit and the region packed EMPTY.
    ('west', 'W', -20.556, -19.340, 19.089, 20.542, EW_FLAT, 1, False),
    # The long east/west wall bands. Bays 0-4 and 10-13 stop 0.30 m short of the
    # walls, but their prop grid stops 1.0 m (west) / 1.4 m (east) short of the bay
    # OUTLINE -- so a row can stand on the outline without ever touching a prop.
    # Where a prop's swept path does reach in (the east band, in every dynamic
    # world), the verifier culls the row for that world and keeps it for the empty
    # ones. Same regions, different result per world, decided by measurement.
    ('west', 'W', -20.556, -19.544, -20.542, 17.786, EW, 1, True),
    ('east', 'E', 19.148, 20.527, -20.542, 3.663, EW, 1, True),
    ('east', 'E', 19.148, 20.527, 9.138, 16.395, EW, 1, True),
]
NOTE = {'north': 'three rows across the top pocket, one along the north-west arm',
        'south': 'shelving on the wall, jacks parked behind the walkway line',
        'east': 'one row down the east wall, in the segments the bays leave free',
        'west': 'corner jacks plus a run down the long west wall'}


def pack(wall, x0, x1, y0, y1, cycle, max_rows):
    """Rows parallel to the wall, packed from the wall inward."""
    # Pull the bounds in 2 cm first. They are written as exactly
    # strip_edge + LANE_MARGIN, so without this the whole first row lands
    # precisely ON the threshold and the verifier rejects it -- 15 units were lost
    # that way, every one of them reported as 'on a walkway strip'.
    x0, x1, y0, y1 = x0 + INSET, x1 - INSET, y0 + INSET, y1 - INSET
    out = []
    flat = wall in 'NS'
    if flat:
        a0, a1 = x0, x1
        near, far = (y1, y0) if wall == 'N' else (y0, y1)
    else:
        a0, a1 = y0, y1
        near, far = (x1, x0) if wall == 'E' else (x0, x1)
    sign = 1.0 if far > near else -1.0
    depth_of = lambda m, yw: extents(m, yw)[1 if flat else 0]
    row_depth = max(depth_of(m, yw) for m, yw in cycle)

    off, rows = 0.0, 0
    while rows < max_rows and abs(far - near) - off >= row_depth:
        a, i, misses, any_placed = a0, 0, 0, False
        while misses < len(cycle):
            m, yw = cycle[i % len(cycle)]
            ex, ey = extents(m, yw)
            length = ex if flat else ey
            if a + length <= a1:
                ac = a + length / 2.0
                dc = near + sign * (off + depth_of(m, yw) / 2.0)
                out.append((m, ac, dc, yw) if flat else (m, dc, ac, yw))
                a += length + RUN_GAP
                misses, any_placed = 0, True
            else:
                misses += 1
            i += 1
        if not any_placed:
            break
        off += row_depth + ROW_GAP
        rows += 1
    return out


GROUPS = []
for g, wall, x0, x1, y0, y1, cyc, nrow, on_bay in REGIONS:
    items = [(m, x, y, yw, None, on_bay)
             for m, x, y, yw in pack(wall, x0, x1, y0, y1, cyc, nrow)]
    done = [grp for grp in GROUPS if grp[0] == g]
    if done:
        done[0][2].extend(items)
    else:
        GROUPS.append((g, NOTE[g], items))

BEGIN = ('    <!-- ===== BEGIN near-wall objects : generated by'
         ' script/add_wall_objects.py ===== -->\n'
         '    <!-- Each group below is plain includes with nothing nested inside,'
         ' so wrapping one -->\n'
         '    <!-- group in an XML comment to hide it is legal. Re-running the'
         ' script restores it. -->\n')
END = ('    <!-- ===== END near-wall objects ===== -->\n')
# No '--' anywhere inside a comment: XML forbids it outright, and ElementTree
# rejects the file while Gazebo's TinyXML2 shrugs and loads it -- exactly the
# split that let a malformed world survive unnoticed once already.
GHEAD = '    <!-- ==== %s wall: %d object(s) : %s ==== -->\n'
GTAIL = '    <!-- ==== end %s wall ==== -->\n'

BLOCK = ('    <include>\n'
         '      <uri>model://%s%s</uri>\n'
         '      <name>%s%s</name>\n'
         '      <pose>%.6f %.6f %.6f 0 0 %.6f</pose>\n'
         '    </include>\n')


def green_quads():
    """Painted walkway rectangles, straight from the ground mesh."""
    ns = '{http://www.collada.org/2005/11/COLLADASchema}'
    dae = os.path.join(ROOT, 'aws-robomaker-small-warehouse-world', 'models',
                       'aws_robomaker_warehouse_GroundB_01', 'meshes',
                       'aws_robomaker_warehouse_GroundB_01_visual.DAE')
    root = ET.parse(dae).getroot()
    arr = {fa.get('id'): [float(x) for x in fa.text.split()]
           for fa in root.iter(ns + 'float_array')}
    pos = next(v for k, v in arr.items() if k.endswith('POSITION-array'))
    uvs = next(v for k, v in arr.items() if k.endswith('UV0-array'))
    tri = next(t for t in root.iter(ns + 'triangles')
               if t.get('material') == 'Material #946569')
    inp = {i.get('semantic'): int(i.get('offset')) for i in tri.findall(ns + 'input')}
    P = [int(x) for x in tri.find(ns + 'p').text.split()]
    st = max(inp.values()) + 1
    out = []
    for i in range(0, len(P), st * 3):
        xy, us = [], []
        for k in range(3):
            v = P[i + k * st + inp['VERTEX']]
            t = P[i + k * st + inp['TEXCOORD']]
            xy.append((pos[v * 3] / 100.0, pos[v * 3 + 1] / 100.0))
            us.append(uvs[t * 2])
        u = (sum(us) / 3) % 1.0
        if 0.342 <= u < 0.590:
            out.append((min(p[0] for p in xy), max(p[0] for p in xy),
                        min(p[1] for p in xy), max(p[1] for p in xy)))
    return out


def existing(text):
    """AABBs of what the world still places, once our own region is gone."""
    clean = R.blank_comments(text)
    box = dict(R.SHELF)
    box.update({k: (v[0], v[1]) for k, v in R.PROP.items()})
    box.update({k: (v[0], v[1]) for k, v in FURNITURE.items()})
    box['ackermann_robot'] = (0.806, 0.485)
    out = []
    for m in re.finditer(r'[ \t]*<include>.*?</include>\n', clean, re.S):
        b = m.group(0)
        u = re.search(r'<uri>model://([^<]+)</uri>', b)
        if not u:
            continue
        full = u.group(1)
        model = full[len(PRE):] if full.startswith(PRE) else full
        if any(s in model for s in ('GroundB', 'RoofB', 'WallB')) or model not in box:
            continue
        p = [float(v) for v in re.search(r'<pose>([^<]*)</pose>', b).group(1).split()]
        dx, dy = box[model]
        c, s = abs(math.cos(p[5])), abs(math.sin(p[5]))
        hw, hh = (dx * c + dy * s) / 2, (dx * s + dy * c) / 2

        # A bay prop is not just where it SPAWNS. In the dynamic worlds the
        # kinematic plugin drives it between bays, and route_all's --edge pushes
        # the destination to the bay's far edge: props reach x = 20.059, while the
        # prop GRID only reaches 18.948. Sizing a wall band off the grid would have
        # parked a 24 m row of shelving directly in their path. Sweep the waypoints.
        pts = [(p[0], p[1])] + [tuple(float(v) for v in wp.split())
                                for wp in re.findall(r'<waypoint>([^<]*)</waypoint>', b)]
        cx = (min(q[0] for q in pts) + max(q[0] for q in pts)) / 2.0
        cy = (min(q[1] for q in pts) + max(q[1] for q in pts)) / 2.0
        hw += (max(q[0] for q in pts) - min(q[0] for q in pts)) / 2.0
        hh += (max(q[1] for q in pts) - min(q[1] for q in pts)) / 2.0
        out.append((model, cx, cy, hw, hh))
    return out


def rect(model, x, y, yaw):
    dx, dy = FURNITURE[model][:2]
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    return (x - (dx * c + dy * s) / 2, x + (dx * c + dy * s) / 2,
            y - (dx * s + dy * c) / 2, y + (dx * s + dy * c) / 2)


def hits(a, b, m=0.0):
    # 1e-9 slack: a unit placed at EXACTLY the margin is legal, and without this
    # the comparison decides it on float noise -- three placements were rejected
    # as 'too close' at a gap of precisely RUN_GAP.
    e = 1e-9
    return (a[0] < b[1] + m - e and a[1] > b[0] - m + e and
            a[2] < b[3] + m - e and a[3] > b[2] - m + e)


def strip_region(text):
    """Delete every BEGIN..END near-wall region, whatever shape its markers are.

    Line-based on purpose. A regex has to know which line the BEGIN token sits on,
    and when that changed between versions the pattern silently matched nothing --
    so re-running stacked a second copy of every object instead of replacing it,
    and the only symptom was 'too close to <itself>'. Walking lines cannot miss.
    """
    n = 0
    while True:
        lines = text.split('\n')
        b = next((i for i, l in enumerate(lines) if 'BEGIN near-wall objects' in l), None)
        if b is None:
            return text, n
        while b > 0 and '<!--' not in lines[b]:
            b -= 1
        e = next((i for i, l in enumerate(lines) if 'END near-wall objects' in l), None)
        if e is None:
            return text, n
        while e < len(lines) - 1 and '-->' not in lines[e]:
            e += 1
        del lines[b:e + 1]
        text = '\n'.join(lines)
        n += 1


def main():
    worlds = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == '--world']
    dry = '--dry-run' in sys.argv
    lanes = green_quads()

    for w in worlds:
        path = w if os.path.isabs(w) else os.path.join(ROOT, w)
        text = open(path).read()

        # Everything this script owns, gone: the delimited region, anything the
        # earlier add_wall_shelves.py left, and the loose trash can (it is
        # re-emitted inside the north group so the group is complete).
        text, n_region = strip_region(text)
        text, n_old = re.subn(
            r'[ \t]*<include>\s*<uri>model://%sShelf[DE]_01</uri>\s*'
            r'<name>%sShelf[DE]_01_1\d\d</name>.*?</include>\n' % (PRE, PRE),
            '', text, flags=re.S)
        text, n_can = re.subn(
            r'[ \t]*<include>\s*<uri>model://%sTrashCanC_01</uri>.*?</include>\n' % PRE,
            '', text, flags=re.S)

        ents = existing(text)
        out, report, placed = [BEGIN], [], 0
        n = 101
        for wall, note, items in GROUPS:
            good = []
            for model, x, y, yaw, fixed, on_bay in items:
                r = rect(model, x, y, yaw)
                why = None
                if not (WX0 < r[0] and r[1] < WX1 and WY0 < r[2] and r[3] < WY1):
                    why = 'outside the wall'
                # A wall band stands ON the bay outline by design: the bays stop
                # 0.30 m short of the wall but their prop grid stops a metre short
                # of the OUTLINE, so that strip is bay-marked floor no prop uses.
                # Clearance from props and their swept paths replaces the rect test.
                for i, bay in enumerate(R.BAYS):
                    if why or on_bay:
                        break
                    if hits(r, bay[1:], BAY_MARGIN):
                        why = 'overlaps bay %d (%s)' % (i, R.ZONE[i])
                for q in lanes:
                    if why:
                        break
                    if hits(r, q, LANE_MARGIN):
                        why = 'on a walkway strip at y=%.2f' % ((q[2] + q[3]) / 2)
                for e in ents:
                    if why:
                        break
                    eb = (e[1] - e[3], e[1] + e[3], e[2] - e[4], e[2] + e[4])
                    if hits(r, eb, RUN_GAP if len(e) > 5 else ENT_MARGIN):
                        why = 'too close to %s at (%.2f, %.2f)' % (e[0], e[1], e[2])
                if why:
                    report.append('      SKIP %-14s (%7.2f,%7.2f): %s'
                                  % (model, x, y, why))
                    continue
                name = fixed if fixed else '%s_%d' % (model, n)
                if not fixed:
                    n += 1
                z = R.FLOOR_TOP - FURNITURE[model][2]
                good.append(BLOCK % (PRE, model, PRE, name, x, y, z, yaw))
                report.append('      ok   %-14s (%7.2f,%7.2f) x[%7.2f,%7.2f]'
                              ' y[%7.2f,%7.2f]' % (model, x, y, r[0], r[1], r[2], r[3]))
                ents.append((model, x, y, (r[1] - r[0]) / 2, (r[3] - r[2]) / 2, 'new'))
                placed += 1
            if good:
                out.append(GHEAD % (wall, len(good), note))
                out.extend(good)
                out.append(GTAIL % wall)
                out.append('\n')
        out.append(END)

        # Insert before the first bay prop, or -- in a world that has none, such as
        # the *_objonwallonly variants -- before </world>. Falling back to the end
        # of the FILE put the whole region after </sdf>: 'junk after document
        # element', which Gazebo would have loaded and ElementTree would not.
        clean_now = R.blank_comments(text)
        anchor = re.search(r'[ \t]*<include>(?:(?!</include>).)*?model://%s'
                           r'(?:Bucket|ClutteringA|ClutteringC|ClutteringD)_01'
                           % PRE, clean_now, re.S)
        if anchor:
            at = anchor.start()
        else:
            close = re.search(r'[ \t]*</world>', clean_now)
            if close is None:
                raise SystemExit('%s has no </world> to insert before'
                                 % os.path.basename(path))
            at = close.start()
        text = text[:at] + ''.join(out) + text[at:]

        print('  %s' % os.path.basename(path))
        if n_region or n_old or n_can:
            print('      cleared: %d region, %d loose shelf, %d trash can'
                  % (n_region, n_old, n_can))
        for line in report:
            print(line)
        print('      placed %d in %d groups' % (placed, len(GROUPS)))
        if not dry:
            open(path, 'w').write(text)

    if dry:
        print('\n(dry run -- nothing written)')


if __name__ == '__main__':
    main()
