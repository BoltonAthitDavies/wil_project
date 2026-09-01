#!/usr/bin/env python3
"""Bay groups that shuttle between bay centres, driven by TrajectoryFollower.

    python3 reshuffle_bays.py [--seed N] [--speed V] [--per-bay-max N] [--dry-run]

Each bay's props are a GROUP. The group's 2D centroid is the thing being navigated;
the goal is another bay's centre. Every object gets waypoints offset by its own
position, so the group translates rigidly.

WHY NOT VelocityControl
    It pins VELOCITY and writes z = 0 every step, cancelling gravity. Contact
    displacement within a step is never undone, so a prop ratchets out of the floor
    and tilts. It also cannot stop. TrajectoryFollower drives with force and torque,
    so gravity still works, and <loop> / waypoints give real destinations.

THREE THINGS THAT SILENTLY BREAK THIS
    1. <link_name> is REQUIRED, and Bucket_01's link is 'body' while every other
       prop's is 'link'. Wrong name -> that model just never moves.
    2. Force is applied along the link's LOCAL +X. An object not already facing its
       goal gets pushed sideways. So every object in a group is pre-oriented to
       yaw = atan2(D.y, D.x); one shared yaw per group.
    3. Force must beat static friction AND scale with mass, or the formation
       shears:  F = m * (a + MU_EFF * g).

       MU_EFF is MEASURED, not taken from the SDF. The declared coefficients
       (prop 0.6, ground 100) predict nothing useful: a bisection in the real
       world showed props immobile at F = m(3 + 0.6g) and moving at
       F = m(10 + 0.6g), which brackets the true effective mu at roughly
       1.0-1.6 -- neither the prop's 0.6 nor the ground's 100. Lowering the
       ground's mu to 0.2 changed nothing, so it is not a simple min/max of
       the two. MU_EFF = 1.6 is the safe upper end of that bracket.

GROUPS NEVER COLLIDE
    Goals are chosen so the groups' swept corridors are pairwise disjoint. That is
    what makes looping safe with zero timing coordination: two groups can never
    meet, at any phase, however far out of sync they drift.

Run after scale_warehouse.py / add_bays.py / fill_bays.py.
Touches small_warehouse_dynamic.world only.
"""
import re, os, sys, math, random, itertools
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
WORLD = os.path.join(ROOT, 'aws-robomaker-small-warehouse-world', 'worlds',
                     'small_warehouse_dynamic', 'small_warehouse_dynamic.world')
STATIC_WORLD = os.path.join(ROOT, 'aws-robomaker-small-warehouse-world', 'worlds',
                            'small_warehouse_static', 'small_warehouse_static.world')
PRE = 'aws_robomaker_warehouse_'

FLOOR_TOP = 0.034223           # ground pose z -0.090092 + slab top 0.124315
G = 9.81
# Effective prop-on-floor friction, measured by bisection in this world (see the
# module docstring). Do NOT substitute the SDF mu values -- they do not predict
# the observed threshold.
MU_EFF = 1.6
# Turning torque also has to beat friction: resisting torque is roughly
# MU_EFF * m * g * r_eff with r_eff ~ 0.6 m for these footprints. The plugin's
# default 50 Nm is fine but 2 Nm is not -- at 2 Nm the groups completed the
# outbound leg and then stalled forever at the far waypoint, unable to turn
# around for the return leg. Scaled with mass, like the force.
TORQUE_K = 25.0                # Nm per kg
MARGIN = 0.25                  # corridor clearance
DWELL = 2.0                    # s paused at each end of the shuttle
WALL = (-20.656, 20.627, -20.642, 20.642)

def _markings():
    """(bay rectangles, N-S lane line x centres, lane-D y range) read FROM THE MESH.

    Nothing about the floor is hard-coded any more: reshape_floor.py can move the
    lanes and resize the bays freely, and this follows. A stale copy of the bay
    table was exactly the kind of desync that bit the launch files.
    """
    import xml.etree.ElementTree as ET
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

    def band(u):
        u %= 1.0
        return 'hazard' if u < 0.342 else ('green' if u < 0.590 else 'other')

    quads = {'hazard': [], 'green': []}
    for i in range(0, len(P), st * 3):
        xy, us = [], []
        for k in range(3):
            v = P[i + k * st + inp['VERTEX']]
            t = P[i + k * st + inp['TEXCOORD']]
            xy.append((pos[v * 3] / 100.0, pos[v * 3 + 1] / 100.0))
            us.append(uvs[t * 2])
        b = band(sum(us) / 3)
        if b in quads:
            quads[b].append((min(p[0] for p in xy), max(p[0] for p in xy),
                             min(p[1] for p in xy), max(p[1] for p in xy)))

    # cluster hazard strips into bay rectangles
    used = [False] * len(quads['hazard'])
    rects = []
    for i in range(len(quads['hazard'])):
        if used[i]:
            continue
        grp = [i]; used[i] = True; changed = True
        while changed:
            changed = False
            bx = (min(quads['hazard'][j][0] for j in grp), max(quads['hazard'][j][1] for j in grp),
                  min(quads['hazard'][j][2] for j in grp), max(quads['hazard'][j][3] for j in grp))
            for k in range(len(quads['hazard'])):
                if used[k]:
                    continue
                q = quads['hazard'][k]
                if q[0] < bx[1] + 0.3 and q[1] > bx[0] - 0.3 and q[2] < bx[3] + 0.3 and q[3] > bx[2] - 0.3:
                    grp.append(k); used[k] = True; changed = True
        rects.append((min(quads['hazard'][j][0] for j in grp), max(quads['hazard'][j][1] for j in grp),
                      min(quads['hazard'][j][2] for j in grp), max(quads['hazard'][j][3] for j in grp)))

    ns_lines = sorted({round((q[0] + q[1]) / 2, 3) for q in quads['green']
                       if (q[3] - q[2]) > (q[1] - q[0])})
    ew = [q for q in quads['green'] if (q[1] - q[0]) > (q[3] - q[2]) and q[0] > 0]
    laneD = (min(q[2] for q in ew), max(q[3] for q in ew)) if ew else (1e9, 1e9)
    rects.sort(key=lambda r: (round((r[0] + r[1]) / 2, 1), r[2]))
    return rects, ns_lines, laneD


_RECTS, _NSL, _LANED = _markings()
# column boundaries from the two N-S lanes; east splits at lane D
_XA = (_NSL[1] + _NSL[2]) / 2 if len(_NSL) >= 4 else 0.0
_XB = (_NSL[3] + _NSL[2]) / 2 if len(_NSL) >= 4 else 0.0


def _zone_of(r):
    cx, cy = (r[0] + r[1]) / 2, (r[2] + r[3]) / 2
    if cx < _NSL[0]:
        return 'west'
    if cx < _NSL[2]:
        return 'stock'
    return 'east_n' if cy > _LANED[1] else 'east_s'


# BAYS: (size class, x0, x1, y0, y1) -- size class only drives the cell grid now.
# The split is RELATIVE, not a fixed metre threshold: bay widths differ between
# floor revisions (6.0/9.0 m before reshape_floor.py, 7.9/14.6 m after), and an
# absolute cut-off classified every old bay as 'small' and broke the sampler.
_W = [r[1] - r[0] for r in _RECTS]
_SPLIT = (min(_W) + max(_W)) / 2
BAYS = [('big' if (r[1] - r[0]) > _SPLIT else 'small', r[0], r[1], r[2], r[3])
        for r in _RECTS]
ZONE = {i: _zone_of(r) for i, r in enumerate(_RECTS)}
T_X, T_Y = 0.2176, 0.1446

CAP = {'big': 6, 'small': 4}   # 3x2 and 2x2 grids

# model -> (dx, dy, collision z_min, mass, mu, link name)
PROP = {
    'Bucket_01':      (0.941, 1.222, 0.005, 2.0, 0.4, 'body'),
    'ClutteringA_01': (2.161, 2.002, 0.030, 2.0, 0.6, 'link'),
    'ClutteringC_01': (1.773, 2.060, 0.030, 1.0, 0.6, 'link'),
    'ClutteringD_01': (1.017, 1.522, 0.333, 1.0, 0.6, 'link'),
}
SHELF = {'ShelfD_01': (3.916, 0.879), 'ShelfE_01': (3.916, 0.879),
         'ShelfF_01': (2.102, 18.047), 'TrashCanC_01': (1.479, 0.909)}

BLOCK = ('    <include>\n'
         '      <uri>model://%s%s</uri>\n'
         '      <name>%s%s_%d</name>\n'
         '      <pose>%.6f %.6f %.6f 0 0 %.6f</pose>\n%s'
         '    </include>\n')
# Props stay STATIC and are driven kinematically. Force-based TrajectoryFollower
# needed <static>false</static>, and 30 dynamic bodies resting on the floor cost
# 3.4x real time (RTF 1.005 -> 0.294, cameras 30 Hz -> 9 Hz). The cost is the
# persistent contact constraints, not the motion: the same 30 dynamic bodies in
# zero gravity, touching nothing, ran at RTF 1.006. Static bodies never enter the
# solver, and a static model moved by pose still carries its collision with it.
def kinematic_block(speed, dwell, pts):
    """Static prop + kinematic waypoint follower. Any number of waypoints."""
    wps = ''.join('          <waypoint>%.4f %.4f</waypoint>\n' % (x, y) for x, y in pts)
    return ('      <static>true</static>\n'
            '      <plugin filename="KinematicTrajectory"\n'
            '              name="kinematic_trajectory::KinematicTrajectory">\n'
            '        <speed>%.3f</speed>\n'
            '        <loop>true</loop>\n'
            '        <dwell>%.2f</dwell>\n'
            '        <waypoints>\n%s'
            '        </waypoints>\n'
            '      </plugin>\n') % (speed, dwell, wps)


def blank_comments(text):
    """Same-length copy with comment bodies blanked to spaces.

    Scanning raw text does not work: a commented block ends '</include> -->', so a
    match starting at its '<include>' cannot close there and runs on to swallow the
    next LIVE block. Blanking first makes every match real; offsets still line up.
    """
    out = list(text)
    for c in re.finditer(r'<!--.*?-->', text, re.S):
        for i in range(c.start(), c.end()):
            if out[i] != '\n':
                out[i] = ' '
    return ''.join(out)


def sit_z(model):
    """Pose z placing the model's collision floor exactly on the floor surface.

    NOT the stock pose z, which buries every prop 20-29 mm. Harmless while static;
    for a dynamic body it is a permanent contact the solver fights every step.
    """
    return FLOOR_TOP - PROP[model][2]


def half(model, yaw):
    dx, dy = PROP[model][:2]
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    return (dx * c + dy * s) / 2, (dx * s + dy * c) / 2


def torque_for(model):
    """Turning torque, scaled with mass so every prop turns at a similar rate."""
    return PROP[model][3] * TORQUE_K


def force_for(model, accel):
    """F = m(a + MU_EFF*g): beat measured static friction, then accelerate at 'a'."""
    m = PROP[model][3]
    return m * (accel + MU_EFF * G)


def cells(bi):
    """Grid slots inside a bay's interior. Big bays 3x2, small 2x2."""
    kind, x0, x1, y0, y1 = BAYS[bi]
    ix0, ix1, iy0, iy1 = x0 + T_X, x1 - T_X, y0 + T_Y, y1 - T_Y
    ncol = 3 if kind == 'big' else 2
    px = [ix0 + (ix1 - ix0) * (i + 0.5) / ncol for i in range(ncol)]
    py = [iy0 + (iy1 - iy0) * 0.25, iy0 + (iy1 - iy0) * 0.75]
    return [(x, y) for y in py for x in px]


def group_aabb(objs):
    return (min(o['x'] - o['hw'] for o in objs), max(o['x'] + o['hw'] for o in objs),
            min(o['y'] - o['hh'] for o in objs), max(o['y'] + o['hh'] for o in objs))


def swept(objs, D):
    """Conservative AABB of the whole translation: union of start and end boxes."""
    x0, x1, y0, y1 = group_aabb(objs)
    return (min(x0, x0 + D[0]) - MARGIN, max(x1, x1 + D[0]) + MARGIN,
            min(y0, y0 + D[1]) - MARGIN, max(y1, y1 + D[1]) + MARGIN)


def overlaps(A, B):
    return A[0] < B[1] and A[1] > B[0] and A[2] < B[3] and A[3] > B[2]


def parse_props(text):
    """(spans, model list, shelf boxes) for live includes. Stacks are flattened."""
    clean = blank_comments(text)
    spans, models, shelves = [], [], []
    for m in re.finditer(r'[ \t]*<include>.*?</include>\n', clean, re.S):
        b = m.group(0)
        uri = re.search(r'model://(\S+?)</uri>', b)
        if not uri:
            continue
        model = uri.group(1)[len(PRE):]
        p = [float(v) for v in re.search(r'<pose>([^<]*)</pose>', b).group(1).split()]
        if model in PROP:
            spans.append((m.start(), m.end()))
            models.append(model)         # identity only; positions are re-derived
        elif model in SHELF:
            dx, dy = SHELF[model]
            c, s = abs(math.cos(p[5])), abs(math.sin(p[5]))
            shelves.append({'x': p[0], 'y': p[1],
                            'hw': (dx * c + dy * s) / 2, 'hh': (dx * s + dy * c) / 2})
    return spans, models, shelves


def main():
    seed = int(sys.argv[sys.argv.index('--seed') + 1]) if '--seed' in sys.argv else 6
    speed = float(sys.argv[sys.argv.index('--speed') + 1]) if '--speed' in sys.argv else 1.5
    pbmax = int(sys.argv[sys.argv.index('--per-bay-max') + 1]) if '--per-bay-max' in sys.argv else 99
    # Fixed net acceleration beats the uniform-cruise formula in practice: sizing
    # 'a' from distance gave 0.08-0.18 m/s^2, i.e. a force only 1-3% above the
    # friction limit, which ODE simply absorbs -- the props never broke static
    # friction at all. A flat 'a' keeps every force comfortably above mu*m*g.
    accel_fixed = float(sys.argv[sys.argv.index('--accel') + 1]) if '--accel' in sys.argv else 0.8
    ncross = int(sys.argv[sys.argv.index('--cross') + 1]) if '--cross' in sys.argv else 2
    dry = '--dry-run' in sys.argv
    rng = random.Random(seed)

    text = open(WORLD).read()
    spans, models, shelves = parse_props(text)
    if not spans:
        raise SystemExit('no live prop includes found -- wrong world file?')
    insert_at = spans[0][0]
    print('  %d props found (stacks flattened -- every prop gets its own floor cell)'
          % len(models))

    # --- choose 6 home bays and deal the props out, respecting capacity ----------
    big = [i for i, b in enumerate(BAYS) if b[0] == 'big']
    small = [i for i, b in enumerate(BAYS) if b[0] == 'small']
    home = sorted(rng.sample(big, 4) + rng.sample(small, 2))
    cap = {i: min(CAP[BAYS[i][0]], pbmax) for i in home}
    if sum(cap.values()) < len(models):
        raise SystemExit('capacity %d < %d props; raise --per-bay-max'
                         % (sum(cap.values()), len(models)))
    rng.shuffle(models)
    counts = {i: 1 for i in home}
    for _ in range(len(models) - len(home)):
        counts[rng.choice([k for k in home if counts[k] < cap[k]])] += 1
    deal, k = {}, 0
    for i in home:
        deal[i] = models[k:k + counts[i]]
        k += counts[i]

    # --- lay each group out on its bay grid (yaw applied later) -----------------
    groups = {}
    for bi in home:
        pos = cells(bi)
        rng.shuffle(pos)
        objs = []
        for mdl, (px, py) in zip(deal[bi], pos):
            hw, hh = half(mdl, 0.0)
            objs.append({'model': mdl, 'x': px, 'y': py, 'hw': hw, 'hh': hh})
        groups[bi] = objs

    # --- pick goals: distinct, reachable, corridors pairwise disjoint -----------
    empty = [i for i in range(len(BAYS)) if i not in home]

    def route_ok(objs, pts):
        """Swept AABB of a whole polyline route, and whether it is clear."""
        x0, x1, y0, y1 = group_aabb(objs)
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        xs, ys = [], []
        for (px, py) in pts:
            xs += [x0 + (px - cx), x1 + (px - cx)]
            ys += [y0 + (py - cy), y1 + (py - cy)]
        S = (min(xs) - MARGIN, max(xs) + MARGIN, min(ys) - MARGIN, max(ys) + MARGIN)
        if any(overlaps(S, (o['x'] - o['hw'], o['x'] + o['hw'],
                            o['y'] - o['hh'], o['y'] + o['hh'])) for o in shelves):
            return None
        if not (WALL[0] < S[0] and S[1] < WALL[1] and WALL[2] < S[2] and S[3] < WALL[3]):
            return None
        return S

    def fits(objs, e):
        kind, x0, x1, y0, y1 = BAYS[e]
        gw = group_aabb(objs)
        return (gw[1] - gw[0]) <= (x1 - x0) - 2 * T_X and (gw[3] - gw[2]) <= (y1 - y0) - 2 * T_Y

    def centre(e):
        kind, x0, x1, y0, y1 = BAYS[e]
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    # --- cross-zone routes: home -> same-zone bay -> DIFFERENT-zone bay --------
    # Collect every feasible route per group, then search for a mutually
    # compatible SET. Greedy selection fails here: the first group's corridor is
    # wide enough to starve every later one, so one group would always win alone.
    cross_cand = defaultdict(list)
    for bi in sorted(groups):
        objs = groups[bi]
        home_pt = (sum(o['x'] for o in objs) / len(objs),
                   sum(o['y'] for o in objs) / len(objs))
        for mid in empty:
            if ZONE[mid] != ZONE[bi] or not fits(objs, mid):
                continue
            for far in empty:
                if far == mid or ZONE[far] == ZONE[bi] or not fits(objs, far):
                    continue
                pts = [home_pt, centre(mid), centre(far)]
                S = route_ok(objs, pts)
                if S is None:
                    continue
                if any(overlaps(S, group_aabb(groups[o])) for o in groups if o != bi):
                    continue
                d = math.dist(pts[0], pts[1]) + math.dist(pts[1], pts[2])
                cross_cand[bi].append((pts, S, mid, far, d))
    for bi in cross_cand:
        cross_cand[bi].sort(key=lambda c: c[4])

    cross = {}
    for n in range(min(ncross, len(cross_cand)), 0, -1):
        for combo in itertools.combinations(sorted(cross_cand), n):
            for pick in itertools.product(*[cross_cand[b][:12] for b in combo]):
                goals = [g for c in pick for g in (c[2], c[3])]
                if len(set(goals)) != len(goals):
                    continue
                if all(not overlaps(pick[i][1], pick[j][1])
                       for i, j in itertools.combinations(range(n), 2)):
                    cross = {b: (c[0], c[1], c[2], c[3])
                             for b, c in zip(combo, pick)}
                    break
            if cross:
                break
        if cross:
            break

    cand = defaultdict(list)
    used_goals = {g for c in cross.values() for g in (c[2], c[3])}
    for bi, objs in groups.items():
        if bi in cross:
            continue
        cx = sum(o['x'] for o in objs) / len(objs)
        cy = sum(o['y'] for o in objs) / len(objs)
        gw = group_aabb(objs)
        for e in empty:
            if e in used_goals:
                continue
            kind, x0, x1, y0, y1 = BAYS[e]
            # Same size class, so "5 big + 3 small empty" holds at BOTH endpoints of
            # the shuttle, not just at spawn.
            if kind != BAYS[bi][0]:
                continue
            if (gw[1] - gw[0]) > (x1 - x0) - 2 * T_X or (gw[3] - gw[2]) > (y1 - y0) - 2 * T_Y:
                continue
            D = ((x0 + x1) / 2 - cx, (y0 + y1) / 2 - cy)
            S = swept(objs, D)
            if any(overlaps(S, (o['x'] - o['hw'], o['x'] + o['hw'],
                                o['y'] - o['hh'], o['y'] + o['hh'])) for o in shelves):
                continue
            if not (WALL[0] < S[0] and S[1] < WALL[1] and WALL[2] < S[2] and S[3] < WALL[3]):
                continue
            # the corridor must also miss every OTHER group's home bay, and every
            # cross-zone route already reserved above
            if any(overlaps(S, group_aabb(groups[o])) for o in groups if o != bi):
                continue
            if any(overlaps(S, c[1]) for c in cross.values()):
                continue
            cand[bi].append((e, D, S, math.hypot(*D)))
    best = None
    for r in range(len(cand), 0, -1):
        for combo in itertools.combinations(sorted(cand), r):
            for pick in itertools.product(*[cand[b] for b in combo]):
                if len({p[0] for p in pick}) != r:
                    continue
                if all(not overlaps(pick[i][2], pick[j][2])
                       for i, j in itertools.combinations(range(r), 2)):
                    best = (combo, pick)
                    break
            if best:
                break
        if best:
            break
    if not best:
        raise SystemExit('no goal assignment with disjoint corridors; try another --seed')
    moving = dict(zip(best[0], best[1]))

    # --- pre-orient to the travel heading, then emit -----------------------------
    blocks, blocks_static, placed = [], [], []
    n = 400
    # Per-group route as OFFSETS from the home pose, so every object in the group
    # gets the same displacement sequence and the formation stays rigid.
    routes = {}
    for bi in home:
        if bi in cross:
            pts = cross[bi][0]
            routes[bi] = [(p[0] - pts[0][0], p[1] - pts[0][1]) for p in pts]
        elif bi in moving:
            routes[bi] = [(0.0, 0.0), moving[bi][1]]

    for bi in home:
        rt = routes.get(bi)
        yaw = math.atan2(rt[1][1], rt[1][0]) if rt else 0.0
        for o in groups[bi]:
            mdl = o['model']
            o['hw'], o['hh'] = half(mdl, yaw)
            plug = ''
            if rt:
                plug = kinematic_block(speed, DWELL,
                                       [(o['x'] + dx, o['y'] + dy) for dx, dy in rt])
            blocks.append(BLOCK % (PRE, mdl, PRE, mdl, n, o['x'], o['y'],
                                   sit_z(mdl), yaw, plug))
            # Same props, same poses, no plugin: small_warehouse_static is the
            # motionless twin of the dynamic world, not a separate layout.
            blocks_static.append(BLOCK % (PRE, mdl, PRE, mdl, n, o['x'], o['y'],
                                          sit_z(mdl), yaw, '      <static>true</static>\n'))
            placed.append((mdl, bi, bool(rt)))
            n += 1

    for (a, b) in reversed(spans):
        text = text[:a] + text[b:]
        if b <= insert_at:
            insert_at -= (b - a)
    text = text[:insert_at] + ''.join(blocks) + text[insert_at:]

    # The wall-side TrashCan is not a bay prop, so it is not part of the
    # disjoint-corridor guarantee. Strip any motion it inherited from an earlier
    # pass rather than let an unplanned mover loose in the aisles.
    text = re.sub(
        r'(<name>%sTrashCanC_01_002</name>\s*\n\s*<pose>[^<]*</pose>\n)'
        r'(?:\s*<static>[^<]*</static>\n)?'
        r'(?:\s*<plugin .*?</plugin>\n)?' % re.escape(PRE),
        lambda m: m.group(1), text, count=1, flags=re.S)

    print('  home bays %s\n' % [i + 1 for i in home])
    print('  %-5s %-5s %-5s %-7s %-22s %-9s %-9s %s'
          % ('bay', 'objs', 'goal', 'dist', 'displacement', 'yaw deg', 'accel', 'cruise'))
    for bi in home:
        if bi in cross:
            pts, S, mid, far = cross[bi]
            d = math.dist(pts[0], pts[1]) + math.dist(pts[1], pts[2])
            print('  %-5d %-5d %-5s %-7.2f CROSS-ZONE: %s -> bay %d (%s) -> bay %d (%s)'
                  % (bi + 1, len(groups[bi]), '3wp', d, ZONE[bi], mid + 1, ZONE[mid],
                     far + 1, ZONE[far]))
            continue
        mv = moving.get(bi)
        if mv:
            e, D, S, d = mv
            print('  %-5d %-5d %-5d %-7.2f (%+7.3f,%+7.3f)      %-9.1f %-9.3f %.2f m/s'
                  % (bi + 1, len(groups[bi]), e + 1, d, D[0], D[1],
                     math.degrees(math.atan2(D[1], D[0])), accel_fixed,
                     math.sqrt(2 * accel_fixed * d)))
        else:
            print('  %-5d %-5d %-5s static' % (bi + 1, len(groups[bi]), '-'))
    print('\n  %d props, %d moving, %d groups shuttling'
          % (len(placed), sum(1 for p in placed if p[2]), len(moving)))
    if dry:
        print('\n(dry run -- nothing written)')
        return
    open(WORLD, 'w').write(text)
    print('\nWrote %s' % os.path.basename(WORLD))

    # --- mirror the same layout into the static world, frozen -------------------
    st = open(STATIC_WORLD).read()
    sspans, _, _ = parse_props(st)
    if sspans:
        at = sspans[0][0]
        for (a, b) in reversed(sspans):
            st = st[:a] + st[b:]
            if b <= at:
                at -= (b - a)
        st = st[:at] + ''.join(blocks_static) + st[at:]
        open(STATIC_WORLD, 'w').write(st)
        print('Wrote %s  (%d props, no plugins)'
              % (os.path.basename(STATIC_WORLD), len(blocks_static)))


if __name__ == '__main__':
    main()
