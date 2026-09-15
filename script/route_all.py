#!/usr/bin/env python3
"""Re-solve the motion of an existing world so EVERY bay group moves.

    python3 script/route_all.py <world.world> [--speed V] [--dwell S]
                                [--same-zone] [--uniform-zone Z,Z]
                                [--along-length] [--furthest] [--sequence]
                                [--dry-run]

    --furthest   pick the LONGEST legal in-zone route instead of the shortest.
    --sequence   in a zone holding several groups, stagger them so the first
                 departs, the second departs once the first has arrived, and the
                 second is back before the first starts home.

                 That nesting is not free. With equal periods it is impossible:
                 requiring g1's whole round trip to fit inside g0's far-end dwell
                 while T+d matches for both reduces to T1 + T0 < 0. So g1 is given
                 HALF g0's period -- it makes two round trips per g0 cycle, the
                 first of which nests -- and g0's dwell is sized to contain it:
                     d0 = 2*T1 + T0 + gap        d1 = (T0 + d0)/2 - T1

                 Sequencing also RELAXES the corridor rule inside that zone: two
                 groups separated in TIME may share space, which is what makes a
                 furthest-bay move possible in a column where every corridor
                 overlaps. The result is checked by sampling the schedule over a
                 full period and testing every pair of group footprints.

    --along-length  travel only along each zone's LONG axis. Every zone here is
                    taller than it is wide (west 38.0 x 7.9 m, stock 38.5 x 14.6,
                    east_s 23.9 x 13.9, east_n 15.2 x 13.9), so this means +/-y.
                    Without it an in-bay shuttle picks whichever axis has more
                    slack inside the BAY, which for the wide stock bays is x --
                    i.e. across the zone's width, not along its length.

    --same-zone   a group may only travel to a bay in its OWN zone (a region
                  enclosed by the green walkway lines), so nothing crosses a
                  walkway. In-bay shuttles are same-zone by definition, and a
                  group alone in its zone (east_n holds only bay 14) can do
                  nothing else.

Prop POSITIONS are left exactly as they are -- only the KinematicTrajectory
waypoints are recomputed. Use this when a generated world has groups stuck static
and you would rather keep the layout than reshuffle it.

WHY A JOINT SOLVE
    reshuffle_bays.py assigns goals greedily and then keeps whatever still fits.
    One wide 3-waypoint cross-zone corridor can block every remaining group -- in
    small_warehouse_dynamic_00 it left bays 11 and 12 with no legal goal at all.
    Solving all groups together, and preferring SHORT routes, finds an assignment
    where all of them move.

The disjoint-corridor invariant is unchanged: a group's swept AABB may not touch
another group's swept AABB, any shelf, any other group's home bay, or the wall.
That is what makes the looping safe with no timing coordination.
"""
import re, os, sys, math, itertools
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bake_map
import importlib.util
_spec = importlib.util.spec_from_file_location('rb', os.path.join(HERE, 'reshuffle_bays.py'))
rb = importlib.util.module_from_spec(_spec)
_argv = sys.argv[:]
sys.argv = ['rb', '--dry-run']
try:
    _spec.loader.exec_module(rb)
except SystemExit:
    pass
sys.argv = _argv

FOOT = {k: (v[0], v[1]) for k, v in rb.PROP.items()}
SHELF = rb.SHELF
M = rb.MARGIN


def boxes(objs):
    return (min(o['x'] - o['hw'] for o in objs), max(o['x'] + o['hw'] for o in objs),
            min(o['y'] - o['hh'] for o in objs), max(o['y'] + o['hh'] for o in objs))


def swept(objs, pts):
    x0, x1, y0, y1 = boxes(objs)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    xs, ys = [], []
    for px, py in pts:
        xs += [x0 + px - cx, x1 + px - cx]
        ys += [y0 + py - cy, y1 + py - cy]
    return (min(xs) - M, max(xs) + M, min(ys) - M, max(ys) + M)


def ov(A, B):
    return A[0] < B[1] and A[1] > B[0] and A[2] < B[3] and A[3] > B[2]


def main():
    world = sys.argv[1]
    speed = float(sys.argv[sys.argv.index('--speed') + 1]) if '--speed' in sys.argv else 2.5
    dwell = float(sys.argv[sys.argv.index('--dwell') + 1]) if '--dwell' in sys.argv else 2.0
    dry = '--dry-run' in sys.argv
    same_zone = '--same-zone' in sys.argv
    along_len = '--along-length' in sys.argv
    furthest = '--furthest' in sys.argv
    edge = '--edge' in sys.argv
    convoy = '--convoy' in sys.argv
    lag = float(sys.argv[sys.argv.index('--lag') + 1]) if '--lag' in sys.argv else 0.0
    sequence = '--sequence' in sys.argv

    # Zone extents = union of that zone's bay rectangles; long axis from those.
    zext = {}
    for i, b in enumerate(rb.BAYS):
        z = rb.ZONE[i]
        e = zext.get(z)
        zext[z] = (min(b[1], e[0]) if e else b[1], max(b[2], e[1]) if e else b[2],
                   min(b[3], e[2]) if e else b[3], max(b[4], e[3]) if e else b[4])
    zlong = {z: ('y' if (e[3] - e[2]) >= (e[1] - e[0]) else 'x') for z, e in zext.items()}
    # --uniform-zone stock,east_s : every group in that zone must travel along the
    # SAME world vector. Enforced inside the search, not filtered afterwards, so
    # backtracking can find a combination that also keeps corridors disjoint.
    uni = set()
    if '--uniform-zone' in sys.argv:
        uni = {z.strip() for z in sys.argv[sys.argv.index('--uniform-zone') + 1].split(',')}

    text = open(world).read()
    clean = rb.blank_comments(text)

    groups, shelves, spans = defaultdict(list), [], {}
    for m in re.finditer(r'[ \t]*<include>.*?</include>\n', clean, re.S):
        blk = m.group(0)
        uri = re.search(r'model://(\S+?)</uri>', blk)
        if not uri:
            continue
        model = uri.group(1)[len(rb.PRE):]
        p = [float(v) for v in re.search(r'<pose>([^<]*)</pose>', blk).group(1).split()]
        c, s = abs(math.cos(p[5])), abs(math.sin(p[5]))
        if model in FOOT:
            dx, dy = FOOT[model]
            bi = next((i for i, b in enumerate(rb.BAYS)
                       if b[1] <= p[0] <= b[2] and b[3] <= p[1] <= b[4]), None)
            groups[bi].append({'x': p[0], 'y': p[1], 'span': (m.start(), m.end()),
                               'hw': (dx * c + dy * s) / 2, 'hh': (dx * s + dy * c) / 2})
        elif model in SHELF:
            dx, dy = SHELF[model]
            shelves.append({'x': p[0], 'y': p[1],
                            'hw': (dx * c + dy * s) / 2, 'hh': (dx * s + dy * c) / 2})

    occ = set(groups)
    empty = [i for i in range(len(rb.BAYS)) if i not in occ]
    # Sequencing lets same-zone groups share space because they are separated in
    # time -- but only if that actually holds. Solve with the relaxation on, check
    # the timeline, and fall back to the strict disjoint-corridor solve if it
    # does not: nesting in time is NOT sufficient on its own, since one group's
    # path can run straight through where the other is parked.
    RELAX = [False]

    def solve(relax):
        RELAX[0] = relax
        # candidate 2-waypoint goals per group, shortest first
        cand = {}
        for bi, objs in groups.items():
            gw = boxes(objs)
            # Anchor every route on the group's AABB CENTRE, never its centroid. The
            # amplitude and fit checks are all AABB-based; mixing in a centroid (which
            # differs for an asymmetric group) lets the box overhang the destination.
            cx, cy = (gw[0] + gw[1]) / 2, (gw[2] + gw[3]) / 2
            lst = []
            for e in empty:
                if same_zone and rb.ZONE[e] != rb.ZONE[bi]:
                    continue
                k, x0, x1, y0, y1 = rb.BAYS[e]
                if (gw[1] - gw[0]) > (x1 - x0) - 2 * rb.T_X or (gw[3] - gw[2]) > (y1 - y0) - 2 * rb.T_Y:
                    continue
                # Park flush against the FAR inside edge of the target bay rather
                # than at its centre -- the longest legal trip into that bay.
                tx, ty = (x0 + x1) / 2, (y0 + y1) / 2
                if edge:
                    vx0, vy0 = tx - cx, ty - cy
                    ghw, ghh = (gw[1] - gw[0]) / 2, (gw[3] - gw[2]) / 2
                    if abs(vy0) >= abs(vx0):
                        ty = (y1 - rb.T_Y - ghh) if vy0 > 0 else (y0 + rb.T_Y + ghh)
                    else:
                        tx = (x1 - rb.T_X - ghw) if vx0 > 0 else (x0 + rb.T_X + ghw)
                pts = [(cx, cy), (tx, ty)]
                if along_len:
                    vx, vy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
                    if zlong[rb.ZONE[bi]] == 'y' and abs(vy) <= abs(vx):
                        continue
                    if zlong[rb.ZONE[bi]] == 'x' and abs(vx) <= abs(vy):
                        continue
                S = swept(objs, pts)
                if any(ov(S, (o['x'] - o['hw'], o['x'] + o['hw'],
                              o['y'] - o['hh'], o['y'] + o['hh'])) for o in shelves):
                    continue
                if not (rb.WALL[0] < S[0] and S[1] < rb.WALL[1]
                        and rb.WALL[2] < S[2] and S[3] < rb.WALL[3]):
                    continue
                # A corridor may not cross another group's home bay -- unless that
                # group is sequenced with this one, in which case they are separated
                # in time and the crossing is safe (verified over the timeline below).
                if any(ov(S, boxes(groups[b])) for b in groups
                       if b != bi and not (RELAX[0] and rb.ZONE[b] == rb.ZONE[bi])):
                    continue
                lst.append((e, pts, S, math.dist(pts[0], pts[1])))
            lst.sort(key=lambda c: -c[3] if furthest else c[3])
            # In-bay shuttles are candidates too, not a last-resort patch. Offering
            # them to the SEARCH lets it backtrack: a greedy pass that hands out
            # bay-to-bay routes first can leave a group with nothing legal left.
            k, x0, x1, y0, y1 = rb.BAYS[bi]
            icx = (x0 + rb.T_X + x1 - rb.T_X) / 2
            icy = (y0 + rb.T_Y + y1 - rb.T_Y) / 2
            slack_x = (x1 - rb.T_X) - (x0 + rb.T_X) - (gw[1] - gw[0])
            slack_y = (y1 - rb.T_Y) - (y0 + rb.T_Y) - (gw[3] - gw[2])
            horiz = slack_x >= slack_y
            if along_len:
                horiz = (zlong[rb.ZONE[bi]] == 'x')
            amp0 = max(0.0, (slack_x if horiz else slack_y) / 2 - 0.05)
            for f in (1.0, 0.75, 0.5, 0.35):
                amp = amp0 * f
                if amp < 0.25:
                    break
                pts = ([(icx - amp, icy), (icx + amp, icy)] if horiz
                       else [(icx, icy - amp), (icx, icy + amp)])
                lst.append((None, pts, swept(objs, pts), 2 * amp))
            cand[bi] = lst

        order = sorted(cand, key=lambda b: len(cand[b]))     # most constrained first
        best = {}

        def search(i, chosen, used):
            if i == len(order):
                return dict(chosen)
            bi = order[i]
            for (e, pts, S, d) in cand[bi]:
                if e is not None and e in used:
                    continue
                # Groups sequenced within one zone are separated in time, so they
                # are allowed to share a corridor; everything else must stay disjoint.
                if any(ov(S, c[2]) for ob, c in chosen.items()
                       if not (RELAX[0] and rb.ZONE[ob] == rb.ZONE[bi])):
                    continue
                    # Sequenced same-zone groups may not overtake: their destinations
                # must keep the same order along the heading as their origins. Without
                # this the follower is routed PAST where the leader is parked and
                # drives straight through it.
                if RELAX[0] and rb.ZONE[bi] in {rb.ZONE[b] for b in chosen}:
                    def _pr(p, v):
                        n = math.hypot(*v) or 1.0
                        return (p[0] * v[0] + p[1] * v[1]) / n
                    vv = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
                    bad_order = False
                    for ob, oc in chosen.items():
                        if rb.ZONE[ob] != rb.ZONE[bi]:
                            continue
                        o0, o1 = oc[1][0], oc[1][1]
                        if (_pr(pts[0], vv) - _pr(o0, vv)) * (_pr(pts[1], vv) - _pr(o1, vv)) < 0:
                            bad_order = True
                            break
                    if bad_order:
                        continue
                if rb.ZONE[bi] in uni:
                    # Compare HEADING, not displacement. Two groups can both travel
                    # straight down while moving 7.88 m and 7.78 m -- their bay-centre
                    # offsets differ -- and requiring equal vectors rejects that.
                    def _unit(p0, p1):
                        vx, vy = p1[0] - p0[0], p1[1] - p0[1]
                        n = math.hypot(vx, vy)
                        return (vx / n, vy / n) if n > 1e-9 else (0.0, 0.0)
                    u1 = _unit(pts[0], pts[1])
                    clash = False
                    for ob, oc in chosen.items():
                        if rb.ZONE[ob] != rb.ZONE[bi]:
                            continue
                        u2 = _unit(oc[1][0], oc[1][1])
                        if abs(u1[0] - u2[0]) > 1e-3 or abs(u1[1] - u2[1]) > 1e-3:
                            clash = True
                            break
                    if clash:
                        continue
                chosen[bi] = (e, pts, S, d)
                if e is not None:
                    used.add(e)
                r = search(i + 1, chosen, used)
                if r:
                    return r
                del chosen[bi]
                if e is not None:
                    used.discard(e)
            return None

        best = search(0, {}, set()) or {}
        print('  %d groups, %d empty bays' % (len(groups), len(empty)))
        for bi in sorted(groups):
            if bi in best:
                e, pts, S, d = best[bi]
                if e is None:
                    print('  bay %-3d %d objs -> in-bay shuttle  %6.2f m travel'
                          % (bi + 1, len(groups[bi]), d))
                else:
                    print('  bay %-3d %d objs -> bay %-3d  %6.2f m   %s -> %s'
                          % (bi + 1, len(groups[bi]), e + 1, d, rb.ZONE[bi], rb.ZONE[e]))
            else:
                print('  bay %-3d %d objs -> NO ROUTE' % (bi + 1, len(groups[bi])))
        if len(best) != len(groups):
            print('\n  could not route every group; nothing written')
            return

        # --- schedule: who departs when, and how long each waits at the far end ----
        sched = {bi: (dwell, 0.0, dwell) for bi in best}
        if sequence:
            byzone = defaultdict(list)
            for bi in best:
                byzone[rb.ZONE[bi]].append(bi)
            for z, members in byzone.items():
                if len(members) < 2:
                    continue
                # Order by position ALONG the direction of travel, furthest
                # first. Leading with the longest route is wrong when groups are
                # packed at one end: the leader then drives straight through the
                # follower's bay while the follower is still parked in it.
                # Whoever is nearest the destination must vacate first, and is
                # then the one still out when the follower returns -- which is
                # exactly the requested nesting.
                def _proj(b):
                    p0, p1 = best[b][1][0], best[b][1][1]
                    vx, vy = p1[0] - p0[0], p1[1] - p0[1]
                    n = math.hypot(vx, vy) or 1.0
                    return (p0[0] * vx + p0[1] * vy) / n
                members.sort(key=lambda b: -_proj(b))
                g0, g1 = members[0], members[1]
                T0 = best[g0][3] / speed
                T1 = best[g1][3] / speed
                if convoy and lag > 0:
                    # STAGGERED convoy. The follower departs `lag` after the
                    # leader. They share a corridor, so they must also turn round
                    # in the right order: on the way back the follower is AHEAD,
                    # so it must leave the far end first. Asymmetric dwell buys
                    # that while keeping both periods identical, which a single
                    # symmetric dwell provably cannot (it needs d0 > d1 + lag AND
                    # d0 == d1).
                    T = max(T0, T1)
                    far1 = dwell                       # follower turns first
                    far0 = far1 + lag + dwell          # leader waits it out
                    P = 2 * T + far0 + dwell           # leader's period
                    home0 = P - 2 * T0 - far0
                    home1 = P - 2 * T1 - far1
                    sched[g0] = (far0, 0.0, max(0.0, home0))
                    sched[g1] = (far1, lag, max(0.0, home1))
                elif convoy:
                    # Both groups sweep, travelling with a fixed lag. EQUAL
                    # periods keep their separation constant forever, so they can
                    # share a corridor safely. Strict nesting (second home first)
                    # is not possible here: it forces the follower's period to be
                    # half the leader's, and its second round trip then meets the
                    # leader coming home -- measured collision at t~47.8 s.
                    P = 2 * (max(T0, T1) + dwell)
                    d0 = P / 2 - T0
                    d1 = P / 2 - T1
                    sched[g0] = (d0, 0.0)
                    # Lag must be ZERO. Any offset makes the leader turn for home
                    # before the follower does, and its return path then runs
                    # through where the follower is still parked (measured
                    # collision at t=12.5 s with a 2 s lag). At zero lag the two
                    # groups translate identically, so their separation is
                    # constant for all time and they can share a corridor.
                    sched[g1] = (d1, 0.0)
                else:
                    gap = 1.0
                    d0 = 2 * T1 + T0 + gap
                    d1 = (T0 + d0) / 2 - T1
                    sched[g0] = (d0, 0.0, d0)
                    sched[g1] = (max(0.0, d1), T0, max(0.0, d1))
                for extra in members[2:]:
                    sched[extra] = (dwell, 0.0, dwell)
                print('  zone %s sequence: bay %d leads (T=%.1fs, dwell %.1fs), '
                      'bay %d follows at t=%.1fs (T=%.1fs, dwell %.1fs)'
                      % (z, g0 + 1, T0, sched[g0][0], g1 + 1, sched[g1][1], T1,
                         sched[g1][0]))

        # --- verify the schedule really separates them in time ----------------------
        def at(bi, t):
            """Group AABB at sim time t, replicating the plugin's phase logic."""
            e, pts, S, d = best[bi]
            dwf, ph, dwh = sched[bi]
            L = math.dist(pts[0], pts[1])
            T = L / speed
            tt = t - ph
            per = 2.0 * T + dwf + dwh
            u = tt % per if per > 0 else 0.0
            if u < T:
                f = u / T
            elif u < T + dwf:
                f = 1.0
            elif u < 2 * T + dwf:
                f = 1.0 - (u - T - dwf) / T
            else:
                f = 0.0
            cx = pts[0][0] + (pts[1][0] - pts[0][0]) * f
            cy = pts[0][1] + (pts[1][1] - pts[0][1]) * f
            gw = boxes(groups[bi])
            ox, oy = cx - (gw[0] + gw[1]) / 2, cy - (gw[2] + gw[3]) / 2
            return (gw[0] + ox, gw[1] + ox, gw[2] + oy, gw[3] + oy)

        periods = [2.0 * math.dist(best[b][1][0], best[b][1][1]) / speed
                   + sched[b][0] + sched[b][2] for b in best]
        horizon = max(periods) * 2 if periods else 0.0
        hits = 0
        for k in range(int(horizon / 0.05) + 1):
            t = k * 0.05
            bs = {bi: at(bi, t) for bi in best}
            for x, y in itertools.combinations(sorted(bs), 2):
                if ov(bs[x], bs[y]):
                    hits += 1
                    if hits <= 3:
                        print('  *** bays %d and %d overlap at t=%.2f s' % (x + 1, y + 1, t))
                    break
        print('  temporal check over %.0f s: %s'
              % (horizon, 'no two groups ever overlap' if hits == 0
                 else '%d sampled instants with an overlap' % hits))

        return best, sched, hits

    best, sched, hits = solve(bool(sequence))
    if sequence and (hits or len(best) != len(groups)):
        print('  relaxed solve failed the timeline; retrying strictly')
        best, sched, hits = solve(False)

    # rewrite: strip old plugin, insert new one, per prop
    edits = []
    for bi, (e, pts, S, d) in best.items():
        # waypoints are per-object offsets from the group's own start, so the
        # formation translates rigidly whether the route is bay-to-bay or in-bay
        D = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
        _gw = boxes(groups[bi])
        off = (pts[0][0] - (_gw[0] + _gw[1]) / 2,
               pts[0][1] - (_gw[2] + _gw[3]) / 2)
        dwf, ph, dwh = sched[bi]
        for o in groups[bi]:
            edits.append((o['span'], o['x'] + off[0], o['y'] + off[1], D, dwf, ph, dwh))
    for (span, x, y, D, dwf, ph, dwh) in sorted(edits, key=lambda t: -t[0][0]):
        a, b = span
        blk = text[a:b]
        blk = re.sub(r'\s*<plugin filename="KinematicTrajectory".*?</plugin>\n', '\n', blk, flags=re.S)
        blk = re.sub(r'\s*<static>[^<]*</static>\n', '\n', blk)
        plug = rb.kinematic_block(speed, dwell, [(x, y), (x + D[0], y + D[1])],
                                  ph, dwf, dwh)
        blk = blk.replace('    </include>\n', plug + '    </include>\n')
        text = text[:a] + blk + text[b:]

    if dry:
        print('\n(dry run -- nothing written)')
        return
    open(world, 'w').write(text)
    print('\nWrote %s -- all %d groups moving' % (os.path.basename(world), len(best)))


if __name__ == '__main__':
    main()
