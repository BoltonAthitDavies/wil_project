#!/usr/bin/env python3
"""Measure where an RTAB-Map database sits in the Gazebo world: the --map-pose
number for viewer.py, fitted to the warehouse instead of guessed.

    python3 script/fit_map_pose.py map/map_tiny.db \
        --world aws-robomaker-small-warehouse-world/worlds/\
small_warehouse_static/small_warehouse_static_tinymap.world --overlay /tmp/fit.png

    -> --map-pose 1.850,8.600,-90.00

THE PROBLEM IT SOLVES
    The 2D viewer draws in GAZEBO WORLD coordinates and nothing else: obstacles and
    walls come from the .world file, ground truth arrives in world coordinates, and
    estimator trails are pushed through a rigid alignment SOLVED AGAINST GROUND
    TRUTH (viewer.py on_est) so they land there too. View.to_screen is a plain
    world->pixel affine with no concept of a frame at all.

    An RTAB-Map occupancy grid is the one thing on screen that is not in that frame.
    It arrives on /map in RTAB-Map's `map` frame, which is anchored wherever the
    front-end initialised. So it draws offset and rotated, and --map-pose is the
    rigid transform that puts it back.

    RViz never has this problem because it does the opposite: its fixed frame is
    `map`, so it moves the ROBOT into the map frame via TF rather than moving the
    map into the world. It therefore needs no equivalent of this number.

WHY A CONSTANT IS THE RIGHT ANSWER
    world->map is genuinely constant -- both are static frames. What moves is
    odom: map->odom is the correction RTAB-Map applies as the front-end drifts.
    So the placement never needs to be live, and composing map->odom with a
    front-end alignment (which is solved once, at the start, and then goes stale
    as odom drifts) actively makes it worse over time. Measure it once per
    database and hard-code it.

HOW IT IS MEASURED
    Brute force over yaw; for each yaw the best translation comes from a single FFT
    cross-correlation of the grid's occupied cells against a blurred score field
    built from the world file's obstacle and wall edges. No odometry, no ground
    truth -- the question asked is only "where do these occupied cells have to sit
    for them to be this warehouse".

    Report the margin. A warehouse of parallel aisles is very nearly rotationally
    ambiguous, so a winner that barely beats a rival 90 or 180 degrees away is not
    a measurement, it is a coin toss. Anything under ~20% deserves the --overlay.

SANITY CHECK, FREE
    The answer should land near the spawn pose of the front-end's BODY frame -- not
    of base_footprint. VINS and ORB-SLAM3 both report the IMU body, which sits
    0.338 m ahead of base_footprint, so a map built from the stock spawn
    (1.8, 9.0, -90 deg) should fit at (1.800, 8.662, -90). That 0.338 m is exactly
    the residual you get from using the spawn pose as the guess. The script prints
    the comparison; a fit far from it means the run did not start at the spawn, or
    the fit is wrong.
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BODY_AHEAD_OF_BASE = 0.338        # metres; see localization_rtabmap.launch.py


def read_pgm(path):
    """P5 loader. Not using cv2/PIL: this is a dependency-free 20 lines, and the
    only tricky part (comments between the header tokens) is handled."""
    with open(path, 'rb') as f:
        data = f.read()
    if data[:2] != b'P5':
        raise SystemExit('%s is not a binary PGM' % path)
    tok, i = [], 2
    while len(tok) < 3:
        while data[i:i + 1].isspace():
            i += 1
        if data[i:i + 1] == b'#':
            while data[i:i + 1] != b'\n':
                i += 1
            continue
        j = i
        while not data[j:j + 1].isspace():
            j += 1
        tok.append(int(data[i:j]))
        i = j
    w, h, _ = tok
    return np.frombuffer(data[i + 1:i + 1 + w * h], np.uint8).reshape(h, w)


def read_yaml(path):
    res, origin = None, None
    for line in open(path):
        if line.startswith('resolution:'):
            res = float(line.split(':')[1])
        elif line.startswith('origin:'):
            origin = [float(v) for v in
                      line.split('[')[1].split(']')[0].split(',')][:2]
    if res is None or origin is None:
        raise SystemExit('no resolution/origin in ' + path)
    return res, origin


def export_db(db, tmp):
    """rtabmap-export writes <name>.pgm/.yaml NEXT TO THE DATABASE, with no way to
    redirect it. Move the pair out rather than leaving litter in map/."""
    stem = os.path.splitext(os.path.basename(db))[0]
    beside = os.path.join(os.path.dirname(os.path.abspath(db)), stem)
    for ext in ('.pgm', '.yaml'):
        if os.path.exists(beside + ext):
            raise SystemExit('%s%s already exists; move it aside first' % (beside, ext))
    subprocess.run(['rtabmap-export', '--map', '--opt', '2', db],
                   check=True, stdout=subprocess.DEVNULL)
    out = os.path.join(tmp, stem + '.pgm')
    for ext in ('.pgm', '.yaml'):
        shutil.move(beside + ext, os.path.join(tmp, stem + ext))
    return out


def edge_points(poly, step):
    out = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        n = max(2, int(np.hypot(*(b - a)) / step) + 1)
        out.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('map', help='an RTAB-Map .db, or an already-exported .pgm')
    ap.add_argument('--world', required=True, help='the .world the map was built in')
    ap.add_argument('--sigma', type=float, default=0.20,
                    help='match tolerance in metres: how far a map cell may sit '
                         'from real geometry and still count (default 0.20)')
    ap.add_argument('--pad', type=float, default=8.0,
                    help='metres of search slack around the warehouse')
    ap.add_argument('--step', type=float, default=0.5,
                    help='coarse yaw step in degrees; refined to 0.1 around the winner')
    ap.add_argument('--spawn', default='1.8,9.0,-90',
                    help='X,Y,YAW_DEG the run started from, for the sanity check')
    ap.add_argument('--overlay', default='',
                    help='write a PNG of the fitted grid over the warehouse')
    a = ap.parse_args()

    from scipy.ndimage import gaussian_filter
    import viewer                                   # for load_obstacles

    tmp = tempfile.mkdtemp(prefix='fitmap.')
    try:
        pgm = export_db(a.map, tmp) if a.map.endswith('.db') else a.map
        res, (ox, oy) = read_yaml(pgm.replace('.pgm', '.yaml'))
        img = read_pgm(pgm)

        # ---- the map's occupied cells, in map-frame metres --------------------
        h = img.shape[0]
        rows, cols = np.nonzero(img == 0)            # 0 = occupied in a ROS map pgm
        if len(rows) < 50:
            raise SystemExit('only %d occupied cells; nothing to fit' % len(rows))
        P = np.stack([ox + (cols + 0.5) * res,
                      oy + (h - 1 - rows + 0.5) * res], 1)

        # ---- the warehouse, in world-frame metres ----------------------------
        obst, wall = viewer.load_obstacles(a.world)
        geo = [edge_points(np.asarray(p, float), res) for _, p, _ in obst]
        if wall is not None:                         # (x0, y0, x1, y1) bbox -> ring
            x0, y0, x1, y1 = wall
            geo.append(edge_points(
                np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float), res))
        Q = np.vstack(geo)
        print('%d occupied cells vs %d world edge samples' % (len(P), len(Q)))

        gx0, gy0 = Q[:, 0].min() - a.pad, Q[:, 1].min() - a.pad
        gw = int((Q[:, 0].max() + a.pad - gx0) / res) + 1
        gh = int((Q[:, 1].max() + a.pad - gy0) / res) + 1

        def rasterise(pts):
            c = np.round((pts[:, 0] - gx0) / res).astype(int)
            r = np.round((pts[:, 1] - gy0) / res).astype(int)
            ok = (c >= 0) & (c < gw) & (r >= 0) & (r < gh)
            out = np.zeros((gh, gw), np.float32)
            np.add.at(out, (r[ok], c[ok]), 1.0)
            return out, int(ok.sum())

        S, _ = rasterise(Q)
        S = gaussian_filter((S > 0).astype(np.float32), a.sigma / res)
        S /= S.max()
        FS = np.fft.rfft2(S)

        def sweep(degrees):
            out = []
            for deg in degrees:
                th = math.radians(deg)
                c, s = math.cos(th), math.sin(th)
                M, n_in = rasterise(P @ np.array([[c, -s], [s, c]]).T)
                if n_in < len(P) * 0.5:              # rotated mostly off the grid
                    continue
                corr = np.fft.irfft2(FS * np.conj(np.fft.rfft2(M)), s=S.shape)
                k = int(np.argmax(corr))
                r, cc = divmod(k, gw)
                # correlation is circular: unwrap the shift to a signed offset
                dx = cc if cc <= gw // 2 else cc - gw
                dy = r if r <= gh // 2 else r - gh
                out.append((corr.flat[k] / len(P), float(deg), dx * res, dy * res))
            return sorted(out, reverse=True)

        cand = sweep(np.arange(-180.0, 180.0, a.step))
        if not cand:
            raise SystemExit('no usable yaw; is --pad large enough?')
        alts = [c for c in cand
                if abs((c[1] - cand[0][1] + 180) % 360 - 180) > 5.0]
        margin = 100.0 * (cand[0][0] / alts[0][0] - 1.0) if alts else float('inf')

        score, deg, tx, ty = sweep(np.arange(cand[0][1] - 1.0,
                                             cand[0][1] + 1.0, 0.1))[0]

        print('\n  --map-pose %.3f,%.3f,%.2f' % (tx, ty, deg))
        print('  agreement %.3f   margin over best rival >5deg away %+.1f%%%s'
              % (score, margin, '' if margin > 20 else '   <-- WEAK, check --overlay'))

        sx, sy, sdeg = (float(v) for v in a.spawn.split(','))
        bx = sx + BODY_AHEAD_OF_BASE * math.cos(math.radians(sdeg))
        by = sy + BODY_AHEAD_OF_BASE * math.sin(math.radians(sdeg))
        print('  spawn of the IMU body frame (%.3f, %.3f, %.1f): fit is %.3f m, '
              '%.2f deg away' % (bx, by, sdeg, math.hypot(tx - bx, ty - by),
                                 (deg - sdeg + 180) % 360 - 180))
        print('  naive guess, the base_footprint spawn (%.3f, %.3f): %.3f m away'
              % (sx, sy, math.hypot(tx - sx, ty - sy)))

        if a.overlay:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            th = math.radians(deg)
            c, s = math.cos(th), math.sin(th)
            V = P @ np.array([[c, -s], [s, c]]).T + [tx, ty]
            fig, ax = plt.subplots(figsize=(9, 12))
            for _, poly, _ in obst:
                q = np.asarray(poly, float)
                ax.fill(q[:, 0], q[:, 1], color='0.78', ec='0.35', lw=0.6)
            if wall is not None:
                x0, y0, x1, y1 = wall
                ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], 'k-', lw=1.5)
            ax.scatter(V[:, 0], V[:, 1], s=1.2, c='crimson', alpha=0.55)
            ax.plot(sx, sy, 'b*', ms=15)
            ax.set_aspect('equal')
            ax.grid(alpha=0.3)
            ax.set_title('--map-pose %.3f,%.3f,%.2f   agreement %.3f'
                         % (tx, ty, deg, score))
            plt.tight_layout()
            plt.savefig(a.overlay, dpi=95)
            print('  wrote ' + a.overlay)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
