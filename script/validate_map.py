#!/usr/bin/env python3
"""Validate an RTAB-Map database map against ground truth: is it the warehouse,
and could the robot have driven its true route on it?

    python3 script/validate_map.py map/map_tiny.db \
        --world aws-robomaker-small-warehouse-world/worlds/\
small_warehouse_static/small_warehouse_static_tinymap.world \
        --gt /path/to/ground_truth.csv --map-pose 1.85,8.60,-90

TWO QUESTIONS, TWO KINDS OF TRUTH
    1. GEOMETRY -- vs the .world file. Does an occupied cell correspond to real
       structure (precision), and is real structure present in the map (recall)?
       Recall is reported BOTH raw and restricted to the region the map actually
       observed; the raw number is dominated by aisles the run never entered and
       says more about route coverage than about mapping.

    2. DRIVABILITY -- vs the ground-truth trajectory. The robot demonstrably drove
       this route, so every pose on it was collision-free in reality. Stamping the
       true footprint onto the map at each pose therefore turns any collision into a
       proven FALSE OBSTACLE, and any unknown cell into a hole the planner would
       have to cross blind. This is the metric that decides whether the map is
       navigable, and it needs no assumption about what a "good" map looks like.

    The footprint is the real one from nav2_ackermann.yaml -- a rectangle, not a
    radius. Using a circumscribed radius (0.577 m) would report collisions in every
    aisle the robot legitimately fits down; using the inscribed one (0.25 m) would
    miss real ones at the corners.

WHAT THE GROUND TRUTH IS AND IS NOT
    The trajectory comes from a DIFFERENT RUN than the one that built the database
    (map_tiny.db was built against the live simulator; its node stamps match no
    recorded dataset). That is fine here and is stated in the plot: the warehouse is
    static, so a route driven through it at any time is a valid drivability probe.
    It is NOT a trajectory-error measurement, and nothing here should be read as one.

    world -> map placement is a constant that must be supplied, not solved live --
    see script/fit_map_pose.py for why, and to measure it for a new database.
"""
import argparse
import math
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_map_pose import export_db, read_pgm, read_yaml, edge_points  # noqa: E402

# nav2_ackermann.yaml: footprint about base_footprint, metres.
FOOTPRINT = np.array([[-0.31, 0.25], [0.52, 0.25], [0.52, -0.25], [-0.31, -0.25]])


def load_gt(path):
    """extract_gt.py CSV -> (N,3) of x, y, yaw. Columns t,x,y,z,qw,qx,qy,qz,..."""
    a = np.loadtxt(path, delimiter=',')
    qw, qx, qy, qz = a[:, 4], a[:, 5], a[:, 6], a[:, 7]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return np.stack([a[:, 1], a[:, 2], yaw], 1)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('map', help='RTAB-Map .db, or an exported .pgm')
    ap.add_argument('--world', required=True)
    ap.add_argument('--gt', required=True, help='CSV from script/extract_gt.py')
    ap.add_argument('--map-pose', default='1.85,8.60,-90',
                    metavar='X,Y,YAW_DEG',
                    help='world placement of the map frame; measure it with '
                         'script/fit_map_pose.py (default: the fitted value for '
                         'map_tiny.db)')
    ap.add_argument('--tol', type=float, default=0.25,
                    help='metres a map cell may sit from real structure and still '
                         'count as correct (default 0.25)')
    ap.add_argument('--out', default='output/compare/rtab_map_vs_gt.png')
    a = ap.parse_args()

    from scipy.ndimage import distance_transform_edt, binary_dilation
    from matplotlib.path import Path as MplPath
    import viewer

    tmp = tempfile.mkdtemp(prefix='valmap.')
    try:
        pgm = export_db(a.map, tmp) if a.map.endswith('.db') else a.map
        res, (ox, oy) = read_yaml(pgm.replace('.pgm', '.yaml'))
        img = read_pgm(pgm)
        h = img.shape[0]

        mx, my, mdeg = (float(v) for v in a.map_pose.split(','))
        th = math.radians(mdeg)
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s], [s, c]])

        def to_world(mask):
            r, cc = np.nonzero(mask)
            p = np.stack([ox + (cc + 0.5) * res, oy + (h - 1 - r + 0.5) * res], 1)
            return p @ R.T + [mx, my]

        occ_w = to_world(img == 0)                     # 0 = occupied
        known_w = to_world(img != 205)                 # 205 = unknown

        # ---- common world raster ---------------------------------------------
        obst, wall = viewer.load_obstacles(a.world)
        gt = load_gt(a.gt)
        polys = [np.asarray(p, float) for _, p, _ in obst]
        x0, y0, x1, y1 = wall
        wall_ring = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

        pad = 1.0
        lo = np.array([min(x0, occ_w[:, 0].min(), gt[:, 0].min()) - pad,
                       min(y0, occ_w[:, 1].min(), gt[:, 1].min()) - pad])
        hi = np.array([max(x1, occ_w[:, 0].max(), gt[:, 0].max()) + pad,
                       max(y1, occ_w[:, 1].max(), gt[:, 1].max()) + pad])
        gw, gh = (np.ceil((hi - lo) / res).astype(int) + 1)

        def rast(pts):
            idx = np.round((pts - lo) / res).astype(int)
            ok = ((idx[:, 0] >= 0) & (idx[:, 0] < gw)
                  & (idx[:, 1] >= 0) & (idx[:, 1] < gh))
            g = np.zeros((gh, gw), bool)
            g[idx[ok, 1], idx[ok, 0]] = True
            return g

        M_occ, M_known = rast(occ_w), rast(known_w)

        # true structure: shelf footprints FILLED (a cell inside a shelf is a real
        # obstacle, not a false positive) plus the wall as a ring
        yy, xx = np.mgrid[0:gh, 0:gw]
        cell_xy = np.stack([lo[0] + xx.ravel() * res, lo[1] + yy.ravel() * res], 1)
        T_occ = np.zeros(gh * gw, bool)
        for q in polys:
            T_occ |= MplPath(q).contains_points(cell_xy)
        T_occ = T_occ.reshape(gh, gw) | rast(edge_points(wall_ring, res * 0.5))
        # Recall is measured against SURFACES, not filled interiors: a stereo
        # camera can only ever see a shelf's outside, so counting its interior as
        # missed geometry would cap recall well below 100 % for a perfect map.
        T_edge = T_occ & binary_dilation(~T_occ, np.ones((3, 3)))

        # ---- 1. geometry ------------------------------------------------------
        d_to_true = distance_transform_edt(~T_occ, sampling=res)
        d_to_map = distance_transform_edt(~M_occ, sampling=res)
        prec = float((d_to_true[M_occ] <= a.tol).mean())
        rec_raw = float((d_to_map[T_edge] <= a.tol).mean())
        seen = T_edge & binary_dilation(M_known, np.ones((3, 3)))
        rec_seen = float((d_to_map[seen] <= a.tol).mean())

        # ---- 2. drivability ---------------------------------------------------
        # footprint interior as body-frame offsets, one sample per cell
        fx = np.arange(FOOTPRINT[:, 0].min(), FOOTPRINT[:, 0].max() + res, res)
        fy = np.arange(FOOTPRINT[:, 1].min(), FOOTPRINT[:, 1].max() + res, res)
        fgx, fgy = np.meshgrid(fx, fy)
        foot = np.stack([fgx.ravel(), fgy.ravel()], 1)
        foot = foot[MplPath(FOOTPRINT).contains_points(foot)]

        cy, sy = np.cos(gt[:, 2]), np.sin(gt[:, 2])
        # (N, F, 2) world points swept by the true footprint along the true route
        wxy = np.empty((len(gt), len(foot), 2))
        wxy[:, :, 0] = gt[:, None, 0] + np.outer(cy, foot[:, 0]) - np.outer(sy, foot[:, 1])
        wxy[:, :, 1] = gt[:, None, 1] + np.outer(sy, foot[:, 0]) + np.outer(cy, foot[:, 1])
        ij = np.round((wxy - lo) / res).astype(int)
        np.clip(ij[:, :, 0], 0, gw - 1, out=ij[:, :, 0])
        np.clip(ij[:, :, 1], 0, gh - 1, out=ij[:, :, 1])
        hit_occ = M_occ[ij[:, :, 1], ij[:, :, 0]].any(1)
        frac_known = M_known[ij[:, :, 1], ij[:, :, 0]].mean(1)
        clear = d_to_map[np.round((gt[:, 1] - lo[1]) / res).astype(int).clip(0, gh - 1),
                         np.round((gt[:, 0] - lo[0]) / res).astype(int).clip(0, gw - 1)]

        blocked = float(hit_occ.mean())
        unmapped = float((frac_known < 0.5).mean())
        drivable = float((~hit_occ & (frac_known >= 0.5)).mean())

        print('MAP GEOMETRY vs world file   (tolerance %.2f m)' % a.tol)
        print('  precision            %.1f %%   (%d occupied cells)'
              % (100 * prec, M_occ.sum()))
        print('  recall, whole building %.1f %%' % (100 * rec_raw))
        print('  recall, region mapped  %.1f %%' % (100 * rec_seen))
        print('DRIVABILITY vs ground-truth route   (%d poses, real footprint)'
              % len(gt))
        print('  clear and mapped     %.1f %%' % (100 * drivable))
        print('  FALSE OBSTACLE       %.1f %%   <- footprint hits an occupied cell'
              % (100 * blocked))
        print('  unmapped (>50%% unknown under footprint)  %.1f %%'
              % (100 * unmapped))
        print('  clearance to nearest mapped obstacle: median %.2f m, p05 %.2f m'
              % (np.median(clear), np.percentile(clear, 5)))

        # ---- plot -------------------------------------------------------------
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        fig = plt.figure(figsize=(16, 11))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1])
        ax = fig.add_subplot(gs[:, 0])
        for q in polys:
            ax.fill(q[:, 0], q[:, 1], color='0.80', ec='0.45', lw=0.7, zorder=1)
        ax.plot(wall_ring[[0, 1, 2, 3, 0], 0], wall_ring[[0, 1, 2, 3, 0], 1],
                'k-', lw=1.6, zorder=1)
        ax.scatter(occ_w[:, 0], occ_w[:, 1], s=1.1, c='crimson', alpha=0.5, zorder=2)
        ok = ~hit_occ & (frac_known >= 0.5)
        ax.plot(gt[ok, 0], gt[ok, 1], '.', ms=2.0, color='#0b7', zorder=3)
        ax.plot(gt[frac_known < 0.5, 0], gt[frac_known < 0.5, 1], '.', ms=2.0,
                color='#f90', zorder=4)
        ax.plot(gt[hit_occ, 0], gt[hit_occ, 1], '.', ms=3.5, color='#c0f', zorder=5)
        ax.plot(gt[0, 0], gt[0, 1], 'b*', ms=17, zorder=6)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.set_xlabel('world x [m]')
        ax.set_ylabel('world y [m]')
        ax.set_title('%s placed at %s\ngrey = true warehouse, red = map cells'
                     % (os.path.basename(a.map), a.map_pose))
        ax.legend(handles=[
            Line2D([], [], ls='', marker='o', ms=6, color='crimson',
                   label='map occupied cell'),
            Line2D([], [], ls='', marker='o', ms=6, color='#0b7',
                   label='GT pose: clear and mapped (%.1f %%)' % (100 * drivable)),
            Line2D([], [], ls='', marker='o', ms=6, color='#f90',
                   label='GT pose: unmapped (%.1f %%)' % (100 * unmapped)),
            Line2D([], [], ls='', marker='o', ms=6, color='#c0f',
                   label='GT pose: FALSE OBSTACLE (%.1f %%)' % (100 * blocked)),
            Line2D([], [], ls='', marker='*', ms=11, color='b', label='spawn'),
        ], loc='upper left', fontsize=9, framealpha=0.93)

        ax2 = fig.add_subplot(gs[0, 1])
        e = d_to_true[M_occ]
        ax2.hist(e, bins=np.arange(0, 2.0 + 0.025, 0.025), color='crimson', alpha=0.8)
        ax2.axvline(a.tol, color='k', ls='--', lw=1.2,
                    label='tolerance %.2f m' % a.tol)
        ax2.set_yscale('log')
        ax2.set_xlabel('distance from a map cell to real structure [m]')
        ax2.set_ylabel('cells (log)')
        ax2.set_title('Geometry: precision %.1f %%   recall %.1f %% of the region '
                      'mapped\n(%.1f %% of the whole building)'
                      % (100 * prec, 100 * rec_seen, 100 * rec_raw))
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(clear, bins=np.arange(0, 3.0 + 0.05, 0.05), color='#0b7', alpha=0.85)
        ax3.axvline(0.25, color='#c0f', ls='--', lw=1.5,
                    label='footprint half-width 0.25 m')
        ax3.axvline(0.80, color='k', ls=':', lw=1.5, label='nav2 inflation 0.80 m')
        ax3.set_xlabel('clearance from the true route to the nearest mapped obstacle [m]')
        ax3.set_ylabel('GT poses')
        ax3.set_title('Drivability: the robot really drove here, so every collision '
                      'below\nis a false obstacle. median clearance %.2f m'
                      % np.median(clear))
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)

        fig.suptitle('RTAB-Map database vs ground truth  --  %s  vs  %s'
                     % (os.path.basename(a.map), os.path.basename(a.gt)),
                     fontsize=13)
        fig.text(0.5, 0.005,
                 'Ground truth is a DIFFERENT run through the same static '
                 'warehouse, so this validates geometry and drivability, not '
                 'trajectory error.', ha='center', fontsize=9, color='0.35')
        plt.tight_layout(rect=(0, 0.015, 1, 0.975))
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        plt.savefig(a.out, dpi=110)
        print('\nwrote ' + a.out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
