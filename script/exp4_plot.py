#!/usr/bin/env python3
"""Plot an Experiment 4 run twice: on the world geometry, and on the prior map.

    python3 script/exp4_plot.py --all
    python3 script/exp4_plot.py --run U_static_P3_circuit_1
    python3 script/exp4_plot.py --all --out docs/report/insights/experiment_4/fig

WHY TWO PANELS
    The two panels answer different questions about the same drive.

    LEFT, on the world: is the estimate where the robot really was? The
    footprints are viewer.load_obstacles -- the convex hull of every collision
    mesh sliced at robot height, the geometry the occupancy bake and the chapter 3
    scene figures use -- so this is the same underlay the rest of the report draws
    trajectories on, and a gap between the two tracks is error against truth.

    RIGHT, on the prior: is the estimate consistent with the map it is localising
    against? The grid comes from rtabmap-export of the database itself. An
    estimate can sit squarely in the map's aisle while both sit half a metre off
    the real one, which is exactly what Experiment 4 measured: an offset of
    0.32-0.59 m on a single-aisle route against 0.02-0.14 m on a route covering
    the building, matching Experiment 3's finding that the realistic maps displace
    their walls by about 0.5 m. One panel cannot show that; two can.

FRAME
    Everything is drawn in the GAZEBO WORLD frame, because that is the frame the
    footprints and the ground truth are already in. The map-frame things -- the
    prior's grid and the published pose -- are brought across with the per-layout
    fit from exp4_goals._align (yaw ~180 deg, residual 0.10 / 0.055 m), and the
    0.338 m lever arm is applied to the ground truth so that it refers to the
    camera/IMU body like every estimate does.
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np
import yaml

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import matplotlib                                              # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.patches import Polygon, Rectangle              # noqa: E402

WORKSPACE = os.path.expanduser('~/wil_project')
sys.path.insert(0, os.path.join(WORKSPACE, 'script'))
import exp4_goals as eg                                        # noqa: E402
import exp4_score as es                                        # noqa: E402
import fit_map_pose as fmp                                     # noqa: E402
import viewer                                                  # noqa: E402

FILL = '#2f3542'          # map_underlay.FILL, so the underlay matches the report
GT_C = '#1b9e77'
EST_C = '#d95f02'
GRID_C = '#34495e'


def world_footprints(world_path):
    """(polygons, wall bounds) in the world frame, from the collision meshes.

    viewer.load_obstacles returns (obstacles, wall): a list of
    (label, Nx2 polygon, authored pose) and the wall as (x0, y0, x1, y1) bounds,
    not a polygon. Drawn exactly as map_underlay.draw_map draws them, so this
    figure's underlay is the same geometry at the same alpha as the chapter 3
    scene figures and every other trajectory plot in the report.
    """
    obstacles, wall = viewer.load_obstacles(world_path)
    polys = [np.asarray(poly, float) for _label, poly, _pose in obstacles
             if len(poly) >= 3]
    return polys, wall


def prior_grid(db):
    """Occupied cell centres of a database's optimised grid, in the MAP frame.

    rtabmap-export writes the pgm/yaml next to the database with no way to
    redirect it, so fit_map_pose.export_db moves the pair into a temp directory
    rather than leaving litter in map/.
    """
    with tempfile.TemporaryDirectory() as tmp:
        try:
            pgm = fmp.export_db(db, tmp)
        except (subprocess.CalledProcessError, SystemExit) as e:
            print(f'  [warn] rtabmap-export failed ({e}); the right panel will '
                  f'show the paths only')
            return None
        img = fmp.read_pgm(pgm)
        res, origin = fmp.read_yaml(os.path.splitext(pgm)[0] + '.yaml')
    # PGM convention: 0 is occupied, 254 free, 205 unknown; row 0 is the TOP, so
    # the y axis is flipped relative to the grid origin.
    occ = np.argwhere(img == 0)
    if not len(occ):
        return None
    h = img.shape[0]
    x = origin[0] + (occ[:, 1] + 0.5) * res
    y = origin[1] + (h - occ[:, 0] - 0.5) * res
    return np.column_stack([x, y])


def run_tracks(spec, world, path, rep):
    """Ground truth and the published pose for one run, both in the world frame."""
    lay = spec['worlds'][world]['layout']
    prior = spec['priors'][lay]['database']
    d = os.path.join(WORKSPACE, 'map', prior, 'exp4', f'{world}_{path}_{rep}')
    rob = os.path.join(d, 'robot_gt.csv')
    est_csv = os.path.join(WORKSPACE, 'map', prior,
                           f'localize_exp4_{world}_{path}_{rep}.csv')
    if not os.path.exists(rob):
        return None
    t, gt = es.load_robot_gt(rob)              # lever arm applied, world frame
    al = eg._align(prior)
    est = est_t = first_fix = None
    if os.path.exists(est_csv):
        rows = [r for r in csv.DictReader(open(est_csv))
                if r['kind'] == 'tf' and r.get('x')]
        if rows:
            em = np.array([[float(r['x']), float(r['y'])] for r in rows])
            est = es.map_to_world(al, em)      # map -> world for drawing
            est_t = np.array([float(r['stamp']) for r in rows])
            # Before the first informative pose the published transform is the
            # seed composed with raw odometry, so it can sit far outside the
            # building and sweep in when a fix lands. Drawing that as one line
            # with the tracked part makes an acquisition transient look like a
            # localisation failure, so it is drawn faded instead.
            inf = [float(r['stamp']) for r in csv.DictReader(open(est_csv))
                   if r['kind'] == 'pose' and r.get('cov_trace')
                   and float(r['cov_trace']) < 1e4]
            first_fix = inf[0] if inf else None
    return {'dir': d, 'prior': prior, 'layout': lay, 'al': al,
            'est_t': est_t if est is not None else None,
            'first_fix': first_fix if est is not None else None,
            'world_file': os.path.join(eg.WORLD_DIR, spec['worlds'][world]['file']),
            'gt': gt, 'est': est, 't': t,
            'goals': es.map_to_world(al, [g['map_xy']
                                          for g in spec['priors'][lay]['paths'][path]])}


def plot_run(spec, world, path, rep, out_dir, grid_cache):
    r = run_tracks(spec, world, path, rep)
    if r is None:
        return None
    polys, wall = world_footprints(r['world_file'])
    if r['prior'] not in grid_cache:
        g = prior_grid(os.path.join(WORKSPACE, 'map', r['prior'], f'{r["prior"]}.db'))
        grid_cache[r['prior']] = g
    grid = grid_cache[r['prior']]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6))
    for ax, which in zip(axes, ('world', 'prior')):
        if which == 'world':
            for p in polys:
                ax.add_patch(Polygon(p, closed=True, facecolor=FILL, alpha=0.5,
                                     edgecolor='none', zorder=0.5))
            if wall is not None:
                x0, y0, x1, y1 = wall
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       edgecolor=FILL, lw=2.0, alpha=0.5,
                                       zorder=0.6))
            ax.set_title('on the world geometry (collision meshes)')
        else:
            if grid is not None:
                gw = es.map_to_world(r['al'], grid)
                ax.scatter(gw[:, 0], gw[:, 1], s=0.7, c=GRID_C, alpha=0.35,
                           linewidths=0, zorder=0.5)
                ax.set_title(f'on the prior map ({r["prior"]}, '
                             f'{len(grid)} occupied cells)')
            else:
                ax.set_title('on the prior map (export unavailable)')
        ax.plot(r['gt'][:, 0], r['gt'][:, 1], color=GT_C, lw=2.0,
                label='ground truth (IMU body)', zorder=3)
        if r['est'] is not None:
            e, et, ff = r['est'], r['est_t'], r['first_fix']
            if ff is not None:
                pre = et <= ff
                if pre.sum() > 1:
                    ax.plot(e[pre, 0], e[pre, 1], color=EST_C, lw=1.0, ls='--',
                            alpha=0.45, zorder=3.05,
                            label='pose before the first fix (seed + odometry)')
                post = et >= ff
                ax.plot(e[post, 0], e[post, 1], color=EST_C, lw=1.4,
                        label='published pose (map to base_link)', zorder=3.1)
            else:
                ax.plot(e[:, 0], e[:, 1], color=EST_C, lw=1.4,
                        label='published pose (map to base_link)', zorder=3.1)
        ax.plot(*r['gt'][0], 'o', ms=7, mfc='white', mec=GT_C, mew=2,
                zorder=4, label='spawn')
        for gx, gy in r['goals']:
            ax.plot(gx, gy, 'x', ms=8, color='#7f8c8d', mew=2, zorder=3.5)
        ax.set_aspect('equal')
        ax.set_xlabel('world x (m)')
        ax.grid(alpha=0.15, lw=0.5)
    axes[0].set_ylabel('world y (m)')
    axes[0].legend(loc='upper left', fontsize=8, framealpha=0.9)

    # one shared extent, so the two panels can be compared cell for cell
    allpts = [r['gt']] + ([r['est']] if r['est'] is not None else [])
    P = np.vstack(allpts)
    pad = 3.0
    for ax in axes:
        ax.set_xlim(P[:, 0].min() - pad, P[:, 0].max() + pad)
        ax.set_ylim(P[:, 1].min() - pad, P[:, 1].max() + pad)

    name = f'{world}_{path}_{rep}'
    fig.suptitle(f'Experiment 4: {name}', y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f'exp4_path_{name}.png')
    fig.savefig(png, dpi=150)
    plt.close(fig)
    # the number the panels are there to show
    err = None
    if r['est'] is not None:
        gm = es.world_to_map(r['al'], r['gt'])
        em = es.world_to_map(r['al'], r['est'])
        # nearest-sample error, enough for a caption; exp4_score does it properly
        d = np.hypot(*(em[:, None, :] - gm[None, :, :]).transpose(2, 0, 1))
        err = float(np.median(d.min(axis=1)))
    print(f'  {name:30} -> {png}'
          + (f'   median closest-approach error {err:.3f} m' if err else ''))
    return png


def summary(spec, out_dir):
    """Error against distance driven, every run on one axis.

    The per-run path figures show WHERE the pose went; this shows whether the
    error grows. That is the claim the slot rests on -- localising against a
    prior bounds the error, where dead reckoning would not -- and a half-by-half
    median in a table states it far less plainly than a flat line over 150 m.

    x is distance driven rather than time so that runs at different speeds are
    comparable, and y is logarithmic because the one diverged run is two orders
    of magnitude above the rest and would otherwise flatten them all.
    """
    fig, ax = plt.subplots(figsize=(10, 5.6))
    cmap = plt.get_cmap('tab10')
    rows = []
    for i, world in enumerate(spec['worlds']):
        lay = spec['worlds'][world]['layout']
        prior = spec['priors'][lay]['database']
        al = eg._align(prior)
        for path in spec['priors'][lay]['paths']:
            for rep in (1, 2, 3):
                rob = os.path.join(WORKSPACE, 'map', prior, 'exp4',
                                   f'{world}_{path}_{rep}', 'robot_gt.csv')
                est = os.path.join(WORKSPACE, 'map', prior,
                                   f'localize_exp4_{world}_{path}_{rep}.csv')
                if not (os.path.exists(rob) and os.path.exists(est)):
                    continue
                t, gt = es.load_robot_gt(rob)
                gm = es.world_to_map(al, gt)
                rec = list(csv.DictReader(open(est)))
                ff = next((float(r['stamp']) for r in rec
                           if r['kind'] == 'pose' and r.get('cov_trace')
                           and float(r['cov_trace']) < 1e4), None)
                E = [r for r in rec if r['kind'] == 'tf' and r.get('x')
                     and (ff is None or float(r['stamp']) >= ff)]
                if len(E) < 5:
                    continue
                et = np.array([float(r['stamp']) for r in E])
                ex = np.array([float(r['x']) for r in E])
                ey = np.array([float(r['y']) for r in E])
                m = (et >= t[0]) & (et <= t[-1])
                err = np.hypot(ex[m] - np.interp(et[m], t, gm[:, 0]),
                               ey[m] - np.interp(et[m], t, gm[:, 1]))
                step = np.hypot(*np.diff(gm, axis=0).T)
                step[step > 1.0] = 0.0
                dist = np.interp(et[m], t, np.r_[0, np.cumsum(step)])
                shape = {'P1_transit': 'straight', 'P2_return': 'square',
                         'P3_circuit': 'figure-8'}.get(path, path)
                lab = f'{world.replace("_", " ")} {shape}'
                # Label by how LONG the error stayed up, not by its peak. A peak
                # threshold called the U figure-of-eight 'lost the pose' when it
                # was above 2 m for 23 s and then recovered to 0.12 m, and said
                # nothing about the square being up for 36 s. Both recovered;
                # neither was lost.
                tt = et[m]
                up = err > 1.0
                longest, start = 0.0, None
                for j, b in enumerate(up):
                    if b and start is None:
                        start = tt[j]
                    elif not b and start is not None:
                        longest = max(longest, tt[j] - start)
                        start = None
                if start is not None:
                    longest = max(longest, tt[-1] - start)
                bad = longest > 10.0
                ax.plot(dist, err, lw=2.2 if bad else 1.3,
                        color=cmap(len(rows) % 10), alpha=0.95 if bad else 0.8,
                        label=lab + (f' (over 1 m for {longest:.0f} s)' if bad else ''))
                rows.append(lab)
    ax.set_yscale('log')
    ax.axhline(0.40, color='#7f8c8d', ls='--', lw=1,
               label="prior graph's own rms, upper end (Exp 3)")
    ax.axhline(0.10, color='#7f8c8d', ls=':', lw=1,
               label="prior graph's own rms, lower end")
    ax.set_xlabel('distance driven since the first fix (m)')
    ax.set_ylabel('planar error against ground truth (m)')
    ax.set_title('Experiment 4: error against distance, every run')
    ax.grid(alpha=0.2, which='both', lw=0.5)
    # Below a centimetre the comparison is noise in the interpolation, and a log
    # axis gives those samples a whole decade of the picture.
    ax.set_ylim(0.01, None)
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5),
              framealpha=0.95)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, 'exp4_error_vs_distance.png')
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f'  {len(rows)} run(s) -> {png}')
    return png


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', metavar='WORLD_PATH_REP')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--summary', action='store_true',
                    help='one axis, every run: error against distance driven')
    ap.add_argument('--out', default=os.path.join(WORKSPACE, 'output', 'exp4_plots'))
    a = ap.parse_args()
    spec = yaml.safe_load(open(os.path.join(WORKSPACE, 'script',
                                            'exp4_protocol.yaml')))
    if a.summary:
        summary(spec, a.out)
        if not a.all and not a.run:
            return
    cache = {}
    done = 0
    for world in spec['worlds']:
        for path in spec['priors'][spec['worlds'][world]['layout']]['paths']:
            for rep in (1, 2, 3):
                name = f'{world}_{path}_{rep}'
                if a.run and a.run != name:
                    continue
                if plot_run(spec, world, path, rep, a.out, cache):
                    done += 1
    if not done:
        sys.exit('no runs plotted; --run takes WORLD_PATH_REP, e.g. '
                 'U_static_P3_circuit_1')
    print(f'\n{done} figure(s) in {a.out}')


if __name__ == '__main__':
    main()
