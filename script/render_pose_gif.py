#!/usr/bin/env python3
"""Animate a pose track over the map it is scored against: three modes.

    # 1. ground truth alone, on the Gazebo world plan           (Experiment 1)
    python3 script/render_pose_gif.py --mode gt \\
        --dataset dataset_allsensor_realistic_001 --out gif/exp1_gt_001.gif

    # 2. ground truth + a VSLAM estimate, on the world plan      (Experiment 1)
    python3 script/render_pose_gif.py --mode gt+est \\
        --dataset dataset_allsensor_realistic_001 --system all \\
        --out gif/exp1_gt_est_001.gif

    # 3. ground truth + the published pose, on the dense map     (Experiment 4)
    python3 script/render_pose_gif.py --mode nav \\
        --run U_static_P3_circuit_1 --out gif/exp4_nav_U_P3.gif

    # size knobs, all modes:  --fps 5 --height 420 --start 30 --end 90
    # an .mp4 --out writes H.264 instead of a GIF, at a fraction of the size

NOTHING HERE DERIVES A NEW TRANSFORM
    Every frame, lever arm and alignment is taken from the script that already
    owns it, so an animation cannot disagree with the number the report quotes:

      mode gt       vio_metrics.load_gt(lever=False) -- the MODEL ORIGIN, which is
                    where the robot actually is. This mode draws no estimate, so
                    there is nothing to move the truth towards.
      mode gt+est   vio_metrics.analyse() -- the same resampled, rigidly aligned
                    (SE(3), scale fixed) series the reported ATE is computed from,
                    with the ground truth already moved to the IMU point. The gap
                    you see between the two tracks IS the error the table reports.
      mode nav      exp4_plot.run_tracks() -- ground truth through
                    exp4_score.load_robot_gt (lever arm applied) and the published
                    pose through exp4_goals._align, map frame to world frame. The
                    estimate is NOT re-aligned here: Experiment 4's finding is a
                    constant offset against the prior, and a fresh rigid fit would
                    absorb exactly the thing being shown.

    The underlay is viewer.load_obstacles in modes gt/gt+est -- the same collision
    geometry the chapter 3 scene figures and every trajectory plot use -- and the
    database's own exported occupancy grid in mode nav, via exp4_plot.prior_grid.

ONE ESTIMATOR PER FILE IN MODE gt+est
    analyse() zeroes each system's clock on its own overlap with the reference, so
    two estimators' `t` arrays do not share an origin. Overlaying them would need a
    common clock that analyse() does not currently return, and faking one would put
    the tracks at the wrong instants. So each estimator is animated against its own
    co-clocked ground truth; --system all writes one file per estimator, suffixed
    with the system name. The GT track is identical in all of them bar its overlap.

SIZE
    A GIF stores a full palette frame per image. At the defaults -- 5 fps, 420 px
    tall, the whole run -- a 100 s route lands in the low tens of MB, and gif/ is
    gitignored, so nothing here enters the repository. --start/--end and --height
    are the size controls; an .mp4 --out is roughly an order of magnitude smaller
    and is written as H.264, so it also plays in a browser and in slideware.
"""

import argparse
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import matplotlib                                              # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.patches import Polygon, Rectangle              # noqa: E402
import numpy as np                                             # noqa: E402
from PIL import Image                                          # noqa: E402

WORKSPACE = os.path.expanduser('~/wil_project')
sys.path.insert(0, os.path.join(WORKSPACE, 'script'))
import exp4_plot as e4p                                        # noqa: E402
import render_dynamic_gif as rdg                               # noqa: E402
import exp4_score as es                                        # noqa: E402
import render_dataset_scene as rds                             # noqa: E402
import vio_metrics as vm                                       # noqa: E402
import viewer                                                  # noqa: E402

# exp4_plot's palette, so these frames match the report's static figures.
FILL, GT_C, EST_C, GRID_C = e4p.FILL, e4p.GT_C, e4p.EST_C, e4p.GRID_C
MOVE_FILL, MOVE_EDGE = '#8d6e63', '#4e342e'     # traffic, against the grey racking

# Every visual arm of Experiment 1: one VINS-Fusion run and ORB-SLAM3's full
# 2x2 of front end (CPU/GPU) against loop closer (active/disabled), which is
# five arms on each of the four recordings -- the twenty runs the chapter's
# status table counts.  The front end and the closer switch are both in the
# directory name; the closer STATE is read from loop_closing.csv by the scoring
# scripts, never from the name.
#
# The unqualified names are the GPU front end, exactly as the chapter's accuracy
# table labels them, so a frame can be set beside that table without a mental
# rename.  PINNED is the subset that table reports; the CPU pair is scored in
# its own subsection, so --system pinned reproduces the headline comparison and
# --system all shows all five.
EXP1_SYSTEMS = [("VINS-Fusion",           "output_vins", "cpu"),
                ("ORB-SLAM3",             "output_orb",  "loopclosing_gpu"),
                ("ORB-SLAM3 (no LC)",     "output_orb",  "noloopclosing_gpu"),
                ("ORB-SLAM3 CPU",         "output_orb",  "loopclosing_cpu"),
                ("ORB-SLAM3 CPU (no LC)", "output_orb",  "noloopclosing_cpu")]
PINNED = ("VINS-Fusion", "ORB-SLAM3", "ORB-SLAM3 (no LC)")


# ---------------------------------------------------------------- underlays

_FEED_CACHE = {}


def _prop_feed(bag):
    """read_bag(), memoised: --system all renders five arms in one process and
    deserialising the whole pose feed each time dominates the run."""
    if bag not in _FEED_CACHE:
        print('  reading prop poses from %s' % os.path.basename(bag))
        _FEED_CACHE[bag] = rdg.read_bag(bag, with_cam=False)
    return _FEED_CACHE[bag]


MOVE_EPS = 0.05          # m; below this a prop is treated as fixed


def world_axes(ax, world_path, bag=None, t0=0.0):
    """Draw the Gazebo footprints and wall; return a per-frame updater or None.

    With `bag`, the props that MOVE during the recording are redrawn at their
    pose at each instant instead of at the pose the .world file authors, so a
    dynamic recording looks dynamic. The feed is
    /world/default/pose/info, whose header stamps are all zero -- its per-message
    receive stamp is wall time, which render_dynamic_gif.read_bag interpolates
    through /clock to sim time. Used verbatim the stamps would pin every prop to
    t=0 and the scene would look static.

    A prop counts as moving if its position wanders more than MOVE_EPS over the
    feed; movers are drawn darker and outlined so traffic is distinguishable
    from fixed racking at a glance, and only their patches are touched per frame.

    `t0` is what the animation's t=0 is in ABSOLUTE sim seconds. The series are
    plotted on a zeroed clock because that is the clock the report's timelines
    quote, while the prop feed is in absolute sim time -- on
    \repo{realistic_002} the recording starts at \SI{46.7}{\second}, so
    indexing the feed with the zeroed value froze every prop at its first pose
    for the first 46.7 seconds and then replayed only the opening third of the
    motion. The offset is applied here, in one place.
    """
    if not world_path:
        sys.exit('no world file resolved for this dataset: '
                 'render_dataset_scene.world_for() found no run_metadata.csv '
                 'carrying a world_path row. Pass --world with the absolute '
                 'path to the .world file.')
    if not os.path.exists(world_path):
        sys.exit('world file recorded in the manifest is missing:\n  %s\n'
                 'Pass --world with a path that exists.' % world_path)
    obstacles, wall = viewer.load_obstacles(world_path)

    feed = None
    if bag:
        feed = _prop_feed(bag)
        if not feed['has_poses'] or not len(feed['pose_t']):
            print('  [note] %s carries no /world/default/pose/info; the props '
                  'stay at their authored pose' % os.path.basename(bag))
            feed = None

    moving = set()
    if feed:
        first = {}
        for fr in feed['pose_frames']:
            for n, q in fr.items():
                if n not in first:
                    first[n] = q
                elif (abs(q[0] - first[n][0]) > MOVE_EPS
                      or abs(q[1] - first[n][1]) > MOVE_EPS):
                    moving.add(n)
        print('  %d of %d props move during the recording'
              % (len(moving), len(first)))

    movers = {}
    for lab, poly, pose in obstacles:
        arr = np.asarray(poly, float)
        if len(arr) < 3:
            continue
        if lab in moving:
            patch = Polygon(arr, closed=True, facecolor=MOVE_FILL,
                            edgecolor=MOVE_EDGE, lw=0.5, alpha=0.75, zorder=2)
            movers[lab] = (patch, arr, pose)
        else:
            patch = Polygon(arr, closed=True, facecolor=FILL, edgecolor='none',
                            alpha=0.30, zorder=1)
        ax.add_patch(patch)

    x0, y0, x1, y1 = wall
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                           edgecolor=FILL, lw=1.0, alpha=0.6, zorder=1))
    pad = 0.5
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)

    if not movers:
        return None
    pose_t, frames_ = feed['pose_t'], feed['pose_frames']

    def update(T):
        k = min(max(int(np.searchsorted(pose_t, T + t0)), 0), len(frames_) - 1)
        live = frames_[k]
        for lab, (patch, arr, p0) in movers.items():
            p1 = live.get(lab)
            if p1 is not None:
                patch.set_xy(rdg.live_poly(arr, p0, p1))
    return update


def grid_axes(ax, db, al, tracks):
    """Draw the prior's exported occupancy grid, brought from map to world frame."""
    g = e4p.prior_grid(db)
    if g is None:
        print('  [warn] no grid exported; the dense panel will show the paths only')
    else:
        w = es.map_to_world(al, g)
        ax.scatter(w[:, 0], w[:, 1], s=0.6, c=GRID_C, alpha=0.55,
                   linewidths=0, zorder=1)
    pts = [t for t in tracks if t is not None and len(t)]
    if pts:
        allp = np.vstack(pts)
        pad = 2.0
        ax.set_xlim(allp[:, 0].min() - pad, allp[:, 0].max() + pad)
        ax.set_ylim(allp[:, 1].min() - pad, allp[:, 1].max() + pad)


# ---------------------------------------------------------------- data loading

def load_gt_only(dataset):
    """Full ground truth at the MODEL ORIGIN, on its own sim clock."""
    cands = vm.gt_candidates('simulation/' + dataset)
    path = next((c for c in cands if os.path.exists(c)), None)
    if path is None:
        sys.exit('no ground truth for %s; looked in:\n  %s'
                 % (dataset, '\n  '.join(cands)))
    r = vm.load_gt(path, lever=False)
    if r is None:
        sys.exit('ground truth at %s is empty' % path)
    t, p = r[0], r[1]
    return t - t[0], p[:, :2], path, float(t[0])


def load_prop_paths(bag, t0):
    """Every moving prop's path, on the same zeroed clock as the other modes.

    The mover test is character-for-character the one world_axes applies --
    more than MOVE_EPS from the FIRST pose in x or y -- so the set that gets a
    coloured trail here is exactly the set that gets a highlighted footprint
    underneath it. Two different thresholds would leave a prop moving with no
    path, or a path under a prop drawn as fixed racking.

    A prop is not guaranteed to appear in every message of the feed, so each
    one carries its own time vector rather than being indexed into a shared
    one. `t0` converts the feed's absolute sim time to the zeroed clock the
    report's timelines quote, the same offset world_axes applies.
    """
    feed = _prop_feed(bag)
    if not feed['has_poses'] or not len(feed['pose_t']):
        sys.exit('%s carries no /world/default/pose/info, so there are no prop '
                 'paths to draw. Mode props needs the pose feed; a static '
                 'dataset has nothing to show.' % os.path.basename(bag))
    ft = np.asarray(feed['pose_t']) - t0
    per = {}
    for i, fr in enumerate(feed['pose_frames']):
        for n, q in fr.items():
            ts, xys = per.setdefault(n, ([], []))
            ts.append(ft[i])
            xys.append((q[0], q[1]))
    out = []
    for n, (ts, xys) in per.items():
        xy = np.asarray(xys, float)
        d = np.abs(xy - xy[0]).max(0)
        if d[0] > MOVE_EPS or d[1] > MOVE_EPS:
            out.append((n, np.asarray(ts), xy))
    out.sort(key=lambda r: r[0])
    if not out:
        sys.exit('no prop moves more than %.2f m in this recording -- it is a '
                 'static dataset, so there is no traffic pattern to show'
                 % MOVE_EPS)
    return out


def load_gt_est(dataset, system):
    """One estimator's aligned series and its co-clocked ground truth."""
    want = [s for s in EXP1_SYSTEMS if s[0] == system]
    if not want:
        sys.exit('unknown --system %r; choose from: %s, or "all" / "pinned"'
                 % (system, ', '.join(repr(s[0]) for s in EXP1_SYSTEMS)))
    run_dir = os.path.join(vm.ROOT, 'output', want[0][1], 'simulation',
                           dataset, want[0][2])
    if not os.path.isdir(run_dir):
        sys.exit('no run directory for %s on %s:\n  %s' % (system, dataset, run_dir))
    res = vm.analyse('simulation/' + dataset, systems=want)
    if res is None or not res[1]:
        sys.exit('nothing to align for %s / %s -- check the run directory exists'
                 % (dataset, system))
    al = res[1][0]
    row = res[0][0]
    print('  %s: ATE %.3f m RMS over %.0f s, %d samples, sim t0 = %.2f s'
          % (row['name'], row['rms'], row['secs'], row['poses'], al['t0']))
    return al['t'], al['ref'][:, :2], al['p'][:, :2], row, al['t0']


def load_nav(run):
    """One Experiment 4 run: ground truth, published pose, prior, grid placement."""
    import yaml
    spec = yaml.safe_load(open(os.path.join(WORKSPACE, 'script',
                                            'exp4_protocol.yaml')))
    for world in spec['worlds']:
        lay = spec['worlds'][world]['layout']
        for path in spec['priors'][lay]['paths']:
            # The repeat is READ from the name, not matched against a fixed
            # list. The campaign renumbers its repeats -- rep 5 is the current
            # one -- and a hardcoded (1, 2, 3) rejected it as an unknown run
            # with the directory sitting right there, which is a confusing way
            # to fail for something the protocol does not bound.
            prefix = f'{world}_{path}_'
            tail = run[len(prefix):]
            if not run.startswith(prefix) or not tail.isdigit():
                continue
            tr = e4p.run_tracks(spec, world, path, int(tail))
            if tr is None:
                sys.exit('run %s has no robot_gt.csv -- the directory is '
                         'map/<prior>/exp4/%s/' % (run, run))
            return tr
    sys.exit('unknown --run %r; it takes WORLD_PATH_REP, e.g. '
             'U_static_P3_circuit_5' % run)


# ---------------------------------------------------------------- rendering

def frames(fig, ax, series, times, fps, trail, note, update=None):
    """Yield one PIL frame per sampled instant, updating only the line data.

    A series is (name, t, xy, colour) or (name, t, xy, colour, kwargs), the
    kwargs overriding the line style -- which is how the acquisition transient
    of mode nav is drawn faded instead of solid.
    """
    lines, heads = [], []
    for s in series:
        name, colour, kw = s[0], s[3], (s[4] if len(s) > 4 else {})
        style = dict(color=colour, lw=1.4, zorder=3, label=name)
        style.update(kw)
        ln, = ax.plot([], [], **style)
        hd, = ax.plot([], [], 'o', color=colour, ms=5, zorder=4,
                      alpha=style.get('alpha', 1.0))
        lines.append(ln)
        heads.append(hd)

    # Legend and captions live in the FIGURE margins, never over the plan: a
    # track that passes under a legend box is a track the viewer cannot check,
    # and these frames exist to be checked against the map.  Only `lines` are
    # passed as handles, so the position markers stay out of the key.
    # A series named '_' is drawn but kept out of the key. Mode props puts one
    # series per moving prop on the axes, and a twelve-entry legend would eat
    # the plan it is there to explain, so the movers share a single entry.
    named = [l for l in lines if not l.get_label().startswith('_')]
    ncol = 2 if len(named) > 1 else 1
    rows = -(-len(named) // ncol)          # ceil, so the reserve matches the key
    if named:
        fig.legend(handles=named, loc='upper center', bbox_to_anchor=(0.5, 0.995),
                   ncol=ncol, fontsize=7, frameon=False)
    fig.subplots_adjust(top=0.88 - 0.045 * rows, bottom=0.14)
    txt = fig.text(0.012, 0.02, '', fontsize=7.5, family='monospace')

    for T in times:
        if update is not None:
            update(T)                  # move the props to their pose at T
        for s, ln, hd in zip(series, lines, heads):
            t, xy = s[1], s[2]
            m = t <= T
            if trail:
                m &= t >= T - trail
            if m.any():
                ln.set_data(xy[m, 0], xy[m, 1])
                hd.set_data(xy[m, 0][-1:], xy[m, 1][-1:])
            else:
                ln.set_data([], [])
                hd.set_data([], [])
        txt.set_text('t = %6.1f s%s' % (T, '    ' + note if note else ''))
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        yield Image.fromarray(buf[:, :, :3].copy())


def write_mp4(images, out, fps):
    """H.264 where it is available, MPEG-4 part 2 where it is not.

    Two writers, and the difference matters for where the file will be played.
    imageio drives the ffmpeg binary that ships inside imageio-ffmpeg, so
    libx264 is reachable without ffmpeg being installed on the host, and the
    result is avc1/yuv420p: it plays in a browser, in Google Slides, in
    PowerPoint and in Keynote, and is several times smaller than the fallback.
    OpenCV's own build here carries no H.264 encoder -- avc1, H264, h264 and
    X264 all fail to open a writer -- so the fallback is mp4v, which VLC and
    PowerPoint play but a browser does not.

    Either writer needs even frame dimensions, and NEITHER of them pads to get
    there. imageio appends `-vf scale=W:H` for a frame that macro_block_size
    does not divide (imageio_ffmpeg/_io.py), and cv2 crops down to it in
    silence. Both change the metres-per-pixel the panel is drawn to, and a
    frame here is a scale drawing -- resampling it would falsify a metre. So an
    odd frame is padded by hand and both writers are handed a size they will
    leave alone, which is why macro_block_size is 1 rather than 2.
    """
    arr = [np.asarray(im)[:, :, :3] for im in images]
    h, w = arr[0].shape[:2]
    if h % 2 or w % 2:
        ph, pw = h + h % 2, w + w % 2
        padded = []
        for im in arr:
            p = np.zeros((ph, pw, 3), dtype=im.dtype)
            p[:h, :w] = im
            padded.append(p)
        arr = padded
        print('  [note] %dx%d is odd; padded to %dx%d rather than rescaled'
              % (w, h, pw, ph))
    try:
        import imageio.v2 as iio
        iio.mimwrite(out, arr, fps=fps, codec='libx264', quality=8,
                     macro_block_size=1)
        return 'H.264'
    except Exception as exc:                      # no imageio, or no libx264
        print('  [note] H.264 writer unavailable (%s); falling back to mp4v, '
              'which browsers will not play' % type(exc).__name__)
    import cv2
    h, w = arr[0].shape[:2]
    h, w = h - h % 2, w - w % 2
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for im in arr:
        vw.write(cv2.cvtColor(im[:h, :w], cv2.COLOR_RGB2BGR))
    vw.release()
    return 'MPEG-4 part 2'


def write(images, out, fps):
    if not images:
        sys.exit('no frames rendered -- check --start/--end against the run length')
    os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
    codec = None
    if out.lower().endswith('.mp4'):
        codec = write_mp4(images, out, fps)
    else:
        images[0].save(out, save_all=True, append_images=images[1:],
                       duration=int(1000 / fps), loop=0, optimize=True)
    mb = os.path.getsize(out) / 1e6
    print('  wrote %s  (%d frames, %.1f MB%s)'
          % (out, len(images), mb, ', ' + codec if codec else ''))
    if mb > 25 and not out.lower().endswith('.mp4'):
        print('  [note] large for a GIF; --height 300, a tighter --start/--end, '
              'or an .mp4 --out all cut this sharply')


def render(series, underlay, title, note, a, out):
    """series: [(name, t, xy, colour[, kwargs])]; underlay: callable(ax).

    `title` is centred over the axes and may carry a newline; keep each line
    under about 45 characters or it runs off a 420 px canvas.  Anything longer
    -- a prior's database name, an alignment convention -- belongs in `note`,
    which is drawn in the bottom margin beside the clock.
    """
    h_in = a.height / 100.0
    fig, ax = plt.subplots(figsize=(h_in * 1.22, h_in), dpi=a.dpi)
    update = underlay(ax)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]', fontsize=8)
    ax.set_ylabel('y [m]', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=8, loc='center', pad=4)
    fig.tight_layout()

    tmax = max(s[1].max() for s in series if len(s[1]))
    tmin = min(s[1].min() for s in series if len(s[1]))
    lo = a.start if a.start is not None else tmin
    hi = a.end if a.end is not None else tmax
    times = np.arange(lo, hi + 1e-9, 1.0 / a.fps)
    print('  %d frames, t = %.1f..%.1f s at %d fps' % (len(times), lo, hi, a.fps))
    ims = list(frames(fig, ax, series, times, a.fps, a.trail, note, update))
    plt.close(fig)
    write(ims, out, a.fps)


def suffixed(out, tag):
    root, ext = os.path.splitext(out)
    return '%s_%s%s' % (root, tag.lower().replace(' ', '').replace('(', '')
                        .replace(')', '').replace('-', ''), ext)


def prop_bag(a, world):
    """The bag to animate props from, or None to leave them at authored pose.

    A missing bag is a degraded render, not a failure: the drive can be
    unmounted, and a still scene with a warning beats no animation at all.
    """
    if a.props == 'authored':
        return None
    dynamic = '_dynamic_' in os.path.basename(world or '')
    if a.props == 'auto' and not dynamic:
        return None
    try:
        return rdg.bag_for(a.dataset)
    except SystemExit as e:
        print('  [note] %s; props stay at their authored pose' % e)
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', required=True,
                    choices=('gt', 'gt+est', 'nav', 'props'))
    ap.add_argument('--dataset', help='modes gt/gt+est, e.g. '
                                      'dataset_allsensor_realistic_001')
    ap.add_argument('--run', help='mode nav, WORLD_PATH_REP')
    ap.add_argument('--system', default='VINS-Fusion',
                    help='mode gt+est: one of %s; "all" for one file per arm, '
                         'or "pinned" for the three the accuracy table reports'
                         % ', '.join(repr(s[0]) for s in EXP1_SYSTEMS))
    ap.add_argument('--world', help='override the world file (modes gt/gt+est)')
    ap.add_argument('--props', default='auto', choices=('auto', 'live', 'authored'),
                    help='modes gt/gt+est: "live" redraws moving props from the '
                         'bag at each instant, "authored" leaves them where the '
                         '.world file puts them, "auto" (default) picks live for '
                         'a _dynamic_ world when its bag is reachable')
    ap.add_argument('--panel', default='dense', choices=('dense', 'world'),
                    help='mode nav: the dense prior grid (default) or the world plan')
    ap.add_argument('--out', required=True, help='.gif or .mp4')
    ap.add_argument('--fps', type=float, default=5.0)
    ap.add_argument('--height', type=int, default=420,
                    help='frame height in pixels AT --dpi 100. It also fixes '
                         'the layout: every font size here is in points, so '
                         'raising this alone widens the view and leaves the '
                         'labels behind. Default 420.')
    ap.add_argument('--dpi', type=int, default=100,
                    help='resolution multiplier. The frame comes out '
                         '--height x --dpi/100 pixels tall and the text grows '
                         'with it, so --dpi 200 doubles the resolution at an '
                         'unchanged layout. Default 100.')
    ap.add_argument('--start', type=float)
    ap.add_argument('--end', type=float)
    ap.add_argument('--trail', type=float,
                    help='draw only the last N seconds of each track')
    a = ap.parse_args()

    if a.mode in ('gt', 'gt+est', 'props') and not a.dataset:
        sys.exit('--dataset is required for --mode %s' % a.mode)
    if a.mode == 'nav' and not a.run:
        sys.exit('--run is required for --mode nav')

    if a.mode == 'gt':
        world = a.world or rds.world_for(a.dataset)
        bag = prop_bag(a, world)
        t, xy, path, t0 = load_gt_only(a.dataset)
        print('%s\n  ground truth: %s (%d samples, model origin, '
              'sim t0 = %.2f s)' % (a.dataset, path, len(t), t0))
        render([('ground truth', t, xy, GT_C)],
               lambda ax: world_axes(ax, world, bag, t0),
               a.dataset, 'ground truth at the model origin', a, a.out)

    elif a.mode == 'props':
        world = a.world or rds.world_for(a.dataset)
        bag = rdg.bag_for(a.dataset)          # mandatory here, not best-effort
        t, xy, path, t0 = load_gt_only(a.dataset)
        paths = load_prop_paths(bag, t0)
        print('%s\n  ground truth: %s (sim t0 = %.2f s)' % (a.dataset, path, t0))
        print('  %d moving props, paths span t = %.1f..%.1f s'
              % (len(paths), min(p[1].min() for p in paths),
                 max(p[1].max() for p in paths)))
        # One colour per prop. Twelve trails in a single colour cross each other
        # into a scribble; distinct hues let a viewer follow ONE box down an
        # aisle, which is the thing being demonstrated.
        cmap = plt.get_cmap('tab20')(np.linspace(0.0, 1.0, 20))
        series = [('robot (ground truth)', t, xy, GT_C, dict(lw=1.8))]
        for i, (_n, pt, pxy) in enumerate(paths):
            series.append(('%d moving obstacles, one colour each' % len(paths)
                           if i == 0 else '_',
                           pt, pxy, cmap[i % 20], dict(lw=1.2, alpha=0.9)))
        lay = 'I' if '_01' in os.path.basename(world or '') else 'U'
        render(series,
               lambda ax: world_axes(ax, world, bag, t0),
               '%s\n%d moving obstacles on the %s layout' % (a.dataset,
                                                             len(paths), lay),
               'paths accumulate; dots are current positions', a, a.out)

    elif a.mode == 'gt+est':
        world = a.world or rds.world_for(a.dataset)
        bag = prop_bag(a, world)
        if a.system == 'all':
            names = [s[0] for s in EXP1_SYSTEMS]
        elif a.system == 'pinned':
            names = list(PINNED)
        else:
            names = [a.system]
        for i, name in enumerate(names):
            print('%s / %s' % (a.dataset, name))
            t, ref, est, row, t0 = load_gt_est(a.dataset, name)
            out = suffixed(a.out, name) if len(names) > 1 else a.out
            render([('ground truth (IMU point)', t, ref, GT_C),
                    ('%s, aligned' % name, t, est, EST_C)],
                   lambda ax, t0=t0: world_axes(ax, world, bag, t0),
                   '%s\n%s -- ATE %.3f m RMS' % (a.dataset, name, row['rms']),
                   'rigid SE(3), scale fixed; GT at the IMU point', a, out)

    else:
        tr = load_nav(a.run)
        if tr['est'] is None:
            sys.exit('run %s has no published-pose CSV beside its prior; '
                     'mode nav needs both tracks' % a.run)
        # Before the first informative pose the published transform is the seed
        # composed with raw odometry, so it can sit far outside the building and
        # sweep in when a fix lands (exp4_plot.run_tracks, and the 73-76 deg
        # startup latch of the acceptance sweep). Drawing that solid makes an
        # acquisition transient look like a localisation failure, so it is
        # split off and faded, exactly as the static figure does.
        et, ff = tr['est_t'], tr['first_fix']
        if ff is None:
            series = [('published pose', et, tr['est'], EST_C)]
        else:
            pre, post = et < ff, et >= ff
            series = []
            if pre.any():
                series.append(('published pose, before first fix',
                               et[pre], tr['est'][pre], EST_C,
                               dict(alpha=0.35, ls='--', lw=1.0)))
            series.append(('published pose', et[post], tr['est'][post], EST_C))
        series.insert(0, ('ground truth (IMU point)', tr['t'],
                          tr['gt'][:, :2], GT_C))
        print('%s\n  prior: %s  (%d GT samples, %d pose samples)'
              % (a.run, tr['prior'], len(tr['t']), len(tr['est_t'])))
        if a.panel == 'world':
            underlay = lambda ax: world_axes(ax, tr['world_file'])          # noqa: E731
            what, note = 'world plan', 'prior %s' % tr['prior']
        else:
            db = os.path.join(WORKSPACE, 'map', tr['prior'], tr['prior'] + '.db')
            underlay = lambda ax: grid_axes(                                # noqa: E731
                ax, db, tr['al'], [tr['gt'][:, :2], tr['est']])
            what, note = 'dense prior grid', 'prior %s' % tr['prior']
        render(series, underlay, '%s\n%s' % (a.run, what), note, a, a.out)


if __name__ == '__main__':
    main()
