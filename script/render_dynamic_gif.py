#!/usr/bin/env python3
"""Animate a dynamic-scene recording: moving props, robot route, and the camera view.

    python3 script/render_dynamic_gif.py --dataset dataset_dynamic_nofloortexture_01_000 \\
        --out gif/dynamic_01_000.gif

    python3 script/render_dynamic_gif.py --dataset ... --out x.gif \\
        --fps 5 --height 400 --start 30 --end 90      # a trimmed segment
    python3 script/render_dynamic_gif.py --dataset ... --out x.mp4   # MP4 via OpenCV

WHAT THE TWO PANELS ARE
    left    top-down scene in the world frame: the floor as the world's own
            ground model renders it, every obstacle footprint at its pose AT THAT
            INSTANT, the ground-truth route drawn up to that instant, and the robot
            with its heading. The static scene figure (render_dataset_scene.py)
            draws props where the world file puts them at t=0; this one draws them
            where they actually were.
    right   the cam0 frame nearest to that instant. Side by side, a viewer can see
            a prop cross the aisle in the plan view and, at the same moment, cross
            the camera's field of view -- which is the whole point of a dynamic
            dataset, and is invisible in any still figure.

THE CLOCK PROBLEM, AGAIN
    /world/default/pose/info carries every model's pose but its header stamps are
    zero on every message. Used verbatim it pins all 30 moving props to t=0 and the
    scene looks static. Its per-message bag receive stamp is wall time, and wall
    does not track sim by a constant offset. So, as yolo_eval.load_bag does, sim
    time is rebuilt from /clock -- sim in the payload, wall in the receive stamp --
    and each pose message's wall stamp is interpolated through that map. The route
    and the camera carry real sim stamps and need none of this.

WHICH PROPS MOVE
    Nothing here assumes it. Every obstacle in the world is drawn at whatever pose
    the feed reports for it at that instant, so a static shelf simply stays where
    it is and a moving bucket moves. On dataset_dynamic_nofloortexture_01_000 the
    feed moves the 30 props numbered 4xx by about 24 m each and leaves the 1xx
    fixtures alone.

SIZE
    A GIF stores a full palette frame per image and compresses photographs badly.
    At the defaults -- 5 fps, 400 px tall, the whole run -- expect tens of MB. The
    --start/--end trim and --height knobs are the size controls; --out with an .mp4
    suffix writes H.264-free MP4 through OpenCV (mp4v) at a fraction of the size.
"""

import argparse
import math
import os
import sqlite3
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import viewer
import vio_metrics as vm
from render_dataset_scene import world_for, to_px, OBJ_FILL, OBJ_EDGE

ROOT = vm.ROOT
FLASH = '/media/ambushee/32E4AAB1E4AA772F/dataset'

ROUTE = (0, 132, 255)
ROBOT = (214, 40, 40)
MOVING_FILL = (198, 96, 32)      # moving props in a different colour to the fixtures
MOVING_EDGE = (120, 52, 12)


def bag_for(name):
    for base in (os.path.join(ROOT, 'dataset'), FLASH):
        d = os.path.join(base, name)
        if os.path.isdir(d):
            dbs = sorted(f for f in os.listdir(d) if f.endswith('.db3'))
            if dbs:
                return os.path.join(d, dbs[0])
    raise SystemExit('no bag found for %s in dataset/ or the flash drive' % name)


def read_bag(bag, with_cam=True):
    """-> clock map, pose tracks (sim-stamped), route, camera frames."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    con = sqlite3.connect('file:%s?mode=ro' % bag, uri=True)
    T = {r[1]: (r[0], r[2]) for r in con.execute('select id,name,type from topics')}

    def rows(topic):
        tid, typ = T[topic]
        mt = get_message(typ)
        for ts, d in con.execute(
                'select timestamp,data from messages where topic_id=? order by timestamp',
                (tid,)):
            yield ts * 1e-9, deserialize_message(bytes(d), mt)

    clock_wall, clock_sim = [], []
    for w, m in rows('/clock'):
        clock_wall.append(w)
        clock_sim.append(m.clock.sec + m.clock.nanosec * 1e-9)
    cw, cs = np.asarray(clock_wall), np.asarray(clock_sim)
    o = np.argsort(cw)
    cw, cs = cw[o], cs[o]

    pose_t, pose_frames = [], []
    # Not every recording carries the prop-pose feed. dataset_map_02 has no
    # /world/default/pose/info at all, so there is nothing to animate the props
    # with and they must stay at their authored pose -- which the caller is told,
    # rather than being shown a still scene that looks like a dynamic one.
    has_poses = '/world/default/pose/info' in T
    for w, m in (rows('/world/default/pose/info') if has_poses else ()):
        frame = {}
        for tr in m.transforms:
            n = tr.child_frame_id
            if not n.startswith('aws_robomaker_warehouse_'):
                continue
            t, q = tr.transform.translation, tr.transform.rotation
            frame[n] = (t.x, t.y, math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                             1.0 - 2.0 * (q.y * q.y + q.z * q.z)))
        pose_t.append(w)
        pose_frames.append(frame)
    pose_t = (np.interp(np.asarray(pose_t), cw, cs) if pose_t
              else np.zeros(0))                          # wall -> sim

    rt, rx, ry, ryaw = [], [], [], []
    for _, m in rows('/ground_truth/odometry'):
        h = m.header.stamp
        p, q = m.pose.pose.position, m.pose.pose.orientation
        rt.append(h.sec + h.nanosec * 1e-9)
        rx.append(p.x)
        ry.append(p.y)
        ryaw.append(math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                               1.0 - 2.0 * (q.y * q.y + q.z * q.z)))
    route = dict(t=np.asarray(rt), x=np.asarray(rx), y=np.asarray(ry),
                 yaw=np.asarray(ryaw))

    cam_t, cam_jpg = [], []
    if with_cam:
        for _, m in rows('/cam0/image_raw/compressed'):
            h = m.header.stamp
            cam_t.append(h.sec + h.nanosec * 1e-9)
            cam_jpg.append(bytes(m.data))
    con.close()
    return dict(pose_t=pose_t, pose_frames=pose_frames, route=route,
                cam_t=np.asarray(cam_t), cam_jpg=cam_jpg, has_poses=has_poses)


def live_poly(poly, p0, p1):
    """Rigid-transform a footprint baked at authored pose p0 to live pose p1.

    Same arithmetic as viewer.MapView._live_poly; restated here so this script
    does not reach into a Qt view class for six lines of geometry.
    """
    dyaw = p1[2] - p0[2]
    c, s = math.cos(dyaw), math.sin(dyaw)
    dx, dy = poly[:, 0] - p0[0], poly[:, 1] - p0[1]
    return np.stack([p1[0] + c * dx - s * dy, p1[1] + s * dx + c * dy], 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, help='e.g. dataset_dynamic_nofloortexture_01_000')
    ap.add_argument('--out', required=True, help='.gif, or .mp4 for OpenCV video')
    ap.add_argument('--fps', type=float, default=5.0, help='output frames per SIM second')
    ap.add_argument('--height', type=int, default=400, help='panel height in px')
    ap.add_argument('--start', type=float, default=None, help='sim-time start, seconds')
    ap.add_argument('--end', type=float, default=None, help='sim-time end, seconds')
    ap.add_argument('--ppm', type=float, default=12.0,
                    help='pixels per metre for the scene render before scaling')
    ap.add_argument('--world',
                    help='world file to draw the floor and objects from. Needed when '
                         'no estimator run manifest records one, which is the case for '
                         'any dataset that has never been run through ORB or VINS.')
    ap.add_argument('--no-cam', action='store_true',
                    help='scene panel only. The camera panel is most of the file '
                         'size: a GIF compresses a plan view well and a photograph '
                         'badly. Drop it when the point is the layout, not the view.')
    a = ap.parse_args()

    name = a.dataset.split('/')[-1]
    bag = bag_for(name)
    world = a.world or world_for('simulation/' + name)
    if world is None:
        raise SystemExit(
            'no world recorded for %s -- it has no estimator run manifest to read one '
            'from. Pass --world <path to .world>; guessing it would put the wrong floor '
            'and the wrong object layout under a real trajectory.' % name)
    if not os.path.isfile(world):
        raise SystemExit('world not on disk: %s' % world)
    print('bag   %s' % bag)
    print('world %s' % os.path.basename(world))

    data = read_bag(bag, with_cam=not a.no_cam)
    route = data['route']
    t0 = route['t'][0] if a.start is None else max(route['t'][0], a.start)
    t1 = route['t'][-1] if a.end is None else min(route['t'][-1], a.end)
    times = np.arange(t0, t1, 1.0 / a.fps)
    print('sim %.2f..%.2f s  -> %d frames at %.1f fps' % (t0, t1, len(times), a.fps))

    # static background: floor + wall ring, obstacles drawn per frame
    dae, concrete, atlas, n_marks = viewer.ground_assets(world)
    floor = viewer.build_floor_array(concrete, a.ppm, '')
    if atlas and n_marks:
        viewer.paint_markings(floor, a.ppm, '', dae_path=dae, atlas_path=atlas)
    bg = Image.fromarray(floor).convert('RGB')
    obstacles, wall = viewer.load_obstacles(world)
    if wall is not None:
        x0, y0, x1, y1 = wall
        c0, r0 = to_px(np.array([[x0, y1]]), a.ppm)[0]
        c1, r1 = to_px(np.array([[x1, y0]]), a.ppm)[0]
        ImageDraw.Draw(bg).rectangle([c0, r0, c1, r1], outline=OBJ_EDGE,
                                     width=max(2, int(round(0.20 * a.ppm))))
    scene_w, scene_h = bg.size
    scale = a.height / scene_h
    out_scene = (int(round(scene_w * scale)), a.height)
    cam_w = 0 if a.no_cam else int(round(1280 * a.height / 720))
    # authored pose per obstacle, and which ones the feed ever moves
    authored = {lab: (np.asarray(poly, float), pose) for lab, poly, pose in obstacles}
    moved = set()
    if data['has_poses'] and data['pose_frames']:
        first = data['pose_frames'][0]
        for fr in data['pose_frames'][::10]:
            for n, p in fr.items():
                q = first.get(n)
                if q and (abs(p[0] - q[0]) > 0.05 or abs(p[1] - q[1]) > 0.05):
                    moved.add(n)
        print('obstacles %d, of which the feed moves %d' % (len(authored), len(moved)))
    else:
        print('obstacles %d; NO /world/default/pose/info in this bag, so props are '
              'drawn at their authored pose and nothing in the scene moves but the '
              'robot' % len(authored))

    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    except OSError:
        font = ImageFont.load_default()

    is_mp4 = a.out.lower().endswith('.mp4')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    frames_out = []
    writer = None
    pose_t = data['pose_t']
    trail_w = max(2, int(round(0.05 * a.ppm)))
    for i, t in enumerate(times):
        # --- scene panel ---
        im = bg.copy()
        dr = ImageDraw.Draw(im)
        if len(pose_t):
            k = min(max(int(np.searchsorted(pose_t, t)), 0), len(pose_t) - 1)
            live = data['pose_frames'][k]
        else:
            live = {}
        for lab, (poly, p0) in authored.items():
            p1 = live.get(lab)
            pts = live_poly(poly, p0, p1) if p1 is not None else poly
            px = [tuple(v) for v in to_px(pts, a.ppm)]
            if len(px) >= 3:
                if lab in moved:
                    dr.polygon(px, fill=MOVING_FILL, outline=MOVING_EDGE)
                else:
                    dr.polygon(px, fill=OBJ_FILL, outline=OBJ_EDGE)
        m = route['t'] <= t
        if m.sum() >= 2:
            pts = [tuple(v) for v in to_px(np.stack([route['x'][m], route['y'][m]], 1), a.ppm)]
            dr.line(pts, fill=ROUTE, width=trail_w)
        j = min(int(np.searchsorted(route['t'], t)), len(route['t']) - 1)
        rx, ry, ryaw = route['x'][j], route['y'][j], route['yaw'][j]
        c, r = to_px(np.array([[rx, ry]]), a.ppm)[0]
        rr = max(4, int(round(0.35 * a.ppm)))
        dr.ellipse([c - rr, r - rr, c + rr, r + rr], fill=ROBOT)
        hx, hy = c + 2.2 * rr * math.cos(ryaw), r - 2.2 * rr * math.sin(ryaw)
        dr.line([(c, r), (hx, hy)], fill=ROBOT, width=max(2, rr // 2))
        im = im.resize(out_scene, Image.LANCZOS)

        # --- camera panel ---
        cam = None
        if not a.no_cam:
            q = min(int(np.searchsorted(data['cam_t'], t)), len(data['cam_t']) - 1)
            jpg = cv2.imdecode(np.frombuffer(data['cam_jpg'][q], np.uint8), cv2.IMREAD_COLOR)
            cam = Image.fromarray(cv2.cvtColor(jpg, cv2.COLOR_BGR2RGB)).resize(
                (cam_w, a.height), Image.BILINEAR)

        # --- compose ---
        W = out_scene[0] + (8 + cam_w if cam is not None else 0)
        canvas = Image.new('RGB', (W, a.height + 22), (24, 24, 24))
        canvas.paste(im, (0, 22))
        if cam is not None:
            canvas.paste(cam, (out_scene[0] + 8, 22))
        d2 = ImageDraw.Draw(canvas)
        # Without the camera panel the canvas is ~400 px wide and the full header
        # does not fit: the timestamp was clipped and the prop count lost. The
        # dataset name is the one part that can go -- it is the filename -- so it
        # is dropped when the string would overrun, rather than the numbers.
        motion = ('props moving: %d' % len(moved)) if data['has_poses'] \
            else 'no prop-pose feed'
        hdr = '%s   sim t = %6.2f s   %s' % (name, t, motion)
        if d2.textlength(hdr, font=font) > W - 12:
            hdr = 'sim t = %6.2f s   %s' % (t, motion)
        d2.text((6, 4), hdr, fill=(235, 235, 235), font=font)
        # Right-aligned at the far edge. Drawn at the panel seam it sat on top of
        # the timestamp text, which runs past the seam at this height.
        if cam is not None:
            lab = 'cam0'
            lw = d2.textlength(lab, font=font)
            d2.text((W - lw - 8, 4), lab, fill=(235, 235, 235), font=font)

        if is_mp4:
            if writer is None:
                writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*'mp4v'),
                                         a.fps, canvas.size)
            writer.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
        else:
            frames_out.append(canvas.quantize(colors=256, method=Image.Quantize.MEDIANCUT))
        if i % 100 == 0:
            print('  frame %d/%d  t=%.1f' % (i, len(times), t))

    if is_mp4:
        writer.release()
    else:
        frames_out[0].save(a.out, save_all=True, append_images=frames_out[1:],
                           duration=int(round(1000.0 / a.fps)), loop=0, optimize=False)
    print('wrote %s  (%.1f MB, %d frames, %dx%d)'
          % (a.out, os.path.getsize(a.out) / 1e6, len(times), W, a.height + 22))


if __name__ == '__main__':
    main()
