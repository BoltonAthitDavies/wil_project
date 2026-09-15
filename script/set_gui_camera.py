#!/usr/bin/env python3
"""Put the GUI's startup camera somewhere it can actually see the warehouse.

    python3 script/set_gui_camera.py --world <a.world> [--tiny] [--dry-run]

THE BUG THIS FIXES
    Every world shipped with

        <camera_pose>0.0 0.0 10.0 0.0 1.5708 -1.5708</camera_pose>

    which is z = 10.0 m, pitched +1.5708 rad -- straight down. The roof slab
    occupies z 9.020 .. 10.124, so that puts the camera INSIDE the roof looking
    into it. The GUI opens, the scene loads, and the viewport is a flat surface:
    "I cannot see the simulation". It is not a rendering failure, it is a camera
    parked inside geometry.

    Looking down from above the roof does not help either -- the roof is opaque.
    The camera has to be under it, so the replacement is an interior corner view.

    GUI-only. <camera_pose> affects nothing about physics, sensors or the baked
    map; the robot's own cameras are unrelated.
"""
import re, os, sys, math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOF_BOTTOM, ROOF_TOP = 9.020, 10.124   # measured from RoofB_01's collision mesh


def pose_for(cx, cy, cz, tx, ty, tz):
    """camera_pose looking from (cx,cy,cz) at (tx,ty,tz). +pitch looks down."""
    d = math.hypot(tx - cx, ty - cy)
    return '%.6f %.6f %.6f 0.000000 %.6f %.6f' % (
        cx, cy, cz, math.atan2(cz - tz, d), math.atan2(ty - cy, tx - cx))


# Rescaled warehouse: interior +-20.6 m, 9.02 m clear. Sit high in the SW corner,
# under the roof, and look at the middle of the floor.
BIG = pose_for(-18.0, -18.0, 8.5, 0.0, 0.0, 1.0)
# model_tiny keeps the pristine 13.98 x 20.91 m shell, so the big pose would be
# outside its walls and see nothing but brick.
TINY = pose_for(-5.5, -8.5, 6.0, 0.0, 0.0, 1.0)


def main():
    worlds = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == '--world']
    dry = '--dry-run' in sys.argv
    force = '--force' in sys.argv
    if not worlds:
        raise SystemExit(__doc__)
    for w in worlds:
        path = w if os.path.isabs(w) else os.path.join(ROOT, w)
        text = open(path).read()
        want = TINY if 'tinymap' in os.path.basename(path) else BIG
        new, n = re.subn(r'<camera_pose>[^<]*</camera_pose>',
                         '<camera_pose>%s</camera_pose>' % want, text)
        old = re.search(r'<camera_pose>([^<]*)</camera_pose>', text)
        if not n:
            print('  %-58s no <camera_pose>, skipped' % os.path.basename(path))
            continue
        # Only touch a camera that is actually buried in the roof. Two things got
        # clobbered by not checking: the archived map_v1 / map_v2 snapshots, which
        # must stay byte-identical to what they recorded, and
        # no_roof_small_warehouse.world, which HAS no roof and so already had a
        # perfectly good pose outside the building.
        if '/map_v' in path.replace(os.sep, '/'):
            print('  %-58s archived snapshot, left alone' % os.path.basename(path))
            continue
        z = float(old.group(1).split()[2])
        # --force targets a world whose pose was set by hand (so it is not in the
        # roof band the guard looks for) but is still unusable -- e.g. z = 2.0
        # pitched straight down, which frames 2.3 m of a 41 m floor.
        if not force and not ROOF_BOTTOM <= z <= ROOF_TOP:
            print('  %-58s z=%.3f is not inside the roof, left alone'
                  % (os.path.basename(path), z))
            continue
        print('  %-58s %s -> %s' % (os.path.basename(path), old.group(1), want))
        if not dry:
            open(path, 'w').write(new)
    if dry:
        print('\n(dry run -- nothing written)')


if __name__ == '__main__':
    main()
