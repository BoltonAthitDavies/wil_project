#!/usr/bin/env python3
"""Where the walking people are at a given simulation time.

    import people_script
    people = people_script.load_actors(world_path)
    for a in people:
        x, y, yaw, clip = people_script.pose_at(a, sim_t)

WHY THIS EXISTS
    The people in the realistic worlds are gz-sim <actor>s on scripted routes.
    gz-sim animates them on the RENDERING side and never writes the animated
    pose back into the entity state, so the bridged /world/default/pose/info
    feed reports every person at the pose it started at, for the whole run.
    Nothing on the ROS side knows where a person is.

    The script is deterministic, though: an actor's pose is a pure function of
    simulation time and the <script> in the world file. This module evaluates
    that function, so the 2-D viewer can draw the people and a detector
    ground truth can be projected for them, both from the world file and the
    simulation clock alone.

WHAT IS REPRODUCED, AND HOW CLOSELY
    gz-sim (SceneManager::CreateActor / RenderUtil) plays the actor's
    trajectories in order; each trajectory's waypoint times are relative to
    its own start, the trajectories are chained end to end, the whole script
    is delayed by <delay_start> and, with <loop>, repeats from the start once
    the last trajectory ends. Position between waypoints is a spline in gz
    (tension per trajectory); here it is linear. The generator writes a
    waypoint every 2 m along a leg, so the two differ by well under 0.2 m,
    which is less than a person's width. Yaw is interpolated the short way.

    The first waypoint of the first trajectory is where a person stands
    before the delay expires and where the script starts each loop.
"""
import math
import xml.etree.ElementTree as ET


class Actor(object):
    __slots__ = ('name', 'delay', 'loop', 'trajs', 'total')

    def __init__(self, name, delay, loop, trajs):
        self.name, self.delay, self.loop = name, delay, loop
        # trajs: [(clip, start_offset, [(t, x, y, yaw), ...]), ...], t relative to
        # the trajectory, offsets cumulative over the script
        self.trajs = []
        t0 = 0.0
        for clip, wps in trajs:
            self.trajs.append((clip, t0, wps))
            t0 += wps[-1][0] if wps else 0.0
        self.total = t0


def _pose6(text):
    v = [float(p) for p in text.split()]
    return v[0], v[1], v[5]


def load_actors(world_path):
    """Every <actor> in the world file with a scripted trajectory."""
    root = ET.parse(world_path).getroot()
    out = []
    for actor in root.iter('actor'):
        script = actor.find('script')
        if script is None:
            continue
        delay = float((script.findtext('delay_start') or '0').strip())
        loop = (script.findtext('loop') or 'true').strip().lower() in ('true', '1')
        trajs = []
        for traj in script.findall('trajectory'):
            wps = []
            for wp in traj.findall('waypoint'):
                x, y, yaw = _pose6(wp.findtext('pose'))
                wps.append((float(wp.findtext('time')), x, y, yaw))
            if wps:
                wps.sort(key=lambda w: w[0])
                trajs.append((traj.get('type', ''), wps))
        if trajs:
            out.append(Actor(actor.get('name', 'actor'), delay, loop, trajs))
    return out


def _lerp_yaw(a, b, f):
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + d * f


def pose_at(actor, sim_t):
    """(x, y, yaw, clip) of the actor at simulation time sim_t (seconds)."""
    clip0, _, wps0 = actor.trajs[0]
    t = sim_t - actor.delay
    if t <= 0.0 or actor.total <= 0.0:
        _, x, y, yaw = wps0[0]
        return x, y, yaw, clip0
    if actor.loop:
        t = math.fmod(t, actor.total)
    elif t >= actor.total:
        clip, _, wps = actor.trajs[-1]
        _, x, y, yaw = wps[-1]
        return x, y, yaw, clip
    for clip, t0, wps in actor.trajs:
        if t < t0 or t > t0 + wps[-1][0]:
            continue
        tt = t - t0
        for (ta, xa, ya, ya_w), (tb, xb, yb, yb_w) in zip(wps, wps[1:]):
            if ta <= tt <= tb:
                f = 0.0 if tb <= ta else (tt - ta) / (tb - ta)
                return (xa + f * (xb - xa), ya + f * (yb - ya),
                        _lerp_yaw(ya_w, yb_w, f), clip)
        _, x, y, yaw = wps[-1]
        return x, y, yaw, clip
    _, x, y, yaw = wps0[0]
    return x, y, yaw, clip0


if __name__ == '__main__':
    import sys
    world = sys.argv[1]
    times = [float(v) for v in sys.argv[2:]] or [0.0, 10.0, 30.0, 60.0]
    for a in load_actors(world):
        print('%s  delay %.1f s  loop %s  %d trajectories, period %.1f s'
              % (a.name, a.delay, a.loop, len(a.trajs), a.total))
        for t in times:
            x, y, yaw, clip = pose_at(a, t)
            print('   t=%6.1f  (%7.2f, %7.2f)  yaw %6.2f  %s' % (t, x, y, yaw, clip))
