#!/usr/bin/env python3
"""Refuse to start a simulation run while a previous one is still alive.

    python3 script/sim_guard.py            # report
    python3 script/sim_guard.py --kill     # SIGINT, then SIGKILL what remains

WHY THIS EXISTS
    sweep_M4, 12 Nov 2026: 28 localisation poses with good covariance (0.0056)
    and ZERO map->odom transforms over 34 s, where M1 and M3 produced 202 and 189
    and had their first transform within 0.2 s of their first pose. The run was
    not short of data and RTAB-Map was not failing to localise.

    A Gazebo from an aborted run 21 minutes earlier was still alive. Its own
    successor said so:

        [ign gazebo-1] [Wrn] [SimulationRunner.cc:695] Found additional
        publishers on /clock, using namespaced clock topic only

    Two /clock publishers make sim time jump, and a TF buffer asked for a
    transform it then judges too old or in the future returns nothing. The pose
    topic does not care -- it is published, not looked up -- so the symptom is
    exactly what M4 shows: poses fine, transforms absent.

    The process is 'ruby', not 'ign' or 'gazebo': ign is a ruby script, so
    `pgrep -x ign` finds nothing and reports a clean machine. That is how the
    leftover survived a cleanup pass that claimed to have removed it.

NEVER pkill -f
    A pattern match on the command line matches this script's own command line,
    and the shell that launched it. Match process NAMES exactly (pgrep -x) and
    signal pids.
"""
import argparse
import os
import signal
import subprocess
import sys
import time

# Exact process names, as pgrep -x matches them. 'ruby' is ign gazebo.
NAMES = ['ruby', 'rtabmap', 'orbslam3_node', 'vins_node', 'parameter_bridge',
         'image_bridge', 'rviz2', 'static_transform_publisher']


def running():
    """{name: [pid, ...]} for everything from a simulation run that is still up."""
    out = {}
    for n in NAMES:
        r = subprocess.run(['pgrep', '-x', n], capture_output=True, text=True)
        pids = [int(p) for p in r.stdout.split()]
        if pids:
            out[n] = pids
    return out


def describe(procs):
    lines = []
    for name, pids in procs.items():
        for pid in pids:
            r = subprocess.run(['ps', '-o', 'etime=,cmd=', '-p', str(pid)],
                               capture_output=True, text=True)
            info = ' '.join(r.stdout.split())[:100]
            lines.append(f'    {name:26} pid {pid:<8} {info}')
    return lines


def kill(procs, wait=4.0):
    for name, pids in procs.items():
        for pid in pids:
            try:
                os.kill(pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    time.sleep(wait)
    left = running()
    for name, pids in left.items():
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    time.sleep(1.0)
    return running()


def require_clean(auto_kill=False):
    """Call this before launching. Returns True when the machine is clear.

    A leftover does not produce an error message in the new run -- it produces
    quietly wrong data -- so this refuses rather than warns.
    """
    procs = running()
    if not procs:
        return True
    print('[sim_guard] a previous simulation run is still alive:')
    for line in describe(procs):
        print(line)
    if not auto_kill:
        print('\n[sim_guard] refusing to start. Two /clock publishers make sim time')
        print('            jump and TF lookups return nothing, which reads as a')
        print('            localisation failure rather than as this. Clear it with:')
        print('                python3 script/sim_guard.py --kill')
        return False
    print('[sim_guard] clearing it')
    left = kill(procs)
    if left:
        print('[sim_guard] could not clear:')
        for line in describe(left):
            print(line)
        return False
    print('[sim_guard] clear')
    return True


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kill', action='store_true')
    a = ap.parse_args()
    procs = running()
    if not procs:
        print('[sim_guard] clear: nothing from a simulation run is up')
        sys.exit(0)
    print('[sim_guard] still alive:')
    for line in describe(procs):
        print(line)
    if a.kill:
        left = kill(procs)
        print('[sim_guard] ' + ('clear' if not left else 'STILL UP: ' + str(left)))
        sys.exit(0 if not left else 1)
    sys.exit(1)
