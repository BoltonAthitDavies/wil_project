#!/usr/bin/env python3
"""Turn a dynamic warehouse world into its static twin.

    python3 script/make_static_world.py SRC.world DST.world

Every active <include> except the robot gets <static>true</static> (includes
already carrying one are left alone, commented-out ones are ignored), and the
LogisticsScheduler / KinematicTrajectory world plugin blocks and the walking
people (<actor> elements) are removed so nothing moves. Everything else (shell, lights, GUI, robot) is copied as is.
"""
import re
import sys

MOVERS = ('LogisticsScheduler', 'KinematicTrajectory')


def make_static(src):
    out, n_tagged, n_plugins, n_actors = [], 0, 0, 0
    # the generator's note about the traffic plugin no longer applies
    src = re.sub(r'    <!-- Traffic: carriers.*?-->\n\n?', '', src, flags=re.S)
    lines = src.splitlines(keepends=True)
    i, in_comment = 0, False
    while i < len(lines):
        line = lines[i]
        # drop the mover plugin blocks (world-level, not the GUI ones)
        if re.search(r'<plugin filename="(%s)"' % '|'.join(MOVERS), line):
            while '</plugin>' not in lines[i]:
                i += 1
            i += 1
            n_plugins += 1
            continue
        # drop the walking people (<actor> elements): nothing moves in a static twin
        if line.strip().startswith('<actor '):
            while '</actor>' not in lines[i]:
                i += 1
            i += 1
            n_actors += 1
            continue
        if line.strip().startswith('<!-- ==== people') or re.match(r'\s*<!-- H\d: ', line):
            i += 1
            continue
        out.append(line)
        if '<!--' in line and '-->' not in line:
            in_comment = True
        if '-->' in line:
            in_comment = False
        if not in_comment and line.strip() == '<include>':
            block = []
            i += 1
            while lines[i].strip() != '</include>':
                block.append(lines[i])
                i += 1
            text = ''.join(block)
            if 'ackermann_robot' not in text and '<static>' not in text:
                block.append('      <static>true</static>\n')
                n_tagged += 1
            out.extend(block)
            out.append(lines[i])
        i += 1
    return ''.join(out), n_tagged, n_plugins, n_actors


if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    text, n_tagged, n_plugins, n_actors = make_static(open(src).read())
    open(dst, 'w').write(text)
    print('%s: tagged %d includes static, removed %d mover plugin block(s) and %d actor(s)' % (dst, n_tagged, n_plugins, n_actors))
