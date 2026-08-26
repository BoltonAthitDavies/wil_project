#!/usr/bin/env python3
"""Car-style keyboard driving for the ackermann robot: throttle pedal + steering wheel.

    python3 ~/wil_project/drive.py

Run it in its own terminal, with the simulation already up. It needs a real TTY,
so use `python3`, not `ros2 launch` -- launch hands its children a pipe for stdin
and the node would never see a keystroke.

WHY THIS EXISTS
    /cmd_vel takes a Twist whose angular.z is a YAW RATE. Hold one key and the
    steering angle you actually get changes with speed, which is nothing like a
    car. Gazebo Fortress's AckermannSteering has no steering-angle input (no
    steer_angle topic, no steering_only mode -- those arrived in Garden), so the
    fix is here: keep the steering ANGLE as state and convert every tick with

        angular.z = v * tan(steer) / wheelbase

    which is the Ackermann bicycle model. The wheel angle is then constant and
    speed-independent, the way a steering wheel behaves.

    Throttle is integrated rather than commanded: holding the key accelerates,
    releasing coasts down through drag, and brake decelerates harder. So what you
    publish is still a speed setpoint, but it now has momentum behind it.

CONTROLS
    w / up      throttle      s / down    brake, then reverse
    a / left    steer left    d / right   steer right
    space       handbrake     q / Ctrl-C  quit (sends a stop first)

KEY-HOLD CAVEAT
    A terminal in raw mode reports key PRESSES, never releases, so "held" is
    inferred from the keyboard's auto-repeat: a key counts as held while a repeat
    arrived within HOLD_WINDOW. X11 waits ~500 ms before repeating, which can feel
    like a stutter right after you press. If that bothers you, shorten the delay:
        xset r rate 200 40
    or raise HOLD_WINDOW below at the cost of a longer coast after release.
"""

import math
import os
import select
import sys
import termios
import time
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.parameter import Parameter

# --- vehicle constants, must match models/ackermann_robot/model.sdf -------------
WHEEL_BASE = 0.42          # <wheel_base>
STEER_LIMIT = 35 * math.pi / 180  # <steering_limit>, 35 degrees

# --- feel; tune these ----------------------------------------------------------
THROTTLE_ACCEL = 0.75       # m/s^2 while throttle held. Kept just under the model's
                           # <max_acceleration>3</max_acceleration> so the plugin's
                           # own limiter is not the binding constraint. Raise both
                           # together (launch arg max_accel) for a punchier car.
BRAKE_DECEL = 6.0          # m/s^2 while brake held
DRAG_DECEL = 0.8           # m/s^2 when neither is held, i.e. coasting
TOP_SPEED = 1.5            # m/s. Independent of the launch arg max_speed, which
                           # clamps at the plugin; this is what the pedal can reach.
REVERSE_SPEED = 1.5        # m/s cap when going backwards

STEER_RATE = 0.6           # rad/s while a steer key is held
CENTER_RATE = 2.4          # rad/s of self-centring when it is not

HOLD_WINDOW = 0.40         # s; see KEY-HOLD CAVEAT above
TICK = 0.02                # s, 50 Hz output

KEYS = {
    'throttle': ('w', '\x1b[A'),
    'brake':    ('s', '\x1b[B'),
    'left':     ('a', '\x1b[D'),
    'right':    ('d', '\x1b[C'),
}


def read_keys(deadline):
    """Collect key tokens until `deadline` (a time.monotonic value).

    Blocking until the deadline is what paces the whole loop at 1/TICK. An
    earlier version passed a fixed timeout instead, so a held key kept stdin
    readable and select never blocked -- the loop then free-ran at ~1400 Hz and
    flooded /cmd_vel. Always drain to a deadline, never to a timeout.
    """
    seen = set()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return seen
        if not select.select([sys.stdin], [], [], remaining)[0]:
            continue
        ch = os.read(sys.stdin.fileno(), 1).decode('utf-8', 'ignore')
        if ch == '\x1b':                    # escape sequence: arrow keys
            rest = ''
            while select.select([sys.stdin], [], [], 0.001)[0] and len(rest) < 2:
                rest += os.read(sys.stdin.fileno(), 1).decode('utf-8', 'ignore')
            seen.add('\x1b' + rest)
        else:
            seen.add(ch)


def main():
    rclpy.init()
    # use_sim_time matters more here than it looks. The pedal integrates speed over
    # dt, and the robot it controls lives in SIM time. Left on the wall clock, the
    # pedal runs 1/RTF times too fast -- at RTF 0.18 the coast-down wipes out 4 m/s
    # in well under a second of simulated time, so touching any other key reads as
    # "the robot stopped". Physics is integrated on the sim clock; key-hold
    # detection stays on the wall clock, because that is about the keyboard.
    node = rclpy.create_node(
        'car_drive_teleop',
        parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
    pub = node.create_publisher(Twist, '/cmd_vel', 10)
    clock = node.get_clock()

    speed = 0.0
    steer = 0.0
    last_press = {name: -1e9 for name in KEYS}
    settings = termios.tcgetattr(sys.stdin)
    print(__doc__.split('CONTROLS')[1].split('KEY-HOLD')[0].strip())
    print()

    try:
        tty.setraw(sys.stdin.fileno())
        prev = None
        warned = False
        next_tick = time.monotonic()
        while True:
            next_tick += TICK
            keys = read_keys(next_tick)
            # Servicing the node is what makes use_sim_time work: the clock is fed
            # by a /clock subscription, and an un-spun node never receives it --
            # clock.now() then sits at 0 forever and the fallback below silently
            # puts you back on the wall clock. Cheap, non-blocking, do not remove.
            rclpy.spin_once(node, timeout_sec=0.0)
            now = time.monotonic()                 # WALL: key-hold detection
            sim = clock.now().nanoseconds * 1e-9   # SIM: physics integration
            if sim <= 0.0:                         # no /clock yet: run standalone
                sim = now
                if not warned:
                    warned = True
                    sys.stderr.write(
                        '\r\n[drive] no /clock yet -- running on the wall clock. '
                        'Pedal feel will be 1/RTF too fast until the sim is up.\r\n')
            if prev is None:
                prev = sim
            dt = min(max(sim - prev, 0.0), 0.2)    # clamp: sim time can jump
            prev = sim

            if 'q' in keys or '\x03' in keys:
                break
            for name, aliases in KEYS.items():
                if any(a in keys for a in aliases):
                    last_press[name] = now
            held = {n: (now - t) < HOLD_WINDOW for n, t in last_press.items()}

            if ' ' in keys:                        # handbrake
                speed = 0.0
                held['throttle'] = held['brake'] = False

            # --- throttle pedal -------------------------------------------------
            if held['throttle']:
                speed += THROTTLE_ACCEL * dt
            elif held['brake']:
                speed -= BRAKE_DECEL * dt
            else:                                  # coast
                drag = DRAG_DECEL * dt
                speed = max(0.0, speed - drag) if speed > 0 else min(0.0, speed + drag)
            speed = max(-REVERSE_SPEED, min(TOP_SPEED, speed))

            # --- steering wheel, self-centring ----------------------------------
            if held['left']:
                steer += STEER_RATE * dt
            elif held['right']:
                steer -= STEER_RATE * dt
            else:
                center = CENTER_RATE * dt
                steer = max(0.0, steer - center) if steer > 0 else min(0.0, steer + center)
            steer = max(-STEER_LIMIT, min(STEER_LIMIT, steer))

            # --- Ackermann bicycle model: angle -> yaw rate ----------------------
            tw = Twist()
            tw.linear.x = speed
            tw.angular.z = speed * math.tan(steer) / WHEEL_BASE
            pub.publish(tw)

            bar = lambda v, lo, hi, n=11: (
                '[' + ''.join('#' if i == round((v - lo) / (hi - lo) * (n - 1)) else '-'
                              for i in range(n)) + ']')
            sys.stdout.write(
                '\r speed {:+5.2f} m/s {}   steer {:+6.1f} deg {}   yaw {:+5.2f} rad/s  '
                .format(speed, bar(speed, -REVERSE_SPEED, TOP_SPEED),
                        math.degrees(steer), bar(steer, -STEER_LIMIT, STEER_LIMIT),
                        tw.angular.z))
            sys.stdout.flush()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        stop = Twist()
        for _ in range(5):                         # make sure the stop lands
            pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()
        print('\nstopped.')


if __name__ == '__main__':
    main()
