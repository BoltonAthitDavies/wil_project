#!/usr/bin/env python3
"""Check that everything which believes it knows the robot's limits still agrees.

    python3 ~/wil_project/script/check_limits.py            # exits non-zero on any FAIL

WHY THIS EXISTS
    model.sdf is the only place the robot is actually defined. Four other files
    carry hand-copied consequences of it, and NOTHING connects them:

        models/ackermann_robot/model.sdf     wheel_base, steering_limit, caps
          -> drive.py                        WHEEL_BASE, STEER_LIMIT
          -> viewer.py                       ROBOT_X0/X1/HALF_W (teleop via drive)
          -> params/nav2_ackermann.yaml      turning radius, lookahead, footprint
          -> launch/small_warehouse.launch.py  max_speed / max_accel defaults

    Every one of those drifts silently. Worse, the two failure modes look nothing
    like a configuration error:

      * AckermannSteering does not reject an infeasible Twist. It clamps the
        turning radius internally and the robot quietly under-turns, so a too-small
        minimum_turning_radius shows up as "nav2 keeps missing corners", not as an
        error in any log.
      * The launch file's max_speed / max_accel work by REWRITING a copy of
        model.sdf before gz parses it. If the launch defaults and the file disagree,
        every run silently shadows the model with a patched copy.

    So this asserts the relations rather than the values. Deriving the numbers
    outright would be wrong: minimum_turning_radius 0.75 against a geometric 0.6355
    is a deliberate margin, not a stale copy, and a checker that "fixed" it would
    be destroying engineering judgement. Each check below therefore states the
    direction it cares about, and margins are reported, not flattened.

THE GEOMETRY, ONCE
    Bicycle model, steering angle d, wheelbase L. The REAR AXLE traces
        R_rear = L / tan(d)
    but nav2 plans base_link, which sits mid-wheelbase and sweeps the larger
        R_base = sqrt(R_rear^2 + (L/2)^2)
    and the curvature bound that /cmd_vel must respect is
        |angular.z| / |linear.x|  <=  k_max = tan(d) / L
    Both bounds are independent of speed. Mixing R_rear and R_base is the classic
    error here -- they differ by 6% and only one of them is what nav2 plans.
"""

import ast
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PKG = os.path.join(ROOT, 'aws-robomaker-small-warehouse-world')
SDF = os.path.join(PKG, 'models', 'ackermann_robot', 'model.sdf')
NAV2 = os.path.join(PKG, 'params', 'nav2_ackermann.yaml')
LAUNCH = os.path.join(PKG, 'launch', 'small_warehouse.launch.py')
DRIVE = os.path.join(HERE, 'drive.py')
VIEWER = os.path.join(HERE, 'viewer.py')


# --- readers -------------------------------------------------------------------

def sdf_plugin_params(path):
    """AckermannSteering's own parameters, read out of the plugin block only.

    Scoped to the block deliberately: <max_velocity> also appears on joints, and a
    document-wide search would pick up whichever came first.
    """
    text = open(path).read()
    m = re.search(r'<plugin[^>]*AckermannSteering.*?</plugin>', text, re.S)
    if not m:
        raise SystemExit('%s: no AckermannSteering plugin block' % path)
    block = m.group(0)
    out = {}
    for tag in ('wheel_base', 'steering_limit', 'wheel_separation', 'kingpin_width',
                'wheel_radius', 'max_velocity', 'max_acceleration'):
        v = re.search(r'<%s>([^<]+)</%s>' % (tag, tag), block)
        if not v:
            raise SystemExit('%s: AckermannSteering has no <%s>' % (path, tag))
        out[tag] = float(v.group(1))
    return out


def sdf_wheel_length(path):
    """Length of the wheel cylinder, i.e. how far the tyres stick out sideways."""
    text = open(path).read()
    m = re.search(r"<link name='front_left_wheel'>.*?<cylinder>.*?"
                  r'<length>([^<]+)</length>', text, re.S)
    if not m:
        raise SystemExit('%s: cannot find the front_left_wheel cylinder' % path)
    return float(m.group(1))


def nav2_params(path):
    import yaml
    doc = yaml.safe_load(open(path))

    def at(*keys):
        node = doc
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                raise SystemExit('%s: missing %s' % (path, '.'.join(keys)))
            node = node[k]
        return node

    ctrl = at('controller_server', 'ros__parameters', 'FollowPath')
    smoo = at('velocity_smoother', 'ros__parameters')
    plan = at('planner_server', 'ros__parameters', 'GridBased')
    glob = at('global_costmap', 'global_costmap', 'ros__parameters')
    return {
        'minimum_turning_radius': float(plan['minimum_turning_radius']),
        'motion_model': str(plan['motion_model_for_search']).upper(),
        'min_lookahead_dist': float(ctrl['min_lookahead_dist']),
        'desired_linear_vel': float(ctrl['desired_linear_vel']),
        'allow_reversing': bool(ctrl['allow_reversing']),
        'use_rotate_to_heading': bool(ctrl['use_rotate_to_heading']),
        'smoother_max_vel': [float(x) for x in smoo['max_velocity']],
        'smoother_max_accel': [float(x) for x in smoo['max_accel']],
        'footprint': ast.literal_eval(glob['footprint']),
    }


def py_const(path, name):
    """Evaluate a module-level `NAME = <expr>` without importing the module.

    Importing viewer.py would pull in rclpy and PyQt5 for two numbers, and would
    fail on a machine that has neither -- which is exactly when you most want to be
    able to run this check. Parsed rather than regexed because these constants are
    written both ways: `WHEEL_BASE = 0.42` but `ROBOT_X0, ROBOT_X1 = -0.300, 0.506`,
    and the values are expressions (`35 * math.pi / 180`), not literals.
    """
    tree = ast.parse(open(path).read())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                value = node.value
            elif (isinstance(target, ast.Tuple)
                  and isinstance(node.value, ast.Tuple)
                  and any(isinstance(e, ast.Name) and e.id == name
                          for e in target.elts)):
                idx = [getattr(e, 'id', None) for e in target.elts].index(name)
                value = node.value.elts[idx]
            else:
                continue
            return float(eval(compile(ast.Expression(value), '<const>', 'eval'),
                              {'math': math, '__builtins__': {}}))
    raise SystemExit('%s: no top-level %s' % (path, name))


def launch_default(path, arg):
    src = open(path).read()
    m = re.search(r"'%s',\s*default_value='([^']+)'" % re.escape(arg), src)
    if not m:
        raise SystemExit('%s: no DeclareLaunchArgument %r' % (path, arg))
    return float(m.group(1))


# --- check plumbing ------------------------------------------------------------

RESULTS = []


def check(name, ok, detail):
    RESULTS.append((bool(ok), name, detail))


def ge(name, got, need, unit='', why=''):
    check(name, got >= need - 1e-9,
          '%.4f%s >= %.4f%s  (margin %+.1f%%)%s'
          % (got, unit, need, unit,
             100.0 * (got - need) / need if need else 0.0,
             '  -- ' + why if why else ''))


def le(name, got, limit, unit='', why=''):
    check(name, got <= limit + 1e-9,
          '%.4f%s <= %.4f%s  (headroom %+.1f%%)%s'
          % (got, unit, limit, unit,
             100.0 * (limit - got) / limit if limit else 0.0,
             '  -- ' + why if why else ''))


def eq(name, got, want, unit='', tol=1e-6, why=''):
    check(name, abs(got - want) <= tol,
          '%.4f%s == %.4f%s%s' % (got, unit, want, unit,
                                  '  -- ' + why if why else ''))


def main():
    sdf = sdf_plugin_params(SDF)
    wheel_len = sdf_wheel_length(SDF)
    nav = nav2_params(NAV2)

    L = sdf['wheel_base']
    d = sdf['steering_limit']
    r_rear = L / math.tan(d)
    r_base = math.hypot(r_rear, L / 2.0)
    k_max = math.tan(d) / L
    half_w = sdf['wheel_separation'] / 2.0 + wheel_len / 2.0

    print(__doc__.split('THE GEOMETRY, ONCE')[0].strip().splitlines()[0])
    print()
    print('model.sdf: wheel_base %.3f m, steering_limit %.4f rad (%.1f deg),'
          % (L, d, math.degrees(d)))
    print('           cap %.1f m/s, accel %.1f m/s^2, track %.3f m + wheel %.3f m'
          % (sdf['max_velocity'], sdf['max_acceleration'],
             sdf['wheel_separation'], wheel_len))
    print('derived:   R_rear %.4f m, R_base %.4f m, k_max %.4f /m, half-width %.4f m'
          % (r_rear, r_base, k_max, half_w))
    print()

    # -- nav2 planner ----------------------------------------------------------
    ge('nav2 minimum_turning_radius covers base_link',
       nav['minimum_turning_radius'], r_base, ' m',
       'nav2 plans base_link, not the rear axle')
    want_model = 'REEDS_SHEPP' if nav['allow_reversing'] else 'DUBIN'
    check('nav2 motion model matches allow_reversing',
          nav['motion_model'] == want_model,
          '%s with allow_reversing=%s (expected %s)'
          % (nav['motion_model'], nav['allow_reversing'], want_model))
    check('nav2 use_rotate_to_heading is off',
          not nav['use_rotate_to_heading'],
          'a car cannot rotate on the spot; true commands w at v=0, i.e. k -> inf')

    # -- nav2 controller -------------------------------------------------------
    ge('RPP min_lookahead_dist keeps emitted curvature feasible',
       nav['min_lookahead_dist'], 2.0 * r_base, ' m',
       'RPP emits k=2/L_d at worst, needs k <= 1/R_base')
    le('desired_linear_vel within the smoother cap',
       nav['desired_linear_vel'], nav['smoother_max_vel'][0], ' m/s')

    # -- nav2 smoother vs the curvature bound ----------------------------------
    le('smoother yaw cap respects the curvature bound',
       nav['smoother_max_vel'][2], k_max * nav['smoother_max_vel'][0], ' rad/s',
       '|wz| <= k_max*|vx| = %.4f' % (k_max * nav['smoother_max_vel'][0]))

    # -- nav2 vs the plugin's own caps -----------------------------------------
    # The bound above says the smoother never DEMANDS more curvature than the robot
    # can do. This says the opposite thing, and it is the one that bites: the
    # smoother must be able to EXECUTE the tightest turn the planner is allowed to
    # plan. max_velocity[2]/max_velocity[0] is the sharpest curvature that can
    # survive the smoother, and scale_velocities preserves the ratio by scaling both
    # components -- so a yaw cap that is low relative to the speed cap silently
    # raises the effective turning radius and the robot cannot track its own plan.
    ge('smoother can execute the planner tightest turn',
       nav['smoother_max_vel'][2] / nav['smoother_max_vel'][0],
       1.0 / nav['minimum_turning_radius'], ' /m',
       'else effective radius is %.2f m against a planned %.2f m'
       % (nav['smoother_max_vel'][0] / nav['smoother_max_vel'][2],
          nav['minimum_turning_radius']))

    le('smoother speed within the plugin cap',
       nav['smoother_max_vel'][0], sdf['max_velocity'], ' m/s',
       'the plugin clamps silently above this')
    le('smoother accel within the plugin cap',
       nav['smoother_max_accel'][0], sdf['max_acceleration'], ' m/s^2')

    # -- footprint -------------------------------------------------------------
    fp = nav['footprint']
    fp_half_w = max(abs(y) for _, y in fp)
    ge('costmap footprint covers the wheel track',
       fp_half_w, half_w, ' m', 'tyres stick out past the chassis')

    # -- the other hand-copies -------------------------------------------------
    eq('drive.py WHEEL_BASE matches model.sdf',
       py_const(DRIVE, 'WHEEL_BASE'), L, ' m')
    eq('drive.py STEER_LIMIT matches model.sdf',
       py_const(DRIVE, 'STEER_LIMIT'), d, ' rad', tol=1e-4)
    eq('viewer.py ROBOT_HALF_W matches the wheel track',
       py_const(VIEWER, 'ROBOT_HALF_W'), half_w, ' m', tol=1e-4)
    vx0, vx1 = py_const(VIEWER, 'ROBOT_X0'), py_const(VIEWER, 'ROBOT_X1')
    fx0, fx1 = min(x for x, _ in fp), max(x for x, _ in fp)
    ge('costmap footprint covers viewer.py front reach', fx1, vx1, ' m')
    le('costmap footprint covers viewer.py rear reach', fx0, vx0, ' m')

    # -- launch defaults vs the file they rewrite ------------------------------
    eq('launch max_speed default matches model.sdf',
       launch_default(LAUNCH, 'max_speed'), sdf['max_velocity'], ' m/s',
       why='a mismatch shadows model.sdf with a patched copy every run')
    eq('launch max_accel default matches model.sdf',
       launch_default(LAUNCH, 'max_accel'), sdf['max_acceleration'], ' m/s^2',
       why='same rewrite path')

    width = max(len(n) for _, n, _ in RESULTS)
    bad = 0
    for ok, name, detail in RESULTS:
        if not ok:
            bad += 1
        print('%s  %-*s  %s' % ('PASS' if ok else 'FAIL', width, name, detail))
    print()
    if bad:
        print('%d of %d checks FAILED.' % (bad, len(RESULTS)))
        print('These are relations, not opinions: a FAIL means some file believes '
              'something about the robot that model.sdf does not support.')
    else:
        print('all %d checks passed.' % len(RESULTS))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
