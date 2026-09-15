#!/usr/bin/env python3
"""Bake an exact nav2 occupancy map from the warehouse world's COLLISION meshes.

    python3 ~/wil_project/script/bake_map.py --preview     # look before you leap
    python3 ~/wil_project/script/bake_map.py               # writes maps/baked/

WHY THIS EXISTS
    Nav2 needs an occupancy grid. The obvious sources are both wrong:

      * maps/002 is in some other SLAM frame entirely -- it does not even contain
        the whole floor (stops 0.6 m short in +y) and wastes 13.7 m past the east
        wall.
      * maps/005 IS world-aligned and is the right resolution/origin, but it is a
        SLAM product: the shelves appear as `#.....#` post outlines because a lidar
        at one height only ever saw the legs. Planning on it sends the robot into
        0.94 m aisles it cannot turn around in.

    The world file knows the truth. Every obstacle is an <include> with a pose, and
    every model has a watertight collision mesh. So we slice the actual collision
    geometry at robot height and rasterize it. The result is exact, regenerates when
    the world changes, and -- unlike a SLAM map -- has no unknown space at all.

THE ALGORITHM, AND WHY THE OBVIOUS ONE FAILS
    The natural approach is "keep every triangle whose z-range overlaps the band,
    fillPoly its xy projection". It does not work, and it fails QUIETLY.

    These meshes are closed boxes. A tall crate's SIDE faces are vertical, so their
    xy projection is a degenerate line with no area. Its only faces with area (the
    top and bottom) lie outside the band. So you get hollow outlines: a 1-px rim
    where the sides were, nothing inside. The wall ring still looks plausible --
    which is exactly what makes it dangerous -- but every crate is empty and the
    planner routes straight through it.

    Instead, cast a vertical ray up through each cell centre and use the even-odd
    rule. The insight that makes this both cheap and exact is that you never need to
    sample z at all:

        a solid intersects the band [z_lo, z_hi] at a cell
        IFF it is solid AT z_lo, OR some surface crossing lies in (z_lo, z_hi]

    so two accumulators per cell suffice -- parity of crossings above z_lo, and a
    count of crossings inside the band. Vertical faces contribute no crossing, which
    is correct, and they drop out naturally as a near-zero 2D cross product.

    Accumulate PER MESH INSTANCE (each is separately watertight) and OR the results;
    accumulating parity across two overlapping solids would cancel them out.

TRANSFORM CHAIN -- THE ORDER IS LOAD-BEARING
        p_raw -> M_node @ p_raw -> * unit_meter -> * mesh_scale
              -> + collision_pose -> + link_pose -> Rz(yaw) @ p + include_xyz

    The <node><matrix> translation column is in ASSET units (centimetres here), so
    the unit scale must multiply the RESULT of the matrix, not the vertices going in.
    Get this backwards and RoofB lands at z = 0.99 m instead of 9.9 m -- i.e. inside
    the band, and your map is 100% occupied.

    Note the node matrix is also what carries the axis permutation. ShelfD/ShelfE,
    ClutteringD and RoofB all declare <up_axis>Z_UP</up_axis> while their raw vertex
    data is plainly Y-up; their node matrix rotates it. So there is NO need for a
    per-model axis table -- just apply the matrix and everything lands Z-up.

WHAT ENDS UP IN THE MAP, AND WHY
    Shelves are SOLID, deliberately. The robot is 0.454 m tall and the shelf deck
    starts at 0.735 m, so it physically fits underneath -- but the gap between two
    shelf banks is 0.94 m wide and 3.9 m long, against a 0.6355 m turning radius.
    It could drive in and would never turn around. It is also a stereo-VIO testbed,
    and the one thing you do not want autonomy to seek out is 0.25 m of featureless
    ceiling. --allow-underpass gives you the other behaviour.

    The four sphere markers ARE obstacles. Their collision radius is 0.25 m, 2.5x
    the 0.1 m visual, and three of them sit on the open floor. Leave them out and
    nav2 plans through them, the robot wedges against an invisible ball, and the
    progress checker aborts with no clue why. If you would rather they were not
    there, delete the four <include> blocks in small_warehouse.world -- the map and
    the simulator have to agree, and the world file is the one that is authoritative.

    Free cells not reachable from the spawn are painted OCCUPIED, not unknown. That
    is the thin strip outside the wall ring where the grid overhangs the building.
    The map then contains no unknown space at all, which takes allow_unknown and
    track_unknown_space off the list of things that can be misconfigured.
"""

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

# --- defaults ------------------------------------------------------------------
# Deliberately identical to maps/005/map.yaml so the baked map and the SLAM map
# overlay pixel-for-pixel -- that is how you eyeball whether the bake is right.
DEF_RES = 0.05
DEF_ORIGIN = (-7.0, -10.5)
DEF_SIZE = (286, 423)

# 0.05 clears the floor's top face (+0.034) without z-fighting it; 2.00 excludes the
# roof (starts at 9.02) with 7 m to spare while keeping shelf decks and crate tops.
DEF_ZMIN, DEF_ZMAX = 0.05, 2.00

# The floor and roof span the entire grid, so they would fill it solid if their
# z-extent ever overlapped the band. The band already excludes them; skipping by
# name as well is a cheap guard against someone widening it.
DEF_SKIP = ['ackermann_robot', 'GroundB', 'RoofB']

DEF_SEED = (1.8, 9.0)          # the robot's spawn pose, from the world file

COLLADA_NS = '{http://www.collada.org/2005/11/COLLADASchema}'


def _floats(text):
    return [float(v) for v in text.split()]


# --- world file ----------------------------------------------------------------

def load_world(path):
    """Return [(model_name, instance_name, (x, y, z, roll, pitch, yaw)), ...].

    ElementTree drops XML comments for free, which is exactly what we want: it
    means the three commented-out DeskC includes are excluded without any special
    handling. A regex over the raw text would wrongly pick them up.
    """
    root = ET.parse(path).getroot()
    world = root.find('world')
    if world is None:
        raise SystemExit('%s has no <world> element' % path)

    out = []
    for inc in world.findall('include'):
        uri = inc.findtext('uri', '').strip()
        if not uri.startswith('model://'):
            continue
        model = uri[len('model://'):].strip('/')
        name = (inc.findtext('name') or model).strip()
        pose = _floats(inc.findtext('pose') or '0 0 0 0 0 0')
        pose += [0.0] * (6 - len(pose))
        out.append((model, name, tuple(pose[:6])))
    return out


def _pose_of(elem):
    txt = elem.findtext('pose') if elem is not None else None
    if not txt:
        return (0.0,) * 6
    v = _floats(txt)
    v += [0.0] * (6 - len(v))
    return tuple(v[:6])


def resolve_uri(rel, models_dir):
    """Resolve the part of a `model://` URI after the scheme to a real path.

    gz does not treat a resource path entry as "a directory of models" -- it joins
    the ENTIRE remainder of the URI onto each entry in turn and takes the first hit.
    That is what lets `model://model_tiny/aws_robomaker_warehouse_ShelfE_01` name a
    model in a sibling tree, which is how small_warehouse_static_tinymap.world
    distinguishes the pristine small warehouse in model_tiny/ from the rescaled
    large one in models/.

    So models_dir may be a single directory (the old behaviour) or a sequence of
    them, searched in order. Falls back to the first root when nothing exists, so
    the caller still gets a path to name in its "missing mesh" warning rather than
    None.
    """
    roots = [models_dir] if isinstance(models_dir, str) else list(models_dir)
    for root in roots:
        path = os.path.join(root, rel)
        if os.path.exists(path):
            return path
    return os.path.join(roots[0], rel)


def load_collisions(models_dir, model):
    """Collision geometry of one model, as ('mesh', path, scale, offset_xyz)
    or ('sphere', radius, offset_xyz) or ('box', size_xyz, offset_xyz).

    models_dir is anything resolve_uri() accepts: one directory, or several to be
    searched in order the way gz searches IGN_GAZEBO_RESOURCE_PATH.

    Only the translation part of link/collision poses is honoured. Every model in
    this asset pack has identity poses there (checked), so a full SE(3) compose
    would be dead code -- but a non-zero translation is at least carried, and a
    non-zero rotation is reported rather than silently ignored.
    """
    sdf_path = resolve_uri(os.path.join(model, 'model.sdf'), models_dir)
    if not os.path.isfile(sdf_path):
        return []

    root = ET.parse(sdf_path).getroot()
    mdl = root.find('model')
    if mdl is None:
        return []

    out = []
    for link in mdl.iter('link'):
        lp = _pose_of(link)
        for col in link.findall('collision'):
            cp = _pose_of(col)
            if any(abs(a) > 1e-9 for a in lp[3:] + cp[3:]):
                print('  ! %s: non-zero link/collision ROTATION is ignored' % model,
                      file=sys.stderr)
            off = np.array([lp[0] + cp[0], lp[1] + cp[1], lp[2] + cp[2]])

            geom = col.find('geometry')
            if geom is None:
                continue

            mesh = geom.find('mesh')
            if mesh is not None:
                uri = (mesh.findtext('uri') or '').strip()
                if uri.startswith('model://'):
                    path = resolve_uri(uri[len('model://'):], models_dir)
                else:
                    path = resolve_uri(os.path.join(model, uri), models_dir)
                scale = np.array(_floats(mesh.findtext('scale') or '1 1 1'))
                out.append(('mesh', path, scale, off))
                continue

            sph = geom.find('sphere')
            if sph is not None:
                out.append(('sphere', float(sph.findtext('radius')), off))
                continue

            box = geom.find('box')
            if box is not None:
                out.append(('box', np.array(_floats(box.findtext('size'))), off))
    return out


# --- COLLADA -------------------------------------------------------------------

_DAE_CACHE = {}


def load_dae(path):
    """(vertices Nx3 in METRES, triangles Mx3 int) with the node matrix applied.

    Every collision DAE in this pack is a single <triangles> block with VERTEX at
    offset 0, so the general polylist/polygons cases are not implemented -- they
    raise rather than silently producing an empty mesh.
    """
    if path in _DAE_CACHE:
        return _DAE_CACHE[path]

    root = ET.parse(path).getroot()

    unit = 1.0
    asset = root.find(COLLADA_NS + 'asset')
    if asset is not None:
        u = asset.find(COLLADA_NS + 'unit')
        if u is not None and u.get('meter'):
            unit = float(u.get('meter'))

    geom = root.find('.//' + COLLADA_NS + 'geometry')
    if geom is None:
        raise ValueError('%s: no <geometry>' % path)
    mesh = geom.find(COLLADA_NS + 'mesh')

    # vertices: <triangles><input semantic="VERTEX" source="#...-VERTEX"> points at
    # a <vertices> element, which in turn points at the real POSITION float_array.
    src = {}
    for s in mesh.findall(COLLADA_NS + 'source'):
        arr = s.find(COLLADA_NS + 'float_array')
        if arr is not None and arr.text:
            src['#' + s.get('id')] = np.array(_floats(arr.text), dtype=np.float64)

    vert_map = {}
    for v in mesh.findall(COLLADA_NS + 'vertices'):
        inp = v.find(COLLADA_NS + 'input')
        vert_map['#' + v.get('id')] = inp.get('source')

    tri = mesh.find(COLLADA_NS + 'triangles')
    if tri is None:
        raise ValueError('%s: only <triangles> is supported, found %s'
                         % (path, [ET.QName(c).localname for c in mesh]))

    stride, vsrc, voff = 0, None, 0
    for inp in tri.findall(COLLADA_NS + 'input'):
        off = int(inp.get('offset', '0'))
        stride = max(stride, off + 1)
        if inp.get('semantic') == 'VERTEX':
            vsrc, voff = inp.get('source'), off
    if vsrc is None:
        raise ValueError('%s: <triangles> has no VERTEX input' % path)
    vsrc = vert_map.get(vsrc, vsrc)

    verts = src[vsrc].reshape(-1, 3)
    idx = np.array(tri.findtext(COLLADA_NS + 'p').split(), dtype=np.int64)
    tris = idx.reshape(-1, stride)[:, voff].reshape(-1, 3)

    # node matrix, then units. Doing it the other way scales the matrix's own
    # translation column a second time -- see the module docstring.
    m = root.find('.//' + COLLADA_NS + 'visual_scene//' + COLLADA_NS + 'matrix')
    if m is not None:
        M = np.array(_floats(m.text), dtype=np.float64).reshape(4, 4)
        verts = verts @ M[:3, :3].T + M[:3, 3]
    verts = verts * unit

    _DAE_CACHE[path] = (verts, tris)
    return verts, tris


# --- rasterization -------------------------------------------------------------

class Grid(object):
    def __init__(self, res, origin, size):
        self.res = res
        self.ox, self.oy = origin
        self.w, self.h = size
        self.xc = self.ox + (np.arange(self.w) + 0.5) * res
        self.yc = self.oy + (np.arange(self.h) + 0.5) * res

    def empty(self):
        return np.zeros((self.h, self.w), dtype=np.int32)


def raster_mesh(grid, verts, tris, z_lo, z_hi):
    """Occupancy of ONE watertight mesh instance over the band, by parity ray-cast."""
    par = grid.empty()
    nin = grid.empty()

    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    e1 = b - a
    e2 = c - a
    denom = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]

    res = grid.res
    for t in range(len(tris)):
        d = denom[t]
        if abs(d) < 1e-12:
            continue                      # vertical face: no crossing, correctly skipped

        ax, ay, az = a[t]
        xs = (ax, b[t][0], c[t][0])
        ys = (ay, b[t][1], c[t][1])
        c0 = int(np.floor((min(xs) - grid.ox) / res))
        c1 = int(np.ceil((max(xs) - grid.ox) / res))
        r0 = int(np.floor((min(ys) - grid.oy) / res))
        r1 = int(np.ceil((max(ys) - grid.oy) / res))
        c0, c1 = max(c0, 0), min(c1 + 1, grid.w)
        r0, r1 = max(r0, 0), min(r1 + 1, grid.h)
        if c0 >= c1 or r0 >= r1:
            continue

        px = grid.xc[c0:c1][None, :] - ax
        py = grid.yc[r0:r1][:, None] - ay

        w1 = (px * e2[t][1] - py * e2[t][0]) / d
        w2 = (e1[t][0] * py - e1[t][1] * px) / d
        w0 = 1.0 - w1 - w2
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue

        z = az + w1 * e1[t][2] + w2 * e2[t][2]
        above = inside & (z > z_lo)
        par[r0:r1, c0:c1] += above
        nin[r0:r1, c0:c1] += above & (z <= z_hi)

    return ((par % 2) == 1) | (nin > 0)


def raster_disc(grid, cx, cy, r):
    x = grid.xc[None, :] - cx
    y = grid.yc[:, None] - cy
    return (x * x + y * y) <= r * r


def transform(verts, offset, scale, pose):
    """Model-local metres -> world, via the include pose (yaw only; roll/pitch are
    always 0 in this world and a full rotation would only hide it if that changed)."""
    x, y, z, roll, pitch, yaw = pose
    if abs(roll) > 1e-9 or abs(pitch) > 1e-9:
        print('  ! include has non-zero roll/pitch, ignored', file=sys.stderr)
    p = verts * scale + offset
    ca, sa = np.cos(yaw), np.sin(yaw)
    out = np.empty_like(p)
    out[:, 0] = ca * p[:, 0] - sa * p[:, 1] + x
    out[:, 1] = sa * p[:, 0] + ca * p[:, 1] + y
    out[:, 2] = p[:, 2] + z
    return out


# --- output --------------------------------------------------------------------

def write_pgm(path, img):
    """Binary P5. Written by hand rather than via cv2.imwrite so the bake does not
    depend on OpenCV at all -- numpy and the stdlib are enough."""
    h, w = img.shape
    with open(path, 'wb') as f:
        f.write(b'P5\n')
        f.write(b'# CREATED BY bake_map.py from small_warehouse.world collision meshes\n')
        f.write(b'%d %d\n255\n' % (w, h))
        f.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())


def write_yaml(path, image_name, res, origin):
    with open(path, 'w') as f:
        f.write('image: %s\n' % image_name)
        f.write('mode: trinary\n')
        f.write('resolution: %f\n' % res)
        f.write('origin: [%f, %f, 0.000000]\n' % (origin[0], origin[1]))
        f.write('negate: 0\n')
        f.write('occupied_thresh: 0.65\n')
        f.write('free_thresh: 0.25\n')


def ascii_preview(occ, cols=78):
    """Coarse '#'/'.' dump, top row = +y, so it reads like the map does on screen."""
    h, w = occ.shape
    step = max(1, int(np.ceil(w / float(cols))))
    out = []
    for r in range(h - 1, -1, -step * 2):
        row = []
        for c in range(0, w, step):
            blk = occ[max(r - step * 2 + 1, 0):r + 1, c:c + step]
            row.append('#' if blk.any() else '.')
        out.append(''.join(row))
    return '\n'.join(out)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg = os.path.join(root, 'aws-robomaker-small-warehouse-world')

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--world', default=os.path.join(
        pkg, 'worlds', 'small_warehouse', 'small_warehouse.world'))
    # Two roots, searched in order, mirroring how gz searches its resource path:
    # the package share dir resolves the two-segment `model://model_tiny/<model>`
    # URIs the tinymap world uses, models/ resolves the plain one-segment ones.
    # Repeatable so an out-of-tree asset directory can be added.
    ap.add_argument('--models', action='append', default=None,
                    help='root to resolve model:// URIs against; repeatable, '
                         'searched in order (default: the package share dir, '
                         'then its models/ subdirectory)')
    ap.add_argument('--out', default=os.path.join(pkg, 'maps', 'baked'))
    ap.add_argument('--resolution', type=float, default=DEF_RES)
    ap.add_argument('--origin', type=float, nargs=2, default=list(DEF_ORIGIN),
                    metavar=('X', 'Y'))
    ap.add_argument('--size', type=int, nargs=2, default=list(DEF_SIZE),
                    metavar=('W', 'H'))
    ap.add_argument('--z-min', type=float, default=DEF_ZMIN)
    ap.add_argument('--z-max', type=float, default=DEF_ZMAX)
    ap.add_argument('--skip-model', action='append', default=None,
                    help='substring match; repeatable. default: %s' % ' '.join(DEF_SKIP))
    ap.add_argument('--drop-spheres', action='store_true',
                    help='omit the sphere markers (see the docstring first)')
    ap.add_argument('--allow-underpass', action='store_true',
                    help='shorthand for --z-max 0.50: let the planner go under shelves')
    ap.add_argument('--unknown-outside', action='store_true',
                    help='emit 205 (unknown) for unreachable cells instead of occupied')
    ap.add_argument('--seed-xy', type=float, nargs=2, default=list(DEF_SEED),
                    metavar=('X', 'Y'), help='reachability seed; the robot spawn')
    ap.add_argument('--preview', action='store_true',
                    help='print an ASCII dump and write nothing')
    args = ap.parse_args()
    # action='append' cannot carry a list default without appending to it.
    if args.models is None:
        args.models = [pkg, os.path.join(pkg, 'models')]

    skip = DEF_SKIP if args.skip_model is None else args.skip_model
    z_lo, z_hi = args.z_min, (0.50 if args.allow_underpass else args.z_max)

    grid = Grid(args.resolution, tuple(args.origin), tuple(args.size))
    print('grid %dx%d @ %.3f m, origin (%.3f, %.3f) -> x[%.2f, %.2f] y[%.2f, %.2f]'
          % (grid.w, grid.h, grid.res, grid.ox, grid.oy,
             grid.ox, grid.ox + grid.w * grid.res,
             grid.oy, grid.oy + grid.h * grid.res))
    print('z band [%.3f, %.3f]' % (z_lo, z_hi))

    occ = np.zeros((grid.h, grid.w), dtype=bool)
    includes = load_world(args.world)
    print('%d live includes in %s' % (len(includes), os.path.basename(args.world)))

    for model, name, pose in includes:
        if any(s in model or s in name for s in skip):
            print('  skip  %-34s (--skip-model)' % name)
            continue

        cols = load_collisions(args.models, model)
        if not cols:
            print('  skip  %-34s (no collision geometry)' % name)
            continue

        before = int(occ.sum())
        for col in cols:
            if col[0] == 'mesh':
                _, path, scale, off = col
                if not os.path.isfile(path):
                    print('  WARN  %-34s missing mesh %s' % (name, path), file=sys.stderr)
                    continue
                verts, tris = load_dae(path)
                wv = transform(verts, off, scale, pose)
                if wv[:, 2].max() < z_lo or wv[:, 2].min() > z_hi:
                    continue                      # wholly outside the band
                occ |= raster_mesh(grid, wv, tris, z_lo, z_hi)

            elif col[0] == 'sphere':
                if args.drop_spheres:
                    continue
                _, r, off = col
                cx, cy, cz = pose[0] + off[0], pose[1] + off[1], pose[2] + off[2]
                # widest cross-section anywhere in the band
                dz = abs(min(max(cz, z_lo), z_hi) - cz)
                if dz >= r:
                    continue
                occ |= raster_disc(grid, cx, cy, np.sqrt(r * r - dz * dz))

            elif col[0] == 'box':
                _, size, off = col
                hx, hy, hz = size / 2.0
                verts = np.array([[sx * hx, sy * hy, sz * hz]
                                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
                tris = np.array([
                    [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                    [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                    [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
                wv = transform(verts, off, np.ones(3), pose)
                if wv[:, 2].max() < z_lo or wv[:, 2].min() > z_hi:
                    continue
                occ |= raster_mesh(grid, wv, tris, z_lo, z_hi)

        print('  bake  %-34s +%6d cells' % (name, int(occ.sum()) - before))

    total = grid.w * grid.h
    print('occupied %d / %d (%.1f%%)' % (occ.sum(), total, 100.0 * occ.sum() / total))

    # --- reachability ----------------------------------------------------------
    sc = int((args.seed_xy[0] - grid.ox) / grid.res)
    sr = int((args.seed_xy[1] - grid.oy) / grid.res)
    unreachable = np.zeros_like(occ)
    if not (0 <= sc < grid.w and 0 <= sr < grid.h):
        print('WARN seed (%.2f, %.2f) is outside the grid; skipping reachability'
              % tuple(args.seed_xy), file=sys.stderr)
    elif occ[sr, sc]:
        print('WARN seed (%.2f, %.2f) is INSIDE an obstacle; skipping reachability'
              % tuple(args.seed_xy), file=sys.stderr)
    else:
        free = ~occ
        seen = np.zeros_like(free)
        stack = [(sr, sc)]
        seen[sr, sc] = True
        while stack:                       # explicit stack: recursion would blow up
            r, c = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.h and 0 <= nc < grid.w and free[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        unreachable = free & ~seen
        print('reachable free %d, unreachable free %d (%s)'
              % (int(seen.sum()), int(unreachable.sum()),
                 'unknown' if args.unknown_outside else 'painted occupied'))

    if args.preview:
        print()
        print(ascii_preview(occ | unreachable))
        print('\n--preview: nothing written.')
        return 0

    # --- write -----------------------------------------------------------------
    img = np.full((grid.h, grid.w), 254, dtype=np.uint8)
    if args.unknown_outside:
        img[unreachable] = 205
    else:
        img[unreachable] = 0
    img[occ] = 0
    img = np.flipud(img)                   # PGM row 0 is the TOP, i.e. max y

    os.makedirs(args.out, exist_ok=True)
    pgm = os.path.join(args.out, 'map.pgm')
    yml = os.path.join(args.out, 'map.yaml')
    write_pgm(pgm, img)
    write_yaml(yml, 'map.pgm', grid.res, (grid.ox, grid.oy))
    print('wrote %s' % pgm)
    print('wrote %s' % yml)
    print('\nNOTE this is a GENERATED file inside the source tree; re-run\n'
          '     colcon build --packages-select aws_robomaker_small_warehouse_world\n'
          '     for the installed copy to pick it up (or use --symlink-install once).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
