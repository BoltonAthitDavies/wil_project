#!/usr/bin/env python3
"""Generate the realistic-logistics dynamic warehouse world.

    python3 script/make_realistic_world.py            # write the world
    python3 script/make_realistic_world.py --plot     # also draw the plan PNG
    python3 script/make_realistic_world.py --check    # checks only, no write

Writes
    aws-robomaker-small-warehouse-world/worlds/small_warehouse_dynamic/
        small_warehouse_dynamic_realistic_nofloortexture.world

WHAT "REALISTIC" MEANS HERE
    The layout follows docs/references/warehouse/ (warehouseLayoutInfo.txt,
    CHAPTER-03-WAREHOUSE-LAYOUT-STORAGE-HANDLING-MHE.pdf and the Mecalux
    warehouse-layout manual, mecalux_warehouse_layout.txt): a warehouse is a
    chain of zones, receiving -> storage -> picking -> packing -> dispatch,
    plus service areas (management office between reception and dispatch,
    MHE charging isolated, general office).  The pattern is the chapter's
    U-shape flow: receiving and dispatch docks share the north wall, the
    fast-moving pick faces sit nearest the docks, bulk/slow stock sits at the
    back and down one side (block storage), and traffic runs in aisles.

    The shell is unchanged (41.28 m square, plain floor, same lights/plugins).
    Only the contents and the traffic are generated.

WHY THE AISLES ARE WHERE THEY ARE
    The four recorded dynamic datasets drove: south along x = 5.25 from the
    north wall, west along y ~ -15.4, north along x ~ -11.2, east along
    y ~ 5.8.  Those four lanes are kept as the main aisles (3 m), so the
    existing teleop route remains drivable in the new layout; --check reports
    the clearance of that route from every static object.

TRAFFIC: JOBS, NOT SHUTTLES
    Four pallet jacks, each on a logistic job driven by the LogisticsScheduler
    system (gz_kinematic_trajectory/LogisticsScheduler.cc): drive to a load,
    take it on the forks, carry it through the aisles, set it down in the zone
    it belongs in, go for the next.  Put-away (receiving -> block storage),
    retrieval (block storage -> dispatch), and two pickers carrying totes from
    the pick faces to the packing stations.  Each job is a 6-move slot
    rotation over two loads and three slots that returns to its starting
    state, so it loops forever without two loads ever claiming one slot; the
    generator simulates every job to prove that before writing.  Speeds are
    1.2-1.3 m/s, walking pace with a loaded jack.

LIMITS THAT DO NOT CHANGE
    Everything is kinematic: contact with the robot resolves as overlap, there
    is no collision response, and two carriers crossing one intersection at
    the same moment pass through each other.  There is no human model in the
    asset set.  The floor is plain (no aisle paint), as the file name says.
"""

import argparse
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_dock_models import DOORS_BY_LAYOUT, DOOR_W, DOOR_H, YI1, YO1   # noqa: E402  the dock doors
PKG = os.path.join(ROOT, 'aws-robomaker-small-warehouse-world')
WORLD_U = os.path.join(PKG, 'worlds', 'small_warehouse_dynamic',
                       'small_warehouse_dynamic_realistic_nofloortexture.world')
WORLD_I = WORLD_U[:-6] + '_01.world'      # the I-shape (through-flow) variant
WORLD = WORLD_U                           # set by select_layout()
GT_DIR = os.path.join(ROOT, 'output', 'compare', 'relogged_20260916', 'ground_truth')

HALF = 20.55          # usable interior half-width; the wall faces are at +-20.64
MARGIN = 0.25         # clearance required between any two static footprints
ROBOT_HALF = 0.42     # robot half-width, for the route-clearance report
CARRY = 1.3           # load centre ahead of the carrier while carried, m (fork length)

# z offsets the existing worlds use per model (the meshes do not sit at z = 0)
Z = {'Bucket_01': 0.029223, 'ClutteringA_01': 0.004223, 'ClutteringC_01': 0.004223,
     'ClutteringD_01': -0.298777, 'ShelfD_01': 0.004923, 'ShelfE_01': 0.004923,
     'DeskC_01': 0.034223, 'PalletJackB_01': 0.022323, 'TrashCanC_01': 0.001223,
     'DockDoor_01': 0.0, 'DockDoorOpen_01': 0.0, 'Trailer_01': 0.0, 'DockYard_01': 0.0}
# zones whose objects sit in or beyond the wall line, outside the "inside the
# walls" rule: the door panels (in the wall thickness) and the yard
OUTSIDE = {'dock doors', 'dock yard'}

YAW90 = math.pi / 2
PI = math.pi


# ----------------------------------------------------------------- footprints
def footprint(model):
    """Model-frame XY bounding box (xmin, ymin, xmax, ymax) of the collision
    mesh, in metres.  The COLLADA files store vertices in centimetres under a
    per-node <matrix> (the FBX exporter rotates the shelf meshes by 90 deg), so
    the raw arrays are transformed by their node before the box is taken --
    reading the arrays alone gives the shelves sideways."""
    d = os.path.join(PKG, 'models', 'aws_robomaker_warehouse_' + model, 'meshes')
    dae = [f for f in os.listdir(d) if f.lower().endswith('.dae') and 'collision' in f.lower()]
    if not dae:
        dae = [f for f in os.listdir(d) if f.lower().endswith('.dae')]
    root = ET.parse(os.path.join(d, dae[0])).getroot()
    ns = root.tag.split('}')[0].strip('{')
    q = lambda t: '{%s}%s' % (ns, t)
    unit = root.find(q('asset') + '/' + q('unit'))
    scale = float(unit.get('meter')) if unit is not None else 1.0
    geoms = {}
    for g in root.iter(q('geometry')):
        for fa in g.iter(q('float_array')):
            if 'position' in (fa.get('id') or '').lower():
                v = [float(t) for t in fa.text.split()]
                geoms[g.get('id')] = list(zip(v[0::3], v[1::3], v[2::3]))
    pts = []

    def walk(node, parent):
        m = node.find(q('matrix'))
        M = parent
        if m is not None:
            a = [float(t) for t in m.text.split()]
            M = [[sum(parent[r][k] * a[k * 4 + c] for k in range(4)) for c in range(4)]
                 for r in range(4)]
        for ig in node.findall(q('instance_geometry')):
            for (x, y, z) in geoms.get(ig.get('url').lstrip('#'), []):
                pts.append((M[0][0] * x + M[0][1] * y + M[0][2] * z + M[0][3],
                            M[1][0] * x + M[1][1] * y + M[1][2] * z + M[1][3]))
        for child in node.findall(q('node')):
            walk(child, M)

    I = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    for vs in root.iter(q('visual_scene')):
        for node in vs.findall(q('node')):
            walk(node, I)
    if not pts:
        pts = [(x, y) for g in geoms.values() for (x, y, z) in g]
    xs = [p[0] * scale for p in pts]; ys = [p[1] * scale for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


FOOT = {m: footprint(m) for m in Z}


def aabb(model, x, y, yaw, pad=0.0):
    x0, y0, x1, y1 = FOOT[model]
    c, s = math.cos(yaw), math.sin(yaw)
    cs = [(x + c * px - s * py, y + s * px + c * py)
          for (px, py) in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
    return (min(p[0] for p in cs) - pad, min(p[1] for p in cs) - pad,
            max(p[0] for p in cs) + pad, max(p[1] for p in cs) + pad)


def extents(model, yaw):
    b = aabb(model, 0.0, 0.0, yaw)
    return b[2] - b[0], b[3] - b[1]


def half_reach(model):
    """Largest half-extent of a model in any heading: the radius its footprint
    can sweep while being carried through a turn."""
    x0, y0, x1, y1 = FOOT[model]
    return max(math.hypot(px, py) for px in (x0, x1) for py in (y0, y1))


def overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


# ---------------------------------------------------------------- the layout
# Zone boundaries (y, north positive):
#   20.55 .. 18.5   dock apron, clear
#   18.5  .. 12.1   staging: receiving (west) / dispatch (east), two rows, the
#                   management office between them, a 2.4 m lane at y 15.4
#   12.1  ..  9.3   fast-moving pick faces: E-W shelves with totes in front
#    9.3  ..  5.08  cross-aisle C (the recorded route's y ~ 5.8 lane)
#    5.08 .. -14.5  storage: rack rows N-S (centre, medium rotation; east,
#                   picking); block storage down the west side
#  -14.5  .. -17.0  cross-aisle E (the recorded route's y ~ -15.4 lane)
#  -17.0  .. -20.55 service band: block storage SW, MHE charging, empty
#                   pallets, general office
# N-S main aisles: A at x -12.7..-9.7 (route x ~ -11.2), B at x 3.75..6.75
# (route x 5.25, also the dock lane from the spawn point).

STATIC = []    # dicts: model x y yaw zone group name
NAMES = {}     # key -> model name, for the loads and carriers the jobs refer to


def at(x, y, slot):
    return abs(x - SLOTS[slot][0]) < 1e-6 and abs(y - SLOTS[slot][1]) < 1e-6


def add(model, x, y, yaw, zone, group=None, key=None):
    """group: objects in one run (a rack row, a pick-face line) touch by design
    and are exempt from the spacing margin against each other.  key: a handle
    for a load or carrier a job refers to."""
    name = 'aws_robomaker_warehouse_%s_%03d' % (model, 100 + len(STATIC))
    STATIC.append(dict(model=model, x=x, y=y, yaw=yaw, zone=zone, group=group, name=name))
    if key:
        NAMES[key] = name
    return name


# slots the jobs move loads between: the load's pose when parked (x, y, yaw)
SLOTS_U = {
    'R_a': (-17.6, 13.2, 0.0),     # receiving staging, row 2, west end: open floor south
    'S1': (-16.3, 2.7, 0.0),       # west block storage, east column, empty
    'S2': (-16.3, -0.5, 0.0),      # west block storage, east column
    'D_a': (14.5, 13.2, 0.0),      # dispatch staging, row 2, above the pick-face gap
    'S3': (-12.5, -18.9, 0.0),     # south-west block, empty
    'S4': (-15.5, -18.9, 0.0),     # south-west block
    'F1': (-3.3, 9.9, 0.0),        # pick face, centre
    'P1': (11.0, -13.3, 0.0),      # packing station, empty
    'P2': (8.5, -13.3, 0.0),       # packing station
    'F3': (17.3, 9.9, 0.0),        # pick face, east
    'P3': (14.0, -13.3, 0.0),      # packing station, empty
    'P4': (16.6, -13.3, 0.0),      # packing station
}
EMPTY_U = {'S1', 'S3', 'P1', 'P3'}
# the carrier's heading when it stands at a slot (it arrives head-on and backs out)
APPROACH_U = {'R_a': YAW90, 'D_a': YAW90, 'S1': PI, 'S2': PI, 'S3': -YAW90, 'S4': -YAW90,
            'F1': YAW90, 'F3': YAW90, 'P1': -YAW90, 'P2': -YAW90, 'P3': -YAW90, 'P4': -YAW90}


def build_static_u():
    # -- staging rows --------------------------------------------------------
    rows = (17.6, 13.2)
    recv_x = [-17.6, -13.7, -9.8, -5.9, -2.0]
    for j, y in enumerate(rows):
        for i, x in enumerate(recv_x):
            m = 'ClutteringA_01' if (i + j) % 2 == 0 else 'ClutteringC_01'
            add(m, x, y, 0.0, 'receiving staging',
                key='L1' if at(x, y, 'R_a') else None)
    disp_x = [10.5, 14.5, 18.5]
    for j, y in enumerate(rows):
        for i, x in enumerate(disp_x):
            m = 'ClutteringD_01' if (i + j) % 2 == 0 else 'ClutteringA_01'
            add(m, x, y, 0.0, 'dispatch staging',
                key='L4' if at(x, y, 'D_a') else None)
    add('PalletJackB_01', -19.6, 15.4, 0.0, 'receiving staging')
    add('PalletJackB_01', 19.75, 15.4, 0.0, 'dispatch staging')
    # management office between reception and dispatch (Mecalux: "ideally ...
    # between these two areas"), west of the dock lane the robot uses
    for y in (17.3, 14.6):
        add('DeskC_01', 1.9, y, YAW90, 'management office')
    add('TrashCanC_01', 1.9, 12.6, 0.0, 'management office')

    # -- fast-moving pick faces, nearest the docks --------------------------
    # the east run has a gap at x 11.3..15.9: the approach to the dispatch slot
    for x in (-7.5, -3.4, 0.7, 9.3, 17.9):
        add('ShelfD_01' if x < 5 else 'ShelfE_01', x, 11.4, 0.0, 'pick faces',
            'pickface-%s' % ('w' if x < 5 else 'e'))
    # no tote at x 10.3: it would sit over the exit of the picking aisle at 10.78
    for x in (-8.5, -5.9, -3.3, -0.7, 1.9, 8.3, 17.3, 19.3):
        key = 'T1' if x == SLOTS['F1'][0] else 'T3' if x == SLOTS['F3'][0] else None
        add('Bucket_01', x, 9.9, 0.0, 'pick faces', key=key)

    # -- storage: centre rack rows, N-S, double (back to back) ----------------
    unit, depth = extents('ShelfD_01', 0.0)   # 3.918 m long, 0.880 m deep
    ys = [5.08 - unit / 2 - k * unit for k in range(5)]
    for cx in (-8.82, -4.26, 0.30):
        for y in ys:
            add('ShelfD_01', cx - depth / 2, y, YAW90, 'storage (medium)', 'rack%.2f' % cx)
            add('ShelfE_01', cx + depth / 2, y, -YAW90, 'storage (medium)', 'rack%.2f' % cx)
    # -- picking racks, east, three units ------------------------------------
    for cx in (8.5, 13.06, 17.62):
        for y in ys[:3]:
            add('ShelfE_01', cx - depth / 2, y, YAW90, 'picking', 'rack%.2f' % cx)
            add('ShelfD_01', cx + depth / 2, y, -YAW90, 'picking', 'rack%.2f' % cx)
    # -- packing area, east south: stations with totes ----------------------
    # desks sit clear of the two picking-aisle exits (x 10.78 and 15.34), and
    # high enough that a picker can turn at the packing lane (y -12.0) with a
    # tote on the forks
    for x in (8.3, 12.6, 17.2, 19.8):
        add('DeskC_01', x, -9.6, 0.0, 'packing')
    for x, key in ((8.5, 'T2'), (16.6, 'T4'), (19.0, None)):
        add('Bucket_01', x, -13.3, 0.0, 'packing', key=key)
    add('TrashCanC_01', 19.6, -7.6, 0.0, 'packing')

    # -- block storage down the west side (slow / bulk) -----------------------
    for i, y in enumerate([-13.3 + 3.2 * k for k in range(6)]):   # 3.2 m pitch: room to swing a pallet in
        for j, x in enumerate((-18.9, -16.3)):
            if at(x, y, 'S1'):
                continue                                   # the empty slot
            m = 'ClutteringA_01' if (i + j) % 2 else 'ClutteringC_01'
            add(m, x, y, 0.0, 'block storage', key='L2' if at(x, y, 'S2') else None)
    # -- service band, south ---------------------------------------------------
    for i, x in enumerate((-18.5, -15.5)):                  # -12.5 is the empty slot; further
        # east would put the exchange cell under the rack ends, too close to turn in
        m = 'ClutteringC_01' if i % 2 else 'ClutteringA_01'
        add(m, x, -18.9, 0.0, 'block storage', key='L3' if x == SLOTS['S4'][0] else None)
    for x in (-6.0, -4.3, -2.6, -0.9):
        add('PalletJackB_01', x, -19.3, YAW90, 'MHE charging')
    add('TrashCanC_01', -7.6, -19.6, 0.0, 'MHE charging')
    for x in (2.0, 3.6):
        add('ClutteringD_01', x, -19.3, 0.0, 'empty pallets')
    for x in (12.0, 15.0, 18.0):
        add('DeskC_01', x, -19.3, 0.0, 'general office')
    add('TrashCanC_01', 8.8, -19.6, 0.0, 'general office')

    add_docks()

    # -- carriers, parked where their job starts ------------------------------
    add('PalletJackB_01', -17.6, 7.8, YAW90, 'traffic', key='C1')
    add('PalletJackB_01', -15.5, -15.9, -YAW90, 'traffic', key='C2')
    add('PalletJackB_01', -3.3, 7.8, YAW90, 'traffic', key='C3')
    add('PalletJackB_01', 17.3, 7.8, YAW90, 'traffic', key='C4')


# ------------------------------------------------------------------- the jobs
# A move is (load, from-slot, to-slot).  Six moves rotate loads a (at s_a) and
# b (at s_b) through the free slot and back to the start:
#   a->free, b->s_a, a->s_b, b->free, a->s_a, b->s_b
# so the job loops forever and no drop ever lands on an occupied slot.
def rot6(a, b, s_a, s_b, s_free):
    return [(a, s_a, s_free), (b, s_b, s_a), (a, s_free, s_b),
            (b, s_a, s_free), (a, s_b, s_a), (b, s_free, s_b)]


def carrier_pose_at(slot):
    """Where the carrier stands while picking or dropping at a slot.  Headings
    are multiples of 90 deg, so the trig is snapped: a cos(-pi/2) of 6e-17
    once produced a 4e-7 m "leg" that the checker read as a westward move."""
    x, y, _ = SLOTS[slot]; yaw = APPROACH[slot]
    c, sn = round(math.cos(yaw), 9), round(math.sin(yaw), 9)
    return x - CARRY * c, y - CARRY * sn


# Paths between slots, as the gotos the carrier drives between its stand-point
# at the source slot and its stand-point at the destination slot.
def path_u_C1(src, dst):
    """Receiving slot R_a (approached from the south across the open floor west
    of the pick faces) and the west block cells (approached from the east off
    the strip beside aisle A), joined through cross-aisle C."""
    C = 7.8; strip = -13.3; rx = SLOTS['R_a'][0]
    if dst == 'R_a':
        return [(strip, SLOTS[src][1]), (strip, C), (rx, C), (rx, SLOTS['R_a'][1] - CARRY)]
    if src == 'R_a':
        return [(rx, C), (strip, C), (strip, SLOTS[dst][1]), (SLOTS[dst][0] + CARRY, SLOTS[dst][1])]
    return [(strip, SLOTS[src][1]), (strip, SLOTS[dst][1]), (SLOTS[dst][0] + CARRY, SLOTS[dst][1])]


def path_u_C2(src, dst):
    """South-west block cells (approached from the north off cross-aisle E) and
    the dispatch slot D_a (approached from the south through the gap in the
    east pick faces), joined by aisle B and cross-aisle C."""
    C = 7.8; B = 5.25; E = -15.9; dx_ = SLOTS['D_a'][0]
    if dst == 'D_a':
        return [(SLOTS[src][0], E), (B, E), (B, C), (dx_, C), (dx_, SLOTS['D_a'][1] - CARRY)]
    if src == 'D_a':
        return [(dx_, C), (B, C), (B, E), (SLOTS[dst][0], E), (SLOTS[dst][0], SLOTS[dst][1] + CARRY)]
    return [(SLOTS[src][0], E), (SLOTS[dst][0], E), (SLOTS[dst][0], SLOTS[dst][1] + CARRY)]


def path_u_C3(src, dst):
    """Centre pick face (from cross-aisle C) and the packing stations (from
    the packing lane), joined by the east picking aisle at x 10.78."""
    C = 7.8; aisle = 10.78; pack = -12.0; fx = SLOTS['F1'][0]
    if dst == 'F1':
        return [(SLOTS[src][0], pack), (aisle, pack), (aisle, C), (fx, C),
                (fx, SLOTS['F1'][1] - CARRY)]
    if src == 'F1':
        return [(fx, C), (aisle, C), (aisle, pack), (SLOTS[dst][0], pack)]
    return [(SLOTS[src][0], pack), (SLOTS[dst][0], pack)]


def path_u_C4(src, dst):
    """East pick face and the packing stations, joined by the picking aisle at
    x 15.34, which runs straight down to the packing lane."""
    C = 7.8; aisle = 15.34; pack = -12.0; fx = SLOTS['F3'][0]
    if dst == 'F3':
        return [(SLOTS[src][0], pack), (aisle, pack), (aisle, C), (fx, C), (fx, SLOTS['F3'][1] - CARRY)]
    if src == 'F3':
        return [(fx, C), (aisle, C), (aisle, pack), (SLOTS[dst][0], pack)]
    return [(SLOTS[src][0], pack), (SLOTS[dst][0], pack)]


# key, speed m/s, handling s, start s, description, moves, path function
CARRIERS_U = [
    ('C1', 1.3, 5.0, 2.0, 'put-away: receiving <-> west block storage',
     rot6('L1', 'L2', 'R_a', 'S2', 'S1'), path_u_C1),
    ('C2', 1.3, 5.0, 9.0, 'retrieval: south-west block <-> dispatch staging',
     rot6('L3', 'L4', 'S4', 'D_a', 'S3'), path_u_C2),
    ('C3', 1.2, 4.0, 5.0, 'picker: centre pick face <-> packing station',
     rot6('T1', 'T2', 'F1', 'P2', 'P1'), path_u_C3),
    ('C4', 1.2, 4.0, 13.0, 'picker: east pick face <-> packing station',
     rot6('T3', 'T4', 'F3', 'P4', 'P3'), path_u_C4),
]

# =================================================================== I-shape
# Through flow (the chapter's "I-shape"): receiving docks on the north wall,
# dispatch docks on the south wall, storage in the middle, the fast-moving
# pick faces between storage and dispatch, packing beside dispatch.  Goods go
# straight through.  Zone bands (y):
#   18.6 .. 12.2   receiving staging, both sides of the robot's dock lane
#   12.2 ..  8.2   cross-aisle C (carrier lane y 9.3)
#    8.2 .. -7.5   storage: 3 double rack rows x 4 units; picking 3 rows x 3
#                  units east; block storage west (4 rows)
#   -7.5 ..-11.2   cross-aisle E' (pallet lane y -9.4 west, picker lane y -9.75)
#  -11.2 ..-13.8   pick faces: totes north of E-W shelves, facing storage
#  -13.8 ..-19.2   dispatch staging (west, centre) and packing (east)
#  -19.2 ..-20.6   dock apron
# The Mecalux rule "management office between reception and dispatch" cannot
# hold with docks on opposite walls; both offices sit on the east wall.
SLOTS_I = {
    'R_a': (-17.6, 13.2, 0.0),     # receiving staging, row 2, west end
    'S1': (-16.3, 4.7, 0.0),       # west block, east column, empty
    'S2': (-16.3, 1.5, 0.0),
    'S3': (-16.3, -4.9, 0.0),      # west block, lower rows (C2), empty
    'S4': (-16.3, -1.7, 0.0),
    'D_a': (-17.6, -15.2, 0.0),    # dispatch staging, row 1, west end
    'F1': (-3.3, -11.85, 0.0),     # pick face totes, facing storage
    'F3': (1.9, -11.85, 0.0),
    'P1': (11.0, -16.3, 0.0),      # packing, empty
    'P2': (8.5, -16.3, 0.0),
    'P3': (14.0, -16.3, 0.0),      # packing, empty
    'P4': (16.6, -16.3, 0.0),
}
EMPTY_I = {'S1', 'S3', 'P1', 'P3'}
APPROACH_I = {'R_a': YAW90, 'D_a': -YAW90, 'S1': PI, 'S2': PI, 'S3': PI, 'S4': PI,
              'F1': -YAW90, 'F3': -YAW90, 'P1': -YAW90, 'P2': -YAW90, 'P3': -YAW90, 'P4': -YAW90}


def build_static_i():
    # -- receiving staging, north, either side of the robot's dock lane ------
    rows = (17.6, 13.2)
    for j, y in enumerate(rows):
        for i, x in enumerate([-17.6, -13.7, -9.8, -5.9, -2.0]):
            add('ClutteringA_01' if (i + j) % 2 == 0 else 'ClutteringC_01', x, y, 0.0,
                'receiving staging', key='L1' if at(x, y, 'R_a') else None)
        for i, x in enumerate([9.0, 12.9, 16.8]):
            add('ClutteringC_01' if (i + j) % 2 == 0 else 'ClutteringA_01', x, y, 0.0, 'receiving staging')
    add('PalletJackB_01', -19.6, 15.4, 0.0, 'receiving staging')
    # -- offices on the east wall: general office north, management mid-way --
    for y in (13.5, 10.9):
        add('DeskC_01', 19.6, y, YAW90, 'general office')
    add('TrashCanC_01', 19.6, 15.9, 0.0, 'general office')
    for y in (1.3, -1.3):
        add('DeskC_01', 19.6, y, YAW90, 'management office')
    add('TrashCanC_01', 19.6, -3.6, 0.0, 'management office')
    # -- storage: centre rack rows, 4 units ------------------------------------
    unit, depth = extents('ShelfD_01', 0.0)
    ys = [7.3 - unit / 2 - k * unit for k in range(4)]   # top 7.3 clears the recorded y~8 leg
    for cx in (-8.82, -4.26, 0.30):
        for y in ys:
            add('ShelfD_01', cx - depth / 2, y, YAW90, 'storage (medium)', 'rack%.2f' % cx)
            add('ShelfE_01', cx + depth / 2, y, -YAW90, 'storage (medium)', 'rack%.2f' % cx)
    # -- picking racks east, 3 units -------------------------------------------
    for cx in (8.5, 13.06, 17.62):
        for y in ys[:3]:
            add('ShelfE_01', cx - depth / 2, y, YAW90, 'picking', 'rack%.2f' % cx)
            add('ShelfD_01', cx + depth / 2, y, -YAW90, 'picking', 'rack%.2f' % cx)
    # -- pick faces between storage and dispatch, totes facing storage --------
    for x in (-7.5, -3.4, 0.7):
        add('ShelfD_01', x, -13.35, 0.0, 'pick faces', 'pickface')
    for x in (-8.5, -5.9, -3.3, -0.7, 1.9):
        key = 'T1' if x == SLOTS['F1'][0] else 'T3' if x == SLOTS['F3'][0] else None
        add('Bucket_01', x, -11.85, 0.0, 'pick faces', key=key)
    # -- dispatch staging, south, west and centre ------------------------------
    for j, y in enumerate((-15.2, -18.2)):
        for i, x in enumerate([-17.6, -13.7, -9.8, -5.9, -2.0]):
            add('ClutteringD_01' if (i + j) % 2 == 0 else 'ClutteringA_01', x, y, 0.0,
                'dispatch staging', key='L4' if at(x, y, 'D_a') else None)
    add('PalletJackB_01', -19.6, -16.7, 0.0, 'dispatch staging')
    # -- packing, south-east: desks, totes, then charging and empty pallets ----
    for x in (8.3, 12.6, 17.2, 19.8):
        add('DeskC_01', x, -12.6, 0.0, 'packing')
    for x, key in ((8.5, 'T2'), (16.6, 'T4'), (19.0, None)):
        add('Bucket_01', x, -16.3, 0.0, 'packing', key=key)
    add('TrashCanC_01', 19.6, -10.6, 0.0, 'packing')
    for x in (8.3, 9.9):
        add('ClutteringD_01', x, -19.3, 0.0, 'empty pallets')
    for x in (14.8, 16.5, 18.2, 19.9):
        add('PalletJackB_01', x, -19.3, YAW90, 'MHE charging')
    # -- block storage west, four rows ---------------------------------------
    for i, y in enumerate((-4.9, -1.7, 1.5, 4.7)):
        for j, x in enumerate((-18.9, -16.3)):
            if at(x, y, 'S1') or at(x, y, 'S3'):
                continue
            key = 'L2' if at(x, y, 'S2') else 'L3' if at(x, y, 'S4') else None
            add('ClutteringA_01' if (i + j) % 2 else 'ClutteringC_01', x, y, 0.0, 'block storage', key=key)
    add_docks()
    # -- carriers, parked where their job starts ------------------------------
    add('PalletJackB_01', -17.6, 9.3, YAW90, 'traffic', key='C1')
    add('PalletJackB_01', -17.6, -9.4, -YAW90, 'traffic', key='C2')
    add('PalletJackB_01', -3.3, -9.75, -YAW90, 'traffic', key='C3')
    add('PalletJackB_01', 1.9, -9.75, -YAW90, 'traffic', key='C4')


def path_i_C1(src, dst):
    """Receiving R_a (from the south, across the open floor west of the racks)
    and the upper west-block cells (from the east, off the strip)."""
    C = 9.3; strip = -13.3; rx = SLOTS['R_a'][0]
    if dst == 'R_a':
        return [(strip, SLOTS[src][1]), (strip, C), (rx, C), (rx, SLOTS['R_a'][1] - CARRY)]
    if src == 'R_a':
        return [(rx, C), (strip, C), (strip, SLOTS[dst][1]), (SLOTS[dst][0] + CARRY, SLOTS[dst][1])]
    return [(strip, SLOTS[src][1]), (strip, SLOTS[dst][1]), (SLOTS[dst][0] + CARRY, SLOTS[dst][1])]


def path_i_C2(src, dst):
    """Lower west-block cells and the dispatch slot D_a (from the north, down
    the open floor west of the pick faces), joined by cross-aisle E'."""
    E = -9.4; strip = -13.3; dx_ = SLOTS['D_a'][0]
    if dst == 'D_a':
        return [(strip, SLOTS[src][1]), (strip, E), (dx_, E), (dx_, SLOTS['D_a'][1] + CARRY)]
    if src == 'D_a':
        return [(dx_, E), (strip, E), (strip, SLOTS[dst][1]), (SLOTS[dst][0] + CARRY, SLOTS[dst][1])]
    return [(strip, SLOTS[src][1]), (strip, SLOTS[dst][1]), (SLOTS[dst][0] + CARRY, SLOTS[dst][1])]


def _path_i_picker(face, aisle):
    def path(src, dst):
        E = -9.75; pack = -15.0; fx = SLOTS[face][0]
        if dst == face:
            return [(SLOTS[src][0], pack), (aisle, pack), (aisle, E), (fx, E), (fx, SLOTS[face][1] + CARRY)]
        if src == face:
            return [(fx, E), (aisle, E), (aisle, pack), (SLOTS[dst][0], pack)]
        return [(SLOTS[src][0], pack), (SLOTS[dst][0], pack)]
    return path


CARRIERS_I = [
    ('C1', 1.3, 5.0, 2.0, 'put-away: receiving <-> west block storage',
     rot6('L1', 'L2', 'R_a', 'S2', 'S1'), path_i_C1),
    ('C2', 1.3, 5.0, 9.0, 'retrieval: west block storage <-> dispatch staging',
     rot6('L3', 'L4', 'S4', 'D_a', 'S3'), path_i_C2),
    ('C3', 1.2, 4.0, 5.0, 'picker: pick face <-> packing station (aisle x 10.78)',
     rot6('T1', 'T2', 'F1', 'P2', 'P1'), _path_i_picker('F1', 10.78)),
    ('C4', 1.2, 4.0, 13.0, 'picker: pick face <-> packing station (aisle x 15.34)',
     rot6('T3', 'T4', 'F3', 'P4', 'P3'), _path_i_picker('F3', 15.34)),
]

LAYOUTS = {
    'u': dict(world=WORLD_U, slots=SLOTS_U, empty=EMPTY_U, approach=APPROACH_U,
              build=build_static_u, carriers=CARRIERS_U,
              title='U-shape flow, carrier jobs', bake='baked_dynamic_realistic',
              labels={'RECEIVING DOCK': (-10, 19.4), 'DISPATCH DOCK': (14, 19.4),
                      'pick faces': (-3.4, 12.6), 'cross-aisle C': (-14, 7.2),
                      'storage (medium rotation)': (-4.3, -1), 'picking': (13, 0),
                      'packing': (13.5, -8.5), 'block storage': (-17.6, -7),
                      'cross-aisle E': (-5, -15.9), 'MHE charging': (-3.5, -17.6),
                      'general office': (15, -17.6), 'mgmt office': (1.9, 19.4),
                      'aisle A': (-11.2, 1), 'aisle B': (5.25, 1)}),
    'i': dict(world=WORLD_I, slots=SLOTS_I, empty=EMPTY_I, approach=APPROACH_I,
              build=build_static_i, carriers=CARRIERS_I,
              title='I-shape (through) flow, carrier jobs', bake='baked_dynamic_realistic_01',
              labels={'RECEIVING DOCK': (-10, 19.4), 'DISPATCH DOCK': (-10, -19.9),
                      'pick faces': (-3.4, -14.6), 'cross-aisle C': (-7, 10.4),
                      'storage (medium rotation)': (-4.3, 0.5), 'picking': (13, 2.5),
                      'packing': (13.5, -14.3), 'block storage': (-17.6, -7.2),
                      "cross-aisle E'": (-7, -8.6), 'MHE charging': (17.3, -18.0),
                      'empty pallets': (9.1, -18.0), 'general office': (18.6, 8.8),
                      'mgmt office': (18.6, 4.0), 'aisle A': (-11.2, 1), 'aisle B': (5.25, 1)}),
}


def add_docks():
    """Dock doors in the wall, trailers at the open ones, the yard outside.
    make_dock_models.py owns the positions and builds the shell with the
    matching openings."""
    for side, sign in (('north', 1.0), ('south', -1.0)):
        for xc, state in DOORS_BY_LAYOUT[LAYOUT][side]:
            add('DockDoor_01' if state == 'closed' else 'DockDoorOpen_01', xc, sign * (YI1 + YO1) / 2,
                0.0 if sign > 0 else PI, 'dock doors', 'docks')
            if state == 'open':
                add('Trailer_01', xc, sign * (YO1 + 0.05), 0.0 if sign > 0 else PI, 'dock yard', 'docks')
        if DOORS_BY_LAYOUT[LAYOUT][side]:
            add('DockYard_01', 0.0, 0.0, 0.0 if sign > 0 else PI, 'dock yard', 'docks')


LAYOUT = 'u'


def select_layout(key):
    global LAYOUT, WORLD, SLOTS, EMPTY, APPROACH, CARRIERS, build_static
    lay = LAYOUTS[key]
    LAYOUT = key; WORLD = lay['world']; SLOTS = lay['slots']; EMPTY = lay['empty']
    APPROACH = lay['approach']; CARRIERS = lay['carriers']; build_static = lay['build']


select_layout('u')


def initial_slots(carrier):
    where = {}
    for (load, src, dst) in carrier[5]:
        where.setdefault(load, src)
    return where


def expand(carrier):
    """The plugin steps for a carrier: ('pick', load, yaw) / ('goto', x, y) /
    ('drop', slot).  Simulates the slot bookkeeping and refuses a job whose
    picks and drops do not line up or that does not return to its start."""
    key, speed, handling, start, desc, moves, path = carrier
    where = initial_slots(carrier)
    occupied = {slot: load for load, slot in where.items()}
    state0 = dict(where)
    steps = []

    def legs(frm, to):
        # the first leg out of a slot is driven backwards when it points
        # straight back along the approach -- a jack backs out, it does not
        # spin round with a pallet on the forks
        here = carrier_pose_at(frm); heading = APPROACH[frm]
        for n, (x, y) in enumerate(path(frm, to)):
            back = n == 0 and math.hypot(x - here[0], y - here[1]) > 1e-6 and \
                math.cos(math.atan2(y - here[1], x - here[0]) - heading) < -0.99
            steps.append(('goto', x, y, back))

    cur = None                                  # slot the carrier stands at
    for (load, src, dst) in moves:
        if where[load] != src:
            raise SystemExit('job %s: %s expected at %s but is at %s' % (key, load, src, where[load]))
        if dst in occupied:
            raise SystemExit('job %s: slot %s occupied by %s when %s arrives'
                             % (key, dst, occupied[dst], load))
        if cur is not None and cur != src:
            legs(cur, src)                      # empty travel to the next pick
        steps.append(('pick', load, APPROACH[src]))
        legs(src, dst)                          # loaded travel
        steps.append(('drop', dst))
        cur = dst
        del occupied[src]; occupied[dst] = load; where[load] = dst
    # the loop closes with the carrier at the last drop; the plugin then drives
    # straight to the first pick, so that leg is emitted too
    first = moves[0][1]
    if cur != first:
        legs(cur, first)
    if where != state0:
        raise SystemExit('job %s does not return to its start: %s -> %s' % (key, state0, where))
    return steps


def replay(carrier):
    """Carrier positions in order: (point, carried load or None, reverse flag,
    heading the carrier had when it picked that load) -- the load's heading
    relative to the carrier is fixed at pick-up, so its box in the carrier
    frame depends on that heading, which differs between slots."""
    where = initial_slots(carrier)
    pts = []; carried = None; pyaw = 0.0
    for s in expand(carrier):
        if s[0] == 'pick':
            slot = where[s[1]]
            pts.append((carrier_pose_at(slot), None, False, 0.0, APPROACH[slot])); carried = s[1]; pyaw = s[2]
        elif s[0] == 'goto':
            pts.append(((s[1], s[2]), carried, s[3], pyaw, None))
        else:
            pts.append((carrier_pose_at(s[1]), carried, False, pyaw, APPROACH[s[1]]))
            where[carried] = s[1]; carried = None
    return pts


# ------------------------------------------------------------------- checks
def check():
    problems = []
    boxes = [(aabb(o['model'], o['x'], o['y'], o['yaw']),
              '%s @ %s (%.1f, %.1f)' % (o['model'], o['zone'], o['x'], o['y']), o['group'])
             for o in STATIC]
    for (b, name, _), o in zip(boxes, STATIC):
        if o['zone'] in OUTSIDE:
            continue
        if b[0] < -HALF or b[1] < -HALF or b[2] > HALF or b[3] > HALF:
            problems.append('outside the walls: ' + name)
    moving = [o['zone'] == 'traffic' for o in STATIC]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if moving[i] or moving[j]:
                continue          # a carrier parks at its first pick: that closeness is the job
            if STATIC[i]['zone'] in OUTSIDE or STATIC[j]['zone'] in OUTSIDE:
                continue          # doors, trailers and the yard slab are stacked by construction
            a = boxes[i][0]; b = boxes[j][0]
            same_run = boxes[i][2] is not None and boxes[i][2] == boxes[j][2]
            pad = -0.001 if same_run else MARGIN
            if overlap((a[0] - pad, a[1] - pad, a[2] + pad, a[3] + pad), b):
                problems.append('static objects %s: %s | %s'
                                % ('overlap' if same_run else 'within %.2f m' % MARGIN,
                                   boxes[i][1], boxes[j][1]))
    return problems


def path_report():
    """Per carrier: the smallest clearance between what its legs sweep and the
    statics that are not part of its own job.  On a straight leg the outfit
    (carrier box, plus the load box riding CARRY ahead with the heading it had
    at pick-up) is swept along the leg; on a reverse leg the carrier faces
    away from the direction of travel, so the outfit is mirrored.  At each
    waypoint the carrier turns in place from the facing it arrived with to
    the facing it leaves with, the shorter way round, and the outfit's
    outline is sampled along that arc.  Negative = overlap.  A leg overlap
    blocks the write; a turn overlap is reported, since a swing clipping a
    corner is a visual blemish rather than a path that cannot exist."""
    out = []
    model_of = {o['name']: o['model'] for o in STATIC}
    label = {o['name']: '%s @ %s (%.1f, %.1f)' % (o['model'], o['zone'], o['x'], o['y'])
             for o in STATIC}
    jack = aabb('PalletJackB_01', 0.0, 0.0, 0.0)

    def box_dist(pt, b):
        dx = max(b[0] - pt[0], 0, pt[0] - b[2]); dy = max(b[1] - pt[1], 0, pt[1] - b[3])
        if dx == 0 and dy == 0:
            return -min(pt[0] - b[0], b[2] - pt[0], pt[1] - b[1], b[3] - pt[1])
        return math.hypot(dx, dy)

    for carrier in CARRIERS:
        key = carrier[0]
        mine = {NAMES[k] for k in initial_slots(carrier)} | {NAMES[key]}
        others = [(aabb(o['model'], o['x'], o['y'], o['yaw']), label[o['name']])
                  for o in STATIC if o['name'] not in mine and o['zone'] != 'traffic']
        pts = replay(carrier)
        worst_leg = (1e9, ''); worst_turn = (1e9, '')
        facing_prev = pts[0][4]                 # parked facing its first slot
        last_outline = None
        for (p0, _, _, _, _), (p1, carried, back, pyaw, stand) in zip(pts, pts[1:]):
            def sweep(at, f0, f1):
                nonlocal worst_turn, outline
                d = f1 - f0
                while d > math.pi: d -= 2 * math.pi
                while d < -math.pi: d += 2 * math.pi
                for k in range(13):
                    a = f0 + d * k / 12
                    c, sn = math.cos(a), math.sin(a)
                    for (ox, oy) in outline:
                        pt = (at[0] + c * ox - sn * oy, at[1] + sn * ox + c * oy)
                        for b, name in others:
                            dd = box_dist(pt, b)
                            if dd < worst_turn[0]:
                                worst_turn = (dd, 'turn at (%.1f, %.1f)%s near %s'
                                              % (at[0], at[1], ' carrying ' + carried if carried else '', name))


            box = jack
            if carried:
                lb = aabb(model_of[NAMES[carried]], CARRY, 0.0, -pyaw)
                box = (min(jack[0], lb[0]), min(jack[1], lb[1]), max(jack[2], lb[2]), max(jack[3], lb[3]))
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            L = math.hypot(dx, dy)
            if L < 1e-3:
                # the stand coincides with the path's last waypoint: no leg,
                # but the carrier still aligns to the slot heading there,
                # with the outfit it arrived with
                if stand is not None and last_outline is not None:
                    outline = last_outline
                    sweep(p1, facing_prev, stand)
                    facing_prev = stand
                continue
            heading = math.atan2(dy, dx)
            facing = heading + (math.pi if back else 0.0)
            # outline points of the outfit in the carrier frame
            x0, y0, x1, y1 = box
            outline = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), ((x0 + x1) / 2, y0),
                       ((x0 + x1) / 2, y1), (x0, (y0 + y1) / 2), (x1, (y0 + y1) / 2)]
            last_outline = outline
            # -- the turn at p0 from the facing we arrived with to this leg's
            if facing_prev is not None:
                sweep(p0, facing_prev, facing)
            # -- the leg itself: outfit swept along the leg, as an AABB
            c, sn = math.cos(facing), math.sin(facing)
            corners = []
            for (ox, oy) in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
                for (px, py) in (p0, p1):
                    corners.append((px + c * ox - sn * oy, py + sn * ox + c * oy))
            corr = (min(q[0] for q in corners), min(q[1] for q in corners),
                    max(q[0] for q in corners), max(q[1] for q in corners))
            for b, name in others:
                dd = max(corr[0] - b[2], b[0] - corr[2], corr[1] - b[3], b[1] - corr[3])
                if dd < worst_leg[0]:
                    worst_leg = (dd, 'leg to (%.1f, %.1f)%s near %s'
                                 % (p1[0], p1[1], ' carrying ' + carried if carried else '', name))
            if os.environ.get('WORLD_DEBUG') == key:
                print('    %s leg %5.1f,%5.1f -> %5.1f,%5.1f %s%s' % (
                    key, p0[0], p0[1], p1[0], p1[1], 'carrying %s ' % carried if carried else '',
                    'REVERSE' if back else ''))
            # -- at a stand the carrier aligns to the slot heading (the plugin's
            # ALIGN phase) with this leg's outfit still on the forks
            if stand is not None:
                sweep(p1, facing, stand)
                facing_prev = stand
            else:
                facing_prev = facing
        out.append((key, carrier[4], len(expand(carrier)), worst_leg, worst_turn))
    return out


def route_clearance():
    import csv, glob
    out = {}
    for f in sorted(glob.glob(os.path.join(GT_DIR, 'dataset_dynamic_*.csv'))):
        pts = [(float(r[1]), float(r[2])) for r in csv.reader(open(f))][::10]
        worst = (1e9, None, None)
        for o in STATIC:
            b = aabb(o['model'], o['x'], o['y'], o['yaw'])
            for (px, py) in pts:
                dx = max(b[0] - px, 0, px - b[2]); dy = max(b[1] - py, 0, py - b[3])
                d = math.hypot(dx, dy)
                if d < worst[0]:
                    worst = (d, '%s @ %s' % (o['model'], o['zone']), (px, py))
        out[os.path.basename(f)[:-4]] = worst
    return out


# ------------------------------------------------------------------- output
INCLUDE = """    <include>
      <uri>model://aws_robomaker_warehouse_%(model)s</uri>
      <name>%(name)s</name>
      <pose>%(x).6f %(y).6f %(z).6f 0 0 %(yaw).6f</pose>%(static)s
    </include>
"""


def plugin_block():
    out = ['    <!-- Traffic: carriers on logistic jobs, driven by the LogisticsScheduler\n'
           '         system (gz_kinematic_trajectory/LogisticsScheduler.cc). A job loops\n'
           '         forever: pick a load, carry it through the aisles, set it down in the\n'
           '         zone it belongs in. Generated by script/make_realistic_world.py. -->\n'
           '    <plugin filename="LogisticsScheduler" name="logistics::LogisticsScheduler">\n']
    for carrier in CARRIERS:
        key, speed, handling, start, desc, moves, path = carrier
        out.append('      <!-- %s: %s -->\n' % (key, desc))
        out.append('      <carrier name="%s">\n' % NAMES[key])
        out.append('        <speed>%.2f</speed>\n' % speed)
        out.append('        <turn_rate>1.5</turn_rate>\n')
        out.append('        <carry_offset>%.2f 0 0</carry_offset>\n' % CARRY)
        out.append('        <handling_time>%.1f</handling_time>\n' % handling)
        out.append('        <start>%.1f</start>\n' % start)
        out.append('        <job>\n')
        for s in expand(carrier):
            if s[0] == 'pick':
                out.append('          <pick load="%s">%.6f</pick>\n' % (NAMES[s[1]], s[2]))
            elif s[0] == 'goto':
                out.append('          <goto%s>%.3f %.3f</goto>\n'
                           % (' reverse="true"' if s[3] else '', s[1], s[2]))
            else:
                x, y, yaw = SLOTS[s[1]]
                # yaw: how the load is parked; heading: the side the jack comes from
                out.append('          <drop heading="%.6f">%.3f %.3f %.6f</drop>\n'
                           % (APPROACH[s[1]], x, y, yaw))
        out.append('        </job>\n      </carrier>\n')
    out.append('    </plugin>\n')
    return ''.join(out)


def emit():
    src = open(WORLD_U).read()                # the shell/gui template is the U world
    head = src[:src.index('    <include>')]
    # a previous generation of this file carries its own plugin block; drop it
    head = re.sub(r'    <!-- Traffic: carriers.*?</plugin>\n\n?', '', head, flags=re.S)
    keep = []
    # one include per match (the body may not contain another <include>): the
    # stock template ends a commented-out run of includes with '</include> -->',
    # and a lazy '.*?' once spanned from that desk (DeskC_01_003, inside a rack)
    # into the ground include and leaked the desk into the output
    for m in re.finditer(r'    <include>\n((?:(?!<include>).)*?)    </include>[^\n]*\n',
                         re.sub(r'<!--.*?-->', '', src, flags=re.S), flags=re.S):
        if any(k in m.group(1) for k in ('RoofB_01', 'GroundB_01_plain')):
            keep.append(m.group(0))
    # the shell: the generated wall with the dock-door openings, under the stock
    # include name so anything keyed on the name still finds the wall
    keep.append('    <include>\n      <uri>model://%s</uri>\n'
                '      <name>aws_robomaker_warehouse_WallB_01_001</name>\n'
                '      <pose>0.0 0.0 0 0 0 0</pose>\n    </include>\n' % DOORS_BY_LAYOUT[LAYOUT]['wall'])
    tail = src[src.index('    <!-- ackermann_robot -->'):]

    out = [head]
    out.append(plugin_block())
    out.append('\n')
    out += keep
    out.append('\n    <!-- ===== BEGIN layout : generated by script/make_realistic_world.py ===== -->\n')
    if LAYOUT == 'u':
        out.append('    <!-- U-shape flow: receiving and dispatch docks on the north wall with the\n'
                   '         management office between them, fast pick faces nearest the docks, rack\n'
                   '         storage in N-S aisles, picking and packing east, block storage west and\n'
                   '         south-west, MHE charging / empty pallets / general office on the south\n'
                   '         wall (Mecalux warehouse-layout manual). Main aisles A (x -11.2) and B\n'
                   '         (x 5.25) and cross-aisles C (y ~7-8) and E (y ~-15.9) match the recorded\n'
                   '         teleop route. Loads and carriers the jobs use are named in the plugin. -->\n')
    else:
        out.append('    <!-- I-shape (through) flow: receiving docks on the north wall, dispatch\n'
                   '         docks on the south wall, rack storage in the middle, the fast pick\n'
                   '         faces between storage and dispatch, packing beside dispatch, block\n'
                   '         storage west, offices on the east wall. Goods go straight through.\n'
                   '         Main aisles A (x -11.2) and B (x 5.25) are kept; the recorded route\'s\n'
                   '         y ~5.8 and y ~-15.4 legs are not drivable in this layout. -->\n')
    moved = set(NAMES.values())
    zone = None
    for o in STATIC:
        if o['zone'] != zone:
            cnt = sum(1 for s in STATIC if s['zone'] == o['zone'])
            out.append('    <!-- ==== %s : %d object(s) ==== -->\n' % (o['zone'], cnt))
            zone = o['zone']
        out.append(INCLUDE % dict(model=o['model'], name=o['name'], x=o['x'], y=o['y'],
                                  z=Z[o['model']], yaw=o['yaw'],
                                  static='\n      <static>true</static>' if o['name'] in moved else ''))
    out.append('    <!-- ===== END layout ===== -->\n\n')
    out.append(tail)
    return ''.join(out)


def plot(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import csv, glob
    colours = {'receiving staging': '#4C72B0', 'dispatch staging': '#DD8452',
               'pick faces': '#55A868', 'storage (medium)': '#8172B3', 'picking': '#64B5CD',
               'packing': '#CCB974', 'block storage': '#937860', 'MHE charging': '#DA8BC3',
               'empty pallets': '#8C8C8C', 'general office': '#C44E52',
               'management office': '#E57373', 'traffic': '#212121',
               'dock doors': '#1565C0', 'dock yard': '#B0BEC5'}
    fig, ax = plt.subplots(figsize=(14.5, 17 if DOORS_BY_LAYOUT[LAYOUT]['south'] else 14))
    ax.add_patch(Rectangle((-20.64, -20.64), 41.28, 41.28, fill=False, lw=2, color='k'))
    seen = set()
    for o in STATIC:
        b = aabb(o['model'], o['x'], o['y'], o['yaw'])
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                               color=colours[o['zone']], alpha=0.85,
                               label=o['zone'] if o['zone'] not in seen else None))
        seen.add(o['zone'])
    for key, (x, y, _) in SLOTS.items():
        ax.add_patch(Rectangle((x - 1.0, y - 1.0), 2.0, 2.0, fill=False, ls=':', lw=1.0,
                               color='#C62828'))
        ax.text(x, y, key, ha='center', va='center', fontsize=7, color='#C62828')
    styles = {'C1': '#C62828', 'C2': '#AD1457', 'C3': '#EF6C00', 'C4': '#6A1B9A'}
    for carrier in CARRIERS:
        pts = [p[0] for p in replay(carrier)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], '-', color=styles[carrier[0]],
                lw=1.0, alpha=0.8, label='%s %s' % (carrier[0], carrier[4]))
    for f in sorted(glob.glob(os.path.join(GT_DIR, 'dataset_dynamic_*.csv')))[:1]:
        pts = [(float(r[1]), float(r[2])) for r in csv.reader(open(f))]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color='k', lw=0.8, alpha=0.5,
                label='recorded teleop route (dyn_00_000)')
    ax.plot(5.25, 20, 'k^', ms=8, label='robot spawn')
    for txt, (x, y) in LAYOUTS[LAYOUT]['labels'].items():
        ax.text(x, y, txt, ha='center', va='center', fontsize=8,
                rotation=90 if txt.startswith('aisle') else 0)
    # the wall openings, drawn as gaps in the shell line
    lay = DOORS_BY_LAYOUT[LAYOUT]
    for xc, state in lay['north']:
        ax.add_patch(Rectangle((xc - DOOR_W / 2, YI1 - 0.05), DOOR_W, YO1 - YI1 + 0.1, color='white', zorder=3))
        ax.text(xc, YO1 + 0.9, 'dock door\n(%s)' % state, ha='center', va='bottom', fontsize=6.5, zorder=4)
    for xc, state in lay['south']:
        ax.add_patch(Rectangle((xc - DOOR_W / 2, -YO1 - 0.05), DOOR_W, YO1 - YI1 + 0.1, color='white', zorder=3))
        ax.text(xc, -YO1 - 0.9, 'dock door\n(%s)' % state, ha='center', va='top', fontsize=6.5, zorder=4)
    ax.set_xlim(-21.5, 21.5); ax.set_ylim(-36.5 if lay['south'] else -21.5, 36.5); ax.set_aspect('equal')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title('%s: %s' % (os.path.basename(WORLD)[:-6], LAYOUTS[LAYOUT]['title']))
    # legend beside the map, not over it
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False,
              borderaxespad=0.0)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches='tight')
    print('plan written to', path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true', help='run the checks, write nothing')
    ap.add_argument('--plot', action='store_true', help='also write the plan PNG next to the world')
    ap.add_argument('--layout', choices=sorted(LAYOUTS), default='u',
                    help="u: U-shape flow (docks on the north wall); i: I-shape through flow "
                         "(receiving north, dispatch south), written to the _01 world")
    a = ap.parse_args()
    select_layout(a.layout)
    print('  layout %s -> %s' % (a.layout, os.path.relpath(WORLD, ROOT)))

    for m in sorted(FOOT):
        dx, dy = extents(m, 0.0)
        print('  footprint %-16s %.3f x %.3f m' % (m, dx, dy))
    build_static()
    for slot in SLOTS:
        x, y, _ = SLOTS[slot]
        occ = [o for o in STATIC if abs(o['x'] - x) < 1e-6 and abs(o['y'] - y) < 1e-6]
        if slot in EMPTY and occ:
            raise SystemExit('slot %s should be empty but holds %s' % (slot, occ[0]['name']))
        if slot not in EMPTY and not occ:
            raise SystemExit('slot %s should hold a load' % slot)
    problems = check()
    print('\n  %d static objects in %d zones, %d carriers' %
          (len(STATIC), len(set(o['zone'] for o in STATIC)), len(CARRIERS)))
    for p in problems:
        print('  PROBLEM:', p)
    for key, desc, nsteps, (d, leg), (dt, turn) in path_report():
        note = '  leg clearance %.2f m (%s); turn sweep %.2f m (%s)' % (d, leg, dt, turn)
        if d < 0:
            problems.append('%s corridor overlaps a static object on %s' % (key, leg))
            note += '  <-- OVERLAP'
        print('  %s %-46s %2d steps\n       %s' % (key, desc, nsteps, note))
    for name, (d, obj, pt) in route_clearance().items():
        flag = '' if d >= ROBOT_HALF + MARGIN else '  <-- tighter than robot half-width + margin'
        print('  route clearance %-42s %.2f m at (%.1f, %.1f) from %s%s'
              % (name, d, pt[0], pt[1], obj, flag))
    if a.plot:
        plot(WORLD[:-6] + '_plan.png')
    if problems:
        print('\n  not written: fix the problems above')
        return 1
    if a.check:
        return 0
    text = emit()
    open(WORLD, 'w').write(text)
    print('  wrote', os.path.relpath(WORLD, ROOT), '(%d lines)' % text.count('\n'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
