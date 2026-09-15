#!/usr/bin/env python3
"""Ground-truth 2D boxes for the warehouse sim, from Gazebo model poses.

Turns `/world/default/pose/info` (every model's true world pose) plus the props'
collision meshes into per-frame 2D bounding boxes in the cam0 image, so YOLO can
be scored against ground truth instead of eyeballed.

WHY THE COLLISION MESH AND NOT A BOX
    Projecting the 8 corners of a 3D axis-aligned box gives a 2D box noticeably
    larger than the object's real silhouette once it is rotated, which inflates
    ground truth and depresses measured IoU for a detector that is actually
    right. We project the mesh vertices themselves and take their 2D extent.

TWO UNIT TRAPS, both silent if you get them wrong:
  * these DAEs declare <unit meter="0.01"> -- the vertex data is in CENTIMETRES.
  * the scene graph carries per-node <matrix> transforms that must be composed;
    the Bucket's is a ~0.58 cm x-offset, small enough to look like noise and big
    enough to shift a box.
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

import numpy as np

WORLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                     'aws-robomaker-small-warehouse-world')
MODELS = os.path.normpath(os.path.join(WORLD, 'models'))

# Gazebo model-name prefix -> YOLO class name. All Cluttering variants are one
# class by decision; the detector was trained that way.
CLASS_OF = {
    'aws_robomaker_warehouse_Bucket_01':      'bucket',
    'aws_robomaker_warehouse_ClutteringA_01': 'box',
    'aws_robomaker_warehouse_ClutteringC_01': 'box',
    'aws_robomaker_warehouse_ClutteringD_01': 'box',
    'aws_robomaker_warehouse_TrashCanC_01':   'bin',
}


# Models that are NOT ground truth but DO contain box-like geometry the detector
# will legitimately fire on. The AWS shelves are modelled loaded with cardboard
# boxes, and the pallet jack carries a pallet. Scoring a detection there as a
# false positive would punish the detector for being right about something this
# ground truth simply does not label, so they become IGNORE regions instead:
# a prediction landing on one counts as neither a hit nor a miss.
#
# (`warehouse_fill` is a directional LIGHT, not geometry -- it turns up in
# pose/info but has nothing to project.)
IGNORE_MODELS = (
    'aws_robomaker_warehouse_ShelfD_01',
    'aws_robomaker_warehouse_ShelfE_01',
    'aws_robomaker_warehouse_ShelfF_01',
    'aws_robomaker_warehouse_PalletJackB_01',
    'aws_robomaker_warehouse_DeskC_01',
)


def ignore_prefix(frame_name: str) -> str | None:
    for p in sorted(IGNORE_MODELS, key=len, reverse=True):
        if frame_name.startswith(p):
            return p
    return None


def model_prefix(frame_name: str) -> str | None:
    """'aws_robomaker_warehouse_Bucket_01_400' -> 'aws_robomaker_warehouse_Bucket_01'.

    pose/info publishes one entry per model AND per link, with instance numbers
    appended. Longest-prefix match keeps ClutteringA/C/D distinct from each other
    even though they share a stem.
    """
    for p in sorted(CLASS_OF, key=len, reverse=True):
        if frame_name.startswith(p):
            return p
    return None


def _dae_vertices(path: str) -> np.ndarray:
    """Collision DAE -> (N,3) vertices in METRES, model frame."""
    root = ET.parse(path).getroot()
    ns = root.tag.split('}')[0].strip('{')
    q = lambda tag: f'{{{ns}}}{tag}'

    unit = 1.0
    u = root.find(f'{q("asset")}/{q("unit")}')
    if u is not None and u.get('meter'):
        unit = float(u.get('meter'))          # 0.01 for these meshes

    # geometry id -> positions array
    geo = {}
    for g in root.iter(q('geometry')):
        arr = g.find(f'{q("mesh")}/{q("source")}/{q("float_array")}')
        if arr is None or not arr.text:
            continue
        v = np.fromstring(arr.text, sep=' ')
        if v.size >= 3:
            geo[g.get('id')] = v[: (v.size // 3) * 3].reshape(-1, 3)

    def node_matrix(node) -> np.ndarray:
        m = node.find(q('matrix'))
        if m is None or not m.text:
            return np.eye(4)
        return np.fromstring(m.text, sep=' ').reshape(4, 4)

    out = []

    def walk(node, parent):
        T = parent @ node_matrix(node)
        for inst in node.findall(q('instance_geometry')):
            gid = (inst.get('url') or '').lstrip('#')
            if gid in geo:
                v = geo[gid]
                out.append((T[:3, :3] @ v.T).T + T[:3, 3])
        for child in node.findall(q('node')):
            walk(child, T)

    for scene in root.iter(q('visual_scene')):
        for node in scene.findall(q('node')):
            walk(node, np.eye(4))

    if not out:                                # no scene graph: raw geometry
        out = list(geo.values())
    return np.vstack(out) * unit


_cache: dict[str, np.ndarray] = {}


def vertices(prefix: str) -> np.ndarray:
    """Cached (N,3) model-frame vertices in metres for a prop model."""
    if prefix in _cache:
        return _cache[prefix]
    d = os.path.join(MODELS, prefix, 'meshes')
    cand = [f for f in os.listdir(d) if f.lower().endswith('.dae')]
    pick = next((f for f in cand if 'collision' in f.lower()), cand[0])
    v = _dae_vertices(os.path.join(d, pick))
    # Decimate: a few thousand points bound the silhouette as tightly as 100k.
    if len(v) > 4000:
        v = v[np.random.default_rng(0).choice(len(v), 4000, replace=False)]
    _cache[prefix] = v
    return v


if __name__ == '__main__':
    print(f"{'model':44s} {'class':7s} {'verts':>7s}   extent x,y,z (m)")
    for p in sorted(CLASS_OF):
        try:
            v = vertices(p)
        except Exception as e:                 # noqa: BLE001
            print(f'{p:44s} FAILED: {e}')
            continue
        e = v.max(0) - v.min(0)
        print(f'{p:44s} {CLASS_OF[p]:7s} {len(v):7d}   '
              f'{e[0]:.2f} x {e[1]:.2f} x {e[2]:.2f}')
