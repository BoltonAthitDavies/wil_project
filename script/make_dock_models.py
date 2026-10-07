#!/usr/bin/env python3
"""Generate the dock-door models for the realistic warehouse world.

    python3 script/make_dock_models.py

Writes, under aws-robomaker-small-warehouse-world/models/:
    aws_robomaker_warehouse_WallB_01_docks/   the warehouse shell with door
                                              openings in the north wall (U-shape)
    aws_robomaker_warehouse_WallB_01_docks_i/ openings north AND south (I-shape)
    aws_robomaker_warehouse_DockDoor_01/      a closed roll-up door (fills an opening)
    aws_robomaker_warehouse_DockDoorOpen_01/  the same door rolled up: drum at the
                                              lintel and the two guide rails
    aws_robomaker_warehouse_Trailer_01/       a docked semi-trailer, rear open
    aws_robomaker_warehouse_DockYard_01/      a slab of yard outside the wall

WHY A NEW WALL MODEL
    The stock shell (WallB_01) has no openings and its dimensions are frozen
    in COLLADA vertex data, not parameters.  Its geometry is simple -- a ring
    13 cm thick, 9.02 m high, 42 x 41.8 m outside -- so instead of editing
    the mesh the shell is REGENERATED here as box slabs, with the north slab
    split around the door openings and a lintel over each.  The original
    model is untouched; the realistic world includes this one instead.

    The original's texture is a plain plaster tile stretched about once over
    each wall (u ~0.58 per wall length, v ~0.88 per wall height); the new
    faces use planar UVs at that scale with the same image, so the shell
    looks the same from inside.

DOCK DOORS
    Five openings on the north (dock) wall, 3.6 m wide, 4.5 m high: two at
    the receiving dock, one on the robot's dock lane (it spawns in front of
    it), two at the dispatch dock.  DOORS below is the single source of their
    positions; make_realistic_world.py imports it to place the door panels
    and keeps that lane clear.  An open door shows a docked trailer with its
    rear doors open, standing on a yard slab, so a camera looking out sees a
    trailer interior rather than the sky colour.

MESH FORMAT
    Minimal COLLADA 1.4.1 written by hand (positions, normals, UVs, one
    triangle list, one textured material), in metres, Z up -- the subset
    Gazebo's Assimp loader needs.  Collision meshes are the same boxes
    without materials.
"""

import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, 'aws-robomaker-small-warehouse-world', 'models')

# shell geometry, from the stock WallB_01 vertex data (metres)
XO0, XO1 = -21.045, 21.045        # outer x
YO0, YO1 = -20.907, 20.907        # outer y
XI0, XI1 = -20.657, 20.627        # inner x (the stock mesh is 3 cm asymmetric)
YI0, YI1 = -20.642, 20.642        # inner y
H = 9.02                          # wall height
U_PER_M, V_PER_M = 0.58 / 41.3, 0.88 / 9.02   # the stock texture's stretch

# the dock doors per layout: x centre and state ('closed' | 'open'), on the
# north wall and on the south wall.  U-shape: receiving and dispatch share the
# north wall.  I-shape (through flow): receiving north, dispatch south.
DOOR_W, DOOR_H = 3.6, 4.5
DOORS_BY_LAYOUT = {
    'u': dict(wall='aws_robomaker_warehouse_WallB_01_docks',
              north=[(-15.0, 'open'), (-8.0, 'closed'), (5.25, 'closed'), (12.0, 'closed'), (18.0, 'open')],
              south=[]),
    'i': dict(wall='aws_robomaker_warehouse_WallB_01_docks_i',
              north=[(-15.0, 'closed'), (-8.0, 'open'), (5.25, 'closed'), (12.0, 'closed'), (18.0, 'open')],
              south=[(-15.0, 'open'), (-8.0, 'closed'), (12.0, 'closed')]),
}
DOORS = DOORS_BY_LAYOUT['u']['north']       # kept for older callers


# ------------------------------------------------------------------- COLLADA
def box_faces(x0, y0, z0, x1, y1, z1, inward=False):
    """Six quads of a box as (4 corners, normal), outward normals (or inward,
    for the inside of a hollow trailer)."""
    faces = [
        ([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], (0, -1, 0)),   # -y
        ([(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)], (0, 1, 0)),    # +y
        ([(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)], (-1, 0, 0)),   # -x
        ([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], (1, 0, 0)),    # +x
        ([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], (0, 0, 1)),    # +z
        ([(x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0)], (0, 0, -1)),   # -z
    ]
    if inward:
        faces = [(list(reversed(c)), (-n[0], -n[1], -n[2])) for c, n in faces]
    return faces


def planar_uv(p, n, su, sv):
    """UV from the two in-plane world axes of a face, at a texture scale."""
    x, y, z = p
    if abs(n[2]) > 0.5:
        return (x * su, y * su)
    a = x if abs(n[1]) > 0.5 else y
    return (a * su, z * sv)


class Mesh:
    def __init__(self):
        self.pos, self.nrm, self.uv, self.idx = [], [], [], []

    def quad(self, corners, normal, su=U_PER_M, sv=V_PER_M):
        base = len(self.pos)
        for c in corners:
            self.pos.append(c); self.nrm.append(normal); self.uv.append(planar_uv(c, normal, su, sv))
        for a, b, c in ((0, 1, 2), (0, 2, 3)):
            self.idx += [base + a, base + b, base + c]

    def box(self, x0, y0, z0, x1, y1, z1, inward=False, su=U_PER_M, sv=V_PER_M, skip=()):
        for k, (corners, normal) in enumerate(box_faces(x0, y0, z0, x1, y1, z1, inward)):
            if k in skip:
                continue
            self.quad(corners, normal, su, sv)

    def dae(self, name, texture=None, colour=(0.8, 0.8, 0.8)):
        n = len(self.pos)
        P = ' '.join('%.4f %.4f %.4f' % p for p in self.pos)
        N = ' '.join('%.3f %.3f %.3f' % v for v in self.nrm)
        T = ' '.join('%.5f %.5f' % t for t in self.uv)
        tris = len(self.idx) // 3
        pidx = ' '.join('%d %d %d' % (i, i, i) for i in self.idx)
        if texture:
            effect = ('<newparam sid="img-surface"><surface type="2D"><init_from>img</init_from></surface></newparam>'
                      '<newparam sid="img-sampler"><sampler2D><source>img-surface</source></sampler2D></newparam>'
                      '<technique sid="common"><lambert><diffuse><texture texture="img-sampler" texcoord="UVSET0"/></diffuse></lambert></technique>')
            images = '<library_images><image id="img"><init_from>%s</init_from></image></library_images>' % texture
        else:
            effect = ('<technique sid="common"><lambert><diffuse><color>%.3f %.3f %.3f 1</color></diffuse></lambert></technique>'
                      % colour)
            images = ''
        return '''<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><contributor><authoring_tool>script/make_dock_models.py</authoring_tool></contributor>
    <unit meter="1" name="meter"/><up_axis>Z_UP</up_axis></asset>
  %s
  <library_effects><effect id="mat-fx"><profile_COMMON>%s</profile_COMMON></effect></library_effects>
  <library_materials><material id="mat" name="mat"><instance_effect url="#mat-fx"/></material></library_materials>
  <library_geometries>
    <geometry id="%s-lib" name="%s"><mesh>
      <source id="%s-pos"><float_array id="%s-POSITION-array" count="%d">%s</float_array>
        <technique_common><accessor source="#%s-POSITION-array" count="%d" stride="3">
          <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
      <source id="%s-nrm"><float_array id="%s-Normal0-array" count="%d">%s</float_array>
        <technique_common><accessor source="#%s-Normal0-array" count="%d" stride="3">
          <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
      <source id="%s-uv"><float_array id="%s-UV0-array" count="%d">%s</float_array>
        <technique_common><accessor source="#%s-UV0-array" count="%d" stride="2">
          <param name="S" type="float"/><param name="T" type="float"/></accessor></technique_common></source>
      <vertices id="%s-vtx"><input semantic="POSITION" source="#%s-pos"/></vertices>
      <triangles material="mat" count="%d">
        <input semantic="VERTEX" source="#%s-vtx" offset="0"/>
        <input semantic="NORMAL" source="#%s-nrm" offset="1"/>
        <input semantic="TEXCOORD" source="#%s-uv" offset="2" set="0"/>
        <p>%s</p></triangles>
    </mesh></geometry>
  </library_geometries>
  <library_visual_scenes><visual_scene id="scene" name="scene">
    <node id="%s" name="%s"><instance_geometry url="#%s-lib">
      <bind_material><technique_common><instance_material symbol="mat" target="#mat">
        <bind_vertex_input semantic="UVSET0" input_semantic="TEXCOORD" input_set="0"/></instance_material></technique_common></bind_material>
    </instance_geometry></node></visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#scene"/></scene>
</COLLADA>
''' % (images, effect, name, name, name, name, 3 * n, P, name, n, name, name, 3 * n, N, name, n,
       name, name, 2 * n, T, name, n, name, name, tris, name, name, name, pidx, name, name, name)


# ------------------------------------------------------------------- models
SDF = '''<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="%(name)s">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry><mesh><uri>model://%(name)s/meshes/%(name)s_collision.DAE</uri></mesh></geometry>
      </collision>
      <visual name="visual">
        <geometry><mesh><uri>model://%(name)s/meshes/%(name)s_visual.DAE</uri></mesh></geometry>
      </visual>
    </link>
  </model>
</sdf>
'''

CONFIG = '''<?xml version="1.0"?>
<model>
  <name>%(name)s</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>
  <author><name>WiL project</name></author>
  <description>%(desc)s. Generated by script/make_dock_models.py.</description>
</model>
'''


def write_model(name, desc, visual, collision, texture_src=None, texture_name=None):
    d = os.path.join(MODELS, name)
    os.makedirs(os.path.join(d, 'meshes'), exist_ok=True)
    tex = None
    if texture_src:
        os.makedirs(os.path.join(d, 'materials', 'textures'), exist_ok=True)
        shutil.copy(texture_src, os.path.join(d, 'materials', 'textures', texture_name))
        tex = '../materials/textures/' + texture_name
    open(os.path.join(d, 'meshes', name + '_visual.DAE'), 'w').write(visual.dae(name + '_visual', tex, visual.colour))
    open(os.path.join(d, 'meshes', name + '_collision.DAE'), 'w').write(collision.dae(name + '_collision'))
    open(os.path.join(d, 'model.sdf'), 'w').write(SDF % dict(name=name))
    open(os.path.join(d, 'model.config'), 'w').write(CONFIG % dict(name=name, desc=desc))
    print('  wrote %-44s %4d tris visual, %4d tris collision' % (name, len(visual.idx) // 3, len(collision.idx) // 3))


def slab_with_openings(m, y0, y1, doors):
    """An east-west wall slab between y0 and y1, split around door openings,
    with a lintel over each opening."""
    edges = [XO0]
    for xc, _ in sorted(doors):
        edges += [xc - DOOR_W / 2, xc + DOOR_W / 2]
    edges.append(XO1)
    for a, b in zip(edges[0::2], edges[1::2]):           # wall segments
        m.box(a, y0, 0, b, y1, H)
    for a, b in zip(edges[1::2], edges[2::2]):           # lintels over the openings
        m.box(a, y0, DOOR_H, b, y1, H)


def wall_with_docks(north, south):
    m = Mesh(); m.colour = (0.8, 0.8, 0.8)
    slab_with_openings(m, YO0, YI0, south)               # south slab
    m.box(XO0, YI0, 0, XI0, YI1, H)                      # west slab
    m.box(XI1, YI0, 0, XO1, YI1, H)                      # east slab
    slab_with_openings(m, YI1, YO1, north)               # north slab
    return m


def door_closed():
    m = Mesh(); m.colour = (0.55, 0.60, 0.66)
    # a roll-up door: the panel across the opening, 8 cm thick, in the middle
    # of the wall thickness; the model origin is the opening's floor centre
    m.box(-DOOR_W / 2, -0.04, 0.0, DOOR_W / 2, 0.04, DOOR_H, su=0.4, sv=1.0 / 0.45)
    for s in (-1, 1):                                    # guide rails
        m.box(s * DOOR_W / 2 - (0.08 if s > 0 else 0), -0.08, 0, s * DOOR_W / 2 + (0.08 if s < 0 else 0), 0.08, DOOR_H + 0.3,
              su=0.4, sv=1.0 / 0.45)
    return m


def door_open():
    m = Mesh(); m.colour = (0.55, 0.60, 0.66)
    m.box(-DOOR_W / 2, -0.28, DOOR_H + 0.02, DOOR_W / 2, 0.28, DOOR_H + 0.58, su=0.4, sv=1.0 / 0.45)   # the drum
    for s in (-1, 1):
        m.box(s * DOOR_W / 2 - (0.08 if s > 0 else 0), -0.08, 0, s * DOOR_W / 2 + (0.08 if s < 0 else 0), 0.08, DOOR_H + 0.3,
              su=0.4, sv=1.0 / 0.45)
    return m


def trailer():
    """A 13.6 m box semi-trailer, bed 1.2 m up, rear end open and the inside
    visible; origin at the rear-centre of the bed, y pointing away from the
    dock."""
    m = Mesh(); m.colour = (0.85, 0.85, 0.82)
    W, L, Hh, BED = 2.55, 13.6, 2.7, 1.2
    t = 0.06
    # floor, roof, two sides, front (closed) -- as thin slabs so both faces draw
    m.box(-W / 2, 0, BED, W / 2, L, BED + t, su=0.2, sv=0.2)
    m.box(-W / 2, 0, BED + Hh - t, W / 2, L, BED + Hh, su=0.2, sv=0.2)
    m.box(-W / 2, 0, BED, -W / 2 + t, L, BED + Hh, su=0.2, sv=0.3)
    m.box(W / 2 - t, 0, BED, W / 2, L, BED + Hh, su=0.2, sv=0.3)
    m.box(-W / 2, L - t, BED, W / 2, L, BED + Hh, su=0.2, sv=0.3)
    # rear doors swung right round, lying flat along the outside of the sides
    for s in (-1, 1):
        m.box(s * (W / 2 + 0.02) - (0.03 if s > 0 else 0), 0.0, BED, s * (W / 2 + 0.02) + (0.03 if s < 0 else 0), 2.4, BED + Hh,
              su=0.2, sv=0.3)
    # chassis and axles
    m.box(-0.5, 1.0, 0.5, 0.5, L - 0.5, BED, su=0.3, sv=0.3)
    for y in (L - 3.6, L - 2.3):
        for s in (-1, 1):
            m.box(s * 1.0 - 0.15, y - 0.5, 0.0, s * 1.0 + 0.15, y + 0.5, 1.0, su=0.3, sv=0.3)
    # landing legs near the front
    for s in (-1, 1):
        m.box(s * 0.9 - 0.08, L - 11.5, 0.0, s * 0.9 + 0.08, L - 11.3, BED, su=0.3, sv=0.3)
    return m


def yard():
    m = Mesh(); m.colour = (0.42, 0.42, 0.40)
    m.box(XO0 - 2, YO1, -0.02, XO1 + 2, YO1 + 18.0, 0.0, su=0.05, sv=0.05)
    return m


def main():
    wall_tex = os.path.join(MODELS, 'aws_robomaker_warehouse_WallB_01', 'materials', 'textures',
                            'aws_robomaker_warehouse_WallB_01.png')
    for key, lay in DOORS_BY_LAYOUT.items():
        w = wall_with_docks(lay['north'], lay['south'])
        write_model(lay['wall'],
                    'Warehouse shell, %s-shape flow: %d dock-door openings north, %d south'
                    % (key.upper(), len(lay['north']), len(lay['south'])),
                    w, w, wall_tex, 'aws_robomaker_warehouse_WallB_01.png')
    d = door_closed(); write_model('aws_robomaker_warehouse_DockDoor_01', 'Closed roll-up dock door, %.1f x %.1f m' % (DOOR_W, DOOR_H), d, d)
    o = door_open(); write_model('aws_robomaker_warehouse_DockDoorOpen_01', 'Rolled-up dock door: drum and guide rails', o, o)
    t = trailer(); write_model('aws_robomaker_warehouse_Trailer_01', 'Docked box semi-trailer, rear open', t, t)
    y = yard(); write_model('aws_robomaker_warehouse_DockYard_01', 'Yard slab outside the dock wall', y, y)
    for key, lay in DOORS_BY_LAYOUT.items():
        print('  %s: north %s; south %s' % (key, ', '.join('%.2f (%s)' % d for d in lay['north']),
                                            ', '.join('%.2f (%s)' % d for d in lay['south']) or '-'))


if __name__ == '__main__':
    main()
