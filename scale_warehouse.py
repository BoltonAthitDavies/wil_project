#!/usr/bin/env python3
"""Scale the warehouse shell (wall / ground / roof) in the horizontal plane.

The AWS warehouse models carry no parametric dimensions -- every size is frozen
into COLLADA <float_array> vertex data, in centimetres.  Enlarging the room means
rewriting those arrays.  Note you CANNOT write "699.023682*2" into a DAE: the
COLLADA float_array grammar is whitespace-separated literal floats, no expressions.

Scales world X and Y independently.  Height (wall 9.02 m, ground slab 12.4 cm,
roof 1.10 m) is deliberately never scaled.

ALWAYS RUN AGAINST PRISTINE MESHES.  The factors are absolute, measured from the
stock geometry, so this is not idempotent -- running it twice compounds.  Do:

    git -C aws-robomaker-small-warehouse-world checkout -- models/
    python3 scale_warehouse.py 3.0105741651 2.0

Usage:
    python3 scale_warehouse.py FACTOR              # uniform, both axes
    python3 scale_warehouse.py FX FY               # per-axis
    python3 scale_warehouse.py FX FY --uv N        # override UV factor
    ... --dry-run

AXIS NOTE
    RoofB_01_collision.DAE is authored in a rotated frame (its node matrix maps
    local Y -> world Z and local Z -> world Y), so its vertex columns read
    (X, height, worldY) and its node <matrix> carries an in-plane X offset that
    must scale with the geometry.  Every other mesh here has an identity matrix.

UV NOTE
    Textures are tiled (UVs run past 1.0, wrap=TRUE), so scaling geometry without
    scaling UVs makes each tile physically larger and the surface blurrier per
    metre -- which costs visual-SLAM features on the floor.

    The ground's UV mapping is PER-FACE ISLANDS, not one global projection: an
    affine fit of (x,y)->(u,v) leaves residuals of 6-8 UV units, and its two
    triangle groups disagree by 65 degrees on where U points.  A uniform scale is
    correct for any such mapping; a per-axis scale is correct for none of them.
    So UVs are scaled isotropically by min(fx, fy): the less-scaled axis keeps
    exact texel density, and the other stretches by fx/fy.  There is no numeric
    fix for that stretch -- it needs a re-unwrap in Blender.

    Wall UVs are left alone entirely: their mapping is diagonal (U and V both
    track X, Y and Z), so scaling would compress the texture vertically.

    CRITICAL: the ground's two materials share ONE UV0 array but use disjoint index
    sets (24 vs 140).  #946568 is tiling concrete (GroundB_01.png) and DOES scale.
    #946569 is an ATLAS (GroundB_02.png): a 4-band colour strip -- hazard stripes
    0.000-0.342, green 0.342-0.590, blue 0.590-0.820, yellow 0.820-1.000 -- where U
    SELECTS THE PAINTED FLOOR-LINE COLOUR.  Scaling those UVs repaints the lines and
    smears them across band edges (a uniform x2 changed 72 of 82 faces and left 18
    straddling).  They must never be scaled, only the concrete's indices.
"""
import re, sys, os

PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'aws-robomaker-small-warehouse-world', 'models')

X, Y, U, _ = 'x', 'y', 'uv', None

# (model, file, array, stride, per-column factor key, values per output line)
TARGETS = [
    ('aws_robomaker_warehouse_WallB_01',   'visual',    'POSITION', 3, (X, Y, _), 3),
    ('aws_robomaker_warehouse_WallB_01',   'collision', 'POSITION', 3, (X, Y, _), 3),
    ('aws_robomaker_warehouse_GroundB_01', 'visual',    'POSITION', 3, (X, Y, _), 3),
    ('aws_robomaker_warehouse_GroundB_01', 'collision', 'POSITION', 3, (X, Y, _), 3),
    ('aws_robomaker_warehouse_RoofB_01',   'visual',    'POSITION', 3, (X, Y, _), 3),
    ('aws_robomaker_warehouse_RoofB_01',   'collision', 'POSITION', 3, (X, _, Y), 3),
    # Only the CONCRETE material's UVs scale -- see UV NOTE.
    ('aws_robomaker_warehouse_GroundB_01', 'visual',    'UV0',      2, (U, U),    2,
     'Material #946568'),
]

# Node <matrix> translation elements that live in the horizontal plane.
# Element 3 is the X translation; element 11 is Z (height) -- never scale that.
MATRIX_FIX = [('aws_robomaker_warehouse_RoofB_01', 'collision', {3: X})]


def uv_indices_for(text, material):
    """UV indices referenced by one material's <triangles> block."""
    import xml.etree.ElementTree as ET
    NS = '{http://www.collada.org/2005/11/COLLADASchema}'
    root = ET.fromstring(text)
    for tri in root.iter(NS + 'triangles'):
        if tri.get('material') != material:
            continue
        inp = {i.get('semantic'): int(i.get('offset')) for i in tri.findall(NS + 'input')}
        P = [int(x) for x in tri.find(NS + 'p').text.split()]
        st = max(inp.values()) + 1
        return set(P[i + inp['TEXCOORD']] for i in range(0, len(P), st))
    raise SystemExit('material %s not found' % material)


def scale_array(text, model, kind, arr, stride, keys, per_line, f, only=None):
    pat = re.compile(r'(<float_array id="%s_%s-%s-array" count="(\d+)">\n)(.*?)(</float_array>)'
                     % (re.escape(model), kind, arr), re.S)
    m = pat.search(text)
    if not m:
        raise SystemExit('array %s not found in %s_%s' % (arr, model, kind))
    head, count, body, tail = m.group(1), int(m.group(2)), m.group(3), m.group(4)
    vals = body.split()
    if len(vals) != count:
        raise SystemExit('%s_%s %s: count=%d but %d values' % (model, kind, arr, count, len(vals)))
    out = []
    for i, v in enumerate(vals):
        k = keys[i % stride]
        if only is not None and (i // stride) not in only:
            k = None
        out.append('%f' % (float(v) * f[k] if k else float(v)))
    lines = [' '.join(out[i:i + per_line]) for i in range(0, len(out), per_line)]
    return text[:m.start()] + head + '\n'.join(lines) + '\n' + tail + text[m.end():], count // stride


def scale_matrix(text, idxs, f):
    m = re.search(r'(<matrix sid="matrix">)([^<]*)(</matrix>)', text)
    if not m:
        raise SystemExit('no node matrix found')
    vals = m.group(2).split()
    for i, k in idxs.items():
        vals[i] = '%f' % (float(vals[i]) * f[k])
    return text[:m.start()] + m.group(1) + ' '.join(vals) + m.group(3) + text[m.end():]


def main():
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    if not a:
        raise SystemExit(__doc__)
    fx = float(a[0])
    fy = float(a[1]) if len(a) > 1 else fx
    uv = float(sys.argv[sys.argv.index('--uv') + 1]) if '--uv' in sys.argv else min(fx, fy)
    f = {X: fx, Y: fy, U: uv}
    dry = '--dry-run' in sys.argv
    print('factors: X x%.10g   Y x%.10g   UV x%.10g\n' % (fx, fy, uv))

    edits = {}
    for entry in TARGETS:
        model, kind, arr, stride, keys, per_line = entry[:6]
        material = entry[6] if len(entry) > 6 else None
        path = os.path.join(PKG, model, 'meshes', '%s_%s.DAE' % (model, kind))
        text = edits.get(path)
        if text is None:
            text = open(path).read()
        geom = text.split('<library_geometries>')[-1].split('</library_geometries>')[0]
        if '*' in geom:
            raise SystemExit('ERROR: %s has "*" inside geometry -- hand-edited expressions '
                             'are not valid COLLADA. git checkout it first.' % path)
        only = uv_indices_for(text, material) if material else None
        text, n = scale_array(text, model, kind, arr, stride, keys, per_line, f, only)
        edits[path] = text
        print('  %-48s %-8s %5d elems  cols %s%s'
              % (os.path.basename(path), arr, n, list(keys),
                 '  ONLY %s (%d idx)' % (material, len(only)) if only else ''))

    for model, kind, idxs in MATRIX_FIX:
        path = os.path.join(PKG, model, 'meshes', '%s_%s.DAE' % (model, kind))
        edits[path] = scale_matrix(edits.get(path) or open(path).read(), idxs, f)
        print('  %-48s %-8s        node translation, elems %s'
              % (os.path.basename(path), 'MATRIX', sorted(idxs)))

    if dry:
        print('\n(dry run -- nothing written)')
        return
    for path, text in edits.items():
        open(path, 'w').write(text)
    print('\nWrote %d files.' % len(edits))


if __name__ == '__main__':
    main()
