"""Flight of the Bumblebee - a striped bee-marble flies down a giant honeycomb wall.
The honeycomb is a piano roll: each column is one semitone (left = low, right = high), each row is one beat.
Every melody note is a tap of the bee on one hex cell, which flashes and stays lit, so the whole flight
leaves a glowing trail; the accompaniment chords glow violet on their own cells. At the end the bee
lands in a sunflower that blooms, and the camera pulls back to reveal the complete flight path."""
import os, sys, json, math, time, colorsys
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json')); NOTES = json.load(open(f'{HERE}/notes.json'))
FPS = 30; FINAL = E['final']; T0 = E['t0']; BEAT = E['beat']
NF = int((FINAL + 4.5) * FPS); T = np.arange(NF) / FPS
rng = np.random.default_rng(7)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'
sc.cycles.samples = int(os.environ.get('SAMP', 6)); sc.cycles.use_adaptive_sampling = True; sc.cycles.adaptive_threshold = 0.06
sc.cycles.use_denoising = True
for k, v in (('denoiser', 'OPENIMAGEDENOISE'), ('use_fast_gi', True)):
    try: setattr(sc.cycles, k, v)
    except Exception: pass
sc.cycles.max_bounces = 4; sc.cycles.diffuse_bounces = 1; sc.cycles.glossy_bounces = 2; sc.cycles.transmission_bounces = 2
sc.render.resolution_x, sc.render.resolution_y = 1080, 1920
sc.render.resolution_percentage = int(os.environ.get('PCT', 50)); sc.render.fps = FPS
sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.25
try: sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Punchy'
except Exception: pass
sc.frame_start, sc.frame_end = 0, NF - 1


def link(o): sc.collection.objects.link(o); return o


def mat(name, col, rough=0.5, **kw):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']; b.inputs['Base Color'].default_value = (*col, 1); b.inputs['Roughness'].default_value = rough
    for k, v in kw.items(): b.inputs[k].default_value = v
    return m


def emis(name, col, s):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    e = nt.nodes.new('ShaderNodeEmission'); e.inputs['Color'].default_value = (*col, 1); e.inputs['Strength'].default_value = s
    o = nt.nodes.new('ShaderNodeOutputMaterial'); nt.links.new(e.outputs[0], o.inputs[0]); return m


WAXCOL = (0.32, 0.16, 0.02)


def glowmat(name, gain=10.0, base=0.0, rough=0.15):
    """object colour = cell colour; object alpha drives emission"""
    m = mat(name, (1, 1, 1), rough, **{'Coat Weight': 0.8, 'Coat Roughness': 0.05})
    nt = m.node_tree; b = nt.nodes['Principled BSDF']
    oi = nt.nodes.new('ShaderNodeObjectInfo')
    mx = nt.nodes.new('ShaderNodeMix'); mx.data_type = 'RGBA'; mx.clamp_factor = True
    rgba = [i for i in mx.inputs if i.type == 'RGBA']; A_, B_ = rgba[0], rgba[1]
    R_ = [o for o in mx.outputs if o.type == 'RGBA'][0]
    A_.default_value = (*WAXCOL, 1)
    fm = nt.nodes.new('ShaderNodeMath'); fm.operation = 'MULTIPLY'; fm.inputs[1].default_value = 2.5; fm.use_clamp = True
    nt.links.new(oi.outputs['Alpha'], fm.inputs[0]); nt.links.new(fm.outputs[0], mx.inputs['Factor'])
    nt.links.new(oi.outputs['Color'], B_); nt.links.new(R_, b.inputs['Base Color'])
    nt.links.new(oi.outputs['Color'], b.inputs['Emission Color'])
    mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY_ADD'; mul.inputs[1].default_value = gain; mul.inputs[2].default_value = base
    nt.links.new(oi.outputs['Alpha'], mul.inputs[0]); nt.links.new(mul.outputs[0], b.inputs['Emission Strength'])
    return m


def key_series(obj, path, idx, vals, eps=1e-5):
    n = len(vals)
    for f in range(n):
        v = vals[f]
        if 0 < f < n - 1 and abs(v - vals[f - 1]) < eps and abs(v - vals[f + 1]) < eps: continue
        getattr(obj, path)[idx] = float(v); obj.keyframe_insert(path, index=idx, frame=f)


def rod(p0, p1, r, mt, verts=10):
    p0, p1 = Vector(p0), Vector(p1); d = p1 - p0
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=d.length, vertices=verts, location=(p0 + p1) / 2)
    o = bpy.context.object; q = Vector((0, 0, 1)).rotation_difference(d.normalized())
    o.rotation_mode = 'QUATERNION'; o.rotation_quaternion = q
    o.matrix_world = Matrix.Translation((p0 + p1) / 2) @ q.to_matrix().to_4x4()
    o.data.materials.append(mt); return o


def gsmooth(x, sigma_s):
    """zero-phase gaussian smoothing over time (no lag)"""
    s = sigma_s * FPS; k = np.arange(-int(4 * s), int(4 * s) + 1); w = np.exp(-0.5 * (k / s) ** 2); w /= w.sum()
    xp = np.concatenate([np.repeat(x[:1], len(k), 0), x, np.repeat(x[-1:], len(k), 0)])
    return np.convolve(xp, w, mode='same')[len(k):-len(k)]


# ---------------- the honeycomb wall (board) ----------------
W = 0.16                     # column spacing (one semitone)
HR = W * math.sqrt(3) / 2    # row spacing (one beat)
R = W / math.sqrt(3) * 0.9   # hex circumradius with a small gap
TH = 0.05                    # cell thickness
PMID = 66                    # pitch at x = 0
ALPHA = math.radians(14)     # board leans back
EX = np.array([1.0, 0, 0])
DN = np.array([0, -math.sin(ALPHA), -math.cos(ALPHA)])     # down along the board (towards the camera a bit)
NN = np.array([0, -math.cos(ALPHA), math.sin(ALPHA)])      # board normal, facing the camera
ZTOP = 27.5; O = np.array([0, 0, ZTOP])
ROWS = E['nbeats'] + 4; PLO, PHI = 31, 97


def cell_c(p, r):
    """centre of the cell face (front surface) for pitch p, row r"""
    u = (p - PMID + 0.5 * (r % 2)) * W
    return O + EX * u + DN * (r * HR + 0.4) + NN * TH


MROT = Matrix(((EX[0], -DN[0], NN[0]), (EX[1], -DN[1], NN[1]), (EX[2], -DN[2], NN[2])))   # local (u, up, n) -> world
hexang = [math.pi / 2 + k * math.pi / 3 for k in range(6)]
HX = np.array([[R * math.cos(a), R * math.sin(a)] for a in hexang])


def hex_mesh(name, r_scale=1.0, bevel=True):
    bm = bmesh.new()
    vs = [bm.verts.new((x * r_scale, y * r_scale, -TH)) for x, y in HX]
    f = bm.faces.new(vs); ext = bmesh.ops.extrude_face_region(bm, geom=[f])
    for v in ext['geom']:
        if isinstance(v, bmesh.types.BMVert): v.co.z += TH
    if bevel: bmesh.ops.bevel(bm, geom=[e for e in bm.edges], offset=0.008, segments=2, affect='EDGES')
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    for p in me.polygons: p.use_smooth = False
    return me


# which cells are animated: lead (bee taps), chords, counter-melody
lead_cells = {}
for i, (t, p, v, b, sub) in enumerate(LEAD):
    lead_cells.setdefault((p, b), []).append((t, v))
FINAL_IDX = len(LEAD) - 1                    # the last note (low A at FINAL) is played by the sunflower, not a cell
lp, lb = LEAD[-1][1], LEAD[-1][3]
lead_cells[(lp, lb)] = [x for x in lead_cells[(lp, lb)] if abs(x[0] - FINAL) > 1e-3]
if not lead_cells[(lp, lb)]: del lead_cells[(lp, lb)]
chord_cells, horn_cells = {}, {}
for t, d, role, p, v in NOTES:
    b = int(math.floor((t - T0) / BEAT + 1e-6))
    if t >= FINAL - 1e-3: continue
    if role == 'chord': chord_cells.setdefault((p, b), []).append((t, d, v))
    if role == 'horn': horn_cells.setdefault((p, b), []).append((t, d, v))
anim_keys = set(lead_cells) | set(chord_cells) | set(horn_cells)

# static wax cells: one joined mesh
WAX = mat('wax', WAXCOL, 0.2, **{'Coat Weight': 0.6, 'Transmission Weight': 0.0, 'Subsurface Weight': 0.3,
                                              'Emission Color': (1.0, 0.5, 0.05, 1), 'Emission Strength': 0.02})
verts, faces = [], []
for r in range(ROWS):
    for p in range(PLO, PHI + 1):
        if (p, r) in anim_keys: continue
        c = cell_c(p, r)
        base = len(verts)
        for k in range(6):
            q = c + EX * HX[k, 0] + (-DN) * HX[k, 1]
            verts.append(q); verts.append(q - NN * TH)
        faces.append([base + 2 * k for k in range(6)])
        for k in range(6):
            a, b2 = base + 2 * k, base + 2 * ((k + 1) % 6)
            faces.append([a, a + 1, b2 + 1, b2])
me = bpy.data.meshes.new('wax'); me.from_pydata([tuple(v) for v in verts], [], faces); me.update()
WAXO = link(bpy.data.objects.new('wax', me)); WAXO.data.materials.append(WAX)
# backboard (dark wood) + hive frame
BOARD = mat('board', (0.08, 0.04, 0.015), 0.6)
FR = mat('frame', (0.32, 0.18, 0.08), 0.55)
bw = (PHI - PLO + 3) * W; bh = ROWS * HR + 1.2
bc = O + EX * ((PLO + PHI) / 2 - PMID) * W + DN * (bh / 2 - 0.2) - NN * TH * 1.2
bpy.ops.mesh.primitive_cube_add(size=1); bo = bpy.context.object
bo.matrix_world = Matrix.Translation(Vector(bc)) @ MROT.to_4x4() @ Matrix.Diagonal((bw, bh, 0.08, 1)); bo.data.materials.append(BOARD)
for sx in (-1, 1):
    c = bc + EX * sx * (bw / 2 + 0.12) + NN * 0.08
    bpy.ops.mesh.primitive_cube_add(size=1); o = bpy.context.object
    o.matrix_world = Matrix.Translation(Vector(c)) @ MROT.to_4x4() @ Matrix.Diagonal((0.24, bh + 0.5, 0.3, 1)); o.data.materials.append(FR)
for sy in (-1, 1):
    c = bc + DN * sy * (bh / 2 + 0.12) + NN * 0.08
    bpy.ops.mesh.primitive_cube_add(size=1); o = bpy.context.object
    o.matrix_world = Matrix.Translation(Vector(c)) @ MROT.to_4x4() @ Matrix.Diagonal((bw + 0.5, 0.24, 0.3, 1)); o.data.materials.append(FR)
# pitch ruler on the top frame: small brass studs on every C
BRASS = mat('brass', (0.9, 0.65, 0.25), 0.25, **{'Metallic': 1.0})
for p in range(PLO, PHI + 1):
    if p % 12 == 0:
        c = O + EX * (p - PMID) * W + DN * 0.1 + NN * 0.25
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.04, location=tuple(c)); bpy.context.object.data.materials.append(BRASS)

# animated cells
CELLM = glowmat('cell', 3.2, 0.0)
CMESH = hex_mesh('cellmesh', bevel=False)
pmin, pmax = 45, 93
def lead_col(p):
    x = (np.clip(p, pmin, pmax) - pmin) / (pmax - pmin)
    return colorsys.hsv_to_rgb(0.05 + 0.08 * x, 1.0 - 0.12 * x, 1.0)
CHORD_COL = (0.5, 0.2, 1.0); HORN_COL = (0.1, 0.45, 1.0)
CELLS = {}
for (p, r) in sorted(anim_keys):
    o = link(bpy.data.objects.new(f'c{p}_{r}', CMESH)); o.data.materials.append(CELLM) if not o.data.materials else None
    c = cell_c(p, r)
    o.matrix_world = Matrix.Translation(Vector(c)) @ MROT.to_4x4()
    CELLS[(p, r)] = o
bpy.context.view_layer.update()
for (p, r), o in CELLS.items():
    a = np.zeros(NF); dip = np.zeros(NF)
    if (p, r) in lead_cells:
        col = lead_col(p)
        for t, v in lead_cells[(p, r)]:
            g = 0.55 + 0.45 * v / 127
            a = np.maximum(a, np.where(T >= t - 0.5 / FPS, 0.12 + g * 1.4 * np.exp(-np.clip(T - t, 0, None) / 0.18), 0))
            dip += np.exp(-((T - t) / 0.05) ** 2) * 0.018
    else:
        col = HORN_COL if (p, r) in horn_cells else CHORD_COL
    for t, d, v in chord_cells.get((p, r), []):
        a = np.maximum(a, np.where(T >= t - 0.5 / FPS, 0.05 + 0.55 * np.exp(-np.clip(T - t, 0, None) / 0.2), 0))
    for t, d, v in horn_cells.get((p, r), []):
        on = (T >= t - 0.5 / FPS) & (T < t + d)
        a = np.maximum(a, np.where(on, 0.55, np.where(T >= t + d, 0.06 + 0.5 * np.exp(-np.clip(T - t - d, 0, None) / 0.3), 0)))
    # finale reveal: the whole trail brightens
    fin = np.clip((T - FINAL - 0.6) / 1.5, 0, 1)
    a = np.where(a > 0.01, a + fin * (0.5 if (p, r) in lead_cells else 0.2), a)
    o.color = (*col, 0); key_series(o, 'color', 3, a, 1e-3)
    if dip.max() > 0:
        base = o.location.copy()
        for i in range(3): key_series(o, 'location', i, [base[i] - NN[i] * dip[f] for f in range(NF)], 1e-5)

# ---------------- the bee (ball) ----------------
BR = 0.095
bpy.ops.mesh.primitive_uv_sphere_add(radius=BR, segments=32, ring_count=16)
BEE = bpy.context.object
for pp in BEE.data.polygons: pp.use_smooth = True
bm_ = bpy.data.materials.new('bee'); bm_.use_nodes = True; nt = bm_.node_tree; b = nt.nodes['Principled BSDF']
b.inputs['Roughness'].default_value = 0.12; b.inputs['Coat Weight'].default_value = 1.0
tc = nt.nodes.new('ShaderNodeTexCoord'); wv = nt.nodes.new('ShaderNodeTexWave'); wv.wave_type = 'BANDS'; wv.bands_direction = 'Z'
wv.inputs['Scale'].default_value = 2.2; wv.inputs['Distortion'].default_value = 0.0
ramp = nt.nodes.new('ShaderNodeValToRGB'); ramp.color_ramp.interpolation = 'CONSTANT'
ramp.color_ramp.elements[0].position = 0.0; ramp.color_ramp.elements[0].color = (0.01, 0.01, 0.01, 1)
ramp.color_ramp.elements[1].position = 0.5; ramp.color_ramp.elements[1].color = (1.0, 0.85, 0.05, 1)
nt.links.new(tc.outputs['Object'], wv.inputs['Vector']); nt.links.new(wv.outputs['Fac'], ramp.inputs['Fac'])
nt.links.new(ramp.outputs['Color'], b.inputs['Base Color'])
nt.links.new(ramp.outputs['Color'], b.inputs['Emission Color']); b.inputs['Emission Strength'].default_value = 0.45
BEE.data.materials.append(bm_)
# wings ride on an empty that follows the bee's position (not its spin)
RIG = link(bpy.data.objects.new('rig', None))
WING = mat('wing', (0.85, 0.95, 1.0), 0.05, **{'Transmission Weight': 0.9, 'Alpha': 0.55, 'IOR': 1.2})
WINGS = []
for sx in (-1, 1):
    piv = link(bpy.data.objects.new(f'wp{sx}', None)); piv.parent = RIG; piv.location = (sx * 0.03, 0, 0.0)
    bpy.ops.mesh.primitive_circle_add(vertices=24, radius=1, fill_type='NGON')
    w = bpy.context.object; w.scale = (0.13, 0.06, 1); w.location = (sx * 0.12, 0, 0); w.data.materials.append(WING); w.parent = piv
    piv.rotation_mode = 'XYZ'; WINGS.append((piv, sx))
lt = bpy.data.lights.new('beel', 'POINT'); lt.energy = 12; lt.color = (1.0, 0.75, 0.3); lt.shadow_soft_size = 0.05
lo = link(bpy.data.objects.new('beel', lt)); lo.parent = RIG; lo.location = (0, 0, 0)

# tap points
taps = [(t, cell_c(p, b) + NN * BR) for (t, p, v, b, sub) in LEAD]
# sunflower position (below the board, centre)
SUN_C = O + DN * (ROWS * HR + 1.9) + NN * 0.9 + EX * 0.0
taps[-1] = (FINAL, SUN_C + NN * BR)
start = (0.0, taps[0][1] + NN * 0.9 - DN * 0.6)
pts = [start] + taps
BP = np.zeros((NF, 3))
tt = np.array([q[0] for q in pts]); PP = np.array([q[1] for q in pts])
for f in range(NF):
    t = T[f]
    if t >= tt[-1]:
        BP[f] = PP[-1]; continue
    i = int(np.searchsorted(tt, t, side='right') - 1); i = max(0, i)
    t0_, t1_ = tt[i], tt[i + 1]; a = (t - t0_) / max(1e-6, t1_ - t0_)
    p0, p1 = PP[i], PP[i + 1]; dist = np.linalg.norm(p1 - p0)
    hgt = 0.03 + min(0.45, 0.22 * dist) if i > 0 else 0.0
    if i == 0: a = a * a * (3 - 2 * a)
    BP[f] = p0 + (p1 - p0) * a + NN * hgt * 4 * a * (1 - a)
# settle on the sunflower after landing: bob with the flower
for i in range(3): key_series(BEE, 'location', i, BP[:, i]); key_series(RIG, 'location', i, BP[:, i])
spin = np.cumsum(np.r_[0, np.linalg.norm(np.diff(BP, axis=0), axis=1)]) / BR
BEE.rotation_mode = 'XYZ'; key_series(BEE, 'rotation_euler', 0, spin * 0.5); key_series(BEE, 'rotation_euler', 2, spin * 0.3)
# orient the rig so the wings sit on the bee's back (facing the camera side, board up = +local y)
RIG.matrix_world = MROT.to_4x4(); RIG.rotation_mode = 'XYZ'; e0 = RIG.rotation_euler.copy()
for i in range(3): RIG.rotation_euler[i] = e0[i]
for i in range(3): key_series(RIG, 'location', i, BP[:, i])
for piv, sx in WINGS:
    piv.location = (sx * 0.02, 0.0, BR * 0.55)
    flap = 0.45 + 0.55 * np.sin(T * 2 * math.pi * 11.3 + (0 if sx > 0 else 0.2))
    flap = np.where(T > FINAL + 0.5, 0.25 + 0.05 * np.sin(T * 3), flap)
    key_series(piv, 'rotation_euler', 1, -sx * flap)

# ---------------- sunflower (plays the last note) ----------------
SUNM = mat('petal', (1.0, 0.72, 0.02), 0.45, **{'Emission Color': (1.0, 0.6, 0.0, 1), 'Emission Strength': 0.0, 'Subsurface Weight': 0.2})
DISC = mat('disc', (0.18, 0.08, 0.02), 0.7, **{'Emission Color': (1.0, 0.45, 0.05, 1), 'Emission Strength': 0.0})
STEM = mat('stem', (0.12, 0.4, 0.08), 0.6)
HEAD = link(bpy.data.objects.new('head', None)); HEAD.matrix_world = Matrix.Translation(Vector(SUN_C)) @ MROT.to_4x4()
bpy.ops.mesh.primitive_cylinder_add(radius=0.42, depth=0.12, vertices=48); dsk = bpy.context.object; dsk.data.materials.append(DISC)
dsk.parent = HEAD; dsk.location = (0, 0, -0.06)
PETALS = []
for k in range(26):
    a = k / 26 * 2 * math.pi; ring = k % 2
    piv = link(bpy.data.objects.new(f'pp{k}', None)); piv.parent = HEAD; piv.location = ((0.4 + 0.03 * ring) * math.cos(a), (0.4 + 0.03 * ring) * math.sin(a), -0.04 - 0.02 * ring)
    piv.rotation_mode = 'ZYX'; piv.rotation_euler = (0, 0, a)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, segments=16, ring_count=8); pt_ = bpy.context.object
    pt_.scale = (0.36, 0.1, 0.02); pt_.location = (0.34, 0, 0); pt_.data.materials.append(SUNM); pt_.parent = piv
    PETALS.append(piv)
rod(SUN_C - NN * 0.1, SUN_C - NN * 0.6 + np.array([0, 0, -6.0]), 0.07, STEM, 12)
for sx in (-1, 1):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, segments=16, ring_count=8, location=tuple(SUN_C - NN * 0.3 + np.array([sx * 0.45, 0, -1.4])))
    lf = bpy.context.object; lf.scale = (0.55, 0.06, 0.2); lf.rotation_euler = (0, sx * 0.4, 0); lf.data.materials.append(STEM)
bloom = np.clip((T - FINAL) / 0.35, 0, 1); bloom = 1 - (1 - bloom) ** 3
bloom = bloom * (1 + 0.12 * np.sin(np.clip(T - FINAL, 0, None) * 14) * np.exp(-np.clip(T - FINAL, 0, None) * 3))
pre = np.clip((T - (FINAL - 6)) / 6, 0, 1) * 0.25            # buds slowly loosen as the bee approaches
for piv in PETALS:
    key_series(piv, 'rotation_euler', 1, -(1.25 * (1 - np.maximum(bloom, pre))))   # folded forward over the disc
for m_, base, peak in ((SUNM, 0.0, 0.5), (DISC, 0.0, 1.5)):
    s_ = m_.node_tree.nodes['Principled BSDF'].inputs['Emission Strength']
    val = base + peak * np.clip(bloom, 0, 1) * (0.6 + 0.4 * np.exp(-np.clip(T - FINAL, 0, None) * 2))
    for f in range(NF):
        if 0 < f < NF - 1 and abs(val[f] - val[f - 1]) < 1e-4 and abs(val[f] - val[f + 1]) < 1e-4: continue
        s_.default_value = float(val[f]); s_.keyframe_insert('default_value', frame=f)
bob = np.where(T > FINAL, 0.08 * np.sin((T - FINAL) * 10) * np.exp(-(T - FINAL) * 3), 0)
hb = HEAD.location.copy()
for i in range(3): key_series(HEAD, 'location', i, hb[i] - NN[i] * bob)

# ---------------- garden: ground, grass tufts, flowers around the hive ----------------
GROUND = mat('ground', (0.12, 0.3, 0.07), 0.8)
zb = (O + DN * (ROWS * HR + 1.0))[2]
bpy.ops.mesh.primitive_plane_add(size=120, location=(0, 0, zb - 6.0)); g = bpy.context.object; g.data.materials.append(GROUND)
FLC = [(0.9, 0.2, 0.4), (0.95, 0.85, 0.2), (0.6, 0.3, 0.9), (1.0, 0.5, 0.1), (0.95, 0.95, 0.95)]
for i in range(70):
    x = rng.uniform(-16, 16); y = rng.uniform(-14, 4)
    if abs(x) < 5.5 and y > -3: continue
    h = rng.uniform(0.6, 1.6); zz = zb - 6.0
    rod((x, y, zz), (x, y, zz + h), 0.025, STEM, 6)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=rng.uniform(0.15, 0.3), segments=12, ring_count=6, location=(x, y, zz + h))
    fo = bpy.context.object; fo.scale[2] = 0.45; fo.data.materials.append(mat(f'fl{i}', FLC[i % 5], 0.5))

# ---------------- world & lights ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True; nt = w.node_tree
bg = nt.nodes['Background']; bg.inputs['Strength'].default_value = 0.7
grad = nt.nodes.new('ShaderNodeTexGradient'); tc = nt.nodes.new('ShaderNodeTexCoord'); mp = nt.nodes.new('ShaderNodeMapping')
mp.inputs['Rotation'].default_value = (0, -math.pi / 2, 0)
ramp = nt.nodes.new('ShaderNodeValToRGB'); ramp.color_ramp.elements[0].color = (0.25, 0.32, 0.18, 1); ramp.color_ramp.elements[1].color = (0.35, 0.55, 0.9, 1)
nt.links.new(tc.outputs['Generated'], mp.inputs['Vector']); nt.links.new(mp.outputs['Vector'], grad.inputs['Vector'])
nt.links.new(grad.outputs['Fac'], ramp.inputs['Fac']); nt.links.new(ramp.outputs['Color'], bg.inputs['Color'])
sun = bpy.data.lights.new('sun', 'SUN'); sun.energy = 1.5; sun.color = (1.0, 0.92, 0.8); sun.angle = 0.1
so = link(bpy.data.objects.new('sun', sun)); so.rotation_euler = (math.radians(50), math.radians(-25), math.radians(-20))

# ---------------- camera: square-on to the wall, wide, follows the flight with zero-lag smoothing ----------------
camd = bpy.data.cameras.new('cam'); camd.lens = 35; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 300
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
tr = cam.constraints.new('TRACK_TO'); tr.target = tgt; tr.track_axis = 'TRACK_NEGATIVE_Z'; tr.up_axis = 'UP_Y'
# position along the board (x, s) of the bee
bx = BP[:, 0]
bs = (BP - O) @ DN
fx = gsmooth(bx, 0.55)
fs = gsmooth(bs, 1.2)
DIST = 10.5
# keep the bee inside the frame: limit horizontal error
fx = bx + np.clip(fx - bx, -1.0, 1.0)
fx = gsmooth(fx, 0.25)
TPn = O[None, :] + EX[None, :] * fx[:, None] + DN[None, :] * np.maximum(fs - 0.9, 2.4)[:, None]
CPn = TPn + NN[None, :] * DIST + np.array([0, 0, 0.6])[None, :]
# intro: a little wider, looking at the top of the hive
intro = np.clip(1 - (T - 0.3) / 2.2, 0, 1); intro = intro * intro * (3 - 2 * intro)
# finale: pull back to show the whole flight path on the honeycomb
fin = np.clip((T - (FINAL + 0.5)) / 3.2, 0, 1); fin = fin * fin * (3 - 2 * fin)
centre = O + DN * (ROWS * HR / 2 + 1.0) + EX * 0.4
full_t = centre; full_c = centre + NN * 44.0 + np.array([0, 0, 1.5])
CPf = CPn + (NN[None, :] * 3.0) * intro[:, None]
TPf = TPn * (1 - fin[:, None]) + full_t[None, :] * fin[:, None]
CPf = CPf * (1 - fin[:, None]) + full_c[None, :] * fin[:, None]
for i in range(3): key_series(cam, 'location', i, CPf[:, i]); key_series(tgt, 'location', i, TPf[:, i])

print('objects', len(bpy.data.objects), 'NF', NF, 'cells', len(CELLS), 'final', FINAL)
if MODE == 'test':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS)); sc.frame_set(f)
        sc.render.filepath = f'{HERE}/test_{f:04d}.png'
        t1 = time.time(); bpy.ops.render.render(write_still=True); print('frame', f, 'sec', round(time.time() - t1, 1))
elif MODE == 'range':
    sc.frame_start, sc.frame_end = int(args[1]), int(args[2])
    sc.render.image_settings.file_format = 'PNG'; sc.render.filepath = f'{HERE}/frames/f_'
    bpy.ops.render.render(animation=True)
