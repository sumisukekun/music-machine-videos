"""Era medley 1901 -> 1680: one marble falls down a tall wall through six eras.
Every lead note = the marble landing on a striker. The marble obeys one constant gravity: pure parabolas between
notes, and at each note a rigid bounce off a tilted striker that never adds speed (layout.py plans where every
striker has to sit so the parabola reaches it exactly on its note; the result is layout.json).
Each era has its own striker shape / palette and a glowing year number on the wall."""
import os, sys, json, math, time, colorsys
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json')); NOTES = json.load(open(f'{HERE}/notes.json'))
FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.5) * FPS); T = np.arange(NF) / FPS
rng = np.random.default_rng(11)

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
sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.3
try: sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Medium High Contrast'
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


def glowmat(name, gain=10.0, base=0.6, rough=0.15):
    """object colour = key colour; object alpha drives emission"""
    m = mat(name, (1, 1, 1), rough, **{'Coat Weight': 0.6}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    oi = nt.nodes.new('ShaderNodeObjectInfo')
    nt.links.new(oi.outputs['Color'], b.inputs['Base Color']); nt.links.new(oi.outputs['Color'], b.inputs['Emission Color'])
    mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY_ADD'; mul.inputs[1].default_value = gain; mul.inputs[2].default_value = base
    nt.links.new(oi.outputs['Alpha'], mul.inputs[0]); nt.links.new(mul.outputs[0], b.inputs['Emission Strength'])
    return m


def key_series(obj, path, idx, vals, eps=1e-5):
    n = len(vals)
    for f in range(n):
        v = vals[f]
        if 0 < f < n - 1 and abs(v - vals[f - 1]) < eps and abs(v - vals[f + 1]) < eps: continue
        getattr(obj, path)[idx] = float(v); obj.keyframe_insert(path, index=idx, frame=f)


def key_alpha(obj, arr, col):
    obj.color = (*col, 0)
    key_series(obj, 'color', 3, arr, 1e-3)


def rod(p0, p1, r, mt, verts=10):
    p0, p1 = Vector(p0), Vector(p1); d = p1 - p0
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=d.length, vertices=verts, location=(p0 + p1) / 2)
    o = bpy.context.object; q = Vector((0, 0, 1)).rotation_difference(d.normalized())
    o.rotation_mode = 'QUATERNION'; o.rotation_quaternion = q
    o.matrix_world = Matrix.Translation((p0 + p1) / 2) @ q.to_matrix().to_4x4()
    o.data.materials.append(mt); return o

SECS = E['secs']

# ---------------- layout: one striker per lead note (planned by layout.py) ----------------
LAY = json.load(open(f'{HERE}/layout.json'))
R = LAY['R']; G = LAY['G']; H0 = LAY['H0']             # marble radius, gravity, first free-fall height
def era_of(t, s):
    if s >= 6: return 6
    return s if t >= SECS[s][2] - 1e-6 else -1
hits = []                                  # merge simultaneous notes (final chord)
for t, p, v, s in LEAD:
    if hits and abs(t - hits[-1][0]) < 1e-3: continue
    hits.append([t, p, v, s, era_of(t, s)])
NH = len(hits)
assert NH == len(LAY['t']) and max(abs(h[0] - t) for h, t in zip(hits, LAY['t'])) < 1e-3, 'run layout.py after arrange.py'
X = np.array(LAY['x']); Z = np.array(LAY['z']); TH = np.array(LAY['tilt']); V0 = np.array(LAY['v0'])
ZTOP = Z.copy()                            # contact height (ball centre at a hit = (X, Z + R))
NRM = np.stack([np.sin(TH), np.cos(TH)], 1)              # striker surface normal (x, z)
CON = np.stack([X, Z + R], 1) - R * NRM                  # contact point on the striker surface
XC = float(np.mean(X))


def pose(o, i, half, base_y=0.0):
    """put striker o so its top surface touches the marble at hit i, tilted to the bounce normal"""
    c = CON[i] - half * NRM[i]
    o.location = (c[0], 0, c[1]); o.rotation_euler = (0, base_y + TH[i], 0)
    return c


# ---------------- materials / palettes ----------------
PAL = [  # (wall colour, striker colour, glow colour)
    ((0.10, 0.035, 0.11), (1.0, 0.72, 0.28), (1.0, 0.75, 0.3)),     # 1901 brass tubes
    ((0.03, 0.045, 0.12), (0.75, 0.9, 1.0), (0.55, 0.85, 1.0)),     # 1877 glass bells
    ((0.025, 0.07, 0.07), (0.6, 0.35, 1.0), (0.55, 0.3, 1.0)),      # 1875 crystal shards
    ((0.07, 0.06, 0.035), (0.55, 0.32, 0.14), (1.0, 0.55, 0.2)),    # 1829 wooden blocks
    ((0.02, 0.07, 0.09), (0.1, 0.6, 0.65), (0.2, 0.95, 0.9)),       # 1783 mosaic tiles
    ((0.11, 0.03, 0.04), (0.93, 0.88, 0.76), (1.0, 0.8, 0.5)),      # 1680 harpsichord keys
]
GLOW = glowmat('glow', gain=9.0, base=0.0, rough=0.25)
GLASS = mat('glass', (1, 1, 1), 0.05, **{'Transmission Weight': 0.85, 'IOR': 1.45, 'Coat Weight': 0.5})
oi_g = GLASS.node_tree.nodes.new('ShaderNodeObjectInfo'); bG = GLASS.node_tree.nodes['Principled BSDF']
GLASS.node_tree.links.new(oi_g.outputs['Color'], bG.inputs['Emission Color']); GLASS.node_tree.links.new(oi_g.outputs['Color'], bG.inputs['Base Color'])
mg = GLASS.node_tree.nodes.new('ShaderNodeMath'); mg.operation = 'MULTIPLY_ADD'; mg.inputs[1].default_value = 8; mg.inputs[2].default_value = 0.5
GLASS.node_tree.links.new(oi_g.outputs['Alpha'], mg.inputs[0]); GLASS.node_tree.links.new(mg.outputs[0], bG.inputs['Emission Strength'])
METAL = glowmat('metal', gain=6.0, base=0.15, rough=0.22)
METAL.node_tree.nodes['Principled BSDF'].inputs['Metallic'].default_value = 1.0
BRACK = mat('bracket', (0.25, 0.25, 0.28), 0.35, **{'Metallic': 0.9})
BLACKKEY = glowmat('blackkey', gain=5.0, base=0.0, rough=0.3)

WALLY = 0.42
STRK = []        # (object, hit time, glow colour)
def finish(o, col, glow, t, m):
    o.data.materials.append(m)
    for p in o.data.polygons: p.use_smooth = True
    o.color = (*col, 0); STRK.append((o, t, glow))
def bracket(x, z):
    rod((x, 0.12, z), (x, WALLY, z), 0.025, BRACK, 8)

for i, (t, p, v, s, era) in enumerate(hits):
    if era == -1:                                   # tunnel chime: small silver bar
        bpy.ops.mesh.primitive_cylinder_add(radius=0.045, depth=0.34, vertices=12)
        o = bpy.context.object; c = pose(o, i, 0.045, math.pi / 2); finish(o, (0.85, 0.87, 0.92), (0.9, 0.95, 1.0), t, METAL); bracket(*c); continue
    if era == 6: continue
    wall, col, glow = PAL[era]
    if era == 0:      # brass tube lying across
        bpy.ops.mesh.primitive_cylinder_add(radius=0.09, depth=0.4, vertices=20)
        o = bpy.context.object; c = pose(o, i, 0.09, math.pi / 2); finish(o, col, glow, t, METAL); bracket(*c)
        tg = np.array([math.cos(TH[i]), -math.sin(TH[i])])
        for dx in (-0.2, 0.2):
            bpy.ops.mesh.primitive_torus_add(major_radius=0.095, minor_radius=0.018, location=(c[0] + dx * tg[0], 0, c[1] + dx * tg[1]), rotation=(0, math.pi / 2 + TH[i], 0))
            bpy.context.object.data.materials.append(BRACK)
    elif era == 1:    # glass bell, mouth down
        bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=0.2, radius2=0.07, depth=0.26)
        o = bpy.context.object; c = pose(o, i, 0.13); finish(o, col, glow, t, GLASS); bracket(*c)
    elif era == 2:    # crystal shard (flat-topped hexagonal prism, tilted facets)
        bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=0.15, depth=0.3)
        o = bpy.context.object
        bm = bmesh.new(); bm.from_mesh(o.data)
        for vv in bm.verts:
            if vv.co.z < 0: vv.co.x *= 0.35; vv.co.y *= 0.35; vv.co.z -= 0.12
        bm.to_mesh(o.data); c = pose(o, i, 0.15); finish(o, col, glow, t, GLASS); bracket(*c)
    elif era == 3:    # wooden block
        bpy.ops.mesh.primitive_cube_add(size=1); o = bpy.context.object; o.scale = (0.4, 0.26, 0.16)
        bpy.ops.object.transform_apply(scale=True)
        bv = o.modifiers.new('b', 'BEVEL'); bv.width = 0.02; bv.segments = 2
        c = pose(o, i, 0.08); finish(o, col if i % 2 else (0.62, 0.38, 0.17), glow, t, GLOW); bracket(*c)
    elif era == 4:    # mosaic tile shelf
        bpy.ops.mesh.primitive_cube_add(size=1); o = bpy.context.object; o.scale = (0.4, 0.3, 0.08)
        bpy.ops.object.transform_apply(scale=True)
        cc = [(0.1, 0.6, 0.65), (0.85, 0.65, 0.2), (0.1, 0.25, 0.7), (0.9, 0.9, 0.85)][i % 4]
        c = pose(o, i, 0.04); finish(o, cc, glow, t, GLOW); bracket(*c)
    elif era == 5:    # harpsichord key (sharps black)
        sharp = (p % 12) in (1, 3, 6, 8, 10)
        bpy.ops.mesh.primitive_cube_add(size=1); o = bpy.context.object; o.scale = (0.4, 0.3, 0.1)
        bpy.ops.object.transform_apply(scale=True)
        bv = o.modifiers.new('b', 'BEVEL'); bv.width = 0.015; bv.segments = 2
        c = pose(o, i, 0.05); finish(o, (0.05, 0.04, 0.04) if sharp else col, glow, t, BLACKKEY if sharp else GLOW); bracket(*c)

# final golden bell
zb = ZTOP[-1]; xb = X[-1]
bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=1.0, radius2=0.35, depth=1.1, location=(xb, 0, zb - 0.55))
BIG = bpy.context.object; finish(BIG, (1.0, 0.75, 0.3), (1.0, 0.8, 0.35), hits[-1][0], METAL)
bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=0.07, location=(xb, 0, zb - 1.1)); bpy.context.object.data.materials.append(METAL)
bpy.context.object.color = (1.0, 0.75, 0.3, 0)
rod((xb, 0.25, zb - 0.3), (xb, WALLY, zb - 0.3), 0.05, BRACK, 12)      # bell hangs from the wall behind it

# ---------------- wall, one tinted panel per era ----------------
sec_top = []
for s in range(6):
    idx = [i for i in range(NH) if hits[i][3] == s]
    sec_top.append(ZTOP[idx[0]] + 1.2)
sec_top.append(ZTOP[-1] - 4.0)
sec_top[0] = 9.0
WALLW = float(X.max() - X.min()) + 30.0
for s in range(6):
    z0, z1 = sec_top[s], sec_top[s + 1]
    wall = PAL[s][0]
    if s == 5: z1 -= 14
    bpy.ops.mesh.primitive_plane_add(size=1, location=(XC, WALLY + 0.03, (z0 + z1) / 2), rotation=(math.pi / 2, 0, 0))
    o = bpy.context.object; o.scale = (WALLW, z0 - z1, 1)
    m = mat(f'wall{s}', wall, 0.75)
    nt = m.node_tree; b = nt.nodes['Principled BSDF']; nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 3
    mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'; mix.inputs['Factor'].default_value = 0.25
    mix.inputs['A'].default_value = (*wall, 1); mix.inputs['B'].default_value = tuple(min(1, c * 1.8) for c in wall) + (1,)
    nt.links.new(nz.outputs['Fac'], mix.inputs['Factor']); nt.links.new(mix.outputs['Result'], b.inputs['Base Color'])
    o.data.materials.append(m)
    # thin gold divider between eras
    if s > 0:
        bpy.ops.mesh.primitive_cube_add(size=1, location=(XC, WALLY, z0)); d = bpy.context.object; d.scale = (WALLW, 0.05, 0.05)
        d.data.materials.append(emis(f'div{s}', PAL[s][2], 2.0))

# year numbers on the wall (pop in as the era starts)
YEARS = []
for s, (year, name, ts, te) in enumerate(SECS):
    idx = [i for i in range(NH) if hits[i][4] == s]
    zy = ZTOP[idx[0]] + 0.55
    cu = bpy.data.curves.new(f'y{s}', 'FONT'); cu.body = ('c.' if year == 1680 else '') + str(year); cu.size = 1.25; cu.extrude = 0.04
    cu.align_x = 'CENTER'; cu.align_y = 'BOTTOM'
    o = link(bpy.data.objects.new(f'year{s}', cu)); o.location = (float(np.mean(X[idx[:6]])), WALLY - 0.06, zy); o.rotation_euler = (math.pi / 2, 0, 0)
    o.data.materials.append(emis(f'ym{s}', PAL[s][2], 4.0)); YEARS.append((o, ts))
    a = np.clip((T - (ts - 0.45)) / 0.35, 0, 1); a = 1 - (1 - a) ** 3
    sc_ = np.where(a > 0, 0.001 + a * (1 + 0.15 * np.sin(a * math.pi)), 0.001)
    for k in range(3): key_series(o, 'scale', k, sc_)

# candles for the 1680 era (decor on the wall edges)
idx5 = [i for i in range(NH) if hits[i][4] == 5]
FLAME = emis('flame', (1.0, 0.6, 0.2), 12.0); WAX = mat('wax', (0.95, 0.92, 0.85), 0.5)
for k in range(8):
    zc = ZTOP[idx5[0]] - k * (ZTOP[idx5[0]] - ZTOP[idx5[-1]]) / 7
    j = idx5[int(np.argmin(np.abs(ZTOP[idx5] - zc)))]; xc = X[j] + (-1.6 if k % 2 else 1.6)
    rod((xc, 0.2, zc - 0.3), (xc, 0.2, zc), 0.06, WAX, 12)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=(xc, 0.2, zc + 0.07)); f_ = bpy.context.object; f_.scale = (1, 1, 1.8); f_.data.materials.append(FLAME)
    rod((xc, 0.2, zc - 0.3), (xc, WALLY, zc - 0.3), 0.02, BRACK, 6)

# ---------------- marble path: gravity parabolas between hits ----------------
BP = np.zeros((NF, 3)); SPIN = np.zeros(NF)
ht = np.array([h[0] for h in hits])
cz = ZTOP + R                               # ball centre at contact
TD = math.sqrt(2 * H0 / G)                  # first drop: released from rest H0 above striker 0
vin_end = abs(LAY['vin'][-1][1])            # landing speed on the big bell
SETTLE = []                                 # small decaying hops on the bell crown, then rest
tb, vb = ht[-1], 0.35 * vin_end
while vb > 0.25 and len(SETTLE) < 6:
    SETTLE.append((tb, vb)); tb += 2 * vb / G; vb *= 0.45
def ball_at(t):
    if t <= ht[0]:
        s_ = max(0.0, t - (ht[0] - TD)); return (X[0], cz[0] + H0 - 0.5 * G * s_ * s_)
    if t >= ht[-1]:
        for t0, v in SETTLE:
            s_ = t - t0
            if s_ <= 2 * v / G: return (X[-1], cz[-1] + v * s_ - 0.5 * G * s_ * s_)
        return (X[-1], cz[-1])
    i = int(np.searchsorted(ht, t, 'right') - 1); s_ = t - ht[i]
    return (X[i] + V0[i][0] * s_, cz[i] + V0[i][1] * s_ - 0.5 * G * s_ * s_)
for f in range(NF):
    bx, bz_ = ball_at(T[f]); BP[f] = (bx, 0, bz_)
for f in range(1, NF): SPIN[f] = SPIN[f - 1] + (BP[f][0] - BP[f - 1][0]) / R        # rolls with its sideways travel
bpy.ops.mesh.primitive_uv_sphere_add(radius=R, segments=32, ring_count=16)
BALL = bpy.context.object
for p in BALL.data.polygons: p.use_smooth = True
BALL.data.materials.append(mat('ball', (0.9, 0.08, 0.1), 0.12, **{'Coat Weight': 1.0, 'Emission Color': (1, 0.15, 0.1, 1), 'Emission Strength': 0.6}))
for k in range(3): key_series(BALL, 'location', k, BP[:, k])
BALL.rotation_mode = 'XYZ'; key_series(BALL, 'rotation_euler', 1, SPIN)
bl = bpy.data.lights.new('balll', 'POINT'); bl.energy = 6; bl.color = (1, 0.3, 0.25); bl.shadow_soft_size = 0.1
blo = link(bpy.data.objects.new('balll', bl)); blo.parent = BALL

# ---------------- striker hit animation: glow + wobble ----------------
for o, t, glow in STRK:
    fh = int(round(t * FPS)); o.color = (*o.color[:3], 0)
    o.color = (*glow, 0); o.keyframe_insert('color', index=3, frame=max(0, fh - 1))
    for df, a in ((0, 1.0), (5, 0.45), (18, 0.12)):
        o.color = (*glow, a); o.keyframe_insert('color', index=3, frame=fh + df)
    o.rotation_mode = 'XYZ'; base = o.rotation_euler[0]
    for df, a in ((-1, 0), (2, 0.18), (5, -0.09), (9, 0.04), (14, 0)):
        o.rotation_euler[0] = base + (a if o is not BIG else a * 0.3); o.keyframe_insert('rotation_euler', index=0, frame=max(0, fh + df))

# ---------------- world & lights ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs['Color'].default_value = (0.012, 0.008, 0.02, 1)
w.node_tree.nodes['Background'].inputs['Strength'].default_value = 1.0

# ---------------- camera: wide, follows the marble down the wall ----------------
def smooth(Xa, k):
    Y = Xa.copy()
    for f in range(1, len(Xa)): Y[f] = Y[f - 1] + (Xa[f] - Y[f - 1]) * (1 - math.exp(-1 / FPS / k))
    for f in range(len(Xa) - 2, -1, -1): Y[f] = Y[f + 1] + (Y[f] - Y[f + 1]) * (1 - (1 - math.exp(-1 / FPS / k)))   # keep forward lag
    return Y
bz = smooth(BP[:, 2].copy(), 0.35); bxs = smooth(BP[:, 0].copy(), 0.6)
fin = np.clip((T - (FINAL - 1.5)) / 2.0, 0, 1); fin = fin * fin * (3 - 2 * fin)
camd = bpy.data.cameras.new('cam'); camd.lens = 30; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
tr = cam.constraints.new('TRACK_TO'); tr.target = tgt; tr.track_axis = 'TRACK_NEGATIVE_Z'; tr.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
for f in range(NF):
    zc = bz[f] - 1.6
    sway = 1.2 * math.sin(T[f] * 0.13)
    d = 15.0 + 3.0 * fin[f]
    CP[f] = (bxs[f] + sway, -d, zc + 1.8)
    TP[f] = (bxs[f] + 0.15 * sway, 0, zc)
for i in range(3): key_series(cam, 'location', i, CP[:, i]); key_series(tgt, 'location', i, TP[:, i])
# key light rides with the camera, rim light from above
for nm, off, e, col, sz in (('key', (3.5, -7, 3.0), 900, (1.0, 0.92, 0.85), 6), ('fill', (-5, -6, -2), 300, (0.6, 0.55, 1.0), 6), ('top', (0, -2, 7), 400, (1, 0.8, 0.9), 5)):
    l_ = bpy.data.lights.new(nm, 'AREA'); l_.energy = e; l_.color = col; l_.size = sz
    o = link(bpy.data.objects.new(nm, l_))
    c = o.constraints.new('TRACK_TO'); c.target = tgt; c.track_axis = 'TRACK_NEGATIVE_Z'; c.up_axis = 'UP_Y'
    for k in range(3): key_series(o, 'location', k, TP[:, k] + off[k])

# ---------------- sanity: marble never passes through a striker, no sudden speed changes ----------------
bad = 0; HW = LAY['HW']
cx_, cz_ = CON[:, 0], CON[:, 1]
for t in np.arange(ht[0] - TD, ht[-1], 1 / 240):
    b = ball_at(t); rx, rz = b[0] - cx_, b[1] - cz_
    u = rx * NRM[:, 1] - rz * NRM[:, 0]; w = rx * NRM[:, 0] + rz * NRM[:, 1]
    du = np.maximum(0, np.abs(u) - HW); dw = np.where(w > 0, w, np.maximum(0, -w - 0.3))
    hit = (np.hypot(du, dw) < R - 0.015) & (np.abs(ht - t) > 0.03); hit[-1] = False
    if hit.any():
        bad += 1
        if bad < 25: print('COL t', round(t, 3), 'strikers', np.where(hit)[0][:5])
SP_ = np.linalg.norm(np.diff(BP, axis=0), axis=1) * FPS
fh_ = set(int(round(t * FPS)) + k for t in list(ht) + [s0 for s0, _ in SETTLE] for k in (-1, 0, 1))
rat = [max(SP_[f], SP_[f - 1]) / max(1e-3, min(SP_[f], SP_[f - 1])) for f in range(1, len(SP_))
       if f not in fh_ and f + 1 not in fh_ and min(SP_[f], SP_[f - 1]) > 0.5]
print('speed u/s max', round(SP_.max(), 2), 'in-flight max frame-to-frame speed ratio', round(max(rat), 3))
print('objects', len(bpy.data.objects), 'NF', NF, 'strikers', len(STRK), 'drop', round(ZTOP[0] - ZTOP[-1], 1), 'collisions', bad)
if MODE == 'check': sys.exit(0)
if MODE == 'test':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS)); sc.frame_set(f)
        sc.render.filepath = f'{HERE}/test_{f:04d}.png'
        t1 = time.time(); bpy.ops.render.render(write_still=True); print('frame', f, 'sec', round(time.time() - t1, 1))
elif MODE == 'range':
    sc.frame_start, sc.frame_end = int(args[1]), int(args[2])
    sc.render.image_settings.file_format = 'PNG'; sc.render.filepath = f'{HERE}/frames/f_'
    bpy.ops.render.render(animation=True)
