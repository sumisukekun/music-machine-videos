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
sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.5      # centred: the blur always reaches a hit <= 1/120 s away
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


# ---------------- fun layer: FX material, ghost trail, hit ripples + sparks, era gates, gears, bass lamps ----------------
def fxmat(name, gain=10.0):
    """transparent where object alpha = 0, glowing in the object colour where alpha = 1"""
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    oi = nt.nodes.new('ShaderNodeObjectInfo'); em = nt.nodes.new('ShaderNodeEmission'); tr_ = nt.nodes.new('ShaderNodeBsdfTransparent')
    mx = nt.nodes.new('ShaderNodeMixShader'); mu = nt.nodes.new('ShaderNodeMath'); mu.operation = 'MULTIPLY'; mu.inputs[1].default_value = gain
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(oi.outputs['Color'], em.inputs['Color']); nt.links.new(oi.outputs['Alpha'], mu.inputs[0]); nt.links.new(mu.outputs[0], em.inputs['Strength'])
    nt.links.new(oi.outputs['Alpha'], mx.inputs['Fac']); nt.links.new(tr_.outputs[0], mx.inputs[1]); nt.links.new(em.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs[0]); return m
FXM = fxmat('fx', 10.0); TRAILM = fxmat('trail', 3.0)

def key_all(o, loc=None, scl=None, alpha=None, col=None):
    if col is not None: o.color = (*col, 0)
    if loc is not None:
        for k in range(3): key_series(o, 'location', k, loc[:, k])
    if scl is not None:
        for k in range(3): key_series(o, 'scale', k, scl)
    if alpha is not None: key_series(o, 'color', 3, alpha, 1e-3)

# ghost trail: copies of the marble a few frames behind it (same path, so it shows the parabola)
for j in range(1, 6):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=R * (1 - 0.13 * j), segments=16, ring_count=8); g_ = bpy.context.object
    g_.data.materials.append(TRAILM)
    loc = np.array([(ball_at(t - j * 0.022)[0], 0.0, ball_at(t - j * 0.022)[1]) for t in T])
    key_all(g_, loc=loc, alpha=np.full(NF, 0.55 * (1 - j / 6.5)), col=(1.0, 0.25, 0.15))

# hit FX pools: an expanding ring around each contact point + three sparks flying out on gravity parabolas
ERA_GLOW = lambda e: (0.9, 0.95, 1.0) if e < 0 else (PAL[e][2] if e < 6 else (1.0, 0.8, 0.35))
NRING, NSPK = 10, 36
ring_loc = np.zeros((NRING, NF, 3)); ring_scl = np.zeros((NRING, NF)); ring_a = np.zeros((NRING, NF)); ring_col = [None] * NRING
spk_loc = np.zeros((NSPK, NF, 3)); spk_scl = np.zeros((NSPK, NF)); spk_a = np.zeros((NSPK, NF))
rng2 = np.random.default_rng(7)
for k, (t0, p, v, s, era) in enumerate(hits):
    c = CON[k]; n_ = NRM[k]; f0 = int(round(t0 * FPS)); r = k % NRING
    big = 3.0 if k == NH - 1 else 1.0
    for df in range(0, 16):
        f = f0 + df
        if f >= NF: break
        u = df / 15
        ring_loc[r, f] = (c[0], -0.02, c[1]); ring_scl[r, f] = big * (0.35 + 1.6 * (1 - (1 - u) ** 2)); ring_a[r, f] = (1 - u) ** 1.5
    ring_col[r] = ERA_GLOW(era)
    for q in range(3):
        sid = (3 * k + q) % NSPK; ang = math.atan2(n_[0], n_[1]) + rng2.uniform(-1.0, 1.0)
        sp = rng2.uniform(1.4, 2.6) * (1.6 if k == NH - 1 else 1.0)
        vx, vz, vy = sp * math.sin(ang), sp * math.cos(ang), -rng2.uniform(0.2, 0.8)
        for df in range(0, 14):
            f = f0 + df
            if f >= NF: break
            tt = df / FPS
            spk_loc[sid, f] = (c[0] + vx * tt, vy * tt, c[1] + vz * tt - 0.5 * G * tt * tt)
            spk_scl[sid, f] = 1 - df / 14; spk_a[sid, f] = 1 - df / 14
for r in range(NRING):
    bpy.ops.mesh.primitive_torus_add(major_radius=0.3, minor_radius=0.018, major_segments=40, minor_segments=6, rotation=(math.pi / 2, 0, 0))
    o = bpy.context.object; o.data.materials.append(FXM)
    for f in range(1, NF):                                      # stay put while invisible
        if ring_a[r, f] == 0: ring_loc[r, f] = ring_loc[r, f - 1]
    key_all(o, loc=ring_loc[r], scl=np.maximum(ring_scl[r], 1e-3), alpha=ring_a[r], col=ring_col[r] or (1, 1, 1))
SPKCOL = np.zeros((NSPK, 3)) + 1
for k in range(NH):
    for q in range(3): SPKCOL[(3 * k + q) % NSPK] = ERA_GLOW(hits[k][4])
for sid in range(NSPK):
    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.028, subdivisions=1); o = bpy.context.object; o.data.materials.append(FXM)
    for f in range(1, NF):
        if spk_a[sid, f] == 0: spk_loc[sid, f] = spk_loc[sid, f - 1]
    key_all(o, loc=spk_loc[sid], scl=np.maximum(spk_scl[sid], 1e-3), alpha=spk_a[sid], col=tuple(SPKCOL[sid]))

# era gates: a glowing ring the marble flies straight through just before the era's first note, year label above
def clear_of_strikers(p, d):
    return np.min(np.hypot(CON[:-1, 0] - p[0], CON[:-1, 1] - p[1])) > d
GATEM = glowmat('gate', gain=8.0, base=0.3, rough=0.25); GATEM.node_tree.nodes['Principled BSDF'].inputs['Metallic'].default_value = 1.0
for s_, (year, name, ts, te) in enumerate(SECS):
    tg = None
    for need in (1.0, 0.85, 0.7):
        for tt in np.arange(ts - 0.06, max(0.2, ts - 1.2), -1 / 120):
            if clear_of_strikers(ball_at(tt), need): tg = tt; break
        if tg is not None: break
    if tg is None: tg = ts - 0.3
    p0, p1 = np.array(ball_at(tg - 0.01)), np.array(ball_at(tg + 0.01)); vv = (p1 - p0) / 0.02
    bpy.ops.mesh.primitive_torus_add(major_radius=0.42, minor_radius=0.045, major_segments=48, minor_segments=10,
                                     location=(ball_at(tg)[0], 0, ball_at(tg)[1]), rotation=(0, math.atan2(vv[0], vv[1]), 0))
    o = bpy.context.object; o.data.materials.append(GATEM); o.color = (*PAL[s_][2], 0)
    fg = int(round(tg * FPS))
    for df, a_ in ((-2, 0.0), (0, 1.0), (8, 0.35), (40, 0.1)):
        o.color = (*PAL[s_][2], a_); o.keyframe_insert('color', index=3, frame=max(0, fg + df))
    gx, gz = ball_at(tg)
    cu = bpy.data.curves.new(f'y{s_}', 'FONT'); cu.body = ('c.' if year == 1680 else '') + str(year); cu.size = 0.95; cu.extrude = 0.04
    cu.align_x = 'CENTER'; cu.align_y = 'BOTTOM'
    yo = link(bpy.data.objects.new(f'year{s_}', cu)); yo.location = (gx, WALLY - 0.06, gz + 0.6); yo.rotation_euler = (math.pi / 2, 0, 0)
    yo.data.materials.append(emis(f'ym{s_}', PAL[s_][2], 4.0))
    a = np.clip((T - (tg - 0.5)) / 0.35, 0, 1); a = 1 - (1 - a) ** 3
    sc_ = np.where(a > 0, 0.001 + a * (1 + 0.15 * np.sin(a * math.pi)), 0.001)
    for k in range(3): key_series(yo, 'scale', k, sc_)

# wall gears (decor, flush on the wall behind the marble): meshing pairs turning at matched speeds
def gear(r, teeth, thick=0.03):
    bm = bmesh.new(); pts = []
    for k in range(teeth * 4):
        a_ = 2 * math.pi * k / (teeth * 4); rr = r * (1.0 if (k // 2) % 2 == 0 else 0.86)
        pts.append(bm.verts.new((rr * math.cos(a_), rr * math.sin(a_), 0)))
    f_ = bm.faces.new(pts); ex = bmesh.ops.extrude_face_region(bm, geom=[f_])
    bmesh.ops.translate(bm, vec=(0, 0, thick), verts=[e for e in ex['geom'] if isinstance(e, bmesh.types.BMVert)])
    me = bpy.data.meshes.new('gear'); bm.to_mesh(me); bm.free()
    return link(bpy.data.objects.new('gear', me))
GEARM = mat('gearm', (0.3, 0.2, 0.09), 0.45, **{'Metallic': 1.0})
placed = []
for i in range(0, NH - 1, 7):
    e = hits[i][4]; col = PAL[max(0, min(5, e if e >= 0 else hits[i][3]))]
    for side in (-1, 1):
        cx_, cz_ = X[i] + side * 1.3, Z[i] - 1.1
        r1 = 0.45 + 0.2 * ((i // 7) % 3) / 2; r2 = r1 * 0.6
        ok = all(np.hypot(CON[:-1, 0] - px, CON[:-1, 1] - pz).min() > rr + 0.35 for px, pz, rr in ((cx_, cz_, r1), (cx_ + side * (r1 + r2) * 0.93, cz_ + 0.15, r2)))
        ok = ok and all(math.hypot(cx_ - a_, cz_ - b_) > 2.4 for a_, b_ in placed)
        if not ok: continue
        placed.append((cx_, cz_))
        for (gx, gz, rr, tn, sgn) in ((cx_, cz_, r1, 12, 1), (cx_ + side * (r1 + r2) * 0.93, cz_ + 0.15, r2, 8, -1)):
            o = gear(rr, tn); o.data.materials.append(GEARM); o.location = (gx, WALLY - 0.035, gz); o.rotation_mode = 'XYZ'
            w_ = sgn * 0.5 * (r1 / rr)
            o.rotation_euler = (math.pi / 2, 0, 0); o.keyframe_insert('rotation_euler', index=1, frame=0)
            o.rotation_euler = (math.pi / 2, w_ * NF / FPS, 0); o.keyframe_insert('rotation_euler', index=1, frame=NF - 1)
            for fc in (o.animation_data.action.fcurves if hasattr(o.animation_data.action, 'fcurves') else []):
                for kp in fc.keyframe_points: kp.interpolation = 'LINEAR'

# bass lamps: little lamps on the wall along the path pulse on every bass note of the accompaniment
BASS = sorted(n[0] for n in NOTES if n[2] == 'bass')
pulse = np.zeros(NF)
for tb_ in BASS:
    f0 = int(round(tb_ * FPS)); m_ = np.arange(f0, min(NF, f0 + 12)); pulse[m_] = np.maximum(pulse[m_], np.exp(-(m_ - f0) / 4.0))
LAMPM = fxmat('lamp', 14.0); LBASE = mat('lampbase', (0.2, 0.18, 0.16), 0.4, **{'Metallic': 0.9})
for i in range(3, NH - 1, 5):
    e = hits[i][4]; col = ERA_GLOW(e if e >= 0 else hits[i][3])
    for side in (-1, 1):
        lx, lz = X[i] + side * 0.9, Z[i] + 0.75
        if np.hypot(CON[:-1, 0] - lx, CON[:-1, 1] - lz).min() < 0.5: continue
        bpy.ops.mesh.primitive_cylinder_add(radius=0.11, depth=0.04, location=(lx, WALLY - 0.02, lz), rotation=(math.pi / 2, 0, 0)); bpy.context.object.data.materials.append(LBASE)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.07, segments=16, ring_count=8, location=(lx, WALLY - 0.08, lz)); o = bpy.context.object
        o.data.materials.append(LAMPM); key_all(o, alpha=0.12 + 0.88 * pulse, col=col)

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
