"""Waltz of the Flowers - 「一晩で咲く魔法の庭」 a night greenhouse, side view.
Every melody onset = a ball landing on a flower part (bud, tulip, leaf, frozen bud, golden bell) that sounds it and opens.
layout.py plans all motion (constant gravity, rigid bounces that never add speed, mechanisms for the speed-ups).
The waltz accompaniment: all open flowers sway in 3/4 (big lean on the bass beat, nods on beats 2-3), the string lights
pulse on the after-beats.  jp_sub.otf = Noto Sans CJK JP Bold (OFL) subset to the glyphs used here.
python3 scene.py -- check | test t1 t2 ... | range f0 f1"""
import os, sys, json, math, time
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from phys import at as seg_at, KEY

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json')); NOTES = json.load(open(f'{HERE}/notes.json'))
LAY = json.load(open(f'{HERE}/layout.json'))
FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.0) * FPS); T = np.arange(NF) / FPS
G = LAY['G']; R = LAY['R']; EV = LAY['ev']
TN = np.array([l['t'] for l in LEAD]); BT = {int(k): v for k, v in E['bar_t'].items()}
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
sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.5
try: sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Punchy'
except Exception: pass
sc.frame_start, sc.frame_end = 0, NF - 1
FONT = bpy.data.fonts.load(f'{HERE}/jp_sub.otf')


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


def glowmat(name, gain=8.0, base=0.0, rough=0.3, metal=0.0, coat=0.3):
    """object colour = base colour; object alpha drives emission (hit flash)"""
    m = mat(name, (1, 1, 1), rough, **{'Coat Weight': coat, 'Metallic': metal}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    oi = nt.nodes.new('ShaderNodeObjectInfo')
    nt.links.new(oi.outputs['Color'], b.inputs['Base Color']); nt.links.new(oi.outputs['Color'], b.inputs['Emission Color'])
    mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY_ADD'; mul.inputs[1].default_value = gain; mul.inputs[2].default_value = base
    nt.links.new(oi.outputs['Alpha'], mul.inputs[0]); nt.links.new(mul.outputs[0], b.inputs['Emission Strength'])
    return m


def fxmat(name, gain=10.0):
    """transparent where object alpha = 0, glowing in the object colour where alpha = 1"""
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    oi = nt.nodes.new('ShaderNodeObjectInfo'); em = nt.nodes.new('ShaderNodeEmission'); tr_ = nt.nodes.new('ShaderNodeBsdfTransparent')
    mx = nt.nodes.new('ShaderNodeMixShader'); mu = nt.nodes.new('ShaderNodeMath'); mu.operation = 'MULTIPLY'; mu.inputs[1].default_value = gain
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(oi.outputs['Color'], em.inputs['Color']); nt.links.new(oi.outputs['Alpha'], mu.inputs[0]); nt.links.new(mu.outputs[0], em.inputs['Strength'])
    nt.links.new(oi.outputs['Alpha'], mx.inputs['Fac']); nt.links.new(tr_.outputs[0], mx.inputs[1]); nt.links.new(em.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs[0]); return m


FROST = []                     # value sockets driven by the global frost amount (cold spell)
def petalmat(name, rough=0.45, sss=0.0, glow=0.0):
    """petal colour = object colour, mixed toward frosty grey-blue by the global frost value"""
    m = mat(name, (1, 1, 1), rough, **{'Coat Weight': 0.2}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    oi = nt.nodes.new('ShaderNodeObjectInfo')
    mx = nt.nodes.new('ShaderNodeMix'); mx.data_type = 'RGBA'; mx.inputs['B'].default_value = (0.55, 0.62, 0.72, 1)
    nt.links.new(oi.outputs['Color'], mx.inputs['A']); nt.links.new(mx.outputs['Result'], b.inputs['Base Color'])
    if glow > 0:
        nt.links.new(oi.outputs['Color'], b.inputs['Emission Color']); b.inputs['Emission Strength'].default_value = glow
    FROST.append(mx.inputs['Factor']); return m


def objcol(name, rough=0.45, metal=0.0):
    m = mat(name, (1, 1, 1), rough, **{'Metallic': metal}); nt = m.node_tree
    oi = nt.nodes.new('ShaderNodeObjectInfo'); nt.links.new(oi.outputs['Color'], nt.nodes['Principled BSDF'].inputs['Base Color'])
    return m


def key_series(obj, path, idx, vals, eps=1e-5):
    n = len(vals)
    for f in range(n):
        v = vals[f]
        if 0 < f < n - 1 and abs(v - vals[f - 1]) < eps and abs(v - vals[f + 1]) < eps: continue
        getattr(obj, path)[idx] = float(v); obj.keyframe_insert(path, index=idx, frame=f)


def key_loc(o, P):
    for k in range(3): key_series(o, 'location', k, P[:, k])


def key_socket(sock, vals, eps=1e-4):
    n = len(vals)
    for f in range(n):
        v = vals[f]
        if 0 < f < n - 1 and abs(v - vals[f - 1]) < eps and abs(v - vals[f + 1]) < eps: continue
        sock.default_value = float(v); sock.keyframe_insert('default_value', frame=f)


_MC = {}
def prim(kind, m, *a):
    key = (kind, m.name if m else None) + tuple(a)
    if key in _MC: return _MC[key]
    bm = bmesh.new()
    if kind == 'cube': bmesh.ops.create_cube(bm, size=1.0)
    elif kind == 'cyl': bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=1.0, radius2=1.0, depth=1.0)
    elif kind == 'cone': bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=a[1], radius2=a[2], depth=a[3])
    elif kind == 'sph': bmesh.ops.create_uvsphere(bm, u_segments=a[0], v_segments=a[1], radius=1.0)
    elif kind == 'ico': bmesh.ops.create_icosphere(bm, subdivisions=a[0], radius=1.0)
    elif kind == 'plane': bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    elif kind == 'petal':                     # a curved petal: lies along +x from the origin, cupped
        L_, W_ = a[0], a[1]; rows = []
        for i in range(7):
            u = i / 6; w = W_ * math.sin(math.pi * min(0.98, 0.15 + 0.85 * u)) * (1 - 0.35 * u)
            z = 0.25 * W_ * (u ** 2) * a[2]
            rows.append([bm.verts.new((L_ * u, -w / 2, z + 0.15 * W_ * a[2])), bm.verts.new((L_ * u, 0, z)), bm.verts.new((L_ * u, w / 2, z + 0.15 * W_ * a[2]))])
        for r0, r1 in zip(rows, rows[1:]):
            bm.faces.new((r0[0], r1[0], r1[1], r0[1])); bm.faces.new((r0[1], r1[1], r1[2], r0[2]))
    me = bpy.data.meshes.new(kind); bm.to_mesh(me); bm.free()
    if kind in ('sph', 'cone', 'petal', 'ico'):
        for p in me.polygons: p.use_smooth = True
    if m: me.materials.append(m)
    if kind == 'petal':
        mod_me = me
    _MC[key] = me; return me


def obj(me, name='o'): return link(bpy.data.objects.new(name, me))


def box(c, size, m, rot=None, name='box'):
    o = obj(prim('cube', m), name); o.location = c; o.scale = size
    if rot is not None: o.rotation_euler = rot
    return o


def rod(p0, p1, r, m, verts=10):
    p0, p1 = Vector(p0), Vector(p1); d = p1 - p0
    o = obj(prim('cyl', m, verts), 'rod'); q = Vector((0, 0, 1)).rotation_difference(d.normalized() if d.length > 1e-9 else Vector((0, 0, 1)))
    o.matrix_world = Matrix.Translation((p0 + p1) / 2) @ q.to_matrix().to_4x4() @ Matrix.Diagonal((r, r, max(1e-4, d.length), 1))
    return o


def sphere(c, r, m, seg=(16, 8), name='sph'):
    o = obj(prim('sph', m, *seg), name); o.location = c; o.scale = (r, r, r); return o


def text(s, loc, size, m, rot=(math.pi / 2, 0, 0), align='CENTER', ext=0.02):
    cu = bpy.data.curves.new('t', 'FONT'); cu.body = s; cu.font = FONT; cu.size = size * 3.0; cu.extrude = ext   # this subset's glyphs come out at 1/3 em
    cu.align_x = align; cu.align_y = 'CENTER'
    o = link(bpy.data.objects.new('txt', cu)); o.location = loc; o.rotation_euler = rot; o.data.materials.append(m); return o


def pulse(times, decay=5.0, length=16, amp=1.0):
    a = np.zeros(NF)
    for t in times:
        f0 = int(round(t * FPS)); m_ = np.arange(max(0, f0), min(NF, f0 + length))
        a[m_] = np.maximum(a[m_], amp * np.exp(-(m_ - f0) / decay))
    return a


def ramp(t0, t1, ease=True):
    u = np.clip((T - t0) / max(1e-6, t1 - t0), 0, 1)
    return u * u * (3 - 2 * u) if ease else u


def smooth(Xa, k):
    Y = Xa.copy(); a = 1 - math.exp(-1 / FPS / k)
    for f in range(1, len(Xa)): Y[f] = Y[f - 1] + (Xa[f] - Y[f - 1]) * a
    for f in range(len(Xa) - 2, -1, -1): Y[f] = Y[f + 1] + (Y[f] - Y[f + 1]) * a
    return Y


# ======================================================================= global timeline values
T_GUST = EV['gust']['t']; T_SHUT = EV['t_shut']; SR0, SR1 = EV['sunrise']['t0'], EV['sunrise']['t1']
cold = ramp(T_GUST, T_GUST + 1.2) * (1 - ramp(T_SHUT - 0.3, T_SHUT + 1.5))          # 0..1 cold spell
dawn = ramp(SR0, SR1 + 1.5)                                                           # 0..1 sunrise
for s_ in FROST: pass
BASS = sorted(t for t, k in E['beats'] if k == 'bass'); CHORD = sorted(t for t, k in E['beats'] if k == 'chord')

# ======================================================================= balls
POS = {n: np.array([seg_at(b['seg'], t) for t in T]) for n, b in LAY['balls'].items()}
M_H = mat('hummingbird', (0.05, 0.75, 0.42), 0.12, **{'Coat Weight': 1.0, 'Metallic': 0.35})
M_HB = mat('hb_breast', (0.95, 0.25, 0.45), 0.25, **{'Coat Weight': 0.6})
M_DROP = mat('drop', (0.55, 0.85, 1.0), 0.02, **{'Transmission Weight': 0.85, 'IOR': 1.33, 'Coat Weight': 1.0})
M_DROPG = emis('dropglow', (0.35, 0.75, 1.0), 6.0)
M_SEED = mat('seed', (1.0, 0.72, 0.18), 0.22, **{'Metallic': 0.8, 'Coat Weight': 0.5})
M_BEAK = mat('beak', (0.08, 0.06, 0.05), 0.4)
M_WING = mat('wing', (0.35, 0.95, 0.7), 0.2, **{'Transmission Weight': 0.3, 'Emission Color': (0.2, 1.0, 0.6, 1), 'Emission Strength': 0.6})
BALL = {}
for n, b in LAY['balls'].items():
    bpy.ops.mesh.primitive_uv_sphere_add(radius=R, segments=32, ring_count=16); o = bpy.context.object; o.name = n
    for p in o.data.polygons: p.use_smooth = True
    o.data.materials.append({'bird': M_H, 'drop': M_DROP, 'seed': M_SEED}[b['kind']])
    key_loc(o, POS[n])
    vel = np.gradient(POS[n], axis=0) * FPS
    if b['kind'] == 'bird':
        # hummingbird: beak + breast patch + wings, facing the direction of travel; wings beat fast in flight
        rig = link(bpy.data.objects.new('birdrig', None)); rig.parent = o
        bk = obj(prim('cone', M_BEAK, 12, 0.03, 0.002, 0.2), 'beak'); bk.parent = rig; bk.location = (R + 0.08, 0, 0.03); bk.rotation_euler = (0, math.pi / 2, 0)
        br = sphere((0.06, -0.02, -0.04), R * 0.8, M_HB); br.parent = rig; br.scale = (R * 0.75, R * 0.8, R * 0.7)
        for side in (-1, 1):
            eye = sphere((R * 0.75, side * R * 0.45, R * 0.35), 0.022, M_BEAK); eye.parent = rig
            wp = link(bpy.data.objects.new('wingpiv', None)); wp.parent = rig; wp.location = (-0.02, side * R * 0.85, R * 0.4)
            w = obj(prim('petal', M_WING, 0.28, 0.12, 0.3), 'wing'); w.parent = wp; w.rotation_euler = (0, 0, side * 1.9)
            flap = np.zeros(NF)
            for f in range(NF):
                sp = np.linalg.norm(vel[f]); flap[f] = (0.6 * math.sin(T[f] * 2 * math.pi * 9) if sp > 0.05 else 0.15 * math.sin(T[f] * 2 * math.pi * 1.5)) * side
            wp.rotation_mode = 'XYZ'; key_series(wp, 'rotation_euler', 0, flap, 1e-3)
        head = np.zeros(NF); cur = 0.0
        for f in range(NF):
            if abs(vel[f, 0]) > 0.2: cur = 0.0 if vel[f, 0] > 0 else math.pi
            head[f] = cur
        rig.rotation_mode = 'XYZ'; key_series(rig, 'rotation_euler', 2, smooth(head, 0.08))
    else:
        sp = np.zeros(NF)
        for f in range(1, NF): sp[f] = sp[f - 1] + (POS[n][f, 0] - POS[n][f - 1, 0]) / R
        o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, sp, 1e-4)
        if b['kind'] == 'drop':
            core = sphere((0, 0, 0), R * 0.45, M_DROPG); core.parent = o
    BALL[n] = o

# ======================================================================= struck flower parts
def frame_from_n(n):
    n = np.array(n, float); n /= np.linalg.norm(n)
    xa = np.cross([0, 1, 0], n); xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return Matrix(((xa[0], ya[0], n[0]), (xa[1], ya[1], n[1]), (xa[2], ya[2], n[2])))


def pitch_col(p):
    h = ((p * 7) % 12) / 12.0                                 # circle of fifths -> neighbouring notes get distinct hues
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, 0.55, 1.0); return (r, g, b)


PITCH = {i: l['p'] for i, l in enumerate(LEAD)}
SECOF = {i: l['sec'] for i, l in enumerate(LEAD)}
M_KEY = glowmat('key', gain=9.0, base=0.45, rough=0.35)
M_ICE = glowmat('ice', gain=8.0, base=0.1, rough=0.08, coat=1.0)
M_BELL = glowmat('bell', gain=7.0, base=0.6, rough=0.25, metal=0.6)
M_LEAFK = glowmat('leafkey', gain=8.0, base=0.35, rough=0.4)
M_PET = petalmat('petal', 0.5, glow=0.35)
M_STEM = mat('stem', (0.16, 0.42, 0.12), 0.6)
M_LEAF = petalmat('leaf', 0.55)
M_SEPAL = mat('sepal', (0.2, 0.5, 0.15), 0.6)
M_OBJ = objcol('objcol', 0.5)
M_FX = fxmat('fx', 8.0)
FLOWER_COL = {'horn1': [(1.0, 0.45, 0.62), (1.0, 0.75, 0.82), (0.95, 0.3, 0.45)], 'horn2': [(0.7, 0.45, 1.0), (0.85, 0.7, 1.0), (0.55, 0.35, 0.95)],
              'cold': [(0.75, 0.88, 1.0), (0.9, 0.95, 1.0)], 'climb': [(1.0, 0.82, 0.3), (1.0, 0.6, 0.25)],
              'horn3': [(1.0, 0.55, 0.2), (1.0, 0.85, 0.35), (1.0, 0.4, 0.5)], 'coda': [(1.0, 0.95, 0.5), (1.0, 0.6, 0.7), (0.7, 0.85, 1.0)]}
FLOWERS = []         # (pivot empty, t_bloom, kind)
KEYS = []


def flower(c, t, kind, col, size=1.0, n_pet=6, petal_len=0.24, droop=-0.35):
    """a flower head at c (upright): petals that grow open at time t (time-lapse bloom), sway with the waltz"""
    piv = link(bpy.data.objects.new('flower', None)); piv.location = c
    for k in range(n_pet):
        a = 2 * math.pi * k / n_pet + (0.3 if n_pet % 2 else 0)
        pe = obj(prim('petal', M_PET, petal_len, petal_len * 0.62, 1.0), 'petal'); pe.parent = piv
        pe.rotation_euler = (0, -droop, a); pe.color = (*col, 1)
    FLOWERS.append((piv, t, kind, size)); return piv


for n, b in LAY['balls'].items():
    for h in b['hits']:
        k = h['kind']
        if k not in KEY: continue
        hw, hd, th = KEY[k]; nrm = np.array(h['n'], float); nrm /= np.linalg.norm(nrm)
        contact = np.array(h['p']) - R * nrm; ctr = contact - nrm * th
        i = h['i']; t = h['t']; sec = SECOF[i]; pc = pitch_col(PITCH[i])
        Mx = Matrix.Translation(Vector(ctr)) @ frame_from_n(nrm).to_4x4()
        if k in ('bud', 'petal', 'ice'):
            o = obj(prim('cyl', M_ICE if k == 'ice' else M_KEY, 20), 'disc'); o.matrix_world = Mx @ Matrix.Diagonal((hw, hd, 2 * th, 1))
            o.color = (*pc, 0)
            sep = obj(prim('sph', M_SEPAL, 12, 6), 'sepal'); sep.location = ctr - np.array([0, 0, th + 0.03]); sep.scale = (hw * 0.95, hd * 0.95, 0.07)
            cols = FLOWER_COL.get(sec, FLOWER_COL['horn1'])
            col = cols[i % len(cols)] if k != 'ice' else (0.85, 0.93, 1.0)
            flower(tuple(ctr - np.array([0, 0, th + 0.01])), t, k, col, n_pet=5 if k == 'ice' else 7,
                   petal_len=0.28 if k == 'petal' else 0.4)
        elif k == 'cup':
            o = obj(prim('cyl', M_KEY, 20), 'cupbase'); o.matrix_world = Mx @ Matrix.Diagonal((hw, hd, 2 * th, 1)); o.color = (*pc, 0)
        elif k == 'leaf':
            o = obj(prim('sph', M_LEAFK, 16, 8), 'leafkey'); o.matrix_world = Mx @ Matrix.Diagonal((hw * 1.25, hd * 1.1, th * 1.2, 1))
            o.color = (*[0.25 + 0.75 * c_ for c_ in pc], 0)
            lf = obj(prim('petal', M_LEAF, 0.42, 0.3, 0.5), 'leaf'); lf.location = tuple(ctr - np.array([0, -0.12, th + 0.012])); lf.rotation_euler = (0, 0.05, -math.pi / 2)
            lf.color = (0.18, 0.55, 0.2, 1)
        elif k == 'bell':
            o = obj(prim('cone', M_BELL, 24, 2.0, 1.0, 1.0), 'bell'); o.matrix_world = Mx @ Matrix.Translation((0, 0, th - 0.11)) @ Matrix.Diagonal((hw, hd, 0.22, 1))
            o.color = (1.0, 0.78, 0.3, 0)
        elif k == 'sunflower':
            continue
        KEYS.append(dict(o=o, c=ctr, n=nrm, t=t, kind=k, ball=n, i=i, contact=contact, hw=hw, th=th))

# hit flash + dip on every struck part
for K in KEYS:
    o = K['o']; fh = int(round(K['t'] * FPS)); base = o.location.copy(); nv = Vector(K['n']); col = tuple(o.color[:3])
    o.keyframe_insert('color', index=3, frame=max(0, fh - 1))
    for df, a in ((0, 1.0), (4, 0.55), (18, 0.18)):
        o.color = (*col, a); o.keyframe_insert('color', index=3, frame=fh + df)
    for df, d in ((-1, 0), (1, -0.025), (4, 0.008), (8, 0)):
        o.location = base + nv * d; o.keyframe_insert('location', frame=max(0, fh + df))

# hit sparkles: expanding ring + pollen burst (FX pool)
NRING = 16
ring_loc = np.zeros((NRING, NF, 3)); ring_scl = np.full((NRING, NF), 1e-3); ring_a = np.zeros((NRING, NF)); ring_c = [(1, 1, 1)] * NRING
for k, K in enumerate(sorted(KEYS, key=lambda b: b['t'])):
    r = k % NRING; f0 = int(round(K['t'] * FPS))
    for df in range(14):
        f = f0 + df
        if f >= NF: break
        u = df / 13; ring_loc[r, f] = K['contact'] + np.array([0, -0.2, 0.02]); ring_scl[r, f] = 0.3 + 1.2 * (1 - (1 - u) ** 2)
        ring_a[r, f] = (1 - u) ** 1.6
    ring_c[r] = pitch_col(PITCH[K['i']])
for r in range(NRING):
    bpy.ops.mesh.primitive_torus_add(major_radius=0.2, minor_radius=0.012, major_segments=32, minor_segments=5, rotation=(math.pi / 2, 0, 0))
    o = bpy.context.object; o.data.materials.append(M_FX); o.color = (*ring_c[r], 0)
    for f in range(1, NF):
        if ring_a[r, f] == 0: ring_loc[r, f] = ring_loc[r, f - 1]
    key_loc(o, ring_loc[r])
    for k in range(3): key_series(o, 'scale', k, ring_scl[r])
    key_series(o, 'color', 3, ring_a[r], 1e-3)
# pollen motes: little glowing specks thrown up from each opening flower (pool)
NPOL = 40
pl_loc = np.zeros((NPOL, NF, 3)); pl_a = np.zeros((NPOL, NF))
for k, K in enumerate(sorted([K for K in KEYS if K['kind'] in ('bud', 'petal', 'ice', 'bell', 'cup')], key=lambda b: b['t'])):
    for j in range(2):
        r = (2 * k + j) % NPOL; f0 = int(round(K['t'] * FPS)); v = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.5, 0.1), rng.uniform(0.8, 1.6)])
        for df in range(24):
            f = f0 + df
            if f >= NF: break
            s_ = df / FPS; pl_loc[r, f] = K['contact'] + v * s_ + np.array([0, 0, -0.9 * s_ * s_]); pl_a[r, f] = (1 - df / 24) ** 1.3
for r in range(NPOL):
    o = sphere((0, 0, 0), 0.025, M_FX, (8, 4), 'pollen'); o.color = (1.0, 0.9, 0.5, 0)
    for f in range(1, NF):
        if pl_a[r, f] == 0: pl_loc[r, f] = pl_loc[r, f - 1]
    key_loc(o, pl_loc[r]); key_series(o, 'color', 3, pl_a[r], 1e-3)

# ======================================================================= flower bloom / wilt / sway animation
# one shared waltz sway: lean on the bass beat, small nods on the after-beats
sway = np.zeros(NF)
for f in range(NF):
    t = T[f]; v = 0.0
    for tb in BASS[max(0, np.searchsorted(BASS, t) - 2):np.searchsorted(BASS, t) + 1]:
        if t >= tb: v += 0.09 * math.exp(-(t - tb) / 0.35) * math.sin(min(math.pi, (t - tb) / 0.3 * math.pi / 2 + 0.3))
    for tc in CHORD[max(0, np.searchsorted(CHORD, t) - 2):np.searchsorted(CHORD, t) + 1]:
        if t >= tc: v -= 0.035 * math.exp(-(t - tc) / 0.15)
    sway[f] = v
SWAY = link(bpy.data.objects.new('sway', None)); SWAY.rotation_mode = 'XYZ'; key_series(SWAY, 'rotation_euler', 1, sway, 1e-4)
for piv, t, kind, size in FLOWERS:
    s = np.zeros(NF)
    for f in range(NF):
        u = (T[f] - t) / 0.32
        if u <= 0: s[f] = 0.12
        else: s[f] = 0.12 + 0.88 * min(1.0, u) ** 0.7 + 0.12 * math.sin(min(u, 2.5) * math.pi * 1.2) * math.exp(-1.5 * u)
    wilt = 1.0 - 0.38 * cold * (T >= t)                          # droop + shrink in the cold, back (bigger) when warm again
    reb = 1.0 + 0.12 * ramp(T_SHUT, T_SHUT + 2.0) * (t < T_SHUT)
    s = s * wilt * reb * size
    for k in range(2): key_series(piv, 'scale', k, s, 1e-3)
    key_series(piv, 'scale', 2, s * (1 - 0.45 * cold * (T >= t)), 1e-3)
    cr = piv.constraints.new('COPY_ROTATION'); cr.target = SWAY; cr.mix_mode = 'ADD'; cr.use_x = False; cr.use_z = False
for sock in FROST: key_socket(sock, 0.75 * cold)

# ======================================================================= stems, soil beds
def stem(top, bottom_z, r=0.022, leaves=True, back=0.0):
    """stem from just under a flower down to the soil (kinked back behind the lane when back > 0)"""
    top = np.array(top, float)
    if back:
        mid = top + np.array([0, back, -0.12]); rod(top, mid, r, M_STEM, 8); top = mid
    rod(top, (top[0], top[1], bottom_z), r, M_STEM, 8)
    if leaves and top[2] - bottom_z > 0.5:
        zl = bottom_z + (top[2] - bottom_z) * rng.uniform(0.3, 0.6)
        lf = obj(prim('petal', M_LEAF, 0.3, 0.16, 0.7), 'stemleaf'); lf.location = (top[0], top[1], zl)
        lf.rotation_euler = (0, -0.5, rng.choice([0.3, math.pi - 0.3]) + rng.uniform(-0.3, 0.3)); lf.color = (0.15, 0.48, 0.16, 1)


hkeys = [K for K in KEYS if K['ball'] == 'H']
HX = np.array([K['c'][0] for K in hkeys]); HZ = np.array([K['c'][2] for K in hkeys])
cups = EV['cups']
def soil_z(x):
    m = np.abs(HX - x) < 1.4
    zs = list(HZ[m]) + [c['p'][2] - R for c in cups if abs(c['p'][0] - x) < 1.4]
    return (min(zs) if zs else float(np.interp(x, HX, HZ))) - 1.15
SX = np.arange(-4.0, HX.max() + 8.0, 0.5); SZ = np.array([soil_z(x) for x in SX])
SZ = np.minimum.accumulate(SZ[::-1])[::-1] * 0 + np.array([min(SZ[max(0, i - 2):i + 3]) for i in range(len(SZ))])
def ground(x): return float(np.interp(x, SX, SZ))
M_SOIL = mat('soil', (0.12, 0.07, 0.04), 0.95)
M_PLANK = mat('plank', (0.42, 0.26, 0.13), 0.7)
M_IRON = mat('iron', (0.12, 0.2, 0.16), 0.4, **{'Metallic': 0.7})
M_GLASS = mat('glass', (0.8, 0.9, 1.0), 0.05, **{'Transmission Weight': 1.0, 'IOR': 1.1, 'Alpha': 0.25})
M_STONE = mat('stone', (0.22, 0.12, 0.08), 0.85)
# terraced planter boxes along the H lane
for x0 in np.arange(SX[0], SX[-1], 1.0):
    z = ground(x0 + 0.5)
    box((x0 + 0.5, 0.0, z - 0.15), (1.02, 1.3, 0.3), M_SOIL, name='soil')
    box((x0 + 0.5, -0.66, z - 0.35), (1.02, 0.06, 0.62), M_PLANK, name='plank')
    box((x0 + 0.5, 0.0, z - 1.6), (1.02, 1.4, 2.3), M_STONE, name='bedwall')
for K in hkeys:
    if K['kind'] in ('bud', 'ice', 'petal'): stem(K['c'] - np.array([0, 0, 0.06]), ground(K['c'][0]))
# tulip cups (H rests in them): petals close around the bird while it rests, burst open before the spring flick
M_TULIP = petalmat('tulip', 0.4, glow=0.15)
TULIP_COL = [(1.0, 0.3, 0.4), (1.0, 0.55, 0.15), (0.9, 0.3, 0.9), (0.95, 0.95, 0.95), (1.0, 0.8, 0.2), (1.0, 0.4, 0.6)]
for ci, c in enumerate(cups):
    p = np.array(c['p']); base = p - np.array([0, 0, R + 0.05])
    hp = link(bpy.data.objects.new('tulip', None)); hp.location = tuple(base)
    pets = []
    for k in range(6):
        a = 2 * math.pi * k / 6
        pv = link(bpy.data.objects.new('tpiv', None)); pv.parent = hp; pv.location = (0.1 * math.cos(a), 0.1 * math.sin(a), 0.0); pv.rotation_euler = (0, 0, a)
        pe = obj(prim('petal', M_TULIP, 0.34, 0.24, 1.6), 'tpetal'); pe.parent = pv; pe.color = (*TULIP_COL[ci % 6], 1)
        pets.append(pv)
    # petal angle above horizontal: open (0.15) on arrival, closes to 1.25 (cupped) while resting, flares (-0.2) at the flick
    ang = np.full(NF, 0.15)
    for f in range(NF):
        t = T[f]
        if t < c['t_in']: ang[f] = 0.15
        elif t < c['t_launch'] - 0.25: ang[f] = 0.15 + 1.1 * min(1, (t - c['t_in']) / 0.5) ** 0.8
        elif t < c['t_launch']: ang[f] = 1.25 - 1.45 * (t - (c['t_launch'] - 0.25)) / 0.25
        else: ang[f] = -0.2 + 0.4 * min(1, (t - c['t_launch']) / 1.0)
    for pv in pets:
        pv.rotation_mode = 'XYZ'; key_series(pv, 'rotation_euler', 1, -ang, 1e-3)
    stem(base, ground(base[0]), r=0.035)
    # glow of the tulip while the bird rests in it
    lt = bpy.data.lights.new('tulipglow', 'POINT'); lt.color = TULIP_COL[ci % 6]; lt.shadow_soft_size = 0.2
    lo = link(bpy.data.objects.new('tulipglow', lt)); lo.location = tuple(p + np.array([0, -0.3, 0.1]))
    g_ = np.where((T > c['t_in']) & (T < c['t_launch']), 40.0, 0.0) * (1 - np.exp(-np.maximum(0, T - c['t_in']) / 0.3))
    key_socket(lt.energy if False else lt.id_data.__class__ and None, []) if False else None
    for f in range(NF):
        if 0 < f < NF - 1 and abs(g_[f] - g_[f - 1]) < 0.05 and abs(g_[f] - g_[f + 1]) < 0.05: continue
        lt.energy = float(g_[f]); lt.keyframe_insert('energy', frame=f)
    # spring stamen: a coiled stalk that compresses when the bird lands and kicks at the launch
    sp = rod((0, 0, 0), (0, 0, 1), 0.03, M_IRON, 8); sp.parent = hp
    sz = np.full(NF, 0.1)
    for f in range(NF):
        t = T[f]
        if c['t_in'] <= t < c['t_launch']: sz[f] = 0.06
        elif c['t_launch'] <= t < c['t_launch'] + 0.08: sz[f] = 0.06 + 0.12 * (t - c['t_launch']) / 0.08
        elif t >= c['t_launch'] + 0.08: sz[f] = 0.18 - 0.08 * min(1, (t - c['t_launch'] - 0.08) / 0.4)
    key_series(sp, 'scale', 2, sz, 1e-4)

M_BARK = mat('bark', (0.3, 0.18, 0.1), 0.85)
# ======================================================================= greenhouse structure, sky, moon
X0W, X1W = -6.0, float(SX[-1]) + 4.0
ZF = float(SZ.min()) - 3.0                                     # floor level (deep enough for every bed)
ZR = EV['tree']['z_rack'] + 2.4                                # eaves / roof line
YB = 2.6                                                       # back glass wall
box(((X0W + X1W) / 2, 0.3, ZF), (X1W - X0W + 4, 8, 0.2), M_STONE, name='floor')
for x in np.arange(X0W, X1W + 0.1, 2.0):
    rod((x, YB, ZF), (x, YB, ZR), 0.035, M_IRON, 8)                       # mullions of the back glass wall
    rod((x, YB, ZR), (x, YB - 1.6, ZR + 1.4), 0.045, M_IRON, 8)          # roof rafters
for z in np.arange(ZF, ZR + 0.1, 2.4): rod((X0W, YB, z), (X1W, YB, z), 0.025, M_IRON, 6)
gl = box(((X0W + X1W) / 2, YB + 0.05, (ZF + ZR) / 2), (X1W - X0W, 0.02, ZR - ZF), M_GLASS, name='backglass')
# night sky beyond the glass: gradient (+ dawn), stars, moon disc
SKY = bpy.data.materials.new('sky'); SKY.use_nodes = True; nt = SKY.node_tree; nt.nodes.clear()
tc = nt.nodes.new('ShaderNodeTexCoord'); sepz = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(tc.outputs['Object'], sepz.inputs[0])
mpz = nt.nodes.new('ShaderNodeMapRange'); mpz.inputs['From Min'].default_value = -0.5; mpz.inputs['From Max'].default_value = 0.5
nt.links.new(sepz.outputs['Y'], mpz.inputs['Value'])
night = nt.nodes.new('ShaderNodeValToRGB'); night.color_ramp.elements[0].color = (0.02, 0.03, 0.12, 1); night.color_ramp.elements[1].color = (0.0, 0.0, 0.02, 1)
dawnr = nt.nodes.new('ShaderNodeValToRGB'); dawnr.color_ramp.elements[0].color = (1.0, 0.45, 0.15, 1); dawnr.color_ramp.elements[1].color = (0.25, 0.35, 0.75, 1)
e_ = dawnr.color_ramp.elements.new(0.35); e_.color = (1.0, 0.7, 0.55, 1)
nt.links.new(mpz.outputs[0], night.inputs['Fac']); nt.links.new(mpz.outputs[0], dawnr.inputs['Fac'])
mixd = nt.nodes.new('ShaderNodeMix'); mixd.data_type = 'RGBA'
nt.links.new(night.outputs['Color'], mixd.inputs['A']); nt.links.new(dawnr.outputs['Color'], mixd.inputs['B'])
key_socket(mixd.inputs['Factor'], dawn)
noise = nt.nodes.new('ShaderNodeTexVoronoi'); noise.inputs['Scale'].default_value = 140.0
nt.links.new(tc.outputs['Object'], noise.inputs['Vector'])
st = nt.nodes.new('ShaderNodeValToRGB'); st.color_ramp.elements[0].position = 0.0; st.color_ramp.elements[0].color = (1, 1, 1, 1); st.color_ramp.elements[1].position = 0.035; st.color_ramp.elements[1].color = (0, 0, 0, 1)
nt.links.new(noise.outputs['Distance'], st.inputs['Fac'])
stm = nt.nodes.new('ShaderNodeMath'); stm.operation = 'MULTIPLY'; nt.links.new(st.outputs['Color'], stm.inputs[0]); key_socket(stm.inputs[1], 2.5 * (1 - dawn))
add = nt.nodes.new('ShaderNodeMix'); add.data_type = 'RGBA'; add.blend_type = 'ADD'; add.inputs['Factor'].default_value = 1.0
nt.links.new(mixd.outputs['Result'], add.inputs['A']); nt.links.new(stm.outputs[0], add.inputs['B'])
em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Strength'].default_value = 1.0; out = nt.nodes.new('ShaderNodeOutputMaterial')
nt.links.new(add.outputs['Result'], em.inputs['Color']); nt.links.new(em.outputs[0], out.inputs[0])
skyo = obj(prim('plane', SKY), 'sky'); skyo.location = ((X0W + X1W) / 2, 40, (ZF + ZR) / 2 + 4); skyo.rotation_euler = (math.pi / 2, 0, 0); skyo.scale = (X1W - X0W + 120, 70, 1)
M_MOONS = emis('moon', (1.0, 0.95, 0.8), 3.0)
moon = sphere((4.0, 39, ZR - 1.0), 2.6, M_MOONS, (32, 16), 'moon'); moon.scale = (2.6, 0.1, 2.6)
# the rising sun (far right, behind the glass)
M_SUN = emis('sun', (1.0, 0.75, 0.35), 6.0)
sunc = np.array(EV['sunflower']['p'])
sund = sphere((0, 39, 0), 3.2, M_SUN, (32, 16), 'sundisc'); sund.scale = (3.2, 0.1, 3.2)
key_loc(sund, np.stack([np.full(NF, sunc[0] + 6.0), np.full(NF, 39.0), sunc[2] - 9 + 14 * dawn], 1))
# string lights under the rafters: warm bulbs that pulse on the waltz after-beats, dim in the cold
lamp_a = 0.35 + 0.65 * pulse(CHORD, decay=2.5, length=8)
M_LAMP = emis('lamp', (1.0, 0.8, 0.45), 1.0)
key_socket(M_LAMP.node_tree.nodes['Emission'].inputs['Strength'], (3.0 + 12 * lamp_a) * (1 - 0.7 * cold) * (1 - 0.6 * dawn))
for x0 in np.arange(X0W, X1W, 2.0):
    for k in range(8):
        u = (k + 0.5) / 8; x = x0 + 2.0 * u; z = ZR - 0.3 - 0.45 * math.sin(math.pi * u)
        sphere((x, YB - 0.5, z), 0.06, M_LAMP, (10, 6), 'bulb')
    rod((x0, YB - 0.5, ZR - 0.3), (x0 + 2.0, YB - 0.5, ZR - 0.3), 0.008, M_BEAK, 4)

# ---------------- moon lamp (start): the bird is inside; the two halves swing open on the first bass note
mp = np.array(EV['moon']['p'])
M_BRASS = mat('brass', (0.85, 0.6, 0.25), 0.3, **{'Metallic': 0.9})
M_MOONL = mat('moonlamp', (1.0, 0.92, 0.65), 0.3, **{'Emission Color': (1.0, 0.9, 0.6, 1), 'Emission Strength': 2.5})
rod((mp[0], 0, mp[2] + 0.25), (mp[0], 0, ZR + 1.0), 0.02, M_BRASS, 6)
for side in (-1, 1):
    hp = link(bpy.data.objects.new('mhinge', None)); hp.location = (mp[0] + side * 0.02, 0, mp[2] + 0.24)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.25, segments=32, ring_count=16); hm = bpy.context.object
    bm = bmesh.new(); bm.from_mesh(hm.data)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.x * side < -1e-4], context='VERTS'); bm.to_mesh(hm.data); bm.free()
    for p in hm.data.polygons: p.use_smooth = True
    sol = hm.modifiers.new('s', 'SOLIDIFY'); sol.thickness = 0.015
    hm.data.materials.append(M_MOONL); hm.parent = hp; hm.location = (0, 0, -0.24)
    rot = np.zeros(NF)
    for f in range(NF):
        u = (T[f] - EV['moon']['t_open']) / 0.25
        if u > 0: rot[f] = side * (min(1.0, u) ** 0.6 * 1.6 + 0.1 * math.sin(7 * u) * math.exp(-1.5 * u))
    hp.rotation_mode = 'XYZ'; key_series(hp, 'rotation_euler', 1, rot)
mlt = bpy.data.lights.new('moonlamp', 'POINT'); mlt.energy = 60; mlt.color = (1.0, 0.9, 0.7); mlt.shadow_soft_size = 0.3
link(bpy.data.objects.new('moonlamp', mlt)).location = (mp[0], -0.4, mp[2])

# ======================================================================= watering cans + drops + vines + ponds
M_CAN = mat('can', (0.25, 0.55, 0.65), 0.35, **{'Metallic': 0.7})
for ci, c in enumerate(EV['cans']):
    sp_ = np.array(c['spout'])
    piv = link(bpy.data.objects.new('can', None)); piv.location = tuple(sp_ + np.array([0, 0, R]))
    # body behind/above the spout tip (spout points down-left toward the leaves when tipped)
    body = rod((0.75, 0, 0.25), (0.75, 0, 0.75), 0.28, M_CAN, 24); body.parent = piv
    rim = rod((0.75, 0, 0.75), (0.75, 0, 0.78), 0.3, M_CAN, 24); rim.parent = piv
    spt = rod((0.0, 0, 0.0), (0.55, 0, 0.35), 0.05, M_CAN, 10); spt.parent = piv
    rose = obj(prim('cone', M_CAN, 16, 0.09, 0.04, 0.1), 'rose'); rose.parent = piv; rose.location = (0.02, 0, 0.0); rose.rotation_euler = (0, -2.1, 0)
    hd = rod((1.0, 0, 0.35), (1.15, 0, 0.6), 0.025, M_CAN, 8); hd.parent = piv
    rod((sp_[0] + 0.75, 0.35 + sp_[1], sp_[2] + 0.9), (sp_[0] + 0.75, 0.35 + sp_[1], ZR), 0.012, M_BEAK, 6)   # hanging wire
    ang = np.zeros(NF)
    for f in range(NF):
        u = (T[f] - c['t_tip']) / 0.12
        if u > 0: ang[f] = 0.45 * min(1, u) + 0.06 * math.sin(9 * u) * math.exp(-2 * u)
    piv.rotation_mode = 'XYZ'; key_series(piv, 'rotation_euler', 1, ang)
    text('じょうろ' if ci % 3 == 0 else '', (sp_[0] + 0.75, sp_[1] - 0.3, sp_[2] + 0.55), 0.13, M_OBJ)
    if c.get('lever'):
        lv = np.array(c['lever'])
        box((lv[0], lv[1], lv[2] - R - 0.03), (0.3, 0.3, 0.04), M_BRASS, name='levercup')
        rod((lv[0] + 0.15, lv[1], lv[2] - R - 0.03), (sp_[0] + 0.75, sp_[1], sp_[2] + 0.3), 0.02, M_BRASS, 6)
    if c.get('string_from'):
        sf = np.array(c['string_from'])
        rod((sf[0], 0.05, sf[2] - 0.6), (sp_[0] + 1.1, sp_[1] + 0.05, sp_[2] + 0.55), 0.008, mat('string', (0.9, 0.85, 0.7), 0.8), 4)
# vines (behind the drop lane) holding the leaves
for K in KEYS:
    if K['kind'] == 'leaf':
        c_ = K['c']; rod((c_[0], c_[1], c_[2] - 0.03), (c_[0], c_[1] + 0.32, c_[2] - 0.08), 0.012, M_STEM, 6)
        rod((c_[0], c_[1] + 0.32, c_[2] - 0.08), (c_[0], c_[1] + 0.32, ZR), 0.016, M_STEM, 6)
M_WATER = mat('water', (0.2, 0.45, 0.6), 0.03, **{'Transmission Weight': 0.6, 'Coat Weight': 1.0})
for pd in EV['ponds']:
    p = np.array(pd['p'])
    bpy.ops.mesh.primitive_cylinder_add(radius=0.42, depth=0.22, vertices=32, location=(p[0], p[1], p[2] - R - 0.18)); bs = bpy.context.object; bs.data.materials.append(M_STONE)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.38, depth=0.02, vertices=32, location=(p[0], p[1], p[2] - R - 0.07)); wt = bpy.context.object; wt.data.materials.append(M_WATER)
    for dx_ in (-0.38, 0.38): rod((p[0] + dx_, p[1] + 0.1, p[2] - R - 0.1), (p[0] + dx_, p[1] + 0.1, ZR), 0.01, M_BEAK, 4)   # hanging basin
    for k in range(3):
        lp = obj(prim('cyl', M_LEAF, 24), 'lily'); lp.location = (p[0] + 0.2 * math.cos(k * 2.1), p[1] + 0.2 * math.sin(k * 2.1), p[2] - R - 0.055); lp.scale = (0.11, 0.11, 0.008); lp.color = (0.2, 0.6, 0.25, 1)
    # splash crown
    for k in range(10):
        sd = sphere((0, 0, 0), 0.03, M_FX, (8, 4), 'splash'); sd.color = (0.6, 0.85, 1.0, 0)
        a = 2 * math.pi * k / 10; v = np.array([0.8 * math.cos(a), 0.5 * math.sin(a), 2.2])
        P = np.zeros((NF, 3)); A = np.zeros(NF)
        for f in range(NF):
            s_ = T[f] - pd['t']
            if 0 <= s_ < 0.6: P[f] = p + np.array([0, 0, -R]) + v * s_ + np.array([0, 0, -0.5 * G * s_ * s_]); A[f] = 1 - s_ / 0.6
            else: P[f] = p
        key_loc(sd, P); key_series(sd, 'color', 3, A, 1e-3)

# ======================================================================= the golden bell tree, rack, gutters, buckets, rope, window, windmill
tr = EV['tree']; XT = tr['x']; ZRK = tr['z_rack']
rod((XT, 0.5, ZF), (XT, 0.5, ZRK - 0.3), 0.42, M_BARK, 16)
for k in range(6): rod((XT, 0.5, ZF + 0.3), (XT + 0.9 * math.cos(k), 0.5 + 0.6 * math.sin(k), ZF - 0.2), 0.12, M_BARK, 8)
gkeys = [K for K in KEYS if K['kind'] == 'bell']
for K in gkeys:                                           # each bell on a short stalk to a branch behind it
    c_ = K['c']; yb = c_[1] + 0.32
    rod((c_[0], c_[1], c_[2] - 0.12), (c_[0], yb, c_[2] - 0.32), 0.014, M_STEM, 6)
for gn in ('G1', 'G2', 'G4'):
    ks = sorted([K for K in gkeys if K['ball'] == gn], key=lambda K: K['t'])
    pts = [(XT, ks[0]['c'][1] + 0.32, ks[0]['c'][2] - 0.05)] + [(K['c'][0], K['c'][1] + 0.32, K['c'][2] - 0.32) for K in ks]
    for a, b in zip(pts, pts[1:]):
        rod(a, b, 0.07, M_BARK, 8)
        for u in (0.25, 0.75):
            q = np.array(a) * (1 - u) + np.array(b) * u; sphere(tuple(q + np.array([0, -0.08, 0.05])), 0.035, M_LAMP, (8, 4), 'fairy')
    rod(pts[-1], (pts[-1][0], pts[-1][1], pts[-1][2] - 0.5), 0.03, M_BARK, 6)
# leaves on the tree for a fuller crown
for k in range(220):
    K = gkeys[k % len(gkeys)]; c_ = K['c'] + np.array([rng.uniform(-0.5, 0.5), 0.4 + rng.uniform(0, 0.7), rng.uniform(-0.6, 0.1)])
    lf = obj(prim('petal', M_LEAF, 0.45, 0.26, 0.6), 'treeleaf'); lf.location = tuple(c_); lf.rotation_euler = (rng.uniform(-0.5, 0.5), rng.uniform(-0.6, 0.2), rng.uniform(0, 6.28)); lf.color = (0.2, 0.42, 0.12, 1)
# seed rack on top of the trunk: a box with two flap gates; G3 waits behind G1 in the magazine
box((XT, 0.0, ZRK - R - 0.06), (1.2, 1.1, 0.08), M_BRASS, name='rack')
for side in (-1, 1): box((XT + side * 0.62, 0.0, ZRK - 0.02), (0.04, 1.1, 0.3), M_BRASS, name='rackwall')
for rk in EV['rack']:
    s_ = np.array(rk['slot']); d = rk['dir']
    hp = link(bpy.data.objects.new('gate', None)); hp.location = (s_[0] + d * (R + 0.04), s_[1], s_[2] + 0.2)
    fl = box((0, 0, -0.17), (0.03, 0.3, 0.32), M_IRON, name='flap'); fl.parent = hp
    ang = np.zeros(NF)
    for f in range(NF):
        u = (T[f] - rk['t_gate'] + 0.04) / 0.08
        if u > 0: ang[f] = d * 1.3 * min(1, u) * (1 - min(1, max(0, (T[f] - rk['t_gate'] - 0.4) / 0.3)))
    hp.rotation_mode = 'XYZ'; key_series(hp, 'rotation_euler', 1, -ang)
# gutters
M_GUT = mat('gutter', (0.6, 0.62, 0.66), 0.3, **{'Metallic': 0.9})
for sd, gu in EV['gutters'].items():
    x0, x1, z0 = gu['x0'], gu['x1'], gu['z0']; th = math.radians(8.0); s = float(sd)
    zc = lambda x: z0 - abs(x - x0) * math.tan(th) - R
    for y in (-0.42, 0.42):
        rod((x0, y, zc(x0) + 0.08), (x1, y, zc(x1) + 0.08), 0.03, M_GUT, 8)
    xh = x1 - s * (R + 0.06)                                   # the bottom stops short of the end: the drop hole
    ln = abs(xh - x0) / math.cos(th)
    bot = box(((x0 + xh) / 2, 0.0, (zc(x0) + zc(xh)) / 2 - 0.015), (ln, 0.88, 0.03), M_GUT, rot=(0, s * th, 0), name='gutter')
    for x in np.linspace(x0, x1, 5): rod((x, 0.45, zc(x)), (x, 0.45, ZR), 0.012, M_IRON, 6)
def ch_end_z(sd):
    gu = EV['gutters'][sd]; return gu['z0'] - abs(gu['x1'] - gu['x0']) * math.tan(math.radians(8.0))
# counterweight buckets on the rope; window in the roof above the tree; rope over pulleys
BK = {}
for sd, bk in EV['buckets'].items():
    keys = np.array(bk['keys']); zb = np.interp(T, keys[:, 0], keys[:, 1]); BK[sd] = zb
    piv = link(bpy.data.objects.new('bucket', None))
    key_loc(piv, np.stack([np.full(NF, bk['x']), np.full(NF, -0.075), zb - R - 0.04], 1))
    bpy.ops.mesh.primitive_cylinder_add(radius=0.56, depth=0.75, vertices=32, location=(0, 0, 0.375)); bo = bpy.context.object
    bm = bmesh.new(); bm.from_mesh(bo.data); bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.normal.z > 0.9], context='FACES'); bm.to_mesh(bo.data); bm.free()
    bo.modifiers.new('s', 'SOLIDIFY').thickness = 0.025; bo.data.materials.append(M_BRASS); bo.parent = piv
    for y in (-0.56, 0.56): rod((0, y, 0.75), (0, 0, 1.3), 0.01, M_BEAK, 4).parent = piv
    sd_ = float(sd); box((bk['x'] + sd_ * (R + 0.04), 0.0, ch_end_z(sd) + 0.05), (0.04, 0.9, 0.35), M_GUT, name='endstop')
    rp = rod((0, 0, 0), (0, 0, 1), 0.012, mat('rope', (0.85, 0.75, 0.55), 0.9), 6)       # rope from bucket bail to pulley
    PZ = ZR + 0.2
    P = np.stack([np.full(NF, bk['x']), np.zeros(NF), zb - R - 0.04 + 1.3], 1)
    key_loc(rp, P); key_series(rp, 'scale', 2, PZ - P[:, 2], 1e-4)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.18, minor_radius=0.04, location=(bk['x'], 0, PZ), rotation=(math.pi / 2, 0, 0)); bpy.context.object.data.materials.append(M_IRON)
    # ratchet notch counter next to each bucket: 2 lamps per side light up when that side's balls have arrived
    text('おもり', (bk['x'], -0.6, zb[0] + 0.0), 0.18, M_OBJ).hide_render = True
# window: roof pane hinged at its top edge, flung open by the gust; each ratchet notch pulls it a quarter shut
WX = XT + 1.6; WZ = ZR + 1.2
open_frac = np.zeros(NF)
for sd, zb in BK.items():
    bk = EV['buckets'][sd]; open_frac += 0.5 * (zb - bk['z_low']) / (bk['z_high'] - bk['z_low'])
whp = link(bpy.data.objects.new('window', None)); whp.location = (WX - 1.2, YB - 1.0, WZ + 0.6)
wp = box((1.2, 0, -0.5), (2.4, 0.05, 1.2), M_GLASS, name='pane'); wp.parent = whp
for dz in (0, -1.0): rod((0, 0, dz), (2.4, 0, dz), 0.03, M_IRON, 6).parent = whp
for dx in (0, 1.2, 2.4): rod((dx, 0, 0), (dx, 0, -1.0), 0.03, M_IRON, 6).parent = whp
whp.rotation_mode = 'XYZ'; key_series(whp, 'rotation_euler', 0, -1.1 * open_frac, 1e-4)
for sd, bk in EV['buckets'].items():
    rod((bk['x'], 0, ZR + 0.2), (WX, YB - 1.0, WZ - 0.3), 0.012, mat('rope2', (0.85, 0.75, 0.55), 0.9), 6)
# thermometer on the trunk: the red column drops in the cold
M_RED = emis('thermo', (1.0, 0.12, 0.08), 3.0)
tz0 = ZRK - 2.6
rod((XT, -0.36, tz0), (XT, -0.36, tz0 + 1.4), 0.07, M_GLASS, 12)
sphere((XT, -0.36, tz0 - 0.05), 0.11, M_RED, (16, 8))
col_ = rod((0, 0, 0), (0, 0, 1), 0.04, M_RED, 10); col_.location = (XT, -0.37, tz0)
key_series(col_, 'scale', 2, 1.25 - 0.95 * cold, 1e-3)
# windmill on the roof: spins while the wind blows; drive shaft + gears down to the rack cam
wmx, wmz = XT, ZR + 2.6
rod((wmx, YB - 1.4, ZR + 1.0), (wmx, YB - 1.4, wmz), 0.06, M_IRON, 8)
hub = link(bpy.data.objects.new('mill', None)); hub.location = (wmx, YB - 1.6, wmz)
for k in range(4):
    bl = box((0, 0, 0.75), (0.32, 0.03, 1.3), M_OBJ, name='blade'); bl.color = (0.95, 0.95, 0.9, 1)
    pv = link(bpy.data.objects.new('bpiv', None)); pv.parent = hub; pv.rotation_euler = (0, k * math.pi / 2, 0); bl.parent = pv
wind = cold.copy()
spin = np.cumsum(wind * 9.0 / FPS)
hub.rotation_mode = 'XYZ'; key_series(hub, 'rotation_euler', 1, spin, 1e-4)
rod((wmx, YB - 1.4, ZR + 1.0), (XT, 0.0, ZRK + 0.4), 0.03, M_IRON, 8)
for zz, r_ in ((ZRK + 0.45, 0.22), (ZR + 0.95, 0.18)):
    gp = link(bpy.data.objects.new('gear', None)); gp.location = (XT if zz < ZR else wmx, 0.0 if zz < ZR else YB - 1.4, zz)
    bpy.ops.mesh.primitive_cylinder_add(radius=r_, depth=0.06, vertices=20, location=(0, 0, 0)); gb = bpy.context.object; gb.data.materials.append(M_BRASS); gb.parent = gp
    for k in range(10):
        tt_ = box((r_ * math.cos(k * 0.628), r_ * math.sin(k * 0.628), 0), (0.07, 0.07, 0.06), M_BRASS, rot=(0, 0, k * 0.628), name='tooth'); tt_.parent = gp
    gp.rotation_mode = 'XYZ'; key_series(gp, 'rotation_euler', 2, spin * (1.6 if zz < ZR else -1.6), 1e-4)
# snow / frost flakes blown in from the open window (pool, each re-used every 2.4 s)
M_SNOW = emis('snow', (0.85, 0.92, 1.0), 2.0)
for k in range(160):
    fl = obj(prim('ico', M_SNOW, 1), 'flake'); fl.scale = (0.025, 0.025, 0.025)
    ph = rng.uniform(0, 2.4); y0 = rng.uniform(-1.6, 1.2); z0 = rng.uniform(-1.0, 3.0)
    P = np.zeros((NF, 3)); vis = np.zeros(NF)
    for f in range(NF):
        t = T[f]
        if cold[f] <= 0.02: P[f] = (WX, YB, WZ + 5); continue
        s_ = ((t - T_GUST + ph) % 2.4)
        x = WX - 0.5 - s_ * 6.5 + 0.2 * math.sin(3 * t + k)
        P[f] = (x, y0, WZ - 0.5 - s_ * (WZ - (z0 - 4.0)) / 2.4 + 0.15 * math.cos(2 * t + k)); vis[f] = cold[f]
    key_loc(fl, P)
    sc_ = 0.025 * vis + 1e-4
    for c_ in range(3): key_series(fl, 'scale', c_, sc_, 1e-4)


# ---------------- background foliage (behind the lanes, never struck): ferns, palms, big leaves; warm light pools
M_FOL = mat('foliage', (0.06, 0.2, 0.08), 0.6, **{'Coat Weight': 0.2})
M_FOL2 = mat('foliage2', (0.12, 0.3, 0.1), 0.6)
for x in np.arange(X0W, X1W, 0.9):
    zg = ground(x)
    for j in range(3):
        lf = obj(prim('petal', M_FOL if j % 2 else M_FOL2, rng.uniform(0.8, 1.6), rng.uniform(0.35, 0.6), 0.5), 'fern')
        lf.location = (x + rng.uniform(-0.4, 0.4), rng.uniform(1.3, 2.2), zg - 0.1)
        lf.rotation_euler = (rng.uniform(-0.3, 0.3), -rng.uniform(0.6, 1.3), rng.choice([0.0, math.pi]) + rng.uniform(-0.6, 0.6))
for x in np.arange(X0W + 2, X1W, 5.5):                           # potted palms
    zg = ground(x); top = zg + rng.uniform(2.6, 3.6); y = 1.9
    rod((x, y, zg - 0.6), (x, y, top), 0.07, M_BARK if 'M_BARK' in dir() else M_PLANK, 8)
    for k in range(8):
        lf = obj(prim('petal', M_FOL2, 1.3, 0.4, 0.8), 'palm'); lf.location = (x, y, top)
        lf.rotation_euler = (0, 0.5 + rng.uniform(-0.2, 0.3), k * math.pi / 4 + rng.uniform(-0.2, 0.2))
for x in np.arange(X0W + 1, X1W, 4.0):                           # warm lanterns on posts (light pools)
    zg = ground(x)
    lt = bpy.data.lights.new('lantern', 'POINT'); lt.color = (1.0, 0.72, 0.4); lt.shadow_soft_size = 0.25
    lo = link(bpy.data.objects.new('lantern', lt)); lo.location = (x, 1.1, zg + 2.2)
    for f in range(0, NF, 3): lt.energy = float(70 * (1 - 0.65 * cold[f]) * (1 - 0.5 * dawn[f])); lt.keyframe_insert('energy', frame=f)
    sphere((x, 1.1, zg + 2.2), 0.09, M_LAMP, (12, 6), 'lanternbulb')
    rod((x, 1.1, zg + 2.1), (x, 1.1, zg - 0.6), 0.025, M_IRON, 6)

# ======================================================================= signs (keywords in the first 30 s), plant labels
M_SIGN = mat('sign', (0.45, 0.28, 0.14), 0.75)
M_TXT_W = emis('txtw', (1.0, 0.97, 0.9), 1.6); M_TXT_Y = emis('txty', (1.0, 0.85, 0.35), 2.0); M_TXT_P = emis('txtp', (1.0, 0.6, 0.75), 2.0)
def signboard(c, w, h, lines, hang=True):
    c = np.array(c, float)
    box((c[0], c[1] + 0.06, c[2]), (w, 0.06, h), M_SIGN, name='sign')
    if hang:
        for dx in (-w / 2 + 0.1, w / 2 - 0.1): rod((c[0] + dx, c[1] + 0.06, c[2] + h / 2), (c[0] + dx, c[1] + 0.06, ZR), 0.01, M_BEAK, 4)
    else:
        rod((c[0], c[1] + 0.1, c[2] - h / 2), (c[0], c[1] + 0.1, ground(c[0])), 0.04, M_SIGN, 6)
    for s_, dz, size, m_ in lines: text(s_, (c[0], c[1] - 0.005, c[2] + dz), size, m_, ext=0.01)
# opening title boards (hang above the first beds)
signboard((1.0, -0.9, 7.0), 3.9, 2.1, [('チャイコフスキー', 0.62, 0.38, M_TXT_W), ('くるみ割り人形', 0.0, 0.42, M_TXT_Y), ('「花のワルツ」', -0.64, 0.46, M_TXT_P)])
gz = ground(5.0)
signboard((5.2, -0.95, gz + 1.0), 2.5, 0.62, [('クラシック', 0.0, 0.44, M_TXT_W)], hang=False)
signboard((9.8, -0.95, ground(9.8) + 0.95), 2.0, 0.62, [('ビー玉', 0.0, 0.46, M_TXT_Y)], hang=False)
signboard((14.2, -0.95, ground(14.2) + 0.95), 2.3, 0.62, [('からくり', 0.0, 0.46, M_TXT_W)], hang=False)
signboard((18.8, -0.95, ground(18.8) + 1.0), 2.0, 0.62, [('ワルツ', 0.0, 0.46, M_TXT_P)], hang=False)
signboard((24.5, -0.95, ground(24.5) + 1.0), 2.8, 0.62, [('バレエ音楽', 0.0, 0.44, M_TXT_Y)], hang=False)
signboard((5.3, -0.9, ground(5.3) + 5.6), 3.0, 0.7, [('魔法の温室', 0.0, 0.5, M_TXT_W)])
# end board with the comment question
sf_ = np.array(EV['sunflower']['p'])
signboard((sf_[0], -0.9, sf_[2] + 3.9), 4.4, 0.8, [('何の花が一番好き？', 0.0, 0.46, M_TXT_Y)])

# ======================================================================= the giant sunflower (final chord)
M_SFD = mat('sfdisc', (0.25, 0.12, 0.04), 0.7, **{'Emission Color': (1, 0.6, 0.2, 1), 'Emission Strength': 0.0})
sfp = sf_ - np.array([0, 0, R + 0.05])
dsk = obj(prim('cyl', M_SFD, 40), 'sfdisc'); dsk.location = tuple(sfp); dsk.scale = (0.5, 0.5, 0.06)
ems = np.where(T >= FINAL, 3.0 * np.exp(-np.maximum(0, T - FINAL) / 2.0) + 0.6, 0.0)
key_socket(M_SFD.node_tree.nodes['Principled BSDF'].inputs['Emission Strength'], ems)
rod((sfp[0], 0.1, sfp[2] - 0.08), (sfp[0], 0.1, ground(sfp[0])), 0.07, M_STEM, 12)
sfpiv = link(bpy.data.objects.new('sunflower', None)); sfpiv.location = tuple(sfp)
for k in range(22):
    a = 2 * math.pi * k / 22
    pe = obj(prim('petal', M_PET, 0.6, 0.22, 0.6), 'sfpetal'); pe.parent = sfpiv; pe.location = (0.42 * math.cos(a), 0.42 * math.sin(a), 0)
    pe.rotation_euler = (0, -0.55, a); pe.color = (1.0, 0.78, 0.1, 1)
ssc = np.where(T < FINAL - 0.05, 0.2, np.minimum(1.0, (T - FINAL + 0.05) / 0.35) ** 0.6 * 1.0 + 0.2 * (T < FINAL - 0.05))
ssc = np.where(T < FINAL - 0.05, 0.2, 0.2 + 0.85 * np.minimum(1.0, (T - FINAL + 0.05) / 0.4) ** 0.6)
for k in range(3): key_series(sfpiv, 'scale', k, ssc, 1e-3)
for k in range(8):                                     # leaves on the big stem
    lf = obj(prim('petal', M_LEAF, 0.7, 0.4, 0.6), 'sfleaf'); lf.location = (sfp[0], 0.1, sfp[2] - 0.8 - 0.45 * k); lf.rotation_euler = (0, -0.3, (k % 2) * math.pi + 0.2); lf.color = (0.2, 0.5, 0.15, 1)
# confetti of petals at the final chord
for k in range(80):
    cf = obj(prim('petal', M_PET, 0.09, 0.06, 0.5), 'conf'); cf.color = (*[(1, 0.5, 0.7), (1, 0.85, 0.3), (0.8, 0.6, 1), (1, 1, 1)][k % 4], 1)
    a0 = rng.uniform(0, 2 * math.pi); sp = rng.uniform(1.0, 3.0); vy = rng.uniform(-0.8, 0.8)
    P = np.zeros((NF, 3)); RT = np.zeros((NF, 3))
    for f in range(NF):
        s_ = T[f] - FINAL
        if s_ < 0: P[f] = sfp; continue
        burst = 1 - math.exp(-s_ / 0.3)
        P[f] = (sfp[0] + sp * math.cos(a0) * burst + 0.15 * math.sin(3 * s_ + k), sfp[1] - 0.3 + vy * burst, sfp[2] + 0.5 + sp * math.sin(a0) * 0.7 * burst - 0.5 * s_)
        RT[f] = (s_ * (3 + k % 5), s_ * (2 + k % 3), s_)
    key_loc(cf, P)
    sc_ = np.where(T >= FINAL, 1.0, 0.001)
    for c_ in range(3): key_series(cf, 'scale', c_, sc_); key_series(cf, 'rotation_euler', c_, RT[:, c_])

# ======================================================================= lights & world
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes['Background']
key_socket(bg.inputs['Strength'], 0.25 + 0.6 * dawn)
wc = np.stack([0.05 + 0.6 * dawn, 0.07 + 0.35 * dawn, 0.18 - 0.03 * dawn], 1)
for f in range(NF):
    if f % 6 and f < NF - 1: continue
    bg.inputs['Color'].default_value = (*wc[f], 1); bg.inputs['Color'].keyframe_insert('default_value', frame=f)
moonl = bpy.data.lights.new('moonlight', 'SUN'); moonl.angle = 0.1
mo = link(bpy.data.objects.new('moonlight', moonl)); mo.rotation_euler = (math.radians(55), math.radians(15), math.radians(-30))
for f in range(0, NF, 3):
    moonl.energy = float(0.9 + 0.8 * cold[f] - 0.6 * dawn[f]); moonl.keyframe_insert('energy', frame=f)
    moonl.color = (0.55 + 0.45 * dawn[f], 0.65 + 0.15 * dawn[f], 1.0 - 0.45 * dawn[f]); moonl.keyframe_insert('color', frame=f)
sunl = bpy.data.lights.new('sunrise', 'SUN'); sunl.angle = 0.05; sunl.color = (1.0, 0.7, 0.4)
so = link(bpy.data.objects.new('sunrise', sunl)); so.rotation_euler = (math.radians(80), math.radians(-60), math.radians(160))
key_series(sunl, 'energy', 0, 6.0 * dawn) if False else None
for f in range(0, NF, 3): sunl.energy = float(5.0 * dawn[f]); sunl.keyframe_insert('energy', frame=f)
# warm fill lights following the camera (set up after the camera path)

# ======================================================================= camera
def ball_pos(n, t): return seg_at(LAY['balls'][n]['seg'], t)
FOC = np.zeros((NF, 3)); WID = np.full(NF, 5.6)
def active(t):
    """who carries the melody right now (+ look-ahead target)"""
    best = None
    for i, l in enumerate(LEAD):
        if l['t'] >= t - 0.05: best = i; break
    if best is None: best = len(LEAD) - 1
    return best
owner = {}
for n, b in LAY['balls'].items():
    for h in b['hits']:
        if h['kind'] in KEY or h['kind'] == 'cup': owner.setdefault(h['i'], n)
for f in range(NF):
    t = T[f]
    i = active(t); n = owner.get(i, 'H')
    p = ball_pos(n, t); q = ball_pos(n, LEAD[i]['t'])
    FOC[f] = 0.6 * p + 0.4 * q
    if 7.5 < t < 16.0: FOC[f] = FOC[f] + np.array([0.3, 0, -0.9])      # keep H's tulip (below) in the frame with the drops
    if T_GUST + 7.5 < t < T_SHUT + 0.5 and n.startswith('G'): FOC[f] = FOC[f] + np.array([0, 0, 1.2])
# story shots (blended in/out over 0.4 s)
def shot(t0, t1, foc, wid, ease=0.4):
    w_ = np.clip(np.minimum((T - t0 + ease) / ease, (t1 + ease - T) / ease), 0, 1); w_ = w_ * w_ * (3 - 2 * w_)
    FOC[:] = FOC * (1 - w_[:, None]) + np.array(foc)[None, :] * w_[:, None]; WID[:] = WID * (1 - w_) + wid * w_
XT_, ZRK_ = EV['tree']['x'], EV['tree']['z_rack']
shot(-1, 0.9, (0.9, 0, 4.2), 5.8)
cc = np.array(EV['cups'][3]['p'])
shot(29.6, 30.0, ((cc[0] + XT_) / 2 + 0.4, 0, (cc[2] + ZRK_) / 2 - 0.3), 6.2)          # H shelters, the first seed drops
for rk in EV['rack'][1:]:
    shot(rk['t_gate'] - 0.5, rk['t_gate'] + 0.15, (XT_, 0, ZRK_ - 0.6), 6.0)            # each gate: rack, window, windmill
for f in range(NF):
    t = T[f]
    if t >= FINAL + 0.4:
        u = min(1.0, (t - FINAL - 0.4) / 3.0); u = u * u * (3 - 2 * u)
        FOC[f] = (1 - u) * FOC[f] + u * np.array([EV['sunflower']['p'][0] - 7.0, 0, EV['sunflower']['p'][2] + 3.5]); WID[f] = (1 - u) * WID[f] + u * 14.6
FOC[:, 2] += 0.6
FOC[:, 0] = smooth(FOC[:, 0], 0.32); FOC[:, 2] = smooth(FOC[:, 2], 0.45); WID = smooth(WID, 0.5)
LENS = 30.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 300
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24
    CP[f] = (FOC[f, 0], -D, FOC[f, 2] + 0.12 * D)
    TP[f] = (FOC[f, 0], 0.0, FOC[f, 2])
key_loc(cam, CP); key_loc(tgt, TP)
for dx, en, col in ((-1.5, 260.0, (1.0, 0.85, 0.65)), (2.5, 160.0, (0.75, 0.85, 1.0))):
    fl_ = bpy.data.lights.new('fill', 'AREA'); fl_.size = 3.0; fl_.color = col
    fo = link(bpy.data.objects.new('fill', fl_)); fo.parent = cam; fo.location = (dx, 2.5, 1.0)
    for f in range(0, NF, 3): fl_.energy = float(en * (1 - 0.35 * cold[f])); fl_.keyframe_insert('energy', frame=f)


# ======================================================================= checks
def check():
    info = {}
    gaps = []
    for K in KEYS:
        f = int(round(K['t'] * FPS)); p = POS[K['ball']][min(NF - 1, f)]
        gaps.append(np.linalg.norm(p - (K['contact'] + R * K['n'])))
    info['contact_gap_at_hit_frame_max'] = round(max(gaps), 3)
    info['keys'] = len(KEYS); info['flowers'] = len(FLOWERS); info['objects'] = len(bpy.data.objects); info['NF'] = NF
    # ball size on screen (fraction of frame width)
    info['ball_frac_of_width_min'] = round(float((2 * R / WID).min()), 3); info['ball_frac_during_melody_min'] = round(float((2 * R / WID[T < FINAL]).min()), 3)
    return info


if MODE in ('check', 'test'): print(check())
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
