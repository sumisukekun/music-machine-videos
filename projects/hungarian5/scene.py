"""Hungarian Dance No. 5 - night rooftop chase (夜の屋根の上の追いかけっこ). Side view across two rows of roofs under the moon.
Thief (black ball with a green karakusa bundle, front row) strikes every melody note on roof tiles (xylophone), chimney
bells, gutter brackets and deck slats; the police ball (blue, cap with siren lamp, back row) strikes every bass note on
copper plates; the siren lamp flashes on every off-beat chord. layout.py plans all motion; this file builds and animates.
python3 scene.py -- check | test t1 t2 ... | range f0 f1"""
import os, sys, json, math, time
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json')); BASS = json.load(open(f'{HERE}/bass.json'))
LAY = json.load(open(f'{HERE}/layout.json'))
FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.2) * FPS); T = np.arange(NF) / FPS
G = LAY['G']; R = LAY['R']; EV = LAY['ev']; YT, YC = LAY['YT'], LAY['YC']
SECS = {s['name']: s for s in E['secs']}
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
sc.cycles.transparent_max_bounces = 6
sc.render.resolution_x, sc.render.resolution_y = 1080, 1920
sc.render.resolution_percentage = int(os.environ.get('PCT', 50)); sc.render.fps = FPS
sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.5
try: sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Punchy'
except Exception: pass
sc.frame_start, sc.frame_end = 0, NF - 1
FONT = bpy.data.fonts.load(f'{HERE}/jp_h5.otf')


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


def key_value(sock, vals, eps=1e-4):
    n = len(vals)
    for f in range(n):
        if 0 < f < n - 1 and abs(vals[f] - vals[f - 1]) < eps and abs(vals[f] - vals[f + 1]) < eps: continue
        sock.default_value = float(vals[f]); sock.keyframe_insert('default_value', frame=f)


_MC = {}
def prim(kind, m, *a):
    key = (kind, m.name if m else None) + tuple(a)
    if key in _MC: return _MC[key]
    bm = bmesh.new()
    if kind == 'cube': bmesh.ops.create_cube(bm, size=1.0)
    elif kind == 'cyl': bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=1.0, radius2=1.0, depth=1.0)
    elif kind == 'cone': bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=a[1], radius2=a[2], depth=a[3])
    elif kind == 'sph': bmesh.ops.create_uvsphere(bm, u_segments=a[0], v_segments=a[1], radius=1.0)
    elif kind == 'plane': bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    me = bpy.data.meshes.new(kind); bm.to_mesh(me); bm.free()
    if kind in ('sph', 'cone'):
        for p in me.polygons: p.use_smooth = True
    if m: me.materials.append(m)
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
    cu = bpy.data.curves.new('t', 'FONT'); cu.body = s; cu.font = FONT; cu.size = size; cu.extrude = ext
    cu.align_x = align; cu.align_y = 'CENTER'
    o = link(bpy.data.objects.new('txt', cu)); o.location = loc; o.rotation_euler = rot; o.data.materials.append(m); return o


def pulse_alpha(times, decay=5.0, length=16, amp=1.0):
    a = np.zeros(NF)
    for t in times:
        f0 = int(round(t * FPS)); m_ = np.arange(max(0, f0), min(NF, f0 + length))
        a[m_] = np.maximum(a[m_], amp * np.exp(-(m_ - f0) / decay))
    return a


def frame_from_n(n):
    n = np.array(n, float); n /= np.linalg.norm(n)
    xa = np.cross([0, 1, 0], n); xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return Matrix(((xa[0], ya[0], n[0]), (xa[1], ya[1], n[1]), (xa[2], ya[2], n[2])))


def smooth(Xa, k):
    Y = np.array(Xa, float).copy(); a = 1 - math.exp(-1 / FPS / k)
    for f in range(1, len(Y)): Y[f] = Y[f - 1] + (Y[f] - Y[f - 1]) * a
    for f in range(len(Y) - 2, -1, -1): Y[f] = Y[f + 1] + (Y[f] - Y[f + 1]) * a
    return Y


# ---------------- motion ----------------
def seg_at(g, t):
    p = np.array(g['p']); u = t - g['t0']
    if g['k'] == 'rest': return p
    if g['k'] == 'fly': return p + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
    return p + np.array(g['u']) * (g['v0'] * u + 0.5 * g['a'] * u * u)
class BallPath:
    def __init__(s, d): s.seg = d['seg']; s.hits = d['hits']
    def at(s, t):
        for g in s.seg:
            if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9: return seg_at(g, t)
        return seg_at(s.seg[0], s.seg[0]['t0']) if t < s.seg[0]['t0'] else seg_at(s.seg[-1], s.seg[-1]['t1'])
BP = {n: BallPath(d) for n, d in LAY['balls'].items()}
POS = {n: np.array([b.at(t) for t in T]) for n, b in BP.items()}
HITS = {n: sorted(b.hits, key=lambda h: h['t']) for n, b in BP.items()}

# ---------------- materials ----------------
M_THIEF = mat('thief', (0.07, 0.07, 0.085), 0.16, **{'Coat Weight': 1.0, 'Coat Roughness': 0.05})
M_COP = mat('cop', (0.05, 0.25, 0.95), 0.15, **{'Coat Weight': 1.0})
M_EYE = mat('eye', (0.97, 0.97, 0.95), 0.25); M_PUPIL = mat('pupil', (0.01, 0.01, 0.01), 0.2, **{'Coat Weight': 1.0})
M_CAP = mat('cap', (0.03, 0.05, 0.2), 0.45); M_BADGE = mat('badge', (1.0, 0.75, 0.2), 0.25, **{'Metallic': 1.0})
M_KARA = None
M_TILE = glowmat('tile', gain=9.0, base=0.0, rough=0.4)          # xylophone bars (rosewood) - object colour = wood, flash amber
M_PLATE = glowmat('plate', gain=7.0, base=0.0, rough=0.3, metal=0.6)
M_BELL = glowmat('bell', gain=6.0, base=0.15, rough=0.25, metal=0.8)
M_STEEL = mat('steel', (0.5, 0.52, 0.56), 0.35, **{'Metallic': 0.9})
M_DARK = mat('dark', (0.04, 0.04, 0.05), 0.6)
M_ROOF = mat('roof', (0.16, 0.17, 0.2), 0.55)               # dark slate roof boards
M_ROOF2 = mat('roof2', (0.2, 0.13, 0.11), 0.6)
M_WALL = objcol('wall', 0.85)
M_BRICK = mat('brick', (0.42, 0.16, 0.1), 0.8)
M_WOOD = mat('wood', (0.45, 0.28, 0.14), 0.6)
M_RAIL = mat('rail', (0.65, 0.6, 0.5), 0.6)
M_OBJ = objcol('objcol', 0.5)
M_FX = fxmat('fx', 8.0)
M_TXT_W = emis('txtw', (1, 1, 1), 1.6); M_TXT_Y = emis('txty', (1, 0.82, 0.25), 2.0); M_TXT_B = emis('txtb', (0.45, 0.7, 1.0), 2.2)
M_TXT_G = emis('txtg', (0.45, 1.0, 0.45), 2.0)

# karakusa (唐草) furoshiki: green with swirly white pattern
M_KARA = bpy.data.materials.new('kara'); M_KARA.use_nodes = True; nt = M_KARA.node_tree; bs = nt.nodes['Principled BSDF']
bs.inputs['Roughness'].default_value = 0.75
tc = nt.nodes.new('ShaderNodeTexCoord'); wv = nt.nodes.new('ShaderNodeTexWave'); wv.wave_type = 'RINGS'; wv.inputs['Scale'].default_value = 6.0
wv.inputs['Distortion'].default_value = 9.0; wv.inputs['Detail'].default_value = 1.5
nt.links.new(tc.outputs['Object'], wv.inputs['Vector'])
rp = nt.nodes.new('ShaderNodeValToRGB'); rp.color_ramp.elements[0].color = (0.02, 0.32, 0.12, 1); rp.color_ramp.elements[1].color = (0.92, 0.95, 0.9, 1)
rp.color_ramp.elements[0].position = 0.72; rp.color_ramp.elements[1].position = 0.8
nt.links.new(wv.outputs['Fac'], rp.inputs['Fac']); nt.links.new(rp.outputs['Color'], bs.inputs['Base Color'])

# ---------------- balls ----------------
BALL = {}; FACE = {}
for n in ('thief', 'cop'):
    root = link(bpy.data.objects.new(n, None)); key_loc(root, POS[n])
    body = sphere((0, 0, 0), R, M_THIEF if n == 'thief' else M_COP, (32, 16), name=n + '_body'); body.parent = root
    sp = np.zeros(NF)
    for f in range(1, NF): sp[f] = sp[f - 1] + (POS[n][f, 0] - POS[n][f - 1, 0]) / R
    body.rotation_mode = 'XYZ'; key_series(body, 'rotation_euler', 1, sp)
    BALL[n] = root
    # eyes (do not roll), on the camera side
    eyes = []
    for sx in (-1, 1):
        d = np.array([0.36 * sx + 0.12, -0.86, 0.34]); d /= np.linalg.norm(d)
        e_ = sphere(tuple(d * R * 0.86), R * 0.3, M_EYE, (16, 8), name='eye'); e_.parent = root; e_.scale = (R * 0.3, R * 0.18, R * 0.34)
        pu = sphere((0, 0, 0), R * 0.15, M_PUPIL, (12, 6), name='pupil'); pu.parent = root
        eyes.append((d, pu))
    FACE[n] = eyes
    if n == 'thief':                      # green karakusa bundle tied on its back + knot
        bd = sphere((-R * 0.55, 0.05, R * 0.95), R * 0.62, M_KARA, (20, 10), name='bundle'); bd.parent = root; bd.scale = (R * 0.7, R * 0.6, R * 0.55)
        kn = sphere((R * 0.05, -0.02, R * 0.98), R * 0.18, M_KARA, (12, 6), name='knot'); kn.parent = root
        for sgn in (-1, 1):
            ear = sphere((R * 0.18, -0.02 + 0.06 * sgn, R * 1.05), R * 0.12, M_KARA, (10, 5)); ear.parent = root; ear.scale = (R * 0.16, R * 0.06, R * 0.1)
    else:                                 # police cap with badge and a siren lamp
        cp = rod((0, 0, R * 0.72), (0, 0, R * 1.08), R * 0.72, M_CAP, 24); cp.parent = root
        vz = rod((0, -R * 0.0, R * 0.72), (0, 0, R * 0.76), R * 0.88, M_CAP, 24); vz.parent = root
        bg = sphere((0, -R * 0.7, R * 0.92), R * 0.12, M_BADGE, (12, 6)); bg.parent = root; bg.scale = (R * 0.13, R * 0.03, R * 0.13)
        sl = sphere((0, 0, R * 1.14), R * 0.22, M_FX, (16, 8), name='siren'); sl.parent = root
        FACE['siren'] = sl
# pupils: look ahead (+x) normally, at the other ball during the stand-offs
STAND = [(EV['decks'][k]['T']['t0'] + 0.5 * (EV['decks'][k]['T']['t_edge'] - EV['decks'][k]['T']['t0']), EV['decks'][k]['T']['t_edge'] + 0.25)
         for k in EV['decks']]
STAND.append((EV['collide']['t'] - 0.6, EV['collide']['t'] + 1.0))
for n in ('thief', 'cop'):
    other = 'cop' if n == 'thief' else 'thief'
    for d, pu in FACE[n]:
        P = np.zeros((NF, 3))
        look = np.zeros((NF, 3))
        for f in range(NF):
            t = T[f]; v = POS[n][min(NF - 1, f + 2)] - POS[n][max(0, f - 2)]
            ahead = np.array([1.0, -0.3, -0.25 + 2.0 * float(np.clip(v[2], -0.15, 0.15))])
            w = 0.0
            for a, b in STAND:
                if a - 0.4 < t < b + 0.4: w = max(w, min(1, (t - a + 0.4) / 0.4, (b + 0.4 - t) / 0.4))
            if T[f] < 1.0: w = 0.0
            to = POS[other][f] - POS[n][f]; to = to / max(1e-6, np.linalg.norm(to))
            look[f] = (1 - w) * ahead / np.linalg.norm(ahead) + w * (to + np.array([0, -0.9, 0]))
        for k in range(3): look[:, k] = smooth(look[:, k], 0.06)
        for f in range(NF):
            l = look[f] - d * (look[f] @ d); l = l / max(1e-6, np.linalg.norm(l)) * R * 0.13
            P[f] = d * R * 0.98 + l
        key_loc(pu, P)
# siren lamp: red / blue flashes on every off-beat chord (and the whistle at the start)
PAH = [p[0] for p in E['pahs']]
sa = pulse_alpha(PAH, decay=2.5, length=8) + pulse_alpha([0.12, 0.42, 0.72], decay=3, length=9)
sl = FACE['siren']; key_series(sl, 'color', 3, np.clip(0.15 + sa, 0, 1.2), 1e-3)
cols = np.zeros((NF, 3)); par = 0
for f in range(NF):
    k = int(np.searchsorted(PAH, T[f] + 1e-6)); par = k % 2
    cols[f] = (1.0, 0.08, 0.05) if par else (0.1, 0.35, 1.0)
for k in range(3): key_series(sl, 'color', k, cols[:, k], 1e-3)

# ---------------- struck objects: one per hit ----------------
TILE_W = {'tile': 0.17, 'plate': 0.2, 'gut': 0.1, 'slat': 0.16, 'bell': 0.2, 'cbell': 0.2}
BARS = []
PITCHT = {round(l['t'], 4): l['p'] for l in LEAD}
def tile_col(n, kind, t):
    if n == 'thief':
        if kind in ('bell', 'cbell'): return (1.0, 0.72, 0.3)
        return (0.62, 0.3, 0.14)
    return (0.75, 0.38, 0.22)
for n, hs in HITS.items():
    for h in hs:
        k = h['kind']; t = h['t']
        if k == 'gbell': continue
        nrm = np.array(h['n'], float); nrm /= np.linalg.norm(nrm)
        c = np.array(h['p']) - R * nrm
        if k in ('bell', 'cbell'):
            o = obj(prim('cone', M_BELL, 24, 0.13, 0.05, 0.16), 'bell'); th = 0.08; ctr = c - nrm * th
            o.matrix_world = Matrix.Translation(Vector(ctr)) @ frame_from_n(nrm).to_4x4()
        else:
            w = h.get('w', TILE_W.get(k, 0.17)); depth = 0.8 if k in ('tile', 'plate', 'slat') else 0.22; TH = 0.06
            if k == 'plate': depth = 0.85
            th = TH / 2; ctr = c - nrm * th
            o = obj(prim('cube', M_PLATE if n == 'cop' else M_TILE), 'tile')
            o.matrix_world = Matrix.Translation(Vector(ctr)) @ frame_from_n(nrm).to_4x4() @ Matrix.Diagonal((w, depth, TH, 1))
        o.color = (*tile_col(n, k, t), 0)
        BARS.append(dict(o=o, c=ctr, n=nrm, t=t, th=th, kind=k, ball=n, contact=c, roll=h.get('roll', False)))
for B_ in BARS:
    o = B_['o']; fh = int(round(B_['t'] * FPS)); base = o.location.copy(); nv = Vector(B_['n'])
    col = tuple(o.color[:3]); flash = (1.0, 0.75, 0.3) if B_['ball'] == 'thief' else (0.35, 0.75, 1.0)
    o.color = (*col, 0); o.keyframe_insert('color', frame=max(0, fh - 1))
    for df, a, mix in ((0, 1.0, 1.0), (4, 0.45, 0.7), (18, 0.0, 0.0)):
        cc = tuple(col[i] * (1 - mix) + flash[i] * mix for i in range(3))
        o.color = (*cc, a); o.keyframe_insert('color', frame=fh + df)
    if not B_['roll']:
        for df, d in ((-1, 0), (1, -0.022), (4, 0.006), (8, 0)):
            o.location = base + nv * d; o.keyframe_insert('location', frame=max(0, fh + df))
# hit rings
NRING = 12
ring_loc = np.zeros((NRING, NF, 3)); ring_scl = np.full((NRING, NF), 1e-3); ring_a = np.zeros((NRING, NF)); ring_c = [(1, 1, 1)] * NRING
rk = 0
for B_ in sorted(BARS, key=lambda b: b['t']):
    r = rk % NRING; rk += 1; f0 = int(round(B_['t'] * FPS))
    for df in range(12):
        f = f0 + df
        if f >= NF: break
        u = df / 11; ring_loc[r, f] = B_['contact'] + np.array([0, -0.35, 0.02]); ring_scl[r, f] = 0.25 + 1.0 * (1 - (1 - u) ** 2)
        ring_a[r, f] = 0.8 * (1 - u) ** 1.6
    ring_c[r] = (1.0, 0.7, 0.3) if B_['ball'] == 'thief' else (0.35, 0.7, 1.0)
for r in range(NRING):
    bpy.ops.mesh.primitive_torus_add(major_radius=0.2, minor_radius=0.012, major_segments=32, minor_segments=5, rotation=(math.pi / 2, 0, 0))
    o = bpy.context.object; o.data.materials.append(M_FX); o.color = (*ring_c[r], 0)
    for f in range(1, NF):
        if ring_a[r, f] == 0: ring_loc[r, f] = ring_loc[r, f - 1]
    key_loc(o, ring_loc[r])
    for k in range(3): key_series(o, 'scale', k, ring_scl[r])
    key_series(o, 'color', 3, ring_a[r], 1e-3)

# ---------------- houses: roofs follow the struck tiles, gaps where the ball leaps ----------------
WIN = []          # windows: (x, y, z, w, h, delay, lit)
def roof_groups(n):
    hs = [h for h in HITS[n] if h['kind'] in ('tile', 'plate', 'bell')]
    groups = []; cur = []
    for h in hs:
        if cur:
            pv = cur[-1]; dx = h['p'][0] - pv['p'][0]; dz = h['p'][2] - pv['p'][2]
            if dx > 1.15 or dz < -0.42 or dz > 0.3 or h['t'] - pv['t'] > 0.95:
                groups.append(cur); cur = []
        cur.append(h)
    if cur: groups.append(cur)
    return groups
SPECIAL = []      # x ranges (per lane) occupied by gutter/deck structures
gu = EV['gutter']
SPECIAL.append(('thief', gu['p0'][0] - 0.2, gu['p1'][0] + 0.2))
for dk_ in EV['decks'].values():
    for ln, d in dk_.items(): SPECIAL.append(('thief' if ln == 'T' else 'cop', d['x0'] - 0.3, d['x1'] + 0.3))
HOUSES = []
WALLC = [(0.32, 0.27, 0.3), (0.28, 0.3, 0.36), (0.36, 0.3, 0.25), (0.26, 0.28, 0.27), (0.34, 0.26, 0.24)]
def house(x0, x1, ztop, y, depth, roofmat=M_ROOF, seed=0, eave=True):
    ztop = float(ztop); zb = ztop - 4.5
    hb = box(((x0 + x1) / 2, y, (ztop + zb) / 2 - 0.05), (x1 - x0, depth, ztop - zb - 0.1), M_WALL, name='house')
    hb.color = (*WALLC[seed % len(WALLC)], 1)
    if eave: box(((x0 + x1) / 2, y, ztop - 0.04), (x1 - x0 + 0.24, depth + 0.24, 0.08), roofmat, name='eave')
    HOUSES.append((x0, x1, ztop, y))
    # windows on the camera side (front face) - 2 rows
    nw = max(1, int((x1 - x0) / 0.85))
    for row in range(2):
        zc = ztop - 0.75 - row * 0.95
        for k in range(nw):
            xc = x0 + (k + 0.5) * (x1 - x0) / nw
            WIN.append((xc, y - depth / 2 - 0.012, zc, 0.36, 0.42, rng.uniform(0, 1), rng.random() < 0.35))
for n, y in (('thief', YT), ('cop', YC)):
    for gi, g in enumerate(roof_groups(n)):
        xs = np.array([h['p'][0] for h in g]); zs = np.array([h['p'][2] - R for h in g])
        x0, x1 = xs.min() - 0.35, xs.max() + 0.35
        # roof board under the tiles: polyline that stays under every tile
        X = np.arange(x0, x1 + 0.01, 0.15); Z = np.array([zs[np.abs(xs - x) < 0.5].min() if (np.abs(xs - x) < 0.5).any() else np.interp(x, xs, zs) for x in X]) - 0.09
        bm = bmesh.new(); vs = []
        for x, z in zip(X, Z):
            vs.append([bm.verts.new((x, y - 0.5, z)), bm.verts.new((x, y + 0.5, z)), bm.verts.new((x, y + 0.5, z - 0.12)), bm.verts.new((x, y - 0.5, z - 0.12))])
        for a, b in zip(vs, vs[1:]):
            for k in range(4): bm.faces.new((a[k], b[k], b[(k + 1) % 4], a[(k + 1) % 4]))
        bm.faces.new(vs[0][::-1]); bm.faces.new(vs[-1])
        me = bpy.data.meshes.new('roof'); bm.to_mesh(me); bm.free(); ro = link(bpy.data.objects.new('roof', me)); ro.data.materials.append(M_ROOF if n == 'thief' else M_ROOF2)
        # house body below (top = lowest roof point), with a sloped wall following the roof (stepped, 0.6 wide)
        for xa in np.arange(x0, x1 - 1e-6, 0.6):
            xb = min(x1, xa + 0.6); zt_ = float(np.interp([xa, xb], X, Z).min()) - 0.12
            box(((xa + xb) / 2, y, zt_ - 2.25), (xb - xa + 0.004, 1.0, 4.5), M_WALL, name='hb').color = (*WALLC[(gi + (n == 'cop')) % 5], 1)
        HOUSES.append((x0, x1, float(Z.min()), y))
        nw = max(1, int((x1 - x0) / 0.8))
        for row in range(3):
            for k in range(nw):
                xc = x0 + (k + 0.5) * (x1 - x0) / nw; zc = float(np.interp(xc, X, Z)) - 0.75 - row * 0.95
                WIN.append((xc, y - 0.512, zc, 0.34, 0.42, rng.uniform(0, 1), rng.random() < 0.28))
        # chimney under every chimney bell
        for h in g:
            if h['kind'] == 'bell':
                c = np.array(h['p']) - R * np.array(h['n'])
                zr = float(np.interp(c[0], X, Z))
                box((c[0], y, (c[2] - 0.17 + zr) / 2), (0.24, 0.3, c[2] - 0.17 - zr), M_BRICK, name='chimney')

# start chimney (thief) and the police lookout tower (cop)
ts = EV['thief_start']; ps = np.array(ts['p'])
zr0 = HITS['thief'][0]['p'][2] - R - 0.2
box((ps[0], YT, (ps[2] - R - 0.05 + zr0) / 2), (0.36, 0.42, ps[2] - R - 0.05 - zr0), M_BRICK, name='chim0')
lid = link(bpy.data.objects.new('lidpiv', None)); lid.location = (ps[0] + 0.18, YT, ps[2] - R - 0.03)
lp = box((-0.2, 0, 0), (0.42, 0.48, 0.05), M_DARK, name='lid'); lp.parent = lid
la = np.zeros(NF)
for f in range(NF):
    u = (T[f] - ts['t_rel'] + 0.03) / 0.18
    if u > 0: la[f] = min(1.0, u) ** 2 * 1.7
lid.rotation_mode = 'XYZ'; key_series(lid, 'rotation_euler', 1, -la)
cs = EV['cop_start']; pc = np.array(cs['p'])
zc0 = HITS['cop'][0]['p'][2] - R - 0.2
for dx in (-0.25, 0.25):
    for dy in (-0.25, 0.25): rod((pc[0] + dx, YC + dy, pc[2] - R - 0.08), (pc[0] + dx, YC + dy, zc0 - 3), 0.025, M_STEEL, 6)
box((pc[0], YC, pc[2] + 0.55), (0.62, 0.62, 0.06), M_DARK, name='towerroof')
hat = link(bpy.data.objects.new('hatchpiv', None)); hat.location = (pc[0] + 0.3, YC, pc[2] - R - 0.04)
hp = box((-0.3, 0, 0), (0.6, 0.6, 0.06), M_STEEL, name='hatch'); hp.parent = hat
ha = np.zeros(NF)
for f in range(NF):
    u = (T[f] - cs['t_rel'] + 0.03) / 0.15
    if u > 0: ha[f] = min(1.0, u) ** 2 * 1.6
hat.rotation_mode = 'XYZ'; key_series(hat, 'rotation_euler', 1, ha)
lamp = sphere((pc[0], YC, pc[2] + 0.66), 0.09, M_FX); lamp.color = (1, 0.1, 0.05, 1)
key_series(lamp, 'color', 3, np.clip(0.3 + 0.7 * (np.sin(T * 12) > 0) * (T < 3.0), 0, 1), 1e-3)
# name labels at the start
lab = [(text('どろぼう', (ps[0] + 0.05, YT - 0.4, ps[2] + 0.55), 0.26, M_TXT_G), BALL['thief'], (0.05, -0.4, 0.5)),
       (text('おまわりさん', (pc[0], YC - 0.4, pc[2] + 0.95), 0.24, M_TXT_B), BALL['cop'], (0, -0.4, 0.58))]
for tx, b_, off in lab:
    tx.parent = None
    P = np.array([POS['thief' if b_.name == 'thief' else 'cop'][f] + np.array(off) for f in range(NF)])
    key_loc(tx, P)
    s_ = np.clip(np.minimum((T - 0.1) / 0.2, (3.6 - T) / 0.4), 0.001, 1)
    for k in range(3): key_series(tx, 'scale', k, s_)

# ---------------- eave gutter (rolling, accelerating) ----------------
p0 = np.array(gu['p0']); p1 = np.array(gu['p1']); uG = np.array(gu['u'])
nG = np.array([-uG[2], 0, uG[0]])
a0 = p0 - nG * (R + 0.03) - uG * 0.3; a1 = p1 - nG * (R + 0.03) + uG * 0.15
L = np.linalg.norm(a1 - a0)
gpiv = link(bpy.data.objects.new('gutter', None)); gpiv.matrix_world = Matrix.Translation(Vector((a0 + a1) / 2)) @ frame_from_n(nG).to_4x4()
bpy.ops.mesh.primitive_cylinder_add(radius=0.2, depth=L, vertices=24, rotation=(0, math.pi / 2, 0))
gc = bpy.context.object; bm = bmesh.new(); bm.from_mesh(gc.data)
bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z > 0.02], context='VERTS'); bm.to_mesh(gc.data); bm.free()
gc.modifiers.new('s', 'SOLIDIFY').thickness = 0.02; gc.data.materials.append(M_STEEL); gc.parent = gpiv; gc.location = (0, 0, 0.17)
for s_ in np.linspace(0, L, 6):
    q = a0 + uG * s_
    rod(q, q + np.array([0, 0.45, 0.25]), 0.012, M_STEEL, 6)
# downspout at the end
rod(a1 + np.array([0.05, 0, -0.05]), a1 + np.array([0.05, 0, -3.5]), 0.07, M_STEEL, 12)
# the house the gutter hangs on
zg = max(p0[2], p1[2]) + 0.15
house(p0[0] - 0.2, p1[0] + 0.2, float(min(p0[2], p1[2]) + 0.55), YT + 0.25, 0.5, seed=3)

# ---------------- slatted rooftop decks (物干し台) with laundry ----------------
M_CLOTH = objcol('cloth', 0.9)
for name, dk_ in EV['decks'].items():
    for ln, d in dk_.items():
        y = d['y']; x0, x1, z = d['x0'], d['x1'], d['z'] - R
        box(((x0 + x1) / 2 - 0.15, y, z - 0.07), (x1 - x0 + 0.3, 0.95, 0.06), M_WOOD, name='deck')
        for xx in np.arange(x0 - 0.2, x1 + 0.01, 0.9):
            rod((xx, y - 0.45, z - 0.1), (xx, y - 0.45, z - 1.8), 0.03, M_WOOD, 6); rod((xx, y + 0.45, z - 0.1), (xx, y + 0.45, z - 1.8), 0.03, M_WOOD, 6)
        # roof under the deck
        house(x0 - 0.5, x1 + 0.05, z - 1.8, y, 1.0, roofmat=M_ROOF if ln == 'T' else M_ROOF2, seed=len(HOUSES))
        if ln == 'C':       # clothesline with laundry behind the cop's deck
            for xx in (x0 - 0.2, x1):
                rod((xx, y + 0.55, z - 0.1), (xx, y + 0.55, z + 1.25), 0.025, M_RAIL, 6)
            rod((x0 - 0.2, y + 0.55, z + 1.2), (x1, y + 0.55, z + 1.2), 0.008, M_RAIL, 4)
            for k, xx in enumerate(np.arange(x0 + 0.2, x1 - 0.2, 0.75)):
                cl = box((xx, y + 0.56, z + 0.92), (0.42, 0.02, 0.5), M_CLOTH, name='cloth')
                cl.color = (*[(0.9, 0.9, 0.85), (0.85, 0.35, 0.3), (0.35, 0.5, 0.85), (0.95, 0.85, 0.4)][k % 4], 1)
        # edge stop rail (the ball creeps over the edge: no rail on the lane, small posts beside it)
        rod((x1 + 0.02, y - 0.47, z), (x1 + 0.02, y - 0.47, z + 0.35), 0.02, M_RAIL, 6)
        rod((x1 + 0.02, y + 0.47, z), (x1 + 0.02, y + 0.47, z + 0.35), 0.02, M_RAIL, 6)

# ---------------- finale: chord bells on stands, the great bell in its tower ----------------
gb = EV['gbell']; XG = gb['x']; ZT = gb['z_top']
for B_ in BARS:
    if B_['kind'] == 'cbell':
        c = B_['c']; rod((c[0], c[1], c[2] - 0.08), (c[0], c[1], ZT - 2.4), 0.025, M_STEEL, 8)
# great bell: crown plate at ZT, bell body below, tower frame around
BELLR = 0.75
bell_piv = link(bpy.data.objects.new('gbell', None)); bell_piv.location = (XG, 0, ZT)
bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=BELLR, radius2=0.42, depth=1.1, location=(0, 0, 0))
gbo = bpy.context.object; gbo.parent = bell_piv; gbo.location = (0, 0, -0.62)
for p_ in gbo.data.polygons: p_.use_smooth = True
gbo.data.materials.append(M_BELL); gbo.color = (1.0, 0.7, 0.3, 0)
crown = rod((0, 0, -0.08), (0, 0, 0), 0.5, M_BELL, 32); crown.parent = bell_piv; crown.color = (1.0, 0.75, 0.35, 0)
lip = rod((0, 0, -1.2), (0, 0, -1.12), BELLR + 0.05, M_BELL, 48); lip.parent = bell_piv; lip.color = (1.0, 0.7, 0.3, 0)
tg_ = gb['t']
for o in (gbo, crown, lip): key_series(o, 'color', 3, pulse_alpha([tg_], decay=14, length=60) * 0.45, 1e-3)
sw = np.zeros(NF)
for f in range(NF):
    u = T[f] - tg_
    if u > 0: sw[f] = 0.06 * math.sin(u * 5.0) * math.exp(-u * 0.8)
bell_piv.rotation_mode = 'XYZ'; key_series(bell_piv, 'rotation_euler', 1, sw)
for dx in (-1.25, 1.25):
    box((XG + dx, 0, ZT - 2.2), (0.22, 1.9, 5.6), M_BRICK, name='tower')
box((XG, 0, ZT + 0.85), (2.9, 2.0, 0.18), M_BRICK, name='towertop')
bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=2.1, radius2=0.0, depth=1.1, location=(XG, 0, ZT + 1.5), rotation=(0, 0, math.pi / 4))
bpy.context.object.data.materials.append(M_ROOF2)
rod((XG, 0, ZT + 2.05), (XG, 0, ZT + 2.6), 0.02, M_BADGE, 6)
box((XG, 0, ZT - 5.0), (2.7, 1.9, 0.3), M_BRICK, name='towerbase')
# shock rings from the gong + collision spark
for k in range(4):
    bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=0.02, major_segments=48, minor_segments=5, rotation=(math.pi / 2, 0, 0))
    o = bpy.context.object; o.data.materials.append(M_FX); o.color = (1.0, 0.8, 0.4, 0); o.location = (XG, -0.9, ZT - 0.6)
    ts_ = tg_ + 0.18 * k; sc_ = np.full(NF, 0.01); a_ = np.zeros(NF)
    for f in range(NF):
        u = (T[f] - ts_) / 1.4
        if 0 <= u <= 1: sc_[f] = 0.6 + 5.0 * (1 - (1 - u) ** 2); a_[f] = 0.9 * (1 - u) ** 1.5
    for c_ in range(3): key_series(o, 'scale', c_, sc_)
    key_series(o, 'color', 3, a_, 1e-3)
cl_ = EV['collide']
spk = sphere(tuple(np.array(cl_['p']) + np.array([0, -0.2, 0])), 0.2, M_FX); spk.color = (1, 0.95, 0.6, 0)
fa = pulse_alpha([cl_['t']], decay=3, length=10); key_series(spk, 'color', 3, fa, 1e-3)
for c_ in range(3): key_series(spk, 'scale', c_, 0.05 + 0.4 * fa)
# fireworks over the harbour after the gong (visual only)
FW_C = [(1, 0.4, 0.3), (0.4, 0.8, 1), (1, 0.85, 0.3), (0.6, 1, 0.5), (1, 0.5, 0.9)]
for k in range(5):
    t0 = tg_ + 0.35 + 0.42 * k; c0 = np.array([XG - 3.2 + 1.6 * k + rng.uniform(-0.3, 0.3), 9.0, ZT + 4.0 + rng.uniform(-0.6, 0.9)])
    for j in range(28):
        th, ph = rng.uniform(0, 2 * math.pi), math.acos(rng.uniform(-1, 1))
        d = np.array([math.sin(ph) * math.cos(th), math.sin(ph) * math.sin(th) * 0.3, math.cos(ph)])
        o = sphere((0, 0, 0), 0.06, M_FX, (8, 4)); o.color = (*FW_C[k], 0)
        P = np.zeros((NF, 3)); A = np.zeros(NF)
        for f in range(NF):
            u = T[f] - t0
            if u < 0: P[f] = c0; continue
            r_ = 2.4 * (1 - math.exp(-u * 2.5)); P[f] = c0 + d * r_ + np.array([0, 0, -0.35 * u * u]); A[f] = max(0, 1 - u / 1.6) * 1.5
        key_loc(o, P); key_series(o, 'color', 3, A, 1e-3)

# ---------------- police searchlight: snaps on in the first frames and follows the thief all night ----------------
SRC = POS['thief'] + np.array([-7.0, -1.6, -4.5])
for k in range(3): SRC[:, k] = smooth(SRC[:, k], 0.8)
beam = link(bpy.data.objects.new('beampiv', None)); key_loc(beam, SRC)
bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=0.03, radius2=0.8, depth=1.0, location=(0, 0, 0.5))
bc = bpy.context.object; bc.parent = beam; bc.data.materials.append(M_FX); bc.color = (1.0, 0.95, 0.75, 0)
tgt_ = link(bpy.data.objects.new('beamtgt', None)); TP = POS['thief'].copy(); key_loc(tgt_, TP)
cn = beam.constraints.new('TRACK_TO'); cn.target = tgt_; cn.track_axis = 'TRACK_Z'; cn.up_axis = 'UP_Y'
dist = np.linalg.norm(TP - SRC, axis=1)
on = np.clip((T - 0.08) / 0.05, 0, 1)
key_series(bc, 'color', 3, 0.03 * on + 0.05 * pulse_alpha([0.1], decay=4, length=14), 1e-4)
key_series(beam, 'scale', 2, np.maximum(0.01, dist * (on > 0) * 1.02), 1e-3)
key_series(beam, 'scale', 0, np.maximum(0.01, on), 1e-3); key_series(beam, 'scale', 1, np.maximum(0.01, on), 1e-3)
spot = bpy.data.lights.new('spot', 'SPOT'); spot.spot_size = math.radians(7); spot.spot_blend = 0.5; spot.color = (1, 0.95, 0.82)
so = link(bpy.data.objects.new('spot', spot)); key_loc(so, SRC)
cn = so.constraints.new('TRACK_TO'); cn.target = tgt_; cn.track_axis = 'TRACK_NEGATIVE_Z'; cn.up_axis = 'UP_Y'
for f in range(NF):
    if 0 < f < NF - 1 and abs(on[f] - on[f - 1]) < 1e-4 and abs(on[f] - on[f + 1]) < 1e-4: continue
    spot.energy = float(5000 * on[f]); spot.keyframe_insert('energy', frame=f)
# the siren lamp lights the surroundings (red / blue) on every off-beat chord
sir = bpy.data.lights.new('siren', 'POINT'); sir.shadow_soft_size = 0.05
sio = link(bpy.data.objects.new('sirenlight', sir)); sio.parent = BALL['cop']; sio.location = (0, -0.1, R * 1.3)
for f in range(NF):
    if 0 < f < NF - 1 and abs(sa[f] - sa[f - 1]) < 1e-3 and abs(sa[f] - sa[f + 1]) < 1e-3 and (cols[f] == cols[f - 1]).all(): continue
    sir.energy = float(3 + 45 * min(1, sa[f])); sir.keyframe_insert('energy', frame=f)
    sir.color = tuple(cols[f]); sir.keyframe_insert('color', frame=f)
ex = text('!', (0, 0, 0), 0.45, M_TXT_Y); ex.parent = BALL['thief']; ex.location = (0.05, -0.3, 0.5)
es = np.clip(np.minimum((T - 0.15) / 0.12, (1.0 - T) / 0.2), 0.001, 1.2)
for k in range(3): key_series(ex, 'scale', k, es)

# ---------------- sky, moon, stars, background town ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes['Background']; bg.inputs['Color'].default_value = (0.03, 0.05, 0.12, 1); bg.inputs['Strength'].default_value = 0.6
moon = bpy.data.lights.new('moon', 'SUN'); moon.energy = 0.9; moon.angle = 0.05; moon.color = (0.7, 0.8, 1.0)
mo = link(bpy.data.objects.new('moonlight', moon)); mo.rotation_euler = (math.radians(55), math.radians(30), math.radians(-25))
rim = bpy.data.lights.new('rim', 'SUN'); rim.energy = 2.6; rim.angle = 0.1; rim.color = (0.75, 0.85, 1.0)
ro_ = link(bpy.data.objects.new('rim', rim)); ro_.rotation_euler = (math.radians(-60), math.radians(0), math.radians(10))   # from behind (+y) and above
# camera rig (needed for the sky that rides along)
def run_mid(t, f):
    a, b = POS['thief'][f], POS['cop'][f]
    return 0.62 * a + 0.38 * b
FOC = np.zeros((NF, 3)); WID = np.zeros(NF)
t_end = gb['t']
for f in range(NF):
    t = T[f]; m = run_mid(t, f)
    FOC[f] = m + np.array([0.45, 0, -0.05]); WID[f] = max(3.7, abs(POS['thief'][f][0] - POS['cop'][f][0]) + 2.6)
    if t < 1.2: FOC[f] = (POS['thief'][f] + POS['cop'][f]) / 2 + np.array([0.3, 0, -0.3]); WID[f] = 4.6
    for nm in EV['decks']:                           # stand-offs: a bit closer on the two balls
        d = EV['decks'][nm]['T']
        if d['t0'] + 1.0 < t < d['t_edge'] + 0.3: WID[f] = 3.7
    if t > t_end - 1.2:                              # the great bell: pull back to see the town light up
        u = min(1, (t - t_end + 1.2) / 1.2); u2 = min(1, max(0, (t - t_end - 0.6) / 2.2))
        target = np.array([XG - 0.2, 0, ZT + 0.05 + 1.2 * u2]); FOC[f] = (1 - u) * FOC[f] + u * target
        WID[f] = (1 - u) * WID[f] + u * (4.0 + 3.4 * (u2 * u2 * (3 - 2 * u2)))
WID = np.clip(WID, 3.6, 11.0)
FOC[:, 0] = smooth(FOC[:, 0], 0.3); FOC[:, 2] = smooth(FOC[:, 2], 0.45); WID = smooth(WID, 0.5)
LENS = 35.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 500
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TPc = np.zeros((NF, 3))
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24
    CP[f] = (FOC[f, 0] - 0.05 * D, -D, FOC[f, 2] + 0.1 * D)
    TPc[f] = (FOC[f, 0], 0.0, FOC[f, 2])
key_loc(cam, CP); key_loc(tgt, TPc)
# sky dome riding with the camera: gradient + stars + moon
skyp = link(bpy.data.objects.new('skypiv', None)); key_loc(skyp, np.stack([CP[:, 0], np.zeros(NF), CP[:, 2]], 1))
SKY = bpy.data.materials.new('sky'); SKY.use_nodes = True; nt = SKY.node_tree; nt.nodes.clear()
tc = nt.nodes.new('ShaderNodeTexCoord'); sepz = nt.nodes.new('ShaderNodeSeparateXYZ'); rp = nt.nodes.new('ShaderNodeValToRGB')
rp.color_ramp.elements[0].color = (0.3, 0.2, 0.42, 1); rp.color_ramp.elements[1].color = (0.02, 0.03, 0.12, 1)
e_ = rp.color_ramp.elements.new(0.42); e_.color = (0.08, 0.13, 0.34, 1)
mp = nt.nodes.new('ShaderNodeMapRange'); mp.inputs['From Min'].default_value = -30; mp.inputs['From Max'].default_value = 45
em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Strength'].default_value = 1.0; out = nt.nodes.new('ShaderNodeOutputMaterial')
nt.links.new(tc.outputs['Object'], sepz.inputs[0]); nt.links.new(sepz.outputs['Z'], mp.inputs['Value']); nt.links.new(mp.outputs[0], rp.inputs['Fac'])
nt.links.new(rp.outputs['Color'], em.inputs['Color']); nt.links.new(em.outputs[0], out.inputs[0])
skyo = obj(prim('plane', SKY), 'sky'); skyo.parent = skyp; skyo.location = (0, 120, 0); skyo.rotation_euler = (math.pi / 2, 0, 0); skyo.scale = (260, 160, 1)
M_STAR = emis('star', (1, 0.97, 0.9), 6.0)
bm = bmesh.new()
for k in range(500):
    x, z = rng.uniform(-110, 110), rng.uniform(-20, 75); s_ = rng.uniform(0.08, 0.22)
    vs = [bm.verts.new((x - s_, 0, z)), bm.verts.new((x, 0, z - s_)), bm.verts.new((x + s_, 0, z)), bm.verts.new((x, 0, z + s_))]
    bm.faces.new(vs)
me = bpy.data.meshes.new('stars'); bm.to_mesh(me); bm.free(); st_ = link(bpy.data.objects.new('stars', me)); st_.data.materials.append(M_STAR)
st_.parent = skyp; st_.location = (0, 118, 0)
M_MOON = emis('moon', (1.0, 0.95, 0.78), 5.0)
mn = sphere((0, 0, 0), 4.2, M_MOON, (48, 24), name='moon'); mn.parent = skyp; mn.location = (11, 110, 13)
hal = obj(prim('plane', None), 'halo'); hal.parent = skyp; hal.location = (11, 112, 13); hal.rotation_euler = (math.pi / 2, 0, 0); hal.scale = (34, 34, 1)
HAL = bpy.data.materials.new('halo'); HAL.use_nodes = True; nt = HAL.node_tree; nt.nodes.clear()
tc = nt.nodes.new('ShaderNodeTexCoord'); gr = nt.nodes.new('ShaderNodeTexGradient'); gr.gradient_type = 'SPHERICAL'
mpp = nt.nodes.new('ShaderNodeMapping'); mpp.inputs['Scale'].default_value = (2, 2, 2)
nt.links.new(tc.outputs['Object'], mpp.inputs['Vector']); nt.links.new(mpp.outputs[0], gr.inputs['Vector'])
em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Color'].default_value = (0.7, 0.75, 1.0, 1)
mu = nt.nodes.new('ShaderNodeMath'); mu.operation = 'POWER'; mu.inputs[1].default_value = 2.5
nt.links.new(gr.outputs['Fac'], mu.inputs[0]); nt.links.new(mu.outputs[0], em.inputs['Strength'])
trn = nt.nodes.new('ShaderNodeBsdfTransparent'); add = nt.nodes.new('ShaderNodeAddShader'); out = nt.nodes.new('ShaderNodeOutputMaterial')
nt.links.new(em.outputs[0], add.inputs[0]); nt.links.new(trn.outputs[0], add.inputs[1]); nt.links.new(add.outputs[0], out.inputs[0])
hal.data.materials.append(HAL)
# background town on the hill: layered rows of houses, each with windows
xsP = POS['thief'][:, 0]; zsP = POS['thief'][:, 2]
o_ = np.argsort(xsP); xsP, zsP = xsP[o_], zsP[o_]
def path_z(x): return float(np.interp(x, xsP, zsP))
XMIN, XMAX = -14.0, XG + 18
M_BGW = objcol('bgwall', 0.9)
for layer, (yb, dz, sh, hmin, hmax) in enumerate(((3.2, -2.7, 0.55, 0.8, 2.3), (7.5, -3.6, 0.42, 1.0, 2.6), (14.0, -5.0, 0.3, 1.0, 3.0), (24.0, -6.8, 0.2, 1.0, 3.6))):
    x = XMIN
    while x < XMAX:
        wdt = rng.uniform(1.0, 2.4); h = rng.uniform(hmin, hmax); zb = path_z(x + wdt / 2) + dz
        if layer == 0 and any(x0 - 0.5 < x + wdt / 2 < x1 + 0.5 for x0, x1, _, _ in HOUSES): x += wdt + 0.2; continue
        c = (sh * 0.22, sh * 0.24, sh * 0.36)
        b_ = box((x + wdt / 2, yb + layer * 0.8, zb + h / 2 - 3.5), (wdt, 1.4, h + 7.0), M_BGW, name='bg'); b_.color = (*c, 1)
        if rng.random() < 0.5:        # pitched roof
            bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=wdt * 0.72, radius2=0.0, depth=0.6 + 0.2 * rng.random(), location=(x + wdt / 2, yb + layer * 0.8, zb + h + 0.3), rotation=(0, 0, math.pi / 4))
            o = bpy.context.object; o.scale = (1, 0.55, 1); o.data.materials.append(M_BGW); o.color = (c[0] * 0.7, c[1] * 0.7, c[2] * 0.8, 1)
        nwx = max(1, int(wdt / 0.55)); nwz = max(1, int(h / 0.75))
        for i in range(nwx):
            for j in range(nwz):
                if rng.random() < 0.75:
                    WIN.append((x + (i + 0.5) * wdt / nwx, yb + layer * 0.8 - 0.71, zb + h - 0.45 - j * 0.75, 0.24, 0.3, rng.uniform(0, 1), rng.random() < 0.25))
        x += wdt + rng.uniform(0.05, 0.5)
# all windows in one mesh: attribute 'delay' = when it lights up in the final wave, 'lit' = lit from the start
bm = bmesh.new(); dl = bm.faces.layers.float.new('delay'); lt = bm.faces.layers.float.new('lit'); hu = bm.faces.layers.float.new('hue')
for (x, y, z, wd, ht, r_, lit) in WIN:
    vs = [bm.verts.new((x - wd / 2, y, z - ht / 2)), bm.verts.new((x + wd / 2, y, z - ht / 2)), bm.verts.new((x + wd / 2, y, z + ht / 2)), bm.verts.new((x - wd / 2, y, z + ht / 2))]
    fc = bm.faces.new(vs); fc[dl] = 0.35 + abs(x - XG) * 0.07 + 0.25 * r_; fc[lt] = 1.0 if lit else 0.0; fc[hu] = rng.random()
me = bpy.data.meshes.new('windows'); bm.to_mesh(me); bm.free(); wo = link(bpy.data.objects.new('windows', me))
WM = bpy.data.materials.new('win'); WM.use_nodes = True; nt = WM.node_tree; nt.nodes.clear()
at_d = nt.nodes.new('ShaderNodeAttribute'); at_d.attribute_name = 'delay'; at_l = nt.nodes.new('ShaderNodeAttribute'); at_l.attribute_name = 'lit'
at_h = nt.nodes.new('ShaderNodeAttribute'); at_h.attribute_name = 'hue'
tv = nt.nodes.new('ShaderNodeValue'); tv.name = 'TIME'
sub = nt.nodes.new('ShaderNodeMath'); sub.operation = 'SUBTRACT'; nt.links.new(tv.outputs[0], sub.inputs[0]); nt.links.new(at_d.outputs['Fac'], sub.inputs[1])
gt = nt.nodes.new('ShaderNodeMath'); gt.operation = 'GREATER_THAN'; gt.inputs[1].default_value = 0.0; nt.links.new(sub.outputs[0], gt.inputs[0])
mx_ = nt.nodes.new('ShaderNodeMath'); mx_.operation = 'MAXIMUM'; nt.links.new(gt.outputs[0], mx_.inputs[0]); nt.links.new(at_l.outputs['Fac'], mx_.inputs[1])
st = nt.nodes.new('ShaderNodeMath'); st.operation = 'MULTIPLY_ADD'; st.inputs[1].default_value = 3.2; st.inputs[2].default_value = 0.06
nt.links.new(mx_.outputs[0], st.inputs[0])
rpw = nt.nodes.new('ShaderNodeValToRGB'); rpw.color_ramp.elements[0].color = (1.0, 0.62, 0.25, 1); rpw.color_ramp.elements[1].color = (1.0, 0.88, 0.6, 1)
nt.links.new(at_h.outputs['Fac'], rpw.inputs['Fac'])
emw = nt.nodes.new('ShaderNodeEmission'); nt.links.new(rpw.outputs['Color'], emw.inputs['Color']); nt.links.new(st.outputs[0], emw.inputs['Strength'])
ow = nt.nodes.new('ShaderNodeOutputMaterial'); nt.links.new(emw.outputs[0], ow.inputs[0])
wo.data.materials.append(WM)
tvals = np.where(T < tg_, -100.0, T - tg_)
key_value(tv.outputs[0], tvals, 1e-3)

# ---------------- end text (comment bait) ----------------
hud = link(bpy.data.objects.new('hud', None)); hud.parent = cam; hud.location = (0, 0, -3.0)
HH = 3.0 * 24 / LENS
q_ = text('どっちが勝った？', (0, HH * 0.28, 0), 0.125, M_TXT_W, rot=(0, 0, 0), ext=0.0); q_.parent = hud
qs = np.clip((T - (tg_ + 1.3)) / 0.35, 0.001, 1.0); qs = np.where(qs < 1, qs * (1 + 0.25 * np.sin(qs * math.pi)), 1)
for k in range(3): key_series(q_, 'scale', k, qs)
for o in bpy.data.objects:
    if o.parent is hud: o.visible_shadow = False

# ---------------- sanity checks ----------------
def check():
    info = {}
    gaps = []
    for B_ in BARS:
        n = B_['ball']; f = int(round(B_['t'] * FPS)); p = POS[n][min(NF - 1, f)]
        gaps.append(np.linalg.norm(p - (B_['contact'] + R * B_['n'])))
    info['contact_gap_at_hit_frame_max'] = round(max(gaps), 3); info['contact_gap_mean'] = round(float(np.mean(gaps)), 3)
    info['bars'] = len(BARS); info['objects'] = len(bpy.data.objects); info['NF'] = NF; info['windows'] = len(WIN)
    return info

if MODE in ('check', 'test') and (MODE == 'check' or os.environ.get('CHECK')):
    print(check())
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
