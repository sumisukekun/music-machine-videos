"""Beethoven 5 - 「運命の扉」 (The Door of Fate): side view of a cut-away stone castle at night.
Every melody note = a ball striking a brass key / bell / bowl / the door plates / the gong (planned in layout.py:
constant gravity, bounces never add speed). Sleeping balls are dark; a ball glows once it has been woken (first hit).
Every bowl landing lights a window - the castle wakes up. Dawn comes with the horn call; at the end the door rises,
light floods out and the castle's balls pour out to the visitor.
Accompaniment: the wall torches flare on every bass note, the clockwork gears tick on the bass notes.
python3 scene.py -- check | test t1 t2 ... | range f0 f1"""
import os, sys, json, math, time
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json')); NOTES = json.load(open(f'{HERE}/notes.json'))
LAY = json.load(open(f'{HERE}/layout.json'))
FPS = 30; FINAL = E['final']; NF = int((FINAL + 3.8) * FPS); T = np.arange(NF) / FPS
G = LAY['G']; R = LAY['R']; EV = LAY['ev']; KD = LAY['keydim']; DOOR = LAY['door']
TN = np.array([l[0] for l in LEAD])
rng = np.random.default_rng(5)

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


def glowmat(name, gain=8.0, base=0.0, rough=0.3, metal=0.0, coat=0.5):
    """object colour = base colour; object alpha drives emission (hit flash / awake glow)"""
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


def winmat(name):
    """window glass: dark blue-black glass when off, warm glow (object colour) as object alpha rises"""
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    oi = nt.nodes.new('ShaderNodeObjectInfo'); em = nt.nodes.new('ShaderNodeEmission'); gl = nt.nodes.new('ShaderNodeBsdfPrincipled')
    gl.inputs['Base Color'].default_value = (0.02, 0.03, 0.06, 1); gl.inputs['Roughness'].default_value = 0.15
    mx = nt.nodes.new('ShaderNodeMixShader'); mu = nt.nodes.new('ShaderNodeMath'); mu.operation = 'MULTIPLY'; mu.inputs[1].default_value = 7.0
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(oi.outputs['Color'], em.inputs['Color']); nt.links.new(oi.outputs['Alpha'], mu.inputs[0]); nt.links.new(mu.outputs[0], em.inputs['Strength'])
    nt.links.new(oi.outputs['Alpha'], mx.inputs['Fac']); nt.links.new(gl.outputs[0], mx.inputs[1]); nt.links.new(em.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs[0]); return m


def objcol(name, rough=0.45, metal=0.0):
    m = mat(name, (1, 1, 1), rough, **{'Metallic': metal}); nt = m.node_tree
    oi = nt.nodes.new('ShaderNodeObjectInfo'); nt.links.new(oi.outputs['Color'], nt.nodes['Principled BSDF'].inputs['Base Color'])
    return m


def stonemat(name, c1, c2, scale=(2.0, 4.0), mortar=(0.05, 0.05, 0.06), rough=0.85, bump=0.35):
    """procedural stone blocks (object coordinates)"""
    m = mat(name, c1, rough); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    tc = nt.nodes.new('ShaderNodeTexCoord'); mp = nt.nodes.new('ShaderNodeMapping'); mp.inputs['Scale'].default_value = (scale[0], scale[0], scale[1])
    br = nt.nodes.new('ShaderNodeTexBrick'); br.inputs['Color1'].default_value = (*c1, 1); br.inputs['Color2'].default_value = (*c2, 1)
    br.inputs['Mortar'].default_value = (*mortar, 1); br.inputs['Scale'].default_value = 1.0; br.inputs['Mortar Size'].default_value = 0.025
    br.offset = 0.5
    nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 9.0
    mix = nt.nodes.new('ShaderNodeMixRGB'); mix.blend_type = 'MULTIPLY'; mix.inputs['Fac'].default_value = 0.35
    # stones in the X-Z plane: swizzle so that brick rows run along x, courses along z
    sep = nt.nodes.new('ShaderNodeSeparateXYZ'); com = nt.nodes.new('ShaderNodeCombineXYZ')
    nt.links.new(tc.outputs['Object'], mp.inputs['Vector']); nt.links.new(mp.outputs['Vector'], sep.inputs[0])
    nt.links.new(sep.outputs['X'], com.inputs['X']); nt.links.new(sep.outputs['Z'], com.inputs['Y']); nt.links.new(sep.outputs['Y'], com.inputs['Z'])
    nt.links.new(com.outputs[0], br.inputs['Vector']); nt.links.new(tc.outputs['Object'], nz.inputs['Vector'])
    nt.links.new(br.outputs['Color'], mix.inputs['Color1']); nt.links.new(nz.outputs['Color'], mix.inputs['Color2'])
    nt.links.new(mix.outputs[0], b.inputs['Base Color'])
    bm_ = nt.nodes.new('ShaderNodeBump'); bm_.inputs['Strength'].default_value = bump
    nt.links.new(br.outputs['Fac'], bm_.inputs['Height']); nt.links.new(bm_.outputs['Normal'], b.inputs['Normal'])
    return m


def key_series(obj, path, idx, vals, eps=1e-5):
    n = len(vals)
    for f in range(n):
        v = vals[f]
        if 0 < f < n - 1 and abs(v - vals[f - 1]) < eps and abs(v - vals[f + 1]) < eps: continue
        getattr(obj, path)[idx] = float(v); obj.keyframe_insert(path, index=idx, frame=f)


def key_loc(o, P):
    for k in range(3): key_series(o, 'location', k, P[:, k])


def key_node(sock, vals, eps=1e-3):
    for f in range(len(vals)):
        if 0 < f < len(vals) - 1 and abs(vals[f] - vals[f - 1]) < eps and abs(vals[f] - vals[f + 1]) < eps: continue
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
    elif kind == 'bowl':
        bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=1.0)
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z > 0.05], context='VERTS')
    elif kind == 'bell':                                # bell profile revolved (mouth at z = 0, crown at z = 1)
        prof = [(0.0, 1.0), (0.42, 0.98), (0.55, 0.85), (0.6, 0.55), (0.72, 0.2), (0.95, 0.0), (0.88, 0.0)]
        rings = []
        for k in range(24):
            a_ = 2 * math.pi * k / 24
            rings.append([bm.verts.new((r * math.cos(a_), r * math.sin(a_), z)) for r, z in prof])
        for k in range(24):
            r0, r1 = rings[k], rings[(k + 1) % 24]
            for j in range(len(prof) - 1):
                if prof[j][0] == 0 and prof[j + 1][0] == 0: continue
                vs = [r0[j], r1[j], r1[j + 1], r0[j + 1]]
                if r0[j] is r1[j]: vs = [r0[j], r1[j + 1], r0[j + 1]]
                try: bm.faces.new(vs)
                except Exception: pass
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    elif kind == 'gear':
        teeth, r0, r1, th = a
        vs_t, vs_b = [], []
        for k in range(teeth * 4):
            ang = 2 * math.pi * k / (teeth * 4); r = r1 if k % 4 in (1, 2) else r0
            vs_t.append(bm.verts.new((r * math.cos(ang), th / 2, r * math.sin(ang))))
            vs_b.append(bm.verts.new((r * math.cos(ang), -th / 2, r * math.sin(ang))))
        bm.faces.new(vs_t); bm.faces.new(vs_b[::-1])
        for k in range(len(vs_t)):
            k2 = (k + 1) % len(vs_t); bm.faces.new((vs_t[k], vs_t[k2], vs_b[k2], vs_b[k]))
    me = bpy.data.meshes.new(kind); bm.to_mesh(me); bm.free()
    if kind in ('sph', 'cone', 'bowl', 'bell'):
        for p in me.polygons: p.use_smooth = True
    if m: me.materials.append(m)
    _MC[key] = me; return me


def obj(me, name='o'): return link(bpy.data.objects.new(name, me))


def box(c, size, m, rot=None, name='box'):
    o = obj(prim('cube', m), name); o.location = c; o.scale = size
    if rot is not None: o.rotation_euler = rot
    return o


def rod(p0, p1, r, m, verts=8, name='rod'):
    p0, p1 = Vector(p0), Vector(p1); d = p1 - p0
    o = obj(prim('cyl', m, verts), name); q = Vector((0, 0, 1)).rotation_difference(d.normalized() if d.length > 1e-9 else Vector((0, 0, 1)))
    o.matrix_world = Matrix.Translation((p0 + p1) / 2) @ q.to_matrix().to_4x4() @ Matrix.Diagonal((r, r, max(1e-4, d.length), 1))
    return o


def sphere(c, r, m, seg=(16, 8), name='sph'):
    o = obj(prim('sph', m, *seg), name); o.location = c; o.scale = (r, r, r); return o


def text(s, loc, size, m, rot=(math.pi / 2, 0, 0), align='CENTER', ext=0.02):
    cu = bpy.data.curves.new('t', 'FONT'); cu.body = s; cu.font = FONT; cu.size = size; cu.extrude = ext
    cu.align_x = align; cu.align_y = 'CENTER'
    o = link(bpy.data.objects.new('txt', cu)); o.location = loc; o.rotation_euler = rot; o.data.materials.append(m); return o


def pulse(times, decay=5.0, length=16, amp=1.0):
    a = np.zeros(NF)
    for t in times:
        f0 = int(round(t * FPS)); m_ = np.arange(max(0, f0), min(NF, f0 + length))
        a[m_] = np.maximum(a[m_], amp * np.exp(-(m_ - f0) / decay))
    return a


def smoothstep(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


def frame_from_n(n):
    n = np.array(n, float); n /= np.linalg.norm(n)
    xa = np.cross([0, 1, 0], n); xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return Matrix(((xa[0], ya[0], n[0]), (xa[1], ya[1], n[1]), (xa[2], ya[2], n[2])))


# ---------------- ball motion ----------------
def seg_at(g, t):
    p = np.array(g['p'])
    if g['k'] == 'rest': return p
    u = t - g['t0']; a = np.array(g['a']) if g['k'] == 'acc' else np.array([0, 0, -G])
    return p + np.array(g['v']) * u + 0.5 * a * u * u
class BallPath:
    def __init__(s, seg): s.seg = seg
    def at(s, t):
        for g in s.seg:
            if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9: return seg_at(g, t)
        return seg_at(s.seg[0], s.seg[0]['t0']) if t < s.seg[0]['t0'] else seg_at(s.seg[-1], s.seg[-1]['t1'])
BP = {n: BallPath(d['seg']) for n, d in LAY['balls'].items()}
POS = {n: np.array([b.at(t) for t in T]) for n, b in BP.items()}
HITS = {n: d['hits'] for n, d in LAY['balls'].items()}
FIRST = {n: min(h['t'] for h in d['hits']) for n, d in LAY['balls'].items()}

# ---------------- materials ----------------
M_BALL = glowmat('ball', gain=2.2, base=0.0, rough=0.18, metal=0.25, coat=1.0)
M_KEY = glowmat('key', gain=7.0, base=0.0, rough=0.28, metal=0.65, coat=0.3)
_b = M_KEY.node_tree.nodes['Principled BSDF']
for l_ in list(M_KEY.node_tree.links):
    if l_.to_socket == _b.inputs['Base Color']: M_KEY.node_tree.links.remove(l_)
_mixk = M_KEY.node_tree.nodes.new('ShaderNodeMixRGB'); _mixk.inputs['Fac'].default_value = 0.35
_mixk.inputs['Color1'].default_value = (0.85, 0.62, 0.26, 1)
M_KEY.node_tree.links.new(M_KEY.node_tree.nodes['Object Info'].outputs['Color'], _mixk.inputs['Color2'])
M_KEY.node_tree.links.new(_mixk.outputs[0], _b.inputs['Base Color'])
M_BELL = glowmat('bellm', gain=6.0, base=0.08, rough=0.25, metal=0.85, coat=0.2)
M_BOWL = glowmat('bowl', gain=6.0, base=0.0, rough=0.25, metal=0.85, coat=0.2)
M_FX = fxmat('fx', 8.0)
M_WIN = winmat('win')
M_IRON = mat('iron', (0.08, 0.08, 0.09), 0.45, **{'Metallic': 0.85})
M_BRASS = mat('brass', (0.78, 0.55, 0.22), 0.3, **{'Metallic': 0.9})
M_WOOD = mat('wood', (0.24, 0.13, 0.06), 0.65)
M_WOODL = mat('woodl', (0.38, 0.22, 0.1), 0.6)
M_ROPE = glowmat('rope', gain=9.0, base=0.0, rough=0.8)
_b = M_ROPE.node_tree.nodes['Principled BSDF']
for l_ in list(M_ROPE.node_tree.links):
    if l_.to_socket == _b.inputs['Base Color']: M_ROPE.node_tree.links.remove(l_)
_b.inputs['Base Color'].default_value = (0.09, 0.06, 0.04, 1)
M_STONE = stonemat('stone', (0.23, 0.24, 0.28), (0.17, 0.18, 0.22), scale=(1.6, 3.2))
M_STONE_OUT = stonemat('stoneout', (0.2, 0.21, 0.25), (0.14, 0.15, 0.19), scale=(1.2, 2.4), bump=0.5)
M_FLOOR = stonemat('floor', (0.16, 0.15, 0.15), (0.11, 0.1, 0.1), scale=(1.2, 1.2))
M_GROUND = mat('ground', (0.05, 0.06, 0.07), 0.25, **{'Coat Weight': 0.6})          # wet cobbles outside
M_DOOR = mat('door', (0.2, 0.1, 0.045), 0.7)
M_FLAME = emis('flame', (1.0, 0.45, 0.12), 6.0)
M_TXT = emis('txt', (1.0, 0.85, 0.45), 4.0)
M_OBJ = objcol('objcol', 0.5)

# ---------------- balls ----------------
BCOL = {'visitor': (0.92, 0.94, 1.0), 'inner': (0.95, 0.2, 0.12), 'sleeper': (1.0, 0.62, 0.18), 'runner': (0.25, 0.85, 0.95),
        'bell': (1.0, 0.82, 0.3), 'walker': (0.55, 1.0, 0.45), 'twin': (0.75, 0.82, 1.0), 'hammer': (1.0, 0.18, 0.25), 'plain': (1, 0.7, 0.3)}
BALL = {}
for n, d in LAY['balls'].items():
    o = obj(prim('sph', M_BALL, 32, 16), n); o.scale = (R, R, R)
    col = BCOL.get(d['kind'], (1, 0.7, 0.3))
    if n.startswith('C'): col = (0.25, 0.85, 0.95)
    if n in ('Q1', 'Q2', 'V'): col = (0.95, 0.85, 0.6)
    key_loc(o, POS[n])
    # asleep (dark, alpha 0) until its first hit, then awake: glows softly; flashes at each hit
    awake = np.where(T >= FIRST[n] - 1e-6, 0.12 if n != 'K' else 0.25, 0.0)
    if n == 'K': awake[:] = 0.25
    a = np.maximum(awake, pulse([h['t'] for h in d['hits']], decay=3.0, length=10, amp=0.9))
    base = np.array(col)
    o.color = (*(base * 0.35), 0)
    dim = np.where(T >= FIRST[n] - 1e-6, 1.0, 0.35) if n != 'K' else np.ones(NF)
    for c in range(3): key_series(o, 'color', c, base[c] * dim, 1e-3)
    key_series(o, 'color', 3, a, 1e-3)
    sp = np.zeros(NF); P_ = POS[n]
    for f in range(1, NF): sp[f] = sp[f - 1] + (P_[f, 0] - P_[f - 1, 0]) / R
    o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, sp)
    BALL[n] = o


def pitch_col(p):
    """warm brass with a hue per pitch class (C red ... B violet), lighter when higher"""
    h = ((p % 12) / 12.0); import colorsys
    r, g, b = colorsys.hsv_to_rgb((0.08 + 0.85 * h) % 1.0, 0.7, 1.0)
    return (r, g, b)


# ---------------- keys / bells / bowls: one per hit ----------------
KEYS = []
BACKY = 2.55
for n, d in LAY['balls'].items():
    for h in d['hits']:
        k = h['kind']; i = h['i']; t = h['t']
        nrm = np.array(h['n'], float); nrm /= np.linalg.norm(nrm); c = np.array(h['p']) - R * nrm
        pitch = LEAD[i][1] if 'twin' not in k else (h.get('twin_pitch') or LEAD[i][1])
        if k in ('plate', 'gong'): continue
        if k in ('bowl', 'twinbowl', 'ropebowl', 'lever'):
            rr = 0.24 if k != 'ropebowl' else 0.3
            o = obj(prim('bowl', M_BOWL), 'bowl'); o.location = (c[0], c[1], c[2] + rr - 0.01); o.scale = (rr, rr, rr * 0.85)
            col = pitch_col(pitch) if k != 'lever' else (1.0, 0.3, 0.2)
            KEYS.append(dict(o=o, t=t, col=col, n=np.array([0, 0, 1.0]), kind='bowl'))
            # bracket to the back wall (or a post for the front lane)
            if c[1] >= -0.1: rod((c[0], c[1] + rr * 0.8, c[2] + 0.05), (c[0], BACKY, c[2] + 0.05), 0.022, M_IRON, 6)
            else: rod((c[0], c[1], c[2] - 0.02), (c[0], c[1], c[2] - 0.35), 0.03, M_IRON, 6)
            continue
        if k == 'bell':
            o = obj(prim('bell', M_BELL), 'bell'); s_ = 0.26
            o.matrix_world = Matrix.Translation(Vector(c - nrm * 0.24)) @ frame_from_n(nrm).to_4x4() @ Matrix.Diagonal((s_, s_, 0.24, 1))
            KEYS.append(dict(o=o, t=t, col=pitch_col(pitch), n=nrm, kind='bell'))
            rod((c[0], c[1], c[2]), (c[0], c[1], c[2] + 0.55), 0.012, M_IRON, 6)
            continue
        dk = 'twinkey' if d['kind'] == 'twin' else ('step' if k == 'step' else 'key')
        w, dp, th = KD[dk]
        ctr = c - nrm * th / 2
        o = obj(prim('cube', M_KEY), 'key')
        o.matrix_world = Matrix.Translation(Vector(ctr)) @ frame_from_n(nrm).to_4x4() @ Matrix.Diagonal((w, dp, th, 1))
        KEYS.append(dict(o=o, t=t, col=pitch_col(pitch) if k != 'step' else (0.75, 0.75, 0.8), n=nrm, kind=k))
        if k == 'step':                                     # iron step on a stone post outside
            box((c[0], c[1], (c[2] - th + 0.0) / 2 - 0.02), (0.2, 0.45, c[2] - th - 0.02), M_STONE_OUT)
        elif c[1] >= -0.1:
            rod((ctr[0], ctr[1] + dp / 2 - 0.05, ctr[2] - 0.02), (ctr[0], BACKY, ctr[2] - 0.02), 0.012, M_IRON, 6)
        else:
            rod((ctr[0], ctr[1], ctr[2] - th / 2), (ctr[0], ctr[1], ctr[2] - 0.3), 0.018, M_IRON, 6)
for K_ in KEYS:
    o = K_['o']; fh = int(round(K_['t'] * FPS)); base = o.location.copy(); nv = Vector(K_['n'])
    o.color = (*K_['col'], 0.0)
    o.color = (*K_['col'], 0.05); o.keyframe_insert('color', index=3, frame=max(1, fh - 1))
    lit = 0.1 if K_['kind'] in ('bowl',) else 0.0
    o.color = (*K_['col'], 0.05); o.keyframe_insert('color', index=3, frame=0)
    for df, a in ((0, 1.0), (4, 0.5), (16, 0.14 + lit)):
        o.color = (*K_['col'], a); o.keyframe_insert('color', index=3, frame=fh + df)
    for c_ in range(3):
        o.keyframe_insert('color', index=c_, frame=0)
    if K_['kind'] != 'bowl':
        for df, dd in ((-1, 0), (1, -0.022), (4, 0.008), (8, 0)):
            o.location = base + nv * dd; o.keyframe_insert('location', frame=max(0, fh + df))
    else:
        for df, dd in ((-1, 0), (1, -0.035), (5, 0.01), (10, 0)):
            o.location = base + Vector((0, 0, dd)); o.keyframe_insert('location', frame=max(0, fh + df))

# hit rings + sparks (FX pool)
NRING = 16
ring_loc = np.zeros((NRING, NF, 3)); ring_scl = np.full((NRING, NF), 1e-3); ring_a = np.zeros((NRING, NF)); ring_c = [(1, 0.8, 0.5)] * NRING
ALLH = sorted([(h['t'], np.array(h['p']) - R * np.array(h['n']), LEAD[h['i']][1]) for d in LAY['balls'].values() for h in d['hits']], key=lambda x: x[0])
for k, (t, c, p) in enumerate(ALLH):
    r = k % NRING; f0 = int(round(t * FPS))
    for df in range(13):
        f = f0 + df
        if f >= NF: break
        u = df / 12; ring_loc[r, f] = c + np.array([0, -0.3, 0]); ring_scl[r, f] = 0.25 + 1.1 * (1 - (1 - u) ** 2); ring_a[r, f] = (1 - u) ** 1.7
for r in range(NRING):
    bpy.ops.mesh.primitive_torus_add(major_radius=0.2, minor_radius=0.012, major_segments=32, minor_segments=5, rotation=(math.pi / 2, 0, 0))
    o = bpy.context.object; o.data.materials.append(M_FX); o.color = (1.0, 0.8, 0.45, 0)
    for f in range(1, NF):
        if ring_a[r, f] == 0: ring_loc[r, f] = ring_loc[r, f - 1]
    key_loc(o, ring_loc[r])
    for k in range(3): key_series(o, 'scale', k, ring_scl[r])
    key_series(o, 'color', 3, ring_a[r], 1e-3)


def sparks(t, c, n=10, col=(1.0, 0.7, 0.3), sp=1.6, life=0.6, up=1.5, size=0.03, dirx=0.0):
    for k in range(n):
        o = sphere((0, 0, 0), size, M_FX, (8, 4)); o.color = (*col, 0)
        v = np.array([rng.normal(dirx, 0.8) * sp, rng.normal(-0.4, 0.4), rng.uniform(0.3, 1.0) * up])
        P_ = np.zeros((NF, 3)); A_ = np.zeros(NF)
        for f in range(NF):
            u = T[f] - t
            if u < 0: P_[f] = c; continue
            P_[f] = c + v * u + np.array([0, 0, -0.5 * G * u * u])
            A_[f] = max(0, 1 - u / life) if u < life else 0
        key_loc(o, P_); key_series(o, 'color', 3, A_, 1e-3)


# ---------------- the castle ----------------
XMAX = max(h['p'][0] for d in LAY['balls'].values() for h in d['hits']) + 2.2
ZTOP = 11.2
# back wall (cut-away castle: we look in through the missing front wall)
bw = box(((0.5 + XMAX) / 2, BACKY + 0.25, ZTOP / 2), (XMAX - 0.5, 0.5, ZTOP), M_STONE, name='backwall')
# floor inside, ground outside
box(((0.5 + XMAX) / 2, 0.6, -0.2), (XMAX - 0.5, 4.8, 0.4), M_FLOOR, name='floor')
box((-30, 0, -0.2), (60, 90, 0.4), M_GROUND, name='ground')
# door tower wall (above the door) and the end tower wall
box((0.25, 0.8, (DOOR['h'] + ZTOP + 0.6) / 2), (0.5, 4.4, ZTOP + 0.6 - DOOR['h']), M_STONE_OUT, name='doorwall')
box((0.25, -1.3, DOOR['h'] / 2), (0.5, 0.5, DOOR['h']), M_STONE_OUT, name='jamb')
box((0.25, 2.0, DOOR['h'] / 2), (0.5, 2.0, DOOR['h']), M_STONE_OUT, name='jamb')
# stone arch over the door on the outer face
for k in range(11):
    a_ = math.pi * k / 10
    cy, cz = -1.2 * math.cos(a_), DOOR['h'] - 0.05 + 0.75 * math.sin(a_)
    box((-0.05, cy, cz), (0.14, 0.3, 0.3), M_STONE_OUT, rot=(a_ - math.pi / 2, 0, 0), name='voussoir')
for yy in (-1.4, 1.4):
    sphere((-0.22, yy, 2.55), 0.08, emis('lant', (1.0, 0.65, 0.3), 9.0))
    rod((0.0, yy, 2.7), (-0.22, yy, 2.7), 0.015, M_IRON, 6)
    L_out = bpy.data.lights.new('lantern', 'POINT'); L_out.color = (1.0, 0.6, 0.3); L_out.energy = 30; L_out.shadow_soft_size = 0.15
    lo = link(bpy.data.objects.new('lantern', L_out)); lo.location = (-0.45, yy, 2.5)
box((XMAX + 0.25, 0.6, ZTOP / 2), (0.5, 4.8, ZTOP), M_STONE_OUT, name='endwall')
# battlements on the roof line
for x in np.arange(0.0, XMAX + 0.5, 0.7):
    box((x + 0.2, 0.6, ZTOP + 0.25), (0.4, 4.8, 0.5), M_STONE_OUT, name='merlon')
box(((XMAX) / 2, 0.6, ZTOP - 0.1), (XMAX + 0.5, 4.8, 0.2), M_STONE_OUT, name='roofline')
# cut-away edges of the front wall (thick stone frame at the bottom and top, gives the dollhouse look)
box(((0.5 + XMAX) / 2, -1.45, 0.12), (XMAX - 0.5, 0.3, 0.24), M_STONE_OUT, name='cutlow')
# the top gallery floor (wooden walkway under the homeward walk)
gal = [h['p'] for n in ('P1', 'P2', 'P3', 'P4') for h in HITS[n] if h['kind'] == 'key']
if gal:
    gx0 = min(p[0] for p in gal) - 0.3; gx1 = max(p[0] for p in gal) + 0.4; gz = min(p[2] for p in gal) - R - 0.35
    box(((gx0 + gx1) / 2, 0.9, gz), (gx1 - gx0, 3.2, 0.12), M_WOOD, name='gallery')
    for x in np.arange(gx0 + 0.3, gx1, 1.4):
        rod((x, -0.65, gz + 0.06), (x, -0.65, gz + 0.55), 0.02, M_WOOD, 6)
    rod((gx0, -0.65, gz + 0.55), (gx1, -0.65, gz + 0.55), 0.025, M_WOOD, 6)
# stone pillars on the back wall between the towers
for x in np.arange(2.0, XMAX, 3.6):
    box((x, BACKY - 0.12, ZTOP / 2), (0.45, 0.3, ZTOP), M_STONE_OUT, name='pillar')

# windows: one per bowl landing (the castle wakes up window by window) + a few that stay dark
WIN = []
for w in EV['windows']:
    p = np.array(w['p'])
    o = box((p[0], BACKY - 0.02, p[2] + 0.45), (0.26, 0.04, 0.36), M_WIN, name='win')
    top = obj(prim('cyl', M_WIN, 16), 'wint'); top.location = (p[0], BACKY - 0.02, p[2] + 0.63); top.rotation_euler = (math.pi / 2, 0, 0); top.scale = (0.13, 0.13, 0.04)
    a = np.where(T >= w['t'], 1.0, 0.0) * (0.75 + 0.25 * np.exp(-np.maximum(0, T - w['t']) / 0.4))
    for oo in (o, top):
        oo.color = (1.0, 0.62, 0.25, 0); key_series(oo, 'color', 3, a, 1e-3)
    box((p[0], BACKY - 0.04, p[2] + 0.26), (0.34, 0.06, 0.04), M_STONE_OUT)
    box((p[0], BACKY - 0.05, p[2] + 0.5), (0.02, 0.02, 0.5), M_IRON); box((p[0], BACKY - 0.05, p[2] + 0.5), (0.26, 0.02, 0.02), M_IRON)
    WIN.append(w['t'])
n_lit = np.array([np.sum(np.array(WIN) <= t) for t in T]) / max(1, len(WIN))

# torches between the windows: flames flicker and flare on every bass note
BASS = sorted(set(round(n_[0], 3) for n_ in NOTES if n_[2] in ('pizz', 'pbass')))
flare = pulse(BASS, decay=3.0, length=10)
TORCH = []
for x in np.arange(1.6, XMAX - 0.5, 2.9):
    for z in (1.7, 6.0):
        c = (x, BACKY - 0.25, z)
        rod((x, BACKY - 0.02, z - 0.25), (x, BACKY - 0.25, z - 0.05), 0.025, M_IRON, 6)
        f_ = obj(prim('cone', M_FLAME, 10, 0.06, 0.0, 0.2), 'flame'); f_.location = (x, BACKY - 0.25, z + 0.06)
        TORCH.append(f_)
fl_master = link(bpy.data.objects.new('flame_master', None))
fs = 1.0 + 0.35 * flare + 0.08 * np.sin(T * 23) + 0.05 * np.sin(T * 37)
key_series(fl_master, 'scale', 2, fs, 1e-3)
for f_ in TORCH:
    cs = f_.constraints.new('COPY_SCALE'); cs.target = fl_master; cs.use_x = False; cs.use_y = False; cs.use_offset = True
em_fl = M_FLAME.node_tree.nodes['Emission']
key_node(em_fl.inputs['Strength'], 5.0 + 9.0 * flare)
# a few warm point lights near the torches (the hall gets warmer as windows light)
for x in np.arange(3.0, XMAX - 1, 6.0):
    for z in (2.0, 6.5):
        L_ = bpy.data.lights.new('tl', 'POINT'); L_.color = (1.0, 0.55, 0.25); L_.shadow_soft_size = 0.4
        lo = link(bpy.data.objects.new('tl', L_)); lo.location = (x, 1.2, z)
        en = 40.0 + 160.0 * n_lit + 40.0 * flare
        for f in range(0, NF, 3): L_.energy = float(en[f]); L_.keyframe_insert('energy', frame=f)

# clockwork gears on the back wall: tick forward on every bass note once the castle is awake
M_GEAR = mat('gear', (0.55, 0.38, 0.16), 0.35, **{'Metallic': 0.9})
tick = np.zeros(NF); ang = 0.0; bi = 0
bass_f = sorted(int(round(t * FPS)) for t in BASS)
for f in range(NF):
    while bi < len(bass_f) and bass_f[bi] <= f:
        if T[f] > TN[8]: ang += 0.09
        bi += 1
    tick[f] = ang
tick_s = np.copy(tick)
for f in range(1, NF): tick_s[f] = tick_s[f - 1] + (tick[f] - tick_s[f - 1]) * 0.55
GEARS = [(4.6, 9.2, 0.9, 18), (6.15, 9.85, 0.55, 11), (14.8, 2.2, 1.0, 20), (16.55, 3.05, 0.62, 12), (24.0, 0.9, 0.75, 14),
         (30.6, 2.2, 1.1, 22), (32.4, 3.5, 0.6, 12), (20.5, 8.6, 0.7, 14)]
for k, (x, z, r, teeth) in enumerate(GEARS):
    if x > XMAX - 0.6: continue
    g_ = obj(prim('gear', M_GEAR, teeth, r * 0.88, r, 0.08), 'gear'); g_.location = (x, BACKY - 0.1, z); g_.rotation_mode = 'XYZ'
    key_series(g_, 'rotation_euler', 1, tick_s * (1 if k % 2 == 0 else -1) * (1.0 / r) * 0.8, 1e-4)
    rod((x, BACKY - 0.16, z), (x, BACKY, z), r * 0.18, M_IRON, 12)

# ---------------- hatches, ramps, cords, baskets ----------------
for hc in EV['hatches']:
    p = np.array(hc['p']); t0 = hc['t']
    for dx in (-0.19, 0.19):
        box((p[0] + dx, p[1], p[2] + 0.04), (0.035, 0.36, 0.34), M_WOOD, name='hwall')
    box((p[0], p[1] + 0.19, p[2] + 0.04), (0.38, 0.03, 0.34), M_WOOD, name='hwall')
    box((p[0], p[1], p[2] + 0.22), (0.42, 0.4, 0.04), M_WOODL, name='hroof')
    if p[1] >= -0.1: rod((p[0], p[1] + 0.2, p[2] + 0.1), (p[0], BACKY, p[2] + 0.1), 0.02, M_IRON, 6)
    # trap flap, hinged at its -x edge, swings down when the cord pulls
    hg = link(bpy.data.objects.new('hinge', None)); hg.location = (p[0] - 0.19, p[1], p[2] - R - 0.03)
    fl = box((0.19, 0, 0), (0.38, 0.36, 0.03), M_WOODL, name='flap'); fl.parent = hg
    a = np.where(T >= t0, np.minimum(1, (T - t0) / 0.12) ** 2 * 1.75, 0.0)
    hg.rotation_mode = 'XYZ'; key_series(hg, 'rotation_euler', 1, a)
for r_ in EV['ramps']:
    top = np.array(r_['top']); end = np.array(r_['end']); u = np.array(r_['u'])
    nn = np.array([-u[2], 0, u[0]]) if u[0] > 0 else np.array([u[2], 0, -u[0]])
    if nn[2] < 0: nn = -nn
    a0 = top - nn * (R + 0.03) - u * 0.25; a1 = end - nn * (R + 0.03)
    if r_.get('road'):
        a0 = top - nn * (R + 0.03) - u * 3.0
        o = rod(a0, a1, 0.001, M_STONE_OUT)            # (direction only)
        L_ = np.linalg.norm(a1 - a0); mid = (a0 + a1) / 2
        bpy.data.objects.remove(o)
        rb = box(mid - nn * 0.25, (L_, 1.6, 0.5), M_STONE_OUT, rot=(0, -math.atan2(u[2], u[0]), 0), name='road')
        continue
    mid = (a0 + a1) / 2; L_ = np.linalg.norm(a1 - a0)
    box(mid - nn * 0.03, (L_, 0.32, 0.05), M_WOOD, rot=(0, -math.atan2(u[2], u[0]), 0), name='ramp')
    for side in (-0.16, 0.16):
        box(mid + nn * 0.04 + np.array([0, side, 0]), (L_, 0.03, 0.1), M_WOODL, rot=(0, -math.atan2(u[2], u[0]), 0), name='rail')
    if top[1] >= -0.1: rod((a0[0], 0.1, a0[2] - 0.05), (a0[0], BACKY, a0[2] - 0.05), 0.02, M_IRON, 6)
    # latch at the top: a little gate that flips up when its cord pulls
    if not r_.get('shelf'):
        lg = link(bpy.data.objects.new('latch', None)); lg.location = tuple(top + u * 0.17 - nn * 0.02)
        gb = box((0, 0, 0.12), (0.03, 0.3, 0.24), M_IRON, name='latchgate'); gb.parent = lg
for cd in EV['cords']:
    s_ = np.array(cd['src']); d_ = np.array(cd['dst']); t0 = cd['t']
    # cord: runs from the trigger key down/up to the latch; sags until pulled, then goes taut (and glows briefly)
    mid_up = (s_ + d_) / 2
    o = rod(s_, d_, 0.007, M_ROPE, 5, name='cord'); o.color = (1.0, 0.7, 0.3, 0.0)
    a = pulse([t0], decay=4, length=14, amp=1.0) * 0.9
    key_series(o, 'color', 3, a, 1e-3)
for bk in EV['baskets']:
    p = np.array(bk['p'])
    if bk['kind'] == 'puddle': continue
    o = obj(prim('bowl', M_WOODL), 'basket'); o.location = (p[0], p[1], p[2] + 0.2); o.scale = (0.24, 0.24, 0.22)
    if bk['kind'] == 'weight':                     # the gong ball's basket hangs from a pulley rope
        rod((p[0], p[1], p[2] + 0.2), (p[0], p[1], p[2] + 1.2), 0.01, M_IRON, 5)
    elif p[1] >= -0.1:
        rod((p[0], p[1] + 0.22, p[2] + 0.05), (p[0], BACKY, p[2] + 0.05), 0.02, M_IRON, 6)

# ---------------- the door, plates, gong, bell rope, lever ----------------
open_t = EV['door_open']; LV = EV['lever']
door_z = np.zeros(NF)
for k, lv in enumerate(LV[:-1]): door_z += np.where(T >= lv['t'], 0.0, 0.0) + smoothstep((T - lv['t']) / 0.12) * 0.16
door_z += smoothstep((T - open_t) / 0.45) * (DOOR['h'] + 0.4)
dr = link(bpy.data.objects.new('door', None))
slab = box((0.25, 0, DOOR['h'] / 2), (0.46, 2.1, DOOR['h']), M_DOOR, name='doorslab'); slab.parent = dr
for yy in np.arange(-0.9, 0.95, 0.3):            # plank seams on both faces
    for xs in (0.015, 0.485):
        sm = box((xs - 0.02 if xs < 0.3 else xs + 0.02, yy + 0.15, DOOR['h'] / 2), (0.012, 0.02, DOOR['h'] - 0.02), M_IRON, name='seam'); sm.parent = dr
for z in np.arange(0.35, DOOR['h'], 0.75):
    for xs in (0.0, 0.5):
        b_ = box((xs, 0, z), (0.04, 2.1, 0.12), M_IRON, name='band'); b_.parent = dr
        for yy in np.arange(-0.9, 0.95, 0.3):
            st = sphere((xs + (0.02 if xs > 0.3 else -0.02), yy, z), 0.03, M_IRON, (8, 4)); st.parent = dr
# knocker ring (outside) and inner plates are part of the door
for pl in EV['plates']:
    p = np.array(pl['p']); n = np.array(pl['n'])
    o = obj(prim('cube', M_KEY), 'plate'); o.location = (p[0] - n[0] * 0.025, p[1], p[2]); o.scale = (0.05, 0.5, 0.42)
    o.parent = dr
    o.color = (1.0, 0.72, 0.35, 0.05)
    key_series(o, 'color', 3, 0.05 + pulse([pl['t']], decay=5, length=24), 1e-3)
    if pl['n'][0] < 0:                              # the iron knocker ring outside
        bpy.ops.mesh.primitive_torus_add(major_radius=0.17, minor_radius=0.025, location=(p[0] - 0.07, p[1], p[2] + 0.3), rotation=(0, math.pi / 2, 0))
        rg = bpy.context.object; rg.data.materials.append(M_BRASS); rg.parent = dr
    sparks(pl['t'], p + n * 0.05, n=12, col=(1.0, 0.75, 0.4), sp=1.0, up=1.2, dirx=n[0] * 0.8)
key_series(dr, 'location', 2, door_z, 1e-4)
# door shudder on each knock
shud = np.zeros(NF)
for pl in EV['plates']:
    u = T - pl['t']; shud += np.where(u > 0, 0.012 * np.sin(u * 60) * np.exp(-u * 8), 0)
key_series(dr, 'location', 0, shud, 1e-5)
# arch over the door + lantern
for k in range(9):
    a_ = math.pi * k / 8
    box((0.25, 1.05 * math.cos(a_) * 0, DOOR['h'] + 0.05 + 0.25 * math.sin(a_)), (0.6, 2.4, 0.12), M_STONE_OUT, name='arch') if k == 4 else None

# puddle where the visitor waits
pd = EV['baskets'][0]['p']
pud = obj(prim('cyl', mat('puddle', (0.02, 0.03, 0.05), 0.02, **{'Coat Weight': 1.0}), 32), 'puddle')
pud.location = (pd[0], pd[1], 0.002); pud.scale = (0.45, 0.6, 0.004)
splash_t = EV['baskets'][0]['t']
for k in range(10):
    o = sphere((0, 0, 0), 0.025, M_FX, (8, 4)); o.color = (0.7, 0.85, 1.0, 0)
    ang = 2 * math.pi * k / 10; v = np.array([math.cos(ang) * 0.8, math.sin(ang) * 0.8, 1.6])
    P_ = np.zeros((NF, 3)); A_ = np.zeros(NF)
    for f in range(NF):
        u = T[f] - splash_t
        P_[f] = np.array(pd) + (v * u + np.array([0, 0, -0.5 * G * u * u]) if u > 0 else 0)
        A_[f] = max(0, 1 - u / 0.45) if 0 < u < 0.45 else 0
    key_loc(o, P_); key_series(o, 'color', 3, A_, 1e-3)
# the gong
gg = EV['gong']; gp = np.array(gg['p'])
gpiv = link(bpy.data.objects.new('gongpiv', None)); gpiv.location = (gp[0] + 0.05, 0, gp[2] + 0.9)
disc = obj(prim('cyl', M_BELL, 40), 'gong'); disc.parent = gpiv; disc.location = (0, 0, -0.9); disc.rotation_euler = (0, math.pi / 2, 0)
disc.scale = (0.55, 0.55, 0.04); disc.color = (1.0, 0.7, 0.3, 0)
key_series(disc, 'color', 3, pulse([gg['t']], decay=12, length=60), 1e-3)
sw = np.where(T > gg['t'], 0.25 * np.sin((T - gg['t']) * 5.5) * np.exp(-(T - gg['t']) * 1.0), 0)
gpiv.rotation_mode = 'XYZ'; key_series(gpiv, 'rotation_euler', 1, sw, 1e-4)
for yy in (-0.45, 0.45):
    rod((gp[0] + 0.05, yy, gp[2] + 0.9), (gp[0] + 0.05, yy, gp[2] - 0.75), 0.03, M_WOOD, 6)
rod((gp[0] + 0.05, -0.5, gp[2] + 0.9), (gp[0] + 0.05, 0.5, gp[2] + 0.9), 0.04, M_WOOD, 8)
sparks(gg['t'], gp + [-0.05, 0, 0], n=16, col=(1, 0.75, 0.35), sp=1.4, up=1.6, dirx=-0.8)
# bell rope (bowl -> bell-tower hatch): the long rope glows when pulled
br_ = EV['bellrope']
# lever: a beam on a fulcrum; each hammer ball's landing pushes it down a notch, the chain lifts the door
lv0 = np.array(LV[0]['p'])
lvp = link(bpy.data.objects.new('leverpiv', None)); lvp.location = (lv0[0] - 0.7, 0, lv0[2] - R - 0.12)
beam = box((0.55, 0, 0), (1.6, 1.3, 0.07), M_WOOD, name='leverbeam'); beam.parent = lvp
la = np.zeros(NF)
for k, lv in enumerate(LV): la += smoothstep((T - lv['t']) / 0.1) * (0.07 if k < 2 else 0.2)
lvp.rotation_mode = 'XYZ'; key_series(lvp, 'rotation_euler', 1, la, 1e-4)
box((lv0[0] - 0.7, 0, (lv0[2] - R - 0.12) / 2 - 0.05), (0.18, 0.5, lv0[2] - R - 0.12), M_STONE_OUT, name='fulcrum')
# chain from the lever's short arm up over a pulley above the door down to the door top
chain_top = (0.25, 0, DOOR['h'] + 0.9)
rod((lv0[0] - 1.2, 0, lv0[2] - 0.05), (lv0[0] - 1.2, 0, DOOR['h'] + 0.9), 0.02, M_IRON, 6)
rod((lv0[0] - 1.2, 0, DOOR['h'] + 0.9), chain_top, 0.02, M_IRON, 6)
pul = obj(prim('cyl', M_IRON, 24), 'pulley'); pul.location = chain_top; pul.rotation_euler = (math.pi / 2, 0, 0); pul.scale = (0.12, 0.12, 0.08)

# ---------------- outside: night sky, moon, lightning, dawn ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes['Background']
dawn_t = EV['dawn']
dawn = smoothstep((T - dawn_t) / 14.0)
flash_t = [EV['plates'][0]['t'] + 0.12, EV['plates'][1]['t'] + 0.2, EV['gong']['t'] + 0.05, EV['plates'][2]['t'] + 0.15]
flash = np.zeros(NF)
for t in flash_t:
    for dt_, a_ in ((0.0, 1.0), (0.13, 0.6)):
        f0 = int(round((t + dt_) * FPS))
        for df in range(5):
            if 0 <= f0 + df < NF: flash[f0 + df] = max(flash[f0 + df], a_ * (1 - df / 5))
night = np.array([0.02, 0.03, 0.08]); morning = np.array([0.55, 0.45, 0.38])
for f in range(0, NF):
    if 0 < f < NF - 1 and abs(dawn[f] - dawn[f - 1]) < 1e-3 and flash[f] == 0 and flash[f - 1] == 0 and flash[f + 1] == 0: continue
    c = night * (1 - dawn[f]) + morning * dawn[f] + flash[f] * np.array([0.6, 0.65, 0.9])
    bg.inputs['Color'].default_value = (*c, 1); bg.inputs['Color'].keyframe_insert('default_value', frame=f)
    bg.inputs['Strength'].default_value = 0.5 + 0.6 * dawn[f] + 2.5 * flash[f]; bg.inputs['Strength'].keyframe_insert('default_value', frame=f)
SKY = bpy.data.materials.new('sky'); SKY.use_nodes = True; nt = SKY.node_tree; nt.nodes.clear()
tc = nt.nodes.new('ShaderNodeTexCoord'); sepz = nt.nodes.new('ShaderNodeSeparateXYZ'); mp = nt.nodes.new('ShaderNodeMapRange')
mp.inputs['From Min'].default_value = -0.5; mp.inputs['From Max'].default_value = 0.5
rpn = nt.nodes.new('ShaderNodeValToRGB'); rpn.color_ramp.elements[0].color = (0.03, 0.04, 0.12, 1); rpn.color_ramp.elements[1].color = (0.005, 0.008, 0.03, 1)
rpd = nt.nodes.new('ShaderNodeValToRGB'); rpd.color_ramp.elements[0].color = (1.0, 0.55, 0.25, 1); rpd.color_ramp.elements[1].color = (0.35, 0.5, 0.85, 1)
mix = nt.nodes.new('ShaderNodeMixRGB'); em = nt.nodes.new('ShaderNodeEmission'); out = nt.nodes.new('ShaderNodeOutputMaterial')
nt.links.new(tc.outputs['Generated'], sepz.inputs[0]); nt.links.new(sepz.outputs['Y'], mp.inputs['Value'])
nt.links.new(mp.outputs[0], rpn.inputs['Fac']); nt.links.new(mp.outputs[0], rpd.inputs['Fac'])
nt.links.new(rpn.outputs['Color'], mix.inputs['Color1']); nt.links.new(rpd.outputs['Color'], mix.inputs['Color2'])
nt.links.new(mix.outputs[0], em.inputs['Color']); nt.links.new(em.outputs[0], out.inputs[0])
mp.inputs['From Min'].default_value = 0.0; mp.inputs['From Max'].default_value = 1.0
key_node(mix.inputs['Fac'], dawn)
key_node(em.inputs['Strength'], 1.0 + 0.8 * dawn + 6.0 * flash)
skyp = obj(prim('plane', SKY), 'sky'); skyp.location = (8, 40, 6); skyp.rotation_euler = (math.pi / 2, 0, 0); skyp.scale = (140, 70, 1)
moon = sphere((-14, 38, 15), 1.6, emis('moon', (0.85, 0.9, 1.0), 6.0), (24, 12))
mo_em = moon.data.materials[0].node_tree.nodes['Emission']; key_node(mo_em.inputs['Strength'], 6.0 * (1 - dawn) + 0.3)
sun_ob = sphere((-30, 39, -4), 3.0, emis('sunball', (1.0, 0.7, 0.35), 0.0), (24, 12))
su_em = sun_ob.data.materials[0].node_tree.nodes['Emission']; key_node(su_em.inputs['Strength'], 9.0 * dawn)
sz = -4 + 16 * dawn
key_series(sun_ob, 'location', 2, sz, 1e-3)
# lightning bolts (jagged emissive tubes in the sky)
for k, t in enumerate(flash_t):
    pts = [np.array([-4.0 + 3 * k, 30, 22])]
    for j in range(9): pts.append(pts[-1] + np.array([rng.normal(0, 1.2), 0, -2.4]))
    bm = bmesh.new()
    for a_, b_ in zip(pts, pts[1:]):
        o = rod(a_, b_, 0.12, M_FX, 6, name='bolt'); o.color = (0.8, 0.85, 1.0, 0)
        key_series(o, 'color', 3, np.where(np.abs(T - t) < 0.07, 1.0, 0.0) + np.where(np.abs(T - t - 0.13) < 0.04, 0.6, 0.0), 1e-3)
    bm.free()
# moonlight / sunlight
moonl = bpy.data.lights.new('moon', 'SUN'); moonl.angle = 0.05
ml = link(bpy.data.objects.new('moonl', moonl)); ml.rotation_euler = (math.radians(55), math.radians(-30), math.radians(-25))
for f in range(0, NF):
    if 0 < f < NF - 1 and abs(dawn[f] - dawn[f - 1]) < 1e-3 and flash[f] == 0 and flash[f - 1] == 0 and flash[f + 1] == 0: continue
    c = np.array([0.55, 0.65, 1.0]) * (1 - dawn[f]) + np.array([1.0, 0.75, 0.5]) * dawn[f]
    moonl.color = tuple(c); moonl.keyframe_insert('color', frame=f)
    moonl.energy = 0.6 + 2.6 * dawn[f] + 7.0 * flash[f]; moonl.keyframe_insert('energy', frame=f)
# fill light from the open front (the cut-away side) so the machinery reads at night
fill = bpy.data.lights.new('fill', 'AREA'); fill.shape = 'RECTANGLE'; fill.size = 40; fill.size_y = 14; fill.color = (0.75, 0.8, 1.0)
fo = link(bpy.data.objects.new('fill', fill)); fo.location = (16, -9, 7); fo.rotation_euler = (math.radians(90), 0, 0)
for f in range(0, NF, 3):
    fill.energy = float(2600 + 2600 * n_lit[f] + 2600 * dawn[f]); fill.keyframe_insert('energy', frame=f)

# ---------------- finale: light floods out of the door, the castle's balls pour out ----------------
glow = bpy.data.lights.new('doorlight', 'AREA'); glow.shape = 'RECTANGLE'; glow.size = 2.0; glow.size_y = 3.0; glow.color = (1.0, 0.85, 0.6)
go = link(bpy.data.objects.new('doorlight', glow)); go.location = (1.4, 0, 1.6); go.rotation_euler = (0, math.radians(-90), 0)
gl_e = 9000 * smoothstep((T - open_t) / 0.35) * (1 - 0.5 * smoothstep((T - open_t - 1.5) / 1.5))
for f in range(0, NF):
    if f > 0 and abs(gl_e[f] - gl_e[f - 1]) < 1 and f < NF - 1: continue
    glow.energy = float(gl_e[f]); glow.keyframe_insert('energy', frame=f)
lp = box((1.0, 0.0, 1.6), (0.05, 2.0, 3.2), M_FX, name='doorglow'); lp.color = (1.0, 0.85, 0.6, 0)
key_series(lp, 'color', 3, 1.4 * smoothstep((T - open_t) / 0.3), 1e-3)
PCOL = [(1.0, 0.62, 0.18), (0.25, 0.85, 0.95), (0.55, 1.0, 0.45), (0.75, 0.82, 1.0), (1.0, 0.82, 0.3), (1.0, 0.25, 0.3)]
for pb in EV['pour']:
    o = obj(prim('sph', M_BALL, 24, 12), 'pour'); P_ = np.array([seg_at([g for g in pb['seg'] if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9][0], t) for t in T])
    key_loc(o, P_)
    s_ = R * smoothstep((T - pb['t0']) / 0.07)
    for c_ in range(3): key_series(o, 'scale', c_, np.maximum(1e-4, s_), 1e-5)
    col = PCOL[pb['col']]; o.color = (*col, 0.35)
# closing question
qt = text('扉の向こうに\n何があった？', (0, 0, 0), 0.3, M_TXT, ext=0.004)
qt.data.space_line = 1.05
T_Q = open_t + 1.6

# ---------------- camera ----------------
def smooth(Xa, k):
    Y = Xa.copy(); a = 1 - math.exp(-1 / FPS / k)
    for f in range(1, len(Xa)): Y[f] = Y[f - 1] + (Xa[f] - Y[f - 1]) * a
    for f in range(len(Xa) - 2, -1, -1): Y[f] = Y[f + 1] + (Y[f] - Y[f + 1]) * a
    return Y
ACT = {}
for n, d in LAY['balls'].items():
    ts = [h['t'] for h in d['hits']]; ACT[n] = (min(ts) - 0.55, max(ts) + 0.35)
FOC = np.zeros((NF, 3)); WID = np.zeros(NF)
last = np.array([0.0, 0, 1.3])
for f in range(NF):
    t = T[f]
    act = [n for n, (a, b) in ACT.items() if a <= t <= b]
    if act:
        P_ = np.array([POS[n][f] for n in act])
        lo_, hi_ = P_.min(0), P_.max(0)
        c = (lo_ + hi_) / 2; span = hi_ - lo_
        wid = max(4.6, span[0] + 2.2, (span[2] + 1.9) * 9 / 16)
        FOC[f] = c + np.array([0, 0, 0.25]); WID[f] = min(wid, 7.0); last = FOC[f].copy()
    else:
        FOC[f] = last; WID[f] = WID[f - 1] if f else 4.6
# story overrides
def blend_to(t0, t1, foc, wid, ease=0.35):
    for f in range(NF):
        t = T[f]
        if t0 - ease <= t <= t1 + ease:
            u = 1.0 if t0 <= t <= t1 else (1 - abs(t - (t0 if t < t0 else t1)) / ease)
            u = smoothstep(np.array(u)).item()
            FOC[f] = FOC[f] * (1 - u) + np.array(foc) * u; WID[f] = WID[f] * (1 - u) + wid * u
pl0 = EV['plates']
blend_to(0.0, TN[7] + 0.35, (-0.1, 0, 1.25), 4.9, ease=0.2)                        # the door: both sides
# follow the cord from the door to the first chute (fermata)
r1 = EV['ramps'][2]
for f in range(NF):
    t = T[f]; t0, t1 = TN[7] + 0.35, TN[8] - 0.1
    if t0 < t < t1:
        u = smoothstep(np.array((t - t0) / (t1 - t0))).item()
        FOC[f] = np.array([0.25, 0, 1.3]) * (1 - u) + (np.array(r1['end']) + [-0.6, 0, 0.6]) * u; WID[f] = 4.9 + 2.4 * math.sin(math.pi * u)
# reveal: the gong rings, the camera pulls back - the castle is huge; then dive to the door for the second knock
tg = EV['gong']['t']
blend_to(tg + 0.15, tg + 0.55, (9.0, 0, 5.6), 15.0, ease=0.3)
blend_to(TN[54] - 0.15, TN[57] + 0.45, (0.6, 0, 1.8), 4.8, ease=0.35)
# follow the second cord to the second tower
r2 = EV['ramps'][3]
for f in range(NF):
    t = T[f]; t0, t1 = TN[57] + 0.45, TN[58] - 0.15
    if t0 < t < t1:
        u = smoothstep(np.array((t - t0) / (t1 - t0))).item()
        FOC[f] = np.array([0.6, 0, 1.8]) * (1 - u) + (np.array(r2['end']) + [0.6, 0, -0.4]) * u; WID[f] = 4.8 + 3.0 * math.sin(math.pi * u)
# bell rope: tilt up along the rope
blend_to(TN[182] + 0.05, TN[183] - 0.05, tuple((np.array(br_['p0']) + np.array(br_['p1'])) / 2 + [0.5, 0, 0]), 6.2, ease=0.2)
# twins: frame both
blend_to(TN[252] - 0.3, TN[310], (EV['twins']['xc'], 0, float(np.mean([POS['T1a'][int(TN[252] * FPS), 2], POS['T6a'][int(TN[310] * FPS), 2]]))), 5.6, ease=0.5)
# finale: the lever and the door together, then the door
blend_to(TN[339] - 0.3, open_t - 0.05, ((lv0[0] + 0.25) / 2 + 0.2, 0, 2.7), 6.6, ease=0.4)
blend_to(open_t + 0.1, open_t + 2.0, (-0.6, 0, 1.7), 6.0, ease=0.25)
blend_to(open_t + 2.3, 200, (2.5, 0, 3.2), 8.6, ease=0.9)
WID = np.clip(WID, 4.4, 16.0)
FOC[:, 0] = smooth(FOC[:, 0], 0.28); FOC[:, 2] = smooth(FOC[:, 2], 0.35); WID = smooth(WID, 0.4)
LENS = 32.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 400
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
YAW = np.zeros(NF)
YAW += np.where(T < TN[3] + 0.5, 30.0, 0.0)
YAW += np.where((T >= TN[3] + 0.5) & (T < TN[7] + 0.3), 14.0, 0.0)
YAW += np.where((T >= open_t - 0.25) & (T < open_t + 2.1), 24.0, 0.0)
YAW = smooth(YAW, 0.3)
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24; yw = math.radians(YAW[f])
    CP[f] = (FOC[f, 0] - D * math.sin(yw), -D * math.cos(yw), FOC[f, 2] + 0.12 * D)
    TP[f] = (FOC[f, 0], 0.0, FOC[f, 2])
key_loc(cam, CP); key_loc(tgt, TP)
# closing question on the camera (HUD), fades in
hud = link(bpy.data.objects.new('hud', None)); hud.parent = cam; hud.location = (0, 0, -3.0)
HH = 3.0 * 24 / LENS
qt.parent = hud; qt.location = (0, HH * 0.27, 0); qt.rotation_euler = (0, 0, 0); qt.visible_shadow = False
qs = smoothstep((T - T_Q) / 0.5)
for c_ in range(3): key_series(qt, 'scale', c_, np.maximum(1e-3, qs), 1e-4)


# ---------------- checks ----------------
def check():
    info = {}
    gaps = []
    for n, d in LAY['balls'].items():
        for h in d['hits']:
            f = int(round(h['t'] * FPS)); gaps.append(np.linalg.norm(POS[n][min(NF - 1, f)] - np.array(h['p'])))
    info['contact_gap_at_hit_frame_max'] = round(max(gaps), 3)
    info['keys'] = len(KEYS); info['objects'] = len(bpy.data.objects); info['NF'] = NF
    info['cam_width_range'] = (round(float(WID.min()), 2), round(float(WID.max()), 2))
    info['ball_pct_of_width_min'] = round(float(2 * R / WID.max() * 100), 1)
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
