"""Mozart Symphony No. 40 - Night city chase. Side view of a hillside city at night.
Red marble A and blue marble B chase each other over the rooftops: every violin I note = one of them landing on a glowing
neon key (or the edge tile of a roof it then rolls along); every viola note = the night train rolling over a rail bar of
the elevated railway; every cello/bass note = the bronze street ball on a lamp-post bell or a bar of its tram rail.
layout.py plans all motion; this file builds and animates it.
Fonts: en_bold.ttf / en_serif_bold.ttf = DejaVu Sans Bold / DejaVu Serif Bold (DejaVu fonts license),
jp_sub.otf = Noto Sans CJK JP Black (SIL OFL) subset to the characters used here (make_font.py).
python3 scene.py -- test t1 t2 ... | range f0 f1 | pieces"""
import os, sys, json, math, time
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); MEL = json.load(open(f'{HERE}/mel.json'))
ACC = json.load(open(f'{HERE}/acc.json')); BASS = json.load(open(f'{HERE}/bass.json'))
LAY = json.load(open(f'{HERE}/layout.json'))
FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.3) * FPS); T = np.arange(NF) / FPS
G = LAY['G']; EV = LAY['ev']; LN = LAY['lanes']; RR = LAY['R']
GP, FORTE, T0, Q = E['gp'], E['forte'], E['t0'], E['q']
BT = {int(k): v for k, v in E['bt'].items()}
BLACK0, BLACK1 = BT[42] + 2 * Q + 0.25, BT[293]           # the general pause: blackout
rng = np.random.default_rng(40)

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
FONT = bpy.data.fonts.load(f'{HERE}/jp_sub.otf')
FONT_EN = bpy.data.fonts.load(f'{HERE}/en_bold.ttf'); FONT_ENS = bpy.data.fonts.load(f'{HERE}/en_serif_bold.ttf')
TXT_K = 2.75
EN_K = 0.75


def link(o): sc.collection.objects.link(o); return o


def mat(name, col, rough=0.5, **kw):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']; b.inputs['Base Color'].default_value = (*col, 1); b.inputs['Roughness'].default_value = rough
    for k, v in kw.items():
        if k == 'Emission Color': v = (*v, 1)
        b.inputs[k].default_value = v
    return m


def emis(name, col, s):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    e = nt.nodes.new('ShaderNodeEmission'); e.inputs['Color'].default_value = (*col, 1); e.inputs['Strength'].default_value = s
    o = nt.nodes.new('ShaderNodeOutputMaterial'); nt.links.new(e.outputs[0], o.inputs[0]); return m


def glowmat(name, gain=8.0, base=0.0, rough=0.3, metal=0.0, coat=0.5):
    """object colour = base colour; object alpha drives emission (hit flash)"""
    m = mat(name, (1, 1, 1), rough, **{'Coat Weight': coat, 'Metallic': metal}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    oi = nt.nodes.new('ShaderNodeObjectInfo')
    nt.links.new(oi.outputs['Color'], b.inputs['Base Color']); nt.links.new(oi.outputs['Color'], b.inputs['Emission Color'])
    mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY_ADD'; mul.inputs[1].default_value = gain; mul.inputs[2].default_value = base
    nt.links.new(oi.outputs['Alpha'], mul.inputs[0]); nt.links.new(mul.outputs[0], b.inputs['Emission Strength'])
    return m


def fxmat(name, gain=10.0):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    oi = nt.nodes.new('ShaderNodeObjectInfo'); em = nt.nodes.new('ShaderNodeEmission'); tr_ = nt.nodes.new('ShaderNodeBsdfTransparent')
    mx = nt.nodes.new('ShaderNodeMixShader'); mu = nt.nodes.new('ShaderNodeMath'); mu.operation = 'MULTIPLY'; mu.inputs[1].default_value = gain
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(oi.outputs['Color'], em.inputs['Color']); nt.links.new(oi.outputs['Alpha'], mu.inputs[0]); nt.links.new(mu.outputs[0], em.inputs['Strength'])
    nt.links.new(oi.outputs['Alpha'], mx.inputs['Fac']); nt.links.new(tr_.outputs[0], mx.inputs[1]); nt.links.new(em.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs[0]); return m


def key_series(obj, path, idx, vals, eps=1e-5):
    n = len(vals)
    for f in range(n):
        v = vals[f]
        if 0 < f < n - 1 and abs(v - vals[f - 1]) < eps and abs(v - vals[f + 1]) < eps: continue
        getattr(obj, path)[idx] = float(v); obj.keyframe_insert(path, index=idx, frame=f)


def key_loc(o, P):
    for k in range(3): key_series(o, 'location', k, P[:, k])


_MC = {}
def prim(kind, m, *a):
    key = (kind, m.name if m else None) + tuple(a)
    if key in _MC: return _MC[key]
    bm = bmesh.new()
    if kind == 'cube': bmesh.ops.create_cube(bm, size=1.0)
    elif kind == 'bcube':
        bmesh.ops.create_cube(bm, size=1.0); bmesh.ops.bevel(bm, geom=bm.edges[:] + bm.verts[:], offset=0.18, segments=3, affect='EDGES')
    elif kind == 'cyl': bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=1.0, radius2=1.0, depth=1.0)
    elif kind == 'cone': bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=a[1], radius2=a[2], depth=a[3])
    elif kind == 'sph': bmesh.ops.create_uvsphere(bm, u_segments=a[0], v_segments=a[1], radius=1.0)
    elif kind == 'plane': bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    elif kind == 'bell':                                   # bell, mouth down, crown flat at z = 0, height 1, mouth radius 1
        prof = [(0.0, 0.0), (0.42, 0.0), (0.5, -0.08), (0.55, -0.35), (0.62, -0.62), (0.8, -0.86), (1.0, -1.0), (0.96, -1.0), (0.78, -0.88), (0.58, -0.64),
                (0.5, -0.36), (0.44, -0.1), (0.0, -0.1)]
        segs = a[0]; rings = []
        for k in range(segs):
            ph = 2 * math.pi * k / segs
            rings.append([bm.verts.new((r * math.cos(ph), r * math.sin(ph), z)) for r, z in prof])
        for k in range(segs):
            r0, r1 = rings[k], rings[(k + 1) % segs]
            for j in range(len(prof) - 1): bm.faces.new((r0[j], r1[j], r1[j + 1], r0[j + 1]))
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    me = bpy.data.meshes.new(kind); bm.to_mesh(me); bm.free()
    if kind in ('sph', 'cone', 'bell', 'cyl', 'bcube'):
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


def _latin(s): return all(ord(c) < 0x2E80 for c in s)


def text(s, loc, size, m, rot=(math.pi / 2, 0, 0), align='CENTER', ext=0.02, font=None):
    cu = bpy.data.curves.new('t', 'FONT'); cu.body = s; cu.font = font or FONT; cu.size = size * TXT_K; cu.extrude = ext
    if _latin(s) and cu.font == FONT: cu.font = FONT_EN; cu.size *= EN_K
    elif _latin(s) and cu.font == FONT_ENS: cu.size *= EN_K
    cu.align_x = align; cu.align_y = 'CENTER'
    o = link(bpy.data.objects.new('txt', cu)); o.location = loc; o.rotation_euler = rot; o.data.materials.append(m); return o


def fit_w(o, w):
    bpy.context.view_layer.update()
    if w and o.dimensions.x > w: k = w / o.dimensions.x; o.scale = (k, k, k)
    return o


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


def seg_at(g, t):
    if g['k'] == 'rest': return np.array(g['p'])
    if g['k'] == 'fly':
        u = t - g['t0']; return np.array(g['p']) + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
    ts = np.array(g['ts']); ps = np.array(g['ps'])
    return np.array([np.interp(t, ts, ps[:, c]) for c in range(3)])


class BallPath:
    def __init__(s, d): s.seg = d['seg']; s.hits = d['hits']; s.r = d['r']
    def at(s, t):
        for g in s.seg:
            if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9: return seg_at(g, t)
        return seg_at(s.seg[0], s.seg[0]['t0']) if t < s.seg[0]['t0'] else seg_at(s.seg[-1], s.seg[-1]['t1'])
BP = {n: BallPath(d) for n, d in LAY['balls'].items()}
POS = {n: np.array([b.at(t) for t in T]) for n, b in BP.items()}


def smooth(Xa, k):
    Y = Xa.copy(); a = 1 - math.exp(-1 / FPS / k)
    for f in range(1, len(Xa)): Y[f] = Y[f - 1] + (Xa[f] - Y[f - 1]) * a
    for f in range(len(Xa) - 2, -1, -1): Y[f] = Y[f + 1] + (Y[f] - Y[f + 1]) * a
    return Y


def strip_mesh(X, Z, y0, y1, th, m, name, bottom=None):
    """an extruded profile: top edge (X, Z), from y0 to y1, `th` thick (or down to `bottom`)"""
    bm = bmesh.new(); vs = []
    for x, z in zip(X, Z):
        zb = (z - th) if bottom is None else bottom
        vs.append([bm.verts.new((x, y0, z)), bm.verts.new((x, y1, z)), bm.verts.new((x, y1, zb)), bm.verts.new((x, y0, zb))])
    for a, b in zip(vs, vs[1:]):
        for k in range(4): bm.faces.new((a[k], b[k], b[(k + 1) % 4], a[(k + 1) % 4]))
    bm.faces.new(vs[0][::-1]); bm.faces.new(vs[-1])
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new(name, me)); o.data.materials.append(m)
    return o


def flash(o, t, col, n=None, base=0.0, amp=1.0):
    fh = int(round(t * FPS))
    o.color = (*col, base); o.keyframe_insert('color', index=3, frame=max(0, fh - 1))
    for df, a in ((0, amp), (4, 0.55 * amp), (18, max(base, 0.12))):
        o.color = (*col, a); o.keyframe_insert('color', index=3, frame=fh + df)
    if n is not None:
        b0 = o.location.copy(); nv = Vector(n)
        for df, d in ((-1, 0), (1, -0.03), (4, 0.01), (8, 0)):
            o.location = b0 + nv * d; o.keyframe_insert('location', frame=max(0, fh + df))


# ---------------- city lights: global keyed values drive every window (base level, forte wave, blackout) ----------------
LIT0 = np.zeros(NF); LIT1 = np.zeros(NF); FRONT = np.zeros(NF)
xcam_guess = 0.5 * (POS['A'][:, 0] + POS['B'][:, 0])
for f in range(NF):
    t = T[f]
    base = 0.18 + 0.12 * min(1.0, t / 30.0)
    if BLACK0 <= t < BLACK1: base = 0.0
    elif t >= BLACK1: base = min(0.55, 0.3 * (t - BLACK1) / (FINAL - BLACK1) + 0.12) if t < FINAL else min(0.5, 0.42 + 0.3 * (t - FINAL))
    LIT0[f] = base
    LIT1[f] = 0.0 if (t < FORTE or BLACK0 <= t) else 0.28
    FRONT[f] = -1e3 if t < FORTE else float(np.interp(FORTE, T, xcam_guess)) - 6.0 + 22.0 * (t - FORTE)
FRONT = np.minimum(FRONT, 1e4)
WINM = bpy.data.materials.new('windows'); WINM.use_nodes = True
_nt = WINM.node_tree; _b = _nt.nodes['Principled BSDF']; _b.inputs['Base Color'].default_value = (0.035, 0.04, 0.07, 1); _b.inputs['Roughness'].default_value = 0.6
_geo = _nt.nodes.new('ShaderNodeNewGeometry')
_sep = _nt.nodes.new('ShaderNodeSeparateXYZ'); _nt.links.new(_geo.outputs['Position'], _sep.inputs[0])
def _m(op, a, b=None, v2=None):
    n = _nt.nodes.new('ShaderNodeMath'); n.operation = op
    if isinstance(a, (int, float)): n.inputs[0].default_value = a
    else: _nt.links.new(a, n.inputs[0])
    if b is not None:
        if isinstance(b, (int, float)): n.inputs[1].default_value = b
        else: _nt.links.new(b, n.inputs[1])
    return n.outputs[0]
WX, WZ = 0.36, 0.46
gx = _m('DIVIDE', _sep.outputs['X'], WX); gy = _m('DIVIDE', _sep.outputs['Y'], WX); gz = _m('DIVIDE', _sep.outputs['Z'], WZ)
fx_ = _m('FRACT', _m('ADD', gx, gy)); fz_ = _m('FRACT', gz)
mx_ = _m('MULTIPLY', _m('GREATER_THAN', fx_, 0.2), _m('LESS_THAN', fx_, 0.78))
mz_ = _m('MULTIPLY', _m('GREATER_THAN', fz_, 0.25), _m('LESS_THAN', fz_, 0.8))
mask = _m('MULTIPLY', mx_, mz_)
_cmb = _nt.nodes.new('ShaderNodeCombineXYZ')
_nt.links.new(_m('FLOOR', _m('ADD', gx, gy)), _cmb.inputs[0]); _nt.links.new(_m('FLOOR', gz), _cmb.inputs[1])
_wn = _nt.nodes.new('ShaderNodeTexWhiteNoise'); _wn.noise_dimensions = '3D'; _nt.links.new(_cmb.outputs[0], _wn.inputs['Vector'])
_v0 = _nt.nodes.new('ShaderNodeValue'); _v0.name = 'LIT0'; _v1 = _nt.nodes.new('ShaderNodeValue'); _v1.name = 'LIT1'
_vf = _nt.nodes.new('ShaderNodeValue'); _vf.name = 'FRONT'
wave = _m('MINIMUM', _m('MAXIMUM', _m('MULTIPLY', _m('SUBTRACT', _vf.outputs[0], _sep.outputs['X']), 0.5), 0.0), 1.0)
litf = _m('ADD', _v0.outputs[0], _m('MULTIPLY', _v1.outputs[0], wave))
_sepc = _nt.nodes.new('ShaderNodeSeparateColor'); _nt.links.new(_wn.outputs['Color'], _sepc.inputs[0])
lit = _m('LESS_THAN', _sepc.outputs[0], litf)
em = _m('MULTIPLY', mask, lit)
_ramp = _nt.nodes.new('ShaderNodeValToRGB'); _nt.links.new(_sepc.outputs[1], _ramp.inputs[0])
_ramp.color_ramp.elements[0].color = (1.0, 0.62, 0.28, 1); _ramp.color_ramp.elements[1].color = (0.75, 0.88, 1.0, 1)
_nt.links.new(_ramp.outputs[0], _b.inputs['Emission Color'])
_nt.links.new(_m('MULTIPLY', em, 3.2), _b.inputs['Emission Strength'])
for f in range(0, NF, 2):
    for node, arr in ((_v0, LIT0), (_v1, LIT1), (_vf, FRONT)):
        node.outputs[0].default_value = float(arr[f]); node.outputs[0].keyframe_insert('default_value', frame=f)

# ---------------- materials ----------------
M_RED = mat('red', (0.95, 0.06, 0.05), 0.08, **{'Coat Weight': 1.0, 'Emission Color': (1.0, 0.15, 0.1), 'Emission Strength': 0.6})
M_BLUE = mat('blue', (0.05, 0.3, 1.0), 0.08, **{'Coat Weight': 1.0, 'Emission Color': (0.2, 0.5, 1.0), 'Emission Strength': 0.6})
M_TRAINB = mat('trainball', (1.0, 0.85, 0.4), 0.15, **{'Metallic': 0.6, 'Emission Color': (1.0, 0.85, 0.5), 'Emission Strength': 1.2})
M_BRONZEB = mat('bronzeball', (0.85, 0.5, 0.2), 0.2, **{'Metallic': 0.9, 'Coat Weight': 0.6})
M_KEY = glowmat('key', gain=10.0, base=0.3, rough=0.08, coat=1.0)
M_BELL = glowmat('bell', gain=5.0, base=0.0, rough=0.25, metal=0.85)
M_RAILB = glowmat('railbar', gain=8.0, base=0.15, rough=0.2, metal=0.4)
M_STEEL = mat('steel', (0.55, 0.58, 0.62), 0.3, **{'Metallic': 0.9})
M_DSTEEL = mat('dsteel', (0.12, 0.13, 0.15), 0.4, **{'Metallic': 0.8})
M_ROOF = mat('roof', (0.09, 0.09, 0.11), 0.7)
M_ROOFRIM = mat('roofrim', (0.25, 0.25, 0.28), 0.5)
M_ASPH = mat('asphalt', (0.03, 0.03, 0.035), 0.35, **{'Coat Weight': 0.6})
M_LINE = emis('roadline', (1.0, 0.85, 0.5), 0.6)
M_LAMP = emis('lamp', (1.0, 0.8, 0.5), 18.0)
M_NEONR = emis('neonr', (1.0, 0.15, 0.3), 9.0); M_NEONB = emis('neonb', (0.2, 0.6, 1.0), 9.0); M_NEONY = emis('neony', (1.0, 0.8, 0.2), 8.0)
M_FX = fxmat('fx', 8.0); M_TRAIL = fxmat('trail', 4.0)
M_TXT = emis('txtw', (1.0, 0.97, 0.92), 5.0); M_TXT_Y = emis('txty', (1.0, 0.82, 0.35), 7.0)
M_TXT_R = emis('txtr', (1.0, 0.3, 0.35), 7.0); M_TXT_B = emis('txtb', (0.35, 0.7, 1.0), 7.0)
M_BOARD = mat('board', (0.02, 0.02, 0.03), 0.4, **{'Coat Weight': 0.5})
M_GOLD = mat('gold', (1.0, 0.72, 0.3), 0.22, **{'Metallic': 0.95})

PITCH_COL = [(1.0, 0.12, 0.12), (1.0, 0.35, 0.05), (1.0, 0.6, 0.05), (1.0, 0.85, 0.1), (0.55, 1.0, 0.1), (0.1, 1.0, 0.3),
             (0.05, 1.0, 0.75), (0.05, 0.7, 1.0), (0.15, 0.35, 1.0), (0.45, 0.2, 1.0), (0.75, 0.2, 1.0), (1.0, 0.2, 0.65)]
def pcol(p, light=0.0):
    c = PITCH_COL[p % 12]; return tuple(min(1, x * (1 - light) + light) for x in c)

# ---------------- balls + glowing trails ----------------
BALL = {}
BMAT = {'A': M_RED, 'B': M_BLUE, 'train': M_TRAINB, 'bass': M_BRONZEB}
TRCOL = {'A': (1.0, 0.2, 0.15), 'B': (0.25, 0.55, 1.0), 'train': (1.0, 0.85, 0.45), 'bass': (1.0, 0.6, 0.25)}
for n, b in BP.items():
    rr = b.r
    bpy.ops.mesh.primitive_uv_sphere_add(radius=rr, segments=32, ring_count=16); o = bpy.context.object; o.name = n
    for p in o.data.polygons: p.use_smooth = True
    o.data.materials.append(BMAT[n])
    key_loc(o, POS[n])
    sp = np.zeros(NF)
    for f in range(1, NF): sp[f] = sp[f - 1] + (POS[n][f, 0] - POS[n][f - 1, 0]) / rr
    o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, sp)
    BALL[n] = o
    nt_ = 7 if n in ('A', 'B') else 4
    for j in range(1, nt_ + 1):
        g_ = sphere((0, 0, 0), rr * (1 - 0.1 * j), M_TRAIL, (16, 8), 'trail')
        lag = j * 0.03
        P = np.array([b.at(max(0.0, t - lag)) for t in T]); key_loc(g_, P)
        sp_ = np.linalg.norm(np.gradient(P, axis=0), axis=1) * FPS
        g_.color = (*TRCOL[n], 0); key_series(g_, 'color', 3, np.clip((sp_ - 0.4) / 1.5, 0, 1) * 0.5 * (1 - j / (nt_ + 1.5)), 2e-3)
for n, col, e_ in (('A', (1.0, 0.3, 0.25), 25), ('B', (0.3, 0.55, 1.0), 25), ('train', (1.0, 0.85, 0.55), 40)):
    l_ = bpy.data.lights.new('l' + n, 'POINT'); l_.color = col; l_.energy = e_; l_.shadow_soft_size = 0.2
    lo = link(bpy.data.objects.new('l' + n, l_)); lo.parent = BALL[n]
# the train's headlight: a spot riding ahead of the ball
hl = bpy.data.lights.new('headlight', 'SPOT'); hl.color = (1.0, 0.9, 0.7); hl.energy = 300; hl.spot_size = math.radians(30); hl.spot_blend = 0.6
hlo = link(bpy.data.objects.new('headlight', hl)); hlo.location = tuple(np.array(EV['train']['station']) + [0.2, 0, 0])
key_loc(hlo, POS['train'] + np.array([0.15, 0, 0.05])); hlo.rotation_euler = (0, math.radians(90), 0)

PIECES = []
# ---------------- melody keys (neon tiles on posts) ----------------
KEY_W, KEY_D, KEY_TH = 0.32, 0.48, 0.08
KEYS = {'A': [], 'B': []}
for n in ('A', 'B'):
    for h in BP[n].hits:
        if h['kind'] != 'key' or h['pitch'] is None: continue
        nv = np.array(h['n']); top = np.array(h['p']) - RR[n] * nv
        col = pcol(h['pitch'], 0.15)
        o = obj(prim('bcube', M_KEY), 'key')
        o.matrix_world = Matrix.Translation(Vector(top - nv * KEY_TH / 2)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((KEY_W, KEY_D, KEY_TH, 1))
        o.color = (*col, 0.0); flash(o, h['t'], col, nv, base=0.12)
        KEYS[n].append(dict(t=h['t'], top=top, n=nv, col=col))
        PIECES.append(dict(kind='key', t=h['t'], top=top, n=nv, hw=KEY_W / 2, hd=KEY_D / 2, th=KEY_TH / 2, ball=n))

# ---------------- rooftop rows: buildings under each lane (flat roofs under keys, sloped roofs where a ball rolls) ----------------
RAMPS = EV['ramps']
ROW = {'A': (LN['A'] - 0.45, LN['A'] + 0.42), 'B': (LN['B'] - 0.42, LN['B'] + 0.45)}
FACADE = [mat(f'facade{k}', c, 0.6) for k, c in enumerate(((0.07, 0.07, 0.1), (0.1, 0.08, 0.09), (0.06, 0.08, 0.1), (0.09, 0.09, 0.08)))]
ROOFTOPS = {'A': [], 'B': []}
STREET_Z = None
def lane_floor(n):
    """lowest point of the ball over each x (sampled): no roof may come above it"""
    P = POS[n]; xs_ = np.arange(P[:, 0].min() - 3, P[:, 0].max() + 3, 0.1)
    fz = np.full(len(xs_), np.nan)
    for f in range(NF):
        j = int((P[f, 0] - xs_[0]) / 0.1)
        for jj in (j - 2, j - 1, j, j + 1, j + 2, j + 3):
            if 0 <= jj < len(xs_):
                v = P[f, 2] - RR[n]
                fz[jj] = v if np.isnan(fz[jj]) else min(fz[jj], v)
    ok = ~np.isnan(fz); fz = np.interp(xs_, xs_[ok], fz[ok])
    return xs_, fz
for n in ('A', 'B'):
    y0, y1 = ROW[n]
    xs_, fz = lane_floor(n)
    myr = [r for r in RAMPS if r['who'] == n]
    spans = []
    for r in myr:
        d = np.array([math.cos(r['alpha']), 0, -math.sin(r['alpha'])]); nn = np.array(r['n'])
        a = np.array(r['p0']) - RR[n] * nn - d * 0.06; b = np.array(r['lip']) - RR[n] * nn
        spans.append((a[0], b[0], a, b, r))
    # flat-roof buildings between the sloped ones: chunks of 1.6 - 3.2 m
    occupied = sorted([(s[0], s[1]) for s in spans])
    x = xs_[0] + 1.0; xend = xs_[-1] - 1.0; k = 0
    while x < xend:
        w = rng.uniform(1.6, 3.2)
        x1 = min(x + w, xend)
        for a_, b_ in occupied:
            if a_ - 0.05 < x1 and b_ + 0.05 > x:
                if a_ - 0.05 > x + 0.6: x1 = a_ - 0.05
                else: x = b_ + 0.05; x1 = None
                break
        if x1 is None: continue
        m_ = (xs_ >= x) & (xs_ <= x1)
        if m_.sum() < 2: x = x1 + 0.12; continue
        zr = float(fz[m_].min()) - rng.uniform(0.55, 0.85)
        bot = zr - 40
        strip_mesh(np.array([x, x1]), np.array([zr, zr]), y0, y1, 0.0, FACADE[k % 4], 'bldg', bottom=bot); k += 1
        WIN_ = strip_mesh(np.array([x + 0.02, x1 - 0.02]), np.array([zr - 0.35, zr - 0.35]), y0 - 0.01, y0 - 0.005, 0.0, WINM, 'winA', bottom=bot)
        box(((x + x1) / 2, (y0 + y1) / 2, zr + 0.04), (x1 - x, y1 - y0 + 0.04, 0.08), M_ROOFRIM)
        ROOFTOPS[n].append((x, x1, zr))
        x = x1 + rng.uniform(0.1, 0.35)
    for a_x, b_x, a, b, r in spans:
        X = np.array([a[0], b[0]]); Z = np.array([a[2], b[2]])
        strip_mesh(X, Z, y0, y1, 0.0, FACADE[1], 'rampbldg', bottom=min(Z) - 40)
        strip_mesh(X + np.array([0.02, -0.02]), Z - 0.35, y0 - 0.01, y0 - 0.005, 0.0, WINM, 'winR', bottom=min(Z) - 40)
        strip_mesh(X, Z + 0.001, y0, y1, 0.06, M_ROOF if r['kind'] == 'roof' else M_DSTEEL, 'slope')
        for yy in (y0 + 0.02, y1 - 0.02):
            strip_mesh(X, Z + 0.09, yy - 0.02, yy + 0.02, 0.09, M_NEONR if n == 'A' else M_NEONB, 'rail')
        # the landing tile of a roof roll (that landing is a note)
        for bar in r['bars']:
            nn = np.array(r['n']); top = np.array(bar['p']) - RR[n] * nn
            col = pcol(bar['pitch'], 0.15)
            if r['kind'] == 'roof':
                o = obj(prim('bcube', M_KEY), 'rooftile')
                o.matrix_world = Matrix.Translation(Vector(top - nn * 0.03)) @ frame_from_n(nn).to_4x4() @ Matrix.Diagonal((0.16, y1 - y0 - 0.08, 0.06, 1))
                PIECES.append(dict(kind='rooftile', t=bar['t'], top=top, n=nn, hw=0.08, hd=(y1 - y0) / 2 - 0.04, th=0.03, ball=n))
            else:                                               # xylophone fire escape bars
                o = obj(prim('bcube', M_KEY), 'xbar')
                o.matrix_world = Matrix.Translation(Vector(top - nn * 0.03)) @ frame_from_n(nn).to_4x4() @ Matrix.Diagonal((0.1, y1 - y0 - 0.1, 0.06, 1))
                PIECES.append(dict(kind='xbar', t=bar['t'], top=top, n=nn, hw=0.05, hd=(y1 - y0) / 2 - 0.05, th=0.03, ball=n))
            o.color = (*col, 0.0); flash(o, bar['t'], col, base=0.15)
        if r['kind'] == 'xylo':
            q = (a + b) / 2
            text('Fire-escape Xylophone', (q[0], y0 - 0.06, q[2] + 0.75), 0.22, M_TXT_Y)
# posts from each key down to its roof
def roof_under(n, x):
    for x0, x1, zr in ROOFTOPS[n]:
        if x0 <= x <= x1: return zr
    return None
for n in ('A', 'B'):
    for k_ in KEYS[n]:
        top = k_['top']; zr = roof_under(n, top[0])
        zr = zr if zr is not None else top[2] - 1.2
        rod((top[0], top[1], top[2] - KEY_TH - 0.01), (top[0], top[1], zr), 0.03, M_STEEL, 8)
        box((top[0], top[1], zr + 0.05), (0.18, 0.18, 0.1), M_DSTEEL)

# ---------------- the street (front): asphalt, lamp posts with the bass bells, the tram rail ----------------
xsS, fzS = lane_floor('bass')
STREET = np.minimum.accumulate(np.round((fzS - 1.5) / 0.25) * 0.25)
def street_z(x): return float(np.interp(x, xsS, STREET))
yS = LN['bass']
strip_mesh(xsS[::5], STREET[::5], yS - 1.2, ROW['A'][0], 0.3, M_ASPH, 'street')
for x0 in np.arange(xsS[0], xsS[-1], 1.6):
    zz = street_z(x0) + 0.005
    box((x0, yS - 0.55, zz), (0.7, 0.06, 0.01), M_LINE)
for h in BP['bass'].hits:
    if h['kind'] != 'key' or h['pitch'] is None: continue
    nv = np.array(h['n']); top = np.array(h['p']) - RR['bass'] * nv; p = h['pitch']
    sz = 0.38 + 0.008 * (50 - p)
    o = obj(prim('bell', M_BELL, 32), 'bell')
    o.matrix_world = Matrix.Translation(Vector(top)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((sz * 0.75, sz * 0.75, sz, 1))
    o.color = (1.0, 0.62, 0.25, 0); flash(o, h['t'], (1.0, 0.65, 0.3), base=0.05)
    zb = street_z(top[0])
    rod(top - nv * sz, (top[0], top[1], zb), 0.045, M_DSTEEL, 10)
    rod((top[0], top[1], top[2] - sz - 0.25), (top[0] + 0.35, top[1] - 0.1, top[2] - sz - 0.05), 0.02, M_DSTEEL, 6)
    sphere((top[0] + 0.38, top[1] - 0.1, top[2] - sz - 0.1), 0.07, M_LAMP, (10, 6))
    PIECES.append(dict(kind='bell', t=h['t'], top=top, n=nv, hw=0.42 * sz * 0.75, hd=0.42 * sz * 0.75, th=0.05, ball='bass'))
for r in [r for r in RAMPS if r['who'] == 'bass']:
    d = np.array([math.cos(r['alpha']), 0, -math.sin(r['alpha'])]); nn = np.array(r['n'])
    a = np.array(r['p0']) - RR['bass'] * nn - d * 0.4; b = np.array(r['lip']) - RR['bass'] * nn + d * 0.1
    y0 = a[1]
    for yy in (y0 - 0.26, y0 + 0.26): rod((a[0], yy, a[2] + 0.04), (b[0], yy, b[2] + 0.04), 0.025, M_STEEL, 8)
    for s in np.arange(-0.4, r['L'] + 0.1, 1.4):
        q = np.array(r['p0']) - RR['bass'] * nn + d * s
        for yy in (y0 - 0.26, y0 + 0.26): rod((q[0], yy, q[2]), (q[0], yy, street_z(q[0])), 0.03, M_DSTEEL, 6)
    for bar in r['bars']:
        top = np.array(bar['p']) - RR['bass'] * nn
        o = obj(prim('bcube', M_BELL), 'tbar')
        o.matrix_world = Matrix.Translation(Vector(top - nn * 0.03)) @ frame_from_n(nn).to_4x4() @ Matrix.Diagonal((0.1, 0.62, 0.06, 1))
        o.color = (1.0, 0.65, 0.3, 0.0); flash(o, bar['t'], (1.0, 0.7, 0.35), base=0.05)
        PIECES.append(dict(kind='tbar', t=bar['t'], top=top, n=nn, hw=0.05, hd=0.31, th=0.03, ball='bass'))
    q = np.array(r['p0']) + d * min(2.0, r['L'] * 0.3)
    text('Tram of Basses', (q[0], y0 - 0.45, q[2] + 0.75), 0.2, M_TXT_Y)

# ---------------- the elevated railway (train): rails, sleepers, pylons, note bars ----------------
TRv = EV['train']; aT = TRv['alpha']; dT = np.array([math.cos(aT), 0, -math.sin(aT)]); nT = np.array([math.sin(aT), 0, math.cos(aT)])
r0 = np.array(TRv['rail0']); st = np.array(TRv['station']); aE = TRv['alpha_e']; dE = np.array([math.cos(aE), 0, -math.sin(aE)])
RT = RR['train']; yT = LN['train']
L_rail = TRv['v'] * (TRv['t_end'] - BP['train'].hits[0]['t']) + 2.0
rail_a = r0 - RT * nT; rail_b = r0 + dT * L_rail - RT * nT
st_a = st - RT * np.array([math.sin(aE), 0, math.cos(aE)]) - dE * 0.4
for yy in (yT - 0.16, yT + 0.16):
    rod((st_a[0], yy, st_a[2] + 0.03), (rail_a[0], yy, rail_a[2] + 0.03), 0.022, M_STEEL, 8)
    rod((rail_a[0], yy, rail_a[2] + 0.03), (rail_b[0], yy, rail_b[2] + 0.03), 0.022, M_STEEL, 8)
for s in np.arange(0, L_rail, 0.9):
    q = rail_a + dT * s
    box(tuple(q - nT * 0.05), (0.08, 0.5, 0.05), M_DSTEEL)
    if (int(s / 0.9)) % 5 == 0:
        rod((q[0], yT, q[2] - 0.08), (q[0], yT, q[2] - 30), 0.09, M_DSTEEL, 10)
        rod((q[0], yT - 0.3, q[2] - 0.12), (q[0], yT + 0.3, q[2] - 0.12), 0.05, M_DSTEEL, 6)
strip_mesh(np.array([rail_a[0], rail_b[0]]), np.array([rail_a[2] - 0.1, rail_b[2] - 0.1]), yT - 0.3, yT + 0.3, 0.08, M_DSTEEL, 'girder')
# the station (start): a small platform with a lit canopy
box(tuple(st + np.array([-0.6, 0, -RT - 0.1])), (1.6, 1.0, 0.2), M_DSTEEL)
box(tuple(st + np.array([-0.6, 0, 1.2])), (1.8, 1.1, 0.06), M_DSTEEL)
for dx in (-1.3, 0.1): rod(st + np.array([dx, -0.45, -0.1]), st + np.array([dx, -0.45, 1.2]), 0.03, M_STEEL, 6)
box(tuple(st + np.array([-0.6, -0.55, 1.05])), (1.6, 0.04, 0.1), M_NEONY)
text('NIGHT LINE', tuple(st + np.array([-0.6, -0.6, 1.45])), 0.2, M_TXT_Y)
done = {}
for h in BP['train'].hits:
    k = h['ev']
    done.setdefault(k, []).append(h)
for k, hs in done.items():
    ps = sorted(h['pitch'] for h in hs); pc = np.array(hs[0]['p']); top = pc - RT * nT
    if len(ps) == 1: parts = [(ps[0], 0.0, 0.5)]
    else: parts = [(ps[0], -0.13, 0.24), (ps[1], 0.13, 0.24)]
    for p, dy, w in parts:
        o = obj(prim('bcube', M_RAILB), 'rbar')
        o.matrix_world = Matrix.Translation(Vector(top - nT * 0.025 + np.array([0, dy, 0]))) @ frame_from_n(nT).to_4x4() @ Matrix.Diagonal((0.09, w, 0.05, 1))
        col = pcol(p, 0.3); o.color = (*col, 0.0); flash(o, hs[0]['t'], col, base=0.12)
    PIECES.append(dict(kind='rbar', t=hs[0]['t'], top=top, n=nT, hw=0.045, hd=0.25, th=0.025, ball='train'))

# ---------------- the final rooftop: the bowl bell where the two meet ----------------
bw = EV['bowl']; bc = np.array([bw['x'], bw['y'], bw['z']])
bowl = obj(prim('bell', M_BELL, 48), 'bowl')
bowl.matrix_world = Matrix.Translation(Vector(bc + [0, 0, 0.0])) @ Matrix.Diagonal((bw['r'], bw['r'], -0.32, 1))
bowl.color = (1.0, 0.75, 0.35, 0.05); key_series(bowl, 'color', 3, np.maximum(0.06, pulse_alpha([FINAL], decay=45, length=150)), 1e-3)
for pp in (bw['pa'], bw['pb']):
    PIECES.append(dict(kind='bowl', t=FINAL, top=np.array(pp) - [0, 0, RR['A']], n=np.array([0, 0, 1.0]), hw=0.3, hd=0.3, th=0.05, ball='A' if pp == bw['pa'] else 'B'))
zr_b = bc[2] - 0.45
rod(bc - [0, 0, 0.32], (bc[0], bc[1], zr_b), 0.08, M_GOLD, 12)
box((bc[0], bc[1], zr_b - 0.05), (3.4, 2.2, 0.1), M_ROOFRIM)
strip_mesh(np.array([bc[0] - 1.7, bc[0] + 1.7]), np.array([zr_b - 0.1, zr_b - 0.1]), bc[1] - 1.1, bc[1] + 1.1, 0.0, FACADE[2], 'finalbldg', bottom=zr_b - 40)
strip_mesh(np.array([bc[0] - 1.68, bc[0] + 1.68]), np.array([zr_b - 0.45, zr_b - 0.45]), bc[1] - 1.11, bc[1] - 1.105, 0.0, WINM, 'winF', bottom=zr_b - 40)
for dx in np.linspace(-1.6, 1.6, 9):
    sphere((bc[0] + dx, bc[1] - 1.1, zr_b + 0.05), 0.05, M_NEONY, (8, 6))

# ---------------- the skyline behind: towers with windows, rooftop neon, antenna lights ----------------
XMIN = min(POS['A'][:, 0].min(), POS['train'][:, 0].min()) - 8; XMAX = max(POS['A'][:, 0].max(), POS['B'][:, 0].max()) + 10
def zcity(x): return float(np.interp(x, POS['A'][:, 0], POS['A'][:, 2]))
for row, (yb, hmin, hmax, wmin, wmax) in enumerate(((3.6, 1.5, 6.0, 1.6, 3.0), (6.5, 4.0, 11.0, 2.5, 4.5), (11.0, 7.0, 18.0, 3.5, 6.0))):
    x = XMIN
    while x < XMAX:
        w = rng.uniform(wmin, wmax); h = rng.uniform(hmin, hmax); zt_ = zcity(x + w / 2) + h
        yy = yb + rng.uniform(0, 1.5)
        strip_mesh(np.array([x, x + w]), np.array([zt_, zt_]), yy, yy + w * 0.8, 0.0, FACADE[row % 4], 'tower', bottom=zt_ - 60)
        strip_mesh(np.array([x + 0.05, x + w - 0.05]), np.array([zt_ - 0.4, zt_ - 0.4]), yy - 0.01, yy - 0.005, 0.0, WINM, 'towerwin', bottom=zt_ - 60)
        if rng.uniform() < 0.35:
            rod((x + w / 2, yy + 0.4, zt_), (x + w / 2, yy + 0.4, zt_ + rng.uniform(0.8, 2.0)), 0.03, M_DSTEEL, 6)
            sphere((x + w / 2, yy + 0.4, zt_ + 2.0), 0.06, M_NEONR, (8, 6))
        if rng.uniform() < 0.3:
            box((x + w / 2, yy - 0.03, zt_ - 0.2), (w * 0.7, 0.04, 0.08), (M_NEONR, M_NEONB, M_NEONY)[int(rng.integers(3))])
        x += w + rng.uniform(0.2, 1.2)

# ---------------- forte (bar 28): searchlights sweep the sky; finale fireworks ----------------
BEAMS = []
for k in range(4):
    xb = float(np.interp(FORTE, T, xcam_guess)) + (-4.5 + 3.0 * k)
    zb_ = zcity(xb) - 1.0
    piv = link(bpy.data.objects.new('beampiv', None)); piv.location = (xb, 8.0, zb_)
    cn = obj(prim('cone', M_FX, 24, 0.05, 1.2, 1.0), 'beam'); cn.parent = piv; cn.location = (0, 0, 6.0); cn.scale = (1, 1, 12.0)
    cn.color = (0.85, 0.9, 1.0, 0)
    a_ = np.array([0.0 if (t < FORTE or t > BLACK0) else 0.12 * min(1.0, (t - FORTE) / 0.3) for t in T])
    key_series(cn, 'color', 3, a_, 1e-3)
    ang = np.array([0.5 * math.sin(0.9 * (t - FORTE) + k * 1.3) for t in T])
    piv.rotation_mode = 'XYZ'; key_series(piv, 'rotation_euler', 1, ang, 1e-3)
FW = []
FWC = [(1.0, 0.3, 0.3), (0.3, 0.6, 1.0), (1.0, 0.85, 0.35), (1.0, 0.45, 0.9), (0.4, 1.0, 0.6)]
for k in range(7):
    t0 = FINAL + 0.15 + 0.42 * k; FW.append(t0)
    c0 = bc + np.array([rng.uniform(-3.5, 3.5), 4.5 + rng.uniform(0, 2.0), rng.uniform(3.5, 6.5)])
    col = FWC[k % 5]
    for j in range(26):
        s_ = sphere((0, 0, 0), 0.06, M_FX, (8, 6)); s_.color = (*col, 0)
        th = rng.uniform(0, 2 * math.pi); ph = math.acos(rng.uniform(-1, 1)); sp = rng.uniform(2.0, 3.0)
        v = sp * np.array([math.sin(ph) * math.cos(th), math.sin(ph) * math.sin(th) * 0.4, math.cos(ph)])
        P = np.zeros((NF, 3)); A_ = np.zeros(NF)
        for f in range(NF):
            u = T[f] - t0
            P[f] = c0 + (v * u + np.array([0, 0, -0.3 * G * u * u]) if u > 0 else 0)
            A_[f] = 0 if u < 0 or u > 1.6 else (1 - u / 1.6) ** 1.2
        key_loc(s_, P); key_series(s_, 'color', 3, A_, 1e-3)
EV['fireworks'] = FW
for k in range(160):                                           # confetti over the bowl
    cf = obj(prim('plane', M_FX), 'conf'); cf.color = (*FWC[k % 5], 1)
    a0 = rng.uniform(0, 2 * math.pi); spd = rng.uniform(1.0, 3.2)
    P = np.zeros((NF, 3)); RT_ = np.zeros((NF, 3))
    for f in range(NF):
        s_ = T[f] - FINAL
        if s_ < 0: P[f] = bc; continue
        burst = 1 - math.exp(-s_ / 0.3)
        P[f] = (bc[0] + spd * math.cos(a0) * 1.4 * burst + 0.15 * math.sin(3 * s_ + k), bc[1] - 0.6 + 0.4 * math.sin(a0) * burst, bc[2] + 0.4 + spd * abs(math.sin(a0)) * 1.0 * burst - 0.5 * s_)
        RT_[f] = (s_ * (3 + k % 5), s_ * (2 + k % 3), s_ * 1.5)
    key_loc(cf, P)
    sc_ = np.where(T >= FINAL, 1.0, 0.001)
    for c_ in range(3): key_series(cf, 'scale', c_, sc_ * 0.07); key_series(cf, 'rotation_euler', c_, RT_[:, c_])
    cf.color = (*FWC[k % 5], 0.0); key_series(cf, 'color', 3, np.where(T >= FINAL, 0.35, 0.0), 1e-3)

# hit sparks: three sparks fly out of every melody key
NSPK = 48; spk_loc = np.zeros((NSPK, NF, 3)); spk_a = np.zeros((NSPK, NF)); spk_c = [(1, 1, 1)] * NSPK
sid = 0
for n in ('A', 'B'):
    for h in BP[n].hits:
        if h['pitch'] is None: continue
        c = np.array(h['p']) - RR[n] * np.array(h['n']); f0 = int(round(h['t'] * FPS))
        for q in range(3):
            ang = rng.uniform(0, 2 * math.pi); sp = rng.uniform(1.2, 2.2)
            v = np.array([sp * math.cos(ang) * 0.8, -rng.uniform(0.3, 0.8), rng.uniform(1.0, 2.2)])
            k = sid % NSPK; sid += 1; spk_c[k] = TRCOL[n]
            for df in range(0, 12):
                f = f0 + df
                if f >= NF: break
                tt = df / FPS; spk_loc[k, f] = c + v * tt + np.array([0, 0, -0.5 * G * tt * tt]); spk_a[k, f] = 1 - df / 12
for k in range(NSPK):
    o = sphere((0, 0, 0), 0.03, M_FX, (6, 4)); o.color = (*spk_c[k], 0)
    for f in range(1, NF):
        if spk_a[k, f] == 0: spk_loc[k, f] = spk_loc[k, f - 1]
    key_loc(o, spk_loc[k]); key_series(o, 'color', 3, spk_a[k], 1e-3)

# ---------------- world: night sky, moon ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bgn = w.node_tree.nodes['Background']; bgn.inputs['Color'].default_value = (0.02, 0.03, 0.08, 1)
for f in range(0, NF, 3):
    t = T[f]; s = 0.35 if not (BLACK0 <= t < BLACK1) else 0.08
    bgn.inputs['Strength'].default_value = s; bgn.inputs['Strength'].keyframe_insert('default_value', frame=f)
mn = bpy.data.lights.new('moon', 'SUN'); mn.color = (0.6, 0.72, 1.0); mn.angle = 0.05
mno = link(bpy.data.objects.new('moon', mn)); mno.rotation_euler = (math.radians(55), math.radians(-15), math.radians(-25))
for f in range(0, NF, 3):
    t = T[f]; mn.energy = float(0.5 if not (BLACK0 <= t < BLACK1) else 0.12); mn.keyframe_insert('energy', frame=f)

# ---------------- camera: follow the two marbles, keep street and train in the shot ----------------
FOC = np.zeros((NF, 3)); WID = np.zeros(NF)
pA, pB = POS['A'], POS['B']
for f in range(NF):
    t = T[f]
    FOC[f] = np.array([0.5 * (pA[f, 0] + pB[f, 0]) + 0.3, 0, 0.5 * (pA[f, 2] + pB[f, 2]) + 0.15])
    WID[f] = 5.7
    if t < T0 + 0.6: WID[f] = 6.8
    if BLACK0 - 0.3 < t < BLACK1 + 0.5: WID[f] = 7.2
    if t > FINAL - 1.2:
        u = min(1, (t - (FINAL - 1.2)) / 1.4)
        FOC[f] = (1 - u) * FOC[f] + u * np.array([bc[0], 0, bc[2] + 1.1]); WID[f] = (1 - u) * WID[f] + u * 6.4
FOC[:, 0] = smooth(FOC[:, 0], 0.45); FOC[:, 2] = smooth(FOC[:, 2], 0.7); WID = smooth(WID, 0.5)
LENS = 32.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 500
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24
    CP[f] = (FOC[f, 0], -D, FOC[f, 2] + 0.12 * D)
    TP[f] = (FOC[f, 0], 0.0, FOC[f, 2] + 0.3)
key_loc(cam, CP); key_loc(tgt, TP)
for k, (dx, dz, e_) in enumerate(((-3.0, 5.5, 450), (3.0, 5.0, 350))):
    al = bpy.data.lights.new(f'fill{k}', 'AREA'); al.size = 6.0; al.color = (0.6, 0.7, 1.0) if k == 0 else (1.0, 0.75, 0.55)
    ao = link(bpy.data.objects.new(f'fill{k}', al))
    key_loc(ao, np.stack([FOC[:, 0] + dx, np.full(NF, -4.0), FOC[:, 2] + dz], 1)); ao.rotation_euler = (math.radians(-30), 0, 0)
    for f in range(0, NF, 3):
        t = T[f]; al.energy = float(e_ * (0.25 if BLACK0 <= t < BLACK1 else 1.0)); al.keyframe_insert('energy', frame=f)

# ---------------- HUD (rides with the camera) ----------------
HUD_D = 3.0
def hud_text(s, x, y, size, m, font=None, align='CENTER'):
    o = text(s, (x, y, -HUD_D), size, m, rot=(0, 0, 0), align=align, ext=0.0, font=font); o.parent = cam; return o
def vis(o, a):
    s0 = o.scale[0]
    for c in range(3): key_series(o, 'scale', c, np.maximum(0.001, a * s0), 1e-3)
M_HUDBG = mat('hudbg', (0.0, 0.0, 0.0), 1.0, **{'Alpha': 0.55})
def window(t0, t1, fade=0.3):
    return np.array([0.0 if t < t0 - fade or t > t1 + fade else min(1.0, (t - t0 + fade) / fade, (t1 + fade - t) / fade) for t in T])
hook = window(0.0, 3.6)
h1 = fit_w(hud_text('Two marbles. One night city.', 0, 0.9, 0.08, M_TXT), 0.92); vis(h1, hook)
h2 = fit_w(hud_text('Will they meet?', 0, 0.79, 0.11, M_TXT_Y), 0.92); vis(h2, hook)
h3 = fit_w(hud_text('2つのビー玉、最後に会える？', 0, 0.69, 0.045, M_TXT), 0.9); vis(h3, hook)
hb_ = obj(prim('plane', M_HUDBG), 'hudbg'); hb_.parent = cam; hb_.location = (0, 0.8, -HUD_D - 0.01); hb_.scale = (1.3, 0.32, 1); vis(hb_, hook)
# who has the melody: a small RED / BLUE tag at the top
own = {}
for m in MEL: own.setdefault(m[4], []).append(m[0])
def playing(who, t):
    ts = own.get(who, []) + own.get('AB', [])
    return any(abs(t - x) < 0.6 for x in ts)
tagA = np.array([1.0 if (playing('A', t) and 3.8 < t < BLACK0) else 0.0 for t in T]); tagA = np.clip(smooth(tagA, 0.08), 0, 1)
tagB = np.array([1.0 if (playing('B', t) and 3.8 < t < BLACK0) else 0.0 for t in T]); tagB = np.clip(smooth(tagB, 0.08), 0, 1)
ta = hud_text('RED', -0.3, 0.9, 0.07, M_TXT_R); vis(ta, 0.35 + 0.65 * tagA * (T > 3.8))
tb_ = hud_text('BLUE', 0.3, 0.9, 0.07, M_TXT_B); vis(tb_, 0.35 + 0.65 * tagB * (T > 3.8))
vs_ = hud_text('vs', 0.0, 0.9, 0.05, M_TXT); vis(vs_, ((T > 3.8) & (T < BLACK0)).astype(float))
for o_ in (ta, tb_):
    a0 = o_.scale[0]
    vis(o_, np.where((T > 3.8) & (T < BLACK0), 1.0, 0.0) * (0.55 + 0.45 * (tagA if o_ is ta else tagB)))
bo = fit_w(hud_text('BLACKOUT!', 0, 0.86, 0.1, M_TXT_Y), 0.9); vis(bo, window(BLACK0 + 0.05, BLACK1 - 0.1, 0.12))
fin = window(FINAL + 0.6, FINAL + 9, 0.3)
f1 = fit_w(hud_text('They met!', 0, 0.9, 0.11, M_TXT_Y), 0.9); vis(f1, fin)
f2 = fit_w(hud_text('Mozart – Symphony No. 40', 0, 0.79, 0.07, M_TXT), 0.92); vis(f2, fin)
f3 = fit_w(hud_text('モーツァルト 交響曲第40番', 0, 0.71, 0.045, M_TXT), 0.9); vis(f3, fin)
fb_ = obj(prim('plane', M_HUDBG), 'hudbg2'); fb_.parent = cam; fb_.location = (0, 0.81, -HUD_D - 0.01); fb_.scale = (1.3, 0.3, 1); vis(fb_, fin)

# ---------------- billboards on the towers behind (keywords, readable on a phone; first 30 s) ----------------
M_BFRAME = mat('bframe', (0.15, 0.15, 0.17), 0.3, **{'Metallic': 0.9})
def billboard(lines, x, w, h, zc, y=3.3, neon=M_NEONY):
    box((x, y + 0.05, zc), (w, 0.06, h), M_BOARD, name='board')
    for sgn in (1, -1):
        box((x, y, zc + sgn * h / 2), (w + 0.1, 0.08, 0.06), neon)
        box((x + sgn * w / 2, y, zc), (0.06, 0.08, h + 0.1), neon)
        rod((x + sgn * w * 0.35, y + 0.1, zc - h / 2), (x + sgn * w * 0.35, y + 0.1, zc - h / 2 - 6), 0.06, M_DSTEEL, 8)
    tot = sum(sz * 1.35 for _, sz, _, _ in lines); zz = zc + tot / 2
    for s_, size, mt, fnt in lines:
        zz -= size * 1.35
        fit_w(text(s_, (x, y - 0.03, zz + size * 0.6), size, mt, font=fnt), w - 0.35)
x_ = lambda t: float(np.interp(t, T, FOC[:, 0]))
bz_ = lambda t: float(np.interp(t, T, FOC[:, 2])) + 2.9
S, J = FONT_ENS, FONT
for tb, w_, h_, lines, neon in ((4.6, 5.6, 2.3, [('Mozart', 0.62, M_TXT_Y, S), ('Symphony No. 40', 0.42, M_TXT, S), ('in G minor, K.550', 0.28, M_TXT, J),
                                                  ('モーツァルト 交響曲第40番 ト短調', 0.19, M_TXT, J)], M_NEONY),
                                 (10.6, 5.2, 1.8, [('Classical Music', 0.42, M_TXT_B, J), ('× Marble Machine', 0.34, M_TXT, J), ('クラシック × ビー玉からくり', 0.19, M_TXT, J)], M_NEONB),
                                 (16.6, 5.0, 1.8, [('Night City Chase', 0.42, M_TXT_R, J), ('Red vs Blue', 0.32, M_TXT, J), ('夜の街の追いかけっこ', 0.19, M_TXT, J)], M_NEONR),
                                 (22.6, 5.0, 1.8, [('Satisfying', 0.42, M_TXT_Y, J), ('Marble ASMR', 0.36, M_TXT, J), ('ビー玉 マーブルマシン', 0.19, M_TXT, J)], M_NEONY),
                                 (28.6, 5.2, 1.8, [('Molto Allegro', 0.42, M_TXT_B, S), ('1st movement · 1788', 0.3, M_TXT, J), ('第1楽章 モルト・アレグロ', 0.19, M_TXT, J)], M_NEONB)):
    billboard(lines, x_(tb) + 0.4, w_, h_, bz_(tb), neon=neon)

EV_OUT = dict(pieces=[dict(kind=p['kind'], t=p['t'], top=list(map(float, p['top'])), n=list(map(float, p['n'])), hw=p['hw'], hd=p['hd'], th=p['th'], ball=p['ball']) for p in PIECES],
              rooftops={k: v for k, v in ROOFTOPS.items()}, street=[xsS.tolist(), STREET.tolist()], fireworks=FW, cam=dict(foc=FOC.tolist(), wid=WID.tolist()))
if MODE == 'pieces': json.dump(EV_OUT, open(f'{HERE}/pieces.json', 'w'))

if MODE == 'test':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS)); sc.frame_set(f)
        sc.render.filepath = f'{HERE}/test_{f:04d}.png'
        t1 = time.time(); bpy.ops.render.render(write_still=True); print('frame', f, 'sec', round(time.time() - t1, 1))
elif MODE == 'range':
    sc.frame_start, sc.frame_end = int(args[1]), int(args[2])
    sc.render.image_settings.file_format = 'PNG'; sc.render.filepath = f'{HERE}/frames/f_'
    bpy.ops.render.render(animation=True)
