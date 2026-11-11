"""Morning Mood - 夜明けの山の村 (a mountain village before dawn), side view along a hillside course down to the fjord.
Every melody note = the flute ball (glass) or the oboe ball (amber) landing on a key of that pitch; every accompaniment
onset = the dawn ball (copper) pushing through a bell gate on the back rail (one bell per chord note). Gates on the ramps open
when the last note of a phrase (a lever key) is struck; the ropeway gondolas carry the resting ball ahead; the sky gets lighter
with every note; the counterweight hauls the sun (a giant gold marble) over the mountains on the final chord.
layout.py plans all motion; this file builds and animates it.
jp_sub.otf = Noto Sans CJK JP Black (SIL OFL) subset to the characters used here (make_font.py).
python3 scene.py -- test t1 t2 ... | range f0 f1"""
import os, sys, json, math, time
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
LAY = json.load(open(f'{HERE}/layout.json'))
FPS = 30; FINAL = E['final']; NF = int((FINAL + 6.0) * FPS); T = np.arange(NF) / FPS
G = LAY['G']; EV = LAY['ev']; RB = LAY['R']['mel']; RD = LAY['R']['dawn']
TN = np.array([l[0] for l in LEAD]); T0, BAR, E8 = E['t0'], E['bar'], E['e8']
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
sc.cycles.transparent_max_bounces = 6
sc.render.resolution_x, sc.render.resolution_y = 1080, 1920
sc.render.resolution_percentage = int(os.environ.get('PCT', 50)); sc.render.fps = FPS
sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.5
try: sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Punchy'
except Exception: pass
sc.frame_start, sc.frame_end = 0, NF - 1
FONT = bpy.data.fonts.load(f'{HERE}/jp_sub.otf')
TXT_K = 2.75


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


def objcol(name, rough=0.45, metal=0.0, emit=0.0):
    m = mat(name, (1, 1, 1), rough, **{'Metallic': metal}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    oi = nt.nodes.new('ShaderNodeObjectInfo'); nt.links.new(oi.outputs['Color'], b.inputs['Base Color'])
    if emit:
        nt.links.new(oi.outputs['Color'], b.inputs['Emission Color']); b.inputs['Emission Strength'].default_value = emit
    return m


def key_series(obj, path, idx, vals, eps=1e-5):
    n = len(vals)
    for f in range(n):
        v = vals[f]
        if 0 < f < n - 1 and abs(v - vals[f - 1]) < eps and abs(v - vals[f + 1]) < eps: continue
        getattr(obj, path)[idx] = float(v); obj.keyframe_insert(path, index=idx, frame=f)


def key_loc(o, P):
    for k in range(3): key_series(o, 'location', k, P[:, k])


def key_sock(sock, vals, step=3, idx=None):
    """keyframe a node socket default_value (scalar or colour) every `step` frames"""
    for f in list(range(0, NF, step)) + [NF - 1]:
        v = vals[f]
        if idx is None and np.ndim(v) == 0: sock.default_value = float(v); sock.keyframe_insert('default_value', frame=f)
        else:
            for c in range(len(v)): sock.default_value[c] = float(v[c])
            sock.keyframe_insert('default_value', frame=f)


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
    elif kind == 'prism':                                   # roof: triangular prism along x, ridge at z=0.5
        v = [bm.verts.new(p) for p in ((-0.5, -0.5, 0), (-0.5, 0.5, 0), (-0.5, 0, 0.5), (0.5, -0.5, 0), (0.5, 0.5, 0), (0.5, 0, 0.5))]
        for f in ((0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (0, 2, 5, 3)): bm.faces.new([v[i] for i in f])
    elif kind == 'bell':                                    # bell, mouth down, crown at z = 0, height 1, mouth radius 1
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
    elif kind == 'bowl':                                    # open bowl, rim radius 1 at z=0, depth 0.6
        segs = a[0]; prof = [(1.0, 0.0), (0.96, -0.3), (0.8, -0.52), (0.5, -0.62), (0.0, -0.64)]
        rings = []
        for k in range(segs):
            ph = 2 * math.pi * k / segs
            rings.append([bm.verts.new((r * math.cos(ph), r * math.sin(ph), z)) for r, z in prof])
        for k in range(segs):
            r0, r1 = rings[k], rings[(k + 1) % segs]
            for j in range(len(prof) - 1): bm.faces.new((r0[j], r1[j], r1[j + 1], r0[j + 1]))
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
        bmesh.ops.solidify(bm, geom=bm.faces[:], thickness=0.06)
    me = bpy.data.meshes.new(kind); bm.to_mesh(me); bm.free()
    if kind in ('sph', 'cone', 'bell', 'cyl', 'bcube', 'bowl'):
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


def text(s, loc, size, m, rot=(math.pi / 2 - 0.12, 0, 0), align='CENTER', ext=0.02):
    cu = bpy.data.curves.new('t', 'FONT'); cu.body = s; cu.font = FONT; cu.size = size * TXT_K; cu.extrude = ext
    cu.align_x = align; cu.align_y = 'CENTER'
    o = link(bpy.data.objects.new('txt', cu)); o.location = loc; o.rotation_euler = rot; o.data.materials.append(m); return o


def tube(pts, r, m, name='tube', seg=8):
    """polyline tube (curve with bevel)"""
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'; cu.bevel_depth = r; cu.bevel_resolution = 2
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for k, p in enumerate(pts): sp.points[k].co = (*p, 1)
    o = link(bpy.data.objects.new(name, cu)); o.data.materials.append(m); return o


def strip_mesh(X, Z, y0, y1, th, m, name):
    bm = bmesh.new(); vs = []
    for x, z in zip(X, Z):
        vs.append([bm.verts.new((x, y0, z)), bm.verts.new((x, y1, z)), bm.verts.new((x, y1, z - th)), bm.verts.new((x, y0, z - th))])
    for a, b in zip(vs, vs[1:]):
        for k in range(4): bm.faces.new((a[k], b[k], b[(k + 1) % 4], a[(k + 1) % 4]))
    bm.faces.new(vs[0][::-1]); bm.faces.new(vs[-1])
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new(name, me)); o.data.materials.append(m)
    for p in o.data.polygons: p.use_smooth = True
    return o


def frame_from_n(n):
    n = np.array(n, float); n /= np.linalg.norm(n)
    xa = np.cross([0, 1, 0], n); xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return Matrix(((xa[0], ya[0], n[0]), (xa[1], ya[1], n[1]), (xa[2], ya[2], n[2])))


def pulse(times, decay=5.0, length=18, amp=1.0):
    a = np.zeros(NF)
    for t in times:
        f0 = int(round(t * FPS)); m_ = np.arange(max(0, f0), min(NF, f0 + length))
        a[m_] = np.maximum(a[m_], amp * np.exp(-(m_ - f0) / decay))
    return a


def smooth(Xa, k):
    Y = np.array(Xa, float).copy(); a = 1 - math.exp(-1 / FPS / k)
    for f in range(1, len(Y)): Y[f] = Y[f - 1] + (Y[f] - Y[f - 1]) * a
    for f in range(len(Y) - 2, -1, -1): Y[f] = Y[f + 1] + (Y[f] - Y[f + 1]) * a
    return Y


def step(t0, w=0.3):
    """0 -> 1 smoothstep starting at t0 over w seconds (per frame array)"""
    u = np.clip((T - t0) / w, 0, 1); return u * u * (3 - 2 * u)


# ---------------- ball paths ----------------
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

# ---------------- dawn progress: every note makes the sky a little lighter ----------------
cnt = np.searchsorted(TN, T + 1e-6) / len(TN)
SUN = EV['sun']
def sun_h(t):
    u = t - SUN['t_rise']
    if u <= 0: return 0.0
    if u <= SUN['t_acc']: return 0.5 * SUN['a'] * u * u
    w = min(u - SUN['t_acc'], SUN['t_brake']); vs = SUN['vmax']; dec = vs / SUN['t_brake']
    return 0.5 * SUN['a'] * SUN['t_acc'] ** 2 + vs * w - 0.5 * dec * w * w
SUNH = np.array([sun_h(t) for t in T]); SUNU = SUNH / SUN['h']
DAWN = smooth(0.08 + 0.62 * cnt, 0.5)
DAWN = np.maximum(DAWN, 0.7 + 0.3 * SUNU) * (T >= SUN['t_rise']) + DAWN * (T < SUN['t_rise'])
def lerpc(keys, d):
    xs = [k[0] for k in keys]; return np.array([np.interp(d, xs, [k[1][c] for k in keys]) for c in range(3)])
ZEN = [(0.0, (0.004, 0.006, 0.03)), (0.35, (0.02, 0.03, 0.11)), (0.62, (0.08, 0.12, 0.32)), (0.75, (0.16, 0.28, 0.6)), (1.0, (0.3, 0.55, 1.0))]
HOR = [(0.0, (0.02, 0.025, 0.07)), (0.35, (0.3, 0.12, 0.24)), (0.62, (0.95, 0.42, 0.25)), (0.75, (1.0, 0.62, 0.3)), (1.0, (1.0, 0.88, 0.6))]
ZC = np.array([lerpc(ZEN, d) for d in DAWN]); HC = np.array([lerpc(HOR, d) for d in DAWN])

# ---------------- materials ----------------
M_FLUTE = mat('flute', (0.75, 0.9, 1.0), 0.05, **{'Coat Weight': 1.0, 'Transmission Weight': 0.0, 'Emission Color': (0.6, 0.85, 1.0), 'Emission Strength': 0.8})
M_OBOE = mat('oboe', (0.85, 0.42, 0.08), 0.25, **{'Coat Weight': 1.0, 'Emission Color': (1.0, 0.5, 0.15), 'Emission Strength': 0.5})
M_DAWN = mat('dawnb', (0.95, 0.45, 0.25), 0.2, **{'Metallic': 0.9, 'Coat Weight': 0.5})
M_KEY = glowmat('key', gain=9.0, base=0.55, rough=0.08, coat=1.0)
M_BELL = glowmat('bell', gain=7.0, base=0.15, rough=0.25, metal=0.85)
M_GOLD = mat('gold', (1.0, 0.72, 0.28), 0.25, **{'Metallic': 0.9})
M_WOOD = mat('wood', (0.36, 0.2, 0.1), 0.6)
M_WOOD2 = mat('wood2', (0.55, 0.33, 0.17), 0.55)
M_RED = mat('redroof', (0.55, 0.08, 0.06), 0.6)
M_WALL = objcol('wall', 0.7)
M_ROOF = objcol('roof', 0.6)
M_WIN = glowmat('win', gain=14.0, base=0.0, rough=0.4, coat=0.0)
M_GRASS = mat('grass', (0.08, 0.2, 0.07), 0.9)
M_GRASS2 = mat('grass2', (0.12, 0.26, 0.08), 0.9)
M_ROCK = mat('rock', (0.22, 0.22, 0.25), 0.8)
M_MTN = mat('mtn', (0.16, 0.18, 0.26), 0.85)
M_MTN2 = mat('mtn2', (0.1, 0.12, 0.2), 0.9)
M_SNOW = mat('snow', (0.8, 0.85, 0.95), 0.6)
M_TREE = mat('tree', (0.03, 0.12, 0.06), 0.8)
M_STEEL = mat('steel', (0.55, 0.57, 0.62), 0.3, **{'Metallic': 0.9})
M_DARK = mat('dark', (0.05, 0.05, 0.06), 0.6)
M_CABLE = mat('cable', (0.12, 0.12, 0.14), 0.4, **{'Metallic': 0.7})
M_WATER = mat('water', (0.02, 0.05, 0.1), 0.04, **{'Coat Weight': 1.0})
M_BASKET = mat('basket', (0.6, 0.42, 0.2), 0.7)
M_OBJ = objcol('objcol', 0.5)
M_LAMP = glowmat('lamp', gain=16.0, base=0.0, rough=0.3, coat=0.0)
M_FX = fxmat('fx', 8.0)
M_TXT = emis('txtw', (1.0, 0.95, 0.85), 4.0); M_TXT_G = emis('txtg', (1.0, 0.78, 0.3), 6.0); M_TXT_B = emis('txtb', (0.6, 0.85, 1.0), 5.0)
M_TXT_O = emis('txto', (1.0, 0.6, 0.25), 5.0)
M_SIGN = mat('sign', (0.25, 0.13, 0.06), 0.6)
M_SUN = emis('sun', (1.0, 0.72, 0.25), 18.0)
def glowdisc(name, gain=6.0):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    tc = nt.nodes.new('ShaderNodeTexCoord'); gr = nt.nodes.new('ShaderNodeTexGradient'); gr.gradient_type = 'SPHERICAL'
    mp = nt.nodes.new('ShaderNodeMapping'); mp.inputs['Scale'].default_value = (2, 2, 2)
    pw = nt.nodes.new('ShaderNodeMath'); pw.operation = 'POWER'; pw.inputs[1].default_value = 2.5
    oi = nt.nodes.new('ShaderNodeObjectInfo'); mu = nt.nodes.new('ShaderNodeMath'); mu.operation = 'MULTIPLY'
    em = nt.nodes.new('ShaderNodeEmission'); g2 = nt.nodes.new('ShaderNodeMath'); g2.operation = 'MULTIPLY'; g2.inputs[1].default_value = gain
    tr_ = nt.nodes.new('ShaderNodeBsdfTransparent'); mx = nt.nodes.new('ShaderNodeMixShader'); out = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(tc.outputs['Object'], mp.inputs[0]); nt.links.new(mp.outputs[0], gr.inputs[0]); nt.links.new(gr.outputs['Fac'], pw.inputs[0])
    nt.links.new(pw.outputs[0], mu.inputs[0]); nt.links.new(oi.outputs['Alpha'], mu.inputs[1])
    nt.links.new(oi.outputs['Color'], em.inputs['Color']); nt.links.new(mu.outputs[0], g2.inputs[0]); nt.links.new(g2.outputs[0], em.inputs['Strength'])
    nt.links.new(mu.outputs[0], mx.inputs['Fac']); nt.links.new(tr_.outputs[0], mx.inputs[1]); nt.links.new(em.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs[0]); return m
M_STAR = glowdisc('starglow', 6.0)
M_GLOWD = glowdisc('sunglowd', 3.0)

PITCH_COL = [(1.0, 0.12, 0.12), (1.0, 0.35, 0.05), (1.0, 0.6, 0.05), (1.0, 0.85, 0.1), (0.55, 1.0, 0.1), (0.1, 1.0, 0.3),
             (0.05, 1.0, 0.75), (0.05, 0.7, 1.0), (0.15, 0.35, 1.0), (0.45, 0.2, 1.0), (0.75, 0.2, 1.0), (1.0, 0.2, 0.65)]
def pcol(p, light=0.0):
    c = PITCH_COL[p % 12]; return tuple(min(1, x * (1 - light) + light) for x in c)

# ---------------- balls ----------------
BALL = {}
for n, b in BP.items():
    rr = b.r
    bpy.ops.mesh.primitive_uv_sphere_add(radius=rr, segments=32, ring_count=16); o = bpy.context.object; o.name = n
    for p in o.data.polygons: p.use_smooth = True
    o.data.materials.append({'flute': M_FLUTE, 'oboe': M_OBOE, 'dawn': M_DAWN}[n])
    bpy.ops.mesh.primitive_torus_add(major_radius=rr * 1.0, minor_radius=rr * 0.11, major_segments=32, minor_segments=8)
    hb = bpy.context.object; hb.parent = o; hb.data.materials.append({'flute': M_GOLD, 'oboe': M_WOOD, 'dawn': M_GOLD}[n])
    key_loc(o, POS[n])
    sp = np.zeros(NF)
    for f in range(1, NF): sp[f] = sp[f - 1] + (POS[n][f, 0] - POS[n][f - 1, 0]) / rr
    o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, sp)
    BALL[n] = o
# the morning star: a glow halo around the flute ball until it lands on the first key
halo = obj(prim('plane', M_STAR), 'halo'); halo.parent = BALL['flute']; halo.rotation_euler = (math.pi / 2, 0, 0)
halo.location = (0, -0.25, 0); halo.color = (0.7, 0.85, 1.0, 1)
ha = np.clip(1 - (T - TN[0]) / 0.4, 0, 1) * 0.5 + 0.08 * (T < TN[0])
hs = np.where(T < TN[0] + 0.4, 1.6, 0.001)
key_series(halo, 'color', 3, ha, 1e-3)
for k in range(2): key_series(halo, 'scale', k, hs, 1e-3)
# a soft light that rides with each melody ball (they glow in the dark)
for n, col, en in (('flute', (0.7, 0.85, 1.0), 25), ('oboe', (1.0, 0.65, 0.3), 18)):
    ld = bpy.data.lights.new('bl' + n, 'POINT'); ld.color = col; ld.energy = en; ld.shadow_soft_size = 0.2
    lo = link(bpy.data.objects.new('bl' + n, ld)); lo.parent = BALL[n]; lo.location = (0, -0.5, 0.35)

# ---------------- course ground (grassy terraces, only ever going down) ----------------
allc = sorted([(h['p'][0], h['p'][2]) for n in ('flute', 'oboe') for h in BP[n].hits])
AX = np.array([a[0] for a in allc]); AZ = np.array([a[1] for a in allc])
FLX = np.arange(-25.0, AX.max() + 9, 0.5)
FLZ = np.array([AZ[np.abs(AX - x) < 3.0].min() if (np.abs(AX - x) < 3.0).any() else np.nan for x in FLX])
s_ = ~np.isnan(FLZ); FLZ = np.interp(FLX, FLX[s_], FLZ[s_]) - 1.3
for _ in range(4): FLZ = np.minimum(FLZ, np.convolve(np.pad(FLZ, 2, mode='edge'), np.ones(5) / 5, 'valid'))
FLZ = np.minimum.accumulate(FLZ)
X_SHORE = AX.max() + 2.5
Z_WATER = float(FLZ[FLX <= X_SHORE].min()) - 2.4
FLZ = np.where(FLX > X_SHORE, np.maximum(Z_WATER - 1.0, FLZ - 0.9 * (FLX - X_SHORE)), FLZ)
def floor_z_at(x): return float(np.interp(x, FLX, FLZ))
_dp = POS['dawn']; _o = np.argsort(_dp[:, 0])
FLZF = np.minimum(FLZ, np.interp(FLX, _dp[_o, 0], _dp[_o, 2]) - RD - 0.75)
FLZF = np.minimum.accumulate(FLZF)
FLZF = np.where(FLX > X_SHORE, np.maximum(Z_WATER - 1.0, FLZF - 0.9 * (FLX - X_SHORE)), FLZF)
def front_z_at(x): return float(np.interp(x, FLX, FLZF))
strip_mesh(FLX, FLZ, -0.9, 3.6, 3.5, M_GRASS, 'meadow')
strip_mesh(FLX, FLZF, -12.0, -0.9, 0.6, M_GRASS, 'meadowfront')
strip_mesh(FLX, FLZ + 0.02, -1.0, -0.85, 0.02 + FLZ.max() - FLZF.min() + 1, M_ROCK, 'terracewall') if False else None
strip_mesh(FLX, FLZ + np.where(FLX > X_SHORE, 0, 1.0 + 0.3 * np.sin(FLX * 0.3)), 3.6, 14.0, 1.6, M_GRASS2, 'backslope')
strip_mesh(FLX, FLZ + np.where(FLX > X_SHORE, 0, 2.6 + 0.6 * np.sin(FLX * 0.17 + 1)), 14.0, 26.0, 2.6, M_GRASS, 'backslope2')
# stones and flowers on the near meadow
M_FLOW = objcol('flow', 0.5, emit=0.6)
for k in range(220):
    x = rng.uniform(FLX[0] + 10, FLX[-1] - 10); y = rng.uniform(-9.5, -2.2); z = front_z_at(x)
    if k % 3 == 0:
        o = sphere((x, y, z + 0.02), rng.uniform(0.08, 0.2), M_ROCK, (8, 6)); o.scale[2] *= 0.5
    else:
        o = sphere((x, y, z + 0.05), 0.05, M_FLOW, (6, 4)); o.color = (*[(1, 0.9, 0.95), (1, 0.85, 0.2), (0.6, 0.7, 1.0)][k % 3], 1)

# ---------------- keys: one per melody hit ----------------
PIECES = []
KEY_TH = 0.12
def riser(objs, t_hit, anchor, lead=1.2, stay=0.9):
    anchor = Vector(anchor)
    e = link(bpy.data.objects.new('pop', None)); e.location = anchor
    for o in objs: o.parent = e; o.matrix_parent_inverse = Matrix.Translation(-anchor)
    s_ = np.zeros(NF)
    for f in range(NF):
        t = T[f]; u1 = (t - (t_hit - lead)) / 0.3; u2 = (t - (t_hit + stay)) / 0.35
        if u1 <= 0: a = 0.0
        elif u1 < 1: a = 1.18 * math.sin(u1 * math.pi / 2) if u1 < 0.7 else 1.18 - 0.18 * (u1 - 0.7) / 0.3
        else: a = 1.0
        b = 1.0 - min(1, max(0, u2)) ** 1.5
        s_[f] = max(0.001, a * b)
    for k in range(3): key_series(e, 'scale', k, s_, 1e-4)
    return e


def flash(o, t, col, base_loc=None, nv=None):
    fh = int(round(t * FPS))
    o.color = (*col, 0); o.keyframe_insert('color', index=3, frame=max(0, fh - 1))
    for df, a in ((0, 1.0), (4, 0.55), (18, 0.08)):
        o.color = (*col, a); o.keyframe_insert('color', index=3, frame=fh + df)
    if base_loc is not None:
        for df, d in ((-1, 0), (1, -0.03), (4, 0.01), (8, 0)):
            o.location = base_loc + nv * d; o.keyframe_insert('location', frame=max(0, fh + df))


RING = []
for n in ('flute', 'oboe'):
    for h in BP[n].hits:
        if h['kind'] not in ('key', 'lever', 'pkey', 'reunion'): continue
        nv = np.array(h['n']); top = np.array(h['p']) - RB * nv
        w, d = 0.42, (0.6 if h['kind'] in ('key', 'lever') else 0.4)
        ctr = top - nv * KEY_TH / 2
        o = obj(prim('bcube', M_KEY), 'key')
        o.matrix_world = Matrix.Translation(Vector(ctr)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((w, d, KEY_TH, 1))
        col = pcol(h['pitch'], 0.0); o.color = (*col, 0)
        flash(o, h['t'], col, o.location.copy(), Vector(nv))
        # wooden (oboe) or silver (flute) frame under the key
        fr_ = obj(prim('cube', M_WOOD if n == 'oboe' else M_STEEL), 'kfr')
        fr_.matrix_world = Matrix.Translation(Vector(ctr - nv * 0.08)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((w + 0.06, d * 0.5, 0.05, 1))
        zf = floor_z_at(top[0])
        po = rod((top[0], top[1] + 0.1, top[2] - KEY_TH - 0.08), (top[0], top[1] + 0.1, zf - 0.2), 0.03, M_WOOD if n == 'oboe' else M_STEEL, 6)
        objs = [o, fr_, po]
        if h['kind'] == 'lever':
            # lever: a rod from the key's far end up to the ramp gate (drawn with the ramp)
            pass
        stay = 0.9 if h['kind'] != 'lever' else 1.6
        if h['t'] > 2.0: riser(objs, h['t'], top, lead=1.2, stay=stay)
        RING.append((h['t'], top, (0.75, 0.9, 1.0) if n == 'flute' else (1.0, 0.7, 0.35)))
        PIECES.append(dict(t=h['t'], top=top, n=nv, ball=n))

# ---------------- ramps with gates (the waiting ball), levers ----------------
LBL = {'flute': ('フルート', M_TXT_B), 'oboe': ('オーボエ', M_TXT_O)}
for k, rp in enumerate(EV['ramps']):
    hd = np.array(rp['head']); lp = np.array(rp['lip']); a = rp['alpha']
    # chute: two side rails + floor, from a little behind the head to the lip
    back = hd - 0.25 * np.array([math.cos(a), 0, -math.sin(a)])
    for dy in (-0.17, 0.17):
        rod(back + [0, dy, -0.05], lp + [0, dy, -0.05], 0.025, M_WOOD2, 8)
    fl_ = obj(prim('cube', M_WOOD), 'chute'); mid = (back + lp) / 2 - [0, 0, RB + 0.02]
    Lc = float(np.linalg.norm(lp - back))
    fl_.matrix_world = Matrix.Translation(Vector(mid)) @ Matrix.Rotation(a, 4, 'Y') @ Matrix.Diagonal((Lc, 0.34, 0.04, 1))
    for pp in (back, lp):
        rod(pp + [0, 0.22, -RB - 0.04], (pp[0], pp[1] + 0.22, floor_z_at(pp[0]) - 0.2), 0.035, M_WOOD, 6)
    # gate: a small bar across the chute in front of the ball; lifts when the lever key is struck
    gx = hd + (RB + 0.04) * np.array([math.cos(a), 0, -math.sin(a)])
    gate = box((gx[0], gx[1], gx[2] - 0.02), (0.05, 0.42, 0.34), M_RED, rot=(0, -a, 0), name='gate')
    up = step(rp['t_open'] + 0.01, 0.07) * 0.42
    key_series(gate, 'location', 2, gx[2] - 0.02 + up)
    # lever linkage: rod from the lever key up to the gate (glows when pulled)
    lv = EV['levers'][k]; ptop = np.array(lv['p']) - RB * np.array(lv['n'])
    link_ = rod((ptop[0] + 0.21, ptop[1] + 0.32, ptop[2] - 0.04), (gx[0], gx[1] + 0.25, gx[2] + 0.2), 0.02, M_LAMP, 6); link_.color = (1.0, 0.7, 0.3, 0)
    for df, aa in ((-1, 0.0), (0, 1.0), (10, 0.3), (30, 0.0)):
        link_.color = (1.0, 0.7, 0.3, aa); link_.keyframe_insert('color', index=3, frame=max(0, int(round(lv['t'] * FPS)) + df))
    # label of the waiting ball (keyword sign)
    nm_, mt = LBL[rp['ball']]
    text(nm_, (hd[0] - 0.2, hd[1] - 0.3, hd[2] + 0.62), 0.26, mt)
# baskets
for bk in EV['baskets']:
    p = np.array(bk['p'])
    o = obj(prim('bowl', M_BASKET, 20), 'basket'); o.location = (p[0], p[1], p[2] - RB + 0.13); o.scale = (0.3, 0.3, 0.3)

# ---------------- ropeway: cable, buckets, towers; driven by the windmill ----------------
M_BUCKET = mat('bucket', (0.75, 0.15, 0.1), 0.35, **{'Coat Weight': 0.7})
HANG = 0.85
for k, gd in enumerate(EV['gondolas']):
    route = np.array(gd['route']); cab = route + [0, 0, RB + 0.1 + HANG]
    tube(cab.tolist(), 0.018, M_CABLE, 'cable')
    # towers along the high part of the cable
    for j in range(0, len(cab), 40):
        c = cab[j]
        if c[1] < 0.9: continue
        rod((c[0], c[1] + 0.05, c[2] + 0.1), (c[0], c[1] + 0.05, floor_z_at(c[0]) + 1.0), 0.06, M_WOOD, 6)
        rod((c[0] - 0.3, c[1] + 0.05, c[2] + 0.08), (c[0] + 0.3, c[1] + 0.05, c[2] + 0.08), 0.04, M_WOOD, 6)
    # bucket follows the ball during the ride, waits at the basket before, flies back up after
    ts = np.array(gd['ts']); ps = np.array(gd['ps'])
    bp = np.zeros((NF, 3)); tilt = np.zeros(NF); scl = np.ones(NF)
    for f in range(NF):
        t = T[f]
        if t <= ts[0]: q = ps[0]
        elif t <= ts[-1]: q = np.array([np.interp(t, ts, ps[:, c]) for c in range(3)])
        else:
            u = min(1.0, (t - ts[-1]) / 1.2); q = ps[-1] + np.array([0.0, 1.2 * u, 0.9 * u * u + (0.35 if not gd['drop'] else 0.0) * min(1, u * 4)])
        bp[f] = q
        if gd['drop'] and t > ts[-1] + 0.5: tilt[f] = min(1.0, (t - ts[-1] - 0.5) / 0.12) * 1.2
        if t > ts[-1] + 1.4: scl[f] = 0.001
    hold = link(bpy.data.objects.new('bucket', None)); key_loc(hold, bp); hold.rotation_mode = 'XYZ'; key_series(hold, 'rotation_euler', 0, -tilt)
    for c in range(3): key_series(hold, 'scale', c, scl, 1e-3)
    bk = obj(prim('bowl', M_BUCKET, 20), 'bk'); bk.parent = hold; bk.location = (0, 0, -RB + 0.12); bk.scale = (0.28, 0.28, 0.3)
    rod((0, 0, 0.12), (0, 0, RB + 0.1 + HANG), 0.015, M_STEEL, 6).parent = hold
    wh = obj(prim('cyl', M_STEEL, 16), 'trolley'); wh.parent = hold; wh.location = (0, 0, RB + 0.1 + HANG); wh.scale = (0.09, 0.09, 0.05)
    wh.rotation_euler = (math.pi / 2, 0, 0)
    rf = obj(prim('prism', M_BUCKET), 'roof'); rf.parent = hold; rf.location = (0, 0, RB + 0.32); rf.scale = (0.5, 0.5, 0.25)
# windmill (drives the ropeway): spins up when the first gondola leaves
WM = EV['wake']['windmill']
wx = EV['baskets'][0]['p'][0] - 3.0; wy = 9.0; wz = floor_z_at(wx) + 1.0
tw = obj(prim('cone', M_WALL, 8, 1.1, 0.7, 4.5), 'mill'); tw.location = (wx, wy, wz + 2.25); tw.color = (0.85, 0.8, 0.7, 1)
rfm = obj(prim('cone', M_ROOF, 8, 0.85, 0.0, 1.2), 'millroof'); rfm.location = (wx, wy, wz + 5.1); rfm.color = (0.45, 0.1, 0.07, 1)
hub = link(bpy.data.objects.new('hub', None)); hub.location = (wx, wy - 0.95, wz + 4.2); hub.rotation_mode = 'XYZ'
for j in range(4):
    arm = link(bpy.data.objects.new('arm', None)); arm.parent = hub; arm.rotation_euler = (0, j * math.pi / 2, 0)
    box((0, 0, 1.6), (0.12, 0.05, 3.2), M_WOOD, name='spar').parent = arm
    sail = box((0.32, -0.02, 1.9), (0.55, 0.02, 2.4), M_OBJ, name='sail'); sail.parent = arm; sail.color = (0.9, 0.88, 0.8, 1)
sphere((0, 0, 0), 0.22, M_GOLD).parent = hub
om = np.clip((T - WM) / 2.5, 0, 1) * 1.6
key_series(hub, 'rotation_euler', 1, np.cumsum(om) / FPS)
# gear pair at the windmill foot (visible drive) + a sign
for gx_, gz_, r_, sgn in ((wx - 1.0, wz + 0.6, 0.55, 1), (wx - 1.0 + 0.95, wz + 0.6, 0.4, -1.375)):
    g = link(bpy.data.objects.new('gear', None)); g.location = (gx_, wy - 1.3, gz_); g.rotation_mode = 'XYZ'
    d = obj(prim('cyl', M_GOLD, 32), 'gd'); d.parent = g; d.scale = (r_, r_, 0.08); d.rotation_euler = (math.pi / 2, 0, 0)
    nt_ = int(r_ * 22)
    for j in range(nt_):
        ph = 2 * math.pi * j / nt_
        tth = box((r_ * math.cos(ph), 0, r_ * math.sin(ph)), (0.1, 0.08, 0.1), M_GOLD, rot=(0, -ph, 0)); tth.parent = g
    key_series(g, 'rotation_euler', 1, sgn * np.cumsum(om) / FPS * 2.0)

# ---------------- the dawn rail: start shed, rails, bell gates ----------------
RL = EV['rail']; dawnP = POS['dawn']
Y_R = RL['y']
rail_ts = np.arange(RL['t_rel'], RL['t_end'] + 1e-9, 0.25)
rp_ = np.array([BP['dawn'].at(t) for t in rail_ts])
for dy in (-0.16, 0.16):
    tube((rp_ + [0, dy, -RD + 0.02]).tolist(), 0.03, M_STEEL, 'rail')
for j in range(0, len(rp_), 6):
    c = rp_[j]
    box((c[0], c[1], c[2] - RD - 0.03), (0.08, 0.5, 0.05), M_WOOD, name='sleeper')
    if j % 12 == 0: rod((c[0], c[1] + 0.3, c[2] - RD - 0.05), (c[0], c[1] + 0.3, front_z_at(c[0]) - 0.1), 0.04, M_WOOD, 6)
pt = np.array(RL['p_top'])
shed = box((pt[0] - 0.35, pt[1], pt[2] + 0.1), (0.9, 0.8, 0.8), M_WALL, name='shed'); shed.color = (0.4, 0.25, 0.15, 1)
sr = obj(prim('prism', M_ROOF), 'shedroof'); sr.location = (pt[0] - 0.35, pt[1], pt[2] + 0.5); sr.scale = (1.1, 1.0, 0.5); sr.color = (0.5, 0.1, 0.08, 1)
text('夜明けの玉', (pt[0] + 0.2, pt[1] - 0.55, pt[2] + 1.2), 0.24, M_TXT_G)
for gt in EV['gates']:
    x, z, t = gt['x'], gt['z'], gt['t']; ps = gt['pitches']; nb = len(ps)
    big = gt['kind'] == 'final'
    H = 0.5 if not big else 0.75
    for dy in (-0.32, 0.32): rod((x + 0.02, Y_R + dy, z - RD - 0.05), (x + 0.02, Y_R + dy, z + H), 0.025, M_WOOD2, 6)
    rod((x + 0.02, Y_R - 0.36, z + H), (x + 0.02, Y_R + 0.36, z + H), 0.025, M_WOOD2, 6)
    # paddle hanging from the cross bar: pushed by the ball at t, swings out and back
    hinge = link(bpy.data.objects.new('hinge', None)); hinge.location = (x + 0.02, Y_R, z + H); hinge.rotation_mode = 'XYZ'
    pad = box((0, 0, -(H - 0.1) / 2 - 0.05), (0.04, 0.44, H - 0.1), M_WOOD, name='paddle'); pad.parent = hinge
    ang = np.zeros(NF)
    for f in range(NF):
        u = T[f] - t
        if u > 0: ang[f] = 0.55 * math.sin(min(u, 0.25) / 0.25 * math.pi / 2) * math.exp(-max(0, u - 0.25) * 2.2) * math.cos(max(0, u - 0.25) * 7)
    key_series(hinge, 'rotation_euler', 1, -ang, 1e-3)
    # bells on top: one per chord note, low notes larger
    for j, p in enumerate(ps):
        yy = Y_R + (j - (nb - 1) / 2) * (0.24 if nb > 3 else 0.28)
        sz = (0.17 + 0.005 * (64 - p)) * (1.4 if big else 1.0)
        bo = obj(prim('bell', M_BELL, 20), 'bell')
        loc = Vector((x + 0.02, yy, z + H + 0.06 + sz))
        bo.location = loc; bo.scale = (sz * 0.7, sz * 0.7, sz)
        col = pcol(p, 0.15); bo.color = (*col, 0)
        flash(bo, t, col)
        bo.rotation_mode = 'XYZ'
        wob = np.zeros(NF)
        for f in range(NF):
            u = T[f] - t
            if u > 0: wob[f] = 0.3 * math.sin(u * 22) * math.exp(-u * 3)
        key_series(bo, 'rotation_euler', 0, wob, 1e-3)
        RING.append((t, np.array(loc), col))
# the latch funnel at the rail end
pf = np.array(RL['p_fun'])
fun = obj(prim('bowl', M_GOLD, 24), 'funnel'); fun.location = (pf[0], pf[1], pf[2] - RD + 0.15); fun.scale = (0.42, 0.42, 0.5)
fz = pf[2] - RD + 0.15 - 0.2 * step(RL['t_fun'], 0.6)
key_series(fun, 'location', 2, fz)

# ---------------- final cups (sunrise lever) ----------------
cupz = {}
for nme, pc in EV['cups'].items():
    pc = np.array(pc)
    cu = obj(prim('bowl', M_KEY, 24), 'cup'); cu.location = (pc[0], pc[1], pc[2] - RB + 0.14); cu.scale = (0.3, 0.3, 0.32)
    p = LEAD[-1][1][0 if nme == 'flute' else 1]; col = pcol(p, 0.2); cu.color = (*col, 0)
    flash(cu, FINAL, col)
    key_series(cu, 'location', 2, pc[2] - RB + 0.14 - EV['cup_drop'] * step(FINAL, EV['cup_sink']))
    rod((pc[0], pc[1], pc[2] - RB - 0.1), (pc[0], pc[1], floor_z_at(pc[0])), 0.05, M_GOLD, 8)
    RING.append((FINAL, pc - [0, 0, RB], col))

# hit rings (FX pool)
NRING = 14
RING.sort(key=lambda r: r[0])
ring_loc = np.zeros((NRING, NF, 3)); ring_scl = np.full((NRING, NF), 1e-3); ring_a = np.zeros((NRING, NF)); ring_c = [(1, 1, 1)] * NRING
for k, (t, top, c) in enumerate(RING):
    r = k % NRING; f0 = int(round(t * FPS))
    for df in range(14):
        f = f0 + df
        if f >= NF: break
        u = df / 13; ring_loc[r, f] = top + np.array([0, -0.32, 0.02]); ring_scl[r, f] = 0.35 + 1.3 * (1 - (1 - u) ** 2)
        ring_a[r, f] = (1 - u) ** 1.6
    ring_c[r] = c
for r in range(NRING):
    bpy.ops.mesh.primitive_torus_add(major_radius=0.25, minor_radius=0.014, major_segments=32, minor_segments=5, rotation=(math.pi / 2, 0, 0))
    o = bpy.context.object; o.data.materials.append(M_FX); o.color = (*ring_c[r], 0)
    for f in range(1, NF):
        if ring_a[r, f] == 0: ring_loc[r, f] = ring_loc[r, f - 1]
    key_loc(o, ring_loc[r])
    for k in range(3): key_series(o, 'scale', k, ring_scl[r])
    key_series(o, 'color', 3, ring_a[r], 1e-3)

# ---------------- village: houses with windows, barn + rooster, lanterns, trees ----------------
HOUSE_COL = [(0.75, 0.2, 0.12), (0.92, 0.85, 0.7), (0.2, 0.35, 0.55), (0.85, 0.6, 0.2), (0.3, 0.5, 0.3)]
WINDOWS = []                                                # (object, time lit or None)
def house(x, y, w, d, h, col, wins=3, dark_until=None):
    z = floor_z_at(x) + 1.0 + (0.3 if y > 5 else 0.0)
    b = box((x, y, z + h / 2), (w, d, h), M_WALL, name='house'); b.color = (*col, 1)
    r = obj(prim('prism', M_ROOF), 'roof'); r.location = (x, y, z + h); r.scale = (w + 0.3, d + 0.3, h * 0.7); r.color = (0.25, 0.08, 0.05, 1)
    ws = []
    for j in range(wins):
        wx_ = x - w / 2 + (j + 0.5) * w / wins
        wo = box((wx_, y - d / 2 - 0.02, z + h * 0.55), (min(0.45, w / wins * 0.55), 0.03, 0.5), M_WIN, name='win'); wo.color = (1.0, 0.72, 0.35, 0)
        ws.append(wo)
    rod((x + w * 0.3, y, z + h + 0.1), (x + w * 0.3, y, z + h + 0.9), 0.12, M_ROCK, 8)
    return ws
xs_win = [w['x'] for w in EV['wake']['windows']]
hx = np.linspace(min(xs_win) - 1.0, max(xs_win) + 1.0, 8)
win_objs = []
for k, x in enumerate(hx):
    win_objs += house(x, 5.0 + 0.8 * (k % 2), 1.6, 1.4, 1.6 + 0.3 * (k % 3), HOUSE_COL[k % 5], wins=3 + (k % 2))
win_objs.sort(key=lambda o: o.location.x)
wt = sorted(w['t'] for w in EV['wake']['windows'])
for j, wo in enumerate(win_objs):
    t = wt[min(len(wt) - 1, int(j * len(wt) / len(win_objs)))]
    f0 = int(round(t * FPS))
    wo.keyframe_insert('color', index=3, frame=max(0, f0 - 1)); wo.color = (1.0, 0.72, 0.35, 1.0); wo.keyframe_insert('color', index=3, frame=f0 + 2)
    RING.append((t, np.array(wo.location), (1.0, 0.75, 0.4)))
# other houses (lit gradually by the dawn) + trees
for k, x in enumerate(np.arange(-14, FLX[-1] - 20, 7.5)):
    if min(xs_win) - 3 < x < max(xs_win) + 3: continue
    ws = house(x + rng.uniform(-1, 1), 8.5 + rng.uniform(0, 3), 1.8, 1.5, 1.8, HOUSE_COL[(k + 2) % 5], wins=2)
    for wo in ws:
        t = float(np.interp(wo.location.x, [0, 85], [3.0, 50.0])) + rng.uniform(-1, 1)
        f0 = int(round(max(0.5, t) * FPS)); wo.keyframe_insert('color', index=3, frame=max(0, f0 - 1)); wo.color = (1.0, 0.72, 0.35, 0.7)
        wo.keyframe_insert('color', index=3, frame=f0 + 4)
for k in range(140):
    x = rng.uniform(FLX[0] + 5, FLX[-1] - 5); y = rng.uniform(11, 24)
    if y < 0 and rng.uniform() < 0.6: continue
    z = floor_z_at(x) + (1.0 if y > 3.6 else 0.0) + (1.6 if y > 14 else 0.0); h = rng.uniform(1.6, 3.6)
    t_ = obj(prim('cone', M_TREE, 8, 1.0, 0.0, 1.0), 'tree'); t_.location = (x, y, z + h / 2 + 0.3); t_.scale = (h * 0.3, h * 0.3, h)
    rod((x, y, z), (x, y, z + 0.4), 0.08, M_WOOD, 6)
# barn + rooster (crows when the flute starts its second call)
RT = EV['wake']['rooster']
bx_ = float(np.interp(RT, T, POS['flute'][:, 0])) + 1.5
bz_ = floor_z_at(bx_) + 1.0
barn = box((bx_, 4.6, bz_ + 1.0), (2.2, 1.6, 2.0), M_WALL, name='barn'); barn.color = (0.55, 0.12, 0.08, 1)
br = obj(prim('prism', M_ROOF), 'barnroof'); br.location = (bx_, 4.6, bz_ + 2.0); br.scale = (2.5, 1.9, 1.2); br.color = (0.2, 0.2, 0.22, 1)
roo = link(bpy.data.objects.new('rooster', None)); roo.location = (bx_, 4.6, bz_ + 2.62)
body = sphere((0, 0, 0.18), 0.22, M_OBJ, (16, 8)); body.parent = roo; body.color = (0.95, 0.9, 0.85, 1); body.scale = (0.28, 0.2, 0.22)
tail = obj(prim('cone', M_OBJ, 8, 1, 0, 1), 'tail'); tail.parent = roo; tail.location = (-0.25, 0, 0.32); tail.scale = (0.12, 0.08, 0.35)
tail.rotation_euler = (0, -0.6, 0); tail.color = (0.15, 0.3, 0.2, 1)
for lx in (-0.06, 0.06): rod((lx, 0, -0.02), (lx, 0, 0.08), 0.015, M_GOLD, 6).parent = roo
neck = link(bpy.data.objects.new('neck', None)); neck.parent = roo; neck.location = (0.15, 0, 0.3); neck.rotation_mode = 'XYZ'
head = sphere((0.06, 0, 0.18), 0.1, M_OBJ, (12, 8)); head.parent = neck; head.color = (0.95, 0.9, 0.85, 1)
comb = sphere((0.04, 0, 0.29), 0.07, M_LAMP, (8, 6)); comb.parent = neck; comb.scale = (0.08, 0.03, 0.06); comb.color = (1.0, 0.1, 0.05, 0.1)
beak = obj(prim('cone', M_GOLD, 8, 1, 0, 1), 'beak'); beak.parent = neck; beak.location = (0.19, 0, 0.18); beak.scale = (0.035, 0.035, 0.09); beak.rotation_euler = (0, math.pi / 2, 0)
crow = np.zeros(NF)
for f in range(NF):
    u = T[f] - RT
    if 0 < u < 2.2: crow[f] = -0.7 * math.sin(min(1, u / 0.25) * math.pi / 2) * (1 if u < 1.8 else (2.2 - u) / 0.4)
key_series(neck, 'rotation_euler', 1, crow, 1e-3)
for df, aa in ((-1, 0.1), (0, 1.0), (50, 1.0), (60, 0.2)):
    comb.color = (1.0, 0.1, 0.05, aa); comb.keyframe_insert('color', index=3, frame=max(0, int(round(RT * FPS)) + df))
ko = text('コケコッコー！', (bx_ + 0.3, 4.2, bz_ + 3.6), 0.34, M_TXT_G)
ks = np.where((T > RT + 0.1) & (T < RT + 1.9), 1.0, 0.001)
for c in range(3): key_series(ko, 'scale', c, ks, 1e-3)
# lanterns along the near path: each flute note of bars 17-20 lights one
for ln in EV['wake']['lanterns']:
    x = ln['x']; zf = floor_z_at(x)
    rod((x, 0.9, zf), (x, 0.9, zf + 1.1), 0.03, M_DARK, 6)
    lo = sphere((x, 0.9, zf + 1.2), 0.1, M_LAMP, (12, 8)); lo.color = (1.0, 0.75, 0.4, 0)
    f0 = int(round(ln['t'] * FPS)); lo.keyframe_insert('color', index=3, frame=max(0, f0 - 1)); lo.color = (1.0, 0.75, 0.4, 0.9)
    lo.keyframe_insert('color', index=3, frame=f0 + 1)
    obj(prim('cone', M_DARK, 8, 0.14, 0.02, 0.12), 'lcap').location = (x, 0.9, zf + 1.34)
# fjord at the end
xw0 = FLX[-1] - 1.0
water = box((X_SHORE + 100, 10, Z_WATER), (200, 140, 0.1), M_WATER, name='fjord')

# ---------------- mountains (far), the sun elevator ----------------
def ridge(y, zbase, amp, seed, m, snow=True, x0=FLX[0] - 40, x1=FLX[-1] + 60):
    r_ = np.random.default_rng(seed)
    X = np.arange(x0, x1, 2.0); Z = np.zeros_like(X)
    for k in range(5):
        Z += amp / (k + 1) * np.sin(X * (0.025 * (k + 1) * (1.3 + r_.uniform())) + r_.uniform(0, 6))
    Z = zbase + amp + Z
    strip_mesh(X, Z, y, y + 6, Z.max() - zbase + 30, m, 'ridge')
    if False:
        for x, z in zip(X[::3], Z[::3]):
            if z > zbase + amp * 1.5:
                c = obj(prim('cone', M_SNOW, 8, 1, 0, 1), 'snow'); c.location = (x, y - 0.5, z - 1.2); c.scale = (4.0, 1.0, 2.8)
    return X, Z
ridge(75, -6, 10, 3, M_MTN2, x1=FLX[-1] + 160)
RX, RZ = ridge(32, -7, 4.5, 5, M_MTN, x1=FLX[-1] + 160)
# the sun: a giant gold marble behind the near ridge, hauled up by a crane (rope over a pulley, counterweight behind it)
SR = 4.0
xs_sun = float(EV['cups']['flute'][0]) + 3.0
ridge_z = float(np.interp(xs_sun, RX, RZ))
SY = 46.0
sz0 = ridge_z - SR - 0.4
M_SUNB = mat('sunball', (1.0, 0.55, 0.1), 0.15, **{'Metallic': 0.5, 'Coat Weight': 1.0, 'Emission Color': (1.0, 0.42, 0.06), 'Emission Strength': 1.1})
sunh = link(bpy.data.objects.new('sunh', None)); sunh.location = (xs_sun, SY, sz0)
sun = sphere((0, 0, 0), SR, M_SUNB, (48, 24)); sun.parent = sunh
bpy.ops.mesh.primitive_torus_add(major_radius=SR * 1.0, minor_radius=SR * 0.1, major_segments=64, minor_segments=12)
sb = bpy.context.object; sb.parent = sunh; sb.rotation_euler = (0.45, 0.35, 0); sb.data.materials.append(mat('sunband', (1.0, 0.9, 0.75), 0.2, **{'Emission Color': (1.0, 0.85, 0.6), 'Emission Strength': 1.2}))
key_series(sunh, 'location', 2, sz0 + SUNH)
sunh.rotation_mode = 'XYZ'; key_series(sunh, 'rotation_euler', 1, -SUNH / SR * 0.5)
glow = obj(prim('plane', M_GLOWD), 'sunglow'); glow.location = (xs_sun, SY + 6, sz0); glow.rotation_euler = (math.pi / 2, 0, 0)
glow.color = (1.0, 0.45, 0.1, 0); glow.scale = (SR * 7, SR * 7, 1)
key_series(glow, 'location', 2, sz0 + SUNH)
key_series(glow, 'color', 3, 0.1 + 0.35 * SUNU, 1e-3)
# crane: A-frame straddling the sun, pulley on top, rope down to the sun's hook
twz = sz0 + SUN['h'] + SR + 4.0
for dx in (-SR - 1.2, SR + 1.2):
    rod((xs_sun + dx * 1.25, SY, ridge_z - 3), (xs_sun + dx, SY, twz), 0.3, M_WOOD, 8)
rod((xs_sun - SR - 1.6, SY, twz), (xs_sun + SR + 1.6, SY, twz), 0.32, M_WOOD, 8)
pul = obj(prim('cyl', M_STEEL, 32), 'pulley'); pul.location = (xs_sun, SY, twz + 0.9); pul.scale = (0.9, 0.9, 0.35); pul.rotation_euler = (math.pi / 2, 0, 0)
pul.rotation_mode = 'XYZ'; key_series(pul, 'rotation_euler', 1, SUNH / 0.9)
ropeh = link(bpy.data.objects.new('ropeh', None)); key_series(ropeh, 'location', 2, sz0 + SUNH + SR)
rp_o = obj(prim('cyl', M_CABLE, 8), 'rope'); rp_o.location = (xs_sun, SY, 0)
rz0 = sz0 + SUNH + SR; rlen = twz + 0.9 - rz0
key_series(rp_o, 'location', 2, (rz0 + twz + 0.9) / 2); rp_o.scale = (0.08, 0.08, 1); key_series(rp_o, 'scale', 2, np.maximum(0.05, rlen))
# the latch line from the funnel to the crane foot: a spark runs along it when the dawn ball drops in
foot = np.array([xs_sun + SR + 2.5, SY - 2.0, ridge_z - 2.0])
rl = rod((pf[0], pf[1], pf[2] - 0.5), tuple(foot), 0.04, M_CABLE, 6)
spk = sphere((0, 0, 0), 0.25, M_LAMP, (12, 8)); spk.color = (1.0, 0.75, 0.3, 0.6)
P_ = np.zeros((NF, 3)); S_ = np.full(NF, 0.001)
p0_ = np.array([pf[0], pf[1], pf[2] - 0.5])
for f in range(NF):
    u = (T[f] - SUN['t_latch']) / (SUN['t_rise'] - SUN['t_latch'])
    P_[f] = p0_ + (foot - p0_) * min(1, max(0, u)) ** 1.5
    if 0 <= u <= 1.0: S_[f] = 1.0 + 0.8 * u
key_loc(spk, P_)
for c in range(3): key_series(spk, 'scale', c, S_, 1e-3)

# ---------------- sky, stars, lights (keyed to the dawn progress) ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes['Background']; bg.inputs['Strength'].default_value = 0.35
key_sock(bg.inputs['Color'], [(*ZC[f] * 0.6 + HC[f] * 0.4, 1) for f in range(NF)])
SKY = bpy.data.materials.new('sky'); SKY.use_nodes = True; nt = SKY.node_tree; nt.nodes.clear()
tc = nt.nodes.new('ShaderNodeTexCoord'); sep = nt.nodes.new('ShaderNodeSeparateXYZ')
cz = nt.nodes.new('ShaderNodeRGB'); ch = nt.nodes.new('ShaderNodeRGB')
mr = nt.nodes.new('ShaderNodeMapRange'); mr.inputs['From Min'].default_value = 0.3; mr.inputs['From Max'].default_value = 0.75
mix = nt.nodes.new('ShaderNodeMixRGB'); vo_ = nt.nodes.new('ShaderNodeTexVoronoi'); vo_.inputs['Scale'].default_value = 300.0
st = nt.nodes.new('ShaderNodeMapRange'); st.inputs['From Min'].default_value = 0.04; st.inputs['From Max'].default_value = 0.0
st.inputs['To Min'].default_value = 0.0
stm = nt.nodes.new('ShaderNodeMath'); stm.operation = 'MULTIPLY'
add = nt.nodes.new('ShaderNodeMixRGB'); add.blend_type = 'ADD'; add.inputs['Fac'].default_value = 1.0
em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Strength'].default_value = 1.0; out = nt.nodes.new('ShaderNodeOutputMaterial')
nt.links.new(tc.outputs['Generated'], sep.inputs[0]); nt.links.new(sep.outputs['Y'], mr.inputs['Value'])
nt.links.new(mr.outputs[0], mix.inputs['Fac']); nt.links.new(ch.outputs[0], mix.inputs[1]); nt.links.new(cz.outputs[0], mix.inputs[2])
nt.links.new(tc.outputs['Object'], vo_.inputs['Vector']); nt.links.new(vo_.outputs['Distance'], st.inputs['Value'])
nt.links.new(st.outputs[0], stm.inputs[0]); nt.links.new(sep.outputs['Y'], stm.inputs[1])
nt.links.new(mix.outputs[0], add.inputs[1]); nt.links.new(stm.outputs[0], add.inputs[2])
nt.links.new(add.outputs[0], em.inputs['Color']); nt.links.new(em.outputs[0], out.inputs[0])
key_sock(cz.outputs[0], [(*ZC[f], 1) for f in range(NF)]); key_sock(ch.outputs[0], [(*HC[f], 1) for f in range(NF)])
key_sock(st.inputs['To Max'], [float(3.0 * max(0, 1 - DAWN[f] / 0.7)) for f in range(NF)])
SKYX0, SKYX1 = FLX[0] - 80, FLX[-1] + 120
skyo = obj(prim('plane', SKY), 'sky'); skyo.location = ((SKYX0 + SKYX1) / 2, 120, 30); skyo.rotation_euler = (math.pi / 2, 0, 0)
skyo.scale = (SKYX1 - SKYX0 + 200, 160, 1); skyo.visible_shadow = False
moon = bpy.data.lights.new('moon', 'SUN'); moon.angle = 0.05; moon.color = (0.6, 0.72, 1.0)
mo = link(bpy.data.objects.new('moon', moon)); mo.rotation_euler = (math.radians(50), math.radians(-20), math.radians(-30))
sunl = bpy.data.lights.new('sunl', 'SUN'); sunl.angle = 0.03; sunl.color = (1.0, 0.72, 0.42)
so = link(bpy.data.objects.new('sunl', sunl)); so.rotation_euler = (math.radians(-78), math.radians(0), math.radians(15))
fill = bpy.data.lights.new('fill', 'SUN'); fill.color = (1.0, 0.85, 0.7)
fo = link(bpy.data.objects.new('fill', fill)); fo.rotation_euler = (math.radians(62), math.radians(18), math.radians(-12))
for f in list(range(0, NF, 3)) + [NF - 1]:
    d = DAWN[f]
    moon.energy = float(0.9 * max(0.15, 1 - d)); moon.keyframe_insert('energy', frame=f)
    sunl.energy = float(4.0 * max(0, (d - 0.3) / 0.7) ** 1.5); sunl.keyframe_insert('energy', frame=f)
    fill.energy = float(0.25 + 1.6 * d); fill.keyframe_insert('energy', frame=f)

# ---------------- signs (keywords for search, readable on a phone) ----------------
def sign(lines, x, y, z, w, h, plate=True):
    z = front_z_at(x) + h / 2 + 0.3
    if plate:
        box((x, y + 0.08, z), (w, 0.12, h), M_SIGN, rot=(-0.12, 0, 0), name='plate')
        for sgn in (1, -1): box((x, y + 0.08 + 0.12 * sgn * h / 2, z + 0.99 * sgn * h / 2), (w + 0.12, 0.16, 0.08), M_GOLD, rot=(-0.12, 0, 0))
        for xx in (x - w / 2 + 0.25, x + w / 2 - 0.25): rod((xx, y + 0.12, z - h / 2), (xx, y + 0.12, front_z_at(xx) - 0.1), 0.06, M_WOOD, 8)
    zz = z + h / 2 - 0.06
    for s_, size, mt in lines:
        zz -= size * 1.25
        text(s_, (x, y - 0.03 + 0.12 * (zz + size * 0.15 - z), zz + size * 0.15), size, mt)
fx_ = lambda t, n='flute': float(np.interp(t, T, POS[n][:, 0])); fz_ = lambda t, n='flute': float(np.interp(t, T, POS[n][:, 2]))
sign([('グリーグ「朝」', 0.62, M_TXT_G), ('ペール・ギュント組曲より', 0.32, M_TXT), ('ビー玉からくり × クラシック', 0.3, M_TXT_B)],
     0.9, -3.4, -2.6, 5.0, 2.05)
sign([('朝の気分 Morning Mood', 0.36, M_TXT_G), ('夜明けの山の村', 0.32, M_TXT)], fx_(8.0) + 1.2, -3.4, fz_(8.0) - 2.7, 4.4, 1.25)
sign([('ロープウェイ', 0.34, M_TXT_O), ('マーブルマシン', 0.3, M_TXT)], fx_(13.0, 'oboe') + 1.4, -3.4, fz_(13.0, "oboe") - 2.7, 3.2, 1.25)
sign([('クラシック名曲', 0.36, M_TXT_G), ('エドヴァルド・グリーグ', 0.3, M_TXT)], fx_(18.0, 'oboe') + 1.3, -3.4, fz_(18.0, "oboe") - 2.7, 4.2, 1.25)
sign([('ノルウェーの作曲家', 0.32, M_TXT), ('グリーグ 1875年作曲', 0.34, M_TXT_G)], fx_(23.5) + 1.3, -3.4, fz_(23.5) - 2.7, 3.8, 1.25)
sign([('ビー玉 × 鉄琴 × からくり', 0.32, M_TXT_B), ('ペール・ギュント「朝」', 0.32, M_TXT_G)], fx_(28.5) + 1.3, -3.4, fz_(28.5) - 2.7, 4.6, 1.25)
# hook in the night sky (first seconds) and the closing greeting
def sky_text(s, size, mt, t0, t1, loc):
    o = text(s, loc, size, mt, rot=(math.pi / 2, 0, 0))
    a = np.clip((T - t0) / 0.3, 0, 1) * np.clip((t1 - T) / 0.4, 0, 1)
    for c in range(3): key_series(o, 'scale', c, np.maximum(0.001, a), 1e-3)
    return o
sky_text('玉が鳴るたび', 0.62, M_TXT, 0.0, 6.0, (1.0, 1.2, 5.6))
sky_text('夜が明けていく', 0.62, M_TXT_G, 0.3, 6.0, (1.0, 1.2, 4.75))
sky_text('おはよう！', 3.2, M_TXT_G, SUN['t_rise'] + 2.4, 99, (xs_sun, 40.0, twz + 3.2))

# ---------------- final burst: petals + light rays ----------------
cup_c = (np.array(EV['cups']['flute']) + np.array(EV['cups']['oboe'])) / 2
TB = SUN['t_rise'] + SUN['t_acc']
for k in range(120):
    cf = obj(prim('plane', M_LAMP), 'petal'); cf.color = (*[(1, 0.5, 0.7), (1, 0.8, 0.2), (1, 1, 1), (0.5, 0.8, 1)][k % 4], 0.25)
    a0 = rng.uniform(0.2, math.pi - 0.2); spd = rng.uniform(2.0, 5.0); vy = rng.uniform(-1.5, 2.5)
    P = np.zeros((NF, 3)); RT_ = np.zeros((NF, 3))
    for f in range(NF):
        s_ = T[f] - TB
        if s_ < 0: P[f] = cup_c; continue
        b_ = 1 - math.exp(-s_ / 0.45)
        P[f] = cup_c + [spd * math.cos(a0) * 1.4 * b_ + 0.2 * math.sin(3 * s_ + k), vy * b_ - 0.5, spd * math.sin(a0) * 0.9 * b_ - 0.6 * s_ + 0.3]
        RT_[f] = (s_ * (3 + k % 5), s_ * (2 + k % 3), s_ * 1.5)
    key_loc(cf, P)
    sc_ = np.where(T >= TB, 0.16, 0.001)
    for c_ in range(3): key_series(cf, 'scale', c_, sc_); key_series(cf, 'rotation_euler', c_, RT_[:, c_])

# ---------------- camera: follow the ball that is playing; the sun reveal pulls back ----------------
act = np.zeros(NF)                       # 0 = flute, 1 = oboe
def tbar(b): return T0 + (b - 1) * BAR
for f in range(NF):
    t = T[f]
    if tbar(5) - 0.5 <= t < tbar(9) - 0.5 or tbar(13) - 0.5 <= t < tbar(17) - 0.5: act[f] = 1.0
act = smooth(act, 0.35)
FOC = POS['flute'] * (1 - act[:, None]) + POS['oboe'] * act[:, None]
FOC = FOC + np.array([0.8, 0, 0.75])
st0 = np.clip((T - (T0 - 0.2)) / 0.6, 0, 1)
FOC[:, 2] = FOC[:, 2] * st0 + 1.3 * (1 - st0); FOC[:, 0] = FOC[:, 0] * st0 + 1.1 * (1 - st0)
WID = np.full(NF, 6.2)
WID[T < T0 + 0.2] = 6.2
rev = step(SUN['t_rise'] - 0.6, 2.4)
REV_F = np.array([xs_sun - 1.0, 0, float(cup_c[2]) + 4.0])
FOC = FOC * (1 - rev[:, None]) + REV_F * rev[:, None]
WID = WID * (1 - rev) + 17.0 * rev
FOC[:, 0] = smooth(FOC[:, 0], 0.5); FOC[:, 2] = smooth(FOC[:, 2], 0.7); WID = smooth(WID, 0.6)
LENS = 32.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 800
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24
    hk = 0.2 * (1 - rev[f]) + 0.05 * rev[f]
    CP[f] = (FOC[f, 0], -D, FOC[f, 2] + hk * D)
    TP[f] = (FOC[f, 0], 0.0, FOC[f, 2] + 0.2 + rev[f] * 2.0)
key_loc(cam, CP); key_loc(tgt, TP)

if MODE == 'test':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS)); sc.frame_set(f)
        sc.render.filepath = f'{HERE}/test_{f:04d}.png'
        t1 = time.time(); bpy.ops.render.render(write_still=True); print('frame', f, 'sec', round(time.time() - t1, 1))
elif MODE == 'range':
    sc.frame_start, sc.frame_end = int(args[1]), int(args[2])
    sc.render.image_settings.file_format = 'PNG'; sc.render.filepath = f'{HERE}/frames/f_'
    bpy.ops.render.render(animation=True)
if MODE == 'probe':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS)); sc.frame_set(f)
        print('t', tt_, 'cam', [round(v, 2) for v in cam.matrix_world.translation], 'WID', round(WID[f], 2), 'FOC', np.round(FOC[f], 2),
              'fl', np.round(POS['flute'][f], 2), 'ob', np.round(POS['oboe'][f], 2), 'dawn', np.round(POS['dawn'][f], 2), 'D', round(DAWN[f], 2))
