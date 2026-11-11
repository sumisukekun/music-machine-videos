"""Eine kleine Nachtmusik - 夜の宮殿の灯り (the lights of the night palace). Side view of a long palace hall at night.
Every violin I note = the lamp ball landing on a candlestick key (its candle lights in the first half); every violin II note =
the dancing couple on porcelain keys; every cello note = the gold ball on a gold bell or a bass bar of its track.
layout.py plans all motion (constant gravity, bounces that never add speed, rolling ramps, bucket lifts, see-saw);
this file builds and animates it.  Fonts: jp_sub.otf / jp_serif.otf = Noto Sans / Serif CJK JP Black (SIL OFL) subsets
of the characters used here (make_font.py).
python3 scene.py -- test t1 t2 ... | range f0 f1 | probe t ..."""
import os, sys, json, math, time
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
DUO = json.load(open(f'{HERE}/duo.json')); BASS = json.load(open(f'{HERE}/bass.json'))
LAY = json.load(open(f'{HERE}/layout.json'))
FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.3) * FPS); T = np.arange(NF) / FPS
G = LAY['G']; EV = LAY['ev']; LN = LAY['lanes']; ZO = LAY['zoff']
RL, RP, RB = LAY['R']['lamp'], LAY['R']['duo'], LAY['R']['bass']
GUST, DACAPO, T0, Q = E['gust'], E['dacapo'], E['t0'], E['q']
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
FONT = bpy.data.fonts.load(f'{HERE}/jp_sub.otf'); FONT_S = bpy.data.fonts.load(f'{HERE}/jp_serif.otf')
TXT_K = 2.75                 # these fonts render at ~0.36 x nominal size per glyph: scale so that a glyph is ~size wide
TXT_K = 2.75                 # this font renders at ~0.36 x nominal size per glyph: scale so that a glyph is ~size wide


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


def glowmat(name, gain=8.0, base=0.0, rough=0.3, metal=0.0, coat=0.5, trans=0.0):
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


_MC = {}
def prim(kind, m, *a):
    key = (kind, m.name if m else None) + tuple(a)
    if key in _MC: return _MC[key]
    bm = bmesh.new()
    if kind == 'cube': bmesh.ops.create_cube(bm, size=1.0)
    elif kind == 'bcube':
        bmesh.ops.create_cube(bm, size=1.0); bmesh.ops.bevel(bm, geom=bm.edges[:] + bm.verts[:], offset=0.18, segments=3, affect='EDGES')
    elif kind == 'suzu':                                   # round bell (suzu) with a slot and a ring on top
        bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=1.0)
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
    elif kind == 'icicle':                                 # hanging icicle, top at z = 0, length 1
        bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=1.0, radius2=0.0, depth=1.0)
        bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, -0.5))
        for v in bm.verts: v.co.z = -v.co.z - 1.0 if False else v.co.z
        bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=Matrix.Rotation(math.pi, 3, 'X'))
        bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, -0.5))
    me = bpy.data.meshes.new(kind); bm.to_mesh(me); bm.free()
    if kind in ('sph', 'cone', 'bell', 'icicle', 'cyl', 'bcube', 'suzu'):
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


def text(s, loc, size, m, rot=(math.pi / 2 - 0.3, 0, 0), align='CENTER', ext=0.02):
    cu = bpy.data.curves.new('t', 'FONT'); cu.body = s; cu.font = FONT; cu.size = size * TXT_K; cu.extrude = ext
    cu.align_x = align; cu.align_y = 'CENTER'
    o = link(bpy.data.objects.new('txt', cu)); o.location = loc; o.rotation_euler = rot; o.data.materials.append(m); return o


def pulse_alpha(times, decay=5.0, length=16, amp=1.0):
    a = np.zeros(NF)
    for t in times:
        f0 = int(round(t * FPS)); m_ = np.arange(max(0, f0), min(NF, f0 + length))
        a[m_] = np.maximum(a[m_], amp * np.exp(-(m_ - f0) / decay))
    return a


def text(s, loc, size, m, rot=(math.pi / 2 - 0.3, 0, 0), align='CENTER', ext=0.02, font=None):
    cu = bpy.data.curves.new('t', 'FONT'); cu.body = s; cu.font = font or FONT_S; cu.size = size * TXT_K; cu.extrude = ext
    cu.align_x = align; cu.align_y = 'CENTER'
    o = link(bpy.data.objects.new('txt', cu)); o.location = loc; o.rotation_euler = rot; o.data.materials.append(m); return o


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


def step_series(times_vals, default=0.0):
    """piecewise-constant series over frames from [(t, value), ...] (sorted)"""
    a = np.full(NF, default, float)
    for t, v in times_vals: a[T >= t - 1e-9] = v
    return a


def smooth(Xa, k):
    Y = Xa.copy(); a = 1 - math.exp(-1 / FPS / k)
    for f in range(1, len(Xa)): Y[f] = Y[f - 1] + (Xa[f] - Y[f - 1]) * a
    for f in range(len(Xa) - 2, -1, -1): Y[f] = Y[f + 1] + (Y[f] - Y[f + 1]) * a
    return Y


# ---------------- materials ----------------
M_GOLD = mat('gold', (1.0, 0.72, 0.3), 0.22, **{'Metallic': 0.95})
M_GOLDD = mat('goldd', (0.55, 0.36, 0.12), 0.35, **{'Metallic': 0.9})
M_GOLDB = mat('goldball', (1.0, 0.74, 0.25), 0.16, **{'Metallic': 0.85, 'Coat Weight': 0.6})
M_ROSE = mat('rose', (0.95, 0.25, 0.42), 0.1, **{'Coat Weight': 1.0, 'Emission Color': (1.0, 0.3, 0.5), 'Emission Strength': 0.35})
M_PEARL = mat('pearl', (0.95, 0.93, 0.9), 0.1, **{'Coat Weight': 1.0, 'Emission Color': (0.9, 0.9, 1.0), 'Emission Strength': 0.25})
M_LAMPB = glowmat('lampball', gain=14.0, base=0.15, rough=0.08, coat=1.0)
M_KEY = glowmat('key', gain=9.0, base=0.25, rough=0.06, coat=1.0)
M_BELL = glowmat('bell', gain=5.0, base=0.0, rough=0.25, metal=0.85)
M_WAX = mat('wax', (0.92, 0.86, 0.74), 0.45, **{'Subsurface Weight': 0.3})
M_FLAME = emis('flame', (1.0, 0.55, 0.16), 22.0)
M_FLAMEC = emis('flamecore', (1.0, 0.9, 0.6), 30.0)
M_EMBER = emis('ember', (1.0, 0.3, 0.05), 6.0)
M_WALL = mat('wall', (0.32, 0.05, 0.07), 0.6)
M_PANEL = mat('panel', (0.18, 0.03, 0.05), 0.5, **{'Coat Weight': 0.3})
M_COL = mat('column', (0.85, 0.82, 0.75), 0.35)
M_DARK = mat('dark', (0.03, 0.03, 0.04), 0.6)
M_WOOD = mat('wood', (0.35, 0.16, 0.07), 0.45, **{'Coat Weight': 0.5})
M_STEEL = mat('steel', (0.6, 0.62, 0.68), 0.3, **{'Metallic': 0.9})
M_GLASS = mat('glass', (0.8, 0.9, 1.0), 0.05, **{'Alpha': 0.15})
M_VELVET = mat('velvet', (0.5, 0.03, 0.06), 0.8, **{'Sheen Weight': 0.8})
M_OBJ = objcol('objcol', 0.5)
M_LAMP = objcol('lampcol', 0.4, emit=6.0)
M_FX = fxmat('fx', 8.0)
M_TXT = emis('txtw', (1.0, 0.95, 0.85), 4.0); M_TXT_G = emis('txtg', (1.0, 0.78, 0.35), 6.0); M_TXT_P = emis('txtp', (1.0, 0.6, 0.7), 5.0)
M_PLATE = mat('plate', (0.06, 0.02, 0.03), 0.3, **{'Coat Weight': 1.0})
# marble floor: checker of cream and dark green marble
M_FLOOR = bpy.data.materials.new('floor'); M_FLOOR.use_nodes = True; _nt = M_FLOOR.node_tree; _b = _nt.nodes['Principled BSDF']
_tc = _nt.nodes.new('ShaderNodeTexCoord'); _ck = _nt.nodes.new('ShaderNodeTexChecker'); _ck.inputs['Scale'].default_value = 1.0
_ck.inputs['Color1'].default_value = (0.42, 0.36, 0.3, 1); _ck.inputs['Color2'].default_value = (0.03, 0.08, 0.07, 1)
_mp = _nt.nodes.new('ShaderNodeMapping'); _mp.inputs['Scale'].default_value = (0.8, 0.8, 0.8)
_nt.links.new(_tc.outputs['Object'], _mp.inputs['Vector']); _nt.links.new(_mp.outputs[0], _ck.inputs['Vector'])
_nt.links.new(_ck.outputs['Color'], _b.inputs['Base Color']); _b.inputs['Roughness'].default_value = 0.12; _b.inputs['Coat Weight'].default_value = 1.0

PITCH_COL = [(1.0, 0.12, 0.12), (1.0, 0.35, 0.05), (1.0, 0.6, 0.05), (1.0, 0.85, 0.1), (0.55, 1.0, 0.1), (0.1, 1.0, 0.3),
             (0.05, 1.0, 0.75), (0.05, 0.7, 1.0), (0.15, 0.35, 1.0), (0.45, 0.2, 1.0), (0.75, 0.2, 1.0), (1.0, 0.2, 0.65)]
def pcol(p, light=0.0):
    c = PITCH_COL[p % 12]; return tuple(min(1, x * (1 - light) + light) for x in c)

# ---------------- balls ----------------
BALL = {}
lamp_glow = np.ones(NF)
for f in range(NF):
    t = T[f]
    if t < GUST: lamp_glow[f] = 1.0
    elif t < DACAPO: lamp_glow[f] = max(0.06, math.exp(-(t - GUST) / 0.12))
    else: lamp_glow[f] = min(1.0, 0.06 + 1.6 * (1 - math.exp(-(t - DACAPO) / 0.08)))
for n, b in BP.items():
    rr = b.r
    bpy.ops.mesh.primitive_uv_sphere_add(radius=rr, segments=32, ring_count=16); o = bpy.context.object; o.name = n
    for p in o.data.polygons: p.use_smooth = True
    o.data.materials.append({'lamp': M_LAMPB, 'rose': M_ROSE, 'pearl': M_PEARL, 'bass': M_GOLDB}[n])
    bpy.ops.mesh.primitive_torus_add(major_radius=rr * 1.0, minor_radius=rr * 0.12, major_segments=32, minor_segments=8)
    hb = bpy.context.object; hb.parent = o; hb.data.materials.append(M_GOLD if n != 'bass' else M_GOLDD)
    key_loc(o, POS[n])
    sp = np.zeros(NF)
    for f in range(1, NF): sp[f] = sp[f - 1] + (POS[n][f, 0] - POS[n][f - 1, 0]) / rr
    o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, sp)
    BALL[n] = o
BALL['lamp'].color = (1.0, 0.5, 0.12, 1.0); key_series(BALL['lamp'], 'color', 3, lamp_glow, 1e-3)
ll = bpy.data.lights.new('lamplight', 'POINT'); ll.color = (1.0, 0.62, 0.3); ll.shadow_soft_size = 0.25
llo = link(bpy.data.objects.new('lamplight', ll)); llo.parent = BALL['lamp']
for f in range(0, NF, 2):
    ll.energy = float(45 * lamp_glow[f]); ll.keyframe_insert('energy', frame=f)


def strip_mesh(X, Z, y0, y1, th, m, name):
    bm = bmesh.new(); vs = []
    for x, z in zip(X, Z):
        vs.append([bm.verts.new((x, y0, z)), bm.verts.new((x, y1, z)), bm.verts.new((x, y1, z - th)), bm.verts.new((x, y0, z - th))])
    for a, b in zip(vs, vs[1:]):
        for k in range(4): bm.faces.new((a[k], b[k], b[(k + 1) % 4], a[(k + 1) % 4]))
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new(name, me)); o.data.materials.append(m)
    return o


def riser(objs, t_hit, anchor, lead=1.2, stay=0.9, pop=0.3, shrink=0.35):
    """objects pop into existence (scale around `anchor`) at t_hit-lead and vanish after t_hit+stay"""
    anchor = Vector(anchor)
    e = link(bpy.data.objects.new('pop', None)); e.location = anchor
    for o in objs: o.parent = e; o.matrix_parent_inverse = Matrix.Translation(-anchor)
    s_ = np.zeros(NF)
    for f in range(NF):
        t = T[f]; u1 = (t - (t_hit - lead)) / pop; u2 = (t - (t_hit + stay)) / shrink
        if u1 <= 0: a = 0.0
        elif u1 < 1: a = 1.18 * math.sin(u1 * math.pi / 2) if u1 < 0.7 else 1.18 - 0.18 * (u1 - 0.7) / 0.3
        else: a = 1.0
        b = 1.0 - min(1, max(0, u2)) ** 1.5
        s_[f] = max(0.001, a * b)
    for k in range(3): key_series(e, 'scale', k, s_, 1e-4)
    return e


def flash(o, t, col, n=None, base=0.0):
    fh = int(round(t * FPS))
    o.color = (*col, base); o.keyframe_insert('color', index=3, frame=max(0, fh - 1))
    for df, a in ((0, 1.0), (4, 0.55), (18, max(base, 0.1))):
        o.color = (*col, a); o.keyframe_insert('color', index=3, frame=fh + df)
    if n is not None:
        b0 = o.location.copy(); nv = Vector(n)
        for df, d in ((-1, 0), (1, -0.03), (4, 0.01), (8, 0)):
            o.location = b0 + nv * d; o.keyframe_insert('location', frame=max(0, fh + df))


# ---------------- floor of the hall: below every lamp contact, only ever steps down ----------------
allc = sorted([(h['p'][0], h['p'][2]) for h in BP['lamp'].hits])
AX = np.array([a[0] for a in allc]); AZ = np.array([a[1] for a in allc])
FLX = np.arange(-16.0, AX.max() + 24, 0.25)
FLZ = np.array([AZ[np.abs(AX - x) < 2.0].min() if (np.abs(AX - x) < 2.0).any() else np.nan for x in FLX])
s_ = ~np.isnan(FLZ); FLZ = np.interp(FLX, FLX[s_], FLZ[s_]) - 1.35
FLZ = np.minimum.accumulate(FLZ)
FLZ = np.round(FLZ / 0.18) * 0.18                       # marble stair steps of 18 cm
def floor_z_at(x): return float(np.interp(x, FLX, FLZ))
WALL_Y = 4.7
strip_mesh(FLX, FLZ, -6.0, WALL_Y, 0.4, M_FLOOR, 'floor')
# back wall (burgundy) with gold-framed panels, white pilasters, tall arched windows
WX0, WX1 = FLX[0], FLX[-1]
bpy.ops.mesh.primitive_plane_add(size=1.0); wall = bpy.context.object; wall.data.materials.append(M_WALL)
wall.location = ((WX0 + WX1) / 2, WALL_Y + 0.05, -6); wall.rotation_euler = (math.pi / 2, 0, 0); wall.scale = (WX1 - WX0, 40, 1)
M_NIGHT = emis('night', (0.05, 0.1, 0.3), 1.2)
WIN = []
for k, x0 in enumerate(np.arange(WX0 + 3, WX1 - 2, 6.0)):
    zf = floor_z_at(x0)
    for dx in (-1.5, 1.5):                                   # pilasters
        box((x0 + dx + 3.0, WALL_Y - 0.15, zf + 5.5), (0.5, 0.3, 11.0), M_COL)
        box((x0 + dx + 3.0, WALL_Y - 0.32, zf + 0.25), (0.62, 0.1, 0.5), M_GOLDD)
    # tall window (night sky) with gold frame and red curtains
    box((x0, WALL_Y + 0.02, zf + 4.6), (1.9, 0.02, 4.4), M_NIGHT, name='winsky')
    for xx in (x0 - 1.0, x0 + 1.0): box((xx, WALL_Y - 0.06, zf + 4.6), (0.1, 0.12, 4.5), M_GOLD)
    box((x0, WALL_Y - 0.06, zf + 2.35), (2.1, 0.12, 0.1), M_GOLD); box((x0, WALL_Y - 0.06, zf + 6.85), (2.1, 0.12, 0.1), M_GOLD)
    box((x0, WALL_Y - 0.05, zf + 4.6), (0.05, 0.08, 4.4), M_GOLD)
    for xx in (x0 - 1.35, x0 + 1.35): box((xx, WALL_Y - 0.25, zf + 4.3), (0.45, 0.12, 5.6), M_VELVET, name='curtain')
    WIN.append((x0, zf))
    box((x0 + 3.0, WALL_Y - 0.03, zf + 1.6), (3.4, 0.04, 1.8), M_PANEL); box((x0 + 3.0, WALL_Y - 0.05, zf + 1.6), (3.5, 0.03, 1.9), M_GOLDD)
# stars in the windows
for x0, zf in WIN:
    for j in range(6):
        sphere((x0 + rng.uniform(-0.85, 0.85), WALL_Y - 0.01, zf + rng.uniform(2.6, 6.6)), 0.02, M_TXT, (6, 4))

# ---------------- candles: (flame object, lit intervals) ----------------
CANDLES = []                    # dict(fl=flame obj, x, on=[(t0, t1)], who)
FUSE_V = 9.0
X_BRAZ = EV['brazier'][0]
def t_relight(x): return DACAPO + 0.35 + abs(x - X_BRAZ) / FUSE_V
X_WIN_GUST = float(np.interp(GUST, T, POS['lamp'][:, 0])) + 2.6
def t_out(x): return GUST + 0.12 + max(0.0, X_WIN_GUST - x) / 14.0


def candle(base, h=0.22, r=0.045, scale=1.0, who='lamp'):
    """wax candle standing on `base`; returns the flame (a parent empty that is scaled 0..1)"""
    c = rod(base, Vector(base) + Vector((0, 0, h * scale)), r * scale, M_WAX, 10)
    fe = link(bpy.data.objects.new('flame', None)); fe.location = Vector(base) + Vector((0, 0, h * scale + 0.005))
    f1 = obj(prim('cone', M_FLAME, 10, 1.0, 0.0, 1.0), 'fl'); f1.parent = fe; f1.location = (0, 0, 0.06 * scale); f1.scale = (0.035 * scale, 0.035 * scale, 0.12 * scale)
    f2 = sphere((0, 0, 0.03 * scale), 0.032 * scale, M_FLAMEC, (10, 6)); f2.parent = fe
    return c, fe


def animate_flame(fe, on, k):
    s = np.zeros(NF); lean = np.zeros(NF)
    for t0, t1 in on:
        for f in range(NF):
            t = T[f]
            if t < t0 or t > t1 + 0.25: continue
            a = min(1.0, (t - t0) / 0.12)
            if t > t1: a *= max(0.0, 1 - (t - t1) / 0.2); lean[f] = -0.9 * min(1, (t - t1) / 0.12)
            s[f] = max(s[f], a)
    s = np.maximum(s, 0.001)
    for c in range(3): key_series(fe, 'scale', c, s, 2e-3)
    fe.rotation_mode = 'XYZ'; key_series(fe, 'rotation_euler', 1, lean, 1e-3)


# ---------------- lamp keys: candlesticks (first half and da capo), moon keys (piano section), the brazier ----------------
KEY_W, KEY_D, KEY_TH = 0.32, 0.52, 0.08
PIECES = []
for h in BP['lamp'].hits:
    kd = h['kind']
    if kd not in ('candle', 'dark', 'candle2', 'brazier'): continue
    nv = np.array(h['n']); top = np.array(h['p']) - RL * nv
    col = pcol(h['pitch'], 0.0)
    zf = floor_z_at(top[0])
    if kd == 'brazier':
        o = obj(prim('cyl', M_EMBER, 32), 'coals'); o.matrix_world = Matrix.Translation(Vector(top - nv * 0.03)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((0.42, 0.42, 0.06, 1))
        bowl = obj(prim('cone', M_GOLD, 32, 0.25, 0.46, 0.3), 'bowl'); bowl.location = tuple(top + [0, 0, -0.2])
        for a_ in range(3):
            ph = 2 * math.pi * a_ / 3 + 0.3
            rod((top[0] + 0.2 * math.cos(ph), top[1] + 0.2 * math.sin(ph), top[2] - 0.3), (top[0] + 0.42 * math.cos(ph), top[1] + 0.42 * math.sin(ph), zf), 0.03, M_GOLD, 8)
        EV['braz_top'] = top.tolist()
        PIECES.append(dict(kind=kd, t=h['t'], top=top, n=nv, hw=0.42, hd=0.42, th=0.06, ball='lamp'))
        continue
    ctr = top - nv * KEY_TH / 2
    o = obj(prim('bcube', M_KEY), 'key')
    o.matrix_world = Matrix.Translation(Vector(ctr)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((KEY_W, KEY_D, KEY_TH, 1))
    if kd == 'dark': col = pcol(h['pitch'], 0.45)
    o.color = (*col, 0.0); flash(o, h['t'], col, nv, base=0.08 if kd != 'dark' else 0.0)
    stem = rod((top[0], top[1], top[2] - KEY_TH - 0.01), (top[0], top[1], zf), 0.03, M_GOLD, 8)
    foot = obj(prim('cone', M_GOLD, 16, 0.14, 0.04, 0.12), 'foot'); foot.location = (top[0], top[1], zf + 0.06)
    parts = [o, stem, foot]
    if kd in ('candle', 'candle2'):
        arm = rod((top[0], top[1] + 0.2, top[2] - 0.05), (top[0], top[1] + 0.36, top[2] - 0.05), 0.015, M_GOLD, 6)
        cup = obj(prim('cyl', M_GOLD, 12), 'cup'); cup.location = (top[0], top[1] + 0.36, top[2] - 0.05); cup.scale = (0.06, 0.06, 0.02)
        cw, fe = candle((top[0], top[1] + 0.36, top[2] - 0.04))
        parts += [arm, cup, cw, fe]
        on = [(h['t'], t_out(top[0]))] if kd == 'candle' else []
        on.append((t_relight(top[0]) if kd == 'candle' else h['t'], 1e9))
        CANDLES.append(dict(fl=fe, x=top[0], on=on, who='lamp'))
    PIECES.append(dict(kind=kd, t=h['t'], top=top, n=nv, hw=KEY_W / 2, hd=KEY_D / 2, th=KEY_TH / 2, ball='lamp'))
    if h['t'] > T0 + 0.1: riser(parts, h['t'], top, lead=1.2, stay=1e6)

# ---------------- ramps (lamp xylophone, bass tracks) ----------------
for R_ in EV['ramps']:
    al = R_['alpha']; p0 = np.array(R_['p0']); r = R_['r']; d = np.array([math.cos(al), 0, -math.sin(al)]); nrm = np.array(R_['n'])
    L = R_['L']; who = R_['who']
    s0, s1 = -0.35, L + 0.05
    a = p0 - r * nrm + d * s0; b = p0 - r * nrm + d * s1
    y0 = p0[1]
    hw = 0.17 if who == 'lamp' else 0.24
    for yy in (y0 - hw, y0 + hw):
        rod((a[0], yy, a[2] + 0.04), (b[0], yy, b[2] + 0.04), 0.025, M_GOLD, 8)
    for s in np.arange(s0 + 0.1, s1, 1.2):
        q = p0 - r * nrm + d * s
        zb_ = floor_z_at(q[0]) if who == 'lamp' else q[2] - 0.9
        for yy in (y0 - hw, y0 + hw): rod((q[0], yy, q[2] + 0.02), (q[0], yy, zb_), 0.025, M_GOLDD, 6)
    if who == 'bass':                                       # the track sits on a narrow shelf of the upper gallery
        X = np.linspace(a[0] - 0.4, b[0] + 0.4, 30); Zs_ = np.interp(X, [a[0], b[0]], [a[2], b[2]]) - 0.9
        strip_mesh(X, Zs_, y0 - 0.45, y0 + 0.45, 0.12, M_WOOD, 'shelf')
        strip_mesh(X, Zs_ + 0.12, y0 - 0.5, y0 - 0.45, 0.16, M_GOLD, 'shelfrail')
    for j, bar in enumerate(R_['bars']):
        pc = np.array(bar['p']); top = pc - r * nrm
        col = pcol(bar['pitch'], 0.1 if who == 'lamp' else 0.3)
        o = obj(prim('bcube', M_KEY if who == 'lamp' else M_BELL), 'xbar')
        o.matrix_world = Matrix.Translation(Vector(top - nrm * 0.03)) @ frame_from_n(nrm).to_4x4() @ Matrix.Diagonal((0.09, 2 * hw + 0.06, 0.06, 1))
        o.color = (*col, 0.0); flash(o, bar['t'], col if who == 'lamp' else (1.0, 0.75, 0.35), base=0.05)
        PIECES.append(dict(kind='xbar', t=bar['t'], top=top, n=nrm, hw=0.045, hd=hw + 0.03, th=0.03, ball=who))
        if who == 'lamp':
            cw, fe = candle((top[0], y0 + hw + 0.12, top[2] + 0.0), h=0.16, r=0.035)
            on = [(bar['t'], t_out(top[0])), (t_relight(top[0]), 1e9)] if R_['pass'] == 1 else [(bar['t'], 1e9)]
            CANDLES.append(dict(fl=fe, x=top[0], on=on, who='lamp'))
    tag = 'シロフォン坂' if who == 'lamp' else 'ベースの線路'
    if R_['pass'] == 1:
        q = p0 - r * nrm + d * (L * 0.3)
        text(tag, (q[0], y0 - 0.55, q[2] + (0.75 if who == 'lamp' else 1.2)), 0.26, M_TXT_G, font=FONT)

# ---------------- couple keys (porcelain, pop up and vanish) and bass bells ----------------
for n in ('rose', 'pearl'):
    for h in BP[n].hits:
        if h['kind'] not in ('key', 'uni'): continue                 # one porcelain tile per dancer (same pitch for a single note)
        nv = np.array(h['n']); top = np.array(h['p']) - RP * nv
        o = obj(prim('bcube', M_KEY), 'duokey2')
        o.matrix_world = Matrix.Translation(Vector(top - nv * 0.05)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((0.38, 0.38, 0.1, 1))
        col = pcol(h['pitch'], 0.25); o.color = (*col, 0); flash(o, h['t'], col, nv)
        po = rod((top[0], top[1], top[2] - 0.1), (top[0], top[1], top[2] - 0.7), 0.025, M_GOLD, 8)
        PIECES.append(dict(kind=h['kind'], t=h['t'], top=top, n=nv, hw=0.19, hd=0.19, th=0.05, ball=n))
        QUICK = h['t'] > E['grid'][-2][2] - 0.2                         # last bars: the couple slides over to the platform
        if h['t'] > T0 + 0.1: riser([o, po], h['t'], top, **(dict(lead=0.15, stay=0.05, pop=0.1, shrink=0.15) if QUICK else dict(lead=0.35, pop=0.15)))
for h in BP['bass'].hits:
    if h['kind'] != 'bell': continue
    nv = np.array(h['n']); top = np.array(h['p']) - RB * nv; p = h['pitch']
    sz = 0.34 + 0.01 * (55 - p)
    o = obj(prim('bell', M_BELL, 32), 'bell')
    o.matrix_world = Matrix.Translation(Vector(top)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((sz * 0.75, sz * 0.75, sz, 1))
    o.color = (1.0, 0.72, 0.3, 0); flash(o, h['t'], (1.0, 0.72, 0.3))
    po = rod(top - nv * sz, top - nv * sz - np.array([0, 0, 0.7]), 0.04, M_GOLDD, 8)
    PIECES.append(dict(kind='bell', t=h['t'], top=top, n=nv, hw=0.42 * sz * 0.75, hd=0.42 * sz * 0.75, th=0.05, ball='bass'))
    if h['t'] > T0 + 0.1: riser([o, po], h['t'], top)
# couple: a gallery balcony under their keys; pitch names of the couple / bass shown on small plaques are not needed

def gallery(name, y, depth, below, kinds):
    hs = [h for n in name for h in BP[n].hits if h['kind'] in kinds]
    xs_ = np.array([h['p'][0] for h in hs]); zz = np.array([h['p'][2] for h in hs])
    X = np.arange(xs_.min() - 1.2, xs_.max() + 1.2, 0.5)
    near = np.array([(np.abs(xs_ - x) < 1.2).any() for x in X])
    Z = np.array([zz[np.abs(xs_ - x) < 1.6].min() if (np.abs(xs_ - x) < 1.6).any() else np.nan for x in X])
    ok = ~np.isnan(Z); Z = np.interp(X, X[ok], Z[ok]) - below
    for _ in range(3): Z = np.minimum(Z, np.convolve(np.pad(Z, 3, mode='edge'), np.ones(7) / 7, 'valid'))
    segs = []; cur = []
    for j in range(len(X)):
        if near[j]: cur.append(j)
        elif cur: segs.append(cur); cur = []
    if cur: segs.append(cur)
    for sg in segs:
        if len(sg) < 2: continue
        Xs, Zs = X[sg], Z[sg]
        strip_mesh(Xs, Zs, y - depth, y + depth, 0.18, M_WOOD, 'gallery')
        strip_mesh(Xs, Zs + 0.3, y - depth - 0.06, y - depth, 0.34, M_GOLD, 'galrail')
        for x, z in list(zip(Xs, Zs))[::3]: rod((x, y - depth - 0.03, z), (x, y - depth - 0.03, z + 0.3), 0.012, M_GOLD, 6)
        for x, z in list(zip(Xs, Zs))[::12]: rod((x, y + depth - 0.1, z - 0.18), (x, y + depth - 0.1, floor_z_at(x)), 0.12, M_COL, 12)
    return X, Z
GX_D, GZ_D = gallery(['rose', 'pearl'], LN['duo'], 0.55, 0.85, ('uni', 'key'))
gallery(['bass'], LN['bass'], 0.5, 0.9 + 0.35, ('bell',))

# ---------------- bucket lifts (bass), with a winch gear ----------------
for el in EV['elev']:
    pin = np.array(el['p_in']); H = el['H']
    bk = link(bpy.data.objects.new('bucket', None)); bk.location = tuple(pin - [0, 0, RB])
    box((0, 0, -0.04), (0.8, 0.7, 0.06), M_GOLDD).parent = bk
    for sx in (-1, 1):
        box((0.4 * sx, 0, 0.12), (0.05, 0.7, 0.3), M_GOLDD).parent = bk
        box((0, 0.35 * sx, 0.12), (0.8, 0.05, 0.3), M_GOLDD).parent = bk
    Z = np.zeros(NF); RT = np.zeros(NF)
    for f in range(NF):
        t = T[f]; u = (t - el['t_up0']) / (el['t_up1'] - el['t_up0']); u = min(1, max(0, u)); u = u * u * (3 - 2 * u)
        Z[f] = pin[2] - RB + H * u
        v = (t - (el['t_rel'] - 0.12)) / 0.12
        RT[f] = 0.0 if v < 0 else -0.55 * min(1, v) * (1 if t < el['t_rel'] + 0.6 else max(0, 1 - (t - el['t_rel'] - 0.6) / 0.4))
    key_series(bk, 'location', 2, Z); bk.rotation_mode = 'XYZ'; key_series(bk, 'rotation_euler', 1, RT, 1e-3)
    top = pin[2] + H + 1.2
    for sy in (-0.45, 0.45): rod((pin[0], pin[1] + sy, pin[2] - RB - 0.8), (pin[0], pin[1] + sy, top), 0.035, M_STEEL, 8)
    rod((pin[0], pin[1] - 0.45, top), (pin[0], pin[1] + 0.45, top), 0.04, M_STEEL, 8)
    gear = link(bpy.data.objects.new('gear', None)); gear.location = (pin[0], pin[1] - 0.55, top)
    g0 = obj(prim('cyl', M_GOLD, 32), 'gearw'); g0.parent = gear; g0.scale = (0.32, 0.32, 0.06); g0.rotation_euler = (math.pi / 2, 0, 0)
    for k in range(12):
        ph = 2 * math.pi * k / 12
        tth = box((0.36 * math.cos(ph), 0, 0.36 * math.sin(ph)), (0.08, 0.06, 0.08), M_GOLD, rot=(0, -ph, 0)); tth.parent = gear
    ch = rod((pin[0], pin[1], top), (pin[0], pin[1], pin[2] - RB + 0.2), 0.012, M_STEEL, 6)
    ang = np.array([-(Z[f] - Z[0]) / 0.32 for f in range(NF)])
    gear.rotation_mode = 'XYZ'; key_series(gear, 'rotation_euler', 1, ang, 1e-4)
    text('リフト', (pin[0] - 0.75, pin[1] - 0.6, pin[2] + 0.2), 0.22, M_TXT_G, font=FONT)

# ---------------- the gust: a window bursts open, wind streaks, every candle goes out ----------------
xg = X_WIN_GUST; zfg = floor_z_at(xg)
box((xg, WALL_Y + 0.03, zfg + 4.6), (2.4, 0.02, 4.6), M_NIGHT, name='gustsky')
for side in (-1, 1):
    hg = link(bpy.data.objects.new('casement', None)); hg.location = (xg + side * 1.2, WALL_Y - 0.12, zfg + 4.6)
    pane = box((-side * 0.6, 0, 0), (1.2, 0.05, 4.4), M_GLASS, name='pane'); pane.parent = hg
    for zz in (-2.2, 0, 2.2): box((-side * 0.6, -0.02, zz), (1.22, 0.07, 0.07), M_GOLD).parent = hg
    box((-side * 1.18, -0.02, 0), (0.07, 0.07, 4.4), M_GOLD).parent = hg; box((-side * 0.02, -0.02, 0), (0.07, 0.07, 4.4), M_GOLD).parent = hg
    an = np.zeros(NF)
    for f in range(NF):
        u = (T[f] - GUST) / 0.22
        if u > 0: an[f] = side * (1.9 * min(1, u) ** 1.5 + 0.08 * math.exp(-(T[f] - GUST - 0.22) / 0.4) * math.sin(18 * (T[f] - GUST)) * (u > 1))
    hg.rotation_mode = 'XYZ'; key_series(hg, 'rotation_euler', 2, an, 1e-4)
    cur = box((xg + side * 1.55, WALL_Y - 0.35, zfg + 4.3), (0.5, 0.12, 5.6), M_VELVET, name='curtainG')
    cur.rotation_mode = 'XYZ'
    rx = np.array([(-0.5 * math.exp(-max(0, T[f] - GUST) / 1.2) * (1 + 0.3 * math.sin(9 * T[f])) if T[f] > GUST else 0.0) for f in range(NF)])
    key_series(cur, 'rotation_euler', 0, rx, 1e-3)
moon = bpy.data.lights.new('moonshaft', 'SPOT'); moon.color = (0.55, 0.7, 1.0); moon.spot_size = math.radians(40); moon.spot_blend = 0.5
mo = link(bpy.data.objects.new('moonshaft', moon)); mo.location = (xg, WALL_Y + 1.5, zfg + 7.5); mo.rotation_euler = (math.radians(-35), 0, 0)
for f in range(0, NF, 3):
    t = T[f]; moon.energy = float(0 if t < GUST else 4000 * min(1, (t - GUST) / 0.3) * (1 if t < DACAPO else max(0.25, 1 - (t - DACAPO) / 2)))
    moon.keyframe_insert('energy', frame=f)
for k in range(36):                                          # wind streaks (fly out of the window, across the hall)
    st = obj(prim('cube', M_FX), 'wind'); st.color = (0.75, 0.85, 1.0, 0)
    t0 = GUST + 0.05 + rng.uniform(0, 0.9); sp = rng.uniform(9, 16); zz = zfg + rng.uniform(1.0, 6.5); yy = rng.uniform(-1.5, 3.8)
    P = np.zeros((NF, 3)); A = np.zeros(NF)
    for f in range(NF):
        u = T[f] - t0
        P[f] = (xg - sp * max(0, u), yy if u > 0 else WALL_Y, zz - 0.4 * max(0, u))
        A[f] = 0 if u < 0 or u > 1.4 else 0.35 * math.sin(math.pi * u / 1.4)
    key_loc(st, P); st.scale = (0.9, 0.02, 0.02); key_series(st, 'color', 3, A, 1e-3)

# ---------------- wall sconces (3 candles each) and ceiling chandeliers: lit by the fuse at the da capo ----------------
SCON = np.arange(WX0 + 1.5, WX1 - 1.0, 1.8)
for k, x0 in enumerate(SCON):
    zf = floor_z_at(x0); zc = zf + 2.6
    rod((x0, WALL_Y - 0.02, zc), (x0, WALL_Y - 0.3, zc), 0.025, M_GOLD, 6)
    o = sphere((x0, WALL_Y - 0.1, zc + 0.05), 0.07, M_GOLD, (10, 6))
    for dx in (-0.2, 0.0, 0.2):
        zz = zc + (0.1 if dx == 0 else 0.0)
        rod((x0, WALL_Y - 0.3, zc), (x0 + dx, WALL_Y - 0.3, zz), 0.015, M_GOLD, 6)
        cw, fe = candle((x0 + dx, WALL_Y - 0.3, zz), h=0.18, r=0.035)
        CANDLES.append(dict(fl=fe, x=x0, on=[(t_relight(x0), 1e9)], who='sconce'))
# the fuse: a gold cord along the wall under the sconces; a spark runs both ways from the brazier
FZ = lambda x: floor_z_at(x) + 2.45
fx_ = np.arange(WX0, WX1, 0.5)
for xa, xb in zip(fx_, fx_[1:]): rod((xa, WALL_Y - 0.22, FZ(xa)), (xb, WALL_Y - 0.22, FZ(xb)), 0.012, M_GOLDD, 5)
bt = np.array(EV['braz_top'])
rod(bt + [0, 0.3, 0.0], (bt[0], WALL_Y - 0.22, FZ(bt[0])), 0.012, M_GOLDD, 5)
for sgn in (-1, 1):
    spk = sphere((0, 0, 0), 0.07, M_FX); spk.color = (1.0, 0.75, 0.3, 0)
    P = np.zeros((NF, 3)); A = np.zeros(NF)
    for f in range(NF):
        u = T[f] - DACAPO
        if u < 0: P[f] = bt; continue
        if u < 0.35:
            w = u / 0.35; P[f] = bt + (np.array([bt[0], WALL_Y - 0.22, FZ(bt[0])]) - bt) * w
        else:
            x = bt[0] + sgn * FUSE_V * (u - 0.35); P[f] = (x, WALL_Y - 0.22, FZ(x))
        A[f] = 1.0 if WX0 < P[f][0] < WX1 else 0.0
    key_loc(spk, P); key_series(spk, 'color', 3, A, 1e-3)
DECO = []
def chandelier(c, s=1.0, bulbs=8, chain=8.0, t_on=None, count=True):
    cx, cy, cz = c
    bpy.ops.mesh.primitive_torus_add(major_radius=0.55 * s, minor_radius=0.045 * s, major_segments=40, minor_segments=8, location=(cx, cy, cz))
    bpy.context.object.data.materials.append(M_GOLD)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.3 * s, minor_radius=0.035 * s, major_segments=32, minor_segments=8, location=(cx, cy, cz + 0.35 * s))
    bpy.context.object.data.materials.append(M_GOLD)
    rod((cx, cy, cz - 0.3 * s), (cx, cy, cz + 0.6 * s), 0.05 * s, M_GOLD, 10)
    if chain: rod((cx, cy, cz + 0.6 * s), (cx, cy, cz + chain), 0.02, M_GOLD, 6)
    fls = []
    for j in range(bulbs):
        ph = 2 * math.pi * j / bulbs
        bx, by = cx + 0.55 * s * math.cos(ph), cy + 0.55 * s * math.sin(ph)
        cw, fe = candle((bx, by, cz + 0.03 * s), h=0.14 * s, r=0.03 * s, scale=1.0)
        sphere((bx, by, cz - 0.05 * s), 0.035 * s, M_GLASS, (8, 6))
        fls.append(fe)
        if t_on is not None: CANDLES.append(dict(fl=fe, x=cx, on=[(t_on + 0.02 * j, 1e9)], who='deco' if count else 'grand'))
    return fls
for x0 in np.arange(WX0 + 6, WX1 - 4, 12.0):
    DECO.append(x0); chandelier((x0, 2.6, floor_z_at(x0) + 8.2), 1.25, bulbs=8, t_on=t_relight(x0) + 0.2)
# the dark chandelier the balls drop from
c0 = EV['chandelier0']
chandelier((c0['x'], c0['y'], c0['z'] + RL + 0.25), 1.0, bulbs=8, chain=6.0)
hinge0 = link(bpy.data.objects.new('hinge0', None)); hinge0.location = (c0['x'] - 0.3, c0['y'], c0['z'] - RL - 0.05)
door0 = box((0.3, 0, 0.02), (0.6, 0.5, 0.04), M_GOLD, name='trapdoor0'); door0.parent = hinge0
ang = np.array([min(1.0, max(0, (T[f] - c0['t_release'] + 0.02) / 0.18)) ** 2 * 1.6 for f in range(NF)])
hinge0.rotation_mode = 'XYZ'; key_series(hinge0, 'rotation_euler', 1, ang)
for k_, n in (('duo_drop', 'rose'), ('bass_drop', 'bass')):
    d0 = EV[k_]; rr = 0.5 if n == 'rose' else 0.6
    chandelier((d0['x'], d0['y'], d0['z'] + 0.45), rr, bulbs=6, chain=5.0)

# ---------------- brazier: glows in the dark, bursts into flame at the da capo ----------------
bfe = link(bpy.data.objects.new('brazflame', None)); bfe.location = tuple(bt + [0, 0, 0.02])
for k in range(7):
    ph = 2 * math.pi * k / 7
    f1 = obj(prim('cone', M_FLAME, 10, 1.0, 0.0, 1.0), 'bfl'); f1.parent = bfe
    f1.location = (0.22 * math.cos(ph), 0.22 * math.sin(ph), 0.25); f1.scale = (0.1, 0.1, 0.5)
f1 = obj(prim('cone', M_FLAMEC, 12, 1.0, 0.0, 1.0), 'bflc'); f1.parent = bfe; f1.location = (0, 0, 0.35); f1.scale = (0.16, 0.16, 0.7)
bs = np.array([0.001 if T[f] < DACAPO else min(1.25, 1.6 * (1 - math.exp(-(T[f] - DACAPO) / 0.06))) * (0.92 + 0.08 * math.sin(19 * T[f])) for f in range(NF)])
for c in range(3): key_series(bfe, 'scale', c, bs, 2e-3)
bl = bpy.data.lights.new('brazlight', 'POINT'); bl.color = (1.0, 0.5, 0.2); bl.shadow_soft_size = 0.4
blo = link(bpy.data.objects.new('brazlight', bl)); blo.location = tuple(bt + [0, -0.3, 0.6])
for f in range(0, NF, 2):
    t = T[f]; bl.energy = float(0 if t < DACAPO - 6 else (25 if t < DACAPO else 25 + 900 * math.exp(-(t - DACAPO) / 0.5) + 120)); bl.keyframe_insert('energy', frame=f)
burst = sphere(tuple(bt + [0, -0.2, 0.3]), 0.5, M_FX); burst.color = (1.0, 0.6, 0.25, 0)
key_series(burst, 'color', 3, 0.9 * pulse_alpha([DACAPO], decay=4, length=16), 1e-3)
CANDLES.append(dict(fl=None, x=bt[0], on=[(DACAPO, 1e9)], who='lamp'))
for k in range(40):                                          # sparks out of the brazier
    sp_ = sphere((0, 0, 0), 0.035, M_FX); sp_.color = (1.0, 0.7, 0.3, 0)
    v = np.array([rng.uniform(-1.6, 1.6), rng.uniform(-1.2, 0.6), rng.uniform(2.0, 4.5)])
    P = np.zeros((NF, 3)); A = np.zeros(NF)
    for f in range(NF):
        u = T[f] - DACAPO
        P[f] = bt + (v * u + np.array([0, 0, -0.5 * G * u * u]) if u > 0 else 0)
        A[f] = 0 if u < 0 or u > 1.2 else (1 - u / 1.2)
    key_loc(sp_, P); key_series(sp_, 'color', 3, A, 1e-3)

# ---------------- finale: see-saw, the cup, the couple's cup bells, the grand chandelier ----------------
ss = EV['seesaw']; piv_ = np.array(ss['pivot']); ARM = ss['arm']; TH0 = ss['th0']; cupc = np.array(ss['cup'])
sw_ = link(bpy.data.objects.new('seesaw', None)); sw_.location = tuple(piv_)
pl = box((0, 0, -0.03), (0.5, 2 * ARM + 0.25, 0.06), M_WOOD, name='plank'); pl.parent = sw_
box((0, ARM, 0.03), (0.7, 0.7, 0.06), M_GOLD, name='farpad').parent = sw_
bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=0.35, radius2=0.03, depth=0.5, location=(piv_[0], piv_[1], piv_[2] - 0.28))
bpy.context.object.data.materials.append(M_STEEL)
rod((piv_[0], piv_[1], piv_[2] - 0.5), (piv_[0], piv_[1], floor_z_at(piv_[0])), 0.12, M_GOLDD, 12)
tl_ = ss['t_launch']; swa = np.zeros(NF)
for f in range(NF):
    u = (T[f] - tl_) / ss['t_swing']
    swa[f] = TH0 if u <= 0 else TH0 - 2 * TH0 * min(1.0, u)
sw_.rotation_mode = 'XYZ'; key_series(sw_, 'rotation_euler', 0, swa)
zplat0 = cupc[2] - RL - 0.06
plat = link(bpy.data.objects.new('platform', None)); plat.location = (cupc[0], cupc[1], zplat0)
PXs = ss['px']; XPAIR = EV['final_cups'][0]['cup'][0] * 0.5 + EV['final_cups'][1]['cup'][0] * 0.5
LPL = (XPAIR - cupc[0]) + 1.25
box((PXs - cupc[0], 0, -0.03), (LPL, 0.85, 0.06), M_GOLD, name='platdisc').parent = plat
cupo = obj(prim('bell', M_KEY, 24), 'cup'); cupo.parent = plat; cupo.location = (0, 0, 0.0); cupo.scale = (0.3, 0.3, -0.22)
ccol = pcol(LEAD[-1][1], 0.2); cupo.color = (*ccol, 0); flash(cupo, LEAD[-1][0], ccol, base=0.15)
PIECES.append(dict(kind='cup', t=LEAD[-1][0], top=cupc - [0, 0, RL], n=np.array([0, 0, 1.0]), hw=0.3, hd=0.3, th=0.05, ball='lamp'))
rise = 2 * ARM * math.sin(TH0)
pz = np.array([zplat0 + rise * min(1.0, max(0.0, (T[f] - tl_) / ss['t_swing'])) for f in range(NF)])
key_series(plat, 'location', 2, pz)
for gx in (PXs - cupc[0] - LPL / 2 - 0.08, PXs - cupc[0] + LPL / 2 + 0.08): rod((cupc[0] + gx, cupc[1], floor_z_at(cupc[0])), (cupc[0] + gx, cupc[1], zplat0 + rise + 0.3), 0.03, M_STEEL, 8)
text('シーソー', (piv_[0] + 1.0, piv_[1] - 0.6, piv_[2] - 0.3), 0.24, M_TXT_G, font=FONT)
for fc in EV['final_cups']:
    p = np.array(fc['cup']); col = pcol(DUO[-1][1][0], 0.3)
    o = obj(prim('bell', M_KEY, 24), 'duocup'); o.parent = plat; o.location = (p[0] - cupc[0], p[1] - cupc[1], p[2] - RP - zplat0); o.scale = (0.27, 0.27, -0.2)
    po_ = rod((p[0] - cupc[0], p[1] - cupc[1], 0.0), (p[0] - cupc[0], p[1] - cupc[1], p[2] - RP - zplat0 - 0.2), 0.04, M_GOLD, 8); po_.parent = plat
    o.color = (*col, 0.0); flash(o, DUO[-1][0], col, base=0.15)
    PIECES.append(dict(kind='cup', t=DUO[-1][0], top=p - [0, 0, RP], n=np.array([0, 0, 1.0]), hw=0.2, hd=0.2, th=0.05, ball=fc['ball']))
    pb_ = np.array(fc['bell']); colb = pcol(fc['pitch'], 0.2)
    sb = obj(prim('bell', M_BELL, 32), 'smallbell')
    sb.matrix_world = Matrix.Translation(Vector(pb_ + [0, 0, RP + 0.03])) @ Matrix.Diagonal((0.46, 0.46, 0.3, 1))   # inner crown = ball top (lower half of the ball shows)
    sb.color = (*colb, 0.05); flash(sb, FINAL, colb, base=0.25)
    rod(pb_ + [0, 0, RP + 0.03], pb_ + [0, 0, RP + 1.6], 0.025, M_GOLD, 8)
    PIECES.append(dict(kind='smallbell', t=FINAL, top=pb_ + [0, 0, RP], n=np.array([0, 0, -1.0]), hw=0.15, hd=0.15, th=0.05, ball=fc['ball']))
bbx = EV['bigbell']; bz0 = bbx['z']
gb = obj(prim('bell', M_BELL, 48), 'bigbell')
gb.matrix_world = Matrix.Translation(Vector((bbx['x'], bbx['y'], bz0 + 0.04))) @ Matrix.Diagonal((0.62, 0.62, 0.38, 1))   # inner crown = ball top (lower half shows)
gb.color = (1.0, 0.75, 0.35, 0)
key_series(gb, 'color', 3, np.maximum(0.06, pulse_alpha([FINAL], decay=40, length=150)), 1e-3)
PIECES.append(dict(kind='bigbell', t=FINAL, top=np.array([bbx['x'], bbx['y'], bz0]), n=np.array([0, 0, -1.0]), hw=0.35, hd=0.35, th=0.05, ball='lamp'))
N_LAMP = sum(1 for c in CANDLES if c['who'] == 'lamp'); N_SC = sum(1 for c in CANDLES if c['who'] == 'sconce'); N_DECO = sum(1 for c in CANDLES if c['who'] == 'deco')
N_GRAND = 525 - N_LAMP - N_SC - N_DECO
print('candles: lamp', N_LAMP, 'sconce', N_SC, 'deco', N_DECO, 'grand', N_GRAND)
assert 40 <= N_GRAND <= 260, N_GRAND
rings = []; left = N_GRAND; rr_ = 1.0
while left > 0:
    nb = min(left, int(2 * math.pi * rr_ / 0.16)); rings.append((rr_, nb)); left -= nb; rr_ += 0.45
gz = max([bz0] + [fc['bell'][2] + RP for fc in EV['final_cups']]) + 1.0
rod((bbx['x'], bbx['y'], bz0 + 0.04), (bbx['x'], bbx['y'], gz), 0.04, M_GOLD, 8)
rod((PXs, bbx['y'], gz), (PXs, bbx['y'], gz + 6.0), 0.06, M_GOLD, 10)
for j, (rr_, nb) in enumerate(rings):
    zr = gz - 0.25 * j
    bpy.ops.mesh.primitive_torus_add(major_radius=rr_, minor_radius=0.04, major_segments=64, minor_segments=8, location=(PXs, bbx['y'], zr))
    bpy.context.object.data.materials.append(M_GOLD)
    for i in range(nb):
        ph = 2 * math.pi * (i + 0.5 * j) / nb
        bx, by = PXs + rr_ * math.cos(ph), bbx['y'] + rr_ * math.sin(ph)
        cw, fe = candle((bx, by, zr + 0.03), h=0.1, r=0.022)
        CANDLES.append(dict(fl=fe, x=bx, on=[(FINAL + 0.08 + 0.12 * j + 0.1 * i / nb, 1e9)], who='grand'))
    for i in range(0, nb, 3):
        ph = 2 * math.pi * (i + 0.5 * j) / nb
        o = sphere((PXs + rr_ * math.cos(ph), bbx['y'] + rr_ * math.sin(ph), zr - 0.12), 0.04, M_GLASS, (8, 6))
gl_light = bpy.data.lights.new('grandlight', 'POINT'); gl_light.color = (1.0, 0.8, 0.55); gl_light.shadow_soft_size = 1.0
glo = link(bpy.data.objects.new('grandlight', gl_light)); glo.location = (PXs, -0.8, gz - 0.6)
gglow = np.array([0.0 if t < FINAL else min(1.0, (t - FINAL) / 0.6) for t in T])
for f in range(0, NF, 2):
    gl_light.energy = float(2500 * gglow[f]); gl_light.keyframe_insert('energy', frame=f)
LAMPC = [(1.0, 0.8, 0.4), (1.0, 0.55, 0.65), (1.0, 0.95, 0.8), (0.95, 0.75, 1.0)]
for k in range(120):                                          # gold confetti
    cf = obj(prim('plane', M_LAMP), 'conf'); cf.color = (*LAMPC[k % 4], 1)
    a0 = rng.uniform(0, 2 * math.pi); spd = rng.uniform(1.0, 3.4); vy = rng.uniform(-1.5, 0.5)
    P = np.zeros((NF, 3)); RT = np.zeros((NF, 3))
    for f in range(NF):
        s_ = T[f] - FINAL
        if s_ < 0: P[f] = (PXs, bbx['y'], bz0); continue
        burst = 1 - math.exp(-s_ / 0.3)
        P[f] = (PXs + spd * math.cos(a0) * 1.3 * burst + 0.15 * math.sin(3 * s_ + k), bbx['y'] + vy * burst - 0.4, bz0 + spd * math.sin(a0) * 0.8 * burst - 0.55 * s_)
        RT[f] = (s_ * (3 + k % 5), s_ * (2 + k % 3), s_ * 1.5)
    key_loc(cf, P)
    sc_ = np.where(T >= FINAL, 1.0, 0.001)
    for c_ in range(3): key_series(cf, 'scale', c_, sc_ * 0.07); key_series(cf, 'rotation_euler', c_, RT[:, c_])

# ---------------- animate every flame; the counter ----------------
for k, c in enumerate(CANDLES):
    if c['fl'] is not None: animate_flame(c['fl'], c['on'], k)
_fe = M_FLAME.node_tree.nodes['Emission']
for f in range(0, NF, 2):                                   # a gentle common flicker of all flames
    _fe.inputs['Strength'].default_value = float(22 * (0.88 + 0.07 * math.sin(2.1 * f) + 0.05 * math.sin(0.73 * f))); _fe.inputs['Strength'].keyframe_insert('default_value', frame=f)
CNT = np.zeros(NF, int)
for c in CANDLES:
    for t0, t1 in c['on']: CNT += ((T >= t0) & (T < t1)).astype(int)
print('counter max', CNT.max(), 'final', CNT[-1], 'at gust', CNT[int(GUST * FPS)], 'before da capo', CNT[int(DACAPO * FPS) - 1])
# global warmth of the hall: follows the number of lit candles
BRI = smooth(np.minimum(1.0, 0.04 + CNT / 300.0) * 1.0, 0.25)

# ---------------- lights ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes['Background']; bg.inputs['Color'].default_value = (0.03, 0.04, 0.09, 1)
for f in range(0, NF, 3):
    bg.inputs['Strength'].default_value = float(0.05 + 0.55 * BRI[f]); bg.inputs['Strength'].keyframe_insert('default_value', frame=f)
mn = bpy.data.lights.new('moon', 'SUN'); mn.color = (0.6, 0.72, 1.0); mn.angle = 0.05
mno = link(bpy.data.objects.new('moon', mn)); mno.rotation_euler = (math.radians(50), math.radians(-10), math.radians(-20))
for f in range(0, NF, 3):
    t = T[f]; mn.energy = float(0.12 + 0.25 * min(1.0, t / 8.0) + (0.45 if GUST < t < DACAPO + 1.5 else 0.0)); mn.keyframe_insert('energy', frame=f)

# ---------------- camera: follow the lamp ball, keep the gallery orchestra in the shot ----------------
FOC = np.zeros((NF, 3)); WID = np.zeros(NF)
lP = POS['lamp']; bP = POS['bass']
for f in range(NF):
    t = T[f]
    FOC[f] = np.array([lP[f, 0] + 0.5, 0, 0.55 * lP[f, 2] + 0.45 * (bP[f, 2] - ZO['bass']) + 1.45])
    WID[f] = 5.9
    if DACAPO - 0.2 < t < DACAPO + 4.5: WID[f] = 5.9 + 1.8 * math.sin(math.pi * min(1, (t - DACAPO + 0.2) / 4.7))
    if t > FINAL - 1.6:
        u = min(1, (t - (FINAL - 1.6)) / 1.0)
        FOC[f] = (1 - u) * FOC[f] + u * np.array([PXs, 0, (gz + cupc[2]) / 2 + 0.2]); WID[f] = (1 - u) * WID[f] + u * 8.2
FOC[:, 0] = smooth(FOC[:, 0], 0.5); FOC[:, 2] = smooth(FOC[:, 2], 0.8); WID = smooth(WID, 0.5)
LENS = 32.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 500
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24
    CP[f] = (FOC[f, 0], -D, FOC[f, 2] + 0.2 * D)
    TP[f] = (FOC[f, 0], 0.0, FOC[f, 2] + 0.2)
key_loc(cam, CP); key_loc(tgt, TP)
# hall lights: warm area lights riding with the camera (their power follows the lit candles)
for k, (dx, dz) in enumerate(((-3.0, 6.5), (3.0, 6.5), (0.0, 3.0))):
    al = bpy.data.lights.new(f'hall{k}', 'AREA'); al.size = 6.0; al.color = (1.0, 0.72, 0.45)
    ao = link(bpy.data.objects.new(f'hall{k}', al))
    key_loc(ao, np.stack([FOC[:, 0] + dx, np.full(NF, -3.5 if k < 2 else 1.5), FOC[:, 2] + dz], 1))
    ao.rotation_euler = (math.radians(-25 if k < 2 else 0), 0, 0) if k < 2 else (0, 0, 0)
    for f in range(0, NF, 3):
        al.energy = float((1100 if k < 2 else 600) * BRI[f] + 10); al.keyframe_insert('energy', frame=f)
# spot on the dancing couple and the lamp
for n, col in (('rose', (1.0, 0.85, 0.8)), ('lamp', (1.0, 0.85, 0.7))):
    spd = bpy.data.lights.new('spot' + n, 'SPOT'); spd.spot_size = math.radians(18); spd.spot_blend = 0.6
    spd.color = col; spd.shadow_soft_size = 0.3
    so = link(bpy.data.objects.new('spot' + n, spd)); stg = link(bpy.data.objects.new('spottgt' + n, None))
    P = POS[n].copy(); Ps = P.copy()
    for k in range(3): Ps[:, k] = smooth(P[:, k], 0.25)
    key_loc(stg, Ps - np.array([0, 0, 0.25])); key_loc(so, Ps + np.array([-2.0, -6.0, 9.0]))
    c_ = so.constraints.new('TRACK_TO'); c_.target = stg; c_.track_axis = 'TRACK_NEGATIVE_Z'; c_.up_axis = 'UP_Y'
    for f in range(0, NF, 3):
        spd.energy = float(150 + 1800 * BRI[f]); spd.keyframe_insert('energy', frame=f)

# ---------------- HUD (rides with the camera): hook, counter ----------------
HUD_D = 3.0
def hud_text(s, x, y, size, m, font=FONT, align='CENTER'):
    o = text(s, (x, y, -HUD_D), size, m, rot=(0, 0, 0), align=align, ext=0.0, font=font); o.parent = cam; return o
def vis(o, a):
    for c in range(3): key_series(o, 'scale', c, np.maximum(0.001, a), 1e-3)
M_HUDW = emis('hudw', (1.0, 0.97, 0.9), 3.0); M_HUDG = emis('hudg', (1.0, 0.75, 0.3), 4.0)
M_HUDBG = mat('hudbg', (0.0, 0.0, 0.0), 1.0, **{'Alpha': 0.55})
hook_a = np.array([1.0 if t < 3.6 else max(0.0, 1 - (t - 3.6) / 0.3) for t in T])
h1 = hud_text('真っ暗な宮殿', 0, 0.86, 0.085, M_HUDW); vis(h1, hook_a)
h2 = hud_text('ビー玉だけで全部ともせる？', 0, 0.74, 0.07, M_HUDG); vis(h2, hook_a)
hb_ = obj(prim('plane', M_HUDBG), 'hudbg'); hb_.parent = cam; hb_.location = (0, 0.8, -HUD_D - 0.01); hb_.scale = (1.3, 0.3, 1); vis(hb_, hook_a)
# counter: '灯り' + 3 digits, digit glyph objects switched by scale
cnt_a = np.array([0.0 if t < 3.6 else min(1.0, (t - 3.6) / 0.3) for t in T])
lab = hud_text('灯り', -0.3, 0.9, 0.075, M_HUDG); vis(lab, cnt_a)
cb_ = obj(prim('plane', M_HUDBG), 'cntbg'); cb_.parent = cam; cb_.location = (-0.12, 0.9, -HUD_D - 0.01); cb_.scale = (0.62, 0.16, 1); vis(cb_, cnt_a)
DIG_X = [-0.14, -0.065, 0.01]
for pos in range(3):
    dv = (CNT // (10 ** (2 - pos))) % 10
    shown = CNT >= 10 ** (2 - pos) if pos < 2 else np.ones(NF, bool)
    for dgt in range(10):
        o = hud_text(str(dgt), DIG_X[pos], 0.9, 0.105, M_HUDW)
        vis(o, ((dv == dgt) & shown).astype(float) * cnt_a)
tot = hud_text('/ 525', 0.17, 0.9, 0.06, M_HUDG); vis(tot, cnt_a * (T > FINAL + 0.9))

# ---------------- banners on the back wall (keywords for search, readable on a phone) ----------------
def banner(lines, x, w, h, zc):
    y = WALL_Y - 0.45
    box((x, y + 0.05, zc), (w, 0.05, h), M_VELVET, name='banner')
    for sgn in (1, -1):
        box((x, y, zc + sgn * h / 2), (w + 0.1, 0.08, 0.08), M_GOLD)
        box((x + sgn * w / 2, y, zc), (0.08, 0.08, h + 0.1), M_GOLD)
    rod((x - w / 2, y + 0.05, zc + h / 2 + 0.05), (x - w / 2 + 0.6, WALL_Y - 0.05, zc + h / 2 + 1.0), 0.015, M_GOLD, 6)
    rod((x + w / 2, y + 0.05, zc + h / 2 + 0.05), (x + w / 2 - 0.6, WALL_Y - 0.05, zc + h / 2 + 1.0), 0.015, M_GOLD, 6)
    tot_h = sum(sz * 1.3 for _, sz, _, _ in lines); zz = zc + tot_h / 2
    for s_, size, mt, fnt in lines:
        zz -= size * 1.3
        text(s_, (x, y - 0.03, zz + size * 0.62), size, mt, rot=(math.pi / 2, 0, 0), font=fnt)
x_ = lambda t: float(np.interp(t, T, POS['lamp'][:, 0]))
bz_ = lambda t: float(np.interp(t, T, FOC[:, 2])) + 2.25
S, J = FONT_S, FONT
for tb, w_, h_, lines in ((5.2, 5.6, 2.5, [('アイネ・クライネ', 0.56, M_TXT_G, S), ('ナハトムジーク', 0.56, M_TXT_G, S), ('モーツァルト（1787）', 0.34, M_TXT, J)]),
                          (11.0, 5.2, 1.7, [('クラシック × ビー玉からくり', 0.34, M_TXT_P, J), ('セレナーデ ト長調 K.525', 0.34, M_TXT, J)]),
                          (16.8, 4.8, 1.7, [('夜の宮殿の舞踏会', 0.44, M_TXT_G, S), ('ビー玉が燭台をともす', 0.32, M_TXT, J)]),
                          (22.6, 5.0, 1.7, [('モーツァルトの名曲', 0.4, M_TXT_G, J), ('弦楽合奏 第1楽章 アレグロ', 0.3, M_TXT, J)]),
                          (28.4, 4.8, 1.7, [('月明かりのピアノ', 0.4, M_TXT_P, S), ('マーブルマシン', 0.36, M_TXT_G, J)])):
    banner(lines, x_(tb) + 0.6, w_, h_, bz_(tb))
# the answer under the chandelier
k525 = text('モーツァルト K.525', (PXs, -1.3, cupc[2] - 0.9), 0.36, M_TXT_G, font=FONT_S)
vis(k525, np.array([0.0 if t < FINAL + 0.9 else min(1.0, (t - FINAL - 0.9) / 0.3) for t in T]))

EV_OUT = dict(pieces=[dict(kind=p['kind'], t=p['t'], top=list(map(float, p['top'])), n=list(map(float, p['n'])), hw=p['hw'], hd=p['hd'], th=p['th'], ball=p['ball']) for p in PIECES],
              counter=dict(final=int(CNT[-1]), max=int(CNT.max()), gust=int(CNT[int(GUST * FPS)]), dark=int(CNT[int(DACAPO * FPS) - 1])))
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
if MODE == 'probe':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS))
        print('t', tt_, 'FOC', np.round(FOC[f], 2), 'WID', round(WID[f], 2), 'lamp', np.round(POS['lamp'][f], 2), 'rose', np.round(POS['rose'][f], 2), 'bass', np.round(POS['bass'][f], 2), 'cnt', CNT[f])
