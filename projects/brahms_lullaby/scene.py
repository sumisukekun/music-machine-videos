"""Brahms' Lullaby - 'Sleepy marbles: how many sheep until the baby sleeps?'. Side view of a dreamy night land drawn
like a baby's room: a quilt meadow, cotton clouds, star ornaments hanging on threads, a crib with a baby mobile at the end.
Every voice note = the star marble striking the chime of a hanging star (it lights up and stays lit);
every piano right-hand note = the twin marbles on pastel toy blocks; every left-hand note = a sheep on a drum
(one sheep per bar: out of its cloud, three drums, over the fence, into its pillow, asleep).
layout.py plans all motion (constant gravity, bounces that never add speed, the see-saw); this file builds and animates it.
Fonts: en_bold.ttf = Poppins Bold (SIL OFL); jp_sub.otf = Noto Sans CJK JP Bold (SIL OFL) subset (make_font.py).
python3 scene.py -- test t1 t2 ... | range f0 f1 | probe t ... | pieces"""
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
FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.6) * FPS); T = np.arange(NF) / FPS
G = LAY['G']; EV = LAY['ev']; LN = LAY['lanes']; ZO = LAY['zoff']; V = LAY['v']
RS, RT, RH, RG = LAY['R']['star'], LAY['R']['twins'], LAY['R']['sheep'], LAY['R']['giant']
T0, Q, BAR = E['t0'], E['q'], E['bar']
T_OUT = EV['lights_out']
rng = np.random.default_rng(7)
ST_T, ST_Z = np.array(EV['stage'][0]), np.array(EV['stage'][1])
def zs(t): return float(np.interp(t, ST_T, ST_Z))
SX_T, SX_X = np.array(EV['stage_x'][0]), np.array(EV['stage_x'][1])
def xs(t): return float(np.interp(t, SX_T, SX_X))

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
try: sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Base Contrast'
except Exception: pass
sc.view_settings.exposure = 0.25
sc.frame_start, sc.frame_end = 0, NF - 1
FONT_EN = bpy.data.fonts.load(f'{HERE}/en_bold.ttf'); FONT_JP = bpy.data.fonts.load(f'{HERE}/jp_sub.otf')
TXT_K = 2.75; EN_K = 0.62


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
    """object colour = base colour; object alpha drives emission (hit flash / lit)"""
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


def objcol(name, rough=0.45, metal=0.0, emit=0.0, sheen=0.0, coat=0.0):
    m = mat(name, (1, 1, 1), rough, **{'Metallic': metal, 'Sheen Weight': sheen, 'Coat Weight': coat}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
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
        bmesh.ops.create_cube(bm, size=1.0); bmesh.ops.bevel(bm, geom=bm.edges[:] + bm.verts[:], offset=a[0] if a else 0.18, segments=3, affect='EDGES')
    elif kind == 'cyl': bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=1.0, radius2=1.0, depth=1.0)
    elif kind == 'cone': bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=a[1], radius2=a[2], depth=a[3])
    elif kind == 'sph': bmesh.ops.create_uvsphere(bm, u_segments=a[0], v_segments=a[1], radius=1.0)
    elif kind == 'plane': bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    elif kind == 'star':                                   # puffy 5-point star in the xz plane (faces -y), outer radius 1
        n5 = 5; ri = 0.45; th = a[0] if a else 0.22
        pts = []
        for j in range(2 * n5):
            r_ = 1.0 if j % 2 == 0 else ri; ph = math.pi / 2 + math.pi * j / n5
            pts.append((r_ * math.cos(ph), r_ * math.sin(ph)))
        cf = bm.verts.new((0, -th, 0)); cb = bm.verts.new((0, th, 0))
        ring_f = [bm.verts.new((x, -th * 0.35, z)) for x, z in pts]; ring_b = [bm.verts.new((x, th * 0.35, z)) for x, z in pts]
        for j in range(2 * n5):
            j2 = (j + 1) % (2 * n5)
            bm.faces.new((cf, ring_f[j2], ring_f[j])); bm.faces.new((cb, ring_b[j], ring_b[j2]))
            bm.faces.new((ring_f[j], ring_f[j2], ring_b[j2], ring_b[j]))
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
    elif kind == 'crescent':                               # crescent moon in the xz plane (opening +x), radius 1; a[1]: 0 whole, +1/-1 half
        th = a[0]; half = a[1] if len(a) > 1 else 0; N_ = 36
        a0, a1 = math.radians(47.4), math.radians(70.8)          # where the cut-out circle (r 0.78 at x 0.42) meets the moon
        def arc(cx, r, t0, t1, n): return [(cx + r * math.cos(t0 + (t1 - t0) * j / n), r * math.sin(t0 + (t1 - t0) * j / n)) for j in range(n + 1)]
        if half == 0: pts = arc(0, 1, a0, 2 * math.pi - a0, N_) + arc(0.42, 0.78, 2 * math.pi - a1, a1, N_)[1:-1]
        elif half > 0: pts = arc(0, 1, a0, math.pi, N_ // 2) + arc(0.42, 0.78, math.pi, a1, N_ // 2)[:-1]
        else: pts = arc(0, 1, math.pi, 2 * math.pi - a0, N_ // 2) + arc(0.42, 0.78, 2 * math.pi - a1, math.pi, N_ // 2)[1:]
        vf = [bm.verts.new((x, -th, z)) for x, z in pts]; vb = [bm.verts.new((x, th, z)) for x, z in pts]
        bm.faces.new(vf[::-1]); bm.faces.new(vb)
        for j in range(len(pts)):
            j2 = (j + 1) % len(pts); bm.faces.new((vf[j], vf[j2], vb[j2], vb[j]))
    me = bpy.data.meshes.new(kind); bm.to_mesh(me); bm.free()
    if kind in ('sph', 'cone', 'bell', 'cyl', 'bcube'):
        for p in me.polygons: p.use_smooth = True
    if m: me.materials.append(m)
    _MC[key] = me; return me


def obj(me, name='o'): return link(bpy.data.objects.new(name, me))


def box(c, size, m, rot=None, name='box', bev=False):
    o = obj(prim('bcube' if bev else 'cube', m), name); o.location = c; o.scale = size
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
    cu = bpy.data.curves.new('t', 'FONT'); cu.body = s; cu.extrude = ext
    if _latin(s): cu.font = font or FONT_EN; cu.size = size * TXT_K * EN_K
    else: cu.font = FONT_JP; cu.size = size * TXT_K
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
SHEEPN = sorted(n for n in BP if n.startswith('sheep')) + ['giant']


def smooth(Xa, k):
    Y = Xa.copy(); a = 1 - math.exp(-1 / FPS / k)
    for f in range(1, len(Xa)): Y[f] = Y[f - 1] + (Xa[f] - Y[f - 1]) * a
    for f in range(len(Xa) - 2, -1, -1): Y[f] = Y[f + 1] + (Y[f] - Y[f + 1]) * a
    return Y


def flash(o, t, col, n=None, base=0.0, after=None):
    fh = int(round(t * FPS))
    o.color = (*col, base); o.keyframe_insert('color', index=3, frame=max(0, fh - 1))
    for df, a in ((0, 1.0), (4, 0.6), (18, after if after is not None else max(base, 0.1))):
        o.color = (*col, a); o.keyframe_insert('color', index=3, frame=fh + df)
    if n is not None:
        b0 = o.location.copy(); nv = Vector(n)
        for df, d in ((-1, 0), (1, -0.03), (4, 0.01), (8, 0)):
            o.location = b0 + nv * d; o.keyframe_insert('location', frame=max(0, fh + df))


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


# ---------------- lighting state: warm night-light until the room light goes out, then the starry sky ----------------
OUT = np.array([0.0 if t < T_OUT else min(1.0, (t - T_OUT) / 0.5) for t in T])          # 0 = lamp on, 1 = dark
STARS_ON = np.array([0.0 if t < T_OUT + 0.15 else min(1.0, (t - T_OUT - 0.15) / 1.2) for t in T])

# ---------------- materials ----------------
M_GOLD = mat('gold', (1.0, 0.78, 0.38), 0.25, **{'Metallic': 0.9})
M_GOLDD = mat('goldd', (0.6, 0.42, 0.18), 0.35, **{'Metallic': 0.85})
M_THREAD = emis('thread', (0.95, 0.85, 0.65), 0.6)
M_STARB = glowmat('starball', gain=10.0, base=1.2, rough=0.1, coat=1.0)
M_PINK = mat('pink', (1.0, 0.45, 0.62), 0.12, **{'Coat Weight': 1.0, 'Emission Color': (1.0, 0.5, 0.65), 'Emission Strength': 0.25})
M_BLUE = mat('blue', (0.4, 0.68, 1.0), 0.12, **{'Coat Weight': 1.0, 'Emission Color': (0.45, 0.7, 1.0), 'Emission Strength': 0.25})
M_KEY = glowmat('key', gain=7.0, base=0.12, rough=0.25, coat=0.8)
M_CHIME = glowmat('chime', gain=6.0, base=0.0, rough=0.2, metal=0.8)
M_STARO = glowmat('staro', gain=9.0, base=0.0, rough=0.35, coat=0.6)
M_WOOL = mat('wool', (0.97, 0.96, 0.93), 0.85, **{'Sheen Weight': 1.0, 'Subsurface Weight': 0.15})
M_FACE = mat('face', (0.07, 0.06, 0.07), 0.5)
M_EYE = mat('eye', (1.0, 1.0, 1.0), 0.2)
M_PUPIL = mat('pupil', (0.02, 0.02, 0.03), 0.1, **{'Coat Weight': 1.0})
M_BLUSH = mat('blush', (1.0, 0.55, 0.62), 0.6)
M_CLOUD = mat('cloud', (0.95, 0.95, 1.0), 0.9, **{'Emission Color': (0.75, 0.78, 1.0), 'Emission Strength': 0.12, 'Sheen Weight': 0.6})
M_DRUM = objcol('drum', 0.35, coat=0.8)
M_SKIN = mat('drumskin', (0.98, 0.94, 0.86), 0.55)
M_WOOD = mat('wood', (0.62, 0.4, 0.22), 0.45, **{'Coat Weight': 0.4})
M_WHITEW = mat('whitewood', (0.95, 0.93, 0.88), 0.5)
M_PILLOW = objcol('pillow', 0.7, sheen=0.8)
M_OBJ = objcol('objcol', 0.5)
M_FX = fxmat('fx', 8.0)
M_TXT = emis('txtw', (1.0, 0.97, 0.9), 3.0); M_TXT_G = emis('txtg', (1.0, 0.82, 0.45), 4.0); M_TXT_P = emis('txtp', (1.0, 0.7, 0.82), 3.5)
M_TXT_B = emis('txtb', (0.7, 0.85, 1.0), 3.5)
M_INK = mat('ink', (0.25, 0.14, 0.08), 0.6)
M_SKINB = mat('babyskin', (1.0, 0.8, 0.68), 0.5, **{'Subsurface Weight': 0.3})
M_BLANKET = mat('blanket', (0.55, 0.72, 1.0), 0.8, **{'Sheen Weight': 0.8})
M_CAP = mat('cap', (1.0, 0.75, 0.82), 0.8, **{'Sheen Weight': 0.8})
M_MOON = glowmat('moon', gain=6.0, base=0.6, rough=0.4, coat=0.5)
# quilt meadow: pastel patchwork (checker of two pastel greens, mapped in world coords) with a soft stitched grid
M_QUILT = bpy.data.materials.new('quilt'); M_QUILT.use_nodes = True; _nt = M_QUILT.node_tree; _b = _nt.nodes['Principled BSDF']
_tc = _nt.nodes.new('ShaderNodeTexCoord'); _ck = _nt.nodes.new('ShaderNodeTexChecker'); _ck.inputs['Scale'].default_value = 1.0
_ck.inputs['Color1'].default_value = (0.5, 0.72, 0.58, 1); _ck.inputs['Color2'].default_value = (0.86, 0.8, 0.62, 1)
_mp = _nt.nodes.new('ShaderNodeMapping'); _mp.inputs['Scale'].default_value = (1.6, 1.6, 1.6)
_nt.links.new(_tc.outputs['Object'], _mp.inputs['Vector']); _nt.links.new(_mp.outputs[0], _ck.inputs['Vector'])
_nt.links.new(_ck.outputs['Color'], _b.inputs['Base Color']); _b.inputs['Roughness'].default_value = 0.85; _b.inputs['Sheen Weight'].default_value = 0.7

PITCH_COL = [(1.0, 0.45, 0.45), (1.0, 0.6, 0.4), (1.0, 0.78, 0.4), (1.0, 0.9, 0.45), (0.75, 1.0, 0.5), (0.5, 1.0, 0.6),
             (0.45, 1.0, 0.85), (0.45, 0.85, 1.0), (0.5, 0.62, 1.0), (0.7, 0.55, 1.0), (0.88, 0.55, 1.0), (1.0, 0.55, 0.85)]
def pcol(p, light=0.0):
    c = PITCH_COL[p % 12]; return tuple(min(1, x * (1 - light) + light) for x in c)


# ---------------- balls: star marble, twins ----------------
BALL = {}
for n, mm in (('star', M_STARB), ('pink', M_PINK), ('blue', M_BLUE)):
    rr = BP[n].r
    bpy.ops.mesh.primitive_uv_sphere_add(radius=rr, segments=32, ring_count=16); o = bpy.context.object; o.name = n
    for p in o.data.polygons: p.use_smooth = True
    o.data.materials.append(mm)
    key_loc(o, POS[n])
    sp = np.zeros(NF)
    for f in range(1, NF): sp[f] = sp[f - 1] + (POS[n][f, 0] - POS[n][f - 1, 0]) / rr
    o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, sp)
    BALL[n] = o
    if n == 'star':                                       # a little gold star inside the glowing marble
        st = obj(prim('star', M_GOLD, 0.3), 'starcore'); st.parent = o; st.scale = (rr * 0.62, rr * 0.62, rr * 0.62)
BALL['star'].color = (1.0, 0.72, 0.28, 0.35)
sl = bpy.data.lights.new('starlight', 'POINT'); sl.color = (1.0, 0.85, 0.55); sl.shadow_soft_size = 0.2; sl.energy = 30
slo = link(bpy.data.objects.new('starlight', sl)); slo.parent = BALL['star']

# ---------------- sheep ----------------
SHEEP_OBJ = {}
FACE_DIR = np.array([1.0, -0.45, 0.0]); FACE_DIR /= np.linalg.norm(FACE_DIR)     # they look right and a little at the camera
SLEEP_T = {s['name']: s['t_sleep'] for s in EV['sheep']}; SLEEP_T['giant'] = EV['seesaw']['t_launch'] + 0.4


def make_sheep(name, r):
    root = link(bpy.data.objects.new(name, None))
    core = sphere((0, 0, 0), r * 0.8, M_WOOL, (20, 10), 'core'); core.parent = root
    golden = math.pi * (3 - math.sqrt(5)); npf = 26
    for j in range(npf):
        zz = 1 - 2 * (j + 0.5) / npf; rr_ = math.sqrt(1 - zz * zz); ph = golden * j
        d = np.array([rr_ * math.cos(ph), rr_ * math.sin(ph), zz])
        if d @ FACE_DIR > 0.55: continue                      # leave room for the face
        pf = sphere(tuple(d * r * 0.68), r * 0.32, M_WOOL, (12, 6), 'puff'); pf.parent = root
    fc = FACE_DIR * r * 0.78 + np.array([0, 0, -0.08 * r])
    head = sphere(tuple(fc), r * 0.42, M_FACE, (16, 8), 'head'); head.parent = root; head.scale = (r * 0.4, r * 0.36, r * 0.46)
    side = np.cross([0, 0, 1], FACE_DIR); side /= np.linalg.norm(side)
    eyes = []
    for s_ in (-1, 1):
        ep = fc + FACE_DIR * r * 0.3 + side * s_ * r * 0.17 + np.array([0, 0, 0.08 * r])
        eye = link(bpy.data.objects.new('eyeroot', None)); eye.parent = root; eye.location = tuple(ep)
        ew = sphere((0, 0, 0), 1, M_EYE, (12, 6), 'eyew'); ew.parent = eye; ew.scale = (r * 0.11, r * 0.11, r * 0.11)
        pu = sphere(tuple(FACE_DIR * r * 0.07), 1, M_PUPIL, (10, 5), 'pupil'); pu.parent = eye; pu.scale = (r * 0.065, r * 0.065, r * 0.065)
        eyes.append(eye)
        ear = sphere(tuple(fc + side * s_ * r * 0.42 + np.array([0, 0, 0.12 * r]) - FACE_DIR * 0.1 * r), 1, M_FACE, (10, 5), 'ear')
        ear.parent = root; ear.scale = (r * 0.09, r * 0.2, r * 0.07); ear.rotation_euler = (0, 0, math.atan2(side[1], side[0]))
        bl = sphere(tuple(fc + FACE_DIR * r * 0.28 + side * s_ * r * 0.27 - np.array([0, 0, 0.1 * r])), 1, M_BLUSH, (8, 4), 'blush')
        bl.parent = root; bl.scale = (r * 0.06, r * 0.06, r * 0.04)
    return root, eyes


for n in SHEEPN:
    r = RG if n == 'giant' else RH
    root, eyes = make_sheep(n, r)
    key_loc(root, POS[n])
    ts_ = SLEEP_T[n]
    # blink now and then while awake, eyes shut (sleeping) after landing in the pillow
    ez = np.ones(NF)
    for f in range(NF):
        t = T[f]
        if t > ts_: ez[f] = max(0.12, 1 - (t - ts_) / 0.35)
        elif (t * 1.3 + hash(n) % 7) % 3.2 < 0.12: ez[f] = 0.15
    for e_ in eyes: key_series(e_, 'scale', 2, ez, 1e-3)
    # sleeping: a slow breathing
    br = np.array([1.0 if T[f] < ts_ + 0.4 else 1 + 0.035 * math.sin(2 * math.pi * (T[f] - ts_) / 2.6) for f in range(NF)])
    key_series(root, 'scale', 2, br, 1e-4)
    SHEEP_OBJ[n] = root

# ---------------- stage: quilt meadow (front lane), always stepping down ----------------
pts = []
for s in EV['sheep']:
    d1 = np.array(s['drums'][0]); p4 = np.array(s['pillow'])
    pts += [(d1[0] - 0.6, d1[2] - 1.0), (d1[0] + 1.0, s['drums'][1][2] - 1.0), (p4[0], p4[2] - RH - 0.22)]
gd = np.array(EV['giant']['drum']); far = np.array(EV['seesaw']['far']); piv = np.array(EV['seesaw']['pivot'])
pts += [(gd[0], gd[2] - 1.1), (far[0], piv[2] - 1.3)]
pts.sort()
GX = np.array([p[0] for p in pts]); GZv = np.array([p[1] for p in pts])
FLX = np.arange(-14.0, GX.max() + 30, 0.25)
FLZ = np.array([GZv[np.abs(GX - x) < 1.2].min() if (np.abs(GX - x) < 1.2).any() else np.nan for x in FLX])
ok_ = ~np.isnan(FLZ); FLZ = np.interp(FLX, FLX[ok_], FLZ[ok_])
FLZ = np.minimum.accumulate(FLZ)
for _ in range(3): FLZ = np.minimum(FLZ, np.convolve(np.pad(FLZ, 6, mode='edge'), np.ones(13) / 13, 'valid'))
def ground_z(x): return float(np.interp(x, FLX, FLZ))


def strip_mesh(X, Z, y0, y1, th, m, name, Z1=None):
    bm = bmesh.new(); vs = []
    Z1 = Z if Z1 is None else Z1
    for x, z, z1 in zip(X, Z, Z1):
        vs.append([bm.verts.new((x, y0, z)), bm.verts.new((x, y1, z1)), bm.verts.new((x, y1, z1 - th)), bm.verts.new((x, y0, z - th))])
    for a, b in zip(vs, vs[1:]):
        for k in range(4): bm.faces.new((a[k], b[k], b[(k + 1) % 4], a[(k + 1) % 4]))
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new(name, me)); o.data.materials.append(m)
    for p in o.data.polygons: p.use_smooth = True
    return o


DROP_F = 1.6                                             # the quilt hangs over a step in front of the sheep lane
strip_mesh(FLX, FLZ, -1.0, 1.0, DROP_F, M_QUILT, 'meadow')
strip_mesh(FLX, FLZ - DROP_F, -7.0, -1.0, 0.4, M_QUILT, 'meadowfront')
strip_mesh(FLX, FLZ, 1.0, 7.0, 0.5, M_QUILT, 'meadowback', Z1=FLZ + 0.6)

# ---------------- star chimes: a gold chime bar in front of a hanging puffy star ----------------
PIECES = []
SWAY = 0.15
def sway_dx(t, th):
    u = np.maximum(0.0, np.asarray(t) - th)
    return np.where(np.asarray(t) > th, -SWAY * (1 - np.exp(-u / 0.25)) + 0.03 * np.exp(-u / 1.2) * np.sin(5.0 * u), 0.0)
STARS = []                                            # (ornament centre, t_hit)
for h in BP['star'].hits:
    if h['kind'] != 'chime': continue
    nv = np.array(h['n']); top = np.array(h['p']) - RS * nv; p = h['pitch']
    col = pcol(p, 0.35)
    bar = obj(prim('bcube', M_CHIME, 0.3), 'chime')
    bar.matrix_world = Matrix.Translation(Vector(top - nv * 0.03)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((0.4, 0.4, 0.06, 1))
    bar.color = (1.0, 0.85, 0.5, 0.0); flash(bar, h['t'], (1.0, 0.85, 0.5), nv)
    ssz = 0.26 + 0.012 * (p - 63)
    oc = np.array([top[0], LN['star'] + 0.48, top[2] - 0.12])
    so = obj(prim('star', M_STARO, 0.25), 'staro'); so.location = tuple(oc); so.scale = (ssz, ssz, ssz)
    so.color = (*col, 0.0); flash(so, h['t'], col, base=0.0, after=0.32)
    br_ = rod(top - nv * 0.05 + np.array([0, 0.18, 0]), oc + np.array([0, -0.1, 0.05]), 0.012, M_GOLD, 6)
    th_ = rod(oc + np.array([0, 0, ssz * 0.95]), oc + np.array([0, 0, 14.0]), 0.006, M_THREAD, 4)
    STARS.append((oc, h['t'], so, col, ssz))
    PIECES.append(dict(kind='chime', t=h['t'], top=top, n=nv, hw=0.2, hd=0.2, th=0.03, ball='star', rs=(1.2, 1e6, 0.3, 0.35), sway=SWAY))
    e_ = riser([bar, so, br_, th_], h['t'], top, lead=1.2, stay=1e6)
    key_series(e_, 'location', 0, top[0] + sway_dx(T, h['t']), 1e-5)            # the struck ornament is nudged back and sways

# ---------------- twin keys: pastel toy blocks that pop up and vanish, a cloud road under them ----------------
for n in ('pink', 'blue'):
    for h in BP[n].hits:
        if h['kind'] not in ('key', 'uni'): continue
        nv = np.array(h['n']); top = np.array(h['p']) - RT * nv
        if h['kind'] == 'uni' and n == 'blue': continue                     # one wide block for both twins
        o = obj(prim('bcube', M_KEY, 0.12), 'block')
        w = (0.84, 0.72, 0.16) if h['kind'] == 'uni' else (0.4, 0.4, 0.16)
        cc = top - nv * w[2] / 2
        if h['kind'] == 'uni':
            other = [x for x in BP['blue'].hits if abs(x['t'] - h['t']) < 1e-6][0]
            cc = 0.5 * (cc + np.array(other['p']) - RT * np.array(other['n']) - nv * w[2] / 2)
            top = cc + nv * w[2] / 2
        o.matrix_world = Matrix.Translation(Vector(cc)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((*w, 1))
        col = pcol(h['pitch'], 0.3); o.color = (*col, 0); flash(o, h['t'], col, nv)
        po = rod((cc[0], cc[1], cc[2] - 0.08), (cc[0], cc[1], cc[2] - 0.75), 0.025, M_WHITEW, 8)
        QUICK = h['t'] > E['grid'][-2][2] - 0.2
        rp = dict(lead=0.15, stay=0.05, pop=0.1, shrink=0.15) if QUICK else dict(lead=0.35, stay=0.9, pop=0.15, shrink=0.35)
        PIECES.append(dict(kind=h['kind'], t=h['t'], top=top, n=nv, hw=w[0] / 2, hd=w[1] / 2, th=w[2] / 2, ball='pair' if h['kind'] == 'uni' else n,
                           rs=(rp['lead'], rp['stay'], rp['pop'], rp['shrink'])))
        riser([o, po], h['t'], top, **rp)


# ---------------- moon cradles: on a long note the ball drops into a crescent-moon cradle (it rings), rests, the moon opens ----------------
M_CRADLE = glowmat('cradle', gain=6.0, base=0.15, rough=0.3, coat=0.8)
for cr in EV['cradles']:
    pc = np.array(cr['p']); twins = cr['who'] == 'twins'
    r_ = RT if twins else RS
    S_ = 0.5
    sx, sy = (1.55, 0.9) if twins else (1.0, 0.55)
    # after the -90 deg turn about y the opening faces up; the inner circle's bottom (0.36 S below the moon centre) = ball bottom
    ctr = pc + np.array([0, 0, -r_ + 0.36 * S_])
    col = (1.0, 0.86, 0.5) if not twins else (0.85, 0.8, 1.0)
    top = pc - np.array([0, 0, r_])
    stay = cr['t_rel'] - cr['t'] + 0.1                     # the moon opens, then fades away
    LEADC, POPC = (0.15, 0.1) if twins else (1.0, 0.3)
    pe = link(bpy.data.objects.new('cradlepop', None)); pe.location = tuple(top)
    for half, sgn in ((1, -1), (-1, 1)):                  # prim z>0 half -> x<0 side after the turn
        horn = ctr + np.array([sgn * 0.737 * S_ * sx, 0, 0.676 * S_])
        hg = link(bpy.data.objects.new('cradlehinge', None)); hg.location = tuple(horn)
        bo = obj(prim('crescent', M_CRADLE, 0.5, half), 'cradle'); bo.parent = hg
        bo.location = tuple(ctr - horn); bo.rotation_euler = (0, -math.pi / 2, 0); bo.scale = (S_, sy, S_ * sx)
        bo.color = (*col, 0.15); flash(bo, cr['t'], col, base=0.15, after=0.25)
        ang = np.zeros(NF)
        for f in range(NF):
            u = (T[f] - (cr['t_rel'] - 0.15)) / 0.4
            ang[f] = 0.0 if u <= 0 else sgn * 1.25 * min(1.0, u) ** 0.8 * (1.0 if T[f] < cr['t_rel'] + 0.8 else max(0.0, 1 - (T[f] - cr['t_rel'] - 0.8) / 0.8))
        hg.rotation_mode = 'XYZ'; key_series(hg, 'rotation_euler', 1, -ang, 1e-4)
        th_ = rod(horn, horn + np.array([0, 0, 12.0]), 0.006, M_THREAD, 4)
        for o_ in (hg, th_): o_.parent = pe; o_.matrix_parent_inverse = Matrix.Translation(-Vector(top))
    PIECES.append(dict(kind='cradle', t=cr['t'], top=top, n=np.array([0, 0, 1.0]), hw=(0.42 if twins else 0.22), hd=(0.38 if twins else 0.2), th=0.04,
                       ball='pair' if twins else 'star', own=(cr['t'] - 0.03, cr['t_rel'] + 0.46), rs=(LEADC, stay, POPC, 0.35)))
    s_ = np.zeros(NF)
    for f in range(NF):
        t = T[f]; u1 = (t - (cr['t'] - LEADC)) / POPC; u2 = (t - (cr['t'] + stay)) / 0.35
        a = 0.0 if u1 <= 0 else min(1.0, u1)
        s_[f] = max(0.001, a * (1.0 - min(1, max(0, u2)) ** 1.5))
    for k in range(3): key_series(pe, 'scale', k, s_, 1e-4)


def cloud_road(names, y, depth, below):
    hs = [h for n in names for h in BP[n].hits if h['kind'] in ('key', 'uni')]
    xs_ = np.array([h['p'][0] for h in hs]); zz = np.array([h['p'][2] for h in hs])
    X = np.arange(xs_.min() - 1.0, xs_.max() + 0.6, 0.4)
    Z = np.array([zz[np.abs(xs_ - x) < 1.4].min() if (np.abs(xs_ - x) < 1.4).any() else np.nan for x in X])
    ok = ~np.isnan(Z); Z = np.interp(X, X[ok], Z[ok]) - below
    for _ in range(3): Z = np.minimum(Z, np.convolve(np.pad(Z, 3, mode='edge'), np.ones(7) / 7, 'valid'))
    strip_mesh(X, Z, y - depth, y + depth, 0.16, M_CLOUD, 'cloudroad')
    for x, z in list(zip(X, Z))[::2]:                      # cotton puffs along its edges
        for yy in (y - depth, y + depth):
            sphere((x + rng.uniform(-0.1, 0.1), yy, z - 0.06), rng.uniform(0.13, 0.2), M_CLOUD, (12, 6), 'puff')
    return X, Z
cloud_road(['pink', 'blue'], LN['twins'], 0.36, 0.9)

# ---------------- sheep: clouds, drums, fences, pillows ----------------
def cloud_puff(c, w, m=M_CLOUD, seed=0, scale=1.0):
    r_ = np.random.default_rng(seed); parts = []
    for j in range(7):
        dx = (j - 3) / 3 * w * 0.5; rr = (0.22 + 0.12 * (1 - abs(j - 3) / 3)) * scale
        parts.append(sphere((c[0] + dx, c[1] + r_.uniform(-0.1, 0.1), c[2] + r_.uniform(-0.03, 0.06) * scale), rr, m, (14, 7), 'cloudp'))
    return parts


def drum(top, n, p, t, size=1.0):
    """drum: pastel body, cream skin; its skin top = contact top, normal n"""
    rr = (0.3 + 0.012 * max(0, 50 - p)) * size; hh = 0.3 * size
    col = pcol(p, 0.25)
    M = Matrix.Translation(Vector(top)) @ frame_from_n(n).to_4x4()
    skin = obj(prim('cyl', M_SKIN, 32), 'skin'); skin.matrix_world = M @ Matrix.Translation((0, 0, -0.01)) @ Matrix.Diagonal((rr, rr, 0.02, 1))
    body = obj(prim('cyl', M_DRUM, 32), 'drumbody'); body.matrix_world = M @ Matrix.Translation((0, 0, -0.02 - hh / 2)) @ Matrix.Diagonal((rr * 1.02, rr * 1.02, hh, 1))
    body.color = (*col, 1)
    rim = obj(prim('cyl', M_GOLD, 32), 'rim'); rim.matrix_world = M @ Matrix.Translation((0, 0, -0.03)) @ Matrix.Diagonal((rr * 1.06, rr * 1.06, 0.035, 1))
    glow = obj(prim('cyl', M_FX, 32), 'drumglow'); glow.matrix_world = M @ Matrix.Translation((0, 0, 0.005)) @ Matrix.Diagonal((rr * 0.9, rr * 0.9, 0.01, 1))
    glow.color = (*pcol(p, 0.5), 0); key_series(glow, 'color', 3, 0.5 * pulse_alpha([t], decay=5, length=20), 1e-3)
    bot = np.array(top) - np.array(n) * (hh + 0.02)
    rod(bot, (bot[0], bot[1], ground_z(bot[0]) - 0.05), 0.05 * size, M_WOOD, 8)
    return rr


for s in EV['sheep']:
    hs = [h for h in BP[s['name']].hits if h['kind'] == 'drum']
    for h in hs:
        nv = np.array(h['n']); top = np.array(h['p']) - RH * nv
        rr = drum(top, nv, h['pitch'], h['t'])
        PIECES.append(dict(kind='drum', t=h['t'], top=top, n=nv, hw=rr * 0.72, hd=rr * 0.72, th=0.03, ball=s['name']))
    # the cloud it waits on; it parts when the sheep drops
    c = np.array(s['cloud']) - [0, 0, RH + 0.12]
    parts = cloud_puff(c, 0.9, seed=hash(s['name']) % 1000)
    for j, pp in enumerate(parts):
        side = -1 if j < 3 else (1 if j > 3 else 0)
        P = np.zeros((NF, 3)); base = np.array(pp.location)
        for f in range(NF):
            u = (T[f] - s['t_release']) / 0.25
            P[f] = base + ([side * 0.5 * min(1, max(0, u)) ** 0.7, 0, 0] if side else [0, 0, 0])
        key_loc(pp, P)
        if side == 0: key_series(pp, 'scale', 2, np.where(T > s['t_release'], np.maximum(0.001, pp.scale[2] * (1 - np.clip((T - s['t_release']) / 0.15, 0, 1))), pp.scale[2]), 1e-4)
    # fence (rails along x, under the top of the jump)
    fz = s['fence']['z_top']; fx = s['fence']['x']; gz = ground_z(fx)
    for dx in (-0.1, 0.0, 0.1):
        box((fx + dx, 0.0, (fz + gz) / 2), (0.06, 0.06, fz - gz), M_WHITEW)
        o = obj(prim('cone', M_WHITEW, 4, 0.045, 0.0, 0.07), 'tip'); o.location = (fx + dx, 0.0, fz + 0.03); o.rotation_euler = (0, 0, math.pi / 4)
    for zz in (gz + 0.25 * (fz - gz), gz + 0.72 * (fz - gz)): box((fx, -0.04, zz), (0.3, 0.04, 0.06), M_WHITEW)
    PIECES.append(dict(kind='fence', t=-1, top=np.array([fx, 0.0, fz + 0.07]), n=np.array([0, 0, 1.0]), hw=0.15, hd=0.05, th=(fz - gz) / 2 + 0.04, ball='none'))
    # pillow (squashes when the sheep lands)
    p4 = np.array(s['pillow']); pt = p4[2] - RH
    pe = link(bpy.data.objects.new('pillowe', None)); pe.location = (p4[0], p4[1], pt)
    pw = obj(prim('bcube', M_PILLOW, 0.3), 'pillow'); pw.parent = pe; pw.location = (0, 0, -0.12); pw.scale = (0.82, 0.7, 0.24)
    pw.color = (*pcol(int(rng.integers(0, 12)), 0.55), 1)
    sq = np.array([1.0 if T[f] < s['t_sleep'] else 1 - 0.28 * math.exp(-(T[f] - s['t_sleep']) / 0.12) * math.cos(14 * (T[f] - s['t_sleep'])) for f in range(NF)])
    key_series(pe, 'scale', 2, sq, 1e-4)
    rod((p4[0], p4[1], pt - 0.24), (p4[0], p4[1], ground_z(p4[0]) - 0.05), 0.08, M_WHITEW, 8)
    PIECES.append(dict(kind='pillow', t=s['t_sleep'], top=np.array([p4[0], p4[1], pt]), n=np.array([0, 0, 1.0]), hw=0.41, hd=0.35, th=0.12, ball=s['name'], own=(s['t_sleep'] - 0.03, 1e9)))

# ---------------- 'Zzz' over the sleeping sheep ----------------
for n in SHEEPN:
    ts_ = SLEEP_T[n]; base = POS[n][-1] + np.array([0.1, -0.2, (RG if n == 'giant' else RH) + 0.15])
    for j in range(3):
        z_ = text('z', (0, 0, 0), 0.16 + 0.05 * j, M_TXT_B, rot=(math.pi / 2, 0, 0.2), ext=0.0)
        P = np.zeros((NF, 3)); A = np.zeros(NF)
        for f in range(NF):
            u = ((T[f] - ts_ - 0.5 - 0.7 * j) % 2.4) / 2.4
            P[f] = base + np.array([0.25 * u + 0.08 * math.sin(6 * u), 0, 0.7 * u])
            A[f] = 0.0 if T[f] < ts_ + 0.5 + 0.7 * j else math.sin(math.pi * u)
        key_loc(z_, P)
        for c_ in range(3): key_series(z_, 'scale', c_, np.maximum(0.001, A), 1e-3)

# ---------------- the giant sheep's cloud and drum, the see-saw, platform, cups, the baby mobile, the crib ----------------
gs = EV['giant']; gdrum = [h for h in BP['giant'].hits if h['kind'] == 'drum'][0]
nv = np.array(gdrum['n']); top = np.array(gdrum['p']) - RG * nv
rr = drum(top, nv, gdrum['pitch'], gdrum['t'], size=1.9)
PIECES.append(dict(kind='drum', t=gdrum['t'], top=top, n=nv, hw=rr * 0.72, hd=rr * 0.72, th=0.03, ball='giant'))
c = np.array(gs['cloud']) - [0, 0, RG + 0.2]
parts = cloud_puff(c, 1.9, seed=99, scale=2.0)
for j, pp in enumerate(parts):
    side = -1 if j < 3 else (1 if j > 3 else 0)
    P = np.zeros((NF, 3)); base = np.array(pp.location)
    for f in range(NF):
        u = (T[f] - gs['t_release']) / 0.25
        P[f] = base + ([side * 1.0 * min(1, max(0, u)) ** 0.7, 0, 0] if side else [0, 0, 0])
    key_loc(pp, P)
    if side == 0: key_series(pp, 'scale', 2, np.where(T > gs['t_release'], np.maximum(0.001, pp.scale[2] * (1 - np.clip((T - gs['t_release']) / 0.15, 0, 1))), pp.scale[2]), 1e-4)
ss = EV['seesaw']; piv_ = np.array(ss['pivot']); ARM = ss['arm']; TH0 = ss['th0']; cupc = np.array(ss['cup']); PXs = ss['px']
sw_ = link(bpy.data.objects.new('seesaw', None)); sw_.location = tuple(piv_)
pl = box((0, 0, -0.03), (0.55, 2 * ARM + 0.3, 0.06), M_WOOD, name='plank'); pl.parent = sw_
fp = box((0, -ARM, 0.03), (0.9, 0.9, 0.06), M_GOLD, name='farpad'); fp.parent = sw_
CHB = []
for j, (p_, dx) in enumerate(zip(gs['chord'], (-0.32, 0.0, 0.32))):         # the LH chord bells on the far end
    b_ = obj(prim('bell', M_CHIME, 24), 'chordbell'); b_.parent = sw_; b_.location = (dx * 1.2, -ARM - 0.5, 0.42 - 0.04 * j); b_.scale = (0.17, 0.17, 0.22)
    b_.color = (1.0, 0.8, 0.45, 0); flash(b_, ss['t_launch'], (1.0, 0.8, 0.45)); CHB.append(b_)
    rr2 = rod((dx * 1.2, -ARM - 0.5, 0.42 - 0.04 * j), (dx * 1.2, -ARM - 0.5, 0.0), 0.02, M_GOLD, 6); rr2.parent = sw_
    hm = rod((dx * 1.2, -ARM - 0.05, 0.08), (dx * 1.2, -ARM - 0.45, 0.32), 0.018, M_GOLD, 6); hm.parent = sw_
cone = obj(prim('cone', M_WOOD, 4, 0.38, 0.03, 0.55), 'fulcrum'); cone.location = (piv_[0], piv_[1], piv_[2] - 0.3)
rod((piv_[0], piv_[1], piv_[2] - 0.55), (piv_[0], piv_[1], ground_z(piv_[0]) - 0.1), 0.12, M_WOOD, 12)
tl_ = ss['t_launch']; swa = np.zeros(NF)
for f in range(NF):
    u = (T[f] - tl_) / ss['t_swing']
    swa[f] = -TH0 if u <= 0 else -TH0 + 2 * TH0 * min(1.0, u)
sw_.rotation_mode = 'XYZ'; key_series(sw_, 'rotation_euler', 0, swa)
zplat0 = cupc[2] - RS - 0.06
plat = link(bpy.data.objects.new('platform', None)); plat.location = (cupc[0], cupc[1], zplat0)
XPAIR = 0.5 * (EV['final_cups'][0]['cup'][0] + EV['final_cups'][1]['cup'][0])
LPL = abs(XPAIR - cupc[0]) + 1.3
box((PXs - cupc[0], 0, -0.03), (LPL, 0.95, 0.06), M_GOLD, name='platdisc').parent = plat
cupo = obj(prim('bell', M_KEY, 24), 'cup'); cupo.parent = plat; cupo.location = (0, 0, 0.0); cupo.scale = (0.32, 0.32, -0.22)
ccol = pcol(LEAD[-1][1], 0.3); cupo.color = (*ccol, 0); flash(cupo, LEAD[-1][0], ccol, base=0.15)
PIECES.append(dict(kind='cup', t=LEAD[-1][0], top=cupc - [0, 0, RS], n=np.array([0, 0, 1.0]), hw=0.3, hd=0.3, th=0.05, ball='star', own=(LEAD[-1][0] - 0.03, 1e9), life=(0, tl_)))
rise = ss['rise']
pz = np.array([zplat0 + rise * min(1.0, max(0.0, (T[f] - tl_) / ss['t_swing'])) for f in range(NF)])
key_series(plat, 'location', 2, pz)
for gx in (PXs - cupc[0] - LPL / 2 - 0.08, PXs - cupc[0] + LPL / 2 + 0.08):
    rod((cupc[0] + gx, cupc[1], ground_z(cupc[0]) - 0.2), (cupc[0] + gx, cupc[1], zplat0 + rise + 0.3), 0.03, M_GOLDD, 8)
for fc in EV['final_cups']:
    p = np.array(fc['cup']); col = pcol(DUO[-1][1][0], 0.3)
    o = obj(prim('bell', M_KEY, 24), 'twincup'); o.parent = plat; o.location = (p[0] - cupc[0], p[1] - cupc[1], p[2] - RT - zplat0); o.scale = (0.27, 0.27, -0.2)
    po_ = rod((p[0] - cupc[0], p[1] - cupc[1], 0.0), (p[0] - cupc[0], p[1] - cupc[1], p[2] - RT - zplat0 - 0.2), 0.04, M_GOLD, 8); po_.parent = plat
    o.color = (*col, 0.0); flash(o, DUO[-1][0], col, base=0.15)
    PIECES.append(dict(kind='cup', t=DUO[-1][0], top=p - [0, 0, RT], n=np.array([0, 0, 1.0]), hw=0.2, hd=0.2, th=0.05, ball=fc['ball'], own=(DUO[-1][0] - 0.03, 1e9), life=(0, tl_)))
# the baby mobile: a cross arm on a long rod; moon bell over the star's cup, two star bells over the twins' cups; it turns after the chord
mb = EV['moonbell']; mz = mb['z']
MOB = link(bpy.data.objects.new('mobile', None)); MOB.location = (PXs, LN['star'], mz + 1.0)
rod((PXs, LN['star'], mz + 1.0), (PXs, LN['star'], mz + 16.0), 0.035, M_GOLD, 8)
MOBP = []
moonb = obj(prim('bell', M_CHIME, 40), 'moonbell'); moonb.location = (mb['x'], mb['y'], mz + 0.03); moonb.scale = (0.55, 0.55, 0.36); MOBP.append(moonb)
moonb.color = (1.0, 0.85, 0.5, 0); key_series(moonb, 'color', 3, np.maximum(0.05, pulse_alpha([FINAL], decay=30, length=120)), 1e-3)
PIECES.append(dict(kind='moonbell', t=FINAL, top=np.array([mb['x'], mb['y'], mz]), n=np.array([0, 0, -1.0]), hw=0.3, hd=0.3, th=0.05, ball='star', life=(0, FINAL + 0.9)))
MOX, MOZ = mb['x'] - 0.15, mz + 0.62                       # the sleepy crescent moon above its bell (opening to the right, face on the left)
moon = obj(prim('crescent', M_MOON, 0.12, 0), 'moonface'); moon.location = (MOX, mb['y'] - 0.05, MOZ); moon.scale = (0.5, 0.5, 0.5); MOBP.append(moon)
moon.color = (1.0, 0.88, 0.5, 0.3); key_series(moon, 'color', 3, 0.3 + 0.7 * np.clip((T - FINAL) / 0.3, 0, 1), 1e-3)
eye = sphere((MOX - 0.32, mb['y'] - 0.13, MOZ + 0.08), 0.04, M_PUPIL, (10, 5), 'mooneye'); MOBP.append(eye)
key_series(eye, 'scale', 2, np.where(T < FINAL + 0.15, 0.04, 0.007), 1e-4)
MOBP.append(sphere((MOX - 0.38, mb['y'] - 0.12, MOZ - 0.08), 0.035, M_BLUSH, (8, 4), 'moonblush'))
MOBP.append(rod((mb['x'], mb['y'], mz + 0.03), (mb['x'], mb['y'], mz + 1.0), 0.012, M_GOLD, 6))
for fc in EV['final_cups']:
    pb_ = np.array(fc['bell']); colb = pcol(fc['pitch'], 0.3)
    sb = obj(prim('bell', M_CHIME, 32), 'starbell'); sb.location = tuple(pb_ + [0, 0, RT + 0.03]); sb.scale = (0.4, 0.4, 0.27); MOBP.append(sb)
    sb.color = (*colb, 0.05); flash(sb, FINAL, colb, base=0.25)
    so = obj(prim('star', M_STARO, 0.25), 'mobstar'); so.location = tuple(pb_ + [0, -0.05, RT + 0.6]); so.scale = (0.24, 0.24, 0.24); MOBP.append(so)
    so.color = (*colb, 0.3); flash(so, FINAL, colb, base=0.3, after=0.6)
    MOBP.append(rod(pb_ + [0, 0, RT + 0.03], pb_ + [0, 0, RT + 0.36], 0.01, M_GOLD, 6))
    MOBP.append(rod(pb_ + [0, 0, RT + 0.84], (pb_[0], pb_[1], mz + 1.0), 0.01, M_GOLD, 6))
    PIECES.append(dict(kind='starbell', t=FINAL, top=pb_ + [0, 0, RT], n=np.array([0, 0, -1.0]), hw=0.15, hd=0.15, th=0.05, ball=fc['ball'], life=(0, FINAL + 0.9)))
xa_ = [fc['bell'][0] for fc in EV['final_cups']] + [mb['x']]
MOBP.append(rod((min(xa_) - 0.05, LN['star'], mz + 1.0), (max(xa_) + 0.05, LN['star'], mz + 1.0), 0.03, M_GOLD, 8))
for o_ in MOBP: o_.parent = MOB; o_.matrix_parent_inverse = Matrix.Translation(-Vector(MOB.location))
MOB.rotation_mode = 'XYZ'
key_series(MOB, 'rotation_euler', 2, np.array([0.0 if t < FINAL + 0.9 else 0.35 * (t - FINAL - 0.9) ** 1.5 for t in T]), 1e-4)
# crib with the baby, beside the see-saw
CX, CY, CZ = PXs + 2.7, 3.3, ground_z(PXs) + 0.5
box((CX, CY, CZ + 0.05), (2.6, 1.3, 0.25), M_WHITEW, bev=True)                  # mattress base
box((CX, CY, CZ + 0.25), (2.4, 1.15, 0.2), mat('sheet', (1.0, 0.97, 0.92), 0.8), bev=True)
for xx in np.linspace(CX - 1.3, CX + 1.3, 13):
    rod((xx, CY - 0.66, CZ - 0.1), (xx, CY - 0.66, CZ + 1.15), 0.025, M_WHITEW, 8)
for zz in (CZ - 0.1, CZ + 1.15): rod((CX - 1.35, CY - 0.66, zz), (CX + 1.35, CY - 0.66, zz), 0.045, M_WHITEW, 8)
for xx in (CX - 1.35, CX + 1.35): rod((xx, CY - 0.7, ground_z(CX) - 0.3), (xx, CY - 0.7, CZ + 1.4), 0.07, M_WHITEW, 10)
box((CX + 0.25, CY, CZ + 0.45), (1.6, 1.0, 0.28), M_BLANKET, bev=True)
head = sphere((CX - 0.8, CY - 0.05, CZ + 0.62), 0.34, M_SKINB, (24, 12), 'babyhead')
cap = sphere((CX - 0.86, CY + 0.02, CZ + 0.72), 0.36, M_CAP, (24, 12), 'cap'); cap.scale = (0.37, 0.37, 0.3)
BEYES = []
for dx in (-0.12, 0.1):
    ep = sphere((CX - 0.8 + dx, CY - 0.36, CZ + 0.62), 0.045, M_PUPIL, (10, 5), 'babyeye'); BEYES.append(ep)
    key_series(ep, 'scale', 2, np.array([0.045 if t < FINAL + 0.25 else max(0.006, 0.045 - (t - FINAL - 0.25) * 0.15) for t in T]), 1e-4)
    sphere((CX - 0.8 + dx * 1.5, CY - 0.33, CZ + 0.52), 0.05, M_BLUSH, (8, 4), 'babyblush').scale = (0.06, 0.03, 0.04)
smile = rod((CX - 0.86, CY - 0.36, CZ + 0.5), (CX - 0.74, CY - 0.36, CZ + 0.5), 0.012, M_PUPIL, 6)
# a warm night-light lamp beside the crib (switched off after the chord)
LX = CX - 1.0
rod((LX, CY + 1.0, ground_z(LX) - 0.2), (LX, CY + 1.0, CZ + 1.4), 0.05, M_WHITEW, 8)
shade = obj(prim('cone', mat('shade', (1.0, 0.85, 0.6), 0.6, **{'Emission Color': (1.0, 0.75, 0.4), 'Emission Strength': 3.0}), 24, 0.45, 0.28, 0.5), 'shade')
shade.location = (LX, CY + 1.0, CZ + 1.6)
_sh = shade.data.materials[0].node_tree.nodes['Principled BSDF']
for f in range(0, NF, 2):
    _sh.inputs['Emission Strength'].default_value = float(3.0 * (1 - OUT[f]) + 0.0); _sh.inputs['Emission Strength'].keyframe_insert('default_value', frame=f)

# ---------------- signposts in the meadow (search keywords, English first, Japanese below) ----------------
def signboard(lines, x, w, h, zc, y=-2.3):
    box((x, y + 0.06, zc), (w, 0.1, h), M_WOOD, bev=True, name='sign')
    box((x, y - 0.0, zc), (w - 0.18, 0.04, h - 0.18), mat('signface', (0.98, 0.93, 0.82), 0.7), name='signface')
    for sx in (-1, 1): rod((x + sx * (w / 2 - 0.3), y + 0.1, zc - h / 2), (x + sx * (w / 2 - 0.3), y + 0.1, ground_z(x) - DROP_F - 0.1), 0.06, M_WOOD, 8)
    tot_h = sum(sz * 1.35 for _, sz, _ in lines); zz = zc + tot_h / 2
    for s_, size, mt in lines:
        zz -= size * 1.35
        fit_w(text(s_, (x, y - 0.05, zz + size * 0.67), size, mt, rot=(math.pi / 2, 0, 0)), w - 0.4)
M_SIGN1 = mat('signink1', (0.42, 0.2, 0.5), 0.5); M_SIGN2 = mat('signink2', (0.2, 0.32, 0.6), 0.5); M_SIGN3 = mat('signink3', (0.55, 0.3, 0.15), 0.5)
for tb, w_, h_, lines in ((3.2, 4.6, 2.1, [("Brahms' Lullaby", 0.5, M_SIGN1), ('Wiegenlied Op.49 No.4 (1868)', 0.26, M_SIGN3), ('ブラームスの子守歌', 0.24, M_SIGN2)]),
                          (9.4, 4.4, 1.9, [('Johannes Brahms', 0.44, M_SIGN1), ('Classical music for sleep', 0.28, M_SIGN3), ('クラシック 子守唄', 0.24, M_SIGN2)]),
                          (15.6, 4.4, 1.9, [('Marble Machine', 0.48, M_SIGN2), ('Satisfying · ASMR', 0.3, M_SIGN3), ('ビー玉マシン', 0.24, M_SIGN1)]),
                          (21.8, 4.4, 1.9, [('Baby Sleep Music', 0.44, M_SIGN1), ('Count the sheep', 0.3, M_SIGN3), ('赤ちゃん 寝かしつけ', 0.24, M_SIGN2)]),
                          (28.0, 4.4, 1.9, [('Music Box Lullaby', 0.42, M_SIGN2), ('Sleepy marbles', 0.3, M_SIGN3), ('オルゴール 子守歌', 0.24, M_SIGN1)])):
    xx = xs(tb) + 0.4
    signboard(lines, xx, w_, h_ - 0.2, ground_z(xx) - DROP_F + 0.75 + h_ / 2)

# ---------------- background: big soft clouds far behind, a few toy blocks in the foreground ----------------
for k, x0 in enumerate(np.arange(-10, FLX[-1], 7.0)):
    zz = ground_z(x0) + rng.uniform(5.0, 9.5)
    for j in range(5):
        sphere((x0 + j * 0.8 + rng.uniform(-0.2, 0.2), 9.0 + rng.uniform(0, 1.5), zz + rng.uniform(-0.3, 0.4)), rng.uniform(0.6, 1.1), M_CLOUD, (16, 8), 'bigcloud')
BLK = 'BRAHMS'
for k, x0 in enumerate(np.arange(xs(6.3), xs(40), 6.2)):
    for j in range(2):
        b_ = box((x0 + 0.5 * j, -1.6, ground_z(x0) - DROP_F + 0.21), (0.42, 0.42, 0.42), M_OBJ, rot=(0, 0, rng.uniform(-0.3, 0.3)), bev=True, name='toyblock')
        b_.color = (*pcol(int(rng.integers(0, 12)), 0.4), 1)

# ---------------- constellation: after the lights go out, lines join the lit stars in the order they were played ----------------
CON_T0 = T_OUT + 0.6; CON_DUR = 2.6
for j in range(len(STARS) - 1):
    a = STARS[j][0]; b = STARS[j + 1][0]
    ln = rod(a, b, 0.018, M_FX, 4); ln.color = (0.8, 0.88, 1.0, 0)
    tj = CON_T0 + CON_DUR * (1 - j / max(1, len(STARS) - 2))         # drawn from the end (the crib) back to the start
    key_series(ln, 'color', 3, np.clip((T - tj) / 0.25, 0, 1) * 0.55, 1e-3)
# every lit star shines brighter in the dark
for oc, th, so, col, ssz in STARS:
    a_ = np.where(T < th, 0.0, 0.32) + 0.5 * STARS_ON * (T >= th)
    fh = int(round(th * FPS))
    for k in range(fh, min(NF, fh + 19)): a_[k] = max(a_[k], np.exp(-(k - fh) / 5.0))
    so.animation_data_clear(); key_series(so, 'color', 3, a_, 1e-3)
# glow-in-the-dark sky stars
for k in range(260):
    ph = rng.uniform(-1.0, 1.0); x_ = rng.uniform(-14, FLX[-1]); y_ = rng.uniform(7.0, 13.0); z_ = ground_z(x_) + rng.uniform(1.0, 13.0)
    s_ = sphere((x_, y_, z_), rng.uniform(0.03, 0.07), M_FX, (6, 3), 'skystar'); s_.color = (0.85, 0.9, 1.0, 0)
    tw = 0.6 + 0.4 * np.sin(rng.uniform(1.5, 4.0) * T + rng.uniform(0, 6.28))
    key_series(s_, 'color', 3, np.round(STARS_ON * tw * rng.uniform(0.4, 1.0), 2) + 0.06, 0.02)

# ---------------- the counter: sheep asleep ----------------
CNT = np.zeros(NF, int)
for n in SHEEPN: CNT += (T >= (SLEEP_T[n] if n != 'giant' else EV['seesaw']['t_launch'])).astype(int)
print('sheep counter final', CNT[-1])

# ---------------- world & lights ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes['Background']
_wt = w.node_tree; _tc2 = _wt.nodes.new('ShaderNodeTexCoord'); _sp = _wt.nodes.new('ShaderNodeSeparateXYZ'); _cr = _wt.nodes.new('ShaderNodeValToRGB')
_wt.links.new(_tc2.outputs['Generated'], _sp.inputs[0]); _cr.color_ramp.interpolation = 'EASE'; _wt.links.new(_sp.outputs['Z'], _cr.inputs['Fac'])
_cr.color_ramp.elements[0].position = 0.02; _cr.color_ramp.elements[0].color = (0.16, 0.09, 0.26, 1)
_cr.color_ramp.elements[1].position = 0.22; _cr.color_ramp.elements[1].color = (0.015, 0.025, 0.1, 1)
_wt.links.new(_cr.outputs['Color'], bg.inputs['Color'])
for f in range(0, NF, 3):
    bg.inputs['Strength'].default_value = float(0.9 * (1 - OUT[f]) + 0.3 * OUT[f]); bg.inputs['Strength'].keyframe_insert('default_value', frame=f)
mn = bpy.data.lights.new('moon', 'SUN'); mn.color = (0.65, 0.75, 1.0); mn.angle = 0.08
mno = link(bpy.data.objects.new('moon', mn)); mno.rotation_euler = (math.radians(45), math.radians(-12), math.radians(-25))
for f in range(0, NF, 3):
    mn.energy = float(0.35 + 0.25 * OUT[f]); mn.keyframe_insert('energy', frame=f)

# ---------------- camera: follow the stage, keep all three lanes in the shot; at the end turn to look back along the trail ----------------
FOC = np.zeros((NF, 3)); WID = np.zeros(NF)
for f in range(NF):
    t = T[f]
    FOC[f] = np.array([xs(min(t, FINAL - 1.2)) + 0.3, 0, zs(t) + 1.8]); WID[f] = 5.6
TEND = FINAL - 2.2
X_END = PXs + 0.9; Z_END = 0.5 * (ground_z(PXs) + mz) + 0.4
for f in range(NF):
    t = T[f]
    if t > TEND:
        u = min(1, (t - TEND) / 1.6); u = u * u * (3 - 2 * u)
        FOC[f] = (1 - u) * FOC[f] + u * np.array([X_END, 0, Z_END]); WID[f] = (1 - u) * WID[f] + u * 7.4
FOC[:, 0] = smooth(FOC[:, 0], 0.5); FOC[:, 2] = smooth(FOC[:, 2], 0.9); WID = smooth(WID, 0.5)
LENS = 32.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 500
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
YAW = np.array([0.0 if t < T_OUT + 0.3 else 0.55 * (lambda u: u * u * (3 - 2 * u))(min(1.0, (t - T_OUT - 0.3) / 2.8)) for t in T])
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24
    yw = YAW[f]
    CP[f] = FOC[f] + np.array([D * math.sin(yw), -D * math.cos(yw), 0.2 * D + 3.0 * yw])
    TP[f] = FOC[f] + np.array([-3.5 * yw, 0, 0.2])
key_loc(cam, CP); key_loc(tgt, TP)
# warm room light riding with the camera (off when the lamp goes out), a cool fill
for k, (dx, dz, en) in enumerate(((-3.0, 6.5, 1400), (3.5, 5.5, 1000), (0.0, 1.0, 450))):
    al = bpy.data.lights.new(f'room{k}', 'AREA'); al.size = 6.0; al.color = (1.0, 0.78, 0.55) if k < 2 else (0.75, 0.8, 1.0)
    ao = link(bpy.data.objects.new(f'room{k}', al))
    key_loc(ao, np.stack([FOC[:, 0] + dx, np.full(NF, -4.5 if k < 2 else -6.0), FOC[:, 2] + dz], 1))
    ao.rotation_euler = (math.radians(-28), 0, 0) if k < 2 else (math.radians(-80), 0, 0)
    for f in range(0, NF, 3):
        al.energy = float(en * (1 - 0.93 * OUT[f]) if k < 2 else en * (1 - 0.5 * OUT[f])); al.keyframe_insert('energy', frame=f)
# a soft spot on the star marble
spd = bpy.data.lights.new('spotstar', 'SPOT'); spd.spot_size = math.radians(16); spd.spot_blend = 0.7; spd.color = (1.0, 0.9, 0.75); spd.shadow_soft_size = 0.3
so_ = link(bpy.data.objects.new('spotstar', spd)); stg = link(bpy.data.objects.new('spottgt', None))
Ps = POS['star'].copy()
for k in range(3): Ps[:, k] = smooth(Ps[:, k], 0.25)
key_loc(stg, Ps - np.array([0, 0, 0.25])); key_loc(so_, Ps + np.array([-2.0, -6.0, 9.0]))
c_ = so_.constraints.new('TRACK_TO'); c_.target = stg; c_.track_axis = 'TRACK_NEGATIVE_Z'; c_.up_axis = 'UP_Y'
for f in range(0, NF, 3):
    spd.energy = float(900 * (1 - 0.7 * OUT[f])); spd.keyframe_insert('energy', frame=f)

# ---------------- HUD (rides with the camera): hook, counter, ending ----------------
HUD_D = 3.0
def hud_text(s, x, y, size, m, align='CENTER'):
    o = text(s, (x, y, -HUD_D), size, m, rot=(0, 0, 0), align=align, ext=0.0); o.parent = cam; return o
def vis(o, a):
    s0 = tuple(o.scale)
    for c in range(3): key_series(o, 'scale', c, np.maximum(0.001, a * s0[c]), 1e-3)
M_HUDW = emis('hudw', (1.0, 0.97, 0.92), 3.0); M_HUDG = emis('hudg', (1.0, 0.82, 0.45), 4.0); M_HUDP = emis('hudp', (1.0, 0.72, 0.85), 3.5)
M_HUDBG = mat('hudbg', (0.02, 0.02, 0.06), 1.0, **{'Alpha': 0.5})
hook_a = np.array([1.0 if t < 3.9 else max(0.0, 1 - (t - 3.9) / 0.3) for t in T])
h1 = fit_w(hud_text('How many sheep', 0, 0.9, 0.1, M_HUDW), 0.95); vis(h1, hook_a)
h2 = fit_w(hud_text('until the baby sleeps?', 0, 0.79, 0.085, M_HUDG), 0.95); vis(h2, hook_a)
h3 = hud_text('赤ちゃんが眠るまで ひつじは何匹？', 0, 0.71, 0.044, M_HUDW); vis(h3, hook_a)
hb_ = obj(prim('plane', M_HUDBG), 'hudbg'); hb_.parent = cam; hb_.location = (0, 0.81, -HUD_D - 0.01); hb_.scale = (0.75, 0.11, 1); vis(hb_, hook_a)
cnt_a = np.array([0.0 if t < 3.9 else min(1.0, (t - 3.9) / 0.3) for t in T]) * np.array([1.0 if t < FINAL + 0.8 else max(0.0, 1 - (t - FINAL - 0.8) / 0.3) for t in T])
lab = hud_text('SHEEP', -0.09, 0.9, 0.075, M_HUDG, align='RIGHT'); vis(lab, cnt_a)
cb_ = obj(prim('plane', M_HUDBG), 'cntbg'); cb_.parent = cam; cb_.location = (0.0, 0.9, -HUD_D - 0.01); cb_.scale = (0.3, 0.045, 1); vis(cb_, cnt_a)
DIG_X = [0.0, 0.075]
for pos in range(2):
    dv = (CNT // (10 ** (1 - pos))) % 10
    shown = CNT >= 10 if pos == 0 else np.ones(NF, bool)
    xo = DIG_X[pos] - (0.0 if pos == 0 else 0.0)
    for dgt in range(10):
        o = hud_text(str(dgt), xo, 0.9, 0.11, M_HUDW)
        vis(o, ((dv == dgt) & shown).astype(float) * cnt_a)
cjp = hud_text('ひつじ', 0.16, 0.9, 0.04, M_HUDW, align='LEFT'); vis(cjp, cnt_a)
end_a = np.array([0.0 if t < FINAL + 1.0 else min(1.0, (t - FINAL - 1.0) / 0.4) for t in T])
e1 = fit_w(hud_text('25 sheep... Good night', 0, 0.86, 0.085, M_HUDW), 0.95); vis(e1, end_a)
e2 = fit_w(hud_text("Brahms' Lullaby", 0, 0.76, 0.07, M_HUDG), 0.95); vis(e2, end_a)
e3 = hud_text('ひつじ25匹で おやすみなさい', 0, 0.69, 0.044, M_HUDW); vis(e3, end_a)
eb_ = obj(prim('plane', M_HUDBG), 'endbg'); eb_.parent = cam; eb_.location = (0, 0.79, -HUD_D - 0.01); eb_.scale = (0.75, 0.1, 1); vis(eb_, end_a)

EV_OUT = dict(pieces=[dict(kind=p['kind'], t=p['t'], top=list(map(float, p['top'])), n=list(map(float, p['n'])), hw=float(p['hw']), hd=float(p['hd']), th=float(p['th']), ball=p['ball'],
                           rs=p.get('rs'), own=p.get('own'), life=p.get('life'), sway=p.get('sway')) for p in PIECES], counter=dict(final=int(CNT[-1])))
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
        print('t', tt_, 'FOC', np.round(FOC[f], 2), 'star', np.round(POS['star'][f], 2), 'pink', np.round(POS['pink'][f], 2), 'cnt', CNT[f])
