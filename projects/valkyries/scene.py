"""Ride of the Valkyries - 8 riders: who reaches Valhalla first? Side view of a storm sky over the Valkyrie rock, then a storm
bridge and the golden bridge of Valhalla. Every theme note = a rider landing on a key of that pitch (eight riders, eight lanes),
every bass note = the thunder ball on a drum. layout.py plans all motion; this file builds and animates it.
jp_sub.otf = Noto Sans CJK JP Black (SIL OFL) subset to the characters used here (make_font.py); en_bold.ttf = DejaVu Sans Bold.
python3 scene.py -- test t1 t2 ... | range f0 f1 | probe t ..."""
import os, sys, json, math, time
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
LAY = json.load(open(f'{HERE}/layout.json'))
FPS = 30; FINAL = E['final']; NF = int((FINAL + 6.2) * FPS); T = np.arange(NF) / FPS
G = LAY['G']; EV = LAY['ev']; LN = LAY['lane']
TN = np.array([l[0] for l in LEAD])
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
FONT = bpy.data.fonts.load(f'{HERE}/jp_sub.otf')
FONT_EN = bpy.data.fonts.load(f'{HERE}/en_bold.ttf')
TXT_K = 2.75


def link(o): sc.collection.objects.link(o); return o


def mat(name, col, rough=0.5, **kw):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']; b.inputs['Base Color'].default_value = (*col, 1); b.inputs['Roughness'].default_value = rough
    for k, v in kw.items():
        if k in ('Emission Color', 'Sheen Tint') and len(v) == 3: v = (*v, 1)
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


def objcol(name, rough=0.45, metal=0.0, emit=0.0, coat=0.0):
    m = mat(name, (1, 1, 1), rough, **{'Metallic': metal, 'Coat Weight': coat}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    oi = nt.nodes.new('ShaderNodeObjectInfo'); nt.links.new(oi.outputs['Color'], b.inputs['Base Color'])
    if emit:
        nt.links.new(oi.outputs['Color'], b.inputs['Emission Color']); b.inputs['Emission Strength'].default_value = emit
    return m


def key_series(obj, path, idx, vals, eps=1e-5, const=False):
    n = len(vals); fc_target = obj
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
    elif kind in ('bell', 'gbell'):
        # bell, mouth down, height 1, mouth radius 1. 'gbell' (great bell): a hemispherical crown of radius 0.55 / 1.15 at the top
        if kind == 'bell':
            prof = [(0.0, 0.0), (0.42, 0.0), (0.5, -0.08), (0.55, -0.35), (0.62, -0.62), (0.8, -0.86), (1.0, -1.0), (0.96, -1.0), (0.78, -0.88),
                    (0.58, -0.64), (0.5, -0.36), (0.44, -0.1), (0.0, -0.1)]
        else:
            rc = 0.55 / 1.15
            prof = [(rc * math.sin(math.radians(a_)), rc * (math.cos(math.radians(a_)) - 1)) for a_ in range(0, 91, 6)]
            prof += [(0.52, -0.55), (0.6, -0.75), (0.8, -0.92), (1.0, -1.0), (0.96, -1.0), (0.76, -0.9), (0.56, -0.7), (0.46, -0.5), (0.0, -0.45)]
        segs = a[0]; rings = []
        for k in range(segs):
            ph = 2 * math.pi * k / segs
            rings.append([bm.verts.new((r * math.cos(ph), r * math.sin(ph), z)) for r, z in prof])
        for k in range(segs):
            r0, r1 = rings[k], rings[(k + 1) % segs]
            for j in range(len(prof) - 1): bm.faces.new((r0[j], r1[j], r1[j + 1], r0[j + 1]))
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    me = bpy.data.meshes.new(kind); bm.to_mesh(me); bm.free()
    if kind in ('sph', 'cone', 'bell', 'gbell', 'cyl', 'bcube'):
        for p in me.polygons: p.use_smooth = True
    if m: me.materials.append(m)
    _MC[key] = me; return me


def obj(me, name='o'): return link(bpy.data.objects.new(name, me))


def box(c, size, m, rot=None, name='box', bevel=False):
    o = obj(prim('bcube' if bevel else 'cube', m), name); o.location = c; o.scale = size
    if rot is not None: o.rotation_euler = rot
    return o


def rod(p0, p1, r, m, verts=10):
    p0, p1 = Vector(p0), Vector(p1); d = p1 - p0
    o = obj(prim('cyl', m, verts), 'rod'); q = Vector((0, 0, 1)).rotation_difference(d.normalized() if d.length > 1e-9 else Vector((0, 0, 1)))
    o.matrix_world = Matrix.Translation((p0 + p1) / 2) @ q.to_matrix().to_4x4() @ Matrix.Diagonal((r, r, max(1e-4, d.length), 1))
    return o


def sphere(c, r, m, seg=(16, 8), name='sph'):
    o = obj(prim('sph', m, *seg), name); o.location = c; o.scale = (r, r, r); return o


def text(s, loc, size, m, rot=(math.pi / 2 - 0.33, 0, 0), align='CENTER', ext=0.02):
    cu = bpy.data.curves.new('t', 'FONT'); cu.body = s; cu.font = FONT_EN if _latin(s) else FONT; cu.size = size * TXT_K * (EN_K if _latin(s) else 1.0); cu.extrude = ext
    cu.align_x = align; cu.align_y = 'CENTER'
    o = link(bpy.data.objects.new('txt', cu)); o.location = loc; o.rotation_euler = rot; o.data.materials.append(m); return o


# English-first on-screen text (2026-11-12): Latin strings use DejaVu Sans Bold (en_bold.ttf, free DejaVu font license),
# Japanese subtitles keep the Noto CJK subset.
def _latin(s): return all(ord(c) < 0x2E80 for c in s)
EN_K = 0.75                  # DejaVu renders ~1.45x taller than the CJK font at the same size: match the line heights


def fit_w(o, w):
    """shrink a text object uniformly so that it is at most w wide"""
    bpy.context.view_layer.update()
    if w and o.dimensions.x > w: k = w / o.dimensions.x; o.scale = (k, k, k)
    return o


def label(en, jp, loc, size, m, mj=None, k=0.5, w=None, **kw):
    """English line at loc, a small Japanese line just below it (in the text plane). Returns [en_obj, jp_obj]."""
    o = fit_w(text(en, loc, size, m, **kw), w)
    if not jp: return [o]
    R = o.rotation_euler.to_matrix()
    d = R @ Vector((0, -(size * 0.62 + size * k * 0.62) * o.scale[0] / 1.0, 0))
    o2 = fit_w(text(jp, tuple(Vector(loc) + d), size * k, mj or M_TXT, **kw), w)
    return [o, o2]


def frame_from_n(n):
    n = np.array(n, float); n /= np.linalg.norm(n)
    xa = np.cross([0, 1, 0], n); xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return Matrix(((xa[0], ya[0], n[0]), (xa[1], ya[1], n[1]), (xa[2], ya[2], n[2])))


def smooth(Xa, k):
    Y = Xa.copy(); a = 1 - math.exp(-1 / FPS / k)
    for f in range(1, len(Xa)): Y[f] = Y[f - 1] + (Xa[f] - Y[f - 1]) * a
    for f in range(len(Xa) - 2, -1, -1): Y[f] = Y[f + 1] + (Y[f] - Y[f + 1]) * a
    return Y


# ---------------- ball paths ----------------
def at_many(b, ts):
    out = np.zeros((len(ts), 3))
    for g in b['seg']:
        m = (ts >= g['t0'] - 1e-9) & (ts <= g['t1'] + 1e-9)
        if not m.any(): continue
        tt = ts[m]
        if g['k'] == 'rest': out[m] = g['p']
        elif g['k'] == 'fly':
            u = (tt - g['t0'])[:, None]; out[m] = np.array(g['p']) + np.array(g['v']) * u + np.array([0, 0, -0.5 * G]) * u * u
        else:
            gt = np.array(g['ts']); gp = np.array(g['ps'])
            out[m] = np.stack([np.interp(tt, gt, gp[:, c]) for c in range(3)], 1)
    s0 = b['seg'][0]; s1 = b['seg'][-1]
    def one(g, t):
        if g['k'] == 'rest': return np.array(g['p'])
        if g['k'] == 'fly':
            u = t - g['t0']; return np.array(g['p']) + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
        return np.array(g['ps'][0] if t <= g['t0'] else g['ps'][-1])
    out[ts < s0['t0']] = one(s0, s0['t0']); out[ts > s1['t1']] = one(s1, s1['t1'])
    return out
def strip_mesh(X, Z, y0, y1, th, m, name):
    bm = bmesh.new(); vs = []
    for x, z in zip(X, Z):
        vs.append([bm.verts.new((x, y0, z)), bm.verts.new((x, y1, z)), bm.verts.new((x, y1, z - th)), bm.verts.new((x, y0, z - th))])
    for a, b in zip(vs, vs[1:]):
        for k in range(4): bm.faces.new((a[k], b[k], b[(k + 1) % 4], a[(k + 1) % 4]))
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new(name, me)); o.data.materials.append(m)
    return o


def riser(objs, t_hit, anchor, lead=1.2, stay=0.9):
    """objects pop into existence (scale around `anchor`) at t_hit-lead and vanish after t_hit+stay"""
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


def flash(o, t, col, n=(0, 0, 1), dip=0.03, levels=((0, 1.0), (4, 0.55), (18, 0.1))):
    fh = int(round(t * FPS)); base = o.location.copy(); nv = Vector(n)
    o.color = (*col, 0); o.keyframe_insert('color', index=3, frame=max(0, fh - 1))
    for df, a in levels:
        o.color = (*col, a); o.keyframe_insert('color', index=3, frame=fh + df)
    for df, d in ((-1, 0), (1, -dip), (4, dip * 0.3), (8, 0)):
        o.location = base + nv * d; o.keyframe_insert('location', frame=max(0, fh + df))


BL = LAY['balls']
POS = {n: at_many(b, T) for n, b in BL.items()}
RIDN = [f'v{j}' for j in range(8)]
NAMES = EV['names']
T9 = E['bar_t']['9']; T17 = E['bar_t']['17']
LEAP = EV['leap']; BR = EV['bridge']; FIN = EV['finish']
def tri_up(t, t0, t1): return min(1.0, max(0.0, (t - t0) / (t1 - t0)))
# dawn: 0 = storm night, 1 = golden sunrise (the clouds part during the leap)
DAWN = np.array([tri_up(t, LEAP['t0'] - 0.2, T17 + 0.8) for t in T]); DAWN = DAWN * DAWN * (3 - 2 * DAWN)

# ---------------- materials ----------------
M_GOLD = mat('gold', (1.0, 0.72, 0.28), 0.2, **{'Metallic': 1.0})
M_GOLDD = mat('goldd', (0.55, 0.36, 0.14), 0.32, **{'Metallic': 0.9})
M_IRON = mat('iron', (0.12, 0.13, 0.16), 0.35, **{'Metallic': 0.9})
M_STEEL = mat('steel', (0.6, 0.65, 0.75), 0.25, **{'Metallic': 1.0})
M_ROCK = mat('rock', (0.09, 0.085, 0.1), 0.85)
M_ROCK2 = mat('rock2', (0.16, 0.15, 0.17), 0.8)
M_STONE = mat('stone', (0.2, 0.2, 0.24), 0.7)
M_STONEW = mat('stonew', (0.75, 0.68, 0.58), 0.6)
M_MARBLE = mat('marble', (0.92, 0.88, 0.8), 0.25, **{'Coat Weight': 0.5})
M_WOOD = mat('wood', (0.28, 0.14, 0.06), 0.55)
M_COPPER = mat('copper', (0.85, 0.42, 0.22), 0.25, **{'Metallic': 1.0})
M_HEAD = glowmat('drumhead', gain=5.0, base=0.0, rough=0.6)
M_KEY = glowmat('key', gain=9.0, base=0.45, rough=0.12, coat=1.0)
M_BELL = glowmat('fbell', gain=5.0, base=0.1, rough=0.18, metal=0.95)
M_RIDER = objcol('rider', 0.06, metal=0.3, emit=0.6, coat=1.0)
M_THUNDER = mat('thunder', (0.08, 0.1, 0.16), 0.2, **{'Metallic': 0.8, 'Coat Weight': 1.0, 'Emission Color': (0.35, 0.6, 1.0), 'Emission Strength': 0.6})
M_FX = fxmat('fx', 8.0)
M_BOLT = emis('bolt', (0.75, 0.85, 1.0), 40.0)
M_TXT = emis('txtw', (1.0, 0.97, 0.92), 4.0); M_TXT_G = emis('txtg', (1.0, 0.78, 0.3), 6.0); M_TXT_B = emis('txtb', (0.5, 0.8, 1.0), 6.0)
M_TXT_R = emis('txtr', (1.0, 0.4, 0.35), 5.0)
M_PLATE = mat('plate', (0.03, 0.035, 0.06), 0.3, **{'Coat Weight': 1.0})
M_LAMP = objcol('lamp', 0.4, emit=8.0)
M_GATEGLOW = emis('gateglow', (1.0, 0.85, 0.55), 0.0)
M_CLOUD = mat('cloud', (0.25, 0.27, 0.33), 0.9)
# the clouds and the sky change colour at dawn (keyed material inputs)
_cb = M_CLOUD.node_tree.nodes['Principled BSDF'].inputs['Base Color']
for f in (0, int((LEAP['t0'] - 0.2) * FPS), int((T17 + 0.8) * FPS)):
    u = DAWN[min(f, NF - 1)]
    _cb.default_value = tuple(np.array([0.34, 0.37, 0.48, 1]) * (1 - u) + np.array([0.75, 0.5, 0.42, 1]) * u)
    _cb.keyframe_insert('default_value', frame=f)
PITCH_COL = [(1.0, 0.12, 0.12), (1.0, 0.35, 0.05), (1.0, 0.6, 0.05), (1.0, 0.85, 0.1), (0.55, 1.0, 0.1), (0.1, 1.0, 0.3),
             (0.05, 1.0, 0.75), (0.05, 0.7, 1.0), (0.15, 0.35, 1.0), (0.45, 0.2, 1.0), (0.75, 0.2, 1.0), (1.0, 0.2, 0.65)]
def pcol(p, light=0.0):
    c = PITCH_COL[p % 12]; return tuple(min(1, x * (1 - light) + light) for x in c)
RCOL = [(1.0, 0.2, 0.2), (1.0, 0.55, 0.1), (1.0, 0.9, 0.2), (0.3, 1.0, 0.35), (0.2, 0.85, 1.0), (0.3, 0.4, 1.0), (0.75, 0.3, 1.0),
        (1.0, 0.4, 0.8)]

# ---------------- balls (the eight riders with light trails, the thunder ball) ----------------
BALL = {}
for n, b in BL.items():
    rr = b['r']
    o = obj(prim('sph', M_RIDER if b['role'] == 'rider' else M_THUNDER, 32, 16), n); o.scale = (rr, rr, rr)
    P = POS[n]; key_loc(o, P)
    sp = np.zeros(NF)
    for f in range(1, NF): sp[f] = sp[f - 1] + (P[f, 0] - P[f - 1, 0]) / rr
    o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, sp)
    bd = obj(bpy.data.meshes.new('band'), 'band')
    bm = bmesh.new(); bmesh.ops.create_cone(bm, cap_ends=False, segments=32, radius1=1.006, radius2=1.006, depth=0.18); bm.to_mesh(bd.data); bm.free()
    bd.data.materials.append(M_GOLD if b['role'] == 'rider' else M_STEEL); bd.parent = o; bd.rotation_euler = (math.pi / 2, 0, 0)
    if b['role'] == 'rider':
        j = int(n[1:]); o.color = (*RCOL[j], 1)
        # light trail: ghosts a few frames behind, fading
        for g_ in range(1, 6):
            gh = obj(prim('sph', M_FX, 12, 6), 'trail'); gh.color = (*RCOL[j], 0.55 * (1 - g_ / 6))
            sc_ = rr * (1 - 0.12 * g_)
            Pg = np.concatenate([np.repeat(P[:1], g_, 0), P[:-g_]])
            mv = np.linalg.norm(P - Pg, axis=1) > 0.04                 # only while moving
            key_loc(gh, Pg)
            for k in range(3): key_series(gh, 'scale', k, np.where(mv, sc_, 0.001))
    BALL[n] = o

# ---------------- keys (one per rider note), drums, final bells ----------------
def floor_under(x, z): return z - 1.6
def stem(top, m=M_GOLDD, r=0.03):
    return rod((top[0], top[1], top[2] - 0.05), (top[0], top[1], top[2] - 1.5), r, m, 8)
KEY_TH = 0.12
RING_EV = []
for j, n in enumerate(RIDN):
    b = BL[n]
    for h in b['hits']:
        if h['kind'] == 'cup': continue
        nv = np.array(h['n']) if np.linalg.norm(h['vout']) > 0 else np.array([0, 0, 1.0])
        top = np.array(h['p']) - b['r'] * nv; p = h['pitch']; col = pcol(p, 0.0)
        if h['kind'] == 'final':
            # the bells of the final chord: golden bowls on the finish line, they stay
            sz = 0.55 - 0.012 * (p - 47)
            o = obj(prim('bell', M_BELL, 32), 'fbell')
            o.matrix_world = Matrix.Translation(Vector(top - np.array([0, 0, sz * 0.15]))) @ Matrix.Diagonal((sz * 0.62, sz * 0.62, -sz * 0.6, 1))
            o.color = (*pcol(p, 0.3), 0); flash(o, h['t'], pcol(p, 0.3), (0, 0, 1), dip=0.0, levels=((0, 1.0), (6, 0.7), (40, 0.35)))
            st_ = rod(top - np.array([0, 0, sz * 0.7]), top - np.array([0, 0, 2.2]), 0.05, M_GOLDD, 10)
            riser([o, st_], h['t'], top, lead=0.6, stay=60.0)        # rises after the last note (no overlap with the last key)
            RING_EV.append((h['t'], top, pcol(p, 0.4)))
            continue
        w_ = 0.32
        o = obj(prim('bcube', M_KEY), 'key')
        o.matrix_world = Matrix.Translation(Vector(top - nv * KEY_TH / 2)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((w_, 0.5 if p >= 57 else 0.56, KEY_TH, 1))
        o.color = (*col, 0); flash(o, h['t'], col, nv)
        objs = [o, stem(top)]
        if h['t'] > 1.0: riser(objs, h['t'], top, lead=1.0, stay=0.8)
        RING_EV.append((h['t'], top, pcol(p, 0.4)))
B_ = BL['bass']
for h in B_['hits']:
    nv = np.array(h['n']) if np.linalg.norm(h['vout']) > 0 else np.array([0, 0, 1.0])
    top = np.array(h['p']) - B_['r'] * nv; fin = h['kind'] == 'final'
    s = 0.42 if not fin else 0.62
    e = link(bpy.data.objects.new('drum', None)); e.matrix_world = Matrix.Translation(Vector(top)) @ frame_from_n(nv).to_4x4()
    bowl = obj(prim('cone', M_COPPER if h['kind'] != 'lever' else M_GOLD, 32, 1.0, 0.55, 1.0), 'kettle'); bowl.parent = e
    bowl.location = (0, 0, -0.5 * s); bowl.scale = (s, s, s * 0.9)
    head = obj(prim('cyl', M_HEAD, 32), 'head'); head.parent = e; head.location = (0, 0, -0.012); head.scale = (s * 1.02, s * 1.02, 0.02)
    head.color = (0.6, 0.8, 1.0, 0); flash(head, h['t'], (0.6, 0.8, 1.0), (0, 0, 1), dip=0.0)
    rim = obj(prim('cyl', M_STEEL, 32), 'rim'); rim.parent = e; rim.location = (0, 0, -0.02); rim.scale = (s * 1.06, s * 1.06, 0.04)
    for a_ in range(3):
        ph = 2 * math.pi * a_ / 3
        lg = rod((s * 0.6 * math.cos(ph), s * 0.6 * math.sin(ph), -0.8 * s), (s * 0.7 * math.cos(ph), s * 0.7 * math.sin(ph), -2.0), 0.025, M_STEEL, 6)
        lg.parent = e
    if h['kind'] == 'lever':
        # the lever drum: its rod runs forward to the drawbridge chain
        rd = rod((s, 0, -0.3 * s), (s + 1.6, 0, 0.6), 0.04, M_GOLD); rd.parent = e
    RING_EV.append((h['t'], top, (0.6, 0.8, 1.0)))
    if not fin and h['t'] > 1.0: riser([e], h['t'], top, lead=1.1, stay=0.8)

# ---------------- act 1: the Valkyrie rock with the eight gates, the start line ----------------
S1 = EV['s1']; LANE = EV['lane']
xs_perch = min(s['perch'][0] for s in S1); zs_perch = max(s['perch'][2] for s in S1)
# the rock: a dark crag behind / under the gates
for k in range(9):
    yy = -1.0 + k * LANE * 1.0
    box((xs_perch - 0.5 - 1.3, yy, zs_perch - 3.0), (2.6, LANE * 1.15, 7.0), M_ROCK, name='crag', bevel=True).rotation_euler = (0, 0.05 * (k % 3 - 1), 0.04 * (k % 2))
for k in range(14):
    x_ = xs_perch - 3.5 + rng.uniform(-1.5, 1.5); y_ = rng.uniform(-1, 9); z_ = zs_perch + rng.uniform(-6, 1.5)
    o = sphere((x_, y_, z_), rng.uniform(0.8, 1.8), M_ROCK2, (8, 6), 'boulder'); o.scale = (o.scale[0], o.scale[1] * 0.8, o.scale[2] * 1.3)
FL_T = []
for s in S1:
    pe = np.array(s['perch']); lp = np.array(s['lip']); r_ = BL['v0']['r']
    u = (lp - pe) / max(1e-6, np.linalg.norm(lp - pe)); L_ = np.linalg.norm(lp - pe)
    c_ = (pe + lp) / 2 - np.array([0, 0, r_ + 0.035])
    rp = box(tuple(c_ - u * 0.15), (L_ + 0.6, 0.5, 0.07), M_STONE, name='gramp'); rp.rotation_euler = (0, math.atan2(-u[2], u[0]), 0)
    box(tuple(pe - np.array([0.35, 0, r_ + 0.06])), (0.6, 0.6, 0.1), M_STONE, name='gledge')
    # the gate: an arch (two posts + lintel) and a flap that drops at the release
    for sy in (-1, 1): box(tuple(pe + np.array([0.05, sy * 0.32, 0.2])), (0.1, 0.08, 0.9), M_IRON, name='gpost')
    box(tuple(pe + np.array([0.05, 0, 0.68])), (0.14, 0.75, 0.12), M_GOLD, name='glintel', bevel=True)
    fl = box(tuple(pe + np.array([r_ + 0.05, 0, 0.02])), (0.04, 0.55, 0.5), M_IRON, name='gflap')
    f0 = int(s['gate'] * FPS); fl.keyframe_insert('location', frame=max(0, f0 - 1))
    fl.location = tuple(pe + np.array([r_ + 0.05, 0, -0.65])); fl.keyframe_insert('location', frame=f0 + 2)
    j = s['j']
    lt = sphere(tuple(pe + np.array([-0.2, 0, 0.95])), 0.07, M_LAMP, (10, 6)); lt.color = (*RCOL[j], 1)
    # name plate over the gate (English, readable)
# the start line: eight cups, a short steep ramp each, one long gate bar
ST = EV['start']; zc = ST['z']; xc = ST['x']
for j in range(8):
    y = j * LANE
    cp = obj(prim('cone', M_GOLD, 24, 0.34, 0.2, 0.16), 'cup'); cp.location = (xc, y, zc - BL['v0']['r'] - 0.02)
    u = np.array([math.cos(ST['alpha']), 0, -math.sin(ST['alpha'])]); r_ = BL['v0']['r']
    a0 = np.array([xc, y, zc - r_]); a1 = a0 + u * (ST['L'] + 0.12)
    rp = box(tuple((a0 + a1) / 2 - np.array([0, 0, 0.04])), (np.linalg.norm(a1 - a0) + 0.3, 0.5, 0.06), M_STONEW, name='sramp')
    rp.rotation_euler = (0, math.atan2(-u[2], u[0]), 0)
    rod((xc, y, zc - r_ - 0.1), (xc, y, zc - 3.0), 0.06, M_STONEW, 10)
    lt = sphere((xc - 0.42, y, zc - 0.05), 0.07, M_LAMP, (10, 6)); lt.color = (*RCOL[j], 1)
box((xc - 0.6, 3.5 * LANE, zc - 0.55), (1.2, 8 * LANE + 0.6, 0.3), M_STONEW, name='startledge')
gb = box((xc + BL['v0']['r'] + 0.06, 3.5 * LANE, zc + 0.05), (0.06, 8 * LANE + 0.4, 0.22), M_GOLD, name='startbar')
f0 = int(ST['t'] * FPS); gb.keyframe_insert('location', frame=f0 - 1)
gb.location = (xc + BL['v0']['r'] + 0.06, 3.5 * LANE, zc - 0.9); gb.keyframe_insert('location', frame=f0 + 3)
# checkered START banner on two posts behind the lanes
for k in range(16):
    for r in range(2):
        m_ = M_MARBLE if (k + r) % 2 == 0 else M_PLATE
        box((xc + 0.2, -0.6 + k * 0.42, zc + 2.3 + r * 0.42), (0.05, 0.42, 0.42), m_, name='chk')
for yy in (-0.8, 6.0): rod((xc + 0.2, yy, zc - 2.0), (xc + 0.2, yy, zc + 3.2), 0.06, M_IRON)

# ---------------- the storm bridge (act 2) and the golden bridge of Valhalla (act 3) ----------------
allc = sorted([(h['p'][0], h['p'][2], h['t']) for n in RIDN + ['bass'] for h in BL[n]['hits'] if h['t'] > T9 - 1])
AX = np.array([a[0] for a in allc]); AZ = np.array([a[1] for a in allc]); AT = np.array([a[2] for a in allc])
FLX = np.arange(xc + 0.5, AX.max() + 8, 0.25)
FLZ = np.array([AZ[np.abs(AX - x) < 1.6].min() if (np.abs(AX - x) < 1.6).any() else np.nan for x in FLX])
s_ = ~np.isnan(FLZ); FLZ = np.interp(FLX, FLX[s_], FLZ[s_]) - 1.7
for _ in range(3): FLZ = np.minimum(FLZ, np.convolve(np.pad(FLZ, 2, mode='edge'), np.ones(5) / 5, 'valid'))
FLZ = np.minimum.accumulate(FLZ)
XB = BR['x_hinge']
mS = FLX < XB - 0.1; mG = FLX > XB + 4.2
strip_mesh(FLX[mS], FLZ[mS], -2.6, 8 * LANE + 0.4, 0.5, M_STONE, 'stormbridge')
strip_mesh(FLX[mG], FLZ[mG], -2.6, 8 * LANE + 0.4, 0.5, M_MARBLE, 'goldbridge')
for yy in (-2.65, 8 * LANE + 0.45):
    strip_mesh(FLX[mS], FLZ[mS] + 0.12, yy - 0.06, yy + 0.06, 0.2, M_IRON, 'rail')
    strip_mesh(FLX[mG], FLZ[mG] + 0.12, yy - 0.06, yy + 0.06, 0.2, M_GOLD, 'rail')
# glowing lane lines on the storm bridge (blue) and the golden bridge (gold)
M_LINE_B = emis('lineb', (0.3, 0.6, 1.0), 3.0); M_LINE_G = emis('lineg', (1.0, 0.75, 0.35), 3.0)
for j in range(9):
    yy = (j - 0.5) * LANE
    strip_mesh(FLX[mS], FLZ[mS] + 0.004, yy - 0.02, yy + 0.02, 0.01, M_LINE_B, 'line')
    strip_mesh(FLX[mG], FLZ[mG] + 0.004, yy - 0.02, yy + 0.02, 0.01, M_LINE_G, 'line')
# support pillars under the bridges
for x0 in np.arange(FLX[0] + 1.0, FLX[-1], 4.0):
    z0 = float(np.interp(x0, FLX, FLZ))
    for yy in (-2.2, 8 * LANE):
        rod((x0, yy, z0 - 0.5), (x0, yy, z0 - 12.0), 0.35, M_STONE if x0 < XB else M_MARBLE, 12)
# the drawbridge: hinge at XB, falls from upright to flat between the lever and the landing
zb_ = float(np.interp(XB + 4.3, FLX, FLZ))
hinge = link(bpy.data.objects.new('hinge', None)); hinge.location = (XB, 3.5 * LANE - 0.6, zb_)
db = box((2.15, 0, -0.25), (4.3, 8 * LANE + 3.0, 0.5), M_WOOD, name='drawbridge'); db.parent = hinge
for k in range(6): box((0.4 + k * 0.7, 0, 0.01), (0.08, 8 * LANE + 3.0, 0.04), M_IRON, name='plankband').parent = hinge
for sy in (-1, 1):
    ch = rod((4.2, sy * (4 * LANE + 1.4), 0.0), (-0.2, sy * (4 * LANE + 1.4), 4.5), 0.04, M_IRON, 6); ch.parent = hinge
for f in range(NF):
    u = min(1.0, max(0.0, (T[f] - BR['t0']) / (BR['t1'] - BR['t0'])))
    ang = -(math.pi / 2) * (1 - u * u)                     # falls with growing speed
    if 0 < f < NF - 1 and (T[f] < BR['t0'] - 0.1 or T[f] > BR['t1'] + 0.1): continue
    hinge.rotation_euler = (0, ang, 0); hinge.keyframe_insert('rotation_euler', index=1, frame=f)
box((XB - 0.25, 3.5 * LANE - 0.6, zb_ - 0.2), (0.5, 8 * LANE + 3.4, 0.6), M_GOLD, name='hingebeam', bevel=True)

# ---------------- Valhalla: the golden castle and its great gate at the finish ----------------
XF, ZF = FIN['x'], FIN['z']
XG = XF + 1.3
zfl = float(np.interp(XF, FLX, FLZ))
for sy, yy in ((1, 8 * LANE + 1.4),):
    rod((XG, yy, zfl - 1), (XG, yy, zfl + 9), 1.1, M_STONEW, 24)
    o = obj(prim('cone', M_GOLD, 24, 1.35, 0.0, 3.0), 'roof'); o.location = (XG, yy, zfl + 10.5)
# the gate: two doors on hinges at its sides, a golden arch, two gears
GW = 8 * LANE + 3.6; GH = 6.0; yc = 3.5 * LANE - 0.6
box((XG + 0.15, yc, zfl + GH + 0.6), (0.9, GW + 1.6, 1.4), M_STONEW, name='lintel', bevel=True)
gl = box((XG + 0.6, yc, zfl + GH / 2), (0.05, GW, GH), M_GATEGLOW, name='gatelight')
TG = FINAL + 0.25
for sy in (-1, 1):
    hg = link(bpy.data.objects.new('doorhinge', None)); hg.location = (XG, yc + sy * GW / 2, zfl)
    d = box((0.0, -sy * GW / 4, GH / 2), (0.25, GW / 2, GH), M_GOLD, name='door', bevel=True); d.parent = hg
    for k in range(4): box((-0.14, -sy * GW / 4, 0.8 + k * 1.4), (0.05, GW / 2 - 0.2, 0.12), M_GOLDD, name='doorband').parent = hg
    hg.rotation_euler = (0, 0, 0); hg.keyframe_insert('rotation_euler', index=2, frame=int(TG * FPS))
    hg.rotation_euler = (0, 0, sy * math.radians(100)); hg.keyframe_insert('rotation_euler', index=2, frame=int((TG + 1.4) * FPS))
    gr = obj(prim('cyl', M_GOLDD, 24), 'gear'); gr.location = (XG - 0.4, yc + sy * (GW / 2 + 0.9), zfl + GH + 0.2)
    gr.rotation_euler = (math.pi / 2, 0, 0); gr.scale = (0.8, 0.8, 0.2)
    for k in range(12):
        ph = 2 * math.pi * k / 12; tooth = box((0.95 * math.cos(ph), 0.95 * math.sin(ph), 0), (0.25, 0.18, 0.9), M_GOLDD, name='tooth')
        tooth.parent = gr; tooth.rotation_euler = (0, 0, ph)
    gr.keyframe_insert('rotation_euler', index=2, frame=int(TG * FPS))
    gr.rotation_euler = (math.pi / 2, 0, sy * 4 * math.pi); gr.keyframe_insert('rotation_euler', index=2, frame=int((TG + 1.4) * FPS))
_ge = M_GATEGLOW.node_tree.nodes['Emission'].inputs['Strength']
_ge.default_value = 0.0; _ge.keyframe_insert('default_value', frame=int(TG * FPS))
_ge.default_value = 10.0; _ge.keyframe_insert('default_value', frame=int((TG + 1.2) * FPS))
# the castle behind (far, golden towers; seen from the leap on)
for k in range(9):
    xx = XF - 22 + k * 5.5 + rng.uniform(-1, 1); yy = 24 + rng.uniform(0, 8); hh = rng.uniform(9, 20)
    zb0 = zfl - 6
    rod((xx, yy, zb0), (xx, yy, zb0 + hh), rng.uniform(1.0, 1.8), M_STONEW, 16)
    o = obj(prim('cone', M_GOLD, 16, 2.2, 0.0, 4.0), 'roof'); o.location = (xx, yy, zb0 + hh + 2.0)
    box((xx + 2.7, yy, zb0 + hh * 0.45), (4.5, 1.2, hh * 0.9), M_STONEW, name='wall')

# ---------------- clouds (a sea of cloud under the bridges, storm clouds above) ----------------
for k in range(150):
    x_ = rng.uniform(-14, XG + 16); y_ = rng.uniform(-14, 30)
    zref = float(np.interp(x_, FLX, FLZ)) if x_ > FLX[0] else 0.0
    z_ = zref - rng.uniform(4.0, 8.0)
    o = sphere((x_, y_, z_), rng.uniform(1.5, 3.5), M_CLOUD, (12, 8), 'cloud'); o.scale = (o.scale[0] * 1.8, o.scale[1] * 1.3, o.scale[2] * 0.55)
for k in range(40):
    x_ = rng.uniform(-10, 14); y_ = rng.uniform(10, 26)
    o = sphere((x_, y_, rng.uniform(7, 12)), rng.uniform(2.0, 4.0), M_CLOUD, (12, 8), 'stormcloud'); o.scale = (o.scale[0] * 2, o.scale[1], o.scale[2] * 0.5)

# ---------------- lightning bolts (on the cloud drop and on every downbeat of the storm, bars 1-16) ----------------
BOLT_T = [EV['cloud_drop']['t']] + [h['t'] for h in B_['hits'] if h['t'] < LEAP['t0'] - 0.1 and
                                     abs(((h['t'] - E['t0']) / E['bar']) % 1) < 1e-3]
def bolt_mesh(p0, p1, seed):
    r_ = np.random.default_rng(seed); pts = [np.array(p0, float)]
    for k in range(1, 9):
        u = k / 9; q = np.array(p0) + (np.array(p1) - np.array(p0)) * u + np.array([r_.uniform(-0.8, 0.8), r_.uniform(-0.3, 0.3), 0])
        pts.append(q)
    pts.append(np.array(p1, float)); return pts
CAMX = None                                  # filled after the camera is known
BOLTS = []

# ---------------- hit rings ----------------
NRING = 16
ring_loc = np.zeros((NRING, NF, 3)); ring_scl = np.full((NRING, NF), 1e-3); ring_a = np.zeros((NRING, NF)); ring_c = [(1, 1, 1)] * NRING
for k, (t, p, c) in enumerate(sorted(RING_EV, key=lambda e: e[0])):
    r = k % NRING; f0 = int(round(t * FPS))
    for df in range(12):
        f = f0 + df
        if f >= NF: break
        u = df / 11; ring_loc[r, f] = p + np.array([0, -0.3, 0.02]); ring_scl[r, f] = 0.35 + 1.3 * (1 - (1 - u) ** 2)
        ring_a[r, f] = (1 - u) ** 1.6
    ring_c[r] = c
for r in range(NRING):
    bpy.ops.mesh.primitive_torus_add(major_radius=0.25, minor_radius=0.015, major_segments=32, minor_segments=5, rotation=(math.pi / 2, 0, 0))
    o = bpy.context.object; o.data.materials.append(M_FX); o.color = (*ring_c[r], 0)
    for f in range(1, NF):
        if ring_a[r, f] == 0: ring_loc[r, f] = ring_loc[r, f - 1]
    key_loc(o, ring_loc[r])
    for k in range(3): key_series(o, 'scale', k, ring_scl[r])
    key_series(o, 'color', 3, ring_a[r], 1e-3)

# ---------------- finish: sparks and confetti out of the open gate ----------------
M_SPARK = fxmat('spark', 12.0)
for s in range(120):
    d = np.array([rng.uniform(-1.0, -0.2), rng.uniform(-1, 1), rng.uniform(0.3, 1.2)]); d /= np.linalg.norm(d)
    sp_ = rng.uniform(3.0, 6.5); t0_ = TG + 0.6 + rng.uniform(0, 1.2)
    c0 = np.array([XG + 0.4, yc + rng.uniform(-GW / 3, GW / 3), zfl + rng.uniform(1, GH - 1)])
    o = sphere(tuple(c0), 0.07, M_SPARK, (6, 4), 'spark'); col = RCOL[s % 8]; o.color = (*col, 0)
    Pm = np.zeros((NF, 3)); A_ = np.zeros(NF); Sc = np.full(NF, 0.001)
    for f in range(NF):
        u = T[f] - t0_
        if u < 0: Pm[f] = c0; continue
        Pm[f] = c0 + d * sp_ * (1 - math.exp(-u * 1.2)) / 1.2 + np.array([0, 0, -0.9 * u * u])
        A_[f] = max(0.0, 1 - u / 2.6) ** 1.2; Sc[f] = 1.0 if u < 2.7 else 0.001
    key_loc(o, Pm); key_series(o, 'color', 3, A_, 1e-3)
    for k in range(3): key_series(o, 'scale', k, Sc * 0.07)

# ---------------- camera: the start rock in act 1, then follow the pack; wide for the leap and the finish ----------------
PR = np.stack([POS[n] for n in RIDN])                    # (8, NF, 3)
PACKX = PR[:, :, 0].mean(0); PACKZ = PR[:, :, 2].mean(0)
FOC = np.zeros((NF, 3)); WID = np.zeros(NF)
yl = 3.5 * LANE
for f in range(NF):
    t = T[f]
    a1 = np.array([0.4 + 0.06 * t, yl * 0.75, 1.4 - 0.02 * t]); w1 = 8.8
    a2 = np.array([PACKX[f] + 0.7, yl * 0.7, PACKZ[f] - 0.3]); w2 = 6.4
    u = min(1, max(0, (t - (T9 - 1.2)) / 2.4)); u = u * u * (3 - 2 * u)
    FOC[f] = (1 - u) * a1 + u * a2; WID[f] = (1 - u) * w1 + u * w2
    # the leap: pull back (the castle appears), then in again
    lw = min(1, max(0, (t - (LEAP['t0'] - 0.5)) / 1.2)) * (1 - min(1, max(0, (t - (T17 + 0.6)) / 1.8)))
    lw = lw * lw * (3 - 2 * lw); WID[f] += 4.5 * lw; FOC[f, 2] += 1.2 * lw
    fw = min(1, max(0, (t - (FINAL - 2.2)) / 2.0)); fw = fw * fw * (3 - 2 * fw)
    FOC[f] = (1 - fw) * FOC[f] + fw * np.array([XF + 0.2, yl * 0.7, ZF + 1.0]); WID[f] = (1 - fw) * WID[f] + fw * 9.5
FOC[:, 0] = smooth(FOC[:, 0], 0.5); FOC[:, 2] = smooth(FOC[:, 2], 0.7); WID = smooth(WID, 0.5)
LENS = 32.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 600
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24
    fw = min(1, max(0, (T[f] - (FINAL - 2.2)) / 2.0)); fw = fw * fw * (3 - 2 * fw)
    dv = (1 - fw) * np.array([0.0, -1.0]) + fw * np.array([-0.75, -0.66]); dv /= np.linalg.norm(dv)   # turn towards the gate
    CP[f] = (FOC[f, 0] + dv[0] * D, FOC[f, 1] + dv[1] * D, FOC[f, 2] + 0.36 * D)
    TP[f] = (FOC[f, 0], FOC[f, 1], FOC[f, 2] + 0.25)
key_loc(cam, CP); key_loc(tgt, TP)

# lightning bolts in the sky behind the course
for k, tb in enumerate(BOLT_T):
    f = int(round(tb * FPS)); f = min(f, NF - 1)
    if k == 0:
        cd = EV['cloud_drop']['p']; p0 = (cd[0] - 0.6, cd[1] + 0.5, cd[2] + 6.0); p1 = (cd[0], cd[1], cd[2] + 0.4)
    else:
        xb = FOC[f, 0] + rng.uniform(-4, 4); p0 = (xb, 18.0, FOC[f, 2] + 11); p1 = (xb + rng.uniform(-2, 2), 18.0, FOC[f, 2] - 4)
    pts = bolt_mesh(p0, p1, 100 + k)
    for a, b in zip(pts, pts[1:]):
        r_ = rod(a, b, 0.06 if k else 0.05, M_BOLT, 6)
        for df, hv in ((-1, True), (0, False), (4, True)):
            r_.hide_render = hv; r_.keyframe_insert('hide_render', frame=max(0, f + df))
    BOLTS.append(f)

# ---------------- signs (search keywords, English first, readable on a phone) ----------------
def sign(lines, x, y, z, w, h, plate=True, tilt=0.33):
    if plate:
        box((x, y + 0.08, z), (w, 0.12, h), M_PLATE, rot=(-tilt, 0, 0), name='plate')
        for sgn in (1, -1): box((x, y + 0.08 + math.sin(tilt) * sgn * h / 2, z + math.cos(tilt) * sgn * h / 2), (w + 0.1, 0.16, 0.07), M_GOLD, rot=(-tilt, 0, 0))
    zz = z + h / 2
    for s_, size, mt in lines:
        zz -= size * 1.25
        fit_w(text(s_, (x, y - 0.03 + math.tan(tilt) * (zz + size * 0.15 - z), zz + size * 0.15), size, mt, rot=(math.pi / 2 - tilt, 0, 0)),
              (w - 0.3) if plate else None)
sign([('RIDE OF THE VALKYRIES', 0.5, M_TXT_G), ('Richard Wagner · Die Walküre', 0.34, M_TXT), ('Classical music × Marble race', 0.3, M_TXT_B),
      ('ワーグナー「ワルキューレの騎行」 ビー玉レース', 0.22, M_TXT)], 1.0, -3.2, -2.9, 7.4, 2.5)
def course_z(x): return float(np.interp(x, FLX, FLZ))
for x_, ls in ((PACKX[int((T9 + 4.5) * FPS)] + 2.5, [('8 VALKYRIES', 0.36, M_TXT_G), ('Epic classical · marble run', 0.24, M_TXT), ('8人のワルキューレ 大レース', 0.17, M_TXT)]),
               (PACKX[int((T9 + 10.5) * FPS)] + 2.5, [('WHO WINS?', 0.42, M_TXT_R), ('Comment your rider!', 0.26, M_TXT), ('誰が勝つ？ コメントで予想', 0.17, M_TXT)])):
    sign(ls, x_, -3.6, course_z(x_) + 0.9, 3.8, 1.35)
x_v = XG - 1.0
sign([('VALHALLA', 0.6, M_TXT_G), ('ヴァルハラ城', 0.24, M_TXT)], XG - 0.3, -0.9, zfl + GH + 1.9, 0.1, 0.9, plate=False, tilt=0.25)

# ---------------- HUD (rides with the camera, near the top of the frame) ----------------
hud = link(bpy.data.objects.new('hud', None)); hud.parent = cam
HZ = 6.0; hh_ = HZ * 24 / LENS; ww_ = hh_ * 9 / 16
hud.location = (0, hh_ * 0.355, -HZ)
pl_ = box((0, 0, -0.02), (ww_ * 0.9, hh_ * 0.105, 0.02), M_PLATE, name='hudplate'); pl_.parent = hud
for sgn in (1, -1):
    gb_ = box((0, sgn * hh_ * 0.0525, -0.01), (ww_ * 0.92, hh_ * 0.006, 0.02), M_GOLD, name='hudgold'); gb_.parent = hud
TS = hh_ * 0.032
def hud_text(s, y, size, m, vis):
    o = fit_w(text(s, (0, y, 0), size, m, rot=(0, 0, 0)), ww_ * 0.84); o.parent = hud
    s0 = o.scale[0]; v = np.where(vis, s0, 0.0005)
    for k in range(3): key_series(o, 'scale', k, v, 1e-6)
    return o
t_hook1 = T9 - 0.2
hud_text('8 RIDERS – WHO ARRIVES FIRST?', TS * 0.45, TS * 1.0, M_TXT_G, T < t_hook1)
hud_text('8人のワルキューレ 誰が一番に着く？', -TS * 0.85, TS * 0.42, M_TXT, T < t_hook1)
# act 1: riders on the start line k/8
cnt = EV['count']; ct_ = np.array([c[0] for c in cnt]); cv_ = np.array([c[1] for c in cnt])
CV = np.array([cv_[max(0, np.searchsorted(ct_, t, 'right') - 1)] for t in T])
for k in range(9):
    hud_text(f'ON THE START LINE: {k}/8', -TS * 1.75, TS * 0.55, M_TXT_B, (CV == k) & (T < t_hook1))
# the race: the leader (largest x, with a little hysteresis)
LEADER = np.full(NF, -1)
cur = -1
for f in range(NF):
    t = T[f]
    if t < T9 + 0.6 or t >= FINAL - 0.02: continue
    xs = PR[:, f, 0]; best = int(np.argmax(xs))
    if cur < 0 or xs[best] > xs[cur] + 0.08: cur = best
    LEADER[f] = cur
for j in range(8):
    hud_text(f'LEADER: #{j + 1} {NAMES[j]}', TS * 0.45, TS * 0.95, M_TXT_G, LEADER == j)
racing = (LEADER >= 0)
hud_text('先頭のワルキューレ', -TS * 0.85, TS * 0.42, M_TXT, racing)
NCH = int(sum(1 for f in range(1, NF) if LEADER[f] != LEADER[f - 1] and LEADER[f] >= 0 and LEADER[f - 1] >= 0))
fin_v = T >= FINAL - 0.02
hud_text('PHOTO FINISH – ALL 8!', TS * 0.45, TS * 1.05, M_TXT_G, fin_v)
hud_text('全員同着！ 8人そろってヴァルハラへ', -TS * 0.85, TS * 0.42, M_TXT, fin_v)
# a finish line across the lanes just before the bells
box((XF, yl, ZF - BL['v0']['r'] - 0.62), (0.12, 8 * LANE + 0.6, 0.03), M_MARBLE, name='finishline')
for k in range(20):
    box((XF - 0.12, -0.3 + k * (8 * LANE + 0.6) / 20 - 0.01, ZF - BL['v0']['r'] - 0.6), (0.12, (8 * LANE + 0.6) / 20, 0.035),
        M_PLATE if k % 2 else M_MARBLE, name='chk2')

# ---------------- lights ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bgn = w.node_tree.nodes['Background']
for f in (0, int((LEAP['t0'] - 0.2) * FPS), int((T17 + 0.8) * FPS)):
    u = DAWN[min(f, NF - 1)]
    bgn.inputs['Color'].default_value = tuple(np.array([0.05, 0.07, 0.16, 1]) * (1 - u) + np.array([0.3, 0.17, 0.12, 1]) * u)
    bgn.inputs['Color'].keyframe_insert('default_value', frame=f)
bgn.inputs['Strength'].default_value = 0.9
key_ = bpy.data.lights.new('key', 'SUN'); key_.angle = 0.15
ko = link(bpy.data.objects.new('key', key_)); ko.rotation_euler = (math.radians(50), math.radians(-15), math.radians(-25))
for f in (0, int((LEAP['t0'] - 0.2) * FPS), int((T17 + 0.8) * FPS)):
    u = DAWN[min(f, NF - 1)]
    key_.energy = 2.6 * (1 - u) + 3.2 * u; key_.keyframe_insert('energy', frame=f)
    key_.color = tuple(np.array([0.6, 0.72, 1.0]) * (1 - u) + np.array([1.0, 0.8, 0.55]) * u); key_.keyframe_insert('color', frame=f)
fl_ = bpy.data.lights.new('flash', 'SUN'); fl_.energy = 0.0; fl_.color = (0.75, 0.85, 1.0); fl_.angle = 0.3
flo = link(bpy.data.objects.new('flash', fl_)); flo.rotation_euler = (math.radians(30), math.radians(25), math.radians(160))
for f in BOLTS:
    for df, e_ in ((-1, 0.0), (0, 9.0), (2, 2.5), (5, 0.0)):
        fl_.energy = e_; fl_.keyframe_insert('energy', frame=max(0, f + df))
fil = bpy.data.lights.new('fill', 'SUN'); fil.energy = 0.6; fil.color = (0.55, 0.65, 1.0)
fo = link(bpy.data.objects.new('fill', fil)); fo.rotation_euler = (math.radians(70), math.radians(30), math.radians(160))
spd = bpy.data.lights.new('spot', 'SPOT'); spd.energy = 2600; spd.spot_size = math.radians(30); spd.spot_blend = 0.6
spd.color = (1.0, 0.92, 0.8); spd.shadow_soft_size = 0.4
so_ = link(bpy.data.objects.new('spot', spd)); stg = link(bpy.data.objects.new('spottgt', None))
key_loc(stg, FOC - np.array([0, 0, 0.4])); key_loc(so_, FOC + np.array([-2.0, -8.0, 10.0]))
c_ = so_.constraints.new('TRACK_TO'); c_.target = stg; c_.track_axis = 'TRACK_NEGATIVE_Z'; c_.up_axis = 'UP_Y'
ga = bpy.data.lights.new('gatelight', 'AREA'); ga.size = 6.0; ga.color = (1.0, 0.85, 0.6); ga.energy = 0.0
gao = link(bpy.data.objects.new('gatelight', ga)); gao.location = (XG + 1.2, yc, zfl + GH / 2); gao.rotation_euler = (0, math.radians(-90), 0)
ga.keyframe_insert('energy', frame=int(TG * FPS)); ga.energy = 2500.0; ga.keyframe_insert('energy', frame=int((TG + 1.2) * FPS))

if MODE == 'test':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS)); sc.frame_set(f)
        sc.render.filepath = f'{HERE}/test_{f:04d}.png'
        t1_ = time.time(); bpy.ops.render.render(write_still=True); print('frame', f, 'sec', round(time.time() - t1_, 1))
elif MODE == 'range':
    sc.frame_start, sc.frame_end = int(args[1]), int(args[2])
    sc.render.image_settings.file_format = 'PNG'; sc.render.filepath = f'{HERE}/frames/f_'
    bpy.ops.render.render(animation=True)
elif MODE == 'probe':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS)); sc.frame_set(f)
        print('t', tt_, 'cam', [round(v, 2) for v in cam.matrix_world.translation], 'WID', round(WID[f], 2), 'FOC', np.round(FOC[f], 2))
print('scene objects', len(bpy.data.objects), 'leader changes', NCH)
