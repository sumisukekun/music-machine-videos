"""Ode to Joy - 年末の大合唱 (the year-end choir), side view of a great concert hall on New Year's Eve.
Every melody note = the choir balls landing on a key of that pitch (one long key for all the lanes that sing), every bass note =
the bronze ball on a gold bell, every chord on beats 2 and 4 = the pair on two small bells, and in the last 8 bars every chord =
the rain of balls on the nine great bells. layout.py plans all motion; this file builds and animates it.
jp_sub.otf = Noto Sans CJK JP Black (SIL OFL) subset to the characters used here (make_font.py).
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
FPS = 30; FINAL = E['final']; NF = int((FINAL + 5.5) * FPS); T = np.arange(NF) / FPS
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
BL = LAY['balls']
POS = {n: at_many(b, T) for n, b in BL.items()}
MELN = [n for n, b in BL.items() if b['role'] == 'melody']
RAINN = [n for n, b in BL.items() if b['role'] == 'rain']

# ---------------- materials ----------------
M_GOLD = mat('gold', (1.0, 0.7, 0.25), 0.22, **{'Metallic': 0.95})
M_GOLDD = mat('goldd', (0.6, 0.38, 0.12), 0.35, **{'Metallic': 0.9})
M_SOLO = mat('solo', (1.0, 0.78, 0.3), 0.08, **{'Metallic': 0.85, 'Coat Weight': 1.0, 'Emission Color': (1.0, 0.7, 0.25), 'Emission Strength': 0.5})
M_BRONZE = mat('bronze', (0.95, 0.5, 0.25), 0.18, **{'Metallic': 0.85, 'Coat Weight': 0.6, 'Emission Color': (1.0, 0.45, 0.2), 'Emission Strength': 0.35})
M_SILVER = mat('silver', (0.85, 0.9, 1.0), 0.12, **{'Metallic': 0.7, 'Coat Weight': 0.8, 'Emission Color': (0.7, 0.85, 1.0), 'Emission Strength': 0.4})
M_CHOIR = mat('choir', (1, 1, 1), 0.1, **{'Coat Weight': 1.0, 'Emission Strength': 0.45})
_nt = M_CHOIR.node_tree; _oi = _nt.nodes.new('ShaderNodeObjectInfo')
_nt.links.new(_oi.outputs['Color'], _nt.nodes['Principled BSDF'].inputs['Base Color']); _nt.links.new(_oi.outputs['Color'], _nt.nodes['Principled BSDF'].inputs['Emission Color'])
M_RAIN = glowmat('rain', gain=7.0, base=0.35, rough=0.12, coat=1.0)
M_KEY = glowmat('key', gain=9.0, base=0.35, rough=0.12, coat=1.0)
M_BELL = glowmat('bell', gain=4.0, base=0.0, rough=0.22, metal=0.9)
M_TINK = glowmat('tink', gain=8.0, base=0.3, rough=0.12, metal=0.4, coat=1.0)
M_GBELL = glowmat('gbell', gain=3.0, base=0.05, rough=0.2, metal=0.95)
M_WOOD = mat('wood', (0.16, 0.07, 0.03), 0.22, **{'Coat Weight': 1.0})
M_WOODL = mat('woodl', (0.42, 0.2, 0.08), 0.35, **{'Coat Weight': 0.6})
M_VELVET = mat('velvet', (0.42, 0.015, 0.03), 0.85, **{'Sheen Weight': 1.0, 'Sheen Tint': (1.0, 0.4, 0.4)})
M_CURTAIN = mat('curtain', (0.55, 0.02, 0.04), 0.8, **{'Sheen Weight': 1.0, 'Sheen Tint': (1.0, 0.5, 0.5)})
M_WALL = mat('wall', (0.13, 0.05, 0.05), 0.7)
M_DARK = mat('dark', (0.03, 0.025, 0.03), 0.6)
M_PLATE = mat('plate', (0.05, 0.02, 0.03), 0.3, **{'Coat Weight': 1.0})
M_CREAM = mat('cream', (0.9, 0.82, 0.68), 0.5)
M_SEAT = mat('seat', (0.55, 0.03, 0.05), 0.6)
M_WIN = emis('window', (0.05, 0.1, 0.28), 1.2)
M_LAMP = objcol('lamp', 0.4, emit=6.0)
M_FX = fxmat('fx', 8.0)
M_TXT = emis('txtw', (1.0, 0.95, 0.85), 4.0); M_TXT_G = emis('txtg', (1.0, 0.75, 0.3), 6.0); M_TXT_R = emis('txtr', (1.0, 0.35, 0.3), 5.0)
M_CANDLE = emis('candle', (1.0, 0.75, 0.4), 18.0)
PITCH_COL = [(1.0, 0.12, 0.12), (1.0, 0.35, 0.05), (1.0, 0.6, 0.05), (1.0, 0.85, 0.1), (0.55, 1.0, 0.1), (0.1, 1.0, 0.3),
             (0.05, 1.0, 0.75), (0.05, 0.7, 1.0), (0.15, 0.35, 1.0), (0.45, 0.2, 1.0), (0.75, 0.2, 1.0), (1.0, 0.2, 0.65)]
def pcol(p, light=0.0):
    c = PITCH_COL[p % 12]; return tuple(min(1, x * (1 - light) + light) for x in c)
CHOIR_COL = [(1.0, 0.25, 0.3), (0.3, 0.6, 1.0), (1.0, 0.95, 0.9), (0.4, 1.0, 0.55), (1.0, 0.55, 0.15), (0.75, 0.4, 1.0)]

# ---------------- balls ----------------
BALL = {}
for n, b in BL.items():
    rr = b['r']; role = b['role']
    seg = (16, 8) if role == 'rain' else (32, 16)
    o = obj(prim('sph', None, *seg), n); o.scale = (rr, rr, rr)
    if n == 'solo': o.data = prim('sph', M_SOLO, *seg)
    elif role == 'melody':
        o.data = prim('sph', M_CHOIR, *seg); j = int(n[1:]); o.color = (*CHOIR_COL[j % len(CHOIR_COL)], 1)
    elif role == 'bass': o.data = prim('sph', M_BRONZE, *seg)
    elif role == 'pair': o.data = prim('sph', M_SILVER, *seg)
    else: o.data = prim('sph', M_RAIN, *seg)
    P = POS[n]
    if role == 'rain':
        # hidden inside the organ pipe until it drops out; colour of its bell while flying, warm gold once seated
        r_ = EV['rain'][int(n[1:])]; h = b['hits'][0]; c0 = pcol(h['pitch'], 0.15)
        f_out = int(math.floor(r_['t_out'] * FPS)); f_seat = int(math.ceil(r_['t_seat'] * FPS))
        sc_ = np.where(T >= r_['t_out'] - 1e-6, rr, 0.0005)
        for k in range(3): key_series(o, 'scale', k, sc_)
        for k in range(3): key_series(o, 'location', k, P[:, k])
        o.color = (*c0, 0.0); o.keyframe_insert('color', frame=0)
        fh = int(round(h['t'] * FPS))
        o.color = (*c0, 1.0); o.keyframe_insert('color', frame=fh)
        o.color = (*c0, 0.15); o.keyframe_insert('color', frame=fh + 8)
        o.color = (*c0, 0.15); o.keyframe_insert('color', frame=f_seat)
        o.color = (1.0, 0.82, 0.4, 1.0); o.keyframe_insert('color', frame=f_seat + 2)
        o.color = (1.0, 0.82, 0.4, 0.45); o.keyframe_insert('color', frame=f_seat + 12)
        fe = int((FINAL + 0.05) * FPS)
        if fe > f_seat + 12:
            o.color = (1.0, 0.82, 0.4, 0.45); o.keyframe_insert('color', frame=fe)
        o.color = (1.0, 0.86, 0.5, 1.4); o.keyframe_insert('color', frame=max(fe + 3, f_seat + 13))
    else:
        key_loc(o, P)
        sp = np.zeros(NF)
        for f in range(1, NF): sp[f] = sp[f - 1] + (P[f, 0] - P[f - 1, 0]) / rr
        o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, sp)
        # a thin band so the spin reads
        bd = obj(bpy.data.meshes.new('band'), 'band')
        bm = bmesh.new(); bmesh.ops.create_cone(bm, cap_ends=False, segments=32, radius1=1.005, radius2=1.005, depth=0.14); bm.to_mesh(bd.data); bm.free()
        bd.data.materials.append(M_GOLD if role != 'pair' else M_GOLDD); bd.parent = o; bd.rotation_euler = (math.pi / 2, 0, 0)
    BALL[n] = o

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


# ---------------- the stage floor (terraces below every contact) ----------------
allc = sorted([(h['p'][0], h['p'][2]) for n, b in BL.items() if b['role'] != 'rain' for h in b['hits']])
AX = np.array([a[0] for a in allc]); AZ = np.array([a[1] for a in allc])
FLX = np.arange(-12.0, AX.max() + 26, 0.5)
FLZ = np.array([AZ[np.abs(AX - x) < 3.0].min() if (np.abs(AX - x) < 3.0).any() else np.nan for x in FLX])
s_ = ~np.isnan(FLZ); FLZ = np.interp(FLX, FLX[s_], FLZ[s_]) - 1.0
for _ in range(4): FLZ = np.minimum(FLZ, np.convolve(np.pad(FLZ, 2, mode='edge'), np.ones(5) / 5, 'valid'))
FLZ = np.minimum.accumulate(FLZ)
def floor_z_at(x): return float(np.interp(x, FLX, FLZ))
strip_mesh(FLX, FLZ, -7.0, 22.0, 0.4, M_WOOD, 'stage')
# gold inlay lines on the stage (perspective cue) and footlights along the front edge
for yy in (-4.5, -0.9, 7.0):
    strip_mesh(FLX, FLZ + 0.004, yy - 0.03, yy + 0.03, 0.01, M_GOLD, 'inlay')
for x0 in np.arange(-6.0, FLX[-1], 2.2):
    o = sphere((x0, -6.8, floor_z_at(x0) + 0.12), 0.1, M_LAMP, (12, 8)); o.color = (1.0, 0.72, 0.4, 1)

# ---------------- melody keys: one per note, long enough for every lane that sings it ----------------
KEY_TH = 0.12
for h in BL['solo']['hits']:
    nv = np.array(h['n']); top = np.array(h['p']) - BL['solo']['r'] * nv
    lanes = h['lanes']; depth = (lanes - 1) * LN + 0.5
    ctr = top - nv * KEY_TH / 2 + np.array([0, (lanes - 1) * LN / 2, 0])
    o = obj(prim('bcube', M_KEY), 'key')
    o.matrix_world = Matrix.Translation(Vector(ctr)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((0.42, depth, KEY_TH, 1))
    col = pcol(h['pitch'], 0.0); o.color = (*col, 0)
    flash(o, h['t'], col, nv)
    objs = [o]
    for yy in sorted(set([0.0, (lanes - 1) * LN])):
        objs.append(rod((top[0], yy, top[2] - KEY_TH - 0.02), (top[0], yy, floor_z_at(top[0]) - 0.3), 0.03, M_GOLDD, 8))
    if h['kind'] == 'lever':
        # the lever key: a gold rod from its front end runs to the bass ball's gate
        L_ = EV['bass_gate']; g = np.array(L_['ledge'])
        objs.append(rod((top[0], -0.35, top[2] - 0.06), (top[0], -0.35, top[2] + 0.35), 0.035, M_GOLD))
        objs.append(rod((top[0], -0.35, top[2] + 0.35), (g[0] - 0.25, g[1], top[2] + 0.35), 0.03, M_GOLD))
        objs.append(rod((g[0] - 0.25, g[1], top[2] + 0.35), (g[0] - 0.25, g[1], g[2] + 0.25), 0.03, M_GOLD))
    if h['kind'] == 'final' or h['t'] < 1.6: continue
    riser(objs, h['t'], top, lead=1.2, stay=0.9)

# ---------------- bass bells (gold, mouth down) and pair bells (small round bells) ----------------
for n in ('bass', 'pairlo', 'pairhi'):
    b = BL[n]
    for h in b['hits']:
        if h['kind'] == 'seesaw': continue
        nv = np.array(h['n']) if np.linalg.norm(h['vout']) > 0 else np.array([0, 0, 1.0])
        top = np.array(h['p']) - b['r'] * nv; p = h['pitch']
        if n == 'bass':
            sz = 0.32 + 0.012 * (38 - p)
            o = obj(prim('bell', M_BELL, 32), 'bell')
            o.matrix_world = Matrix.Translation(Vector(top)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((sz * 0.75, sz * 0.75, sz, 1))
            col = (1.0, 0.7, 0.3); bot = top - nv * sz
        else:
            rs = 0.13 + 0.004 * (62 - p); col = pcol(p, 0.15)
            cs = top - nv * rs
            o = obj(prim('sph', M_TINK, 24, 12), 'suzu'); o.location = tuple(cs); o.scale = (rs, rs, rs)
            bd = obj(prim('cyl', M_GOLD, 24), 'band'); bd.parent = o; bd.scale = (1.01, 1.01, 0.12)
            bot = top - nv * 2 * rs
        o.color = (*col, 0)
        flash(o, h['t'], col, nv)
        po = rod(bot, (bot[0], bot[1], floor_z_at(bot[0]) - 0.2), 0.035 if n == 'bass' else 0.022, M_GOLDD, 8)
        if not h.get('final'): riser([o, po], h['t'], top, lead=1.2, stay=0.75)

# ---------------- the podium (start) ----------------
pd = EV['podium']; zf = floor_z_at(0.0)
box((0.0, 0.0, (pd['z'] - 0.2 + zf) / 2), (0.7, 0.7, pd['z'] - 0.2 - zf), M_VELVET, name='podium')
box((0.0, 0.0, pd['z'] - 0.2 - 0.04), (0.9, 0.9, 0.08), M_GOLD, name='podtop', bevel=True)
# the trapdoor of the podium top: the ball sits on it and drops through when it opens
for sgn in (-1, 1):
    fl = box((0.0, 0.0, pd['z'] - 0.2 + 0.0), (0.42, 0.42, 0.03), M_GOLDD, name='trap')
    fl.location = (sgn * 0.21, 0, pd['z'] - 0.21)
    fl.scale = (0.4, 0.42, 0.03)
    tr = pd['t_release']
    fl.rotation_euler = (0, 0, 0); fl.keyframe_insert('rotation_euler', index=1, frame=int(tr * FPS))
    fl.rotation_euler = (0, sgn * 1.4, 0); fl.keyframe_insert('rotation_euler', index=1, frame=int(tr * FPS) + 4)

# ---------------- the bass ball's ledge, gate and ramp ----------------
bg = EV['bass_gate']; led = np.array(bg['ledge']); lip = np.array(bg['lip']); RB_ = BL['bass']['r']
uB = (lip - led) / max(1e-6, np.linalg.norm(lip - led))
surf = lambda p: p - np.array([0, 0, RB_])
L_r = np.linalg.norm(lip - led)
c_ = (surf(led) + surf(lip)) / 2 - np.array([0, 0, 0.04])
rp = box(tuple(c_), (L_r + 0.5, 0.5, 0.08), M_WOODL, name='bramp')
rp.rotation_euler = (0, math.atan2(-(lip - led)[2], (lip - led)[0]) * 1.0, 0)
box(tuple(surf(led) - np.array([0.3, 0, 0.06])), (0.6, 0.6, 0.1), M_WOODL, name='bledge')
rod(surf(led) - np.array([0.3, 0, 0.1]), (led[0] - 0.3, led[1], floor_z_at(led[0]) - 0.2), 0.05, M_GOLDD)
rod(surf(lip) - np.array([0, 0, 0.1]), (lip[0], lip[1], floor_z_at(lip[0]) - 0.2), 0.05, M_GOLDD)
gate = box(tuple(led + np.array([RB_ + 0.05, 0, 0])), (0.04, 0.55, 2 * RB_ + 0.1), M_GOLD, name='bgate')
f0 = int(bg['t'] * FPS)
gate.keyframe_insert('location', frame=f0)
gate.location = tuple(led + np.array([RB_ + 0.05, 0, 2 * RB_ + 0.25])); gate.keyframe_insert('location', frame=f0 + 3)
# side rails
for sgn in (-1, 1):
    rod(surf(led) + np.array([-0.3, sgn * 0.3, 0.12]), surf(lip) + np.array([0, sgn * 0.3, 0.12]), 0.02, M_GOLD)

# ---------------- the see-saw (pair catapult) ----------------
ss = EV['seesaw']; piv = np.array(ss['pivot']); U = np.array(ss['u']); ARM = ss['arm']; TH0 = ss['th0']
AXS = np.array([U[1] * -1, U[0], 0.0]) * -1          # horizontal axis perpendicular to U
AXS = np.array([math.sin(ss['phi']), math.cos(ss['phi']), 0.0])
plank_e = link(bpy.data.objects.new('plank_pivot', None)); plank_e.location = tuple(piv)
yaw = math.atan2(U[1], U[0])
plank_e.rotation_mode = 'ZYX'
pl = box((0, 0, -0.03), (2 * ARM + 0.3, 0.26, 0.06), M_WOODL, name='plank', bevel=True); pl.parent = plank_e
bell_ss = obj(prim('bell', M_BELL, 32), 'ssbell'); bell_ss.parent = plank_e
szs = ss['bell_h']; bell_ss.location = (ARM, 0, szs); bell_ss.scale = (szs * 0.75, szs * 0.75, szs)
cupo = obj(prim('cone', M_GOLD, 24, 0.33, 0.2, 0.14), 'cup'); cupo.parent = plank_e; cupo.location = (-ARM, 0, 0.06)
rod(piv - np.array([0, 0, 0.0]), (piv[0], piv[1], floor_z_at(piv[0]) - 0.2), 0.08, M_GOLDD)
axl = rod(piv - AXS * 0.2, piv + AXS * 0.2, 0.05, M_GOLD)
th = np.array([ss['th0'] if t < ss['t'] else (ss['th0'] - 2 * ss['th0'] * min(1, ((t - ss['t']) / ss['t_swing'])) ** 2) for t in T])
plank_e.rotation_euler = (0, 0, yaw)
for f in range(NF):
    if 0 < f < NF - 1 and abs(th[f] - th[f - 1]) < 1e-6 and abs(th[f] - th[f + 1]) < 1e-6: continue
    plank_e.rotation_euler = (0, -th[f], yaw); plank_e.keyframe_insert('rotation_euler', frame=f)
hS = [h for h in BL['bass']['hits'] if h['kind'] == 'seesaw'][0]
bell_ss.color = (1.0, 0.7, 0.3, 0); flash(bell_ss, hS['t'], (1.0, 0.7, 0.3), (0, 0, 1), dip=0.0)

# ---------------- the joiners' balconies: perch, gate flap, ramp ----------------
for J in EV['join']:
    pe = np.array(J['perch']); lp = np.array(J['lip']); r_ = BL['solo']['r']
    u = (lp - pe) / max(1e-6, np.linalg.norm(lp - pe))
    L_ = np.linalg.norm(lp - pe)
    c_ = (pe + lp) / 2 - np.array([0, 0, r_ + 0.035])
    rp = box(tuple(c_ - u * 0.1), (L_ + 0.55, 0.42, 0.06), M_WOODL, name='jramp')
    rp.rotation_euler = (0, math.atan2(-u[2], u[0]), 0)
    st = box(tuple(pe - np.array([0.28, 0, r_ + 0.05])), (0.4, 0.42, 0.08), M_VELVET, name='jseat')
    rod(pe - np.array([0.28, 0, r_ + 0.1]), (pe[0] - 0.28, pe[1], floor_z_at(pe[0]) - 0.2), 0.03, M_GOLDD)
    for sgn in (-1, 1): rod(pe + np.array([-0.45, sgn * 0.22, -r_ + 0.15]), lp + np.array([0, sgn * 0.22, -r_ + 0.15]), 0.015, M_GOLD)
    fl = box(tuple(pe + np.array([r_ + 0.03, 0, -r_ * 0.2])), (0.03, 0.4, 0.3), M_GOLD, name='jflap')
    f0 = int(J['t_rel'] * FPS)
    fl.keyframe_insert('location', frame=f0 - 1)
    fl.location = tuple(pe + np.array([r_ + 0.03, 0, -r_ - 0.35])); fl.keyframe_insert('location', frame=f0 + 2)
    lampo = sphere(tuple(pe + np.array([-0.5, 0, 0.35])), 0.06, M_LAMP, (10, 6)); lampo.color = (1.0, 0.8, 0.5, 1)
# balcony fronts (velvet + gold rail) under each group of perches
for grp in (EV['join'][:4], EV['join'][4:]):
    xs = [J['perch'][0] for J in grp]; ys = [J['perch'][1] for J in grp]; zs = [J['perch'][2] for J in grp]
    x0, x1 = min(xs) - 0.9, max(xs) + 0.4; y1 = max(ys) + 0.6; zb = min(zs) - 0.55
    box(((x0 + x1) / 2, y1 + 0.1, zb + 0.3), (x1 - x0, 0.12, 0.9), M_VELVET, name='balc')
    rod((x0, y1 + 0.05, zb + 0.78), (x1, y1 + 0.05, zb + 0.78), 0.03, M_GOLD)

# ---------------- the riser, its 987 seats, the curtain, the bells and the organ ----------------
ri = EV['riser']; XC, YR, ZR = ri['xc'], ri['yr'], ri['zr']; NROW, NCOL = ri['nrow'], ri['ncol']
PXs, RISE, DEPTH = ri['px'], ri['rise'], ri['depth']
W_ = NCOL * PXs + 0.6
for k in range(NROW):
    zt = ZR + k * RISE
    box((XC, YR + k * DEPTH + DEPTH / 2 + 4.0, (zt + floor_z_at(XC) - 1) / 2), (W_, DEPTH + 8.0 if False else DEPTH, zt - floor_z_at(XC) + 1), M_WOOD, name='tier')
    box((XC, YR + k * DEPTH + 0.01, zt - 0.06), (W_, 0.02, 0.12), M_VELVET, name='tierfront')
    box((XC, YR + k * DEPTH + 0.025, zt - 0.005), (W_, 0.05, 0.012), M_GOLD, name='tieredge')
# fix: tiers as stacked boxes from the floor
for o in [o for o in bpy.data.objects if o.name.startswith('tier') and not o.name.startswith(('tierfront', 'tieredge'))]:
    k = int(round((o.location.y - 4.0 - YR - DEPTH / 2) / DEPTH)); zt = ZR + k * RISE; zb = floor_z_at(XC) - 1.0
    o.location = (XC, YR + k * DEPTH + DEPTH / 2, (zt + zb) / 2); o.scale = (W_, DEPTH, zt - zb)
# seat rings (one mesh)
bm = bmesh.new()
for r_ in range(NROW):
    for c_ in range(NCOL):
        k = NROW - 1 - r_
        x, y, z = ri['x_col'][c_], YR + k * DEPTH + DEPTH / 2, ZR + k * RISE
        bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=0.15, radius2=0.12, depth=0.03,
                              matrix=Matrix.Translation((x, y, z + 0.012)))
me = bpy.data.meshes.new('seats'); bm.to_mesh(me); bm.free(); so = link(bpy.data.objects.new('seats', me)); so.data.materials.append(M_SEAT)
# the side walls of the riser
for sx in (-1, 1):
    xw = XC + sx * (W_ / 2 + 0.15)
    box((xw, YR + NROW * DEPTH / 2, (ZR + NROW * RISE + floor_z_at(XC) - 1) / 2), (0.3, NROW * DEPTH, NROW * RISE + 1 - (ZR - floor_z_at(XC))), M_GOLDD, name='rwall')
bel = EV['bells']; RT = bel['r']; LT = bel['length']
for k, (x, p) in enumerate(zip(bel['x'], bel['pitches'])):
    # a golden tubular bell lying across x, hung from the organ by two cords
    o = obj(prim('cyl', M_GBELL, 32), 'tube'); o.location = (x, bel['y'], bel['z']); o.rotation_euler = (0, math.pi / 2, 0)
    o.scale = (RT, RT, LT)
    for sx in (-1, 1):
        cap = obj(prim('cyl', M_GOLDD, 32), 'tcap'); cap.location = (x + sx * LT / 2, bel['y'], bel['z']); cap.rotation_euler = (0, math.pi / 2, 0)
        cap.scale = (RT * 1.12, RT * 1.12, 0.05)
        rod((x + sx * LT * 0.38, bel['y'], bel['z'] + RT), (x + sx * LT * 0.38, bel['y'], bel['z'] + RT + bel['h_pipe'] - 0.1), 0.012, M_GOLDD, 6)
    col = pcol(p, 0.2); o.color = (*col, 0)
    ts_ = sorted(set(round(t, 4) for (t, pp), h in zip(LAY['bellhits'], [None] * len(LAY['bellhits'])) if pp == p))
    ts_ = sorted(set(round(BL[n]['hits'][0]['t'], 4) for n in RAINN if BL[n]['hits'][0].get('tube') == k))
    for t in ts_:
        fh = int(round(t * FPS))
        o.color = (*col, 0.0); o.keyframe_insert('color', index=3, frame=max(0, fh - 1))
        o.color = (*col, 1.0); o.keyframe_insert('color', index=3, frame=fh)
        o.color = (*col, 0.15); o.keyframe_insert('color', index=3, frame=fh + 10)
    # the organ's slot mouth above the tube
    mz = bel['z'] + RT + bel['h_pipe']
    box((x, bel['y'], mz + 0.15), (LT + 1.4, 0.7, 0.3), M_GOLD, name='slot', bevel=True)
    box((x, bel['y'], mz + 0.01), (LT + 1.3, 0.5, 0.02), M_DARK, name='slotmouth')
    rod((x, bel['y'] + 0.2, mz + 0.3), (x, bel['y'] + 0.2, mz + 6.0 + 0.7 * (k % 3)), 0.24, M_GOLD, 20)
    if k % 2 == 0:
        lt = bpy.data.lights.new(f'bl{k}', 'POINT'); lt.energy = 0.0; lt.color = pcol(p, 0.4); lt.shadow_soft_size = 0.5
        lo = link(bpy.data.objects.new(f'bl{k}', lt)); lo.location = (x + 0.46, bel['y'] - 1.0, bel['z'])
        tl = sorted(set(round(BL[n]['hits'][0]['t'], 4) for n in RAINN if BL[n]['hits'][0].get('tube') in (k, k + 1)))
        for t in tl:
            fh = int(round(t * FPS))
            lt.energy = 0.0; lt.keyframe_insert('energy', frame=max(0, fh - 1)); lt.energy = 500.0; lt.keyframe_insert('energy', frame=fh)
            lt.energy = 0.0; lt.keyframe_insert('energy', frame=fh + 9)
# decorative organ pipes behind the bells (the organ front)
for k, x in enumerate(np.linspace(XC - W_ / 2 + 0.3, XC + W_ / 2 - 0.3, 34)):
    hh = 5.0 + 3.0 * math.sin(math.pi * k / 33) + 0.6 * (k % 2)
    zb_ = ZR + NROW * RISE + 1.0
    rod((x, YR + NROW * DEPTH + 1.0, zb_), (x, YR + NROW * DEPTH + 1.0, zb_ + hh), 0.17, M_GOLD, 16)
box((XC, YR + NROW * DEPTH + 1.4, ZR + NROW * RISE + 3.5), (W_ + 1.2, 0.4, 9.0), M_WALL, name='organcase')
# the curtain: two velvet halves with folds, slide apart from T_CURTAIN
ct = EV['curtain']
def curtain_half(sgn):
    bm = bmesh.new(); hw = W_ / 2 + 0.6; zb_ = floor_z_at(XC) - 0.5; zt_ = bel['z'] + bel['h_pipe'] + 6.0
    nx, nz = 60, 2; verts = []
    for i in range(nx + 1):
        u = i / nx; x = sgn * u * hw; y = 0.25 * math.sin(u * 26 * math.pi)
        verts.append([bm.verts.new((x, y, zb_ + (zt_ - zb_) * j / nz)) for j in range(nz + 1)])
    for i in range(nx):
        for j in range(nz): bm.faces.new((verts[i][j], verts[i + 1][j], verts[i + 1][j + 1], verts[i][j + 1]))
    me = bpy.data.meshes.new('curtain'); bm.to_mesh(me); bm.free()
    for p in me.polygons: p.use_smooth = True
    o = link(bpy.data.objects.new('curtain', me)); o.data.materials.append(M_CURTAIN); o.location = (XC, YR - 0.6, 0)
    for f in range(NF):
        u = min(1, max(0, (T[f] - ct['t0']) / (ct['t1'] - ct['t0']))); u = u * u * (3 - 2 * u)
        o.location.x = XC + sgn * u * (hw + 0.5); o.scale.x = 1 - 0.75 * u
        if 0 < f < NF - 1 and (T[f] < ct['t0'] - 0.1 or T[f] > ct['t1'] + 0.1): continue
        o.keyframe_insert('location', index=0, frame=f); o.keyframe_insert('scale', index=0, frame=f)
    return o
curtain_half(-1); curtain_half(1)
# curtain valance (gold) and the proscenium
box((XC, YR - 0.9, bel['z'] + bel['h_pipe'] + 5.0), (W_ + 3.0, 0.4, 1.2), M_GOLD, name='valance', bevel=True)
for sx in (-1, 1):
    box((XC + sx * (W_ / 2 + 1.3), YR - 0.9, (floor_z_at(XC) + bel['z'] + 9) / 2), (1.2, 0.8, bel['z'] + 9 - floor_z_at(XC)), M_GOLDD, name='prosc', bevel=True)

# ---------------- the hall: back wall with tall windows (snowy night), chandeliers ----------------
YW = 24.0
strip_mesh(FLX, FLZ + 30.0, YW, YW + 0.5, 40.0, M_WALL, 'backwall')
for x0 in np.arange(-6.0, FLX[-1], 6.0):
    zf_ = floor_z_at(x0)
    box((x0, YW - 0.05, zf_ + 9.0), (2.4, 0.1, 9.0), M_WIN, name='win')
    box((x0, YW - 0.12, zf_ + 9.0), (0.12, 0.1, 9.0), M_GOLDD, name='mull')
    box((x0, YW - 0.12, zf_ + 9.0), (2.4, 0.1, 0.12), M_GOLDD, name='mull')
    for sx in (-1, 1): box((x0 + sx * 1.6, YW - 0.2, zf_ + 9.0), (0.5, 0.3, 18.0), M_GOLDD, name='pilaster')
# falling snow behind the windows: a few hundred dots drifting down
SN = 260
snx = rng.uniform(-8, FLX[-1], SN); snz0 = rng.uniform(0, 16, SN); snv = rng.uniform(0.4, 0.9, SN)
M_SNOW = emis('snow', (0.85, 0.9, 1.0), 2.0)
for k in range(SN):
    xw = round((snx[k] + 6.0) / 6.0) * 6.0 - 6.0
    x = xw + rng.uniform(-1.1, 1.1); zf_ = floor_z_at(xw)
    o = sphere((x, YW + 0.4, 0), 0.035, M_SNOW, (6, 4))
    zz = zf_ + 4.6 + ((snz0[k] - snv[k] * T) % 9.0)
    key_series(o, 'location', 2, zz)
def chandelier(c, s=1.0, bulbs=12):
    cx, cy, cz = c
    bpy.ops.mesh.primitive_torus_add(major_radius=0.6 * s, minor_radius=0.05 * s, major_segments=40, minor_segments=8, location=(cx, cy, cz))
    bpy.context.object.data.materials.append(M_GOLD)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.33 * s, minor_radius=0.04 * s, major_segments=32, minor_segments=8, location=(cx, cy, cz + 0.4 * s))
    bpy.context.object.data.materials.append(M_GOLD)
    rod((cx, cy, cz - 0.35 * s), (cx, cy, cz + 9.0), 0.03 * s, M_GOLD, 8)
    for j in range(bulbs):
        ph = 2 * math.pi * j / bulbs
        bx, by = cx + 0.6 * s * math.cos(ph), cy + 0.6 * s * math.sin(ph)
        sphere((bx, by, cz + 0.12 * s), 0.06 * s, M_CANDLE, (8, 6))
for x0 in np.arange(4.0, FLX[-1], 9.0):
    chandelier((x0, 10.0, floor_z_at(x0) + 9.5), 1.4)

# ---------------- signs (keywords for search, readable on a phone) ----------------
def sign(lines, x, y, z, w, h, plate=True):
    if plate:
        box((x, y + 0.08, z), (w, 0.12, h), M_PLATE, rot=(-0.33, 0, 0), name='plate')
        for sgn in (1, -1): box((x, y + 0.08 + 0.324 * sgn * h / 2, z + 0.946 * sgn * h / 2), (w + 0.1, 0.16, 0.07), M_GOLD, rot=(-0.33, 0, 0))
        for xx in (x - w / 2 + 0.2, x + w / 2 - 0.2): rod((xx, y + 0.1, z - h / 2), (xx, y + 0.1, floor_z_at(xx)), 0.05, M_GOLD, 8)
    zz = z + h / 2
    for s_, size, mt in lines:
        zz -= size * 1.25
        fit_w(text(s_, (x, y - 0.03 + 0.344 * (zz + size * 0.15 - z), zz + size * 0.15), size, mt), (w - 0.3) if plate else None)
PS = POS['solo']
def xs_(t): return float(np.interp(t, T, PS[:, 0]))
def zs_(t): return float(np.interp(t, T, PS[:, 2]))
sign([('BEETHOVEN', 0.46, M_TXT), ('Symphony No. 9 "Ode to Joy"', 0.3, M_TXT_G), ('Classical × Marble Machine', 0.26, M_TXT_R),
      ('ベートーヴェン 交響曲第9番「歓喜の歌」', 0.17, M_TXT)],
     xs_(0.5) + 0.5, -3.6, zs_(0.5) - 3.4, 4.3, 1.6)
sign([("New Year's Eve Choir", 0.34, M_TXT_G), ('Classical Music', 0.3, M_TXT), ('年末の大合唱 クラシック名曲', 0.18, M_TXT)], xs_(7.0) + 1.5, -3.4, zs_(7.0) - 1.9, 3.6, 1.3)
sign([('Satisfying Marble Machine', 0.32, M_TXT_G), ('A choir of marbles', 0.3, M_TXT), ('ビー玉マーブルマシンで合唱', 0.18, M_TXT)], xs_(13.0) + 1.0, -3.6, zs_(13.0) - 2.2, 4.0, 1.3)
sign([('Symphony No. 9 – 4th mvt.', 0.32, M_TXT_G), ("Beethoven's Ode to Joy", 0.28, M_TXT), ('第九 第4楽章「歓喜の歌」', 0.18, M_TXT)], xs_(19.5) + 1.0, -3.6, zs_(19.5) - 2.4, 4.0, 1.25)
sign([('How many will join?', 0.36, M_TXT_R), ('New Year classical', 0.28, M_TXT_G), ('1人から何人に？ 年越しクラシック', 0.18, M_TXT)], xs_(26.0) + 1.0, -3.6, zs_(26.0) - 2.4, 4.0, 1.3)
sign([('Jumping in', 0.26, M_TXT_G), ('from the audience!', 0.22, M_TXT), ('観客席から飛び入り参加！', 0.14, M_TXT)], EV['join'][0]['perch'][0] - 1.2, EV['join'][0]['perch'][1] + 1.0, EV['join'][0]['perch'][2] + 0.9, 2.2, 0.8)
sign([('THE GRAND CHOIR', 0.5, M_TXT_G), ('ひな壇の大合唱団', 0.26, M_TXT)], XC, YR - 1.2, bel['z'] + bel['h_pipe'] + 5.0, 0.1, 0.8, plate=False)

# ---------------- hit rings (FX pool) for melody keys and bass bells ----------------
NRING = 14
ring_loc = np.zeros((NRING, NF, 3)); ring_scl = np.full((NRING, NF), 1e-3); ring_a = np.zeros((NRING, NF)); ring_c = [(1, 1, 1)] * NRING
evs = [(h['t'], np.array(h['p']) - BL['solo']['r'] * np.array(h['n']), pcol(h['pitch'], 0.4)) for h in BL['solo']['hits']]
evs += [(h['t'], np.array(h['p']) - BL['bass']['r'] * np.array(h['n']), (1, 0.8, 0.4)) for h in BL['bass']['hits']]
for k, (t, p, c) in enumerate(sorted(evs, key=lambda e: e[0])):
    r = k % NRING; f0 = int(round(t * FPS))
    for df in range(14):
        f = f0 + df
        if f >= NF: break
        u = df / 13; ring_loc[r, f] = p + np.array([0, -0.32, 0.02]); ring_scl[r, f] = 0.35 + 1.4 * (1 - (1 - u) ** 2)
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

# ---------------- fireworks over the riser at the final chord ----------------
NB, NS = 7, 46
M_SPARK = fxmat('spark', 12.0)
FWC = [(1.0, 0.3, 0.3), (1.0, 0.8, 0.3), (0.4, 0.7, 1.0), (0.5, 1.0, 0.5), (1.0, 0.5, 1.0), (1.0, 1.0, 0.85), (1.0, 0.6, 0.2)]
TOPZ = bel['z'] + 1.6
BURSTS = []
for b in range(NB):
    tb = FINAL + 0.05 + 0.35 * b
    cb = np.array([XC + (b - 3) * 2.3 + rng.uniform(-0.5, 0.5), YR - 2.5, TOPZ + rng.uniform(-0.5, 1.5)])
    BURSTS.append((tb, cb))
    for s in range(NS):
        d = rng.normal(size=3); d /= np.linalg.norm(d); sp_ = rng.uniform(2.5, 4.0)
        o = sphere(tuple(cb), 0.09, M_SPARK, (6, 4)); col = FWC[b % len(FWC)]; o.color = (*col, 0)
        Pm = np.zeros((NF, 3)); A_ = np.zeros(NF); Sc = np.full(NF, 0.001)
        for f in range(NF):
            u = T[f] - tb
            if u < 0: Pm[f] = cb; continue
            Pm[f] = cb + d * sp_ * (1 - math.exp(-u * 1.6)) / 1.6 + np.array([0, 0, -0.6 * u * u])
            A_[f] = max(0.0, 1 - u / 2.2) ** 1.3; Sc[f] = 1.0 if u < 2.3 else 0.001
        key_loc(o, Pm); key_series(o, 'color', 3, A_, 1e-3)
        for k in range(3): key_series(o, 'scale', k, Sc * 0.09)
# a 2027 banner of light above the riser after the last seat is taken
t_last = max(r['t_seat'] for r in EV['rain'])
bn = text('HAPPY NEW YEAR', (XC, YR - 1.6, bel['z'] + 2.3), 1.05, M_TXT_G, rot=(math.pi / 2 - 0.2, 0, 0))
bn.scale = (0.001,) * 3; bn.keyframe_insert('scale', frame=int(t_last * FPS))
bn.scale = (1, 1, 1); bn.keyframe_insert('scale', frame=int(t_last * FPS) + 8)

# ---------------- camera: follow the soloist (and its lanes), widen for the riser ----------------
FOC = np.zeros((NF, 3)); WID = np.zeros(NF)
lanes_t = np.interp(T, [h['t'] for h in BL['solo']['hits']], [h['lanes'] for h in BL['solo']['hits']])
bz = bel['z'] + bel['h_pipe'] * 0.4
for f in range(NF):
    t = T[f]; s = PS[f]
    if t < E['t0'] - 0.2:
        FOC[f] = s + np.array([0.5, 0, -1.0]); WID[f] = 6.0
    else:
        FOC[f] = s + np.array([1.0, (lanes_t[f] - 1) * LN * 0.35, -0.5 - 0.06 * lanes_t[f]])
        WID[f] = 6.6 + 0.12 * lanes_t[f]
    tw = ct['t0'] - 0.6
    if t > tw:
        u = min(1, (t - tw) / 2.6); u = u * u * (3 - 2 * u)
        wide = np.array([XC, YR + 0.2 * NROW * DEPTH, (ZR + bz) / 2 - 1.0])
        FOC[f] = (1 - u) * FOC[f] + u * wide; WID[f] = (1 - u) * WID[f] + u * 19.0
FOC[:, 0] = smooth(FOC[:, 0], 0.45); FOC[:, 1] = smooth(FOC[:, 1], 0.8); FOC[:, 2] = smooth(FOC[:, 2], 0.6); WID = smooth(WID, 0.5)
LENS = 32.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 500
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24
    CP[f] = (FOC[f, 0], FOC[f, 1] - D, FOC[f, 2] + 0.34 * D)
    TP[f] = (FOC[f, 0], FOC[f, 1], FOC[f, 2] + 0.2)
key_loc(cam, CP); key_loc(tgt, TP)

# ---------------- the counter (合唱団 ○人), riding with the camera near the top of the frame ----------------
hud = link(bpy.data.objects.new('hud', None)); hud.parent = cam
HZ = 6.0; hh_ = HZ * 24 / LENS                         # visible height at this distance
hud.location = (0, hh_ * 0.36, -HZ)
pl_ = box((0, 0, -0.02), (hh_ * 9 / 16 * 0.62, hh_ * 0.075, 0.02), M_PLATE, name='hudplate'); pl_.parent = hud
for sgn in (1, -1):
    gb_ = box((0, sgn * hh_ * 0.0375, -0.01), (hh_ * 9 / 16 * 0.64, hh_ * 0.006, 0.02), M_GOLD, name='hudgold'); gb_.parent = hud
ts_h = hh_ * 0.034
DW = ts_h * 0.95                                     # digit slot spacing (DejaVu digits)
X_LBL = hh_ * 9 / 16 * 0.105 - DW - 0.75 * ts_h      # right edge of the label, a little left of the first digit
t1 = fit_w(text('CHOIR:', (X_LBL, ts_h * 0.26, 0), ts_h * 1.2, M_TXT, rot=(0, 0, 0), align='RIGHT'), X_LBL + hh_ * 9 / 16 * 0.29); t1.parent = hud
t2 = text('合唱団', (X_LBL, -ts_h * 0.64, 0), ts_h * 0.45, M_TXT, rot=(0, 0, 0), align='RIGHT'); t2.parent = hud
cnt = EV['count']
ctimes = np.array([c[0] for c in cnt]); cvals = np.array([c[1] for c in cnt])
CV = np.array([cvals[max(0, np.searchsorted(ctimes, t, 'right') - 1)] for t in T])
for slot in range(3):
    for dgt in range(10):
        o = text(str(dgt), (hh_ * 9 / 16 * 0.105 + (slot - 1) * DW, 0, 0), ts_h * 1.3, M_TXT_G, rot=(0, 0, 0)); o.parent = hud
        vis = np.zeros(NF)
        for f in range(NF):
            v = int(CV[f]); s_ = f'{v:<3d}'                # left-aligned right after 'CHOIR'
            vis[f] = 1.0 if s_[slot] == str(dgt) else 0.0
        if not vis.any(): o.hide_render = True; continue
        for k in range(3): key_series(o, 'scale', k, np.maximum(0.001, vis))
# bump the plate when the count changes
bump = np.ones(NF)
for f in range(1, NF):
    if CV[f] != CV[f - 1]:
        for df in range(6):
            if f + df < NF: bump[f + df] = max(bump[f + df], 1.0 + 0.08 * (1 - df / 6))
for k in range(2): key_series(hud, 'scale', k, bump, 1e-3)

# ---------------- lights ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bgn = w.node_tree.nodes['Background']; bgn.inputs['Color'].default_value = (0.03, 0.015, 0.012, 1); bgn.inputs['Strength'].default_value = 0.9
key_ = bpy.data.lights.new('key', 'SUN'); key_.energy = 2.6; key_.angle = 0.2; key_.color = (1.0, 0.85, 0.65)
ko = link(bpy.data.objects.new('key', key_)); ko.rotation_euler = (math.radians(55), math.radians(-12), math.radians(-20))
fil = bpy.data.lights.new('fill', 'SUN'); fil.energy = 0.7; fil.color = (0.6, 0.7, 1.0)
fo = link(bpy.data.objects.new('fill', fil)); fo.rotation_euler = (math.radians(70), math.radians(30), math.radians(160))
spd = bpy.data.lights.new('spot', 'SPOT'); spd.energy = 3500; spd.spot_size = math.radians(22); spd.spot_blend = 0.5
spd.color = (1.0, 0.88, 0.7); spd.shadow_soft_size = 0.3
so_ = link(bpy.data.objects.new('spot', spd)); stg = link(bpy.data.objects.new('spottgt', None))
Ps = PS.copy()
for k in range(3): Ps[:, k] = smooth(PS[:, k], 0.25)
Ps[:, 1] = (lanes_t - 1) * LN * 0.5
key_loc(stg, Ps - np.array([0, 0, 0.25])); key_loc(so_, Ps + np.array([-2.0, -7.0, 9.0]))
c_ = so_.constraints.new('TRACK_TO'); c_.target = stg; c_.track_axis = 'TRACK_NEGATIVE_Z'; c_.up_axis = 'UP_Y'
# riser light (comes on with the curtain)
ra = bpy.data.lights.new('riser', 'AREA'); ra.size = 16.0; ra.color = (1.0, 0.85, 0.7); ra.energy = 0.0
rao = link(bpy.data.objects.new('riserlight', ra)); rao.location = (XC, YR - 6.0, ZR + 12.0); rao.rotation_euler = (math.radians(35), 0, 0)
ra.keyframe_insert('energy', frame=int(ct['t0'] * FPS)); ra.energy = 9000.0; ra.keyframe_insert('energy', frame=int(ct['t1'] * FPS))

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
        print('t', tt_, 'cam', [round(v, 2) for v in cam.matrix_world.translation], 'WID', round(WID[f], 2), 'FOC', np.round(FOC[f], 2),
              'solo', np.round(PS[f], 2))
print('scene objects', len(bpy.data.objects))
