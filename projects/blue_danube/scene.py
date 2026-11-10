"""The Blue Danube - 氷上の舞踏会 (a ball on the frozen Danube), side view of a long terraced ice ballroom at night.
Every melody note = the dancing couple (red + white) landing on an ice key of that pitch; every bass note = the gold ball
landing on a gold bell; every chord = a small couple landing on two ice bells. layout.py plans all motion (constant gravity,
bounces that never add speed, sled / door / slide / see-saw mechanisms); this file builds and animates it.
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
FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.5) * FPS); T = np.arange(NF) / FPS
G = LAY['G']; EV = LAY['ev']; LY = LAY['lane_y']; DXP = LAY['dxp']
RP, RB, RC = LAY['R']['pair'], LAY['R']['bass'], LAY['R']['chord']
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


def frame_from_n(n):
    n = np.array(n, float); n /= np.linalg.norm(n)
    xa = np.cross([0, 1, 0], n); xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return Matrix(((xa[0], ya[0], n[0]), (xa[1], ya[1], n[1]), (xa[2], ya[2], n[2])))


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

# ---------------- materials ----------------
M_RED = mat('red', (0.9, 0.04, 0.08), 0.12, **{'Coat Weight': 1.0, 'Emission Color': (1.0, 0.1, 0.12), 'Emission Strength': 0.6})
M_WHITE = mat('white', (0.93, 0.94, 0.97), 0.1, **{'Coat Weight': 1.0, 'Emission Color': (0.85, 0.9, 1.0), 'Emission Strength': 0.35})
M_GOLD = mat('gold', (1.0, 0.7, 0.25), 0.22, **{'Metallic': 0.9})
M_GOLDB = mat('goldball', (1.0, 0.72, 0.22), 0.18, **{'Metallic': 0.85, 'Coat Weight': 0.6})
M_KEY = glowmat('icekey', gain=9.0, base=0.45, rough=0.06, coat=1.0)
M_BELL = glowmat('bell', gain=5.0, base=0.0, rough=0.25, metal=0.85)
M_TINK = glowmat('tink', gain=9.0, base=0.6, rough=0.1, metal=0.2, coat=1.0)
M_ICE = mat('ice', (0.05, 0.13, 0.26), 0.04, **{'Coat Weight': 1.0, 'Emission Color': (0.1, 0.25, 0.5), 'Emission Strength': 0.05})
M_ICEPOST = mat('icepost', (0.3, 0.5, 0.75), 0.1, **{'Emission Color': (0.3, 0.55, 0.95), 'Emission Strength': 0.25})
M_POST = mat('post', (0.08, 0.12, 0.22), 0.3, **{'Metallic': 0.7})
M_PLATE = mat('plate', (0.03, 0.05, 0.14), 0.3, **{'Coat Weight': 1.0})
M_SNOW = mat('snow', (0.7, 0.78, 0.92), 0.75)
M_STEEL = mat('steel', (0.6, 0.62, 0.68), 0.3, **{'Metallic': 0.9})
M_DARK = mat('dark', (0.04, 0.05, 0.08), 0.6)
M_WOOD = mat('wood', (0.45, 0.22, 0.1), 0.5)
M_SLED = mat('sled', (0.8, 0.08, 0.06), 0.3, **{'Coat Weight': 0.8})
M_GLASS = mat('glass', (0.8, 0.9, 1.0), 0.05, **{'Alpha': 0.12, 'Emission Color': (0.4, 0.7, 1.0), 'Emission Strength': 0.15})
M_WATER = mat('water', (0.01, 0.02, 0.05), 0.05)
M_OBJ = objcol('objcol', 0.5)
M_LAMP = objcol('lampcol', 0.4, emit=6.0)
M_FX = fxmat('fx', 8.0)
M_TXT = emis('txtw', (0.85, 0.95, 1.0), 4.0); M_TXT_G = emis('txtg', (1.0, 0.8, 0.35), 6.0); M_TXT_P = emis('txtp', (1.0, 0.55, 0.75), 5.0)
M_CANDLE = emis('candle', (1.0, 0.75, 0.4), 18.0)

# ---------------- balls ----------------
BALL = {}
COLB = {'red': M_RED, 'white': M_WHITE, 'bass': M_GOLDB}
CHC = {'ch2lo': (0.35, 0.65, 1.0), 'ch2hi': (0.75, 0.88, 1.0), 'ch3lo': (1.0, 0.55, 0.75), 'ch3hi': (1.0, 0.85, 0.92)}
M_CHB = mat('chball', (1, 1, 1), 0.12, **{'Coat Weight': 1.0, 'Emission Strength': 0.5}); _nt = M_CHB.node_tree
_oi = _nt.nodes.new('ShaderNodeObjectInfo'); _nt.links.new(_oi.outputs['Color'], _nt.nodes['Principled BSDF'].inputs['Base Color']); _nt.links.new(_oi.outputs['Color'], _nt.nodes['Principled BSDF'].inputs['Emission Color'])
for n, b in BP.items():
    rr = b.r
    bpy.ops.mesh.primitive_uv_sphere_add(radius=rr, segments=32, ring_count=16); o = bpy.context.object; o.name = n
    for p in o.data.polygons: p.use_smooth = True
    if n in COLB: o.data.materials.append(COLB[n])
    else: o.data.materials.append(M_CHB); o.color = (*CHC[n], 1)
    # a ribbon band so the spin reads
    bpy.ops.mesh.primitive_torus_add(major_radius=rr * 1.0, minor_radius=rr * 0.12, major_segments=32, minor_segments=8)
    hb = bpy.context.object; hb.parent = o
    hb.data.materials.append({'red': M_WHITE, 'white': M_RED, 'bass': M_WHITE}.get(n, M_GOLD))
    key_loc(o, POS[n])
    sp = np.zeros(NF)
    for f in range(1, NF): sp[f] = sp[f - 1] + (POS[n][f, 0] - POS[n][f - 1, 0]) / rr
    o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, sp)
    BALL[n] = o

def strip_mesh(X, Z, y0, y1, th, m, name):
    bm = bmesh.new(); vs = []
    for x, z in zip(X, Z):
        vs.append([bm.verts.new((x, y0, z)), bm.verts.new((x, y1, z)), bm.verts.new((x, y1, z - th)), bm.verts.new((x, y0, z - th))])
    for a, b in zip(vs, vs[1:]):
        for k in range(4): bm.faces.new((a[k], b[k], b[(k + 1) % 4], a[(k + 1) % 4]))
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new(name, me)); o.data.materials.append(m)
    for p in o.data.polygons: p.use_smooth = True
    return o

# ---------------- keys and bells: one per hit ----------------
PITCH_COL = [(1.0, 0.12, 0.12), (1.0, 0.35, 0.05), (1.0, 0.6, 0.05), (1.0, 0.85, 0.1), (0.55, 1.0, 0.1), (0.1, 1.0, 0.3),
             (0.05, 1.0, 0.75), (0.05, 0.7, 1.0), (0.15, 0.35, 1.0), (0.45, 0.2, 1.0), (0.75, 0.2, 1.0), (1.0, 0.2, 0.65)]
def pcol(p, light=0.0):
    c = PITCH_COL[p % 12]; return tuple(min(1, x * (1 - light) + light) for x in c)
PIECES = []          # dict(o, kind, t, top, n, hw, hd, th, ball)
KEY_TH = 0.12
def riser(objs, t_hit, anchor, lead=2.6, stay=1.2):
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


def add_piece(o, kind, t, top, n, hw, hd, th, ball, flash_col, flash=True):
    PIECES.append(dict(o=o, kind=kind, t=t, top=np.array(top), n=np.array(n), hw=hw, hd=hd, th=th, ball=ball))
    if flash:
        fh = int(round(t * FPS)); base = o.location.copy(); nv = Vector(n)
        o.color = (*flash_col, 0); o.keyframe_insert('color', index=3, frame=max(0, fh - 1))
        for df, a in ((0, 1.0), (4, 0.55), (18, 0.1)):
            o.color = (*flash_col, a); o.keyframe_insert('color', index=3, frame=fh + df)
        for df, d in ((-1, 0), (1, -0.03), (4, 0.01), (8, 0)):
            o.location = base + nv * d; o.keyframe_insert('location', frame=max(0, fh + df))


def floor_z_at(x): return float(np.interp(x, FLX, FLZ))


# floor (frozen river terraces): below every contact
allc = sorted([(h['p'][0], h['p'][2]) for b in BP.values() for h in b.hits])
AX = np.array([a[0] for a in allc]); AZ = np.array([a[1] for a in allc])
FLX = np.arange(-14.0, AX.max() + 20, 0.5)
FLZ = np.array([AZ[np.abs(AX - x) < 3.0].min() if (np.abs(AX - x) < 3.0).any() else np.nan for x in FLX])
s_ = ~np.isnan(FLZ); FLZ = np.interp(FLX, FLX[s_], FLZ[s_]) - 1.15
for _ in range(4): FLZ = np.minimum(FLZ, np.convolve(np.pad(FLZ, 2, mode='edge'), np.ones(5) / 5, 'valid'))
FLZ = np.minimum.accumulate(FLZ)                           # the frozen river only ever steps down

pair_shared = {}
for n in ('red', 'white'):
    for h in BP[n].hits:
        if h['kind'] in ('crack', 'reunion'): pair_shared.setdefault((h['ev'], h['kind']), []).append(h)
KEYS = []
for (ev, kind), hs in pair_shared.items():
    pa = np.mean([h['p'] for h in hs], axis=0); n = np.array(hs[0]['n'])
    top = pa - RP * n; w = 2 * DXP + 0.44; d = 2 * LAY['dyp'] + 0.42
    ctr = top - n * KEY_TH / 2
    o = obj(prim('bcube', M_KEY), 'widekey')
    o.matrix_world = Matrix.Translation(Vector(ctr)) @ frame_from_n(n).to_4x4() @ Matrix.Diagonal((w, d, KEY_TH, 1))
    col = pcol(hs[0]['pitch'], 0.0); o.color = (*col, 0)
    add_piece(o, kind, hs[0]['t'], top, n, w / 2, d / 2, KEY_TH / 2, 'pair', col)
    KEYS.append(dict(o=o, top=top, kind=kind, t=hs[0]['t'], w=w, ev=ev))
for n in ('red', 'white'):
    for h in BP[n].hits:
        if h['kind'] not in ('key', 'uni', 'half', 'floe', 'low'): continue
        nv = np.array(h['n']); top = np.array(h['p']) - RP * nv; w = 0.42; d = 0.6
        ctr = top - nv * KEY_TH / 2
        o = obj(prim('bcube', M_KEY), 'key')
        o.matrix_world = Matrix.Translation(Vector(ctr)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((w, d, KEY_TH, 1))
        col = pcol(h['pitch'], 0.0); o.color = (*col, 0)
        add_piece(o, h['kind'], h['t'], top, nv, w / 2, d / 2, KEY_TH / 2, n, col)
        KEYS.append(dict(o=o, top=top, kind=h['kind'], t=h['t'], w=w, ev=h['ev']))
# ice posts under the keys (to the floor); crack key / floe keys get their own look
for K in KEYS:
    c = K['top']
    if K['kind'] == 'crack': continue
    z0 = c[2] - KEY_TH - 0.02; zf = floor_z_at(c[0])
    po = rod((c[0], c[1], z0), (c[0], c[1], zf - 0.3), 0.035, M_POST, 8)
    if K['t'] > 2.0: riser([K['o'], po], K['t'], c, lead=1.2, stay=0.9)
# bells
BELL_T = {}; TIER = {}
for n, b in BP.items():
    for h in b.hits:
        if h['kind'] not in ('bell', 'tink', 'fincup'): continue
        nv = np.array(h['n']) if h['kind'] != 'fincup' else np.array([0, 0, 1.0])
        r = b.r; top = np.array(h['p']) - r * nv
        p = h['pitch']
        if h['kind'] == 'bell':
            sz = 0.3 + 0.008 * (62 - p); m_ = M_BELL; col = (1.0, 0.72, 0.3)
            o = obj(prim('bell', m_, 32), 'bell')
            o.matrix_world = Matrix.Translation(Vector(top)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((sz * 0.75, sz * 0.75, sz, 1))
            hw = 0.42 * sz * 0.75; th = 0.05
        elif h['kind'] == 'tink':
            rs = 0.15 + 0.004 * (66 - p); col = pcol(p, 0.1)
            cs = top - nv * rs                                     # sphere centre: its surface normal at the contact = n
            o = obj(prim('suzu', M_TINK), 'suzu'); o.location = tuple(cs); o.scale = (rs, rs, rs)
            sl_ = obj(prim('cube', M_DARK), 'slot'); sl_.parent = o; sl_.location = (0, -0.55, -0.55); sl_.scale = (1.1, 0.5, 0.12); sl_.rotation_euler = (0.8, 0, 0)
            bd = obj(prim('cyl', M_GOLD, 24), 'band'); bd.parent = o; bd.scale = (1.01, 1.01, 0.12)
            hw = rs * 0.5; th = 0.05; sz = 2 * rs
        else:                                           # final cup bell (mouth up): the ball lands in it and stays
            col = pcol(p, 0.4); sz = 0.32
            o = obj(prim('bell', M_TINK, 24), 'fincup')
            o.matrix_world = Matrix.Translation(Vector(top + [0, 0, -0.0])) @ Matrix.Diagonal((sz, sz, -sz * 0.9, 1))
            hw = 0.2; th = 0.05
        o.color = (*col, 0)
        add_piece(o, h['kind'], h['t'], top, nv, hw, hw, th, n, col)
        bot = top - nv * (sz if h['kind'] == 'bell' else (2 * rs if h['kind'] == 'tink' else 0.3))
        zf = floor_z_at(top[0]) - 0.2
        TIER.setdefault(n[:3] if n != 'bass' else 'bass', []).append((top[0], bot[2], bot[1]))
        po = rod(bot, (bot[0], bot[1], bot[2] - 0.6), 0.025 if h['kind'] == 'tink' else 0.04, M_POST, 8)
        if h['t'] > 2.5: riser([o, po], h['t'], top, lead=1.2, stay=0.9 if h['kind'] != 'fincup' else 99)
        BELL_T.setdefault(n, []).append(h['t'])

# hit rings (FX pool)
NRING = 16
ring_loc = np.zeros((NRING, NF, 3)); ring_scl = np.full((NRING, NF), 1e-3); ring_a = np.zeros((NRING, NF)); ring_c = [(1, 1, 1)] * NRING
for k, P_ in enumerate(sorted([p for p in PIECES if p['kind'] in ('key', 'uni', 'half', 'floe', 'low', 'crack', 'reunion', 'bell')], key=lambda b: b['t'])):
    r = k % NRING; f0 = int(round(P_['t'] * FPS))
    for df in range(14):
        f = f0 + df
        if f >= NF: break
        u = df / 13; ring_loc[r, f] = P_['top'] + np.array([0, -0.32, 0.02]); ring_scl[r, f] = 0.35 + 1.4 * (1 - (1 - u) ** 2)
        ring_a[r, f] = (1 - u) ** 1.6
    ring_c[r] = (0.7, 0.9, 1.0) if P_['kind'] != 'bell' else (1, 0.8, 0.4)
for r in range(NRING):
    bpy.ops.mesh.primitive_torus_add(major_radius=0.25, minor_radius=0.014, major_segments=32, minor_segments=5, rotation=(math.pi / 2, 0, 0))
    o = bpy.context.object; o.data.materials.append(M_FX); o.color = (*ring_c[r], 0)
    for f in range(1, NF):
        if ring_a[r, f] == 0: ring_loc[r, f] = ring_loc[r, f - 1]
    key_loc(o, ring_loc[r])
    for k in range(3): key_series(o, 'scale', k, ring_scl[r])
    key_series(o, 'color', 3, ring_a[r], 1e-3)

# ---------------- the frozen Danube: terraced ice floor, snow banks, dark water in the crack ----------------
strip_mesh(FLX, FLZ, -6.0, 9.0, 0.4, M_ICE, 'river')
# snow banks: near bank (foreground) and far bank
strip_mesh(FLX, FLZ + 0.25, -9.0, -6.0, 1.0, M_SNOW, 'nearbank')
strip_mesh(FLX, FLZ + 0.6, 9.0, 14.0, 1.2, M_SNOW, 'farbank')
# ---------------- ballroom balconies for the orchestra (one per tier), with strings of lights ----------------
LAMPC = [(1.0, 0.75, 0.4), (0.6, 0.85, 1.0), (1.0, 0.5, 0.7), (0.7, 1.0, 0.8)]
M_BALC = mat('balc', (0.12, 0.2, 0.38), 0.15, **{'Coat Weight': 1.0, 'Emission Color': (0.2, 0.4, 0.8), 'Emission Strength': 0.25})
for tier, pts in TIER.items():
    pts = sorted(pts); bx_ = np.array([p[0] for p in pts]); bz_ = np.array([p[1] for p in pts]); by_ = float(np.median([p[2] for p in pts]))
    X = np.arange(bx_[0] - 1.5, bx_[-1] + 1.5, 0.25)
    Zt = np.array([bz_[np.abs(bx_ - x) < 1.6].min() if (np.abs(bx_ - x) < 1.6).any() else np.nan for x in X])
    ok_ = ~np.isnan(Zt); Zt = np.interp(X, X[ok_], Zt[ok_]) - 0.55
    for _ in range(3): Zt = np.minimum(Zt, np.convolve(np.pad(Zt, 3, mode='edge'), np.ones(7) / 7, 'valid'))
    strip_mesh(X, Zt, by_ - 0.45, by_ + 0.45, 0.14, M_BALC, 'balcony')
    strip_mesh(X, Zt + 0.22, by_ - 0.5, by_ - 0.45, 0.05, M_GOLD, 'rail')
    for k, (x, z) in enumerate(zip(X[::4], Zt[::4])):
        rod((x, by_ - 0.475, z), (x, by_ - 0.475, z + 0.2), 0.012, M_GOLD, 6)
        o = sphere((x, by_ - 0.5, z - 0.08), 0.045, M_LAMP, (8, 6)); o.color = (*LAMPC[k % 4], 1)
    for x, z in list(zip(X, Zt))[::24]:
        rod((x, by_ + 0.3, z - 0.14), (x, by_ + 0.3, floor_z_at(x)), 0.05, M_POST, 8)
# decorative chandeliers above the far lanes (no bells) every 13 m
def chandelier(c, s=1.0, bulbs=10, name='chand', chain=True):
    cx, cy, cz = c
    bpy.ops.mesh.primitive_torus_add(major_radius=0.55 * s, minor_radius=0.045 * s, major_segments=40, minor_segments=8, location=(cx, cy, cz))
    bpy.context.object.data.materials.append(M_GOLD)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.3 * s, minor_radius=0.035 * s, major_segments=32, minor_segments=8, location=(cx, cy, cz + 0.35 * s))
    bpy.context.object.data.materials.append(M_GOLD)
    rod((cx, cy, cz - 0.3 * s), (cx, cy, cz + 0.6 * s), 0.05 * s, M_GOLD, 10)
    for j in range(bulbs):
        ph = 2 * math.pi * j / bulbs
        bx, by = cx + 0.55 * s * math.cos(ph), cy + 0.55 * s * math.sin(ph)
        rod((bx, by, cz), (bx, by, cz + 0.14 * s), 0.025 * s, M_WHITE, 6)
        sphere((bx, by, cz + 0.19 * s), 0.05 * s, M_CANDLE, (8, 6))
        ic = obj(prim('icicle', M_ICEPOST, 8), 'drop'); ic.location = (bx, by, cz - 0.04 * s); ic.scale = (0.03 * s, 0.03 * s, 0.22 * s)
    if chain: rod((cx, cy, cz + 0.6 * s), (cx, cy, cz + 9.0), 0.02, M_GOLD, 6)
for x0 in np.arange(10.0, FLX[-1] - 12, 11.0):
    chandelier((x0, 4.5, floor_z_at(x0) + 10.5), 1.2, name='deco')
# lanterns on the near bank (low, out of the way)
for x0 in np.arange(-6.0, FLX[-1], 4.5):
    zf = floor_z_at(x0) + 0.25
    o = sphere((x0, -6.3, zf + 0.25), 0.12, M_LAMP, (12, 8)); o.color = (1.0, 0.75, 0.45, 1)

# ---------------- sky: gradient, stars, aurora curtains, distant city ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes['Background']; bg.inputs['Color'].default_value = (0.02, 0.04, 0.12, 1); bg.inputs['Strength'].default_value = 0.3
moon = bpy.data.lights.new('moon', 'SUN'); moon.energy = 0.6; moon.angle = 0.05; moon.color = (0.7, 0.8, 1.0)
mo = link(bpy.data.objects.new('moon', moon)); mo.rotation_euler = (math.radians(55), math.radians(-15), math.radians(-25))
fill = bpy.data.lights.new('fill', 'SUN'); fill.energy = 0.35; fill.color = (1.0, 0.8, 0.6)
fo = link(bpy.data.objects.new('fill', fill)); fo.rotation_euler = (math.radians(70), math.radians(30), math.radians(160))
SKY = bpy.data.materials.new('sky'); SKY.use_nodes = True; nt = SKY.node_tree; nt.nodes.clear()
tc = nt.nodes.new('ShaderNodeTexCoord'); sepz = nt.nodes.new('ShaderNodeSeparateXYZ'); rp = nt.nodes.new('ShaderNodeValToRGB')
rp.color_ramp.elements[0].color = (0.05, 0.08, 0.25, 1); rp.color_ramp.elements[1].color = (0.0, 0.0, 0.03, 1)
e_ = rp.color_ramp.elements.new(0.35); e_.color = (0.08, 0.05, 0.22, 1)
mp = nt.nodes.new('ShaderNodeMapRange'); mp.inputs['From Min'].default_value = -0.5; mp.inputs['From Max'].default_value = 0.5
vo_ = nt.nodes.new('ShaderNodeTexVoronoi'); vo_.inputs['Scale'].default_value = 140.0
try: vo_.feature = 'F1'
except Exception: pass
st = nt.nodes.new('ShaderNodeMapRange'); st.inputs['From Min'].default_value = 0.035; st.inputs['From Max'].default_value = 0.0
st.inputs['To Min'].default_value = 0.0; st.inputs['To Max'].default_value = 3.0
add = nt.nodes.new('ShaderNodeMixRGB'); add.blend_type = 'ADD'; add.inputs['Fac'].default_value = 1.0
em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Strength'].default_value = 1.0; out = nt.nodes.new('ShaderNodeOutputMaterial')
nt.links.new(tc.outputs['Generated'], sepz.inputs[0]); nt.links.new(tc.outputs['Object'], vo_.inputs['Vector'])
mp2 = nt.nodes.new('ShaderNodeMapRange'); mp2.inputs['From Min'].default_value = 0.0; mp2.inputs['From Max'].default_value = 1.0
nt.links.new(sepz.outputs['Y'], mp2.inputs['Value']); nt.links.new(mp2.outputs[0], rp.inputs['Fac'])
nt.links.new(vo_.outputs['Distance'], st.inputs['Value'])
nt.links.new(rp.outputs['Color'], add.inputs[1]); nt.links.new(st.outputs[0], add.inputs[2])
nt.links.new(add.outputs[0], em.inputs['Color']); nt.links.new(em.outputs[0], out.inputs[0])
SKYX0, SKYX1 = FLX[0] - 20, FLX[-1] + 30
skyo = obj(prim('plane', SKY), 'sky'); skyo.location = ((SKYX0 + SKYX1) / 2, 60, -2); skyo.rotation_euler = (math.pi / 2, 0, 0)
skyo.scale = (SKYX1 - SKYX0 + 80, 90, 1)
# aurora: two curtains of animated emissive noise (green -> teal -> pink), transparent at the edges
def aurora(y, z, h, c1, c2, speed, name):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    tc = nt.nodes.new('ShaderNodeTexCoord'); mpn = nt.nodes.new('ShaderNodeMapping')
    nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 2.2; nz.inputs['Detail'].default_value = 3.0
    wv = nt.nodes.new('ShaderNodeTexWave'); wv.inputs['Scale'].default_value = 3.0; wv.inputs['Distortion'].default_value = 6.0
    sep = nt.nodes.new('ShaderNodeSeparateXYZ')
    ramp = nt.nodes.new('ShaderNodeValToRGB'); ramp.color_ramp.elements[0].color = (*c1, 1); ramp.color_ramp.elements[1].color = (*c2, 1)
    fall = nt.nodes.new('ShaderNodeValToRGB'); fe = fall.color_ramp.elements
    fe[0].position = 0.0; fe[0].color = (0, 0, 0, 1); fe[1].position = 1.0; fe[1].color = (0, 0, 0, 1)
    e1 = fe.new(0.25); e1.color = (1, 1, 1, 1); e2 = fe.new(0.55); e2.color = (0.35, 0.35, 0.35, 1)
    mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'
    mul2 = nt.nodes.new('ShaderNodeMath'); mul2.operation = 'MULTIPLY'; mul2.inputs[1].default_value = 2.6
    emn = nt.nodes.new('ShaderNodeEmission'); trn = nt.nodes.new('ShaderNodeBsdfTransparent'); mix = nt.nodes.new('ShaderNodeMixShader')
    clamp = nt.nodes.new('ShaderNodeClamp')
    o_ = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(tc.outputs['Generated'], mpn.inputs['Vector']); nt.links.new(mpn.outputs[0], wv.inputs['Vector']); nt.links.new(mpn.outputs[0], nz.inputs['Vector'])
    nt.links.new(tc.outputs['Generated'], sep.inputs[0]); nt.links.new(sep.outputs['Y'], fall.inputs['Fac'])
    nt.links.new(wv.outputs['Fac'], mul.inputs[0]); nt.links.new(fall.outputs['Color'], mul.inputs[1])
    nt.links.new(nz.outputs['Fac'], ramp.inputs['Fac']); nt.links.new(ramp.outputs['Color'], emn.inputs['Color'])
    nt.links.new(mul.outputs[0], mul2.inputs[0]); nt.links.new(mul2.outputs[0], emn.inputs['Strength'])
    nt.links.new(mul.outputs[0], clamp.inputs['Value']); nt.links.new(clamp.outputs[0], mix.inputs['Fac'])
    nt.links.new(trn.outputs[0], mix.inputs[1]); nt.links.new(emn.outputs[0], mix.inputs[2]); nt.links.new(mix.outputs[0], o_.inputs[0])
    for f, val in ((0, 0.0), (NF - 1, speed * NF / FPS)):
        mpn.inputs['Location'].default_value[0] = val; mpn.inputs['Location'].keyframe_insert('default_value', index=0, frame=f)
    for fc in (m.node_tree.animation_data.action.fcurves if m.node_tree.animation_data and hasattr(m.node_tree.animation_data.action, 'fcurves') else []):
        for kp in fc.keyframe_points: kp.interpolation = 'LINEAR'
    o = obj(prim('plane', m), name); o.location = ((SKYX0 + SKYX1) / 2, y, z); o.rotation_euler = (math.pi / 2, 0, 0)
    o.scale = (SKYX1 - SKYX0 + 60, h, 1); o.visible_shadow = False
    return o
AUR = [aurora(55, 16, 40, (0.1, 1.0, 0.45), (0.1, 0.7, 1.0), 0.004, 'aur1'), aurora(50, 10, 30, (0.4, 1.0, 0.6), (1.0, 0.35, 0.8), -0.006, 'aur2')]
# distant hills on the far bank
M_HILL = mat('hill', (0.02, 0.035, 0.08), 0.9)
for k, x0 in enumerate(np.arange(SKYX0, SKYX1, 14.0)):
    o = sphere((x0 + rng.uniform(-4, 4), 70, floor_z_at(min(max(x0, FLX[0]), FLX[-1])) - 6), 1.0, M_HILL, (24, 12)); o.scale = (rng.uniform(14, 22), 6, rng.uniform(7, 11))

# ---------------- falling snow: 3 transparent layers of soft dots in front of the camera ----------------
def snow_layer(dist, scale, speed, name, strength=1.6):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    tc = nt.nodes.new('ShaderNodeTexCoord'); mpn = nt.nodes.new('ShaderNodeMapping')
    vor = nt.nodes.new('ShaderNodeTexVoronoi'); vor.inputs['Scale'].default_value = scale
    try: vor.inputs['Randomness'].default_value = 1.0
    except Exception: pass
    mr = nt.nodes.new('ShaderNodeMapRange'); mr.inputs['From Min'].default_value = 0.09; mr.inputs['From Max'].default_value = 0.02
    emn = nt.nodes.new('ShaderNodeEmission'); emn.inputs['Color'].default_value = (0.9, 0.95, 1.0, 1); emn.inputs['Strength'].default_value = strength
    trn = nt.nodes.new('ShaderNodeBsdfTransparent'); mix = nt.nodes.new('ShaderNodeMixShader'); o_ = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(tc.outputs['UV'], mpn.inputs['Vector']); nt.links.new(mpn.outputs[0], vor.inputs['Vector'])
    nt.links.new(vor.outputs['Distance'], mr.inputs['Value']); nt.links.new(mr.outputs[0], mix.inputs['Fac'])
    nt.links.new(trn.outputs[0], mix.inputs[1]); nt.links.new(emn.outputs[0], mix.inputs[2]); nt.links.new(mix.outputs[0], o_.inputs[0])
    for f, val in ((0, 0.0), (NF - 1, speed * NF / FPS)):
        mpn.inputs['Location'].default_value[1] = val; mpn.inputs['Location'].keyframe_insert('default_value', index=1, frame=f)
        mpn.inputs['Location'].default_value[0] = -0.3 * val; mpn.inputs['Location'].keyframe_insert('default_value', index=0, frame=f)
    bpy.ops.mesh.primitive_plane_add(size=1.0); o = bpy.context.object; o.name = name; o.data.materials.append(m)
    o.visible_shadow = False; o.visible_glossy = False
    return o, dist
SNOW = [snow_layer(4.0, 9.0, 0.035, 'snow1', 1.2), snow_layer(7.0, 16.0, 0.025, 'snow2', 0.9)]

# ---------------- opening chandelier (the couple waits inside) + small chandeliers for the orchestra's first drops ----------------
c0 = EV['chandelier0']
chandelier((c0['x'], 0.0, c0['z'] + RP + 0.25), 1.15, name='chand0')
box((c0['x'], 0.0, c0['z'] - RP - 0.03), (2 * DXP + 0.5, 0.45, 0.04), M_GOLD, name='trapdoor0')   # floor that opens
hinge0 = link(bpy.data.objects.new('hinge0', None)); hinge0.location = (c0['x'] - DXP - 0.25, 0.0, c0['z'] - RP - 0.05)
door0 = [o for o in bpy.data.objects if o.name.startswith('trapdoor0')][0]; door0.parent = hinge0; door0.location = (DXP + 0.25, 0, 0.02)
ang = np.zeros(NF)
for f in range(NF):
    u = (T[f] - c0['t_release'] + 0.02) / 0.18
    if u > 0: ang[f] = min(1.0, u) ** 2 * 1.6
hinge0.rotation_mode = 'XYZ'; key_series(hinge0, 'rotation_euler', 1, ang)
for n in ('bass', 'ch2lo', 'ch3lo'):
    g = BP[n].seg[0]; p = np.array(g['p'])
    if n == 'bass': cxs = p[0]
    else: cxs = p[0] + LAY['dxc']
    chandelier((cxs, p[1], p[2] + 0.45), 0.7, bulbs=8, name='chandS')

# ---------------- crack: the shared key splits, a glowing crack runs over the ice, red's floe drifts away ----------------
cr = EV['crack']; t_cr = cr['t']; pc_ = np.array(cr['p'])
ck = [K for K in KEYS if K['kind'] == 'crack'][0]
ko = ck['o']
# after the couple has left (next landing), the two halves of the cracked key fall into the water
t_fall = t_cr + 0.2
halves = []
for side in (-1, 1):
    h_ = obj(prim('cube', M_KEY), 'half'); h_.matrix_world = ko.matrix_world.copy() @ Matrix.Translation((side * 0.25, 0, 0)) @ Matrix.Diagonal((0.48, 1, 1, 1))
    h_.color = ko.color; halves.append(h_)
    base = h_.location.copy(); P = np.zeros((NF, 3)); RT = np.zeros(NF)
    for f in range(NF):
        u = T[f] - t_fall
        P[f] = base if u < 0 else base + Vector((side * 0.3 * u, -0.2 * u, -0.5 * G * u * u))
        RT[f] = 0 if u < 0 else side * 2.0 * u
    key_loc(h_, P); h_.rotation_mode = 'XYZ'
    r0 = h_.rotation_euler.copy(); key_series(h_, 'rotation_euler', 1, r0[1] + RT)
    sc_ = np.where(T < t_fall + 0.8, 1.0, 0.001); key_series(h_, 'scale', 0, sc_ * h_.scale[0]); key_series(h_, 'scale', 1, sc_ * h_.scale[1]); key_series(h_, 'scale', 2, sc_ * h_.scale[2])
# hide the whole key once the halves take over (it flashed on the hit)
ko_sc = np.where(T < t_fall, 1.0, 0.001)
for k in range(3): key_series(ko, 'scale', k, ko_sc * ko.scale[k])
# glowing crack line on the ice floor, growing along +x from under the key, then a dark lead of water between the floes
M_CRACK = emis('crack', (0.5, 0.85, 1.0), 12.0)
xc0 = pc_[0]; zfl = lambda x: floor_z_at(x) + 0.005
pts = [(xc0 - 1.0, -0.62)]
x = xc0 - 1.0
while x < EV['sled']['p_stop'][0] + 4:
    x += rng.uniform(0.35, 0.7); pts.append((x, -0.62 + rng.uniform(-0.12, 0.12)))
crk = []
for k in range(len(pts) - 1):
    (x0, y0), (x1, y1) = pts[k], pts[k + 1]
    o = rod((x0, y0, zfl(x0)), (x1, y1, zfl(x1)), 0.03, M_CRACK, 6); crk.append((o, x0))
for o, x0 in crk:
    tg = t_cr + max(0.0, (x0 - xc0 + 1.0)) / 9.0
    s = np.where(T < tg, 0.001, 1.0); key_series(o, 'scale', 0, s * o.scale[0]); key_series(o, 'scale', 1, s * o.scale[1])
cbloom = sphere((pc_[0], pc_[1] - 0.45, pc_[2] - 0.1), 0.3, M_FX); cbloom.color = (0.6, 0.9, 1.0, 0)
key_series(cbloom, 'color', 3, 0.6 * pulse_alpha([t_cr], decay=3, length=12), 1e-3)
# red's floe: a slab under red's sinking keys, tilting toward the camera
fl_keys = sorted([K for K in KEYS if K['kind'] in ('half', 'floe') and K['top'][1] < -0.01], key=lambda K: K['top'][0])
if fl_keys:
    xs = [K['top'][0] for K in fl_keys]
    for K in fl_keys:
        c = K['top']; zf = floor_z_at(c[0])
    x0f, x1f = xs[0] - 0.8, EV['sled']['seat'][0] + 0.6
    zf0 = floor_z_at(x0f)
    o = box(((x0f + x1f) / 2, -1.25, zf0 - 0.05), (x1f - x0f, 1.6, 0.25), M_ICE, name='floe')
    o.rotation_euler = (math.radians(-5), math.radians(2), 0)

# ---------------- sled ride (red) ----------------
sl = EV['sled']; seat = np.array(sl['seat'])
M_RUN = mat('runner', (0.75, 0.75, 0.8), 0.2, **{'Metallic': 1.0})
sled = link(bpy.data.objects.new('sled', None))
box((0, 0, -RP - 0.04), (0.9, 0.55, 0.08), M_SLED, name='sledbody').parent = sled
box((-0.4, 0, -RP + 0.1), (0.08, 0.55, 0.3), M_SLED, name='sledback').parent = sled
for yy in (-0.24, 0.24):
    r_ = rod((-0.45, yy, -RP - 0.2), (0.45, yy, -RP - 0.2), 0.022, M_RUN, 8); r_.parent = sled
    r2 = rod((0.45, yy, -RP - 0.2), (0.6, yy, -RP + 0.0), 0.022, M_RUN, 8); r2.parent = sled
    for xx in (-0.25, 0.25):
        r3 = rod((xx, yy, -RP - 0.2), (xx, yy, -RP - 0.08), 0.018, M_RUN, 6); r3.parent = sled
SLB = []
for k, xx in enumerate((-0.3, -0.1, 0.1, 0.3)):
    b_ = sphere((xx, -0.3, -RP + 0.02), 0.045, M_GOLD); b_.parent = sled; SLB.append(b_)
# sled position: carries red from the seat until the stop; before = waiting at the top, after = parked at the snowbank
SP = np.zeros((NF, 3))
for f in range(NF):
    t = T[f]
    if t < sl['t_seat']: SP[f] = seat
    elif t <= sl['t_stop']: SP[f] = POS['red'][f]
    else: SP[f] = np.array(sl['p_stop'])
key_loc(sled, SP)
pit = np.zeros(NF)
for f in range(NF):
    u = T[f] - sl['t_seat']
    if 0 <= u <= sl['t_inc']: pit[f] = sl['theta']
    elif u > sl['t_inc']: pit[f] = sl['theta'] * max(0, 1 - (u - sl['t_inc']) / 0.25)
    if T[f] > sl['t_stop']: pit[f] = -0.12 * math.exp(-(T[f] - sl['t_stop']) / 0.25) * math.sin(14 * (T[f] - sl['t_stop']))
sled.rotation_mode = 'XYZ'; key_series(sled, 'rotation_euler', 1, pit)
for k, b_ in enumerate(SLB):                                  # jingle bells shake while moving
    dz = np.zeros(NF)
    for f in range(NF):
        if sl['t_seat'] < T[f] < sl['t_stop'] + 0.3: dz[f] = 0.025 * math.sin(40 * T[f] + k * 1.7)
    key_series(b_, 'location', 2, -RP + 0.02 + dz)
# chute: bed + side rails from the seat down the incline and along the run-out
def chute_pts(x0, x1, n=60):
    ps = []
    for x in np.linspace(x0, x1, n):
        dxs = x - seat[0]; Linc = sl['L1'] * math.cos(sl['theta'])
        z = seat[2] - dxs * math.tan(sl['theta']) if dxs <= Linc else seat[2] - sl['L1'] * math.sin(sl['theta'])
        ps.append((x, z))
    return ps
cps = chute_pts(seat[0] - 1.0, sl['p_stop'][0] + 0.6)
X_ = np.array([p[0] for p in cps]); Z_ = np.array([p[1] for p in cps]) - RP - 0.22
strip_mesh(X_, Z_, seat[1] - 0.45, seat[1] + 0.45, 0.08, M_SNOW, 'chute')
for yy in (seat[1] - 0.45, seat[1] + 0.45):
    strip_mesh(X_, Z_ + 0.14, yy - 0.03, yy + 0.03, 0.16, M_WOOD, 'rail')
for x, z in list(zip(X_, Z_))[::6]:
    rod((x, seat[1] - 0.4, z - 0.08), (x, seat[1] - 0.4, floor_z_at(x)), 0.04, M_WOOD, 6)
    rod((x, seat[1] + 0.4, z - 0.08), (x, seat[1] + 0.4, floor_z_at(x)), 0.04, M_WOOD, 6)
ps_ = np.array(sl['p_stop'])
bank = sphere((ps_[0] + 0.75, ps_[1], ps_[2] - RP - 0.1), 1.0, M_SNOW, (24, 12)); bank.scale = (0.45, 0.7, 0.45)
puff = sphere((ps_[0] + 0.5, ps_[1], ps_[2]), 0.5, M_FX); puff.color = (1, 1, 1, 0)
key_series(puff, 'color', 3, pulse_alpha([sl['t_stop']], decay=6, length=24, amp=0.7), 1e-3)
text('そり', (seat[0] + 0.1, seat[1] - 0.75, seat[2] + 0.75), 0.32, M_TXT_P)

# ---------------- revolving door (white) ----------------
dr = EV['door']; ax_ = np.array(dr['axis'])
zfloor = ax_[2] - RP
M_DOORF = mat('doorfloor', (0.75, 0.82, 0.95), 0.2, **{'Metallic': 0.5})
rod((ax_[0], ax_[1], zfloor - 0.08), (ax_[0], ax_[1], zfloor), 0.95, M_DOORF, 40)
rod((ax_[0], ax_[1], zfloor - 0.08), (ax_[0], ax_[1], floor_z_at(ax_[0])), 0.75, M_ICEPOST, 24)
rod((ax_[0], ax_[1], zfloor + 1.05), (ax_[0], ax_[1], zfloor + 1.15), 1.0, M_GOLD, 40)          # roof
o = sphere((ax_[0], ax_[1], zfloor + 1.15), 1.0, M_ICEPOST, (24, 12)); o.scale = (0.95, 0.95, 0.45)
bpy.ops.mesh.primitive_cylinder_add(radius=0.97, depth=1.05, vertices=48, location=(ax_[0], ax_[1], zfloor + 0.525))
gl = bpy.context.object; gl.data.materials.append(M_GLASS)
bm = bmesh.new(); bm.from_mesh(gl.data)                                  # openings at -x (entry) and +x (exit), caps removed
dele = [f for f in bm.faces if abs(f.normal.z) > 0.9 or abs(f.calc_center_median().y) < 0.5]
bmesh.ops.delete(bm, geom=dele, context='FACES'); bm.to_mesh(gl.data); bm.free()
gl.modifiers.new('s', 'SOLIDIFY').thickness = 0.02
rotor = link(bpy.data.objects.new('rotor', None)); rotor.location = (ax_[0], ax_[1], zfloor)
rod((0, 0, 0), (0, 0, 1.05), 0.05, M_GOLD, 12).parent = rotor
for k in range(4):
    ph = math.pi / 4 + k * math.pi / 2
    wg = box((0.47 * math.cos(ph), 0.47 * math.sin(ph), 0.5), (0.9, 0.03, 0.95), M_GLASS, rot=(0, 0, ph), name='wing'); wg.parent = rotor
    fr = box((0.47 * math.cos(ph), 0.47 * math.sin(ph), 0.5), (0.92, 0.05, 0.05), M_GOLD, rot=(0, 0, ph), name='wingframe'); fr.parent = rotor
    fr2 = box((0.93 * math.cos(ph), 0.93 * math.sin(ph), 0.5), (0.04, 0.05, 0.95), M_GOLD, rot=(0, 0, ph), name='wingedge'); fr2.parent = rotor
# rotor angle: the ball sits in the middle of a compartment (wings at ball angle +-45 deg); the door turns slowly before and
# after the ride at the same speed (it is motor driven)
rang = np.array([math.pi - dr['omega'] * (t - dr['t_in']) - math.pi / 4 - math.pi / 4 for t in T])
rotor.rotation_mode = 'XYZ'; key_series(rotor, 'rotation_euler', 2, rang)
text('回転扉', (ax_[0], ax_[1] - 1.05, zfloor + 1.55), 0.3, M_TXT_G)

# ---------------- ice slide + ice chimes (white) ----------------
sd = EV['slide']; p0s = np.array(sd['p0']); al = sd['alpha']; Ls = sd['L']
ca, sa = math.cos(al), math.sin(al)
ch_top = []
xs_ = np.linspace(p0s[0] - 0.2, p0s[0] + (Ls + 0.3) * ca, 50)
zs_ = p0s[2] - (xs_ - p0s[0]) * math.tan(al) - RP - 0.04
strip_mesh(xs_, zs_, -0.35, 0.35, 0.07, M_ICE, 'slidebed')
for yy in (-0.38, 0.38): strip_mesh(xs_, zs_ + 0.12, yy - 0.03, yy + 0.03, 0.16, M_ICEPOST, 'sliderail')
for x, z in list(zip(xs_, zs_))[::7]:
    rod((x, 0.0, z - 0.07), (x, 0.0, floor_z_at(x)), 0.05, M_ICEPOST, 8)
# arch beam above the slide, parallel to it; each chime hangs from it so its tip just touches the ball top at its note
beam_off = 1.05
bx = np.array([xs_[0], xs_[-1]]); bz = p0s[2] - (bx - p0s[0]) * math.tan(al) + beam_off
rod((bx[0], 0, bz[0]), (bx[1], 0, bz[1]), 0.045, M_GOLD, 8)
for xx in (bx[0], bx[1]):
    zz = p0s[2] - (xx - p0s[0]) * math.tan(al) + beam_off
    for yy in (-0.45, 0.45): rod((xx, yy, zz), (xx, yy, floor_z_at(xx)), 0.04, M_GOLD, 8)
    rod((xx, -0.45, zz), (xx, 0.45, zz), 0.04, M_GOLD, 8)
CHIME_O = []
for chn in sd['chimes']:
    p = np.array(chn['p']); tip = p + [0, 0, RP + 0.01]
    zb = p0s[2] - (p[0] - p0s[0]) * math.tan(al) + beam_off
    piv = link(bpy.data.objects.new('chpiv', None)); piv.location = (p[0], 0, zb)
    L_ = zb - tip[2]
    ic = obj(prim('icicle', M_TINK, 10), 'chime'); ic.parent = piv; ic.location = (0, 0, 0); ic.scale = (0.05, 0.05, L_)
    col = pcol(chn['pitch'], 0.35); ic.color = (*col, 0)
    add_piece(ic, 'chime', chn['t'], tip, [0, 0, -1], 0.05, 0.05, 0.01, 'white', col, flash=False)
    fh = int(round(chn['t'] * FPS))
    ic.color = (*col, 0); ic.keyframe_insert('color', index=3, frame=max(0, fh - 1))
    for df, a in ((0, 1.0), (5, 0.5), (20, 0.12)):
        ic.color = (*col, a); ic.keyframe_insert('color', index=3, frame=fh + df)
    sw = np.zeros(NF)
    for f in range(NF):
        u = T[f] - chn['t']
        if u > 0: sw[f] = -0.35 * math.exp(-u / 0.6) * math.sin(9 * u)
    piv.rotation_mode = 'XYZ'; key_series(piv, 'rotation_euler', 1, sw)
    CHIME_O.append(ic)
text('すべり台', (p0s[0] + 1.5, -0.6, p0s[2] + 1.6), 0.3, M_TXT_P)

# ---------------- reunion sparkle ----------------
ru = EV['reunion']
for k in range(24):
    sp_ = sphere((0, 0, 0), 0.05, M_FX); sp_.color = (*([(1, 0.5, 0.7), (1, 0.9, 0.5), (0.7, 0.9, 1)][k % 3]), 0)
    a0 = 2 * math.pi * k / 24; P = np.zeros((NF, 3)); A = np.zeros(NF)
    for f in range(NF):
        u = T[f] - ru['t']
        P[f] = (ru['x'] + (0.25 + 1.6 * (1 - math.exp(-u / 0.3))) * math.cos(a0) * (u > 0), -0.3, ru['z'] + 0.2 + (0.25 + 1.6 * (1 - math.exp(-u / 0.3))) * math.sin(a0) * (u > 0))
        A[f] = 0 if u < 0 else math.exp(-u / 0.5)
    key_loc(sp_, P); key_series(sp_, 'color', 3, A, 1e-3)

# ---------------- finale: see-saw, cups, the gold ball, the great chandelier bell ----------------
ss = EV['seesaw']; piv_ = np.array(ss['pivot']); ARM = ss['arm']; TH0 = ss['th0']; cupc = np.array(ss['cup'])
DYP = LAY['dyp']
sw_ = link(bpy.data.objects.new('seesaw', None)); sw_.location = tuple(piv_)
pl = box((0, 0, -0.03), (0.5, 2 * ARM + 0.2, 0.06), M_WOOD, name='plank'); pl.parent = sw_
box((0, ARM, 0.03), (0.6, 0.6, 0.06), M_GOLD, name='farpad').parent = sw_
bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=0.35, radius2=0.03, depth=0.5, location=(piv_[0], piv_[1], piv_[2] - 0.28))
bpy.context.object.data.materials.append(M_STEEL)
rod((piv_[0], piv_[1], piv_[2] - 0.5), (piv_[0], piv_[1], floor_z_at(piv_[0])), 0.12, M_POST, 12)
tl_ = ss['t_launch']; swa = np.zeros(NF)
for f in range(NF):
    u = (T[f] - tl_) / ss['t_swing']
    swa[f] = TH0 if u <= 0 else TH0 - 2 * TH0 * min(1.0, u)          # near end (y = 0) down before, up after
sw_.rotation_mode = 'XYZ'; key_series(sw_, 'rotation_euler', 0, swa)
# launch platform: a gold disc on two vertical guide rails carrying both cups; the plank's near end pushes it straight up
zplat0 = cupc[2] - RP - 0.06
plat = link(bpy.data.objects.new('platform', None)); plat.location = (cupc[0], 0, zplat0)
box((0, 0, -0.03), (1.15, 0.85, 0.06), M_GOLD, name='platdisc').parent = plat
for side, (dx, dy) in (('red', (DXP, -DYP)), ('white', (-DXP, DYP))):
    cup = obj(prim('bell', M_TINK, 24), 'cup'); cup.parent = plat
    cup.location = (dx, dy, 0.0); cup.scale = (0.28, 0.28, -0.2)
    col = pcol(LEAD[137][1][0], 0.3); fh = int(round(TN[137] * FPS))
    for df, a in ((-1, 0), (0, 1.0), (5, 0.5), (24, 0.15)):
        cup.color = (*col, a); cup.keyframe_insert('color', index=3, frame=max(0, fh + df))
rise = 2 * ARM * math.sin(TH0)
pz = np.array([zplat0 + rise * min(1.0, max(0.0, (T[f] - tl_) / ss['t_swing'])) for f in range(NF)])
key_series(plat, 'location', 2, pz)
for gx in (-0.62, 0.62):
    rod((cupc[0] + gx, 0, zplat0 - 0.3), (cupc[0] + gx, 0, zplat0 + rise + 0.25), 0.03, M_STEEL, 8)
text('シーソー', (piv_[0] + 1.0, piv_[1] - 0.3, piv_[2] - 0.55), 0.24, M_TXT_G)
# great chandelier + bell above the cups
bbx = EV['bigbell']; bz0 = bbx['z']
gb = obj(prim('bell', M_BELL, 48), 'bigbell')
gb.matrix_world = Matrix.Translation(Vector((bbx['x'], 0, bz0 + 0.95))) @ Matrix.Diagonal((0.8, 0.8, 0.95, 1))
# the ball tops touch the inside of the crown: crown inner surface at bz0 (the shell is 0.1 x 0.95 thick)
gb.color = (1.0, 0.75, 0.35, 0)
key_series(gb, 'color', 3, np.maximum(0.06, pulse_alpha([FINAL], decay=40, length=150)), 1e-3)
add_piece(gb, 'bigbell', FINAL, (bbx['x'], 0, bz0), [0, 0, -1], 0.35, 0.35, 0.05, 'pair', (1, 0.8, 0.4), flash=False)
chandelier((bbx['x'], 0, bz0 + 1.25), 2.0, bulbs=16, name='grand')
M_GRAND = emis('grandglow', (1.0, 0.8, 0.5), 1.0); ge = M_GRAND.node_tree.nodes['Emission']
gglow = pulse_alpha([FINAL], decay=60, length=200)
for f in range(0, NF, 2):
    v = float(1.5 + 30 * gglow[f])
    ge.inputs['Strength'].default_value = v; ge.inputs['Strength'].keyframe_insert('default_value', frame=f)
for j in range(24):
    ph = 2 * math.pi * j / 24
    o = sphere((bbx['x'] + 1.6 * math.cos(ph), 1.6 * math.sin(ph), bz0 + 1.0 + 0.3 * math.sin(3 * ph)), 0.08, M_GRAND, (8, 6))
gl_light = bpy.data.lights.new('grandlight', 'POINT'); gl_light.color = (1.0, 0.8, 0.55); gl_light.shadow_soft_size = 0.6
glo = link(bpy.data.objects.new('grandlight', gl_light)); glo.location = (bbx['x'], -0.5, bz0 + 0.4)
for f in range(0, NF, 2):
    gl_light.energy = float(30 + 1800 * gglow[f]); gl_light.keyframe_insert('energy', frame=f)
# final burst of ice-crystal confetti
for k in range(110):
    cf = obj(prim('plane', M_LAMP), 'conf'); cf.color = (*LAMPC[k % 4], 1)
    a0 = rng.uniform(0, 2 * math.pi); spd = rng.uniform(1.0, 3.2); vy = rng.uniform(-1.5, 0.5)
    P = np.zeros((NF, 3)); RT = np.zeros((NF, 3))
    for f in range(NF):
        s_ = T[f] - FINAL
        if s_ < 0: P[f] = (bbx['x'], 0, bz0); continue
        burst = 1 - math.exp(-s_ / 0.3)
        P[f] = (bbx['x'] + spd * math.cos(a0) * 1.2 * burst + 0.15 * math.sin(3 * s_ + k), vy * burst - 0.4, bz0 + spd * math.sin(a0) * 0.8 * burst - 0.55 * s_)
        RT[f] = (s_ * (3 + k % 5), s_ * (2 + k % 3), s_ * 1.5)
    key_loc(cf, P)
    sc_ = np.where(T >= FINAL, 1.0, 0.001)
    for c_ in range(3): key_series(cf, 'scale', c_, sc_ * 0.07); key_series(cf, 'rotation_euler', c_, RT[:, c_])

# ---------------- signs (keywords for search, readable on a phone) ----------------
def sign(lines, x, y, z, w, h, m_txt=M_TXT, plate=True):
    if plate:
        box((x, y + 0.08, z), (w, 0.12, h), M_PLATE, rot=(-0.3, 0, 0), name='plate')
        for sgn in (1, -1): box((x, y + 0.08 + 0.2955 * sgn * h / 2, z + 0.955 * sgn * h / 2), (w + 0.1, 0.16, 0.07), M_GOLD, rot=(-0.3, 0, 0))
        for xx in (x - w / 2 + 0.2, x + w / 2 - 0.2): rod((xx, y + 0.1, z - h / 2), (xx, y + 0.1, floor_z_at(xx)), 0.05, M_GOLD, 8)
    zz = z + h / 2
    for s_, size, mt in lines:
        zz -= size * 1.25
        text(s_, (x, y - 0.03 + 0.31 * (zz + size * 0.15 - z), zz + size * 0.15), size, mt)
x_ = lambda t: float(np.interp(t, T, (POS['red'][:, 0] + POS['white'][:, 0]) / 2))
z_ = lambda t: float(np.interp(t, T, (POS['red'][:, 2] + POS['white'][:, 2]) / 2))
sign([('美しく青きドナウ', 0.66, M_TXT_G), ('ヨハン・シュトラウス2世（1867）', 0.31, M_TXT), ('ワルツ × ビー玉からくり', 0.38, M_TXT_P)],
     x_(1.2) + 0.7, -1.9, z_(1.2) - 0.75, 5.6, 2.0)
sign([('ズン（1拍目）', 0.26, M_TXT_G)], BP['bass'].hits[0]['p'][0] + 0.9, LY['bass'] - 0.45, BP['bass'].hits[0]['p'][2] + 0.25, 1.6, 0.36, plate=False)
sign([('チャッ（2拍目）', 0.26, M_TXT)], BP['ch2lo'].hits[0]['p'][0] - 0.2, LY['ch2'] - 0.45, BP['ch2lo'].hits[0]['p'][2] + 0.45, 1.6, 0.36, plate=False)
sign([('チャッ（3拍目）', 0.26, M_TXT_P)], BP['ch3lo'].hits[0]['p'][0] + 1.4, LY['ch3'] - 0.45, BP['ch3lo'].hits[0]['p'][2] + 0.6, 1.6, 0.36, plate=False)
sign([('ワルツは3拍子', 0.36, M_TXT_G), ('ズン・チャッ・チャッ', 0.3, M_TXT)], x_(8.5) + 1.5, -1.7, z_(8.5) - 1.0, 3.6, 1.15)
sign([('氷上の舞踏会', 0.42, M_TXT_P), ('凍ったドナウ川', 0.28, M_TXT)], x_(15.5) + 1.5, -1.7, z_(15.5) - 1.0, 3.4, 1.15)
sign([('クラシック名曲', 0.36, M_TXT_G), ('美しく青きドナウ', 0.3, M_TXT)], x_(22.5) + 1.5, -1.7, z_(22.5) - 1.0, 3.6, 1.15)
sign([('ヨハン・シュトラウス2世', 0.3, M_TXT_G), ('ウィンナ・ワルツ', 0.3, M_TXT)], x_(28.0) + 1.5, -1.7, z_(28.0) - 1.4, 4.0, 1.15)

# ---------------- camera: follow the couple, widen while they are apart ----------------
def smooth(Xa, k):
    Y = Xa.copy(); a = 1 - math.exp(-1 / FPS / k)
    for f in range(1, len(Xa)): Y[f] = Y[f - 1] + (Xa[f] - Y[f - 1]) * a
    for f in range(len(Xa) - 2, -1, -1): Y[f] = Y[f + 1] + (Y[f] - Y[f + 1]) * a
    return Y
FOC = np.zeros((NF, 3)); WID = np.zeros(NF)
rP, wP = POS['red'], POS['white']
for f in range(NF):
    t = T[f]; r_, w_ = rP[f], wP[f]
    if t < E['t0'] + 0.3:
        FOC[f] = (r_ + w_) / 2 + np.array([0.6, 0, 1.6]); WID[f] = 6.2
    elif t > FINAL - 0.5:
        FOC[f] = np.array([bbx['x'], 0, (bz0 + EV['seesaw']['cup'][2]) / 2 + 0.2]); WID[f] = 7.6
    else:
        FOC[f] = (r_ + w_) / 2 + np.array([0.9, 0, 1.9])
        WID[f] = max(abs(r_[0] - w_[0]) + 4.0, abs(r_[2] - w_[2]) * 9 / 16 * 1.2 + 5.0, 5.8)
WID = np.clip(WID, 5.8, 9.5)
FOC[:, 0] = smooth(FOC[:, 0], 0.45); FOC[:, 2] = smooth(FOC[:, 2], 0.6); WID = smooth(WID, 0.7)
LENS = 32.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 500
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24
    CP[f] = (FOC[f, 0], -D, FOC[f, 2] + 0.34 * D)
    TP[f] = (FOC[f, 0], 0.0, FOC[f, 2] + 0.2)
key_loc(cam, CP); key_loc(tgt, TP)
# ballroom spotlights: one on each dancer, from high in front
for n, col in (('red', (1.0, 0.85, 0.75)), ('white', (0.8, 0.9, 1.0))):
    spd = bpy.data.lights.new('spot' + n, 'SPOT'); spd.energy = 2500; spd.spot_size = math.radians(16); spd.spot_blend = 0.6
    spd.color = col; spd.shadow_soft_size = 0.3
    so = link(bpy.data.objects.new('spot' + n, spd))
    stg = link(bpy.data.objects.new('spottgt' + n, None))
    P = POS[n].copy(); Ps = P.copy()
    for k in range(3): Ps[:, k] = smooth(P[:, k], 0.25)
    key_loc(stg, Ps - np.array([0, 0, 0.25])); key_loc(so, Ps + np.array([-2.0, -6.0, 9.0]))
    c_ = so.constraints.new('TRACK_TO'); c_.target = stg; c_.track_axis = 'TRACK_NEGATIVE_Z'; c_.up_axis = 'UP_Y'
# snow layers ride with the camera
for o, dist in SNOW:
    o.parent = cam; o.location = (0, 0, -dist); o.rotation_euler = (0, 0, 0)
    hh = dist * 24 / LENS; o.scale = (hh * 9 / 16 * 1.1, hh * 1.1, 1)

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
              'red', np.round(POS['red'][f], 2), 'white', np.round(POS['white'][f], 2))
if MODE == 'who':
    import bpy_extras
    f = int(round(float(args[1]) * FPS)); sc.frame_set(f); dg = bpy.context.evaluated_depsgraph_get()
    from bpy_extras.object_utils import world_to_camera_view
    big = []
    for o in bpy.data.objects:
        if o.type not in ('MESH', 'CURVE', 'FONT'): continue
        bb = [o.matrix_world @ Vector(c) for c in o.bound_box]
        uv = [world_to_camera_view(sc, cam, p) for p in bb]
        xs = [u.x for u in uv]; ys = [u.y for u in uv]; zs = [u.z for u in uv]
        if max(zs) < 0: continue
        x0, x1 = max(0, min(xs)), min(1, max(xs)); y0, y1 = max(0, min(ys)), min(1, max(ys))
        if x1 > x0 and y1 > y0: big.append(((x1 - x0) * (y1 - y0), o.name, round(x0, 2), round(x1, 2), round(y0, 2), round(y1, 2)))
    for b in sorted(big, reverse=True)[:15]: print('WHO', b)
