"""Gymnopedie No.1 - 'The Music Box Tower' (オルゴールの塔). A glass music-box tower at night; the camera circles it with the
marbles, so the tower seems to turn slowly like a music box.
Every melody note = the PEARL landing on a glass key of that pitch (outer lane); every bass note = the NIGHT ball landing on
a bass bar (inner lane); every chord = the plunger that the bass bar kicks up falling back onto its chime crown one beat later.
Struck keys stay lit; on the final chord the whole tower flares up, then the lift carries both marbles back to the crown
and the lights go out behind it -> the last frame is the first frame (loop).  layout.py plans all motion.
jp_sub.otf = Noto Sans CJK JP Black (SIL OFL) subset (make_font.py); en_bold.ttf = DejaVu Sans Bold (DejaVu font license).
python3 scene.py -- test t1 t2 ... | range f0 f1 | probe t ..."""
import os, sys, json, math, time
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json')); NIGHT = E['night']
LAY = json.load(open(f'{HERE}/layout.json'))
FPS = 30; FINAL = E['final']; T_END = LAY['ev']['t_end']; NF = int(round(T_END * FPS)); T = np.arange(NF) / FPS
G = LAY['G']; EV = LAY['ev']; GEO = LAY['geo']
T0, BAR, BEAT = E['t0'], E['bar'], E['beat']
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
sc.cycles.max_bounces = 4; sc.cycles.diffuse_bounces = 1; sc.cycles.glossy_bounces = 2; sc.cycles.transmission_bounces = 3
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


def objcol(name, rough=0.45, metal=0.0, emit=0.0, coat=0.0):
    m = mat(name, (1, 1, 1), rough, **{'Metallic': metal, 'Coat Weight': coat}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
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


def key_scale(o, s, base=(1, 1, 1)):
    for k in range(3): key_series(o, 'scale', k, np.asarray(s) * base[k], 1e-4)


def key_val(sock, vals, step=2):
    for f in range(0, NF, step):
        sock.default_value = float(vals[f]); sock.keyframe_insert('default_value', frame=f)


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
    elif kind == 'ico': bmesh.ops.create_icosphere(bm, subdivisions=a[0], radius=1.0)
    elif kind == 'plane': bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    elif kind == 'bowl':                                    # open bowl, rim radius 1 at z=0, depth 0.64
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
    elif kind == 'flower':                                 # 5 petals + centre, petals in the xy plane, radius ~1
        for k in range(5):
            ph = 2 * math.pi * k / 5
            g = bmesh.ops.create_uvsphere(bm, u_segments=10, v_segments=6, radius=1.0)
            vs = g['verts']
            bmesh.ops.scale(bm, vec=(0.55, 0.36, 0.1), verts=vs)
            bmesh.ops.translate(bm, vec=(0.5, 0, 0.02), verts=vs)
            bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(ph, 3, 'Z'), verts=vs)
    elif kind == 'fcentre':
        bmesh.ops.create_uvsphere(bm, u_segments=10, v_segments=6, radius=1.0)
        bmesh.ops.scale(bm, vec=(0.28, 0.28, 0.16), verts=bm.verts); bmesh.ops.translate(bm, vec=(0, 0, 0.06), verts=bm.verts)
    elif kind == 'leaf':                                   # flat leaf, long axis x, from x = 0 to 1
        bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=6, radius=1.0)
        bmesh.ops.scale(bm, vec=(0.5, 0.26, 0.04), verts=bm.verts); bmesh.ops.translate(bm, vec=(0.5, 0, 0), verts=bm.verts)
    me = bpy.data.meshes.new(kind); bm.to_mesh(me); bm.free()
    if kind not in ('cube', 'plane'):
        for p in me.polygons: p.use_smooth = True
    if m: me.materials.append(m)
    _MC[key] = me; return me


def obj(me, name='o'): return link(bpy.data.objects.new(name, me))


def box(c, size, m, rot=None, name='box', kind='cube'):
    o = obj(prim(kind, m), name); o.location = c; o.scale = size
    if rot is not None: o.rotation_euler = rot
    return o


def rod(p0, p1, r, m, verts=10, r1=None):
    p0, p1 = Vector(p0), Vector(p1); d = p1 - p0
    me = prim('cyl', m, verts) if r1 is None else prim('cone', m, verts, 1.0, r1 / r, 1.0)
    o = obj(me, 'rod'); q = Vector((0, 0, 1)).rotation_difference(d.normalized() if d.length > 1e-9 else Vector((0, 0, 1)))
    o.matrix_world = Matrix.Translation((p0 + p1) / 2) @ q.to_matrix().to_4x4() @ Matrix.Diagonal((r, r, max(1e-4, d.length), 1))
    return o


def sphere(c, r, m, seg=(16, 8), name='sph'):
    o = obj(prim('sph', m, *seg), name); o.location = c; o.scale = (r, r, r); return o


def text(s, loc, size, m, rot=(math.pi / 2 - 0.25, 0, 0), align='CENTER', ext=0.02):
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


def smoothstep(x): x = min(1.0, max(0.0, x)); return x * x * (3 - 2 * x)


def pop_scale(t_in, t_out=None, grow=0.3, shrink=0.35, over=1.15):
    """0 -> 1 (with overshoot) from t_in; back to 0 after t_out"""
    s_ = np.zeros(NF)
    for f in range(NF):
        t = T[f]; u1 = (t - t_in) / grow
        if u1 <= 0: a = 0.0
        elif u1 < 0.7: a = over * math.sin(u1 / 0.7 * math.pi / 2)
        elif u1 < 1: a = over - (over - 1) * (u1 - 0.7) / 0.3
        else: a = 1.0
        b = 1.0 if t_out is None else 1.0 - min(1, max(0, (t - t_out) / shrink)) ** 1.5
        s_[f] = max(0.001, a * b)
    return s_


def popper(objs, anchor, t_in, t_out=None, grow=0.3, shrink=0.35):
    anchor = Vector(anchor)
    e = link(bpy.data.objects.new('pop', None)); e.location = anchor
    for o in objs: o.parent = e; o.matrix_parent_inverse = Matrix.Translation(-anchor)
    key_scale(e, pop_scale(t_in, t_out, grow, shrink))
    return e






def tube(pts, r, m, name='tube', seg=8):
    """polyline tube (curve with bevel)"""
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'; cu.bevel_depth = r; cu.bevel_resolution = 2
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for k, p in enumerate(pts): sp.points[k].co = (*p, 1)
    o = link(bpy.data.objects.new(name, cu)); o.data.materials.append(m); return o


def smooth(Xa, k):
    Y = np.array(Xa, float).copy(); a = 1 - math.exp(-1 / FPS / k)
    for f in range(1, len(Y)): Y[f] = Y[f - 1] + (Y[f] - Y[f - 1]) * a
    for f in range(len(Y) - 2, -1, -1): Y[f] = Y[f + 1] + (Y[f] - Y[f + 1]) * a
    return Y


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


PITCH_COL = [(1.0, 0.12, 0.12), (1.0, 0.35, 0.05), (1.0, 0.6, 0.05), (1.0, 0.85, 0.1), (0.55, 1.0, 0.1), (0.1, 1.0, 0.3),
             (0.05, 1.0, 0.75), (0.05, 0.7, 1.0), (0.15, 0.35, 1.0), (0.45, 0.2, 1.0), (0.75, 0.2, 1.0), (1.0, 0.2, 0.65)]
def pcol(p, light=0.0):
    c = PITCH_COL[p % 12]; return tuple(min(1, x * (1 - light) + light) for x in c)


# ================================================================ motion from layout.json
BP = {n: BallPath(d) for n, d in LAY['balls'].items()}
POS = {n: np.array([b.at(t) for t in T]) for n, b in BP.items()}
RB = {n: d['r'] for n, d in LAY['balls'].items()}
PITCH = GEO['pitch']; Z_OFF = GEO.get('z_off', -3.0); TM = GEO['tm']; W_C = GEO['w_c']
def thn(t): return 0.05 + 0.30 * max(0.0, t - T0) / (TM - T0) if t <= TM else 0.35 + W_C * (t - TM)
def zc(th): return Z_OFF - PITCH * th / (2 * math.pi)
LIFT = EV['lift']
def sstep(u): u = min(1.0, max(0.0, u)); return u * u * (3 - 2 * u)
def trap(L, Tt, ta):
    vc = L / (Tt - ta)
    def s(u):
        if u <= 0: return 0.0
        if u >= Tt: return L
        if u < ta: return 0.5 * vc / ta * u * u
        if u > Tt - ta: w = Tt - u; return L - 0.5 * vc / ta * w * w
        return 0.5 * vc * ta + vc * (u - ta)
    return s
UP_S = trap(LIFT['z_h'] - LIFT['z_end'], LIFT['up'][1] - LIFT['up'][0], 1.6)
def z_follow(t): return zc(thn(t)) - 1.0
def z_car(t):
    Zh, Ze = LIFT['z_h'], LIFT['z_end']; s0, s1 = LIFT['sink']
    if t <= s0: return Zh
    if t <= FINAL - 3.0: return Zh + (z_follow(t) - Zh) * sstep((t - s0) / (s1 - s0))
    if t <= FINAL: zf = z_follow(FINAL - 3.0); return zf + (Ze - zf) * sstep((t - (FINAL - 3.0)) / 2.6)
    if t <= LIFT['up'][0]: return Ze
    return Ze + UP_S(t - LIFT['up'][0])
ZCAR = np.array([z_car(t) for t in T])
# chord (struck pitch) of every hit, for colours
def cyl(th, r, z): return np.array([r * math.cos(th), r * math.sin(th), z])
def az(p): return math.atan2(p[1], p[0])
def face_out(th, tilt=0.0):
    """rotation so that a text / plate in the XY plane faces radially outwards at azimuth th"""
    return (math.pi / 2 - tilt, 0, th + math.pi / 2)
Z_BOT = LIFT['z_end'] - RB['pearl'] - 0.35           # lift floor at the bottom
R_CORE = 1.6

# ================================================================ materials
M_PEARL = mat('pearl', (0.95, 0.93, 0.9), 0.12, **{'Coat Weight': 1.0, 'Sheen Weight': 0.6, 'Emission Color': (0.8, 0.85, 1.0), 'Emission Strength': 0.35})
M_NIGHTB = mat('nightb', (0.05, 0.12, 0.6), 0.05, **{'Coat Weight': 1.0, 'Emission Color': (0.15, 0.3, 1.0), 'Emission Strength': 0.6})
M_KEY = glowmat('key', gain=10.0, base=0.25, rough=0.1, coat=1.0)
M_BELL = glowmat('bell', gain=7.0, base=0.15, rough=0.25, metal=0.8)
M_GOLD = mat('gold', (1.0, 0.72, 0.3), 0.25, **{'Metallic': 1.0})
M_GOLD2 = mat('gold2', (0.9, 0.6, 0.22), 0.4, **{'Metallic': 1.0})
M_WOOD = mat('wood', (0.22, 0.08, 0.05), 0.35, **{'Coat Weight': 0.8})
M_GLASS = mat('glass', (0.75, 0.85, 1.0), 0.02, **{'Transmission Weight': 1.0, 'IOR': 1.2})
M_DARK = mat('dark', (0.02, 0.02, 0.04), 0.5)
M_FELT = mat('felt', (0.5, 0.05, 0.08), 0.9)
M_CREAM = mat('cream', (0.92, 0.88, 0.8), 0.35, **{'Coat Weight': 0.6})
M_OBJ = objcol('objcol', 0.4)
M_LAMP = glowmat('lamp', gain=14.0, base=0.0, rough=0.3, coat=0.0)
M_FX = fxmat('fx', 8.0)
M_WATER = mat('water', (0.01, 0.015, 0.04), 0.03, **{'Coat Weight': 1.0})
M_TXT = emis('txtw', (1.0, 0.95, 0.88), 3.0); M_TXT_G = emis('txtg', (1.0, 0.75, 0.3), 5.0); M_TXT_B = emis('txtb', (0.6, 0.8, 1.0), 4.0)
M_PLATE = mat('plate', (0.03, 0.03, 0.08), 0.3, **{'Coat Weight': 1.0})
M_LANTERN = emis('lantern', (1.0, 0.7, 0.35), 2.2)
M_MOON = glowdisc('moonglow', 4.0)

# ---------------- glow (lit) state of every key: flash at the hit, stays lit, flares at the final, dark again behind the
# rising lift (the music box rewinds)
T_FLARE = FINAL + 0.25
def lit_curve(t_hit, z):
    a = np.zeros(NF)
    u = T - t_hit
    on = u >= 0
    a[on] = 0.32 + 0.68 * np.exp(-u[on] / 0.18)
    # final flare: a wave from the foot to the crown (1.2 s)
    tf = T_FLARE + 1.2 * min(1, max(0, (z - LIFT['z_end']) / (LIFT['z_h'] - LIFT['z_end'])))
    w = T - tf
    a = np.where(w >= 0, np.maximum(a, 0.55 + 0.45 * np.exp(-w / 0.4)), a)
    # lights out when the lift passes on its way up (keys above the lift stay lit until it reaches them)
    off = np.array([1.0 if ZCAR[f] < z - 0.5 or T[f] < LIFT['up'][0] else max(0.0, 1 - (ZCAR[f] - z + 0.5) / 2.0) for f in range(NF)])
    return a * off


def key_alpha(o, a):
    key_series(o, 'color', 3, a, 2e-3)


PITCH_C = {}
def pc(p, light=0.0): return pcol(p, light)


def kframe(top, n):
    n = np.array(n, float); n /= np.linalg.norm(n); th = az(top)
    tg = np.array([-math.sin(th), math.cos(th), 0.0]); xa = tg - (tg @ n) * n; xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return Matrix(((xa[0], ya[0], n[0]), (xa[1], ya[1], n[1]), (xa[2], ya[2], n[2])))


def place(o, ctr, M, scl):
    o.matrix_world = Matrix.Translation(Vector(ctr)) @ M.to_4x4() @ Matrix.Diagonal((*scl, 1))


def bump(o, t, nv, d=0.035):
    fh = int(round(t * FPS)); base = o.location.copy(); nv = Vector(nv)
    for df, dd in ((-1, 0), (1, -d), (4, d * 0.3), (9, 0)):
        o.location = base + nv * dd; o.keyframe_insert('location', frame=max(0, fh + df))


RING = []                                            # (t, position, colour) hit rings
LIGHTS = []                                          # every sound strike: time (counter)

# ================================================================ balls
BALL = {}
for n, b in BP.items():
    rr = b.r
    bpy.ops.mesh.primitive_uv_sphere_add(radius=rr, segments=32, ring_count=16); o = bpy.context.object; o.name = n
    for p in o.data.polygons: p.use_smooth = True
    o.data.materials.append(M_PEARL if n == 'pearl' else M_NIGHTB)
    if n == 'night':
        bpy.ops.mesh.primitive_torus_add(major_radius=rr * 1.0, minor_radius=rr * 0.09, major_segments=32, minor_segments=8)
        hb = bpy.context.object; hb.parent = o; hb.data.materials.append(M_GOLD)
    key_loc(o, POS[n])
    sp = np.zeros(NF)
    for f in range(1, NF):
        d = POS[n][f] - POS[n][f - 1]; sp[f] = sp[f - 1] + math.hypot(d[0], d[1]) / rr
    o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, sp)
    BALL[n] = o
for n, col, en in (('pearl', (0.85, 0.9, 1.0), 30), ('night', (0.35, 0.5, 1.0), 30)):
    ld = bpy.data.lights.new('bl' + n, 'POINT'); ld.color = col; ld.energy = en; ld.shadow_soft_size = 0.2
    lo = link(bpy.data.objects.new('bl' + n, ld)); lo.location = (0, 0, 0)
    P_ = POS[n].copy(); rad = np.hypot(P_[:, 0], P_[:, 1]) + 1e-6
    P_[:, 0] *= (rad + 0.6) / rad; P_[:, 1] *= (rad + 0.6) / rad; P_[:, 2] += 0.4
    key_loc(lo, P_)

# ================================================================ pearl keys (outer lane)
EXT = LAY['ext']
for h in BP['pearl'].hits:
    if h['kind'] not in ('key', 'lever', 'chord'): continue
    nv = np.array(h['n']); top = np.array(h['p']) - RB['pearl'] * nv; M = kframe(top, nv)
    ex = EXT['pearl']; th = 0.12
    ctr = top - nv * th / 2
    o = obj(prim('bcube', M_KEY), 'key'); place(o, ctr, M, (ex[0] * 2 - 0.04, ex[1] * 2 - 0.04, th))
    col = pc(h['pitch'][0]) if h['kind'] != 'chord' else (1.0, 0.85, 0.5); o.color = (*col, 0)
    key_alpha(o, lit_curve(h['t'], top[2])); bump(o, h['t'], nv)
    # gold frame + cantilever arm to the core
    fr = obj(prim('cube', M_GOLD2), 'kfr'); place(fr, ctr - nv * 0.09, M, (ex[0] * 2 + 0.02, ex[1] * 2 - 0.2, 0.04))
    th_ = az(top)
    rod(top - nv * 0.12, cyl(th_, R_CORE, top[2] - 0.35), 0.035, M_GOLD2, 8)
    if h['kind'] == 'chord':                         # chord plate: four chimes hang under the plate
        for j, p in enumerate(h['pitch']):
            off = (j - 1.5) * 0.12
            c = top + Vector((0, 0, 0)).to_tuple() if False else top
            q = c + np.array(M.col[0]) * off - nv * 0.2
            ch = obj(prim('cyl', M_BELL, 12), 'chime'); ch.location = Vector(q - [0, 0, 0.18]); ch.scale = (0.04, 0.04, 0.36)
            cc = pc(p); ch.color = (*cc, 0); key_alpha(ch, lit_curve(h['t'], top[2]))
    if h['kind'] == 'lever':                         # lever key: a little gold flag shows it
        fl = rod(top + np.array(M.col[1]) * 0.25, top + np.array(M.col[1]) * 0.25 + [0, 0, 0.45], 0.012, M_GOLD, 6)
        fg = box(tuple(top + np.array(M.col[1]) * 0.25 + [0, 0, 0.38]), (0.02, 0.18, 0.11), M_FELT, rot=(0, 0, th_), name='flag')
    RING.append((h['t'], top, pc(h['pitch'][0], 0.3)))
LIGHTS += [l[0] for l in LEAD]

# ================================================================ night bass bars + plungers (hammers) + chime crowns
HAM = EV['hammers']
nh = {h['ev']: h for h in BP['night'].hits if h.get('ev') is not None}
first_ib = min(hm['ib'] for hm in HAM)
def plunger_xy(top, k=0, nk=1):
    th = az(top); r = math.hypot(top[0], top[1]) - 0.62
    return cyl(th + (k - (nk - 1) / 2) * 0.36 / r, r, 0)[:2]
for h in BP['night'].hits:
    if h['kind'] not in ('bass', 'lever'): continue
    nv = np.array(h['n']); top = np.array(h['p']) - RB['night'] * nv; M = kframe(top, nv)
    ex = EXT['night']; th = 0.15
    o = obj(prim('bcube', M_WOOD), 'bass'); place(o, top - nv * th / 2, M, (ex[0] * 2 - 0.04, ex[1] * 2 - 0.04, th))
    gl = obj(prim('bcube', M_KEY), 'bassglow'); place(gl, top - nv * 0.01, M, (ex[0] * 2 - 0.16, ex[1] * 2 - 0.16, 0.03))
    col = pc(min(h['pitch'])); gl.color = (*col, 0); key_alpha(gl, lit_curve(h['t'], top[2]))
    bump(o, h['t'], nv); bump(gl, h['t'], nv)
    th_ = az(top)
    rod(top - nv * 0.15, cyl(th_, R_CORE, top[2] - 0.4), 0.04, M_GOLD2, 8)
    RING.append((h['t'], top, pc(min(h['pitch']), 0.3)))
LIGHTS += [l[0] for l in NIGHT]
byib = {}
for hm in HAM: byib.setdefault(hm['ib'], []).append(hm)
for ib, hms in byib.items():
    hb = nh[ib]; nv = np.array(hb['n']); top = np.array(hb['p']) - RB['night'] * nv
    for k, hm in enumerate(sorted(hms, key=lambda x: x['dt'])):
        xy = plunger_xy(top, k, len(hms)); zt = top[2]
        # chime crown: one small bell per chord note, in a ring around the plunger foot
        ps = hm['pitches']; nb = len(ps)
        for j, p in enumerate(ps):
            ph = 2 * math.pi * j / nb + 0.4
            loc = Vector((xy[0] + 0.12 * math.cos(ph), xy[1] + 0.12 * math.sin(ph), zt - 0.22))
            bo = obj(prim('bell', M_BELL, 16), 'bell'); bo.location = loc; bo.scale = (0.075, 0.075, 0.11)
            cc = pc(p, 0.1); bo.color = (*cc, 0); key_alpha(bo, lit_curve(hm['t'], zt))
            RING.append((hm['t'], np.array(loc), cc))
        hub = obj(prim('cyl', M_GOLD2, 16), 'hub'); hub.location = (xy[0], xy[1], zt - 0.12); hub.scale = (0.07, 0.07, 0.04)
        # plunger: kicked up at the bass hit, flies up and falls back (ballistic) onto the crown one (two) beat(s) later
        hmax = G * hm['dt'] ** 2 / 8
        gtop = zt - 0.1 + hmax + 0.75
        gd = obj(prim('cyl', M_GLASS, 16), 'guide'); gd.location = (xy[0], xy[1], (zt - 0.1 + gtop) / 2); gd.scale = (0.075, 0.075, gtop - zt + 0.1)
        rod((xy[0], xy[1], gtop), (xy[0], xy[1], gtop + 0.03), 0.09, M_GOLD, 12)
        pl = link(bpy.data.objects.new('plunger', None))
        rd = obj(prim('cyl', M_GOLD, 12), 'prod'); rd.parent = pl; rd.location = (0, 0, 0.3); rd.scale = (0.03, 0.03, 0.5)
        hd = sphere((0, 0, 0.58), 0.06, M_GOLD, (12, 8)); hd.parent = pl
        tip = sphere((0, 0, 0.04), 0.045, M_FELT, (12, 8)); tip.parent = pl
        zz = np.full(NF, zt - 0.1)
        v0 = G * hm['dt'] / 2
        for f in range(NF):
            u = T[f] - hm['t_kick']
            if 0 < u < hm['dt']: zz[f] = zt - 0.1 + v0 * u - 0.5 * G * u * u
        pl.location = (xy[0], xy[1], 0); key_series(pl, 'location', 2, zz)
        # lever arm from the bass bar to the plunger foot
        rod((top - nv * 0.08), (xy[0], xy[1], zt - 0.1), 0.02, M_GOLD2, 6)

# ================================================================ glass core, gold rings, crown, base
Z_TOP = 0.0
Z_DOOR = Z_BOT + 3.2
core = obj(prim('cyl', M_GLASS, 48), 'core'); core.location = (0, 0, (Z_TOP + Z_DOOR) / 2); core.scale = (R_CORE, R_CORE, Z_TOP - Z_DOOR)
core.visible_shadow = False
for z in np.arange(Z_DOOR, Z_TOP + 0.01, 1.3):
    bpy.ops.mesh.primitive_torus_add(major_radius=R_CORE + 0.02, minor_radius=0.045, major_segments=48, minor_segments=8, location=(0, 0, z))
    bpy.context.object.data.materials.append(M_GOLD)
# open arcade at the foot (the marbles jump in between the columns)
DOOR_A = [az(POS['pearl'][int((FINAL - 0.4) * FPS)]), az(POS['night'][int((FINAL - 0.3) * FPS)])]
for k in range(10):
    a = 2 * math.pi * k / 10 + 0.31
    if min(abs(math.remainder(a - d, 2 * math.pi)) for d in DOOR_A + [PSI_ for PSI_ in (-LIFT['psi'], LIFT['psi'])]) < 0.42: continue
    rod(cyl(a, R_CORE, Z_BOT - 0.4), cyl(a, R_CORE, Z_DOOR), 0.07, M_GOLD, 10)
bpy.ops.mesh.primitive_torus_add(major_radius=R_CORE + 0.02, minor_radius=0.06, major_segments=48, minor_segments=8, location=(0, 0, Z_DOOR))
bpy.context.object.data.materials.append(M_GOLD)
# crown deck (ring around the lift shaft) with a plaque
deck = obj(prim('cyl', M_WOOD, 48), 'deck'); deck.location = (0, 0, Z_TOP - 0.06); deck.scale = (R_CORE, R_CORE, 0.12)
bpy.ops.mesh.primitive_torus_add(major_radius=R_CORE, minor_radius=0.05, major_segments=48, minor_segments=8, location=(0, 0, Z_TOP))
bpy.context.object.data.materials.append(M_GOLD)
# music-box base at the foot, on a dark lake
Z_BASE = Z_BOT - 0.4
bs = obj(prim('cyl', M_WOOD, 64), 'base'); bs.location = (0, 0, Z_BASE - 0.6); bs.scale = (5.6, 5.6, 1.2)
for zz_ in (Z_BASE, Z_BASE - 1.2):
    bpy.ops.mesh.primitive_torus_add(major_radius=5.6, minor_radius=0.07, major_segments=96, minor_segments=8, location=(0, 0, zz_))
    bpy.context.object.data.materials.append(M_GOLD)
lake = box((0, 0, Z_BASE - 1.3), (400, 400, 0.1), M_WATER, name='lake')

# ================================================================ lift: platform, two cups (with flaps), lantern, wind-up key
liftc = link(bpy.data.objects.new('lift', None)); liftc.location = (0, 0, 0)
key_series(liftc, 'location', 2, ZCAR)
plat = obj(prim('cyl', M_WOOD, 40), 'plat'); plat.parent = liftc; plat.location = (0, 0, -RB['pearl'] - 0.06); plat.scale = (1.48, 1.48, 0.1)
bpy.ops.mesh.primitive_torus_add(major_radius=1.48, minor_radius=0.04, major_segments=48, minor_segments=8, location=(0, 0, -RB['pearl']))
t_ = bpy.context.object; t_.parent = liftc; t_.data.materials.append(M_GOLD)
FLAPT = {r['ball']: r['t_open'] for r in EV['ramps']}
CUPBELLS = {'pearl': LEAD[-1][1], 'night': NIGHT[-1][1]}
for nme, sgn in (('pearl', -1), ('night', 1)):
    a = sgn * LIFT['psi']; c = cyl(a, LIFT['r_c'], 0)
    cu = obj(prim('bowl', M_GOLD, 24), 'cup'); cu.parent = liftc; cu.location = (c[0], c[1], -RB[nme] + 0.05); cu.scale = (0.3, 0.3, 0.3)
    # bells around the cup (the final chord)
    ps = CUPBELLS[nme]
    for j, p in enumerate(ps):
        ph = a + math.pi + (j - (len(ps) - 1) / 2) * 0.55
        q = c + np.array([0.36 * math.cos(ph), 0.36 * math.sin(ph), 0.0])
        bo = obj(prim('bell', M_BELL, 16), 'cbell'); bo.parent = liftc; bo.location = (q[0], q[1], 0.12); bo.scale = (0.09, 0.09, 0.13)
        cc = pc(p, 0.1); bo.color = (*cc, 0); key_alpha(bo, lit_curve(FINAL, LIFT['z_end']))
    # flap: the front lip of the cup, drops when the chute opens (and is up again when the lift comes back)
    hz = link(bpy.data.objects.new('flaph', None)); hz.parent = liftc
    fdir = np.array([math.cos(a), math.sin(a)])
    for r_ in EV['ramps']:
        if r_['ball'] == nme: d_ = np.array(r_['dir'][:2]); fdir = d_ / np.linalg.norm(d_)
    fa = math.atan2(fdir[1], fdir[0])
    hz.location = (c[0] + fdir[0] * 0.27, c[1] + fdir[1] * 0.27, -RB[nme]); hz.rotation_mode = 'XYZ'; hz.rotation_euler = (0, 0, fa)
    fl = box((0, 0, 0.12), (0.04, 0.34, 0.24), M_GOLD, name='flap'); fl.parent = hz
    ang = np.zeros(NF)
    to = FLAPT[nme]
    ang += np.array([sstep((t - to) / 0.08) for t in T]) * 1.4
    ang -= np.array([sstep((t - (to + 2.5)) / 0.4) for t in T]) * 1.4
    key_series(hz, 'rotation_euler', 1, ang, 1e-3)
    RING.append((FINAL, cyl(a, LIFT['r_c'], LIFT['z_end'] - RB[nme]), (1.0, 0.85, 0.5)))
LIGHTS += []
# lantern: glass cage + warm light; roof with the music-box wind-up key
for k in range(6):
    a = 2 * math.pi * k / 6 + math.pi / 6
    r_ = rod(cyl(a, 0.62, -RB['pearl']), cyl(a, 0.62, 1.4), 0.03, M_GOLD, 8); r_.parent = liftc
lg = obj(prim('cyl', M_LANTERN, 24), 'lanternglow'); lg.parent = liftc; lg.location = (0, 0, 0.45); lg.scale = (0.2, 0.2, 0.5)
roof = obj(prim('cone', M_GOLD, 32, 0.8, 0.12, 0.5), 'roof'); roof.parent = liftc; roof.location = (0, 0, 1.62)
ll = bpy.data.lights.new('lantern', 'POINT'); ll.color = (1.0, 0.72, 0.4); ll.energy = 70; ll.shadow_soft_size = 0.4
llo = link(bpy.data.objects.new('lanternl', ll)); llo.parent = liftc; llo.location = (0, 0, 0.6)
wk = link(bpy.data.objects.new('windkey', None)); wk.parent = liftc; wk.location = (0, 0, 2.05); wk.rotation_mode = 'XYZ'
rod((0, 0, -0.2), (0, 0, 0.45), 0.06, M_GOLD, 12).parent = wk
for sgn in (1, -1):
    lobe = obj(prim('cyl', M_GOLD, 32), 'lobe'); lobe.parent = wk; lobe.location = (sgn * 0.42, 0, 0.62); lobe.scale = (0.36, 0.36, 0.06)
    lobe.rotation_euler = (math.pi / 2, 0, 0)
    hole = obj(prim('cyl', M_DARK, 24), 'hole'); hole.parent = wk; hole.location = (sgn * 0.45, 0, 0.62); hole.scale = (0.17, 0.17, 0.08)
    hole.rotation_euler = (math.pi / 2, 0, 0)
wang = np.array([(-2 * math.pi * sstep(t / 0.15) if t < 0.2 else -2 * math.pi) - 0.12 * max(0, min(t, FINAL) - 0.2) for t in T])
wang = wang + np.array([2 * math.pi * 1.1 * sstep((t - LIFT['up'][0]) / (LIFT['up'][1] - LIFT['up'][0])) for t in T])
key_series(wk, 'rotation_euler', 2, wang, 1e-3)
# lift cable up to the pulley above the crown
pz = LIFT['z_h'] + 3.4
pul = obj(prim('cyl', M_GOLD, 32), 'pulley'); pul.location = (0, 0.95, pz); pul.scale = (0.4, 0.4, 0.1); pul.rotation_euler = (0, math.pi / 2, 0)
for s_ in (-1, 1): rod((0, s_ * 1.55, Z_TOP), (0, s_ * 0.6, pz + 0.1), 0.06, M_GOLD2, 8)
rod((0, -0.6, pz + 0.1), (0, 1.0, pz + 0.1), 0.06, M_GOLD2, 8)
cab = obj(prim('cyl', M_DARK, 8), 'cable'); cab.location = (0, 0.55, 0); cab.scale = (0.025, 0.025, 1)
key_series(cab, 'location', 2, (ZCAR + 2.2 + pz) / 2); key_series(cab, 'scale', 2, np.maximum(0.05, pz - ZCAR - 2.2))
pul.rotation_mode = 'XYZ'; key_series(pul, 'rotation_euler', 0, -ZCAR / 0.4)

# ================================================================ chutes (start: from the lift cups; end: from the cradle)
def chute_mesh(head, lip, dirv, back=0.15):
    head = np.array(head); lip = np.array(lip); dirv = np.array(dirv)
    b0 = head - dirv * back
    side = np.cross(dirv, [0, 0, 1.0]); side /= np.linalg.norm(side)
    for s_ in (-1, 1):
        rod(b0 + side * s_ * 0.17 - [0, 0, 0.1], lip + side * s_ * 0.17 - [0, 0, 0.1], 0.022, M_GOLD, 8)
    fl_ = obj(prim('cube', M_WOOD), 'chute'); mid = (b0 + lip) / 2 - [0, 0, RB['pearl'] + 0.02]
    Lc = float(np.linalg.norm(lip - b0)); M = kframe(mid, np.cross(side, dirv))
    fl_.matrix_world = Matrix.Translation(Vector(mid)) @ Matrix(((dirv[0], side[0], 0), (dirv[1], side[1], 0), (dirv[2], side[2], 1))).to_4x4() @ Matrix.Diagonal((Lc, 0.34, 0.04, 1))
    nz = np.cross(dirv, side); nz /= np.linalg.norm(nz)
    fl_.matrix_world = Matrix.Translation(Vector(mid)) @ Matrix(((dirv[0], side[0], nz[0]), (dirv[1], side[1], nz[1]), (dirv[2], side[2], nz[2]))).to_4x4() @ Matrix.Diagonal((Lc, 0.34, 0.04, 1))
for r_ in EV['ramps']:
    hd = np.array(r_['head']); dv = np.array(r_['dir']); hd2 = hd + dv * 0.3      # the chute starts at the lift's rim
    chute_mesh(hd2, r_['lip'], dv, back=0.0)
ec = EV['endchute']
chute_mesh(ec['head'], ec['lip'], ec['dir'])
crd = obj(prim('bowl', M_GOLD, 20), 'cradle'); hc = np.array(ec['head']); crd.location = (hc[0], hc[1], hc[2] - RB['night'] + 0.06); crd.scale = (0.3, 0.3, 0.25)
rod((hc[0], hc[1], hc[2] - 0.3), cyl(az(hc), R_CORE, hc[2] - 0.6), 0.04, M_GOLD2, 8)
# the hammer stroke that opens the pearl's flap: a glowing wire from that plunger to the crown
hlev = [hm for hm in HAM if abs(hm['t'] - [l for l in EV['levers'] if l.get('kind') == 'hammer'][0]['t']) < 1e-6][0]
hb = nh[hlev['ib']]; tp_ = np.array(hb['p']) - RB['night'] * np.array(hb['n']); xy = plunger_xy(tp_)
wire = tube([(xy[0], xy[1], tp_[2] - 0.2), (xy[0] * 0.9, xy[1] * 0.9, Z_TOP - 0.1), cyl(-LIFT['psi'] + 0.2, R_CORE + 0.1, Z_TOP + 0.05).tolist()], 0.02, M_LAMP, 'wire')
wire.color = (1.0, 0.8, 0.4, 0)
f0 = int(round(hlev['t'] * FPS))
for df, aa in ((-1, 0.0), (0, 1.0), (12, 0.4), (40, 0.0)):
    wire.color = (1.0, 0.8, 0.4, aa); wire.keyframe_insert('color', index=3, frame=max(0, f0 + df))

# ================================================================ carousel horses (ride the pearl over its long notes)
HCOL = {'A': (0.95, 0.92, 0.88), 'B': (0.55, 0.75, 1.0), 'C': (1.0, 0.65, 0.75)}
LV = {l['horse']: l for l in EV['levers'] if l.get('horse')}
def horse_model(col):
    root = link(bpy.data.objects.new('horse', None))
    def part(o): o.parent = root; return o
    m = M_OBJ
    b = part(sphere((0, 0, -0.38), 1, m, (16, 10))); b.scale = (0.42, 0.16, 0.17); b.color = (*col, 1)
    nk = part(obj(prim('cyl', m, 10), 'neck')); nk.location = (0.32, 0, -0.2); nk.scale = (0.08, 0.08, 0.32); nk.rotation_euler = (0, 0.5, 0); nk.color = (*col, 1)
    hd = part(sphere((0.45, 0, -0.04), 1, m, (12, 8))); hd.scale = (0.17, 0.09, 0.09); hd.rotation_euler = (0, 0.5, 0); hd.color = (*col, 1)
    mn = part(sphere((0.28, 0, -0.1), 1, M_GOLD, (10, 6))); mn.scale = (0.1, 0.05, 0.16); mn.rotation_euler = (0, 0.5, 0)
    for lx, ang in ((0.25, 0.5), (-0.25, -0.5)):
        for ly in (-0.07, 0.07):
            lg_ = part(obj(prim('cyl', m, 8), 'leg')); lg_.location = (lx + 0.08 * np.sign(ang), ly, -0.6); lg_.scale = (0.035, 0.035, 0.32)
            lg_.rotation_euler = (0, ang, 0); lg_.color = (*col, 1)
    tl = part(obj(prim('cone', M_GOLD, 8, 1, 0, 1), 'tail')); tl.location = (-0.47, 0, -0.38); tl.scale = (0.05, 0.05, 0.3); tl.rotation_euler = (0, -2.3, 0)
    pole = part(obj(prim('cyl', M_GOLD, 12), 'pole')); pole.location = (0, 0, 1.0); pole.scale = (0.025, 0.025, 2.6)
    return root
for rd_ in EV['rides']:
    lvh = LV[rd_['horse']]
    ts = np.array(rd_['ts']); ps = np.array(rd_['ps'])
    sad0 = np.array(lvh['saddle']); hov = np.array(lvh['hover']); tdrop = lvh['t_drop']
    P_ = np.zeros((NF, 3)); YAW = np.zeros(NF); TIP = np.zeros(NF)
    for f in range(NF):
        t = T[f]
        if t <= ts[0]: q = sad0
        elif t <= ts[-1]: q = np.array([np.interp(t, ts, ps[:, c]) for c in range(3)])
        else: q = hov + [0, 0, 1.3 * sstep((t - tdrop - 0.05) / 0.6)]
        P_[f] = q; YAW[f] = az(q) + math.pi / 2
        if t > tdrop - 0.02: TIP[f] = 0.9 * sstep((t - tdrop + 0.02) / 0.1) * (1 - sstep((t - tdrop - 0.8) / 0.5))
    hroot = link(bpy.data.objects.new('hroot', None)); key_loc(hroot, P_ - np.array([0, 0, RB['pearl'] - 0.05]))
    hroot.rotation_mode = 'XYZ'; YAW = np.unwrap(YAW); key_series(hroot, 'rotation_euler', 2, YAW, 1e-3)
    hm_ = horse_model(HCOL[rd_['horse']]); hm_.parent = hroot
    bob = link(bpy.data.objects.new('saddle', None)); bob.parent = hroot; bob.rotation_mode = 'XYZ'
    sd = obj(prim('bowl', M_FELT, 20), 'saddlecup'); sd.parent = bob; sd.scale = (0.27, 0.27, 0.22)
    key_series(bob, 'rotation_euler', 0, TIP, 1e-3)
    # canopy: a gold band with bulbs above the ride
    route = np.array(rd_['route']); zt_ = route[:, 2].max() + 3.4
    can = np.array([cyl(az(q), math.hypot(q[0], q[1]), zt_) for q in route])
    tube(can.tolist(), 0.07, M_GOLD, 'canopy')
    for j in range(0, len(can), 6):
        bl = sphere(tuple(can[j] - [0, 0, 0.09]), 0.06, M_LAMP, (8, 6)); bl.color = (1.0, 0.8, 0.45, 0.6)
    SFXH = None

# ================================================================ hit rings (FX pool)
NRING = 16
RING.sort(key=lambda r: r[0])
ring_loc = np.zeros((NRING, NF, 3)); ring_scl = np.full((NRING, NF), 1e-3); ring_a = np.zeros((NRING, NF)); ring_c = [(1, 1, 1)] * NRING
ring_rot = np.zeros((NRING, NF))
for k, (t, top, c) in enumerate(RING):
    r = k % NRING; f0 = int(round(t * FPS))
    for df in range(14):
        f = f0 + df
        if f >= NF or f < 0: break
        u = df / 13; ring_loc[r, f] = np.array(top) + [0, 0, 0.02]; ring_scl[r, f] = 0.35 + 1.2 * (1 - (1 - u) ** 2)
        ring_a[r, f] = (1 - u) ** 1.6
    ring_c[r] = c
for r in range(NRING):
    bpy.ops.mesh.primitive_torus_add(major_radius=0.25, minor_radius=0.014, major_segments=32, minor_segments=5)
    o = bpy.context.object; o.data.materials.append(M_FX); o.color = (*ring_c[r], 0)
    for f in range(1, NF):
        if ring_a[r, f] == 0: ring_loc[r, f] = ring_loc[r, f - 1]
    key_loc(o, ring_loc[r])
    for k in range(3): key_series(o, 'scale', k, ring_scl[r])
    key_series(o, 'color', 3, ring_a[r], 1e-3)

# ================================================================ sky, moon, stars, lights
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes['Background']; bg.inputs['Color'].default_value = (0.012, 0.018, 0.06, 1); bg.inputs['Strength'].default_value = 1.2
SKY = bpy.data.materials.new('sky'); SKY.use_nodes = True; nt = SKY.node_tree; nt.nodes.clear()
tc = nt.nodes.new('ShaderNodeTexCoord'); sep = nt.nodes.new('ShaderNodeSeparateXYZ')
cz = nt.nodes.new('ShaderNodeRGB'); cz.outputs[0].default_value = (0.004, 0.008, 0.04, 1)
ch = nt.nodes.new('ShaderNodeRGB'); ch.outputs[0].default_value = (0.05, 0.06, 0.2, 1)
mr = nt.nodes.new('ShaderNodeMapRange'); mr.inputs['From Min'].default_value = 0.25; mr.inputs['From Max'].default_value = 0.8
mix = nt.nodes.new('ShaderNodeMixRGB'); vo_ = nt.nodes.new('ShaderNodeTexVoronoi'); vo_.inputs['Scale'].default_value = 420.0
st = nt.nodes.new('ShaderNodeMapRange'); st.inputs['From Min'].default_value = 0.035; st.inputs['From Max'].default_value = 0.0
st.inputs['To Min'].default_value = 0.0; st.inputs['To Max'].default_value = 2.5
stm = nt.nodes.new('ShaderNodeMath'); stm.operation = 'MULTIPLY'
add = nt.nodes.new('ShaderNodeMixRGB'); add.blend_type = 'ADD'; add.inputs['Fac'].default_value = 1.0
em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Strength'].default_value = 1.0; out = nt.nodes.new('ShaderNodeOutputMaterial')
nt.links.new(tc.outputs['Generated'], sep.inputs[0]); nt.links.new(sep.outputs['Z'], mr.inputs['Value'])
nt.links.new(mr.outputs[0], mix.inputs['Fac']); nt.links.new(ch.outputs[0], mix.inputs[1]); nt.links.new(cz.outputs[0], mix.inputs[2])
nt.links.new(tc.outputs['Generated'], vo_.inputs['Vector']); nt.links.new(vo_.outputs['Distance'], st.inputs['Value'])
nt.links.new(st.outputs[0], stm.inputs[0]); nt.links.new(sep.outputs['Z'], stm.inputs[1])
nt.links.new(mix.outputs[0], add.inputs[1]); nt.links.new(stm.outputs[0], add.inputs[2])
nt.links.new(add.outputs[0], em.inputs['Color']); nt.links.new(em.outputs[0], out.inputs[0])
# a sky dome (sphere seen from inside) so the stars surround the orbiting camera
bpy.ops.mesh.primitive_uv_sphere_add(radius=300, segments=48, ring_count=24, location=(0, 0, -10))
dome = bpy.context.object; dome.data.materials.append(SKY); dome.visible_shadow = False
for p in dome.data.polygons: p.use_smooth = True
# the moon (follows behind the tower, opposite the camera, a little to the side)
moon = sphere((0, 0, 0), 4.0, emis('moonm', (1.0, 0.95, 0.82), 1.6), (32, 16), 'moon')
mglow = obj(prim('plane', M_MOON), 'moonglow'); mglow.color = (0.7, 0.8, 1.0, 0.5)
sun = bpy.data.lights.new('moonl', 'SUN'); sun.angle = 0.04; sun.color = (0.62, 0.72, 1.0); sun.energy = 2.2
so = link(bpy.data.objects.new('moonl', sun))
fill = bpy.data.lights.new('fill', 'SUN'); fill.color = (1.0, 0.8, 0.6); fill.energy = 0.8
fo = link(bpy.data.objects.new('fill', fill))

# ================================================================ camera: circles the tower with the marbles
LENS = 32.0
PH0 = 0.25
th_course = np.array([thn(t) for t in T])
PH = np.where(T < TM - 1.0, PH0, th_course - 0.02)
PH = np.where(T > FINAL, 6 * math.pi + PH0, PH)
PH = smooth(PH, 1.2)
# focus height: between the two marbles, the crown at the start and at the end
mz = (POS['pearl'][:, 2] + POS['night'][:, 2]) / 2
ZF0 = -0.9
FZ = np.where(T < TM - 0.5, ZF0, mz + 0.9)
rev_mid = np.array([sstep((t - 21.0) / 1.6) * (1 - sstep((t - 25.4) / 1.6)) for t in T])
rev_end = np.array([sstep((t - (FINAL + 0.2)) / 1.8) for t in T])
back = np.array([sstep((t - (LIFT['up'][0] + 1.6)) / 3.4) for t in T])
ztower = (LIFT['z_h'] + LIFT['z_end']) / 2
FZ = FZ * (1 - rev_end) + ztower * rev_end
FZ = FZ * (1 - back) + ZF0 * back
FZ = smooth(FZ, 0.8)
WID = 6.2 + 3.0 * rev_mid
WID = WID * (1 - rev_end) + 16.5 * rev_end
WID = WID * (1 - back) + 6.2 * back
WID = smooth(WID, 0.5)
RF = 2.9 * (1 - rev_end * (1 - back)) + 0.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 900
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24
    ph = PH[f]; fx = cyl(ph, RF[f], FZ[f])
    el = 0.16
    CP[f] = fx + np.array([math.cos(ph), math.sin(ph), 0]) * D * math.cos(el) + [0, 0, D * math.sin(el)]
    TP[f] = fx
key_loc(cam, CP); key_loc(tgt, TP)
# moon + lights placed relative to the camera direction
MP = np.array([cyl(PH[f] + math.pi - 0.75, 170, 95 + FZ[f]) for f in range(NF)])
key_loc(moon, MP)
key_loc(mglow, MP + np.array([[math.cos(PH[f] + math.pi), math.sin(PH[f] + math.pi), 0] for f in range(NF)]) * 2)
mglow.rotation_mode = 'XYZ'; key_series(mglow, 'rotation_euler', 2, PH + math.pi / 2); mglow.rotation_euler[0] = math.pi / 2
mglow.scale = (70, 70, 1)
so.rotation_mode = 'XYZ'; so.rotation_euler = (math.radians(55), 0, 0); key_series(so, 'rotation_euler', 2, PH + math.pi / 2 + 0.55 - math.pi / 2)
fo.rotation_mode = 'XYZ'; fo.rotation_euler = (math.radians(65), 0, 0); key_series(fo, 'rotation_euler', 2, PH - math.pi / 2 + 0.6)

# ================================================================ text: crown plaque, banners (keywords), HUD hook + counter
def wtext(lines, th, r, z, plate_w=None, plate_h=None, tilt=0.0):
    """a plaque facing outwards at azimuth th; lines = [(text, size, material)] from the top"""
    rot = face_out(th, tilt); out_ = np.array([math.cos(th), math.sin(th), 0])
    if plate_w:
        pl = box(tuple(cyl(th, r - 0.06, z)), (plate_w, 0.06, plate_h), M_PLATE, rot=(0, 0, th + math.pi / 2), name='plaque')
        for s_ in (1, -1): box(tuple(cyl(th, r - 0.03, z + s_ * plate_h / 2)), (plate_w + 0.1, 0.08, 0.06), M_GOLD, rot=(0, 0, th + math.pi / 2))
    tot = sum(s * 1.3 for _, s, _ in lines); zz = z + tot / 2
    objs = []
    for s_, size, mt in lines:
        zz -= size * 1.3 / 2
        o = fit_w(text(s_, tuple(cyl(th, r, zz)), size, mt, rot=rot), (plate_w - 0.3) if plate_w else None); objs.append(o)
        zz -= size * 1.3 / 2
    return objs
def at_t(t, n='night'): return POS[n][int(round(t * FPS))]
# crown plaque (seen in the first seconds and at the very end)
wtext([('GYMNOPÉDIE No.1', 0.34, M_TXT_G), ('Erik Satie · 1888', 0.22, M_TXT), ('ジムノペディ第1番 エリック・サティ', 0.17, M_TXT)],
      PH0 - 0.05, R_CORE + 0.3, -4.75, plate_w=2.3, plate_h=1.15)
# banners along the spiral, just below the outer lane (first 30 s: search keywords)
for t_, l1, l2, l3 in ((9.0, 'Classical Piano', 'Satie – Gymnopédie No.1', 'クラシック ピアノ 癒し'),
                       (14.5, 'Marble Machine', 'Music Box Tower', 'ビー玉 マーブルマシン オルゴール'),
                       (20.5, 'Relaxing Sleep Music', 'Satisfying ASMR', '睡眠用BGM 作業用 ASMR'),
                       (27.0, 'Erik Satie', 'French composer, 1888', 'フランスの作曲家 サティ')):
    th = thn(t_) + 0.05; zb = min(POS['pearl'][int(t_ * FPS), 2], POS['night'][int(t_ * FPS), 2]) - 2.6
    wtext([(l1, 0.3, M_TXT_G), (l2, 0.21, M_TXT), (l3, 0.17, M_TXT)], th, 4.75, zb, plate_w=3.0, plate_h=1.05)
# HUD (rides with the camera)
hud = link(bpy.data.objects.new('hud', None)); hud.parent = cam
HZ = 6.0; hh_ = HZ * 24 / LENS; ww_ = hh_ * 9 / 16
hud.location = (0, 0, -HZ)
def hud_text(s, y, size, mt, t0, t1, fade=0.35, x=0.0, align='CENTER'):
    o = fit_w(text(s, (x, y, 0), size, mt, rot=(0, 0, 0), align=align), ww_ * 0.9); o.parent = hud
    k0 = o.scale[0]
    a = np.clip((T - t0) / fade, 0, 1) * np.clip((t1 - T) / fade, 0, 1)
    for c in range(3): key_series(o, 'scale', c, np.maximum(0.001, a * k0), 1e-3)
    return o
TL = T_END
# hook (also back in the last second -> seamless loop)
for t0_, t1_ in ((-1.0, 6.2), (TL - 1.4, TL + 5)):
    hud_text('This tower is a music box', hh_ * 0.40, hh_ * 0.022, M_TXT, t0_, t1_)
    hud_text('この塔は オルゴール', hh_ * 0.352, hh_ * 0.026, M_TXT, t0_, t1_)
    hud_text('Can 2 marbles light it up?', hh_ * 0.315, hh_ * 0.02, M_TXT_G, t0_ + 0.6, t1_)
    hud_text('2つの玉で 塔をぜんぶ灯せる？', hh_ * 0.265, hh_ * 0.026, M_TXT, t0_ + 0.6, t1_)
hud_text('Satie – Gymnopédie No.1', -hh_ * 0.39, hh_ * 0.019, M_TXT_G, -1, 6.2)
hud_text('Classical piano · Marble machine · Music box', -hh_ * 0.43, hh_ * 0.015, M_TXT, -1, 6.2)
hud_text('ALL LIT!', hh_ * 0.33, hh_ * 0.04, M_TXT_G, T_FLARE + 1.0, LIFT['up'][0] + 2.5)
hud_text('ぜんぶ灯った！', hh_ * 0.26, hh_ * 0.032, M_TXT, T_FLARE + 1.0, LIFT['up'][0] + 2.5)
# counter: LIGHTS nnn/105 (from the first melody note until the lift starts up)
NTOT = len(LIGHTS); LT = np.sort(np.array(LIGHTS))
CV = np.searchsorted(LT, T + 1e-6)
cvis = (T > TM - 0.8) & (T < LIFT['up'][0] + 2.5)
ts_h = hh_ * 0.03; DW = ts_h * 0.95
yC = hh_ * 0.42
lab = text('LIGHTS', (-ww_ * 0.06, yC, 0), ts_h * 0.9, M_TXT, rot=(0, 0, 0), align='RIGHT'); lab.parent = hud
lab2 = text('灯り', (-ww_ * 0.06, yC - ts_h * 1.05, 0), ts_h * 0.7, M_TXT, rot=(0, 0, 0), align='RIGHT'); lab2.parent = hud
tot = text(f'/{NTOT}', (ww_ * 0.07 + 2.6 * DW, yC, 0), ts_h * 0.9, M_TXT, rot=(0, 0, 0), align='LEFT'); tot.parent = hud
for o_ in (lab, lab2, tot):
    for c in range(3): key_series(o_, 'scale', c, np.where(cvis, 1.0, 0.001))
for slot in range(3):
    for dgt in range(10):
        o = text(str(dgt), (ww_ * 0.07 + (slot - 0.3) * DW, yC, 0), ts_h * 1.15, M_TXT_G, rot=(0, 0, 0)); o.parent = hud
        vis = np.zeros(NF)
        for f in range(NF):
            s_ = f'{int(CV[f]):>3d}'
            vis[f] = 1.0 if (s_[slot] == str(dgt) and cvis[f]) else 0.0
        if not vis.any(): o.hide_render = True; continue
        for k in range(3): key_series(o, 'scale', k, np.maximum(0.001, vis))

# ================================================================ final: sparkles rising around the tower
for k in range(140):
    sp_ = obj(prim('sph', M_LAMP, 6, 4), 'spark'); sp_.color = (*[(1, 0.85, 0.5), (0.7, 0.85, 1.0), (1, 0.7, 0.85)][k % 3], 0.8)
    a0 = rng.uniform(0, 2 * math.pi); r0 = rng.uniform(2.0, 5.5); z0 = rng.uniform(LIFT['z_end'], LIFT['z_h'])
    vz = rng.uniform(0.3, 1.2); tb = T_FLARE + 1.2 * (z0 - LIFT['z_end']) / (LIFT['z_h'] - LIFT['z_end'])
    P_ = np.zeros((NF, 3)); S_ = np.full(NF, 0.001)
    for f in range(NF):
        u = T[f] - tb
        P_[f] = cyl(a0 + 0.15 * max(0, u), r0 + 0.2 * max(0, u), z0 + vz * max(0, u))
        if 0 <= u < 4.5: S_[f] = 0.05 * min(1, u / 0.2) * (1 - u / 4.5)
    key_loc(sp_, P_)
    for c in range(3): key_series(sp_, 'scale', c, S_, 1e-4)
gl_ = bpy.data.lights.new('towerglow', 'POINT'); gl_.color = (1.0, 0.8, 0.5); gl_.shadow_soft_size = 3.0
glo = link(bpy.data.objects.new('towerglow', gl_)); glo.location = (0, 0, ztower)
for f in list(range(0, NF, 3)) + [NF - 1]:
    u = T[f] - T_FLARE
    gl_.energy = float(12000 * (sstep(u / 1.2) * (1 - sstep((T[f] - LIFT['up'][0]) / 4.0)))); gl_.keyframe_insert('energy', frame=f)

if MODE == 'test':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS)); sc.frame_set(f)
        sc.render.filepath = f'{HERE}/test_{f:04d}.png'
        t1 = time.time(); bpy.ops.render.render(write_still=True); print('frame', f, 'sec', round(time.time() - t1, 1))
elif MODE == 'range':
    sc.frame_start, sc.frame_end = int(args[1]), int(args[2])
    sc.render.image_settings.file_format = 'PNG'; sc.render.filepath = f'{HERE}/frames/f_'
    bpy.ops.render.render(animation=True)
elif MODE == 'probe':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS)); sc.frame_set(f)
        print('t', tt_, 'cam', [round(v, 2) for v in cam.matrix_world.translation], 'WID', round(WID[f], 2), 'FZ', round(FZ[f], 2),
              'pearl', np.round(POS['pearl'][f], 2), 'night', np.round(POS['night'][f], 2), 'lift', round(ZCAR[f], 2))
print('scene objects', len(bpy.data.objects))
