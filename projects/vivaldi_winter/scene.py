"""Vivaldi 'Winter' II Largo - "Will the snowmen smile?", side view of a long log cabin on a blizzard night, cut open towards
the camera. Every solo-violin note = the ember ball landing on a copper key of that pitch (front lane, by the hearth); cello =
the iron ball on firewood log drums (bench, back lane); the two pizzicato violin parts = raindrops falling from the sky onto
two rows of glass chimes outside the windows (Vivaldi's 'La Pioggia'). layout.py plans all motion (constant gravity, bounces
that never add speed, stove pop, shelf flap, copper rails, bellows, chain elevator, roof chute); this file builds and animates it.
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
FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.5) * FPS); T = np.arange(NF) / FPS
G = LAY['G']; EV = LAY['ev']; LN = LAY['lanes']; KL, KS = LAY['key_lead'], LAY['key_stay']; RN = LAY['rain']
T0 = E['t0']; T_GUST, T_BEL = E['gust'], EV['bellows']['t']; GZB = LAY['gzb']; VV = LAY['v']
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
def strip_mesh(X, Z, y0, y1, th, m, name, Z1=None, m_front=None):
    """extruded profile along x: top surface z = Z (Z1 at the back edge if given), thickness th; the front face (y0 side)
    gets m_front if given"""
    Z1 = Z if Z1 is None else Z1
    bm = bmesh.new(); vs = []
    for x, z, z1 in zip(X, Z, Z1):
        vs.append([bm.verts.new((x, y0, z)), bm.verts.new((x, y1, z1)), bm.verts.new((x, y1, z1 - th)), bm.verts.new((x, y0, z - th))])
    for a, b in zip(vs, vs[1:]):
        for k in range(4):
            f_ = bm.faces.new((a[k], b[k], b[(k + 1) % 4], a[(k + 1) % 4]))
            if k == 3 and m_front is not None: f_.material_index = 1
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new(name, me)); o.data.materials.append(m)
    if m_front is not None: o.data.materials.append(m_front)
    return o


# ---------------- pitch colours ----------------
PITCH_COL = [(1.0, 0.25, 0.3), (1.0, 0.45, 0.2), (1.0, 0.65, 0.15), (1.0, 0.85, 0.2), (0.6, 1.0, 0.2), (0.2, 1.0, 0.4),
             (0.1, 1.0, 0.8), (0.15, 0.75, 1.0), (0.3, 0.45, 1.0), (0.6, 0.35, 1.0), (0.85, 0.35, 1.0), (1.0, 0.35, 0.7)]
def pcol(p, light=0.0):
    c = PITCH_COL[p % 12]; return tuple(min(1, x * (1 - light) + light) for x in c)
SNOWC = (0.92, 0.95, 1.0)


def flash(o, t, col, before=SNOWC, keep_col=True):
    """object colour: `before` (snow white) until the hit, then the pitch colour with an emission flash"""
    fh = int(round(t * FPS))
    o.color = (*before, 0); o.keyframe_insert('color', frame=max(0, fh - 1))
    for df, a in ((0, 1.0), (4, 0.55), (18, 0.12)):
        o.color = (*(col if keep_col else before), a); o.keyframe_insert('color', frame=fh + df)


def bump(o, t, nv, depth=0.03):
    fh = int(round(t * FPS)); base = o.location.copy(); nv = Vector(nv)
    for df, d in ((-1, 0), (1, -depth), (4, depth * 0.3), (8, 0)):
        o.location = base + nv * d; o.keyframe_insert('location', frame=max(0, fh + df))

BP = {n: BallPath(d) for n, d in LAY['balls'].items()}
POS = {n: np.array([b.at(t) for t in T]) for n, b in BP.items()}
MAIN = ('mel', 'bas')


def hold(P, act):
    """inactive frames keep the nearest earlier active position (or the next one before the first activity), so that a
    hidden object never streaks across the frame under motion blur. Positions change only between two inactive frames."""
    P = P.copy(); act = np.asarray(act, bool); idx = np.nonzero(act)[0]
    if not len(idx): return P
    P[:idx[0]] = P[idx[0]]
    last = idx[0]
    for f in range(idx[0], len(P)):
        if act[f]: last = f
        else: P[f] = P[last]
    # one inactive frame before every new activity: move there already
    for f in range(1, len(P)):
        if act[f] and not act[f - 1]: P[f - 1] = P[f]
    return P


def gust_curve():
    """fire / lamp factor: 1 normally; the gust (bar 12) blows it down to 0.22; the bellows (bar 14) roars it up to 1.7 -> 1"""
    g = np.ones(NF)
    for f in range(NF):
        t = T[f]
        if T_GUST <= t < T_BEL: g[f] = 1 - 0.78 * smoothstep((t - T_GUST) / 0.6) + 0.08 * math.sin(t * 23) * (t > T_GUST + 0.6)
        elif t >= T_BEL: g[f] = 1 + 0.7 * math.exp(-(t - T_BEL) / 1.2) * smoothstep((t - T_BEL) / 0.15)
        if t < T_GUST or t >= T_BEL: g[f] *= 1 + 0.06 * math.sin(t * 17.0) + 0.04 * math.sin(t * 29.0 + 1.0)
    return g
GF = gust_curve()
WARM = np.clip((T - FINAL) / 1.5, 0, 1)                          # the final warmth
SNOWSTOP = np.clip((T - FINAL - 0.4) / 1.8, 0, 1)

# ---------------- materials ----------------
M_EMBER = mat('ember', (1.0, 0.42, 0.1), 0.25, **{'Emission Color': (1.0, 0.38, 0.06), 'Emission Strength': 4.0, 'Coat Weight': 1.0})
key_val(M_EMBER.node_tree.nodes['Principled BSDF'].inputs['Emission Strength'], 2.5 + 2.5 * GF + 6 * WARM, 2)
M_IRON = mat('iron', (0.07, 0.07, 0.08), 0.28, **{'Metallic': 1.0})
M_GOLD = mat('gold', (1.0, 0.72, 0.28), 0.22, **{'Metallic': 0.9})
M_COPPER = mat('copper', (0.95, 0.5, 0.3), 0.25, **{'Metallic': 0.95})
M_WHITE = mat('white', (0.95, 0.95, 0.95), 0.3)
M_BLACK = mat('coal', (0.02, 0.02, 0.02), 0.5)
M_CARROT = mat('carrot', (1.0, 0.38, 0.05), 0.5)
M_SCARF = objcol('scarf', 0.7)
M_KEY = glowmat('copperkey', gain=7.0, base=0.15, rough=0.25, metal=0.6, coat=0.6)
M_DRUM = glowmat('drumhead', gain=5.0, base=0.15, rough=0.5)
M_CHIME = glowmat('chime', gain=9.0, base=0.35, rough=0.05, coat=1.0)
M_CUP = glowmat('cupbell', gain=6.0, base=0.15, rough=0.2, metal=0.8)
M_BARK = mat('bark', (0.2, 0.12, 0.07), 0.8)
M_BIRCH = mat('birch', (0.85, 0.82, 0.75), 0.7)
M_STEEL = mat('steel', (0.5, 0.52, 0.56), 0.3, **{'Metallic': 0.9})
M_DARK = mat('darkmetal', (0.08, 0.08, 0.09), 0.4, **{'Metallic': 0.8})
M_SNOW = mat('snow', (0.88, 0.92, 1.0), 0.6, **{'Subsurface Weight': 0.2})
M_SNOWMAN = mat('snowman', (0.92, 0.95, 1.0), 0.55, **{'Subsurface Weight': 0.25, 'Emission Color': (1.0, 0.75, 0.45), 'Emission Strength': 0.0})
key_val(M_SNOWMAN.node_tree.nodes['Principled BSDF'].inputs['Emission Strength'], 0.0 + 0.25 * WARM, 3)
M_LEATHER = mat('leather', (0.35, 0.16, 0.07), 0.6)
M_GLASS = bpy.data.materials.new('glass'); M_GLASS.use_nodes = True
_nt = M_GLASS.node_tree; _nt.nodes.clear()
_tr = _nt.nodes.new('ShaderNodeBsdfTransparent'); _gl = _nt.nodes.new('ShaderNodeBsdfGlossy'); _gl.inputs['Roughness'].default_value = 0.05
_mx = _nt.nodes.new('ShaderNodeMixShader'); _mx.inputs['Fac'].default_value = 0.12; _o = _nt.nodes.new('ShaderNodeOutputMaterial')
_nt.links.new(_tr.outputs[0], _mx.inputs[1]); _nt.links.new(_gl.outputs[0], _mx.inputs[2]); _nt.links.new(_mx.outputs[0], _o.inputs[0])
M_OBJ = objcol('objcol', 0.5)
M_LAMP = objcol('lampcol', 0.4, emit=5.0)
M_FX = fxmat('fx', 8.0)
M_FLAME = fxmat('flame', 10.0)
M_TXT = emis('txtw', (1.0, 1.0, 1.0), 3.0); M_TXT_G = emis('txtg', (1.0, 0.8, 0.35), 4.5); M_TXT_B = emis('txtb', (0.6, 0.85, 1.0), 4.0)
M_TXT_O = emis('txto', (1.0, 0.55, 0.25), 4.5)
M_FROST = emis('frosttxt', (0.85, 0.93, 1.0), 2.2)


def wood_mat(name, c1, c2, scale=3.0, axis='Z', bump=0.3):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; b = nt.nodes['Principled BSDF']
    b.inputs['Roughness'].default_value = 0.7
    tc = nt.nodes.new('ShaderNodeTexCoord'); wv = nt.nodes.new('ShaderNodeTexWave'); wv.wave_type = 'BANDS'
    wv.bands_direction = axis; wv.inputs['Scale'].default_value = scale; wv.inputs['Distortion'].default_value = 3.0
    wv.inputs['Detail'].default_value = 2.0
    nt.links.new(tc.outputs['Object'], wv.inputs['Vector'])
    cr = nt.nodes.new('ShaderNodeValToRGB'); cr.color_ramp.elements[0].color = (*c1, 1); cr.color_ramp.elements[1].color = (*c2, 1)
    nt.links.new(wv.outputs['Fac'], cr.inputs['Fac']); nt.links.new(cr.outputs['Color'], b.inputs['Base Color'])
    bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = bump
    nt.links.new(wv.outputs['Fac'], bp.inputs['Height']); nt.links.new(bp.outputs[0], b.inputs['Normal'])
    return m
M_LOGWALL = wood_mat('logwall', (0.22, 0.11, 0.05), (0.45, 0.26, 0.12), 2.2, 'Z', 0.5)
M_FLOOR = wood_mat('floor', (0.3, 0.16, 0.07), (0.52, 0.32, 0.16), 1.6, 'X', 0.2)
M_WOOD = wood_mat('wood', (0.35, 0.19, 0.08), (0.6, 0.38, 0.18), 4.0, 'X', 0.2)
M_STONE = bpy.data.materials.new('stone'); M_STONE.use_nodes = True; _nt = M_STONE.node_tree; _b = _nt.nodes['Principled BSDF']
_br = _nt.nodes.new('ShaderNodeTexBrick'); _br.inputs['Scale'].default_value = 3.0; _br.inputs['Mortar Size'].default_value = 0.03
_br.inputs['Color1'].default_value = (0.45, 0.42, 0.4, 1); _br.inputs['Color2'].default_value = (0.3, 0.28, 0.27, 1); _br.inputs['Mortar'].default_value = (0.12, 0.11, 0.1, 1)
_tc = _nt.nodes.new('ShaderNodeTexCoord'); _mp = _nt.nodes.new('ShaderNodeMapping'); _mp.inputs['Rotation'].default_value = (math.pi / 2, 0, 0)
_nt.links.new(_tc.outputs['Object'], _mp.inputs['Vector']); _nt.links.new(_mp.outputs[0], _br.inputs['Vector'])
_nt.links.new(_br.outputs['Color'], _b.inputs['Base Color']); _b.inputs['Roughness'].default_value = 0.85

# ---------------- balls ----------------
BALL = {}
for n, b in BP.items():
    rr = b.r
    bpy.ops.mesh.primitive_uv_sphere_add(radius=rr, segments=32, ring_count=16); o = bpy.context.object; o.name = n
    for p in o.data.polygons: p.use_smooth = True
    o.data.materials.append(M_EMBER if n == 'mel' else M_IRON)
    key_loc(o, POS[n])
    bpy.ops.mesh.primitive_torus_add(major_radius=rr * 1.0, minor_radius=rr * 0.12, major_segments=32, minor_segments=8)
    hb = bpy.context.object; hb.parent = o; hb.data.materials.append(M_GOLD if n == 'bas' else M_GOLD)
    sp = np.zeros(NF)
    for f in range(1, NF): sp[f] = sp[f - 1] + (POS[n][f, 0] - POS[n][f - 1, 0]) / rr
    o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, sp)
    BALL[n] = o
# the ember carries its own warm light
eml = bpy.data.lights.new('emberlight', 'POINT'); eml.color = (1.0, 0.55, 0.25); eml.shadow_soft_size = 0.2
emo = link(bpy.data.objects.new('emberlight', eml)); key_loc(emo, POS['mel'] + [0, -0.35, 0.25])
for f in range(0, NF, 2): eml.energy = float(18 + 22 * GF[f] + 260 * WARM[f]); eml.keyframe_insert('energy', frame=f)

# ---------------- floors: hearth floor (ember lane), bench (iron-ball lane), yard outside ----------------
LANE_GROUND = {}
for role in MAIN:
    b = BP[role]; r = b.r
    pts = sorted([(h['p'][0], h['p'][2]) for h in b.hits if h['kind'] not in ('lantern',)])
    if role == 'mel':
        for rl in EV['rails']: pts += [(rl['p0'][0], rl['p0'][2]), (rl['p1'][0], rl['p1'][2])]
        lp = EV['lift']['p0']
        for x in np.arange(lp[0], lp[0] + 10, 0.5): pts.append((x, lp[2]))
    else:
        lb = b.hits[-1]['p']
        for x in np.arange(lb[0], lb[0] + 8, 0.5): pts.append((x, lb[2]))
    pts.sort()
    AX = np.array([p[0] for p in pts]); AZ = np.array([p[1] for p in pts])
    X = np.arange(-10.0, AX.max() + 10, 0.4)
    Zg = np.array([AZ[np.abs(AX - x) < 1.2].min() if (np.abs(AX - x) < 1.2).any() else np.nan for x in X])
    ok_ = ~np.isnan(Zg); Zg = np.interp(X, X[ok_], Zg[ok_]) - r - 0.42
    for _ in range(4): Zg = np.minimum(Zg, np.convolve(np.pad(Zg, 2, mode='edge'), np.ones(5) / 5, 'valid'))
    LANE_GROUND[role] = (X, Zg)
def ground(role, x): X, Z = LANE_GROUND[role]; return float(np.interp(x, X, Z))
def yard(x): return -GZB * (x - RN['rain1']['dx']) / VV
X_END = EV['chute']['p1'][0] - 0.15                                   # end wall of the cabin
strip_mesh(LANE_GROUND['mel'][0], LANE_GROUND['mel'][1], -2.2, 0.85, 3.0, M_FLOOR, 'floor_mel')
strip_mesh(LANE_GROUND['bas'][0], LANE_GROUND['bas'][1], 0.85, 2.55, 3.0, M_FLOOR, 'bench', m_front=M_WOOD)
XY = np.arange(-14, 100, 1.0)
strip_mesh(XY, np.array([yard(x) for x in XY]), 2.75, 30.0, 2.0, M_SNOW, 'yard')
# snow outside the end of the cabin (front)
XS = np.arange(X_END, X_END + 30, 0.5)
strip_mesh(XS, np.array([ground('mel', X_END) - 0.05 + 0.04 * math.sin(x * 1.3) for x in XS]), -8.0, 2.76, 3.0, M_SNOW, 'endsnow')

# ---------------- back wall with windows (modules of 2.4 m, each at its bench height) ----------------
MOD = 2.4; WIN_W = 1.6; SILL, HEAD, TOP = 0.3, 1.55, 1.8
WALL_Y0, WALL_Y1 = 2.55, 2.75
FIREPL = [8.4, 24.0, 38.4, EV['bellows']['p'][0]]
WINDOWS = []; FIRES = []
xm = -4.8
while xm < X_END - 0.6:
    zb = ground('bas', xm + MOD / 2)
    x1 = min(xm + MOD, X_END)
    fire_here = [xf for xf in FIREPL if xm <= xf < xm + MOD]
    # post (solid log wall) on the left of the module
    box((xm + 0.4, (WALL_Y0 + WALL_Y1) / 2, zb + TOP / 2 - 0.5), (0.8, WALL_Y1 - WALL_Y0, TOP + 1.0), M_LOGWALL, name='post')
    if fire_here:
        box((xm + 0.4 + (x1 - xm - 0.4) / 2, (WALL_Y0 + WALL_Y1) / 2, zb + TOP / 2 - 0.5), (x1 - xm - 0.8 + 0.01, WALL_Y1 - WALL_Y0, TOP + 1.0),
            M_LOGWALL, name='wallpanel')
        FIRES.append(dict(x=fire_here[0], z=zb))
    elif x1 - xm > 1.0:
        wx0, wx1 = xm + 0.8, x1
        cx = (wx0 + wx1) / 2; ww = wx1 - wx0
        box((cx, (WALL_Y0 + WALL_Y1) / 2, zb + SILL / 2 - 0.5), (ww + 0.01, WALL_Y1 - WALL_Y0, SILL + 1.0), M_LOGWALL, name='sillwall')
        box((cx, (WALL_Y0 + WALL_Y1) / 2, zb + (HEAD + TOP) / 2), (ww + 0.01, WALL_Y1 - WALL_Y0, TOP - HEAD), M_LOGWALL, name='header')
        box((cx, WALL_Y0 - 0.06, zb + SILL + 0.02), (ww, 0.18, 0.06), M_WOOD, name='sill')
        box((cx, WALL_Y0 - 0.06, zb + SILL + 0.06), (ww - 0.05, 0.16, 0.03), M_SNOW, name='sillsnow')
        WINDOWS.append(dict(x=cx, w=ww, z0=zb + SILL, z1=zb + HEAD))
    xm += MOD
# end wall (seen edge-on) and the top beam
box((X_END + 0.1, 1.2, ground('mel', X_END) + 2.3), (0.25, 4.0, 5.2), M_LOGWALL, name='endwall')

# casement windows: two glass halves per window, hinged at the sides; the gust swings them open, the bellows shuts them
OPEN = np.zeros(NF)
for f in range(NF):
    t = T[f]
    if T_GUST - 0.05 <= t < T_BEL + 0.25: OPEN[f] = smoothstep((t - T_GUST + 0.05) / 0.35) * (1 + 0.08 * math.sin(t * 9))
    elif T_BEL + 0.25 <= t < T_BEL + 0.6: OPEN[f] = 1 - smoothstep((t - T_BEL - 0.25) / 0.35)
for wdw in WINDOWS:
    for side in (-1, 1):
        hx = wdw['x'] + side * wdw['w'] / 2
        hg = link(bpy.data.objects.new('hinge', None)); hg.location = (hx, WALL_Y0 + 0.02, (wdw['z0'] + wdw['z1']) / 2)
        hw_ = wdw['w'] / 2
        gl = box((-side * hw_ / 2, 0, 0), (hw_ - 0.04, 0.02, wdw['z1'] - wdw['z0'] - 0.04), M_GLASS, name='pane'); gl.parent = hg
        for zz in (-1, 1):
            fr = box((-side * hw_ / 2, 0, zz * (wdw['z1'] - wdw['z0']) / 2), (hw_, 0.06, 0.06), M_WOOD, name='frame'); fr.parent = hg
        for xx in (0.0, -side * hw_):
            fr = box((xx, 0, 0), (0.06, 0.06, wdw['z1'] - wdw['z0']), M_WOOD, name='frame'); fr.parent = hg
        fr = box((-side * hw_ / 2, 0, 0), (hw_, 0.04, 0.04), M_WOOD, name='mullion'); fr.parent = hg
        hg.rotation_mode = 'XYZ'; key_series(hg, 'rotation_euler', 2, side * OPEN * 1.6 * (0.9 + 0.2 * rng.random()), 1e-3)
    # frost in the corners
    for cx_, cz_ in ((wdw['x'] - wdw['w'] / 2 + 0.15, wdw['z0'] + 0.12), (wdw['x'] + wdw['w'] / 2 - 0.15, wdw['z0'] + 0.12)):
        o = sphere((cx_, WALL_Y0 - 0.005, cz_), 0.22, M_SNOW, (12, 6)); o.scale = (0.3, 0.01, 0.16)

# ---------------- fireplaces (stone, flames, light) and the stove at the start ----------------
FL_LIGHTS = []
def flames(c, w, h, k0, boost=None):
    """a few emissive flame tongues (fx material) at c, size w x h; flicker + the gust / bellows factor"""
    for j in range(5):
        fo = obj(prim('cone', M_FLAME, 10, 1.0, 0.05, 1.0), 'flame'); fo.color = ((1.0, 0.45 + 0.1 * j, 0.1), 1)[0] + (0.9,)
        xx = c[0] + (j - 2) / 2.0 * w * 0.45
        fo.location = (xx, c[1], c[2] + h * 0.45)
        sz = np.array([(0.55 + 0.35 * math.sin(T[f] * (7 + j) + j * 1.7) ** 2) * GF[f] * (1.0 if boost is None else boost[f]) for f in range(NF)])
        key_scale(fo, np.maximum(0.05, sz), (w * 0.22, 0.08, h))
    lt = bpy.data.lights.new('fire', 'POINT'); lt.color = (1.0, 0.55, 0.25); lt.shadow_soft_size = 0.5
    lo = link(bpy.data.objects.new('fire', lt)); lo.location = (c[0], c[1] - 0.6, c[2] + h * 0.6)
    for f in range(0, NF, 2): lt.energy = float(k0 * GF[f] * (1.0 if boost is None else boost[f]) * (1 + 0.6 * WARM[f])); lt.keyframe_insert('energy', frame=f)
    FL_LIGHTS.append(lo)
for fp in FIRES:
    x, zb = fp['x'], fp['z']
    box((x, WALL_Y0 - 0.25, zb + 0.75), (1.9, 0.5, 1.5), M_STONE, name='hearth')
    box((x, WALL_Y0 - 0.51, zb + 0.6), (1.1, 0.04, 0.9), M_DARK, name='firebox')
    box((x, WALL_Y0 - 0.32, zb + 1.6), (2.2, 0.7, 0.2), M_WOOD, name='mantel')
    box((x, WALL_Y0 - 0.1, zb + TOP + 0.4), (1.2, 0.4, 2.0), M_STONE, name='chimneybreast')
    for j in range(3): rod((x - 0.3 + 0.3 * j, WALL_Y0 - 0.6, zb + 0.22), (x - 0.2 + 0.3 * j, WALL_Y0 - 0.85, zb + 0.18), 0.07, M_BARK, 8)
    flames((x, WALL_Y0 - 0.65, zb + 0.25), 0.9, 0.55, 140)
    # candles on the mantel
    for dx in (-0.8, 0.8):
        rod((x + dx, WALL_Y0 - 0.35, zb + 1.7), (x + dx, WALL_Y0 - 0.35, zb + 1.95), 0.04, M_WHITE, 8)
        cf = obj(prim('cone', M_FLAME, 8, 1.0, 0.05, 1.0), 'candle'); cf.location = (x + dx, WALL_Y0 - 0.35, zb + 2.02); cf.color = (1.0, 0.7, 0.3, 0.9)
        key_scale(cf, np.maximum(0.05, GF * (0.9 + 0.1 * np.sin(T * 13 + dx))), (0.03, 0.03, 0.09))
# the stove at the start (it pops the ember out at T_POP)
ST = np.array(EV['stove']['p']); T_POP = EV['stove']['t']
box((ST[0] - 0.1, ST[1] + 0.05, ground('mel', ST[0]) + 0.55), (0.8, 0.8, 1.1), M_DARK, name='stove')
rod((ST[0] - 0.1, ST[1] + 0.3, ground('mel', ST[0]) + 1.1), (ST[0] - 0.1, ST[1] + 0.3, ground('mel', ST[0]) + 6.0), 0.12, M_DARK, 16)
for dx in (-0.3, 0.3):
    rod((ST[0] - 0.1 + dx, ST[1] - 0.25, ground('mel', ST[0]) - 0.1), (ST[0] - 0.1 + dx, ST[1] - 0.25, ground('mel', ST[0]) + 0.05), 0.05, M_DARK, 8)
hd = link(bpy.data.objects.new('stovedoor', None)); hd.location = (ST[0] + 0.3, ST[1] - 0.06, ST[2])
dr = box((-0.0, 0, 0), (0.05, 0.42, 0.42), M_GOLD, name='door'); dr.parent = hd
hd.rotation_mode = 'XYZ'
key_series(hd, 'rotation_euler', 1, np.array([0.0 if t < T_POP - 0.05 else min(1.0, (t - T_POP + 0.05) / 0.12) * 1.3 * (1 - 0.3 * min(1, max(0, t - T_POP - 0.4))) for t in T]))
flames((ST[0] - 0.15, ST[1] - 0.1, ST[2] - 0.25), 0.35, 0.3, 60)

# ---------------- keys: copper keys (ember), log drums (iron ball) ----------------
PIECES = []
KEY_TH = 0.08
for role in MAIN:
    b = BP[role]; r = b.r; ext = LN[role]['ext']
    for h in b.hits:
        if h['kind'] not in ('key', 'drum', 'bellows'): continue
        nv = np.array(h['n']); top = np.array(h['p']) - r * nv; col = pcol(h['pitch'])
        zf = ground(role, top[0])
        if role == 'bas':
            if h['kind'] == 'bellows': continue
            hd = obj(prim('cyl', M_DRUM, 24), 'drumhead')
            hd.matrix_world = Matrix.Translation(Vector(top - nv * 0.03)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((ext[0] + 0.02, ext[0] + 0.02, 0.06, 1))
            st = rod(top - nv * 0.06, (top[0], top[1], zf - 0.2), ext[0] + 0.01, M_BIRCH, 16)
            ring = obj(prim('cyl', M_BARK, 24), 'drumring'); ring.matrix_world = Matrix.Translation(Vector(top - nv * 0.08)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((ext[0] + 0.035, ext[0] + 0.035, 0.05, 1))
            parts = [hd, st, ring]; flash(hd, h['t'], col, before=(0.85, 0.72, 0.5)); fo = hd
        else:
            ko = obj(prim('bcube', M_KEY), 'key')
            ko.matrix_world = Matrix.Translation(Vector(top - nv * KEY_TH / 2)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((2 * ext[0], 2 * ext[1], KEY_TH, 1))
            po = rod(top - nv * KEY_TH, (top[0], top[1], zf - 0.2), 0.025, M_COPPER, 8)
            parts = [ko, po]; flash(ko, h['t'], col, before=(0.9, 0.5, 0.3)); fo = ko
        bump(fo, h['t'], nv, 0.02)
        popper(parts, (top[0], top[1], zf - 0.2), h['t'] - KL, h['t'] + KS - 0.3, grow=0.3, shrink=0.3)
        PIECES.append(dict(t=h['t'], top=top, kind=h['kind'], role=role, col=col))
# the iron ball's last drum (final chord): a big kettle drum
hf = BP['bas'].hits[-1]; ftop = np.array(hf['p']) - BP['bas'].r * np.array([0, 0, 1.0]); zff = ground('bas', ftop[0])
fd_ = obj(prim('cyl', M_DRUM, 32), 'findrum'); fd_.location = tuple(ftop - [0, 0, 0.03]); fd_.scale = (0.42, 0.42, 0.06)
flash(fd_, FINAL, (1.0, 0.7, 0.3), before=(0.85, 0.72, 0.5))
rod(ftop - [0, 0, 0.06], (ftop[0], ftop[1], zff - 0.2), 0.4, M_COPPER, 24, r1=0.3)
popper([fd_], tuple(ftop), FINAL - 1.2, None)
PIECES.append(dict(t=FINAL, top=ftop, kind='findrum', role='bas', col=(1, 0.7, 0.3)))
# copper rails for the long notes
for rl in EV['rails']:
    p0 = np.array(rl['p0']); p1 = np.array(rl['p1']); r = BP['mel'].r
    zf = ground('mel', p0[0]); parts = []
    for yy in (-0.11, 0.11):
        parts.append(rod(p0 + [-0.15, yy, -r * 0.82], p1 + [0.05, yy, -r * 0.82], 0.022, M_COPPER, 8))
    for xx in (p0[0] - 0.1, p1[0]):
        parts.append(rod((xx, 0, p0[2] - r * 0.82), (xx, 0, zf - 0.2), 0.02, M_COPPER, 6))
        parts.append(rod((xx, -0.11, p0[2] - r * 0.82), (xx, 0.11, p0[2] - r * 0.82), 0.018, M_COPPER, 6))
    popper(parts, ((p0[0] + p1[0]) / 2, 0, zf - 0.2), rl['t_land'] - KL, rl['t_off'] + KS - 0.3)

# the bellows (iron ball's drum at bar 14): top board = drum, leather folds, nozzle into the fireplace behind
bl = EV['bellows']; bn = np.array(bl['n']); btop = np.array(bl['p']) - BP['bas'].r * bn; zfb = ground('bas', btop[0])
bel_root = link(bpy.data.objects.new('bellows', None)); bel_root.location = (btop[0], btop[1], zfb)
SQ = np.array([1.0 - 0.35 * math.exp(-max(0, t - T_BEL) / 0.25) * (t >= T_BEL) * min(1, (t - T_BEL) / 0.05 + 0.0) for t in T])
bh = btop[2] - zfb
body = obj(prim('cyl', M_LEATHER, 16), 'bellowsbody'); body.parent = bel_root; body.location = (0, 0, 0)
key_scale(body, np.ones(NF), (0.34, 0.3, 1.0))
key_series(body, 'scale', 2, bh * SQ - 0.05, 1e-4); key_series(body, 'location', 2, (bh * SQ - 0.05) / 2, 1e-4)
board = obj(prim('cyl', M_DRUM, 24), 'bellowsboard'); board.parent = bel_root; board.scale = (0.3, 0.3, 0.06)
key_series(board, 'location', 2, bh * SQ - 0.03, 1e-4)
flash(board, T_BEL, (1.0, 0.6, 0.2), before=(0.6, 0.35, 0.15))
rod((btop[0], btop[1] + 0.25, zfb + 0.25), (btop[0], WALL_Y0 - 0.55, zfb + 0.4), 0.06, M_GOLD, 10, r1=0.03)
PIECES.append(dict(t=T_BEL, top=btop, kind='bellows', role='bas', col=(1.0, 0.6, 0.2)))

# ---------------- the rain: glass chimes outside (static rows: every chime is struck once) + the drops ----------------
for cid, c in LAY['chimes'].items():
    R_ = RN[c['role']]; ext = R_['ext']; nv = np.array(c['n']); top = np.array(c['top']); col = pcol(c['pitch'], 0.15)
    ch = obj(prim('bcube', M_CHIME), 'chime')
    ch.matrix_world = Matrix.Translation(Vector(top - nv * ext[2])) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((2 * ext[0], 2 * ext[1], 2 * ext[2], 1))
    for t_ in c['times'][:1]: flash(ch, t_, col, before=(0.55, 0.7, 0.85))
    rod(top - nv * 2 * ext[2], (top[0], top[1], yard(top[0]) - 0.1), 0.012, M_DARK, 6)
    for t_ in c['times']: PIECES.append(dict(t=t_, top=top, kind='chime', role=c['role'], col=col))
# the drops: a pool of little ice-blue balls; each lives from its release (high above) until it sinks into the snow
M_DROP = mat('drop', (0.7, 0.9, 1.0), 0.05, **{'Emission Color': (0.6, 0.85, 1.0), 'Emission Strength': 3.0, 'Transmission Weight': 0.3})
drops = sorted(LAY['drops'], key=lambda d: d['t_rel'])
def drop_at(d, t):
    if t < d['t']: u = t - d['t_rel']; return np.array(d['p_rel']) + np.array([0, 0, -0.5 * G * u * u])
    u = t - d['t']; return np.array(d['p']) + np.array(d['vout']) * u + np.array([0, 0, -0.5 * G * u * u])
for d in drops:
    vz = d['vout'][2]; h = d['p'][2] - d['ground']
    d['t_end'] = d['t'] + (vz + math.sqrt(vz * vz + 2 * G * h)) / G
slots = []
for d in drops:
    for s_ in slots:
        if s_[-1]['t_end'] < d['t_rel'] - 0.1: s_.append(d); break
    else: slots.append([d])
print('raindrop pool', len(slots))
HIDE = np.array([0.0, 0.0, -80.0])
for s_ in slots:
    o = sphere((0, 0, 0), RN['rain1']['r'], M_DROP, (12, 6)); P = np.zeros((NF, 3)); act = np.zeros(NF, bool)
    for d in s_:
        f0 = int(math.ceil(d['t_rel'] * FPS)); f1 = int(math.floor(d['t_end'] * FPS))
        for f in range(max(0, f0), min(NF, f1 + 1)): P[f] = drop_at(d, T[f]); act[f] = True
    key_loc(o, hold(P, act)); key_scale(o, np.where(act, RN['rain1']['r'], 1e-4), (1, 1, 1.6))

# ---------------- hit rings / sparks ----------------
NRING = 20
ring_loc = np.zeros((NRING, NF, 3)); ring_a = np.zeros((NRING, NF)); ring_s = np.full((NRING, NF), 1e-3)
for k, P_ in enumerate(sorted([p for p in PIECES if p['kind'] != 'chime'], key=lambda b: b['t'])):
    r = k % NRING; f0 = int(round(P_['t'] * FPS))
    for df in range(14):
        f = f0 + df
        if f >= NF: break
        u = df / 13; ring_loc[r, f] = P_['top'] + np.array([0, -0.3, 0.02]); ring_s[r, f] = 0.3 + 1.1 * (1 - (1 - u) ** 2)
        ring_a[r, f] = (1 - u) ** 1.6
for r in range(NRING):
    bpy.ops.mesh.primitive_torus_add(major_radius=0.25, minor_radius=0.014, major_segments=32, minor_segments=5, rotation=(math.pi / 2, 0, 0))
    o = bpy.context.object; o.data.materials.append(M_FX); o.color = (1.0, 0.8, 0.5, 0)
    for f in range(1, NF):
        if ring_a[r, f] == 0: ring_loc[r, f] = ring_loc[r, f - 1]
    key_loc(o, ring_loc[r]); key_scale(o, ring_s[r]); key_series(o, 'color', 3, ring_a[r], 1e-3)
# ember sparks: a few tiny sparks fly up from every melody key (warm)
NSP = 36
melk = sorted([p for p in PIECES if p['role'] == 'mel'], key=lambda b: b['t'])
SPP = np.tile(HIDE, (NSP, NF, 1)); SPA = np.zeros((NSP, NF))
for k, P_ in enumerate(melk):
    for j in range(3):
        s = (3 * k + j) % NSP; f0 = int(round(P_['t'] * FPS)); a0 = rng.uniform(0.4, 2.7); sp_ = rng.uniform(0.8, 1.6)
        for df in range(16):
            f = f0 + df
            if f >= NF: break
            u = df / FPS
            SPP[s, f] = P_['top'] + [sp_ * math.cos(a0) * u, -0.25, sp_ * math.sin(a0) * u + 0.6 * u - 0.5 * 1.5 * u * u]
            SPA[s, f] = (1 - df / 16) * 0.9
for s in range(NSP):
    o = obj(prim('ico', M_FX, 1), 'spark'); o.color = (1.0, 0.55, 0.15, 0)
    key_loc(o, hold(SPP[s], SPA[s] > 0)); key_scale(o, np.full(NF, 0.025)); key_series(o, 'color', 3, SPA[s], 1e-3)
# chime splashes: a small ring on every chime hit
NCR = 16
cr_loc = np.tile(HIDE, (NCR, NF, 1)); cr_a = np.zeros((NCR, NF)); cr_s = np.full((NCR, NF), 1e-3)
for k, P_ in enumerate(sorted([p for p in PIECES if p['kind'] == 'chime'], key=lambda b: b['t'])):
    r = k % NCR; f0 = int(round(P_['t'] * FPS))
    for df in range(8):
        f = f0 + df
        if f >= NF: break
        u = df / 7; cr_loc[r, f] = P_['top'] + [0, -0.02, 0.03]; cr_s[r, f] = 0.15 + 0.5 * (1 - (1 - u) ** 2); cr_a[r, f] = (1 - u) ** 1.5
for r in range(NCR):
    bpy.ops.mesh.primitive_torus_add(major_radius=0.25, minor_radius=0.02, major_segments=24, minor_segments=4)
    o = bpy.context.object; o.data.materials.append(M_FX); o.color = (0.6, 0.85, 1.0, 0)
    key_loc(o, hold(cr_loc[r], cr_a[r] > 0)); key_scale(o, cr_s[r]); key_series(o, 'color', 3, cr_a[r], 1e-3)

# ---------------- the shelf flap that drops the iron ball, the opening stove burst ----------------
PB = np.array(EV['shelf']['p']); T_FLAP = EV['shelf']['t']; rb = BP['bas'].r
box((PB[0] - 0.55, PB[1] + 0.05, PB[2] - rb - 0.05), (0.7, 0.6, 0.06), M_WOOD, name='shelf')
rod((PB[0] - 0.85, PB[1] + 0.05, PB[2] - rb - 0.05), (PB[0] - 0.85, PB[1] + 0.05, ground('bas', PB[0]) - 0.1), 0.04, M_WOOD, 8)
hg = link(bpy.data.objects.new('flaphinge', None)); hg.location = (PB[0] - rb - 0.04, PB[1], PB[2] - rb - 0.03)
fp = box((rb + 0.04, 0, 0), (2 * rb + 0.08, 2 * rb + 0.1, 0.05), M_WOOD, name='flap'); fp.parent = hg
hg.rotation_mode = 'XYZ'; key_series(hg, 'rotation_euler', 1, np.array([0.0 if t < T_FLAP - 0.02 else min(1.0, (t - T_FLAP + 0.02) / 0.15) ** 2 * 1.6 for t in T]))
for k in range(40):
    sh = obj(prim('ico', M_FX, 1), 'popspark'); sh.color = (1.0, 0.5, 0.12, 0)
    a0 = rng.uniform(-0.3, 1.9); sp_ = rng.uniform(1.0, 3.0)
    P = np.tile(HIDE, (NF, 1)); A = np.zeros(NF)
    for f in range(NF):
        u = T[f] - T_POP
        if u < 0 or u > 1.0: continue
        P[f] = ST + np.array([0.3 + sp_ * math.cos(a0) * u, -0.2 - 0.6 * u, sp_ * math.sin(a0) * u - 0.5 * 3.0 * u * u])
        A[f] = max(0, 1 - u / 1.0)
    key_loc(sh, hold(P, A > 0)); key_scale(sh, np.where(A > 0, 0.035, 1e-4)); key_series(sh, 'color', 3, A, 1e-3)

# ---------------- finale: inclined chain elevator, smoke-jack fan, roof chute, the hero snowman's lantern ----------------
LF = EV['lift']; CH = EV['chute']; LT = EV['lantern']
pc = np.array(LF['p0']); ptop = np.array(LF['top']); ud = (ptop - pc) / np.linalg.norm(ptop - pc); rM = BP['mel'].r
nrm = np.array([-ud[2], 0, ud[0]])                                   # perpendicular in the xz plane (up-left)
for yy in (-0.3, 0.3):
    rod(pc + [0, yy, 0] - nrm * (rM + 0.25) - ud * 0.4, ptop + [0, yy, 0] - nrm * (rM + 0.25) + ud * 0.3, 0.05, M_STEEL, 8)
for s_ in np.arange(0, np.linalg.norm(ptop - pc) + 0.01, 1.4):
    q = pc + ud * s_ - nrm * (rM + 0.25)
    rod(q + [0, -0.3, 0], q + [0, 0.3, 0], 0.03, M_STEEL, 6)
    zf = ground('mel', q[0])
    if q[2] - zf > 0.3: rod(q + [0, 0.35, 0], (q[0], 0.35, zf - 0.1), 0.05, M_STEEL, 8)
# sprockets + chain links running with the bucket
SPR = []
for q in (pc - nrm * (rM + 0.25) - ud * 0.4, ptop - nrm * (rM + 0.25) + ud * 0.3):
    sp = link(bpy.data.objects.new('sprocket', None)); sp.location = tuple(q)
    w_ = obj(prim('cyl', M_GOLD, 20), 'sprocketw'); w_.parent = sp; w_.scale = (0.22, 0.22, 0.06); w_.rotation_euler = (math.pi / 2, 0, 0)
    for j in range(3): box((0, 0, 0), (0.46, 0.04, 0.04), M_STEEL, rot=(0, j * math.pi / 3, 0)).parent = sp
    SPR.append(sp)
S_LIFT = np.array([np.dot(POS['mel'][f] - pc, ud) if LF['t_go'] <= T[f] <= LF['t_top'] else (0.0 if T[f] < LF['t_go'] else np.linalg.norm(ptop - pc)) for f in range(NF)])
for sp in SPR: sp.rotation_mode = 'XYZ'; key_series(sp, 'rotation_euler', 1, S_LIFT / 0.22, 1e-3)
Lc = np.linalg.norm(ptop - pc) + 0.7
for j in range(16):
    o = box((0, 0, 0), (0.08, 0.5, 0.05), M_DARK, name='chainlink'); P = np.zeros((NF, 3))
    for f in range(NF):
        s_ = (j * 2 * Lc / 16 + S_LIFT[f]) % (2 * Lc)
        base = pc - nrm * (rM + 0.25) - ud * 0.4
        P[f] = base + ud * s_ if s_ < Lc else base + ud * (2 * Lc - s_) - nrm * 0.3
    key_loc(o, P)
# the bucket bell (catches the ember at bar 16 beat 3, rides up, tips it onto the chute)
bk = link(bpy.data.objects.new('bucket', None))
BKP = np.array([pc if T[f] < LF['t_go'] else (POS['mel'][f] if T[f] <= LF['t_top'] else ptop) for f in range(NF)])
key_loc(bk, BKP)
bkr = link(bpy.data.objects.new('bucketrot', None)); bkr.parent = bk
bkr.rotation_mode = 'XYZ'
key_series(bkr, 'rotation_euler', 1, np.array([0.0 if t < LF['t_tip'] else -0.9 * smoothstep((t - LF['t_tip']) / 0.2) * (1 - smoothstep((t - LF['t_tip'] - 0.8) / 0.6)) for t in T]), 1e-3)
cupm = obj(prim('bell', M_CUP, 28), 'bucketbell'); cupm.parent = bkr
cupm.matrix_parent_inverse = Matrix(); cupm.location = (0, 0, -rM - 0.02 + 0.26); cupm.scale = (rM + 0.08, rM + 0.08, -0.26)
cupm.color = (1.0, 0.75, 0.3, 0); flash(cupm, LF['t_c'], (1.0, 0.7, 0.3), before=(0.9, 0.6, 0.3))
for yy in (-1, 1): rod((0, yy * (rM + 0.1), -rM), (0, yy * 0.3, -rM - 0.1), 0.02, M_STEEL, 6).parent = bkr
PIECES.append(dict(t=LF['t_c'], top=pc - [0, 0, rM], kind='bucket', role='mel', col=(1, 0.8, 0.4)))
# the grand fireplace with the chimney and the smoke-jack fan (drives the chain through the shaft)
xf = ptop[0] - 0.2; zbf = ground('bas', xf)
box((xf, WALL_Y0 - 0.25, zbf + 0.75), (1.9, 0.5, 1.5), M_STONE, name='grandhearth')
box((xf, WALL_Y0 - 0.51, zbf + 0.6), (1.1, 0.04, 0.9), M_DARK, name='grandfirebox')
flames((xf, WALL_Y0 - 0.65, zbf + 0.25), 0.9, 0.6, 160)
box((xf, WALL_Y0 - 0.1, (zbf + 1.5 + ptop[2] + 1.0) / 2), (1.0, 0.6, ptop[2] + 1.0 - zbf - 1.5), M_STONE, name='chimney')
fan = link(bpy.data.objects.new('fan', None)); fan.location = (xf, WALL_Y0 - 0.45, ptop[2] + 0.55)
for j in range(6):
    bl_ = box((0.28, 0, 0), (0.5, 0.03, 0.14), M_GOLD, rot=(0.4, 0, 0)); bl_.parent = fan
    bl_.matrix_parent_inverse = Matrix.Rotation(j * math.pi / 3, 4, 'Y')
hub = sphere((xf, WALL_Y0 - 0.45, ptop[2] + 0.55), 0.09, M_GOLD, (10, 6))
fan.rotation_mode = 'XYZ'; key_series(fan, 'rotation_euler', 1, -(S_LIFT / 0.22 + T * 1.5 * (T > T_BEL)), 1e-3)
rod((xf, WALL_Y0 - 0.45, ptop[2] + 0.55), tuple(SPR[1].location + Vector((0, 0.0, 0))), 0.035, M_STEEL, 8)
text('Smoke-jack', (xf + 0.1, WALL_Y0 - 0.5, ptop[2] + 1.2), 0.11, M_TXT_G)
# the roof of the cabin's end and the copper chute on it
p_r0 = np.array(CH['p0']); p_e = np.array(CH['p1']); dch = (p_e - p_r0) / np.linalg.norm(p_e - p_r0)
roof_n = np.array([-dch[2], 0, dch[0]])
for yy in (-0.12, 0.12): rod(p_r0 + [-0.25, yy, 0] - roof_n * rM * 0.85, p_e + [0.02, yy, 0] - roof_n * rM * 0.85, 0.022, M_COPPER, 8)
rc = (p_r0 + p_e) / 2 - roof_n * (rM + 0.12); Lr = np.linalg.norm(p_e - p_r0) + 1.6
ro = box(tuple(rc + [-0.5, 1.0, 0]), (Lr + 1.0, 4.6, 0.16), M_WOOD, rot=(0, -math.atan2(dch[2], dch[0]), 0), name='roof')
rs = box(tuple(rc + [-0.5, 1.0, 0] + roof_n * 0.12), (Lr + 1.0, 4.6, 0.1), M_SNOW, rot=(0, -math.atan2(dch[2], dch[0]), 0), name='roofsnow')
for k in range(9):          # icicles along the eave
    ic = obj(prim('cone', M_GLASS, 8, 1.0, 0.0, 1.0), 'icicle'); ic.location = (p_e[0] + 0.1, -1.0 + 0.5 * k, p_e[2] - rM - 0.35)
    ic.rotation_euler = (math.pi, 0, 0); ic.scale = (0.05, 0.05, 0.18 + 0.12 * rng.random())

FROST = [('VIVALDI', 'Winter · Largo', 'ヴィヴァルディ「冬」'), ('The Four Seasons', 'L\'Inverno · 1725', '「四季」より 第2楽章'),
         ('Classical Music', 'Relaxing', 'クラシック音楽'), ('Marble Machine', 'Satisfying', 'ビー玉マシン'),
         ('Rain =', 'Pizzicato', '雨音はピチカート'), ('Baroque Music', 'A. Vivaldi', 'バロック音楽'),
         ('Fireplace', 'ASMR', '暖炉の音 ASMR'), ('Snowy Night', 'Cozy Cabin', '雪の夜の山小屋'),
         ('Winter Largo', 'Violin', '冬 ラルゴ'), ('Marble Music', 'Calm', '癒しのビー玉音楽')]
def signboard(wdw, lines):
    en1, en2, jp = lines
    cx = wdw['x']; zb = wdw['z0'] - SILL; z = zb + TOP + 0.62; bw = 2.25
    box((cx, WALL_Y0 + 0.05, z), (bw, 0.08, 1.0), M_WOOD, rot=(-0.15, 0, 0), name='signboard')
    box((cx, WALL_Y0 + 0.06, z + 0.52), (bw + 0.1, 0.2, 0.08), M_SNOW, rot=(-0.15, 0, 0))
    for xx in (cx - bw / 2 + 0.2, cx + bw / 2 - 0.2): rod((xx, WALL_Y0 + 0.1, zb + TOP - 0.05), (xx, WALL_Y0 + 0.1, z - 0.4), 0.04, M_WOOD, 6)
    rot = (math.pi / 2 - 0.15, 0, 0)
    fit_w(text(en1, (cx, WALL_Y0 - 0.04, z + 0.26), 0.15, M_TXT_G, rot=rot), bw - 0.2)
    fit_w(text(en2, (cx, WALL_Y0 - 0.02, z - 0.04), 0.11, M_TXT, rot=rot), bw - 0.2)
    fit_w(text(jp, (cx, WALL_Y0 - 0.0, z - 0.32), 0.1, M_TXT, rot=rot), bw - 0.2)
fw = [w_ for w_ in WINDOWS if w_['x'] > 0.5][::2]
for wdw, lines in zip(fw, FROST): signboard(wdw, lines)
# ---------------- snowmen: a row in the yard, the hero with the lantern outside the end of the cabin ----------------
SNOWMEN = []
def snowman(base, s, face_dir, t_smile, lantern=None, hero=False):
    """three snowballs, coal eyes, carrot nose, a mouth of 5 coal dots (frown -> smile at t_smile), scarf, branch arms"""
    root = link(bpy.data.objects.new('snowman', None)); root.location = tuple(base)
    parts = []
    for rr, zz in ((0.62, 0.55), (0.46, 1.38), (0.34, 2.02)):
        o = sphere((0, 0, zz * s), rr * s, M_SNOWMAN, (24, 12)); o.parent = root
    head_c = np.array([0, 0, 2.02 * s]); fd = np.array(face_dir) / np.linalg.norm(face_dir)
    side = np.cross([0, 0, 1], fd)
    for sd in (-1, 1):
        o = sphere(tuple(head_c + fd * 0.3 * s + side * sd * 0.12 * s + [0, 0, 0.08 * s]), 0.045 * s, M_BLACK, (10, 6)); o.parent = root
    nose = obj(prim('cone', M_CARROT, 12, 1.0, 0.0, 1.0), 'nose'); nose.parent = root
    nq = Vector((0, 0, 1)).rotation_difference(Vector(tuple(fd)))
    nose.matrix_parent_inverse = Matrix(); nose.matrix_basis = Matrix.Translation(Vector(tuple(head_c + fd * 0.42 * s))) @ nq.to_matrix().to_4x4() @ Matrix.Diagonal((0.055 * s, 0.055 * s, 0.26 * s, 1))
    # mouth dots: curvature c goes -1 (frown) -> +1 (smile)
    CURV = np.array([-1.0 if t < t_smile else -1.0 + 2.0 * smoothstep((t - t_smile) / 0.35) for t in T])
    for j in range(5):
        u = (j - 2) / 2.0
        o = sphere((0, 0, 0), 0.032 * s, M_BLACK, (8, 6)); o.parent = root
        P = np.array([head_c + fd * 0.3 * s + side * u * 0.15 * s + [0, 0, -0.1 * s + 0.06 * s * c * (u * u - 0.5)] for c in CURV])
        key_loc(o, P)
    sc_ = obj(prim('cyl', M_SCARF, 20), 'scarf'); sc_.parent = root
    sc_.location = (0, 0, 1.72 * s); sc_.scale = (0.4 * s, 0.4 * s, 0.1 * s); sc_.color = (0.9, 0.15, 0.12, 1) if hero else (rng.choice([0.2, 0.9]), 0.3 + 0.4 * rng.random(), rng.choice([0.2, 0.8]), 1)
    tail = box(tuple([0, 0, 1.5 * s] + fd * 0.38 * s + side * 0.12 * s), (0.14 * s, 0.03 * s, 0.4 * s), M_SCARF, name='scarftail'); tail.parent = root; tail.color = sc_.color
    for sd in (-1, 1):
        a0 = np.array([0, 0, 1.45 * s]) + side * sd * 0.42 * s
        a1 = a0 + side * sd * 0.55 * s + [0, 0, 0.35 * s]
        arm = rod(a0, a1, 0.025 * s, M_BARK, 6); arm.parent = root
    # shivering until the smile (cold!), then a happy little hop
    SH = np.array([0.04 * math.sin(T[f] * 31 + base[0]) * (T[f] < t_smile) for f in range(NF)])
    root.rotation_mode = 'XYZ'; key_series(root, 'rotation_euler', 0, SH, 1e-4)
    HOP = np.array([base[2] + (0.18 * s * math.sin(math.pi * min(1, (t - t_smile - 0.35) / 0.4)) if t_smile + 0.35 <= t <= t_smile + 0.75 else 0.0) for t in T])
    key_series(root, 'location', 2, HOP, 1e-4)
    SNOWMEN.append(dict(root=root, base=base, t=t_smile))
    return root
# the hero (front, outside the end wall), holding the lantern up on a stick
lp = np.array(LT['p']); zgh = ground('mel', lp[0]) - 0.05
hero_base = np.array([lp[0] + 0.75, -0.15, zgh])
hs = 1.0
snowman(hero_base, hs, (-1.0, -0.8, 0), FINAL + 0.08, hero=True)
rod(hero_base + [-0.55, 0, 1.6], lp + [0.0, 0, 0.55], 0.03, M_BARK, 6)
# the lantern: a gold cage with a bell cup inside (the ember lands in it -> it lights)
lb = obj(prim('bell', M_CUP, 28), 'lanternbell'); lb.location = tuple(lp + [0, 0, -rM - 0.02 + 0.24]); lb.scale = (rM + 0.07, rM + 0.07, -0.24)
lb.color = (1.0, 0.75, 0.3, 0); flash(lb, FINAL, (1.0, 0.75, 0.35), before=(0.8, 0.6, 0.3))
for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
    rod(lp + [dx * (rM + 0.06), dy * (rM + 0.06), -rM - 0.02], lp + [dx * (rM + 0.06), dy * (rM + 0.06), rM + 0.35], 0.015, M_GOLD, 6)
rf = obj(prim('cone', M_GOLD, 4, 1.0, 0.1, 1.0), 'lanternroof'); rf.location = tuple(lp + [0, 0, rM + 0.45]); rf.scale = (rM + 0.15, rM + 0.15, 0.2)
rf.rotation_euler = (0, 0, math.pi / 4)
PIECES.append(dict(t=FINAL, top=lp - [0, 0, rM], kind='lantern', role='mel', col=(1, 0.8, 0.4)))
# the row in the yard; the warmth runs back along it after the final chord
ROW_X = [x for x in np.arange(1.0, X_END + 1.0, 7.4)]
ROW_X.sort(key=lambda x: -x)
LANT = []
SIGN_X = [w_['x'] for w_ in fw[:len(FROST)]]
def clear_of_signs(x):
    for _ in range(20):
        near = [sx for sx in SIGN_X if abs(sx - x) < 1.5]
        if not near: return x
        x = near[0] + 1.55 * (1 if x >= near[0] else -1)
    return x
ROW_X = [clear_of_signs(x) for x in ROW_X]
for k, x in enumerate(ROW_X):
    t_s = FINAL + 0.45 + 0.17 * k
    s = 1.2 + 0.1 * rng.random()
    base = np.array([x, 6.0 + rng.uniform(-0.2, 0.2), yard(x) + 0.75])
    mound = sphere((x, base[1], yard(x) - 0.3), 1.0, M_SNOW, (24, 12)); mound.scale = (1.6, 1.4, 1.1)
    snowman(base, s, (-0.25, -1.0, 0), t_s)
    lc = base + [0.8, -0.4, 1.0]
    rod(base + [0.8, -0.4, -0.1], lc - [0, 0, 0.12], 0.03, M_BARK, 6)
    lg = sphere(tuple(lc), 0.13, M_LAMP, (12, 8)); lg.color = (0.15, 0.12, 0.1, 1)
    lg.keyframe_insert('color', frame=int(t_s * FPS)); lg.color = (1.0, 0.65, 0.3, 1); lg.keyframe_insert('color', frame=int(t_s * FPS) + 6)
    LANT.append((t_s, lc))
for t_s, lc in LANT[:6]:
    lt = bpy.data.lights.new('lantern', 'POINT'); lt.color = (1.0, 0.6, 0.3); lt.shadow_soft_size = 0.3
    lo = link(bpy.data.objects.new('lantern', lt)); lo.location = tuple(lc + [0, -0.3, 0])
    lt.energy = 0.0; lt.keyframe_insert('energy', frame=int(t_s * FPS)); lt.energy = 60.0; lt.keyframe_insert('energy', frame=int(t_s * FPS) + 8)
hl = bpy.data.lights.new('herolantern', 'POINT'); hl.color = (1.0, 0.65, 0.3); hl.shadow_soft_size = 0.2
hlo = link(bpy.data.objects.new('herolantern', hl)); hlo.location = tuple(lp + [0, -0.4, 0.1])
hl.energy = 0.0; hl.keyframe_insert('energy', frame=int(FINAL * FPS)); hl.energy = 220.0; hl.keyframe_insert('energy', frame=int(FINAL * FPS) + 5)
hl.energy = 120.0; hl.keyframe_insert('energy', frame=int(FINAL * FPS) + 40)
# warm burst at the final chord
for k in range(80):
    sh = obj(prim('ico', M_FX, 1), 'finspark'); sh.color = (1.0, 0.7, 0.3, 0)
    a0 = rng.uniform(0, 2 * math.pi); el = rng.uniform(-0.3, 1.2); sp_ = rng.uniform(1.0, 3.5)
    P = np.tile(HIDE, (NF, 1)); A = np.zeros(NF)
    for f in range(NF):
        u = T[f] - FINAL
        if u < 0 or u > 2.5: continue
        P[f] = lp + np.array([sp_ * math.cos(a0) * math.cos(el) * u, sp_ * math.sin(a0) * math.cos(el) * u * 0.6, sp_ * math.sin(el) * u - 0.5 * 1.2 * u * u])
        A[f] = max(0, 1 - u / 2.5)
    key_loc(sh, hold(P, A > 0)); key_scale(sh, np.where(A > 0, 0.04, 1e-4)); key_series(sh, 'color', 3, A, 1e-3)

# ---------------- gust: snow blows in through the open windows ----------------
NG = 140
for k in range(NG):
    o = obj(prim('ico', M_FX, 1), 'gustflake'); o.color = (0.9, 0.95, 1.0, 0)
    P = np.tile(HIDE, (NF, 1)); A = np.zeros(NF)
    t_k = T_GUST + 0.2 + rng.uniform(0, T_BEL - T_GUST - 0.4); life = rng.uniform(0.8, 1.4)
    # path: from a window near the camera's x at t_k across the room towards the camera
    x0 = float(np.interp(t_k, T, POS['mel'][:, 0])) + rng.uniform(-2.5, 4.0); z0 = ground('bas', x0) + rng.uniform(0.6, 2.0)
    vel = np.array([rng.uniform(-1.5, 0.5), -rng.uniform(4.0, 7.0), rng.uniform(-0.8, 0.4)])
    for f in range(NF):
        u = T[f] - t_k
        if u < 0 or u > life: continue
        P[f] = np.array([x0, WALL_Y0 + 0.4, z0]) + vel * u + [0.2 * math.sin(7 * u + k), 0, 0.15 * math.sin(5 * u + k)]
        A[f] = 0.8 * min(1, u / 0.1) * (1 - u / life)
    key_loc(o, hold(P, A > 0)); key_scale(o, np.where(A > 0, rng.uniform(0.025, 0.05), 1e-4)); key_series(o, 'color', 3, A, 1e-3)

# ---------------- wall lamps along the hall ----------------
for k, x in enumerate(np.arange(4.0, X_END - 2, 7.2)):
    zb = ground('bas', x)
    lt = bpy.data.lights.new('lamp', 'POINT'); lt.color = (1.0, 0.62, 0.32); lt.shadow_soft_size = 0.3
    lo = link(bpy.data.objects.new('lamp', lt)); lo.location = (x, 1.0, zb + 2.6)
    for f in range(0, NF, 3): lt.energy = float(120 * GF[f] * (1 + 0.3 * WARM[f])); lt.keyframe_insert('energy', frame=f)

# ---------------- text: frosted-window keywords, lane labels, telops ----------------
TR = (math.pi / 2 - 0.3, 0, 0)
label('Melody · Solo Violin', 'メロディ（独奏ヴァイオリン）', (2.2, -1.0, ground('mel', 2.2) + 0.05), 0.12, M_TXT_O, k=0.6, rot=(math.pi / 2 - 0.9, 0, 0))
label('Bass · Cello', '低音（チェロ）', (2.6, 0.83, ground('bas', 2.6) - 0.3), 0.11, M_TXT_G, k=0.6, rot=(math.pi / 2, 0, 0))
label('Rain · Pizzicato', '雨（ピチカート）', (3.4, 4.4, yard(3.4) + 3.1), 0.13, M_TXT_B, k=0.6, rot=TR)

# ---------------- sky, moon, snow layers, aurora ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bgw = w.node_tree.nodes['Background']; bgw.inputs['Color'].default_value = (0.05, 0.08, 0.18, 1)
key_val(bgw.inputs['Strength'], 0.35 + 0.25 * WARM, 6)
moon = bpy.data.lights.new('moon', 'SUN'); moon.angle = 0.05; moon.color = (0.6, 0.72, 1.0)
mo = link(bpy.data.objects.new('moon', moon)); mo.rotation_euler = (math.radians(55), math.radians(-15), math.radians(20))
key_val_l = [(f, 0.35 + 0.5 * WARM[f]) for f in range(0, NF, 6)]
for f, v in key_val_l: moon.energy = float(v); moon.keyframe_insert('energy', frame=f)
SKY = bpy.data.materials.new('sky'); SKY.use_nodes = True; nt = SKY.node_tree; nt.nodes.clear()
tc = nt.nodes.new('ShaderNodeTexCoord'); sepz = nt.nodes.new('ShaderNodeSeparateXYZ')
rw = nt.nodes.new('ShaderNodeValToRGB'); rw.color_ramp.elements[0].color = (0.12, 0.16, 0.3, 1); rw.color_ramp.elements[1].color = (0.01, 0.015, 0.05, 1)
vor = nt.nodes.new('ShaderNodeTexVoronoi'); vor.inputs['Scale'].default_value = 260.0
mr = nt.nodes.new('ShaderNodeMapRange'); mr.inputs['From Min'].default_value = 0.06; mr.inputs['From Max'].default_value = 0.0
stars = nt.nodes.new('ShaderNodeMath'); stars.operation = 'MULTIPLY'
wav = nt.nodes.new('ShaderNodeTexWave'); wav.inputs['Scale'].default_value = 1.6; wav.inputs['Distortion'].default_value = 6.0; wav.bands_direction = 'X'
aur = nt.nodes.new('ShaderNodeValToRGB'); aur.color_ramp.elements[0].color = (0, 0, 0, 1); aur.color_ramp.elements[1].color = (0.1, 1.0, 0.5, 1)
aur.color_ramp.elements[0].position = 0.6
aum = nt.nodes.new('ShaderNodeMath'); aum.operation = 'MULTIPLY'
add1 = nt.nodes.new('ShaderNodeMixRGB'); add1.blend_type = 'ADD'; add1.inputs['Fac'].default_value = 1.0
add2 = nt.nodes.new('ShaderNodeMixRGB'); add2.blend_type = 'ADD'; add2.inputs['Fac'].default_value = 1.0
aumix = nt.nodes.new('ShaderNodeMixRGB'); aumix.blend_type = 'MULTIPLY'; aumix.inputs['Fac'].default_value = 1.0
em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Strength'].default_value = 1.0; out = nt.nodes.new('ShaderNodeOutputMaterial')
nt.links.new(tc.outputs['Generated'], sepz.inputs[0]); nt.links.new(sepz.outputs['Y'], rw.inputs['Fac'])
nt.links.new(tc.outputs['Generated'], vor.inputs['Vector']); nt.links.new(vor.outputs['Distance'], mr.inputs['Value'])
nt.links.new(mr.outputs[0], stars.inputs[0])
nt.links.new(tc.outputs['Generated'], wav.inputs['Vector']); nt.links.new(wav.outputs['Fac'], aur.inputs['Fac'])
nt.links.new(rw.outputs['Color'], add1.inputs['Color1']); nt.links.new(stars.outputs[0], add1.inputs['Color2'])
nt.links.new(aur.outputs['Color'], aumix.inputs['Color1'])
nt.links.new(add1.outputs[0], add2.inputs['Color1']); nt.links.new(aumix.outputs[0], add2.inputs['Color2'])
nt.links.new(add2.outputs[0], em.inputs['Color']); nt.links.new(em.outputs[0], out.inputs[0])
key_val(stars.inputs[1], 0.25 + 2.5 * WARM, 6)
aumix.inputs['Color2'].default_value = (0, 0, 0, 1)
for f in range(0, NF, 6):
    v = float(0.9 * WARM[f]); aumix.inputs['Color2'].default_value = (v, v, v, 1); aumix.inputs['Color2'].keyframe_insert('default_value', frame=f)
skyo = obj(prim('plane', SKY), 'sky'); skyo.location = (35, 70, 0); skyo.rotation_euler = (math.pi / 2, 0, 0); skyo.scale = (240, 120, 1)
mdisc = sphere((-6, 66, 22), 2.2, emis('moondisc', (0.85, 0.9, 1.0), 3.0), (24, 12))
# forest silhouettes far behind the yard
M_PINE = mat('pine', (0.05, 0.12, 0.1), 0.8)
for k, x0 in enumerate(np.arange(-12, X_END + 30, 2.6)):
    y0 = 13 + rng.uniform(0, 6); zg = yard(x0) - 0.3; hgt = rng.uniform(3.0, 5.5)
    o = obj(prim('cone', M_PINE, 10, 1.0, 0.0, 1.0), 'pine'); o.location = (x0 + rng.uniform(-0.8, 0.8), y0, zg + hgt / 2); o.scale = (hgt * 0.28, hgt * 0.28, hgt)
    o2 = obj(prim('cone', M_SNOW, 10, 1.0, 0.0, 1.0), 'pinesnow'); o2.location = (o.location[0], y0 - 0.1, zg + hgt * 0.72); o2.scale = (hgt * 0.15, hgt * 0.15, hgt * 0.45)


def flake_layer(dist, scale, speed, name, strength=1.4):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    tc = nt.nodes.new('ShaderNodeTexCoord'); mpn = nt.nodes.new('ShaderNodeMapping')
    vor = nt.nodes.new('ShaderNodeTexVoronoi'); vor.inputs['Scale'].default_value = scale
    mr = nt.nodes.new('ShaderNodeMapRange'); mr.inputs['From Min'].default_value = 0.085; mr.inputs['From Max'].default_value = 0.02
    emn = nt.nodes.new('ShaderNodeEmission'); emn.inputs['Color'].default_value = (0.95, 0.97, 1.0, 1)
    trn = nt.nodes.new('ShaderNodeBsdfTransparent'); mix = nt.nodes.new('ShaderNodeMixShader'); o_ = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(tc.outputs['UV'], mpn.inputs['Vector']); nt.links.new(mpn.outputs[0], vor.inputs['Vector'])
    nt.links.new(vor.outputs['Distance'], mr.inputs['Value']); nt.links.new(mr.outputs[0], mix.inputs['Fac'])
    nt.links.new(trn.outputs[0], mix.inputs[1]); nt.links.new(emn.outputs[0], mix.inputs[2]); nt.links.new(mix.outputs[0], o_.inputs[0])
    # wind: the blizzard blows sideways, much harder in the gust; the snow stops after the final chord
    wind = np.array([1.0 + (2.5 if T_GUST <= t < T_BEL else 0.0) for t in T]) * (1 - 0.8 * SNOWSTOP)
    pos = np.cumsum(wind) / FPS * speed
    for f in range(0, NF, 3):
        mpn.inputs['Location'].default_value[1] = float(pos[f]); mpn.inputs['Location'].keyframe_insert('default_value', index=1, frame=f)
        mpn.inputs['Location'].default_value[0] = float(-0.9 * pos[f]); mpn.inputs['Location'].keyframe_insert('default_value', index=0, frame=f)
    key_val(emn.inputs['Strength'], strength * (1 - SNOWSTOP), 3)
    bpy.ops.mesh.primitive_plane_add(size=1.0); o = bpy.context.object; o.name = name; o.data.materials.append(m)
    o.visible_shadow = False; o.visible_glossy = False
    return o, dist
FLAKES = [flake_layer(5.0, 9.0, 0.05, 'flake1', 1.0), flake_layer(9.0, 16.0, 0.035, 'flake2', 0.7), flake_layer(16.0, 26.0, 0.025, 'flake3', 0.5)]

# ---------------- camera ----------------
def smooth(Xa, k):
    Y = Xa.copy(); a = 1 - math.exp(-1 / FPS / k)
    for f in range(1, len(Xa)): Y[f] = Y[f - 1] + (Xa[f] - Y[f - 1]) * a
    for f in range(len(Xa) - 2, -1, -1): Y[f] = Y[f + 1] + (Y[f] - Y[f + 1]) * a
    return Y
FOC = np.zeros((NF, 3)); WID = np.zeros(NF); EL = np.zeros(NF); AZ = np.zeros(NF)
YC = 1.8
for f in range(NF):
    t = T[f]
    pm, pb = POS['mel'][f], POS['bas'][f]
    xs = [pm[0], pb[0]]
    z0 = min(pm[2], pb[2]) - 0.9; z1 = max(pm[2], pb[2]) + 2.0
    FOC[f] = (np.mean(xs) + 0.7, YC, (z0 + z1) / 2 + 0.35); WID[f] = max(5.3, (z1 - z0 + 3.4) * 9 / 16, max(xs) - min(xs) + 3.4)
    EL[f] = math.radians(17)
    if t < T0 + 1.0:                                 # the opening: a little wider, looking at the stove and the first snowman
        u = smoothstep((t - 0.0) / (T0 + 1.0))
        FOC[f] = FOC[f] * u + np.array([0.6, 2.4, ground('mel', 0) + 2.2]) * (1 - u); WID[f] = WID[f] * u + 6.8 * (1 - u)
    if t > LF['t_c'] - 1.0:                          # the finale: frame the elevator, the chute and the hero
        u = smoothstep((t - LF['t_c'] + 1.0) / 2.0)
        tgt = np.array([0.5 * (pm[0] + lp[0]) + 0.3, 0.9, 0.5 * (pm[2] + lp[2]) + 0.2])
        FOC[f] = FOC[f] * (1 - u) + tgt * u; WID[f] = WID[f] * (1 - u) + 7.6 * u
    if t > FINAL + 0.2:                              # pull back and turn to look back along the row of snowmen
        u = smoothstep((t - FINAL - 0.2) / 3.2)
        FOC[f] = FOC[f] * (1 - u) + np.array([lp[0] - 6.0, 4.5, lp[2] - 0.5]) * u; WID[f] = WID[f] * (1 - u) + 13.0 * u
        AZ[f] = math.radians(38) * u; EL[f] = math.radians(17 + 8 * u)
WID = np.clip(WID, 5.2, 14.0)
for k, kk in ((0, 0.6), (1, 0.8), (2, 0.8)): FOC[:, k] = smooth(FOC[:, k], kk)
WID = smooth(WID, 0.8)
LENS = 30.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 600
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24
    CP[f] = (FOC[f, 0] + D * math.cos(EL[f]) * math.sin(AZ[f]), FOC[f, 1] - D * math.cos(EL[f]) * math.cos(AZ[f]), FOC[f, 2] + D * math.sin(EL[f]))
    TP[f] = FOC[f]
key_loc(cam, CP); key_loc(tgt, TP)
for o, dist in FLAKES:
    o.parent = cam; o.location = (0, 0, -dist); o.rotation_euler = (0, 0, 0)
    hh = dist * 24 / LENS; o.scale = (hh * 9 / 16 * 1.1, hh * 1.1, 1)


# ---------------- telops (parented to the camera) ----------------
def telop(lines, y, t_in, t_out, dist=4.0):
    """lines: [(text, size, material)], stacked downwards from y (fraction of the half frame height, +1 = top)"""
    hh = dist * 24 / LENS / 2
    root = link(bpy.data.objects.new('telop', None)); root.parent = cam; root.location = (0, y * hh, -dist)
    zz = 0.0; objs = []
    for s_, size, mt in lines:
        o = text(s_, (0, zz, 0), size * hh, mt, rot=(0, 0, 0), ext=0.0); o.parent = root
        fit_w(o, hh * 9 / 16 * 2 * 0.92); objs.append(o)
        zz -= size * hh * TXT_K * 0.62 * 1.25
    key_scale(root, pop_scale(t_in, t_out, 0.3, 0.3, 1.08))
    return root
telop([('Will the snowmen smile?', 0.075, M_TXT), ('雪だるまは笑うかな？', 0.05, M_TXT_B)], 0.78, 0.05, 4.6)
telop([('VIVALDI – "Winter"', 0.07, M_TXT_G), ('The Four Seasons · II. Largo', 0.045, M_TXT), ('Classical Music × Marble Machine', 0.038, M_TXT_O),
       ('ヴィヴァルディ「四季」より「冬」第2楽章', 0.034, M_TXT)], 0.78, 4.9, 11.0)
telop([('Relaxing · Satisfying · ASMR', 0.045, M_TXT_B), ('癒しのクラシック × ビー玉', 0.034, M_TXT)], 0.8, 11.4, 17.0)
telop([('BLIZZARD!', 0.09, M_TXT_B), ('吹雪で火が消えそう！', 0.045, M_TXT)], 0.78, T_GUST + 0.2, T_GUST + 3.6)
telop([('The bellows saves the fire!', 0.06, M_TXT_O), ('ふいごで火が復活！', 0.045, M_TXT)], 0.78, T_BEL + 0.1, T_BEL + 3.4)
telop([('Chimney lift ↑', 0.06, M_TXT_G), ('煙突リフトで外へ！', 0.045, M_TXT)], 0.8, LF['t_c'] + 0.2, LF['t_c'] + 3.0)
telop([('THEY ALL SMILED!', 0.085, M_TXT_G), ('みんな笑顔に！', 0.05, M_TXT)], 0.8, FINAL + 0.6, None)

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
        f = int(round(float(tt_) * FPS))
        print('t', tt_, 'WID', round(WID[f], 2), 'FOC', np.round(FOC[f], 2), {n: np.round(POS[n][f], 2).tolist() for n in BP})
