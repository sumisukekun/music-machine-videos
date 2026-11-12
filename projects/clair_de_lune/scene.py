"""Clair de Lune - 'Can you hear the moonlight?'  A moonlit Japanese water garden of terraced ponds, side view at night.
Every melody note = the silver / gold twin marbles striking two lotus-leaf keys (or the bars of a bamboo slide); every
harmony note = a water drop from a bamboo spout ringing a glass bowl; every bass note = the moon-stone. Each note fills the
moon in the sky a little (counter MOON %); the last chord makes it full and puts the full moon on the pond.
layout.py plans all motion (constant gravity, bounces never add speed, zip line, pulley lift, slides, see-saw, shishi-odoshi,
rolling channel); this file builds and animates it.  Fonts: jp_sub.otf / jp_serif.otf = Noto Sans / Serif CJK JP (SIL OFL)
subsets of the characters used here (make_font.py); en_bold.ttf / en_serif_bold.ttf = DejaVu Sans / Serif Bold.
python3 scene.py -- test t1 t2 ... | range f0 f1 | probe t ..."""
import os, sys, json, math, time, colorsys
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); TW = json.load(open(f'{HERE}/twins.json'))
DR = json.load(open(f'{HERE}/drops.json')); BS = json.load(open(f'{HERE}/bass.json')); NOTES = json.load(open(f'{HERE}/notes.json'))
LAY = json.load(open(f'{HERE}/layout.json'))
FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.4) * FPS); T = np.arange(NF) / FPS
G = LAY['G']; EV = LAY['ev']; LN = LAY['lanes']; OFFS = LAY['off']
RT, RB, RD = LAY['R']['twin'], LAY['R']['moon'], LAY['R']['drop']
T0, E8 = E['t0'], E['e8']
rng = np.random.default_rng(17)

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


def glowmat(name, base_col=None, gain=8.0, base=0.0, rough=0.3, metal=0.0, coat=0.5):
    """object colour = base colour (or base_col); object alpha drives emission (hit flash) in the object colour"""
    m = mat(name, base_col or (1, 1, 1), rough, **{'Coat Weight': coat, 'Metallic': metal}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    oi = nt.nodes.new('ShaderNodeObjectInfo')
    if base_col is None: nt.links.new(oi.outputs['Color'], b.inputs['Base Color'])
    nt.links.new(oi.outputs['Color'], b.inputs['Emission Color'])
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


def key_loc(o, P, step=1):
    if step == 1:
        for k in range(3): key_series(o, 'location', k, P[:, k])
        return
    for f in list(range(0, NF, step)) + [NF - 1]:
        o.location = tuple(map(float, P[f])); o.keyframe_insert('location', frame=f)


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
    elif kind == 'ring':                                   # flat ring (ripple), radius 1, width a[1]
        segs = a[0]; w = a[1]; vi, vo = [], []
        for k in range(segs):
            ph = 2 * math.pi * k / segs
            vi.append(bm.verts.new(((1 - w) * math.cos(ph), (1 - w) * math.sin(ph), 0))); vo.append(bm.verts.new((math.cos(ph), math.sin(ph), 0)))
        for k in range(segs): bm.faces.new((vi[k], vo[k], vo[(k + 1) % segs], vi[(k + 1) % segs]))
    elif kind in ('bowl', 'leaf', 'flower', 'bell'):
        if kind == 'bowl':      # glass bowl bell, mouth up, rim at z = 0, radius 1, depth 0.75
            prof = [(0.0, -0.75), (0.45, -0.72), (0.78, -0.55), (0.95, -0.25), (1.0, 0.0), (0.93, 0.0), (0.88, -0.22), (0.72, -0.48), (0.42, -0.63), (0.0, -0.66)]
        elif kind == 'leaf':    # lotus leaf: a shallow cup, top at z = 0, radius 1, with a rim
            prof = [(0.0, -0.05), (0.5, -0.03), (0.9, 0.02), (1.0, 0.06), (0.97, 0.0), (0.5, -0.09), (0.0, -0.12)]
        elif kind == 'bell':    # bronze bowl gong, mouth up
            prof = [(0.0, -0.5), (0.5, -0.47), (0.85, -0.32), (1.0, 0.0), (0.92, 0.0), (0.8, -0.25), (0.45, -0.38), (0.0, -0.41)]
        else:                   # lotus flower cup
            prof = [(0.0, -0.4), (0.35, -0.38), (0.7, -0.2), (1.0, 0.15), (0.9, 0.12), (0.6, -0.12), (0.3, -0.25), (0.0, -0.27)]
        segs = a[0]; rings = []
        for k in range(segs):
            ph = 2 * math.pi * k / segs
            rr_ = 1.0 + (0.1 * math.cos(5 * ph) if kind == 'flower' else 0.0)
            rings.append([bm.verts.new((r * rr_ * math.cos(ph), r * rr_ * math.sin(ph), z + (0.12 * max(0, math.cos(5 * ph)) * r if kind == 'flower' else 0))) for r, z in prof])
        for k in range(segs):
            r0, r1 = rings[k], rings[(k + 1) % segs]
            for j in range(len(prof) - 1): bm.faces.new((r0[j], r1[j], r1[j + 1], r0[j + 1]))
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    me = bpy.data.meshes.new(kind); bm.to_mesh(me); bm.free()
    if kind in ('sph', 'cone', 'cyl', 'bcube', 'bowl', 'leaf', 'flower', 'bell'):
        for p in me.polygons: p.use_smooth = True
    if m: me.materials.append(m)
    _MC[key] = me; return me


def obj(me, name='o'): return link(bpy.data.objects.new(name, me))


def box(c, size, m, rot=None, name='box', kind='cube'):
    o = obj(prim(kind, m), name); o.location = c; o.scale = size
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
    if _latin(s): cu.font = FONT_ENS if cu.font == FONT_S else FONT_EN; cu.size *= EN_K
    cu.align_x = align; cu.align_y = 'CENTER'
    o = link(bpy.data.objects.new('txt', cu)); o.location = loc; o.rotation_euler = rot; o.data.materials.append(m); return o


def fit_w(o, w):
    bpy.context.view_layer.update()
    if w and o.dimensions.x > w: k = w / o.dimensions.x; o.scale = (k, k, k)
    return o


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
MAIN = ('silver', 'gold', 'moon')
POS = {n: np.array([BP[n].at(t) for t in T]) for n in MAIN}


def smooth(Xa, k):
    Y = Xa.copy(); a = 1 - math.exp(-1 / FPS / k)
    for f in range(1, len(Xa)): Y[f] = Y[f - 1] + (Xa[f] - Y[f - 1]) * a
    for f in range(len(Xa) - 2, -1, -1): Y[f] = Y[f + 1] + (Y[f] - Y[f + 1]) * a
    return Y


def pulse(times, decay=5.0, length=16, amp=1.0):
    a = np.zeros(NF)
    for t in times:
        f0 = int(round(t * FPS)); m_ = np.arange(max(0, f0), min(NF, f0 + length))
        a[m_] = np.maximum(a[m_], amp * np.exp(-(m_ - f0) / decay))
    return a


def flash(o, t, col, n=None, base=0.0, dip=0.03):
    fh = int(round(t * FPS))
    o.color = (*col, base); o.keyframe_insert('color', index=3, frame=max(0, fh - 1))
    for df, a in ((0, 1.0), (5, 0.55), (22, max(base, 0.12))):
        o.color = (*col, a); o.keyframe_insert('color', index=3, frame=fh + df)
    if n is not None:
        b0 = o.location.copy(); nv = Vector(n)
        for df, d in ((-1, 0), (1, -dip), (4, dip * 0.3), (8, 0)):
            o.location = b0 + nv * d; o.keyframe_insert('location', frame=max(0, fh + df))


def strip_mesh(X, Z, y0, y1, th, m, name):
    bm = bmesh.new(); vs = []
    for x, z in zip(X, Z):
        vs.append([bm.verts.new((x, y0, z)), bm.verts.new((x, y1, z)), bm.verts.new((x, y1, z - th)), bm.verts.new((x, y0, z - th))])
    for a, b in zip(vs, vs[1:]):
        for k in range(4): bm.faces.new((a[k], b[k], b[(k + 1) % 4], a[(k + 1) % 4]))
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new(name, me)); o.data.materials.append(m)
    return o


def riser(objs, t_hit, anchor, lead=1.2, rise=0.4, depth=0.9):
    """a group rises out of the water (and unfolds) between t_hit-lead and t_hit-lead+rise, then stays"""
    anchor = Vector(anchor)
    e = link(bpy.data.objects.new('rise', None)); e.location = anchor
    for o in objs: o.parent = e; o.matrix_parent_inverse = Matrix.Translation(-anchor)
    Z = np.zeros(NF); S = np.zeros(NF)
    for f in range(NF):
        u = (T[f] - (t_hit - lead)) / rise
        if u <= 0: Z[f] = -depth; S[f] = 0.001
        elif u < 1: w = u * u * (3 - 2 * u); Z[f] = -depth * (1 - w); S[f] = max(0.001, w * (1 + 0.15 * math.sin(math.pi * w)))
        else: Z[f] = 0.0; S[f] = 1.0
    key_series(e, 'location', 2, anchor.z + Z, 1e-4)
    for k in (0, 1): key_series(e, 'scale', k, S, 1e-4)
    key_series(e, 'scale', 2, np.maximum(0.001, np.minimum(1, S * 1.0)), 1e-4)
    return e


def pcol(p, light=0.0):
    h = ((p % 12) * 7 % 12) / 12.0                                # circle of fifths around the hue wheel
    r, g, b = colorsys.hsv_to_rgb((0.55 + h) % 1.0, 0.7, 1.0)
    return tuple(min(1, x * (1 - light) + light) for x in (r, g, b))


# ---------------- materials ----------------
M_SILVER = mat('silver', (0.85, 0.9, 1.0), 0.05, **{'Metallic': 0.9, 'Coat Weight': 1.0, 'Emission Color': (0.75, 0.85, 1.0), 'Emission Strength': 0.6})
M_GOLDB = mat('goldball', (1.0, 0.76, 0.32), 0.1, **{'Metallic': 0.85, 'Coat Weight': 1.0, 'Emission Color': (1.0, 0.7, 0.3), 'Emission Strength': 0.5})
M_DROP = mat('drop', (0.55, 0.85, 1.0), 0.02, **{'Coat Weight': 1.0, 'Emission Color': (0.45, 0.8, 1.0), 'Emission Strength': 2.2})
M_LEAF = glowmat('leaf', base_col=(0.08, 0.32, 0.12), gain=7.0, base=0.0, rough=0.35, coat=0.6)
M_STEM = mat('stem', (0.05, 0.2, 0.07), 0.6)
M_BOWL = glowmat('bowl', gain=7.0, base=0.18, rough=0.05, coat=1.0)
M_BAR = glowmat('bar', base_col=(0.62, 0.5, 0.28), gain=7.0, base=0.0, rough=0.35, coat=0.4)
M_BRONZE = glowmat('bronze', base_col=(0.55, 0.36, 0.16), gain=5.0, base=0.0, rough=0.25, metal=0.9)
M_BAMBOO = mat('bamboo', (0.28, 0.42, 0.12), 0.35, **{'Coat Weight': 0.6})
M_BAMBOOD = mat('bamboodark', (0.12, 0.2, 0.06), 0.5)
M_WOOD = mat('wood', (0.22, 0.12, 0.06), 0.5, **{'Coat Weight': 0.3})
M_ROPE = mat('rope', (0.55, 0.45, 0.3), 0.8)
M_STONE = mat('stone', (0.16, 0.17, 0.19), 0.75)
M_STONED = mat('stoned', (0.07, 0.075, 0.09), 0.85)
M_MOSS = mat('moss', (0.03, 0.07, 0.04), 0.9)
M_WALLST = mat('wallstone', (0.06, 0.07, 0.07), 0.85)
M_LANTW = emis('lanternwin', (1.0, 0.62, 0.28), 6.0)
M_PAPER = mat('paper', (1.0, 0.8, 0.55), 0.6, **{'Emission Color': (1.0, 0.6, 0.3), 'Emission Strength': 2.5})
M_PINK = glowmat('pink', base_col=(0.95, 0.5, 0.65), gain=5.0, base=0.0, rough=0.4, coat=0.3)
M_FX = fxmat('fx', 6.0)
M_FALL = mat('waterfall', (0.6, 0.75, 0.9), 0.1, **{'Alpha': 0.35, 'Emission Color': (0.5, 0.65, 0.9), 'Emission Strength': 0.35})
M_TXT = emis('txtw', (1.0, 0.96, 0.9), 3.0); M_TXT_G = emis('txtg', (1.0, 0.8, 0.45), 4.0); M_TXT_B = emis('txtb', (0.7, 0.85, 1.0), 4.0)
M_SIGN = mat('signboard', (0.12, 0.07, 0.04), 0.5, **{'Coat Weight': 0.4})
# moon-stone: pale cratered stone, its glow grows with the moon
M_MOONB = bpy.data.materials.new('moonball'); M_MOONB.use_nodes = True; _nt = M_MOONB.node_tree; _b = _nt.nodes['Principled BSDF']
_nz = _nt.nodes.new('ShaderNodeTexNoise'); _nz.inputs['Scale'].default_value = 3.5; _nz.inputs['Detail'].default_value = 6.0
_cr = _nt.nodes.new('ShaderNodeValToRGB'); _cr.color_ramp.elements[0].color = (0.45, 0.46, 0.5, 1); _cr.color_ramp.elements[1].color = (0.92, 0.92, 0.88, 1)
_cr.color_ramp.elements[0].position = 0.35; _cr.color_ramp.elements[1].position = 0.65
_nt.links.new(_nz.outputs['Fac'], _cr.inputs['Fac']); _nt.links.new(_cr.outputs['Color'], _b.inputs['Base Color'])
_nt.links.new(_cr.outputs['Color'], _b.inputs['Emission Color']); _b.inputs['Roughness'].default_value = 0.7
_oi = _nt.nodes.new('ShaderNodeObjectInfo'); _mu = _nt.nodes.new('ShaderNodeMath'); _mu.operation = 'MULTIPLY'; _mu.inputs[1].default_value = 4.0
_nt.links.new(_oi.outputs['Alpha'], _mu.inputs[0]); _nt.links.new(_mu.outputs[0], _b.inputs['Emission Strength'])
# water: dark mirror with a slow ripple bump
M_WATER = bpy.data.materials.new('water'); M_WATER.use_nodes = True; _nt = M_WATER.node_tree; _b = _nt.nodes['Principled BSDF']
_b.inputs['Base Color'].default_value = (0.004, 0.012, 0.03, 1); _b.inputs['Roughness'].default_value = 0.04; _b.inputs['Coat Weight'].default_value = 0.6
_tc = _nt.nodes.new('ShaderNodeTexCoord'); _nz = _nt.nodes.new('ShaderNodeTexNoise'); _nz.inputs['Scale'].default_value = 1.2
_bp = _nt.nodes.new('ShaderNodeBump'); _bp.inputs['Strength'].default_value = 0.12
_nt.links.new(_tc.outputs['Object'], _nz.inputs['Vector']); _nt.links.new(_nz.outputs['Fac'], _bp.inputs['Height']); _nt.links.new(_bp.outputs['Normal'], _b.inputs['Normal'])

# ---------------- balls ----------------
BALL = {}
moon_frac_src = sorted(set((round(n[0], 5), n[3]) for n in NOTES))
TOTAL = len(moon_frac_src)
struck = np.array([sum(1 for t, p in moon_frac_src if t <= T[f] + 1e-6) for f in range(NF)])
FRAC = struck / TOTAL                                            # share of the score already struck
FRAC_S = smooth(FRAC.astype(float), 0.12)
for n in MAIN:
    rr = BP[n].r
    bpy.ops.mesh.primitive_uv_sphere_add(radius=rr, segments=40, ring_count=20); o = bpy.context.object; o.name = n
    for p in o.data.polygons: p.use_smooth = True
    o.data.materials.append({'silver': M_SILVER, 'gold': M_GOLDB, 'moon': M_MOONB}[n])
    key_loc(o, POS[n])
    sp = np.zeros(NF)
    for f in range(1, NF): sp[f] = sp[f - 1] + (POS[n][f, 0] - POS[n][f - 1, 0]) / rr
    o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, sp)
    BALL[n] = o
mg = np.array([0.12 + 0.5 * FRAC_S[f] + (2.2 * min(1, (T[f] - FINAL) / 0.5) if T[f] > FINAL else 0.0) for f in range(NF)])
BALL['moon'].color = (1, 1, 1, 0.1); key_series(BALL['moon'], 'color', 3, mg, 1e-3)
for n, col, en in (('silver', (0.7, 0.82, 1.0), 9.0), ('gold', (1.0, 0.75, 0.4), 7.0)):
    ld = bpy.data.lights.new('l' + n, 'POINT'); ld.color = col; ld.energy = en; ld.shadow_soft_size = 0.2
    lo = link(bpy.data.objects.new('l' + n, ld)); lo.parent = BALL[n]
mlt = bpy.data.lights.new('lmoon', 'POINT'); mlt.color = (0.8, 0.85, 1.0); mlt.shadow_soft_size = 0.4
mlo = link(bpy.data.objects.new('lmoon', mlt)); mlo.parent = BALL['moon']
for f in range(0, NF, 3): mlt.energy = float(4 + 40 * mg[f]); mlt.keyframe_insert('energy', frame=f)

# ---------------- water terraces (the level only ever steps down) ----------------
cons = []
for n in ('silver', 'gold'):
    for h in BP[n].hits: cons.append((h['p'][0], h['p'][2] - RT - 0.85))
for h in BP['moon'].hits: cons.append((h['p'][0], h['p'][2] - RB - 0.55))
for p in EV['channel']['path']: cons.append((p[0], p[2] - RB - 0.5))
cons.append((EV['gong']['p'][0], EV['gong']['p'][2] - RB - 1.1)); cons.append((EV['flowers']['c'][0], EV['flowers']['c'][2] - 0.9))
cons.append((EV['lift']['p_in'][0], EV['lift']['p_in'][2] - 0.75)); cons.append((EV['seesaw']['piv'][0], EV['seesaw']['piv'][2] - 1.1))
cons.append((EV['shishi']['piv'][0], EV['shishi']['piv'][2] - 1.0))
for r_ in EV['ramps']:
    if 'bars' in r_:
        for bar in r_['bars']: cons.append((bar['p'][0], bar['p'][2] - 0.9))
CX = np.array([c[0] for c in cons]); CZ = np.array([c[1] for c in cons])
WX = np.arange(-14.0, CX.max() + 22, 0.25)
WZ = np.array([CZ[np.abs(CX - x) < 1.6].min() if (np.abs(CX - x) < 1.6).any() else np.nan for x in WX])
ok = ~np.isnan(WZ); WZ = np.interp(WX, WX[ok], WZ[ok]); WZ = np.minimum.accumulate(WZ)
STEP = 0.7
WZ = np.floor(WZ / STEP) * STEP
def water_z(x): return float(np.interp(x, WX, WZ))
terr = []; j0 = 0
for j in range(1, len(WX) + 1):
    if j == len(WX) or WZ[j] != WZ[j0]: terr.append((WX[j0], WX[j - 1] + 0.25, WZ[j0])); j0 = j
YW0, YW1 = -7.0, 2.65
for k, (x0, x1, z) in enumerate(terr):
    box(((x0 + x1) / 2, (YW0 + YW1) / 2, z - 0.02), (x1 - x0 + 0.02, YW1 - YW0, 0.04), M_WATER, name='water')
    if k + 1 < len(terr):
        z2 = terr[k + 1][2]
        box((x1, (YW0 + YW1) / 2, (z + z2) / 2 - 0.03), (0.05, YW1 - YW0, z - z2 + 0.05), M_FALL, name='fall')
        box((x1 - 0.12, (YW0 + YW1) / 2, (z + z2) / 2 - 0.1), (0.22, YW1 - YW0, z - z2 + 0.0), M_STONED, name='weir')
        for j in range(6):                                          # spray where each fall lands
            sp_ = sphere((x1 + 0.15, rng.uniform(-3.0, 2.2), z2 + 0.03), 0.06, M_FX, (8, 6)); sp_.color = (0.8, 0.9, 1.0, 0.25)
print('terraces', len(terr), 'water z', round(WZ[0], 2), '->', round(WZ[-1], 2))
# back bank: a mossy stone wall rising from the water to the bowls, with the bowls on posts
BANKX = np.arange(WX[0], WX[-1], 0.5)
btops = np.array([d['top'][0] for d in EV['drops']]); bz = np.array([d['top'][2] for d in EV['drops']])
BZ = np.array([bz[np.abs(btops - x) < 2.5].min() - 0.55 if (np.abs(btops - x) < 2.5).any() else np.nan for x in BANKX])
ok = ~np.isnan(BZ); BZ = np.interp(BANKX, BANKX[ok], BZ[ok])
BZ = np.minimum.accumulate(BZ)
def bank_z(x): return float(np.interp(x, BANKX, BZ))
strip_mesh(BANKX, BZ, 2.65, 7.5, 0.3, M_MOSS, 'bank')
strip_mesh(BANKX, BZ, 2.65, 2.75, 30.0, M_STONED, 'bankwall')
for x in np.arange(BANKX[0], BANKX[-1], 0.9):                      # wall stones
    zb_ = bank_z(x); zw_ = water_z(x)
    for zz in np.arange(zw_ + 0.2, zb_ - 0.1, 0.5):
        box((x + rng.uniform(-0.15, 0.15), 2.62, zz + rng.uniform(-0.06, 0.06)), (rng.uniform(0.55, 0.85), 0.12, rng.uniform(0.3, 0.46)), M_WALLST, name='wallstone', kind='bcube')
# bamboo grove + far hill (silhouettes)
for x in np.arange(BANKX[0], BANKX[-1], 0.55):
    zb_ = bank_z(x); h = rng.uniform(2.6, 4.6); yy = rng.uniform(5.2, 7.2)
    rod((x, yy, zb_ - 0.2), (x + rng.uniform(-0.3, 0.3), yy, zb_ + h), rng.uniform(0.05, 0.08), M_BAMBOOD, 6)
HX = np.arange(BANKX[0] - 10, BANKX[-1] + 10, 2.0)
HZ = np.array([bank_z(min(max(x, BANKX[0]), BANKX[-1])) + 3.2 + 1.2 * math.sin(0.11 * x) + 0.8 * math.sin(0.27 * x + 1) for x in HX])
strip_mesh(HX, HZ, 14.0, 15.0, 40.0, mat('hill', (0.01, 0.02, 0.03), 0.9), 'hill')

# ---------------- ripples ----------------
RIPPLES = []
def ripple(x, y, z, t, size=1.2, dur=1.6, amp=0.5, col=(0.75, 0.88, 1.0)):
    RIPPLES.append((x, y, z, t, size, dur, amp, col))
def build_ripples():
    for x, y, z, t, size, dur, amp, col in RIPPLES:
        for k, dl in enumerate((0.0, 0.25)):
            o = obj(prim('ring', M_FX, 48, 0.06), 'ripple'); o.location = (x, y, z + 0.012); o.color = (*col, 0)
            S = np.zeros(NF); A = np.zeros(NF)
            for f in range(NF):
                u = (T[f] - t - dl) / dur
                if 0 <= u <= 1: S[f] = 0.05 + size * (1 - (1 - u) ** 2) * (1 - 0.3 * k); A[f] = amp * (1 - u) ** 1.5 * (1 - 0.4 * k)
            S = np.maximum(S, 0.001)
            for c in (0, 1): key_series(o, 'scale', c, S, 1e-4)
            key_series(o, 'color', 3, A, 1e-3)

# ---------------- lotus keys of the twins (rise out of the water before they are struck) ----------------
LEAF_R = 0.21
for n in ('silver', 'gold'):
    for h in BP[n].hits:
        if h['kind'] not in ('lotus', 'latch'): continue
        nv = np.array(h['n']); top = np.array(h['p']) - RT * nv; col = pcol(h['pitch'], 0.15)
        zw = water_z(top[0])
        if h['kind'] == 'lotus':
            o = obj(prim('leaf', M_LEAF, 28), 'leaf')
            o.matrix_world = Matrix.Translation(Vector(top)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((LEAF_R, LEAF_R, LEAF_R, 1))
            o.color = (*col, 0.0); flash(o, h['t'], col, nv, base=0.1)
            stem = rod(top - nv * 0.03, (top[0] + 0.04, top[1], zw - 0.1), 0.022, M_STEM, 6)
            parts = [o, stem]
        else:                                                         # the latch lever key (bronze)
            o = obj(prim('bcube', M_BRONZE), 'latchkey')
            o.matrix_world = Matrix.Translation(Vector(top - nv * 0.03)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((0.42, 0.42, 0.06, 1))
            o.color = (*col, 0.0); flash(o, h['t'], col, nv, base=0.1, dip=0.06)
            stem = rod(top - nv * 0.06, (top[0], top[1], zw - 0.1), 0.035, M_STONE, 8)
            parts = [o, stem]
        riser(parts, h['t'], top)
        ripple(top[0], top[1], zw, h['t'], size=1.0, amp=0.55, col=tuple(0.5 + 0.5 * c for c in col))
# latch cord: from the lever keys up to the gate of the moon-stone perch
lk = [h for h in BP['silver'].hits if h['kind'] == 'latch'][0]
pr = EV['perch']['p']
rod((lk['p'][0], 0.0, lk['p'][2] - RT - 0.08), (lk['p'][0], 0.0, lk['p'][2] + 0.9), 0.012, M_ROPE, 5)
rod((lk['p'][0], 0.0, lk['p'][2] + 0.9), (pr[0] - 0.5, pr[1], pr[2] - RB - 0.1), 0.012, M_ROPE, 5)

# ---------------- willow (the dew drops hang from a branch tip) ----------------
wl = EV['willow']; wx, wz = wl['x'], wl['z']
M_WILLOW = mat('willow', (0.1, 0.3, 0.12), 0.6, **{'Emission Color': (0.2, 0.4, 0.25), 'Emission Strength': 0.05})
trunk_b = (wx - 2.6, 1.8, water_z(wx) - 0.3); trunk_t = (wx - 2.2, 1.6, wz + 2.6)
rod(trunk_b, trunk_t, 0.32, M_WOOD, 12)
arcs = []
for k in range(7):
    ax = wx - 2.2 + rng.uniform(-1.6, 1.4); ay = rng.uniform(0.8, 2.6); az = wz + rng.uniform(1.6, 2.8)
    rod(trunk_t, (ax, ay, az), 0.05, M_WOOD, 6)
    L_ = rng.uniform(1.5, 3.6)
    for j in range(3):
        x_ = ax + rng.uniform(-0.4, 0.4); y_ = ay + rng.uniform(-0.3, 0.3)
        rod((x_, y_, az), (x_ + rng.uniform(-0.15, 0.15), y_, az - L_ * rng.uniform(0.6, 1.0)), 0.012, M_WILLOW, 4)
# the two strands holding the drops (they spring up when the drops let go)
for n in ('silver', 'gold'):
    p0 = np.array(BP[n].at(0.0))
    st = link(bpy.data.objects.new('strand', None)); st.location = tuple(p0 + [0, 0, RT])
    s1 = rod((0, 0, 0), (0, 0, 1.6), 0.012, M_WILLOW, 4); s1.parent = st
    for j in range(4):
        lf = sphere((0.03 * (-1) ** j, 0, 0.25 + 0.35 * j), 0.04, M_WILLOW, (8, 4)); lf.scale = (0.025, 0.012, 0.09); lf.parent = st
    rel = wl['t_release']
    Zs = np.array([0.0 if T[f] < rel else 0.35 * (1 - math.exp(-(T[f] - rel) / 0.15)) * (1 + 0.25 * math.exp(-(T[f] - rel) / 0.5) * math.sin(14 * (T[f] - rel))) for f in range(NF)])
    key_series(st, 'location', 2, p0[2] + RT + Zs, 1e-4)
rod(trunk_t, (wx, 0.0, wz + RT + 1.6), 0.03, M_WOOD, 6)

# ---------------- zip line + lantern gondola ----------------
gd = EV['gondola']; gtop = np.array(gd['top']); gdir = np.array(gd['dir'])
HB = 0.75                                                          # rope above the pair centre
ra = gtop + [0, 0, HB] - gdir * 0.9; rb = gtop + [0, 0, HB] + gdir * (gd['L'] + 0.55)
rod(ra, rb, 0.015, M_ROPE, 6)
for pp in (ra, rb):
    rod((pp[0], pp[1], water_z(pp[0]) - 0.2), (pp[0], pp[1], pp[2] + 0.25), 0.07, M_BAMBOO, 10)
    sphere((pp[0], pp[1], pp[2] + 0.28), 0.08, M_BAMBOO, (8, 6))
box(tuple(rb - gdir * 0.25 - [0, 0, 0.05]), (0.12, 0.3, 0.3), M_WOOD, name='buffer')
gond = link(bpy.data.objects.new('gondola', None))
for n_, o_ in (('tray', box((0, 0, -RT - 0.04), (0.98, 0.98, 0.06), M_WOOD)),):
    o_.parent = gond
for sx in (-1, 1):
    box((sx * 0.48, 0, -RT + 0.05), (0.04, 0.98, 0.14), M_WOOD).parent = gond
    box((0, sx * 0.48, -RT + 0.05), (0.98, 0.04, 0.14), M_WOOD).parent = gond
for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
    o_ = rod((sx * 0.46, sy * 0.46, -RT), (0, 0, HB - 0.08), 0.008, M_ROPE, 4); o_.parent = gond
o_ = box((0, 0, HB - 0.04), (0.18, 0.08, 0.1), M_WOOD); o_.parent = gond
lan = obj(prim('cyl', M_PAPER, 16), 'lantern'); lan.parent = gond; lan.location = (0, 0, -RT - 0.3); lan.scale = (0.14, 0.14, 0.26)
gl = bpy.data.lights.new('glan', 'POINT'); gl.color = (1.0, 0.6, 0.3); gl.energy = 25; glo = link(bpy.data.objects.new('glan', gl)); glo.parent = gond; glo.location = (0, 0, -RT - 0.3)
GP = np.zeros((NF, 3))
for f in range(NF):
    t = T[f]; u = t - gd['t0']; us = gd['t_stop'] - gd['t0']
    s = 0.0 if u <= 0 else (gd['vt'] * (min(u, us) - gd['tau'] * (1 - math.exp(-min(u, us) / gd['tau']))))
    GP[f] = gtop + gdir * s
key_loc(gond, GP)
gond.rotation_mode = 'XYZ'
swing = np.array([0.0 if T[f] < gd['t_stop'] else -0.22 * math.exp(-(T[f] - gd['t_stop']) / 0.7) * math.sin(5.5 * (T[f] - gd['t_stop'])) for f in range(NF)])
key_series(gond, 'rotation_euler', 1, swing, 1e-4)

# ---------------- well-bucket lift (tsurube): pulley frame, the pair's bucket, the water counter-bucket ----------------
lf = EV['lift']; pin = np.array(lf['p_in']); H = lf['H']
xl, zl = pin[0], pin[2]; top_z = zl + H + 1.25
for sy in (-0.62, 0.62): rod((xl - 0.6, sy, water_z(xl) - 0.2), (xl - 0.6, sy, top_z), 0.06, M_WOOD, 8)
rod((xl - 0.6, -0.7, top_z), (xl - 0.6, 1.7, top_z), 0.06, M_WOOD, 8)
for sy in (1.4,): rod((xl - 0.6, sy, water_z(xl) - 0.2), (xl - 0.6, sy, top_z), 0.06, M_WOOD, 8)
pul = link(bpy.data.objects.new('pulley', None)); pul.location = (xl, 0.5, top_z - 0.05)
pw = obj(prim('cyl', M_WOOD, 24), 'pulw'); pw.parent = pul; pw.scale = (0.28, 0.28, 0.08); pw.rotation_euler = (math.pi / 2, 0, 0)
for k in range(6):
    o_ = box((0, 0, 0), (0.5, 0.03, 0.04), M_ROPE, rot=(0, k * math.pi / 6, 0)); o_.parent = pul
LZ = np.zeros(NF); TIPR = np.zeros(NF)
for f in range(NF):
    t = T[f]; u = min(1.0, max(0.0, (t - lf['t_up0']) / (lf['t_up1'] - lf['t_up0']))); u = u * u * (3 - 2 * u)
    LZ[f] = H * u
    v = (t - (lf['t_tip'] - 0.12)) / 0.12
    TIPR[f] = 0.0 if v < 0 else 0.75 * min(1, v) * (1 if t < lf['t_tip'] + 0.8 else max(0, 1 - (t - lf['t_tip'] - 0.8) / 0.5))
bk = link(bpy.data.objects.new('bucketA', None)); bk.location = tuple(pin + [0.5, 0, -RT - 0.06])
box((-0.5, 0, 0.0), (1.0, 0.98, 0.05), M_WOOD).parent = bk
for sx in (-1, 1):
    box((-0.5 + sx * 0.5, 0, 0.16), (0.05, 0.98, 0.32), M_WOOD).parent = bk
    box((-0.5, sx * 0.49, 0.16), (1.0, 0.05, 0.32), M_WOOD).parent = bk
key_series(bk, 'location', 2, pin[2] - RT - 0.06 + LZ, 1e-4); bk.rotation_mode = 'XYZ'; key_series(bk, 'rotation_euler', 1, TIPR, 1e-4)
rod((xl, 0.0, top_z - 0.3), (xl, 0.0, top_z - 0.31), 0.01, M_ROPE, 4)
ropeA = obj(prim('cyl', M_ROPE, 6), 'ropeA')
RA = np.array([(xl, 0.0, (top_z - 0.3 + pin[2] - RT + 0.3 + LZ[f]) / 2) for f in range(NF)])
key_loc(ropeA, RA)
key_series(ropeA, 'scale', 2, np.array([max(0.01, top_z - 0.3 - (pin[2] - RT + 0.3 + LZ[f])) for f in range(NF)]), 1e-4)
for c in (0, 1): ropeA.scale[c] = 0.012
cb = link(bpy.data.objects.new('bucketB', None))
obj(prim('cyl', M_WOOD, 16), 'bkB').parent = cb; cb.children[0].scale = (0.32, 0.32, 0.45)
wat = obj(prim('cyl', M_WATER, 16), 'bkBw'); wat.parent = cb; wat.scale = (0.29, 0.29, 0.02); wat.location = (0, 0, 0.1)
CBZ = np.array([top_z - 0.9 - LZ[f] for f in range(NF)])                # the water bucket sinks as the pair rises
key_loc(cb, np.stack([np.full(NF, xl), np.full(NF, 1.0), CBZ], 1))
rod((xl, 1.0, top_z - 0.3), (xl, 1.0, top_z - 0.6), 0.012, M_ROPE, 4)
key_series(pul, 'rotation_euler', 1, LZ / 0.28, 1e-4)
sp_ = rod((xl + 0.6, 1.45, top_z + 0.5), (xl + 0.05, 1.0, top_z - 0.45), 0.05, M_BAMBOO, 8)       # spout filling the counter-bucket
stream = rod((xl + 0.05, 1.0, top_z - 0.45), (xl + 0.05, 1.0, top_z - 0.95), 0.02, M_FALL, 6)

# ---------------- bamboo xylophone slides (one rail pair and one row of bars per ball) ----------------
for R_ in EV['ramps']:
    if 'bars' not in R_: continue
    al = R_['alpha']; p0 = np.array(R_['p0']); d = np.array([math.cos(al), 0, -math.sin(al)]); nrm = np.array(R_['n']); L = R_['L']
    for n, o in OFFS.items():
        o = np.array(o); base = p0 + o - RT * nrm
        a = base + d * (-0.25 - o[0]); b = base + d * (L + 0.1)
        for yy in (-0.19, 0.19):
            rod(a + [0, yy, 0.03], b + [0, yy, 0.03], 0.03, M_BAMBOO, 8)
        for s in np.arange(0.0, L + 0.1, 1.3):
            q = base + d * s
            for yy in (-0.19, 0.19): rod(q + [0, yy, 0], (q[0], q[1] + yy, water_z(q[0]) - 0.1), 0.025, M_BAMBOOD, 6)
        for bar in R_['bars']:
            pc = np.array(bar['p']) + o; top = pc - RT * nrm
            pitch = TW[bar['ev']]['hi'] if (n == 'silver' or TW[bar['ev']]['lo'] is None) else TW[bar['ev']]['lo']
            col = pcol(pitch, 0.2)
            ob = obj(prim('bcube', M_BAR), 'xbar')
            ob.matrix_world = Matrix.Translation(Vector(top - nrm * 0.03)) @ frame_from_n(nrm).to_4x4() @ Matrix.Diagonal((0.09, 0.42, 0.06, 1))
            ob.color = (*col, 0.0); flash(ob, bar['t'], col, base=0.05)

# ---------------- moon-stone: perch tower with a trapdoor, the A-flat slab ----------------
pr = np.array(EV['perch']['p']); sl = np.array(EV['slab']['p'])
slh = [h for h in BP['moon'].hits if h['kind'] == 'slab'][0]; sn = np.array(slh['n'])
PIL = (pr[0] + 0.1, 2.35)
rod((PIL[0], PIL[1], water_z(PIL[0]) - 0.3), (PIL[0], PIL[1], pr[2] + 0.4), 0.28, M_STONE, 14)
box((PIL[0], PIL[1], pr[2] + 0.5), (0.9, 0.9, 0.25), M_STONE, kind='bcube')
rod((PIL[0], PIL[1] - 0.2, pr[2] - RB - 0.12), (pr[0], pr[1] + 0.35, pr[2] - RB - 0.12), 0.06, M_STONE, 8)
ringp = obj(prim('cyl', M_STONED, 24), 'perchring'); ringp.location = (pr[0], pr[1], pr[2] - RB - 0.06); ringp.scale = (0.5, 0.5, 0.05)
for side in (-1, 1):                                               # trapdoor halves
    hg = link(bpy.data.objects.new('trap', None)); hg.location = (pr[0] + side * 0.36, pr[1], pr[2] - RB + 0.02)
    dd = box((-side * 0.18, 0, 0), (0.36, 0.62, 0.04), M_WOOD); dd.parent = hg
    an = np.array([0.0 if T[f] < EV['perch']['t_rel'] - 0.02 else side * min(1.4, 1.4 * ((T[f] - EV['perch']['t_rel'] + 0.02) / 0.12) ** 2) for f in range(NF)])
    hg.rotation_mode = 'XYZ'; key_series(hg, 'rotation_euler', 1, an, 1e-4)
slab_top = sl - RB * sn
so = obj(prim('bcube', M_BRONZE), 'slab')
so.matrix_world = Matrix.Translation(Vector(slab_top - sn * 0.05)) @ frame_from_n(sn).to_4x4() @ Matrix.Diagonal((0.8, 0.8, 0.1, 1))
so.color = (1.0, 0.75, 0.4, 0.0); flash(so, slh['t'], (1.0, 0.75, 0.4))
rod(slab_top - sn * 0.1, (PIL[0], PIL[1] - 0.2, slab_top[2] - 0.3), 0.06, M_STONE, 8)
# ---------------- stone see-saw ----------------
ss = EV['seesaw']; PIV = np.array(ss['piv']); LA, LB, TH0 = ss['la'], ss['lb'], ss['th0']
plank = link(bpy.data.objects.new('plank', None)); plank.location = tuple(PIV)
box(((LB - LA) / 2, 0, -0.06), (LA + LB + 0.3, 1.0, 0.12), M_WOOD).parent = plank
cupA = obj(prim('leaf', M_BRONZE, 28), 'cupA'); cupA.parent = plank; cupA.location = (-LA, 0, ss['cup_d']); cupA.scale = (0.62, 0.62, 1.2)
cupA.color = (0.9, 0.7, 0.4, 0.0)
plate = obj(prim('cyl', M_BRONZE, 32), 'plateB'); plate.parent = plank; plate.location = (LB, 0, ss['plate'] / 2); plate.scale = (0.45, 0.45, ss['plate'])
plate.color = (1.0, 0.75, 0.4, 0.0); flash(plate, ss['t_hit'], (1.0, 0.75, 0.4))
TH = np.array([TH0 - (2 * TH0) * min(1.0, max(0.0, (T[f] - ss['t_hit']) / ss['t_sw'])) for f in range(NF)])
plank.rotation_mode = 'XYZ'; key_series(plank, 'rotation_euler', 1, -TH, 1e-5)
bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=0.5, radius2=0.06, depth=0.6, location=(PIV[0], PIV[1], PIV[2] - 0.38))
bpy.context.object.data.materials.append(M_STONE)
rod((PIV[0], 0, PIV[2] - 0.6), (PIV[0], 0, water_z(PIV[0]) - 0.2), 0.25, M_STONE, 12)
# ---------------- stone channel with bronze bass bars ----------------
CHP = np.array(EV['channel']['path'])
cn = len(CHP)
# U-channel: floor + two walls following the moon-stone's path (floor under the ball's contact point)
dirs = np.gradient(CHP[:, :2], axis=0); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
side = np.stack([-dirs[:, 1], dirs[:, 0]], 1)
def chan_mesh(off0, off1, z0, z1, m, name):
    bm = bmesh.new(); vs = []
    for k in range(cn):
        p = CHP[k]; s_ = side[k]
        q0 = (p[0] + s_[0] * off0, p[1] + s_[1] * off0); q1 = (p[0] + s_[0] * off1, p[1] + s_[1] * off1)
        zf = p[2] - RB
        vs.append([bm.verts.new((q0[0], q0[1], zf + z1)), bm.verts.new((q1[0], q1[1], zf + z1)), bm.verts.new((q1[0], q1[1], zf + z0)), bm.verts.new((q0[0], q0[1], zf + z0))])
    for a, b in zip(vs, vs[1:]):
        for k in range(4): bm.faces.new((a[k], b[k], b[(k + 1) % 4], a[(k + 1) % 4]))
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new(name, me)); o.data.materials.append(m); return o
chan_mesh(-0.5, 0.5, -0.16, 0.0, M_STONE, 'chanfloor')
chan_mesh(-0.5, -0.38, -0.16, 0.32, M_STONE, 'chanwall'); chan_mesh(0.38, 0.5, -0.16, 0.32, M_STONE, 'chanwall')
for k in range(0, cn, 14):
    p = CHP[k]; rod((p[0], p[1], p[2] - RB - 0.16), (p[0], p[1], water_z(p[0]) - 0.2), 0.16, M_STONE, 10)
for cb_ in EV['channel']['bars']:
    p = np.array(cb_['p']); top = p - [0, 0, RB]
    ob = obj(prim('bcube', M_BRONZE), 'cbar'); ob.location = tuple(top - [0, 0, 0.03]); ob.scale = (0.12, 0.8, 0.06)
    ob.color = (1.0, 0.75, 0.4, 0.0); flash(ob, cb_['t'], (1.0, 0.75, 0.4))
# ---------------- shishi-odoshi ----------------
sh = EV['shishi']; SPV = np.array(sh['piv']); L1, L2, STH = sh['l1'], sh['l2'], sh['th0']
tube = link(bpy.data.objects.new('shishi', None)); tube.location = tuple(SPV)
tb_ = obj(prim('cyl', M_BAMBOO, 20), 'tube'); tb_.parent = tube; tb_.rotation_euler = (0, math.pi / 2, 0)
tb_.scale = (0.15, 0.15, L1 + L2); tb_.location = ((L2 - L1) / 2, 0, -0.18)
for xx in (-L1 + 0.05, 0.0, L2 - 0.6): rr_ = obj(prim('cyl', M_BAMBOOD, 20), 'node'); rr_.parent = tube; rr_.rotation_euler = (0, math.pi / 2, 0); rr_.scale = (0.16, 0.16, 0.03); rr_.location = (xx, 0, -0.18)
padb = box((-L1, 0, -sh['pad'] / 2), (1.0, 1.0, sh['pad']), M_WOOD, kind='bcube'); padb.parent = tube
opn = obj(prim('cyl', M_WATER, 20), 'tubewater'); opn.parent = tube; opn.rotation_euler = (0, math.pi / 2, 0); opn.scale = (0.12, 0.12, 0.02); opn.location = (L2 + 0.0, 0, -0.18)
SHT = np.zeros(NF)
for f in range(NF):
    t = T[f]; t1 = sh['t_tip']; t2 = t1 + sh['t_sw']; t3 = t2 + 0.45; t4 = t3 + 0.3
    if t < t1: SHT[f] = STH
    elif t < t2: SHT[f] = STH - 2 * STH * (t - t1) / sh['t_sw']
    elif t < t3: SHT[f] = -STH
    elif t < t4: u = (t - t3) / 0.3; SHT[f] = -STH + 2 * STH * u * u
    else: SHT[f] = STH
tube.rotation_mode = 'XYZ'; key_series(tube, 'rotation_euler', 1, -SHT, 1e-5)
for sy in (-0.5, 0.5):
    rod((SPV[0], sy, SPV[2] + 0.05), (SPV[0], sy, water_z(SPV[0]) - 0.2), 0.05, M_WOOD, 8)
rod((SPV[0], -0.55, SPV[2]), (SPV[0], 0.55, SPV[2]), 0.04, M_WOOD, 8)
padtop = SPV + np.array([-L1 * math.cos(STH), 0, -L1 * math.sin(STH)])
box((padtop[0], 0, padtop[2] - sh['pad'] - 0.35), (0.7, 0.8, 0.5), M_STONE, kind='bcube')              # the stone it knocks on
otip = SPV + np.array([L2 * math.cos(STH), 0, L2 * math.sin(STH)])
kak = rod((otip[0] + 1.6, 0.6, otip[2] + 1.4), (otip[0] + 0.1, 0.0, otip[2] + 0.55), 0.06, M_BAMBOO, 8)
strm = rod((otip[0] + 0.1, 0.0, otip[2] + 0.55), (otip[0] + 0.05, 0.0, otip[2] - 0.05), 0.025, M_FALL, 6)
for k in range(24):                                                # water spilling when it tips
    sp_ = sphere((0, 0, 0), 0.05, M_FX, (8, 6)); sp_.color = (0.7, 0.85, 1.0, 0)
    tdn = SPV + np.array([L2 * math.cos(STH), 0, -L2 * math.sin(STH)])
    v = np.array([rng.uniform(0.2, 1.4), rng.uniform(-0.6, 0.6), rng.uniform(-0.5, 1.2)])
    P = np.zeros((NF, 3)); A = np.zeros(NF)
    for f in range(NF):
        u = T[f] - (sh['t_tip'] + sh['t_sw'])
        P[f] = tdn + (v * u + np.array([0, 0, -0.5 * G * u * u]) if u > 0 else 0)
        A[f] = 0 if u < 0 or u > 0.7 else 0.5 * (1 - u / 0.7)
    key_loc(sp_, P); key_series(sp_, 'color', 3, A, 1e-3)

# ---------------- glass bowls under bamboo spouts (the drops) ----------------
DROPO = []
sp_pts = sorted([(d['spout'][0], d['spout'][2]) for d in EV['drops']])
PX_ = np.array([p[0] for p in sp_pts]); PZ_ = np.array([p[1] for p in sp_pts]) + RD + 0.32
YP = LN['drops'] + 0.0
px_all = np.arange(PX_.min() - 4, PX_.max() + 4, 0.6)
pz_all = np.interp(px_all, PX_, PZ_)
pz_all = np.maximum(pz_all, np.array([bank_z(x) + 1.6 for x in px_all]))
for xa, xb, za, zb in zip(px_all, px_all[1:], pz_all, pz_all[1:]):
    rod((xa, YP + 0.55, za), (xb, YP + 0.55, zb), 0.07, M_BAMBOO, 8)
for x in px_all[::5]:
    rod((x, YP + 0.55, float(np.interp(x, px_all, pz_all)) + 0.05), (x, YP + 0.55, bank_z(x) - 0.1), 0.045, M_BAMBOOD, 6)
for d in EV['drops']:
    top = np.array(d['top']); p = d['pitch']; col = pcol(p, 0.25)
    sz = 0.25 + 0.005 * (72 - p)
    bw = obj(prim('bowl', M_BOWL, 32), 'bowl'); bw.location = tuple(top); bw.scale = (sz, sz, sz)
    bw.color = (*col, 0.0); flash(bw, d['t'], col, base=0.15)
    rod((top[0], top[1], top[2] - sz * 0.7), (top[0], top[1], bank_z(top[0]) - 0.05), 0.05, M_STONE, 8)
    wtr = obj(prim('cyl', M_WATER, 20), 'bowlwater'); wtr.location = (top[0], top[1], top[2] - 0.04); wtr.scale = (sz * 0.88, sz * 0.88, 0.01)
    spz = d['spout'][2] + RD
    pz = float(np.interp(top[0], px_all, pz_all))
    rod((top[0], YP + 0.55, pz), (top[0], top[1] + 0.02, pz - 0.05), 0.035, M_BAMBOO, 6)
    rod((top[0], top[1], pz - 0.05), (top[0], top[1], spz + 0.02), 0.03, M_BAMBOO, 6)
    # the drop: grows at the spout tip, falls, splashes
    b = BP[d['name']]; dr_ = sphere((0, 0, 0), RD, M_DROP, (20, 10), d['name'])
    Pd = np.array([b.at(t) for t in T]); key_loc(dr_, Pd)
    t_rel = d['t'] - EV['tf_drop']
    S = np.array([0.001 if T[f] < t_rel - 0.6 else (min(1.0, (T[f] - t_rel + 0.6) / 0.55) if T[f] < d['t'] else max(0.001, 1 - (T[f] - d['t']) / 0.07)) for f in range(NF)])
    stretch = np.array([1.0 + (0.35 * min(1, (T[f] - t_rel) / 0.3) if t_rel < T[f] < d['t'] else 0.0) for f in range(NF)])
    for c in (0, 1): key_series(dr_, 'scale', c, S * RD / np.sqrt(stretch), 1e-4)
    key_series(dr_, 'scale', 2, S * RD * stretch, 1e-4)
    for k in range(4):
        sp_ = sphere((0, 0, 0), 0.03, M_FX, (6, 4)); sp_.color = (0.6, 0.9, 1.0, 0)
        ph = 2 * math.pi * k / 4 + rng.uniform(0, 1); v = np.array([0.7 * math.cos(ph), 0.7 * math.sin(ph), rng.uniform(1.0, 1.6)])
        P = np.zeros((NF, 3)); A = np.zeros(NF)
        for f in range(NF):
            u = T[f] - d['t']
            P[f] = top + (v * u + np.array([0, 0, -0.5 * G * u * u]) if u > 0 else 0)
            A[f] = 0 if u < 0 or u > 0.4 else 0.9 * (1 - u / 0.4)
        key_loc(sp_, P); key_series(sp_, 'color', 3, A, 1e-3)
    ripple(top[0], top[1], top[2] - 0.035, d['t'], size=sz * 0.85, dur=1.0, amp=0.7, col=tuple(0.5 + 0.5 * c for c in col))

# ---------------- the great gong and the two lotus-flower bells ----------------
gp = np.array(EV['gong']['p']); zwg = water_z(gp[0])
gg = obj(prim('bell', M_BRONZE, 48), 'gong'); gg.location = tuple(gp - [0, 0, RB - 0.12]); gg.scale = (0.75, 0.75, 0.75)
gg.color = (1.0, 0.8, 0.45, 0.0); key_series(gg, 'color', 3, np.maximum(0.0, pulse([FINAL], decay=30, length=140)), 1e-3)
rod((gp[0], gp[1], gp[2] - RB - 0.2), (gp[0], gp[1], zwg - 0.2), 0.3, M_STONE, 16)
isl = obj(prim('cyl', M_STONE, 32), 'island'); isl.location = (gp[0], gp[1], zwg + 0.02); isl.scale = (1.0, 1.0, 0.12)
fl = np.array(EV['flowers']['c'])
for n in ('silver', 'gold'):
    pc = fl + np.array(OFFS[n]); fh = [h for h in BP[n].hits if h['kind'] == 'flower'][0]; col = pcol(fh['pitch'], 0.3)
    fo = obj(prim('flower', M_PINK, 40), 'flower'); fo.location = tuple(pc - [0, 0, RT - 0.08]); fo.scale = (0.3, 0.3, 0.3)
    fo.color = (*col, 0.0); flash(fo, FINAL, col, base=0.35)
    rod(pc - [0, 0, RT + 0.05], (pc[0], pc[1], water_z(pc[0]) - 0.1), 0.025, M_STEM, 6)
    riser([fo], FINAL - 0.4, pc - [0, 0, RT], lead=2.2, rise=0.6, depth=0.6)
    for k in range(6):
        lf_ = obj(prim('leaf', M_LEAF, 20), 'pad'); lf_.location = (pc[0] + rng.uniform(-1, 1), pc[1] + rng.uniform(-1.4, 0.6), water_z(pc[0]) + 0.02)
        lf_.scale = (0.32, 0.32, 0.32); lf_.color = (0.4, 0.8, 0.5, 0.0)
# the full moon on the water (reflection), appears with the last chord and wavers in the ripples
refl = obj(prim('cyl', M_FX, 48), 'moonrefl'); refl.location = (gp[0], -1.6, zwg + 0.015); refl.color = (0.9, 0.93, 1.0, 0)
ra_ = np.array([0.0 if T[f] < FINAL + 0.2 else 0.3 * min(1, (T[f] - FINAL - 0.2) / 1.4) for f in range(NF)])
key_series(refl, 'color', 3, ra_, 1e-3)
for f in range(0, NF, 2):
    w = 1 + (0.12 * math.sin(7 * T[f]) * math.exp(-max(0, T[f] - FINAL) / 1.5) if T[f] > FINAL else 0)
    refl.scale = (1.3 * w, 0.75 / w, 0.004); refl.keyframe_insert('scale', frame=f)
for k in range(5): ripple(gp[0], gp[1], zwg, FINAL + 0.18 * k, size=5.5, dur=3.2, amp=0.6)
for k in range(60):                                                # splash
    sp_ = sphere((0, 0, 0), 0.05, M_FX, (6, 4)); sp_.color = (0.75, 0.9, 1.0, 0)
    ph = rng.uniform(0, 2 * math.pi); v = np.array([1.4 * math.cos(ph), 1.0 * math.sin(ph), rng.uniform(1.8, 3.6)])
    P = np.zeros((NF, 3)); A = np.zeros(NF)
    for f in range(NF):
        u = T[f] - FINAL
        P[f] = gp + (v * u + np.array([0, 0, -0.5 * G * u * u]) if u > 0 else 0)
        A[f] = 0 if u < 0 or u > 1.1 else 0.8 * (1 - u / 1.1)
    key_loc(sp_, P); key_series(sp_, 'color', 3, A, 1e-3)
build_ripples()

# ---------------- stone lanterns on the bank, lit in a chain by the moon-stone's see-saw strike; fireflies ----------------
T_WAKE = EV['seesaw']['t_hit']; X_WAKE = PIV[0]
LANT = []
for x in np.arange(BANKX[0] + 4, BANKX[-1] - 2, 7.5):
    zb_ = bank_z(x); y0 = 4.0
    box((x, y0, zb_ + 0.12), (0.6, 0.6, 0.24), M_STONE, kind='bcube')
    rod((x, y0, zb_ + 0.24), (x, y0, zb_ + 0.95), 0.1, M_STONE, 10)
    box((x, y0, zb_ + 1.1), (0.55, 0.55, 0.32), M_STONE, kind='bcube')
    win = box((x, y0 - 0.26, zb_ + 1.12), (0.3, 0.04, 0.2), M_LANTW, name='lwin')
    bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=0.62, radius2=0.08, depth=0.32, location=(x, y0, zb_ + 1.45))
    bpy.context.object.rotation_euler = (0, 0, math.pi / 4); bpy.context.object.data.materials.append(M_STONE)
    sphere((x, y0, zb_ + 1.65), 0.08, M_STONE, (8, 6))
    t_on = T_WAKE + 0.25 + abs(x - X_WAKE) / 14.0
    s_ = np.array([0.001 if T[f] < t_on else min(1.0, (T[f] - t_on) / 0.25) for f in range(NF)])
    for c in range(3): key_series(win, 'scale', c, np.array([0.3, 0.04, 0.2])[c] * s_, 1e-4)
    ll = bpy.data.lights.new('lant', 'POINT'); ll.color = (1.0, 0.6, 0.3); ll.shadow_soft_size = 0.3
    lo = link(bpy.data.objects.new('lant', ll)); lo.location = (x, y0 - 0.45, zb_ + 1.12)
    for f in range(0, NF, 3): ll.energy = float(60 * (0 if T[f] < t_on else min(1, (T[f] - t_on) / 0.3))); ll.keyframe_insert('energy', frame=f)
    LANT.append(x)
M_FF = fxmat('firefly', 9.0)
for k in range(90):
    x0 = rng.uniform(X_WAKE - 6, gp[0] + 6); y0 = rng.uniform(-2.5, 2.4); z0 = water_z(x0) + rng.uniform(0.3, 1.2)
    ff = sphere((0, 0, 0), 0.035, M_FF, (6, 4)); ff.color = (0.85, 1.0, 0.45, 0)
    t_on = T_WAKE + 0.1 + abs(x0 - X_WAKE) / 10.0 + rng.uniform(0, 0.6); ph = rng.uniform(0, 6.28); w = rng.uniform(0.6, 1.4)
    P = np.zeros((NF, 3)); A = np.zeros(NF)
    for f in range(NF):
        u = max(0.0, T[f] - t_on)
        P[f] = (x0 + 0.5 * math.sin(w * u + ph), y0 + 0.3 * math.sin(0.7 * w * u), z0 + 1.2 * (1 - math.exp(-u / 1.5)) + 0.25 * math.sin(1.3 * w * u + ph))
        A[f] = 0 if T[f] < t_on else min(1.0, u / 0.4) * (0.55 + 0.45 * math.sin(3.1 * w * u + ph))
    key_loc(ff, P, step=3); key_series(ff, 'color', 3, A, 2e-3)

# ---------------- reeds and irises at the front ----------------
for x in np.arange(WX[0] + 1, WX[-1] - 1, 0.7):
    zw = water_z(x)
    for j in range(2):
        xx = x + rng.uniform(-0.3, 0.3); yy = rng.uniform(-6.5, -4.5); h = rng.uniform(0.8, 1.8)
        rod((xx, yy, zw - 0.1), (xx + rng.uniform(-0.2, 0.2), yy, zw + h), 0.015, M_BAMBOOD, 4)

# ---------------- lights ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
SKY = (0.012, 0.024, 0.065)
bg = w.node_tree.nodes['Background']; bg.inputs['Color'].default_value = (*SKY, 1); bg.inputs['Strength'].default_value = 1.0
mn = bpy.data.lights.new('moonlight', 'SUN'); mn.color = (0.62, 0.72, 1.0); mn.angle = 0.03
mno = link(bpy.data.objects.new('moonlight', mn)); mno.rotation_euler = (math.radians(48), math.radians(-18), math.radians(-25))
for f in range(0, NF, 3):
    mn.energy = float(0.7 + 1.3 * FRAC_S[f] + (0.8 * min(1, (T[f] - FINAL) / 0.8) if T[f] > FINAL else 0)); mn.keyframe_insert('energy', frame=f)

# ---------------- camera: follow the pair, keep the bowls and the moon-stone in view ----------------
C2 = 0.5 * (POS['silver'] + POS['gold'])
FOC = np.zeros((NF, 3)); WID = np.zeros(NF)
for f in range(NF):
    t = T[f]
    FOC[f] = np.array([C2[f, 0] + 0.6, 0, C2[f, 2] + 1.2]); WID[f] = 5.9
    if t < 3.2: FOC[f] = np.array([wx + 0.4, 0, wz - 0.4]); WID[f] = 6.2
    if 32.6 < t < 37.6:                                             # the moon-stone falls, the see-saw throws the pair
        u = math.sin(math.pi * (t - 32.6) / 5.0); FOC[f] = (1 - u) * FOC[f] + u * np.array([PIV[0] + 0.8, 0, PIV[2] + 1.9]); WID[f] = 5.9 + 1.6 * u
    if t > FINAL - 2.0:
        u = min(1, (t - (FINAL - 2.0)) / 1.6); u = u * u * (3 - 2 * u)
        FOC[f] = (1 - u) * FOC[f] + u * np.array([gp[0] + 0.3, 0, gp[2] + 1.0]); WID[f] = (1 - u) * WID[f] + u * 7.6
FOC[:, 0] = smooth(FOC[:, 0], 0.45); FOC[:, 2] = smooth(FOC[:, 2], 0.7); WID = smooth(WID, 0.4)
LENS = 32.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 600
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24
    CP[f] = (FOC[f, 0], -D, FOC[f, 2] + 0.18 * D)
    TP[f] = (FOC[f, 0], 0.0, FOC[f, 2] + 0.2)
key_loc(cam, CP); key_loc(tgt, TP)
for k, (dx, dz, en) in enumerate(((-3.0, 6.0, 700), (3.5, 5.0, 500))):              # soft moonlight fill riding with the camera
    al = bpy.data.lights.new(f'fill{k}', 'AREA'); al.size = 7.0; al.color = (0.65, 0.75, 1.0); al.energy = en
    ao = link(bpy.data.objects.new(f'fill{k}', al))
    key_loc(ao, np.stack([FOC[:, 0] + dx, np.full(NF, -4.5), FOC[:, 2] + dz], 1)); ao.rotation_euler = (math.radians(-30), 0, 0)

# ---------------- the moon (rides with the camera, far away): a disc whose phase follows the notes struck ----------------
MD = 120.0; MR = 4.4; MX, MY = 11.5, 34.5
M_MOON = emis('moon', (1.0, 0.97, 0.88), 6.0)
md = obj(prim('cyl', M_MOON, 64), 'moondisc'); md.parent = cam; md.location = (MX, MY, -MD); md.scale = (MR, MR, 0.01); md.rotation_euler = (0, 0, 0)
M_OCC = emis('occluder', tuple(c * 1.25 for c in SKY), 1.0)
oc = obj(prim('cyl', M_OCC, 64), 'occ'); oc.parent = cam; oc.scale = (MR * 1.01, MR * 1.01, 0.01)
OCX = np.array([MX - MR * (0.25 + 1.95 * FRAC_S[f]) for f in range(NF)])
key_series(oc, 'location', 0, OCX, 1e-4); oc.location[1] = MY; oc.location[2] = -MD + 1.0
oc.keyframe_insert('location', index=1, frame=0); oc.keyframe_insert('location', index=2, frame=0)
halo = obj(prim('cyl', M_FX, 64), 'halo'); halo.parent = cam; halo.location = (MX, MY, -MD - 2.0); halo.scale = (MR * 1.9, MR * 1.9, 0.01)
halo.color = (0.55, 0.65, 1.0, 0)
key_series(halo, 'color', 3, 0.03 + 0.1 * FRAC_S + np.array([0.12 * min(1, max(0, T[f] - FINAL) / 0.8) for f in range(NF)]), 1e-3)
for k in range(70):                                                 # stars
    sx, sy = rng.uniform(-40, 40), rng.uniform(-10, 70)
    if math.hypot(sx - MX, sy - MY) < MR * 2.4: continue
    st = sphere((sx, sy, -MD - 5), rng.uniform(0.08, 0.2), M_TXT, (6, 4)); st.parent = cam
# a soft cloud bank drifting under the moon
M_CLOUD = fxmat('cloud', 1.2)
for k in range(5):
    cl = obj(prim('sph', M_CLOUD, 16, 8), 'cloud'); cl.parent = cam; cl.color = (0.35, 0.42, 0.6, 0.18)
    cx0 = rng.uniform(-25, 25); cy0 = MY - rng.uniform(6, 14)
    cl.scale = (rng.uniform(9, 16), rng.uniform(1.4, 2.4), 0.5)
    key_loc(cl, np.stack([cx0 + 0.25 * T, np.full(NF, cy0), np.full(NF, -MD + 3.0 + k)], 1), step=30)

# ---------------- HUD (rides with the camera) ----------------
HUD_D = 3.0
def hud_text(s, x, y, size, m, font=FONT, align='CENTER'):
    o = text(s, (x, y, -HUD_D), size, m, rot=(0, 0, 0), align=align, ext=0.0, font=font); o.parent = cam; return o
def vis(o, a):
    s0 = o.scale[0]
    for c in range(3): key_series(o, 'scale', c, np.maximum(0.001, a * s0), 1e-3)
M_HUDW = emis('hudw', (1.0, 0.97, 0.92), 3.0); M_HUDG = emis('hudg', (1.0, 0.85, 0.5), 4.0); M_HUDB = emis('hudb', (0.75, 0.88, 1.0), 4.0)
M_HUDBG = mat('hudbg', (0.0, 0.0, 0.0), 1.0, **{'Alpha': 0.5})
hook_a = np.array([1.0 if t < 3.9 else max(0.0, 1 - (t - 3.9) / 0.35) for t in T])
h1 = fit_w(hud_text('Can you hear', 0, 0.62, 0.105, M_HUDW, font=FONT_S), 0.95); vis(h1, hook_a)
h2 = fit_w(hud_text('the moonlight?', 0, 0.5, 0.105, M_HUDB, font=FONT_S), 0.95); vis(h2, hook_a)
h3 = fit_w(hud_text('月の光、聴こえますか？', 0, 0.4, 0.052, M_HUDW, font=FONT_S), 0.9); vis(h3, hook_a)
sub_a = np.array([0.0 if t < 3.9 else (min(1.0, (t - 3.9) / 0.3) if t < 9.5 else max(0.0, 1 - (t - 9.5) / 0.4)) for t in T])
s1 = fit_w(hud_text('Every note fills the moon', 0, 0.42, 0.06, M_HUDW), 0.92); vis(s1, sub_a)
s2 = fit_w(hud_text('音が鳴るたび 月が満ちていく', 0, 0.355, 0.04, M_HUDW), 0.9); vis(s2, sub_a)
# counter: MOON nn% (digit glyphs switched by scale)
PCT_ = np.minimum(100, np.floor(FRAC * 100 + 1e-9)).astype(int)
cnt_a = np.array([0.0 if t < 3.9 else min(1.0, (t - 3.9) / 0.3) for t in T]) * (T < FINAL + 0.6)
CXH, CYH = 0.29, 0.62                                               # under the moon (camera space -> HUD plane)
lab = hud_text('MOON', CXH, CYH + 0.075, 0.05, M_HUDB); vis(lab, cnt_a)
DIG_X = [CXH - 0.07, CXH - 0.005, CXH + 0.06]
for pos in range(3):
    dv = (PCT_ // (10 ** (2 - pos))) % 10
    shown = PCT_ >= 10 ** (2 - pos) if pos < 2 else np.ones(NF, bool)
    for dgt in range(10):
        o = hud_text(str(dgt), DIG_X[pos], CYH - 0.005, 0.085, M_HUDW)
        vis(o, ((dv == dgt) & shown).astype(float) * cnt_a)
pc_ = hud_text('%', CXH + 0.125, CYH - 0.01, 0.06, M_HUDW); vis(pc_, cnt_a)
fin_a = np.array([0.0 if t < FINAL + 0.6 else min(1.0, (t - FINAL - 0.6) / 0.3) for t in T])
f1 = fit_w(hud_text('FULL MOON  100%', 0, 0.56, 0.085, M_HUDG, font=FONT_S), 0.9); vis(f1, fin_a)
f2 = fit_w(hud_text('満月', 0, 0.475, 0.05, M_HUDW, font=FONT_S), 0.9); vis(f2, fin_a)
fq_a = np.array([0.0 if t < FINAL + 1.6 else min(1.0, (t - FINAL - 1.6) / 0.4) for t in T])
q1 = fit_w(hud_text('Did you hear it?', 0, -0.7, 0.075, M_HUDW, font=FONT_S), 0.9); vis(q1, fq_a)
q2 = fit_w(hud_text('月の光、聴こえた？', 0, -0.79, 0.04, M_HUDW, font=FONT_S), 0.9); vis(q2, fq_a)

# ---------------- signboards at the pond's edge (search keywords, English first) and mechanism labels ----------------
def signboard(x, lines, w, h):
    w = min(w, 4.1)
    zw = water_z(x); y = -1.7; zc = zw + 0.35 + h / 2
    box((x, y + 0.04, zc), (w, 0.06, h), M_SIGN, kind='bcube')
    for sx in (-1, 1): rod((x + sx * (w / 2 - 0.15), y + 0.08, zw - 0.3), (x + sx * (w / 2 - 0.15), y + 0.08, zc + h / 2 + 0.1), 0.05, M_WOOD, 8)
    tot = sum(sz * 1.32 for _, sz, _, _ in lines); zz = zc + tot / 2
    for s_, size, mt, fnt in lines:
        zz -= size * 1.32
        fit_w(text(s_, (x, y - 0.01, zz + size * 0.62), size, mt, font=fnt), w - 0.3)
def xcam(t): return float(np.interp(t, T, FOC[:, 0]))
S, J = FONT_S, FONT
for tb, w_, h_, lines in ((4.6, 5.0, 1.75, [('Clair de Lune', 0.56, M_TXT_G, S), ('Claude Debussy (1905)', 0.3, M_TXT, J), ('月の光　ドビュッシー', 0.24, M_TXT, J)]),
                          (10.2, 4.8, 1.45, [('Classical Piano', 0.44, M_TXT_B, S), ('Suite bergamasque', 0.27, M_TXT, J), ('クラシック ピアノ名曲', 0.22, M_TXT, J)]),
                          (15.8, 4.8, 1.45, [('Marble Machine ASMR', 0.4, M_TXT_G, J), ('Satisfying · Relaxing', 0.27, M_TXT, J), ('ビー玉マーブルマシン', 0.22, M_TXT, J)]),
                          (21.4, 4.8, 1.45, [('Sleep Music', 0.46, M_TXT_B, S), ('Calm night · Moonlight', 0.27, M_TXT, J), ('癒しの睡眠用BGM', 0.22, M_TXT, J)]),
                          (27.4, 4.8, 1.45, [('Moonlit Garden', 0.44, M_TXT_G, S), ('Japanese water garden', 0.27, M_TXT, J), ('月夜の日本庭園', 0.22, M_TXT, J)])):
    signboard(xcam(tb) + 0.4, lines, w_, h_)
def tag(en, jp, loc, size=0.22):
    o = fit_w(text(en, loc, size, M_TXT_G), 2.6)
    if jp: fit_w(text(jp, (loc[0], loc[1], loc[2] - size * 0.95), size * 0.62, M_TXT), 2.6)
tag('Zip line', 'ロープウェイ', (gtop[0] + 1.3, -0.9, gtop[2] + 1.35))
tag('Well bucket lift', 'つるべ', (xl - 0.2, -0.9, top_z + 0.45))
r1 = [r for r in EV['ramps'] if r.get('kind') == 'slide1'][0]
tag('Bamboo xylophone', '竹の木琴', (r1['p0'][0] + 2.6, -0.9, r1['p0'][2] + 0.95))
tag('Moon-stone', '月の石', (pr[0] + 0.1, 0.6, pr[2] + RB + 0.55), 0.26)
tag('See-saw', 'シーソー', (PIV[0] - 0.4, -0.9, PIV[2] - 0.85))
tag('Shishi-odoshi', 'ししおどし', (SPV[0] + 0.2, -0.9, SPV[2] - 0.95))
tag('Glass bowls', 'ガラスの鉢', (EV['drops'][2]['top'][0] + 0.2, 2.4, EV['drops'][2]['top'][2] + 0.65), 0.2)

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
        print('t', tt_, 'FOC', np.round(FOC[f], 2), 'WID', round(WID[f], 2), 'pct', PCT_[f], {n: np.round(POS[n][f], 2).tolist() for n in MAIN})
elif MODE == 'save':
    bpy.ops.wm.save_as_mainfile(filepath='/tmp/cdl.blend')
