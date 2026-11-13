"""O Tannenbaum - 'A fallen star grows a giant Christmas tree'.  Side view of a snowy forest at night; the camera rises with the tree.
Every melody note = a ball landing on a key / bell of that pitch: Stella (the fallen-star marble) plays half of every phrase on ice keys
(each key she strikes makes the newest tier of the fir grow), glass ornaments play the other half on the tier's bells and bounce into
its branches (and light up); in verses 2 and 3 pinecones drop onto log drums for the bass. Stella rides a machine up to the next row
after each of her phrases; at the end the star cannon fires her onto the tree top on the final chord.
layout.py plans all motion (constant gravity, bounces that never add speed, mechanisms for every speed-up); this file builds it.
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
FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.6) * FPS); T = np.arange(NF) / FPS
G = LAY['G']; EV = LAY['ev']; BALLS = LAY['balls']; RR = LAY['R']; YT = LAY['yt']; TIER_H = LAY['tier_h']; Y_PINE = LAY['y_pine']
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


def link(o): sc.collection.objects.link(o); return o


def mat(name, col, rough=0.5, **kw):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']; b.inputs['Base Color'].default_value = (*col, 1); b.inputs['Roughness'].default_value = rough
    for k, v in kw.items():
        if k in ('Emission Color',) and len(v) == 3: v = (*v, 1)
        b.inputs[k].default_value = v
    return m


def emis(name, col, s):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    e = nt.nodes.new('ShaderNodeEmission'); e.inputs['Color'].default_value = (*col, 1); e.inputs['Strength'].default_value = s
    o = nt.nodes.new('ShaderNodeOutputMaterial'); nt.links.new(e.outputs[0], o.inputs[0]); return m


def glowmat(name, gain=8.0, base=0.0, rough=0.3, metal=0.0, coat=0.5):
    """object colour = base colour; object alpha drives emission (hit flash / light on)"""
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


def objcol(name, rough=0.45, metal=0.0, coat=0.0):
    m = mat(name, (1, 1, 1), rough, **{'Metallic': metal, 'Coat Weight': coat}); nt = m.node_tree
    oi = nt.nodes.new('ShaderNodeObjectInfo'); nt.links.new(oi.outputs['Color'], nt.nodes['Principled BSDF'].inputs['Base Color'])
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
        bmesh.ops.create_cube(bm, size=1.0); bmesh.ops.bevel(bm, geom=bm.edges[:] + bm.verts[:], offset=0.12, segments=2, affect='EDGES')
    elif kind == 'cyl': bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=1.0, radius2=1.0, depth=1.0)
    elif kind == 'cone': bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=a[1], radius2=a[2], depth=a[3])
    elif kind == 'sph': bmesh.ops.create_uvsphere(bm, u_segments=a[0], v_segments=a[1], radius=1.0)
    elif kind == 'plane': bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    elif kind == 'torus':
        R_, r_ = a[0], a[1]; nu, nv = 24, 8; vs = []
        for i in range(nu):
            ph = 2 * math.pi * i / nu
            vs.append([bm.verts.new(((R_ + r_ * math.cos(2 * math.pi * j / nv)) * math.cos(ph), (R_ + r_ * math.cos(2 * math.pi * j / nv)) * math.sin(ph),
                                     r_ * math.sin(2 * math.pi * j / nv))) for j in range(nv)])
        for i in range(nu):
            for j in range(nv): bm.faces.new((vs[i][j], vs[(i + 1) % nu][j], vs[(i + 1) % nu][(j + 1) % nv], vs[i][(j + 1) % nv]))
    elif kind == 'bell':
        prof = [(0.0, 0.0), (0.42, 0.0), (0.5, -0.08), (0.55, -0.35), (0.62, -0.62), (0.8, -0.86), (1.0, -1.0), (0.96, -1.0), (0.78, -0.88),
                (0.58, -0.64), (0.5, -0.36), (0.44, -0.1), (0.0, -0.1)]
        segs = a[0]; rings = []
        for k in range(segs):
            ph = 2 * math.pi * k / segs
            rings.append([bm.verts.new((r * math.cos(ph), r * math.sin(ph), z)) for r, z in prof])
        for k in range(segs):
            r0, r1 = rings[k], rings[(k + 1) % segs]
            for j in range(len(prof) - 1): bm.faces.new((r0[j], r1[j], r1[j + 1], r0[j + 1]))
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    elif kind == 'star':                                       # 5-point star in the x-z plane, thickness a[0] along y
        pts = []
        for i in range(10):
            ang = math.pi / 2 + i * math.pi / 5; r = 1.0 if i % 2 == 0 else 0.45
            pts.append((r * math.cos(ang), r * math.sin(ang)))
        th = a[0]; c0 = bm.verts.new((0, -th, 0)); c1 = bm.verts.new((0, th, 0))
        rf = [bm.verts.new((x, -th * 0.3, z)) for x, z in pts]; rb = [bm.verts.new((x, th * 0.3, z)) for x, z in pts]
        for i in range(10):
            j = (i + 1) % 10
            bm.faces.new((c0, rf[i], rf[j])); bm.faces.new((c1, rb[j], rb[i])); bm.faces.new((rf[i], rb[i], rb[j], rf[j]))
    me = bpy.data.meshes.new(kind); bm.to_mesh(me); bm.free()
    if kind in ('sph', 'cone', 'bell', 'cyl', 'bcube', 'torus'):
        for p in me.polygons: p.use_smooth = True
    if m: me.materials.append(m)
    _MC[key] = me; return me


def obj(me, name='o'): return link(bpy.data.objects.new(name, me))


def box(c, size, m, rot=None, name='box', bevel=False):
    o = obj(prim('bcube' if bevel else 'cube', m), name); o.location = c; o.scale = size
    if rot is not None: o.rotation_euler = rot
    return o


def rod_matrix(p0, p1, r):
    p0, p1 = Vector(p0), Vector(p1); d = p1 - p0
    q = Vector((0, 0, 1)).rotation_difference(d.normalized() if d.length > 1e-9 else Vector((0, 0, 1)))
    return Matrix.Translation((p0 + p1) / 2) @ q.to_matrix().to_4x4() @ Matrix.Diagonal((r, r, max(1e-4, d.length), 1))


def rod(p0, p1, r, m, verts=10):
    o = obj(prim('cyl', m, verts), 'rod'); o.matrix_world = rod_matrix(p0, p1, r); return o


def sphere(c, r, m, seg=(16, 8), name='sph'):
    o = obj(prim('sph', m, *seg), name); o.location = c; o.scale = (r, r, r); return o


def _latin(s): return all(ord(c) < 0x2E80 for c in s)
EN_K = 0.75
JP_K = 2.1                                   # the CJK subset renders at about a third of the em size in Blender


def text(s, loc, size, m, rot=(math.pi / 2, 0, 0), align='CENTER', ext=0.01):
    cu = bpy.data.curves.new('t', 'FONT'); cu.body = s; cu.font = FONT_EN if _latin(s) else FONT
    cu.size = size * (EN_K if _latin(s) else JP_K); cu.extrude = ext
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


def flash(o, t, col, n=(0, 0, 1), dip=0.025, levels=((0, 1.0), (4, 0.5), (16, 0.1))):
    fh = int(round(t * FPS)); base = o.location.copy(); nv = Vector(n)
    o.color = (*col, 0); o.keyframe_insert('color', index=3, frame=max(0, fh - 1))
    for df, a in levels:
        o.color = (*col, a); o.keyframe_insert('color', index=3, frame=fh + df)
    for df, d in ((-1, 0), (1, -dip), (4, dip * 0.3), (8, 0)):
        o.location = base + nv * d; o.keyframe_insert('location', frame=max(0, fh + df))


def clock_node(nt):
    v = nt.nodes.new('ShaderNodeValue'); v.outputs[0].default_value = 0.0
    v.outputs[0].keyframe_insert('default_value', frame=0)
    v.outputs[0].default_value = NF / FPS; v.outputs[0].keyframe_insert('default_value', frame=NF)
    return v


def lightmat(name, col, on=4.0, flash_=14.0, off_col=(0.04, 0.03, 0.02), attr='t_on', tau=0.35, off_emit=0.0):
    """face attribute `attr` = switch-on time: dark bulb before, light after (with a short flash)"""
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    at = nt.nodes.new('ShaderNodeAttribute'); at.attribute_type = 'GEOMETRY'; at.attribute_name = attr
    ck = clock_node(nt)
    sub = nt.nodes.new('ShaderNodeMath'); sub.operation = 'SUBTRACT'; nt.links.new(ck.outputs[0], sub.inputs[0]); nt.links.new(at.outputs['Fac'], sub.inputs[1])
    gt = nt.nodes.new('ShaderNodeMath'); gt.operation = 'GREATER_THAN'; gt.inputs[1].default_value = 0.0; nt.links.new(sub.outputs[0], gt.inputs[0])
    mx = nt.nodes.new('ShaderNodeMath'); mx.operation = 'MAXIMUM'; mx.inputs[1].default_value = 0.0; nt.links.new(sub.outputs[0], mx.inputs[0])
    ex = nt.nodes.new('ShaderNodeMath'); ex.operation = 'MULTIPLY'; ex.inputs[1].default_value = -1.0 / tau; nt.links.new(mx.outputs[0], ex.inputs[0])
    ee = nt.nodes.new('ShaderNodeMath'); ee.operation = 'EXPONENT'; nt.links.new(ex.outputs[0], ee.inputs[0])
    fl = nt.nodes.new('ShaderNodeMath'); fl.operation = 'MULTIPLY_ADD'; fl.inputs[1].default_value = flash_; fl.inputs[2].default_value = on
    nt.links.new(ee.outputs[0], fl.inputs[0])
    st = nt.nodes.new('ShaderNodeMath'); st.operation = 'MULTIPLY'; nt.links.new(fl.outputs[0], st.inputs[0]); nt.links.new(gt.outputs[0], st.inputs[1])
    st2 = nt.nodes.new('ShaderNodeMath'); st2.operation = 'ADD'; st2.inputs[1].default_value = off_emit; nt.links.new(st.outputs[0], st2.inputs[0])
    em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Color'].default_value = (*col, 1); nt.links.new(st2.outputs[0], em.inputs['Strength'])
    gl = nt.nodes.new('ShaderNodeBsdfPrincipled'); gl.inputs['Base Color'].default_value = (*off_col, 1); gl.inputs['Roughness'].default_value = 0.15
    ad = nt.nodes.new('ShaderNodeAddShader'); nt.links.new(gl.outputs[0], ad.inputs[0]); nt.links.new(em.outputs[0], ad.inputs[1])
    nt.links.new(ad.outputs[0], out.inputs[0])
    return m


def quads_mesh(name, quads, tons, m, attr='t_on'):
    me = bpy.data.meshes.new(name); vs = []; fs = []
    for q in quads:
        k = len(vs); vs += [tuple(p) for p in q]; fs.append((k, k + 1, k + 2, k + 3))
    me.from_pydata(vs, [], fs); me.update()
    a = me.attributes.new(attr, 'FLOAT', 'FACE'); a.data.foreach_set('value', [float(t) for t in tons])
    me.materials.append(m); return link(bpy.data.objects.new(name, me))


def strip_mesh(X, Z, y0, y1, th, m, name):
    bm = bmesh.new(); vs = []
    for x, z in zip(X, Z):
        vs.append([bm.verts.new((x, y0, z)), bm.verts.new((x, y1, z)), bm.verts.new((x, y1, z - th)), bm.verts.new((x, y0, z - th))])
    for a, b in zip(vs, vs[1:]):
        for k in range(4): bm.faces.new((a[k], b[k], b[(k + 1) % 4], a[(k + 1) % 4]))
    bm.faces.new(vs[0][::-1]); bm.faces.new(vs[-1])
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new(name, me)); o.data.materials.append(m)
    return o



def smooth01(u): u = min(1.0, max(0.0, u)); return u * u * (3 - 2 * u)


ROWOBJ = {}                                          # objects of row k: they grow out of the tree just before the row is used
def snap(): return set(bpy.data.objects)
def assign(before, k, reg=None):
    for o in set(bpy.data.objects) - before:
        if o.parent is None: (ROWOBJ if reg is None else reg).setdefault(int(k), []).append(o)
MACHOBJ = {}                                         # machine k: appears with row k, folds back into the tree once it has been used
def assign_m(before, k): assign(before, k, MACHOBJ)


# ---------------- ball paths ----------------
def at_seg(seg, t):
    for g in seg:
        if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9:
            if g['k'] == 'rest': return np.array(g['p'])
            if g['k'] == 'fly':
                u = t - g['t0']; return np.array(g['p']) + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
            P = np.array(g['pts']); u = (t - g['t0']) / max(1e-9, g['t1'] - g['t0']) * (len(P) - 1)
            i = min(len(P) - 2, int(u)); f = u - i; return P[i] * (1 - f) + P[i + 1] * f
    return at_seg(seg, seg[0]['t0']) if t < seg[0]['t0'] else at_seg(seg, seg[-1]['t1'])
POS = {n: np.array([at_seg(b['seg'], t) for t in T]) for n, b in BALLS.items()}
PS = POS['stella']
DEV = EV['dev']; TIERS = EV['tiers']; Z_TOP = EV['z_top']
RS = RR['stella']
T_V2 = E['bass'][0][0] - 0.6; T_V3 = TN[[i for i in range(len(LEAD)) if LEAD[i][4] == 2][0]]
ZG = TIERS[0]['zb'] - 0.12                         # snow ground
M_PER_U = 30.0 / (Z_TOP + RS - ZG)                 # the finished tree is 30 m tall

# ---------------- materials ----------------
M_STELLA = glowmat('stella', gain=7.0, base=1.2, rough=0.12, metal=0.7, coat=1.0)
M_ORN = glowmat('orn', gain=3.5, base=0.0, rough=0.05, metal=0.6, coat=1.0)
M_KEY = glowmat('key', gain=7.0, base=0.25, rough=0.08, coat=1.0)
M_BELL = glowmat('bell', gain=6.0, base=0.15, rough=0.22, metal=0.8)
M_DRUM = glowmat('drum', gain=10.0, base=0.0, rough=0.4, coat=0.5)
M_OBJ = objcol('objcol', 0.45)
M_FX = fxmat('fx', 8.0)
M_GOLD = mat('gold', (1.0, 0.72, 0.28), 0.22, **{'Metallic': 0.95})
M_GOLDD = mat('goldd', (0.65, 0.42, 0.14), 0.35, **{'Metallic': 0.9})
M_BRASS = mat('brass', (0.85, 0.6, 0.25), 0.3, **{'Metallic': 0.9})
M_STEEL = mat('steel', (0.6, 0.62, 0.66), 0.28, **{'Metallic': 0.9})
M_DARK = mat('dark', (0.04, 0.035, 0.03), 0.5)
M_WOOD = mat('wood', (0.3, 0.15, 0.07), 0.5, **{'Coat Weight': 0.3})
M_WOODL = mat('woodl', (0.55, 0.33, 0.15), 0.5)
M_RED = mat('red', (0.7, 0.04, 0.05), 0.35, **{'Coat Weight': 0.6})
M_WHITE = mat('white', (0.92, 0.92, 0.95), 0.4)
M_SNOW = mat('snow', (0.85, 0.9, 1.0), 0.55, **{'Subsurface Weight': 0.3, 'Subsurface Radius': (0.3, 0.4, 0.6)})
M_ICE = mat('ice', (0.6, 0.85, 1.0), 0.05, **{'Transmission Weight': 0.6, 'Coat Weight': 1.0, 'Emission Color': (0.3, 0.6, 1.0), 'Emission Strength': 0.3})
M_TXT = emis('txt', (1.0, 0.97, 0.9), 2.2)
M_TXT_G = emis('txtg', (1.0, 0.78, 0.3), 3.0)
M_TXT_B = emis('txtb', (0.55, 0.85, 1.0), 2.6)
M_PLATE = mat('plate', (0.05, 0.08, 0.16), 0.45, **{'Coat Weight': 0.5})
M_PINE = mat('pine', (0.32, 0.17, 0.07), 0.6)
nt = M_PINE.node_tree; bsdf = nt.nodes['Principled BSDF']
vo_ = nt.nodes.new('ShaderNodeTexVoronoi'); vo_.inputs['Scale'].default_value = 9.0
bp_ = nt.nodes.new('ShaderNodeBump'); bp_.inputs['Strength'].default_value = 0.9; nt.links.new(vo_.outputs['Distance'], bp_.inputs['Height'])
nt.links.new(bp_.outputs['Normal'], bsdf.inputs['Normal'])
mr_ = nt.nodes.new('ShaderNodeMix'); mr_.data_type = 'RGBA'; mr_.inputs[6].default_value = (0.12, 0.06, 0.02, 1); mr_.inputs[7].default_value = (0.5, 0.3, 0.12, 1)
nt.links.new(vo_.outputs['Distance'], mr_.inputs[0]); nt.links.new(mr_.outputs[2], bsdf.inputs['Base Color'])


def pitch_col(p):
    cols = [(1.0, 0.3, 0.3), (1.0, 0.6, 0.2), (1.0, 0.9, 0.3), (0.45, 1.0, 0.45), (0.3, 0.95, 0.9), (0.4, 0.65, 1.0), (0.8, 0.5, 1.0)]
    deg = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 10: 6, 11: 6, 1: 0, 3: 1, 6: 3, 8: 4}[p % 12]
    return cols[deg]


# ---------------- balls ----------------
def make_ball(name, r, m, spin=True, seg=(32, 16)):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, segments=seg[0], ring_count=seg[1]); o = bpy.context.object; o.name = name
    for p in o.data.polygons: p.use_smooth = True
    o.data.materials.append(m); key_loc(o, POS[name])
    if spin:
        sp = np.zeros(NF)
        for f in range(1, NF): sp[f] = sp[f - 1] + (POS[name][f, 0] - POS[name][f - 1, 0]) / r
        o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, sp)
    return o
BST = make_ball('stella', RS, M_STELLA)
glow = np.full(NF, 0.25); glow[T < 1.4] = 1.0
glow += pulse([FINAL], decay=10, length=60, amp=1.2)
BST.color = (1.0, 0.78, 0.32, 0.25); key_series(BST, 'color', 3, glow, 1e-3)
# a little five-point star engraved on Stella (so she reads as "the star")
st_ = obj(prim('star', M_GOLD, 0.25), 'stellastar'); st_.parent = BST; st_.location = (0, -RS * 0.97, 0); st_.scale = (0.08, 0.08, 0.08)
ORN_COLS = [(0.9, 0.05, 0.08), (1.0, 0.7, 0.15), (0.1, 0.35, 0.95), (0.85, 0.85, 0.92), (0.1, 0.65, 0.3), (0.8, 0.15, 0.75), (1.0, 0.45, 0.1),
            (0.3, 0.85, 1.0)]
for n, b in BALLS.items():
    b4 = snap()
    if b['kind'] == 'orn':
        h = b['hits'][0]; j = int(n[3:]); col = ORN_COLS[(j * 3) % len(ORN_COLS)]
        o = obj(prim('sph', M_ORN, 24, 12), n); o.scale = (RR['orn'],) * 3; key_loc(o, POS[n])
        for p in o.data.polygons: p.use_smooth = True
        cap = rod((0, 0, 0.85), (0, 0, 1.2), 0.28, M_GOLD, 10); cap.parent = o
        lit = np.array([0.0 if T[f] < h['t_cup'] else 0.35 + 0.65 * math.exp(-(T[f] - h['t_cup']) / 0.35) for f in range(NF)])
        lit += 0.8 * pulse([FINAL], decay=8, length=40)
        o.color = (*col, 0); key_series(o, 'color', 3, lit, 1e-3); o.color = (*col, float(lit[0]))
    elif b['kind'] == 'pine':
        make_ball(n, RR['pine'], M_PINE, spin=True, seg=(20, 10))
    elif b['kind'] == 'snowball':
        make_ball(n, RR['snowball'], M_SNOW, spin=False, seg=(24, 12))
    if b['kind'] == 'orn': assign(b4, b['hits'][0]['tier'])
    elif b['kind'] == 'pine': assign(b4, E['bass'][b['hits'][0]['bass_j']][5])
    elif b['kind'] == 'snowball': assign_m(b4, int(n[4:]))

# ---------------- camera ----------------
FOC = np.zeros((NF, 3)); WID = np.full(NF, 6.0)
H_ALL = (Z_TOP + 0.9 - ZG + 0.3) / 0.72; W_ALL = H_ALL * 9 / 16; ZTREE_MID = (ZG - 0.3 + Z_TOP + 0.9) / 2 - 0.015 * H_ALL
for f in range(NF):
    t = T[f]; h = PS[f]
    FOC[f] = (0.22 * h[0], 0.0, h[2] + 0.7)
    if t < 1.6:                                       # opening: the sky with the falling star, then down to the snow
        u = min(1, max(0, (t - 0.3) / 1.3)); u = u * u * (3 - 2 * u)
        f0 = np.array([-1.2, 0, 3.4]); FOC[f] = f0 * (1 - u) + FOC[f] * u
    tw = 41.7; te = 44.5                              # mid-video reveal: how tall it is already (start of verse 3)
    if tw < t < te + 1.0:
        u = min(1, (t - tw) / 1.2) * min(1, max(0, (te + 1.0 - t) / 1.0)); u = u * u * (3 - 2 * u)
        WID[f] = 6.0 + 2.2 * u; FOC[f, 2] -= 1.4 * u
    tf_ = EV['cannon']['t_fire'] - 0.9
    if t > tf_:                                       # the finale: the whole giant tree
        u = min(1, (t - tf_) / 2.6); u = u * u * (3 - 2 * u)
        FOC[f] = (1 - u) * FOC[f] + u * np.array([0.0, 0.0, ZTREE_MID]); WID[f] = (1 - u) * WID[f] + u * W_ALL
FOC[:, 0] = smooth(FOC[:, 0], 0.6); FOC[:, 2] = smooth(FOC[:, 2], 0.45); WID = smooth(WID, 0.4)
LENS = 32.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 400
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TPc = np.zeros((NF, 3)); DD = np.zeros(NF)
for f in range(NF):
    D_ = WID[f] / (9 / 16) * LENS / 24; DD[f] = D_
    CP[f] = (FOC[f, 0], 0.8 - D_, FOC[f, 2] + 0.05 * D_)
    TPc[f] = (FOC[f, 0], 0.8, FOC[f, 2])
key_loc(cam, CP); key_loc(tgt, TPc)
def fz_cam(t): return float(np.interp(t, T, FOC[:, 2]))

# ---------------- the fir: 9 tiers that grow note by note ----------------
M_FIR = mat('fir', (0.02, 0.16, 0.05), 0.75)
nt = M_FIR.node_tree; bsdf = nt.nodes['Principled BSDF']
nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 22.0; nz.inputs['Detail'].default_value = 6.0
bmp = nt.nodes.new('ShaderNodeBump'); bmp.inputs['Strength'].default_value = 0.7; nt.links.new(nz.outputs['Fac'], bmp.inputs['Height'])
nt.links.new(bmp.outputs['Normal'], bsdf.inputs['Normal'])
mg = nt.nodes.new('ShaderNodeMix'); mg.data_type = 'RGBA'; mg.inputs[6].default_value = (0.008, 0.07, 0.025, 1); mg.inputs[7].default_value = (0.04, 0.24, 0.07, 1)
nt.links.new(nz.outputs['Fac'], mg.inputs[0])
geo = nt.nodes.new('ShaderNodeNewGeometry'); sxyz = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(geo.outputs['Normal'], sxyz.inputs[0])
mrng = nt.nodes.new('ShaderNodeMapRange'); mrng.inputs['From Min'].default_value = 0.5; mrng.inputs['From Max'].default_value = 0.8
nt.links.new(sxyz.outputs['Z'], mrng.inputs['Value'])
mul_ = nt.nodes.new('ShaderNodeMath'); mul_.operation = 'MULTIPLY'; nt.links.new(mrng.outputs[0], mul_.inputs[0]); nt.links.new(nz.outputs['Fac'], mul_.inputs[1])
ms = nt.nodes.new('ShaderNodeMix'); ms.data_type = 'RGBA'; ms.inputs[7].default_value = (0.85, 0.9, 1.0, 1)
nt.links.new(mul_.outputs[0], ms.inputs[0]); nt.links.new(mg.outputs[2], ms.inputs[6]); nt.links.new(ms.outputs[2], bsdf.inputs['Base Color'])


def tier_mesh(R, H, seed, name):
    rr_ = np.random.default_rng(seed); bm = bmesh.new(); NS, NZ = 48, 26; rings = []
    ph = rr_.uniform(0, 6.28, 3)
    for iz in range(NZ + 1):
        z = H * iz / NZ; rl = R * (1 - 0.65 * z / H)
        saw = (z / 0.62) % 1.0
        ring = []
        for a in range(NS):
            th = 2 * math.pi * a / NS
            jag = 1 + 0.06 * math.sin(9 * th + ph[0]) + 0.04 * math.sin(14 * th + ph[1] + z * 3)
            r = max(0.02, rl * jag - 0.03 - 0.09 * saw * (1 - z / H))
            ring.append(bm.verts.new((r * math.cos(th), r * math.sin(th), z)))
        rings.append(ring)
    for iz in range(NZ):
        for a in range(NS):
            b_ = (a + 1) % NS; bm.faces.new((rings[iz][a], rings[iz][b_], rings[iz + 1][b_], rings[iz + 1][a]))
    # underside of the skirt (droops in to the trunk)
    inner = [bm.verts.new((0.18 * R * math.cos(2 * math.pi * a / NS), 0.18 * R * math.sin(2 * math.pi * a / NS), 0.45)) for a in range(NS)]
    for a in range(NS):
        b_ = (a + 1) % NS; bm.faces.new((rings[0][b_], rings[0][a], inner[a], inner[b_]))
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); me.materials.append(M_FIR)
    for p in me.polygons: p.use_smooth = True
    return me


M_TLIGHT = lightmat('tlight', (1.0, 0.8, 0.42), on=6.0, flash_=16.0, off_col=(0.02, 0.06, 0.02))
TIER_OBJ = []; GROW = np.zeros((9, NF))
for tr in TIERS:
    k = tr['k']; R = tr['R']
    o = link(bpy.data.objects.new(f'tier{k}', tier_mesh(R, TIER_H, k + 3, f'tier{k}'))); o.location = (0, YT, tr['zb'])
    s0 = 0.14 if k == 0 else 0.0
    g = np.zeros(NF)
    for tg in tr['t_grow']:
        u = np.clip((T - tg) / 0.22, 0, 1); g += (1 - (1 - u) ** 3) / len(tr['t_grow'])
    s = s0 + (1 - s0) * g
    GROW[k] = s
    for c_ in range(3): key_series(o, 'scale', c_, np.maximum(1e-3, s), 1e-4)
    TIER_OBJ.append(o)
    # lights on the front half (local coordinates; they grow with the tier)
    rr_ = np.random.default_rng(k + 11); q_ = []; tq = []
    for q in range(40):
        z = rr_.uniform(0.12, 0.85) * TIER_H; r = R * (1 - 0.65 * z / TIER_H) * 0.97; a = rr_.uniform(-1.25, 1.25)
        px, py = r * math.sin(a), -r * math.cos(a); d = 0.04
        q_.append([(px - d, py - 0.02, z), (px, py - 0.02, z - d), (px + d, py - 0.02, z), (px, py - 0.02, z + d)]); tq.append(tr['t_lit'] + 0.03 * q)
    lo = quads_mesh(f'tl{k}', q_, tq, M_TLIGHT); lo.parent = o
    # gold hooks where its ornaments land (local)
    for o_ in EV['orn']:
        if o_['tier'] != k: continue
        c = np.array(o_['cup']) - np.array([0, YT, tr['zb']])
        cu = obj(prim('torus', M_GOLD, 0.06, 0.012), 'cup'); cu.location = (c[0], c[1], c[2] - RR['orn'] + 0.015); cu.parent = o
# trunk + stand
rod((0, YT, ZG - 0.05), (0, YT, TIERS[0]['zb'] + 0.6), 0.28, mat('trunk', (0.18, 0.09, 0.04), 0.8), 16)
# the leader on top of the last tier and the star holder (grow with tier 8)
t8 = TIERS[8]; zl0 = TIER_H; zl1 = Z_TOP - RS - t8['zb'] - 0.02
ld = obj(prim('cone', M_FIR, 16, 0.12, 0.04, 1.0), 'leader'); ld.location = (0, 0, (zl0 - 0.2 + zl1) / 2); ld.scale = (1, 1, zl1 - zl0 + 0.2); ld.parent = TIER_OBJ[8]
hold = obj(prim('torus', M_GOLD, 0.12, 0.03), 'holder'); hold.location = (0, 0, zl1 + 0.02); hold.parent = TIER_OBJ[8]
# the big star that appears around Stella on the final chord
top = np.array(EV['cannon']['top'])
M_TSTAR = lightmat('tstar', (1.0, 0.8, 0.3), on=9.0, flash_=30.0, off_col=(0.1, 0.075, 0.03))
BIGSTAR = obj(prim('star', M_TSTAR, 0.06), 'bigstar'); BIGSTAR.location = (top[0], top[1] - 0.05, top[2]); BIGSTAR.scale = (0.001,) * 3
me_ = BIGSTAR.data.copy(); BIGSTAR.data = me_
a_ = me_.attributes.new('t_on', 'FLOAT', 'FACE'); a_.data.foreach_set('value', [FINAL] * len(me_.polygons))
fs_ = int(round(FINAL * FPS))
BIGSTAR.keyframe_insert('scale', frame=fs_); BIGSTAR.scale = (0.75,) * 3; BIGSTAR.keyframe_insert('scale', frame=fs_ + 6)
BIGSTAR.scale = (0.6,) * 3; BIGSTAR.keyframe_insert('scale', frame=fs_ + 14)

# ---------------- Stella's ice keys (one per note) on snowy log beams ----------------
M_LOG = mat('log', (0.22, 0.12, 0.06), 0.7)
KEYHITS = [h for h in BALLS['stella']['hits'] if h['kind'] in ('key', 'rollkey')]
for h in KEYHITS:
    b4 = snap()
    n_ = np.array(h['n']); c = np.array(h['p']) - RS * n_; col = pitch_col(h['pitch'])
    o = obj(prim('bcube', M_KEY), 'key')
    if h['kind'] == 'rollkey':
        d = 1 if h['roll'] >= 0 else -1; L_ = abs(h['roll']) + 0.07 - 0.05
        o.matrix_world = Matrix.Translation(Vector(c + np.array([d * (L_ / 2 - 0.07), 0, -0.03]))) @ Matrix.Diagonal((L_, 0.42, 0.06, 1))
    else:
        o.matrix_world = Matrix.Translation(Vector(c - n_ * 0.03)) @ frame_from_n(n_).to_4x4() @ Matrix.Diagonal((0.12, 0.42, 0.06, 1))
    o.color = (*col, 0)
    lv = ((0, 1.0), (4, 0.5), (16, 0.1)) if h['kind'] == 'key' else ((0, 1.0), (int(h['t_roll'] * FPS), 0.7), (int(h['t_roll'] * FPS) + 12, 0.1))
    flash(o, h['t'], col, n=n_, levels=lv)
    for dy in (-0.15, 0.15):
        rod((c[0], dy, c[2] - 0.05), (c[0], dy, c[2] - 0.34), 0.012, M_BRASS, 6)
    assign(b4, LEAD[h['i']][8])
for k in range(9):
    b4 = snap()
    hs = [h for h in KEYHITS if LEAD[h['i']][8] == k]
    X = np.array([h['p'][0] for h in hs] + [h['p'][0] + h['roll'] for h in hs if h['kind'] == 'rollkey'])
    Z = np.array([h['p'][2] - RS - 0.36 for h in hs] + [h['p'][2] - RS - 0.36 for h in hs if h['kind'] == 'rollkey'])
    o_ = np.argsort(X); X, Z = X[o_], Z[o_]
    XX = np.linspace(X[0] - 0.25, X[-1] + 0.25, 30); ZZ = np.interp(XX, X, Z)
    ZM = np.array([Z[np.abs(X - x) < 0.3].min() if (np.abs(X - x) < 0.3).any() else z for x, z in zip(XX, ZZ)])
    strip_mesh(XX, ZM, -0.24, 0.24, 0.12, M_LOG, 'beam')
    strip_mesh(XX, ZM + 0.03, -0.25, 0.25, 0.04, M_SNOW, 'beamsnow')
    for x_ in (XX[0] + 0.15, XX[-1] - 0.15):                          # brackets back into the tree
        z_ = float(np.interp(x_, XX, ZM)) - 0.06
        rod((x_, 0.2, z_), (x_ * 0.7, YT - 1.2, z_ - 0.15), 0.04, M_LOG, 8)
    assign(b4, k)

# ---------------- ornament bells (height = pitch), garland rail, clips; folded away into the tree once the tier is lit ----------------
for tr in TIERS:
    bl = tr['bells']; yo = tr['y_orn']; zr = tr['rail']
    b4 = snap()
    rig = link(bpy.data.objects.new(f'bellrig{tr["k"]}', None)); rig.location = (0, yo, zr)
    objs = []
    for b_ in bl:
        o_ = BALLS[b_['ball']]; h = o_['hits'][0]
        n_ = np.array(h['n']); c = np.array(h['p']) - RR['orn'] * n_
        be = obj(prim('bell', M_BELL, 20), 'bell')
        be.matrix_world = Matrix.Translation(Vector(c)) @ frame_from_n(n_).to_4x4() @ Matrix.Diagonal((0.1, 0.1, 0.17, 1))
        col = (1.0, 0.82, 0.4); be.color = (*col, 0); flash(be, h['t'], col, n=n_, dip=0.02)
        objs.append(be)
        objs.append(rod((c[0], yo, c[2] - 0.15), (c[0], yo, min(b['z'] for b in bl) - 0.55), 0.012, M_BRASS, 6))
    xs = [b_['x'] for b_ in bl]; zb0 = min(b_['z'] for b_ in bl) - 0.55
    objs.append(rod((min(xs) - 0.15, yo, zb0), (max(xs) + 0.15, yo, zb0), 0.03, M_GOLDD, 8))
    objs.append(rod((min(xs) - 0.2, yo, zr), (max(xs) + 0.2, yo, zr), 0.035, M_GOLD, 10))
    gx = np.linspace(min(xs) - 0.2, max(xs) + 0.2, 16)
    for xa_, xb_ in zip(gx, gx[1:]):
        objs.append(rod((xa_, yo + 0.06, zr + 0.06 + 0.03 * math.sin(xa_ * 9)), (xb_, yo + 0.06, zr + 0.06 + 0.03 * math.sin(xb_ * 9)), 0.05,
                        mat('garland', (0.02, 0.25, 0.08), 0.6), 6))
    for x_ in (min(xs) - 0.2, max(xs) + 0.2):
        objs.append(rod((x_, yo, zr), (x_, YT - 0.6, zr + 0.2), 0.025, M_GOLD, 8))
    for b_ in bl:
        o_ = BALLS[b_['ball']]; p0 = np.array(o_['seg'][0]['p']); tr_rel = o_['seg'][0]['t1']
        piv = link(bpy.data.objects.new('clip', None)); piv.location = (p0[0], yo, zr - 0.02)
        for sgn in (1, -1):
            jaw = box((sgn * 0.04, 0, -0.05), (0.025, 0.06, 0.09), M_GOLD); jaw.parent = piv
        ang = np.array([0.0 if T[f] < tr_rel else min(1.0, (T[f] - tr_rel) / 0.08) * 0.6 for f in range(NF)])
        key_series(piv, 'scale', 0, 1.0 + 0.8 * ang, 1e-4); objs.append(piv)
    for o in objs:
        o.parent = rig; o.matrix_parent_inverse = Matrix.Translation(-Vector(rig.location))
    tf0 = tr['t_lit'] + 0.25
    sc_ = np.array([1.0 if T[f] < tf0 else max(1e-3, 1 - min(1, (T[f] - tf0) / 0.5) ** 2) for f in range(NF)])
    for c_ in range(3): key_series(rig, 'scale', c_, sc_, 1e-4)
    assign(b4, tr['k'])

# ---------------- pinecone branches, log drums, troughs (verses 2, 3) ----------------
M_TRIG = glowmat('trig', gain=10.0, base=0.3, rough=0.3)
for pr_ in EV['pine']:
    b4 = snap()
    k = pr_['k']; z_ = pr_['rail']
    rod((pr_['x0'], Y_PINE, z_), (pr_['x1'], Y_PINE, z_), 0.045, M_LOG, 10)
    for x_ in (pr_['x0'], pr_['x1']): rod((x_, Y_PINE, z_), (x_ * 0.8, YT - 1.3, z_ + 0.3), 0.04, M_LOG, 8)
    cups_ = [np.array(h['cup']) for b in BALLS.values() if b['kind'] == 'pine' for h in b['hits'] if 'cup' in h and E['bass'][h['bass_j']][5] == k]
    if cups_:                                                         # the trough the pinecones land in
        xa_, xb_ = min(c[0] for c in cups_) - 0.22, max(c[0] for c in cups_) + 0.22; cy, cz = cups_[0][1], cups_[0][2] - RR['pine']
        box(((xa_ + xb_) / 2, cy, cz - 0.03), (xb_ - xa_, 0.32, 0.05), M_WOOD)
        box(((xa_ + xb_) / 2, cy - 0.16, cz + 0.05), (xb_ - xa_, 0.03, 0.16), M_WOOD)
        for x_ in (xa_, xb_): box((x_, cy, cz + 0.05), (0.03, 0.32, 0.16), M_WOOD)
        for x_ in (xa_ + 0.1, xb_ - 0.1): rod((x_, cy + 0.1, cz - 0.05), (x_, cy + 0.6, cz - 0.6), 0.02, M_BRASS, 6)
    assign(b4, k)
for name, b in BALLS.items():
    if b['kind'] != 'pine': continue
    b4 = snap()
    h = b['hits'][0]; n_ = np.array(h['n']); c = np.array(h['p']) - RR['pine'] * n_
    rd = 0.2 if h['kind'] != 'bassfinal' else 0.25
    o = obj(prim('cyl', M_DRUM if h['kind'] != 'trigger' else M_TRIG, 24), 'drumhead')
    o.matrix_world = Matrix.Translation(Vector(c - n_ * 0.015)) @ frame_from_n(n_).to_4x4() @ Matrix.Diagonal((rd, rd, 0.03, 1))
    col = (0.95, 0.82, 0.6) if h['kind'] != 'trigger' else (0.9, 0.1, 0.06)
    o.color = (*col, 0.0 if h['kind'] != 'trigger' else 0.3); flash(o, h['t'], col, n=n_, dip=0.03)
    shell = obj(prim('cyl', M_RED if h['kind'] == 'bassfinal' else M_WOOD, 24), 'drumshell')
    shell.matrix_world = Matrix.Translation(Vector(c - n_ * 0.17)) @ frame_from_n(n_).to_4x4() @ Matrix.Diagonal((rd * 0.97, rd * 0.97, 0.26, 1))
    for zz in (0.06, 0.28): rod(c - n_ * zz + [0, 0, 0], c - n_ * (zz + 0.025), rd * 1.01, M_GOLDD, 24)
    rod((c[0], c[1] + 0.1, c[2] - 0.25), (c[0], c[1] + 0.6, c[2] - 0.6), 0.02, M_BRASS, 6)
    piv = link(bpy.data.objects.new('pclip', None)); p0 = np.array(b['seg'][0]['p']); piv.location = (p0[0], Y_PINE, p0[2] + RR['pine'] + 0.03)
    for sgn in (1, -1):
        jaw = box((sgn * 0.045, 0, -0.03), (0.025, 0.07, 0.08), M_GOLDD); jaw.parent = piv
    tr_rel = b['seg'][0]['t1']
    key_series(piv, 'scale', 0, np.array([1.0 if T[f] < tr_rel else 1.0 + 0.8 * min(1.0, (T[f] - tr_rel) / 0.08) for f in range(NF)]), 1e-4)
    assign(b4, E['bass'][h['bass_j']][5])

# ---------------- machines ----------------
def follow(objs, P, anchor):
    """parent objects to an empty that follows positions P[f] (anchor = the position objects were built at)"""
    e = link(bpy.data.objects.new('follow', None)); e.location = anchor
    for o in objs: o.parent = e; o.matrix_parent_inverse = Matrix.Translation(-Vector(anchor))
    key_loc(e, P); return e


def bucket(c, w, d, h, m, open_side=0):
    """open bucket around ball centre c (floor at c.z - RS - 0.03); open_side = +1/-1: no wall on that x side (a gate there)"""
    zf_ = c[2] - RS - 0.03; objs = [box((c[0], c[1], zf_ - 0.02), (w, d, 0.04), m)]
    for sx in (-1, 1):
        if sx != open_side: objs.append(box((c[0] + sx * w / 2, c[1], zf_ + h / 2), (0.03, d, h), m))
    for sy in (-1, 1): objs.append(box((c[0], c[1] + sy * d / 2, zf_ + h / 2), (w, 0.03, h), m))
    return objs


def hold_path(D, t_end):
    """the car follows Stella from the catch until t_end, before: at the catch, after: where she left it"""
    pc = np.array(D['catch']); i_end = min(NF - 1, int(round(t_end * FPS)))
    return np.array([pc if T[f] < D['t_catch'] else (PS[f] if T[f] <= t_end else PS[i_end]) for f in range(NF)])


def label(s, p, size=0.13, w=1.3):
    o = fit_w(text(s, tuple(p), size, M_TXT_G), w)
    bpy.context.view_layer.update()
    box((p[0], p[1] + 0.03, p[2]), (o.dimensions.x + 0.16, 0.02, size * 1.25), M_PLATE)
    return o


M_LIFT = mat('lift', (0.75, 0.08, 0.06), 0.3, **{'Coat Weight': 0.8})
M_LIFT2 = mat('lift2', (0.1, 0.35, 0.65), 0.3, **{'Coat Weight': 0.8})
M_GATE = mat('gatec', (0.95, 0.78, 0.25), 0.3, **{'Metallic': 0.5})
CH = {c['k']: c for c in EV['chutes']}
AL = math.radians(16.0); L_CH = 0.36


def chute(c, ice=False):
    top_ = np.array(c['top']); u = np.array(c['u']); d = 1 if u[0] > 0 else -1
    nrm = np.array([d * math.sin(AL), 0, math.cos(AL)])
    s0 = top_ - nrm * (RS + 0.03) - u * 0.22; s1 = top_ + u * L_CH - nrm * (RS + 0.03)
    pl_ = obj(prim('cube', M_ICE if ice else M_WOODL), 'chute')
    pl_.matrix_world = Matrix.Translation(Vector((s0 + s1) / 2)) @ Matrix.Rotation(-d * AL, 4, 'Y') @ Matrix.Diagonal((np.linalg.norm(s1 - s0), 0.38, 0.04, 1))
    for dy in (-0.2, 0.2): rod(s0 + [0, dy, 0.1], s1 + [0, dy, 0.1], 0.016, M_GOLD, 6)
    rod(s1 - [0, 0, 0.03], (s1[0], 0.15, s1[2] - 0.6), 0.025, M_BRASS, 8)
    # the gate: a flap just in front of the ball, hinged above, swings up when the chute opens
    hg = link(bpy.data.objects.new('gate', None)); gp = top_ + u * (RS + 0.03) + nrm * 0.24; hg.location = tuple(gp)
    fl_ = box((0, 0, -0.14), (0.035, 0.34, 0.26), M_GATE); fl_.parent = hg
    ang = np.array([0.0 if T[f] < c['t_gate'] else -min(1.0, (T[f] - c['t_gate']) / 0.15) * 1.4 * d for f in range(NF)])
    hg.rotation_mode = 'XYZ'; key_series(hg, 'rotation_euler', 1, ang)
    box(tuple(gp + [0, 0, 0.03]), (0.06, 0.42, 0.05), M_DARK)
    return top_, u, d


SIGNS = []
for D in DEV:
    b4 = snap()
    k = D['k']; pc = np.array(D['catch']); d = D['d']
    if D['kind'] in ('lift', 'chair', 'balloon'):
        top_, u, dd = chute(CH[k])
        P = hold_path(D, D['t_gate'] + 0.02)
        tg = D['t_gate']
    if D['kind'] == 'lift':
        # back to the bottom after she has left
        i0 = int(round((tg + 0.6) * FPS))
        for f in range(i0, NF):
            uu = min(1, (T[f] - tg - 0.6) / 1.6); uu = uu * uu * (3 - 2 * uu); P[f] = top_ * (1 - uu) + pc * uu
        objs = bucket(pc, 0.42, 0.42, 0.22, M_LIFT if k == 0 else M_LIFT2, open_side=-d)
        follow(objs, P, pc)
        xr = pc[0] + d * 0.3
        for dy in (-0.26, 0.26):
            rod((xr, dy, pc[2] - 0.3), (top_[0] + d * 0.3, dy, top_[2] + 0.15), 0.025, M_STEEL, 8)
            rod((pc[0], dy, pc[2] - RS - 0.1), (top_[0], dy, top_[2] - RS - 0.1), 0.012, M_STEEL, 6)
        box((top_[0] + d * 0.45, 0.45, top_[2] + 0.1), (0.24, 0.3, 0.24), M_DARK, name='motor')
        sp = obj(prim('cyl', M_RED, 20), 'pulley'); sp.location = (top_[0] + d * 0.45, 0.28, top_[2] + 0.1); sp.rotation_euler = (math.pi / 2, 0, 0); sp.scale = (0.12, 0.12, 0.05)
        sp.rotation_mode = 'XYZ'; key_series(sp, 'rotation_euler', 1, np.cumsum(np.r_[0, np.linalg.norm(np.diff(P, axis=0), axis=1)]) / 0.14)
        SIGNS.append(('ELEVATOR', (pc[0] + d * 0.45, -0.45, pc[2] + 0.75)))
    elif D['kind'] == 'chair':
        cab0 = pc + np.array([0, 0, 0.62]) - (top_ - pc) / np.linalg.norm(top_ - pc) * 0.35
        cab1 = top_ + np.array([0, 0, 0.62]) + (top_ - pc) / np.linalg.norm(top_ - pc) * 0.35
        rod(cab0, cab1, 0.014, M_DARK, 6)
        for cb in (cab0, cab1):
            wh = obj(prim('torus', M_STEEL, 0.16, 0.025), 'sheave'); wh.location = tuple(cb + [0, 0, -0.0]); wh.rotation_euler = (math.pi / 2, 0, 0)
            rod(cb + [0, 0.06, 0], (cb[0] + d * 0.15, 0.5, cb[2] - 0.9), 0.04, M_STEEL, 8)
        seat = [box((pc[0], 0, pc[2] - RS - 0.06), (0.4, 0.38, 0.05), M_LIFT2), box((pc[0] + d * 0.21, 0, pc[2] - RS + 0.08), (0.04, 0.38, 0.3), M_LIFT2)]
        seat.append(rod((pc[0], 0.0, pc[2] - RS - 0.04), (pc[0] + d * 0.02, 0.0, pc[2] + 0.62), 0.014, M_STEEL, 6))
        seat.append(rod((pc[0] - 0.2, 0.17, pc[2] - RS - 0.04), (pc[0] + d * 0.02, 0.0, pc[2] + 0.62), 0.01, M_STEEL, 5))
        seat.append(rod((pc[0] - 0.2, -0.17, pc[2] - RS - 0.04), (pc[0] + d * 0.02, 0.0, pc[2] + 0.62), 0.01, M_STEEL, 5))
        follow(seat, P, pc)
        SIGNS.append(('CHAIRLIFT', (pc[0] + d * 0.5, -0.45, pc[2] + 0.8)))
    elif D['kind'] == 'balloon':
        i0 = int(round((tg + 0.5) * FPS))
        for f in range(i0, NF):
            uu = (T[f] - tg - 0.5); P[f] = top_ + np.array([d * 0.25 * uu, 0.9 * uu, 0.55 * uu + 0.05 * uu * uu])
        objs = bucket(pc, 0.44, 0.44, 0.24, mat('basket', (0.45, 0.28, 0.1), 0.7), open_side=-d)
        env = sphere(pc + [0, 0, 1.25], 1.0, mat('envelope', (0.85, 0.08, 0.08), 0.5, **{'Sheen Weight': 0.6}), (24, 16)); env.scale = (0.6, 0.6, 0.72); objs.append(env)
        for a_ in range(6):
            band = sphere(pc + [0, 0, 1.25], 1.0, M_WHITE, (24, 16)); band.scale = (0.61, 0.08, 0.73); band.rotation_euler = (0, 0, a_ * math.pi / 6); objs.append(band)
        for sx in (-1, 1):
            for sy in (-1, 1): objs.append(rod(pc + [sx * 0.2, sy * 0.2, -RS + 0.2], pc + [sx * 0.3, sy * 0.3, 0.68], 0.008, M_DARK, 4))
        fl = obj(prim('cone', emis('flame', (1.0, 0.55, 0.15), 14.0), 12, 0.07, 0.0, 0.3), 'flame'); fl.location = tuple(pc + [0, 0, 0.62]); objs.append(fl)
        e = follow(objs, P, pc)
        fsc = 0.15 + pulse([s_[1] for s_ in LAY['sfx'] if s_[0] == 'burner' and abs(s_[1] - D['m0']) < 1.5], decay=9, length=30, amp=1.0)
        for c_ in range(3): key_series(fl, 'scale', c_, np.maximum(0.05, fsc), 1e-3)
        SIGNS.append(('HOT-AIR BALLOON', (pc[0] + d * 0.2, -0.75, pc[2] - 0.55)))
    elif D['kind'] == 'catapult':
        piv = np.array(D['pivot']); ARM = D['arm']; TH = math.radians(D['th']); tl = D['t_launch']
        sw = link(bpy.data.objects.new('seesaw', None)); sw.location = tuple(piv)
        plank = box((0, 0, 0), (0.3, 2 * ARM + 0.4, 0.06), M_WOOD, name='plank'); plank.parent = sw
        cup_ = bucket(np.array([0, -ARM, RS + 0.05]), 0.32, 0.32, 0.12, M_LIFT)
        for o in cup_: o.parent = sw
        pad = box((0, ARM, 0.06), (0.36, 0.36, 0.05), M_RED); pad.parent = sw
        bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=0.25, radius2=0.03, depth=0.35, location=(piv[0], piv[1], piv[2] - 0.2), rotation=(0, 0, math.pi / 4))
        bpy.context.object.data.materials.append(M_STEEL)
        rod((piv[0], piv[1], piv[2] - 0.37), (piv[0], piv[1] + 0.4, piv[2] - 1.4), 0.04, M_LOG, 8)
        ang = np.array([TH if T[f] < tl else TH - 2 * TH * smooth01((T[f] - tl) / 0.12) for f in range(NF)])
        sw.rotation_mode = 'XYZ'; key_series(sw, 'rotation_euler', 0, ang)
        # the snowy branch the snowball sits on, the latch and the cord from Stella's cup
        ps_ = np.array(D['snow_start']); pe_ = np.array(D['snow_end']); alb = math.radians(D['alb'])
        nb = np.array([0, -math.sin(alb), math.cos(alb)])
        b0 = ps_ - nb * (RR['snowball'] + 0.03) + np.array([0, 0.35, 0.06]); b1 = pe_ - nb * (RR['snowball'] + 0.03)
        br = obj(prim('cube', M_LOG), 'branch'); br.matrix_world = Matrix.Translation(Vector((b0 + b1) / 2)) @ Matrix.Rotation(alb, 4, 'X') @ Matrix.Diagonal((0.36, np.linalg.norm(b1 - b0), 0.08, 1))
        bs_ = obj(prim('cube', M_SNOW), 'branchsnow'); bs_.matrix_world = Matrix.Translation(Vector((b0 + b1) / 2 + nb * 0.05)) @ Matrix.Rotation(alb, 4, 'X') @ Matrix.Diagonal((0.4, np.linalg.norm(b1 - b0), 0.04, 1))
        rod(b0, (b0[0] * 0.6, YT - 0.9, b0[2] + 0.2), 0.06, M_LOG, 8)
        lt = link(bpy.data.objects.new('latch', None)); lt.location = tuple(ps_ + np.array([0, -RR['snowball'] - 0.04, 0.2]))
        lf_ = box((0, 0, -0.14), (0.3, 0.03, 0.24), M_GATE); lf_.parent = lt
        lang = np.array([0.0 if T[f] < D['t_snow'] - 0.04 else min(1.0, (T[f] - D['t_snow'] + 0.04) / 0.12) * 1.4 for f in range(NF)])
        lt.rotation_mode = 'XYZ'; key_series(lt, 'rotation_euler', 0, lang)
        rod((pc[0] + d * 0.17, 0.05, pc[2] - 0.05), tuple(ps_ + np.array([d * 0.17, -RR['snowball'] - 0.05, 0.3])), 0.008, M_WHITE, 4)
        SIGNS.append(('SNOWBALL CATAPULT', (pc[0] - d * 0.3, -0.45, pc[2] - 0.75)))
    elif D['kind'] == 'final':
        pcan = np.array(D['cannon']); v0 = np.array(D['v0']); uC = v0 / np.linalg.norm(v0); tf_ = D['t_fire']
        P = hold_path(D, tf_)
        M_CAN = mat('cannon', (0.08, 0.25, 0.14), 0.3, **{'Metallic': 0.6, 'Coat Weight': 0.8})
        objs = [rod(pc - uC * 0.6, pc + uC * 0.12, 0.2, M_CAN, 24)]
        for u_ in (-0.5, -0.1, 0.1): objs.append(rod(pc + uC * (u_ - 0.02), pc + uC * (u_ + 0.02), 0.22, M_GOLD, 24))
        follow(objs, P, pc)
        for dx in (-0.25, 0.25):
            rod((pc[0] + d * 0.35 + dx * 0, -0.3 + dx, pc[2] - 0.4), (pcan[0] + d * 0.35, pcan[1] - 0.3 + dx, pcan[2] - 0.4), 0.03, M_STEEL, 8)
        pist = rod((0, 0, -0.05), (0, 0, 0.05), 0.15, M_STEEL, 16)
        PP = np.zeros((NF, 3))
        for f in range(NF):
            uu = (T[f] - (tf_ - 0.04)) / 0.04
            s_ = -0.45 if uu < 0 else (-0.45 + 0.45 * min(1, uu) if T[f] < tf_ + 0.5 else -0.45 + 0.45 * max(0, 1 - (T[f] - tf_ - 0.5) / 0.6))
            PP[f] = P[f] + uC * (s_ - RS - 0.02)
        key_loc(pist, PP); pist.rotation_mode = 'QUATERNION'; pist.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(Vector(uC))
        trg = np.array(EV['cannon']['trigger'])
        rod((trg[0] + 0.1, trg[1] + 0.2, trg[2] - RR['pine'] - 0.12), (pcan[0], pcan[1] - 0.3, pcan[2] - 0.5), 0.022, mat('trigrod', (0.8, 0.1, 0.06), 0.3), 6)
        smoke = sphere(tuple(pcan + uC * 0.3), 0.3, M_FX, (12, 8)); smoke.color = (1, 0.95, 0.8, 0)
        fs = int(round(tf_ * FPS)); sa = np.zeros(NF); ss = np.full(NF, 0.01)
        for df in range(22):
            if fs + df < NF: sa[fs + df] = 0.9 * (1 - df / 22) ** 1.5; ss[fs + df] = 0.3 + 1.1 * (1 - (1 - df / 22) ** 2)
        key_series(smoke, 'color', 3, sa, 1e-3)
        for c_ in range(3): key_series(smoke, 'scale', c_, ss * 0.3)
        SIGNS.append(('STAR CANNON', (pcan[0] + d * 0.2, pcan[1] - 0.5, pcan[2] - 0.8)))
    s_, p_ = SIGNS[-1]; label(s_, p_)
    assign_m(b4, k)

# ---------------- intro: the snow cup and the ice chute ----------------
it = EV['intro']; cupp = np.array(it['cup'])
chute(CH[-1], ice=True)
rod((cupp[0] - 0.1, 0.0, cupp[2] - RS - 0.06), (cupp[0] - 0.1, 0.0, ZG), 0.17, M_LOG, 12)
cap_ = sphere((cupp[0] - 0.1, 0.0, cupp[2] - RS - 0.05), 1.0, M_SNOW, (20, 10)); cap_.scale = (0.3, 0.3, 0.09)
nest = obj(prim('torus', M_GOLD, 0.2, 0.04), 'nest'); nest.location = (cupp[0], 0, cupp[2] - RS + 0.04)
label('FALLEN STAR', (cupp[0] - 0.1, -0.35, cupp[2] - 0.75))
# shooting-star trail during the fall
NTR = 14
for q in range(NTR):
    o = obj(prim('sph', M_FX, 12, 6), 'trail'); o.color = (1.0, 0.85, 0.45, 0)
    Lq = np.zeros((NF, 3)); A_ = np.zeros(NF); S_ = np.full(NF, 1e-3)
    for f in range(NF):
        tq = T[f] - 0.035 * (q + 1)
        if T[f] <= it['t_land'] + 0.05 and tq >= 0:
            Lq[f] = at_seg(BALLS['stella']['seg'], min(tq, it['t_land'])); A_[f] = 0.8 * (1 - q / NTR); S_[f] = RS * (1 - 0.6 * q / NTR)
        else: Lq[f] = Lq[f - 1] if f else np.array(it['sky'])
    key_loc(o, Lq); key_series(o, 'color', 3, A_, 1e-3)
    for c_ in range(3): key_series(o, 'scale', c_, S_)
# snow puff on landing
for q in range(10):
    o = obj(prim('sph', M_FX, 10, 6), 'puff'); o.color = (0.9, 0.95, 1.0, 0)
    a0 = 2 * math.pi * q / 10; Lq = np.zeros((NF, 3)); A_ = np.zeros(NF); S_ = np.full(NF, 1e-3)
    for f in range(NF):
        s_ = T[f] - it['t_land']
        Lq[f] = cupp + np.array([0.5 * math.cos(a0) * min(s_, 0.6), -0.2, 0.25 * abs(math.sin(a0)) * min(max(s_, 0), 0.6) + 0.05])
        if 0 <= s_ < 0.8: A_[f] = 0.7 * (1 - s_ / 0.8); S_[f] = 0.08 + 0.18 * s_
    key_loc(o, Lq); key_series(o, 'color', 3, A_, 1e-3)
    for c_ in range(3): key_series(o, 'scale', c_, S_)

# ---------------- hit rings (FX pool) ----------------
HITS = [(h['t'], np.array(h['p']) - BALLS[n]['r'] * np.array(h['n']), n, h['kind']) for n, b in BALLS.items() for h in b['hits']
        if h['kind'] in ('key', 'rollkey', 'bell', 'bass', 'trigger', 'bassfinal', 'top', 'land')]
HITS.sort(key=lambda x: x[0])
NRING = 12
ring_loc = np.zeros((NRING, NF, 3)); ring_scl = np.full((NRING, NF), 1e-3); ring_a = np.zeros((NRING, NF))
for k, (t, c, n, kind) in enumerate(HITS):
    r = k % NRING; f0 = int(round(t * FPS))
    big = 2.4 if kind in ('top', 'land', 'bassfinal') else (1.0 if n == 'stella' else 0.7)
    for df in range(12):
        f = f0 + df
        if f >= NF: break
        u = df / 11; ring_loc[r, f] = c + np.array([0, -0.25, 0.02]); ring_scl[r, f] = big * (0.25 + 1.0 * (1 - (1 - u) ** 2))
        ring_a[r, f] = (1 - u) ** 1.6
for r in range(NRING):
    o = obj(prim('torus', M_FX, 0.2, 0.012), 'ring'); o.rotation_euler = (math.pi / 2, 0, 0); o.color = (1.0, 0.88, 0.6, 0)
    for f in range(1, NF):
        if ring_a[r, f] == 0: ring_loc[r, f] = ring_loc[r, f - 1]
    key_loc(o, ring_loc[r])
    for k in range(3): key_series(o, 'scale', k, ring_scl[r])
    key_series(o, 'color', 3, ring_a[r], 1e-3)
# growth sparkles: at the top of the growing tier on each of Stella's notes; cup sparkles when an ornament lands
SPK = []
for tr in TIERS:
    for tg in tr['t_grow']:
        f_ = min(NF - 1, int(round((tg + 0.2) * FPS))); s_ = GROW[tr['k']][f_]
        SPK.append((tg + 0.05, np.array([0.0, YT - tr['R'] * 0.2 - 0.3, tr['zb'] + TIER_H * s_ * 0.92]), 0.22))
for o_ in EV['orn']: SPK.append((o_['t_cup'], np.array(o_['cup']) + [0, -0.15, 0], 0.13))
SPK.sort(key=lambda s: s[0])
NSP = 10
for q in range(NSP):
    o = obj(prim('star', M_FX, 0.02), 'spark'); o.color = (1.0, 0.92, 0.6, 0)
    Ls = np.zeros((NF, 3)); A_ = np.zeros(NF); S_ = np.full(NF, 1e-3); RZ = np.zeros(NF)
    for j, (t_, p_, sz) in enumerate(SPK):
        if j % NSP != q: continue
        f0 = int(round(t_ * FPS))
        for df in range(12):
            if f0 + df < NF: Ls[f0 + df] = p_; A_[f0 + df] = (1 - df / 12) ** 1.3; S_[f0 + df] = sz * (0.6 + 0.8 * df / 12); RZ[f0 + df] = df * 0.15
    for f in range(1, NF):
        if A_[f] == 0: Ls[f] = Ls[f - 1]
    key_loc(o, Ls); key_series(o, 'color', 3, A_, 1e-3); key_series(o, 'rotation_euler', 1, RZ)
    for c_ in range(3): key_series(o, 'scale', c_, S_)
# finale: a burst of golden sparks from the star
for q in range(28):
    o = obj(prim('star', M_FX, 0.02), 'burst'); o.color = (1.0, 0.85, 0.4, 0)
    a0 = 2 * math.pi * q / 28 + 0.1; sp = 2.2 + 1.2 * (q % 3)
    Ls = np.zeros((NF, 3)); A_ = np.zeros(NF); S_ = np.full(NF, 1e-3)
    for f in range(NF):
        s_ = T[f] - FINAL
        dr = sp * 0.45 * (1 - math.exp(-max(0, s_) / 0.45))
        Ls[f] = top + np.array([dr * math.cos(a0), -0.3, dr * math.sin(a0) - 0.4 * max(0, s_ - 0.5) ** 2])
        if 0 <= s_ < 2.6: A_[f] = (1 - s_ / 2.6) ** 1.2; S_[f] = 0.12
    key_loc(o, Ls); key_series(o, 'color', 3, A_, 1e-3)
    for c_ in range(3): key_series(o, 'scale', c_, S_)

# ---------------- rows grow out of the tree: row k appears while Stella rides up to it ----------------
for k, objs in ROWOBJ.items():
    if k == 0: continue
    D_ = DEV[k - 1]; ta = D_['t_catch'] + 0.15
    zr_ = float(np.median([h['p'][2] for h in KEYHITS if LEAD[h['i']][8] == k]))
    rig = link(bpy.data.objects.new(f'rowrig{k}', None)); rig.location = (0, YT - 1.0, zr_)
    for o in objs:
        o.parent = rig; o.matrix_parent_inverse = Matrix.Translation(-Vector(rig.location))
    sv = np.array([1e-3 if T[f] < ta else min(1.0, 1e-3 + (1 - (1 - min(1, (T[f] - ta) / 0.7)) ** 3) * (1 + 0.08 * math.sin(math.pi * min(1, (T[f] - ta) / 0.7)))) for f in range(NF)])
    for c_ in range(3): key_series(rig, 'scale', c_, sv, 1e-4)


for k, objs in MACHOBJ.items():
    D_ = DEV[k]
    ta = DEV[k - 1]['t_catch'] + 0.15 if k > 0 else -1.0
    tf_ = {'lift': D_.get('t_gate', 0) + 2.3, 'chair': D_.get('t_gate', 0) + 0.9, 'balloon': D_.get('t_gate', 0) + 3.0,
           'catapult': D_.get('t_launch', 0) + 1.4, 'final': 1e9}[D_['kind']]
    pc_ = np.array(D_['catch'])
    rig = link(bpy.data.objects.new(f'machrig{k}', None)); rig.location = (pc_[0] * 0.5, YT - 1.0, pc_[2] + 1.0)
    for o in objs:
        o.parent = rig; o.matrix_parent_inverse = Matrix.Translation(-Vector(rig.location))
    sv = np.ones(NF)
    for f in range(NF):
        if T[f] < ta: sv[f] = 1e-3
        elif T[f] < ta + 0.7: u = (T[f] - ta) / 0.7; sv[f] = 1e-3 + (1 - (1 - u) ** 3) * (1 + 0.08 * math.sin(math.pi * u))
        if T[f] > tf_: sv[f] = max(1e-3, 1 - min(1, (T[f] - tf_) / 0.6) ** 2)
    for c_ in range(3): key_series(rig, 'scale', c_, sv, 1e-4)


# ---------------- the snowy forest at night ----------------
M_GROUND = mat('ground', (0.8, 0.86, 0.97), 0.6)
nt = M_GROUND.node_tree; bsdf = nt.nodes['Principled BSDF']
nzg = nt.nodes.new('ShaderNodeTexNoise'); nzg.inputs['Scale'].default_value = 1.5
bpg = nt.nodes.new('ShaderNodeBump'); bpg.inputs['Strength'].default_value = 0.35; nt.links.new(nzg.outputs['Fac'], bpg.inputs['Height'])
nt.links.new(bpg.outputs['Normal'], bsdf.inputs['Normal'])
gnd = box((0, 60, ZG - 0.5), (400, 260, 1.0), M_GROUND, name='ground')
for q in range(14):                                                           # snow drifts near the stage
    x_ = rng.uniform(-9, 9); y_ = rng.uniform(1.5, 9)
    if abs(x_) < 2.6 and y_ < 5: continue
    m_ = sphere((x_, y_, ZG - 0.1), 1.0, M_GROUND, (20, 10)); m_.scale = (rng.uniform(0.8, 2.0), rng.uniform(0.6, 1.4), rng.uniform(0.25, 0.5))
M_FOREST = M_FIR
TREEPOS = []
for q in range(150):
    x_ = rng.uniform(-45, 45); y_ = rng.uniform(6.5, 70)
    if abs(x_) < 6 and y_ < 11: continue
    TREEPOS.append((x_, y_, rng.uniform(4.0, 8.0) * (1 + y_ / 140)))
for x_, y_, h_ in TREEPOS:
    rod((x_, y_, ZG - 0.2), (x_, y_, ZG + 0.25 * h_), 0.04 * h_, M_LOG, 8)
    for j, (z0, z1, r0) in enumerate(((0.15, 0.6, 0.3), (0.4, 0.85, 0.22), (0.62, 1.0, 0.15))):
        c_ = obj(prim('cone', M_FOREST, 20, 1.0, 0.0, 1.0), 'ftree'); c_.location = (x_, y_, ZG + h_ * (z0 + z1) / 2); c_.scale = (r0 * h_, r0 * h_, (z1 - z0) * h_)
# a little village at the foot of the tree (shows the scale at the end)
M_HOUSE = mat('house', (0.45, 0.22, 0.12), 0.7); M_ROOF = mat('roof', (0.8, 0.85, 0.95), 0.6)
M_WIN = emis('window', (1.0, 0.7, 0.3), 6.0)
for q, (x_, y_, s_) in enumerate(((-5.2, 5.5, 1.0), (-7.0, 8.0, 1.2), (5.6, 6.0, 1.1), (7.6, 9.5, 0.9), (-9.5, 11.0, 1.3), (9.8, 12.5, 1.2),
                                  (-3.8, 10.5, 0.8), (4.2, 11.5, 0.85))):
    w_, h_ = 1.2 * s_, 0.9 * s_
    box((x_, y_, ZG + h_ / 2), (w_, w_ * 0.8, h_), M_HOUSE)
    rf = obj(prim('cone', M_ROOF, 4, 1.0, 0.0, 1.0), 'roof'); rf.location = (x_, y_, ZG + h_ + 0.3 * s_); rf.scale = (w_ * 0.78, w_ * 0.64, 0.6 * s_); rf.rotation_euler = (0, 0, math.pi / 4)
    for dx in (-0.25, 0.25): box((x_ + dx * w_, y_ - w_ * 0.41, ZG + h_ * 0.55), (0.18 * s_, 0.02, 0.22 * s_), M_WIN)
    box((x_ + 0.3 * w_, y_, ZG + h_ + 0.45 * s_), (0.14 * s_, 0.14 * s_, 0.5 * s_), mat('chimney', (0.3, 0.12, 0.08), 0.8))
    pl = bpy.data.lights.new('houselight', 'POINT'); pl.energy = 25.0 * s_; pl.color = (1.0, 0.6, 0.25)
    po = link(bpy.data.objects.new('houselight', pl)); po.location = (x_, y_ - w_ * 0.6, ZG + h_ * 0.6)
# sky: mountains and the moon ride along with the camera height (far away)
skyrig = link(bpy.data.objects.new('skyrig', None))
key_series(skyrig, 'location', 2, 0.8 * (FOC[:, 2] - FOC[0, 2]))
M_MTN = mat('mtn', (0.12, 0.16, 0.3), 0.8)
nt = M_MTN.node_tree; bsdf = nt.nodes['Principled BSDF']
geo2 = nt.nodes.new('ShaderNodeNewGeometry'); sx2 = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(geo2.outputs['Position'], sx2.inputs[0])
mr2 = nt.nodes.new('ShaderNodeMapRange'); mr2.inputs['From Min'].default_value = 18.0; mr2.inputs['From Max'].default_value = 26.0
nt.links.new(sx2.outputs['Z'], mr2.inputs['Value'])
mx2 = nt.nodes.new('ShaderNodeMix'); mx2.data_type = 'RGBA'; mx2.inputs[6].default_value = (0.1, 0.13, 0.26, 1); mx2.inputs[7].default_value = (0.75, 0.82, 0.95, 1)
nt.links.new(mr2.outputs[0], mx2.inputs[0]); nt.links.new(mx2.outputs[2], bsdf.inputs['Base Color'])
for x_, h_, r_ in ((-60, 34, 50), (-15, 40, 55), (35, 30, 48), (80, 38, 52), (10, 26, 40)):
    mo = obj(prim('cone', M_MTN, 9, 1.0, 0.0, 1.0), 'mtn'); mo.location = (x_, 230, ZG + h_ / 2 - 6); mo.scale = (r_, r_ * 0.5, h_); mo.parent = skyrig
moon = sphere((-16, 300, 52), 9.0, emis('moon', (1.0, 0.96, 0.85), 4.0), (32, 16)); moon.parent = skyrig
# world: night-sky gradient + stars
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True; nt = w.node_tree
bgn = nt.nodes['Background']
tc_ = nt.nodes.new('ShaderNodeTexCoord'); sxw = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(tc_.outputs['Generated'], sxw.inputs[0])
grad = nt.nodes.new('ShaderNodeMapRange'); grad.inputs['From Min'].default_value = 0.45; grad.inputs['From Max'].default_value = 0.8
nt.links.new(sxw.outputs['Z'], grad.inputs['Value'])
skyc = nt.nodes.new('ShaderNodeMix'); skyc.data_type = 'RGBA'; skyc.inputs[6].default_value = (0.05, 0.08, 0.2, 1); skyc.inputs[7].default_value = (0.006, 0.01, 0.035, 1)
nt.links.new(grad.outputs[0], skyc.inputs[0])
vor = nt.nodes.new('ShaderNodeTexVoronoi'); vor.inputs['Scale'].default_value = 260.0; nt.links.new(tc_.outputs['Generated'], vor.inputs['Vector'])
stl = nt.nodes.new('ShaderNodeMath'); stl.operation = 'LESS_THAN'; stl.inputs[1].default_value = 0.035; nt.links.new(vor.outputs['Distance'], stl.inputs[0])
stm = nt.nodes.new('ShaderNodeMath'); stm.operation = 'MULTIPLY'; stm.inputs[1].default_value = 3.0; nt.links.new(stl.outputs[0], stm.inputs[0])
sta = nt.nodes.new('ShaderNodeMix'); sta.data_type = 'RGBA'; sta.inputs[7].default_value = (1.0, 1.0, 1.0, 1)
stc = nt.nodes.new('ShaderNodeMath'); stc.operation = 'MULTIPLY'; nt.links.new(stm.outputs[0], stc.inputs[0]); nt.links.new(grad.outputs[0], stc.inputs[1])
stc2 = nt.nodes.new('ShaderNodeMath'); stc2.operation = 'MINIMUM'; stc2.inputs[1].default_value = 1.0; nt.links.new(stc.outputs[0], stc2.inputs[0])
nt.links.new(stc2.outputs[0], sta.inputs[0]); nt.links.new(skyc.outputs[2], sta.inputs[6])
nt.links.new(sta.outputs[2], bgn.inputs['Color']); bgn.inputs['Strength'].default_value = 1.0
# falling snow around the camera
snowrig = link(bpy.data.objects.new('snowrig', None)); key_loc(snowrig, np.stack([FOC[:, 0], np.zeros(NF), FOC[:, 2]], 1))
M_FLAKE = emis('flake', (0.9, 0.95, 1.0), 1.5)
for q in range(110):
    fl_ = obj(prim('sph', M_FLAKE, 6, 4), 'flake'); fl_.parent = snowrig; r_ = rng.uniform(0.018, 0.04); fl_.scale = (r_,) * 3
    x0 = rng.uniform(-4.5, 4.5); y0 = rng.uniform(-3.5, 6.0); vz = rng.uniform(0.45, 0.8); per = 12.0 / vz; ph = rng.uniform(0, per)
    fl_.location = (x0, y0, 0.0)
    zf = np.array([6.5 - 12.0 * (((T[f] + ph) % per) / per) for f in range(NF)])
    keys = {0, NF - 1}
    for f in range(1, NF):
        if zf[f] > zf[f - 1]: keys |= {f - 1, f}
    for f in sorted(keys):
        fl_.location[2] = float(zf[f]); fl_.keyframe_insert('location', index=2, frame=f)

# ---------------- HUD (rides with the camera): height counter + question on top, keyword captions at the bottom ----------------
hud = link(bpy.data.objects.new('hud', None)); hud.parent = cam
HZ = 6.0; hh_ = HZ * 24 / LENS; WW = hh_ * 9 / 16
hud.location = (0, 0, -HZ)
top_g = link(bpy.data.objects.new('hudtop', None)); top_g.parent = hud; top_g.location = (0, hh_ * 0.415, 0)
pl_ = box((0, 0, -0.02), (WW * 0.9, hh_ * 0.1, 0.02), emis('hudpanel', (0.01, 0.025, 0.06), 1.0), name='hudplate'); pl_.parent = top_g
for sgn in (1, -1):
    gb_ = box((0, sgn * hh_ * 0.05, -0.01), (WW * 0.92, hh_ * 0.006, 0.02), M_GOLD, name='hudgold'); gb_.parent = top_g
ts_h = hh_ * 0.03
t1_ = text('HEIGHT', (-WW * 0.12, ts_h * 0.85, 0), ts_h * 1.25, M_TXT, rot=(0, 0, 0), align='RIGHT'); t1_.parent = top_g
tq = text('How tall will it grow?', (0, -ts_h * 0.3, 0), ts_h * 1.1, M_TXT_G, rot=(0, 0, 0)); tq.parent = top_g; fit_w(tq, WW * 0.84)
tj = text('もみの木はどこまで伸びる？', (0, -ts_h * 1.25, 0), ts_h * 0.5, M_TXT_B, rot=(0, 0, 0)); tj.parent = top_g
# height (m) = tree top above the snow
HT = np.zeros(NF)
for f in range(NF):
    zt = max(TIERS[k]['zb'] + TIER_H * GROW[k][f] for k in range(9) if GROW[k][f] > 1e-3)
    if GROW[8][f] > 0.999: zt = max(zt, Z_TOP - RS)
    if T[f] >= FINAL: zt = Z_TOP + RS
    HT[f] = (zt - ZG) * M_PER_U
HT = np.round(HT, 1)
DW = ts_h * 0.98; XD0 = -WW * 0.05
slots = [(0, 'tens'), (1, 'ones'), (2, 'dot'), (3, 'tenths')]
def digit_str(v): return f'{v:4.1f}'
for slot in range(4):
    xs_ = XD0 + DW * (0.5, 1.5, 2.15, 2.8)[slot]
    chars = '.' if slot == 2 else '0123456789'
    for ch in chars:
        o = text(ch, (xs_, ts_h * 0.85, 0), ts_h * 1.9, M_TXT_G, rot=(0, 0, 0), align='CENTER'); o.parent = top_g
        vis = np.array([1.0 if digit_str(HT[f])[slot] == ch else 0.0 for f in range(NF)])
        if not vis.any(): o.hide_render = True; continue
        for c_ in range(3): key_series(o, 'scale', c_, np.maximum(0.001, vis))
um = text('m', (XD0 + 3.45 * DW, ts_h * 0.85, 0), ts_h * 1.5, M_TXT_G, rot=(0, 0, 0), align='LEFT'); um.parent = top_g
bump = np.ones(NF)
for f in range(1, NF):
    if HT[f] - HT[f - 1] > 0.05:
        for df in range(5):
            if f + df < NF: bump[f + df] = max(bump[f + df], 1.0 + 0.04 * (1 - df / 5))
for c_ in range(2): key_series(top_g, 'scale', c_, bump, 1e-3)
# bottom captions (search keywords in the first 30 s; English first, Japanese smaller)
CAPS = [(0.0, 7.4, [('O Christmas Tree', 1.5, M_TXT_G), ('"O Tannenbaum" · German Christmas carol (1824)', 0.9, M_TXT), ('もみの木（ドイツ民謡・クリスマスソング）', 0.68, M_TXT_B)]),
        (7.4, 14.8, [('Christmas Marble Machine', 1.25, M_TXT_G), ('Satisfying marble music · ASMR', 0.95, M_TXT), ('クリスマス ビー玉からくり', 0.68, M_TXT_B)]),
        (14.8, 23.0, [('Every note grows the tree!', 1.15, M_TXT_G), ('Classic Christmas song · marble music', 0.9, M_TXT), ('音符ひとつで木が伸びる', 0.68, M_TXT_B)]),
        (23.0, 30.5, [('Verse 2: pinecones play the bass', 1.0, M_TXT_G), ('Ernst Anschütz 1824 · German folk song', 0.9, M_TXT), ('2番：まつぼっくりがベースで参加', 0.68, M_TXT_B)]),
        (T_V3 - 1.2, T_V3 + 5.5, [('Verse 3: higher than the forest!', 1.0, M_TXT_G), ('3番：森よりも高く！', 0.68, M_TXT_B)]),
        (FINAL + 0.4, 999, [('MERRY CHRISTMAS!', 1.6, M_TXT_G), ('One fallen star · a 30 m Christmas tree', 0.9, M_TXT), ('メリークリスマス', 0.75, M_TXT_B)])]
for t0_, t1c, lines in CAPS:
    g_ = link(bpy.data.objects.new('cap', None)); g_.parent = hud; g_.location = (0, -hh_ * 0.375, 0)
    nl = len(lines); hgt = sum(s * ts_h * 1.3 for _, s, _ in lines) + ts_h * 0.6
    pnl = box((0, 0, -0.02), (WW * 0.94, hgt, 0.02), emis('cappanel', (0.008, 0.02, 0.05), 1.0), name='cappl'); pnl.parent = g_
    for sgn in (1, -1):
        gb_ = box((0, sgn * hgt / 2, -0.01), (WW * 0.96, hh_ * 0.005, 0.02), M_GOLD); gb_.parent = g_
    yy = hgt / 2 - ts_h * 0.3
    for s_, sz, mt in lines:
        yy -= sz * ts_h * 1.3
        o = text(s_, (0, yy + sz * ts_h * 0.45, 0), sz * ts_h, mt, rot=(0, 0, 0)); o.parent = g_; fit_w(o, WW * 0.88)
    vis = np.array([max(0.001, min(1, (T[f] - t0_) / 0.25, (t1c - T[f]) / 0.25)) if t0_ <= T[f] <= t1c else 0.001 for f in range(NF)])
    for c_ in range(3): key_series(g_, 'scale', c_, vis, 1e-3)
for o in bpy.data.objects:
    p_ = o.parent
    while p_ is not None and p_ is not hud: p_ = p_.parent
    if p_ is hud: o.visible_shadow = False

# ---------------- lights ----------------
sun = bpy.data.lights.new('moonlight', 'SUN'); sun.energy = 1.4; sun.color = (0.62, 0.72, 1.0); sun.angle = math.radians(2)
so = link(bpy.data.objects.new('moonlight', sun)); so.rotation_euler = (math.radians(55), math.radians(18), math.radians(160))
key_ = bpy.data.lights.new('key', 'AREA'); key_.size = 6.0; key_.color = (1.0, 0.84, 0.62); key_.energy = 700.0
ko = link(bpy.data.objects.new('key', key_))
fill = bpy.data.lights.new('fill', 'AREA'); fill.size = 8.0; fill.color = (0.55, 0.65, 1.0); fill.energy = 220.0
fo_ = link(bpy.data.objects.new('fill', fill))
for f in range(0, NF, 5):
    ko.location = (CP[f, 0] - 2.5, -4.5, FOC[f, 2] + 5.0); ko.rotation_euler = (math.radians(45), 0, math.radians(-25))
    ko.keyframe_insert('location', frame=f); ko.keyframe_insert('rotation_euler', frame=f)
    fo_.location = (CP[f, 0] + 4.0, -5.0, FOC[f, 2] + 1.0); fo_.rotation_euler = (math.radians(75), 0, math.radians(35))
    fo_.keyframe_insert('location', frame=f); fo_.keyframe_insert('rotation_euler', frame=f)
for tr in TIERS:
    gl = bpy.data.lights.new('tierglow', 'POINT'); gl.color = (1.0, 0.75, 0.4); gl.shadow_soft_size = 1.0
    go = link(bpy.data.objects.new('tierglow', gl)); go.location = (0, YT - tr['R'] - 0.8, tr['zb'] + 1.0)
    for t_, e_ in ((0, 0.0), (tr['t_lit'] - 0.05, 0.0), (tr['t_lit'] + 0.3, 90.0), (FINAL - 0.05, 90.0), (FINAL + 0.2, 220.0), (FINAL + 1.5, 130.0)):
        gl.energy = e_; gl.keyframe_insert('energy', frame=max(0, int(t_ * FPS)))
sg = bpy.data.lights.new('starglow', 'POINT'); sg.color = (1.0, 0.82, 0.45)
sgo = link(bpy.data.objects.new('starglow', sg)); sgo.location = (top[0], top[1] - 0.8, top[2])
for t_, e_ in ((0, 0.0), (FINAL - 0.02, 0.0), (FINAL + 0.1, 500.0), (FINAL + 1.2, 260.0)):
    sg.energy = e_; sg.keyframe_insert('energy', frame=int(t_ * FPS))
stl_ = bpy.data.lights.new('stellaglow', 'POINT'); stl_.color = (1.0, 0.8, 0.4); stl_.energy = 30.0; stl_.shadow_soft_size = 0.2
slo = link(bpy.data.objects.new('stellaglow', stl_)); slo.parent = BST; slo.location = (0, -0.3, 0.1)

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
        print('t', tt_, 'cam', [round(v, 2) for v in cam.matrix_world.translation], 'WID', round(WID[f], 2), 'FOC', np.round(FOC[f], 2), 'H', HT[f])
print('scene objects', len(bpy.data.objects))
