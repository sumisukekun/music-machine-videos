"""Deck the Halls - 'Can 9 marbles decorate a Christmas tree?'  Side view of a long candle-lit hall on Christmas Eve.
Every melody note = a ball landing on a key / bell of that pitch: Holly (red marble) plays the verse lines on xylophone keys,
nine glass ornaments play every 'Fa-la-la' on the bells of the next bare tree and bounce into it (and light up), the Yule-log ball
plays the bass in verse 2 on wooden bass bars; the star marble lands on the top of the last tree on the final chord.
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
FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.3) * FPS); T = np.arange(NF) / FPS
G = LAY['G']; EV = LAY['ev']; BALLS = LAY['balls']; RR = LAY['R']; TRG = LAY['tree']
Y_ORN, Y_BASS = LAY['y_orn'], LAY['y_bass']
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


def text(s, loc, size, m, rot=(math.pi / 2, 0, 0), align='CENTER', ext=0.01):
    cu = bpy.data.curves.new('t', 'FONT'); cu.body = s; cu.font = FONT_EN if _latin(s) else FONT
    cu.size = size * (EN_K if _latin(s) else 1.0); cu.extrude = ext
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
PH = POS['holly']; PY = POS['yule']
DEV = EV['dev']; TREES = EV['trees']
T_V2 = E['bass'][0][0]

# ---------------- materials ----------------
M_HOLLY = mat('holly', (0.8, 0.02, 0.03), 0.08, **{'Coat Weight': 1.0})
M_HOLLYLEAF = mat('hleaf', (0.03, 0.35, 0.08), 0.35, **{'Coat Weight': 0.6})
M_YULE = mat('yule', (1.0, 0.62, 0.18), 0.18, **{'Metallic': 0.95})
M_STARB = mat('starb', (1.0, 0.85, 0.3), 0.15, **{'Metallic': 0.8, 'Emission Color': (1.0, 0.8, 0.3), 'Emission Strength': 2.5})
M_GOLD = mat('gold', (1.0, 0.72, 0.28), 0.22, **{'Metallic': 0.95})
M_GOLDD = mat('goldd', (0.65, 0.42, 0.14), 0.35, **{'Metallic': 0.9})
M_BRASS = mat('brass', (0.85, 0.6, 0.25), 0.3, **{'Metallic': 0.9})
M_STEEL = mat('steel', (0.6, 0.6, 0.64), 0.28, **{'Metallic': 0.9})
M_DARK = mat('dark', (0.04, 0.035, 0.03), 0.5)
M_WOOD = mat('wood', (0.32, 0.16, 0.07), 0.45, **{'Coat Weight': 0.4})
M_WOODL = mat('woodl', (0.55, 0.32, 0.14), 0.5)
M_RED = mat('red', (0.6, 0.03, 0.04), 0.4, **{'Sheen Weight': 0.5})
M_GREEN = mat('green', (0.02, 0.22, 0.08), 0.5)
M_WHITE = mat('white', (0.9, 0.88, 0.82), 0.5)
M_KEY = glowmat('key', gain=7.0, base=0.0, rough=0.3, coat=0.3)
M_BELL = glowmat('bell', gain=6.0, base=0.15, rough=0.22, metal=0.8)
M_BASSBAR = glowmat('bassbar', gain=14.0, base=0.0, rough=0.35, coat=0.7)
M_ORN = glowmat('orn', gain=3.5, base=0.0, rough=0.05, metal=0.6, coat=1.0)
M_OBJ = objcol('objcol', 0.45)
M_FX = fxmat('fx', 8.0)
M_TXT = emis('txt', (1.0, 0.97, 0.9), 2.2)
M_TXT_G = emis('txtg', (1.0, 0.75, 0.25), 3.0)
M_TXT_R = emis('txtr', (1.0, 0.3, 0.25), 2.6)
M_TXT_D = mat('txtd', (0.08, 0.05, 0.03), 0.5)
M_PLATE = mat('plate', (0.06, 0.12, 0.07), 0.45, **{'Coat Weight': 0.5})


def pitch_col(p, sat=1.0):
    h = ((p - 53) % 12) / 12.0
    cols = [(1.0, 0.25, 0.2), (1.0, 0.55, 0.15), (1.0, 0.85, 0.2), (0.5, 0.9, 0.3), (0.2, 0.8, 0.6), (0.3, 0.6, 1.0), (0.75, 0.45, 1.0)]
    c = cols[int(h * 7) % 7]; return tuple(1 - sat + sat * x for x in c)


# ---------------- balls ----------------
def make_ball(name, r, m, spin=True):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, segments=32, ring_count=16); o = bpy.context.object; o.name = name
    for p in o.data.polygons: p.use_smooth = True
    o.data.materials.append(m); key_loc(o, POS[name])
    if spin:
        sp = np.zeros(NF)
        for f in range(1, NF): sp[f] = sp[f - 1] + (POS[name][f, 0] - POS[name][f - 1, 0]) / r
        o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, sp)
    return o
BH = make_ball('holly', RR['holly'], M_HOLLY)
# holly sprig on Holly (two leaves and berries), so the red marble reads as "Holly"
for sgn in (1, -1):
    lf = sphere((sgn * 0.06, -0.02, RR['holly'] * 0.92), 1.0, M_HOLLYLEAF, (12, 6)); lf.scale = (0.075, 0.03, 0.028); lf.rotation_euler = (0.3, 0, sgn * 0.5)
    lf.parent = BH
for dx in (-0.025, 0.025):
    b_ = sphere((dx, -0.04, RR['holly'] * 1.0), 0.024, mat('berry', (0.9, 0.0, 0.02), 0.1), (10, 6)); b_.parent = BH
BY = make_ball('yule', RR['bass'], M_YULE)
BS = make_ball('star', RR['star'], M_STARB, spin=False)
ORN_COLS = [(0.85, 0.05, 0.08), (1.0, 0.7, 0.15), (0.1, 0.35, 0.9), (0.85, 0.85, 0.9), (0.1, 0.6, 0.25), (0.75, 0.15, 0.7), (1.0, 0.4, 0.1),
            (0.3, 0.8, 0.95), (0.95, 0.85, 0.3)]
ORN = {}
for n, b in BALLS.items():
    if b['kind'] != 'orn': continue
    h = b['hits'][0]; j = int(n[3:]) % 9; col = ORN_COLS[(j * 4 + int(n[3:]) // 9) % 9]
    o = obj(prim('sph', M_ORN, 24, 12), n); o.scale = (RR['orn'],) * 3; key_loc(o, POS[n])
    for p in o.data.polygons: p.use_smooth = True
    cap = rod((0, 0, 0.85), (0, 0, 1.2), 0.28, M_GOLD, 10); cap.parent = o
    lit = np.zeros(NF); tc_ = h['t_cup']
    for f in range(NF):
        u = T[f] - tc_
        lit[f] = 0.0 if u < 0 else 0.35 + 0.65 * math.exp(-u / 0.35)
    lit += 0.8 * pulse([FINAL], decay=8, length=40)
    o.color = (*col, 0); key_series(o, 'color', 3, lit, 1e-3)
    o.color = (*col, float(lit[0]))
    ORN[n] = o

TOP = np.array(EV['cannon']['top'])
# ---------------- camera ----------------
FOC = np.zeros((NF, 3)); WID = np.zeros(NF)
TXS = np.array([tr['x'] for tr in TREES])
REFT = [(TN[[i for i in range(len(LEAD)) if LEAD[i][6] == 'orn' and LEAD[i][8] == k][0]], TN[[i for i in range(len(LEAD)) if LEAD[i][6] == 'orn' and LEAD[i][8] == k][-1]]) for k in range(8)]
for f in range(NF):
    t = T[f]; h = PH[f]
    # during each refrain, lean toward the tree being decorated
    wt = 0.0; xt = h[0]; ztop = h[2] + 3.0
    for k, (ta, tb) in enumerate(REFT):
        u = min(1.0, max(0.0, (t - ta + 0.6) / 0.9)) * min(1.0, max(0.0, (tb + 0.9 - t) / 0.9))
        if u > wt: wt = u; xt = TREES[k]['x']; ztop = TREES[k]['rail'] + 0.4
    x = (1 - 0.7 * wt) * (h[0] + 0.7) + 0.7 * wt * xt
    zlow = h[2] - 0.6
    if t > T_V2 - 1.0:
        zlow = min(zlow, PY[f][2] - 0.4) * min(1, (t - T_V2 + 1.0) / 2.0) + zlow * (1 - min(1, (t - T_V2 + 1.0) / 2.0))
    zhigh = max(h[2] + 1.2, (1 - wt) * (h[2] + 2.2) + wt * ztop)
    FOC[f] = (x, 0.0, (zlow + zhigh) / 2)
    WID[f] = max(4.6, (zhigh - zlow + 1.4) * 9 / 16)
    if t < 1.6:                                      # opening: close on the gift box
        u = min(1, max(0, (t - 0.4) / 1.2)); u = u * u * (3 - 2 * u)
        f0 = np.array([EV['giftbox']['p'][0] + 0.6, 0, EV['giftbox']['p'][2] + 0.6]); FOC[f] = f0 * (1 - u) + FOC[f] * u; WID[f] = 3.4 * (1 - u) + WID[f] * u
    tw = EV['cannon']['t_fire'] - 0.6
    if t > tw:                                       # the reveal: the whole last tree, star on top
        u = min(1, (t - tw) / 1.8); u = u * u * (3 - 2 * u)
        wide = np.array([TREES[7]['x'] - 0.4, 0.0, (PY[f][2] - 0.5 + TOP[2] + 0.9) / 2])
        FOC[f] = (1 - u) * FOC[f] + u * wide; WID[f] = (1 - u) * WID[f] + u * max(5.6, (TOP[2] + 0.9 - PY[f][2] + 0.8) * 9 / 16)
FOC[:, 0] = smooth(FOC[:, 0], 0.45); FOC[:, 2] = smooth(FOC[:, 2], 0.6); WID = smooth(WID, 0.6)
LENS = 32.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 300
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TPc = np.zeros((NF, 3))
for f in range(NF):
    D_ = WID[f] / (9 / 16) * LENS / 24
    CP[f] = (FOC[f, 0], -D_, FOC[f, 2] + 0.16 * D_)
    TPc[f] = (FOC[f, 0], 0.6, FOC[f, 2] - 0.15)
key_loc(cam, CP); key_loc(tgt, TPc)

def fz_cam(t): return float(np.interp(t, T, FOC[:, 2]))

# ---------------- Holly's keys (one per note), course rails ----------------
KEYS = []
for h in BALLS['holly']['hits']:
    if h['kind'] != 'key': continue
    n_ = np.array(h['n']); c = np.array(h['p']) - RR['holly'] * n_
    o = obj(prim('bcube', M_KEY), 'key')
    o.matrix_world = Matrix.Translation(Vector(c - n_ * 0.03)) @ frame_from_n(n_).to_4x4() @ Matrix.Diagonal((0.22, 0.42, 0.06, 1))
    col = pitch_col(h['pitch']); o.color = (*col, 0)
    flash(o, h['t'], col, n=n_)
    KEYS.append((c, h))
    for dy in (-0.15, 0.15):
        rod((c[0], dy, c[2] - 0.05), (c[0], dy, c[2] - 0.32), 0.012, M_BRASS, 6)
# one wooden rail under each line of keys
for k in range(8):
    hs = [h for h in BALLS['holly']['hits'] if h['kind'] == 'key' and LEAD[h['i']][8] == k]
    X = np.array([h['p'][0] for h in hs]); Z = np.array([h['p'][2] - RR['holly'] - 0.32 for h in hs])
    XX = np.linspace(X[0] - 0.25, X[-1] + 0.25, 24); ZZ = np.interp(XX, X, Z)
    ZM = np.array([Z[np.abs(X - x) < 0.4].min() if (np.abs(X - x) < 0.4).any() else z for x, z in zip(XX, ZZ)])
    strip_mesh(XX, ZM, -0.22, 0.22, 0.07, M_WOOD, 'rail')
    strip_mesh(XX, ZM + 0.012, -0.235, -0.205, 0.02, M_GOLD, 'railtrim')

# ---------------- floor, hall walls (follow the course down: a grand stepped hall) ----------------
allz = [h['p'][2] for b in BALLS.values() for h in b['hits']] + [tr['zb'] for tr in TREES]
allx = [h['p'][0] for b in BALLS.values() for h in b['hits']] + [tr['x'] for tr in TREES]
XA, XB = -6.0, max(allx) + 8.0
STEPX = np.arange(XA, XB + 3.0, 3.0)
def floor_at(x):
    zs = [tr['zb'] for tr in TREES if abs(tr['x'] - x) < 2.6]
    zs += [h['p'][2] - 0.45 for h in BALLS['yule']['hits'] if abs(h['p'][0] - x) < 1.6]
    zs += [h['p'][2] - 1.0 for h in BALLS['holly']['hits'] if abs(h['p'][0] - x) < 1.6]
    zs += [EV['yule']['start'][2] - 0.8] if abs(EV['yule']['start'][0] - x) < 2.0 else []
    if not zs: zs = [min(TREES, key=lambda tr: abs(tr['x'] - x))['zb']]
    return min(zs)
FLZ = np.array([floor_at(x + 1.5) for x in STEPX])
for i in range(1, len(FLZ)): FLZ[i] = min(FLZ[i], FLZ[i - 1] + 0.0) if False else FLZ[i]
FLZ = np.minimum.accumulate(FLZ)                                   # the hall only steps down
M_FLOOR = mat('floor', (0.25, 0.12, 0.06), 0.35, **{'Coat Weight': 0.6})
nt = M_FLOOR.node_tree; bsdf = nt.nodes['Principled BSDF']
wv = nt.nodes.new('ShaderNodeTexWave'); wv.inputs['Scale'].default_value = 3.0; wv.inputs['Distortion'].default_value = 4.0
mr = nt.nodes.new('ShaderNodeMix'); mr.data_type = 'RGBA'; mr.inputs[6].default_value = (0.28, 0.14, 0.06, 1); mr.inputs[7].default_value = (0.16, 0.07, 0.03, 1)
nt.links.new(wv.outputs['Fac'], mr.inputs[0]); nt.links.new(mr.outputs[2], bsdf.inputs['Base Color'])
M_WALL = mat('wall', (0.1, 0.03, 0.03), 0.7)
nt = M_WALL.node_tree; bsdf = nt.nodes['Principled BSDF']
gm = nt.nodes.new('ShaderNodeNewGeometry'); sx_ = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(gm.outputs['Position'], sx_.inputs[0])
mo = nt.nodes.new('ShaderNodeMath'); mo.operation = 'MULTIPLY'; mo.inputs[1].default_value = 1 / 0.45; nt.links.new(sx_.outputs['X'], mo.inputs[0])
fr = nt.nodes.new('ShaderNodeMath'); fr.operation = 'FRACT'; nt.links.new(mo.outputs[0], fr.inputs[0])
gt1 = nt.nodes.new('ShaderNodeMath'); gt1.operation = 'GREATER_THAN'; gt1.inputs[1].default_value = 0.5; nt.links.new(fr.outputs[0], gt1.inputs[0])
ln_ = nt.nodes.new('ShaderNodeMath'); ln_.operation = 'LESS_THAN'; ln_.inputs[1].default_value = 0.04; nt.links.new(fr.outputs[0], ln_.inputs[0])
m1 = nt.nodes.new('ShaderNodeMix'); m1.data_type = 'RGBA'; m1.inputs[6].default_value = (0.32, 0.035, 0.04, 1); m1.inputs[7].default_value = (0.24, 0.02, 0.03, 1)
nt.links.new(gt1.outputs[0], m1.inputs[0])
m2 = nt.nodes.new('ShaderNodeMix'); m2.data_type = 'RGBA'; m2.inputs[7].default_value = (0.6, 0.4, 0.12, 1)
nt.links.new(m1.outputs[2], m2.inputs[6]); nt.links.new(ln_.outputs[0], m2.inputs[0]); nt.links.new(m2.outputs[2], bsdf.inputs['Base Color'])
YW = 5.6
M_WIN = mat('win', (0.02, 0.04, 0.1), 0.05, **{'Emission Color': (0.12, 0.2, 0.45), 'Emission Strength': 1.2})
M_SNOWSILL = mat('snowsill', (0.9, 0.93, 1.0), 0.6)
M_FIRE = emis('fire', (1.0, 0.45, 0.1), 9.0)
for i, x0 in enumerate(STEPX[:-1]):
    zf = FLZ[i]
    box((x0 + 1.5, 1.2, zf - 0.15), (3.0, 10.0, 0.3), M_FLOOR, name='floor')
    if i > 0 and FLZ[i - 1] > zf + 1e-3:                                       # stair riser
        box((x0, 1.2, (zf + FLZ[i - 1]) / 2), (0.06, 10.0, FLZ[i - 1] - zf), M_WOODL, name='riser')
    box((x0 + 1.5, YW, zf + 6.0), (3.02, 0.2, 12.0), M_WALL, name='wall')
    box((x0 + 1.5, YW - 0.12, zf + 0.6), (3.02, 0.1, 1.2), M_WOOD, name='wainscot')
    box((x0 + 1.5, YW - 0.18, zf + 1.22), (3.02, 0.08, 0.08), M_GOLDD, name='dado')
# windows (night), fireplaces, candle sconces, along the back wall
for i, x0 in enumerate(STEPX[:-1]):
    zf = FLZ[i]; xc = x0 + 1.5
    if i % 3 == 0:
        box((xc, YW - 0.1, zf + 3.6), (1.3, 0.05, 2.2), M_WIN, name='window')
        for dx in (-0.66, 0.0, 0.66): box((xc + dx, YW - 0.14, zf + 3.6), (0.06, 0.05, 2.3), M_WOODL)
        box((xc, YW - 0.14, zf + 3.6), (1.38, 0.05, 0.06), M_WOODL)
        box((xc, YW - 0.2, zf + 2.47), (1.5, 0.2, 0.08), M_SNOWSILL)
    elif i % 3 == 1:
        box((xc, YW - 0.35, zf + 0.75), (1.8, 0.5, 1.5), mat('brick', (0.35, 0.1, 0.06), 0.8), name='fireplace')
        box((xc, YW - 0.62, zf + 0.6), (1.0, 0.05, 0.9), M_DARK)
        box((xc, YW - 0.66, zf + 0.35), (0.8, 0.04, 0.35), M_FIRE, name='fire')
        box((xc, YW - 0.4, zf + 1.55), (2.1, 0.7, 0.12), M_WOOD, name='mantel')
        for dx in (-0.6, -0.2, 0.2, 0.6):                                       # stockings
            box((xc + dx, YW - 0.8, zf + 1.25), (0.16, 0.06, 0.4), M_RED); box((xc + dx + 0.05, YW - 0.8, zf + 1.06), (0.24, 0.06, 0.12), M_RED)
            box((xc + dx, YW - 0.82, zf + 1.42), (0.18, 0.07, 0.07), M_WHITE)
        fl = bpy.data.lights.new('firel', 'POINT'); fl.energy = 60.0; fl.color = (1.0, 0.5, 0.2); fl.shadow_soft_size = 0.3
        fo = link(bpy.data.objects.new('firel', fl)); fo.location = (xc, YW - 1.2, zf + 0.6)
    else:
        for dx in (-0.8, 0.8):
            box((xc + dx, YW - 0.2, zf + 2.8), (0.12, 0.2, 0.06), M_BRASS)
            rod((xc + dx, YW - 0.3, zf + 2.83), (xc + dx, YW - 0.3, zf + 3.1), 0.03, M_WHITE, 8)
            sphere((xc + dx, YW - 0.3, zf + 3.16), 0.04, emis('flame', (1.0, 0.7, 0.3), 12.0), (8, 6))
        box((xc, YW - 0.12, zf + 3.4), (1.0, 0.06, 1.3), M_GOLDD, name='frame')
        box((xc, YW - 0.16, zf + 3.4), (0.86, 0.04, 1.16), mat('paint', (0.1, 0.18, 0.12), 0.6))

# ---------------- string of lights along the hall: one bulb per Holly key, lit when she strikes it ----------------
M_BULB = lightmat('bulb', (1.0, 0.75, 0.35), on=5.0, flash_=18.0)
quads = []; tons = []; swag = []
hk = [h for h in BALLS['holly']['hits'] if h['kind'] == 'key']
zs_course = smooth(np.array([PH[min(NF - 1, int(h['t'] * FPS))][2] for h in hk]), 0.2) if False else None
for j, h in enumerate(hk):
    x = h['p'][0]; zt = fz_cam(h['t']) + 3.25
    swag.append((x, zt))
    r = 0.07
    quads.append([(x - r, YW - 0.3, zt), (x, YW - 0.3, zt - r * 1.3), (x + r, YW - 0.3, zt), (x, YW - 0.3, zt + r * 1.3)]); tons.append(h['t'])
quads_mesh('bulbs', quads, tons, M_BULB)
SX = np.array([s[0] for s in swag]); SZ = np.array([s[1] for s in swag])
XX = np.arange(SX[0] - 1, SX[-1] + 1, 0.12); ZZ = np.interp(XX, SX, SZ) + 0.12
strip_mesh(XX, ZZ, YW - 0.31, YW - 0.27, 0.03, M_DARK, 'wire')
for g0 in np.arange(SX[0] - 1, SX[-1] + 1, 1.5):                                  # evergreen garland swags under the wire
    pts = [(g0 + 1.5 * u, np.interp(g0 + 1.5 * u, SX, SZ) - 0.1 - 0.35 * math.sin(math.pi * u)) for u in np.linspace(0, 1, 9)]
    for (xa_, za_), (xb_, zb_) in zip(pts, pts[1:]): rod((xa_, YW - 0.35, za_), (xb_, YW - 0.35, zb_), 0.07, M_GREEN, 8)

# ---------------- trees: cone with tier steps (inside the planned surface), lights, cups, star, pedestal ----------------
RBASE, HT, YT = TRG['rbase'], TRG['ht'], TRG['yt']
M_TREE = mat('tree', (0.02, 0.16, 0.05), 0.75)
nt = M_TREE.node_tree; bsdf = nt.nodes['Principled BSDF']
nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 28.0; nz.inputs['Detail'].default_value = 6.0
bmp = nt.nodes.new('ShaderNodeBump'); bmp.inputs['Strength'].default_value = 0.6; nt.links.new(nz.outputs['Fac'], bmp.inputs['Height'])
nt.links.new(bmp.outputs['Normal'], bsdf.inputs['Normal'])
mr = nt.nodes.new('ShaderNodeMix'); mr.data_type = 'RGBA'; mr.inputs[6].default_value = (0.01, 0.08, 0.025, 1); mr.inputs[7].default_value = (0.05, 0.28, 0.08, 1)
nt.links.new(nz.outputs['Fac'], mr.inputs[0]); nt.links.new(mr.outputs[2], bsdf.inputs['Base Color'])
M_TLIGHT = lightmat('tlight', (1.0, 0.82, 0.45), on=6.0, flash_=16.0, off_col=(0.02, 0.06, 0.02))
M_TSTAR = lightmat('tstar', (1.0, 0.8, 0.3), on=8.0, flash_=25.0, off_col=(0.1, 0.075, 0.03))
M_VELVET = mat('velvet', (0.45, 0.02, 0.04), 0.6, **{'Sheen Weight': 1.0})
tl_q = []; tl_t = []
for tr in TREES:
    s = tr['s']; rb = RBASE * s; ht = HT * s; zb = tr['zb']; xT = tr['x']
    bm = bmesh.new(); NS, NZ = 40, 60; rings = []
    for iz in range(NZ + 1):
        z = ht * iz / NZ; rlin = rb * (1 - z / ht)
        saw = ((z / (0.85 * s)) % 1.0)
        r = max(0.0, rlin - 0.025 - 0.07 * saw)
        rings.append([bm.verts.new((xT + r * math.cos(2 * math.pi * a / NS), YT + r * math.sin(2 * math.pi * a / NS), zb + z)) for a in range(NS)])
    for iz in range(NZ):
        for a in range(NS):
            b_ = (a + 1) % NS; bm.faces.new((rings[iz][a], rings[iz][b_], rings[iz + 1][b_], rings[iz + 1][a]))
    me = bpy.data.meshes.new('tree'); bm.to_mesh(me); bm.free(); me.materials.append(M_TREE)
    for p in me.polygons: p.use_smooth = True
    link(bpy.data.objects.new('tree', me))
    # pedestal (tree stand) down to the floor
    fz = float(np.interp(xT, STEPX + 1.5, FLZ))
    rod((xT, YT, min(fz, zb) - 0.02), (xT, YT, zb + 0.4), 0.2, mat('trunk', (0.2, 0.1, 0.04), 0.7), 16)
    sk = rod((xT, YT, fz), (xT, YT, fz + 0.06), rb * 0.75, M_VELVET, 32)
    rod((xT, YT, fz + 0.055), (xT, YT, fz + 0.075), rb * 0.77, M_WHITE, 32)
    if zb - fz > 0.3:                                    # a tall tree stand (brass) when the tree stands on a step above the floor
        rod((xT, YT, fz), (xT, YT, zb), 0.32, M_GOLDD, 24)
    # tree lights: switch on with the refrain's last ornament
    t_on = max(o_['t_cup'] for o_ in EV['orn'] if o_['tree'] == tr['k'])
    rr_ = np.random.default_rng(tr['k'] + 5)
    for q in range(46):
        z = rr_.uniform(0.25, 0.92) * ht; r = rb * (1 - z / ht) - 0.02; a = rr_.uniform(-1.2, 1.2)
        px, py = xT + r * math.sin(a), YT - r * math.cos(a)
        d = 0.035; tl_q.append([(px - d, py - 0.01, zb + z), (px, py - 0.01, zb + z - d), (px + d, py - 0.01, zb + z), (px, py - 0.01, zb + z + d)])
        tl_t.append(t_on + 0.04 * q)
    # tree-top star (lit when the tree is done; the big one on the last tree is lit by the star marble)
    if tr['k'] < 7:
        st = obj(prim('star', M_TSTAR, 0.06), 'tstar'); st.location = (xT, YT - 0.05, zb + ht + 0.12); st.scale = (0.2,) * 3
        me_ = st.data.copy(); st.data = me_
        a = me_.attributes.new('t_on', 'FLOAT', 'FACE'); a.data.foreach_set('value', [t_on + 0.3] * len(me_.polygons))
    # the cups (small gold hooks) the ornaments land in
    for o_ in EV['orn']:
        if o_['tree'] != tr['k']: continue
        c = np.array(o_['cup'])
        cu = obj(prim('torus', M_GOLD, 0.06, 0.012), 'cup'); cu.location = (c[0], c[1], c[2] - RR['orn'] + 0.015)
        rod((c[0], c[1], c[2] - RR['orn'] + 0.01), (c[0], c[1] + 0.12, c[2] - RR['orn'] - 0.02), 0.01, M_GOLD, 5)
    # number plate on the pedestal
    fit_w(text(f'TREE {tr["k"] + 1}', (xT, YT - rb * 0.78, fz + 0.25), 0.18, M_TXT_G), 1.2)
quads_mesh('treelights', tl_q, tl_t, M_TLIGHT)
# the big star on the last tree
tr7 = TREES[7]; top = np.array(EV['cannon']['top'])
BIGSTAR = obj(prim('star', M_TSTAR, 0.07), 'bigstar'); BIGSTAR.location = (top[0], top[1] - 0.02, top[2]); BIGSTAR.scale = (0.42,) * 3
me_ = BIGSTAR.data.copy(); BIGSTAR.data = me_
a = me_.attributes.new('t_on', 'FLOAT', 'FACE'); a.data.foreach_set('value', [FINAL] * len(me_.polygons))
rod((top[0], YT + 0.05, top[2] - 0.45), (top[0], YT + 0.05, tr7['top'] - 0.1), 0.025, M_GOLDD, 8)

# ---------------- ornament bells (height = pitch), the garland rail with the ornaments' clips ----------------
for tr in TREES:
    bl = tr['bells']
    for b_ in bl:
        o_ = BALLS[b_['ball']]; h = o_['hits'][0]
        n_ = np.array(h['n']); c = np.array(h['p']) - RR['orn'] * n_
        be = obj(prim('bell', M_BELL, 20), 'bell')
        be.matrix_world = Matrix.Translation(Vector(c)) @ frame_from_n(n_).to_4x4() @ Matrix.Diagonal((0.1, 0.1, 0.17, 1))
        col = (1.0, 0.78, 0.35); be.color = (*col, 0); flash(be, h['t'], col, n=n_, dip=0.02)
        rod((c[0], Y_ORN, c[2] - 0.15), (c[0], Y_ORN, tr['bells'][0]['z'] - 0.75 + 0.0), 0.012, M_BRASS, 6)
    xs = [b_['x'] for b_ in bl]; zb0 = min(b_['z'] for b_ in bl) - 0.75
    rod((min(xs) - 0.15, Y_ORN, zb0), (max(xs) + 0.15, Y_ORN, zb0), 0.03, M_GOLDD, 8)          # bell bar
    zr = tr['rail']
    rod((min(xs) - 0.2, Y_ORN, zr), (max(xs) + 0.2, Y_ORN, zr), 0.035, M_GOLD, 10)            # garland rail
    for x_ in (min(xs) - 0.2, max(xs) + 0.2):
        rod((x_, Y_ORN, zr), (x_, Y_ORN + 0.6, zr + 0.25), 0.025, M_GOLD, 8)
        rod((x_, Y_ORN + 0.6, zr + 0.25), (x_, YT - 0.2, zr + 0.25), 0.025, M_GOLD, 8)
    gx = np.linspace(min(xs) - 0.2, max(xs) + 0.2, 16)
    for xa_, xb_ in zip(gx, gx[1:]):
        rod((xa_, Y_ORN + 0.06, zr + 0.06 + 0.03 * math.sin(xa_ * 9)), (xb_, Y_ORN + 0.06, zr + 0.06 + 0.03 * math.sin(xb_ * 9)), 0.05, M_GREEN, 6)
    for b_ in bl:                                                                                  # clip on the rail above each bell
        o_ = BALLS[b_['ball']]; p0 = np.array(o_['seg'][0]['p']); tr_rel = o_['seg'][0]['t1']
        piv = link(bpy.data.objects.new('clip', None)); piv.location = (p0[0], Y_ORN, zr - 0.02)
        for sgn in (1, -1):
            jaw = box((sgn * 0.04, 0, -0.05), (0.025, 0.06, 0.09), M_GOLD); jaw.parent = piv
        ang = np.array([0.0 if T[f] < tr_rel else min(1.0, (T[f] - tr_rel) / 0.08) * 0.6 for f in range(NF)])
        piv.rotation_mode = 'XYZ'; key_series(piv, 'scale', 0, 1.0 + 0.8 * ang, 1e-4)

# ---------------- bass bars (verse 2) on posts, the trigger lever, the spring cannon ----------------
M_BASSWOOD = mat('basswood', (0.4, 0.2, 0.08), 0.35, **{'Coat Weight': 0.6})
for h in BALLS['yule']['hits']:
    n_ = np.array(h['n']); c = np.array(h['p']) - RR['bass'] * n_
    w, d = (0.3, 0.36) if h['kind'] != 'bassfinal' else (0.42, 0.42)
    o = obj(prim('bcube', M_BASSBAR), 'bassbar')
    o.matrix_world = Matrix.Translation(Vector(c - n_ * 0.04)) @ frame_from_n(n_).to_4x4() @ Matrix.Diagonal((w, d, 0.08, 1))
    col = (0.42, 0.17, 0.05) if h['kind'] != 'trigger' else (0.75, 0.05, 0.04)
    o.color = (*col, 0); flash(o, h['t'], col, n=n_, dip=0.03)
    fz = float(np.interp(c[0], STEPX + 1.5, FLZ))
    for dy in (-0.12, 0.12): rod((c[0], c[1] + dy, c[2] - 0.08), (c[0], c[1] + dy, fz), 0.02, M_BRASS, 6)
can = EV['cannon']; pC = np.array(can['p']); vC = np.array(can['v']); uC = vC / np.linalg.norm(vC)
trg = np.array(EV['trigger']['p'])
barrel = rod(pC - uC * 0.55, pC + uC * 0.18, 0.17, mat('cannon', (0.08, 0.25, 0.12), 0.3, **{'Metallic': 0.6, 'Coat Weight': 0.8}), 24)
for u_ in (-0.5, -0.05, 0.16):
    rod(pC + uC * (u_ - 0.02), pC + uC * (u_ + 0.02), 0.19, M_GOLD, 24)
fz = float(np.interp(pC[0], STEPX + 1.5, FLZ))
rod(pC - uC * 0.5, (pC[0], pC[1], fz), 0.06, M_GOLDD, 10)
# spring plunger inside (visible through the muzzle as it shoots up): piston from 0.45 below up to the muzzle at the fire time
pist = rod((0, 0, -0.05), (0, 0, 0.05), 0.13, M_STEEL, 16); pist.parent = None
PP = np.zeros((NF, 3))
for f in range(NF):
    u = (T[f] - (can['t_fire'] - 0.04)) / 0.04
    s_ = -0.42 if u < 0 else (-0.42 + 0.42 * min(1, u) if T[f] < can['t_fire'] + 0.5 else -0.42 + 0.42 * max(0, 1 - (T[f] - can['t_fire'] - 0.5) / 0.6))
    PP[f] = pC + uC * (s_ - RR['star'] - 0.02)
key_loc(pist, PP); pist.rotation_mode = 'QUATERNION'; pist.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(Vector(uC))
# trigger link: a rod from the trigger bar back to the cannon's latch
rod((trg[0], trg[1] + 0.2, trg[2] - RR['bass'] - 0.1), (pC[0], pC[1] - 0.15, pC[2] - 0.35), 0.022, M_STEEL, 6)
fit_w(text('STAR CANNON', (pC[0] + 0.2, pC[1] - 0.45, pC[2] - 0.75), 0.14, M_TXT_G), 1.4)
smoke = sphere(tuple(pC + uC * 0.25), 0.3, M_FX, (12, 8)); smoke.color = (1, 0.9, 0.7, 0)
fs = int(round(can['t_fire'] * FPS)); sa = np.zeros(NF); ss = np.full(NF, 0.01)
for df in range(22):
    if fs + df < NF: sa[fs + df] = 0.9 * (1 - df / 22) ** 1.5; ss[fs + df] = 0.3 + 1.1 * (1 - (1 - df / 22) ** 2)
key_series(smoke, 'color', 3, sa, 1e-3)
for k in range(3): key_series(smoke, 'scale', k, ss * 0.3)

# ---------------- the gift box at the start (spring lid, spring plunger) ----------------
gb = EV['giftbox']; pg = np.array(gb['p']); RH = RR['holly']
def gift(c, size, col, ribbon, t_open, t_close=None, t_reopen=None, name='gift', open_front=False):
    """open-top gift box; the lid is hinged at the back edge"""
    w, d, h = size; zb_ = c[2]
    box((c[0], c[1], zb_ + 0.02), (w, d, 0.04), col)
    for sx in ((-1,) if open_front else (-1, 1)): box((c[0] + sx * w / 2, c[1], zb_ + h / 2), (0.04, d, h), col)
    for sy in (-1, 1): box((c[0], c[1] + sy * d / 2, zb_ + h / 2), (w, 0.04, h), col)
    box((c[0], c[1] - d / 2 - 0.022, zb_ + h / 2), (0.08, 0.01, h), ribbon)
    for sx in ((-1,) if open_front else (-1, 1)): box((c[0] + sx * (w / 2 + 0.022), c[1], zb_ + h / 2), (0.01, 0.08, h), ribbon)
    hg = link(bpy.data.objects.new('lidhinge', None)); hg.location = (c[0], c[1] + d / 2, zb_ + h)
    lid = box((0, -d / 2, 0.03), (w + 0.06, d + 0.06, 0.06), col); lid.parent = hg
    bw = box((0, -d / 2, 0.065), (0.08, d + 0.07, 0.012), ribbon); bw.parent = hg
    for sgn in (1, -1):
        lp = sphere((sgn * 0.07, -d / 2, 0.1), 1.0, ribbon, (12, 6)); lp.scale = (0.08, 0.035, 0.045); lp.parent = hg
    ang = np.zeros(NF)
    for f in range(NF):
        t = T[f]; a = 0.0
        if t_open is not None and t >= t_open: a = 1.9 * min(1, (t - t_open) / 0.18) + 0.12 * math.sin(14 * (t - t_open)) * math.exp(-4 * (t - t_open))
        if t_close is not None and t >= t_close: a = 1.9 * max(0, 1 - (t - t_close) / 0.25)
        if t_reopen is not None and t >= t_reopen: a = 1.9 * min(1, (t - t_reopen) / 0.12) + 0.15 * math.sin(16 * (t - t_reopen)) * math.exp(-3 * (t - t_reopen))
        ang[f] = a
    hg.rotation_mode = 'XYZ'; key_series(hg, 'rotation_euler', 0, -ang, 1e-4)
    return hg
M_GIFT1 = mat('gift1', (0.05, 0.3, 0.15), 0.35, **{'Coat Weight': 0.5}); M_RIB = mat('ribbon', (1.0, 0.75, 0.2), 0.25, **{'Metallic': 0.6})
gift((pg[0] - 0.12, 0, pg[2] - RH - 0.04), (0.62, 0.5, 0.42), M_GIFT1, M_RIB, gb['t_lid'], open_front=True)
# the box's front wall (facing +x) is the low side so she can leave; the plunger behind her
box((pg[0] - 0.12, 0, pg[2] - RH - 0.1), (0.9, 0.7, 0.08), M_WOOD, name='shelf')
rod((pg[0] - 0.12, 0, pg[2] - RH - 0.14), (pg[0] - 0.12, 0, floor_at(pg[0]) + 0.0), 0.04, M_BRASS, 8)
pl = rod((-0.32, 0, 0), (0, 0, 0), 0.05, M_STEEL, 12)
bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=0.03, rotation=(0, math.pi / 2, 0)); ph_ = bpy.context.object; ph_.data.materials.append(M_RED); ph_.parent = pl
px = np.zeros(NF)
for f in range(NF):
    t = T[f]; u = (t - (gb['t_push'] - 0.05)) / 0.05
    px[f] = pg[0] - RH - 0.02 - 0.2 + (0.2 * min(1, max(0, u)) if t < gb['t_push'] + 0.4 else 0.2 * max(0, 1 - (t - gb['t_push'] - 0.4) / 0.5))
pl.location = (0, 0, pg[2]); key_series(pl, 'location', 0, px)
# the box wall in front of Holly is cut down so she can roll out: drawn as the open front above
# ---------------- machines ----------------
def bucket(c, w, d, h, m):
    """open bucket around ball centre c: floor at c.z - RH - 0.02"""
    zf_ = c[2] - RH - 0.03; objs = []
    objs.append(box((c[0], c[1], zf_ - 0.02), (w, d, 0.04), m))
    for sx in (-1, 1): objs.append(box((c[0] + sx * w / 2, c[1], zf_ + h / 2), (0.03, d, h), m))
    for sy in (-1, 1): objs.append(box((c[0], c[1] + sy * d / 2, zf_ + h / 2), (w, 0.03, h), m))
    return objs


def follow(objs, P, anchor):
    """parent objects to an empty that follows positions P[f] (anchor = the position objects were built at)"""
    e = link(bpy.data.objects.new('follow', None)); e.location = anchor
    for o in objs: o.parent = e; o.matrix_parent_inverse = Matrix.Translation(-Vector(anchor))
    key_loc(e, P); return e


def trapdoor(c, t_open, m, w=0.3, d=0.3, parent=None):
    """floor flap under ball centre c, hinged at its +x edge, swings down at t_open"""
    hg = link(bpy.data.objects.new('trap', None)); hg.location = (c[0] + w / 2, c[1], c[2] - RH - 0.035)
    fl_ = box((-w / 2, 0, -0.015), (w, d, 0.03), m); fl_.parent = hg
    ang = np.array([0.0 if T[f] < t_open else min(1.0, (T[f] - t_open) / 0.12) ** 2 * 1.5 for f in range(NF)])
    hg.rotation_mode = 'XYZ'; key_series(hg, 'rotation_euler', 1, ang)
    if parent is not None: hg.parent = parent; hg.matrix_parent_inverse = parent.matrix_world.inverted() if False else Matrix.Translation(-parent.location)
    return hg


M_LIFT = mat('lift', (0.75, 0.08, 0.06), 0.3, **{'Coat Weight': 0.8})
M_LIFT2 = mat('lift2', (0.1, 0.35, 0.6), 0.3, **{'Coat Weight': 0.8})
M_TRAIN = mat('train', (0.05, 0.25, 0.12), 0.3, **{'Coat Weight': 0.8})
SIGNS = []
for D in DEV:
    k = D['k']; pc = np.array(D['catch'])
    if D['kind'] == 'lift':
        pr = np.array(D['rel'])
        # inclined lift: two guide rails, a bucket that slides up them, a motor box at the top
        d_ = (pr - pc) / np.linalg.norm(pr - pc)
        for dy in (-0.22, 0.22):
            rod(pc + [0, dy, -RH - 0.15] - d_ * 0.2, pr + [0, dy, -RH - 0.15] + d_ * 0.25, 0.022, M_STEEL, 8)
        box(tuple(pr + d_ * 0.3 + [0.05, 0, -RH - 0.1]), (0.22, 0.5, 0.22), M_DARK, name='motor')
        for p_ in (pc - d_ * 0.2, pr + d_ * 0.25):
            rod(p_ + [0, 0, -RH - 0.15], (p_[0], 0, floor_at(p_[0])), 0.03, M_BRASS, 8)
        objs = bucket(pc, 0.4, 0.4, 0.2, M_LIFT if k == 0 else M_LIFT2)
        f0 = int(D['t_catch'] * FPS)
        P = np.array([PH[f] if T[f] >= D['t_catch'] else pc for f in range(NF)])
        P = np.array([P[f] if T[f] <= D['t_rel'] else pr for f in range(NF)])
        e = follow(objs, P, pc)
        hg = trapdoor(pc, D['t_rel'], M_DARK, 0.36, 0.36); hg.parent = e; hg.matrix_parent_inverse = Matrix.Translation(-Vector(pc))
        SIGNS.append(('LIFT', pc + [0.35, -0.3, -0.5]))
    elif D['kind'] == 'wheel':
        C = np.array(D['centre']); r = D['r']; hang = D['hang']; pr = np.array(D['rel'])
        hub = link(bpy.data.objects.new('wheel', None)); hub.location = tuple(C)
        ring = obj(prim('torus', M_GOLD, 1.0, 0.025), 'ring'); ring.parent = hub; ring.rotation_euler = (math.pi / 2, 0, 0); ring.scale = (r, r, r)
        ring.location = (0, -0.26, 0)
        ring2 = obj(prim('torus', M_GOLD, 1.0, 0.025), 'ring'); ring2.parent = hub; ring2.rotation_euler = (math.pi / 2, 0, 0); ring2.scale = (r, r, r)
        ring2.location = (0, 0.26, 0)
        for a in range(8):
            th = a * math.pi / 4
            for yy in (-0.26, 0.26):
                sp = rod((0, yy, 0), (r * math.cos(th), yy, r * math.sin(th)), 0.015, M_GOLD, 6); sp.parent = hub; sp.matrix_parent_inverse = Matrix.Translation(-Vector(C))
                sp.matrix_world = rod_matrix(C + [0, yy, 0], C + [r * math.cos(th), yy, r * math.sin(th)], 0.015)
        rod(C + [0, -0.32, 0], C + [0, 0.45, 0], 0.04, M_STEEL, 10)
        for sx in (-1, 1): rod(C + [0, 0.45, 0], (C[0] + sx * 0.5, 0.45, floor_at(C[0])), 0.035, M_BRASS, 8)
        # wheel angle: the gondola's rim point angle; rotate the hub with it (clockwise seen from the front)
        ang = np.zeros(NF)
        for f in range(NF):
            q = PH[f] - C + [0, 0, hang]
            if D['t_catch'] <= T[f] <= D['t_rel']: ang[f] = math.atan2(q[2], q[0])
            elif T[f] < D['t_catch']: ang[f] = math.pi
            else: ang[f] = 0.0
        hub.rotation_mode = 'XYZ'; key_series(hub, 'rotation_euler', 1, -(ang - math.pi))
        # gondola (stays upright, hangs from the rim point)
        P = np.array([PH[f] if D['t_catch'] <= T[f] <= D['t_rel'] else (pc if T[f] < D['t_catch'] else pr) for f in range(NF)])
        objs = bucket(pc, 0.38, 0.38, 0.18, M_LIFT2)
        objs.append(rod(pc + [-0.19, 0, 0.0], pc + [0, 0, hang], 0.012, M_GOLD, 6)); objs.append(rod(pc + [0.19, 0, 0.0], pc + [0, 0, hang], 0.012, M_GOLD, 6))
        e = follow(objs, P, pc)
        hg = trapdoor(pc, D['t_rel'], M_DARK, 0.34, 0.34); hg.parent = e; hg.matrix_parent_inverse = Matrix.Translation(-Vector(pc))
        SIGNS.append(('FERRIS WHEEL', C + [0, -0.4, -r - 0.55]))
    elif D['kind'] == 'ramp':
        A, Bp, En = np.array(D['A']), np.array(D['Bp']), np.array(D['end'])
        al = math.radians(D['alpha']); nrm = np.array([math.sin(al), 0, math.cos(al)])
        for p0_, p1_ in ((A - [0.2, 0, -0.04], Bp), (Bp, En)):
            s0 = p0_ - nrm * (RH + 0.03); s1 = p1_ - nrm * (RH + 0.03)
            pl_ = rod(s0, s1, 1.0, M_WOODL, 4); pl_.matrix_world = Matrix.Translation(Vector((s0 + s1) / 2)) @ Matrix.Rotation(al, 4, 'Y') @ Matrix.Diagonal((np.linalg.norm(s1 - s0), 0.36, 0.04, 1))
            pl_.data = prim('cube', M_WOODL)
            for dy in (-0.19, 0.19): rod(s0 + [0, dy, 0.09], s1 + [0, dy, 0.09], 0.018, M_GOLD, 6)
            rod(s1 - [0, 0, 0.03], (s1[0], 0, floor_at(s1[0])), 0.03, M_BRASS, 8)
        rod(A - [0.25, 0, RH + 0.05], (A[0] - 0.25, 0, floor_at(A[0])), 0.03, M_BRASS, 8)
        box(tuple(A - [RH + 0.04, 0, -0.02]), (0.04, 0.36, 0.3), M_WOODL)
        for gx, tg in ((A, D['tA']), (Bp, D['tB'])):                    # gates: flaps in front of the ball, swing up when open
            hg = link(bpy.data.objects.new('gate', None)); gp = gx + np.array([math.cos(al), 0, -math.sin(al)]) * (RH + 0.02) + nrm * (0.25)
            hg.location = tuple(gp)
            fl_ = box((0, 0, -0.14), (0.04, 0.34, 0.26), mat('gatec', (0.9, 0.75, 0.2), 0.3, **{'Metallic': 0.5})); fl_.parent = hg
            ang = np.array([0.0 if T[f] < tg else -min(1.0, (T[f] - tg) / 0.15) * 1.4 for f in range(NF)])
            hg.rotation_mode = 'XYZ'; key_series(hg, 'rotation_euler', 1, ang)
            box(tuple(gp + [0, 0, 0.03]), (0.06, 0.42, 0.05), M_DARK)
        SIGNS.append(('GATES', A + [0.7, -0.35, -0.75]))
    elif D['kind'] == 'seesaw':
        piv = np.array(D['pivot']); ARM = D['arm']; th0 = math.radians(D['th0'])
        sw = link(bpy.data.objects.new('seesaw', None)); sw.location = tuple(piv)
        plank = box((0, 0, 0), (2 * ARM + 0.36, 0.3, 0.06), M_WOOD, name='plank'); plank.parent = sw
        cup_ = bucket(np.array([ARM, 0, RH + 0.05]), 0.34, 0.32, 0.14, M_LIFT)
        for o in cup_: o.parent = sw
        tdoor = trapdoor(np.array([ARM, 0, RH + 0.05]), D['t_rel'], M_DARK, 0.3, 0.3); tdoor.parent = sw
        bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=0.25, radius2=0.03, depth=0.35, location=(piv[0], piv[1], piv[2] - 0.2)); bpy.context.object.data.materials.append(M_STEEL)
        rod((piv[0], 0, piv[2] - 0.37), (piv[0], 0, floor_at(piv[0])), 0.04, M_BRASS, 8)
        ang = np.zeros(NF)
        for f in range(NF):
            u = (T[f] - D['t_tip']) / (D['t_tipped'] - D['t_tip']); u = min(1, max(0, u)); u = u * u * (3 - 2 * u)
            ang[f] = th0 - 2 * th0 * u
        sw.rotation_mode = 'XYZ'; key_series(sw, 'rotation_euler', 1, -ang)
        # chain reaction: the left end's push rod lifts the gate of the hearth chute (Yule ball) when it comes up
        yl = EV['yule']; ys = np.array(yl['start'])
        gateY = link(bpy.data.objects.new('ygate', None)); gx_ = ys[0] + RR['bass'] + 0.03; gateY.location = (gx_, Y_BASS, ys[2] + 0.32)
        gfl = box((0, 0, -0.2), (0.05, 0.42, 0.34), mat('ygatec', (0.85, 0.1, 0.08), 0.3)); gfl.parent = gateY
        gang = np.array([0.0 if T[f] < yl['t_release'] - 0.06 else -min(1.0, (T[f] - yl['t_release'] + 0.06) / 0.12) * 1.3 for f in range(NF)])
        gateY.rotation_mode = 'XYZ'; key_series(gateY, 'rotation_euler', 1, gang)
        # the link: from the see-saw's left end to the gate arm, a bar that slides up with the end
        lk = link(bpy.data.objects.new('link', None))
        lbar = rod((0, 0, 0), (0, 0, 1), 0.02, M_STEEL, 6)
        LP = np.zeros((NF, 3))
        for f in range(NF):
            a_ = ang[f]; LP[f] = piv + np.array([-ARM * math.cos(a_), 0, -ARM * math.sin(a_)])
        for f in range(NF):
            if f % 2 and f not in (NF - 1,): continue
            p0_ = LP[f] + [0, -0.18, 0]; p1_ = np.array([gx_ + 0.02, Y_BASS + 0.22, ys[2] + 0.32]) + [0, 0, 0.0]
            mw = rod_matrix(p0_, (p0_[0], Y_BASS + 0.22, p0_[2]), 0.02)
            lbar.matrix_world = mw
            lbar.keyframe_insert('location', frame=f); lbar.keyframe_insert('rotation_euler', frame=f); lbar.keyframe_insert('scale', frame=f)
        # gate arm the bar pushes (from the bar end to the gate hinge)
        rod((LP[0][0], Y_BASS + 0.22, ys[2] + 0.32), (gx_, Y_BASS + 0.22, ys[2] + 0.32), 0.018, M_STEEL, 6)
        SIGNS.append(('SEE-SAW', piv + [0, -0.35, -0.75]))
    elif D['kind'] == 'train':
        pr = np.array(D['rel'])
        zt_ = pc[2] - RH - 0.25
        for dy in (-0.14, 0.14): rod((pc[0] - 1.0, dy, zt_), (pr[0] + 1.2, dy, zt_), 0.018, M_STEEL, 6)
        for xx in np.arange(pc[0] - 1.0, pr[0] + 1.2, 0.22): box((xx, 0, zt_ - 0.03), (0.06, 0.42, 0.04), M_WOOD)
        for xx in (pc[0] - 0.9, (pc[0] + pr[0]) / 2, pr[0] + 1.1): rod((xx, 0, zt_ - 0.05), (xx, 0, floor_at(xx)), 0.035, M_BRASS, 8)
        P = np.array([PH[f] if D['t_catch'] <= T[f] <= D['t_rel'] else (pc if T[f] < D['t_catch'] else pr) for f in range(NF)])
        objs = bucket(pc, 0.44, 0.38, 0.16, mat('wagon', (0.75, 0.55, 0.1), 0.3, **{'Metallic': 0.4}))
        # locomotive in front of the wagon
        lx = pc[0] + 0.62
        objs.append(box((lx, 0, zt_ + 0.2), (0.5, 0.36, 0.24), M_TRAIN))
        objs.append(box((lx - 0.12, 0, zt_ + 0.42), (0.24, 0.34, 0.22), M_TRAIN))
        objs.append(box((lx - 0.12, 0, zt_ + 0.55), (0.3, 0.4, 0.04), M_RED))
        objs.append(rod((lx + 0.12, 0, zt_ + 0.32), (lx + 0.12, 0, zt_ + 0.56), 0.05, M_DARK, 10))
        objs.append(box((lx + 0.27, 0, zt_ + 0.12), (0.05, 0.3, 0.12), M_RED))
        for wx in (pc[0] - 0.14, pc[0] + 0.14, lx - 0.15, lx + 0.15):
            for dy in (-0.19, 0.19):
                wh = rod((wx, dy - 0.02, zt_ + 0.07), (wx, dy + 0.02, zt_ + 0.07), 0.07, M_RED, 14); objs.append(wh)
        e = follow(objs, P, pc)
        hg = trapdoor(pc, D['t_rel'], M_DARK, 0.4, 0.34); hg.parent = e; hg.matrix_parent_inverse = Matrix.Translation(-Vector(pc))
        # smoke puffs while it runs
        for q in range(7):
            pf = sphere((0, 0, 0), 0.08, M_FX, (8, 6)); pf.color = (0.95, 0.95, 1.0, 0)
            t0 = D['m0'] + q * 0.4; LPf = np.zeros((NF, 3)); A_ = np.zeros(NF); S_ = np.full(NF, 0.001)
            for f in range(NF):
                u = (T[f] - t0) / 1.0
                base = P[min(NF - 1, int(t0 * FPS))] + [0.74, 0, -RH + 0.33]
                LPf[f] = base + [-0.3 * max(0, u), 0, 0.5 * max(0, u)]
                if 0 <= u <= 1: A_[f] = 0.5 * (1 - u); S_[f] = 0.5 + 1.2 * u
            key_loc(pf, LPf); key_series(pf, 'color', 3, A_, 1e-3)
            for c_ in range(3): key_series(pf, 'scale', c_, np.where(S_ > 0.01, S_ * 0.11, 0.001))
        SIGNS.append(('TOY TRAIN', pc + [1.1, -0.35, -0.75]))
    elif D['kind'] == 'gift':
        gift((pc[0], 0, pc[2] - RH - 0.04), (0.6, 0.5, 0.36), mat('gift2', (0.7, 0.05, 0.08), 0.35, **{'Coat Weight': 0.5}), M_RIB,
             0.0, t_close=D['t_catch'] + 0.6, t_reopen=FINAL)
        fz = floor_at(pc[0])
        box((pc[0], 0, (pc[2] - RH - 0.06 + fz) / 2), (0.7, 0.6, pc[2] - RH - 0.06 - fz), M_WOODL, name='giftstand')
for s_, p_ in SIGNS:
    fit_w(text(s_, tuple(p_), 0.12, M_TXT_G), 1.2)

# ---------------- the toy fireplace + hearth chute for the Yule-log ball (front) ----------------
yl = EV['yule']; ys = np.array(yl['start']); ye = np.array(yl['end'])
al = math.radians(yl['alpha']); nrm = np.array([math.sin(al), 0, math.cos(al)])
s0 = ys - nrm * (RR['bass'] + 0.03) - [0.3, 0, 0]; s1 = ye - nrm * (RR['bass'] + 0.03)
pl_ = obj(prim('cube', M_WOODL), 'chute')
pl_.matrix_world = Matrix.Translation(Vector((s0 + s1) / 2)) @ Matrix.Rotation(al, 4, 'Y') @ Matrix.Diagonal((np.linalg.norm(s1 - s0), 0.46, 0.05, 1))
for dy in (-0.24, 0.24): rod(s0 + [0, dy, 0.1], s1 + [0, dy, 0.1], 0.02, M_GOLD, 6)
fz = floor_at(ys[0])
fpx = ys[0] - 0.1
box((fpx, Y_BASS, (s0[2] + fz) / 2 - 0.05), (1.3, 0.6, s0[2] - fz - 0.1), mat('brick2', (0.45, 0.13, 0.08), 0.85), name='toyfire')
box((fpx, Y_BASS - 0.3, (s0[2] + fz) / 2 - 0.25), (0.7, 0.04, 0.6), M_DARK)
box((fpx, Y_BASS - 0.32, (s0[2] + fz) / 2 - 0.42), (0.55, 0.03, 0.25), M_FIRE)
for dx in (-0.3, 0.0, 0.3):
    box((fpx + dx, Y_BASS - 0.36, s0[2] - 0.35), (0.14, 0.04, 0.3), M_RED); box((fpx + dx, Y_BASS - 0.37, s0[2] - 0.2), (0.16, 0.05, 0.06), M_WHITE)
for xx in np.arange(s0[0] + 0.6, s1[0], 0.6): rod((xx, Y_BASS, np.interp(xx, [s0[0], s1[0]], [s0[2], s1[2]]) - 0.03), (xx, Y_BASS, floor_at(xx)), 0.03, M_BRASS, 8)
fit_w(text('YULE LOG BALL', (fpx, Y_BASS - 0.36, s0[2] - 0.62), 0.13, M_TXT_G), 1.2)
fl = bpy.data.lights.new('toyfire', 'POINT'); fl.energy = 25.0; fl.color = (1.0, 0.5, 0.2)
fo = link(bpy.data.objects.new('toyfire', fl)); fo.location = (fpx, Y_BASS - 0.8, (s0[2] + fz) / 2)

# ---------------- hit rings (FX pool) on keys, bells and bass bars ----------------
HITS = [(h['t'], np.array(h['p']) - BALLS[n]['r'] * np.array(h['n']), n) for n, b in BALLS.items() for h in b['hits']
        if h['kind'] in ('key', 'bell', 'bass', 'trigger', 'bassfinal', 'star')]
HITS.sort(key=lambda x: x[0])
NRING = 12
ring_loc = np.zeros((NRING, NF, 3)); ring_scl = np.full((NRING, NF), 1e-3); ring_a = np.zeros((NRING, NF))
ring_c = [(1, 1, 1)] * NRING
for k, (t, c, n) in enumerate(HITS):
    r = k % NRING; f0 = int(round(t * FPS))
    big = 2.2 if n in ('star', 'yule') else (1.0 if n == 'holly' else 0.65)
    for df in range(12):
        f = f0 + df
        if f >= NF: break
        u = df / 11; ring_loc[r, f] = c + np.array([0, -0.25, 0.02]); ring_scl[r, f] = big * (0.25 + 1.0 * (1 - (1 - u) ** 2))
        ring_a[r, f] = (1 - u) ** 1.6
    ring_c[r] = (1, 1, 1)
for r in range(NRING):
    o = obj(prim('torus', M_FX, 0.2, 0.012), 'ring'); o.rotation_euler = (math.pi / 2, 0, 0); o.color = (1.0, 0.85, 0.55, 0)
    for f in range(1, NF):
        if ring_a[r, f] == 0: ring_loc[r, f] = ring_loc[r, f - 1]
    key_loc(o, ring_loc[r])
    for k in range(3): key_series(o, 'scale', k, ring_scl[r])
    key_series(o, 'color', 3, ring_a[r], 1e-3)
# sparkles where each ornament lands in its cup
NSP = 10
for q in range(NSP):
    o = obj(prim('star', M_FX, 0.02), 'spark'); o.color = (1.0, 0.9, 0.6, 0)
    Ls = np.zeros((NF, 3)); A_ = np.zeros(NF); S_ = np.full(NF, 1e-3)
    for j, o_ in enumerate(EV['orn']):
        if j % NSP != q: continue
        f0 = int(round(o_['t_cup'] * FPS))
        for df in range(10):
            if f0 + df < NF: Ls[f0 + df] = np.array(o_['cup']) + [0, -0.15, 0]; A_[f0 + df] = (1 - df / 10) ** 1.3; S_[f0 + df] = 0.12 + 0.12 * df / 10
    for f in range(1, NF):
        if A_[f] == 0: Ls[f] = Ls[f - 1]
    key_loc(o, Ls); key_series(o, 'color', 3, A_, 1e-3)
    for c_ in range(3): key_series(o, 'scale', c_, S_)

# ---------------- signs on the back wall (search keywords; English first, Japanese below) ----------------
def wall_sign(lines, x, z, w, h, y=YW - 0.25):
    box((x, y + 0.04, z), (w, 0.05, h), M_PLATE, name='plate')
    for sg in (1, -1): box((x, y, z + sg * h / 2), (w + 0.08, 0.07, 0.06), M_GOLD)
    for sg in (1, -1): box((x + sg * w / 2, y, z), (0.06, 0.07, h + 0.08), M_GOLD)
    zz = z + h / 2 - 0.08
    for s_, size, mt in lines:
        zz -= size * 1.2
        fit_w(text(s_, (x, y - 0.02, zz + size * 0.5), size, mt), w - 0.3)
def hx(t): return float(np.interp(t, T, PH[:, 0]))
def hz(t): return float(np.interp(t, T, PH[:, 2]))
wall_sign([('DECK THE HALLS', 0.62, M_TXT_G), ('Traditional Welsh Christmas Carol', 0.3, M_TXT), ('"Nos Galan" · Wales 1794', 0.26, M_TXT),
           ('ひいらぎかざろう（ウェールズ民謡）', 0.3, M_TXT)], hx(1.5) + 0.9, fz_cam(1.5) + 1.35, 5.6, 2.6)
wall_sign([('Christmas Marble Machine', 0.4, M_TXT_G), ('Satisfying marble music · ASMR', 0.3, M_TXT), ('クリスマス ビー玉からくり', 0.24, M_TXT)],
          hx(12.0) + 1.0, fz_cam(12.0) + 1.75, 5.2, 1.7)
wall_sign([('Every "Fa-la-la" = 9 ornaments!', 0.36, M_TXT_G), ('Classic Christmas song · Carol', 0.28, M_TXT), ('「ファララララ」で飾りが9個', 0.24, M_TXT)],
          hx(21.0) + 1.0, fz_cam(21.0) + 1.75, 5.4, 1.7)
wall_sign([('Verse 2: the Yule ball joins!', 0.36, M_TXT_G), ('2番：ユールの玉がベースで参加', 0.24, M_TXT)],
          hx(33.5) + 1.2, fz_cam(33.5) + 1.9, 5.0, 1.2)

# ---------------- counter (ORNAMENTS n) + question, riding with the camera at the top ----------------
hud = link(bpy.data.objects.new('hud', None)); hud.parent = cam
HZ = 6.0; hh_ = HZ * 24 / LENS; WW = hh_ * 9 / 16
hud.location = (0, hh_ * 0.425, -HZ)
pl_ = box((0, 0, -0.02), (WW * 0.86, hh_ * 0.09, 0.02), emis('hudpanel', (0.015, 0.05, 0.025), 1.0), name='hudplate'); pl_.parent = hud
for sgn in (1, -1):
    gb_ = box((0, sgn * hh_ * 0.045, -0.01), (WW * 0.88, hh_ * 0.007, 0.02), M_GOLD, name='hudgold'); gb_.parent = hud
ts_h = hh_ * 0.03
t1_ = text('ORNAMENTS', (-WW * 0.06, ts_h * 0.75, 0), ts_h * 1.25, M_TXT, rot=(0, 0, 0), align='RIGHT'); t1_.parent = hud
t2_ = text('How many will the trees get?', (0, -ts_h * 0.3, 0), ts_h * 0.95, M_TXT_G, rot=(0, 0, 0)); t2_.parent = hud
t3_ = text('飾りはいくつ？', (0, -ts_h * 0.95, 0), ts_h * 0.5, M_TXT, rot=(0, 0, 0)); t3_.parent = hud
fit_w(t2_, WW * 0.82)
cups_t = sorted(o_['t_cup'] for o_ in EV['orn'])
CV = np.array([int(np.searchsorted(cups_t, t, 'right')) for t in T])
DW = ts_h * 1.15; XD0 = WW * 0.02
for slot in range(2):
    for dgt in range(10):
        o = text(str(dgt), (XD0 + slot * DW, ts_h * 0.75, 0), ts_h * 1.9, M_TXT_G, rot=(0, 0, 0)); o.parent = hud
        vis = np.array([1.0 if f'{int(CV[f]):>2d}'[slot] == str(dgt) else 0.0 for f in range(NF)])
        if not vis.any(): o.hide_render = True; continue
        for k in range(3): key_series(o, 'scale', k, np.maximum(0.001, vis))
star_lbl = text('+ ★', (XD0 + 2 * DW + 0.1, ts_h * 0.75, 0), ts_h * 1.5, M_TXT_G, rot=(0, 0, 0), align='LEFT'); star_lbl.parent = hud
sv = np.where(T >= FINAL, 1.0, 0.001)
for k in range(3): key_series(star_lbl, 'scale', k, sv)
bump = np.ones(NF)
for f in range(1, NF):
    if CV[f] != CV[f - 1]:
        for df in range(6):
            if f + df < NF: bump[f + df] = max(bump[f + df], 1.0 + 0.05 * (1 - df / 6))
for k in range(2): key_series(hud, 'scale', k, bump, 1e-3)
for o in bpy.data.objects:
    if o.parent is hud: o.visible_shadow = False

# ---------------- ending: MERRY CHRISTMAS ----------------
endo = []
endo += [text('MERRY CHRISTMAS!', (tr7['x'], 0.9, top[2] - 1.3), 0.55, M_TXT_G)]
endo += [text('72 ornaments + 1 star', (tr7['x'], 0.9, top[2] - 1.85), 0.3, M_TXT)]
endo += [text('メリークリスマス', (tr7['x'], 0.9, top[2] - 2.25), 0.24, M_TXT)]
for o_ in endo:
    fit_w(o_, 5.0); k_ = o_.scale[0]
    o_.location.x = tr7['x'] - 0.4 - 2.6 if False else o_.location.x
    o_.scale = (0.001,) * 3; o_.keyframe_insert('scale', frame=int((FINAL + 0.5) * FPS))
    o_.scale = (k_,) * 3; o_.keyframe_insert('scale', frame=int((FINAL + 0.5) * FPS) + 8)
# confetti from the gift box at the end
CONF = 70; gpos = np.array(DEV[7]['catch'])
for k in range(CONF):
    cf = obj(prim('plane', M_OBJ), 'conf'); cf.color = (*ORN_COLS[k % 9], 1)
    a0 = rng.uniform(-0.6, 0.6); sp = rng.uniform(3.0, 5.5); vy = rng.uniform(-0.8, 0.8)
    P = np.zeros((NF, 3)); RT = np.zeros((NF, 3))
    for f in range(NF):
        s_ = T[f] - FINAL
        if s_ < 0: P[f] = gpos; continue
        vx_, vz_ = sp * math.sin(a0), sp * math.cos(a0)
        tt = min(s_, 0.5); drag = 1 - math.exp(-s_ / 0.35)
        P[f] = gpos + np.array([vx_ * 0.35 * drag + 0.1 * math.sin(3 * s_ + k), vy * drag, vz_ * 0.35 * drag - 0.55 * max(0, s_ - 0.3)])
        RT[f] = (s_ * (3 + k % 5), s_ * (2 + k % 3), s_ * 1.5)
    key_loc(cf, P)
    sc_ = np.where(T >= FINAL, 1.0, 0.001)
    for c_ in range(3): key_series(cf, 'scale', c_, sc_ * 0.07); key_series(cf, 'rotation_euler', c_, RT[:, c_])

# ---------------- lights ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bgn = w.node_tree.nodes['Background']; bgn.inputs['Color'].default_value = (0.03, 0.025, 0.04, 1); bgn.inputs['Strength'].default_value = 1.0
key_ = bpy.data.lights.new('key', 'AREA'); key_.size = 6.0; key_.color = (1.0, 0.82, 0.6); key_.energy = 900.0
ko = link(bpy.data.objects.new('key', key_))
fill = bpy.data.lights.new('fill', 'AREA'); fill.size = 8.0; fill.color = (0.55, 0.65, 1.0); fill.energy = 250.0
fo_ = link(bpy.data.objects.new('fill', fill))
for f in range(0, NF, 5):
    ko.location = (CP[f, 0] - 2.5, -4.5, FOC[f, 2] + 5.0); ko.rotation_euler = (math.radians(45), 0, math.radians(-25))
    ko.keyframe_insert('location', frame=f); ko.keyframe_insert('rotation_euler', frame=f)
    fo_.location = (CP[f, 0] + 4.0, -5.0, FOC[f, 2] + 1.0); fo_.rotation_euler = (math.radians(75), 0, math.radians(35))
    fo_.keyframe_insert('location', frame=f); fo_.keyframe_insert('rotation_euler', frame=f)
# warm glow of each tree once it is lit
for tr in TREES:
    t_on = max(o_['t_cup'] for o_ in EV['orn'] if o_['tree'] == tr['k'])
    gl = bpy.data.lights.new('treeglow', 'POINT'); gl.color = (1.0, 0.75, 0.4); gl.shadow_soft_size = 1.0
    go = link(bpy.data.objects.new('treeglow', gl)); go.location = (tr['x'], YT - 1.6, tr['zb'] + HT * tr['s'] * 0.45)
    for t_, e_ in ((0, 0.0), (t_on - 0.05, 0.0), (t_on + 0.3, 120.0), (FINAL - 0.05, 120.0), (FINAL + 0.2, 260.0), (FINAL + 1.5, 160.0)):
        gl.energy = e_; gl.keyframe_insert('energy', frame=max(0, int(t_ * FPS)))
sg = bpy.data.lights.new('starglow', 'POINT'); sg.color = (1.0, 0.8, 0.4)
sgo = link(bpy.data.objects.new('starglow', sg)); sgo.location = (top[0], top[1] - 0.6, top[2])
for t_, e_ in ((0, 0.0), (FINAL - 0.02, 0.0), (FINAL + 0.1, 300.0), (FINAL + 1.0, 120.0)):
    sg.energy = e_; sg.keyframe_insert('energy', frame=int(t_ * FPS))

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
        print('t', tt_, 'cam', [round(v, 2) for v in cam.matrix_world.translation], 'WID', round(WID[f], 2), 'FOC', np.round(FOC[f], 2))
print('scene objects', len(bpy.data.objects))
if MODE == 'ray':
    from bpy_extras.view3d_utils import region_2d_to_vector_3d
    f = int(round(float(args[1]) * FPS)); sc.frame_set(f); dg = bpy.context.evaluated_depsgraph_get()
    cmw = cam.matrix_world; fr = cam.data.view_frame(scene=sc)
    for v in np.linspace(0.05, 0.95, 19):
        u = 0.5; tr_, br, bl_, tl = [cmw @ p for p in fr]
        pt = bl_ + (br - bl_) * u + (tl - bl_) * (1 - v)
        d = (pt - cmw.translation).normalized()
        hit, loc, nrm_, idx, ob, _ = sc.ray_cast(dg, cmw.translation, d)
        print('v', round(v, 2), ob.name if hit else None, [round(c, 2) for c in loc] if hit else '')
