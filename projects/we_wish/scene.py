"""We Wish You a Merry Christmas - 'Can ONE caroler light up the whole village?'  Side view of a snowy village street, Christmas Eve.
Every melody note = a ball landing on a key / bell of that pitch: Carol (red marble in a Santa hat) plays the verse phrases on
stepping-stone keys, the last note of each phrase rings a doorbell and that house lights up (12 houses); 22 snowballs per refrain drop
off a canopy onto bells of the bakery / church / town hall; the figgy-pudding ball plays the bass in verses 2-3; Carol lands on the
great bell on top of the town-square tree on the final 'Year'. Then the camera pulls back: the village sits on a giant Christmas pudding.
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
FPS = 30; FINAL = E['final']; NF = int((FINAL + 5.2) * FPS); T = np.arange(NF) / FPS
G = LAY['G']; EV = LAY['ev']; BALLS = LAY['balls']; RR = LAY['R']
Y_SNOW, Y_BASS, YH = LAY['y_snow'], LAY['y_bass'], LAY['yh']
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


def smooth01_(u): u = min(1.0, max(0.0, u)); return u * u * (3 - 2 * u)


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
PC = POS['carol']; PP = POS['pudding']
DEV = EV['dev']; BLD = EV['bld']; HOUSES = EV['houses']
RC, RS, RP = RR['carol'], RR['snow'], RR['bass']
T_V2 = E['bass'][0][0]
GREAT = np.array(EV['great']['p']); DRUM = np.array(EV['drum']['p'])
SNOWS = sorted([n for n in BALLS if BALLS[n]['kind'] == 'snow'], key=lambda n: int(n[4:]))

# ---------------- materials ----------------
M_CAROL = mat('carol', (0.85, 0.02, 0.03), 0.08, **{'Coat Weight': 1.0})
M_HATW = mat('hatw', (0.95, 0.95, 0.95), 0.8, **{'Sheen Weight': 1.0})
M_PUD = mat('pud', (0.16, 0.06, 0.03), 0.55)
nt = M_PUD.node_tree; bsdf = nt.nodes['Principled BSDF']
nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 18.0; nz.inputs['Detail'].default_value = 4.0
cr = nt.nodes.new('ShaderNodeValToRGB'); cr.color_ramp.elements[0].position = 0.55; cr.color_ramp.elements[0].color = (0.17, 0.065, 0.03, 1)
cr.color_ramp.elements[1].position = 0.62; cr.color_ramp.elements[1].color = (0.05, 0.015, 0.02, 1)
nt.links.new(nz.outputs['Fac'], cr.inputs['Fac']); nt.links.new(cr.outputs['Color'], bsdf.inputs['Base Color'])
bmp = nt.nodes.new('ShaderNodeBump'); bmp.inputs['Strength'].default_value = 0.4; nt.links.new(nz.outputs['Fac'], bmp.inputs['Height'])
nt.links.new(bmp.outputs['Normal'], bsdf.inputs['Normal'])
M_ICING = mat('icing', (0.97, 0.96, 0.92), 0.3, **{'Coat Weight': 0.6})
M_LEAF = mat('leaf', (0.03, 0.33, 0.08), 0.3, **{'Coat Weight': 0.6})
M_BERRY = mat('berry', (0.9, 0.0, 0.02), 0.1, **{'Coat Weight': 1.0})
M_SNOWB = mat('snowball', (0.92, 0.95, 1.0), 0.55)
nt = M_SNOWB.node_tree; bsdf = nt.nodes['Principled BSDF']
nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 40.0; nz.inputs['Detail'].default_value = 6.0
bmp = nt.nodes.new('ShaderNodeBump'); bmp.inputs['Strength'].default_value = 0.5; nt.links.new(nz.outputs['Fac'], bmp.inputs['Height'])
nt.links.new(bmp.outputs['Normal'], bsdf.inputs['Normal'])
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
M_SNOW = mat('snow', (0.82, 0.86, 0.95), 0.6)
nt = M_SNOW.node_tree; bsdf = nt.nodes['Principled BSDF']
nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 6.0; nz.inputs['Detail'].default_value = 5.0
bmp = nt.nodes.new('ShaderNodeBump'); bmp.inputs['Strength'].default_value = 0.25; nt.links.new(nz.outputs['Fac'], bmp.inputs['Height'])
nt.links.new(bmp.outputs['Normal'], bsdf.inputs['Normal'])
def brickmat(name, c1, c2, mortar, scale=4.0):
    m = mat(name, c1, 0.8); nt_ = m.node_tree; b_ = nt_.nodes['Principled BSDF']
    bt_ = nt_.nodes.new('ShaderNodeTexBrick'); bt_.inputs['Color1'].default_value = (*c1, 1); bt_.inputs['Color2'].default_value = (*c2, 1)
    bt_.inputs['Mortar'].default_value = (*mortar, 1); bt_.inputs['Scale'].default_value = scale; bt_.inputs['Mortar Size'].default_value = 0.025
    tc_ = nt_.nodes.new('ShaderNodeTexCoord'); mp_ = nt_.nodes.new('ShaderNodeMapping'); nt_.links.new(tc_.outputs['Object'], mp_.inputs[0])
    sp_ = nt_.nodes.new('ShaderNodeCombineXYZ'); sx_ = nt_.nodes.new('ShaderNodeSeparateXYZ'); gm_ = nt_.nodes.new('ShaderNodeNewGeometry')
    nt_.links.new(gm_.outputs['Position'], sx_.inputs[0])
    ad_ = nt_.nodes.new('ShaderNodeMath'); ad_.operation = 'ADD'; nt_.links.new(sx_.outputs['X'], ad_.inputs[0]); nt_.links.new(sx_.outputs['Y'], ad_.inputs[1])
    nt_.links.new(ad_.outputs[0], sp_.inputs[0]); nt_.links.new(sx_.outputs['Z'], sp_.inputs[1])
    nt_.links.new(sp_.outputs[0], bt_.inputs['Vector'])
    nt_.links.new(bt_.outputs['Color'], b_.inputs['Base Color'])
    bp_ = nt_.nodes.new('ShaderNodeBump'); bp_.inputs['Strength'].default_value = 0.3; nt_.links.new(bt_.outputs['Fac'], bp_.inputs['Height']); nt_.links.new(bp_.outputs['Normal'], b_.inputs['Normal'])
    return m
M_STONE = brickmat('stone', (0.33, 0.31, 0.33), (0.25, 0.24, 0.27), (0.6, 0.62, 0.68), 2.5)
M_KEY = glowmat('key', gain=7.0, base=0.0, rough=0.25, coat=0.6)
M_BELL = glowmat('bell', gain=6.0, base=0.15, rough=0.22, metal=0.8)
M_BASSBAR = glowmat('bassbar', gain=14.0, base=0.0, rough=0.35, coat=0.7)
M_OBJ = objcol('objcol', 0.45)
M_FX = fxmat('fx', 8.0)
M_TXT = emis('txt', (1.0, 0.97, 0.9), 2.2)
M_TXT_G = emis('txtg', (1.0, 0.75, 0.25), 3.0)
M_TXT_R = emis('txtr', (1.0, 0.3, 0.25), 2.6)
M_PLATE = mat('plate', (0.35, 0.03, 0.04), 0.45, **{'Coat Weight': 0.5})


def pitch_col(p, sat=1.0):
    h = ((p - 55) % 12) / 12.0
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
BC = make_ball('carol', RC, M_CAROL)
# Santa hat: follows Carol's position (does not spin with her)
hatf = link(bpy.data.objects.new('hatf', None)); key_loc(hatf, PC)
hb = obj(prim('torus', M_HATW, 0.105, 0.035), 'hatbrim'); hb.location = (0, 0, RC * 0.78); hb.parent = hatf
hc = obj(prim('cone', M_CAROL, 20, 0.11, 0.015, 0.26), 'hatcone'); hc.location = (0.035, 0, RC * 0.78 + 0.12); hc.rotation_euler = (0, 0.35, 0); hc.parent = hatf
pp_ = sphere((0.085, 0, RC * 0.78 + 0.24), 0.04, M_HATW, (10, 6)); pp_.parent = hatf
BP = make_ball('pudding', RP, M_PUD)
ic = sphere((0, 0, RP * 0.35), 1.0, M_ICING, (24, 12)); ic.scale = (RP * 0.97, RP * 0.97, RP * 0.7); ic.parent = BP
for sgn in (1, -1):
    lf = sphere((sgn * 0.07, 0, RP * 1.0), 1.0, M_LEAF, (12, 6)); lf.scale = (0.085, 0.035, 0.025); lf.rotation_euler = (0.2, 0, sgn * 0.45); lf.parent = BP
for dx in (-0.025, 0.025): b_ = sphere((dx, -0.03, RP * 1.05), 0.026, M_BERRY, (10, 6)); b_.parent = BP
SNOWO = {}
for n in SNOWS:
    o = obj(prim('sph', M_SNOWB, 20, 10), n); o.scale = (RS,) * 3; key_loc(o, POS[n])
    o.rotation_euler = (rng.uniform(0, 6), rng.uniform(0, 6), 0); SNOWO[n] = o

# ---------------- camera ----------------
FOC = np.zeros((NF, 3)); WID = np.zeros(NF)
REFT = [(EV['snow'][0]['t_hit'], 0)]
REFW = []
for k in range(3):
    ss = [s for s in EV['snow'] if s['bld'] == k]
    REFW.append((ss[0]['t_hit'], ss[-1]['t_hit'], np.array([s['t_hit'] for s in ss]), np.array([BLD[k]['bells'][j]['x'] for j in range(len(ss))])))
T_WIDE = FINAL - 2.6; T_PULL = FINAL + 0.7; T_PULL1 = FINAL + 3.6
PUD_C = None   # set below (giant pudding)
for f in range(NF):
    t = T[f]; h = PC[f]
    wt = 0.0; xt = h[0]; ztop = h[2] + 2.0
    for k, (ta, tb, tts, xbs) in enumerate(REFW):
        u = min(1.0, max(0.0, (t - ta + 0.6) / 0.9)) * min(1.0, max(0.0, (tb + 0.9 - t) / 0.9))
        if u > wt: wt = u; xt = float(np.interp(t, tts, xbs)); ztop = BLD[k]['rail'] + 0.45
    x = (1 - 0.45 * wt) * (h[0] + 0.6) + 0.45 * wt * xt
    zlow = h[2] - 0.6
    if t > T_V2 - 1.5:
        a_ = min(1, (t - T_V2 + 1.5) / 2.0)
        zlow = min(zlow, PP[f][2] - 0.85) * a_ + zlow * (1 - a_)
    zhigh = max(h[2] + 2.0, (1 - wt) * (h[2] + 2.0) + wt * ztop)
    FOC[f] = (x, 0.0, (zlow + zhigh) / 2)
    WID[f] = max(4.5, (zhigh - zlow + 1.3) * 9 / 16)
    if t < 1.6:                                      # opening: close on the carolers' gift box
        u = min(1, max(0, (t - 0.4) / 1.2)); u = u * u * (3 - 2 * u)
        f0 = np.array([EV['hut']['p'][0] + 0.7, 0, EV['hut']['p'][2] + 0.75]); FOC[f] = f0 * (1 - u) + FOC[f] * u; WID[f] = 3.6 * (1 - u) + WID[f] * u
    if t > T_WIDE:                                   # the finale: wheel, great bell, tree, drum
        u = min(1, (t - T_WIDE) / 1.8); u = u * u * (3 - 2 * u)
        wide = np.array([(GREAT[0] - 2.9 + DRUM[0] + 0.5) / 2, 0.0, (DRUM[2] - 0.9 + GREAT[2] + 2.0) / 2])
        FOC[f] = (1 - u) * FOC[f] + u * wide; WID[f] = (1 - u) * WID[f] + u * max(7.0, (GREAT[2] + 2.0 - DRUM[2] + 1.6) * 9 / 16)
FOC[:, 0] = smooth(FOC[:, 0], 0.45); FOC[:, 2] = smooth(FOC[:, 2], 0.6); WID = smooth(WID, 0.6)
FOC0 = FOC.copy(); WID0 = WID.copy()
LENS = 32.0


def fz_cam(t): return float(np.interp(t, T, FOC0[:, 2]))
def fx_cam(t): return float(np.interp(t, T, FOC0[:, 0]))


# ---------------- floor: snowy terraces that step down with the course ----------------
allx = [h['p'][0] for b in BALLS.values() for h in b['hits']]
XA, XB = -8.0, max(allx) + 9.0
STEPX = np.arange(XA, XB + 3.0, 3.0)
def floor_need(x):
    zs = [h['zb'] for h in HOUSES if abs(h['x'] - x) < 1.6]
    zs += [b['zb'] for b in BLD if abs(b['x'] - x) < b['w'] / 2 + 0.5]
    zs += [h['p'][2] - 0.5 for h in BALLS['pudding']['hits'] if abs(h['p'][0] - x) < 1.6]
    zs += [h['p'][2] - 1.0 for h in BALLS['carol']['hits'] if abs(h['p'][0] - x) < 1.6]
    zs += [PC[f][2] - 0.9 for f in range(0, NF, 3) if abs(PC[f][0] - x) < 1.6]
    ps_ = EV['pudding']['start']
    zs += [ps_[2] - 0.8] if abs(ps_[0] - x) < 2.0 else []
    if abs(GREAT[0] - x) < 2.5: zs += [GREAT[2] - RC - 0.5 - 5.0]
    if abs(DRUM[0] - x) < 1.5: zs += [DRUM[2] - RP - 0.6]
    if not zs: zs = [min(HOUSES, key=lambda hh: abs(hh['x'] - x))['zb']]
    return min(zs)
FLZ = np.array([min(floor_need(x + d) for d in (0.0, 1.5, 3.0)) for x in STEPX])
FLZ = np.minimum.accumulate(FLZ)
def floor_at(x): return float(FLZ[min(len(FLZ) - 1, max(0, int((x - XA) // 3.0)))])
Y0F, Y1F = -8.0, 7.5
# ---------------- the giant Christmas pudding under the village (seen in the final pull-back) ----------------
# a sphere below every terrace (fitted offline: radius 130, centre x = -6), the terraces reach down to it
PX0, PY0, PRAD = -6.0, 1.5, 130.0
def dome_z(x, y, zc): return zc + math.sqrt(max(0.0, PRAD ** 2 - (x - PX0) ** 2 - (y - PY0) ** 2))
PZC = min(FLZ[i] - 0.3 - dome_z(x0 + dx, y_, 0.0) for i, x0 in enumerate(STEPX[:-1]) for dx in (0.0, 3.0) for y_ in (Y0F, Y1F, 1.5))
for i, x0 in enumerate(STEPX[:-1]):
    zf = FLZ[i]; zbot = min(dome_z(x0 + dx, y_, PZC) for dx in (0.0, 3.0) for y_ in (Y0F, Y1F)) - 0.5
    box((x0 + 1.5, (Y0F + Y1F) / 2, (zf + zbot) / 2), (3.02, Y1F - Y0F, zf - zbot), M_SNOW, name='terrace')
    if i > 0 and FLZ[i - 1] > zf + 1e-3:                                       # snowy stone riser
        box((x0, (Y0F + Y1F) / 2, (zf + FLZ[i - 1]) / 2), (0.08, Y1F - Y0F, FLZ[i - 1] - zf), M_STONE, name='riser')
        box((x0 + 0.04, (Y0F + Y1F) / 2, FLZ[i - 1] + 0.02), (0.2, Y1F - Y0F, 0.06), M_SNOW, name='lip')
PUD_C = np.array([PX0, PY0, PZC])
bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=96, v_segments=48, radius=PRAD)
for v in list(bm.verts):
    if v.co.z < -2.0: bm.verts.remove(v)
me = bpy.data.meshes.new('pudding'); bm.to_mesh(me); bm.free()
for p in me.polygons: p.use_smooth = True
GP = link(bpy.data.objects.new('giantpud', me)); GP.location = tuple(PUD_C)
M_GPUD = mat('gpud', (0.16, 0.06, 0.03), 0.5)
nt = M_GPUD.node_tree; bsdf = nt.nodes['Principled BSDF']
gm = nt.nodes.new('ShaderNodeNewGeometry'); sxyz = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(gm.outputs['Position'], sxyz.inputs[0])
nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 0.35; nz.inputs['Detail'].default_value = 5.0
cr = nt.nodes.new('ShaderNodeValToRGB'); cr.color_ramp.elements[0].position = 0.55; cr.color_ramp.elements[0].color = (0.085, 0.032, 0.014, 1)
cr.color_ramp.elements[1].position = 0.6; cr.color_ramp.elements[1].color = (0.03, 0.008, 0.012, 1)
nt.links.new(nz.outputs['Fac'], cr.inputs['Fac'])
# icing: white above a wavy line (drips) at ~55 % of the dome height
wv = nt.nodes.new('ShaderNodeTexWave'); wv.wave_type = 'RINGS' if False else 'BANDS'; wv.bands_direction = 'X'
wv.inputs['Scale'].default_value = 0.035; wv.inputs['Distortion'].default_value = 2.5; wv.inputs['Detail'].default_value = 1.0
nt.links.new(gm.outputs['Position'], wv.inputs['Vector'])
lim = nt.nodes.new('ShaderNodeMath'); lim.operation = 'MULTIPLY_ADD'; lim.inputs[1].default_value = 0.12 * PRAD; lim.inputs[2].default_value = PZC + 0.5 * PRAD
nt.links.new(wv.outputs['Fac'], lim.inputs[0])
gt = nt.nodes.new('ShaderNodeMath'); gt.operation = 'GREATER_THAN'; nt.links.new(sxyz.outputs['Z'], gt.inputs[0]); nt.links.new(lim.outputs[0], gt.inputs[1])
mx = nt.nodes.new('ShaderNodeMix'); mx.data_type = 'RGBA'; nt.links.new(gt.outputs[0], mx.inputs[0]); nt.links.new(cr.outputs['Color'], mx.inputs[6])
mx.inputs[7].default_value = (0.95, 0.94, 0.9, 1); nt.links.new(mx.outputs[2], bsdf.inputs['Base Color'])
rg = nt.nodes.new('ShaderNodeMix'); rg.data_type = 'FLOAT'; nt.links.new(gt.outputs[0], rg.inputs[0]); rg.inputs[2].default_value = 0.6; rg.inputs[3].default_value = 0.25
nt.links.new(rg.outputs[0], bsdf.inputs['Roughness'])
GP.data.materials.append(M_GPUD)
# plate, tablecloth, giant candles, giant holly sprig, crackers
box((PX0, PY0, PZC - 1.6), (16 * PRAD, 16 * PRAD, 2.0), mat('cloth', (0.45, 0.03, 0.05), 0.8, **{'Sheen Weight': 0.6}), name='table')
rod((PX0, PY0, PZC - 0.6), (PX0, PY0, PZC + 0.2), PRAD * 1.32, mat('china', (0.95, 0.95, 0.93), 0.15, **{'Coat Weight': 1.0}), 96)
rod((PX0, PY0, PZC - 0.55), (PX0, PY0, PZC + 0.25), PRAD * 1.33, M_GOLD, 96).scale[2] *= 1.0
rod((PX0, PY0, PZC + 0.21), (PX0, PY0, PZC + 0.27), PRAD * 1.25, mat('china2', (0.92, 0.92, 0.9), 0.2), 96)
M_CANDLE = mat('candle', (0.85, 0.05, 0.05), 0.4)
CANDLES = [(PX0 - 2.0 * PRAD, PY0 + 1.1 * PRAD), (PX0 + 2.25 * PRAD, PY0 + 0.9 * PRAD), (PX0 + 0.6 * PRAD, PY0 + 2.6 * PRAD)]
KS = PRAD / 62.0
for cx_, cy_ in CANDLES:
    rod((cx_, cy_, PZC - 0.6), (cx_, cy_, PZC + 70 * KS), 7.0 * KS, M_CANDLE, 32)
    rod((cx_, cy_, PZC - 0.6), (cx_, cy_, PZC + 2.0 * KS), 12.0 * KS, M_GOLD, 32)
    fl_ = sphere((cx_, cy_, PZC + 76 * KS), 1.0, emis('cflame', (1.0, 0.6, 0.2), 30.0), (16, 8)); fl_.scale = (2.4 * KS, 2.4 * KS, 6.0 * KS)
    cl = bpy.data.lights.new('candle', 'POINT'); cl.color = (1.0, 0.6, 0.3); cl.shadow_soft_size = 4.0
    link(bpy.data.objects.new('candle', cl)).location = (cx_, cy_ - 8 * KS, PZC + 78 * KS)
    for t_, e_ in ((0, 0.0), (FINAL + 0.7, 0.0), (FINAL + 3.0, 2.0e6 * KS * KS)):
        cl.energy = e_; cl.keyframe_insert('energy', frame=int(t_ * FPS))
# giant holly sprig on the pudding's summit, behind the village
HS = np.array([PX0 - 0.42 * PRAD, PY0 + 6.0, dome_z(PX0 - 0.42 * PRAD, PY0 + 6.0, PZC) - 1.5 * KS])
for a_, rz in ((-0.5, 0.3), (0.6, -0.4), (2.4, 0.2)):
    lf = sphere(tuple(HS + KS * np.array([9 * math.cos(a_), 9 * math.sin(a_) * 0.4, 3.0])), 1.0, M_LEAF, (24, 10)); lf.scale = (11 * KS, 4.2 * KS, 1.6 * KS)
    lf.rotation_euler = (0.2, rz, a_)
for d_ in ((-1.6, -1.5, 5.5), (1.4, -1.8, 5.2), (0.0, -0.2, 7.2)): sphere(tuple(HS + KS * np.array(d_)), 2.3 * KS, M_BERRY, (20, 10))
# brandy flames (flambe) after the reveal
FLAMES = []
M_FLAME = fxmat('flame', 6.0)
rr_ = np.random.default_rng(3)
for q in range(64):
    az = math.radians(rr_.uniform(-175, -5)); el = math.radians(rr_.uniform(2, 50))
    nv = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    c_ = PUD_C + nv * PRAD * 0.995
    fo = obj(prim('cone', M_FLAME, 12, 1.0, 0.0, 2.0), 'flame'); fo.location = tuple(c_); fo.color = (0.25, 0.42, 1.0, 0) if q % 3 else (0.55, 0.4, 1.0, 0)
    up = 0.85 * np.array([0, 0, 1.0]) + 0.15 * nv; up /= np.linalg.norm(up)
    fo.rotation_mode = 'QUATERNION'; fo.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(Vector(up))
    h_ = rr_.uniform(0.3, 0.55) * PRAD; wd = rr_.uniform(0.045, 0.07) * PRAD
    t0 = T_PULL + 0.4 + 0.025 * q
    A_ = np.array([0.0 if T[f] < t0 else min(1, (T[f] - t0) / 0.5) * (0.75 + 0.25 * math.sin(9 * T[f] + q)) for f in range(NF)])
    key_series(fo, 'color', 3, A_ * 0.6, 1e-3)
    for c3 in range(2): key_series(fo, 'scale', c3, np.maximum(0.001, np.minimum(A_ * 2, 1.0) * wd), 1e-3)
    key_series(fo, 'scale', 2, np.maximum(0.001, A_ * h_ / 2.0 * (1 + 0.15 * np.sin(13 * T + q))), 1e-3)
    FLAMES.append(fo)

# ---------------- camera (with the final pull-back) ----------------
FOCR = np.array([PX0 + 0.1 * PRAD, 0.0, PZC + PRAD * 0.62]); WIDR = PRAD * 2.45
for f in range(NF):
    t = T[f]
    if t > T_PULL:
        u = min(1.0, (t - T_PULL) / (T_PULL1 - T_PULL)); u = u * u * u * (u * (6 * u - 15) + 10)
        lw = (1 - u) * math.log(WID0[f]) + u * math.log(WIDR)
        g_ = (math.exp(lw) - WID0[f]) / max(1e-6, WIDR - WID0[f])
        FOC[f] = (1 - g_) * FOC0[f] + g_ * FOCR; WID[f] = math.exp(lw)
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 6000
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TPc = np.zeros((NF, 3))
for f in range(NF):
    D_ = WID[f] / (9 / 16) * LENS / 24
    CP[f] = (FOC[f, 0], -D_, FOC[f, 2] + 0.16 * D_)
    TPc[f] = (FOC[f, 0], 0.6, FOC[f, 2] - 0.15)
key_loc(cam, CP); key_loc(tgt, TPc)
camd.clip_start = 0.1
CLIP = np.array([0.1 if WID[f] < 10 else 0.5 for f in range(NF)])
for f in range(NF):
    if f == 0 or f == NF - 1 or abs(CLIP[f] - CLIP[f - 1]) > 1e-6 or (f + 1 < NF and abs(CLIP[f + 1] - CLIP[f]) > 1e-6):
        camd.clip_start = float(CLIP[f]); camd.keyframe_insert('clip_start', frame=f)

# ---------------- Carol's stepping-stone keys (one per note), doorbells, the street rails ----------------
for h in BALLS['carol']['hits']:
    if h['kind'] not in ('key', 'door'): continue
    n_ = np.array(h['n']); c = np.array(h['p']) - RC * n_
    o = obj(prim('bcube', M_KEY), 'key')
    big = h['kind'] == 'door'
    o.matrix_world = Matrix.Translation(Vector(c - n_ * 0.03)) @ frame_from_n(n_).to_4x4() @ Matrix.Diagonal((0.22, 0.42, 0.06, 1))
    col = pitch_col(h['pitch']) if not big else (1.0, 0.75, 0.25)
    o.color = (*col, 0)
    flash(o, h['t'], col, n=n_)
    for dy in (-0.15, 0.15):
        rod((c[0], dy, c[2] - 0.05), (c[0], dy, c[2] - 0.32), 0.012, M_BRASS, 6)
    if big:                                            # doorbell: a little gold bell on the key's back edge + a bell-pull wire to the house
        bb = obj(prim('bell', M_GOLD, 16), 'dbell'); bb.location = (c[0], 0.3, c[2] + 0.2); bb.scale = (0.07, 0.07, 0.1)
        rod((c[0], 0.3, c[2] - 0.05), (c[0], 0.3, c[2] + 0.2), 0.01, M_BRASS, 6)
        hs = [hh for hh in HOUSES if hh['n'] == h['house']][0]
        rod((c[0], 0.3, c[2] + 0.2), (hs['x'] + 0.0, YH - 0.05, floor_at(hs['x']) + 1.75), 0.008, M_GOLDD, 4)
for k in range(3):
    hs = [h for h in BALLS['carol']['hits'] if h['kind'] in ('key', 'door') and LEAD[h['i']][7] == k]
    X = np.array([h['p'][0] for h in hs]); Z = np.array([h['p'][2] - RC - 0.32 for h in hs])
    XX = np.linspace(X[0] - 0.25, X[-1] + 0.25, 40); ZZ = np.interp(XX, X, Z)
    ZM = np.array([Z[np.abs(X - x) < 0.4].min() if (np.abs(X - x) < 0.4).any() else z for x, z in zip(XX, ZZ)])
    strip_mesh(XX, ZM, -0.22, 0.22, 0.07, M_WOOD, 'rail')
    strip_mesh(XX, ZM + 0.012, -0.235, -0.205, 0.02, M_GOLD, 'railtrim')
    strip_mesh(XX, ZM + 0.012, 0.205, 0.235, 0.02, M_GOLD, 'railtrim')
    for x_ in XX[::5]:
        rod((x_, 0, float(np.interp(x_, XX, ZM)) - 0.07), (x_, 0, floor_at(x_)), 0.03, M_DARK, 8)

# ---------------- houses: windows / door lamp / wreath light up when their doorbell rings ----------------
M_WIN = lightmat('win', (1.0, 0.62, 0.28), on=3.2, flash_=10.0, off_col=(0.03, 0.04, 0.07))
M_LAMP = lightmat('lamp', (1.0, 0.8, 0.45), on=10.0, flash_=20.0, off_col=(0.1, 0.1, 0.12))
M_WREATHL = lightmat('wreathl', (1.0, 0.3, 0.2), on=7.0, flash_=14.0, off_col=(0.2, 0.02, 0.02))
WALLS = [(0.75, 0.62, 0.48), (0.55, 0.15, 0.12), (0.85, 0.8, 0.68), (0.32, 0.42, 0.55), (0.42, 0.5, 0.35), (0.7, 0.45, 0.3)]
win_q = []; win_t = []; lamp_q = []; lamp_t = []
for j, hs in enumerate(HOUSES):
    x = hs['x']; zb = hs['zb']; t_on = hs['t_on']
    w, d, hw = 2.0, 1.9, 1.75
    fz_ = floor_at(x)
    if zb - fz_ > 0.02:
        box((x, YH + d / 2 - 0.2, (zb + fz_) / 2), (w + 0.5, d + 0.9, zb - fz_), M_STONE, name='plinth')
        box((x, YH + d / 2 - 0.2, zb - 0.02), (w + 0.56, d + 0.96, 0.08), M_SNOW)
    wm = mat(f'hwall{j}', WALLS[j % 6], 0.75)
    box((x, YH + d / 2, zb + hw / 2), (w, d, hw), wm, name='house')
    box((x, YH - 0.02, zb + 0.06), (w + 0.06, 0.06, 0.12), M_DARK)
    # gable roof (ridge along y) with snow
    rh = 0.85
    for sgn in (1, -1):
        L_ = math.hypot(w / 2 + 0.2, rh)
        ang = math.atan2(rh, w / 2 + 0.2)
        cxr = x + sgn * (w / 4 + 0.1)
        rf = box((cxr, YH + d / 2, zb + hw + rh / 2), (L_, d + 0.3, 0.1), mat(f'roof{j}', (0.25, 0.08, 0.06) if j % 2 else (0.12, 0.13, 0.16), 0.6))
        rf.rotation_euler = (0, sgn * ang, 0)
        sn = box((cxr - sgn * 0.02, YH + d / 2, zb + hw + rh / 2 + 0.08), (L_ * 1.01, d + 0.34, 0.08), M_SNOW); sn.rotation_euler = (0, sgn * ang, 0)
    # front gable triangle
    me_ = bpy.data.meshes.new('gable'); me_.from_pydata([(x - w / 2, YH, zb + hw), (x + w / 2, YH, zb + hw), (x, YH, zb + hw + rh)], [], [(0, 1, 2)])
    go_ = link(bpy.data.objects.new('gable', me_)); go_.data.materials.append(wm)
    box((x + 0.55, YH + 1.2, zb + hw + 0.75), (0.25, 0.25, 0.7), mat('brick', (0.4, 0.13, 0.08), 0.8), name='chimney')
    box((x + 0.55, YH + 1.2, zb + hw + 1.12), (0.3, 0.3, 0.06), M_SNOW)
    # door, wreath, lamp
    box((x, YH - 0.03, zb + 0.5), (0.5, 0.06, 1.0), mat('door', (0.25, 0.06, 0.04) if j % 2 == 0 else (0.05, 0.18, 0.1), 0.4), name='door')
    wr = obj(prim('torus', M_GREEN, 0.13, 0.045), 'wreath'); wr.location = (x, YH - 0.08, zb + 0.72); wr.rotation_euler = (math.pi / 2, 0, 0)
    for q in range(8):
        a_ = 2 * math.pi * q / 8; px, pz = x + 0.13 * math.cos(a_), zb + 0.72 + 0.13 * math.sin(a_); d_ = 0.022
        lamp_q.append([(px - d_, YH - 0.13, pz), (px, YH - 0.13, pz - d_), (px + d_, YH - 0.13, pz), (px, YH - 0.13, pz + d_)]); lamp_t.append(t_on + 0.03 * q)
    for dx in (-0.62, 0.62):                               # two windows with cross bars
        x0_, z0_ = x + dx - 0.25, zb + 0.75
        win_q.append([(x0_, YH - 0.02, z0_), (x0_ + 0.5, YH - 0.02, z0_), (x0_ + 0.5, YH - 0.02, z0_ + 0.6), (x0_, YH - 0.02, z0_ + 0.6)]); win_t.append(t_on + 0.05)
        box((x + dx, YH - 0.04, z0_ + 0.3), (0.04, 0.03, 0.62), M_WHITE); box((x + dx, YH - 0.04, z0_ + 0.3), (0.52, 0.03, 0.04), M_WHITE)
        box((x + dx, YH - 0.1, z0_ - 0.02), (0.6, 0.16, 0.05), M_SNOW)
    win_q.append([(x - 0.15, YH - 0.02, zb + hw + 0.12), (x + 0.15, YH - 0.02, zb + hw + 0.12), (x + 0.15, YH - 0.02, zb + hw + 0.45), (x - 0.15, YH - 0.02, zb + hw + 0.45)])
    win_t.append(t_on + 0.25)
    box((x + 0.42, YH - 0.12, zb + 1.15), (0.05, 0.1, 0.05), M_DARK)
    lp = sphere((x + 0.42, YH - 0.2, zb + 1.1), 0.07, M_LAMP, (12, 8)); me2 = lp.data.copy(); lp.data = me2
    a = me2.attributes.new('t_on', 'FLOAT', 'FACE'); a.data.foreach_set('value', [t_on] * len(me2.polygons))
    # warm light spilling on the snow
    hl = bpy.data.lights.new('hl', 'POINT'); hl.color = (1.0, 0.65, 0.35); hl.shadow_soft_size = 0.4
    ho = link(bpy.data.objects.new('hl', hl)); ho.location = (x, YH - 0.7, zb + 1.2)
    for t_, e_ in ((0, 0.0), (t_on - 0.03, 0.0), (t_on + 0.1, 70.0), (t_on + 0.6, 35.0), (FINAL - 0.03, 35.0), (FINAL + 0.15, 90.0), (FINAL + 1.2, 40.0)):
        hl.energy = e_; hl.keyframe_insert('energy', frame=max(0, int(t_ * FPS)))
    fit_w(text(f'No.{hs["n"] + 1}', (x, YH - 0.06, zb + 1.27), 0.13, M_TXT_G), 0.4)
quads_mesh('windows', win_q, win_t, M_WIN)
M_LANT = emis('lantern', (1.0, 0.75, 0.4), 6.0)
for j, hs in enumerate(HOUSES):
    xl = hs['x'] + 1.25
    if any(abs(xl - b_['x']) < b_['w'] / 2 + 0.4 for b_ in BLD): continue
    zt_ = hs['zb'] + 2.25; yl = 1.45
    rod((xl, yl, floor_at(xl)), (xl, yl, zt_), 0.035, M_DARK, 8)
    rod((xl, yl, zt_), (xl, yl - 0.25, zt_), 0.02, M_DARK, 6)
    box((xl, yl - 0.25, zt_ - 0.12), (0.12, 0.12, 0.18), M_LANT); box((xl, yl - 0.25, zt_ - 0.01), (0.17, 0.17, 0.04), M_DARK)
    ll = bpy.data.lights.new('lamp', 'POINT'); ll.energy = 14.0; ll.color = (1.0, 0.7, 0.4); ll.shadow_soft_size = 0.1
    link(bpy.data.objects.new('lamp', ll)).location = (xl, yl - 0.35, zt_ - 0.15)
quads_mesh('wreathlights', lamp_q, lamp_t, M_WREATHL)

# ---------------- refrain buildings: bakery, church, town hall; their bells (height = pitch), canopy with the snowballs ----------------
YF = 2.42                                               # facade
SIGN_TXT = {0: ('CHRISTMAS PUDDING BAKERY', 'プディング屋'), 1: ('ST. NICHOLAS CHURCH', '教会'), 2: ('TOWN HALL', '町役場')}
BCOL = {0: (0.62, 0.36, 0.2), 1: (0.7, 0.68, 0.62), 2: (0.45, 0.22, 0.18)}
bwin_q = []; bwin_t = []
for b in BLD:
    k = b['k']; xB = b['x']; w = b['w']; zb = b['zb']; zl = b['ledge']; zr = b['rail']
    fz_ = floor_at(xB)
    if zb - fz_ > 0.02:
        box((xB, YF + 1.1, (zb + fz_) / 2), (w + 0.4, 3.2, zb - fz_), M_STONE, name='plinth')
        box((xB, YF + 1.1, zb - 0.02), (w + 0.46, 3.26, 0.08), M_SNOW)
    t_on = max(s['t_land'] for s in EV['snow'] if s['bld'] == k)
    wm = brickmat(f'bwall{k}', BCOL[k], tuple(c * 0.8 for c in BCOL[k]), (0.75, 0.72, 0.68), 3.0)
    top = zr + 1.4
    box((xB, YF + 1.3, (zb + top) / 2), (w, 2.6, top - zb), wm, name='bld')
    # gable / steeple / clock tower
    if k == 1:
        box((xB - w / 2 + 0.7, YF + 1.0, top + 1.1), (1.2, 1.2, 2.2), wm)
        sp_ = obj(prim('cone', mat('spire', (0.15, 0.2, 0.25), 0.5), 4, 1.0, 0.0, 1.0), 'spire'); sp_.location = (xB - w / 2 + 0.7, YF + 1.0, top + 3.0)
        sp_.scale = (0.95, 0.95, 1.6); sp_.rotation_euler = (0, 0, math.pi / 4)
        st = obj(prim('star', M_GOLD, 0.06), 'cstar'); st.location = (xB - w / 2 + 0.7, YF + 0.95, top + 4.0); st.scale = (0.25,) * 3
    if k == 2:
        box((xB + w / 2 - 0.9, YF + 1.0, top + 1.3), (1.4, 1.4, 2.6), wm)
        ck = rod((xB + w / 2 - 0.9, YF + 0.25, top + 1.6), (xB + w / 2 - 0.9, YF + 0.29, top + 1.6), 0.5, M_WHITE, 32); ck.rotation_euler = (math.pi / 2, 0, 0)
        ck.matrix_world = rod_matrix((xB + w / 2 - 0.9, YF + 0.27, top + 1.6), (xB + w / 2 - 0.9, YF + 0.31, top + 1.6), 0.5)
        rod((xB + w / 2 - 0.9, YF + 0.25, top + 1.6), (xB + w / 2 - 0.9, YF + 0.25, top + 1.95), 0.025, M_DARK, 6)
        rod((xB + w / 2 - 0.9, YF + 0.25, top + 1.6), (xB + w / 2 - 0.65, YF + 0.25, top + 1.6), 0.025, M_DARK, 6)
    me_ = bpy.data.meshes.new('bgable'); me_.from_pydata([(xB - w / 2, YF, top), (xB + w / 2, YF, top), (xB, YF, top + 1.3)], [], [(0, 1, 2)])
    go_ = link(bpy.data.objects.new('bgable', me_)); go_.data.materials.append(wm)
    for sgn in (1, -1):
        L_ = math.hypot(w / 2 + 0.2, 1.3); ang = math.atan2(1.3, w / 2 + 0.2)
        rf = box((xB + sgn * (w / 4 + 0.1), YF + 1.3, top + 0.65), (L_, 2.9, 0.12), M_SNOW); rf.rotation_euler = (0, sgn * ang, 0)
    # ledge (balcony) the snowballs land on, with a low railing at the back edge only
    box((xB, (1.08 + YF) / 2 + 0.02, zl - 0.06), (w - 0.1, YF - 1.04, 0.12), M_SNOW, name='ledge')
    box((xB, 1.1, zl - 0.14), (w - 0.1, 0.06, 0.1), M_GOLDD)
    for x_ in np.linspace(xB - w / 2 + 0.2, xB + w / 2 - 0.2, 4): rod((x_, 1.12, zl - 0.2), (x_, YF - 0.05, zl - 0.6), 0.03, M_GOLDD, 6)
    # canopy: from the facade down to the front edge; the snowball rail with clips along its edge
    c0 = np.array([0, YF, zr + 0.62]); c1 = np.array([0, 0.6, zr + 0.14])
    L_ = np.linalg.norm(c1 - c0); ang = math.atan2(c0[2] - c1[2], c0[1] - c1[1])
    cn = box((xB, (c0[1] + c1[1]) / 2, (c0[2] + c1[2]) / 2), (w + 0.4, L_, 0.08), mat(f'canopy{k}', (0.6, 0.06, 0.07) if k != 1 else (0.1, 0.25, 0.15), 0.5))
    cn.rotation_euler = (ang, 0, 0)
    cs = box((xB, (c0[1] + c1[1]) / 2, (c0[2] + c1[2]) / 2 + 0.07), (w + 0.42, L_, 0.07), M_SNOW); cs.rotation_euler = (ang, 0, 0)
    for x_ in np.arange(xB - w / 2 - 0.1, xB + w / 2 + 0.2, 0.45):       # scalloped valance stripes
        box((x_, 0.62, zr + 0.06), (0.22, 0.03, 0.14), M_WHITE)
    xs = [bb['x'] for bb in b['bells']]
    rod((min(xs) - 0.2, Y_SNOW, zr), (max(xs) + 0.2, Y_SNOW, zr), 0.03, M_GOLD, 10)
    for bb in b['bells']:
        o_ = BALLS[bb['ball']]; p0 = np.array(o_['seg'][0]['p']); tr_rel = o_['seg'][0]['t1']
        piv = link(bpy.data.objects.new('clip', None)); piv.location = (p0[0], Y_SNOW, zr - 0.02)
        for sgn in (1, -1):
            jaw = box((sgn * 0.05, 0, -0.06), (0.025, 0.06, 0.1), M_GOLD); jaw.parent = piv
        ang_ = np.array([0.0 if T[f] < tr_rel else min(1.0, (T[f] - tr_rel) / 0.08) * 0.6 for f in range(NF)])
        piv.rotation_mode = 'XYZ'; key_series(piv, 'scale', 0, 1.0 + 0.8 * ang_, 1e-4)
    # bells (height = pitch) on rods down to a bar fixed to the facade
    zbar = min(bb['z'] for bb in b['bells']) - 0.42
    for bb in b['bells']:
        o_ = BALLS[bb['ball']]; h = o_['hits'][0]
        n_ = np.array(h['n']); c = np.array(h['p']) - RS * n_
        be = obj(prim('bell', M_BELL, 20), 'bell')
        be.matrix_world = Matrix.Translation(Vector(c)) @ frame_from_n(n_).to_4x4() @ Matrix.Diagonal((0.105, 0.105, 0.17, 1))
        col = pitch_col(h['pitch']); be.color = (*col, 0); flash(be, h['t'], col, n=n_, dip=0.02)
        rod((c[0], Y_SNOW, c[2] - 0.16), (c[0], Y_SNOW, zbar), 0.012, M_BRASS, 6)
    rod((min(xs) - 0.15, Y_SNOW, zbar), (max(xs) + 0.15, Y_SNOW, zbar), 0.03, M_GOLDD, 8)
    for x_ in (min(xs) - 0.15, max(xs) + 0.15): rod((x_, Y_SNOW, zbar), (x_, YF, zbar), 0.025, M_GOLDD, 6)
    # windows: upper floor behind the ledge, shop windows below
    for x_ in np.linspace(xB - w / 2 + 0.6, xB + w / 2 - 0.6, 4):
        z0_ = zl + 0.12
        bwin_q.append([(x_ - 0.28, YF - 0.02, z0_), (x_ + 0.28, YF - 0.02, z0_), (x_ + 0.28, YF - 0.02, z0_ + 0.85), (x_ - 0.28, YF - 0.02, z0_ + 0.85)])
        bwin_t.append(t_on)
        z1_ = zb + 0.5
        if zl - 0.6 - z1_ > 0.8:
            for zz_ in np.arange(z1_, zl - 1.3, 1.25):
                bwin_q.append([(x_ - 0.26, YF - 0.02, zz_), (x_ + 0.26, YF - 0.02, zz_), (x_ + 0.26, YF - 0.02, zz_ + 0.7), (x_ - 0.26, YF - 0.02, zz_ + 0.7)])
                bwin_t.append(t_on + 0.2)
                box((x_, YF - 0.08, zz_ - 0.03), (0.62, 0.12, 0.05), M_SNOW)
    en, jp = SIGN_TXT[k]
    sz = zl - 0.42
    box((xB, YF - 0.05, sz), (min(w - 0.3, 5.0), 0.06, 0.42), M_PLATE)
    fit_w(text(en, (xB, YF - 0.1, sz + 0.05), 0.26, M_TXT_G), min(w - 0.5, 4.7))
    fit_w(text(jp, (xB, YF - 0.1, sz - 0.14), 0.11, M_TXT), 2.0)
    bl_ = bpy.data.lights.new('bl', 'POINT'); bl_.color = (1.0, 0.7, 0.4); bl_.shadow_soft_size = 0.6
    bo = link(bpy.data.objects.new('bl', bl_)); bo.location = (xB, YF - 1.0, zl + 0.7)
    for t_, e_ in ((0, 10.0), (t_on - 0.05, 10.0), (t_on + 0.3, 160.0), (FINAL + 0.2, 220.0), (FINAL + 1.2, 140.0)):
        bl_.energy = e_; bl_.keyframe_insert('energy', frame=max(0, int(t_ * FPS)))
quads_mesh('bwindows', bwin_q, bwin_t, M_WIN)

# the bakery's lever: snowball 15 lands in its bucket, the lever tips and pulls the latch rope of the pudding chute
LV = EV['lever']; lp_ = np.array(LV['p'])
lpiv = lp_ + np.array([0.45, 0, -RS - 0.03])
lev = link(bpy.data.objects.new('lever', None)); lev.location = tuple(lpiv)
arm = box((0.0, 0, 0), (1.3, 0.06, 0.05), M_STEEL); arm.parent = lev
for o_ in [box((-0.45 + dx, 0, 0.07), (0.03, 0.3, 0.14), M_RED) for dx in (-0.15, 0.15)] + [box((-0.45, sg * 0.15, 0.07), (0.3, 0.03, 0.14), M_RED) for sg in (-1, 1)] \
        + [box((-0.45, 0, 0.0), (0.32, 0.32, 0.03), M_RED)]:
    o_.parent = lev
rod(tuple(lpiv - [0, 0, 0.0]), tuple(lpiv - [0, 0, 0.35]), 0.03, M_STEEL, 8)
LANG = LAY['ev']['lever']['ang']
lang = np.array([0.0 if T[f] < LV['t'] else smooth01_((T[f] - LV['t']) / 0.25) * LANG for f in range(NF)])
lev.rotation_mode = 'XYZ'; key_series(lev, 'rotation_euler', 1, -lang)
# the snowball in the bucket rides down with it
# rope: lever's right end -> down the facade -> along the ground -> the chute gate
pdx = EV['pudding']; ps0 = np.array(pdx['start'])
rope_pts = [lpiv + [0.62, 0, 0], np.array([lpiv[0] + 0.62, YF - 0.1, floor_at(lpiv[0]) + 0.05]), np.array([ps0[0] + RP + 0.04, YF - 0.1, floor_at(ps0[0]) + 0.05]),
            np.array([ps0[0] + RP + 0.04, Y_BASS + 0.3, floor_at(ps0[0]) + 0.05]), np.array([ps0[0] + RP + 0.04, Y_BASS + 0.3, ps0[2] + 0.3])]
for a_, b_ in zip(rope_pts, rope_pts[1:]): rod(tuple(a_), tuple(b_), 0.012, mat('rope', (0.75, 0.6, 0.35), 0.8), 6)

# ---------------- the pudding stall + chute (front lane) ----------------
al = math.radians(pdx['alpha']); nrm = np.array([math.sin(al), 0, math.cos(al)]); pe0 = np.array(pdx['end'])
s0 = ps0 - nrm * (RP + 0.03) - [0.3, 0, 0]; s1 = pe0 - nrm * (RP + 0.03)
pl_ = obj(prim('cube', M_WOODL), 'chute')
pl_.matrix_world = Matrix.Translation(Vector((s0 + s1) / 2)) @ Matrix.Rotation(al, 4, 'Y') @ Matrix.Diagonal((np.linalg.norm(s1 - s0), 0.5, 0.05, 1))
for dy in (-0.26, 0.26): rod(s0 + [0, dy, 0.12], s1 + [0, dy, 0.12], 0.02, M_GOLD, 6)
for xx in (s0[0] + 0.1, s1[0] - 0.05): rod((xx, Y_BASS, np.interp(xx, [s0[0], s1[0]], [s0[2], s1[2]]) - 0.03), (xx, Y_BASS, floor_at(xx)), 0.03, M_BRASS, 8)
# stall roof over the waiting pudding
for dx in (-0.45, 0.45):
    for dy in (-0.3, 0.3): rod((ps0[0] - 0.1 + dx, Y_BASS + dy, floor_at(ps0[0])), (ps0[0] - 0.1 + dx, Y_BASS + dy, ps0[2] + 0.75), 0.025, M_WOOD, 6)
box((ps0[0] - 0.1, Y_BASS, ps0[2] + 0.78), (1.1, 0.8, 0.06), mat('stallroof', (0.75, 0.08, 0.08), 0.4))
box((ps0[0] - 0.1, Y_BASS, ps0[2] + 0.84), (1.12, 0.82, 0.05), M_SNOW)
box((ps0[0] - 0.1, Y_BASS - 0.42, ps0[2] + 0.62), (1.0, 0.03, 0.22), M_PLATE)
fit_w(text('PUDDING', (ps0[0] - 0.1, Y_BASS - 0.45, ps0[2] + 0.62), 0.13, M_TXT_G), 0.92)
gateP = link(bpy.data.objects.new('pgate', None)); gx_ = ps0[0] + RP + 0.04; gateP.location = (gx_, Y_BASS, ps0[2] + 0.3)
gfl = box((0, 0, -0.2), (0.05, 0.44, 0.34), mat('pgatec', (0.85, 0.75, 0.2), 0.3, **{'Metallic': 0.5})); gfl.parent = gateP
gang = np.array([0.0 if T[f] < pdx['t_release'] - 0.06 else -min(1.0, (T[f] - pdx['t_release'] + 0.06) / 0.12) * 1.3 for f in range(NF)])
gateP.rotation_mode = 'XYZ'; key_series(gateP, 'rotation_euler', 1, gang)

# ---------------- the carolers' gift box at the start (spring lid, spring plunger) ----------------
gb = EV['hut']; pg = np.array(gb['p'])
def gift(c, size, col, ribbon, t_open, open_front=False):
    w, d, h = size; zb_ = c[2]
    box((c[0], c[1], zb_ + 0.02), (w, d, 0.04), col)
    for sx in ((-1,) if open_front else (-1, 1)): box((c[0] + sx * w / 2, c[1], zb_ + h / 2), (0.04, d, h), col)
    for sy in (-1, 1): box((c[0], c[1] + sy * d / 2, zb_ + h / 2), (w, 0.04, h), col)
    box((c[0], c[1] - d / 2 - 0.022, zb_ + h / 2), (0.08, 0.01, h), ribbon)
    hg = link(bpy.data.objects.new('lidhinge', None)); hg.location = (c[0], c[1] + d / 2, zb_ + h)
    lid = box((0, -d / 2, 0.03), (w + 0.06, d + 0.06, 0.06), col); lid.parent = hg
    bw = box((0, -d / 2, 0.065), (0.08, d + 0.07, 0.012), ribbon); bw.parent = hg
    for sgn in (1, -1):
        lp = sphere((sgn * 0.07, -d / 2, 0.1), 1.0, ribbon, (12, 6)); lp.scale = (0.08, 0.035, 0.045); lp.parent = hg
    ang = np.zeros(NF)
    for f in range(NF):
        t = T[f]
        if t >= t_open: ang[f] = 1.9 * min(1, (t - t_open) / 0.18) + 0.12 * math.sin(14 * (t - t_open)) * math.exp(-4 * (t - t_open))
    hg.rotation_mode = 'XYZ'; key_series(hg, 'rotation_euler', 0, -ang, 1e-4)
    return hg
M_GIFT1 = mat('gift1', (0.05, 0.3, 0.15), 0.35, **{'Coat Weight': 0.5}); M_RIB = mat('ribbon', (1.0, 0.75, 0.2), 0.25, **{'Metallic': 0.6})
gift((pg[0] - 0.12, 0, pg[2] - RC - 0.04), (0.62, 0.5, 0.42), M_GIFT1, M_RIB, gb['t_lid'], open_front=True)
box((pg[0] - 0.12, 0, pg[2] - RC - 0.1), (0.9, 0.7, 0.08), M_WOOD, name='shelf')
rod((pg[0] - 0.12, 0, pg[2] - RC - 0.14), (pg[0] - 0.12, 0, floor_at(pg[0])), 0.04, M_BRASS, 8)
pl = rod((-0.32, 0, 0), (0, 0, 0), 0.05, M_STEEL, 12)
bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=0.03, rotation=(0, math.pi / 2, 0)); ph_ = bpy.context.object; ph_.data.materials.append(M_RED); ph_.parent = pl
px = np.zeros(NF)
for f in range(NF):
    t = T[f]; u = (t - (gb['t_push'] - 0.05)) / 0.05
    px[f] = pg[0] - RC - 0.02 - 0.2 + (0.2 * min(1, max(0, u)) if t < gb['t_push'] + 0.4 else 0.2 * max(0, 1 - (t - gb['t_push'] - 0.4) / 0.5))
pl.location = (0, 0, pg[2]); key_series(pl, 'location', 0, px)
fit_w(text('CAROLER', (pg[0] - 0.12, -0.27, pg[2] - RC + 0.2), 0.1, M_TXT_G), 0.55)


# ---------------- machines ----------------
def bucket(c, w, d, h, m):
    zf_ = c[2] - RC - 0.03; objs = []
    objs.append(box((c[0], c[1], zf_ - 0.02), (w, d, 0.04), m))
    for sx in (-1, 1): objs.append(box((c[0] + sx * w / 2, c[1], zf_ + h / 2), (0.03, d, h), m))
    for sy in (-1, 1): objs.append(box((c[0], c[1] + sy * d / 2, zf_ + h / 2), (w, 0.03, h), m))
    return objs


def follow(objs, P, anchor):
    e = link(bpy.data.objects.new('follow', None)); e.location = anchor
    for o in objs: o.parent = e; o.matrix_parent_inverse = Matrix.Translation(-Vector(anchor))
    key_loc(e, P); return e


def trapdoor(c, t_open, m, w=0.3, d=0.3):
    hg = link(bpy.data.objects.new('trap', None)); hg.location = (c[0] + w / 2, c[1], c[2] - RC - 0.035)
    fl_ = box((-w / 2, 0, -0.015), (w, d, 0.03), m); fl_.parent = hg
    ang = np.array([0.0 if T[f] < t_open else min(1.0, (T[f] - t_open) / 0.12) ** 2 * 1.5 for f in range(NF)])
    hg.rotation_mode = 'XYZ'; key_series(hg, 'rotation_euler', 1, ang)
    return hg


M_LIFT = mat('lift', (0.75, 0.08, 0.06), 0.3, **{'Coat Weight': 0.8})
M_LIFT2 = mat('lift2', (0.1, 0.35, 0.6), 0.3, **{'Coat Weight': 0.8})
M_RLIGHT = lightmat('rimlight', (1.0, 0.8, 0.4), on=6.0, flash_=10.0, off_col=(0.3, 0.25, 0.15))
SIGNS = []
for D in DEV:
    k = D['k']; pc = np.array(D['catch']); pr = np.array(D['rel'])
    if D['kind'] in ('gondola', 'sleigh'):
        P = np.array([PC[f] if D['t_catch'] <= T[f] <= D['t_rel'] else (pc if T[f] < D['t_catch'] else pr) for f in range(NF)])
    if D['kind'] == 'gondola':
        HC = 0.9                                               # cable height above the ball
        a0 = pc + [-0.1, 0, HC]; a1 = pr + [0.5, 0, HC]
        rod(a0, a1, 0.012, M_STEEL, 6)
        for a_ in (a0, a1):                                    # stations: a post behind the course with an arm and a bull wheel
            rod((a_[0], 0.55, a_[2] + 0.2), (a_[0], 0.55, floor_at(a_[0])), 0.06, M_WOOD, 8)
            rod((a_[0], 0.55, a_[2] + 0.12), (a_[0], 0.0, a_[2] + 0.12), 0.03, M_WOOD, 6)
            rod(a_ + [0, -0.05, 0.12], a_ + [0, 0.05, 0.12], 0.13, M_RED, 20)
            box((a_[0], 0.3, a_[2] + 0.33), (0.4, 0.75, 0.08), M_SNOW)
        objs = bucket(pc, 0.42, 0.4, 0.2, M_LIFT)
        for dy in (-0.215, 0.215):                             # U-shaped hanger (clear of the ball's plane)
            objs.append(rod(pc + [0, dy, -RC + 0.17], pc + [0, dy, HC - 0.05], 0.014, M_STEEL, 6))
        objs.append(rod(pc + [0, -0.215, HC - 0.05], pc + [0, 0.215, HC - 0.05], 0.014, M_STEEL, 6))
        objs.append(box(tuple(pc + [0, 0, HC + 0.03]), (0.16, 0.08, 0.1), M_DARK))
        objs.append(box(tuple(pc + [0, -0.205, 0.0]), (0.42, 0.01, 0.12), M_GOLD))
        e = follow(objs, P, pc)
        hg = trapdoor(pc, D['t_rel'], M_DARK, 0.38, 0.38); hg.parent = e; hg.matrix_parent_inverse = Matrix.Translation(-Vector(pc))
        SIGNS.append(('ROPEWAY', pc + [0.4, -0.3, -0.62]))
    elif D['kind'] == 'sleigh':
        zt_ = pc[2] - RC - 0.25
        for dy in (-0.27, 0.27): rod((pc[0] - 0.3, dy, zt_), (pr[0] + 1.6, dy, zt_), 0.018, M_STEEL, 6)
        for xx in np.arange(pc[0] - 0.3, pr[0] + 1.6, 0.22):
            if abs(xx - pr[0]) < 0.35: continue                  # the drop through the track at the end: no sleeper there
            box((xx, 0, zt_ - 0.03), (0.06, 0.62, 0.04), M_WOOD)
        for xx in (pc[0] - 0.2, (pc[0] + pr[0]) / 2, pr[0] + 1.5):
            for dy in (-0.27, 0.27): rod((xx, dy, zt_ - 0.02), (xx, dy, floor_at(xx)), 0.03, M_BRASS, 8)
        objs = bucket(pc, 0.44, 0.38, 0.16, M_LIFT)
        # sleigh body + curled runners
        for dy in (-0.2, 0.2): objs.append(box(tuple(pc + [-0.05, dy, -RC - 0.12]), (0.62, 0.06, 0.1), M_RED))
        for dx in (-0.33, 0.23): objs.append(box(tuple(pc + [dx, 0, -RC - 0.12]), (0.1, 0.42, 0.1), M_RED))
        for dy in (-0.2, 0.2):
            objs.append(rod(pc + [-0.35, dy, -RC - 0.2], pc + [0.32, dy, -RC - 0.2], 0.02, M_GOLD, 6))
            objs.append(rod(pc + [0.32, dy, -RC - 0.2], pc + [0.42, dy, -RC - 0.05], 0.02, M_GOLD, 6))
        # toy reindeer in front
        rx = pc[0] + 0.75; rz = zt_ + 0.25
        objs.append(box((rx, 0, rz + 0.08), (0.42, 0.18, 0.2), mat('deer', (0.4, 0.22, 0.1), 0.6)))
        objs.append(box((rx + 0.26, 0, rz + 0.3), (0.16, 0.14, 0.16), mat('deer', (0.4, 0.22, 0.1), 0.6)))
        objs.append(sphere((rx + 0.36, -0.0, rz + 0.3), 0.04, M_BERRY, (10, 6)))
        for dy in (-0.05, 0.05):
            objs.append(rod((rx + 0.24, dy, rz + 0.38), (rx + 0.2, dy * 3, rz + 0.58), 0.012, M_WOODL, 5))
        for lx in (-0.15, 0.15):
            for dy in (-0.06, 0.06): objs.append(rod((rx + lx, dy, rz - 0.02), (rx + lx, dy, zt_ + 0.02), 0.02, mat('deer', (0.4, 0.22, 0.1), 0.6), 5))
        objs.append(rod((rx - 0.21, 0, rz + 0.1), (pc[0] + 0.25, 0, pc[2] - RC - 0.1), 0.01, M_GOLD, 4))
        e = follow(objs, P, pc)
        hg = trapdoor(pc, D['t_rel'], M_DARK, 0.4, 0.34); hg.parent = e; hg.matrix_parent_inverse = Matrix.Translation(-Vector(pc))
        SIGNS.append(('SLEIGH RIDE', pc + [1.1, -0.35, -0.75]))
    elif D['kind'] == 'wheel':
        Cw = np.array(D['centre']); r = D['r']; hang = D['hang']
        hub = link(bpy.data.objects.new('wheel', None)); hub.location = tuple(Cw)
        for yy in (-0.26, 0.26):
            ring = obj(prim('torus', M_GOLD, 1.0, 0.025), 'ring'); ring.parent = hub; ring.rotation_euler = (math.pi / 2, 0, 0); ring.scale = (r, r, r)
            ring.location = (0, yy, 0)
        for a in range(12):
            th = a * math.pi / 6
            for yy in (-0.26, 0.26):
                sp = rod((0, yy, 0), (r * math.cos(th), yy, r * math.sin(th)), 0.015, M_GOLD, 6); sp.parent = hub
                sp.matrix_parent_inverse = Matrix.Translation(-Vector(Cw))
                sp.matrix_world = rod_matrix(Cw + [0, yy, 0], Cw + [r * math.cos(th), yy, r * math.sin(th)], 0.015)
        # rim bulbs (on the front ring), lit from the catch
        bq = []; bt = []
        for a in range(36):
            th = 2 * math.pi * a / 36; px_, pz_ = r * math.cos(th), r * math.sin(th); d_ = 0.03
            bq.append([(px_ - d_, -0.3, pz_), (px_, -0.3, pz_ - d_), (px_ + d_, -0.3, pz_), (px_, -0.3, pz_ + d_)]); bt.append(D['t_catch'] + 0.02 * a)
        bo_ = quads_mesh('rimbulbs', bq, bt, M_RLIGHT); bo_.parent = hub; bo_.matrix_parent_inverse = Matrix.Translation(-Vector(Cw))
        rod(Cw + [0, -0.32, 0], Cw + [0, 0.62, 0], 0.04, M_STEEL, 10)
        for sx in (-1.0, 0.0): rod(Cw + [0, 0.6, 0], (Cw[0] + sx, 0.6, floor_at(Cw[0] + sx)), 0.04, M_BRASS, 8)
        ang = np.zeros(NF)
        for f in range(NF):
            q = PC[f] - Cw + [0, 0, hang]
            if D['t_catch'] <= T[f] <= D['t_rel']: ang[f] = math.atan2(q[2], q[0])
            elif T[f] < D['t_catch']: ang[f] = math.pi
            else: ang[f] = 0.0
        hub.rotation_mode = 'XYZ'; key_series(hub, 'rotation_euler', 1, -(ang - math.pi))
        P = np.array([PC[f] if D['t_catch'] <= T[f] <= D['t_rel'] else (pc if T[f] < D['t_catch'] else pr) for f in range(NF)])
        objs = bucket(pc, 0.38, 0.38, 0.18, M_LIFT2)
        for dy in (-0.205, 0.205):
            objs.append(rod(pc + [-0.19, dy, 0.0], pc + [0, dy, hang], 0.012, M_GOLD, 6)); objs.append(rod(pc + [0.19, dy, 0.0], pc + [0, dy, hang], 0.012, M_GOLD, 6))
        e = follow(objs, P, pc)
        hg = trapdoor(pc, D['t_rel'], M_DARK, 0.34, 0.34); hg.parent = e; hg.matrix_parent_inverse = Matrix.Translation(-Vector(pc))
        SIGNS.append(('CHRISTMAS WHEEL', Cw + [0, -0.45, -r - 0.45]))
for s_, p_ in SIGNS:
    fit_w(text(s_, tuple(p_), 0.12, M_TXT_G), 1.3)

# ---------------- the town-square tree with the great bell on top ----------------
M_TREE = mat('tree', (0.02, 0.16, 0.05), 0.75)
nt = M_TREE.node_tree; bsdf = nt.nodes['Principled BSDF']
nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 28.0; nz.inputs['Detail'].default_value = 6.0
bmp = nt.nodes.new('ShaderNodeBump'); bmp.inputs['Strength'].default_value = 0.6; nt.links.new(nz.outputs['Fac'], bmp.inputs['Height'])
nt.links.new(bmp.outputs['Normal'], bsdf.inputs['Normal'])
mr = nt.nodes.new('ShaderNodeMix'); mr.data_type = 'RGBA'; mr.inputs[6].default_value = (0.01, 0.08, 0.025, 1); mr.inputs[7].default_value = (0.05, 0.28, 0.08, 1)
nt.links.new(nz.outputs['Fac'], mr.inputs[0]); nt.links.new(mr.outputs[2], bsdf.inputs['Base Color'])
M_TLIGHT = lightmat('tlight', (1.0, 0.82, 0.45), on=6.0, flash_=16.0, off_col=(0.02, 0.06, 0.02))
crown = GREAT - [0, 0, RC]
TREE_TOP = crown[2] - 0.5; TREE_H = 5.0; TREE_RB = 1.12; TX = GREAT[0]
tzb = TREE_TOP - TREE_H
bm = bmesh.new(); NS, NZ = 40, 60; rings = []
for iz in range(NZ + 1):
    z = TREE_H * iz / NZ; rlin = TREE_RB * (1 - z / TREE_H); saw = ((z / 0.8) % 1.0)
    r = max(0.0, rlin - 0.025 - 0.07 * saw * (1 - z / TREE_H))
    rings.append([bm.verts.new((TX + r * math.cos(2 * math.pi * a / NS), r * math.sin(2 * math.pi * a / NS), tzb + z)) for a in range(NS)])
for iz in range(NZ):
    for a in range(NS):
        b_ = (a + 1) % NS; bm.faces.new((rings[iz][a], rings[iz][b_], rings[iz + 1][b_], rings[iz + 1][a]))
me = bpy.data.meshes.new('tree'); bm.to_mesh(me); bm.free(); me.materials.append(M_TREE)
for p in me.polygons: p.use_smooth = True
link(bpy.data.objects.new('tree', me))
rod((TX, 0, floor_at(TX)), (TX, 0, tzb + 0.3), 0.18, mat('trunk', (0.2, 0.1, 0.04), 0.7), 16)
tl_q = []; tl_t = []; rr_ = np.random.default_rng(11)
for q in range(70):
    z = rr_.uniform(0.15, 0.95) * TREE_H; r = TREE_RB * (1 - z / TREE_H) - 0.01; a = rr_.uniform(-1.3, 1.3)
    px_, py_ = TX + r * math.sin(a), -r * math.cos(a); d = 0.035
    tl_q.append([(px_ - d, py_ - 0.01, tzb + z), (px_, py_ - 0.01, tzb + z - d), (px_ + d, py_ - 0.01, tzb + z), (px_, py_ - 0.01, tzb + z + d)])
    tl_t.append(FINAL + 0.01 * q)
quads_mesh('treelights', tl_q, tl_t, M_TLIGHT)
gbell = obj(prim('bell', M_BELL, 32), 'greatbell'); gbell.location = tuple(crown + [0, 0, -0.03]); gbell.scale = (0.34, 0.34, 0.48)
gbell.color = (1.0, 0.8, 0.3, 0); flash(gbell, FINAL, (1.0, 0.8, 0.3), dip=0.03, levels=((0, 1.0), (6, 0.7), (40, 0.35)))
dish = rod(tuple(crown - [0, 0, 0.04]), tuple(crown), 0.12, M_GOLD, 24)
rod((TX, 0, TREE_TOP - 0.2), tuple(crown - [0, 0, 0.3]), 0.03, M_GOLDD, 8)
fit_w(text('GREAT BELL', (TX, -1.25, tzb + 0.6), 0.14, M_TXT_G), 1.2)

# ---------------- pudding's bass bars (verses 2-3) on posts, the big drum ----------------
for h in BALLS['pudding']['hits']:
    n_ = np.array(h['n']); c = np.array(h['p']) - RP * n_
    fz = floor_at(c[0])
    if h['kind'] == 'drum':
        dr = rod(tuple(c - [0, 0, 0.42]), tuple(c - [0, 0, 0.02]), 0.42, mat('drum', (0.7, 0.04, 0.05), 0.35, **{'Coat Weight': 0.8}), 32)
        rod(tuple(c - [0, 0, 0.03]), tuple(c), 0.43, mat('drumhead', (0.95, 0.92, 0.85), 0.5), 32)
        for zz in (0.06, 0.38): rod(tuple(c - [0, 0, zz + 0.02]), tuple(c - [0, 0, zz - 0.02]), 0.44, M_GOLD, 32)
        rod(tuple(c - [0, 0, 0.42]), (c[0], c[1], fz), 0.06, M_BRASS, 8)
        hd = obj(prim('cyl', M_BASSBAR, 32), 'drumglow'); hd.matrix_world = rod_matrix(c - [0, 0, 0.012], c + [0, 0, 0.004], 0.4)
        hd.color = (1.0, 0.85, 0.6, 0); flash(hd, h['t'], (1.0, 0.85, 0.6), dip=0.0)
        continue
    o = obj(prim('bcube', M_BASSBAR), 'bassbar')
    o.matrix_world = Matrix.Translation(Vector(c - n_ * 0.04)) @ frame_from_n(n_).to_4x4() @ Matrix.Diagonal((0.32, 0.38, 0.08, 1))
    col = (0.5, 0.2, 0.06)
    o.color = (*col, 0); flash(o, h['t'], col, n=n_, dip=0.03)
    for dy in (-0.12, 0.12): rod((c[0], c[1] + dy, c[2] - 0.08), (c[0], c[1] + dy, fz), 0.02, M_BRASS, 6)
fit_w(text('BASS DRUM', (DRUM[0], Y_BASS - 0.5, DRUM[2] - RP - 0.25), 0.12, M_TXT_G), 1.0)

# ---------------- hit rings (FX pool), snow puffs where snowballs land ----------------
HITS = [(h['t'], np.array(h['p']) - BALLS[n]['r'] * np.array(h['n']), n) for n, b in BALLS.items() for h in b['hits']
        if h['kind'] in ('key', 'door', 'bell', 'bass', 'drum', 'great')]
HITS.sort(key=lambda x: x[0])
NRING = 12
ring_loc = np.zeros((NRING, NF, 3)); ring_scl = np.full((NRING, NF), 1e-3); ring_a = np.zeros((NRING, NF))
for k, (t, c, n) in enumerate(HITS):
    r = k % NRING; f0 = int(round(t * FPS))
    big = 2.4 if (n == 'pudding' and t > FINAL - 0.01) or (n == 'carol' and t > FINAL - 0.01) else (1.3 if n == 'pudding' else (1.0 if n == 'carol' else 0.7))
    for df in range(12):
        f = f0 + df
        if f >= NF: break
        u = df / 11; ring_loc[r, f] = c + np.array([0, -0.25, 0.02]); ring_scl[r, f] = big * (0.25 + 1.0 * (1 - (1 - u) ** 2))
        ring_a[r, f] = (1 - u) ** 1.6
for r in range(NRING):
    o = obj(prim('torus', M_FX, 0.2, 0.012), 'ring'); o.rotation_euler = (math.pi / 2, 0, 0); o.color = (1.0, 0.85, 0.55, 0)
    for f in range(1, NF):
        if ring_a[r, f] == 0: ring_loc[r, f] = ring_loc[r, f - 1]
    key_loc(o, ring_loc[r])
    for k in range(3): key_series(o, 'scale', k, ring_scl[r])
    key_series(o, 'color', 3, ring_a[r], 1e-3)
NSP = 10
for q in range(NSP):
    o = obj(prim('sph', M_FX, 8, 6), 'puff'); o.color = (0.9, 0.95, 1.0, 0)
    Ls = np.zeros((NF, 3)); A_ = np.zeros(NF); S_ = np.full(NF, 1e-3)
    for j, s_ in enumerate(EV['snow']):
        if j % NSP != q: continue
        f0 = int(round(s_['t_land'] * FPS))
        for df in range(10):
            if f0 + df < NF: Ls[f0 + df] = np.array(s_['land']) + [0, -0.05, -RS * 0.6]; A_[f0 + df] = 0.35 * (1 - df / 10) ** 1.3; S_[f0 + df] = 0.1 + 0.16 * df / 10
    for f in range(1, NF):
        if A_[f] == 0: Ls[f] = Ls[f - 1]
    key_loc(o, Ls); key_series(o, 'color', 3, A_, 1e-3)
    for c_ in range(3): key_series(o, 'scale', c_, S_ * (1.0 if c_ != 2 else 0.5))

# ---------------- keyword signboards behind the houses (English first, Japanese below) ----------------
def board(lines, x, z, w, h, y=6.2):
    box((x, y + 0.04, z), (w, 0.06, h), M_PLATE, name='board')
    for sg in (1, -1): box((x, y, z + sg * h / 2), (w + 0.1, 0.08, 0.07), M_GOLD)
    for sg in (1, -1): box((x + sg * w / 2, y, z), (0.07, 0.08, h + 0.1), M_GOLD)
    box((x, y - 0.02, z + h / 2 + 0.06), (w + 0.1, 0.12, 0.08), M_SNOW)
    for sg in (-0.4, 0.4): rod((x + sg * w, y + 0.1, z - h / 2), (x + sg * w, y + 0.1, floor_at(x + sg * w)), 0.06, M_WOOD, 8)
    zz = z + h / 2 - 0.1
    for s_, size, mt in lines:
        zz -= size * 1.22
        fit_w(text(s_, (x, y - 0.03, zz + size * 0.5), size, mt), w - 0.35)
def cx(t): return fx_cam(t)
board([('WE WISH YOU A MERRY CHRISTMAS', 0.52, M_TXT_G), ('Traditional English Christmas Carol', 0.4, M_TXT), ('おめでとうクリスマス（イギリス民謡）', 0.44, M_TXT)],
      cx(2.6) + 0.2, fz_cam(2.6) + 2.55, 6.4, 2.25)
board([('Christmas Marble Machine', 0.5, M_TXT_G), ('Satisfying marble music · ASMR', 0.4, M_TXT), ('クリスマス ビー玉からくり', 0.44, M_TXT)],
      cx(9.0) + 0.3, fz_cam(9.0) + 2.6, 6.0, 2.1)
board([('Carol singers · Snowy village', 0.46, M_TXT_G), ('Classic Christmas song · Xmas', 0.4, M_TXT), ('雪の村のキャロル隊', 0.44, M_TXT)],
      cx(24.5) + 0.3, fz_cam(24.5) + 2.6, 6.0, 2.1)

# ---------------- counter (HOUSES LIT n/12) + question, riding with the camera at the top ----------------
hud = link(bpy.data.objects.new('hud', None)); hud.parent = cam
HZ = 6.0; hh_ = HZ * 24 / LENS; WW = hh_ * 9 / 16
hud.location = (0, hh_ * 0.425, -HZ)
pl_ = box((0, 0, -0.02), (WW * 0.88, hh_ * 0.095, 0.02), emis('hudpanel', (0.05, 0.012, 0.015), 1.0), name='hudplate'); pl_.parent = hud
for sgn in (1, -1):
    gb_ = box((0, sgn * hh_ * 0.0475, -0.01), (WW * 0.9, hh_ * 0.007, 0.02), M_GOLD, name='hudgold'); gb_.parent = hud
ts_h = hh_ * 0.03
t1_ = text('HOUSES LIT', (-WW * 0.1, ts_h * 0.75, 0), ts_h * 1.25, M_TXT, rot=(0, 0, 0), align='RIGHT'); t1_.parent = hud
t2_ = text('Can ONE caroler light up the whole village?', (0, -ts_h * 0.3, 0), ts_h * 0.95, M_TXT_G, rot=(0, 0, 0)); t2_.parent = hud
t3_ = text('1人で村じゅうの灯りをともせる？', (0, -ts_h * 1.15, 0), ts_h * 1.0, M_TXT, rot=(0, 0, 0)); t3_.parent = hud
fit_w(t2_, WW * 0.84)
lit_t = sorted(h['t_on'] for h in HOUSES)
CV = np.array([int(np.searchsorted(lit_t, t + 1e-6, 'right')) for t in T])
DW = ts_h * 1.15; XD0 = -WW * 0.02
for slot in range(2):
    for dgt in range(10):
        o = text(str(dgt), (XD0 + slot * DW, ts_h * 0.75, 0), ts_h * 1.9, M_TXT_G, rot=(0, 0, 0)); o.parent = hud
        vis = np.array([1.0 if f'{int(CV[f]):>2d}'[slot] == str(dgt) else 0.0 for f in range(NF)])
        if not vis.any(): o.hide_render = True; continue
        for k in range(3): key_series(o, 'scale', k, np.maximum(0.001, vis))
o = text('/ 12', (XD0 + 2 * DW - 0.02, ts_h * 0.75, 0), ts_h * 1.4, M_TXT, rot=(0, 0, 0), align='LEFT'); o.parent = hud
bump = np.ones(NF)
for f in range(1, NF):
    if CV[f] != CV[f - 1]:
        for df in range(6):
            if f + df < NF: bump[f + df] = max(bump[f + df], 1.0 + 0.06 * (1 - df / 6))
hide = np.where(T > T_PULL + 0.2, 0.001, 1.0)
for k in range(2): key_series(hud, 'scale', k, bump * hide, 1e-3)
key_series(hud, 'scale', 2, hide, 1e-3)
for o in bpy.data.objects:
    if o.parent is hud: o.visible_shadow = False

# ---------------- ending texts (on the camera): MERRY CHRISTMAS, then the reveal line ----------------
endh = link(bpy.data.objects.new('endh', None)); endh.parent = cam; endh.location = (0, hh_ * 0.3, -HZ)
ep = box((0, -ts_h * 1.4, -0.03), (WW * 0.92, ts_h * 5.6, 0.02), emis('endpanel', (0.05, 0.012, 0.015), 1.0), name='endplate'); ep.parent = endh
for sgn in (1, -1):
    eg = box((0, -ts_h * 1.4 + sgn * ts_h * 2.8, -0.02), (WW * 0.94, hh_ * 0.007, 0.02), M_GOLD); eg.parent = endh
e1 = text('MERRY CHRISTMAS!', (0, 0, 0), ts_h * 2.0, M_TXT_G, rot=(0, 0, 0)); e1.parent = endh; fit_w(e1, WW * 0.86)
e2 = text('12 houses + 65 snowballs + 1 pudding', (0, -ts_h * 1.9, 0), ts_h * 0.9, M_TXT, rot=(0, 0, 0)); e2.parent = endh; fit_w(e2, WW * 0.84)
e3 = text('メリークリスマス', (0, -ts_h * 3.1, 0), ts_h * 0.95, M_TXT, rot=(0, 0, 0)); e3.parent = endh
sv = np.array([0.001 if T[f] < FINAL + 0.25 else min(1.0, (T[f] - FINAL - 0.25) / 0.25) for f in range(NF)])
sv = np.where(T > T_PULL + 0.9, np.maximum(0.001, 1 - (T - T_PULL - 0.9) / 0.3), sv)
for k in range(3): key_series(endh, 'scale', k, np.maximum(0.001, sv), 1e-3)
revh = link(bpy.data.objects.new('revh', None)); revh.parent = cam; revh.location = (0, -hh_ * 0.33, -HZ)
rp = box((0, -ts_h * 0.4, -0.03), (WW * 0.92, ts_h * 5.4, 0.02), emis('revpanel', (0.05, 0.012, 0.015), 1.0), name='revplate'); rp.parent = revh
for sgn in (1, -1):
    rg = box((0, -ts_h * 0.4 + sgn * ts_h * 2.7, -0.02), (WW * 0.94, hh_ * 0.007, 0.02), M_GOLD); rg.parent = revh
r1 = text('The village was on a', (0, ts_h * 1.2, 0), ts_h * 1.25, M_TXT, rot=(0, 0, 0)); r1.parent = revh
r2 = text('CHRISTMAS PUDDING!', (0, -ts_h * 0.6, 0), ts_h * 1.9, M_TXT_G, rot=(0, 0, 0)); r2.parent = revh; fit_w(r2, WW * 0.86)
r3 = text('村はクリスマスプディングの上だった！', (0, -ts_h * 2.2, 0), ts_h * 0.95, M_TXT, rot=(0, 0, 0)); r3.parent = revh; fit_w(r3, WW * 0.8)
rv = np.array([0.001 if T[f] < T_PULL1 - 0.3 else min(1.0, (T[f] - T_PULL1 + 0.3) / 0.3) for f in range(NF)])
for k in range(3): key_series(revh, 'scale', k, np.maximum(0.001, rv), 1e-3)
for o in bpy.data.objects:
    if o.parent in (endh, revh): o.visible_shadow = False

# ---------------- falling snow (two periodic layers riding with the camera; gone in the pull-back) ----------------
for li, (depth, nfl, spd) in enumerate(((5.0, 150, 0.55), (9.0, 300, 0.4))):
    PER = 12.0; vs = []; fs = []
    rr_ = np.random.default_rng(20 + li)
    for q in range(nfl):
        x_, y_ = rr_.uniform(-5, 5), rr_.uniform(0, PER); s_ = rr_.uniform(0.012, 0.024) * (1.0 if li == 0 else 1.5)
        for yy in (y_ - PER, y_):
            k_ = len(vs); vs += [(x_ - s_, yy, 0), (x_, yy - s_, 0), (x_ + s_, yy, 0), (x_, yy + s_, 0)]; fs.append((k_, k_ + 1, k_ + 2, k_ + 3))
    me_ = bpy.data.meshes.new('flakes'); me_.from_pydata(vs, [], fs); me_.materials.append(emis(f'flake{li}', (0.85, 0.9, 1.0), 0.8))
    fl = link(bpy.data.objects.new('flakes', me_)); fl.parent = cam; fl.visible_shadow = False
    per_s = PER / spd; t_ = -per_s * rr_.uniform(0, 1)
    while t_ < NF / FPS + 0.1:
        fl.location = (0, PER * 0.5, -depth); fl.keyframe_insert('location', frame=t_ * FPS)
        fl.location = (0, -PER * 0.5, -depth); fl.keyframe_insert('location', frame=(t_ + per_s) * FPS - 0.01)
        t_ += per_s
    scl = np.where(T > T_PULL + 0.3, 0.001, 1.0)
    for c_ in range(3): key_series(fl, 'scale', c_, scl, 1e-3)

# ---------------- lights, world ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bgn = w.node_tree.nodes['Background']; bgn.inputs['Color'].default_value = (0.012, 0.016, 0.04, 1); bgn.inputs['Strength'].default_value = 1.0
# stars + far dark wall of the dining room (only the giant candles light it)
rr_ = np.random.default_rng(4); sq = []
for q in range(500):
    x_, z_ = rr_.uniform(-140, 200), rr_.uniform(-10, 160); s_ = rr_.uniform(0.06, 0.16)
    sq.append([(x_ - s_, 300, z_), (x_, 300, z_ - s_), (x_ + s_, 300, z_), (x_, 300, z_ + s_)])
me_ = bpy.data.meshes.new('stars'); vs = []; fs = []
for qd in sq:
    k_ = len(vs); vs += qd; fs.append((k_, k_ + 1, k_ + 2, k_ + 3))
me_.from_pydata(vs, [], fs); me_.materials.append(emis('star', (0.9, 0.92, 1.0), 3.0)); link(bpy.data.objects.new('stars', me_))
moon = bpy.data.lights.new('moon', 'SUN'); moon.color = (0.6, 0.7, 1.0); moon.angle = math.radians(3)
mo = link(bpy.data.objects.new('moon', moon)); mo.rotation_euler = (math.radians(50), 0, math.radians(-30))
for t_, e_ in ((0, 0.35), (T_PULL, 0.35), (T_PULL1, 1.6), (NF / FPS, 1.6)):
    moon.energy = e_; moon.keyframe_insert('energy', frame=int(t_ * FPS))
key_ = bpy.data.lights.new('key', 'AREA'); key_.size = 6.0; key_.color = (1.0, 0.85, 0.7); key_.energy = 500.0
ko = link(bpy.data.objects.new('key', key_))
fill = bpy.data.lights.new('fill', 'AREA'); fill.size = 8.0; fill.color = (0.55, 0.65, 1.0); fill.energy = 260.0
fo_ = link(bpy.data.objects.new('fill', fill))
for f in range(0, NF, 5):
    ko.location = (CP[f, 0] - 2.5, -4.5, FOC[f, 2] + 5.0); ko.rotation_euler = (math.radians(45), 0, math.radians(-25))
    ko.keyframe_insert('location', frame=f); ko.keyframe_insert('rotation_euler', frame=f)
    fo_.location = (CP[f, 0] + 4.0, -5.0, FOC[f, 2] + 1.0); fo_.rotation_euler = (math.radians(75), 0, math.radians(35))
    fo_.keyframe_insert('location', frame=f); fo_.keyframe_insert('rotation_euler', frame=f)
fl_ = bpy.data.lights.new('flamel', 'POINT'); fl_.color = (0.35, 0.5, 1.0); fl_.shadow_soft_size = 20.0
flo = link(bpy.data.objects.new('flamel', fl_)); flo.location = (PX0, PY0 - PRAD - 20 * KS, PZC + 10 * KS)
for t_, e_ in ((0, 0.0), (T_PULL + 0.6, 0.0), (T_PULL + 1.6, 4e5 * KS * KS), (NF / FPS, 4e5 * KS * KS)):
    fl_.energy = e_; fl_.keyframe_insert('energy', frame=int(t_ * FPS))
rl = bpy.data.lights.new('revkey', 'AREA'); rl.size = PRAD * 1.5; rl.color = (1.0, 0.75, 0.5)
rlo = link(bpy.data.objects.new('revkey', rl)); rlo.location = (PX0 - 0.6 * PRAD, PY0 - 2.5 * PRAD, PZC + 2.2 * PRAD)
rlo.rotation_euler = (math.radians(60), 0, math.radians(-15))
for t_, e_ in ((0, 0.0), (T_PULL + 0.3, 0.0), (T_PULL1, 1.2e6 * KS * KS), (NF / FPS, 1.2e6 * KS * KS)):
    rl.energy = e_; rl.keyframe_insert('energy', frame=int(t_ * FPS))
gl = bpy.data.lights.new('bellglow', 'POINT'); gl.color = (1.0, 0.8, 0.4)
glo = link(bpy.data.objects.new('bellglow', gl)); glo.location = tuple(crown + [0, -0.7, -0.3])
for t_, e_ in ((0, 0.0), (FINAL - 0.02, 0.0), (FINAL + 0.1, 260.0), (FINAL + 1.2, 120.0)):
    gl.energy = e_; gl.keyframe_insert('energy', frame=int(t_ * FPS))
tg = bpy.data.lights.new('treeglow', 'POINT'); tg.color = (1.0, 0.75, 0.4); tg.shadow_soft_size = 1.0
tgo = link(bpy.data.objects.new('treeglow', tg)); tgo.location = (TX, -1.6, tzb + 2.2)
for t_, e_ in ((0, 0.0), (FINAL - 0.02, 0.0), (FINAL + 0.4, 180.0)):
    tg.energy = e_; tg.keyframe_insert('energy', frame=int(t_ * FPS))
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
