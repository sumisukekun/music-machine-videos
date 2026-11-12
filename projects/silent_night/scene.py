"""Silent Night - 'Can marbles light up the whole village?'  Side view of a snowy hillside village on Christmas Eve.
Every melody note = the lantern ball (and the children) landing on a key of that pitch (one long key for all the lanes that
sing), every bass note = the snowball on a gold bell, every inner-voice pair = the two ornaments on two small bells. Every strike
lights one light of the village (windows, street lamps, the stars of the fir trees); the great church bell lights the rest.
layout.py plans all motion; this file builds and animates it.
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
VIL = EV['village']; FLR = EV['floor']
FLX = np.array(FLR['x']); FLZ = np.array(FLR['z']); FLZM = np.array(FLR['zm']); Y_STEP = FLR['y_step']
def floor_z_at(x): return float(np.interp(x, FLX, FLZ))
def mfloor_z_at(x): return float(np.interp(x, FLX, FLZM))
def ground_at(x, y): return mfloor_z_at(x) if y > Y_STEP else floor_z_at(x)

# ---------------- a clock for the shaders (seconds), for the lights that switch on at a stored time ----------------
def clock_node(nt):
    v = nt.nodes.new('ShaderNodeValue'); v.outputs[0].default_value = 0.0
    v.outputs[0].keyframe_insert('default_value', frame=0)
    v.outputs[0].default_value = NF / FPS; v.outputs[0].keyframe_insert('default_value', frame=NF)
    fc = nt.animation_data.action.fcurves if hasattr(nt.animation_data.action, 'fcurves') else None
    return v


def lightmat(name, col, on=4.0, flash_=14.0, off_col=(0.02, 0.025, 0.05), attr='t_on', tau=0.35):
    """face attribute `attr` = switch-on time: dark glass before, warm light after (with a short flash)"""
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
    em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Color'].default_value = (*col, 1); nt.links.new(st.outputs[0], em.inputs['Strength'])
    gl = nt.nodes.new('ShaderNodeBsdfPrincipled'); gl.inputs['Base Color'].default_value = (*off_col, 1); gl.inputs['Roughness'].default_value = 0.08
    ad = nt.nodes.new('ShaderNodeAddShader'); nt.links.new(gl.outputs[0], ad.inputs[0]); nt.links.new(em.outputs[0], ad.inputs[1])
    nt.links.new(ad.outputs[0], out.inputs[0])
    return m


def quads_mesh(name, quads, tons, m, attr='t_on'):
    """quads: list of 4 corner points; one face each, face attribute attr = tons"""
    me = bpy.data.meshes.new(name); vs = []; fs = []
    for q in quads:
        k = len(vs); vs += [tuple(p) for p in q]; fs.append((k, k + 1, k + 2, k + 3))
    me.from_pydata(vs, [], fs); me.update()
    a = me.attributes.new(attr, 'FLOAT', 'FACE'); a.data.foreach_set('value', [float(t) for t in tons])
    me.materials.append(m); return link(bpy.data.objects.new(name, me))


def boxes_mesh(name, items, mats):
    """items: (centre, size, rot_z, mat_index); one mesh"""
    bm = bmesh.new()
    for c, s_, rz, mi in items:
        r = bmesh.ops.create_cube(bm, size=1.0)
        M_ = Matrix.Translation(Vector(c)) @ Matrix.Rotation(rz, 4, 'Z') @ Matrix.Diagonal((s_[0], s_[1], s_[2], 1))
        bmesh.ops.transform(bm, matrix=M_, verts=r['verts'])
        for f in {f for v in r['verts'] for f in v.link_faces}: f.material_index = mi
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    for m in mats: me.materials.append(m)
    return link(bpy.data.objects.new(name, me))


# ---------------- materials ----------------
M_GOLD = mat('gold', (1.0, 0.7, 0.25), 0.22, **{'Metallic': 0.95})
M_GOLDD = mat('goldd', (0.6, 0.38, 0.12), 0.35, **{'Metallic': 0.9})
M_LANTERN = mat('lantern', (1.0, 0.8, 0.4), 0.1, **{'Coat Weight': 1.0, 'Emission Color': (1.0, 0.62, 0.22), 'Emission Strength': 6.0})
M_SNOWB = mat('snowball', (0.92, 0.95, 1.0), 0.55, **{'Sheen Weight': 0.6, 'Emission Color': (0.6, 0.75, 1.0), 'Emission Strength': 0.15})
M_ORN = mat('orn', (1, 1, 1), 0.08, **{'Metallic': 0.75, 'Coat Weight': 1.0})
_nt = M_ORN.node_tree; _oi = _nt.nodes.new('ShaderNodeObjectInfo'); _nt.links.new(_oi.outputs['Color'], _nt.nodes['Principled BSDF'].inputs['Base Color'])
M_CHILD = mat('child', (1, 1, 1), 0.15, **{'Coat Weight': 1.0, 'Emission Strength': 0.6})
_nt = M_CHILD.node_tree; _oi = _nt.nodes.new('ShaderNodeObjectInfo')
_nt.links.new(_oi.outputs['Color'], _nt.nodes['Principled BSDF'].inputs['Base Color']); _nt.links.new(_oi.outputs['Color'], _nt.nodes['Principled BSDF'].inputs['Emission Color'])
M_KEY = glowmat('key', gain=9.0, base=0.35, rough=0.12, coat=1.0)
M_BELL = glowmat('bell', gain=4.0, base=0.0, rough=0.22, metal=0.9)
M_TINK = glowmat('tink', gain=8.0, base=0.3, rough=0.12, metal=0.4, coat=1.0)
M_WOOD = mat('wood', (0.16, 0.08, 0.04), 0.5)
M_WOODL = mat('woodl', (0.38, 0.2, 0.09), 0.5)
M_SNOW = mat('snow', (0.82, 0.88, 1.0), 0.45, **{'Sheen Weight': 0.5, 'Sheen Tint': (0.7, 0.8, 1.0)})
M_SNOWD = mat('snowd', (0.55, 0.62, 0.8), 0.6)
M_IRON = mat('iron', (0.05, 0.05, 0.06), 0.4, **{'Metallic': 0.8})
M_RED = mat('red', (0.6, 0.04, 0.04), 0.35, **{'Coat Weight': 0.8})
M_FIR = mat('fir', (0.02, 0.12, 0.06), 0.7)
M_PLATE = mat('plate', (0.12, 0.06, 0.03), 0.5)
WALLS = [mat('wall0', (0.55, 0.45, 0.35), 0.8), mat('wall1', (0.42, 0.16, 0.12), 0.8), mat('wall2', (0.28, 0.33, 0.42), 0.8),
         mat('wall3', (0.55, 0.42, 0.2), 0.8), M_SNOW, mat('roof', (0.2, 0.07, 0.05), 0.7), mat('stone', (0.35, 0.36, 0.42), 0.85)]
M_LAMP = objcol('lamp', 0.4, emit=6.0)
M_FX = fxmat('fx', 8.0)
M_TXT = emis('txtw', (1.0, 0.95, 0.85), 4.0); M_TXT_G = emis('txtg', (1.0, 0.75, 0.3), 6.0); M_TXT_R = emis('txtr', (1.0, 0.4, 0.35), 5.0)
M_TXT_B = emis('txtb', (0.6, 0.85, 1.0), 5.0)
M_WIN = lightmat('window', (1.0, 0.68, 0.32), on=5.0, flash_=22.0)
M_LAMPGL = lightmat('lampglass', (1.0, 0.78, 0.45), on=9.0, flash_=25.0)
M_STAR = lightmat('treestar', (1.0, 0.86, 0.3), on=12.0, flash_=30.0, off_col=(0.15, 0.12, 0.05))
M_SKYSTAR = lightmat('skystar', (0.85, 0.9, 1.0), on=7.0, flash_=6.0, off_col=(0.0, 0.0, 0.0), tau=0.6)
PITCH_COL = [(1.0, 0.12, 0.12), (1.0, 0.35, 0.05), (1.0, 0.6, 0.05), (1.0, 0.85, 0.1), (0.55, 1.0, 0.1), (0.1, 1.0, 0.3),
             (0.05, 1.0, 0.75), (0.05, 0.7, 1.0), (0.15, 0.35, 1.0), (0.45, 0.2, 1.0), (0.75, 0.2, 1.0), (1.0, 0.2, 0.65)]
def pcol(p, light=0.0):
    c = PITCH_COL[p % 12]; return tuple(min(1, x * (1 - light) + light) for x in c)
CHILD_COL = [(1.0, 0.25, 0.3), (0.3, 0.6, 1.0), (0.4, 1.0, 0.55), (1.0, 0.85, 0.3), (0.8, 0.45, 1.0)]

# ---------------- balls ----------------
BALL = {}
for n, b in BL.items():
    rr = b['r']; role = b['role']
    o = obj(prim('sph', None, 32, 16), n); o.scale = (rr, rr, rr)
    band = M_GOLD
    if n == 'solo':
        o.data = prim('sph', M_LANTERN, 32, 16)
        lt = bpy.data.lights.new('lanternlight', 'POINT'); lt.energy = 40.0; lt.color = (1.0, 0.65, 0.3); lt.shadow_soft_size = 0.2
        lo = link(bpy.data.objects.new('lanternlight', lt)); lo.parent = o; lo.location = (0, -1.6, 1.4)
    elif role == 'melody':
        o.data = prim('sph', M_CHILD, 32, 16); j = int(n[1:]); o.color = (*CHILD_COL[(j - 1) % len(CHILD_COL)], 1); band = M_SNOW
    elif role == 'bass': o.data = prim('sph', M_SNOWB, 32, 16); band = M_SNOWD
    else:
        o.data = prim('sph', M_ORN, 32, 16); o.color = (0.85, 0.05, 0.06, 1) if n == 'pairlo' else (0.1, 0.55, 0.2, 1)
    P = POS[n]
    key_loc(o, P)
    sp = np.zeros(NF)
    for f in range(1, NF): sp[f] = sp[f - 1] + (P[f, 0] - P[f - 1, 0]) / rr
    o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, sp)
    bd = obj(bpy.data.meshes.new('band'), 'band')
    bm = bmesh.new(); bmesh.ops.create_cone(bm, cap_ends=False, segments=32, radius1=1.005, radius2=1.005, depth=0.14); bm.to_mesh(bd.data); bm.free()
    bd.data.materials.append(band); bd.parent = o; bd.rotation_euler = (math.pi / 2, 0, 0)
    if role == 'pair':
        cap = obj(prim('cyl', M_GOLD, 16), 'ocap'); cap.parent = o; cap.location = (0, 0, 1.0); cap.scale = (0.3, 0.3, 0.25)
    BALL[n] = o

# ---------------- ground: the melody terrace, the front ground, the hill behind (snow) ----------------
X0g, X1g = FLX[0], FLX[-1]
strip_mesh(FLX, FLZ, -12.0, Y_STEP, 0.6, M_SNOW, 'ground_front')
strip_mesh(FLX, FLZM, Y_STEP, VIL['yn'] - 0.3, 0.6, M_SNOW, 'ground_mel')
# the bank between the front ground and the terrace (a snowy stone wall)
bm = bmesh.new(); vs = []
for x, z0, z1 in zip(FLX, FLZ, FLZM):
    vs.append((bm.verts.new((x, Y_STEP, z0 - 0.5)), bm.verts.new((x, Y_STEP, z1))))
for a, b in zip(vs, vs[1:]): bm.faces.new((a[0], b[0], b[1], a[1]))
me = bpy.data.meshes.new('bank'); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new('bank', me)); o.data.materials.append(WALLS[6])
strip_mesh(FLX, FLZM + 0.05, Y_STEP - 0.12, Y_STEP + 0.12, 0.1, M_SNOW, 'bankcap')
# the hill: a grid rising behind the first row of houses
YN = VIL['yn']; HILL = VIL['hill']
bm = bmesh.new(); ys_ = np.concatenate([[YN - 0.3], np.arange(YN, 75.0, 1.5)]); grid = []
for x in FLX[::2]:
    grid.append([bm.verts.new((x, y, mfloor_z_at(x) + HILL * max(0.0, y - YN) - 0.02 + 0.08 * math.sin(1.3 * x + 0.7 * y))) for y in ys_])
for a, b in zip(grid, grid[1:]):
    for j in range(len(ys_) - 1): bm.faces.new((a[j], b[j], b[j + 1], a[j + 1]))
me = bpy.data.meshes.new('hill'); bm.to_mesh(me); bm.free()
for p in me.polygons: p.use_smooth = True
o = link(bpy.data.objects.new('hill', me)); o.data.materials.append(M_SNOW)

# ---------------- melody keys: one per note, long enough for every lane that sings it ----------------
KEY_TH = 0.12
LIFT = EV['lift']
for h in BL['solo']['hits']:
    nv = np.array(h['n']); top = np.array(h['p']) - BL['solo']['r'] * nv
    lanes = h['lanes']; depth = (lanes - 1) * LN + 0.5
    if h['kind'] == 'lift': continue
    if h['kind'] == 'final': nv = np.array([0, 0, 1.0]); top = np.array(h['p']) - np.array([0, 0, BL['solo']['r']])
    ctr = top - nv * KEY_TH / 2 + np.array([0, (lanes - 1) * LN / 2, 0])
    o = obj(prim('bcube', M_KEY), 'key')
    o.matrix_world = Matrix.Translation(Vector(ctr)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((0.42, depth, KEY_TH, 1))
    col = pcol(h['pitch'], 0.0); o.color = (*col, 0)
    flash(o, h['t'], col, nv)
    objs = [o]
    for yy in sorted(set([0.0, (lanes - 1) * LN])):
        objs.append(rod((top[0], yy, top[2] - KEY_TH - 0.02), (top[0], yy, mfloor_z_at(top[0]) - 0.3), 0.03, M_WOOD, 8))
    if h['kind'] == 'lever':
        L_ = EV['bass_gate']; g = np.array(L_['ledge'])
        objs.append(rod((top[0], -0.3, top[2] - 0.06), (top[0], -0.3, top[2] + 0.35), 0.035, M_GOLD))
        objs.append(rod((top[0], -0.3, top[2] + 0.35), (g[0] - 0.25, g[1], top[2] + 0.35), 0.03, M_GOLD))
        objs.append(rod((g[0] - 0.25, g[1], top[2] + 0.35), (g[0] - 0.25, g[1], g[2] + 0.25), 0.03, M_GOLD))
    if h['kind'] == 'final' or h['t'] < 1.6: continue
    riser(objs, h['t'], top, lead=1.2, stay=0.9)

# ---------------- bass bells (gold, mouth down) and pair bells (small round bells) ----------------
for n in ('bass', 'pairlo', 'pairhi'):
    b = BL[n]
    for h in b['hits']:
        if h['kind'] in ('seesaw', 'churchbell'): continue
        nv = np.array(h['n']) if np.linalg.norm(h['vout']) > 0 else np.array([0, 0, 1.0])
        top = np.array(h['p']) - b['r'] * nv; p = h['pitch']
        if n == 'bass':
            sz = 0.32 + 0.012 * (50 - p)
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
        po = rod(bot, (bot[0], bot[1], ground_at(bot[0], bot[1]) - 0.2), 0.035 if n == 'bass' else 0.022, M_GOLDD, 8)
        if not h.get('final'): riser([o, po], h['t'], top, lead=1.2, stay=0.75)

# ---------------- the great church bell (the snowball's last note) in its snowy wooden belfry ----------------
hC = [h for h in BL['bass']['hits'] if h['kind'] == 'churchbell'][0]
topC = np.array(hC['p']) - np.array([0, 0, BL['bass']['r']]); SZC = 0.95
cb = obj(prim('gbell', M_BELL, 48), 'churchbell')
cb.location = tuple(topC); cb.scale = (SZC * 0.8, SZC * 0.8, SZC)
cb.color = (1.0, 0.72, 0.3, 0.0); flash(cb, hC['t'], (1.0, 0.72, 0.3), (0, 0, 1), dip=0.0, levels=((0, 1.0), (8, 0.6), (40, 0.3)))
gz = floor_z_at(topC[0])
for sx in (-1, 1):
    rod((topC[0] + sx * 1.0, topC[1], gz - 0.2), (topC[0] + sx * 1.0, topC[1], topC[2] + 0.35), 0.09, M_WOOD, 12)
box((topC[0], topC[1], topC[2] + 0.42), (2.4, 0.3, 0.14), M_WOOD, name='belfrybeam')
rod((topC[0], topC[1], topC[2] + 0.35), tuple(topC), 0.04, M_IRON, 8)
# little roof with snow
for sx in (-1, 1):
    r_ = box((topC[0] + sx * 0.62, topC[1], topC[2] + 0.75), (1.45, 0.75, 0.08), M_WOOD, rot=(0, sx * 0.5, 0), name='belfryroof')
    r_ = box((topC[0] + sx * 0.62, topC[1], topC[2] + 0.81), (1.45, 0.8, 0.06), M_SNOW, rot=(0, sx * 0.5, 0), name='belfrysnow')
bl_ = bpy.data.lights.new('belllight', 'POINT'); bl_.energy = 0.0; bl_.color = (1.0, 0.75, 0.4); bl_.shadow_soft_size = 0.6
blo = link(bpy.data.objects.new('belllight', bl_)); blo.location = (topC[0], topC[1] - 1.2, topC[2] - 0.4)
fC = int(round(hC['t'] * FPS))
bl_.keyframe_insert('energy', frame=fC - 1); bl_.energy = 2500.0; bl_.keyframe_insert('energy', frame=fC); bl_.energy = 600.0
bl_.keyframe_insert('energy', frame=fC + 30)

# ---------------- street lamps (one per bass strike) and little fir trees (one per ornament strike) ----------------
glass = []; gt_ = []
for L_ in VIL['lamps']:
    x, y, zb = L_['x'], L_['y'], L_['zb']; hh = 1.25
    rod((x, y, zb - 0.1), (x, y, zb + hh), 0.035, M_IRON, 8)
    box((x, y, zb + hh + 0.02), (0.22, 0.22, 0.04), M_IRON, name='lampbase')
    box((x, y, zb + hh + 0.3), (0.26, 0.26, 0.05), M_IRON, name='lamptop')
    box((x, y, zb + hh + 0.35), (0.3, 0.3, 0.05), M_SNOW, name='lampsnow')
    c = np.array([x, y, zb + hh + 0.16]); a_, b_ = 0.1, 0.12
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        if dy: q = [c + [-a_, dy * a_, -b_], c + [a_, dy * a_, -b_], c + [a_, dy * a_, b_], c + [-a_, dy * a_, b_]]
        else: q = [c + [dx * a_, -a_, -b_], c + [dx * a_, a_, -b_], c + [dx * a_, a_, b_], c + [dx * a_, -a_, b_]]
        glass.append(q); gt_.append(L_['t'])
quads_mesh('lampglass', glass, gt_, M_LAMPGL)
stars = []; st_ = []
for Tr in VIL['trees']:
    x, y, zb, s_ = Tr['x'], Tr['y'], Tr['zb'], Tr['s']
    for k, (r0, z0, hh) in enumerate(((0.38, 0.12, 0.45), (0.3, 0.42, 0.4), (0.2, 0.7, 0.35))):
        o = obj(prim('cone', M_FIR, 12, r0 * s_, 0.02, hh * s_), 'fir'); o.location = (x, y, zb + (z0 + hh / 2) * s_)
        o2 = obj(prim('cone', M_SNOW, 12, r0 * s_ * 0.75, 0.01, hh * s_ * 0.4), 'firsnow'); o2.location = (x, y, zb + (z0 + hh * 0.85) * s_)
    rod((x, y, zb - 0.1), (x, y, zb + 0.15 * s_), 0.05, M_WOOD, 6)
    c = np.array([x, y - 0.02, zb + 1.12 * s_]); r1, r2 = 0.13 * s_, 0.055 * s_
    pts = [c + np.array([math.sin(k * math.pi / 5) * (r1 if k % 2 == 0 else r2), 0, math.cos(k * math.pi / 5) * (r1 if k % 2 == 0 else r2)]) for k in range(10)]
    for k in range(0, 10, 2):
        stars.append([c, pts[k - 1], pts[k], pts[(k + 1) % 10]]); st_.append(Tr['t'])
quads_mesh('treestars', stars, st_, M_STAR)

# ---------------- the lantern (start): a lamp post with a glass lantern; its floor drops open ----------------
ln = EV['lantern']; zf = mfloor_z_at(0.0)
rod((ln['x'] - 0.55, 0.0, zf - 0.2), (ln['x'] - 0.55, 0.0, ln['z'] + 0.75), 0.06, M_IRON, 10)
rod((ln['x'] - 0.55, 0.0, ln['z'] + 0.75), (ln['x'], 0.0, ln['z'] + 0.75), 0.04, M_IRON, 8)
box((ln['x'], 0, ln['z'] + 0.5), (0.55, 0.55, 0.06), M_IRON, name='lanterntop')
box((ln['x'], 0, ln['z'] + 0.56), (0.6, 0.6, 0.06), M_SNOW, name='lanternsnow')
for sx in (-1, 1):
    for sy in (-1, 1): rod((ln['x'] + sx * 0.26, sy * 0.26, ln['z'] - 0.24), (ln['x'] + sx * 0.26, sy * 0.26, ln['z'] + 0.48), 0.018, M_IRON, 6)
tr = ln['t_release']
for sgn in (-1, 1):
    fl = box((ln['x'] + sgn * 0.13, 0, ln['z'] - 0.23), (0.26, 0.54, 0.03), M_IRON, name='trap')
    fl.keyframe_insert('rotation_euler', index=1, frame=int(tr * FPS))
    fl.rotation_euler = (0, sgn * 1.4, 0); fl.keyframe_insert('rotation_euler', index=1, frame=int(tr * FPS) + 4)

# ---------------- the snowball's roof ledge, gate and snowy chute ----------------
bg = EV['bass_gate']; led = np.array(bg['ledge']); lip = np.array(bg['lip']); RB_ = BL['bass']['r']
surf = lambda p: p - np.array([0, 0, RB_])
L_r = np.linalg.norm(lip - led)
c_ = (surf(led) + surf(lip)) / 2 - np.array([0, 0, 0.04])
ang = math.atan2(-(lip - led)[2], (lip - led)[0])
box(tuple(c_), (L_r + 0.5, 0.6, 0.08), M_WOODL, rot=(0, ang, 0), name='bramp')
box(tuple(c_ + np.array([0, 0, -0.06])), (L_r + 0.5, 0.66, 0.05), M_SNOW, rot=(0, ang, 0), name='brampsnow')
box(tuple(surf(led) - np.array([0.35, 0, 0.06])), (0.7, 0.7, 0.1), M_WOODL, name='bledge')
# the little hut the snowball waits on (a snowy roof under the ledge)
hx = led[0] - 0.4; hz0 = floor_z_at(hx)
box((hx, led[1], (hz0 + surf(led)[2] - 0.12) / 2), (1.0, 0.8, surf(led)[2] - 0.12 - hz0), WALLS[1], name='snowhut')
rod(surf(lip) - np.array([0, 0, 0.1]), (lip[0], lip[1], floor_z_at(lip[0]) - 0.2), 0.05, M_WOOD)
gate = box(tuple(led + np.array([RB_ + 0.05, 0, 0])), (0.04, 0.6, 2 * RB_ + 0.1), M_RED, name='bgate')
f0 = int(bg['t'] * FPS)
gate.keyframe_insert('location', frame=f0)
gate.location = tuple(led + np.array([RB_ + 0.05, 0, 2 * RB_ + 0.25])); gate.keyframe_insert('location', frame=f0 + 3)
for sgn in (-1, 1):
    rod(surf(led) + np.array([-0.3, sgn * 0.32, 0.12]), surf(lip) + np.array([0, sgn * 0.32, 0.12]), 0.02, M_WOOD)

# ---------------- the sled see-saw (ornament catapult) ----------------
ss = EV['seesaw']; piv = np.array(ss['pivot']); U = np.array(ss['u']); ARM = ss['arm']; TH0 = ss['th0']
AXS = np.array([math.sin(ss['phi']), math.cos(ss['phi']), 0.0])
plank_e = link(bpy.data.objects.new('plank_pivot', None)); plank_e.location = tuple(piv)
yaw = math.atan2(U[1], U[0]); plank_e.rotation_mode = 'ZYX'
pl = box((0, 0, -0.03), (2 * ARM + 0.3, 0.34, 0.06), M_WOODL, name='sled', bevel=True); pl.parent = plank_e
for sy in (-1, 1):
    rn = box((0, sy * 0.15, -0.1), (2 * ARM + 0.2, 0.03, 0.03), M_RED, name='runner'); rn.parent = plank_e
    tip = box((ARM + 0.13, sy * 0.15, -0.03), (0.04, 0.03, 0.16), M_RED, name='runnertip'); tip.parent = plank_e
bell_ss = obj(prim('bell', M_BELL, 32), 'ssbell'); bell_ss.parent = plank_e
szs = ss['bell_h']; bell_ss.location = (ARM, 0, szs); bell_ss.scale = (szs * 0.75, szs * 0.75, szs)
cupo = obj(prim('cone', M_RED, 24, 0.33, 0.22, 0.14), 'basket'); cupo.parent = plank_e; cupo.location = (-ARM, 0, 0.06)
rod(tuple(piv), (piv[0], piv[1], floor_z_at(piv[0]) - 0.2), 0.12, M_WOOD)
rod(tuple(piv - AXS * 0.25), tuple(piv + AXS * 0.25), 0.06, M_IRON)
th = np.array([TH0 if t < ss['t'] else (TH0 - 2 * TH0 * min(1, ((t - ss['t']) / ss['t_swing'])) ** 2) for t in T])
for f in range(NF):
    if 0 < f < NF - 1 and abs(th[f] - th[f - 1]) < 1e-6 and abs(th[f] - th[f + 1]) < 1e-6: continue
    plank_e.rotation_euler = (0, -th[f], yaw); plank_e.keyframe_insert('rotation_euler', frame=f)
hS = [h for h in BL['bass']['hits'] if h['kind'] == 'seesaw'][0]
bell_ss.color = (1.0, 0.7, 0.3, 0); flash(bell_ss, hS['t'], (1.0, 0.7, 0.3), (0, 0, 1), dip=0.0)

# ---------------- the lift between the verses: bucket (its floor = the C key), front mast, gate, chute ----------------
hL = [h for h in BL['solo']['hits'] if h['kind'] == 'lift'][0]
bot = np.array(LIFT['bottom']); per = np.array(LIFT['perch']); lipL = np.array(LIFT['lip']); RS_ = BL['solo']['r']
buck = link(bpy.data.objects.new('bucket', None)); buck.location = tuple(bot - np.array([0, 0, RS_]))
kf = box((0, 0, -KEY_TH / 2), (0.5, 0.5, KEY_TH), M_KEY, name='liftkey', bevel=True); kf.parent = buck
kf.color = (*pcol(hL['pitch']), 0); flash(kf, hL['t'], pcol(hL['pitch']), (0, 0, 1), dip=0.0)
for sx, sy, w_, d_ in ((-1, 0, 0.04, 0.5), (0, -1, 0.5, 0.04), (0, 1, 0.5, 0.04)):
    wl = box((sx * 0.25, sy * 0.25, 0.12), (w_, d_, 0.24), M_IRON, name='bucketwall'); wl.parent = buck
zl = np.array([np.interp(t, [LIFT['t_up0'], LIFT['t_up1']], [0, 1]) for t in T]); zl = zl * zl * (3 - 2 * zl)
key_series(buck, 'location', 2, bot[2] - RS_ + LIFT['rise'] * zl)
mast_x = bot[0]
for dx in (-0.3, 0.3):
    rod((mast_x + dx, -0.42, mfloor_z_at(mast_x) - 0.2), (mast_x + dx, -0.42, per[2] + 0.9), 0.04, M_IRON, 8)
box((mast_x, -0.42, per[2] + 0.95), (0.8, 0.1, 0.1), M_IRON, name='liftbeam')
pul = obj(prim('cyl', M_GOLD, 24), 'pulley'); pul.location = (mast_x, -0.3, per[2] + 0.95); pul.rotation_euler = (math.pi / 2, 0, 0); pul.scale = (0.12, 0.12, 0.06)
cab = rod((mast_x, -0.3, per[2] + 0.83), (mast_x, -0.3, per[2] - 0.2), 0.008, M_IRON, 6)
# the chute from the top of the lift (its gate opens at t_rel)
uL = (lipL - per) / np.linalg.norm(lipL - per)
cc = (per + lipL) / 2 - np.array([0, 0, RS_ + 0.04])
angL = math.atan2(-(lipL - per)[2], (lipL - per)[0])
box(tuple(cc + uL * 0.15), (np.linalg.norm(lipL - per) + 0.3, 0.42, 0.06), M_WOODL, rot=(0, angL, 0), name='lchute')
box(tuple(cc + uL * 0.15 + np.array([0, 0, -0.045])), (np.linalg.norm(lipL - per) + 0.3, 0.46, 0.04), M_SNOW, rot=(0, angL, 0), name='lchutesnow')
for sgn in (-1, 1): rod(per + np.array([0.1, sgn * 0.21, -RS_ + 0.12]), lipL + np.array([0, sgn * 0.21, -RS_ + 0.12]), 0.015, M_GOLD)
rod(lipL - np.array([0, 0, RS_ + 0.08]), (lipL[0], -0.3, mfloor_z_at(lipL[0]) - 0.2), 0.04, M_WOOD)
lg = box(tuple(per + np.array([RS_ + 0.06, 0, -0.02])), (0.03, 0.42, 0.32), M_RED, name='liftgate')
f0 = int(LIFT['t_rel'] * FPS)
lg.keyframe_insert('location', frame=f0 - 1); lg.location = tuple(per + np.array([RS_ + 0.06, 0, 0.4])); lg.keyframe_insert('location', frame=f0 + 2)

# ---------------- the children's houses: door, doorstep, ramp ----------------
CH_H = [hs for hs in VIL['houses'] if hs['kind'] == 'child']
for J, hs in zip(EV['join'], CH_H):
    pe = np.array(J['perch']); lp = np.array(J['lip']); r_ = BL['solo']['r']
    x0, x1 = pe[0] - 0.7, pe[0] + r_ + 0.06; y0, y1 = pe[1] - 0.22, pe[1] + 0.48; zb = pe[2] - r_ - 0.05; hh = hs['h']
    jc = J['j'] - 1
    for c_, s_, m_ in ((((x0 + x1) / 2, (y0 + y1) / 2, zb + hh / 2), (x1 - x0, y1 - y0, hh), WALLS[jc % 4]),):
        box(c_, s_, m_, name='chouse')
    # roof (two slopes along x) with snow
    for sy in (-1, 1):
        box(((x0 + x1) / 2, (y0 + y1) / 2 + sy * 0.2, zb + hh + 0.14), (x1 - x0 + 0.2, 0.48, 0.06), WALLS[5], rot=(-sy * 0.6, 0, 0), name='croof')
        box(((x0 + x1) / 2, (y0 + y1) / 2 + sy * 0.2, zb + hh + 0.19), (x1 - x0 + 0.24, 0.5, 0.05), M_SNOW, rot=(-sy * 0.6, 0, 0), name='croofsnow')
    # snowy rock pillar under the house
    zg = mfloor_z_at(pe[0])
    if zb - zg > 0.1:
        box(((x0 + x1) / 2, (y0 + y1) / 2, (zb + zg) / 2 - 0.1), (x1 - x0 - 0.15, y1 - y0 - 0.1, zb - zg + 0.2), WALLS[6], name='pillar')
    # the door on the +x wall: hinged at its back edge, swings open at t_rel - 0.25
    hinge = link(bpy.data.objects.new('hinge', None)); hinge.location = (x1 + 0.01, pe[1] + 0.2, zb)
    dr = box((0, -0.2, 0.25), (0.03, 0.4, 0.5), M_RED, name='door'); dr.parent = hinge
    td = J['t_rel'] - 0.25
    hinge.keyframe_insert('rotation_euler', index=2, frame=int(td * FPS)); hinge.rotation_euler = (0, 0, 1.6)
    hinge.keyframe_insert('rotation_euler', index=2, frame=int(td * FPS) + 6)
    # light spilling out of the open door
    dl = bpy.data.lights.new('doorlight', 'POINT'); dl.energy = 0.0; dl.color = (1.0, 0.7, 0.4); dl.shadow_soft_size = 0.2
    dlo = link(bpy.data.objects.new('doorlight', dl)); dlo.location = (x1 - 0.2, pe[1], zb + 0.35)
    dl.keyframe_insert('energy', frame=int(td * FPS)); dl.energy = 25.0; dl.keyframe_insert('energy', frame=int(td * FPS) + 6)
    # ramp from the doorstep to the lip
    u = (lp - pe) / max(1e-6, np.linalg.norm(lp - pe)); L_ = np.linalg.norm(lp - pe)
    if L_ > 0.05:
        c_ = (pe + lp) / 2 - np.array([0, 0, r_ + 0.035])
        box(tuple(c_ + u * 0.05), (L_ + 0.12, 0.4, 0.06), M_WOODL, rot=(0, math.atan2(-u[2], u[0]), 0), name='jramp')
        box(tuple(c_ + u * 0.05 + np.array([0, 0, -0.04])), (L_ + 0.12, 0.44, 0.04), M_SNOW, rot=(0, math.atan2(-u[2], u[0]), 0), name='jrampsnow')
        rod(lp - np.array([0, 0, r_ + 0.06]), (lp[0], lp[1], mfloor_z_at(lp[0]) - 0.2), 0.03, M_WOOD, 6)
    box((pe[0] - 0.05, pe[1], zb - 0.02), (0.6, 0.42, 0.04), M_WOODL, name='doorstep')

# ---------------- the village: house bodies, roofs (snow), chimneys, the church; all windows in one mesh ----------------
HS = VIL['houses']; keep = set(VIL['keep'])
items = []
for k, hs in enumerate(HS):
    if hs['kind'] == 'child' or k not in keep: continue
    x, yf, zb, w, d, h = hs['x'], hs['y'], hs['zb'], hs['w'], hs['d'], hs['h']
    if hs['kind'] == 'church':
        items.append(((x, yf + d / 2, zb + h / 2 - 0.5), (w, d, h + 1.0), 0.0, 0))
        continue
    mi = (k * 7) % 4
    items.append(((x, yf + d / 2, zb + h / 2 - 0.6), (w, d, h + 1.2), 0.0, mi))
    if k % 3 == 0:
        items.append(((x + w * 0.25, yf + d * 0.35, zb + h + 0.45), (0.16, 0.16, 0.5), 0.0, 6))
boxes_mesh('houses', items, WALLS)
# the roofs: tilted slabs (roof + snow layer) and the gable triangles, one mesh
bm = bmesh.new()
for k, hs in enumerate(HS):
    if hs['kind'] in ('child', 'church') or k not in keep: continue
    x, yf, zb, w, d, h = hs['x'], hs['y'], hs['zb'], hs['w'], hs['d'], hs['h']
    pitch = 0.62; half = d / 2 + 0.12; rl = half / math.cos(pitch)
    for sy in (-1, 1):
        for th_, m_i, lift_ in ((0.07, 0, 0.0), (0.06, 1, 0.06)):
            cy = yf + d / 2 + sy * half / 2; cz = zb + h + half * math.tan(pitch) / 2 + lift_
            r = bmesh.ops.create_cube(bm, size=1.0)
            M_ = Matrix.Translation((x, cy, cz)) @ Matrix.Rotation(-sy * pitch, 4, 'X') @ Matrix.Diagonal((w + 0.25 + 0.04 * m_i, rl + 0.03 * m_i, th_, 1))
            bmesh.ops.transform(bm, matrix=M_, verts=r['verts'])
            for f in {f for v in r['verts'] for f in v.link_faces}: f.material_index = m_i
    for xx in (x - w / 2, x + w / 2):
        f_ = bm.faces.new((bm.verts.new((xx, yf, zb + h)), bm.verts.new((xx, yf + d, zb + h)), bm.verts.new((xx, yf + d / 2, zb + h + d / 2 * math.tan(pitch)))))
        f_.material_index = 2
me = bpy.data.meshes.new('roofs'); bm.to_mesh(me); bm.free()
for m_ in (WALLS[5], M_SNOW, WALLS[0]): me.materials.append(m_)
link(bpy.data.objects.new('roofs', me))
# the church: tower with a spire, a cross, and snow on the nave
ch = EV['church']
xC, yC, zC, wC, dC, hC_ = ch['x'], ch['y'], ch['zb'], ch['w'], ch['d'], ch['h']
box((xC, yC + dC / 2, zC + hC_ + 0.45), (wC + 0.3, dC + 0.3, 0.9), WALLS[5], rot=(0, 0, 0), name='naveroof')
box((xC, yC + dC / 2, zC + hC_ + 0.93), (wC + 0.35, dC + 0.35, 0.08), M_SNOW, name='navesnow')
xt_ = xC - wC / 2 - 0.55
box((xt_, yC + 0.7, zC + 2.6), (1.1, 1.1, 5.6), WALLS[0], name='tower')
spire = obj(prim('cone', WALLS[5], 4, 0.85, 0.02, 2.4), 'spire'); spire.location = (xt_, yC + 0.7, zC + 5.4 + 1.2); spire.rotation_euler = (0, 0, math.pi / 4)
sn = obj(prim('cone', M_SNOW, 4, 0.5, 0.02, 1.2), 'spiresnow'); sn.location = (xt_, yC + 0.69, zC + 5.4 + 1.95); sn.rotation_euler = (0, 0, math.pi / 4)
rod((xt_, yC + 0.7, zC + 7.8), (xt_, yC + 0.7, zC + 8.5), 0.04, M_GOLD, 8); rod((xt_ - 0.2, yC + 0.7, zC + 8.25), (xt_ + 0.2, yC + 0.7, zC + 8.25), 0.035, M_GOLD, 8)
# windows (dark until their time): the church gets a round window and a tall tower window too
quads = []; tons = []
for w_ in VIL['wins']:
    x, y, z, ww, hh = w_['x'], w_['y'], w_['z'], w_['w'], w_['h']
    quads.append([(x - ww / 2, y, z - hh / 2), (x + ww / 2, y, z - hh / 2), (x + ww / 2, y, z + hh / 2), (x - ww / 2, y, z + hh / 2)]); tons.append(w_['t'])
church_t = min(w_['t'] for w_ in VIL['wins'] if HS[w_['house']]['kind'] == 'church')
for zz in (1.6, 3.2, 4.4):
    quads.append([(xt_ - 0.2, yC + 0.14, zC + zz), (xt_ + 0.2, yC + 0.14, zC + zz), (xt_ + 0.2, yC + 0.14, zC + zz + 0.6), (xt_ - 0.2, yC + 0.14, zC + zz + 0.6)])
    tons.append(church_t)
win_o = quads_mesh('windows', quads, tons, M_WIN)
# window frames: a thin dark cross on every window (one mesh)
bm = bmesh.new()
for q in quads:
    q = np.array(q); c = q.mean(0); ww = q[1][0] - q[0][0]; hh = q[2][2] - q[1][2]
    for sz_ in ((ww + 0.04, 0.012, 0.03), (0.03, 0.012, hh + 0.04)):
        r = bmesh.ops.create_cube(bm, size=1.0); bmesh.ops.transform(bm, matrix=Matrix.Translation(Vector(c) + Vector((0, -0.01, 0))) @ Matrix.Diagonal((*sz_, 1)), verts=r['verts'])
    r = bmesh.ops.create_cube(bm, size=1.0); bmesh.ops.transform(bm, matrix=Matrix.Translation(Vector(c) + Vector((0, -0.03, -hh / 2 - 0.03))) @ Matrix.Diagonal((ww + 0.12, 0.08, 0.05, 1)), verts=r['verts'])
me = bpy.data.meshes.new('wframes'); bm.to_mesh(me); bm.free(); me.materials.append(M_SNOW); link(bpy.data.objects.new('wframes', me))

# ---------------- night sky: mountains, moon, stars (a few from the start; the sky fills at the end) ----------------
XS0, XS1 = FLX[0] - 20, FLX[-1] + 20
bm = bmesh.new(); prev = None
for x in np.arange(XS0, XS1, 2.0):
    hgt = 18 + 7 * math.sin(x * 0.07) + 4 * math.sin(x * 0.19 + 1.0) + 2 * math.sin(x * 0.5)
    v0 = bm.verts.new((x, 95.0, -20.0)); v1 = bm.verts.new((x, 95.0, hgt))
    if prev: bm.faces.new((prev[0], v0, v1, prev[1]))
    prev = (v0, v1)
me = bpy.data.meshes.new('mountains'); bm.to_mesh(me); bm.free(); me.materials.append(mat('mount', (0.12, 0.15, 0.25), 0.8)); link(bpy.data.objects.new('mountains', me))
moon = sphere((8.0, 160.0, 62.0), 4.0, emis('moon', (1.0, 0.95, 0.82), 6.0), (32, 16))
srng = np.random.default_rng(11)
NSTAR = 2600
sx = srng.uniform(XS0 - 40, XS1 + 40, NSTAR); sz = srng.uniform(14, 120, NSTAR); ss_ = srng.uniform(0.08, 0.3, NSTAR) * (1 + (srng.uniform(size=NSTAR) > 0.97) * 1.6)
st_on = np.where(srng.uniform(size=NSTAR) < 0.08, 0.0, FINAL + 0.8 + 2.6 * srng.uniform(size=NSTAR) ** 0.7)
sq = []
for x, z, r in zip(sx, sz, ss_):
    sq.append([(x - r, 170.0, z), (x, 170.0, z - r), (x + r, 170.0, z), (x, 170.0, z + r)])
quads_mesh('stars', sq, st_on, M_SKYSTAR)
# a shooting star after the sky is full
M_SHOOT = fxmat('shoot', 14.0)
sh = sphere((0, 168.0, 0), 0.5, M_SHOOT, (8, 6)); sh.color = (0.9, 0.95, 1.0, 0)
tS = FINAL + 3.6; xs0 = float(np.array(LAY['balls']['solo']['hits'][-1]['p'])[0])
for f in range(NF):
    u = (T[f] - tS) / 0.9
    if -0.05 < u < 1.05 or f in (0, NF - 1):
        uu = min(1, max(0, u))
        sh.location = (xs0 - 30 + 50 * uu, 168.0, 95 - 22 * uu); sh.keyframe_insert('location', frame=f)
        sh.color = (0.9, 0.95, 1.0, (math.sin(math.pi * uu) if 0 < u < 1 else 0.0)); sh.keyframe_insert('color', index=3, frame=f)
        sh.scale = (2.5, 0.5, 0.5); sh.rotation_euler = (0, math.atan2(22, 50), 0)

# ---------------- falling snow: sheets of flakes sliding down and wrapping (linear keys only at the wraps) ----------------
M_FLAKE = emis('flake', (0.85, 0.9, 1.0), 1.6)
XA, XB = -10.0, FLX[-1] + 4
def flake_sheet(n, y0, y1, zlo, zhi, seed, r=0.025):
    rr_ = np.random.default_rng(seed); bm = bmesh.new()
    for _ in range(n):
        x = rr_.uniform(XA, XB); y = rr_.uniform(y0, y1); z = rr_.uniform(0, zhi - zlo)
        bmesh.ops.create_icosphere(bm, subdivisions=1, radius=r * rr_.uniform(0.7, 1.4), matrix=Matrix.Translation((x, y, z)))
    me = bpy.data.meshes.new('flakes'); bm.to_mesh(me); bm.free(); me.materials.append(M_FLAKE)
    return link(bpy.data.objects.new('flakes', me))
ZLO = float(min(FLZ.min(), -12.0)); ZHI = 9.0; HH = ZHI - ZLO; VS = 0.55; PER = HH / VS
for k in range(2):
    for y0, y1, n, sd in ((-9.0, -1.0, 1800, 1), (-1.0, 6.0, 1500, 2)):
        o = flake_sheet(n, y0, y1, ZLO, ZHI, sd * 10 + k)
        o.location.x = 0.15 * k
        # sheet offset z(t) = ZLO + HH - s(t), s = (VS t + k HH) mod 2 HH: two sheets always cover [ZLO, ZHI]; the jump back to the
        # top happens while the sheet is entirely below the scene
        tend = NF / FPS; keys = [(0.0, ZLO + HH - ((k * HH) % (2 * HH)))]
        m_ = 1
        while True:
            tw = (m_ * 2 * HH - k * HH) / VS; m_ += 1
            if tw <= 0: continue
            if tw >= tend: break
            keys += [(tw - 0.002, ZLO - HH), (tw, ZLO + HH)]
        keys.append((tend, ZLO + HH - ((VS * tend + k * HH) % (2 * HH))))
        for t_, z_ in keys:
            o.location.z = z_; o.keyframe_insert('location', index=2, frame=t_ * FPS)

# ---------------- signs (keywords for search, readable on a phone): English first, Japanese below ----------------
def sign(lines, x, y, z, w, h, plate=True):
    if plate:
        box((x, y + 0.08, z), (w, 0.12, h), M_PLATE, rot=(-0.33, 0, 0), name='plate')
        for sgn in (1, -1): box((x, y + 0.08 + 0.324 * sgn * h / 2, z + 0.946 * sgn * h / 2), (w + 0.1, 0.16, 0.07), M_WOODL, rot=(-0.33, 0, 0))
        box((x, y + 0.08 + 0.324 * h / 2 + 0.03, z + 0.946 * h / 2 + 0.08), (w + 0.2, 0.22, 0.09), M_SNOW, rot=(-0.33, 0, 0), name='signsnow')
        for xx in (x - w / 2 + 0.2, x + w / 2 - 0.2): rod((xx, y + 0.1, z - h / 2), (xx, y + 0.1, floor_z_at(xx) - 0.2), 0.05, M_WOOD, 8)
    zz = z + h / 2
    for s_, size, mt in lines:
        zz -= size * 1.25
        fit_w(text(s_, (x, y - 0.03 + 0.344 * (zz + size * 0.15 - z), zz + size * 0.15), size, mt), (w - 0.3) if plate else None)
PS = POS['solo']
def xs_(t): return float(np.interp(t, T, PS[:, 0]))
def zs_(t): return float(np.interp(t, T, PS[:, 2]))
SY, SDZ = -4.8, 3.3
sign([('SILENT NIGHT', 0.5, M_TXT), ('Franz Gruber · 1818', 0.3, M_TXT_G), ('Christmas Carol × Marble Machine', 0.24, M_TXT_B),
      ('「きよしこの夜」グルーバー作曲', 0.17, M_TXT)], xs_(0.6) + 0.6, SY, zs_(0.6) - SDZ, 4.4, 1.7)
sign([('Christmas Eve Village', 0.32, M_TXT_G), ('Classical Christmas Music', 0.28, M_TXT), ('クリスマス・イブの村 クリスマスソング', 0.17, M_TXT)],
     xs_(7.5) + 1.0, SY, zs_(7.5) - SDZ, 3.9, 1.3)
sign([('Satisfying Marble Machine', 0.32, M_TXT_G), ('Every note lights a window', 0.27, M_TXT), ('ビー玉マーブルマシン 音で灯りがともる', 0.17, M_TXT)],
     xs_(14.0) + 1.0, SY, zs_(14.0) - SDZ, 4.1, 1.3)
sign([('Stille Nacht, heilige Nacht', 0.3, M_TXT_G), ('Christmas Music Box ASMR', 0.28, M_TXT), ('「聖夜」オルゴール風 クラシック', 0.17, M_TXT)],
     xs_(20.0) + 1.0, SY, zs_(20.0) - SDZ, 4.1, 1.3)
sign([('Can marbles light up', 0.34, M_TXT_R), ('the whole village?', 0.34, M_TXT_R), ('ビー玉で村じゅうを灯せる？', 0.18, M_TXT)],
     xs_(25.2) + 1.0, SY, zs_(25.2) - SDZ, 4.0, 1.45)
J0 = EV['join'][0]
sign([('Here come the children!', 0.3, M_TXT_G), ('子どもたちも出てきた！', 0.18, M_TXT)], xs_(33.6) + 1.0, SY, zs_(33.6) - SDZ, 3.6, 0.9)
sign([('Up to verse 2!', 0.3, M_TXT_G), ('2番へ エレベーター', 0.17, M_TXT)], xs_(29.6) + 0.6, SY, zs_(29.6) - SDZ, 3.0, 0.85)

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

# ---------------- camera: follow the lantern ball (and its lanes, and the snowball), widen to the whole valley at the end ----------------
FOC = np.zeros((NF, 3)); WID = np.zeros(NF)
lanes_t = np.interp(T, [h['t'] for h in BL['solo']['hits']], [h['lanes'] for h in BL['solo']['hits']])
PB = POS['bass']; tb0 = EV['bass_gate']['t']
for f in range(NF):
    t = T[f]; s = PS[f]
    wb = min(1.0, max(0.0, (t - tb0) / 1.5))                 # once the snowball plays, keep it in the frame as well
    zc = s[2] - 0.6 - wb * 0.35 * max(0.0, s[2] - PB[f, 2])
    FOC[f] = (s[0] + 1.0, (lanes_t[f] - 1) * LN * 0.35 - wb * 0.5, zc - 0.05 * lanes_t[f])
    WID[f] = 5.5 + 0.12 * lanes_t[f] + wb * 0.6
    if t < E['t0'] - 0.2: FOC[f] = s + np.array([0.6, 0, -1.2]); WID[f] = 6.0
    tw = FINAL + 0.2
    if t > tw:
        u = min(1, (t - tw) / 3.0); u = u * u * (3 - 2 * u)
        ch_ = EV['church']
        wide = np.array([ch_['x'] - 5.0, 12.0, ch_['zb'] + 5.5])
        FOC[f] = (1 - u) * FOC[f] + u * wide; WID[f] = (1 - u) * WID[f] + u * 26.0
FOC[:, 0] = smooth(FOC[:, 0], 0.45); FOC[:, 1] = smooth(FOC[:, 1], 0.8); FOC[:, 2] = smooth(FOC[:, 2], 0.7); WID = smooth(WID, 0.5)
LENS = 32.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 600
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TPc = np.zeros((NF, 3))
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24
    CP[f] = (FOC[f, 0], FOC[f, 1] - D, FOC[f, 2] + 0.34 * D)
    TPc[f] = (FOC[f, 0], FOC[f, 1], FOC[f, 2] + 0.2)
key_loc(cam, CP); key_loc(tgt, TPc)

# ---------------- the counter (LIGHTS n / 1818), riding with the camera near the top of the frame ----------------
hud = link(bpy.data.objects.new('hud', None)); hud.parent = cam
HZ = 6.0; hh_ = HZ * 24 / LENS
hud.location = (0, hh_ * 0.36, -HZ)
WW = hh_ * 9 / 16
pl_ = box((0, 0, -0.02), (WW * 0.78, hh_ * 0.078, 0.02), M_PLATE, name='hudplate'); pl_.parent = hud
for sgn in (1, -1):
    gb_ = box((0, sgn * hh_ * 0.039, -0.01), (WW * 0.8, hh_ * 0.006, 0.02), M_GOLD, name='hudgold'); gb_.parent = hud
ts_h = hh_ * 0.032
DW = ts_h * 0.95
XD0 = -WW * 0.02                                   # first digit slot (4 digits, right-aligned), then ' / 1818'
t1 = text('LIGHTS', (XD0 - 0.9 * ts_h, ts_h * 0.26, 0), ts_h * 1.1, M_TXT, rot=(0, 0, 0), align='RIGHT'); t1.parent = hud
t2 = text('灯り', (XD0 - 0.9 * ts_h, -ts_h * 0.66, 0), ts_h * 0.45, M_TXT, rot=(0, 0, 0), align='RIGHT'); t2.parent = hud
t3 = text('/ 1818', (XD0 + 3 * DW + 0.7 * ts_h, 0, 0), ts_h * 1.0, M_TXT_G, rot=(0, 0, 0), align='LEFT'); t3.parent = hud
cnt = EV['count']
ctimes = np.array([c[0] for c in cnt]); cvals = np.array([c[1] for c in cnt])
CV = np.array([cvals[max(0, np.searchsorted(ctimes, t, 'right') - 1)] for t in T])
for slot in range(4):
    for dgt in range(10):
        o = text(str(dgt), (XD0 + slot * DW, 0, 0), ts_h * 1.3, M_TXT_G, rot=(0, 0, 0)); o.parent = hud
        vis = np.zeros(NF)
        for f in range(NF):
            s_ = f'{int(CV[f]):>4d}'
            vis[f] = 1.0 if s_[slot] == str(dgt) else 0.0
        if not vis.any(): o.hide_render = True; continue
        for k in range(3): key_series(o, 'scale', k, np.maximum(0.001, vis))
bump = np.ones(NF)
for f in range(1, NF):
    if CV[f] != CV[f - 1] and T[f] < FINAL:
        for df in range(6):
            if f + df < NF: bump[f + df] = max(bump[f + df], 1.0 + 0.06 * (1 - df / 6))
for k in range(2): key_series(hud, 'scale', k, bump, 1e-3)

# ---------------- the ending: MERRY CHRISTMAS over the valley ----------------
chE = EV['church']
bn = text('MERRY CHRISTMAS', (chE['x'] - 5.0, 9.0, chE['zb'] + 14.0), 2.0, M_TXT_G, rot=(math.pi / 2 - 0.25, 0, 0))
bn2 = text('1818 lights for the carol of 1818', (chE['x'] - 5.0, 9.0, chE['zb'] + 11.6), 1.0, M_TXT, rot=(math.pi / 2 - 0.25, 0, 0))
bn3 = text('「きよしこの夜」1818年 初演', (chE['x'] - 5.0, 9.0, chE['zb'] + 9.9), 0.85, M_TXT, rot=(math.pi / 2 - 0.25, 0, 0))
fit_w(bn, 15.0); fit_w(bn2, 14.0); fit_w(bn3, 12.0)
tB = EV['village']['wave'][1] + 0.2
for o_ in (bn, bn2, bn3):
    k_ = o_.scale[0]; o_.scale = (0.001,) * 3; o_.keyframe_insert('scale', frame=int(tB * FPS))
    o_.scale = (k_,) * 3; o_.keyframe_insert('scale', frame=int(tB * FPS) + 8)

# ---------------- lights ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bgn = w.node_tree.nodes['Background']; bgn.inputs['Color'].default_value = (0.012, 0.02, 0.055, 1); bgn.inputs['Strength'].default_value = 1.0
moonl = bpy.data.lights.new('moon', 'SUN'); moonl.energy = 0.55; moonl.angle = 0.05; moonl.color = (0.6, 0.72, 1.0)
mo = link(bpy.data.objects.new('moonlight', moonl)); mo.rotation_euler = (math.radians(50), math.radians(-25), math.radians(-160))
fil = bpy.data.lights.new('fill', 'SUN'); fil.energy = 0.18; fil.color = (0.5, 0.6, 1.0)
fo = link(bpy.data.objects.new('fill', fil)); fo.rotation_euler = (math.radians(70), math.radians(20), math.radians(10))
# the valley glows warmer as more windows are lit
glow = bpy.data.lights.new('valley', 'AREA'); glow.size = 30.0; glow.color = (1.0, 0.7, 0.4); glow.energy = 0.0
go = link(bpy.data.objects.new('valley', glow)); go.rotation_euler = (math.radians(-60), 0, 0)
for f in range(0, NF, 6):
    go.location = (CP[f, 0], 6.0, FOC[f, 2] - 2.0)
    go.keyframe_insert('location', frame=f)
    glow.energy = 2.5 * float(CV[f]); glow.keyframe_insert('energy', frame=f)

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
