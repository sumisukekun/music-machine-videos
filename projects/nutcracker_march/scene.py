"""The Nutcracker - March: 'Toy Soldiers vs the Mouse King - who wins?'.  Side view of a toy battlefield in the snow
under the Christmas tree at night.  Every key that lights up is struck by a marble at that moment and sounds that note:
soldier marbles on red toy drums and golden bell plates, mouse marbles on cheese keys and dark drums, mouse marbles from
the cheese blimp and golden shells from the soldiers' balloon on little keys (one marble = one note).
layout.py plans all motion (constant gravity, bounces that never add speed, cups / gondolas); this file builds and
animates it.  Fonts: en_bold.ttf = Poppins Bold (SIL OFL); jp_sub.otf = Noto Sans CJK JP Bold (SIL OFL) subset (make_font.py).
python3 scene.py -- test t1 t2 ... | range f0 f1 | probe t ... | pieces"""
import os, sys, json, math, time
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); LAY = json.load(open(f'{HERE}/layout.json')); VO = json.load(open(f'{HERE}/voices.json'))
FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.6) * FPS); T = np.arange(NF) / FPS
G = LAY['G']; EV = LAY['ev']; LN = LAY['lanes']; R_ = LAY['R']
V_ = EV['v']; SLOPE = EV['slope']; T0 = E['t0']
def xs(t): return V_ * (t - T0)
def zs(t): return SLOPE * (t - T0)
SECT = {s: t for s, t in E['secs']}
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
try: sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Base Contrast'
except Exception: pass
sc.view_settings.exposure = 0.3
sc.frame_start, sc.frame_end = 0, NF - 1
FONT_EN = bpy.data.fonts.load(f'{HERE}/en_bold.ttf'); FONT_JP = bpy.data.fonts.load(f'{HERE}/jp_sub.otf')
TXT_K = 2.75; EN_K = 0.62


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
    """object colour = base colour; object alpha drives emission (hit flash / lit)"""
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


def objcol(name, rough=0.45, metal=0.0, emit=0.0, sheen=0.0, coat=0.0):
    m = mat(name, (1, 1, 1), rough, **{'Metallic': metal, 'Sheen Weight': sheen, 'Coat Weight': coat}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    oi = nt.nodes.new('ShaderNodeObjectInfo'); nt.links.new(oi.outputs['Color'], b.inputs['Base Color'])
    if emit:
        nt.links.new(oi.outputs['Color'], b.inputs['Emission Color']); b.inputs['Emission Strength'].default_value = emit
    return m


def stripes(name, c1, c2, scale=6.0, axis='Z', rough=0.3, coat=0.8, twist=0.0, radial=False):
    """candy stripes in object coordinates (candy canes, peppermint swirl when radial)"""
    m = mat(name, c1, rough, **{'Coat Weight': coat}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    tc = nt.nodes.new('ShaderNodeTexCoord'); sep = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(tc.outputs['Object'], sep.inputs[0])
    f = nt.nodes.new('ShaderNodeMath')
    if radial:                                             # peppermint: angle around z
        at = nt.nodes.new('ShaderNodeMath'); at.operation = 'ARCTAN2'; nt.links.new(sep.outputs['Y'], at.inputs[0]); nt.links.new(sep.outputs['X'], at.inputs[1])
        f.operation = 'MULTIPLY'; f.inputs[1].default_value = scale / math.pi; nt.links.new(at.outputs[0], f.inputs[0])
    else:
        ad = nt.nodes.new('ShaderNodeMath'); ad.operation = 'MULTIPLY_ADD'; ad.inputs[1].default_value = twist
        nt.links.new(sep.outputs['X'], ad.inputs[0]); nt.links.new(sep.outputs[axis], ad.inputs[2])
        f.operation = 'MULTIPLY'; f.inputs[1].default_value = scale; nt.links.new(ad.outputs[0], f.inputs[0])
    fr = nt.nodes.new('ShaderNodeMath'); fr.operation = 'PINGPONG'; fr.inputs[1].default_value = 1.0; nt.links.new(f.outputs[0], fr.inputs[0])
    gt = nt.nodes.new('ShaderNodeMath'); gt.operation = 'GREATER_THAN'; gt.inputs[1].default_value = 0.5; nt.links.new(fr.outputs[0], gt.inputs[0])
    mx = nt.nodes.new('ShaderNodeMix'); mx.data_type = 'RGBA'; mx.inputs[6].default_value = (*c1, 1); mx.inputs[7].default_value = (*c2, 1)
    nt.links.new(gt.outputs[0], mx.inputs[0]); nt.links.new(mx.outputs[2], b.inputs['Base Color'])
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
        bmesh.ops.create_cube(bm, size=1.0); bmesh.ops.bevel(bm, geom=bm.edges[:] + bm.verts[:], offset=a[0] if a else 0.18, segments=3, affect='EDGES')
    elif kind == 'cyl': bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=1.0, radius2=1.0, depth=1.0)
    elif kind == 'tube':                                   # open cylinder (glass cup ring), radius 1, height 1
        bmesh.ops.create_cone(bm, cap_ends=False, segments=a[0], radius1=1.0, radius2=1.0, depth=1.0)
    elif kind == 'cone': bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=a[1], radius2=a[2], depth=a[3])
    elif kind == 'sph': bmesh.ops.create_uvsphere(bm, u_segments=a[0], v_segments=a[1], radius=1.0)
    elif kind == 'plane': bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    elif kind == 'star':                                   # puffy 5-point star in the xz plane (faces -y), outer radius 1
        n5 = 5; ri = 0.45; th = a[0] if a else 0.22
        pts = []
        for j in range(2 * n5):
            r_ = 1.0 if j % 2 == 0 else ri; ph = math.pi / 2 + math.pi * j / n5
            pts.append((r_ * math.cos(ph), r_ * math.sin(ph)))
        cf = bm.verts.new((0, -th, 0)); cb = bm.verts.new((0, th, 0))
        ring_f = [bm.verts.new((x, -th * 0.35, z)) for x, z in pts]; ring_b = [bm.verts.new((x, th * 0.35, z)) for x, z in pts]
        for j in range(2 * n5):
            j2 = (j + 1) % (2 * n5)
            bm.faces.new((cf, ring_f[j2], ring_f[j])); bm.faces.new((cb, ring_b[j], ring_b[j2]))
            bm.faces.new((ring_f[j], ring_f[j2], ring_b[j2], ring_b[j]))
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
    elif kind == 'crescent':                               # crescent moon in the xz plane (opening +x), radius 1
        th = a[0]; N_ = 36
        a0, a1 = math.radians(47.4), math.radians(70.8)
        def arc(cx, r, t0, t1, n): return [(cx + r * math.cos(t0 + (t1 - t0) * j / n), r * math.sin(t0 + (t1 - t0) * j / n)) for j in range(n + 1)]
        pts = arc(0, 1, a0, 2 * math.pi - a0, N_) + arc(0.42, 0.78, 2 * math.pi - a1, a1, N_)[1:-1]
        vf = [bm.verts.new((x, -th, z)) for x, z in pts]; vb = [bm.verts.new((x, th, z)) for x, z in pts]
        bm.faces.new(vf[::-1]); bm.faces.new(vb)
        for j in range(len(pts)):
            j2 = (j + 1) % len(pts); bm.faces.new((vf[j], vf[j2], vb[j2], vb[j]))
    elif kind == 'bowl':                                   # half sphere shell, opening up, radius 1
        bmesh.ops.create_uvsphere(bm, u_segments=a[0], v_segments=a[0] // 2, radius=1.0)
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z > 0.02], context='VERTS')
    me = bpy.data.meshes.new(kind); bm.to_mesh(me); bm.free()
    if kind in ('sph', 'cone', 'bell', 'cyl', 'bcube', 'tube', 'bowl'):
        for p in me.polygons: p.use_smooth = True
    if m: me.materials.append(m)
    _MC[key] = me; return me


def obj(me, name='o'): return link(bpy.data.objects.new(name, me))


def box(c, size, m, rot=None, name='box', bev=False):
    o = obj(prim('bcube' if bev else 'cube', m), name); o.location = c; o.scale = size
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
    cu = bpy.data.curves.new('t', 'FONT'); cu.body = s; cu.extrude = ext
    if _latin(s): cu.font = font or FONT_EN; cu.size = size * TXT_K * EN_K
    else: cu.font = FONT_JP; cu.size = size * TXT_K
    cu.align_x = align; cu.align_y = 'CENTER'
    o = link(bpy.data.objects.new('txt', cu)); o.location = loc; o.rotation_euler = rot; o.data.materials.append(m); return o


def fit_w(o, w):
    bpy.context.view_layer.update()
    if w and o.dimensions.x > w: k = w / o.dimensions.x; o.scale = (k, k, k)
    return o


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



def smooth(Xa, k):
    Y = Xa.copy(); a = 1 - math.exp(-1 / FPS / k)
    for f in range(1, len(Xa)): Y[f] = Y[f - 1] + (Xa[f] - Y[f - 1]) * a
    for f in range(len(Xa) - 2, -1, -1): Y[f] = Y[f + 1] + (Y[f] - Y[f + 1]) * a
    return Y


def flash(o, t, col, n=None, base=0.0, after=None, amp=0.03):
    fh = int(round(t * FPS))
    o.color = (*col, base); o.keyframe_insert('color', index=3, frame=max(0, fh - 1))
    for df, a in ((0, 1.0), (4, 0.6), (18, after if after is not None else max(base, 0.1))):
        o.color = (*col, a); o.keyframe_insert('color', index=3, frame=fh + df)
    if n is not None:
        b0 = o.location.copy(); nv = Vector(n)
        for df, d in ((-1, 0), (1, -amp), (4, amp / 3), (8, 0)):
            o.location = b0 + nv * d; o.keyframe_insert('location', frame=max(0, fh + df))


def scale_curve(t_in, t_out, pop=0.3, shrink=0.35, overshoot=True):
    s_ = np.zeros(NF)
    for f in range(NF):
        t = T[f]; u1 = (t - t_in) / pop; u2 = (t - t_out) / shrink
        if u1 <= 0: a = 0.0
        elif u1 < 1: a = (1.18 * math.sin(u1 * math.pi / 2) if u1 < 0.7 else 1.18 - 0.18 * (u1 - 0.7) / 0.3) if overshoot else u1
        else: a = 1.0
        s_[f] = max(0.001, a * (1.0 - min(1, max(0, u2)) ** 1.5))
    return s_


def riser(objs, t_hit, anchor, lead=1.2, stay=0.9, pop=0.3, shrink=0.35, overshoot=True):
    """objects pop into existence (scale around `anchor`) at t_hit-lead and vanish after t_hit+stay"""
    anchor = Vector(anchor)
    e = link(bpy.data.objects.new('pop', None)); e.location = anchor
    for o in objs: o.parent = e; o.matrix_parent_inverse = Matrix.Translation(-anchor)
    s_ = scale_curve(t_hit - lead, t_hit + stay, pop, shrink, overshoot)
    for k in range(3): key_series(e, 'scale', k, s_, 1e-4)
    return e


PITCH_COL = [(1.0, 0.45, 0.55), (1.0, 0.6, 0.45), (1.0, 0.78, 0.4), (1.0, 0.92, 0.5), (0.72, 1.0, 0.55), (0.5, 1.0, 0.7),
             (0.45, 1.0, 0.92), (0.45, 0.82, 1.0), (0.55, 0.62, 1.0), (0.72, 0.55, 1.0), (0.9, 0.55, 1.0), (1.0, 0.55, 0.85)]
def pcol(p, light=0.0):
    c = PITCH_COL[p % 12]; return tuple(min(1, x * (1 - light) + light) for x in c)


def glassfx(name, col, alpha=0.35, glow=0.3):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    gl = nt.nodes.new('ShaderNodeBsdfGlossy'); gl.inputs['Color'].default_value = (*col, 1); gl.inputs['Roughness'].default_value = 0.05
    em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Color'].default_value = (*col, 1); em.inputs['Strength'].default_value = glow
    ad = nt.nodes.new('ShaderNodeAddShader'); nt.links.new(gl.outputs[0], ad.inputs[0]); nt.links.new(em.outputs[0], ad.inputs[1])
    tr_ = nt.nodes.new('ShaderNodeBsdfTransparent'); mx = nt.nodes.new('ShaderNodeMixShader'); mx.inputs['Fac'].default_value = alpha
    lw = nt.nodes.new('ShaderNodeLayerWeight'); lw.inputs['Blend'].default_value = 0.35
    mf = nt.nodes.new('ShaderNodeMath'); mf.operation = 'MULTIPLY_ADD'; mf.inputs[1].default_value = 0.6; mf.inputs[2].default_value = alpha
    nt.links.new(lw.outputs['Facing'], mf.inputs[0]); mc = nt.nodes.new('ShaderNodeClamp'); nt.links.new(mf.outputs[0], mc.inputs[0]); nt.links.new(mc.outputs[0], mx.inputs['Fac'])
    nt.links.new(tr_.outputs[0], mx.inputs[1]); nt.links.new(ad.outputs[0], mx.inputs[2])
    o = nt.nodes.new('ShaderNodeOutputMaterial'); nt.links.new(mx.outputs[0], o.inputs[0]); return m



# ================================================================== the march
MAIN = ['cap', 's2', 's3', 'drum', 'king', 'm2', 'm3', 'rat']
SOLD = ['cap', 's2', 's3', 'drum']; MICE = ['king', 'm2', 'm3', 'rat']
PIECES = []

# ---------------- materials ----------------
M_GOLD = mat('gold', (1.0, 0.76, 0.33), 0.25, **{'Metallic': 0.9})
M_GOLDD = mat('goldd', (0.6, 0.42, 0.18), 0.35, **{'Metallic': 0.85})
M_RED = mat('toyred', (0.78, 0.04, 0.05), 0.28, **{'Coat Weight': 1.0, 'Coat Roughness': 0.08})
M_BLUE = mat('toyblue', (0.05, 0.12, 0.5), 0.3, **{'Coat Weight': 1.0})
M_WHITE = mat('toywhite', (0.95, 0.94, 0.9), 0.35, **{'Coat Weight': 0.6})
M_BLACK = mat('toyblack', (0.02, 0.02, 0.025), 0.3, **{'Coat Weight': 0.8})
M_SKIN = mat('skin', (1.0, 0.96, 0.88), 0.5)
M_CAPB = stripes('capball', (0.82, 0.05, 0.06), (1.0, 0.78, 0.3), scale=3.0, axis='Z', twist=0.0, rough=0.15, coat=1.0)
M_SOLB = stripes('solball', (0.8, 0.06, 0.08), (0.97, 0.95, 0.9), scale=3.0, axis='Z', rough=0.15, coat=1.0)
M_DRUMB = stripes('drumball', (0.08, 0.16, 0.6), (1.0, 0.8, 0.35), scale=4.0, axis='Z', twist=1.2, rough=0.15, coat=1.0)
M_KINGB = mat('kingball', (0.3, 0.27, 0.34), 0.25, **{'Coat Weight': 1.0, 'Sheen Weight': 0.6, 'Sheen Tint': (0.7, 0.4, 1.0, 1.0)})
M_MOUSEB = mat('mouseball', (0.52, 0.5, 0.52), 0.35, **{'Coat Weight': 0.8, 'Sheen Weight': 0.8})
M_RATB = mat('ratball', (0.28, 0.22, 0.2), 0.35, **{'Coat Weight': 0.8, 'Sheen Weight': 0.8})
M_EAR = mat('ear', (1.0, 0.55, 0.62), 0.5, **{'Subsurface Weight': 0.3})
M_EARO = mat('earout', (0.45, 0.43, 0.46), 0.5)
M_PURP = mat('purple', (0.35, 0.08, 0.55), 0.35, **{'Coat Weight': 0.7})
M_CHEESE = mat('cheese', (1.0, 0.72, 0.12), 0.45, **{'Subsurface Weight': 0.2, 'Coat Weight': 0.3})
M_CHEESED = mat('cheesed', (0.75, 0.48, 0.05), 0.5)
M_WOOD = mat('wood', (0.42, 0.24, 0.11), 0.6)
M_SNOW = mat('snow', (0.62, 0.68, 0.85), 0.6, **{'Subsurface Weight': 0.25, 'Sheen Weight': 0.6, 'Emission Color': (0.4, 0.45, 0.8), 'Emission Strength': 0.02})
M_KEY = glowmat('bellplate', gain=9.0, base=0.1, rough=0.15, metal=0.6, coat=1.0)
M_CHEESEK = glowmat('cheesekey', gain=9.0, base=0.05, rough=0.4, coat=0.4)
M_DRUMGLOW = glowmat('drumskin', gain=6.0, base=0.0, rough=0.5)
M_OBJ = objcol('objcol', 0.4, coat=0.6)
M_FX = fxmat('fx', 8.0)
M_FX2 = fxmat('fx2', 3.0)
M_GLASSC = glassfx('glasscup', (0.85, 0.9, 1.0), 0.18, 0.35)
M_TROUGH = glassfx('trough', (0.8, 0.85, 1.0), 0.12, 0.15)
M_PELA = mat('pela', (0.55, 0.53, 0.56), 0.3, **{'Coat Weight': 0.8})
M_SHELL = glowmat('shell', gain=6.0, base=0.6, rough=0.15, metal=0.7, coat=0.8)
M_TREE = mat('fir', (0.04, 0.24, 0.12), 0.75, **{'Sheen Weight': 0.5})
M_TRUNK = mat('trunk', (0.3, 0.17, 0.08), 0.8)
M_ROPE = mat('rope', (0.35, 0.27, 0.18), 0.7)
M_STEEL = mat('steel', (0.7, 0.72, 0.78), 0.25, **{'Metallic': 1.0})
M_CABLE = mat('cable', (0.12, 0.1, 0.1), 0.6)


def ring_mesh(r0, r1, h, seg=40):
    """a ring (annulus) in the xy plane, z from -h/2 to h/2"""
    bm = bmesh.new(); vs = []
    for k in range(seg):
        a = 2 * math.pi * k / seg; c, s_ = math.cos(a), math.sin(a)
        vs.append([bm.verts.new((r * c, r * s_, z)) for r, z in ((r0, -h / 2), (r1, -h / 2), (r1, h / 2), (r0, h / 2))])
    for k in range(seg):
        a, b = vs[k], vs[(k + 1) % seg]
        for j in range(4): bm.faces.new((a[j], b[j], b[(j + 1) % 4], a[(j + 1) % 4]))
    me = bpy.data.meshes.new('ring'); bm.to_mesh(me); bm.free(); return me


# ---------------- marbles + their hats (hats follow the marble, they do not roll) ----------------
BALL = {}
BMAT = dict(cap=M_CAPB, s2=M_SOLB, s3=M_SOLB, drum=M_DRUMB, king=M_KINGB, m2=M_MOUSEB, m3=M_MOUSEB, rat=M_RATB)
HAT = {}
for n in MAIN:
    rr = R_[n]
    bpy.ops.mesh.primitive_uv_sphere_add(radius=rr, segments=32, ring_count=16); o = bpy.context.object; o.name = n
    for p in o.data.polygons: p.use_smooth = True
    o.data.materials.append(BMAT[n]); key_loc(o, POS[n])
    sp = np.zeros(NF)
    for f in range(1, NF): sp[f] = sp[f - 1] + (POS[n][f, 0] - POS[n][f - 1, 0]) / rr
    o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, sp)
    BALL[n] = o
    h = link(bpy.data.objects.new(f'hat_{n}', None)); key_loc(h, POS[n]); HAT[n] = h
    if n in SOLD:                                      # tin-soldier shako: black cylinder, gold band, plume
        k = 1.25 if n == 'cap' else 1.0
        sh = obj(prim('cyl', M_BLACK if n != 'drum' else M_BLUE, 24), 'shako'); sh.parent = h; sh.location = (0, 0, rr * 0.95 + 0.11 * k); sh.scale = (rr * 0.55, rr * 0.55, 0.24 * k)
        bd = obj(prim('cyl', M_GOLD, 24), 'band'); bd.parent = h; bd.location = (0, 0, rr * 0.95 + 0.02); bd.scale = (rr * 0.58, rr * 0.58, 0.05)
        pl = sphere((0, 0, 0), 0.06 * k, M_RED if n != 'cap' else M_WHITE, (12, 6), 'plume'); pl.parent = h; pl.location = (0.0, 0, rr * 0.95 + 0.26 * k); pl.scale = (0.05 * k, 0.05 * k, 0.09 * k)
        if n == 'cap':
            for sx in (-1, 1):
                ep = obj(prim('cyl', M_GOLD, 16), 'epaulet'); ep.parent = h; ep.location = (sx * rr * 0.98, -0.02, rr * 0.25); ep.rotation_euler = (0, math.pi / 2, 0); ep.scale = (0.07, 0.07, 0.05)
    else:                                              # mouse ears (+ the King's crown)
        for sx in (-1, 1):
            ea = obj(prim('cyl', M_EARO, 24), 'ear'); ea.parent = h; ea.location = (sx * rr * 0.62, 0.0, rr * 0.86); ea.rotation_euler = (math.pi / 2, 0, sx * 0.25); ea.scale = (rr * 0.42, rr * 0.42, 0.03)
            ei = obj(prim('cyl', M_EAR, 24), 'earin'); ei.parent = h; ei.location = (sx * rr * 0.62, -0.02, rr * 0.86); ei.rotation_euler = (math.pi / 2, 0, sx * 0.25); ei.scale = (rr * 0.3, rr * 0.3, 0.02)
        if n == 'king':
            cr = obj(ring_mesh(rr * 0.42, rr * 0.5, 0.12, 32), 'crown'); cr.data.materials.append(M_GOLD); cr.parent = h; cr.location = (0, 0, rr * 0.98)
            for j in range(6):
                a = 2 * math.pi * j / 6
                sp_ = obj(prim('cone', M_GOLD, 4, 0.05, 0.0, 0.14), 'spike'); sp_.parent = h; sp_.location = (rr * 0.46 * math.cos(a), rr * 0.46 * math.sin(a), rr * 0.98 + 0.12)
            gm = sphere((0, 0, 0), 0.035, M_RED, (8, 4), 'gem'); gm.parent = h; gm.location = (0, -rr * 0.5, rr * 0.98)
        tl = obj(prim('cyl', M_EAR, 6), 'tail'); tl.parent = h
        tl.location = (-rr * 1.05, 0.05, -rr * 0.35); tl.rotation_euler = (0, -2.2, 0); tl.scale = (0.014, 0.014, 0.4)
lt = bpy.data.lights.new('caplight', 'POINT'); lt.color = (1.0, 0.75, 0.5); lt.shadow_soft_size = 0.2; lt.energy = 25
link(bpy.data.objects.new('caplight', lt)).parent = BALL['cap']

# pellets: mouse marbles (grey with ears) and golden shells
PEL = sorted(n for n in BP if len(n) == 4 and n[1:].isdigit())
PK = {k['name']: k for k in EV['pellet_keys']}
for n in PEL:
    b = BP[n]; hp = [h for h in b.hits if h['kind'] == 'pellet'][0]; p_ = hp['pitch']
    rp_ = b.r; army = n[0] == 'a'
    o = obj(prim('sph', M_PELA if army else M_SHELL, 16, 8), n); o.scale = (rp_, rp_, rp_)
    key_loc(o, POS[n])
    t_show = EV['pellet_show'][n]; t_out = b.hits[-1]['t'] + 0.08
    s_ = scale_curve(t_show, t_out, 0.2, 0.12, overshoot=False)
    for k in range(3): key_series(o, 'scale', k, s_ * rp_, 1e-5)
    if army:
        for sx in (-1, 1):
            ea = obj(prim('cyl', M_EAR, 12), 'pear'); ea.parent = o; ea.location = (sx * 0.6, 0, 0.85); ea.rotation_euler = (math.pi / 2, 0, 0); ea.scale = (0.42, 0.42, 0.25)
    else:
        col = (1.0, 0.72, 0.25); o.color = (*col, 0.3); flash(o, hp['t'], (1.0, 0.85, 0.5), base=0.3, after=0.5)

# ---------------- ground: snow, stepping down under the lowest marbles ----------------
pts = []
for n in ('drum', 'rat'):
    for h in BP[n].hits:
        pts.append((h['p'][0], h['p'][2] - R_[n] - 1.15))
pts += [(x, zs(x / V_ + T0) - 2.2) for x in np.arange(-25, xs(FINAL) + 40, 2.0)]
pts.sort()
GX = np.array([p[0] for p in pts]); GZv = np.array([p[1] for p in pts])
FLX = np.arange(-25.0, GX.max() + 30, 0.25)
FLZ = np.array([GZv[np.abs(GX - x) < 1.4].min() if (np.abs(GX - x) < 1.4).any() else np.nan for x in FLX])
ok_ = ~np.isnan(FLZ); FLZ = np.interp(FLX, FLX[ok_], FLZ[ok_])
FLZ = np.minimum.accumulate(FLZ)
for _ in range(3): FLZ = np.minimum(FLZ, np.convolve(np.pad(FLZ, 6, mode='edge'), np.ones(13) / 13, 'valid'))
def ground_z(x): return float(np.interp(x, FLX, FLZ))


def strip_mesh(X, Z, y0, y1, th, m, name, Z1=None):
    bm = bmesh.new(); vs = []
    Z1 = Z if Z1 is None else Z1
    for x, z, z1 in zip(X, Z, Z1):
        vs.append([bm.verts.new((x, y0, z)), bm.verts.new((x, y1, z1)), bm.verts.new((x, y1, z1 - th)), bm.verts.new((x, y0, z - th))])
    for a, b in zip(vs, vs[1:]):
        for k in range(4): bm.faces.new((a[k], b[k], b[(k + 1) % 4], a[(k + 1) % 4]))
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new(name, me)); o.data.materials.append(m)
    for p in o.data.polygons: p.use_smooth = True
    return o


strip_mesh(FLX, FLZ, -7.0, 9.0, 1.2, M_SNOW, 'snowfield', Z1=FLZ + 0.5)
for x0 in np.arange(-22, FLX[-1], 1.9):
    sphere((x0 + rng.uniform(-0.5, 0.5), rng.uniform(-5.5, -3.6), ground_z(x0) - 0.3), rng.uniform(0.5, 0.9), M_SNOW, (14, 7), 'drift').scale[2] *= 0.35


def post(top, r=0.03, m=None):
    return rod((top[0], top[1], top[2]), (top[0], top[1], ground_z(top[0]) - 0.1), r, m or M_GOLDD, 6)


# ---------------- keys of the bouncing marbles ----------------
def drum_key(top, n, p, t, rr, body_m, rim_m, skin_col, hh=0.26):
    M = Matrix.Translation(Vector(top)) @ frame_from_n(n).to_4x4()
    skin = obj(prim('cyl', M_DRUMGLOW, 32), 'skin'); skin.matrix_world = M @ Matrix.Translation((0, 0, -0.01)) @ Matrix.Diagonal((rr, rr, 0.02, 1))
    flash(skin, t, pcol(p, 0.45), base=0.0, after=0.08)
    body = obj(prim('cyl', body_m, 32), 'drumbody'); body.matrix_world = M @ Matrix.Translation((0, 0, -0.02 - hh / 2)) @ Matrix.Diagonal((rr * 1.02, rr * 1.02, hh, 1))
    rim = obj(prim('cyl', rim_m, 32), 'rim'); rim.matrix_world = M @ Matrix.Translation((0, 0, -0.03)) @ Matrix.Diagonal((rr * 1.07, rr * 1.07, 0.04, 1))
    rim2 = obj(prim('cyl', rim_m, 32), 'rim2'); rim2.matrix_world = M @ Matrix.Translation((0, 0, -0.02 - hh)) @ Matrix.Diagonal((rr * 1.07, rr * 1.07, 0.04, 1))
    bot = np.array(top) - np.array(n) * (hh + 0.04)
    return [skin, body, rim, rim2]


def bar_key(top, n, p, t, m, w=0.42, d=0.4, th=0.08, base=0.08, after=0.3, light=0.1):
    M = Matrix.Translation(Vector(top)) @ frame_from_n(n).to_4x4()
    k_ = obj(prim('bcube', m, 0.25), 'bar'); k_.matrix_world = M @ Matrix.Translation((0, 0, -th / 2)) @ Matrix.Diagonal((w, d, th, 1))
    col = pcol(p, light); k_.color = (*col, base); flash(k_, t, col, Vector(n), base=base, after=after)
    return [k_]


KEY_STYLE = {}
for nm_ in MAIN:
    b = BP[nm_]; rr_ = b.r
    for h in b.hits:
        if h['kind'] != 'key': continue
        nv = np.array(h['n']); top = np.array(h['p']) - rr_ * nv; pp = h['pitch']; t = h['t']; p0 = max(pp)
        lead, stay = 0.4, 0.75
        if nm_ == 'cap':
            objs = bar_key(top, nv, p0, t, M_KEY, w=0.4, d=0.4, light=0.15); hw, hd, th = 0.2, 0.2, 0.04; lead, stay = 0.55, 0.9
        elif nm_ in ('s2', 's3'):
            objs = drum_key(top, nv, p0, t, 0.25, M_RED, M_GOLD, (1, 1, 1), 0.22); hw = hd = 0.25 * 0.72; th = 0.03
        elif nm_ == 'drum':
            objs = drum_key(top, nv, p0, t, 0.31, M_BLUE, M_GOLD, (1, 1, 1), 0.3); hw = hd = 0.31 * 0.72; th = 0.03
        elif nm_ == 'king':
            objs = bar_key(top, nv, p0, t, M_CHEESEK, w=0.4, d=0.4, th=0.1, base=0.05, after=0.35, light=0.0); hw, hd, th = 0.2, 0.2, 0.05
            if t > E['bar_t']['22'] + 1.0: lead, stay = 0.55, max(0.9, E['bar_t']['24'] + 14 * E['u'] - t + 0.6)   # the grand staircase stays until he is in the cage
        elif nm_ in ('m2', 'm3'):
            objs = bar_key(top, nv, p0, t, M_CHEESEK, w=0.4, d=0.4, th=0.1, base=0.05, after=0.3); hw, hd, th = 0.2, 0.2, 0.05
        else:
            objs = drum_key(top, nv, p0, t, 0.3, M_PURP, M_GOLDD, (1, 1, 1), 0.28); hw = hd = 0.3 * 0.72; th = 0.03
        riser(objs, t, top, lead=lead, stay=stay, pop=0.18, shrink=0.3)
        PIECES.append(dict(kind='key', t=t, top=top, n=nv, hw=hw, hd=hd, th=th, ball=nm_, rs=(lead, stay, 0.18, 0.3)))

# ---------------- the drummer's xylophone ramps (the bass runs of bars 81 / 83) ----------------
for rp in EV['ramps']:
    who = rp['who']; u = np.array(rp['u']); n = np.array(rp['n']); pa = np.array(rp['p_a']); R0 = R_[who]
    sk = rp['s_keys']; s_end = float(np.dot(np.array(rp['p_end']) - pa, u))
    hk = [h for h in BP[who].hits if h['kind'] in ('rampland', 'roll') and rp['t_a'] - 1e-6 <= h['t'] <= rp['t_end']]
    Mr = Matrix(((u[0], 0, n[0]), (0, 1, 0), (u[2], 0, n[2])))
    objs = []
    for j, s_ in enumerate(sk):
        c = pa + u * s_ - n * (R0 + 0.04)
        b_ = obj(prim('bcube', M_KEY, 0.2), 'rampbar'); b_.matrix_world = Matrix.Translation(Vector(c)) @ Mr.to_4x4() @ Matrix.Diagonal((0.15, 0.62, 0.08, 1))
        col = pcol(max(hk[j]['pitch']), 0.25); b_.color = (*col, 0.1); flash(b_, hk[j]['t'], col, base=0.1, after=0.25)
        objs.append(b_)
        PIECES.append(dict(kind='roll', t=hk[j]['t'], top=c + n * 0.04, n=n, hw=0.075, hd=0.31, th=0.04, ball=who, own=(rp['t_a'] - 0.03, rp['t_end'] + 0.03),
                           life=(rp['t_a'] - 0.6, rp['t_end'] + 0.7)))
    for sy in (-0.33, 0.33):
        a_ = pa + u * (sk[0] - 0.25) - n * (R0 + 0.1) + np.array([0, sy, 0]); b2 = pa + u * (s_end + 0.05) - n * (R0 + 0.1) + np.array([0, sy, 0])
        objs.append(rod(a_, b2, 0.03, M_GOLD, 8))
    mid = pa + u * (0.5 * s_end) - n * (R0 + 0.14)
    objs.append(post(mid, 0.04))
    riser(objs, rp['t_a'], pa - n * R0, lead=0.55, stay=(rp['t_end'] - rp['t_a']) + 0.5, pop=0.18, shrink=0.3)

# ---------------- pellet keys (small cheese keys / golden keys) and troughs ----------------
for k in EV['pellet_keys']:
    top = np.array(k['top']); nv = np.array(k['n']); army = k['name'][0] == 'a'
    objs = bar_key(top, nv, k['p'], k['t'], M_CHEESEK if army else M_KEY, w=0.15, d=0.4, th=0.07, base=0.05, after=0.4, light=0.1)
    riser(objs, k['t'], top, lead=0.45, stay=0.55, pop=0.15, shrink=0.25)
    PIECES.append(dict(kind='pkey', t=k['t'], top=top, n=nv, hw=0.075, hd=0.2, th=0.035, ball=k['name'], rs=(0.45, 0.55, 0.15, 0.25)))
TRO = []
for tag, pre in (('armyB', 'a'), ('armyC', 'a'), ('shell', 's')):
    td = EV['troughs'][tag]
    lands = [BP[n].hits[-1] for n in PEL if n[0] == pre]
    if tag == 'armyB': lands = [h for h in lands if h['t'] < SECT['army']]
    if tag == 'armyC': lands = [h for h in lands if h['t'] >= SECT['army']]
    xx = [h['p'][0] for h in lands]; yy = [h['p'][1] for h in lands]; zz = [h['p'][2] for h in lands]
    x0, x1 = min(xx) - 0.6, max(xx) + 0.6; y0, y1 = min(yy) - 0.35, max(yy) + 0.35
    # a gently sloping snow-filled glass trough under the landing points (the bed follows the landing heights)
    XX = np.linspace(x0, x1, 60)
    srt = sorted(zip(xx, zz)); ZB = np.interp(XX, [a for a, _ in srt], [b for _, b in srt]) - 0.09
    bed = strip_mesh(XX, ZB, y0, y1, 0.06, M_SNOW, 'troughbed')
    wall1 = strip_mesh(XX, ZB + 0.12, y0 - 0.03, y0, 0.2, M_TROUGH, 'troughwall'); wall2 = strip_mesh(XX, ZB + 0.12, y1, y1 + 0.03, 0.2, M_TROUGH, 'troughwall')
    rim1 = strip_mesh(XX, ZB + 0.13, y0 - 0.05, y0 + 0.01, 0.03, M_GOLD if pre == 's' else M_CHEESED, 'troughrim')
    rim2 = strip_mesh(XX, ZB + 0.13, y1 - 0.01, y1 + 0.05, 0.03, M_GOLD if pre == 's' else M_CHEESED, 'troughrim')
    tmin = min(h['t'] for h in lands) - 2.0; tmax = max(h['t'] for h in lands) + 1.0
    for o in (bed, wall1, wall2, rim1, rim2):
        sc_ = scale_curve(tmin, tmax, 0.4, 0.5, overshoot=False)
        o.location = ((x0 + x1) / 2, (y0 + y1) / 2, float(np.mean(ZB)))
        o.data.transform(Matrix.Translation(-Vector(o.location)))
        for k in range(3): key_series(o, 'scale', k, sc_, 1e-4)
    TRO.append((tag, x0, x1, (y0 + y1) / 2, float(np.mean(ZB))))

# ---------------- cups, carts and gondolas ----------------
CABLES = []
for cp in EV['cups']:
    who = cp['who']; r_ = R_[who]; t0, t1 = cp['t'], cp['t_rel']
    p0 = np.array(cp['p']); v = np.array(cp['v_rel'])
    root = link(bpy.data.objects.new('cup', None))
    P = np.array([POS[who][f] if t0 <= T[f] <= t1 else (p0 if T[f] < t0 else POS[who][int(round(t1 * FPS))] + v * (T[f] - t1)) for f in range(NF)])
    key_loc(root, P)
    ring = obj(prim('tube', M_GLASSC, 32, 0.12), 'cupring'); ring.parent = root; ring.location = (0, 0, -r_ * 0.25); ring.scale = (r_ * 1.35, r_ * 1.35, r_ * 1.1)
    gold = obj(ring_mesh(r_ * 1.33, r_ * 1.45, 0.05, 32), 'cuprim'); gold.data.materials.append(M_GOLD if who in SOLD else M_CHEESED); gold.parent = root; gold.location = (0, 0, r_ * 0.33)
    hinge = link(bpy.data.objects.new('cuphinge', None)); hinge.parent = root; hinge.location = (-r_ * 1.3, 0, -r_ - 0.02)
    door = obj(prim('cyl', M_GOLD if who in SOLD else M_CHEESED, 24), 'cupdoor'); door.parent = hinge; door.location = (r_ * 1.3, 0, -0.015); door.scale = (r_ * 1.3, r_ * 1.3, 0.03)
    ang = np.array([0.0 if T[f] < t1 - 0.06 else min(1.0, (T[f] - t1 + 0.06) / 0.1) * 1.6 for f in range(NF)])
    hinge.rotation_mode = 'XYZ'; key_series(hinge, 'rotation_euler', 1, ang, 1e-4)
    for j, a_ in enumerate((0.0, 2.1, 4.2)):
        rod((r_ * 1.38 * math.cos(a_), r_ * 1.38 * math.sin(a_), r_ * 0.35), (0, 0, r_ * 0.35 + 0.5), 0.008, M_GOLD, 4).parent = root
    # the hanger up to the cable / rope
    if cp['kind'] == 'gondola':
        zc = float(P[int(round((t0 + t1) / 2 * FPS)), 2]) + 0.95
    else:
        zc = float(max(p0[2], P[int(round(t1 * FPS)), 2])) + 1.4
    hg = obj(prim('cyl', M_ROPE, 6), 'hanger'); hg.parent = None
    HP_ = P + np.array([0, 0, r_ * 0.35 + 0.5]); L_ = np.maximum(0.02, zc - HP_[:, 2])
    key_loc(hg, HP_ + np.stack([np.zeros(NF), np.zeros(NF), L_ / 2], 1))
    hg.scale = (0.014, 0.014, 1.0); key_series(hg, 'scale', 2, L_, 1e-4)
    wh = obj(prim('cyl', M_RED if who in SOLD else M_PURP, 16), 'trolley'); wh.rotation_euler = (math.pi / 2, 0, 0); wh.scale = (0.11, 0.11, 0.06)
    key_loc(wh, np.stack([HP_[:, 0], HP_[:, 1], np.full(NF, zc + 0.02)], 1))
    xa, xb = p0[0] - 0.8, P[int(round(t1 * FPS)), 0] + 1.0
    cab = rod((xa, p0[1], zc + 0.08), (xb, p0[1], zc + 0.08), 0.011, M_CABLE, 6); CABLES.append((xa, xb, p0[1], zc))
    sc_ = scale_curve(t0 - 0.2, t1 + 0.6, 0.15, 0.35)
    for k in range(3): key_series(root, 'scale', k, sc_, 1e-4)
    for o in (hg, wh):
        key_series(o, 'scale', 0, np.where(sc_ > 0.01, o.scale[0], 0.0001), 1e-5); key_series(o, 'scale', 1, np.where(sc_ > 0.01, o.scale[1], 0.0001), 1e-5)
    # the cable towers (gondolas): toy-like red/white pylons at both ends
    PIECES.append(dict(kind='cup', t=t0, top=p0 - [0, 0, r_], n=np.array([0, 0, 1.0]), hw=r_ * 1.3, hd=r_ * 1.3, th=0.04, ball=who, own=(t0 - 0.03, t1 + 0.45),
                       life=(t0 - 0.2, t0 - 0.03)))
# the final cups / cage / podiums (absorb the last note and keep the marble)
for n in MAIN:
    fc = EV['final_cups'][n]; p_ = np.array(fc['p']); r_ = R_[n]; t_ = fc['t']
    root = link(bpy.data.objects.new('fcup', None)); root.location = tuple(p_)
    ring = obj(prim('tube', M_GLASSC, 32, 0.12), 'fring'); ring.parent = root; ring.location = (0, 0, -r_ * 0.25); ring.scale = (r_ * 1.35, r_ * 1.35, r_ * 1.1)
    fl_ = obj(prim('cyl', M_DRUMGLOW, 32), 'ffloor'); fl_.parent = root; fl_.location = (0, 0, -r_ - 0.02); fl_.scale = (r_ * 1.3, r_ * 1.3, 0.03)
    fl_.color = (1, 1, 1, 0); flash(fl_, t_, pcol(max(VO[n][-1]['p']), 0.4), base=0.0, after=0.25)
    rim = obj(ring_mesh(r_ * 1.33, r_ * 1.45, 0.05, 32), 'frim'); rim.data.materials.append(M_GOLD if n in SOLD else M_CHEESED); rim.parent = root; rim.location = (0, 0, r_ * 0.33)
    post(p_ - np.array([0, 0, r_ + 0.05]), 0.05, M_RED if n in SOLD else M_PURP)
    if n == 'king':                                    # the cage closes over the Mouse King
        cage = link(bpy.data.objects.new('cage', None)); cage.parent = root
        for j in range(10):
            a_ = 2 * math.pi * j / 10
            rod((r_ * 1.5 * math.cos(a_), r_ * 1.5 * math.sin(a_), -r_), (r_ * 1.5 * math.cos(a_), r_ * 1.5 * math.sin(a_), r_ * 1.6), 0.015, M_STEEL, 6).parent = cage
        tp_ = obj(prim('cone', M_STEEL, 24, r_ * 1.6, 0.05, 0.3), 'cagetop'); tp_.parent = cage; tp_.location = (0, 0, r_ * 1.75)
        cz = np.array([3.0 if T[f] < t_ + 0.15 else max(0.0, 3.0 - 3.0 * min(1.0, (T[f] - t_ - 0.15) / 0.35) ** 2) for f in range(NF)])
        key_series(cage, 'location', 2, cz, 1e-4)
    sc_ = scale_curve(t_ - 0.12, 1e9, 0.1, 0.3)
    for k in range(3): key_series(root, 'scale', k, sc_, 1e-4)
    PIECES.append(dict(kind='fcup', t=t_, top=p_ - [0, 0, r_], n=np.array([0, 0, 1.0]), hw=r_ * 1.3, hd=r_ * 1.3, th=0.04, ball=n, own=(t_ - 0.03, 1e9),
                       life=(t_ - 0.12, 1e9)))

# ---------------- the start shelves: the toy fort (soldiers) and the cheese tower (mice) ----------------
for n in MAIN:
    s = EV['start'][n]; p_ = np.array(s['p']); r_ = R_[n]; tr_ = s['t_release']
    shelf = box(tuple(p_ + np.array([0.55, 0, -r_ - 0.05])), (0.75, 0.6, 0.1), M_WOOD if n in SOLD else M_CHEESE, bev=True, name='shelf')
    hinge = link(bpy.data.objects.new('trap', None)); hinge.location = tuple(p_ + np.array([-r_ * 1.1, 0, -r_ - 0.05]))
    trap = box((0, 0, 0), (r_ * 2.2, 0.6, 0.06), M_GOLD, name='trapdoor'); trap.parent = hinge; trap.location = (r_ * 1.1, 0, 0)
    hinge.rotation_mode = 'XYZ'; key_series(hinge, 'rotation_euler', 1, np.array([0.0 if T[f] < tr_ - 0.04 else min(1.0, (T[f] - tr_ + 0.04) / 0.1) * 1.5 for f in range(NF)]), 1e-4)
    post(p_ + np.array([0.75, 0.0, -r_ - 0.1]), 0.05, M_WOOD if n in SOLD else M_CHEESED)

# ---------------- scenery: the toy fort, the cheese castle, the Mouse King's gate ----------------
def crenel_wall(x0, x1, y, z0, h, m_wall, m_top, d=0.6):
    box(((x0 + x1) / 2, y, z0 + h / 2), (x1 - x0, d, h), m_wall, name='wall')
    for xc in np.arange(x0 + 0.25, x1 - 0.1, 0.7): box((xc, y, z0 + h + 0.18), (0.38, d, 0.36), m_top, name='crenel')


def tower(x, y, z0, h, r, m_body, m_roof, flag_m=None):
    o = obj(prim('cyl', m_body, 32), 'tower'); o.location = (x, y, z0 + h / 2); o.scale = (r, r, h)
    rf = obj(prim('cone', m_roof, 32, r * 1.25, 0.0, r * 1.6), 'roof'); rf.location = (x, y, z0 + h + r * 0.8)
    if flag_m is not None:
        rod((x, y, z0 + h + r * 1.5), (x, y, z0 + h + r * 1.5 + 1.0), 0.02, M_GOLD, 6)
        fl = box((x + 0.3, y, z0 + h + r * 1.5 + 0.8), (0.6, 0.02, 0.36), flag_m, name='flag')
        return fl
    return None


# toy fort (red / white) where the soldiers start
gx = ground_z(0.0)
crenel_wall(-9.0, 4.5, 6.0, gx, 4.2, M_WHITE, M_RED, 0.8)
for xt_ in (-8.0, -3.0, 3.5): tower(xt_, 6.0, gx, 5.3, 0.85, M_RED, M_BLUE, M_RED)
box((-2.0, 5.5, gx + 1.3), (1.8, 0.2, 2.6), M_WOOD, name='gate')
# cheese castle (B and the army)
xb0 = xs(SECT['mice']) - 4.0; xb1 = xs(SECT['finale']) - 6.0
for xc in np.arange(xb0, xb1, 3.2):
    gz = ground_z(xc); hh = rng.uniform(3.5, 6.0)
    blk = box((xc, rng.uniform(6.2, 7.4), gz + hh / 2), (rng.uniform(2.2, 3.0), 1.2, hh), M_CHEESE, bev=True, name='cheese')
    for k in range(5):
        hole = sphere((xc + rng.uniform(-1.0, 1.0), 5.45, gz + rng.uniform(0.6, hh - 0.4)), rng.uniform(0.15, 0.35), M_CHEESED, (12, 6), 'hole')
        hole.scale[1] = 0.25
    if rng.random() < 0.45: tower(xc + 0.6, 7.6, gz + hh, 1.4, 0.45, M_CHEESE, M_PURP, M_PURP)
# the Mouse King's gate at the end: its flag flips at the final chord
xg = EV['final_cups']['cap']['p'][0] + 0.9; gz = ground_z(xg)
crenel_wall(xg - 6.0, xg + 6.0, 6.5, gz, 4.6, M_CHEESE, M_CHEESED, 1.0)
for xt_ in (xg - 5.0, xg + 5.0): tower(xt_, 6.5, gz, 6.0, 0.9, M_CHEESE, M_PURP, None)
pole_top = (xg, 6.4, gz + 9.2)
rod((xg, 6.4, gz + 4.6), pole_top, 0.04, M_GOLD, 8)
FLAG_M = link(bpy.data.objects.new('flagmouse', None)); FLAG_S = link(bpy.data.objects.new('flagsold', None))
fm = box((xg + 1.25, 6.38, gz + 8.3), (2.4, 0.03, 1.5), M_PURP, name='flagm'); fm.parent = FLAG_M
for sx in (-1, 1):
    e_ = obj(prim('cyl', M_EARO, 16), 'flagear'); e_.parent = FLAG_M; e_.location = (xg + 1.25 + sx * 0.42, 6.34, gz + 8.55); e_.rotation_euler = (math.pi / 2, 0, 0); e_.scale = (0.3, 0.3, 0.02)
fs = box((xg + 1.25, 6.38, gz + 8.3), (2.4, 0.03, 1.5), M_RED, name='flags'); fs.parent = FLAG_S
box((xg + 1.25, 6.35, gz + 8.3), (2.4, 0.03, 0.36), M_WHITE, name='flagstripe').parent = FLAG_S
st2 = obj(prim('star', M_GOLD, 0.2), 'flagstar'); st2.parent = FLAG_S; st2.location = (xg + 1.25, 6.3, gz + 8.3); st2.scale = (0.45, 0.45, 0.45)
sw = np.array([1.0 if T[f] < FINAL + 0.1 else max(0.001, 1 - (T[f] - FINAL - 0.1) / 0.3) for f in range(NF)])
for k in range(3): key_series(FLAG_M, 'scale', k, sw, 1e-3)
FLAG_M.location = (0, 0, 0)
up = np.array([0.0 if T[f] < FINAL + 0.35 else min(1.0, (T[f] - FINAL - 0.35) / 0.6) for f in range(NF)])
key_series(FLAG_S, 'location', 2, -2.2 * (1 - up), 1e-4); key_series(FLAG_S, 'scale', 2, np.maximum(0.001, up), 1e-3)

# far background: a giant Christmas tree with big warm bokeh lights, presents, the window
for xt_ in np.arange(-20, FLX[-1] + 20, 34.0):
    gz = ground_z(xt_) - 1.0
    for k, (r0, z0, h) in enumerate(((7.5, 0.0, 9.0), (6.2, 5.5, 8.0), (4.8, 10.5, 7.0), (3.2, 15.0, 6.0))):
        c_ = obj(prim('cone', M_TREE, 40, r0, 0.0, h), 'tree'); c_.location = (xt_, 24.0, gz + z0 + h / 2)
    obj(prim('cyl', M_TRUNK, 16), 'trunk').location = (xt_, 24.0, gz - 1.0)
    for k in range(70):
        zz = rng.uniform(1.0, 20.0); rr = max(0.5, 7.5 * (1 - zz / 21.5)); a_ = rng.uniform(-1.5, 1.5)
        s_ = sphere((xt_ + rr * math.sin(a_), 24.0 - rr * math.cos(a_) - 0.2, gz + zz), rng.uniform(0.12, 0.22), M_FX, (8, 4), 'treelight')
        s_.color = (*[(1.0, 0.75, 0.35), (1.0, 0.3, 0.25), (0.4, 0.7, 1.0), (1.0, 0.9, 0.6), (0.5, 1.0, 0.5)][k % 5], 0)
        key_series(s_, 'color', 3, np.round(0.55 + 0.35 * np.sin(rng.uniform(1.0, 3.0) * T + rng.uniform(0, 6.28)), 2), 0.03)
    for k in range(5):
        xx_ = xt_ + rng.uniform(-9, 9); sz = rng.uniform(1.2, 2.4)
        pb = box((xx_, rng.uniform(15.0, 18.0), ground_z(xx_) + sz / 2 - 0.2), (sz, sz, sz), M_OBJ, bev=True, name='present')
        pb.color = (*[(0.8, 0.1, 0.1), (0.1, 0.4, 0.2), (0.15, 0.25, 0.7), (0.85, 0.65, 0.2)][k % 4], 1)
        box((xx_, rng.uniform(15.0, 18.0) * 0 + pb.location[1], pb.location[2]), (sz * 1.01, sz * 0.18, sz * 1.01), M_GOLD, name='ribbon')
for k in range(300):
    x0 = rng.uniform(-20, FLX[-1] + 5); y0 = rng.uniform(-4.0, 10.0); z_top = ground_z(x0) + rng.uniform(3, 16)
    fl_ = sphere((x0, y0, z_top), rng.uniform(0.02, 0.045), M_FX, (6, 3), 'flake'); fl_.color = (1.0, 1.0, 1.0, 0.3)
    vz = -rng.uniform(0.3, 0.5); vx = rng.uniform(-0.12, 0.08)
    fl_.location = (x0, y0, z_top); fl_.keyframe_insert('location', frame=0)
    fl_.location = (x0 + vx * T[-1], y0, z_top + vz * T[-1]); fl_.keyframe_insert('location', frame=NF - 1)

# ---------------- blimps: the cheese blimp (mouse army) and the soldiers' balloon (golden shells) ----------------
def blimp(name, body_m, fin_m, length=2.6, rad=0.55):
    root = link(bpy.data.objects.new(name, None))
    b_ = obj(prim('sph', body_m, 32, 16), 'blimpbody'); b_.parent = root; b_.location = (0, 0, rad + 0.25); b_.scale = (length / 2, rad, rad)
    for sz_ in (-1, 1):
        f_ = box((0, 0, 0), (0.45, 0.04, 0.4), fin_m, name='fin'); f_.parent = root; f_.location = (-length / 2 + 0.15, 0, rad + 0.25 + sz_ * 0.35)
    g_ = box((0, 0, 0), (1.9, 0.5, 0.18), fin_m, bev=True, name='gondola'); g_.parent = root; g_.location = (0, 0, 0.12)
    for dx in (-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75):
        hh = obj(prim('cyl', M_BLACK, 16), 'hatch'); hh.parent = root; hh.location = (dx, 0, 0.02); hh.scale = (0.1, 0.1, 0.02)
    return root


BL_A = blimp('blimp_army', M_CHEESE, M_PURP); BL_S = blimp('balloon_sold', M_RED, M_GOLD)
for sx in (-1, 1):
    e_ = obj(prim('cyl', M_EARO, 24), 'blimpear'); e_.parent = BL_A; e_.location = (1.0, sx * 0.35, 1.25); e_.rotation_euler = (math.pi / 2, 0, sx * 0.4); e_.scale = (0.3, 0.3, 0.04)
st_ = box((0, 0, 0), (0.6, 0.05, 0.6), M_WHITE, name='star'); st_.parent = BL_S; st_.location = (0, -0.56, 0.8); st_.rotation_euler = (0, math.pi / 4, 0)
BLM = EV['blimps']
def blimp_track(segs, park):
    """segs: list of (t0, t1, y, z, dx); the hatch line flies at x = xs(t) + dx between t0 and t1; it glides in/out"""
    P = np.zeros((NF, 3))
    for f in range(NF):
        t = T[f]; p = None
        for k, (a, b, y, z, dx) in enumerate(segs):
            if a <= t <= b: p = (xs(t) + dx, y, z); break
        if p is None:
            # between / outside: glide smoothly (in from the right-top before, out to the left-top after)
            before = [s for s in segs if s[0] > t]; after = [s for s in segs if s[1] < t]
            if before and after:
                a = after[-1]; b = before[0]; u = smoother((t - a[1]) / (b[0] - a[1]))
                p = (xs(t) + a[4] + (b[4] - a[4]) * u, a[2] + (b[2] - a[2]) * u, a[3] + (b[3] - a[3]) * u + 1.2 * math.sin(math.pi * u))
            elif before:
                b = before[0]; u = smoother((b[0] - t) / 2.5); p = (xs(t) + b[4] + 3.0 * u, b[2], b[3] + 3.5 * u)
            else:
                a = after[-1]; u = smoother((t - a[1]) / 2.5); p = (xs(t) + a[4] - 4.0 * u, a[2], a[3] + 3.5 * u)
        P[f] = p
    return P
def smoother(u): u = min(1.0, max(0.0, u)); return u * u * u * (u * (6 * u - 15) + 10)
PA = blimp_track([(BLM['armyB']['t0'] - 0.3, BLM['armyB']['t1'] + 0.3, BLM['armyB']['y'], BLM['armyB']['z'], BLM['armyB']['dx']),
                  (BLM['armyC']['t0'] - 0.3, BLM['armyC']['t1'] + 0.3, BLM['armyC']['y'], BLM['armyC']['z'], BLM['armyC']['dx'])], None)
PS = blimp_track([(BLM['shell']['t0'] - 0.3, BLM['shell']['t1'] + 0.3, BLM['shell']['y'], BLM['shell']['z'], BLM['shell']['dx'])], None)
key_loc(BL_A, PA); key_loc(BL_S, PS)

# ---------------- signboards (search keywords: English first, Japanese below) ----------------
M_SIGNF = mat('signface', (1.0, 0.95, 0.86), 0.6)
M_INKR = mat('inkred', (0.6, 0.05, 0.08), 0.5); M_INKB = mat('inkblue', (0.08, 0.15, 0.45), 0.5); M_INKK = mat('inkbrown', (0.35, 0.2, 0.08), 0.5)
def banner(lines, x, w, h, zc, y, frame_m):
    box((x, y + 0.06, zc), (w, 0.12, h), frame_m, bev=True, name='sign')
    box((x, y - 0.01, zc), (w - 0.22, 0.04, h - 0.22), M_SIGNF, name='signface')
    tot_h = sum(sz * 1.35 for _, sz, _ in lines); zz = zc + tot_h / 2
    for s_, size, mt in lines:
        zz -= size * 1.35
        fit_w(text(s_, (x, y - 0.06, zz + size * 0.67), size, mt, rot=(math.pi / 2, 0, 0)), w - 0.45)
SIGNS = ((2.6, 5.4, 2.0, [('The Nutcracker', 0.46, M_INKR), ('March', 0.5, M_INKB), ('くるみ割り人形「行進曲」', 0.24, M_INKK)]),
         (8.4, 5.4, 1.9, [('Tchaikovsky', 0.52, M_INKR), ('Classical music · Ballet', 0.3, M_INKB), ('チャイコフスキー　バレエ音楽', 0.24, M_INKK)]),
         (14.2, 5.4, 1.9, [('Marble Machine', 0.5, M_INKB), ('Satisfying · ASMR', 0.32, M_INKR), ('ビー玉マシン　からくり', 0.24, M_INKK)]),
         (20.6, 5.4, 1.9, [('Mouse King', 0.5, M_INKK), ('vs Toy Soldiers', 0.36, M_INKR), ('ネズミの王様 vs おもちゃの兵隊', 0.22, M_INKB)]),
         (26.8, 5.4, 1.9, [('Christmas Music', 0.48, M_INKR), ('Classical marble race', 0.3, M_INKB), ('クリスマス　クラシック', 0.24, M_INKK)]))
for tb, w_, h_, lines in SIGNS:
    xx = xs(tb) + 0.4; zc = zs(tb) + 5.35
    banner(lines, xx, w_, h_, zc, 5.0, M_RED if tb < 15 else M_CHEESED)
    for sx in (-1, 1): rod((xx + sx * (w_ / 2 - 0.4), 5.12, zc - h_ / 2), (xx + sx * (w_ / 2 - 0.4), 5.12, ground_z(xx) - 0.2), 0.06, M_GOLD if tb < 15 else M_CHEESED, 8)

# ---------------- the score ----------------
SC = EV['score_events']
TS_S = sorted(t for t, s in SC if s == 'S'); TS_M = sorted(t for t, s in SC if s == 'M')
CNT_S = np.array([int(np.searchsorted(TS_S, t + 1e-6, side='right')) for t in T])
CNT_M = np.array([int(np.searchsorted(TS_M, t + 1e-6, side='right')) for t in T])
print('score final', CNT_S[-1], CNT_M[-1], 'before final chord', int(np.searchsorted(TS_S, FINAL - 1e-3)), int(np.searchsorted(TS_M, FINAL - 1e-3)))

# ---------------- world & lights ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes['Background']
_wt = w.node_tree; _tc2 = _wt.nodes.new('ShaderNodeTexCoord'); _sp = _wt.nodes.new('ShaderNodeSeparateXYZ'); _cr = _wt.nodes.new('ShaderNodeValToRGB')
_wt.links.new(_tc2.outputs['Generated'], _sp.inputs[0]); _cr.color_ramp.interpolation = 'EASE'; _wt.links.new(_sp.outputs['Z'], _cr.inputs['Fac'])
_cr.color_ramp.elements[0].position = 0.02; _cr.color_ramp.elements[0].color = (0.16, 0.06, 0.08, 1)
_cr.color_ramp.elements[1].position = 0.3; _cr.color_ramp.elements[1].color = (0.015, 0.02, 0.07, 1)
_wt.links.new(_cr.outputs['Color'], bg.inputs['Color']); bg.inputs['Strength'].default_value = 0.9
mn = bpy.data.lights.new('moon', 'SUN'); mn.color = (0.65, 0.72, 1.0); mn.angle = 0.08; mn.energy = 0.6
mno = link(bpy.data.objects.new('moon', mn)); mno.rotation_euler = (math.radians(50), math.radians(-12), math.radians(-25))

# ---------------- camera: follow the stage; a little wider for the army; at the end pull back to the gate ----------------
FOC = np.zeros((NF, 3)); WID = np.zeros(NF)
for f in range(NF):
    t = T[f]
    FOC[f] = np.array([xs(min(t, FINAL)) + 0.35, 0, zs(min(t, FINAL)) + 3.0]); WID[f] = 5.7
    ua = smoother((t - (SECT['army'] - 1.0)) / 1.5) * (1 - smoother((t - (SECT['finale'] - 1.2)) / 1.5))
    WID[f] += 0.5 * ua; FOC[f, 0] += 0.35 * ua
    if t < T0 + 0.3:                                   # the opening: a little wider on the fort
        u0 = smoother((T0 + 0.3 - t) / 1.2); WID[f] += 0.8 * u0
    if t > FINAL - 0.2:
        u2 = smoother((t - FINAL + 0.2) / 2.4)
        FOC[f] = (1 - u2) * FOC[f] + u2 * np.array([xg - 0.3, 0, zs(FINAL) + 4.3]); WID[f] = (1 - u2) * WID[f] + u2 * 8.2
FOC[:, 0] = smooth(FOC[:, 0], 0.4); FOC[:, 2] = smooth(FOC[:, 2], 0.7); WID = smooth(WID, 0.45)
LENS = 32.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 500
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24
    CP[f] = FOC[f] + np.array([0, -D, 0.14 * D]); TP[f] = FOC[f] + np.array([0, 0, 0.2])
key_loc(cam, CP); key_loc(tgt, TP)
for k, (dx, dz, en, col) in enumerate(((-3.0, 6.0, 1300, (1.0, 0.8, 0.6)), (3.5, 5.0, 900, (1.0, 0.72, 0.6)), (0.0, 1.0, 380, (0.7, 0.78, 1.0)))):
    al = bpy.data.lights.new(f'fill{k}', 'AREA'); al.size = 6.0; al.color = col; al.energy = en
    ao = link(bpy.data.objects.new(f'fill{k}', al))
    key_loc(ao, np.stack([FOC[:, 0] + dx, np.full(NF, -4.5 if k < 2 else -6.0), FOC[:, 2] + dz], 1))
    ao.rotation_euler = (math.radians(-28), 0, 0) if k < 2 else (math.radians(-80), 0, 0)
# purple mood light while the mice rule (B and the army)
pl_ = bpy.data.lights.new('mouselight', 'AREA'); pl_.size = 8.0; pl_.color = (0.6, 0.35, 1.0)
plo = link(bpy.data.objects.new('mouselight', pl_)); key_loc(plo, np.stack([FOC[:, 0] + 1.0, np.full(NF, 3.0), FOC[:, 2] + 7.0], 1)); plo.rotation_euler = (0, 0, 0)
mood = np.array([800.0 * smoother((t - SECT['mice'] + 0.6) / 1.0) * (1 - smoother((t - SECT['finale'] + 0.8) / 1.0)) for t in T])
for f in range(0, NF, 3):
    pl_.energy = float(mood[f]); pl_.keyframe_insert('energy', frame=f)

# ---------------- fireworks & confetti on the final chord ----------------
FW_C = [np.array([xg - 3.0, 4.0, zs(FINAL) + 8.0]), np.array([xg + 1.5, 4.5, zs(FINAL) + 9.0]), np.array([xg - 0.5, 3.5, zs(FINAL) + 7.0])]
for k in range(240):
    c0 = FW_C[k % 3]; tb_ = FINAL + 0.25 * (k % 3); d_ = rng.normal(0, 1, 3); d_ /= np.linalg.norm(d_); v = d_ * rng.uniform(2.5, 4.0)
    s_ = sphere(tuple(c0), rng.uniform(0.07, 0.12), M_FX2, (6, 3), 'spark'); s_.color = (*[(1.0, 0.25, 0.2), (1.0, 0.8, 0.2), (0.3, 0.6, 1.0), (0.4, 1.0, 0.4), (1.0, 0.4, 0.9)][k % 5], 0)
    P = np.array([c0 + v * max(0, t - tb_) + np.array([0, 0, -0.5 * 1.5 * max(0, t - tb_) ** 2]) for t in T])
    key_loc(s_, P); key_series(s_, 'color', 3, np.where(T < tb_, 0.0, np.clip(1.3 - (T - tb_) / 1.6, 0, 1)), 1e-3)
for k in range(120):
    p0 = np.array([xg + rng.uniform(-4, 3), rng.uniform(-1.5, 2.5), zs(FINAL) + rng.uniform(8.5, 10.5)])
    s_ = box(tuple(p0), (0.12, 0.02, 0.07), M_OBJ, name='confetti'); s_.color = (*pcol(int(rng.integers(0, 12)), 0.2), 1)
    vz = -rng.uniform(0.6, 1.1); vx = rng.uniform(-0.3, 0.3); w_ = rng.uniform(4, 9)
    P = np.array([p0 + np.array([vx * max(0, t - FINAL) + 0.15 * math.sin(w_ * t), 0, vz * max(0, t - FINAL) + (0 if t > FINAL else 30)]) for t in T])
    key_loc(s_, P); s_.rotation_mode = 'XYZ'; key_series(s_, 'rotation_euler', 0, w_ * T * 0.5, 1e-3)

# ---------------- HUD (rides with the camera) ----------------
HUD_D = 3.0
def hud_text(s, x, y, size, m, align='CENTER'):
    o = text(s, (x, y, -HUD_D), size, m, rot=(0, 0, 0), align=align, ext=0.0); o.parent = cam; return o
def vis(o, a):
    s0 = tuple(o.scale)
    for c in range(3): key_series(o, 'scale', c, np.maximum(0.001, a * s0[c]), 1e-3)
M_HUDW = emis('hudw', (1.0, 0.97, 0.94), 3.0); M_HUDG = emis('hudg', (1.0, 0.8, 0.4), 4.0); M_HUDR = emis('hudr', (1.0, 0.3, 0.25), 4.0)
M_HUDY = emis('hudy', (1.0, 0.82, 0.25), 4.0); M_HUDP = emis('hudp', (0.8, 0.6, 1.0), 4.0)
M_HUDBG = mat('hudbg', (0.03, 0.02, 0.06), 1.0, **{'Alpha': 0.55})
def window(t0, t1, fade=0.3): return np.array([0.0 if t < t0 else min(1.0, (t - t0) / fade) * (1.0 if t < t1 else max(0.0, 1 - (t - t1) / fade)) for t in T])
def hud_bg(x, y, sx, sy, a):
    b_ = obj(prim('plane', M_HUDBG), 'hudbg'); b_.parent = cam; b_.location = (x, y, -HUD_D - 0.01); b_.scale = (sx, sy, 1); vis(b_, a)
hook_a = window(-1, 4.2)
vis(fit_w(hud_text('Toy Soldiers vs Mouse King', 0, 0.97, 0.082, M_HUDW), 0.98), hook_a)
vis(fit_w(hud_text('Who wins? Count every note!', 0, 0.875, 0.078, M_HUDG), 0.98), hook_a)
vis(hud_text('おもちゃの兵隊 vs ネズミの王様　どっちが勝つ？', 0, 0.8, 0.04, M_HUDW), hook_a)
hud_bg(0, 0.89, 0.8, 0.13, hook_a)
cnt_a = window(4.2, 1e9)
YC = 0.93
vis(hud_text('SOLDIERS', -0.27, YC + 0.075, 0.05, M_HUDR), cnt_a); vis(hud_text('MICE', 0.27, YC + 0.075, 0.05, M_HUDY), cnt_a)
vis(hud_text('兵隊', -0.27, YC - 0.075, 0.032, M_HUDW), cnt_a); vis(hud_text('ネズミ', 0.27, YC - 0.075, 0.032, M_HUDW), cnt_a)
vis(hud_text(':', 0, YC, 0.1, M_HUDW), cnt_a)
hud_bg(0, YC, 0.62, 0.115, cnt_a)
DX = 0.07
for team, CNT, x0, mt in (('S', CNT_S, -0.34, M_HUDR), ('M', CNT_M, 0.2, M_HUDY)):
    for pos in range(3):
        dv = (CNT // (10 ** (2 - pos))) % 10
        for dgt in range(10):
            o = hud_text(str(dgt), x0 + DX * pos, YC, 0.1, M_HUDW)
            vis(o, ((dv == dgt)).astype(float) * cnt_a)
def caption(en, jp, t0, t1, m=M_HUDG, y=0.72):
    a = window(t0, t1)
    vis(fit_w(hud_text(en, 0, y, 0.085, m), 0.98), a); vis(hud_text(jp, 0, y - 0.075, 0.042, M_HUDW), a); hud_bg(0, y - 0.03, 0.8, 0.085, a)
caption('The Mouse King attacks!', 'ネズミの王様が来た！', SECT['mice'] - 0.6, SECT['mice'] + 2.6, M_HUDP)
caption('The grand staircase!', '王様の大階段！', E['bar_t']['22'] + 1.0, E['bar_t']['24'] + 0.9, M_HUDP)
n_army_c = sum(1 for k in EV['pellet_keys'] if k['name'][0] == 'a' and k['t'] >= SECT['army'] - 0.5)
caption(f'Mouse army: {n_army_c} marbles!', f'ネズミの大軍 {n_army_c}匹！', SECT['army'] - 0.3, SECT['army'] + 3.0, M_HUDY)
caption('Mice are winning...', 'ネズミが逆転…！', SECT['army'] + 7.5, SECT['army'] + 10.5, M_HUDY)
caption('Soldiers strike back!', '兵隊の反撃！', SECT['finale'] - 0.6, SECT['finale'] + 2.6, M_HUDR)
caption('Last chord decides!', '最後の和音で決まる！', FINAL - 3.2, FINAL - 0.4, M_HUDG)
end_a = window(FINAL + 0.9, 1e9, 0.4)
vis(fit_w(hud_text(f'SOLDIERS WIN {CNT_S[-1]}:{CNT_M[-1]}', 0, -0.5, 0.1, M_HUDR), 0.98), end_a)
vis(fit_w(hud_text('...by ONE note!', 0, -0.61, 0.085, M_HUDG), 0.98), end_a)
vis(fit_w(hud_text('The Nutcracker · March (Tchaikovsky)', 0, -0.7, 0.05, M_HUDW), 0.98), end_a)
vis(hud_text('1音差で兵隊の勝ち！　くるみ割り人形「行進曲」', 0, -0.77, 0.036, M_HUDW), end_a)
hud_bg(0, -0.635, 0.82, 0.19, end_a)

EV_OUT = dict(pieces=[dict(kind=p['kind'], t=p['t'], top=list(map(float, p['top'])), n=list(map(float, p['n'])), hw=float(p['hw']), hd=float(p['hd']), th=float(p['th']), ball=p['ball'],
                           rs=p.get('rs'), own=p.get('own'), life=p.get('life')) for p in PIECES], counter=dict(S=int(CNT_S[-1]), M=int(CNT_M[-1])))
if MODE == 'pieces': json.dump(EV_OUT, open(f'{HERE}/pieces.json', 'w'))

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
        f = int(round(float(tt_) * FPS))
        print('t', tt_, 'FOC', np.round(FOC[f], 2), 'W', round(WID[f], 2), {n: np.round(POS[n][f], 2).tolist() for n in MAIN}, 'score', CNT_S[f], CNT_M[f])
