"""Dance of the Sugar Plum Fairy - 'Can one marble light up the Candy Castle?'.  Side view of a candy castle on a snowy
Christmas night.  Every celesta note = the fairy marble on a glass key (it lights up and stays lit) or a sugar pearl on a
crystal key / a tree ornament; every pizzicato chord = the peppermint marble on a peppermint disc; every bass note = the
chocolate truffle on a chocolate drum (runs = it rolls down a little xylophone ramp).
layout.py plans all motion (constant gravity, bounces that never add speed, cups, the lift); this file builds and animates it.
Fonts: en_bold.ttf = Poppins Bold (SIL OFL); jp_sub.otf = Noto Sans CJK JP Bold (SIL OFL) subset (make_font.py).
python3 scene.py -- test t1 t2 ... | range f0 f1 | probe t ... | pieces"""
import os, sys, json, math, time
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); LAY = json.load(open(f'{HERE}/layout.json'))
FA = json.load(open(f'{HERE}/fairy.json')); MI = json.load(open(f'{HERE}/mint.json')); CO = json.load(open(f'{HERE}/cocoa.json'))
FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.6) * FPS); T = np.arange(NF) / FPS
G = LAY['G']; EV = LAY['ev']; LN = LAY['lanes']; R_ = LAY['R']
RF, RM, RC, RP = R_['fairy'], R_['mint'], R_['cocoa'], R_['pearl']
rng = np.random.default_rng(7)
ST_T, ST_Z = np.array(EV['stage'][0]), np.array(EV['stage'][1])
def zs(t): return float(np.interp(t, ST_T, ST_Z))
SX_T, SX_X = np.array(EV['stage_x'][0]), np.array(EV['stage_x'][1])
def xs(t): return float(np.interp(t, SX_T, SX_X))
SECT = {s: t for s, t in E['secs']}

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
PEARLN = sorted(n for n in BP if n.startswith('p'))


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


# ---------------- materials ----------------
M_GOLD = mat('gold', (1.0, 0.78, 0.38), 0.25, **{'Metallic': 0.9})
M_GOLDD = mat('goldd', (0.6, 0.42, 0.18), 0.35, **{'Metallic': 0.85})
M_THREAD = mat('thread', (0.7, 0.55, 0.3), 0.4, **{'Metallic': 0.8})
M_FAIRY = glowmat('fairyball', gain=8.0, base=1.6, rough=0.08, coat=1.0)
M_MINT = stripes('mintball', (1.0, 0.98, 0.96), (0.92, 0.08, 0.16), scale=4.0, axis='Z', twist=1.6, rough=0.12, coat=1.0)
M_COCOA = mat('cocoaball', (0.24, 0.12, 0.06), 0.28, **{'Coat Weight': 0.9, 'Coat Roughness': 0.15})
M_PEARL = glowmat('pearl', gain=5.0, base=0.9, rough=0.15, coat=0.8)
M_KEY = glowmat('glasskey', gain=7.0, base=0.15, rough=0.12, coat=1.0)
M_DISC = stripes('peppermint', (1.0, 0.97, 0.95), (0.9, 0.05, 0.15), scale=6.0, radial=True, rough=0.2, coat=1.0)
M_CHIME = glowmat('chime', gain=6.0, base=0.0, rough=0.2, metal=0.8)
M_DRUM = mat('choc', (0.28, 0.14, 0.06), 0.35, **{'Coat Weight': 0.7})
M_CREAM = mat('cream', (1.0, 0.93, 0.82), 0.5, **{'Sheen Weight': 0.4})
M_SNOW = mat('snow', (0.92, 0.95, 1.0), 0.6, **{'Subsurface Weight': 0.25, 'Sheen Weight': 0.6, 'Emission Color': (0.55, 0.62, 0.9), 'Emission Strength': 0.04})
M_GINGER = mat('gingerbread', (0.55, 0.3, 0.13), 0.7)
M_ICING = mat('icing', (1.0, 0.98, 0.96), 0.4, **{'Coat Weight': 0.5})
M_PINK = mat('pinkfrost', (1.0, 0.62, 0.78), 0.45, **{'Coat Weight': 0.5})
M_MINTF = mat('mintfrost', (0.55, 0.92, 0.8), 0.45, **{'Coat Weight': 0.5})
M_LILAC = mat('lilac', (0.72, 0.6, 1.0), 0.45, **{'Coat Weight': 0.5})
M_CANE = stripes('cane', (1.0, 0.98, 0.95), (0.9, 0.06, 0.12), scale=5.0, axis='Z', twist=0.0, rough=0.2, coat=1.0)
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


M_GLASS = mat('glass', (0.85, 0.92, 1.0), 0.05, **{'Transmission Weight': 0.9, 'IOR': 1.2, 'Coat Weight': 1.0, 'Alpha': 0.55})
M_GLASSC = glassfx('glasscup', (0.85, 0.8, 1.0), 0.18, 0.35)
M_PORC = mat('porcelain', (0.98, 0.97, 0.95), 0.2, **{'Coat Weight': 1.0})
M_TREE = mat('fir', (0.06, 0.32, 0.16), 0.75, **{'Sheen Weight': 0.5})
M_TRUNK = mat('trunk', (0.3, 0.17, 0.08), 0.8)
M_OBJ = objcol('objcol', 0.4, coat=0.6)
M_FX = fxmat('fx', 8.0)
M_WIN = glowmat('window', gain=7.0, base=0.0, rough=0.5)
M_CLOUD = mat('cloud', (0.95, 0.95, 1.0), 0.9, **{'Emission Color': (0.8, 0.8, 1.0), 'Emission Strength': 0.2, 'Sheen Weight': 0.6})
M_MOON = mat('moonmat', (1.0, 0.92, 0.65), 0.4, **{'Emission Color': (1.0, 0.88, 0.55), 'Emission Strength': 2.5})


# ---------------- balls ----------------
BALL = {}
for n, mm, rr in (('fairy', M_FAIRY, RF), ('mint', M_MINT, RM), ('cocoa', M_COCOA, RC)):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=rr, segments=32, ring_count=16); o = bpy.context.object; o.name = n
    for p in o.data.polygons: p.use_smooth = True
    o.data.materials.append(mm)
    key_loc(o, POS[n])
    sp = np.zeros(NF)
    for f in range(1, NF): sp[f] = sp[f - 1] + (POS[n][f, 0] - POS[n][f - 1, 0]) / rr
    o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, sp)
    BALL[n] = o
BALL['fairy'].color = (1.0, 0.45, 0.95, 0.3)
st = obj(prim('star', M_GOLD, 0.3), 'fairycore'); st.parent = BALL['fairy']; st.scale = (RF * 0.6, RF * 0.6, RF * 0.6)
fl = bpy.data.lights.new('fairylight', 'POINT'); fl.color = (1.0, 0.6, 1.0); fl.shadow_soft_size = 0.2; fl.energy = 35
flo = link(bpy.data.objects.new('fairylight', fl)); flo.parent = BALL['fairy']
# a sparkle trail behind the fairy
for j in range(10):
    s_ = sphere((0, 0, 0), 0.05 * (1 - j / 12), M_FX, (8, 4), 'trail'); s_.color = (1.0, 0.7, 1.0, 0.6 * (1 - j / 10))
    lag = 2 + 2 * j
    P = np.array([POS['fairy'][max(0, f - lag)] for f in range(NF)]) + np.array([0, 0.05, 0])
    key_loc(s_, P)
# sugar pearls: hidden until they appear in the sieve / cloud, glowing after they land
for n in PEARLN:
    b = BP[n]; hp = [h for h in b.hits if h['kind'] == 'pearl'][0]; p_ = hp['pitch']
    rp_ = b.r; o = obj(prim('sph', M_PEARL, 16, 8), n); o.scale = (rp_, rp_, rp_)
    key_loc(o, POS[n])
    t_rel = [g for g in b.seg if g['k'] == 'fly'][0]['t0']; t_show = EV['pearl_show'][n]
    t_out = b.hits[-1]['t'] - 0.04 if n.startswith('pc') else hp['t'] + 0.25
    s_ = scale_curve(t_show, t_out, min(0.2, t_rel - t_show), 0.13 if n.startswith('pc') else 0.2, overshoot=False) * rp_
    for k in range(3): key_series(o, 'scale', k, s_, 1e-5)
    col = pcol(p_, 0.0); o.color = (*col, 0.45); flash(o, hp['t'], col, base=0.45, after=0.6)

# ---------------- ground: snow, always stepping down ----------------
pts = []
for h in BP['cocoa'].hits:
    if h['kind'] in ('drum', 'rampland', 'roll', 'cup', 'catch', 'lever', 'cupfinal'):
        pts.append((h['p'][0], h['p'][2] - RC - (1.0 if h['kind'] in ('drum', 'cupfinal') else 1.4)))
tr = EV['tree']; pts += [(tr['x'] + dx, tr['z0']) for dx in np.linspace(-4, 4, 9)]
pts.sort()
GX = np.array([p[0] for p in pts]); GZv = np.array([p[1] for p in pts])
FLX = np.arange(-16.0, GX.max() + 30, 0.25)
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


strip_mesh(FLX, FLZ, -6.0, 7.5, 1.2, M_SNOW, 'snowfield', Z1=FLZ + 0.4)
# snow drifts (soft bumps) along the front
for x0 in np.arange(-14, FLX[-1], 1.7):
    sphere((x0 + rng.uniform(-0.5, 0.5), rng.uniform(-4.5, -2.8), ground_z(x0) - 0.35), rng.uniform(0.5, 0.9), M_SNOW, (14, 7), 'drift').scale[2] *= 0.35

# ---------------- fairy keys: glass keys hanging on gold threads; they light up and stay lit ----------------
PIECES = []
LIT_T = []                                                   # every lit light (for the counter)
KEYS = []
for h in BP['fairy'].hits:
    if h['kind'] != 'key': continue
    nv = np.array(h['n']); top = np.array(h['p']) - RF * nv; p = h['pitch']; col = pcol(p, 0.08)
    k_ = obj(prim('bcube', M_KEY, 0.25), 'glasskey')
    k_.matrix_world = Matrix.Translation(Vector(top - nv * 0.04)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((0.4, 0.38, 0.08, 1))
    k_.color = (*col, 0.06); flash(k_, h['t'], col, nv, base=0.06, after=0.45)
    th1 = rod(top + np.array([-0.17, 0.12, 0.0]), top + np.array([0.0, 0.12, 4.5]), 0.005, M_THREAD, 4)
    th2 = rod(top + np.array([0.17, 0.12, 0.0]), top + np.array([0.0, 0.12, 4.5]), 0.005, M_THREAD, 4)
    gem = obj(prim('star', M_KEY, 0.25), 'keystar'); gem.location = tuple(top + np.array([0, 0.15, 0.28])); gem.scale = (0.1, 0.1, 0.1)
    gem.color = (*col, 0.0); flash(gem, h['t'], col, base=0.0, after=0.6)
    stay_ = 0.6 if any(abs(h['t'] - x) < 1e-6 for x in EV['fairy_temp_keys']) else 1e6
    riser([k_, th1, th2, gem], h['t'], top, lead=1.1, stay=stay_)
    PIECES.append(dict(kind='key', t=h['t'], top=top, n=nv, hw=0.2, hd=0.19, th=0.04, ball='fairy', rs=(1.1, stay_, 0.3, 0.35)))
    LIT_T.append(h['t']); KEYS.append((top, h['t'], col))

# ---------------- mint: peppermint discs with a little bell per chord note ----------------
for h in BP['mint'].hits:
    if h['kind'] != 'chord': continue
    nv = np.array(h['n']); top = np.array(h['p']) - RM * nv; pp = h['pitch']
    M = Matrix.Translation(Vector(top)) @ frame_from_n(nv).to_4x4()
    d_ = obj(prim('cyl', M_DISC, 40), 'disc'); d_.matrix_world = M @ Matrix.Translation((0, 0, -0.06)) @ Matrix.Diagonal((0.31, 0.31, 0.12, 1))
    rim = obj(prim('cyl', M_GLASSC, 40), 'discrim'); rim.matrix_world = M @ Matrix.Translation((0, 0, -0.065)) @ Matrix.Diagonal((0.33, 0.33, 0.1, 1))
    objs = [d_, rim]
    for j, p_ in enumerate(pp):
        bx = (j - (len(pp) - 1) / 2) * 0.17
        b_ = obj(prim('bell', M_CHIME, 16), 'chordbell'); b_.parent = None
        b_.matrix_world = M @ Matrix.Translation((bx, 0.32, 0.3)) @ Matrix.Diagonal((0.085, 0.085, 0.11, 1))
        c_ = pcol(p_, 0.3); b_.color = (*c_, 0); flash(b_, h['t'], c_)
        rr_ = rod(M @ Vector((bx, 0.32, 0.3)), M @ Vector((bx, 0.32, -0.04)), 0.012, M_GOLD, 6)
        objs += [b_, rr_]
    po = rod(top - nv * 0.12, (top[0], top[1], ground_z(top[0]) - 0.1), 0.035, M_CANE, 10)
    objs.append(po)
    gl = obj(prim('cyl', M_FX, 32), 'discglow'); gl.matrix_world = M @ Matrix.Translation((0, 0, 0.005)) @ Matrix.Diagonal((0.3, 0.3, 0.01, 1))
    gl.color = (1.0, 0.75, 0.8, 0); key_series(gl, 'color', 3, 0.5 * pulse_alpha([h['t']], decay=5, length=20), 1e-3); objs.append(gl)
    riser(objs, h['t'], top, lead=0.35, stay=0.8, pop=0.15, shrink=0.3)
    PIECES.append(dict(kind='chord', t=h['t'], top=top, n=nv, hw=0.23, hd=0.23, th=0.06, ball='mint', rs=(0.35, 0.8, 0.15, 0.3)))

# ---------------- cocoa: chocolate drums (and the ramps) ----------------
def drum(top, n, p, t, size=1.0):
    rr = min(0.32, 0.27 + 0.004 * max(0, 52 - p)) * size; hh = 0.3 * size
    M = Matrix.Translation(Vector(top)) @ frame_from_n(n).to_4x4()
    skin = obj(prim('cyl', M_CREAM, 32), 'skin'); skin.matrix_world = M @ Matrix.Translation((0, 0, -0.01)) @ Matrix.Diagonal((rr, rr, 0.02, 1))
    body = obj(prim('cyl', M_DRUM, 32), 'drumbody'); body.matrix_world = M @ Matrix.Translation((0, 0, -0.02 - hh / 2)) @ Matrix.Diagonal((rr * 1.02, rr * 1.02, hh, 1))
    rim = obj(prim('cyl', M_GOLD, 32), 'rim'); rim.matrix_world = M @ Matrix.Translation((0, 0, -0.03)) @ Matrix.Diagonal((rr * 1.06, rr * 1.06, 0.035, 1))
    glow = obj(prim('cyl', M_FX, 32), 'drumglow'); glow.matrix_world = M @ Matrix.Translation((0, 0, 0.005)) @ Matrix.Diagonal((rr * 0.9, rr * 0.9, 0.01, 1))
    glow.color = (*pcol(p, 0.5), 0); key_series(glow, 'color', 3, 0.55 * pulse_alpha([t], decay=5, length=20), 1e-3)
    bot = np.array(top) - np.array(n) * (hh + 0.02)
    leg = rod(bot, (bot[0], bot[1], ground_z(bot[0]) - 0.05), 0.05 * size, M_CANE, 8)
    return rr, [skin, body, rim, glow, leg]


for h in BP['cocoa'].hits:
    if h['kind'] != 'drum': continue
    nv = np.array(h['n']); top = np.array(h['p']) - RC * nv; pp = h['pitch']
    rr, objs = drum(top, nv, pp[0], h['t'])
    riser(objs, h['t'], top, lead=0.35, stay=0.8, pop=0.15, shrink=0.3)
    PIECES.append(dict(kind='drum', t=h['t'], top=top, n=nv, hw=rr * 0.72, hd=rr * 0.72, th=0.03, ball='cocoa', rs=(0.35, 0.8, 0.15, 0.3)))
for rp in EV['ramps']:
    u = np.array(rp['u']); n = np.array(rp['n']); pa = np.array(rp['p_a']); R0 = RC
    sk = rp['s_keys']; s_end = float(np.dot(np.array(rp['p_end']) - pa, u))
    objs = []
    hk = [h for h in BP['cocoa'].hits if h['kind'] in ('rampland', 'roll') and rp['t_a'] - 1e-6 <= h['t'] <= rp['t_end']]
    Mr = Matrix(((u[0], 0, n[0]), (0, 1, 0), (u[2], 0, n[2])))
    for j, s_ in enumerate(sk):
        c = pa + u * s_ - n * (R0 + 0.04)
        b_ = obj(prim('bcube', M_KEY, 0.2), 'rampbar'); b_.matrix_world = Matrix.Translation(Vector(c)) @ Mr.to_4x4() @ Matrix.Diagonal((0.16, 0.62, 0.08, 1))
        col = pcol(hk[j]['pitch'][0], 0.25); b_.color = (*col, 0.1); flash(b_, hk[j]['t'], col, base=0.1, after=0.2)
        objs.append(b_)
        PIECES.append(dict(kind='roll', t=hk[j]['t'], top=c + n * 0.04, n=n, hw=0.08, hd=0.31, th=0.04, ball='cocoa', own=(rp['t_a'] - 0.03, rp['t_end'] + 0.03),
                           life=(rp['t_a'] - 0.6, rp['t_end'] + 0.7)))
    for sy in (-0.33, 0.33):
        a_ = pa + u * (sk[0] - 0.25) - n * (R0 + 0.1) + np.array([0, sy, 0]); b2 = pa + u * (s_end + 0.05) - n * (R0 + 0.1) + np.array([0, sy, 0])
        objs.append(rod(a_, b2, 0.03, M_GOLD, 8))
    mid = pa + u * (0.5 * s_end) - n * (R0 + 0.14)
    objs.append(rod(mid, (mid[0], mid[1], ground_z(mid[0]) - 0.05), 0.04, M_CANE, 8))
    riser(objs, rp['t_a'], pa - n * R0, lead=0.5, stay=(rp['t_end'] - rp['t_a']) + 0.5, pop=0.18, shrink=0.3)

# ---------------- glass cups (the waiting places) ----------------
for cp in EV['cups']:
    who = cp['who']; r_ = {'fairy': RF, 'mint': RM, 'cocoa': RC}[who]
    p0 = np.array(cp['p']); p1 = np.array(cp['p_rel']); v = np.array(cp['v']); t0, t1 = cp['t'], cp['t_rel']
    moving = np.linalg.norm(v) > 1e-6
    root = link(bpy.data.objects.new('cup', None))
    P = np.zeros((NF, 3))
    for f in range(NF):
        t = T[f]; P[f] = p0 + v * (min(max(t, t0), t1 + 1.0) - t0)
    key_loc(root, P)
    ring = obj(prim('tube', M_GLASSC, 32, 0.12), 'cupring'); ring.parent = root; ring.location = (0, 0, -r_ * 0.25); ring.scale = (r_ * 1.35, r_ * 1.35, r_ * 1.1)
    gold = obj(prim('tube', M_GOLD, 32, 0.2), 'cuprim'); gold.parent = root; gold.location = (0, 0, r_ * 0.33); gold.scale = (r_ * 1.4, r_ * 1.4, 0.04)
    hinge = link(bpy.data.objects.new('cuphinge', None)); hinge.parent = root; hinge.location = (-r_ * 1.3, 0, -r_ - 0.02)
    door = obj(prim('cyl', M_GOLD, 24), 'cupdoor'); door.parent = hinge; door.location = (r_ * 1.3, 0, -0.015); door.scale = (r_ * 1.3, r_ * 1.3, 0.03)
    ang = np.array([0.0 if T[f] < t1 - 0.06 else min(1.0, (T[f] - t1 + 0.06) / 0.1) * 1.6 for f in range(NF)])
    hinge.rotation_mode = 'XYZ'; key_series(hinge, 'rotation_euler', 1, ang, 1e-4)
    # it hangs from a candy-cane trolley on a rope (moving) or from a gold hook (waiting)
    hook = rod((0, 0, r_ * 0.35), (0, 0, r_ * 0.35 + 1.4), 0.012, M_GOLD, 6); hook.parent = root
    for j, ang_ in enumerate((0.0, 2.1, 4.2)):
        rod((r_ * 1.35 * math.cos(ang_), r_ * 1.35 * math.sin(ang_), r_ * 0.35), (0, 0, r_ * 0.35 + 0.5), 0.008, M_GOLD, 4).parent = root
    if moving:
        a_ = p0 + np.array([0, 0, r_ * 0.35 + 1.4]) - v * 0.6; b_ = p1 + v * 1.2 + np.array([0, 0, r_ * 0.35 + 1.4])
        rope = rod(a_, b_, 0.012, M_THREAD, 4)
        wheel = obj(prim('cyl', M_CANE, 16), 'trolley'); wheel.parent = root; wheel.location = (0, 0, r_ * 0.35 + 1.42); wheel.rotation_euler = (math.pi / 2, 0, 0); wheel.scale = (0.09, 0.09, 0.05)
        sc_ = scale_curve(t0 - 0.6, t1 + 0.35, 0.25, 0.3)
        e_ = link(bpy.data.objects.new('ropepop', None)); rope.parent = e_
        for k in range(3): key_series(e_, 'scale', k, np.where(sc_ > 0.01, 1.0, 0.001), 1e-4)
    sc_ = scale_curve(t0 - 0.7, t1 + 0.35, 0.25, 0.35)
    for k in range(3): key_series(root, 'scale', k, sc_, 1e-4)
    if not moving:
        PIECES.append(dict(kind='cup', t=t0, top=p0 - [0, 0, r_], n=np.array([0, 0, 1.0]), hw=r_ * 1.3, hd=r_ * 1.3, th=0.04, ball=who, own=(t0 - 0.03, t1 + 0.45),
                           life=(t0 - 0.4, t1 - 0.05)))
    if cp['kind'] == 'cup': LIT_T.append(t0) if who == 'fairy' else None

# ---------------- the lever cup (cocoa, bar 20): a candy-cane lever knocks the latch of the sugar bowl ----------------
lv = EV['lever']; bw = EV['bowl']; HP = EV['harp']
HK = {int(k): v for k, v in HP['keys'].items()}
hx0 = min(v['x'] for v in HK.values()); hx1 = max(v['x'] for v in HK.values())
SIEVE_Z0 = HP['sieve_z']
BOWL_C = np.array([hx0 - 0.95, HP['y'], SIEVE_Z0 + 1.15])
plv = np.array(lv['p'])
piv = plv + np.array([0.9, 0.0, -RC - 0.25])
LV = link(bpy.data.objects.new('lever', None)); LV.location = tuple(piv)
arm_far = BOWL_C + np.array([0.0, 0.0, -0.55])
for p_a, p_b, w_ in ((piv, plv + np.array([0, 0, -RC - 0.05]), 0.05), (piv, arm_far, 0.04)):
    r_ = rod(p_a, p_b, w_, M_CANE, 10); r_.parent = LV; r_.matrix_parent_inverse = Matrix.Translation(-Vector(piv))
fulc = obj(prim('cone', M_GOLD, 4, 0.2, 0.02, 0.35), 'fulcrum'); fulc.location = tuple(piv + np.array([0, 0, -0.18]))
rod(piv + np.array([0, 0, -0.35]), (piv[0], piv[1], ground_z(piv[0]) - 0.1), 0.05, M_GOLD, 8)
la = np.array([0.0 if T[f] < lv['t'] else min(1.0, (T[f] - lv['t']) / 0.12) * 0.12 for f in range(NF)])
LV.rotation_mode = 'XYZ'; key_series(LV, 'rotation_euler', 1, la, 1e-4)
key_series(LV, 'scale', 0, scale_curve(lv['t'] - 1.2, HP['t_down'], 0.3, 0.4)); key_series(LV, 'scale', 1, scale_curve(lv['t'] - 1.2, HP['t_down'], 0.3, 0.4))
key_series(LV, 'scale', 2, scale_curve(lv['t'] - 1.2, HP['t_down'], 0.3, 0.4))

# ---------------- the giant sugar bowl, the sieve, the crystal harp (rises out of the snow), the glass trough ----------------
BOWL = link(bpy.data.objects.new('bowl', None)); BOWL.location = tuple(BOWL_C + np.array([0.55, 0, -0.3]))
bo = obj(prim('bowl', M_PORC, 40), 'sugarbowl'); bo.parent = BOWL; bo.location = (-0.55, 0, 0.3); bo.scale = (0.85, 0.85, 0.75)
brim = obj(prim('tube', M_GOLD, 40, 0.04), 'bowlrim'); brim.parent = BOWL; brim.location = (-0.55, 0, 0.32); brim.scale = (0.86, 0.86, 0.04)
sug = obj(prim('sph', M_PEARL, 24, 12), 'sugarheap'); sug.parent = BOWL; sug.location = (-0.55, 0, 0.25); sug.scale = (0.78, 0.78, 0.3); sug.color = (1.0, 0.9, 1.0, 0.5)
for dx in (-0.95, -0.15): rod((BOWL_C[0] + dx + 0.0, HP['y'] + 0.5, BOWL_C[2] - 0.6), (BOWL_C[0] + dx, HP['y'] + 0.5, ground_z(BOWL_C[0]) - 0.1), 0.04, M_GOLD, 8)
tip = np.array([0.0 if T[f] < bw['t_tip0'] else min(1.0, (T[f] - bw['t_tip0']) / (bw['t_tip1'] - bw['t_tip0'])) ** 1.5 * 1.15 for f in range(NF)])
BOWL.rotation_mode = 'XYZ'; key_series(BOWL, 'rotation_euler', 1, tip, 1e-4)
latch = box(tuple(BOWL_C + np.array([-0.2, 0, -0.5])), (0.5, 0.12, 0.08), M_GOLD, name='latch')
key_series(latch, 'rotation_euler', 1, np.array([0.0 if T[f] < bw['t_latch'] else min(1.0, (T[f] - bw['t_latch']) / 0.08) * -1.2 for f in range(NF)]), 1e-4)
# the sugar stream from the bowl into the sieve (while the cadenza rains)
pearl_t = sorted(BP[n].hits[0]['t'] for n in PEARLN if n.startswith('pc'))
stream = rod(BOWL_C + np.array([0.3, 0, 0.1]), BOWL_C + np.array([0.85, 0, -1.0]), 0.09, M_FX, 12); stream.color = (1.0, 0.9, 1.0, 0)
key_series(stream, 'color', 3, np.array([0.0 if not (bw['t_tip1'] - 0.2 < T[f] < pearl_t[-1] - 0.3) else 0.5 for f in range(NF)]), 1e-3)
H_UP = HP['t_up']; H_DN = HP['t_down']
HARP = link(bpy.data.objects.new('harp', None))
hz = np.array([-6.0 + 6.0 * min(1.0, max(0.0, (T[f] - H_UP) / 0.8)) ** 0.7 - 6.0 * min(1.0, max(0.0, (T[f] - H_DN) / 0.7)) ** 1.5 for f in range(NF)])
key_series(HARP, 'location', 2, hz, 1e-4)
NH = np.array(HP['n'])
for p_, kk in sorted(HK.items()):
    top = np.array([kk['x'], HP['y'], kk['z']])
    k_ = obj(prim('bcube', M_KEY, 0.25), 'harpkey'); k_.parent = HARP
    k_.matrix_world = Matrix.Translation(Vector(top - NH * 0.035)) @ frame_from_n(NH).to_4x4() @ Matrix.Diagonal((0.28, 0.66, 0.08, 1))
    col = pcol(p_, 0.1); k_.color = (*col, 0.12)
    hits_ = [BP[n].hits[0]['t'] for n in PEARLN if n.startswith('pc') and BP[n].hits[0]['pitch'] == p_]
    a_ = np.maximum(np.where(T > min(hits_), 0.35, 0.12), pulse_alpha(hits_, decay=6, length=20))
    key_series(k_, 'color', 3, a_, 2e-3)
    leg = rod(top - NH * 0.08, (top[0], top[1] + 0.1, kk['z'] - 3.0), 0.025, M_GOLD, 6); leg.parent = HARP
    for n in PEARLN:
        if not n.startswith('pc'): continue
        h0 = BP[n].hits[0]
        if h0['pitch'] == p_:
            PIECES.append(dict(kind='harp', t=h0['t'], top=top, n=NH, hw=0.13, hd=0.32, th=0.035, ball=n, life=(H_UP + 0.8, H_DN)))
            LIT_T.append(h0['t'])
for yy in (HP['y'] + 0.25, HP['y'] - 0.25):
    rod((hx0 - 0.3, yy, HK[min(HK)]['z'] - 0.12), (hx1 + 0.3, yy, HK[max(HK)]['z'] - 0.12), 0.03, M_GOLD, 8).parent = HARP
# the sieve (glass trough above the keys) follows the key line
sv = rod((hx0 - 0.4, HP['y'], SIEVE_Z0 + 0.0), (hx1 + 0.3, HP['y'], SIEVE_Z0 + 0.055 * (max(HK) - min(HK)) + 0.0), 0.2, glassfx('sieve', (0.95, 0.85, 1.0), 0.12, 0.2), 16)
sv.parent = HARP
# the glass trough where the pearls land (glowing pile)
land = np.array([BP[n].hits[-1]['p'] for n in PEARLN if n.startswith('pc')])
tz = HP['z_trough']
tro = box(((land[:, 0].min() + land[:, 0].max()) / 2, land[:, 1].mean(), tz - 0.04), (land[:, 0].max() - land[:, 0].min() + 0.8, land[:, 1].max() - land[:, 1].min() + 0.7, 0.08), M_GLASSC, name='trough')
tro.parent = HARP
for sx in (-1, 1):
    box(((land[:, 0].min() + land[:, 0].max()) / 2, land[:, 1].mean() + sx * ((land[:, 1].max() - land[:, 1].min()) / 2 + 0.35), tz + 0.12), (land[:, 0].max() - land[:, 0].min() + 0.8, 0.05, 0.3), M_GLASSC, name='troughwall').parent = HARP

# ---------------- the crescent moon the fairy starts in, the gingerbread chimney the guards start on ----------------
mp = np.array(EV['moon']['p'])
MO = link(bpy.data.objects.new('moonroot', None))
key_loc(MO, np.array([POS['fairy'][f] if T[f] < EV['moon']['t_release'] else mp for f in range(NF)]) + np.array([0, 0, -RF - 0.02]))
moon = obj(prim('crescent', M_MOON, 0.1), 'crescent'); moon.parent = MO; moon.location = (0.32, 0.12, 0.62); moon.rotation_euler = (0, -math.pi / 2, 0); moon.scale = (0.75, 0.6, 0.75)
door = obj(prim('cyl', M_GOLD, 24), 'moondoor'); door.parent = MO; door.scale = (0.26, 0.26, 0.03)
mr = rod((0, 0.12, 0.7), (0, 0.12, 9.0), 0.012, M_THREAD, 4); mr.parent = MO
rod(np.array(EV['moon']['p0']) + np.array([-2, 0.12, 9.0 - RF]), mp + np.array([3, 0.12, 9.0 - RF]), 0.015, M_THREAD, 4)
tr_ = EV['moon']['t_release']
key_series(door, 'location', 0, np.array([0.0 if T[f] < tr_ - 0.05 else min(1.0, (T[f] - tr_ + 0.05) / 0.12) * 0.5 for f in range(NF)]), 1e-4)
for nm_ in ('mint', 'cocoa'):
    s0 = EV[f'{nm_}_start']; p0 = np.array(s0['p']); r_ = RM if nm_ == 'mint' else RC
    box(tuple(p0 + np.array([-0.55, 0.0, -r_ - 0.08])), (0.7, 0.8, 0.16), M_GINGER, bev=True, name='ledge')
    trp = box(tuple(p0 + np.array([0, 0.0, -r_ - 0.04])), (0.5, 0.6, 0.06), M_ICING, name='trapdoor')
    key_series(trp, 'location', 0, np.array([p0[0] + (0.0 if T[f] < s0['t_release'] - 0.05 else min(1.0, (T[f] - s0['t_release'] + 0.05) / 0.12) * 0.55) for f in range(NF)]), 1e-4)
    rod(p0 + np.array([-0.85, 0, -r_ - 0.15]), (p0[0] - 0.85, p0[1], ground_z(p0[0] - 0.85) - 0.1), 0.07, M_CANE, 10)

# ---------------- the Christmas tree, the lift, the star bell, the sugar cloud ----------------
tx, ty, tz0, R0T, HT = tr['x'], tr['y'], tr['z0'], tr['r0'], tr['h']
rod((tx, ty, tz0 - 0.6), (tx, ty, tz0 + 1.2), 0.45, M_TRUNK, 16)
for j, (z_a, z_b, r_a) in enumerate(((0.6, 4.2, 1.0), (2.8, 6.6, 0.74), (5.0, HT, 0.5))):
    c_ = obj(prim('cone', M_TREE, 40, 1.0, 0.0, 1.0), 'fir'); c_.location = (tx, ty, tz0 + (z_a + z_b) / 2); ra = R0T * r_a * 1.05
    c_.scale = (ra, ra, z_b - z_a)
    sn = obj(prim('cone', M_SNOW, 40, 1.0, 0.0, 1.0), 'firsnow'); sn.location = (tx, ty, tz0 + z_b - (z_b - z_a) * 0.16); sn.scale = (ra * 0.24, ra * 0.24, (z_b - z_a) * 0.3)
ORN = EV['ornaments']
for j, o_ in enumerate(ORN):
    p = np.array(o_['p']); n = np.array(o_['n']); col = pcol(o_['pitch'] + j * 5, 0.0)
    r_o = o_['r']
    ob = sphere(tuple(p - n * r_o), r_o, M_CHIME, (20, 10), 'ornament'); ob.color = (*col, 0.04); flash(ob, o_['t'], col, base=0.04, after=0.6)
    cap = obj(prim('cyl', M_GOLD, 12), 'orncap'); cap.location = tuple(p - n * r_o + np.array([0, 0, r_o])); cap.scale = (0.05, 0.05, 0.05)
    PIECES.append(dict(kind='ornament', t=o_['t'], top=p, n=n, hw=r_o * 0.7, hd=r_o * 0.7, th=r_o * 0.7, ball=[nn for nn in PEARLN if nn.startswith('pt') and abs(BP[nn].hits[0]['t'] - o_['t']) < 1e-6 and BP[nn].hits[0]['pitch'] == o_['pitch']][0]))
    LIT_T.append(o_['t'])
# garland of lights (lit on the final chord)
for k in range(150):
    u = k / 150; z_ = tz0 + 0.5 + u * (HT - 1.0); r_ = R0T * (1 - (z_ - tz0) / HT) + 0.05; ph = 2 * math.pi * 6.5 * u
    if math.cos(ph) > 0.3: continue
    s_ = sphere((tx + r_ * math.sin(ph), ty - r_ * math.cos(ph), z_), 0.07, M_FX, (6, 3), 'garland'); s_.color = (*pcol(k * 5, 0.2), 0.0)
    key_series(s_, 'color', 3, np.clip((T - FINAL - 0.02 * k % 0.6) / 0.15, 0, 1) * 0.9, 1e-3)
lf = EV['lift']; cup_l = np.array(lf['cup']); ud = np.array(lf['ud']); p_l = np.array(lf['p_launch'])
rail0 = cup_l - ud * 0.6 - np.array([0, 0, RF + 0.1]); rail1 = p_l + ud * 0.25 - np.array([0, 0, RF + 0.1])
for sx in (-0.2, 0.2): rod(rail0 + np.array([sx, 0, 0]), rail1 + np.array([sx, 0, 0]), 0.04, M_CANE, 10)
for k in range(9):
    q = rail0 + (rail1 - rail0) * k / 8; rod(q + np.array([-0.2, 0, 0]), q + np.array([0.2, 0, 0]), 0.025, M_GOLD, 6)
rod(rail0, (rail0[0], rail0[1], ground_z(rail0[0]) - 0.1), 0.06, M_CANE, 10)
LC = link(bpy.data.objects.new('liftcup', None))
PL = np.zeros((NF, 3)); spin = np.zeros(NF)
for f in range(NF):
    t = T[f]
    if t < lf['t0']: s_ = 0.0
    elif t < lf['t_launch']: s_ = 0.5 * lf['acc'] * (t - lf['t0']) ** 2
    else: s_ = lf['L']
    PL[f] = cup_l + ud * s_ - np.array([0, 0, RF]); spin[f] = s_ / 0.25
key_loc(LC, PL)
cup_ = obj(prim('bowl', M_GLASSC, 32), 'liftbowl'); cup_.parent = LC; cup_.location = (0, 0, RF * 0.95); cup_.scale = (RF * 1.45, RF * 1.45, RF * 1.0)
cg = obj(prim('tube', M_GOLD, 32, 0.15), 'liftrim'); cg.parent = LC; cg.location = (0, 0, RF * 0.95); cg.scale = (RF * 1.5, RF * 1.5, 0.04)
car = box((0, 0, -0.08), (0.5, 0.18, 0.1), M_GOLD, name='carriage'); car.parent = LC
gear = obj(prim('cyl', M_GOLD, 12), 'gear'); gear.location = tuple(rail0 + np.array([0.0, -0.25, 0.0])); gear.rotation_euler = (math.pi / 2, 0, 0); gear.scale = (0.35, 0.35, 0.08)
gear.rotation_mode = 'XYZ'; key_series(gear, 'rotation_euler', 2, spin, 1e-3)
for k in range(8):
    t_ = box(tuple(rail0 + np.array([0.0, -0.25, 0.0])), (0.12, 0.12, 0.12), M_GOLD, name='tooth'); t_.parent = gear
    t_.matrix_parent_inverse = Matrix.Identity(4); t_.location = (0.38 * math.cos(k * math.pi / 4), 0.38 * math.sin(k * math.pi / 4), 0); t_.scale = (0.3, 0.3, 1.2)
chain = rod(rail0 + np.array([0, -0.25, 0]), rail1 + np.array([0, -0.25, 0]), 0.012, M_GOLDD, 4)
sp_ = np.array(EV['star']['p'])
sbell = obj(prim('bell', M_CHIME, 32), 'starbell'); sbell.location = tuple(sp_ + np.array([0, 0, 0.02])); sbell.scale = (0.42, 0.42, 0.34)
sbell.color = (1.0, 0.85, 0.45, 0.15); key_series(sbell, 'color', 3, np.maximum(0.15, pulse_alpha([FINAL], decay=30, length=120)), 1e-3)
star = obj(prim('star', M_CHIME, 0.3), 'treestar'); star.location = tuple(sp_ + np.array([0, 0.0, 0.85])); star.scale = (0.62, 0.62, 0.62)
star.color = (1.0, 0.82, 0.35, 0.1); key_series(star, 'color', 3, np.where(T < FINAL, 0.1, 1.0), 1e-3)
rod(sp_ + np.array([0, 0, 0.02]), sp_ + np.array([0, 0, 0.3]), 0.02, M_GOLD, 6)
rod((tx, ty, tz0 + HT - 0.2), sp_ + np.array([0, 0, 0.3]), 0.03, M_GOLD, 6)
PIECES.append(dict(kind='starbell', t=FINAL, top=sp_, n=np.array([0, 0, -1.0]), hw=0.25, hd=0.25, th=0.05, ball='fairy', life=(0, FINAL + 0.8)))
LIT_T.append(FINAL)
CL = EV['cloud']
for bz in CL['bands']:                                    # three rings of sugar clouds around the tree
    rb = R0T * max(0.0, 1 - (bz - tz0) / HT) + 0.55
    for k in range(13):
        az = math.radians(-92 + 184 * k / 12)
        pf = sphere((tx + rb * math.sin(az), ty - rb * math.cos(az), bz + 0.28 + rng.uniform(-0.05, 0.12)), rng.uniform(0.26, 0.38), M_CLOUD, (14, 7), 'sugarcloud')
        s0 = pf.scale[0]
        sc_ = np.clip((T - CL['t_open'] + 1.8) / 0.8, 0.001, 1) * s0
        for c_ in range(3): key_series(pf, 'scale', c_, sc_, 1e-4)

# ---------------- candy castle in the background, windows light with the fairy's notes ----------------
WIN = []
x_lo, x_hi = -12.0, FLX[-1] - 10
x_ = x_lo; k = 0
TOWERM = [M_PINK, M_MINTF, M_LILAC, M_ICING]
while x_ < x_hi:
    gz = ground_z(x_); yb = 9.5 + 1.2 * math.sin(k * 1.7)
    hgt = 7.5 + 3.0 * ((k * 7) % 5) / 4; rad = 1.1 + 0.3 * (k % 3)
    rod((x_, yb, gz - 1.0), (x_, yb, gz + hgt), rad, M_GINGER, 24)
    for zz in np.arange(gz + 1.5, gz + hgt - 0.5, 1.6): rod((x_, yb, zz), (x_, yb, zz + 0.14), rad + 0.06, M_ICING, 24)
    roof = obj(prim('cone', TOWERM[k % 4], 24, 1.0, 0.0, 1.0), 'roof'); roof.location = (x_, yb, gz + hgt + 1.3); roof.scale = (rad * 1.3, rad * 1.3, 2.6)
    sphere((x_, yb, gz + hgt + 2.7), 0.25, M_PINK if k % 2 else M_MINTF, (12, 6), 'gumdrop')
    for zz in np.arange(gz + 2.3, gz + hgt - 0.4, 1.6):
        for dxw in ((-0.45, 0.45) if rad > 1.25 else (0.0,)):
            w_ = obj(prim('bcube', M_WIN, 0.3), 'window'); w_.location = (x_ + dxw, yb - rad - 0.02, zz); w_.scale = (0.38, 0.06, 0.62)
            WIN.append([w_, x_ + dxw])
    # wall to the next tower
    wx = x_ + 3.4
    if wx < x_hi:
        box((wx, yb + 0.6, gz + 1.8), (4.2, 1.0, 5.6), M_GINGER, name='wall')
        box((wx, yb + 0.05, gz + 4.6), (4.3, 0.2, 0.3), M_ICING, name='icing')
        for j in range(5): box((wx - 1.7 + 0.85 * j, yb + 0.1, gz + 4.95), (0.35, 0.3, 0.45), M_ICING, name='merlon')
        for dxw in (-1.0, 0.0, 1.0):
            w_ = obj(prim('bcube', M_WIN, 0.3), 'window'); w_.location = (wx + dxw, yb + 0.07, gz + 2.6); w_.scale = (0.45, 0.06, 0.75)
            WIN.append([w_, wx + dxw])
    x_ += 6.8; k += 1
# light the windows with the fairy's keys: each lit key turns on the nearest dark window
WIN_T = {}
order = sorted(range(len(WIN)), key=lambda i: WIN[i][1])
for top, th, col in KEYS:
    cands = [i for i in order if i not in WIN_T]
    if not cands: break
    i = min(cands, key=lambda i: abs(WIN[i][1] - top[0]))
    WIN_T[i] = th
for i, (w_, wx) in enumerate(WIN):
    th = WIN_T.get(i, FINAL + 0.3 + 0.004 * (i % 60))
    c_ = (1.0, 0.8, 0.45) if i % 3 else (1.0, 0.65, 0.85)
    w_.color = (*c_, 0.0)
    a_ = np.where(T < th, 0.0, 0.42); fh = int(round(th * FPS))
    for kf in range(max(0, fh), min(NF, fh + 12)): a_[kf] = max(a_[kf], 0.42 + 0.8 * math.exp(-(kf - fh) / 3))
    key_series(w_, 'color', 3, a_, 1e-3)

# ---------------- candy decorations in front: candy canes, lollipops, gumdrops ----------------
for x0 in np.arange(-10, FLX[-1] - 5, 3.1):
    gz = ground_z(x0); kind = int(rng.integers(0, 3)); yy = rng.uniform(-3.4, -2.6)
    if kind == 0:
        rod((x0, yy, gz - 0.3), (x0, yy, gz + 1.4), 0.08, M_CANE, 12)
        for j in range(8):
            a0 = math.pi * j / 8; a1 = math.pi * (j + 1) / 8
            rod((x0 + 0.25 - 0.25 * math.cos(a0), yy, gz + 1.4 + 0.25 * math.sin(a0)), (x0 + 0.25 - 0.25 * math.cos(a1), yy, gz + 1.4 + 0.25 * math.sin(a1)), 0.08, M_CANE, 12)
    elif kind == 1:
        rod((x0, yy, gz - 0.3), (x0, yy, gz + 1.1), 0.03, M_ICING, 8)
        lp = obj(prim('cyl', stripes(f'lolli{len(_MC)}', (1.0, 0.95, 0.98), pcol(int(rng.integers(0, 12)), 0.1), scale=5.0, radial=True), 32), 'lolli')
        lp.location = (x0, yy, gz + 1.45); lp.rotation_euler = (math.pi / 2, 0, 0); lp.scale = (0.4, 0.4, 0.08)
    else:
        for j in range(3):
            g_ = sphere((x0 + 0.45 * j, yy, gz - 0.05), 0.3, M_OBJ, (16, 8), 'gumdrop'); g_.scale[2] = 0.24; g_.color = (*pcol(int(rng.integers(0, 12)), 0.1), 1)

# ---------------- signboards (search keywords, English first, Japanese below) ----------------
M_SIGNI1 = mat('signink1', (0.55, 0.12, 0.3), 0.5); M_SIGNI2 = mat('signink2', (0.18, 0.3, 0.62), 0.5); M_SIGNI3 = mat('signink3', (0.45, 0.24, 0.1), 0.5)
M_SIGNF = mat('signface', (1.0, 0.95, 0.88), 0.6)
def signboard(lines, x, w, h, zc, y=-2.2):
    box((x, y + 0.06, zc), (w, 0.12, h), M_GINGER, bev=True, name='sign')
    box((x, y - 0.01, zc), (w - 0.2, 0.04, h - 0.2), M_SIGNF, name='signface')
    for sx in (-1, 1): rod((x + sx * (w / 2 - 0.35), y + 0.12, zc - h / 2), (x + sx * (w / 2 - 0.35), y + 0.12, ground_z(x) - 0.3), 0.07, M_CANE, 10)
    tot_h = sum(sz * 1.35 for _, sz, _ in lines); zz = zc + tot_h / 2
    for s_, size, mt in lines:
        zz -= size * 1.35
        fit_w(text(s_, (x, y - 0.06, zz + size * 0.67), size, mt, rot=(math.pi / 2, 0, 0)), w - 0.45)
SIGNS = ((2.6, 4.9, 2.4, [('Dance of the', 0.36, M_SIGNI3), ('Sugar Plum Fairy', 0.5, M_SIGNI1), ('くるみ割り人形「金平糖の精の踊り」', 0.24, M_SIGNI2)]),
         (8.2, 4.7, 2.2, [('Tchaikovsky', 0.52, M_SIGNI1), ('The Nutcracker · Ballet', 0.3, M_SIGNI3), ('チャイコフスキー　クラシック', 0.24, M_SIGNI2)]),
         (14.0, 4.7, 2.2, [('Marble Machine', 0.5, M_SIGNI2), ('Satisfying · ASMR', 0.32, M_SIGNI3), ('ビー玉マシン　からくり', 0.24, M_SIGNI1)]),
         (19.8, 4.7, 2.2, [('Christmas Music', 0.48, M_SIGNI1), ('Celesta · Music Box', 0.32, M_SIGNI3), ('クリスマス　オルゴール', 0.24, M_SIGNI2)]),
         (25.6, 4.7, 2.2, [('Candy Castle', 0.5, M_SIGNI2), ('Classical music', 0.32, M_SIGNI3), ('お菓子の城　バレエ音楽', 0.24, M_SIGNI1)]))
for tb, w_, h_, lines in SIGNS:
    xx = xs(tb) + 0.2
    signboard(lines, xx, w_, h_, ground_z(xx) + 0.45 + h_ / 2)

# ---------------- night sky: stars, the big moon, snowfall ----------------
for k in range(240):
    x_ = rng.uniform(-20, FLX[-1] + 10); s_ = sphere((x_, rng.uniform(16, 24), ground_z(x_) + rng.uniform(2.0, 22.0)), rng.uniform(0.04, 0.1), M_FX, (6, 3), 'skystar')
    s_.color = (0.85, 0.9, 1.0, 0); key_series(s_, 'color', 3, np.round(0.35 + 0.3 * np.sin(rng.uniform(1.5, 4.0) * T + rng.uniform(0, 6.28)), 2), 0.02)
for xm in (6.0, 50.0):
    sphere((xs(xm) - 6, 30.0, zs(xm) + 11.0), 2.6, M_MOON, (32, 16), 'bigmoon')
for k in range(380):
    x0 = rng.uniform(-14, FLX[-1] + 5); y0 = rng.uniform(-4.5, 6.0); z_top = ground_z(x0) + rng.uniform(4, 22)
    fl_ = sphere((x0, y0, z_top), rng.uniform(0.025, 0.05), M_FX, (6, 3), 'flake'); fl_.color = (1.0, 1.0, 1.0, 0.35)
    vz = -rng.uniform(0.35, 0.6); vx = rng.uniform(-0.15, 0.1)
    fl_.location = (x0, y0, z_top); fl_.keyframe_insert('location', frame=0)
    fl_.location = (x0 + vx * T[-1], y0, z_top + vz * T[-1]); fl_.keyframe_insert('location', frame=NF - 1)

# ---------------- the counter ----------------
LIT_T = sorted(LIT_T)
CNT = np.array([int(np.searchsorted(LIT_T, t + 1e-6, side='right')) for t in T])
print('lights counter final', CNT[-1], 'of', len(LIT_T))

# ---------------- world & lights ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes['Background']
_wt = w.node_tree; _tc2 = _wt.nodes.new('ShaderNodeTexCoord'); _sp = _wt.nodes.new('ShaderNodeSeparateXYZ'); _cr = _wt.nodes.new('ShaderNodeValToRGB')
_wt.links.new(_tc2.outputs['Generated'], _sp.inputs[0]); _cr.color_ramp.interpolation = 'EASE'; _wt.links.new(_sp.outputs['Z'], _cr.inputs['Fac'])
_cr.color_ramp.elements[0].position = 0.02; _cr.color_ramp.elements[0].color = (0.2, 0.08, 0.3, 1)
_cr.color_ramp.elements[1].position = 0.25; _cr.color_ramp.elements[1].color = (0.01, 0.02, 0.09, 1)
_wt.links.new(_cr.outputs['Color'], bg.inputs['Color']); bg.inputs['Strength'].default_value = 0.9
mn = bpy.data.lights.new('moon', 'SUN'); mn.color = (0.62, 0.72, 1.0); mn.angle = 0.08; mn.energy = 0.55
mno = link(bpy.data.objects.new('moon', mn)); mno.rotation_euler = (math.radians(48), math.radians(-14), math.radians(-25))

# ---------------- camera: follow the stage; pull back for the sugar rain; at the end show the whole tree ----------------
FOC = np.zeros((NF, 3)); WID = np.zeros(NF)
for f in range(NF):
    t = T[f]; u0 = min(1.0, max(0.0, (t - 6.0) / 2.5)); u0 = u0 * u0 * (3 - 2 * u0)
    FOC[f] = np.array([xs(min(t, FINAL)) + 0.1, 0, zs(t) + 2.4 + 1.9 * (1 - u0)]); WID[f] = 5.7
HC = np.array([(hx0 + hx1) / 2 - 0.4, 0, (HP['z_trough'] + SIEVE_Z0) / 2 + 0.6])
TC0, TC1 = bw['t_tip0'] - 0.6, H_DN - 0.2
TEND = lf['t0'] - 0.8
END_F = np.array([tx - 0.3, 0, tz0 + HT * 0.5 + 0.9])
for f in range(NF):
    t = T[f]
    u = min(1.0, max(0.0, (t - TC0) / 1.2)) * (1 - min(1.0, max(0.0, (t - TC1) / 1.2))); u = u * u * (3 - 2 * u)
    FOC[f] = (1 - u) * FOC[f] + u * HC; WID[f] = (1 - u) * WID[f] + u * 8.6
    if t > TEND:
        u2 = min(1.0, (t - TEND) / 2.2); u2 = u2 * u2 * (3 - 2 * u2)
        FOC[f] = (1 - u2) * FOC[f] + u2 * END_F; WID[f] = (1 - u2) * WID[f] + u2 * 8.0
FOC[:, 0] = smooth(FOC[:, 0], 0.45); FOC[:, 2] = smooth(FOC[:, 2], 0.8); WID = smooth(WID, 0.45)
LENS = 32.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 500
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24
    CP[f] = FOC[f] + np.array([0, -D, 0.16 * D]); TP[f] = FOC[f] + np.array([0, 0, 0.25])
key_loc(cam, CP); key_loc(tgt, TP)
for k, (dx, dz, en, col) in enumerate(((-3.0, 6.0, 1300, (1.0, 0.8, 0.6)), (3.5, 5.0, 900, (1.0, 0.75, 0.85)), (0.0, 1.0, 420, (0.7, 0.78, 1.0)))):
    al = bpy.data.lights.new(f'fill{k}', 'AREA'); al.size = 6.0; al.color = col; al.energy = en
    ao = link(bpy.data.objects.new(f'fill{k}', al))
    key_loc(ao, np.stack([FOC[:, 0] + dx, np.full(NF, -4.5 if k < 2 else -6.0), FOC[:, 2] + dz], 1))
    ao.rotation_euler = (math.radians(-28), 0, 0) if k < 2 else (math.radians(-80), 0, 0)
spd = bpy.data.lights.new('spotfairy', 'SPOT'); spd.spot_size = math.radians(16); spd.spot_blend = 0.7; spd.color = (1.0, 0.85, 1.0); spd.shadow_soft_size = 0.3; spd.energy = 800
so_ = link(bpy.data.objects.new('spotfairy', spd)); stg = link(bpy.data.objects.new('spottgt', None))
Ps = POS['fairy'].copy()
for k in range(3): Ps[:, k] = smooth(Ps[:, k], 0.25)
key_loc(stg, Ps - np.array([0, 0, 0.25])); key_loc(so_, Ps + np.array([-2.0, -6.0, 9.0]))
c_ = so_.constraints.new('TRACK_TO'); c_.target = stg; c_.track_axis = 'TRACK_NEGATIVE_Z'; c_.up_axis = 'UP_Y'

# ---------------- confetti & sparks on the final chord (decoration) ----------------
for k in range(90):
    p0 = sp_ + np.array([0, -0.3, 0.6]); v = np.array([rng.normal(0, 2.2), rng.normal(-0.5, 0.6), rng.uniform(2.0, 5.5)])
    s_ = sphere(tuple(p0), rng.uniform(0.04, 0.08), M_FX, (6, 3), 'spark'); s_.color = (*pcol(int(rng.integers(0, 12)), 0.3), 0)
    P = np.array([p0 + v * max(0, t - FINAL) + np.array([0, 0, -0.5 * 3.0 * max(0, t - FINAL) ** 2]) for t in T])
    key_loc(s_, P); key_series(s_, 'color', 3, np.where(T < FINAL, 0.0, np.clip(1.2 - (T - FINAL) / 2.5, 0, 1)), 1e-3)

# ---------------- HUD (rides with the camera) ----------------
HUD_D = 3.0
def hud_text(s, x, y, size, m, align='CENTER'):
    o = text(s, (x, y, -HUD_D), size, m, rot=(0, 0, 0), align=align, ext=0.0); o.parent = cam; return o
def vis(o, a):
    s0 = tuple(o.scale)
    for c in range(3): key_series(o, 'scale', c, np.maximum(0.001, a * s0[c]), 1e-3)
M_HUDW = emis('hudw', (1.0, 0.97, 0.94), 3.0); M_HUDG = emis('hudg', (1.0, 0.82, 0.45), 4.0); M_HUDP = emis('hudp', (1.0, 0.68, 0.95), 4.0)
M_HUDBG = mat('hudbg', (0.03, 0.02, 0.08), 1.0, **{'Alpha': 0.5})
def window(t0, t1, fade=0.3): return np.array([0.0 if t < t0 else min(1.0, (t - t0) / fade) * (1.0 if t < t1 else max(0.0, 1 - (t - t1) / fade)) for t in T])
hook_a = window(-1, 4.0)
vis(fit_w(hud_text('Can ONE marble light up', 0, 0.97, 0.085, M_HUDW), 0.95), hook_a)
vis(fit_w(hud_text('the Candy Castle?', 0, 0.875, 0.1, M_HUDP), 0.95), hook_a)
vis(hud_text('1個のビー玉で お菓子の城は光る？', 0, 0.795, 0.044, M_HUDW), hook_a)
hb_ = obj(prim('plane', M_HUDBG), 'hudbg'); hb_.parent = cam; hb_.location = (0, 0.89, -HUD_D - 0.01); hb_.scale = (0.78, 0.13, 1); vis(hb_, hook_a)
cnt_a = window(4.0, FINAL + 0.9)
vis(hud_text('LIGHTS', -0.06, 0.9, 0.07, M_HUDG, align='RIGHT'), cnt_a)
cb_ = obj(prim('plane', M_HUDBG), 'cntbg'); cb_.parent = cam; cb_.location = (0.04, 0.9, -HUD_D - 0.01); cb_.scale = (0.36, 0.05, 1); vis(cb_, cnt_a)
vis(hud_text('個の灯り', 0.24, 0.9, 0.036, M_HUDW, align='LEFT'), cnt_a)
DX = 0.07
for pos in range(3):
    dv = (CNT // (10 ** (2 - pos))) % 10
    shown = CNT >= 10 ** (2 - pos) if pos < 2 else np.ones(NF, bool)
    for dgt in range(10):
        o = hud_text(str(dgt), 0.02 + DX * pos, 0.9, 0.1, M_HUDW)
        vis(o, ((dv == dgt) & shown).astype(float) * cnt_a)
rain_a = window(bw['t_tip1'] + 0.2, bw['t_tip1'] + 2.4)
vis(fit_w(hud_text('SUGAR RAIN!', 0, 0.78, 0.1, M_HUDP), 0.95), rain_a)
vis(hud_text('砂糖の雨がハープに！', 0, 0.71, 0.042, M_HUDW), rain_a)
lift_a = window(lf['t0'] - 0.2, lf['t_launch'] - 0.3)
vis(fit_w(hud_text('Up to the star!', 0, -0.55, 0.09, M_HUDG), 0.95), lift_a)
vis(hud_text('ツリーのてっぺんへ！', 0, -0.63, 0.042, M_HUDW), lift_a)
end_a = window(FINAL + 1.0, 1e9, 0.4)
vis(fit_w(hud_text(f'{CNT[-1]} LIGHTS!', 0, -0.5, 0.11, M_HUDG), 0.95), end_a)
vis(fit_w(hud_text('Merry Christmas', 0, -0.62, 0.075, M_HUDW), 0.95), end_a)
vis(fit_w(hud_text('Dance of the Sugar Plum Fairy', 0, -0.71, 0.055, M_HUDP), 0.95), end_a)
vis(hud_text(f'灯り{CNT[-1]}個 メリークリスマス！', 0, -0.78, 0.04, M_HUDW), end_a)
eb_ = obj(prim('plane', M_HUDBG), 'endbg'); eb_.parent = cam; eb_.location = (0, -0.63, -HUD_D - 0.01); eb_.scale = (0.8, 0.2, 1); vis(eb_, end_a)

EV_OUT = dict(pieces=[dict(kind=p['kind'], t=p['t'], top=list(map(float, p['top'])), n=list(map(float, p['n'])), hw=float(p['hw']), hd=float(p['hd']), th=float(p['th']), ball=p['ball'],
                           rs=p.get('rs'), own=p.get('own'), life=p.get('life')) for p in PIECES], counter=dict(final=int(CNT[-1])))
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
        print('t', tt_, 'FOC', np.round(FOC[f], 2), 'W', round(WID[f], 2), 'fairy', np.round(POS['fairy'][f], 2), 'mint', np.round(POS['mint'][f], 2), 'cocoa', np.round(POS['cocoa'][f], 2), 'cnt', CNT[f])
