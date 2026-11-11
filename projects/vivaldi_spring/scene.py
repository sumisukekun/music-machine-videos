"""Vivaldi 'Spring' - 「春をはこぶビー玉」, side view of a snowy terraced hillside that turns into spring behind the balls.
Every melody note = the green 'spring' ball landing on an ice key of that pitch (front terrace); violin 2 = the pink ball
(middle terrace); cello = the gold 'earth' ball on tree-stump drums (back terrace); the birdsong = two little bird balls on the
bud bells of the cherry branches. layout.py plans all motion (constant gravity, bounces that never add speed, icicles, nest
levers, leaf, ropeway, trapdoors); this file builds and animates it.
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
G = LAY['G']; EV = LAY['ev']; LN = LAY['lanes']; KL, KS = LAY['key_lead'], LAY['key_stay']
TB0, TB1 = E['b_start'], E['b_end']
rng = np.random.default_rng(11)

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
BP = {n: BallPath(d) for n, d in LAY['balls'].items()}
POS = {n: np.array([b.at(t) for t in T]) for n, b in BP.items()}
MAIN = ('mel', 'har', 'bas'); BIRDS = ('birdA', 'birdB')

# ---------------- season: 0 = deep winter ... 1 = full spring (follows the music) ----------------
SEAS = np.clip((T - 1.0) / (FINAL - 1.0), 0, 1) ** 0.9
SEAS = np.maximum(SEAS, 0.0); BLOOM = np.clip((T - FINAL) / 1.2, 0, 1)
# melt front along x: the furthest point any ball has reached (+ a little), the snow melts behind it
FRONT = np.maximum.accumulate(np.max(np.stack([POS[n][:, 0] for n in BP]), 0)) + 0.6
FRONT[T < E['t0'] - 0.02] = -30.0
FRONT[T >= FINAL] = FRONT[T >= FINAL] + 60 * BLOOM[T >= FINAL] ** 2

# ---------------- materials ----------------
M_MEL = mat('mel', (0.25, 0.85, 0.2), 0.1, **{'Coat Weight': 1.0, 'Emission Color': (0.4, 1.0, 0.3), 'Emission Strength': 0.5})
M_HAR = mat('har', (1.0, 0.35, 0.6), 0.1, **{'Coat Weight': 1.0, 'Emission Color': (1.0, 0.4, 0.7), 'Emission Strength': 0.5})
M_BAS = mat('bas', (1.0, 0.7, 0.22), 0.2, **{'Metallic': 0.85, 'Coat Weight': 0.6})
M_BA = mat('birdA', (1.0, 0.85, 0.1), 0.15, **{'Coat Weight': 1.0, 'Emission Color': (1.0, 0.8, 0.2), 'Emission Strength': 0.5})
M_BB = mat('birdB', (0.2, 0.6, 1.0), 0.15, **{'Coat Weight': 1.0, 'Emission Color': (0.3, 0.6, 1.0), 'Emission Strength': 0.5})
M_WHITE = mat('white', (0.95, 0.95, 0.95), 0.3)
M_BLACK = mat('black', (0.01, 0.01, 0.01), 0.2, **{'Coat Weight': 1.0})
M_BEAK = mat('beak', (1.0, 0.45, 0.05), 0.3)
M_GOLD = mat('gold', (1.0, 0.72, 0.28), 0.22, **{'Metallic': 0.9})
M_KEY = glowmat('icekey', gain=9.0, base=0.35, rough=0.08, coat=1.0)
M_DRUM = glowmat('drumhead', gain=6.0, base=0.25, rough=0.35)
M_BUD = glowmat('bud', gain=8.0, base=0.3, rough=0.2, coat=0.5)
M_CUP = glowmat('cupbell', gain=6.0, base=0.15, rough=0.2, metal=0.8)
M_BARK = mat('bark', (0.2, 0.11, 0.06), 0.8)
M_WOOD = mat('wood', (0.5, 0.3, 0.14), 0.6)
M_STEEL = mat('steel', (0.6, 0.62, 0.68), 0.3, **{'Metallic': 0.9})
M_CABLE = mat('cable', (0.15, 0.15, 0.17), 0.4, **{'Metallic': 0.6})
M_ICE = mat('ice', (0.85, 0.95, 1.0), 0.05, **{'Transmission Weight': 1.0, 'IOR': 1.31, 'Emission Color': (0.5, 0.8, 1.0), 'Emission Strength': 0.15})
M_SNOWF = mat('snowflat', (0.85, 0.9, 1.0), 0.7)
M_POST = mat('post', (0.35, 0.4, 0.5), 0.4, **{'Metallic': 0.5})
M_OBJ = objcol('objcol', 0.5)
M_PET = objcol('petal', 0.45, emit=0.25, coat=0.3)
M_LAMP = objcol('lampcol', 0.4, emit=5.0)
M_FX = fxmat('fx', 8.0)
M_TXT = emis('txtw', (1.0, 1.0, 1.0), 3.0); M_TXT_G = emis('txtg', (1.0, 0.85, 0.35), 4.5); M_TXT_P = emis('txtp', (1.0, 0.6, 0.8), 4.0)
M_TXT_GR = emis('txtgr', (0.55, 1.0, 0.45), 3.5)
M_SIGN = mat('sign', (0.3, 0.17, 0.08), 0.6)

# ---------------- balls ----------------
BALL = {}
COLB = {'mel': M_MEL, 'har': M_HAR, 'bas': M_BAS, 'birdA': M_BA, 'birdB': M_BB}
BAND = {'mel': M_WHITE, 'har': M_WHITE, 'bas': M_MEL}
for n, b in BP.items():
    rr = b.r
    bpy.ops.mesh.primitive_uv_sphere_add(radius=rr, segments=32, ring_count=16); o = bpy.context.object; o.name = n
    for p in o.data.polygons: p.use_smooth = True
    o.data.materials.append(COLB[n])
    key_loc(o, POS[n])
    if n in MAIN:
        bpy.ops.mesh.primitive_torus_add(major_radius=rr * 1.0, minor_radius=rr * 0.12, major_segments=32, minor_segments=8)
        hb = bpy.context.object; hb.parent = o; hb.data.materials.append(BAND[n])
        sp = np.zeros(NF)
        for f in range(1, NF): sp[f] = sp[f - 1] + (POS[n][f, 0] - POS[n][f - 1, 0]) / rr
        o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, sp)
    else:                                                    # a little bird: beak, eyes, tail tuft; faces +x
        bk = obj(prim('cone', M_BEAK, 12, 1.0, 0.0, 1.0), 'beak'); bk.parent = o
        bk.location = (rr * 1.05, 0, 0.0); bk.rotation_euler = (0, math.pi / 2, 0); bk.scale = (rr * 0.32, rr * 0.32, rr * 0.55)
        for yy in (-1, 1):
            ey = sphere((rr * 0.62, yy * rr * 0.5, rr * 0.45), rr * 0.16, M_BLACK, (10, 6)); ey.parent = o
            ey.location = (rr * 0.62, yy * rr * 0.55, rr * 0.42)
        tl = obj(prim('cone', COLB[n], 10, 1.0, 0.1, 1.0), 'tail'); tl.parent = o
        tl.location = (-rr * 1.15, 0, rr * 0.3); tl.rotation_euler = (0, -math.pi / 2 - 0.5, 0); tl.scale = (rr * 0.35, rr * 0.18, rr * 0.6)
        wg = obj(prim('leaf', COLB[n]), 'wing'); wg.parent = o
        wg.location = (-rr * 0.2, -rr * 0.95, rr * 0.1); wg.scale = (rr * 1.1, rr * 1.4, rr * 1.0); wg.rotation_euler = (0.2, 0, math.pi + 0.2)
        # wing flap while in the air
        fl = np.zeros(NF)
        for f in range(1, NF - 1):
            vz = (POS[n][f + 1, 2] - POS[n][f - 1, 2]) * FPS / 2
            fl[f] = 0.2 + 0.5 * math.sin(T[f] * 40) * min(1, abs(vz) / 1.5)
        wg.rotation_mode = 'XYZ'; key_series(wg, 'rotation_euler', 0, fl, 1e-3)
    BALL[n] = o


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


# ---------------- terraces (one per main lane), snow -> grass behind the melt front ----------------
def terrain_mat(name, grass, snow=(0.88, 0.92, 1.0), wall=False):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; b = nt.nodes['Principled BSDF']
    b.inputs['Roughness'].default_value = 0.8
    geo = nt.nodes.new('ShaderNodeNewGeometry'); sep = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(geo.outputs['Position'], sep.inputs[0])
    val = nt.nodes.new('ShaderNodeValue'); val.name = 'front'
    nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 0.9; nz.inputs['Detail'].default_value = 3.0
    nt.links.new(geo.outputs['Position'], nz.inputs['Vector'])
    sub = nt.nodes.new('ShaderNodeMath'); sub.operation = 'SUBTRACT'; nt.links.new(val.outputs[0], sub.inputs[0]); nt.links.new(sep.outputs['X'], sub.inputs[1])
    mad = nt.nodes.new('ShaderNodeMath'); mad.operation = 'MULTIPLY_ADD'; mad.inputs[1].default_value = 1.6
    nt.links.new(nz.outputs['Fac'], mad.inputs[0]); nt.links.new(sub.outputs[0], mad.inputs[2])
    mr = nt.nodes.new('ShaderNodeMapRange'); mr.inputs['From Min'].default_value = 0.95; mr.inputs['From Max'].default_value = 1.25
    nt.links.new(mad.outputs[0], mr.inputs['Value'])
    # grass colour varies with a second noise (meadow patches)
    nz2 = nt.nodes.new('ShaderNodeTexNoise'); nz2.inputs['Scale'].default_value = 5.0; nt.links.new(geo.outputs['Position'], nz2.inputs['Vector'])
    gr = nt.nodes.new('ShaderNodeValToRGB'); gr.color_ramp.elements[0].color = (*[c * 0.55 for c in grass], 1); gr.color_ramp.elements[1].color = (*grass, 1)
    nt.links.new(nz2.outputs['Fac'], gr.inputs['Fac'])
    mix = nt.nodes.new('ShaderNodeMixRGB'); mix.inputs['Color1'].default_value = (*snow, 1)
    nt.links.new(mr.outputs[0], mix.inputs['Fac']); nt.links.new(gr.outputs['Color'], mix.inputs['Color2'])
    nt.links.new(mix.outputs[0], b.inputs['Base Color'])
    key_val(val.outputs[0], FRONT, 2)
    return m
M_TER = terrain_mat('terrace', (0.3, 0.62, 0.15))
M_WALL = terrain_mat('rockwall', (0.24, 0.3, 0.13), snow=(0.8, 0.84, 0.92))
_nt = M_WALL.node_tree; _b = _nt.nodes['Principled BSDF']
_vo = _nt.nodes.new('ShaderNodeTexVoronoi'); _vo.inputs['Scale'].default_value = 2.5
_gm = _nt.nodes.new('ShaderNodeNewGeometry'); _nt.links.new(_gm.outputs['Position'], _vo.inputs['Vector'])
_bp = _nt.nodes.new('ShaderNodeBump'); _bp.inputs['Strength'].default_value = 0.6; _nt.links.new(_vo.outputs['Distance'], _bp.inputs['Height']); _nt.links.new(_bp.outputs[0], _b.inputs['Normal'])
M_WALL_UNUSED = bpy.data.materials.new('stonewall'); M_WALL_UNUSED.use_nodes = True; _nt = M_WALL_UNUSED.node_tree; _b = _nt.nodes['Principled BSDF']
_br = _nt.nodes.new('ShaderNodeTexBrick'); _br.inputs['Scale'].default_value = 1.2; _br.inputs['Mortar Size'].default_value = 0.025
_br.inputs['Color1'].default_value = (0.42, 0.4, 0.38, 1); _br.inputs['Color2'].default_value = (0.3, 0.29, 0.28, 1); _br.inputs['Mortar'].default_value = (0.12, 0.11, 0.1, 1)
_br.offset = 0.5; _br.squash = 1.0
_tc = _nt.nodes.new('ShaderNodeTexCoord'); _mp = _nt.nodes.new('ShaderNodeMapping'); _mp.inputs['Rotation'].default_value = (math.pi / 2, 0, 0)
_nt.links.new(_tc.outputs['Object'], _mp.inputs['Vector']); _nt.links.new(_mp.outputs[0], _br.inputs['Vector'])
_nt.links.new(_br.outputs['Color'], _b.inputs['Base Color']); _b.inputs['Roughness'].default_value = 0.85
_bp = _nt.nodes.new('ShaderNodeBump'); _bp.inputs['Strength'].default_value = 0.4; _nt.links.new(_br.outputs['Fac'], _bp.inputs['Height']); _nt.links.new(_bp.outputs[0], _b.inputs['Normal'])
LANE_GROUND = {}
for role in MAIN:
    b = BP[role]; r = b.r
    pts = sorted([(h['p'][0], h['p'][2]) for h in b.hits])
    # under the ropeway the ground follows a straight line between the cadence and the ritornello start
    cx, cz = EV['rope'][role]['pts'][0][0], EV['rope'][role]['pts'][0][2]
    dx_, dz_ = EV['rope'][role]['pts'][-1][0], EV['rope'][role]['pts'][-1][2] - 0.5 * G * LAY['t_drop'] ** 2
    for x in np.arange(cx, dx_, 0.5): pts.append((x, cz + (dz_ - cz) * (x - cx) / (dx_ - cx)))
    pts.sort()
    AX = np.array([p[0] for p in pts]); AZ = np.array([p[1] for p in pts])
    X = np.arange(-16.0, AX.max() + 22, 0.4)
    Zg = np.array([AZ[np.abs(AX - x) < 1.4].min() if (np.abs(AX - x) < 1.4).any() else np.nan for x in X])
    ok_ = ~np.isnan(Zg); Zg = np.interp(X, X[ok_], Zg[ok_]) - r - 0.42
    for _ in range(4): Zg = np.minimum(Zg, np.convolve(np.pad(Zg, 2, mode='edge'), np.ones(5) / 5, 'valid'))
    LANE_GROUND[role] = (X, Zg)
def ground(role, x): X, Z = LANE_GROUND[role]; return float(np.interp(x, X, Z))
YE = {'mel': (-1.6, 0.85), 'har': (0.85, 2.55), 'bas': (2.55, 4.2)}
for role in MAIN:
    X, Z = LANE_GROUND[role]
    strip_mesh(X, Z, YE[role][0], YE[role][1], 7.0, M_TER, 'terrace_' + role, m_front=M_WALL)
# orchard terrace behind (trunks of the bird trees stand here)
Xo = LANE_GROUND['bas'][0]; Zo = LANE_GROUND['bas'][1] + 0.55
strip_mesh(Xo, Zo, 4.2, 10.0, 8.0, M_TER, 'orchard', m_front=M_WALL)
# near snow bank in front of the camera
strip_mesh(LANE_GROUND['mel'][0], LANE_GROUND['mel'][1] - 0.6, -4.5, -1.6, 4.0, M_TER, 'nearbank')

# ---------------- keys: ice keys (mel / har), stump drums (bas), bud bells (birds) ----------------
PIECES = []
FLOWERS = []          # (t, location, colour, size)
KEY_TH = 0.1
for role in MAIN:
    b = BP[role]; r = b.r; ext = LN[role]['ext']
    for h in b.hits:
        if h['kind'] not in ('key', 'drum', 'lever'): continue
        nv = np.array(h['n']); top = np.array(h['p']) - r * nv; col = pcol(h['pitch'])
        zf = ground(role, top[0])
        if role == 'bas':
            hd = obj(prim('cyl', M_DRUM, 24), 'drumhead')
            hd.matrix_world = Matrix.Translation(Vector(top - nv * 0.03)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((ext[0] + 0.03, ext[0] + 0.03, 0.06, 1))
            st = rod(top - nv * 0.06, (top[0], top[1], zf - 0.2), ext[0] + 0.01, M_BARK, 16)
            ring = obj(prim('cyl', M_GOLD, 24), 'drumring'); ring.matrix_world = Matrix.Translation(Vector(top - nv * 0.07)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((ext[0] + 0.05, ext[0] + 0.05, 0.04, 1))
            parts = [hd, st, ring]; flash(hd, h['t'], col, before=(0.85, 0.8, 0.75)); fo = hd
        else:
            ko = obj(prim('bcube', M_KEY), 'key')
            ko.matrix_world = Matrix.Translation(Vector(top - nv * KEY_TH / 2)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((2 * ext[0], 2 * ext[1], KEY_TH, 1))
            po = rod(top - nv * KEY_TH, (top[0], top[1], zf - 0.2), 0.03, M_POST, 8)
            parts = [ko, po]; flash(ko, h['t'], col); fo = ko
        bump(fo, h['t'], -nv * 0 + nv, 0.025)
        t_in = h['t'] - KL
        popper(parts, (top[0], top[1], zf - 0.2), t_in, h['t'] + KS - 0.3, grow=0.3, shrink=0.3)
        PIECES.append(dict(t=h['t'], top=top, kind=h['kind'], role=role, col=col))
        if role != 'bas':
            FLOWERS.append((h['t'], (top[0] + rng.uniform(-0.1, 0.1), top[1] - LN[role]['ext'][1] - 0.22 - rng.uniform(0, 0.25), zf), pcol(h['pitch'], 0.2), 0.2 + 0.05 * rng.random()))
            FLOWERS.append((h['t'] + 0.1, (top[0] + rng.uniform(-0.3, 0.3), top[1] - LN[role]['ext'][1] - 0.6 - rng.uniform(0, 0.3), zf), pcol(h['pitch'] + 5, 0.3), 0.17))
            if role == 'har' and rng.random() < 0.5:
                FLOWERS.append((h['t'] + 0.15, (top[0] + rng.uniform(-0.2, 0.2), top[1] + 0.5 + rng.uniform(0, 0.2), zf), pcol(h['pitch'] + 3, 0.35), 0.13))
        else:
            FLOWERS.append((h['t'], (top[0] + rng.uniform(-0.15, 0.15), top[1] - 0.45, zf), (1.0, 0.9, 0.3), 0.15))

# lever under bird A's nest (from the earth ball's bar 13 drum) and under bird B's nest (from bird A's bud bell)
def lever_arm(p_key, nest, t, m=M_WOOD):
    a = np.array(p_key); b_ = np.array(nest)
    piv = link(bpy.data.objects.new('lever', None)); piv.location = tuple((a + b_) / 2)
    o = rod(a - (a + b_) / 2, b_ - (a + b_) / 2 - np.array([0, 0, 0.0]), 0.035, m, 8); o.parent = piv
    pv = sphere(tuple((a + b_) / 2), 0.07, M_GOLD, (12, 6))
    ax = np.cross(b_ - a, [0, 0, 1.0]); ax /= np.linalg.norm(ax) + 1e-9
    rot = np.zeros(NF)
    for f in range(NF):
        u = T[f] - t
        rot[f] = 0.0 if u < 0 else -0.08 * min(1, u / 0.06)
    piv.rotation_mode = 'AXIS_ANGLE'
    for f in range(NF):
        if 0 < f < NF - 1 and rot[f] == rot[f - 1] == rot[f + 1]: continue
        piv.rotation_axis_angle = (float(rot[f]), *ax); piv.keyframe_insert('rotation_axis_angle', frame=f)
    return piv

# bud bells on the bird branches
BR_PTS = {n: [] for n in BIRDS}
for n in BIRDS:
    b = BP[n]; r = b.r; ext = LN[n]['ext']
    for h in b.hits:
        if h['kind'] not in ('bud', 'lever'): continue
        nv = np.array(h['n']); top = np.array(h['p']) - r * nv; col = pcol(h['pitch'], 0.2)
        pad = obj(prim('cyl', M_BUD, 20), 'bud')
        pad.matrix_world = Matrix.Translation(Vector(top - nv * 0.03)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((ext[0] + 0.03, ext[1] + 0.02, 0.06, 1))
        flash(pad, h['t'], col, before=(0.35, 0.6, 0.25))
        bump(pad, h['t'], nv, 0.02)
        zb = top[2] - 0.45
        tw = rod(top - nv * 0.06, (top[0] - 0.12, top[1], zb), 0.022, M_BARK, 6)
        popper([pad, tw], (top[0] - 0.12, top[1], zb), h['t'] - KL, h['t'] + KS - 0.3)
        BR_PTS[n].append((top[0], zb))
        PIECES.append(dict(t=h['t'], top=top, kind='bud', role=n, col=col))
        # the blossom that opens where the bud was (stays on the twig)
        FLOWERS.append((h['t'], (top[0] - 0.12 + rng.uniform(-0.05, 0.05), top[1] - 0.22, zb + 0.12), (1.0, 0.72, 0.85), 0.13))
for n, h_ in (('bas', EV['leverA']), ('birdA', EV['leverB'])):
    p = np.array(h_['p']) - BP[n].r * np.array([0, 0, 1.0])
    nest = np.array(h_['nest']) - [0, 0, BP['birdA' if n == 'bas' else 'birdB'].r + 0.08]
    lever_arm(p + [0.0, 0.25, -0.05], nest + [-0.25, -0.2, 0.0], h_['t'])

# bird branches: a long gnarled branch under each bird lane, held by trunks standing on the orchard terrace
for n in BIRDS:
    pts = sorted(BR_PTS[n])
    for key_ in ('nest', 'home'):
        pass
    xs = np.array([p[0] for p in pts]); zs = np.array([p[1] for p in pts])
    X = np.arange(xs.min() - 2.0, xs.max() + 2.5, 0.6)
    Zb = np.array([zs[np.abs(xs - x) < 1.0].min() if (np.abs(xs - x) < 1.0).any() else np.nan for x in X])
    ok_ = ~np.isnan(Zb); Zb = np.interp(X, X[ok_], Zb[ok_])
    y = LN[n]['y']
    for k in range(len(X) - 1):
        rod((X[k], y, Zb[k] - 0.02), (X[k + 1], y, Zb[k + 1] - 0.02), 0.06, M_BARK, 8)
    for x0 in np.arange(X[0] + 0.5, X[-1], 6.5):
        zt = float(np.interp(x0, X, Zb)); zg = float(np.interp(x0, Xo, Zo))
        rod((x0, y + 0.25, zg - 0.3), (x0, y + 0.05, zt), 0.16, M_BARK, 12, r1=0.08)

# ---------------- flowers: each opens at its hit and stays ----------------
for t, loc, col, sz in FLOWERS:
    st = rod((loc[0], loc[1], loc[2] - 0.05), (loc[0], loc[1], loc[2] + sz * 1.3), 0.012, M_MEL, 5)
    fl = obj(prim('flower', M_PET), 'flower'); fl.location = (loc[0], loc[1], loc[2] + sz * 1.3); fl.scale = (sz, sz, sz)
    fl.rotation_euler = (-0.9, rng.uniform(-0.3, 0.3), rng.uniform(0, 6.28)); fl.color = (*col, 1)
    ce = obj(prim('fcentre', M_LAMP), 'fcentre'); ce.parent = fl; ce.color = (1.0, 0.85, 0.2, 1)
    popper([st, fl], (loc[0], loc[1], loc[2] - 0.05), t + 0.02, None, grow=0.35)

# ---------------- hit rings ----------------
NRING = 14
ring_loc = np.zeros((NRING, NF, 3)); ring_a = np.zeros((NRING, NF)); ring_s = np.full((NRING, NF), 1e-3); ring_c = [(1, 1, 1)] * NRING
for k, P_ in enumerate(sorted(PIECES, key=lambda b: b['t'])):
    r = k % NRING; f0 = int(round(P_['t'] * FPS))
    for df in range(14):
        f = f0 + df
        if f >= NF: break
        u = df / 13; ring_loc[r, f] = P_['top'] + np.array([0, -0.3, 0.02]); ring_s[r, f] = 0.3 + 1.2 * (1 - (1 - u) ** 2)
        ring_a[r, f] = (1 - u) ** 1.6
    ring_c[r] = P_['col']
for r in range(NRING):
    bpy.ops.mesh.primitive_torus_add(major_radius=0.25, minor_radius=0.014, major_segments=32, minor_segments=5, rotation=(math.pi / 2, 0, 0))
    o = bpy.context.object; o.data.materials.append(M_FX); o.color = (1, 1, 1, 0)
    for f in range(1, NF):
        if ring_a[r, f] == 0: ring_loc[r, f] = ring_loc[r, f - 1]
    key_loc(o, ring_loc[r]); key_scale(o, ring_s[r]); key_series(o, 'color', 3, ring_a[r], 1e-3)
# per-hit colour of the ring pool: one colour per ring object is enough (the flash colour is the key's own)

# ---------------- opening: snow cloud and the three ice blocks ----------------
M_CLOUD = mat('cloud', (0.85, 0.88, 0.95), 0.9)
cx0 = 0.0
for role in MAIN:
    pc = np.array(EV['icicles'][role]); r = BP[role].r
    for k in range(6):
        sphere((pc[0] + rng.uniform(-0.55, 0.55), pc[1] + rng.uniform(0.1, 0.4), pc[2] + r + 0.62 + rng.uniform(0, 0.15)), rng.uniform(0.18, 0.3), M_CLOUD, (16, 8))
    t_rel = LAY['sfx'][0][1]
    for side in (-1, 1):
        hb = box((pc[0] + side * (r + 0.04) / 2, pc[1], pc[2]), ((r + 0.06), 2 * r + 0.12, 2 * r + 0.2), M_ICE, name='iceblock')
        P = np.zeros((NF, 3)); A = np.ones(NF)
        for f in range(NF):
            u = T[f] - t_rel
            P[f] = hb.location if u < 0 else Vector(hb.location) + Vector((side * 1.2 * u, -0.6 * u, -0.5 * G * u * u + 0.8 * u))
        key_loc(hb, P); key_scale(hb, np.where(T < t_rel + 0.7, 1.0, 0.001), tuple(hb.scale))
    ic = obj(prim('cone', M_ICE, 10, 1.0, 0.0, 1.0), 'icicle'); ic.location = (pc[0], pc[1], pc[2] + r + 0.33); ic.scale = (0.08, 0.08, 0.3)
    ic.rotation_euler = (math.pi, 0, 0)
    key_scale(ic, np.where(T < t_rel, 1.0, 0.001), (0.08, 0.08, 0.3))
# burst of ice shards at the release
for k in range(30):
    role = MAIN[k % 3]; pc = np.array(EV['icicles'][role]); t_rel = LAY['sfx'][0][1]
    sh = obj(prim('ico', M_FX, 1), 'shard'); sh.color = (0.7, 0.9, 1.0, 0)
    a0 = rng.uniform(0, 2 * math.pi); sp_ = rng.uniform(1.0, 2.5)
    P = np.zeros((NF, 3)); A = np.zeros(NF)
    for f in range(NF):
        u = T[f] - t_rel
        if u < 0: P[f] = pc; continue
        P[f] = pc + np.array([sp_ * math.cos(a0) * u, -0.5 * u, sp_ * math.sin(a0) * u - 0.5 * G * u * u])
        A[f] = max(0, 1 - u / 0.8) * 0.8
    key_loc(sh, P); key_scale(sh, np.full(NF, 0.04)); key_series(sh, 'color', 3, A, 1e-3)

# ---------------- nests, leaves, homes ----------------
M_NEST = mat('nest', (0.45, 0.3, 0.15), 0.9)
def nest(c, r, t_open=None, name='nest'):
    """a bowl of twigs; the floor flap (under the ball) swings open at t_open"""
    c = np.array(c)
    bpy.ops.mesh.primitive_torus_add(major_radius=r + 0.1, minor_radius=0.07, major_segments=24, minor_segments=8, location=tuple(c - [0, 0, r * 0.35]))
    bpy.context.object.data.materials.append(M_NEST)
    bpy.ops.mesh.primitive_torus_add(major_radius=r + 0.05, minor_radius=0.05, major_segments=24, minor_segments=8, location=tuple(c - [0, 0, r * 0.7]))
    bpy.context.object.data.materials.append(M_NEST)
    if t_open is None:
        box(tuple(c - [0, 0, r + 0.025]), (2 * r + 0.1, 2 * r + 0.1, 0.05), M_NEST)
        return
    hinge = link(bpy.data.objects.new('nesthinge', None)); hinge.location = tuple(c - [r + 0.06, 0, r + 0.03])
    fl = box((r + 0.06, 0, 0.0), (2 * r + 0.1, 2 * r + 0.1, 0.05), M_NEST, name='nestflap'); fl.parent = hinge
    ang = np.array([0.0 if t < t_open - 0.02 else min(1.0, (t - t_open + 0.02) / 0.15) ** 2 * 1.7 for t in T])
    hinge.rotation_mode = 'XYZ'; key_series(hinge, 'rotation_euler', 1, ang)
for n, key_ in (('birdA', 'nestA'), ('birdB', 'nestB')):
    nd = EV[key_]; r = BP[n].r
    nest(nd['nest'], r, nd['t_drop'])
    nest(nd['home'], r)
    for lf in nd['leaves']:
        # zip line: the basket waits at the top station, glides with the bird, hits the stopper
        p0 = np.array(lf['p']); ps = np.array(lf['p_stop']); dv = np.array(lf['dir'])
        hgt = r + 0.55
        c0 = p0 + [0, 0, hgt] - dv * 0.6; c1 = ps + [0, 0, hgt] + dv * 0.25
        rod(c0, c1, 0.015, M_CABLE, 6)
        for c_ in (c0, c1):
            zg = float(np.interp(c_[0], Xo, Zo))
            rod((c_[0], c_[1] + 0.2, zg - 0.3), (c_[0], c_[1] + 0.2, c_[2] + 0.2), 0.05, M_BARK, 8)
            rod((c_[0], c_[1] + 0.2, c_[2] + 0.1), (c_[0], c_[1], c_[2] + 0.1), 0.03, M_BARK, 6)
        box(tuple(ps + [0, 0, hgt] + dv * 0.18), (0.08, 0.12, 0.12), M_GOLD, name='stopper')
        bk = link(bpy.data.objects.new('basket', None))
        BKP = np.zeros((NF, 3))
        for f in range(NF):
            t = T[f]
            BKP[f] = p0 if t < lf['t_land'] else (POS[n][f] if t <= lf['t_stop'] else ps)
        key_loc(bk, BKP)
        bpy.ops.mesh.primitive_torus_add(major_radius=r + 0.06, minor_radius=0.03, major_segments=24, minor_segments=6, location=(0, 0, -r * 0.4))
        o_ = bpy.context.object; o_.data.materials.append(M_NEST); o_.parent = bk
        box((0, 0, -r - 0.02), (2 * r + 0.1, 2 * r + 0.1, 0.04), M_NEST).parent = bk
        for dx in (-1, 1): rod((dx * (r + 0.06), 0, -r * 0.4), (0, 0, hgt), 0.012, M_BARK, 5).parent = bk
        sphere((0, 0, hgt), 0.04, M_GOLD, (8, 6)).parent = bk
        text('ジップライン', tuple(p0 + [0.9, -0.5, hgt + 0.35]), 0.22, M_TXT)

# ---------------- ropeway: cables, stations, gondolas with cup bells and trapdoors ----------------
HANG = 0.95
GCOL = {'mel': (0.3, 0.8, 0.3), 'har': (1.0, 0.45, 0.65), 'bas': (1.0, 0.75, 0.3)}
for role in MAIN:
    rp = EV['rope'][role]; r = BP[role].r; pts = [np.array(p) for p in rp['pts']]
    cab = [p + [0, 0, r + HANG] for p in pts]
    for k in range(3): rod(cab[k], cab[k + 1], 0.018, M_CABLE, 6)
    for k, p in enumerate(cab):
        zg = ground(role, p[0])
        rod((p[0], p[1] + 0.35, zg - 0.2), (p[0], p[1] + 0.35, p[2] + 0.35), 0.06, M_STEEL, 8)
        rod((p[0], p[1] + 0.35, p[2] + 0.3), (p[0], p[1], p[2] + 0.3), 0.04, M_STEEL, 6)
        wh = link(bpy.data.objects.new('wheel', None)); wh.location = (p[0], p[1], p[2] + 0.12)
        w_ = obj(prim('cyl', M_GOLD, 20), 'wheelo'); w_.parent = wh; w_.scale = (0.18, 0.18, 0.05); w_.rotation_euler = (math.pi / 2, 0, 0)
        for j in range(4):
            sp_ = box((0, 0, 0), (0.34, 0.03, 0.03), M_STEEL, rot=(0, j * math.pi / 4, 0)); sp_.parent = wh
        # the pulley turns with the cable: angle = distance travelled / radius
        s_ = np.array([np.linalg.norm(POS[role][f] - pts[0]) if rp['t_go'] <= T[f] else 0.0 for f in range(NF)])
        trav = np.zeros(NF)
        for f in range(1, NF):
            trav[f] = trav[f - 1] + (np.linalg.norm(POS[role][f] - POS[role][f - 1]) if rp['t_go'] <= T[f] <= rp['t_stop'] + 0.05 else 0.0)
        wh.rotation_mode = 'XYZ'; key_series(wh, 'rotation_euler', 1, trav / 0.18, 1e-3)
    # gondola rides with the ball between t_go and t_stop (the ball sits in the cup), stays at the end station after
    gd = link(bpy.data.objects.new('gondola', None))
    GP = np.zeros((NF, 3))
    for f in range(NF):
        t = T[f]
        GP[f] = pts[0] if t < rp['t_go'] else (POS[role][f] if t < rp['t_open'] else pts[-1])
    key_loc(gd, GP)
    bpy.ops.mesh.primitive_torus_add(major_radius=r + 0.07, minor_radius=0.035, major_segments=28, minor_segments=8, location=(0, 0, -r * 0.45))
    cup = bpy.context.object; cup.data.materials.append(M_CUP); cup.parent = gd; cup.color = (*GCOL[role], 0)
    bpy.ops.mesh.primitive_torus_add(major_radius=r + 0.05, minor_radius=0.03, major_segments=28, minor_segments=8, location=(0, 0, -r * 0.05))
    c2 = bpy.context.object; c2.data.materials.append(M_CUP); c2.parent = gd; c2.color = (*GCOL[role], 0)
    for o_ in (cup, c2): flash(o_, TB0, (1.0, 0.85, 0.4), before=GCOL[role])
    for side in (-1, 1):
        hg = link(bpy.data.objects.new('flaphinge', None)); hg.parent = gd; hg.location = (side * (r + 0.09), 0, -r - 0.03)
        fp = box((-side * (r + 0.09) / 2, 0, 0), (r + 0.09, 2 * r + 0.2, 0.04), M_WOOD, name='flap'); fp.parent = hg
        ang = np.array([0.0 if t < rp['t_open'] - 0.02 else -side * min(1.0, (t - rp['t_open'] + 0.02) / 0.1) * 1.5 for t in T])
        hg.rotation_mode = 'XYZ'; key_series(hg, 'rotation_euler', 1, ang)
    for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        rod((dx * (r + 0.12), dy * (r + 0.12), -r - 0.05), (dx * (r + 0.12), dy * (r + 0.12), r + 0.45), 0.02, M_STEEL, 6).parent = gd
    roof = box((0, 0, r + 0.5), (2 * r + 0.4, 2 * r + 0.4, 0.07), M_OBJ, name='roof'); roof.parent = gd; roof.color = (*GCOL[role], 1)
    rod((0, 0, r + 0.5), (0, 0, r + HANG), 0.025, M_STEEL, 6).parent = gd
    tr_ = sphere((0, 0, r + HANG), 0.07, M_GOLD, (10, 6)); tr_.parent = gd

# ---------------- background: cherry trees (snow-capped -> blossoms behind the front), mountains ----------------
CLUS = []
for k, x0 in enumerate(np.arange(-8.0, Xo[-1] - 4, 4.2)):
    y0 = 9.0 + rng.uniform(-0.6, 1.6); zg = float(np.interp(x0, Xo, Zo)); hgt = rng.uniform(2.6, 3.6)
    rod((x0, y0, zg - 0.5), (x0, y0, zg + hgt), 0.2, M_BARK, 10, r1=0.1)
    for j in range(4):
        a0 = rng.uniform(0, 2 * math.pi); L_ = rng.uniform(1.0, 1.8)
        tip = (x0 + L_ * math.cos(a0), y0 + 0.4 * L_ * math.sin(a0), zg + hgt * rng.uniform(0.6, 0.95) + 0.6 * L_)
        rod((x0, y0, zg + hgt * 0.55), tip, 0.08, M_BARK, 6, r1=0.03)
        CLUS.append((x0, tip, rng.uniform(0.35, 0.55)))
    CLUS.append((x0, (x0, y0, zg + hgt + 0.3), 0.7))
SAK = [(1.0, 0.72, 0.82), (1.0, 0.82, 0.88), (0.98, 0.62, 0.76)]
for k, (x0, c, s) in enumerate(CLUS):
    o = obj(prim('ico', M_PET, 2), 'blossom'); o.location = c; o.scale = (s * 1.2, s * 0.9, s * 0.8)
    # snow white until the melt front passes the tree, then blossom pink
    f_pass = int(np.searchsorted(FRONT, x0 - 1.0))
    sk = SAK[k % 3]
    o.color = (*SNOWC, 1); o.keyframe_insert('color', frame=0)
    if f_pass < NF:
        o.keyframe_insert('color', frame=f_pass); o.color = (*sk, 1); o.keyframe_insert('color', frame=min(NF - 1, f_pass + 20))
        sc_ = np.array([1.0 if f < f_pass else 1.0 + 0.25 * min(1, (f - f_pass) / 20) for f in range(NF)])
        key_scale(o, sc_, (s * 1.2, s * 0.9, s * 0.8))
M_MTN = mat('mtn', (0.35, 0.45, 0.7), 0.9)
for k, x0 in enumerate(np.arange(-30, Xo[-1] + 40, 16.0)):
    o = sphere((x0 + rng.uniform(-5, 5), 55, float(np.interp(min(max(x0, Xo[0]), Xo[-1]), Xo, Zo)) - 12), 1.0, M_MTN, (24, 12))
    o.scale = (rng.uniform(16, 24), 8, rng.uniform(10, 15))
    cap = sphere(tuple(o.location + Vector((0, -1.5, o.scale[2] * 0.62))), 1.0, M_SNOWF, (20, 10)); cap.scale = (o.scale[0] * 0.45, 6, o.scale[2] * 0.42)

# ---------------- the giant cherry tree and the three golden bells (final chord) ----------------
FB = EV['final_bells']
fx_ = np.mean([FB[r]['p'][0] for r in MAIN]); fz_ = max(FB[r]['p'][2] for r in MAIN)
TRUNK = np.array([fx_ + 1.6, 6.0, float(np.interp(fx_, Xo, Zo))])
rod(tuple(TRUNK - [0, 0, 1.0]), tuple(TRUNK + [0, 0, fz_ - TRUNK[2] + 4.0]), 0.75, M_BARK, 20, r1=0.4)
GT = []
for role in MAIN:
    p = np.array(FB[role]['p']); r = BP[role].r
    bl = obj(prim('bell', M_CUP, 32), 'finbell')
    bl.matrix_world = Matrix.Translation(Vector(p - [0, 0, r + 0.02 - 0.3])) @ Matrix.Diagonal((0.34, 0.34, -0.3, 1))
    bl.color = (1.0, 0.75, 0.3, 0)
    key_series(bl, 'color', 3, np.maximum(0.05, pulse_alpha([FINAL], decay=25, length=120)), 1e-3)
    # a branch from the trunk to the bell
    rod(tuple(p - [0, 0, r + 0.32]), tuple(TRUNK + [0, 0, p[2] - TRUNK[2] - 1.0]), 0.12, M_BARK, 8, r1=0.3)
    popper([bl], tuple(p - [0, 0, r + 0.3]), FINAL - 1.6, None)
    PIECES.append(dict(t=FINAL, top=p - [0, 0, r], kind='finbell', role=role, col=(1, 0.8, 0.4)))
crown_c = TRUNK + [0, 0, fz_ - TRUNK[2] + 4.0]
for k in range(70):
    a0 = rng.uniform(0, 2 * math.pi); rr_ = rng.uniform(0.5, 4.2)
    c = crown_c + [rr_ * math.cos(a0) * 1.3, rng.uniform(-1.5, 1.5) + 0.5, rr_ * math.sin(a0) * 0.6 + rng.uniform(-0.5, 1.2)]
    o = obj(prim('ico', M_PET, 2), 'bigbloom'); o.location = tuple(c); o.color = (*SAK[k % 3], 1)
    s = rng.uniform(0.7, 1.3); t_b = FINAL + 0.05 + 0.6 * rr_ / 4.2
    key_scale(o, 0.25 + 0.75 * pop_scale(t_b, None, 0.5, over=1.1), (s, s * 0.8, s * 0.75))
    o.color = (*SNOWC, 1); o.keyframe_insert('color', frame=int(t_b * FPS)); o.color = (*SAK[k % 3], 1); o.keyframe_insert('color', frame=int(t_b * FPS) + 8)
for j in range(5):
    a0 = -0.4 + j * 0.45
    rod(tuple(crown_c - [0, 0, 2.2]), tuple(crown_c + [3.2 * math.cos(a0 + 1.2), 0.3, 2.0 * math.sin(a0 + 1.2) - 0.3]), 0.22, M_BARK, 8, r1=0.08)
fl_ = bpy.data.lights.new('finlight', 'POINT'); fl_.color = (1.0, 0.85, 0.6); fl_.shadow_soft_size = 1.0
flo = link(bpy.data.objects.new('finlight', fl_)); flo.location = tuple(crown_c - [0, 3.0, 1.5])
gg = pulse_alpha([FINAL], decay=45, length=200)
for f in range(0, NF, 2):
    fl_.energy = float(5 + 6000 * gg[f]); fl_.keyframe_insert('energy', frame=f)
# petal storm after the final chord
for k in range(260):
    cf = obj(prim('leaf', M_PET), 'petal'); cf.color = (*SAK[k % 3], 1)
    a0 = rng.uniform(0, 2 * math.pi); spd = rng.uniform(1.0, 4.0); vy = rng.uniform(-2.5, 0.2)
    P = np.zeros((NF, 3)); RT = np.zeros((NF, 3))
    for f in range(NF):
        s_ = T[f] - FINAL
        if s_ < 0: P[f] = crown_c; continue
        burst = 1 - math.exp(-s_ / 0.4)
        P[f] = crown_c + [spd * math.cos(a0) * 1.6 * burst + 0.3 * math.sin(2 * s_ + k), vy * burst - 0.8, spd * math.sin(a0) * 0.9 * burst - 0.7 * s_ - 1.0]
        RT[f] = (s_ * (3 + k % 5), s_ * (2 + k % 3), s_ * 1.5)
    key_loc(cf, P); key_scale(cf, np.where(T >= FINAL, 1.0, 0.001), (0.12, 0.12, 0.12))
    for c_ in range(3): key_series(cf, 'rotation_euler', c_, RT[:, c_])

# ---------------- signs (keywords for search, readable on a phone) ----------------
def sign(lines, x, y, z, w, h, pole_role='mel'):
    box((x, y + 0.08, z), (w, 0.1, h), M_SIGN, rot=(-0.12, 0, 0), name='plate')
    box((x, y + 0.1, z + h / 2 + 0.05), (w + 0.12, 0.2, 0.1), M_SNOWF, rot=(-0.12, 0, 0))      # snow on top
    for xx in (x - w / 2 + 0.25, x + w / 2 - 0.25): rod((xx, y + 0.16, z - h / 2), (xx, y + 0.16, ground(pole_role, xx) - 0.2), 0.05, M_SIGN, 8)
    zz = z + h / 2
    for s_, size, mt in lines:
        zz -= size * 1.28
        text(s_, (x, y - 0.0 + 0.12 * (zz + size * 0.15 - z), zz + size * 0.15), size, mt, rot=(math.pi / 2 - 0.12, 0, 0))
def mx(t): return float(np.interp(t, T, POS['mel'][:, 0]))
def mz(t): return float(np.interp(t, T, POS['mel'][:, 2]))
def osign(lines, x, w, h):
    zg = float(np.interp(x, Xo, Zo)); sign(lines, x, 4.55, zg + 0.35 + h / 2, w, h, pole_role='bas')
osign([('ヴィヴァルディ', 0.42, M_TXT_G), ('「四季」より「春」', 0.46, M_TXT_P), ('第1楽章 × ビー玉からくり', 0.23, M_TXT)], 2.3, 3.6, 1.6)
for tt_, lines, wd in ((6.5, [('バロック音楽', 0.34, M_TXT_G), ('1725年の名曲', 0.28, M_TXT)], 3.0),
                       (12.0, [('ビー玉が通ると', 0.28, M_TXT), ('春になる', 0.4, M_TXT_GR)], 3.0),
                       (18.0, [('クラシック名曲', 0.32, M_TXT_G), ('マーブルマシン', 0.28, M_TXT)], 3.0),
                       (24.0, [('このあと', 0.26, M_TXT), ('鳥のさえずり', 0.38, M_TXT_P)], 3.2)):
    x_ = mx(tt_) + 1.8
    sign(lines, x_, 0.62, ground('mel', x_) + 1.45, wd, 1.0)
for role, lbl, mt in (('mel', '春のメロディ', M_TXT_GR), ('har', '花の和音', M_TXT_P), ('bas', '大地の低音', M_TXT_G)):
    p0 = POS[role][int(2.0 * FPS)]
    text(lbl, (2.3 + (0.6 if role == 'bas' else 0), p0[1] - 0.75, ground(role, 2.3) + 0.55), 0.3, mt)
nA = EV['nestA']['nest']; nB = EV['nestB']['nest']
text('ねむる鳥', (nA[0] - 1.2, nA[1] - 0.6, nA[2] + 0.45), 0.26, M_TXT_G)
text('もう1羽', (nB[0] - 1.1, nB[1] - 0.6, nB[2] + 0.45), 0.26, M_TXT_P)
rp = EV['rope']['har']['pts']
text('ロープウェイ', (rp[1][0] + 1.4, 1.2, rp[1][2] + 1.55), 0.26, M_TXT)
text('満開の桜', (crown_c[0], crown_c[1] - 2.6, crown_c[2] + 2.6), 0.5, M_TXT_P)

# ---------------- sky, sun, falling snow -> falling petals ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes['Background']; bg.inputs['Color'].default_value = (0.5, 0.6, 0.8, 1)
key_val(bg.inputs['Strength'], 0.45 + 0.35 * SEAS, 6)
sun = bpy.data.lights.new('sun', 'SUN'); sun.angle = 0.08
so = link(bpy.data.objects.new('sun', sun)); so.rotation_euler = (math.radians(50), math.radians(-12), math.radians(-30))
for f in range(0, NF, 6):
    s_ = SEAS[f]
    sun.energy = float(2.6 + 1.8 * s_ + 1.2 * BLOOM[f]); sun.keyframe_insert('energy', frame=f)
    sun.color = (0.75 + 0.25 * s_, 0.82 + 0.1 * s_, 1.0 - 0.25 * s_); sun.keyframe_insert('color', frame=f)
fill = bpy.data.lights.new('fill', 'SUN'); fill.energy = 0.5; fill.color = (0.7, 0.8, 1.0)
fo = link(bpy.data.objects.new('fill', fill)); fo.rotation_euler = (math.radians(70), math.radians(30), math.radians(160))
SKY = bpy.data.materials.new('sky'); SKY.use_nodes = True; nt = SKY.node_tree; nt.nodes.clear()
tc = nt.nodes.new('ShaderNodeTexCoord'); sepz = nt.nodes.new('ShaderNodeSeparateXYZ')
rw = nt.nodes.new('ShaderNodeValToRGB'); rw.color_ramp.elements[0].color = (0.7, 0.8, 0.98, 1); rw.color_ramp.elements[1].color = (0.2, 0.35, 0.8, 1)
rs = nt.nodes.new('ShaderNodeValToRGB'); rs.color_ramp.elements[0].color = (1.0, 0.75, 0.6, 1); rs.color_ramp.elements[1].color = (0.25, 0.55, 1.0, 1)
e2 = rs.color_ramp.elements.new(0.4); e2.color = (0.75, 0.85, 1.0, 1)
mixs = nt.nodes.new('ShaderNodeMixRGB'); em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Strength'].default_value = 1.0
out = nt.nodes.new('ShaderNodeOutputMaterial')
nt.links.new(tc.outputs['Generated'], sepz.inputs[0]); nt.links.new(sepz.outputs['Y'], rw.inputs['Fac']); nt.links.new(sepz.outputs['Y'], rs.inputs['Fac'])
nt.links.new(rw.outputs['Color'], mixs.inputs['Color1']); nt.links.new(rs.outputs['Color'], mixs.inputs['Color2'])
nt.links.new(mixs.outputs[0], em.inputs['Color']); nt.links.new(em.outputs[0], out.inputs[0])
key_val(mixs.inputs['Fac'], np.minimum(1, SEAS * 1.15), 6)
SKYX0, SKYX1 = Xo[0] - 30, Xo[-1] + 40
skyo = obj(prim('plane', SKY), 'sky'); skyo.location = ((SKYX0 + SKYX1) / 2, 70, 0); skyo.rotation_euler = (math.pi / 2, 0, 0)
skyo.scale = (SKYX1 - SKYX0 + 80, 110, 1)
# sun disc rising at the end
sund = sphere((0, 66, 0), 3.0, emis('sundisc', (1.0, 0.85, 0.6), 6.0), (24, 12))


def flake_layer(dist, scale, speed, name, strength=1.4):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    tc = nt.nodes.new('ShaderNodeTexCoord'); mpn = nt.nodes.new('ShaderNodeMapping')
    vor = nt.nodes.new('ShaderNodeTexVoronoi'); vor.inputs['Scale'].default_value = scale
    mr = nt.nodes.new('ShaderNodeMapRange'); mr.inputs['From Min'].default_value = 0.085; mr.inputs['From Max'].default_value = 0.02
    mixc = nt.nodes.new('ShaderNodeMixRGB'); mixc.inputs['Color1'].default_value = (0.95, 0.97, 1.0, 1); mixc.inputs['Color2'].default_value = (1.0, 0.6, 0.78, 1)
    emn = nt.nodes.new('ShaderNodeEmission'); emn.inputs['Strength'].default_value = strength
    trn = nt.nodes.new('ShaderNodeBsdfTransparent'); mix = nt.nodes.new('ShaderNodeMixShader'); o_ = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(tc.outputs['UV'], mpn.inputs['Vector']); nt.links.new(mpn.outputs[0], vor.inputs['Vector'])
    nt.links.new(vor.outputs['Distance'], mr.inputs['Value']); nt.links.new(mr.outputs[0], mix.inputs['Fac'])
    nt.links.new(mixc.outputs[0], emn.inputs['Color'])
    nt.links.new(trn.outputs[0], mix.inputs[1]); nt.links.new(emn.outputs[0], mix.inputs[2]); nt.links.new(mix.outputs[0], o_.inputs[0])
    for f, val in ((0, 0.0), (NF - 1, speed * NF / FPS)):
        mpn.inputs['Location'].default_value[1] = val; mpn.inputs['Location'].keyframe_insert('default_value', index=1, frame=f)
        mpn.inputs['Location'].default_value[0] = -0.3 * val; mpn.inputs['Location'].keyframe_insert('default_value', index=0, frame=f)
    key_val(mixc.inputs['Fac'], np.clip((SEAS - 0.25) * 2.0, 0, 1), 6)
    bpy.ops.mesh.primitive_plane_add(size=1.0); o = bpy.context.object; o.name = name; o.data.materials.append(m)
    o.visible_shadow = False; o.visible_glossy = False
    return o, dist
FLAKES = [flake_layer(4.0, 10.0, 0.03, 'flake1', 1.1), flake_layer(7.0, 17.0, 0.022, 'flake2', 0.8)]

# ---------------- camera ----------------
def smooth(Xa, k):
    Y = Xa.copy(); a = 1 - math.exp(-1 / FPS / k)
    for f in range(1, len(Xa)): Y[f] = Y[f - 1] + (Xa[f] - Y[f - 1]) * a
    for f in range(len(Xa) - 2, -1, -1): Y[f] = Y[f + 1] + (Y[f] - Y[f + 1]) * a
    return Y
FOC = np.zeros((NF, 3)); WID = np.zeros(NF)
YC = 2.0; YB = 5.0
for f in range(NF):
    t = T[f]
    if t < TB0 - 0.3 or TB1 - 1.2 < t:
        ps = [POS[n][f] for n in MAIN]
        z0 = min(p[2] for p in ps) - 1.0; z1 = max(p[2] for p in ps) + 0.8
        if t < 4.0: z1 += 2.2 * (1 - smoothstep((t - 2.5) / 1.5))
        xs = [p[0] for p in ps]
        FOC[f] = (np.mean(xs) + 0.9, YC, (z0 + z1) / 2); WID[f] = max(5.6, (z1 - z0 + 2.6) * 9 / 16, max(xs) - min(xs) + 3.5)
    else:
        act = [n for n in BIRDS if T[f] > EV['nestB']['t_drop'] - 1.2 or n == 'birdA']
        ps = [POS[n][f] for n in act] + [POS['bas'][f] + [0, 0, 0.5]]
        z0 = min(p[2] for p in ps) - 0.6; z1 = max(p[2] for p in ps) + 0.8
        xs = [POS[n][f][0] for n in act]
        FOC[f] = (np.mean(xs) + 0.6, YB, (z0 + z1) / 2); WID[f] = max(4.4, (z1 - z0 + 1.6) * 9 / 16, max(xs) - min(xs) + 3.0)
    if t > FINAL - 0.3:
        u = smoothstep((t - FINAL - 0.3) / 2.6)
        FOC[f] = FOC[f] * (1 - u) + np.array([crown_c[0] - 0.6, YC + 1.0, (crown_c[2] + fz_) / 2 - 0.8]) * u
        WID[f] = WID[f] * (1 - u) + 9.5 * u
WID = np.clip(WID, 4.4, 10.0)
for k, kk in ((0, 0.5), (1, 0.9), (2, 0.7)): FOC[:, k] = smooth(FOC[:, k], kk)
WID = smooth(WID, 0.8)
LENS = 32.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 600
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3)); EL = math.radians(27)
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24
    CP[f] = (FOC[f, 0], FOC[f, 1] - D * math.cos(EL), FOC[f, 2] + D * math.sin(EL))
    TP[f] = (FOC[f, 0], FOC[f, 1], FOC[f, 2])
key_loc(cam, CP); key_loc(tgt, TP)
for o, dist in FLAKES:
    o.parent = cam; o.location = (0, 0, -dist); o.rotation_euler = (0, 0, 0)
    hh = dist * 24 / LENS; o.scale = (hh * 9 / 16 * 1.1, hh * 1.1, 1)
# the sun disc rises behind the giant tree at the end
SP_ = np.array([[crown_c[0] + 4, 66, crown_c[2] - 14 + 16 * smoothstep((t - FINAL + 1.0) / 3.5)] for t in T])
key_loc(sund, SP_)

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
