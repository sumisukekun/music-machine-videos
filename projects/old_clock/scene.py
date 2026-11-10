"""Grandfather's Clock (H. C. Work, 1876) - '100年時計の一生' inside a tall cut-away grandfather clock.
Every melody note = a marble landing on a brass chime bar (one bar per note, flashes on its hit); the bass = the clock's
chime rods, struck by hammers on beats 1 and 3 (the pendulum's extremes). layout.py plans all motion (one gravity,
bounces that never gain speed, cups / trapdoors / lever rails / the clock weights as elevators); this file builds it.
The clock ages while the camera descends through it (1876 -> 1976): darker wood, tarnished brass, dust, cobwebs,
dimmer lamps; the pendulum's swing dies on the last note; the child marble is catapulted out to a small new clock.
python3 scene.py -- test t1 t2 ... | range f0 f1"""
import os, sys, json, math, time
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json')); NOTES = json.load(open(f'{HERE}/notes.json'))
LAY = json.load(open(f'{HERE}/layout.json'))
FPS = 30; DUR = E['dur']; NF = int(DUR * FPS); T = np.arange(NF) / FPS
G = LAY['G']; R = LAY['R']; RK = LAY['RK']; EV = LAY['ev']; Q = E['q']; TB1 = E['tb1']; FINAL = E['final']
TN = np.array([l[0] for l in LEAD])
rng = np.random.default_rng(3)

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
sc.render.resolution_x, sc.render.resolution_y = 1080, 1920
sc.render.resolution_percentage = int(os.environ.get('PCT', 50)); sc.render.fps = FPS
sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.5
try: sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Punchy'
except Exception: pass
sc.frame_start, sc.frame_end = 0, NF - 1
FONT = bpy.data.fonts.load(f'{HERE}/jp_sub.otf')
FONT_S = bpy.data.fonts.load(f'{HERE}/jp_serif.otf')


def link(o): sc.collection.objects.link(o); return o


def mat(name, col, rough=0.5, **kw):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']; b.inputs['Base Color'].default_value = (*col, 1); b.inputs['Roughness'].default_value = rough
    for k, v in kw.items(): b.inputs[k].default_value = v
    return m


def emis(name, col, s):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    e = nt.nodes.new('ShaderNodeEmission'); e.inputs['Color'].default_value = (*col, 1); e.inputs['Strength'].default_value = s
    o = nt.nodes.new('ShaderNodeOutputMaterial'); nt.links.new(e.outputs[0], o.inputs[0]); return m


def glowmat(name, gain=8.0, rough=0.3, metal=0.0, coat=0.3):
    """object colour = base colour; object alpha drives emission (hit flash)"""
    m = mat(name, (1, 1, 1), rough, **{'Coat Weight': coat, 'Metallic': metal}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    oi = nt.nodes.new('ShaderNodeObjectInfo')
    nt.links.new(oi.outputs['Color'], b.inputs['Base Color']); nt.links.new(oi.outputs['Color'], b.inputs['Emission Color'])
    mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = gain
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


def key_socket(sock, vals, eps=1e-4):
    n = len(vals)
    for f in range(n):
        v = vals[f]
        if 0 < f < n - 1 and abs(v - vals[f - 1]) < eps and abs(v - vals[f + 1]) < eps: continue
        sock.default_value = float(v); sock.keyframe_insert('default_value', frame=f)


def key_color(sock, cols, eps=1e-3):
    n = len(cols)
    for f in range(n):
        c = cols[f]
        if 0 < f < n - 1 and np.abs(c - cols[f - 1]).max() < eps and np.abs(c - cols[f + 1]).max() < eps: continue
        sock.default_value = (*map(float, c), 1); sock.keyframe_insert('default_value', frame=f)


_MC = {}
def prim(kind, m, *a):
    key = (kind, m.name if m else None) + tuple(a)
    if key in _MC: return _MC[key]
    bm = bmesh.new()
    if kind == 'cube': bmesh.ops.create_cube(bm, size=1.0)
    elif kind == 'cyl': bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=1.0, radius2=1.0, depth=1.0)
    elif kind == 'cone': bmesh.ops.create_cone(bm, cap_ends=a[4] if len(a) > 4 else True, segments=a[0], radius1=a[1], radius2=a[2], depth=a[3])
    elif kind == 'sph': bmesh.ops.create_uvsphere(bm, u_segments=a[0], v_segments=a[1], radius=1.0)
    elif kind == 'plane': bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    elif kind == 'torus':
        bm.free(); bpy.ops.mesh.primitive_torus_add(major_radius=a[0], minor_radius=a[1], major_segments=a[2], minor_segments=6)
        o = bpy.context.object; me = o.data; bpy.data.objects.remove(o)
        if m: me.materials.append(m)
        _MC[key] = me; return me
    me = bpy.data.meshes.new(kind); bm.to_mesh(me); bm.free()
    if kind in ('sph', 'cone', 'cyl'):
        for p in me.polygons: p.use_smooth = kind != 'cyl' or len(p.vertices) == 4
    if m: me.materials.append(m)
    _MC[key] = me; return me


def obj(me, name='o'): return link(bpy.data.objects.new(name, me))


def box(c, size, m, rot=None, name='box'):
    o = obj(prim('cube', m), name); o.location = c; o.scale = size
    if rot is not None: o.rotation_euler = rot
    return o


def rod(p0, p1, r, m, verts=10):
    p0, p1 = Vector(p0), Vector(p1); d = p1 - p0
    o = obj(prim('cyl', m, verts), 'rod'); q = Vector((0, 0, 1)).rotation_difference(d.normalized() if d.length > 1e-9 else Vector((0, 0, 1)))
    o.matrix_world = Matrix.Translation((p0 + p1) / 2) @ q.to_matrix().to_4x4() @ Matrix.Diagonal((r, r, max(1e-4, d.length), 1))
    return o


def sphere(c, r, m, seg=(16, 8), name='sph'):
    o = obj(prim('sph', m, *seg), name); o.location = c; o.scale = (r, r, r); return o


def text(s, loc, size, m, rot=(math.pi / 2, 0, 0), align='CENTER', ext=0.0, font=None):
    cu = bpy.data.curves.new('t', 'FONT'); cu.body = s; cu.font = font or FONT; cu.size = size; cu.extrude = ext
    cu.align_x = align; cu.align_y = 'CENTER'
    o = link(bpy.data.objects.new('txt', cu)); o.location = loc; o.rotation_euler = rot; o.data.materials.append(m); return o


def pulse(times, decay=5.0, length=16, amp=1.0):
    a = np.zeros(NF)
    for t in times:
        f0 = int(round(t * FPS)); m_ = np.arange(max(0, f0), min(NF, f0 + length))
        a[m_] = np.maximum(a[m_], amp * np.exp(-(m_ - f0) / decay))
    return a


def smooth(Xa, k):
    Y = Xa.copy(); a = 1 - math.exp(-1 / FPS / k)
    for f in range(1, len(Xa)): Y[f] = Y[f - 1] + (Xa[f] - Y[f - 1]) * a
    for f in range(len(Xa) - 2, -1, -1): Y[f] = Y[f + 1] + (Y[f] - Y[f + 1]) * a
    return Y


# ---------------- paths ----------------
def seg_at(g, t):
    if g['k'] == 'pend':
        th = g['sgn'] * g['amp'] * math.sin(g['w'] * (t - g['t0'])); pv = np.array(g['piv'])
        return pv + g['L'] * np.array([math.sin(th), 0, -math.cos(th)])
    p = np.array(g['p'])
    if g['k'] == 'rest': return p
    u = t - g['t0']
    if g['k'] == 'fly': return p + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
    return p + np.array(g['v']) * u + 0.5 * np.array(g['a']) * u * u


class Path:
    def __init__(s, b): s.seg = b['seg']; s.t0 = np.array([g['t0'] for g in s.seg]); s.r = b['r']; s.hits = b['hits']
    def at(s, t):
        if t < s.t0[0]: return seg_at(s.seg[0], s.t0[0])
        j = max(0, min(int(np.searchsorted(s.t0, t, side='right')) - 1, len(s.seg) - 1)); g = s.seg[j]
        return seg_at(g, min(t, g['t1']))
PATHS = {n: Path(b) for n, b in LAY['balls'].items()}
POS = {n: np.array([p.at(t) for t in T]) for n, p in PATHS.items()}

# ---------------- time / ageing ----------------
T_STOP = FINAL                                  # the clock stops on the last note
def age(t): return float(np.clip((t - 6.0) / (T_STOP - 6.0), 0, 1))
AGE = np.array([age(t) for t in T])

# ---------------- materials ----------------
M_WOOD = mat('wood', (0.32, 0.1, 0.04), 0.35, **{'Coat Weight': 0.6})          # mahogany, polished; ages below
bw = M_WOOD.node_tree.nodes['Principled BSDF']
key_color(bw.inputs['Base Color'], np.array([(0.32 - 0.2 * a, 0.1 - 0.06 * a, 0.04 - 0.025 * a) for a in AGE]))
key_socket(bw.inputs['Coat Weight'], 0.6 * (1 - AGE) + 0.02)
key_socket(bw.inputs['Roughness'], 0.3 + 0.5 * AGE)
M_WOOD2 = mat('wood2', (0.45, 0.2, 0.08), 0.5)
bw2 = M_WOOD2.node_tree.nodes['Principled BSDF']
key_color(bw2.inputs['Base Color'], np.array([(0.45 - 0.25 * a, 0.2 - 0.1 * a, 0.08 - 0.04 * a) for a in AGE]))
M_BACK = mat('back', (0.1, 0.045, 0.025), 0.6)
bk_ = M_BACK.node_tree.nodes['Principled BSDF']
key_color(bk_.inputs['Base Color'], np.array([(0.1 - 0.05 * a, 0.045 - 0.02 * a, 0.025 - 0.01 * a) for a in AGE]))
M_BRASS = mat('brass', (1.0, 0.72, 0.3), 0.22, **{'Metallic': 1.0})
bb = M_BRASS.node_tree.nodes['Principled BSDF']
key_color(bb.inputs['Base Color'], np.array([(1.0 - 0.45 * a, 0.72 - 0.35 * a, 0.3 - 0.12 * a) for a in AGE]))
key_socket(bb.inputs['Roughness'], 0.2 + 0.45 * AGE)
M_STEEL = mat('steel', (0.6, 0.6, 0.62), 0.3, **{'Metallic': 0.9})
M_DARK = mat('dark', (0.03, 0.03, 0.035), 0.5)
M_IVORY = mat('ivory', (0.92, 0.88, 0.76), 0.4)
M_KEY = glowmat('key', gain=5.0, rough=0.25, metal=0.6, coat=0.4)
M_ROD = glowmat('rod', gain=3.0, rough=0.25, metal=0.8, coat=0.2)
M_OBJ = objcol('objcol', 0.45)
M_FX = fxmat('fx', 6.0)
M_TXT_GOLD = emis('txtgold', (1.0, 0.78, 0.35), 1.6)
M_TXT_W = emis('txtw', (1, 0.97, 0.9), 1.4)
M_TXT_Y = emis('txty', (1, 0.82, 0.3), 1.8)
M_TXT_D = mat('txtd', (0.06, 0.04, 0.03), 0.5)
M_BALL = {'gold': mat('bgold', (1.0, 0.75, 0.25), 0.12, **{'Metallic': 0.6, 'Coat Weight': 1.0, 'Emission Color': (1.0, 0.7, 0.25, 1), 'Emission Strength': 0.8}),
          'rose': mat('brose', (1.0, 0.3, 0.45), 0.08, **{'Coat Weight': 1.0, 'Emission Color': (1.0, 0.3, 0.45, 1), 'Emission Strength': 0.9}),
          'kid': mat('bkid', (0.2, 0.6, 1.0), 0.06, **{'Coat Weight': 1.0, 'Emission Color': (0.25, 0.6, 1.0, 1), 'Emission Strength': 1.1})}

# ---------------- marbles ----------------
BALL = {}
for n, P in POS.items():
    rr = PATHS[n].r
    o = obj(prim('sph', M_BALL[n], 32, 16), n); o.scale = (rr, rr, rr)
    key_loc(o, P)
    # spin with travel (rolling look): about y from x-motion, about x from y-motion
    sy = np.zeros(NF); sx = np.zeros(NF)
    for f in range(1, NF):
        d = P[f] - P[f - 1]
        sy[f] = sy[f - 1] + d[0] / rr; sx[f] = sx[f - 1] - d[1] / rr
    o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, sy); key_series(o, 'rotation_euler', 0, sx)
    BALL[n] = o
# swirl stripe so the spin reads (a thin band)
for n, o in BALL.items():
    rr = PATHS[n].r
    bd = obj(prim('torus', M_IVORY if n != 'gold' else M_DARK, 1.0, 0.06, 32), 'band'); bd.parent = o; bd.scale = (1.0, 1.0, 1.0)
    bd.rotation_euler = (0.4, 0.2, 0)

# ---------------- chime bars: one per hit (flash + dip on the hit) ----------------
def pitch_col(p):
    h = ((p - 59) % 12) / 12.0
    return (1.0, 0.62 + 0.22 * h, 0.22 + 0.25 * h)
BARS = []
HITS = {(n, h['i']): h for n, b in LAY['balls'].items() for h in b['hits']}
for K in LAY['keys']:
    M = np.array(K['M']); c = np.array(K['c'])
    o = obj(prim('cube', M_KEY), 'bar')
    o.matrix_world = Matrix.Translation(Vector(c)) @ Matrix(M.tolist()).to_4x4() @ Matrix.Diagonal((2 * K['hw'], 2 * K['hd'], 2 * K['th'], 1))
    col = pitch_col(K['pitch']); o.color = (*col, 0)
    BARS.append(dict(o=o, c=c, n=M[:, 2], t=K['t'], ball=K['ball'], i=K['i'], hd=K['hd'], kind=HITS[(K['ball'], K['i'])]['kind']))
    # two small posts to the back: the bar hangs on pins (stop short of the pendulum plane)
    back = c + M[:, 1] * K['hd']
    for sx_ in (-1, 1):
        pp = c + M[:, 0] * sx_ * K['hw'] * 0.55 + M[:, 1] * K['hd']
        rod(pp, (pp[0], 0.62, pp[2]), 0.011, M_STEEL, 6)
for B_ in BARS:
    o = B_['o']; fh = int(round(B_['t'] * FPS)); base = o.location.copy(); nv = Vector(B_['n'])
    col = tuple(o.color[:3])
    o.color = (*col, 0); o.keyframe_insert('color', index=3, frame=max(0, fh - 1))
    for df, a in ((0, 1.0), (3, 0.55), (14, 0.12), (40, 0.0)):
        o.color = (*col, a); o.keyframe_insert('color', index=3, frame=fh + df)
    if B_['kind'] in ('lever', 'seesaw'): continue
    for df, d in ((-1, 0), (1, -0.02), (4, 0.006), (8, 0)):
        o.location = base + nv * d; o.keyframe_insert('location', frame=max(0, fh + df))
# hit rings + sparks (pool)
NRING = 10
ring_loc = np.zeros((NRING, NF, 3)); ring_scl = np.full((NRING, NF), 1e-3); ring_a = np.zeros((NRING, NF))
for k, B_ in enumerate(sorted(BARS, key=lambda b: b['t'])):
    r = k % NRING; f0 = int(round(B_['t'] * FPS))
    for df in range(14):
        f = f0 + df
        if f >= NF: break
        u = df / 13; ring_loc[r, f] = B_['c'] + np.array([0, -B_['hd'] - 0.02, 0.02]); ring_scl[r, f] = 0.25 + 1.1 * (1 - (1 - u) ** 2)
        ring_a[r, f] = (1 - u) ** 1.6
for r in range(NRING):
    o = obj(prim('torus', M_FX, 0.2, 0.012, 32), 'ring'); o.rotation_euler = (math.pi / 2, 0, 0); o.color = (1.0, 0.8, 0.45, 0)
    for f in range(1, NF):
        if ring_a[r, f] == 0: ring_loc[r, f] = ring_loc[r, f - 1]
    key_loc(o, ring_loc[r])
    for k in range(3): key_series(o, 'scale', k, ring_scl[r])
    key_series(o, 'color', 3, ring_a[r], 1e-3)

# ---------------- the clock case ----------------
Z_BASE = EV['base']['z_base']; Z_ROOM = EV['base']['z_room']
Z_TOP_IN = 31.75                                   # interior top (under the hood / plate)
XW = 1.9; YB = 1.05; YF = -0.82
Z_HOOD0, Z_HOOD1 = 31.2, 35.4
# side walls + back wall (interior), front frame pillars
for sx_ in (-1, 1):
    box((sx_ * (XW + 0.06), 0.1, (Z_ROOM + Z_HOOD1) / 2), (0.14, 1.95, Z_HOOD1 - Z_ROOM), M_WOOD, name='side')
    box((sx_ * (XW + 0.08), YF - 0.06, (Z_ROOM + Z_HOOD1) / 2), (0.26, 0.14, Z_HOOD1 - Z_ROOM), M_WOOD, name='pillar')
# back wall: wood panels with inlay strips (texture-free: rows of slightly different boards)
for k, z0 in enumerate(np.arange(Z_BASE - 0.2, Z_HOOD1, 1.6)):
    b = box((0, YB + 0.03, z0 + 0.8), (2 * XW, 0.06, 1.58), M_BACK, name='board')
    box((0, YB - 0.005, z0 + 0.01), (2 * XW, 0.02, 0.025), M_BRASS, name='inlay')
# base plinth (closed front below the interior floor), floor of the interior
box((0, 0.1, (Z_ROOM + Z_BASE) / 2), (2 * XW + 0.4, 1.95, Z_BASE - Z_ROOM), M_WOOD, name='plinth')
box((0, 0.1, Z_BASE - 0.02), (2 * XW, 1.9, 0.04), M_WOOD2, name='floorin')
box((0, YF - 0.1, Z_ROOM + 0.12), (2 * XW + 0.6, 0.3, 0.24), M_WOOD, name='skirting')
# the hatch in the left wall of the base (open: a dark hole with a brass frame) at the child's crossing height
hz_ = EV['hatch_out']['z']; hx_ = EV['hatch_out']['x']
sgnw = 1 if hx_ > 0 else -1
hole = box((sgnw * (XW + 0.06), 0.0, hz_), (0.2, 0.42, 0.42), M_DARK, name='hatchhole')
for dz_ in (-0.23, 0.23):
    box((sgnw * (XW + 0.14), 0.0, hz_ + dz_), (0.04, 0.5, 0.04), M_BRASS)
for dy_ in (-0.23, 0.23):
    box((sgnw * (XW + 0.14), dy_, hz_), (0.04, 0.04, 0.5), M_BRASS)
# hood: dial, plates, pediment
box((0, 0.1, (Z_HOOD0 + Z_HOOD1) / 2 + 0.1), (2 * XW + 0.5, 2.0, Z_HOOD1 - Z_HOOD0 + 0.2), M_WOOD, name='hoodback') if False else None
box((0, YF - 0.05, Z_HOOD1 - 0.15), (2 * XW + 0.6, 0.3, 0.3), M_WOOD, name='cornice')
box((0, YF + 0.05, (Z_TOP_IN + Z_HOOD1) / 2), (2 * XW, 0.12, Z_HOOD1 - Z_TOP_IN), M_WOOD2, name='hoodface')
DIAL_C = np.array([0.0, YF - 0.03, 33.15]); DIAL_R = 1.12
dl = obj(prim('cyl', M_IVORY, 64), 'dial'); dl.location = DIAL_C; dl.rotation_euler = (math.pi / 2, 0, 0); dl.scale = (DIAL_R, DIAL_R, 0.04)
ring = obj(prim('torus', M_BRASS, 1.0, 0.045, 64), 'dialring'); ring.location = DIAL_C + np.array([0, -0.03, 0]); ring.rotation_euler = (math.pi / 2, 0, 0)
ring.scale = (DIAL_R + 0.02, DIAL_R + 0.02, DIAL_R + 0.02)
ROMAN = ['XII', 'I', 'II', 'III', 'IIII', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI']
for k, s_ in enumerate(ROMAN):
    a = math.radians(90 - 30 * k)
    p_ = DIAL_C + np.array([0.86 * DIAL_R * math.cos(a), -0.05, 0.86 * DIAL_R * math.sin(a)])
    tx = text(s_, p_, 0.2, M_TXT_D, font=FONT_S); tx.rotation_euler = (math.pi / 2, 0, a - math.pi / 2)
# hands: hour + minute (minute hand moves a little with every tick), seconds hand ticks
def hand(length, width, m, z_off):
    piv = link(bpy.data.objects.new('hand', None)); piv.location = DIAL_C + np.array([0, -0.06 - z_off, 0])
    h = box((0, 0, length / 2 - 0.08), (width, 0.012, length), m); h.parent = piv
    return piv
HH = hand(0.6, 0.07, M_DARK, 0.0); MH = hand(0.9, 0.045, M_DARK, 0.02)
TICKS = np.array([TB1 + k * 2 * Q for k in range(-2, int((FINAL - TB1) / (2 * Q)) + 1)])
def ticks_before(t): return np.searchsorted(TICKS, t + 1e-6)
mang = np.array([-(math.radians(60) + ticks_before(t) * math.radians(6)) for t in T])
hang = np.array([-(math.radians(305) + ticks_before(t) * math.radians(0.5)) for t in T])
HH.rotation_mode = 'XYZ'; MH.rotation_mode = 'XYZ'
key_series(HH, 'rotation_euler', 1, -hang); key_series(MH, 'rotation_euler', 1, -mang)
sphere(DIAL_C + np.array([0, -0.1, 0]), 0.05, M_BRASS)
# moon-phase arch above the dial
arc = obj(prim('cyl', M_OBJ, 48), 'arch'); arc.location = DIAL_C + np.array([0, 0.02, DIAL_R + 0.25]); arc.rotation_euler = (math.pi / 2, 0, 0)
arc.scale = (0.85, 0.5, 0.03); arc.color = (0.08, 0.12, 0.35, 1)
for k in range(9):
    sphere(arc.location + Vector((rng.uniform(-0.6, 0.6), -0.05, rng.uniform(-0.05, 0.4))), 0.02, M_TXT_Y)
# title plate (brass) under the dial + composer plate
PL_Z = 31.48
box((0, YF - 0.07, PL_Z), (3.0, 0.04, 0.5), M_DARK, name='plate')
for sz_ in (-0.26, 0.26): box((0, YF - 0.09, PL_Z + sz_), (3.0, 0.02, 0.03), M_BRASS)
text('大きな古時計', (0, YF - 0.12, PL_Z), 0.4, M_TXT_GOLD, font=FONT_S, ext=0.004)
box((0, YF - 0.07, Z_HOOD1 + 0.3), (3.4, 0.05, 0.55), M_WOOD2, name='pediment')
text('ビー玉からくり', (0, YF - 0.11, Z_HOOD1 + 0.3), 0.36, M_TXT_GOLD, font=FONT_S, ext=0.005)
# finials
for sx_ in (-1, 0, 1):
    sphere((sx_ * 1.8, YF, Z_HOOD1 + 0.68), 0.12, M_BRASS)

# ---------------- wall signs: composer, era years, labels (Japanese keywords, readable size) ----------------
def wall_sign(s, x, z, size, m=M_TXT_GOLD, font=None, plate=True, y=YB - 0.03):
    if plate:
        w = 0.62 * size * max(2, len(s)) + 0.2
        box((x, y + 0.005, z), (w, 0.02, size * 1.5), M_BRASS if plate == 'brass' else M_DARK)
    return text(s, (x, y - 0.02, z), size, m, font=font or FONT_S, ext=0.004)
zz = {n: [h['p'][2] for h in b['hits']] for n, b in LAY['balls'].items()}
wall_sign('作曲 ヘンリー・クレイ・ワーク', 0.25, 30.6, 0.22, M_TXT_W, plate=True)
SEC_Z = {'1876': 27.4, '1900': 20.6, '1925': 14.2, '1950': 7.6, '1976': -2.5}
for yr, z in SEC_Z.items():
    wall_sign(yr, 0.0, z, 0.55, M_TXT_GOLD, plate=False)
wall_sign('オルゴール仕掛け', 0.0, 28.6, 0.26, M_TXT_W, plate=True)

# ---------------- pendulum: long rod from the movement to the bob in the base; ticks at the extremes ----------------
PIV = np.array([0.0, 0.8, 31.9]); BOB_Z = Z_BASE + 0.85
LPEND = PIV[2] - BOB_Z
A0 = 0.4 / LPEND
def pend_amp(t):
    if t < 52.625: return A0
    if t < T_STOP: return A0 * (1 - 0.6 * (t - 52.625) / (T_STOP - 52.625))
    return 0.4 * A0 * math.exp(-(t - T_STOP) / 0.45)
TH = np.array([pend_amp(t) * math.cos(math.pi * (t - TB1) / (2 * Q)) for t in T])
pv = link(bpy.data.objects.new('pend', None)); pv.location = PIV; pv.rotation_mode = 'XYZ'
key_series(pv, 'rotation_euler', 1, TH, 1e-6)
pr = rod((0, 0, 0), (0, 0, -LPEND + 0.3), 0.025, M_BRASS, 8); pr.parent = pv
bob = obj(prim('cyl', M_BRASS, 48), 'bob'); bob.parent = pv; bob.location = (0, 0, -LPEND); bob.rotation_euler = (math.pi / 2, 0, 0); bob.scale = (0.42, 0.42, 0.07)
bc = obj(prim('cyl', M_IVORY, 32), 'bobface'); bc.parent = pv; bc.location = (0, -0.08, -LPEND); bc.rotation_euler = (math.pi / 2, 0, 0); bc.scale = (0.2, 0.2, 0.01)
# escapement at the top: escape wheel steps one tooth per tick, anchor rocks with the pendulum
EW_C = np.array([0.55, 0.88, 31.25])
ew = obj(prim('cyl', M_BRASS, 30), 'escwheel'); ew.location = EW_C; ew.rotation_euler = (math.pi / 2, 0, 0); ew.scale = (0.32, 0.32, 0.03)
ewr = link(bpy.data.objects.new('ewrot', None)); ewr.location = EW_C; ewr.rotation_mode = 'XYZ'
for k in range(15):
    a = 2 * math.pi * k / 15
    t_ = box((0.36 * math.cos(a), 0, 0.36 * math.sin(a)), (0.09, 0.03, 0.05), M_BRASS, rot=(0, -a, 0)); t_.parent = ewr
ew.parent = ewr; ew.location = (0, 0, 0)
steps = np.array([ticks_before(t) for t in T]) * (2 * math.pi / 30)
key_series(ewr, 'rotation_euler', 1, steps)
anc = link(bpy.data.objects.new('anchor', None)); anc.location = (0.0, 0.84, 31.6); anc.rotation_mode = 'XYZ'
for sx_ in (-1, 1):
    arm = rod((0, 0, 0), (sx_ * 0.42 + 0.55, 0, -0.25), 0.03, M_STEEL, 8); arm.parent = anc
key_series(anc, 'rotation_euler', 1, TH * 6, 1e-6)
wall_sign('振り子', 0.0, 29.75, 0.3, M_TXT_W, plate=True, y=0.7)
# big gears of the movement near the top (turn with the escapement)
for (gx, gz, gr, ratio) in ((-1.25, 31.0, 0.45, -0.5), (1.3, 30.4, 0.35, 0.8)):
    gp = link(bpy.data.objects.new('gear', None)); gp.location = (gx, 0.9, gz); gp.rotation_mode = 'XYZ'
    gw = obj(prim('cyl', M_BRASS, 40), 'gearw'); gw.parent = gp; gw.rotation_euler = (math.pi / 2, 0, 0); gw.scale = (gr, gr, 0.03)
    for k in range(18):
        a = 2 * math.pi * k / 18
        tt_ = box(((gr + 0.04) * math.cos(a), 0, (gr + 0.04) * math.sin(a)), (0.07, 0.03, 0.06), M_BRASS, rot=(0, -a, 0)); tt_.parent = gp
    key_series(gp, 'rotation_euler', 1, steps * ratio)
wall_sign('歯車', -1.25, 30.25, 0.22, M_TXT_W, plate=False, y=0.85)

# ---------------- bass chime rods along the side walls + hammers (beats 1 and 3) ----------------
BASS = [(n[0], n[3]) for n in NOTES if n[2] == 'bass']
RODX = {38: -1.7, 40: -1.52, 43: 1.52, 45: 1.7}
RODY = 0.95
rod_mats = {}
for p, x in RODX.items():
    m = glowmat(f'rodm{p}', gain=2.5, rough=0.25, metal=0.85, coat=0.2)
    r_ = rod((x, RODY, Z_BASE), (x, RODY, Z_TOP_IN), 0.045 if p < 42 else 0.04, m, 16)
    r_.color = (1.0, 0.75, 0.4, 0)
    times = [t for t, pp in BASS if pp == p]
    key_series(r_, 'color', 3, pulse(times, decay=5, length=24, amp=0.9), 1e-3)
HAM_Z = np.arange(Z_BASE + 1.0, Z_TOP_IN - 0.5, 4.2)
lift = {p: np.zeros(NF) for p in RODX}
for t, p in BASS:
    f0 = int(round(t * FPS))
    for df in range(-8, 6):
        f = f0 + df
        if 0 <= f < NF:
            u = df / 8.0
            lift[p][f] = max(lift[p][f], 0.7 * (1 - abs(u)) ** 0.8 if df < 0 else 0.25 * (1 - df / 6))
for p, x in RODX.items():
    master = link(bpy.data.objects.new(f'hm{p}', None)); master.rotation_mode = 'XYZ'
    sgn = -1 if x < 0 else 1
    key_series(master, 'rotation_euler', 2, sgn * lift[p], 1e-4)
    for z in HAM_Z:
        hp = link(bpy.data.objects.new('hpiv', None)); hp.location = (x, RODY - 0.42, z); hp.rotation_mode = 'XYZ'
        a_ = rod((0, 0, 0), (0, 0.33, 0), 0.012, M_STEEL, 6); a_.parent = hp
        hd = obj(prim('cyl', M_WOOD2, 16), 'hamhead'); hd.parent = hp; hd.location = (0, 0.36, 0); hd.scale = (0.05, 0.05, 0.1)
        cr_ = hp.constraints.new('COPY_ROTATION'); cr_.target = master
        box((x, RODY - 0.47, z), (0.05, 0.08, 0.12), M_BRASS)

# ---------------- weights (the clock's elevators): brass cylinders with a dish, hanging on two chains ----------------
W = LAY['weights']; W_OFF = LAY['w_off']
CUPS = LAY['cups']
def w_seat(name, t):
    w = W[name]; return np.array([w['x'], 0.0, max(w['z0'] - w['v'] * t, w['zstop'])])
TIPS = {}            # (name) -> list of (t_tip, dir)
for c in CUPS:
    if c['kind'] == 'weight' and c['ball'] in ('gold', 'rose'):
        TIPS.setdefault(c['ref'], []).append((c['t_out'], -1 if W[c['ref']]['x'] > 0 else 1))
    if c['kind'] == 'weight' and c['ball'] == 'kid':
        TIPS.setdefault(c['ref'], []).append((c['t_out'], -1 if W[c['ref']]['x'] > 0 else 1))
for name, w in W.items():
    root = link(bpy.data.objects.new('w_' + name, None))
    P = np.array([w_seat(name, t) for t in T]); key_loc(root, P)
    body = obj(prim('cyl', M_BRASS, 32), 'wbody'); body.parent = root; body.location = (0, 0, -W_OFF - 0.25); body.scale = (0.17, 0.17, 0.62)
    wb_ = obj(prim('cyl', M_DARK, 32), 'wband'); wb_.parent = root; wb_.location = (0, 0, -W_OFF - 0.1); wb_.scale = (0.175, 0.175, 0.05)
    tilt = link(bpy.data.objects.new('wtilt', None)); tilt.parent = root; tilt.rotation_mode = 'XYZ'
    dish = obj(prim('cone', M_BRASS, 32, 0.12, 0.26, 0.16, False), 'dish'); dish.parent = tilt; dish.location = (0, 0, -0.08)
    dish.scale = (1.0, 1.75, 1.0)
    sol = dish.modifiers.new('s', 'SOLIDIFY'); sol.thickness = 0.015
    ang = np.zeros(NF)
    for (tt, d) in TIPS.get(name, []):
        for f in range(NF):
            u = (T[f] - tt) / 0.12
            if -0.0 <= u: ang[f] = max(ang[f], math.radians(30) * min(1, u) * (1 if T[f] < tt + 0.7 else max(0, 1 - (T[f] - tt - 0.7) / 0.3)))
    sgn = -1 if w['x'] > 0 else 1
    key_series(tilt, 'rotation_euler', 1, sgn * ang, 1e-4)
    # chains: two thin rods from the dish rim (y = +-0.42) up to the top (they move with the weight: long fixed-length rods)
    for sy_ in (-0.44, 0.44):
        ch = rod((0, sy_, 0), (0, sy_, 60.0), 0.012, M_STEEL, 6); ch.parent = root
    box((0, 0, 0.0), (0.06, 0.9, 0.03), M_STEEL).parent = root
    lab = text('おもり', (0, -0.18, -W_OFF - 0.25), 0.11, M_TXT_D, font=FONT_S); lab.parent = root
# pulleys at the top
for name, w in W.items():
    pu = obj(prim('cyl', M_BRASS, 24), 'pulley'); pu.location = (w['x'], 0, Z_TOP_IN - 0.25); pu.rotation_euler = (math.pi / 2, 0, 0); pu.scale = (0.46, 0.46, 0.05)

# ---------------- cups / hoppers with trapdoors, the hatch, the nest, the cradle, lever rails, the see-saw ----------------
def funnel(c, r=0.2, m=M_BRASS):
    f = obj(prim('cone', m, 24, 0.09, r, 0.24, False), 'funnel'); f.location = (c[0], c[1], c[2] - 0.02)
    sol = f.modifiers.new('s', 'SOLIDIFY'); sol.thickness = 0.012
    return f
def flap(c, t_open, t_close=None, r=0.1):
    hp = link(bpy.data.objects.new('flap', None)); hp.location = (c[0] - r, c[1], c[2] - 0.15); hp.rotation_mode = 'XYZ'
    fl = box((r, 0, 0), (2 * r, 2 * r, 0.02), M_BRASS); fl.parent = hp
    ang = np.zeros(NF)
    for f in range(NF):
        u = (T[f] - t_open) / 0.08
        if u > 0: ang[f] = min(1, u) * 1.4
        if t_close and T[f] > t_close: ang[f] = max(0, 1.4 * (1 - (T[f] - t_close) / 0.3))
    key_series(hp, 'rotation_euler', 1, ang, 1e-4)
    rod((c[0] + r + 0.05, c[1], c[2] - 0.15), (c[0] + r + 0.05, c[1], c[2] - 0.45), 0.012, M_STEEL, 6)
for c in CUPS:
    if c['kind'] == 'cup':
        p_ = np.array(c['p']) - np.array([0, 0, 0.05])
        funnel(p_); flap(p_, c['t_out'] if c['t_out'] < 79 else 999)
        rod((p_[0], p_[1] + 0.2, p_[2]), (p_[0], 0.62, p_[2]), 0.015, M_STEEL, 6)
    elif c['kind'] == 'tipcup':
        p_ = np.array(c['p']) - np.array([0, 0, 0.04])
        hp = link(bpy.data.objects.new('tipc', None)); hp.location = p_; hp.rotation_mode = 'XYZ'
        cu = obj(prim('cone', M_BRASS, 24, 0.07, 0.15, 0.14, False), 'tcup'); cu.parent = hp
        ang = np.zeros(NF)
        for f in range(NF):
            u = (T[f] - c['t_out']) / 0.12
            if u > 0: ang[f] = math.radians(30) * min(1, u) * max(0, 1 - max(0, T[f] - c['t_out'] - 0.8) / 0.4)
        key_series(hp, 'rotation_euler', 1, -c.get('tip', 1) * ang, 1e-4)
        rod((p_[0], p_[1] + 0.12, p_[2]), (p_[0], 0.62, p_[2]), 0.015, M_STEEL, 6)
# the hatch under the dial: a tipping cup; the rose's hopper: funnel + flap
ht = EV['hatch']; hp_ = np.array(ht['p']) - np.array([0, 0, 0.05])
hh = link(bpy.data.objects.new('hatch', None)); hh.location = hp_; hh.rotation_mode = 'XYZ'
hc = obj(prim('cone', M_BRASS, 24, 0.1, 0.21, 0.2, False), 'hcup'); hc.parent = hh
ang = np.array([math.radians(30) * min(1, max(0, (t - ht['t']) / 0.12)) for t in T]); key_series(hh, 'rotation_euler', 1, -ang, 1e-4)
rod(hp_ + np.array([0, 0.2, 0]), (hp_[0], 0.62, hp_[2]), 0.015, M_STEEL, 6)
rp = EV['rose_hopper']; rp_ = np.array(rp['p']) - np.array([0, 0, 0.05])
funnel(rp_, 0.22); flap(rp_, rp['t'])
rod((rp_[0], rp_[1], rp_[2] + 0.3), (rp_[0], rp_[1], rp_[2] + 1.6), 0.13, M_OBJ, 24).color = (0.9, 0.85, 0.8, 1)   # the hopper's glass-like tube
# nest + cradle
cr = EV['cradle']; nest = np.array(cr['nest']); kc = np.array(cr['p'])
funnel(nest - np.array([0, 0, 0.05]), 0.17); flap(nest - np.array([0, 0, 0.05]), E['t_tag'] * 0 + TN[41] - 0.5)
cradle = link(bpy.data.objects.new('cradle', None)); cradle.location = kc - np.array([0, 0, 0.1])
cb_ = obj(prim('cone', M_OBJ, 24, 0.06, 0.13, 0.08, False), 'cradlecup'); cb_.parent = cradle; cb_.color = (0.55, 0.75, 1.0, 1)
cz = np.zeros(NF)
for f in range(NF):
    u = (T[f] - cr['t']) / 0.06
    if 0 < u < 1: cz[f] = 0.18 * u
    elif u >= 1: cz[f] = 0.18 * max(0, 1 - (T[f] - cr['t'] - 0.06) / 0.4)
key_series(cradle, 'location', 2, kc[2] - 0.1 + cz, 1e-4)
u40 = EV['unison40']
for pp in (u40['g'], u40['r']):     # linkage from each lever bar to the cradle
    rod((pp[0], 0.0, pp[2] - 0.2), (kc[0], 0.0, kc[2] - 0.2), 0.014, M_STEEL, 6)
rod((kc[0], 0.0, kc[2] - 0.2), (kc[0], 0.62, kc[2] - 0.2), 0.02, M_STEEL, 6)
sparkle = sphere(kc + np.array([0, -0.15, 0.1]), 0.35, M_FX); sparkle.color = (0.6, 0.85, 1.0, 0)
key_series(sparkle, 'color', 3, pulse([cr['t']], decay=6, length=24, amp=0.8), 1e-3)
for rl in LAY['rails']:
    p0 = np.array(rl['p0']) - np.array([0, 0, R + 0.02]); p1 = np.array(rl['p1']) - np.array([0, 0, R + 0.02])
    for dy in (-0.07, 0.07):
        rod(p0 + np.array([0, dy, 0]), p1 + np.array([0, dy, 0]), 0.016, M_BRASS, 8)
# the see-saw in the base
ss = EV['seesaw']; piv = np.array(ss['pivot']); arm = ss['arm']; phi0 = ss['phi0']; side = ss['side']
gold_end = np.array(ss['gold_end']); kid_seat = np.array(ss['kid_seat'])
sw = link(bpy.data.objects.new('seesaw', None)); sw.location = piv; sw.rotation_mode = 'XYZ'
L_ = abs(kid_seat[0] - gold_end[0]) / math.cos(phi0) + 0.3
plank = box((0, 0, 0), (L_, 0.75, 0.06), M_WOOD2, name='plank'); plank.parent = sw
kcup = obj(prim('cone', M_OBJ, 24, 0.06, 0.14, 0.1, False), 'kcup'); kcup.parent = sw; kcup.location = (side * arm, kid_seat[1], 0.07); kcup.color = (0.55, 0.75, 1.0, 1)
pc_ = obj(prim('cone', M_BRASS, 4, 0.25, 0.02, 0.35), 'fulcrum'); pc_.location = piv - np.array([0, 0, 0.2])
swa = np.array([(-phi0 if T[f] < ss['t'] else -phi0 + 2 * phi0 * min(1, (T[f] - ss['t']) / ss['t_flip'])) for f in range(NF)])
key_series(sw, 'rotation_euler', 1, side * swa, 1e-5)
# the struck bar of the last note sits on the gold end of the plank (the lever bar = a bar object already: re-parent)
for B_ in BARS:
    if B_['kind'] == 'seesaw':
        o = B_['o']; mw = o.matrix_world.copy(); o.parent = sw; o.matrix_parent_inverse = sw.matrix_world.inverted() @ Matrix.Translation((0, 0, 0))
        o.matrix_world = mw
        o.animation_data_clear()
        col = tuple(o.color[:3]); fh = int(round(B_['t'] * FPS))
        for df, a in ((-1, 0), (0, 1.0), (3, 0.55), (14, 0.12), (40, 0.0)):
            o.color = (*col, a); o.keyframe_insert('color', index=3, frame=max(0, fh + df))

# ---------------- dust, cobwebs (ageing), framed family portraits ----------------
NDUST = 70
for k in range(NDUST):
    d_ = obj(prim('sph', M_FX, 6, 4), 'dust'); d_.scale = (0.012, 0.012, 0.012); d_.color = (1.0, 0.9, 0.7, 0)
    t_on = 30 + 30 * k / NDUST
    zc = rng.uniform(Z_BASE, 16)
    P = np.zeros((NF, 3)); A = np.zeros(NF)
    for f in range(NF):
        t = T[f]
        P[f] = (rng.uniform(-1.6, 1.6) if f == 0 else P[f - 1][0] + 0.004 * math.sin(t * 0.7 + k),
                -0.3 + 0.6 * math.sin(k), zc - 0.05 * t + 0.2 * math.sin(t * 0.3 + k))
        A[f] = 0.35 * min(1, max(0, (t - t_on) / 3))
    key_loc(d_, P); key_series(d_, 'color', 3, A, 1e-3)
def cobweb(c, s, t_on, flip=1):
    wb = bpy.data.meshes.new('web'); bm = bmesh.new(); v0 = bm.verts.new((0, 0, 0))
    rays = [bm.verts.new((flip * s * math.cos(a), 0, -s * math.sin(a))) for a in np.linspace(0, math.pi / 2, 6)]
    for r_ in rays: bm.edges.new((v0, r_))
    for k in range(1, 4):
        pts = [bm.verts.new((flip * s * k / 4 * math.cos(a), 0, -s * k / 4 * math.sin(a))) for a in np.linspace(0, math.pi / 2, 6)]
        for a_, b_ in zip(pts, pts[1:]): bm.edges.new((a_, b_))
    bm.to_mesh(wb); bm.free(); o = link(bpy.data.objects.new('web', wb)); o.location = c
    sk = o.modifiers.new('sk', 'SKIN')
    for v in wb.skin_vertices[0].data: v.radius = (0.004, 0.004)
    o.data.materials.append(M_FX); o.color = (0.9, 0.9, 0.95, 0)
    key_series(o, 'color', 3, np.clip((T - t_on) / 3, 0, 1) * 0.25, 1e-3)
for (x_, z_, s_, t_on, fl) in ((-1.8, 6.0, 0.6, 38, 1), (1.8, 2.0, 0.7, 44, -1), (-1.8, -2.0, 0.8, 50, 1), (1.8, -5.0, 0.6, 53, -1), (-1.8, 9.5, 0.5, 34, 1)):
    cobweb((x_, -0.5, z_), s_, t_on, fl)
def portrait(x, z, members, label=None):
    box((x, YB - 0.04, z), (0.62, 0.03, 0.5), M_BRASS)
    box((x, YB - 0.06, z), (0.5, 0.02, 0.38), M_IVORY)
    n = len(members)
    for k, (col, r_) in enumerate(members):
        sphere((x - 0.15 * (n - 1) / 2 + 0.15 * k, YB - 0.09, z - 0.08 + r_), r_, M_OBJ).color = (*col, 1)
GOLDC, ROSEC, KIDC = (1.0, 0.7, 0.2), (0.95, 0.35, 0.45), (0.2, 0.55, 1.0)
portrait(1.35, 26.3, [(GOLDC, 0.06)])
portrait(-1.35, 19.4, [(GOLDC, 0.06), (ROSEC, 0.06)])
portrait(1.35, 13.0, [(GOLDC, 0.06), (KIDC, 0.045), (ROSEC, 0.06)])
portrait(-1.35, 5.8, [(GOLDC, 0.06), (KIDC, 0.055), (ROSEC, 0.06)])
portrait(1.35, -1.2, [(GOLDC, 0.06), (ROSEC, 0.06)])

# ---------------- room, rug, window light, the little clock ----------------
Z_R = Z_ROOM
box((-6, 1.5, Z_R - 0.05), (26, 12, 0.1), M_WOOD2, name='floor')
for k in range(-12, 8):
    box((k * 0.9 + 0.45, 1.5, Z_R + 0.001), (0.012, 12, 0.004), M_DARK)
M_WALL = mat('wallpaper', (0.55, 0.42, 0.3), 0.8)
box((-6, 3.0, Z_R + 6), (26, 0.1, 12), M_WALL, name='wall')
for k in range(-14, 6):
    box((k * 0.6, 2.94, Z_R + 2.2), (0.04, 0.02, 4.4), M_WOOD2)
lt = EV['little']; LX = lt['pivot'][0]; LY = lt['pivot'][1]
rugx0 = EV['floor_hits'][-1]['p'][0] if EV.get('floor_hits') else LX + 1.5
rug = box(((rugx0 + LX - 0.6) / 2, 0.3, Z_R + 0.006), (abs(rugx0 - LX) + 1.4, 1.6, 0.012), M_OBJ, name='rug'); rug.color = (0.45, 0.08, 0.08, 1)
# little clock: body on four legs, pendulum hangs below the body between the legs, glass front shows a music box
LP = lt['L']; LPIV = np.array(lt['pivot'])
BODY_Z0 = LPIV[2] + 0.05; BODY_H = 0.85; BW = 0.9
lc = box((LX, LY + 0.05, BODY_Z0 + BODY_H / 2), (BW, 0.55, BODY_H), M_WOOD, name='littlebody')
box((LX, LY + 0.05, BODY_Z0 + BODY_H + 0.08), (BW + 0.12, 0.62, 0.12), M_WOOD, name='littletop')
for sx_ in (-1, 1):
    for sy_ in (-1, 1):
        rod((LX + sx_ * (BW / 2 - 0.05), LY + 0.05 + sy_ * 0.22, Z_R), (LX + sx_ * (BW / 2 - 0.05), LY + 0.05 + sy_ * 0.22, BODY_Z0), 0.03, M_WOOD2, 8)
ldial = obj(prim('cyl', M_IVORY, 48), 'ldial'); ldial.location = (LX, LY - 0.24, BODY_Z0 + 0.6); ldial.rotation_euler = (math.pi / 2, 0, 0); ldial.scale = (0.24, 0.24, 0.02)
obj(prim('torus', M_BRASS, 0.25, 0.015, 48), 'lring').location = (LX, LY - 0.26, BODY_Z0 + 0.6)
bpy.data.objects[-1].rotation_euler = (math.pi / 2, 0, 0) if False else (0, 0, 0)
# music box inside (seen through the opening below the dial): pinned drum + comb (5 tines)
TAG = E['tag']; TAGP = sorted(set(p for t, p, v, d in TAG))
DR_C = np.array([LX, LY - 0.18, BODY_Z0 + 0.22])
box((LX, LY - 0.21, BODY_Z0 + 0.22), (0.7, 0.02, 0.3), M_DARK, name='mbwin')
drum = link(bpy.data.objects.new('drum', None)); drum.location = DR_C + np.array([0, -0.05, 0]); drum.rotation_mode = 'XYZ'
dc = obj(prim('cyl', M_BRASS, 32), 'drumcyl'); dc.parent = drum; dc.rotation_euler = (0, math.pi / 2, 0); dc.scale = (0.07, 0.07, 0.55)
T_DRUM0 = TAG[0][0] - 0.6; OMEGA = 1.2
for t, p, v, d in TAG:
    k = TAGP.index(p); xk = -0.22 + 0.11 * k
    a = math.pi / 2 - OMEGA * (t - T_DRUM0)            # pin reaches the top (comb) at the note time
    pin = rod((xk, 0.07 * math.cos(a), 0.07 * math.sin(a)), (xk, 0.085 * math.cos(a), 0.085 * math.sin(a)), 0.006, M_STEEL, 6); pin.parent = drum
dang = np.array([0.0 if t < T_DRUM0 else -OMEGA * (t - T_DRUM0) for t in T])
key_series(drum, 'rotation_euler', 0, dang, 1e-5)
for k, p in enumerate(TAGP):
    xk = LX - 0.22 + 0.11 * k
    tn = box((xk, DR_C[1] - 0.05, DR_C[2] + 0.12), (0.05, 0.02, 0.12), M_KEY, name='tine'); tn.color = (*pitch_col(p), 0)
    times = [t for t, pp, v, d in TAG if pp == p]
    key_series(tn, 'color', 3, pulse(times, decay=5, length=20), 1e-3)
text('オルゴール', (LX, LY - 0.3, BODY_Z0 + BODY_H - 0.07), 0.09, M_TXT_GOLD, font=FONT_S)
# little pendulum: the child marble rides in its bob cup
lpv = link(bpy.data.objects.new('lpend', None)); lpv.location = LPIV; lpv.rotation_mode = 'XYZ'
lr = rod((0, 0, 0), (0, 0, -LP + RK + 0.03), 0.01, M_BRASS, 6); lr.parent = lpv
lcup = obj(prim('cone', M_BRASS, 24, 0.05, 0.12, 0.07, False), 'lcup'); lcup.parent = lpv; lcup.location = (0, 0, -LP - RK + 0.02)
lth = np.array([0.0 if t < lt['t_arr'] else -lt['dir'] * lt['amp'] * math.sin(math.pi / Q * (t - lt['t_arr'])) for t in T])
lth = np.array([lt['dir'] * lt['amp'] * math.sin(math.pi / Q * (t - lt['t_arr'])) if t >= lt['t_arr'] else 0.0 for t in T])
key_series(lpv, 'rotation_euler', 1, -lth, 1e-5)
# small hands on the little clock start moving when the child arrives
lh = link(bpy.data.objects.new('lhand', None)); lh.location = (LX, LY - 0.27, BODY_Z0 + 0.6); lh.rotation_mode = 'XYZ'
box((0, 0, 0.09), (0.02, 0.01, 0.18), M_DARK).parent = lh
lha = np.array([0.0 if t < lt['t_arr'] else -0.6 * (t - lt['t_arr']) for t in T]); key_series(lh, 'rotation_euler', 1, -lha, 1e-5)
text('2026', (LX, LY - 0.3, BODY_Z0 + 0.42), 0.07, M_TXT_D, font=FONT_S)

# ---------------- lights & world ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes['Background']; bg.inputs['Color'].default_value = (0.05, 0.04, 0.035, 1)
key_socket(bg.inputs['Strength'], 0.6 - 0.35 * AGE)
FOCUS = None   # set below; lamps follow the camera focus
# warm lamp light that dims with age; a cool moonlight at the stop; morning sun in the epilogue
lamp = bpy.data.lights.new('lamp', 'AREA'); lamp.size = 3.0; lamp.shape = 'DISK'
lo = link(bpy.data.objects.new('lamp', lamp)); lo.rotation_euler = (math.radians(70), 0, math.radians(-25))
key_socket(lamp.energy if False else bpy.data.lights['lamp'], [0]) if False else None
sun = bpy.data.lights.new('sun', 'SUN'); sun.angle = 0.2
so = link(bpy.data.objects.new('sun', sun)); so.rotation_euler = (math.radians(55), math.radians(-30), math.radians(-35))
fill = bpy.data.lights.new('fill', 'POINT'); fill.shadow_soft_size = 1.0
fo = link(bpy.data.objects.new('fill', fill))
e_lamp = np.array([900 - 450 * a if t < T_STOP + 0.5 else 300 + 500 * min(1, max(0, (t - 61.5) / 2.0)) for t, a in zip(T, AGE)])
c_lamp = np.array([(1.0, 0.82 - 0.12 * a, 0.62 - 0.25 * a) if t < 61.0 else (1.0, 0.9, 0.75) for t, a in zip(T, AGE)])
for f in range(NF):
    if 0 < f < NF - 1 and abs(e_lamp[f] - e_lamp[f - 1]) < 1e-3 and abs(e_lamp[f] - e_lamp[f + 1]) < 1e-3: continue
    lamp.energy = float(e_lamp[f]); lamp.keyframe_insert('energy', frame=f)
for f in range(0, NF, 6):
    lamp.color = tuple(map(float, c_lamp[f])); lamp.keyframe_insert('color', frame=f)
e_sun = np.array([1.6 - 0.9 * a if t < 61 else 0.7 + 1.8 * min(1, (t - 61) / 2.5) for t, a in zip(T, AGE)])
for f in range(0, NF, 3):
    sun.energy = float(e_sun[f]); sun.keyframe_insert('energy', frame=f)
    sun.color = (1.0, 0.92 - 0.2 * AGE[f], 0.8 - 0.35 * AGE[f]) if T[f] < 61 else (1.0, 0.95, 0.85); sun.keyframe_insert('color', frame=f)
fill.energy = 120; fill.color = (1.0, 0.85, 0.65)

# ---------------- camera: follows the playing marbles down the clock ----------------
def active_focus(t):
    pts = []; wts = []
    for n, P_ in PATHS.items():
        hs = [h['t'] for h in P_.hits]
        if not hs: continue
        d = min(abs(t - x) for x in hs)
        wgt = math.exp(-d / 0.9)
        if wgt > 0.02: pts.append(P_.at(t)); wts.append(wgt)
    if not pts: return None
    wts = np.array(wts); return (np.array(pts) * wts[:, None]).sum(0) / wts.sum()
FOC = np.zeros((NF, 3)); WID = np.full(NF, 4.3)
last = np.array([0, 0, 30.0])
for f in range(NF):
    t = T[f]
    a = active_focus(t)
    if a is not None: last = a
    FOC[f] = last
    if t < 2.4:                                    # opening: the hood (dial, title plate) and the hatch
        u = max(0.0, (t - 1.3) / 1.1)
        FOC[f] = (1 - u) * np.array([0.0, 0, 31.6]) + u * np.array([last[0] * 0.3, 0, last[2] + 0.8]); WID[f] = 4.5
    elif t > FINAL + 0.25:                         # epilogue: follow the child out of the clock to the little clock
        pk = PATHS['kid'].at(t)
        if t < lt['t_arr'] - 0.3:
            FOC[f] = np.array([pk[0], 0, max(pk[2], Z_R + 0.6) + 0.2]); WID[f] = 3.4
        else:
            FOC[f] = np.array([LX, 0, Z_R + 1.15]); WID[f] = 2.3
    else:
        FOC[f][0] *= 0.35                           # keep the clock's interior framed
        FOC[f][2] += 0.4
FOC[:, 0] = smooth(FOC[:, 0], 0.5); FOC[:, 2] = smooth(FOC[:, 2], 0.45); WID = smooth(WID, 0.6)
LENS = 30.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 400
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24
    CP[f] = (FOC[f, 0], -D - 0.1, FOC[f, 2] + 0.12 * D)
    TP[f] = (FOC[f, 0], 0.2, FOC[f, 2])
key_loc(cam, CP); key_loc(tgt, TP)
key_loc(lo, CP + np.array([1.5, 0.5, 2.5])); key_loc(fo, CP * np.array([1, 0, 1]) + np.array([-1.0, -1.5, 1.0]))

# ---------------- HUD (on the camera): hook, era year, closing question ----------------
hud = link(bpy.data.objects.new('hud', None)); hud.parent = cam; hud.location = (0, 0, -3.0)
HH_ = 3.0 * 24 / LENS; HW_ = HH_ * 9 / 16
M_PANEL = emis('panel', (0.02, 0.015, 0.01), 1.0)
def hud_text(s, y, size, m, t0, t1, panel=True):
    o = text(s, (0, y, 0.01), size * 1.75, m, rot=(0, 0, 0)); o.parent = hud; o.visible_shadow = False
    vis = np.where((T >= t0) & (T < t1), 1.0, 0.001)
    for k in range(3): key_series(o, 'scale', k, vis)
    if panel:
        w_ = min(HW_ * 0.98, size * 1.75 * 1.0 * len(s) + 0.12)
        pn = box((0, y, 0.0), (w_, size * 2.4, 0.004), M_PANEL, name='panel'); pn.parent = hud; pn.visible_shadow = False
        for k in range(3): key_series(pn, 'scale', k, vis * np.array([w_, size * 2.4, 0.004])[k])
    return o
hud_text('100年動いた古時計', HH_ / 2 - 0.3, 0.12, M_TXT_Y, 0.0, 3.6)
hud_text('中をのぞくと…', HH_ / 2 - 0.62, 0.1, M_TXT_W, 0.0, 3.6)
hud_text('♪ 大きな古時計（ビー玉オルゴール）', -HH_ / 2 + 0.3, 0.055, M_TXT_W, 2.0, 12.0)
ERAS = [('1876年', 0.0, 11.4), ('1900年', 11.4, 21.0), ('1925年', 21.0, 31.6), ('1950年', 31.6, 42.0), ('1965年', 42.0, 52.6), ('1976年', 52.6, 61.5), ('2026年', 63.2, 99)]
for s_, t0, t1 in ERAS:
    hud_text(s_, HH_ / 2 - 0.2, 0.1, M_TXT_GOLD, max(t0, 3.6), t1)
hud_text('あなたの家にも', HH_ / 2 - 0.5, 0.11, M_TXT_W, 66.8, 99)
hud_text('古い時計ある？', HH_ / 2 - 0.8, 0.11, M_TXT_Y, 66.8, 99)

# ---------------- render ----------------
if MODE == 'test':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS)); sc.frame_set(f)
        sc.render.filepath = f'{HERE}/test_{f:04d}.png'
        t1 = time.time(); bpy.ops.render.render(write_still=True); print('frame', f, 'sec', round(time.time() - t1, 1))
elif MODE == 'range':
    sc.frame_start, sc.frame_end = int(args[1]), int(args[2])
    sc.render.image_settings.file_format = 'PNG'; sc.render.filepath = f'{HERE}/frames/f_'
    bpy.ops.render.render(animation=True)
