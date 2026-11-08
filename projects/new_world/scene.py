"""New World Symphony IV - the ball train crosses the continent (玉の大陸横断鉄道), side view following the train.
The locomotive (black) rolls on its rail and clacks over a xylophone sleeper on every melody note; the four cars bounce
on their bells (red = bass beats 1/3, yellow/green/blue = chord beats 2/4, cello line in bars 26-33).
All motion comes from layout.py.  jp_nw.otf = Noto Sans CJK JP Bold (OFL) subset to the glyphs used here.   python3 scene.py -- check | test t1 t2 ... | range f0 f1"""
import os, sys, json, math, time
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix, Quaternion

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json')); ACC = json.load(open(f'{HERE}/acc.json'))
LAY = json.load(open(f'{HERE}/layout.json'))
FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.5) * FPS); T = np.arange(NF) / FPS
G = LAY['G']; RL = LAY['RL']; RC = LAY['RC']; YL = LAY['YL']; YC = LAY['YC']; EV = LAY['ev']
Q = E['q']; BAR = E['bar']; T0 = E['t0']
def tb(b): return T0 + (b - 1) * BAR
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
sc.render.resolution_x, sc.render.resolution_y = 1080, 1920
sc.render.resolution_percentage = int(os.environ.get('PCT', 50)); sc.render.fps = FPS
sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.5
try: sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Punchy'
except Exception: pass
sc.frame_start, sc.frame_end = 0, NF - 1
FONT = bpy.data.fonts.load(f'{HERE}/jp_nw.otf')


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


def glowmat(name, gain=8.0, base=0.0, rough=0.3, metal=0.0):
    """object colour = base colour; object alpha drives emission (hit flash)"""
    m = mat(name, (1, 1, 1), rough, **{'Coat Weight': 0.4, 'Metallic': metal}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    oi = nt.nodes.new('ShaderNodeObjectInfo')
    nt.links.new(oi.outputs['Color'], b.inputs['Base Color']); nt.links.new(oi.outputs['Color'], b.inputs['Emission Color'])
    mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY_ADD'; mul.inputs[1].default_value = gain; mul.inputs[2].default_value = base
    nt.links.new(oi.outputs['Alpha'], mul.inputs[0]); nt.links.new(mul.outputs[0], b.inputs['Emission Strength'])
    return m


def fxmat(name, gain=10.0):
    """transparent where object alpha = 0, glowing in the object colour where alpha = 1"""
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    oi = nt.nodes.new('ShaderNodeObjectInfo'); em = nt.nodes.new('ShaderNodeEmission'); tr_ = nt.nodes.new('ShaderNodeBsdfTransparent')
    mx = nt.nodes.new('ShaderNodeMixShader'); mu = nt.nodes.new('ShaderNodeMath'); mu.operation = 'MULTIPLY'; mu.inputs[1].default_value = gain
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(oi.outputs['Color'], em.inputs['Color']); nt.links.new(oi.outputs['Alpha'], mu.inputs[0]); nt.links.new(mu.outputs[0], em.inputs['Strength'])
    nt.links.new(oi.outputs['Alpha'], mx.inputs['Fac']); nt.links.new(tr_.outputs[0], mx.inputs[1]); nt.links.new(em.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs[0]); return m


def smokemat(name):
    """soft white steam / dust: diffuse, object alpha = opacity"""
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    oi = nt.nodes.new('ShaderNodeObjectInfo'); df = nt.nodes.new('ShaderNodeBsdfDiffuse'); tr_ = nt.nodes.new('ShaderNodeBsdfTransparent')
    mx = nt.nodes.new('ShaderNodeMixShader'); out = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(oi.outputs['Color'], df.inputs['Color'])
    nt.links.new(oi.outputs['Alpha'], mx.inputs['Fac']); nt.links.new(tr_.outputs[0], mx.inputs[1]); nt.links.new(df.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs[0]); return m


def objcol(name, rough=0.45, metal=0.0):
    m = mat(name, (1, 1, 1), rough, **{'Metallic': metal}); nt = m.node_tree
    oi = nt.nodes.new('ShaderNodeObjectInfo'); nt.links.new(oi.outputs['Color'], nt.nodes['Principled BSDF'].inputs['Base Color'])
    return m


def key_series(o, path, idx, vals, eps=1e-5):
    n = len(vals)
    for f in range(n):
        v = vals[f]
        if 0 < f < n - 1 and abs(v - vals[f - 1]) < eps and abs(v - vals[f + 1]) < eps: continue
        getattr(o, path)[idx] = float(v); o.keyframe_insert(path, index=idx, frame=f)


def key_loc(o, P):
    for k in range(3): key_series(o, 'location', k, P[:, k])


def key_val(sock, vals, eps=1e-3):
    n = len(vals)
    for f in range(n):
        if 0 < f < n - 1 and abs(vals[f] - vals[f - 1]) < eps and abs(vals[f] - vals[f + 1]) < eps: continue
        sock.default_value = float(vals[f]); sock.keyframe_insert('default_value', frame=f)


def key_col(sock, cols, eps=1e-3):
    n = len(cols)
    for f in range(n):
        if 0 < f < n - 1 and np.abs(cols[f] - cols[f - 1]).max() < eps and np.abs(cols[f] - cols[f + 1]).max() < eps: continue
        sock.default_value = (*map(float, cols[f]), 1); sock.keyframe_insert('default_value', frame=f)


_MC = {}
def prim(kind, m, *a):
    key = (kind, m.name if m else None) + tuple(a)
    if key in _MC: return _MC[key]
    bm = bmesh.new()
    if kind == 'cube': bmesh.ops.create_cube(bm, size=1.0)
    elif kind == 'cyl': bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=1.0, radius2=1.0, depth=1.0)
    elif kind == 'cone': bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=a[1], radius2=a[2], depth=a[3])
    elif kind == 'sph': bmesh.ops.create_uvsphere(bm, u_segments=a[0], v_segments=a[1], radius=1.0)
    elif kind == 'plane': bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    elif kind == 'ico': bmesh.ops.create_icosphere(bm, subdivisions=a[0], radius=1.0)
    me = bpy.data.meshes.new(kind); bm.to_mesh(me); bm.free()
    if kind in ('sph', 'cone', 'ico') or (kind == 'cyl' and a[0] > 12):
        for p in me.polygons: p.use_smooth = True
    if m: me.materials.append(m)
    _MC[key] = me; return me


def obj(me, name='o'): return link(bpy.data.objects.new(name, me))


def box(c, size, m, rot=None, name='box'):
    o = obj(prim('cube', m), name); o.location = c; o.scale = size
    if rot is not None: o.rotation_euler = rot
    return o


def rod(p0, p1, r, m, verts=8):
    p0, p1 = Vector(p0), Vector(p1); d = p1 - p0
    o = obj(prim('cyl', m, verts), 'rod'); q = Vector((0, 0, 1)).rotation_difference(d.normalized() if d.length > 1e-9 else Vector((0, 0, 1)))
    o.matrix_world = Matrix.Translation((p0 + p1) / 2) @ q.to_matrix().to_4x4() @ Matrix.Diagonal((r, r, max(1e-4, d.length), 1))
    return o


def sphere(c, r, m, seg=(16, 8), name='sph'):
    o = obj(prim('sph', m, *seg), name); o.location = c; o.scale = (r, r, r); return o


def text(s, loc, size, m, rot=(math.pi / 2, 0, 0), align='CENTER', ext=0.02):
    cu = bpy.data.curves.new('t', 'FONT'); cu.body = s; cu.font = FONT; cu.size = size; cu.extrude = ext
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
    up = np.array([0, 1.0, 0]) if abs(n[1]) < 0.9 else np.array([1.0, 0, 0])
    xa = np.cross(up, n); xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return Matrix(((xa[0], ya[0], n[0]), (xa[1], ya[1], n[1]), (xa[2], ya[2], n[2])))


def smooth(Xa, k):
    Y = Xa.copy(); a = 1 - math.exp(-1 / FPS / k)
    for f in range(1, len(Xa)): Y[f] = Y[f - 1] + (Xa[f] - Y[f - 1]) * a
    for f in range(len(Xa) - 2, -1, -1): Y[f] = Y[f + 1] + (Y[f] - Y[f + 1]) * a
    return Y


# ---------------- motion from the layout ----------------
def seg_at(g, t):
    p = np.array(g['p']); u = t - g['t0']
    if g['k'] == 'rest': return p
    if g['k'] == 'roll': return p + np.array(g['d']) * (g['v0'] * u + 0.5 * g['a'] * u * u)
    return p + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
class BallPath:
    def __init__(s, d): s.seg = d['seg']; s.hits = d['hits']; s.R = d['R']; s.y = d['y']
    def at(s, t):
        for g in s.seg:
            if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9: return seg_at(g, t)
        return seg_at(s.seg[0], s.seg[0]['t0']) if t < s.seg[0]['t0'] else seg_at(s.seg[-1], s.seg[-1]['t1'])
NAMES = ['loco', 'red', 'yellow', 'green', 'blue']
BP = {n: BallPath(LAY['balls'][n]) for n in NAMES}
POS = {n: np.array([BP[n].at(t) for t in T]) for n in NAMES}
LH = BP['loco'].hits
_lx = np.array([h['p'][0] for h in LH]); _lz = np.array([h['p'][2] for h in LH]) - RL
def rail_z(x): return float(np.interp(x, _lx, _lz))
LX = POS['loco'][:, 0]
def x_at_bar(b): return float(BP['loco'].at(tb(b))[0])
X = {b: x_at_bar(b) for b in range(1, 45)}
TR = EV['trestle']; RIM = EV['rim']; LEAP = EV['leap']
X_C0, X_C1 = TR['x0'] + 0.4, RIM['x']                    # canyon (between the rims)
X_TUN0, X_TUN1 = X[22] - 1.2, X[25] + 0.2                # summit tunnel
X_RIV0, X_RIV1 = X[17] - 0.4, X[18] + 0.1                # iron bridge over the river
X_END = float(POS['loco'][-1, 0])
def in_tunnel(x): return X_TUN0 < x < X_TUN1


# ---------------- materials ----------------
M_LOCO = mat('loco', (0.025, 0.025, 0.03), 0.18, **{'Coat Weight': 1.0, 'Metallic': 0.3})
CARCOL = {'red': (0.85, 0.06, 0.05), 'yellow': (1.0, 0.72, 0.05), 'green': (0.1, 0.65, 0.22), 'blue': (0.1, 0.32, 0.95)}
M_CAR = {n: mat('car_' + n, c, 0.15, **{'Coat Weight': 1.0}) for n, c in CARCOL.items()}
M_GOLD = mat('gold', (1.0, 0.7, 0.25), 0.25, **{'Metallic': 1.0})
M_BRASS = glowmat('brass', gain=6.0, base=0.0, rough=0.25, metal=0.8)
M_KEY = glowmat('key', gain=7.0, base=0.0, rough=0.35)
M_STEEL = mat('steel', (0.5, 0.52, 0.56), 0.3, **{'Metallic': 0.9})
M_IRON = mat('iron', (0.12, 0.13, 0.15), 0.45, **{'Metallic': 0.7})
M_DARK = mat('dark', (0.04, 0.04, 0.05), 0.6)
M_WOOD = mat('wood', (0.42, 0.26, 0.13), 0.75)
M_WOODD = mat('woodd', (0.26, 0.15, 0.07), 0.8)
M_GRAVEL = mat('gravel', (0.33, 0.3, 0.27), 0.95)
M_OBJ = objcol('objcol', 0.5)
M_FX = fxmat('fx', 8.0)
M_SMOKE = smokemat('smoke')
M_TXT_W = emis('txtw', (1, 1, 1), 1.4); M_TXT_Y = emis('txty', (1, 0.82, 0.25), 2.0)
M_TXT_D = mat('txtd', (0.05, 0.05, 0.08), 0.5)
M_LAMP = emis('lamp', (1.0, 0.8, 0.45), 6.0)
M_WATER = mat('water', (0.05, 0.25, 0.4), 0.05, **{'Coat Weight': 0.5})

# ---------------- balls ----------------
BALL = {}
for n in NAMES:
    R = BP[n].R
    bpy.ops.mesh.primitive_uv_sphere_add(radius=R, segments=40, ring_count=20); o = bpy.context.object; o.name = n
    for p in o.data.polygons: p.use_smooth = True
    o.data.materials.append(M_LOCO if n == 'loco' else M_CAR[n])
    key_loc(o, POS[n]); BALL[n] = o
    if n == 'loco':                       # gold wheel-rim band shows the rolling; headlamp + chimney cap ride along
        bpy.ops.mesh.primitive_torus_add(major_radius=R * 1.0, minor_radius=0.018, major_segments=40, minor_segments=8, rotation=(math.pi / 2, 0, 0))
        band = bpy.context.object; band.data.materials.append(M_GOLD); band.parent = o
        bpy.ops.mesh.primitive_torus_add(major_radius=R * 0.72, minor_radius=0.014, major_segments=40, minor_segments=8, rotation=(math.pi / 2, 0, 0), location=(0, -0.11, 0))
        b2 = bpy.context.object; b2.data.materials.append(M_GOLD); b2.parent = o
        for k in range(6):                # spokes so the wheel visibly turns
            a = k * math.pi / 3; sp = rod((0, -0.12, 0), (0.12 * math.cos(a), -0.12, 0.12 * math.sin(a)), 0.01, M_GOLD, 6); sp.parent = o
        rot = np.zeros(NF)
        for f in range(1, NF): rot[f] = rot[f - 1] + (POS[n][f, 0] - POS[n][f - 1, 0]) / R
        o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, rot)
    else:                                 # passenger car: a lit window band around the middle (stays level)
        bpy.ops.mesh.primitive_torus_add(major_radius=R * 0.99, minor_radius=0.026, major_segments=36, minor_segments=8)
        wb = bpy.context.object; wb.data.materials.append(emis('win_' + n, (1.0, 0.85, 0.5), 3.0)); wb.parent = o
        bpy.ops.mesh.primitive_torus_add(major_radius=R * 1.0, minor_radius=0.012, major_segments=36, minor_segments=6, location=(0, 0, 0.045))
        bpy.context.object.data.materials.append(M_GOLD); bpy.context.object.parent = o
# headlamp (in front of the loco, not rolling) + its light for the tunnel and the night
hl = sphere((0, 0, 0), 0.05, emis('headlamp', (1, 0.92, 0.7), 25.0), (12, 6), 'headlamp')
HLP = POS['loco'] + np.array([RL * 0.92, -0.05, 0.06]); key_loc(hl, HLP)
lt = bpy.data.lights.new('headlight', 'SPOT'); lt.energy = 0; lt.spot_size = math.radians(70); lt.color = (1, 0.9, 0.7); lt.shadow_soft_size = 0.05
lo_ = link(bpy.data.objects.new('headlight', lt)); lo_.rotation_euler = (0, math.radians(90 + 12), 0); key_loc(lo_, HLP + np.array([0.05, 0, 0]))
pl = bpy.data.lights.new('trainglow', 'POINT'); pl.energy = 0; pl.color = (1, 0.8, 0.55); pl.shadow_soft_size = 0.3
plo = link(bpy.data.objects.new('trainglow', pl)); key_loc(plo, (POS['loco'] + POS['green']) / 2 + np.array([0, -1.0, 0.8]))
# couplings: thin glowing links between consecutive balls
M_COUP = emis('coup', (1.0, 0.78, 0.4), 2.0)
for a, b in zip(NAMES, NAMES[1:]):
    c = obj(prim('cyl', M_COUP, 8), 'coupling'); c.rotation_mode = 'QUATERNION'
    Pa, Pb = POS[a], POS[b]; loc = (Pa + Pb) / 2; d = Pb - Pa; L_ = np.linalg.norm(d, axis=1)
    on = (T >= EV['gantry']['t'] + 0.85).astype(float)
    qs = np.zeros((NF, 4))
    for f in range(NF):
        q = Vector((0, 0, 1)).rotation_difference(Vector(d[f] / max(1e-6, L_[f]))); qs[f] = (q.w, q.x, q.y, q.z)
    key_loc(c, loc)
    for k in range(4): key_series(c, 'rotation_quaternion', k, qs[:, k], 1e-4)
    sl = np.maximum(1e-3, L_ - BP[a].R - BP[b].R)
    key_series(c, 'scale', 0, np.maximum(1e-4, 0.008 * on)); key_series(c, 'scale', 1, np.maximum(1e-4, 0.008 * on)); key_series(c, 'scale', 2, sl * np.maximum(1e-3, on))

# ---------------- melody sleepers (the loco's rail keys) ----------------
def pitch_col(p):
    h = ((p - 59) * 0.083) % 1.0
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0); return (r, g, b)
BARS = []          # dict(o, contact, n, t, kind, ball, hw, hd, th, c)
KEYW, KEYD, KEYT = 0.16, 0.46, 0.06
for i, h in enumerate(LH):
    k = h['kind']; n = np.array(h['n']); c = np.array(h['p']) - RL * n
    if k == 'bell': continue
    if k == 'cymbal':
        o = obj(prim('cyl', M_BRASS, 32), 'cymbal'); th = 0.02; scl = (0.26, 0.26, th * 2); col = (1.0, 0.75, 0.3)
        hw = hd = 0.26
    else:
        o = obj(prim('cube', M_KEY), 'key'); th = KEYT / 2; scl = (KEYW, KEYD, KEYT); col = pitch_col(h['pitch'])
        hw, hd = KEYW / 2, KEYD / 2
        if k in ('lever',): col = (1.0, 0.25, 0.15)
    ctr = c - n * th + n * 0.004
    o.matrix_world = Matrix.Translation(Vector(ctr)) @ frame_from_n(n).to_4x4() @ Matrix.Diagonal((*scl, 1))
    o.color = (*col, 0)
    BARS.append(dict(o=o, c=ctr, n=n, t=h['t'], kind=k, ball='loco', contact=c, hw=hw, hd=hd, th=th, t_fall=h.get('t_fall'), x=c[0], col=col))

# ---------------- car bells / drums ----------------
M_DRUMSKIN = glowmat('drumskin', gain=5.0, base=0.0, rough=0.6)
for n in NAMES[1:]:
    for h in BP[n].hits:
        k = h['kind']; nv = np.array(h['n']); c = np.array(h['p']) - RC * nv
        if k == 'cup':
            continue
        if n == 'red' and k in ('drum', 'tunnel', 'rim'):
            o = obj(prim('cyl', M_DRUMSKIN, 24), 'drum'); th = 0.05; scl = (0.15, 0.15, 0.1); col = (1.0, 0.92, 0.82)
            rimo = None
        elif k == 'trestle':
            o = obj(prim('cube', M_KEY), 'plank'); th = 0.025; scl = (0.2, 0.26, 0.05); col = tuple(0.55 * np.array(CARCOL[n]) + 0.45 * np.array([0.8, 0.55, 0.3]))
        else:
            o = obj(prim('cyl', M_BRASS, 24), 'bell'); th = 0.02; scl = (0.13, 0.13, 0.04)
            col = tuple(0.5 * np.array(CARCOL[n]) + 0.5 * np.array([1.0, 0.78, 0.35]))
        ctr = c - nv * th + nv * 0.004
        o.matrix_world = Matrix.Translation(Vector(ctr)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((*scl, 1))
        o.color = (*col, 0)
        hw = scl[0] if k != 'trestle' else 0.1; hd = scl[1] if k != 'trestle' else 0.13
        BARS.append(dict(o=o, c=ctr, n=nv, t=h['t'], kind=k, ball=n, contact=c, hw=hw, hd=hd, th=th, t_fall=h.get('t_fall'), x=c[0], col=col))

# hit flash + dip (+ the trestle pieces fall when the collapse reaches them)
for B_ in BARS:
    o = B_['o']; fh = int(round(B_['t'] * FPS)); base = o.location.copy(); nv = Vector(B_['n'])
    col = tuple(o.color[:3])
    o.color = (*col, 0); o.keyframe_insert('color', index=3, frame=max(0, fh - 1))
    for df, a in ((0, 1.0), (4, 0.45), (18, 0.1)):
        o.color = (*col, a); o.keyframe_insert('color', index=3, frame=fh + df)
    for df, d in ((-1, 0), (1, -0.022), (4, 0.006), (8, 0)):
        o.location = base + nv * d; o.keyframe_insert('location', frame=max(0, fh + df))
    if B_['t_fall'] is not None:
        tf_ = B_['t_fall']; f0 = int(math.floor(tf_ * FPS))
        o.keyframe_insert('location', frame=f0)
        rx = rng.uniform(-1.5, 1.5); o.rotation_mode = 'XYZ'; r0 = o.rotation_euler.copy(); o.keyframe_insert('rotation_euler', frame=f0)
        for f in range(f0 + 1, min(NF, f0 + 75), 3):
            u = f / FPS - tf_
            o.location = base + Vector((0, 0, -0.5 * G * u * u)); o.keyframe_insert('location', frame=f)
            o.rotation_euler = (r0[0] + rx * u, r0[1] + 0.8 * rx * u, r0[2]); o.keyframe_insert('rotation_euler', frame=f)

# hit rings (FX pool) and steam puffs from the loco on every melody note
NRING = 16
ring_loc = np.zeros((NRING, NF, 3)); ring_scl = np.full((NRING, NF), 1e-3); ring_a = np.zeros((NRING, NF)); ring_c = [(1, 1, 1)] * NRING
for k, B_ in enumerate(sorted(BARS, key=lambda b: b['t'])):
    r = k % NRING; f0 = int(round(B_['t'] * FPS))
    for df in range(12):
        f = f0 + df
        if f >= NF: break
        u = df / 11; ring_loc[r, f] = B_['contact'] + np.array([0, -0.3, 0.0]); ring_scl[r, f] = 0.25 + 0.9 * (1 - (1 - u) ** 2); ring_a[r, f] = (1 - u) ** 1.6
    ring_c[r] = B_['col']
for r in range(NRING):
    bpy.ops.mesh.primitive_torus_add(major_radius=0.2, minor_radius=0.012, major_segments=28, minor_segments=4, rotation=(math.pi / 2, 0, 0))
    o = bpy.context.object; o.data.materials.append(M_FX); o.color = (*ring_c[r], 0)
    for f in range(1, NF):
        if ring_a[r, f] == 0: ring_loc[r, f] = ring_loc[r, f - 1]
    key_loc(o, ring_loc[r])
    for k in range(3): key_series(o, 'scale', k, ring_scl[r])
    key_series(o, 'color', 3, ring_a[r], 1e-3)
NPUFF = 14
for r in range(NPUFF):
    o = obj(prim('ico', M_SMOKE, 2), 'puff'); o.color = (0.95, 0.95, 0.97, 0)
    P = np.zeros((NF, 3)); S = np.full(NF, 1e-3); A = np.zeros(NF)
    mine = [h for i, h in enumerate(LH) if i % NPUFF == r and h['kind'] != 'bell']
    for h in mine:
        f0 = int(round(h['t'] * FPS)); big = 1.0 + (0.8 if tb(8) - 0.1 < h['t'] < tb(10) else 0.0)
        for df in range(26):
            f = f0 + df
            if f >= NF: break
            u = df / 25; P[f] = np.array(h['p']) + np.array([-0.35 * u * big, 0.1, RL + 0.08 + 1.1 * u * big]); S[f] = (0.09 + 0.3 * u) * big
            A[f] = 0.28 * (1 - u) ** 1.5
    for f in range(1, NF):
        if A[f] == 0: P[f] = P[f - 1]
    key_loc(o, P)
    for k in range(3): key_series(o, 'scale', k, S, 1e-4)
    key_series(o, 'color', 3, A, 1e-3)

# ---------------- the loco's rail (steel beam the sleepers sit on) ----------------
def rail_strip(x0, x1, name, m, w=0.13, depth=0.1, y=YL, step=0.25, zf=rail_z):
    X_ = np.arange(x0, x1 + 1e-6, step)
    if len(X_) < 2: return None
    bm = bmesh.new(); vs = []
    for x in X_:
        z = zf(x)
        vs.append([bm.verts.new((x, y - w / 2, z)), bm.verts.new((x, y + w / 2, z)), bm.verts.new((x, y + w / 2, z - depth)), bm.verts.new((x, y - w / 2, z - depth))])
    for a, b in zip(vs, vs[1:]):
        for k in range(4): bm.faces.new((a[k], b[k], b[(k + 1) % 4], a[(k + 1) % 4]))
    bm.faces.new(vs[0][::-1]); bm.faces.new(vs[-1])
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new(name, me)); o.data.materials.append(m)
    return o
x_lip = LEAP['lip'][0]; x_land = LEAP['land'][0]
rail_strip(-3.0, TR['x0'], 'rail_a', M_IRON)
rail_strip(x_land - 0.3, X_END + 0.6, 'rail_c', M_IRON)
# on the trestle the rail is made of planks (one per sleeper gap) so it can fall apart
tr_keys = sorted([b for b in BARS if b['ball'] == 'loco' and b['t_fall'] is not None], key=lambda b: b['x'])
edges = [TR['x0']] + [(a['x'] + b['x']) / 2 for a, b in zip(tr_keys, tr_keys[1:])] + [x_lip + 0.12]
def wave_t(x): return TR['t_w0'] + (x - TR['x0']) / TR['vw']
TRESTLE_PIECES = []
for a, b in zip(edges, edges[1:]):
    o = rail_strip(a, b, 'plank', M_WOOD, w=0.15, step=max(0.05, (b - a) / 2))
    if o is None: continue
    xm = (a + b) / 2; passes = float(np.interp(b, LX, T))       # the loco has left the plank
    TRESTLE_PIECES.append((o, max(wave_t(xm), passes + 0.06), xm))
# the up-turned lip at the end of the last span
lip_c = np.array(LEAP['lip']) - np.array([0, 0, RL])
kick = box(tuple(lip_c + np.array([-0.12, 0, 0.03])), (0.32, 0.15, 0.04), M_WOOD, rot=(0, -math.radians(30), 0), name='lip')
TRESTLE_PIECES.append((kick, max(wave_t(x_lip), LEAP['t0'] + 0.06), x_lip))

# ---------------- terrain ----------------
def hash2(x, y):
    return (math.sin(x * 0.37 + y * 1.71) * 0.5 + math.sin(x * 0.11 - y * 0.53) * 0.8 + math.sin(x * 0.023 + y * 0.09) * 1.6)
Z_FLOOR = TR['z'] - 13.0
def ground(x, y):
    zr = rail_z(min(max(x, -3), X_END + 30)) - 0.55
    if x < -3: zr = rail_z(-3) - 0.55
    if x > X_END: zr = rail_z(X_END) - 0.55
    ay = abs(y - 0.4)
    near = zr - 0.25 * max(0, ay - 1.6)                      # embankment shoulders
    if y < -1.4: near = zr - 0.5 - 0.15 * (-1.4 - y)          # foreground slopes away from the camera
    far = zr - 0.3 + 0.18 * hash2(x, y) * min(1, max(0, y - 3) / 6)
    z = near if y < 3 else far
    # river under the iron bridge
    if X_RIV0 < x < X_RIV1: z -= 2.6 * math.sin(math.pi * (x - X_RIV0) / (X_RIV1 - X_RIV0)) ** 0.5
    # the canyon
    if X_C0 - 2 < x < X_C1 + 2:
        u = min((x - (X_C0 - 2)) / 2.5, ((X_C1 + 2) - x) / 2.5, 1.0)
        if u > 0: z = z * (1 - u) + (Z_FLOOR + 0.6 * hash2(x, y) * 0.3) * u
    if x > X_C1 + 2 - 1e-6 or x > x_land - 0.5:
        pass
    # mesas / mountains beyond
    if y > 12: z += 0.5 * max(0, hash2(x * 0.7, y))
    return z
def build_terrain():
    xs = np.arange(-14, X_END + 30, 0.6); ys = np.concatenate([np.arange(-14, -2, 1.0), np.arange(-2, 4, 0.35), np.arange(4, 26, 1.0), np.arange(26, 90, 4.0)])
    bm = bmesh.new(); V = {}
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            V[i, j] = bm.verts.new((x, y, ground(x, y)))
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            bm.faces.new((V[i, j], V[i + 1, j], V[i + 1, j + 1], V[i, j + 1]))
    me = bpy.data.meshes.new('terrain'); bm.to_mesh(me); bm.free()
    for p in me.polygons: p.use_smooth = True
    o = link(bpy.data.objects.new('terrain', me)); return o
TER = build_terrain()
M_TER = bpy.data.materials.new('ter'); M_TER.use_nodes = True; nt = M_TER.node_tree; bs = nt.nodes['Principled BSDF']; bs.inputs['Roughness'].default_value = 0.92
geo = nt.nodes.new('ShaderNodeNewGeometry'); sxyz = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(geo.outputs['Position'], sxyz.inputs[0])
mr = nt.nodes.new('ShaderNodeMapRange'); mr.inputs['From Min'].default_value = -14; mr.inputs['From Max'].default_value = X_END + 30
nt.links.new(sxyz.outputs['X'], mr.inputs['Value'])
ramp = nt.nodes.new('ShaderNodeValToRGB'); cr = ramp.color_ramp
span = X_END + 44
def at_x(x): return (x + 14) / span
STOPS = [(-14, (0.36, 0.34, 0.3)), (16, (0.4, 0.38, 0.3)), (26, (0.42, 0.45, 0.16)), (55, (0.55, 0.46, 0.16)), (68, (0.72, 0.45, 0.22)),
         (86, (0.45, 0.4, 0.38)), (X_C0 - 1, (0.62, 0.3, 0.16)), (X_C1 + 2, (0.6, 0.28, 0.14)), (X_C1 + 6, (0.16, 0.3, 0.12)),
         (X_END - 12, (0.14, 0.26, 0.12)), (X_END - 4, (0.3, 0.3, 0.32)), (X_END + 30, (0.3, 0.3, 0.32))]
cr.elements[0].position = at_x(STOPS[0][0]); cr.elements[0].color = (*STOPS[0][1], 1)
cr.elements[1].position = at_x(STOPS[-1][0]); cr.elements[1].color = (*STOPS[-1][1], 1)
for x_, c_ in STOPS[1:-1]:
    e_ = cr.elements.new(at_x(x_)); e_.color = (*c_, 1)
nt.links.new(mr.outputs[0], ramp.inputs['Fac'])
# rock strata by height (canyon walls), and a little noise
wave = nt.nodes.new('ShaderNodeTexWave'); wave.wave_type = 'BANDS'; wave.bands_direction = 'Z'; wave.inputs['Scale'].default_value = 0.35; wave.inputs['Distortion'].default_value = 2.0
nt.links.new(geo.outputs['Position'], wave.inputs['Vector'])
noi = nt.nodes.new('ShaderNodeTexNoise'); noi.inputs['Scale'].default_value = 0.6; nt.links.new(geo.outputs['Position'], noi.inputs['Vector'])
mix1 = nt.nodes.new('ShaderNodeMix'); mix1.data_type = 'RGBA'; mix1.blend_type = 'MULTIPLY'; mix1.inputs['Factor'].default_value = 0.35
nt.links.new(ramp.outputs['Color'], mix1.inputs[6]); nt.links.new(wave.outputs['Color'], mix1.inputs[7])
mix2 = nt.nodes.new('ShaderNodeMix'); mix2.data_type = 'RGBA'; mix2.blend_type = 'OVERLAY'; mix2.inputs['Factor'].default_value = 0.25
nt.links.new(mix1.outputs[2], mix2.inputs[6]); nt.links.new(noi.outputs['Color'], mix2.inputs[7])
nt.links.new(mix2.outputs[2], bs.inputs['Base Color'])
TER.data.materials.append(M_TER)
# gravel beds under the rows of the track
def bed(x0, x1, y0, y1, zoff=-0.48):
    X_ = np.arange(x0, x1, 0.5)
    if len(X_) < 2: return
    bm = bmesh.new(); vs = []
    for x in X_:
        z = rail_z(x) + zoff
        vs.append((bm.verts.new((x, y0, z)), bm.verts.new((x, y1, z))))
    for a, b in zip(vs, vs[1:]): bm.faces.new((a[0], b[0], b[1], a[1]))
    me = bpy.data.meshes.new('bed'); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new('bed', me)); o.data.materials.append(M_GRAVEL)
for x0, x1 in ((-3, X_RIV0), (X_RIV1, X_C0 - 1.5), (X_C1 + 1.0, X_END + 4)):
    bed(x0, x1, -0.75, 1.3)
# posts from the loco rail down to the bed (except on bridges), and under every car bell
for x in np.arange(-2.5, X_END, 1.4):
    if X_RIV0 < x < X_RIV1 or X_C0 - 1.5 < x < X_C1 + 1.0: continue
    rod((x, YL, rail_z(x) - 0.1), (x, YL, rail_z(x) - 0.5), 0.035, M_IRON, 6)
def post_floor(x, y):
    if X_RIV0 < x < X_RIV1: return rail_z(x) - 0.25          # the bridge deck
    if X_C0 - 1.5 < x < X_C1 + 1.0: return None
    return rail_z(x) - 0.48
for B_ in BARS:
    if B_['ball'] == 'loco' or B_['kind'] == 'trestle': continue
    c = B_['contact']; zb = post_floor(c[0], c[1])
    if zb is None: zb = Z_FLOOR if c[0] < RIM['x'] else RIM['z'] - 0.5
    if c[2] - 0.08 > zb: rod((c[0], c[1], c[2] - 0.06), (c[0], c[1], zb), 0.018, M_STEEL, 6)

# ---------------- departure station: engine shed, platform, sign, loading gantry + lever ----------------
zr0 = rail_z(0)
shed = box((-3.4, 0.5, zr0 + 1.0), (4.0, 3.6, 2.8), mat('shed', (0.35, 0.12, 0.08), 0.8), name='shed')
box((-1.38, YL, zr0 + 0.55), (0.06, 0.8, 1.1), M_DARK, name='door')          # dark doorway the loco rolls out of
box((-3.4, 0.5, zr0 + 2.55), (4.4, 4.0, 0.3), mat('roof', (0.2, 0.2, 0.22), 0.6), name='roofshed')
for k in range(3): sphere((-1.35, -0.9, zr0 + 1.6 - 0.3 * k), 0.06, M_LAMP, (10, 6))
plat_z = rail_z(6) - 0.2
box((7.0, 2.0, plat_z - 0.25), (14.0, 1.4, 0.5), mat('platform', (0.6, 0.58, 0.54), 0.8), name='platform')
for x in (1.5, 5.5, 9.5, 13.5):
    rod((x, 2.5, rail_z(x) - 0.4), (x, 2.5, rail_z(x) + 2.3), 0.05, M_IRON, 8)
    sphere((x, 2.45, rail_z(x) + 2.1), 0.07, M_LAMP, (10, 6))
box((7.5, 2.0, rail_z(7.5) + 2.45), (14.5, 1.8, 0.12), mat('canopy', (0.25, 0.32, 0.28), 0.6), name='canopy')
sign = box((4.2, 2.62, rail_z(4.2) + 1.25), (2.6, 0.06, 0.8), mat('signboard', (0.95, 0.94, 0.9), 0.6), name='sign')
text('東の駅', (4.2, 2.58, rail_z(4.2) + 1.4), 0.32, M_TXT_D, ext=0.003)
text('行き先：？？？', (4.2, 2.58, rail_z(4.2) + 1.07), 0.2, mat('txtr', (0.75, 0.05, 0.05), 0.5), ext=0.003)
# loading gantry with four cars on trap floors
g = EV['gantry']; gz = g['z'] - RC
gx = [g['x%d' % c] for c in range(4)]
for c in range(4):
    # trapdoor hinged at its back edge
    hz = link(bpy.data.objects.new('hinge', None)); hz.location = (gx[c] + 0.16, YC[c], gz - 0.02)
    dr = box((-0.16, 0, 0), (0.32, 0.26, 0.04), M_OBJ, name='trap'); dr.color = (*CARCOL[NAMES[c + 1]], 1); dr.parent = hz
    ang = np.zeros(NF)
    for f in range(NF):
        u = (T[f] - g['t']) / 0.18
        if u > 0: ang[f] = min(1.0, u) ** 2 * 1.5
    hz.rotation_mode = 'XYZ'; key_series(hz, 'rotation_euler', 1, ang)
gx0, gx1 = min(gx) - 0.5, max(gx) + 0.5
for xx in (gx0, gx1):
    for yy in (YC[0] - 0.3, YC[-1] + 0.3):
        rod((xx, yy, rail_z(xx) - 0.45), (xx, yy, gz + 0.5), 0.045, M_IRON, 8)
box(((gx0 + gx1) / 2, (YC[0] + YC[-1]) / 2, gz - 0.06), (gx1 - gx0 + 0.1, YC[-1] - YC[0] + 0.7, 0.05), M_IRON, name='gantryframe')
box(((gx0 + gx1) / 2, YC[0] - 0.32, gz + 0.5), (gx1 - gx0 + 0.1, 0.05, 0.08), M_IRON)
# hole in the gantry floor under each car is the trapdoor; signs
text('のりば', ((gx0 + gx1) / 2, YC[0] - 0.36, gz + 0.75), 0.22, M_TXT_Y)
# lever: the red sleeper is linked to the gantry by a rod that kicks the trapdoor bolts
lev = EV['lever']; lp = np.array(lev['p'])
rod((lp[0], YL - 0.24, lp[2] - RL - 0.05), (lp[0], YL - 0.24, gz + 0.3), 0.02, M_STEEL, 6)
rod((lp[0], YL - 0.24, gz + 0.3), (gx0, YC[0] - 0.3, gz + 0.3), 0.02, M_STEEL, 6)

# ---------------- telegraph poles + wires along the line ----------------
M_POLE = mat('pole', (0.3, 0.2, 0.1), 0.8)
pole_x = [x for x in np.arange(2.0, X_END + 6, 4.5) if not (X_C0 - 2 < x < X_C1 + 2) and not in_tunnel(x) and not (X_RIV0 - 0.5 < x < X_RIV1 + 0.5)]
YP = 2.4
for x in pole_x:
    zb = ground(x, YP); zt = rail_z(x) + 2.6
    rod((x, YP, zb - 0.2), (x, YP, zt), 0.05, M_POLE, 8); rod((x - 0.35, YP, zt - 0.2), (x + 0.35, YP, zt - 0.2), 0.025, M_POLE, 6)
for a, b in zip(pole_x, pole_x[1:]):
    if b - a > 5: continue
    for dx in (-0.3, 0.3):
        pts = [(a + (b - a) * u, YP, rail_z(a + (b - a) * u) + 2.4 - 0.25 * math.sin(math.pi * u)) for u in np.linspace(0, 1, 7)]
        for p0, p1 in zip(pts, pts[1:]): rod((p0[0] + dx * 0, p0[1], p0[2]), (p1[0], p1[1], p1[2]), 0.006, M_DARK, 4)

# ---------------- prairie & mesa scenery ----------------
M_MESA = mat('mesa', (0.62, 0.3, 0.16), 0.9)
for k in range(16):
    x = rng.uniform(24, 96); y = rng.uniform(26, 70); w = rng.uniform(3, 8); h = rng.uniform(2.5, 7)
    o = obj(prim('cone', M_MESA, 7, 1.0, 0.75, 1.0), 'mesa'); o.location = (x, y, ground(x, min(y, 25)) + h / 2 - 0.3); o.scale = (w, w * 0.6, h)
for k in range(10):                                         # snowy peaks behind the tunnel
    x = rng.uniform(X_TUN0 - 10, X_TUN1 + 6); y = rng.uniform(14, 45); h = rng.uniform(7, 14)
    o = obj(prim('cone', mat('peak%d' % k, (0.42, 0.4, 0.42), 0.9), 6, 1.0, 0.05, 1.0), 'peak'); o.location = (x, y, rail_z(x) + h / 2 - 1); o.scale = (h * 0.8, h * 0.6, h)
    s_ = obj(prim('cone', mat('snow', (0.92, 0.94, 0.98), 0.6), 6, 1.0, 0.05, 1.0), 'snow'); s_.location = (x, y - 0.05, rail_z(x) + h - 1 - h * 0.12); s_.scale = (h * 0.8 * 0.26, h * 0.6 * 0.26, h * 0.26)
M_CACT = mat('cactus', (0.15, 0.4, 0.16), 0.7)
for k in range(26):
    x = rng.uniform(20, 82); y = rng.uniform(3.2, 9)
    if X_RIV0 - 1 < x < X_RIV1 + 1: continue
    zb = ground(x, y); h = rng.uniform(0.6, 1.2)
    rod((x, y, zb), (x, y, zb + h), 0.09, M_CACT, 10); rod((x + 0.12, y, zb + h * 0.5), (x + 0.25, y, zb + h * 0.85), 0.05, M_CACT, 8)
# buffalo herd silhouettes on the prairie
M_BUF = mat('buffalo', (0.16, 0.1, 0.06), 0.9)
for k in range(9):
    x = 32 + k * 2.6 + rng.uniform(-0.6, 0.6); y = 11 + rng.uniform(-1, 3); zb = ground(x, y)
    b_ = sphere((x, y, zb + 0.45), 0.5, M_BUF, (12, 8)); b_.scale = (0.6, 0.35, 0.38)
    sphere((x + 0.42, y, zb + 0.5), 0.2, M_BUF, (10, 6))
# windmill
wm_x = 47.0; wm_z = ground(wm_x, 6.5)
rod((wm_x, 6.5, wm_z), (wm_x, 6.5, wm_z + 3.2), 0.06, M_IRON, 6)
wm = link(bpy.data.objects.new('windmill', None)); wm.location = (wm_x, 6.3, wm_z + 3.2)
for k in range(8):
    a = k * math.pi / 4; bl_ = box((0.35 * math.cos(a), 0, 0.35 * math.sin(a)), (0.5, 0.02, 0.12), M_STEEL, rot=(0, -a, 0)); bl_.parent = wm
wm.rotation_mode = 'XYZ'; key_series(wm, 'rotation_euler', 1, -T * 1.6)

# ---------------- iron truss bridge over the river (bar 17) ----------------
xb0, xb1 = X_RIV0, X_RIV1
zr_b = rail_z(xb0)
water = box(((xb0 + xb1) / 2, 6, zr_b - 2.4), (xb1 - xb0 + 1.5, 40, 0.1), M_WATER, name='river')
for yy in (YL - 0.35, YC[-1] + 0.3):
    for x in np.arange(xb0, xb1 + 0.01, (xb1 - xb0) / 4):
        rod((x, yy, rail_z(x) - 0.25), (x, yy, rail_z(x) + 1.25), 0.03, M_IRON, 6)
    xs_ = list(np.arange(xb0, xb1 + 0.01, (xb1 - xb0) / 4))
    for a, b in zip(xs_, xs_[1:]):
        rod((a, yy, rail_z(a) + 1.25), (b, yy, rail_z(b) + 1.25), 0.035, M_IRON, 6)
        rod((a, yy, rail_z(a) - 0.25), (b, yy, rail_z(b) + 1.25), 0.025, M_IRON, 6)
        rod((a, yy, rail_z(a) - 0.25), (b, yy, rail_z(b) - 0.25), 0.04, M_IRON, 6)
box(((xb0 + xb1) / 2, 0.35, rail_z((xb0 + xb1) / 2) - 0.3), (xb1 - xb0, 1.9, 0.08), M_IRON, name='deck')
for x in (xb0, xb1): rod((x, 0.35, rail_z(x) - 0.3), (x, 0.35, zr_b - 2.5), 0.12, mat('pier', (0.5, 0.48, 0.45), 0.9), 8)

# ---------------- summit tunnel (cut-away: back wall + vault, stone portals) ----------------
M_ROCK = mat('rock', (0.28, 0.26, 0.25), 0.95)
xt0, xt1 = X_TUN0, X_TUN1; xm_ = (xt0 + xt1) / 2
box((xm_, YC[-1] + 0.75, rail_z(xm_) + 1.0), (xt1 - xt0, 0.3, 4.0), M_ROCK, name='tunnelwall')
mtn = obj(prim('cone', M_ROCK, 8, 1.0, 0.1, 1.0), 'mountain'); mtn.location = (xm_, 6.0, rail_z(xm_) + 3.5); mtn.scale = ((xt1 - xt0) * 0.8, 6.0, 9.0)
for xx in (xt0, xt1):
    for yy in (YC[-1] + 0.5,):
        rod((xx, yy, rail_z(xx) - 0.5), (xx, yy, rail_z(xx) + 1.8), 0.16, mat('portal', (0.45, 0.42, 0.38), 0.9), 8)
    box((xx, 0.35, rail_z(xx) + 1.95), (0.5, 2.1, 0.4), mat('portal2', (0.45, 0.42, 0.38), 0.9))
# tunnel lamps on the back wall flash with every melody note in there (so the dark is alive)
tun_notes = [h['t'] for h in LH if h['kind'] == 'tunnel']
M_TLAMP = emis('tlamp', (1.0, 0.6, 0.25), 1.0)
key_val(M_TLAMP.node_tree.nodes['Emission'].inputs['Strength'], 0.6 + 10 * pulse_alpha(tun_notes, decay=3.0, length=10))
for x in np.arange(xt0 + 0.8, xt1, 1.6): sphere((x, YC[-1] + 0.58, rail_z(x) + 1.3), 0.06, M_TLAMP, (10, 6))

# ---------------- canyon trestle: wooden bents for both decks, collapse from bar 30 ----------------
M_BENT = M_WOODD
def car_deck_z(x):
    """bottom of the cars' bells on the trestle (lower deck)"""
    cs = [b for b in BARS if b['ball'] != 'loco' and b['kind'] == 'trestle']
    xs_ = np.array([b['contact'][0] for b in cs]); zs_ = np.array([b['contact'][2] for b in cs])
    o_ = np.argsort(xs_); return float(np.interp(x, xs_[o_], zs_[o_])) - 0.28
cdx = [b['contact'][0] for b in BARS if b['ball'] != 'loco' and b['kind'] == 'trestle']
for x in np.arange(TR['x0'] + 0.2, x_lip + 0.01, 1.25):
    pieces = []
    zt_ = rail_z(x) - 0.1
    for yy in (YL - 0.12, YL + 0.12):
        pieces.append(rod((x, yy, zt_), (x, yy, Z_FLOOR), 0.05, M_BENT, 6))
    pieces.append(rod((x, YL - 0.12, zt_ - 0.2), (x, YL + 0.12, zt_ - 0.2), 0.03, M_BENT, 4))
    for k in range(4):                                        # X-bracing down the bent
        z0_ = zt_ - 0.5 - k * 2.4; z1_ = z0_ - 2.4
        if z1_ < Z_FLOOR: break
        pieces.append(rod((x, YL - 0.12, z0_), (x + 0.6, YL + 0.12, z1_), 0.025, M_BENT, 4))
    if min(cdx) - 0.5 < x < max(cdx) + 0.6:                    # the cars' lower deck on the same bent
        zc_ = car_deck_z(x)
        pieces.append(rod((x, YC[0] - 0.1, zc_), (x, YC[-1] + 0.15, zc_), 0.04, M_BENT, 4))
        for yy in (YC[0] - 0.1, YC[-1] + 0.15): pieces.append(rod((x, yy, zc_), (x, yy, Z_FLOOR), 0.045, M_BENT, 6))
    tf_ = wave_t(x)
    passes = max([np.interp(x + 0.7, LX, T)] + [np.interp(x + 0.5, POS[n][:, 0], T) for n in NAMES[1:]])
    tf_ = max(tf_, passes + 0.1)
    TRESTLE_PIECES += [(p_, tf_, x) for p_ in pieces]
for x0_ in np.arange(min(cdx) - 0.4, max(cdx) + 0.4, 0.8):     # car deck stringers
    p_ = rod((x0_, (YC[0] + YC[-1]) / 2, car_deck_z(x0_)), (x0_ + 0.8, (YC[0] + YC[-1]) / 2, car_deck_z(x0_ + 0.8)), 0.035, M_BENT, 4)
    passes = max(np.interp(x0_ + 1.3, POS[n][:, 0], T) for n in NAMES[1:])
    TRESTLE_PIECES.append((p_, max(wave_t(x0_), passes + 0.1), x0_))
for B_ in BARS:                                                 # posts under the cars' trestle planks
    if B_['kind'] == 'trestle' and B_['ball'] != 'loco':
        c = B_['contact']; p_ = rod((c[0], c[1], c[2] - 0.05), (c[0], c[1], car_deck_z(c[0])), 0.016, M_BENT, 4)
        TRESTLE_PIECES.append((p_, B_['t_fall'], c[0]))
DUST = []
for o, tf_, x in TRESTLE_PIECES:
    f0 = int(math.floor(tf_ * FPS)); base = o.location.copy(); o.rotation_mode = 'XYZ'; r0 = o.rotation_euler.copy()
    if f0 >= NF: continue
    o.keyframe_insert('location', frame=f0); o.keyframe_insert('rotation_euler', frame=f0)
    sx = rng.uniform(-0.6, 0.6); zc = base.z
    for f in range(f0 + 1, min(NF, f0 + 80), 2):
        u = f / FPS - tf_
        o.location = base + Vector((0.3 * u, 0, -0.5 * G * u * u)); o.keyframe_insert('location', frame=f)
        o.rotation_euler = (r0[0] + sx * u, r0[1] + 0.5 * u, r0[2]); o.keyframe_insert('rotation_euler', frame=f)
    DUST.append((x, tf_ + math.sqrt(max(0.1, 2 * (zc - Z_FLOOR) / G))))
# dust clouds rise from the canyon floor where the wreckage lands
DUST.sort(key=lambda d: d[1]); dust_ev = []
for x, t in DUST:
    if not dust_ev or abs(t - dust_ev[-1][1]) > 0.35: dust_ev.append((x, t))
for k, (x, t) in enumerate(dust_ev[:40]):
    o = obj(prim('ico', M_SMOKE, 2), 'dust'); o.color = (0.75, 0.55, 0.4, 0)
    f0 = int(t * FPS); S = np.full(NF, 1e-3); A = np.zeros(NF); P = np.tile([x, 0.3, Z_FLOOR + 0.5], (NF, 1)).astype(float)
    for f in range(max(0, f0), min(NF, f0 + 60)):
        u = (f - f0) / 59; S[f] = 0.6 + 2.2 * u; A[f] = 0.8 * (1 - u) ** 1.2; P[f, 2] = Z_FLOOR + 0.5 + 1.5 * u
    key_loc(o, P)
    for c in range(3): key_series(o, 'scale', c, S, 1e-3)
    key_series(o, 'color', 3, A, 1e-3)
for k in range(9):
    x = X_C0 + (X_C1 - X_C0) * (k + 0.5) / 9 + rng.uniform(-2, 2); y = rng.uniform(30, 55); w = rng.uniform(3, 6)
    zt_b = rail_z(x) + rng.uniform(-1.5, 2.5)
    o = obj(prim('cone', M_MESA, 7, 1.0, 0.8, 1.0), 'butte'); o.location = (x, y, (zt_b + Z_FLOOR) / 2); o.scale = (w, w * 0.6, zt_b - Z_FLOOR)
canyon_river = box(((X_C0 + X_C1) / 2, 6, Z_FLOOR + 0.15), (X_C1 - X_C0 + 4, 50, 0.1), M_WATER, name='canyonriver')

# ---------------- evening valley: pines, crossing gates, city skyline ----------------
M_PINE = mat('pine', (0.08, 0.24, 0.12), 0.85)
for k in range(60):
    x = rng.uniform(X_C1 + 3, X_END - 6); y = rng.uniform(3.0, 22)
    zb = ground(x, y); h = rng.uniform(1.0, 2.6)
    o = obj(prim('cone', M_PINE, 7, 1.0, 0.02, 1.0), 'pine'); o.location = (x, y, zb + h / 2); o.scale = (h * 0.32, h * 0.32, h)
xg = X[40] + 2.0
for yy in (YL - 0.7, YC[-1] + 0.45):
    zg = rail_z(xg) - 0.45
    rod((xg, yy, zg), (xg, yy, zg + 1.3), 0.04, M_STEEL, 8)
    arm = box((xg - 0.7, yy, zg + 1.05), (1.4, 0.04, 0.06), M_OBJ, name='gatearm'); arm.color = (0.95, 0.95, 0.95, 1)
ACC_BEATS = [a['t'] for a in ACC if a['bar'] >= 38]
M_XLAMP = emis('xlamp', (1.0, 0.1, 0.05), 1.0)
key_val(M_XLAMP.node_tree.nodes['Emission'].inputs['Strength'], 0.3 + 14 * pulse_alpha(ACC_BEATS, decay=3.0, length=9))
for yy in (YL - 0.7, YC[-1] + 0.45):
    for dx in (-0.12, 0.12): sphere((xg + dx, yy - 0.05, rail_z(xg) + 0.95), 0.06, M_XLAMP, (10, 6))
M_WIN = bpy.data.materials.new('windows'); M_WIN.use_nodes = True; nt = M_WIN.node_tree; bs = nt.nodes['Principled BSDF']
bs.inputs['Base Color'].default_value = (0.08, 0.09, 0.12, 1); bs.inputs['Roughness'].default_value = 0.6
tc = nt.nodes.new('ShaderNodeTexCoord'); bri = nt.nodes.new('ShaderNodeTexBrick'); bri.inputs['Scale'].default_value = 1.0
bri.inputs['Color1'].default_value = (1, 0.8, 0.45, 1); bri.inputs['Color2'].default_value = (0.3, 0.25, 0.15, 1); bri.inputs['Mortar'].default_value = (0, 0, 0, 1)
bri.inputs['Mortar Size'].default_value = 0.35; bri.inputs['Brick Width'].default_value = 0.6; bri.inputs['Row Height'].default_value = 0.5
nt.links.new(tc.outputs['Object'], bri.inputs['Vector']); nt.links.new(bri.outputs['Color'], bs.inputs['Emission Color'])
WIN_ON = np.clip((T - 54.0) / 6.0, 0, 1) * 2.2
key_val(bs.inputs['Emission Strength'], WIN_ON)
for k in range(34):
    x = X_END - 14 + rng.uniform(0, 34); y = rng.uniform(10, 34); w = rng.uniform(1.2, 3.0); h = rng.uniform(2.0, 9.0)
    o = box((x, y, rail_z(min(x, X_END)) - 0.5 + h / 2), (w, w, h), M_WIN, name='bldg')

# ---------------- arrival station "新世界": bell at the end of the line, car cups, clock tower, fireworks ----------------
pe = POS['loco'][-1]
zs_ = rail_z(X_END)
bell_c = pe + np.array([RL + 0.22, 0, 0.05])
bpiv = link(bpy.data.objects.new('bellpivot', None)); bpiv.location = (bell_c[0], YL, bell_c[2] + 0.55)
bell = obj(prim('cone', M_BRASS, 32, 0.24, 0.1, 0.42), 'bigbell'); bell.parent = bpiv; bell.location = (0, 0, -0.55); bell.color = (1.0, 0.75, 0.3, 0)
key_series(bell, 'color', 3, pulse_alpha([FINAL], decay=12, length=60), 1e-3)
sw = np.zeros(NF)
for f in range(NF):
    u = T[f] - FINAL
    if u > 0: sw[f] = 0.35 * math.sin(5.5 * u) * math.exp(-1.2 * u)
bpiv.rotation_mode = 'XYZ'; key_series(bpiv, 'rotation_euler', 1, sw)
for yy in (YL - 0.35, YL + 0.35): rod((bell_c[0], yy, zs_ - 0.45), (bell_c[0], yy, bell_c[2] + 0.75), 0.04, M_IRON, 8)
rod((bell_c[0], YL - 0.35, bell_c[2] + 0.75), (bell_c[0], YL + 0.35, bell_c[2] + 0.75), 0.04, M_IRON, 8)
for n in NAMES[1:]:
    h = BP[n].hits[-1]; c = np.array(h['p'])
    bpy.ops.mesh.primitive_cylinder_add(radius=0.17, depth=0.14, vertices=28, location=(c[0], c[1], c[2] - RC + 0.04))
    cup = bpy.context.object; cup.data.materials.append(M_BRASS); cup.color = (*CARCOL[n], 0)
    bm = bmesh.new(); bm.from_mesh(cup.data); bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.normal.z > 0.9], context='FACES'); bm.to_mesh(cup.data); bm.free()
    cup.modifiers.new('s', 'SOLIDIFY').thickness = 0.02
    key_series(cup, 'color', 3, pulse_alpha([FINAL], decay=10, length=50), 1e-3)
    rod((c[0], c[1], c[2] - RC - 0.03), (c[0], c[1], zs_ - 0.48), 0.025, M_STEEL, 6)
    BARS.append(dict(o=cup, c=np.array([c[0], c[1], c[2] - RC - 0.02]), n=np.array([0, 0, 1.0]), t=FINAL, kind='cup', ball=n,
                     contact=c - np.array([0, 0, RC]), hw=0.17, hd=0.17, th=0.02, t_fall=None, x=c[0], col=CARCOL[n]))
BARS.append(dict(o=bell, c=bell_c, n=np.array([-1.0, 0, 0]), t=FINAL, kind='bell', ball='loco', contact=pe + np.array([RL, 0, 0]),
                 hw=0.2, hd=0.2, th=0.2, t_fall=None, x=bell_c[0], col=(1, 0.75, 0.3)))
xs_st = X_END - 6
box((xs_st, 2.0, zs_ - 0.45), (13.0, 1.4, 0.5), mat('platform2', (0.62, 0.6, 0.56), 0.8), name='platform2')
box((xs_st, 2.0, zs_ + 2.45), (13.0, 1.8, 0.12), mat('canopy2', (0.35, 0.18, 0.12), 0.6), name='canopy2')
for x in np.arange(xs_st - 6, xs_st + 6.5, 3.0):
    rod((x, 2.6, zs_ - 0.4), (x, 2.6, zs_ + 2.4), 0.05, M_IRON, 8); sphere((x, 2.55, zs_ + 2.2), 0.08, M_LAMP, (10, 6))
sign2 = box((X_END - 3.0, 2.68, zs_ + 1.35), (2.8, 0.06, 0.9), mat('signboard2', (0.95, 0.94, 0.9), 0.6), name='sign2')
M_SIGN = emis('signtxt', (1.0, 0.75, 0.2), 1.0)
key_val(M_SIGN.node_tree.nodes['Emission'].inputs['Strength'], 0.05 + 4.0 * np.clip((T - FINAL) / 0.3, 0, 1))
text('新世界', (X_END - 3.0, 2.63, zs_ + 1.47), 0.42, M_SIGN, ext=0.003)
text('NEW WORLD', (X_END - 3.0, 2.63, zs_ + 1.11), 0.17, M_TXT_D, ext=0.003)
# clock tower
ctx, cty = X_END - 1.0, 6.0
box((ctx, cty, zs_ + 3.0), (1.8, 1.8, 7.0), mat('tower', (0.55, 0.3, 0.2), 0.8), name='tower')
cone_ = obj(prim('cone', mat('towerroof', (0.15, 0.3, 0.25), 0.6), 4, 1.0, 0.05, 1.0), 'roof'); cone_.location = (ctx, cty, zs_ + 7.5); cone_.scale = (1.4, 1.4, 1.6); cone_.rotation_euler = (0, 0, math.pi / 4)
M_FACE = emis('clockface', (1.0, 0.95, 0.8), 1.0)
key_val(M_FACE.node_tree.nodes['Emission'].inputs['Strength'], 0.6 + 2.5 * np.clip((T - 56) / 6, 0, 1) + 6 * pulse_alpha([FINAL], decay=15, length=60))
face = obj(prim('cyl', M_FACE, 32), 'face'); face.location = (ctx, cty - 0.92, zs_ + 5.6); face.scale = (0.65, 0.65, 0.04); face.rotation_euler = (math.pi / 2, 0, 0)
for L_, wdt, ang in ((0.45, 0.04, 0.0), (0.32, 0.05, -math.pi / 2)):        # 9 o'clock - arrival time
    hnd = box((ctx + 0.5 * L_ * math.sin(ang), cty - 0.97, zs_ + 5.6 + 0.5 * L_ * math.cos(ang)), (wdt, 0.02, L_), M_DARK, rot=(0, ang, 0))
# fireworks: bursts over the city right after the final chord (each burst = one sparkle in the audio)
FW = [(FINAL + 0.15, -1.5, 4.8, (1, 0.3, 0.2)), (FINAL + 0.75, 2.0, 5.6, (0.3, 0.7, 1)), (FINAL + 1.35, -3.5, 6.2, (1, 0.85, 0.2)),
      (FINAL + 1.95, 1.0, 6.8, (0.4, 1, 0.4)), (FINAL + 2.55, -0.8, 5.2, (1, 0.4, 0.9)), (FINAL + 3.15, 3.2, 6.0, (1, 1, 1))]
NSP = 36
for tf_, dx, hz, col in FW:
    cx_, cz_ = X_END + dx, zs_ + hz; cy_ = 9.0
    dirs = rng.normal(size=(NSP, 3)); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    for k in range(NSP):
        o = obj(prim('ico', M_FX, 1), 'spark'); o.color = (*col, 0)
        P = np.tile([cx_, cy_, cz_], (NF, 1)).astype(float); A = np.zeros(NF); S = np.full(NF, 1e-3)
        f0 = int(round(tf_ * FPS))
        for f in range(f0, min(NF, f0 + 45)):
            u = (f - f0) / FPS; sp = 2.6 * (1 - math.exp(-u / 0.25))
            P[f] = (cx_ + dirs[k, 0] * sp, cy_ + dirs[k, 1] * sp * 0.4, cz_ + dirs[k, 2] * sp - 0.6 * u * u); A[f] = (1 - (f - f0) / 45) ** 1.3; S[f] = 0.05
        key_loc(o, P); key_series(o, 'color', 3, A, 1e-3)
        for c in range(3): key_series(o, 'scale', c, S, 1e-4)
fwl = bpy.data.lights.new('fwlight', 'POINT'); fwl.energy = 0; fwl.shadow_soft_size = 2.0
fwo = link(bpy.data.objects.new('fwlight', fwl)); fwo.location = (X_END, 4.0, zs_ + 5.5)
en = 2500 * pulse_alpha([f[0] for f in FW], decay=6, length=30)
for f in range(NF):
    if 0 < f < NF - 1 and abs(en[f] - en[f - 1]) < 1 and abs(en[f] - en[f + 1]) < 1: continue
    fwl.energy = float(en[f]); fwl.keyframe_insert('energy', frame=f)

# ---------------- world: time of day ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bgn = w.node_tree.nodes['Background']
KT = [(0, (0.55, 0.42, 0.5), 0.5, (1.0, 0.7, 0.55), 2.2, 12),
      (12, (0.45, 0.6, 0.9), 0.7, (1.0, 0.93, 0.82), 3.2, 40),
      (33.0, (0.45, 0.6, 0.9), 0.7, (1.0, 0.93, 0.82), 3.2, 40),
      (33.8, (0.08, 0.08, 0.11), 0.14, (1.0, 0.9, 0.8), 0.12, 40),
      (38.2, (0.08, 0.08, 0.11), 0.14, (1.0, 0.9, 0.8), 0.12, 40),
      (39.0, (0.6, 0.55, 0.6), 0.65, (1.0, 0.82, 0.6), 3.0, 30),
      (52.0, (0.7, 0.5, 0.45), 0.6, (1.0, 0.65, 0.4), 2.6, 15),
      (60.0, (0.35, 0.25, 0.35), 0.35, (1.0, 0.45, 0.3), 1.2, 6),
      (66.0, (0.06, 0.07, 0.15), 0.25, (0.6, 0.65, 1.0), 0.25, 30),
      (80.0, (0.06, 0.07, 0.15), 0.25, (0.6, 0.65, 1.0), 0.25, 30)]
kt_t = np.array([k[0] for k in KT])
def interp(i):
    return np.stack([np.interp(T, kt_t, [np.array(k[i], float)[c] if np.ndim(k[i]) else k[i] for k in KT]) for c in range(3)], 1) if np.ndim(KT[0][i]) else np.interp(T, kt_t, [k[i] for k in KT])
WC, WS, SC, SS, SE = interp(1), interp(2), interp(3), interp(4), interp(5)
key_col(bgn.inputs['Color'], WC); key_val(bgn.inputs['Strength'], WS)
sun = bpy.data.lights.new('sun', 'SUN'); sun.angle = 0.12
so = link(bpy.data.objects.new('sun', sun)); so.rotation_mode = 'XYZ'
for f in range(0, NF, 3):
    sun.energy = float(SS[f]); sun.keyframe_insert('energy', frame=f)
    sun.color = tuple(map(float, SC[f])); sun.keyframe_insert('color', frame=f)
    so.rotation_euler = (math.radians(90 - SE[f]), math.radians(-28), math.radians(-35)); so.keyframe_insert('rotation_euler', frame=f)
lt_e = np.where((T > 33.5) & (T < 38.6), 90.0, 0.0) + np.clip((T - 58) / 6, 0, 1) * 30.0
pl_e = np.where((T > 33.5) & (T < 38.6), 25.0, 0.0) + np.clip((T - 60) / 5, 0, 1) * 18.0
for f in range(NF):
    if 0 < f < NF - 1 and lt_e[f] == lt_e[f - 1] == lt_e[f + 1] and pl_e[f] == pl_e[f - 1] == pl_e[f + 1]: continue
    lt.energy = float(lt_e[f]); lt.keyframe_insert('energy', frame=f); pl.energy = float(pl_e[f]); pl.keyframe_insert('energy', frame=f)
# sky backdrop: gradient wall far behind, colours follow the time of day; stars at night
SKY = bpy.data.materials.new('sky'); SKY.use_nodes = True; nt = SKY.node_tree; nt.nodes.clear()
tc = nt.nodes.new('ShaderNodeTexCoord'); sepz = nt.nodes.new('ShaderNodeSeparateXYZ'); mp = nt.nodes.new('ShaderNodeMapRange')
mp.inputs['From Min'].default_value = -0.5; mp.inputs['From Max'].default_value = 0.5
mixs = nt.nodes.new('ShaderNodeMix'); mixs.data_type = 'RGBA'
em = nt.nodes.new('ShaderNodeEmission'); out = nt.nodes.new('ShaderNodeOutputMaterial')
nt.links.new(tc.outputs['Object'], sepz.inputs[0]); nt.links.new(sepz.outputs['Y'], mp.inputs['Value']); nt.links.new(mp.outputs[0], mixs.inputs['Factor'])
nt.links.new(mixs.outputs[2], em.inputs['Color']); nt.links.new(em.outputs[0], out.inputs[0])
star = nt.nodes.new('ShaderNodeTexVoronoi'); star.inputs['Scale'].default_value = 140.0
st_r = nt.nodes.new('ShaderNodeMath'); st_r.operation = 'LESS_THAN'; st_r.inputs[1].default_value = 0.03
nt.links.new(tc.outputs['Object'], star.inputs['Vector']); nt.links.new(star.outputs['Distance'], st_r.inputs[0])
st_m = nt.nodes.new('ShaderNodeMath'); st_m.operation = 'MULTIPLY'; nt.links.new(st_r.outputs[0], st_m.inputs[0])
addc = nt.nodes.new('ShaderNodeMix'); addc.data_type = 'RGBA'; addc.blend_type = 'ADD'
nt.links.new(st_m.outputs[0], addc.inputs['Factor']); nt.links.new(mixs.outputs[2], addc.inputs[6]); addc.inputs[7].default_value = (1, 1, 1, 1)
nt.links.new(addc.outputs[2], em.inputs['Color'])
SKYK = [(0, (1.0, 0.62, 0.45), (0.35, 0.4, 0.7)), (12, (0.75, 0.85, 1.0), (0.2, 0.42, 0.85)), (39, (1.0, 0.85, 0.6), (0.3, 0.5, 0.85)),
        (52, (1.0, 0.6, 0.35), (0.45, 0.35, 0.6)), (60, (0.95, 0.35, 0.2), (0.2, 0.15, 0.35)), (66, (0.12, 0.12, 0.25), (0.02, 0.02, 0.07)), (80, (0.12, 0.12, 0.25), (0.02, 0.02, 0.07))]
sk_t = np.array([k[0] for k in SKYK])
LOWC = np.stack([np.interp(T, sk_t, [k[1][c] for k in SKYK]) for c in range(3)], 1)
HIC = np.stack([np.interp(T, sk_t, [k[2][c] for k in SKYK]) for c in range(3)], 1)
key_col(mixs.inputs[6], LOWC); key_col(mixs.inputs[7], HIC)
key_val(st_m.inputs[1], np.clip((T - 62) / 4, 0, 1) * 1.5)
key_val(em.inputs['Strength'], np.where((T > 33.8) & (T < 38.2), 0.05, 1.0))
for x0 in np.arange(-30, X_END + 60, 60.0):
    o = obj(prim('plane', SKY), 'sky'); o.location = (x0 + 30, 95, rail_z(min(max(x0 + 30, 0), X_END)) + 10); o.rotation_euler = (math.pi / 2, 0, 0); o.scale = (60.2, 75, 1)
# big low sun disc at dawn / sunset
M_SUND = emis('sundisc', (1.0, 0.6, 0.3), 1.0)
key_val(M_SUND.node_tree.nodes['Emission'].inputs['Strength'], np.where(T < 13, 6.0, 0.0) + np.where((T > 50) & (T < 64), 6.0, 0.0))
for x_, z_ in ((10, 3.0), (150, -2.5)):
    o = obj(prim('cyl', M_SUND, 40), 'sundisc'); o.location = (x_ + 6, 90, rail_z(x_) + z_); o.scale = (3.2, 3.2, 0.1); o.rotation_euler = (math.pi / 2, 0, 0)

# ---------------- camera ----------------
CEN = (POS['loco'] * 1.2 + POS['red'] + POS['yellow'] + POS['green'] + POS['blue']) / 5.2
FOC = np.zeros((NF, 3)); WID = np.zeros(NF)
for f in range(NF):
    t = T[f]; c = CEN[f].copy()
    zl = POS['loco'][f, 2]; zc = np.mean([POS[n][f, 2] for n in NAMES[1:]])
    FOC[f] = (c[0] - 0.1, 0, 0.5 * zl + 0.5 * zc + 0.85); WID[f] = 4.8
    if t < 3.0:                       # opening: shed, ramp, sign and the cars waiting on the gantry
        u = t / 3.0; FOC[f] = (POS['loco'][f, 0] + 1.0 + 1.0 * u, 0, rail_z(POS['loco'][f, 0]) + 1.2); WID[f] = 5.4
    elif t < tb(8) + 0.4:
        FOC[f] = (POS['loco'][f, 0] + 0.9, 0, rail_z(POS['loco'][f, 0]) + 1.2); WID[f] = 5.2
    if tb(26) + 0.3 < t < tb(34) + 0.6:   # trestle: wider and lower so the collapse, the canyon and the leap are visible
        WID[f] = 5.6; FOC[f, 2] = 0.5 * zl + 0.5 * zc + 0.2
    if t > FINAL + 0.3:               # final: pull back over the station, city and fireworks
        u = min(1.0, (t - FINAL - 0.3) / 2.2); u = u * u * (3 - 2 * u)
        FOC[f] = FOC[f] * (1 - u) + np.array([X_END - 1.0, 0, zs_ + 3.2]) * u; WID[f] = 4.8 * (1 - u) + 8.5 * u
FOC[:, 0] = smooth(FOC[:, 0], 0.25); FOC[:, 2] = smooth(FOC[:, 2], 0.55); WID = smooth(WID, 0.6)
LENS = 32.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 400
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24
    CP[f] = (FOC[f, 0], -D, FOC[f, 2] + 0.1 * D)
    TP[f] = (FOC[f, 0], 0.4, FOC[f, 2])
key_loc(cam, CP); key_loc(tgt, TP)

# ---------------- HUD: route map (東の駅 -> ？？？ -> 新世界), hook question ----------------
hud = link(bpy.data.objects.new('hud', None)); hud.parent = cam; hud.location = (0, 0, -3.0)
HH = 3.0 * 24 / LENS; HWd = HH * 9 / 16
pz = HH / 2 - 0.2
M_PANEL = emis('panel', (0.02, 0.025, 0.05), 1.0)
pn = box((0, pz - 0.02, 0), (HWd * 0.96, 0.17, 0.004), M_PANEL, name='panel'); pn.parent = hud
x0h, x1h = -HWd * 0.2, HWd * 0.2
ln = box((0, pz - 0.03, 0.003), (x1h - x0h, 0.012, 0.002), emis('route', (0.9, 0.75, 0.4), 1.5), name='routeline'); ln.parent = hud
t1_ = text('東の駅', (x0h - 0.16, pz - 0.03, 0.004), 0.07, M_TXT_W, rot=(0, 0, 0)); t1_.parent = hud
q_ = text('？？？', (x1h + 0.16, pz - 0.03, 0.004), 0.07, M_TXT_Y, rot=(0, 0, 0)); q_.parent = hud
nw_ = text('新世界', (x1h + 0.16, pz - 0.03, 0.004), 0.07, M_TXT_Y, rot=(0, 0, 0)); nw_.parent = hud
swp = np.clip((T - FINAL - 0.1) / 0.25, 0, 1)
key_series(q_, 'scale', 1, np.maximum(1e-3, 1 - swp), 1e-3); key_series(nw_, 'scale', 1, np.maximum(1e-3, swp), 1e-3)
dot = sphere((0, 0, 0), 0.03, emis('dot', (1, 1, 1), 3.0), (10, 6)); dot.parent = hud
prog = np.clip(POS['loco'][:, 0] / X_END, 0, 1)
DP = np.stack([x0h + (x1h - x0h) * prog, np.full(NF, pz - 0.03), np.full(NF, 0.006)], 1); key_loc(dot, DP)
for k, n in enumerate(NAMES[1:]):
    d2 = sphere((0, 0, 0), 0.016, emis('dot' + n, CARCOL[n], 3.0), (8, 6)); d2.parent = hud
    key_loc(d2, DP - np.array([0.045 * (k + 1), 0, 0]))
hk = text('この列車、どこ行き？', (0, pz - 0.3, 0.004), 0.11, M_TXT_W, rot=(0, 0, 0)); hk.parent = hud
hs_ = np.where(T < 3.2, 1.0, np.clip(1 - (T - 3.2) / 0.3, 0, 1)); key_series(hk, 'scale', 0, np.maximum(1e-3, hs_), 1e-3)
ar = text('新世界に到着！', (0, -HH / 2 + 0.55, 0.004), 0.16, M_TXT_Y, rot=(0, 0, 0)); ar.parent = hud
as_ = np.clip((T - FINAL - 0.8) / 0.3, 0, 1); key_series(ar, 'scale', 0, np.maximum(1e-3, as_), 1e-3)
cq = text('どこ行きの列車だと思った？', (0, -HH / 2 + 0.36, 0.004), 0.085, M_TXT_W, rot=(0, 0, 0)); cq.parent = hud
cs_ = np.clip((T - FINAL - 1.6) / 0.3, 0, 1); key_series(cq, 'scale', 0, np.maximum(1e-3, cs_), 1e-3)
for o in bpy.data.objects:
    if o.parent is hud: o.visible_shadow = False


# ---------------- sanity checks ----------------
def check():
    info = {}
    gaps = []
    for B_ in BARS:
        n = B_['ball']; f = int(round(B_['t'] * FPS)); p = POS[n][min(NF - 1, f)]
        gaps.append(np.linalg.norm(p - (B_['contact'] + BP[n].R * B_['n'])))
    info['contact_gap_at_hit_frame_max'] = round(max(gaps), 3)
    info['bars'] = len(BARS); info['objects'] = len(bpy.data.objects); info['NF'] = NF
    return info


if MODE == 'check':
    json.dump({'dust': [round(t, 4) for x, t in dust_ev[:40] if t < NF / FPS], 'fw': [f[0] for f in FW]}, open(f'{HERE}/sfx_scene.json', 'w'))
if MODE == 'check' or os.environ.get('CHECK'): print(check())
if MODE == 'check': sys.exit(0)
if MODE == 'test':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS)); sc.frame_set(f)
        sc.render.filepath = f'{HERE}/test_{f:04d}.png'
        t1 = time.time(); bpy.ops.render.render(write_still=True); print('frame', f, 'sec', round(time.time() - t1, 1))
elif MODE == 'range':
    sc.frame_start, sc.frame_end = int(args[1]), int(args[2])
    sc.render.image_settings.file_format = 'PNG'; sc.render.filepath = f'{HERE}/frames/f_'
    bpy.ops.render.render(animation=True)
