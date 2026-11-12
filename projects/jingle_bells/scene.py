"""Jingle Bells - 'Who wins the sleigh race?' Red and green jingle-bell marbles race down a snowy mountain at night.
Every melody note = a racer landing on an ice key of that pitch (layout.py plans all motion: constant gravity, bounces that
never add speed, rolling ramps, the ski lift is the only energy input). Accompaniment, all struck on screen:
  bass        snowmen beat their timpani on every bass note (one master mallet swing, copied by every snowman)
  off-beats   candy-cane hammers ring the golden sleigh bells on every off-beat chord
python3 scene.py -- check | test t1 t2 ... | range f0 f1 | probe t1 ..."""
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
G = LAY['G']; R = LAY['R']; EV = LAY['ev']; LY = LAY['lane_y']; DZL = LAY['dzl']
TN = np.array([l[0] for l in LEAD]); SECS = dict(E['secs'])
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
FONT = bpy.data.fonts.load(f'{HERE}/jp_sub.otf')
FONT_EN = bpy.data.fonts.load(f'{HERE}/en_bold.ttf')


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


def glowmat(name, gain=8.0, base=0.0, rough=0.3, metal=0.0, trans=0.0):
    """object colour = base colour; object alpha drives emission (hit flash)"""
    m = mat(name, (1, 1, 1), rough, **{'Coat Weight': 0.5, 'Metallic': metal}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    if trans: b.inputs['Transmission Weight'].default_value = trans
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


def objcol(name, rough=0.45, metal=0.0, emit=0.0):
    m = mat(name, (1, 1, 1), rough, **{'Metallic': metal}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
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


def key_socket(sock, vals, eps=1e-3):
    for f in range(len(vals)):
        if 0 < f < len(vals) - 1 and abs(vals[f] - vals[f - 1]) < eps and abs(vals[f] - vals[f + 1]) < eps: continue
        sock.default_value = float(vals[f]); sock.keyframe_insert('default_value', frame=f)


_MC = {}
def prim(kind, m, *a):
    key = (kind, m.name if m else None) + tuple(a)
    if key in _MC: return _MC[key]
    bm = bmesh.new()
    if kind == 'cube': bmesh.ops.create_cube(bm, size=1.0)
    elif kind == 'cyl': bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=1.0, radius2=1.0, depth=1.0)
    elif kind == 'cone': bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=a[1], radius2=a[2], depth=a[3])
    elif kind == 'sph': bmesh.ops.create_uvsphere(bm, u_segments=a[0], v_segments=a[1], radius=1.0)
    elif kind == 'ico': bmesh.ops.create_icosphere(bm, subdivisions=a[0], radius=1.0)
    elif kind == 'plane': bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    elif kind == 'torus':
        me = bpy.data.meshes.new('torus'); bpy.ops.mesh.primitive_torus_add(major_radius=a[0], minor_radius=a[1], major_segments=a[2], minor_segments=a[3])
        o = bpy.context.object; me = o.data; bpy.data.objects.remove(o)
        for p in me.polygons: p.use_smooth = True
        if m: me.materials.append(m)
        _MC[key] = me; bm.free(); return me
    me = bpy.data.meshes.new(kind); bm.to_mesh(me); bm.free()
    if kind in ('sph', 'cone', 'ico'):
        for p in me.polygons: p.use_smooth = True
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


def cone(c, r1, r2, h, m, seg=16, name='cone'):
    o = obj(prim('cone', m, seg, r1, r2, h), name); o.location = c; return o


def _latin(s): return all(ord(c) < 0x2E80 for c in s)
EN_K = 0.75
TXT_ROT = (math.pi / 2 - 0.08, 0, 0)


def text(s, loc, size, m, rot=TXT_ROT, align='CENTER', ext=0.02):
    cu = bpy.data.curves.new('t', 'FONT'); cu.body = s; cu.font = FONT_EN if _latin(s) else FONT
    cu.size = size * (EN_K if _latin(s) else 1.0); cu.extrude = ext
    cu.align_x = align; cu.align_y = 'CENTER'
    o = link(bpy.data.objects.new('txt', cu)); o.location = loc; o.rotation_euler = rot; o.data.materials.append(m); return o


def fit_w(o, w):
    bpy.context.view_layer.update()
    if w and o.dimensions.x > w: k = w / o.dimensions.x; o.scale = (k, k, k)
    return o


def label(en, jp, loc, size, m, mj=None, k=0.5, w=None, gap=0.95, **kw):
    """English line at loc, a small Japanese line just below it"""
    o = fit_w(text(en, loc, size, m, **kw), w)
    if not jp: return [o]
    o2 = fit_w(text(jp, (loc[0], loc[1] - 0.01, loc[2] - size * gap), size * k, mj or m, **kw), w)
    return [o, o2]


def pulse_alpha(times, NFr=NF, decay=5.0, length=16, amp=1.0):
    a = np.zeros(NFr)
    for t in times:
        f0 = int(round(t * FPS)); m_ = np.arange(max(0, f0), min(NFr, f0 + length))
        a[m_] = np.maximum(a[m_], amp * np.exp(-(m_ - f0) / decay))
    return a


def smooth(Xa, k):
    Y = Xa.copy(); a = 1 - math.exp(-1 / FPS / k)
    for f in range(1, len(Xa)): Y[f] = Y[f - 1] + (Xa[f] - Y[f - 1]) * a
    for f in range(len(Xa) - 2, -1, -1): Y[f] = Y[f + 1] + (Y[f] - Y[f + 1]) * a
    return Y


# ---------------- ball motion ----------------
def seg_at(g, t):
    if g['k'] == 'rest': return np.array(g['p'])
    if g['k'] == 'fly':
        u = t - g['t0']; return np.array(g['p']) + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
    ts = np.array(g['ts']); ps = np.array(g['ps'])
    return np.array([np.interp(t, ts, ps[:, c]) for c in range(3)])


class BallPath:
    def __init__(s, d): s.seg = d['seg']; s.hits = d['hits']; s.team = d['team']
    def at(s, t):
        for g in s.seg:
            if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9: return seg_at(g, t)
        return seg_at(s.seg[0], s.seg[0]['t0']) if t < s.seg[0]['t0'] else seg_at(s.seg[-1], s.seg[-1]['t1'])
BP = {n: BallPath(d) for n, d in LAY['balls'].items()}
POS = {n: np.array([b.at(t) for t in T]) for n, b in BP.items()}
PR, PG = POS['red'], POS['green']

# ---------------- materials ----------------
M_RED = mat('red', (0.8, 0.03, 0.04), 0.18, **{'Metallic': 0.75, 'Coat Weight': 1.0})
M_GRN = mat('green', (0.03, 0.55, 0.12), 0.18, **{'Metallic': 0.75, 'Coat Weight': 1.0})
M_GOLD = mat('gold', (1.0, 0.7, 0.22), 0.25, **{'Metallic': 1.0})
M_SLOT = mat('slot', (0.02, 0.02, 0.02), 0.5)
M_KEY = glowmat('key', gain=9.0, base=0.15, rough=0.15, trans=0.3)
M_BELL = glowmat('bell', gain=7.0, base=0.05, rough=0.25, metal=0.9)
M_STEEL = mat('steel', (0.6, 0.62, 0.68), 0.3, **{'Metallic': 0.9})
M_DARK = mat('dark', (0.04, 0.04, 0.05), 0.6)
M_WOOD = mat('wood', (0.42, 0.24, 0.12), 0.6)
M_SNOW = mat('snow', (0.9, 0.93, 1.0), 0.55, **{'Subsurface Weight': 0.2, 'Coat Weight': 0.15})
M_ICE = mat('ice', (0.55, 0.8, 1.0), 0.08, **{'Transmission Weight': 0.6, 'Emission Color': (0.2, 0.55, 1.0), 'Emission Strength': 0.25})
M_CAVE = mat('cavewall', (0.05, 0.16, 0.35), 0.35, **{'Emission Color': (0.05, 0.25, 0.6), 'Emission Strength': 0.35})
M_PINE = mat('pine', (0.02, 0.16, 0.08), 0.8)
M_TREE = mat('xtree', (0.02, 0.3, 0.1), 0.7)
M_OBJ = objcol('objcol', 0.5)
M_OBJE = objcol('objcol_e', 0.4, emit=2.5)
M_FX = fxmat('fx', 8.0)
M_TXT = emis('txt', (1, 1, 1), 2.2); M_TXT_Y = emis('txty', (1, 0.82, 0.3), 3.0); M_TXT_R = emis('txtr', (1, 0.18, 0.15), 3.5)
M_TXT_G = emis('txtg', (0.25, 1, 0.35), 3.5); M_TXT_C = emis('txtc', (0.6, 0.9, 1.0), 3.0)
M_PANEL = mat('panel', (0.05, 0.04, 0.08), 0.6)

# ---------------- the racers: jingle bells (sphere + slot + top loop), spinning with travel ----------------
BALL = {}
for n, b in BP.items():
    o = obj(prim('sph', M_RED if n == 'red' else M_GRN, 40, 20), n); o.scale = (R, R, R)
    sl = obj(prim('torus', M_SLOT, 1.0, 0.035, 40, 6), 'slot'); sl.parent = o; sl.scale = (1.0, 1.0, 1.0)
    sl2 = obj(prim('torus', M_SLOT, 1.0, 0.035, 40, 6), 'slot2'); sl2.parent = o; sl2.rotation_euler = (math.pi / 2, 0, 0); sl2.scale = (0.999, 0.999, 0.999)
    lp = obj(prim('torus', M_GOLD, 0.22, 0.06, 20, 6), 'loop'); lp.parent = o; lp.location = (0, 0, 1.05); lp.rotation_euler = (math.pi / 2, 0, 0)
    cap = obj(prim('sph', M_GOLD, 16, 8), 'cap'); cap.parent = o; cap.location = (0, 0, 0.93); cap.scale = (0.25, 0.25, 0.12)
    key_loc(o, POS[n])
    sp = np.zeros(NF)
    for f in range(1, NF): sp[f] = sp[f - 1] + (POS[n][f, 0] - POS[n][f - 1, 0]) / R
    o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, sp)
    BALL[n] = o

# ---------------- keys: one per hit ----------------
def frame_from_n(n):
    n = np.array(n, float); n /= np.linalg.norm(n)
    xa = np.cross([0, 1, 0], n); xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return Matrix(((xa[0], ya[0], n[0]), (xa[1], ya[1], n[1]), (xa[2], ya[2], n[2])))
def pitch_col(p):
    h = ((p - 63) % 12) / 12.0
    cols = [(0.55, 0.85, 1.0), (0.7, 0.75, 1.0), (0.85, 0.7, 1.0), (1.0, 0.75, 0.9), (1.0, 0.85, 0.65), (0.75, 1.0, 0.85)]
    return cols[int(h * 6) % 6]
KEYS = []
KW, KD, KT = 0.16, 0.62, 0.07
for n, b in BP.items():
    team = b.team
    for h in b.hits:
        k = h['kind']
        if k in ('star', 'ornament'): continue
        nrm = np.array(h['n'], float); nrm /= np.linalg.norm(nrm)
        c = np.array(h['p']) - R * nrm
        if k == 'crystal':
            o = obj(prim('cone', M_KEY, 6, 1.0, 0.75, 1.0), 'crystal'); th = 0.09
            ctr = c - nrm * th; scl = (0.13, 0.13, 2 * th)
        elif k == 'snowbank':
            o = obj(prim('cube', M_KEY), 'key'); th = KT / 2; ctr = c - nrm * th; scl = (0.5, KD, KT)
        elif k == 'switch':
            o = obj(prim('cube', M_KEY), 'key'); th = KT / 2; ctr = c - nrm * th; scl = (0.28, KD, KT)
        else:
            o = obj(prim('cube', M_KEY), 'key'); th = KT / 2; ctr = c - nrm * th; scl = (KW, KD, KT)
        o.matrix_world = Matrix.Translation(Vector(ctr)) @ frame_from_n(nrm).to_4x4() @ Matrix.Diagonal((*scl, 1))
        col = pitch_col(h['pitch'])
        o.color = (*col, 0)
        KEYS.append(dict(o=o, c=ctr, n=nrm, t=h['t'], kind=k, team=team, ball=n, contact=c, pitch=h['pitch']))
for K_ in KEYS:
    o = K_['o']; fh = int(round(K_['t'] * FPS)); base = o.location.copy(); nv = Vector(K_['n']); col = tuple(o.color[:3])
    o.color = (*col, 0); o.keyframe_insert('color', index=3, frame=max(0, fh - 1))
    for df, a in ((0, 1.0), (4, 0.45), (18, 0.1)):
        o.color = (*col, a); o.keyframe_insert('color', index=3, frame=fh + df)
    for df, d in ((-1, 0), (1, -0.022), (4, 0.007), (8, 0)):
        o.location = base + nv * d; o.keyframe_insert('location', frame=max(0, fh + df))

# hit rings (FX pool)
NRING = 12
ring_loc = np.zeros((NRING, NF, 3)); ring_scl = np.full((NRING, NF), 1e-3); ring_a = np.zeros((NRING, NF)); ring_c = [(1, 1, 1)] * NRING
for k, K_ in enumerate(sorted(KEYS, key=lambda b: b['t'])):
    r = k % NRING; f0 = int(round(K_['t'] * FPS))
    for df in range(14):
        f = f0 + df
        if f >= NF: break
        u = df / 13; ring_loc[r, f] = K_['contact'] + np.array([0, -0.4, -0.1]); ring_scl[r, f] = 0.35 + 1.2 * (1 - (1 - u) ** 2)
        ring_a[r, f] = (1 - u) ** 1.6
    ring_c[r] = (1.0, 0.45, 0.4) if K_['team'] == 'R' else (0.5, 1.0, 0.55)
for r in range(NRING):
    o = obj(prim('torus', M_FX, 0.22, 0.012, 32, 5), 'ring'); o.rotation_euler = (math.pi / 2, 0, 0); o.color = (*ring_c[r], 0)
    for f in range(1, NF):
        if ring_a[r, f] == 0: ring_loc[r, f] = ring_loc[r, f - 1]
    key_loc(o, ring_loc[r])
    for k in range(3): key_series(o, 'scale', k, ring_scl[r])
    key_series(o, 'color', 3, ring_a[r], 1e-3)

# ---------------- lanes: two rails under each lane's keys, posts, string lights ----------------
def lane_pts(name, t0=-1, t1=1e9, kinds=('key', 'switch', 'snowbank')):
    pts = [(h['p'][0], h['p'][2] - R - KT) for h in BP[name].hits if t0 <= h['t'] <= t1 and h['kind'] in kinds]
    pts.sort(); return pts
LANE_SEGS = []
M_RAIL_R = mat('railr', (0.75, 0.08, 0.06), 0.35, **{'Metallic': 0.6}); M_RAIL_G = mat('railg', (0.06, 0.5, 0.12), 0.35, **{'Metallic': 0.6})
M_BULB_R = emis('bulbr', (1.0, 0.25, 0.15), 6.0); M_BULB_G = emis('bulbg', (0.25, 1.0, 0.3), 6.0); M_BULB_W = emis('bulbw', (1.0, 0.85, 0.5), 6.0)
def rails(pts, y, mr, mb, name):
    if len(pts) < 2: return None
    xs = np.array([p[0] for p in pts]); zs = np.array([p[1] for p in pts])
    X = np.arange(xs[0] - 0.3, xs[-1] + 0.3, 0.2)
    Zm = np.array([zs[np.abs(xs - x) < 0.4].min() if (np.abs(xs - x) < 0.4).any() else np.interp(x, xs, zs) for x in X])
    for yy in (y - KD / 2 + 0.04, y + KD / 2 - 0.04):
        bm = bmesh.new(); vs = []
        for x, z in zip(X, Zm):
            vs.append([bm.verts.new((x, yy - 0.03, z)), bm.verts.new((x, yy + 0.03, z)), bm.verts.new((x, yy + 0.03, z - 0.08)), bm.verts.new((x, yy - 0.03, z - 0.08))])
        for a, b in zip(vs, vs[1:]):
            for k in range(4): bm.faces.new((a[k], b[k], b[(k + 1) % 4], a[(k + 1) % 4]))
        me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new(name, me)); o.data.materials.append(mr)
    for x, z in list(zip(X, Zm))[::3]:
        sphere((x, y - KD / 2 - 0.02, z - 0.02), 0.035, mb, (8, 4))
    LANE_SEGS.append((X, Zm, y))
    return X, Zm
t_sw = EV['switch']['t']; t_c2 = SECS['C2']
LR = rails(lane_pts('red'), LY['R'], M_RAIL_R, M_BULB_R, 'laneR')
LG1 = rails(lane_pts('green', t1=t_sw + 0.01), LY['G'], M_RAIL_G, M_BULB_G, 'laneG1')
LG2 = rails(lane_pts('green', t0=t_c2 - 0.01), LY['G'], M_RAIL_G, M_BULB_G, 'laneG2')

# ---------------- terrain: snow slope under the lanes, a ravine in front of the ice cave ----------------
allx = np.concatenate([LR[0]]); allz = np.concatenate([LR[1]])
def surf_z(x): return float(np.interp(x, allx, allz)) - 1.25
CAVE = EV['cave']; cave_hits = [h for h in BP['green'].hits if h['kind'] == 'crystal']
CX0 = cave_hits[0]['p'][0] - 1.5; CXC = EV['elevator']['p0'][0] - 0.9; CX1 = EV['elevator']['p0'][0] + 3.8   # cave | open pit
CZ_FLOOR = min(h['p'][2] for h in cave_hits) - R - 0.5
FX0, FX1 = -14.0, EV['tree']['x'] + 16
XG = np.arange(FX0, FX1, 1.0); YG = np.array([-14.0, -9.0, -6.0, -3.5, -2.0, -1.25, -1.24, 1.3, 2.5, 5.0, 9.0, 14.0, 22.0])
def terr_z(x, y):
    z = surf_z(x)
    if y > 1.3: z += 0.12 * (y - 1.3) + 0.3 * math.sin(0.4 * x + y)        # gently rising behind the course
    if CX0 - 3 < x < CX1 + 3 and y < -1.24:                                  # ravine in front of the ice cave (cut-away)
        u = min(1.0, (x - (CX0 - 3)) / 3.0, ((CX1 + 3) - x) / 3.0)
        z = (1 - u) * z + u * (CZ_FLOOR - 1.0)
    elif CXC - 0.5 < x < CX1 + 3 and y <= 2.5:                               # open pit: the ski lift rises out of it
        u = min(1.0, (x - (CXC - 0.5)) / 0.5, ((CX1 + 3) - x) / 3.0)
        z = (1 - u) * z + u * (CZ_FLOOR - 1.0)
    return z
bm = bmesh.new(); V_ = [[bm.verts.new((x, y, terr_z(x, y))) for y in YG] for x in XG]
for i in range(len(XG) - 1):
    for j in range(len(YG) - 1):
        if YG[j] == -1.25 and CX0 <= XG[i] and XG[i + 1] <= CXC: continue      # the cave mouth stays open
        bm.faces.new((V_[i][j], V_[i + 1][j], V_[i + 1][j + 1], V_[i][j + 1]))
me = bpy.data.meshes.new('terrain'); bm.to_mesh(me); bm.free()
for p in me.polygons: p.use_smooth = True
TER = link(bpy.data.objects.new('terrain', me)); TER.data.materials.append(M_SNOW)
# the ice cave: ceiling slab front face, back wall, floor, icicles
xs_c = np.arange(CX0, CXC + 0.01, 0.5)
xs_w = np.arange(CX0, CX1 + 0.01, 0.5)
bm = bmesh.new(); vs = []
for x in xs_c: vs.append((bm.verts.new((x, -1.25, surf_z(x))), bm.verts.new((x, -1.25, surf_z(x) - 0.55))))
for a, b in zip(vs, vs[1:]): bm.faces.new((a[0], b[0], b[1], a[1]))
me = bpy.data.meshes.new('cut'); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new('cavecut', me)); o.data.materials.append(M_ICE)
bm = bmesh.new(); vs = []
for x in xs_w: vs.append((bm.verts.new((x, 2.6, surf_z(x) - 0.2)), bm.verts.new((x, 2.6, CZ_FLOOR - 1.2))))
for a, b in zip(vs, vs[1:]): bm.faces.new((a[0], b[0], b[1], a[1]))
me = bpy.data.meshes.new('wall'); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new('cavewall', me)); o.data.materials.append(M_CAVE)
bm = bmesh.new(); vs = []
for x in xs_c: vs.append((bm.verts.new((x, -1.25, surf_z(x) - 0.55)), bm.verts.new((x, 2.6, surf_z(x) - 0.55))))
for a, b in zip(vs, vs[1:]): bm.faces.new((a[0], b[0], b[1], a[1]))
me = bpy.data.meshes.new('ceil'); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new('ceil', me)); o.data.materials.append(M_CAVE)
for h in cave_hits:                           # each crystal grows out of an ice pillar
    c = np.array(h['p']) - R * np.array(h['n'])
    cone((c[0], c[1], (c[2] - 0.18 + CZ_FLOOR - 0.8) / 2), 0.1, 0.1, max(0.05, c[2] - 0.18 - CZ_FLOOR + 0.8), M_ICE, 6)
for x in np.arange(CX0 + 0.3, CXC, 0.55):
    L_ = rng.uniform(0.2, 0.45); zt = surf_z(x) - 0.55
    o = cone((x + rng.uniform(-0.2, 0.2), rng.uniform(-0.9, 1.2), zt - L_ / 2), 0.07, 0.0, L_, M_ICE, 6); o.rotation_euler = (math.pi, 0, 0)
box(((CX0 + CX1) / 2, 0.68, CZ_FLOOR - 1.1), (CX1 - CX0, 3.86, 0.3), M_CAVE, name='cavefloor')
cl = bpy.data.lights.new('cavelight', 'AREA'); cl.size = 6; cl.color = (0.3, 0.6, 1.0); cl.energy = 900
co_ = link(bpy.data.objects.new('cavelight', cl)); co_.location = ((CX0 + CX1) / 2, -2.5, CZ_FLOOR + 1.6); co_.rotation_euler = (math.radians(80), 0, 0)

# posts from the lanes down to the snow
for X, Zm, y in LANE_SEGS:
    for x, z in list(zip(X, Zm))[::10]:
        zb = terr_z(x, y) if not (CX0 - 1 < x < CXC) else surf_z(x)
        if z - 0.1 > zb: rod((x, y, z - 0.08), (x, y, zb - 0.2), 0.035, M_WOOD, 6)

# ---------------- pine trees with snow, distant mountains, sky, moon, falling snow ----------------
pine_me = []
for k in range(3):
    bm = bmesh.new()
    for lvl in range(4):
        r1 = 0.9 - 0.18 * lvl; zc = 0.6 + 0.55 * lvl
        bmesh.ops.create_cone(bm, cap_ends=True, segments=10, radius1=r1, radius2=0.05, depth=0.9, matrix=Matrix.Translation((0, 0, zc)))
    me = bpy.data.meshes.new('pine'); bm.to_mesh(me); bm.free(); me.materials.append(M_PINE); pine_me.append(me)
snowcap = prim('cone', M_SNOW, 10, 0.5, 0.02, 0.5)
for x in np.arange(FX0, FX1, 2.3):
    for y0 in (rng.uniform(3.2, 5.5), rng.uniform(7, 11)):
        xx = x + rng.uniform(-0.8, 0.8); s = rng.uniform(0.8, 1.6)
        o = obj(pine_me[rng.integers(3)], 'pine'); o.location = (xx, y0, terr_z(xx, y0) - 0.1); o.scale = (s, s, s * rng.uniform(1.0, 1.4))
        o2 = obj(snowcap, 'cap'); o2.location = (xx, y0, terr_z(xx, y0) - 0.1 + s * 2.35 * o.scale[2] / s); o2.scale = (s, s, s)
SKY = bpy.data.materials.new('sky'); SKY.use_nodes = True; nt = SKY.node_tree; nt.nodes.clear()
tc = nt.nodes.new('ShaderNodeTexCoord'); sepz = nt.nodes.new('ShaderNodeSeparateXYZ'); rp = nt.nodes.new('ShaderNodeValToRGB')
rp.color_ramp.elements[0].color = (0.06, 0.05, 0.2, 1); rp.color_ramp.elements[1].color = (0.0, 0.01, 0.05, 1)
e_ = rp.color_ramp.elements.new(0.35); e_.color = (0.02, 0.04, 0.14, 1)
mp = nt.nodes.new('ShaderNodeMapRange'); mp.inputs['From Min'].default_value = -0.5; mp.inputs['From Max'].default_value = 0.5
st = nt.nodes.new('ShaderNodeTexVoronoi'); st.inputs['Scale'].default_value = 140.0
try: st.feature = 'F1'
except Exception: pass
stm = nt.nodes.new('ShaderNodeMapRange'); stm.inputs['From Min'].default_value = 0.0; stm.inputs['From Max'].default_value = 0.035
stm.inputs['To Min'].default_value = 6.0; stm.inputs['To Max'].default_value = 0.0; stm.clamp = True
add = nt.nodes.new('ShaderNodeMath'); add.operation = 'ADD'; add.inputs[1].default_value = 1.0
mixc = nt.nodes.new('ShaderNodeMixRGB'); mixc.blend_type = 'ADD'
em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Strength'].default_value = 1.0; out = nt.nodes.new('ShaderNodeOutputMaterial')
nt.links.new(tc.outputs['Generated'], sepz.inputs[0]); nt.links.new(sepz.outputs['Z'], rp.inputs['Fac'])
nt.links.new(tc.outputs['Generated'], st.inputs['Vector']); nt.links.new(st.outputs['Distance'], stm.inputs['Value'])
nt.links.new(rp.outputs['Color'], mixc.inputs[1]); nt.links.new(stm.outputs[0], mixc.inputs['Fac']); mixc.inputs[2].default_value = (0.9, 0.9, 1.0, 1)
nt.links.new(mixc.outputs[0], em.inputs['Color']); nt.links.new(em.outputs[0], out.inputs[0])
zmid = 0.5 * (allz[0] + allz[-1])
sky = obj(prim('plane', SKY), 'sky'); sky.location = ((FX0 + FX1) / 2, 60, zmid); sky.rotation_euler = (math.pi / 2, 0, 0)
sky.scale = (FX1 - FX0 + 160, 140, 1)
moon = sphere((40.0, 55, allz[0] + 14), 3.2, emis('moon', (1.0, 0.95, 0.8), 6.0), (32, 16))
M_MTN = mat('mtn', (0.25, 0.3, 0.45), 0.8)
for k, x in enumerate(np.arange(FX0 - 20, FX1 + 30, 17.0)):
    h = rng.uniform(14, 26); z0 = float(np.interp(x, allx, allz)) - 12
    o = cone((x, rng.uniform(38, 50), z0 + h / 2), rng.uniform(13, 20), 0.5, h, M_MTN, 7)
    cone((o.location.x, o.location.y - 0.6, z0 + h * 0.8), o.data.vertices[0].co.length * 0.0 + h * 0.25, 0.3, h * 0.4, M_SNOW, 7)
# falling snow: two stacked sheets of flakes on the camera, sliding down and wrapping
M_FLAKE = emis('flake', (0.9, 0.95, 1.0), 1.2)

# ---------------- accompaniment devices ----------------
BASS_T = [b[0] for b in E['bass']]; STAB_T = [s[0] for s in E['stabs']]
# master swings: up between beats, down on the hit
def swing(times, amp=0.6):
    up = np.zeros(NF); tt = np.array(times)
    for f in range(NF):
        t = T[f]; k = np.searchsorted(tt, t); nxt = tt[k] if k < len(tt) else 1e9; prv = tt[k - 1] if k > 0 else -1e9
        if nxt - prv < 0.8:
            u = (t - prv) / (nxt - prv); up[f] = amp * math.sin(math.pi * u) ** 0.8
        elif 0 < nxt - t < 0.25: up[f] = amp * math.sin(math.pi * 0.5 * (1 - (nxt - t) / 0.25))
    return up
MAL = link(bpy.data.objects.new('mallet_master', None)); MAL.rotation_mode = 'XYZ'; key_series(MAL, 'rotation_euler', 1, -swing(BASS_T, 0.7))
HAM = link(bpy.data.objects.new('hammer_master', None)); HAM.rotation_mode = 'XYZ'; key_series(HAM, 'rotation_euler', 1, swing(STAB_T, 0.55))
skin_a = pulse_alpha(BASS_T, decay=2.0, length=7); bell_a = pulse_alpha(STAB_T, decay=2.0, length=6)
M_SKIN = mat('skin', (0.95, 0.9, 0.8), 0.5, **{'Emission Color': (1.0, 0.8, 0.5)}); key_socket(M_SKIN.node_tree.nodes['Principled BSDF'].inputs['Emission Strength'], 4 * skin_a)
M_SBELL = mat('sbell', (1.0, 0.72, 0.2), 0.2, **{'Metallic': 1.0, 'Emission Color': (1.0, 0.75, 0.3)})
key_socket(M_SBELL.node_tree.nodes['Principled BSDF'].inputs['Emission Strength'], 0.2 + 6 * bell_a)
M_COPPER = mat('copper', (0.75, 0.35, 0.18), 0.25, **{'Metallic': 1.0})
M_CARROT = mat('carrot', (1.0, 0.4, 0.05), 0.5)
M_CANE = mat('cane', (0.95, 0.95, 0.95), 0.3)
M_SCARF_R = mat('scarfr', (0.85, 0.06, 0.06), 0.7); M_SCARF_G = mat('scarfg', (0.06, 0.6, 0.15), 0.7)
snowman_xs = [x for x in np.arange(4.0, EV['tree']['x'] - 6, 8.5) if not (CX0 - 3 < x < CX1 + 3)]
for k, x in enumerate(snowman_xs):
    y = -2.3; z = terr_z(x, y) - 0.05
    sphere((x, y, z + 0.38), 0.42, M_SNOW, (20, 10)); sphere((x, y, z + 0.98), 0.3, M_SNOW, (20, 10)); sphere((x, y, z + 1.42), 0.21, M_SNOW, (20, 10))
    o = cone((x + 0.0, y - 0.3, z + 1.44), 0.04, 0.0, 0.22, M_CARROT, 8); o.rotation_euler = (math.pi / 2, 0, 0)
    for dx in (-0.08, 0.08): sphere((x + dx, y - 0.18, z + 1.5), 0.025, M_DARK, (8, 4))
    rod((x, y, z + 1.6), (x, y, z + 1.62), 0.2, M_DARK, 12); rod((x, y, z + 1.62), (x, y, z + 1.85), 0.13, M_DARK, 12)
    sc_ = obj(prim('torus', M_SCARF_R if k % 2 == 0 else M_SCARF_G, 0.22, 0.05, 20, 6), 'scarf'); sc_.location = (x, y, z + 1.24)
    # timpani in front-right of the snowman
    dx_ = 0.75
    o = sphere((x + dx_, y - 0.2, z + 0.72), 0.36, M_COPPER, (24, 12)); o.scale = (0.36, 0.36, 0.28)
    rod((x + dx_, y - 0.2, z + 0.72), (x + dx_, y - 0.2, z + 0.74), 0.37, M_SKIN, 24)
    for a in range(3):
        ang = a * 2.1; rod((x + dx_ + 0.3 * math.cos(ang), y - 0.2 + 0.3 * math.sin(ang), z + 0.6), (x + dx_ + 0.3 * math.cos(ang), y - 0.2 + 0.3 * math.sin(ang), z + 0.0), 0.015, M_STEEL, 6)
    piv = link(bpy.data.objects.new('piv', None)); piv.location = (x + 0.25, y - 0.2, z + 1.0)
    m_ = rod((0, 0, 0), (0.55, 0, -0.2), 0.02, M_WOOD, 6); m_.parent = piv
    hd = sphere((0.57, 0, -0.21), 0.06, M_SNOW); hd.parent = piv
    cr_ = piv.constraints.new('COPY_ROTATION'); cr_.target = MAL
bell_xs = [x for x in np.arange(8.3, EV['tree']['x'] - 6, 8.5)]
for k, x in enumerate(bell_xs):
    y = 1.45; zb = terr_z(x, y)
    ztop = float(np.interp(x, allx, allz)) + DZL + 1.6
    if CX0 - 1 < x < CX1 + 1: zb = surf_z(x)
    rod((x, y, zb - 0.2), (x, y, ztop + 0.25), 0.05, M_CANE, 8)
    rod((x - 0.6, y, ztop + 0.25), (x + 0.1, y, ztop + 0.25), 0.04, M_CANE, 8)
    for j, dx in enumerate((-0.5, -0.32, -0.14)):
        rod((x + dx, y, ztop + 0.25), (x + dx, y, ztop + 0.02), 0.008, M_STEEL, 5)
        sphere((x + dx, y, ztop - 0.06), 0.09, M_SBELL, (16, 8))
    piv = link(bpy.data.objects.new('hpiv', None)); piv.location = (x + 0.05, y - 0.15, ztop - 0.35)
    hm = rod((0, 0, 0), (-0.38, 0, 0.32), 0.022, M_CANE, 6); hm.parent = piv
    hh = rod((-0.4, -0.05, 0.3), (-0.4, 0.05, 0.3), 0.05, M_SCARF_R, 10); hh.parent = piv
    cr_ = piv.constraints.new('COPY_ROTATION'); cr_.target = HAM

# ---------------- start: summit gate, start ramps, countdown lamps, banner ----------------
T_GATE = EV['gate']['t']
for team in ('R', 'G'):
    st_ = EV['start'][team]; top = np.array(st_['top']); lip = np.array(st_['lip']); u = np.array(st_['u'])
    nrm = np.array([u[2] * -1, 0, u[0]]) if False else np.array([math.sin(EV['gate']['alpha']), 0, math.cos(EV['gate']['alpha'])])
    a = top - u * 0.4 - nrm * R; b = lip - nrm * R
    for dy in (-0.2, 0.2): rod((a[0], a[1] + dy, a[2]), (b[0], b[1] + dy, b[2]), 0.03, M_RAIL_R if team == 'R' else M_RAIL_G, 8)
    rod((a[0], a[1], a[2] - 0.05), (a[0], a[1], terr_z(a[0], a[1]) - 0.2), 0.05, M_WOOD, 8)
    rod((b[0], b[1], b[2] - 0.05), (b[0], b[1], terr_z(b[0], b[1]) - 0.2), 0.05, M_WOOD, 8)
    # gate: a flap in front of the ball that swings down
    hp = link(bpy.data.objects.new('gate', None)); hp.location = tuple(top + u * (R + 0.04) - nrm * R)
    fl = box((0, 0, 0.18), (0.04, 0.5, 0.36), M_OBJ, name='flap'); fl.parent = hp; fl.color = ((0.9, 0.1, 0.1, 1) if team == 'R' else (0.1, 0.8, 0.2, 1))
    ang = np.zeros(NF)
    for f in range(NF):
        uu = (T[f] - (T_GATE - 0.06)) / 0.12
        if uu > 0: ang[f] = min(1.0, uu) * 1.6
    hp.rotation_mode = 'XYZ'; key_series(hp, 'rotation_euler', 1, ang)
st_g = np.array(EV['start']['G']['top']); st_r = np.array(EV['start']['R']['top'])
# countdown lamps (3 red, then green on the gate)
lamp_t = [T_GATE - 0.75, T_GATE - 0.5, T_GATE - 0.25, T_GATE]
for k, tl in enumerate(lamp_t):
    col = (1, 0.15, 0.1) if k < 3 else (0.2, 1, 0.3)
    o = sphere((st_g[0] - 0.9 + 0.42 * k, 1.1, st_g[2] + 1.05), 0.15, M_OBJE, (16, 8)); o.color = (*col, 1)
    sc_v = np.where(T >= tl, 1.0, 0.25)
    for c_ in range(3): key_series(o, 'color', c_, np.array([col[c_]] * NF) * sc_v)
box((st_g[0] - 0.27, 1.25, st_g[2] + 1.05), (1.75, 0.08, 0.45), M_PANEL, name='lampboard')
# start arch + banner (keywords)
xa = st_g[0] - 1.4
for yy in (-1.3, 1.3):
    rod((xa, yy, terr_z(xa, yy) - 0.3), (xa, yy, st_g[2] + 3.2), 0.07, M_CANE, 10)
    rod((xa + 3.4, yy, terr_z(xa + 3.4, yy) - 0.3), (xa + 3.4, yy, st_g[2] + 3.2), 0.07, M_CANE, 10)
box((xa + 1.7, 1.35, st_g[2] + 2.6), (4.2, 0.1, 1.55), M_PANEL, name='banner')
label('JINGLE BELLS', 'ジングルベル', (xa + 1.7, 1.25, st_g[2] + 2.95), 0.62, M_TXT_Y, M_TXT, k=0.6, w=3.9, gap=0.75)
label('James Lord Pierpont · 1857', 'ピアポント作曲（1857年）', (xa + 1.7, 1.25, st_g[2] + 2.06), 0.3, M_TXT, M_TXT, k=0.95, w=3.9, gap=1.1)
label('WHO WINS THE SLEIGH RACE?', 'ソリ玉レース、勝つのはどっち？', (xa + 1.7, 1.0, st_g[2] + 4.35), 0.6, M_TXT, M_TXT_Y, k=0.7, w=5.0, gap=0.9)
text('RED', (st_r[0] - 0.75, LY['R'] - 0.4, st_r[2] + 0.1), 0.3, M_TXT_R)
text('GREEN', (st_g[0] - 0.85, LY['G'] - 0.6, st_g[2] + 0.1), 0.3, M_TXT_G)

# ---------------- billboards along the course (keywords in the first 30 s) ----------------
BOARDS = [(4.0, 'Christmas marble machine', 'クリスマスのビー玉からくり'),
          (8.0, 'Red vs Green · Marble race', '赤 vs 緑 ビー玉レース'),
          (12.5, '"One Horse Open Sleigh"', '原題「ワン・ホース・オープン・スレイ」'),
          (17.5, 'Satisfying Christmas ASMR', '癒やしのクリスマスASMR'),
          (22.5, 'Classic Xmas song · 1857', '定番クリスマスソング'),
          (27.5, 'Jingle Bells – Pierpont', 'ジングルベル（ピアポント）')]
def board(x_center, en, jp, ztop, w=3.6, h=1.25, y=2.2):
    zb = terr_z(x_center, y)
    for dx in (-w / 2 + 0.2, w / 2 - 0.2): rod((x_center + dx, y + 0.05, zb - 0.3), (x_center + dx, y + 0.05, ztop - h / 2), 0.05, M_WOOD, 8)
    box((x_center, y + 0.06, ztop), (w, 0.08, h), M_PANEL, name='board')
    rod((x_center - w / 2, y - 0.02, ztop + h / 2), (x_center + w / 2, y - 0.02, ztop + h / 2), 0.03, M_CANE, 8)
    for k in range(9): sphere((x_center - w / 2 + w * (k + 0.5) / 9, y - 0.05, ztop + h / 2 + 0.02), 0.04, [M_BULB_R, M_BULB_G, M_BULB_W][k % 3], (8, 4))
    label(en, jp, (x_center, y - 0.02, ztop + 0.2), 0.42, M_TXT, M_TXT_Y, k=0.75, w=w - 0.3, gap=0.85)
for tb, en, jp in BOARDS:
    x_ = float(np.interp(tb, T, 0.5 * (PR[:, 0] + PG[:, 0]))) + 1.0
    zr = float(np.interp(tb, T, PR[:, 2]))
    board(x_, en, jp, zr + DZL + 2.15)

# ---------------- track switch + cave signs ----------------
sw = EV['switch']; psw = np.array(sw['p'])
label('ICE CAVE ↓', '氷の洞窟', (psw[0] + 0.8, LY['G'] - 0.45, psw[2] + 1.0), 0.32, M_TXT_C, M_TXT, k=0.7, gap=0.85)
# trapdoor: the green lane ends at the switch plate; a hatch in the snow opens as green drops through
hp = link(bpy.data.objects.new('hatch', None)); hp.location = (psw[0] + 0.25, LY['G'], surf_z(psw[0]) + 0.02)
hz = box((0.5, 0, 0), (1.0, 0.8, 0.05), M_WOOD, name='hatch'); hz.parent = hp
ang = np.zeros(NF)
for f in range(NF):
    uu = (T[f] - (sw['t'] - 0.4)) / 0.3
    if uu > 0: ang[f] = -min(1.0, uu) * 1.9
hp.rotation_mode = 'XYZ'; key_series(hp, 'rotation_euler', 1, ang)

# ---------------- chairlift: cable, towers, bull wheel (gear), the bucket ----------------
el = EV['elevator']; p0e = np.array(el['p0']); p1e = np.array(el['p1'])
HANG = 1.0
c0 = p0e + np.array([0, 0, HANG + R]); c1 = p1e + np.array([0, 0, HANG + R])
dcab = (c1 - c0) / np.linalg.norm(c1 - c0)
rod(c0 - dcab * 0.6, c1 + dcab * 0.6, 0.02, M_STEEL, 6)
rod(c0 - dcab * 0.6 + np.array([0, 0.25, 0.3]), c1 + dcab * 0.6 + np.array([0, 0.25, 0.3]), 0.02, M_STEEL, 6)
for cc in (c0, c1):
    rod((cc[0], cc[1] + 0.4, cc[2] + 0.2), (cc[0], cc[1] + 0.4, (CZ_FLOOR - 1.0) if cc is c0 else terr_z(cc[0], 1.3) - 0.3), 0.09, M_STEEL, 10)
GEARS = []
for cc, rr in ((c0, 0.55), (c1, 0.75)):
    gp = link(bpy.data.objects.new('gear', None)); gp.location = (cc[0], cc[1] + 0.3, cc[2] + 0.15); gp.rotation_mode = 'XYZ'
    d = rod((0, -0.05, 0), (0, 0.05, 0), rr, M_GOLD, 32); d.parent = gp
    for k in range(16):
        a = 2 * math.pi * k / 16
        tth = box((rr * math.cos(a), 0, rr * math.sin(a)), (0.12, 0.1, 0.12), M_GOLD, rot=(0, -a, 0)); tth.parent = gp
    for k in range(4):
        a = math.pi * k / 4; sp_ = rod((-rr * math.cos(a), -0.06, -rr * math.sin(a)), (rr * math.cos(a), -0.06, rr * math.sin(a)), 0.03, M_DARK, 6); sp_.parent = gp
    GEARS.append((gp, rr))
BK = np.array([BP['green'].at(t) for t in T])
moving = (T >= el['t0'] - 2) & (T <= el['t1'] + 0.01)
s_cab = np.array([float((BK[f] - p0e) @ dcab) if el['t0'] <= T[f] <= el['t1'] else (0.0 if T[f] < el['t0'] else float((p1e - p0e) @ dcab)) for f in range(NF)])
for gp, rr in GEARS: key_series(gp, 'rotation_euler', 1, s_cab / rr)
bucket = link(bpy.data.objects.new('bucket', None))
BUCK = np.zeros((NF, 3)); TILTB = np.zeros(NF)
for f in range(NF):
    t = T[f]
    if t < el['t0']: BUCK[f] = p0e
    elif t <= el['t1']: BUCK[f] = BK[f]
    else: BUCK[f] = p1e
    if el['t1'] < t: TILTB[f] = -min(1.0, (t - el['t1']) / (el['t_tip'] - el['t1'])) * 0.9 * (1.0 if t < el['t_tip'] + 1.5 else max(0.0, 1 - (t - el['t_tip'] - 1.5)))
key_loc(bucket, BUCK); bucket.rotation_mode = 'XYZ'; key_series(bucket, 'rotation_euler', 1, -TILTB)
cup_o = obj(prim('cone', M_OBJ, 24, 0.3, 0.24, 0.3), 'cup'); cup_o.location = (0, 0, -R + 0.08); cup_o.parent = bucket; cup_o.color = (0.95, 0.75, 0.15, 1)
sol_ = cup_o.modifiers.new('s', 'SOLIDIFY'); sol_.thickness = 0.03
hang = rod((0, 0, 0.0), (0, 0, HANG + R), 0.025, M_STEEL, 6); hang.parent = bucket
grip = box((0, 0, HANG + R), (0.2, 0.12, 0.1), M_DARK); grip.parent = bucket
label('SKI LIFT ↑', 'スキーリフト', (c0[0] + 1.6, LY['G'] - 1.0, c0[2] + 0.9), 0.4, M_TXT_Y, M_TXT, k=0.65, gap=0.85)

# ---------------- the ski jump: in-run profile, tower, lip ----------------
jp_ = EV['jump']
def inrun_profile(L, alpha, beta, rk):
    xs, zs = [], []
    for s in np.linspace(0, L, 60): xs.append(s * math.cos(alpha)); zs.append(-s * math.sin(alpha))
    x1, z1 = xs[-1], zs[-1]; cx, cz = x1 + rk * math.sin(alpha), z1 + rk * math.cos(alpha)
    for a in np.linspace(-alpha, beta, 30)[1:]: xs.append(cx + rk * math.sin(a)); zs.append(cz - rk * math.cos(a))
    return np.array(xs), np.array(zs)
ix, iz = inrun_profile(jp_['L'], jp_['alpha'], jp_['beta'], jp_['rk'])
ix = ix + jp_['x_top']; iz = iz + jp_['z_top'] - R
yj = LY['G']
bm = bmesh.new(); vs = []
for x, z in zip(ix, iz): vs.append([bm.verts.new((x, yj - 0.4, z)), bm.verts.new((x, yj + 0.4, z)), bm.verts.new((x, yj + 0.4, z - 0.12)), bm.verts.new((x, yj - 0.4, z - 0.12))])
for a, b in zip(vs, vs[1:]):
    for k in range(4): bm.faces.new((a[k], b[k], b[(k + 1) % 4], a[(k + 1) % 4]))
me = bpy.data.meshes.new('inrun'); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new('inrun', me)); o.data.materials.append(M_SNOW)
for dy in (-0.42, 0.42):
    for (x0_, z0_), (x1_, z1_) in zip(list(zip(ix, iz))[:-1:3], list(zip(ix, iz))[3::3]):
        rod((x0_, yj + dy, z0_ + 0.12), (x1_, yj + dy, z1_ + 0.12), 0.025, M_RAIL_G, 6)
for x, z in list(zip(ix, iz))[::6]:
    zb = terr_z(x, yj) if not (CX0 < x < CX1) else CZ_FLOOR - 1.0
    for dy in (-0.45, 0.45): rod((x, yj + dy, z - 0.1), (x, yj + dy, zb - 0.3), 0.05, M_WOOD, 6)
for k in range(0, len(ix) - 6, 6): sphere((ix[k], yj - 0.46, iz[k] + 0.15), 0.04, M_BULB_W, (8, 4))
label('SKI JUMP', 'ジャンプ台', (ix[len(ix) // 3] + 0.3, yj - 0.6, iz[len(ix) // 3] + 1.2), 0.6, M_TXT_Y, M_TXT, k=0.65, gap=0.85)
# snow bank at the landing
pl = np.array(jp_['p_land'])
o = sphere((pl[0] + 0.1, LY['G'], pl[2] - R - 0.2), 0.85, M_SNOW, (24, 12)); o.scale = (1.0, 0.5, 0.18)
box((pl[0] + 0.1, LY['G'], pl[2] - R - 0.25), (1.6, 0.8, 0.12), M_WOOD, name='bankbase')
for dx in (-0.6, 0.8): rod((pl[0] + dx, LY['G'], pl[2] - R - 0.3), (pl[0] + dx, LY['G'], terr_z(pl[0] + dx, LY['G']) - 0.3), 0.05, M_WOOD, 8)
label('SNOW BANK', '雪のクッション', (pl[0] + 0.1, LY['G'] - 0.8, pl[2] + 0.9), 0.3, M_TXT, M_TXT_Y, k=0.7, gap=0.85)

# ---------------- finish gate, tapes ----------------
fin = EV['finish']; t_tape = fin['t']
x_fl = 0.5 * (fin['x_green'] + fin['x_red']) - 0.1                 # the line: green is past it at t_tape, red is not
x_fl = fin['x_green'] - 0.25
zr_f = float(np.interp(t_tape, T, PR[:, 2])); zg_f = float(np.interp(t_tape, T, PG[:, 2]))
M_CHK = bpy.data.materials.new('chk'); M_CHK.use_nodes = True; nt = M_CHK.node_tree
ck = nt.nodes.new('ShaderNodeTexChecker'); ck.inputs['Scale'].default_value = 8.0; ck.inputs['Color1'].default_value = (0.02, 0.02, 0.02, 1); ck.inputs['Color2'].default_value = (0.95, 0.95, 0.95, 1)
nt.links.new(ck.outputs['Color'], nt.nodes['Principled BSDF'].inputs['Base Color'])
o = box((x_fl, 1.2, (zr_f - 2.0 + zg_f + 2.2) / 2), (0.2, 0.2, zg_f + 2.2 - zr_f + 2.0), M_CHK, name='finpost')
box((x_fl, 0.2, zg_f + 2.2), (0.25, 2.2, 0.55), M_CHK, name='finbar')
for zz_ in (zg_f + 0.05, zr_f + 0.05): box((x_fl, 0.0, zz_ - 0.35), (0.06, 2.0, 0.06), M_CHK, name='finline')
label('FINISH', 'ゴール', (x_fl, -0.95, zg_f + 2.9), 0.6, M_TXT, M_TXT_Y, k=0.65, gap=0.8)
M_TAPE = mat('tape', (1.0, 0.85, 0.2), 0.5)
T_BREAK = {}
for name, yy, zz in (('green', LY['G'], zg_f), ('red', LY['R'], zr_f)):
    P_ = POS[name]; tb = None
    for f in range(NF):
        if P_[f, 0] + R >= x_fl and T[f] > SECS['C2']: tb = T[f]; break
    T_BREAK[name] = tb
    for side, (ya, yb) in enumerate(((yy - 0.4, yy), (yy + 0.4, yy))):
        pv = link(bpy.data.objects.new('tp', None)); pv.location = (x_fl, ya, zz + 0.05)
        seg = box(((0, (yb - ya) / 2, 0)), (0.02, abs(yb - ya), 0.07), M_TAPE, name='tape'); seg.parent = pv
        rot = np.zeros(NF)
        for f in range(NF):
            u = (T[f] - tb) / 0.5
            if u > 0: rot[f] = (1 - math.exp(-3 * u)) * 1.3 + 0.1 * math.sin(10 * u) * math.exp(-2 * u)
        pv.rotation_mode = 'XYZ'; key_series(pv, 'rotation_euler', 2, rot * (1 if side == 0 else -1)); key_series(pv, 'rotation_euler', 1, rot * 0.5)
print('tape break', T_BREAK)

# ---------------- the Christmas tree: star, ornaments, light chain ----------------
tr = EV['tree']; pst = np.array(tr['star']); por = np.array(tr['ornament'])
XT, YT = tr['x'], LY['G']
z_top = pst[2] - R - 0.05; z_bot = terr_z(XT, YT) - 0.1; HT = z_top - z_bot
HALF = math.atan(1.0 / 2.0)
for k in range(6):
    u0 = k / 6; u1 = (k + 1.35) / 6
    za = z_bot + HT * u0; zb = min(z_top, z_bot + HT * u1)
    ra = (z_top - za) * 0.5 + 0.1; rb = max(0.05, (z_top - zb) * 0.5 * 0.4)
    cone((XT, YT, (za + zb) / 2), ra, rb, zb - za, M_TREE, 24)
rod((XT, YT, z_bot - 0.4), (XT, YT, z_bot + 0.3), 0.25, M_WOOD, 12)
# star (behind the winner's cup), lights up on the final chord
def star_mesh(r1, r2, th):
    bm = bmesh.new(); top = []; bot = []
    for k in range(10):
        a = math.pi / 2 + k * math.pi / 5; r_ = r1 if k % 2 == 0 else r2
        top.append(bm.verts.new((r_ * math.cos(a), th, r_ * math.sin(a)))); bot.append(bm.verts.new((r_ * math.cos(a), -th, r_ * math.sin(a))))
    c1 = bm.verts.new((0, th * 2.5, 0)); c2 = bm.verts.new((0, -th * 2.5, 0))
    for k in range(10):
        bm.faces.new((top[k], top[(k + 1) % 10], c1)); bm.faces.new((bot[(k + 1) % 10], bot[k], c2))
    me = bpy.data.meshes.new('star'); bm.to_mesh(me); bm.free(); return me
M_STAR = mat('star', (1.0, 0.8, 0.25), 0.2, **{'Metallic': 0.8, 'Emission Color': (1.0, 0.8, 0.3)})
star_a = np.where(T >= FINAL, 1.0, 0.0) * (8 + 4 * np.exp(-np.maximum(0, T - FINAL) / 0.3))
key_socket(M_STAR.node_tree.nodes['Principled BSDF'].inputs['Emission Strength'], star_a + 0.05)
so = obj(star_mesh(0.62, 0.26, 0.05), 'star'); so.data.materials.append(M_STAR); so.location = (XT, YT + 0.32, pst[2] + 0.05)
cupS = obj(prim('cone', M_GOLD, 24, 0.27, 0.18, 0.18), 'starcup'); cupS.location = (pst[0], pst[1], pst[2] - R + 0.04)
sol_ = cupS.modifiers.new('s', 'SOLIDIFY'); sol_.thickness = 0.02
cupO = obj(prim('cone', M_GOLD, 24, 0.27, 0.16, 0.16), 'orncup'); cupO.location = (por[0], por[1], por[2] - R + 0.03)
rod((por[0], por[1] + 0.05, por[2] - R - 0.05), (XT, YT, por[2] - R - 0.05), 0.02, M_GOLD, 6)
sol_ = cupO.modifiers.new('s', 'SOLIDIFY'); sol_.thickness = 0.02
# ornaments (static) + light chain (spiral from the top down, lit one after another after the final chord)
OC = [(0.9, 0.1, 0.1), (1.0, 0.8, 0.2), (0.2, 0.4, 1.0), (0.9, 0.9, 0.95), (0.8, 0.2, 0.7)]
for k in range(26):
    u = rng.uniform(0.12, 0.92); zz = z_top - HT * u; rr = HT * u * 0.5 * 0.9; a = rng.uniform(math.pi * 0.9, math.pi * 2.1)
    o = sphere((XT + rr * math.cos(a), YT + rr * math.sin(a), zz), 0.11, M_OBJ, (16, 8)); o.color = (*OC[k % 5], 1)
NL = 70; M_LIGHT = objcol('tlight', 0.3, emit=0.0)
nt = M_LIGHT.node_tree; bsdf = nt.nodes['Principled BSDF']; oi = [n for n in nt.nodes if n.type == 'OBJECT_INFO'][0]
mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = 12.0
nt.links.new(oi.outputs['Alpha'], mul.inputs[0]); nt.links.new(mul.outputs[0], bsdf.inputs['Emission Strength']); nt.links.new(oi.outputs['Color'], bsdf.inputs['Emission Color'])
LC = [(1, 0.3, 0.2), (1, 0.85, 0.3), (0.3, 1, 0.4), (0.4, 0.6, 1)]
for k in range(NL):
    u = 0.06 + 0.9 * k / NL; zz = z_top - HT * u; rr = HT * u * 0.5 + 0.03; a = -math.pi / 2 + k * 0.9
    o = sphere((XT + rr * math.cos(a), YT + rr * math.sin(a), zz), 0.055, M_LIGHT, (8, 4))
    tl = FINAL + 0.08 + 0.028 * k
    al = np.where(T >= tl, 1.0, 0.0) * (0.75 + 0.25 * np.sin(6 * T + k)) + pulse_alpha([tl], decay=4, length=14, amp=1.5)
    o.color = (*LC[k % 4], 0); key_series(o, 'color', 3, al, 2e-3)
# winner banner + sparkle burst
ban = link(bpy.data.objects.new('banner', None)); ban.location = (XT, YT - 1.2, pst[2] + 2.0)
bt = label('GREEN WINS!', '緑チームの勝ち！', (0, 0, 0), 0.72, M_TXT_G, M_TXT, k=0.6, gap=0.75, rot=(math.pi / 2 - 0.08, 0, 0))
mc = label('Merry Christmas', '', (0, 0, -1.1), 0.36, M_TXT_Y, None, rot=(math.pi / 2 - 0.08, 0, 0))
for o in bt + mc: o.parent = ban
bsc = np.full(NF, 0.001)
for f in range(NF):
    u = (T[f] - FINAL - 0.25) / 0.35
    if u > 0: bsc[f] = min(1.0, u) + 0.12 * math.sin(9 * u) * math.exp(-2 * u)
for c_ in range(3): key_series(ban, 'scale', c_, bsc)
for k in range(70):
    sp_ = obj(prim('ico', M_FX, 1), 'spark'); sp_.color = (*[(1, 0.85, 0.3), (1, 1, 1), (0.4, 1, 0.5), (1, 0.4, 0.3)][k % 4], 0)
    a = rng.uniform(0, 2 * math.pi); vv = rng.uniform(1.0, 3.2); vz = rng.uniform(0.5, 3.0)
    P_ = np.zeros((NF, 3)); A_ = np.zeros(NF)
    for f in range(NF):
        s_ = T[f] - FINAL
        if s_ < 0: P_[f] = pst; continue
        P_[f] = pst + np.array([vv * math.cos(a) * s_, -0.4 * s_, vz * s_ - 0.5 * 2.0 * s_ * s_]) * (1 - 0.15 * s_)
        A_[f] = max(0.0, 1 - s_ / 2.6)
    key_loc(sp_, P_); key_series(sp_, 'color', 3, A_, 1e-3)
    for c_ in range(3): key_series(sp_, 'scale', c_, np.where(T >= FINAL, 0.035, 0.001))

# ---------------- world & lights ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bgn = w.node_tree.nodes['Background']; bgn.inputs['Color'].default_value = (0.03, 0.05, 0.12, 1); bgn.inputs['Strength'].default_value = 0.7
moonl = bpy.data.lights.new('moonlight', 'SUN'); moonl.energy = 1.6; moonl.angle = 0.1; moonl.color = (0.7, 0.8, 1.0)
mo = link(bpy.data.objects.new('moonlight', moonl)); mo.rotation_euler = (math.radians(55), math.radians(-20), math.radians(-25))
fil = bpy.data.lights.new('fill', 'SUN'); fil.energy = 0.6; fil.color = (1.0, 0.75, 0.55)
fo = link(bpy.data.objects.new('fill', fil)); fo.rotation_euler = (math.radians(75), math.radians(35), math.radians(160))

# ---------------- camera ----------------
el_t0, el_t1 = el['t0'], el['t1']; t_lip = jp_['t_lip']
FOC = np.zeros((NF, 3)); WID = np.zeros(NF)
pst_c = np.array([XT, 0, z_top - HT * 0.3])
for f in range(NF):
    t = T[f]; r_, g_ = PR[f], PG[f]
    xs = [r_[0], g_[0]]; zs = [r_[2], g_[2]]
    if t < T_GATE + 0.35:
        FOC[f] = (st_g[0] + 0.25, 0, st_g[2] + 0.75); WID[f] = 5.6; continue
    if t > FINAL - 0.5:
        u = min(1.0, (t - (FINAL - 0.5)) / 1.2)
        FOC[f] = (1 - u) * np.array([np.mean(xs) + 0.4, 0, np.mean(zs) + 0.6]) + u * np.array([XT - 0.1, 0, z_top - 0.2]); WID[f] = (1 - u) * 5.5 + u * 5.2
        continue
    if el_t0 - 1.0 < t < SECS['C2'] + 0.4:                           # the lift and the jump: follow green (red when nearby)
        wg = 0.85
        FOC[f] = wg * g_ + (1 - wg) * r_ + np.array([0.8, 0, 0.3]); FOC[f][1] = 0
        WID[f] = 6.2 + 1.8 * min(1.0, max(0.0, (t - el_t1 + 1.5) / 1.5)) * (1.0 if t < t_lip + 0.4 else max(0, 1 - (t - t_lip - 0.4) / 1.2))
        continue
    FOC[f] = (np.mean(xs) + 0.5, 0, np.mean(zs) + 0.55)
    span_x = abs(xs[0] - xs[1]) + 3.2; span_z = abs(zs[0] - zs[1]) + 3.4
    WID[f] = max(span_x, span_z * 9 / 16, 5.2)
WID = np.clip(WID, 4.8, 8.2)
FOC[:, 0] = smooth(FOC[:, 0], 0.35); FOC[:, 2] = smooth(FOC[:, 2], 0.45); WID = smooth(WID, 0.6)
LENS = 32.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 400
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
ELEV = smooth(np.where((T > t_sw - 0.3) & (T < el_t0 + 0.5), 0.0, 0.12), 0.5)        # level camera while green is in the cave
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24
    CP[f] = (FOC[f, 0], -D, FOC[f, 2] + ELEV[f] * D)
    TP[f] = (FOC[f, 0], 0.0, FOC[f, 2])
key_loc(cam, CP); key_loc(tgt, TP)
# follow light (warm), so the racers always read
fl_ = bpy.data.lights.new('follow', 'POINT'); fl_.energy = 900; fl_.color = (1.0, 0.85, 0.7); fl_.shadow_soft_size = 1.0
flo = link(bpy.data.objects.new('follow', fl_)); key_loc(flo, FOC + np.array([-1.0, -3.5, 3.0]))
# falling snow on the camera
FL = []
for sheet in range(2):
    bm = bmesh.new()
    for k in range(160):
        x = rng.uniform(-4, 4); z = rng.uniform(-4, 4); y = rng.uniform(4, 9)
        bmesh.ops.create_icosphere(bm, subdivisions=1, radius=rng.uniform(0.012, 0.03), matrix=Matrix.Translation((x, z, -y)))
    me = bpy.data.meshes.new('snowfl'); bm.to_mesh(me); bm.free(); me.materials.append(M_FLAKE)
    o = link(bpy.data.objects.new('snowfl', me)); o.parent = cam
    zz = 4.0 - ((T * 0.55 + sheet * 4.0) % 8.0)
    key_series(o, 'location', 1, zz, 1e-6); key_series(o, 'location', 0, 0.3 * np.sin(T * 0.4 + sheet), 1e-6)
    for f in range(NF):                                              # constant interpolation at the wrap
        pass
    FL.append(o)

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
        f = int(round(float(tt_) * FPS))
        print('t', tt_, 'FOC', np.round(FOC[f], 2), 'WID', round(WID[f], 2), 'red', np.round(PR[f], 2), 'green', np.round(PG[f], 2))
print('scene objects', len(bpy.data.objects), 'NF', NF)
