"""William Tell galop - Red vs White sports-day relay (紅白リレー), side view of a long downhill race course.
Every melody note = a ball landing on a xylophone bar (lane bar, pit bar, hurdle, podium...). layout.py plans all motion
(constant gravity, rigid bounces that never add speed, mechanisms for the speed-ups); this file builds and animates.
Accompaniment: cheer-squad taiko drums are struck on every bass note, bunting lamps flash on every off-beat chord.
python3 scene.py -- check | test t1 t2 ... | range f0 f1"""
import os, sys, json, math, time
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json')); NOTES = json.load(open(f'{HERE}/notes.json'))
LAY = json.load(open(f'{HERE}/layout.json'))
FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.5) * FPS); T = np.arange(NF) / FPS
G = LAY['G']; R = LAY['R']; EV = LAY['ev']; LY = LAY['lane_y']
TN = np.array([l[0] for l in LEAD]); SECS = dict((k, v) for k, v in E['secs'])
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
sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.5      # centred: blur reaches a hit <= 1/120 s away
try: sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Punchy'
except Exception: pass
sc.frame_start, sc.frame_end = 0, NF - 1
FONT = bpy.data.fonts.load(f'{HERE}/jp_sub.otf')


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
    m = mat(name, (1, 1, 1), rough, **{'Coat Weight': 0.5, 'Metallic': metal}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
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


def objcol(name, rough=0.45, metal=0.0):
    """plain material coloured by object colour (one material for many differently coloured objects)"""
    m = mat(name, (1, 1, 1), rough, **{'Metallic': metal}); nt = m.node_tree
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
    """shared mesh (one per shape + material): creating objects from data is far faster than bpy.ops"""
    key = (kind, m.name if m else None) + tuple(a)
    if key in _MC: return _MC[key]
    bm = bmesh.new()
    if kind == 'cube': bmesh.ops.create_cube(bm, size=1.0)
    elif kind == 'cyl': bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=1.0, radius2=1.0, depth=1.0)
    elif kind == 'cone': bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=a[1], radius2=a[2], depth=a[3])
    elif kind == 'sph': bmesh.ops.create_uvsphere(bm, u_segments=a[0], v_segments=a[1], radius=1.0)
    elif kind == 'plane': bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    me = bpy.data.meshes.new(kind); bm.to_mesh(me); bm.free()
    if kind in ('sph', 'cone'):
        for p in me.polygons: p.use_smooth = True
    if m: me.materials.append(m)
    _MC[key] = me; return me


def obj(me, name='o'): return link(bpy.data.objects.new(name, me))


def box(c, size, m, rot=None, bevel=0.0, name='box'):
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


def text(s, loc, size, m, rot=(math.pi / 2, 0, 0), align='CENTER', ext=0.02):
    cu = bpy.data.curves.new('t', 'FONT'); cu.body = s; cu.font = FONT; cu.size = size; cu.extrude = ext
    cu.align_x = align; cu.align_y = 'CENTER'
    o = link(bpy.data.objects.new('txt', cu)); o.location = loc; o.rotation_euler = rot; o.data.materials.append(m); return o


def pulse_alpha(times, NFr=NF, decay=5.0, length=16, amp=1.0):
    a = np.zeros(NFr)
    for t in times:
        f0 = int(round(t * FPS)); m_ = np.arange(max(0, f0), min(NFr, f0 + length))
        a[m_] = np.maximum(a[m_], amp * np.exp(-(m_ - f0) / decay))
    return a


# ---------------- balls (motion from layout segments) ----------------
class BallPath:
    def __init__(s, d): s.seg = d['seg']; s.hits = d['hits']; s.team = d['team']
    def at(s, t):
        for g in s.seg:
            if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9:
                p = np.array(g['p'])
                if g['k'] == 'rest': return p
                u = t - g['t0']; return p + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
        return np.array(s.seg[0]['p']) if t < s.seg[0]['t0'] else s.at(s.seg[-1]['t1'])
BP = {n: BallPath(d) for n, d in LAY['balls'].items()}
POS = {n: np.array([b.at(t) for t in T]) for n, b in BP.items()}

# active runner of each team over time (camera, scoreboard)
tR, tW = EV['handoff']['R']['t'], EV['handoff']['W']['t']
def run_pos(team, t):
    if team == 'R': return BP['R1' if t < tR else 'R2'].at(t)
    return BP['W1' if t < tW else 'W2'].at(t)
XS = 0.0; XF = (EV['finish']['xW'] + EV['finish']['xR']) / 2

# ---------------- materials ----------------
M_RED = mat('red', (0.85, 0.05, 0.06), 0.15, **{'Coat Weight': 1.0})
M_WHITE = mat('white', (0.92, 0.92, 0.9), 0.12, **{'Coat Weight': 1.0})
M_GOLD = mat('gold', (1.0, 0.68, 0.18), 0.3, **{'Metallic': 0.55})
M_KEY = glowmat('key', gain=7.0, base=0.0, rough=0.35)
M_KEY.node_tree.nodes['Principled BSDF'].inputs['Coat Weight'].default_value = 0.1
M_BELL = glowmat('bell', gain=6.0, base=0.1, rough=0.3, metal=0.5)
M_STEEL = mat('steel', (0.55, 0.56, 0.6), 0.3, **{'Metallic': 0.9})
M_DARK = mat('dark', (0.06, 0.06, 0.07), 0.6)
M_TRACK = mat('track', (0.62, 0.22, 0.14), 0.8)
M_LINE = mat('line', (0.95, 0.95, 0.95), 0.5)
M_WOOD = mat('wood', (0.55, 0.35, 0.18), 0.6)
M_GRASS = mat('grass', (0.18, 0.5, 0.15), 0.9)
M_MUD = mat('mud', (0.22, 0.13, 0.06), 0.9)
M_PITWALL = mat('pitwall', (0.36, 0.26, 0.17), 0.9)
M_BLUE = mat('mat', (0.1, 0.3, 0.85), 0.7)
M_OBJ = objcol('objcol', 0.5)
M_FX = fxmat('fx', 8.0)
M_TXT_W = emis('txtw', (1, 1, 1), 1.3); M_TXT_R = emis('txtr', (1, 0.15, 0.1), 1.6); M_TXT_Y = emis('txty', (1, 0.85, 0.2), 1.6)
M_TXT_RD = mat('txtrd', (0.75, 0.02, 0.03), 0.6)
M_TXT_D = mat('txtd', (0.05, 0.05, 0.08), 0.5)

# balls
BALL = {}
for n, b in BP.items():
    rr = R
    bpy.ops.mesh.primitive_uv_sphere_add(radius=rr, segments=32, ring_count=16); o = bpy.context.object; o.name = n
    for p in o.data.polygons: p.use_smooth = True
    o.data.materials.append({'R': M_RED, 'W': M_WHITE, 'g': M_GOLD}[n[0]])
    # team headband (hachimaki) so the runners read as athletes
    if n != 'gold':
        bpy.ops.mesh.primitive_torus_add(major_radius=rr * 1.0, minor_radius=0.022, major_segments=32, minor_segments=8)
        hb = bpy.context.object; hb.data.materials.append(M_WHITE if n[0] == 'R' else M_RED); hb.parent = o; hb.rotation_euler = (0.35, 0, 0)
    key_loc(o, POS[n])
    # spin with travel
    sp = np.zeros(NF)
    for f in range(1, NF): sp[f] = sp[f - 1] + (POS[n][f, 0] - POS[n][f - 1, 0]) / rr
    o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, sp)
    BALL[n] = o

# ---------------- bars: one per hit ----------------
BAR_W = {'bar': 0.16, 'trap': 0.5, 'pit': 0.18, 'mat': 0.9, 'hurdle': 0.16, 'finish': 0.3, 'step': 0.16, 'podium': 0.5, 'bell': 0.2}
BAR_TH = 0.07
BARS = []          # (obj, centre, frame-normal, t, half thickness, half width, half depth)
def frame_from_n(n):
    n = np.array(n, float); n /= np.linalg.norm(n)
    xa = np.cross([0, 1, 0], n); xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return Matrix(((xa[0], ya[0], n[0]), (xa[1], ya[1], n[1]), (xa[2], ya[2], n[2])))
def pitch_col(p, team):
    h = ((p - 60) % 12) / 12.0
    if team == 'R': return (0.95, 0.12 + 0.3 * h, 0.08 + 0.1 * h)
    if team == 'W': return (0.85 + 0.1 * h, 0.88 + 0.1 * h, 0.95)
    return (1.0, 0.75, 0.3)
PITCH = {i: l[1] for i, l in enumerate(LEAD)}
handled_special = set()
for n, b in BP.items():
    team = n[0]
    for h in b.hits:
        k = h['kind']; t = h['t']; i = h['i']
        if k == 'trap' or (k == 'podium' and h['vout'] == [0, 0, 0] and False): pass
        nrm = np.array(h['n'], float); nrm /= np.linalg.norm(nrm)
        if k == 'podium':                         # podium top is built separately (one block per team); no bar
            continue
        c = np.array(h['p']) - R * nrm
        w = BAR_W.get(k, 0.16); depth = 0.95 if k in ('bar', 'hurdle', 'pit', 'trap', 'mat', 'finish') else 0.5
        if n == 'gold': w, depth = 0.22, 0.5
        ctr = c - nrm * BAR_TH / 2
        mm = M_BELL if n == 'gold' or k == 'finish' else M_KEY
        if n == 'gold':                            # brass bell, mouth down, struck on its crown
            o = obj(prim('cone', mm, 24, 0.1, 0.04, 0.14), 'bell'); ctr = c - nrm * 0.07; th = 0.07; scl = (1, 1, 1)
        else:
            o = obj(prim('cube', mm), 'bar'); th = BAR_TH / 2; scl = (w, depth, BAR_TH)
        o.matrix_world = Matrix.Translation(Vector(ctr)) @ frame_from_n(nrm).to_4x4() @ Matrix.Diagonal((*scl, 1))
        col = (0.3, 0.32, 0.38) if k == 'pit' else ((0.15, 0.4, 0.95) if k == 'mat' else
              ((1.0, 0.82, 0.3) if k == 'finish' else pitch_col(PITCH[i], team if n != 'gold' else 'g')))
        o.color = (*col, 0)
        BARS.append(dict(o=o, c=ctr, n=nrm, t=t, th=th, hw=w / 2, hd=depth / 2, kind=k, team=team, ball=n, i=i, contact=c))

# hit flash + dip
for B_ in BARS:
    o = B_['o']; fh = int(round(B_['t'] * FPS)); base = o.location.copy(); nv = Vector(B_['n'])
    col = tuple(o.color[:3])
    o.color = (*col, 0); o.keyframe_insert('color', index=3, frame=max(0, fh - 1))
    for df, a in ((0, 1.0), (4, 0.5), (16, 0.12)):
        o.color = (*col, a); o.keyframe_insert('color', index=3, frame=fh + df)
    for df, d in ((-1, 0), (1, -0.025), (4, 0.008), (8, 0)):
        o.location = base + nv * d; o.keyframe_insert('location', frame=max(0, fh + df))

# hit rings (FX pool)
NRING = 14
ring_loc = np.zeros((NRING, NF, 3)); ring_scl = np.full((NRING, NF), 1e-3); ring_a = np.zeros((NRING, NF)); ring_c = [(1, 1, 1)] * NRING
for k, B_ in enumerate(sorted(BARS, key=lambda b: b['t'])):
    r = k % NRING; f0 = int(round(B_['t'] * FPS))
    for df in range(14):
        f = f0 + df
        if f >= NF: break
        u = df / 13; ring_loc[r, f] = B_['contact'] + np.array([0, -0.5 if B_['team'] != 'g' else -0.3, 0]); ring_scl[r, f] = 0.3 + 1.3 * (1 - (1 - u) ** 2)
        ring_a[r, f] = (1 - u) ** 1.6
    ring_c[r] = (1.0, 0.45, 0.35) if B_['team'] == 'R' else ((0.8, 0.9, 1.0) if B_['team'] == 'W' else (1, 0.8, 0.3))
for r in range(NRING):
    bpy.ops.mesh.primitive_torus_add(major_radius=0.22, minor_radius=0.014, major_segments=32, minor_segments=5, rotation=(math.pi / 2, 0, 0))
    o = bpy.context.object; o.data.materials.append(M_FX); o.color = (*ring_c[r], 0)
    for f in range(1, NF):
        if ring_a[r, f] == 0: ring_loc[r, f] = ring_loc[r, f - 1]
    key_loc(o, ring_loc[r])
    for k in range(3): key_series(o, 'scale', k, ring_scl[r])
    key_series(o, 'color', 3, ring_a[r], 1e-3)

# ---------------- lane boards (track strips under the bars) ----------------
def lane_points(names, t0=-1, t1=1e9, kinds=None):
    pts = []
    for n in names:
        for h in BP[n].hits:
            if t0 <= h['t'] <= t1 and (kinds is None or h['kind'] in kinds):
                pts.append((h['p'][0], h['p'][2] - R - BAR_TH - 0.1))
    pts.sort(); return pts
def strip(pts, y, width, m, th=0.08, name='strip', lines=True):
    """board following pts (x, ztop) at lane y"""
    if len(pts) < 2: return
    xs = np.array([p[0] for p in pts]); zs = np.array([p[1] for p in pts])
    # smooth-ish: running minimum so the board never pokes above a bar
    bm = bmesh.new(); prev = None
    X = np.arange(xs[0] - 0.4, xs[-1] + 0.4, 0.25); Z = np.interp(X, xs, zs)
    Zm = np.array([zs[(xs > x - 0.45) & (xs < x + 0.45)].min() if ((xs > x - 0.45) & (xs < x + 0.45)).any() else z for x, z in zip(X, Z)])
    vs = []
    for x, z in zip(X, Zm):
        vs.append([bm.verts.new((x, y - width / 2, z)), bm.verts.new((x, y + width / 2, z)), bm.verts.new((x, y + width / 2, z - th)), bm.verts.new((x, y - width / 2, z - th))])
    for a, b in zip(vs, vs[1:]):
        for k in range(4): bm.faces.new((a[k], b[k], b[(k + 1) % 4], a[(k + 1) % 4]))
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new(name, me)); o.data.materials.append(m)
    if lines:
        for yy in (y - width / 2 + 0.03, y + width / 2 - 0.03):
            bm = bmesh.new(); vs = []
            for x, z in zip(X, Zm):
                vs.append([bm.verts.new((x, yy - 0.025, z + 0.004)), bm.verts.new((x, yy + 0.025, z + 0.004))])
            for a, b in zip(vs, vs[1:]): bm.faces.new((a[0], b[0], b[1], a[1]))
            me = bpy.data.meshes.new('ln'); bm.to_mesh(me); bm.free(); lo = link(bpy.data.objects.new('ln', me)); lo.data.materials.append(M_LINE)
    # posts down to the field
    for x, z in list(zip(X, Zm))[::12]:
        rod((x, y, z - th), (x, y, z - 2.6), 0.03, M_STEEL, 6)
    return X, Zm
t_trap = EV['trap']['t']; t_mud = EV['trap']['mud_t']; t_land = EV['catapult']['t_land']
LR = strip(lane_points(['R1', 'R2'], t1=EV['finish']['t'] + 0.01), LY['R'], 1.0, M_TRACK, name='laneR')
LW1 = strip(lane_points(['W1', 'W2'], t1=t_trap + 0.01), LY['W'], 1.0, M_TRACK, name='laneW1')
LW2 = strip(lane_points(['W2'], t0=t_land - 0.01, t1=EV['finish']['t'] + 0.01), LY['W'], 1.0, M_TRACK, name='laneW2')
# coda steps boards (short planks under each step bar)
for n in ('R2', 'W2'):
    for h in BP[n].hits:
        if h['kind'] == 'step':
            c = np.array(h['p']); rod((c[0], c[1], c[2] - R - BAR_TH), (c[0], c[1], c[2] - 2.5), 0.025, M_STEEL, 6)
# field below everything (follows the course down)
allx = np.concatenate([LR[0], LW2[0]]); allz = np.concatenate([LR[1], LW2[1]])
o_ = np.argsort(allx); allx, allz = allx[o_], allz[o_]
def course_z(x): return float(np.interp(x, allx, allz))
_lowx = np.array([h['p'][0] for b in BP.values() for h in b.hits]); _lowz = np.array([h['p'][2] for b in BP.values() for h in b.hits])
_lo = np.argsort(_lowx); _lowx, _lowz = _lowx[_lo], _lowz[_lo]
def low_z(x, w=14.0):
    m = np.abs(_lowx - x) < w
    return float(_lowz[m].min()) if m.any() else course_z(x)
FX0, FX1 = -12.0, allx[-1] + 25
bm = bmesh.new(); vs = []
for x in np.arange(FX0, FX1, 2.0):
    z = low_z(x) - 1.4
    vs.append((bm.verts.new((x, -60, z)), bm.verts.new((x, 14, z))))
for a, b in zip(vs, vs[1:]): bm.faces.new((a[0], b[0], b[1], a[1]))
me = bpy.data.meshes.new('field'); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new('field', me)); o.data.materials.append(M_GRASS)

# ---------------- stadium: stands with crowd, bunting flags, cheer-squad drums ----------------
CROWD = mat('crowd', (1, 1, 1), 0.8); nt = CROWD.node_tree; bsdf = nt.nodes['Principled BSDF']
vor = nt.nodes.new('ShaderNodeTexVoronoi'); vor.inputs['Scale'].default_value = 6.0
tco = nt.nodes.new('ShaderNodeNewGeometry'); nt.links.new(tco.outputs['Position'], vor.inputs['Vector'])
ramp = nt.nodes.new('ShaderNodeValToRGB'); cr = ramp.color_ramp
cols = [(0.55, 0.2, 0.2), (0.6, 0.6, 0.62), (0.2, 0.25, 0.45), (0.55, 0.48, 0.25), (0.12, 0.12, 0.14), (0.5, 0.35, 0.4)]
cr.elements[0].color = (*cols[0], 1); cr.elements[1].color = (*cols[1], 1); cr.elements[1].position = 1.0; cr.interpolation = 'CONSTANT'
for k, c_ in enumerate(cols[2:]):
    e_ = cr.elements.new((k + 1) / len(cols)); e_.color = (*c_, 1)
nt.links.new(vor.outputs['Color'], ramp.inputs['Fac']) if 'Color' in vor.outputs else None
try: nt.links.new(vor.outputs['Distance'], ramp.inputs['Fac'])
except Exception: pass
sep = nt.nodes.new('ShaderNodeSeparateColor'); nt.links.new(vor.outputs['Color'], sep.inputs[0]); nt.links.new(sep.outputs[0], ramp.inputs['Fac'])
nt.links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
for x0 in np.arange(FX0, FX1, 20.0):
    z0 = course_z(x0 + 10) - 2.6
    for k in range(6):                       # stepped stand rows
        box((x0 + 10, 7.0 + k * 0.8, z0 + 0.6 + k * 0.55), (20.02, 0.8, 0.3 + k * 1.1 if False else 0.55), M_DARK if k % 2 else M_STEEL, name='step')
        box((x0 + 10, 6.75 + k * 0.8, z0 + 0.95 + k * 0.55), (20.02, 0.35, 0.25), CROWD, name='crowd')
    box((x0 + 10, 11.8, z0 + 4.8), (20.02, 0.4, 2.2), mat('roofwall', (0.85, 0.85, 0.9), 0.6), name='back')
# bunting (万国旗): strings of little triangle flags between poles above the far side of the course
FLAGC = [(0.9, 0.1, 0.1), (1, 0.85, 0.1), (0.1, 0.5, 0.9), (0.1, 0.7, 0.3), (1, 1, 1), (0.9, 0.4, 0.1)]
flag_objs = []
for x0 in np.arange(FX0, FX1, 9.0):
    zb = course_z(x0) + 3.8; zb1 = course_z(x0 + 9) + 3.8
    rod((x0, 3.0, course_z(x0) - 2.6), (x0, 3.0, zb + 0.3), 0.04, M_STEEL, 6)
    for k in range(14):
        u = (k + 0.5) / 14; x = x0 + 9 * u; z = zb + (zb1 - zb) * u - 0.6 * math.sin(math.pi * u)
        bm = bmesh.new(); a = bm.verts.new((-0.16, 0, 0)); b = bm.verts.new((0.16, 0, 0)); c = bm.verts.new((0, 0, -0.38)); bm.faces.new((a, b, c))
        me = bpy.data.meshes.new('fl'); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new('flag', me))
        o.location = (x, 3.0, z); o.data.materials.append(M_OBJ); o.color = (*FLAGC[(k + int(x0)) % 6], 1)
        flag_objs.append(o)
# bunting lamps: flash on every off-beat chord stab
STAB = sorted(set(round(n[0], 3) for n in NOTES if n[2] == 'hnst'))
lamp_a = 0.08 + 0.92 * pulse_alpha(STAB, decay=2.5, length=7)
M_LAMP = emis('lamp', (1.0, 0.85, 0.5), 1.0)
em_l = M_LAMP.node_tree.nodes['Emission']
for f in range(NF):                                  # one shared animated strength for all bunting lamps
    if 0 < f < NF - 1 and abs(lamp_a[f] - lamp_a[f - 1]) < 1e-3 and abs(lamp_a[f] - lamp_a[f + 1]) < 1e-3: continue
    em_l.inputs['Strength'].default_value = float(0.5 + 14 * lamp_a[f]); em_l.inputs['Strength'].keyframe_insert('default_value', frame=f)
for x0 in np.arange(FX0, FX1, 9.0):
    zb = course_z(x0) + 3.8; zb1 = course_z(x0 + 9) + 3.8
    for k in range(0, 14, 2):
        u = (k + 1.0) / 14; x = x0 + 9 * u; z = zb + (zb1 - zb) * u - 0.6 * math.sin(math.pi * u) + 0.05
        sphere((x, 2.95, z), 0.07, M_LAMP, (10, 6))
# cheer-squad taiko drums (red and white), struck on every bass note
BASS = sorted(set(round(n[0], 3) for n in NOTES if n[2] == 'tuba'))
TIMP = sorted(set(round(n[0], 3) for n in NOTES if n[2] == 'timp'))
M_DRUM = mat('drum', (0.45, 0.12, 0.05), 0.5)
skin_a = pulse_alpha(BASS, decay=2.0, length=6)
SKIN = {}
for team, col in (('R', (0.95, 0.25, 0.2)), ('W', (0.95, 0.95, 0.95))):
    m = mat('skin' + team, col, 0.6); b = m.node_tree.nodes['Principled BSDF']; b.inputs['Emission Color'].default_value = (*col, 1)
    for f in range(NF):
        if 0 < f < NF - 1 and abs(skin_a[f] - skin_a[f - 1]) < 1e-3 and abs(skin_a[f] - skin_a[f + 1]) < 1e-3: continue
        b.inputs['Emission Strength'].default_value = float(3.0 * skin_a[f]); b.inputs['Emission Strength'].keyframe_insert('default_value', frame=f)
    SKIN[team] = m
# one master mallet swing (lifted between beats, down on every bass note); every mallet copies it
up = np.zeros(NF)
for f in range(NF):
    t = T[f]; k = np.searchsorted(BASS, t); nxt = BASS[k] if k < len(BASS) else 1e9; prv = BASS[k - 1] if k > 0 else -1e9
    if nxt - prv < 0.6 and prv > 0:
        u = (t - prv) / (nxt - prv); up[f] = 0.55 * math.sin(math.pi * u) ** 0.7
MAL = link(bpy.data.objects.new('mallet_master', None)); MAL.rotation_mode = 'XYZ'; key_series(MAL, 'rotation_euler', 1, -up)
for x0 in np.arange(8.0, FX1 - 20, 10.0):
    for side, team in ((0, 'R'), (1, 'W')):
        col = (0.95, 0.2, 0.15) if team == 'R' else (0.95, 0.95, 0.95)
        x = x0 + side * 1.2; z = low_z(x) - 1.4 + 0.75; y = 4.2
        rod((x, y - 0.25, z), (x, y + 0.25, z), 0.42, M_DRUM, 24)
        rod((x, y - 0.27, z), (x, y - 0.25, z), 0.4, SKIN[team], 24)
        rod((x, y + 0.1, z - 0.42), (x, y + 0.1, z - 0.8), 0.06, M_WOOD, 6)
        piv = link(bpy.data.objects.new('piv', None)); piv.location = (x + 0.55, y - 0.45, z + 0.45)
        m_ = rod((0, 0, 0), (-0.5, 0, -0.38), 0.025, M_WOOD, 6); m_.parent = piv
        hd = sphere((-0.52, 0, -0.4), 0.07, M_OBJ); hd.color = (*col, 1); hd.parent = piv
        cr_ = piv.constraints.new('COPY_ROTATION'); cr_.target = MAL

# ---------------- start: bell tower, trigger + pistol, start pads + plungers ----------------
tg = EV['trigger']; T_GUN = tg['t']; trig = np.array(tg['p'])
gh = BP['gold'].hits
YG = LY['gold']
for h in gh:                                         # each bell hangs on a short rod from the sloped beam above
    c = np.array(h['p']) - R * np.array(h['n'])
    rod((c[0], YG + 0.25, c[2] - 0.14), (c[0], YG + 0.25, c[2] + 0.35), 0.012, M_STEEL, 6)
    rod((c[0], YG, c[2] - 0.14), (c[0], YG + 0.25, c[2] - 0.14), 0.012, M_STEEL, 6)
c0b = np.array(gh[0]['p']); c1b = np.array(gh[-1]['p'])
rod((c0b[0] - 0.3, YG + 0.25, c0b[2] + 0.2), (c1b[0] + 0.3, YG + 0.25, c1b[2] + 0.2), 0.04, M_WOOD, 8)
for xb_, zb_ in ((c0b[0] - 0.3, c0b[2] + 0.2), (c1b[0] + 0.3, c1b[2] + 0.2)):
    rod((xb_, YG + 0.25, zb_), (xb_, YG + 0.25, course_z(0) - 2.6), 0.05, M_WOOD, 8)
text('スタート', (0.1, -1.6, 0.9), 0.4, M_TXT_W)
# drop chute for the gold ball
ch = EV['chute']; c0 = np.array(ch['start']); c1 = np.array(ch['mouth'])
rod(c0 - [0.5, 0, 0], c1 - [0.05, 0, 0], 0.2, M_GOLD, 20)                 # the gold ball waits inside, rolls out on the first beat
rod(c0 - [0.5, 0, 0.2], (c0[0] - 0.5, c0[1], course_z(0) - 2.6), 0.05, M_WOOD, 8)
# trigger lever cup + pistol
box((trig[0], YG, trig[2] - R - 0.03), (0.34, 0.3, 0.05), M_STEEL)
rod((trig[0], YG, trig[2] - R - 0.05), (trig[0], YG, course_z(0) - 2.6), 0.035, M_STEEL, 8)
pist = box((trig[0] + 0.35, YG, trig[2] + 0.1), (0.42, 0.1, 0.12), M_DARK, name='pistol')
box((trig[0] + 0.2, YG, trig[2] - 0.02), (0.1, 0.09, 0.2), M_DARK)
smoke = sphere((trig[0] + 0.7, YG, trig[2] + 0.12), 0.25, M_FX)
smoke.color = (1, 1, 0.95, 0)
fg = int(round(T_GUN * FPS)); sa = np.zeros(NF); ss = np.full(NF, 0.01)
for df in range(24):
    if fg + df < NF: sa[fg + df] = 0.9 * (1 - df / 24) ** 1.5; ss[fg + df] = 0.3 + 1.0 * (1 - (1 - df / 24) ** 2)
key_series(smoke, 'color', 3, sa, 1e-3)
for k in range(3): key_series(smoke, 'scale', k, ss * 0.25)
for team in ('R', 'W'):
    st = EV['start'][team]; p = np.array(st['p'])
    box((p[0], p[1], p[2] - R - 0.05), (0.5, 0.9, 0.1), M_OBJ, name='pad').color = ((0.9, 0.2, 0.2, 1) if team == 'R' else (0.95, 0.95, 0.95, 1))
    rod((p[0], p[1], p[2] - R - 0.1), (p[0], p[1], p[2] - 2.0), 0.04, M_STEEL, 8)
    # plunger: rod behind the ball, shoots forward at the gun, then retracts
    pl = rod((-0.6, 0, 0), (0, 0, 0), 0.06, M_STEEL, 12); pl.location = (p[0] - R - 0.02, p[1], p[2])
    bpy.ops.mesh.primitive_cylinder_add(radius=0.12, depth=0.04, location=(0, 0, 0), rotation=(0, math.pi / 2, 0)); ph = bpy.context.object; ph.data.materials.append(M_DARK); ph.parent = pl
    px = np.full(NF, p[0] - R - 0.02 - 0.25)
    for f in range(NF):
        u = (T[f] - (T_GUN - 0.05)) / 0.05
        if 0 <= u <= 1: px[f] = p[0] - R - 0.27 + 0.25 * u
        elif T[f] > T_GUN: px[f] = p[0] - R - 0.02 - 0.25 * min(1, (T[f] - T_GUN) / 0.6)
    key_series(pl, 'location', 0, px)
    rod((p[0] - 0.9, p[1], p[2]), (p[0] - 0.9, p[1], p[2] - 2.0), 0.04, M_STEEL, 8)
    text('紅組' if team == 'R' else '白組', (p[0] - 0.6, p[1] - 0.5, p[2] + 0.65), 0.32, M_TXT_R if team == 'R' else M_TXT_W)

# ---------------- baton zone: 2nd runners wait on tees, 1st runners drop into team baskets ----------------
for team in ('R', 'W'):
    ho = EV['handoff'][team]; p2 = np.array(ho['p2']); pb = np.array(ho['basket'])
    rod((p2[0], p2[1], p2[2] - R), (p2[0], p2[1], p2[2] - R - 0.5), 0.03, M_STEEL, 8)   # tee
    bpy.ops.mesh.primitive_torus_add(major_radius=0.07, minor_radius=0.02, location=(p2[0], p2[1], p2[2] - R + 0.01)); bpy.context.object.data.materials.append(M_STEEL)
    # basket: open cup the 1st runner falls into
    bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=0.3, vertices=24, location=(pb[0], pb[1], pb[2] - R + 0.15 - 0.02))
    bk = bpy.context.object; bk.data.materials.append(M_OBJ); bk.color = ((0.9, 0.15, 0.1, 1) if team == 'R' else (0.95, 0.95, 0.95, 1))
    bm = bmesh.new(); bm.from_mesh(bk.data)
    top = [f for f in bm.faces if f.normal.z > 0.9]; bmesh.ops.delete(bm, geom=top, context='FACES')
    bm.to_mesh(bk.data); bm.free()
    sol = bk.modifiers.new('s', 'SOLIDIFY'); sol.thickness = 0.03
    rod((pb[0], pb[1], pb[2] - R - 0.02), (pb[0], pb[1], pb[2] - 2.2), 0.03, M_STEEL, 6)
    text('バトンゾーン', (p2[0], p2[1] - 0.55, p2[2] + 0.75), 0.22, M_TXT_Y)

# ---------------- pit: trapdoor, mud pit, see-saw catapult, weight ball ----------------
tp = EV['trap']; pt = np.array(tp['p'])
# trapdoor panel hinged at its far edge, swings down right after white lands on it
hinge = link(bpy.data.objects.new('hinge', None)); hinge.location = (pt[0] + 0.3, pt[1], pt[2] - R - 0.02)
door = box((-0.3, 0, -0.03), (0.6, 0.95, 0.05), M_OBJ, name='door'); door.color = (0.95, 0.85, 0.2, 1); door.parent = hinge
ang = np.zeros(NF)
for f in range(NF):
    u = (T[f] - tp['t'] - 0.02) / 0.22
    if u > 0: ang[f] = min(1.0, u) ** 2 * 1.45
hinge.rotation_mode = 'XYZ'; key_series(hinge, 'rotation_euler', 1, -ang)
mud = lane_points(['W2'], t0=t_mud - 0.01, t1=SECS['br'] + 0.01)
PX = [m_[0] for m_ in mud]; PZ = [m_[1] for m_ in mud]
pit_z = min(PZ) if PZ else pt[2] - 2
ss_ = EV['seesaw']; cup = np.array(ss_['cup']); piv = np.array(ss_['pivot'])
px0, px1 = pt[0] + 0.32, piv[0] + 1.6
box(((px0 + px1) / 2, LY['W'], pit_z - 0.15), (px1 - px0, 1.1, 0.3), M_MUD, name='mudfloor')
box(((px0 + px1) / 2, LY['W'] + 0.6, (pt[2] - 0.3 + pit_z - 0.3) / 2), (px1 - px0, 0.08, pt[2] - pit_z), M_PITWALL, name='pitwall')   # back wall only: cut-away pit
for xx in (px0, px1):
    box((xx, LY['W'], pit_z - 0.3), (0.08, 1.2, 0.6), M_PITWALL)
text('落とし穴', (pt[0] + 1.5, LY['W'] - 0.7, pt[2] + 0.7), 0.28, M_TXT_Y)
# see-saw: plank on a pivot; cup at white's end (left), weight lands on the right end
sw = link(bpy.data.objects.new('seesaw', None)); sw.location = tuple(piv)
arm = ss_['arm']
plank = box((0, 0, 0), (2 * arm + 0.3, 0.3, 0.06), M_WOOD, name='plank'); plank.parent = sw
cupo = box((-arm, 0, 0.06), (0.22, 0.28, 0.06), M_STEEL); cupo.parent = sw
bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=0.25, radius2=0.02, depth=0.3, location=(piv[0], piv[1], piv[2] - 0.18)); bpy.context.object.data.materials.append(M_STEEL)
th0 = math.atan2((cup[2] - R - 0.03) - piv[2], arm)          # left end down (white's cup)
th0 = -abs(math.asin(max(-0.9, min(0.9, (piv[2] - (cup[2] - R - 0.09)) / arm))))
swa = np.full(NF, th0)
tl_ = ss_['t_launch']
for f in range(NF):
    u = (T[f] - tl_) / 0.07
    if u > 0: swa[f] = th0 + (-2 * th0) * min(1, u)
sw.rotation_mode = 'XYZ'; key_series(sw, 'rotation_euler', 1, -swa)
# weight ball: sits in a hopper above the right end, drops on the downbeat of bridge bar 3
RWt = ss_['weight_r']; wx = piv[0] + arm - 0.05
w_top = piv[2] + arm * math.sin(-th0) + RWt + 0.06
hop_z = w_top + ss_['fall']
bpy.ops.mesh.primitive_uv_sphere_add(radius=RWt, segments=32, ring_count=16); wb = bpy.context.object
for p in wb.data.polygons: p.use_smooth = True
wb.data.materials.append(mat('weight', (0.15, 0.25, 0.75), 0.25, **{'Coat Weight': 1.0}))
WP = np.zeros((NF, 3))
for f in range(NF):
    t = T[f]; trel = ss_['t_release']
    if t < trel: z = hop_z
    elif t < tl_: z = hop_z - 0.5 * G * (t - trel) ** 2
    else: z = piv[2] - arm * math.sin(-th0) + RWt + 0.06
    WP[f] = (wx, LY['W'], z)
key_loc(wb, WP)
# hopper (open-bottom funnel with a flap) holding the weight
rod((wx, LY['W'] + 0.5, hop_z + RWt + 0.1), (wx, LY['W'] + 0.5, pit_z - 0.3), 0.05, M_STEEL, 8)
box((wx, LY['W'] + 0.25, hop_z + RWt + 0.12), (0.2, 0.55, 0.05), M_STEEL)
text('シーソー', (piv[0], LY['W'] - 0.65, pit_z + 0.3), 0.22, M_TXT_Y)
# landing mat
la = np.array(EV['catapult']['land'])
box((la[0], la[1], la[2] - R - 0.12), (1.2, 1.0, 0.18), M_BLUE, name='gymmat')

# ---------------- hurdles in the C section (frames under the hurdle bars) ----------------
M_HURD = mat('hurd', (0.95, 0.95, 0.95), 0.4)
for B_ in BARS:
    if B_['kind'] == 'hurdle':
        c = B_['c']; y0 = c[1]
        for dy in (-0.42, 0.42):
            rod((c[0], y0 + dy, c[2] - 0.03), (c[0], y0 + dy, c[2] - 0.45), 0.02, M_HURD, 6)
            rod((c[0] - 0.15, y0 + dy, c[2] - 0.45), (c[0] + 0.15, y0 + dy, c[2] - 0.45), 0.02, M_DARK, 6)

# ---------------- finish: tape, gate, podium, kusudama, victory bells ----------------
tf = EV['finish']['t']
xt = float(BP['W2'].at(tf - 0.06)[0])
zt = max(BP['W2'].at(tf - 0.06)[2], BP['R2'].at(tf + 0.07)[2])
pW_t = BP['W2'].at(tf - 0.06); zt = float(pW_t[2])
T_TAPE = None
for t in np.arange(tf - 0.3, tf, 1 / 240):
    if BP['W2'].at(t)[0] >= xt: T_TAPE = t; break
T_TAPE = T_TAPE or tf - 0.06
assert BP['R2'].at(T_TAPE)[0] < xt - 0.2, 'red must be behind at the tape'
for yy in (LY['W'] - 0.7, LY['R'] + 0.7):
    rod((xt, yy, zt - 2.6), (xt, yy, zt + 1.3), 0.022, M_WHITE, 10)
box((xt, (LY['W'] + LY['R']) / 2, zt + 1.3), (0.1, LY['R'] - LY['W'] + 1.5, 0.32), M_OBJ, name='goalbar').color = (0.9, 0.1, 0.1, 1)
text('ゴール', (xt, LY['W'] - 0.78, zt + 1.3), 0.26, M_TXT_W, rot=(math.pi / 2, 0, 0))
# tape: two halves (from each post to the middle), break and swing back when white touches
yl, yr = LY['W'] - 0.7, LY['R'] + 0.7; ym = LY['W']
M_TAPE = mat('tape', (0.95, 0.1, 0.25), 0.6)
for side, (ya, yb) in enumerate(((yl, ym), (yr, ym))):
    pv = link(bpy.data.objects.new('tp', None)); pv.location = (xt, ya, zt)
    seg = box(((0, (yb - ya) / 2, 0)), (0.03, abs(yb - ya), 0.09), M_TAPE, name='tape'); seg.parent = pv
    rot = np.zeros(NF)
    for f in range(NF):
        u = (T[f] - T_TAPE) / 0.5
        if u > 0: rot[f] = (1 - math.exp(-3 * u)) * 1.25 + 0.1 * math.sin(10 * u) * math.exp(-2 * u)
    pv.rotation_mode = 'XYZ'; key_series(pv, 'rotation_euler', 2, rot * (1 if side == 0 else -1))
    key_series(pv, 'rotation_euler', 1, rot * 0.6)
# podium blocks (1 = white, taller; 2 = red)
pod = EV['podium']; ground = min(pod['W']['p'][2], pod['R']['p'][2]) - 0.75
for team, num in (('W', '1'), ('R', '2')):
    p = np.array(pod[team]['p']); ztop = p[2] - R
    box((p[0], p[1], (ztop + ground) / 2), (0.8, 0.75, ztop - ground), M_OBJ, name='podium').color = ((0.97, 0.97, 0.97, 1) if team == 'W' else (0.9, 0.15, 0.12, 1))
    box((p[0], p[1], ztop - 0.02), (0.82, 0.77, 0.04), M_GOLD)
    text(num, (p[0], p[1] - 0.39, ztop - 0.38), 0.55, M_TXT_Y if team == 'W' else M_TXT_W)
    # podium-top contact plates flash on the ping-pong hits
    o = box((p[0], p[1], ztop - 0.005), (0.3, 0.3, 0.012), M_KEY, name='podkey'); o.color = (1, 0.85, 0.3, 0)
    hts = [h['t'] for h in BP[team + '2'].hits if h['kind'] == 'podium' and abs(h['p'][2] - p[2]) < 0.05]
    key_series(o, 'color', 3, pulse_alpha(hts, decay=4.0, length=14), 1e-3)
box(((pod['W']['p'][0] + pod['R']['p'][0]) / 2, 0, ground - 0.1), (4.0, 2.2, 0.2), M_GRASS)
# victory bells + gold balls from the kusudama for the final tutti
cx = (pod['W']['p'][0] + pod['R']['p'][0]) / 2
ztopW = pod['W']['p'][2]
GN = EV['gold_notes']
BELLS = []
for k, i in enumerate(GN):
    t = TN[i]; u = (k / max(1, len(GN) - 2)) if i != GN[-1] else 0.5
    bx = cx - 1.5 + 3.0 * ((k * 0.37) % 1.0) if i != GN[-1] else cx + 0.6
    bz = ztopW + 0.55 + 0.25 * math.sin(k * 1.3)
    by = 0.75
    bl = obj(prim('cone', M_BELL, 24, 0.2, 0.08, 0.24), 'vbell'); bl.location = (bx, by, bz - 0.12); bl.color = (1.0, 0.8, 0.35, 0)
    box((bx, by, bz + 0.1 + 1.5 - 0.12), (0.3, 0.3, 0.04), M_GOLD, name='rack')
    key_series(bl, 'color', 3, pulse_alpha([t], decay=6, length=24), 1e-3)
    rod((bx, by, bz - 0.24), (bx, by + 0.5, bz - 0.24), 0.015, M_STEEL, 6)
    # gold ball: released from rest 2.2 above, hits the bell crown, bounces off backward
    gbo = sphere((0, 0, 0), 0.1, M_GOLD)
    h0 = 1.5; tdrop = math.sqrt(2 * h0 / G); P = np.zeros((NF, 3))
    for f in range(NF):
        tt = T[f]
        if tt < t - tdrop: P[f] = (bx, by, bz + 0.1 + h0)
        elif tt < t: s_ = tt - (t - tdrop); P[f] = (bx, by, bz + 0.1 + h0 - 0.5 * G * s_ * s_)
        else:
            s_ = tt - t; vz0 = 0.45 * G * tdrop
            P[f] = (bx + 0.4 * s_, by + 1.2 * s_, bz + 0.1 + vz0 * s_ - 0.5 * G * s_ * s_)
    key_loc(gbo, P)
    BELLS.append((bx, by, bz, t))
# kusudama above podium 1, opens on the final chord, banner + confetti
kx, kz = pod['W']['p'][0], ztopW + 2.35
rod((kx, 0.0, kz + 0.45), (kx, 0.0, kz + 3.0), 0.02, M_STEEL, 6)
M_KUSU = mat('kusu', (1.0, 0.8, 0.1), 0.4)
for side in (-1, 1):
    hp = link(bpy.data.objects.new('khinge', None)); hp.location = (kx, -0.0, kz + 0.42)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.42, segments=32, ring_count=16); hm = bpy.context.object
    bm = bmesh.new(); bm.from_mesh(hm.data)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.x * side < -1e-4], context='VERTS'); bm.to_mesh(hm.data); bm.free()
    sol = hm.modifiers.new('s', 'SOLIDIFY'); sol.thickness = 0.02
    hm.data.materials.append(M_KUSU); hm.parent = hp; hm.location = (0, 0, -0.42)
    rot = np.zeros(NF)
    for f in range(NF):
        u = (T[f] - FINAL) / 0.35
        if u > 0: rot[f] = side * min(1.0, u) ** 0.6 * 2.2 + side * 0.08 * math.sin(8 * u) * math.exp(-u)
    hp.rotation_mode = 'XYZ'; key_series(hp, 'rotation_euler', 1, rot)
ban = link(bpy.data.objects.new('banner', None)); ban.location = (kx, -0.02, kz - 0.05)
bpl = box((0, 0, -1.1), (1.2, 0.02, 2.2), M_WHITE, name='banner'); bpl.parent = ban
bt = text('白組\n優勝', (0, -0.03, -1.1), 0.5, M_TXT_RD, ext=0.005); bt.parent = ban
bt.data.space_line = 0.95
bsc = np.full(NF, 0.001)
for f in range(NF):
    u = (T[f] - FINAL - 0.12) / 0.4
    if u > 0: bsc[f] = min(1.0, u) + 0.08 * math.sin(9 * u) * math.exp(-2 * u)
key_series(ban, 'scale', 2, bsc); key_series(ban, 'scale', 0, np.where(bsc > 0.01, 1.0, 0.001)); key_series(ban, 'scale', 1, np.where(bsc > 0.01, 1.0, 0.001))
CONF = 90
for k in range(CONF):
    cf = obj(prim('plane', M_OBJ), 'conf')
    cf.color = (*FLAGC[k % 6], 1)
    ang0 = rng.uniform(0, 2 * math.pi); sp = rng.uniform(0.6, 2.2); vy = rng.uniform(-0.8, 0.8)
    P = np.zeros((NF, 3)); RT = np.zeros((NF, 3))
    for f in range(NF):
        s_ = T[f] - FINAL
        if s_ < 0: P[f] = (kx, 0, kz + 0.2); RT[f] = 0; continue
        burst = 1 - math.exp(-s_ / 0.25)
        x = kx + sp * math.cos(ang0) * 0.6 * burst + 0.12 * math.sin(3 * s_ + k)
        z = kz + 0.2 + sp * math.sin(ang0) * 0.4 * burst - 0.75 * s_
        P[f] = (x, vy * burst, z); RT[f] = (s_ * (3 + k % 5), s_ * (2 + k % 3), s_ * 1.5)
    key_loc(cf, P)
    sc_ = np.where(T >= FINAL, 1.0, 0.001)
    for c_ in range(3): key_series(cf, 'scale', c_, sc_ * 0.08); key_series(cf, 'rotation_euler', c_, RT[:, c_])

# ---------------- world & sun ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes['Background']; bg.inputs['Color'].default_value = (0.4, 0.6, 0.95, 1); bg.inputs['Strength'].default_value = 0.55
sun = bpy.data.lights.new('sun', 'SUN'); sun.energy = 2.6; sun.angle = 0.15; sun.color = (1.0, 0.96, 0.9)
so = link(bpy.data.objects.new('sun', sun)); so.rotation_euler = (math.radians(50), math.radians(-25), math.radians(-20))
# sky backdrop (gradient wall far behind)
SKY = bpy.data.materials.new('sky'); SKY.use_nodes = True; nt = SKY.node_tree; nt.nodes.clear()
tc = nt.nodes.new('ShaderNodeTexCoord'); sepz = nt.nodes.new('ShaderNodeSeparateXYZ'); rp = nt.nodes.new('ShaderNodeValToRGB')
rp.color_ramp.elements[0].color = (0.45, 0.65, 0.95, 1); rp.color_ramp.elements[1].color = (0.08, 0.28, 0.75, 1)
mp = nt.nodes.new('ShaderNodeMapRange'); mp.inputs['From Min'].default_value = 0; mp.inputs['From Max'].default_value = 14
em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Strength'].default_value = 0.8; out = nt.nodes.new('ShaderNodeOutputMaterial')
nt.links.new(tc.outputs['Object'], sepz.inputs[0]); nt.links.new(sepz.outputs['Z'], mp.inputs['Value']); nt.links.new(mp.outputs[0], rp.inputs['Fac'])
nt.links.new(rp.outputs['Color'], em.inputs['Color']); nt.links.new(em.outputs[0], out.inputs[0])
for x0 in np.arange(FX0, FX1, 40.0):
    zc = course_z(x0 + 20)
    o = obj(prim('plane', SKY), 'sky'); o.location = (x0 + 20, 14, zc); o.rotation_euler = (math.pi / 2, 0, 0); o.scale = (40.1, 30, 1)

# ---------------- camera: follow the race, widen for the big story beats ----------------
def smooth(Xa, k):
    Y = Xa.copy(); a = 1 - math.exp(-1 / FPS / k)
    for f in range(1, len(Xa)): Y[f] = Y[f - 1] + (Xa[f] - Y[f - 1]) * a
    for f in range(len(Xa) - 2, -1, -1): Y[f] = Y[f + 1] + (Y[f] - Y[f + 1]) * a
    return Y
FOC = np.zeros((NF, 3)); WID = np.zeros(NF)
podc = np.array([cx, 0, pod['W']['p'][2] + 0.2])
for f in range(NF):
    t = T[f]
    r_, w_ = run_pos('R', t), run_pos('W', t)
    if t < T_GUN + 0.2:                                     # opening: tower + both runners at the start
        FOC[f] = (-3.2, 0, 1.9); WID[f] = 8.4
    elif t > tf + 1.6:                                       # podium
        FOC[f] = podc + np.array([0, 0, 0.5]); WID[f] = 4.6
    else:
        FOC[f] = (r_ + w_) / 2 + np.array([0.5, 0, 0.45]); WID[f] = abs(r_[0] - w_[0]) + 3.6
    if t_trap - 0.3 < t < EV['catapult']['t_land'] + 0.8:     # white's misfortune and the catapult: frame white, red at the edge
        wgt = 0.85
        FOC[f] = wgt * w_ + (1 - wgt) * r_ + np.array([0.6, 0, 0.9]); WID[f] = min(6.6, abs(r_[0] - w_[0]) * 0.55 + 4.0)
WID = np.clip(WID, 4.2, 9.0)
FOC[:, 0] = smooth(FOC[:, 0], 0.35); FOC[:, 2] = smooth(FOC[:, 2], 0.5); WID = smooth(WID, 0.6)
LENS = 32.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 400
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24                       # distance so that the frame is WID wide at the lanes
    CP[f] = (FOC[f, 0], -D, FOC[f, 2] + 0.16 * D)
    TP[f] = (FOC[f, 0], 0.0, FOC[f, 2] - 0.25 * D / 8)
key_loc(cam, CP); key_loc(tgt, TP)

# ---------------- scoreboard (HUD on the camera): progress bars, 1st-place lamp ----------------
hud = link(bpy.data.objects.new('hud', None)); hud.parent = cam; hud.location = (0, 0, -3.0)
HH = 3.0 * 24 / LENS; HWd = HH * 9 / 16                    # visible size at the hud plane
M_PANEL = emis('panel', (0.02, 0.03, 0.06), 1.0)
pz = HH / 2 - 0.16
pn = box((0, pz, 0), (HWd * 0.9, 0.24, 0.005), M_PANEL, rot=None, name='panel'); pn.parent = hud
for o in [pn]: o.visible_shadow = False
rows = {'R': pz + 0.055, 'W': pz - 0.055}
bar_len = HWd * 0.5; bx0 = -HWd * 0.45 + 0.24
progress = {}
for team in ('R', 'W'):
    xs_ = np.array([run_pos(team, t)[0] for t in T]); progress[team] = np.clip((xs_ - XS) / (XF - XS), 0, 1)
raw = {team: np.array([run_pos(team, t)[0] for t in T]) for team in ('R', 'W')}
lead_t = np.where(raw['W'] > raw['R'], 1.0, 0.0)
lead_t[T >= tf - 0.1] = 1.0                                   # white crossed the line first: ranking frozen
for team in ('R', 'W'):
    yr = rows[team]
    tl = text('紅' if team == 'R' else '白', (bx0 - 0.1, yr, 0.003), 0.12, M_TXT_R if team == 'R' else M_TXT_W, rot=(0, 0, 0)); tl.parent = hud; tl.visible_shadow = False
    bgb = box((bx0 + bar_len / 2, yr, 0.002), (bar_len, 0.04, 0.003), emis('bbg', (0.15, 0.15, 0.2), 1.0), name='bbg'); bgb.parent = hud; bgb.visible_shadow = False
    fill = box((0.5, 0, 0), (1.0, 0.04, 0.003), emis('fill' + team, (1, 0.12, 0.08) if team == 'R' else (0.95, 0.95, 1.0), 2.5), name='fill')
    fp = link(bpy.data.objects.new('fp', None)); fp.parent = hud; fp.location = (bx0, yr, 0.004); fill.parent = fp
    key_series(fp, 'scale', 0, np.maximum(1e-3, progress[team] * bar_len), 1e-4)
    # rank lamps: "1位" lit for the leader
    first = lead_t if team == 'W' else 1 - lead_t
    for label, on in (('1位', first), ('2位', 1 - first)):
        tx = text(label, (bx0 + bar_len + 0.15, yr, 0.004), 0.1, M_TXT_Y if label == '1位' else M_TXT_W, rot=(0, 0, 0)); tx.parent = hud; tx.visible_shadow = False
        key_series(tx, 'scale', 0, np.maximum(1e-3, on), 1e-3); key_series(tx, 'scale', 1, np.maximum(1e-3, on), 1e-3)
for o in bpy.data.objects:
    if o.parent is hud or (o.parent is not None and o.parent.parent is hud): o.visible_shadow = False

# ---------------- sanity checks ----------------
def check():
    info = {}
    # 1) contact distance at the frame nearest each hit
    gaps = []
    for B_ in BARS:
        n = B_['ball']; f = int(round(B_['t'] * FPS)); p = POS[n][min(NF - 1, f)]
        gaps.append(np.linalg.norm(p - (B_['contact'] + R * B_['n'])))
    info['contact_gap_at_hit_frame_max'] = round(max(gaps), 3); info['contact_gap_mean'] = round(float(np.mean(gaps)), 3)
    # 2) collisions: ball surface entering any bar box other than at its own hit (+-0.03 s)
    bad = 0; worst = 0
    for n, b in BP.items():
        for t in np.arange(0.2, FINAL + 1.0, 1 / 240):
            p = b.at(t)
            for B_ in BARS:
                if abs(B_['c'][0] - p[0]) > 0.6 or abs(B_['c'][2] - p[2]) > 0.6 or abs(B_['c'][1] - p[1]) > 1.0: continue
                if B_['ball'] == n and abs(B_['t'] - t) < 0.03: continue
                Mx = frame_from_n(B_['n']); rel = Mx.transposed() @ Vector(p - B_['c'])
                d = Vector((max(0, abs(rel.x) - B_['hw']), max(0, abs(rel.y) - B_['hd']), max(0, abs(rel.z) - B_['th'])))
                pen = R - d.length
                if pen > 0.01:
                    bad += 1; worst = max(worst, pen)
                    if bad <= 12: print('COL', n, round(t, 3), 'bar of', B_['ball'], B_['i'], B_['kind'], 'pen', round(pen, 3))
    info['collision_samples'] = bad; info['collision_worst'] = round(worst, 3)
    # 3) ball-ball overlap (except the hand-off contact)
    names = list(BP); mind = 9
    for f in range(NF):
        for a in range(len(names)):
            for c in range(a + 1, len(names)):
                d = np.linalg.norm(POS[names[a]][f] - POS[names[c]][f])
                mind = min(mind, d)
    info['min_ball_distance'] = round(mind, 3)
    # 4) in-flight frame-to-frame speed ratio (frames not adjacent to a hit/mechanism event)
    ev_t = [h['t'] for b in BP.values() for h in b.hits] + [T_GUN, EV['handoff']['R']['t'], EV['handoff']['W']['t'],
           EV['handoff']['R']['tland'], EV['handoff']['W']['tland'], EV['seesaw']['t_catch'], EV['seesaw']['t_launch']]
    evf = set(int(round(t * FPS)) + k for t in ev_t for k in (-1, 0, 1, 2))
    worst = 1.0
    for n in names:
        sp = np.linalg.norm(np.diff(POS[n], axis=0), axis=1) * FPS
        for f in range(1, len(sp)):
            if f in evf or f + 1 in evf or min(sp[f], sp[f - 1]) < 0.5: continue
            worst = max(worst, max(sp[f], sp[f - 1]) / min(sp[f], sp[f - 1]))
    info['inflight_speed_ratio_max'] = round(worst, 3)
    info['bars'] = len(BARS); info['objects'] = len(bpy.data.objects); info['NF'] = NF
    return info

if MODE in ('check', 'test') and (MODE == 'check' or os.environ.get('CHECK')):
    print(check())
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
