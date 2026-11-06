"""In the Hall of the Mountain King - a crystal cave inside the mountain.
Three glowing marbles roll down a triple-helix track paved with crystal keys (one key per melody note: the marble
strikes it and it rings). Marble 1 starts alone at the top; marble 2 and 3 are released from their own gates when the
music grows. Stone hammers on pivots beat the drum-stones / timpani, a bronze gong and a pendulum on the stalactite
chimes. At the end the marbles plunge into the giant crystal, which flares and bursts into shards."""
import os, sys, json, math, time, colorsys
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json')); NOTES = json.load(open(f'{HERE}/notes.json'))
FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.5) * FPS); T = np.arange(NF) / FPS
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
sc.render.resolution_x, sc.render.resolution_y = 1080, 1920
sc.render.resolution_percentage = int(os.environ.get('PCT', 50)); sc.render.fps = FPS
sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.3
try: sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Medium High Contrast'
except Exception: pass
sc.frame_start, sc.frame_end = 0, NF - 1
sc.render.use_persistent_data = True


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


def glowmat(name, gain=10.0, base=0.6, rough=0.15):
    """object colour = key colour; object alpha drives emission"""
    m = mat(name, (1, 1, 1), rough, **{'Coat Weight': 0.6}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    oi = nt.nodes.new('ShaderNodeObjectInfo')
    nt.links.new(oi.outputs['Color'], b.inputs['Base Color']); nt.links.new(oi.outputs['Color'], b.inputs['Emission Color'])
    mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY_ADD'; mul.inputs[1].default_value = gain; mul.inputs[2].default_value = base
    nt.links.new(oi.outputs['Alpha'], mul.inputs[0]); nt.links.new(mul.outputs[0], b.inputs['Emission Strength'])
    return m


def key_series(obj, path, idx, vals, eps=1e-5):
    n = len(vals)
    for f in range(n):
        v = vals[f]
        if 0 < f < n - 1 and abs(v - vals[f - 1]) < eps and abs(v - vals[f + 1]) < eps: continue
        getattr(obj, path)[idx] = float(v); obj.keyframe_insert(path, index=idx, frame=f)


def key_alpha(obj, arr, col):
    obj.color = (*col, 0)
    key_series(obj, 'color', 3, arr, 1e-3)


def rod(p0, p1, r, mt, verts=10):
    p0, p1 = Vector(p0), Vector(p1); d = p1 - p0
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=d.length, vertices=verts, location=(p0 + p1) / 2)
    o = bpy.context.object; q = Vector((0, 0, 1)).rotation_difference(d.normalized())
    o.rotation_mode = 'QUATERNION'; o.rotation_quaternion = q
    o.matrix_world = Matrix.Translation((p0 + p1) / 2) @ q.to_matrix().to_4x4()
    o.data.materials.append(mt); return o


def kf(obj, path, idx, pts):
    """sparse keyframes: list of (time_sec, value)"""
    for t, v in pts:
        if idx is None: setattr(obj, path, v); obj.keyframe_insert(path, frame=t * FPS)
        else: getattr(obj, path)[idx] = v; obj.keyframe_insert(path, index=idx, frame=t * FPS)


def lin_fc(obj):
    if obj.animation_data and obj.animation_data.action:
        try:
            for fc in obj.animation_data.action.fcurves:
                for k in fc.keyframe_points: k.interpolation = 'LINEAR'
        except Exception: pass


def smooth(X, k):
    Y = X.copy()
    for f in range(1, len(X)): Y[f] = Y[f - 1] + (X[f] - Y[f - 1]) * (1 - math.exp(-1 / FPS / k))
    return Y


HITS = E['hits']
BEATS = np.array(E['beats'])                      # [global beat, time]
def gbeat(t): return np.interp(t, BEATS[:, 1], BEATS[:, 0])

# ---------------- track geometry: triple helix ----------------
R = 2.4; PITCH = 1.5; K = 0.42                    # radius, rise per turn, track length per beat
LANE_COL = [(1.0, 0.55, 0.15), (0.2, 0.85, 1.0), (1.0, 0.25, 0.85)]
S_START = [0.0, K * 64, K * 128]                  # lanes 2 & 3 join at cycle 2 / 3
LAST_BEAT = 183; S_END = K * LAST_BEAT + 0.35
ZBOT = 2.95; ZTOP = ZBOT + S_END / (2 * math.pi * R) * PITCH
def hel(s, lane, r=R):
    th = s / R + lane * 2 * math.pi / 3 + 0.4
    return np.array([r * math.cos(th), r * math.sin(th), ZTOP - s / (2 * math.pi * R) * PITCH])
def tang(s, lane):
    a, b = hel(s - 0.01, lane), hel(s + 0.01, lane); d = b - a; return d / np.linalg.norm(d)

RAIL = mat('rail', (0.75, 0.62, 0.4), 0.3, **{'Metallic': 1.0})
DARKM = mat('iron', (0.08, 0.07, 0.07), 0.45, **{'Metallic': 0.8})
def curve_obj(name, pts, bevel, mt):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'; cu.bevel_depth = bevel; cu.bevel_resolution = 2
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for i, p in enumerate(pts): sp.points[i].co = (*p, 1)
    o = link(bpy.data.objects.new(name, cu)); o.data.materials.append(mt); return o
HALF = 0.17                                         # half track width
for L in range(3):
    ss = np.arange(S_START[L] - 0.5, S_END + 0.01, 0.06)
    for side in (-1, 1):
        curve_obj(f'rail{L}{side}', [hel(s, L, R + side * HALF) + np.array([0, 0, 0.035]) for s in ss], 0.022, RAIL)
# outer support rods + brackets (rods stand outside the helix, brackets reach the outer rail)
NR = 12
for i in range(NR):
    a = i * 2 * math.pi / NR
    rod((3.0 * math.cos(a), 3.0 * math.sin(a), 0.0), (3.0 * math.cos(a), 3.0 * math.sin(a), ZTOP + 0.6), 0.045, DARKM, 8)
    for L in range(3):
        # s values where lane L crosses angle a
        th0 = L * 2 * math.pi / 3 + 0.4
        n = 0
        while True:
            s = R * ((a - th0) % (2 * math.pi) + 2 * math.pi * n); n += 1
            if s > S_END + 0.4: break
            if s < S_START[L] - 0.5: continue
            p = hel(s, L, R + HALF + 0.02)
            rod((p[0], p[1], p[2]), (3.0 * math.cos(a), 3.0 * math.sin(a), p[2]), 0.02, DARKM, 6)

# ---------------- crystal keys (one per lead note) ----------------
def pitch_col(p, L):
    h = ((p - 59) * 0.045 + [0.07, 0.52, 0.85][L]) % 1
    return colorsys.hsv_to_rgb(h, 0.65, 1.0)
def keymat(name, gain, rough):
    """unlit keys are dark smoky crystal tinted with the key colour; alpha (hit) lights them up"""
    m = mat(name, (1, 1, 1), rough, **{'Coat Weight': 0.7}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    oi = nt.nodes.new('ShaderNodeObjectInfo')
    dim = nt.nodes.new('ShaderNodeMath'); dim.operation = 'MULTIPLY_ADD'; dim.inputs[1].default_value = 1.6; dim.inputs[2].default_value = 0.12
    nt.links.new(oi.outputs['Alpha'], dim.inputs[0])
    mix = nt.nodes.new('ShaderNodeVectorMath'); mix.operation = 'SCALE'
    nt.links.new(oi.outputs['Color'], mix.inputs[0]); nt.links.new(dim.outputs[0], mix.inputs['Scale'])
    nt.links.new(mix.outputs[0], b.inputs['Base Color']); nt.links.new(oi.outputs['Color'], b.inputs['Emission Color'])
    g = nt.nodes.new('ShaderNodeMath'); g.operation = 'MULTIPLY'; g.inputs[1].default_value = gain
    nt.links.new(oi.outputs['Alpha'], g.inputs[0]); nt.links.new(g.outputs[0], b.inputs['Emission Strength'])
    return m
KEYM = keymat('keym', 14.0, 0.08)
SHM = keymat('shardm', 18.0, 0.05)
def unit_key():
    bpy.ops.mesh.primitive_cube_add(size=1); o = bpy.context.object; o.scale = (0.15, 0.31, 0.045)
    bpy.ops.object.transform_apply(scale=True); bev = o.modifiers.new('b', 'BEVEL'); bev.width = 0.012; bev.segments = 2
    return o.data
KEYMESH = unit_key(); bpy.data.objects.remove(bpy.context.object)
bpy.ops.mesh.primitive_cone_add(vertices=6, radius1=0.035, radius2=0.0, depth=1.0, location=(0, 0, 0.5))
o = bpy.context.object; SHMESH = o.data; bpy.data.objects.remove(o)
KEYS = []
for t, L, p, v, gb in LEAD:
    s = K * gb; c = hel(s, L); tg = tang(s, L)
    yaw = math.atan2(tg[1], tg[0]); pitch_ = -math.asin(tg[2])
    k_ = link(bpy.data.objects.new('key', KEYMESH)); k_.data.materials.append(KEYM) if not k_.data.materials else None
    k_.location = c; k_.rotation_euler = (0, pitch_, yaw)
    col = pitch_col(p, L)
    sh = link(bpy.data.objects.new('shard', SHMESH)); sh.data.materials.append(SHM) if not sh.data.materials else None
    inner = hel(s, L, R - HALF - 0.05); hgt = min(0.34, 0.1 + 0.013 * (p - 59 - 12 * L))
    sh.location = inner + np.array([0, 0, 0.02]); sh.scale = (1, 1, hgt); sh.rotation_euler = (0, 0, yaw)
    KEYS.append((t, k_, sh, col, c[2]))
for t, k_, sh, col, z0 in KEYS:
    k_.color = (*col, 0.0); sh.color = (*col, 0.0)
    for o in (k_, sh):
        for tt, a in ((0, 0.0), (t - 1 / FPS, 0.0), (t, 1.0), (t + 0.18, 0.35), (t + 0.9, 0.12)):
            if tt < 0: continue
            o.color[3] = a; o.keyframe_insert('color', index=3, frame=tt * FPS)
    for tt, dz in ((t - 1 / FPS, 0.0), (t + 0.02, -0.02), (t + 0.14, 0.0)):
        k_.location[2] = z0 + dz; k_.keyframe_insert('location', index=2, frame=tt * FPS)
    sh.scale[2] = sh.scale[2]
    for tt, sc_ in ((t - 1 / FPS, 1.0), (t + 0.03, 1.25), (t + 0.25, 1.0)):
        sh.scale[0] = sc_; sh.keyframe_insert('scale', index=0, frame=tt * FPS); sh.scale[1] = sc_; sh.keyframe_insert('scale', index=1, frame=tt * FPS)

# gates at each lane start (a little lever that drops when the marble is released)
LEADT = {L: min(x[0] for x in LEAD if x[1] == L) for L in range(3)}
GATES = []
for L in range(3):
    s0 = S_START[L] - 0.32; p = hel(s0 + 0.15, L); tg = tang(s0, L)
    piv = link(bpy.data.objects.new(f'gate{L}', None)); piv.location = hel(s0 + 0.15, L, R + HALF + 0.03) + np.array([0, 0, 0.05])
    piv.rotation_euler = (0, 0, math.atan2(tg[1], tg[0]))
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -0.17, 0.08)); g = bpy.context.object; g.scale = (0.03, 0.36, 0.16)
    g.data.materials.append(mat(f'gatem{L}', LANE_COL[L], 0.3, **{'Metallic': 0.7})); g.parent = piv
    piv.rotation_mode = 'XYZ'; GATES.append(piv)
    # stopper post + a little feeder cup above lane 2/3 gates
    rod(hel(s0 - 0.25, L, R) + np.array([0, 0, 0.03]), hel(s0 - 0.25, L, R) + np.array([0, 0, 0.25]), 0.03, RAIL, 8)

# ---------------- marbles ----------------
RB = 0.12
MARB = []
for L in range(3):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=RB, segments=24, ring_count=16); m_ = bpy.context.object
    for pg in m_.data.polygons: pg.use_smooth = True
    mm = mat(f'marb{L}', LANE_COL[L], 0.05, **{'Emission Color': (*LANE_COL[L], 1), 'Emission Strength': 9.0, 'Coat Weight': 1.0})
    m_.data.materials.append(mm)
    lt = bpy.data.lights.new(f'ml{L}', 'POINT'); lt.color = LANE_COL[L]; lt.energy = 14; lt.shadow_soft_size = 0.05
    lo = link(bpy.data.objects.new(f'ml{L}', lt)); lo.parent = m_
    MARB.append(m_)
# final ramps: from each lane end down to the crystal's cup
TF = E['final']; CUPZ = 2.62
RAMP = {}
for L in range(3):
    a = hel(S_END, L); th = math.atan2(a[1], a[0])
    b = np.array([0.32 * math.cos(th + 0.5), 0.32 * math.sin(th + 0.5), CUPZ + 0.05])
    RAMP[L] = (a, b)
    d = b - a; n = np.array([-d[1], d[0], 0]); n /= np.linalg.norm(n)
    for side in (-1, 1):
        curve_obj(f'ramp{L}{side}', [a + n * side * HALF + np.array([0, 0, 0.035]), b + n * side * 0.08 + np.array([0, 0, 0.035])], 0.022, RAIL)
    rod(a + np.array([0, 0, -0.02]), np.array([a[0], a[1], 0]), 0.03, DARKM, 6)
t_last = float(np.interp(LAST_BEAT, BEATS[:, 0], BEATS[:, 1]))
BPS0 = E['bpm'][0] / 60
def gbeat_ext(t): return gbeat(t) if t >= E['t0'] else (t - E['t0']) * BPS0
T_MOVE = {}
for L, m_ in enumerate(MARB):
    P = np.zeros((NF, 3)); vis = np.ones(NF)
    s_rest = S_START[L] - 0.34
    # ramp: helix to S_END, then straight to the cup; constant acceleration so it lands exactly on the final chord
    a_r, b_r = RAMP[L]; s0 = K * LAST_BEAT; d1 = S_END - s0; d2 = float(np.linalg.norm(b_r - a_r)); D = d1 + d2
    v0 = K * E['bpm'][-1] / 60; tau = TF - t_last; acc = 2 * (D - v0 * tau) / tau ** 2
    for f in range(NF):
        t = T[f]
        if t <= t_last:
            s = max(s_rest, K * gbeat_ext(t)); P[f] = hel(s, L) + np.array([0, 0, 0.0225 + RB])
            if L not in T_MOVE and s > s_rest + 1e-4: T_MOVE[L] = t
        elif t <= TF:
            x = v0 * (t - t_last) + 0.5 * acc * (t - t_last) ** 2
            if x <= d1: P[f] = hel(s0 + x, L) + np.array([0, 0, 0.0225 + RB])
            else: u_ = (x - d1) / d2; P[f] = a_r * (1 - u_) + b_r * u_ + np.array([0, 0, 0.0225 + RB])
        else:
            P[f] = b_r + np.array([0, 0, RB - 0.45 * min(1, (t - TF) / 0.25)]); vis[f] = 0 if t > TF + 0.3 else 1
    for i in range(3): key_series(m_, 'location', i, P[:, i])
    spin = np.cumsum(np.r_[0, np.linalg.norm(np.diff(P, axis=0), axis=1)]) / RB
    m_.rotation_mode = 'XYZ'; key_series(m_, 'rotation_euler', 0, spin * 0.6); key_series(m_, 'rotation_euler', 1, spin * 0.8)
    sc_ = np.where(vis > 0, 1.0, 0.0); key_series(m_, 'scale', 0, sc_); key_series(m_, 'scale', 1, sc_); key_series(m_, 'scale', 2, sc_)
for L, piv in enumerate(GATES):
    tr = T_MOVE.get(L, 0.5) - 0.14
    kf(piv, 'rotation_euler', 0, [(0, 0.0), (max(0, tr - 0.06), 0.0), (tr + 0.1, -1.5)])

# ---------------- the giant crystal (center, under the helix) ----------------
CRY = mat('crystal', (0.55, 0.3, 1.0), 0.03, **{'Transmission Weight': 0.85, 'IOR': 1.6, 'Emission Color': (0.6, 0.35, 1.0, 1), 'Emission Strength': 0.08, 'Roughness': 0.02})
bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=0.62, depth=2.2, location=(0, 0, 1.1)); gc = bpy.context.object; gc.data.materials.append(CRY)
bpy.ops.mesh.primitive_cone_add(vertices=6, radius1=0.62, radius2=0.42, depth=0.45, location=(0, 0, 2.42)); cup = bpy.context.object; cup.data.materials.append(CRY)
small = []
for i in range(9):
    a = i * 0.7 + rng.uniform(0, 0.3); r = 0.7 + rng.uniform(0, 0.5); h = rng.uniform(0.6, 1.5)
    bpy.ops.mesh.primitive_cone_add(vertices=6, radius1=0.16, radius2=0.0, depth=h, location=(r * math.cos(a), r * math.sin(a), h / 2 - 0.05),
                                    rotation=(rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35), 0))
    o = bpy.context.object; o.data.materials.append(CRY); small.append(o)
cn = CRY.node_tree.nodes['Principled BSDF'].inputs['Emission Strength']
for t, v in ((0, 0.08), (TF - 0.05, 0.3), (TF, 22.0), (TF + 0.4, 10.0), (TF + 2.5, 5.0), (TF + 4.4, 4.0)):
    cn.default_value = v; cn.keyframe_insert('default_value', frame=t * FPS)
cl = bpy.data.lights.new('crl', 'POINT'); cl.color = (0.7, 0.45, 1.0); cl.shadow_soft_size = 0.6
clo = link(bpy.data.objects.new('crl', cl)); clo.location = (0, 0, 2.9)
for t, v in ((0, 20), (TF - 0.05, 40), (TF, 2600), (TF + 0.5, 1300), (TF + 4.4, 900)): cl.energy = v; cl.keyframe_insert('energy', frame=t * FPS)
# burst of shards on the final chord
SHB = emis('burst', (0.75, 0.55, 1.0), 12.0)
for i in range(26):
    bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=0.05, radius2=0, depth=0.22); o = bpy.context.object; o.data.materials.append(SHB)
    a = rng.uniform(0, 2 * math.pi); el = rng.uniform(0.3, 1.2); sp = rng.uniform(3.0, 5.5)
    v0 = np.array([math.cos(a) * math.cos(el), math.sin(a) * math.cos(el), math.sin(el)]) * sp
    p0 = np.array([0, 0, CUPZ + 0.1])
    for tt in np.arange(0, 1.6, 1 / FPS):
        p = p0 + v0 * tt + np.array([0, 0, -4.9 * tt * tt])
        if p[2] < 0.05: p[2] = 0.05
        o.location = p; o.keyframe_insert('location', frame=(TF + tt) * FPS)
        o.rotation_euler = (tt * 7 + i, tt * 5, i); o.keyframe_insert('rotation_euler', frame=(TF + tt) * FPS)
    o.location = p0; o.keyframe_insert('location', frame=(TF - 1 / FPS) * FPS)
    for tt, s in ((0, 0.0), (TF - 1 / FPS, 0.0), (TF, 1.0), (TF + 1.4, 1.0), (TF + 1.7, 0.0)):
        o.scale = (s, s, s); o.keyframe_insert('scale', frame=tt * FPS)

# ---------------- percussion machines: hammers on pivots ----------------
STONE = mat('stone', (0.16, 0.14, 0.15), 0.85)
WOOD = mat('wood', (0.25, 0.14, 0.07), 0.7)
BRONZE = mat('bronze', (0.8, 0.5, 0.2), 0.25, **{'Metallic': 1.0})
def swing_curve(times, rest, cock, strike, dt_c=0.22):
    """angle per frame: rests at `rest`, rises to `cock` before each hit, slams to `strike` exactly at the hit"""
    a = np.full(NF, rest, float)
    ts = sorted(times)
    for i, th in enumerate(ts):
        prev = ts[i - 1] if i else -9
        c = min(dt_c, 0.6 * (th - prev))
        t1 = th - c
        m = (T >= t1) & (T < th); u = (T[m] - t1) / c
        lift = np.sin(np.clip(u / 0.55, 0, 1) * math.pi / 2)              # up
        down = np.clip((u - 0.55) / 0.45, 0, 1) ** 2                       # accelerate down
        a[m] = rest + (cock - rest) * lift * (1 - down) + (strike - rest) * down
        m2 = (T >= th) & (T < th + 0.2); u2 = (T[m2] - th) / 0.2
        a[m2] = np.minimum(a[m2], strike + (rest - strike) * np.sin(u2 * math.pi / 2)) if strike < rest else np.maximum(a[m2], strike + (rest - strike) * np.sin(u2 * math.pi / 2))
    return a
def drum_station(name, pos, face, rad, hgt, tilt_col, stone_ids):
    """a drum-stone on the floor with a hammer pivoting on an A-frame beside it; returns (hammer pivot, drum top obj)"""
    x, y = pos
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=rad, depth=hgt, location=(x, y, hgt / 2)); d = bpy.context.object
    d.data.materials.append(STONE); bv = d.modifiers.new('b', 'BEVEL'); bv.width = 0.05; bv.segments = 3
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=rad * 0.86, depth=0.03, location=(x, y, hgt + 0.01)); skin = bpy.context.object
    sm = glowmat(f'skin{name}', gain=8, base=0.05, rough=0.4); skin.data.materials.append(sm); skin.color = (*tilt_col, 0)
    # A-frame posts behind the drum (relative to `face` direction)
    ux, uy = math.cos(face), math.sin(face)          # from drum toward the pivot
    px, py = x + ux * (rad + 0.55), y + uy * (rad + 0.55); ph = hgt + 0.9
    vx, vy = -uy, ux
    for sgn in (-1, 1):
        rod((px + vx * 0.3 * sgn, py + vy * 0.3 * sgn, 0), (px + vx * 0.3 * sgn, py + vy * 0.3 * sgn, ph + 0.08), 0.04, WOOD, 8)
    rod((px - vx * 0.34, py - vy * 0.34, ph), (px + vx * 0.34, py + vy * 0.34, ph), 0.035, DARKM, 8)
    piv = link(bpy.data.objects.new(f'hp{name}', None)); piv.location = (px, py, ph); piv.rotation_mode = 'XYZ'
    piv.rotation_euler = (0, 0, face + math.pi)      # local +X points from pivot toward the drum centre
    L_ = rad + 0.55
    arm = rod((0, 0, 0), (L_, 0, 0), 0.05, WOOD, 8); arm.parent = piv
    bpy.ops.mesh.primitive_cube_add(size=1, location=(L_, 0, -0.15)); hd = bpy.context.object; hd.scale = (0.42, 0.4, 0.36)
    hd.data.materials.append(STONE); hd.parent = piv; bv = hd.modifiers.new('b', 'BEVEL'); bv.width = 0.04; bv.segments = 2
    # strike angle: hammer face touches drum top (pivot is 0.9 above top, head bottom 0.27 below arm)
    strike = math.asin(min(0.99, (0.9 - 0.33) / L_))
    return piv, skin, strike
ST = {}
ST[0] = drum_station('bd', (-2.45, -3.0), math.radians(90), 0.8, 0.75, (1.0, 0.4, 0.15), [0])
ST[1] = drum_station('t1', (2.45, -3.0), math.radians(90), 0.62, 0.95, (1.0, 0.75, 0.3), [1])
ST[2] = drum_station('t2', (0.0, -3.95), math.radians(90), 0.5, 0.85, (0.5, 0.8, 1.0), [2])
for sid, (piv, skin, strike) in ST.items():
    ts = [h[0] for h in HITS if h[1] == sid]
    # positive y-rotation tips +X (head) down; rest slightly raised, cock high, strike = touching
    a = swing_curve(ts, -0.45, -1.0, strike)
    key_series(piv, 'rotation_euler', 1, a)
    al = np.zeros(NF)
    for th, s_, r_, p_, v_ in [h for h in HITS if h[1] == sid]:
        al = np.maximum(al, np.where(T >= th, np.exp(-(T - th) / 0.2) * (v_ / 127), 0))
    key_series(skin, 'color', 3, al, 2e-3)
# gong (crash) hanging in a frame at the back-left, struck by a hammer
GX, GY = -2.6, 3.4
for sgn in (-1, 1): rod((GX + sgn * 1.0, GY, 0), (GX + sgn * 1.0, GY, 3.1), 0.06, WOOD, 8)
rod((GX - 1.1, GY, 3.05), (GX + 1.1, GY, 3.05), 0.06, WOOD, 8)
gpiv = link(bpy.data.objects.new('gongpiv', None)); gpiv.location = (GX, GY, 3.0); gpiv.rotation_mode = 'XYZ'
for sgn in (-1, 1): c_ = rod((sgn * 0.3, 0, 0), (sgn * 0.3, 0, -0.55), 0.01, DARKM, 6); c_.parent = gpiv
bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.8, depth=0.05, location=(0, 0, -1.35), rotation=(math.pi / 2, 0, 0)); gg = bpy.context.object
gg.data.materials.append(BRONZE); gg.parent = gpiv
bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.22, depth=0.07, location=(0, -0.02, -1.35), rotation=(math.pi / 2, 0, 0)); bpy.context.object.data.materials.append(BRONZE); bpy.context.object.parent = gpiv
gts = [h[0] for h in HITS if h[1] == 4] + [TF]
wob = np.zeros(NF)
for th in gts: wob += np.where(T >= th, np.sin((T - th) * 11) * np.exp(-(T - th) * 1.6), 0) * 0.12
key_series(gpiv, 'rotation_euler', 0, wob)
# gong beater: pendulum mallet hanging from a separate arm in front of the gong
mpiv = link(bpy.data.objects.new('malpiv', None)); mpiv.location = (GX, GY - 1.0, 3.0); mpiv.rotation_mode = 'XYZ'
rod((GX, GY - 1.0, 3.05), (GX, GY, 3.05), 0.04, WOOD, 8)
ma = rod((0, 0, 0), (0, 0, -1.25), 0.025, WOOD, 8); ma.parent = mpiv
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.13, location=(0, 0, -1.33)); mh = bpy.context.object; mh.data.materials.append(mat('felt', (0.6, 0.08, 0.06), 0.9)); mh.parent = mpiv
# hits when the mallet swings forward (+Y) to touch the gong (0.9 away from pivot line)
gstrike = math.asin(0.84 / 1.33)
key_series(mpiv, 'rotation_euler', 0, swing_curve(gts, 0.0, -0.6, gstrike, dt_c=0.35))
# stalactite chimes from the cave ceiling + pendulum striker
CHX, CHY, CHZ = -1.3, 1.6, 13.2
CHM = glowmat('chm', gain=10, base=0.25, rough=0.05)
CH = []
for i in range(5):
    x = CHX + (i - 2) * 0.32; ln = 1.4 + 0.35 * abs(i - 2) * -1 + 0.5
    bpy.ops.mesh.primitive_cone_add(vertices=6, radius1=0.09, radius2=0.0, depth=ln, location=(x, CHY, CHZ - ln / 2), rotation=(math.pi, 0, 0))
    o = bpy.context.object; o.data.materials.append(CHM); o.color = (0.6, 0.85, 1.0, 0); CH.append(o)
cts = [h[0] for h in HITS if h[1] == 3] + [TF]
spiv = link(bpy.data.objects.new('chpiv', None)); spiv.location = (CHX, CHY - 0.8, CHZ); spiv.rotation_mode = 'XYZ'
sa = rod((0, 0, 0), (0, 0, -1.1), 0.015, RAIL, 6); sa.parent = spiv
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.08, location=(0, 0, -1.15)); sb = bpy.context.object; sb.data.materials.append(RAIL); sb.parent = spiv
cstrike = math.asin(min(0.99, 0.72 / 1.15))
key_series(spiv, 'rotation_euler', 0, swing_curve(cts, 0.0, -0.5, cstrike, dt_c=0.4))
al = np.zeros(NF)
for th in cts: al = np.maximum(al, np.where(T >= th, np.exp(-(T - th) / 0.6), 0))
for o in CH: key_series(o, 'color', 3, al, 2e-3)

# ---------------- the cave ----------------
ROCK = mat('rock', (0.11, 0.09, 0.1), 0.9)
nt = ROCK.node_tree; b = nt.nodes['Principled BSDF']
nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 3.0; nz.inputs['Detail'].default_value = 8
cr = nt.nodes.new('ShaderNodeValToRGB'); cr.color_ramp.elements[0].color = (0.04, 0.035, 0.045, 1); cr.color_ramp.elements[1].color = (0.2, 0.17, 0.18, 1)
bm_ = nt.nodes.new('ShaderNodeBump'); bm_.inputs['Strength'].default_value = 0.6
nt.links.new(nz.outputs['Fac'], cr.inputs['Fac']); nt.links.new(cr.outputs['Color'], b.inputs['Base Color'])
nt.links.new(nz.outputs['Fac'], bm_.inputs['Height']); nt.links.new(bm_.outputs['Normal'], b.inputs['Normal'])
bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0)); fl = bpy.context.object; fl.data.materials.append(ROCK)
m = fl.modifiers.new('sub', 'SUBSURF'); m.levels = 4; m.subdivision_type = 'SIMPLE'
m = fl.modifiers.new('d', 'DISPLACE'); tx = bpy.data.textures.new('fl', 'CLOUDS'); tx.noise_scale = 1.5; m.texture = tx; m.strength = 0.25; m.mid_level = 0.8
bpy.ops.mesh.primitive_uv_sphere_add(radius=12, segments=48, ring_count=32, location=(0, 3.0, 4.5)); cave = bpy.context.object
cave.scale = (1.0, 0.8, 0.95); cave.data.materials.append(ROCK)
bm = bmesh.new(); bm.from_mesh(cave.data)
bmesh.ops.delete(bm, geom=[f_ for f_ in bm.faces if f_.calc_center_median().y < -2.2], context='FACES'); bm.to_mesh(cave.data); bm.free()
for pg in cave.data.polygons: pg.use_smooth = True
m = cave.modifiers.new('d', 'DISPLACE'); tx = bpy.data.textures.new('cv', 'CLOUDS'); tx.noise_scale = 2.2; m.texture = tx; m.strength = 1.6
cave.visible_camera = True
# stalactites & stalagmites + glowing wall crystals
for i in range(60):
    a = rng.uniform(0, 2 * math.pi); r = rng.uniform(4.5, 9)
    x, y = r * math.cos(a), 3.0 + 0.8 * r * math.sin(a)
    ln = rng.uniform(0.8, 3.0)
    if (i % 2 and y < 1.5) or y < -4: continue
    up = i % 2
    z0 = 0 if up else 14.5 - abs(x) * 0.25
    bpy.ops.mesh.primitive_cone_add(vertices=7, radius1=rng.uniform(0.15, 0.5), radius2=0.0, depth=ln, location=(x, y, z0 + (ln / 2 if up else -ln / 2)),
                                    rotation=(0 if up else math.pi, 0, 0))
    bpy.context.object.data.materials.append(ROCK)
GEMC = [(0.5, 0.3, 1.0), (0.2, 0.7, 1.0), (1.0, 0.35, 0.8)]
for i in range(40):
    a = rng.uniform(0.3, math.pi - 0.3) if i % 3 else rng.uniform(0, 2 * math.pi); r = rng.uniform(5, 8.5)
    x, y = r * math.cos(a) * (1 if i % 2 else -1), 3.0 + r * math.sin(a) * 0.8
    z = rng.uniform(0, 1.2)
    if y < 1.0: continue
    h = rng.uniform(0.3, 0.9)
    bpy.ops.mesh.primitive_cone_add(vertices=6, radius1=0.09 * h * 2, radius2=0.0, depth=h, location=(x, y, z), rotation=(rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), 0))
    bpy.context.object.data.materials.append(emis(f'gem{i}', GEMC[i % 3], 3.0))

# ---------------- world & lights ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs['Color'].default_value = (0.004, 0.004, 0.008, 1)
for nm, loc, e, col, sz in (('key', (-6, -9, 12), 900, (0.65, 0.7, 1.0), 6), ('rim', (6, 4, 9), 500, (0.8, 0.45, 1.0), 5),
                             ('fill', (2, -8, 2), 150, (1.0, 0.6, 0.35), 4)):
    l_ = bpy.data.lights.new(nm, 'AREA'); l_.energy = e; l_.color = col; l_.size = sz
    o = link(bpy.data.objects.new(nm, l_)); o.location = loc
    tgt0 = link(bpy.data.objects.new(nm + 't', None)); tgt0.location = (0, 0, 5)
    c = o.constraints.new('TRACK_TO'); c.target = tgt0; c.track_axis = 'TRACK_NEGATIVE_Z'; c.up_axis = 'UP_Y'
    if nm == 'key':
        for t, v in ((0, 900), (TF - 0.05, 900), (TF + 0.1, 1400), (TF + 4, 1300)): l_.energy = v; l_.keyframe_insert('energy', frame=t * FPS)

# ---------------- camera: wide, slow, whole machine in view ----------------
camd = bpy.data.cameras.new('cam'); camd.lens = 35; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
lead_z = np.array([hel(K * gbeat(min(t, t_last)), 0)[2] for t in T])
u = np.clip(T / TF, 0, 1)
H = 15.6 - 1.2 * np.sin(u * math.pi) ** 1.5           # visible height: whole tower, gently closer mid-piece
fin = np.clip((T - (TF - 1.2)) / 2.5, 0, 1); fin = fin * fin * (3 - 2 * fin)
H = H * (1 - fin) + 16.2 * fin
zc = 5.9 + 0.2 * (lead_z - 6.6)                       # follow the leading marble just a little
zc = smooth(zc, 1.2) * (1 - fin) + 6.0 * fin
az = math.radians(-100) + math.radians(32) * (u * u * (3 - 2 * u))
dist = H * 35 / 24
CP = np.stack([dist * np.cos(az), dist * np.sin(az), zc + 1.6], 1); TP = np.stack([0 * zc, 0.4 + 0 * zc, zc], 1)
for i in range(3): key_series(cam, 'location', i, CP[:, i]); key_series(tgt, 'location', i, TP[:, i])

print('objects', len(bpy.data.objects), 'NF', NF, 'keys', len(KEYS), 'ZTOP', round(ZTOP, 2))
if MODE == 'test':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS)); sc.frame_set(f)
        sc.render.filepath = f'{HERE}/test_{f:04d}.png'
        t1 = time.time(); bpy.ops.render.render(write_still=True); print('frame', f, 'sec', round(time.time() - t1, 1))
elif MODE == 'range':
    sc.frame_start, sc.frame_end = int(args[1]), int(args[2])
    sc.render.image_settings.file_format = 'PNG'; sc.render.filepath = f'{HERE}/frames/f_'
    bpy.ops.render.render(animation=True)
