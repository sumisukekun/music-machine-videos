"""William Tell galop - an Alpine switchback railway.
A train of four marbles gallops down a zig-zag track cut into a cliff face. The track is paved with tuned key-sleepers:
one key per melody note, so the locomotive ball strikes each key exactly when the note sounds (keys light up and dip).
Fanfare at the summit station, waterfall beside the cliff, and at the bottom the train crosses a stone-arch bridge over
a gorge; on the final chord a crossbow on the far cliff fires an arrow through the apple on the buffer stop."""
import os, sys, json, math, time, colorsys
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix, Quaternion

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.5) * FPS); T = np.arange(NF) / FPS
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
sc.cycles.max_bounces = 4; sc.cycles.diffuse_bounces = 1; sc.cycles.glossy_bounces = 2; sc.cycles.transmission_bounces = 3
sc.render.resolution_x, sc.render.resolution_y = 1080, 1920
sc.render.resolution_percentage = int(os.environ.get('PCT', 50)); sc.render.fps = FPS
sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.3
try: sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Medium High Contrast'
except Exception: pass
sc.frame_start, sc.frame_end = 0, NF - 1


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
    m = mat(name, (1, 1, 1), rough, **{'Coat Weight': 0.6, 'Metallic': 0.3}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
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


def box(loc, size, mt, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=rot)
    o = bpy.context.object; o.scale = size; o.data.materials.append(mt); return o


def smoothstep(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


# ================= the track path =================
V = 1.45                        # galop speed (units / s)
T_GO = 0.35                     # the train leaves the station
NROW = 12; RT = 0.7             # rows, turn radius
H = 1.6                         # drop per row + turn
t_fin = LEAD[-1][0]


def s_of_t(t):
    t = np.asarray(t, float)
    a = np.clip(t - T_GO, 0, None)
    s = V * np.where(a < 0.4, a * a / 0.8, a - 0.2)          # gentle start (0 -> V over 0.4 s)
    sf_ = V * (t_fin - T_GO - 0.2)
    tau = np.clip(t - t_fin, 0, None); TB = 2 * 0.62 / V      # brake into the buffer after the final note
    return np.where(t > t_fin, sf_ + V * (np.minimum(tau, TB) - np.minimum(tau, TB) ** 2 / (2 * TB)), s)


S_FIN = float(s_of_t(t_fin)); S_STOP = float(s_of_t(t_fin + 5))
# solve row length W so that the final note lands 2.4 units right of the bridge centre on the last row
X_FIN = 0.9
W = (S_FIN - (NROW - 1) * math.pi * RT - X_FIN) / (NROW - 1 + 0.5)
LROW = W + math.pi * RT; G = H / LROW
print('W', round(W, 2), 'S_FIN', round(S_FIN, 1))


def path(s):
    """position on the centre line + unit tangent for arc length s (s<0: flat station siding)"""
    if s < 0:
        return np.array([-W / 2 + s, 0.0, 0.0]), np.array([1.0, 0, 0])
    k = int(s // LROW); u = s - k * LROW
    z0 = -k * H
    if k >= NROW - 1:                 # last row continues straight (bridge, buffer stop)
        k = NROW - 1; u = s - k * LROW; z0 = -k * H
        return np.array([-W / 2 + u, 0.0, z0 - G * u]), np.array([1.0, 0, -G]) / math.hypot(1, G)
    even = k % 2 == 0; y = 0.0 if even else -2 * RT
    if u <= W:
        x = (-W / 2 + u) if even else (W / 2 - u)
        d = np.array([1.0 if even else -1.0, 0, -G]); return np.array([x, y, z0 - G * u]), d / np.linalg.norm(d)
    v = (u - W) / (math.pi * RT)       # 0..1 around the half turn
    z = z0 - G * u
    if even:
        th = math.pi / 2 - math.pi * v; c = (W / 2, -RT)
        p = np.array([c[0] + RT * math.cos(th), c[1] + RT * math.sin(th), z])
        d = np.array([math.sin(th), -math.cos(th), 0.0]) * math.pi * RT
    else:
        th = -math.pi / 2 - math.pi * v; c = (-W / 2, -RT)
        p = np.array([c[0] + RT * math.cos(th), c[1] + RT * math.sin(th), z])
        d = np.array([math.sin(th), -math.cos(th), 0.0]) * math.pi * RT
    d = np.array([d[0], d[1], -G * math.pi * RT]); return p, d / np.linalg.norm(d)


def frame(s):
    p, tg = path(s); up = np.array([0, 0, 1.0])
    lat = np.cross(up, tg); lat /= np.linalg.norm(lat); upn = np.cross(tg, lat)
    return p, tg, lat, upn


Z_BR = -(NROW - 1) * H            # bridge row start height
Z_END = float(path(S_STOP)[0][2])

# ================= materials =================
ROCK = mat('rock', (0.32, 0.30, 0.29), 0.85)
nt = ROCK.node_tree; b_ = nt.nodes['Principled BSDF']
nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 1.4; nz.inputs['Detail'].default_value = 8
cr = nt.nodes.new('ShaderNodeValToRGB'); cr.color_ramp.elements[0].color = (0.2, 0.16, 0.13, 1); cr.color_ramp.elements[1].color = (0.52, 0.45, 0.38, 1)
nt.links.new(nz.outputs['Fac'], cr.inputs['Fac']); nt.links.new(cr.outputs['Color'], b_.inputs['Base Color'])
bmp = nt.nodes.new('ShaderNodeBump'); bmp.inputs['Strength'].default_value = 0.5
nt.links.new(nz.outputs['Fac'], bmp.inputs['Height']); nt.links.new(bmp.outputs['Normal'], b_.inputs['Normal'])
SNOW = mat('snow', (0.88, 0.91, 0.97), 0.55, **{'Subsurface Weight': 0.15})
STEEL = mat('steel', (0.55, 0.56, 0.6), 0.3, **{'Metallic': 1.0})
DARK = mat('dark', (0.08, 0.08, 0.09), 0.4, **{'Metallic': 0.8})
WOODM = mat('wood', (0.35, 0.2, 0.1), 0.6)
STONE = mat('stone', (0.55, 0.52, 0.48), 0.8)
PINE = mat('pine', (0.04, 0.17, 0.08), 0.75)
RED = mat('red', (0.75, 0.06, 0.05), 0.4)
BRASS = mat('brass', (0.95, 0.7, 0.3), 0.25, **{'Metallic': 1.0})

# ================= rails, stringers, keys =================
SS = np.arange(-1.6, S_STOP + 0.9, 0.08)
FR = [frame(s) for s in SS]


def curve(name, offs, bev, mt):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'; cu.bevel_depth = bev; cu.bevel_resolution = 2
    sp = cu.splines.new('POLY'); sp.points.add(len(SS) - 1)
    for i, (p, tg, lat, up) in enumerate(FR):
        q = p + lat * offs[0] + up * offs[1]; sp.points[i].co = (*q, 1)
    o = link(bpy.data.objects.new(name, cu)); o.data.materials.append(mt); return o


GAUGE = 0.12
for side in (-1, 1):
    curve(f'rail{side}', (side * GAUGE, 0.02), 0.014, STEEL)        # running rails (ball sits on them)
    curve(f'guard{side}', (side * 0.25, 0.12), 0.01, BRASS)          # guard rails
    curve(f'str{side}', (side * 0.27, -0.1), 0.03, DARK)             # stringers under the keys
# guard-rail posts
for s in np.arange(-1.5, S_STOP + 0.8, 0.6):
    p, tg, lat, up = frame(s)
    for side in (-1, 1):
        rod(p + lat * side * 0.27 - up * 0.08, p + lat * side * 0.25 + up * 0.12, 0.008, BRASS, 6)

R_BALL = 0.18
BALL_UP = math.sqrt(R_BALL ** 2 - GAUGE ** 2) + 0.02 + 0.014      # ball centre above the centre line
KEYM = glowmat('key', 9.0, 0.25)
pitches = sorted(set(l[1] for l in LEAD)); pmin, pmax = min(pitches), max(pitches)
KEYS = []
for i, (t, p, v, d) in enumerate(LEAD):
    s = float(s_of_t(t)); P_, tg, lat, up = frame(s)
    c = P_ + up * (BALL_UP - R_BALL - 0.004 - 0.025)              # key top just under the ball's lowest point
    bpy.ops.mesh.primitive_cube_add(size=1); o = bpy.context.object
    M = Matrix(((tg[0], lat[0], up[0]), (tg[1], lat[1], up[1]), (tg[2], lat[2], up[2])))
    o.matrix_world = Matrix.Translation(Vector(c)) @ M.to_4x4() @ Matrix.Diagonal((0.085, 0.58, 0.05, 1))
    o.modifiers.new('b', 'BEVEL').width = 0.008
    o.data.materials.append(KEYM)
    hue = 0.02 + 0.75 * (p - pmin) / max(1, pmax - pmin)
    col = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
    KEYS.append((o, t, col, s))
bpy.context.view_layer.update()
for o, t, col, s in KEYS:
    a = np.where(T >= t - 0.5 / FPS, 0.10 + 0.9 * np.exp(-np.clip(T - t, 0, None) / 0.3), 0.0)
    # finale: a wave of light runs back up the mountain from the apple
    a = np.maximum(a, np.where(T >= FINAL + (S_FIN - s) / 40.0, 0.8 * np.exp(-np.clip(T - FINAL - (S_FIN - s) / 40.0, 0, None) / 0.8) + 0.3, 0))
    key_alpha(o, a, col)
    dz = -0.02 * np.exp(-np.clip(T - t, 0, None) / 0.07) * (T >= t - 0.5 / FPS)
    key_series(o, 'location', 2, o.location.z + dz)

# ================= supports: cantilever arms back into the cliff =================
CLIFF_Y = 1.05
ARMS = []
for s in np.arange(0.4, S_FIN - LROW * 0.0, 1.25):
    p, tg, lat, up = frame(s)
    if s > (NROW - 1) * LROW - 0.5: break              # the bridge row has its own piers
    a0 = np.array([p[0], CLIFF_Y + 0.2, p[2] - 0.13]); a1 = np.array([p[0], p[1] - 0.3, p[2] - 0.13])
    ok = True
    for s2 in np.arange(max(0, s - 30), min(S_FIN, s + 30), 0.1):
        if abs(s2 - s) < 1.0: continue
        q = path(s2)[0]
        if abs(q[0] - p[0]) < 0.45 and min(a0[1], a1[1]) - 0.4 < q[1] < max(a0[1], a1[1]) + 0.4 and -0.35 < q[2] - a0[2] < 0.55: ok = False; break
    if ok:
        rod(a1, a0, 0.022, STEEL, 8)
        ARMS.append(s)

# ================= the cliff, peaks, trees =================
ZTOP = 1.6
bpy.ops.mesh.primitive_grid_add(x_subdivisions=120, y_subdivisions=230, size=1)
cl = bpy.context.object; cl.data.materials.append(ROCK)
Z_RIV = -(NROW - 1) * H - 7.5
bm = bmesh.new(); bm.from_mesh(cl.data)
for v_ in bm.verts:
    x = v_.co.x * 22; z = ZTOP - (v_.co.y + 0.5) * (ZTOP - Z_RIV + 0.5)
    n = (math.sin(x * 1.3 + z * 0.4) * 0.5 + math.sin(z * 2.1 + x * 0.3) * 0.35 + rng.random() * 0.25)
    v_.co = Vector((x, CLIFF_Y + 0.25 + 0.35 * n + 0.04 * (z % 2.4 < 0.25), z))
bm.to_mesh(cl.data); cl.data.update()
for p_ in cl.data.polygons: p_.use_smooth = True
# snow cap + summit plateau behind the station
box((0, CLIFF_Y + 2.5, ZTOP - 0.3), (26, 4, 1.4), ROCK)
box((0, CLIFF_Y + 2.6, ZTOP + 0.42), (26, 3.8, 0.08), SNOW)
for i in range(7):     # distant peaks
    x = -16 + i * 5.5 + rng.uniform(-1, 1); hh = rng.uniform(9, 15); r = rng.uniform(5, 8)
    bpy.ops.mesh.primitive_cone_add(vertices=7, radius1=r, depth=hh, location=(x * 2.2, 55 + rng.uniform(0, 10), ZTOP - 4 + hh / 2))
    o = bpy.context.object; o.scale = (2.2, 2.2, 1.6); o.data.materials.append(mat(f'peak{i}', (0.42, 0.48, 0.6), 0.9))
    bpy.ops.mesh.primitive_cone_add(vertices=7, radius1=r * 0.38, depth=hh * 0.38, location=(x * 2.2, o.location.y, ZTOP - 4 + (hh - hh * 0.19) * 1.6))
    bpy.context.object.scale = (2.2, 2.2, 1.6); bpy.context.object.data.materials.append(SNOW)


def pine(x, y, z, s=1.0):
    rod((x, y, z), (x, y, z + 0.35 * s), 0.05 * s, WOODM, 6)
    for j in range(3):
        bpy.ops.mesh.primitive_cone_add(vertices=8, radius1=(0.42 - j * 0.1) * s, depth=0.6 * s, location=(x, y, z + (0.45 + j * 0.32) * s))
        bpy.context.object.data.materials.append(PINE)


for x in (-8.5, -7.2, -6.0, 5.2, 6.4, 7.6, 8.8):
    pine(x, CLIFF_Y + 1.2 + rng.uniform(0, 1.5), ZTOP + 0.5, rng.uniform(0.9, 1.4))

# summit station: platform, roof, bell
SX = -W / 2 - 1.0
box((SX, 0, -0.3), (2.6, 1.3, 0.25), STONE)
for dx in (-1.1, 1.1):
    for dy in (-0.5, 0.5): rod((SX + dx, dy, -0.18), (SX + dx, dy, 1.25), 0.04, WOODM, 6)
box((SX, 0, 1.32), (2.8, 1.5, 0.12), RED, rot=(0.12, 0, 0))
box((SX - 1.0, 0.5, 0.45), (0.5, 0.25, 1.1), mat('sign', (0.9, 0.85, 0.7), 0.6))

# ================= bridge, gorge, river =================
Z_RIV = Z_BR - 7.5
X_BR0, X_BR1 = -W / 2 - 0.5, W / 2 + 1.6
zb = lambda x: Z_BR - G * (x + W / 2)
# deck under the bridge row
for x in np.arange(X_BR0, X_BR1, 0.5):
    box((x + 0.25, 0, zb(x + 0.25) - 0.22), (0.52, 0.8, 0.12), STONE, rot=(0, math.atan(G), 0))
# arches + piers
NA = 4; span = (X_BR1 - X_BR0) / NA
for i in range(NA + 1):
    x = X_BR0 + i * span
    box((x, 0, (zb(x) - 0.3 + Z_RIV) / 2), (0.45, 0.75, zb(x) - 0.3 - Z_RIV), STONE)
for i in range(NA):
    xa = X_BR0 + i * span; xc = xa + span / 2
    for j in range(16):
        x = xa + (j + 0.5) * span / 16; a = math.acos(max(-1, min(1, (x - xc) / (span / 2))))
        zsp = zb(xc) - 0.3 - span * 0.55                         # springing line
        zz = zsp + span * 0.45 * math.sin(a)                    # arch intrados: highest at the crown
        box((x, 0, (zz + zb(x) - 0.28) / 2), (span / 16 + 0.02, 0.62, max(0.08, zb(x) - 0.28 - zz)), STONE)
# gorge walls and river
box((-W / 2 - 5.2, 0.5, (Z_BR + Z_RIV) / 2 - 0.5), (6, 6, Z_BR - Z_RIV + 1), ROCK)
box((W / 2 + 5.0, 0.5, (Z_BR + Z_RIV) / 2 - 1.5), (6, 6, Z_BR - Z_RIV + 1), ROCK)
WATER = mat('water', (0.03, 0.12, 0.13), 0.12)
box((0, -2, Z_RIV), (30, 14, 0.1), WATER)
for x in (-9.0, -7.6, 7.2, 8.6, 9.8):
    pine(x, rng.uniform(-1.5, 1.5), Z_BR + (0.1 if x < 0 else -2.8), rng.uniform(1.0, 1.5))

# waterfall down the left of the cliff
FALL = bpy.data.materials.new('fall'); FALL.use_nodes = True; nt = FALL.node_tree; nt.nodes.clear()
outm = nt.nodes.new('ShaderNodeOutputMaterial'); mix = nt.nodes.new('ShaderNodeMixShader')
tr_ = nt.nodes.new('ShaderNodeBsdfTransparent'); em_ = nt.nodes.new('ShaderNodeEmission'); em_.inputs['Color'].default_value = (0.75, 0.88, 1.0, 1); em_.inputs['Strength'].default_value = 0.9
tc = nt.nodes.new('ShaderNodeTexCoord'); mp = nt.nodes.new('ShaderNodeMapping'); nzf = nt.nodes.new('ShaderNodeTexNoise')
nzf.inputs['Scale'].default_value = 3.0; nzf.inputs['Detail'].default_value = 6
mp.inputs['Scale'].default_value = (5.0, 1.0, 0.7)
crf = nt.nodes.new('ShaderNodeValToRGB'); crf.color_ramp.elements[0].position = 0.45; crf.color_ramp.elements[1].position = 0.7
nt.links.new(tc.outputs['Object'], mp.inputs['Vector']); nt.links.new(mp.outputs['Vector'], nzf.inputs['Vector'])
nt.links.new(nzf.outputs['Fac'], crf.inputs['Fac']); nt.links.new(crf.outputs['Color'], mix.inputs['Fac'])
nt.links.new(tr_.outputs[0], mix.inputs[1]); nt.links.new(em_.outputs[0], mix.inputs[2]); nt.links.new(mix.outputs[0], outm.inputs['Surface'])
mp.inputs['Location'].default_value[2] = 0; mp.inputs['Location'].keyframe_insert('default_value', index=2, frame=0)
mp.inputs['Location'].default_value[2] = 4.0 * NF / FPS; mp.inputs['Location'].keyframe_insert('default_value', index=2, frame=NF)
XF = W / 2 + RT + 1.3
bpy.ops.mesh.primitive_plane_add(size=1, location=(XF, CLIFF_Y + 0.15, (ZTOP + Z_RIV) / 2), rotation=(math.pi / 2, 0, 0))
wf = bpy.context.object; wf.scale = (1.1, ZTOP - Z_RIV, 1); wf.data.materials.append(FALL)
bpy.ops.mesh.primitive_plane_add(size=1, location=(XF + 0.05, CLIFF_Y + 0.1, (ZTOP + Z_RIV) / 2), rotation=(math.pi / 2, 0, 0))
wf2 = bpy.context.object; wf2.scale = (0.7, ZTOP - Z_RIV, 1); wf2.data.materials.append(FALL)

# ================= signal lamps at the hairpins: pulse on the off-beat chords =================
LAMP = bpy.data.materials.new('lamp'); LAMP.use_nodes = True; nt = LAMP.node_tree; nt.nodes.clear()
le = nt.nodes.new('ShaderNodeEmission'); le.inputs['Color'].default_value = (1.0, 0.65, 0.25, 1)
nt.links.new(le.outputs[0], nt.nodes.new('ShaderNodeOutputMaterial').inputs['Surface'])
for k in range(NROW - 1):
    s = k * LROW + W + math.pi * RT / 2; p, tg, lat, up = frame(s)
    out = np.array([p[0] - (W / 2 if k % 2 else -W / 2), p[1] + RT, 0]); out[2] = 0
    out = out / (np.linalg.norm(out) + 1e-9)
    base = p + out * 0.42
    rod(base - np.array([0, 0, 0.1]), base + np.array([0, 0, 0.45]), 0.02, DARK, 6)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.07, location=base + np.array([0, 0, 0.5])); bpy.context.object.data.materials.append(LAMP)
off = [b + E['q'] / 2 for b in E['beats'] if E['secs'][1][1] - 0.01 <= b <= FINAL]
lv = np.full(NF, 2.0)
for tb in off: lv = np.maximum(lv, np.where(T >= tb, 2.0 + 14 * np.exp(-np.clip(T - tb, 0, None) / 0.09), 2.0))
lv = np.where(T >= FINAL, 22.0, lv)
for f in range(NF):
    if 0 < f < NF - 1 and abs(lv[f] - lv[f - 1]) < 0.05 and abs(lv[f] - lv[f + 1]) < 0.05: continue
    le.inputs['Strength'].default_value = float(lv[f]); le.inputs['Strength'].keyframe_insert('default_value', frame=f)

# ================= the train: four marbles =================
BCOL = [((0.85, 0.08, 0.05), 'loco'), ((0.1, 0.35, 0.9), 'c1'), ((0.1, 0.7, 0.3), 'c2'), ((0.95, 0.75, 0.1), 'c3')]
GAP = 0.4
TRAIN = []
for i, (col, nm) in enumerate(BCOL):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=R_BALL, segments=32, ring_count=16)
    b = bpy.context.object; b.name = nm
    for p_ in b.data.polygons: p_.use_smooth = True
    if i == 0: b.data.materials.append(mat('locom', col, 0.08, **{'Coat Weight': 1.0, 'Emission Color': (1, 0.2, 0.1, 1), 'Emission Strength': 0.6}))
    else: b.data.materials.append(mat(f'car{i}', col, 0.05, **{'Coat Weight': 1.0, 'Transmission Weight': 0.6}))
    TRAIN.append(b)
# headlamp on the locomotive
hl = bpy.data.lights.new('head', 'POINT'); hl.energy = 6; hl.color = (1.0, 0.85, 0.6); hl.shadow_soft_size = 0.05
HL = link(bpy.data.objects.new('head', hl))
SB = s_of_t(T)
LOCO = np.zeros((NF, 3))
for i, b in enumerate(TRAIN):
    pos = np.zeros((NF, 3)); q = Quaternion(); qs = []
    for f in range(NF):
        s = SB[f] - i * GAP; p, tg, lat, up = frame(s); pos[f] = p + up * BALL_UP
        if f:
            ds = np.linalg.norm(pos[f] - pos[f - 1])
            if ds > 1e-6: q = Quaternion(Vector(-lat), ds / R_BALL) @ q
        qs.append(q.copy())
    for k in range(3): key_series(b, 'location', k, pos[:, k])
    b.rotation_mode = 'QUATERNION'
    for f in range(NF):
        if f % 2 and f != NF - 1: continue
        b.rotation_quaternion = qs[f]; b.keyframe_insert('rotation_quaternion', frame=f)
    if i == 0: LOCO = pos
for k in range(3): key_series(HL, 'location', k, LOCO[:, k] + np.array([0, 0, 0.35])[k])

# ================= buffer stop, apple, crossbow =================
pB, tgB, latB, upB = frame(S_STOP + R_BALL + 0.04)
box(pB + upB * 0.12 + tgB * 0.06, (0.12, 0.62, 0.3), RED)
for side in (-1, 1): rod(pB + latB * side * 0.25 - upB * 0.1, pB + latB * side * 0.25 + upB * 0.35, 0.03, DARK, 6)
APX = pB + tgB * 0.35
rod(APX - np.array([0, 0, 0.15]), APX + np.array([0, 0, 1.05]), 0.035, WOODM, 8)
AP0 = APX + np.array([0, 0, 1.28])
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.22, segments=24, ring_count=12, location=AP0); APPLE = bpy.context.object
for p_ in APPLE.data.polygons: p_.use_smooth = True
APPLE.data.materials.append(mat('apple', (0.75, 0.04, 0.03), 0.25, **{'Coat Weight': 0.8}))
APPLE.scale = (1, 1, 0.9)
st = rod((0, 0, 0.17), (0.03, 0, 0.3), 0.012, WOODM, 6); st.parent = APPLE; st.matrix_parent_inverse = APPLE.matrix_world.inverted()
# crossbow on a rock ledge across the gorge, aimed along the bridge at the apple
CB0 = np.array([-W / 2 - RT - 1.3, 0.0, 0.0]); CB0[2] = AP0[2] + G * (AP0[0] - CB0[0])
aim = AP0 - CB0; L_AIM = np.linalg.norm(aim); aimn = aim / L_AIM
box((CB0[0] - 0.4, 0.2, CB0[2] - 1.2), (1.8, 1.8, 1.2), STONE)
rod(CB0 - np.array([0, 0, 0.62]), CB0 - np.array([0, 0, 0.08]), 0.05, WOODM, 8)
CBS = rod(CB0 - aimn * 0.7, CB0 + aimn * 0.38, 0.06, WOODM, 8)           # stock
TIPS = []
for side in (-1, 1):
    tip = CB0 + aimn * 0.3 + np.array([0, -0.08, side * 0.6]) - aimn * 0.2
    rod(CB0 + aimn * 0.32, tip, 0.025, STEEL, 6); TIPS.append(tip)
# bow-string: two segments from each tip to a nock that is pulled back, then snaps forward
NOCK_BACK = CB0 - aimn * 0.32; NOCK_REST = CB0 + aimn * 0.12
T_REL = FINAL - L_AIM / 22.0
STR = []
for side, tip in enumerate(TIPS):
    rod((0, 0, 0), (0, 0, 1), 0.006, mat(f'string{side}', (0.9, 0.85, 0.75), 0.6), 4); s_ = bpy.context.object
    STR.append((s_, tip))
for f in range(NF):
    t = T[f]
    nk = NOCK_BACK if t < T_REL else NOCK_REST + (NOCK_BACK - NOCK_REST) * math.exp(-(t - T_REL) * 30) * math.cos((t - T_REL) * 60)
    for s_, tip in STR:
        d = Vector(tip) - Vector(nk); q = Vector((0, 0, 1)).rotation_difference(d.normalized())
        s_.matrix_world = Matrix.Translation(Vector(nk)) @ q.to_matrix().to_4x4() @ Matrix.Diagonal((1, 1, d.length, 1))
        if f in (0, NF - 1) or abs(t - T_REL) < 0.4:
            s_.keyframe_insert('location', frame=f); s_.keyframe_insert('rotation_euler', frame=f); s_.keyframe_insert('scale', frame=f)
# arrow
ARROW = link(bpy.data.objects.new('arrow', None))
sh = rod((0, 0, -0.5), (0, 0, 0.32), 0.015, WOODM, 6); sh.parent = ARROW
bpy.ops.mesh.primitive_cone_add(vertices=8, radius1=0.04, depth=0.12, location=(0, 0, 0.38)); hd = bpy.context.object; hd.data.materials.append(STEEL); hd.parent = ARROW
for k in range(3):
    a = k * 2 * math.pi / 3
    bx = box((0.035 * math.cos(a), 0.035 * math.sin(a), -0.42), (0.004, 0.06, 0.14), RED, rot=(0, 0, a + math.pi / 2)); bx.parent = ARROW
qa = Vector((0, 0, 1)).rotation_difference(Vector(aimn)); ARROW.rotation_mode = 'QUATERNION'
TIP_OFF = 0.44 + 0.08
apos = np.zeros((NF, 3)); arot = []
for f in range(NF):
    t = T[f]
    if t < T_REL: c = NOCK_BACK + aimn * 0.5
    elif t < FINAL: c = NOCK_BACK + aimn * 0.5 + aimn * (L_AIM - 0.5 + 0.32 - TIP_OFF + 0.06) * ((t - T_REL) / (FINAL - T_REL))
    else: c = AP0 - aimn * (TIP_OFF - 0.06)
    apos[f] = c
    wob = 0.0 if t < FINAL else 0.08 * math.exp(-(t - FINAL) * 4) * math.sin((t - FINAL) * 40)
    arot.append(Quaternion(Vector(np.cross(aimn, [0, 0, 1])).normalized(), wob) @ qa)
for k in range(3): key_series(ARROW, 'location', k, apos[:, k])
for f in range(NF):
    if T[f] < FINAL - 0.05 and f not in (0,): continue
    ARROW.rotation_quaternion = arot[f]; ARROW.keyframe_insert('rotation_quaternion', frame=f)
ARROW.rotation_quaternion = arot[0]; ARROW.keyframe_insert('rotation_quaternion', frame=0)
# apple jolt
jo = np.where(T >= FINAL, 0.06 * np.exp(-(T - FINAL) * 6) * np.sin((T - FINAL) * 30), 0)
key_series(APPLE, 'location', 0, AP0[0] + jo * aimn[0]); APPLE.location.y = AP0[1]; APPLE.location.z = AP0[2]
# sparkle burst
SPK = emis('spark', (1.0, 0.85, 0.4), 30)
for i in range(70):
    d = rng.normal(0, 1, 3); d /= np.linalg.norm(d); d[2] = abs(d[2]) * 0.8 + 0.2
    sp_ = rng.uniform(1.5, 4.0)
    bpy.ops.mesh.primitive_ico_sphere_add(radius=rng.uniform(0.015, 0.035), subdivisions=1); o = bpy.context.object; o.data.materials.append(SPK)
    tt = np.clip(T - FINAL, 0, None)
    pos = AP0[None] + d[None] * sp_ * tt[:, None] + np.array([0, 0, -1.6])[None] * tt[:, None] ** 2
    pos = np.where((T >= FINAL)[:, None], pos, AP0[None] + np.array([0, 0.0, -0.0]))
    scl = np.where((T >= FINAL) & (tt < 1.6), np.clip(1 - tt / 1.6, 0, 1), 0.0)
    for k in range(3): key_series(o, 'location', k, pos[:, k])
    for k in range(3): key_series(o, 'scale', k, scl)
# target board behind the apple (rings face the crossbow)
WHITE = mat('white', (0.9, 0.88, 0.82), 0.5)
for k, (rr, mt) in enumerate(((0.5, RED), (0.38, WHITE), (0.26, RED), (0.14, WHITE))):
    c0 = AP0 + aimn * (0.32 - 0.012 * k)
    rod(c0, c0 - aimn * 0.04, rr, mt, 32)
rod(AP0 + aimn * 0.36 - np.array([0, 0, 1.5]), AP0 + aimn * 0.36 - np.array([0, 0, 0.4]), 0.04, WOODM, 8)
# a lamp that flares on the apple at the hit
al = bpy.data.lights.new('applel', 'POINT'); al.color = (1, 0.8, 0.5); al.shadow_soft_size = 0.2
ALo = link(bpy.data.objects.new('applel', al)); ALo.location = AP0 + np.array([0, -0.4, 0.2])
for f in range(NF):
    e = 0 if T[f] < FINAL else 120 * math.exp(-(T[f] - FINAL) * 2.5)
    if f in (0, NF - 1) or abs(T[f] - FINAL) < 2.5: al.energy = e; al.keyframe_insert('energy', frame=f)

# ================= world & sun =================
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True; nt = w.node_tree
bg = nt.nodes['Background']; bg.inputs['Strength'].default_value = 1.0
grad = nt.nodes.new('ShaderNodeTexGradient'); tc = nt.nodes.new('ShaderNodeTexCoord'); mp = nt.nodes.new('ShaderNodeMapping')
mp.inputs['Rotation'].default_value = (0, -math.pi / 2, 0)
ramp = nt.nodes.new('ShaderNodeValToRGB'); ramp.color_ramp.elements[0].color = (0.55, 0.68, 0.85, 1); ramp.color_ramp.elements[1].color = (0.12, 0.26, 0.55, 1)
nt.links.new(tc.outputs['Generated'], mp.inputs['Vector']); nt.links.new(mp.outputs['Vector'], grad.inputs['Vector'])
nt.links.new(grad.outputs['Fac'], ramp.inputs['Fac']); nt.links.new(ramp.outputs['Color'], bg.inputs['Color'])
sun = bpy.data.lights.new('sun', 'SUN'); sun.energy = 4.0; sun.color = (1.0, 0.92, 0.8); sun.angle = 0.05
so = link(bpy.data.objects.new('sun', sun)); so.rotation_euler = (math.radians(50), 0, math.radians(-35))
fill = bpy.data.lights.new('fill', 'SUN'); fill.energy = 0.6; fill.color = (0.6, 0.75, 1.0)
fo_ = link(bpy.data.objects.new('fill', fill)); fo_.rotation_euler = (math.radians(70), 0, math.radians(150))

# ================= camera: wide, calm, follows the train's height =================
camd = bpy.data.cameras.new('cam'); camd.lens = 50; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
WID = 7.9                                                     # visible width (units)
def gsmooth(x, sig):
    k = np.arange(-int(4 * sig * FPS), int(4 * sig * FPS) + 1) / FPS; g = np.exp(-0.5 * (k / sig) ** 2); g /= g.sum()
    xp = np.concatenate([np.full(len(k) // 2, x[0]), x, np.full(len(k) // 2, x[-1])]); return np.convolve(xp, g, 'valid')
tz = gsmooth(LOCO[:, 2] - 1.3, 1.6)
tz = np.minimum(tz, -3.0)
fin = smoothstep((T - (FINAL - 7.0)) / 5.0)
wid = WID + (10.6 - WID) * fin
tz = tz * (1 - fin) + (Z_BR + 0.3) * fin
tx = 0.25 * np.sin(T / 7.0) * (1 - fin) + (-0.6) * fin
D = (wid * 16 / 9 / 2) * camd.lens / 12.0
TP = np.stack([tx, np.full(NF, -0.75), tz], 1)
CP = np.stack([tx * 0.6, -0.75 - D * math.cos(math.radians(17)), tz + D * math.sin(math.radians(17))], 1)
for k in range(3): key_series(cam, 'location', k, CP[:, k]); key_series(tgt, 'location', k, TP[:, k])

print('objects', len(bpy.data.objects), 'NF', NF, 'keys', len(KEYS), 'arms', len(ARMS), 'Z_BR', Z_BR)
if MODE == 'test':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS)); sc.frame_set(f)
        sc.render.filepath = f'{HERE}/test_{f:04d}.png'
        t1 = time.time(); bpy.ops.render.render(write_still=True); print('frame', f, 'sec', round(time.time() - t1, 1))
elif MODE == 'range':
    sc.frame_start, sc.frame_end = int(args[1]), int(args[2])
    sc.render.image_settings.file_format = 'PNG'; sc.render.filepath = f'{HERE}/frames/f_'
    bpy.ops.render.render(animation=True)
