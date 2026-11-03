"""Can-can ball launcher: pipes fire balls in arcs at instruments; each hit = the note you hear."""
import os, sys, json, math, colorsys, time
import bpy, bmesh
import numpy as np
from mathutils import Vector, Euler, Matrix

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json'))
FPS = 30; T_END = 66.0; NF = int(T_END * FPS); FINAL = E['final']
G = 9.8; TAU = 0.8
rng = np.random.default_rng(21)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'
sc.cycles.samples = int(os.environ.get('SAMP', 5))
sc.cycles.use_adaptive_sampling = True; sc.cycles.adaptive_threshold = 0.08
sc.cycles.use_denoising = True
for k, v in (('denoiser', 'OPENIMAGEDENOISE'), ('denoising_quality', 'FAST'), ('denoising_prefilter', 'FAST'),
             ('use_fast_gi', True), ('ao_bounces_render', 1)):
    try: setattr(sc.cycles, k, v)
    except Exception: pass
sc.cycles.max_bounces = 4; sc.cycles.diffuse_bounces = 1; sc.cycles.glossy_bounces = 2
sc.cycles.caustics_reflective = False; sc.cycles.caustics_refractive = False
sc.cycles.sample_clamp_indirect = 4
sc.render.resolution_x, sc.render.resolution_y = 720, 1280
sc.render.resolution_percentage = int(os.environ.get('PCT', 66))
sc.render.fps = FPS
sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.5
sc.render.use_persistent_data = True; sc.render.use_overwrite = False
try: sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Punchy'
except Exception: pass
sc.frame_start, sc.frame_end = 0, NF - 1


def link(o): sc.collection.objects.link(o); return o


def mat(name, col, rough=0.5, metal=0.0, **kw):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*col, 1); b.inputs['Roughness'].default_value = rough
    b.inputs['Metallic'].default_value = metal
    for k, v in kw.items(): b.inputs[k].default_value = v
    return m, b


def glowmat(name, col, rough=0.3, metal=0.0, gain=6.0):
    """object colour alpha drives emission (flash on hit)"""
    m, b = mat(name, col, rough, metal)
    nn = m.node_tree.nodes; ll = m.node_tree.links
    oi = nn.new('ShaderNodeObjectInfo'); mul = nn.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = gain
    ll.new(oi.outputs['Alpha'], mul.inputs[0]); ll.new(mul.outputs[0], b.inputs['Emission Strength'])
    ll.new(oi.outputs['Color'], b.inputs['Emission Color'])
    return m


def ss(x): x = min(max(x, 0), 1); return x * x * (3 - 2 * x)

_TB = bmesh.new(); bmesh.ops.create_uvsphere(_TB, u_segments=16, v_segments=10, radius=1.0)
_TV = np.array([v.co[:] for v in _TB.verts]); _TF = [[v.index for v in f.verts] for f in _TB.faces]; _TB.free()

# ---------------- stage ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bgn = w.node_tree.nodes['Background']; bgn.inputs['Color'].default_value = (0.03, 0.015, 0.03, 1); bgn.inputs['Strength'].default_value = 1
FLOOR, fb = mat('floor', (0.35, 0.18, 0.08), 0.35)
bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 4, 0)); bpy.context.object.data.materials.append(FLOOR)
WALL, _ = mat('backwall', (0.05, 0.015, 0.03), 0.8)
bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 6.5, 10), rotation=(math.pi / 2, 0, 0)); bpy.context.object.data.materials.append(WALL)
CURT, _ = mat('curtain', (0.45, 0.02, 0.04), 0.6, **{'Sheen Weight': 0.6})
for sx in (-1, 1):
    bm = bmesh.new()
    for i in range(40):
        for j in range(2):
            pass
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=60, y_subdivisions=2, size=1, location=(sx * 5.4, 1.0, 7.0), rotation=(math.pi / 2, 0, 0))
    c = bpy.context.object; c.scale = (3.0, 15, 1)
    for v in c.data.vertices: v.co.z += 0.06 * math.sin(v.co.x * 60)
    c.data.materials.append(CURT)
    bpy.ops.object.shade_smooth()
# valance
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 1.2, 12.6)); v_ = bpy.context.object; v_.scale = (14, 0.3, 1.4); v_.data.materials.append(CURT)
GOLD, _ = mat('gold', (1.0, 0.7, 0.25), 0.25, 1.0)
COPPER, _ = mat('copper', (0.95, 0.5, 0.3), 0.22, 1.0)
DARKM, _ = mat('dark', (0.04, 0.04, 0.05), 0.4, 0.6)

# spotlights
for sx, col, e in ((-4, (1, 0.8, 0.6), 5000), (4, (0.8, 0.85, 1), 5000), (0, (1, 0.9, 0.75), 4000)):
    ld = bpy.data.lights.new(f'spot{sx}', 'SPOT'); ld.energy = e; ld.color = col; ld.spot_size = math.radians(55); ld.spot_blend = 0.4
    ld.shadow_soft_size = 0.4
    o = link(bpy.data.objects.new(f'spot{sx}', ld)); o.location = (sx, -6, 11)
    tr = o.constraints.new('TRACK_TO'); e_ = link(bpy.data.objects.new(f'spt{sx}', None)); e_.location = (sx * 0.3, 2, 4.5); tr.target = e_
    tr.track_axis = 'TRACK_NEGATIVE_Z'; tr.up_axis = 'UP_Y'
fl = bpy.data.lights.new('fill', 'AREA'); fl.energy = 900; fl.size = 8; fl.color = (0.8, 0.6, 1.0)
link(bpy.data.objects.new('fill', fl)).location = (0, -8, 3)

# ---------------- instruments ----------------
targets = {}       # key -> (hit point, object for flash)
INST = []          # (object, key) for flashing
mel_p = sorted({p for _, p in E['mel']})
nx = len(mel_p)
XY_Z = 7.2; XY_Y = 2.6
BARC = {}
for i, p in enumerate(mel_p):
    x = -3.9 + 7.8 * i / (nx - 1)
    L = 1.9 - 1.0 * i / (nx - 1)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, XY_Y, XY_Z))
    o = bpy.context.object; o.scale = (0.4, 0.14, L); bpy.ops.object.transform_apply(scale=True)
    bev = o.modifiers.new('b', 'BEVEL'); bev.width = 0.03; bev.segments = 2; bpy.ops.object.modifier_apply(modifier='b')
    rgb = colorsys.hsv_to_rgb(i / nx * 0.85, 0.75, 1.0)
    o.data.materials.append(glowmat(f'bar{p}', (0.55, 0.32, 0.15), 0.35, gain=8)); o['rgb'] = rgb; o.color = (*rgb, 0)
    targets[('mel', p)] = Vector((x, XY_Y - 0.08, XY_Z + 0.1)); INST.append((o, ('mel', p)))
    for sz in (1, -1):   # mounting cord pins
        bpy.ops.mesh.primitive_cylinder_add(radius=0.025, depth=0.25, location=(x, XY_Y - 0.05, XY_Z + sz * L * 0.32), rotation=(math.pi / 2, 0, 0))
        bpy.context.object.data.materials.append(GOLD)
# xylophone frame
for zz in (XY_Z + 1.05, XY_Z - 1.05):
    bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=8.6, location=(0, XY_Y + 0.1, zz), rotation=(0, math.pi / 2, 0))
    bpy.context.object.data.materials.append(GOLD)

# bass marimba (left) with resonator tubes
bass_p = sorted({p for _, p in E['bass']})
for i, p in enumerate(bass_p):
    x = -4.0 + i * 0.62
    L = 2.0 - 0.13 * i
    zc = 3.6
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, 2.0, zc)); o = bpy.context.object
    o.scale = (0.5, 0.16, L); bpy.ops.object.transform_apply(scale=True)
    o.data.materials.append(glowmat(f'bass{p}', (0.3, 0.12, 0.06), 0.35, gain=8)); o['rgb'] = (0.3, 0.55, 1.0); o.color = (0.3, 0.55, 1.0, 0)
    targets[('bass', p)] = Vector((x, 1.9, zc + 0.15)); INST.append((o, ('bass', p)))
    bpy.ops.mesh.primitive_cylinder_add(radius=0.16, depth=1.6, location=(x, 2.25, zc - L / 2 - 0.85)); bpy.context.object.data.materials.append(GOLD)

# chord bells (right): three tubular chimes
CH_C = Vector((3.1, 2.0, 4.0))
chime_objs = []
for k in range(3):
    bpy.ops.mesh.primitive_cylinder_add(radius=0.11, depth=2.6 - 0.3 * k, location=(CH_C.x - 0.35 + 0.35 * k, CH_C.y, CH_C.z))
    o = bpy.context.object; o.data.materials.append(glowmat(f'chime{k}', (0.8, 0.8, 0.85), 0.2, metal=1.0, gain=6)); o['rgb'] = (0.3, 1.0, 0.5); o.color = (0.3, 1.0, 0.5, 0)
    chime_objs.append(o); INST.append((o, ('chord', 0)))
bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=1.3, location=(CH_C.x, CH_C.y, CH_C.z + 1.45), rotation=(0, math.pi / 2, 0)); bpy.context.object.data.materials.append(GOLD)
targets[('chord', 0)] = CH_C + Vector((0, -0.12, 0.3))

# kick drum (front, centre-right) and snare (right)
DRUM, _ = mat('drumshell', (0.6, 0.02, 0.05), 0.25, **{'Coat Weight': 1.0})
HEADM = glowmat('head', (0.92, 0.9, 0.85), 0.5, gain=3)
bpy.ops.mesh.primitive_cylinder_add(radius=0.95, depth=0.7, location=(1.2, 0.6, 1.0), rotation=(math.pi / 2, 0, 0)); kd = bpy.context.object; kd.data.materials.append(DRUM)
bpy.ops.mesh.primitive_cylinder_add(radius=0.9, depth=0.02, location=(1.2, 0.24, 1.0), rotation=(math.pi / 2, 0, 0)); kh = bpy.context.object
kh.data.materials.append(HEADM); kh['rgb'] = (1, 0.3, 0.2); kh.color = (1, 0.3, 0.2, 0); INST.append((kh, ('kick', 0)))
targets[('kick', 0)] = Vector((1.2, 0.2, 1.1))
bpy.ops.mesh.primitive_cylinder_add(radius=0.45, depth=0.25, location=(3.3, 0.8, 1.6)); sd = bpy.context.object; sd.data.materials.append(DRUM)
bpy.ops.mesh.primitive_cylinder_add(radius=0.43, depth=0.02, location=(3.3, 0.8, 1.74)); sh = bpy.context.object
sh.data.materials.append(HEADM); sh['rgb'] = (1, 0.8, 0.3); sh.color = (1, 0.8, 0.3, 0); INST.append((sh, ('snare', 0)))
bpy.ops.mesh.primitive_cylinder_add(radius=0.03, depth=1.5, location=(3.3, 0.8, 0.75)); bpy.context.object.data.materials.append(DARKM)
targets[('snare', 0)] = Vector((3.3, 0.8, 1.78))
# crash cymbal (top right)
cpiv = link(bpy.data.objects.new('cpiv', None)); cpiv.location = (3.4, 1.6, 9.6); cpiv.rotation_mode = 'XYZ'
bpy.ops.mesh.primitive_cone_add(radius1=0.95, radius2=0.1, depth=0.1, vertices=48, location=(0, 0, 0))
cy = bpy.context.object; cy.parent = cpiv; cy.rotation_euler = (math.radians(70), 0, 0)
cy.data.materials.append(glowmat('cym', (1.0, 0.72, 0.25), 0.2, metal=1.0, gain=4)); cy['rgb'] = (1, 0.85, 0.4); cy.color = (1, 0.85, 0.4, 0)
INST.append((cy, ('crash', 0)))
targets[('crash', 0)] = Vector((3.4, 1.35, 9.6))

# ---------------- launcher pipes (fan at the bottom centre) ----------------
keys = list(targets.keys())
# order pipes by target x so trajectories cross as little as possible
keys.sort(key=lambda k: targets[k].x)
pipes = {}
for j, k in enumerate(keys):
    x = -2.6 + 5.2 * j / (len(keys) - 1)
    base = Vector((x, -1.4, 0.4 + 0.25 * (j % 2)))
    T = targets[k]
    v0 = (T - base) / TAU + Vector((0, 0, 0.5 * G * TAU))
    d = v0.normalized()
    mouth = base + d * 0.9
    v0 = (T - mouth) / TAU + Vector((0, 0, 0.5 * G * TAU)); d = v0.normalized()
    bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=1.8, location=mouth - d * 0.9)
    pp = bpy.context.object; pp.rotation_mode = 'QUATERNION'; pp.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d)
    pp.data.materials.append(COPPER)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.12, minor_radius=0.03, location=mouth)
    rim = bpy.context.object; rim.rotation_mode = 'QUATERNION'; rim.rotation_quaternion = pp.rotation_quaternion
    rim.data.materials.append(glowmat(f'rim{j}', (1, 0.7, 0.3), 0.2, metal=1.0, gain=10)); rim['rgb'] = (1, 0.75, 0.35); rim.color = (1, 0.75, 0.35, 0)
    pipes[k] = (mouth, rim)
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -1.5, 0.25)); bb = bpy.context.object; bb.scale = (6.2, 1.2, 0.5); bb.data.materials.append(DARKM)

# ---------------- balls ----------------
BALLM = {'mel': glowmat('bmel', (1.0, 0.8, 0.3), 0.15, metal=1.0, gain=1.0), 'bass': glowmat('bbass', (0.3, 0.55, 1.0), 0.15, metal=1.0, gain=1.0),
         'chord': glowmat('bch', (0.3, 1.0, 0.5), 0.15, metal=1.0, gain=1.0), 'kick': glowmat('bkick', (1.0, 0.3, 0.25), 0.15, metal=1.0, gain=1.0),
         'snare': glowmat('bsn', (1.0, 0.95, 0.9), 0.15, metal=1.0, gain=1.0), 'crash': glowmat('bcr', (1.0, 0.9, 0.5), 0.15, metal=1.0, gain=2.5)}
shots = [(t, ('mel', p)) for t, p in E['mel']] + [(t, ('bass', p)) for t, p in E['bass']] + [(x[0], ('chord', 0)) for x in E['chord']] + \
        [(t, ('kick', 0)) for t in E['kick']] + [(t, ('snare', 0)) for t in E['snare']] + [(t, ('crash', 0)) for t in E['crash']]
shots.sort()
sph = {}
for kind in BALLM:
    r = 0.18 if kind in ('kick', 'crash') else 0.13
    me = bpy.data.meshes.new('ball_' + kind); me.from_pydata((_TV * r).tolist(), [], _TF)
    for p_ in me.polygons: p_.use_smooth = True
    me.materials.append(BALLM[kind]); sph[kind] = me
balls = []
for i, (t, k) in enumerate(shots):
    o = link(bpy.data.objects.new(f'b{i}', sph[k[0]])); o.color = (1, 1, 1, 1.0)
    balls.append((o, t, k))


def ball_anim():
    for o, t, k in balls:
        P0, _ = pipes[k]; P1 = targets[k]
        v = (P1 - P0) / TAU + Vector((0, 0, 0.5 * G * TAU))
        vin = v + Vector((0, 0, -G * TAU))
        # rebound: back toward camera and down, damped
        vout = Vector((vin.x * 0.25 + rng.uniform(-0.4, 0.4), -abs(vin.y) * 0.35 - 0.8, abs(vin.z) * 0.25 + 1.0))
        t0 = t - TAU; f0 = max(0, int(math.floor(t0 * FPS)) - 1); f2 = min(NF - 1, int((t + 0.9) * FPS))
        frames = [0, f0] + list(range(f0 + 1, f2 + 1)) + [min(NF - 1, f2 + 1)]
        for f in frames:
            tt = f / FPS
            if tt < t0:
                pos, vis = P0, False
            elif tt <= t:
                a = tt - t0; pos = P0 + v * a + Vector((0, 0, -0.5 * G * a * a)); vis = True
            elif tt <= t + 0.9:
                a = tt - t; pos = P1 + vout * a + Vector((0, 0, -0.5 * G * a * a)); vis = True
            else:
                pos, vis = P1, False
            o.location = pos; o.hide_render = not vis
            o.keyframe_insert('location', frame=f); o.keyframe_insert('hide_render', frame=f)


def flash_anim():
    by = {}
    for t, k in shots: by.setdefault(k, []).append(t)
    for o, k in INST:
        ts = by.get(k, []); rgb = o['rgb']; base = o.location.copy()
        fr = {0, NF - 1}
        for t in ts:
            hf = int(math.ceil(t * FPS)); fr |= set(range(max(0, hf - 1), min(NF, hf + 9)))
        for f in sorted(fr):
            tt = f / FPS; g = 0.0; sh = 0.0
            for t in ts:
                if 0 <= tt - t < 0.5:
                    g = max(g, math.exp(-(tt - t) / 0.12)); sh += 0.035 * math.exp(-(tt - t) / 0.06) * math.cos(70 * (tt - t))
            o.color = (*rgb, g); o.location = base + Vector((0, sh, 0))
            o.keyframe_insert('color', frame=f); o.keyframe_insert('location', frame=f)
    # pipe mouths flash when they fire
    for k, (mouth, rim) in pipes.items():
        ts = [t - TAU for t in by.get(k, [])]
        fr = {0, NF - 1}
        for t in ts:
            hf = int(math.ceil(t * FPS)); fr |= set(range(max(0, hf - 1), min(NF, hf + 5)))
        for f in sorted(fr):
            tt = f / FPS; g = 0.0
            for t in ts:
                if 0 <= tt - t < 0.3: g = max(g, math.exp(-(tt - t) / 0.06))
            rim.color = (*rim['rgb'], g); rim.keyframe_insert('color', frame=f)
    # cymbal swing
    for f in range(NF):
        tt = f / FPS; a = 0.0
        for c in E['crash']:
            if 0 <= tt - c < 2.5: a += 0.3 * math.exp(-(tt - c) / 0.5) * math.sin(2 * math.pi * 2.3 * (tt - c))
        cpiv.rotation_euler = (a, 0.5 * a, 0); cpiv.keyframe_insert('rotation_euler', frame=f)

# ---------------- finale confetti ----------------
CONF = [glowmat(f'conf{k}', c, 0.4, gain=0.6) for k, c in enumerate(((1, 0.2, 0.3), (1, 0.85, 0.2), (0.2, 0.7, 1), (0.3, 1, 0.4), (1, 1, 1)))]
conf = []
for k in range(140):
    bpy.ops.mesh.primitive_plane_add(size=0.12, location=(0, 0, -5))
    o = bpy.context.object; o.data.materials.append(CONF[k % len(CONF)]); o.color = (1, 1, 1, 1)
    conf.append((o, Vector((rng.uniform(-4.5, 4.5), rng.uniform(-1, 2.5), rng.uniform(11, 14))), rng.uniform(0, 1.5), rng.uniform(0.8, 1.5)))


def confetti_anim():
    for o, p0, delay, spd in conf:
        o.rotation_mode = 'XYZ'
        for f in [0, int((FINAL + delay) * FPS) - 1] + list(range(int((FINAL + delay) * FPS), NF, 3)):
            tt = f / FPS - FINAL - delay
            if tt < 0: o.location = (0, 0, -5)
            else:
                o.location = p0 + Vector((0.3 * math.sin(tt * 3 + delay * 5), 0, -spd * tt))
                o.rotation_euler = (tt * 5 + delay, tt * 3, tt * 4)
            o.keyframe_insert('location', frame=f); o.keyframe_insert('rotation_euler', frame=f)

# ---------------- camera ----------------
cd = bpy.data.cameras.new('cam'); cd.lens = 52; cd.sensor_fit = 'HORIZONTAL'; cd.sensor_width = 36; cd.clip_end = 300
cam = link(bpy.data.objects.new('cam', cd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trk = cam.constraints.new('TRACK_TO'); trk.target = tgt; trk.track_axis = 'TRACK_NEGATIVE_Z'; trk.up_axis = 'UP_Y'
secs = dict((n, t) for t, n in E['sections'])
KEYS = [
    (0.0, (1.8, -8.0, 1.6), (0.0, 0.0, 2.0)),               # close on the pipes
    (secs['A'], (0.0, -12.5, 4.8), (0.0, 2.0, 5.0)),        # whole stage
    (secs['B'] - 2, (-2.5, -10.5, 6.5), (-0.5, 2.0, 6.6)),   # xylophone side
    (secs['B'] + 6, (2.6, -10.0, 3.0), (1.5, 1.5, 3.2)),     # drums & chimes
    (secs['A2'], (0.0, -12.8, 5.0), (0.0, 2.0, 5.1)),
    (secs['A2'] + 9, (-2.8, -9.6, 3.6), (-2.0, 1.5, 4.0)),   # bass marimba
    (secs['finale'], (0.0, -13.0, 5.4), (0.0, 2.0, 5.6)),
    (T_END, (0.0, -14.5, 6.2), (0.0, 2.0, 6.2)),
]


def cam_state(t):
    for k in range(len(KEYS) - 1):
        if KEYS[k][0] <= t <= KEYS[k + 1][0]:
            a = ss((t - KEYS[k][0]) / max(1e-3, KEYS[k + 1][0] - KEYS[k][0]))
            return Vector(KEYS[k][1]).lerp(Vector(KEYS[k + 1][1]), a), Vector(KEYS[k][2]).lerp(Vector(KEYS[k + 1][2]), a)
    return Vector(KEYS[-1][1]), Vector(KEYS[-1][2])


def cam_anim():
    for f in range(NF):
        t = f / FPS; p, l = cam_state(t)
        for c in E['crash']:
            if 0 <= t - c < 0.4:
                kk = math.exp(-(t - c) / 0.1); p = p + Vector((0.08 * kk * math.sin(90 * t), 0, 0.06 * kk * math.sin(77 * t)))
        cam.location = p; tgt.location = l
        cam.keyframe_insert('location', frame=f); tgt.keyframe_insert('location', frame=f)


t0 = time.time()
ball_anim(); flash_anim(); confetti_anim(); cam_anim()
print('built', round(time.time() - t0, 1), 'balls', len(balls), 'pipes', len(pipes))

if MODE == 'test':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS)); sc.frame_set(f)
        sc.render.filepath = f'{HERE}/test_{f:04d}.png'
        t1 = time.time(); bpy.ops.render.render(write_still=True)
        print('frame', f, 'sec', round(time.time() - t1, 1))
elif MODE == 'range':
    sc.frame_start, sc.frame_end = int(args[1]), int(args[2])
    sc.render.image_settings.file_format = 'PNG'
    sc.render.filepath = f'{HERE}/frames/f_'
    bpy.ops.render.render(animation=True)
