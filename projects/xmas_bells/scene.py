"""Carol of the Bells - a glowing ball rolls down a spiral track wrapped around a Christmas tree.
The track is paved with glass keys: one key per note of the glockenspiel line, so fast runs become dense keyboards.
Each key lights as the ball rolls over it; golden bells on the tree swing on the bass; the star blazes at the end."""
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


# ---------------- the tree ----------------
ZB, ZT = 0.0, 7.4                       # trunk base, tip
def tree_r(z): return np.maximum(0.05, 2.0 * np.clip(1 - (z - 0.5) / (ZT - 0.5), 0, None) ** 1.05)
NEEDLE = mat('needle', (0.02, 0.16, 0.06), 0.7)
NEEDLE2 = mat('needle2', (0.03, 0.22, 0.08), 0.65)
for i in range(9):                                   # stacked frilly cones
    z0 = 0.5 + i * 0.72; h = 1.25; r0 = tree_r(z0) * 1.05
    bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=r0, radius2=0.02, depth=h, location=(0, 0, z0 + h / 2))
    o = bpy.context.object; o.data.materials.append(NEEDLE if i % 2 else NEEDLE2)
    bm = bmesh.new(); bm.from_mesh(o.data)
    for v in bm.verts:                               # scalloped hem
        if v.co.z < -h / 2 + 1e-3:
            a = math.atan2(v.co.y, v.co.x); v.co.z += 0.06 * math.cos(a * 12)
    bm.to_mesh(o.data)
    m = o.modifiers.new('d', 'DISPLACE'); tx = bpy.data.textures.new(f'tn{i}', 'CLOUDS'); tx.noise_scale = 0.15; m.texture = tx; m.strength = 0.06
    o.modifiers.new('s', 'SUBSURF').levels = 1
rod((0, 0, 0), (0, 0, 0.9), 0.22, mat('trunk', (0.2, 0.1, 0.05), 0.8), 16)
# fairy lights on the tree surface (twinkle in 4 groups)
LIGHT_COLS = [(1.0, 0.75, 0.35), (1.0, 0.3, 0.3), (0.4, 0.7, 1.0), (1.0, 0.9, 0.6)]
LMATS = [emis(f'fl{k}', c, 6.0) for k, c in enumerate(LIGHT_COLS)]
FL = {k: [] for k in range(4)}
for i in range(260):
    z = 0.7 + (ZT - 1.0) * (i / 260) ** 0.85; a = i * 2.39996
    r = tree_r(z) * 0.98
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.03, segments=8, ring_count=6, location=(r * math.cos(a), r * math.sin(a), z))
    o = bpy.context.object; k = i % 4; o.data.materials.append(LMATS[k]); FL[k].append(o)
# ornaments (glossy balls)
for i in range(70):
    z = 0.9 + (ZT - 1.6) * rng.random(); a = rng.uniform(0, 2 * math.pi); r = tree_r(z) * 1.0
    col = [(0.8, 0.05, 0.05), (0.85, 0.65, 0.15), (0.75, 0.75, 0.8), (0.1, 0.2, 0.7)][i % 4]
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.07, segments=16, ring_count=10, location=(r * math.cos(a), r * math.sin(a), z - 0.05))
    o = bpy.context.object; o.data.materials.append(mat(f'orn{i}', col, 0.08, **{'Metallic': 0.9}))
    for p in o.data.polygons: p.use_smooth = True
# golden bells that swing on the bass
GOLD = mat('gold', (1.0, 0.7, 0.25), 0.2, **{'Metallic': 1.0})
BELLS = []
for i in range(10):
    z = 1.2 + i * 0.55; a = i * 2.1 + 0.6; r = tree_r(z) * 1.02
    piv = link(bpy.data.objects.new(f'bellp{i}', None)); piv.location = (r * math.cos(a), r * math.sin(a), z)
    piv.rotation_mode = 'XYZ'; piv.rotation_euler = (0, 0, a)
    bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=0.12, radius2=0.04, depth=0.18, location=(0, 0, -0.12))
    b = bpy.context.object; b.data.materials.append(GOLD); b.parent = piv
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.035, location=(0, 0, -0.22)); c = bpy.context.object; c.data.materials.append(GOLD); c.parent = piv
    BELLS.append(piv)
# star
bm = bmesh.new(); pts = []
for k in range(10):
    a = math.pi / 2 + k * math.pi / 5; r = 0.42 if k % 2 == 0 else 0.18
    pts.append(bm.verts.new((r * math.cos(a), 0, r * math.sin(a))))
f = bm.faces.new(pts); bmesh.ops.extrude_face_region(bm, geom=[f])
for v in bm.verts[len(pts):]: v.co.y += 0.08
me = bpy.data.meshes.new('star'); bm.to_mesh(me)
STAR = link(bpy.data.objects.new('star', me)); STAR.location = (0, 0, ZT + 0.25)
SMAT = mat('starm', (1.0, 0.8, 0.3), 0.2, **{'Metallic': 0.6, 'Emission Color': (1.0, 0.8, 0.35, 1)})
STAR.data.materials.append(SMAT)

# ---------------- the spiral track ----------------
TURNS = 6.2; Z0, Z1 = ZT - 0.6, 0.55
uu = np.linspace(0, 1, 6000)
def helix(u):
    z = Z0 + (Z1 - Z0) * u; r = tree_r(z) + 0.42; a = -TURNS * 2 * math.pi * u - 0.5
    return np.stack([r * np.cos(a), r * np.sin(a), z], -1) if np.ndim(u) else np.array([r * math.cos(a), r * math.sin(a), z])
HP = helix(uu); seg = np.linalg.norm(np.diff(HP, axis=0), axis=1); S = np.concatenate([[0], np.cumsum(seg)]); SLEN = S[-1]
t_first, t_last = LEAD[0][0], LEAD[-1][0]
# ball speed: the ball covers the track from t=0.2 to the final note
V = SLEN / (t_last - 0.2)
def s_of_t(t): return np.clip((t - 0.2) * V, 0, SLEN)
def u_of_s(s): return np.interp(s, S, uu)
def pt(s): return helix(u_of_s(s))
def frame_at(s):
    """position, tangent, outward normal"""
    p = pt(s); p2 = pt(min(SLEN, s + 0.02)); p1 = pt(max(0, s - 0.02))
    tg = (p2 - p1); tg /= np.linalg.norm(tg) + 1e-9
    outw = np.array([p[0], p[1], 0.0]); outw /= np.linalg.norm(outw) + 1e-9
    return p, tg, outw
RAIL = mat('rail', (0.85, 0.85, 0.9), 0.15, **{'Metallic': 1.0})
W = 0.075                                             # half gauge
for side in (-1, 1):
    cu = bpy.data.curves.new(f'rail{side}', 'CURVE'); cu.dimensions = '3D'; cu.bevel_depth = 0.009; cu.bevel_resolution = 2
    sp = cu.splines.new('POLY'); ss = np.linspace(0, SLEN, 1400); sp.points.add(len(ss) - 1)
    for i, s in enumerate(ss):
        p, tg, ow = frame_at(s); q = p + ow * side * W + np.array([0, 0, 0.02])
        sp.points[i].co = (*q, 1)
    o = link(bpy.data.objects.new(f'rail{side}', cu)); o.data.materials.append(RAIL)
# brackets back to the tree every ~0.7 m
for s in np.arange(0.3, SLEN, 0.9):
    p, tg, ow = frame_at(s); rod(p - ow * W + np.array([0, 0, 0.01]), p - ow * 0.36 + np.array([0, 0, -0.06]), 0.007, RAIL, 6)
# keys: one glass key per lead note, length = distance to the next note
KEYM = glowmat('key', 4.0, 0.15)
KEYS = []
pitches = sorted(set(p for _, p, _ in LEAD)); pmin, pmax = min(pitches), max(pitches)
for i, (t, p, v) in enumerate(LEAD):
    s0 = s_of_t(t); s1 = s_of_t(LEAD[i + 1][0]) if i + 1 < len(LEAD) else s0 + 0.3
    L = max(0.03, min(0.32, (s1 - s0) * 0.82))
    p0, tg, ow = frame_at(s0 + L / 2)
    up = np.cross(ow, tg); up /= np.linalg.norm(up)
    bpy.ops.mesh.primitive_cube_add(size=1)
    o = bpy.context.object; o.scale = (L, 2 * W - 0.012, 0.022)
    M = Matrix(((tg[0], -ow[0], up[0]), (tg[1], -ow[1], up[1]), (tg[2], -ow[2], up[2])))
    o.matrix_world = Matrix.Translation(Vector(p0) + Vector(up) * 0.004) @ M.to_4x4() @ Matrix.Diagonal((L, 2 * W - 0.012, 0.022, 1))
    m = o.modifiers.new('b', 'BEVEL'); m.width = 0.006; m.segments = 2
    o.data.materials.append(KEYM)
    hue = ((p - pmin) / max(1, pmax - pmin)) * 0.8
    col = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    KEYS.append((o, t, col, Vector(p0), Vector(up)))
bpy.context.view_layer.update()
for o, t, col, P, up in KEYS:
    a = np.where(T >= t - 0.5 / FPS, 0.12 + 0.88 * np.exp(-np.clip(T - t, 0, None) / 0.22), 0.0)
    key_alpha(o, a, col)
    # little dip as the ball passes
    d = np.exp(-((T - t) / 0.06) ** 2) * 0.012
    base = o.location.copy()
    key_series(o, 'location', 2, base.z - d)

# ---------------- the ball ----------------
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.055, segments=32, ring_count=16)
BALL = bpy.context.object
for p in BALL.data.polygons: p.use_smooth = True
BM = mat('ballm', (1.0, 0.3, 0.7), 0.05, **{'Coat Weight': 1.0, 'Emission Color': (1.0, 0.25, 0.65, 1), 'Emission Strength': 1.2})
BALL.data.materials.append(BM)
ld = bpy.data.lights.new('bl', 'POINT'); ld.energy = 8; ld.color = (1.0, 0.6, 0.85); ld.shadow_soft_size = 0.05
lo = link(bpy.data.objects.new('bl', ld)); lo.parent = BALL
BP = np.zeros((NF, 3))
for f in range(NF):
    t = T[f]; p, tg, ow = frame_at(s_of_t(t)); BP[f] = p + np.array([0, 0, 0.02 + 0.055])
# before start: ball drops from the star
drop = T < 0.2
BP[drop] = BP[np.argmax(~drop)] + np.array([0, 0, 1]) * (0.2 - T[drop])[:, None] * 2.0
# after the last note: hop off onto the gifts
endf = int(t_last * FPS)
for f in range(endf, NF):
    a = T[f] - t_last; p, tg, ow = frame_at(SLEN)
    BP[f] = p + np.array([0, 0, 0.075]) + tg * 0.6 * a + ow * 0.5 * a + np.array([0, 0, 1.4 * a - 4.9 * a * a])
    BP[f, 2] = max(BP[f, 2], 0.3 + 0.055)
for i in range(3): key_series(BALL, 'location', i, BP[:, i])
# rolling spin
spin = np.cumsum(np.r_[0, np.linalg.norm(np.diff(BP, axis=0), axis=1)]) / 0.055
BALL.rotation_mode = 'XYZ'; key_series(BALL, 'rotation_euler', 0, spin * 0.7); key_series(BALL, 'rotation_euler', 1, spin * 0.7)

# ---------------- ground, gifts, snowman, sign ----------------
SNOW = mat('snow', (0.85, 0.88, 0.95), 0.6, **{'Subsurface Weight': 0.2})
bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0)); g = bpy.context.object; g.data.materials.append(SNOW)
m = g.modifiers.new('sub', 'SUBSURF'); m.levels = 3; m.subdivision_type = 'SIMPLE'
m = g.modifiers.new('d', 'DISPLACE'); tx = bpy.data.textures.new('sn', 'CLOUDS'); tx.noise_scale = 2.0; m.texture = tx; m.strength = 0.15
GIFTC = [(0.8, 0.08, 0.1), (0.1, 0.45, 0.2), (0.15, 0.3, 0.8), (0.9, 0.75, 0.2), (0.85, 0.85, 0.9)]
for i in range(9):
    a = i * 0.7 + 0.3 + rng.normal(0, 0.1); r = 2.5 + rng.uniform(0, 0.8); s = rng.uniform(0.3, 0.55)
    bpy.ops.mesh.primitive_cube_add(size=s, location=(r * math.cos(a), r * math.sin(a), s / 2), rotation=(0, 0, rng.uniform(0, 1)))
    o = bpy.context.object; o.data.materials.append(mat(f'gift{i}', GIFTC[i % 5], 0.35))
    rib = mat(f'rib{i}', GIFTC[(i + 3) % 5], 0.2, **{'Metallic': 0.5})
    for ax in ((1.02, 0.15, 1.02), (0.15, 1.02, 1.02)):
        bpy.ops.mesh.primitive_cube_add(size=s, location=o.location, rotation=o.rotation_euler); r_ = bpy.context.object; r_.scale = ax; r_.data.materials.append(rib)
# snowman
SX, SY_ = 3.6, -2.4
for z, r in ((0.45, 0.5), (1.15, 0.36), (1.65, 0.25)):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(SX, SY_, z)); o = bpy.context.object; o.data.materials.append(SNOW)
    for p in o.data.polygons: p.use_smooth = True
COAL = mat('coal', (0.02, 0.02, 0.02), 0.6)
for dx in (-0.08, 0.08):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.03, location=(SX - 0.2, SY_ + dx - 0.12, 1.72)); bpy.context.object.data.materials.append(COAL)
bpy.ops.mesh.primitive_cone_add(radius1=0.04, depth=0.2, location=(SX - 0.3, SY_ - 0.12, 1.62), rotation=(0, -math.pi / 2, 0.4)); bpy.context.object.data.materials.append(mat('carrot', (1, 0.4, 0.05), 0.5))
bpy.ops.mesh.primitive_cylinder_add(radius=0.27, depth=0.06, location=(SX, SY_, 1.85)); bpy.context.object.data.materials.append(mat('scarf', (0.8, 0.05, 0.08), 0.6))
# 3D sign
cu = bpy.data.curves.new('sign', 'FONT'); cu.body = 'Merry Christmas'; cu.size = 0.4; cu.extrude = 0.05; cu.align_x = 'CENTER'
SIGN = link(bpy.data.objects.new('sign', cu)); SIGN.location = (0, -3.6, 0.15); SIGN.rotation_euler = (math.pi / 2, 0, 0)
SIGN.data.materials.append(mat('signm', (0.85, 0.08, 0.1), 0.3, **{'Emission Color': (1.0, 0.15, 0.15, 1), 'Emission Strength': 1.5}))

# ---------------- snow fall (particles) ----------------
bpy.ops.mesh.primitive_plane_add(size=16, location=(0, 0, 10)); em = bpy.context.object; em.hide_render = True
ps = em.modifiers.new('snow', 'PARTICLE_SYSTEM').particle_system; pst = ps.settings
pst.count = 1600; pst.frame_start = -300; pst.frame_end = NF; pst.lifetime = 400; pst.normal_factor = 0; pst.factor_random = 0.2
pst.effector_weights.gravity = 0.03; pst.render_type = 'OBJECT'; pst.particle_size = 1.0; pst.size_random = 0.6
bpy.ops.mesh.primitive_ico_sphere_add(radius=0.011, subdivisions=1, location=(0, 0, -20)); flake = bpy.context.object
flake.data.materials.append(emis('flake', (0.9, 0.95, 1.0), 2.0)); pst.instance_object = flake
pst.brownian_factor = 0.15

# ---------------- animation of tree lights, bells, star ----------------
bars = [E['t0'] + b * E['bar'] for b in range(64)]
for k, objs in FL.items():
    a = np.full(NF, 0.55)
    for b, tb in enumerate(bars):
        if b % 4 == k: a = np.maximum(a, 0.55 + 0.9 * np.exp(-np.clip(T - tb, 0, None) / 0.35) * (T >= tb))
    a = np.where(T > FINAL - 0.05, 2.0, a)
    m_ = LMATS[k].node_tree.nodes['Emission']
    for f in range(0, NF):
        if 0 < f < NF - 1 and abs(a[f] - a[f - 1]) < 1e-3 and abs(a[f] - a[f + 1]) < 1e-3: continue
        m_.inputs['Strength'].default_value = 6.0 * a[f]; m_.inputs['Strength'].keyframe_insert('default_value', frame=f)
bass_t = [n[0] for n in NOTES if n[2] in ('cello', 'cpizz', 'pbass')]
sw = np.zeros(NF)
for tb in sorted(set(round(x, 3) for x in bass_t)):
    a = T - tb; sw += np.where(a >= 0, np.sin(a * 9) * np.exp(-a * 2.5), 0) * 0.35
for i, b in enumerate(BELLS):
    key_series(b, 'rotation_euler', 1, sw * (1 if i % 2 else -1))
sg = np.clip((T - FINAL) / 0.2, 0, 1)
key_series(STAR, 'scale', 0, 1 + 0.3 * sg); key_series(STAR, 'scale', 2, 1 + 0.3 * sg)
sn = SMAT.node_tree.nodes['Principled BSDF'].inputs['Emission Strength']
for f in range(NF):
    v = 1.5 + 7 * sg[f] * (0.85 + 0.15 * math.sin(T[f] * 8))
    if f in (0, NF - 1) or sg[f] > 0 or (f > 0 and sg[f - 1] > 0): sn.default_value = v; sn.keyframe_insert('default_value', frame=f)
STAR.rotation_mode = 'XYZ'; key_series(STAR, 'rotation_euler', 2, T * 0.6)
stl = bpy.data.lights.new('starl', 'POINT'); stl.color = (1, 0.8, 0.4); stl.shadow_soft_size = 0.3
sto = link(bpy.data.objects.new('starl', stl)); sto.location = STAR.location
for f in range(NF):
    if f in (0, NF - 1) or sg[f] > 0 or (f > 0 and sg[f - 1] > 0): stl.energy = 20 + 220 * sg[f]; stl.keyframe_insert('energy', frame=f)

# ---------------- world & lights ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True; nt = w.node_tree
bg = nt.nodes['Background']; bg.inputs['Strength'].default_value = 1.0
grad = nt.nodes.new('ShaderNodeTexGradient'); tc = nt.nodes.new('ShaderNodeTexCoord'); mp = nt.nodes.new('ShaderNodeMapping')
mp.inputs['Rotation'].default_value = (0, -math.pi / 2, 0)
ramp = nt.nodes.new('ShaderNodeValToRGB'); ramp.color_ramp.elements[0].color = (0.004, 0.006, 0.02, 1); ramp.color_ramp.elements[1].color = (0.03, 0.05, 0.12, 1)
nt.links.new(tc.outputs['Generated'], mp.inputs['Vector']); nt.links.new(mp.outputs['Vector'], grad.inputs['Vector'])
nt.links.new(grad.outputs['Fac'], ramp.inputs['Fac']); nt.links.new(ramp.outputs['Color'], bg.inputs['Color'])
for nm, loc, e, col, sz in (('moon', (-8, -10, 12), 400, (0.6, 0.7, 1.0), 4), ('warm', (6, -6, 3), 300, (1.0, 0.7, 0.4), 3)):
    l_ = bpy.data.lights.new(nm, 'AREA'); l_.energy = e; l_.color = col; l_.size = sz
    o = link(bpy.data.objects.new(nm, l_)); o.location = loc
    c = o.constraints.new('TRACK_TO'); c.target = STAR if nm == 'moon' else BALL; c.track_axis = 'TRACK_NEGATIVE_Z'; c.up_axis = 'UP_Y'

# ---------------- camera: chases the ball around the tree, pulls back for the finale ----------------
camd = bpy.data.cameras.new('cam'); camd.lens = 26; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24
camd.dof.use_dof = True; camd.dof.aperture_fstop = 4.0
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
tr = cam.constraints.new('TRACK_TO'); tr.target = tgt; tr.track_axis = 'TRACK_NEGATIVE_Z'; tr.up_axis = 'UP_Y'
camd.dof.focus_object = BALL
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
for f in range(NF):
    p = BP[f]
    ang = math.atan2(p[1], p[0]) + 0.45                   # sit slightly behind the ball on its orbit
    rr = math.hypot(p[0], p[1]) + 2.3
    CP[f] = (rr * math.cos(ang), rr * math.sin(ang), p[2] + 1.0)
    TP[f] = BP[f] * 0.8 + np.array([0, 0, BP[f][2] - 0.15]) * 0.2
# smooth camera
def smooth(X, k):
    Y = X.copy()
    for f in range(1, len(X)): Y[f] = Y[f - 1] + (X[f] - Y[f - 1]) * (1 - math.exp(-1 / FPS / k))
    return Y
CP = smooth(CP, 0.35); TP = smooth(TP, 0.2)
# intro: start wide on the star; finale: pull back to the whole tree
wide_c = np.array([0.0, -13.5, 3.6]); wide_t = np.array([0, 0, 3.9])
intro = np.clip(1 - (T - 0.6) / 1.6, 0, 1); intro = intro * intro * (3 - 2 * intro)
fin = np.clip((T - (FINAL - 0.8)) / 2.5, 0, 1); fin = fin * fin * (3 - 2 * fin)
wv = np.maximum(intro, fin)[:, None]
star_c = np.array([0.0, -3.2, ZT + 0.6]); star_t = np.array([0, 0, ZT - 0.2])
CPf = CP * (1 - wv) + np.where(T[:, None] < 5, star_c, wide_c) * wv
TPf = TP * (1 - wv) + np.where(T[:, None] < 5, star_t, wide_t) * wv
for i in range(3): key_series(cam, 'location', i, CPf[:, i]); key_series(tgt, 'location', i, TPf[:, i])
fo = np.where(fin > 0.5, 1, 0)
print('objects', len(bpy.data.objects), 'NF', NF, 'track', round(SLEN, 1), 'V', round(V, 2), 'keys', len(KEYS))
if MODE == 'test':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS)); sc.frame_set(f)
        sc.render.filepath = f'{HERE}/test_{f:04d}.png'
        t1 = time.time(); bpy.ops.render.render(write_still=True); print('frame', f, 'sec', round(time.time() - t1, 1))
elif MODE == 'range':
    sc.frame_start, sc.frame_end = int(args[1]), int(args[2])
    sc.render.image_settings.file_format = 'PNG'; sc.render.filepath = f'{HERE}/frames/f_'
    bpy.ops.render.render(animation=True)
