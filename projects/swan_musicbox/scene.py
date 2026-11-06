"""Swan Lake - a giant antique music box on a moonlit lake.
A pinned brass cylinder (pins = pearls) turns once through the whole piece; every pin plucks one steel comb tooth
exactly on its note (the tooth bends, snaps back, quivers and glints). The cylinder drives a real gear train up to a
spinning air-brake governor; a spring barrel with a winding key sits on the other end. A swan automaton glides on a
brass rail around the box, leaving ripples; at the climax feathers swirl and the swan spreads its wings on the final chord."""
import os, sys, json, math, time, colorsys
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); COMB = json.load(open(f'{HERE}/comb.json')); NOTES = json.load(open(f'{HERE}/notes.json'))
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
sc.view_settings.exposure = 0.35


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


def stmat(name, col, rough=0.25, gain=8.0, metal=1.0):
    """metal whose emission = object colour * object alpha (for teeth glints)"""
    m = mat(name, col, rough, **{'Metallic': metal}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    oi = nt.nodes.new('ShaderNodeObjectInfo')
    nt.links.new(oi.outputs['Color'], b.inputs['Emission Color'])
    mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = gain
    nt.links.new(oi.outputs['Alpha'], mul.inputs[0]); nt.links.new(mul.outputs[0], b.inputs['Emission Strength'])
    return m


def smooth_obj(o):
    for p in o.data.polygons: p.use_smooth = True
    return o


def box(loc, size, mt, rot=(0, 0, 0), bev=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=rot)
    o = bpy.context.object; o.scale = size; o.data.materials.append(mt)
    if bev: m = o.modifiers.new('b', 'BEVEL'); m.width = bev; m.segments = 2
    return o


def gear(name, n, R, th, mt, x, cy, cz):
    """spur gear, axis along X, centred at (x, cy, cz). returns object (rotation_euler.x animates it)"""
    m_ = 2 * R / n; ro, ri = R + m_, R - 1.25 * m_
    bm = bmesh.new(); pts = []
    for k in range(n):
        a0 = 2 * math.pi * k / n; da = 2 * math.pi / n
        for f, r in ((0.0, ri), (0.18, ro), (0.42, ro), (0.6, ri)):
            a = a0 + f * da; pts.append(bm.verts.new((0, r * math.cos(a), r * math.sin(a))))
    face = bm.faces.new(pts)
    r = bmesh.ops.extrude_face_region(bm, geom=[face])
    for v in [e for e in r['geom'] if isinstance(e, bmesh.types.BMVert)]: v.co.x += th
    for v in bm.verts: v.co.x -= th / 2
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    o = link(bpy.data.objects.new(name, me)); o.location = (x, cy, cz); o.data.materials.append(mt)
    o.rotation_mode = 'XYZ'
    # hub + spokes look: a darker inner disc
    return o


# ---------------- materials ----------------
WOOD = mat('wood', (0.16, 0.05, 0.025), 0.35, **{'Coat Weight': 0.8, 'Coat Roughness': 0.05})
WOOD2 = mat('wood2', (0.26, 0.10, 0.04), 0.4, **{'Coat Weight': 0.6})
BRASS = mat('brass', (0.95, 0.68, 0.32), 0.22, **{'Metallic': 1.0})
DBRASS = mat('dbrass', (0.6, 0.4, 0.18), 0.3, **{'Metallic': 1.0})
STEEL = mat('steel', (0.8, 0.82, 0.86), 0.18, **{'Metallic': 1.0})
VELVET = mat('velvet', (0.012, 0.02, 0.08), 0.9, **{'Sheen Weight': 1.0})
PEARL = mat('pearl', (0.95, 0.93, 0.9), 0.12, **{'Coat Weight': 1.0, 'Subsurface Weight': 0.2})
STONE = mat('stone', (0.22, 0.24, 0.3), 0.7)
WHITE = mat('swanw', (0.95, 0.95, 0.97), 0.55, **{'Subsurface Weight': 0.25, 'Sheen Weight': 0.6})

# ---------------- lake, island, far shore ----------------
bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, 0)); water = bpy.context.object
WM = mat('water', (0.01, 0.02, 0.05), 0.04, **{'Coat Weight': 0.0})
nt = WM.node_tree; bsdf = nt.nodes['Principled BSDF']
nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 3.0; nz.inputs['Detail'].default_value = 3
bmp = nt.nodes.new('ShaderNodeBump'); bmp.inputs['Strength'].default_value = 0.08
nt.links.new(nz.outputs['Fac'], bmp.inputs['Height']); nt.links.new(bmp.outputs['Normal'], bsdf.inputs['Normal'])
water.data.materials.append(WM)
bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=2.45, depth=0.5, location=(0, 0, 0.1)); isl = bpy.context.object; isl.data.materials.append(STONE)
m = isl.modifiers.new('b', 'BEVEL'); m.width = 0.06; m.segments = 3
bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=2.3, depth=0.04, location=(0, 0, 0.37)); bpy.context.object.data.materials.append(DBRASS)
# far shore: dark hills, a castle with a few warm windows
SIL = mat('sil', (0.006, 0.008, 0.016), 0.9)
for i in range(14):
    x = -40 + i * 6.2 + rng.uniform(-2, 2); h = rng.uniform(2.5, 6)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(x, 46 + rng.uniform(-3, 3), 0)); o = bpy.context.object
    o.scale = (rng.uniform(5, 8), 3, h); o.data.materials.append(SIL)
WIN = emis('win', (1.0, 0.6, 0.25), 6.0)
CX, CY = 9, 42
for dx, w_, h_ in ((0, 4, 6), (-3.2, 1.4, 9), (3.2, 1.4, 9), (0, 1.2, 12)):
    box((CX + dx, CY, h_ / 2), (w_, w_, h_), SIL)
    bpy.ops.mesh.primitive_cone_add(radius1=w_ * 0.75, depth=2.4, location=(CX + dx, CY, h_ + 1.2)); bpy.context.object.data.materials.append(SIL)
for k in range(7):
    box((CX + rng.uniform(-3.4, 3.4), CY - 2.1, rng.uniform(2, 8)), (0.25, 0.1, 0.4), WIN)
# stars + moon
STARM = emis('starm', (0.8, 0.85, 1.0), 4.0)
for i in range(260):
    a = rng.uniform(-1.2, 1.2); e = rng.uniform(0.12, 0.9); r = 90
    bpy.ops.mesh.primitive_ico_sphere_add(radius=rng.uniform(0.05, 0.16), subdivisions=1,
                                          location=(r * math.sin(a), r * math.cos(a), r * math.sin(e) * 0.8 + 6))
    bpy.context.object.data.materials.append(STARM)
bpy.ops.mesh.primitive_uv_sphere_add(radius=3.6, segments=32, ring_count=16, location=(-11, 80, 15))
MOON = bpy.context.object; MOON.data.materials.append(emis('moonm', (0.95, 0.95, 1.0), 6.0)); smooth_obj(MOON)

# ---------------- the music box case ----------------
BX, BY, BZ0 = 4.4, 2.1, 0.39          # outer size x, y ; base z
box((0, 0, BZ0 + 0.06), (BX, BY, 0.12), WOOD, bev=0.02)                       # floor
box((0, -BY / 2 + 0.05, BZ0 + 0.3), (BX, 0.1, 0.5), WOOD, bev=0.015)          # low front wall
box((0, BY / 2 - 0.05, BZ0 + 0.55), (BX, 0.1, 1.0), WOOD, bev=0.015)          # back wall
for sx in (-1, 1): box((sx * (BX / 2 - 0.05), 0, BZ0 + 0.3), (0.1, BY, 0.5), WOOD, bev=0.015)
for z in (BZ0 + 0.56, BZ0 + 1.06):                                           # brass trim
    box((0, -BY / 2 - 0.005, z - 0.5 if z < 1 else z - 0.5), (BX + 0.02, 0.02, 0.025), BRASS)
box((0, -BY / 2 - 0.012, BZ0 + 0.3), (BX * 0.6, 0.01, 0.22), WOOD2)            # front inlay panel
for sx in (-1, 1):
    for sy in (-1, 1):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.11, location=(sx * (BX / 2 - 0.12), sy * (BY / 2 - 0.12), BZ0 - 0.02)); bpy.context.object.data.materials.append(BRASS)
# lid, opened back ~105 deg, velvet inside with a brass frame
lid = link(bpy.data.objects.new('lidp', None)); lid.location = (0, BY / 2, BZ0 + 1.05)
for o in (box((0, -BY / 2, 0.05), (BX, BY, 0.1), WOOD, bev=0.02), box((0, -BY / 2, -0.004), (BX - 0.3, BY - 0.3, 0.01), VELVET),
          box((0, -BY / 2, -0.002), (BX - 0.22, BY - 0.22, 0.008), BRASS)):
    o.parent = lid
MIRROR = mat('mirror', (0.9, 0.92, 0.95), 0.02, **{'Metallic': 1.0})
bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.5, depth=0.01, location=(0, -BY / 2, -0.012))
mo = bpy.context.object; mo.scale = (1.5, 1, 1); mo.data.materials.append(MIRROR); mo.parent = lid
bpy.ops.mesh.primitive_torus_add(major_radius=0.52, minor_radius=0.025, location=(0, -BY / 2, -0.015))
mr = bpy.context.object; mr.scale = (1.5, 1, 1); mr.data.materials.append(BRASS); mr.parent = lid
lid.rotation_euler = (math.radians(-105), 0, 0)

# ---------------- cylinder ----------------
CYY, CYZ, R = 0.28, BZ0 + 0.78, 0.48
CXL = 1.42                                   # half length
OMEGA = 2 * math.pi / 72.0                   # one turn = 72 s (whole piece fits on one revolution)
cylp = link(bpy.data.objects.new('cyl', None)); cylp.location = (0, CYY, CYZ); cylp.rotation_mode = 'XYZ'
bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=R, depth=2 * CXL, rotation=(0, math.pi / 2, 0))
cyl = bpy.context.object; cyl.data.materials.append(BRASS); cyl.parent = cylp; smooth_obj(cyl)
m = cyl.modifiers.new('b', 'BEVEL'); m.width = 0.01; m.segments = 2
bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.05, depth=2 * CXL + 0.5, rotation=(0, math.pi / 2, 0))
bpy.context.object.data.materials.append(STEEL); bpy.context.object.parent = cylp
for sx in (-1, 1):                                       # bearing posts (fixed)
    rod((sx * (CXL + 0.2), CYY, BZ0 + 0.12), (sx * (CXL + 0.2), CYY, CYZ), 0.06, BRASS, 16)
    box((sx * (CXL + 0.2), CYY, BZ0 + 0.16), (0.2, 0.3, 0.06), BRASS)

# ---------------- comb + teeth (one tooth per pitch, bass on the left) ----------------
PITCHES = sorted({c[1] for c in COMB}); NT = len(PITCHES)
PITCH_X = {p: -CXL + 0.08 + (2 * CXL - 0.16) * i / (NT - 1) for i, p in enumerate(PITCHES)}
TIPY = CYY - R - 0.012                       # tooth tip just inside the pin circle
COMBY = TIPY - 0.62                          # comb base
box((0, COMBY - 0.06, CYZ - 0.02), (2 * CXL + 0.3, 0.2, 0.16), STEEL, bev=0.01)
for sx in (-1, 1): rod((sx * (CXL + 0.05), COMBY - 0.06, BZ0 + 0.12), (sx * (CXL + 0.05), COMBY - 0.06, CYZ - 0.08), 0.05, BRASS, 12)
box((0, COMBY - 0.06, CYZ + 0.075), (2 * CXL + 0.32, 0.22, 0.03), BRASS)      # clamp bar
for k in range(9):
    bpy.ops.mesh.primitive_cylinder_add(radius=0.022, depth=0.03, location=(-CXL + k * CXL / 4, COMBY - 0.06, CYZ + 0.1)); bpy.context.object.data.materials.append(STEEL)
TMAT = stmat('tooth', (0.82, 0.84, 0.9), 0.15, 9.0)
TEETH = {}
tw = (2 * CXL - 0.16) / (NT - 1) * 0.72
for p in PITCHES:
    x = PITCH_X[p]; L = TIPY - (COMBY + 0.04)
    piv = link(bpy.data.objects.new(f'tp{p}', None)); piv.location = (x, COMBY + 0.04, CYZ); piv.rotation_mode = 'XYZ'
    lw = 1.0 + (max(PITCHES) - p) / 40                   # bass teeth a bit thicker (brass weights underneath)
    o = box((0, L / 2, 0), (tw, L, 0.016 * lw), TMAT); o.parent = piv
    if p < 56:
        w_ = box((0, L * 0.45, -0.03), (tw * 0.9, L * 0.35, 0.03), BRASS); w_.parent = piv
    TEETH[p] = (piv, o, L)
# pins (pearls) on the cylinder: one per comb note; contact at the front (angle pi) exactly at the note time
PLUCK_A = math.pi + 0.015
pins = []
for t, p, v in COMB:
    phi = PLUCK_A - OMEGA * t
    x = PITCH_X[p]
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.011, segments=10, ring_count=6, location=(x, (R + 0.008) * math.cos(phi), (R + 0.008) * math.sin(phi)))
    o = bpy.context.object; o.data.materials.append(PEARL); o.parent = cylp; pins.append(o)
key_series(cylp, 'rotation_euler', 0, OMEGA * T)

# tooth motion: pushed down by the pin, released at the note, quivers and glints
for p, (piv, o, L) in TEETH.items():
    ang = np.zeros(NF); glow = np.zeros(NF)
    for t, pp, v in COMB:
        if pp != p: continue
        t_c = t - 0.07 / OMEGA                                     # contact ~0.07 rad before release
        a = (T - t_c) / (t - t_c)
        push = np.where((a >= 0) & (a < 1), np.clip(a, 0, 1) ** 1.5, 0) * (0.022 / L)
        rel = T - t
        quiv = np.where(rel >= 0, np.cos(rel * 2 * math.pi * 5.5) * np.exp(-rel / 0.35), 0) * (0.010 / L) * (0.5 + v / 254)
        ang += push + quiv
        glow = np.maximum(glow, np.where(rel >= -1 / FPS, np.exp(-np.clip(rel, 0, None) / 0.45), 0) * (0.4 + v / 160))
    key_series(piv, 'rotation_euler', 0, -ang)       # positive X rotation lifts tip; minus = pushed down
    key_alpha(o, glow, (0.55, 0.75, 1.0))

# ---------------- gear train to the governor (right) + spring barrel & key (left) ----------------
GX = CXL + 0.08
G1 = gear('g1', 56, 0.42, 0.04, BRASS, GX, CYY, CYZ)
mod = 2 * 0.42 / 56
def place_mesh(cA, rA, rB, ang):
    return (cA[0] + (rA + rB) * math.cos(ang), cA[1] + (rA + rB) * math.sin(ang))
g2c = place_mesh((CYY, CYZ), 0.42, 10 * mod / 2, math.radians(70))
G2 = gear('g2', 10, 10 * mod / 2, 0.05, STEEL, GX, *g2c)
W3 = gear('w3', 40, 0.30, 0.035, BRASS, GX + 0.09, *g2c)
mod3 = 2 * 0.30 / 40
g4c = place_mesh(g2c, 0.30, 8 * mod3 / 2, math.radians(160))
G4 = gear('g4', 8, 8 * mod3 / 2, 0.05, STEEL, GX + 0.09, *g4c)
W5 = gear('w5', 32, 0.22, 0.03, BRASS, GX + 0.17, *g4c)
mod5 = 2 * 0.22 / 32
g6c = place_mesh(g4c, 0.22, 8 * mod5 / 2, math.radians(95))
G6 = gear('g6', 8, 8 * mod5 / 2, 0.05, STEEL, GX + 0.17, *g6c)
w1 = OMEGA; w2 = -w1 * 56 / 10; w4 = -w2 * 40 / 8; w6 = -w4 * 32 / 8
for g, w_, n in ((G1, w1, 56), (G2, w2, 10), (W3, w2, 40), (G4, w4, 8), (W5, w4, 32), (G6, w6, 8)):
    ph = (math.pi / n) if n <= 10 else 0.0
    key_series(g, 'rotation_euler', 0, w_ * T + ph)
for c, x0, x1 in ((g2c, GX - 0.05, GX + 0.14), (g4c, GX + 0.04, GX + 0.22), (g6c, GX + 0.12, GX + 0.42)):
    rod((x0, *c), (x1, *c), 0.012, STEEL, 8)
# air-brake fan on the last arbor
fan = link(bpy.data.objects.new('fan', None)); fan.location = (GX + 0.36, *g6c); fan.rotation_mode = 'XYZ'
for s in (-1, 1):
    v_ = box((0, s * 0.09, 0), (0.06, 0.16, 0.004), BRASS); v_.parent = fan
key_series(fan, 'rotation_euler', 0, w6 * T)
# spring barrel + winding key (left end)
BXL = -CXL - 0.08
SB = gear('barrel', 70, 0.36, 0.06, BRASS, BXL - 0.04, CYY, CYZ - 0.0)
bpy.data.objects['barrel'].location = (BXL - 0.04, CYY - 0.0, CYZ)
P0 = gear('p0', 14, 0.36 * 14 / 70, 0.05, STEEL, BXL - 0.04, CYY, CYZ)
# barrel sits below-front of the cylinder pinion
bc = place_mesh((CYY, CYZ), 0.36 * 14 / 70, 0.36, math.radians(-120))
SB.location = (BXL - 0.04, *bc); P0.location = (BXL - 0.04, CYY, CYZ)
bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.3, depth=0.22, location=(BXL - 0.2, *bc), rotation=(0, math.pi / 2, 0))
drum = bpy.context.object; drum.data.materials.append(DBRASS)
wind = np.clip((T - 0.15) / 1.1, 0, 1)
key_series(P0, 'rotation_euler', 0, OMEGA * T + math.pi / 14)
key_series(SB, 'rotation_euler', 0, -OMEGA * T * 14 / 70 + wind * 2.2)
# winding key outside the left wall
kp = link(bpy.data.objects.new('keyp', None)); kp.location = (-BX / 2 - 0.02, *bc); kp.rotation_mode = 'XYZ'
for o in (rod((0, 0, 0), (-0.35, 0, 0), 0.03, BRASS, 12),):
    o.parent = kp
bt_ = box((-0.4, 0, 0), (0.05, 0.42, 0.16), BRASS, bev=0.02); bt_.parent = kp
for s in (-1, 1):
    bpy.ops.mesh.primitive_torus_add(major_radius=0.1, minor_radius=0.025, location=(-0.4, s * 0.2, 0), rotation=(0, math.pi / 2, 0))
    bpy.context.object.data.materials.append(BRASS); bpy.context.object.parent = kp
key_series(kp, 'rotation_euler', 0, wind * 2 * math.pi * 1.5 + 0.02 * T)

# ---------------- swan automaton on a brass rail ----------------
RR = 3.25
bpy.ops.mesh.primitive_torus_add(major_radius=RR, minor_radius=0.018, major_segments=160, minor_segments=8, location=(0, 0, 0.03)); bpy.context.object.data.materials.append(BRASS)
for k in range(12):                                   # little posts holding the rail
    a = k * math.pi / 6; rod((RR * math.cos(a), RR * math.sin(a), -0.3), (RR * math.cos(a), RR * math.sin(a), 0.03), 0.015, BRASS, 6)
swan = link(bpy.data.objects.new('swan', None)); swan.rotation_mode = 'XYZ'
def part(o): o.parent = swan; return o
bpy.ops.mesh.primitive_uv_sphere_add(radius=1, segments=32, ring_count=16, location=(0, 0, 0.17)); b_ = part(smooth_obj(bpy.context.object)); b_.scale = (0.42, 0.23, 0.17); b_.data.materials.append(WHITE)
bpy.ops.mesh.primitive_cone_add(radius1=0.12, depth=0.3, location=(-0.42, 0, 0.27), rotation=(0, -math.radians(65), 0)); t_ = part(smooth_obj(bpy.context.object)); t_.scale = (1, 1.4, 1); t_.data.materials.append(WHITE)
cu = bpy.data.curves.new('neck', 'CURVE'); cu.dimensions = '3D'; cu.bevel_depth = 0.042; cu.bevel_resolution = 4
sp = cu.splines.new('BEZIER'); sp.bezier_points.add(2)
for bp_, co, hl, hr in zip(sp.bezier_points, ((0.28, 0, 0.24), (0.27, 0, 0.55), (0.40, 0, 0.78)),
                           ((0.2, 0, 0.18), (0.36, 0, 0.45), (0.33, 0, 0.78)), ((0.36, 0, 0.30), (0.18, 0, 0.66), (0.47, 0, 0.78))):
    bp_.co = co; bp_.handle_left = hl; bp_.handle_right = hr
pts_ = sp.bezier_points; pts_[0].radius = 1.35; pts_[1].radius = 0.95; pts_[2].radius = 0.8
neck = part(link(bpy.data.objects.new('neck', cu))); neck.data.materials.append(WHITE)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.06, location=(0.43, 0, 0.79)); hd = part(smooth_obj(bpy.context.object)); hd.scale = (1.3, 0.9, 0.9); hd.data.materials.append(WHITE)
bpy.ops.mesh.primitive_cone_add(radius1=0.026, depth=0.12, location=(0.54, 0, 0.765), rotation=(0, math.radians(105), 0)); part(bpy.context.object).data.materials.append(mat('beak', (1.0, 0.35, 0.05), 0.4))
BLK = mat('blk', (0.01, 0.01, 0.01), 0.3)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.026, location=(0.495, 0, 0.785)); part(bpy.context.object).data.materials.append(BLK)
for s in (-1, 1):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.009, location=(0.46, s * 0.045, 0.8)); part(bpy.context.object).data.materials.append(BLK)
# crown (automaton touch)
bpy.ops.mesh.primitive_cone_add(vertices=5, radius1=0.045, radius2=0.06, depth=0.04, location=(0.42, 0, 0.86)); part(bpy.context.object).data.materials.append(BRASS)
# wings: hinge at the shoulder, 7 feathers each, folded along the body -> fan open & lift on the final chord
WINGS = []
for s in (-1, 1):
    hinge = part(link(bpy.data.objects.new(f'wing{s}', None))); hinge.location = (0.12, s * 0.16, 0.28); hinge.rotation_mode = 'XYZ'
    feathers = []
    for k in range(7):
        fp = link(bpy.data.objects.new(f'fp{s}{k}', None)); fp.parent = hinge; fp.rotation_mode = 'XYZ'
        Lf = 0.30 + 0.05 * k
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1, segments=16, ring_count=8, location=(-Lf / 2, 0, 0))
        f_ = smooth_obj(bpy.context.object); f_.scale = (Lf / 2, 0.06, 0.018); f_.data.materials.append(WHITE); f_.parent = fp
        feathers.append(fp)
    WINGS.append((s, hinge, feathers))
# swan path: two laps, easing to a stop at the front (facing the camera side) on the final chord
A_END = -math.pi / 2
u = np.clip((T - 0.8) / (FINAL - 0.8), 0, 1); ue = 1 - (1 - u) ** 1.6
ang = A_END - 2 * 2 * math.pi * (1 - ue)
SW = np.stack([RR * np.cos(ang), RR * np.sin(ang)], 1)
bob = 0.012 * np.sin(T * 2.1)
key_series(swan, 'location', 0, SW[:, 0]); key_series(swan, 'location', 1, SW[:, 1]); key_series(swan, 'location', 2, 0.02 + bob)
yaw = ang + math.pi / 2                                 # tangent (counter-clockwise)
fin = np.clip((T - FINAL + 0.25) / 0.9, 0, 1); fin = fin * fin * (3 - 2 * fin)
yaw = yaw * (1 - fin) + (math.pi * 1.5 - 0.35) * fin      # turn to show its profile+front to camera at the end
key_series(swan, 'rotation_euler', 2, yaw); key_series(swan, 'rotation_euler', 1, 0.02 * np.sin(T * 1.3))
for s, hinge, feathers in WINGS:
    # folded: lying back along the body ; spread: lifted ~75 deg, swept out
    key_series(hinge, 'rotation_euler', 0, s * (0.12 + fin * 1.25))
    key_series(hinge, 'rotation_euler', 2, s * (-0.05 - fin * 0.35))
    key_series(hinge, 'rotation_euler', 1, -0.1 - fin * 0.25)
    for k, fp in enumerate(feathers):
        key_series(fp, 'rotation_euler', 1, -(0.03 * k) - fin * (0.13 * k))
        fp.rotation_euler[2] = s * (0.02 * k)

# ---------------- ripples: swan wake + bass notes from the island ----------------
RIP = emis('rip', (0.65, 0.75, 1.0), 1.0)
nt_ = RIP.node_tree; em_ = nt_.nodes['Emission']; oi = nt_.nodes.new('ShaderNodeObjectInfo')
mul = nt_.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = 0.7
nt_.links.new(oi.outputs['Alpha'], mul.inputs[0]); nt_.links.new(mul.outputs[0], em_.inputs['Strength'])
ev = []
for tw_ in np.arange(1.0, FINAL + 3.5, 0.75):
    f = min(NF - 1, int(tw_ * FPS)); ev.append((tw_, SW[f, 0], SW[f, 1], 0.9))
bass_comb = sorted({round(c[0], 3) for c in COMB if c[1] < 56})
for k, tb_ in enumerate(bass_comb):
    a_ = -math.pi / 2 + (k % 5 - 2) * 0.5; ev.append((tb_, 2.5 * math.cos(a_), 2.5 * math.sin(a_), 1.4))
ev.append((FINAL, 0, 0, 3.0))
ev.sort()
POOL = 26; rings = []
for i in range(POOL):
    bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=0.008, major_segments=64, minor_segments=4, location=(0, 0, 0.012))
    o = bpy.context.object; o.data.materials.append(RIP); rings.append(o)
LIFE = 2.6
for i, o in enumerate(rings):
    X = np.zeros(NF); Y = np.zeros(NF); S = np.full(NF, 0.001); A = np.zeros(NF)
    for (t0, x, y, sz) in ev[i::POOL]:
        rel = T - t0; on = (rel >= 0) & (rel < LIFE)
        X[on] = x; Y[on] = y; S[on] = 0.15 + sz * (rel[on] / LIFE) ** 0.6; A[on] = (1 - rel[on] / LIFE) ** 1.5
    key_series(o, 'location', 0, X); key_series(o, 'location', 1, Y)
    key_series(o, 'scale', 0, S); key_series(o, 'scale', 1, S)
    key_alpha(o, A, (1, 1, 1))

# ---------------- feathers swirling at the climax ----------------
FEA = mat('fea', (0.97, 0.97, 1.0), 0.6, **{'Emission Color': (0.8, 0.85, 1.0, 1), 'Emission Strength': 0.3})
T_A3 = [s for n, s in E['secs'] if n == 'A3'][0]
for i in range(110):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, segments=10, ring_count=6)
    o = smooth_obj(bpy.context.object); o.scale = (0.05, 0.016, 0.004); o.data.materials.append(FEA); o.rotation_mode = 'XYZ'
    t_start = T_A3 + rng.uniform(0, FINAL - T_A3) if i < 50 else FINAL + rng.uniform(-0.05, 0.25)
    r0 = rng.uniform(0.6, 3.0); a0 = rng.uniform(0, 2 * math.pi); z0 = rng.uniform(0.3, 1.2)
    burst = i >= 50
    rel = np.clip(T - t_start, 0, None); live = T >= t_start
    a = a0 + rel * (0.9 if not burst else 1.4) / max(0.6, r0 * 0.5)
    r = r0 + rel * (0.25 if not burst else 0.55)
    z = z0 + rel * (0.45 if not burst else 1.1) + 0.08 * np.sin(rel * 3 + i)
    if burst:                                           # start from the swan's wings
        f = min(NF - 1, int(FINAL * FPS)); sx_, sy_ = SW[f]
        k_ = np.clip(rel / 0.6, 0, 1)
        X = (sx_ + rng.normal(0, 0.15)) * (1 - k_) + r * np.cos(a) * k_; Y = (sy_ + rng.normal(0, 0.15)) * (1 - k_) + r * np.sin(a) * k_
        z = 0.5 * (1 - k_) + z * k_
    else:
        X = r * np.cos(a); Y = r * np.sin(a)
    X = np.where(live, X, 0); Y = np.where(live, Y, 0); Z = np.where(live, z, -3)
    for j, arr in enumerate((X, Y, Z)): key_series(o, 'location', j, arr)
    for j in range(3): key_series(o, 'rotation_euler', j, rel * rng.uniform(1, 4) + rng.uniform(0, 6))

# ---------------- world & lights ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True; nt = w.node_tree
bg = nt.nodes['Background']; bg.inputs['Strength'].default_value = 1.0
grad = nt.nodes.new('ShaderNodeTexGradient'); tc = nt.nodes.new('ShaderNodeTexCoord'); mp = nt.nodes.new('ShaderNodeMapping')
mp.inputs['Rotation'].default_value = (0, -math.pi / 2, 0)
ramp = nt.nodes.new('ShaderNodeValToRGB'); ramp.color_ramp.elements[0].color = (0.003, 0.005, 0.018, 1); ramp.color_ramp.elements[1].color = (0.02, 0.035, 0.09, 1)
nt.links.new(tc.outputs['Generated'], mp.inputs['Vector']); nt.links.new(mp.outputs['Vector'], grad.inputs['Vector'])
nt.links.new(grad.outputs['Fac'], ramp.inputs['Fac']); nt.links.new(ramp.outputs['Color'], bg.inputs['Color'])
tg0 = link(bpy.data.objects.new('lt', None)); tg0.location = (0, 0, 1.0)
for nm, rot, e, col in (('moonl', (math.radians(55), 0, math.radians(200)), 1.6, (0.55, 0.65, 1.0)),
                        ('fill', (math.radians(60), 0, math.radians(-35)), 0.55, (0.6, 0.7, 1.0)),
                        ('warm', (math.radians(55), 0, math.radians(35)), 0.45, (1.0, 0.75, 0.45))):
    l_ = bpy.data.lights.new(nm, 'SUN'); l_.energy = e; l_.color = col; l_.angle = math.radians(3)
    o = link(bpy.data.objects.new(nm, l_)); o.rotation_euler = rot
lp = bpy.data.lights.new('lamp', 'POINT'); lp.energy = 60; lp.color = (1.0, 0.8, 0.55); lp.shadow_soft_size = 0.08
lo_ = link(bpy.data.objects.new('lamp', lp)); lo_.location = (0, -0.9, CYZ + 1.1)
# final chord: everything glows a little brighter
fl = bpy.data.lights.new('finl', 'POINT'); fl.color = (0.8, 0.85, 1.0); fl.shadow_soft_size = 1.0
flo = link(bpy.data.objects.new('finl', fl)); flo.location = (0, -2.6, 1.6); flo.visible_glossy = False
fg = np.clip((T - FINAL) / 0.3, 0, 1) * np.exp(-np.clip(T - FINAL - 0.3, 0, None) / 3.0)
for f in range(NF):
    if f in (0, NF - 1) or fg[f] > 0 or (f + 1 < NF and fg[f + 1] > 0): fl.energy = 5 + 300 * fg[f]; fl.keyframe_insert('energy', frame=f)

# ---------------- camera: wide establishing shot, then calm medium-wide framings (never close) ----------------
camd = bpy.data.cameras.new('cam'); camd.lens = 28; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24
camd.dof.use_dof = True; camd.dof.aperture_fstop = 8.0; camd.clip_end = 300
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
tr = cam.constraints.new('TRACK_TO'); tr.target = tgt; tr.track_axis = 'TRACK_NEGATIVE_Z'; tr.up_axis = 'UP_Y'
camd.dof.focus_object = tgt
S_ = dict(E['secs'])
KF = [  # time, camera, target
    (0.0, (-3.0, -26.0, 8.0), (0, 0, 2.6)),
    (3.4, (-2.2, -12.6, 6.4), (0, 0, 1.0)),
    (S_['A2'] - 1, (-0.6, -12.0, 6.0), (0, 0, 1.0)),
    (S_['A2'] + 3, (5.4, -10.8, 6.0), (0.4, 0, 1.0)),
    (S_['B1'] + 1, (4.2, -12.5, 8.2), (0, 0, 0.8)),
    (S_['B2'] + 1, (-5.0, -11.8, 6.6), (-0.3, 0, 1.0)),
    (S_['A3'] - 1, (-4.0, -12.8, 6.2), (0, -0.4, 1.0)),
    (FINAL - 1.5, (0.0, -14.2, 5.2), (0, -0.9, 1.15)),
    (FINAL + 4.5, (0.0, -13.0, 4.9), (0, -0.9, 1.2)),
]
kt = np.array([k[0] for k in KF]); kc = np.array([k[1] for k in KF]); ktg = np.array([k[2] for k in KF])
def interp(arr):
    out = np.zeros((NF, 3))
    for f in range(NF):
        t = T[f]; i = min(len(kt) - 2, max(0, np.searchsorted(kt, t) - 1))
        a = np.clip((t - kt[i]) / (kt[i + 1] - kt[i]), 0, 1); a = a * a * (3 - 2 * a)
        out[f] = arr[i] * (1 - a) + arr[i + 1] * a
    return out
CPf = interp(kc); TPf = interp(ktg)
for i in range(3): key_series(cam, 'location', i, CPf[:, i]); key_series(tgt, 'location', i, TPf[:, i])

print('objects', len(bpy.data.objects), 'NF', NF, 'pins', len(pins), 'teeth', NT)
if MODE == 'test':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS)); sc.frame_set(f)
        sc.render.filepath = f'{HERE}/test_{f:04d}.png'
        t1 = time.time(); bpy.ops.render.render(write_still=True); print('frame', f, 'sec', round(time.time() - t1, 1))
elif MODE == 'range':
    sc.frame_start, sc.frame_end = int(args[1]), int(args[2])
    sc.render.image_settings.file_format = 'PNG'; sc.render.filepath = f'{HERE}/frames/f_'
    bpy.ops.render.render(animation=True)
