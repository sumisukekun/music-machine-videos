"""Happy Clap - a wooden karakuri marionette theatre plays "If You're Happy and You Know It".
Brass cannons lob marbles onto a rainbow marimba (the melody). Above the stage a pinned cam barrel turns;
each pin trips the follower, the marionette cross jerks and the puppet claps / stomps / taps its shoulders.
At every verse the barrel slides to the next track (click!) and the sign flips. Finale: everything + confetti."""
import os, sys, json, math, colorsys, time
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json'))
FPS = 30; FINAL = E['final']; T_END = FINAL + 4.4; NF = int(T_END * FPS); BEAT = E['beat']
G = 9.8; TAU = 0.5
rng = np.random.default_rng(11)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'
sc.cycles.samples = int(os.environ.get('SAMP', 5))
sc.cycles.use_adaptive_sampling = True; sc.cycles.adaptive_threshold = 0.06
sc.cycles.use_denoising = True
for k, v in (('denoiser', 'OPENIMAGEDENOISE'), ('denoising_quality', 'HIGH'), ('denoising_prefilter', 'ACCURATE'),
             ('use_fast_gi', True), ('ao_bounces_render', 1)):
    try: setattr(sc.cycles, k, v)
    except Exception: pass
sc.cycles.max_bounces = 5; sc.cycles.diffuse_bounces = 2; sc.cycles.glossy_bounces = 2; sc.cycles.transparent_max_bounces = 6
sc.cycles.caustics_reflective = False; sc.cycles.caustics_refractive = False
sc.cycles.sample_clamp_indirect = 4
sc.render.resolution_x, sc.render.resolution_y = 1080, 1920
sc.render.resolution_percentage = int(os.environ.get('PCT', 50))
sc.render.fps = FPS
sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.3
try: sc.render.motion_blur_position = 'START'
except Exception: pass
sc.render.use_persistent_data = True; sc.render.use_overwrite = False
try: sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Punchy'
except Exception: pass
sc.frame_start, sc.frame_end = 0, NF - 1
TT = np.arange(NF) / FPS


def link(o): sc.collection.objects.link(o); return o


def mat(name, col, rough=0.5, metal=0.0, **kw):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*col, 1); b.inputs['Roughness'].default_value = rough
    b.inputs['Metallic'].default_value = metal
    for k, v in kw.items(): b.inputs[k].default_value = v
    return m


def glowmat(name, col, rough=0.3, metal=0.0, gain=6.0, base=0.0):
    """object colour alpha drives emission (flash on hit)"""
    m = mat(name, col, rough, metal); b = m.node_tree.nodes['Principled BSDF']
    nn = m.node_tree.nodes; ll = m.node_tree.links
    oi = nn.new('ShaderNodeObjectInfo'); mul = nn.new('ShaderNodeMath'); mul.operation = 'MULTIPLY_ADD'
    mul.inputs[1].default_value = gain; mul.inputs[2].default_value = base
    ll.new(oi.outputs['Alpha'], mul.inputs[0]); ll.new(mul.outputs[0], b.inputs['Emission Strength'])
    ll.new(oi.outputs['Color'], b.inputs['Emission Color'])
    return m


def emis(name, col, strength=6.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Color'].default_value = (*col, 1); em.inputs['Strength'].default_value = strength
    out = nt.nodes.new('ShaderNodeOutputMaterial'); nt.links.new(em.outputs[0], out.inputs[0])
    return m


def key_series(obj, path, idx, vals, eps=1e-5, every=False):
    n = len(vals)
    for f in range(n):
        v = vals[f]
        if not every and 0 < f < n - 1 and abs(v - vals[f - 1]) < eps and abs(v - vals[f + 1]) < eps: continue
        getattr(obj, path)[idx] = float(v); obj.keyframe_insert(path, index=idx, frame=f)


def key_vec(obj, path, arr, every=False):
    for i in range(arr.shape[1]): key_series(obj, path, i, arr[:, i], every=every)


def key_alpha(obj, arr, col):
    obj.color = (*col, 0)
    n = len(arr)
    for f in range(n):
        v = arr[f]
        if 0 < f < n - 1 and abs(v - arr[f - 1]) < 1e-4 and abs(v - arr[f + 1]) < 1e-4: continue
        obj.color[3] = float(v); obj.keyframe_insert('color', index=3, frame=f)


def prim(kind, mt, **kw):
    getattr(bpy.ops.mesh, 'primitive_' + kind + '_add')(**kw)
    o = bpy.context.object
    if mt: o.data.materials.append(mt)
    return o


def box(mt, size, loc, rot=(0, 0, 0), bevel=0.0, parent=None):
    o = prim('cube', mt, size=1, location=loc, rotation=rot); o.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        m = o.modifiers.new('b', 'BEVEL'); m.width = bevel; m.segments = 3; m.limit_method = 'ANGLE'
    if parent: setp(o, parent)
    return o


def setp(o, parent):
    mw = o.matrix_world.copy(); o.parent = parent; o.matrix_world = mw
    return o


def rod(p0, p1, r, mt, verts=12):
    p0, p1 = Vector(p0), Vector(p1); d = p1 - p0
    o = prim('cylinder', mt, radius=r, depth=d.length, vertices=verts, location=(p0 + p1) / 2)
    o.rotation_mode = 'QUATERNION'; o.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d.normalized())
    return o


def smooth(o):
    for p in o.data.polygons: p.use_smooth = True
    return o


def empty(name, loc, parent=None):
    e = link(bpy.data.objects.new(name, None)); e.location = loc
    if parent: e.parent = parent
    e.rotation_mode = 'XYZ'
    return e


def ease(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


# ---------------- materials ----------------
WOOD = mat('wood', (0.55, 0.32, 0.16), 0.45)
WOODL = mat('woodl', (0.85, 0.62, 0.38), 0.4)
WOODD = mat('woodd', (0.25, 0.13, 0.06), 0.5)
GOLD = mat('gold', (1.0, 0.76, 0.35), 0.2, 1.0)
BRASS = mat('brass', (0.95, 0.68, 0.32), 0.25, 1.0)
IRON = mat('iron', (0.2, 0.2, 0.22), 0.35, 1.0)
RED = mat('red', (0.75, 0.05, 0.06), 0.45)
VELVET = mat('velvet', (0.45, 0.02, 0.05), 0.8, **{'Sheen Weight': 0.6, 'Sheen Tint': (1, 0.4, 0.4, 1)})
BLUE = mat('blue', (0.08, 0.22, 0.65), 0.45)
SKIN = mat('skin', (1.0, 0.78, 0.6), 0.45, **{'Coat Weight': 0.4})
BLACK = mat('black', (0.02, 0.02, 0.025), 0.15, **{'Coat Weight': 0.8})
WHITE = mat('white', (0.95, 0.93, 0.9), 0.4)
STRING = mat('string', (0.95, 0.92, 0.85), 0.6)
CHEEK = glowmat('cheek', (1.0, 0.35, 0.4), 0.5, gain=3.0)
YEL = mat('yel', (1.0, 0.75, 0.1), 0.35)
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs['Color'].default_value = (0.05, 0.03, 0.03, 1)
w.node_tree.nodes['Background'].inputs['Strength'].default_value = 1.0

# ---------------- theatre ----------------
SY = 0.25                       # puppet plane
FLOOR = 0.3
box(WOOD, (3.6, 2.2, 0.7), (0, 0.05, FLOOR - 0.35), bevel=0.01)                 # stage block
for i in range(12):                                                               # floor boards
    box(WOODL if i % 2 else mat(f'fb{i}', (0.78, 0.55, 0.32), 0.42), (3.6, 0.17, 0.02), (0, -0.98 + 0.18 * i + 0.09, FLOOR + 0.005))
box(GOLD, (3.6, 0.05, 0.06), (0, -1.06, FLOOR - 0.02))                            # apron trim
box(RED, (3.6, 0.02, 0.5), (0, -1.075, FLOOR - 0.3))                              # apron face
for i in range(9): prim('uv_sphere', GOLD, radius=0.03, location=(-1.6 + 0.4 * i, -1.09, FLOOR - 0.3), segments=12, ring_count=8)
box(mat('wall', (0.16, 0.06, 0.04), 0.7), (4.0, 0.1, 5.0), (0, 1.25, 2.3))          # back wall
for i in range(9):
    box(mat(f'pan{i}', (0.22, 0.1, 0.05), 0.5), (0.36, 0.04, 2.2), (-1.6 + 0.4 * i, 1.19, 1.5), bevel=0.02)
# wall stars
for i in range(14):
    x, z = rng.uniform(-1.5, 1.5), rng.uniform(1.0, 3.4)
    o = prim('ico_sphere', emis(f'ws{i}', (1.0, 0.8, 0.45), 2.0), radius=0.015, location=(x, 1.16, z), subdivisions=1)


def curtain(x0, side):
    """pleated velvet curtain"""
    bm = bmesh.new(); nx, nz = 40, 30; W, H = 0.75, 4.3
    vs = []
    for j in range(nz + 1):
        row = []
        for i in range(nx + 1):
            u = i / nx; z = H * j / nz
            gather = 1 - 0.25 * (1 - j / nz) ** 2
            x = x0 + side * u * W * gather
            y = 0.85 + 0.06 * math.sin(u * 2 * math.pi * 7)
            row.append(bm.verts.new((x, y, FLOOR + z - 0.02)))
        vs.append(row)
    for j in range(nz):
        for i in range(nx):
            bm.faces.new([vs[j][i], vs[j][i + 1], vs[j + 1][i + 1], vs[j + 1][i]] if side > 0 else [vs[j][i + 1], vs[j][i], vs[j + 1][i], vs[j + 1][i + 1]])
    me = bpy.data.meshes.new('cur'); bm.to_mesh(me); o = link(bpy.data.objects.new('cur', me)); o.data.materials.append(VELVET)
    smooth(o); m = o.modifiers.new('s', 'SOLIDIFY'); m.thickness = 0.02
    return o


curtain(-1.0, -1); curtain(1.0, 1)
# valance with string lights
for i in range(13):
    x = -1.8 + 0.3 * i
    o = prim('cylinder', VELVET, radius=0.17, depth=0.02, location=(x, 0.65, 4.3), rotation=(math.pi / 2, 0, 0), vertices=24)
box(VELVET, (3.8, 0.1, 0.5), (0, 0.7, 4.55))
box(GOLD, (3.8, 0.04, 0.04), (0, 0.63, 4.3))
BULBS = []
for i in range(15):
    x = -1.4 + 0.2 * i; z = 4.2 - 0.12 * math.sin(math.pi * (i % 5) / 4) ** 2
    hue = [0.0, 0.12, 0.33, 0.58, 0.8][i % 5]
    col = colorsys.hsv_to_rgb(hue, 0.6, 1.0)
    o = prim('uv_sphere', glowmat(f'bulb{i}', col, 0.2, gain=12, base=1.5), radius=0.035, location=(x, 0.55, z), segments=12, ring_count=8)
    BULBS.append((o, col, i))

# ---------------- marimba ----------------
MEL = E['mel']
PITCHES = sorted(set(p for _, p in MEL))
NBAR = len(PITCHES); MY = -1.32; MZ = 0.16
BW = 0.14; BX0 = -(NBAR - 1) * BW / 2
BARS = {}
box(WOODD, (NBAR * BW + 0.2, 0.05, 0.05), (0, MY - 0.17, MZ - 0.06))
box(WOODD, (NBAR * BW + 0.2, 0.05, 0.05), (0, MY + 0.17, MZ - 0.06))
for i, p in enumerate(PITCHES):
    x = BX0 + i * BW; L = 0.42 - 0.012 * i
    col = colorsys.hsv_to_rgb(i / NBAR * 0.85, 0.75, 1.0)
    o = box(glowmat(f'bar{i}', col, 0.35, gain=10), (BW * 0.86, L, 0.035), (x, MY, MZ), bevel=0.008)
    rod((x, MY, MZ - 0.05), (x, MY, MZ - 0.35 - 0.01 * (NBAR - i)), 0.025, GOLD)      # resonator tubes
    BARS[p] = (o, col, x)
# brass cannons (left/right) at the pit
CANN = []
for s in (-1, 1):
    base = empty(f'cb{s}', (s * 1.02, MY + 0.05, -0.12))
    prim('cylinder', WOODD, radius=0.09, depth=0.08, location=base.location, vertices=24)
    piv = empty(f'cp{s}', (0, 0, 0.05), base)
    piv.rotation_euler = (0, s * -0.55, 0)
    brl = prim('cylinder', BRASS, radius=0.045, depth=0.26, location=(0, 0, 0.12), vertices=24); brl.parent = piv
    rim = prim('torus', GOLD, major_radius=0.05, minor_radius=0.012, location=(0, 0, 0.25)); rim.parent = piv
    flash = prim('uv_sphere', glowmat(f'mf{s}', (1, 0.8, 0.4), gain=8), radius=0.03, location=(0, 0, 0.26)); flash.parent = piv
    CANN.append((base, piv, brl, flash, s))
bpy.context.view_layer.update()

# ---------------- puppet ----------------
PUP = empty('pup', (0, SY, FLOOR + 0.64))
TOR = empty('torso', (0, 0, 0), PUP)
o = prim('cylinder', RED, radius=0.2, depth=0.52, location=(0, SY, FLOOR + 0.64 + 0.3), vertices=32); smooth(o); setp(o, TOR)
m = o.modifiers.new('b', 'BEVEL'); m.width = 0.06; m.segments = 4
o = prim('cylinder', BLUE, radius=0.205, depth=0.12, location=(0, SY, FLOOR + 0.64 + 0.04), vertices=32); smooth(o); setp(o, TOR)
box(GOLD, (0.42, 0.03, 0.04), (0, SY - 0.19, FLOOR + 0.64 + 0.1), parent=TOR)
for i in range(3):
    b = prim('uv_sphere', GOLD, radius=0.025, location=(0, SY - 0.2, FLOOR + 0.64 + 0.22 + 0.11 * i), segments=12, ring_count=8); setp(b, TOR)
for s in (-1, 1):
    ep = prim('uv_sphere', GOLD, radius=0.07, location=(s * 0.23, SY, FLOOR + 0.64 + 0.55)); smooth(ep); setp(ep, TOR); ep.scale = (1, 1, 0.55)
NECK = empty('neck', (0, 0, 0.6), TOR)
HEAD = empty('head', (0, 0, 0.05), NECK)
bpy.context.view_layer.update()
hz = FLOOR + 0.64 + 0.6 + 0.05 + 0.24
o = prim('uv_sphere', SKIN, radius=0.24, location=(0, SY, hz), segments=48, ring_count=24); smooth(o); setp(o, HEAD)
o = prim('uv_sphere', SKIN, radius=0.05, location=(0, SY - 0.24, hz - 0.02)); smooth(o); setp(o, HEAD)              # nose
EYES = []; CHEEKS = []
for s in (-1, 1):
    e_ = prim('uv_sphere', BLACK, radius=0.035, location=(s * 0.085, SY - 0.215, hz + 0.05)); smooth(e_); setp(e_, HEAD)
    e_.scale = (1, 0.6, 1.25); EYES.append(e_)
    hl = prim('uv_sphere', emis(f'hl{s}', (1, 1, 1), 3), radius=0.009, location=(s * 0.085 + 0.012, SY - 0.24, hz + 0.065)); setp(hl, e_)
    ck = prim('uv_sphere', CHEEK, radius=0.05, location=(s * 0.14, SY - 0.17, hz - 0.05)); smooth(ck); setp(ck, HEAD); ck.scale = (1, 0.4, 0.75)
    ck.color = (1, 0.3, 0.35, 0.0); CHEEKS.append(ck)
mo = prim('torus', mat('mouth', (0.5, 0.03, 0.05), 0.3), major_radius=0.08, minor_radius=0.014, location=(0, SY - 0.205, hz - 0.08),
          rotation=(math.pi / 2 - 0.35, 0, 0), major_segments=32, minor_segments=8)
bm = bmesh.new(); bm.from_mesh(mo.data)
bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.y > -0.012], context='VERTS'); bm.to_mesh(mo.data); setp(mo, HEAD)
# hat: red cap with yellow pompom
hat = prim('cone', RED, radius1=0.2, radius2=0.03, depth=0.3, location=(0, SY, hz + 0.3), vertices=32); smooth(hat); setp(hat, HEAD)
hb = prim('torus', WHITE, major_radius=0.2, minor_radius=0.04, location=(0, SY, hz + 0.16)); smooth(hb); setp(hb, HEAD)
POM = prim('uv_sphere', YEL, radius=0.06, location=(0, SY, hz + 0.47)); smooth(POM); setp(POM, HEAD)
HANDS = {}; SHO = {}; ELB = {}; HIP = {}; FOOT = {}
for s in (-1, 1):
    sh = empty(f'sh{s}', (s * 0.27, 0, 0.53), TOR); SHO[s] = sh
    bpy.context.view_layer.update(); P0 = sh.matrix_world.translation.copy()
    o = prim('uv_sphere', WOODL, radius=0.055, location=P0); smooth(o); setp(o, sh)
    o = prim('cylinder', RED, radius=0.045, depth=0.26, location=P0 + Vector((0, 0, -0.15)), vertices=16); smooth(o); setp(o, sh)
    el = empty(f'el{s}', (0, 0, -0.3), sh); ELB[s] = el
    bpy.context.view_layer.update(); P1 = el.matrix_world.translation.copy()
    o = prim('uv_sphere', WOODL, radius=0.045, location=P1); smooth(o); setp(o, el)
    o = prim('cylinder', RED, radius=0.04, depth=0.2, location=P1 + Vector((0, 0, -0.12)), vertices=16); smooth(o); setp(o, el)
    o = prim('torus', WHITE, major_radius=0.042, minor_radius=0.014, location=P1 + Vector((0, 0, -0.22))); setp(o, el)
    hd = prim('uv_sphere', WHITE, radius=0.07, location=P1 + Vector((0, 0, -0.29))); smooth(hd); setp(hd, el)
    hd.scale = (0.55, 1.0, 1.05)                                       # flat mitten, palm faces inward
    th = prim('uv_sphere', WHITE, radius=0.03, location=P1 + Vector((0, -0.055, -0.26))); smooth(th); setp(th, el)
    HANDS[s] = empty(f'hand{s}', (0, 0, -0.32), el)
    hp = empty(f'hip{s}', (s * 0.1, 0, 0.0), PUP); HIP[s] = hp
    bpy.context.view_layer.update(); P2 = hp.matrix_world.translation.copy()
    o = prim('cylinder', BLUE, radius=0.065, depth=0.42, location=P2 + Vector((0, 0, -0.22)), vertices=16); smooth(o); setp(o, hp)
    o = prim('uv_sphere', WOODL, radius=0.06, location=P2 + Vector((0, 0, -0.43))); smooth(o); setp(o, hp)
    bt = prim('uv_sphere', BLACK, radius=0.09, location=P2 + Vector((0, -0.05, -0.56))); smooth(bt); setp(bt, hp); bt.scale = (0.85, 1.4, 0.6)
    FOOT[s] = empty(f'knee{s}', (0, 0, -0.3), hp)
HEADPT = empty('headpt', (0, 0, 0.62), HEAD)

# ---------------- cam barrel + marionette cross ----------------
BZ = 3.45; BY = SY - 0.05; BR = 0.15
TRK = 0.2; NTRK = 5
BAR_ROOT = empty('barrel_slide', (0, BY, BZ))
BAR_ROT = empty('barrel_rot', (0, 0, 0), BAR_ROOT)
bl = NTRK * TRK + 0.2
o = prim('cylinder', WOODL, radius=BR, depth=bl, location=(0, BY, BZ), rotation=(0, math.pi / 2, 0), vertices=48); smooth(o); setp(o, BAR_ROT)
for i in range(NTRK + 1):
    o = prim('cylinder', GOLD, radius=BR + 0.012, depth=0.015, location=((i - NTRK / 2) * TRK, BY, BZ), rotation=(0, math.pi / 2, 0), vertices=48); setp(o, BAR_ROT)
for s in (-1, 1):
    o = prim('cylinder', IRON, radius=0.025, depth=0.5, location=(s * 1.15, BY, BZ), rotation=(0, math.pi / 2, 0)); setp(o, BAR_ROOT)
    box(WOODD, (0.08, 0.08, 1.4), (s * 1.45, BY, BZ + 0.3))
box(WOODD, (3.0, 0.08, 0.08), (0, BY + 0.05, BZ + 1.0))
# gear on the barrel end + driving pinion
def gear(r, n, thick, mt, loc, rot):
    o = prim('cylinder', mt, radius=r, depth=thick, location=loc, rotation=rot, vertices=n * 2)
    bm_ = bmesh.new(); bm_.from_mesh(o.data)
    for v in bm_.verts:
        a = math.atan2(v.co.y, v.co.x); k = int(round(a / (math.pi / n)))
        if k % 2 == 0: v.co.x *= (r + 0.02) / r; v.co.y *= (r + 0.02) / r
    bm_.to_mesh(o.data); return o
g1 = gear(0.2, 20, 0.04, BRASS, (-(bl / 2 + 0.03), BY, BZ), (0, math.pi / 2, 0)); setp(g1, BAR_ROT)
CRANK = empty('crank', (-(bl / 2 + 0.03), BY, BZ + 0.33))
g2 = gear(0.11, 11, 0.04, BRASS, CRANK.location, (0, math.pi / 2, 0)); setp(g2, CRANK)
o = rod(CRANK.location, CRANK.location + Vector((-0.08, 0, 0.18)), 0.012, IRON); setp(o, CRANK)
o = prim('uv_sphere', RED, radius=0.035, location=CRANK.location + Vector((-0.08, 0, 0.2))); setp(o, CRANK)
CRANK.rotation_mode = 'XYZ'; BAR_ROT.rotation_mode = 'XYZ'

GEST = E['gest']; SECS = E['secs']
VSTART = [s['t'] for s in SECS]
ROT_PERIOD = E['verse_beats'] * BEAT
OMG = -2 * math.pi / ROT_PERIOD


def phi(t): return OMG * t
READ = math.pi          # pin points toward the camera (-y) when it trips the follower
for g in GEST:
    trk = min(g['v'], NTRK - 1)
    a = READ - phi(g['t'])            # barrel-local angle about +x
    x = (trk - (NTRK - 1) / 2) * TRK
    d = Vector((0, math.cos(a), math.sin(a)))
    c = Vector((x, BY, BZ)) + d * (BR + 0.03)
    pn = rod(Vector((x, BY, BZ)) + d * (BR - 0.01), c, 0.018, GOLD, verts=10); setp(pn, BAR_ROT)
    kn = prim('uv_sphere', mat(f'pk{g["v"]}', colorsys.hsv_to_rgb([0.0, 0.6, 0.33, 0.12, 0.85][g['v']], 0.8, 1), 0.3), radius=0.027, location=c,
              segments=12, ring_count=8)
    setp(kn, BAR_ROT)
# follower lever: pivot behind/below the barrel front, tip rides on the pins
FOL = empty('follower', (0, BY - 0.42, BZ - 0.02))
o = rod(FOL.location, FOL.location + Vector((0, 0.24, 0.0)), 0.016, IRON); setp(o, FOL)
o = prim('uv_sphere', BRASS, radius=0.03, location=FOL.location); setp(o, FOL)
box(WOODD, (0.05, 0.05, 0.3), (0.06, BY - 0.42, BZ - 0.17))
box(WOODD, (1.45, 0.05, 0.05), (0.06 + 0.72, BY - 0.42, BZ - 0.32))
# marionette cross hangs from the follower by a cord; strings to hands, knees, head
CROSS = empty('cross', (0, SY, 2.9))
o = box(WOOD, (0.7, 0.05, 0.05), (0, SY, 2.9), bevel=0.01); setp(o, CROSS)
o = box(WOOD, (0.05, 0.45, 0.05), (0, SY, 2.9), bevel=0.01); setp(o, CROSS)
CROSSPTS = {}
for nm, p in (('hl', (-0.35, 0, 0)), ('hr', (0.35, 0, 0)), ('kl', (-0.12, -0.22, 0)), ('kr', (0.12, -0.22, 0)), ('hd', (0, 0.22, 0)),
              ('top', (0, 0, 0.03))):
    CROSSPTS[nm] = empty('cp_' + nm, p, CROSS)
    if nm != 'top':
        o = prim('uv_sphere', GOLD, radius=0.018, location=Vector(p) + CROSS.location); setp(o, CROSS)
bpy.context.view_layer.update()


def string(a, b, r=0.004):
    cu = bpy.data.curves.new('str', 'CURVE'); cu.dimensions = '3D'; cu.bevel_depth = r; cu.bevel_resolution = 1
    sp = cu.splines.new('POLY'); sp.points.add(1)
    A = a.matrix_world.translation; B = b.matrix_world.translation
    sp.points[0].co = (*A, 1); sp.points[1].co = (*B, 1)
    o = link(bpy.data.objects.new('str', cu)); o.data.materials.append(STRING)
    for i, e_ in enumerate((a, b)):
        hk = o.modifiers.new(f'h{i}', 'HOOK'); hk.object = e_; hk.vertex_indices_set([i])
        hk.matrix_inverse = e_.matrix_world.inverted(); hk.center = e_.matrix_world.translation
    return o


TIP = empty('fol_tip', (0, 0.24, 0), FOL)
bpy.context.view_layer.update()
string(CROSSPTS['hl'], HANDS[-1]); string(CROSSPTS['hr'], HANDS[1])
string(CROSSPTS['kl'], FOOT[-1]); string(CROSSPTS['kr'], FOOT[1]); string(CROSSPTS['hd'], HEADPT)
string(TIP, CROSSPTS['top'], 0.006)

# ---------------- sign board ----------------
SIGN = empty('sign', (0, SY + 0.1, 4.0))
o = box(WOODL, (1.8, 0.05, 0.38), (0, SY + 0.12, 4.0), bevel=0.02); setp(o, SIGN)
o = box(GOLD, (1.86, 0.03, 0.42), (0, SY + 0.14, 4.0), bevel=0.01); setp(o, SIGN)
FONT = bpy.data.fonts.load(f'{HERE}/jp_sub.otf')
TXTS = ['しあわせなら手をたたこう', '♪ 手をたたこう', '♪ 足ならそう', '♪ 肩たたこう', '♪ みんなでぜんぶ！']
TOBJ = []
TXTM = mat('txt', (0.5, 0.05, 0.08), 0.4)
for i, s_ in enumerate(TXTS):
    cu = bpy.data.curves.new(f't{i}', 'FONT'); cu.body = s_; cu.font = FONT; cu.align_x = 'CENTER'; cu.align_y = 'CENTER'
    cu.size = 0.26 if i else 0.21; cu.extrude = 0.006
    o = link(bpy.data.objects.new(f't{i}', cu)); o.data.materials.append(TXTM)
    o.location = (0, -0.035, 0); o.rotation_euler = (math.pi / 2, 0, 0); o.parent = SIGN
    TOBJ.append(o)
bpy.context.view_layer.update()
for o in TOBJ:
    o.data.size *= min(1.55 / o.dimensions.x, 0.24 / o.dimensions.y)

# ---------------- lights & camera ----------------
def light(kind, name, loc, e, col, size=1.0, target=None, spot=None):
    ld = bpy.data.lights.new(name, kind); ld.energy = e; ld.color = col
    if kind == 'AREA': ld.size = size
    if kind == 'SPOT': ld.spot_size = spot; ld.spot_blend = 0.4; ld.shadow_soft_size = 0.15
    o = link(bpy.data.objects.new(name, ld)); o.location = loc
    if target is not None:
        tr = o.constraints.new('TRACK_TO'); tr.target = target; tr.track_axis = 'TRACK_NEGATIVE_Z'; tr.up_axis = 'UP_Y'
    return o


aim = empty('aim', (0, SY, 1.4))
light('AREA', 'key', (-2.2, -3.0, 3.6), 380, (1.0, 0.85, 0.7), 2.0, aim)
light('AREA', 'fill', (2.4, -2.6, 1.6), 120, (0.85, 0.85, 1.0), 2.5, aim)
SPOT = light('SPOT', 'spot', (0.4, -2.0, 5.2), 900, (1.0, 0.92, 0.75), target=aim, spot=0.5)
light('AREA', 'top', (0, -0.3, 4.9), 90, (1.0, 0.9, 0.8), 1.5, aim)
light('AREA', 'pit', (0, -2.4, -0.2), 70, (1.0, 0.8, 0.6), 1.5, empty('pitaim', (0, MY, MZ)))
camd = bpy.data.cameras.new('cam'); camd.lens = 35; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgo = empty('ctgt', (0, 0, 1.5))
tr = cam.constraints.new('TRACK_TO'); tr.target = tgo; tr.track_axis = 'TRACK_NEGATIVE_Z'; tr.up_axis = 'UP_Y'
camd.dof.use_dof = True; camd.dof.aperture_fstop = 5.6

# ================= animation =================
def pulse(t, tg, a=0.11, r=0.22):
    """0..1 shape peaking at tg (fast rise, softer release)"""
    x = np.zeros_like(t)
    m1 = (t > tg - a) & (t <= tg); x[m1] = ease((t[m1] - (tg - a)) / a)
    m2 = (t > tg) & (t < tg + r); x[m2] = 1 - ease((t[m2] - tg) / r)
    return x


def lift(t, tg, up=0.3, dn=0.09):
    """foot lift: rises slowly, slams down at tg"""
    x = np.zeros_like(t)
    m1 = (t > tg - up - dn) & (t <= tg - dn); x[m1] = ease((t[m1] - (tg - up - dn)) / up)
    m2 = (t > tg - dn) & (t <= tg); x[m2] = 1 - ((t[m2] - (tg - dn)) / dn) ** 2
    return x


T = TT
# which "mode" (pose family) is active: blend weights per verse
def win(t0, t1, fade=0.35):
    return ease((T - t0 + fade) / fade) * (1 - ease((T - t1) / fade))
VE = VSTART + [FINAL - 0.2]
W_CLAP = win(VE[0] + 0.2, VE[1] - 0.1) + win(VE[3], VE[4] + 0.15)
W_STOMP = win(VE[1], VE[2] - 0.1)
W_TAP = win(VE[2], VE[3] - 0.1)
W_FIN = ease((T - (FINAL - 0.12)) / 0.12) * (1 - ease((T - (FINAL + 2.6)) / 0.6))
clapP = np.zeros(NF); stompL = {-1: np.zeros(NF), 1: np.zeros(NF)}; tapP = {-1: np.zeros(NF), 1: np.zeros(NF)}; fol = np.zeros(NF)
cheek = np.zeros(NF)
for g in GEST:
    tg = g['t']; k = g['kind']
    fol = np.maximum(fol, pulse(T, tg, 0.12, 0.2))
    cheek = np.maximum(cheek, pulse(T, tg, 0.05, 0.5))
    side = -1 if g['n'] == 0 else 1
    if k in ('clap', 'all'): clapP = np.maximum(clapP, pulse(T, tg, 0.1, 0.18))
    if k in ('stomp', 'all'): stompL[side] = np.maximum(stompL[side], lift(T, tg))
    if k == 'tap': tapP[side] = np.maximum(tapP[side], pulse(T, tg, 0.1, 0.2))
# beat bounce
beat_ph = (T - VSTART[0]) / BEAT
active = (T > VSTART[0] - 0.5) & (T < FINAL + 0.5)
bob = np.where(active, np.abs(np.sin(np.pi * beat_ph)), 0)
sway = np.where(active, np.sin(np.pi * beat_ph / 2), 0)
# wake-up during intro: head lifts from slump
wake = ease((T - 1.4) / 1.2)
stomp_dip = sum(np.maximum(0, pulse(T, g['t'], 0.03, 0.18)) for g in GEST if g['kind'] in ('stomp', 'all', 'final'))
jump = np.zeros(NF)
m = (T > FINAL - 0.35) & (T < FINAL + 0.45)
tj = (T[m] - (FINAL - 0.35)) / 0.8; jump[m] = 0.28 * np.sin(np.pi * tj) * (tj < 1)

pz = FLOOR + 0.64 + 0.018 * bob * (1 - W_FIN) - 0.04 * np.clip(stomp_dip, 0, 1) + jump
key_series(PUP, 'location', 2, pz)
PUP.rotation_mode = 'XYZ'
key_series(PUP, 'rotation_euler', 1, 0.05 * sway * (1 - W_FIN))
key_series(TOR, 'rotation_euler', 0, 0.05 * np.clip(stomp_dip, 0, 1) - 0.06 * (1 - wake))
key_series(NECK, 'rotation_euler', 0, -0.45 * (1 - wake) + 0.06 * bob - 0.1 * clapP)
key_series(NECK, 'rotation_euler', 1, -0.12 * sway)
key_series(NECK, 'rotation_euler', 2, 0.15 * (tapP[1] - tapP[-1]))
# blink: eyes squash at a few times + happy squint on every gesture
blink = np.zeros(NF)
for tb in list(np.arange(2.6, FINAL, 3.7)) + [FINAL]:
    blink = np.maximum(blink, pulse(T, tb, 0.06, 0.08))
for e_ in EYES: key_series(e_, 'scale', 2, 1.25 * (1 - 0.85 * np.maximum(blink, 0.6 * W_FIN)))
for ck in CHEEKS: key_alpha(ck, np.clip(cheek * 0.9 + 0.15 * W_FIN, 0, 1), (1, 0.3, 0.35))

# arms: base poses blended
for s in (-1, 1):
    # rest: hang, slightly out
    shx = np.zeros(NF); shy = -s * 0.12 * np.ones(NF); shz = np.zeros(NF); elx = -0.15 * np.ones(NF)
    # clap ready -> shut
    o_ = 1 - clapP
    cx = -1.1; cz = -s * (0.46 * (1 - o_) - 0.18 * o_); cex = -0.55
    # stomp: hands on hips (elbows out)
    swing = stompL[-s] - stompL[s]
    stx, sty, stz, stex = -0.75 * swing - 0.1, -s * 0.15, 0.0, -0.5 - 0.3 * np.abs(swing)
    # tap: hand to own shoulder, tapping
    tp = tapP[s]
    tx, ty, tz, tex = -0.45, -s * 0.15, -s * 0.15, -2.25 - 0.45 * tp
    # finale: arms up in a V
    fx, fy, fz, fex = -0.15, -s * 2.5, 0.0, -0.1
    wr = np.clip(1 - W_CLAP - W_STOMP - W_TAP - W_FIN, 0, 1)
    SX = wr * shx + W_CLAP * cx + W_STOMP * stx + W_TAP * tx + W_FIN * fx
    SY_ = wr * shy + W_CLAP * 0 + W_STOMP * sty + W_TAP * ty + W_FIN * fy
    SZ = wr * shz + W_CLAP * cz + W_STOMP * stz + W_TAP * tz + W_FIN * fz
    EX = wr * elx + W_CLAP * cex + W_STOMP * stex + W_TAP * tex + W_FIN * fex
    SX = SX + 0.04 * bob * (wr + W_STOMP)
    for i, arr in enumerate((SX, SY_, SZ)): key_series(SHO[s], 'rotation_euler', i, arr)
    key_series(ELB[s], 'rotation_euler', 0, EX)
    key_series(HIP[s], 'rotation_euler', 0, -0.85 * stompL[s] - 0.25 * W_FIN * (s > 0) * 0)
    key_series(HIP[s], 'rotation_euler', 1, -s * 0.12 * W_FIN)

# barrel: rotation + slides one track per verse; crank turns
trk = np.zeros(NF)
for i, ts in enumerate(VSTART + [FINAL - 0.25]):
    trk += ease((T - (ts - 0.55)) / 0.25) * (1 if i > 0 else 0)
key_series(BAR_ROOT, 'location', 0, -((trk) - (NTRK - 1) / 2) * TRK)
run = ease((T - 0.2) / 1.0)
ang = np.cumsum(run) / FPS * OMG
ang = ang - np.interp(E['gest'][0]['t'], T, ang) + phi(E['gest'][0]['t'])     # exact phase at gesture times
key_series(BAR_ROT, 'rotation_euler', 0, ang)
key_series(CRANK, 'rotation_euler', 0, -ang * 0.2 / 0.11)
key_series(FOL, 'rotation_euler', 0, 0.28 * fol)
# the cross follows the follower tip + tilts with each gesture kind
cz_ = 2.9 + 0.07 * fol - 0.04 * (1 - wake)
key_series(CROSS, 'location', 2, cz_)
CROSS.rotation_mode = 'XYZ'
key_series(CROSS, 'rotation_euler', 1, 0.25 * (stompL[1] - stompL[-1]) + 0.2 * (tapP[1] - tapP[-1]) + 0.04 * sway)
key_series(CROSS, 'rotation_euler', 0, -0.25 * clapP)

# sign flips at each verse
for i, o in enumerate(TOBJ):
    t_on = 0.0 if i == 0 else VSTART[i - 1] - 0.55
    t_off = VSTART[i] - 0.55 if i < len(VSTART) else 1e9
    vis = ((T >= t_on) & (T < t_off)).astype(float)
    key_series(o, 'scale', 0, vis); key_series(o, 'scale', 1, vis); key_series(o, 'scale', 2, vis)
flip = np.zeros(NF)
for ts in VSTART: flip += np.where(np.abs(T - (ts - 0.55)) < 0.25, np.cos(np.pi * (T - (ts - 0.55)) / 0.5), 0) * 0 + \
        np.where((T > ts - 0.8) & (T < ts - 0.3), np.sin(np.pi * (T - (ts - 0.8)) / 0.5) * 1.0, 0)
SIGN.rotation_mode = 'XYZ'; key_series(SIGN, 'rotation_euler', 0, 1.4 * flip)

# string lights: chase on the beat
for o, col, i in BULBS:
    b_ = np.floor(np.maximum(0, beat_ph)).astype(int)
    on = ((b_ + i) % 3 == 0).astype(float) * active + 0.15
    on = np.maximum(on, 1.0 * (T > FINAL - 0.05))
    key_alpha(o, on, col)

# ---------------- marbles on the marimba ----------------
GV = np.array([0, 0, -G])
NPOOL = 18
balls = []
BMS = {}
for i in range(NPOOL):
    o = prim('uv_sphere', None, radius=0.052, location=(0, 0, -3), segments=24, ring_count=12); smooth(o)
    balls.append(o)
pos = [np.tile(np.array([0, 0, -3.0]), (NF, 1)) for _ in range(NPOOL)]
mcol = [None] * NPOOL
barflash = {p: np.zeros(NF) for p in PITCHES}
bardip = {p: np.zeros(NF) for p in PITCHES}
canrec = {-1: np.zeros(NF), 1: np.zeros(NF)}
busy = np.zeros(NPOOL)
bpy.context.view_layer.update()
mouth = {}
for base, piv, brl, flash, s in CANN:
    mouth[s] = np.array(flash.matrix_world.translation)
PCOL = {}
for k, (t, p) in enumerate(MEL):
    bo, col, bx = BARS[p]
    s = 1 if bx < 0 else -1                                   # fire from the opposite side: crossing arcs
    t0 = t - TAU
    j = int(np.argmin(busy)); busy[j] = t + 0.9
    p0 = mouth[s]; p1 = np.array([bx + rng.normal(0, 0.01), MY + rng.normal(0, 0.02), MZ + 0.0175 + 0.035])
    v0 = (p1 - p0 - 0.5 * GV * TAU ** 2) / TAU
    f0, f1 = int(math.ceil(t0 * FPS)), int(t * FPS)
    for f in range(max(0, f0), min(NF, f1 + 1)):
        tau = f / FPS - t0; pos[j][f] = p0 + v0 * tau + 0.5 * GV * tau * tau
    vb = np.array([rng.normal(0, 0.25) - 0.15 * s, -0.9, 1.6])
    for f in range(f1 + 1, min(NF, int((t + 0.85) * FPS))):
        tau = f / FPS - t; pp = p1 + vb * tau + 0.5 * GV * tau * tau
        pos[j][f] = pp if pp[2] > -0.9 else np.array([0, 0, -3.0])
    if f1 + 1 < NF: pos[j][f1 + 1:][pos[j][f1 + 1:, 2] < -0.89] = np.array([0, 0, -3.0])
    barflash[p] = np.maximum(barflash[p], np.exp(-np.clip(T - t, 0, None) / 0.15) * (T >= t - 0.5 / FPS))
    bardip[p] = np.maximum(bardip[p], pulse(T, t, 0.02, 0.15))
    canrec[s] = np.maximum(canrec[s], pulse(T, t0, 0.01, 0.2))
    mcol[j] = mcol[j] or []; mcol[j].append((t0, tuple(c ** 2.2 for c in col)))
# each ball keeps one colour per flight: colour via object colour keyed at launch (constant)
BALLM = mat('ballm', (1, 1, 1), 0.08, **{'Coat Weight': 1.0})
nn = BALLM.node_tree.nodes; oi = nn.new('ShaderNodeObjectInfo'); BALLM.node_tree.links.new(oi.outputs['Color'], nn['Principled BSDF'].inputs['Base Color'])
for j, o in enumerate(balls):
    o.data.materials.append(BALLM)
    key_vec(o, 'location', pos[j])
    for (t0, col) in (mcol[j] or []):
        f = max(0, int(t0 * FPS) - 1); o.color = (*col, 1)
        o.keyframe_insert('color', frame=f)
    ad = o.animation_data
    from bpy_extras import anim_utils
    cb = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
    for fc in cb.fcurves:
        if fc.data_path == 'color':
            for kp in fc.keyframe_points: kp.interpolation = 'CONSTANT'
for p in PITCHES:
    bo, col, bx = BARS[p]
    key_alpha(bo, np.clip(barflash[p], 0, 1), col)
    key_series(bo, 'location', 2, MZ - 0.012 * bardip[p])
for base, piv, brl, flash, s in CANN:
    key_series(brl, 'location', 2, 0.12 - 0.05 * canrec[s])
    key_alpha(flash, np.clip(canrec[s] * 1.0, 0, 1), (1, 0.75, 0.35))

# ---------------- effects: stars (clap), rings (stomp), notes (tap), confetti (final) ----------------
def star_mesh(r1=0.05, r2=0.022, th=0.012):
    bm_ = bmesh.new(); top = []; bot = []
    for i in range(10):
        a = math.pi / 2 + i * math.pi / 5; r = r1 if i % 2 == 0 else r2
        top.append(bm_.verts.new((r * math.cos(a), th, r * math.sin(a)))); bot.append(bm_.verts.new((r * math.cos(a), -th, r * math.sin(a))))
    ct = bm_.verts.new((0, th * 2, 0)); cb_ = bm_.verts.new((0, -th * 2, 0))
    for i in range(10):
        j = (i + 1) % 10
        bm_.faces.new([ct, top[j], top[i]]); bm_.faces.new([cb_, bot[i], bot[j]]); bm_.faces.new([top[i], top[j], bot[j], bot[i]])
    me = bpy.data.meshes.new('star'); bm_.to_mesh(me); return me


STARME = star_mesh()
FXS = []
def pool(n, make):
    return [make(i) for i in range(n)]


def mk_star(i):
    o = link(bpy.data.objects.new(f'star{i}', STARME))
    o.data.materials.append(emis('stm', (1.0, 0.82, 0.25), 9.0) if i == 0 else bpy.data.materials['stm'])
    return o


STARS = pool(36, mk_star)
RINGM = emis('ringm', (1.0, 0.85, 0.55), 4.0)
def mk_ring(i):
    o = prim('torus', RINGM, major_radius=0.1, minor_radius=0.012, location=(0, 0, -3), major_segments=32, minor_segments=6)
    return o
RINGS = pool(10, mk_ring)
NOTEM = emis('notem', (0.6, 0.9, 1.0), 5.0)
def mk_note(i):
    cu = bpy.data.curves.new(f'n{i}', 'FONT'); cu.body = '♪'; cu.font = FONT; cu.size = 0.16; cu.extrude = 0.01
    cu.align_x = 'CENTER'; cu.align_y = 'CENTER'
    o = link(bpy.data.objects.new(f'n{i}', cu)); o.data.materials.append(NOTEM); o.rotation_euler = (math.pi / 2, 0, 0)
    return o
NOTES_FX = pool(10, mk_note)

hand_y = SY - 0.45; clap_pt = np.array([0, SY - 0.48, FLOOR + 0.64 + 0.53 + 0.02])
def fx_pool(objs, events, life, fn):
    """events: list of (t, data); fn(age, data, k) -> (loc, scale, rotz) for one object; objects assigned round-robin"""
    n = len(objs); L = [np.tile([0, 0, -3.0], (NF, 1)) for _ in objs]; S = [np.zeros(NF) for _ in objs]; R = [np.zeros(NF) for _ in objs]
    c = 0
    for (t, dat, k) in events:
        j = c % n; c += 1
        f0, f1 = max(0, int(t * FPS)), min(NF, int((t + life) * FPS))
        for f in range(f0, f1):
            age = f / FPS - t
            if age < 0: continue
            lo, scl, rz = fn(age, dat, k); L[j][f] = lo; S[j][f] = scl; R[j][f] = rz
    for j, o in enumerate(objs):
        key_vec(o, 'location', L[j])
        for i in range(3): key_series(o, 'scale', i, S[j])
        key_series(o, 'rotation_euler', 1, R[j])


ev = []
for g in GEST:
    if g['kind'] in ('clap', 'all', 'final'):
        for k in range(6 if g['kind'] != 'final' else 12):
            ang_ = 2 * math.pi * k / (6 if g['kind'] != 'final' else 12) + rng.uniform(-0.3, 0.3)
            ev.append((g['t'], (ang_, rng.uniform(0.9, 1.4)), k))
ev.sort(key=lambda e_: e_[0])
fx_pool(STARS, ev, 0.55, lambda a, d, k: (clap_pt + np.array([math.cos(d[0]), -0.15, math.sin(d[0])]) * d[1] * 0.6 * (1 - math.exp(-a / 0.12)),
                                           max(0.0, 1.3 * (1 - a / 0.55)) * min(1, a / 0.04), a * 6 + d[0]))
ev = []
for g in GEST:
    if g['kind'] in ('stomp', 'all', 'final'):
        side = -1 if g['n'] == 0 else 1
        for sd in ((side,) if g['kind'] != 'final' else (-1, 1)):
            ev.append((g['t'], sd, 0))
fx_pool(RINGS, ev, 0.45, lambda a, d, k: (np.array([d * 0.1, SY - 0.06, FLOOR + 0.015]), 0.6 + 3.5 * a, 0.0))
for o in RINGS: o.rotation_euler = (0, 0, 0)
ev = [(g['t'], -1 if g['n'] == 0 else 1, 0) for g in GEST if g['kind'] == 'tap']
fx_pool(NOTES_FX, ev, 0.9, lambda a, d, k: (np.array([d * (0.3 + 0.25 * a), SY - 0.3, FLOOR + 0.64 + 0.62 + 0.55 * a]),
                                             max(0, min(1, a / 0.08)) * (1 - ease((a - 0.6) / 0.3)), d * 0.3 * math.sin(a * 8)))
for o in NOTES_FX: o.rotation_mode = 'XYZ'; o.rotation_euler[0] = math.pi / 2

# confetti from both curtains at the finale
CONF = []
for i in range(140):
    col = colorsys.hsv_to_rgb(rng.uniform(0, 1), 0.8, 1.0)
    o = box(mat(f'cf{i}', col, 0.4, **{'Emission Color': (*col, 1), 'Emission Strength': 0.6}), (0.035, 0.002, 0.022), (0, 0, -3))
    o.rotation_mode = 'XYZ'; CONF.append(o)
    s = -1 if i % 2 else 1
    p0 = np.array([s * 1.05, -0.3, 0.6]); v = np.array([-s * rng.uniform(0.8, 2.2), rng.uniform(-0.9, 0.2), rng.uniform(4.0, 6.2)])
    tl = FINAL + rng.uniform(-0.02, 0.15)
    L = np.tile([0, 0, -3.0], (NF, 1)); RX = np.zeros(NF); RZ = np.zeros(NF)
    ph0 = rng.uniform(0, 6.28); w0 = rng.uniform(5, 12)
    for f in range(int(tl * FPS), NF):
        a = f / FPS - tl
        if a < 0: continue
        # drag: velocity decays to a slow flutter fall
        k = 2.2; vt = 0.35
        x = p0 + (v / k) * (1 - math.exp(-k * a)) + np.array([0.06 * math.sin(w0 * a * 0.5 + ph0), 0, 0])
        x[2] = p0[2] + (v[2] + vt / 1) / k * (1 - math.exp(-k * a)) - vt * a * 1.0 - 0.0
        if x[2] < FLOOR + 0.01 and abs(x[0]) < 1.6: x[2] = FLOOR + 0.01
        L[f] = x; RX[f] = ph0 + w0 * a; RZ[f] = ph0 * 0.7 + w0 * 0.6 * a
    key_vec(o, 'location', L); key_series(o, 'rotation_euler', 0, RX); key_series(o, 'rotation_euler', 2, RZ)
# confetti cannons sit at the curtain edges
for s in (-1, 1):
    o = prim('cylinder', GOLD, radius=0.06, depth=0.22, location=(s * 1.05, -0.3, 0.5), rotation=(0, -s * 0.35, 0), vertices=20)
    prim('cylinder', WOODD, radius=0.07, depth=0.1, location=(s * 1.05, -0.3, FLOOR + 0.05), vertices=20)
    prim('uv_sphere', RED, radius=0.05, location=(s * 1.09, -0.3, 0.39))

# spotlight swells at the finale
SPOT.data.energy = 900
for f, e_ in ((0, 150), (int(1.6 * FPS), 450), (int(FINAL * FPS) - 2, 450), (int(FINAL * FPS) + 2, 800)):
    SPOT.data.energy = e_; SPOT.data.keyframe_insert('energy', frame=f)

# ---------------- camera plan ----------------
WIDE = ((0.0, -6.4, 1.9), (0, SY, 1.85), 30)
MED = ((0.0, -5.7, 1.15), (0, SY - 0.4, 1.3), 30)
LOWC = ((0.35, -3.7, 0.95), (0, MY + 0.6, 0.75), 30)
TOPC = ((0.0, -4.1, 3.2), (0, SY, 3.4), 28)
PUPC = ((-0.55, -5.4, 1.0), (0, SY - 0.4, 1.25), 30)
shots = [(0.0, WIDE, (0.0, -5.6, 1.95))]
for v, ts in enumerate(VSTART):
    shots.append((ts - 1.1, TOPC, (0.0, -3.85, 3.2)))          # barrel shifts + sign flips
    shots.append((ts + 0.6, MED if v % 2 == 0 else PUPC, None))
    l3 = ts + (0.5 + 2 * E['line']) * BEAT                         # line 3: melody close-up on the marimba
    shots.append((l3, LOWC, (-0.35, -3.5, 0.9)))
    shots.append((l3 + 4 * BEAT + 0.3, PUPC if v % 2 == 0 else MED, None))
shots.append((FINAL - 2.0, WIDE, (0.0, -6.6, 2.05)))
shots.sort(key=lambda s_: s_[0])
CL = np.zeros((NF, 3)); CT = np.zeros((NF, 3)); LN = np.zeros(NF)
for i, (t0, (loc, tg, lens), loc2) in enumerate(shots):
    t1 = shots[i + 1][0] if i + 1 < len(shots) else T_END + 1
    m = (T >= t0) & (T < t1)
    u = ((T[m] - t0) / max(0.1, t1 - t0))[:, None]
    a, b = np.array(loc), np.array(loc2 if loc2 else (loc[0] + 0.12, loc[1] + 0.25, loc[2]))
    CL[m] = a + (b - a) * u; CT[m] = np.array(tg); LN[m] = lens
for f in range(NF):
    cam.location = CL[f]; cam.keyframe_insert('location', frame=f)
    tgo.location = CT[f]; tgo.keyframe_insert('location', frame=f)
    camd.lens = LN[f]; camd.keyframe_insert('lens', frame=f)
    camd.dof.focus_distance = float(np.linalg.norm(CT[f] - CL[f])); camd.keyframe_insert('dof.focus_distance', frame=f)
from bpy_extras import anim_utils
for idb in (cam, tgo, camd):
    ad = idb.animation_data
    cb = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
    for fc in cb.fcurves:
        for kp in fc.keyframe_points: kp.interpolation = 'CONSTANT'

print('objects', len(bpy.data.objects), 'NF', NF)
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
