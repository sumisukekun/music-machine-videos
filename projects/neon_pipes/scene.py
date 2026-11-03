"""Neon Pipes - a modern take on the classic pipe-launcher music machine.
Chrome launcher tubes fire glowing balls in exact ballistic arcs: plucked neon strings (melody), a radial
"sun" xylophone (counter-melody / chords), giant bass strings, and a drum kit. Balls bounce into catcher cups."""
import os, sys, json, math, colorsys, time
import bpy, bmesh
import numpy as np
from mathutils import Vector

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json'))
FPS = 30; T_END = 66.0; NF = int(T_END * FPS); FINAL = E['final']; BEAT = E['beat']; TB0 = E['t0']
G = 9.8; TAU = 0.8
rng = np.random.default_rng(7)
bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
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
sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.45
try: sc.render.motion_blur_position = 'START'   # shutter opens on the frame: hard camera cuts stay clean
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


def glowmat(name, col, rough=0.3, metal=0.0, gain=6.0):
    """object colour alpha drives emission (flash on hit)"""
    m = mat(name, col, rough, metal); b = m.node_tree.nodes['Principled BSDF']
    nn = m.node_tree.nodes; ll = m.node_tree.links
    oi = nn.new('ShaderNodeObjectInfo'); mul = nn.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = gain
    ll.new(oi.outputs['Alpha'], mul.inputs[0]); ll.new(mul.outputs[0], b.inputs['Emission Strength'])
    ll.new(oi.outputs['Color'], b.inputs['Emission Color'])
    return m


def ss(x): x = min(max(x, 0), 1); return x * x * (3 - 2 * x)


def key_series(obj, path, idx, vals, eps=1e-5):
    """keyframe an array of per-frame values, skipping frames where nothing changes"""
    n = len(vals)
    for f in range(n):
        v = vals[f]
        if 0 < f < n - 1 and abs(v - vals[f - 1]) < eps and abs(v - vals[f + 1]) < eps: continue
        getattr(obj, path)[idx] = float(v); obj.keyframe_insert(path, index=idx, frame=f)


def prim(kind, mt, **kw):
    getattr(bpy.ops.mesh, 'primitive_' + kind + '_add')(**kw)
    o = bpy.context.object
    if mt: o.data.materials.append(mt)
    return o




def rod(p0, p1, r, mt, verts=12):
    p0, p1 = Vector(p0), Vector(p1); d = p1 - p0
    o = prim('cylinder', mt, radius=r, depth=d.length, vertices=verts, location=(p0 + p1) / 2)
    o.rotation_mode = 'QUATERNION'; o.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d.normalized())
    return o


def neon(name, col, strength=12.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Color'].default_value = (*col, 1); em.inputs['Strength'].default_value = strength
    out = nt.nodes.new('ShaderNodeOutputMaterial'); nt.links.new(em.outputs[0], out.inputs[0])
    return m


NOTES = json.load(open(f'{HERE}/notes.json'))
S = dict((n, t) for t, n in E['sections'])
GV = Vector((0, 0, -G))
BR = 0.075

# ---------------- materials ----------------
CHROME = mat('chrome', (0.9, 0.92, 0.95), 0.08, 1.0)
SATIN = mat('satin', (0.7, 0.72, 0.78), 0.28, 1.0)
DARKM = mat('darkmetal', (0.03, 0.03, 0.04), 0.3, 0.8)
BLACKG = mat('blackgloss', (0.008, 0.008, 0.012), 0.12, 0.0, **{'Coat Weight': 1.0})
FLOORM = mat('floor', (0.012, 0.012, 0.02), 0.32, 0.3)
WALLM = mat('wall', (0.018, 0.014, 0.03), 0.6, 0.2)
CYAN = (0.05, 0.85, 1.0); MAG = (1.0, 0.12, 0.75); ORANGE = (1.0, 0.45, 0.06); LIME = (0.55, 1.0, 0.15); VIOLET = (0.45, 0.2, 1.0)
NE_C = neon('neon_c', CYAN, 10); NE_M = neon('neon_m', MAG, 10); NE_V = neon('neon_v', VIOLET, 8); NE_O = neon('neon_o', ORANGE, 10)

# ---------------- stage ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bgn = w.node_tree.nodes['Background']; bgn.inputs['Color'].default_value = (0.004, 0.003, 0.012, 1); bgn.inputs['Strength'].default_value = 1
prim('plane', FLOORM, size=60, location=(0, 2, 0))
for i in range(-12, 13):          # floor grid of faint neon lines
    o = prim('cube', NE_V, size=1, location=(i * 1.0, 2, 0.002)); o.scale = (0.012, 30, 0.002)
for j in range(-12, 18):
    o = prim('cube', NE_V, size=1, location=(0, j * 1.0, 0.002)); o.scale = (24, 0.012, 0.002)
NE_V.node_tree.nodes['Emission'].inputs['Strength'].default_value = 1.2
o = prim('plane', WALLM, size=40, location=(0, 8.0, 10), rotation=(math.pi / 2, 0, 0))
for sx in (-1, 1):
    o = prim('plane', WALLM, size=40, location=(sx * 6.5, 2, 10), rotation=(0, math.pi / 2, 0))
# wall neon strips and decorative pipes with glowing collars
for k, x in enumerate(np.linspace(-5.5, 5.5, 9)):
    o = prim('cube', NE_C if k % 2 == 0 else NE_M, size=1, location=(x, 7.95, 4.5)); o.scale = (0.05, 0.02, 9)
for z, mt in ((1.2, CHROME), (6.4, SATIN), (7.2, CHROME)):
    rod((-6.4, 7.7, z), (6.4, 7.7, z), 0.12, mt, 24)
for x in (-4.8, -3.6, 3.6, 4.8):
    rod((x, 7.6, 0), (x, 7.6, 9), 0.1, CHROME, 20)
    for z in (1.0, 2.6, 4.2, 5.8):
        prim('torus', NE_C if x < 0 else NE_M, major_radius=0.13, minor_radius=0.025, location=(x, 7.6, z))
for sx in (-1, 1):
    for z in (0.8, 3.0):
        rod((sx * 6.3, -6, z), (sx * 6.3, 7.8, z), 0.11, CHROME, 20)
    o = prim('cube', NE_M if sx < 0 else NE_C, size=1, location=(sx * 6.45, 1, 5.2)); o.scale = (0.02, 14, 0.05)

def area(name, loc, aim, col, e, size):
    ld = bpy.data.lights.new(name, 'AREA'); ld.energy = e; ld.color = col; ld.size = size
    o = link(bpy.data.objects.new(name, ld)); o.location = loc
    tr = o.constraints.new('TRACK_TO'); t = link(bpy.data.objects.new(name + 't', None)); t.location = aim; tr.target = t
    tr.track_axis = 'TRACK_NEGATIVE_Z'; tr.up_axis = 'UP_Y'
area('key', (0, -4, 9), (0, 1.5, 0), (0.85, 0.88, 1.0), 2600, 5)
area('magL', (-6, -2, 4), (0, 1.5, 1), (1.0, 0.25, 0.8), 1500, 4)
area('cyanR', (6, -1, 4), (0, 1.5, 1), (0.2, 0.8, 1.0), 1500, 4)
area('rim', (0, 9, 6), (0, 0, 1), (0.6, 0.4, 1.0), 1200, 6)

# ---------------- helpers ----------------
INST = {}     # key -> list of (obj, kind) responders
TARGET = {}   # key -> (hit point, launch point, catcher point, flight time, ball colour)
LAUNCH = {}   # key -> launcher record (ring obj, tube obj)

def tube(base, top, r, ring_mt):
    """chrome launcher tube from base to top with a neon collar at the mouth"""
    t = rod(base, top, r, CHROME, 24)
    d = (Vector(top) - Vector(base)).normalized()
    ring = prim('torus', glowmat('ring', (0.6, 0.6, 0.65), 0.2, 1.0, 14), major_radius=r * 1.05, minor_radius=r * 0.28, location=top)
    ring.rotation_mode = 'QUATERNION'; ring.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d)
    return t, ring

def cup(p, r=0.13, mt=None):
    c = prim('cylinder', mt or SATIN, radius=r, depth=0.28, vertices=20, location=(p[0], p[1], p[2] - 0.14))
    prim('cylinder', DARKM, radius=r * 0.82, depth=0.01, vertices=20, location=(p[0], p[1], p[2] + 0.002))
    prim('torus', CHROME, major_radius=r, minor_radius=0.018, location=p)
    rod((p[0], p[1], 0), (p[0], p[1], p[2] - 0.28), 0.03, SATIN, 8)
    return c

def hue(i, n, h0, h1, s=0.85):
    return colorsys.hsv_to_rgb((h0 + (h1 - h0) * i / max(1, n - 1)) % 1.0, s, 1.0)

def pset(*roles): return sorted({e[3] for e in NOTES if e[2] in roles})

# ---------------- melody strings (plucked neon strings) ----------------
STR_P = pset('mel', 'arp', 'final_str'); n = len(STR_P)
YS, YL, YT = 4.2, 2.85, 3.8
frame_mt = BLACKG
rod((-3.15, YS, 0.25), (3.15, YS, 0.25), 0.09, CHROME, 20)
for i, p in enumerate(STR_P):
    x = -2.8 + 5.6 * i / (n - 1); top = 4.0 - 1.3 * i / (n - 1); col = hue(i, n, 0.5, 0.58)
    s = rod((x, YS, 0.3), (x, YS, top), 0.014, glowmat(f's{p}', (0.8, 0.82, 0.9), 0.2, 1.0, 9), 8); s.color = (*col, 0.15)
    prim('cylinder', CHROME, radius=0.035, depth=0.12, vertices=12, location=(x, YS, top + 0.04))   # tuning peg
    zh = 1.55 + 0.25 * math.sin(i * 1.7)
    hitp = Vector((x, YS - BR - 0.016, zh))
    l0 = Vector((x, YL, 0.0)); l1 = Vector((x, YL + 0.08, 0.55))
    t, ring = tube(l0, l1, 0.06, None)
    catch = Vector((x, YT, 0.42))
    TARGET[('str', p)] = (hitp, l1 + (l1 - l0).normalized() * 0.03, catch, 0.55, col)
    INST[('str', p)] = [(s, 'string')]; LAUNCH[('str', p)] = (t, ring)
# string frame (top beam follows the peg line) + catcher trough
rod((-3.0, YS, 4.05), (3.0, YS, 2.75), 0.06, CHROME, 16)
for x in (-3.15, 3.15): rod((x, YS, 0), (x, YS, 4.2 if x < 0 else 2.9), 0.07, CHROME, 16)
o = prim('cube', SATIN, size=1, location=(0, YT, 0.3)); o.scale = (6.2, 0.26, 0.04)
for sy in (-1, 1):
    o = prim('cube', CHROME, size=1, location=(0, YT + sy * 0.13, 0.38)); o.scale = (6.2, 0.02, 0.16)
o = prim('cube', NE_C, size=1, location=(0, YT - 0.14, 0.46)); o.scale = (6.2, 0.01, 0.015)
rod((-3.0, YL, 0.05), (3.0, YL, 0.05), 0.08, CHROME, 16)   # manifold feeding the tubes

# ---------------- bass strings (big, behind) ----------------
BAS_P = pset('bass', 'final_bass'); n = len(BAS_P)
YB, YBL, YBT = 6.2, 5.35, 5.8
rod((-3.0, YB, 5.6), (3.0, YB, 5.6), 0.1, CHROME, 20); rod((-3.0, YB, 0.3), (3.0, YB, 0.3), 0.1, CHROME, 20)
for x in (-3.05, 3.05): rod((x, YB, 0), (x, YB, 5.7), 0.09, CHROME, 16)
for i, p in enumerate(BAS_P):
    x = -2.6 + 5.2 * i / (n - 1); col = hue(i, n, 0.03, 0.1, 0.95)
    s = rod((x, YB, 0.35), (x, YB, 5.55), 0.032 - 0.014 * i / (n - 1), glowmat(f'b{p}', (0.95, 0.7, 0.4), 0.25, 1.0, 7), 10); s.color = (*col, 0.12)
    zh = 3.25 + 0.5 * (i % 3) / 2
    hitp = Vector((x, YB - BR - 0.03, zh))
    l0 = Vector((x, YBL, 0.0)); l1 = Vector((x, YBL + 0.06, 0.6))
    t, ring = tube(l0, l1, 0.075, None)
    TARGET[('bass', p)] = (hitp, l1 + Vector((0, 0.01, 0.03)), Vector((x, YBT, 0.42)), 0.62, col)
    INST[('bass', p)] = [(s, 'string')]; LAUNCH[('bass', p)] = (t, ring)
o = prim('cube', SATIN, size=1, location=(0, YBT, 0.3)); o.scale = (6.0, 0.26, 0.04)
for sy in (-1, 1):
    o = prim('cube', CHROME, size=1, location=(0, YBT + sy * 0.13, 0.38)); o.scale = (6.0, 0.02, 0.16)
o = prim('cube', NE_O, size=1, location=(0, YB + 0.12, 5.75)); o.scale = (6.0, 0.015, 0.02)

# ---------------- the sun: radial xylophone with a central launcher cone ----------------
SUN_P = pset('xmel', 'xlead', 'stab', 'final_sun'); n = len(SUN_P)
C = Vector((0, 0.7, 0)); ZB = 0.5
cone = prim('cone', CHROME, radius1=0.55, radius2=0.22, depth=1.2, vertices=48, location=(C.x, C.y, 0.6))
prim('torus', NE_M, major_radius=0.23, minor_radius=0.03, location=(C.x, C.y, 1.2))
prim('cylinder', BLACKG, radius=0.95, depth=0.3, vertices=64, location=(C.x, C.y, 0.15))
prim('torus', NE_M, major_radius=0.96, minor_radius=0.02, location=(C.x, C.y, 0.3))
SUN_ANG = {}
for i, p in enumerate(SUN_P):
    a = -math.pi / 2 + 2 * math.pi * (i + 0.5) / n + math.pi / n * 0      # starts facing the camera
    a = math.pi * 1.5 - 2 * math.pi * (i + 0.5) / n
    d = Vector((math.cos(a), math.sin(a), 0)); L = 1.35 - 0.55 * i / (n - 1); r0 = 1.1
    col = hue(i, n, 0.83, 1.12)
    mid = C + d * (r0 + L / 2)
    b = prim('cube', None, size=1, location=(mid.x, mid.y, ZB)); b.scale = (L, 0.2, 0.06); b.rotation_euler = (0, 0, a)
    bpy.ops.object.transform_apply(scale=True)
    bv = b.modifiers.new('b', 'BEVEL'); bv.width = 0.015; bv.segments = 2
    b.data.materials.append(glowmat(f'x{p}', (0.85, 0.8, 0.75), 0.18, 1.0, 9)); b.color = (*col, 0.08)
    for f_ in (0.22, 0.78):
        q = C + d * (r0 + L * f_); rod((q.x, q.y, 0), (q.x, q.y, ZB - 0.03), 0.02, SATIN, 8)
    hitp = C + d * (r0 + L * 0.55); hitp.z = ZB + 0.03 + BR
    mouth = C + d * 0.17; mouth.z = 1.24
    cp = C + d * (r0 + L + 0.42); cp.z = 0.42; cup(cp, 0.12)
    TARGET[('sun', p)] = (hitp, mouth, cp, 0.5, col)
    INST[('sun', p)] = [(b, 'bar')]; SUN_ANG[p] = a
prim('torus', NE_V, major_radius=2.98, minor_radius=0.02, location=(C.x, C.y, 0.02))

# ---------------- drum kit (front) ----------------
SHELL = mat('shell', (0.02, 0.02, 0.03), 0.15, 0.6, **{'Coat Weight': 1.0})
HEADM = glowmat('head', (0.1, 0.1, 0.12), 0.35, 0.0, 4)
CYMM = glowmat('cym', (0.95, 0.75, 0.35), 0.18, 1.0, 4)
DR = {  # key: (x, y, head z, radius, kind)
    'kick': (0.0, -3.0, 0.62, 0.62, 'drum'), 'snare': (-1.15, -2.5, 0.85, 0.34, 'drum'),
    'tom1': (0.95, -2.25, 1.05, 0.3, 'drum'), 'tom2': (1.65, -2.75, 0.95, 0.36, 'drum'), 'tom3': (1.2, -3.55, 0.78, 0.45, 'drum'),
    'hat': (-1.8, -2.05, 1.25, 0.3, 'cym'), 'crash': (-1.55, -3.45, 1.65, 0.5, 'cym')}
DCOL = {'kick': MAG, 'snare': (1, 1, 1), 'tom1': CYAN, 'tom2': (0.3, 0.6, 1), 'tom3': VIOLET, 'hat': LIME, 'crash': ORANGE}
DRUMS = {}
LX = {'hat': -1.2, 'crash': -0.8, 'snare': -0.4, 'kick': 0.0, 'tom1': 0.4, 'tom3': 0.8, 'tom2': 1.2}
for k_, (x, y, z, r, kind) in DR.items():
    if kind == 'drum':
        sh = prim('cylinder', SHELL, radius=r, depth=z - 0.08, vertices=48, location=(x, y, (z - 0.08) / 2 + 0.04))
        for zz in (z - 0.02, 0.1):
            prim('torus', CHROME, major_radius=r + 0.01, minor_radius=0.022, location=(x, y, zz))
        prim('torus', NE_C if k_ != 'kick' else NE_M, major_radius=r + 0.015, minor_radius=0.008, location=(x, y, (z + 0.1) / 2))
        hd = prim('cylinder', HEADM, radius=r * 0.97, depth=0.02, vertices=48, location=(x, y, z - 0.01)); hd.color = (*DCOL[k_], 0)
        DRUMS[k_] = hd; resp = [(hd, 'head')]
    else:
        rod((x, y, 0), (x, y, z - 0.05), 0.022, CHROME, 10)
        cy = prim('cone', CYMM, radius1=r, radius2=0.04, depth=0.06, vertices=48, location=(x, y, z)); cy.color = (*DCOL[k_], 0)
        cy.rotation_mode = 'XYZ'
        if k_ == 'hat':
            cy2 = prim('cone', CYMM, radius1=r, radius2=0.04, depth=0.05, vertices=48, location=(x, y, z - 0.07), rotation=(math.pi, 0, 0))
        DRUMS[k_] = cy; resp = [(cy, 'cym')]
    # launcher on the far side (towards the sun), aiming over to the head
    dvec = Vector((x - 0.0, y - (-1.2), 0)); dvec.z = 0; dvec.normalize()
    lb = Vector((x * 0.55, -1.35, 0.0)); lt = lb + Vector((-dvec.x * -0.12, -0.14, 0.62))
    lx = LX[k_]; lb = Vector((lx, -1.3, 0.0)); lt = Vector((lx, -1.42, 0.62))
    t, ring = tube(lb, lt, 0.07, None)
    hitp = Vector((x + 0.25 * r * (1 if x >= 0 else -1) * 0.3, y + 0.25 * r, z + BR + 0.01))
    if kind == 'cym': hitp = Vector((x + 0.55 * r * (1 if x > -1.7 else 1), y + 0.4 * r, z + 0.03 + BR))
    away = Vector((hitp.x - lt.x, hitp.y - lt.y, 0)).normalized()
    cp = Vector((x, y, 0)) + away * (r + 0.42); cp.z = 0.42
    cup(cp, 0.12)
    TARGET[('drum', k_)] = (hitp, lt + Vector((0, 0, 0.03)), cp, 0.55, DCOL[k_])
    INST[('drum', k_)] = resp; LAUNCH[('drum', k_)] = (t, ring)
rod((-1.3, -1.35, 0.05), (1.3, -1.35, 0.05), 0.08, CHROME, 16)

# ---------------- shots ----------------
ROLEKEY = {'mel': 'str', 'arp': 'str', 'final_str': 'str', 'bass': 'bass', 'final_bass': 'bass',
           'xmel': 'sun', 'xlead': 'sun', 'stab': 'sun', 'final_sun': 'sun'}
hits = {}
for t, d, role, p, v in NOTES:
    if role in ROLEKEY: key = (ROLEKEY[role], p)
    else: key = ('drum', 'hat' if role == 'ohat' else role)
    hits.setdefault((key, round(t, 3)), v)
HITS = sorted(((t, key, v) for (key, t), v in hits.items()), key=lambda z: z[0])


def arc(p0, p1, T):
    v0 = (p1 - p0 - 0.5 * GV * T * T) / T
    return lambda tt: p0 + v0 * tt + 0.5 * GV * tt * tt


def drop_time(p0, p1, up=1.6):
    dz = p1.z - p0.z      # bounce with a small upward speed, land in the catcher
    return (up + math.sqrt(up * up - 2 * G * dz)) / G

SHOTS = []   # (t_launch, t_hit, t_end, f1, f2)
for t, key, v in HITS:
    hp, lp, cp, T, col = TARGET[key]
    T2 = min(0.75, max(0.3, drop_time(hp, cp)))
    SHOTS.append((t - T, t, t + T2, arc(lp, hp, T), arc(hp, cp, T2), col, key))

# ball pool: reuse a ball once its previous shot has ended
BALLM = glowmat('ball', (0.9, 0.9, 0.95), 0.12, 1.0, 7)
pool = []   # [obj, free_at, frames dict]
for sh in SHOTS:
    tl, th, te = sh[0], sh[1], sh[2]
    for rec in pool:
        if rec[1] < tl - 0.15: break
    else:
        o = prim('uv_sphere', BALLM, radius=BR, segments=16, ring_count=8, location=(0, 0, -3)); bpy.ops.object.shade_smooth()
        rec = [o, -1, []]; pool.append(rec)
    rec[1] = te; rec[2].append(sh)
print('shots', len(SHOTS), 'balls', len(pool))
for o, _, shots in pool:
    o.color = (1, 1, 1, 0.6)
    last_end = -10
    for tl, th, te, f1, f2, col, key in shots:
        fa = max(0, int(math.floor(tl * FPS))); fb = min(NF - 1, int(math.ceil(te * FPS)))
        o.color = (*col, 1.0); o.keyframe_insert('color', frame=max(0, fa - 1))
        for f in range(max(0, fa - 1), fb + 2):
            if f >= NF: break
            tt = f / FPS
            if tt < tl: p, vis = f1(0), False
            elif tt <= th: p, vis = f1(tt - tl), True
            elif tt <= te: p, vis = f2(tt - th), True
            else: p, vis = f2(te - th), False
            if f == fa and tl > 0: vis = True
            o.location = p; o.hide_render = not vis
            o.keyframe_insert('location', frame=f); o.keyframe_insert('hide_render', frame=f)
    if not shots: o.hide_render = True
    o.hide_render = True; o.keyframe_insert('hide_render', frame=0) if not shots or shots[0][0] * FPS > 1 else None

# ---------------- responses: glow, string vibration, launcher recoil ----------------
def env_series(times, tau=0.12, amp=None):
    a = np.zeros(NF)
    for k, t in enumerate(times):
        m = (TT >= t) & (TT < t + 6 * tau)
        a[m] = np.maximum(a[m], (amp[k] if amp is not None else 1.0) * np.exp(-(TT[m] - t) / tau))
    return a

by_key = {}
for t, key, v in HITS: by_key.setdefault(key, []).append((t, v))
for key, lst in by_key.items():
    ts = [t for t, _ in lst]; amps = [0.45 + 0.55 * v / 127 for _, v in lst]
    for o, kind in INST[key]:
        base_a = o.color[3]
        glow = base_a + env_series(ts, 0.18 if kind == 'string' else 0.12, amps) * (1.6 if kind != 'cym' else 1.2)
        key_series(o, 'color', 3, glow, 3e-3)
        if kind == 'string':
            vib = np.zeros(NF)
            for t in ts:
                m = (TT >= t) & (TT < t + 0.8)
                vib[m] = np.maximum(vib[m], np.exp(-(TT[m] - t) / 0.22) * np.abs(np.sin(2 * np.pi * 7.5 * (TT[m] - t) + 1.2)))
            sx = 1 + 2.8 * vib
            key_series(o, 'scale', 0, sx, 2e-3); key_series(o, 'scale', 1, sx, 2e-3)
        elif kind in ('bar', 'head'):
            z0 = o.location.z; dz = env_series(ts, 0.07) * (0.03 if kind == 'bar' else 0.02)
            key_series(o, 'location', 2, z0 - dz, 1e-4)
        elif kind == 'cym':
            wob = np.zeros(NF)
            for t in ts:
                m = (TT >= t) & (TT < t + 1.5)
                wob[m] += 0.12 * np.exp(-(TT[m] - t) / 0.35) * np.sin(2 * np.pi * 2.6 * (TT[m] - t))
            key_series(o, 'rotation_euler', 0, wob, 1e-4)
            key_series(o, 'rotation_euler', 1, wob * 0.6, 1e-4)
# launchers: neon collar flashes + tube kick on each launch
launch_times = {}
for sh in SHOTS: launch_times.setdefault(sh[6], []).append(sh[0])
for key, (t_, ring) in LAUNCH.items():
    ts = launch_times.get(key, [])
    col = TARGET[key][4]; ring.color = (*col, 0.08)
    key_series(ring, 'color', 3, 0.08 + 1.8 * env_series(ts, 0.1), 3e-3)
    z0 = t_.location.z
    key_series(t_, 'location', 2, z0 - 0.035 * env_series(ts, 0.05) + 0.02 * env_series([x + 0.05 for x in ts], 0.04), 1e-4)
# sun cone pulses with every sun launch
sun_l = sorted(sh[0] for sh in SHOTS if sh[6][0] == 'sun')
key_series(cone, 'scale', 2, 1 - 0.04 * env_series(sun_l, 0.05), 1e-4)

# ---------------- finale: neon shower ----------------
CONF = [neon(f'cf{k}', c, 6) for k, c in enumerate((CYAN, MAG, ORANGE, LIME, VIOLET))]
for k in range(140):
    o = prim('ico_sphere', CONF[k % len(CONF)], radius=0.035, subdivisions=1, location=(0, 0, -5))
    p0 = Vector((rng.uniform(-3.5, 3.5), rng.uniform(-3.5, 6), rng.uniform(5, 8))); delay = rng.uniform(0, 1.2); spd = rng.uniform(1.0, 2.2)
    for f in [0, int((FINAL + delay) * FPS) - 1] + list(range(int((FINAL + delay) * FPS), NF, 3)):
        tt = f / FPS - FINAL - delay
        o.location = (0, 0, -5) if tt < 0 else p0 + Vector((0.25 * math.sin(tt * 3 + delay * 5), 0, -spd * tt))
        o.keyframe_insert('location', frame=f)

# ---------------- camera ----------------
cd = bpy.data.cameras.new('cam'); cd.sensor_fit = 'HORIZONTAL'; cd.sensor_width = 36; cd.clip_end = 200
cam = link(bpy.data.objects.new('cam', cd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trk = cam.constraints.new('TRACK_TO'); trk.target = tgt; trk.track_axis = 'TRACK_NEGATIVE_Z'; trk.up_axis = 'UP_Y'
cd.dof.use_dof = False
A2, Bs, BK, A3, FN = S['A2'], S['B'], S['break'], S['A3'], S['final']
SHOTCAM = [  # start, end, pos0, look0, pos1, look1, lens   (hard cut between shots, eased move within)
    (0.0, 5.0, (-3.4, 2.3, 0.9), (0.0, 4.2, 1.5), (-2.6, 1.9, 1.6), (0.6, 4.2, 1.9), 22),   # low along the string launchers
    (5.0, 10.4, (0.0, -6.6, 5.6), (0.0, 1.4, 0.9), (0.9, -5.8, 4.6), (0.0, 1.4, 1.0), 20),  # whole machine
    (10.4, 15.0, (2.7, -1.3, 3.0), (0.0, 0.7, 0.4), (-2.5, -1.1, 2.7), (0.0, 0.8, 0.4), 22), # over the sun
    (15.0, A2, (-1.4, 1.6, 1.25), (0.0, 4.2, 1.6), (1.6, 1.8, 1.5), (0.4, 4.2, 1.8), 24),    # strings close
    (A2, A2 + 7.0, (0.0, -2.4, 6.4), (0.0, 0.65, 0.0), (1.0, -2.0, 5.4), (0.0, 0.65, 0.0), 24),  # top-down sun
    (A2 + 7.0, A2 + 12.0, (-0.8, -5.9, 2.0), (0.0, -2.6, 0.8), (0.9, -5.6, 1.8), (0.2, -2.7, 0.9), 24),  # drum kit
    (A2 + 12.0, Bs, (0.0, -5.4, 2.6), (0.0, 1.0, 1.2), (0.0, -6.8, 5.4), (0.0, 1.6, 1.0), 20),  # pull back
    (Bs, Bs + 6.0, (-2.8, -1.7, 1.25), (0.0, 0.7, 0.6), (2.6, -1.7, 1.3), (0.0, 0.7, 0.6), 22),  # low on the sun
    (Bs + 6.0, BK, (0.0, 3.0, 0.9), (0.0, 6.2, 3.4), (0.9, 3.4, 1.2), (0.0, 6.2, 3.6), 20),  # bass strings
    (BK, A3, (1.9, -5.3, 1.4), (0.0, -2.8, 1.0), (-1.7, -5.1, 1.6), (0.0, -2.8, 1.0), 22),  # drum break
    (A3, FN, (0.0, -6.8, 5.6), (0.0, 1.4, 1.0), (-2.0, -5.8, 4.2), (0.0, 1.6, 1.2), 20),  # full machine
    (FN, T_END + 0.1, (0.0, -6.6, 5.4), (0.0, 1.6, 1.3), (0.0, -8.6, 7.0), (0.0, 1.8, 1.6), 20),
]


def cam_state(t):
    for t0, t1, p0, l0, p1, l1, lens in SHOTCAM:
        if t0 <= t < t1:
            a = ss((t - t0) / (t1 - t0)) * 0.6 + 0.4 * (t - t0) / (t1 - t0)
            return Vector(p0).lerp(Vector(p1), a), Vector(l0).lerp(Vector(l1), a), lens
    return Vector(SHOTCAM[-1][4]), Vector(SHOTCAM[-1][5]), SHOTCAM[-1][6]


crash_t = [t for t, key, v in HITS if key == ('drum', 'crash')]
for f in range(NF):
    t = f / FPS; p, l, lens = cam_state(t)
    for c in crash_t:
        if 0 <= t - c < 0.4:
            kk = math.exp(-(t - c) / 0.1); p = p + Vector((0.03 * kk * math.sin(90 * t), 0, 0.025 * kk * math.sin(77 * t)))
    cam.location = p; tgt.location = l; cd.lens = lens
    cam.keyframe_insert('location', frame=f); tgt.keyframe_insert('location', frame=f); cd.keyframe_insert('lens', frame=f)
from bpy_extras import anim_utils
for idb in (cam, tgt, cd):      # constant keys: hard cuts stay crisp under motion blur
    ad = idb.animation_data
    cb = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
    for fc in cb.fcurves:
        for kp in fc.keyframe_points: kp.interpolation = 'CONSTANT'

print('objects', len(bpy.data.objects))

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
