"""Chip Fab - a bright high-tech IC factory that plays music.
Melody: laser-etch heads stamp circuits onto chips riding a conveyor. Counter-melody: lasers light dies on a
spinning iridescent wafer. Bass: a chip mounter places finished chips onto a motherboard. Drums: hydraulic
press (kick), rivet stamper (snare), ejector pistons (toms), stack lights (hats), ceiling strobes (crash)."""
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
WHITE = mat('white', (0.78, 0.8, 0.84), 0.32)
PANEL = mat('panel', (0.42, 0.48, 0.58), 0.35)
GREY = mat('grey', (0.2, 0.21, 0.23), 0.35, 0.3)
CHROME = mat('chrome', (0.9, 0.92, 0.95), 0.08, 1.0)
SATIN = mat('satin', (0.72, 0.74, 0.78), 0.25, 1.0)
GOLD = mat('gold', (1.0, 0.76, 0.33), 0.2, 1.0)
EPOXY = mat('epoxy', (0.02, 0.022, 0.026), 0.3)
RUBBER = mat('rubber', (0.03, 0.03, 0.035), 0.6)
GLASS = mat('glass', (0.85, 0.92, 1.0), 0.02, 0.0, **{'Transmission Weight': 1.0, 'IOR': 1.45})
FLOORM = mat('floor', (0.32, 0.36, 0.42), 0.2)
BLUE = (0.1, 0.5, 1.0); CYAN = (0.1, 0.9, 1.0); MINT = (0.2, 1.0, 0.6); AMBER = (1.0, 0.65, 0.1); PINK = (1.0, 0.3, 0.6)
NE_B = neon('ne_b', BLUE, 6); NE_C = neon('ne_c', CYAN, 6); NE_W = neon('ne_w', (1, 1, 1), 5)


def glowpattern(name, col, scale=9.0, gain=10.0, base=(0.02, 0.022, 0.026), gold=False):
    """circuit-trace look: brick mortar lines -> emission mask; object alpha drives strength"""
    m = mat(name, base, 0.3, 1.0 if gold else 0.0); nt = m.node_tree; nn = nt.nodes; ll = nt.links
    b = nn['Principled BSDF']
    br = nn.new('ShaderNodeTexBrick'); br.inputs['Scale'].default_value = scale; br.inputs['Mortar Size'].default_value = 0.035
    br.inputs['Color1'].default_value = (0, 0, 0, 1); br.inputs['Color2'].default_value = (0, 0, 0, 1); br.inputs['Mortar'].default_value = (1, 1, 1, 1)
    br.offset = 0.37; br.squash = 1.6
    oi = nn.new('ShaderNodeObjectInfo')
    m1 = nn.new('ShaderNodeMath'); m1.operation = 'MULTIPLY'; m1.inputs[1].default_value = gain
    m2 = nn.new('ShaderNodeMath'); m2.operation = 'MULTIPLY'
    ll.new(oi.outputs['Alpha'], m1.inputs[0]); ll.new(m1.outputs[0], m2.inputs[0]); ll.new(br.outputs['Fac'], m2.inputs[1])
    ll.new(m2.outputs[0], b.inputs['Emission Strength']); ll.new(oi.outputs['Color'], b.inputs['Emission Color'])
    return m


def beammat(name):
    """laser beam: invisible when object alpha is 0, bright emission when flashed"""
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear(); nn = nt.nodes; ll = nt.links
    oi = nn.new('ShaderNodeObjectInfo'); tr = nn.new('ShaderNodeBsdfTransparent'); em = nn.new('ShaderNodeEmission')
    mul = nn.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = 30
    cl = nn.new('ShaderNodeClamp'); mx = nn.new('ShaderNodeMixShader'); out = nn.new('ShaderNodeOutputMaterial')
    ll.new(oi.outputs['Alpha'], mul.inputs[0]); ll.new(mul.outputs[0], em.inputs['Strength']); ll.new(oi.outputs['Color'], em.inputs['Color'])
    ll.new(oi.outputs['Alpha'], cl.inputs['Value']); ll.new(cl.outputs[0], mx.inputs[0])
    ll.new(tr.outputs[0], mx.inputs[1]); ll.new(em.outputs[0], mx.inputs[2]); ll.new(mx.outputs[0], out.inputs[0])
    return m
BEAM = beammat('beam')


def box(loc, size, mt, bevel=0.0):
    o = prim('cube', mt, size=1, location=loc); o.scale = size; bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        bv = o.modifiers.new('b', 'BEVEL'); bv.width = bevel; bv.segments = 2
    return o


def env_series(times, tau=0.12, amp=None):
    a = np.zeros(NF)
    for k, t in enumerate(times):
        m = (TT >= t) & (TT < t + 6 * tau)
        a[m] = np.maximum(a[m], (amp[k] if amp is not None else 1.0) * np.exp(-(TT[m] - t) / tau))
    return a


def stamp_series(times, pre=0.07, tau=0.1):
    """0 -> 1 as a tool comes down to hit exactly at t, then springs back"""
    a = np.zeros(NF)
    for t in times:
        m = (TT >= t - pre) & (TT < t); a[m] = np.maximum(a[m], ((TT[m] - t + pre) / pre) ** 2)
        m = (TT >= t) & (TT < t + 6 * tau); a[m] = np.maximum(a[m], np.exp(-(TT[m] - t) / tau))
    return a


def hue(i, n, h0, h1, s=0.8):
    return colorsys.hsv_to_rgb((h0 + (h1 - h0) * i / max(1, n - 1)) % 1.0, s, 1.0)


def pset(*roles): return sorted({e[3] for e in NOTES if e[2] in roles})
def times(*roles, p=None): return sorted({round(e[0], 4) for e in NOTES if e[2] in roles and (p is None or e[3] == p)})

# ---------------- clean-room stage (bright) ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bgn = w.node_tree.nodes['Background']; bgn.inputs['Color'].default_value = (0.75, 0.8, 0.9, 1); bgn.inputs['Strength'].default_value = 0.12
prim('plane', FLOORM, size=60, location=(0, 2, 0))
for i in range(-8, 9):                 # floor safety/aisle lines
    pass
for x in (-4.9, 4.9):
    box((x, 1.0, 0.004), (0.08, 12, 0.006), mat('yel', (1.0, 0.75, 0.05), 0.4))
box((0, 7.0, 3.5), (16, 0.2, 7), PANEL)
for sx in (-1, 1): box((sx * 6.2, 1.5, 3.5), (0.2, 14, 7), PANEL)
for k, x in enumerate(np.linspace(-5, 5, 6)):          # window bands with blue light
    box((x, 6.88, 3.8), (1.2, 0.02, 1.6), NE_B if k % 2 == 0 else NE_C)
    box((x, 6.86, 3.8), (1.36, 0.04, 1.76), GREY)
CEIL = []
cm = neon('ceil', (1.0, 1.0, 1.0), 2.0)
for x in (-3.6, -1.2, 1.2, 3.6):
    for y in (-2.0, 1.0, 4.0):
        CEIL.append(box((x, y, 6.0), (1.8, 1.2, 0.05), cm))
box((0, 1.5, 6.06), (13, 12, 0.08), PANEL)
cl = w.node_tree.nodes['Background']


def area(name, loc, aim, col, e, size):
    ld = bpy.data.lights.new(name, 'AREA'); ld.energy = e; ld.color = col; ld.size = size
    o = link(bpy.data.objects.new(name, ld)); o.location = loc
    tr = o.constraints.new('TRACK_TO'); t = link(bpy.data.objects.new(name + 't', None)); t.location = aim; tr.target = t
    tr.track_axis = 'TRACK_NEGATIVE_Z'; tr.up_axis = 'UP_Y'
    return ld
KEYL = area('key', (0, 0.5, 5.8), (0, 0.5, 0), (1.0, 0.98, 0.95), 1100, 7)
area('front', (0, -7, 4.5), (0, 1, 1), (0.9, 0.95, 1.0), 450, 6)
area('blueL', (-5.5, 1, 3), (0, 1, 1), (0.5, 0.75, 1.0), 700, 3)
area('warmR', (5.5, 1, 3), (0, 1, 1), (1.0, 0.85, 0.7), 700, 3)

# ---------------- chips (one shared mesh) ----------------
def chip_mesh(name, s=0.26, h=0.035, pins=6):
    bm = bmesh.new()
    def cube(c, d, mi):
        r = bmesh.ops.create_cube(bm, size=1.0)
        for v in r['verts']: v.co = Vector((c[0] + v.co.x * d[0], c[1] + v.co.y * d[1], c[2] + v.co.z * d[2]))
        for f in {f for v in r['verts'] for f in v.link_faces}: f.material_index = mi
    cube((0, 0, h / 2), (s, s, h), 0)
    for k in range(pins):
        u = -s / 2 + s * (k + 0.5) / pins
        for sx in (-1, 1):
            cube((sx * (s / 2 + 0.018), u, 0.012), (0.036, s / pins * 0.45, 0.012), 1)
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    me.materials.append(EPOXY); me.materials.append(GOLD)
    return me
CHIPME = chip_mesh('chip')
PATME = bpy.data.meshes.new('pat')
bmp = bmesh.new(); bmesh.ops.create_grid(bmp, x_segments=1, y_segments=1, size=0.11); bmp.to_mesh(PATME); bmp.free()


def make_chip(name, loc, patmat, col):
    c = link(bpy.data.objects.new(name, CHIPME)); c.location = loc
    p = link(bpy.data.objects.new(name + 'p', PATME.copy() if False else PATME)); p.parent = c; p.location = (0, 0, 0.0365)
    if not p.data.materials: p.data.materials.append(patmat)
    p.color = (*col, 0.0)
    return c, p
PATM = glowpattern('chippat', CYAN, scale=6.0, gain=9.0)

# ---------------- etch line: conveyor + laser heads (melody) ----------------
YE = 3.2; ZB = 0.95; V = 0.35; SP = 0.42; XS = -5.4; XE = 5.4
box((0, YE, ZB / 2 - 0.03), (10.2, 0.7, ZB - 0.06), WHITE, 0.02)
belt = box((0, YE, ZB - 0.015), (10.0, 0.62, 0.03), RUBBER)
for sy in (-1, 1): box((0, YE + sy * 0.34, ZB + 0.02), (10.2, 0.04, 0.08), SATIN)
box((0, YE - 0.37, ZB - 0.2), (10.2, 0.012, 0.03), NE_C)
for sx in (-1, 1):   # wafer feeder / packing stations hiding the ends of the belt
    box((sx * 5.4, YE, 0.95), (1.4, 1.2, 1.9), WHITE, 0.04)
    box((sx * 5.4 - sx * 0.71, YE, 1.15), (0.02, 0.7, 0.3), GREY)
    box((sx * 5.4 - sx * 0.715, YE, 1.6), (0.01, 0.9, 0.12), NE_B)
STR_P = pset('mel', 'arp', 'final_str'); nS = len(STR_P)
HX = {p: -3.0 + 6.0 * i / (nS - 1) for i, p in enumerate(STR_P)}
NCHIP = int((T_END + (XE - XS) / V) / (SP / V)) + 2
CHIPS = []
for j in range(NCHIP):
    t0 = j * SP / V - (XE - XS) / V           # chip j enters at XS at time t0
    c, p = make_chip(f'c{j}', (XS, YE, ZB), PATM, CYAN)
    c.rotation_euler = (0, 0, 0)
    for f in (0, NF - 1):
        c.location = (XS + V * (f / FPS - t0), YE, ZB); c.keyframe_insert('location', index=0, frame=f)
    CHIPS.append((c, p, t0))
def chip_at(x, t):
    j = int(round(((t - (x - XS) / V) + (XE - XS) / V) / (SP / V)))
    return j if 0 <= j < NCHIP else None
# gantry
for sx in (-1, 1):
    box((sx * 3.5, YE, 1.6), (0.14, 0.14, 3.2), WHITE, 0.02)
box((0, YE, 3.15), (7.2, 0.3, 0.26), WHITE, 0.03)
box((0, YE - 0.155, 3.15), (7.0, 0.01, 0.04), NE_B)
HEADS = {}
for i, p in enumerate(STR_P):
    x = HX[p]; col = hue(i, nS, 0.5, 0.95)
    rod((x, YE, 2.4), (x, YE, 3.02), 0.03, CHROME, 12)
    h = box((x, YE, 2.25), (0.24, 0.3, 0.34), WHITE, 0.03)
    noz = prim('cone', GREY, radius1=0.035, radius2=0.08, depth=0.12, vertices=16, location=(0, 0, -0.23)); noz.parent = h
    led = box((0, -0.152, 0.07), (0.16, 0.01, 0.05), glowmat(f'led{p}', (0.1, 0.1, 0.1), 0.3, 0, 8)); led.parent = h; led.color = (*col, 0.25)
    bm_ = prim('cylinder', BEAM, radius=0.012, depth=1.0, vertices=8, location=(0, 0, 0)); bm_.parent = h; bm_.color = (*col, 0)
    HEADS[p] = (h, led, bm_, col)

# ---------------- wafer station (counter-melody) ----------------
WC = Vector((0, 0.5, 0)); WZ = 0.78; WR = 1.12; OMEGA = 0.18
box((WC.x, WC.y, 0.3), (2.9, 2.9, 0.6), WHITE, 0.05)
box((WC.x, WC.y - 1.452, 0.42), (2.2, 0.01, 0.04), NE_C)
prim('cylinder', SATIN, radius=1.22, depth=0.16, vertices=96, location=(WC.x, WC.y, 0.68))
WAFM = mat('wafer', (0.55, 0.57, 0.62), 0.07, 1.0)
try:
    WAFM.node_tree.nodes['Principled BSDF'].inputs['Thin Film Thickness'].default_value = 420
    WAFM.node_tree.nodes['Principled BSDF'].inputs['Thin Film IOR'].default_value = 1.9
except Exception: pass
wafer = prim('cylinder', WAFM, radius=WR, depth=0.02, vertices=128, location=(WC.x, WC.y, WZ))
DIEM = glowmat('die', (0.06, 0.07, 0.12), 0.25, 0.6, 4.0)
DIES = []
st = 0.122
for a_ in np.arange(-WR, WR, st):
    for b_ in np.arange(-WR, WR, st):
        u, v = a_ + st / 2, b_ + st / 2
        if u * u + v * v < (WR - 0.09) ** 2:
            d = box((u, v, 0.012), (st * 0.86, st * 0.86, 0.004), DIEM); d.parent = wafer
            d.color = (*hue(len(DIES) % 7, 7, 0.38, 0.62), 0.0); DIES.append([d, u, v, []])
key_series(wafer, 'rotation_euler', 2, OMEGA * TT)
SUN_P = pset('xmel', 'xlead', 'stab', 'final_sun'); nU = len(SUN_P)
PROBE = {}
for i, p in enumerate(SUN_P):
    a = math.pi * 1.5 + 2 * math.pi * (i + 0.5) / nU; d = Vector((math.cos(a), math.sin(a), 0))
    col = hue(i, nU, 0.3, 0.75)
    base = WC + d * 1.38; top = base + Vector((0, 0, 1.55))
    rod(base + Vector((0, 0, 0.6)), top, 0.025, CHROME, 10)
    hd = box(top + Vector((0, 0, 0.0)) - d * 0.05, (0.13, 0.13, 0.16), WHITE, 0.02)
    lamp = prim('uv_sphere', glowmat(f'pl{p}', (0.1, 0.1, 0.1), 0.3, 0, 8), radius=0.04, location=top + Vector((0, 0, 0.11))); lamp.color = (*col, 0.3)
    rt = 0.32 + 0.66 * ((i * 7) % nU) / (nU - 1)
    tgt = WC + d * rt; tgt.z = WZ + 0.016
    nz = top - d * 0.05 - Vector((0, 0, 0.09))
    bm_ = rod(nz, tgt, 0.01, BEAM, 8); bm_.color = (*col, 0)
    PROBE[p] = (lamp, bm_, tgt, col)

# ---------------- motherboard + chip mounter (bass) ----------------
MB = Vector((0, -2.25, 0.82)); MW, MD = 3.4, 1.9
box((MB.x, MB.y, 0.4), (3.9, 2.4, 0.8), WHITE, 0.05)
box((MB.x, MB.y - 1.21, 0.62), (3.0, 0.01, 0.04), NE_B)
BOARDM = glowpattern('board', AMBER, scale=4.0, gain=4.0, base=(0.02, 0.09, 0.07))
board = box((MB.x, MB.y, MB.z), (MW, MD, 0.03), BOARDM); board.color = (*AMBER, 0.15)
bass_ev = [e for e in NOTES if e[2] in ('bass', 'final_bass')]
PLACE = []
for e in bass_ev:
    beat = round(((e[0] - E['t0']) / E['beat']) % 4, 3)
    if beat in (0.0, 2.5) and e[2] == 'bass': PLACE.append(e[0])
PLACE = sorted(set(PLACE))
COLS = 14; ROWS = int(math.ceil(len(PLACE) / COLS))
SLOTS = []
for k in range(len(PLACE)):
    r, c = divmod(k, COLS)
    if r % 2: c = COLS - 1 - c                          # snake order: short gantry moves
    SLOTS.append(Vector((MB.x - MW / 2 + 0.17 + (MW - 0.34) * c / (COLS - 1), MB.y + MD / 2 - 0.18 - (MD - 0.36) * r / max(1, ROWS - 1), MB.z + 0.015)))
PLACED = []
BPAT = glowpattern('bpat', AMBER, scale=6.0, gain=9.0)
for k, sl in enumerate(SLOTS):
    c, p = make_chip(f'pc{k}', sl, BPAT, AMBER); c.scale = (0.7, 0.7, 0.7)
    PLACED.append((c, p))
# gantry: rails along y, beam along x (moves in y), head (moves in x)
for sx in (-1, 1):
    box((sx * 1.95, MB.y, 1.55), (0.1, 2.5, 0.1), SATIN)
    for sy in (-1, 1): box((sx * 1.95, MB.y + sy * 1.2, 1.18), (0.1, 0.1, 0.75), WHITE)
gbeam = box((0, MB.y, 1.66), (4.0, 0.16, 0.14), WHITE, 0.02)
gled = box((0, -0.082, 0), (3.6, 0.01, 0.03), NE_C); gled.parent = gbeam; gled.location = (0, -0.082, 0)
ghead = box((0, MB.y, 1.4), (0.2, 0.22, 0.32), WHITE, 0.02)
gnoz = rod((0, 0, -0.16), (0, 0, -0.42), 0.022, CHROME, 10); gnoz.parent = ghead; gnoz.location = (0, 0, -0.29)
carry, carryp = make_chip('carry', (0, 0, -0.45), BPAT, AMBER); carry.parent = ghead; carry.location = (0, 0, -0.47); carry.scale = (0.7, 0.7, 0.7)
# feeder reels (decor)
for k in range(4):
    r_ = prim('cylinder', GREY, radius=0.32, depth=0.06, vertices=48, location=(-2.25 + 0.0, MB.y - 0.7 + k * 0.45, 1.15), rotation=(0, math.pi / 2, 0))
    prim('cylinder', NE_B if k % 2 else NE_C, radius=0.33, depth=0.01, vertices=48, location=(-2.29, MB.y - 0.7 + k * 0.45, 1.15), rotation=(0, math.pi / 2, 0))

# ---------------- drums ----------------
# kick: hydraulic press (right)
PX, PY = 3.3, 0.5
for sx in (-1, 1): box((PX + sx * 0.55, PY, 1.4), (0.18, 0.5, 2.8), WHITE, 0.03)
box((PX, PY, 2.85), (1.4, 0.6, 0.3), WHITE, 0.03); box((PX, PY, 0.35), (1.4, 0.8, 0.7), GREY, 0.03)
box((PX, PY - 0.41, 0.5), (1.0, 0.01, 0.05), NE_B)
anvil = box((PX, PY, 0.78), (0.7, 0.5, 0.16), glowmat('anvil', (0.6, 0.62, 0.66), 0.2, 1.0, 6)); anvil.color = (*PINK, 0)
rod((PX, PY, 1.8), (PX, PY, 2.7), 0.14, CHROME, 24)
ram = box((PX, PY, 1.62), (0.82, 0.56, 0.36), WHITE, 0.03)
rled = box((0, -0.285, 0), (0.6, 0.01, 0.05), glowmat('rled', (0.1, 0.1, 0.1), 0.3, 0, 8)); rled.parent = ram; rled.location = (0, -0.285, 0); rled.color = (*PINK, 0.2)
tray = box((PX, PY, 0.89), (0.5, 0.4, 0.05), EPOXY)
# snare: rivet stamper (left)
SX_, SY_ = -3.3, 0.5
box((SX_, SY_, 0.45), (0.9, 0.8, 0.9), WHITE, 0.03); box((SX_ + 0.32, SY_, 1.35), (0.16, 0.4, 1.8), WHITE, 0.02)
box((SX_ + 0.05, SY_, 2.25), (0.75, 0.4, 0.22), WHITE, 0.02)
srod = rod((SX_, SY_, 1.4), (SX_, SY_, 2.15), 0.05, CHROME, 16)
stool = prim('cylinder', glowmat('stool', (0.85, 0.86, 0.9), 0.2, 1.0, 7), radius=0.16, depth=0.18, vertices=32, location=(SX_, SY_, 1.32)); stool.color = (*CYAN, 0)
splate = box((SX_, SY_, 0.92), (0.5, 0.5, 0.04), glowmat('splate', (0.3, 0.32, 0.36), 0.3, 1.0, 5)); splate.color = (*CYAN, 0)
# toms: three ejector pistons (front left)
TOMS = {}
for k, (r_, x_, y_, col) in enumerate(((0.13, -3.4, -1.0, MINT), (0.16, -3.0, -1.6, BLUE), (0.2, -3.5, -2.3, PINK))):
    rl = ('tom1', 'tom2', 'tom3')[k]
    prim('cylinder', WHITE, radius=r_ + 0.06, depth=0.8, vertices=32, location=(x_, y_, 0.4))
    prim('torus', NE_C, major_radius=r_ + 0.065, minor_radius=0.012, location=(x_, y_, 0.8))
    cap = prim('cylinder', glowmat(f'cap{k}', (0.2, 0.2, 0.22), 0.3, 0.5, 8), radius=r_, depth=0.1, vertices=32, location=(x_, y_, 0.83)); cap.color = (*col, 0.1)
    TOMS[rl] = cap
# hats: stack-light towers + a ticking flipper at the end of the conveyor
STACKS = []
for x_ in (-4.3, 4.3):
    rod((x_, YE - 0.6, 0), (x_, YE - 0.6, 1.6), 0.03, SATIN, 10)
    segs = []
    for k, col in enumerate((MINT, AMBER, (1, 0.15, 0.15))):
        o = prim('cylinder', glowmat(f'st{x_}{k}', (0.3, 0.3, 0.3), 0.2, 0, 10), radius=0.07, depth=0.12, vertices=20, location=(x_, YE - 0.6, 1.66 + k * 0.13)); o.color = (*col, 0.08)
        segs.append(o)
    STACKS.append(segs)
flip = box((4.1, YE - 0.2, ZB + 0.12), (0.05, 0.36, 0.14), CHROME); flip.rotation_mode = 'XYZ'

# ======================= animation =======================
FIN = E['final']
# --- etch heads + chips ---
etch = {}
for p, (h, led, bm_, col) in HEADS.items():
    ts = times('mel', 'arp', 'final_str', p=p)
    st_ = stamp_series(ts, 0.06, 0.09)
    hz = 2.25 - 0.3 * st_
    key_series(h, 'location', 2, hz, 1e-4)
    L = (hz - 0.29) - (ZB + 0.036)
    key_series(bm_, 'scale', 2, L, 1e-4); key_series(bm_, 'location', 2, -0.29 - L / 2, 1e-4)
    key_series(bm_, 'color', 3, env_series(ts, 0.07), 3e-3)
    key_series(led, 'color', 3, 0.25 + 1.5 * env_series(ts, 0.15), 3e-3)
    for t in ts:
        j = chip_at(HX[p], t)
        if j is not None: etch.setdefault(j, []).append(t)
for j, ts in etch.items():
    c, p, t0 = CHIPS[j]; ts = sorted(ts)
    lvl = np.zeros(NF)
    for t in ts: lvl += 0.3 * (TT >= t)
    a = np.minimum(1.2, lvl) + 1.5 * env_series(ts, 0.12)
    key_series(p, 'color', 3, a, 3e-3)
# --- wafer dies ---
for p, (lamp, bm_, tgt, col) in PROBE.items():
    ts = times('xmel', 'xlead', 'stab', 'final_sun', p=p)
    key_series(bm_, 'color', 3, env_series(ts, 0.08), 3e-3)
    key_series(lamp, 'color', 3, 0.3 + 1.5 * env_series(ts, 0.15), 3e-3)
    for t in ts:
        th = OMEGA * t; q = tgt - WC
        u = math.cos(-th) * q.x - math.sin(-th) * q.y; v = math.sin(-th) * q.x + math.cos(-th) * q.y
        k = min(range(len(DIES)), key=lambda i: (DIES[i][1] - u) ** 2 + (DIES[i][2] - v) ** 2)
        DIES[k][3].append(t)
fin_flash = env_series([FIN], 0.8)
for d, u, v, ts in DIES:
    lit = np.zeros(NF)
    if ts: lit = 0.55 * (TT >= min(ts))
    ring = math.hypot(u, v) / WR          # finale: a light wave runs out from the centre
    wave = env_series([FIN + 0.5 * ring], 0.35)
    a = lit + 1.8 * env_series(sorted(ts), 0.12) + 1.2 * wave + 0.5 * (TT >= FIN + 0.5 * ring)
    key_series(d, 'color', 3, a, 3e-3)
# --- chip mounter ---
hx = np.zeros(NF); hy = np.zeros(NF)
pts = [(PLACE[0] - 2.0, SLOTS[0])] + [(t, s) for t, s in zip(PLACE, SLOTS)]
for f in range(NF):
    t = TT[f]
    if t <= PLACE[0]: P = SLOTS[0]
    elif t >= PLACE[-1]: P = SLOTS[-1]
    else:
        k = int(np.searchsorted(PLACE, t))          # moving from slot k-1 to slot k
        ta, tb = PLACE[k - 1] + 0.12, PLACE[k] - 0.07
        a = 0 if t < ta else (1 if t > tb else ss((t - ta) / max(1e-3, tb - ta)))
        P = SLOTS[k - 1].lerp(SLOTS[k], a)
    hx[f], hy[f] = P.x, P.y
key_series(ghead, 'location', 0, hx, 1e-4); key_series(ghead, 'location', 1, hy, 1e-4); key_series(gbeam, 'location', 1, hy, 1e-4)
hz = 1.45 - (1.45 - (MB.z + 0.015 + 0.47 * 0.7 + 0.03)) * stamp_series(PLACE, 0.07, 0.09)
key_series(ghead, 'location', 2, hz, 1e-4)
vis = np.ones(NF)
for t in PLACE: vis[(TT >= t) & (TT < t + 0.16)] = 0
for f in range(NF):
    if f == 0 or f == NF - 1 or vis[f] != vis[f - 1] or vis[f] != vis[f + 1]:
        carry.hide_render = carryp.hide_render = bool(vis[f] < 0.5)
        carry.keyframe_insert('hide_render', frame=f); carryp.keyframe_insert('hide_render', frame=f)
for k, (c, p) in enumerate(PLACED):
    t = PLACE[k]; f0 = int(round(t * FPS))
    for f, hid in ((0, True), (max(0, f0 - 1), True), (f0, False)):
        c.hide_render = p.hide_render = hid; c.keyframe_insert('hide_render', frame=f); p.keyframe_insert('hide_render', frame=f)
    a = 0.45 * (TT >= t) + 1.6 * env_series([t], 0.15) + 1.0 * env_series([FIN + 0.03 * k], 0.4) + 0.5 * (TT >= FIN + 0.03 * k)
    key_series(p, 'color', 3, a, 3e-3)
bt = [e[0] for e in bass_ev]
key_series(board, 'color', 3, 0.15 + 0.45 * env_series(bt, 0.12) + 0.9 * (TT >= FIN) * np.minimum(1, (TT - FIN) / 0.8 + 0.001), 3e-3)
# --- kick press, snare stamper, toms, hats ---
kt = times('kick'); st_ = stamp_series(kt, 0.09, 0.12)
key_series(ram, 'location', 2, 1.62 - 0.52 * st_, 1e-4)
key_series(anvil, 'color', 3, 1.4 * env_series(kt, 0.12), 3e-3); key_series(rled, 'color', 3, 0.2 + 1.6 * env_series(kt, 0.15), 3e-3)
snt = times('snare'); s2 = stamp_series(snt, 0.06, 0.08)
key_series(stool, 'location', 2, 1.32 - 0.31 * s2, 1e-4); key_series(srod, 'location', 2, srod.location.z - 0.31 * s2, 1e-4)
key_series(stool, 'color', 3, 1.4 * env_series(snt, 0.1), 3e-3); key_series(splate, 'color', 3, 1.2 * env_series(snt, 0.1), 3e-3)
for rl, cap in TOMS.items():
    tt_ = times(rl); key_series(cap, 'location', 2, 0.83 + 0.2 * env_series(tt_, 0.07), 1e-4)
    key_series(cap, 'color', 3, 0.1 + 2.0 * env_series(tt_, 0.12), 3e-3)
ht = times('hat'); oht = times('ohat')
for segs in STACKS:
    key_series(segs[0], 'color', 3, 0.08 + 1.6 * env_series(ht, 0.07), 3e-3)
    key_series(segs[1], 'color', 3, 0.08 + 1.8 * env_series(oht, 0.2) + 1.5 * (TT >= FIN), 3e-3)
    key_series(segs[2], 'color', 3, 0.08 + 1.8 * env_series(times('crash'), 0.3), 3e-3)
key_series(flip, 'rotation_euler', 2, 0.5 * stamp_series(ht, 0.05, 0.06), 1e-4)
# --- crash: ceiling strobes ---
ct = times('crash'); cs = 2.0 + 8.0 * env_series(ct, 0.22)
sock = cm.node_tree.nodes['Emission'].inputs['Strength']
for f in range(NF):
    if f == 0 or f == NF - 1 or abs(cs[f] - cs[f - 1]) > 0.02 or abs(cs[f] - cs[f + 1]) > 0.02:
        sock.default_value = float(cs[f]); sock.keyframe_insert('default_value', frame=f)

# ---------------- camera ----------------
cd = bpy.data.cameras.new('cam'); cd.sensor_fit = 'HORIZONTAL'; cd.sensor_width = 36; cd.clip_end = 200
cam = link(bpy.data.objects.new('cam', cd)); sc.camera = cam
tgo = link(bpy.data.objects.new('tgt', None))
trk = cam.constraints.new('TRACK_TO'); trk.target = tgo; trk.track_axis = 'TRACK_NEGATIVE_Z'; trk.up_axis = 'UP_Y'
A2, Bs, BK, A3 = S['A2'], S['B'], S['break'], S['A3']
SHOTCAM = [
    (0.0, 5.0, (-3.4, 0.9, 1.8), (0.0, 3.2, 1.8), (-1.8, 1.0, 1.9), (1.2, 3.2, 1.8), 22),    # along the etch line
    (5.0, 10.4, (0.0, -6.2, 4.6), (0.0, 1.0, 1.1), (1.0, -5.6, 4.0), (0.0, 1.0, 1.1), 20),   # whole factory
    (10.4, 15.0, (1.7, -1.1, 2.6), (0.0, 0.5, 0.7), (-1.7, -1.0, 2.5), (0.0, 0.5, 0.7), 24),  # wafer
    (15.0, A2, (-1.6, 1.8, 1.6), (-0.6, 3.2, 1.6), (1.6, 1.8, 1.7), (0.8, 3.2, 1.6), 26),    # heads close
    (A2, A2 + 7.0, (0.0, -0.5, 4.4), (0.0, 0.5, 0.7), (0.5, -0.3, 3.8), (0.0, 0.5, 0.7), 26),  # wafer from above
    (A2 + 7.0, A2 + 12.0, (1.9, -1.6, 1.6), (3.3, 0.5, 1.3), (2.4, -1.8, 1.9), (3.3, 0.5, 1.3), 24),  # press
    (A2 + 12.0, Bs, (0.0, -4.6, 2.4), (0.0, 1.0, 1.2), (0.0, -7.0, 5.0), (0.0, 1.0, 1.0), 20),
    (Bs, Bs + 6.0, (-2.2, -1.5, 1.35), (0.0, 0.5, 0.8), (2.2, -1.5, 1.35), (0.0, 0.5, 0.8), 22),  # wafer low
    (Bs + 6.0, BK, (0.6, -4.5, 2.6), (0.0, -2.2, 0.9), (-0.6, -4.3, 2.3), (0.0, -2.2, 0.9), 24),  # chip mounter
    (BK, A3, (0.0, -5.2, 1.6), (0.0, 0.5, 1.2), (0.0, -4.5, 1.9), (0.0, 0.5, 1.2), 18),        # drum break
    (A3, FIN, (0.0, -6.2, 4.6), (0.0, 0.8, 1.1), (-2.2, -5.4, 3.8), (0.0, 0.8, 1.1), 20),
    (FIN, T_END + 0.1, (0.0, -4.4, 3.4), (0.0, -1.4, 0.8), (0.0, -7.6, 5.6), (0.0, 0.6, 1.0), 22),
]


def cam_state(t):
    for t0, t1, p0, l0, p1, l1, lens in SHOTCAM:
        if t0 <= t < t1:
            a = ss((t - t0) / (t1 - t0)) * 0.6 + 0.4 * (t - t0) / (t1 - t0)
            return Vector(p0).lerp(Vector(p1), a), Vector(l0).lerp(Vector(l1), a), lens
    return Vector(SHOTCAM[-1][4]), Vector(SHOTCAM[-1][5]), SHOTCAM[-1][6]


for f in range(NF):
    t = f / FPS; p, l, lens = cam_state(t)
    for c in kt:
        if 0 <= t - c < 0.3 and (A2 + 7.0 <= t < A2 + 12.0 or BK <= t < A3):
            kk = math.exp(-(t - c) / 0.07); p = p + Vector((0, 0, 0.02 * kk * math.sin(80 * t)))
    cam.location = p; tgo.location = l; cd.lens = lens
    cam.keyframe_insert('location', frame=f); tgo.keyframe_insert('location', frame=f); cd.keyframe_insert('lens', frame=f)
from bpy_extras import anim_utils
for idb in (cam, tgo, cd):
    ad = idb.animation_data
    cb = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
    for fc in cb.fcurves:
        for kp in fc.keyframe_points: kp.interpolation = 'CONSTANT'
for c, p, t0 in CHIPS:     # conveyor chips: straight-line motion
    ad = c.animation_data; cb = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
    for fc in cb.fcurves:
        for kp in fc.keyframe_points: kp.interpolation = 'LINEAR'

print('objects', len(bpy.data.objects), 'chips', NCHIP, 'dies', len(DIES), 'placed', len(PLACE), 'lit dies', sum(1 for d in DIES if d[3]))

if MODE == 'dbg': exec(open('/tmp/dbg.py').read())
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
