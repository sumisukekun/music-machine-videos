"""Toccata and Fugue in D minor - 'The Midnight Cathedral' (真夜中の大聖堂), side view of a marble organ in a gothic cathedral.
Every organ note = a marble on a key (ivory / ebony keys; the key glows, the organ pipes of that pitch light up behind).
layout.py plans all motion (constant gravity, bounces that never add speed, rolling ramps, visible mechanisms);
this file builds and animates it. Fonts: DejaVu Sans / Serif Bold (en_bold.ttf, en_serif_bold.ttf, DejaVu license),
Noto Sans CJK JP Black subset (jp_sub.otf, SIL OFL, make_font.py).
python3 scene.py -- test t1 t2 ... | range f0 f1"""
import os, sys, json, math, time
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
EVJ = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json')); LAY = json.load(open(f'{HERE}/layout.json'))
S = EVJ['secs']; FPS = 30; FINAL = EVJ['final']; T_END = S['end']
NF = int((T_END + 3.4) * FPS); T = np.arange(NF) / FPS
G = LAY['G']; R = LAY['R']; P = LAY['parts']; KEYS = LAY['keys']
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
sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.5
try: sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Punchy'
except Exception: pass
sc.frame_start, sc.frame_end = 0, NF - 1
FONT = bpy.data.fonts.load(f'{HERE}/jp_sub.otf')
FONT_EN = bpy.data.fonts.load(f'{HERE}/en_bold.ttf'); FONT_SERIF = bpy.data.fonts.load(f'{HERE}/en_serif_bold.ttf')
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def link(o): sc.collection.objects.link(o); return o


def mat(name, col, rough=0.5, **kw):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']; b.inputs['Base Color'].default_value = (*col, 1); b.inputs['Roughness'].default_value = rough
    for k, v in kw.items():
        if k in ('Emission Color',) and len(v) == 3: v = (*v, 1)
        b.inputs[k].default_value = v
    return m


def emis(name, col, s):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    e = nt.nodes.new('ShaderNodeEmission'); e.inputs['Color'].default_value = (*col, 1); e.inputs['Strength'].default_value = s
    o = nt.nodes.new('ShaderNodeOutputMaterial'); nt.links.new(e.outputs[0], o.inputs[0]); return m


def glowmat(name, gain=8.0, base=0.0, rough=0.3, metal=0.0, coat=0.4, glow=None):
    """object colour = base colour; object alpha drives emission (hit flash). glow: fixed emission colour (else = object colour)"""
    m = mat(name, (1, 1, 1), rough, **{'Coat Weight': coat, 'Metallic': metal}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    oi = nt.nodes.new('ShaderNodeObjectInfo')
    nt.links.new(oi.outputs['Color'], b.inputs['Base Color'])
    if glow is None: nt.links.new(oi.outputs['Color'], b.inputs['Emission Color'])
    else: b.inputs['Emission Color'].default_value = (*glow, 1)
    mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY_ADD'; mul.inputs[1].default_value = gain; mul.inputs[2].default_value = base
    nt.links.new(oi.outputs['Alpha'], mul.inputs[0]); nt.links.new(mul.outputs[0], b.inputs['Emission Strength'])
    return m


def fxmat(name, gain=10.0):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    oi = nt.nodes.new('ShaderNodeObjectInfo'); em = nt.nodes.new('ShaderNodeEmission'); tr_ = nt.nodes.new('ShaderNodeBsdfTransparent')
    mx = nt.nodes.new('ShaderNodeMixShader'); mu = nt.nodes.new('ShaderNodeMath'); mu.operation = 'MULTIPLY'; mu.inputs[1].default_value = gain
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(oi.outputs['Color'], em.inputs['Color']); nt.links.new(oi.outputs['Alpha'], mu.inputs[0]); nt.links.new(mu.outputs[0], em.inputs['Strength'])
    nt.links.new(oi.outputs['Alpha'], mx.inputs['Fac']); nt.links.new(tr_.outputs[0], mx.inputs[1]); nt.links.new(em.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs[0]); return m


def objcol(name, rough=0.45, metal=0.0):
    m = mat(name, (1, 1, 1), rough, **{'Metallic': metal}); nt = m.node_tree
    oi = nt.nodes.new('ShaderNodeObjectInfo'); nt.links.new(oi.outputs['Color'], nt.nodes['Principled BSDF'].inputs['Base Color'])
    return m


def stone(name, c1, c2, scale=3.0, rough=0.85):
    m = mat(name, c1, rough); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    tc = nt.nodes.new('ShaderNodeTexCoord'); nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = scale
    nz.inputs['Detail'].default_value = 6.0
    br = nt.nodes.new('ShaderNodeTexBrick'); br.inputs['Scale'].default_value = 0.9; br.inputs['Mortar Size'].default_value = 0.015
    br.inputs['Color1'].default_value = (*c1, 1); br.inputs['Color2'].default_value = (*c2, 1); br.inputs['Mortar'].default_value = (0.05, 0.045, 0.04, 1)
    nt.links.new(tc.outputs['Object'], br.inputs['Vector']); nt.links.new(tc.outputs['Object'], nz.inputs['Vector'])
    mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'; mix.inputs['Factor'].default_value = 0.35
    nt.links.new(br.outputs['Color'], mix.inputs[6]); nt.links.new(nz.outputs['Color'], mix.inputs[7])
    nt.links.new(mix.outputs[2], b.inputs['Base Color'])
    bump = nt.nodes.new('ShaderNodeBump'); bump.inputs['Strength'].default_value = 0.25
    nt.links.new(br.outputs['Fac'], bump.inputs['Height']); nt.links.new(bump.outputs['Normal'], b.inputs['Normal'])
    return m


def key_series(obj, path, idx, vals, eps=1e-5):
    n = len(vals)
    for f in range(n):
        v = vals[f]
        if 0 < f < n - 1 and abs(v - vals[f - 1]) < eps and abs(v - vals[f + 1]) < eps: continue
        getattr(obj, path)[idx] = float(v); obj.keyframe_insert(path, index=idx, frame=f)


def key_loc(o, Pa):
    for k in range(3): key_series(o, 'location', k, Pa[:, k])


_MC = {}
def prim(kind, m, *a):
    key = (kind, m.name if m else None) + tuple(a)
    if key in _MC: return _MC[key]
    bm = bmesh.new()
    if kind == 'cube': bmesh.ops.create_cube(bm, size=1.0)
    elif kind == 'cyl': bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=1.0, radius2=1.0, depth=1.0)
    elif kind == 'cone': bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=a[1], radius2=a[2], depth=a[3])
    elif kind == 'sph': bmesh.ops.create_uvsphere(bm, u_segments=a[0], v_segments=a[1], radius=1.0)
    elif kind == 'plane': bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    elif kind == 'torus':
        bm.free(); bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=a[0], major_segments=a[1], minor_segments=8)
        o = bpy.context.object; me = o.data; bpy.data.objects.remove(o)
        if m: me.materials.append(m)
        _MC[key] = me; return me
    me = bpy.data.meshes.new(kind); bm.to_mesh(me); bm.free()
    if kind in ('sph', 'cone', 'cyl'):
        for p in me.polygons: p.use_smooth = kind == 'sph' or abs(p.normal.z) < 0.5
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


def _latin(s): return all(ord(c) < 0x2E80 for c in s)
EN_K = 0.75


def text(s, loc, size, m, rot=(math.pi / 2, 0, 0), align='CENTER', ext=0.02, font=None):
    cu = bpy.data.curves.new('t', 'FONT'); cu.body = s
    cu.font = font or (FONT_EN if _latin(s) else FONT); cu.size = size * (EN_K if _latin(s) else 1.0); cu.extrude = ext
    cu.align_x = align; cu.align_y = 'CENTER'
    o = link(bpy.data.objects.new('txt', cu)); o.location = loc; o.rotation_euler = rot; o.data.materials.append(m); return o


def fit_w(o, w):
    bpy.context.view_layer.update()
    if w and o.dimensions.x > w: k = w / o.dimensions.x; o.scale = (k, k, k)
    return o


def frame_from(n, tdir=None):
    n = np.array(n, float); n /= np.linalg.norm(n)
    td = np.array(tdir, float) if tdir is not None else np.cross([0, 1, 0], n)
    td = td - n * np.dot(td, n)
    if np.linalg.norm(td) < 1e-6: td = np.cross([0, 1, 0], n)
    if np.linalg.norm(td) < 1e-6: td = np.array([1.0, 0, 0])
    td /= np.linalg.norm(td); dd = np.cross(n, td)
    return Matrix(((td[0], dd[0], n[0]), (td[1], dd[1], n[1]), (td[2], dd[2], n[2])))


def pulse(times, decay=5.0, length=16, amp=1.0):
    a = np.zeros(NF)
    for t in times:
        f0 = int(round(t * FPS)); m_ = np.arange(max(0, f0), min(NF, f0 + length))
        a[m_] = np.maximum(a[m_], amp * np.exp(-(m_ - f0) / decay))
    return a


def held(ons, offs, peak=1.0, sus=0.45, decay=4.0):
    """glow while a note is held: flash on the attack, then a steady glow until the release"""
    a = np.zeros(NF)
    for t0, t1 in zip(ons, offs):
        f0 = int(round(t0 * FPS)); f1 = int(round(t1 * FPS))
        for f in range(max(0, f0), min(NF, f1 + 6)):
            v = sus + (peak - sus) * math.exp(-(f - f0) / decay) if f <= f1 else sus * math.exp(-(f - f1) / 2.0)
            a[f] = max(a[f], v)
    return a


def smooth(Xa, k):
    Y = Xa.copy(); a = 1 - math.exp(-1 / FPS / k)
    for f in range(1, len(Xa)): Y[f] = Y[f - 1] + (Xa[f] - Y[f - 1]) * a
    for f in range(len(Xa) - 2, -1, -1): Y[f] = Y[f + 1] + (Y[f] - Y[f + 1]) * a
    return Y


def ease(u): u = min(1.0, max(0.0, u)); return u * u * (3 - 2 * u)


# ---------------- ball paths ----------------
GZ = np.array([0, 0, -G])
def seg_at(g, t):
    p = np.array(g['p']); u = t - g['t0']
    if g['k'] == 'rest': return p
    if g['k'] == 'fly': return p + np.array(g['v']) * u + 0.5 * GZ * u * u
    if g['k'] == 'line': return p + np.array(g['d']) * (g['v0'] * u + 0.5 * g['a'] * u * u)
    f = 0 if g['t1'] <= g['t0'] else min(1, max(0, (t - g['t0']) / (g['t1'] - g['t0'])))
    if g['k'] == 'ease': f = f * f * (3 - 2 * f)
    return p + (np.array(g['p1']) - p) * f


def positions(segs, ts):
    out = np.zeros((len(ts), 3)); j = 0
    for k, t in enumerate(ts):
        while j < len(segs) - 1 and t > segs[j]['t1'] + 1e-9: j += 1
        g = segs[j]
        if t < segs[0]['t0']: out[k] = seg_at(segs[0], segs[0]['t0'])
        elif t > g['t1']: out[k] = seg_at(g, g['t1'])
        else: out[k] = seg_at(g, t)
    return out


BALLS = LAY['balls']
POS = {n: positions(b['seg'], T) for n, b in BALLS.items()}

# ---------------- materials ----------------
M_SILVER = mat('silver', (0.92, 0.93, 0.96), 0.08, **{'Metallic': 1.0})
M_IVORYB = mat('ivoryball', (0.95, 0.9, 0.78), 0.18, **{'Coat Weight': 1.0})
M_GOLDB = mat('goldball', (1.0, 0.72, 0.25), 0.16, **{'Metallic': 1.0})
M_BRONZE = mat('bronze', (0.62, 0.36, 0.18), 0.25, **{'Metallic': 1.0})
M_RAIN = mat('rainball', (1.0, 0.75, 0.3), 0.2, **{'Metallic': 1.0})
M_BRASS = mat('brass', (0.85, 0.62, 0.3), 0.3, **{'Metallic': 1.0})
M_IRON = mat('iron', (0.08, 0.075, 0.07), 0.45, **{'Metallic': 0.8})
M_WOOD = mat('wood', (0.32, 0.16, 0.07), 0.55)
M_WOODD = mat('wooddark', (0.12, 0.055, 0.03), 0.5)
M_TIN = mat('tin', (0.5, 0.52, 0.56), 0.25, **{'Metallic': 1.0})
M_STONE = stone('stone', (0.17, 0.15, 0.14), (0.12, 0.11, 0.105), 2.0)
M_STONED = stone('stoned', (0.08, 0.075, 0.08), (0.06, 0.055, 0.06), 2.5)
M_FLOOR = stone('floor', (0.1, 0.09, 0.085), (0.05, 0.045, 0.045), 1.6, 0.22)
M_KEY = glowmat('key', gain=6.0, rough=0.25, coat=0.6)
M_PIPE = glowmat('pipe', gain=4.0, rough=0.22, metal=1.0, coat=0.0, glow=(1.0, 0.62, 0.25))
M_FX = fxmat('fx', 8.0)
M_OBJ = objcol('objcol', 0.5)
M_TXT = emis('txt', (1.0, 0.95, 0.85), 2.2); M_TXTG = emis('txtg', (1.0, 0.75, 0.3), 3.0); M_TXTJ = emis('txtj', (0.9, 0.85, 0.75), 1.6)
M_PLATE = mat('plate', (0.04, 0.03, 0.03), 0.4)
M_CANDLE = mat('candle', (0.95, 0.92, 0.82), 0.6)
M_FLAME = emis('flame', (1.0, 0.55, 0.15), 18.0)
M_GLASSWIN = emis('lancet', (0.04, 0.08, 0.4), 0.8)
M_RED = glowmat('lockglow', gain=6.0, rough=0.3, metal=0.6, glow=(1.0, 0.15, 0.05))
M_SPARK = fxmat('spark', 14.0)

# ---------------- balls ----------------
BOBJ = {}
for n, b in BALLS.items():
    kind = b['kind']; r = b['r']
    seg = (32, 16) if kind != 'rain' else (14, 7)
    m = {'silver': M_SILVER, 'ivory': M_IVORYB, 'gold': M_GOLDB, 'giant': M_BRONZE, 'rain': M_RAIN}[kind]
    o = obj(prim('sph', m, *seg), n); o.scale = (r, r, r)
    Pn = POS[n]
    if kind in ('ivory',):                 # hidden in the vault until they fall
        pass
    key_loc(o, Pn)
    if kind != 'rain':
        sp = np.zeros(NF); sx = np.zeros(NF)
        for f in range(1, NF):
            d = Pn[f] - Pn[f - 1]; sp[f] = sp[f - 1] + d[0] / r; sx[f] = sx[f - 1] - d[1] / r
        o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, sp, 1e-4); key_series(o, 'rotation_euler', 0, sx, 1e-4)
    BOBJ[n] = o
# giants get a band so the rolling reads
for n in ('GIANT1', 'GIANT2'):
    tr = obj(prim('torus', M_BRASS, 0.06, 32), 'band'); tr.parent = BOBJ[n]; tr.scale = (1.01, 1.01, 1.01); tr.rotation_euler = (math.pi / 2, 0, 0)

# ---------------- keys ----------------
def key_col(p):
    pc = p % 12
    if pc in (1, 3, 6, 8, 10): return (0.06, 0.05, 0.05)
    return (0.93, 0.88, 0.74)
HUES = {0: (1.0, 0.25, 0.2), 1: (1.0, 0.45, 0.1), 2: (1.0, 0.75, 0.2), 3: (0.8, 1.0, 0.2), 4: (0.3, 1.0, 0.4), 5: (0.2, 1.0, 0.8),
        6: (0.2, 0.7, 1.0), 7: (0.3, 0.4, 1.0), 8: (0.6, 0.3, 1.0), 9: (0.95, 0.3, 1.0), 10: (1.0, 0.3, 0.7), 11: (1.0, 0.35, 0.45)}
KOBJ = []
for ki, k in enumerate(KEYS):
    if k['kind'] == 'final': continue
    n = np.array(k['n']); Mx = frame_from(n, k.get('tdir'))
    c = np.array(k['c']) - n * k['th'] / 2
    p0 = k['ps'][-1]
    mkey = mat(f'key{ki}', (1, 1, 1), 0.25)
    o = obj(prim('cube', M_KEY), f'key_{ki}')
    o.matrix_world = Matrix.Translation(Vector(c)) @ Mx.to_4x4() @ Matrix.Diagonal((k['w'], k['d'], k['th'], 1))
    col = key_col(p0) if k['kind'] not in ('pedal',) else (0.55, 0.32, 0.15)
    hue = HUES[p0 % 12]
    # the key shows its own colour, the flash mixes in the pitch hue
    ons = k['t']; offs = k['toff']
    a = held(ons, offs, 1.0, 0.35 if k['kind'] in ('cradle', 'console', 'chordkey', 'pedal') else 0.0, 4.0)
    if k['kind'] not in ('cradle', 'console', 'chordkey', 'pedal'): a = np.maximum(a, pulse(ons, 4.0, 14))
    o.color = (*col, 0)
    # colour: blend towards the hue while lit (keyframed RGB follows the alpha)
    for ch in range(3):
        key_series(o, 'color', ch, col[ch] + (hue[ch] - col[ch]) * np.clip(a, 0, 1) * 0.8, 1e-3)
    key_series(o, 'color', 3, a, 1e-3)
    # dip on each hit
    base = Vector(c); nv = Vector(n)
    for t in ons:
        fh = int(round(t * FPS))
        for df, d in ((-1, 0), (1, -0.02), (4, 0.006), (8, 0)):
            o.location = base + nv * d; o.keyframe_insert('location', frame=max(0, fh + df))
    KOBJ.append((ki, o))

# hit sparks (ring pool)
NRING = 24
HITS = sorted([(t, np.array(k['c']), np.array(k['n']), k['ps'][-1]) for k in KEYS if k['kind'] != 'final' for t in k['t']], key=lambda h: h[0])
ring_loc = np.zeros((NRING, NF, 3)); ring_scl = np.full((NRING, NF), 1e-3); ring_a = np.zeros((NRING, NF)); ring_rot = np.zeros((NRING, NF, 3))
ring_c = [(1, 1, 1)] * NRING
for j, (t, c, n, p0) in enumerate(HITS):
    r_ = j % NRING; f0 = int(round(t * FPS))
    eul = Vector((0, 0, 1)).rotation_difference(Vector(n)).to_euler()
    for df in range(12):
        f = f0 + df
        if f >= NF: break
        u = df / 11; ring_loc[r_, f] = c + n * 0.01; ring_scl[r_, f] = 0.12 + 0.3 * (1 - (1 - u) ** 2); ring_a[r_, f] = (1 - u) ** 1.5
        ring_rot[r_, f] = eul
    ring_c[r_] = HUES[p0 % 12]
for r_ in range(NRING):
    o = obj(prim('torus', M_FX, 0.05, 24), 'ring'); o.color = (1.0, 0.85, 0.6, 0)
    for f in range(1, NF):
        if ring_a[r_, f] == 0: ring_loc[r_, f] = ring_loc[r_, f - 1]; ring_rot[r_, f] = ring_rot[r_, f - 1]
    key_loc(o, ring_loc[r_])
    for k in range(3): key_series(o, 'scale', k, ring_scl[r_]); key_series(o, 'rotation_euler', k, ring_rot[r_, :, k], 1e-3)
    key_series(o, 'color', 3, ring_a[r_], 1e-3)

# supports: one dark iron stringer under each lane of keys (per region), a few posts down to the floor
from collections import defaultdict
LANES = defaultdict(list)
for ki, k in enumerate(KEYS):
    if k['region'] in ('stairs1', 'stairs2', 'cascade') and k['kind'] == 'bar':
        LANES[(k['region'], k['ball'])].append(np.array(k['c']))
for (reg, bn), cs in LANES.items():
    cs = sorted(cs, key=lambda c: c[0])
    if reg == 'stairs1': cs = [c for c in cs if c[0] > 1.0]
    pts = [c + np.array([0, 0.0, -0.1]) for c in cs]
    for a_, b_ in zip(pts, pts[1:]): rod(a_, b_, 0.022, M_IRON, 6)
    for c in pts[::5]:
        rod(c, (c[0], c[1], (0.0 if c[0] < 2.3 else -5.6)), 0.018, M_IRON, 6)
for ki, k in enumerate(KEYS):
    if k['region'] in ('solo', 'bar10') and k['kind'] in ('bar', 'chordkey', 'cradle'):
        c = np.array(k['c']); rod((c[0], c[1] + 0.12, c[2] - 0.08), (c[0], c[1] + 0.12, -5.6), 0.014, M_IRON, 6)

# ---------------- the cathedral ----------------
X0, X1 = -6.0, 34.0
# floor of the nave (low) and the raised chancel / portal block
box(((X0 + X1) / 2, 2.0, -5.75), (X1 - X0, 16, 0.3), M_FLOOR, name='nave')
box((-1.0, -0.6, -2.8), (8.6, 4.2, 5.6), M_STONE, name='chancel')
box((2.3, -0.6, -2.8), (0.3, 4.3, 5.7), M_STONED, name='chancel_edge')
# back wall (nave) and the portal wall around the door
box(((X0 + X1) / 2, 3.6, 8.0), (X1 - X0, 0.4, 28.0), M_STONE, name='backwall')
D = P['door']
PW_X0, PW_X1 = -4.4, 2.35
box(((PW_X0 + D['x0']) / 2, D['y'] + 0.25, 7.0), (D['x0'] - PW_X0, 0.5, 14.0), M_STONE, name='portalL')
box(((D['x1'] + PW_X1) / 2, D['y'] + 0.25, 7.0), (PW_X1 - D['x1'], 0.5, 14.0), M_STONE, name='portalR')
box((0, D['y'] + 0.25, (D['z1'] + 14.0) / 2), (D['x1'] - D['x0'], 0.5, 14.0 - D['z1']), M_STONE, name='portalT')
# pointed arch moulding around the door
for side in (-1, 1):
    rod((side * (D['x1'] + 0.12), D['y'] - 0.05, 0), (side * (D['x1'] + 0.12), D['y'] - 0.05, D['z1']), 0.12, M_STONED, 12)
    pts = [(side * (D['x1'] + 0.12) * math.cos(a_) , D['z1'] + 1.5 * math.sin(a_)) for a_ in np.linspace(0, math.pi / 2, 8)]
    for (xa, za), (xb, zb) in zip(pts, pts[1:]): rod((xa, D['y'] - 0.05, za), (xb, D['y'] - 0.05, zb), 0.12, M_STONED, 10)
# rose window above the door (behind the three galleries)
RW_C = (0.0, D['y'] - 0.01, 8.0); RW_R = 2.0
M_ROSE = []
rose_cols = [(0.85, 0.03, 0.06), (0.03, 0.12, 0.9), (0.95, 0.55, 0.03), (0.03, 0.55, 0.2), (0.45, 0.05, 0.75)]
for ring_i, (r0, r1, nseg) in enumerate(((0.0, 0.55, 8), (0.6, 1.3, 16), (1.35, 1.9, 24))):
    for s_ in range(nseg):
        a0 = 2 * math.pi * s_ / nseg + 0.02; a1 = 2 * math.pi * (s_ + 1) / nseg - 0.02
        bm = bmesh.new(); vs = []
        for rr, aa in ((r0 + 0.03, a0), (r1, a0), (r1, a1), (r0 + 0.03, a1)):
            vs.append(bm.verts.new((rr * math.cos(aa), 0, rr * math.sin(aa))))
        bm.faces.new(vs); me = bpy.data.meshes.new('rose'); bm.to_mesh(me); bm.free()
        col = rose_cols[(s_ + ring_i * 2) % len(rose_cols)]
        mm = emis(f'rose{ring_i}_{s_ % 5}', col, 1.7) if f'rose{ring_i}_{s_ % 5}' not in bpy.data.materials else bpy.data.materials[f'rose{ring_i}_{s_ % 5}']
        o = link(bpy.data.objects.new('rosepane', me)); o.location = RW_C; o.data.materials.append(mm); M_ROSE.append(mm)
for k_ in range(24):
    aa = 2 * math.pi * k_ / 24; r0_ = 0.58 if k_ % 3 else 0.0
    rod((RW_C[0] + r0_ * math.cos(aa), RW_C[1] - 0.02, RW_C[2] + r0_ * math.sin(aa)), (RW_C[0] + 1.95 * math.cos(aa), RW_C[1] - 0.02, RW_C[2] + 1.95 * math.sin(aa)), 0.025, M_STONED, 6)
for rr in (0.58, 1.32, 1.95):
    t_ = obj(prim('torus', M_STONED, 0.04 / rr, 48), 'tracery'); t_.location = (RW_C[0], RW_C[1] - 0.02, RW_C[2]); t_.scale = (rr, rr, rr); t_.rotation_euler = (math.pi / 2, 0, 0)
# tall lancet windows along the nave
for xw in np.arange(5.0, X1, 4.5):
    box((xw, 3.38, 4.5), (0.9, 0.02, 6.0), M_GLASSWIN, name='lancet')
    rod((xw, 3.36, 1.5), (xw, 3.36, 7.5), 0.03, M_STONED, 6)
    for zz in (2.5, 4.5, 6.5): rod((xw - 0.45, 3.36, zz), (xw + 0.45, 3.36, zz), 0.025, M_STONED, 6)
# pillars along the nave (in front of the back wall)
for xp in np.arange(2.8, X1, 4.5):
    rod((xp, 3.0, -5.6), (xp, 3.0, 14.0), 0.32, M_STONE, 16)
    rod((xp, 3.0, -5.6), (xp, 3.0, -5.2), 0.42, M_STONED, 16)

# ---------------- organ pipes (light up with their pitch) ----------------
ACTIVE = {}
for r in LEAD:
    for p in r[2]: ACTIVE.setdefault(p, []).append((r[0], r[0] + r[1]))
def pipe_flat(x0, x1, zb, y, pitches, scale=1.0, r0=0.09):
    """one 'flat' of façade pipes: tallest in the middle, mouths at zb"""
    ps = sorted(pitches); n = len(ps)
    order = sorted(range(n), key=lambda i: abs(i - (n - 1) / 2))
    xs = np.linspace(x0, x1, n)
    by_len = sorted(ps)                                   # lowest pitch = longest pipe -> centre
    out = []
    for slot, p in zip(order, by_len):
        L_ = scale * 4.2 * 2 ** (-(p - 38) / 24) + 0.6
        rr = r0 * (1.0 + 0.6 * 2 ** (-(p - 38) / 24))
        x = xs[slot]
        o = rod((x, y, zb + 0.35), (x, y, zb + 0.35 + L_), rr, M_PIPE, 16)
        foot = obj(prim('cone', M_PIPE, 16, 1.0, 0.25, 1.0), 'foot'); foot.location = (x, y, zb + 0.175); foot.scale = (rr, rr, 0.35)
        mouth = box((x, y - rr * 0.8, zb + 0.5), (rr * 1.1, 0.02, rr * 0.9), M_PLATE, name='mouth')
        col = (0.33, 0.34, 0.38)
        a = held([q[0] for q in ACTIVE.get(p, [])], [q[1] for q in ACTIVE.get(p, [])], 1.0, 0.55, 5.0)
        for ob in (o, foot):
            ob.color = (*col, 0); key_series(ob, 'color', 3, a, 1e-3)
        out.append(o)
    return out
USED = sorted(ACTIVE)
door_p = sorted(set(p for r in LEAD if r[3] in ('call1', 'call2', 'call3', 'ped', 'chord', 'dmaj', 'inner') for p in r[2]))
pipe_flat(-4.1, -2.3, 9.6, D['y'] - 0.08, door_p[::2], 0.75)
pipe_flat(2.25, 4.1, 9.6, D['y'] - 0.08, door_p[1::2], 0.75)
fast_p = sorted(set(p for r in LEAD if r[3] == 'trio' for p in r[2]))
pipe_flat(3.6, 10.0, 3.6, 3.25, fast_p[::2], 1.0)
pipe_flat(10.6, 17.0, 3.6, 3.25, fast_p[1::2], 1.0)
end_p = sorted(set(p for r in LEAD if r[3] in ('chord2', 'solo', 'final', 'ped') for p in r[2]))
pipe_flat(17.6, 22.5, 1.2, 3.25, end_p, 1.05)
pipe_flat(28.4, 32.0, -2.0, 3.25, sorted(set(LEAD[-1][2])), 1.3, 0.12)
assert set(door_p) | set(fast_p) | set(end_p) >= set(USED)

# ---------------- the three galleries (calls) ----------------
M_LOCKB = mat('lockbody', (0.5, 0.36, 0.15), 0.3, **{'Metallic': 1.0})
LOCKS = {1: (1.62, 4.25), 2: (-1.62, 3.05), 3: (1.62, 1.85)}
for c, v in P['calls'].items():
    c = int(c)
    ps = np.array(v['p_start']); ce = np.array(v['chute_end']); dch = np.array(v['dch'])
    # gallery ledge + inclined chute (two rails) + gate
    for dy in (-0.11, 0.11):
        rod(ps - dch * 0.2 + [0, dy, -0.13], ce + [0, dy, -0.13], 0.025, M_BRASS, 8)
    box(ps - dch * 0.45 + [0, 0, -0.2], (0.6, 0.5, 0.08), M_WOODD, name='gallery')
    rod(ps - dch * 0.45 + [0, 0.3, -0.25], ps - dch * 0.45 + [0, 0.3, -1.2], 0.03, M_BRASS, 6)
    # shaft walls (frames around the wall keys), cradle stand, ramp rail + flick hammer
    w1 = np.array(v['wall1']); w2 = np.array(v['wall2']); cr = np.array(v['cradle'])
    for w in (w1, w2):
        sx = np.sign(w[0] - (w1[0] + w2[0]) / 2)
        box((w[0] + sx * 0.04, 0.0, (w[2] + cr[2]) / 2 + 0.25), (0.03, 0.46, abs(w[2] - cr[2]) + 1.0), M_WOODD, name='shaftwall')
    p0 = np.array(v['ramp_p0']); dr = np.array(v['ramp_dir']); nr = np.array(v['ramp_n'])
    a_ = p0 - R * nr - nr * 0.07 - dr * 0.25; b_ = p0 + dr * v['ramp_len'] - R * nr - nr * 0.07
    for dy in (-0.23, 0.23): rod(a_ + [0, dy, 0], b_ + [0, dy, 0], 0.022, M_BRASS, 8)
    rod((a_ + b_) / 2 + [0, 0.25, 0], (a_ + b_) / 2 + [0, 0.25, -1.2], 0.025, M_BRASS, 6)
    # flick hammer: pivot behind the cradle, swings through the marble's position at t_rel
    hp = link(bpy.data.objects.new('hammer', None)); hp.location = tuple(p0 - dr * 0.42 + [0, 0.0, 0.15])
    hh = rod((0, 0, 0), tuple(dr * 0.22 - [0, 0, 0.14]), 0.02, M_IRON, 6); hh.parent = hp
    hd = sphere(tuple(dr * 0.24 - [0, 0, 0.15]), 0.06, M_BRASS); hd.parent = hp
    ang = np.zeros(NF)
    for f in range(NF):
        u = (T[f] - (v['t_rel'] - 0.12)) / 0.12
        if 0 <= u <= 1: ang[f] = -0.6 * math.sin(math.pi / 2 * u)
        elif u > 1: ang[f] = -0.6 * max(0, 1 - (u - 1) * 0.25)
    hp.rotation_mode = 'XYZ'; key_series(hp, 'rotation_euler', 1, ang * v['s_r'] * -1, 1e-4)
    # lock lever: a plate on an arm, tips under the marble; a rod runs to its padlock on the door frame
    lv = np.array(v['lever']); ldn = np.array(v['lever_dn'])
    piv = link(bpy.data.objects.new('lever', None)); s_side = 1 if lv[0] > 0 else -1
    piv.location = (lv[0] + s_side * 0.45, 0.0, lv[2] - R - 0.04)
    plate = box((-s_side * 0.45, 0, 0), (0.42, 0.42, 0.05), M_BRASS, name='leverplate'); plate.parent = piv
    arm = rod((0, 0, 0), (-s_side * 0.45, 0, 0), 0.025, M_IRON, 6); arm.parent = piv
    ang = np.array([0.0 if t < v['t_lock'] else -0.32 * ease((t - v['t_lock']) / 0.3) for t in T]) * -s_side
    piv.rotation_mode = 'XYZ'; key_series(piv, 'rotation_euler', 1, ang, 1e-4)
    box(piv.location + Vector((s_side * 0.05, 0.25, -0.1)), (0.1, 0.1, 0.4), M_IRON, name='leverpost')
    lx, lz = LOCKS[c]
    rod((piv.location.x, 0.2, piv.location.z), (lx + s_side * 0.18, D['y'] - 0.22, lz + 0.18), 0.018, M_IRON, 6)
    # padlock: glows red, pops open and drops away when the lever is pressed
    t_un = v['t_lock'] + 0.25
    lk = link(bpy.data.objects.new('lock', None)); lk.location = (lx, D['y'] - 0.22, lz)
    body = box((0, 0, 0), (0.42, 0.16, 0.34), M_LOCKB, name='lockbody'); body.parent = lk
    sh = obj(prim('torus', M_RED, 0.12, 24), 'shackle'); sh.parent = lk; sh.location = (0, 0, 0.2); sh.scale = (0.14, 0.14, 0.18); sh.rotation_euler = (math.pi / 2, 0, 0)
    sh.color = (0.6, 0.05, 0.03, 1)
    glo = np.array([0.6 + 0.4 * math.sin(6 * t) if t < t_un else max(0, 1 - (t - t_un) * 2) for t in T])
    key_series(sh, 'color', 3, glo, 1e-3)
    lt = text(f'LOCK {c}', (0, -0.09, 0.0), 0.12, M_TXT); lt.parent = lk
    zz = np.array([0.0 if t < t_un + 0.1 else -0.5 * G * (t - t_un - 0.1) ** 2 for t in T]); zz = np.maximum(zz, -(lz - 0.15))
    rotl = np.array([0.0 if t < t_un + 0.1 else min(1.2, (t - t_un - 0.1) * 3) for t in T])
    lk.rotation_mode = 'XYZ'; key_series(lk, 'location', 2, lz + zz, 1e-4); key_series(lk, 'rotation_euler', 1, rotl * s_side, 1e-4)
    # chain from the lock across the door seam
    for k_ in range(6):
        xx = lx - s_side * (0.25 + 0.27 * k_)
        ch = obj(prim('torus', M_IRON, 0.25, 12), 'chain'); ch.location = (xx, D['y'] - 0.2, lz); ch.scale = (0.11, 0.11, 0.11)
        ch.rotation_euler = (math.pi / 2 if k_ % 2 else 0, 0, 0)
        key_series(ch, 'location', 2, lz + np.array([0.0 if t < t_un + 0.1 else -min(lz + 0.5, 0.5 * G * (t - t_un - 0.1) ** 2) for t in T]), 1e-4)
    sp_ = sphere((lx, D['y'] - 0.35, lz), 0.4, M_SPARK); sp_.color = (0.4, 1.0, 0.5, 0)
    key_series(sp_, 'color', 3, pulse([t_un], 3.0, 12), 1e-3)
    for k in range(3): key_series(sp_, 'scale', k, 0.2 + 1.2 * pulse([t_un], 6.0, 12), 1e-3)

# ---------------- GIANT I: shelf, pedal, socket; the great doors ----------------
g1 = P['giant1']; sh = np.array(g1['shelf'])
shp = link(bpy.data.objects.new('shelf', None)); shp.location = (sh[0] - 0.4, sh[1], sh[2] - 0.32 - 0.04)
pl = box((0.4, 0, 0), (0.85, 0.75, 0.07), M_WOOD, name='shelfplate'); pl.parent = shp
key_series(shp, 'rotation_euler', 1, np.array([0.0 if t < g1['t_rel'] else min(1.3, 9 * (t - g1['t_rel']) ** 2) for t in T]), 1e-4)
rod((sh[0] - 0.4, sh[1], sh[2] - 0.36), (sh[0] - 0.4, sh[1], 0.0), 0.04, M_IRON, 8)
rod((sh[0] - 0.4, sh[1], sh[2] - 0.36), (P['calls']['3']['lever'][0] + 0.45, 0.0, P['calls']['3']['lever'][2] - R - 0.04), 0.018, M_IRON, 6)
pd = np.array(g1['pedal'])
pedo = [o for ki, o in KOBJ if KEYS[ki]['kind'] == 'pedal' and KEYS[ki]['region'] == 'door'][0]
box((pd[0], pd[1], pd[2] - 0.2), (1.0, 1.0, 0.18), M_STONED, name='pedalbase')
text('PEDAL D', (pd[0], pd[1] - 0.52, pd[2] - 0.1), 0.13, M_TXTG)
so = np.array(g1['socket']); box((so[0], so[1], so[2] - 0.32 - 0.05), (0.6, 0.7, 0.1), M_IRON, name='socket')
# chain from the pedal up to the door winch
rod((pd[0] - 0.4, pd[1] + 0.3, pd[2]), (D['x1'] + 0.35, D['y'] - 0.2, D['z1'] + 0.4), 0.025, M_IRON, 6)
M_DOOR = mat('door', (0.3, 0.14, 0.06), 0.55)
dw = (D['x1'] - D['x0']) / 2
DO = []
for side in (-1, 1):
    lf = link(bpy.data.objects.new('leaf', None)); lf.location = (side * dw / 2, D['y'] + 0.12, 0)
    panel = box((0, 0, D['z1'] / 2), (dw - 0.02, 0.14, D['z1']), M_DOOR, name='doorleaf'); panel.parent = lf
    for zz in (0.8, 2.4, 4.0):
        bnd = box((0, -0.08, zz), (dw - 0.06, 0.03, 0.14), M_IRON, name='doorband'); bnd.parent = lf
    rg = obj(prim('torus', M_BRASS, 0.12, 24), 'ring'); rg.parent = lf; rg.location = (-side * (dw / 2 - 0.25), -0.1, 2.3); rg.scale = (0.12, 0.12, 0.12)
    rg.rotation_euler = (math.pi / 2, 0, 0)
    xo = np.array([side * dw / 2 + side * (dw + 0.05) * ease((t - D['t_open0']) / (D['t_open1'] - D['t_open0'])) for t in T])
    key_series(lf, 'location', 0, xo, 1e-4)
    DO.append(lf)
# inside the door: warm sanctuary, the trio's ledge and chutes
box((0, 3.3, 2.5), (3.0, 0.1, 5.0), emis('inner', (1.0, 0.45, 0.12), 2.0), name='innerwall')
box((0, 2.4, -0.05), (3.0, 1.8, 0.1), M_FLOOR, name='innerfloor')
for i, v in P['trio_start'].items():
    st = np.array(v['rest']); ce = np.array(v['chute_end']); dc = np.array(v['dir'])
    for dx in (-0.1, 0.1): rod(st - dc * 0.2 + [dx, 0, -0.13], ce + [dx, 0, -0.13], 0.02, M_BRASS, 8)
    rod(st + [0, 0.1, -0.15], (st[0], st[1] + 0.1, 0.0), 0.025, M_BRASS, 6)
    rod(ce + [0, 0.05, -0.15], (ce[0], ce[1] + 0.05, 0.0), 0.02, M_BRASS, 6)

# ---------------- the console (chord keys) + vault hatches ----------------
CO = P['console']
box((0.25, CO['y'] - 0.05, CO['z'] / 2 - 0.08), (3.1, 0.75, CO['z'] - 0.2), M_WOODD, name='consolebody')
box((0.25, CO['y'] - 0.38, CO['z'] - 0.02), (3.15, 0.06, 0.22), M_BRASS, name='consolerail')
box((1.86, CO['y'] - 0.05, (CO['inner_z']) / 2 - 0.05), (0.75, 0.75, CO['inner_z'] - 0.15), M_WOODD, name='innerbody')
text('CONSOLE', (0.25, CO['y'] - 0.42, CO['z'] - 0.35), 0.12, M_TXTG)
# hatches in the dark vault: a beam with a trapdoor above every falling ivory marble
for b_ in CO['balls']:
    p0 = POS[b_['name']][0]
    box((p0[0], p0[1], p0[2] + R + 0.06), (0.32, 0.32, 0.08), M_IRON, name='hatch')
box((0.3, CO['y'], CO['vault'] + 0.9), (4.4, 0.5, 0.5), M_STONED, name='vaultbeam')

# ---------------- stairs signage, lift, landing ----------------
def sign(en, jp, loc, size, w=None, m=M_TXT, rot=(math.pi / 2, 0, 0)):
    o = fit_w(text(en, loc, size, m, rot=rot), w)
    out = [o]
    if jp: out.append(fit_w(text(jp, (loc[0], loc[1], loc[2] - size * 0.95), size * 0.5, M_TXTJ, rot=rot), w))
    return out
LF = P['lift']
cups = {int(i): np.array(c) for i, c in LF['cups'].items()}
rack = link(bpy.data.objects.new('rack', None)); rack.location = (0, 0, 0)
for i, c in cups.items():
    cp = obj(prim('cyl', M_BRASS, 20), 'cup'); cp.parent = rack; cp.location = (c[0], c[1], c[2] - 0.03); cp.scale = (0.2, 0.2, 0.06)
    rod((c[0], c[1], c[2] - 0.06), (c[0], c[1] + 0.25, c[2] - 0.06), 0.02, M_IRON, 6).parent = rack
xs_ = [c[0] for c in cups.values()]
bar_ = box((np.mean(xs_), 0.68 + 0.25, min(c[2] for c in cups.values()) - 0.12), (max(xs_) - min(xs_) + 0.5, 1.0, 0.06), M_IRON, name='rackbar'); bar_.parent = rack
key_series(rack, 'location', 2, np.array([LF['h'] * ease((t - LF['t0']) / (LF['t1'] - LF['t0'])) for t in T]), 1e-4)
for xx in (min(xs_) - 0.3, max(xs_) + 0.3):
    rod((xx, 1.45, min(c[2] for c in cups.values()) - 1.0), (xx, 1.45, max(c[2] for c in cups.values()) + LF['h'] + 0.6), 0.05, M_IRON, 8)
sign('LIFT', '', (np.mean(xs_), 1.5, max(c[2] for c in cups.values()) + LF['h'] + 0.9), 0.25, m=M_TXTG)
LD = P['landing']
for i, c in LD['cradles'].items():
    c = np.array(c); box((c[0], c[1], c[2] - 0.03), (0.3, 0.32, 0.06), M_WOODD, name='landcradle')
    rod((c[0], c[1] + 0.1, c[2] - 0.06), (c[0], c[1] + 0.1, -5.6), 0.02, M_BRASS, 6)
    gp = link(bpy.data.objects.new('gate', None)); gp.location = (c[0] + 0.2, c[1], c[2] - 0.02)
    gb = box((0, 0, 0.17), (0.04, 0.34, 0.34), M_IRON, name='gatebar'); gb.parent = gp
    tr_ = LD['t_rel'][str(i)]
    key_series(gp, 'rotation_euler', 1, np.array([0.0 if t < tr_ - 0.08 else min(1.5, 20 * (t - tr_ + 0.08)) for t in T]), 1e-4)
cz = np.mean([c[2] for c in LD['cradles'].values()]); cx = np.mean([c[0] for c in LD['cradles'].values()])
sign('FERMATA... WAIT', '', (cx, 1.4, cz + 0.85), 0.16, 1.8, M_TXTG)
# signs along the stairs (early search keywords)
s1 = [np.array(k['c']) for k in KEYS if k['region'] == 'stairs1']
sign('PRESTISSIMO!', 'プレスティッシモ', (np.mean([c[0] for c in s1]) + 0.3, 1.4, max(c[2] for c in s1) + 1.25), 0.32, 2.8)
s2 = [np.array(k['c']) for k in KEYS if k['region'] == 'stairs2']
sign('ONE OCTAVE HIGHER', '1オクターブ上', (np.mean([c[0] for c in s2]), 1.4, max(c[2] for c in s2) + 1.1), 0.22, 2.8)
s3 = [np.array(k['c']) for k in KEYS if k['region'] == 'cascade']
sign('THE CASCADE', '下降音階', (np.mean([c[0] for c in s3]) - 1.0, 1.4, max(c[2] for c in s3) + 0.5), 0.26, 2.8)

# ---------------- bar 10-12: catch cradles, Giant II, spout, solo, golden pipes, final keys ----------------
for i, v in P['catch'].items():
    c = np.array(v['cradle']); box((c[0], c[1], c[2] - 0.03), (0.3, 0.32, 0.06), M_WOODD, name='catch')
    rod((c[0], c[1] + 0.1, c[2] - 0.06), (c[0], c[1] + 0.1, -5.6), 0.02, M_BRASS, 6)
FI = P['final']; ZF = FI['zf']
box((25.6, 2.25, ZF - 1.4), (8.0, 1.3, 0.2), M_IRON, name='troughfloor')
box((25.6, 1.55, ZF - 0.9), (8.0, 0.06, 1.0), M_WOODD, name='troughfront')
box((25.6, 2.95, ZF - 0.4), (8.0, 0.06, 2.0), M_WOODD, name='troughback')
g2 = P['giant2']; st = np.array(g2['start']); ed = np.array(g2['edge'])
nrm = np.array([math.sin(math.radians(20)), 0, math.cos(math.radians(20))])
a_ = st - nrm * (0.32 + 0.04) - np.array([math.cos(math.radians(20)), 0, -math.sin(math.radians(20))]) * 0.4; b_ = ed - nrm * (0.32 + 0.04)
ledge = rod(a_, b_, 0.05, M_IRON, 8)
for dy in (-0.25, 0.25): rod(a_ + [0, dy, 0], b_ + [0, dy, 0], 0.03, M_BRASS, 8)
rod(a_, (a_[0], a_[1], -5.6), 0.04, M_IRON, 6)
sign('GIANT II', '', (st[0] - 0.2, st[1] - 0.6, st[2] + 0.6), 0.16, m=M_TXTG)
bk = np.array(g2['bucket'])
bkt = link(bpy.data.objects.new('bucket', None)); bkt.location = tuple(bk)
bo = obj(prim('cyl', M_IRON, 24), 'bucketbody'); bo.parent = bkt; bo.location = (0, 0, 0.2); bo.scale = (0.42, 0.42, 0.4)
key_series(bkt, 'location', 2, np.array([bk[2] - g2['drop'] * ease((t - g2['t_cw0']) / (g2['t_cw1'] - g2['t_cw0'])) for t in T]), 1e-4)
rod((bk[0], bk[1], bk[2] + 0.4), (bk[0], bk[1], FI['pipes']['z1'] + 0.6), 0.015, M_IRON, 6)
# stop pin of the pedal (drops at t_off)
pin = rod((bk[0] - 0.25, 0.95, ZF - 0.4), (bk[0] + 0.25, 0.95, ZF - 0.4), 0.04, M_BRASS, 8)
key_series(pin, 'location', 2, np.array([ZF - 0.4 - (0.0 if t < g2['t_off'] - 0.05 else min(0.4, (t - g2['t_off'] + 0.05) * 4)) for t in T]), 1e-4)
# spout for the first drop (from the first golden pipe)
spt = np.array(P['solo']['spout']); px0 = FI['pipes']['xs'][0]
rod((px0, FI['pipes']['y'], spt[2] + 0.35), (spt[0], spt[1], spt[2] + 0.2), 0.12, M_BRASS, 16)
rod((spt[0], spt[1], spt[2] + 0.2), (spt[0], spt[1], spt[2] - 0.12), 0.13, M_BRASS, 16)
sign('THE FIRST DROP', '', (spt[0] - 0.3, spt[1] - 0.2, spt[2] + 0.75), 0.15, m=M_TXTG)
# solo ramp rails, flick hammer, shaft frame
so_ = P['solo']; p0 = np.array(so_['ramp_p0']); dr = np.array(so_['ramp_dir']); nr = np.array(so_['ramp_n'])
a_ = p0 - R * nr - nr * 0.07 - dr * 0.25; b_ = p0 + dr * so_['ramp_len'] - R * nr - nr * 0.07
for dy in (-0.22, 0.22): rod(a_ + [0, dy, 0], b_ + [0, dy, 0], 0.02, M_BRASS, 8)
hp = link(bpy.data.objects.new('hammer2', None)); hp.location = tuple(p0 - dr * 0.42 + [0, 0, 0.15])
hh = rod((0, 0, 0), tuple(dr * 0.22 - [0, 0, 0.14]), 0.02, M_IRON, 6); hh.parent = hp
hd = sphere(tuple(dr * 0.24 - [0, 0, 0.15]), 0.06, M_BRASS); hd.parent = hp
key_series(hp, 'rotation_euler', 1, np.array([0.0 if t < so_['t_kick'] - 0.12 else 0.6 * math.sin(math.pi / 2 * min(1, (t - so_['t_kick'] + 0.12) / 0.12)) for t in T]), 1e-4)
shx = [s_[0] for s_ in so_['shaft']]; shz = [s_[2] for s_ in so_['shaft']]
for xx in (min(shx) - 0.05, max(shx) + 0.05):
    box((xx, Y_S := so_['spout'][1], (min(shz) + max(shz)) / 2), (0.03, 0.44, max(shz) - min(shz) + 0.6), M_WOODD, name='trillshaft')
sign('TRILL', '', ((min(shx) + max(shx)) / 2, Y_S - 0.3, max(shz) + 0.55), 0.14, m=M_TXTG)
# final keys: flat front lip + slope (glow with every strike)
tilt = FI['tilt']
FKO = []
for ki, k in enumerate(KEYS):
    if k['kind'] != 'final': continue
    x = k['c'][0]; w = k['w']; p0_ = k['ps'][0]
    col = key_col(p0_)
    lip = box((x, (FI['y0'] + FI['yl']) / 2, ZF - 0.07), (w, FI['yl'] - FI['y0'], 0.14), M_KEY, name='finallip')
    ylen = (FI['y1'] - FI['yl']) / math.cos(tilt)
    slope = obj(prim('cube', M_KEY), 'finalslope')
    ax_y = Vector((0, math.cos(tilt), -math.sin(tilt))); n_ = Vector((0, math.sin(tilt), math.cos(tilt)))
    c_ = Vector((x, FI['yl'], ZF)) + ax_y * ylen / 2 - n_ * 0.07
    slope.matrix_world = Matrix.Translation(c_) @ Matrix(((1, 0, 0), tuple(ax_y), tuple(n_))).transposed().to_4x4() @ Matrix.Diagonal((w, ylen, 0.14, 1))
    a = np.maximum(held(k['t'][:1], k['toff'][:1], 1.0, 0.5, 6.0), 0.0)
    if p0_ == 38: a = np.maximum(held(k['t'][:1], k['toff'][:1], 1.0, 0.45, 5.0), held([FI['t_final']], [FI['t_final'] + EVJ['hold']], 1.0, 0.6, 8.0))
    else: a = held([k['t'][0]], [FI['t_final'] + EVJ['hold']], 1.0, 0.6, 8.0)
    a = np.clip(a + 0.6 * pulse(k['t'], 2.0, 6), 0, 1.6)
    hue = HUES[p0_ % 12]
    for o in (lip, slope):
        o.color = (*col, 0)
        for ch in range(3): key_series(o, 'color', ch, col[ch] + (hue[ch] - col[ch]) * np.clip(a, 0, 1) * 0.7, 1e-3)
        key_series(o, 'color', 3, a, 1e-3)
    box((x, 0.2, ZF - 0.6), (w * 0.8, 1.0, 0.9), M_WOODD, name='finalstand')
    lab = NM[p0_ % 12] + str(p0_ // 12 - 1)
    text(lab, (x, FI['y0'] - 0.02, ZF - 0.07), 0.09, M_TXTG)
    FKO.append((p0_, lip))
box((25.4, -0.7, ZF - 0.9), (5.4, 0.1, 0.8), M_WOODD, name='pedalboardfront')
text('FINAL CHORD · D MINOR', (25.4, -0.76, ZF - 0.75), 0.13, M_TXTG)
# golden pipes: back half shell + rings + feeder wheel at the mouth
PI = FI['pipes']
M_PIPEG = mat('pipegold', (1.0, 0.7, 0.28), 0.2, **{'Metallic': 1.0})
for x in PI['xs']:
    bm = bmesh.new(); vs_b, vs_t = [], []
    for a_i in range(13):
        aa = math.pi * a_i / 12
        vs_b.append(bm.verts.new((x + PI['r'] * math.cos(aa), PI['y'] + PI['r'] * math.sin(aa), PI['z0'])))
        vs_t.append(bm.verts.new((x + PI['r'] * math.cos(aa), PI['y'] + PI['r'] * math.sin(aa), PI['z1'])))
    for a_i in range(12): bm.faces.new((vs_b[a_i], vs_b[a_i + 1], vs_t[a_i + 1], vs_t[a_i]))
    me = bpy.data.meshes.new('pipeshell'); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new('pipeshell', me)); o.data.materials.append(M_PIPEG)
    sol_ = o.modifiers.new('s', 'SOLIDIFY'); sol_.thickness = 0.015
    for zz in np.arange(PI['z0'] + 0.3, PI['z1'], 0.9):
        rg = obj(prim('torus', M_PIPEG, 0.12, 24), 'pring'); rg.location = (x, PI['y'], zz); rg.scale = (PI['r'] + 0.012, PI['r'] + 0.012, PI['r'] + 0.012)
    wh = link(bpy.data.objects.new('feeder', None)); wh.location = (x, PI['y'] - 0.02, PI['z0'] - 0.02)
    for sp_i in range(4):
        sp_o = box((0, 0, 0), (0.24, 0.02, 0.025), M_IRON, name='spoke'); sp_o.parent = wh; sp_o.rotation_euler = (0, sp_i * math.pi / 4, 0)
    spin = np.array([0.0 if t < PI['t_gate'] else (t - PI['t_gate']) * PI['u'] / (PI['r']) for t in T])
    wh.rotation_mode = 'XYZ'; key_series(wh, 'rotation_euler', 1, spin, 1e-4)
# crown of the golden pipes + plaque
box((np.mean(PI['xs']), PI['y'], PI['z1'] + 0.12), (PI['xs'][-1] - PI['xs'][0] + 0.5, 0.4, 0.24), M_PIPEG, name='crown')
plq = text('BWV 565', (np.mean(PI['xs']), PI['y'] - 0.25, PI['z1'] + 0.75), 0.75, M_TXTG, font=FONT_SERIF)
plq2 = text('THE GOLDEN PIPES', (np.mean(PI['xs']), PI['y'] - 0.25, PI['z1'] + 0.3), 0.2, M_TXT)
# final burst: sparks from every final key at the end of the flood
M_BURST = fxmat('burst', 12.0)
for j in range(70):
    o = sphere((0, 0, 0), 0.035, M_BURST, (8, 4)); o.color = (1.0, 0.75 + 0.2 * rng.random(), 0.3, 0)
    x0_ = rng.uniform(PI['xs'][0], PI['xs'][-1]); t0 = FI['t_last'] + rng.uniform(-0.1, 0.4)
    v0 = np.array([rng.uniform(-1.5, 1.5), rng.uniform(-1.5, -0.3), rng.uniform(2.5, 6.0)])
    Pp = np.zeros((NF, 3)); al = np.zeros(NF)
    for f in range(NF):
        u = T[f] - t0
        if u < 0: Pp[f] = (x0_, 0.0, ZF + 0.2); continue
        Pp[f] = (x0_ + v0[0] * u, v0[1] * u, ZF + 0.2 + v0[2] * u - 0.5 * 3.0 * u * u); al[f] = max(0, 1 - u / 2.2)
    key_loc(o, Pp); key_series(o, 'color', 3, al, 1e-3)

# ---------------- title plaque, hymn board (search keywords in the first 30 s) ----------------
box((0.0, D['y'] - 0.02, 10.95), (4.6, 0.08, 1.35), M_PLATE, name='titleplate')
box((0.0, D['y'] - 0.04, 11.65), (4.7, 0.06, 0.05), M_BRASS); box((0.0, D['y'] - 0.04, 10.27), (4.7, 0.06, 0.05), M_BRASS)
fit_w(text('J. S. BACH', (0, D['y'] - 0.1, 11.33), 0.44, M_TXTG, font=FONT_SERIF), 4.2)
fit_w(text('Toccata and Fugue in D minor, BWV 565', (0, D['y'] - 0.1, 10.88), 0.26, M_TXT, font=FONT_SERIF), 4.3)
fit_w(text('バッハ「トッカータとフーガ ニ短調」', (0, D['y'] - 0.1, 10.5), 0.2, M_TXTJ), 4.2)
box((-2.85, -0.3, 2.85), (1.15, 0.1, 2.6), M_WOODD, name='hymnboard')
box((-2.85, -0.36, 4.16), (1.2, 0.06, 0.08), M_BRASS); box((-2.85, -0.36, 1.54), (1.2, 0.06, 0.08), M_BRASS)
for k_, (en, jp) in enumerate((('PIPE ORGAN', 'パイプオルガン'), ('MARBLE MACHINE', 'ビー玉マシン'), ('CLASSICAL · BAROQUE', 'クラシック'),
                                ('SATISFYING ASMR', ''))):
    zz = 3.85 - k_ * 0.6
    fit_w(text(en, (-2.85, -0.37, zz), 0.15, M_TXT), 1.0)
    if jp: fit_w(text(jp, (-2.85, -0.37, zz - 0.2), 0.1, M_TXTJ), 1.0)
rod((-2.85, -0.3, 1.5), (-2.85, -0.3, 0.0), 0.04, M_WOODD, 6)

# candles along the chancel edge and the nave
CANDLES = [(-3.4, -1.5, 0.0), (-2.2, -1.5, 0.0), (1.9, -1.5, 0.0), (-3.9, -1.0, 0.0)] + [(x, 2.6, -5.6) for x in np.arange(4.0, 33.0, 2.25)]
for (x, y, z) in CANDLES:
    hgt = 0.5 if z == 0.0 else 0.9
    rod((x, y, z), (x, y, z + hgt), 0.05, M_CANDLE, 10)
    fl = sphere((x, y, z + hgt + 0.06), 0.035, M_FLAME, (8, 6)); fl.scale = (0.035, 0.035, 0.07)
    li = bpy.data.lights.new('cl', 'POINT'); li.energy = 6.0 if z == 0.0 else 12.0; li.color = (1.0, 0.6, 0.3); li.shadow_soft_size = 0.1
    lo = link(bpy.data.objects.new('cl', li)); lo.location = (x, y - 0.1, z + hgt + 0.25)

# ---------------- world & lights ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bgn = w.node_tree.nodes['Background']; bgn.inputs['Color'].default_value = (0.004, 0.005, 0.014, 1); bgn.inputs['Strength'].default_value = 1.0
moon = bpy.data.lights.new('moon', 'SUN'); moon.energy = 0.35; moon.color = (0.5, 0.62, 1.0); moon.angle = 0.05
mo = link(bpy.data.objects.new('moon', moon)); mo.rotation_euler = (math.radians(55), math.radians(-30), math.radians(15))
def area(loc, rot, size, energy, col):
    li = bpy.data.lights.new('a', 'AREA'); li.energy = energy; li.size = size; li.color = col
    o = link(bpy.data.objects.new('a', li)); o.location = loc; o.rotation_euler = rot; return li
def spot(loc, aim, energy, col, angle=40, size=0.6):
    li = bpy.data.lights.new('s', 'SPOT'); li.energy = energy; li.color = col; li.spot_size = math.radians(angle); li.spot_blend = 0.6
    li.shadow_soft_size = size
    o = link(bpy.data.objects.new('s', li)); o.location = loc
    d = Vector(aim) - Vector(loc); o.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler(); return li
spot((-2.5, -6.5, 12.0), (0.0, 0.0, 6.5), 1800.0, (1.0, 0.8, 0.55), 38)       # the galleries
spot((3.0, -6.0, 6.0), (0.5, 0.0, 1.5), 1300.0, (1.0, 0.78, 0.5), 42)          # door, console, pedal
spot((6.0, -6.5, 7.0), (6.0, 0.6, 1.8), 2000.0, (1.0, 0.8, 0.55), 45)          # stairs / lift
spot((13.0, -6.5, 7.0), (13.0, 0.6, 1.5), 2000.0, (1.0, 0.8, 0.55), 45)        # upper stairs, landing
spot((18.5, -6.5, 5.0), (18.0, 0.6, -1.0), 2000.0, (1.0, 0.8, 0.55), 45)       # cascade
spot((23.5, -6.5, 3.0), (23.5, 0.4, -2.8), 1700.0, (1.0, 0.8, 0.55), 48)       # solo / pedal II
area((0.0, 1.8, 11.0), (math.radians(-160), 0, 0), 3.0, 120.0, (0.5, 0.6, 1.0))            # rose window glow
inner = area((0.0, 3.1, 2.6), (math.radians(90), 0, 0), 2.6, 0.0, (1.0, 0.55, 0.22))        # light from inside the door
for f in range(NF):
    v = 1400.0 * ease((T[f] - D['t_open0']) / (D['t_open1'] - D['t_open0']))
    if f in (0, NF - 1) or f % 6 == 0:
        inner.energy = v; inner.keyframe_insert('energy', frame=f)
fl_ = area((25.4, -2.5, 3.0), (math.radians(40), 0, 0), 6.0, 0.0, (1.0, 0.75, 0.35))          # the flood lights up the golden pipes
for f in range(NF):
    v = 1400.0 * ease((T[f] - FI['t_final'] + 0.2) / 0.6)
    if f % 3 == 0 or f == NF - 1: fl_.energy = v; fl_.keyframe_insert('energy', frame=f)

# ---------------- camera ----------------
CK = [  # t, x, z, width
    (0.0, 0.0, 6.7, 7.4), (1.2, 0.1, 7.3, 6.6), (2.4, 0.45, 7.8, 5.0), (4.7, 0.45, 7.7, 5.0), (5.7, -0.25, 6.5, 5.0), (9.1, -0.25, 6.2, 5.0),
    (10.3, 0.4, 4.9, 5.0), (13.6, 0.6, 4.4, 5.2), (14.9, 0.9, 3.4, 6.2), (16.1, 0.2, 3.6, 7.0), (18.6, 0.4, 2.5, 6.0),
    (20.6, 0.9, 1.6, 5.0)]
FOC = np.zeros((NF, 3)); WID = np.zeros(NF)
tri = ['T0', 'T1', 'T2']
for f in range(NF):
    t = T[f]
    if t <= CK[-1][0]:
        k = max(i for i in range(len(CK)) if CK[i][0] <= t); k1 = min(k + 1, len(CK) - 1)
        u = 0 if k1 == k else (t - CK[k][0]) / (CK[k1][0] - CK[k][0]); u = ease(u)
        FOC[f] = (CK[k][1] + (CK[k1][1] - CK[k][1]) * u, 0, CK[k][2] + (CK[k1][2] - CK[k][2]) * u); WID[f] = CK[k][3] + (CK[k1][3] - CK[k][3]) * u
    elif t < 41.0:
        pm = np.mean([POS[n][f] for n in tri], axis=0); FOC[f] = (pm[0] + 0.45, 0, pm[2] + 0.3); WID[f] = 4.3
    elif t < 46.5:
        pm = np.mean([POS[n][f] for n in tri], axis=0); g_ = POS['GIANT2'][f]; d_ = POS['DROP'][f]
        FOC[f] = ((pm[0] + g_[0] + d_[0]) / 3 + 0.3, 0, (pm[2] + g_[2]) / 2 + 0.4); WID[f] = 5.6
    elif t < FI['t_final'] - 0.9:
        d_ = POS['DROP'][f]; FOC[f] = (d_[0] + 0.35, 0, d_[2] + 0.2); WID[f] = 4.6
    elif t < FI['t_last'] + 0.3:
        u = ease((t - (FI['t_final'] - 0.9)) / 1.4)
        FOC[f] = (24.6 + 0.9 * u, 0, -3.0 + 2.6 * u); WID[f] = 5.0 + 3.6 * u
    else:
        u = ease((t - FI['t_last'] - 0.3) / 2.5)
        FOC[f] = (25.5 - 1.5 * u, 0, -0.4 + 1.3 * u); WID[f] = 8.6 + 3.4 * u
FOC[:, 0] = smooth(FOC[:, 0], 0.3); FOC[:, 2] = smooth(FOC[:, 2], 0.35); WID = smooth(WID, 0.45)
LENS = 35.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 400
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
for f in range(NF):
    Dd = WID[f] / (9 / 16) * LENS / 24
    CP[f] = (FOC[f, 0], -Dd, FOC[f, 2] + 0.12 * Dd)
    TP[f] = (FOC[f, 0], 0.4, FOC[f, 2] - 0.02 * Dd)
key_loc(cam, CP); key_loc(tgt, TP)

# ---------------- HUD: hook (first seconds) + marble counter ----------------
hud = link(bpy.data.objects.new('hud', None)); hud.parent = cam
HZ = 6.0; hh_ = HZ * 24 / LENS; hw_ = hh_ * 9 / 16
hud.location = (0, 0, -HZ)
def hudtext(s, x, y, size, m, align='CENTER', font=None, w=None):
    o = fit_w(text(s, (x, y, 0), size, m, rot=(0, 0, 0), align=align, ext=0.0, font=font), w); o.parent = hud; o.visible_shadow = False; return o
M_HPL = emis('hudplate', (0.01, 0.01, 0.02), 1.0)
hook = link(bpy.data.objects.new('hook', None)); hook.parent = hud; hook.location = (0, -hh_ * 0.355, 0)
hp_ = box((0, 0, -0.01), (hw_ * 0.98, hh_ * 0.15, 0.005), M_HPL, name='hookplate'); hp_.parent = hook
for o_ in (hudtext('Can 3 marbles unlock', 0, hh_ * 0.036, hh_ * 0.042, M_TXT, w=hw_ * 0.92), hudtext('the cathedral organ?', 0, -hh_ * 0.012, hh_ * 0.042, M_TXT, w=hw_ * 0.92),
           hudtext('3つのビー玉で大聖堂の扉は開く？', 0, -hh_ * 0.052, hh_ * 0.027, M_TXTJ, w=hw_ * 0.9)):
    o_.parent = hook
vis = np.array([1.0 if t < 7.2 else max(0.001, 1 - (t - 7.2) * 3) for t in T])
for k in range(3): key_series(hook, 'scale', k, np.maximum(0.001, vis), 1e-3)
# counter (top right): MARBLES PLAYED: N
cnt = np.array(P['counter'])
CV = np.array([int(np.searchsorted(cnt, t, 'right')) for t in T])
ctr = link(bpy.data.objects.new('ctr', None)); ctr.parent = hud; ctr.location = (hw_ * 0.18, hh_ * 0.43, 0)
cp_ = box((0.0, 0, -0.01), (hw_ * 0.6, hh_ * 0.085, 0.005), M_HPL, name='ctrplate'); cp_.parent = ctr
l1 = hudtext('MARBLES', -hw_ * 0.12, hh_ * 0.014, hh_ * 0.034, M_TXT, w=hw_ * 0.3); l1.parent = ctr
l2 = hudtext('演奏したビー玉', -hw_ * 0.12, -hh_ * 0.024, hh_ * 0.018, M_TXTJ, w=hw_ * 0.3); l2.parent = ctr
DW = hh_ * 0.027
for slot in range(3):
    for dgt in range(10):
        o = hudtext(str(dgt), hw_ * 0.14 + (slot - 1) * DW, 0, hh_ * 0.052, M_TXTG); o.parent = ctr
        vv = np.zeros(NF)
        for f in range(NF):
            s_ = f'{CV[f]:>3d}'; vv[f] = 1.0 if s_[slot] == str(dgt) else 0.0
        if not vv.any(): o.hide_render = True; continue
        for k in range(3): key_series(o, 'scale', k, np.maximum(0.001, vv))
show = np.array([0.001 if t < 7.0 else min(1.0, (t - 7.0) * 4) for t in T])
bump = np.ones(NF)
for f in range(1, NF):
    if CV[f] != CV[f - 1] and CV[f] < 30:
        for df in range(6):
            if f + df < NF: bump[f + df] = max(bump[f + df], 1.0 + 0.12 * (1 - df / 6))
for k in range(3): key_series(ctr, 'scale', k, np.maximum(0.001, show * bump), 1e-3)
# end card: '565 marbles = BWV 565'
endc = link(bpy.data.objects.new('endc', None)); endc.parent = hud; endc.location = (0, -hh_ * 0.33, 0)
ep = box((0, 0, -0.01), (hw_ * 0.98, hh_ * 0.16, 0.005), M_HPL, name='endplate'); ep.parent = endc
for o_ in (hudtext('565 marbles = BWV 565', 0, hh_ * 0.03, hh_ * 0.042, M_TXTG, w=hw_ * 0.92),
           hudtext('Did you notice?', 0, -hh_ * 0.022, hh_ * 0.03, M_TXT, w=hw_ * 0.9),
           hudtext('ビー玉565個 = BWV 565', 0, -hh_ * 0.058, hh_ * 0.02, M_TXTJ, w=hw_ * 0.9)):
    o_.parent = endc
ev_ = np.array([0.001 if t < FI['t_last'] + 0.35 else min(1.0, (t - FI['t_last'] - 0.35) * 3) for t in T])
for k in range(3): key_series(endc, 'scale', k, ev_, 1e-3)
for o in bpy.data.objects:
    p_ = o
    while p_.parent is not None: p_ = p_.parent
    if p_ is cam: o.visible_shadow = False


# ---------------- run ----------------
if MODE == 'test':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS)); sc.frame_set(f)
        sc.render.filepath = f'{HERE}/test_{f:04d}.png'
        t1 = time.time(); bpy.ops.render.render(write_still=True); print('frame', f, 'sec', round(time.time() - t1, 1))
elif MODE == 'range':
    sc.frame_start, sc.frame_end = int(args[1]), int(args[2])
    sc.render.image_settings.file_format = 'PNG'; sc.render.filepath = f'{HERE}/frames/f_'
    bpy.ops.render.render(animation=True)
elif MODE == 'info':
    print('NF', NF, 'objects', len(bpy.data.objects))
