"""The Entertainer (Scott Joplin, 1902) - 'Saloon player piano' marble machine. Side view along a long Western saloon.
All motion comes from layout.py (one gravity, bounces never add speed, rolls at constant speed, mechanisms for the rest).
Melody: glass bells (short notes, the ball bounces) and brass chime rails (held notes, the ball rolls along while it rings).
Bass: ZUN lands on the left end of each floor see-saw on the beat, CHA on the right end on the off-beat (oom-pah).
jp_sub.otf = Noto Sans CJK JP Bold (SIL OFL) subset to the glyphs used here.
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
G = LAY['G']; R = LAY['R']; EV = LAY['ev']; YL = LAY['Y']; V = LAY['V']
T0 = E['t0']; BEAT = E['beat']
def tb(b): return T0 + b * BEAT
rng = np.random.default_rng(7)
FLOOR = -0.35

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


def glowmat(name, gain=8.0, base=0.0, rough=0.3, metal=0.0, coat=0.5):
    """object colour = base colour; object alpha drives emission (hit flash)"""
    m = mat(name, (1, 1, 1), rough, **{'Coat Weight': coat, 'Metallic': metal}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
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
    m = mat(name, (1, 1, 1), rough, **{'Metallic': metal}); nt = m.node_tree
    oi = nt.nodes.new('ShaderNodeObjectInfo'); nt.links.new(oi.outputs['Color'], nt.nodes['Principled BSDF'].inputs['Base Color'])
    return m


def woodmat(name, c1, c2, scale=6.0, rough=0.45, stretch=(1, 8, 1), world=False):
    m = mat(name, c1, rough); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    tc = nt.nodes.new('ShaderNodeTexCoord'); mp = nt.nodes.new('ShaderNodeMapping'); mp.inputs['Scale'].default_value = stretch
    if world:
        gp = nt.nodes.new('ShaderNodeNewGeometry'); nt.links.new(gp.outputs['Position'], mp.inputs['Vector'])
        wv = nt.nodes.new('ShaderNodeTexWave'); wv.inputs['Scale'].default_value = scale; wv.inputs['Distortion'].default_value = 6.0
        rp = nt.nodes.new('ShaderNodeValToRGB'); rp.color_ramp.elements[0].color = (*c1, 1); rp.color_ramp.elements[1].color = (*c2, 1)
        nt.links.new(mp.outputs[0], wv.inputs['Vector']); nt.links.new(wv.outputs['Fac'], rp.inputs['Fac']); nt.links.new(rp.outputs['Color'], b.inputs['Base Color'])
        return m
    wv = nt.nodes.new('ShaderNodeTexWave'); wv.inputs['Scale'].default_value = scale; wv.inputs['Distortion'].default_value = 6.0
    rp = nt.nodes.new('ShaderNodeValToRGB'); rp.color_ramp.elements[0].color = (*c1, 1); rp.color_ramp.elements[1].color = (*c2, 1)
    nt.links.new(tc.outputs['Object'], mp.inputs['Vector']); nt.links.new(mp.outputs[0], wv.inputs['Vector'])
    nt.links.new(wv.outputs['Fac'], rp.inputs['Fac']); nt.links.new(rp.outputs['Color'], b.inputs['Base Color'])
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
    key = (kind, m.name if m else None) + tuple(a)
    if key in _MC: return _MC[key]
    bm = bmesh.new()
    if kind == 'cube': bmesh.ops.create_cube(bm, size=1.0)
    elif kind == 'cyl': bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=1.0, radius2=1.0, depth=1.0)
    elif kind == 'cone': bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=a[1], radius2=a[2], depth=a[3])
    elif kind == 'sph': bmesh.ops.create_uvsphere(bm, u_segments=a[0], v_segments=a[1], radius=1.0)
    elif kind == 'dome':
        bmesh.ops.create_uvsphere(bm, u_segments=a[0], v_segments=a[1], radius=1.0)
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < -0.05], context='VERTS')
    elif kind == 'plane': bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    elif kind == 'bowl':
        bmesh.ops.create_uvsphere(bm, u_segments=a[0], v_segments=a[1], radius=1.0)
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z > 0.25], context='VERTS')
    elif kind == 'torus':
        bm.free(); bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=a[0], major_segments=a[1], minor_segments=6)
        o = bpy.context.object; me = o.data; bpy.data.objects.remove(o)
        if m: me.materials.append(m)
        _MC[key] = me; return me
    me = bpy.data.meshes.new(kind); bm.to_mesh(me); bm.free()
    if kind in ('sph', 'cone', 'dome', 'bowl'):
        for p in me.polygons: p.use_smooth = True
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


def text(s, loc, size, m, rot=(math.pi / 2, 0, 0), align='CENTER', ext=0.01):
    cu = bpy.data.curves.new('t', 'FONT'); cu.body = s; cu.font = FONT; cu.size = size; cu.extrude = ext
    cu.align_x = align; cu.align_y = 'CENTER'
    o = link(bpy.data.objects.new('txt', cu)); o.location = loc; o.rotation_euler = rot; o.data.materials.append(m); return o


def pulse_alpha(times, NFr=NF, decay=5.0, length=16, amp=1.0):
    a = np.zeros(NFr)
    for t in times:
        f0 = int(round(t * FPS)); m_ = np.arange(max(0, f0), min(NFr, f0 + length))
        a[m_] = np.maximum(a[m_], amp * np.exp(-(m_ - f0) / decay))
    return a


def key_flash(o, t, peak=1.0, hold=0.0, tail=16):
    """object alpha: 0 before the hit, peak at the hit frame, decays (held for 'hold' seconds for rails)"""
    fh = int(round(t * FPS)); col = tuple(o.color[:3])
    o.color = (*col, 0); o.keyframe_insert('color', index=3, frame=max(0, fh - 1))
    o.color = (*col, peak); o.keyframe_insert('color', index=3, frame=fh)
    fhold = fh + int(round(hold * FPS))
    if hold > 0: o.color = (*col, 0.7 * peak); o.keyframe_insert('color', index=3, frame=fhold)
    o.color = (*col, 0.35 * peak); o.keyframe_insert('color', index=3, frame=fhold + 4)
    o.color = (*col, 0.06); o.keyframe_insert('color', index=3, frame=fhold + tail)


def smooth(Xa, k):
    Y = Xa.copy(); a = 1 - math.exp(-1 / FPS / k)
    for f in range(1, len(Xa)): Y[f] = Y[f - 1] + (Xa[f] - Y[f - 1]) * a
    for f in range(len(Xa) - 2, -1, -1): Y[f] = Y[f + 1] + (Y[f] - Y[f + 1]) * a
    return Y


# ---------------- ball paths ----------------
def seg_at(g, t):
    p = np.array(g['p'])
    if g['k'] == 'rest': return p
    u = t - g['t0']
    if g['k'] == 'lin': return p + (np.array(g['q']) - p) * (u / max(1e-9, g['t1'] - g['t0']))
    return p + np.array(g['v']) * u + 0.5 * np.array(g['a']) * u * u


class BallPath:
    def __init__(s, d): s.seg = d['seg']; s.hits = d['hits']
    def at(s, t):
        for g in s.seg:
            if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9: return seg_at(g, t)
        return seg_at(s.seg[0], s.seg[0]['t0']) if t < s.seg[0]['t0'] else seg_at(s.seg[-1], s.seg[-1]['t1'])
BP = {n: BallPath(d) for n, d in LAY['balls'].items()}
POS = {n: np.array([b.at(t) for t in T]) for n, b in BP.items()}

# ---------------- materials ----------------
M_IVORY = mat('ivory', (0.93, 0.9, 0.82), 0.12, **{'Coat Weight': 1.0})
M_KEYBLK = mat('keyblack', (0.02, 0.02, 0.025), 0.2, **{'Coat Weight': 1.0})
M_RED = mat('red', (0.8, 0.04, 0.03), 0.15, **{'Coat Weight': 1.0})
M_GOLD = mat('gold', (1.0, 0.72, 0.25), 0.25, **{'Metallic': 1.0})
M_BRASS = mat('brass', (0.85, 0.6, 0.25), 0.3, **{'Metallic': 1.0})
M_WALNUT = woodmat('walnut', (0.22, 0.1, 0.04), (0.38, 0.19, 0.08), 9.0, 0.3)
M_ZUN = mat('zun', (0.35, 0.06, 0.02), 0.15, **{'Coat Weight': 1.0})
M_BRASSBALL = mat('brassball', (0.95, 0.75, 0.35), 0.18, **{'Metallic': 1.0})
M_HAT = mat('hat', (0.42, 0.25, 0.12), 0.7)
M_HATBAND = mat('hatband', (0.08, 0.05, 0.03), 0.6)
M_STEEL = mat('steel', (0.5, 0.5, 0.52), 0.35, **{'Metallic': 0.9})
M_DARK = mat('dark', (0.04, 0.03, 0.03), 0.6)
M_MAHO = woodmat('maho', (0.2, 0.05, 0.02), (0.36, 0.11, 0.045), 1.0, 0.22, (0.4, 3, 3), world=True)
M_PLANK = woodmat('plank', (0.3, 0.16, 0.07), (0.45, 0.26, 0.12), 1.2, 0.5, (0.6, 3, 3), world=True)
M_WALL = woodmat('wall', (0.13, 0.065, 0.03), (0.2, 0.1, 0.045), 0.6, 0.75, (1.5, 1, 0.3), world=True)
M_FLOOR = woodmat('floor', (0.12, 0.065, 0.035), (0.2, 0.11, 0.055), 0.5, 0.45, (0.25, 2.2, 1), world=True)
M_PIANO = woodmat('pianowood', (0.07, 0.025, 0.012), (0.15, 0.06, 0.025), 1.0, 0.18, (0.7, 3, 3), world=True)
M_VELVET = mat('velvet', (0.22, 0.01, 0.02), 0.9)
M_FELT = mat('felt', (0.08, 0.25, 0.12), 0.95)
M_PAPER = mat('paper', (0.86, 0.78, 0.6), 0.85)
M_BEER = mat('beer', (0.95, 0.5, 0.06), 0.12, **{'Coat Weight': 1.0, 'Emission Color': (1.0, 0.55, 0.1, 1), 'Emission Strength': 0.25})
M_FOAM = mat('foam', (0.97, 0.95, 0.88), 0.9)
M_GLASS = mat('glass', (0.85, 0.92, 0.95), 0.03, **{'Transmission Weight': 1.0, 'IOR': 1.45})
M_BELL = glowmat('bell', gain=9.0, base=0.35, rough=0.08, coat=1.0)
M_RAIL = glowmat('rail', gain=6.0, base=0.0, rough=0.25, metal=0.85)
M_PAD = glowmat('pad', gain=6.0, base=0.0, rough=0.35, coat=0.6)
M_OBJ = objcol('objcol', 0.5)
M_FX = fxmat('fx', 8.0)
M_TXT_W = emis('txtw', (1, 0.95, 0.85), 2.0); M_TXT_Y = emis('txty', (1, 0.75, 0.25), 3.0)
M_TXT_R = emis('txtr', (1, 0.2, 0.12), 3.0); M_TXT_D = mat('txtd', (0.06, 0.03, 0.02), 0.6); M_TXT_RD = mat('txtrd', (0.55, 0.04, 0.03), 0.6)
M_NEON_P = emis('neonp', (1.0, 0.3, 0.6), 4.0); M_NEON_C = emis('neonc', (0.3, 0.8, 1.0), 4.0)
M_LAMP = emis('lamp', (1.0, 0.72, 0.38), 6.0)


def pitch_col(p, kind='bell'):
    h = ((p - 60) % 12) / 12.0
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, 0.9, 0.95)
    return (r, g, b)


# ---------------- balls ----------------
BALL = {}
BMAT = {'coin': M_GOLD, 'P': M_IVORY, 'C': M_RED, 'Z': M_ZUN, 'Ch': M_BRASSBALL}
for n in BP:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=R, segments=32, ring_count=16); o = bpy.context.object; o.name = n
    for p in o.data.polygons: p.use_smooth = True
    o.data.materials.append(BMAT[n])
    if n == 'Z':     # orange band
        bpy.ops.mesh.primitive_torus_add(major_radius=R * 0.99, minor_radius=0.025, major_segments=32, minor_segments=8)
        st = bpy.context.object; st.data.materials.append(mat('zband', (1.0, 0.45, 0.08), 0.3)); st.parent = o
    if n == 'P':     # piano-key stripe
        bpy.ops.mesh.primitive_torus_add(major_radius=R * 0.99, minor_radius=0.03, major_segments=32, minor_segments=8)
        st = bpy.context.object; st.data.materials.append(M_KEYBLK); st.parent = o
    if n == 'coin':
        bpy.ops.mesh.primitive_torus_add(major_radius=R * 0.98, minor_radius=0.018, major_segments=32, minor_segments=6, rotation=(math.pi / 2, 0, 0))
        st = bpy.context.object; st.data.materials.append(M_BRASS); st.parent = o
    key_loc(o, POS[n])
    sp = np.zeros(NF)
    for f in range(1, NF): sp[f] = sp[f - 1] + (POS[n][f, 0] - POS[n][f - 1, 0]) / R
    o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, sp)
    BALL[n] = o
# cowboy hat for C (does not spin with the ball)
hat = link(bpy.data.objects.new('hat', None))
brim = obj(prim('cyl', M_HAT, 32), 'brim'); brim.scale = (0.25, 0.25, 0.018); brim.parent = hat; brim.location = (0, 0, 0)
crown = obj(prim('cone', M_HAT, 24, 1.0, 0.8, 1.0), 'crown'); crown.scale = (0.13, 0.11, 0.12); crown.parent = hat; crown.location = (0, 0, 0.065)
band = obj(prim('cyl', M_HATBAND, 24), 'band'); band.scale = (0.133, 0.113, 0.022); band.parent = hat; band.location = (0, 0, 0.02)
for side in (-1, 1):     # curled brim edges
    cu = obj(prim('cyl', M_HAT, 16), 'curl'); cu.scale = (0.05, 0.25, 0.03); cu.parent = hat; cu.location = (0, side * 0.0, 0.0)
HP = POS['C'] + np.array([0, 0, R * 0.78])
hat.rotation_euler = (0.0, -0.18, 0.0)
key_loc(hat, HP)
# little wobble of the hat on every landing
hw = np.zeros(NF)
for h in BP['C'].hits:
    f0 = int(round(h['t'] * FPS))
    for df in range(10):
        if f0 + df < NF: hw[f0 + df] += 0.12 * math.exp(-df / 3) * math.cos(df * 1.7)
key_series(hat, 'rotation_euler', 1, -0.18 + hw, 1e-3)

# ---------------- melody bells and chime rails ----------------
RB = 0.1
BELLS = []; RAILS = []; LANEPTS = {}
def lane_key(n, h):
    if n == 'coin': return 'coin'
    if n == 'P': return 'PA' if h['t'] < EV['mug']['t_load'] else 'PA2'
    return 'CB' if h['t'] < tb(71.4) else 'CA2'
for n in ('coin', 'P', 'C'):
    for h in BP[n].hits:
        k = h['kind']
        if k not in ('bell', 'cbell', 'rail', 'juke'): continue
        p = np.array(h['p']); nv = np.array(h['n']); nv /= np.linalg.norm(nv)
        col = pitch_col(h['pitch']) if k != 'juke' else (1.0, 0.78, 0.3)
        if n == 'coin': col = tuple(0.55 + 0.45 * c for c in col)
        if k == 'rail':
            x0 = p[0] - 0.12; x1 = p[0] + h['roll']; zt = p[2] - R
            rl = box(((x0 + x1) / 2, p[1], zt - 0.025), (x1 - x0, 0.12, 0.05), M_RAIL, name='rail'); rl.color = (*col, 0)
            # brass end caps + groove lines
            for xx in (x0, x1): box((xx, p[1], zt - 0.025), (0.03, 0.14, 0.06), M_BRASS)
            key_flash(rl, h['t'], 1.0, hold=h['roll'] / max(0.1, abs(h['vout'][0])), tail=14)
            RAILS.append(dict(o=rl, x0=x0, x1=x1, z=zt, y=p[1], t=h['t'], ball=n, i=h['li'], hold=h['roll']))
            LANEPTS.setdefault(lane_key(n, h), []).append((x0 + 0.08, zt - 0.05)); LANEPTS[lane_key(n, h)].append((x1 - 0.08, zt - 0.05))
        else:
            c = p - (R + RB) * nv
            dm = obj(prim('dome', M_BELL, 24, 12), 'bell'); dm.location = c; dm.scale = (RB, RB, RB * 0.9); dm.color = (*col, 0)
            collar = box((c[0], c[1], c[2] - 0.012), (0.23, 0.23, 0.025), M_BRASS, name='collar')
            knob = sphere((c[0], c[1], c[2] - 0.035), 0.03, M_BRASS)
            key_flash(dm, h['t'], 1.0)
            BELLS.append(dict(o=dm, c=c, t=h['t'], ball=n, i=h['li'], contact=p - R * nv, n=nv, kind=k))
            LANEPTS.setdefault(lane_key(n, h), []).append((c[0], c[2] - 0.05))
# hit dip of each bell (a small knock)
for b_ in BELLS:
    o = b_['o']; fh = int(round(b_['t'] * FPS)); base = o.location.copy()
    for df, d in ((-1, 0), (1, -0.02), (4, 0.006), (8, 0)):
        o.location = base + Vector((0, 0, d)); o.keyframe_insert('location', frame=max(0, fh + df))

# lane boards (polished counter tops under the bells), stepped, with brass posts to each bell
def lane_board(pts, y, width=0.36, name='board'):
    pts = sorted(pts); xs = np.array([p[0] for p in pts]); zs = np.array([p[1] for p in pts])
    X = np.arange(xs[0] - 0.35, xs[-1] + 0.35, 0.1)
    Z = np.array([zs[(xs > x - 0.5) & (xs < x + 0.5)].min() if ((xs > x - 0.5) & (xs < x + 0.5)).any() else np.interp(x, xs, zs) for x in X]) - 0.16
    # stepped: hold the level in 1.2 m pieces (top = lowest bell base of the piece)
    out = []
    for x0 in np.arange(X[0], X[-1], 1.2):
        m_ = (X >= x0) & (X < x0 + 1.2)
        if m_.any(): out.append((x0, min(x0 + 1.2, X[-1] + 0.05), float(Z[m_].min())))
    for x0, x1, z in out:
        box(((x0 + x1) / 2, y, z - 0.04), (x1 - x0 + 0.002, width, 0.08), M_MAHO, name=name)
        box(((x0 + x1) / 2, y - width / 2 - 0.01, z - 0.02), (x1 - x0 + 0.002, 0.02, 0.05), M_BRASS)
    for x0, x1, z in out[::2]:
        rod(((x0 + x1) / 2, y, z - 0.08), ((x0 + x1) / 2, y, FLOOR), 0.05, M_MAHO, 8)
    return out
BOARDS = {}
for key, y in (('coin', YL['coin']), ('PA', YL['P']), ('PA2', YL['P']), ('CB', YL['C']), ('CA2', YL['C'])):
    if key in LANEPTS: BOARDS[key] = lane_board(LANEPTS[key], y)
def board_z(key, x):
    for x0, x1, z in BOARDS[key]:
        if x0 - 0.05 <= x <= x1 + 0.05: return z
    return min(z for _, _, z in BOARDS[key])
for b_ in BELLS:
    key = lane_key(b_['ball'], dict(t=b_['t']))
    zb = board_z(key, b_['c'][0])
    rod((b_['c'][0], b_['c'][1], b_['c'][2] - 0.04), (b_['c'][0], b_['c'][1], zb), 0.014, M_BRASS, 8)
for r_ in RAILS:
    key = lane_key(r_['ball'], dict(t=r_['t']))
    for xx in (r_['x0'] + 0.06, r_['x1'] - 0.06):
        rod((xx, r_['y'], r_['z'] - 0.05), (xx, r_['y'], board_z(key, xx)), 0.014, M_BRASS, 8)

# hit rings (FX pool)
NRING = 16
ring_loc = np.zeros((NRING, NF, 3)); ring_scl = np.full((NRING, NF), 1e-3); ring_a = np.zeros((NRING, NF)); ring_c = [(1, 1, 1)] * NRING
HITS_FX = [(b['t'], b['contact'], 'm') for b in BELLS] + [(r['t'], np.array([BP[r['ball']].at(r['t'])[0], r['y'], r['z']]), 'm') for r in RAILS]
for k, (t, c, kind) in enumerate(sorted(HITS_FX, key=lambda a: a[0])):
    r = k % NRING; f0 = int(round(t * FPS))
    for df in range(12):
        f = f0 + df
        if f >= NF: break
        u = df / 11; ring_loc[r, f] = c + np.array([0, -0.2, 0.02]); ring_scl[r, f] = 0.25 + 0.9 * (1 - (1 - u) ** 2)
        ring_a[r, f] = (1 - u) ** 1.6
for r in range(NRING):
    o = obj(prim('torus', M_FX, 0.06, 32), 'ring'); o.rotation_euler = (math.pi / 2, 0, 0); o.scale = (0.2, 0.2, 0.2)
    o.color = (1.0, 0.85, 0.55, 0)
    for f in range(1, NF):
        if ring_a[r, f] == 0: ring_loc[r, f] = ring_loc[r, f - 1]
    key_loc(o, ring_loc[r])
    for k in range(3): key_series(o, 'scale', k, ring_scl[r] * 0.22)
    key_series(o, 'color', 3, ring_a[r], 1e-3)

# ---------------- bass: see-saws on the floor rail ----------------
PADW, PADD, PADT = 0.24, 0.3, 0.04
H0 = 0.06                                 # pad centre above the plank axis
TH0 = math.radians(7.0)
def frame_from_n(n):
    n = np.array(n, float); n /= np.linalg.norm(n)
    xa = np.cross([0, 1, 0], n); xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return Matrix(((xa[0], ya[0], n[0]), (xa[1], ya[1], n[1]), (xa[2], ya[2], n[2])))
def roty(a): return Matrix.Rotation(a, 3, 'Y')
zh = [h for h in BP['Z'].hits if h['kind'] in ('saw', 'cat')]
chh = [h for h in BP['Ch'].hits if h['kind'] in ('saw', 'chacup')]
PADS = []
used = set()
def pad_center(h): n = np.array(h['n']); n /= np.linalg.norm(n); return np.array(h['p']) - (R + PADT / 2) * n, n
def solve_saw(cL, cR, thL, thR):
    """plank pivot X,Z and arms so that the left pad centre is cL at angle thL and the right pad centre cR at angle thR"""
    # left:  cL = P0 + roty(thL) (-aL, 0, H0);  right: cR = P0 + roty(thR) (aR, 0, H0)
    # unknowns X, Z, aL, aR (linear)
    A = np.array([[1, 0, -math.cos(thL), 0], [0, 1, math.sin(thL), 0], [1, 0, 0, math.cos(thR)], [0, 1, 0, -math.sin(thR)]])
    bvec = np.array([cL[0] - H0 * math.sin(thL), cL[2] - H0 * math.cos(thL), cR[0] - H0 * math.sin(thR), cR[2] - H0 * math.cos(thR)])
    X, Zp, aL, aR = np.linalg.solve(A, bvec)
    return X, Zp, aL, aR
SAWS = []
def make_saw(hL, hR, kind='saw'):
    """hL: hit on the left end (or None), hR: hit on the right end (or None).  normal see-saw: left hit first (config +TH0),
    right hit later (config -TH0).  catapult: left = CHA's cup (latched until ZUN lands on the right end)."""
    y = YL['bass']
    if kind == 'cat':
        cL, nL = pad_center(hL); cR, nR = pad_center(hR)
        th = -math.atan2(cR[2] - cL[2], cR[0] - cL[0])          # plank along the line between the two (latched)
        Lc = math.hypot(cR[0] - cL[0], cR[2] - cL[2]); aL = 0.62 * Lc; aR = Lc - aL     # cup on the long arm
        P0 = cL - np.array(roty(th)) @ np.array([-aL, 0, H0]); X, Zp = P0[0], P0[2]
        thL, thR = th, th
        keys = [(hL['t'], th), (hR['t'], th), (hR['t'] + 0.2, th + 2 * TH0 + 0.05)]
    elif hL is not None and hR is not None:
        cL, nL = pad_center(hL); cR, nR = pad_center(hR)
        for thd in (7.0, 10.0, 13.0, 16.0):           # steeper rock when the two ends sit at different heights
            th_ = math.radians(thd); thL, thR = th_, -th_
            X, Zp, aL, aR = solve_saw(cL, cR, thL, thR)
            if min(aL, aR) >= 0.15 and max(aL, aR) <= 1.2: break
        else: return False
        keys = [(hL['t'], th_), (hL['t'] + 0.08, -th_), (hR['t'], -th_), (hR['t'] + 0.08, th_)]
    else:
        h = hL or hR; c, nn = pad_center(h); left = hL is not None
        thc = TH0 if left else -TH0
        a = 0.33
        X = c[0] + (a * math.cos(thc) - H0 * math.sin(thc)) * (1 if left else -1)
        # solve exactly for the pivot so the pad centre is c at the hit angle
        loc = np.array([-a if left else a, 0, H0]); M = np.array(roty(thc))
        P0 = c - M @ loc; X, Zp = P0[0], P0[2]
        aL, aR = (a, 0.3) if left else (0.3, a)
        keys = [(h['t'], thc), (h['t'] + 0.08, -thc)]
        cL = c if left else None; cR = c if not left else None; nL = nn if left else None; nR = nn if not left else None
        thL, thR = (thc, None) if left else (None, thc)
    piv = link(bpy.data.objects.new('piv', None)); piv.location = (X, y, Zp); piv.rotation_mode = 'XYZ'
    yo = -0.34 if kind == 'cat' else 0.0           # the long catapult arm runs in front of the rail (cross-arms to its ends)
    plank = box(((aR - aL) / 2, yo, 0), (aL + aR + 0.12, 0.2 if kind != 'cat' else 0.12, 0.05), M_PLANK if kind != 'cat' else M_BRASS, name='plank'); plank.parent = piv
    if kind == 'cat':
        for xx in (-aL, aR):
            arm = box((xx, yo / 2, -0.005), (0.06, abs(yo), 0.04), M_BRASS, name='catarm'); arm.parent = piv
    hub = obj(prim('cyl', M_BRASS, 16), 'hub'); hub.parent = piv; hub.rotation_euler = (math.pi / 2, 0, 0); hub.scale = (0.045, 0.045, 0.26)
    hub.location = (0, yo, 0)
    # post + A-frame stand down to the floor rail
    rod((X, y + yo - 0.1, Zp), (X - 0.12, y + yo - 0.1, FLOOR + 0.12), 0.022, M_WALNUT, 6)
    rod((X, y + yo + 0.1, Zp), (X + 0.12, y + yo + 0.1, FLOOR + 0.12), 0.022, M_WALNUT, 6)
    for hh, side, th_hit, nn, cc in ((hL, -1, thL, nL if hL is not None else None, cL), (hR, 1, thR, nR if hR is not None else None, cR)):
        if hh is None:
            if kind != 'cat':      # counterweight on the empty end
                cw = sphere((side * 0.3, 0, -0.02), 0.06, M_STEEL); cw.parent = piv
            continue
        a = aL if side < 0 else aR
        is_cup = hh['kind'] == 'chacup'
        mloc = np.array(roty(th_hit)).T @ np.array(frame_from_n(nn))       # pad orientation in the plank frame
        if is_cup:
            pd = obj(prim('bowl', M_PAD, 24, 12), 'cup'); pd.scale = (0.17, 0.17, 0.17)
            pd.matrix_parent_inverse = Matrix.Identity(4)
            pd.parent = piv; pd.location = (side * a, 0, H0 + PADT / 2 + R - 0.17 * 1.0 + 0.0)
        else:
            pd = obj(prim('cube', M_PAD), 'pad'); pd.parent = piv
            pd.matrix_basis = Matrix.Translation((side * a, 0, H0)) @ Matrix(mloc).to_4x4() @ Matrix.Diagonal((PADW, PADD, PADT, 1))
        ball = 'Z' if hh in zh else 'Ch'
        col = (1.0, 0.35, 0.12) if ball == 'Z' else (1.0, 0.85, 0.3)
        pd.color = (*col, 0)
        key_flash(pd, hh['t'], 1.0, tail=12)
        PADS.append(dict(o=pd, t=hh['t'], ball=ball, h=hh, c=cc, n=nn, cup=is_cup))
    # rotation keys (angle = +left-end-up, the same roty() as the solver)
    keys = sorted(keys)
    piv.rotation_euler = (0, keys[0][1], 0); piv.keyframe_insert('rotation_euler', index=1, frame=0)
    for t, a in keys:
        piv.rotation_euler = (0, a, 0); piv.keyframe_insert('rotation_euler', index=1, frame=max(1, int(round(t * FPS))))
    SAWS.append(dict(piv=piv, X=X, Z=Zp, aL=aL, aR=aR, keys=keys, kind=kind, x0=X - aL - 0.06, x1=X + aR + 0.06))
    return True
# catapults first
cat_pairs = []
for c in EV['cups']:
    hL = [h for h in chh if abs(h['t'] - c['t_cup']) < 1e-4][0]
    hR = [h for h in BP['Z'].hits if abs(h['t'] - c['t_throw']) < 1e-4][0]
    make_saw(hL, hR, 'cat'); used.add(id(hL)); used.add(id(hR))
for hz in zh:
    if id(hz) in used: continue
    m = [h for h in chh if id(h) not in used and abs(h['t'] - (hz['t'] + 1.5 * BEAT)) < 1e-3]
    if m and make_saw(hz, m[0]): used.add(id(hz)); used.add(id(m[0]))
for h in zh:
    if id(h) not in used: make_saw(h, None); used.add(id(h))
for h in chh:
    if id(h) not in used: make_saw(None, h); used.add(id(h))
# floor rail under the see-saws
xs_b = [s['X'] for s in SAWS]
box(((min(xs_b) + max(xs_b)) / 2, YL['bass'], FLOOR + 0.06), (max(xs_b) - min(xs_b) + 2.0, 0.5, 0.12), M_WALNUT, name='floorrail')
box(((min(xs_b) + max(xs_b)) / 2, YL['bass'] - 0.26, FLOOR + 0.13), (max(xs_b) - min(xs_b) + 2.0, 0.03, 0.03), M_BRASS)

# ---------------- hook: the player piano, coin chute, start lever, hopper ----------------
PX0, PX1 = -2.3, 1.15; PY0, PY1 = 0.45, 1.7; PTOP = EV['chute']['mouth'][2] - R - 0.03
box(((PX0 + PX1) / 2, (PY0 + PY1) / 2, (FLOOR + PTOP) / 2), (PX1 - PX0, PY1 - PY0, PTOP - FLOOR), M_PIANO, name='piano')
box(((PX0 + PX1) / 2, PY0 - 0.02, PTOP - 0.05), (PX1 - PX0 + 0.12, 0.08, 0.1), M_BRASS)
# keyboard
KZ = 1.55
box(((PX0 + PX1) / 2, PY0 - 0.25, KZ - 0.08), (PX1 - PX0 - 0.2, 0.5, 0.12), M_PIANO)
nk = 28; kw = (PX1 - PX0 - 0.4) / nk
KEYS = []
for k in range(nk):
    x = PX0 + 0.2 + kw * (k + 0.5)
    ko = box((x, PY0 - 0.27, KZ), (kw * 0.92, 0.42, 0.05), M_IVORY, name='key'); KEYS.append(ko)
    if k % 7 not in (2, 6) and k < nk - 1:
        box((x + kw / 2, PY0 - 0.2, KZ + 0.04), (kw * 0.55, 0.28, 0.05), M_KEYBLK)
# player-piano keys go down by themselves on the melody (the piano 'plays' what the balls play)
lead_t = [l[0] for l in LEAD]
for k, ko in enumerate(KEYS):
    ts = [l[0] for l in LEAD if (l[1] - 55) % nk == k]
    d = np.zeros(NF)
    for t in ts:
        f0 = int(round(t * FPS))
        for df in range(5):
            if 0 <= f0 + df < NF: d[f0 + df] = max(d[f0 + df], 0.025 * (1 - df / 5))
    key_series(ko, 'location', 2, KZ - d, 1e-4)
# roll window with a scrolling perforated paper roll
WZ0, WZ1 = 2.35, 3.95
box(((PX0 + PX1) / 2, PY0 - 0.03, (WZ0 + WZ1) / 2), (PX1 - PX0 - 0.5, 0.06, WZ1 - WZ0 + 0.16), M_BRASS)
M_ROLL = mat('roll', (0.9, 0.85, 0.7), 0.8); nt = M_ROLL.node_tree; bsdf = nt.nodes['Principled BSDF']
tc_ = nt.nodes.new('ShaderNodeTexCoord'); mp_ = nt.nodes.new('ShaderNodeMapping'); mp_.inputs['Scale'].default_value = (16, 1, 9)
vor = nt.nodes.new('ShaderNodeTexVoronoi'); vor.feature = 'F1'; vor.inputs['Scale'].default_value = 1.0
mr = nt.nodes.new('ShaderNodeMapRange'); mr.inputs['From Min'].default_value = 0.24; mr.inputs['From Max'].default_value = 0.27
mix_ = nt.nodes.new('ShaderNodeMix'); mix_.data_type = 'RGBA'; mix_.inputs[6].default_value = (0.05, 0.03, 0.02, 1); mix_.inputs[7].default_value = (0.9, 0.85, 0.7, 1)
nt.links.new(tc_.outputs['Object'], mp_.inputs['Vector']); nt.links.new(mp_.outputs[0], vor.inputs['Vector'])
nt.links.new(vor.outputs['Distance'], mr.inputs['Value']); nt.links.new(mr.outputs[0], mix_.inputs[0]); nt.links.new(mix_.outputs[2], bsdf.inputs['Base Color'])
roll_o = box(((PX0 + PX1) / 2, PY0 - 0.065, (WZ0 + WZ1) / 2), (PX1 - PX0 - 0.7, 0.01, WZ1 - WZ0), M_ROLL, name='roll')
off = np.zeros(NF)
for f in range(1, NF): off[f] = off[f - 1] + (0.05 if T[f] > 0.25 else 0.0) * (1.0 if T[f] < FINAL + 0.2 else 0.0)
for f in range(0, NF, 3):
    mp_.inputs['Location'].default_value = (0, 0, float(off[f])); mp_.inputs['Location'].keyframe_insert('default_value', frame=f)
# lamps on the piano front light up when the coin drops
M_PL = emis('pianolamp', (1.0, 0.6, 0.25), 1.0); em_pl = M_PL.node_tree.nodes['Emission']
for f, s in ((0, 0.3), (int((T0 - 0.05) * FPS), 0.3), (int(T0 * FPS) + 2, 9.0), (int(T0 * FPS) + 10, 5.0)):
    em_pl.inputs['Strength'].default_value = s; em_pl.inputs['Strength'].keyframe_insert('default_value', frame=f)
for x in np.linspace(PX0 + 0.25, PX1 - 0.25, 7):
    sphere((x, PY0 - 0.06, WZ1 + 0.25), 0.055, M_PL, (12, 6))
    sphere((x, PY0 - 0.06, WZ0 - 0.2), 0.045, M_PL, (12, 6))
text('自動ピアノ', ((PX0 + PX1) / 2, PY0 - 0.08, 1.05), 0.36, M_TXT_Y)
text('1曲 5セント', ((PX0 + PX1) / 2, PY0 - 0.08, 4.45), 0.24, M_TXT_W)
# coin chute along the piano top (open half-pipe), coin rolls out of its mouth
ch = EV['chute']; c0 = np.array(ch['start']); c1 = np.array(ch['mouth'])
box(((c0[0] + c1[0]) / 2 - 0.2, 0, c0[2] - R - 0.02), (c1[0] - c0[0] + 0.5, 0.36, 0.04), M_BRASS)
for yy in (-0.17, 0.17): box(((c0[0] + c1[0]) / 2 - 0.2, yy, c0[2] - R + 0.05), (c1[0] - c0[0] + 0.5, 0.025, 0.14), M_BRASS)
rod((c0[0] - 0.4, 0, c0[2] - R - 0.04), (c0[0] - 0.4, 0.5, PTOP), 0.03, M_BRASS)
# start lever (coin end <-> P's cup), pivot between
L1 = EV['lever1']; lc = np.array(L1['coin']); lp = np.array(L1['cup'])
piv1 = link(bpy.data.objects.new('piv1', None)); pv = (lc + lp) / 2 - np.array([0, 0, 0.22]); piv1.location = tuple(pv)
d1 = lp - lc; L_ = np.linalg.norm(d1[[0, 1]]); yaw = math.atan2(d1[1], d1[0])
piv1.rotation_mode = 'XYZ'; piv1.rotation_euler = (0, 0, yaw)
lev = box((0, 0, 0.0), (L_ + 0.3, 0.12, 0.05), M_WALNUT, name='lever'); lev.parent = piv1
lp_plate = box((-L_ / 2, 0, 0.22 - R - 0.03), (0.26, 0.26, 0.04), M_PAD, name='leverplate'); lp_plate.parent = piv1; lp_plate.color = (1.0, 0.8, 0.3, 0)
rod_ = obj(prim('cyl', M_BRASS, 8), 'stalk'); rod_.parent = piv1; rod_.location = (-L_ / 2, 0, 0.1); rod_.scale = (0.02, 0.02, 0.2)
cupo = obj(prim('bowl', M_BRASS, 24, 12), 'pcup'); cupo.parent = piv1; cupo.location = (L_ / 2, 0, 0.22 - 0.02); cupo.scale = (0.19, 0.19, 0.19)
rod2 = obj(prim('cyl', M_BRASS, 8), 'stalk'); rod2.parent = piv1; rod2.location = (L_ / 2, 0, 0.08); rod2.scale = (0.02, 0.02, 0.16)
key_flash(lp_plate, L1['t'], 1.0, tail=20)
f1 = int(round(L1['t'] * FPS))
for f, a in ((0, 0.0), (f1, 0.0), (f1 + 3, -0.16)):
    piv1.rotation_euler = (0, a, yaw); piv1.keyframe_insert('rotation_euler', index=1, frame=f)
rod((pv[0], pv[1], pv[2]), (pv[0], pv[1], FLOOR), 0.05, M_WALNUT, 8)
text('スタート', (pv[0], pv[1] - 0.45, pv[2] - 0.45), 0.2, M_TXT_Y)
# the two 40-degree chutes: ZUN and CHA wait behind a stop gate, the gate flips up, they roll down and fly out level
hz_ = EV['hopper']
for nm, lab in (('Z', 'ズン'), ('Ch', 'チャ')):
    g = np.array(hz_[nm]['gate']); mo = np.array(hz_[nm]['mouth']); th = hz_[nm]['theta']
    dch = np.array([math.cos(th), 0, -math.sin(th)]); nc = np.array([math.sin(th), 0, math.cos(th)])
    Lc = np.linalg.norm(mo - g)
    rot = (0, th, 0)
    ctr = (g + mo) / 2 - nc * (R + 0.025) - dch * 0.1
    box(tuple(ctr), (Lc + 0.35, 0.34, 0.05), M_BRASS, rot=rot, name='chute')
    for sy in (-0.18, 0.18): box(tuple(ctr + np.array([0, sy, 0]) + nc * 0.08), (Lc + 0.35, 0.025, 0.2), M_BRASS, rot=rot, name='chutewall')
    # curved lip at the bottom (level exit)
    box(tuple(mo - np.array([0.08, 0, R + 0.025])), (0.22, 0.34, 0.05), M_BRASS, name='chutelip')
    rod(tuple(ctr - nc * 0.05), (ctr[0], ctr[1], FLOOR), 0.03, M_BRASS, 6)
    # stop gate just below the waiting ball, hinged at its top: flips up when the cable pulls
    hp_ = g + dch * (R + 0.03) + nc * (2 * R - 0.02)
    hg = link(bpy.data.objects.new('hinge', None)); hg.location = tuple(hp_); hg.rotation_mode = 'XYZ'
    flap = box((0, 0, -0.17), (0.03, 0.3, 0.34), M_RED, name='gate'); flap.parent = hg; flap.rotation_euler = (0, th, 0)
    tg = hz_[nm]['t_open']; ang = np.zeros(NF)
    for f in range(NF):
        u = (T[f] - tg + 0.1) / 0.1
        if u > 0: ang[f] = min(1.0, u) * 1.4
    key_series(hg, 'rotation_euler', 1, -ang)
    # glass magazine tube above the waiting ball, label
    tube = obj(prim('cyl', M_GLASS, 24), 'tube'); tube.location = (g[0], g[1], g[2] + 0.45); tube.scale = (0.18, 0.18, 0.55)
    box((g[0], g[1], g[2] + 0.75), (0.44, 0.44, 0.08), M_WALNUT)
    text(lab, (g[0], g[1] - 0.3, g[2] + 1.0), 0.28, M_TXT_Y if nm == 'Z' else M_TXT_W)
# cable from the start lever to the hopper gates
rod(pv + np.array([0.0, 0, -0.1]), np.array(hz_['Ch']['gate']) + np.array([0.0, 0, 0.3]), 0.008, M_DARK, 4)

# ---------------- balcony with the swing doors (C's entrance) ----------------
bl = EV['balcony']; bz = bl['edge'][2] - R; bx0 = bl['start'][0] - 0.8; bx1 = bl['edge'][0]
box(((bx0 + bx1) / 2, YL['C'], bz - 0.05), (bx1 - bx0, 0.38, 0.1), M_PLANK, name='balcony')
box(((bx0 + bx1) / 2, YL['C'] - 0.2, bz + 0.25), (bx1 - bx0, 0.03, 0.03), M_WALNUT)
for xx in np.arange(bx0 + 0.1, bx1, 0.25): rod((xx, YL['C'] - 0.2, bz), (xx, YL['C'] - 0.2, bz + 0.25), 0.012, M_WALNUT, 6)
for xx in (bx0 + 0.1, bx1 - 0.1): rod((xx, YL['C'], bz + 0.05), (xx, YL['C'], 7.6), 0.02, M_DARK, 6)
dx = bl['door'][0]; td = bl['t_door']
for yy in (YL['C'] - 0.24, YL['C'] + 0.22):
    rod((dx, yy, bz), (dx, yy, bz + 0.95), 0.035, M_WALNUT, 8)
box((dx, YL['C'] - 0.01, bz + 0.98), (0.12, 0.6, 0.1), M_WALNUT)
text('SALOON', (dx - 0.07, YL['C'] - 0.01, bz + 1.15), 0.17, M_TXT_Y, rot=(math.pi / 2, 0, 0))
for side, yh in ((1, YL['C'] - 0.22), (-1, YL['C'] + 0.2)):        # batwing leaves hinged at the frame posts
    hg = link(bpy.data.objects.new('dhinge', None)); hg.location = (dx, yh, bz + 0.15)
    leaf = box((0, side * 0.105, 0.24), (0.035, 0.2, 0.42), M_PLANK, name='leaf'); leaf.parent = hg
    # the ball pushes the leaf: smallest opening angle that keeps the leaf (hinge -> tip, 0.2 m) clear of the ball
    ang = np.zeros(NF); push_end = None; phi = 0.0
    def seg_d(ph, pc):
        a_ = np.array([dx, yh]); b_ = a_ + np.array([0.2 * math.sin(ph), side * 0.2 * math.cos(ph)])
        ab = b_ - a_; u_ = max(0, min(1, (pc - a_) @ ab / (ab @ ab))); return np.linalg.norm(pc - (a_ + u_ * ab))
    # leaves fly open just as the ball arrives (fast, ahead of it), then swing back and forth, damped
    t_touch = td - (R + 0.03) / bl['u']
    for f in range(NF):
        u = T[f] - (t_touch - 0.07)
        if u <= 0: continue
        t_clear = td + (R + 0.25) / bl['u']                                  # the ball is through the doorway
        if u < 0.12: ang[f] = 1.5 * math.sin(0.5 * math.pi * u / 0.12)
        elif T[f] < t_clear: ang[f] = 1.5
        else:
            w_ = T[f] - t_clear; ang[f] = 1.5 * math.cos(w_ * 6.0) * math.exp(-w_ / 0.5)
    hg.rotation_mode = 'XYZ'; key_series(hg, 'rotation_euler', 2, -side * ang, 1e-3)

# ---------------- mug ropeway ----------------
MP = EV['mug']['path']
mt = np.array([m[0] for m in MP]); mpnt = np.array([m[1] for m in MP])
ext0 = mpnt[0] + np.array([-80, 0, 0]); ext1 = mpnt[-1] + np.array([80, 0, 0.0])
vx_ = EV['mug']['vxm']
mt = np.concatenate([[mt[0] - 80 / vx_], mt, [mt[-1] + 80 / vx_]]); mpnt = np.vstack([ext0, mpnt, ext1])
def mug_at(t):
    t = min(max(t, mt[0]), mt[-1]); k = int(np.searchsorted(mt, t) - 1); k = max(0, min(k, len(mt) - 2))
    u = (t - mt[k]) / (mt[k + 1] - mt[k]); return mpnt[k] + (mpnt[k + 1] - mpnt[k]) * u
HANG = 0.55
for a, b in zip(mpnt[:-1], mpnt[1:]):
    rod(a + [0, 0, HANG], b + [0, 0, HANG], 0.012, M_DARK, 6)
for q in mpnt[1:-1]:
    wh = obj(prim('cyl', M_BRASS, 24), 'pulley'); wh.location = q + [0, 0, HANG]; wh.rotation_euler = (math.pi / 2, 0, 0); wh.scale = (0.12, 0.12, 0.05)
    rod(q + [0, 0, HANG], (q[0], q[1] + 0.9, 7.6), 0.02, M_DARK, 6)
T_TIP = EV['mug']['t_tip']
MUGR = 0.19; MUGH = 0.38
for kk, dt_ in enumerate((0.0, -3.2, 3.2, 6.4, -6.4)):
    root = link(bpy.data.objects.new('mug', None)); root.rotation_mode = 'XYZ'
    PM = np.array([mug_at(t + dt_) for t in T])
    key_loc(root, PM)
    # body: glass cylinder + beer + foam, hanger
    g_ = obj(prim('cyl', M_BEER, 24), 'mugglass'); g_.parent = root; g_.scale = (MUGR, MUGR, MUGH / 2); g_.location = (0, 0, MUGH / 2 - R - 0.03)
    bb = obj(prim('cyl', M_BEER, 24), 'beer'); bb.parent = root; bb.scale = (MUGR - 0.02, MUGR - 0.02, 0.12); bb.location = (0, 0, 0.12 - R - 0.02)
    fo = obj(prim('cyl', M_FOAM, 24), 'foam'); fo.parent = root; fo.scale = (MUGR - 0.015, MUGR - 0.015, 0.03); fo.location = (0, 0, 0.27 - R)
    hdl = obj(prim('torus', M_BRASS, 0.2, 24), 'handle'); hdl.parent = root; hdl.scale = (0.1, 0.1, 0.1); hdl.rotation_euler = (math.pi / 2, 0, 0); hdl.location = (0, MUGR + 0.06, MUGH / 2 - R)
    rod_h = obj(prim('cyl', M_DARK, 6), 'hanger'); rod_h.parent = root; rod_h.scale = (0.01, 0.01, (HANG - (MUGH - R)) / 2); rod_h.location = (0, 0, (HANG + MUGH - R) / 2)
    # tipping at the unloading cam
    ang = np.zeros(NF)
    for f in range(NF):
        u = (T[f] + dt_ - (T_TIP - 0.12)) / 0.12
        if u > 0: ang[f] = min(1.0, u) * 1.9 * (1 if T[f] + dt_ < T_TIP + 1.6 else max(0.0, 1 - (T[f] + dt_ - T_TIP - 1.6) / 0.5))
    key_series(root, 'rotation_euler', 1, ang, 1e-3)
# splash when P lands in the mug
spl = sphere((0, 0, 0), 0.25, M_FX); spl.color = (1.0, 0.9, 0.6, 0)
ft = int(round(EV['mug']['t_load'] * FPS)); sa = np.zeros(NF); ss = np.full(NF, 0.01)
for df in range(14):
    if ft + df < NF: sa[ft + df] = 0.8 * (1 - df / 14) ** 1.5; ss[ft + df] = 0.3 + 0.8 * (1 - (1 - df / 14) ** 2)
spl.location = tuple(np.array(MP[1][1]) + [0, 0, 0.15])
key_series(spl, 'color', 3, sa, 1e-3)
for k in range(3): key_series(spl, 'scale', k, ss * 0.5)

# ---------------- jukebox + troughs ----------------
J = EV['juke']; jp = J['p']
hx = [J['p'][n]['hole'][0] for n in ('P', 'C')]; hz = min(J['p'][n]['hole'][2] for n in ('P', 'C'))
JX0, JX1 = min(hx) - 0.35, max(hx) + 0.45; JTOP = hz - 0.12
M_JUKE = glowmat('jukebody', gain=2.2, base=0.0, rough=0.2, coat=1.0)
jb = box(((JX0 + JX1) / 2, 0.0, (FLOOR + JTOP) / 2), (JX1 - JX0, 0.9, JTOP - FLOOR), M_PIANO, name='jukelow')
# tall arch behind
ARC_X = (JX0 + JX1) / 2
arch = box((ARC_X, 0.85, (FLOOR + 3.6) / 2), (JX1 - JX0 + 0.3, 0.7, 3.6 - FLOOR), M_PIANO, name='jukearch')
dome_ = obj(prim('dome', M_PIANO, 32, 16), 'archdome'); dome_.location = (ARC_X, 0.85, 3.6); dome_.scale = ((JX1 - JX0 + 0.3) / 2, 0.35, 0.8)
JB = []
for k, (yy, rr) in enumerate(((0.48, 0.0), (0.47, 0.18), (0.46, 0.36))):
    for side in (-1, 1):
        st = box((ARC_X + side * ((JX1 - JX0) / 2 - 0.05 - rr * 0.3), yy, 2.0), (0.08, 0.04, 3.0 - rr), M_JUKE, name='jband'); JB.append(st)
    arc = obj(prim('torus', M_JUKE, 0.05, 48), 'jarc'); arc.location = (ARC_X, yy, 3.55); arc.rotation_euler = (math.pi / 2, 0, 0)
    arc.scale = ((JX1 - JX0) / 2 - 0.05 - rr * 0.3,) * 3; JB.append(arc)
grill = box((ARC_X, 0.47, 1.55), (JX1 - JX0 - 0.5, 0.03, 1.2), M_BRASS, name='grill')
text('ジュークボックス', (ARC_X, 0.44, 2.55), 0.2, M_TXT_W)
# top slot + lid (closes after the balls drop in)
box((ARC_X, -0.47, JTOP + 0.03), (JX1 - JX0, 0.06, 0.06), M_BRASS); box((ARC_X, 0.47, JTOP + 0.03), (JX1 - JX0, 0.06, 0.06), M_BRASS)
lh = link(bpy.data.objects.new('lidh', None)); lh.location = (ARC_X, 0.42, JTOP + 0.06)
lid = box((0, -0.42, 0.0), (JX1 - JX0 - 0.05, 0.84, 0.04), M_PIANO, name='lid'); lid.parent = lh
TL = J['t_in'] + 0.35; ang = np.zeros(NF)
for f in range(NF):
    u = (T[f] - TL) / 0.22
    ang[f] = 1.75 if u < 0 else max(0.0, 1.75 * (1 - u * u)) if u < 1 else 0.06 * math.exp(-(u - 1) * 3) * abs(math.sin((u - 1) * 10))
lh.rotation_mode = 'XYZ'; key_series(lh, 'rotation_euler', 0, -ang, 1e-3)
# jukebox glow: off until the final chord, then rainbow bands
for k, o in enumerate(JB):
    import colorsys
    for f in range(0, NF):
        t = T[f]
        if t < FINAL - 0.05: a = 0.04
        else: a = min(1.0, (t - FINAL + 0.05) / 0.15)
        if f % 2 and 0 < f < NF - 1 and t < FINAL - 0.2: continue
        h_ = (k * 0.17 + max(0, t - FINAL) * 0.35) % 1.0; r_, g_, b_ = colorsys.hsv_to_rgb(h_, 0.8, 1.0)
        o.color = (r_, g_, b_, a); o.keyframe_insert('color', frame=f)
# felt troughs for ZUN / CHA at the end of the rail
for nm in ('Z', 'Ch'):
    tp = np.array(EV['trough'][nm]['p'])
    box((tp[0], tp[1], tp[2] - R - 0.03), (0.5, 0.4, 0.05), M_FELT, name='trough')
    for sx in (-1, 1): box((tp[0] + sx * 0.25, tp[1], tp[2] - R + 0.04), (0.04, 0.4, 0.16), M_WALNUT)
    rod((tp[0], tp[1], tp[2] - R - 0.05), (tp[0], tp[1], FLOOR), 0.04, M_WALNUT, 6)
# comment question on a chalkboard, flips in after the final chord
cb = link(bpy.data.objects.new('cboard', None)); cb.location = (ARC_X - 0.9, 1.6, 5.35)
bd = box((0, 0, 0), (4.5, 0.06, 1.05), mat('chalk', (0.05, 0.1, 0.07), 0.9), name='chalkboard'); bd.parent = cb
fr = box((0, 0.01, 0), (4.64, 0.05, 1.19), M_WALNUT); fr.parent = cb
tq = text('この曲、どこで聴いた？', (0, -0.05, 0), 0.38, M_TXT_W); tq.parent = cb; tq.location = (0, -0.05, 0)
scb = np.full(NF, 0.001)
for f in range(NF):
    u = (T[f] - FINAL - 0.5) / 0.35
    if u > 0: scb[f] = min(1.0, u) + 0.06 * math.sin(10 * u) * math.exp(-2 * u)
for k in range(3): key_series(cb, 'scale', k, np.maximum(0.001, scb), 1e-3)
# confetti from the jukebox top
FLAGC = [(0.95, 0.2, 0.2), (1, 0.8, 0.2), (0.2, 0.6, 1.0), (0.3, 0.9, 0.4), (1, 1, 1), (0.9, 0.4, 0.9)]
for k in range(80):
    cf = obj(prim('plane', M_OBJ), 'conf'); cf.color = (*FLAGC[k % 6], 1)
    ang0 = rng.uniform(0.3, math.pi - 0.3); sp = rng.uniform(1.5, 3.5); vy = rng.uniform(-0.8, 0.5); t0c = FINAL + 0.1 + rng.uniform(0, 0.3)
    P = np.zeros((NF, 3)); RT = np.zeros((NF, 3))
    for f in range(NF):
        s_ = T[f] - t0c
        if s_ < 0: P[f] = (ARC_X, 0.5, 3.8); continue
        P[f] = (ARC_X + sp * math.cos(ang0) * s_ * 0.7, 0.5 + vy * s_, 3.8 + sp * math.sin(ang0) * s_ - 2.0 * s_ * s_ / (1 + s_) - 0.4 * s_)
        RT[f] = (s_ * (3 + k % 5), s_ * (2 + k % 3), s_ * 1.5)
    key_loc(cf, P)
    sc_ = np.where(T >= t0c, 1.0, 0.001)
    for c_ in range(3): key_series(cf, 'scale', c_, sc_ * 0.09); key_series(cf, 'rotation_euler', c_, RT[:, c_])

# ---------------- the saloon ----------------
XMIN, XMAX = -8.0, max(JX1, max(xs_b)) + 10.0
fl = obj(prim('plane', M_FLOOR), 'floor'); fl.location = ((XMIN + XMAX) / 2, -2.0, FLOOR); fl.scale = (XMAX - XMIN, 14, 1)
WY = 3.3
wl = obj(prim('plane', M_WALL), 'wall'); wl.location = ((XMIN + XMAX) / 2, WY, 3.6); wl.rotation_euler = (math.pi / 2, 0, 0); wl.scale = (XMAX - XMIN, 8.2, 1)
box(((XMIN + XMAX) / 2, WY - 0.05, 0.6), (XMAX - XMIN, 0.06, 1.9), M_VELVET, name='wainscot')
box(((XMIN + XMAX) / 2, WY - 0.08, 1.58), (XMAX - XMIN, 0.08, 0.06), M_BRASS)
for x in np.arange(XMIN, XMAX, 6.0):          # ceiling beams + pilasters
    box((x, 0.5, 7.75), (0.3, 9.0, 0.3), M_WALNUT)
    box((x, WY - 0.12, 3.6), (0.28, 0.2, 8.0), M_WALNUT)
# back bar: shelves with bottles behind the counters
BOTC = [(0.15, 0.4, 0.15), (0.45, 0.22, 0.06), (0.6, 0.55, 0.3), (0.12, 0.25, 0.35), (0.5, 0.1, 0.1)]
M_BOT = [mat(f'bot{k}', c, 0.05, **{'Coat Weight': 1.0}) for k, c in enumerate(BOTC)]
for z in (2.15, 2.85):
    box(((XMIN + XMAX) / 2, WY - 0.3, z - 0.03), (XMAX - XMIN, 0.42, 0.05), M_WALNUT)
for x in np.arange(XMIN + 0.3, XMAX, 0.42):
    for z in (2.15, 2.85):
        k = int(rng.integers(0, 5)); hgt = rng.uniform(0.28, 0.42)
        bo = obj(prim('cyl', M_BOT[k], 12), 'bottle'); bo.location = (x + rng.uniform(-0.05, 0.05), WY - 0.3, z + hgt / 2); bo.scale = (0.06, 0.06, hgt / 2)
        nk_ = obj(prim('cyl', M_BOT[k], 8), 'neck'); nk_.location = (bo.location[0], WY - 0.3, z + hgt + 0.06); nk_.scale = (0.022, 0.022, 0.07)
# hanging oil lamps (+ real point lights), brighten for the finale
LAMPS = []
for x in np.arange(XMIN + 2, XMAX, 5.0):
    sphere((x, 0.9, 6.55), 0.16, M_LAMP, (16, 8)); rod((x, 0.9, 6.7), (x, 0.9, 7.6), 0.012, M_DARK, 4)
    box((x, 0.9, 6.78), (0.36, 0.36, 0.05), M_BRASS)
    ld = bpy.data.lights.new('lamp', 'POINT'); ld.energy = 420; ld.color = (1.0, 0.68, 0.36); ld.shadow_soft_size = 0.35
    lo = link(bpy.data.objects.new('lamp', ld)); lo.location = (x, 0.6, 6.3); LAMPS.append(ld)
# posters & signs on the back wall (search keywords in the first 30 s)
def poster(x, z, lines, w=1.5, h=1.15, m_txt=M_TXT_D, sizes=None, paper=M_PAPER):
    box((x, WY - 0.06, z), (w, 0.02, h), paper, name='poster')
    box((x, WY - 0.05, z), (w + 0.08, 0.015, h + 0.08), M_WALNUT)
    n_ = len(lines)
    for k, s in enumerate(lines):
        sz = (sizes or [0.26] * n_)[k]
        text(s, (x, WY - 0.08, z + (n_ - 1) * 0.5 * 0.34 - k * 0.34), sz, m_txt)
# big neon title above the piano
box((0.0, WY - 0.1, 6.15), (5.6, 0.06, 1.75), M_DARK, name='signboard')
box((0.0, WY - 0.09, 6.15), (5.72, 0.04, 1.87), M_BRASS)
text('ジ・エンターテイナー', (0.0, WY - 0.16, 6.45), 0.72, M_NEON_P)
text('スコット・ジョプリン（1902年）', (0.0, WY - 0.16, 5.7), 0.4, M_TXT_Y)
poster(4.2, 4.35, ['ラグタイム', 'RAGTIME'], sizes=[0.36, 0.22])
poster(9.6, 4.35, ['からくり', 'ビー玉'], sizes=[0.36, 0.36], m_txt=M_TXT_RD)
poster(15.4, 4.35, ['西部の酒場', 'SALOON'], sizes=[0.32, 0.24])
poster(21.2, 4.35, ['ピアノ名曲', 'ジョプリン'], sizes=[0.32, 0.3], m_txt=M_TXT_RD)
poster(27.0, 4.35, ['ビー玉', '自動ピアノ'], sizes=[0.4, 0.3])
poster(33.0, 4.35, ['ラグタイム', 'ピアノ'], sizes=[0.36, 0.34], m_txt=M_TXT_RD)
# WANTED poster: the hat ball (foreshadows the stranger)
wx = 39.0
poster(wx, 4.35, ['WANTED', '', '指名手配'], sizes=[0.3, 0.2, 0.26])
sphere((wx, WY - 0.1, 4.33), 0.17, M_RED, (24, 12))
whb = obj(prim('cyl', M_HAT, 24), 'whb'); whb.location = (wx, WY - 0.12, 4.47); whb.scale = (0.26, 0.26, 0.015); whb.rotation_euler = (0.25, 0, 0)
whc = obj(prim('cone', M_HAT, 24, 1.0, 0.8, 1.0), 'whc'); whc.location = (wx, WY - 0.12, 4.55); whc.scale = (0.13, 0.11, 0.1)
poster(45.0, 4.35, ['ジ・エンター', 'テイナー'], sizes=[0.32, 0.32])
for x, s in ((60, 'BAR'), (80, 'SALOON'), (100, 'BAR'), (118, 'ラグタイム')):
    box((x, WY - 0.08, 4.45), (2.0, 0.04, 0.7), M_DARK)
    text(s, (x, WY - 0.13, 4.45), 0.38, M_NEON_C if s != 'ラグタイム' else M_NEON_P)
# wagon wheels
for x in np.arange(12.5, XMAX - 4, 17.0):
    ww = obj(prim('torus', M_WALNUT, 0.08, 32), 'wheel'); ww.location = (x, WY - 0.1, 5.7); ww.rotation_euler = (math.pi / 2, 0, 0); ww.scale = (0.6, 0.6, 0.6)
    for k in range(8):
        a_ = k * math.pi / 8; rod((x - 0.58 * math.cos(a_), WY - 0.1, 5.7 - 0.58 * math.sin(a_)), (x + 0.58 * math.cos(a_), WY - 0.1, 5.7 + 0.58 * math.sin(a_)), 0.018, M_WALNUT, 6)

# ---------------- world & key lights ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes['Background']; bg.inputs['Color'].default_value = (0.05, 0.03, 0.02, 1); bg.inputs['Strength'].default_value = 0.35
key = bpy.data.lights.new('key', 'AREA'); key.energy = 1300; key.size = 6.0; key.color = (1.0, 0.85, 0.65)
KEY = link(bpy.data.objects.new('key', key))
fill = bpy.data.lights.new('fill', 'AREA'); fill.energy = 260; fill.size = 10.0; fill.color = (0.6, 0.7, 1.0)
FILL = link(bpy.data.objects.new('fill', fill))

# ---------------- camera: follows the music along the saloon ----------------
def act_x(t):
    if t < tb(7.0): return BP['coin'].at(t)[0]
    if t < EV['mug']['t_load']: xm = BP['P'].at(t)[0]
    elif t < tb(71.5): xm = BP['C'].at(t)[0]
    else: xm = 0.5 * (BP['P'].at(t)[0] + BP['C'].at(t)[0])
    xb = 0.5 * (BP['Z'].at(t)[0] + BP['Ch'].at(t)[0] + LAY['S'] * 0.5) if t > EV['hopper']['Z']['t_open'] else xm
    return 0.55 * xm + 0.45 * xb
FOC = np.zeros((NF, 3)); WID = np.zeros(NF)
for f in range(NF):
    t = T[f]
    if t < T0 + 0.6:
        FOC[f] = (0.0, 0, 3.55); WID[f] = 4.8
    elif t > FINAL + 0.5:
        FOC[f] = (ARC_X - 0.9, 0, 2.75); WID[f] = 5.6
    else:
        FOC[f] = (act_x(t) + 0.35, 0, 2.55 if t > tb(7.0) else 3.3); WID[f] = 4.3
FOC[:, 0] = smooth(FOC[:, 0], 0.45); FOC[:, 2] = smooth(FOC[:, 2], 0.6); WID = smooth(WID, 0.6)
LENS = 30.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 400
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24
    CP[f] = (FOC[f, 0], -D, FOC[f, 2] + 0.12 * D)
    TP[f] = (FOC[f, 0], 0.0, FOC[f, 2] - 0.02 * D)
key_loc(cam, CP); key_loc(tgt, TP)
KP = CP + np.array([1.5, 2.5, 5.0]); key_loc(KEY, KP); KEY.rotation_euler = (math.radians(35), 0, 0)
FP = CP + np.array([-4.0, 3.0, 1.0]); key_loc(FILL, FP); FILL.rotation_euler = (math.radians(80), 0, math.radians(-30))
# the lamps brighten at the finale
for f in (0, int((FINAL - 0.05) * FPS), int((FINAL + 0.3) * FPS)):
    for ld in LAMPS:
        ld.energy = 420 if f < FINAL * FPS else 800; ld.keyframe_insert('energy', frame=f)


# ---------------- sanity checks ----------------
def check():
    info = {}
    gaps = []
    for b_ in BELLS:
        f = int(round(b_['t'] * FPS)); p = POS[b_['ball']][min(NF - 1, f)]
        gaps.append(np.linalg.norm(p - b_['c']) - (R + RB))
    info['bell_contact_gap_at_frame_max'] = round(float(np.max(np.abs(gaps))), 3)
    rg = []
    for r_ in RAILS:
        f = int(round(r_['t'] * FPS)); p = POS[r_['ball']][min(NF - 1, f)]
        rg.append(abs(p[2] - R - r_['z']))
    info['rail_contact_gap_at_frame_max'] = round(float(max(rg)), 3)
    # pads: world position of the pad centre at the hit frame vs planned
    dg = []
    sc.frame_set(0)
    for pd in PADS:
        if pd['cup']: continue
        f = int(round(pd['t'] * FPS)); sc.frame_set(f)
        wpos = np.array(pd['o'].matrix_world.translation); dg.append(np.linalg.norm(wpos - pd['c']))
    info['pad_centre_error_at_hit_frame_max'] = round(float(max(dg)), 3) if dg else None
    # moving parts vs balls, every frame: pads / cups / planks (oriented boxes) and the beer mugs
    movers = [o for o in bpy.data.objects if o.name.startswith(('pad', 'plank', 'catarm', 'cup', 'leverplate', 'lever', 'pcup', 'gate', 'leaf'))]
    own = {}
    for pd in PADS: own.setdefault(pd['o'].name, []).append((pd['ball'], pd['t']))
    hits_pp = 0; worst = 0.0; ex = []
    for f in range(0, NF):
        sc.frame_set(f); t = T[f]
        for o in movers:
            Mi = o.matrix_world.inverted()
            for n in BALL:
                p = POS[n][f]
                loc = Mi @ Vector(p)
                # unit cube (+-0.5) in local space; scale back to world distance approximately
                sx, sy, sz = o.matrix_world.to_scale()
                d = Vector((max(0, abs(loc.x) - 0.5) * sx, max(0, abs(loc.y) - 0.5) * sy, max(0, abs(loc.z) - 0.5) * sz)).length
                if o.name.startswith(('cup', 'pcup')): d = max(0.0, (Vector(p) - o.matrix_world.translation).length - 0.17 - R) if loc.z > -0.2 else 9
                if d < R - 0.015:
                    mine = any(b_ == n and abs(t - th) < 0.07 for b_, th in own.get(o.name, []))
                    if o.name.startswith(('cup', 'pcup')) and n in ('Ch', 'P'): mine = True      # balls resting in their cups
                    if o.name.startswith('leverplate') and n == 'coin': mine = True
                    if not mine:
                        hits_pp += 1; worst = max(worst, R - d)
                        if len(ex) < 6: ex.append((o.name, n, round(t, 2), round(R - d, 3)))
    info['moving_parts_collisions'] = hits_pp; info['moving_parts_worst_pen'] = round(worst, 3); info['moving_parts_examples'] = ex
    ss = sorted([s_ for s_ in SAWS if s_['kind'] != 'cat'], key=lambda s_: s_['x0'])
    info['seesaw_plank_overlaps'] = sum(1 for a_, b_ in zip(ss, ss[1:]) if b_['x0'] < a_['x1'] - 0.01)
    info['bells'] = len(BELLS); info['rails'] = len(RAILS); info['pads'] = len(PADS); info['saws'] = len(SAWS)
    info['objects'] = len(bpy.data.objects); info['NF'] = NF
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
