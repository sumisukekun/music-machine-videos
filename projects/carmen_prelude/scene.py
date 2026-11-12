"""Carmen Prelude - 闘牛場のビー玉 (marbles in the bullring), side view of a long bullring at sunset.
Every melody note = the two toreros (red + gold) landing on a xylophone key of that pitch (one wide key for a shared note,
two keys for a third / octave); every bass note = the bull (black, horns) landing on a drum; every off-beat chord = the
two spectators landing on two brass bells in the stands; the tremolo = the crowd wave, one marble per note on the chime
rails.  layout.py plans all motion (constant gravity, bounces that never add speed, ramp / trebuchet mechanisms); this file
builds and animates it.  jp_sub.otf = Noto Sans CJK JP Black (SIL OFL) subset to the characters used here (make_font.py).
python3 scene.py -- test t1 t2 ... | range f0 f1"""
import os, sys, json, math, time
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
LAY = json.load(open(f'{HERE}/layout.json'))
FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.5) * FPS); T = np.arange(NF) / FPS
G = LAY['G']; EV = LAY['ev']; LY = LAY['lane_y']; DYP = LAY['dyp']; KW = LAY['kw']
RP, RB, RC, RM = LAY['R']['pair'], LAY['R']['bull'], LAY['R']['spec'], LAY['R']['marble']
SECS = E['secs']; T0 = E['t0']
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
sc.cycles.transparent_max_bounces = 6
sc.render.resolution_x, sc.render.resolution_y = 1080, 1920
sc.render.resolution_percentage = int(os.environ.get('PCT', 50)); sc.render.fps = FPS
sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.5
try: sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Punchy'
except Exception: pass
sc.frame_start, sc.frame_end = 0, NF - 1
FONT = bpy.data.fonts.load(f'{HERE}/jp_sub.otf')
FONT_EN = bpy.data.fonts.load(f'{HERE}/en_bold.ttf')
TXT_K = 2.75


def link(o): sc.collection.objects.link(o); return o


def mat(name, col, rough=0.5, **kw):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']; b.inputs['Base Color'].default_value = (*col, 1); b.inputs['Roughness'].default_value = rough
    for k, v in kw.items():
        if k == 'Emission Color': v = (*v, 1)
        b.inputs[k].default_value = v
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
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    oi = nt.nodes.new('ShaderNodeObjectInfo'); em = nt.nodes.new('ShaderNodeEmission'); tr_ = nt.nodes.new('ShaderNodeBsdfTransparent')
    mx = nt.nodes.new('ShaderNodeMixShader'); mu = nt.nodes.new('ShaderNodeMath'); mu.operation = 'MULTIPLY'; mu.inputs[1].default_value = gain
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(oi.outputs['Color'], em.inputs['Color']); nt.links.new(oi.outputs['Alpha'], mu.inputs[0]); nt.links.new(mu.outputs[0], em.inputs['Strength'])
    nt.links.new(oi.outputs['Alpha'], mx.inputs['Fac']); nt.links.new(tr_.outputs[0], mx.inputs[1]); nt.links.new(em.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs[0]); return m


def objcol(name, rough=0.45, metal=0.0, emit=0.0, coat=0.0):
    m = mat(name, (1, 1, 1), rough, **{'Metallic': metal, 'Coat Weight': coat}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    oi = nt.nodes.new('ShaderNodeObjectInfo'); nt.links.new(oi.outputs['Color'], b.inputs['Base Color'])
    if emit:
        nt.links.new(oi.outputs['Color'], b.inputs['Emission Color']); b.inputs['Emission Strength'].default_value = emit
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
    elif kind == 'bcube':
        bmesh.ops.create_cube(bm, size=1.0); bmesh.ops.bevel(bm, geom=bm.edges[:] + bm.verts[:], offset=0.18, segments=3, affect='EDGES')
    elif kind == 'cyl': bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=1.0, radius2=1.0, depth=1.0)
    elif kind == 'cone': bmesh.ops.create_cone(bm, cap_ends=True, segments=a[0], radius1=a[1], radius2=a[2], depth=a[3])
    elif kind == 'sph': bmesh.ops.create_uvsphere(bm, u_segments=a[0], v_segments=a[1], radius=1.0)
    elif kind == 'plane': bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    elif kind == 'bell':                                   # bell, mouth down, crown flat at z = 0, height 1, mouth radius 1
        prof = [(0.0, 0.0), (0.42, 0.0), (0.5, -0.08), (0.55, -0.35), (0.62, -0.62), (0.8, -0.86), (1.0, -1.0), (0.96, -1.0), (0.78, -0.88), (0.58, -0.64),
                (0.5, -0.36), (0.44, -0.1), (0.0, -0.1)]
        segs = a[0]; rings = []
        for k in range(segs):
            ph = 2 * math.pi * k / segs
            rings.append([bm.verts.new((r * math.cos(ph), r * math.sin(ph), z)) for r, z in prof])
        for k in range(segs):
            r0, r1 = rings[k], rings[(k + 1) % segs]
            for j in range(len(prof) - 1): bm.faces.new((r0[j], r1[j], r1[j + 1], r0[j + 1]))
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    elif kind == 'bowl':                                   # open bowl (cup), rim at z = 0, radius 1, depth 0.7
        prof = [(1.0, 0.0), (0.97, -0.3), (0.85, -0.55), (0.6, -0.7), (0.0, -0.72), (0.0, -0.8), (0.65, -0.78), (0.92, -0.6), (1.05, -0.3), (1.08, 0.0)]
        segs = a[0]; rings = []
        for k in range(segs):
            ph = 2 * math.pi * k / segs
            rings.append([bm.verts.new((r * math.cos(ph), r * math.sin(ph), z)) for r, z in prof])
        for k in range(segs):
            r0, r1 = rings[k], rings[(k + 1) % segs]
            for j in range(len(prof) - 1): bm.faces.new((r0[j], r1[j], r1[j + 1], r0[j + 1]))
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    elif kind == 'arch':                                   # half ring (arch) in the x-z plane, radius 1.05
        segs = 20; rings = []
        for i in range(segs + 1):
            ph = math.pi * i / segs; c = Vector((1.05 * math.cos(ph), 0, 1.05 * math.sin(ph)))
            rings.append([bm.verts.new(c + Vector((0.12 * math.cos(ph) * math.cos(2 * math.pi * k / 6), 0.12 * math.sin(2 * math.pi * k / 6), 0.12 * math.sin(ph) * math.cos(2 * math.pi * k / 6)))) for k in range(6)])
        for i in range(segs):
            for k in range(6): bm.faces.new((rings[i][k], rings[i][(k + 1) % 6], rings[i + 1][(k + 1) % 6], rings[i + 1][k]))
    elif kind == 'horn':                                   # curved bull horn: base at origin, curving up and forward
        segs = 10; rings = []
        for i in range(9):
            u = i / 8; ang = u * 1.5
            c = Vector((0.0, 0.55 * math.sin(ang) * 0.6, 0.55 * (1 - math.cos(ang)) + 0.4 * u))
            r = 0.16 * (1 - u) + 0.012
            rings.append([bm.verts.new(c + Vector((r * math.cos(2 * math.pi * k / segs), 0, r * math.sin(2 * math.pi * k / segs)))) for k in range(segs)])
        for i in range(8):
            for k in range(segs): bm.faces.new((rings[i][k], rings[i][(k + 1) % segs], rings[i + 1][(k + 1) % segs], rings[i + 1][k]))
    me = bpy.data.meshes.new(kind); bm.to_mesh(me); bm.free()
    if kind in ('sph', 'cone', 'bell', 'cyl', 'bcube', 'bowl', 'horn', 'arch'):
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


def text(s, loc, size, m, rot=(math.pi / 2 - 0.3, 0, 0), align='CENTER', ext=0.02):
    cu = bpy.data.curves.new('t', 'FONT'); cu.body = s; cu.font = FONT_EN if _latin(s) else FONT; cu.size = size * TXT_K * (EN_K if _latin(s) else 1.0); cu.extrude = ext
    cu.align_x = align; cu.align_y = 'CENTER'
    o = link(bpy.data.objects.new('txt', cu)); o.location = loc; o.rotation_euler = rot; o.data.materials.append(m); return o


# English-first on-screen text (2026-11-12): Latin strings use DejaVu Sans Bold (en_bold.ttf, free DejaVu font license),
# Japanese subtitles keep the Noto CJK subset.
def _latin(s): return all(ord(c) < 0x2E80 for c in s)
EN_K = 0.75                  # DejaVu renders ~1.45x taller than the CJK font at the same size: match the line heights


def fit_w(o, w):
    """shrink a text object uniformly so that it is at most w wide"""
    bpy.context.view_layer.update()
    if w and o.dimensions.x > w: k = w / o.dimensions.x; o.scale = (k, k, k)
    return o


def label(en, jp, loc, size, m, mj=None, k=0.5, w=None, **kw):
    """English line at loc, a small Japanese line just below it (in the text plane). Returns [en_obj, jp_obj]."""
    o = fit_w(text(en, loc, size, m, **kw), w)
    if not jp: return [o]
    R = o.rotation_euler.to_matrix()
    d = R @ Vector((0, -(size * 0.62 + size * k * 0.62) * o.scale[0] / 1.0, 0))
    o2 = fit_w(text(jp, tuple(Vector(loc) + d), size * k, mj or M_TXT, **kw), w)
    return [o, o2]


def pulse_alpha(times, decay=5.0, length=16, amp=1.0):
    a = np.zeros(NF)
    for t in times:
        f0 = int(round(t * FPS)); m_ = np.arange(max(0, f0), min(NF, f0 + length))
        a[m_] = np.maximum(a[m_], amp * np.exp(-(m_ - f0) / decay))
    return a


def frame_from_n(n):
    n = np.array(n, float); n /= np.linalg.norm(n)
    xa = np.cross([0, 1, 0], n); xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return Matrix(((xa[0], ya[0], n[0]), (xa[1], ya[1], n[1]), (xa[2], ya[2], n[2])))


def smooth(Xa, k):
    Y = Xa.copy(); a = 1 - math.exp(-1 / FPS / k)
    for f in range(1, len(Xa)): Y[f] = Y[f - 1] + (Xa[f] - Y[f - 1]) * a
    for f in range(len(Xa) - 2, -1, -1): Y[f] = Y[f + 1] + (Y[f] - Y[f + 1]) * a
    return Y


# ---------------- ball paths ----------------
def seg_at(g, t):
    if g['k'] == 'rest': return np.array(g['p'])
    if g['k'] == 'fly':
        u = t - g['t0']; return np.array(g['p']) + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
    ts = np.array(g['ts']); ps = np.array(g['ps'])
    return np.array([np.interp(t, ts, ps[:, c]) for c in range(3)])


class BallPath:
    def __init__(s, d): s.seg = d['seg']; s.hits = d['hits']; s.r = d['r']
    def at(s, t):
        for g in s.seg:
            if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9: return seg_at(g, t)
        return seg_at(s.seg[0], s.seg[0]['t0']) if t < s.seg[0]['t0'] else seg_at(s.seg[-1], s.seg[-1]['t1'])
BP = {n: BallPath(d) for n, d in LAY['balls'].items()}
POS = {n: np.array([b.at(t) for t in T]) for n, b in BP.items()}
MARBLES = sorted(n for n in BP if n.startswith('m'))

# ---------------- materials ----------------
M_RED = mat('red', (0.85, 0.02, 0.04), 0.1, **{'Coat Weight': 1.0, 'Emission Color': (1.0, 0.08, 0.06), 'Emission Strength': 0.5})
M_GOLDB = mat('goldball', (1.0, 0.72, 0.2), 0.15, **{'Metallic': 0.9, 'Coat Weight': 0.6, 'Emission Color': (1.0, 0.7, 0.2), 'Emission Strength': 0.25})
M_GOLD = mat('gold', (1.0, 0.7, 0.25), 0.22, **{'Metallic': 0.9})
M_BLACK = mat('bull', (0.015, 0.015, 0.02), 0.18, **{'Coat Weight': 1.0})
M_HORN = mat('horn', (0.95, 0.88, 0.72), 0.35, **{'Coat Weight': 0.4})
M_EYE = emis('eye', (1.0, 0.05, 0.02), 25.0)
M_KEY = glowmat('key', gain=9.0, base=0.12, rough=0.18, coat=1.0)
M_DRUM = glowmat('drumskin', gain=6.0, base=0.0, rough=0.5)
M_BELL = glowmat('bell', gain=7.0, base=0.0, rough=0.2, metal=0.9)
M_DRUMBODY = mat('drumbody', (0.6, 0.03, 0.03), 0.3, **{'Coat Weight': 0.8})
M_SAND = bpy.data.materials.new('sand'); M_SAND.use_nodes = True
_nt = M_SAND.node_tree; _b = _nt.nodes['Principled BSDF']; _b.inputs['Roughness'].default_value = 0.95
_nz = _nt.nodes.new('ShaderNodeTexNoise'); _nz.inputs['Scale'].default_value = 22.0; _nz.inputs['Detail'].default_value = 8.0
_rp = _nt.nodes.new('ShaderNodeValToRGB'); _rp.color_ramp.elements[0].color = (0.6, 0.36, 0.12, 1); _rp.color_ramp.elements[1].color = (0.95, 0.66, 0.3, 1)
_bu = _nt.nodes.new('ShaderNodeBump'); _bu.inputs['Strength'].default_value = 0.35; _nt.links.new(_nz.outputs['Fac'], _bu.inputs['Height']); _nt.links.new(_bu.outputs['Normal'], _b.inputs['Normal'])
_nt.links.new(_nz.outputs['Fac'], _rp.inputs['Fac']); _nt.links.new(_rp.outputs['Color'], _b.inputs['Base Color'])
M_BARRERA = mat('barrera', (0.55, 0.03, 0.03), 0.45, **{'Coat Weight': 0.3})
M_BARTOP = mat('bartop', (0.95, 0.75, 0.15), 0.4)
M_WHITE = mat('whitewash', (0.92, 0.88, 0.8), 0.7)
M_OCHRE = mat('ochre', (0.85, 0.55, 0.18), 0.6)
M_STONE = mat('stone', (0.55, 0.5, 0.45), 0.8)
M_SEAT = mat('seat', (0.6, 0.55, 0.5), 0.75)
M_WOOD = mat('wood', (0.4, 0.2, 0.08), 0.5)
M_STEEL = mat('steel', (0.5, 0.5, 0.55), 0.3, **{'Metallic': 0.9})
M_DARK = mat('dark', (0.02, 0.015, 0.015), 0.7)
M_VELVET = mat('velvet', (0.5, 0.0, 0.05), 0.6, **{'Sheen Weight': 0.8})
M_ROPE = mat('rope', (0.75, 0.6, 0.35), 0.8)
M_POST = mat('post', (0.35, 0.22, 0.12), 0.4, **{'Metallic': 0.5})
M_PLATE = mat('plate', (0.12, 0.02, 0.03), 0.3, **{'Coat Weight': 1.0})
M_OBJ = objcol('objcol', 0.4, coat=0.8)
M_LAMP = objcol('lampcol', 0.4, emit=5.0)
M_FX = fxmat('fx', 8.0)
M_TXT = emis('txtw', (1.0, 0.95, 0.85), 4.0); M_TXT_G = emis('txtg', (1.0, 0.75, 0.25), 6.0); M_TXT_R = emis('txtr', (1.0, 0.2, 0.15), 6.0)

# ---------------- balls ----------------
BALL = {}
for n, b in BP.items():
    rr = b.r
    o = obj(prim('sph', None, 32, 16), n); o.scale = (rr, rr, rr)
    if n == 'red': o.data = o.data.copy(); o.data.materials.append(M_RED)
    elif n == 'gold': o.data = o.data.copy(); o.data.materials.append(M_GOLDB)
    elif n == 'bull': o.data = o.data.copy(); o.data.materials.append(M_BLACK)
    else:
        o.data = o.data.copy(); o.data.materials.append(M_OBJ)
        if n.startswith('s'): o.color = ((1.0, 0.85, 0.1, 1) if n.endswith('lo') else (0.1, 0.45, 1.0, 1))
        else: o.color = (*[(1, 0.15, 0.15), (1, 0.85, 0.1), (0.2, 0.6, 1), (0.2, 0.9, 0.4), (1, 0.45, 0.8), (1, 1, 1), (1, 0.5, 0.1)][int(n[1:]) % 7], 1)
    if n in ('red', 'gold', 'bull'):
        bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=0.1, major_segments=32, minor_segments=8)
        hb = bpy.context.object; hb.parent = o
        hb.data.materials.append({'red': M_GOLD, 'gold': M_RED, 'bull': M_DRUMBODY}[n])
    key_loc(o, POS[n])
    sp = np.zeros(NF)
    for f in range(1, NF): sp[f] = sp[f - 1] + (POS[n][f, 0] - POS[n][f - 1, 0]) / rr
    o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, sp)
    BALL[n] = o
# the bull's head (does not roll): two horns and glowing eyes, facing +x
head = link(bpy.data.objects.new('bullhead', None)); key_loc(head, POS['bull'])
def horn_seg(a, b, r0, r1):
    a, b = Vector(a), Vector(b); d = b - a
    o = obj(prim('cone', M_HORN, 12, 1.0, r1 / r0, 1.0), 'horn'); q = Vector((0, 0, 1)).rotation_difference(d.normalized())
    o.matrix_world = Matrix.Translation((a + b) / 2) @ q.to_matrix().to_4x4() @ Matrix.Diagonal((r0, r0, d.length, 1)); o.parent = head
for side in (-1, 1):
    p0 = (RB * 0.25, side * RB * 0.62, RB * 0.62); p1 = (RB * 0.35, side * RB * 1.25, RB * 0.95); p2 = (RB * 0.6, side * RB * 1.45, RB * 1.55)
    horn_seg(p0, p1, 0.085, 0.06); horn_seg(p1, p2, 0.06, 0.012)
    ey = sphere((RB * 0.78, side * RB * 0.5, RB * 0.38), 0.06, M_EYE, (12, 8)); ey.parent = head
ring_ = link(bpy.data.objects.new('nosering', None)); ring_.parent = head
bpy.ops.mesh.primitive_torus_add(major_radius=0.08, minor_radius=0.016, location=(RB * 0.99, 0, -RB * 0.12), rotation=(0, math.pi / 2, 0))
bpy.context.object.data.materials.append(M_GOLD); bpy.context.object.parent = head

# ---------------- floor ----------------
def floor_z_at(x): return float(np.interp(x, FLX, FLZ))
allc = sorted([(h['p'][0], h['p'][2]) for n in ('red', 'gold', 'bull') for h in BP[n].hits])
AX = np.array([a[0] for a in allc]); AZ = np.array([a[1] for a in allc])
FLX = np.arange(-25.0, AX.max() + 30, 0.5)
FLZ = np.array([AZ[np.abs(AX - x) < 3.0].min() if (np.abs(AX - x) < 3.0).any() else np.nan for x in FLX])
s_ = ~np.isnan(FLZ); FLZ = np.interp(FLX, FLX[s_], FLZ[s_]) - 1.25
for _ in range(6): FLZ = np.minimum(FLZ, np.convolve(np.pad(FLZ, 3, mode='edge'), np.ones(7) / 7, 'valid'))
FLZ = np.convolve(np.pad(FLZ, 4, mode='edge'), np.ones(9) / 9, 'valid') - 0.05


def strip_mesh(X, Z, y0, y1, th, m, name, Z1=None):
    bm = bmesh.new(); vs = []
    Z1 = Z if Z1 is None else Z1
    for x, z, z1 in zip(X, Z, Z1):
        vs.append([bm.verts.new((x, y0, z)), bm.verts.new((x, y1, z1)), bm.verts.new((x, y1, z1 - th)), bm.verts.new((x, y0, z - th))])
    for a, b in zip(vs, vs[1:]):
        for k in range(4): bm.faces.new((a[k], b[k], b[(k + 1) % 4], a[(k + 1) % 4]))
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new(name, me)); o.data.materials.append(m)
    for p in o.data.polygons: p.use_smooth = True
    return o


# ---------------- keys, drums, bells: one per hit, they pop up 1.2 s before and vanish after ----------------
PITCH_COL = [(1.0, 0.12, 0.12), (1.0, 0.35, 0.05), (1.0, 0.6, 0.05), (1.0, 0.85, 0.1), (0.55, 1.0, 0.1), (0.1, 1.0, 0.3),
             (0.05, 1.0, 0.75), (0.05, 0.7, 1.0), (0.15, 0.35, 1.0), (0.45, 0.2, 1.0), (0.75, 0.2, 1.0), (1.0, 0.2, 0.65)]
def pcol(p, light=0.0):
    c = PITCH_COL[p % 12]; return tuple(min(1, x * (1 - light) + light) for x in c)
PIECES = []
KEY_TH = 0.1


def riser(objs, t_hit, anchor, lead=1.2, stay=0.9):
    anchor = Vector(anchor)
    e = link(bpy.data.objects.new('pop', None)); e.location = anchor
    for o in objs: o.parent = e; o.matrix_parent_inverse = Matrix.Translation(-anchor)
    s_ = np.zeros(NF)
    for f in range(NF):
        t = T[f]; u1 = (t - (t_hit - lead)) / 0.3; u2 = (t - (t_hit + stay)) / 0.35
        if u1 <= 0: a = 0.0
        elif u1 < 1: a = 1.18 * math.sin(u1 * math.pi / 2) if u1 < 0.7 else 1.18 - 0.18 * (u1 - 0.7) / 0.3
        else: a = 1.0
        b = 1.0 - min(1, max(0, u2)) ** 1.5
        s_[f] = max(0.001, a * b)
    for k in range(3): key_series(e, 'scale', k, s_, 1e-4)
    return e


def flash(o, t, col, n=None, dip=0.03, keys=((0, 1.0), (4, 0.55), (18, 0.1))):
    fh = int(round(t * FPS))
    o.color = (*col, 0); o.keyframe_insert('color', index=3, frame=max(0, fh - 1))
    for df, a in keys:
        o.color = (*col, a); o.keyframe_insert('color', index=3, frame=fh + df)
    if n is not None and dip:
        base = o.location.copy(); nv = Vector(n)
        for df, d in ((-1, 0), (1, -dip), (4, dip * 0.3), (8, 0)):
            o.location = base + nv * d; o.keyframe_insert('location', frame=max(0, fh + df))


def add_post(top, zf, r=0.03, m=M_POST):
    return rod((top[0], top[1], top[2]), (top[0], top[1], zf - 0.3), r, m, 8)


# toreros' keys: shared (one wide key) or split
pair_hits = {}
for n in ('red', 'gold'):
    for h in BP[n].hits: pair_hits.setdefault((h['ev'], h['kind']), []).append((n, h))
KEYS = []
for (ev, kind), lst in pair_hits.items():
    if kind not in ('uni', 'key'): continue
    groups = [lst] if kind == 'uni' else [[x] for x in lst]
    for grp in groups:
        hs = [h for _, h in grp]; nv = np.array(hs[0]['n'])
        pa = np.mean([h['p'] for h in hs], axis=0); top = pa - RP * nv
        d = (2 * DYP + 0.44) if len(grp) == 2 else 0.44
        ctr = top - nv * KEY_TH / 2
        o = obj(prim('bcube', M_KEY), 'key')
        o.matrix_world = Matrix.Translation(Vector(ctr)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((KW, d, KEY_TH, 1))
        col = pcol(hs[0]['pitch']); flash(o, hs[0]['t'], col, nv)
        PIECES.append(dict(kind='key', t=hs[0]['t'], top=top, n=nv))
        po = add_post(top - nv * KEY_TH, floor_z_at(top[0]), 0.025)
        if hs[0]['t'] > T0 + 0.5: riser([o, po], hs[0]['t'], top)
        KEYS.append(dict(t=hs[0]['t'], top=top))
# bull drums
DRUMS = []
for h in BP['bull'].hits:
    nv = np.array(h['n']) if h['kind'] != 'findrum' else np.array([0, 0, 1.0])
    top = np.array(h['p']) - RB * nv
    big = h['kind'] == 'findrum'
    rr = 0.24 if not big else 0.34; hh = 0.3 if not big else 0.42
    fr = frame_from_n(nv).to_4x4()
    skin = obj(prim('cyl', M_DRUM, 32), 'skin'); skin.matrix_world = Matrix.Translation(Vector(top - nv * 0.01)) @ fr @ Matrix.Diagonal((rr, rr, 0.02, 1))
    body = obj(prim('cyl', M_DRUMBODY, 32), 'drum'); body.matrix_world = Matrix.Translation(Vector(top - nv * (hh / 2 + 0.02))) @ fr @ Matrix.Diagonal((rr * 1.02, rr * 1.02, hh, 1))
    rim = obj(prim('cyl', M_GOLD, 32), 'rim'); rim.matrix_world = Matrix.Translation(Vector(top - nv * 0.03)) @ fr @ Matrix.Diagonal((rr * 1.07, rr * 1.07, 0.05, 1))
    skin.color = (1.0, 0.85, 0.6, 0); flash(skin, h['t'], (1.0, 0.8, 0.5), None)
    po = add_post(top - nv * hh, floor_z_at(top[0]), 0.045)
    objs = [skin, body, rim, po]
    if h['kind'] != 'findrum' and h['t'] > T0 + 0.5: riser(objs, h['t'], top)
    elif h['kind'] == 'findrum': riser(objs, h['t'], top, stay=99)
    PIECES.append(dict(kind='drum', t=h['t'], top=top, n=nv)); DRUMS.append(dict(t=h['t'], top=top, kind=h['kind']))
# spectators' bells (mouth down, the ball lands on the crown); final bells mouth up
BELLS = []
for n in [k for k in BP if k.startswith('s')]:
    for h in BP[n].hits:
        if h['kind'] not in ('bell', 'finbell'): continue
        nv = np.array(h['n']) if h['kind'] == 'bell' else np.array([0, 0, 1.0])
        top = np.array(h['p']) - RC * nv; p = h['pitch']; sz = 0.26 + 0.006 * (64 - p)
        o = obj(prim('bell', M_BELL, 24), 'bell')
        if h['kind'] == 'bell':
            o.matrix_world = Matrix.Translation(Vector(top)) @ frame_from_n(nv).to_4x4() @ Matrix.Diagonal((sz * 0.7, sz * 0.7, sz, 1))
        else:
            o.matrix_world = Matrix.Translation(Vector(top)) @ Matrix.Diagonal((0.32, 0.32, -0.28, 1))
        col = pcol(p, 0.25); flash(o, h['t'], col, nv if h['kind'] == 'bell' else None, dip=0.02)
        bot = top - nv * (sz if h['kind'] == 'bell' else 0.28)
        po = rod(bot, (bot[0], bot[1], bot[2] - 0.7), 0.02, M_POST, 8)
        if h['kind'] == 'bell' and h['t'] > T0 + 0.5: riser([o, po], h['t'], top)
        elif h['kind'] == 'finbell': riser([o, po], h['t'], top, stay=99)
        PIECES.append(dict(kind='bell', t=h['t'], top=top, n=nv)); BELLS.append(dict(t=h['t'], top=top, bot=bot))

# hit rings (FX pool) and sand puffs under the bull
NRING = 14
ring_loc = np.zeros((NRING, NF, 3)); ring_scl = np.full((NRING, NF), 1e-3); ring_a = np.zeros((NRING, NF)); ring_c = [(1, 1, 1)] * NRING
for k, P_ in enumerate(sorted([p for p in PIECES if p['kind'] in ('key', 'drum')], key=lambda b: b['t'])):
    r = k % NRING; f0 = int(round(P_['t'] * FPS))
    for df in range(12):
        f = f0 + df
        if f >= NF: break
        u = df / 11; ring_loc[r, f] = P_['top'] + np.array([0, -0.3, 0.02]); ring_scl[r, f] = (0.3 if P_['kind'] == 'key' else 0.55) + 1.2 * (1 - (1 - u) ** 2)
        ring_a[r, f] = (1 - u) ** 1.6
    ring_c[r] = (1.0, 0.9, 0.6) if P_['kind'] == 'key' else (1.0, 0.55, 0.25)
for r in range(NRING):
    bpy.ops.mesh.primitive_torus_add(major_radius=0.25, minor_radius=0.014, major_segments=32, minor_segments=5, rotation=(math.pi / 2, 0, 0))
    o = bpy.context.object; o.data.materials.append(M_FX); o.color = (*ring_c[r], 0)
    for f in range(1, NF):
        if ring_a[r, f] == 0: ring_loc[r, f] = ring_loc[r, f - 1]
    key_loc(o, ring_loc[r])
    for k in range(3): key_series(o, 'scale', k, ring_scl[r])
    key_series(o, 'color', 3, ring_a[r], 1e-3)

# ---------------- the bullring: sand, barrera, callejon, stands with the crowd, arcade, flags ----------------
strip_mesh(FLX, FLZ, -9.0, 2.05, 0.5, M_SAND, 'arena')
strip_mesh(FLX, FLZ - 0.05, 2.05, 4.3, 0.5, M_SAND, 'callejon')
M_LINE = mat('chalk', (0.95, 0.93, 0.88), 0.8)
for yl in (-2.6, -2.75): strip_mesh(FLX, FLZ + 0.006, yl - 0.05, yl + 0.05, 0.01, M_LINE, 'tercio')
BARH = 1.15
strip_mesh(FLX, FLZ + BARH, 1.98, 2.1, BARH + 0.3, M_BARRERA, 'barrera')
strip_mesh(FLX, FLZ + BARH + 0.06, 1.95, 2.13, 0.08, M_BARTOP, 'bartop')
# the gate where the bull bursts into the ring (A2): where its path crosses the barrera
bp_ = POS['bull']; fcross = int(np.argmax(bp_[:, 1] < 2.05 + RB * 0.2))
x_gate = float(bp_[fcross, 0]); t_gate = T[fcross]
gz = floor_z_at(x_gate)
gap = box((x_gate, 2.04, gz + BARH / 2), (2.3, 0.2, BARH + 0.6), M_DARK, name='gatehole')
gap.hide_render = True
door = link(bpy.data.objects.new('doorhinge', None)); door.location = (x_gate - 1.1, 1.98, gz)
dpanel = box((1.1 * 0.5 + 0.0, 0, BARH / 2 + 0.02), (1.1, 0.08, BARH), M_BARTOP, name='door'); dpanel.parent = door
dpanel2 = box((1.65, 0, BARH / 2 + 0.02), (1.1, 0.08, BARH), M_BARTOP, name='door2'); dpanel2.parent = door
dang = np.array([0.0 if t < t_gate - 0.45 else -min(1.0, (t - (t_gate - 0.45)) / 0.25) ** 1.5 * 1.7 for t in T])
door.rotation_mode = 'XYZ'; key_series(door, 'rotation_euler', 2, dang)
text('TORIL', (x_gate, 1.9, gz + BARH + 0.42), 0.3, M_TXT_G, rot=(math.pi / 2, 0, 0))
# stands: rising tiers of seats along the whole ring
TIER0_Y, TIER_D, TIER_H = 4.5, 0.85, 0.55
SX = FLX[::2]; SZ = np.array([floor_z_at(x) for x in SX])
def stand_z(x, k): return floor_z_at(x) + 2.4 + k * TIER_H
for k in range(9):
    y0 = TIER0_Y + k * TIER_D
    strip_mesh(SX, SZ + 2.4 + k * TIER_H, y0, y0 + TIER_D, 2.4 + k * TIER_H + 1.5, M_SEAT if k % 2 else M_STONE, f'tier{k}')
strip_mesh(SX, SZ + 2.4 - 0.05, 4.25, 4.5, 2.6, M_WHITE, 'standwall')
M_DRAPE = objcol('drape', 0.7)
for k, x in enumerate(np.arange(SX[0] + 3, SX[-1], 7.5)):                 # hanging drapes on the stand wall
    zt = floor_z_at(x) + 2.3
    d_ = box((x, 4.2, zt - 0.55), (1.4, 0.04, 1.1), M_DRAPE, name='drape'); d_.color = ((0.75, 0.02, 0.05, 1) if k % 2 else (1.0, 0.75, 0.05, 1))
    s_ = box((x, 4.18, zt - 0.55), (1.4, 0.03, 0.3), M_DRAPE, name='drapestripe'); s_.color = ((1.0, 0.75, 0.05, 1) if k % 2 else (0.75, 0.02, 0.05, 1))
strip_mesh(SX, SZ + 2.55, 4.22, 4.32, 0.18, M_OCHRE, 'standrail')
# the crowd: rows of marbles sitting on the tiers (one mesh, random colours via a colour attribute)
CROWD_COLS = [(0.9, 0.1, 0.1), (1, 0.85, 0.15), (0.15, 0.5, 1), (0.2, 0.85, 0.35), (1, 0.45, 0.75), (0.95, 0.95, 0.95), (1, 0.5, 0.1), (0.6, 0.3, 0.9)]
M_CROWD = bpy.data.materials.new('crowd'); M_CROWD.use_nodes = True
_nt = M_CROWD.node_tree; _b = _nt.nodes['Principled BSDF']; _b.inputs['Roughness'].default_value = 0.15; _b.inputs['Coat Weight'].default_value = 0.8
_at = _nt.nodes.new('ShaderNodeAttribute'); _at.attribute_name = 'col'; _nt.links.new(_at.outputs['Color'], _b.inputs['Base Color'])
for chunk in range(0, len(SX) - 1, 40):
    bm = bmesh.new(); cl = bm.loops.layers.color.new('col')
    for x in np.arange(SX[chunk], SX[min(len(SX) - 1, chunk + 40)], 0.42):
        for k in range(1, 9):
            if rng.random() < 0.22: continue
            r_ = 0.13; c = (x + rng.uniform(-0.1, 0.1), TIER0_Y + k * TIER_D + 0.35, stand_z(x, k) + r_)
            col = CROWD_COLS[rng.integers(len(CROWD_COLS))]
            g = bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=5, radius=r_)
            bmesh.ops.translate(bm, verts=g['verts'], vec=c)
            for v in g['verts']:
                for lp in v.link_loops: lp[cl] = (*col, 1)
    me = bpy.data.meshes.new('crowd'); bm.to_mesh(me); bm.free()
    for p in me.polygons: p.use_smooth = True
    me.materials.append(M_CROWD); link(bpy.data.objects.new('crowd', me))
# arcade on top (Seville style): white pillars, ochre arches, tiled roof
ytop = TIER0_Y + 9 * TIER_D
for x in np.arange(SX[0], SX[-1], 2.4):
    zt = stand_z(x, 9)
    box((x, ytop + 0.2, zt + 1.3), (0.3, 0.3, 2.6), M_WHITE, name='pillar')
    ao = obj(prim('arch', M_OCHRE), 'arch'); ao.location = (x + 1.2, ytop + 0.2, zt + 2.0)
strip_mesh(SX, SZ + 2.4 + 9 * TIER_H + 3.3, ytop - 0.1, ytop + 0.5, 0.5, M_WHITE, 'arcadebeam')
strip_mesh(SX, SZ + 2.4 + 9 * TIER_H + 3.6, ytop - 0.4, ytop + 1.6, 0.25, mat('tiles', (0.55, 0.15, 0.06), 0.6), 'roof',
           Z1=SZ + 2.4 + 9 * TIER_H + 4.4)
# bunting: red / yellow pennants along the barrera and the stand rail
M_FLAG = objcol('flag', 0.6)
for x in np.arange(SX[0], SX[-1], 0.7):
    for yy, zz in ((1.93, floor_z_at(x) + BARH + 0.05), (4.2, floor_z_at(x) + 2.5)):
        f_ = obj(prim('cone', M_FLAG, 3, 0.16, 0.0, 0.3), 'pennant'); f_.location = (x, yy, zz - 0.12); f_.rotation_euler = (math.pi, 0, 0)
        f_.scale = (1, 0.15, 1); f_.color = ((0.85, 0.05, 0.05, 1) if int(x / 0.7) % 2 else (1.0, 0.8, 0.05, 1))

# ---------------- sky: sunset gradient, low sun, a few clouds ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes['Background']; bg.inputs['Color'].default_value = (0.9, 0.5, 0.3, 1); bg.inputs['Strength'].default_value = 0.35
sun = bpy.data.lights.new('sun', 'SUN'); sun.energy = 4.0; sun.angle = 0.04; sun.color = (1.0, 0.72, 0.45)
so_ = link(bpy.data.objects.new('sun', sun)); so_.rotation_euler = (math.radians(62), math.radians(-28), math.radians(-35))
fill = bpy.data.lights.new('fill', 'SUN'); fill.energy = 0.6; fill.color = (0.6, 0.65, 1.0)
fo = link(bpy.data.objects.new('fill', fill)); fo.rotation_euler = (math.radians(60), math.radians(30), math.radians(160))
SKY = bpy.data.materials.new('sky'); SKY.use_nodes = True; nt = SKY.node_tree; nt.nodes.clear()
tc = nt.nodes.new('ShaderNodeTexCoord'); sep = nt.nodes.new('ShaderNodeSeparateXYZ'); rp = nt.nodes.new('ShaderNodeValToRGB')
ce = rp.color_ramp.elements
ce[0].position = 0.0; ce[0].color = (1.0, 0.45, 0.12, 1); ce[1].position = 1.0; ce[1].color = (0.12, 0.08, 0.3, 1)
e_ = ce.new(0.25); e_.color = (1.0, 0.3, 0.25, 1); e_ = ce.new(0.55); e_.color = (0.45, 0.15, 0.4, 1)
em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Strength'].default_value = 1.4; out = nt.nodes.new('ShaderNodeOutputMaterial')
nz2 = nt.nodes.new('ShaderNodeTexNoise'); nz2.inputs['Scale'].default_value = 3.0; nz2.inputs['Detail'].default_value = 4.0
mr2 = nt.nodes.new('ShaderNodeMapRange'); mr2.inputs['From Min'].default_value = 0.55; mr2.inputs['From Max'].default_value = 0.75
mix = nt.nodes.new('ShaderNodeMixRGB'); mix.blend_type = 'MIX'; mix.inputs['Color2'].default_value = (1.0, 0.65, 0.5, 1)
nt.links.new(tc.outputs['Generated'], sep.inputs[0]); nt.links.new(sep.outputs['Y'], rp.inputs['Fac'])
nt.links.new(tc.outputs['Object'], nz2.inputs['Vector']); nt.links.new(nz2.outputs['Fac'], mr2.inputs['Value'])
nt.links.new(rp.outputs['Color'], mix.inputs['Color1']); nt.links.new(mr2.outputs[0], mix.inputs['Fac'])
nt.links.new(mix.outputs[0], em.inputs['Color']); nt.links.new(em.outputs[0], out.inputs[0])
SKYX0, SKYX1 = FLX[0] - 30, FLX[-1] + 40
skyo = obj(prim('plane', SKY), 'sky'); skyo.location = ((SKYX0 + SKYX1) / 2, 70, 10); skyo.rotation_euler = (math.pi / 2, 0, 0)
skyo.scale = (SKYX1 - SKYX0 + 120, 110, 1); skyo.visible_shadow = False
M_SUND = emis('sundisk', (1.0, 0.42, 0.12), 1.6)
sd_ = sphere((40.0, 68, 7.0), 3.2, M_SUND, (32, 16)); sd_.visible_shadow = False

# ---------------- intro: the president's box with the ramp, and the toril cage ----------------
rmp = EV['ramp']; top_, lip_ = np.array(rmp['top']), np.array(rmp['lip']); al = rmp['alpha']
ca_, sa_ = math.cos(al), math.sin(al)
xs_ = np.linspace(top_[0] - 0.3, lip_[0], 20); zs_ = top_[2] - (xs_ - top_[0]) * math.tan(al) - RP - 0.03
strip_mesh(xs_, zs_, -0.62, 0.62, 0.08, M_WOOD, 'ramp')
for yy in (-0.62, 0.62): strip_mesh(xs_, zs_ + 0.16, yy - 0.04, yy + 0.04, 0.2, M_GOLD, 'ramprail')
zb = lip_[2] - RP - 0.2
box((lip_[0] - 1.6, 0.0, zb - 0.5), (3.6, 1.8, 1.0), M_VELVET, name='palco')
box((lip_[0] - 1.6, -0.92, zb - 0.02), (3.7, 0.08, 0.12), M_GOLD, name='palcorail')
for x in np.arange(lip_[0] - 3.3, lip_[0] + 0.1, 0.45): rod((x, -0.92, zb - 1.0), (x, -0.92, zb), 0.03, M_GOLD, 8)
rod((lip_[0] - 3.4, 0, zb - 1.0), (lip_[0] - 3.4, 0, floor_z_at(lip_[0]) - 0.2), 0.12, M_WHITE, 12)
rod((lip_[0] + 0.1, 0, zb - 1.0), (lip_[0] + 0.1, 0, floor_z_at(lip_[0]) - 0.2), 0.12, M_WHITE, 12)
gate = link(bpy.data.objects.new('rampgate', None)); gate.location = (top_[0] + RP + 0.1, 0, top_[2] + RP + 0.25)
gp = box((0, 0, -0.25), (0.06, 1.2, 0.5), M_GOLD, name='gatebar'); gp.parent = gate
gang = np.array([0.0 if t < rmp['t0'] - 0.1 else -min(1.0, (t - rmp['t0'] + 0.1) / 0.15) * 1.4 for t in T])
gate.rotation_mode = 'XYZ'; key_series(gate, 'rotation_euler', 1, gang)
# toril: dark cage behind the barrera, the bull's eyes glow inside; a trapdoor drops it on its first drum
tr_ = np.array(EV['toril']['p'])
box((tr_[0], tr_[1] + 0.2, tr_[2] + 0.1), (1.3, 1.1, 1.3), M_DARK, name='torilcage').visible_shadow = False
for yy in np.linspace(tr_[1] - 0.5, tr_[1] + 0.5, 1):
    pass
for x in np.linspace(tr_[0] - 0.6, tr_[0] + 0.6, 7): rod((x, tr_[1] - 0.6, tr_[2] - RB - 0.1), (x, tr_[1] - 0.6, tr_[2] + 0.7), 0.025, M_STEEL, 6)
tdh = link(bpy.data.objects.new('torilhinge', None)); tdh.location = (tr_[0] - 0.55, tr_[1], tr_[2] - RB - 0.04)
tdp = box((0.55, 0, 0), (1.1, 0.9, 0.06), M_WOOD, name='torildoor'); tdp.parent = tdh
tang = np.array([0.0 if t < EV['toril']['t'] - 0.02 else min(1.0, (t - EV['toril']['t'] + 0.02) / 0.18) ** 2 * 1.6 for t in T])
tdh.rotation_mode = 'XYZ'; key_series(tdh, 'rotation_euler', 1, tang)
cagewall = box((tr_[0], tr_[1] + 0.2, tr_[2] - RB - 0.6), (1.3, 1.1, 1.1), M_WHITE, name='torilbase')
rod((tr_[0], tr_[1], tr_[2] - RB - 1.1), (tr_[0], tr_[1], floor_z_at(tr_[0]) - 0.2), 0.15, M_WHITE, 10)

# ---------------- the trebuchet ----------------
tb_ = EV['treb']; piv = np.array(tb_['pivot']); LA, LS = tb_['L'], tb_['Ls']
T_ROT = tb_['t_rel'] - tb_['t_s']
def arm_phi(t):
    u = min(max(0.0, t - tb_['t_s']), T_ROT); ph = tb_['phi0'] - 0.5 * tb_['alpha'] * u * u
    if t > tb_['t_rel']:                                      # hits the padded stop: a small damped recoil
        w_ = t - tb_['t_rel']; ph += 0.06 * math.exp(-w_ / 0.25) * math.sin(w_ * 18)
    return ph
zf_t = floor_z_at(piv[0])
for yy in (-0.62, 0.62):
    for dx in (-1.0, 1.0): rod((piv[0], yy, piv[2]), (piv[0] + dx, yy, zf_t - 0.1), 0.07, M_WOOD, 8)
    rod((piv[0] - 1.0, yy, zf_t + 0.25), (piv[0] + 1.0, yy, zf_t + 0.25), 0.05, M_WOOD, 8)
rod((piv[0], -0.7, piv[2]), (piv[0], 0.7, piv[2]), 0.06, M_STEEL, 12)
arm = link(bpy.data.objects.new('arm', None)); arm.location = tuple(piv)
beam = box((0.5 * (LA - LS), 0, 0), (LA + LS, 0.16, 0.14), M_WOOD, name='beam'); beam.parent = arm
cross = box((LA, 0, 0), (0.1, 2 * DYP + 0.3, 0.08), M_GOLD, name='cupbar'); cross.parent = arm
arot = np.array([-arm_phi(t) for t in T]); arm.rotation_mode = 'XYZ'; key_series(arm, 'rotation_euler', 1, arot)
# the two cups ride on the tip and stay upright (gimbal)
cupe = link(bpy.data.objects.new('cups', None))
TIP = np.array([piv + LA * np.array([math.cos(arm_phi(t)), 0, math.sin(arm_phi(t))]) for t in T]); key_loc(cupe, TIP)
cupc = np.array(tb_['cup'])
for side, yo in (('red', -DYP), ('gold', DYP)):
    cu = obj(prim('bowl', M_BELL, 24), 'cup'); cu.parent = cupe; cu.location = (0, yo, -RP * 0.35); cu.scale = (RP * 1.15, RP * 1.15, RP * 1.15)
    hcup = [h for h in BP[side].hits if h['kind'] == 'cup'][0]
    col = pcol(hcup['pitch'], 0.2); flash(cu, hcup['t'], col, None)
    rod((0, yo, -RP * 0.35 - 0.06), (0, yo, -RP * 0.35 - 0.3), 0.025, M_GOLD, 6).parent = cupe
# counterweight (hangs from the short arm), latch and the trigger rope from the bull's trigger drum
cwe = link(bpy.data.objects.new('cw', None))
CW = np.array([piv + LS * np.array([math.cos(arm_phi(t) - math.pi), 0, math.sin(arm_phi(t) - math.pi)]) for t in T]); key_loc(cwe, CW)
cwb = box((0, 0, -0.45), (0.75, 0.7, 0.7), M_STONE, name='counterweight'); cwb.parent = cwe
rod((0, 0, 0), (0, 0, -0.1), 0.03, M_STEEL, 6).parent = cwe
text('1t', (0, -0.36, -0.45), 0.26, M_TXT_G, rot=(math.pi / 2, 0, 0)).parent = cwe
trig = [d for d in DRUMS if d['kind'] == 'trigger'][0]; cw0 = CW[0]
latch = box((cw0[0], 0, cw0[2] - 0.9), (0.08, 0.5, 0.08), M_STEEL, name='latch')
lz = np.array([cw0[2] - 0.9 if t < tb_['t_sig'] else cw0[2] - 0.9 - min(1.0, (t - tb_['t_sig']) / 0.06) * 0.3 for t in T])
key_series(latch, 'location', 2, lz)
ttop = trig['top']
rope = rod((ttop[0] + 0.45, ttop[1], ttop[2] - 0.2), (cw0[0], 0.0, cw0[2] - 0.9), 0.02, M_ROPE, 6)
rope.color = (1, 0.8, 0.3, 0)
label('TREBUCHET', 'トレブシェット（投石機）', (piv[0] - 1.2, -1.0, piv[2] + 1.62), 0.26, M_TXT_G, k=0.6)
label('Step here…', 'ここを踏むと…', (ttop[0], ttop[1] - 0.55, ttop[2] + 0.8), 0.22, M_TXT_R, k=0.6)
# the toreros' final cups, the spectators' box seats
fc = np.array(EV['final_cups'])
for side, yo in (('red', -DYP), ('gold', DYP)):
    hf = [h for h in BP[side].hits if h['kind'] == 'fincup'][0]
    cu = obj(prim('bowl', M_BELL, 24), 'fincup'); cu.location = (fc[0], yo, fc[2] - RP * 0.35); cu.scale = (RP * 1.15, RP * 1.15, RP * 1.15)
    flash(cu, hf['t'], pcol(hf['pitch'], 0.2), None, keys=((0, 1.0), (6, 0.6), (40, 0.25)))
    po = rod((fc[0], yo, fc[2] - RP * 0.35 - 0.12), (fc[0], yo, floor_z_at(fc[0])), 0.04, M_GOLD, 8)
    riser([cu, po], hf['t'], (fc[0], yo, fc[2]), stay=99)
for p_ in EV['seats']:
    cu = obj(prim('bowl', M_VELVET, 20), 'seat'); cu.location = (p_[0], p_[1], p_[2] - RC * 0.35); cu.scale = (RC * 1.2, RC * 1.2, RC * 1.2)
    rod((p_[0], p_[1], p_[2] - RC), (p_[0], p_[1], p_[2] - 0.8), 0.02, M_GOLD, 6)

# ---------------- the crowd wave (tremolo): ledges, chime rails, gutters, two last bells ----------------
cw_ = EV['crowd']
rows = {}
for n in MARBLES:
    h = BP[n].hits[0]; rows.setdefault(h['kind'] + h['row'], []).append((n, h))
M_RAIL = glowmat('rail', gain=4.0, base=0.0, rough=0.2, metal=0.9)
for key_, lst in rows.items():
    lst.sort(key=lambda a: a[1]['t'])
    if key_.startswith('rail'):
        P0 = np.array(lst[0][1]['p']); P1 = np.array(lst[-1][1]['p'])
        a_ = P0 - np.array([0.25, 0, RM + 0.035]); b_ = P1 + np.array([0.25, 0, 0]) - np.array([0, 0, RM + 0.035])
        rl = rod(a_, b_, 0.035, M_RAIL, 12); rl.color = (1.0, 0.8, 0.4, 0)
        key_series(rl, 'color', 3, 0.5 * pulse_alpha([h['t'] for _, h in lst], decay=3, length=8), 1e-3)
        g0 = a_ + np.array([0, 0.17, -0.02]); g1 = b_ + np.array([0, 0.17, -0.02])
        strip_mesh(np.array([g0[0], g1[0]]), np.array([g0[2], g1[2]]), g0[1] - 0.09, g0[1] + 0.09, 0.05, M_WOOD, 'gutter')
        for xx in np.arange(a_[0], b_[0], 1.2): rod((xx, a_[1], np.interp(xx, [a_[0], b_[0]], [a_[2], b_[2]])), (xx, a_[1], floor_z_at(xx) + 1.5), 0.02, M_POST, 6)
    for n, h in lst:
        p = np.array(h['p']); led = p + np.array([0, 0, 0.5 * G * EV['crowd']['td'] ** 2])
        hold = link(bpy.data.objects.new('holder', None)); hold.location = tuple(led - np.array([0, 0, RM]))
        fl_ = box((0.0, 0.0, -0.01), (0.15, 0.15, 0.02), M_GOLD, name='flap'); fl_.parent = hold
        ft = h['t'] - EV['crowd']['td']
        hang = np.array([0.0 if t < ft - 0.03 else min(1.0, (t - ft + 0.03) / 0.08) * 1.4 for t in T])
        hold.rotation_mode = 'XYZ'; key_series(hold, 'rotation_euler', 1, hang)
        rod(led - np.array([0, 0, RM + 0.02]), led - np.array([0, -0.0, RM + 0.4]), 0.01, M_POST, 5)
        fxs = sphere(p - np.array([0, 0, RM]), 0.12, M_FX, (10, 6)); fxs.color = (1.0, 0.85, 0.5, 0)
        key_series(fxs, 'color', 3, 0.8 * pulse_alpha([h['t']], decay=2, length=6), 1e-3)
        if h['kind'] == 'cbell':
            o = obj(prim('bell', M_BELL, 24), 'cbell'); top = p - np.array([0, 0, RM])
            o.matrix_world = Matrix.Translation(Vector(top)) @ Matrix.Diagonal((0.18, 0.18, 0.26, 1))
            flash(o, h['t'], pcol(h['pitch'], 0.2), None)
            rod(top - np.array([0, 0, 0.26]), top - np.array([0, 0, 0.8]), 0.02, M_POST, 6)
label('CROWD WAVE', '観客席のウェーブ', ((cw_['x0'] + cw_['x1']) / 2, 3.0, cw_['z'] + 1.42), 0.32, M_TXT_G, k=0.55)

# ---------------- finale: flower shower (petals and carnations from the stands) ----------------
M_PET = objcol('petal', 0.5, emit=0.6)
bfx = np.array(EV['bull_final']); FC = (fc + bfx) / 2
PETC = [(0.9, 0.05, 0.1), (1.0, 0.35, 0.5), (1.0, 0.95, 0.9), (1.0, 0.75, 0.1), (0.85, 0.1, 0.35)]
for k in range(150):
    pt = obj(prim('sph', M_PET, 6, 4), 'petal'); pt.color = (*PETC[k % 5], 1)
    x0 = FC[0] + rng.uniform(-5.5, 4.0); y0 = rng.uniform(3.5, 7.5); z0 = FC[2] + rng.uniform(4.5, 7.5)
    vx = rng.uniform(-0.6, 0.6); vy = rng.uniform(-3.8, -1.6); ts_ = FINAL + rng.uniform(-0.1, 0.5)
    P = np.zeros((NF, 3)); RT = np.zeros((NF, 3))
    for f in range(NF):
        s_ = T[f] - ts_
        if s_ < 0: P[f] = (x0, y0, z0); continue
        P[f] = (x0 + vx * s_ + 0.25 * math.sin(2.5 * s_ + k), y0 + vy * (1 - math.exp(-s_ / 0.8)) * 0.8 * 1.25, z0 + 1.6 * s_ * math.exp(-s_ / 0.4) - 1.1 * s_)
        RT[f] = (s_ * (3 + k % 5), s_ * (2 + k % 3), s_ * 1.5)
    key_loc(pt, P)
    sc_ = np.where(T >= ts_, 1.0, 0.001)
    key_series(pt, 'scale', 0, sc_ * 0.16); key_series(pt, 'scale', 1, sc_ * 0.11); key_series(pt, 'scale', 2, sc_ * 0.035)
    for c_ in range(3): key_series(pt, 'rotation_euler', c_, RT[:, c_])
ole = text('¡OLÉ!', (FC[0], 2.6, fc[2] + 3.2), 0.9, M_TXT_G, rot=(math.pi / 2, 0, 0))
osz = np.array([0.001 if t < FINAL else min(1.0, (t - FINAL) / 0.25) for t in T])
for k in range(3): key_series(ole, 'scale', k, osz)

# ---------------- signs (keywords for search, readable on a phone) ----------------
def sign(lines, x, y, z, w, h, plate=True):
    if plate:
        box((x, y + 0.08, z), (w, 0.12, h), M_PLATE, rot=(-0.3, 0, 0), name='plate')
        for sgn in (1, -1): box((x, y + 0.08 + 0.2955 * sgn * h / 2, z + 0.955 * sgn * h / 2), (w + 0.1, 0.16, 0.07), M_GOLD, rot=(-0.3, 0, 0))
        for xx in (x - w / 2 + 0.2, x + w / 2 - 0.2): rod((xx, y + 0.1, z - h / 2), (xx, y + 0.1, floor_z_at(xx)), 0.05, M_GOLD, 8)
    zz = z + h / 2
    for s_, size, mt in lines:
        zz -= size * 1.25 + (0.05 if zz < z + h / 2 - 0.1 else 0.0)        # a little air between the Latin lines
        fit_w(text(s_, (x, y - 0.03 + 0.31 * (zz + size * 0.15 - z), zz + size * 0.15), size, mt), w - 0.3)
x_ = lambda t: float(np.interp(t, T, (POS['red'][:, 0] + POS['gold'][:, 0]) / 2))
z_ = lambda t: float(np.interp(t, T, (POS['red'][:, 2] + POS['gold'][:, 2]) / 2))
sign([('CARMEN – Bizet', 0.52, M_TXT_R), ('Prelude (1875) · Classical', 0.3, M_TXT), ('Opera × Marble Machine', 0.28, M_TXT_G),
      ('カルメン 前奏曲（ビゼー）', 0.2, M_TXT)],
     -0.4, -2.4, z_(T0) - 1.05, 5.0, 2.2)
sign([('Toreros vs the Bull', 0.34, M_TXT_G), ('Will they escape?', 0.38, M_TXT_R), ('赤い闘牛士 vs 黒い牛 逃げ切れる？', 0.18, M_TXT)], x_(5.4) + 0.6, -1.8, z_(5.4) - 0.8, 4.2, 1.3)
sign([('Georges Bizet', 0.4, M_TXT_G), ('Carmen – Prelude', 0.3, M_TXT), ('ジョルジュ・ビゼー「カルメン」', 0.18, M_TXT)], x_(10.2) + 0.6, -1.8, z_(10.2) - 0.8, 4.0, 1.3)
sign([('Plaza de Toros', 0.38, M_TXT_G), ('Bullring · Seville, Spain', 0.28, M_TXT), ('闘牛場 スペイン・セビリア', 0.18, M_TXT)], x_(14.0) + 0.6, -1.8, z_(14.0) - 0.8, 4.4, 1.25)
sign([('The bull breaks in!', 0.42, M_TXT_R), ('Bullfight entrance march', 0.26, M_TXT), ('牛、乱入！ 闘牛の入場曲', 0.18, M_TXT)], x_(18.6) + 0.6, -1.8, z_(18.6) - 0.85, 3.6, 1.35)
sign([('Classical Music', 0.38, M_TXT_G), ('Opera "Carmen"', 0.3, M_TXT), ('クラシック名曲 オペラ『カルメン』', 0.18, M_TXT)], x_(23.4) + 0.6, -1.8, z_(23.4) - 0.8, 4.0, 1.3)
sign([('Satisfying', 0.38, M_TXT_G), ('Marble Machine ASMR', 0.3, M_TXT), ('ビゼー「カルメン」ビー玉マーブルマシン', 0.18, M_TXT)], x_(28.0) + 0.6, -1.8, z_(28.0) - 0.8, 4.2, 1.3)
sign([('SHOWDOWN!', 0.42, M_TXT_R), ('いざ、勝負！', 0.18, M_TXT)], x_(41.0) + 0.4, -1.8, z_(41.0) - 0.8, 3.0, 0.75)

# ---------------- camera ----------------
FOC = np.zeros((NF, 3)); WID = np.zeros(NF)
rP, gP, bP = POS['red'], POS['gold'], POS['bull']
pv = (rP + gP) / 2
TL = EV['land']['t']; TREL = tb_['t_rel']
for f in range(NF):
    t = T[f]; c = pv[f]; bb = bP[f]
    if t < T0 + 0.4:
        FOC[f] = np.array([-0.6, 0, 1.2]); WID[f] = 5.6
        continue
    if SECS['B3'] - 0.6 < t < TREL + 0.15:                    # the charge: the waiting toreros, the bull storms in from the left
        FOC[f] = np.array([cupc[0] - 1.4, 0, cupc[2] + 1.3]); WID[f] = 6.2
        continue
    if TREL + 0.15 <= t < TL + 0.4:                          # the flight
        FOC[f] = c + np.array([1.0, 0, 0.4]); WID[f] = 6.8
        continue
    if t > FINAL - 1.0:
        FOC[f] = np.array([fc[0] - 0.7, 0, (fc[2] + bfx[2]) / 2 + 0.9]); WID[f] = 5.8
        continue
    dx = bb[0] - c[0]
    near = abs(dx) < 6.5 and abs(bb[2] - c[2]) < 4.0
    FOC[f] = c + np.array([0.9, 0, 0.9])
    if near:
        FOC[f, 0] = 0.6 * (c[0] + 0.9) + 0.4 * (bb[0] + 0.5); FOC[f, 2] = 0.65 * (c[2] + 0.9) + 0.35 * (bb[2] + 0.3)
        WID[f] = max(4.8, abs(dx) + 3.2, abs(bb[2] - c[2]) * 9 / 16 * 1.3 + 3.8)
    else: WID[f] = 4.8
WID = np.clip(WID, 4.6, 8.0)
FOC[:, 0] = smooth(FOC[:, 0], 0.4); FOC[:, 2] = smooth(FOC[:, 2], 0.55); WID = smooth(WID, 0.6)
LENS = 32.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 600
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24
    CP[f] = (FOC[f, 0], -D, FOC[f, 2] + 0.42 * D)
    TP[f] = (FOC[f, 0], 0.9, FOC[f, 2] + 0.2)
key_loc(cam, CP); key_loc(tgt, TP)
SUNX = smooth(FOC[:, 0] + 9.0, 2.0); key_series(sd_, 'location', 0, SUNX)          # the setting sun stays in the background
# spotlights on the toreros and the bull
for n, col, en in (('red', (1.0, 0.85, 0.7), 1800), ('bull', (1.0, 0.6, 0.45), 1400)):
    spd = bpy.data.lights.new('spot' + n, 'SPOT'); spd.energy = en; spd.spot_size = math.radians(18); spd.spot_blend = 0.6
    spd.color = col; spd.shadow_soft_size = 0.3
    so = link(bpy.data.objects.new('spot' + n, spd))
    stg = link(bpy.data.objects.new('spottgt' + n, None))
    P = POS[n].copy(); Ps = P.copy()
    for k in range(3): Ps[:, k] = smooth(P[:, k], 0.25)
    key_loc(stg, Ps - np.array([0, 0, 0.25])); key_loc(so, Ps + np.array([-2.0, -6.0, 9.0]))
    c_ = so.constraints.new('TRACK_TO'); c_.target = stg; c_.track_axis = 'TRACK_NEGATIVE_Z'; c_.up_axis = 'UP_Y'

if MODE == 'test':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS)); sc.frame_set(f)
        sc.render.filepath = f'{HERE}/test_{f:04d}.png'
        t1 = time.time(); bpy.ops.render.render(write_still=True); print('frame', f, 'sec', round(time.time() - t1, 1))
elif MODE == 'range':
    sc.frame_start, sc.frame_end = int(args[1]), int(args[2])
    sc.render.image_settings.file_format = 'PNG'; sc.render.filepath = f'{HERE}/frames/f_'
    bpy.ops.render.render(animation=True)
if MODE == 'probe':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS)); sc.frame_set(f)
        print('t', tt_, 'cam', [round(v, 2) for v in cam.matrix_world.translation], 'WID', round(WID[f], 2), 'FOC', np.round(FOC[f], 2),
              'red', np.round(POS['red'][f], 2), 'bull', np.round(POS['bull'][f], 2))
