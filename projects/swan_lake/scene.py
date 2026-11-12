"""Swan Lake - 'White swan or black swan?'  A moonlit lake: a white pearl (Odette) and a black pearl (Odile) play
Tchaikovsky's oboe theme on floating keys; a glowing moon ring with two gondolas lifts them back up; at the end both draw the
two halves of a giant heart and the lake lights up two swans.  layout.py plans every motion (constant gravity, bounces that never
add speed except on the swan-wing spring keys that visibly flap); this file builds and animates it.
Accompaniment: bass notes = floating bell buoys struck by hammers, chord changes = the lotus lanterns breathe.
Fonts: en_serif.ttf / en_bold.ttf = DejaVu (free licence), jp_sub.otf = Noto Sans CJK JP Black (SIL OFL) subset (make_font.py).
python3 scene.py -- test t1 t2 ... | range f0 f1"""
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
G = LAY['G']; R = LAY['R']; EV = LAY['ev']; RG = LAY['ring']
TN = np.array([l[0] for l in LEAD])
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
sc.render.use_persistent_data = True
F_JP = bpy.data.fonts.load(f'{HERE}/jp_sub.otf'); F_EN = bpy.data.fonts.load(f'{HERE}/en_bold.ttf'); F_SER = bpy.data.fonts.load(f'{HERE}/en_serif.ttf')


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


def glowmat(name, gain=8.0, base=None, emit=None, rough=0.25, metal=0.0, coat=0.6):
    """object alpha drives the emission strength (hit flash); base / emission colour = object colour unless given"""
    m = mat(name, base or (1, 1, 1), rough, **{'Coat Weight': coat, 'Metallic': metal}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    oi = nt.nodes.new('ShaderNodeObjectInfo')
    if base is None: nt.links.new(oi.outputs['Color'], b.inputs['Base Color'])
    if emit is None: nt.links.new(oi.outputs['Color'], b.inputs['Emission Color'])
    else: b.inputs['Emission Color'].default_value = (*emit, 1)
    mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = gain
    nt.links.new(oi.outputs['Alpha'], mul.inputs[0]); nt.links.new(mul.outputs[0], b.inputs['Emission Strength'])
    return m


def fxmat(name, gain=10.0):
    """transparent where object alpha = 0, glowing in the object colour where alpha = 1"""
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    oi = nt.nodes.new('ShaderNodeObjectInfo'); em = nt.nodes.new('ShaderNodeEmission'); tr_ = nt.nodes.new('ShaderNodeBsdfTransparent')
    mx = nt.nodes.new('ShaderNodeMixShader'); mu = nt.nodes.new('ShaderNodeMath'); mu.operation = 'MULTIPLY'; mu.inputs[1].default_value = gain
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(oi.outputs['Color'], em.inputs['Color']); nt.links.new(oi.outputs['Alpha'], mu.inputs[0]); nt.links.new(mu.outputs[0], em.inputs['Strength'])
    cl = nt.nodes.new('ShaderNodeMath'); cl.operation = 'MINIMUM'; cl.inputs[1].default_value = 1.0
    nt.links.new(oi.outputs['Alpha'], cl.inputs[0])
    nt.links.new(cl.outputs[0], mx.inputs['Fac']); nt.links.new(tr_.outputs[0], mx.inputs[1]); nt.links.new(em.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs[0]); return m


def objcol(name, rough=0.45, metal=0.0):
    m = mat(name, (1, 1, 1), rough, **{'Metallic': metal}); nt = m.node_tree
    oi = nt.nodes.new('ShaderNodeObjectInfo'); nt.links.new(oi.outputs['Color'], nt.nodes['Principled BSDF'].inputs['Base Color'])
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
    elif kind == 'plane': bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    elif kind == 'torus':
        R0, r0, n0, n1 = a
        vs = [[bm.verts.new(((R0 + r0 * math.cos(v)) * math.cos(u), (R0 + r0 * math.cos(v)) * math.sin(u), r0 * math.sin(v)))
               for v in np.linspace(0, 2 * math.pi, n1, endpoint=False)] for u in np.linspace(0, 2 * math.pi, n0, endpoint=False)]
        for i in range(n0):
            for j in range(n1):
                bm.faces.new((vs[i][j], vs[(i + 1) % n0][j], vs[(i + 1) % n0][(j + 1) % n1], vs[i][(j + 1) % n1]))
    me = bpy.data.meshes.new(kind); bm.to_mesh(me); bm.free()
    if kind in ('sph', 'cone', 'torus', 'cyl'):
        for p in me.polygons: p.use_smooth = kind != 'cyl'
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


def text(s, loc, size, m, font=None, rot=(math.pi / 2, 0, 0), align='CENTER', ext=0.0):
    cu = bpy.data.curves.new('t', 'FONT'); cu.body = s; cu.font = font or F_EN; cu.size = size; cu.extrude = ext
    cu.align_x = align; cu.align_y = 'CENTER'
    o = link(bpy.data.objects.new('txt', cu)); o.location = loc; o.rotation_euler = rot; o.data.materials.append(m); return o


def pulse(times, decay=5.0, length=16, amp=1.0):
    a = np.zeros(NF)
    for t in times:
        f0 = int(round(t * FPS)); m_ = np.arange(max(0, f0), min(NF, f0 + length))
        a[m_] = np.maximum(a[m_], amp * np.exp(-(m_ - f0) / decay))
    return a


def time_node(nt):
    v = nt.nodes.new('ShaderNodeValue'); fc = v.outputs[0].driver_add('default_value'); fc.driver.expression = 'frame / 30'
    return v.outputs[0]


def mnode(nt, op, a, b=None, c=None):
    m = nt.nodes.new('ShaderNodeMath'); m.operation = op
    for k, x in enumerate((a, b, c)):
        if x is None: continue
        if isinstance(x, (int, float)): m.inputs[k].default_value = float(x)
        else: nt.links.new(x, m.inputs[k])
    return m.outputs[0]


def dots_mat(name, kind, gain=6.0):
    """emissive dots: per-vertex 'ton' (switch-on time) and colour; alpha follows the kind's envelope in the shader"""
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    tm = time_node(nt)
    at = nt.nodes.new('ShaderNodeAttribute'); at.attribute_name = 'ton'; at.attribute_type = 'GEOMETRY'
    ca = nt.nodes.new('ShaderNodeVertexColor'); ca.layer_name = 'col'
    age = mnode(nt, 'SUBTRACT', tm, at.outputs['Fac'])
    on = mnode(nt, 'GREATER_THAN', age, 0.0)
    if kind == 'loop':        # bright when dropped, dim trail, gone after ~6 s
        a = mnode(nt, 'ADD', mnode(nt, 'MULTIPLY', mnode(nt, 'EXPONENT', mnode(nt, 'DIVIDE', age, -1.5)), 0.85), 0.15)
        a = mnode(nt, 'MULTIPLY', a, mnode(nt, 'EXPONENT', mnode(nt, 'DIVIDE', mnode(nt, 'MAXIMUM', mnode(nt, 'SUBTRACT', age, 5.0), 0.0), -0.8)))
    elif kind == 'tutti':     # stays: the heart; flares on the final chord
        a = mnode(nt, 'ADD', mnode(nt, 'MULTIPLY', mnode(nt, 'EXPONENT', mnode(nt, 'DIVIDE', age, -0.7)), 0.5), 0.5)
        a = mnode(nt, 'ADD', a, mnode(nt, 'MULTIPLY', mnode(nt, 'GREATER_THAN', tm, FINAL), 1.0))
    else:                     # lanterns of the two swans: fade in
        a = mnode(nt, 'MULTIPLY', mnode(nt, 'MINIMUM', mnode(nt, 'DIVIDE', age, 0.3), 1.0), 1.5)
    a = mnode(nt, 'MULTIPLY', a, on)
    em = nt.nodes.new('ShaderNodeEmission'); nt.links.new(ca.outputs['Color'], em.inputs['Color'])
    nt.links.new(mnode(nt, 'MULTIPLY', a, gain), em.inputs['Strength'])
    tr_ = nt.nodes.new('ShaderNodeBsdfTransparent'); mx = nt.nodes.new('ShaderNodeMixShader'); out = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(mnode(nt, 'MINIMUM', a, 1.0), mx.inputs['Fac']); nt.links.new(tr_.outputs[0], mx.inputs[1]); nt.links.new(em.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs[0]); return m


def make_dots(name, pts, tons, cols, r, m, flat=False):
    bm = bmesh.new(); ton_l = bm.verts.layers.float.new('ton')
    base = bmesh.new(); bmesh.ops.create_uvsphere(base, u_segments=8, v_segments=4, radius=1.0)
    proto = [v.co.copy() for v in base.verts]; faces = [[v.index for v in f.verts] for f in base.faces]; base.free()
    vcols = []
    for p, t0, c in zip(pts, tons, cols):
        vs = []
        for co in proto:
            v = bm.verts.new((p[0] + co.x * r, p[1] + co.y * r, p[2] + co.z * r * (0.45 if flat else 1.0))); v[ton_l] = float(t0); vs.append(v)
        for fc in faces: bm.faces.new([vs[i] for i in fc])
        vcols += [c] * len(proto)
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    ca = me.color_attributes.new('col', 'FLOAT_COLOR', 'POINT')
    for k, c in enumerate(vcols): ca.data[k].color = (*c, 1)
    for p_ in me.polygons: p_.use_smooth = True
    me.materials.append(m); o = obj(me, name); return o


def smoothstep(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


# ---------------- motion from layout ----------------
RT = np.array(EV['ring']['t']); RTH = np.array(EV['ring']['th'])
def theta(g, t): return float(np.interp(t, RT, RTH)) + (0 if g == 'A' else math.pi)
def pivot(g, t): th = theta(g, t); return np.array([RG['r'] * math.cos(th), 0.0, RG['zc'] + RG['r'] * math.sin(th)])
def at(seg, t):
    for g in seg:
        if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9:
            if g['k'] == 'rest': return np.array(g['p'])
            if g['k'] == 'ride': return pivot(g['g'], t) - np.array([0, 0, RG['hang']])
            u = t - g['t0']; return np.array(g['p']) + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
    return at(seg, seg[0]['t0']) if t < seg[0]['t0'] else at(seg, seg[-1]['t1'])
BALLS = LAY['balls']
POS = {n: np.array([at(b['seg'], t) for t in T]) for n, b in BALLS.items()}
HITS = {n: b['hits'] for n, b in BALLS.items()}

# ---------------- materials ----------------
M_W = mat('pearlW', (0.93, 0.94, 0.97), 0.18, **{'Coat Weight': 1.0, 'Emission Color': (0.75, 0.82, 1.0, 1), 'Emission Strength': 0.35})
M_K = mat('pearlK', (0.025, 0.02, 0.03), 0.12, **{'Coat Weight': 1.0, 'Metallic': 0.3})
nt = M_K.node_tree; bk = nt.nodes['Principled BSDF']; lw = nt.nodes.new('ShaderNodeLayerWeight'); lw.inputs['Blend'].default_value = 0.35
mlt = nt.nodes.new('ShaderNodeMath'); mlt.operation = 'MULTIPLY'; mlt.inputs[1].default_value = 4.0
nt.links.new(lw.outputs['Facing'], mlt.inputs[0]); nt.links.new(mlt.outputs[0], bk.inputs['Emission Strength'])
bk.inputs['Emission Color'].default_value = (0.55, 0.15, 0.95, 1)
M_GOLD = mat('gold', (1.0, 0.72, 0.3), 0.25, **{'Metallic': 0.85})
M_GOLDG = mat('goldglow', (1.0, 0.75, 0.35), 0.3, **{'Metallic': 0.6, 'Emission Color': (1.0, 0.78, 0.4, 1), 'Emission Strength': 1.2})
M_RED = mat('ruby', (0.75, 0.05, 0.25), 0.2, **{'Emission Color': (0.9, 0.1, 0.4, 1), 'Emission Strength': 1.0})
M_KEYW = glowmat('keyW', gain=6.0, rough=0.2)
M_KEYK = glowmat('keyK', gain=9.0, base=(0.03, 0.025, 0.05), rough=0.15, coat=1.0)
M_PLATE = glowmat('plate', gain=5.0, base=(1.0, 0.75, 0.35), emit=(1.0, 0.8, 0.45), rough=0.25, metal=0.8)
M_STEEL = mat('steel', (0.6, 0.62, 0.68), 0.3, **{'Metallic': 0.9})
M_DARK = mat('dark', (0.02, 0.025, 0.035), 0.6)
M_WINGW = mat('wingW', (0.95, 0.95, 0.97), 0.5, **{'Emission Color': (0.8, 0.85, 1.0, 1), 'Emission Strength': 0.25})
M_WINGK = mat('wingK', (0.02, 0.015, 0.03), 0.35, **{'Coat Weight': 1.0})
M_OBJ = objcol('objcol', 0.5)
M_FX = fxmat('fx', 6.0)
M_TXT = emis('txt', (1, 1, 1), 2.2); M_TXT_G = emis('txtg', (1.0, 0.82, 0.45), 2.5); M_TXT_V = emis('txtv', (0.8, 0.55, 1.0), 2.5)
M_TXT_B = emis('txtb', (0.75, 0.88, 1.0), 2.0)

# ---------------- balls ----------------
BALL = {}
for n in ('W', 'K'):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=R, segments=40, ring_count=20); o = bpy.context.object; o.name = n
    for p in o.data.polygons: p.use_smooth = True
    o.data.materials.append(M_W if n == 'W' else M_K)
    # little tiara: Odette's gold crown, Odile's ruby one
    bpy.ops.mesh.primitive_torus_add(major_radius=R * 0.62, minor_radius=0.014, major_segments=24, minor_segments=6, location=(0, 0, R * 0.8))
    tia = bpy.context.object; tia.data.materials.append(M_GOLD if n == 'W' else M_RED); tia.parent = o
    for k in range(5):
        a = 2 * math.pi * k / 5
        sp = sphere((R * 0.62 * math.cos(a), R * 0.62 * math.sin(a), R * 0.8 + 0.03), 0.018, M_GOLDG if n == 'W' else M_RED); sp.parent = o
    key_loc(o, POS[n]); BALL[n] = o

# ---------------- keys (one per hit) ----------------
KEY = {'plate': (0.13, 0.36), 'key': (0.17, 0.5), 'wing': (0.2, 0.62), 'lever': (0.22, 0.44), 'winglever': (0.22, 0.5)}
TH_KEY = 0.07
def key_frame(h):
    n = np.array(h['n'], float); n /= np.linalg.norm(n)
    d = np.array(h['vin'], float) + np.array(h['vout'], float); d[2] = 0
    if np.linalg.norm(d) < 1e-6: d = np.array([1.0, 0, 0])
    xa = d - (d @ n) * n; xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return Matrix(((xa[0], ya[0], n[0]), (xa[1], ya[1], n[1]), (xa[2], ya[2], n[2])))
def pitch_col(p, team):
    h = ((p - 59) % 24) / 24.0
    if team == 'W': return (0.78 + 0.2 * h, 0.85 + 0.1 * math.sin(6 * h), 1.0 - 0.15 * h)
    return (0.55 + 0.35 * h, 0.12 + 0.1 * h, 1.0 - 0.3 * h)
PITCH = {i: l[1] for i, l in enumerate(LEAD)}
KEYS = []
for n in ('W', 'K'):
    for h in HITS[n]:
        k = h['kind']
        if k == 'bell': continue
        nrm = np.array(h['n'], float); nrm /= np.linalg.norm(nrm); c = np.array(h['p']) - R * nrm
        w, d = KEY[k]; ctr = c - nrm * TH_KEY / 2; Mx = key_frame(h)
        mm = M_PLATE if k == 'plate' else (M_KEYW if n == 'W' else M_KEYK)
        o = obj(prim('cube', mm), 'key')
        o.matrix_world = Matrix.Translation(Vector(ctr)) @ Mx.to_4x4() @ Matrix.Diagonal((w, d, TH_KEY, 1))
        col = (1.0, 0.8, 0.45) if k == 'plate' else pitch_col(PITCH[h['i']], n)
        o.color = (*col, 0.12 if k != 'plate' else 0.15)
        KEYS.append(dict(o=o, c=ctr, n=nrm, t=h['t'], kind=k, team=n, i=h['i'], contact=c, M=Mx, w=w, d=d, col=col))
# keys bloom out of the night 1.8 s before they are needed and fade after (the tutti keys stay: they hold the heart)
def show_curve(t):
    t_on = -9 if 'plate' in K_['kind'] else t - 1.8
    t_off = 1e9 if (t >= EV['nearmiss']['t1'] - 0.01 or K_['kind'] == 'plate' and False) else t + 2.6
    if K_['kind'] == 'plate': t_off = t + 3.0
    return np.array([smoothstep((tt - t_on) / 0.45) * (1 - smoothstep((tt - t_off) / 0.45)) for tt in T])
for K_ in KEYS:
    K_['show'] = np.maximum(1e-3, show_curve(K_['t']))
    o = K_['o']; o.scale = (K_['w'], K_['d'], TH_KEY)
    for k, sz in enumerate((K_['w'], K_['d'], TH_KEY)): key_series(o, 'scale', k, sz * K_['show'], 1e-5)
# flash + dip
for K_ in KEYS:
    o = K_['o']; fh = int(round(K_['t'] * FPS)); base = o.location.copy(); nv = Vector(K_['n']); col = K_['col']
    base_a = o.color[3]
    o.color = (*col, base_a); o.keyframe_insert('color', index=3, frame=max(0, fh - 1))
    for df, a in ((0, 1.0), (4, 0.55), (18, 0.3)):
        o.color = (*col, a); o.keyframe_insert('color', index=3, frame=fh + df)
    for df, dd in ((-1, 0), (1, -0.03), (4, 0.01), (8, 0)):
        o.location = base + nv * dd; o.keyframe_insert('location', frame=max(0, fh + df))
# swan-wing spring keys: two feathered wings fold flat beside the key and flap up when the ball lands (the visible launch)
def wing_mesh(side, m):
    bm = bmesh.new()
    pts = [(0, 0), (0.06, 0.22), (0.02, 0.42), (-0.06, 0.55), (-0.12, 0.40), (-0.16, 0.18), (-0.12, 0.0)]
    vs = [bm.verts.new((x, side * y, 0)) for x, y in pts]
    bm.faces.new(vs if side > 0 else vs[::-1])
    me = bpy.data.meshes.new('wing'); bm.to_mesh(me); bm.free(); me.materials.append(m)
    sol = None; return me
WME = {(s, t): wing_mesh(s, M_WINGW if t == 'W' else M_WINGK) for s in (-1, 1) for t in ('W', 'K')}
for K_ in KEYS:
    if 'wing' not in K_['kind']: continue
    Mx = K_['M']; ctr = Vector(K_['c']); fh = int(round(K_['t'] * FPS))
    for side in (-1, 1):
        hinge = link(bpy.data.objects.new('hinge', None))
        hinge.matrix_world = Matrix.Translation(ctr + Mx @ Vector((0.0, side * K_['d'] / 2, TH_KEY / 2))) @ Mx.to_4x4()
        for k in range(3): key_series(hinge, 'scale', k, K_['show'], 1e-4)
        wg = obj(WME[(side, K_['team'])], 'wing'); wg.parent = hinge
        sol = wg.modifiers.new('s', 'SOLIDIFY'); sol.thickness = 0.012
        ang = np.zeros(NF)
        for f in range(max(0, fh - 2), min(NF, fh + 16)):
            u = f - fh
            ang[f] = 0 if u < 0 else (1.25 * min(1, (u + 1) / 3) if u < 3 else 1.25 * math.exp(-(u - 3) / 4))
        hinge.rotation_mode = 'XYZ'
        rot0 = hinge.rotation_euler.copy()
        # flap about the key's travel axis (local x): rebuild the local rotation per frame with a child empty
        flap = link(bpy.data.objects.new('flap', None)); flap.parent = hinge; wg.parent = flap
        key_series(flap, 'rotation_euler', 0, side * ang)

# ---------------- ripples on the water under every hit + under the balls' landings in gondolas ----------------
NRING = 18
ring_loc = np.zeros((NRING, NF, 3)); ring_scl = np.full((NRING, NF), 1e-3); ring_a = np.zeros((NRING, NF)); ring_c = [(1, 1, 1)] * NRING
for k, K_ in enumerate(sorted(KEYS, key=lambda b: b['t'])):
    r = k % NRING; f0 = int(round(K_['t'] * FPS))
    for df in range(36):
        f = f0 + df
        if f >= NF: break
        u = df / 35; ring_loc[r, f] = (K_['c'][0], K_['c'][1], 0.012); ring_scl[r, f] = 0.25 + 1.6 * (1 - (1 - u) ** 2)
        ring_a[r, f] = 0.7 * (1 - u) ** 1.5
    ring_c[r] = (0.75, 0.85, 1.0) if K_['team'] == 'W' else (0.7, 0.35, 1.0)
for r in range(NRING):
    o = obj(prim('torus', M_FX, 0.3, 0.012, 48, 6), 'ripple'); o.color = (*ring_c[r], 0)
    for f in range(1, NF):
        if ring_a[r, f] == 0: ring_loc[r, f] = ring_loc[r, f - 1]
    key_loc(o, ring_loc[r])
    for k in range(2): key_series(o, 'scale', k, ring_scl[r])
    key_series(o, 'color', 3, ring_a[r], 1e-3)

# ---------------- light trails: dots dropped along every flight (loops fade, the tutti heart stays) ----------------
T_TU = EV['nearmiss']['t1']
M_TRL = dots_mat('trailLoop', 'loop', 5.0); M_TRT = dots_mat('trailTutti', 'tutti', 5.0)
for n in ('W', 'K'):
    pts = []; last = None
    for g in BALLS[n]['seg']:
        if g['k'] != 'fly': continue
        for t in np.arange(g['t0'], g['t1'], 1 / 120):
            u = t - g['t0']; p = np.array(g['p']) + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
            if last is None or np.linalg.norm(p - last) > (0.14 if t < T_TU else 0.1): pts.append((t, p)); last = p
    col = (0.85, 0.92, 1.0) if n == 'W' else (0.75, 0.35, 1.0)
    for grp, m_ in ((False, M_TRL), (True, M_TRT)):
        sel = [(t, p) for t, p in pts if (t >= T_TU) == grp]
        make_dots('trail' + n, [p for t, p in sel], [t for t, p in sel], [col] * len(sel), 0.035, m_)

# ---------------- the moon ring: two rings of light, beads, rollers, gondolas ----------------
ZC, RR, HG = RG['zc'], RG['r'], RG['hang']
M_RING = mat('ring', (1.0, 0.86, 0.6), 0.25, **{'Metallic': 0.7, 'Emission Color': (1.0, 0.85, 0.55, 1), 'Emission Strength': 1.4})
rg = obj(prim('torus', M_RING, RR, 0.1, 128, 12), 'ring'); rg.location = (0, 0, ZC); rg.rotation_euler = (math.pi / 2, 0, 0)
spin = link(bpy.data.objects.new('ringspin', None)); spin.location = (0, 0, ZC); spin.rotation_mode = 'XYZ'
M_BEAD = emis('bead', (1.0, 0.9, 0.7), 6.0)
for k in range(36):
    a = 2 * math.pi * k / 36
    b_ = sphere((RR * math.cos(a), -0.11, RR * math.sin(a)), 0.03, M_BEAD, (8, 4)); b_.parent = spin
    b2 = sphere((RR * math.cos(a), 0.11, RR * math.sin(a)), 0.03, M_BEAD, (8, 4)); b2.parent = spin
th0 = theta('A', 0.0)
key_series(spin, 'rotation_euler', 1, np.array([-(theta('A', t) - th0) for t in T]))
for a in (-math.pi / 2 - 0.45, -math.pi / 2 + 0.45):          # rollers carrying the ring (outside, below)
    c = np.array([(RR + 0.22) * math.cos(a), 0, ZC + (RR + 0.22) * math.sin(a)])
    rod(c + [0, -0.18, 0], c + [0, 0.18, 0], 0.12, M_GOLD, 20)
    rod(c, (c[0], 0.0, -0.5), 0.05, M_GOLD, 8)
c = np.array([0, 0, ZC + RR + 0.24]); rod(c + [0, -0.18, 0], c + [0, 0.18, 0], 0.12, M_GOLD, 20)    # top guide roller
for sgn in (-1, 1):                                               # the guide roller's arch behind the ring
    rod((0, 0.25, ZC + RR + 0.24), (sgn * (RR + 0.7), 0.25, ZC + 0.6), 0.05, M_GOLD, 8)
    rod((sgn * (RR + 0.7), 0.25, ZC + 0.6), (sgn * (RR + 0.7), 0.25, -0.5), 0.06, M_GOLD, 8)
GOND = {}
REL = {'A': [EV['lever_A2']['t'], EV['lever_B2']['t']], 'B': [EV['lever_B']['t']]}
for g in ('A', 'B'):
    gp = link(bpy.data.objects.new('gond' + g, None))
    key_loc(gp, np.array([pivot(g, t) for t in T]))
    cup_z = -HG - R - 0.03
    bm = bmesh.new(); bmesh.ops.create_cone(bm, cap_ends=False, segments=24, radius1=0.2, radius2=0.24, depth=0.2)
    me = bpy.data.meshes.new('cup'); bm.to_mesh(me); bm.free(); cu = obj(me, 'cup'); cu.data.materials.append(M_GOLD)
    cu.location = (0, 0, cup_z + 0.1); cu.parent = gp
    s_ = cu.modifiers.new('s', 'SOLIDIFY'); s_.thickness = 0.02
    rod((0, 0, 0), (0, 0.24, -0.15), 0.018, M_GOLD, 8).parent = gp
    rod((0, 0.24, -0.15), (0, 0.24, cup_z + 0.2), 0.018, M_GOLD, 8).parent = gp
    sphere((0, 0, 0), 0.05, M_GOLDG).parent = gp
    # trapdoor floor: hinged at the back, swings down at each release
    fl_h = link(bpy.data.objects.new('flh', None)); fl_h.parent = gp; fl_h.location = (0, 0.2, cup_z)
    fl = obj(prim('cyl', M_GOLD, 24), 'floor'); fl.parent = fl_h; fl.location = (0, -0.2, 0); fl.scale = (0.2, 0.2, 0.02)
    ang = np.zeros(NF)
    for tr in REL[g]:
        for f in range(NF):
            u = T[f] - tr
            if -0.05 < u < 0.9: ang[f] = max(ang[f], 1.4 * min(1, (u + 0.05) / 0.12) * (1 if u < 0.5 else max(0, 1 - (u - 0.5) / 0.4)))
    fl_h.rotation_mode = 'XYZ'; key_series(fl_h, 'rotation_euler', 0, -ang)
    GOND[g] = gp

# ---------------- crescent cradle + harps (intro) ----------------
cr = EV['cradle']; CW = np.array(cr['W']); CK = np.array(cr['K'])
M_MOONC = mat('moonc', (1.0, 0.93, 0.7), 0.3, **{'Emission Color': (1.0, 0.9, 0.6, 1), 'Emission Strength': 2.5})
bm = bmesh.new()
for k in range(40):                                             # crescent: two arcs (outer / inner) as a closed flat band, extruded
    pass
outer = [(1.05 * math.cos(a), 0, 1.05 * math.sin(a)) for a in np.linspace(math.pi * 1.05, math.pi * 1.95, 40)]
inner = [(0.86 * math.cos(a), 0, 0.86 * math.sin(a) - 0.12) for a in np.linspace(math.pi * 1.95, math.pi * 1.05, 40)]
vs = [bm.verts.new(p) for p in outer + inner]; bm.faces.new(vs)
me = bpy.data.meshes.new('cres'); bm.to_mesh(me); bm.free()
cres = obj(me, 'crescent'); cres.data.materials.append(M_MOONC); cres.location = (0, 0.0, CW[2] + 1.3)
so_ = cres.modifiers.new('s', 'SOLIDIFY'); so_.thickness = 0.25; so_.offset = 0
for tr, p in ((cr['tW'], CW), (cr['tK'], CK)):                  # a gold cup under each ball, the floor opens
    bm = bmesh.new(); bmesh.ops.create_cone(bm, cap_ends=False, segments=24, radius1=0.17, radius2=0.21, depth=0.14)
    me = bpy.data.meshes.new('ccup'); bm.to_mesh(me); bm.free(); cu = obj(me, 'ccup'); cu.data.materials.append(M_GOLD)
    cu.location = (p[0], 0, p[2] - R + 0.04); s_ = cu.modifiers.new('s', 'SOLIDIFY'); s_.thickness = 0.02
    rod((p[0] * 1.15, 0, p[2] - 0.05), (p[0] * 1.6, 0, CW[2] + 0.38), 0.015, M_GOLD, 6)
    fh_ = link(bpy.data.objects.new('cfh', None)); fh_.location = (p[0] + (0.17 if p[0] > 0 else -0.17), 0, p[2] - R - 0.03)
    fl = obj(prim('cyl', M_GOLD, 24), 'cfloor'); fl.parent = fh_; fl.location = (-0.17 if p[0] > 0 else 0.17, 0, 0); fl.scale = (0.17, 0.17, 0.015)
    ang = np.array([1.5 * smoothstep((t - tr + 0.04) / 0.1) for t in T])
    fh_.rotation_mode = 'XYZ'; key_series(fh_, 'rotation_euler', 1, ang * (1 if p[0] > 0 else -1))
# harps: plates hang on gold strings from a golden arch
for n in ('W', 'K'):
    pl = [K_ for K_ in KEYS if K_['team'] == n and K_['kind'] == 'plate']
    rail = []
    for K_ in pl:                                    # each plate hangs from a gold rail behind it on a short stem
        c = np.array(K_['c']); q = c + np.array(K_['M'] @ Vector((0, 0.17, 0))); q = q + np.array([0, 0.12, 0])
        rod(c + np.array(K_['M'] @ Vector((0, 0.16, 0))), q + [0, 0, 0.3], 0.008, M_GOLDG, 6); rail.append(q + [0, 0, 0.3])
    for a_, b_ in zip(rail, rail[1:]): rod(a_, b_, 0.022, M_GOLD, 8)
    rod(rail[0], (rail[0][0] * 0.4, 0.1, CW[2] + 0.35), 0.022, M_GOLD, 8)
    sphere(rail[-1], 0.05, M_GOLDG)

# ---------------- the black swan's reed nest + gate, levers, wires ----------------
ne = EV['nest']; NP = np.array(ne['p'])
M_REED = mat('reed', (0.05, 0.06, 0.04), 0.7)
bm = bmesh.new(); bmesh.ops.create_cone(bm, cap_ends=False, segments=28, radius1=0.2, radius2=0.34, depth=0.24)
me = bpy.data.meshes.new('nest'); bm.to_mesh(me); bm.free(); nst = obj(me, 'nest'); nst.data.materials.append(M_REED)
nst.location = (NP[0], NP[1], NP[2] - R + 0.08); s_ = nst.modifiers.new('s', 'SOLIDIFY'); s_.thickness = 0.04
for k in range(14):                                               # reeds around it
    a = 2 * math.pi * k / 14; r_ = 0.36 + 0.05 * (k % 3)
    rod((NP[0] + r_ * math.cos(a), NP[1] + r_ * math.sin(a), 0.0), (NP[0] + (r_ + 0.06) * math.cos(a), NP[1] + (r_ + 0.06) * math.sin(a), NP[2] + 0.2 + 0.25 * (k % 2)), 0.012, M_REED, 5)
gate_h = link(bpy.data.objects.new('gh', None)); gate_h.location = (NP[0] + 0.2, NP[1], NP[2] - R - 0.035)
gf = obj(prim('cyl', M_REED, 24), 'gate'); gf.parent = gate_h; gf.location = (-0.2, 0, 0); gf.scale = (0.2, 0.2, 0.02)
gate_h.rotation_mode = 'XYZ'; key_series(gate_h, 'rotation_euler', 1, np.array([1.5 * smoothstep((t - ne['t_open'] + 0.03) / 0.1) for t in T]))
LBL = text('Odile', (NP[0] + 0.55, NP[1] - 0.1, NP[2] + 0.55), 0.26, M_TXT_V, font=F_SER, align='LEFT')
LBLj = text('黒鳥オディール', (NP[0] + 0.55, NP[1] - 0.1, NP[2] + 0.25), 0.16, M_TXT_V, font=F_JP, align='LEFT')
# wires: each lever key flashes a wire to what it releases
M_WIRE = fxmat('wire', 4.0)
WIRES = [(EV['lever_A'], [NP[0], NP[1], NP[2] - 0.3], (0.8, 0.9, 1.0)),
         (EV['lever_A2'], [0, 0.3, ZC + RR + 0.24], (0.75, 0.4, 1.0)),
         (EV['lever_B'], [0, 0.3, ZC + RR + 0.24], (0.8, 0.9, 1.0)),
         (EV['lever_B2'], [0, 0.3, ZC + RR + 0.24], (0.75, 0.4, 1.0))]
for lv, tgt, col in WIRES:
    p0 = np.array(lv['p']) - np.array([0, 0, R + TH_KEY]); p1 = np.array(tgt)
    mid = (p0 + p1) / 2 + np.array([0, 0.5, 0.0]); pts = [p0 + (mid - p0) * u * 2 if u < 0.5 else mid + (p1 - mid) * (u - 0.5) * 2 for u in np.linspace(0, 1, 9)]
    base_a = pulse([lv['t']], decay=6, length=24, amp=1.0) + 0.06
    for a_, b_ in zip(pts, pts[1:]):
        o = rod(a_, b_, 0.012, M_WIRE, 6); o.color = (*col, 0.06); key_series(o, 'color', 3, base_a, 1e-3)

# ---------------- final bell: a golden lotus bell at the tip of the heart ----------------
bl = EV['bell']; BP_ = np.array(bl['p'])
M_BELL = glowmat('bell', gain=7.0, base=(1.0, 0.78, 0.38), emit=(1.0, 0.85, 0.5), rough=0.2, metal=0.85)
bel = obj(prim('cone', M_BELL, 40, 0.75, 0.22, 0.6), 'bell'); bel.location = (BP_[0], BP_[1], BP_[2] - R - 0.32); bel.color = (1, 0.8, 0.4, 0.1)
key_series(bel, 'color', 3, 0.1 + pulse([FINAL], decay=25, length=140, amp=1.2))
for sx in (-1, 1):
    bm = bmesh.new(); bmesh.ops.create_cone(bm, cap_ends=True, segments=24, radius1=0.16, radius2=0.2, depth=0.08)
    me = bpy.data.meshes.new('bcup'); bm.to_mesh(me); bm.free(); o = obj(me, 'bcup'); o.data.materials.append(M_GOLD)
    o.location = (sx * 0.4, BP_[1], BP_[2] - R - 0.02)
M_PETAL = mat('petal', (1.0, 0.85, 0.9), 0.4, **{'Emission Color': (1.0, 0.7, 0.85, 1), 'Emission Strength': 0.8})
for k in range(10):
    a = 2 * math.pi * k / 10
    pv = link(bpy.data.objects.new('petal', None)); pv.location = (BP_[0] + 0.7 * math.cos(a), BP_[1] + 0.7 * math.sin(a), BP_[2] - R - 0.6)
    pv.rotation_euler = (0, 0, a)
    pe = sphere((0.35, 0, 0), 1.0, M_PETAL, (16, 8)); pe.scale = (0.45, 0.16, 0.05); pe.parent = pv; pe.rotation_euler = (0, -0.5, 0)
rod((BP_[0], BP_[1], BP_[2] - R - 0.6), (BP_[0], BP_[1], -0.3), 0.06, M_GOLD, 10)

# ---------------- bass buoys: floating bells struck by hammers on every bass note ----------------
BASS = E['bass']; PITCHES = sorted(set(p for t, p, v in BASS))
M_BUOY = glowmat('buoy', gain=6.0, base=(0.55, 0.62, 0.8), emit=(0.7, 0.8, 1.0), rough=0.25, metal=0.8)
for j, pb in enumerate(PITCHES):
    a = math.pi * (0.18 + 0.64 * j / (len(PITCHES) - 1)); rr = 4.4
    x, y = -rr * math.cos(a), 0.6 + 1.6 * math.sin(a)
    sz = 0.32 - 0.012 * (pb - PITCHES[0])
    rod((x, y, -0.2), (x, y, 0.12), sz * 1.1, M_DARK, 20)
    b_ = obj(prim('cone', M_BUOY, 28, sz, sz * 0.45, sz * 1.2), 'buoybell'); b_.location = (x, y, 0.12 + sz * 0.6); b_.color = (0.7, 0.8, 1, 0.05)
    tms = [t for t, p, v in BASS if p == pb]
    key_series(b_, 'color', 3, 0.05 + pulse(tms, decay=4, length=20, amp=1.0), 1e-3)
    piv = link(bpy.data.objects.new('hpiv', None)); piv.location = (x, y + 0.45, 0.12 + sz * 1.2 + 0.35)
    arm = rod((0, 0, 0), (0, -0.42, -0.3), 0.015, M_GOLD, 6); arm.parent = piv
    hd = sphere((0, -0.44, -0.32), 0.05, M_GOLD); hd.parent = piv
    up = np.zeros(NF); tms_a = np.array(tms)
    for f in range(NF):
        t = T[f]; k = np.searchsorted(tms_a, t); nxt = tms_a[k] if k < len(tms_a) else 1e9
        prv = tms_a[k - 1] if k > 0 else -1e9
        if nxt - t < 0.25: up[f] = 0.6 * ((nxt - t) / 0.25)
        elif t - prv < 0.25: up[f] = 0.6 * ((t - prv) / 0.25) ** 0.6
        else: up[f] = 0.6
    piv.rotation_mode = 'XYZ'; key_series(piv, 'rotation_euler', 0, up)
    rt = np.zeros(NF)

# ---------------- lake, sky, moon, shore ----------------
M_WATER = bpy.data.materials.new('water'); M_WATER.use_nodes = True; nt = M_WATER.node_tree
bw = nt.nodes['Principled BSDF']; bw.inputs['Base Color'].default_value = (0.005, 0.012, 0.03, 1); bw.inputs['Roughness'].default_value = 0.035
nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 3.5; nz.inputs['Detail'].default_value = 4
tco = nt.nodes.new('ShaderNodeTexCoord'); mp_ = nt.nodes.new('ShaderNodeMapping'); mp_.inputs['Scale'].default_value = (1, 3, 1)
nt.links.new(tco.outputs['Object'], mp_.inputs[0]); nt.links.new(mp_.outputs[0], nz.inputs['Vector'])
bump = nt.nodes.new('ShaderNodeBump'); bump.inputs['Strength'].default_value = 0.06; bump.inputs['Distance'].default_value = 0.02
nt.links.new(nz.outputs['Fac'], bump.inputs['Height']); nt.links.new(bump.outputs['Normal'], bw.inputs['Normal'])
lake = obj(prim('plane', M_WATER), 'lake'); lake.scale = (600, 600, 1)
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True; nt = w.node_tree
bg = nt.nodes['Background']
tcw = nt.nodes.new('ShaderNodeTexCoord'); sep = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(tcw.outputs['Generated'], sep.inputs[0])
grad = nt.nodes.new('ShaderNodeValToRGB'); grad.color_ramp.elements[0].color = (0.02, 0.05, 0.12, 1); grad.color_ramp.elements[1].color = (0.003, 0.006, 0.02, 1)
grad.color_ramp.elements[0].position = 0.0; grad.color_ramp.elements[1].position = 0.5
nt.links.new(sep.outputs['Z'], grad.inputs['Fac'])
vor = nt.nodes.new('ShaderNodeTexVoronoi'); vor.inputs['Scale'].default_value = 180.0; nt.links.new(tcw.outputs['Generated'], vor.inputs['Vector'])
mr = nt.nodes.new('ShaderNodeMapRange'); mr.inputs['From Min'].default_value = 0.0; mr.inputs['From Max'].default_value = 0.07
mr.inputs['To Min'].default_value = 1.0; mr.inputs['To Max'].default_value = 0.0
nt.links.new(vor.outputs['Distance'], mr.inputs['Value'])
pw_ = nt.nodes.new('ShaderNodeMath'); pw_.operation = 'POWER'; pw_.inputs[1].default_value = 6.0; nt.links.new(mr.outputs[0], pw_.inputs[0])
up_ = nt.nodes.new('ShaderNodeMath'); up_.operation = 'GREATER_THAN'; up_.inputs[1].default_value = 0.0; nt.links.new(sep.outputs['Z'], up_.inputs[0])
st = nt.nodes.new('ShaderNodeMath'); st.operation = 'MULTIPLY'; nt.links.new(pw_.outputs[0], st.inputs[0]); nt.links.new(up_.outputs[0], st.inputs[1])
st2 = nt.nodes.new('ShaderNodeMath'); st2.operation = 'MULTIPLY'; st2.inputs[1].default_value = 1.6; nt.links.new(st.outputs[0], st2.inputs[0])
add = nt.nodes.new('ShaderNodeMixRGB'); add.blend_type = 'ADD'; add.inputs['Fac'].default_value = 1.0
nt.links.new(grad.outputs['Color'], add.inputs[1]); nt.links.new(st2.outputs[0], add.inputs[2])
nt.links.new(add.outputs['Color'], bg.inputs['Color']); bg.inputs['Strength'].default_value = 1.0
# the moon: big disc low over the far shore, halo, its reflection does the rest
MOONP = np.array([0.0, 170.0, 46.0])
M_MOON = emis('moon', (1.0, 0.95, 0.82), 5.0)
mo = obj(prim('cyl', M_MOON, 64), 'moon'); mo.location = tuple(MOONP); mo.rotation_euler = (math.pi / 2, 0, 0); mo.scale = (6.5, 6.5, 0.1)
HALO = bpy.data.materials.new('halo'); HALO.use_nodes = True; nt = HALO.node_tree; nt.nodes.clear()
tc_ = nt.nodes.new('ShaderNodeTexCoord'); gr = nt.nodes.new('ShaderNodeTexGradient'); gr.gradient_type = 'SPHERICAL'
em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Color'].default_value = (0.6, 0.7, 1.0, 1)
mx = nt.nodes.new('ShaderNodeMixShader'); trn = nt.nodes.new('ShaderNodeBsdfTransparent'); out = nt.nodes.new('ShaderNodeOutputMaterial')
pw2 = nt.nodes.new('ShaderNodeMath'); pw2.operation = 'POWER'; pw2.inputs[1].default_value = 3.0
nt.links.new(tc_.outputs['Object'], gr.inputs['Vector']); nt.links.new(gr.outputs['Fac'], pw2.inputs[0])
ms = nt.nodes.new('ShaderNodeMath'); ms.operation = 'MULTIPLY'; ms.inputs[1].default_value = 1.4
nt.links.new(pw2.outputs[0], ms.inputs[0]); nt.links.new(ms.outputs[0], em.inputs['Strength'])
nt.links.new(pw2.outputs[0], mx.inputs['Fac']); nt.links.new(trn.outputs[0], mx.inputs[1]); nt.links.new(em.outputs[0], mx.inputs[2])
nt.links.new(mx.outputs[0], out.inputs[0])
ha = obj(prim('plane', HALO), 'halo'); ha.location = tuple(MOONP + [0, 2, 0]); ha.rotation_euler = (math.pi / 2, 0, 0); ha.scale = (60, 60, 1)
# far shore: dark hills + pine silhouettes all around
M_HILL = mat('hill', (0.004, 0.006, 0.012), 0.9)
for k in range(64):
    a = 2 * math.pi * k / 64; r_ = 140 + 25 * math.sin(3 * a) + 10 * math.sin(7 * a)
    h_ = 9 + 6 * math.sin(5 * a + 1) + 4 * math.sin(11 * a)
    o = obj(prim('cone', M_HILL, 8, 1.0, 0.0, 1.0), 'hill'); o.location = (r_ * math.sin(a), r_ * math.cos(a), h_ / 2 - 1)
    o.scale = (28, 18, h_)
for k in range(260):
    a = rng.uniform(0, 2 * math.pi); r_ = rng.uniform(95, 125); h_ = rng.uniform(4, 9)
    o = obj(prim('cone', M_HILL, 6, 1.0, 0.0, 1.0), 'pine'); o.location = (r_ * math.sin(a), r_ * math.cos(a), h_ / 2 - 0.5); o.scale = (h_ * 0.22, h_ * 0.22, h_)
sun = bpy.data.lights.new('moonlight', 'SUN'); sun.energy = 0.45; sun.angle = 0.02; sun.color = (0.7, 0.8, 1.0)
so = link(bpy.data.objects.new('moonlight', sun)); so.rotation_euler = (math.radians(-62), 0, math.radians(180))
fill = bpy.data.lights.new('fill', 'AREA'); fill.energy = 900; fill.size = 30; fill.color = (0.55, 0.65, 1.0)
fo = link(bpy.data.objects.new('fill', fill)); fo.location = (0, -30, 25); fo.rotation_euler = (math.radians(50), 0, 0)

# ---------------- lotus lanterns: breathe on every chord change; the two swans of lanterns light up at the end ----------------
CHT = [t for t, c in E['chords']]
M_LAN = bpy.data.materials.new('lantern'); M_LAN.use_nodes = True; nt = M_LAN.node_tree; nt.nodes.clear()
vlan = nt.nodes.new('ShaderNodeValue')
lan_a = 0.25 + 0.35 * pulse(CHT, decay=10, length=40, amp=1.0); lan_a[T >= FINAL] += 1.0
for f in range(NF):
    if 0 < f < NF - 1 and abs(lan_a[f] - lan_a[f - 1]) < 2e-3 and abs(lan_a[f] - lan_a[f + 1]) < 2e-3: continue
    vlan.outputs[0].default_value = float(lan_a[f]); vlan.outputs[0].keyframe_insert('default_value', frame=f)
em_ = nt.nodes.new('ShaderNodeEmission'); em_.inputs['Color'].default_value = (1.0, 0.72, 0.4, 1)
nt.links.new(mnode(nt, 'MULTIPLY', vlan.outputs[0], 7.0), em_.inputs['Strength'])
o_ = nt.nodes.new('ShaderNodeOutputMaterial'); nt.links.new(em_.outputs[0], o_.inputs[0])
HEART = np.array(EV['heart']); TIP = HEART[-1]
lp = []
while len(lp) < 70:
    x, y = rng.uniform(-16, 16), rng.uniform(-22, 9)
    if abs(x) < 3.2 and abs(y) < 2.5: continue
    lp.append((x, y, 0.02))
make_dots('lanterns', lp, [0] * len(lp), [(1, 1, 1)] * len(lp), 0.09, M_LAN, flat=True)
# swan outlines (bodies + heads) around the heart: right swan, then mirrored
def hp(t): return np.array([16 * math.sin(t) ** 3, 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t) - 5])
SCH = float(np.linalg.norm(HEART[1] - HEART[0]) / np.linalg.norm(hp(math.pi / (len(HEART) - 1)) - hp(0))) if False else None
xs = HEART[:, 0]; ys = HEART[:, 1]
def heart_at_y(y, side=1):
    k = np.argmin(np.abs(ys - y) + (xs < 0) * 99); return HEART[k]
Hlow = HEART[int(len(HEART) * 0.72)]; Hmid = HEART[int(len(HEART) * 0.5)]
body = [Hlow, Hlow + [2.6, -0.3], Hlow + [5.0, 0.6], Hlow + [6.8, 2.6], Hlow + [7.4, 4.6], Hlow + [6.2, 4.4],   # belly -> tail
        Hlow + [5.0, 5.0], Hlow + [3.4, 6.4], Hlow + [1.8, 6.6], Hmid + [0.4, -0.2]]                                    # wing back to the neck
from math import comb
def catmull(P, n=24):
    P = np.array(P, float); P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]]); out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for u in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    out.append(P[-2]); return np.array(out)
wingl = [Hlow + [1.6, 1.6], Hlow + [3.4, 2.4], Hlow + [5.2, 3.6]]
head = [(0.25, 0.9), (0.55, 1.35), (1.05, 1.5), (1.5, 1.25), (1.55, 0.75)]   # head crown near the cusp (the heart line is the neck)
beak = [(0.25, 0.9), (0.0, 0.62), (0.3, 0.72)]
SW_PTS = []
for poly, spacing in ((catmull(body), 0.32), (catmull(wingl), 0.32), (catmull(head, 10), 0.18), (np.array(beak), 0.12)):
    s_ = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))])
    for d in np.arange(0, s_[-1] + 1e-6, spacing):
        SW_PTS.append(np.array([np.interp(d, s_, poly[:, 0]), np.interp(d, s_, poly[:, 1])]))
EYE = np.array([1.05, 1.15])
M_SWAN = dots_mat('swanlan', 'swan', 7.0)
sp_, st_, sc_ = [], [], []
for sgn, col in ((1, (0.9, 0.95, 1.0)), (-1, (0.8, 0.45, 1.0))):
    for j, p in enumerate(SW_PTS + [EYE]):
        x, y = sgn * p[0], p[1]
        sp_.append((x, y, 0.03)); st_.append(FINAL + 0.15 + 0.06 * np.linalg.norm(np.array([x, y]) - TIP)); sc_.append(col)
make_dots('swans', sp_, st_, sc_, 0.11, M_SWAN, flat=True)

# ---------------- signs on the lake (keywords inside the first 30 s) ----------------
M_SIGN = mat('sign', (0.12, 0.08, 0.05), 0.6)
def sign(loc, lines, w_, rotz=0.0):
    pv = link(bpy.data.objects.new('signp', None)); pv.location = loc; pv.rotation_euler = (0, 0, rotz)
    hgt = sum(s[1] * 1.35 for s in lines) + 0.25
    b_ = box((0, 0.03, -hgt / 2 + 0.12), (w_, 0.05, hgt), M_SIGN); b_.parent = pv
    rod((0, 0.06, -hgt + 0.12), (0, 0.06, -loc[2] - 0.3), 0.04, M_SIGN, 6).parent = pv
    z = 0.0
    for s_, size, m_, f_ in lines:
        o = text(s_, (0, -0.01, z), size, m_, font=f_); o.parent = pv; z -= size * 1.35
    return pv
sign((-3.3, -0.3, 2.6), [('SWAN LAKE', 0.36, M_TXT_G, F_SER), ('Tchaikovsky', 0.24, M_TXT, F_SER), ('白鳥の湖 チャイコフスキー', 0.2, M_TXT, F_JP)], 2.6, rotz=0.15)
sign((4.6, 0.7, 2.2), [('Marble machine', 0.26, M_TXT_B, F_EN), ('Classical · Ballet', 0.22, M_TXT, F_EN), ('ビー玉からくり・クラシック', 0.2, M_TXT, F_JP)], 2.7, rotz=-0.15)
text('Odette', (-1.7, -0.4, 8.25), 0.24, M_TXT_B, font=F_SER, align='RIGHT')
text('白鳥オデット', (-1.7, -0.4, 7.98), 0.15, M_TXT_B, font=F_JP, align='RIGHT')

# ---------------- camera ----------------
def smooth(Xa, k):
    Y = Xa.copy(); a = 1 - math.exp(-1 / FPS / k)
    for f in range(1, len(Xa)): Y[f] = Y[f - 1] + (Xa[f] - Y[f - 1]) * a
    for f in range(len(Xa) - 2, -1, -1): Y[f] = Y[f + 1] + (Y[f] - Y[f + 1]) * a
    return Y
SEC = {s: t for s, t in E['secs']}
def owner(t):
    if t < 2.45: return 'cradle'
    if t < SEC['A2'] - 0.2: return 'W'
    if t < SEC['B'] - 0.35: return 'K'
    if t < SEC['B2'] - 0.35: return 'W'
    if t < EV['nearmiss']['t0'] - 0.5: return 'K'
    if t < SEC['tutti'] + 0.6: return 'near'
    if t < 58.4: return 'W'
    if t < 63.0: return 'K'
    return 'both'
FOC = np.zeros((NF, 3)); WID = np.zeros(NF)
for f in range(NF):
    t = T[f]; o_ = owner(t)
    if o_ == 'cradle': FOC[f] = (0.0, -0.5, 7.7); WID[f] = 4.4
    elif o_ == 'near': FOC[f] = (0.0, -0.2, 5.2); WID[f] = 3.6
    elif o_ == 'both': FOC[f] = (POS['W'][f] + POS['K'][f]) / 2; WID[f] = max(4.6, abs(POS['W'][f][0] - POS['K'][f][0]) + 3.0)
    else: FOC[f] = POS[o_][f]; WID[f] = 4.4
    if o_ in ('W', 'K'): FOC[f][2] = FOC[f][2] * 0.8 + 0.6
TF = FINAL + 0.4
FOC[:, 0] = smooth(FOC[:, 0], 0.45); FOC[:, 1] = smooth(FOC[:, 1], 0.45); FOC[:, 2] = smooth(FOC[:, 2], 0.7); WID = smooth(np.clip(WID, 3.4, 9.0), 0.6)
LENS = 30.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 1000
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam; cam.rotation_mode = 'XYZ'
CP = np.zeros((NF, 3)); CR_ = np.zeros((NF, 3))
TOPC = np.array([0.0, -5.6, 56.0])
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24
    pitch = math.radians(14); yaw = 0.0
    cp = FOC[f] + np.array([-math.sin(yaw) * math.cos(pitch) * D, -math.cos(yaw) * math.cos(pitch) * D, math.sin(pitch) * D])
    rot = np.array([math.pi / 2 - pitch, 0, -yaw])
    u = smoothstep((T[f] - TF) / 2.3)
    if u > 0:
        cp = cp * (1 - u) + TOPC * u
        rot = rot * (1 - u) + np.array([0.0, 0, 0]) * u
    CP[f] = cp; CR_[f] = rot
key_loc(cam, CP)
for k in range(3): key_series(cam, 'rotation_euler', k, CR_[:, k], 1e-6)

# ---------------- HUD captions (English first, Japanese below) ----------------
hud = link(bpy.data.objects.new('hud', None)); hud.parent = cam; hud.location = (0, 0, -3.0)
HH = 3.0 * 24 / LENS; HWd = HH * 9 / 16
def cap(lines, y, t0, t1, fade=0.25):
    grp = link(bpy.data.objects.new('cap', None)); grp.parent = hud; grp.location = (0, y, 0)
    yy = 0.0
    for s_, size, m_, f_ in lines:
        o = text(s_, (0, yy, 0), size * HH, m_, font=f_, rot=(0, 0, 0)); o.parent = grp
        bpy.context.view_layer.update()
        dims = o.dimensions
        if dims.x > HWd * 0.92: o.scale = (HWd * 0.92 / dims.x,) * 3
        yy -= size * HH * 1.3
    s = np.array([smoothstep((t - t0) / fade) * (1 - smoothstep((t - t1) / fade)) for t in T])
    for k in range(3): key_series(grp, 'scale', k, np.maximum(1e-3, s), 1e-3)
    return grp
TOPY = HH * 0.38; BOTY = -HH * 0.33
cap([('White swan or black swan?', 0.052, M_TXT, F_SER), ('白鳥と黒鳥、どっち？', 0.064, M_TXT_G, F_JP)], TOPY, -1, 3.5)
cap([('SWAN LAKE', 0.07, M_TXT_G, F_SER), ('Tchaikovsky · Classical music', 0.032, M_TXT, F_EN), ('白鳥の湖 ／ チャイコフスキー', 0.056, M_TXT, F_JP)], BOTY, 0.3, 7.6)
cap([('Odette, the white swan', 0.04, M_TXT_B, F_SER), ('白鳥オデットが歌う', 0.056, M_TXT_B, F_JP)], TOPY, 3.9, 9.0)
cap([('Marble machine · Satisfying ASMR', 0.034, M_TXT, F_EN), ('ビー玉のからくり・癒しのASMR', 0.053, M_TXT_G, F_JP)], BOTY, 8.2, 13.6)
cap([('Odile, the black swan,', 0.04, M_TXT_V, F_SER), ('copies her song!', 0.04, M_TXT_V, F_SER), ('黒鳥オディールがそっくりまねる', 0.056, M_TXT_V, F_JP)], TOPY, 14.4, 20.0)
cap([('Ballet music (1876) · Act II', 0.032, M_TXT, F_EN), ('バレエ音楽「白鳥の湖」第2幕より 情景', 0.049, M_TXT_G, F_JP)], BOTY, 19.0, 26.5)
cap([('Near miss!', 0.05, M_TXT_G, F_SER), ('すれ違い！', 0.064, M_TXT_G, F_JP)], TOPY, 48.6, 50.6, fade=0.12)
cap([('Now they sing together', 0.04, M_TXT, F_SER), ('ふたりで一緒に', 0.056, M_TXT_G, F_JP)], TOPY, 51.0, 55.5)
cap([('White swan or black swan?', 0.052, M_TXT, F_SER), ('Tell us in the comments', 0.032, M_TXT_G, F_EN), ('どっちが好き？コメントで教えて', 0.053, M_TXT, F_JP)], TOPY, FINAL + 1.3, 99)
for o in bpy.data.objects:
    p_ = o.parent
    while p_ is not None:
        if p_ is hud: o.visible_shadow = False; break
        p_ = p_.parent

if MODE == 'test':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS)); sc.frame_set(f)
        sc.render.filepath = f'{HERE}/test_{f:04d}.png'
        t1 = time.time(); bpy.ops.render.render(write_still=True); print('frame', f, 'sec', round(time.time() - t1, 1))
elif MODE == 'range':
    sc.frame_start, sc.frame_end = int(args[1]), int(args[2])
    sc.render.image_settings.file_format = 'PNG'; sc.render.filepath = f'{HERE}/frames/f_'
    bpy.ops.render.render(animation=True)
