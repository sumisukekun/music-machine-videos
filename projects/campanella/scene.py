"""La Campanella - the bell tower at night (鐘の塔). Every note = a ball striking one bell of that pitch.
layout.py plans all motion (constant gravity, bounces that never add speed, lift + launcher as the only energy sources);
this file builds and animates. python3 scene.py -- check | test t1 t2 ... | range f0 f1"""
import os, sys, json, math, time
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from motion import L as LAY, PATHS, G, R, lift_z, lift_phi, bell_r
NOTES = json.load(open(f'{HERE}/notes.json')); E = json.load(open(f'{HERE}/events.json'))
FPS = 30; NF = int(E['dur'] * FPS); T = np.arange(NF) / FPS
ZD = LAY['deck']; TT = LAY['t']; GBd = LAY['giant']; PHF = LAY['phf']
_phi_t = np.array(LAY['phi']['t']); _phi_v = np.array(LAY['phi']['v'])
def PHI(t): return float(np.interp(t, _phi_t, _phi_v))
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
sc.cycles.max_bounces = 4; sc.cycles.diffuse_bounces = 1; sc.cycles.glossy_bounces = 2; sc.cycles.transmission_bounces = 3
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


def glowmat(name, gain=8.0, rough=0.3, metal=0.0, coat=0.3):
    """object colour = base colour; object alpha drives emission (hit flash)"""
    m = mat(name, (1, 1, 1), rough, **{'Coat Weight': coat, 'Metallic': metal}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    oi = nt.nodes.new('ShaderNodeObjectInfo')
    nt.links.new(oi.outputs['Color'], b.inputs['Base Color']); nt.links.new(oi.outputs['Color'], b.inputs['Emission Color'])
    mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = gain
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
    elif kind == 'tor':
        me = bpy.data.meshes.new('tor'); bm.free()
        bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=a[0], major_segments=a[1], minor_segments=6)
        o = bpy.context.object; me = o.data; bpy.data.objects.remove(o)
        if m: me.materials.append(m)
        for p in me.polygons: p.use_smooth = True
        _MC[key] = me; return me
    elif kind == 'bell': bell_profile(bm, *a)
    me = bpy.data.meshes.new(kind); bm.to_mesh(me); bm.free()
    if kind in ('sph', 'cone', 'bell', 'cyl'):
        for p in me.polygons: p.use_smooth = True
    if m: me.materials.append(m)
    _MC[key] = me; return me


def bell_profile(bm, seg=20):
    """unit bell: crown top at z=0, mouth at z=-1.5 (radius 1), shoulder radius 0.45; lathe"""
    prof = [(0.0, 0.0), (0.30, -0.02), (0.45, -0.10), (0.50, -0.30), (0.55, -0.70), (0.68, -1.10), (0.88, -1.38), (1.0, -1.5), (0.93, -1.52), (0.80, -1.45)]
    rings = []
    for (r, z) in prof:
        ring = []
        for k in range(seg):
            a = 2 * math.pi * k / seg; ring.append(bm.verts.new((r * math.cos(a), r * math.sin(a), z)))
        rings.append(ring)
    for i in range(len(rings) - 1):
        for k in range(seg):
            bm.faces.new((rings[i][k], rings[i][(k + 1) % seg], rings[i + 1][(k + 1) % seg], rings[i + 1][k]))
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)


def obj(me, name='o'): return link(bpy.data.objects.new(name, me))


def box(c, size, m, rot=None, name='box'):
    o = obj(prim('cube', m), name); o.location = c; o.scale = size
    if rot is not None: o.rotation_euler = rot
    return o


def rod(p0, p1, r, m, verts=8):
    p0, p1 = Vector(p0), Vector(p1); d = p1 - p0
    o = obj(prim('cyl', m, verts), 'rod'); q = Vector((0, 0, 1)).rotation_difference(d.normalized() if d.length > 1e-9 else Vector((0, 0, 1)))
    o.matrix_world = Matrix.Translation((p0 + p1) / 2) @ q.to_matrix().to_4x4() @ Matrix.Diagonal((r, r, max(1e-4, d.length), 1))
    return o


def sphere(c, r, m, seg=(16, 8), name='sph'):
    o = obj(prim('sph', m, *seg), name); o.location = c; o.scale = (r, r, r); return o


def tube_curve(pts, r, m, name='rail'):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'; cu.bevel_depth = r; cu.bevel_resolution = 2; cu.use_fill_caps = True
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for i, p in enumerate(pts): sp.points[i].co = (*p, 1)
    o = link(bpy.data.objects.new(name, cu)); o.data.materials.append(m); return o


def text(s, loc, size, m, rot=(0, 0, 0), align='CENTER', ext=0.0):
    cu = bpy.data.curves.new('t', 'FONT'); cu.body = s; cu.font = FONT; cu.size = size; cu.extrude = ext
    cu.align_x = align; cu.align_y = 'CENTER'
    o = link(bpy.data.objects.new('txt', cu)); o.location = loc; o.rotation_euler = rot; o.data.materials.append(m); return o


def frame_from_n(n):
    n = np.array(n, float); n /= np.linalg.norm(n)
    h = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    xa = np.cross(h, n); xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return Matrix(((xa[0], ya[0], n[0]), (xa[1], ya[1], n[1]), (xa[2], ya[2], n[2])))


def hsv(h, s, v):
    import colorsys; return colorsys.hsv_to_rgb(h % 1.0, s, v)


# ---------------- materials ----------------
def bellmat(name, gain=7.0):
    """polished bronze; object colour = flash colour (pitch), object alpha = flash strength"""
    m = mat(name, (1.0, 0.64, 0.25), 0.17, **{'Metallic': 1.0}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    oi = nt.nodes.new('ShaderNodeObjectInfo'); nt.links.new(oi.outputs['Color'], b.inputs['Emission Color'])
    mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = gain
    nt.links.new(oi.outputs['Alpha'], mul.inputs[0]); nt.links.new(mul.outputs[0], b.inputs['Emission Strength'])
    return m
M_BELL = bellmat('bell', 7.0)
M_BRASS = mat('brass', (0.78, 0.55, 0.25), 0.3, **{'Metallic': 0.95})
M_DBRASS = mat('dbrass', (0.35, 0.24, 0.12), 0.4, **{'Metallic': 0.9})
M_STONE = mat('stone', (0.32, 0.30, 0.36), 0.8)
M_DSTONE = mat('dstone', (0.12, 0.12, 0.16), 0.85)
M_GLASS = bpy.data.materials.new('glass'); M_GLASS.use_nodes = True
_b = M_GLASS.node_tree.nodes['Principled BSDF']; _b.inputs['Base Color'].default_value = (0.75, 0.9, 1.0, 1); _b.inputs['Roughness'].default_value = 0.05
_b.inputs['Transmission Weight'].default_value = 1.0; _b.inputs['IOR'].default_value = 1.2; _b.inputs['Alpha'].default_value = 0.35
M_HERO = mat('hero', (0.95, 0.96, 1.0), 0.04, **{'Metallic': 1.0, 'Emission Color': (0.5, 0.75, 1.0, 1), 'Emission Strength': 0.35})
M_COPPER = mat('copper', (0.9, 0.42, 0.22), 0.12, **{'Metallic': 1.0})
M_SILVER = mat('silver', (0.78, 0.82, 0.9), 0.1, **{'Metallic': 1.0})
M_GOLD = mat('goldball', (1.0, 0.72, 0.25), 0.12, **{'Metallic': 1.0})
M_PEARL = mat('pearl', (0.95, 0.93, 0.9), 0.15, **{'Coat Weight': 1.0})
M_FX = fxmat('fx', 8.0)
M_LAMP = emis('lamp', (1.0, 0.7, 0.35), 6.0)
M_WIN = emis('win', (1.0, 0.62, 0.3), 3.0)
M_TXT_W = emis('txtw', (1, 1, 1), 1.4); M_TXT_Y = emis('txty', (1, 0.82, 0.3), 1.8)
M_OBJ = objcol('objcol', 0.5)
M_WIRE = glowmat('wire', gain=10.0, rough=0.4, metal=0.5)

# ---------------- balls ----------------
POS = {n: np.array([p.at(t) for t in T]) for n, p in PATHS.items()}
BALLM = {'hero': M_HERO, 'twin': M_COPPER, 'rh': M_SILVER, 'lh': M_GOLD, 'drop': M_PEARL}
BALL = {}
for n, b in LAY['balls'].items():
    bpy.ops.mesh.primitive_uv_sphere_add(radius=R, segments=32, ring_count=16); o = bpy.context.object; o.name = n
    for p in o.data.polygons: p.use_smooth = True
    o.data.materials.append(BALLM[b['kind']])
    key_loc(o, POS[n])
    sp = np.zeros(NF)
    for f in range(1, NF): sp[f] = sp[f - 1] + np.linalg.norm(POS[n][f, :2] - POS[n][f - 1, :2]) / R
    o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 0, sp * 0.7); key_series(o, 'rotation_euler', 1, sp * 0.7)
    BALL[n] = o
# hero glow light
hl = bpy.data.lights.new('herolight', 'POINT'); hl.energy = 6.0; hl.color = (0.6, 0.8, 1.0); hl.shadow_soft_size = 0.2
hlo = link(bpy.data.objects.new('herolight', hl)); hlo.visible_camera = False; hlo.parent = BALL['S']

# ---------------- bells: one per hit ----------------
PCOL = [hsv(0.0 + 0.8 * k / 12, 0.75, 1.0) for k in range(12)]     # pitch-class tint on the gold
BELLS = []
for n, b in LAY['balls'].items():
    for h in b['hits']:
        if h['kind'] == 'giant': continue
        i = h['i']; p = NOTES[i][1]; rb = bell_r(p)
        nr = np.array(h['n'], float); nr /= np.linalg.norm(nr)
        ax = nr if h['kind'] == 'bell' else np.array([0, 0, 1.0])
        c = np.array(h['p']) - R * nr
        o = obj(prim('bell', M_BELL, 20), 'bell')
        o.matrix_world = Matrix.Translation(Vector(c)) @ frame_from_n(ax).to_4x4() @ Matrix.Diagonal((rb, rb, rb, 1))
        base = PCOL[p % 12]
        o.color = (*base, 0.0)
        BELLS.append(dict(o=o, t=h['t'], c=c, ax=ax, rb=rb, ball=n, p=p, kind=b['kind']))
# flash on the hit + a small sway; all bells light up at the final chord (wave from the bottom up)
T_FIN = TT['hit']
for B in BELLS:
    o = B['o']; fh = int(round(B['t'] * FPS)); col = tuple(o.color[:3])
    o.color = (*col, 0.0); o.keyframe_insert('color', index=3, frame=max(0, fh - 1))
    for df, a in ((0, 1.0), (3, 0.6), (10, 0.25), (24, 0.06)):
        o.color = (*col, a); o.keyframe_insert('color', index=3, frame=fh + df)
    fw = int(round((T_FIN + 0.05 + B['c'][2] / 45.0) * FPS))
    if fw > fh + 24:
        o.color = (*col, 0.06); o.keyframe_insert('color', index=3, frame=fw - 1)
    o.color = (*col, 0.9); o.keyframe_insert('color', index=3, frame=max(fw, fh + 25))
    o.color = (*col, 0.45); o.keyframe_insert('color', index=3, frame=max(fw, fh + 25) + 20)
    base = o.matrix_world.copy(); o.rotation_mode = 'QUATERNION'
    q0 = base.to_quaternion(); loc0 = base.to_translation()
    for df, d in ((-1, 0), (1, -0.02), (4, 0.008), (9, 0)):
        o.location = loc0 + Vector(-B['ax'] * d * 2.0 * B['rb']); o.keyframe_insert('location', frame=max(0, fh + df))
# hit rings (FX pool)
NRING = 18
ring_loc = np.zeros((NRING, NF, 3)); ring_scl = np.full((NRING, NF), 1e-3); ring_a = np.zeros((NRING, NF)); ring_rot = np.zeros((NRING, NF, 3))
ring_c = [(1, 1, 1)] * NRING
for k, B in enumerate(sorted(BELLS, key=lambda b: b['t'])):
    r = k % NRING; f0 = int(round(B['t'] * FPS))
    for df in range(12):
        f = f0 + df
        if f >= NF: break
        u = df / 11; ring_loc[r, f] = B['c'] - B['ax'] * 0.7 * B['rb']; ring_scl[r, f] = B['rb'] * (1.2 + 3.0 * (1 - (1 - u) ** 2))
        ring_a[r, f] = (1 - u) ** 1.5
    ring_c[r] = PCOL[B['p'] % 12]
for r in range(NRING):
    o = obj(prim('tor', M_FX, 0.05, 32), 'ring'); o.color = (*ring_c[r], 0)
    for f in range(1, NF):
        if ring_a[r, f] == 0: ring_loc[r, f] = ring_loc[r, f - 1]
    key_loc(o, ring_loc[r])
    for k in range(3): key_series(o, 'scale', k, ring_scl[r])
    key_series(o, 'color', 3, ring_a[r], 1e-3)

# ---------------- lane rails: every bell stands on a short stem on its ball's spiral rail ----------------
RAILM = {'hero': M_BRASS, 'twin': M_BRASS, 'rh': M_BRASS, 'lh': M_DBRASS}
for n, b in LAY['balls'].items():
    bs = [B for B in BELLS if B['ball'] == n]
    if not bs or b['kind'] == 'drop': continue
    pts = []
    for B in sorted(bs, key=lambda x: x['t']):
        mouth = B['c'] - B['ax'] * 1.5 * B['rb']
        foot = np.array([mouth[0], mouth[1], mouth[2] - 0.22])
        rod(B['c'] - B['ax'] * 1.0 * B['rb'], foot, 0.018, M_DBRASS)
        pts.append(foot)
    if n in ('S', 'K'):
        for q in pts: rod(q, (q[0], q[1], 0.0), 0.02, M_DBRASS)           # intro bells stand on posts on the plaza
    if len(pts) >= 2:
        P = np.array(pts); sm = [P[0]]
        for a_, b_ in zip(P, P[1:]):
            for u in (0.33, 0.67): sm.append(a_ + (b_ - a_) * u)
            sm.append(b_)
        tube_curve(sm, 0.03, RAILM[b['kind']])
        # every ~4th foot: a strut to the tower core so the rails are carried
        for j, q in enumerate(pts[::4]):
            rr = math.hypot(q[0], q[1])
            if rr > 1.0 and n not in ('S', 'K'):
                rod(q, (q[0] * 0.9 / rr, q[1] * 0.9 / rr, q[2] - 0.15), 0.014, M_DBRASS)

# ---------------- gates (release flaps), wires from the lift's trip contacts, cups, baskets ----------------
for tr in LAY['trips']:
    b = LAY['balls'][tr['ball']]; g = b['gate']; gp = np.array(g['p']); tg = np.array(g['tang'])
    shelf = box(gp - np.array([0, 0, R + 0.03]) - tg * 0.12, (0.42, 0.42, 0.05), M_BRASS, name='gate')
    shelf.rotation_euler = (0, 0, math.atan2(tg[1], tg[0]))
    flap = box(gp + tg * (R + 0.04), (0.04, 0.36, 0.36), M_DBRASS, name='flap'); flap.rotation_euler = (0, 0, math.atan2(tg[1], tg[0]))
    f0 = int(round(g['t'] * FPS)) - 2
    flap.keyframe_insert('location', frame=f0); flap.location = Vector(gp + tg * (R + 0.04) + np.array([0, 0, 0.42])); flap.keyframe_insert('location', frame=f0 + 3)
    # signal wire: from the trip contact on the lift guide (at the lift's height when it trips) to the gate
    rr = math.hypot(gp[0], gp[1]); a = math.atan2(gp[1], gp[0])
    p0 = np.array([0.92 * math.cos(a), 0.92 * math.sin(a), tr['z_lever']])
    w = rod(p0, gp - np.array([0, 0, R + 0.05]), 0.012, M_WIRE); w.color = (0.4, 0.8, 1.0, 0.0)
    w.keyframe_insert('color', index=3, frame=f0 - 1); w.color = (0.4, 0.8, 1.0, 1.0); w.keyframe_insert('color', index=3, frame=f0 + 1)
    w.color = (0.4, 0.8, 1.0, 0.0); w.keyframe_insert('color', index=3, frame=f0 + 14)
    lamp = sphere(p0, 0.06, M_WIRE); lamp.color = (0.4, 0.8, 1.0, 0.15)
    lamp.keyframe_insert('color', index=3, frame=f0 - 1); lamp.color = (0.4, 0.8, 1.0, 1.0); lamp.keyframe_insert('color', index=3, frame=f0 + 1)
    lamp.color = (0.4, 0.8, 1.0, 0.3); lamp.keyframe_insert('color', index=3, frame=f0 + 20)
for n, b in LAY['balls'].items():
    for c in b['cups']:
        cp = np.array(c['p']); v = np.array(c['v']); tg = v / np.linalg.norm(v)
        cup = obj(prim('cone', M_BRASS, 16, 0.12, 0.24, 0.16), 'cup'); cup.location = cp - np.array([0, 0, R - 0.02])
        cup.rotation_mode = 'XYZ'; f0 = int(round(c['t_out'] * FPS))
        cup.keyframe_insert('rotation_euler', frame=f0 - 2)
        ax_ = np.cross([0, 0, 1], tg); cup.rotation_euler = (ax_[0] * 0.6, ax_[1] * 0.6, 0); cup.keyframe_insert('rotation_euler', frame=f0 + 2)
        cup.rotation_euler = (0, 0, 0); cup.keyframe_insert('rotation_euler', frame=f0 + 14)
    if b['basket']:
        bp = np.array(b['basket']['p'])
        bk = obj(prim('cone', M_DBRASS, 16, 0.16, 0.26, 0.2), 'basket'); bk.location = bp - np.array([0, 0, R - 0.04])

# ---------------- tower: plaza, core shaft, lift, deck, belfry, giant bell, roof ----------------
ZTOP = ZD + 3.9
base = obj(prim('cyl', M_STONE, 64), 'plaza'); base.location = (0, 0, -0.15); base.scale = (14, 14, 0.3)
for k in range(48):                                   # plaza ring of lanterns
    a = 2 * math.pi * k / 48; p = (9.5 * math.cos(a), 9.5 * math.sin(a), 0)
    rod(p, (p[0], p[1], 1.1), 0.03, M_DBRASS); sphere((p[0], p[1], 1.2), 0.09, M_LAMP)
for z in np.arange(0.0, ZD + 0.01, 1.6):
    o = obj(prim('tor', M_BRASS, 0.035, 48), 'band'); o.location = (0, 0, z); o.scale = (0.92, 0.92, 0.92)
for k in range(4):
    a = 2 * math.pi * k / 4 + 0.4
    rod((0.84 * math.cos(a), 0.84 * math.sin(a), 0), (0.84 * math.cos(a), 0.84 * math.sin(a), ZD), 0.025, M_BRASS)
# lift platform (turns with the spiral) + S's gate bell standing on it is part of BELLS (cradle) -> carried
plat = obj(prim('cyl', M_BRASS, 40), 'platform'); plat.scale = (0.78, 0.78, 0.08)
pivot = link(bpy.data.objects.new('liftpivot', None)); plat.parent = pivot; plat.location = (0, 0, -0.04)
lz = np.array([lift_z(t) for t in T]); lp = np.array([lift_phi(t) for t in T])
key_series(pivot, 'location', 2, lz); key_series(pivot, 'rotation_euler', 2, lp)
lamp_l = sphere((0, 0, 0.12), 0.07, M_LAMP); lamp_l.parent = pivot; lamp_l.location = (-0.5, 0.0, 0.1)
lift_light = bpy.data.lights.new('liftlight', 'POINT'); lift_light.energy = 25; lift_light.color = (1.0, 0.75, 0.45)
llo = link(bpy.data.objects.new('liftlight', lift_light)); llo.visible_camera = False; llo.parent = pivot; llo.location = (0, 0, 1.0)
# S's gate bell rides the platform: re-parent the cradle bell of S to the pivot
z0 = LAY['lift']['z0']
for B in BELLS:
    if B['ball'] == 'S' and abs(B['t'] - TT['ferm']) < 1e-3:
        o = B['o']; mw = o.matrix_world.copy()
        o.animation_data_clear(); o.color = (*o.color[:3], 0.0)
        fh = int(round(B['t'] * FPS))
        for df, a in ((-1, 0.0), (0, 1.0), (3, 0.6), (10, 0.25), (24, 0.06)):
            o.color = (*o.color[:3], a); o.keyframe_insert('color', index=3, frame=max(0, fh + df))
        fw = int(round((T_FIN + 0.05 + ZD / 45.0) * FPS))
        o.color = (*o.color[:3], 0.06); o.keyframe_insert('color', index=3, frame=fw - 1)
        o.color = (*o.color[:3], 0.9); o.keyframe_insert('color', index=3, frame=fw)
        o.parent = pivot; o.matrix_parent_inverse = Matrix.Identity(4)
        o.matrix_basis = Matrix.Translation(Vector((0, 0, -z0))) @ mw
        # at the top the bell tips forward: its cradle lets the hero roll out onto the chute
        tilt_f = int(round(TT['tip'] * FPS))
        o.rotation_mode = 'XYZ'
# deck (belfry floor): ring slab, gutter rim
deck = obj(prim('cyl', M_STONE, 72), 'deck'); deck.location = (0, 0, ZD + 0.045); deck.scale = (4.95, 4.95, 0.15)
rim = obj(prim('tor', M_BRASS, 0.05, 72), 'rim'); rim.location = (0, 0, ZD + 0.2); rim.scale = (4.9, 4.9, 1.6)
for k in range(10):                                   # belfry columns
    a = 2 * math.pi * k / 10 + 0.15
    rod((5.15 * math.cos(a), 5.15 * math.sin(a), ZD - 0.5), (5.15 * math.cos(a), 5.15 * math.sin(a), ZTOP), 0.12, M_STONE)
roof = obj(prim('cone', M_DSTONE, 48, 5.8, 0.0, 5.0), 'roof'); roof.location = (0, 0, ZTOP + 2.5)
sph_top = sphere((0, 0, ZTOP + 5.2), 0.25, M_BRASS)
# giant bell (hangs from a beam), struck by the hero on its sound bow
gh = GBd['c'][2] - GBd['z_mouth']; grb = GBd['r_mouth']
gb = obj(prim('bell', M_BELL, 48), 'giantbell')
gb.matrix_world = Matrix.Translation(Vector((0, 0, GBd['c'][2]))) @ Matrix.Diagonal((grb, grb, gh / 1.5, 1))
gb.color = (1.0, 0.78, 0.35, 0.0)
fh = int(round(T_FIN * FPS))
gb.keyframe_insert('color', index=3, frame=fh - 1)
for df, a in ((0, 1.6), (6, 1.0), (30, 0.6), (90, 0.45)):
    gb.color = (1.0, 0.78, 0.35, a); gb.keyframe_insert('color', index=3, frame=fh + df)
gb.rotation_mode = 'XYZ'
for df, a in ((-1, 0.0), (4, 0.05), (14, -0.035), (26, 0.022), (40, -0.012), (60, 0.0)):
    gb.rotation_euler = (a * -math.sin(PHF + 0.42), a * math.cos(PHF + 0.42), 0); gb.keyframe_insert('rotation_euler', frame=fh + df)
rod((0, 0, GBd['c'][2]), (0, 0, ZTOP), 0.07, M_DBRASS)
rod((-4.9, 0, ZTOP - 0.1), (4.9, 0, ZTOP - 0.1), 0.1, M_DBRASS); rod((0, -4.9, ZTOP - 0.1), (0, 4.9, ZTOP - 0.1), 0.1, M_DBRASS)
# chute + launcher + goal cup
ch = LAY['chute']; d = np.array([math.cos(ch['phi']), math.sin(ch['phi']), 0])
zc0 = ZD + LAY['p_s0'][2] - z0 - R - 0.03
rod(np.array([d[0] * 0.55, d[1] * 0.55, zc0]), np.array([d[0] * 2.45, d[1] * 2.45, zc0 - ch['drop']]), 0.05, M_BRASS)
for s_ in (-1, 1):
    o_ = np.array([-d[1], d[0], 0]) * 0.13 * s_
    rod(np.array([d[0] * 0.55, d[1] * 0.55, zc0 + 0.06]) + o_, np.array([d[0] * 2.45, d[1] * 2.45, zc0 - ch['drop'] + 0.06]) + o_, 0.025, M_BRASS)
PL = np.array(LAY['launcher'])
lcup = obj(prim('cone', M_DBRASS, 20, 0.2, 0.25, 0.14), 'launcher'); lcup.location = PL - np.array([0, 0, R + 0.02])
plunger = obj(prim('cyl', M_BRASS, 16), 'plunger'); plunger.scale = (0.1, 0.1, 0.5)
ff = int(round(TT['fire'] * FPS)); pz = PL[2] - R - 0.35
plunger.location = (PL[0], PL[1], pz); plunger.keyframe_insert('location', frame=ff - 1)
plunger.location = (PL[0], PL[1], pz + 0.2); plunger.keyframe_insert('location', frame=ff + 1); plunger.keyframe_insert('location', frame=ff + 12)
plunger.location = (PL[0], PL[1], pz); plunger.keyframe_insert('location', frame=ff + 30)
spring = obj(prim('tor', M_DBRASS, 0.12, 24), 'spring'); spring.location = (PL[0], PL[1], pz - 0.1); spring.scale = (0.16, 0.16, 0.16)
GO = np.array(LAY['goal'])
gcup = obj(prim('cone', M_BRASS, 20, 0.14, 0.26, 0.12), 'goal'); gcup.location = GO - np.array([0, 0, R - 0.02])
# spokes: trapdoor hatches under the belfry beams, clock hand that opens them one after another
for sp in LAY['spokes']:
    for bn in sp['balls']:
        g = LAY['balls'][bn]['gate']; gp = np.array(g['p'])
        sh = box(gp - np.array([0, 0, R + 0.03]), (0.36, 0.36, 0.04), M_BRASS, name='hatch')
        f0 = int(round(sp['t_open'] * FPS))
        sh.rotation_mode = 'XYZ'; sh.keyframe_insert('rotation_euler', frame=f0 - 1)
        tg = np.array(g['tang']); sh.rotation_euler = (tg[0] * 1.4, tg[1] * 1.4, 0); sh.keyframe_insert('rotation_euler', frame=f0 + 2)
        rod(gp + np.array([0, 0, -R - 0.02]), (gp[0], gp[1], ZTOP - 0.1), 0.012, M_DBRASS)
HZ = ZD + 3.15
hand = link(bpy.data.objects.new('hand', None)); hand.location = (0, 0, HZ)
arm = rod((0, 0, 0), (4.6, 0, 0), 0.06, M_BRASS); arm.parent = hand
tip = sphere((4.6, 0, 0), 0.14, M_LAMP); tip.parent = hand
ha = np.zeros(NF); keys = [(TT['top'], LAY['spokes'][0]['ang'] - 0.5)] + [(sp['t_open'], sp['ang']) for sp in LAY['spokes']]
kt = np.array([k[0] for k in keys]); ka = np.array([k[1] for k in keys])
for f in range(NF): ha[f] = np.interp(T[f], kt, ka)
key_series(hand, 'rotation_euler', 2, ha)

# ---------------- town, sky, moon, stars, fireworks ----------------
for k in range(70):
    a = rng.uniform(0, 2 * math.pi); r_ = rng.uniform(34, 70); w = rng.uniform(1.6, 3.4); h = rng.uniform(2.0, 7.0)
    if abs((a - PHF + math.pi) % (2 * math.pi) - math.pi) < 0.35: continue          # keep the final pull-back clear
    hb = box((r_ * math.cos(a), r_ * math.sin(a), h / 2), (w, w, h), M_DSTONE, name='house'); hb.rotation_euler = (0, 0, a)
    for j in range(int(h // 1.2)):
        if rng.random() < 0.55:
            box(((r_ - w / 2 - 0.01) * math.cos(a), (r_ - w / 2 - 0.01) * math.sin(a), 0.8 + j * 1.2), (0.05, 0.35, 0.45), M_WIN, rot=(0, 0, a))
gnd = obj(prim('cyl', M_DSTONE, 64), 'ground'); gnd.location = (0, 0, -0.32); gnd.scale = (120, 120, 0.1)
w_ = bpy.data.worlds.new('W'); sc.world = w_; w_.use_nodes = True
nt = w_.node_tree; bgn = nt.nodes['Background']
tc = nt.nodes.new('ShaderNodeTexCoord'); sep = nt.nodes.new('ShaderNodeSeparateXYZ'); rp = nt.nodes.new('ShaderNodeValToRGB')
rp.color_ramp.elements[0].position = 0.0; rp.color_ramp.elements[0].color = (0.10, 0.05, 0.16, 1)
rp.color_ramp.elements[1].position = 0.55; rp.color_ramp.elements[1].color = (0.005, 0.01, 0.045, 1)
mp = nt.nodes.new('ShaderNodeMapRange'); mp.inputs['From Min'].default_value = -0.05; mp.inputs['From Max'].default_value = 0.9
nt.links.new(tc.outputs['Generated'], sep.inputs[0]); nt.links.new(sep.outputs['Z'], mp.inputs['Value']); nt.links.new(mp.outputs[0], rp.inputs['Fac'])
nt.links.new(rp.outputs['Color'], bgn.inputs['Color']); bgn.inputs['Strength'].default_value = 1.0
moon_l = bpy.data.lights.new('moon', 'SUN'); moon_l.energy = 0.6; moon_l.color = (0.65, 0.75, 1.0); moon_l.angle = 0.02
mo = link(bpy.data.objects.new('moon', moon_l)); mo.rotation_euler = (math.radians(55), 0, math.radians(30))
moon = sphere((-70, 60, 55), 3.2, emis('moonm', (0.95, 0.95, 1.0), 3.0), seg=(32, 16))
M_STAR = emis('star', (1, 1, 1), 6.0)
for k in range(700):
    v = rng.normal(0, 1, 3); v[2] = abs(v[2]) + 0.12; v /= np.linalg.norm(v)
    sphere(v * 150, rng.uniform(0.08, 0.25), M_STAR, seg=(6, 3))
# floor lanterns on the core (warm points that light the bells nearby)
for z in np.arange(1.5, ZD, 3.2):
    for k in range(3):
        a = 2 * math.pi * k / 3 + z
        sphere((0.98 * math.cos(a), 0.98 * math.sin(a), z), 0.07, M_LAMP)
    pl = bpy.data.lights.new('lan', 'POINT'); pl.energy = 30; pl.color = (1.0, 0.7, 0.4); pl.shadow_soft_size = 0.5
    po = link(bpy.data.objects.new('lan', pl)); po.location = (0, 0, z); po.visible_camera = False; po.visible_glossy = False; po.visible_transmission = False; pl.shadow_soft_size = 0.1
FW = LAY['fireworks']
NP = 70
for k, tf_ in enumerate(FW):
    a = PHF + rng.uniform(-0.9, 0.9) + math.pi * 0.0; dist = rng.uniform(8, 16)
    ctr = np.array([dist * math.cos(a), dist * math.sin(a), ZD + rng.uniform(6, 14)])
    hue = rng.uniform(0, 1)
    dirs = rng.normal(0, 1, (NP, 3)); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    for j in range(NP):
        o = obj(prim('sph', M_FX, 6, 3), 'spark'); col = hsv(hue + rng.uniform(-0.06, 0.06), 0.7, 1.0); o.color = (*col, 0)
        P = np.zeros((NF, 3)); S_ = np.full(NF, 1e-3); A = np.zeros(NF)
        sp_ = rng.uniform(3.5, 5.0)
        for f in range(NF):
            u = T[f] - tf_
            if u < 0: P[f] = ctr; continue
            P[f] = ctr + dirs[j] * sp_ * (1 - math.exp(-u * 1.6)) / 1.6 + np.array([0, 0, -0.6 * u * u]); S_[f] = 0.09; A[f] = max(0.0, 1.0 - u / 1.6) ** 1.3
        key_loc(o, P)
        for c_ in range(3): key_series(o, 'scale', c_, S_)
        key_series(o, 'color', 3, A, 1e-3)

# ---------------- camera: rises with the music, orbits with the spiral ----------------
def smooth(Xa, k):
    Y = Xa.copy(); a = 1 - math.exp(-1 / FPS / k)
    for f in range(1, len(Xa)): Y[f] = Y[f - 1] + (Xa[f] - Y[f - 1]) * a
    for f in range(len(Xa) - 2, -1, -1): Y[f] = Y[f + 1] + (Y[f] - Y[f + 1]) * a
    return Y
ZF = np.zeros(NF); WID = np.zeros(NF); AZ = np.zeros(NF)
prim_balls = [n for n, b in LAY['balls'].items() if b['kind'] in ('rh', 'lh', 'twin', 'hero')]
spans = {n: (b['hits'][0]['t'] - 0.7, b['hits'][-1]['t'] + 0.25) for n, b in LAY['balls'].items() if b['hits']}
for f in range(NF):
    t = T[f]
    zs = []; ws = []
    for n in prim_balls:
        if n == 'S' and t > TT['ferm'] + 0.5: continue
        t0, t1 = spans[n]
        if t0 <= t <= t1:
            wgt = 1.0 + len(LAY['balls'][n]['hits']) / 40.0
            zs.append(POS[n][f, 2] * wgt); ws.append(wgt)
    zl = lift_z(t) + 0.8
    if zs: z_ = sum(zs) / sum(ws); ZF[f] = 0.72 * z_ + 0.28 * zl
    else: ZF[f] = ZF[f - 1] if f else 1.0
    AZ[f] = PHI(min(t, 56.6)) + 0.02
    WID[f] = 7.6
    if t < 2.6: WID[f] = 7.6 + 6.0 * (1 - t / 2.6) ** 2; ZF[f] = 1.3 + 3.5 * (1 - t / 2.6) ** 2
    if t > 55.0:                                    # finale: the belfry
        u = min(1, (t - 55.0) / 1.6); ZF[f] = ZF[f] * (1 - u) + (ZD + 1.4) * u; WID[f] = 7.6 + 2.2 * u
    if t > T_FIN + 0.35:                           # reveal: pull back and up, the whole tower lights
        u = min(1, (t - T_FIN - 0.35) / 3.2); u = u * u * (3 - 2 * u)
        ZF[f] = (ZD + 1.4) * (1 - u) + (ZD * 0.55 + 2) * u; WID[f] = 9.8 + 22.0 * u
ZF = smooth(ZF, 0.45); WID = smooth(WID, 0.3); AZ = smooth(AZ, 0.25)
LENS = 35.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 600
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24
    a = AZ[f]
    CP[f] = (D * math.cos(a), D * math.sin(a), max(0.45, ZF[f] - 0.07 * D))
    TP[f] = (0, 0, ZF[f] + 0.06 * D)
key_loc(cam, CP); key_loc(tgt, TP)
# warm key light that travels with the camera (lights the bells in front)
kl = bpy.data.lights.new('key', 'AREA'); kl.energy = 900; kl.size = 6; kl.color = (1.0, 0.85, 0.65)
klo = link(bpy.data.objects.new('key', kl))
KP = np.zeros((NF, 3))
for f in range(NF):
    a = AZ[f] + 0.6; KP[f] = (9 * math.cos(a), 9 * math.sin(a), ZF[f] + 5)
key_loc(klo, KP)
ktc = klo.constraints.new('TRACK_TO'); ktc.target = tgt; ktc.track_axis = 'TRACK_NEGATIVE_Z'; ktc.up_axis = 'UP_Y'

# ---------------- HUD text ----------------
hud = link(bpy.data.objects.new('hud', None)); hud.parent = cam; hud.location = (0, 0, -3.0)
HH = 3.0 * 24 / LENS; HWd = HH * 9 / 16
def hud_text(s, y, size, m, t0, t1):
    o = text(s, (0, y, 0), size, m); o.parent = hud; o.visible_shadow = False
    sc_ = np.array([1e-3 if not (t0 <= t < t1) else 1.0 for t in T])
    for c_ in range(3): key_series(o, 'scale', c_, sc_, 1e-4)
    return o
hud_text('この塔の鐘', HH / 2 - 0.28, 0.17, M_TXT_W, 0.0, 4.0)
hud_text('ぜんぶで何個鳴る？', HH / 2 - 0.5, 0.17, M_TXT_Y, 0.0, 4.0)
hud_text('ラ・カンパネラ（リスト）', -HH / 2 + 0.3, 0.1, M_TXT_W, 0.0, 4.0)
hud_text('鐘の滝 96連打', HH / 2 - 0.32, 0.14, M_TXT_Y, 38.8, 42.0)
hud_text('何個鳴ったか数えられた？', HH / 2 - 0.3, 0.13, M_TXT_W, 61.8, 99)
hud_text('答えはコメントで！', HH / 2 - 0.5, 0.12, M_TXT_Y, 62.3, 99)


# ---------------- sanity checks ----------------
def check():
    info = {}
    gaps = []
    for n, b in LAY['balls'].items():
        for h in b['hits']:
            f = int(round(h['t'] * FPS)); gaps.append(np.linalg.norm(POS[n][min(NF - 1, f)] - np.array(h['p'])))
    info['contact_gap_at_hit_frame_max'] = round(max(gaps), 3)
    # ball size on screen: fraction of the frame width at the lane in front (r ~ 3)
    fr = []
    for f in range(0, NF, 15):
        D = WID[f] / (9 / 16) * LENS / 24; wl = WID[f] * (D - 3) / D; fr.append(2 * R / wl)
    info['ball_frac_of_width_min_max'] = (round(min(fr), 3), round(max(fr), 3))
    info['bells'] = len(BELLS) + 1; info['objects'] = len(bpy.data.objects); info['NF'] = NF
    return info


if MODE in ('check', 'test'): print(check())
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
