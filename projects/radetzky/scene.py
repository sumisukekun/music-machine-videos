"""Radetzky March - New Year's concert of marbles (ビー玉のニューイヤーコンサート), vertical cut-away of a gold concert hall.
layout.py plans every motion (constant gravity, bounces that never add speed, rolling with g sin(slope)); this file builds and
animates it. Every melody note = a gold player marble striking a key that slides out of a spiral glockenspiel tower.
Every beat = the conductor landing on his drum podium. Every clap = the audience marbles landing in their seats.
python3 scene.py -- test t1 t2 ... | range f0 f1"""
import os, sys, json, math, time
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix, Quaternion

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json')); NOTES = json.load(open(f'{HERE}/notes.json'))
LAY = json.load(open(f'{HERE}/layout.json'))
FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.5) * FPS); T = np.arange(NF) / FPS
G = LAY['G']; EV = LAY['ev']; GEO = LAY['geo']; RAD = LAY['radii']
TN = np.array([l[0] for l in LEAD]); KB = lambda k: E['t0'] + k * E['bar']
rng = np.random.default_rng(3)

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
sc.cycles.transparent_max_bounces = 8
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


def glowmat(name, gain=8.0, base=0.0, rough=0.3, metal=0.0):
    """object colour = base colour; object alpha drives emission (hit flash)"""
    m = mat(name, (1, 1, 1), rough, **{'Coat Weight': 0.4, 'Metallic': metal}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
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


def glassmat(name, tint=(0.85, 0.95, 1.0), opacity=0.22):
    """cheap clear glass: mostly transparent + a glossy sheen (balls inside stay readable)"""
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    tr_ = nt.nodes.new('ShaderNodeBsdfTransparent'); tr_.inputs['Color'].default_value = (*tint, 1)
    gl = nt.nodes.new('ShaderNodeBsdfGlossy'); gl.inputs['Roughness'].default_value = 0.08
    fr = nt.nodes.new('ShaderNodeLayerWeight'); fr.inputs['Blend'].default_value = 0.35
    mu = nt.nodes.new('ShaderNodeMath'); mu.operation = 'MULTIPLY_ADD'; mu.inputs[1].default_value = 0.8; mu.inputs[2].default_value = opacity
    mx = nt.nodes.new('ShaderNodeMixShader'); out = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(fr.outputs['Facing'], mu.inputs[0]); nt.links.new(mu.outputs[0], mx.inputs['Fac'])
    nt.links.new(tr_.outputs[0], mx.inputs[1]); nt.links.new(gl.outputs[0], mx.inputs[2]); nt.links.new(mx.outputs[0], out.inputs[0])
    return m


def objcol(name, rough=0.45, metal=0.0, coat=0.0):
    m = mat(name, (1, 1, 1), rough, **{'Metallic': metal, 'Coat Weight': coat}); nt = m.node_tree
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


def key_quat(o, Qs):
    o.rotation_mode = 'QUATERNION'
    for k in range(4): key_series(o, 'rotation_quaternion', k, Qs[:, k], 1e-6)


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
        R_, r_, nu, nv = a
        vs = [[bm.verts.new(((R_ + r_ * math.cos(2 * math.pi * j / nv)) * math.cos(2 * math.pi * i / nu),
                             (R_ + r_ * math.cos(2 * math.pi * j / nv)) * math.sin(2 * math.pi * i / nu), r_ * math.sin(2 * math.pi * j / nv)))
               for j in range(nv)] for i in range(nu)]
        for i in range(nu):
            for j in range(nv):
                bm.faces.new((vs[i][j], vs[(i + 1) % nu][j], vs[(i + 1) % nu][(j + 1) % nv], vs[i][(j + 1) % nv]))
    me = bpy.data.meshes.new(kind); bm.to_mesh(me); bm.free()
    if kind in ('sph', 'cone', 'torus', 'cyl'):
        for p in me.polygons: p.use_smooth = kind != 'cyl' or len(p.vertices) == 4
    if m: me.materials.append(m)
    _MC[key] = me; return me


def obj(me, name='o'): return link(bpy.data.objects.new(name, me))


def box(c, size, m, rot=None, name='box'):
    o = obj(prim('cube', m), name); o.location = c; o.scale = size
    if rot is not None: o.rotation_euler = rot
    return o


def rod(p0, p1, r, m, verts=12):
    p0, p1 = Vector(p0), Vector(p1); d = p1 - p0
    o = obj(prim('cyl', m, verts), 'rod'); q = Vector((0, 0, 1)).rotation_difference(d.normalized() if d.length > 1e-9 else Vector((0, 0, 1)))
    o.matrix_world = Matrix.Translation((p0 + p1) / 2) @ q.to_matrix().to_4x4() @ Matrix.Diagonal((r, r, max(1e-4, d.length), 1))
    return o


def sphere(c, r, m, seg=(24, 12), name='sph'):
    o = obj(prim('sph', m, *seg), name); o.location = c; o.scale = (r, r, r); return o


def torus(c, R_, r_, m, rot=(0, 0, 0), name='tor', nu=32, nv=8):
    o = obj(prim('torus', m, R_, r_, nu, nv), name); o.location = c; o.rotation_euler = rot; return o


def text(s, loc, size, m, rot=(math.pi / 2, 0, 0), align='CENTER', ext=0.02):
    cu = bpy.data.curves.new('t', 'FONT'); cu.body = s; cu.font = FONT; cu.size = size; cu.extrude = ext
    cu.align_x = align; cu.align_y = 'CENTER'
    o = link(bpy.data.objects.new('txt', cu)); o.location = loc; o.rotation_euler = rot; o.data.materials.append(m); return o


def tube(pts, r, m, name='tube', res=4):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'; cu.bevel_depth = r; cu.bevel_resolution = res; cu.use_fill_caps = False
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for k, p in enumerate(pts): sp.points[k].co = (*p, 1)
    o = link(bpy.data.objects.new(name, cu)); o.data.materials.append(m); return o


def smooth_pts(pts, n=6):
    """corner-rounded polyline (quadratic bezier at each corner)"""
    pts = [np.array(p, float) for p in pts]; out = [pts[0]]
    for a, b, c in zip(pts, pts[1:], pts[2:]):
        la, lc = np.linalg.norm(b - a), np.linalg.norm(c - b); rr = min(0.35, la / 2, lc / 2)
        p0 = b + (a - b) / la * rr; p2 = b + (c - b) / lc * rr
        for k in range(n + 1):
            u = k / n; out.append((1 - u) ** 2 * p0 + 2 * u * (1 - u) * b + u * u * p2)
    out.append(pts[-1]); return out


def pulse_alpha(times, NFr=NF, decay=5.0, length=16, amp=1.0):
    a = np.zeros(NFr)
    for t in times:
        f0 = int(round(t * FPS)); m_ = np.arange(max(0, f0), min(NFr, f0 + length))
        a[m_] = np.maximum(a[m_], amp * np.exp(-(m_ - f0) / decay))
    return a


def smoothstep(u): u = np.clip(u, 0, 1); return u * u * (3 - 2 * u)


# ---------------- ball paths ----------------
def seg_at(seg, t):
    for g in seg:
        if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9:
            if g['k'] == 'rest': return np.array(g['p'])
            if g['k'] == 'fly':
                u = t - g['t0']; return np.array(g['p']) + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
            P = g['P']; x = (t - g['t0']) / g['dt']; i = int(min(len(P) - 2, max(0, math.floor(x)))); f = x - i
            return np.array(P[i]) * (1 - f) + np.array(P[i + 1]) * f
    if t < seg[0]['t0']: return seg_at(seg, seg[0]['t0'])
    return seg_at(seg, seg[-1]['t1'])
BALLS = LAY['balls']
POS = {n: np.array([seg_at(b['seg'], t) for t in T]) for n, b in BALLS.items()}


def roll_quats(P, r):
    """rolling spin from the horizontal motion (axis = z x v)"""
    q = Quaternion((1, 0, 0, 0)); out = [tuple(q)]
    for f in range(1, len(P)):
        d = P[f] - P[f - 1]; dh = np.array([d[0], d[1], 0.0]); L = np.linalg.norm(dh)
        if L > 1e-6:
            ax = Vector(np.cross([0, 0, 1], dh / L)); q = Quaternion(ax, L / r) @ q
        out.append(tuple(q))
    return np.array(out)


# ---------------- materials ----------------
M_GOLD = mat('gold', (1.0, 0.7, 0.28), 0.22, **{'Metallic': 1.0})
M_GOLD_D = mat('golddull', (0.75, 0.5, 0.2), 0.4, **{'Metallic': 1.0})
M_SILVER = mat('silver', (0.92, 0.93, 0.97), 0.12, **{'Metallic': 1.0})
M_KEY = glowmat('key', gain=6.0, base=0.0, rough=0.18, metal=1.0)
M_BLACK = mat('black', (0.015, 0.015, 0.02), 0.12, **{'Coat Weight': 1.0})
M_WHITE = mat('white', (0.95, 0.95, 0.93), 0.35)
M_RED = mat('red', (0.9, 0.04, 0.05), 0.12, **{'Coat Weight': 1.0})
M_VELVET = mat('velvet', (0.42, 0.03, 0.05), 0.8, **{'Sheen Weight': 0.8})
M_CARPET = mat('carpet', (0.35, 0.03, 0.04), 0.9)
M_CREAM = mat('cream', (0.93, 0.82, 0.6), 0.55)
M_WOOD = mat('wood', (0.38, 0.2, 0.08), 0.45, **{'Coat Weight': 0.4})
M_STAGE = mat('stage', (0.5, 0.29, 0.12), 0.35, **{'Coat Weight': 0.5})
M_BRICK = mat('brick', (0.22, 0.12, 0.08), 0.85)
M_DARK = mat('dark', (0.04, 0.035, 0.04), 0.6)
M_STEEL = mat('steel', (0.6, 0.6, 0.64), 0.3, **{'Metallic': 0.9})
M_GLASS = glassmat('glass', (0.9, 0.97, 1.0), 0.1)
M_GLASS_F = glassmat('glassf', (1.0, 0.92, 0.75), 0.3)
M_OBJ = objcol('objcol', 0.12, coat=1.0)
M_OBJM = objcol('objcolm', 0.5)
M_FX = fxmat('fx', 6.0)
M_EYE = mat('eye', (1, 1, 1), 0.3); M_PUPIL = mat('pupil', (0.01, 0.01, 0.01), 0.2)
M_TXT_W = emis('txtw', (1, 1, 1), 2.0); M_TXT_Y = emis('txty', (1, 0.82, 0.3), 2.5); M_TXT_R = emis('txtr', (1, 0.25, 0.2), 3.0)
M_LAMP = emis('lamp', (1.0, 0.8, 0.5), 6.0)
M_BAND = emis('band', (0.02, 0.01, 0.015), 1.0)
M_WIN = emis('win', (1.0, 0.75, 0.45), 1.2)

# ==================================================================== hall
XC = GEO['XC']; Y_BACK = 8.15; Z_CEIL = 14.4; X_L = -5.4; X_R = 5.4
base_floor = EV['stairs']['floor'] - RAD['late']          # basement floor at the bottom of the stairs
# stage
box((XC - 0.4, -0.15, -0.09), (X_R - X_L + 0.2, 3.7, 0.18), M_STAGE, name='stage')
box((XC - 0.4, -2.02, -0.09), (X_R - X_L + 0.2, 0.06, 0.2), M_GOLD, name='stagelip')
# back wall with pilasters and glowing arched windows
box((XC - 0.4, Y_BACK + 0.1, 7.0), (X_R - X_L + 0.4, 0.2, 15.0), M_CREAM, name='backwall')
for x in np.arange(X_L + 0.5, X_R, 1.55):
    box((x, Y_BACK - 0.02, 9.6), (0.22, 0.14, 9.6), M_GOLD_D, name='pilaster')
for x in np.arange(X_L + 1.27, X_R - 0.5, 1.55):
    box((x, Y_BACK - 0.01, 13.45), (0.8, 0.05, 1.1), M_WIN, name='win')
    sphere((x, Y_BACK - 0.01, 14.0), 0.4, M_WIN, (16, 8)).scale = (0.4, 0.05, 0.4)
# side walls (gold boxes)
for xw in (X_L, X_R):
    box((xw, 3.0, 7.0), (0.2, 10.4, 15.0), M_CREAM, name='side')
    for z in (5.0, 8.0, 11.0):
        box((xw + (0.12 if xw < 0 else -0.12), 3.0, z), (0.06, 9.8, 0.08), M_GOLD_D)
# ceiling with a coffered gold trim and the conductor's hatch
box((XC - 0.4, 3.0, Z_CEIL + 0.15), (X_R - X_L + 0.4, 10.4, 0.3), M_CREAM, name='ceiling')
box((XC - 0.4, -2.18, Z_CEIL - 0.05), (X_R - X_L + 0.4, 0.12, 0.25), M_GOLD, name='ceilfront')
pod = np.array(GEO['podium'])
hatch_hinge = link(bpy.data.objects.new('hatch', None)); hatch_hinge.location = (pod[0] - 0.55, pod[1], Z_CEIL)
hd = box((0.55, 0, 0), (1.1, 1.1, 0.06), M_GOLD, name='hatchdoor'); hd.parent = hatch_hinge
ha = np.array([min(1.0, max(0.0, (t - 0.0) / 0.25)) ** 0.5 * 1.7 for t in T])
hatch_hinge.rotation_mode = 'XYZ'; key_series(hatch_hinge, 'rotation_euler', 1, ha)
# risers and seats
R_A = RAD['aud']
rows, cols = GEO['rows'], GEO['cols']
def seat_pos(r, c): return np.array([XC + (c - (cols - 1) / 2) * GEO['seat_dx'], GEO['seat_y0'] + r * GEO['row_dy'], GEO['seat_z0'] + r * GEO['row_dz'] + R_A])
for r in range(rows):
    y = GEO['seat_y0'] + r * GEO['row_dy']; zt = GEO['seat_z0'] + r * GEO['row_dz'] - 0.12
    w = (cols + 0.6) * GEO['seat_dx']
    box((XC, y + 0.05, zt - 0.5), (w, GEO['row_dy'] + 0.02, 1.0), M_CARPET, name='riser')
    box((XC, y - GEO['row_dy'] / 2 + 0.06, zt + 0.01), (w, 0.05, 0.03), M_GOLD, name='riserlip')
    for c in range(cols):
        s = seat_pos(r, c)
        box((s[0], s[1] + 0.02, zt + 0.06), (0.5, 0.48, 0.12), M_VELVET, name='seat')
        box((s[0], s[1] + 0.3, zt + 0.32), (0.5, 0.08, 0.5), M_VELVET, name='seatback')
        box((s[0], s[1] + 0.31, zt + 0.58), (0.52, 0.1, 0.04), M_GOLD_D)
# front of the risers: gold balustrade
box((XC, GEO['seat_y0'] - 0.42, GEO['seat_z0'] - 0.3), ((cols + 0.6) * GEO['seat_dx'], 0.1, 0.5), M_GOLD_D, name='balus')
for x in np.linspace(XC - (cols / 2) * GEO['seat_dx'], XC + (cols / 2) * GEO['seat_dx'], 12):
    rod((x, GEO['seat_y0'] - 0.5, GEO['seat_z0'] - 0.55), (x, GEO['seat_y0'] - 0.5, GEO['seat_z0'] - 0.05), 0.04, M_GOLD, 8)
# dark velvet drape behind the orchestra (gold towers read against it)
M_DRAPE = mat('drape', (0.16, 0.012, 0.02), 0.85, **{'Sheen Weight': 1.0})
bm = bmesh.new(); xs = np.linspace(X_L + 0.2, X_R - 0.2, 160); vs = []
for x in xs:
    yy = 1.9 + 0.09 * math.sin(x * 9.0)
    vs.append((bm.verts.new((x, yy, 0.0)), bm.verts.new((x, yy, GEO['seat_z0'] - 0.55))))
for a, b in zip(vs, vs[1:]): bm.faces.new((a[0], b[0], b[1], a[1]))
me = bpy.data.meshes.new('drape'); bm.to_mesh(me); bm.free()
for p in me.polygons: p.use_smooth = True
link(bpy.data.objects.new('drape', me)).data.materials.append(M_DRAPE)
box((XC - 0.4, 1.9, GEO['seat_z0'] - 0.6), (X_R - X_L, 0.2, 0.12), M_GOLD_D)

# ==================================================================== balls
def make_ball(name, r, m, eyes=False, stripe=None):
    o = obj(prim('sph', m, 32, 16), name); o.scale = (r, r, r)
    if eyes:
        for sx in (-1, 1):
            e = sphere((sx * 0.36, -0.86, 0.22), 0.26, M_EYE, (12, 8)); e.parent = o
            pu = sphere((sx * 0.36, -1.08, 0.18), 0.13, M_PUPIL, (10, 6)); pu.parent = o
    if stripe is not None:
        st = torus((0, 0, 0), 1.0, 0.09, stripe, (0, 0, 0)); st.parent = o
    return o
BO = {}
for n in ('G1', 'G2'):
    o = make_ball(n, RAD[n], M_GOLD if n == 'G1' else M_SILVER, stripe=M_RED if n == 'G1' else M_GOLD)
    key_loc(o, POS[n]); key_quat(o, roll_quats(POS[n], RAD[n])); BO[n] = o
# conductor: black, white collar, baton
cnd = make_ball('cond', RAD['cond'], M_BLACK); key_loc(cnd, POS['cond']); BO['cond'] = cnd
col_ = torus((0, 0, 0.72), 0.62, 0.12, M_WHITE); col_.parent = cnd
for sx in (-1, 1):
    tl = obj(prim('cone', M_BLACK, 4, 0.3, 0.02, 1.0), 'tail'); tl.parent = cnd; tl.location = (sx * 0.3, 0.75, -0.75); tl.rotation_euler = (0.5, 0, sx * 0.2)
# baton: pivot at the conductor's right side, swings down on every beat
bat_piv = link(bpy.data.objects.new('batonpiv', None)); bat_piv.parent = cnd; bat_piv.location = (1.0, 0, 0.3)
bt_ = rod((0, 0, 0), (0, 0, 1.6), 0.045, M_WHITE, 8); bt_.parent = bat_piv
CH = EV['cond_hops']
ba = np.zeros(NF)
beats = [b for b, tau in CH]
for f in range(NF):
    t = T[f]; k = np.searchsorted(beats, t)
    if 0 < k < len(beats):
        u = (t - beats[k - 1]) / (beats[k] - beats[k - 1]); ba[f] = 0.9 * math.sin(math.pi * u) - 0.35
    elif k == 0: ba[f] = 0.6
bat_piv.rotation_mode = 'XYZ'; key_series(bat_piv, 'rotation_euler', 1, ba)
# the late marble: red, eyes, gold band; a glowing trail so it can be followed through the whole hall
late = make_ball('late', RAD['late'], M_RED, eyes=True); key_loc(late, POS['late']); BO['late'] = late
band = torus((0, 0, 0.55), 0.84, 0.08, M_GOLD); band.parent = late
TR_N = 7
for k in range(1, TR_N + 1):
    tr_ = sphere((0, 0, 0), RAD['late'] * (0.85 - 0.08 * k), M_FX, (12, 6)); tr_.color = (1.0, 0.3, 0.15, 0)
    d = 2 * k; P = np.concatenate([np.repeat(POS['late'][:1], d, 0), POS['late'][:-d]])
    key_loc(tr_, P)
    sp = np.linalg.norm(POS['late'] - np.concatenate([POS['late'][:1], POS['late'][:-1]]), axis=1) * FPS
    alpha = np.clip(sp / 1.5, 0, 1) * (0.55 - 0.06 * k) * ((T > EV['late_door']['t_rel']) & (T < FINAL + 0.2))
    key_series(tr_, 'color', 3, alpha, 2e-3)
# audience: hops from layout (land on the beat = clap)
AUD = LAY['aud']
APOS = {}
PAL = [(0.15, 0.45, 0.95), (0.1, 0.75, 0.55), (0.98, 0.75, 0.15), (0.6, 0.35, 0.9), (0.95, 0.45, 0.6), (0.3, 0.85, 0.95),
       (0.9, 0.9, 0.95), (0.55, 0.8, 0.25), (0.98, 0.55, 0.2)]
SEATED = []
for r in range(rows):
    for c in range(cols):
        if [r, c] == EV['empty']: continue
        s = seat_pos(r, c); z = np.zeros(NF)
        for tl, tau in AUD[f'{r}_{c}']:
            f0 = max(0, int(math.ceil((tl - tau) * FPS))); f1 = min(NF - 1, int(math.floor(tl * FPS)))
            for f in range(f0, f1 + 1):
                u = T[f] - (tl - tau)
                if 0 <= u <= tau: z[f] = max(z[f], 0.5 * G * u * (tau - u))
        P = np.tile(s, (NF, 1)); P[:, 2] += z; APOS[(r, c)] = P
        o = make_ball(f'a{r}_{c}', R_A, M_OBJ, eyes=True); o.color = (*PAL[(r * 5 + c * 3) % len(PAL)], 1)
        key_loc(o, P); SEATED.append(o)
        o.rotation_euler = (0, 0, rng.uniform(-0.25, 0.25))
# spotlight beam on the empty seat (fades when the late marble sits down)
seat = np.array(EV['seat'])
beam = obj(prim('cone', M_FX, 32, 0.75, 0.12, 1.0), 'beam'); beam.location = (seat[0], seat[1], (Z_CEIL + seat[2]) / 2)
beam.scale = (1, 1, Z_CEIL - seat[2]); beam.color = (1.0, 0.92, 0.7, 0)
ba_ = 0.07 * (1 - smoothstep((T - FINAL) / 0.6)) * smoothstep(T / 0.8)
key_series(beam, 'color', 3, ba_, 1e-4)
spot = bpy.data.lights.new('spot', 'SPOT'); spot.energy = 400; spot.spot_size = 0.5; spot.color = (1, 0.9, 0.7); spot.shadow_soft_size = 0.2
so_ = link(bpy.data.objects.new('spot', spot)); so_.location = (seat[0], seat[1] - 0.3, Z_CEIL - 0.3); so_.rotation_euler = (0.05, 0, 0)
for f in range(0, NF, 3):
    spot.energy = float(400 * (1 - smoothstep((T[f] - FINAL) / 0.6))); spot.keyframe_insert('energy', frame=f)

# flower arrangements along the stage front (Vienna New Year's concert)
FLC = [(1.0, 0.25, 0.35), (1.0, 0.85, 0.3), (1.0, 0.55, 0.15), (0.95, 0.95, 0.95), (0.85, 0.2, 0.6)]
M_LEAF = mat('leaf', (0.08, 0.35, 0.1), 0.6)
for x in np.arange(X_L + 0.35, X_R - 0.2, 0.42):
    if abs(x - pod[0]) < 0.8: continue
    sphere((x, -1.82, 0.05), 0.2, M_LEAF, (12, 6)).scale = (0.22, 0.16, 0.12)
    for k in range(5):
        fo_ = sphere((x + rng.uniform(-0.15, 0.15), -1.85 + rng.uniform(-0.06, 0.06), 0.12 + rng.uniform(0, 0.1)), 0.06, M_OBJM, (10, 6))
        fo_.color = (*FLC[rng.integers(len(FLC))], 1)

# ==================================================================== podium drum (struck by the conductor on every beat)
dz_top = pod[2]
rod((pod[0], pod[1], 0.0), (pod[0], pod[1], dz_top - 0.02), 0.62, M_RED, 32)
for zz in (0.06, dz_top - 0.05):
    torus((pod[0], pod[1], zz), 0.63, 0.045, M_GOLD)
for k in range(10):
    a = 2 * math.pi * k / 10
    rod((pod[0] + 0.64 * math.cos(a), pod[1] + 0.64 * math.sin(a), 0.08), (pod[0] + 0.64 * math.cos(a + 0.6), pod[1] + 0.64 * math.sin(a + 0.6), dz_top - 0.07), 0.015, M_GOLD, 6)
head = obj(prim('cyl', M_KEY, 32), 'drumhead'); head.location = (pod[0], pod[1], dz_top - 0.01); head.scale = (0.6, 0.6, 0.02)
head.color = (1.0, 0.93, 0.8, 0)
key_series(head, 'color', 3, 0.05 + pulse_alpha([b for b, tau in CH] + [E['t0']], decay=3, length=10), 1e-3)

# ==================================================================== spiral glockenspiel towers
TW = {k: np.array(v) for k, v in GEO['tower'].items()}
Z_TOP = GEO['z_top']; R_P = RAD['G1']
KEY_W, KEY_D, KEY_T = 0.2, 0.46, 0.065
COL_R = 0.17
PC = lambda p: ((p - 62) % 12) / 12.0
def key_col(p):
    h = PC(p); return (1.0, 0.78 - 0.18 * h, 0.35 + 0.2 * h)
KEYS = []
for n in ('G1', 'G2'):
    for h in BALLS[n]['hits']:
        if h['kind'] != 'key': continue
        tw = h['tower']; c = TW[tw]
        nrm = np.array(h['n'], float); nrm /= np.linalg.norm(nrm)
        ct = np.array(h['p']) - R_P * nrm                       # contact point
        rad = np.array([ct[0] - c[0], ct[1] - c[1], 0.0]); rad /= np.linalg.norm(rad)
        ya = rad - nrm * (rad @ nrm); ya /= np.linalg.norm(ya); xa = np.cross(ya, nrm)
        Mx = Matrix(((xa[0], ya[0], nrm[0]), (xa[1], ya[1], nrm[1]), (xa[2], ya[2], nrm[2])))
        ctr = ct - nrm * KEY_T / 2
        KEYS.append(dict(ball=n, tower=tw, i=h['i'], t=h['t'], c=ctr, contact=ct, n=nrm, M=Mx, rad=rad, pitch=LEAD[h['i']][1], w=h['kw'], win=h['win']))
for K in KEYS:
    UNF0, UNF1, FLD0, FLD1 = K['win']                         # slide out before the hit, back in after (layout.py)
    o = obj(prim('cube', M_KEY), 'key'); o.rotation_mode = 'QUATERNION'; o.rotation_quaternion = K['M'].to_quaternion()
    arm = rod((0, -0.5 * KEY_D, 0), (0, -0.5 * KEY_D - 1.0, 0), 0.03, M_GOLD, 8); arm.parent = o
    o.color = (*key_col(K['pitch']), 0)
    rk = np.linalg.norm(K['c'][:2] - TW[K['tower']]); L = rk - COL_R - KEY_D / 2
    arm.scale = (0.03, 0.03, L); arm.location = (0, -(KEY_D / 2 + L / 2), 0); arm.rotation_euler = (math.pi / 2, 0, 0)
    colp = np.array([TW[K['tower']][0], TW[K['tower']][1], K['c'][2]]) + K['rad'] * COL_R
    t = K['t']; fh = int(round(t * FPS))
    fr = range(max(0, int((t - UNF0) * FPS) - 1), min(NF, int((t + FLD1) * FPS) + 2))
    s_ = np.array([smoothstep((T[f] - (t - UNF0)) / (UNF0 - UNF1)) * (1 - smoothstep((T[f] - (t + FLD0)) / (FLD1 - FLD0))) for f in range(NF)])
    s_ = np.maximum(s_, 1e-3)
    loc = colp[None, :] + (K['c'] - colp)[None, :] * s_[:, None]
    dip = np.zeros(NF)
    for df, d in ((1, -0.02), (3, 0.006), (6, 0)):
        if fh + df < NF: dip[fh + df] = d
    loc += K['n'][None, :] * dip[:, None]
    for k in range(3):
        key_series(o, 'location', k, loc[:, k]); key_series(o, 'scale', k, s_ * (K['w'], KEY_D, KEY_T)[k])
    a = np.zeros(NF)
    for df in range(18):
        if fh + df < NF: a[fh + df] = math.exp(-df / 4.0)
    key_series(o, 'color', 3, a, 1e-3)
    K['o'] = o
# tower columns, crowns, plinths
for tw, c in TW.items():
    rod((c[0], c[1], 0.0), (c[0], c[1], Z_TOP + 0.75), COL_R, M_GOLD, 24)
    rod((c[0], c[1], 0.0), (c[0], c[1], 0.25), 0.45, M_GOLD_D, 32)
    sphere((c[0], c[1], Z_TOP + 0.85), 0.2, M_GOLD); obj(prim('cone', M_GOLD, 24, 0.22, 0.0, 0.5), 'crown').location = (c[0], c[1], Z_TOP + 1.15)
    for z in np.arange(0.6, Z_TOP + 0.6, 0.6): torus((c[0], c[1], z), COL_R + 0.01, 0.025, M_GOLD_D)
# lift cups: follow the ball while it is aboard, otherwise the planned moves
for tw, c in TW.items():
    runs = [r_ for r_ in LAY['runs'] if r_['tower'] == tw]; bn = runs[0]['ball']
    kf = LAY['lift'][tw]; kt = np.array([k_[0] for k_ in kf])
    follow = [(0.0, runs[0]['t_rel'])] + [(runs[k]['t_catch'], runs[k + 1]['t_rel']) for k in range(len(runs) - 1)] + [(runs[-1]['t_catch'], 99)]
    ANG = np.zeros(NF); ZZ = np.zeros(NF)
    for f in range(NF):
        t = T[f]
        if any(a <= t <= b for a, b in follow):
            p = POS[bn][f]; ANG[f] = math.atan2(p[1] - c[1], p[0] - c[0]); ZZ[f] = p[2]; continue
        k = int(np.searchsorted(kt, t)) - 1; k = max(0, min(len(kf) - 2, k))
        a0, a1 = kf[k], kf[k + 1]; u = smoothstep((t - a0[0]) / max(1e-6, a1[0] - a0[0]))
        da = (a1[1] - a0[1] + math.pi) % (2 * math.pi) - math.pi
        ANG[f] = a0[1] + da * u; ZZ[f] = a0[2] + (a1[2] - a0[2]) * u
    ANG = np.unwrap(ANG)
    piv = link(bpy.data.objects.new('liftpiv' + tw, None)); piv.location = (c[0], c[1], 0)
    key_series(piv, 'location', 2, ZZ); piv.rotation_mode = 'XYZ'; key_series(piv, 'rotation_euler', 2, ANG)
    dirn = GEO['dirn'][tw]
    parts = [torus((0, 0, -0.16), COL_R + 0.03, 0.05, M_SILVER), rod((COL_R, 0, -0.16), (0.72, 0, -0.16), 0.035, M_SILVER, 8),
             torus((0.86, 0, -0.165), 0.14, 0.035, M_SILVER)]
    beta = math.radians(24)
    ce = np.array([0.86, 0, 0]) + np.array([0, dirn * math.cos(beta), -math.sin(beta)]) * 0.33
    for sy in (-1, 1):
        off = np.array([sy * 0.11, 0, -0.17])
        parts.append(rod(np.array([0.86, 0, 0]) + off, ce + off, 0.022, M_SILVER, 6))
    for p_ in parts: p_.parent = piv

# ==================================================================== basement: lobby door, stairs (struck on the off-beats), corridor
LB = BALLS['late']
yb = EV['late_door']['p'][1]
bz0 = base_floor - 0.05
box((XC - 0.4, yb + 1.2, (bz0 - 0.18 - 0.2) / 2), (X_R - X_L + 0.2, 0.2, abs(bz0) + 0.2), M_BRICK, name='basewall')
box((XC - 0.4, yb, bz0 - 0.12), (X_R - X_L + 0.2, 2.2, 0.24), M_DARK, name='basefloor')
for x in np.arange(X_L + 0.4, X_R, 1.2):
    sphere((x, yb + 1.08, -0.5), 0.07, M_LAMP, (10, 6))
door = np.array(EV['late_door']['p'])
box((door[0] - 0.35, yb + 0.3, door[2] + 0.15), (0.7, 0.08, 1.1), M_WOOD, name='doorpanel')
box((door[0] - 0.05, yb - 0.1, door[2] - R_A - 0.04), (0.9, 0.9, 0.08), M_CARPET, name='doorstep')
text('ロビー', (door[0] + 0.1, yb - 0.2, door[2] + 0.5), 0.36, M_TXT_Y)
dl = door + np.array([0, 0, -RAD['late']])
STEPS = [h for h in LB['hits'] if h['kind'] in ('step', 'carpet')]
M_STEPK = glowmat('stepk', gain=3.0, rough=0.5)
for k, h in enumerate(STEPS):
    c = np.array(h['p']) - np.array([0, 0, RAD['late']])
    if h['kind'] == 'carpet': continue
    nx = STEPS[k + 1]['p'][0] if k + 1 < len(STEPS) else c[0] + 0.5
    o = box((c[0], c[1], c[2] - 0.06), (0.42, 0.9, 0.12), M_STEPK, name='step'); o.color = (0.55, 0.3, 0.12, 0)
    key_series(o, 'color', 3, pulse_alpha([h['t']], decay=3, length=10), 1e-3)
    box((c[0], c[1], (c[2] - 0.12 + bz0) / 2), (0.4, 0.88, max(0.02, c[2] - 0.12 - bz0)), M_WOOD, name='riserstep')
corr = [np.array(q) for q in EV['corridor']['pts']]
cp = [q - np.array([0, 0, RAD['late'] + 0.03]) for q in corr]
for a, b in zip(cp, cp[1:]):
    d = b - a; L = np.linalg.norm(d); mid = (a + b) / 2; ang = math.atan2(d[2], d[0])
    box(mid, (L + 0.02, 0.9, 0.06), M_CARPET, rot=(0, -ang, 0), name='corridor')
text('こっち？', (0.6, yb - 0.2, -0.75), 0.75, M_TXT_W)

# ==================================================================== service elevator
el = EV['elevator']; ex, ey = el['x'], el['y']
for sx in (-0.5, 0.5):
    for sy in (-0.45, 0.45):
        rod((ex + sx, ey + sy, bz0), (ex + sx, ey + sy, GEO['cat_z'] + 0.9), 0.04, M_GOLD_D, 8)
box((ex, ey + 0.5, (bz0 + GEO['cat_z'] + 0.9) / 2), (1.05, 0.06, GEO['cat_z'] + 0.9 - bz0), M_BRICK, name='shaftback')
cab = link(bpy.data.objects.new('cabin', None))
cz = np.array([POS['late'][f, 2] if el['t_in'] - 0.1 <= T[f] <= el['t_arr'] + 0.35 else (el['z0'] if T[f] < el['t_in'] else el['z1']) for f in range(NF)])
cz = cz - RAD['late']
key_series(cab, 'location', 2, cz); cab.location = (ex, ey, 0)
cab.keyframe_insert('location', index=0, frame=0); cab.keyframe_insert('location', index=1, frame=0)
flo_piv = link(bpy.data.objects.new('cabfloor', None)); flo_piv.parent = cab; flo_piv.location = (-0.42, 0, 0)
fl = box((0.42, 0, -0.04), (0.86, 0.86, 0.08), M_GOLD, name='cabfloor'); fl.parent = flo_piv
tlt = np.array([0.08 * smoothstep((t - EV['catwalk']['t_tilt']) / 0.25) if t < EV['catwalk']['t_tilt'] + 2.5 else 0.08 * (1 - smoothstep((t - EV['catwalk']['t_tilt'] - 2.5) / 0.5)) for t in T])
flo_piv.rotation_mode = 'XYZ'; key_series(flo_piv, 'rotation_euler', 1, tlt)
for o_ in [box((0.45, 0, 0.45), (0.05, 0.86, 0.9), M_GLASS, name='cabwall'), box((0, 0.43, 0.45), (0.9, 0.04, 0.9), M_GOLD_D),
           box((0, 0, 0.93), (0.96, 0.92, 0.06), M_GOLD)]:
    o_.parent = cab
lamp_c = sphere((0, -0.3, 0.88), 0.06, M_LAMP, (10, 6)); lamp_c.parent = cab
# floor indicator lamps flash as the cabin passes; the ding bell at the top is struck by the cabin's hammer on arrival
M_IND = glowmat('ind', gain=8, base=0.05)
for zf in np.arange(-1.5, GEO['cat_z'], 2.2):
    o = sphere((ex - 0.56, ey - 0.46, zf), 0.07, M_IND, (10, 6)); o.color = (1, 0.6, 0.2, 0)
    tp = [T[f] for f in range(1, NF) if (cz[f - 1] - zf) * (cz[f] - zf) <= 0 and cz[f] != cz[f - 1]]
    key_series(o, 'color', 3, pulse_alpha(tp, decay=6, length=20), 1e-3)
bell = obj(prim('cone', M_KEY, 24, 0.22, 0.08, 0.26), 'ding'); bell.location = (ex, ey - 0.1, GEO['cat_z'] + 1.12); bell.color = (1, 0.85, 0.4, 0)
key_series(bell, 'color', 3, pulse_alpha([el['t_arr']], decay=6, length=24), 1e-3)
hm = rod((0, 0, 0.94), (0, 0, 1.08), 0.03, M_STEEL, 8); hm.parent = cab
text('業務用エレベーター', (ex, ey - 0.6, bz0 + 0.95), 0.22, M_TXT_W, rot=(math.pi / 2, 0, 0))

# ==================================================================== catwalk + see-saw catapult + weight
cw = [np.array(q) - np.array([0, 0, RAD['late'] + 0.02]) for q in EV['catwalk']['pts']]
for a, b in zip(cw[:-2], cw[1:-1]):
    d = b - a; L = np.linalg.norm(d); mid = (a + b) / 2; ang = math.atan2(d[2], d[0])
    box(mid - np.array([0, 0, 0.03]), (L + 0.03, 0.55, 0.06), M_GOLD_D, rot=(0, -ang, 0), name='catwalk')
for x in (cw[1][0], cw[2][0]):
    rod((x, cw[0][1], Z_CEIL), (x, cw[0][1], cw[1][2] - 0.05), 0.02, M_STEEL, 6)
ss = EV['seesaw']; piv = np.array(ss['pivot']); ARM, ARMW, TILT = ss['arm'], ss['arm_w'], math.radians(ss['tilt'])
sw = link(bpy.data.objects.new('seesaw', None)); sw.location = tuple(piv); sw.rotation_mode = 'XYZ'
pl = box(((ARM - ARMW) / 2, 0, 0), (ARM + ARMW + 0.2, 0.36, 0.07), M_WOOD, name='plank'); pl.parent = sw
cupo = torus((ARM, 0, 0.06), 0.15, 0.04, M_GOLD); cupo.parent = sw
cone_ = obj(prim('cone', M_GOLD_D, 4, 0.3, 0.03, 0.55), 'fulcrum'); cone_.location = (piv[0], piv[1], piv[2] - 0.3)
rod((piv[0], piv[1], piv[2] - 0.55), (piv[0], piv[1], Z_CEIL), 0.02, M_STEEL, 6)
tl_ = ss['t_launch']
swa = np.array([TILT if t < tl_ else TILT - 2 * TILT * smoothstep((t - tl_) / 0.08) for t in T])
key_series(sw, 'rotation_euler', 1, swa)
RW = 0.26
w_top = lambda a: piv + np.array([-ARMW * math.cos(a), 0, ARMW * math.sin(a)])
wb = sphere((0, 0, 0), RW, mat('weight', (0.35, 0.2, 0.08), 0.25, **{'Metallic': 1.0}), (32, 16), 'weight')
WP = np.zeros((NF, 3))
for f in range(NF):
    t = T[f]; trel = ss['t_release']; a = swa[f]
    end = w_top(a) + np.array([0, 0, 0.035 + RW])
    hz = w_top(TILT)[2] + 0.035 + RW + ss['fall']
    if t < trel: WP[f] = (end[0], end[1], hz)
    elif t < tl_: WP[f] = (end[0], end[1], hz - 0.5 * G * (t - trel) ** 2)
    else: WP[f] = end
key_loc(wb, WP)
hz0 = w_top(TILT)[2] + 0.035 + RW + ss['fall']
box((WP[0, 0], piv[1], hz0 + RW + 0.06), (0.62, 0.62, 0.05), M_STEEL)
latch = link(bpy.data.objects.new('latch', None)); latch.location = (WP[0, 0] + 0.31, piv[1], hz0 - RW - 0.02)
lf = box((-0.3, 0, 0), (0.6, 0.5, 0.04), M_STEEL, name='latch'); lf.parent = latch
latch.rotation_mode = 'XYZ'; key_series(latch, 'rotation_euler', 1, np.array([-1.4 * smoothstep((t - ss['t_release']) / 0.1) for t in T]))
for dx in (-0.31, 0.31):
    rod((WP[0, 0] + dx, piv[1], hz0 + RW + 0.06), (WP[0, 0] + dx, piv[1], Z_CEIL), 0.018, M_STEEL, 6)

# ==================================================================== glass pipe on the back wall + golden glass funnel
PI = EV['pipe']
R_PIPE = 0.28
tube(PI['main'], R_PIPE, M_GLASS, 'pipe1')
tube(PI['spur'], R_PIPE, M_GLASS, 'spur')
tube(PI['leg3'], R_PIPE, M_GLASS, 'pipe3')
cap = sphere(PI['spur'][-1], R_PIPE + 0.01, M_GOLD, (16, 8))
for q in PI['main'][::12] + PI['leg3'][::10]:
    rod((q[0], q[1] + 0.25, q[2]), (q[0], Y_BACK, q[2]), 0.025, M_GOLD_D, 6)
# bristle mouth (funnel opening) catching the flying marble
mo = np.array(PI['mouth'])
m1 = obj(prim('cone', M_GLASS_F, 32, 0.55, R_PIPE, 0.45), 'mouth'); m1.location = mo + np.array([0, -0.15, 0.02]); m1.rotation_euler = (math.pi / 2, 0, 0)
torus(mo + np.array([0, -0.37, 0.02]), 0.55, 0.04, M_GOLD, (math.pi / 2, 0, 0))
text('迷子ルート？', (PI['spur'][-1][0] - 0.4, PI['y'] - 0.4, PI['spur'][-1][2] + 0.45), 0.3, M_TXT_W)
# junction flap: flips when the marble rolls back
J = np.array(PI['spur'][0])
flap = link(bpy.data.objects.new('flap', None)); flap.location = (J[0], J[1] - 0.3, J[2] + 0.1)
fp = box((0, 0, 0.15), (0.04, 0.05, 0.3), M_GOLD, name='flap'); fp.parent = flap
flap.rotation_mode = 'XYZ'; key_series(flap, 'rotation_euler', 1, np.array([0.5 - 1.0 * smoothstep((t - PI['t_back']) / 0.15) for t in T]))
FU = EV['funnel']; fc = np.array(FU['c'])
Bk = FU['depth'] / (1 / FU['rmin'] - 1 / FU['r0'])
zf = lambda r: fc[2] + FU['depth'] - Bk * (1 / r - 1 / FU['r0'])
bm = bmesh.new(); NS = 48; prof = np.linspace(FU['rmin'], FU['r0'] + 0.04, 18)
ring = [[bm.verts.new((r * math.cos(2 * math.pi * k / NS) + fc[0], r * math.sin(2 * math.pi * k / NS) + fc[1], zf(min(r, FU['r0'])) - 0.02)) for k in range(NS)] for r in prof]
for a, b in zip(ring, ring[1:]):
    for k in range(NS): bm.faces.new((a[k], a[(k + 1) % NS], b[(k + 1) % NS], b[k]))
me = bpy.data.meshes.new('funnel'); bm.to_mesh(me); bm.free()
for p in me.polygons: p.use_smooth = True
fo = link(bpy.data.objects.new('funnel', me)); fo.data.materials.append(M_GLASS_F)
torus((fc[0], fc[1], zf(FU['r0']) - 0.02), FU['r0'] + 0.04, 0.05, M_GOLD)
sp_ = EV['spout']['p']
rod((fc[0], fc[1], zf(FU['rmin']) - 0.02), (fc[0], fc[1], sp_[2] - 0.05), FU['rmin'] + 0.11, M_GLASS_F, 24)
torus((fc[0], fc[1], sp_[2] - 0.05), FU['rmin'] + 0.11, 0.035, M_GOLD)
for a in (0.4, 2.5, 4.6):
    rod((fc[0] + (FU['r0'] + 0.04) * math.cos(a), fc[1] + (FU['r0'] + 0.04) * math.sin(a), zf(FU['r0'])), (fc[0] + 0.5 * math.cos(a), fc[1] + 0.5 * math.sin(a), Z_CEIL), 0.015, M_GOLD_D, 6)

# ==================================================================== chandeliers, lights, confetti
STABT = E['stab']
M_CRY = glowmat('crystal', gain=10, base=0.6, rough=0.1)
cry = []
for cx_, cy_ in ((XC - 3.3, 1.2), (XC + 2.9, 1.2)):
    cz_ = Z_CEIL - 0.75
    rod((cx_, cy_, Z_CEIL), (cx_, cy_, cz_ + 0.3), 0.025, M_GOLD, 6)
    torus((cx_, cy_, cz_), 0.55, 0.04, M_GOLD); torus((cx_, cy_, cz_ - 0.25), 0.32, 0.035, M_GOLD)
    for k in range(14):
        a = 2 * math.pi * k / 14
        o = sphere((cx_ + 0.55 * math.cos(a), cy_ + 0.55 * math.sin(a), cz_ - 0.08), 0.06, M_CRY, (10, 6)); o.color = (1, 0.85, 0.6, 0); cry.append(o)
    lt = bpy.data.lights.new('chand', 'POINT'); lt.energy = 900; lt.color = (1, 0.82, 0.6); lt.shadow_soft_size = 0.6
    link(bpy.data.objects.new('chand', lt)).location = (cx_, cy_, cz_ - 0.4)
ca = pulse_alpha(STABT, decay=3, length=10, amp=0.6)
for o in cry: key_series(o, 'color', 3, ca, 2e-3)
# key light from the camera side + warm fill + stage footlights
for loc, en, size, col in (((XC, -14, 13), 9000, 12, (1.0, 0.92, 0.82)), ((XC - 6, -6, 3), 1500, 6, (1.0, 0.8, 0.6)), ((XC + 6, -6, 9), 1500, 6, (1.0, 0.85, 0.7))):
    ld = bpy.data.lights.new('area', 'AREA'); ld.energy = en; ld.size = size; ld.color = col
    lo = link(bpy.data.objects.new('area', ld)); lo.location = loc
    d = Vector((XC, 3.0, 6.0)) - Vector(loc); lo.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
for x in np.arange(X_L + 0.5, X_R - 0.3, 0.7):
    sphere((x, -1.98, 0.04), 0.05, M_LAMP, (8, 6))
lb = bpy.data.lights.new('base', 'AREA'); lb.energy = 300; lb.size = 8; lb.color = (1, 0.7, 0.45)
lbo = link(bpy.data.objects.new('base', lb)); lbo.location = (XC, yb - 1.5, -0.6); lbo.rotation_euler = (math.radians(70), 0, 0)
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs['Color'].default_value = (0.25, 0.16, 0.1, 1); w.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.25
# confetti + gold streamers from the ceiling on the final chord
CONFC = [(1, 0.82, 0.25), (0.95, 0.15, 0.2), (1, 1, 1), (0.3, 0.6, 1), (0.4, 0.9, 0.5), (1, 0.5, 0.8)]
for k in range(140):
    cf = obj(prim('plane', M_OBJM), 'conf'); cf.color = (*CONFC[k % 6], 1)
    x0 = rng.uniform(X_L + 0.3, X_R - 0.3); y0 = rng.uniform(-1.5, 7.0); z0 = Z_CEIL - 0.1
    vx = rng.uniform(-0.6, 0.6); fall = rng.uniform(0.9, 1.6); ph = rng.uniform(0, 6.28)
    P = np.zeros((NF, 3)); RT = np.zeros((NF, 3)); S = np.zeros(NF)
    t0 = FINAL + rng.uniform(0.0, 0.25)
    for f in range(NF):
        s_ = T[f] - t0
        if s_ < 0: P[f] = (x0, y0, z0); continue
        burst = 1 - math.exp(-s_ / 0.2)
        P[f] = (x0 + vx * burst + 0.15 * math.sin(3 * s_ + ph), y0, z0 - 1.6 * burst - fall * s_); RT[f] = (s_ * (3 + k % 5), s_ * (2 + k % 3), s_ * 1.7)
        S[f] = 0.09
    key_loc(cf, P)
    for c_ in range(3): key_series(cf, 'scale', c_, np.maximum(S, 1e-4)); key_series(cf, 'rotation_euler', c_, RT[:, c_])

# ==================================================================== camera (fixed wide shot of the whole cut-away, slight breathing)
LENS = 50.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 400
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
FOC = np.zeros((NF, 3)); WID = np.zeros(NF)
for f in range(NF):
    t = T[f]
    u = smoothstep((t - 50.0) / 3.0)
    WID[f] = 11.7 - 2.5 * u
    FOC[f] = (XC + 0.1 - 0.1 * u, 3.0, 5.6 + 1.9 * u)
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24
    CP[f] = (FOC[f, 0], FOC[f, 1] - D, FOC[f, 2] + 0.1 * D); TP[f] = FOC[f]
key_loc(cam, CP); key_loc(tgt, TP)


# ---------------- captions (HUD on the camera) ----------------
def hud_text(s, xy, size, m, t0, t1, name='cap'):
    hud = link(bpy.data.objects.new('hud', None)); hud.parent = cam; hud.location = (0, 0, -3.0)
    o = text(s, (xy[0], xy[1], 0.002), size, m, rot=(0, 0, 0), ext=0.0); o.parent = hud; o.visible_shadow = False
    bd = box((0, xy[1], -0.002), (HWd * 2.2, size * 1.45, 0.001), M_BAND, name='band'); bd.parent = hud; bd.visible_shadow = False
    vis = np.array([1.0 if t0 <= t <= t1 else 1e-3 for t in T])
    for ob in (o, bd):
        for k in range(3): key_series(ob, 'scale', k, vis * (1 if ob is o else np.array([HWd * 2.2, size * 1.45, 0.001])[k]), 1e-4)
    return o
HH = 3.0 * 24 / LENS; HWd = HH * 9 / 16
hud_text('観客が全員ビー玉の', (0, HH / 2 - 0.12, 0), 0.095, M_TXT_W, 0.0, 3.6)
hud_text('ニューイヤーコンサート', (0, HH / 2 - 0.26, 0), 0.115, M_TXT_Y, 0.0, 3.6)
hud_text('…1つだけ空席がある', (0, -HH / 2 + 0.2, 0), 0.09, M_TXT_W, 1.6, 6.0)


def follow_text(s, ball, t0, t1, off, size, m):
    o = text(s, (0, 0, 0), size, m)
    P = POS[ball] + np.array(off)[None, :]
    key_loc(o, P)
    vis = np.array([1.0 if t0 <= t <= t1 else 1e-3 for t in T])
    for k in range(3): key_series(o, 'scale', k, vis, 1e-4)
    return o
lab = link(bpy.data.objects.new('labp', None)); lab.location = (door[0] + 2.4, -1.75, -0.5)
lt_ = text('← 遅刻した玉', (0, -0.01, 0), 0.8, M_TXT_R); lt_.parent = lab
lp_ = box((0, 0.03, 0), (2.7, 0.02, 0.62), M_BAND, name='plate'); lp_.parent = lab
vis = np.array([1.0 if EV['late_door']['t_rel'] - 0.8 <= t <= EV['stairs']['t1'] else 1e-3 for t in T])
for k in range(3): key_series(lab, 'scale', k, vis, 1e-4)
follow_text('まちがえた！', 'late', PI['t_junction'] + 0.2, PI['t_back'] + 0.6, (-0.3, -1.2, 0.8), 1.2, M_TXT_R)
hud_text('間に合った！', (0, HH / 2 - 0.2, 0), 0.17, M_TXT_Y, FINAL + 0.05, 99)


# ---------------- sanity checks ----------------
def check():
    info = {}
    gaps = []
    for K in KEYS:
        f = int(round(K['t'] * FPS)); p = POS[K['ball']][min(NF - 1, f)]
        gaps.append(np.linalg.norm(p - (K['contact'] + R_P * K['n'])))
    info['key_contact_gap_max'] = round(max(gaps), 3)
    info['keys'] = len(KEYS); info['objects'] = len(bpy.data.objects); info['NF'] = NF
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
