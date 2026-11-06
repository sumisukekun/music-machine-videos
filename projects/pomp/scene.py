"""Pomp and Circumstance - a royal palace clockwork ceremony.
A golden marble descends a switch-back grand staircase on the palace front. Every step it lands on is one melody note:
the brass step presses a little hammer that strikes the organ pipe standing behind it (pipe length = pitch).
At the row ends the marble swings round brass U-chutes. Automaton trumpeters on the balcony, driven by visible
pneumatic pistons and link rods, raise their trumpets for the fanfares and the grand reprise; banners unroll for the
reprise; at the final chord the marble drops into a cup whose lever hauls a cable that lifts the crown, and fireworks burst."""
import os, sys, json, math, time
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix, Quaternion

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.5) * FPS); T = np.arange(NF) / FPS
rng = np.random.default_rng(7)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'
sc.cycles.samples = int(os.environ.get('SAMP', 6)); sc.cycles.use_adaptive_sampling = True; sc.cycles.adaptive_threshold = 0.06
sc.cycles.use_denoising = True
for k, v in (('denoiser', 'OPENIMAGEDENOISE'), ('use_fast_gi', True)):
    try: setattr(sc.cycles, k, v)
    except Exception: pass
sc.cycles.max_bounces = 4; sc.cycles.diffuse_bounces = 1; sc.cycles.glossy_bounces = 2; sc.cycles.transmission_bounces = 1
sc.render.resolution_x, sc.render.resolution_y = 1080, 1920
sc.render.resolution_percentage = int(os.environ.get('PCT', 50)); sc.render.fps = FPS
sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.3
try: sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Medium High Contrast'
except Exception: pass
sc.frame_start, sc.frame_end = 0, NF - 1


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


def glowmat(name, base_col, metal=1.0, rough=0.22, gain=12.0, base=0.0):
    """metallic surface; object colour = glow colour, object alpha drives emission"""
    m = mat(name, base_col, rough, **{'Metallic': metal}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    oi = nt.nodes.new('ShaderNodeObjectInfo')
    nt.links.new(oi.outputs['Color'], b.inputs['Emission Color'])
    mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY_ADD'; mul.inputs[1].default_value = gain; mul.inputs[2].default_value = base
    nt.links.new(oi.outputs['Alpha'], mul.inputs[0]); nt.links.new(mul.outputs[0], b.inputs['Emission Strength'])
    return m


def fadeemis(name, gain=25.0):
    """pure emission; colour from object colour, strength from object alpha"""
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    oi = nt.nodes.new('ShaderNodeObjectInfo'); e = nt.nodes.new('ShaderNodeEmission')
    mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = gain
    nt.links.new(oi.outputs['Color'], e.inputs['Color']); nt.links.new(oi.outputs['Alpha'], mul.inputs[0]); nt.links.new(mul.outputs[0], e.inputs['Strength'])
    o = nt.nodes.new('ShaderNodeOutputMaterial'); nt.links.new(e.outputs[0], o.inputs[0]); return m


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


def box(c, s, mt, bev=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=c); o = bpy.context.object; o.scale = s
    if bev: m = o.modifiers.new('b', 'BEVEL'); m.width = bev; m.segments = 2
    o.data.materials.append(mt); return o


def cyl(c, r, h, mt, verts=24, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=h, vertices=verts, location=c, rotation=rot)
    o = bpy.context.object; o.data.materials.append(mt)
    for p in o.data.polygons: p.use_smooth = True
    return o


def sph(c, r, mt, seg=24):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, segments=seg, ring_count=seg // 2, location=c)
    o = bpy.context.object; o.data.materials.append(mt)
    for p in o.data.polygons: p.use_smooth = True
    return o


def moving_rod(name, P0, P1, r, mt):
    """cylinder whose ends follow two per-frame point arrays (NF,3)"""
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=1.0, vertices=8)
    o = bpy.context.object; o.name = name; o.data.materials.append(mt); o.rotation_mode = 'QUATERNION'
    D = P1 - P0; Ln = np.linalg.norm(D, axis=1); M = (P0 + P1) / 2
    Q = np.zeros((NF, 4))
    for f in range(NF):
        q = Vector((0, 0, 1)).rotation_difference(Vector(D[f] / max(Ln[f], 1e-6))); Q[f] = (q.w, q.x, q.y, q.z)
    for i in range(3): key_series(o, 'location', i, M[:, i])
    for i in range(4): key_series(o, 'rotation_quaternion', i, Q[:, i])
    key_series(o, 'scale', 2, Ln); return o


def smooth(X, k):
    Y = X.copy()
    for f in range(1, len(X)): Y[f] = Y[f - 1] + (X[f] - Y[f - 1]) * (1 - math.exp(-1 / FPS / k))
    return Y


def ss(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


# ---------------- materials ----------------
GOLD = mat('gold', (1.0, 0.72, 0.32), 0.18, **{'Metallic': 1.0})
BRASS = mat('brass', (0.95, 0.68, 0.32), 0.3, **{'Metallic': 1.0})
STONE = mat('stone', (0.5, 0.46, 0.4), 0.75)
STONE2 = mat('stone2', (0.42, 0.38, 0.33), 0.8)
WALLB = mat('wallb', (0.55, 0.12, 0.12), 0.7)
NAVY = mat('navy', (0.03, 0.05, 0.18), 0.6)
RED = mat('red', (0.75, 0.04, 0.05), 0.5)
BLACK = mat('black', (0.02, 0.02, 0.02), 0.6)
SKIN = mat('skin', (0.9, 0.72, 0.6), 0.6)
WIN = emis('win', (1.0, 0.68, 0.32), 3.0)
WALL_Y = 0.62
TREAD_M = glowmat('tread', (0.95, 0.7, 0.35), gain=6.0)
PIPE_M = glowmat('pipe', (0.92, 0.9, 0.86), metal=1.0, rough=0.15, gain=16.0)

# ---------------- the staircase path ----------------
V = 0.5; HS = 0.07; RB = 0.085; W = 1.7; R = 0.24; LB = math.pi * R; TF = math.sqrt(2 * HS / 9.81); DF = V * TF
TL = [l[0] for l in LEAD]; PL = [l[1] for l in LEAD]; NL = len(TL)
PRE = 1.2                                  # roll time from the release gate to the first landing
T_REL = TL[0] - PRE
Z_TOP = 8.6                                # ball-centre height on the first tread
ZBAL = Z_TOP + 0.75
segs = []   # (kind, t0, t1, data)  kind: 'line' (x0,z,dir) | 'drop' (x0,z0,dir) | 'bend' (cx,cz,dir)
x = -W - 0.15; z = Z_TOP + HS; dirn = 1; t = T_REL; rowlen = 0.0
TREADS = []  # (x0, x1, ztop, note_index or None)
ROW = [0]; TROW = []
tread_start = x
def add_line(L, note_idx):
    global x, t
    segs.append(('line', t, t + L / V, (x, z, dirn))); x += dirn * L; t += L / V
def add_drop():
    global x, z, t
    segs.append(('drop', t, t + TF, (x, z, dirn))); x += dirn * DF; z -= HS; t += TF
def add_bend():
    global x, z, t, dirn
    segs.append(('bend', t, t + LB / V, (x, z - R, dirn))); z -= 2 * R; dirn = -dirn; t += LB / V; ROW[0] += 1
# pre-roll from the gate to the first landing
cur_note = None
gaps = [PRE] + [TL[i + 1] - TL[i] for i in range(NL - 1)]
for gi, g in enumerate(gaps):
    note_idx = gi - 1            # tread belongs to the note that landed at its start (None for the pre-roll)
    L = V * g
    room = W - dirn * x
    if note_idx >= 0 and L >= LB + DF + 0.12 and (room < 1.3 or room < L + 0.25):
        s1 = max(0.06, min(room, L - LB - DF - 0.06))
        x0 = x; add_line(s1, note_idx); TREADS.append((x0, x, z, note_idx, dirn)); TROW.append(ROW[0])
        add_bend()
        s2 = L - s1 - LB - DF
        x0 = x; add_line(s2, None); TREADS.append((x0, x, z, None, dirn)); TROW.append(ROW[0])
        add_drop()
    else:
        x0 = x; add_line(L - DF, note_idx); TREADS.append((x0, x, z, note_idx if note_idx >= 0 else -1, dirn)); TROW.append(ROW[0])
        add_drop()
# last note: roll on to the cup
x0 = x; END_ROLL = 0.75; add_line(END_ROLL, NL - 1); TREADS.append((x0, x, z, NL - 1, dirn)); TROW.append(ROW[0])
T_CUP = t; CUP_X = x + dirn * 0.16; CUP_Z = z - 0.28; END_DIR = dirn
# centre the whole staircase (including the U-chutes) on x = 0
_ex = []
for kind, t0, t1, d in segs:
    _ex.append(d[0])
    if kind == 'bend': _ex.append(d[0] + d[2] * (R + RB + 0.03))
    if kind == 'line': _ex.append(d[0] + d[2] * V * (t1 - t0))
XO = -(min(_ex) + max(_ex)) / 2
segs = [(k_, a_, b_, (d_[0] + XO,) + tuple(d_[1:])) for k_, a_, b_, d_ in segs]
TREADS = [(a_ + XO, b_ + XO, c_, d_, e_) for a_, b_, c_, d_, e_ in TREADS]
x += XO; CUP_X += XO; X_START = -W - 0.15 + XO
xs = [s[3][0] for s in segs]
print('path: segments', len(segs), 'x range', round(min(_ex) + XO, 2), round(max(_ex) + XO, 2), 'XO', round(XO, 2), 'z bottom', round(z, 2), 'cup t', round(T_CUP, 2))
LAND_T = {}   # check landings
for s in segs:
    if s[0] == 'drop': pass


def ball_at(tq):
    if tq <= T_REL: return np.array([X_START, 0.0, Z_TOP + HS])
    for kind, t0, t1, d in segs:
        if t0 <= tq <= t1:
            u = tq - t0
            if kind == 'line': return np.array([d[0] + d[2] * V * u, 0.0, d[1]])
            if kind == 'drop': return np.array([d[0] + d[2] * V * u, 0.0, d[1] - 0.5 * 9.81 * u * u])
            if kind == 'bend':
                th = V * u / R; return np.array([d[0] + d[2] * R * math.sin(th), 0.0, d[1] + R * math.cos(th)])
    # into the cup
    u = tq - T_CUP; fall = min(0.28, 0.5 * 9.81 * u * u)
    hx = min(0.16, V * u)
    return np.array([x + END_DIR * hx, 0.0, z - fall])


BP = np.array([ball_at(tq) for tq in T])
# verify landings land on the beat
for i, tl in enumerate(TL[:8]):
    f = int(round(tl * FPS)); print('note', i, round(tl, 2), 'z', round(BP[f][2], 3), round(BP[f - 2][2], 3))

# ---------------- treads, pipes, hammers ----------------
pmin, pmax = min(PL), max(PL)
PIPES = []
for (x0, x1, zc, ni, dr) in TREADS:
    lo, hi = min(x0, x1), max(x0, x1)
    if ni is not None and ni >= 0: lo, hi = (lo - DF - 0.01, hi) if dr > 0 else (lo, hi + DF + 0.01)
    ztop = zc - RB
    o = box(((lo + hi) / 2, 0.0, ztop - 0.02), (hi - lo, 0.26, 0.04), TREAD_M, 0.008)
    # front lip rail and risers
    box(((lo + hi) / 2, -0.135, ztop + 0.012), (hi - lo, 0.012, 0.03), GOLD)
    if ni is not None and ni >= 0:
        tn = TL[ni]
        a = np.where(T >= tn - 0.5 / FPS, 0.15 + 0.85 * np.exp(-np.clip(T - tn, 0, None) / 0.35), 0.0)
        key_alpha(o, a, (1.0, 0.62, 0.22))
        d = np.exp(-((T - tn - 0.02) / 0.05) ** 2) * 0.012
        key_series(o, 'location', 2, ztop - 0.02 - d)
    # back shelf for the pipes
    box(((lo + hi) / 2, 0.235, ztop - 0.07), (hi - lo + 0.002, 0.17, 0.035), GOLD)
    if ni is None or ni < 0: continue
    PIPES.append((ni, (x0 + (DF + 0.06) * dr), ztop, TREADS.index((x0, x1, zc, ni, dr))))
for k, (ni, px, ztop, ti) in enumerate(PIPES):
    base = ztop - 0.05
    q = (pmax - PL[ni]) / max(1, pmax - pmin)
    hgt = 0.3 + 0.45 * q; rr = 0.045 + 0.025 * q
    if TROW[ti] == 0: hgt = min(hgt, (ZBAL - 0.16) - base - 0.06)
    py = 0.43 if TROW[ti] % 2 == 0 else 0.54
    pz = base + hgt / 2
    pipe = cyl((px, py, pz), rr, hgt, PIPE_M, 20)
    box((px, py - rr * 0.92, base + 0.07), (rr * 0.9, 0.006, 0.035), BLACK)
    bpy.ops.mesh.primitive_cone_add(radius1=rr * 0.35, radius2=rr, depth=0.06, vertices=16, location=(px, py, base - 0.01))
    bpy.context.object.data.materials.append(PIPE_M)
    rod((px, py, base - 0.04), (px, WALL_Y, base - 0.04), 0.012, BRASS, 6)        # wall bracket
    tn = TL[ni]
    a = np.where(T >= tn - 0.5 / FPS, 0.06 + 0.94 * np.exp(-np.clip(T - tn, 0, None) / 0.5), 0.0)
    key_alpha(pipe, a, (1.0, 0.75, 0.35))
    # hammer: hinged on the shelf front, its tail under the tread; the step pushes it into the pipe
    hp = link(bpy.data.objects.new(f'hpiv{k}', None)); hp.location = (px + 0.07, 0.17, base + 0.02); hp.rotation_mode = 'XYZ'
    hm = rod((0, 0, 0), (0, py - rr - 0.03 - 0.17, 0.07), 0.008, BRASS, 6); hm.parent = hp
    hh = sph((0, py - rr - 0.035 - 0.17, 0.07), 0.024, BRASS, 12); hh.parent = hp
    kick = np.where(T >= tn, np.exp(-np.clip(T - tn, 0, None) / 0.07), 0) * np.where(T >= tn - 0.04, 1, 0)
    key_series(hp, 'rotation_euler', 0, -0.35 + 0.35 * kick)

# ---------------- U-chutes at the row ends ----------------
for kind, t0, t1, d in segs:
    if kind != 'bend': continue
    cx, cz, dr = d
    for yy in (-0.13, 0.13):
        cu = bpy.data.curves.new('ch', 'CURVE'); cu.dimensions = '3D'; cu.bevel_depth = 0.012; cu.bevel_resolution = 2
        sp = cu.splines.new('POLY'); n = 40; sp.points.add(n - 1)
        for i in range(n):
            th = math.pi * i / (n - 1); rr = R + RB + 0.012
            sp.points[i].co = (cx + dr * rr * math.sin(th), yy, cz + rr * math.cos(th), 1)
        link(bpy.data.objects.new('chute', cu)).data.materials.append(GOLD)
    # back plate
    bm = bmesh.new(); vs = []
    for i in range(33):
        th = math.pi * i / 32; rr = R + RB + 0.03
        vs.append((bm.verts.new((cx + dr * rr * math.sin(th), 0.15, cz + rr * math.cos(th))), bm.verts.new((cx, 0.15, cz))))
    for i in range(32):
        try: bm.faces.new((vs[i][0], vs[i + 1][0], vs[i][1]))
        except ValueError: pass
    me = bpy.data.meshes.new('cb'); bm.to_mesh(me); ob = link(bpy.data.objects.new('cbk', me)); ob.data.materials.append(WALLB)

# ---------------- release gate + cup + crown cable ----------------
gx, gz = X_START, Z_TOP + HS - RB
box((gx - 0.1, 0.0, gz - 0.02), (0.45, 0.26, 0.04), GOLD)
cyl((gx - 0.25, 0.0, gz + 0.25), 0.11, 0.5, GOLD, 24)                       # magazine tube holding spare marbles
for k in range(3): sph((gx - 0.25, 0.0, gz + 0.1 + RB + k * 0.17), RB, GOLD)
GATE = link(bpy.data.objects.new('gatep', None)); GATE.location = (gx + 0.12, 0.0, gz); GATE.rotation_mode = 'XYZ'
g_ = box((0, 0, 0.09), (0.02, 0.24, 0.18), BRASS); g_.parent = GATE
op = ss((T - (T_REL - 0.25)) / 0.2); key_series(GATE, 'rotation_euler', 1, op * 1.4)
# cup on a lever
CUPP = link(bpy.data.objects.new('cupp', None)); CUPP.location = (CUP_X - END_DIR * 0.45, 0.0, CUP_Z - 0.05); CUPP.rotation_mode = 'XYZ'
bpy.ops.mesh.primitive_cone_add(radius1=0.06, radius2=0.13, depth=0.16, vertices=32, location=(END_DIR * 0.45, 0, 0.08))
cup = bpy.context.object; cup.data.materials.append(GOLD); cup.parent = CUPP
la = rod((-END_DIR * 0.5, 0, 0.0), (END_DIR * 0.45, 0, 0.0), 0.02, BRASS); la.parent = CUPP
cyl((CUP_X - END_DIR * 0.45, 0.0, CUP_Z - 0.12), 0.035, 0.14, STONE2, 12)
dip = ss((T - T_CUP - 0.05) / 0.5)
key_series(CUPP, 'rotation_euler', 1, END_DIR * 0.28 * dip)
# ---------------- palace ----------------
WALL_Y = 0.62
ZB = z - RB - 0.6                     # courtyard level
print('ground z', round(ZB, 2))
H = Z_TOP + 1.6 - ZB
box((0, WALL_Y + 0.2, ZB + H / 2), (9.0, 0.4, H), STONE)
for xx in (-4.2, -3.25, 3.25, 4.2):
    cyl((xx, WALL_Y - 0.05, ZB + H / 2), 0.2, H, STONE2, 24)
    box((xx, WALL_Y - 0.05, ZB + H - 0.1), (0.55, 0.55, 0.2), STONE2)
for zz in np.arange(ZB + 1.2, ZB + H - 0.6, 1.6):
    for xx in (-3.75, 3.75):
        box((xx, WALL_Y - 0.01, zz), (0.55, 0.02, 0.95), WIN)
        box((xx, WALL_Y - 0.03, zz + 0.55), (0.75, 0.06, 0.1), STONE2)
PANEL = mat('panel', (0.22, 0.02, 0.04), 0.55, **{'Sheen Weight': 0.5})
pz0, pz1 = ZB + 0.5, ZBAL - 0.2
box((0, WALL_Y - 0.01, (pz0 + pz1) / 2), (5.0, 0.02, pz1 - pz0), PANEL)
for a_, b_ in (((-2.5, pz0), (2.5, pz0)), ((-2.5, pz1), (2.5, pz1)), ((-2.5, pz0), (-2.5, pz1)), ((2.5, pz0), (2.5, pz1))):
    rod((a_[0], WALL_Y - 0.03, a_[1]), (b_[0], WALL_Y - 0.03, b_[1]), 0.035, GOLD, 10)
# side wings
for sx in (-1, 1):
    box((sx * 7.0, WALL_Y + 1.2, ZB + (H - 2) / 2), (5.0, 2.0, H - 2), STONE)
    for zz in np.arange(ZB + 1.2, ZB + H - 2.6, 1.6):
        for xx in np.arange(5.2, 9.2, 1.2): box((sx * xx, WALL_Y + 0.19, zz), (0.5, 0.02, 0.9), WIN)
# cornice, balcony with balustrade
ZBAL = Z_TOP + 0.75
box((0, WALL_Y - 0.25, ZBAL - 0.08), (6.6, 1.2, 0.16), STONE2, 0.02)
for xx in np.linspace(-3.1, 3.1, 27): cyl((xx, WALL_Y - 0.8, ZBAL + 0.2), 0.04, 0.4, STONE, 10)
box((0, WALL_Y - 0.8, ZBAL + 0.42), (6.4, 0.12, 0.06), STONE2)
box((0, WALL_Y + 0.2, ZB + H + 0.15), (9.4, 0.7, 0.3), STONE2, 0.03)
# dome + drum
ZD = ZB + H + 0.3
cyl((0, WALL_Y + 1.6, ZD + 0.6), 1.6, 1.2, STONE, 48)
for k in range(16):
    a = 2 * math.pi * k / 16
    if math.sin(a) < 0.2: box((1.62 * math.cos(a), WALL_Y + 1.6 + 1.62 * math.sin(a), ZD + 0.6), (0.18, 0.18, 0.7), WIN).rotation_euler = (0, 0, a)
bpy.ops.mesh.primitive_uv_sphere_add(radius=1.65, segments=48, ring_count=24, location=(0, WALL_Y + 1.6, ZD + 1.2))
dm = bpy.context.object; dm.scale = (1, 1, 0.85); dm.data.materials.append(mat('dome', (0.25, 0.45, 0.42), 0.35, **{'Metallic': 0.6}))
for p in dm.data.polygons: p.use_smooth = True
ZLANT = ZD + 1.2 + 1.65 * 0.85
cyl((0, WALL_Y + 1.6, ZLANT + 0.2), 0.42, 0.4, STONE2, 32)
# crown (rises out of the lantern at the end)
CROWN = link(bpy.data.objects.new('crown', None)); CROWN.location = (0, WALL_Y + 1.6, ZLANT + 0.25)
CRG = mat('crowng', (1.0, 0.75, 0.3), 0.12, **{'Metallic': 1.0, 'Emission Color': (1, 0.7, 0.3, 1), 'Emission Strength': 0.3})
bpy.ops.mesh.primitive_torus_add(major_radius=0.3, minor_radius=0.05, location=(0, 0, 0.0)); o = bpy.context.object; o.data.materials.append(CRG); o.parent = CROWN
cyl((0, 0, 0.12), 0.29, 0.22, CRG, 32).parent = CROWN
JEW = [mat('ruby', (0.8, 0.02, 0.05), 0.05, **{'Transmission Weight': 0.6}), mat('sapph', (0.05, 0.1, 0.8), 0.05, **{'Transmission Weight': 0.6})]
for k in range(8):
    a = 2 * math.pi * k / 8
    bpy.ops.mesh.primitive_cone_add(radius1=0.06, radius2=0.0, depth=0.22, vertices=8, location=(0.29 * math.cos(a), 0.29 * math.sin(a), 0.33))
    c = bpy.context.object; c.data.materials.append(CRG); c.parent = CROWN
    sph((0.29 * math.cos(a), 0.29 * math.sin(a), 0.46), 0.035, CRG, 10).parent = CROWN
    sph((0.31 * math.cos(a), 0.31 * math.sin(a), 0.12), 0.04, JEW[k % 2], 12).parent = CROWN
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, location=(0, 0, 0.3)); o = bpy.context.object; o.scale = (1, 1, 0.8); o.data.materials.append(RED); o.parent = CROWN
sph((0, 0, 0.55), 0.06, CRG, 12).parent = CROWN
rod((0, 0, 0.6), (0, 0, 0.78), 0.015, CRG).parent = CROWN; rod((-0.07, 0, 0.71), (0.07, 0, 0.71), 0.015, CRG).parent = CROWN
rise = ss((T - T_CUP - 0.25) / 1.8)
key_series(CROWN, 'location', 2, ZLANT + 0.25 - 0.55 + (0.55 + 0.75) * rise)
CROWN.rotation_mode = 'XYZ'; key_series(CROWN, 'rotation_euler', 2, np.where(T > T_CUP, (T - T_CUP) * 0.5, 0))
cl_ = bpy.data.lights.new('crownl', 'POINT'); cl_.color = (1, 0.8, 0.45); cl_.shadow_soft_size = 0.4
clo = link(bpy.data.objects.new('crownl', cl_)); clo.location = (0, WALL_Y + 0.6, ZLANT + 1.2)
for f in range(NF):
    if f in (0, NF - 1) or rise[f] > 0 or (f > 0 and rise[f - 1] > 0): cl_.energy = 5 + 300 * rise[f]; cl_.keyframe_insert('energy', frame=f)
# cable: the cup's lever tail rises and draws a cable (floor pulley -> up the pilaster -> gantry over the lantern) that lifts the crown
phi = END_DIR * 0.28 * dip
PIV = np.array([CUP_X - END_DIR * 0.45, 0.0, CUP_Z - 0.05])
TAIL = np.stack([PIV[0] - END_DIR * 0.5 * np.cos(phi), np.zeros(NF), PIV[2] + 0.5 * np.sin(phi) * END_DIR], 1)
FP = np.array([TAIL[0, 0], 0.05, ZB + 0.12]); PX = 2.45 * (1 if TAIL[0, 0] > 0 else -1)
moving_rod('cab0', TAIL, np.tile(FP, (NF, 1)), 0.008, GOLD)
GZ = ZLANT + 1.75
for a_, b_ in ((FP, (PX, 0.05, ZB + 0.12)), ((PX, 0.05, ZB + 0.12), (PX, 0.05, ZB + H + 0.55)), ((PX, 0.05, ZB + H + 0.55), (PX * 0.3, WALL_Y + 1.6, GZ)), ((PX * 0.3, WALL_Y + 1.6, GZ), (0, WALL_Y + 1.6, GZ))):
    rod(a_, b_, 0.008, GOLD, 6)
for c_ in (FP, (PX, 0.05, ZB + 0.12), (PX, 0.05, ZB + H + 0.55), (0, WALL_Y + 1.6, GZ)): cyl(c_, 0.07, 0.03, BRASS, 20, rot=(math.pi / 2, 0, 0))
for sx_ in (-1, 1): rod((sx_ * 0.6, WALL_Y + 1.6, ZLANT - 0.12), (sx_ * 0.25, WALL_Y + 1.6, GZ + 0.05), 0.03, BRASS, 10)   # gantry legs
rod((-0.3, WALL_Y + 1.6, GZ + 0.06), (0.3, WALL_Y + 1.6, GZ + 0.06), 0.03, BRASS, 10)
CZ = ZLANT + 0.25 - 0.55 + (0.55 + 0.75) * rise
moving_rod('cab1', np.tile([0, WALL_Y + 1.6, GZ - 0.04], (NF, 1)), np.stack([np.zeros(NF), np.full(NF, WALL_Y + 1.6), CZ + 0.78], 1), 0.008, GOLD)
# ---------------- banners (unroll for the grand reprise) ----------------
TG0 = E['grand'][0]
for sx in (-1, 1):
    bx = sx * 2.68; btop = ZBAL - 0.25
    rollr = cyl((bx, WALL_Y - 0.08, btop), 0.05, 0.5, GOLD, 16, rot=(0, math.pi / 2, 0))
    bpy.ops.mesh.primitive_plane_add(size=1, location=(bx, WALL_Y - 0.1, btop))
    bn = bpy.context.object; bn.rotation_euler = (math.pi / 2, 0, 0)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    for v in bn.data.vertices: v.co.z -= 0.5
    bn.data.materials.append(mat(f'ban{sx}', (0.55, 0.03, 0.06), 0.6))
    bn.scale = (0.42, 1, 1)
    un = ss((T - TG0) / 1.4); key_series(bn, 'scale', 2, 0.02 + 3.0 * un)
    rollr.rotation_mode = 'XYZ'; key_series(rollr, 'rotation_euler', 0, un * 3.0 / 0.05)
    # gold crest on the banner (follows its bottom)
    em_ = sph((bx, WALL_Y - 0.12, btop - 0.4), 0.12, GOLD, 16); em_.scale = (1, 0.3, 1)
    key_series(em_, 'location', 2, btop - 0.02 - (3.0 * un) * 0.55)
    key_series(em_, 'scale', 0, np.clip(un * 3, 0, 1)); key_series(em_, 'scale', 2, np.clip(un * 3, 0, 1))

# ---------------- automaton trumpeters on the balcony ----------------
blow = np.zeros(NF)
for t0, d in E['blow']:
    blow = np.maximum(blow, ss((T - (t0 - 0.35)) / 0.2) * (1 - ss((T - (t0 + d + 0.05)) / 0.25)))
g0, g1 = E['grand']
blow = np.maximum(blow, ss((T - g0) / 0.35) * (1 - ss((T - g1) / 0.5)))
bob = np.zeros(NF)
for tn in TL:
    if tn >= g0: bob += np.where(T >= tn - 0.05, np.exp(-np.clip(T - tn, 0, None) / 0.15), 0) * 0.06
ANG = 1.05 - 1.2 * blow - bob * blow         # rotation about x of the arm (rest: trumpet pointing down-forward)
TRUMP = GOLD
for k, xx in enumerate(np.linspace(-2.2, 2.2, 5)):
    yb = WALL_Y - 0.45; zf = ZBAL
    for dx in (-0.07, 0.07): cyl((xx + dx, yb, zf + 0.25), 0.05, 0.5, NAVY, 12)
    cyl((xx, yb, zf + 0.72), 0.15, 0.48, RED, 20)
    cyl((xx, yb, zf + 0.5), 0.155, 0.05, GOLD, 20)
    sph((xx, yb, zf + 1.07), 0.1, SKIN, 16)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.14, location=(xx, yb, zf + 1.28)); hat = bpy.context.object; hat.scale = (0.9, 0.9, 1.6); hat.data.materials.append(BLACK)
    for dx in (-0.17, 0.17): sph((xx + dx, yb, zf + 0.93), 0.05, GOLD, 10)
    # right arm with trumpet (pivot at shoulder)
    sh = link(bpy.data.objects.new(f'sh{k}', None)); sh.location = (xx + 0.2, yb - 0.02, zf + 0.9); sh.rotation_mode = 'XYZ'
    rod((0, 0, 0), (0, -0.28, 0), 0.04, RED, 10).parent = sh
    rod((0, -0.28, 0), (-0.12, -0.32, 0.05), 0.012, TRUMP, 8).parent = sh
    rod((-0.12, -0.32, 0.05), (-0.12, -0.62, 0.05), 0.013, TRUMP, 8).parent = sh
    bpy.ops.mesh.primitive_cone_add(radius1=0.015, radius2=0.075, depth=0.16, vertices=20, location=(-0.12, -0.7, 0.05), rotation=(math.pi / 2, 0, 0))
    bl_ = bpy.context.object; bl_.data.materials.append(TRUMP); bl_.parent = sh
    # flag hanging from the trumpet
    box((-0.12, -0.5, -0.05), (0.004, 0.18, 0.16), RED).parent = sh
    rod((0, 0, 0), (0, 0.16, 0), 0.015, BRASS, 6).parent = sh        # lever tail behind the shoulder
    key_series(sh, 'rotation_euler', 0, ANG)
    # pneumatic piston below the lever tail; a link rod joins them
    tail = np.stack([np.full(NF, xx + 0.2), yb - 0.02 + 0.16 * np.cos(ANG), zf + 0.9 + 0.16 * np.sin(ANG)], 1)
    PY = yb - 0.02 + 0.13
    cyl((xx + 0.2, PY, zf + 0.33), 0.035, 0.3, BRASS, 16)
    ptop = np.stack([np.full(NF, xx + 0.2), np.full(NF, PY), tail[:, 2] - 0.28], 1)
    prd = cyl((0, 0, 0), 0.012, 0.3, GOLD, 8)
    key_series(prd, 'location', 0, ptop[:, 0]); key_series(prd, 'location', 1, ptop[:, 1]); key_series(prd, 'location', 2, ptop[:, 2] - 0.15)
    moving_rod(f'lnk{k}', ptop, tail, 0.01, GOLD)
# air line along the balcony
rod((-2.5, WALL_Y - 0.32, ZBAL + 0.2), (2.5, WALL_Y - 0.32, ZBAL + 0.2), 0.02, BRASS, 8)

# ---------------- the marble ----------------
BALL = sph((0, 0, 0), RB, mat('ballm', (1.0, 0.78, 0.35), 0.08, **{'Metallic': 1.0, 'Emission Color': (1, 0.7, 0.3, 1), 'Emission Strength': 2.0}), 32)
for i in range(3): key_series(BALL, 'location', i, BP[:, i])
spin = np.cumsum(np.r_[0, np.linalg.norm(np.diff(BP, axis=0), axis=1)]) / RB
BALL.rotation_mode = 'XYZ'; key_series(BALL, 'rotation_euler', 1, np.cumsum(np.r_[0, np.diff(BP[:, 0])]) / RB)
ld = bpy.data.lights.new('bl', 'POINT'); ld.energy = 6; ld.color = (1.0, 0.75, 0.4); ld.shadow_soft_size = 0.05
lo = link(bpy.data.objects.new('bl', ld)); lo.parent = BALL; lo.location = (0, -0.25, 0.1)

# ---------------- fireworks ----------------
FW = fadeemis('fw', 60.0)
bpy.ops.mesh.primitive_ico_sphere_add(radius=0.12, subdivisions=1, location=(0, 0, -50)); proto = bpy.context.object; proto.data.materials.append(FW)
COLS = [(1.0, 0.75, 0.3), (1.0, 0.2, 0.2), (0.9, 0.9, 1.0), (0.3, 0.6, 1.0), (1.0, 0.5, 0.9)]
bursts = [(FINAL + 0.05, -3.2, 11.5), (FINAL + 0.5, 3.4, 12.5), (FINAL + 1.0, -0.5, 14.0), (FINAL + 1.5, -4.6, 13.2), (FINAL + 2.0, 4.6, 11.0), (FINAL + 2.6, 1.8, 13.6), (FINAL + 3.1, -2.2, 14.6)]
for bi, (tb_, bx, bz) in enumerate(bursts):
    col = COLS[bi % len(COLS)]; c = np.array([bx, 7.0, ZB + bz])
    # rocket trail
    rk = proto.copy(); rk.data = proto.data; link(rk); rk.color = (1, 0.8, 0.5, 0)
    tau = (T - (tb_ - 0.9)) / 0.9
    RP = np.stack([np.full(NF, c[0]), np.full(NF, c[1]), ZB + (c[2] - ZB) * (1 - (1 - np.clip(tau, 0, 1)) ** 2)], 1)
    for i in range(3): key_series(rk, 'location', i, RP[:, i])
    key_series(rk, 'color', 3, np.where((tau >= 0) & (tau <= 1), 0.6, 0.0), 1e-3)
    for j in range(64):
        d = rng.normal(0, 1, 3); d /= np.linalg.norm(d); sp_ = rng.uniform(3.0, 3.8)
        o = proto.copy(); o.data = proto.data; link(o); o.color = (*col, 0)
        tt = np.clip(T - tb_, 0, None)
        P = c + d[None, :] * sp_ * (1 - np.exp(-tt[:, None] * 1.6)) / 1.6 * 1.6 - np.array([0, 0, 0.5]) * tt[:, None] ** 2
        for i in range(3): key_series(o, 'location', i, P[:, i], 1e-4)
        al = np.where(T >= tb_, np.exp(-tt / 0.9) * (0.7 + 0.3 * np.sin(tt * 40 + j)), 0.0)
        key_series(o, 'color', 3, al, 2e-3)
    fl = bpy.data.lights.new(f'fwl{bi}', 'POINT'); fl.color = col; fl.shadow_soft_size = 1.0
    flo = link(bpy.data.objects.new(f'fwl{bi}', fl)); flo.location = tuple(c)
    for f in range(NF):
        e = 8000 * math.exp(-max(0, T[f] - tb_) / 0.5) if T[f] >= tb_ else 0
        if f in (0, NF - 1) or abs(T[f] - tb_) < 3: fl.energy = e; fl.keyframe_insert('energy', frame=f)

# ---------------- ground, world, lights ----------------
bpy.ops.mesh.primitive_plane_add(size=80, location=(0, 0, ZB)); g = bpy.context.object; g.data.materials.append(mat('court', (0.32, 0.3, 0.28), 0.55))
for sx in (-1, 1):   # flame torches on the courtyard
    cyl((sx * 2.5, -1.2, ZB + 0.6), 0.05, 1.2, BRASS, 12)
    fo = sph((sx * 2.5, -1.2, ZB + 1.3), 0.1, emis(f'fl{sx}', (1.0, 0.5, 0.15), 18.0), 12); fo.scale = (1, 1, 1.6)
    tl_ = bpy.data.lights.new(f'tl{sx}', 'POINT'); tl_.energy = 60; tl_.color = (1, 0.55, 0.2)
    link(bpy.data.objects.new(f'tl{sx}', tl_)).location = (sx * 2.5, -1.3, ZB + 1.6)
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True; nt = w.node_tree
bg = nt.nodes['Background']; bg.inputs['Strength'].default_value = 1.0
grad = nt.nodes.new('ShaderNodeTexGradient'); tc = nt.nodes.new('ShaderNodeTexCoord'); mp = nt.nodes.new('ShaderNodeMapping')
mp.inputs['Rotation'].default_value = (0, -math.pi / 2, 0)
ramp = nt.nodes.new('ShaderNodeValToRGB'); ramp.color_ramp.elements[0].color = (0.01, 0.012, 0.03, 1); ramp.color_ramp.elements[1].color = (0.03, 0.05, 0.13, 1)
nt.links.new(tc.outputs['Generated'], mp.inputs['Vector']); nt.links.new(mp.outputs['Vector'], grad.inputs['Vector'])
nt.links.new(grad.outputs['Fac'], ramp.inputs['Fac']); nt.links.new(ramp.outputs['Color'], bg.inputs['Color'])
STARM = emis('star', (0.9, 0.92, 1.0), 6.0)
for i in range(220):
    sph((rng.uniform(-22, 22), rng.uniform(25, 30), ZB + rng.uniform(6, 40)), rng.uniform(0.02, 0.06), STARM, 6)
for nm, loc, e, col, sz, tgt in (('key', (-5, -9, Z_TOP + 1.0), 420, (1.0, 0.85, 0.65), 5, (0, 0.6, Z_TOP - 2)),
                                ('fill', (6, -8, ZB + 3), 220, (0.55, 0.65, 1.0), 6, (0, 0.6, ZB + 3)),
                                ('top', (0, -3, ZB + H + 4), 400, (1.0, 0.8, 0.6), 4, (0, 0.6, ZB + H))):
    l_ = bpy.data.lights.new(nm, 'AREA'); l_.energy = e; l_.color = col; l_.size = sz
    o = link(bpy.data.objects.new(nm, l_)); o.location = loc
    tg_ = link(bpy.data.objects.new(nm + 't', None)); tg_.location = tgt
    c = o.constraints.new('TRACK_TO'); c.target = tg_; c.track_axis = 'TRACK_NEGATIVE_Z'; c.up_axis = 'UP_Y'

# ---------------- camera: wide, gentle; follows the active rows, pulls back for the crown ----------------
camd = bpy.data.cameras.new('cam'); camd.lens = 32; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
tr = cam.constraints.new('TRACK_TO'); tr.target = tgt; tr.track_axis = 'TRACK_NEGATIVE_Z'; tr.up_axis = 'UP_Y'
DIST = float(os.environ.get('DIST', 13.2))
zb = smooth(BP[:, 2], 1.6)
zmid = np.clip(zb + 0.9, ZB + 3.2, ZBAL - 0.6)
intro = 1 - ss((T - T_REL + 0.5) / 2.5)                       # start framed on the trumpeters
zt = zmid * (1 - intro) + (ZBAL + 0.2) * intro
fin = ss((T - (T_CUP - 0.3)) / 2.6)
ztf = zt * (1 - fin) + (ZB + H * 0.78) * fin
dist = DIST + 6.5 * fin
drift = 0.6 * np.sin(T / 22.0)
CPos = np.stack([drift + 0.3, -dist, ztf + 0.6 + 0.6 * fin], 1); TPos = np.stack([drift * 0.2, np.full(NF, 0.6), ztf], 1)
for i in range(3): key_series(cam, 'location', i, CPos[:, i]); key_series(tgt, 'location', i, TPos[:, i])

print('objects', len(bpy.data.objects), 'NF', NF, 'treads', len(TREADS), 'pipes', len(PIPES))
if MODE == 'test':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS)); sc.frame_set(f)
        sc.render.filepath = f'{HERE}/test_{f:04d}.png'
        t1 = time.time(); bpy.ops.render.render(write_still=True); print('frame', f, 'sec', round(time.time() - t1, 1))
elif MODE == 'range':
    sc.frame_start, sc.frame_end = int(args[1]), int(args[2])
    sc.render.image_settings.file_format = 'PNG'; sc.render.filepath = f'{HERE}/frames/f_'
    bpy.ops.render.render(animation=True)
