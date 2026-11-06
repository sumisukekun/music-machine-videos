"""Csikos Post - a marble horse race on a sports-day stadium.
Four marbles race two laps of a red tartan oval. Every melody note is a xylophone hurdle that pops up out of the
track in one lane exactly where that lane's marble is at that instant: the marble hops over it, grazes the bar and
the bar rings. The intro is played by the marbles bouncing on the bars inside the starting stalls, the gate opens
with a hammer stroke on the golden gate bar, a band machine in the infield (bass drum / snare / cymbal) plays the
accompaniment strokes, and the finish tape snaps on the final chord."""
import os, sys, json, math, time
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix, Quaternion

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.5) * FPS); T = np.arange(NF) / FPS
SECS = dict((s, t) for s, t in E['secs']); TA = SECS['A']
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
sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.25
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


def glowmat(name, gain=10.0, base=0.3, rough=0.25):
    """object colour = base colour; object alpha drives emission"""
    m = mat(name, (1, 1, 1), rough, **{'Coat Weight': 0.5}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    oi = nt.nodes.new('ShaderNodeObjectInfo')
    nt.links.new(oi.outputs['Color'], b.inputs['Base Color']); nt.links.new(oi.outputs['Color'], b.inputs['Emission Color'])
    mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY_ADD'; mul.inputs[1].default_value = gain; mul.inputs[2].default_value = base
    nt.links.new(oi.outputs['Alpha'], mul.inputs[0]); nt.links.new(mul.outputs[0], b.inputs['Emission Strength'])
    return m


def key_series(obj, path, idx, vals, eps=1e-5):
    n = len(vals)
    for f in range(n):
        v = vals[f]
        if 0 < f < n - 1 and abs(v - vals[f - 1]) < eps and abs(v - vals[f + 1]) < eps: continue
        getattr(obj, path)[idx] = float(v); obj.keyframe_insert(path, index=idx, frame=f)


def fcurves(obj):
    ad = obj.animation_data; act = ad.action
    if hasattr(act, 'fcurves'): return act.fcurves
    return act.layers[0].strips[0].channelbag(ad.action_slot).fcurves


def key_pts(obj, path, idx, pts, interp='BEZIER'):
    """sparse keys [(time_sec, value)]"""
    for t, v in pts:
        getattr(obj, path)[idx] = float(v); obj.keyframe_insert(path, index=idx, frame=t * FPS)
    if obj.animation_data and obj.animation_data.action:
        for fc in fcurves(obj):
            if fc.data_path == path and fc.array_index == idx:
                for kp in fc.keyframe_points: kp.interpolation = interp
    return obj


def flash_pts(hits, peak=1.0, decay=0.45, base=0.0):
    pts = [(0, base)]
    for h in hits:
        pts += [(h - 1.5 / FPS, base), (h, peak), (h + decay * 0.35, base + (peak - base) * 0.35), (h + decay, base)]
    pts.sort(); out = []
    for t, v in pts:
        if out and t <= out[-1][0] + 1e-4: out[-1] = (out[-1][0], max(out[-1][1], v)); continue
        out.append((t, v))
    return out


def rod(p0, p1, r, mt, verts=10):
    p0, p1 = Vector(p0), Vector(p1); d = p1 - p0
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=d.length, vertices=verts, location=(p0 + p1) / 2)
    o = bpy.context.object; q = Vector((0, 0, 1)).rotation_difference(d.normalized())
    o.rotation_mode = 'QUATERNION'; o.rotation_quaternion = q
    o.matrix_world = Matrix.Translation((p0 + p1) / 2) @ q.to_matrix().to_4x4()
    o.data.materials.append(mt); return o


def box(loc, size, mt, name='box'):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc); o = bpy.context.object; o.scale = size; o.name = name
    o.data.materials.append(mt); return o


def smooth01(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


# ---------------- stadium geometry ----------------
SH = 3.0                    # half length of the straights
RI = 1.6; LW = 0.48; NL = 4; RO = RI + NL * LW
RL = [RI + LW * (i + 0.5) for i in range(NL)]
BR = 0.14                   # marble radius
HB = 0.19                   # hurdle bar top height
S0 = SH - 1.0               # start / finish line (y = -1 on the right-hand straight)


def loop_len(r): return 4 * SH + 2 * math.pi * r


def pos(r, s):
    """point + unit tangent on the stadium loop of radius r, arc length s from (r,-SH) going +y"""
    L = loop_len(r); s = s % L
    if s < 2 * SH: return np.array([r, -SH + s]), np.array([0.0, 1.0])
    s -= 2 * SH
    if s < math.pi * r:
        a = s / r; return np.array([r * math.cos(a), SH + r * math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    s -= math.pi * r
    if s < 2 * SH: return np.array([-r, SH - s]), np.array([0.0, -1.0])
    s -= 2 * SH; a = math.pi + s / r
    return np.array([r * math.cos(a), -SH + r * math.sin(a)]), np.array([-math.sin(a), math.cos(a)])


def ring_mesh(name, r1, r2, z, mt, n=400):
    bm = bmesh.new(); vs = []
    for k in range(n):
        f = k / n
        p1, _ = pos(r1, f * loop_len(r1)); p2, _ = pos(r2, f * loop_len(r2))
        vs.append((bm.verts.new((p1[0], p1[1], z)), bm.verts.new((p2[0], p2[1], z))))
    for k in range(n):
        a, b = vs[k], vs[(k + 1) % n]; bm.faces.new((a[0], b[0], b[1], a[1]))
    me = bpy.data.meshes.new(name); bm.to_mesh(me); o = link(bpy.data.objects.new(name, me)); o.data.materials.append(mt)
    return o


def fill_mesh(name, r, z, mt, n=300):
    bm = bmesh.new(); c = bm.verts.new((0, 0, z)); vs = []
    for k in range(n):
        p, _ = pos(r, k / n * loop_len(r)); vs.append(bm.verts.new((p[0], p[1], z)))
    for k in range(n): bm.faces.new((c, vs[k], vs[(k + 1) % n]))
    me = bpy.data.meshes.new(name); bm.to_mesh(me); o = link(bpy.data.objects.new(name, me)); o.data.materials.append(mt)
    return o


TARTAN = mat('tartan', (0.62, 0.13, 0.08), 0.75)
WHITE = mat('white', (0.9, 0.9, 0.9), 0.5)
GRASS = mat('grass', (0.12, 0.35, 0.1), 0.85)
GRASS2 = mat('grass2', (0.16, 0.42, 0.12), 0.85)
ring_mesh('track', RI - 0.06, RO + 0.06, 0.0, TARTAN)
for k in range(NL + 1):
    r = RI + k * LW; ring_mesh(f'line{k}', r - 0.012, r + 0.012, 0.002, WHITE)
fill_mesh('infield', RI - 0.06, 0.0, GRASS2)
bpy.ops.mesh.primitive_plane_add(size=80, location=(0, 0, -0.003)); bpy.context.object.data.materials.append(GRASS)
# mowing stripes in the infield
for k in range(-3, 4):
    if k % 2: box((0, k * 0.9, 0.0015), (2 * RI - 0.3, 0.45, 0.001), GRASS)
box(((RI + RO) / 2, -1.0, 0.004), (RO - RI, 0.07, 0.002), WHITE, 'finishline')
# kerb
ring_mesh('kerb', RI - 0.12, RI - 0.06, 0.02, WHITE)

# ---------------- race timing ----------------
L_LANE = [loop_len(r) for r in RL]
WIN = 1                                    # the blue marble wins (lane 2)
FINISH_FRAC = [0.982, 1.0, 0.968, 0.952]  # fraction of 2 laps done at the final chord
NAMES = ['red', 'blue', 'yellow', 'green']
BCOL = [(0.85, 0.06, 0.05), (0.05, 0.25, 0.9), (0.95, 0.75, 0.05), (0.05, 0.6, 0.15)]
TAU = 0.45
def prog(t):
    x = np.maximum(t - TA, 0.0); return x - TAU * (1 - np.exp(-x / TAU))
P_F = prog(FINAL)
WOB = [(0.010, 2.0, 0.3), (0.012, 3.0, 2.0), (0.011, 2.5, 4.1), (0.009, 3.5, 1.2)]
S_REST = S0 - 0.24
def lane_s(i, t):
    t = np.asarray(t, float)
    p = (t - TA) / (FINAL - TA)
    g = prog(np.minimum(t, FINAL)) / P_F * FINISH_FRAC[i]
    a, n, ph = WOB[i]; g = g + a * np.sin(math.pi * np.clip(p, 0, 1)) * np.sin(2 * math.pi * n * np.clip(p, 0, 1) + ph) * np.clip(p, 0, 1) ** 0.5
    s = S_REST + (2 * L_LANE[i] + (S0 - S_REST)) * g
    # after the finish: coast and slow down
    v_end = (2 * L_LANE[i] + (S0 - S_REST)) * FINISH_FRAC[i] / P_F * (1 - math.exp(-(FINAL - TA) / TAU))
    post = np.maximum(t - FINAL, 0)
    return np.where(t > FINAL, s + v_end * 1.3 * (1 - np.exp(-post / 1.3)), s)

# ---------------- note -> lane assignment ----------------
intro_notes = [l for l in LEAD if l[0] < TA - 1e-3]
gate_note = [l for l in LEAD if abs(l[0] - TA) < 1e-3]
race = [l for l in LEAD if l[0] > TA + 1e-3]
last = [-9.0] * NL; HITS = {i: [] for i in range(NL)}
for k, (t, p, v) in enumerate(race):
    if abs(t - FINAL) < 1e-3: lane = WIN
    else:
        cand = [i for i in range(NL) if not (i == WIN and t > FINAL - 0.55)]
        lane = max(cand, key=lambda i: (t - last[i], -abs(i - 1.5)))
    HITS[lane].append((t, p, v)); last[lane] = t
INTRO = {i: [] for i in range(NL)}
for k, (t, p, v) in enumerate(intro_notes): INTRO[[1, 2, 0, 3][k % 4]].append((t, p, v))
print('hits per lane', [len(HITS[i]) for i in range(NL)], 'min gap', min(min(np.diff([h[0] for h in HITS[i]])) for i in range(NL)))

PCOL = [(0.95, 0.15, 0.15), (0.95, 0.4, 0.1), (0.95, 0.7, 0.1), (0.75, 0.9, 0.1), (0.3, 0.85, 0.2), (0.1, 0.8, 0.55),
        (0.1, 0.7, 0.9), (0.15, 0.4, 0.95), (0.4, 0.25, 0.95), (0.65, 0.2, 0.9), (0.9, 0.2, 0.75), (0.95, 0.2, 0.45)]
def pcol(p): return PCOL[(p * 7) % 12]   # circle of fifths colours: neighbouring keys look alike

# ---------------- marbles: positions per frame ----------------
STALL_TOP = 0.06
BP = np.zeros((NL, NF, 3)); TAN = np.zeros((NL, NF, 2))
for i in range(NL):
    s = lane_s(i, T); xy = np.zeros((NF, 2)); tg = np.zeros((NF, 2))
    for f in range(NF): xy[f], tg[f] = pos(RL[i], s[f])
    z = np.full(NF, BR)
    # inside the stall: resting on the stall bar, bouncing for the intro notes
    contacts = [h[0] for h in INTRO[i]]
    zb = np.full(NF, STALL_TOP)
    if contacts:
        pts = [contacts[0] - 0.34] + contacts + [TA]
        for a, b in zip(pts[:-1], pts[1:]):
            g = b - a; H = min(0.26, 1.6 * g * g + 0.02)
            m = (T >= a) & (T < b); u = (T[m] - a) / g
            zb[m] = STALL_TOP + 4 * H * u * (1 - u)
    else:
        m = (T >= TA - 0.34) & (T < TA); u = (T[m] - (TA - 0.34)) / 0.34; zb[m] = STALL_TOP + 4 * 0.08 * u * (1 - u)
    # leaving the stall: drop off the stall bar onto the track
    leave = np.clip((T - TA) / 0.22, 0, 1)
    z_stall = BR + zb * (1 - leave) ** 2
    z = np.where(T < TA + 0.22, z_stall, BR)
    # hurdle hops (apex = bar contact at the note time)
    ht = [h[0] for h in HITS[i]]
    for k, h in enumerate(ht):
        gp = h - (ht[k - 1] if k else TA + 0.15); gn = (ht[k + 1] - h) if k + 1 < len(ht) else 9
        D = min(0.44, 2 * gp * 0.98, 2 * gn * 0.98)
        m = np.abs(T - h) < D / 2; u = (T[m] - h) / (D / 2)
        z[m] = np.maximum(z[m], BR + HB * (1 - u * u))
    BP[i, :, 0] = xy[:, 0]; BP[i, :, 1] = xy[:, 1]; BP[i, :, 2] = z; TAN[i] = tg

BALLS = []
for i in range(NL):
    m = bpy.data.materials.new(f'ball{i}'); m.use_nodes = True; nt = m.node_tree; b = nt.nodes['Principled BSDF']
    b.inputs['Roughness'].default_value = 0.12; b.inputs['Coat Weight'].default_value = 1.0
    tc = nt.nodes.new('ShaderNodeTexCoord'); sep = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(tc.outputs['Object'], sep.inputs[0])
    ab = nt.nodes.new('ShaderNodeMath'); ab.operation = 'ABSOLUTE'; nt.links.new(sep.outputs['Z'], ab.inputs[0])
    lt = nt.nodes.new('ShaderNodeMath'); lt.operation = 'LESS_THAN'; lt.inputs[1].default_value = 0.035; nt.links.new(ab.outputs[0], lt.inputs[0])
    mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'; mix.inputs['A'].default_value = (*BCOL[i], 1); mix.inputs['B'].default_value = (0.95, 0.95, 0.95, 1)
    nt.links.new(lt.outputs[0], mix.inputs['Factor']); nt.links.new(mix.outputs['Result'], b.inputs['Base Color'])
    b.inputs['Emission Color'].default_value = (*BCOL[i], 1)
    oi = nt.nodes.new('ShaderNodeObjectInfo'); mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = 4.0
    nt.links.new(oi.outputs['Alpha'], mul.inputs[0]); nt.links.new(mul.outputs[0], b.inputs['Emission Strength'])
    bpy.ops.mesh.primitive_uv_sphere_add(radius=BR, segments=32, ring_count=16)
    o = bpy.context.object; o.name = f'marble_{NAMES[i]}'; o.data.materials.append(m)
    for p in o.data.polygons: p.use_smooth = True
    o.rotation_mode = 'QUATERNION'
    for c in range(3): key_series(o, 'location', c, BP[i, :, c])
    # rolling
    q = Quaternion(); Q = []
    for f in range(NF):
        if f:
            d = Vector((BP[i, f, 0] - BP[i, f - 1, 0], BP[i, f, 1] - BP[i, f - 1, 1], 0)); L = d.length
            if L > 1e-6:
                ax = Vector((0, 0, 1)).cross(d / L); q = Quaternion(ax, L / BR) @ q
        Q.append(q.copy())
    Qa = np.array([[x.w, x.x, x.y, x.z] for x in Q])
    for c in range(4): key_series(o, 'rotation_quaternion', c, Qa[:, c], 1e-4)
    allh = [h[0] for h in INTRO[i]] + [h[0] for h in HITS[i]] + ([TA] if True else [])
    o.color = (1, 1, 1, 0)
    for t, v in flash_pts(sorted(allh), 1.0, 0.35): o.color[3] = v; o.keyframe_insert('color', index=3, frame=t * FPS)
    BALLS.append(o)

# ---------------- pop-up xylophone hurdles ----------------
BARM = glowmat('bar', gain=9.0, base=0.5, rough=0.3)
STEEL = mat('steel', (0.75, 0.76, 0.8), 0.25, **{'Metallic': 1.0})
DARK = mat('dark', (0.04, 0.04, 0.05), 0.6)
def hurdle_mesh():
    bm = bmesh.new()
    def cube(cx, cy, cz, sx, sy, sz, mi):
        r = bmesh.ops.create_cube(bm, size=1.0)
        for v in r['verts']: v.co.x = v.co.x * sx + cx; v.co.y = v.co.y * sy + cy; v.co.z = v.co.z * sz + cz
        for fc in {f for v in r['verts'] for f in v.link_faces}: fc.material_index = mi
    cube(0, 0, HB - 0.025, 0.37, 0.065, 0.05, 0)
    for sx in (-0.17, 0.17): cube(sx, 0, (HB - 0.05 - 0.35) / 2, 0.024, 0.024, HB - 0.05 + 0.35, 1)
    me = bpy.data.meshes.new('hurdle'); bm.to_mesh(me); me.materials.append(BARM); me.materials.append(STEEL)
    return me
HME = hurdle_mesh()
DOWN = -(HB + 0.03)
NH = 0
for i in range(NL):
    for (t, p, v) in HITS[i]:
        s = float(lane_s(i, t)); xy, tg = pos(RL[i], s)
        o = link(bpy.data.objects.new(f'h{i}_{NH}', HME)); NH += 1
        o.rotation_euler = (0, 0, math.atan2(tg[1], tg[0]) - math.pi / 2)
        o.location = (xy[0], xy[1], DOWN)
        o.color = (*pcol(p), 0)
        up0 = t - 0.55; up1 = t - 0.22; dn0 = t + 0.28; dn1 = t + 0.6
        if abs(t - FINAL) < 1e-3: dn0, dn1 = t + 3.0, t + 3.6
        key_pts(o, 'location', 2, [(up0, DOWN), (up1, 0.0), (dn0, 0.0), (dn1, DOWN)])
        for tt, vv in flash_pts([t], 1.0 + 0.6 * (v > 100), 0.5): o.color[3] = vv; o.keyframe_insert('color', index=3, frame=tt * FPS)

# ---------------- starting gate (stalls, doors, golden gate bar + starter hammer) ----------------
GATE = link(bpy.data.objects.new('gate', None))
GREEN = mat('gatepaint', (0.1, 0.45, 0.25), 0.4)
GOLDM = glowmat('goldbar', gain=8.0, base=0.2, rough=0.2)
yg = -1.0; y_back = S_REST - SH - 0.2
parts = []
for k in range(NL + 1):
    x = RI + k * LW
    parts.append(box((x, (yg + y_back) / 2, 0.24), (0.035, yg - y_back, 0.48), GREEN))
parts.append(box(((RI + RO) / 2, y_back, 0.3), (RO - RI + 0.035, 0.035, 0.6), GREEN))
for x in (RI - 0.05, RO + 0.05): parts.append(box((x, yg, 0.4), (0.06, 0.06, 0.8), GREEN))
gb = box(((RI + RO) / 2, yg, 0.68), (RO - RI + 0.2, 0.08, 0.07), GOLDM, 'gatebar'); gb.color = (1.0, 0.75, 0.25, 0); parts.append(gb)
for tt, vv in flash_pts([TA], 1.4, 0.6): gb.color[3] = vv; gb.keyframe_insert('color', index=3, frame=tt * FPS)
# stall bars the marbles bounce on
SB = []
for i in range(NL):
    p_, _ = pos(RL[i], S_REST)
    sb = link(bpy.data.objects.new(f'stall{i}', HME)); sb.scale = (1, 1, 1)
    sb.location = (p_[0], p_[1], STALL_TOP - HB); sb.color = (*pcol(int(np.median([h[1] for h in INTRO[i]])) if INTRO[i] else 64), 0)
    for tt, vv in flash_pts([h[0] for h in INTRO[i]], 1.0, 0.45): sb.color[3] = vv; sb.keyframe_insert('color', index=3, frame=tt * FPS)
    parts.append(sb)
# doors (two leaves per stall, hinged at the dividers)
DOORS = []
for i in range(NL):
    for side in (0, 1):
        x_h = RI + i * LW + (0.02 if side == 0 else LW - 0.02)
        piv = link(bpy.data.objects.new(f'door{i}{side}', None)); piv.location = (x_h, yg - 0.02, 0); parts.append(piv)
        d = box(((LW / 2 - 0.03) / 2 * (1 if side == 0 else -1), 0, 0.22), (LW / 2 - 0.035, 0.025, 0.4), WHITE); d.parent = piv
        piv.rotation_mode = 'XYZ'
        sgn = 1 if side == 0 else -1
        key_pts(piv, 'rotation_euler', 2, [(TA - 0.02, 0), (TA + 0.18, sgn * 1.62), (TA + 0.32, sgn * 1.52)])
        DOORS.append(piv)
# starter hammer on the outer post
HP = link(bpy.data.objects.new('hammerpiv', None)); HP.location = (RO + 0.32, yg, 1.05); parts.append(HP)
arm = rod((0, 0, 0), (-0.36, 0, -0.3), 0.022, STEEL); arm.parent = HP
hd = box((-0.4, 0, -0.33), (0.13, 0.09, 0.09), mat('hammerhead', (0.85, 0.1, 0.08), 0.3)); hd.parent = HP
HP.rotation_mode = 'XYZ'
key_pts(HP, 'rotation_euler', 1, [(0, 0.45), (TA - 0.6, 0.45), (TA - 0.25, 0.8), (TA, 0.06), (TA + 0.12, 0.2), (TA + 0.5, 0.45)])
parts.append(rod((RO + 0.32, yg, 0), (RO + 0.32, yg, 1.05), 0.03, GREEN))
for o in parts:
    if o.parent is None: o.parent = GATE
# after the start the whole gate rolls sideways off the track on two rails
for y in (yg - 0.1, y_back + 0.1):
    rod((RO + 0.1, y, 0.01), (RO + 3.2, y, 0.01), 0.018, STEEL)
key_pts(GATE, 'location', 0, [(TA + 1.4, 0.0), (TA + 3.6, 2.9)])
for d in DOORS:   # doors fold back flat while rolling away
    pass

# ---------------- finish tape ----------------
lap1 = []
for i in range(NL):
    s = lane_s(i, T); f = int(np.argmax(s >= S0 + L_LANE[i])); lap1.append(T[f])
T_TAPE = max(lap1) + 1.2
FINAL_LAP_T = lap1[WIN]
TAPE_Z = HB + 2 * BR
posts = []
for x in (RI - 0.16, RO + 0.16):
    po = rod((x, -1.0, -1.0), (x, -1.0, TAPE_Z + 0.12), 0.03, WHITE)
    key_pts(po, 'location', 2, [(T_TAPE - 0.01, po.location.z - 0.6), (T_TAPE + 0.8, po.location.z)])
    posts.append(po)
TAPEM = mat('tape', (0.95, 0.95, 0.95), 0.5)
halves = []
for side, x in ((0, RI - 0.16), (1, RO + 0.16)):
    piv = link(bpy.data.objects.new(f'tapepiv{side}', None)); piv.location = (x, -1.0, TAPE_Z)
    w = (RO - RI + 0.32) / 2
    t_ = box(((w / 2) * (1 if side == 0 else -1), 0, 0), (w, 0.012, 0.05), TAPEM); t_.parent = piv
    stripe = box(((w / 2) * (1 if side == 0 else -1), -0.007, 0), (w, 0.002, 0.016), mat(f'tapered{side}', (0.85, 0.05, 0.05), 0.5)); stripe.parent = piv
    piv.rotation_mode = 'XYZ'
    key_pts(piv, 'location', 2, [(T_TAPE - 0.01, TAPE_Z - 0.6), (T_TAPE + 0.8, TAPE_Z)])
    sg = -1 if side == 0 else 1
    key_pts(piv, 'rotation_euler', 2, [(FINAL, 0.0), (FINAL + 0.35, sg * 1.45), (FINAL + 0.6, sg * 1.62), (FINAL + 0.9, sg * 1.55)])
    key_pts(piv, 'rotation_euler', 0, [(FINAL, 0.0), (FINAL + 0.5, -0.5), (FINAL + 1.2, -1.2)])
    halves.append(piv)

# ---------------- infield band machine: bass drum, snare, cymbal ----------------
PERC = E['perc']
bd_t = [p[0] for p in PERC if p[1] in ('bd', 'timp')]
sn_t = [p[0] for p in PERC if p[1] == 'sn']
roll = [p[0] for p in PERC if p[1] == 'roll']
cr_t = [p[0] for p in PERC if p[1] == 'crash']
SHELL = mat('shell', (0.75, 0.08, 0.08), 0.25, **{'Coat Weight': 1.0})
HEAD = glowmat('head', gain=3.0, base=0.0, rough=0.5)
BRASS = mat('brass', (0.95, 0.7, 0.3), 0.18, **{'Metallic': 1.0})
DX, DY = 0.0, 1.0
bpy.ops.mesh.primitive_cylinder_add(radius=0.48, depth=0.34, vertices=48, location=(DX, DY, 0.62), rotation=(math.pi / 2, 0, 0))
bpy.context.object.data.materials.append(SHELL)
bpy.ops.mesh.primitive_cylinder_add(radius=0.46, depth=0.01, vertices=48, location=(DX, DY - 0.175, 0.62), rotation=(math.pi / 2, 0, 0))
bdh = bpy.context.object; bdh.data.materials.append(HEAD); bdh.color = (1, 0.95, 0.85, 0)
for tt, vv in flash_pts(bd_t, 0.8, 0.3): bdh.color[3] = vv; bdh.keyframe_insert('color', index=3, frame=tt * FPS)
for x in (-0.3, 0.3): rod((DX + x, DY, 0), (DX + x * 0.8, DY, 0.3), 0.02, STEEL)
BDP = link(bpy.data.objects.new('bdpiv', None)); BDP.location = (DX, DY - 0.5, 1.25)
rod((0, 0, 0), (0, 0.0, -0.55), 0.016, mat('wood', (0.55, 0.35, 0.18), 0.5)).parent = BDP
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.07, location=(0, 0, -0.6)); bb = bpy.context.object; bb.data.materials.append(WHITE); bb.parent = BDP
rod((DX, DY - 0.5, 0), (DX, DY - 0.5, 1.25), 0.025, STEEL)
BDP.rotation_mode = 'XYZ'
def strokes(times, rest, back, hitv, lead=0.16, recover=0.22):
    pts = [(0, rest)]
    for t in sorted(times):
        a = max(t - lead, pts[-1][0] + 0.02)
        if a > pts[-1][0] + 0.02: pts.append((a, back))
        pts.append((t, hitv)); pts.append((t + recover * 0.5, rest))
    return pts
key_pts(BDP, 'rotation_euler', 0, strokes(bd_t, 0.08, -0.3, 0.44))
# snare
SX, SY = -0.9, 0.0
bpy.ops.mesh.primitive_cylinder_add(radius=0.28, depth=0.16, vertices=40, location=(SX, SY, 0.55)); bpy.context.object.data.materials.append(mat('snshell', (0.85, 0.85, 0.9), 0.15, **{'Metallic': 1.0}))
bpy.ops.mesh.primitive_cylinder_add(radius=0.27, depth=0.01, vertices=40, location=(SX, SY, 0.635)); snh = bpy.context.object; snh.data.materials.append(HEAD); snh.color = (1, 1, 1, 0)
for tt, vv in flash_pts(sn_t, 0.7, 0.2): snh.color[3] = vv; snh.keyframe_insert('color', index=3, frame=tt * FPS)
rod((SX, SY, 0), (SX, SY, 0.47), 0.02, STEEL)
SNP = []
for k, dx in enumerate((-0.12, 0.12)):
    sp = link(bpy.data.objects.new(f'snpiv{k}', None)); sp.location = (SX + dx, SY + 0.55, 0.8)
    rod((0, 0, 0), (0, -0.5, -0.13), 0.012, mat('stick', (0.8, 0.6, 0.35), 0.4)).parent = sp
    sp.rotation_mode = 'XYZ'; SNP.append(sp)
rod((SX - 0.12, SY + 0.55, 0), (SX - 0.12, SY + 0.55, 0.8), 0.02, STEEL); rod((SX + 0.12, SY + 0.55, 0), (SX + 0.12, SY + 0.55, 0.8), 0.02, STEEL)
alt = [[], []]
for k, t in enumerate(sorted(sn_t)): alt[k % 2].append(t)
for k in range(2):
    pts = strokes(alt[k], 0.0, -0.35, 0.05, lead=0.12)
    for r0 in roll:   # tremolo for the intro roll
        for j in range(int(E['bar'] * 22)):
            tj = r0 + j / 22 + k / 44; pts.append((tj, -0.12 if j % 2 == 0 else 0.05))
    pts.sort(); ded = []
    for t, v in pts:
        if ded and t <= ded[-1][0] + 1e-3: continue
        ded.append((t, v))
    key_pts(SNP[k], 'rotation_euler', 0, ded, 'LINEAR')
# cymbal
CX, CY = 0.9, 0.1
rod((CX, CY, 0), (CX, CY, 1.05), 0.018, STEEL)
CP_ = link(bpy.data.objects.new('cympiv', None)); CP_.location = (CX, CY, 1.08)
bpy.ops.mesh.primitive_cone_add(radius1=0.36, radius2=0.04, depth=0.05, vertices=48, location=(0, 0, 0)); cy = bpy.context.object; cy.data.materials.append(BRASS); cy.parent = CP_
CP_.rotation_mode = 'XYZ'
wob = np.zeros(NF)
for t in cr_t:
    a = T - t; wob += np.where(a >= 0, 0.35 * np.sin(a * 14) * np.exp(-a * 2.2), 0)
key_series(CP_, 'rotation_euler', 0, -0.35 + wob, 1e-4)
CMP = link(bpy.data.objects.new('cymham', None)); CMP.location = (CX + 0.55, CY - 0.1, 1.5)
rod((0, 0, 0), (-0.4, 0, -0.25), 0.014, STEEL).parent = CMP
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=(-0.42, 0, -0.27)); bpy.context.object.data.materials.append(WHITE); bpy.context.object.parent = CMP
rod((CX + 0.55, CY - 0.1, 0), (CX + 0.55, CY - 0.1, 1.5), 0.02, STEEL)
CMP.rotation_mode = 'XYZ'; key_pts(CMP, 'rotation_euler', 1, strokes(cr_t, 0.0, 0.35, -0.14, 0.2))

# ---------------- scoreboard, grandstand, flags ----------------
YB = SH + RO + 1.6
box((0, YB + 2.75, 1.7), (9.5, 0.3, 3.4), mat('stand', (0.75, 0.78, 0.85), 0.7))
SEATC = [(0.85, 0.15, 0.1), (0.95, 0.85, 0.85), (0.1, 0.3, 0.8), (0.95, 0.7, 0.1)]
for k in range(6):
    box((0, YB + 0.05 + k * 0.45, (0.3 + k * 0.5) / 2), (9.3, 0.45, 0.3 + k * 0.5), mat(f'seat{k}', SEATC[k % 4], 0.6))
box((0, YB + 2.6, 3.9), (3.4, 0.25, 1.2), DARK)
def text(body, loc, size, col, strength=3.0):
    cu = bpy.data.curves.new('t', 'FONT'); cu.body = body; cu.size = size; cu.align_x = 'CENTER'; cu.align_y = 'CENTER'
    o = link(bpy.data.objects.new('t', cu)); o.location = loc; o.rotation_euler = (math.pi / 2, 0, 0)
    o.data.materials.append(emis('txt' + body, col, strength)); return o
TXT = [text('LAP 1 / 2', (0, YB + 2.46, 3.9), 0.62, (1.0, 0.75, 0.2)),
       text('FINAL LAP', (0, YB + 2.46, 3.9), 0.62, (1.0, 0.3, 0.2)),
       text('WINNER!', (0, YB + 2.46, 3.9), 0.62, BCOL[WIN], 6.0)]
spans = [(0, FINAL_LAP_T), (FINAL_LAP_T, FINAL), (FINAL, 999)]
for o, (a, b) in zip(TXT, spans):
    for c in range(3): key_pts(o, 'scale', c, [(0, 1.0 if a == 0 else 0.0), (a, 1.0), (b, 0.0)] if a > 0 else [(0, 1.0), (b, 0.0)], 'CONSTANT')
# string of bunting flags (sports-day "manko-ku-ki")
FLAGC = [(0.9, 0.1, 0.1), (1, 1, 1), (0.1, 0.3, 0.85), (0.95, 0.75, 0.05), (0.1, 0.6, 0.2), (0.9, 0.35, 0.6)]
FM = [mat(f'flag{k}', c, 0.6) for k, c in enumerate(FLAGC)]
for (x0, y0), (x1, y1) in (((-4.8, YB - 0.4), (4.8, YB - 0.4)), ((-4.8, YB - 0.4), (-4.6, -SH - RO - 0.6)), ((4.8, YB - 0.4), (4.6, -SH - RO - 0.6))):
    n = 26 if abs(x1 - x0) > 1 else 40
    for x in (x0, x1): rod((x, y0 if x == x0 else y1, 0), (x, y0 if x == x0 else y1, 3.0), 0.04, WHITE)
    for k in range(n):
        u = (k + 0.5) / n; x = x0 + (x1 - x0) * u; y = y0 + (y1 - y0) * u; z = 2.95 - 0.55 * 4 * u * (1 - u)
        bm = bmesh.new(); a = bm.verts.new((-0.11, 0, 0)); b = bm.verts.new((0.11, 0, 0)); c = bm.verts.new((0, 0, -0.26)); bm.faces.new((a, b, c))
        me = bpy.data.meshes.new('fl'); bm.to_mesh(me); fo = link(bpy.data.objects.new('fl', me)); fo.data.materials.append(FM[k % 6])
        fo.location = (x, y, z); fo.rotation_mode = 'XYZ'; fo.rotation_euler = (0, 0, math.atan2(y1 - y0, x1 - x0))
        fo.rotation_euler[0] = 0.0; fo.keyframe_insert('rotation_euler', index=0, frame=0)
        fc = [f for f in fcurves(fo) if f.data_path == 'rotation_euler'][0]
        if True:
            md = fc.modifiers.new('NOISE'); md.scale = 25; md.strength = 0.5; md.phase = rng.uniform(0, 100)
    # the string
    pts_ = [(x0 + (x1 - x0) * u, y0 + (y1 - y0) * u, 2.95 - 0.55 * 4 * u * (1 - u)) for u in np.linspace(0, 1, 30)]
    for a, b in zip(pts_[:-1], pts_[1:]): rod(a, b, 0.008, WHITE, 6)

# ---------------- confetti on the final chord ----------------
CONF = [mat(f'cf{k}', c, 0.5, **{'Emission Color': (*c, 1), 'Emission Strength': 0.6}) for k, c in enumerate(FLAGC)]
bpy.ops.mesh.primitive_plane_add(size=0.06); cfm = bpy.context.object.data; bpy.data.objects.remove(bpy.context.object)
NCF = 220; tt_ = T[T >= FINAL - 0.05]
for k in range(NCF):
    side = k % 2; x0 = RI - 0.4 if side == 0 else RO + 0.4
    v0 = np.array([(1 if side == 0 else -1) * rng.uniform(0.3, 1.6), rng.uniform(-0.9, 0.9), rng.uniform(3.2, 5.5)])
    o = link(bpy.data.objects.new('cf', cfm)); o.data.materials.append(CONF[k % 6]) if not o.data.materials else None
    o.material_slots[0].link = 'OBJECT'; o.material_slots[0].material = CONF[k % 6]
    o.rotation_mode = 'XYZ'
    t0 = FINAL + rng.uniform(0, 0.12)
    o.location = (x0, -1.0, -0.5); o.keyframe_insert('location', frame=0); o.keyframe_insert('location', frame=t0 * FPS - 1)
    fl = rng.uniform(0, 6.28); dr = rng.uniform(0.6, 1.0)
    for f in range(int(t0 * FPS), NF, 2):
        a = f / FPS - t0
        vz = v0[2] * math.exp(-a * 1.8) - 1.2 * (1 - math.exp(-a * 1.8))
        z = (v0[2] + 1.2) / 1.8 * (1 - math.exp(-a * 1.8)) - 1.2 * a + 0.4
        x = x0 + v0[0] * (1 - math.exp(-a * 1.5)) / 1.5 + 0.08 * math.sin(a * 6 + fl)
        y = -1.0 + v0[1] * (1 - math.exp(-a * 1.5)) / 1.5
        o.location = (x, y, max(z, 0.01)); o.keyframe_insert('location', frame=f)
        o.rotation_euler = (a * 7 * dr + fl, a * 5 + fl, a * 3); o.keyframe_insert('rotation_euler', frame=f)

# ---------------- world & lights ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True; nt = w.node_tree
bg = nt.nodes['Background']
try:
    sky = nt.nodes.new('ShaderNodeTexSky')
    for st in ('MULTIPLE_SCATTERING', 'NISHITA'):
        try: sky.sky_type = st; break
        except Exception: pass
    sky.sun_elevation = math.radians(38); sky.sun_rotation = math.radians(200)
    sky.sun_intensity = 0.4; nt.links.new(sky.outputs[0], bg.inputs['Color']); bg.inputs['Strength'].default_value = 0.06
except Exception:
    bg.inputs['Color'].default_value = (0.45, 0.65, 1.0, 1); bg.inputs['Strength'].default_value = 0.8
sun = bpy.data.lights.new('sun', 'SUN'); sun.energy = 3.2; sun.angle = math.radians(4); sun.color = (1.0, 0.95, 0.88)
so = link(bpy.data.objects.new('sun', sun)); so.rotation_euler = (math.radians(48), 0, math.radians(-30))

# ---------------- camera: one wide, steady view of the whole stadium ----------------
camd = bpy.data.cameras.new('cam'); camd.lens = float(os.environ.get('LENS', 30)); camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
tr = cam.constraints.new('TRACK_TO'); tr.target = tgt; tr.track_axis = 'TRACK_NEGATIVE_Z'; tr.up_axis = 'UP_Y'
def zp_smooth(X, tau):
    a = 1 - math.exp(-1 / FPS / tau); Y = X.copy()
    for f in range(1, len(X)): Y[f] = Y[f - 1] + a * (X[f] - Y[f - 1])
    for f in range(len(X) - 2, -1, -1): Y[f] = Y[f + 1] + a * (Y[f] - Y[f + 1])
    return Y
pack = BP[:, :, :2].mean(0)
TG = np.stack([0.75 * pack[:, 0], 0.95 * pack[:, 1] - 0.2, np.full(NF, 0.25)], 1)
for c in range(2): TG[:, c] = zp_smooth(TG[:, c], 0.9)
dist = 13.0 + 1.2 * smooth01(1 - (T - 1.0) / 3.0) + 2.5 * smooth01((T - FINAL - 0.3) / 2.5)
az = np.radians(10) * np.tanh(TG[:, 0] / 1.5)                 # swing a little towards the side the pack is on
el = np.radians(40)
CPx = TG[:, 0] + dist * np.cos(el) * np.sin(az); CPy = TG[:, 1] - dist * np.cos(el) * np.cos(az); CPz = TG[:, 2] + dist * np.sin(el)
for c, arr in enumerate((CPx, CPy, CPz)): key_series(cam, 'location', c, arr, 1e-4)
for c in range(3): key_series(tgt, 'location', c, TG[:, c], 1e-4)

print('objects', len(bpy.data.objects), 'NF', NF, 'hurdles', NH, 'lap1', [round(x, 1) for x in lap1], 'tape', round(T_TAPE, 1))
if MODE == 'test':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS)); sc.frame_set(f)
        sc.render.filepath = f'{HERE}/test_{f:04d}.png'
        t1 = time.time(); bpy.ops.render.render(write_still=True); print('frame', f, 'sec', round(time.time() - t1, 1))
elif MODE == 'range':
    sc.frame_start, sc.frame_end = int(args[1]), int(args[2])
    sc.render.image_settings.file_format = 'PNG'; sc.render.filepath = f'{HERE}/frames/f_'
    bpy.ops.render.render(animation=True)
