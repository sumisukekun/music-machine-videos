"""Note Bounce - glowing balls hop from notehead to notehead across a tilted 3D score on parchment.
Yellow ball = right-hand arpeggio (treble staff), cyan ball = left-hand bass (bass staff).
Each landing lights the notehead and throws a puff of little hearts."""
import os, sys, json, math, time, colorsys
import bpy, bmesh
import numpy as np
from mathutils import Vector

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); NOTES = json.load(open(f'{HERE}/notes.json'))
FPS = 30; FINAL = E['final']; T_END = FINAL + 4.0; NF = int(T_END * FPS); BAR = E['bar']; T0 = E['t0']
rng = np.random.default_rng(3)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'
sc.cycles.samples = int(os.environ.get('SAMP', 6)); sc.cycles.use_adaptive_sampling = True; sc.cycles.adaptive_threshold = 0.06
sc.cycles.use_denoising = True
for k, v in (('denoiser', 'OPENIMAGEDENOISE'), ('denoising_quality', 'HIGH'), ('use_fast_gi', True)):
    try: setattr(sc.cycles, k, v)
    except Exception: pass
sc.cycles.max_bounces = 4; sc.cycles.diffuse_bounces = 1; sc.cycles.glossy_bounces = 1; sc.cycles.transparent_max_bounces = 4
sc.render.resolution_x, sc.render.resolution_y = 1080, 1920
sc.render.resolution_percentage = int(os.environ.get('PCT', 50)); sc.render.fps = FPS
sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.35
try: sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Medium High Contrast'
except Exception: pass
sc.frame_start, sc.frame_end = 0, NF - 1
TT = np.arange(NF) / FPS


def link(o): sc.collection.objects.link(o); return o


def mat(name, col, rough=0.5, **kw):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']; b.inputs['Base Color'].default_value = (*col, 1); b.inputs['Roughness'].default_value = rough
    for k, v in kw.items(): b.inputs[k].default_value = v
    return m


def glowmat(name, col, gain=14.0, base_col=(0.03, 0.025, 0.02)):
    """dark ink notehead; object colour alpha drives an emissive glow"""
    m = mat(name, base_col, 0.35); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    oi = nt.nodes.new('ShaderNodeObjectInfo'); mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = gain
    nt.links.new(oi.outputs['Alpha'], mul.inputs[0]); nt.links.new(mul.outputs[0], b.inputs['Emission Strength'])
    nt.links.new(oi.outputs['Color'], b.inputs['Emission Color'])
    return m


def emis(name, col, s):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    e = nt.nodes.new('ShaderNodeEmission'); e.inputs['Color'].default_value = (*col, 1); e.inputs['Strength'].default_value = s
    o = nt.nodes.new('ShaderNodeOutputMaterial'); nt.links.new(e.outputs[0], o.inputs[0]); return m


def key_series(obj, path, idx, vals, eps=1e-5):
    n = len(vals)
    for f in range(n):
        v = vals[f]
        if 0 < f < n - 1 and abs(v - vals[f - 1]) < eps and abs(v - vals[f + 1]) < eps: continue
        getattr(obj, path)[idx] = float(v); obj.keyframe_insert(path, index=idx, frame=f)


def key_alpha(obj, arr, col):
    obj.color = (*col, 0)
    for f in range(len(arr)):
        if 0 < f < len(arr) - 1 and abs(arr[f] - arr[f - 1]) < 1e-4 and abs(arr[f] - arr[f + 1]) < 1e-4: continue
        obj.color[3] = float(arr[f]); obj.keyframe_insert('color', index=3, frame=f)


# ---------------- geometry of the score ----------------
SPX = 0.85                    # metres per second along the staff
LS = 0.07                     # staff line spacing; one diatonic step = LS/2
TRE_Y = 0.30; BAS_Y = -0.30    # centre (middle line) of each staff
LETTER = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 6, 11: 6}   # pc -> letter index (C=0..B=6), sharps share


def dia(p):  # diatonic step number
    return (p // 12) * 7 + LETTER[p % 12]


def note_y(p, voice):
    if voice == 0: return TRE_Y + (dia(p) - dia(71)) * LS / 2      # B4 on middle line
    return BAS_Y + (dia(p) - dia(50)) * LS / 2                      # D3 on middle line


def tx(t): return (t - T0) * SPX


X_MIN = tx(0) - 2.0; X_MAX = tx(FINAL) + 3.0
# parchment: long plane with procedural noise
PAPER = bpy.data.materials.new('paper'); PAPER.use_nodes = True; nt = PAPER.node_tree
bsdf = nt.nodes['Principled BSDF']; bsdf.inputs['Roughness'].default_value = 0.75
noi = nt.nodes.new('ShaderNodeTexNoise'); noi.inputs['Scale'].default_value = 3.0; noi.inputs['Detail'].default_value = 8
ramp = nt.nodes.new('ShaderNodeValToRGB')
ramp.color_ramp.elements[0].color = (0.22, 0.13, 0.06, 1); ramp.color_ramp.elements[1].color = (0.5, 0.34, 0.17, 1)
nt.links.new(noi.outputs['Fac'], ramp.inputs['Fac']); nt.links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
bpy.ops.mesh.primitive_plane_add(size=1, location=((X_MIN + X_MAX) / 2, 0, 0)); paper = bpy.context.object
paper.scale = (X_MAX - X_MIN, 2.6, 1); paper.data.materials.append(PAPER)
INK = mat('ink', (0.02, 0.016, 0.012), 0.4)
bm = bmesh.new()
def add_box(c, sx, sy, sz):
    r = bmesh.ops.create_cube(bm, size=1.0)
    for v in r['verts']: v.co = Vector((c[0] + v.co.x * sx, c[1] + v.co.y * sy, c[2] + v.co.z * sz))
for yc in (TRE_Y, BAS_Y):
    for k in range(-2, 3): add_box((( X_MIN + X_MAX) / 2, yc + k * LS, 0.001), X_MAX - X_MIN, 0.006, 0.002)
# bar lines
for b in range(E['nbars'] + 2):
    x = tx(T0 + b * BAR) - 0.12
    for yc in (TRE_Y, BAS_Y): add_box((x, yc, 0.001), 0.008, 4 * LS, 0.002)
me = bpy.data.meshes.new('staff'); bm.to_mesh(me); bm.free()
staff = link(bpy.data.objects.new('staff', me)); staff.data.materials.append(INK)
# bar numbers along the bottom edge (like the reference)
for b in range(E['nbars'] + 1):
    cu = bpy.data.curves.new(f'bn{b}', 'FONT'); cu.body = str(b + 1); cu.size = 0.07; cu.align_x = 'CENTER'
    o = link(bpy.data.objects.new(f'bn{b}', cu)); o.location = (tx(T0 + b * BAR) + 0.2, BAS_Y - 0.26, 0.001); o.data.materials.append(INK)
# clefs drawn as simple glyphs at the start
for yc, s in ((TRE_Y, '𝄞'), (BAS_Y, '𝄢')):
    cu = bpy.data.curves.new('clef', 'FONT'); cu.body = 'G' if s == '𝄞' else 'F'; cu.size = 0.22
    o = link(bpy.data.objects.new('clef', cu)); o.location = (tx(T0) - 0.55, yc - 0.08, 0.001); o.data.materials.append(INK)

# ---------------- noteheads ----------------
COL = {0: (1.0, 0.55, 0.05), 1: (0.05, 0.75, 1.0)}
heads = []
NHM = {v: glowmat(f'nh{v}', COL[v], 6.0) for v in (0, 1)}
STEMM = INK
for i, (t, d, v, p, vel) in enumerate(NOTES):
    x = tx(t); y = note_y(p, v)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=LS * 0.55, location=(x, y, 0.012), segments=20, ring_count=10)
    o = bpy.context.object; o.scale = (1.25, 0.85, 0.35); o.rotation_euler = (0, 0, 0.35)
    o.data.materials.append(NHM[v])
    for pl in o.data.polygons: pl.use_smooth = True
    # ledger lines for notes outside the staff
    yc = TRE_Y if v == 0 else BAS_Y
    step = round((y - yc) / (LS / 2))
    for s in range(6, abs(step) + 1, 2):
        yy = yc + math.copysign(s, step) * LS / 2
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x, yy, 0.002)); l = bpy.context.object; l.scale = (0.14, 0.005, 0.002); l.data.materials.append(INK)
    # stems (up for treble, down for bass), quarter-ish length
    if d < 2.5:
        sx = x + (LS * 0.6 if v == 0 else -LS * 0.6); sl = 3.2 * LS
        sy = y + (sl / 2 if v == 0 else -sl / 2)
        bpy.ops.mesh.primitive_cube_add(size=1, location=(sx, sy, 0.003)); s_ = bpy.context.object; s_.scale = (0.006, sl, 0.003); s_.data.materials.append(INK)
    heads.append((o, t, v, Vector((x, y, 0.012))))

# ---------------- balls ----------------
BALL_R = 0.045
balls = {}
for v in (0, 1):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=BALL_R, segments=32, ring_count=16)
    b = bpy.context.object; b.data.materials.append(emis(f'ball{v}', COL[v], 3.5))
    for pl in b.data.polygons: pl.use_smooth = True
    ld = bpy.data.lights.new(f'bl{v}', 'POINT'); ld.energy = 6; ld.color = COL[v]; ld.shadow_soft_size = 0.05
    lo = link(bpy.data.objects.new(f'bl{v}', ld)); lo.parent = b
    balls[v] = b
hits = {0: [], 1: []}
for o, t, v, P in heads:
    if not hits[v] or abs(hits[v][-1][0] - t) > 1e-3: hits[v].append((t, P, [o]))
    else: hits[v][-1][2].append(o)
for v, hl in hits.items():
    if v == 0:   # chord at the end: land on the highest note
        last = hl[-1]; hl[-1] = (last[0], max((h.location for h in last[2]), key=lambda q: q.y).copy(), last[2])
    pos = np.zeros((NF, 3))
    ZB = 0.012 + BALL_R * 0.9
    for f in range(NF):
        t = TT[f]
        if t <= hl[0][0]:
            P = hl[0][1]; a = max(0.0, hl[0][0] - t)
            pos[f] = (P.x - 0.0, P.y, ZB + 4.9 * a * a * 0.6 if a < 1.2 else ZB + 4.9 * 1.44 * 0.6)
            continue
        if t >= hl[-1][0]:
            P = hl[-1][1]; a = t - hl[-1][0]
            pos[f] = (P.x, P.y, ZB + 0.12 * abs(math.sin(min(a, 1.5) * 6)) * math.exp(-a * 2.5)); continue
        k = max(i for i in range(len(hl)) if hl[i][0] <= t)
        ta, Pa, _ = hl[k]; tb, Pb, _ = hl[k + 1]
        u = (t - ta) / (tb - ta); h = 0.05 + 0.32 * min(1.0, (tb - ta))
        pos[f] = (Pa.x + (Pb.x - Pa.x) * u, Pa.y + (Pb.y - Pa.y) * u, ZB + 4 * h * u * (1 - u))
    for i in range(3): key_series(balls[v], 'location', i, pos[:, i])
    balls[v]['path'] = 1
    globals()[f'POS{v}'] = pos
# notehead glow: lights on impact, stays softly lit (trail of played notes)
for o, t, v, P in heads:
    a = np.where(TT >= t - 0.5 / FPS, 0.35 + 0.65 * np.exp(-np.clip(TT - t, 0, None) / 0.18), 0.0)
    key_alpha(o, a, COL[v])

# ---------------- hearts ----------------
def heart_mesh():
    bm_ = bmesh.new(); pts = []
    for k in range(48):
        a = 2 * math.pi * k / 48
        x = 16 * math.sin(a) ** 3; y = 13 * math.cos(a) - 5 * math.cos(2 * a) - 2 * math.cos(3 * a) - math.cos(4 * a)
        pts.append(bm_.verts.new((x / 17, y / 17, 0)))
    f = bm_.faces.new(pts); bmesh.ops.triangulate(bm_, faces=[f])
    me_ = bpy.data.meshes.new('heart'); bm_.to_mesh(me_); return me_
HM = heart_mesh()
HMAT = {v: emis(f'hm{v}', COL[v], 3.0) for v in (0, 1)}
cam = None
POOL = {v: [] for v in (0, 1)}
NPOOL = {0: 36, 1: 18}
for v in (0, 1):
    for i in range(NPOOL[v]):
        o = link(bpy.data.objects.new(f'h{v}_{i}', HM.copy())); o.data.materials.append(HMAT[v]); o.scale = (0, 0, 0)
        POOL[v].append(o)
for v in (0, 1):
    L = [np.tile([0, 0, -5.0], (NF, 1)) for _ in POOL[v]]; S = [np.zeros(NF) for _ in POOL[v]]
    c = 0
    for t, P, objs in hits[v]:
        nh = 3 if v == 0 else 5
        if t >= FINAL - 0.01: nh = 12
        for k in range(nh):
            j = c % len(POOL[v]); c += 1
            ang = rng.uniform(0, 2 * math.pi); sp = rng.uniform(0.15, 0.35); up = rng.uniform(0.35, 0.7); life = rng.uniform(0.5, 0.8)
            for f in range(max(0, int(t * FPS)), min(NF, int((t + life) * FPS))):
                a = f / FPS - t
                if a < 0: continue
                L[j][f] = (P.x + math.cos(ang) * sp * a, P.y + math.sin(ang) * sp * a, P.z + 0.03 + up * a - 0.25 * a * a)
                S[j][f] = 0.035 * min(1, a / 0.06) * (1 - a / life) * (1.6 if t >= FINAL - 0.01 else 1)
    for j, o in enumerate(POOL[v]):
        for i in range(3): key_series(o, 'location', i, L[j][:, i])
        for i in range(3): key_series(o, 'scale', i, S[j])

# ---------------- light & camera ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs['Color'].default_value = (0.01, 0.008, 0.006, 1); w.node_tree.nodes['Background'].inputs['Strength'].default_value = 1
rig = link(bpy.data.objects.new('rig', None))
camd = bpy.data.cameras.new('cam'); camd.lens = 32; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24
camd.dof.use_dof = True; camd.dof.aperture_fstop = 2.8
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam; cam.parent = rig
tgt = link(bpy.data.objects.new('tgt', None)); tgt.parent = rig
tr = cam.constraints.new('TRACK_TO'); tr.target = tgt; tr.track_axis = 'TRACK_NEGATIVE_Z'; tr.up_axis = 'UP_Y'
cam.location = (-1.1, -1.7, 1.6); tgt.location = (0.45, -0.05, 0.0)
camd.dof.focus_object = tgt
for nm, loc, e, col, sz in (('key', (-0.6, -0.8, 1.6), 38, (1.0, 0.8, 0.55), 1.6), ('rim', (1.2, 1.0, 0.9), 14, (0.6, 0.75, 1.0), 1.5)):
    ld = bpy.data.lights.new(nm, 'AREA'); ld.energy = e; ld.color = col; ld.size = sz
    lo = link(bpy.data.objects.new(nm, ld)); lo.parent = rig; lo.location = loc
    c_ = lo.constraints.new('TRACK_TO'); c_.target = tgt; c_.track_axis = 'TRACK_NEGATIVE_Z'; c_.up_axis = 'UP_Y'
# rig follows the leading (treble) ball, smoothed
xs = POS0[:, 0]
k = 0.9; sm = np.zeros(NF); sm[0] = xs[0]
for f in range(1, NF): sm[f] = sm[f - 1] + (xs[f] - sm[f - 1]) * (1 - math.exp(-1 / FPS / k))
sm = np.maximum(sm, xs - 0.6)
key_series(rig, 'location', 0, sm)
rig.rotation_mode = 'XYZ'
# slow sway + pull back at the finale
sway = 0.08 * np.sin(TT * 0.35)
key_series(rig, 'rotation_euler', 2, sway)
fin = np.clip((TT - FINAL) / 2.0, 0, 1); fin = fin * fin * (3 - 2 * fin)
key_series(cam, 'location', 2, 1.6 + 0.5 * fin); key_series(cam, 'location', 0, -1.1 - 0.3 * fin)

print('objects', len(bpy.data.objects), 'NF', NF)
if MODE == 'test':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS)); sc.frame_set(f)
        sc.render.filepath = f'{HERE}/test_{f:04d}.png'
        t1 = time.time(); bpy.ops.render.render(write_still=True); print('frame', f, 'sec', round(time.time() - t1, 1))
elif MODE == 'range':
    sc.frame_start, sc.frame_end = int(args[1]), int(args[2])
    sc.render.image_settings.file_format = 'PNG'; sc.render.filepath = f'{HERE}/frames/f_'
    bpy.ops.render.render(animation=True)
