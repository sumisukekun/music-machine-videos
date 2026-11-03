"""Twinkle Stars (Mozart K.265 complete) - glowing marbles on star-shaped plates; every variation re-colours the machine.
Based on the marble-run machine: bounce down a spiral of brass plates and wire rails, one note per hit.
A new ball joins with every new voice of the canon (Victor de Martrin style macro look)."""
import os, sys, json, math, colorsys, time
import bpy, bmesh
import numpy as np
from mathutils import Vector

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json'))
FPS = 30; T_END = E['final'] + 3.5; NF = int(T_END * FPS); FINAL = E['final']; BEAT = E['beat']; TB0 = E['t0']
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
for k, v in (('denoiser', 'OPENIMAGEDENOISE'), ('denoising_quality', 'HIGH'), ('denoising_prefilter', 'ACCURATE'),
             ('use_fast_gi', True), ('ao_bounces_render', 1)):
    try: setattr(sc.cycles, k, v)
    except Exception: pass
sc.cycles.max_bounces = 4; sc.cycles.diffuse_bounces = 1; sc.cycles.glossy_bounces = 2
sc.cycles.caustics_reflective = False; sc.cycles.caustics_refractive = False
sc.cycles.sample_clamp_indirect = 4
sc.render.resolution_x, sc.render.resolution_y = 720, 1280
sc.render.resolution_percentage = int(os.environ.get('PCT', 66))
sc.render.fps = FPS
sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.25
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



sc.cycles.max_bounces = 8; sc.cycles.transmission_bounces = 6; sc.cycles.glossy_bounces = 3
from mathutils import Matrix, Quaternion
NOTES = json.load(open(f'{HERE}/notes.json'))
GV = Vector((0, 0, -G))
BR = 0.034; S_H = 0.6; CZ = 0.22; Z0 = 15.0
NB = 3
RAD = [0.56, 0.86, 0.36]
DIRS = [1, 1, 1, 1, 1, 1]; OMG = 1.05
PHI = [0.0, -0.1, 0.12]
BCOL = [(1.0, 0.78, 0.25), (0.25, 0.6, 1.0), (1.0, 0.35, 0.6)]
SECT = E['sections']
def sec_of(t):
    k = 0
    for i, (a, b_, n_) in enumerate(SECT):
        if t >= a - 1.5: k = i
    return k
PAL = [((1.0, 0.76, 0.33), 0.16), ((0.93, 0.94, 0.97), 0.1), ((1.0, 0.6, 0.52), 0.16), ((0.42, 0.6, 1.0), 0.18), ((0.95, 0.5, 0.3), 0.2),
       ((0.96, 0.86, 0.62), 0.14), ((0.82, 0.84, 0.88), 0.22), ((0.72, 0.5, 1.0), 0.16), ((1.0, 0.76, 0.33), 0.1), ((0.93, 0.94, 0.97), 0.16),
       ((1.0, 0.6, 0.52), 0.12), ((0.42, 0.6, 1.0), 0.12), ((1.0, 0.8, 0.4), 0.08)]

def anchor(b, t):
    th = PHI[b] + DIRS[b] * OMG * t
    return Vector((RAD[b] * math.cos(th), RAD[b] * math.sin(th), Z0 - CZ * t))

def tangent(b, t):
    th = PHI[b] + DIRS[b] * OMG * t
    return Vector((-math.sin(th), math.cos(th), 0)) * DIRS[b]

# ---------------- materials ----------------
BRASS = mat('brass', (1.0, 0.74, 0.36), 0.16, 1.0)
BRONZE = mat('bronze', (0.85, 0.5, 0.25), 0.22, 1.0)
WIRE = mat('wire', (1.0, 0.8, 0.45), 0.12, 1.0)
GLASS = mat('glass', (1.0, 1.0, 1.0), 0.0, 0.0, **{'Transmission Weight': 1.0, 'IOR': 1.5})
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
nt = w.node_tree; bgn = nt.nodes['Background']
grad = nt.nodes.new('ShaderNodeTexGradient'); grad.gradient_type = 'LINEAR'
tc = nt.nodes.new('ShaderNodeTexCoord'); mp = nt.nodes.new('ShaderNodeMapping'); mp.inputs['Rotation'].default_value = (0, -math.pi / 2, 0)
ramp = nt.nodes.new('ShaderNodeValToRGB')
ramp.color_ramp.elements[0].position = 0.35; ramp.color_ramp.elements[0].color = (0.006, 0.008, 0.03, 1)
ramp.color_ramp.elements[1].position = 0.65; ramp.color_ramp.elements[1].color = (0.05, 0.035, 0.12, 1)
nt.links.new(tc.outputs['Generated'], mp.inputs['Vector']); nt.links.new(mp.outputs['Vector'], grad.inputs['Vector'])
nt.links.new(grad.outputs['Fac'], ramp.inputs['Fac']); nt.links.new(ramp.outputs['Color'], bgn.inputs['Color'])
bgn.inputs['Strength'].default_value = 1.0
vor = nt.nodes.new('ShaderNodeTexVoronoi'); vor.inputs['Scale'].default_value = 140
nt.links.new(tc.outputs['Generated'], vor.inputs['Vector'])
mr = nt.nodes.new('ShaderNodeMapRange'); mr.inputs['From Min'].default_value = 0.0; mr.inputs['From Max'].default_value = 0.11
mr.inputs['To Min'].default_value = 1.0; mr.inputs['To Max'].default_value = 0.0; mr.clamp = True
nt.links.new(vor.outputs['Distance'], mr.inputs['Value'])
pw = nt.nodes.new('ShaderNodeMath'); pw.operation = 'POWER'; pw.inputs[1].default_value = 4.0; nt.links.new(mr.outputs[0], pw.inputs[0])
mx = nt.nodes.new('ShaderNodeMix'); mx.data_type = 'RGBA'; mx.blend_type = 'ADD'; mx.inputs['Factor'].default_value = 1.0
nt.links.new(ramp.outputs['Color'], mx.inputs['A']); nt.links.new(pw.outputs[0], mx.inputs['B'])
nt.links.new(mx.outputs['Result'], bgn.inputs['Color'])

# lights ride down with the machine
rig = link(bpy.data.objects.new('rig', None))
key_series(rig, 'location', 2, Z0 - CZ * TT)
def area(name, loc, col, e, size):
    ld = bpy.data.lights.new(name, 'AREA'); ld.energy = e; ld.color = col; ld.size = size
    o = link(bpy.data.objects.new(name, ld)); o.parent = rig; o.location = loc
    tr = o.constraints.new('TRACK_TO'); tr.target = rig; tr.track_axis = 'TRACK_NEGATIVE_Z'; tr.up_axis = 'UP_Y'
area('key', (-2.5, -2.0, 2.5), (1.0, 0.9, 0.8), 1000, 2.0)
area('fill', (2.5, -1.5, 0.5), (0.55, 0.65, 1.0), 450, 2.5)
area('top', (0.0, 1.0, 3.5), (0.85, 0.85, 1.0), 500, 2.0)

# ---------------- ball paths ----------------
class Path:
    def __init__(self): self.seg = []         # (ta, tb, fn) fn(tau) -> pos
    def add_arc(self, ta, tb, p0, p1):
        T = tb - ta; v0 = (p1 - p0 - 0.5 * GV * T * T) / T
        self.seg.append((ta, tb, lambda tau, p0=p0, v0=v0: p0 + v0 * tau + 0.5 * GV * tau * tau))
        return v0, v0 + GV * T
    def add_fn(self, ta, tb, fn): self.seg.append((ta, tb, fn))
    def pos(self, t):
        if t < self.seg[0][0] or t > self.seg[-1][1]: return None
        for ta, tb, fn in self.seg:
            if ta <= t <= tb: return fn(t - ta)
        return None

PATHS = []; PLATES = []; RAILS = []; TINES = []; CRADLES = []
FINT = E['final']; THF = OMG * FINT + 0.35
BOWL = Vector((0.62 * math.cos(THF), 0.62 * math.sin(THF), Z0 - CZ * FINT - 0.62))
FAST = 0.26
def helix_fn(b, ta, tb, za, zb, sag=0.0):
    th0 = PHI[b] + DIRS[b] * OMG * ta; th1 = PHI[b] + DIRS[b] * OMG * tb; T = tb - ta; R = RAD[b]
    def rf(tau):
        a = tau / T; th = th0 + (th1 - th0) * a
        return Vector((R * math.cos(th), R * math.sin(th), za + (zb - za) * a - sag * math.sin(math.pi * a)))
    return rf
for b in range(NB):
    seen = set(); ns = []
    for t, d, ball, p, v in NOTES:
        if ball == b and round(t, 3) not in seen: seen.add(round(t, 3)); ns.append((t, p, d))
    # group fast passages into runs: the ball rolls along a wire rail and ticks brass tines
    runs = []; cur = [ns[0]]
    for nt_ in ns[1:]:
        if nt_[0] - cur[-1][0] < FAST: cur.append(nt_)
        else: runs.append(cur); cur = [nt_]
    runs.append(cur)
    path = Path()
    t0 = runs[0][0][0]; P0 = anchor(b, t0)
    W = P0 + Vector((0, 0, 0.45)) - tangent(b, t0) * 0.15
    path.add_fn(max(-1.0, t0 - 8.0), t0 - 0.4, lambda tau, W=W: W.copy())
    vin = path.add_arc(t0 - 0.4, t0, W, P0)[1]
    CRADLES.append((b, W, tangent(b, t0), t0 - 0.4))
    for ri, run in enumerate(runs):
        ta, tb = run[0][0], run[-1][0]; Pa, Pb = anchor(b, ta), anchor(b, tb)
        if len(run) > 1:   # roll across the tines
            rf = helix_fn(b, ta, tb, Pa.z, Pb.z); path.add_fn(ta, tb, rf); RAILS.append((b, rf, tb - ta, ta))
            for t, p, d in run: TINES.append((b, rf(t - ta), tangent(b, t), p, t))
            eps = 0.02; vroll = (rf(min(tb - ta, eps)) - rf(0)) / eps
        else: vroll = None
        if ri + 1 < len(runs):
            t1 = runs[ri + 1][0][0]; P1 = anchor(b, t1); dt = t1 - tb
            if dt >= 0.55:
                tA, tB = tb + 0.14, t1 - 0.14
                Q0 = anchor(b, tA) + Vector((0, 0, 0.05 + CZ * 0.14)); Q1 = anchor(b, tB) + Vector((0, 0, 0.09 - CZ * 0.14))
                vout = path.add_arc(tb, tA, Pb, Q0)[0]
                rf2 = helix_fn(b, tA, tB, Q0.z, Q1.z, 0.012); path.add_fn(tA, tB, rf2); RAILS.append((b, rf2, tB - tA, tA))
                vin_next = path.add_arc(tB, t1, Q1, P1)[1]
            else:
                vout, vin_next = path.add_arc(tb, t1, Pb, P1)
        else:
            ang0 = 2 * math.pi * b / NB; TL = 0.62 + 0.05 * b
            Lp = BOWL + Vector((0.09 * math.cos(ang0), 0.09 * math.sin(ang0), 0.03 + BR))
            vout = path.add_arc(tb, tb + TL, Pb, Lp)[0]
            def settle(tau, ang0=ang0, TL=TL):
                tt = tau; r = 0.088 + 0.012 * math.exp(-tt / 0.6)
                ang = ang0 + 2.6 * 0.8 * (1 - math.exp(-tt / 0.8)) - 0.0 * tt
                z = 0.03 + BR + (0.1 - r) * 0.6
                return BOWL + Vector((r * math.cos(ang), r * math.sin(ang), z))
            path.add_fn(tb + TL, T_END + 0.5, settle); vin_next = None
        # landing plate at the start of this run (the first note), unless it is a tine run start handled below
        vland_out = vroll if vroll is not None else vout
        n = (vland_out - vin); n = n.normalized() if n.length > 1e-6 else Vector((0, 0, 1))
        if len(run) == 1:
            PLATES.append((b, Pa, n, tangent(b, ta), run[0][1], ta))
        if len(run) > 1:   # also need a plate where the last tine hands over to the flight
            nb_ = (vout - vroll); nb_ = nb_.normalized() if nb_.length > 1e-6 else Vector((0, 0, 1))
        vin = vin_next
    PATHS.append(path)
print('plates', len(PLATES), 'rails', len(RAILS), 'tines', len(TINES))

# ---------------- static geometry: plates / bass chimes / pins in single meshes ----------------
def frame_from(n, tng):
    x = (tng - n * tng.dot(n)); x = x.normalized() if x.length > 1e-6 else n.orthogonal().normalized()
    y = n.cross(x)
    return Matrix((x, y, n)).transposed()   # columns x, y, n

def add_box(bm, M, c, size):
    r = bmesh.ops.create_cube(bm, size=1.0)
    for v in r['verts']:
        v.co = c + M @ Vector((v.co.x * size[0], v.co.y * size[1], v.co.z * size[2]))

def add_cyl(bm, M, c, r_, L, seg=12):
    r = bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=r_, radius2=r_, depth=L)
    for v in r['verts']: v.co = c + M @ v.co

def add_star(bm, M, c, ro, th, pts=5):
    ri = ro * 0.46; top = []; bot = []
    for k in range(2 * pts):
        r_ = ro if k % 2 == 0 else ri; a_ = math.pi / 2 + math.pi * k / pts
        v = Vector((r_ * math.cos(a_), r_ * math.sin(a_), 0))
        top.append(bm.verts.new(c + M @ (v + Vector((0, 0, th / 2))))); bot.append(bm.verts.new(c + M @ (v - Vector((0, 0, th / 2)))))
    ct = bm.verts.new(c + M @ Vector((0, 0, th / 2 + th * 0.8))); cb = bm.verts.new(c + M @ Vector((0, 0, -th / 2)))
    n_ = len(top)
    for k in range(n_):
        k2 = (k + 1) % n_
        bm.faces.new((top[k], top[k2], ct)); bm.faces.new((bot[k2], bot[k], cb)); bm.faces.new((top[k], bot[k], bot[k2], top[k2]))
NS = len(SECT)
BMS = {kind: [bmesh.new() for _ in range(NS)] for kind in ('pl', 'tb', 'pin')}
for b, P, n, tng, p, t in PLATES:
    M = frame_from(n, tng); k = sec_of(t)
    if b == 1:       # bass: horizontal chime tubes
        r_ = 0.011; L = 0.10 + 0.08 * max(0, 62 - p) / 20
        Mt = M @ Matrix.Rotation(math.pi / 2, 3, 'Y')
        add_cyl(BMS['tb'][k], Mt, P - n * (BR + r_), r_, L)
        add_cyl(BMS['pin'][k], M, P - n * (BR + 2 * r_ + 0.016), 0.0025, 0.03, 6)
    else:            # melody: little stars
        ro = max(0.035, 0.06 - 0.0007 * (p - 64)); th = 0.006
        c = P - n * (BR + th / 2)
        add_star(BMS['pl'][k], M, c, ro, th)
        add_cyl(BMS['pin'][k], M, c - n * 0.018, 0.0022, 0.032, 6)
        r = bmesh.ops.create_uvsphere(BMS['pin'][k], u_segments=8, v_segments=6, radius=0.0055)
        for v in r['verts']: v.co = v.co + c - n * 0.036
for b, P, tng, p, t in TINES:   # tiny stars across the rails, struck while rolling
    rad = Vector((P.x, P.y, 0)).normalized(); M = frame_from(Vector((0, 0, 1)), rad); k = sec_of(t)
    add_star(BMS['pl'][k], M, P - Vector((0, 0, BR * 0.72 + 0.005)), 0.026, 0.004)
SECM = []
for k in range(NS):
    col, rough = PAL[k % len(PAL)]
    mp_ = mat(f'metal{k}', col, rough, 1.0); mt_ = mat(f'tube{k}', tuple(c * 0.85 for c in col), rough + 0.06, 1.0)
    mw_ = mat(f'wire{k}', tuple(min(1, c * 1.05) for c in col), 0.1, 1.0); SECM.append((mp_, mt_, mw_))
    for kind, mt in (('pl', mp_), ('tb', mt_), ('pin', mw_)):
        me = bpy.data.meshes.new(f'{kind}{k}'); BMS[kind][k].to_mesh(me); BMS[kind][k].free()
        if len(me.vertices) == 0: continue
        me.materials.append(mt); o = link(bpy.data.objects.new(f'{kind}{k}', me))
        for poly in me.polygons: poly.use_smooth = kind != 'pl'
# twin-wire rails, one curve per variation (colour changes with the music)
CUS = [None] * NS
for b, rf, T, t in RAILS:
    k = sec_of(t)
    if CUS[k] is None:
        cu = bpy.data.curves.new(f'rails{k}', 'CURVE'); cu.dimensions = '3D'; cu.bevel_depth = 0.0018; cu.bevel_resolution = 2; CUS[k] = cu
    cu = CUS[k]
    nseg = max(16, int(T * 24)); pts = [rf(T * j / nseg) for j in range(nseg + 1)]   # dense enough that long rails follow the curve
    for side in (-1, 1):
        sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
        for j, q in enumerate(pts):
            rad = Vector((q.x, q.y, 0)).normalized()
            sp.points[j].co = (*(q + rad * side * 0.013 - Vector((0, 0, BR * 0.72))), 1)
for k, cu in enumerate(CUS):
    if cu is None: continue
    co = link(bpy.data.objects.new(f'rails{k}', cu)); co.data.materials.append(SECM[k][2])
# central spine
spine = rod((0, 0, Z0 + 1.0), (0, 0, Z0 - CZ * T_END - 1.0), 0.012, BRASS, 16)
# release cradles: a small brass cup on an arm; it tips over to let the waiting ball go
CRAD = []
for b, W, tng, trel in CRADLES:
    piv = link(bpy.data.objects.new(f'piv{b}', None)); piv.location = W - Vector((0, 0, BR + 0.006)); piv.rotation_mode = 'QUATERNION'
    base = Vector((0, 0, 1)).rotation_difference(tng)
    cup_ = prim('uv_sphere', BRASS, radius=BR * 1.25, segments=24, ring_count=12, location=(0, 0, 0)); bpy.ops.object.shade_smooth()
    bm_ = bmesh.new(); bm_.from_mesh(cup_.data)
    bmesh.ops.delete(bm_, geom=[v for v in bm_.verts if v.co.z > -BR * 0.25], context='VERTS'); bm_.to_mesh(cup_.data); bm_.free()
    so = cup_.modifiers.new('s', 'SOLIDIFY'); so.thickness = 0.003
    cup_.parent = piv; cup_.location = (0, 0, BR + 0.006)
    rim = prim('torus', BRASS, major_radius=BR * 1.2, minor_radius=0.0025, location=(0, 0, 0)); rim.parent = piv; rim.location = (0, 0, BR * 0.78)
    radial = Vector((W.x, W.y, 0)).normalized()
    rod(W - Vector((0, 0, BR + 0.006)), Vector((0, 0, W.z - BR - 0.006)) + radial * 0.02, 0.003, WIRE, 8)
    prim('uv_sphere', WIRE, radius=0.007, location=W - Vector((0, 0, BR + 0.006)))
    ax = tng.normalized()
    for f in range(NF):
        tt = TT[f]; a = 0.0 if tt < trel - 0.08 else min(1.0, (tt - trel + 0.08) / 0.18)
        a = a * a * (3 - 2 * a) * (1.25 if tt < trel + 0.6 else 1.25 - 1.25 * min(1, (tt - trel - 0.6) / 0.5))
        if f == 0 or abs(tt - trel) < 1.5:
            piv.rotation_quaternion = Quaternion(ax, a); piv.keyframe_insert('rotation_quaternion', frame=f)
# finale: glass bowl on a brass stand
bowl = prim('uv_sphere', GLASS, radius=0.17, segments=48, ring_count=24, location=BOWL + Vector((0, 0, 0.17)))
bpy.ops.object.shade_smooth()
bm_ = bmesh.new(); bm_.from_mesh(bowl.data)
bmesh.ops.delete(bm_, geom=[v for v in bm_.verts if v.co.z > -0.05], context='VERTS'); bm_.to_mesh(bowl.data); bm_.free()
bowl.modifiers.new('s', 'SOLIDIFY').thickness = 0.006
prim('torus', BRASS, major_radius=0.162, minor_radius=0.005, location=BOWL + Vector((0, 0, 0.17 - 0.05)))
rod(BOWL - Vector((0, 0, 0.0)), BOWL - Vector((0, 0, 0.5)), 0.008, BRASS, 12)
prim('cylinder', BRASS, radius=0.07, depth=0.012, vertices=32, location=BOWL - Vector((0, 0, 0.5)))
rod(BOWL - Vector((0, 0, 0.25)), Vector((0, 0, BOWL.z - 0.25)), 0.005, BRASS, 8)

# ---------------- balls ----------------
BALLS = []
for b in range(NB):
    o = prim('uv_sphere', GLASS, radius=BR, segments=32, ring_count=16, location=(0, 0, -50)); bpy.ops.object.shade_smooth()
    core_m = mat(f'core{b}', BCOL[b], 0.15, 0.0, **{'Emission Strength': 2.2}); core_m.node_tree.nodes['Principled BSDF'].inputs['Emission Color'].default_value = (*BCOL[b], 1)
    c = prim('uv_sphere', core_m, radius=BR * 0.86, segments=24, ring_count=8, location=(0, 0, 0)); bpy.ops.object.shade_smooth()
    c.parent = o; c.location = (0, 0, 0); c.scale = (1.0, 1.0, 1.0)
    o.rotation_mode = 'QUATERNION'
    BALLS.append(o)
BPOS = np.zeros((NB, NF, 3)); BVIS = np.zeros((NB, NF), bool)
for b, o in enumerate(BALLS):
    q = Quaternion(); prev = None
    for f in range(NF):
        p = PATHS[b].pos(TT[f])
        if p is None:
            if f > 0 and BVIS[b, f - 1]:
                o.hide_render = True; o.keyframe_insert('hide_render', frame=f)
            continue
        BPOS[b, f] = p; BVIS[b, f] = True
        if prev is not None:
            d = p - prev; dh = Vector((d.x, d.y, 0))
            if dh.length > 1e-6:
                ax = Vector((0, 0, 1)).cross(dh).normalized(); q = Quaternion(ax, d.length / BR) @ q
        prev = p
        o.location = p; o.rotation_quaternion = q; o.hide_render = False
        o.keyframe_insert('location', frame=f); o.keyframe_insert('rotation_quaternion', frame=f)
        if f == 0 or not BVIS[b, f - 1]:
            o.keyframe_insert('hide_render', frame=f)
            if f > 0:
                o.hide_render = True; o.keyframe_insert('hide_render', frame=f - 1); o.hide_render = False
    if not BVIS[b, 0]:
        o.hide_render = True; o.keyframe_insert('hide_render', frame=0)

# ---------------- camera: follow shots on the newest ball, group orbits ----------------
def gaussian_filter1d(x, sig, axis=0, mode='nearest'):
    k = np.exp(-0.5 * (np.arange(-3 * sig, 3 * sig + 1) / sig) ** 2); k /= k.sum(); h = len(k) // 2
    xp = np.concatenate([np.repeat(x[:1], h, 0), x, np.repeat(x[-1:], h, 0)], 0)
    return np.stack([np.convolve(xp[:, c], k, mode='valid') for c in range(x.shape[1])], 1)
SM = np.stack([gaussian_filter1d(BPOS[b], 6, axis=0, mode='nearest') for b in range(NB)])
FIN = E['final']
cd = bpy.data.cameras.new('cam'); cd.sensor_fit = 'HORIZONTAL'; cd.sensor_width = 36; cd.clip_start = 0.02; cd.clip_end = 100
cd.dof.use_dof = True; cd.dof.aperture_fstop = 6.0
cam = link(bpy.data.objects.new('cam', cd)); sc.camera = cam
tgo = link(bpy.data.objects.new('tgt', None))
trk = cam.constraints.new('TRACK_TO'); trk.target = tgo; trk.track_axis = 'TRACK_NEGATIVE_Z'; trk.up_axis = 'UP_Y'
# (start, end, mode, ball, dist, height, az0, az1, lens)
# camera plan: each variation gets 3-4 shots cycling through close follows and group views
STY = [('follow', 0, 0.6, 0.15, -1.9, -1.4, 52), ('group', 0, 0.85, 0.3, -0.6, 0.2, 45), ('follow', 1, 0.6, 0.18, -1.2, -0.7, 52),
       ('group', 0, 0.6, 0.8, -0.4, 0.4, 35), ('follow', 0, 0.45, 0.1, 0.8, 1.3, 58), ('group', 0, 0.9, 0.08, 0.6, -0.2, 45),
       ('follow', 2, 0.55, 0.15, -2.4, -2.0, 55)]
SH = []; si = 0
bounds = [0.0] + [sct[0] - 0.4 for sct in SECT[1:]] + [FIN - 0.1]
for k in range(len(SECT)):
    t0, t1 = bounds[k], bounds[k + 1]; n_ = max(2, int(round((t1 - t0) / 8.0)))
    for j in range(n_):
        st = STY[si % len(STY)]; si += 1
        if st[1] == 2 and not any(e[2] == 2 and t0 <= e[0] < t1 for e in NOTES): st = STY[0]
        SH.append((t0 + (t1 - t0) * j / n_, t0 + (t1 - t0) * (j + 1) / n_) + st)
SH.append((FIN - 0.1, T_END + 0.1, 'group', 0, 1.0, 0.3, 0.0, 0.6, 40))
for f in range(NF):
    t = TT[f]
    for t0, t1, mode, b, dist, h, a0, a1, lens in SH:
        if t0 <= t < t1: break
    a = (t - t0) / (t1 - t0); az = a0 + (a1 - a0) * ss(a)
    if mode == 'follow':
        look = Vector(SM[b, f]) if BVIS[b, f] else anchor(b, t)
        # view from outside the spiral, a little ahead along the ball's travel
        rad = Vector((look.x, look.y, 0)).normalized()
        off = Vector((math.cos(az), math.sin(az), 0))
        off = (rad * 0.8 + off * 0.6).normalized()
        pos = look + off * dist + Vector((0, 0, h)); foc = (pos - Vector(BPOS[b, f])).length if BVIS[b, f] else dist
    else:
        if t >= FIN - 0.1:      # finale: hold on the last plates and pull back while the balls fly out
            look0 = Vector((0.6 * math.cos(OMG * FIN), 0.6 * math.sin(OMG * FIN), Z0 - CZ * FIN))
            look = look0.lerp(BOWL + Vector((0, 0, 0.08)), ss(min(1, a * 1.6)))
            radial = Vector((math.cos(THF - 0.5 + 0.3 * az), math.sin(THF - 0.5 + 0.3 * az), 0))
            pos = look + radial * (0.75 - 0.2 * ss(a)) + Vector((0, 0, 0.32 - 0.04 * ss(a)))
            foc = (pos - look).length
            cam.location = pos; tgo.location = look; cd.lens = lens; cd.dof.focus_distance = foc
            cam.keyframe_insert('location', frame=f); tgo.keyframe_insert('location', frame=f)
            cd.keyframe_insert('lens', frame=f); cd.keyframe_insert('dof.focus_distance', frame=f)
            continue
        zc = Z0 - CZ * t; th = OMG * t
        vis = [Vector(SM[k, f]) for k in range(NB) if BVIS[k, f]]
        look = sum(vis, Vector()) / len(vis) if vis else Vector((0.6 * math.cos(th), 0.6 * math.sin(th), zc))
        look = look.lerp(Vector((0.6 * math.cos(th), 0.6 * math.sin(th), zc)), 0.5)
        radial = Vector((math.cos(th + az * 0.5), math.sin(th + az * 0.5), 0))
        pos = look + radial * dist + Vector((0, 0, h))
        foc = (pos - look).length - 0.3
    cam.location = pos; tgo.location = look; cd.lens = lens; cd.dof.focus_distance = foc
    cam.keyframe_insert('location', frame=f); tgo.keyframe_insert('location', frame=f)
    cd.keyframe_insert('lens', frame=f); cd.keyframe_insert('dof.focus_distance', frame=f)
from bpy_extras import anim_utils
for idb in (cam, tgo, cd):
    ad = idb.animation_data
    cb = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
    for fc in cb.fcurves:
        for kp in fc.keyframe_points: kp.interpolation = 'CONSTANT'
try: sc.render.motion_blur_position = 'START'
except Exception: pass

# ---------------- twinkle: a little star flash on every hit ----------------
SPM = bpy.data.materials.new('spark'); SPM.use_nodes = True; nt_ = SPM.node_tree; nt_.nodes.clear()
oi_ = nt_.nodes.new('ShaderNodeObjectInfo'); em_ = nt_.nodes.new('ShaderNodeEmission'); tr_ = nt_.nodes.new('ShaderNodeBsdfTransparent')
mxs = nt_.nodes.new('ShaderNodeMixShader'); mu_ = nt_.nodes.new('ShaderNodeMath'); mu_.operation = 'MULTIPLY'; mu_.inputs[1].default_value = 25
out_ = nt_.nodes.new('ShaderNodeOutputMaterial')
nt_.links.new(oi_.outputs['Alpha'], mu_.inputs[0]); nt_.links.new(mu_.outputs[0], em_.inputs['Strength']); nt_.links.new(oi_.outputs['Color'], em_.inputs['Color'])
nt_.links.new(oi_.outputs['Alpha'], mxs.inputs[0]); nt_.links.new(tr_.outputs[0], mxs.inputs[1]); nt_.links.new(em_.outputs[0], mxs.inputs[2]); nt_.links.new(mxs.outputs[0], out_.inputs[0])
bms_ = bmesh.new()
for k in range(8):                        # 4 long + 4 short rays
    a_ = math.pi / 4 * k; L = 1.0 if k % 2 == 0 else 0.45; w_ = 0.07
    d = Vector((math.cos(a_), math.sin(a_), 0)); nrm = Vector((-d.y, d.x, 0))
    v1 = bms_.verts.new(nrm * w_); v2 = bms_.verts.new(d * L); v3 = bms_.verts.new(-nrm * w_); bms_.faces.new((v1, v2, v3))
SPME = bpy.data.meshes.new('spark'); bms_.to_mesh(SPME); bms_.free(); SPME.materials.append(SPM)
hits = []
for t, d, ball, p, v in NOTES:
    f = int(round(t * FPS))
    if 0 <= f < NF and BVIS[ball, f]: hits.append((t, ball, Vector(BPOS[ball, f])))
POOL = []
for t, ball, P in hits:
    for rec in POOL:
        if rec[1] < t - 0.05: break
    else:
        o = link(bpy.data.objects.new('spark', SPME)); o.color = (1, 1, 1, 0); o.scale = (0, 0, 0)
        o.constraints.new('TRACK_TO').target = cam; o.constraints[0].track_axis = 'TRACK_Z'; o.constraints[0].up_axis = 'UP_Y'
        rec = [o, -9, []]; POOL.append(rec)
    rec[1] = t + 0.32; rec[2].append((t, ball, P))
for o, _, lst in POOL:
    for t, ball, P in lst:
        f0 = int(round(t * FPS)); col = BCOL[ball]
        for df, sc_, al in ((-1, 0.0, 0.0), (0, 0.06, 1.0), (2, 0.045, 0.7), (5, 0.02, 0.3), (9, 0.0, 0.0)):
            f = f0 + df
            if 0 <= f < NF:
                o.location = P; o.scale = (sc_, sc_, sc_); o.color = (*[0.6 + 0.4 * c for c in col], al)
                o.keyframe_insert('location', frame=f); o.keyframe_insert('scale', frame=f); o.keyframe_insert('color', frame=f)
print('sparks', len(hits), 'pool', len(POOL))
# ---------------- variation titles (bottom of the frame, fade in/out) ----------------
TXM = bpy.data.materials.new('title'); TXM.use_nodes = True; nt_ = TXM.node_tree; nt_.nodes.clear()
oi_ = nt_.nodes.new('ShaderNodeObjectInfo'); em_ = nt_.nodes.new('ShaderNodeEmission'); tr_ = nt_.nodes.new('ShaderNodeBsdfTransparent')
mxs = nt_.nodes.new('ShaderNodeMixShader'); out_ = nt_.nodes.new('ShaderNodeOutputMaterial'); em_.inputs['Strength'].default_value = 2.5
em_.inputs['Color'].default_value = (1.0, 0.92, 0.7, 1)
nt_.links.new(oi_.outputs['Alpha'], mxs.inputs[0]); nt_.links.new(tr_.outputs[0], mxs.inputs[1]); nt_.links.new(em_.outputs[0], mxs.inputs[2]); nt_.links.new(mxs.outputs[0], out_.inputs[0])
LBL = {'Theme': 'Theme'}
for k, (ta, tb, nm) in enumerate(SECT):
    txt = 'Theme' if nm == 'Theme' else 'Variation ' + str(k)
    cu_ = bpy.data.curves.new(f'tt{k}', 'FONT'); cu_.body = txt; cu_.align_x = 'CENTER'; cu_.size = 0.05; cu_.extrude = 0.0
    to = link(bpy.data.objects.new(f'tt{k}', cu_)); to.data.materials.append(TXM); to.parent = cam
    to.location = (0, -0.29, -1.0); to.color = (1, 1, 1, 0)
    for tt_, al in ((ta - 0.6, 0.0), (ta - 0.2, 1.0), (ta + 3.0, 1.0), (ta + 3.8, 0.0)):
        f = int(tt_ * FPS)
        if 0 <= f < NF: to.color = (1, 1, 1, al); to.keyframe_insert('color', frame=f)
print('objects', len(bpy.data.objects))
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
