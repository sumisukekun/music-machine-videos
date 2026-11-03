"""Can-can v2 - precision ball machine.
14 aiming turrets (yaw ring gears + pinions, recoil) fire balls in exact ballistic arcs; each hit is a note.
Balls drop into rails, run down chutes, are metered by collectors, lifted and returned to the magazine.
Clockwork gear train + metronome step on every beat, cam-driven hi-hat sticks play 8ths/16ths."""
import os, sys, json, math, colorsys, time
import bpy, bmesh
import numpy as np
from mathutils import Vector

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json'))
FPS = 30; T_END = 66.0; NF = int(T_END * FPS); FINAL = E['final']; BEAT = E['beat']; TB0 = E['t0']
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
for k, v in (('denoiser', 'OPENIMAGEDENOISE'), ('denoising_quality', 'FAST'), ('denoising_prefilter', 'FAST'),
             ('use_fast_gi', True), ('ao_bounces_render', 1)):
    try: setattr(sc.cycles, k, v)
    except Exception: pass
sc.cycles.max_bounces = 4; sc.cycles.diffuse_bounces = 1; sc.cycles.glossy_bounces = 2
sc.cycles.caustics_reflective = False; sc.cycles.caustics_refractive = False
sc.cycles.sample_clamp_indirect = 4
sc.render.resolution_x, sc.render.resolution_y = 720, 1280
sc.render.resolution_percentage = int(os.environ.get('PCT', 66))
sc.render.fps = FPS
sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.45
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


def rod(p0, p1, r, mt, verts=10):
    p0, p1 = Vector(p0), Vector(p1); d = p1 - p0
    o = prim('cylinder', mt, radius=r, depth=d.length, vertices=verts, location=(p0 + p1) / 2)
    o.rotation_mode = 'QUATERNION'; o.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d.normalized())
    return o


def gear_mesh(name, n, r, plane='XZ', inner=0.62):
    m = 2 * r / n; ra, rd = r + 0.9 * m, r - 1.1 * m; ri = rd * inner; pa = 2 * math.pi / n
    outer = []
    for k in range(n):
        a = k * pa
        outer += [(rd, a - 0.27 * pa), (ra, a - 0.13 * pa), (ra, a + 0.13 * pa), (rd, a + 0.27 * pa)]
    bm = bmesh.new()
    def P(rr, a):
        c, s = rr * math.cos(a), rr * math.sin(a)
        return (c, 0, s) if plane == 'XZ' else (c, s, 0)
    vo = [bm.verts.new(P(rr, a)) for rr, a in outer]
    vi = [bm.verts.new(P(ri, a)) for _, a in outer]
    N = len(vo)
    for i in range(N):
        bm.faces.new((vo[i], vo[(i + 1) % N], vi[(i + 1) % N], vi[i]))
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    return me, ri


BRASS = mat('brass', (1.0, 0.72, 0.32), 0.22, 1.0)
GOLD = mat('gold', (1.0, 0.75, 0.3), 0.18, 1.0)
COPPER = mat('copper', (0.95, 0.5, 0.3), 0.22, 1.0)
STEEL = mat('steel', (0.75, 0.76, 0.8), 0.25, 1.0)
DARKM = mat('dark', (0.035, 0.035, 0.045), 0.35, 0.7)
GUNM = mat('gunmetal', (0.12, 0.13, 0.15), 0.3, 1.0)


def gear(name, n, r, loc, plane='XZ', mt=BRASS, depth=0.08, spokes=5):
    me, ri = gear_mesh(name, n, r, plane)
    o = link(bpy.data.objects.new(name, me)); o.location = loc; me.materials.append(mt)
    so = o.modifiers.new('s', 'SOLIDIFY'); so.thickness = depth; so.offset = 0
    ax = (0, 1, 0) if plane == 'XZ' else (0, 0, 1)
    # hub + spokes as children
    hub = prim('cylinder', mt, radius=0.22 * ri + 0.03, depth=depth * 1.8, vertices=24, location=(0, 0, 0))
    if plane == 'XZ': hub.rotation_euler = (math.pi / 2, 0, 0)
    hub.parent = o
    for s in range(spokes):
        a = 2 * math.pi * s / spokes
        sp = prim('cube', mt, size=1, location=((ri * 0.55) * math.cos(a), 0, (ri * 0.55) * math.sin(a)) if plane == 'XZ' else ((ri * 0.55) * math.cos(a), (ri * 0.55) * math.sin(a), 0))
        if plane == 'XZ': sp.scale = (ri * 1.0, depth * 0.8, ri * 0.14); sp.rotation_euler = (0, -a, 0)
        else: sp.scale = (ri * 1.0, ri * 0.14, depth * 0.8); sp.rotation_euler = (0, 0, a)
        sp.parent = o
    return o


# ---------------- stage ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bgn = w.node_tree.nodes['Background']; bgn.inputs['Color'].default_value = (0.03, 0.015, 0.03, 1); bgn.inputs['Strength'].default_value = 1
FLOOR = mat('floor', (0.33, 0.17, 0.08), 0.32)
prim('plane', FLOOR, size=40, location=(0, 4, 0))
WALL = mat('backwall', (0.05, 0.016, 0.03), 0.8)
prim('plane', WALL, size=40, location=(0, 6.5, 10), rotation=(math.pi / 2, 0, 0))
CURT = mat('curtain', (0.45, 0.02, 0.04), 0.6, **{'Sheen Weight': 0.6})
for sx in (-1, 1):
    c = prim('grid', CURT, x_subdivisions=60, y_subdivisions=2, size=1, location=(sx * 6.7, 1.0, 7.0), rotation=(math.pi / 2, 0, 0))
    c.scale = (3.0, 16, 1)
    for v in c.data.vertices: v.co.z += 0.06 * math.sin(v.co.x * 60)
    bpy.ops.object.shade_smooth()
v_ = prim('cube', CURT, size=1, location=(0, 1.2, 15.6)); v_.scale = (16, 0.3, 1.4)

for sx, col, e, aim in ((-4, (1, 0.8, 0.6), 5200, (-1.2, 2, 5)), (4, (0.8, 0.85, 1), 5200, (1.2, 2, 5)), (0, (1, 0.9, 0.75), 3800, (0, 2, 7)),
                        (0.5, (1, 0.85, 0.6), 2200, (0, 3, 11.3)), (0, (1, 0.9, 0.8), 1600, (0, -1.5, 0.8))):
    ld = bpy.data.lights.new('spot', 'SPOT'); ld.energy = e; ld.color = col; ld.spot_size = math.radians(55); ld.spot_blend = 0.4
    ld.shadow_soft_size = 0.4
    o = link(bpy.data.objects.new('spot', ld)); o.location = (sx, -6, 12) if aim[2] < 10 else (sx, -4, 15)
    if aim[1] < 0: o.location = (0, -6, 6)
    tr = o.constraints.new('TRACK_TO'); e_ = link(bpy.data.objects.new('spt', None)); e_.location = aim; tr.target = e_
    tr.track_axis = 'TRACK_NEGATIVE_Z'; tr.up_axis = 'UP_Y'
fl = bpy.data.lights.new('fill', 'AREA'); fl.energy = 900; fl.size = 8; fl.color = (0.8, 0.6, 1.0)
link(bpy.data.objects.new('fill', fl)).location = (0, -9, 3)

# ---------------- instruments ----------------
TARGET = {}      # key -> hit point (ball centre)
INST = []        # (object, key, rgb) flashing
BR = 0.13        # ball radius


def bar_row(kind, pitches, z, y, x0, x1, L0, L1, wdt, thk, col, hue0, hue1, metal=0.0, gain=8):
    n = len(pitches); objs = {}
    for i, p in enumerate(pitches):
        x = x0 + (x1 - x0) * i / max(1, n - 1); L = L0 + (L1 - L0) * i / max(1, n - 1); wdt = min(wdt, (x1 - x0) / max(1, n - 1) * 0.88)
        o = prim('cube', None, size=1, location=(x, y, z)); o.scale = (wdt, thk, L); bpy.ops.object.transform_apply(scale=True)
        bv = o.modifiers.new('b', 'BEVEL'); bv.width = 0.02; bv.segments = 2
        rgb = colorsys.hsv_to_rgb(hue0 + (hue1 - hue0) * i / max(1, n - 1), 0.75, 1.0)
        o.data.materials.append(glowmat(f'{kind}{p}', col, 0.3 if metal else 0.38, metal, gain)); o.color = (*rgb, 0)
        TARGET[(kind, p)] = Vector((x, y - thk / 2 - BR, z + 0.12 * L)); INST.append((o, (kind, p), rgb)); objs[p] = (o, x, L)
        for sz in (1, -1):
            prim('cylinder', GOLD, radius=0.022, depth=thk + 0.16, vertices=8, location=(x, y, z + sz * L * 0.3), rotation=(math.pi / 2, 0, 0))
    for zz in (z + L0 * 0.3, z - L0 * 0.3):
        rod((x0 - 0.35, y + thk / 2 + 0.05, zz), (x1 + 0.35, y + thk / 2 + 0.05, zz), 0.035, BRASS)
    return objs

ROSE = (0.5, 0.2, 0.09); ALU = (0.82, 0.83, 0.86)
harm_p = sorted({p for _, p in E['harm']}); mel_p = sorted({p for _, p in E['mel']} | {p for _, p in E['gliss']})
chord_p = sorted({p for _, ps in E['chord'] for p in ps}); bass_p = sorted({p for _, p in E['bass']})
HARM = bar_row('harm', harm_p, 9.75, 2.45, -3.3, 3.3, 0.85, 0.5, 0.32, 0.09, ALU, 0.48, 0.62, metal=1.0, gain=7)
MEL = bar_row('mel', mel_p, 7.4, 2.6, -4.0, 4.0, 1.9, 1.0, 0.42, 0.14, ROSE, 0.0, 0.85)
CHORD = bar_row('chord', chord_p, 5.2, 2.3, -3.7, 3.7, 1.3, 0.8, 0.4, 0.12, (1.0, 0.8, 0.45), 0.08, 0.16, metal=1.0, gain=6)
BASS = bar_row('bass', bass_p, 2.95, 2.0, -4.3, -0.3, 2.0, 1.35, 0.38, 0.16, (0.3, 0.12, 0.06), 0.55, 0.66)
for p, (o, x, L) in BASS.items():   # resonators
    prim('cylinder', GOLD, radius=0.15, depth=1.5, vertices=20, location=(x, 2.3, 2.95 - L / 2 - 0.7))
for p, (o, x, L) in CHORD.items():  # vibraphone motor fans (spin)
    pass

# drums
SHELL = mat('drumshell', (0.62, 0.02, 0.05), 0.22, **{'Coat Weight': 1.0})
HEADM = glowmat('head', (0.92, 0.9, 0.85), 0.5, gain=3)
CYM = glowmat('cym', (1.0, 0.72, 0.25), 0.2, metal=1.0, gain=4)


def drum(name, loc, r, depth, tilt, key, rgb):
    piv = link(bpy.data.objects.new(name, None)); piv.location = loc; piv.rotation_euler = (tilt, 0, 0)
    s = prim('cylinder', SHELL, radius=r, depth=depth, vertices=40); s.parent = piv
    h = prim('cylinder', HEADM, radius=r * 0.96, depth=0.02, vertices=40, location=(0, 0, depth / 2 + 0.005)); h.parent = piv
    rim = prim('torus', STEEL, major_radius=r * 1.0, minor_radius=0.025, location=(0, 0, depth / 2)); rim.parent = piv
    for k in range(8):
        a = 2 * math.pi * k / 8
        lug = prim('cube', STEEL, size=1, location=(r * math.cos(a), r * math.sin(a), 0)); lug.scale = (0.05, 0.05, depth * 0.7); lug.parent = piv
    h.color = (*rgb, 0); INST.append((h, key, rgb))
    nrm = piv.matrix_world.to_3x3() @ Vector((0, 0, 1)) if False else Vector((0, -math.sin(tilt), math.cos(tilt)))
    TARGET[key] = Vector(loc) + nrm * (depth / 2 + BR + 0.01)
    return piv

drum('kick', (2.4, 0.9, 1.0), 0.9, 0.7, math.pi / 2, ('kick', 0), (1, 0.3, 0.2))
rod((2.4, 1.2, 0.0), (2.4, 1.2, 0.15), 0.3, DARKM)
drum('snare', (4.0, 1.0, 1.75), 0.42, 0.25, 0.0, ('snare', 0), (1, 0.8, 0.3)); rod((4.0, 1.0, 0), (4.0, 1.0, 1.62), 0.03, STEEL)
TOMS = {50: ((1.15, 1.55, 3.45), 0.34), 47: ((2.15, 1.6, 3.55), 0.39), 45: ((3.2, 1.5, 3.3), 0.44)}
for k, (loc, r) in TOMS.items():
    drum(f'tom{k}', loc, r, 0.36, math.radians(32), ('tom', k), (1, 0.45, 0.85))
    rod((loc[0], loc[1] + 0.25, loc[2] - 0.2), (loc[0], 1.3, 1.9 if loc[0] < 3.3 else 0), 0.03, STEEL)
# crash cymbals (L/R)
CPIV = []
for side, x in ((0, -4.1), (1, 4.1)):
    piv = link(bpy.data.objects.new(f'cpiv{side}', None)); piv.location = (x, 1.8, 11.0); piv.rotation_mode = 'XYZ'
    cy = prim('cone', CYM, radius1=0.78, radius2=0.08, depth=0.08, vertices=48); cy.parent = piv; cy.rotation_euler = (math.radians(68), 0, 0)
    cy.color = (1, 0.85, 0.4, 0); INST.append((cy, ('crash', side), (1, 0.85, 0.4)))
    rod((x, 1.9, 10.9), (x * 1.08, 2.4, 0), 0.03, STEEL)
    TARGET[('crash', side)] = Vector((x, 1.55, 11.0))
    CPIV.append(piv)
# hi-hat: two cymbals + cam-driven sticks
HX, HY, HZ = 4.05, 0.85, 3.0
rod((HX, HY, 0), (HX, HY, HZ + 0.15), 0.025, STEEL)
hat_bot = prim('cone', CYM, radius1=0.4, radius2=0.05, depth=0.05, vertices=40, location=(HX, HY, HZ - 0.04), rotation=(math.pi, 0, 0))
hat_top = prim('cone', CYM, radius1=0.4, radius2=0.05, depth=0.05, vertices=40, location=(HX, HY, HZ + 0.04))
hat_top.color = (1, 0.9, 0.5, 0); hat_bot.color = (1, 0.9, 0.5, 0)
STICKS = []
for j, dx in enumerate((-0.14, 0.14)):
    pv = link(bpy.data.objects.new(f'stickpiv{j}', None)); pv.location = (HX + dx, HY - 0.75, HZ + 0.55); pv.rotation_mode = 'XYZ'
    tip = Vector((HX + dx * 0.4, HY - 0.05, HZ + 0.12)) - pv.location
    st = rod((0, 0, 0), tip, 0.022, mat(f'stick{j}', (0.9, 0.75, 0.5), 0.5)); st.parent = pv
    bd = prim('uv_sphere', mat(f'bead{j}', (0.95, 0.85, 0.6), 0.4), radius=0.045, segments=12, ring_count=8, location=tip); bd.parent = pv
    STICKS.append(pv)
camshaft = link(bpy.data.objects.new('camshaft', None)); camshaft.location = (HX, HY - 0.95, HZ + 0.3); camshaft.rotation_mode = 'XYZ'
rod((HX - 0.45, HY - 0.95, HZ + 0.3), (HX + 0.45, HY - 0.95, HZ + 0.3), 0.03, STEEL)
for j, dx in enumerate((-0.14, 0.14)):
    cm = prim('cylinder', GUNM, radius=0.09, depth=0.05, vertices=3, location=(dx, 0, 0), rotation=(0, math.pi / 2, 0)); cm.parent = camshaft
    cm.rotation_euler = (j * math.pi / 3, math.pi / 2, 0)
prim('cube', GUNM, size=1, location=(HX, HY - 0.95, HZ - 0.2)).scale = (0.9, 0.2, 0.9)

# ---------------- clockwork: gear train + metronome ----------------
GEARS = []   # (obj, Q, ratio)
GY = 3.4
def add_gear(name, n, r, loc, parent=None, phi=0.0, mt=BRASS):
    o = gear(name, n, r, loc, mt=mt)
    if parent is None: Q, ratio = 0.0, 1.0
    else:
        po, pQ, pr, pn = parent
        Q = phi + math.pi - (2 * math.pi / n) * (0.5 - (phi - pQ) * pn / (2 * math.pi))
        ratio = -pr * pn / n
    rec = (o, Q, ratio, n); GEARS.append(rec); return rec

def place(parent_rec, n, r, ang, name, mt=BRASS):
    po = parent_rec[0]; pr_r = parent_rec[0]['r']
    c = Vector(po.location) + Vector((math.cos(ang), 0, math.sin(ang))) * (pr_r + r)
    rec = add_gear(name, n, r, c, (po, parent_rec[1], parent_rec[2], parent_rec[3]), phi=ang, mt=mt)
    rec[0]['r'] = r; return rec

m0 = 0.075
g0 = add_gear('g0', 36, 36 * m0 / 2, (0, GY, 11.6)); g0[0]['r'] = 36 * m0 / 2
gL = place(g0, 22, 22 * m0 / 2, math.radians(200), 'gL')
gL2 = place(gL, 14, 14 * m0 / 2, math.radians(140), 'gL2', COPPER)
gR = place(g0, 18, 18 * m0 / 2, math.radians(-15), 'gR', COPPER)
gR2 = place(gR, 26, 26 * m0 / 2, math.radians(30), 'gR2')
gL3 = place(gL2, 16, 16 * m0 / 2, math.radians(95), 'gL3')
gR3 = place(gR2, 12, 12 * m0 / 2, math.radians(-60), 'gR3', COPPER)
for o, *_ in GEARS:
    rod(o.location + Vector((0, -0.12, 0)), o.location + Vector((0, 3.0, 0)), 0.04, STEEL)
# escapement anchor
anchor = link(bpy.data.objects.new('anchor', None)); anchor.location = (0, GY - 0.12, 11.6 + 36 * m0 / 2 + 0.45); anchor.rotation_mode = 'XYZ'
for s in (-1, 1):
    a = rod((0, 0, 0), (s * 0.5, 0, -0.35), 0.03, STEEL); a.parent = anchor
# metronome
MET = mat('metbody', (0.18, 0.07, 0.03), 0.35, **{'Coat Weight': 1.0})
mb = prim('cone', MET, radius1=0.55, radius2=0.22, depth=2.0, vertices=4, location=(0, 2.0, 11.3)); mb.rotation_euler = (0, 0, math.pi / 4)
metpiv = link(bpy.data.objects.new('metpiv', None)); metpiv.location = (0, 1.5, 10.5); metpiv.rotation_mode = 'XYZ'
mr = rod((0, 0, 0), (0, 0, 1.9), 0.022, STEEL); mr.parent = metpiv
mw = prim('cube', GOLD, size=1, location=(0, 0, 1.25)); mw.scale = (0.2, 0.07, 0.16); mw.parent = metpiv
mlamp = prim('uv_sphere', glowmat('mlamp', (1, 0.3, 0.2), 0.3, gain=10), radius=0.06, location=(0, 1.88, 12.38)); mlamp.color = (1, 0.25, 0.15, 0)

# ---------------- turrets ----------------
PIV_H = {'back': (-1.1, 1.25), 'front': (-2.0, 0.82)}
BARREL = 0.62


class Turret:
    def __init__(self, name, x, tier, kinds, mortar=False):
        y, z = PIV_H[tier]; self.name = name; self.P = Vector((x, y, z)); self.kinds = kinds; self.shots = []; self.mortar = mortar
        base_z = z - 0.32
        prim('cylinder', GUNM, radius=0.2, depth=base_z, vertices=24, location=(x, y, base_z / 2))
        self.yaw = link(bpy.data.objects.new(name + 'yaw', None)); self.yaw.location = (x, y, base_z); self.yaw.rotation_mode = 'XYZ'
        if not mortar:
            ring = gear(name + 'ring', 28, 0.25, (0, 0, 0.03), plane='XY', mt=BRASS, depth=0.05, spokes=4); ring.parent = self.yaw
            self.pin = gear(name + 'pin', 10, 0.25 * 10 / 28, (x + 0.25 + 0.25 * 10 / 28, y, base_z + 0.03), plane='XY', mt=STEEL, depth=0.05, spokes=3)
            self.pin.rotation_mode = 'XYZ'
            for s in (-1, 1):
                pl = prim('cube', GUNM, size=1, location=(s * 0.13, 0, 0.18)); pl.scale = (0.035, 0.22, 0.3); pl.parent = self.yaw
        self.pitch = link(bpy.data.objects.new(name + 'pitch', None)); self.pitch.parent = self.yaw; self.pitch.location = (0, 0, z - base_z); self.pitch.rotation_mode = 'XYZ'
        self.slide = link(bpy.data.objects.new(name + 'slide', None)); self.slide.parent = self.pitch
        rb = 0.075 if not mortar else 0.12
        b = rod((0, -0.18, 0), (0, BARREL, 0), rb, COPPER if not mortar else GUNM, 16); b.parent = self.slide
        jk = prim('cylinder', BRASS, radius=rb + 0.025, depth=0.12, vertices=16, location=(0, 0.05, 0), rotation=(math.pi / 2, 0, 0)); jk.parent = self.slide
        cw = prim('cube', GUNM, size=1, location=(0, -0.26, 0)); cw.scale = (0.12, 0.12, 0.12); cw.parent = self.slide
        self.rim = prim('torus', glowmat(name + 'rim', (1, 0.7, 0.3), 0.2, metal=1.0, gain=12), major_radius=rb + 0.012, minor_radius=0.022,
                        location=(0, BARREL, 0), rotation=(math.pi / 2, 0, 0))
        self.rim.parent = self.slide; self.rim.color = (1, 0.75, 0.35, 0)
        self.lamp = prim('uv_sphere', glowmat(name + 'lamp', (0.2, 1, 0.3), 0.3, gain=10), radius=0.035, segments=10, ring_count=6, location=(0, -0.2, 0.33))
        self.lamp.parent = self.yaw; self.lamp.color = (0.2, 1, 0.35, 0)
        if mortar:
            for s in (-1, 1):
                pl = prim('cube', GUNM, size=1, location=(s * 0.17, 0, 0.2)); pl.scale = (0.04, 0.3, 0.35); pl.parent = self.yaw

    def solve(self, T):
        d = (T - self.P).normalized()
        for _ in range(6):
            M = self.P + d * BARREL
            v0 = (T - M) / TAU + Vector((0, 0, 0.5 * G * TAU)); d = v0.normalized()
        M = self.P + d * BARREL
        return M, (T - M) / TAU + Vector((0, 0, 0.5 * G * TAU)), d

    def add(self, t, key):
        M, v0, d = self.solve(TARGET[key]); self.shots.append((t - TAU, key, M, v0, d)); return M, v0

    def animate(self):
        sh = sorted(self.shots, key=lambda s: s[0])
        if not sh: return
        ang = lambda d: (math.atan2(-d.x, d.y), math.asin(max(-1, min(1, d.z))))
        A = [ang(s[4]) for s in sh]; F = [s[0] for s in sh]
        yaw = np.zeros(NF); pit = np.zeros(NF); rec = np.zeros(NF); glow = np.zeros(NF); lamp = np.zeros(NF)
        rest = (0.0, math.radians(55))
        for f, t in enumerate(TT):
            i = np.searchsorted(F, t + 1e-9) - 1          # last fired shot index
            if i < 0:
                a = ss((t - (F[0] - 0.9)) / 0.6); cur = (rest[0] + (A[0][0] - rest[0]) * a, rest[1] + (A[0][1] - rest[1]) * a)
            elif i >= len(F) - 1:
                cur = A[-1]
            else:
                gap = F[i + 1] - F[i]
                if gap < 0.14:
                    a = (t - F[i]) / gap
                else:
                    m0_ = F[i] + 0.05; m1_ = min(F[i] + 0.05 + 0.24, F[i + 1] - 0.04)
                    a = ss((t - m0_) / max(1e-3, m1_ - m0_))
                cur = (A[i][0] + (A[i + 1][0] - A[i][0]) * a, A[i][1] + (A[i + 1][1] - A[i][1]) * a)
            yaw[f], pit[f] = cur
            if i >= 0:
                dt = t - F[i]; rec[f] = 0.1 * math.exp(-dt / 0.06) * (1 if dt < 0.5 else 0)
                glow[f] = math.exp(-dt / 0.05) if dt < 0.3 else 0
                lamp[f] = math.exp(-dt / 0.12) if dt < 0.6 else 0
        if self.mortar:
            yaw[:] = A[0][0]; pit[:] = A[0][1]
        key_series(self.yaw, 'rotation_euler', 2, yaw)
        key_series(self.pitch, 'rotation_euler', 0, pit)
        key_series(self.slide, 'location', 1, -rec)
        key_series(self.rim, 'color', 3, glow, 1e-3)
        key_series(self.lamp, 'color', 3, lamp, 1e-3)
        if not self.mortar: key_series(self.pin, 'rotation_euler', 2, -yaw * 28 / 10)


T = {}
for name, x, tier, kinds, mort in (
        ('B0', -3.05, 'back', 'bass', 0), ('B1', -2.4, 'back', 'bass', 0), ('H0', -1.62, 'back', 'harm', 0), ('M0', -0.82, 'back', 'mel', 0),
        ('GL', 0.0, 'back', 'gliss', 0), ('M1', 0.82, 'back', 'mel', 0), ('H1', 1.62, 'back', 'harm', 0), ('TM', 2.3, 'back', 'tom', 0),
        ('XR', 3.05, 'back', 'crashR', 1),
        ('XL', -2.95, 'front', 'crashL', 1), ('C0', -1.55, 'front', 'chord', 0), ('C1', -0.52, 'front', 'chord', 0), ('C2', 0.52, 'front', 'chord', 0),
        ('M2', 1.55, 'front', 'mel', 0), ('K', 2.4, 'front', 'kick', 1), ('S', 3.05, 'front', 'snare', 1)):
    T[name] = Turret(name, x, tier, kinds, bool(mort))
# turret platform + magazine feed pipes
prim('cube', DARKM, size=1, location=(0, -1.55, 0.06)).scale = (7.2, 1.7, 0.12)
MAG = Vector((-3.55, -2.6, 2.25))
mag = prim('cube', GUNM, size=1, location=MAG); mag.scale = (0.5, 0.5, 0.42)
prim('cone', BRASS, radius1=0.12, radius2=0.3, depth=0.25, vertices=20, location=MAG + Vector((0, 0, 0.32)))
rod(MAG + Vector((0.25, 0.1, -0.1)), (-3.3, -1.55, 0.3), 0.05, BRASS)
rod((-3.3, -1.55, 0.3), (3.3, -1.55, 0.3), 0.05, BRASS)
for t_ in T.values(): rod((t_.P.x, -1.55, 0.3), (t_.P.x, t_.P.y, 0.3), 0.035, BRASS)

# ---------------- shots ----------------
BALLM = {k: glowmat('b' + k, c, 0.12, metal=1.0, gain=g) for k, c, g in (
    ('mel', (1.0, 0.78, 0.3), 0.8), ('gliss', (1.0, 0.95, 0.85), 2.5), ('harm', (0.5, 0.95, 1.0), 0.8), ('chord', (1.0, 0.55, 0.15), 0.8),
    ('bass', (0.3, 0.55, 1.0), 0.8), ('kick', (1.0, 0.3, 0.25), 0.8), ('snare', (1.0, 0.95, 0.9), 1.0), ('tom', (1.0, 0.4, 0.85), 0.8),
    ('crash', (1.0, 0.9, 0.5), 2.0))}
shots = []   # (hit time, kind(for ball), key, turret)
rr = {'mel': 0}
for i, (t, p) in enumerate(E['mel']):
    shots.append((t, 'mel', ('mel', p), ['M0', 'M1', 'M2'][i % 3]))
for t, p in E['gliss']: shots.append((t, 'gliss', ('mel', p), 'GL'))
for i, (t, p) in enumerate(E['harm']): shots.append((t, 'harm', ('harm', p), ['H0', 'H1'][i % 2]))
for t, ps in E['chord']:
    for j, p in enumerate(sorted(ps)[-3:]): shots.append((t, 'chord', ('chord', p), ['C0', 'C1', 'C2'][j + 3 - len(ps[-3:])] if len(ps) < 3 else ['C0', 'C1', 'C2'][j]))
for i, (t, p) in enumerate(E['bass']): shots.append((t, 'bass', ('bass', p), ['B0', 'B1'][i % 2]))
for t in E['kick']: shots.append((t, 'kick', ('kick', 0), 'K'))
for t in E['snare']: shots.append((t, 'snare', ('snare', 0), 'S'))
for t, k in E['tom']: shots.append((t, 'tom', ('tom', k), 'TM'))
for t, side in E['crash']: shots.append((t, 'crash', ('crash', side), 'XR' if side else 'XL'))
shots.sort(key=lambda s: s[0])

# ---------------- return circuit ----------------
GUT = {   # kind -> (y, z at x_end, z at x_start, x_end(outlet side), chute x)
    'X': dict(y=2.14, zl=6.0, zr=6.45, xa=-4.55, xb=4.25, out=-1),
    'V': dict(y=1.9, zl=4.5, zr=4.1, xa=-3.95, xb=4.55, out=1),
    'B': dict(y=1.55, zl=1.45, zr=1.75, xa=-4.55, xb=-0.05, out=-1),
}
GUTOF = {'mel': 'X', 'gliss': 'X', 'harm': 'X', 'chord': 'V', 'bass': 'B'}
CHUTE = {-1: Vector((-4.82, 1.9, 0.32)), 1: Vector((4.82, 1.9, 0.32))}
def gz(g, x): return g['zl'] + (g['zr'] - g['zl']) * (x - g['xa']) / (g['xb'] - g['xa'])
RAIL = mat('rail', (0.85, 0.65, 0.35), 0.2, 1.0)
def track(p0, p1, wid=0.085, drop=0.1):
    p0, p1 = Vector(p0), Vector(p1); d = (p1 - p0); side = Vector((-d.y, d.x, 0))
    side = side.normalized() if side.length > 1e-6 else Vector((0, 1, 0))
    for s in (-1, 1): rod(p0 + side * s * wid + Vector((0, 0, -drop)), p1 + side * s * wid + Vector((0, 0, -drop)), 0.02, RAIL, 8)
for g in GUT.values():
    track((g['xa'], g['y'], gz(g, g['xa'])), (g['xb'], g['y'], gz(g, g['xb'])))
    for xx in np.linspace(g['xa'], g['xb'], 6):
        rod((xx, g['y'], gz(g, xx) - 0.12), (xx, g['y'] + 0.45, gz(g, xx) - 0.12), 0.015, STEEL)
for s, c in CHUTE.items():
    top = 6.1 if s < 0 else 4.2
    for a in range(3):
        an = 2 * math.pi * a / 3 + 0.3
        rod(c + Vector((0.17 * math.cos(an), 0.17 * math.sin(an), 0)), Vector((c.x, c.y, top)) + Vector((0.17 * math.cos(an), 0.17 * math.sin(an), 0)), 0.018, RAIL, 8)
    for zz in np.arange(0.6, top, 0.9):
        prim('torus', BRASS, major_radius=0.19, minor_radius=0.02, location=(c.x, c.y, zz))
    col = prim('cylinder', GUNM, radius=0.26, depth=0.5, vertices=20, location=(c.x, c.y, 0.25))   # collector (meters balls)
# floor channels & front rail -> lift
LIFT0 = Vector((-3.55, -2.6, 0.2)); LIFT1 = Vector((-3.55, -2.6, 1.9))
FRONT_R = Vector((3.35, -2.6, 0.24)); FRONT_L = Vector((-3.25, -2.6, 0.2))
track(CHUTE[1] + Vector((0, 0, -0.08)), FRONT_R); track(FRONT_R, FRONT_L)
track(CHUTE[-1] + Vector((0, 0, -0.08)), FRONT_L + Vector((-0.05, 0.05, 0)))
liftbox = prim('cube', GUNM, size=1, location=(LIFT0.x, LIFT0.y, 0.18)); liftbox.scale = (0.5, 0.5, 0.36)
for s in (-1, 1): rod((LIFT0.x + s * 0.13, LIFT0.y + 0.14, 0.1), (LIFT0.x + s * 0.13, LIFT0.y + 0.14, 2.1), 0.025, STEEL)
CLEATS = []
for k in range(9):
    c = prim('cube', BRASS, size=1, location=(LIFT0.x, LIFT0.y + 0.1, 0.4)); c.scale = (0.24, 0.04, 0.03); CLEATS.append(c)
LIFT_V = 2.2


def path_keys(o, pts):
    """pts: list of (time, Vector) - linear between them"""
    for t, p in pts:
        f = t * FPS
        if f < 0 or f > NF - 1: continue
        o.location = p; o.keyframe_insert('location', frame=int(round(f)))


collector_free = {-1: 0.0, 1: 0.0}; lift_free = 0.0; HEAD = 0.075
ret = []
sph = {}
for k_, m_ in BALLM.items():
    r = 0.17 if k_ in ('kick', 'crash') else BR
    bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=r)
    me = bpy.data.meshes.new('ball_' + k_); bm.to_mesh(me); bm.free()
    for p_ in me.polygons: p_.use_smooth = True
    me.materials.append(m_); sph[k_] = me

nball = 0
returns = []
for t, kind, key, tn in shots:
    tur = T[tn]; M, v0 = tur.add(t, key)
    o = link(bpy.data.objects.new(f'b{nball}', sph[kind])); o.color = (*BALLM[kind].node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value[:3], 1); nball += 1
    P1 = TARGET[key]; t0 = t - TAU
    vin = v0 + Vector((0, 0, -G * TAU))
    keys = []
    f0 = int(math.floor(t0 * FPS)); f1 = int(math.floor(t * FPS))
    keys.append((max(0, f0 - 1) / FPS, M, False))
    for f in range(f0, f1 + 1):
        a = f / FPS - t0
        if a < 0: continue
        keys.append((f / FPS, M + v0 * a + Vector((0, 0, -0.5 * G * a * a)), True))
    keys.append((t, P1, True))
    gk = GUTOF.get(kind)
    if gk:
        g = GUT[gk]; zt = gz(g, P1.x)
        vz = 0.55; dz = zt - P1.z
        a_land = (vz + math.sqrt(vz * vz - 2 * G * dz)) / G
        vy = (g['y'] - P1.y) / a_land
        for f in range(int(math.floor(t * FPS)) + 1, int(math.ceil((t + a_land) * FPS))):
            a = f / FPS - t
            keys.append((f / FPS, P1 + Vector((0, vy * a, vz * a - 0.5 * G * a * a)), True))
        tl = t + a_land; L = Vector((P1.x, g['y'], zt)); keys.append((tl, L, True))
        xe = g['xa'] if g['out'] < 0 else g['xb']
        E_ = Vector((xe, g['y'], gz(g, xe))); tl2 = tl + (L - E_).length / 3.2; keys.append((tl2, E_, True))
        c = CHUTE[g['out']]; topc = Vector((c.x, c.y, E_.z - 0.05)); tl3 = tl2 + (topc - E_).length / 2.0; keys.append((tl3, topc, True))
        h = topc.z - c.z; tfall = math.sqrt(2 * h / G)
        for q in range(1, 6):
            a = tfall * q / 5; keys.append((tl3 + a, Vector((c.x, c.y, topc.z - 0.5 * G * a * a)), True))
        arr = tl3 + tfall
        returns.append((arr, o, keys, g['out']))
    else:
        if kind == 'crash':
            vout = Vector((1.6 if P1.x > 0 else -1.6, 0.3, 1.0)); dur = 1.3; zfloor = -9
        elif kind == 'snare':
            vout = Vector((0.35, -0.55, 2.0)); dur = 2.0; zfloor = 0.2
        elif kind == 'tom':
            vout = Vector((0.1, -2.1, 1.3)); dur = 2.0; zfloor = 0.2
        else:
            vout = Vector((rng.uniform(-0.3, 0.3), -1.25, 1.5)); dur = 2.0; zfloor = 0.2
        end = t
        for f in range(int(math.floor(t * FPS)) + 1, int((t + dur) * FPS)):
            a = f / FPS - t; p = P1 + vout * a + Vector((0, 0, -0.5 * G * a * a))
            if p.z < zfloor: break
            keys.append((f / FPS, p, True)); end = f / FPS
        keys.append((end + 1 / FPS, keys[-1][1], False))
        ball_end = keys
        for tt, p, vis in keys:
            f = int(round(tt * FPS))
            if 0 <= f < NF:
                o.location = p; o.hide_render = not vis
                o.keyframe_insert('location', frame=f); o.keyframe_insert('hide_render', frame=f)

# metering: collectors then the lift release balls one by one (no overlaps on the shared rails)
returns.sort(key=lambda r: r[0])
lift_q = []
for arr, o, keys, side in returns:
    c = CHUTE[side]
    dep = max(arr + 0.05, collector_free[side]); collector_free[side] = dep + 0.12
    keys.append((dep - 0.02, Vector((c.x, c.y, 0.3)), False))
    keys.append((dep, Vector((c.x, c.y, 0.28)), True))
    if side > 0:
        tA = dep + (FRONT_R - c).length / 3.6; keys.append((tA, FRONT_R, True))
        tB = tA + (FRONT_L - FRONT_R).length / 3.6; keys.append((tB, FRONT_L, True))
    else:
        tB = dep + (FRONT_L - c).length / 3.6; keys.append((tB, FRONT_L, True))
    tC = tB + (LIFT0 - FRONT_L).length / 2.0; keys.append((tC, LIFT0, False))
    lift_q.append((tC, o, keys))
lift_q.sort(key=lambda r: r[0])
for tC, o, keys in lift_q:
    dep = max(tC + 0.05, lift_free); lift_free = dep + 0.13
    keys.append((dep - 0.02, LIFT0, False)); keys.append((dep, LIFT0 + Vector((0, 0, 0.25)), True))
    tD = dep + (LIFT1.z - LIFT0.z - 0.25) / LIFT_V; keys.append((tD, LIFT1, True))
    tE = tD + 0.25; keys.append((tE, MAG + Vector((0, 0, 0.3)), True)); keys.append((tE + 0.04, MAG, False))
    keys.sort(key=lambda k: k[0])
    lastf = -1
    for tt, p, vis in keys:
        f = int(round(tt * FPS))
        if f < 0 or f > NF - 1: continue
        if f == lastf and vis: pass
        o.location = p; o.hide_render = not vis
        o.keyframe_insert('location', frame=f); o.keyframe_insert('hide_render', frame=f); lastf = f

# ---------------- flashes, drives, clockwork ----------------
by = {}
for t, kind, key, tn in shots: by.setdefault(key, []).append(t)
for o, key, rgb in INST:
    ts = np.array(by.get(key, []))
    g = np.zeros(NF); shk = np.zeros(NF)
    for t in ts:
        m = (TT >= t) & (TT < t + 0.6); dt = TT[m] - t
        g[m] = np.maximum(g[m], np.exp(-dt / 0.13)); shk[m] += 0.03 * np.exp(-dt / 0.07) * np.cos(70 * dt)
    key_series(o, 'color', 3, g, 2e-3)
    base = o.location.y
    key_series(o, 'location', 1, base + shk, 2e-4)
for t_ in T.values(): t_.animate()

# cymbal swing
for side, piv in enumerate(CPIV):
    a = np.zeros(NF)
    for t, s in E['crash']:
        if s != side: continue
        m = (TT >= t) & (TT < t + 2.5); dt = TT[m] - t
        a[m] += 0.32 * np.exp(-dt / 0.55) * np.sin(2 * math.pi * 2.2 * dt)
    key_series(piv, 'rotation_euler', 0, a); key_series(piv, 'rotation_euler', 1, 0.5 * a)
# hi-hat sticks (alternate), cam shaft, cymbal flash
hats = np.array(E['hat']); hs = [np.zeros(NF) + 0.5, np.zeros(NF) + 0.5]; hg = np.zeros(NF); hz = np.zeros(NF)
for i, th in enumerate(hats):
    j = i % 2
    m = (TT >= th - 0.08) & (TT < th + 0.16); dt = TT[m] - th
    prof = np.where(dt < 0, 0.5 * (-dt / 0.08), 0.5 * np.clip(dt / 0.14, 0, 1))
    hs[j][m] = np.minimum(hs[j][m], prof)
    m2 = (TT >= th) & (TT < th + 0.25); d2 = TT[m2] - th
    hg[m2] = np.maximum(hg[m2], 0.8 * np.exp(-d2 / 0.06)); hz[m2] -= 0.025 * np.exp(-d2 / 0.05)
for j in range(2): key_series(STICKS[j], 'rotation_euler', 0, hs[j])
key_series(hat_top, 'color', 3, hg, 2e-3); key_series(hat_top, 'location', 2, HZ + 0.04 + hz, 2e-4)
cams = np.where(TT > TB0 - 0.5, (TT - TB0) / BEAT * math.pi, 0)
key_series(camshaft, 'rotation_euler', 0, cams)
# lift cleats
for k, c in enumerate(CLEATS):
    zz = 0.25 + ((TT * LIFT_V + k * 0.2) % 1.8)
    key_series(c, 'location', 2, zz)
# clockwork: master gear steps one tooth per beat (escapement), metronome swings
beats = np.clip((TT - TB0) / BEAT, 0, (FINAL - TB0) / BEAT)
k = np.floor(beats); fr = beats - k
step = np.clip(fr * BEAT / 0.16, 0, 1); step = 1 - (1 - step) ** 3
Amaster = (2 * math.pi / 36) * (k + step)
for o, Q, ratio, n in GEARS:
    key_series(o, 'rotation_euler', 1, -(Q + ratio * Amaster), 1e-4)
anc = 0.12 * np.where((k % 2) == 0, 1, -1) * (1 - np.exp(-fr * BEAT / 0.05))
anc = np.where((TT > TB0) & (TT < FINAL), anc, 0)
key_series(anchor, 'rotation_euler', 1, anc, 1e-4)
damp = np.where(TT < FINAL, 1.0, np.exp(-(TT - FINAL) / 1.2))
mang = np.where(TT >= TB0 - 0.01, 0.42 * np.sin(math.pi * (TT - TB0) / BEAT) * damp, 0)
key_series(metpiv, 'rotation_euler', 1, mang, 1e-4)
ml = np.zeros(NF)
for b in np.arange(TB0, FINAL + 0.01, BEAT):
    m = (TT >= b) & (TT < b + 0.2); ml[m] = np.maximum(ml[m], np.exp(-(TT[m] - b) / 0.06))
key_series(mlamp, 'color', 3, ml, 2e-3)

# ---------------- finale confetti ----------------
CONF = [glowmat(f'conf{k}', c, 0.4, gain=0.6) for k, c in enumerate(((1, 0.2, 0.3), (1, 0.85, 0.2), (0.2, 0.7, 1), (0.3, 1, 0.4), (1, 1, 1)))]
for k in range(160):
    o = prim('plane', CONF[k % len(CONF)], size=0.12, location=(0, 0, -5)); o.color = (1, 1, 1, 1); o.rotation_mode = 'XYZ'
    p0 = Vector((rng.uniform(-4.8, 4.8), rng.uniform(-2, 2.5), rng.uniform(12, 14.5))); delay = rng.uniform(0, 1.5); spd = rng.uniform(0.8, 1.5)
    for f in [0, int((FINAL + delay) * FPS) - 1] + list(range(int((FINAL + delay) * FPS), NF, 3)):
        tt = f / FPS - FINAL - delay
        if tt < 0: o.location = (0, 0, -5)
        else:
            o.location = p0 + Vector((0.3 * math.sin(tt * 3 + delay * 5), 0, -spd * tt)); o.rotation_euler = (tt * 5 + delay, tt * 3, tt * 4)
        o.keyframe_insert('location', frame=f); o.keyframe_insert('rotation_euler', frame=f)

# ---------------- camera ----------------
cd = bpy.data.cameras.new('cam'); cd.sensor_fit = 'HORIZONTAL'; cd.sensor_width = 36; cd.clip_end = 300
cam = link(bpy.data.objects.new('cam', cd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trk = cam.constraints.new('TRACK_TO'); trk.target = tgt; trk.track_axis = 'TRACK_NEGATIVE_Z'; trk.up_axis = 'UP_Y'
S = dict((n, t) for t, n in E['sections'])
KEYS = [  # time, cam pos, look-at, lens
    (0.0, (2.2, -5.6, 1.0), (0.0, -1.3, 1.1), 40),          # turret bank close-up as they aim
    (1.2, (1.6, -6.4, 1.4), (0.0, -1.0, 1.6), 40),
    (4.5, (0.0, -12.8, 5.0), (0.0, 2.0, 5.6), 50),          # whole machine
    (8.0, (-1.8, -8.0, 7.6), (-1.0, 2.2, 7.9), 50),          # xylophone + harmony bells
    (13.0, (1.8, -8.2, 7.9), (1.2, 2.2, 8.2), 50),
    (S['B'], (0.0, -6.2, 11.0), (0.0, 2.8, 11.3), 45),      # clockwork + metronome
    (S['B'] + 3.5, (-0.6, -6.6, 10.6), (0.0, 2.5, 10.8), 45),
    (S['B'] + 5.0, (0.0, -4.9, 1.1), (0.0, 2.2, 5.6), 30),  # behind the turrets: volley view
    (S['B'] + 10.5, (0.3, -5.2, 1.4), (0.0, 2.2, 6.2), 30),
    (S['B'] + 12.0, (4.6, -5.6, 3.6), (3.3, 1.0, 2.6), 42),  # drums, toms, cam-driven hi-hat
    (S['roll'] - 0.5, (4.1, -5.2, 3.2), (3.4, 1.0, 2.5), 42),
    (S['roll'] + 0.4, (0.0, -9.5, 6.5), (0.0, 2.2, 7.0), 46),  # snare roll -> glissando
    (S['A2'], (0.0, -12.8, 5.2), (0.0, 2.0, 5.8), 50),
    (S['A2'] + 4.0, (-1.6, -9.0, 2.6), (-2.4, 1.6, 2.6), 46),  # bass marimba + chute
    (S['A2'] + 8.5, (-0.7, -6.6, 1.2), (-3.0, -1.0, 2.7), 30),  # return rails, collector, lift
    (S['A2'] + 12.0, (0.0, -6.8, 1.3), (-2.4, -0.6, 2.9), 30),
    (S['A2'] + 14.0, (-2.0, -9.6, 5.0), (0.0, 2.0, 5.2), 46),  # chord vibes + volley
    (S['A2'] + 18.0, (2.0, -9.6, 5.4), (0.0, 2.0, 5.4), 46),
    (S['finale'] - 1.0, (0.0, -12.6, 5.4), (0.0, 2.0, 5.9), 50),
    (FINAL, (0.0, -13.6, 5.8), (0.0, 2.0, 6.2), 50),
    (T_END, (0.0, -15.0, 6.4), (0.0, 2.0, 6.4), 50),
]


def cam_state(t):
    for k in range(len(KEYS) - 1):
        if KEYS[k][0] <= t <= KEYS[k + 1][0]:
            a = ss((t - KEYS[k][0]) / max(1e-3, KEYS[k + 1][0] - KEYS[k][0]))
            return (Vector(KEYS[k][1]).lerp(Vector(KEYS[k + 1][1]), a), Vector(KEYS[k][2]).lerp(Vector(KEYS[k + 1][2]), a),
                    KEYS[k][3] + (KEYS[k + 1][3] - KEYS[k][3]) * a)
    return Vector(KEYS[-1][1]), Vector(KEYS[-1][2]), KEYS[-1][3]


crash_t = [c[0] for c in E['crash']]
for f in range(NF):
    t = f / FPS; p, l, lens = cam_state(t)
    for c in crash_t:
        if 0 <= t - c < 0.4:
            kk = math.exp(-(t - c) / 0.1); p = p + Vector((0.06 * kk * math.sin(90 * t), 0, 0.05 * kk * math.sin(77 * t)))
    cam.location = p; tgt.location = l; cd.lens = lens
    cam.keyframe_insert('location', frame=f); tgt.keyframe_insert('location', frame=f); cd.keyframe_insert('lens', frame=f)

print('built balls', nball, 'turrets', len(T), 'objects', len(bpy.data.objects))

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
