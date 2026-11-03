"""Snowy village light show v2: every bulb individually sequenced (per-bulb, per-frame colour from an image)."""
import os, sys, json, math, colorsys, time
import bpy, bmesh
import numpy as np
from mathutils import Vector, Euler, Matrix

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json'))
FPS = 30; T_END = 70.0; NF = int(T_END * FPS); FINAL = E['final']; BEAT = E['beat']; T0 = 0.6
rng = np.random.default_rng(12)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'
sc.cycles.samples = int(os.environ.get('SAMP', 5))
sc.cycles.use_adaptive_sampling = True; sc.cycles.adaptive_threshold = 0.08
sc.cycles.use_denoising = True
for k, v in (('denoiser', 'OPENIMAGEDENOISE'), ('denoising_quality', 'FAST'), ('denoising_prefilter', 'FAST'),
             ('use_fast_gi', True), ('ao_bounces_render', 1)):
    try: setattr(sc.cycles, k, v)
    except Exception: pass
sc.cycles.max_bounces = 3; sc.cycles.diffuse_bounces = 1; sc.cycles.glossy_bounces = 1
sc.cycles.caustics_reflective = False; sc.cycles.caustics_refractive = False
sc.cycles.sample_clamp_indirect = 4
sc.render.resolution_x, sc.render.resolution_y = 720, 1280
sc.render.resolution_percentage = int(os.environ.get('PCT', 66))
sc.render.fps = FPS
sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.35
sc.render.use_persistent_data = True; sc.render.use_overwrite = False
try: sc.view_settings.view_transform = 'Standard'; sc.view_settings.look = 'Medium High Contrast'
except Exception: pass
sc.frame_start, sc.frame_end = 0, NF - 1


def link(o): sc.collection.objects.link(o); return o


def mat(name, col, rough=0.5, metal=0.0, **kw):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*col, 1); b.inputs['Roughness'].default_value = rough
    b.inputs['Metallic'].default_value = metal
    for k, v in kw.items(): b.inputs[k].default_value = v
    return m, b


def emis(name, col, strength):
    m, b = mat(name, (0, 0, 0), 1.0)
    b.inputs['Emission Color'].default_value = (*col, 1); b.inputs['Emission Strength'].default_value = strength
    return m

# ---------------- night sky ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
nt = w.node_tree; nt.nodes.clear(); L = nt.links.new
tc = nt.nodes.new('ShaderNodeTexCoord'); sep = nt.nodes.new('ShaderNodeSeparateXYZ'); rp = nt.nodes.new('ShaderNodeValToRGB')
L(tc.outputs['Generated'], sep.inputs[0]); L(sep.outputs['Z'], rp.inputs[0])
rp.color_ramp.elements[0].position = 0.5; rp.color_ramp.elements[0].color = (0.05, 0.07, 0.14, 1)
rp.color_ramp.elements[1].position = 0.85; rp.color_ramp.elements[1].color = (0.008, 0.012, 0.035, 1)
bg = nt.nodes.new('ShaderNodeBackground'); wo = nt.nodes.new('ShaderNodeOutputWorld')
L(rp.outputs[0], bg.inputs['Color']); bg.inputs['Strength'].default_value = 0.3; L(bg.outputs[0], wo.inputs[0])
ld = bpy.data.lights.new('moon', 'SUN'); ld.energy = 0.12; ld.color = (0.6, 0.72, 1.0); ld.angle = math.radians(6)
link(bpy.data.objects.new('moon', ld)).rotation_euler = Euler((math.radians(55), 0, math.radians(150)))

# ---------------- snowy hillside ----------------
SNOW, sb = mat('snow', (0.7, 0.75, 0.85), 0.6)
n = SNOW.node_tree.nodes
nz = n.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 0.8
bump = n.new('ShaderNodeBump'); bump.inputs['Strength'].default_value = 0.25
SNOW.node_tree.links.new(nz.outputs['Fac'], bump.inputs['Height']); SNOW.node_tree.links.new(bump.outputs['Normal'], sb.inputs['Normal'])


def hill_z(x, y):
    return 0.0 if y < 1.0 else 0.42 * (y - 1.0) + 0.15 * math.sin(x * 0.4)

bpy.ops.mesh.primitive_grid_add(x_subdivisions=60, y_subdivisions=60, size=60, location=(0, 14, 0))
g = bpy.context.object
for v in g.data.vertices:
    wv = g.matrix_world @ v.co
    v.co.z = hill_z(wv.x, wv.y) + 0.08 * math.sin(wv.x * 1.3) * math.cos(wv.y * 0.9)
g.data.materials.append(SNOW)
for p in g.data.polygons: p.use_smooth = True


_TB = bmesh.new(); bmesh.ops.create_icosphere(_TB, subdivisions=1, radius=1.0)
_TV = np.array([v.co[:] for v in _TB.verts]); _TF = np.array([[v.index for v in f.verts] for f in _TB.faces]); _TB.free()


def instance_mesh(name, centers, radii, uv_x=None):
    centers = np.asarray(centers, float); radii = np.asarray(radii, float)
    nv = len(_TV); n_ = len(centers)
    verts = (_TV[None] * radii[:, None, None] + centers[:, None, :]).reshape(-1, 3)
    faces = (_TF[None] + (np.arange(n_) * nv)[:, None, None]).reshape(-1, 3)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts)); me.vertices.foreach_set('co', verts.ravel().astype(np.float32))
    me.loops.add(faces.size); me.loops.foreach_set('vertex_index', faces.ravel().astype(np.int32))
    me.polygons.add(len(faces))
    me.polygons.foreach_set('loop_start', (np.arange(len(faces)) * 3).astype(np.int32))
    me.polygons.foreach_set('loop_total', np.full(len(faces), 3, np.int32))
    if uv_x is not None:
        uvl = me.uv_layers.new(name='UVMap')
        bulb_of_loop = faces.ravel() // nv
        uv = np.stack([np.asarray(uv_x)[bulb_of_loop], np.full(len(bulb_of_loop), 0.5)], 1)
        uvl.data.foreach_set('uv', uv.ravel().astype(np.float32))
    me.update(); me.validate()
    return me

# ---------------- bulb registry ----------------
BULBS = []   # dicts: pos, u, kind, elem, sub


def add_string(pts, kind, elem, sub=0):
    n_ = len(pts)
    for k, p in enumerate(pts):
        BULBS.append({'pos': tuple(p), 'u': k / max(1, n_ - 1), 'kind': kind, 'elem': elem, 'sub': sub})


def line_pts(a, b, step=0.09, skip_first=False):
    a, b = Vector(a), Vector(b); n_ = max(2, int((b - a).length / step) + 1)
    out = [a.lerp(b, i / (n_ - 1)) for i in range(n_)]
    return out[1:] if skip_first else out

# ---------------- houses ----------------
pitches = sorted({m['p'] for m in E['melody']})
WALLS = [(0.6, 0.12, 0.1), (0.85, 0.78, 0.6), (0.2, 0.3, 0.5), (0.25, 0.4, 0.25), (0.55, 0.35, 0.2), (0.7, 0.55, 0.65)]
ROOF, _ = mat('roof', (0.25, 0.1, 0.08), 0.7)
WIN = emis('window', (1.0, 0.65, 0.3), 3.0)
DOOR, _ = mat('door', (0.25, 0.14, 0.07), 0.6)
SLOTS = [(-4.0, 2.6), (-1.35, 2.2), (1.35, 2.5), (4.0, 2.3), (-2.8, 6.4), (0.1, 6.9), (2.9, 6.3), (-2.0, 10.6), (0.8, 11.2), (3.4, 10.4)]
HOUSE_X = {}
for i, (p, (x, y)) in enumerate(zip(pitches, SLOTS)):
    z0 = hill_z(x, y)
    wdt, dep, hgt = rng.uniform(1.6, 2.0), 1.6, rng.uniform(1.2, 1.6)
    rh = 1.0
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, z0 + hgt / 2 - 0.1))
    h = bpy.context.object; h.scale = (wdt, dep, hgt + 0.2)
    h.data.materials.append(mat(f'wall{i}', WALLS[i % len(WALLS)], 0.8)[0])
    for k, (col, ex, lift) in enumerate(((ROOF, 0.25, 0.0), (SNOW, 0.32, 0.07))):
        bm = bmesh.new()
        hw, hd = wdt / 2 + ex, dep / 2 + ex
        zb = z0 + hgt + lift
        vs = [bm.verts.new(v) for v in ((x - hw, y - hd, zb), (x + hw, y - hd, zb), (x + hw, y + hd, zb), (x - hw, y + hd, zb),
                                        (x, y - hd, zb + rh), (x, y + hd, zb + rh))]
        for f in ((0, 4, 5, 3), (1, 2, 5, 4), (0, 1, 4), (3, 5, 2), (0, 3, 2, 1)):
            bm.faces.new([vs[j] for j in f])
        me = bpy.data.meshes.new(f'roof{i}{k}'); bm.to_mesh(me); bm.free(); me.materials.append(col)
        link(bpy.data.objects.new(me.name, me))
    fy = y - dep / 2 - 0.01
    for wx in (-wdt * 0.28, wdt * 0.28):
        bpy.ops.mesh.primitive_plane_add(size=1, location=(x + wx, fy, z0 + hgt * 0.6), rotation=(math.pi / 2, 0, 0))
        wo_ = bpy.context.object; wo_.scale = (0.32, 0.36, 1); wo_.data.materials.append(WIN)
    bpy.ops.mesh.primitive_plane_add(size=1, location=(x, fy, z0 + 0.35), rotation=(math.pi / 2, 0, 0))
    d_ = bpy.context.object; d_.scale = (0.3, 0.6, 1); d_.data.materials.append(DOOR)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x + wdt * 0.28, y + 0.3, z0 + hgt + rh * 0.55))
    ch = bpy.context.object; ch.scale = (0.22, 0.22, 0.6); ch.data.materials.append(mat('brick', (0.4, 0.15, 0.1), 0.8)[0])
    # outline string: bottom-left corner -> up the wall -> up the gable -> down the gable -> down the right wall
    hw, hd, zb = wdt / 2 + 0.33, dep / 2 + 0.33, z0 + hgt + 0.1
    fy2 = y - hd - 0.03; fyw = y - dep / 2 - 0.04
    xl, xr = x - wdt / 2 - 0.02, x + wdt / 2 + 0.02
    path = (line_pts((xl, fyw, z0 + 0.05), (xl, fyw, z0 + hgt - 0.05)) +
            line_pts((x - hw, fy2, zb), (x, fy2, zb + rh + 0.08), skip_first=False) +
            line_pts((x, fy2, zb + rh + 0.08), (x + hw, fy2, zb), skip_first=True) +
            line_pts((xr, fyw, z0 + hgt - 0.05), (xr, fyw, z0 + 0.05)))
    add_string(path, 'outline', p)
    add_string(line_pts((x - hw, fy2, zb - 0.03), (x + hw, fy2, zb - 0.03)), 'eave', p)
    # window frames
    for s_, wx in enumerate((-wdt * 0.28, wdt * 0.28)):
        cx, cz = x + wx, z0 + hgt * 0.6
        rect = (line_pts((cx - 0.2, fyw, cz - 0.22), (cx + 0.2, fyw, cz - 0.22), 0.08) + line_pts((cx + 0.2, fyw, cz - 0.22), (cx + 0.2, fyw, cz + 0.22), 0.08, True)
                + line_pts((cx + 0.2, fyw, cz + 0.22), (cx - 0.2, fyw, cz + 0.22), 0.08, True) + line_pts((cx - 0.2, fyw, cz + 0.22), (cx - 0.2, fyw, cz - 0.22), 0.08, True))
        add_string(rect, 'window', p, s_)
    HOUSE_X[p] = x

# ---------------- trees ----------------
PINE, _ = mat('pine', (0.03, 0.12, 0.06), 0.8)


def tree_body(x, y, hgt):
    z0 = hill_z(x, y)
    for k in range(3):
        rr = hgt * (0.38 - 0.09 * k); zz = z0 + hgt * (0.25 + 0.22 * k)
        bpy.ops.mesh.primitive_cone_add(radius1=rr, depth=hgt * 0.45, vertices=24, location=(x, y, zz + hgt * 0.22))
        bpy.context.object.data.materials.append(PINE)
        bpy.ops.mesh.primitive_cone_add(radius1=rr * 0.75, depth=hgt * 0.2, vertices=24, location=(x, y, zz + hgt * 0.37))
        bpy.context.object.data.materials.append(SNOW)
    return z0

PINES = [(-5.6, 4.5, 2.6), (5.6, 4.8, 2.8), (-5.0, 9.0, 3.0), (5.2, 8.6, 2.6), (-1.0, 14.2, 2.6), (2.5, 14.6, 2.9)]
for k, (x, y, hh) in enumerate(PINES):
    z0 = tree_body(x, y, hh)
    pts = []
    nb = int(hh * 34)
    for j in range(nb):
        u = j / nb; a = u * 2 * math.pi * 6
        rr = hh * 0.37 * (1 - u) + 0.06; zz = z0 + hh * (0.12 + 0.82 * u)
        pts.append((x + rr * math.cos(a), y + rr * math.sin(a), zz))
    add_string(pts, 'pine', k)

# mega tree: 20 vertical strings draped over the big tree, top -> bottom
MX, MY, MH = -3.3, -1.6, 4.4
mz0 = tree_body(MX, MY, MH)
NSTR = 20
for s_ in range(NSTR):
    a = s_ / NSTR * 2 * math.pi
    pts = []
    for j in range(46):
        u = j / 45
        rr = 0.12 + MH * 0.42 * u; zz = mz0 + MH * (1.0 - 0.9 * u)
        pts.append((MX + rr * math.cos(a), MY + rr * math.sin(a), zz))
    add_string(pts, 'mega', 0, s_)
# star outline
star_c = Vector((MX, MY, mz0 + MH + 0.35))
spts = []
for k in range(10):
    a0 = math.pi / 2 + k * math.pi / 5; a1 = math.pi / 2 + (k + 1) * math.pi / 5
    r0 = 0.5 if k % 2 == 0 else 0.21; r1 = 0.21 if k % 2 == 0 else 0.5
    p0 = star_c + Vector((r0 * math.cos(a0), -0.05, r0 * math.sin(a0))); p1 = star_c + Vector((r1 * math.cos(a1), -0.05, r1 * math.sin(a1)))
    spts += line_pts(p0, p1, 0.07, skip_first=k > 0)
add_string(spts, 'star', 0)

# ---------------- garlands across the streets ----------------
POLE = mat('pole', (0.15, 0.1, 0.08), 0.6)[0]
for k, (y, z, span) in enumerate(((0.9, 2.2, 5.5), (4.4, 3.0, 5.0), (8.6, 4.8, 4.6))):
    for px in (-span, span):
        bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=z + 0.2, location=(px, y, hill_z(px, y) + (z + 0.2) / 2))
        bpy.context.object.data.materials.append(POLE)
    zt = hill_z(0, y) + z
    pts = []
    for j in range(110):
        u = j / 109; xx = -span + 2 * span * u
        pts.append((xx, y, zt - 0.45 * (1 - (2 * u - 1) ** 2)))
    add_string(pts, 'garland', k)

# ---------------- arches over the front path ----------------
for a_i, ay in enumerate((-4.6, -3.5, -2.4, -1.3, -0.2)):
    pts = []
    for j in range(44):
        th = math.pi * j / 43
        pts.append((1.25 - 0.95 * math.cos(th), ay, 0.05 + 1.35 * math.sin(th)))
    add_string(pts, 'arch', a_i)

# ---------------- build the bulb mesh with a UV = bulb index ----------------
NB = len(BULBS)
print('bulbs', NB)
radii = [0.06 if b['kind'] in ('mega', 'star', 'arch') else 0.045 for b in BULBS]
bme = instance_mesh('bulbs', [b['pos'] for b in BULBS], radii, (np.arange(NB) + 0.5) / NB)
bulbs_obj = link(bpy.data.objects.new('bulbs', bme))

# ---------------- choreography -> image (rows = frames, columns = bulbs) ----------------
pos = np.array([b['pos'] for b in BULBS]); U = np.array([b['u'] for b in BULBS])
KIND = np.array([b['kind'] for b in BULBS]); ELEM = np.array([b['elem'] for b in BULBS]); SUB = np.array([b['sub'] for b in BULBS])
IDX = np.zeros(NB, int)
for kname in set(KIND):
    for el in set(ELEM[KIND == kname]):
        sel = np.where((KIND == kname) & (ELEM == el))[0]
        IDX[sel] = np.arange(len(sel))
tf = np.arange(NF) / FPS
IMG = np.zeros((NF, NB, 3), np.float32)
X = pos[:, 0]; XN = (X - X.min()) / (X.max() - X.min())


def hsv(h, s=1.0, v=1.0):
    h = np.asarray(h) % 1.0
    i = np.floor(h * 6).astype(int); f = h * 6 - i
    p, q, t_ = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r = np.choose(i % 6, [v, q, p, p, t_, v]); g_ = np.choose(i % 6, [t_, v, v, q, p, p]); b_ = np.choose(i % 6, [p, p, t_, v, v, q])
    return np.stack([r, g_, b_], -1)


def fr(t): return int(np.clip(round(t * FPS), 0, NF - 1))


def put(mask, f0, f1, val):
    """max-blend val (shape (f, nb, 3) or broadcastable) into IMG[f0:f1, mask]"""
    if f1 <= f0: return
    cur = IMG[f0:f1][:, mask]
    IMG[f0:f1, mask] = np.maximum(cur, val)


WARM = np.array([1.0, 0.78, 0.45]); WHITE = np.array([1, 1, 1.0]); RED = np.array([1, 0.08, 0.05]); GREEN = np.array([0.1, 1, 0.2])
GOLD = np.array([1, 0.65, 0.1]); BLUE = np.array([0.15, 0.35, 1.0])
def pitch_col(p): return hsv((p - 62) / 18.0 * 0.85, 0.85, 1.0)

sections = dict((s, t) for t, s in E['sections'])
T_VERSE, T_CHORUS, T_TAG = sections['verse'], sections['chorus'], sections['tag']
beats = T0 + BEAT * np.arange(0, int((FINAL - T0) / BEAT) + 1)
eighths = E['bells']
mel = E['melody']

# 0) power-on during the intro: a front sweeps left->right on each sleigh-bell eighth, leaving a warm glow
intro8 = [t for t in eighths if t < T_VERSE]
for k, t in enumerate(intro8):
    front = (k + 1) / len(intro8)
    m_ = XN <= front
    f0 = fr(t); f1 = fr(T_VERSE)
    IMG[f0:f1, m_] = np.maximum(IMG[f0:f1, m_], WARM * 0.18)
    edge = (XN > front - 1 / len(intro8)) & m_
    dec = np.exp(-(tf[f0:fr(t + 0.4)] - t) / 0.12)[:, None, None]
    put(edge, f0, fr(t + 0.4), WHITE * dec)
# baseline glow after the intro (outline shapes stay faintly readable)
fb = fr(T_VERSE)
for kname, c, lv in (('outline', WARM, 0.025), ('eave', WARM, 0.02), ('window', WARM, 0.04), ('pine', GREEN, 0.03),
                     ('garland', WARM, 0.025), ('mega', GOLD, 0.03), ('arch', WARM, 0.025), ('star', GOLD, 0.4)):
    m_ = KIND == kname
    IMG[fb:, m_] = np.maximum(IMG[fb:, m_], c * lv)

# 1) melody: the note's house outline is DRAWN by a running light (wipe), held while the note sounds, then fades
for j, m in enumerate(mel):
    t, d, p, sect = m['t'], m['d'], m['p'], m['sect']
    if p not in HOUSE_X: continue
    col = pitch_col(p)
    hm = (KIND == 'outline') & (ELEM == p)
    wipe = min(0.35, max(0.12, d * 0.6))
    hold_end = t + max(d * 0.95, wipe + 0.05)
    f0, f1, f2 = fr(t), fr(hold_end), fr(hold_end + 0.35)
    tt = tf[f0:f2][:, None]
    prog = np.clip((tt - t) / wipe, 0, 1)
    uu = U[hm][None, :]
    lit = (uu <= prog).astype(np.float32)
    head = np.exp(-((uu - prog) ** 2) / 0.002) * (tt < t + wipe)          # bright running head
    fade = np.where(tt > hold_end, np.clip(1 - (tt - hold_end) / 0.35, 0, 1), 1.0)
    val = (lit * 0.85 * fade)[..., None] * col + (head * 1.6)[..., None] * WHITE
    put(hm, f0, f2, val)
    # eave sparkles white at the onset, window frames take the note colour
    em_ = (KIND == 'eave') & (ELEM == p)
    put(em_, f0, fr(t + 0.3), (np.exp(-(tf[f0:fr(t + 0.3)] - t) / 0.08))[:, None, None] * WHITE * 1.4)
    wm_ = (KIND == 'window') & (ELEM == p)
    put(wm_, f0, f1, col * 0.9)
    # chorus: every note also sweeps across the whole village roofline (alternating red / green)
    if sect in ('chorus', 'tag'):
        allm = (KIND == 'eave')
        cc = RED if j % 2 == 0 else GREEN
        f3 = fr(t + 0.45); tt2 = tf[f0:f3][:, None]
        front = (tt2 - t) / 0.25
        xv = XN[allm][None, :]
        band = np.exp(-((xv - front) ** 2) / 0.01) + 0.35 * (xv < front) * np.exp(-(tt2 - t) / 0.2)
        put(allm, f0, f3, band[..., None] * cc)

# 2) garlands: theater chase that steps on every sleigh-bell eighth (verse), rainbow chase (chorus)
gm = KIND == 'garland'
for k, t in enumerate(eighths):
    if t < T_VERSE: continue
    t_next = eighths[k + 1] if k + 1 < len(eighths) else t + BEAT / 2
    f0, f1 = fr(t), fr(t_next)
    if t < T_CHORUS:
        on = ((IDX[gm] + k) % 4 == 0)
        col = np.where(((IDX[gm] // 4 + ELEM[gm]) % 2 == 0)[:, None], RED, GREEN)
        val = (on[:, None] * col)[None] * np.exp(-(tf[f0:f1] - t) / 0.18)[:, None, None]
    else:
        tt = tf[f0:f1][:, None]
        hue = IDX[gm][None, :] / 18.0 - tt * 1.6
        val = hsv(hue) * (0.55 + 0.45 * ((IDX[gm] + k) % 2 == 0))[None, :, None]
    put(gm, f0, f1, val)

# 3) arches: a comet runs over each arch on every beat, rippling from the front arch to the back
am = KIND == 'arch'
for b_i, t in enumerate(beats):
    if t < T_VERSE: continue
    big = t >= T_CHORUS
    for a_i in range(5):
        sel = am & (ELEM == a_i)
        ts_ = t + a_i * 0.06
        dur = BEAT * 0.9
        f0, f1 = fr(ts_), fr(ts_ + dur + 0.2)
        tt = tf[f0:f1][:, None]
        prog = (tt - ts_) / dur
        uu = U[sel][None, :]
        col = [RED, GREEN, GOLD, BLUE, WHITE][(b_i + a_i) % 5]
        comet = np.exp(-np.clip(prog - uu, 0, None) / 0.12) * (uu <= prog) * (prog <= 1.2)
        val = comet[..., None] * col * (1.3 if big else 0.9)
        if big and b_i % 2 == 0:   # kick: whole arch flashes white
            val = np.maximum(val, (np.exp(-(tt - ts_) / 0.12))[..., None] * WHITE * 0.8)
        put(sel, f0, f1, val)

# 4) mega tree
mm = KIND == 'mega'; S = SUB[mm]; Um = U[mm]
fv, fc_ = fr(T_VERSE), fr(T_CHORUS)
tt = tf[fv:fc_][:, None]
spiral = ((S[None, :] / NSTR + Um[None, :] * 1.3 - tt * 0.55) % 1.0) < 0.2
hue = (tt * 0.05 + Um[None, :] * 0.2) % 1
put(mm, fv, fc_, spiral[..., None] * hsv(hue, 0.9, 0.9))
for t in E['bass']:
    if T_VERSE <= t < T_CHORUS:
        f0, f1 = fr(t), fr(t + 0.4)
        put(mm, f0, f1, np.exp(-(tf[f0:f1] - t) / 0.15)[:, None, None] * GOLD * 0.5)
# chorus: rotating bars on every beat + bottom-to-top fill on the kicks
for b_i, t in enumerate(beats):
    if not (T_CHORUS <= t < FINAL): continue
    f0, f1 = fr(t), fr(t + BEAT)
    tt = tf[f0:f1][:, None]
    grp = (S // 5) == (b_i % 4)
    col = [RED, GREEN, WHITE, GOLD][(b_i // 4) % 4]
    put(mm, f0, f1, (grp[None, :] * np.exp(-(tt - t) / 0.3))[..., None] * col * 1.1)
    if b_i % 2 == 0:
        prog = (tt - t) / 0.16
        fill = ((1 - Um[None, :]) <= prog) * np.exp(-np.clip(tt - t - 0.16, 0, None) / 0.18)
        put(mm, f0, f1, fill[..., None] * WHITE * 0.9)

# 5) star: shimmer, pulses on kicks
sm = KIND == 'star'; Us = U[sm]
tt = tf[:, None]
IMG[:, sm] = np.maximum(IMG[:, sm], (0.3 + 0.25 * np.sin(Us[None, :] * 25 - tt * 6) ** 2)[..., None] * GOLD * (tf >= T_VERSE)[:, None, None])
for t in E['kick']:
    f0, f1 = fr(t), fr(t + 0.35)
    put(sm, f0, f1, np.exp(-(tf[f0:f1] - t) / 0.12)[:, None, None] * WHITE * 1.4)

# 6) pines: twinkle in the verse, side-to-side strobes on the eighths in the chorus, chord colour washes
pm_ = KIND == 'pine'
twk = rng.random((NF, pm_.sum())) < 0.05
cols_p = hsv(rng.random(pm_.sum()), 0.8, 1.0)
ver = (tf >= T_VERSE)[:, None]
IMG[:, pm_] = np.maximum(IMG[:, pm_], (twk * ver)[..., None] * cols_p[None] * 0.9)
for k, t in enumerate(eighths):
    if not (T_CHORUS <= t < FINAL): continue
    side = (pos[pm_, 0] < 0) if k % 2 == 0 else (pos[pm_, 0] >= 0)
    sel = np.where(pm_)[0][side]; msk = np.zeros(NB, bool); msk[sel] = True
    f0, f1 = fr(t), fr(t + 0.2)
    put(msk, f0, f1, np.exp(-(tf[f0:f1] - t) / 0.07)[:, None, None] * (GREEN if k % 4 < 2 else RED) * 1.1)
CHC = {'G': GREEN, 'C': RED, 'Am': BLUE, 'D': GOLD, 'D7': GOLD, 'A7': WHITE}
for t, ch in E['chords']:
    if t < T_VERSE or t >= FINAL: continue
    f0, f1 = fr(t), fr(t + 0.8)
    tt = tf[f0:f1][:, None]
    rise = (U[pm_][None, :] <= (tt - t) / 0.3) * np.exp(-(tt - t) / 0.45)
    put(pm_, f0, f1, rise[..., None] * CHC.get(ch, WHITE) * 0.8)

# 7) tag build-up: rainbow wave rolling across the whole village
f0, f1 = fr(T_TAG), fr(FINAL)
tt = tf[f0:f1][:, None]
ramp = np.clip((tt - T_TAG) / (FINAL - T_TAG), 0, 1)
val = hsv(XN[None, :] * 1.5 - tt * 1.2, 0.9, 1.0) * (0.25 + 0.6 * ramp)[..., None]
IMG[f0:f1] = np.maximum(IMG[f0:f1], val)

# 8) finale: white burst rolling outward from the mega tree, then everything on in its own colour, slow twinkle
f0 = fr(FINAL)
tt = tf[f0:][:, None]
dist = np.linalg.norm(pos - np.array([MX, MY, mz0 + MH]), axis=1)[None, :]
wave = np.exp(-((dist - (tt - FINAL) * 14) ** 2) / 2.0)
own = np.where((KIND == 'outline')[:, None], 0, 0)  # placeholder
base_cols = np.zeros((NB, 3))
for kname in ('outline', 'eave', 'window'):
    sel = KIND == kname
    base_cols[sel] = np.array([pitch_col(p) for p in ELEM[sel]])
base_cols[KIND == 'pine'] = cols_p
base_cols[KIND == 'mega'] = hsv(SUB[KIND == 'mega'] / NSTR, 0.8, 1.0)
base_cols[KIND == 'garland'] = hsv(IDX[KIND == 'garland'] / 12.0, 0.9, 1.0)
base_cols[KIND == 'arch'] = hsv(U[KIND == 'arch'] + ELEM[KIND == 'arch'] * 0.2, 0.9, 1.0)
base_cols[KIND == 'star'] = GOLD
twinkle = 0.75 + 0.25 * np.sin(tt * 5 + rng.random(NB)[None, :] * 20)
settle = np.clip((tt - FINAL) / 0.5, 0, 1)
IMG[f0:] = np.maximum(IMG[f0:], wave[..., None] * WHITE * 1.5 + (settle * twinkle)[..., None] * base_cols[None] * 0.9)

IMG = np.clip(IMG, 0, 3.0)
print('choreography built', IMG.shape)

img = bpy.data.images.new('show', width=NB, height=NF, alpha=True, float_buffer=True)
img.colorspace_settings.name = 'Non-Color'
rgba = np.concatenate([IMG, np.ones((NF, NB, 1), np.float32)], axis=2)
img.pixels.foreach_set(rgba.ravel())
# img.pack()  (not needed, saves startup time)

BM = bpy.data.materials.new('bulb'); BM.use_nodes = True
bn = BM.node_tree.nodes; bl = BM.node_tree.links; bn.clear()
uvn = bn.new('ShaderNodeUVMap'); uvn.uv_map = 'UVMap'
sepu = bn.new('ShaderNodeSeparateXYZ'); bl.new(uvn.outputs['UV'], sepu.inputs[0])
fv_ = bn.new('ShaderNodeValue'); fv_.name = 'frameV'
cmb = bn.new('ShaderNodeCombineXYZ'); bl.new(sepu.outputs['X'], cmb.inputs['X']); bl.new(fv_.outputs[0], cmb.inputs['Y'])
it = bn.new('ShaderNodeTexImage'); it.image = img; it.interpolation = 'Closest'; it.extension = 'EXTEND'
bl.new(cmb.outputs[0], it.inputs['Vector'])
em = bn.new('ShaderNodeEmission'); bl.new(it.outputs['Color'], em.inputs['Color']); em.inputs['Strength'].default_value = 6.0
glass = bn.new('ShaderNodeBsdfPrincipled'); glass.inputs['Base Color'].default_value = (0.05, 0.05, 0.05, 1); glass.inputs['Roughness'].default_value = 0.3
add_ = bn.new('ShaderNodeAddShader'); bl.new(glass.outputs[0], add_.inputs[0]); bl.new(em.outputs[0], add_.inputs[1])
out = bn.new('ShaderNodeOutputMaterial'); bl.new(add_.outputs[0], out.inputs[0])
bme.materials.append(BM)
fv_.outputs[0].default_value = 0.5 / NF; fv_.outputs[0].keyframe_insert('default_value', frame=0)
fv_.outputs[0].default_value = (NF - 0.5) / NF; fv_.outputs[0].keyframe_insert('default_value', frame=NF - 1)

# ---------------- falling snow ----------------
SF = emis('flake', (0.9, 0.93, 1.0), 1.2)
cent = np.stack([rng.uniform(-9, 9, 5000), rng.uniform(-5, 18, 5000), rng.uniform(-3, 72, 5000)], 1)
fme = instance_mesh('snowfall', cent, rng.uniform(0.018, 0.035, 5000)); fme.materials.append(SF)
flakes = link(bpy.data.objects.new('snowfall', fme))
flakes.location = (0, 0, 0); flakes.keyframe_insert('location', frame=0)
flakes.location = (0, 0, -0.95 * T_END); flakes.keyframe_insert('location', frame=NF)

# ---------------- MERRY CHRISTMAS ----------------
cu = bpy.data.curves.new('txt', 'FONT'); cu.body = 'Merry Christmas'; cu.align_x = 'CENTER'; cu.size = 1.25; cu.extrude = 0.05
txt = link(bpy.data.objects.new('txt', cu)); txt.rotation_euler = (math.pi / 2, 0, 0); txt.location = (0, 4.0, 10.2)
TXT, txb = mat('txt', (1, 0.85, 0.4), 0.3, 1.0); txb.inputs['Emission Color'].default_value = (1, 0.8, 0.35, 1)
cu.materials.append(TXT)
for f, s_, e in ((0, 0.0, 0.0), (fr(FINAL), 0.0, 0.0), (fr(FINAL + 0.6), 1.0, 6.0), (NF - 1, 1.05, 5.0)):
    txt.scale = (s_, s_, s_); txt.keyframe_insert('scale', frame=f)
    txb.inputs['Emission Strength'].default_value = e; txb.inputs['Emission Strength'].keyframe_insert('default_value', frame=f)


def linear_all(idb):
    ad = idb.animation_data
    if not ad or not ad.action: return
    try:
        fcs = list(ad.action.fcurves)
    except Exception:
        fcs = [fc for lay in ad.action.layers for st_ in lay.strips for cb in st_.channelbags for fc in cb.fcurves]
    for fc in fcs:
        for kp in fc.keyframe_points: kp.interpolation = 'LINEAR'

linear_all(flakes); linear_all(BM.node_tree)

# ---------------- camera: closer, livelier moves that visit the details ----------------
cd = bpy.data.cameras.new('cam'); cd.lens = 34; cd.sensor_fit = 'HORIZONTAL'; cd.sensor_width = 36; cd.clip_end = 500
cam = link(bpy.data.objects.new('cam', cd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trk = cam.constraints.new('TRACK_TO'); trk.target = tgt; trk.track_axis = 'TRACK_NEGATIVE_Z'; trk.up_axis = 'UP_Y'


def ss(x): x = min(max(x, 0), 1); return x * x * (3 - 2 * x)

KEYS = [  # (time, cam pos, look)
    (0.0, (2.6, -7.2, 0.9), (1.25, -2.0, 1.0)),          # low through the arches
    (T_VERSE, (1.6, -7.6, 1.6), (0.3, 1.0, 2.0)),
    (T_VERSE + 7.0, (-0.4, -8.6, 2.8), (-1.5, 3.0, 3.0)),
    (T_VERSE + 14.0, (2.0, -9.6, 3.4), (0.6, 5.0, 3.6)),
    (T_CHORUS, (-1.0, -8.0, 2.2), (-2.6, -0.5, 3.0)),       # chorus opens on the mega tree
    (T_CHORUS + 7.0, (0.6, -10.2, 3.8), (0.0, 4.5, 3.6)),
    (T_CHORUS + 14.0, (-1.6, -10.8, 4.4), (0.0, 5.5, 4.2)),
    (FINAL, (0.0, -11.5, 4.8), (0.0, 5.0, 4.6)),
    (FINAL + 5.0, (0.0, -12.8, 6.0), (0.0, 5.0, 6.2)),
    (T_END, (0.0, -12.8, 6.0), (0.0, 5.0, 6.2)),
]


def cam_state(t):
    for k in range(len(KEYS) - 1):
        if KEYS[k][0] <= t <= KEYS[k + 1][0]:
            a = ss((t - KEYS[k][0]) / (KEYS[k + 1][0] - KEYS[k][0]))
            p = Vector(KEYS[k][1]).lerp(Vector(KEYS[k + 1][1]), a); l = Vector(KEYS[k][2]).lerp(Vector(KEYS[k + 1][2]), a)
            return p, l
    return Vector(KEYS[-1][1]), Vector(KEYS[-1][2])


for f in range(NF):
    p, l = cam_state(f / FPS); cam.location = p; tgt.location = l
    cam.keyframe_insert('location', frame=f); tgt.keyframe_insert('location', frame=f)

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
