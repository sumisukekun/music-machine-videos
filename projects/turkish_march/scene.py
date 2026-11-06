"""Rondo alla Turca - an Ottoman-palace clockwork band.
Marbles shoot from brass spouts on a mosaic minaret's balcony and strike a crescent of 15 tuned brass bells
(one bell per pitch of Mozart's melody). Each bell rings, glows and wobbles; the marble drops into a catch funnel.
Two janissary drummer automata strike their davul drums, each driven by a visible eccentric cam -> push-rod -> arm;
a pair of zil cymbals on the minaret clash, driven by a cam at the tower foot through a long push-rod and two links.
Lanterns light up phrase by phrase; the dome and crescent blaze for the final chord."""
import os, sys, json, math, time, colorsys
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix, Quaternion

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.5) * FPS); T = np.arange(NF) / FPS
G = np.array([0, 0, -9.81])
rng = np.random.default_rng(13)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'
sc.cycles.samples = int(os.environ.get('SAMP', 6)); sc.cycles.use_adaptive_sampling = True; sc.cycles.adaptive_threshold = 0.06
sc.cycles.use_denoising = True
for k, v in (('denoiser', 'OPENIMAGEDENOISE'), ('use_fast_gi', True)):
    try: setattr(sc.cycles, k, v)
    except Exception: pass
sc.cycles.max_bounces = 3; sc.cycles.diffuse_bounces = 1; sc.cycles.glossy_bounces = 1; sc.cycles.transmission_bounces = 1; sc.cycles.volume_bounces = 0
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


def glowmat(name, col, gain, base=0.0, rough=0.25, metal=1.0):
    """base colour fixed; object alpha drives emission (object colour = emission colour)"""
    m = mat(name, col, rough, **{'Metallic': metal}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    oi = nt.nodes.new('ShaderNodeObjectInfo')
    nt.links.new(oi.outputs['Color'], b.inputs['Emission Color'])
    mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY_ADD'; mul.inputs[1].default_value = gain; mul.inputs[2].default_value = base
    nt.links.new(oi.outputs['Alpha'], mul.inputs[0]); nt.links.new(mul.outputs[0], b.inputs['Emission Strength'])
    return m


def mosaic(name, c1, c2, c3, scale=18.0, gold=(0.9, 0.65, 0.2)):
    """Iznik-style tile mosaic: voronoi cells coloured from a small palette, thin gold grout"""
    m = mat(name, c1, 0.25, **{'Coat Weight': 0.5}); nt = m.node_tree; b = nt.nodes['Principled BSDF']
    tc = nt.nodes.new('ShaderNodeTexCoord'); vo = nt.nodes.new('ShaderNodeTexVoronoi'); vo.inputs['Scale'].default_value = scale
    vo.feature = 'F1'; nt.links.new(tc.outputs['Object'], vo.inputs['Vector'])
    ramp = nt.nodes.new('ShaderNodeValToRGB'); cr = ramp.color_ramp; cr.interpolation = 'CONSTANT'
    cr.elements[0].color = (*c1, 1); cr.elements[1].position = 0.45; cr.elements[1].color = (*c2, 1)
    e3 = cr.elements.new(0.8); e3.color = (*c3, 1)
    nt.links.new(vo.outputs['Color'], ramp.inputs['Fac'])
    # grout: voronoi edge distance
    vo2 = nt.nodes.new('ShaderNodeTexVoronoi'); vo2.feature = 'DISTANCE_TO_EDGE'; vo2.inputs['Scale'].default_value = scale
    nt.links.new(tc.outputs['Object'], vo2.inputs['Vector'])
    ge = nt.nodes.new('ShaderNodeMath'); ge.operation = 'LESS_THAN'; ge.inputs[1].default_value = 0.05
    nt.links.new(vo2.outputs['Distance'], ge.inputs[0])
    mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'
    nt.links.new(ge.outputs[0], mix.inputs['Factor']); nt.links.new(ramp.outputs['Color'], mix.inputs[6])
    mix.inputs[7].default_value = (*gold, 1)
    nt.links.new(mix.outputs[2], b.inputs['Base Color'])
    mm = nt.nodes.new('ShaderNodeMath'); mm.operation = 'MULTIPLY'; mm.inputs[1].default_value = 0.9
    nt.links.new(ge.outputs[0], mm.inputs[0]); nt.links.new(mm.outputs[0], b.inputs['Metallic'])
    return m


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


def unit_rod(name, r, mt, verts=12):
    """cylinder of length 1 along +Z starting at its origin (for animated rods between moving points)"""
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=1, vertices=verts, location=(0, 0, 0.5))
    o = bpy.context.object; o.name = name
    bpy.ops.object.transform_apply(location=True)
    o.data.materials.append(mt); o.rotation_mode = 'QUATERNION'; return o


def key_rod(o, A, B):
    """animate a unit rod to span A[f] -> B[f] every frame (A, B: (NF,3))"""
    D = B - A; L = np.linalg.norm(D, axis=1)
    Q = np.zeros((NF, 4))
    for f in range(NF):
        q = Vector((0, 0, 1)).rotation_difference(Vector(D[f] / (L[f] + 1e-9))); Q[f] = (q.w, q.x, q.y, q.z)
    for i in range(3): key_series(o, 'location', i, A[:, i])
    for i in range(4): key_series(o, 'rotation_quaternion', i, Q[:, i])
    key_series(o, 'scale', 2, L)


def smoothstep(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


def bisect(fun, target, lo, hi, it=50):
    """vectorised: solve fun(x) = target for x in [lo, hi] (fun monotonic)"""
    lo = np.full_like(target, lo, dtype=float); hi = np.full_like(target, hi, dtype=float)
    up = fun(hi[:1])[0] > fun(lo[:1])[0]
    for _ in range(it):
        mid = (lo + hi) / 2; fm = fun(mid)
        go_hi = (fm < target) if up else (fm > target)
        lo = np.where(go_hi, mid, lo); hi = np.where(go_hi, hi, mid)
    return (lo + hi) / 2


# ---------------- materials ----------------
BRASS = mat('brass', (0.95, 0.68, 0.28), 0.22, **{'Metallic': 1.0})
DARKBRASS = mat('dbrass', (0.6, 0.38, 0.14), 0.35, **{'Metallic': 1.0})
STEEL = mat('steel', (0.75, 0.75, 0.8), 0.2, **{'Metallic': 1.0})
TOWER = mosaic('towerm', (0.02, 0.32, 0.55), (0.05, 0.55, 0.6), (0.85, 0.88, 0.92), 14)
PLINTH = mosaic('plinthm', (0.55, 0.06, 0.06), (0.05, 0.25, 0.55), (0.9, 0.85, 0.75), 22)
WALLM = mosaic('wallm', (0.01, 0.05, 0.16), (0.015, 0.13, 0.17), (0.3, 0.3, 0.36), 45, (0.5, 0.33, 0.1))
MARBLE = mat('marble', (0.86, 0.83, 0.78), 0.3)
ROBE_R = mat('robeR', (0.6, 0.05, 0.06), 0.6); ROBE_B = mat('robeB', (0.06, 0.18, 0.5), 0.6)
SKIN = mat('skin', (0.75, 0.52, 0.38), 0.55); WHITE = mat('hat', (0.92, 0.9, 0.85), 0.6)
DRUMSKIN = mat('dskin', (0.88, 0.8, 0.62), 0.6); DRUMSHELL = mat('dshell', (0.35, 0.12, 0.05), 0.4)

# ---------------- floor & back wall ----------------
FLOORM = mat('floor', (0.5, 0.45, 0.4), 0.35); nt = FLOORM.node_tree
ck = nt.nodes.new('ShaderNodeTexChecker'); ck.inputs['Scale'].default_value = 50
ck.inputs['Color1'].default_value = (0.45, 0.42, 0.37, 1); ck.inputs['Color2'].default_value = (0.03, 0.12, 0.18, 1)
nt.links.new(ck.outputs['Color'], nt.nodes['Principled BSDF'].inputs['Base Color'])
bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 4, 0)); bpy.context.object.data.materials.append(FLOORM)
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 5.2, 5)); w = bpy.context.object; w.scale = (16, 0.4, 10); w.data.materials.append(WALLM)
ARCHGLOW = emis('archglow', (1.0, 0.42, 0.12), 0.45)
for ax in (-2.6, 0.0, 2.6):                                  # three horseshoe-arch niches glowing warm
    bpy.ops.mesh.primitive_cube_add(size=1, location=(ax, 4.98, 1.7)); o = bpy.context.object; o.scale = (1.3, 0.05, 2.4); o.data.materials.append(ARCHGLOW)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.72, depth=0.05, location=(ax, 4.98, 2.95), rotation=(math.pi / 2, 0, 0)); bpy.context.object.data.materials.append(ARCHGLOW)
    for sx in (-1, 1):
        rod((ax + sx * 0.72, 4.92, 0), (ax + sx * 0.72, 4.92, 2.95), 0.07, MARBLE, 16)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.74, minor_radius=0.06, location=(ax, 4.92, 2.95), rotation=(math.pi / 2, 0, 0)); bpy.context.object.data.materials.append(BRASS)

# ---------------- the minaret ----------------
TC = np.array([0.0, 1.2])                                    # tower centre (x, y)
TR = 0.45
bpy.ops.mesh.primitive_cylinder_add(radius=TR, depth=3.3, vertices=16, location=(*TC, 1.65)); o = bpy.context.object; o.data.materials.append(TOWER)
bpy.ops.mesh.primitive_cylinder_add(radius=0.62, depth=0.35, vertices=16, location=(*TC, 0.175)); bpy.context.object.data.materials.append(PLINTH)
ZB = 2.9                                                     # balcony floor height
bpy.ops.mesh.primitive_cylinder_add(radius=0.68, depth=0.08, vertices=32, location=(*TC, ZB - 0.04)); bpy.context.object.data.materials.append(BRASS)
bpy.ops.mesh.primitive_torus_add(major_radius=0.68, minor_radius=0.025, location=(*TC, ZB + 0.18)); bpy.context.object.data.materials.append(BRASS)
for k in range(16):                                          # balustrade posts
    a = k * math.pi / 8; rod((TC[0] + 0.68 * math.cos(a), TC[1] + 0.68 * math.sin(a), ZB), (TC[0] + 0.68 * math.cos(a), TC[1] + 0.68 * math.sin(a), ZB + 0.18), 0.012, BRASS, 6)
bpy.ops.mesh.primitive_cylinder_add(radius=0.36, depth=0.6, vertices=16, location=(*TC, 3.6)); bpy.context.object.data.materials.append(TOWER)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.42, segments=32, ring_count=16, location=(*TC, 3.9)); DOME = bpy.context.object
bm = bmesh.new(); bm.from_mesh(DOME.data)
bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < -0.01], context='VERTS')
for v in bm.verts:                                           # onion-ish point at the top
    if v.co.z > 0: v.co.z *= 1.0 + 0.45 * (v.co.z / 0.42) ** 3
bm.to_mesh(DOME.data); bm.free()
for p in DOME.data.polygons: p.use_smooth = True
DOMEM = glowmat('domem', (0.95, 0.7, 0.3), 9.0, 0.0, 0.2, 1.0); DOME.data.materials.append(DOMEM)
rod((*TC, 4.5), (*TC, 4.72), 0.015, BRASS)
bpy.ops.mesh.primitive_torus_add(major_radius=0.11, minor_radius=0.022, location=(*TC, 4.84), rotation=(math.pi / 2, 0, 0)); CRES = bpy.context.object
CRES.data.materials.append(DOMEM)

# ---------------- bells: one per pitch, on a crescent in front of the tower ----------------
PITCHES = sorted(set(p for _, p, _ in LEAD)); NP = len(PITCHES)
R_BELL = 1.9; ZA = 1.05                                      # bell crescent radius, bell apex height
PHI = np.radians(np.linspace(-74, 74, NP))                   # left (low) -> right (high) as seen by the camera
BELLS = {}
BELLGLOW = glowmat('bellg', (0.95, 0.68, 0.28), 2.5, 0.0, 0.18, 1.0)
for i, p in enumerate(PITCHES):
    phi = PHI[i]; out = np.array([math.sin(phi), -math.cos(phi), 0.0])
    c = np.array([TC[0], TC[1], 0]) + out * R_BELL
    rb = 0.165 - 0.0055 * i                                  # low bells bigger
    zc = ZA - rb                                             # dome centre height
    # mosaic pedestal + brass yoke
    bpy.ops.mesh.primitive_cylinder_add(radius=0.045, depth=zc - 0.25, vertices=12, location=(c[0], c[1], (zc - 0.25) / 2)); bpy.context.object.data.materials.append(DARKBRASS)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.11, depth=0.06, vertices=16, location=(c[0], c[1], 0.03)); bpy.context.object.data.materials.append(PLINTH)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=0.05, vertices=16, location=(c[0], c[1], zc - 0.25)); bpy.context.object.data.materials.append(BRASS)
    rod((c[0], c[1], zc - 0.25), (c[0], c[1], zc - 0.02), 0.015, STEEL, 8)
    piv = link(bpy.data.objects.new(f'bellpiv{i}', None)); piv.location = (c[0], c[1], zc - 0.02)
    piv.rotation_mode = 'XYZ'; piv.rotation_euler = (0, 0, phi)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=rb, segments=32, ring_count=16, location=(0, 0, 0.02))
    b = bpy.context.object; bm = bmesh.new(); bm.from_mesh(b.data)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < -0.25 * rb], context='VERTS'); bm.to_mesh(b.data); bm.free()
    sol = b.modifiers.new('s', 'SOLIDIFY'); sol.thickness = 0.008
    for pp in b.data.polygons: pp.use_smooth = True
    b.data.materials.append(BELLGLOW); b.parent = piv
    bpy.ops.mesh.primitive_torus_add(major_radius=rb * 0.97, minor_radius=0.008, location=(0, 0, 0.02 - 0.25 * rb)); lip = bpy.context.object; lip.data.materials.append(BRASS); lip.parent = piv
    # catch funnel just outside the bell + return pipe into the floor
    n = math.radians(14); nvec = np.array([0, 0, math.cos(n)]) + out * math.sin(n)
    BELLS[p] = dict(i=i, phi=phi, out=out, c=c, rb=rb, zc=zc, piv=piv, bell=b, n=nvec)
bpy.context.view_layer.update()

# ---------------- marbles: ballistic flight from the balcony spouts ----------------
RBALL = 0.045; TF = 0.62; TE = 0.12; VPOST = (0.95, 1.05)      # flight time, emerge time, post-bounce (out, up)
BALL_COLS = [(0.85, 0.05, 0.1), (0.05, 0.5, 0.35), (0.08, 0.2, 0.85), (0.95, 0.7, 0.15), (0.95, 0.95, 1.0)]
for p, d in BELLS.items():
    d['hit'] = d['c'] + np.array([0, 0, d['zc']]) + d['n'] * (d['rb'] + 0.006 + RBALL)  # ball centre at impact
    d['spout'] = np.array([TC[0], TC[1], ZB + 0.11]) + d['out'] * 0.86
    v0 = (d['hit'] - d['spout'] - 0.5 * G * TF * TF) / TF; d['v0'] = v0
    # brass spout tube aligned with the launch direction
    u = v0 / np.linalg.norm(v0); rod(d['spout'] - u * 0.3, d['spout'] + u * 0.01, 0.042, BRASS, 16)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.042, minor_radius=0.01, location=tuple(d['spout'] + u * 0.01)); tt = bpy.context.object
    tt.rotation_mode = 'QUATERNION'; tt.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(Vector(u)); tt.data.materials.append(BRASS)
    # funnel where the post-bounce arc crosses (hit z - 0.13)
    vo_, vu_ = VPOST; tq = (vu_ + math.sqrt(vu_ ** 2 + 2 * 9.81 * 0.13)) / 9.81; d['tq'] = tq
    d['vpost'] = d['out'] * vo_ + np.array([0, 0, vu_])
    fpos = d['hit'] + d['vpost'] * tq + 0.5 * G * tq * tq; d['funnel'] = fpos
    bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=0.035, radius2=0.085, depth=0.12, end_fill_type='NOTHING', location=tuple(fpos - np.array([0, 0, 0.06])))
    fo = bpy.context.object; fo.modifiers.new('s', 'SOLIDIFY').thickness = 0.006; fo.data.materials.append(DARKBRASS)
    rod(fpos - np.array([0, 0, 0.12]), (fpos[0], fpos[1], 0.02), 0.02, DARKBRASS, 10)

events = sorted(LEAD, key=lambda e: e[0])
POOL = []                                                    # each = list of event indices
busy = []
for j, (t, p, v) in enumerate(events):
    t0, t1 = t - TF - TE - 0.08, t + BELLS[p]['tq'] + 0.25
    for k in range(len(POOL)):
        if busy[k] < t0: POOL[k].append(j); busy[k] = t1; break
    else: POOL.append([j]); busy.append(t1)
print('ball pool', len(POOL))
BMATS = [mat(f'ballm{k}', c, 0.04, **{'Coat Weight': 1.0, 'Transmission Weight': 0.0, 'Emission Color': (*c, 1), 'Emission Strength': 0.7}) for k, c in enumerate(BALL_COLS)]
for k, evs in enumerate(POOL):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=RBALL, segments=24, ring_count=12); o = bpy.context.object
    for pp in o.data.polygons: pp.use_smooth = True
    o.data.materials.append(BMATS[k % len(BMATS)])
    P = np.zeros((NF, 3)); vis = np.zeros(NF, bool)
    first = BELLS[events[evs[0]][1]]; P[:] = first['spout'] - first['v0'] / np.linalg.norm(first['v0']) * 0.25
    for j in evs:
        t, p, v = events[j]; d = BELLS[p]; u = d['v0'] / np.linalg.norm(d['v0'])
        start_in = d['spout'] - u * 0.25
        f0 = int(math.floor((t - TF - TE) * FPS)); f1 = int(math.ceil((t + d['tq'] + 0.12) * FPS))
        for f in range(max(0, f0 - 2), min(NF, f1 + 1)):
            a = T[f] - t
            if a < -TF:                                      # sliding out of the spout tube
                s = np.clip((a + TF + TE) / TE, 0, 1); P[f] = start_in + (d['spout'] - start_in) * s
            elif a < 0:
                tt_ = a + TF; P[f] = d['spout'] + d['v0'] * tt_ + 0.5 * G * tt_ * tt_
            elif a < d['tq']:
                P[f] = d['hit'] + d['vpost'] * a + 0.5 * G * a * a
            else:                                            # drops into the funnel
                s = min(1, (a - d['tq']) / 0.12); P[f] = d['funnel'] - np.array([0, 0, 0.16 * s])
            vis[f] = a > -TF - TE * 0.6 and a < d['tq'] + 0.1
        # hold at the funnel bottom until the next shot (hidden)
        nxt = [jj for jj in evs if jj > j]
        if nxt:
            dn = BELLS[events[nxt[0]][1]]; un = dn['v0'] / np.linalg.norm(dn['v0'])
            for f in range(min(NF, f1 + 1), NF): P[f] = dn['spout'] - un * 0.25
    for i in range(3): key_series(o, 'location', i, P[:, i])
    bpy.context.preferences.edit.keyframe_new_interpolation_type = 'CONSTANT'
    for f in range(NF):                                      # visibility (constant keys only at changes)
        if f == 0 or vis[f] != vis[f - 1]:
            o.hide_render = not vis[f]; o.keyframe_insert('hide_render', frame=f)
    bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
    spin = np.cumsum(np.r_[0, np.linalg.norm(np.diff(P, axis=0), axis=1)]) / RBALL
    o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 0, spin * 0.6)

# bell response: glow + wobble about the tangential axis
for p, d in BELLS.items():
    ts = [t for t, pp, v in LEAD if pp == p]; vs = [v for t, pp, v in LEAD if pp == p]
    a = np.zeros(NF); wob = np.zeros(NF)
    for t, v in zip(ts, vs):
        x = T - t; on = x >= -0.5 / FPS
        a = np.maximum(a, np.where(on, (0.25 + 0.75 * v / 127) * np.exp(-np.clip(x, 0, None) / 0.35), 0))
        wob += np.where(x >= 0, np.sin(x * 38) * np.exp(-x * 7) * 0.05 * v / 127, 0)
    fin = smoothstep((T - FINAL) / 0.15) * (0.5 + 0.1 * np.sin(T * 9))
    a = np.maximum(a, fin)
    key_alpha(d['bell'], a, (1.0, 0.75, 0.35))
    key_series(d['piv'], 'rotation_euler', 0, wob)

# ---------------- janissary drummer automata ----------------
drums = E['drum']; LHITS, RHITS = [], []
last = None
for t, v in drums:
    if t >= FINAL - 1e-3: LHITS.append((t, v)); RHITS.append((t, v)); continue
    if last is None or last == 'R': LHITS.append((t, v)); last = 'L'
    else: RHITS.append((t, v)); last = 'R'


def cam_theta(hits, rest_gap=0.24, swing=0.22):
    """unwrapped cam angle: 2*pi*k at rest, pi + 2*pi*k exactly at hit k (lobe down = strike)"""
    th = np.zeros(NF); hs = [h[0] for h in hits]
    for k, t in enumerate(hs):
        gp = hs[k] - hs[k - 1] if k else 9; gn = hs[k + 1] - t if k + 1 < len(hs) else 9
        du = min(swing, gp * 0.48); dd = min(rest_gap, gn * 0.48)
        th = np.where(T >= t - du, 2 * math.pi * k + math.pi * smoothstep((T - (t - du)) / du), th)
        th = np.where(T >= t, 2 * math.pi * k + math.pi + math.pi * smoothstep((T - t) / dd), th)
    return th


def drummer(X, s, hits, robe):
    """X: figure x; s=+1 drum on the right side, -1 left"""
    Y = 1.95; PZ = 0.55
    bpy.ops.mesh.primitive_cube_add(size=1, location=(X + s * 0.25, Y, PZ / 2)); o = bpy.context.object; o.scale = (1.2, 0.6, PZ); o.data.materials.append(PLINTH)
    bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=0.24, radius2=0.15, depth=0.85, location=(X, Y, PZ + 0.425)); bpy.context.object.data.materials.append(robe)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.17, minor_radius=0.03, location=(X, Y, PZ + 0.55)); bpy.context.object.data.materials.append(BRASS)   # sash
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.11, location=(X, Y, PZ + 0.98)); bpy.context.object.data.materials.append(SKIN)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=0.3, vertices=16, location=(X, Y + 0.03, PZ + 1.18), rotation=(-0.25, 0, 0)); bpy.context.object.data.materials.append(WHITE)   # bork hat
    bpy.ops.mesh.primitive_torus_add(major_radius=0.105, minor_radius=0.025, location=(X, Y, PZ + 1.06)); bpy.context.object.data.materials.append(BRASS)
    for dx in (-0.04, 0.04):                                 # moustache-ish dark eyes
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.014, location=(X + dx, Y - 0.1, PZ + 1.0)); bpy.context.object.data.materials.append(mat('eye', (0.02, 0.02, 0.02), 0.3))
    # davul drum on a stand beside the figure, skin facing up
    DX = X + s * 0.55; DZ = PZ + 0.5; DR = 0.21
    bpy.ops.mesh.primitive_cylinder_add(radius=DR, depth=0.26, vertices=32, location=(DX, Y, DZ - 0.13)); bpy.context.object.data.materials.append(DRUMSHELL)
    bpy.ops.mesh.primitive_cylinder_add(radius=DR + 0.004, depth=0.01, vertices=32, location=(DX, Y, DZ)); skin = bpy.context.object; skin.data.materials.append(DRUMSKIN)
    for zz in (DZ, DZ - 0.26):
        bpy.ops.mesh.primitive_torus_add(major_radius=DR + 0.006, minor_radius=0.012, location=(DX, Y, zz)); bpy.context.object.data.materials.append(BRASS)
    for k in range(3):
        a = k * 2 * math.pi / 3 + 0.5; rod((DX + 0.16 * math.cos(a), Y + 0.16 * math.sin(a), DZ - 0.26), (DX + 0.22 * math.cos(a), Y + 0.22 * math.sin(a), PZ), 0.012, STEEL, 6)
    # arm: pivot at the shoulder, mallet reaches the skin centre at the strike angle
    S = np.array([X + s * 0.15, Y, PZ + 0.85]); HEAD = np.array([DX, Y, DZ + 0.035])
    vec = HEAD - S; ARM = np.linalg.norm(vec); ph_hit = math.atan2(vec[2], abs(vec[0])); ph_rest = math.radians(18)
    arm = link(bpy.data.objects.new(f'arm{s}', None)); arm.location = tuple(S); arm.rotation_mode = 'XYZ'
    bpy.ops.mesh.primitive_cylinder_add(radius=0.035, depth=ARM - 0.04, vertices=12, location=(s * (ARM - 0.04) / 2, 0, 0), rotation=(0, math.pi / 2, 0))
    a_ = bpy.context.object; a_.data.materials.append(robe); a_.parent = arm
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.045, location=(s * ARM, 0, 0)); h_ = bpy.context.object; h_.data.materials.append(DRUMSKIN); h_.parent = arm   # mallet head
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=(0, 0, 0)); sh = bpy.context.object; sh.data.materials.append(BRASS); sh.parent = arm
    th = cam_theta(hits)
    # cam -> flat-foot follower -> push-rod (length Lr) -> pin on the arm 0.16 from the shoulder -> arm angle
    LA = 0.16; YR = Y - 0.06
    def Mx(ph): return S[0] + s * LA * np.cos(ph)
    def Mz(ph): return S[2] + LA * np.sin(ph)
    BX = S[0] + s * LA * math.cos((ph_rest + ph_hit) / 2)
    zb_rest = PZ + 0.2; Lr = math.hypot(Mx(np.array([ph_rest]))[0] - BX, Mz(np.array([ph_rest]))[0] - zb_rest)
    def zb_of(ph): return Mz(ph) - np.sqrt(np.maximum(1e-9, Lr ** 2 - (Mx(ph) - BX) ** 2))
    zb_hit = zb_of(np.array([ph_hit]))[0]
    e = (zb_rest - zb_hit) / 2; rc = 0.1; cz = zb_rest - e - rc
    zb = cz + rc + e * np.cos(th)                            # follower foot height set by the cam (theta=0 rest, pi strike)
    ph = bisect(zb_of, zb, ph_hit - 0.05, ph_rest + 0.05)
    key_series(arm, 'rotation_euler', 1, -s * ph)           # rotate about Y: positive ph raises the mallet
    M = np.stack([Mx(ph), np.full(NF, YR), Mz(ph)], 1)
    B = np.stack([np.full(NF, BX), np.full(NF, YR), zb], 1)
    r_ = unit_rod(f'prod{s}', 0.011, STEEL); key_rod(r_, B, M)
    bpy.ops.mesh.primitive_cube_add(size=1); fb = bpy.context.object; fb.scale = (0.2, 0.05, 0.012); fb.data.materials.append(BRASS)   # flat follower foot
    for i in range(3): key_series(fb, 'location', i, B[:, i] + (0.006 if i == 2 else 0))
    for sx_ in (-0.11, 0.11):                                # foot guides
        rod((BX + sx_, YR, zb.min() - 0.02), (BX + sx_, YR, zb.max() + 0.06), 0.008, DARKBRASS, 6)
    cam = link(bpy.data.objects.new(f'cam{s}', None)); cam.location = (BX, YR, cz); cam.rotation_mode = 'XYZ'
    bpy.ops.mesh.primitive_cylinder_add(radius=rc, depth=0.03, vertices=40, location=(0, 0, e), rotation=(math.pi / 2, 0, 0))
    cd = bpy.context.object; cd.data.materials.append(BRASS); cd.parent = cam
    bpy.ops.mesh.primitive_cylinder_add(radius=0.014, depth=0.034, vertices=6, location=(0.06, 0, e), rotation=(math.pi / 2, 0, 0)); bpy.context.object.data.materials.append(DARKBRASS); bpy.context.object.parent = cam
    bpy.ops.mesh.primitive_cylinder_add(radius=0.016, depth=0.18, vertices=12, location=(0, 0, 0), rotation=(math.pi / 2, 0, 0)); bpy.context.object.data.materials.append(STEEL); bpy.context.object.parent = cam
    for sy in (-0.08, 0.08):                                 # bearing blocks
        bpy.ops.mesh.primitive_cube_add(size=1, location=(BX, YR + sy, (PZ + cz) / 2)); o = bpy.context.object; o.scale = (0.05, 0.02, cz - PZ + 0.03); o.data.materials.append(DARKBRASS)
    key_series(cam, 'rotation_euler', 1, th)                 # disc centre offset e along local +Z: top = cz + rc + e*cos(theta)
    # drum skin flex + glow ring
    flex = np.zeros(NF)
    for t, v in hits:
        x = T - t; flex += np.where(x >= 0, np.exp(-np.clip(x, 0, None) * 18) * 0.012 * v / 127, 0)
    key_series(skin, 'location', 2, DZ - flex)
    return S


drummer(-1.18, -1, LHITS, ROBE_R)
drummer(1.18, 1, RHITS, ROBE_B)

# ---------------- zil cymbals on the minaret, cam at the tower foot ----------------
CR = E['crash']
PH5 = math.radians(5); X0 = TC[0] + 0.74 * math.sin(PH5); YC = TC[1] - 0.74 * math.cos(PH5)
th_c = cam_theta([(t, v) for t, v in CR], 0.3, 0.3)
BETA = math.radians(40); AX = np.array([math.cos(BETA), math.sin(BETA), 0.0])
PIVZ = 3.98; PIVX = 0.40; ARML = 0.45; LINK = 0.4; LA2 = 0.2; RL = 2.9
al_open, al_close = math.radians(16), math.asin((PIVX - 0.035) / ARML)
def rz_of(al): return PIVZ - LA2 * np.cos(al) - np.sqrt(np.maximum(1e-9, LINK ** 2 - (PIVX - LA2 * np.sin(al)) ** 2))
rz_open, rz_close = rz_of(np.array([al_open]))[0], rz_of(np.array([al_close]))[0]
ec = abs(rz_open - rz_close) / 2; rcc = 0.09
# the rod's foot rides on the cam: rest (theta=0) = open position
sgn = 1 if rz_open > rz_close else -1
RZ = (rz_open + rz_close) / 2 + sgn * ec * np.cos(th_c)
al = bisect(rz_of, RZ, al_open - 0.02, al_close + 0.02)
arms = []
for sx in (-1, 1):
    PV = np.array([X0, YC, PIVZ]) + sx * PIVX * AX
    piv = link(bpy.data.objects.new(f'cymp{sx}', None)); piv.location = tuple(PV); piv.rotation_mode = 'XYZ'; piv.rotation_euler[2] = BETA
    dtw = PV[:2] - TC; dtw /= np.linalg.norm(dtw); rod(PV, (TC[0] + dtw[0] * 0.33, TC[1] + dtw[1] * 0.33, PIVZ + 0.1), 0.018, BRASS, 8)   # bracket to the tower
    bpy.ops.mesh.primitive_cylinder_add(radius=0.016, depth=ARML, vertices=10, location=(0, 0, -ARML / 2)); a_ = bpy.context.object; a_.data.materials.append(BRASS); a_.parent = piv
    bpy.ops.mesh.primitive_cylinder_add(radius=0.19, depth=0.012, vertices=40, location=(-sx * 0.01, 0, -ARML), rotation=(0, math.pi / 2, 0))
    cy = bpy.context.object; cy.data.materials.append(BELLGLOW); cy.parent = piv
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.04, location=(-sx * 0.012, 0, -ARML)); bpy.context.object.data.materials.append(BRASS); bpy.context.object.parent = piv
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.03, location=(0, 0, 0)); bpy.context.object.data.materials.append(BRASS); bpy.context.object.parent = piv
    key_series(piv, 'rotation_euler', 1, sx * al)            # inner swing
    arms.append((sx, cy))
    rr_ = sx * (PIVX - LA2 * np.sin(al))
    A = np.stack([X0 + rr_ * AX[0], YC + rr_ * AX[1], PIVZ - LA2 * np.cos(al)], 1)
    R = np.stack([np.full(NF, X0), np.full(NF, YC), RZ], 1)
    lk = unit_rod(f'clink{sx}', 0.01, STEEL); key_rod(lk, R, A)
    print('link len', sx, round(float(np.linalg.norm(A - R, axis=1).min()), 4), round(float(np.linalg.norm(A - R, axis=1).max()), 4))
cyma = np.zeros(NF)
for t, v in CR:
    x = T - t; cyma = np.maximum(cyma, np.where(x >= -0.5 / FPS, np.exp(-np.clip(x, 0, None) / 0.4), 0))
for sx, cy in arms: key_alpha(cy, cyma, (1.0, 0.8, 0.4))
Rbot = np.stack([np.full(NF, X0), np.full(NF, YC), RZ - RL], 1); Rtop = np.stack([np.full(NF, X0), np.full(NF, YC), RZ], 1)
pr = unit_rod('cymrod', 0.013, STEEL); key_rod(pr, Rbot, Rtop)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.03, location=(0, 0, 0)); yk = bpy.context.object; yk.data.materials.append(BRASS)
for i in range(3): key_series(yk, 'location', i, Rtop[:, i])
inward = np.array([TC[0] - X0, TC[1] - YC]); inward /= np.linalg.norm(inward)
for zz in (1.2, 2.2):                                       # guide rings tied back to the tower
    bpy.ops.mesh.primitive_torus_add(major_radius=0.025, minor_radius=0.007, location=(X0, YC, zz)); bpy.context.object.data.materials.append(BRASS)
    rod((X0 + inward[0] * 0.025, YC + inward[1] * 0.025, zz), (X0 + inward[0] * 0.3, YC + inward[1] * 0.3, zz), 0.008, BRASS, 6)
foot = RZ - RL
bpy.ops.mesh.primitive_cube_add(size=1); ft = bpy.context.object; ft.scale = (0.2, 0.05, 0.012); ft.data.materials.append(BRASS)
for i in range(3): key_series(ft, 'location', i, Rbot[:, i] - (0.006 if i == 2 else 0))
# rod pushes up (open) when the lobe is up: foot height = czc + rcc + ec*cos(theta) (sgn handled below)
czc = (rz_open + rz_close) / 2 - RL - rcc - 0.012
camc = link(bpy.data.objects.new('camc', None)); camc.location = (X0, YC, czc); camc.rotation_mode = 'XYZ'
bpy.ops.mesh.primitive_cylinder_add(radius=rcc, depth=0.03, vertices=40, location=(0, 0, sgn * ec), rotation=(math.pi / 2, 0, 0)); bpy.context.object.data.materials.append(BRASS); bpy.context.object.parent = camc
bpy.ops.mesh.primitive_cylinder_add(radius=0.013, depth=0.034, vertices=6, location=(0.055, 0, sgn * ec), rotation=(math.pi / 2, 0, 0)); bpy.context.object.data.materials.append(DARKBRASS); bpy.context.object.parent = camc
bpy.ops.mesh.primitive_cylinder_add(radius=0.018, depth=0.3, vertices=12, location=(0, 0.08, 0), rotation=(math.pi / 2, 0, 0)); bpy.context.object.data.materials.append(STEEL); bpy.context.object.parent = camc
for sy in (-0.09, 0.09):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(X0, YC + sy, czc / 2)); o = bpy.context.object; o.scale = (0.05, 0.02, czc + 0.03); o.data.materials.append(DARKBRASS)
key_series(camc, 'rotation_euler', 1, th_c)
print('cym rod range', round(float(RZ.min()), 3), round(float(RZ.max()), 3), 'cam z', round(czc, 3), 'e', round(ec, 4))

# ---------------- lanterns on poles, lit phrase by phrase ----------------
SECS = E['secs']; sec_t = [s for n, s in SECS]
LANT = []
LCOLS = [(1.0, 0.5, 0.15), (1.0, 0.2, 0.15), (0.3, 0.8, 0.4), (0.25, 0.5, 1.0), (1.0, 0.75, 0.3), (0.9, 0.3, 0.8)]
LAMP = glowmat('lamp', (0.9, 0.85, 0.75), 18.0, 0.3, 0.1, 0.0)
POS = [(-2.45, 3.0), (-1.95, 3.9), (-0.7, 3.95), (0.7, 3.95), (1.95, 3.9), (2.45, 3.0)]
for k, (x, y) in enumerate(POS):
    rod((x, y, 0), (x, y, 2.3), 0.03, DARKBRASS, 8)
    rod((x, y, 2.3), (x * 0.92, y - 0.15, 2.36), 0.02, DARKBRASS, 6)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.11, depth=0.26, vertices=6, location=(x * 0.92, y - 0.15, 2.12)); lo = bpy.context.object; lo.data.materials.append(LAMP)
    bpy.ops.mesh.primitive_cone_add(vertices=6, radius1=0.14, radius2=0.01, depth=0.14, location=(x * 0.92, y - 0.15, 2.32)); bpy.context.object.data.materials.append(BRASS)
    bpy.ops.mesh.primitive_cone_add(vertices=6, radius1=0.11, radius2=0.03, depth=0.07, location=(x * 0.92, y - 0.15, 1.96), rotation=(math.pi, 0, 0)); bpy.context.object.data.materials.append(BRASS)
    LANT.append(lo)
LA_ = np.zeros((len(LANT), NF)) + 0.06
lit = []
for m, (n, s0) in enumerate(SECS[1:]):
    s1 = SECS[m + 2][1] if m + 2 < len(SECS) else FINAL
    seg = (T >= s0) & (T < s1)
    if n.startswith('CH') or n == 'CO':
        LA_[:, seg] = 1.0
    else:
        cur = m % 6
        if cur not in lit: lit.append(cur)
        for k in lit: LA_[k, seg] = 0.55
        LA_[cur, seg] = 0.55 + 0.45 * np.exp(-(T[seg] - s0) / 1.2)
LA_[:, T >= FINAL] = 1.0
for k, lo in enumerate(LANT):
    a = LA_[k] * (0.92 + 0.08 * np.sin(T * (7 + k) + k))
    key_alpha(lo, a, LCOLS[k])

# ---------------- dome / crescent finale ----------------
dg = smoothstep((T - FINAL) / 0.25) * (0.85 + 0.15 * np.sin(T * 8))
dg = np.maximum(dg, 0.04)
key_alpha(DOME, dg, (1.0, 0.75, 0.35)); key_alpha(CRES, np.maximum(dg, 0.25), (1.0, 0.85, 0.5))
fl = bpy.data.lights.new('domel', 'POINT'); fl.color = (1, 0.75, 0.4); fl.shadow_soft_size = 0.4
flo = link(bpy.data.objects.new('domel', fl)); flo.location = (TC[0], TC[1] - 0.8, 4.2)
for f in range(NF):
    if f in (0, NF - 1) or dg[f] > 0.05 or (f > 0 and dg[f - 1] > 0.05): fl.energy = 5 + 300 * dg[f]; fl.keyframe_insert('energy', frame=f)

# ---------------- world & lights ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True; nt = w.node_tree
bg = nt.nodes['Background']; bg.inputs['Strength'].default_value = 1.0
grad = nt.nodes.new('ShaderNodeTexGradient'); tc = nt.nodes.new('ShaderNodeTexCoord'); mp = nt.nodes.new('ShaderNodeMapping')
mp.inputs['Rotation'].default_value = (0, -math.pi / 2, 0)
ramp = nt.nodes.new('ShaderNodeValToRGB'); ramp.color_ramp.elements[0].color = (0.05, 0.02, 0.06, 1); ramp.color_ramp.elements[1].color = (0.06, 0.08, 0.22, 1)
nt.links.new(tc.outputs['Generated'], mp.inputs['Vector']); nt.links.new(mp.outputs['Vector'], grad.inputs['Vector'])
nt.links.new(grad.outputs['Fac'], ramp.inputs['Fac']); nt.links.new(ramp.outputs['Color'], bg.inputs['Color'])
for nm, loc, e, col, sz, tg in (('key', (-4.5, -5.5, 6.5), 550, (1.0, 0.82, 0.62), 4, (0, 1.2, 1.2)),
                                ('fill', (5, -4, 3), 350, (0.6, 0.7, 1.0), 4, (0, 1.2, 1.4)),
                                ('rim', (0, 3.5, 6), 400, (1.0, 0.6, 0.35), 3, (0, 1.0, 2.0))):
    l_ = bpy.data.lights.new(nm, 'AREA'); l_.energy = e; l_.color = col; l_.size = sz
    o = link(bpy.data.objects.new(nm, l_)); o.location = loc
    t_ = link(bpy.data.objects.new(nm + 't', None)); t_.location = tg
    c = o.constraints.new('TRACK_TO'); c.target = t_; c.track_axis = 'TRACK_NEGATIVE_Z'; c.up_axis = 'UP_Y'

# ---------------- camera: wide, slow, never close ----------------
camd = bpy.data.cameras.new('cam'); camd.lens = 30; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
tr = cam.constraints.new('TRACK_TO'); tr.target = tgt; tr.track_axis = 'TRACK_NEGATIVE_Z'; tr.up_axis = 'UP_Y'
ang = np.radians(10) * np.sin(2 * math.pi * T / 46 + 0.6)   # gentle orbit +-10 deg
dist = 9.6 - 0.5 * smoothstep((T - 1.0) / 6) + 0.3 * smoothstep((T - (FINAL - 1.0)) / 3.0)
h = 3.7 - 0.3 * smoothstep(T / 6)
CP = np.stack([TC[0] + dist * np.sin(ang), TC[1] - dist * np.cos(ang), h], 1)
TPz = 1.95 + 0.2 * smoothstep((T - (FINAL - 1.0)) / 3.0)
TP = np.stack([np.full(NF, TC[0]), np.full(NF, TC[1] - 0.2), TPz], 1)
for i in range(3): key_series(cam, 'location', i, CP[:, i]); key_series(tgt, 'location', i, TP[:, i])

DBG = os.environ.get('DBG')
if DBG:
    sc.animation_data_clear() if False else None
    cam.animation_data_clear(); tgt.animation_data_clear()
    cx, cy, cz, tx, ty, tz = map(float, DBG.split(','))
    cam.location = (cx, cy, cz); tgt.location = (tx, ty, tz)
print('objects', len(bpy.data.objects), 'NF', NF, 'bells', NP)
if MODE == 'test':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS)); sc.frame_set(f)
        sc.render.filepath = f'{HERE}/test_{f:04d}.png'
        t1 = time.time(); bpy.ops.render.render(write_still=True); print('frame', f, 'sec', round(time.time() - t1, 1))
elif MODE == 'range':
    sc.frame_start, sc.frame_end = int(args[1]), int(args[2])
    sc.render.image_settings.file_format = 'PNG'; sc.render.filepath = f'{HERE}/frames/f_'
    bpy.ops.render.render(animation=True)
