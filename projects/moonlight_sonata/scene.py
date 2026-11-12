"""Moonlight Sonata - 'Where is this lake?'  A black mirror lake at night under the full moon... which turns out to be
the closed lid of a grand piano in a moonlit room.  Side view (camera at -y), x = travel direction.
Every triplet note = a glass marble (trio A/B/C) bouncing on a glass lotus chime (it stays softly lit: the moon road);
every melody note = the silver marble on a silver key (long notes: thrown up by a see-saw whose bowl catches the bass
marble at that instant); every bass note = a dark marble dropped from the moon gondola into a bronze bowl.
Ending: the silver marble lands in the toy lighthouse (it lights up, its beam reveals the room and the piano), the trio
play the last arpeggio, are kicked off the lid by the launch tray and land on the real piano keys with three more marbles
from the gondola: the final chord.  layout.py plans every motion; this file builds and animates it.
Fonts: en_bold.ttf = Poppins Bold (SIL OFL); jp_sub.otf = Noto Sans CJK JP Bold (SIL OFL) subset (make_font.py).
python3 scene.py -- test t1 t2 ... | range f0 f1 | probe t ..."""
import os, sys, json, math, time
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[0] if args else 'test'
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json')); LAY = json.load(open(f'{HERE}/layout.json'))
FPS = 30; FINAL = E['final']; NF = int((FINAL + 7.5) * FPS); T = np.arange(NF) / FPS
G = LAY['G']; EV = LAY['ev']; YL = LAY['Y']; PIA = LAY['piano']; VX = LAY['VX']; T0 = E['t0']
BARS = {int(k): v for k, v in E['bars'].items()}
rng = np.random.default_rng(11)
T_LAMP = EV['lamp']['t']; T_TRAY = EV['tray']['t']; T_FIN1 = BARS[68] + 2.0; T_FIN2 = BARS[69]

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
sc.cycles.transparent_max_bounces = 6
sc.render.resolution_x, sc.render.resolution_y = 1080, 1920
sc.render.resolution_percentage = int(os.environ.get('PCT', 50)); sc.render.fps = FPS
sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.5
try: sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Base Contrast'
except Exception: pass
sc.view_settings.exposure = 0.4
sc.frame_start, sc.frame_end = 0, NF - 1
FONT_EN = bpy.data.fonts.load(f'{HERE}/en_bold.ttf'); FONT_JP = bpy.data.fonts.load(f'{HERE}/jp_sub.otf')


def link(o): sc.collection.objects.link(o); return o


def mat(name, col, rough=0.5, **kw):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']; b.inputs['Base Color'].default_value = (*col, 1); b.inputs['Roughness'].default_value = rough
    for k, v in kw.items():
        if k == 'Emission Color': v = (*v, 1)
        b.inputs[k].default_value = v
    return m


def emis(name, col, s):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    e = nt.nodes.new('ShaderNodeEmission'); e.inputs['Color'].default_value = (*col, 1); e.inputs['Strength'].default_value = s
    o = nt.nodes.new('ShaderNodeOutputMaterial'); nt.links.new(e.outputs[0], o.inputs[0]); return m


def glowmat(name, gain=8.0, base=0.0, rough=0.3, metal=0.0, coat=0.5, trans=0.0):
    """object colour = base colour; object alpha drives emission (hit flash / stays lit)"""
    m = mat(name, (1, 1, 1), rough, **{'Coat Weight': coat, 'Metallic': metal, 'Transmission Weight': trans}); nt = m.node_tree
    b = nt.nodes['Principled BSDF']; oi = nt.nodes.new('ShaderNodeObjectInfo')
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


def key_val(sock, vals, eps=1e-3):
    n = len(vals)
    for f in range(n):
        if 0 < f < n - 1 and abs(vals[f] - vals[f - 1]) < eps and abs(vals[f] - vals[f + 1]) < eps: continue
        sock.default_value = float(vals[f]); sock.keyframe_insert('default_value', frame=f)


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
        vs = [[bm.verts.new(((R_ + r_ * math.cos(v)) * math.cos(u), (R_ + r_ * math.cos(v)) * math.sin(u), r_ * math.sin(v)))
               for v in np.linspace(0, 2 * math.pi, nv, endpoint=False)] for u in np.linspace(0, 2 * math.pi, nu, endpoint=False)]
        for i in range(nu):
            for j in range(nv): bm.faces.new((vs[i][j], vs[(i + 1) % nu][j], vs[(i + 1) % nu][(j + 1) % nv], vs[i][(j + 1) % nv]))
    elif kind == 'bowl':      # open bowl, inner bottom at z=0, rim radius a[0], bottom radius a[1], height a[2], wall a[3]
        rt, rb, h, w = a; nu = 28
        rings = [(rb - w, 0.0), (rt - w, h), (rt, h), (rb, -w), (0.0, -w)]
        pts = []
        for (r_, z_) in rings:
            pts.append([bm.verts.new((r_ * math.cos(u), r_ * math.sin(u), z_)) for u in np.linspace(0, 2 * math.pi, nu, endpoint=False)])
        ctr = bm.verts.new((0, 0, 0))
        for k in range(len(pts) - 1):
            for i in range(nu): bm.faces.new((pts[k][i], pts[k][(i + 1) % nu], pts[k + 1][(i + 1) % nu], pts[k + 1][i]))
        for i in range(nu): bm.faces.new((ctr, pts[0][(i + 1) % nu], pts[0][i]))
    me = bpy.data.meshes.new(kind); bm.to_mesh(me); bm.free()
    if kind in ('sph', 'cone', 'torus', 'bowl', 'cyl'):
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


def _latin(s): return all(ord(c) < 0x2E80 for c in s)
TXT_K = 1.0; EN_K = 1.0
def text(s, loc, size, m, rot=(math.pi / 2, 0, 0), align='CENTER', ext=0.0):
    cu = bpy.data.curves.new('t', 'FONT'); cu.body = s; cu.extrude = ext
    cu.font = FONT_EN if _latin(s) else FONT_JP; cu.size = size
    cu.align_x = align; cu.align_y = 'CENTER'
    o = link(bpy.data.objects.new('txt', cu)); o.location = loc; o.rotation_euler = rot; o.data.materials.append(m); return o


def fit_w(o, w):
    bpy.context.view_layer.update()
    if w and o.dimensions.x > w: k = w / o.dimensions.x; o.scale = (k, k, k)
    return o


def pulse(times, decay=5.0, length=18, amp=1.0, base=None):
    a = np.zeros(NF)
    for t in times:
        f0 = int(round(t * FPS)); m_ = np.arange(max(0, f0), min(NF, f0 + length))
        a[m_] = np.maximum(a[m_], amp * np.exp(-(m_ - f0) / decay))
    return a


def smooth(Xa, k):
    Y_ = Xa.copy(); a = 1 - math.exp(-1 / FPS / k)
    for f in range(1, len(Xa)): Y_[f] = Y_[f - 1] + (Xa[f] - Y_[f - 1]) * a
    for f in range(len(Xa) - 2, -1, -1): Y_[f] = Y_[f + 1] + (Y_[f] - Y_[f + 1]) * a
    return Y_


def ss(u): u = min(1.0, max(0.0, u)); return u * u * (3 - 2 * u)


# ---------------- ball paths ----------------
class BallPath:
    def __init__(s, d): s.seg = d['seg']; s.hits = d['hits']; s.role = d['role']; s.R = d['R']; s.show = d['show']
    def at(s, t):
        for g in s.seg:
            if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9:
                p = np.array(g['p']); u = t - g['t0']
                if g['k'] == 'rest': return p
                if g['k'] == 'lin': return p + np.array(g['v']) * u
                return p + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
        return np.array(s.seg[0]['p']) if t < s.seg[0]['t0'] else s.at(s.seg[-1]['t1'])
BP = {n: BallPath(d) for n, d in LAY['balls'].items()}
POS = {n: np.array([b.at(t) for t in T]) for n, b in BP.items()}

# ---------------- materials ----------------
M_LID = mat('lid', (0.006, 0.006, 0.008), 0.05, **{'Coat Weight': 1.0, 'Coat Roughness': 0.02, 'Specular IOR Level': 0.6})
M_CASE = mat('case', (0.008, 0.008, 0.01), 0.12, **{'Coat Weight': 1.0, 'Coat Roughness': 0.05})
M_IVORY = mat('ivory', (0.85, 0.82, 0.74), 0.35)
M_EBONY = mat('ebony', (0.015, 0.015, 0.018), 0.25, **{'Coat Weight': 0.6})
M_GOLDT = mat('goldtrim', (0.9, 0.66, 0.3), 0.25, **{'Metallic': 1.0})
M_BRASS = mat('brass', (0.85, 0.6, 0.28), 0.22, **{'Metallic': 1.0})
M_GLASSSTEM = mat('stem', (0.15, 0.2, 0.3), 0.15, **{'Metallic': 0.6})
M_STEEL = mat('steel', (0.6, 0.62, 0.68), 0.25, **{'Metallic': 0.95})
M_DARKW = mat('darkwood', (0.06, 0.035, 0.02), 0.5)
M_FLOOR = mat('floor', (0.045, 0.026, 0.014), 0.85)
M_WALL = mat('wall', (0.012, 0.014, 0.022), 0.85)
M_RUG = mat('rug', (0.05, 0.015, 0.02), 0.95)
M_CURT = mat('curtain', (0.05, 0.06, 0.12), 0.9, **{'Sheen Weight': 0.5})
M_OBJ = objcol('objcol', 0.4)
M_FX = fxmat('fx', 6.0)
M_PAD = glowmat('pad', gain=5.0, base=0.0, rough=0.08, coat=1.0, trans=0.35)
M_MKEY = glowmat('mkey', gain=5.0, base=0.0, rough=0.18, metal=0.9, coat=0.5)
M_BOWL = glowmat('bowl', gain=5.0, base=0.0, rough=0.25, metal=1.0)
M_PKEYW = glowmat('pkeyw', gain=2.5, base=0.0, rough=0.35, coat=0.3)
M_PKEYB = glowmat('pkeyb', gain=2.5, base=0.0, rough=0.25, coat=0.6)
M_TXT_W = emis('txtw', (1.0, 0.97, 0.9), 3.0); M_TXT_G = emis('txtg', (1.0, 0.8, 0.45), 3.5); M_TXT_B = emis('txtb', (0.7, 0.85, 1.0), 3.0)
M_PAPER = mat('paper', (0.95, 0.88, 0.7), 0.7, **{'Emission Color': (1.0, 0.75, 0.45), 'Emission Strength': 0.6})
M_INK = mat('ink', (0.08, 0.05, 0.03), 0.6)
# marbles
M_TRIO = {k: mat('trio' + k, c, 0.04, **{'Transmission Weight': 0.85, 'IOR': 1.5, 'Coat Weight': 1.0, 'Emission Color': c, 'Emission Strength': 0.35})
          for k, c in (('A', (0.62, 0.7, 1.0)), ('B', (0.75, 0.95, 1.0)), ('C', (0.92, 0.8, 1.0)))}
M_MEL = mat('mel', (0.95, 0.95, 1.0), 0.12, **{'Metallic': 0.35, 'Coat Weight': 1.0, 'Emission Color': (0.92, 0.94, 1.0), 'Emission Strength': 1.6})
M_BASSB = mat('bassball', (0.05, 0.08, 0.3), 0.15, **{'Metallic': 0.5, 'Coat Weight': 1.0, 'Emission Color': (0.15, 0.25, 0.8), 'Emission Strength': 0.35})
M_FINB = mat('finball', (0.6, 0.75, 1.0), 0.04, **{'Transmission Weight': 0.8, 'IOR': 1.5, 'Coat Weight': 1.0, 'Emission Color': (0.6, 0.75, 1.0), 'Emission Strength': 0.8})

def pitch_col(p):
    """moon palette: low = violet-blue, middle = ice blue, high = pale gold"""
    u = max(0.0, min(1.0, (p - 40) / 36.0))
    cs = [(0.45, 0.4, 1.0), (0.4, 0.7, 1.0), (0.7, 0.95, 1.0), (1.0, 0.92, 0.7), (1.0, 0.8, 0.45)]
    s_ = u * (len(cs) - 1); i = min(len(cs) - 2, int(s_)); f = s_ - i
    return tuple(cs[i][c] * (1 - f) + cs[i + 1][c] * f for c in range(3))

# ---------------- marbles ----------------
BALL = {}
for n, b in BP.items():
    if b.role == 'trio': m_ = M_TRIO[n[-1]]
    elif b.role == 'mel': m_ = M_MEL
    elif b.role == 'bass': m_ = M_BASSB
    else: m_ = M_FINB
    o = obj(prim('sph', m_, 32, 16), n); o.scale = (b.R,) * 3
    P = POS[n]; key_loc(o, P)
    sp = np.zeros(NF)
    for f in range(1, NF): sp[f] = sp[f - 1] + (P[f, 0] - P[f - 1, 0]) / b.R
    o.rotation_mode = 'XYZ'; key_series(o, 'rotation_euler', 1, sp)
    if b.show > -0.5:                                  # inside the gondola until released
        vis_ = (T >= b.show - 1 / FPS).astype(float)
        for k in range(3): key_series(o, 'scale', k, np.maximum(1e-4, vis_ * b.R), 1e-6)
    BALL[n] = o
# silver marble: a little halo so it reads as 'the moon's child'

# ---------------- hits -> struck objects ----------------
def frame_from_n(n):
    n = np.array(n, float); n /= np.linalg.norm(n)
    xa = np.cross([0, 1, 0], n); xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return Matrix(((xa[0], ya[0], n[0]), (xa[1], ya[1], n[1]), (xa[2], ya[2], n[2])))
PIECES = []       # every struck surface, for the pieces dump / verify
def flash(o, t, col, base=0.0, keep=0.0, pre=0.0):
    fh = int(round(t * FPS)); o.color = (*col, pre)
    o.keyframe_insert('color', index=3, frame=0); o.keyframe_insert('color', index=3, frame=max(0, fh - 1))
    for df, a in ((0, 1.0), (3, 0.75), (10, max(keep, 0.3)), (30, keep)):
        o.color = (*col, a); o.keyframe_insert('color', index=3, frame=fh + df)
def dip(o, t, nv, d=0.03):
    fh = int(round(t * FPS)); base = o.location.copy(); nv = Vector(nv)
    for df, dd in ((-1, 0), (1, -d), (4, d * 0.25), (8, 0)):
        o.location = base + nv * dd; o.keyframe_insert('location', frame=max(0, fh + df))

PAD_R, PAD_TH = 0.21, 0.06
RINGS = []                                             # (t, centre, colour, size)
for n, b in BP.items():
    for h in b.hits:
        k = h['kind']; nrm = np.array(h['n']); nrm /= np.linalg.norm(nrm); p = np.array(h['p']); t = h['t']
        c = p - b.R * nrm
        if k == 'pad':                                  # glass lotus chime on a glass stem
            col = pitch_col(h['pitch'])
            o = obj(prim('cyl', M_PAD, 28), 'pad')
            o.matrix_world = Matrix.Translation(Vector(c - nrm * PAD_TH / 2)) @ frame_from_n(nrm).to_4x4() @ Matrix.Diagonal((PAD_R, PAD_R, PAD_TH, 1))
            flash(o, t, col, keep=0.22, pre=0.02); dip(o, t, -nrm, 0.025)
            for j in range(6):                          # petals
                a_ = j * math.pi / 3 + 0.3
                pe = obj(prim('cone', M_PAD, 12, 0.07, 0.0, 0.16), 'petal')
                q = c - nrm * 0.02 + np.array([math.cos(a_) * 0.18, math.sin(a_) * 0.18, -0.01])
                pe.location = q; pe.rotation_euler = (math.cos(a_ + math.pi / 2) * 0.9, math.sin(a_ + math.pi / 2) * -0.9, 0); pe.scale = (1, 0.6, 1)
                flash(pe, t, col, keep=0.15, pre=0.02)
            rod((c[0], c[1], c[2] - PAD_TH), (c[0], c[1], -0.02), 0.014, M_GLASSSTEM, 6)
            RINGS.append((t, c + np.array([0, 0, 0.01]), col, 0.42))
            PIECES.append(dict(kind=k, t=t, top=c.tolist(), n=nrm.tolist(), hw=PAD_R, hd=PAD_R, th=PAD_TH, ball=n))
        elif k == 'key':                                # silver melody key on a post
            col = (0.78, 0.82, 0.95)
            o = obj(prim('cube', M_MKEY), 'mkey')
            o.matrix_world = Matrix.Translation(Vector(c - nrm * 0.035)) @ frame_from_n(nrm).to_4x4() @ Matrix.Diagonal((0.2, 0.52, 0.07, 1))
            flash(o, t, col, keep=0.12, pre=0.03); dip(o, t, -nrm, 0.03)
            rod((c[0], c[1], c[2] - 0.07), (c[0], c[1], -0.02), 0.03, M_STEEL, 8)
            RINGS.append((t, c + np.array([0, 0, 0.01]), (1.0, 0.92, 0.75), 0.6))
            PIECES.append(dict(kind=k, t=t, top=c.tolist(), n=nrm.tolist(), hw=0.1, hd=0.26, th=0.07, ball=n))
        elif k == 'plate':                              # bronze bell plate (trio A plays the low C# pedal)
            o = obj(prim('cyl', M_BOWL, 32), 'plate'); o.location = c - np.array([0, 0, 0.03]); o.scale = (0.24, 0.24, 0.06)
            flash(o, t, (1.0, 0.7, 0.35), keep=0.1, pre=0.02); dip(o, t, (0, 0, -1), 0.03)
            rod((c[0], c[1], c[2] - 0.06), (c[0], c[1], -0.02), 0.04, M_BRASS, 8)
            RINGS.append((t, c + np.array([0, 0, 0.01]), (1.0, 0.7, 0.35), 0.8))
            PIECES.append(dict(kind=k, t=t, top=c.tolist(), n=[0, 0, 1], hw=0.24, hd=0.24, th=0.06, ball=n))

# bass bowls (plain) and see-saws
SG = EV['seesaw_geom']; TH0 = SG['th']; LSS = SG['L']
M_BEAM = mat('beam', (0.25, 0.18, 0.1), 0.4, **{'Coat Weight': 0.5})
def bowl_at(cbot, m=M_BOWL):
    o = obj(prim('bowl', m, 0.3, 0.2, 0.26, 0.035), 'bowl'); o.location = cbot; return o
BOWLS = []
for bb in EV['bass']:
    p = np.array(bb['p']); t = bb['t']; col = (1.0, 0.68, 0.32)
    cb = p - np.array([0, 0, LAY['R']['bass']])
    if bb['kind'] == 'bowl':
        o = bowl_at(cb); flash(o, t, col, keep=0.08, pre=0.02)
        rod((cb[0], cb[1], cb[2] - 0.04), (cb[0], cb[1], -0.02), 0.06, M_BRASS, 10)
        RINGS.append((t, cb + np.array([0, 0, 0.3]), col, 0.9))
        PIECES.append(dict(kind='bowl', t=t, top=cb.tolist(), n=[0, 0, 1], hw=0.22, hd=0.22, th=0.05, ball=bb['ball']))
    elif bb['kind'] == 'ssbowl':
        x = p[0]; piv = Vector((x, (YL['G'] + YL['M']) / 2, SG['zp']))
        pe = link(bpy.data.objects.new('ss_piv', None)); pe.location = piv; pe.rotation_mode = 'XYZ'
        # beam (local: along y, A end at -LSS (melody key), B end at +LSS (bowl)); rest angle +TH0 (A down)
        bm_ = box((0, 0, -0.03), (0.16, 2 * LSS + 0.5, 0.05), M_BEAM, name='beam'); bm_.parent = pe
        mt = [h for h in BP['mel'].hits if abs(h['t'] - t) < 1e-6][0]
        if mt['pitch'] is not None:
            kA = obj(prim('cube', M_MKEY), 'sskey'); kA.parent = pe; kA.location = (0, -LSS, SG['key_th'] / 2 - 0.005); kA.scale = (0.2, 0.52, SG['key_th'])
        else:                                            # S0: an unpitched brass cradle (the silver marble waits in it)
            kA = obj(prim('bowl', M_BRASS, 0.2, 0.12, 0.05, 0.02), 'sscup'); kA.parent = pe; kA.location = (0, -LSS, SG['key_th'] - 0.005)
            kA.color = (1, 1, 1, 0)
        bo = obj(prim('bowl', M_BOWL, 0.3, 0.2, 0.26, 0.035), 'ssbowl'); bo.parent = pe; bo.location = (0, LSS, SG['bowl_bot'])
        ang = np.full(NF, TH0); fh = t * FPS
        for f in range(NF):
            u = (f / FPS - t) / SG['t_tip']
            ang[f] = TH0 if u <= 0 else (TH0 - 2 * TH0 * min(1.0, u))
        key_series(pe, 'rotation_euler', 0, ang, 1e-6)
        # A-frame stand under the pivot
        for sy in (-0.12, 0.12):
            rod((x - 0.18, piv.y + sy, -0.02), (x, piv.y + sy, piv.z - 0.03), 0.025, M_BRASS, 6)
            rod((x + 0.18, piv.y + sy, -0.02), (x, piv.y + sy, piv.z - 0.03), 0.025, M_BRASS, 6)
        rod((x, piv.y - 0.15, piv.z), (x, piv.y + 0.15, piv.z), 0.035, M_BRASS, 8)
        flash(bo, t, col, keep=0.08, pre=0.02)
        if mt['pitch'] is not None: flash(kA, t, (0.78, 0.82, 0.95), keep=0.12, pre=0.03)
        RINGS.append((t, np.array([x, YL['G'], 1.05]), col, 0.9))
        if mt['pitch'] is not None: RINGS.append((t, np.array(mt['p']) - np.array([0, 0, 0.17]), (1.0, 0.92, 0.75), 0.6))
        PIECES.append(dict(kind='ssbowl', t=t, top=(cb).tolist(), n=[0, 0, 1], hw=0.22, hd=0.22, th=0.05, ball=bb['ball']))
        PIECES.append(dict(kind='sskey', t=t, top=(np.array(mt['p']) - np.array([0, 0, LAY['R']['mel']])).tolist(), n=[0, 0, 1], hw=0.1, hd=0.26, th=0.07, ball='mel', silent=mt['pitch'] is None))
    elif bb['kind'] == 'catbowl':
        CAT = dict(t=t, p=p, cb=cb)
# catapult: lever with the bowl (gondola lane) and the launch tray across the trio lanes at the lid's front edge
cat = EV['catapult']; tr = EV['tray']; xt = tr['x']
pe = link(bpy.data.objects.new('cat_bowl', None)); pe.location = CAT['cb']
bo = obj(prim('bowl', M_BOWL, 0.3, 0.2, 0.26, 0.035), 'catbowl'); bo.parent = pe
key_series(pe, 'location', 2, np.array([CAT['cb'][2] - (cat['drop'] * min(1, max(0, (t_ - T_TRAY) / cat['t_tip'])) if t_ > T_TRAY else 0) for t_ in T]), 1e-6)
flash(bo, T_TRAY, (1.0, 0.68, 0.32), keep=0.1, pre=0.02)
RINGS.append((T_TRAY, CAT['cb'] + np.array([0, 0, 0.3]), (1.0, 0.68, 0.32), 1.0))
tray = box((xt, (tr['y0'] + tr['y1']) / 2, tr['z'] - 0.03), (0.42, tr['y1'] - tr['y0'] + 0.1, 0.06), M_BRASS, name='tray')   # unpitched (felt-lined)
for sy in (tr['y0'] - 0.03, tr['y1'] + 0.03):
    lip = box((xt, sy, tr['z'] + 0.03), (0.42, 0.04, 0.12), M_BRASS, name='traylip'); lip.parent = tray; lip.matrix_parent_inverse = tray.matrix_world.inverted()
M_FELT = mat('felt', (0.35, 0.05, 0.06), 0.95)
felt = box((xt, (tr['y0'] + tr['y1']) / 2, tr['z'] - 0.002), (0.36, tr['y1'] - tr['y0'], 0.005), M_FELT, name='felt'); felt.parent = tray; felt.matrix_parent_inverse = tray.matrix_world.inverted()
key_series(tray, 'location', 2, np.array([tr['z'] - 0.03 + (cat['tray_up'] * min(1, max(0, (t_ - T_TRAY) / cat['t_tip'])) if t_ > T_TRAY else 0) for t_ in T]), 1e-6)
# lever from the bowl's stand to under the tray
lev = rod((xt, CAT['cb'][1], 0.3), (xt, tr['y1'], 0.3), 0.04, M_BRASS, 8)
for yy in (CAT['cb'][1] + 0.6, tr['y0'] + 0.5, tr['y1'] - 0.3):
    rod((xt, yy, -0.02), (xt, yy, 0.3), 0.035, M_BRASS, 8)
rod((xt, CAT['cb'][1], -0.02), (xt, CAT['cb'][1], CAT['cb'][2] - 0.05), 0.05, M_BRASS, 8)
for yy in np.linspace(tr['y0'] + 0.1, tr['y1'] - 0.1, 4):
    rod((xt, yy, 0.3), (xt, yy, tr['z'] - 0.06), 0.025, M_BRASS, 6)
PIECES.append(dict(kind='catbowl', t=T_TRAY, top=CAT['cb'].tolist(), n=[0, 0, 1], hw=0.22, hd=0.22, th=0.05, ball=[b['ball'] for b in EV['bass'] if b['kind'] == 'catbowl'][0]))

# hit rings (light halos, pool)
NRING = 16
ring_loc = np.zeros((NRING, NF, 3)); ring_scl = np.full((NRING, NF), 1e-3); ring_a = np.zeros((NRING, NF)); ring_c = [(1, 1, 1)] * NRING
for k, (t, c, col, sz) in enumerate(sorted(RINGS, key=lambda r: r[0])):
    r = k % NRING; f0 = int(round(t * FPS))
    for df in range(16):
        f = f0 + df
        if f >= NF: break
        u = df / 15; ring_loc[r, f] = c; ring_scl[r, f] = sz * (0.45 + 0.9 * (1 - (1 - u) ** 2)); ring_a[r, f] = 0.7 * (1 - u) ** 1.5
    ring_c[r] = col
RING_OBJ = []
for r in range(NRING):
    o = obj(prim('torus', M_FX, 1.0, 0.03, 40, 6), 'ring'); o.color = (*ring_c[r], 0)
    for f in range(1, NF):
        if ring_a[r, f] == 0: ring_loc[r, f] = ring_loc[r, f - 1]
    key_loc(o, ring_loc[r])
    for k in range(2): key_series(o, 'scale', k, ring_scl[r])
    key_series(o, 'scale', 2, ring_scl[r])
    key_series(o, 'color', 3, ring_a[r], 1e-3)
    o.visible_shadow = False

# ---------------- the grand piano ----------------
Y_A0 = PIA['y_a0']; WK = PIA['wk']; ZW = PIA['z_white']; ZB = PIA['z_black']
LF = PIA['lid_front']; KB = PIA['key_back']; WF = PIA['white_front']; BF = PIA['black_front']
Y_C8 = Y_A0 + 52 * WK
Y_BASS = Y_A0 - 1.6; Y_TREB = Y_C8 + 1.6; XTAIL = -4.6
def outline():
    """grand piano outline (top view), counter-clockwise: straight bass side, keyboard front, treble side, bentside, tail"""
    pts = [(LF, Y_BASS), (LF, Y_TREB), (14.0, Y_TREB)]
    for u in np.linspace(0, 1, 30)[1:]:                     # bentside: from the treble side in towards the tail
        x = 14.0 + (-1.0 - 14.0) * u
        pts.append((x, Y_TREB - (Y_TREB - 0.6) * (u * u * (3 - 2 * u))))
    cx, cy = -1.0, Y_BASS + 2.2                              # rounded tail
    rx, ry = -1.0 - XTAIL, 0.6 - cy
    for a in np.linspace(math.pi / 2, math.pi * 1.5, 24)[1:]:
        pts.append((cx + rx * math.cos(a), cy + ry * math.sin(a) if a <= math.pi else cy + 2.2 * math.sin(a)))
    pts.append((cx, Y_BASS))
    return pts
OUT = outline()
def slab(pts, z0, z1, m, name):
    bm = bmesh.new(); lo = [bm.verts.new((x, y, z0)) for x, y in pts]; hi = [bm.verts.new((x, y, z1)) for x, y in pts]
    bm.faces.new(hi); bm.faces.new(lo[::-1])
    for i in range(len(pts)): bm.faces.new((lo[i], lo[(i + 1) % len(pts)], hi[(i + 1) % len(pts)], hi[i]))
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new(name, me)); o.data.materials.append(m)
    return o
lid = slab(OUT, -0.22, 0.0, M_LID, 'lid')
rim = slab([(x - 0.0, y) for x, y in OUT], -4.6, -0.22, M_CASE, 'rim')
Z_FLOOR = -12.5
# gold trim line round the lid edge (thin), hinges on the straight side
for (x0, y0), (x1, y1) in zip(OUT, OUT[1:] + OUT[:1]):
    rod((x0, y0, -0.25), (x1, y1, -0.25), 0.03, M_GOLDT, 5)
# keyboard: front case block, key bed, cheeks, keys
box(((LF + KB) / 2, (Y_BASS + Y_TREB) / 2, (ZW - 0.25 - 0.22) / 2 - 0.0), (KB - LF + 0.02, Y_TREB - Y_BASS, -0.22 - ZW + 0.25), M_CASE, name='fallblock')
box(((KB + WF) / 2 + 0.1, (Y_BASS + Y_TREB) / 2, ZW - 0.55), (WF - KB + 0.2, Y_TREB - Y_BASS, 0.5), M_CASE, name='keybed')
box((WF + 0.12, (Y_BASS + Y_TREB) / 2, ZW - 0.32), (0.22, Y_TREB - Y_BASS, 0.42), M_CASE, name='keyslip')
for yc in (Y_BASS + 0.37, Y_TREB - 0.37):
    box(((KB + WF) / 2 + 0.06, yc, ZW + 0.05), (WF - KB + 0.12, 0.72, 0.62), M_CASE, name='cheek')
nm_ = text('BEETHOVEN', (LF + 0.205, (Y_BASS + Y_TREB) / 2 - 3.0, -1.9), 0.42, M_GOLDT, rot=(math.pi / 2, 0, math.pi / 2), ext=0.01)
KEYS = {}
for m in range(21, 109):
    if m % 12 in (0, 2, 4, 5, 7, 9, 11):
        o_, pc = divmod(m - 12, 12); wi = o_ * 7 + [0, 2, 4, 5, 7, 9, 11].index(pc) - 5
        yc = Y_A0 + (wi + 0.5) * WK
        o = box(((KB + WF) / 2, yc, ZW - 0.11), (WF - KB, WK - 0.012, 0.22), M_PKEYW, name='wkey'); o.color = (0.85, 0.82, 0.74, 0.0)
        KEYS[m] = dict(o=o, top=ZW, y=yc, hw=(WK - 0.012) / 2, x0=KB, x1=WF, black=False)
    else:
        o_, pc = divmod(m + 1 - 12, 12); wi = o_ * 7 + [0, 2, 4, 5, 7, 9, 11].index(pc) - 5
        yc = Y_A0 + wi * WK
        o = box(((KB + BF) / 2, yc, ZB - 0.11), (BF - KB, 0.16, 0.22), M_PKEYB, name='bkey'); o.color = (0.02, 0.02, 0.025, 0.0)
        KEYS[m] = dict(o=o, top=ZB, y=yc, hw=0.08, x0=KB, x1=BF, black=True)
# struck piano keys: glow + go down (and stay down under a resting marble)
khits = {}
for n, b in BP.items():
    for h in b.hits:
        if h['kind'] == 'pkey': khits.setdefault(h['pitch'], []).append((h['t'], np.linalg.norm(h['vout']) < 1e-9, n))
for m, lst in khits.items():
    K = KEYS[m]; o = K['o']; z0 = o.location.z; zz = np.full(NF, z0)
    for t, stay, n in lst:
        for f in range(NF):
            u = f / FPS - t
            if u < 0: continue
            if stay: zz[f] = min(zz[f], z0 - 0.09 * min(1, u / 0.06))
            elif u < 0.5: zz[f] = min(zz[f], z0 - 0.09 * (min(1, u / 0.06) if u < 0.1 else max(0, 1 - (u - 0.1) / 0.3)))
    key_series(o, 'location', 2, zz, 1e-5)
    col = (0.75, 0.85, 1.0) if not K['black'] else (0.5, 0.65, 1.0)
    for t, stay, n in lst: pass
    fh = [int(round(t * FPS)) for t, s_, n in lst]
    o.keyframe_insert('color', index=3, frame=0)
    base = o.color[:3]
    for f_ in fh:
        o.color = (*base, 0.0); o.keyframe_insert('color', index=3, frame=f_ - 1)
        o.color = (*base, 1.0); o.keyframe_insert('color', index=3, frame=f_)
        o.color = (*base, 0.35); o.keyframe_insert('color', index=3, frame=f_ + 12)
    for t, stay, n in lst:
        PIECES.append(dict(kind='pkey', t=t, top=[KB + 0.5, K['y'], K['top']], n=[0, 0, 1], hw=(WF - KB) / 2, hd=K['hw'], th=0.22, ball=n, pitch=m))
# marbles resting on the keys ride them down
for n, b in BP.items():
    for h in b.hits:
        if h['kind'] == 'pkey' and np.linalg.norm(h['vout']) < 1e-9:
            o = BALL[n]; f0 = int(round(h['t'] * FPS))
            zz = POS[n][:, 2].copy()
            for f in range(f0, NF): zz[f] = POS[n][f, 2] - 0.09 * min(1, (f / FPS - h['t']) / 0.06)
            o.animation_data.action = o.animation_data.action
            for f in range(f0, NF, 1):
                o.location.z = float(zz[f]); o.keyframe_insert('location', index=2, frame=f)
                if f > f0 + 3: break
# legs, lyre, pedals, bench
for (lx, ly) in ((LF - 2.0, Y_BASS + 1.2), (LF - 2.0, Y_TREB - 1.2), (XTAIL + 4.5, Y_BASS + 3.0)):
    box((lx, ly, (Z_FLOOR + -4.6) / 2), (1.0, 1.0, -4.6 - Z_FLOOR), M_CASE, name='leg')
    rod((lx, ly, Z_FLOOR + 0.25), (lx, ly, Z_FLOOR), 0.3, M_BRASS, 12)
box((LF - 1.5, (Y_BASS + Y_TREB) / 2, -7.5), (0.5, 1.6, 5.8), M_CASE, name='lyre')
for dy in (-0.45, 0, 0.45):
    box((LF - 0.9, (Y_BASS + Y_TREB) / 2 + dy, Z_FLOOR + 0.6), (1.2, 0.18, 0.08), M_BRASS, name='pedal')
box((WF + 6.5, (Y_BASS + Y_TREB) / 2 - 1.0, Z_FLOOR + 6.0), (4.2, 9.5, 0.9), M_CASE, name='bench')
box((WF + 6.5, (Y_BASS + Y_TREB) / 2 - 1.0, Z_FLOOR + 6.5), (4.0, 9.3, 0.25), mat('cushion', (0.2, 0.05, 0.06), 0.9), name='cushion')
for bx_ in (-1.7, 1.7):
    for by_ in (-4.2, 2.2):
        box((WF + 6.5 + bx_, (Y_BASS + Y_TREB) / 2 + by_, (Z_FLOOR + Z_FLOOR + 6.0) / 2), (0.35, 0.35, 6.0), M_CASE, name='benchleg')
# sheet music lying on the lid near the tail (keywords): Op. 27 No. 2
SHX, SHY = 9.5, 4.6
sh = box((SHX, SHY, 0.012), (4.2, 3.0, 0.02), M_PAPER, name='sheet')
sh.rotation_euler = (0, 0, 0.12)
for k_, (s_, z_) in enumerate((("Sonata quasi una Fantasia", 0.9), ("Op. 27 No. 2", 0.3), ("Moonlight", -0.4))):
    tx = text(s_, (SHX, SHY + z_, 0.03), 0.42 if k_ < 2 else 0.6, M_INK, rot=(0, 0, 0.12)); fit_w(tx, 3.8)
for k_ in range(5):
    for j in range(5):
        rod((SHX - 1.9, SHY - 1.2 - k_ * 0.1, 0.026), (SHX + 1.9, SHY - 0.8 - k_ * 0.1, 0.026), 0.006, M_INK, 4)

# ---------------- room: floor, rug, back wall with the big window, curtains, the moon ----------------
box((10, 0, Z_FLOOR - 0.25), (140, 140, 0.5), M_FLOOR, name='floor')
box((12, -2, Z_FLOOR + 0.02), (52, 34, 0.05), M_RUG, name='rug')
YW = 26.0
WX0, WX1, WZ0, WZ1 = -24.0, 50.0, -7.0, 26.0
box((10, YW + 0.5, 30.5), (150, 1.0, 9.0), M_WALL, name='wall_top')
box((10, YW + 0.5, (Z_FLOOR + WZ0) / 2), (150, 1.0, WZ0 - Z_FLOOR), M_WALL, name='wall_bot')
box((WX0 - 25, YW + 0.5, 10), (50, 1.0, 50), M_WALL, name='wall_l'); box((WX1 + 25, YW + 0.5, 10), (50, 1.0, 50), M_WALL, name='wall_r')
for (a, b_) in (((WX0, WZ0), (WX1, WZ0)), ((WX0, WZ1), (WX1, WZ1)), ((WX0, WZ0), (WX0, WZ1)), ((WX1, WZ0), (WX1, WZ1))):
    rod((a[0], YW, a[1]), (b_[0], YW, b_[1]), 0.45, M_DARKW, 8)
box(((WX0 + WX1) / 2, YW - 1.0, WZ0 - 0.4), (WX1 - WX0 + 2, 2.4, 0.6), M_DARKW, name='sill')
for cx_ in (WX0 - 2.0, WX1 + 2.0):                         # curtains (folds)
    for j in range(7):
        rod((cx_ + (j - 3) * 0.9, YW - 1.5 - 0.3 * (j % 2), Z_FLOOR), (cx_ + (j - 3) * 0.9, YW - 1.5 - 0.3 * (j % 2), 33), 0.6, M_CURT, 10)
box((-40, 0, 10), (1, 140, 50), M_WALL, name='wall_side')
# night sky outside: stars and the moon
SKY = bpy.data.materials.new('sky'); SKY.use_nodes = True; nt = SKY.node_tree; nt.nodes.clear()
tc = nt.nodes.new('ShaderNodeTexCoord'); vor = nt.nodes.new('ShaderNodeTexVoronoi'); vor.inputs['Scale'].default_value = 90.0
try: vor.feature = 'F1'
except Exception: pass
mr = nt.nodes.new('ShaderNodeMapRange'); mr.inputs['From Min'].default_value = 0.0; mr.inputs['From Max'].default_value = 0.06
mr.inputs['To Min'].default_value = 1.0; mr.inputs['To Max'].default_value = 0.0
pw = nt.nodes.new('ShaderNodeMath'); pw.operation = 'POWER'; pw.inputs[1].default_value = 3.0
sep = nt.nodes.new('ShaderNodeSeparateXYZ'); grad = nt.nodes.new('ShaderNodeMapRange')
grad.inputs['From Min'].default_value = -20; grad.inputs['From Max'].default_value = 80
gr = nt.nodes.new('ShaderNodeValToRGB'); gr.color_ramp.elements[0].color = (0.006, 0.011, 0.03, 1); gr.color_ramp.elements[1].color = (0.001, 0.002, 0.007, 1)
em1 = nt.nodes.new('ShaderNodeEmission'); em1.inputs['Strength'].default_value = 1.0
em2 = nt.nodes.new('ShaderNodeEmission'); em2.inputs['Color'].default_value = (0.85, 0.9, 1.0, 1)
mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = 4.0
add = nt.nodes.new('ShaderNodeAddShader'); out = nt.nodes.new('ShaderNodeOutputMaterial')
nt.links.new(tc.outputs['Object'], vor.inputs['Vector']); nt.links.new(vor.outputs['Distance'], mr.inputs['Value'])
nt.links.new(mr.outputs[0], pw.inputs[0]); nt.links.new(pw.outputs[0], mul.inputs[0]); nt.links.new(mul.outputs[0], em2.inputs['Strength'])
nt.links.new(tc.outputs['Object'], sep.inputs[0]); nt.links.new(sep.outputs['Y'], grad.inputs['Value']); nt.links.new(grad.outputs[0], gr.inputs['Fac'])
nt.links.new(gr.outputs['Color'], em1.inputs['Color'])
nt.links.new(em1.outputs[0], add.inputs[0]); nt.links.new(em2.outputs[0], add.inputs[1]); nt.links.new(add.outputs[0], out.inputs[0])
sky = obj(prim('plane', SKY), 'sky'); sky.location = (12, 150, 20); sky.rotation_euler = (math.pi / 2, 0, 0); sky.scale = (400, 260, 1)
sky.visible_shadow = False
MOON_C = Vector((15.0, 120.0, -6.5)); MOON_R = 3.6
MOON = bpy.data.materials.new('moon'); MOON.use_nodes = True; nt = MOON.node_tree; nt.nodes.clear()
tc = nt.nodes.new('ShaderNodeTexCoord'); nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 2.2; nz.inputs['Detail'].default_value = 6
rp = nt.nodes.new('ShaderNodeValToRGB'); rp.color_ramp.elements[0].position = 0.42; rp.color_ramp.elements[0].color = (0.62, 0.66, 0.75, 1)
rp.color_ramp.elements[1].position = 0.62; rp.color_ramp.elements[1].color = (1.0, 0.98, 0.92, 1)
em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Strength'].default_value = 2.6; out = nt.nodes.new('ShaderNodeOutputMaterial')
nt.links.new(tc.outputs['Object'], nz.inputs['Vector']); nt.links.new(nz.outputs['Fac'], rp.inputs['Fac']); nt.links.new(rp.outputs['Color'], em.inputs['Color'])
nt.links.new(em.outputs[0], out.inputs[0])
moon = sphere(tuple(MOON_C), MOON_R, MOON, (48, 24), 'moon'); moon.visible_shadow = False
glowm = fxmat('moonglow', 1.0); mg = obj(prim('sph', glowm, 32, 16), 'moonhalo'); mg.location = MOON_C + Vector((0, 4, 0)); mg.scale = (MOON_R * 1.9,) * 3
mg.color = (0.5, 0.6, 0.9, 0.06); mg.visible_shadow = False

# ---------------- crescent cradle (start), gondola + cable, lighthouse ----------------
cr = EV['cradle']
cx = np.mean([cr[k]['p'][0] for k in 'ABC']); cz = np.mean([cr[k]['p'][2] for k in 'ABC'])
M_MOONGOLD = mat('moongold', (1.0, 0.82, 0.45), 0.2, **{'Metallic': 1.0, 'Emission Color': (1.0, 0.8, 0.45), 'Emission Strength': 0.5})
for k in 'ABC':
    p = np.array(cr[k]['p'])
    ring = obj(prim('torus', M_MOONGOLD, 1.0, 0.25, 32, 8), 'nest'); ring.location = p - np.array([0, 0, 0.17]); ring.scale = (0.19, 0.19, 0.19)
    # hatch: two little flaps that swing down at release
    tr_ = cr[k]['t_release']
    for sgn in (-1, 1):
        fl = link(bpy.data.objects.new('flap', None)); fl.location = p + np.array([sgn * 0.17, 0, -0.2]); fl.rotation_mode = 'XYZ'
        fb = box((-sgn * 0.085, 0, 0), (0.17, 0.3, 0.025), M_MOONGOLD, name='flapb'); fb.parent = fl
        key_series(fl, 'rotation_euler', 1, np.array([0.0 if t_ < tr_ else sgn * min(1.6, (t_ - tr_) * 14) for t_ in T]), 1e-5)
# simple crescent: two overlapping discs approximated by a ring of spheres
for u in np.linspace(0.06, 0.94, 24):
    a = math.pi * (0.62 + 1.25 * u)
    wgt = math.sin(math.pi * u)
    c_ = (cx - 0.15 + 1.05 * math.cos(a), YL['C'] + 0.55, cz + 0.85 + 1.05 * math.sin(a))
    sphere(c_, 0.06 + 0.16 * wgt, M_MOONGOLD, (16, 8), 'cres')
rod((cx - 0.15, YL['C'] + 0.55, -0.02), (cx - 0.15, YL['C'] + 0.55, cz - 0.15), 0.05, M_BRASS, 8)
rod((cx - 0.3, YL['A'], cz - 0.17), (cx + 0.5, YL['C'], cz - 0.17), 0.03, M_BRASS, 6)
for k in 'ABC':
    p = np.array(cr[k]['p']); rod((p[0], p[1], p[2] - 0.17), (cx - 0.15, YL['C'] + 0.55, p[2] - 0.17), 0.02, M_BRASS, 6)

# gondola: a golden crescent cable car with a glass hopper of night marbles
gz = EV['gondola']['z_rel']; GY = YL['G']
CABLE_Z = gz + 1.9
rod((-12, GY, CABLE_Z), (35.5, GY, CABLE_Z), 0.03, M_STEEL, 6)
for px_ in (-12.0, 35.5):                                   # the cable hangs from the ceiling
    rod((px_, GY, CABLE_Z), (px_ + (-6 if px_ < 0 else 6), GY, 48), 0.02, M_STEEL, 4)
    sphere((px_, GY, CABLE_Z), 0.14, M_BRASS)
gp = link(bpy.data.objects.new('gondola', None))
key_loc(gp, np.stack([np.array([VX * (t_ - T0) - 0.35 + 0.5 for t_ in T]), np.full(NF, GY), np.full(NF, gz)], 1))
def gpart(o): o.parent = gp; return o
M_HOP = mat('hopper', (0.75, 0.85, 1.0), 0.05, **{'Transmission Weight': 0.9, 'IOR': 1.3, 'Coat Weight': 1.0})
gpart(box((0, 0, 0.45), (0.9, 2.7, 0.6), M_HOP, name='hopper'))
gpart(box((0, 0, 0.12), (1.0, 2.8, 0.08), M_MOONGOLD, name='gfloor'))
gpart(box((0, 0, 0.8), (1.0, 2.8, 0.08), M_MOONGOLD, name='groof'))
for dy in (-1.3, 1.3):
    gpart(box((0, dy, 0.45), (1.0, 0.08, 0.75), M_MOONGOLD, name='gside'))
for u in np.linspace(0.0, 1.0, 18):                          # crescent ornament on the side facing the camera
    a = math.pi * (0.55 + 1.6 * u); wgt = math.sin(math.pi * u)
    gpart(sphere((0.55 * math.cos(a) * 0.9, -1.45, 0.45 + 0.55 * math.sin(a)), 0.04 + 0.08 * wgt, M_MOONGOLD, (12, 6), 'gcres'))
gpart(rod((0, 0, 0.84), (0, 0, CABLE_Z - gz), 0.04, M_BRASS, 8))
for dx in (-0.25, 0.25):
    gpart(rod((dx, -0.12, CABLE_Z - gz), (dx, 0.12, CABLE_Z - gz), 0.11, M_BRASS, 14))
# a few marbles visible inside the hopper (the stock goes down as they are dropped)
NB = len([b for b in BP.values() if b.role in ('bass', 'fin')])
stock = []
for j in range(14):
    o = obj(prim('sph', M_BASSB, 16, 8), 'stock'); o.parent = gp
    o.location = ((j % 3 - 1) * 0.24, -0.9 + (j // 3) * 0.42, 0.33 + 0.12 * (j % 2)); o.scale = (0.17,) * 3
    stock.append(o)
rel_t = sorted([b.show for b in BP.values() if b.role in ('bass', 'fin') and b.show > -0.5])
for j, o in enumerate(stock):
    # stock marble j disappears when the (NB - j)-th marble from the end has been released
    k = max(0, len(rel_t) - 1 - j * 2)
    tt_ = rel_t[k] if k < len(rel_t) else 999
    key_series(o, 'scale', 0, np.where(T < tt_, 0.17, 1e-4), 1e-6); key_series(o, 'scale', 1, np.where(T < tt_, 0.17, 1e-4), 1e-6); key_series(o, 'scale', 2, np.where(T < tt_, 0.17, 1e-4), 1e-6)
# hatch under the gondola blinks on every drop

# lighthouse (toy): tapered tower with red bands, gallery, glass lantern, open crown; the silver marble lands in the lamp
LP = np.array(EV['lamp']['p']); LX, LYc = LP[0], LP[1]
M_WHITE = mat('lhwhite', (0.85, 0.85, 0.82), 0.4); M_RED = mat('lhred', (0.6, 0.06, 0.05), 0.4)
z_top = LP[2] - 0.18 - 0.55
nseg = 6
for j in range(nseg):
    z0_ = j * z_top / nseg; z1_ = (j + 1) * z_top / nseg
    r0_ = 0.62 - 0.25 * z0_ / z_top; r1_ = 0.62 - 0.25 * z1_ / z_top
    o = obj(prim('cone', M_RED if j % 2 else M_WHITE, 32, 1.0, r1_ / r0_, 1.0), 'tower'); o.location = (LX, LYc, (z0_ + z1_) / 2); o.scale = (r0_, r0_, z1_ - z0_)
o = obj(prim('cyl', M_DARKW, 32), 'gallery'); o.location = (LX, LYc, z_top + 0.04); o.scale = (0.55, 0.55, 0.08)
for a in np.linspace(0, 2 * math.pi, 16, endpoint=False):
    rod((LX + 0.52 * math.cos(a), LYc + 0.52 * math.sin(a), z_top + 0.08), (LX + 0.52 * math.cos(a), LYc + 0.52 * math.sin(a), z_top + 0.3), 0.012, M_DARKW, 4)
M_LANT = mat('lantern', (0.9, 0.95, 1.0), 0.05, **{'Transmission Weight': 0.95, 'IOR': 1.2})
lant = obj(prim('cyl', M_LANT, 32), 'lantern'); lant.location = (LX, LYc, z_top + 0.4); lant.scale = (0.34, 0.34, 0.6)
for a in np.linspace(0, 2 * math.pi, 6, endpoint=False):
    rod((LX + 0.34 * math.cos(a), LYc + 0.34 * math.sin(a), z_top + 0.1), (LX + 0.34 * math.cos(a), LYc + 0.34 * math.sin(a), z_top + 0.72), 0.018, M_DARKW, 4)
crown = obj(prim('torus', M_DARKW, 1.0, 0.18, 32, 8), 'crown'); crown.location = (LX, LYc, z_top + 0.74); crown.scale = (0.36, 0.36, 0.36)
cup = obj(prim('bowl', M_BRASS, 0.24, 0.12, 0.12, 0.03), 'lampcup'); cup.location = (LX, LYc, LP[2] - 0.18)
LAMP_ON = np.array([0.0 if t_ < T_LAMP else min(1.0, (t_ - T_LAMP) / 0.25) for t_ in T])
M_LAMPG = emis('lampglow', (1.0, 0.85, 0.55), 0.0)
lg = sphere((LX, LYc, LP[2]), 0.3, M_LAMPG, (16, 8), 'lampglow'); lg.visible_shadow = False
key_val(M_LAMPG.node_tree.nodes['Emission'].inputs['Strength'], 0.001 + 9.0 * LAMP_ON + 4.0 * pulse([T_LAMP], decay=8, length=40))
# rotating beam (two soft cones) + a spot light that sweeps the room
M_BEAMFX = fxmat('beamfx', 0.6); BEAMS = []
bp = link(bpy.data.objects.new('beampiv', None)); bp.location = (LX, LYc, LP[2]); bp.rotation_mode = 'XYZ'
for sgn in (1, -1):
    bc = obj(prim('cone', M_BEAMFX, 24, 0.05, 1.0, 1.0), 'beam'); bc.parent = bp
    bc.rotation_euler = (0, sgn * math.pi / 2, 0); bc.location = (sgn * 9.0, 0, 0); bc.scale = (1.4, 1.4, 18.0)
    bc.color = (1.0, 0.9, 0.65, 0.0); bc.visible_shadow = False
    BEAMS.append((bc, sgn))
YAWB = np.array([0.0 if t_ < T_LAMP else -0.9 * (t_ - T_LAMP) for t_ in T]) + math.pi / 2
key_series(bp, 'rotation_euler', 2, YAWB, 1e-5)
for sgn in (1, -1):
    sl = bpy.data.lights.new('beamspot', 'SPOT'); sl.spot_size = math.radians(14); sl.spot_blend = 0.6; sl.color = (1.0, 0.85, 0.6); sl.shadow_soft_size = 0.1
    so = link(bpy.data.objects.new('beamspot', sl)); so.parent = bp; so.rotation_euler = (0, sgn * math.pi / 2, 0); so.location = (0, 0, 0)
    for f in range(0, NF, 2): sl.energy = float(20000 * LAMP_ON[f] + 1e-3); sl.keyframe_insert('energy', frame=f)
pl_ = bpy.data.lights.new('lamppt', 'POINT'); pl_.color = (1.0, 0.8, 0.5); pl_.shadow_soft_size = 0.2
po_ = link(bpy.data.objects.new('lamppt', pl_)); po_.location = (LX, LYc, LP[2] + 0.1)
ROOM_ON = np.array([ss((t_ - T_FIN2 - 0.2) / 3.0) for t_ in T])
for f in range(0, NF, 2): pl_.energy = float(600 * LAMP_ON[f] + 2500 * ROOM_ON[f] + 1e-3); pl_.keyframe_insert('energy', frame=f)

# ---------------- keyword signs (paper lanterns on sticks behind the lanes, read in the first 30 s) ----------------
SIGNS = [(2.6, 'MOONLIGHT', 'SONATA', 'ベートーヴェン「月光」'), (8.4, 'Beethoven', 'Piano Sonata No. 14', 'ピアノソナタ第14番'),
         (14.2, 'Marble machine', 'Classical ASMR', 'ビー玉からくり・クラシック')]
YS = -0.35
for sx, en1, en2, jp in SIGNS:
    w_ = 3.3; zc_ = 5.0
    pnl = box((sx, YS + 0.03, zc_), (w_, 0.04, 1.75), M_PAPER, name='sign')
    fit_w(text(en1, (sx, YS - 0.02, zc_ + 0.48), 0.5, M_INK), w_ - 0.3)
    fit_w(text(en2, (sx, YS - 0.02, zc_ - 0.05), 0.42, M_INK), w_ - 0.3)
    fit_w(text(jp, (sx, YS - 0.02, zc_ - 0.55), 0.3, M_INK), w_ - 0.4)
    for dx in (-w_ / 2 + 0.2, w_ / 2 - 0.2):
        rod((sx + dx, YS + 0.08, -0.02), (sx + dx, YS + 0.08, zc_ + 0.95), 0.04, M_DARKW, 6)
        sphere((sx + dx, YS + 0.08, zc_ + 1.03), 0.08, M_TXT_G)

# ---------------- world & lights ----------------
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes['Background']; bg.inputs['Color'].default_value = (0.006, 0.009, 0.02, 1); bg.inputs['Strength'].default_value = 0.5
sun = bpy.data.lights.new('moonlight', 'SUN'); sun.energy = 0.8; sun.angle = 0.03; sun.color = (0.7, 0.8, 1.0)
so = link(bpy.data.objects.new('moonlight', sun))
d_ = Vector((-0.12, -0.88, -0.46)).normalized(); so.rotation_euler = d_.to_track_quat('-Z', 'Y').to_euler()
# a soft cool key light riding with the camera (readability of the marbles)

# ---------------- camera ----------------
def X(t): return VX * (t - T0) - 0.35
FOC = np.zeros((NF, 3)); WID = np.zeros(NF); PIT = np.zeros(NF); YAW = np.zeros(NF)
for f in range(NF):
    t = T[f]
    base = np.array([X(max(t, 0.6)) + 0.35, -3.3, 3.65]); wid = 4.8; pit = 16.0; yw = 0.0
    if t < 2.0: base = np.array([0.55, -3.3, 3.65])
    # reveal: the lighthouse lights up; the camera rises and swings round towards the keyboard end (edge... keys!)
    a1 = ss((t - T_LAMP - 0.4) / 3.4)
    rev = np.array([29.6, -2.6, -1.4]); base = base * (1 - a1) + rev * a1; wid = wid * (1 - a1) + 7.4 * a1
    pit = pit * (1 - a1) + 27 * a1; yw = 68 * a1
    # bar 68: follow the launch onto the keys
    a2 = ss((t - T_TRAY + 0.3) / 1.2) * (1 - ss((t - T_FIN2 - 0.3) / 2.4))
    k2 = np.array([31.0, -2.9, -1.7]); base = base * (1 - a2) + k2 * a2; wid = wid * (1 - a2) + 5.4 * a2
    # finale: the whole piano in the moonlit room
    a3 = ss((t - T_FIN2 - 0.3) / 3.2)
    room = np.array([13.5, 0.5, -4.0]); base = base * (1 - a3) + room * a3; wid = wid * (1 - a3) + 50.0 * a3
    pit = pit * (1 - a3) + 48 * a3; yw = yw * (1 - a3) + 35 * a3
    FOC[f] = base; WID[f] = wid; PIT[f] = pit; YAW[f] = yw
for k in range(3): FOC[:, k] = smooth(FOC[:, k], 0.35)
WID = smooth(WID, 0.35); PIT = smooth(PIT, 0.35); YAW = smooth(YAW, 0.35)
LENS = 32.0
camd = bpy.data.cameras.new('cam'); camd.lens = LENS; camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24; camd.clip_end = 600
cam = link(bpy.data.objects.new('cam', camd)); sc.camera = cam
tgt = link(bpy.data.objects.new('tgt', None))
trc = cam.constraints.new('TRACK_TO'); trc.target = tgt; trc.track_axis = 'TRACK_NEGATIVE_Z'; trc.up_axis = 'UP_Y'
CP = np.zeros((NF, 3)); TP = np.zeros((NF, 3))
for f in range(NF):
    D = WID[f] / (9 / 16) * LENS / 24
    pr = math.radians(PIT[f]); yw = math.radians(YAW[f])
    CP[f] = FOC[f] + np.array([D * math.cos(pr) * math.sin(yw), -D * math.cos(pr) * math.cos(yw), D * math.sin(pr)])
    TP[f] = FOC[f]
key_loc(cam, CP); key_loc(tgt, TP)
# lighthouse beams: visible only while they point away from the camera (no beam over the lens)
for bc, sgn in BEAMS:
    al = np.zeros(NF)
    for f in range(NF):
        yb = YAWB[f]; d = np.array([math.cos(yb), math.sin(yb)]) * sgn
        tc = CP[f, :2] - np.array([LX, LYc]); tc /= max(1e-6, np.linalg.norm(tc))
        al[f] = 0.16 * LAMP_ON[f] * max(0.0, -(d @ tc)) ** 1.5
    key_series(bc, 'color', 3, al, 1e-3)
# the room comes up for the finale (warm light from above + the lamp)
rl = bpy.data.lights.new('roomlight', 'AREA'); rl.size = 30.0; rl.color = (1.0, 0.82, 0.62)
ro = link(bpy.data.objects.new('roomlight', rl)); ro.location = (12, -6, 30)
for f in range(0, NF, 3): rl.energy = float(1e-3 + 42000 * ROOM_ON[f]); rl.keyframe_insert('energy', frame=f)
kl = bpy.data.lights.new('key', 'AREA'); kl.size = 6.0; kl.color = (0.75, 0.82, 1.0)
ko = link(bpy.data.objects.new('key', kl))
key_loc(ko, FOC + np.array([-3.0, -6.0, 7.0])); ko.rotation_euler = (math.radians(42), 0, math.radians(-25))
for f in range(0, NF, 3):
    kl.energy = float(260 * (1 - 0.6 * ss((T[f] - T_FIN1) / 3.0))); kl.keyframe_insert('energy', frame=f)
# ---------------- HUD (rides with the camera): hook, reveal caption, end card ----------------
HUD_D = 3.0
HH = HUD_D * 24 / LENS; HWd = HH * 9 / 16
def hud_text(s, x, y, size, m, align='CENTER'):
    o = text(s, (x * HWd, y * HH / 2, -HUD_D), size, m, rot=(0, 0, 0), align=align); o.parent = cam; o.visible_shadow = False; return o
def vis(o, a):
    s0 = tuple(o.scale)
    for c in range(3): key_series(o, 'scale', c, np.maximum(0.001, a * s0[c]), 1e-3)
def plate(y, h, a, w=0.96):
    p_ = obj(prim('plane', M_HUDBG), 'hudbg'); p_.parent = cam; p_.location = (0, y * HH / 2, -HUD_D - 0.01); p_.scale = (HWd * w, h * HH / 2, 1)
    p_.visible_shadow = False; vis(p_, a); return p_
M_HUDBG = mat('hudbg', (0.0, 0.0, 0.02), 1.0, **{'Alpha': 0.45})
def win(t0, t1, fi=0.3, fo=0.4): return np.array([ss((t - t0) / fi) * (1 - ss((t - t1) / fo)) for t in T])
hook_a = win(-1, 4.3)
o = fit_w(hud_text('MOONLIGHT SONATA', 0, 0.86, 0.13, M_TXT_W), HWd * 1.85); vis(o, hook_a)
o = fit_w(hud_text('Beethoven · Piano Sonata No. 14', 0, 0.785, 0.062, M_TXT_G), HWd * 1.8); vis(o, hook_a)
o = fit_w(hud_text('Where is this lake?', 0, 0.69, 0.095, M_TXT_B), HWd * 1.8); vis(o, hook_a)
o = hud_text('この湖、どこにある？ ベートーヴェン「月光」', 0, 0.62, 0.062, M_TXT_W); fit_w(o, HWd * 1.8); vis(o, hook_a)
plate(0.76, 0.2, hook_a)
# small running title (first 30 s: keywords on screen the whole time)
ttl_a = win(4.3, 31.0)
o = fit_w(hud_text('Beethoven – Moonlight Sonata | marble machine', 0, 0.935, 0.07, M_TXT_W), HWd * 1.9); vis(o, ttl_a)
o = hud_text('ベートーヴェン「月光」第1楽章 ・ ビー玉からくり', 0, 0.875, 0.06, M_TXT_G); fit_w(o, HWd * 1.85); vis(o, ttl_a)
plate(0.905, 0.09, ttl_a)
# reveal caption
rv_a = win(55.2, 59.6)
o = fit_w(hud_text('Wait... this lake is a PIANO?!', 0, 0.85, 0.08, M_TXT_W), HWd * 1.85); vis(o, rv_a)
o = hud_text('えっ…湖じゃなくて ピアノ！？', 0, 0.77, 0.065, M_TXT_G); fit_w(o, HWd * 1.8); vis(o, rv_a)
plate(0.815, 0.12, rv_a)
end_a = win(FINAL + 1.0, 999)
o = fit_w(hud_text('The lake was a grand piano', 0, 0.86, 0.075, M_TXT_W), HWd * 1.85); vis(o, end_a)
o = fit_w(hud_text('Moonlight Sonata – L. v. Beethoven', 0, 0.78, 0.06, M_TXT_G), HWd * 1.8); vis(o, end_a)
o = hud_text('湖の正体はグランドピアノでした', 0, 0.71, 0.062, M_TXT_W); fit_w(o, HWd * 1.8); vis(o, end_a)
o = fit_w(hud_text('Did you guess it?  気づいた？', 0, -0.86, 0.06, M_TXT_B), HWd * 1.8); vis(o, end_a)
plate(0.79, 0.2, end_a); plate(-0.86, 0.07, end_a)

PIECES_OUT = dict(pieces=PIECES)
if MODE == 'pieces': json.dump(PIECES_OUT, open(f'{HERE}/pieces.json', 'w'))
if MODE == 'test':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS)); sc.frame_set(f)
        sc.render.filepath = f'{HERE}/test_{f:04d}.png'
        t1 = time.time(); bpy.ops.render.render(write_still=True); print('frame', f, 'sec', round(time.time() - t1, 1))
elif MODE == 'range':
    sc.frame_start, sc.frame_end = int(args[1]), int(args[2])
    sc.render.image_settings.file_format = 'PNG'; sc.render.filepath = f'{HERE}/frames/f_'
    bpy.ops.render.render(animation=True)
if MODE == 'probe':
    for tt_ in args[1:]:
        f = int(round(float(tt_) * FPS))
        print('t', tt_, 'FOC', np.round(FOC[f], 2), 'WID', round(WID[f], 2), 'cam', np.round(CP[f], 2))
