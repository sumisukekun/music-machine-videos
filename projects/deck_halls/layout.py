"""Deck the Halls - 'Can 9 marbles decorate a Christmas tree?'  Physics + story planner.
Run after arrange.py; writes layout.json (read by scene.py, render_audio.py, verify.py).

Story (side view of a long candle-lit hall on Christmas Eve, the camera travels along +x):
  0-2 s     a dark hall, eight bare Christmas trees. A gift box on a shelf: its spring lid pops, a spring plunger pushes Holly
            (the red marble) out onto the first key.
  lines     Holly plays the verse lines (bars 1-2, 5-6, 9-10, 13-14 of each verse) on xylophone keys; every key she strikes
            lights a bulb of the string of lights along the hall.
  refrains  every 'Fa-la-la' (9 notes) is played by 9 glass ornaments: they hang on the garland rail above the next bare tree,
            drop one by one onto the tree's bells (the bells stand at the height of their pitch) and bounce into the tree,
            where each one lights up. Meanwhile Holly rides a machine to the next line:
              1 elevator   2 paternoster wheel   3 gated ramp   4 see-saw  -> chain reaction: its other end knocks the latch
              of the toy fireplace, the Yule-log ball rolls down the hearth chute and plays the bass in verse 2
              5 toy train   6 elevator   7 paternoster wheel   8 she rolls into the gift box under the last tree
  end       the Yule ball's last-but-one bass note lands on the cannon's trigger: the spring cannon fires the star marble up to
            the top of the last tree; it arrives at the top of its arc on the final chord, the star lights, every tree flashes.
Physics: one constant gravity G; between contacts pure parabolas; every bounce is an impulse along the surface normal (tangential
velocity kept), restitution emin <= e <= emax, so no bounce adds speed; rolling from rest on a ramp: a = (5/7) g sin(alpha); speed-ups
only from visible mechanisms (spring plunger, elevator, wheel, train, spring cannon)."""
import json, os, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = 7.0
RH, RO, RB, RS = 0.16, 0.1, 0.2, 0.12          # Holly, ornaments, Yule (bass) ball, star marble
Y_ORN, Y_BASS = 0.78, -0.88                     # depth planes: Holly y = 0, ornament bells behind, bass lane in front
EMAX, EMIN = 0.97, 0.2
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
TN = np.array([l[0] for l in LEAD]); NN = len(TN); FINAL = E['final']
FPS = 30


class Ball:
    def __init__(s, name, kind, r):
        s.name = name; s.kind = kind; s.r = r; s.seg = []; s.hits = []
    def rest(s, t0, t1, p): s.seg.append(dict(k='rest', t0=float(t0), t1=float(t1), p=list(map(float, p))))
    def fly(s, t0, t1, p0, v0): s.seg.append(dict(k='fly', t0=float(t0), t1=float(t1), p=list(map(float, p0)), v=list(map(float, v0))))
    def path(s, t0, t1, pts): s.seg.append(dict(k='path', t0=float(t0), t1=float(t1), pts=[list(map(float, q)) for q in pts]))
    def at(s, t): return seg_at(s.seg, t)


def seg_at(seg, t):
    for g in seg:
        if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9:
            if g['k'] == 'rest': return np.array(g['p'])
            if g['k'] == 'fly':
                u = t - g['t0']; return np.array(g['p']) + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
            P = np.array(g['pts']); u = (t - g['t0']) / max(1e-9, g['t1'] - g['t0']) * (len(P) - 1)
            i = min(len(P) - 2, int(u)); f = u - i; return P[i] * (1 - f) + P[i + 1] * f
    return np.array(seg_at(seg, seg[0]['t0'])) if t < seg[0]['t0'] else seg_at(seg, seg[-1]['t1'])


def vout_for(dx, dz, dt, dy=0.0): return np.array([dx / dt, dy / dt, dz / dt + 0.5 * G * dt])


def bounce_e(vin, vout):
    d = vout - vin; n = d / max(1e-9, np.linalg.norm(d)); a = vin @ n
    return (-(vout @ n) / a if a < -1e-9 else 9.0), n


def add_hit(ball, i, p, vin, vout, kind, t=None, **kw):
    e, n = bounce_e(np.array(vin, float), np.array(vout, float))
    if np.linalg.norm(vout) < 1e-9: n = np.array([0, 0, 1.0]); e = 0.0
    h = dict(i=int(i), t=float(TN[i] if t is None else t), p=list(map(float, p)), n=list(map(float, n)), e=float(e), kind=kind,
             vin=list(map(float, vin)), vout=list(map(float, vout)), pitch=int(kw.pop('pitch', LEAD[i][1] if i >= 0 else 0)))
    h.update(kw); ball.hits.append(h); return h


def run(ball, p, vin, ts, vx_of, dz_of, ids, kind='key', emax=EMAX, emin=EMIN, tilt_max=36.0, dvx=(-0.25, 0.15), pitches=None):
    """ball centre p (contact) at time ts[0] with incoming velocity vin; hops through ts[1:].
    vx_of(k) target horizontal speed of hop k, dz_of(k) target height change per second of flight.
    Landing height is only lowered (never raised) until the bounce is physical, looking one hop ahead."""
    for k in range(len(ts) - 1):
        dt = ts[k + 1] - ts[k]; dzt = dz_of(k) * dt
        if k + 1 < len(ts) - 1:
            dt2 = ts[k + 2] - ts[k + 1]
            need = (dz_of(k + 1) + 0.5 * G * dt2) / (emax * 0.985)
        else: need = 0.0
        tx = vx_of(k); cx = vin[0]
        vx0 = cx + float(np.clip(tx - cx, dvx[0], dvx[1]))
        vxs = sorted(set(np.round(np.concatenate([[vx0], vx0 + np.linspace(-0.3, 0.1, 21)]), 4)), key=lambda v: abs(v - vx0))
        dzs = sorted(np.concatenate([dzt - np.linspace(0, 0.5 * G * dt * dt + 1.0, 400), dzt + np.linspace(0.003, 0.2, 30)]),
                     key=lambda d: (0 if d <= dzt else 1, abs(d - dzt)))
        best = None; cands = []
        for vx in vxs[:6]:
            for dz in dzs:
                vo = vout_for(vx * dt, dz, dt); e, n = bounce_e(vin, vo)
                if not (emin <= e <= emax) or vo[2] - G * dt > -0.3: continue
                if math.degrees(math.acos(max(-1, min(1, n[2])))) > tilt_max: continue
                cands.append((dz, vo, e, n))
                if -(vo[2] - G * dt) >= need: best = (dz, vo, e, n); break
            if best is not None: break
        if best is None and cands: best = cands[0]; print('WARN need', ball.name, ids[k], round(need, 2))
        assert best is not None, ('no physical hop', ball.name, ids[k])
        dz, vo, e, n = best
        add_hit(ball, ids[k], p, vin, vo, kind, t=ts[k], **({'pitch': pitches[k]} if pitches else {}))
        ball.fly(ts[k], ts[k + 1], p, vo)
        p = p + vo * dt + np.array([0, 0, -0.5 * G * dt * dt]); vin = vo + np.array([0, 0, -G * dt])
    return p, vin


def smooth01(u): u = min(1.0, max(0.0, u)); return u * u * (3 - 2 * u)


def sample(fn, t0, t1, step=1 / 120):
    n = max(2, int(round((t1 - t0) / step)) + 1)
    return [fn(t0 + (t1 - t0) * k / (n - 1)) for k in range(n)]


B = {}; EV = {'dev': [], 'trees': [], 'orn': []}; SFX = []
H = B['holly'] = Ball('holly', 'holly', RH)
LINES = [[i for i in range(NN) if LEAD[i][6] == 'holly' and LEAD[i][8] == k] for k in range(8)]
REFR = [[i for i in range(NN) if LEAD[i][6] == 'orn' and LEAD[i][8] == k] for k in range(8)]
assert all(len(r) == 9 for r in REFR) and all(len(l) == 8 for l in LINES)

# ------------------------------------------------------------------ start: the gift box on the shelf, spring plunger
X0, Z0 = 0.0, 0.0
TF0 = 0.42; VX0 = 0.95
t1 = TN[LINES[0][0]]
p_box = np.array([X0 - VX0 * TF0, 0.0, Z0 + 0.5 * G * TF0 ** 2])
T_PUSH = t1 - TF0
H.rest(0.0, T_PUSH, p_box)
H.fly(T_PUSH, t1, p_box, [VX0, 0, 0])
EV['giftbox'] = dict(p=p_box.tolist(), t_push=T_PUSH, t_lid=T_PUSH - 0.75)
SFX.append(['lid', T_PUSH - 0.75]); SFX.append(['spring', T_PUSH])
p = np.array([X0, 0, Z0]); vin = np.array([VX0, 0, -G * TF0])

VX_LINE = 1.0
flatdz = lambda k: -0.03
DEVS = ['lift', 'wheel', 'ramp', 'seesaw', 'train', 'lift', 'wheel', 'gift']
HDROP = 0.75                                     # trapdoor drop onto the first key of the next line
for k in range(8):
    ids = LINES[k]; ts = [TN[i] for i in ids]
    def dzf(j, ts=ts):                           # after an eighth, the next longer hop goes down a step (energy for the bigger arc)
        if j > 0 and ts[j] - ts[j - 1] < 0.3 and ts[j + 1] - ts[j] > 0.3: return -0.7
        return -0.03
    p, vin = run(H, p, vin, ts, lambda j: VX_LINE, dzf, ids, dvx=(-0.6, 0.6))
    # last note of the line (a quarter): hop into the machine's cup, landing on the refrain's first beat
    il = ids[-1]; tl = TN[il]; tc = TN[REFR[k][0]]; dt = tc - tl
    dev = DEVS[k]; D = dict(kind=dev, k=k, t_catch=float(tc))
    best = None
    for dz in np.linspace(-0.12, -0.9, 80):
        vo = vout_for(VX_LINE * 0.8 * dt, dz, dt); e, n = bounce_e(vin, vo)
        if EMIN <= e <= EMAX and math.degrees(math.acos(n[2])) < 36: best = (dz, vo); break
    assert best, ('catch', k)
    dz, vo = best
    add_hit(H, il, p, vin, vo, 'key')
    H.fly(tl, tc, p, vo)
    pc = p + vo * dt + np.array([0, 0, -0.5 * G * dt * dt]); D['catch'] = pc.tolist(); D['vin'] = (vo + [0, 0, -G * dt]).tolist()
    SFX.append(['catch', float(tc)])
    if k == 7:                                   # the gift box under the last tree: she stays home
        H.rest(tc, FINAL + 10, pc); D['rest'] = pc.tolist(); EV['dev'].append(D); break
    tn = TN[LINES[k + 1][0]]                     # first note of the next line
    tdrop = math.sqrt(2 * HDROP / G)
    if dev == 'lift':
        DZ = 1.05; DXL = 0.75; tr = tn - tdrop; m0, m1 = tc + 0.35, tr - 0.45
        f = lambda t: pc + np.array([DXL, 0, DZ]) * smooth01((t - m0) / (m1 - m0))
        H.rest(tc, m0, pc); H.path(m0, m1, sample(f, m0, m1)); pr = f(m1); H.rest(m1, tr, pr)
        D.update(m0=m0, m1=m1, dz=DZ, dx=DXL, rel=pr.tolist(), t_rel=tr)
        SFX += [['motor', m0], ['clunk', m1], ['trap', tr]]
        H.fly(tr, tn, pr, [0, 0, 0]); p = pr + np.array([0, 0, -HDROP]); vin = np.array([0, 0, -G * tdrop])
    elif dev == 'wheel':
        RW_ = 0.62; HANG = 0.34                  # wheel radius, gondola hang (ball centre below the rim pivot)
        C = pc + np.array([RW_, 0, HANG])        # cup enters at angle 180 deg (left), leaves at 0 deg (right) over the top
        tr = tn - tdrop; m0, m1 = tc + 0.3, tr - 0.35
        def f(t):
            th = math.pi * (1 - smooth01((t - m0) / (m1 - m0)))
            return C + np.array([RW_ * math.cos(th), 0, RW_ * math.sin(th) - HANG])
        H.rest(tc, m0, pc); H.path(m0, m1, sample(f, m0, m1)); pr = f(m1); H.rest(m1, tr, pr)
        D.update(m0=m0, m1=m1, centre=C.tolist(), r=RW_, hang=HANG, rel=pr.tolist(), t_rel=tr)
        SFX += [['ratchet', m0], ['clunk', m1], ['trap', tr]]
        H.fly(tr, tn, pr, [0, 0, 0]); p = pr + np.array([0, 0, -HDROP]); vin = np.array([0, 0, -G * tdrop])
    elif dev == 'ramp':
        # cup = top of ramp 1 (gate A); ramp 1 down-forward to gate B; ramp 2 to the edge, flight onto the first key
        al = math.radians(11.0); a = 5.0 / 7.0 * G * math.sin(al); L1, L2 = 0.95, 0.85; TFL = 0.36
        u_ = np.array([math.cos(al), 0, -math.sin(al)])
        t_r1 = math.sqrt(2 * L1 / a); t_r2 = math.sqrt(2 * L2 / a); v2 = a * t_r2
        tB = tn - TFL - t_r2; tA = tB - 0.42 - t_r1
        assert tA > tc + 0.35, ('ramp timing', tA - tc)
        pA = pc; pB = pA + u_ * L1; pE = pB + u_ * L2
        H.rest(tc, tA, pA)
        H.path(tA, tA + t_r1, sample(lambda t: pA + u_ * 0.5 * a * (t - tA) ** 2, tA, tA + t_r1))
        H.rest(tA + t_r1, tB, pB)
        H.path(tB, tB + t_r2, sample(lambda t: pB + u_ * 0.5 * a * (t - tB) ** 2, tB, tB + t_r2))
        ve = u_ * v2; H.fly(tB + t_r2, tn, pE, ve)
        p = pE + ve * TFL + np.array([0, 0, -0.5 * G * TFL ** 2]); vin = ve + np.array([0, 0, -G * TFL])
        D.update(alpha=math.degrees(al), A=pA.tolist(), Bp=pB.tolist(), end=pE.tolist(), tA=tA, tB=tB, t_stopB=tA + t_r1,
                 t_edge=tB + t_r2, v_edge=float(v2), a=a)
        SFX += [['gate', tA], ['knock', tA + t_r1], ['gate', tB]]
    elif dev == 'seesaw':
        # Holly lands in the cup on the raised right end; her weight tips the plank; the left end flies up and knocks the latch
        ARM = 0.7; TH0 = math.radians(16.0); TTIP = 0.62
        piv = pc - np.array([ARM * math.cos(TH0), 0, ARM * math.sin(TH0) + RH + 0.05])
        def ang(t): return TH0 - 2 * TH0 * smooth01((t - tc - 0.06) / TTIP)
        def f(t):
            a_ = ang(t); return piv + np.array([ARM * math.cos(a_), 0, ARM * math.sin(a_) + RH + 0.05])
        tr = tn - tdrop; m1 = tc + 0.06 + TTIP
        H.path(tc, m1, sample(f, tc, m1)); pr = f(m1); H.rest(m1, tr, pr)
        D.update(pivot=piv.tolist(), arm=ARM, th0=math.degrees(TH0), t_tip=tc + 0.06, t_tipped=m1, rel=pr.tolist(), t_rel=tr)
        SFX += [['seesaw', m1], ['trap', tr]]
        H.fly(tr, tn, pr, [0, 0, 0]); p = pr + np.array([0, 0, -HDROP]); vin = np.array([0, 0, -G * tdrop])
    elif dev == 'train':
        DX = 2.3; tr = tn - tdrop; m0, m1 = tc + 0.35, tr - 0.35
        f = lambda t: pc + np.array([DX * smooth01((t - m0) / (m1 - m0)), 0, 0])
        H.rest(tc, m0, pc); H.path(m0, m1, sample(f, m0, m1)); pr = f(m1); H.rest(m1, tr, pr)
        D.update(m0=m0, m1=m1, dx=DX, rel=pr.tolist(), t_rel=tr)
        SFX += [['whistle', m0 - 0.1], ['chug', m0], ['clunk', m1], ['trap', tr]]
        H.fly(tr, tn, pr, [0, 0, 0]); p = pr + np.array([0, 0, -HDROP]); vin = np.array([0, 0, -G * tdrop])
    EV['dev'].append(D)


def holly_z_band(t0, t1):
    zs = [H.at(t)[2] for t in np.arange(t0, t1, 0.05)]; return min(zs), max(zs)


def holly_x_band(t0, t1):
    xs = [H.at(t)[0] for t in np.arange(t0, t1, 0.05)]; return min(xs), max(xs)


# ------------------------------------------------------------------ trees + ornaments (one tree per refrain)
RBASE, HT, YT = 1.7, 5.4, 2.6                 # cone base radius, height, centre depth
def tree_front_y(tr, x, z):
    r = RBASE * tr['s'] * (1 - (z - tr['zb']) / (HT * tr['s']))
    dx = x - tr['x']
    if r <= abs(dx) + 0.02: return None
    return YT - math.sqrt(r * r - dx * dx)


O_I = 0
for k in range(8):
    ids = REFR[k]; ts = [TN[i] for i in ids]; ps = [LEAD[i][1] for i in ids]
    D = EV['dev'][k]
    t0 = TN[ids[0]] - 0.5; t1_ = TN[ids[-1]] + 0.6
    zlo, zhi = holly_z_band(t0, t1_); xlo, xhi = holly_x_band(t0, t1_)
    s = 1.0 if k < 7 else 1.25                   # the last tree is a little bigger (the star goes on top)
    xT = (xlo + xhi) / 2 + (0.45 if D['kind'] in ('lift', 'wheel', 'seesaw', 'train') else 0.0)
    if k == 7: xT = D['catch'][0] + 1.25
    zb = zlo - 1.15
    tr = dict(k=k, x=float(xT), zb=float(zb), s=s, top=float(zb + HT * s))
    EV['trees'].append(tr)
    zbell0 = max(zhi + 0.85, zb + 1.9)

    pmin = min(ps)
    bells = []
    for j, (i, t, pt) in enumerate(zip(ids, ts, ps)):
        xb = xT - 0.95 * s + 1.9 * s * j / 8
        zc = zbell0 + 0.1 * (pt - pmin)          # bell height follows the pitch
        bells.append((xb, zc))
    zrail = max(b[1] for b in bells) + 0.75
    tr['rail'] = float(zrail); tr['bells'] = []
    for j, (i, t, pt) in enumerate(zip(ids, ts, ps)):
        xb, zc = bells[j]
        o = B[f'orn{O_I}'] = Ball(f'orn{O_I}', 'orn', RO); O_I += 1
        h = zrail - RO - 0.06 - zc; tfall = math.sqrt(2 * h / G)
        p0 = np.array([xb, Y_ORN, zc + h]); pc_ = np.array([xb, Y_ORN, zc]); vin = np.array([0, 0, -G * tfall])
        o.rest(0.0, t - tfall, p0); o.fly(t - tfall, t, p0, [0, 0, 0])
        # bounce back into the tree: cup on the branch, below and a bit outwards
        best = None
        cups = [c for c in tr.setdefault('_cups', [])]
        tree_orns = [B[b_['ball']] for b_ in tr['bells']]
        for zcup in np.linspace(zc - 0.25, zc - 1.1, 18):
          for fx in (0.6, 0.45, 0.75, 0.3):
            xc = xT + fx * (xb - xT)
            yf = tree_front_y(tr, xc, zcup)
            if yf is None: continue
            tgt = np.array([xc, yf + 0.04, zcup])
            if any(np.linalg.norm(tgt - c) < 2 * RO + 0.07 for c in cups): continue
            for tf in np.linspace(0.3, 0.75, 46):
                vo = vout_for(tgt[0] - xb, tgt[2] - zc, tf, tgt[1] - Y_ORN); e, n = bounce_e(vin, vo)
                if not (0.18 <= e <= 0.85) or vo[2] - G * tf > -0.6: continue
                if math.degrees(math.acos(n[2])) > 40: continue
                sc_ = abs(e - 0.45) + 0.3 * abs(tf - 0.45)
                if best is not None and sc_ >= best[0]: continue
                ok = True
                for u in np.linspace(0, tf + 0.3, 24):          # clear of this tree's other ornaments (flying or resting)
                    q = pc_ + vo * min(u, tf) + np.array([0, 0, -0.5 * G * min(u, tf) ** 2])
                    for ob in tree_orns:
                        if np.linalg.norm(q - ob.at(t + u)) < 2 * RO + 0.03: ok = False; break
                    if not ok: break
                if ok: best = (sc_, tgt, tf, vo)
            if best is not None: break
          if best is not None: break
        if best is not None: tr['_cups'].append(best[1])
        assert best is not None, ('ornament', k, j)
        _, tgt, tf, vo = best
        add_hit(o, i, pc_, vin, vo, 'bell', t=t)
        o.fly(t, t + tf, pc_, vo); o.rest(t + tf, FINAL + 10, tgt)
        o.hits[-1]['cup'] = tgt.tolist(); o.hits[-1]['t_cup'] = float(t + tf); o.hits[-1]['tree'] = k
        tr['bells'].append(dict(i=int(i), x=float(xb), z=float(zc), ball=o.name))
        EV['orn'].append(dict(ball=o.name, tree=k, t_rel=float(t - tfall), t_hit=float(t), t_cup=float(t + tf), cup=tgt.tolist()))
        SFX.append(['plink', float(t + tf)])

# ------------------------------------------------------------------ the Yule-log ball: hearth chute, then the bass in verse 2
BASS = E['bass']; BT = [b[0] for b in BASS]
Y = B['yule'] = Ball('yule', 'bass', RB)
D3 = EV['dev'][3]
t_latch = D3['t_tipped']
# the bass lane runs under the course: target = running minimum of Holly's height minus 1.45
_XA, _XB = float(H.at(BT[0])[0]) - 0.3, EV['trees'][7]['x'] + 1.75
_ZA = min(H.at(u)[2] for u in np.arange(BT[0] - 6, BT[0], 0.1)) - 1.15
def lane_target(t):
    """straight, evenly paced lane: from under Holly at bar 17 to the cannon by the last tree; 0.11 down per second"""
    u = (t - BT[0]) / (BT[-2] - BT[0])
    return _XA + (_XB - _XA) * u, _ZA - 0.11 * (t - BT[0])
x1, z1 = lane_target(BT[0])
# chute: rolling from rest (alpha) from the latch time, leaves the chute end and drops onto the first bass bar
alc = math.radians(3.2); ac = 5.0 / 7.0 * G * math.sin(alc); TFB = 0.42
t_roll = BT[0] - TFB - (t_latch + 0.05)
Lc = 0.5 * ac * t_roll ** 2; vc = ac * t_roll
uc = np.array([math.cos(alc), 0, -math.sin(alc)])
pend = np.array([x1, Y_BASS, z1]) - (uc * vc * TFB + np.array([0, 0, -0.5 * G * TFB ** 2]))
pstart = pend - uc * Lc
t_r0 = t_latch + 0.05
Y.rest(0.0, t_r0, pstart)
Y.path(t_r0, t_r0 + t_roll, sample(lambda t: pstart + uc * 0.5 * ac * (t - t_r0) ** 2, t_r0, t_r0 + t_roll, 1 / 60))
ve = uc * vc; Y.fly(t_r0 + t_roll, BT[0], pend, ve)
EV['yule'] = dict(start=pstart.tolist(), end=pend.tolist(), t_release=t_r0, t_edge=t_r0 + t_roll, alpha=math.degrees(alc), v_edge=vc)
SFX += [['latch', t_latch], ['roll', t_r0]]
p = np.array([x1, Y_BASS, z1]); vin = ve + np.array([0, 0, -G * TFB])
# bass hops: last-but-one note lands on the cannon trigger
NB = len(BT)
def bvx(j):
    xt, _ = lane_target(BT[j + 1]); return (xt - Y_pos[0]) / (BT[j + 1] - BT[j])
Y_pos = p.copy()
pp, vv = p, vin
for j in range(NB - 1):
    Y_pos = pp
    xt, zt = lane_target(BT[j + 1]); dtj = BT[j + 1] - BT[j]
    vx_t = float(np.clip((xt - pp[0]) / dtj, 0.3, 1.2))
    dz_t = min(-0.055, (zt - pp[2])) / dtj
    pp, vv = run(Y, pp, vv, [BT[j], BT[j + 1]], lambda q: vx_t, lambda q: dz_t, [-1, -1], kind='bass', pitches=[BASS[j][1], BASS[j + 1][1]],
                 dvx=(-0.12, 0.12))
    Y.hits[-1]['bass_j'] = j
# the final bass note: dead landing on the big bass bar (the ball stays on it)
add_hit(Y, -1, pp, vv, [0, 0, 0], 'bassfinal', t=BT[-1], pitch=BASS[-1][1], bass_j=NB - 1)
Y.rest(BT[-1], FINAL + 10, pp)
trig = [h for h in Y.hits if h.get('bass_j') == NB - 2][0]; trig['kind'] = 'trigger'
EV['trigger'] = dict(t=trig['t'], p=trig['p'])

# ------------------------------------------------------------------ the star marble: spring cannon -> apex at the tree top on the final chord
S = B['star'] = Ball('star', 'star', RS)
tr7 = EV['trees'][7]
top = np.array([tr7['x'], YT, tr7['top'] + 0.12])
TFS = FINAL - trig['t'] - 0.04
t_fire = FINAL - TFS
vz = G * TFS; hS = 0.5 * G * TFS ** 2
p0 = np.array([trig['p'][0] + 0.05, 0.3, top[2] - hS])        # spring cannon right behind the trigger bar
v0 = np.array([(top[0] - p0[0]) / TFS, (top[1] - p0[1]) / TFS, vz])
S.rest(0.0, t_fire, p0); S.fly(t_fire, FINAL, p0, v0)
add_hit(S, -1, top, v0 + np.array([0, 0, -G * TFS]), [0, 0, 0], 'star', t=FINAL, pitch=89)
S.rest(FINAL, FINAL + 10, top)
EV['cannon'] = dict(p=p0.tolist(), t_fire=t_fire, v=v0.tolist(), top=top.tolist())
SFX += [['cannon', t_fire], ['star', FINAL], ['giftpop', FINAL]]


def check():
    info = {}
    hs = [(n, h) for n, b in B.items() for h in b.hits if np.linalg.norm(h['vout']) > 0]
    info['e_range'] = (round(min(h['e'] for n, h in hs), 3), round(max(h['e'] for n, h in hs), 3))
    info['speed_out_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for n, h in hs), 3)
    info['tilt_max'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for n, h in hs), 1)
    cov = set(h['i'] for b in B.values() for h in b.hits if h['i'] >= 0)
    info['melody_uncovered'] = sorted(set(range(NN)) - cov)
    info['bass_hits'] = sum(1 for h in Y.hits)
    info['holly_z'] = (round(min(h['p'][2] for h in H.hits), 2), round(max(h['p'][2] for h in H.hits), 2))
    info['holly_x'] = (round(H.hits[0]['p'][0], 2), round(H.hits[-1]['p'][0], 2))
    info['yule_z'] = (round(min(h['p'][2] for h in Y.hits), 2), round(max(h['p'][2] for h in Y.hits), 2))
    info['trees'] = [(round(t['x'], 2), round(t['zb'], 2)) for t in EV['trees']]
    info['cannon_v'] = round(float(np.linalg.norm(EV['cannon']['v'])), 2)
    return info


for tr_ in EV['trees']: tr_.pop('_cups', None)
if __name__ == '__main__':
    for k_, v_ in check().items(): print(k_, v_)
    out = dict(G=G, R={'holly': RH, 'orn': RO, 'bass': RB, 'star': RS}, y_orn=Y_ORN, y_bass=Y_BASS, tree=dict(rbase=RBASE, ht=HT, yt=YT),
               balls={n: dict(kind=b.kind, r=b.r, seg=b.seg, hits=b.hits) for n, b in B.items()}, ev=EV, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
