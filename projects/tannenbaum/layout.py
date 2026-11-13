"""O Tannenbaum - 'A fallen star grows a giant Christmas tree'.  Physics + story planner.
Run after arrange.py; writes layout.json (read by scene.py, render_audio.py, verify.py).

Story (side view of a snowy forest at night, the camera rises with the tree):
  0-2.6 s   a shooting star (Stella, a golden marble) falls out of the night sky into a snow cup next to a tiny fir sapling;
            the cup's gate opens, she rolls down an ice chute onto the first ice key.
  phrases   9 phrases (3 verses x A-B-A). Half 1 of each phrase: Stella plays it on a row of ice keys in front of the tree; every
            key she strikes makes the tree's newest tier grow a step. Half 2: glass ornaments drop from the garland rail onto the
            bells of that tier (bell height = pitch) and bounce into the branches, where they light up. Meanwhile Stella rides a
            machine up to the next row (the rows zig-zag up the tree):
              0 elevator  1 chairlift  2 snowball catapult  3 hot-air balloon  4 elevator  5 snowball catapult  6 chairlift
              7 hot-air balloon  8 the star lift: it carries her up beside the top, the star cannon fires her onto the top.
  bass      verses 2 and 3: pinecones drop from a branch rail onto log drums under Stella's row (one pinecone per bass note).
  end       the last-but-one bass pinecone lands on the trigger drum: the cannon fires, Stella reaches the top of her arc on the
            final chord and lands on the tree top - the fallen star is the star of the giant tree.
Physics: one constant gravity G; between contacts pure parabolas; every bounce is an impulse along the surface normal (tangential
velocity kept), restitution emin <= e <= emax, so no bounce adds speed; rolling from rest on a ramp: a = (5/7) g sin(alpha); speed-ups
only from visible mechanisms (lifts, chairlift, balloon, the snowball hitting the catapult, the spring cannon)."""
import json, os, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = 7.0
RS, RO, RP, RB = 0.16, 0.1, 0.13, 0.24              # Stella, ornaments, pinecones, catapult snowball
Y_PINE = -0.62                                      # pinecone plane (in front of Stella's keys)
EMAX, EMIN = 0.97, 0.2
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
TN = np.array([l[0] for l in LEAD]); NN = len(TN); FINAL = E['final']
FPS = 30
STEP = 1.8                                          # height between rows
KEY_HW = 0.06
ROLL_MIN, TF_ROLL = 0.8, 0.3                        # long notes (dotted quarter) are rolled along a long key
YT = 3.4                                            # tree axis depth
XSIDE = 2.45                                        # machine column / chute top (|x|)
XKEY0 = 1.6                                         # |x| of the first key of a row
TIER_H = 2.9


def tier_R(k): return 2.5 - 0.15 * k


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


def bounce_ok_grid(vin, VO, emin=EMIN, emax=EMAX, tilt_max=36.0):
    """vectorised: which outgoing velocities VO (N x 3) are physical bounces for incoming vin"""
    D = VO - vin[None]; nrm = np.linalg.norm(D, axis=1); nrm[nrm < 1e-9] = 1e-9; Nn = D / nrm[:, None]
    a = Nn @ vin; e = np.where(a < -1e-9, -(np.sum(VO * Nn, 1)) / np.where(a < -1e-9, a, -1), 9.0)
    return (e >= emin) & (e <= emax) & (Nn[:, 2] >= math.cos(math.radians(tilt_max)))


def feasible(va, dt2, dzt2, tx2, slack):
    """is there a physical next hop (duration dt2) landing no lower than dzt2 - slack, moving the right way?"""
    vxs = np.linspace(tx2 - 1.0, tx2 + 1.0, 21); vxs = vxs[vxs * tx2 > 0]
    dzs = dzt2 - np.linspace(0, slack, 12)
    VX, DZ = np.meshgrid(vxs, dzs); VX = VX.ravel(); DZ = DZ.ravel()
    VO = np.stack([VX, 0 * VX, DZ / dt2 + 0.5 * G * dt2], 1)
    return bool((bounce_ok_grid(va, VO) & (VO[:, 2] - G * dt2 < -0.3)).any())


def run(ball, p, vin, ts, vx_of, dz_of, ids, kind='key', emax=EMAX, emin=EMIN, tilt_max=36.0, pitches=None, beam=60, **_):
    """ball centre p (contact) at time ts[0] with incoming velocity vin; hops through ts[1:] (each hop = one note interval).
    Beam search over the whole row: every hop is a physical bounce (restitution emin..emax, normal tilt <= tilt_max), the row
    loses as little height as possible and keeps its horizontal speed near vx_of(k)."""
    states = [(0.0, np.array(p, float), np.array(vin, float), [])]
    for k in range(len(ts) - 1):
        dt = ts[k + 1] - ts[k]; dzt = dz_of(k) * dt; tx = vx_of(k)
        nxt = []
        if dt >= ROLL_MIN:                       # long note: land dead on a long key, roll along it, fall off its end
            for cost, p_, v_, hist in states:
                vx = v_[0]; tr_ = dt - TF_ROLL; hdrop = 0.5 * G * TF_ROLL ** 2
                pn = p_ + np.array([vx * dt, 0, -hdrop]); vn = np.array([vx, 0, -G * TF_ROLL])
                nxt.append((cost + hdrop * 3.0 + abs(vx - tx) * 3.0 * dt, pn, vn, hist + [('roll', vx)]))
            states = nxt; continue
        for cost, p_, v_, hist in states:
            VXs = np.linspace(tx - 1.2, tx + 1.2, 41); VXs = VXs[(np.abs(VXs) <= 2.4) & (VXs * tx > 0) & (np.abs(VXs) * dt >= 0.1)]
            DZs = dzt - np.linspace(0, 0.5 * G * dt * dt + 1.6, 160)
            VX, DZ = np.meshgrid(VXs, DZs); VX = VX.ravel(); DZ = DZ.ravel()
            VO = np.stack([VX, 0 * VX, DZ / dt + 0.5 * G * dt], 1)
            ok = bounce_ok_grid(v_, VO, emin, emax, tilt_max) & (VO[:, 2] - G * dt < -0.3)
            # clearance: the ball must not clip the key it just left (half width KEY_HW) on its way down to the next one
            for uf in (0.55, 0.7, 0.85, 1.0):
                tt_ = uf * dt; bx = VO[:, 0] * tt_; bz = VO[:, 2] * tt_ - 0.5 * G * tt_ * tt_
                ex = np.maximum(0, np.abs(bx) - KEY_HW); ez = np.maximum(0, np.maximum((-ball.r - 0.06) - bz, bz + ball.r))
                ok &= np.hypot(ex, ez) >= ball.r + 0.012
            if not ok.any(): continue
            idx = np.nonzero(ok)[0]
            c = cost + (dzt - DZ[idx]) * 3.0 + np.abs(VX[idx] - tx) * 3.0 * dt
            # keep, per horizontal speed, the few least-descent options (diversity in arrival speed)
            sel = []
            for vx in np.unique(VX[idx]):
                m = idx[VX[idx] == vx]; cm = c[VX[idx] == vx]; o = np.argsort(cm)[:6:2]
                sel += [(cm[q], m[q]) for q in o]
            for cc, j in sel:
                vo = VO[j]; pn = p_ + vo * dt + np.array([0, 0, -0.5 * G * dt * dt]); vn = vo + np.array([0, 0, -G * dt])
                nxt.append((cc, pn, vn, hist + [vo]))
        assert nxt, ('no physical hop', ball.name, ids[k])
        # prune: rank by cost minus the kinetic energy in hand (as height), keep the best `beam`
        nxt.sort(key=lambda s_: s_[0] - 3.0 * 0.5 * (s_[2] @ s_[2]) / G * 0.6)
        states = nxt[:beam]
    states.sort(key=lambda s_: s_[0])
    cost, pe, ve, hist = states[0]
    for k, vo in enumerate(hist):
        dt = ts[k + 1] - ts[k]
        if isinstance(vo, tuple):
            vx = vo[1]; tr_ = dt - TF_ROLL
            add_hit(ball, ids[k], p, vin, [vx, 0, 0], 'rollkey', t=ts[k], roll=float(vx * tr_), t_roll=float(tr_),
                    **({'pitch': pitches[k]} if pitches else {}))
            ball.path(ts[k], ts[k] + tr_, [p, p + np.array([vx * tr_, 0, 0])])
            pe_ = p + np.array([vx * tr_, 0, 0]); ball.fly(ts[k] + tr_, ts[k + 1], pe_, [vx, 0, 0])
            p = pe_ + np.array([vx * TF_ROLL, 0, -0.5 * G * TF_ROLL ** 2]); vin = np.array([vx, 0, -G * TF_ROLL])
            continue
        add_hit(ball, ids[k], p, vin, vo, kind, t=ts[k], **({'pitch': pitches[k]} if pitches else {}))
        ball.fly(ts[k], ts[k + 1], p, vo)
        p = p + vo * dt + np.array([0, 0, -0.5 * G * dt * dt]); vin = vo + np.array([0, 0, -G * dt])
    return p, vin


def smooth01(u): u = min(1.0, max(0.0, u)); return u * u * (3 - 2 * u)


def sample(fn, t0, t1, step=1 / 120):
    n = max(2, int(round((t1 - t0) / step)) + 1)
    return [fn(t0 + (t1 - t0) * k / (n - 1)) for k in range(n)]


B = {}; EV = {'dev': [], 'tiers': [], 'orn': [], 'pine': []}; SFX = []
S = B['stella'] = Ball('stella', 'stella', RS)
LINES = [[i for i in range(NN) if LEAD[i][7] == 'star' and LEAD[i][8] == k] for k in range(9)]
REFR = [[i for i in range(NN) if LEAD[i][7] in ('orn', 'top') and LEAD[i][8] == k] for k in range(9)]
assert all(len(l) == 9 for l in LINES)

# ------------------------------------------------------------------ chute: rolling from rest, then a short flight onto a key
AL_CH = math.radians(16.0); L_CH = 0.36; TF_CH = 0.28
A_CH = 5.0 / 7.0 * G * math.sin(AL_CH); T_CH = math.sqrt(2 * L_CH / A_CH); V_CH = A_CH * T_CH


def chute_from_key(p1, d):
    """the chute top (ball centre) for a chute that delivers onto key position p1 travelling in direction d (+1/-1)"""
    u = np.array([d * math.cos(AL_CH), 0, -math.sin(AL_CH)])
    ve = u * V_CH
    pe = p1 - (ve * TF_CH + np.array([0, 0, -0.5 * G * TF_CH ** 2]))
    return pe - u * L_CH, u


def ride_chute(ball, ptop, u, t_gate, tn):
    """gate opens at t_gate; roll down; leave the chute end; land at tn (= t_gate + T_CH + TF_CH)"""
    ball.path(t_gate, t_gate + T_CH, sample(lambda t: ptop + u * 0.5 * A_CH * (t - t_gate) ** 2, t_gate, t_gate + T_CH))
    pe = ptop + u * L_CH; ve = u * V_CH
    ball.fly(t_gate + T_CH, tn, pe, ve)
    return pe + ve * TF_CH + np.array([0, 0, -0.5 * G * TF_CH ** 2]), ve + np.array([0, 0, -G * TF_CH])


# ------------------------------------------------------------------ intro: the shooting star
t1 = TN[LINES[0][0]]
p1 = np.array([-XKEY0, 0.0, 0.0])
ptop, u = chute_from_key(p1, +1)
t_gate = t1 - T_CH - TF_CH
T_LAND = 1.25
p_sky = ptop + np.array([-2.0, 0.0, 6.0])
v_sky = vout_for(*(ptop - p_sky)[[0, 2]], T_LAND)
S.fly(0.0, T_LAND, p_sky, v_sky)
vland = v_sky + np.array([0, 0, -G * T_LAND])
add_hit(S, -1, ptop, vland, [0, 0, 0], 'land', t=T_LAND)
S.rest(T_LAND, t_gate, ptop)
p, vin = ride_chute(S, ptop, u, t_gate, t1)
assert np.allclose(p, p1, atol=1e-6)
EV['intro'] = dict(sky=p_sky.tolist(), v=v_sky.tolist(), t_land=T_LAND, cup=ptop.tolist(), t_gate=t_gate, u=u.tolist())
SFX += [['whoosh', 0.1], ['snowland', T_LAND], ['gate', t_gate]]
EV['chutes'] = [dict(top=ptop.tolist(), u=u.tolist(), t_gate=t_gate, k=-1)]

DEVS = ['lift', 'chair', 'catapult', 'balloon', 'lift', 'catapult', 'chair', 'balloon', 'final']
VX_LINE = 0.65
ZROW = [0.0]
for k in range(9):
    d = 1 if k % 2 == 0 else -1
    ids = LINES[k]; ts = [TN[i] for i in ids]
    def dzf(j, ts=ts): return -0.03
    p, vin = run(S, p, vin, ts, lambda j, d=d: VX_LINE * d, dzf, ids)
    # last note of the row: hop into the machine's cup, landing on the first note of the half 2
    il = ids[-1]; tl = TN[il]; tc = TN[REFR[k][0]]; dt = tc - tl
    dev = DEVS[k]; D = dict(kind=dev, k=k, d=d, t_catch=float(tc))
    best = None
    for dz in np.linspace(-0.15, -1.6, 150):
        for dxc in np.linspace(0.5, 0.15, 15):
            vo = vout_for(d * dxc, dz, dt); e, n = bounce_e(vin, vo)
            if EMIN <= e <= EMAX and math.degrees(math.acos(n[2])) < 36: best = (dz, vo); break
        if best: break
    assert best, ('catch', k)
    dz, vo = best
    add_hit(S, il, p, vin, vo, 'key')
    S.fly(tl, tc, p, vo)
    pc = p + vo * dt + np.array([0, 0, -0.5 * G * dt * dt]); D['catch'] = pc.tolist(); D['vin'] = (vo + [0, 0, -G * dt]).tolist()
    SFX.append(['catch', float(tc)])
    if dev == 'final':
        EV['dev'].append(D); break
    tn = TN[LINES[k + 1][0]]
    zn = ZROW[0] + STEP * (k + 1)
    p1 = np.array([d * (XKEY0 - (0.3 if dev == 'catapult' else 0.0)), 0.0, zn]); ZROW.append(zn)
    if dev == 'catapult':
        TFC = 1.3; t_l = tn - TFC
        v0 = vout_for(p1[0] - pc[0], p1[2] - pc[2], TFC)
        S.rest(tc, t_l, pc); S.fly(t_l, tn, pc, v0)
        # the see-saw runs in depth (y): Stella's cup end at y = 0 (down), the far end behind her (up), towards the tree.
        # A lump of snow (the snowball) sits on a sloped branch of the tree above the far end; Stella's landing trips its latch
        # (a cord from the cup); it rolls off the branch towards the camera, falls onto the far end and flips the plank.
        ARM = 0.6; TH = math.radians(22.0)
        piv = pc + np.array([0, ARM * math.cos(TH), ARM * math.sin(TH) - RS - 0.05])
        far_end = piv + np.array([0, ARM * math.cos(TH), ARM * math.sin(TH) + RB + 0.05])
        hdrop = 1.05; tfall = math.sqrt(2 * hdrop / G)
        alb = math.radians(9.0); ab = 5.0 / 7.0 * G * math.sin(alb)
        t_rel = tc + 0.12; t_rollb = t_l - tfall - t_rel
        assert t_rollb > 0.3, t_rollb
        Lb = 0.5 * ab * t_rollb ** 2; vb = ab * t_rollb
        ub = np.array([0, -math.cos(alb), -math.sin(alb)])
        vbe = ub * vb
        p_end = far_end - (vbe * tfall + np.array([0, 0, -0.5 * G * tfall ** 2]))
        p_start = p_end - ub * Lb
        sb = B[f'snow{k}'] = Ball(f'snow{k}', 'snowball', RB)
        sb.rest(0.0, t_rel, p_start)
        sb.path(t_rel, t_rel + t_rollb, sample(lambda t: p_start + ub * 0.5 * ab * (t - t_rel) ** 2, t_rel, t_rel + t_rollb))
        sb.fly(t_rel + t_rollb, t_l, p_end, vbe)
        vimp = vbe + np.array([0, 0, -G * tfall])
        add_hit(sb, -1, far_end, vimp, [0, 0, 0], 'slam', t=t_l)
        # the far end goes down by 2*ARM*sin(TH) right after the slam; the snowball rides it down (path), then rests
        def sbf(t): return far_end - np.array([0, 0, 2 * ARM * math.sin(TH) * smooth01((t - t_l) / 0.12)])
        sb.path(t_l, t_l + 0.12, sample(sbf, t_l, t_l + 0.12)); sb.rest(t_l + 0.12, FINAL + 10, sbf(t_l + 0.12))
        D.update(pivot=piv.tolist(), arm=ARM, th=math.degrees(TH), t_launch=t_l, v0=v0.tolist(), far=far_end.tolist(),
                 snow_start=p_start.tolist(), snow_end=p_end.tolist(), t_snow=t_rel, t_snow_edge=t_rel + t_rollb, alb=math.degrees(alb))
        SFX += [['latch', tc + 0.1], ['roll', t_rel], ['slam', t_l]]
        p = p1; vin = v0 + np.array([0, 0, -G * TFC])
        EV['dev'].append(D); continue
    ptop, u = chute_from_key(p1, -d)
    t_gate = tn - T_CH - TF_CH
    m0, m1 = tc + 0.28, t_gate - 0.22
    if dev == 'lift':
        f = lambda t, pc=pc, ptop=ptop, m0=m0, m1=m1: np.array([pc[0] + (ptop[0] - pc[0]) * smooth01((t - m0) / (m1 - m0) * 1.6),
                                                                  0.0, pc[2] + (ptop[2] - pc[2]) * smooth01((t - m0) / (m1 - m0))])
        SFX += [['motor', m0], ['clunk', m1]]
    elif dev == 'chair':
        f = lambda t, pc=pc, ptop=ptop, m0=m0, m1=m1: pc + (ptop - pc) * smooth01((t - m0) / (m1 - m0))
        SFX += [['ratchet', m0], ['clunk', m1]]
    elif dev == 'balloon':
        def f(t, pc=pc, ptop=ptop, m0=m0, m1=m1, d=d):
            u_ = smooth01((t - m0) / (m1 - m0)); s_ = math.sin(math.pi * u_)
            return pc + (ptop - pc) * u_ + np.array([d * 0.35 * s_, 0, 0])
        SFX += [['burner', m0 - 0.1], ['burner', m0 + 0.7], ['clunk', m1]]
    S.rest(tc, m0, pc); S.path(m0, m1, sample(f, m0, m1)); S.rest(m1, t_gate, ptop)
    assert np.allclose(f(m1), ptop, atol=1e-6)
    p, vin = ride_chute(S, ptop, u, t_gate, tn)
    assert np.allclose(p, p1, atol=1e-6)
    D.update(m0=m0, m1=m1, top=ptop.tolist(), t_gate=t_gate)
    EV['chutes'].append(dict(top=ptop.tolist(), u=u.tolist(), t_gate=t_gate, k=k))
    SFX.append(['gate', t_gate])
    EV['dev'].append(D)


def s_band(t0, t1, ax):
    v = [S.at(t)[ax] for t in np.arange(t0, t1, 0.05)]; return min(v), max(v)


# ------------------------------------------------------------------ tree tiers + ornaments (one tier per phrase)
def tier_front_y(tr, x, z):
    if not (tr['zb'] <= z <= tr['zb'] + TIER_H): return None
    r = tr['R'] * (1 - 0.65 * (z - tr['zb']) / TIER_H)
    if r <= abs(x) + 0.02: return None
    return YT - math.sqrt(r * r - x * x)


O_I = 0
for k in range(9):
    ids = REFR[k]
    ids_b = [i for i in ids if LEAD[i][7] == 'orn']
    hz = [h['p'][2] for h in S.hits if h['kind'] == 'key' and LEAD[h['i']][8] == k]
    zlo, zhi = min(hz), max(hz)
    tr = dict(k=k, zb=float(zlo - 0.55), R=float(tier_R(k)), t_grow=[float(TN[i]) for i in LINES[k]])
    tr['top'] = tr['zb'] + TIER_H
    EV['tiers'].append(tr)
    zbell0 = zhi + 1.0
    ps = [LEAD[i][1] for i in ids_b]; pmin = min(ps)
    yo = YT - tr['R'] * (1 - 0.65 * (zbell0 + 0.15 - tr['zb']) / TIER_H) - 0.42
    tr['y_orn'] = float(yo)
    bells = []
    nb = len(ids_b)
    for j, i in enumerate(ids_b):
        xb = -0.85 + 1.7 * j / (nb - 1)
        if k % 2 == 1: xb = -xb                               # ornaments drop in Stella's direction of the row
        bells.append((xb, zbell0 + 0.07 * (LEAD[i][1] - pmin)))
    zrail = max(b[1] for b in bells) + 0.6
    tr['rail'] = float(zrail); tr['bells'] = []; cups = []
    for j, i in enumerate(ids_b):
        xb, zc = bells[j]; t = TN[i]
        o = B[f'orn{O_I}'] = Ball(f'orn{O_I}', 'orn', RO); O_I += 1
        h = zrail - RO - 0.06 - zc; tfall = math.sqrt(2 * h / G)
        p0 = np.array([xb, yo, zc + h]); pc_ = np.array([xb, yo, zc]); vin = np.array([0, 0, -G * tfall])
        o.rest(0.0, t - tfall, p0); o.fly(t - tfall, t, p0, [0, 0, 0])
        best = None
        tree_orns = [B[b_['ball']] for b_ in tr['bells']]
        for zcup in np.linspace(zc - 0.25, zc - 1.0, 16):
            for fx in (0.75, 0.6, 0.9, 0.45):
                xc = fx * xb
                yf = tier_front_y(tr, xc, zcup)
                if yf is None: continue
                tgt = np.array([xc, yf + 0.04, zcup])
                if any(np.linalg.norm(tgt - c) < 2 * RO + 0.07 for c in cups): continue
                for tf in np.linspace(0.3, 0.75, 46):
                    vo = vout_for(tgt[0] - xb, tgt[2] - zc, tf, tgt[1] - yo); e, n = bounce_e(vin, vo)
                    if not (0.18 <= e <= 0.85) or vo[2] - G * tf > -0.6: continue
                    if math.degrees(math.acos(n[2])) > 40: continue
                    sc_ = abs(e - 0.45) + 0.3 * abs(tf - 0.45)
                    if best is not None and sc_ >= best[0]: continue
                    ok = True
                    for uu in np.linspace(0, tf + 0.3, 24):
                        q = pc_ + vo * min(uu, tf) + np.array([0, 0, -0.5 * G * min(uu, tf) ** 2])
                        for ob in tree_orns:
                            if np.linalg.norm(q - ob.at(t + uu)) < 2 * RO + 0.03: ok = False; break
                        if not ok: break
                    if ok: best = (sc_, tgt, tf, vo)
                if best is not None: break
            if best is not None: break
        assert best is not None, ('ornament', k, j)
        _, tgt, tf, vo = best; cups.append(tgt)
        add_hit(o, i, pc_, vin, vo, 'bell', t=t)
        o.fly(t, t + tf, pc_, vo); o.rest(t + tf, FINAL + 10, tgt)
        o.hits[-1]['cup'] = tgt.tolist(); o.hits[-1]['t_cup'] = float(t + tf); o.hits[-1]['tier'] = k
        tr['bells'].append(dict(i=int(i), x=float(xb), z=float(zc), ball=o.name))
        EV['orn'].append(dict(ball=o.name, tier=k, t_rel=float(t - tfall), t_hit=float(t), t_cup=float(t + tf), cup=tgt.tolist()))
        SFX.append(['plink', float(t + tf)])
    tr['t_lit'] = float(max(o_['t_cup'] for o_ in EV['orn'] if o_['tier'] == k))
Z_TOP = EV['tiers'][8]['top'] + 0.25                       # ball centre on the star holder at the very top
EV['z_top'] = Z_TOP

# ------------------------------------------------------------------ pinecones: the bass in verses 2 and 3
BASS = E['bass']
P_I = 0
for k in range(3, 9):
    bs = [b for b in BASS if b[5] == k]
    d = 1 if k % 2 == 0 else -1
    hz = [h['p'][2] for h in S.hits if h['kind'] == 'key' and LEAD[h['i']][8] == k]
    zd0 = min(hz) - 1.0
    xs = np.linspace(-1.25, 1.35, len(bs)) * d
    zrail = zd0 + 0.62
    cups = []
    EV['pine'].append(dict(k=k, rail=float(zrail), x0=float(min(xs) - 0.2), x1=float(max(xs) + 0.2)))
    for j, b in enumerate(bs):
        t, pitch = b[0], b[1]
        x = float(xs[j]); zc = zd0 + 0.04 * (pitch - 48) * 0.0
        o = B[f'pine{P_I}'] = Ball(f'pine{P_I}', 'pine', RP); P_I += 1
        h = zrail - RP - 0.05 - zc; tfall = math.sqrt(2 * h / G)
        p0 = np.array([x, Y_PINE, zc + h]); pc_ = np.array([x, Y_PINE, zc]); vin = np.array([0, 0, -G * tfall])
        o.rest(0.0, t - tfall, p0); o.fly(t - tfall, t, p0, [0, 0, 0])
        last = (k == 8 and j == len(bs) - 1)
        if last:
            add_hit(o, -1, pc_, vin, [0, 0, 0], 'bassfinal', t=t, pitch=pitch, bass_j=BASS.index(b))
            o.rest(t, FINAL + 10, pc_)
            continue
        best = None
        for tf in np.linspace(0.32, 0.6, 15):
            for dxc in (0.3, 0.22, 0.38):
                tgt = np.array([x + d * dxc, Y_PINE - 0.36, zc - 0.42])
                if any(np.linalg.norm(tgt - c) < 2 * RP + 0.04 for c in cups): continue
                vo = vout_for(tgt[0] - x, tgt[2] - zc, tf, tgt[1] - Y_PINE); e, n = bounce_e(vin, vo)
                if not (0.18 <= e <= 0.8) or math.degrees(math.acos(n[2])) > 35: continue
                best = (tgt, tf, vo); break
            if best: break
        assert best, ('pine', k, j)
        tgt, tf, vo = best; cups.append(tgt)
        kind = 'trigger' if (k == 8 and j == len(bs) - 2) else 'bass'
        add_hit(o, -1, pc_, vin, vo, kind, t=t, pitch=pitch, bass_j=BASS.index(b))
        o.fly(t, t + tf, pc_, vo); o.rest(t + tf, FINAL + 10, tgt)
        o.hits[-1]['cup'] = tgt.tolist(); o.hits[-1]['t_cup'] = float(t + tf)
        SFX.append(['thud', float(t + tf)])
    EV['pine'][-1]['trough_z'] = float(zd0 - 0.42 - RP)

# ------------------------------------------------------------------ the finale: star lift + star cannon
D8 = EV['dev'][8]; pc = np.array(D8['catch']); d = D8['d']
trig = [h for b in B.values() for h in b.hits if h['kind'] == 'trigger'][0]
t_fire = trig['t'] + 0.04
TFS = FINAL - t_fire
top = np.array([0.0, YT, Z_TOP])
hS = 0.5 * G * TFS ** 2
pcan = np.array([d * 2.25, YT, top[2] - hS])
v0 = np.array([(top[0] - pcan[0]) / TFS, 0.0, G * TFS])
m0, m1 = D8['t_catch'] + 0.25, t_fire - 0.3
def f8(t):
    u_ = smooth01((t - m0) / (m1 - m0)); w_ = u_
    return np.array([pc[0] + (pcan[0] - pc[0]) * u_, YT * w_, pc[2] + (pcan[2] - pc[2]) * u_])
S.rest(D8['t_catch'], m0, pc); S.path(m0, m1, sample(f8, m0, m1)); S.rest(m1, t_fire, pcan)
assert np.allclose(f8(m1), pcan, atol=1e-6)
S.fly(t_fire, FINAL, pcan, v0)
add_hit(S, i := REFR[8][-1], top, v0 + np.array([0, 0, -G * TFS]), [0, 0, 0], 'top', t=FINAL)
S.rest(FINAL, FINAL + 10, top)
D8.update(m0=m0, m1=m1, cannon=pcan.tolist(), t_fire=t_fire, v0=v0.tolist(), top=top.tolist())
EV['cannon'] = dict(p=pcan.tolist(), t_fire=t_fire, v=v0.tolist(), top=top.tolist(), trigger=trig['p'])
SFX += [['motor', m0], ['clunk', m1], ['cannon', t_fire], ['star', FINAL]]


def check():
    info = {}
    hs = [(n, h) for n, b in B.items() for h in b.hits if np.linalg.norm(h['vout']) > 0]
    info['e_range'] = (round(min(h['e'] for n, h in hs), 3), round(max(h['e'] for n, h in hs), 3))
    info['speed_out_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for n, h in hs), 3)
    info['tilt_max'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for n, h in hs), 1)
    cov = set(h['i'] for b in B.values() for h in b.hits if h['i'] >= 0)
    info['melody_uncovered'] = sorted(set(range(NN)) - cov)
    info['rows_z'] = [round(min(h['p'][2] for h in S.hits if h['kind'] == 'key' and LEAD[h['i']][8] == k), 2) for k in range(9)]
    info['rows_x'] = [(round(min(h['p'][0] for h in S.hits if h['kind'] == 'key' and LEAD[h['i']][8] == k), 2),
                       round(max(h['p'][0] for h in S.hits if h['kind'] == 'key' and LEAD[h['i']][8] == k), 2)) for k in range(9)]
    info['catch'] = [tuple(round(c, 2) for c in D_['catch']) for D_ in EV['dev']]
    info['z_top'] = round(Z_TOP, 2)
    info['cannon_v'] = round(float(np.linalg.norm(EV['cannon']['v'])), 2)
    info['launch_v'] = [round(float(np.linalg.norm(D_['v0'])), 2) for D_ in EV['dev'] if D_['kind'] == 'catapult']
    return info


if __name__ == '__main__':
    for k_, v_ in check().items(): print(k_, v_)
    out = dict(G=G, R={'stella': RS, 'orn': RO, 'pine': RP, 'snowball': RB}, y_pine=Y_PINE, yt=YT, tier_h=TIER_H,
               balls={n: dict(kind=b.kind, r=b.r, seg=b.seg, hits=b.hits) for n, b in B.items()}, ev=EV, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
