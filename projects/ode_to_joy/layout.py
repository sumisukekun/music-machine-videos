"""Ode to Joy - '年末の大合唱 / 何人集まる？' (the year-end choir: how many will join?). Physics + story planner.
Run after arrange.py; writes layout.json (read by scene.py, render_audio.py, verify.py).

Story (side view of a great concert hall on New Year's Eve, the course runs along +x and slowly steps down):
  0-2.3 s   one gold ball under a spotlight on a tall podium; the counter says 合唱団 1人. It drops onto the first key.
  bars 1-4  it sings the tune alone (every melody note = the ball landing on a key of that pitch)
  bar 4     its long E lands on a LEVER key: the rod opens the GATE of the bass ball's ledge, the bass ball rolls down its
            ramp and rings its first bass bell on bar 5 (2人)
  bar 8     the bass ball lands on the bell on the near end of a SEE-SAW: the far end throws the harmony pair out of its cup;
            they ring their first chord on bar 9 beat 2 (4人)
  12-13     four balls roll off the BALCONY one after another and land in the lanes behind the soloist, each exactly on a
            note, and sing along in unison on the same long keys (8人)
  17-18     eight more from the upper balcony (16人)
  bar 24    the great curtain opens: an empty choir riser with 987 seats and 18 golden tubular bells under the organ
  25-32     the organ pipes rain: 16 balls per beat drop onto the bells that sound the chord of that beat and bounce into
            their seats; the seats they fill draw '2027' (16 + 496 = 512 = 2^9 balls for the Ninth); final chord -> fireworks
Physics: one constant gravity G; between contacts pure parabolas; every bounce is an impulse along the surface normal
(tangential velocity kept) with restitution emin <= e <= emax, so a bounce never adds speed. Ramps: rolling from rest,
a = (5/7) g sin(alpha). The see-saw: the bass ball gives up energy (low e) - more than the pair receives. The rain: from the
organ's slot mouths, free fall onto the top of a tube (contact where the tube's normal is the impulse direction), then into the seat."""
import json, os, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = 7.0
RS, RB, RP, RR = 0.2, 0.3, 0.15, 0.12      # soloist/choir, bass, pair, rain balls
LANE = 0.5                                 # melody lanes: lane j at y = j * LANE (j = 0 soloist)
Y_BASS, Y_PAIR = -1.5, -2.6
DXC = 0.2                                  # pair: half spacing along x
EMAX, EMIN, TILT = 0.97, 0.22, 34.0
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
TN = np.array([l[0] for l in LEAD]); NN = len(TN)
T0, Q, BAR, FINAL = E['t0'], E['q'], E['bar'], E['final']
BT = {int(k): v for k, v in E['bar_t'].items()}
SFX = []; EV = {}
rng = np.random.default_rng(9)


class Ball:
    def __init__(s, name, r, role): s.name = name; s.r = r; s.role = role; s.seg = []; s.hits = []
    def rest(s, t0, t1, p): s.seg.append(dict(k='rest', t0=float(t0), t1=float(t1), p=[float(c) for c in p]))
    def fly(s, t0, t1, p0, v0): s.seg.append(dict(k='fly', t0=float(t0), t1=float(t1), p=[float(c) for c in p0], v=[float(c) for c in v0]))
    def path(s, ts, ps): s.seg.append(dict(k='path', t0=float(ts[0]), t1=float(ts[-1]), ts=[float(t) for t in ts], ps=[[float(c) for c in p] for p in ps]))
    def at(s, t):
        for g in s.seg:
            if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9: return seg_at(g, t)
        if t < s.seg[0]['t0']: return seg_at(s.seg[0], s.seg[0]['t0'])
        return seg_at(s.seg[-1], s.seg[-1]['t1'])
    def hit(s, t, p, vin, vout, kind, pitch=None, ev=None, **kw):
        e, n = bounce_e(np.array(vin, float), np.array(vout, float))
        d = dict(t=float(t), p=[float(c) for c in p], vin=[float(c) for c in vin], vout=[float(c) for c in vout], n=[float(c) for c in n],
                 e=float(e), kind=kind, pitch=pitch, ev=ev); d.update(kw); s.hits.append(d); return d


def seg_at(g, t):
    if g['k'] == 'rest': return np.array(g['p'])
    if g['k'] == 'fly':
        u = t - g['t0']; return np.array(g['p']) + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
    ts = np.array(g['ts']); ps = np.array(g['ps'])
    return np.array([np.interp(t, ts, ps[:, c]) for c in range(3)])


def bounce_e(vin, vout):
    d = vout - vin; nn = np.linalg.norm(d)
    if nn < 1e-9: return 1.0, np.array([0, 0, 1.0])
    n = d / nn; a = vin @ n
    return (-(vout @ n) / a if a < -1e-9 else 9.0), n


KEY_LEAD, KEY_STAY = 1.2, 1.25           # keys exist from t-1.2 s; they start shrinking at t+0.9 s and are gone by t+1.25 s


def axes_of(n):
    up = np.zeros_like(n); up[..., 1] = 1.0
    xa = np.cross(up, n); xa /= np.linalg.norm(xa, axis=-1, keepdims=True) + 1e-12
    return xa, np.cross(n, xa)


def pen_box(P, top, n, ext, R):
    xa, ya = axes_of(np.asarray(n, float))
    c = np.asarray(top) - np.asarray(n) * ext[2]
    if P.ndim == 3 and np.ndim(c) == 2: c, xa, ya, nn = c[:, None], xa[:, None], ya[:, None], np.asarray(n)[:, None]
    else: nn = np.asarray(n)
    rel = P - c
    lx = np.abs((rel * xa).sum(-1)) - ext[0]; ly = np.abs((rel * ya).sum(-1)) - ext[1]; lz = np.abs((rel * nn).sum(-1)) - ext[2]
    d = np.sqrt(np.maximum(lx, 0) ** 2 + np.maximum(ly, 0) ** 2 + np.maximum(lz, 0) ** 2)
    inside = (lx < 0) & (ly < 0) & (lz < 0)
    return np.where(inside, R + 0.1, R - d)


def fly_pts(p, v, taus):
    return p + v * taus[..., None] + np.array([0, 0, -0.5 * G]) * (taus[..., None] ** 2)


def plan(times, p0, vins, xt, zt, yt, emax=EMAX, emin=EMIN, tilt=TILT, W=220, NX=17, NZ=31, vxlim=(0.5, 3.4),
         land_min=0.3, final=None, dzr=(-2.0, 1.0), R=RS, ext=(0.23, 0.22, 0.07), hist0=(), eover=None):
    """beam search over landing points (x, z) for a ball (or a group moving as one) that strikes at `times`
    (see blue_danube/layout.py). eover: {hop index: (emin, emax)} overrides the restitution window of one bounce."""
    ct = math.cos(math.radians(tilt))
    S = [(0.0, np.array(p0, float), [np.array(v, float) for v in vins], (), tuple(hist0), ())]
    n = len(times)
    for k in range(n - 1):
        d = times[k + 1] - times[k]; t1 = times[k + 1]; tk = times[k]
        dn = times[k + 2] - times[k + 1] if k + 2 < n else d
        need = (0.5 * G * dn) ** 2 + vxlim[0] ** 2
        e_lo, e_hi = (eover or {}).get(k, (emin, emax))
        y1 = yt(t1); new = []
        last = (k == n - 2) and final is not None
        tau_out = np.append(np.arange(0.03, d - 0.02, 0.025), d - 0.004)
        for c, p, vl, h, fl, keys in S:
            dy = y1 - p[1]
            if last:
                VX = np.array([(final[0] - p[0]) / d]); DZ = np.array([final[1] - p[2]])
            else:
                vxo = np.linspace(vxlim[0], vxlim[1], NX); dzs = np.linspace(dzr[0], dzr[1], NZ) * max(0.4, d / 0.66)
                VX, DZ = np.meshgrid(vxo, dzs); VX = VX.ravel(); DZ = DZ.ravel()
            VY = np.full_like(VX, dy / d); VZ = DZ / d + 0.5 * G * d
            ok = np.ones(len(VX), bool); NRM = None
            for v in vl:
                dvx, dvy, dvz = VX - v[0], VY - v[1], VZ - v[2]
                nn = np.sqrt(dvx ** 2 + dvy ** 2 + dvz ** 2) + 1e-9
                nx, ny, nz = dvx / nn, dvy / nn, dvz / nn
                if NRM is None: NRM = np.stack([nx, ny, nz], 1)
                a = v[0] * nx + v[1] * ny + v[2] * nz; b = VX * nx + VY * ny + VZ * nz
                e = np.where(a < -1e-9, -b / np.minimum(a, -1e-9), 9)
                ok &= (e >= e_lo) & (e <= e_hi) & (nz >= ct)
            land = VZ - G * d
            ok &= land <= -land_min
            if not ok.any(): continue
            idx = np.nonzero(ok)[0]
            VO = np.stack([VX[idx], VY[idx], VZ[idx]], 1); NR = NRM[idx]; TOP = p - R * NR
            if len(tau_out):
                P_out = p + VO[:, None, :] * tau_out[None, :, None] + np.array([0, 0, -0.5 * G]) * (tau_out[None, :, None] ** 2)
                bad = (pen_box(P_out, TOP, NR, ext, R) > 0.004).any(1)
                for (tkey, ktop, kn) in keys:
                    alive = (tk + tau_out) < tkey + KEY_STAY
                    if alive.any(): bad |= (pen_box(P_out[:, alive], ktop, kn, ext, R) > 0.004).any(1)
            else: bad = np.zeros(len(idx), bool)
            for (f0, fp, fv, f1) in fl:
                lo_, hi_ = max(f0, tk - KEY_LEAD), min(f1, tk - 0.03)
                if hi_ <= lo_: continue
                tt = np.arange(lo_, hi_, 0.025); P_in = fly_pts(np.array(fp), np.array(fv), tt - f0)
                bad |= (pen_box(np.broadcast_to(P_in, (len(idx),) + P_in.shape), TOP, NR, ext, R) > 0.004).any(1)
            X = xt(t1); Z = zt(t1)
            for jj in np.nonzero(~bad)[0]:
                j = idx[jj]
                p1 = np.array([p[0] + VX[j] * d, y1, p[2] + DZ[j]])
                ke = (VX[j] ** 2 + VY[j] ** 2 + land[j] ** 2) * emax ** 2
                wgt = 1.0 + (8.0 * (k / max(1, n - 2)) ** 2 if final is not None else 0.0)
                cc = c + wgt * ((p1[0] - X) ** 2 + (p1[2] - Z) ** 2) + 3.0 * max(0.0, need - ke)
                vo = VO[jj]; vland = np.array([VX[j], VY[j], land[j]])
                fl2 = tuple(f for f in fl if f[3] > tk - KEY_LEAD - 1.0)[-4:] + ((tk, p, vo, t1),)
                ks2 = tuple(kk for kk in keys if kk[0] + KEY_STAY > tk)[-4:] + ((tk, TOP[jj], NR[jj]),)
                new.append((cc, p1, [vland], h + ((float(tk), p, vl, vo),), fl2, ks2))
        if not new: raise RuntimeError(f'plan dead at hop {k} t={times[k]:.2f}')
        new.sort(key=lambda s: s[0]); keep = []; seen = set()
        for s_ in new:
            key = (round(s_[2][0][0] / 0.08), round(s_[2][0][2] / 0.15), round(s_[1][2] / 0.06), round(s_[1][0] / 0.06))
            if key in seen: continue
            seen.add(key); keep.append(s_)
            if len(keep) >= W: break
        S = keep
    best = S[0]
    hits = [dict(t=t, p=p, vins=vl, vout=vo) for t, p, vl, vo in best[3]]
    return hits, best[1], best[2][0], best[4]


def plan_retry(*a, **kw):
    err = None
    for W, NX, NZ in ((220, 17, 31), (400, 21, 41), (700, 25, 51)):
        kw.update(W=W, NX=NX, NZ=NZ)
        try: return plan(*a, **kw)
        except RuntimeError as e_: err = e_; print('retry', e_)
    raise err


def ev_idx(bar, pos):
    for i, l in enumerate(LEAD):
        if l[4] == bar and abs(l[5] - pos) < 1e-6: return i
    raise KeyError((bar, pos))


def solve_bounce(vin, vout_fn, ranges):
    """search the free parameter(s) of an outgoing flight for a legal bounce; returns the first hit"""
    for prm in ranges:
        vo = vout_fn(prm); e, n = bounce_e(np.array(vin), vo)
        if EMIN <= e <= 0.9 and n[2] >= math.cos(math.radians(TILT)) and np.linalg.norm(vo) <= np.linalg.norm(vin): return vo, prm
    raise RuntimeError('no legal bounce')


BALLS = {}
# ============================================================ the soloist: podium drop, then the whole tune
SOLO = Ball('solo', RS, 'melody'); BALLS['solo'] = SOLO
X0, Z0 = 0.0, 0.0
T_DROP = 1.0
H_DROP = 0.5 * G * T_DROP ** 2
SOLO.rest(0.0, T0 - T_DROP, (X0, 0, Z0 + H_DROP)); SOLO.fly(T0 - T_DROP, T0, (X0, 0, Z0 + H_DROP), (0, 0, 0))
EV['podium'] = dict(x=X0, z=Z0 + H_DROP, t_release=T0 - T_DROP)
SFX.append(['release', T0 - T_DROP])
I_GATE = ev_idx(4, 2)                       # the long E of bar 4: the lever key
I_J1 = [ev_idx(12, 3), ev_idx(13, 1), ev_idx(13, 2), ev_idx(13, 3)]
I_J2 = [ev_idx(17, 0), ev_idx(17, 1), ev_idx(17, 2), ev_idx(17, 3), ev_idx(18, 0), ev_idx(18, 1), ev_idx(18, 2), ev_idx(18, 3)]
T25 = BT[25]
V1, V2 = 1.3, 0.62
def xt_mel(t): return X0 + V1 * (min(t, T25) - T0) + V2 * max(0.0, t - T25)
def zt_mel(t): return Z0 - 0.045 * (t - T0)
MH = []                                     # melody hits: dict(t, p, vin, vout)
CH = [(0, 44), (44, 88), (88, NN - 1)]
import pickle, hashlib
_ck = hashlib.md5((open(__file__).read().split('# ---- melody plan cache key end')[0] + json.dumps(LEAD)).encode()).hexdigest()
_cf = f'/tmp/ode_melody_{_ck}.pkl'
# ---- melody plan cache key end
if os.path.exists(_cf): MH, p_, v_ = pickle.load(open(_cf, 'rb'))
else:
    p_, v_ = np.array([X0, 0, Z0]), np.array([0, 0, -G * T_DROP]); hist = ()
    for a, b in CH:
        hs, p_, v_, hist = plan_retry(TN[a:b + 1], p_, [v_], xt_mel, zt_mel, lambda t: 0.0, R=RS, hist0=hist)
        MH += [dict(t=h['t'], p=np.array(h['p']), vin=np.array(h['vins'][0]), vout=np.array(h['vout'])) for h in hs]
    pickle.dump((MH, p_, v_), open(_cf, 'wb'))
print('final contact', np.round(p_, 3), 'vin', np.round(v_, 3))
P_FIN, V_FIN = p_, v_                       # contact of the final note (its bounce is planned below)
assert len(MH) == NN - 1
# the final note: the choir lands on the last key and stays there (a dead landing: e = 0, no bounce) for the bows
T_BENCH = FINAL
P_BENCH = P_FIN
MH.append(dict(t=FINAL, p=P_FIN, vin=V_FIN, vout=np.zeros(3)))
# lanes active at each melody hit (joiners count from the note they land on)
J_IDX = I_J1 + I_J2
LANES_AT = [1 + sum(1 for j in J_IDX if j <= i) for i in range(NN)]
for i, h in enumerate(MH):
    kind = 'lever' if i == I_GATE else ('final' if i == NN - 1 else 'key')
    SOLO.hit(h['t'], h['p'], h['vin'], h['vout'], kind, LEAD[i][1], ev=i, lanes=LANES_AT[i])
    if i + 1 < NN: SOLO.fly(h['t'], MH[i + 1]['t'], h['p'], h['vout'])
SOLO.rest(FINAL, FINAL + 30, P_FIN)
EV['lever'] = dict(t=float(TN[I_GATE]), p=MH[I_GATE]['p'].tolist(), n=SOLO.hits[I_GATE]['n'])

# ============================================================ the joiners: balcony perch -> ramp -> into lane j on a note
ALPHA_J = math.radians(22.0)
JOIN = []
for j, i in enumerate(J_IDX, start=1):
    h = MH[i]; n = np.array(SOLO.hits[i]['n']); vout = h['vout']; tj = h['t']
    pc = h['p'] + np.array([0, j * LANE, 0])
    best = None
    for ej in (0.5, 0.45, 0.55, 0.4, 0.6, 0.35, 0.65, 0.3, 0.7):
        lam = (vout @ n) * (1 + 1 / ej); vin = vout - lam * n
        vx = vin[0]
        if vx < 0.25: continue
        vex = np.array([vx, 0, -vx * math.tan(ALPHA_J)])
        T = (vex[2] - vin[2]) / G
        if T < 0.25: continue
        best = (ej, vin, vex, T); break
    assert best, ('joiner', j)
    ej, vin, vex, T = best
    lip = pc - vex * T + np.array([0, 0, 0.5 * G * T * T])
    s_ex = np.linalg.norm(vex); a = 5 / 7 * G * math.sin(ALPHA_J)
    T_r = s_ex / a; L = s_ex ** 2 / (2 * a)
    u = np.array([math.cos(ALPHA_J), 0, -math.sin(ALPHA_J)])
    perch = lip - u * L
    t_rel = tj - T - T_r
    b = Ball(f'j{j:02d}', RS, 'melody'); BALLS[b.name] = b
    b.rest(0.0, t_rel, perch)
    tt = np.arange(t_rel, t_rel + T_r + 1e-9, 1 / 240.0)
    b.path(tt, [perch + u * (0.5 * a * (t - t_rel) ** 2) for t in tt])
    b.fly(t_rel + T_r, tj, lip, vex)
    off = np.array([0, j * LANE, 0])
    for k in range(i, NN):
        hk = MH[k]
        kind = 'join' if k == i else ('final' if k == NN - 1 else 'key')
        b.hit(hk['t'], hk['p'] + off, vin if k == i else hk['vin'], hk['vout'], kind, LEAD[k][1], ev=k, lanes=LANES_AT[k])
        if k + 1 < NN: b.fly(hk['t'], MH[k + 1]['t'], hk['p'] + off, hk['vout'])
    b.rest(FINAL, FINAL + 30, P_FIN + off)
    JOIN.append(dict(j=j, ev=i, t=float(tj), e=ej, perch=perch.tolist(), lip=lip.tolist(), t_rel=float(t_rel), T_ramp=T_r, L=L,
                     alpha=ALPHA_J, T_fly=T))
    SFX.append(['gate_small', float(t_rel)])
EV['join'] = JOIN

# ============================================================ helpers: a smoothed track of the melody (keeps the band in shot)
MX = np.array([h['p'][0] for h in MH]); MZ = np.array([h['p'][2] for h in MH]); MT = np.array([h['t'] for h in MH])
def track(t, w=1.8):
    m = np.abs(MT - t) < w
    if not m.any(): m = np.argsort(np.abs(MT - t))[:4]
    return float(MX[m].mean()), float(MZ[m].min())

# ============================================================ the bass ball: ledge -> gate -> ramp -> bass bells (beats 1 and 3)
BE = E['bass']; TBASS = np.array([b[0] for b in BE])
XO_B, ZO_B = 0.35, -1.15
I_SEESAW = [k for k, b in enumerate(BE) if b[2] == 8 and b[3] == 2][0]
x1, z1 = track(TBASS[0]); P_B0 = np.array([x1 + XO_B, Y_BASS, z1 + ZO_B])
ALPHA_B = math.radians(24.0)
T_GATE = float(TN[I_GATE]) + 0.06                 # the lever's rod opens the gate a moment after the key is struck
T_FB = 0.36                                       # flight from the ramp lip to the first bell
T_RB = TBASS[0] - T_FB - T_GATE
aB = 5 / 7 * G * math.sin(ALPHA_B); sB = aB * T_RB; LB = 0.5 * aB * T_RB ** 2
uB = np.array([math.cos(ALPHA_B), 0, -math.sin(ALPHA_B)])
vexB = uB * sB
lipB = P_B0 - vexB * T_FB + np.array([0, 0, 0.5 * G * T_FB ** 2])
ledgeB = lipB - uB * LB
vinB0 = vexB + np.array([0, 0, -G * T_FB])
BASS = Ball('bass', RB, 'bass'); BALLS['bass'] = BASS
BASS.rest(0.0, T_GATE, ledgeB)
tt = np.arange(T_GATE, T_GATE + T_RB + 1e-9, 1 / 240.0)
BASS.path(tt, [ledgeB + uB * (0.5 * aB * (t - T_GATE) ** 2) for t in tt])
BASS.fly(T_GATE + T_RB, TBASS[0], lipB, vexB)
EV['bass_gate'] = dict(t=T_GATE, ledge=ledgeB.tolist(), lip=lipB.tolist(), alpha=ALPHA_B, L=LB, T_ramp=T_RB, v_exit=sB)
SFX.append(['gate', T_GATE])
FIN_B = None
def hx_b(t): return track(t)[0] + XO_B
def hz_b(t): return track(t)[1] + ZO_B
hb, pBE, vBE, _ = plan_retry(TBASS, P_B0, [vinB0], hx_b, hz_b, lambda t: Y_BASS, emax=0.975, emin=0.25, vxlim=(0.4, 3.0),
                             dzr=(-1.0, 0.6), R=RB, ext=(0.26, 0.26, 0.2), eover={I_SEESAW: (0.25, 0.5)})
for k, h in enumerate(hb):
    kind = 'seesaw' if k == I_SEESAW else 'bell'
    BASS.hit(h['t'], h['p'], h['vins'][0], h['vout'], kind, BE[k][1], ev=k)
    t1 = hb[k + 1]['t'] if k + 1 < len(hb) else TBASS[-1]
    BASS.fly(h['t'], t1, h['p'], h['vout'])
# last bass bell = the final chord: the bass ball stays on it (dead landing)
BASS.hit(TBASS[-1], pBE, vBE, np.zeros(3), 'bell', BE[-1][1], ev=len(BE) - 1, final=True)
BASS.rest(TBASS[-1], FINAL + 30, pBE)
hS = BASS.hits[I_SEESAW]
KE_loss = 0.5 * (np.linalg.norm(hS['vin']) ** 2 - np.linalg.norm(hS['vout']) ** 2)          # per unit bass mass

# ============================================================ the see-saw and the harmony pair
# the plank lies across the bass lane, pointing diagonally forward (unit vector U from the far end to the near end). Before the
# hit the near end is up (angle +TH0), the far end rests down with the pair in its cup. The bass ball lands on the bell on the
# near end; the plank swings to -TH0 with constant angular acceleration and stops: the cup is moving along its arc at speed
# S_L at that instant and throws the pair (tangent of the arc: forward along U and up).
PE = E['pair']; TPAIR = np.array([p[0] for p in PE])
ARM = 0.85; TH0 = math.radians(25.0); PHI = math.radians(25.0); CUP_H = 0.12 + RP; BELL_H = 0.36
U = np.array([math.cos(PHI), -math.sin(PHI), 0.0]); ZZ = np.array([0, 0, 1.0])
pS = np.array(hS['p']); topS = pS - RB * np.array(hS['n'])
near_up = topS - ZZ * BELL_H
pivot = near_up - ARM * (U * math.cos(TH0) + ZZ * math.sin(TH0))
def plank_d(al): return U * math.cos(al) + ZZ * math.sin(al)
def plank_n(al): return -U * math.sin(al) + ZZ * math.cos(al)
def cup_c(al): return pivot - ARM * plank_d(al) + CUP_H * plank_n(al)
S_L = 5.0
T_SS = float(hS['t']); T_SWING = 4 * ARM * TH0 / S_L; T_L = T_SS + T_SWING
def al_t(t): u = min(1.0, max(0.0, (t - T_SS) / T_SWING)); return TH0 - 2 * TH0 * u * u
CUP = cup_c(TH0); C_L = cup_c(-TH0)
eps = 1e-5; vL = (cup_c(al_t(T_L)) - cup_c(al_t(T_L - eps))) / eps
dtl = TPAIR[0] - T_L
p0P = C_L + vL * dtl + np.array([0, 0, -0.5 * G * dtl * dtl])
vinP0 = vL + np.array([0, 0, -G * dtl])
KE_pair = 2 * (RP / RB) ** 3 * 0.5 * float(vL @ vL)                    # per unit bass mass (2 small balls)
EV['seesaw'] = dict(t=T_SS, pivot=pivot.tolist(), arm=ARM, th0=TH0, phi=PHI, u=U.tolist(), t_swing=T_SWING, cup=CUP.tolist(),
                    launch=C_L.tolist(), t_launch=T_L, v_launch=vL.tolist(), ke_bass_loss=KE_loss, ke_pair=KE_pair, bell_h=BELL_H,
                    cup_h=CUP_H)
SFX.append(['seesaw', T_SS]); SFX.append(['launch', T_L])
XO_P, ZO_P = -0.1, -1.9
x_catch = lambda t: min(1.0, max(0.0, (t - TPAIR[0]) / 4.5))
def hx_p(t): return (1 - x_catch(t)) * (p0P[0] + 0.9 * (t - TPAIR[0])) + x_catch(t) * (track(t)[0] + XO_P)
def hz_p(t): return (1 - x_catch(t)) * (p0P[2] - 0.1 * (t - TPAIR[0])) + x_catch(t) * (track(t)[1] + ZO_P)
AXP = np.array([math.sin(PHI), math.cos(PHI), 0.0])          # the see-saw's pivot axis: the two balls sit side by side along it
def hy_p(t): return p0P[1] + (Y_PAIR - p0P[1]) * x_catch(t) ** 0.7
hp, pPE, vPE, _ = plan_retry(TPAIR, p0P, [vinP0], hx_p, hz_p, hy_p, emax=0.975, emin=0.3, vxlim=(0.3, 3.2),
                             dzr=(-1.0, 0.6), R=RP, ext=(0.16 + 2 * DXC * AXP[0], 0.16 + 2 * DXC * AXP[1], 0.17))
PAIRB = [Ball('pairlo', RP, 'pair'), Ball('pairhi', RP, 'pair')]
OFFP = [-DXC * AXP, DXC * AXP]
for b, o in zip(PAIRB, OFFP):
    BALLS[b.name] = b
    b.rest(0.0, T_SS, CUP + o)
    tt = np.arange(T_SS, T_L + 1e-9, 1 / 240.0)
    b.path(tt, [cup_c(al_t(t)) + o for t in tt])
    b.fly(T_L, TPAIR[0], C_L + o, vL)
    for k, h in enumerate(hp):
        b.hit(h['t'], h['p'] + o, h['vins'][0], h['vout'], 'tink', PE[k][1][0 if o[1] < 0 else 1], ev=k)
        t1 = hp[k + 1]['t'] if k + 1 < len(hp) else TPAIR[-1]
        b.fly(h['t'], t1, h['p'] + o, h['vout'])
for b, o in zip(PAIRB, OFFP):
    b.hit(TPAIR[-1], pPE + o, vPE, np.zeros(3), 'tink', PE[-1][1][0 if o[1] < 0 else 1], ev=len(PE) - 1, final=True)
    b.rest(TPAIR[-1], FINAL + 30, pPE + o)

# ============================================================ the riser (987 seats), the picture, the bells and the rain
from PIL import Image, ImageDraw, ImageFont
NCOL, NROW = 47, 21
PX, RISE, DEPTH = 0.34, 0.3, 0.34
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc'
def picture(size, stroke):
    im = Image.new('L', (NCOL * 8, NROW * 8), 0); d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, size, index=0)
    bb = d.textbbox((0, 0), '2027', font=f, stroke_width=stroke)
    d.text(((im.width - (bb[2] - bb[0])) / 2 - bb[0], (im.height - (bb[3] - bb[1])) / 2 - bb[1]), '2027', font=f, fill=255,
           stroke_width=stroke, stroke_fill=255)
    a = np.array(im.resize((NCOL, NROW), Image.BOX)) > 110
    return a
best = None
for size in range(110, 200, 4):
    for stroke in (0, 2, 4, 6, 8):
        a = picture(size, stroke); c = int(a.sum())
        if a[:, 0].any() or a[:, -1].any() or a[0].any() or a[-1].any(): continue
        if best is None or abs(c - 496) < abs(best[0] - 496): best = (c, size, stroke, a)
PIX = best[3].copy()
# exactly 496: add / drop pixels on the outline (deterministic)
while PIX.sum() > 496:
    r_, c_ = np.argwhere(PIX)[-1]; PIX[r_, c_] = False
add_cand = [(r_, c_) for r_ in range(1, NROW - 1) for c_ in range(1, NCOL - 1) if not PIX[r_, c_] and
            (PIX[r_ - 1, c_] or PIX[r_ + 1, c_] or PIX[r_, c_ - 1] or PIX[r_, c_ + 1])]
k_ = 0
while PIX.sum() < 496:
    PIX[add_cand[k_]] = True; k_ += 1
assert PIX.sum() == 496
# riser placement: centred on the melody's x-range of bars 25-32, behind the choir lanes
x25 = float(MH[ev_idx(25, 0)]['p'][0]); xfin = float(P_FIN[0])
XC = 0.5 * (x25 + xfin) + 0.4
YR = 13 * LANE + 0.8                         # front edge of tier 0
z_mel_end = float(min(h['p'][2] for h in MH[ev_idx(25, 0):]))
ZR = z_mel_end + 0.2                         # top of tier 0
X_COL = XC + (np.arange(NCOL) - (NCOL - 1) / 2) * PX
def seat(r_, c_):
    """row r_ = 0 at the top of the picture -> tier NROW-1-r_"""
    k = NROW - 1 - r_
    return np.array([X_COL[c_], YR + k * DEPTH + DEPTH / 2, ZR + k * RISE + RR])
TOP_Z = ZR + (NROW - 1) * RISE
# the great tubular bells: 18 golden tubes lying across x in front of the riser, two of every pitch (left half / right half,
# same order), so every chord has tubes all across the width. The rain drops out of the organ's slot mouths onto their tops.
TUBE_P = [69, 76, 71, 78, 73, 79, 74, 80, 70]            # order chosen so that chord tones alternate along the row
NT = 18; LT = 0.82; RT = 0.2
TX = XC + (np.arange(NT) - (NT - 1) / 2) * 0.92
TP = TUBE_P + TUBE_P
BY = YR - 0.8                                            # in front of the riser: every seat is behind the tubes
BZ = TOP_Z + 2.4
H_PIPE = 3.4                                             # organ slot mouths above the tube tops
def tube_dist(P):
    """distance from points P (..., 3) to the surface of the nearest tube"""
    d = []
    for x in TX:
        q = np.stack([np.clip(P[..., 0], x - LT / 2, x + LT / 2), np.full(P.shape[:-1], BY), np.full(P.shape[:-1], BZ)], -1)
        d.append(np.linalg.norm(P - q, axis=-1) - RT)
    return np.min(np.stack(d), axis=0)
RAIN = E['rain']; assert len(RAIN) * 16 == 496
RAINB = []; BELLHITS = []
def rain_cands(S, cx):
    """candidate paths (pipe mouth, v0, Tf, contact centre C, vin, vout, T2, n, e), best first, for a ball that strikes the tube
    whose axis is at (., BY, BZ) at x = cx. The tube's normal has no x part, so the ball keeps its x speed: it drifts out of
    the slot mouth with exactly the x speed it needs to reach its seat."""
    out = []
    for T2 in np.linspace(0.75, 1.75, 41):
        for Tf in (0.5, 0.55, 0.6, 0.45, 0.65, 0.4, 0.7):
            n = np.array([0, 0, 1.0])
            for _ in range(60):
                C = np.array([cx, BY, BZ]) + (RT + RR) * n
                vout = (S - C) / T2 + np.array([0, 0, 0.5 * G * T2])
                P_p = np.array([C[0] - vout[0] * Tf, C[1], BZ + RT + H_PIPE])
                v0 = np.array([vout[0], 0, (C[2] - P_p[2]) / Tf + 0.5 * G * Tf]); vin = v0 + np.array([0, 0, -G * Tf])
                d = vout - vin; n2 = d / np.linalg.norm(d)
                if np.linalg.norm(n2 - n) < 1e-9: break
                n = 0.5 * n + 0.5 * n2; n /= np.linalg.norm(n)
            e, n2 = bounce_e(vin, vout)
            if np.linalg.norm(n2 - n) > 1e-5: continue
            if not (0.15 <= e <= 0.88 and n[2] >= math.cos(math.radians(62)) and np.linalg.norm(vout) <= np.linalg.norm(vin)): continue
            vland = vout + np.array([0, 0, -G * T2]); slope = -vland[2] / max(1e-6, abs(vland[1]))
            if slope < 0.35: continue                    # (approach from the front: only the lower tiers are passed)
            tau = np.linspace(0.0, Tf - 0.02, 25); Pin = P_p + v0 * tau[:, None] + np.array([0, 0, -0.5 * G]) * tau[:, None] ** 2
            tau2 = np.linspace(0.02, T2, 50); Pout = C + vout * tau2[:, None] + np.array([0, 0, -0.5 * G]) * tau2[:, None] ** 2
            if tube_dist(np.concatenate([Pin, Pout])).min() - RR < 0.01: continue
            out.append((abs(e - 0.6) + 0.02 * T2 + 0.05 * abs(vout[0]), P_p, v0, Tf, C, vin, vout, T2, n, e))
    out.sort(key=lambda c: c[0])
    return [c[1:] for c in out]
def traj(P_p, v0, Tf, C, vout, T2, t_out, ts):
    """positions at times ts (fall, flight; rest in the seat afterwards; at the pipe mouth before)"""
    tk = t_out + Tf
    u1 = np.clip(ts - t_out, 0, Tf)[:, None]; u2 = np.clip(ts - tk, 0, T2)[:, None]
    p1 = P_p + v0 * u1 + np.array([0, 0, -0.5 * G]) * u1 ** 2
    p2 = C + vout * u2 + np.array([0, 0, -0.5 * G]) * u2 ** 2
    return np.where((ts < tk)[:, None], p1, p2)
TG0 = RAIN[0][0] - 1.0; TG = np.arange(TG0, FINAL + 4.0, 1 / 120.0)
ACC_P = np.full((520, len(TG), 3), 1e3)           # sampled paths of the accepted rain balls (far away before they appear)
NACC = [0]
def clash(cand, t_out):
    """does this path come closer than 2 cm to any accepted rain ball (flying, or already sitting in its seat)?"""
    P_p, v0, Tf, C, vin, vout, T2, n, e = cand
    i0 = int((t_out - TG0) * 120); i1 = min(len(TG), int((t_out + Tf + T2 - TG0) * 120) + 240)
    me = traj(P_p, v0, Tf, C, vout, T2, t_out, TG[i0:i1])
    if NACC[0] == 0: return False
    d = np.linalg.norm(ACC_P[:NACC[0], i0:i1] - me[None], axis=2) - 2 * RR
    return bool(d.min() < 0.02)
def accept(cand, t_out):
    P_p, v0, Tf, C, vin, vout, T2, n, e = cand
    i0 = int((t_out - TG0) * 120)
    ACC_P[NACC[0], i0:] = traj(P_p, v0, Tf, C, vout, T2, t_out, TG[i0:]); NACC[0] += 1
# which seats fill on which beat: every beat takes 16 free seats near the tubes of its chord (chosen at random among the
# nearest), so the picture fills in all over the riser and only becomes readable at the end
POOL = [tuple(p) for p in rng.permutation(np.argwhere(PIX))]
fails = 0
for kb, (tk, ch, bells, bar, pos) in enumerate(RAIN):
    tubes = [i for i in range(NT) if TP[i] in bells]
    quota = [16 // len(tubes) + (1 if q < 16 % len(tubes) else 0) for q in range(len(tubes))]
    rng.shuffle(quota)
    for ti, qn in zip(tubes, quota):
        near = sorted(POOL, key=lambda s: abs(seat(*s)[0] - TX[ti]))[:max(qn, 3 * qn)]
        pick = [near[i] for i in rng.choice(len(near), size=qn, replace=False)]
        pick.sort(key=lambda s: seat(*s)[0])
        for q, s_ in enumerate(pick):
            POOL.remove(s_)
            S = seat(*s_)
            slots = list(TX[ti] + (np.arange(qn) - (qn - 1) / 2) * min(0.3, (LT - 0.16) / max(1, qn - 1)) if qn > 1 else [TX[ti]])
            order_c = [(ti, slots[q])] + [(ti, x) for x in slots if x != slots[q]]
            # fallback: any tube of this chord, anywhere along it (nearest tubes first)
            order_c += [(t2, TX[t2] + dx) for t2 in sorted(tubes, key=lambda t2: abs(TX[t2] - S[0])) for dx in np.linspace(-0.33, 0.33, 7)]
            done = False
            for k_c, (ti_c, cx) in enumerate(order_c):
                for cand in rain_cands(S, cx)[:150 if k_c < qn else 400]:
                    t_out = tk - cand[2]
                    if clash(cand, t_out): continue
                    P_p, v0, Tf, C, vin, vout, T2, n, e = cand
                    b = Ball(f'r{len(RAINB):03d}', RR, 'rain')
                    b.rest(0.0, t_out, P_p); b.fly(t_out, tk, P_p, v0)
                    b.hit(tk, C, vin, vout, 'rainbell', TP[ti_c], ev=kb, tube=int(ti_c))
                    b.fly(tk, tk + T2, C, vout); b.rest(tk + T2, FINAL + 30, S)
                    b.seat = (int(s_[0]), int(s_[1])); b.t_seat = tk + T2; b.t_out = t_out
                    RAINB.append(b); BALLS[b.name] = b; BELLHITS.append([float(tk), TP[ti_c]])
                    accept(cand, t_out); done = True; break
                if done: break
            if not done: fails += 1; print('rain fail', kb, s_, TP[ti])
assert fails == 0, ('rain balls without a legal path', fails)
EV['riser'] = dict(ncol=NCOL, nrow=NROW, px=PX, rise=RISE, depth=DEPTH, xc=XC, yr=YR, zr=ZR, x_col=X_COL.tolist(),
                   pix=PIX.astype(int).tolist(), font_size=best[1], stroke=best[2])
EV['bells'] = dict(pitches=TP, x=TX.tolist(), y=BY, z=BZ, r=RT, length=LT, h_pipe=H_PIPE)
EV['rain'] = [dict(name=b.name, seat=b.seat, t_out=float(b.t_out), t_seat=float(b.t_seat)) for b in RAINB]
T_CURTAIN = BT[24] + 2 * Q                    # the curtain opens during the last long note of bar 24
EV['curtain'] = dict(t0=T_CURTAIN, t1=T_CURTAIN + 1.6)
SFX.append(['curtain', T_CURTAIN])
# head count for the counter (time, count)
CNT = [(0.0, 1), (float(TBASS[0]), 2), (float(TPAIR[0]), 4)]
for J in JOIN: CNT.append((J['t'], None))
for b in RAINB: CNT.append((float(b.t_seat), None))
CNT.sort(key=lambda c: c[0]); cnt = []; c = 0
for t, v in CNT:
    c = v if v is not None else c + 1
    cnt.append((t, c))
EV['count'] = cnt
assert cnt[-1][1] == 512, cnt[-1]


def check():
    info = {}
    hs = [(n, h) for n, b in BALLS.items() for h in b.hits if np.linalg.norm(h['vout']) > 0]
    info['e_range'] = (round(min(h['e'] for n, h in hs), 3), round(max(h['e'] for n, h in hs), 3))
    info['speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / max(1e-6, np.linalg.norm(h['vin'])) for n, h in hs), 3)
    info['tilt_max_deg(keys/bells)'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for n, h in hs if h['kind'] != 'rainbell'), 1)
    info['seesaw_energy(bass loss vs pair gain, per bass mass)'] = (round(KE_loss, 2), round(KE_pair, 2))
    for n in ('solo', 'bass', 'pairlo'):
        b = BALLS[n]; zz = [h['p'][2] for h in b.hits]; xx = [h['p'][0] for h in b.hits]
        info[n] = dict(hits=len(b.hits), z=(round(min(zz), 2), round(max(zz), 2)), x=(round(min(xx), 1), round(max(xx), 1)))
    info['joiners'] = [(J['j'], round(J['e'], 2), round(J['T_fly'], 2), round(J['L'], 2)) for J in JOIN]
    info['bass_ramp'] = (round(LB, 3), round(T_RB, 3), round(sB, 3))
    info['picture'] = (int(PIX.sum()), best[1], best[2])
    info['rain_e'] = (round(min(b.hits[0]['e'] for b in RAINB), 2), round(max(b.hits[0]['e'] for b in RAINB), 2))
    info['count_final'] = cnt[-1]
    return info


if __name__ == '__main__':
    info = check()
    for k, v in info.items(): print(k, v)
    out = dict(G=G, R=dict(solo=RS, bass=RB, pair=RP, rain=RR), lane=LANE, y_bass=Y_BASS, y_pair=Y_PAIR, dxc=DXC,
               balls={n: dict(r=b.r, role=b.role, seg=b.seg, hits=b.hits) for n, b in BALLS.items()},
               lanes_at=LANES_AT, bellhits=BELLHITS, ev=EV, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
