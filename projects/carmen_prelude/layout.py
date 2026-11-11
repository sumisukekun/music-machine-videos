"""Carmen Prelude - '闘牛場のビー玉' (marbles in the bullring). Physics + story planner.
Run after arrange.py; writes layout.json (read by scene.py, render_audio.py, verify.py).

Cast: the two toreros (red + gold glass balls, side by side) play the melody on xylophone keys: one wide key when they
play the same note, two keys for the two notes of a third / octave.  The bull (big black ball with horns) plays the bass
on drums.  Two spectator balls in the stands play the off-beat chords on bells; in the charge the whole crowd joins in.
Story
  intro    the bull's eyes glow behind the toril gate; the toreros roll down the ramp of the president's box and drop
           onto the first key (downbeat)
  A1       paseillo: the toreros parade across the arena, the bull stamps along behind the barrera (callejon)
  A2       the toril gate bursts open: the bull is in the arena and chases them, closing the gap
  B1       side by side: toreros in thirds, the bull's own melody in the bass
  B2       face-off: both hammer the same notes; then the bull hangs back up on the terrace to take a run-up,
           the toreros land in the two cups of the TREBUCHET in the middle of the ring
  B3       the CHARGE: the bull thunders down the terrace stairs (rising chromatic bass), the crowd wave (tremolo)
           sweeps the stands; the bull steps on the trigger plate -> the counterweight drops -> the arm flings the
           toreros sky-high a moment before the bull reaches the empty cups
  A3       the toreros land on the far side right on the downbeat of the theme; the bull catches up; final chord:
           everybody lands together, flower shower
Physics: one constant gravity G; between contacts pure parabolas; every bounce is an impulse along the key's surface
normal (tangential velocity kept) with restitution emin <= e <= emax, so a bounce never adds speed.  The only energy
sources are visible mechanisms: the president's ramp (rolling from rest), the trebuchet (falling counterweight, the
balls ride the arm's cups and leave it at the stop with the cups' velocity), trapdoors (drops from rest)."""
import json, os, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = 7.0
RP, RB, RC, RM = 0.2, 0.36, 0.17, 0.08     # toreros, bull, spectators, crowd marbles
DYP = 0.32                                 # toreros: red on the near side (y = -DYP), gold on the far side
DXC, DYC = 0.24, 0.2                       # spectator couple: lo on the far side, hi on the near side
EMAX, EMIN, TILT = 0.97, 0.22, 34.0
KW = 0.30                                  # key width along the run
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
TN = np.array([l[0] for l in LEAD]); NN = len(TN)
T0, Q, S16, BAR, FINAL = E['t0'], E['q'], E['s16'], E['bar'], E['final']
SECS = E['secs']
V = 3.1                                    # mean run speed of the toreros (m/s)
LANE_Y = {'pair': 0.0, 'bull_callejon': 2.55, 'bull': 1.45, 'spec': 3.6}
BARRERA_Y = 2.05
SFX = []; EV = {}
def tb(bar, pos): return T0 + (bar - 1) * BAR + pos * S16


class Ball:
    def __init__(s, name, r): s.name = name; s.r = r; s.seg = []; s.hits = []
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


KEY_LEAD, KEY_STAY = 1.2, 1.25         # keys / drums / bells exist from t-1.2 s; they shrink from t+0.9 s, gone by t+1.25 s


def axes_of(n):
    up = np.zeros_like(n); up[..., 1] = 1.0
    xa = np.cross(up, n); xa /= np.linalg.norm(xa, axis=-1, keepdims=True) + 1e-12
    return xa, np.cross(n, xa)


def pen_box(P, top, n, ext, R):
    """penetration of balls (centres P, radius R) into boxes whose top centre is `top` with normal n, half extents ext"""
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


def plan(times, p0, vins, xt, zt, yt, emax=EMAX, emin=EMIN, tilt=TILT, W=160, NX=11, NZ=31, vxlim=(0.5, 3.4),
         land_min=0.3, final=None, dzr=(-2.0, 1.0), R=RP, ext=(0.16, 0.22, 0.07), hist0=(), minsep=0.0, wz=1.0):
    """beam search over landing points (x, z) for a ball (or a couple moving as one) that strikes at `times`.
    p0 = contact centre at times[0], vins = incoming velocities at times[0] (all must give a legal bounce).
    xt, zt: target landing x, z (functions of time); yt: landing y (fixed, not searched).
    Legal bounce: emin <= e <= emax, key tilt <= tilt, still falling at least land_min m/s at the next contact, no flight
    through a key that exists at that moment; consecutive keys at least `minsep` apart along x.
    final=(x, z): the last landing must be exactly there.  Returns (hits for times[:-1], p_end, vin_end)."""
    ct = math.cos(math.radians(tilt))
    S = [(0.0, np.array(p0, float), [np.array(v, float) for v in vins], (), tuple(hist0), ())]
    n = len(times)
    for k in range(n - 1):
        d = times[k + 1] - times[k]; t1 = times[k + 1]; tk = times[k]
        dn = times[k + 2] - times[k + 1] if k + 2 < n else d
        need = (0.5 * G * dn) ** 2 + vxlim[0] ** 2
        y1 = yt(t1); new = []
        last = (k == n - 2) and final is not None
        tau_out = np.append(np.arange(0.03, d - 0.02, 0.025), d - 0.004)
        for c, p, vl, h, fl, keys in S:
            dy = y1 - p[1]
            if last:
                VX = np.array([(final[0] - p[0]) / d]); DZ = np.array([final[1] - p[2]])
            else:
                lo = max(vxlim[0], minsep / d)
                vxo = np.linspace(lo, max(lo + 0.05, vxlim[1]), NX)
                dzs = np.unique(np.concatenate([np.linspace(dzr[0], dzr[1], NZ) * max(0.4, d / 0.66),
                                                np.array([-0.035, -0.025, -0.018, -0.012, -0.007, -0.003, 0.004, 0.01, 0.02]) * max(1.0, d / 0.12)]))
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
                ok &= (e >= emin) & (e <= emax) & (nz >= ct)
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
                cc = c + wgt * ((p1[0] - X) ** 2 + wz * (p1[2] - Z) ** 2) + 3.0 * max(0.0, need - ke)
                vo = VO[jj]; vland = np.array([VX[j], VY[j], land[j]])
                fl2 = tuple(f for f in fl if f[3] > tk - KEY_LEAD - 1.0)[-6:] + ((tk, p, vo, t1),)
                ks2 = tuple(kk for kk in keys if kk[0] + KEY_STAY > tk)[-6:] + ((tk, TOP[jj], NR[jj]),)
                new.append((cc, p1, [vland], h + ((float(tk), p, vl, vo),), fl2, ks2))
        if not new: raise RuntimeError(f'plan dead at hop {k} t={times[k]:.2f}')
        new.sort(key=lambda s: s[0]); keep = []; seen = set()
        for s_ in new:
            key = (round(s_[2][0][0] / 0.08), round(s_[2][0][2] / 0.12), round(s_[1][2] / 0.05), round(s_[1][0] / 0.05))
            if key in seen: continue
            seen.add(key); keep.append(s_)
            if len(keep) >= W: break
        S = keep
    best = S[0]
    hits = [dict(t=t, p=p, vins=vl, vout=vo) for t, p, vl, vo in best[3]]
    return hits, best[1], best[2][0]


import pickle, hashlib
def plan_cached(tag, *a, **kw):
    """dev cache (CACHE=1): the plan result for `tag` is reused while its inputs (times, start) are unchanged"""
    key = hashlib.md5(repr((tag, np.round(a[0], 5).tolist(), np.round(a[1], 5).tolist(), [np.round(v, 5).tolist() for v in a[2]],
                            sorted((k, str(v)) for k, v in kw.items() if not callable(v)))).encode()).hexdigest()
    fn = f'/tmp/carmen_cache/{tag}_{key}.pkl'
    if os.environ.get('CACHE') and os.path.exists(fn): return pickle.load(open(fn, 'rb'))
    r = plan_retry(*a, **kw)
    os.makedirs('/tmp/carmen_cache', exist_ok=True); pickle.dump(r, open(fn, 'wb')); return r


def plan_retry(*a, **kw):
    err = None
    for W, NX, NZ in ((kw.pop('W', 160), kw.pop('NX', 11), kw.pop('NZ', 31)), (320, 15, 41), (600, 21, 51)):
        kw.update(W=W, NX=NX, NZ=NZ)
        try: return plan(*a, **kw)
        except RuntimeError as e_: err = e_; print('retry', e_)
    raise err


def solve_hop(vin, dt, dz_lo=-1.5, dz_hi=-0.05, vxs=(1.6, 0.6), vy=0.0, need_fall=0.3):
    """a single hop of duration dt after a bounce with incoming vin: first legal (dz, vx) (highest landing first)"""
    for dz in np.linspace(dz_hi, dz_lo, 300):
        for vx in np.linspace(vxs[0], vxs[1], 21):
            vo = np.array([vx, vy, dz / dt + 0.5 * G * dt]); e, n = bounce_e(vin, vo)
            if EMIN <= e <= EMAX and n[2] >= math.cos(math.radians(TILT)) and vo[2] - G * dt < -need_fall: return vo
    raise RuntimeError('no hop')


R_ = Ball('red', RP); W_ = Ball('gold', RP)
PAIR = {'red': R_, 'gold': W_}
OFF = {'red': np.array([0.0, -DYP, 0.0]), 'gold': np.array([0.0, +DYP, 0.0])}
STRUCK = {}                                # melody event -> list of (ball, pitch, kind)


def strike(i, ball, kind, voice):
    ps = LEAD[i][1]; p = max(ps) if voice == 'hi' else min(ps)
    STRUCK.setdefault(i, []).append((ball, p, kind)); return p


def write_pair(hits, i0, t_end, kind_shared='uni'):
    for k, h in enumerate(hits):
        i = i0 + k; ps = LEAD[i][1]
        t1 = hits[k + 1]['t'] if k + 1 < len(hits) else t_end
        for j, (nme, b) in enumerate(PAIR.items()):
            if len(ps) == 1: kind = kind_shared; p = strike(i, nme, kind, 'hi')
            else: kind = 'key'; p = strike(i, nme, kind, 'hi' if nme == 'red' else 'lo')
            vin = h['vins'][j if len(h['vins']) > 1 else 0]
            b.hit(h['t'], h['p'] + OFF[nme], vin, h['vout'], kind, p, ev=i)
            b.fly(h['t'], t1, h['p'] + OFF[nme], h['vout'])


# ============================================================ intro: the president's ramp
X0, Z0 = 0.0, 0.0
T_ROLL = 1.0; T_DROP = 0.5                 # roll down the ramp from rest, then fly off its lip
A_RAMP = V / T_ROLL; SIN_R = A_RAMP / (5 / 7 * G); ALPHA_R = math.asin(SIN_R)
lip = np.array([X0 - V * T_DROP, 0.0, Z0 + 0.5 * G * T_DROP ** 2])
L_RAMP = 0.5 * A_RAMP * T_ROLL ** 2
top = lip - np.array([L_RAMP * math.cos(ALPHA_R), 0, -L_RAMP * math.sin(ALPHA_R)])
t_roll0 = T0 - T_DROP - T_ROLL
tt = np.append(np.arange(t_roll0, T0 - T_DROP - 1e-6, 1 / 240.0), T0 - T_DROP)
for nme, b in PAIR.items():
    b.rest(0.0, t_roll0, top + OFF[nme])
    ps_ = [top + OFF[nme] + np.array([math.cos(ALPHA_R), 0, -math.sin(ALPHA_R)]) * 0.5 * A_RAMP * (t - t_roll0) ** 2 for t in tt]
    b.path(tt, np.array(ps_))
    b.fly(T0 - T_DROP, T0, lip + OFF[nme], [V, 0, 0])
EV['ramp'] = dict(top=top.tolist(), lip=lip.tolist(), alpha=ALPHA_R, t0=t_roll0, t1=T0 - T_DROP, L=L_RAMP)
SFX.append(['gate', t_roll0 - 0.05])
vin0 = np.array([V, 0.0, -G * T_DROP])

# ============================================================ toreros: A1 .. B2, ending in the trebuchet cups
I_B3 = int(np.searchsorted(TN, SECS['B3'] - 1e-6))          # first melody event after the charge (= A3 downbeat)
I_CUPS = I_B3 - 1                                           # last B2 note: the two cups of the trebuchet
assert LEAD[I_CUPS][4] == 46 and LEAD[I_B3][4] == 51
I_A2 = int(np.searchsorted(TN, SECS['A2'] - 1e-6)); I_B1 = int(np.searchsorted(TN, SECS['B1'] - 1e-6))
I_B2 = int(np.searchsorted(TN, SECS['B2'] - 1e-6))


def pair_z(t):
    """target height of the toreros' keys: gentle descent; in B2 they run down into the ring (the bull stays up)"""
    z = Z0 - 0.025 * (t - T0)
    if t > SECS['B2']: z -= 3.6 * min(1.0, (t - SECS['B2']) / (TN[I_CUPS] - SECS['B2']))
    return z


ts = TN[0:I_B2 + 1]
hitsP, pC1, vC1 = plan_cached('pair1', ts, (X0, 0, Z0), [vin0], lambda t: X0 + V * (t - T0), pair_z, lambda t: 0.0,
                              vxlim=(2.75, 3.6), minsep=KW + 0.04, ext=(0.16, 0.46, 0.07))
write_pair(hitsP, 0, TN[I_B2])
H1T = np.array([h['t'] for h in hitsP] + [TN[I_B2]]); H1X = np.array([h['p'][0] for h in hitsP] + [pC1[0]])
def pair_x_nom(t): return float(np.interp(t, H1T, H1X)) if t <= TN[I_B2] else float(pC1[0] + V * (t - TN[I_B2]))

# ============================================================ the bull (A1 .. B2): follows the toreros
bull = Ball('bull', RB)
BASS = E['bass']
BT = np.array([b[0] for b in BASS])
I_CH = int(np.searchsorted(BT, SECS['B3'] - 1e-6)); I_A3B = int(np.searchsorted(BT, SECS['A3'] - 1e-6))
BULL_DZ = 1.05                                              # the bull's drums stand this much above the toreros' keys


def bull_lag(t):
    """x of the bull relative to the toreros: behind the barrera (A1), chases (A2), level (B1), drops back (B2)"""
    pts = [(T0, -0.6), (SECS['A2'] - 0.5, -0.8), (SECS['A2'] + 1.5, -3.2), (SECS['B1'], -1.2), (SECS['B1'] + 3, 0.2),
           (SECS['B2'], 0.0), (SECS['B3'], -3.0)]
    return float(np.interp(t, [p[0] for p in pts], [p[1] for p in pts]))


def bull_y(t):
    ta = SECS['A2']
    if t < ta - 0.6: return LANE_Y['bull_callejon']
    u = min(1.0, (t - (ta - 0.6)) / 1.0)
    y = LANE_Y['bull_callejon'] + (LANE_Y['bull'] - LANE_Y['bull_callejon']) * u
    tb1, tb2 = tb(43, 0), tb(45, 0)                          # the lunges of the face-off: the bull leans in
    if tb1 - 0.5 < t < tb2:
        y -= 0.42 * min(1.0, (t - tb1 + 0.5) / 0.5) * min(1.0, (tb2 - t) / 0.4)
    return y


p_b0 = np.array([X0 - 0.6, LANE_Y['bull_callejon'], Z0 + BULL_DZ + (RB - RP)])
TDROP_B = 0.7
bull.rest(0.0, T0 - TDROP_B, p_b0 + [0, 0, 0.5 * G * TDROP_B ** 2])
bull.fly(T0 - TDROP_B, T0, p_b0 + [0, 0, 0.5 * G * TDROP_B ** 2], [0, 0, 0])
EV['toril'] = dict(p=(p_b0 + [0, 0, 0.5 * G * TDROP_B ** 2]).tolist(), t=T0 - TDROP_B)
SFX.append(['bull_drop', T0 - TDROP_B])
ts = BT[:I_CH]
def bx(t): return pair_x_nom(t) + bull_lag(t)
def bz(t): return Z0 + BULL_DZ + (RB - RP) - 0.03 * (t - T0)
hb1, pB, vB = plan_cached('bull1', ts, p_b0, [np.array([0, 0, -G * TDROP_B])], bx, bz, bull_y, R=RB, ext=(0.22, 0.22, 0.14),
                          vxlim=(0.0, 5.5), dzr=(-1.2, 0.8), emax=0.975)
bull_end_A = (pB, vB)

# ============================================================ toreros: B2 down into the ring, ending in the trebuchet cups
CHARGE_DX, CHARGE_DZ = 8.0, 1.9                              # the bull's run: this far and this much down to the cups
x_c, z_c = pB[0] + CHARGE_DX, pB[2] - (RB - RP) - CHARGE_DZ
ts = TN[I_B2:I_CUPS + 1]
t2a, t2b = TN[I_B2], TN[I_CUPS]
hitsP2, pC, vC = plan_cached('pair2', ts, pC1, [vC1], lambda t: pC1[0] + (x_c - pC1[0]) * (t - t2a) / (t2b - t2a),
                             lambda t: pC1[2] + (z_c - pC1[2]) * (t - t2a) / (t2b - t2a), lambda t: 0.0,
                             vxlim=(2.75, 4.0), minsep=KW + 0.04, ext=(0.16, 0.46, 0.07))
write_pair(hitsP2, I_B2, TN[I_CUPS])
cupc = pC.copy()                                            # cups contact centre (pair centre), struck at TN[I_CUPS]
for nme, b in PAIR.items():
    p = strike(I_CUPS, nme, 'cup', 'hi' if nme == 'red' else 'lo')
    b.hit(TN[I_CUPS], cupc + OFF[nme], vC, [0, 0, 0], 'cup', p, ev=I_CUPS)

# ============================================================ the trebuchet
L_ARM = 2.0; L_SHORT = 0.8
PHI0, PHI_REL = math.radians(210.0), math.radians(150.0)
pivot = cupc - L_ARM * np.array([math.cos(PHI0), 0, math.sin(PHI0)])
pivot[1] = 0.0
T_SIG = tb(48, 4)                                           # the bull's drum on the trigger plate
T_S = T_SIG + 0.04                                          # pin pulled, the counterweight starts to fall
T_LAND = TN[I_B3]
V_TIP = 6.6
T_ROT = 2 * (PHI0 - PHI_REL) * L_ARM / V_TIP                # constant angular acceleration from rest
ALPHA = (PHI0 - PHI_REL) * 2 / T_ROT ** 2
T_REL = T_S + T_ROT
def arm_phi(t):
    u = min(max(0.0, t - T_S), T_ROT); return PHI0 - 0.5 * ALPHA * u * u
def tip(t, off=np.zeros(3)):
    ph = arm_phi(t); return pivot + L_ARM * np.array([math.cos(ph), 0, math.sin(ph)]) + off
tt = np.append(np.arange(T_S, T_REL - 1e-6, 1 / 240.0), T_REL)
for nme, b in PAIR.items():
    b.rest(TN[I_CUPS], T_S, cupc + OFF[nme])
    b.path(tt, np.array([tip(t, OFF[nme]) for t in tt]))
w_rel = ALPHA * T_ROT
v_rel = L_ARM * w_rel * np.array([math.sin(PHI_REL), 0, -math.cos(PHI_REL)])
p_rel = tip(T_REL)
T_FL = T_LAND - T_REL
p_land = p_rel + v_rel * T_FL + np.array([0, 0, -0.5 * G * T_FL ** 2]); v_land = v_rel + np.array([0, 0, -G * T_FL])
for nme, b in PAIR.items(): b.fly(T_REL, T_LAND, p_rel + OFF[nme], v_rel)
EV['treb'] = dict(pivot=pivot.tolist(), L=L_ARM, Ls=L_SHORT, phi0=PHI0, phi_rel=PHI_REL, t_sig=T_SIG, t_s=T_S, t_rel=T_REL, alpha=ALPHA,
                  cup=cupc.tolist(), v_rel=v_rel.tolist(), p_rel=p_rel.tolist(), t_cups=float(TN[I_CUPS]),
                  apex=float(p_rel[2] + v_rel[2] ** 2 / (2 * G)))
SFX.append(['cups', float(TN[I_CUPS])]); SFX.append(['trigger', T_SIG]); SFX.append(['release', T_S]); SFX.append(['fling', T_REL])
SFX.append(['stop', T_REL + 0.02])

# ============================================================ toreros: A3 after the flight, final cups
ts = TN[I_B3:]
xl, zl = p_land[0], p_land[2]
hitsA3, pF, vF = plan_cached('pairA3', ts, p_land, [v_land], lambda t: xl + V * (t - T_LAND), lambda t: zl - 0.03 * (t - T_LAND), lambda t: 0.0,
                            vxlim=(2.75, 3.6), minsep=KW + 0.04, ext=(0.16, 0.46, 0.07), emin=0.15)
write_pair(hitsA3, I_B3, FINAL)
finc = pF.copy()
for nme, b in PAIR.items():
    p = strike(NN - 1, nme, 'fincup', 'hi' if nme == 'red' else 'lo')
    b.hit(FINAL, finc + OFF[nme], vF, [0, 0, 0], 'fincup', p, ev=NN - 1)
    b.rest(FINAL, FINAL + 20, finc + OFF[nme])
EV['land'] = dict(t=float(T_LAND), p=p_land.tolist()); EV['final_cups'] = finc.tolist()
SFX.append(['land', float(T_LAND)])

# ============================================================ stage track (keeps the cast in the shot)
allh = sorted([(h['t'], h['p'][0], h['p'][2]) for b in PAIR.values() for h in b.hits])
HT = np.array([a[0] for a in allh]); HX = np.array([a[1] for a in allh]); HZ = np.array([a[2] for a in allh])
def track(t, w=1.5):
    m = np.abs(HT - t) < w
    if not m.any(): m = np.argsort(np.abs(HT - t))[:4]
    return float(HX[m].mean()), float(HZ[m].mean())
def pair_x(t): return float(np.interp(t, HT, HX))

# charge: the remaining 16 notes of B3 down the stairs, the last one lands on the drum where the cups stood
ts = BT[I_CH - 1:I_A3B]
BULL_LAST_Y = 1.1                                           # lands right beside the empty cups (outside the frame)
bull_last = np.array([cupc[0] - 0.1, BULL_LAST_Y, cupc[2] - RP + RB + 0.02])
def cy(t):
    u = (t - ts[0]) / (ts[-1] - ts[0]); return BULL_LAST_Y + (LANE_Y['bull'] - BULL_LAST_Y) * (1 - u) ** 1.4
hb2, pB2, vB2 = plan_cached('bull2', ts, pB, [vB], lambda t: pB[0] + (bull_last[0] - pB[0]) * ((t - ts[0]) / (ts[-1] - ts[0])),
                           lambda t: pB[2] + (bull_last[2] - pB[2]) * ((t - ts[0]) / (ts[-1] - ts[0])) ** 0.8, cy,
                           R=RB, ext=(0.22, 0.22, 0.14), vxlim=(0.0, 4.5), dzr=(-1.2, 0.6), final=(bull_last[0], bull_last[2]), emax=0.975, emin=0.08)
# A3: catch up with the toreros, final drum on the final chord
ts = BT[I_A3B - 1:]
x_f = finc[0] - 1.0; z_f = pB2[2] - 0.03 * (FINAL - T_LAND) - 0.15
hb3, pB3, vB3 = plan_cached('bull3', ts, pB2, [vB2], lambda t: pair_x(t) - 5.3 + 4.7 * min(1.0, (t - T_LAND) / 6.0) if t < FINAL - 0.5 else x_f,
                           lambda t: pB2[2] - 0.03 * (t - T_LAND), lambda t: BULL_LAST_Y + (1.45 - BULL_LAST_Y) * min(1.0, max(0.0, (t - ts[0]) / 1.2)) - 0.5 * min(1.0, max(0.0, (t - (FINAL - 2.0)) / 1.5)),
                           R=RB, ext=(0.22, 0.22, 0.14), vxlim=(0.0, 5.5), dzr=(-1.2, 0.8), emax=0.975, emin=0.08)
I_TRIG = int(np.argmin(np.abs(BT - T_SIG))); assert abs(BT[I_TRIG] - T_SIG) < 1e-6
allb = hb1 + hb2 + hb3
for k, h in enumerate(allb):
    a = BASS[k]
    t1 = allb[k + 1]['t'] if k + 1 < len(allb) else FINAL
    kind = 'trigger' if k == I_TRIG else 'drum'
    bull.hit(h['t'], h['p'], h['vins'][0], h['vout'], kind, a[1], ev=k)
    bull.fly(h['t'], t1, h['p'], h['vout'])
assert len(allb) == len(BASS) - 1
pfb = pB3
bull.hit(FINAL, pfb, vB3, [0, 0, 0], 'findrum', BASS[-1][1], ev=len(BASS) - 1)
bull.rest(FINAL, FINAL + 20, pfb)
EV['bull_final'] = pfb.tolist()

# ============================================================ spectators (off-beat chords): couple 1 (A1, A2), couple 2 (A3)
CH = E['chords']; CT = np.array([c[0] for c in CH])
SPEC = {}
def spec_couple(tag, evs, t_drop):
    names = [f'{tag}lo', f'{tag}hi']; offs = [np.array([-DXC, DYC, 0]), np.array([DXC, -DYC, 0])]
    times = np.array([a[0] for a in evs])
    zo = 2.25
    hx = lambda t: track(t)[0] + 0.9
    hz = lambda t: track(t)[1] + zo
    p0 = np.array([hx(times[0]), LANE_Y['spec'], hz(times[0])])
    hits, pE, vE = plan_cached('spec' + tag, times, p0, [np.array([0, 0, -G * t_drop])], hx, hz, lambda t: LANE_Y['spec'], emax=0.975, emin=0.1,
                              vxlim=(0.0, 4.0), dzr=(-1.0, 0.6), R=RC, ext=(0.16, 0.16, 0.17), minsep=0.0)
    balls = [Ball(n, RC) for n in names]
    for b, o in zip(balls, offs):
        pst = p0 + o
        b.rest(0.0, times[0] - t_drop, pst + [0, 0, 0.5 * G * t_drop ** 2]); b.fly(times[0] - t_drop, times[0], pst + [0, 0, 0.5 * G * t_drop ** 2], [0, 0, 0])
        for k, h in enumerate(hits):
            a = evs[k]
            b.hit(h['t'], h['p'] + o, h['vins'][0], h['vout'], 'bell', a[1][0] if o[0] < 0 else a[1][1], ev=CH.index(a), bar=a[2], pos=a[3])
            t1 = hits[k + 1]['t'] if k + 1 < len(hits) else times[-1]
            b.fly(h['t'], t1, h['p'] + o, h['vout'])
    SPEC[tag] = dict(balls=balls, pE=pE, vE=vE, t_last=float(times[-1]), offs=offs, drop=(p0 + [0, 0, 0.5 * G * t_drop ** 2]).tolist())
    return balls
ev1 = [c for c in CH if c[2] <= 32]; ev2 = [c for c in CH if c[2] >= 51]
assert len(ev1) + len(ev2) == len(CH)
sp1 = spec_couple('s1', ev1, 0.7)
sp2 = spec_couple('s2', ev2, 0.7)
# couple 1: last bell (bar 32) -> they sit down in their box seat (padded cups, silent) and watch
S1 = SPEC['s1']; dt = 0.5; vE = S1['vE']; pE = S1['pE']
for b, o in zip(sp1, S1['offs']):
    last = CH.index(ev1[-1])
    vo = solve_hop(vE, dt, vxs=(max(0.4, vE[0]), 0.3))
    b.hit(S1['t_last'], pE + o, vE, vo, 'bell', ev1[-1][1][0] if o[0] < 0 else ev1[-1][1][1], ev=last, bar=ev1[-1][2], pos=ev1[-1][3])
    b.fly(S1['t_last'], S1['t_last'] + dt, pE + o, vo)
    pseat = pE + o + vo * dt + np.array([0, 0, -0.5 * G * dt * dt])
    b.rest(S1['t_last'] + dt, FINAL + 20, pseat)
    EV.setdefault('seats', []).append(pseat.tolist())
SFX.append(['seat', S1['t_last'] + dt])
# couple 2: last regular bell -> the final bells, mouth up (they land and stay) on the final chord
S2 = SPEC['s2']
fin_ev = ev2[-1]; assert abs(fin_ev[0] - FINAL) < 1e-6
# the planner already ended them at the final time (last event = final chord): turn that landing into a rest
for b, o in zip(sp2, S2['offs']):
    pf = S2['pE'] + o
    b.hit(FINAL, pf, S2['vE'], [0, 0, 0], 'finbell', fin_ev[1][0] if o[0] < 0 else fin_ev[1][1], ev=CH.index(fin_ev), bar=66, pos=4)
    b.rest(FINAL, FINAL + 20, pf)
    EV.setdefault('final_bells', []).append(dict(p=pf.tolist(), pitch=fin_ev[1][0] if o[0] < 0 else fin_ev[1][1]))

# ============================================================ the crowd wave (tremolo): marbles released one by one
CR = E['crowd']
RAIL_Y = {'lo': 3.25, 'hi': 3.95}
xw0 = bull_end_A[0][0] - 1.0; xw1 = cupc[0] + 3.0
zr = track(SECS['B3'])[1] + 2.0
crowd = []
TD_C = 0.26                                                # each marble falls 0.26 s from its ledge
for k, (t, p, bar, pos) in enumerate(CR):
    u = (t - CR[0][0]) / (CR[-1][0] - CR[0][0])
    x = xw0 + (xw1 - xw0) * u
    if bar == 50 and pos >= 6: row = 'hi' if p == max(c[1] for c in CR if c[3] == pos and c[2] == 50) else 'lo'; kind = 'cbell'
    else: row = 'lo' if p == 68 else 'hi'; kind = 'rail'
    zc = zr + (0.55 if row == 'hi' else 0.0) - 0.004 * k        # the rails fall gently along the stands
    pc = np.array([x, RAIL_Y[row], zc])
    b = Ball(f'm{k:02d}', RM)
    ledge = pc + [0, 0, 0.5 * G * TD_C ** 2]
    b.rest(0.0, t - TD_C, ledge); b.fly(t - TD_C, t, ledge, [0, 0, 0])
    vin = np.array([0, 0, -G * TD_C]); vo = np.array([0.0, 0.55, 0.45 * G * TD_C])   # glances off backwards into the gutter
    b.hit(t, pc, vin, vo, kind, p, ev=k, row=row)
    tf = 0.3; b.fly(t, t + tf, pc, vo)
    pg = pc + vo * tf + np.array([0, 0, -0.5 * G * tf * tf])
    b.rest(t + tf, FINAL + 20, pg)
    crowd.append(b)
EV['crowd'] = dict(x0=xw0, x1=xw1, z=zr, rail_y=RAIL_Y, td=TD_C)

BALLS = dict(PAIR); BALLS['bull'] = bull
for b in sp1 + sp2 + crowd: BALLS[b.name] = b


# ============================================================ checks
def check():
    info = {}
    hs = [(n, h) for n, b in BALLS.items() for h in b.hits if np.linalg.norm(h['vout']) > 0 and not n.startswith('m')]
    info['e_range'] = (round(min(h['e'] for n, h in hs), 3), round(max(h['e'] for n, h in hs), 3))
    info['speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / max(1e-6, np.linalg.norm(h['vin'])) for n, h in hs), 3)
    info['tilt_max_deg'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for n, h in hs), 1)
    info['lead_events_uncovered'] = sorted(set(range(NN)) - set(STRUCK))
    for n in ('red', 'bull', 's1lo', 's2lo'):
        b = BALLS[n]; zz = [h['p'][2] for h in b.hits]; xx = [h['p'][0] for h in b.hits]
        info[n] = dict(hits=len(b.hits), z=(round(min(zz), 2), round(max(zz), 2)), x=(round(min(xx), 1), round(max(xx), 1)))
    info['treb'] = {k: round(v, 3) for k, v in EV['treb'].items() if isinstance(v, float)}
    info['v_rel'] = round(float(np.linalg.norm(v_rel)), 2); info['land'] = np.round(p_land, 2).tolist()
    return info


if __name__ == '__main__':
    info = check()
    for k, v in info.items(): print(k, v)
    out = dict(G=G, R=dict(pair=RP, bull=RB, spec=RC, marble=RM), lane_y=LANE_Y, barrera_y=BARRERA_Y, dyp=DYP, dxc=DXC, dyc=DYC, kw=KW,
               balls={n: dict(r=b.r, seg=b.seg, hits=b.hits) for n, b in BALLS.items()},
               struck={str(k): v for k, v in STRUCK.items()}, ev=EV, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
