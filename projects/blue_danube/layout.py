"""The Blue Danube - '氷上の舞踏会' (a ball on the frozen Danube). Physics + story planner.
Run after arrange.py; writes layout.json (read by scene.py, render_audio.py, verify.py).

Cast: the dancing couple (red + white glass balls) plays the melody on ice keys; behind them the 'orchestra':
  the big gold ball rings a bass bell on every beat 1 ('ズン'), two small couples ring ice bells on beats 2 and 3 ('チャッ・チャッ').
Story
  0-1.4 s  the couple drops out of a chandelier onto the first key (the pickup note)
  pass 1   they waltz side by side: one wide key when they play the same note, two keys for the two voices of a chord
  crack    the last note of pass 1 is a big shared key: it cracks under both of them, the ice splits
  pass 2   the two halves drift apart: red's floe sinks toward the camera, red hops into a SLED and rides down,
           the sled hits a snowbank and throws red onto the lower keys; white plays alone, rolls into a REVOLVING DOOR,
           is carried round, drops onto the ICE SLIDE and brushes the ice chimes on the way down, then chases red
  reunion  both land on the same key again (bar 21 of pass 2), side by side to the end
  finale   they drop into the two cups of a SEE-SAW; the gold bass ball lands on the other end and throws them up into the
           great chandelier bell -> final chord, the chandelier lights up
Physics: one constant gravity G; between contacts pure parabolas; every bounce is an impulse along the key's surface normal
(tangential velocity kept) with restitution emin <= e <= emax, so a bounce never adds speed. Fixed note times force the keys to
tilt: short hops are taken fast and flat, long hops slow and high (horizontal <-> vertical energy, never gained). The sled
slides with friction, the slide rolls with (5/7) g sin(a) minus rolling resistance, the door turns at constant speed. The
only energy sources are visible mechanisms (the falling gold ball on the see-saw)."""
import json, os, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = 7.0
RP, RB, RC = 0.2, 0.3, 0.19            # couple, bass ball, chord balls
DXP = 0.25                             # couple: red leads by 2*DXP ...
DYP = 0.22                             # ... and dances on the near lane (white on the far lane)
DYC = 0.2                              # chord couples: lo on the far side, hi on the near side
DXC = 0.26                             # chord couples: half spacing
EMAX, EMIN, TILT = 0.97, 0.22, 34.0
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
TN = np.array([l[0] for l in LEAD]); NN = len(TN)
T0, Q, BAR, FINAL = E['t0'], E['q'], E['bar'], E['final']
V = 1.5                                # mean course speed (m/s)
LANE_Y = {'pair': 0.0, 'bass': 1.2, 'ch2': 1.95, 'ch3': 2.85}
SFX = []; EV = {}


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


KEY_LEAD, KEY_STAY = 1.2, 1.25         # keys / bells exist from t-1.2 s; they start shrinking at t+0.9 s and are gone by t+1.25 s


def axes_of(n):
    """n: (..., 3) unit normals -> xa, ya (..., 3)"""
    up = np.zeros_like(n); up[..., 1] = 1.0
    xa = np.cross(up, n); xa /= np.linalg.norm(xa, axis=-1, keepdims=True) + 1e-12
    return xa, np.cross(n, xa)


def pen_box(P, top, n, ext, R):
    """penetration of balls (centres P, radius R) into key boxes whose top centre is `top` with normal n.
    P: (C, S, 3) or (S, 3); top, n: (C, 3) or (3,); ext = (hw, hd, th). returns (C, S) or (S,) penetration (>0 = overlap)"""
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
         land_min=0.3, final=None, wfin=40.0, dzr=(-2.0, 1.0), R=RP, ext=(0.23, 0.22, 0.07), hist0=()):
    """2-D beam search over landing points (x, z) for a ball (or a couple moving as one) that strikes at `times`.
    p0 = (x, y, z) contact centre at times[0], vins = incoming velocities at times[0] (all must give a legal bounce).
    xt, zt: target landing x, z (functions of time); yt: landing y (fixed, not searched).
    Legal bounce: emin <= e <= emax, key tilt <= `tilt`, ball still falling at least land_min m/s at the next contact,
    and no flight passes through a key that exists at that moment (own key after the bounce, earlier keys, the new key
    against earlier flights).  final=(x, z): the last landing must be exactly there.
    hist0: earlier flights ((t0, p, v, t1), ...) to check the first keys against.
    Returns list of hits [dict(t, p, vins, vout)] for times[:-1] and (p_end, vin_end)."""
    ct = math.cos(math.radians(tilt))
    S = [(0.0, np.array(p0, float), [np.array(v, float) for v in vins], (), tuple(hist0), ())]   # cost, p, vins, hits, flights, keys
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
                ok &= (e >= emin) & (e <= emax) & (nz >= ct)
            land = VZ - G * d
            ok &= land <= -land_min
            if not ok.any(): continue
            idx = np.nonzero(ok)[0]
            VO = np.stack([VX[idx], VY[idx], VZ[idx]], 1); NR = NRM[idx]; TOP = p - R * NR
            # (a) outgoing flight vs this key (after leaving it) and vs earlier keys still standing
            if len(tau_out):
                P_out = p + VO[:, None, :] * tau_out[None, :, None] + np.array([0, 0, -0.5 * G]) * (tau_out[None, :, None] ** 2)
                bad = (pen_box(P_out, TOP, NR, ext, R) > 0.004).any(1)
                for (tkey, ktop, kn) in keys:
                    alive = (tk + tau_out) < tkey + KEY_STAY
                    if alive.any(): bad |= (pen_box(P_out[:, alive], ktop, kn, ext, R) > 0.004).any(1)
            else: bad = np.zeros(len(idx), bool)
            # (b) earlier flights vs this key while it stands (from tk - KEY_LEAD)
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
    return hits, best[1], best[2][0]


def plan_retry(*a, **kw):
    err = None
    for W, NX, NZ in ((220, 17, 31), (400, 21, 41), (700, 25, 51)):
        kw.update(W=W, NX=NX, NZ=NZ)
        try: return plan(*a, **kw)
        except RuntimeError as e_: err = e_; print('retry', e_)
    raise err


def apply(ball, hits, kinds, dx=0.0):
    """write the planned hops of a (couple) centre into a ball, shifted by dx along x"""
    sh = np.array([dx, 0, 0])
    for k, h in enumerate(hits):
        t0 = h['t']; t1 = hits[k + 1]['t'] if k + 1 < len(hits) else None
        vin = h['vins'][0] if len(h['vins']) == 1 else h['vins'][kinds[k][2]]
        kd, pitch, _ = kinds[k][0], kinds[k][1], None
        ball.hit(t0, h['p'] + sh, vin, h['vout'], kd, pitch, ev=kinds[k][3])
    return


def fly_between(ball, hits, t_end, p_end, dx=0.0):
    sh = np.array([dx, 0, 0])
    for k, h in enumerate(hits):
        t1 = hits[k + 1]['t'] if k + 1 < len(hits) else t_end
        ball.fly(h['t'], t1, h['p'] + sh, h['vout'])


def ev_of(t): return int(np.argmin(np.abs(TN - t)))
def pitches(i): return LEAD[i][1]


R_ = Ball('red', RP); W_ = Ball('white', RP)
PAIR = {'red': R_, 'white': W_}
OFF = {'red': np.array([+DXP, -DYP, 0.0]), 'white': np.array([-DXP, +DYP, 0.0])}
STRUCK = {}                                # event -> list of (ball, pitch, kind)


def strike(i, ball, kind, voice):
    ps = pitches(i); p = max(ps) if voice == 'hi' else min(ps)
    STRUCK.setdefault(i, []).append((ball, p, kind)); return p


# ============================================================ opening: the couple falls out of the first chandelier
X0, Z0 = 0.0, 0.0
H_DROP = 0.5 * G * 1.0 ** 2
for nme, b in PAIR.items():
    p_ch = np.array([X0, 0.0, Z0 + H_DROP]) + OFF[nme]
    b.rest(0.0, T0 - 1.0, p_ch); b.fly(T0 - 1.0, T0, p_ch, [0, 0, 0])
EV['chandelier0'] = dict(x=X0, z=Z0 + H_DROP, t_release=T0 - 1.0)
SFX.append(['release', T0 - 1.0])
vin0 = np.array([0.0, 0.0, -G * 1.0])

# ============================================================ pass 1 together (events 0..69); crack on event 68
I_CRACK = 68; I_SPLIT = 69
I_SLED = 77; I_W_DOOR = 87; I_R_BACK = 88; I_SLIDE0, I_SLIDE1 = 96, 105; I_W_BACK = 106; I_REUNION = 114
I_CUPS = 137; I_FINAL = 138
ts = TN[0:I_SPLIT + 1]
hits1, pS, vS = plan(ts, (X0, 0, Z0), [vin0], lambda t: X0 + V * (t - T0), lambda t: Z0 - 0.02 * (t - T0), lambda t: 0.0)
for k, h in enumerate(hits1):
    i = k
    for nme, b in PAIR.items():
        ps = pitches(i)
        if len(ps) == 1: kind = 'crack' if i == I_CRACK else 'uni'; p = strike(i, nme, kind, 'hi')
        else: kind = 'key'; p = strike(i, nme, kind, 'hi' if nme == 'red' else 'lo')
        b.hit(h['t'], h['p'] + OFF[nme], h['vins'][0], h['vout'], kind, p, ev=i)
    t1 = hits1[k + 1]['t'] if k + 1 < len(hits1) else TN[I_SPLIT]
    for nme, b in PAIR.items(): b.fly(h['t'], t1, h['p'] + OFF[nme], h['vout'])
EV['crack'] = dict(t=float(TN[I_CRACK]), p=(hits1[I_CRACK]['p']).tolist())
SFX.append(['crack', float(TN[I_CRACK]) + 0.03])
p_split = {n: pS + OFF[n] for n in PAIR}
# the shared key of event 68 cracked: from 69 on two separate halves (each ball its own key)

# ============================================================ pass 2: red's floe sinks toward the camera, red -> sled
YR_LOW = -1.25
t69 = TN[I_SPLIT]
ts = TN[I_SPLIT:I_SLED + 1]
zr0 = p_split['red'][2]
hr, pR, vR = plan(ts, p_split['red'], [vS], lambda t: p_split['red'][0] + V * (t - t69),
                  lambda t: zr0 - 0.45 * (t - t69), lambda t: -DYP + (YR_LOW + DYP) * min(1, (t - t69) / (TN[I_SLED] - t69)) ** 0.8)
for k, h in enumerate(hr):
    i = I_SPLIT + k; ps = pitches(i)
    p = strike(i, 'red', 'key', 'hi')
    R_.hit(h['t'], h['p'], h['vins'][0], h['vout'], 'floe' if k else 'half', p, ev=i)
    t1 = hr[k + 1]['t'] if k + 1 < len(hr) else TN[I_SLED]
    R_.fly(h['t'], t1, h['p'], h['vout'])
# last floe key (77) -> drop into the sled seat
T_SEAT = TN[I_SLED] + 0.42
best = None
for dzs in np.linspace(-0.05, -1.5, 300):
    for vx in np.linspace(1.6, 0.6, 21):
        d = T_SEAT - TN[I_SLED]; vo = np.array([vx, 0.0, dzs / d + 0.5 * G * d])
        e, n = bounce_e(vR, vo)
        if EMIN <= e <= EMAX and n[2] >= math.cos(math.radians(TILT)): best = (vo, dzs); break
    if best: break
vo, dzs = best
p = strike(I_SLED, 'red', 'floe', 'hi')
R_.hit(TN[I_SLED], pR, vR, vo, 'floe', p, ev=I_SLED)
d = T_SEAT - TN[I_SLED]; seat = pR + vo * d + np.array([0, 0, -0.5 * G * d * d])
R_.fly(TN[I_SLED], T_SEAT, pR, vo)
v_seat_in = vo + np.array([0, 0, -G * d])
# sled: the ball's landing kicks it off (momentum share, sled mass = 2 x ball): v0 = vx_in / 3; slides down a chute with
# friction (incline slope THS, mu), then a flat run-out, then a snowbank stops it dead and the ball flies on.
MU = 0.06; THS = math.radians(9.0)
v_s0 = v_seat_in[0] / 3.0
a1 = G * (math.sin(THS) - MU * math.cos(THS)); a2 = -MU * G
T_STOP = TN[I_R_BACK] - 0.36                                  # ball leaves the sled 0.36 s before landing on key 88
T_INC = 2.6                                                   # time on the incline
tt = np.arange(T_SEAT, T_STOP + 1e-9, 1 / 240.0); sled_s = []; sled_p = []
L1 = v_s0 * T_INC + 0.5 * a1 * T_INC ** 2; v1 = v_s0 + a1 * T_INC
for t in tt:
    u = t - T_SEAT
    if u <= T_INC: s_ = v_s0 * u + 0.5 * a1 * u * u; dx = s_ * math.cos(THS); dz = -s_ * math.sin(THS)
    else:
        w = u - T_INC; s2 = v1 * w + 0.5 * a2 * w * w; dx = L1 * math.cos(THS) + s2; dz = -L1 * math.sin(THS)
    sled_p.append(seat + [dx, 0, dz])
sled_p = np.array(sled_p)
v_end = v1 + a2 * (T_STOP - T_SEAT - T_INC)
R_.path(tt, sled_p)
p_throw = sled_p[-1]; v_throw = np.array([v_end, 0.0, 0.0])
dfl = TN[I_R_BACK] - T_STOP
R_.fly(T_STOP, TN[I_R_BACK], p_throw, v_throw)
p88 = p_throw + v_throw * dfl + np.array([0, 0, -0.5 * G * dfl * dfl]); v88 = v_throw + np.array([0, 0, -G * dfl])
EV['sled'] = dict(t_seat=float(T_SEAT), seat=seat.tolist(), t_inc=T_INC, theta=THS, mu=MU, v0=v_s0, v_inc_end=v1, v_end=v_end,
                  t_stop=float(T_STOP), p_stop=p_throw.tolist(), L1=L1, a1=a1, a2=a2)
SFX.append(['sled_land', float(T_SEAT)]); SFX.append(['stop', float(T_STOP)])
for t in np.arange(T_SEAT + 0.08, T_STOP, Q): SFX.append(['jingle', float(t)])

# ============================================================ white: alone on the upper keys (78..87), then the door
ts = TN[I_SPLIT:I_W_DOOR + 1]
zw0 = p_split['white'][2]
hw, pW, vW = plan(ts, p_split['white'], [vS], lambda t: p_split['white'][0] + V * (t - t69), lambda t: zw0 - 0.03 * (t - t69), lambda t: DYP)
for k, h in enumerate(hw):
    i = I_SPLIT + k
    p = strike(i, 'white', 'half' if k == 0 else 'key', 'lo' if i <= I_SLED else 'hi')
    W_.hit(h['t'], h['p'], h['vins'][0], h['vout'], 'half' if k == 0 else 'key', p, ev=i)
    t1 = hw[k + 1]['t'] if k + 1 < len(hw) else TN[I_W_DOOR]
    W_.fly(h['t'], t1, h['p'], h['vout'])
# key 87 -> into the revolving door (lands on the turning floor, padded, silent)
T_DOOR_IN = TN[I_W_DOOR] + 0.4
RD = 0.5                                                      # radius of the ball's circle inside the door
best = None
for dzs in np.linspace(-0.05, -1.0, 200):
    for vx in np.linspace(1.4, 0.5, 19):
        d = T_DOOR_IN - TN[I_W_DOOR]; vo = np.array([vx, 0.0, dzs / d + 0.5 * G * d])
        e, n = bounce_e(vW, vo)
        if EMIN <= e <= EMAX and n[2] >= math.cos(math.radians(TILT)): best = (vo, dzs); break
    if best: break
vo, _ = best
p = strike(I_W_DOOR, 'white', 'key', 'hi')
W_.hit(TN[I_W_DOOR], pW, vW, vo, 'key', p, ev=I_W_DOOR)
d = T_DOOR_IN - TN[I_W_DOOR]; p_in = pW + vo * d + np.array([0, 0, -0.5 * G * d * d])
W_.fly(TN[I_W_DOOR], T_DOOR_IN, pW, vo)
# the door: axis at (p_in.x + RD, y = 0); the ball is carried round the back half (phi: pi -> 0) at constant speed
T_DOOR_OUT = T_DOOR_IN + 2.5
axis = np.array([p_in[0] + RD, p_in[1], p_in[2]])
OMEGA = math.pi / (T_DOOR_OUT - T_DOOR_IN)
tt = np.arange(T_DOOR_IN, T_DOOR_OUT + 1e-9, 1 / 240.0)
dp = []
for t in tt:
    ph = math.pi - OMEGA * (t - T_DOOR_IN)
    dp.append(axis + [RD * math.cos(ph), RD * math.sin(ph), 0])
W_.path(tt, np.array(dp))
EV['door'] = dict(axis=axis.tolist(), r=RD, t_in=float(T_DOOR_IN), t_out=float(T_DOOR_OUT), omega=OMEGA)
SFX.append(['door_in', float(T_DOOR_IN)]); SFX.append(['door_out', float(T_DOOR_OUT)])
# exit: the ball rolls off the door floor onto the slide head, a step 0.12 lower, and starts down the ICE SLIDE (along +x)
p_out = axis + [RD, 0, 0]
T_SL0 = T_DOOR_OUT + 0.18
d = T_SL0 - T_DOOR_OUT; v_exit = np.array([0.55, 0.0, 0.0])
W_.fly(T_DOOR_OUT, T_SL0, p_out, v_exit)
p_sl0 = p_out + v_exit * d + np.array([0, 0, -0.5 * G * d * d])
# slide: straight chute along +x, slope ALPHA; rolling: a = (5/7) g sin(a) - c_rr g cos(a)
T_SL1 = TN[I_W_BACK] - 0.34                                   # leaves the slide lip, flies onto key 106
SL_DROP_TARGET = 1.9
def slide_profile(alpha, crr):
    a = 5 / 7 * G * math.sin(alpha) - crr * G * math.cos(alpha)
    T = T_SL1 - T_SL0; v0 = 0.55
    L = v0 * T + 0.5 * a * T * T
    return a, L, v0 + a * T
ALPHA = math.radians(17.0); CRR = None
for crr in np.linspace(0.05, 0.3, 251):
    a, L, v1_ = slide_profile(ALPHA, crr)
    if L * math.sin(ALPHA) <= SL_DROP_TARGET: CRR = crr; break
A_SL, L_SL, V_SL1 = slide_profile(ALPHA, CRR)
ca, sa = math.cos(ALPHA), math.sin(ALPHA)
def slide_s(t): u = t - T_SL0; return 0.55 * u + 0.5 * A_SL * u * u
def slide_p(t): s_ = slide_s(t); return p_sl0 + np.array([s_ * ca, 0, -s_ * sa])
tt = np.arange(T_SL0, T_SL1 + 1e-9, 1 / 240.0)
W_.path(tt, np.array([slide_p(t) for t in tt]))
# ice chimes hang from the arch over the slide; the ball brushes each at the note time (white = lower voice / single)
CHIMES = []
for i in range(I_SLIDE0, I_SLIDE1 + 1):
    p = strike(i, 'white', 'chime', 'lo'); pc = slide_p(TN[i])
    CHIMES.append(dict(i=i, t=float(TN[i]), p=pc.tolist(), pitch=p))
    W_.hits.append(dict(t=float(TN[i]), p=pc.tolist(), vin=[0, 0, 0], vout=[0, 0, 0], n=[0, 0, 1], e=0.0, kind='chime', pitch=p, ev=i))
p_lip = slide_p(T_SL1); v_lip = np.array([V_SL1 * ca, 0, -V_SL1 * sa])
d = TN[I_W_BACK] - T_SL1
W_.fly(T_SL1, TN[I_W_BACK], p_lip, v_lip)
p106 = p_lip + v_lip * d + np.array([0, 0, -0.5 * G * d * d]); v106 = v_lip + np.array([0, 0, -G * d])
EV['slide'] = dict(t0=float(T_SL0), t1=float(T_SL1), p0=p_sl0.tolist(), alpha=ALPHA, crr=float(CRR), a=A_SL, L=L_SL, v1=V_SL1,
                   lip=p_lip.tolist(), chimes=CHIMES)
SFX.append(['slide_in', float(T_SL0)])

# ============================================================ red: lower keys 88..114 (alone until 96), white chases from 106
ts = TN[I_R_BACK:I_REUNION + 1]
zr88 = p88[2]
# red slows a little (waits for white); its landing on event 114 defines the reunion point
z_reu_t = p106[2] - 0.6
hr2, pR2, vR2 = plan(ts, p88, [v88], lambda t: p88[0] + 1.25 * (t - TN[I_R_BACK]),
                     lambda t: zr88 + (z_reu_t - zr88) * (t - TN[I_R_BACK]) / (TN[I_REUNION] - TN[I_R_BACK]),
                     lambda t: -DYP + (YR_LOW + DYP) * max(0.0, 1 - max(0, t - TN[I_W_BACK]) / (TN[I_REUNION] - TN[I_W_BACK])))
x_reu = float(pR2[0] - DXP); z_reu = float(pR2[2])
assert abs(pR2[1] + DYP) < 1e-6
for k, h in enumerate(hr2):
    i = I_R_BACK + k
    p = strike(i, 'red', 'key', 'hi')
    R_.hit(h['t'], h['p'], h['vins'][0], h['vout'], 'low', p, ev=i)
    t1 = hr2[k + 1]['t'] if k + 1 < len(hr2) else TN[I_REUNION]
    R_.fly(h['t'], t1, h['p'], h['vout'])
ts = TN[I_W_BACK:I_REUNION + 1]
hw2, pW2, vW2 = plan_retry(ts, p106, [v106], lambda t: p106[0] + (x_reu - DXP - p106[0]) * (t - TN[I_W_BACK]) / (TN[I_REUNION] - TN[I_W_BACK]),
                     lambda t: p106[2] + (z_reu - p106[2]) * (t - TN[I_W_BACK]) / (TN[I_REUNION] - TN[I_W_BACK]), lambda t: DYP,
                     final=(x_reu - DXP, z_reu), vxlim=(0.5, 4.2))
for k, h in enumerate(hw2):
    i = I_W_BACK + k
    p = strike(i, 'white', 'key', 'lo')
    W_.hit(h['t'], h['p'], h['vins'][0], h['vout'], 'low', p, ev=i)
    t1 = hw2[k + 1]['t'] if k + 1 < len(hw2) else TN[I_REUNION]
    W_.fly(h['t'], t1, h['p'], h['vout'])
EV['reunion'] = dict(t=float(TN[I_REUNION]), x=float(x_reu), z=float(z_reu))
SFX.append(['reunion', float(TN[I_REUNION])])

# ============================================================ together again: 114..137 (see-saw cups)
ts = TN[I_REUNION:I_CUPS + 1]
pc0 = np.array([x_reu, 0.0, z_reu])
hj, pJ, vJ = plan(ts, pc0, [vR2, vW2], lambda t: x_reu + V * (t - TN[I_REUNION]), lambda t: z_reu - 0.03 * (t - TN[I_REUNION]), lambda t: 0.0)
for k, h in enumerate(hj):
    i = I_REUNION + k; ps = pitches(i)
    for j, (nme, b) in enumerate(PAIR.items()):
        if len(ps) == 1: kind = 'reunion' if i == I_REUNION else 'uni'; p = strike(i, nme, kind, 'hi')
        else: kind = 'key'; p = strike(i, nme, kind, 'hi' if nme == 'red' else 'lo')
        b.hit(h['t'], h['p'] + OFF[nme], h['vins'][j if len(h['vins']) > 1 else 0], h['vout'], kind, p, ev=i)
        t1 = hj[k + 1]['t'] if k + 1 < len(hj) else TN[I_CUPS]
        b.fly(h['t'], t1, h['p'] + OFF[nme], h['vout'])

# ============================================================ finale: cups of the see-saw, the gold ball, the chandelier bell
cupc = pJ.copy()
T_LAUNCH = TN[I_CUPS] + 2 * Q
H_UP = 2.7                                                    # chandelier bell crown above the cup contact centres
T_FL = FINAL - T_LAUNCH
VL = (H_UP + 0.5 * G * T_FL ** 2) / T_FL
ARM = 0.85; TH0 = math.radians(14.0)
PLANK_TOP = 0.06                                              # cup floor above the plank's top surface
pivot = np.array([cupc[0], ARM * math.cos(TH0), cupc[2] - RP - PLANK_TOP + ARM * math.sin(TH0)])   # plank runs along +y; near end (cups) down
for nme, b in PAIR.items():
    pc = cupc + OFF[nme]
    p = strike(I_CUPS, nme, 'cup', 'hi')
    b.hit(TN[I_CUPS], pc, vJ, [0, 0, 0], 'cup', p, ev=I_CUPS)
    b.rest(TN[I_CUPS], T_LAUNCH, pc)
    rise = 2 * ARM * math.sin(TH0)
    b.fly(T_LAUNCH, FINAL, pc, [0, 0, VL])
    pb = pc + [0, 0, H_UP]; vb_in = np.array([0, 0, VL - G * T_FL])
    p = strike(I_FINAL, nme, 'bigbell', 'hi')
    vb_out = np.array([0, 0, -0.45 * vb_in[2]])
    b.hit(FINAL, pb, vb_in, vb_out, 'bigbell', p, ev=I_FINAL)
    # falls back into its cup, which the see-saw now holds 'rise' higher; silent padded landing, rests there
    zc = pc[2] + rise
    a_, b_, c_ = -0.5 * G, vb_out[2], pb[2] - zc
    tf = (-b_ - math.sqrt(b_ * b_ - 4 * a_ * c_)) / (2 * a_)
    b.fly(FINAL, FINAL + tf, pb, vb_out)
    b.rest(FINAL + tf, FINAL + 20, [pc[0], pc[1], zc])
    EV.setdefault('rest_t', []).append(float(FINAL + tf))
EV['seesaw'] = dict(cup=cupc.tolist(), pivot=pivot.tolist(), arm=ARM, th0=TH0, t_launch=float(T_LAUNCH), vl=VL, h_up=H_UP)
EV['bigbell'] = dict(x=float(cupc[0]), y=0.0, z=float(cupc[2] + H_UP + RP), t=float(FINAL))
SFX.append(['launch', float(T_LAUNCH)]); SFX.append(['land_cups', float(EV['rest_t'][0])])

# ============================================================ the orchestra (accompaniment)
# stage track = smoothed mean of the couple's landing points (keeps the orchestra in the shot)
allh = sorted([(h['t'], h['p'][0], h['p'][2]) for b in PAIR.values() for h in b.hits if h['kind'] not in ('chime', 'bigbell')])
HT = np.array([a[0] for a in allh]); HX = np.array([a[1] for a in allh]); HZ = np.array([a[2] for a in allh])
def smooth_track(t, w=2.0):
    m = np.abs(HT - t) < w
    if not m.any(): m = np.argsort(np.abs(HT - t))[:4]
    return float(HX[m].mean()), float(HZ[m].min())
ACC = {}
for role, r in (('bass', RB), ('ch2', RC), ('ch3', RC)):
    evs = [a for a in E['acc'] if a[1] == role]
    times = np.array([a[0] for a in evs])
    if role == 'bass':
        names = ['bass']; offs = [np.zeros(3)]
    else:
        names = [role + 'lo', role + 'hi']; offs = [np.array([-DXC, DYC, 0]), np.array([DXC, -DYC, 0])]
    xo = {'bass': 0.6, 'ch2': -0.2, 'ch3': 1.0}[role]
    zo = {'bass': 2.5, 'ch2': 3.85, 'ch3': 5.2}[role]
    y = LANE_Y[role]
    # first hit: dropped from a small chandelier
    x1, z1 = smooth_track(times[0]); p0 = np.array([x1 + xo, y, z1 + zo])
    tdrop = 0.9
    final_t = None
    ev_reg = evs if role == 'bass' else evs[:-1]              # chord couples: last event = final chord (cups)
    tr = np.array([a[0] for a in ev_reg])
    hz = lambda t: smooth_track(t)[1] + zo
    hx = lambda t: smooth_track(t)[0] + xo
    if role == 'bass':
        far_up = pivot + np.array([0, ARM * math.cos(TH0), ARM * math.sin(TH0) + PLANK_TOP + RB])
        def hz(t, f=hz):
            u = min(1, max(0, (t - (T_LAUNCH - 3.6)) / 3.0)); return (1 - u) * f(t) + u * (far_up[2] + 1.0)
        T33 = T_LAUNCH - 2 * Q
        def hx(t, f=hx):
            u = min(1, max(0, (t - (T_LAUNCH - 3.6)) / 3.0)); return (1 - u) * f(t) + u * (far_up[0] - 1.7 - 1.4 * (T33 - t))
    fin = (far_up[0] - 1.7, far_up[2] + 1.0) if role == 'bass' else None
    hits, pE, vE = plan_retry(tr, p0, [np.array([0, 0, -G * tdrop])], hx, hz, lambda t: y, emax=0.975, emin=0.3, final=fin,
                        vxlim=(0.4, 3.0), dzr=(-1.0, 0.6), R=r, ext=(0.26, 0.26, 0.2) if role == 'bass' else (0.16, 0.16, 0.17))
    balls = [Ball(n, r) for n in names]
    for b, o in zip(balls, offs):
        pst = p0 + o
        b.rest(0.0, tr[0] - tdrop, pst + [0, 0, 0.5 * G * tdrop ** 2]); b.fly(tr[0] - tdrop, tr[0], pst + [0, 0, 0.5 * G * tdrop ** 2], [0, 0, 0])
        for k, h in enumerate(hits):
            a = ev_reg[k]
            pch = a[2][0] if role == 'bass' else (a[2][0] if o[0] < 0 else a[2][1])
            b.hit(h['t'], h['p'] + o, h['vins'][0], h['vout'], 'bell' if role == 'bass' else 'tink', pch, ev=k, beat=a[5])
            t1 = hits[k + 1]['t'] if k + 1 < len(hits) else tr[-1]
            b.fly(h['t'], t1, h['p'] + o, h['vout'])
    ACC[role] = dict(balls=balls, pE=pE, vE=vE, t_last=tr[-1], evs=evs, offs=offs)
# bass: last regular bell (P2 bar 33 beat 1) -> lands on the see-saw's far end on beat 3 (the launch)
bb = ACC['bass']['balls'][0]; pE, vE = ACC['bass']['pE'], ACC['bass']['vE']
a = ACC['bass']['evs'][-1]
end_far = pivot + np.array([0, ARM * math.cos(TH0), ARM * math.sin(TH0) + PLANK_TOP + RB])      # far end is up before the launch
dt = T_LAUNCH - a[0]
vo = np.array([(end_far[0] - pE[0]) / dt, (end_far[1] - pE[1]) / dt, (end_far[2] - pE[2]) / dt + 0.5 * G * dt])
bb.hit(a[0], pE, vE, vo, 'bell', a[2][0], ev=len(ACC['bass']['evs']) - 1, beat=0)
bb.fly(a[0], T_LAUNCH, pE, vo)
vin_far = vo + np.array([0, 0, -G * dt])
bb.hits.append(dict(t=float(T_LAUNCH), p=end_far.tolist(), vin=vin_far.tolist(), vout=[0, 0, 0], n=[0, 0, 1], e=0.0, kind='seesaw', pitch=None, ev=None))
end_far_down = pivot + np.array([0, ARM * math.cos(TH0), -ARM * math.sin(TH0) + PLANK_TOP + RB])
tt = np.arange(T_LAUNCH, T_LAUNCH + 0.08, 1 / 240.0)
EV['seesaw']['t_swing'] = 0.07
bb.path(tt, np.array([end_far + (end_far_down - end_far) * min(1, (t - T_LAUNCH) / 0.07) for t in tt]))
bb.rest(T_LAUNCH + 0.08, FINAL + 20, end_far_down)
EV['seesaw']['far'] = end_far.tolist(); EV['seesaw']['vin_far'] = vin_far.tolist()
SFX.append(['seesaw', float(T_LAUNCH)])
# chord couples: last regular bell -> final bells (cups) on the final chord
FINB = []
for role in ('ch2', 'ch3'):
    A = ACC[role]; a = A['evs'][-1]; pE, vE = A['pE'], A['vE']; dt = FINAL - A['t_last']
    best = None
    for dzs in np.linspace(-0.05, -1.5, 300):
        for vx in np.linspace(max(0.4, vE[0]), 0.4, 15):
            vo = np.array([vx, 0.0, dzs / dt + 0.5 * G * dt]); e, n = bounce_e(vE, vo)
            if EMIN <= e <= EMAX and n[2] >= math.cos(math.radians(TILT)) and vo[2] - G * dt < -0.3: best = vo; break
        if best is not None: break
    vo = best
    for b, o in zip(A['balls'], A['offs']):
        prev = A['evs'][-2]
        pch = prev[2][0] if o[0] < 0 else prev[2][1]
        b.hit(A['t_last'], pE + o, vE, vo, 'tink', pch, ev=len(A['evs']) - 2, beat=prev[5])
        b.fly(A['t_last'], FINAL, pE + o, vo)
        pf = pE + o + vo * dt + np.array([0, 0, -0.5 * G * dt * dt]); vin = vo + np.array([0, 0, -G * dt])
        pch = a[2][0] if o[0] < 0 else a[2][1]
        b.hit(FINAL, pf, vin, [0, 0, 0], 'fincup', pch, ev=len(A['evs']) - 1, beat=0)
        b.rest(FINAL, FINAL + 20, pf)
        FINB.append(dict(ball=b.name, p=pf.tolist(), pitch=pch))
EV['final_bells'] = FINB

BALLS = dict(PAIR)
for role in ('bass', 'ch2', 'ch3'):
    for b in ACC[role]['balls']: BALLS[b.name] = b


# ============================================================ checks
def check():
    info = {}
    hs = [(n, h) for n, b in BALLS.items() for h in b.hits if h['kind'] not in ('chime', 'cup', 'fincup', 'seesaw')]
    ee = [h['e'] for n, h in hs if np.linalg.norm(h['vout']) > 0]
    info['e_range'] = (round(min(ee), 3), round(max(ee), 3))
    sp = [np.linalg.norm(h['vout']) / max(1e-6, np.linalg.norm(h['vin'])) for n, h in hs if np.linalg.norm(h['vout']) > 0]
    info['speed_out_over_in_max'] = round(max(sp), 3)
    til = [math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for n, h in hs if np.linalg.norm(h['vout']) > 0 and h['kind'] != 'bigbell']
    info['tilt_max_deg'] = round(max(til), 1)
    cov = set(STRUCK)
    info['lead_events_uncovered'] = sorted(set(range(NN)) - cov)
    for n, b in BALLS.items():
        zz = [h['p'][2] for h in b.hits]; xx = [h['p'][0] for h in b.hits]
        info[f'{n}'] = dict(hits=len(b.hits), z=(round(min(zz), 2), round(max(zz), 2)), x=(round(min(xx), 1), round(max(xx), 1)))
    info['sled'] = {k: round(v, 3) if isinstance(v, float) else v for k, v in EV['sled'].items() if k in ('v0', 'v_inc_end', 'v_end', 'L1', 'a1')}
    info['slide'] = {k: round(v, 3) for k, v in EV['slide'].items() if k in ('crr', 'a', 'L', 'v1')}
    info['launch_v'] = round(VL, 2)
    return info


if __name__ == '__main__':
    info = check()
    for k, v in info.items(): print(k, v)
    out = dict(G=G, R=dict(pair=RP, bass=RB, chord=RC), lane_y=LANE_Y, dxp=DXP, dyp=DYP, dxc=DXC, dyc=DYC,
               balls={n: dict(r=b.r, seg=b.seg, hits=b.hits) for n, b in BALLS.items()},
               struck={str(k): v for k, v in STRUCK.items()}, ev=EV, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
