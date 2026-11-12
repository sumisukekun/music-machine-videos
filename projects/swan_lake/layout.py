"""Swan Lake - 'White swan or black swan?'  Physics + story planner.  Run after arrange.py; writes layout.json.

Stage (lake surface z = 0, plan x/y, +y = towards the moon):
  the MOON RING: a big glowing ring standing in the lake at the origin (plane y = 0), turning on rollers, two gondolas A / B
  hang from it (opposite each other).  Above it the crescent cradle and two little harps.
Story (every melody note = a ball landing on a key of that pitch):
  bar 1    the crescent cradle rocks: the white ball W rolls out left and plays the rising harp, the cradle rocks back and the
           black ball K plays the falling harp; W dives down to the lake, K hides in its reed nest (soft landing, no note)
  A        the white swan W sings the theme alone on a descending spiral of white keys (front-left of the ring); its last key is a
           lever: it trips the nest gate -> K drops onto its first key
  A'       the black swan K copies the theme note for note on the mirrored spiral (right); W lands in gondola A at the bottom of the
           ring and is carried up.  K's last key trips gondola A's trapdoor at the top -> W drops into the B section
  B / B'   the rising phrase: W on a bigger spiral (left), K on its own (right) - each one's last key frees the other one at the
           top of the ring
  tutti    K's last key frees W at the top; K races in from the right underneath the falling W (near miss, 'sure-chigai') and
           both hit the first tutti note at the same instant; then they cross over: W draws the right half of a giant heart, K the
           left half, in unison on mirrored keys (light trails on the lake) and both land on the final bell at the heart's tip.
Physics: one constant gravity G; between hits pure parabolas; at a hit the key's surface normal is along (vout - vin)
(impulse along the normal, tangential velocity kept) with restitution e <= EMAX, so a bounce never adds speed.  Key heights
come from a dynamic programme over the whole chain of notes.  Lifts (gondolas) and drops from rest (gates, trapdoors) are
the only other motions."""
import json, os, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = 7.0; R = 0.15
EMAX = 0.96; EMIN = 0.3; EW_MAX = 3.2; W_COST = 6.0; TILT_MAX = 34.0; APEX_MAX = 2.6
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
TN = np.array([l[0] for l in LEAD]); NN = len(TN); SEC = [l[4] for l in LEAD]
IDX = {s: [i for i in range(NN) if SEC[i] == s] for s in ('intro', 'A', 'A2', 'B', 'B2', 'tutti', 'final')}
FINAL = E['final']

# ------------------------------------------------------------------ the moon ring and its gondolas
ZC = 3.45; RR = 2.6; HANG = 0.55
def gondola_ball(theta): return np.array([RR * math.cos(theta), 0.0, ZC + RR * math.sin(theta) - HANG])
BOT, TOP = -math.pi / 2, math.pi / 2


class Ball:
    def __init__(s, name): s.name = name; s.seg = []; s.hits = []
    def rest(s, t0, t1, p): s.seg.append(dict(k='rest', t0=float(t0), t1=float(t1), p=list(map(float, p))))
    def fly(s, t0, t1, p0, v0): s.seg.append(dict(k='fly', t0=float(t0), t1=float(t1), p=list(map(float, p0)), v=list(map(float, v0))))
    def ride(s, t0, t1, g): s.seg.append(dict(k='ride', t0=float(t0), t1=float(t1), g=g))


def bounce_e(vin, vout):
    d = vout - vin; n = d / max(1e-9, np.linalg.norm(d)); a = vin @ n
    return (-(vout @ n) / a if a < -1e-9 else 9.0), n


# ------------------------------------------------------------------ plan curves
def spline(pts, n=400):
    """Catmull-Rom through pts (open), resampled by arc length -> (S, XY)"""
    P = np.array(pts, float); P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for u in np.linspace(0, 1, 60, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    out.append(P[-2]); out = np.array(out)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(out, axis=0), axis=1))])
    return s, out
def at_s(S, XY, s): return np.array([np.interp(s, S, XY[:, 0]), np.interp(s, S, XY[:, 1])])


def heart_right(n=600):
    t = np.linspace(0, math.pi, n)
    x = 16 * np.sin(t) ** 3; y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return np.stack([x, y - 5], 1)
HR = heart_right(); _s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(HR, axis=0), axis=1))])
L_T = 33.0; SCH = L_T / _s[-1]; HEART = HR * SCH; S_H = _s * SCH          # cusp at the origin (= the ring), tip at (0, -22*SCH)
TIP = HEART[-1].copy()


# ------------------------------------------------------------------ the DP over key heights
def solve(P, T, z0, vin0, zlo, zhi, z_end=None, step=0.04, e_pref=0.82, name='', apex=APEX_MAX, emin=EMIN, last_hit=True,
          z_pref=None, w_z=0.0, allow_wing=True, lever_last=False):
    """P (n+1, 2) plan contact points, T (n+1) times; hits at 0..n-1 (n-1 is the last bounce), arrival at n.
    z0 fixed (contact centre height of hit 0), vin0 the velocity arriving at hit 0.  zlo/zhi arrays (n+1).
    If z_end is given the arrival height is fixed (a cup / gondola), else free in [zlo, zhi].
    Returns z (n+1) and per-hit (e, n, vin, vout)."""
    n = len(P) - 1
    lo = min(float(np.min(zlo)), z0, z_end if z_end is not None else 99); hi = max(float(np.max(zhi)), z0, z_end if z_end is not None else -99)
    Z = np.arange(lo - step, hi + 2 * step, step)
    def snap(z): return int(np.argmin(np.abs(Z - z)))
    Z[snap(z0)] = z0
    if z_end is not None: Z[snap(z_end)] = z_end
    M = len(Z); BIG = 1e9
    okz = [(Z >= zlo[k] - 1e-9) & (Z <= zhi[k] + 1e-9) for k in range(n + 1)]
    okz[0] = np.zeros(M, bool); okz[0][snap(z0)] = True
    if z_end is not None: okz[n] = np.zeros(M, bool); okz[n][snap(z_end)] = True
    hv = [(P[k + 1] - P[k]) / (T[k + 1] - T[k]) for k in range(n)]
    dts = [T[k + 1] - T[k] for k in range(n)]
    EMN = np.full(n, emin) if np.isscalar(emin) else np.array(emin)
    if lever_last: EMN[n - 1] = 0.05
    def hit_cost(vin, vo_h, vo_z, bz, k=0):
        """vin (...,3) and vout parts broadcastable; returns cost, e"""
        vo = np.stack(np.broadcast_arrays(vo_h[0], vo_h[1], vo_z), -1)
        vo, vi = np.broadcast_arrays(vo, np.asarray(vin, float)); vo_z = vo[..., 2]
        d = vo - vi; nn = np.linalg.norm(d, axis=-1, keepdims=True); nr = d / np.maximum(nn, 1e-9)
        a = (vi * nr).sum(-1); b = (vo * nr).sum(-1)
        e = np.where(a < -1e-9, -b / np.where(a < -1e-9, a, -1), 9.0)
        tilt = np.degrees(np.arccos(np.clip(nr[..., 2], -1, 1)))
        ap = np.where(vo_z > 0, vo_z ** 2 / (2 * G), 0)
        wing = e > EMAX
        c = 30 * (np.minimum(e, EMAX) - e_pref) ** 2 + 0.002 * tilt ** 2 + 0.15 * ap + np.where(wing, W_COST + 1.5 * (e - 1), 0)
        bad = (wing & ((e > EW_MAX) | (not allow_wing))) | (e < EMN[k]) | (tilt > TILT_MAX) | (ap > apex) | (vi[..., 2] > -0.5)
        return np.where(bad, BIG, c), e
    # step 0: hit 0 with fixed vin0 -> state (z0, z1)
    C = np.full((M, M), BIG)   # C[a=z_{k-1}, b=z_k]  (for k=1: a = z0)
    i0 = snap(z0); dt = dts[0]
    voz = (Z - z0) / dt + 0.5 * G * dt
    c0, _ = hit_cost(np.array(vin0), (hv[0][0], hv[0][1]), voz, z0)
    C[i0, :] = np.where(okz[1], c0, BIG)
    if C.min() >= BIG: raise RuntimeError(f'{name}: infeasible at hit 0')
    back = []
    for k in range(1, n):
        # hit k: a = z_{k-1}, b = z_k, c = z_{k+1}  (only the allowed heights of each note)
        Ia = np.nonzero(okz[k - 1])[0]; Ib = np.nonzero(okz[k])[0]; Ic = np.nonzero(okz[k + 1])[0]
        dtp, dt = dts[k - 1], dts[k]
        viz = (Z[Ib][None, :] - Z[Ia][:, None]) / dtp - 0.5 * G * dtp          # (a, b)
        vin = np.stack(np.broadcast_arrays(hv[k - 1][0], hv[k - 1][1], viz), -1)[:, :, None, :]
        voz = (Z[Ic][None, :] - Z[Ib][:, None]) / dt + 0.5 * G * dt            # (b, c)
        hc, _ = hit_cost(vin, (hv[k][0], hv[k][1]), voz[None, :, :], None, k)    # (a, b, c)
        if z_pref is not None: hc = hc + w_z * (Z[Ic][None, None, :] - z_pref[k + 1]) ** 2
        tot = C[np.ix_(Ia, Ib)][:, :, None] + hc
        arg = np.argmin(tot, axis=0)                                          # (b, c) -> index into Ia
        Cn = np.full((M, M), BIG); Cn[np.ix_(Ib, Ic)] = np.take_along_axis(tot, arg[None], 0)[0]
        full = np.zeros((M, M), int); full[np.ix_(Ib, Ic)] = Ia[arg]
        C = Cn; back.append(full)
        if C.min() >= BIG: raise RuntimeError(f'{name}: infeasible at hit {k} of {n} (t={T[k]:.2f}, dt {dtp:.3f} -> {dt:.3f})')
    b, c = np.unravel_index(np.argmin(C), C.shape)
    if C[b, c] >= BIG: raise RuntimeError(f'no feasible chain for {name}')
    zi = [c, b]
    for arg in reversed(back):
        a = arg[zi[-1], zi[-2]]; zi.append(a)
    zi = zi[::-1]
    z = np.array([Z[i] for i in zi]); z[0] = z0
    if z_end is not None: z[-1] = z_end
    # exact hits
    out = []; vin = np.array(vin0, float)
    for k in range(n):
        dt = dts[k]; vo = np.array([hv[k][0], hv[k][1], (z[k + 1] - z[k]) / dt + 0.5 * G * dt])
        e, nr = bounce_e(vin, vo); out.append((e, nr, vin.copy(), vo))
        vin = vo + np.array([0, 0, -G * dt])
    return z, out, vin


def run_chain(ball, idxs, P, T, z, hits, kinds, p_first=None):
    """write fly segments + hits for a solved chain; idxs = note index per hit (None = no note)"""
    for k in range(len(hits)):
        e, nr, vin, vo = hits[k]; p = np.array([P[k][0], P[k][1], z[k]])
        if idxs[k] is not None or kinds[k] != 'none':
            kd = ('wing' if kinds[k] == 'key' else 'winglever') if (e > EMAX and kinds[k] in ('key', 'lever')) else kinds[k]
            ball.hits.append(dict(i=idxs[k], t=float(T[k]), p=p.tolist(), n=nr.tolist(), e=float(e), kind=kd,
                                  vin=vin.tolist(), vout=vo.tolist()))
        ball.fly(T[k], T[k + 1], p, vo)
    return np.array([P[-1][0], P[-1][1], z[-1]])


B = {'W': Ball('W'), 'K': Ball('K')}
EV = {}; SFX = []
INTRO = IDX['intro']; A = IDX['A']; A2 = IDX['A2']; BB = IDX['B']; B2 = IDX['B2']; TU = IDX['tutti']; FI = IDX['final'][0]

# ------------------------------------------------------------------ gondola schedule (ring angle of gondola A; B = A + pi)
# requirements: A at the bottom when W lands after A (t_inW1), A at the top at K's lever (end A'), A at the bottom again at
# W's lever (end of B) [= B at the top: K released], A at the top at K's lever at the end of B' (W released for the tutti)
T_LEV_A = TN[A[-1]]; T_LEV_A2 = TN[A2[-1]]; T_LEV_B = TN[BB[-1]]; T_LEV_B2 = TN[B2[-1]]
HOP_IN = 0.55                                               # lever key -> landing in the gondola
KT = [T_LEV_A + HOP_IN, T_LEV_A2, T_LEV_B, T_LEV_B2]
DR = 0.22                                                   # release angles: W's B starts left of the top, K's B' right of it
KTH = [BOT, TOP + DR, BOT + 2 * math.pi - DR, TOP + 2 * math.pi]
w0 = math.pi / (KT[1] - KT[0])
TH_T = [0.0] + KT + [FINAL + 6]; TH_V = [KTH[0] - w0 * KT[0]] + KTH + [KTH[-1] + w0 * (FINAL + 6 - KT[-1])]
from scipy.interpolate import PchipInterpolator
THA = PchipInterpolator(TH_T, TH_V)
def theta(g, t): return float(THA(t)) + (0 if g == 'A' else math.pi)
def gpos(g, t): return gondola_ball(theta(g, t))

# ------------------------------------------------------------------ intro: crescent cradle + harps
ZCR = 8.7                                                   # ball centres in the cradle
CR_W = np.array([-0.32, 0.0, ZCR]); CR_K = np.array([0.32, 0.0, ZCR])
TF_W = 0.42; TF_K = 0.42                                     # drops from the cradle lips onto the first plates
def drop(p, tf): return p + np.array([0, 0, -0.5 * G * tf * tf]), np.array([0, 0, -G * tf])
# W: 6 rising harp plates (to the left), then the dive to the lake: first A key
A_START = np.array([-1.95, -2.45])
NEST = np.array([1.95, -2.45, 3.9])
LOOP_A = [A_START, (-3.0, -3.9), (-4.7, -4.6), (-6.2, -3.8), (-6.6, -2.3), (-5.7, -1.1), (-4.2, -0.8), (-2.9, -1.2),
          (-1.9, -1.5), (-1.2, -1.2), (-0.55, -0.55), (0.0, 0.0)]
SA, XYA = spline(LOOP_A)

def plan_on(S, XY, ts, t0, t1, s0=0.0, s1=None):
    s1 = S[-1] if s1 is None else s1
    return np.array([at_s(S, XY, s0 + (s1 - s0) * (t - t0) / (t1 - t0)) for t in ts])

def mirror(P): P = np.array(P, float).copy(); P[..., 0] *= -1; return P

b = B['W']
p0, vin0 = drop(CR_W, TF_W)
b.rest(0, TN[INTRO[0]] - TF_W, CR_W); b.fly(TN[INTRO[0]] - TF_W, TN[INTRO[0]], CR_W, [0, 0, 0])
EV['cradle'] = dict(W=CR_W.tolist(), K=CR_K.tolist(), tW=TN[INTRO[0]] - TF_W)
harpW = INTRO[:6]
HD = np.array([-0.55, -0.83]); Ph = [tuple(CR_W[:2] + 0.28 * k * HD) for k in range(6)]
T_IN1 = T_LEV_A + HOP_IN
ts = [TN[i] for i in A]
Pa = plan_on(SA, XYA, ts + [T_IN1], ts[0], T_IN1)
Pa[-1] = gpos('A', T_IN1)[:2]
P = np.vstack([Ph, Pa]); T = np.array([TN[i] for i in harpW] + ts + [T_IN1])
nh = len(harpW); n = len(P) - 1
zlo = np.array([p0[2] - 1.2] * nh + [0.6] * (n + 1 - nh)); zhi = np.array([p0[2] + 0.01] * nh + [4.2] * (n + 1 - nh))
zlo[-2] = 0.42; zhi[-2] = 0.56                              # the lever key just in front of the ring bottom
zpref = np.concatenate([[p0[2]] * nh, np.linspace(3.6, 1.0, n + 1 - nh)])
z, hits, vlast = solve(P, T, p0[2], vin0, zlo, zhi, z_end=gpos('A', T_IN1)[2], name='W intro+A', lever_last=True, z_pref=zpref, w_z=0.4)
kinds = ['plate'] * nh + ['key'] * (len(A) - 1) + ['lever']
run_chain(b, harpW + A, P, T, z, hits, kinds)
EV['lever_A'] = dict(t=T_LEV_A, p=[P[-2][0], P[-2][1], z[-2]])
SFX.append(['nest', T_IN1])
# W rides gondola A up to the top, released by K's lever at the end of A'
b.ride(T_IN1, T_LEV_A2, 'A')

# K: falling harp (to the right), then a dive into the reed nest (soft landing), waits, drops when W's lever trips the gate
k = B['K']; harpK = INTRO[6:]
TF_K = 0.42
T_RK = TN[harpK[0]] - TF_K
k.rest(0, T_RK, CR_K); k.fly(T_RK, TN[harpK[0]], CR_K, [0, 0, 0])
EV['cradle']['tK'] = T_RK
pk0, vk0 = drop(CR_K, TF_K)
Phk = [tuple(CR_K[:2] + 0.28 * j * HD * [-1, 1]) for j in range(6)]
_dz = NEST[2] - (pk0[2] - 0.35); T_NEST = TN[harpK[-1]] + (1.0 + math.sqrt(1 + 14 * -_dz)) / 7
P = np.vstack([Phk, [NEST[:2]]]); T = np.array([TN[i] for i in harpK] + [T_NEST])
zlo = np.array([pk0[2] - 1.4] * 6 + [NEST[2]]); zhi = np.array([pk0[2] + 0.01] * 6 + [NEST[2]])
z, hits, vl = solve(P, T, pk0[2], vk0, zlo, zhi, z_end=NEST[2], name='K harp', apex=1.2, allow_wing=False)
run_chain(k, harpK, P, T, z, hits, ['plate'] * 6)
k.rest(T_NEST, T_LEV_A, NEST)
SFX.append(['nest', T_NEST])
EV['nest'] = dict(p=NEST.tolist(), t_land=T_NEST, t_open=T_LEV_A)
# A': gate opens at W's lever, K drops onto its first key (directly below), mirrored spiral, lever, gondola B
TFG = TN[A2[0]] - T_LEV_A
pk, vk = drop(NEST, TFG)
k.fly(T_LEV_A, TN[A2[0]], NEST, [0, 0, 0]); SFX.append(['gate', T_LEV_A])
LOOP_A2 = [tuple(q) for q in mirror(LOOP_A)]                    # the black swan copies the white one's spiral (mirrored)
SA2, XYA2 = spline(LOOP_A2)
T_IN2 = T_LEV_A2 + HOP_IN
ts = [TN[i] for i in A2]
P = plan_on(SA2, XYA2, ts + [T_IN2], ts[0], T_IN2); P[0] = NEST[:2]; P[-1] = gpos('B', T_IN2)[:2]
T = np.array(ts + [T_IN2]); n = len(P) - 1
zlo = np.full(n + 1, 0.6); zhi = np.full(n + 1, pk[2]); zlo[-2] = 0.42; zhi[-2] = 0.56
z, hits, vl = solve(P, T, pk[2], vk, zlo, zhi, z_end=gpos('B', T_IN2)[2], name="K A'", lever_last=True, z_pref=np.linspace(pk[2], 1.0, n + 1), w_z=0.4)
run_chain(k, A2, P, T, z, hits, ['key'] * (len(A2) - 1) + ['lever'])
EV['lever_A2'] = dict(t=T_LEV_A2, p=[P[-2][0], P[-2][1], z[-2]])
k.ride(T_IN2, T_LEV_B, 'B'); SFX.append(['nest', T_IN2])

# ------------------------------------------------------------------ B (W) and B' (K): bigger spirals, released at the ring top
def released(ball, g, t_rel, t_first):
    p = gpos(g, t_rel); tf = t_first - t_rel
    ball.fly(t_rel, t_first, p, [0, 0, 0]); SFX.append(['latch', t_rel])
    return p + np.array([0, 0, -0.5 * G * tf * tf]), np.array([0, 0, -G * tf])
LOOP_B = [(-0.57, 0.0), (-1.2, -1.4), (-2.2, -3.4), (-3.6, -5.6), (-5.6, -6.5), (-7.3, -5.3), (-7.7, -3.0), (-6.9, -0.9),
          (-5.3, 0.4), (-3.5, 1.2), (-2.0, 1.35), (-0.95, 0.95), (0.0, 0.0)]
SB, XYB = spline(LOOP_B)
pw, vw = released(B['W'], 'A', T_LEV_A2, TN[BB[0]])
T_IN3 = T_LEV_B + HOP_IN
ts = [TN[i] for i in BB]
P = plan_on(SB, XYB, ts + [T_IN3], ts[0], T_IN3); P[0] = pw[:2]; P[-1] = gpos('A', T_IN3)[:2]
T = np.array(ts + [T_IN3]); n = len(P) - 1
zlo = np.full(n + 1, 0.6); zhi = np.full(n + 1, pw[2]); zlo[-2] = 0.42; zhi[-2] = 0.56
z, hits, vl = solve(P, T, pw[2], vw, zlo, zhi, z_end=gpos('A', T_IN3)[2], name='W B', lever_last=True, z_pref=np.linspace(pw[2], 1.0, n + 1), w_z=0.3)
run_chain(B['W'], BB, P, T, z, hits, ['key'] * (len(BB) - 1) + ['lever'])
EV['lever_B'] = dict(t=T_LEV_B, p=[P[-2][0], P[-2][1], z[-2]])
B['W'].ride(T_IN3, T_LEV_B2, 'A'); SFX.append(['nest', T_IN3])
# B' (K): from the top of the ring, its spiral on the right, ending high on the approach to the ring (the near miss)
pk, vk = released(B['K'], 'B', T_LEV_B, TN[B2[0]])
Z_T0 = gpos('A', T_LEV_B2)[2] - 0.5 * G * (TN[TU[0]] - T_LEV_B2) ** 2     # height of W's first tutti key (W drops onto it)
K_T0 = np.array([-0.58, 0.22])                                         # K's first tutti key (left of the ring centre)
K_LAST = np.array([0.30, -0.70])                                        # K's last B' key (the lever)
LOOP_B2 = [(0.57, 0.0), (1.6, 0.6), (3.4, 0.9), (5.4, 0.4), (7.0, -0.8), (7.8, -2.8), (7.3, -5.0), (5.5, -6.3),
           (3.7, -5.4), (3.5, -3.6), (2.7, -2.0), tuple(K_LAST)]
SB2, XYB2 = spline(LOOP_B2)
ts = [TN[i] for i in B2]
P = plan_on(SB2, XYB2, ts, ts[0], ts[-1]); P[0] = pk[:2]
P = np.vstack([P, [K_T0]]); T = np.array(ts + [TN[TU[0]]]); n = len(P) - 1
zlo = np.full(n + 1, 2.2); zhi = np.full(n + 1, pk[2]); zlo[-1] = zhi[-1] = Z_T0; zlo[-2] = Z_T0 - 0.08
z, hits, vl = solve(P, T, pk[2], vk, zlo, zhi, z_end=Z_T0, name="K B'", lever_last=True, z_pref=np.concatenate([np.linspace(pk[2], 3.0, n - 4), [3.6, 4.2, 4.8, 5.0, Z_T0]]), w_z=0.15)
run_chain(B['K'], B2, P, T, z, hits, ['key'] * (len(B2) - 1) + ['lever'])
EV['lever_B2'] = dict(t=T_LEV_B2, p=[P[-2][0], P[-2][1], z[-2]])
vK_t0 = vl; pK_t0 = np.array([K_T0[0], K_T0[1], Z_T0])

# ------------------------------------------------------------------ tutti: W released from the top, both draw the heart
pw, vw = released(B['W'], 'A', T_LEV_B2, TN[TU[0]])
EV['nearmiss'] = dict(t0=T_LEV_B2, t1=TN[TU[0]])
ts = [TN[i] for i in TU] + [FINAL]
# W: right half of the heart (from the cusp at the ring), final bell cup at the tip
BELL_Z = 1.25
PW = plan_on(S_H, HEART, ts, ts[0], ts[-1], s0=0.0, s1=S_H[-1])
PW[0] = pw[:2]
for j in range(1, n_tu := len(PW)):
    d = np.linalg.norm(PW[j] - TIP); PW[j, 0] = PW[j, 0] + 0.45 * math.exp(-d / 1.5)
PW[-1] = TIP + np.array([0.4, 0.0])
T = np.array(ts); n = len(PW) - 1
zlo = np.full(n + 1, 0.8); zhi = np.full(n + 1, pw[2] + 0.4); zlo[-1] = zhi[-1] = BELL_Z
zp = np.linspace(pw[2], BELL_Z + 0.6, n + 1)
z, hits, vl = solve(PW, T, pw[2], vw, zlo, zhi, z_end=BELL_Z, name='W tutti', z_pref=zp, w_z=0.25)
run_chain(B['W'], TU, PW, T, z, hits, ['key'] * len(TU))
pend = np.array([PW[-1][0], PW[-1][1], BELL_Z])
B['W'].hits.append(dict(i=FI, t=float(FINAL), p=pend.tolist(), n=[0, 0, 1], e=0.0, kind='bell', vin=vl.tolist(), vout=[0, 0, 0]))
B['W'].rest(FINAL, FINAL + 8, pend)
zW = z.copy()
# K: left half (mirror of W's plan from the 2nd tutti note on)
PK = mirror(PW); PK[0] = K_T0
z, hits, vl = solve(PK, T, Z_T0, vK_t0, zlo, zhi, z_end=BELL_Z, name='K tutti', z_pref=zW, w_z=2.0)
run_chain(B['K'], TU, PK, T, z, hits, ['key'] * len(TU))
pend = np.array([PK[-1][0], PK[-1][1], BELL_Z])
B['K'].hits.append(dict(i=FI, t=float(FINAL), p=pend.tolist(), n=[0, 0, 1], e=0.0, kind='bell', vin=vl.tolist(), vout=[0, 0, 0]))
B['K'].rest(FINAL, FINAL + 8, pend)
EV['bell'] = dict(p=[TIP[0], TIP[1], BELL_Z], t=FINAL)
EV['heart'] = HEART.tolist()
EV['ring'] = dict(zc=ZC, r=RR, hang=HANG, t=list(np.round(np.arange(0, FINAL + 6, 1 / 240), 5)),
                  th=[round(float(THA(t)), 6) for t in np.arange(0, FINAL + 6, 1 / 240)])
EV['loops'] = {'A': XYA[::6].tolist(), 'A2': XYA2[::6].tolist(), 'B': XYB[::6].tolist(), 'B2': XYB2[::6].tolist()}
for kk, tt in (('cradle_W', EV['cradle']['tW']), ('cradle_K', EV['cradle']['tK'])): SFX.append(['tip', tt - 0.1])
SFX.append(['chime', FINAL])


# ------------------------------------------------------------------ positions + checks
def pos(ball, t):
    for g in ball.seg:
        if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9:
            if g['k'] == 'rest': return np.array(g['p'])
            if g['k'] == 'ride': return gpos(g['g'], t)
            u = t - g['t0']; return np.array(g['p']) + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
    return pos(ball, ball.seg[-1]['t1']) if t > ball.seg[-1]['t1'] else pos(ball, ball.seg[0]['t0'])


def check():
    info = {}
    hs = [(n, h) for n, b in B.items() for h in b.hits]
    ee = [h['e'] for n, h in hs if h['kind'] not in ('bell', 'wing', 'winglever')]
    info['wings'] = sum(1 for n, h in hs if h['kind'] in ('wing', 'winglever')); info['wing_e_max'] = round(max([h['e'] for n, h in hs if h['kind'] == 'wing'] or [0]), 2)
    info['e_range'] = (round(min(ee), 3), round(max(ee), 3))
    ee2 = [h['e'] for n, h in hs if h['kind'] not in ('bell', 'wing', 'lever', 'winglever')]; info['e_range_keys'] = (round(min(ee2), 3), round(max(ee2), 3))
    info['wing_notes'] = {n: [h['i'] for h in b.hits if h['kind'] == 'wing'] for n, b in B.items()}
    sp = [np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for n, h in hs if h['kind'] not in ('bell', 'wing', 'winglever')]
    info['speed_out_over_in_max'] = round(max(sp), 3)
    info['tilt_max_deg'] = round(max(math.degrees(math.acos(h['n'][2])) for n, h in hs), 1)
    cov = {}
    for n, h in hs:
        if h['i'] is not None: cov.setdefault(h['i'], []).append(n)
    info['notes_uncovered'] = [i for i in range(NN) if i not in cov]
    info['tutti_both'] = all(sorted(cov[i]) == ['K', 'W'] for i in TU + [FI])
    # segment continuity
    worst = 0
    for n, b in B.items():
        for g0, g1 in zip(b.seg, b.seg[1:]):
            assert abs(g0['t1'] - g1['t0']) < 1e-6, (n, g0['t1'], g1['t0'])
            worst = max(worst, np.linalg.norm(pos(b, g0['t1'] - 1e-7) - pos(b, g1['t0'] + 1e-7)))
    info['seg_jump_max'] = round(float(worst), 4)
    ts = np.arange(0, FINAL + 1, 1 / 240)
    d = [np.linalg.norm(pos(B['W'], t) - pos(B['K'], t)) for t in ts]
    j = int(np.argmin(d)); info['min_ball_dist'] = (round(float(d[j]), 3), round(float(ts[j]), 2))
    jj = [k for k, t in enumerate(ts) if EV['nearmiss']['t0'] <= t <= EV['nearmiss']['t1']]
    info['nearmiss_gap_surface'] = round(float(min(d[k] for k in jj)) - 2 * R, 3)
    zz = {n: [h['p'][2] for h in b.hits] for n, b in B.items()}
    info['key_z_range'] = {n: (round(min(v), 2), round(max(v), 2)) for n, v in zz.items()}
    return info


if __name__ == '__main__':
    info = check()
    for kk, v in info.items(): print(kk, v)
    out = dict(G=G, R=R, ring=dict(zc=ZC, r=RR, hang=HANG), balls={n: dict(seg=b.seg, hits=b.hits) for n, b in B.items()}, ev=EV, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
