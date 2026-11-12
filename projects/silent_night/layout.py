"""Silent Night - 'Can marbles light up the whole village?'  Physics + story planner.
Run after arrange.py; writes layout.json (read by scene.py, render_audio.py, verify.py).

Story (side view of a snowy hillside village on Christmas Eve; the course runs along +x and slowly steps down the hill):
  0-1.7 s   the village is dark. One lantern ball hangs in a lantern on a post; the counter says LIGHTS 0 / 1818. Its trapdoor
            opens and the ball drops onto the first key: a window lights up (every melody strike lights one window).
  bars 1-4  it plays the tune alone, lighting window after window
  bar 4     its long G lands on a LEVER key: the rod opens the gate of the snowball's roof, the snowball slides down the snowy
            chute and plays the bass from bar 5 (every bass strike lights a street lamp)
  bar 8     the snowball lands on the bell at the end of a SLED see-saw: the far end throws the two ornaments out of the sled's
            basket; they ring the inner voices from bar 9 on (every strike lights the star of a little fir tree)
  13-15     verse 2: five doors open one after another, five children roll down their doorsteps and land in the lanes behind the
            lantern ball, each exactly on a note, and sing the tune along (now every note lights up to six windows at once)
  bar 24    the final chord: the snowball lands on the great church bell; the bell's light runs through the whole valley
            (every window still dark lights up, nearest first: 1818 lights for 1818, the year the carol was first sung) and
            the sky fills with stars
Physics: one constant gravity G; between contacts pure parabolas; every bounce is an impulse along the surface normal
(tangential velocity kept) with restitution emin <= e <= emax, so a bounce never adds speed. Ramps: rolling from rest,
a = (5/7) g sin(alpha). The see-saw: the snowball gives up more energy (low e) than the ornaments receive."""
import json, os, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = 7.0
RS, RB, RP = 0.2, 0.3, 0.15            # lantern ball / children, snowball, ornaments
LANE = 0.5                                 # melody lanes: lane j at y = j * LANE (j = 0 soloist)
Y_BASS, Y_PAIR = -1.5, -2.6
DXC = 0.2                                  # pair: half spacing along x
EMAX, EMIN, TILT = 0.97, 0.22, 34.0
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
TN = np.array([l[0] for l in LEAD]); NN = len(TN)
T0, EI, FINAL = E['t0'], E['eighth'], E['final']
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
         land_min=0.3, final=None, dzr=(-2.0, 1.0), R=RS, ext=(0.23, 0.22, 0.07), hist0=(), eover=None, avoid=None, arad=0.0):
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
            if avoid is not None and len(tau_out):
                A_ = avoid(k, p, tk + tau_out)
                if A_ is not None: bad |= (np.linalg.norm(P_out - A_[None], axis=2) < arad).any(1)
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
NTOTAL = 1818                               # lights in the village (the year the carol was first sung)
# ============================================================ the lantern ball: lantern drop, then the whole tune (both verses)
SOLO = Ball('solo', RS, 'melody'); BALLS['solo'] = SOLO
X0, Z0 = 0.0, 0.0
T_DROP = 1.0
H_DROP = 0.5 * G * T_DROP ** 2
SOLO.rest(0.0, T0 - T_DROP, (X0, 0, Z0 + H_DROP)); SOLO.fly(T0 - T_DROP, T0, (X0, 0, Z0 + H_DROP), (0, 0, 0))
EV['lantern'] = dict(x=X0, z=Z0 + H_DROP, t_release=T0 - T_DROP)
SFX.append(['release', T0 - T_DROP])
I_GATE = ev_idx(4, 3)                       # the long G of bar 4: the lever key
J_CAND = [[(13, 0)], [(13, 3)], [(14, 0)], [(14, 3)], [(15, 0), (15, 2), (15, 3), (16, 0)]]     # first feasible note of each list
V1 = 1.25
def xt_mel(t): return X0 + V1 * (t - T0)
def zt_mel(t): return Z0 - 0.045 * (t - T0)
MH = []
I_LIFT = ev_idx(12, 0)                      # the long C that ends verse 1: the lantern ball lands in the lift's bucket
import pickle, hashlib
_ck = hashlib.md5((json.dumps(LEAD) + repr((G, RS, EMAX, EMIN, TILT, V1, T_DROP, X0, Z0, 'zt0.045', 'lift', 22.0, 2.0, 0.45, 1.5))).encode()).hexdigest()
# ---- melody plan cache key end
def cached(tag, fn):
    cf = f'/tmp/sn_{tag}_{_ck}.pkl'
    if os.path.exists(cf): return pickle.load(open(cf, 'rb'))
    r = fn(); pickle.dump(r, open(cf, 'wb')); return r
def run_a():
    hs, p_, v_, hist = plan_retry(TN[:I_LIFT + 1], np.array([X0, 0, Z0]), [np.array([0, 0, -G * T_DROP])], xt_mel, zt_mel,
                                  lambda t: 0.0, R=RS)
    return [dict(t=h['t'], p=np.array(h['p']), vin=np.array(h['vins'][0]), vout=np.array(h['vout'])) for h in hs], p_, v_
MHA, P_L, V_L = cached('a', run_a)
# ---- the lift: dead landing in the bucket (the bucket floor is the C key), the lift rises, its gate opens, the ball rolls down
# the chute (rolling from rest, a = 5/7 g sin a) and flies onto the first note of verse 2
T12, T13 = float(TN[I_LIFT]), float(TN[I_LIFT + 1])
ALPHA_L = math.radians(22.0); S_EX = 2.0; T_FL = 0.45
aL = 5 / 7 * G * math.sin(ALPHA_L); T_RL = S_EX / aL; L_L = S_EX ** 2 / (2 * aL)
uL = np.array([math.cos(ALPHA_L), 0, -math.sin(ALPHA_L)]); vexL = uL * S_EX
P13 = np.array([0.0, 0.0, zt_mel(T13) + 0.1]); P13[0] = 0.0
lipL_rel = - vexL * T_FL + np.array([0, 0, 0.5 * G * T_FL ** 2])      # lip relative to the landing point
perch_rel = lipL_rel - uL * L_L
P13[0] = P_L[0] - perch_rel[0]                                         # the perch is right above the bucket
P13[2] = P_L[2] - perch_rel[2] + 1.45 - 1.45                            # placeholder, set below
RISE_L = 1.5                                                           # the lift raises the ball by 1.5 m
P13[2] = P_L[2] + RISE_L - perch_rel[2]
lipL = P13 + lipL_rel; perchL = P13 + perch_rel
T_RELL = T13 - T_FL - T_RL; T_UP0 = T12 + 0.2; T_UP1 = T_RELL - 0.12
assert T_UP1 - T_UP0 > 0.5, (T_UP0, T_UP1)
V13 = vexL + np.array([0, 0, -G * T_FL])
EV['lift'] = dict(t_land=T12, t_up0=T_UP0, t_up1=T_UP1, t_rel=T_RELL, bottom=P_L.tolist(), perch=perchL.tolist(), lip=lipL.tolist(),
                  alpha=ALPHA_L, L=L_L, T_ramp=T_RL, rise=RISE_L)
SFX.append(['lift', T_UP0]); SFX.append(['gate_small', T_RELL])
def run_b():
    hs, p_, v_, hist = plan_retry(TN[I_LIFT + 1:], P13, [V13], xt_mel, zt_mel, lambda t: 0.0, R=RS)
    return [dict(t=h['t'], p=np.array(h['p']), vin=np.array(h['vins'][0]), vout=np.array(h['vout'])) for h in hs], p_, v_
MHB, P_FIN, V_FIN = cached('b', run_b)
MH = MHA + [dict(t=T12, p=P_L, vin=V_L, vout=np.zeros(3))] + MHB
print('final contact', np.round(P_FIN, 3), 'vin', np.round(V_FIN, 3))
assert len(MH) == NN - 1
MH.append(dict(t=FINAL, p=P_FIN, vin=V_FIN, vout=np.zeros(3)))          # dead landing on the last key

# ============================================================ the children: doorstep -> door opens -> ramp -> into lane j on a note
ALPHA_J = math.radians(22.0)
def join_solve(i):
    """incoming velocity for a child landing on note i in its lane (same bounce as the lantern ball: the same vout), coming down
    a ramp of angle alpha rolling from rest. Search restitution and ramp angle for a short ramp (about 1 m)."""
    h = MH[i]; vout = h['vout']; n = bounce_e(h['vin'], vout)[1]
    best = None
    for ej in np.linspace(0.35, 0.85, 21):
        lam = (vout @ n) * (1 + 1 / ej); vin = vout - lam * n
        vx = vin[0]
        if vx < 0.25: continue
        for al in np.radians(np.arange(16, 37, 2.0)):
            vex = np.array([vx, 0, -vx * math.tan(al)])
            T = (vex[2] - vin[2]) / G
            if T < 0.25: continue
            L = (vex @ vex) / (2 * 5 / 7 * G * math.sin(al))
            c = abs(L - 1.0) + 0.3 * abs(ej - 0.55)
            if best is None or c < best[0]: best = (c, ej, vin, vex, T, al)
    return None if best is None or not (0.4 < best[0] + 1.0 - 0.0 < 99) else best[1:]
JOIN = []
J_IDX = []
for cands in J_CAND:
    for bp in cands:
        if join_solve(ev_idx(*bp)): J_IDX.append(ev_idx(*bp)); break
    else: raise RuntimeError(('no joiner note', cands))
LANES_AT = [1 + sum(1 for j in J_IDX if j <= i) for i in range(NN)]
def lift_path(b):
    tt = np.arange(T_UP0, T_UP1 + 1e-9, 1 / 240.0)
    u = (tt - T_UP0) / (T_UP1 - T_UP0); u = u * u * (3 - 2 * u)
    b.rest(T12, T_UP0, P_L)
    b.path(tt, [P_L + np.array([0, 0, RISE_L * x]) for x in u])
    b.rest(T_UP1, T_RELL, perchL)
    tt = np.arange(T_RELL, T_RELL + T_RL + 1e-9, 1 / 240.0)
    b.path(tt, [perchL + uL * (0.5 * aL * (t - T_RELL) ** 2) for t in tt])
    b.fly(T_RELL + T_RL, T13, lipL, vexL)
for i, h in enumerate(MH):
    kind = 'lever' if i == I_GATE else ('final' if i == NN - 1 else ('lift' if i == I_LIFT else 'key'))
    SOLO.hit(h['t'], h['p'], h['vin'], h['vout'], kind, LEAD[i][1], ev=i, lanes=LANES_AT[i])
    if i == I_LIFT: lift_path(SOLO)
    elif i + 1 < NN: SOLO.fly(h['t'], MH[i + 1]['t'], h['p'], h['vout'])
SOLO.rest(FINAL, FINAL + 30, P_FIN)
EV['lever'] = dict(t=float(TN[I_GATE]), p=MH[I_GATE]['p'].tolist(), n=SOLO.hits[I_GATE]['n'])
for j, i in enumerate(J_IDX, start=1):
    h = MH[i]; tj = h['t']
    pc = h['p'] + np.array([0, j * LANE, 0])
    best = join_solve(i)
    ej, vin, vex, T, ALPHA_J = best
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
    SFX.append(['door', float(t_rel) - 0.25])
EV['join'] = JOIN

MX = np.array([h['p'][0] for h in MH]); MZ = np.array([h['p'][2] for h in MH]); MT = np.array([h['t'] for h in MH])
def track(t, w=1.8):
    m = np.abs(MT - t) < w
    if not m.any(): m = np.argsort(np.abs(MT - t))[:4]
    return float(MX[m].mean()), float(MZ[m].min())

# ============================================================ the snowball: roof -> gate -> snowy chute -> bass bells
BE = E['bass']; TBASS = np.array([b[0] for b in BE])
XO_B, ZO_B = 0.35, -1.15
I_SEESAW = [k for k, b in enumerate(BE) if b[2] == 8 and b[3] == 3][0]
x1, z1 = track(TBASS[0]); P_B0 = np.array([x1 + XO_B, Y_BASS, z1 + ZO_B])
ALPHA_B = math.radians(24.0)
T_GATE = float(TN[I_GATE]) + 0.06
T_FB = 0.36
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
# the see-saw geometry (needed while planning the snowball: its bounce off the see-saw bell must not cross the ornaments' flight)
ARM = 0.85; TH0 = math.radians(25.0); PHI = math.radians(25.0); CUP_H = 0.12 + RP; BELL_H = 0.36; S_L = 4.6
U = np.array([math.cos(PHI), -math.sin(PHI), 0.0]); ZZ = np.array([0, 0, 1.0])
AXP = np.array([math.sin(PHI), math.cos(PHI), 0.0])
OFFP = [-DXC * AXP, DXC * AXP]
def seesaw_from(pS, nS, T_SS):
    topS = pS - RB * nS; pivot = topS - ZZ * BELL_H - ARM * (U * math.cos(TH0) + ZZ * math.sin(TH0))
    def plank_d(al): return U * math.cos(al) + ZZ * math.sin(al)
    def plank_n(al): return -U * math.sin(al) + ZZ * math.cos(al)
    def cup_c(al): return pivot - ARM * plank_d(al) + CUP_H * plank_n(al)
    T_SWING = 4 * ARM * TH0 / S_L; T_L = T_SS + T_SWING
    def al_t(t): u = min(1.0, max(0.0, (t - T_SS) / T_SWING)); return TH0 - 2 * TH0 * u * u
    C_L = cup_c(-TH0); eps = 1e-5; vL = (cup_c(al_t(T_L)) - cup_c(al_t(T_L - eps))) / eps
    return dict(pivot=pivot, cup_c=cup_c, al_t=al_t, T_SWING=T_SWING, T_L=T_L, C_L=C_L, vL=vL, plank_d=plank_d, plank_n=plank_n)
def avoid_bass(k, p, ts):
    if k != I_SEESAW: return None
    S_ = seesaw_from(p, np.array([0, 0, 1.0]), TBASS[k])
    u = np.clip(ts - S_['T_L'], 0, None)[:, None]
    P_ = S_['C_L'] + S_['vL'] * u + np.array([0, 0, -0.5 * G]) * u ** 2
    P_[ts < S_['T_L']] = S_['cup_c'](TH0)
    return P_
def hx_b(t): return track(t)[0] + XO_B
def hz_b(t): return track(t)[1] + ZO_B
hb, pBE, vBE, _ = cached('bass2' + str(J_IDX), lambda: plan_retry(TBASS, P_B0, [vinB0], hx_b, hz_b, lambda t: Y_BASS, emax=0.975, emin=0.25,
                             vxlim=(0.4, 3.0), dzr=(-1.0, 0.6), R=RB, ext=(0.26, 0.26, 0.2), eover={I_SEESAW: (0.3, 0.68)},
                             avoid=avoid_bass, arad=RB + RP + DXC + 0.2))
for k, h in enumerate(hb):
    kind = 'seesaw' if k == I_SEESAW else 'bell'
    BASS.hit(h['t'], h['p'], h['vins'][0], h['vout'], kind, BE[k][1], ev=k)
    t1 = hb[k + 1]['t'] if k + 1 < len(hb) else TBASS[-1]
    BASS.fly(h['t'], t1, h['p'], h['vout'])
BASS.hit(TBASS[-1], pBE, vBE, np.zeros(3), 'churchbell', BE[-1][1], ev=len(BE) - 1, final=True)
BASS.rest(TBASS[-1], FINAL + 30, pBE)
hS = BASS.hits[I_SEESAW]
KE_loss = 0.5 * (np.linalg.norm(hS['vin']) ** 2 - np.linalg.norm(hS['vout']) ** 2)

# ============================================================ the sled see-saw and the two ornaments
PE = E['pair']; TPAIR = np.array([p[0] for p in PE])
T_SS = float(hS['t']); SS = seesaw_from(np.array(hS['p']), np.array(hS['n']), T_SS)
pivot, cup_c, al_t, T_SWING, T_L, C_L, vL = SS['pivot'], SS['cup_c'], SS['al_t'], SS['T_SWING'], SS['T_L'], SS['C_L'], SS['vL']
CUP = cup_c(TH0)
dtl = TPAIR[0] - T_L
p0P = C_L + vL * dtl + np.array([0, 0, -0.5 * G * dtl * dtl])
vinP0 = vL + np.array([0, 0, -G * dtl])
KE_pair = 2 * (RP / RB) ** 3 * 0.5 * float(vL @ vL)
EV['seesaw'] = dict(t=T_SS, pivot=pivot.tolist(), arm=ARM, th0=TH0, phi=PHI, u=U.tolist(), t_swing=T_SWING, cup=CUP.tolist(),
                    launch=C_L.tolist(), t_launch=T_L, v_launch=vL.tolist(), ke_bass_loss=KE_loss, ke_pair=KE_pair, bell_h=BELL_H,
                    cup_h=CUP_H)
SFX.append(['seesaw', T_SS]); SFX.append(['launch', T_L])
_TBG = np.arange(0.0, FINAL + 1.0, 1 / 240.0); _PBG = np.array([BASS.at(t) for t in _TBG])
def avoid_pair(k, p, ts):
    return np.stack([np.interp(ts, _TBG, _PBG[:, c]) for c in range(3)], 1)
XO_P, ZO_P = -0.1, -1.9
x_catch = lambda t: min(1.0, max(0.0, (t - TPAIR[0]) / 4.5))
def hx_p(t): return (1 - x_catch(t)) * (p0P[0] + 0.9 * (t - TPAIR[0])) + x_catch(t) * (track(t)[0] + XO_P)
def hz_p(t): return (1 - x_catch(t)) * (p0P[2] - 0.1 * (t - TPAIR[0])) + x_catch(t) * (track(t)[1] + ZO_P)
def hy_p(t): return p0P[1] + (Y_PAIR - p0P[1]) * x_catch(t) ** 0.7
hp, pPE, vPE, _ = cached('pair2' + str(J_IDX) + repr(S_L), lambda: plan_retry(TPAIR, p0P, [vinP0], hx_p, hz_p, hy_p, emax=0.975, emin=0.3,
                             vxlim=(0.3, 3.2), dzr=(-1.0, 0.6), R=RP, ext=(0.16 + 2 * DXC * AXP[0], 0.16 + 2 * DXC * AXP[1], 0.17),
                             avoid=avoid_pair, arad=RB + RP + DXC + 0.05))
PAIRB = [Ball('pairlo', RP, 'pair'), Ball('pairhi', RP, 'pair')]
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

# ============================================================ the snowy ground (terraces under every contact)
allc = sorted([(h['p'][0], h['p'][2]) for n, b in BALLS.items() for h in b.hits])
AX = np.array([a[0] for a in allc]); AZ = np.array([a[1] for a in allc])
FLX = np.arange(-30.0, AX.max() + 40, 0.5)
FLZ = np.array([AZ[np.abs(AX - x) < 3.0].min() if (np.abs(AX - x) < 3.0).any() else np.nan for x in FLX])
s_ = ~np.isnan(FLZ); FLZ = np.interp(FLX, FLX[s_], FLZ[s_]) - 1.0
for _ in range(4): FLZ = np.minimum(FLZ, np.convolve(np.pad(FLZ, 2, mode='edge'), np.ones(5) / 5, 'valid'))
FLZ = np.minimum.accumulate(FLZ)
def floor_z_at(x): return float(np.interp(x, FLX, FLZ))
# the melody terrace (under the lanes of the lantern ball and the children, y > Y_STEP) is higher than the front ground
Y_STEP = -0.42
mc = sorted([(h['p'][0], h['p'][2]) for n, b in BALLS.items() if b.role == 'melody' for h in b.hits])
MXc = np.array([a[0] for a in mc]); MZc = np.array([a[1] for a in mc])
FLZM = np.array([MZc[np.abs(MXc - x) < 3.0].min() if (np.abs(MXc - x) < 3.0).any() else np.nan for x in FLX])
s_ = ~np.isnan(FLZM); FLZM = np.interp(FLX, FLX[s_], FLZM[s_]) - 0.9
for _ in range(4): FLZM = np.minimum(FLZM, np.convolve(np.pad(FLZM, 2, mode='edge'), np.ones(5) / 5, 'valid'))
FLZM = np.minimum.accumulate(FLZM)
def mfloor_z_at(x): return float(np.interp(x, FLX, FLZM))
EV['floor'] = dict(x=FLX.tolist(), z=FLZ.tolist(), zm=FLZM.tolist(), y_step=Y_STEP)

# ============================================================ the village: houses, windows, lamps, fir trees, the church
XEND = float(P_FIN[0])
YN = 6 * LANE + 0.9                          # front face of the first row of houses (behind every lane)
HOUSES = []; WINS = []                       # windows: dict(x, y, z, w, h, house, row)
def add_house(x, yf, zb, w, d, h, ncol, nrow, row, kind='house'):
    k = len(HOUSES); HOUSES.append(dict(x=x, y=yf, zb=zb, w=w, d=d, h=h, row=row, kind=kind))
    wx = w / (ncol + 0.6); wz = min(0.42, h / (nrow + 1.2))
    for c in range(ncol):
        for r in range(nrow):
            WINS.append(dict(x=x - w / 2 + wx * (0.8 + c) , y=yf - 0.01, z=zb + h * (r + 0.75) / (nrow + 0.5), w=wx * 0.55,
                             h=wz * 0.8, house=k, row=row))
# the church (behind the end of the course): nave + tower, a round window and tall windows
XCH = XEND + 1.5
add_house(XCH, YN + 0.6, mfloor_z_at(XCH), 3.2, 3.0, 3.0, 4, 2, 0, 'church')
EV['church'] = dict(x=XCH, y=YN + 0.6, zb=mfloor_z_at(XCH), w=3.2, d=3.0, h=3.0)
# the children's houses: around their doorsteps (built in scene.py; their windows count as village windows too)
for J in JOIN:
    pe = np.array(J['perch'])
    add_house(pe[0] - 0.225, pe[1] - 0.22, pe[2] - RS - 0.05, 0.95, 0.7, 0.95, 2, 1, -1, 'child')
# rows of houses climbing the hill behind the course
x_lo, x_hi = -14.0, XEND + 24.0
HILL = 0.42                                   # the hill rises 0.42 m per metre behind the first row
def hill_z(x, y): return mfloor_z_at(x) + HILL * max(0.0, y - YN)
row = 0; yf = YN
while len(WINS) < NTOTAL + 40 or row < 3:
    x = x_lo + rng.uniform(0, 1.2)
    while x < x_hi:
        if abs(x - XCH) > 2.6 or row > 1:
            w = rng.uniform(1.0, 1.6); h = rng.uniform(1.0, 1.7) * (1 + 0.15 * row); ncol = 2 if w < 1.3 else 3
            add_house(x + w / 2, yf + rng.uniform(0, 0.4), hill_z(x, yf) , w, rng.uniform(0.9, 1.4), h, ncol, 2 if h > 1.25 else 1, row)
            x += w + rng.uniform(0.5, 1.4) + 0.12 * row
        else: x += 1.0
    row += 1; yf += 2.3 + 0.25 * row
NROWS = row
# lamps (one per bass strike, in front of the snowball's lane) and fir trees (one per ornament strike)
LAMPS = []; TREES = []
for h in BASS.hits:
    if h['kind'] == 'churchbell': continue
    LAMPS.append(dict(x=float(h['p'][0]), y=-4.05, zb=floor_z_at(h['p'][0]), t=float(h['t'])))
for b in PAIRB:
    for h in b.hits:
        TREES.append(dict(x=float(h['p'][0]) + (0.3 if b.name == 'pairhi' else -0.3), y=-3.45, zb=floor_z_at(h['p'][0]), t=float(h['t']),
                          s=0.75 + 0.25 * rng.uniform()))
NW = NTOTAL - len(LAMPS) - len(TREES)
assert len(WINS) >= NW, (len(WINS), NW)
# keep exactly NW windows: drop the farthest houses' windows (whole houses first, then the last house's top windows)
WINS.sort(key=lambda w: (w['row'] if w['row'] >= 0 else -1, w['house']))
WINS = WINS[:NW]
keep_h = sorted(set(w['house'] for w in WINS))
assert len(WINS) == NW
# every melody strike (each lane) lights the nearest dark window (the children's own houses and the first row first)
WON = np.full(len(WINS), np.inf)
WX = np.array([w['x'] for w in WINS]); WRow = np.array([w['row'] for w in WINS])
church_w = np.array([HOUSES[w['house']]['kind'] == 'church' for w in WINS])
for n, b in BALLS.items():
    if b.role != 'melody': continue
for i, h in enumerate(MH):
    for j in range(LANES_AT[i]):
        x = h['p'][0]
        cost = np.abs(WX - (x + 0.8)) + 1.6 * np.maximum(WRow, 0) + np.where(np.isinf(WON), 0, 1e6) + np.where(church_w, 1e6, 0)
        k = int(np.argmin(cost)); assert cost[k] < 1e5
        WON[k] = float(h['t'])
# the final chord: the great bell: everything still dark lights up, a wave running outwards from the church
CB = np.array(pBE)
dark = np.isinf(WON)
dd = np.array([math.hypot(w['x'] - CB[0], w['y'] - CB[1]) for w in WINS])
order = np.argsort(np.where(dark, dd, 1e9))
nd = int(dark.sum())
T_WAVE0, T_WAVE1 = FINAL + 0.25, FINAL + 3.0
for r_, k in enumerate(order[:nd]):
    u = r_ / max(1, nd - 1)
    WON[k] = T_WAVE0 + (T_WAVE1 - T_WAVE0) * u ** 0.8
assert not np.isinf(WON).any()
for k, w in enumerate(WINS): w['t'] = float(WON[k])
EV['village'] = dict(houses=HOUSES, keep=keep_h, wins=WINS, lamps=LAMPS, trees=TREES, rows=NROWS, yn=YN, hill=HILL, total=NTOTAL,
                     wave=(T_WAVE0, T_WAVE1), n_by_balls=int((WON < FINAL + 0.01).sum()))
SFX.append(['wave', T_WAVE0])
# the counter: lights on at each time
ALLT = sorted([w['t'] for w in WINS] + [l['t'] for l in LAMPS] + [t['t'] for t in TREES])
assert len(ALLT) == NTOTAL
cnt = []; 
for k, t in enumerate(ALLT): cnt.append((float(t), k + 1))
EV['count'] = [(0.0, 0)] + cnt


def check():
    info = {}
    hs = [(n, h) for n, b in BALLS.items() for h in b.hits if np.linalg.norm(h['vout']) > 0]
    info['e_range'] = (round(min(h['e'] for n, h in hs), 3), round(max(h['e'] for n, h in hs), 3))
    info['speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / max(1e-6, np.linalg.norm(h['vin'])) for n, h in hs), 3)
    info['tilt_max_deg'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for n, h in hs), 1)
    info['seesaw_energy(bass loss vs pair gain, per bass mass)'] = (round(KE_loss, 2), round(KE_pair, 2))
    for n in ('solo', 'bass', 'pairlo'):
        b = BALLS[n]; zz = [h['p'][2] for h in b.hits]; xx = [h['p'][0] for h in b.hits]
        info[n] = dict(hits=len(b.hits), z=(round(min(zz), 2), round(max(zz), 2)), x=(round(min(xx), 1), round(max(xx), 1)))
    info['joiners'] = [(J['j'], round(J['e'], 2), round(J['T_fly'], 2), round(J['L'], 2), [round(c, 2) for c in J['perch']]) for J in JOIN]
    info['bass_ramp'] = (round(LB, 3), round(T_RB, 3), round(sB, 3))
    info['village'] = dict(houses=len(keep_h), rows=NROWS, windows=len(WINS), lamps=len(LAMPS), trees=len(TREES),
                           lit_by_balls=EV['village']['n_by_balls'] + len(LAMPS) + len(TREES), total=NTOTAL)
    return info


if __name__ == '__main__':
    info = check()
    for k, v in info.items(): print(k, v)
    out = dict(G=G, R=dict(solo=RS, bass=RB, pair=RP), lane=LANE, y_bass=Y_BASS, y_pair=Y_PAIR, dxc=DXC,
               balls={n: dict(r=b.r, role=b.role, seg=b.seg, hits=b.hits) for n, b in BALLS.items()},
               lanes_at=LANES_AT, ev=EV, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
