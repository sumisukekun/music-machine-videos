"""Ride of the Valkyries - '8 riders: who reaches Valhalla first?'. Physics + story planner.
Run after arrange.py; writes layout.json (read by scene.py, render_audio.py, verify.py).

Story (side view; the course runs along +x; the eight lanes lie one behind the other along +y):
  0-1.8 s   storm over the Valkyrie rock. Lightning: the THUNDER BALL drops out of the cloud onto the first drum (bass),
            the first Valkyrie rolls out of her gate in the rock.
  bars 1-8  the riders one by one: Valkyrie j leaves her gate, plays bar j of the theme alone on her own lane and lands in
            her cup on the start line (the eight cups fill up: RIDERS 1/8 .. 8/8). The thunder ball doubles the theme on drums.
  bar 9     the start gate drops: all eight roll off the line together and RACE, the theme in octaves (8 riders = 8 voices,
            each on her own lane of keys, each at her own pace: the leader changes)
  bar 16    the thunder ball strikes the lever drum: the drawbridge of Valhalla falls, the clouds part (B major, sunrise);
            the riders leap 1.33 s off the end of the storm bridge onto the golden bridge
  17-27     the race goes on in B major across the golden castle
  bar 28    photo finish: all eight land in the eight bells of the final chord at the same instant, on the same line;
            the thunder ball lands on the great drum; the gate of Valhalla opens
Physics: one constant gravity G; between contacts pure parabolas; every bounce is an impulse along the surface normal
(tangential velocity kept) with restitution emin <= e <= emax, so a bounce never adds speed. Ramps: rolling from rest,
a = (5/7) g sin(alpha). Cups and the final bells are dead landings (e = 0)."""
import json, os, math, pickle, hashlib, time
_T0 = time.time()
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = 7.0
RS, RB = 0.22, 0.34                        # riders, thunder ball
LANE = 0.75                                # lane j at y = j * LANE
Y_BASS = -1.7
EMAX, EMIN, TILT = 0.97, 0.22, 34.0
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
T0, BAR, S16, FINAL = E['t0'], E['bar'], E['s16'], E['final']
BT = {int(k): v for k, v in E['bar_t'].items()}
NR = 8
NAMES = ['GERHILDE', 'ORTLINDE', 'WALTRAUTE', 'SCHWERTLEITE', 'HELMWIGE', 'SIEGRUNE', 'GRIMGERDE', 'ROSSWEISSE']
SFX = []; EV = {}
rng = np.random.default_rng(11)


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


SOFT = 0.0
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
            if last and final[1] is None:                 # only the finish line (x) is given: any height
                dzs = np.linspace(dzr[0], dzr[1], NZ) * max(0.4, d / 0.66)
                DZ = dzs; VX = np.full(len(dzs), (final[0] - p[0]) / d)
            elif last:
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
                cc += SOFT * (max(0.0, -land[j] - 3.0) ** 2 + 4.0 * max(0.0, land[j] + 1.0) ** 2 + 4.0 * max(0.0, 0.9 - VX[j]) ** 2)  # landings neither too hard nor too soft
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




KEXT = (0.17, 0.26, 0.07)                  # key half-extents (along x, along y, thickness) used by the planner
BEXT = (0.34, 0.34, 0.16)                  # drums


def ramp_in(p1, t1, alpha, T_r, T_f):
    """a ball at rest on a perch rolls T_r down a ramp of angle alpha (from rest), flies T_f and arrives at p1 at t1"""
    a = 5 / 7 * G * math.sin(alpha); s = a * T_r; L = 0.5 * a * T_r ** 2
    u = np.array([math.cos(alpha), 0, -math.sin(alpha)]); vex = u * s
    lip = np.array(p1, float) - vex * T_f + np.array([0, 0, 0.5 * G * T_f ** 2])
    return dict(perch=lip - u * L, lip=lip, u=u, a=a, vex=vex, vin=vex + np.array([0, 0, -G * T_f]), t_rel=t1 - T_f - T_r,
                T_r=T_r, T_f=T_f, L=L, alpha=alpha, t1=t1)


def ramp_out(perch, t_rel, alpha, T_r, T_f):
    """the same, forward: from a perch released at t_rel; returns the dict with p1 / t1"""
    a = 5 / 7 * G * math.sin(alpha); s = a * T_r; L = 0.5 * a * T_r ** 2
    u = np.array([math.cos(alpha), 0, -math.sin(alpha)]); vex = u * s
    lip = np.array(perch, float) + u * L
    p1 = lip + vex * T_f + np.array([0, 0, -0.5 * G * T_f ** 2])
    return dict(perch=np.array(perch, float), lip=lip, u=u, a=a, vex=vex, vin=vex + np.array([0, 0, -G * T_f]), t_rel=t_rel,
                T_r=T_r, T_f=T_f, L=L, alpha=alpha, t1=t_rel + T_r + T_f, p1=p1)


def put_ramp(b, R):
    t0 = R['t_rel']; tt = list(np.arange(t0, t0 + R['T_r'], 1 / 240.0)) + [t0 + R['T_r']]
    b.path(tt, [R['perch'] + R['u'] * (0.5 * R['a'] * (t - t0) ** 2) for t in tt])
    b.fly(t0 + R['T_r'], R['t1'], R['lip'], R['vex'])


def legal(vin, vout, emin=0.25, emax=0.9):
    e, n = bounce_e(np.asarray(vin, float), np.asarray(vout, float))
    return emin <= e <= emax and n[2] >= math.cos(math.radians(TILT)) and np.linalg.norm(vout) <= np.linalg.norm(vin), e


def smoothstep(u): u = min(1.0, max(0.0, u)); return u * u * (3 - 2 * u)


BALLS = {}
RID = [Ball(f'v{j}', RS, 'rider') for j in range(NR)]
for b in RID: BALLS[b.name] = b
_ck = hashlib.md5((open(__file__).read().split('# ---- plan cache key end')[0] + json.dumps(LEAD)).encode()).hexdigest()
_cf = f'/tmp/valk_plan_{_ck}.pkl'
CACHE = pickle.load(open(_cf, 'rb')) if os.path.exists(_cf) else {}

# ============================================================ act 1: the riders one by one (bars 1-8)
X1, Z1 = -0.6, 2.8                     # first contact of every solo bar (in the rider's own lane)
XC, ZC = 3.3, 0.5                      # the start line: ball centre in the cups
A1, TR1, TF1 = math.radians(28), 0.55, 0.42
S1 = []
for j, b in enumerate(RID):
    y = j * LANE; bar = j + 1
    L1 = [l for l in LEAD if l[4] == bar]; tms = [l[0] for l in L1]
    R = ramp_in((X1, y, Z1), tms[0], A1, TR1, TF1)
    b.rest(0.0, R['t_rel'], R['perch']); put_ramp(b, R)
    SFX.append(['gate', float(R['t_rel'])])
    C = np.array([XC, y, ZC]); key = ('s1', j)
    if key not in CACHE:
        found = None
        for dxl in (0.0, -0.3, 0.3, -0.6, 0.6, -0.9):
            for zl in (0.35, 0.0, 0.7, 1.0, -0.4):
                tL = tms[-1]; xL = XC - 1.15 + dxl; zL = ZC + zl
                xt = lambda t, xL=xL, tL=tL: X1 + (xL - X1) * (t - tms[0]) / (tL - tms[0])
                zt = lambda t, zL=zL, tL=tL: Z1 + (zL - Z1) * (t - tms[0]) / (tL - tms[0])
                try: hs, pL, vL, _ = plan_retry(tms, (X1, y, Z1), [R['vin']], xt, zt, lambda t: y, R=RS, ext=KEXT, dzr=(-1.6, 0.6))
                except RuntimeError: continue
                for Tc in np.linspace(0.3, 0.66, 37):
                    vo = (C - pL) / Tc + np.array([0, 0, 0.5 * G * Tc])
                    ok, e = legal(vL, vo)
                    if ok: found = (hs, pL, vL, vo, Tc); break
                if found: break
            if found: break
        assert found, ('no cup flight', j)
        CACHE[key] = found
    hs, pL, vL, vo, Tc = CACHE[key]
    allh = [dict(t=h['t'], p=np.array(h['p']), vin=np.array(h['vins'][0]), vout=np.array(h['vout'])) for h in hs]
    allh.append(dict(t=tms[-1], p=pL, vin=vL, vout=vo))
    for k, h in enumerate(allh):
        b.hit(h['t'], h['p'], h['vin'], h['vout'], 'key', L1[k][1], ev=LEAD.index(L1[k]), sec=1)
        t1 = allh[k + 1]['t'] if k + 1 < len(allh) else tms[-1] + Tc
        b.fly(h['t'], t1, h['p'], h['vout'])
    t_cup = tms[-1] + Tc
    vcup = vo + np.array([0, 0, -G * Tc])
    b.hit(t_cup, C, vcup, np.zeros(3), 'cup', None)
    S1.append(dict(j=j, gate=R['t_rel'], perch=R['perch'].tolist(), lip=R['lip'].tolist(), alpha=A1, L=R['L'], t_cup=float(t_cup),
                   cup=C.tolist()))
EV['s1'] = S1

# ============================================================ act 2+3: the race (bars 9-27 + the final chord)
A2, TR2, TF2 = math.radians(35), 0.3, 0.3
EMIN_R = 0.12                          # the riders' keys are soft (felt-lined): a hard landing before a 16th can be damped
T9 = BT[9]; T17 = BT[17]
R2 = [ramp_out((XC, j * LANE, ZC), T9 - TF2 - TR2, A2, TR2, TF2) for j in range(NR)]
T_START = R2[0]['t_rel']
assert max(s['t_cup'] for s in S1) < T_START - 0.05, ('a rider is not on the start line in time', [s['t_cup'] for s in S1], T_START)
SFX.append(['start', float(T_START)])
EV['start'] = dict(t=float(T_START), alpha=A2, L=R2[0]['L'], x=XC, z=ZC)
LR = [l for l in LEAD if l[4] >= 9]
TR = [l[0] for l in LR] + [FINAL]
T_LEAP = [l[0] for l in LR if l[4] == 16][-1]               # the long A# of bar 16: the leap off the storm bridge
P1x, P1z = float(R2[0]['p1'][0]), float(R2[0]['p1'][2])
V2, V3, DXLEAP, DZLEAP = 1.5, 1.4, 2.2, -3.0
SL2, SL3 = 0.32, 0.3                    # the course steps down: every bounce loses energy, the height pays for it
def base_x(t):
    if t <= T_LEAP: return P1x + V2 * (t - T9)
    xl = P1x + V2 * (T_LEAP - T9)
    if t <= T17: return xl + DXLEAP * (t - T_LEAP) / (T17 - T_LEAP)
    return xl + DXLEAP + V3 * (t - T17)
def base_z(t):
    if t <= T_LEAP: return P1z - SL2 * (t - T9)
    zl = P1z - SL2 * (T_LEAP - T9)
    if t <= T17: return zl + DZLEAP * (t - T_LEAP) / (T17 - T_LEAP)
    return zl + DZLEAP - SL3 * (t - T17)
XF, ZF = base_x(FINAL), base_z(FINAL)
PER = [11.0, 14.5, 10.0, 12.5, 15.5, 10.5, 13.0, 11.5]; PHI = rng.uniform(0, 2 * math.pi, NR)
def env(t): return smoothstep((t - T9) / 3.0) * (1 - smoothstep((t - (FINAL - 5.5)) / 4.5))
def off(j, t): return 0.65 * math.sin(2 * math.pi * (t - T9) / PER[j] + PHI[j]) * env(t)
_b = [k for k, l in enumerate(LR) if l[5] == 6]                # chunk boundaries on a long note (the next hop is long)
_c1 = min(_b, key=lambda k: abs(k - 30)); _c2 = min(_b, key=lambda k: abs(k - 60))
SOFT = 0.5
CHUNKS = [(0, len(TR) - 1)]                          # one beam search over the whole race
RACE = {}
for j, b in enumerate(RID):
    key = ('race', j)
    if key not in CACHE:
        p_, v_ = R2[j]['p1'], R2[j]['vin']; hist = (); H = []
        y = j * LANE
        for a, c in CHUNKS:
            last = c == len(TR) - 1
            hs, p_, v_, hist = plan_retry(TR[a:c + 1], p_, [v_], lambda t, j=j: base_x(t) + off(j, t), base_z, lambda t, y=y: y,
                                          R=RS, ext=KEXT, final=(XF, None) if last else None, hist0=hist, dzr=(-2.5, 1.0),
                                          emin=EMIN_R)
            H += [dict(t=h['t'], p=np.array(h['p']), vin=np.array(h['vins'][0]), vout=np.array(h['vout'])) for h in hs]
        CACHE[key] = (H, p_, v_); print('race planned', j, round(time.time() - _T0, 1), flush=True)
        pickle.dump(CACHE, open(_cf, 'wb'))
    RACE[j] = CACHE[key]
pickle.dump(CACHE, open(_cf, 'wb'))
# ---- plan cache key end
for j, b in enumerate(RID):
    b.rest(S1[j]['t_cup'], R2[j]['t_rel'], R2[j]['perch']); put_ramp(b, R2[j])
    H, pF, vF = RACE[j]
    assert len(H) == len(LR)
    for k, h in enumerate(H):
        l = LR[k]; p = l[1] - 12 * (j % 2)
        b.hit(h['t'], h['p'], h['vin'], h['vout'], 'key', p, ev=LEAD.index(l), sec=2 if l[4] <= 16 else 3)
        t1 = H[k + 1]['t'] if k + 1 < len(H) else FINAL
        b.fly(h['t'], t1, h['p'], h['vout'])
    b.hit(FINAL, pF, vF, np.zeros(3), 'final', E['fin_p'][j])
    b.rest(FINAL, FINAL + 30, pF)

# ============================================================ the thunder ball (bass on drums)
BE = E['bass']; TB = [b_[0] for b_ in BE]
RT = np.array([h['t'] for h in RID[0].hits if h['kind'] in ('key', 'final')])
def pack_x(t):
    if t < T9: return X1 - 0.3 + (XC + 0.4 - X1 + 0.3) * (t - T0) / (T9 - T0)
    return float(np.mean([np.interp(t, [h['t'] for h in b.hits if h['kind'] != 'cup' and h['t'] >= T9 - 0.01],
                                    [h['p'][0] for h in b.hits if h['kind'] != 'cup' and h['t'] >= T9 - 0.01]) for b in RID]))
def pack_z(t):
    if t < T9: return ZC + 2.2 - 3.4 * (t - T0) / (T9 - T0)
    return base_z(t) - 1.25
I_LEVER = [k for k, b_ in enumerate(BE) if b_[2] == 16 and b_[3] == 0][0]
BASS = Ball('bass', RB, 'bass'); BALLS['bass'] = BASS
xb0, zb0 = pack_x(T0), pack_z(T0)
T_DROP = 1.0
BASS.rest(0.0, T0 - T_DROP, (xb0, Y_BASS, zb0 + 0.5 * G * T_DROP ** 2)); BASS.fly(T0 - T_DROP, T0, (xb0, Y_BASS, zb0 + 0.5 * G * T_DROP ** 2), (0, 0, 0))
EV['cloud_drop'] = dict(t=T0 - T_DROP, p=[xb0, Y_BASS, zb0 + 0.5 * G * T_DROP ** 2])
SFX.append(['thunder', T0 - T_DROP])
key = ('bass',)
SOFT = 0.0
if key not in CACHE:
    CACHE[key] = plan_retry(TB, (xb0, Y_BASS, zb0), [np.array([0, 0, -G * T_DROP])], lambda t: pack_x(t) + 0.2, pack_z,
                            lambda t: Y_BASS, emax=0.975, emin=0.15, vxlim=(0.2, 3.8), dzr=(-2.5, 1.0), R=RB, ext=BEXT,
                            final=None)
    pickle.dump(CACHE, open(_cf, 'wb'))
hb, pBE, vBE, _ = CACHE[key]
for k, h in enumerate(hb):
    kind = 'lever' if k == I_LEVER else 'drum'
    BASS.hit(h['t'], h['p'], h['vins'][0], h['vout'], kind, BE[k][1], ev=k)
    t1 = hb[k + 1]['t'] if k + 1 < len(hb) else TB[-1]
    BASS.fly(h['t'], t1, h['p'], h['vout'])
BASS.hit(TB[-1], pBE, vBE, np.zeros(3), 'final', BE[-1][1], ev=len(BE) - 1)
BASS.rest(TB[-1], FINAL + 30, pBE)
T_LEVER = float(TB[I_LEVER])
EV['lever'] = dict(t=T_LEVER, p=[float(c) for c in hb[I_LEVER]['p']])
# the drawbridge: hinge at the start of the golden bridge, falls from upright to flat between the lever and the landing
x_land = min(float(h['p'][0]) for b in RID for h in b.hits if abs(h['t'] - T17) < 1e-4)
EV['bridge'] = dict(t0=T_LEVER + 0.15, t1=T17 - 0.35, x_hinge=x_land - 1.3, z=base_z(T17) - 0.9)
SFX.append(['lever', T_LEVER]); SFX.append(['bridge', T17 - 0.35])
EV['leap'] = dict(t0=float(T_LEAP), t1=float(T17))
SFX.append(['gateopen', FINAL + 0.25])
EV['finish'] = dict(x=XF, z=float(min(b.hits[-1]['p'][2] for b in RID)), zmax=float(max(b.hits[-1]['p'][2] for b in RID)), t=FINAL)
EV['names'] = NAMES; EV['lane'] = LANE
# riders on the start line (counter for act 1)
EV['count'] = [(0.0, 0)] + [(s['t_cup'], k + 1) for k, s in enumerate(sorted(S1, key=lambda s: s['t_cup']))]


def check():
    info = {}
    hs = [(n, h) for n, b in BALLS.items() for h in b.hits if np.linalg.norm(h['vout']) > 0]
    info['e_range'] = (round(min(h['e'] for n, h in hs), 3), round(max(h['e'] for n, h in hs), 3))
    info['speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / max(1e-6, np.linalg.norm(h['vin'])) for n, h in hs), 3)
    info['tilt_max_deg'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for n, h in hs), 1)
    for b in RID + [BASS]:
        zz = [h['p'][2] for h in b.hits]; xx = [h['p'][0] for h in b.hits]
        info[b.name] = dict(hits=len(b.hits), z=(round(min(zz), 2), round(max(zz), 2)), x=(round(min(xx), 1), round(max(xx), 1)))
    info['cups'] = [round(s['t_cup'], 2) for s in S1]; info['start'] = round(T_START, 2)
    info['finish'] = (round(XF, 2), round(ZF, 2))
    return info


if __name__ == '__main__':
    info = check()
    for k, v in info.items(): print(k, v)
    out = dict(G=G, R=dict(rider=RS, bass=RB), lane=LANE, y_bass=Y_BASS,
               balls={n: dict(r=b.r, role=b.role, seg=b.seg, hits=b.hits) for n, b in BALLS.items()}, ev=EV, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
