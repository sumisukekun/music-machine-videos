"""Hungarian Dance No. 5 - 'night rooftop chase' (夜の屋根の上の追いかけっこ). Physics planner -> layout.json.
Thief (black ball, front lane) strikes every melody note, the police ball (blue, back lane) strikes every bass note.
Motion rules (checked by verify.py):
  * one constant gravity G between contacts (pure parabolas),
  * a bounce keeps the tangential velocity and reverses the normal one with restitution e <= EMAX (speed never grows);
    the tile under the ball is tilted along the impulse, so a tilted tile is what turns falling into running,
  * rolling parts (eave gutter, slatted decks) move with constant acceleration along a straight surface:
    down the gutter  a = 5/7 G sin(theta) (speeds up), on the slatted deck a = -rolling friction (slows down),
  * landing on a gutter/deck is plastic (normal speed absorbed, tangential kept),
  * the ball leaves a deck at its edge with the small creeping speed and falls freely.
Hop heights come out of a beam search: each hop picks (vx, dz) among physically possible bounces, preferring a gentle
descent of the town and the story's gaps between the two balls."""
import json, os, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json')); BASS = json.load(open(f'{HERE}/bass.json'))
G = E['G']; R = 0.15; EMAX = 0.96; EMIN = 0.10; TILT = 42.0
YT, YC = -0.6, 0.6
SEC = {s['name']: s for s in E['secs']}; DK = E['decks']
TT = np.array([l['t'] for l in LEAD]); TB = np.array([b['t'] for b in BASS])


class Ball:
    def __init__(s, name): s.name = name; s.seg = []; s.hits = []
    def rest(s, t0, t1, p): s.seg.append(dict(k='rest', t0=float(t0), t1=float(t1), p=list(map(float, p))))
    def fly(s, t0, t1, p, v): s.seg.append(dict(k='fly', t0=float(t0), t1=float(t1), p=list(map(float, p)), v=list(map(float, v))))
    def roll(s, t0, t1, p, u, v0, a): s.seg.append(dict(k='roll', t0=float(t0), t1=float(t1), p=list(map(float, p)), u=list(map(float, u)), v0=float(v0), a=float(a)))
    def hit(s, i, t, p, n, e, kind, vin, vout, **kw):
        h = dict(i=int(i), t=float(t), p=list(map(float, p)), n=list(map(float, n)), e=float(e), kind=kind, vin=list(map(float, vin)), vout=list(map(float, vout)))
        h.update(kw); s.hits.append(h)
    def at(s, t):
        for g in s.seg:
            if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9: return seg_at(g, t)
        return seg_at(s.seg[0], s.seg[0]['t0']) if t < s.seg[0]['t0'] else seg_at(s.seg[-1], s.seg[-1]['t1'])


def seg_at(g, t):
    p = np.array(g['p']); u = t - g['t0']
    if g['k'] == 'rest': return p
    if g['k'] == 'fly': return p + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
    return p + np.array(g['u']) * (g['v0'] * u + 0.5 * g['a'] * u * u)


def bounce(vin, vout):
    d = vout - vin; n = d / max(1e-9, np.linalg.norm(d)); a = vin @ n
    return (-(vout @ n) / a if a < -1e-9 else 9.0), n


def beam(p0, vin0, times, rate, vxt, xt=None, vy=0.0, K=48, emax=EMAX, emin=EMIN, tilt=TILT, final_vx=None, dvmax=1.2,
         wx=0.0, name='', zt=None, wz=0.0, final_vz=-0.25):
    """hops from times[0] (ball at p0 with incoming vin0, about to bounce) through times[1:].
    rate(t): preferred height change per second, vxt(t): preferred horizontal speed, xt(t): x target (optional).
    returns list of (p, vin, vout, n, e) for each bounce at times[0..-2] and the arrival state at times[-1]."""
    st = [(0.0, np.array(p0, float), np.array(vin0, float), None)]
    hist = []
    dvs = np.linspace(-dvmax, dvmax, 25)
    for k in range(len(times) - 1):
        dt = times[k + 1] - times[k]; t1 = times[k + 1]
        dzs = np.linspace(-1.8 * max(0.3, dt), 0.5 * dt + 0.05, 110)
        for emin_, tilt_, look in ((emin, tilt, True), (emin, tilt, False), (0.04, 52.0, False)):
            new = []
            for si, (c, p, vin, _) in enumerate(st):
                if final_vx is not None and k == len(times) - 2: vxs = np.array([final_vx])
                else: vxs = np.clip(vin[0] + dvs, 0.6, 4.2)
                VX, DZ = np.meshgrid(vxs, dzs, indexing='ij'); VX = VX.ravel(); DZ = DZ.ravel()
                vo = np.stack([VX, np.full_like(VX, vy), DZ / dt + 0.5 * G * dt], 1)
                d = vo - vin; nn = d / np.maximum(1e-9, np.linalg.norm(d, axis=1))[:, None]
                a = nn @ vin; e = -(np.sum(vo * nn, 1)) / np.where(a < -1e-9, a, -1e-9)
                vlim = final_vz if k == len(times) - 2 else -0.25
                ok = (a < -1e-6) & (e >= emin_) & (e <= emax) & (nn[:, 2] >= math.cos(math.radians(tilt_))) & (vo[:, 2] - G * dt < vlim)
                if look and k + 2 < len(times):                        # one-step lookahead: arrive fast enough for the next hop
                    dt2 = times[k + 2] - times[k + 1]
                    vzr = max(0.0, -1.8 * max(0.3, dt2) / dt2 + 0.5 * G * dt2)
                    vxr = final_vx if (final_vx is not None and k + 2 == len(times) - 1) else 0.0
                    va = vo.copy(); va[:, 2] -= G * dt
                    ok &= np.linalg.norm(va, axis=1) * emax * 0.98 >= math.hypot(vxr, vzr)
                if not ok.any(): continue
                idx = np.nonzero(ok)[0]
                til = np.degrees(np.arccos(np.clip(nn[idx, 2], -1, 1)))
                cost = (c + 6.0 * (DZ[idx] - rate(t1) * dt) ** 2 + 1.5 * (VX[idx] - vxt(t1)) ** 2 + 0.8 * (VX[idx] - vin[0]) ** 2
                        + 0.25 * (til / 40) ** 2 + 0.3 * (e[idx] - 0.8) ** 2)
                pn = p[None, :] + vo[idx] * dt + np.array([0, 0, -0.5 * G * dt * dt])
                if xt is not None: cost = cost + wx * (pn[:, 0] - xt(t1)) ** 2
                if zt is not None: cost = cost + wz * (pn[:, 2] - zt(t1)) ** 2
                # keep a few per parent so the beam stays diverse
                o = np.argsort(cost)[:6]
                for j in o:
                    jj = idx[j]
                    vnext = vo[jj] + np.array([0, 0, -G * dt])
                    new.append((float(cost[j]), pn[j], vnext, (si, vo[jj], nn[jj], float(e[jj]))))
            if new: break
            print('WARN relax', name, k, round(times[k], 3), emin_, tilt_, look)
        if not new:
            raise RuntimeError(f'beam dead {name} hop {k} t={times[k]:.3f} dt={dt:.3f}')
        new.sort(key=lambda s: s[0])
        # thin near-duplicates
        kept = []
        for s in new:
            if all(abs(s[1][2] - q[1][2]) > 0.004 or abs(s[2][0] - q[2][0]) > 0.02 for q in kept): kept.append(s)
            if len(kept) >= K: break
        hist.append(st); st = kept
    # backtrack
    best = st[0]; chain = []; cur = best
    for k in range(len(times) - 2, -1, -1):
        si, vo, n, e = cur[3]; prev = hist[k][si]
        chain.append((prev[1], prev[2], vo, n, e)); cur = prev
    chain.reverse()
    return chain, (best[1], best[2])


def apply(ball, chain, times, idxs, kinds):
    for (p, vin, vo, n, e), t0, t1, i, kd in zip(chain, times[:-1], times[1:], idxs, kinds):
        ball.hit(i, t0, p, n, e, kd, vin, vo); ball.fly(t0, t1, p, vo)


T_, C_ = Ball('thief'), Ball('cop')
EVS = {}; SFX = []
def lead_idx(sec, bar=None, pos=None):
    return [k for k, l in enumerate(LEAD) if l['sec'] == sec and (bar is None or l['bar'] == bar) and (pos is None or abs(l['pos'] - pos) < 1e-6)]
def bass_idx(sec, bar=None):
    return [k for k, b in enumerate(BASS) if b['sec'] == sec and (bar is None or b['bar'] == bar)]
SECT_ORDER = [s['name'] for s in E['secs']]
def sec_of(t):
    nm = SECT_ORDER[0]
    for s in E['secs']:
        if t >= s['t'] - 1e-6: nm = s['name']
    return nm
RATE = {'A': -0.16, 'B': -0.12, 'BR': 0.0, 'B2': -0.15, 'TV': -0.75, 'D1': 0, 'V1': -0.6, 'D2': 0, 'V2': -0.6, 'A3': -0.16, 'coda': -0.1}
rate = lambda t: RATE[sec_of(t)]
VXT = {'A': 2.05, 'B': 2.0, 'BR': 2.0, 'B2': 2.0, 'TV': 2.5, 'D1': 2.3, 'V1': 2.6, 'D2': 2.3, 'V2': 2.6, 'A3': 2.1, 'coda': 2.0}
vxt = lambda t: VXT[sec_of(t)]
_tz = np.arange(0, 80, 0.05); _zz = np.cumsum([RATE[sec_of(t)] * 0.05 for t in _tz])
_zz -= np.array([sum(DK[k]['hfall'] for k in DK if DK[k]['t_edge'] < t) for t in _tz])   # the falls off the deck edges
zt_thief = lambda t: float(np.interp(t, _tz, _zz))


# ============================================================ THIEF ============================================================
# start: sits on a chimney lid; the lid tips at the whistle and the thief drops onto the first roof tile
H0 = 1.25; tdrop = math.sqrt(2 * H0 / G); P1 = np.array([0.0, YT, 0.0])
t_rel = TT[0] - tdrop; Pst = P1 + np.array([-0.05, 0, H0])
T_.rest(0, t_rel, Pst); T_.fly(t_rel, TT[0], Pst, [0.05 / tdrop, 0, 0])
EVS['thief_start'] = dict(p=Pst.tolist(), t_rel=t_rel)
vin = np.array([0.05 / tdrop, 0, -G * tdrop]); p = P1


def thief_run(p, vin, ks, kinds=None, **kw):
    times = list(TT[ks]); kinds = kinds or [LEAD[k]['kind'] for k in ks[:-1]]
    chain, (pe, ve) = beam(p, vin, times, rate, vxt, name='thief', **kw)
    apply(T_, chain, times, ks[:-1], kinds)
    return pe, ve


def roll_part(ball, p, vin, times, idxs, u, a, kind, i_land_kind, v0=None, end_extra=0.0):
    """plastic landing at times[0] on a straight surface along u, then constant acceleration a along u.
    one contact (slat) per time. returns (end position, end velocity, end time) at times[-1] + end_extra"""
    u = np.array(u, float); u /= np.linalg.norm(u)
    vt = float(vin @ u) if v0 is None else v0
    nrm = np.array([-u[2], 0, u[0]]) if u[0] > 0 else np.array([0, 0, 1.0])
    t0 = times[0]; t_end = times[-1] + end_extra
    ball.roll(t0, t_end, p, u, vt, a)
    for j, (t, i) in enumerate(zip(times, idxs)):
        tau = t - t0; q = p + u * (vt * tau + 0.5 * a * tau * tau); vq = u * (vt + a * tau)
        vi = vin if j == 0 else vq
        ball.hit(i, t, q, nrm, 0.0, i_land_kind if j == 0 else kind, vi, vq, roll=True)
    tau = t_end - t0
    return p + u * (vt * tau + 0.5 * a * tau * tau), u * (vt + a * tau), t_end


# --- A: bars 1-12 hops, bars 13-15 the eave gutter (rolling, speeding up), then off the gutter end
kA = lead_idx('A'); kG0 = lead_idx('A', 12, 0)[0]; kG1 = lead_idx('A', 14, 4)[0]       # Eb5 (bar 13) .. A4 dotted (bar 15)
p, vin = thief_run(p, vin, list(range(kA[0], kG0 + 1)))
TH_G = math.radians(6.0); uG = np.array([math.cos(TH_G), 0, -math.sin(TH_G)]); aG = 5 / 7 * G * math.sin(TH_G)
gks = list(range(kG0, kG1 + 1))
pg, vg, tg = roll_part(T_, p, vin, list(TT[gks]), gks, uG, aG, 'gut', 'gut')
EVS['gutter'] = dict(p0=p.tolist(), p1=pg.tolist(), t0=float(TT[kG0]), t1=float(tg), u=uG.tolist(), a=aG, i0=kG0, i1=kG1)
# the ball runs off the gutter end (the last bracket) and flies on
k_after = kG1 + 1; dt = TT[k_after] - tg
T_.fly(tg, TT[k_after], pg, vg)
p = pg + vg * dt + np.array([0, 0, -0.5 * G * dt * dt]); vin = vg + np.array([0, 0, -G * dt])
# --- rest of A, B (bars 33-40) up to the landing on the first slatted deck (bar 41 downbeat)
kB = lead_idx('BR')
p, vin = thief_run(p, vin, list(range(k_after, kB[0] + 1)), final_vx=DK['BR']['v0'])


def deck(ball, p, vin, name, idxs_notes, times_notes, lane):
    dk = DK[name]
    assert abs(times_notes[0] - dk['t_start']) < 1e-4
    pe, ve, te = roll_part(ball, p, vin, times_notes, idxs_notes, [1, 0, 0], -dk['a'], 'slat', 'slat', v0=dk['v0'],
                           end_extra=dk['t_edge'] - times_notes[-1])
    EVS.setdefault('decks', {}).setdefault(name, {})[lane] = dict(x0=float(p[0]), x1=float(pe[0]), z=float(p[2]), y=float(p[1]),
                                                                  t0=dk['t_start'], t_edge=dk['t_edge'], vE=dk['vE'])
    tf = dk['tfall']; ball.fly(te, te + tf, pe, ve)
    SFX.append(['creak', te])
    return pe + ve * tf + np.array([0, 0, -0.5 * G * tf * tf]), ve + np.array([0, 0, -G * tf]), te + tf


p, vin, t_ = deck(T_, p, vin, 'BR', kB, list(TT[kB]), 'T')
kB2 = lead_idx('B2'); assert abs(TT[kB2[0]] - t_) < 1e-4
kD1 = lead_idx('D1')
p, vin = thief_run(p, vin, list(range(kB2[0], kD1[0] + 1)), final_vx=DK['D1']['v0'])
p, vin, t_ = deck(T_, p, vin, 'D1', kD1, list(TT[kD1]), 'T')
kV1 = lead_idx('V1'); kD2 = lead_idx('D2')
p, vin = thief_run(p, vin, list(range(kV1[0], kD2[0] + 1)), final_vx=DK['D2']['v0'])
p, vin, t_ = deck(T_, p, vin, 'D2', kD2, list(TT[kD2]), 'T')
kV2 = lead_idx('V2'); k84 = lead_idx('A3', 15, 0)[0]
p, vin = thief_run(p, vin, list(range(kV2[0], k84 + 1)), final_vz=-1.4)
THIEF_84 = (p, vin)

# ============================================================ COP ============================================================
def cop_kinds(ks): return ['plate'] * len(ks)
GAP = {'A': 1.6, 'B': 0.8, 'BR': 0.35, 'B2': 0.6, 'TV': 0.8, 'D1': 0.45, 'V1': 0.6, 'D2': 0.45, 'V2': 0.6, 'A3': 1.1, 'coda': 0.0}
def x_thief(t): return float(T_.at(t)[0])
_zth = np.array([T_.at(t)[2] for t in _tz]); _k = np.ones(21) / 21
_zths = np.convolve(np.pad(_zth, 10, mode='edge'), _k, 'valid')
def z_thief(t): return float(np.interp(t, _tz, _zths))
T84 = float(TT[lead_idx('A3', 15, 0)[0]])
def xt_cop(t):
    g = GAP[sec_of(t)]
    if sec_of(t) == 'A3': g = float(np.interp(t, [T84 - 3.0, T84 - 0.6], [1.1, 0.25]))   # the police closes in at the very end
    return x_thief(t) - g


_xth = np.array([T_.at(t)[0] for t in _tz]); _xths = np.convolve(np.pad(_xth, 10, mode='edge'), _k, 'valid')
_vth = np.gradient(_xths, _tz)
def vxt_cop(t): return float(np.interp(t, _tz, _vth))


def cop_run(p, vin, ks, **kw):
    times = list(TB[ks])
    chain, (pe, ve) = beam(p, vin, times, rate, vxt_cop, xt=xt_cop, wx=2.5, name='cop', zt=z_thief, wz=1.5, **kw)
    apply(C_, chain, times, ks[:-1], ['plate'] * (len(ks) - 1))
    return pe, ve


H0c = 1.45; tdc = math.sqrt(2 * H0c / G); PC1 = np.array([-1.5, YC, -0.1])
Pcs = PC1 + np.array([-0.04, 0, H0c])
C_.rest(0, TB[0] - tdc, Pcs); C_.fly(TB[0] - tdc, TB[0], Pcs, [0.04 / tdc, 0, 0])
EVS['cop_start'] = dict(p=Pcs.tolist(), t_rel=float(TB[0] - tdc))
vin = np.array([0.04 / tdc, 0, -G * tdc]); p = PC1
cB = bass_idx('BR')
p, vin = cop_run(p, vin, list(range(0, cB[0] + 1)), final_vx=DK['BR']['v0'])
p, vin, t_ = deck(C_, p, vin, 'BR', cB, list(TB[cB]), 'C')
cB2 = bass_idx('B2'); assert abs(TB[cB2[0]] - t_) < 1e-4
cD1 = bass_idx('D1')
p, vin = cop_run(p, vin, list(range(cB2[0], cD1[0] + 1)), final_vx=DK['D1']['v0'])
p, vin, t_ = deck(C_, p, vin, 'D1', cD1, list(TB[cD1]), 'C')
cV1 = bass_idx('V1'); cD2 = bass_idx('D2')
p, vin = cop_run(p, vin, list(range(cV1[0], cD2[0] + 1)), final_vx=DK['D2']['v0'])
p, vin, t_ = deck(C_, p, vin, 'D2', cD2, list(TB[cD2]), 'C')
cV2 = bass_idx('V2'); c84 = [k for k in bass_idx('A3', 15)][0]
p, vin = cop_run(p, vin, list(range(cV2[0], c84 + 1)), final_vz=-1.4)
COP_84 = (p, vin)

# ============================================================ FINALE ============================================================
# bar 84: both strike (thief G5 / cop G2), then on beat 2 both land at the same x,z on their own lanes; from there they move
# as mirror images: chord bells (Cm6, D7) -> the D7 bells are tilted inwards -> the two collide in mid-air -> both land on
# the crown of the great bell on the final G.
q = 4 * E['s16']
kq = lead_idx('A3', 15, 4)[0]; cq = [k for k in bass_idx('A3', 15)][1]
kc = lead_idx('coda'); cc = bass_idx('coda')
t84, t84q = TT[k84], TT[kq]; assert abs(TB[cq] - t84q) < 1e-6
t85, t85q, t86 = TT[kc[0]], TT[kc[1]], TT[kc[2]]
pT, vT = THIEF_84; pC, vC = COP_84
best = None
for X1 in np.arange(max(pT[0], pC[0]) + 0.3, max(pT[0], pC[0]) + 1.6, 0.02):
    for Z1 in np.arange(min(pT[2], pC[2]) - 1.2, min(pT[2], pC[2]) + 0.3, 0.02):
        ok = True; cost = 0; res = []
        for pp, vv in ((pT, vT), (pC, vC)):
            vo = np.array([(X1 - pp[0]) / q, 0, (Z1 - pp[2]) / q + 0.5 * G * q]); e, n = bounce(vv, vo)
            til = math.degrees(math.acos(min(1, n[2])))
            if not (EMIN <= e <= EMAX and til <= TILT and vo[2] - G * q < -0.3): ok = False; break
            va = vo + np.array([0, 0, -G * q])
            cost += (vo[0] - 2.0) ** 2 + 0.2 * (til / 40) ** 2 + 2.0 * max(0, 3.2 - np.linalg.norm(va)) ** 2; res.append((vo, e, n))
        if ok and (best is None or cost < best[0]): best = (cost, X1, Z1, res)
assert best, ('finale convergence infeasible', pT, vT, pC, vC)
_, X1, Z1, res = best
for ball, (pp, vv), (vo, e, n), i in ((T_, (pT, vT), res[0], k84), (C_, (pC, vC), res[1], c84)):
    ball.hit(i, t84, pp, n, e, 'tile' if ball is T_ else 'plate', vv, vo); ball.fly(t84, t84q, pp, vo)
vinT = res[0][0] + np.array([0, 0, -G * q]); vinC = res[1][0] + np.array([0, 0, -G * q])
# common motion from t84q: beam over (vx, dz) with both incoming velocities feasible
def common_hop(vins, dt, vx_t, dz_t, vy=0.0):
    best = None
    for vx in np.arange(1.2, 3.2, 0.02):
        for dz in np.arange(-1.2, 0.4, 0.01):
            vo = np.array([vx, 0, dz / dt + 0.5 * G * dt]); okk = True; ee = []
            for s_, vi in enumerate(vins):
                vo_s = vo + np.array([0, vy * (1 if s_ == 0 else -1), 0])
                e, n = bounce(vi, vo_s); til = math.degrees(math.acos(min(1, n[2])))
                if not (EMIN <= e <= EMAX and til <= TILT and vo[2] - G * dt < -0.3): okk = False; break
                ee.append((e, n))
            if okk:
                c = (vx - vx_t) ** 2 + 3 * (dz - dz_t) ** 2
                if best is None or c < best[0]: best = (c, vo, ee)
    assert best, ('common hop infeasible', vins)
    return best[1], best[2]
P_T = np.array([X1, YT, Z1]); P_C = np.array([X1, YC, Z1])
vo1, ee1 = common_hop([vinT, vinC], t85 - t84q, 2.0, -0.25)
T_.hit(kq, t84q, P_T, ee1[0][1], ee1[0][0], 'tile', vinT, vo1); T_.fly(t84q, t85, P_T, vo1)
C_.hit(cq, t84q, P_C, ee1[1][1], ee1[1][0], 'plate', vinC, vo1); C_.fly(t84q, t85, P_C, vo1)
dt = t85 - t84q
P_T = P_T + vo1 * dt + np.array([0, 0, -0.5 * G * dt * dt]); P_C = P_C + vo1 * dt + np.array([0, 0, -0.5 * G * dt * dt])
vin2 = vo1 + np.array([0, 0, -G * dt])
vo2, ee2 = common_hop([vin2, vin2], t85q - t85, 1.9, -0.15)
T_.hit(kc[0], t85, P_T, ee2[0][1], ee2[0][0], 'cbell', vin2, vo2); T_.fly(t85, t85q, P_T, vo2)
C_.hit(cc[0], t85, P_C, ee2[1][1], ee2[1][0], 'cbell', vin2, vo2); C_.fly(t85, t85q, P_C, vo2)
dt = t85q - t85
P_T = P_T + vo2 * dt + np.array([0, 0, -0.5 * G * dt * dt]); P_C = P_C + vo2 * dt + np.array([0, 0, -0.5 * G * dt * dt])
vin3 = vo2 + np.array([0, 0, -G * dt])
# D7 bells tilted inwards: converge at vy, collide at tcol when 2R apart, restitution EB along y
T_COL = t85q + 0.62 * (t86 - t85q); VY = (YC - R) / (T_COL - t85q); EB = 0.35
vo3, ee3 = common_hop([vin3, vin3], t86 - t85q, 1.6, -0.35, vy=VY)
voT = vo3 + np.array([0, VY, 0]); voC = vo3 + np.array([0, -VY, 0])
T_.hit(kc[1], t85q, P_T, ee3[0][1], ee3[0][0], 'cbell', vin3, voT); C_.hit(cc[1], t85q, P_C, ee3[1][1], ee3[1][0], 'cbell', vin3, voC)
T_.fly(t85q, T_COL, P_T, voT); C_.fly(t85q, T_COL, P_C, voC)
u = T_COL - t85q
QT = P_T + voT * u + np.array([0, 0, -0.5 * G * u * u]); QC = P_C + voC * u + np.array([0, 0, -0.5 * G * u * u])
assert abs((QC[1] - QT[1]) - 2 * R) < 1e-6
vcT = voT + np.array([0, 0, -G * u]); vcC = voC + np.array([0, 0, -G * u])
vaT = vcT * np.array([1, -EB, 1]); vaC = vcC * np.array([1, -EB, 1])
u2 = t86 - T_COL
T_.fly(T_COL, t86, QT, vaT); C_.fly(T_COL, t86, QC, vaC)
FT = QT + vaT * u2 + np.array([0, 0, -0.5 * G * u2 * u2]); FC = QC + vaC * u2 + np.array([0, 0, -0.5 * G * u2 * u2])
vfT = vaT + np.array([0, 0, -G * u2]); vfC = vaC + np.array([0, 0, -G * u2])
# land on the great bell's crown (flat top plate): plastic in z, they stop there side by side
T_.hit(kc[2], t86, FT, [0, 0, 1], 0.0, 'gbell', vfT, [0, 0, 0]); C_.hit(cc[2], t86, FC, [0, 0, 1], 0.0, 'gbell', vfC, [0, 0, 0])
T_.rest(t86, 200, FT); C_.rest(t86, 200, FC)
EVS['collide'] = dict(t=float(T_COL), p=((QT + QC) / 2).tolist(), vy=float(VY), e=EB)
EVS['gbell'] = dict(t=float(t86), x=float(FT[0]), z_top=float(FT[2] - R), yT=float(FT[1]), yC=float(FC[1]))
SFX.append(['clonk', float(T_COL)]); SFX.append(['gong', float(t86)])

# ============================================================ checks ============================================================
def check():
    info = {}
    hs = [(b.name, h) for b in (T_, C_) for h in b.hits]
    bb = [h for n, h in hs if not h.get('roll') and h['kind'] not in ('gbell',)]
    info['e_range'] = (round(min(h['e'] for h in bb), 3), round(max(h['e'] for h in bb), 3))
    info['speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for n, h in hs), 3)
    info['tilt_max'] = round(max(math.degrees(math.acos(min(1, h['n'][2]))) for n, h in hs), 1)
    info['thief_notes'] = (len([h for h in T_.hits]), len(LEAD)); info['cop_notes'] = (len(C_.hits), len(BASS))
    for b in (T_, C_):
        z = [h['p'][2] for h in b.hits]; info['z_' + b.name] = (round(min(z), 2), round(max(z), 2))
        x = [h['p'][0] for h in b.hits]; info['x_' + b.name] = (round(min(x), 2), round(max(x), 2))
    gaps = []
    for t in (3, 8, 14, 18, 22, 27, 30, 35, 42, 46, 50, 55, 60, 65, 68):
        gaps.append((t, round(T_.at(t)[0] - C_.at(t)[0], 2)))
    info['gap_thief_minus_cop'] = gaps
    return info


# tile width: never wider than the room around it (a tile must not reach under the neighbouring arcs)
BASEW = {'tile': 0.17, 'plate': 0.2, 'gut': 0.1, 'slat': 0.16, 'bell': 0.2, 'cbell': 0.2}
for b in (T_, C_):
    hs = sorted(b.hits, key=lambda h: h['t'])
    for k, h in enumerate(hs):
        w = BASEW.get(h['kind'], 0.17)
        if h['kind'] in ('tile', 'plate'):
            nb = [abs(hs[j]['p'][0] - h['p'][0]) for j in (k - 1, k + 1) if 0 <= j < len(hs)]
            w = float(np.clip(0.62 * min(nb), 0.08, w)) if nb else w
        h['w'] = w


def _frame(n):
    n = np.array(n, float); n /= np.linalg.norm(n)
    xa = np.cross([0, 1, 0], n); xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return np.stack([xa, ya, n], 1)
_ts = np.arange(0.0, float(TT[-1]) + 1.0, 1 / 240)
_P = {b.name: np.array([b.at(t) for t in _ts]) for b in (T_, C_)}
for b in (T_, C_):                              # shrink a tile until no ball (its own arcs or the other ball) grazes it
    for h in b.hits:
        if h['kind'] not in ('tile', 'plate'): continue
        n = np.array(h['n']); c = np.array(h['p']) - R * n; M = _frame(n); th = 0.03; ctr = c - n * th
        hd = 0.4 if h['kind'] == 'tile' else 0.425
        while h['w'] > 0.06:
            bad = False
            for nm_, Pn in _P.items():
                m = (np.abs(Pn[:, 0] - ctr[0]) < 0.5) & (np.abs(Pn[:, 2] - ctr[2]) < 0.5) & (np.abs(Pn[:, 1] - ctr[1]) < 0.9)
                if nm_ == b.name: m &= np.abs(_ts - h['t']) > 0.03
                if not m.any(): continue
                rel = (Pn[m] - ctr) @ M
                d = np.stack([np.maximum(0, np.abs(rel[:, 0]) - h['w'] / 2), np.maximum(0, np.abs(rel[:, 1]) - hd), np.maximum(0, np.abs(rel[:, 2]) - th)], 1)
                if (R - np.linalg.norm(d, axis=1) > 0.008).any(): bad = True; break
            if not bad: break
            h['w'] = round(h['w'] - 0.01, 3)


if __name__ == '__main__':
    for k, v in check().items(): print(k, v)
    out = dict(G=G, R=R, YT=YT, YC=YC, balls={b.name: dict(seg=b.seg, hits=b.hits) for b in (T_, C_)}, ev=EVS, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
