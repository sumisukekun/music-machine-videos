"""Gymnopedie No.1 - 'The Music Box Tower' (オルゴールの塔). Physics + story planner.
Run after arrange.py; writes layout.json (read by scene.py, render_audio.py, verify.py).

A glass music-box tower at night.  Two marbles wait in the lantern lift on its crown:
  NIGHT (blue, bass ball)  rolls out first and plays the accompaniment on the inner lane: beat 1 = bass bar, beat 2 = chime
                           plate (the chord), all the way down the spiral; its 4th chime plate is a lever that opens the
                           pearl's flap.
  PEARL (white, melody)    rolls out on the outer lane and plays the melody on glass keys.  Every struck key stays lit.
  Long notes: the E4 (9 beats) and the two D5 (5 beats) are lever keys: the pearl bounces into the saddle cup of a
  carousel horse that carries it around the tower and drops it onto the next note exactly on the beat.
  Meanwhile the empty lift sinks inside the glass core, always level with the marbles.
  Final chord: both marbles jump through the arch at the foot of the tower into the lift's two cups; every key of the
  tower flares up; the lift carries both marbles back up to the crown -> the last frame is the first frame (loop).
Physics: constant gravity G; free flight = parabola in world space; bounces along the key normal with EMIN <= e <= EMAX
(never faster); chutes roll with a = (5/7) g sin(a) - c_rr g cos(a); the carousel horses and the lift move with trapezoid
speed profiles."""
import json, os, math, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from helix import *
RP_, RN_ = 0.2, 0.25                       # pearl / night radii
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json')); NIGHT = E['night']
T0, BEAT, BAR, FINAL = E['t0'], E['beat'], E['bar'], E['final']
def tq(v, b): return T0 + (v - 1) * BAR + b * BEAT
TP = np.array([l[0] for l in LEAD]); TNI = np.array([l[0] for l in NIGHT])
R_N, R_P, R_C, PSI = 3.0, 4.0, 1.2, 0.45   # lane radii, lift cup radius, lift cup azimuths (+- PSI)
PITCH = 6.5                                # the spiral drops 6.5 m per turn
DZ_P = -0.7                                # the pearl lane runs 0.7 m below the night lane
Z_OFF = -3.0                               # the first bounce drops the night ball ~3 m below the crown
def zc(th): return Z_OFF - PITCH * th / (2 * math.pi)
TM = tq(3, 1)                              # first melody note
TV28 = tq(28, 0)
TH_END = 6 * math.pi                       # three turns
W_C = (TH_END - 0.25 - 0.35) / (TV28 - TM)
def sstep(u): u = min(1.0, max(0.0, u)); return u * u * (3 - 2 * u)
def thn(t):
    if t <= TM: return 0.05 + 0.30 * max(0.0, t - T0) / (TM - T0)
    return 0.35 + W_C * (t - TM)
def thp(t): return thn(t) - 0.05 - 0.65 * sstep((t - tq(26, 0)) / (2 * BAR))   # the pearl falls back a little at the end
SFX = []; EV = dict(ramps=[], rides=[], levers=[])
STRUCK = {}

# ---------------------------------------------------------------- the lift (car) inside the glass core
Z_H = 1.34                                 # ball centre in the lift cups at the crown
def z_follow(t): return zc(thn(t)) - 1.0
T_SINK0, T_SINK1 = 8.2, 13.2
T_UP0, T_UP1 = FINAL + 1.9, FINAL + 7.3
Z_END = None                               # set by the night's end chute (below)
def z_car(t):
    if t <= T_SINK0: return Z_H
    if t <= FINAL - 3.0: return Z_H + (z_follow(t) - Z_H) * sstep((t - T_SINK0) / (T_SINK1 - T_SINK0))
    if t <= FINAL: zf = z_follow(FINAL - 3.0); return zf + (Z_END - zf) * sstep((t - (FINAL - 3.0)) / 2.6)
    if t <= T_UP0: return Z_END
    return Z_END + UP_S(t - T_UP0)
def cup_pos(name, t):
    a = -PSI if name == 'pearl' else PSI
    return cyl(a, R_C, z_car(t))

# ---------------------------------------------------------------- chutes from the lift cups (crown)
ALPHA = math.radians(30.0); CRR = 0.04
A_RAMP = 5 / 7 * G * math.sin(ALPHA) - CRR * G * math.cos(ALPHA)
FLAP_DELAY = 0.06


def chute(ball, aim_xy, t_open, t_first, d_lip):
    """the ball rolls from rest out of its lift cup down a straight chute (azimuth towards aim_xy) and flies onto its
    first key at t_first; returns (p_contact, v_in)"""
    head = cup_pos(ball.name, 0.0)
    u = np.array([aim_xy[0] - head[0], aim_xy[1] - head[1], 0.0]); u /= np.linalg.norm(u)
    dirv = np.array([u[0] * math.cos(ALPHA), u[1] * math.cos(ALPHA), -math.sin(ALPHA)])
    t0 = t_open + FLAP_DELAY; Tr = t_first - t0 - d_lip
    L = 0.5 * A_RAMP * Tr * Tr; vl = A_RAMP * Tr
    tt = np.arange(t0, t0 + Tr + 1e-9, 1 / 240.0)
    ball.rest(0.0, t0, head)
    ball.path(tt, np.array([head + 0.5 * A_RAMP * (t - t0) ** 2 * dirv for t in tt]))
    lip = head + L * dirv; vlip = vl * dirv
    ball.fly(t0 + Tr, t_first, lip, vlip)
    p1 = lip + vlip * d_lip + np.array([0, 0, -0.5 * G * d_lip ** 2]); v1 = vlip + np.array([0, 0, -G * d_lip])
    EV['ramps'].append(dict(ball=ball.name, head=head.tolist(), lip=lip.tolist(), dir=dirv.tolist(), alpha=ALPHA, L=L, t_open=t_open,
                            t_roll=t0, t_lip=t0 + Tr, v_lip=vl, a=A_RAMP))
    SFX.append(['flap', t_open])
    return p1, v1


PEARL = Ball('pearl', RP_); NIGHTB = Ball('night', RN_)
BALLS = {'pearl': PEARL, 'night': NIGHTB}
EXT_P = (0.23, 0.3, 0.07)                  # pearl key: half length along the course, half depth (radial), half thickness
EXT_N = (0.28, 0.32, 0.08)                 # night bass bar / chime plate


def record(ball, hits, ids, src, kinds=None, t_last=None):
    for k, h in enumerate(hits):
        i = ids[k]; ps = src[i][1]
        kind = (kinds or {}).get(i, src[i][4])
        ball.hit(h['t'], h['p'], h['vins'][0], h['vout'], kind, ps, ev=i)
        STRUCK.setdefault((ball.name, i), []).append(ps)
        t1 = hits[k + 1]['t'] if k + 1 < len(hits) else t_last
        ball.fly(h['t'], t1, h['p'], h['vout'])


# ================================================================ NIGHT: one bass bar per bar (2.28 s hops); every bass bar is
# a see-saw lever: its inner arm carries a hammer that the ball's impact kicks up; the hammer falls back onto the chime
# plate (the chord) exactly one beat later (two hammers in video bar 27: one beat and two beats).
IB = [i for i, l in enumerate(NIGHT) if l[4] in ('bass', 'final')]
TB = TNI[IB]
p1, v1 = chute(NIGHTB, cyl(0.12, 3.0, 0), 0.15, TB[0], 0.9)
import pickle
_CK = '/tmp/gymno_night.pkl'
if os.environ.get('CACHE') and os.path.exists(_CK): hits_n, pEn, vEn = pickle.load(open(_CK, 'rb'))
else:
    import time as _time; _t0 = _time.time()
    hits_n, pEn, vEn, _, _ = plan_retry(TB[:-1], p1, [v1], thn, lambda t: zc(thn(t)), lambda t: R_N, RN_, EXT_N,
                                        wlim=(0.12, 0.6), dzr=(-1.6, 1.0), land_min=1.0)
    print('night planned', round(_time.time() - _t0, 1), 's')
    pickle.dump((hits_n, pEn, vEn), open(_CK, 'wb'))
record(NIGHTB, hits_n, IB[:-2], NIGHT, t_last=TB[-2])
# hammers: chord onset -> the bass bar of its bar
HAM = []
for i, l in enumerate(NIGHT):
    if l[4] != 'chord': continue
    ib = max(j for j in IB if NIGHT[j][5] == l[5])
    HAM.append(dict(i=i, ib=ib, t_kick=float(NIGHT[ib][0]), t=float(l[0]), dt=float(l[0] - NIGHT[ib][0]), pitches=l[1], bar=l[5]))
    STRUCK.setdefault(('night', i), []).append(l[1])
EV['hammers'] = HAM
# the pearl's flap is opened by the night ball's 2nd hammer stroke (video bar 2, beat 2: the D-major chord)
H_LEVER = HAM[1]
assert abs(H_LEVER['t'] - tq(2, 1)) < 1e-6
EV['levers'].append(dict(ball='night', i=H_LEVER['i'], t=H_LEVER['t'], kind='hammer'))


# ================================================================ PEARL
def pidx(v0, b0, v1_, b1):
    return [i for i, l in enumerate(LEAD) if (v0, b0) <= (l[5], l[6]) <= (v1_, b1)]


def ride(ball, p_from, t_from, p_to, t_to, horse, lift=0.55, bob=0.08):
    """carousel horse: the saddle cup carries the ball from p_from to p_to around the tower (trapezoid speed along the path,
    the horse bobs gently)"""
    a0 = math.atan2(p_from[1], p_from[0]); a1 = math.atan2(p_to[1], p_to[0])
    while a1 < a0: a1 += 2 * math.pi
    r0, r1 = np.hypot(*p_from[:2]), np.hypot(*p_to[:2])
    s = np.linspace(0, 1, 600)
    th = a0 + (a1 - a0) * s; rr = r0 + (r1 - r0) * s
    zz = p_from[2] + (p_to[2] - p_from[2]) * s + lift * np.sin(np.pi * s) + bob * np.sin(4 * np.pi * s) * np.sin(np.pi * s)
    pts = np.stack([rr * np.cos(th), rr * np.sin(th), zz], 1)
    sl = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    T = t_to - t_from; sf, vc = trapezoid(sl[-1], T, min(1.1, T / 3))
    tt = np.arange(t_from, t_to + 1e-9, 1 / 240.0)
    ps = np.array([[np.interp(sf(t - t_from), sl, pts[:, c]) for c in range(3)] for t in tt])
    ball.path(tt, ps)
    EV['rides'].append(dict(horse=horse, t0=float(t_from), t1=float(t_to), ts=tt[::4].tolist(), ps=ps[::4].tolist(), vmax=float(vc),
                            route=pts[::6].tolist()))
    SFX.append(['ride_go', float(t_from)])


H_DROP = 1.0; T_DROP = math.sqrt(2 * H_DROP / G)


def lever_ride(i_lev, pL, vL, t_next, th_next, z_next, horse):
    """lever key -> the pearl bounces into the waiting horse's saddle cup, rides, is dropped onto the next key"""
    t = TP[i_lev]; ps = LEAD[i_lev][1]
    thL = math.atan2(pL[1], pL[0])
    tgt = cyl(thL + 0.62 / R_P, R_P + 0.05, pL[2] + 0.12)
    d = 0.5
    vo, pc, vcin = exit_to(pL, vL, d, tgt, spread=0.25, nz=21)
    PEARL.hit(t, pL, vL, vo, 'lever', ps, ev=i_lev); STRUCK.setdefault(('pearl', i_lev), []).append(ps)
    PEARL.fly(t, t + d, pL, vo)
    PEARL.hits.append(dict(t=float(t + d), p=pc.tolist(), vin=vcin.tolist(), vout=[0, 0, 0], n=[0, 0, 1], e=0.0, kind='saddle', pitch=None, ev=None))
    SFX.append(['saddle', float(t + d)])
    p_next = cyl(th_next, R_P, z_next); p_hover = p_next + [0, 0, H_DROP]
    t_go = t + d + 0.35; t_arr = t_next - T_DROP - 0.3
    PEARL.rest(t + d, t_go, pc)
    ride(PEARL, pc, t_go, p_hover, t_arr, horse)
    PEARL.rest(t_arr, t_next - T_DROP, p_hover)
    PEARL.fly(t_next - T_DROP, t_next, p_hover, [0, 0, 0])
    SFX.append(['tip', float(t_next - T_DROP)])
    EV['levers'].append(dict(ball='pearl', i=i_lev, t=float(t), p=pL.tolist(), n=PEARL.hits[-2]['n'], horse=horse,
                             saddle=pc.tolist(), t_saddle=float(t + d), hover=p_hover.tolist(), t_drop=float(t_next - T_DROP)))
    return p_next, np.array([0, 0, -G * T_DROP])


aim = cyl(-0.55, 4.15, 0)
p1, v1 = chute(PEARL, aim, H_LEVER['t'], TP[0], 0.46)
segs = [((3, 1), (9, 0), 'A'), ((12, 0), (15, 0), 'B'), ((16, 2), (20, 0), 'C'), ((21, 2), (28, 0), None)]
hist = (); keys = ()
p_first, v_first = p1, v1
for (a, b, horse) in segs:
    ids = pidx(a[0], a[1], b[0], b[1])
    ts = TP[ids]
    zt = lambda t: zc(thp(t)) + DZ_P
    if horse is None:
        hits, pE, vE, hist, keys = plan_retry(ts, p_first, [v_first], thp, zt, lambda t: R_P, RP_, EXT_P,
                                              wlim=(0.1, 0.8), dzr=(-1.6, 1.0), hist0=hist, keys0=keys)
        record(PEARL, hits, ids[:-1], LEAD, t_last=ts[-1])
        P28, V28 = pE, vE
        break
    hits, pE, vE, hist, keys = plan_retry(ts, p_first, [v_first], thp, zt, lambda t: R_P, RP_, EXT_P, wlim=(0.1, 0.8), dzr=(-1.6, 1.0),
                                          hist0=hist, keys0=keys)
    record(PEARL, hits, ids[:-1], LEAD, t_last=ts[-1])
    i_lev = ids[-1]
    nxt = [s for s in segs if s[0] > b][0][0]
    t_next = tq(*nxt)
    p_first, v_first = lever_ride(i_lev, pE, vE, t_next, thp(t_next), zc(thp(t_next)) + DZ_P, horse)
    hist = (); keys = ()

# ================================================================ the lift's final height: the pearl's last jump (2.28 s) from its
# chord plate needs e ~ 0.85
D_F = FINAL - TP[-2]
cupxy = cyl(-PSI, R_C, 0.0)
vz_need = 0.85 * abs(V28[2])
Z_END = float(P28[2] + (vz_need - 0.5 * G * D_F) * D_F)
print('Z_END', round(Z_END, 3), 'pearl v28 z', round(P28[2], 3))
# last bass bar = lever: the night ball bounces into the end-chute cradle; the chute drops it into its lift cup on the final chord
I_LAST = IB[-2]
t = TNI[I_LAST]; D_CR = 0.5
thL = math.atan2(pEn[1], pEn[0])
from scipy.optimize import brentq
pcup = cyl(PSI, R_C, 0.0)                   # lift cup (x, y); height Z_END
t_open = t + D_CR + 0.12; TT = FINAL - t_open - FLAP_DELAY
def acc(al): return 5 / 7 * G * math.sin(al) - CRR * G * math.cos(al)
def chute_fit(pcr):
    """end chute from the cradle towards the lift cup: roll time Tr and slope alpha so that the ball lands in the cup
    (x, y, z) exactly at FINAL"""
    hz = np.array([pcup[0] - pcr[0], pcup[1] - pcr[1], 0.0]); dist = np.linalg.norm(hz); hz /= dist
    def tr_of(al):
        a_ = acc(al); f = lambda Tr: 0.5 * a_ * Tr * Tr * math.cos(al) + a_ * Tr * math.cos(al) * (TT - Tr) - dist
        if f(TT - 0.02) < 0: return None
        return brentq(f, 0.02, TT - 0.02)
    def zl(al):
        Tr = tr_of(al)
        if Tr is None: return None
        a_ = acc(al); L = 0.5 * a_ * Tr * Tr; vl = a_ * Tr; dl = TT - Tr
        return pcr[2] - L * math.sin(al) - vl * math.sin(al) * dl - 0.5 * G * dl * dl
    als = np.radians(np.arange(6.0, 50.0, 0.5)); zs = [zl(a) for a in als]
    for k in range(len(als) - 1):
        if zs[k] is None or zs[k + 1] is None: continue
        if (zs[k] - Z_END) * (zs[k + 1] - Z_END) <= 0:
            al = brentq(lambda a: zl(a) - Z_END, als[k], als[k + 1], xtol=1e-12)
            Tr = tr_of(al); return al, Tr, hz
    return None
fit = None
for cdz in (0.1, -0.3, -0.7, -1.1, -1.5):
    for cdr in (0.35, 0.6, 0.85):
        vo, pcr, vcin = exit_to(pEn, vEn, D_CR, cyl(thL + 0.12, R_N - cdr, pEn[2] + cdz), spread=0.25, nz=21)
        fit = chute_fit(pcr)
        if fit: break
    if fit: break
assert fit, 'no end chute'
al, Tr, hz = fit
a_ = acc(al); L = 0.5 * a_ * Tr * Tr; vl = a_ * Tr; dl = TT - Tr
print('cradle', np.round(pcr, 2), 'end chute', round(math.degrees(al), 2), 'deg, roll', round(Tr, 3), 's, flight', round(dl, 3), 's')
NIGHTB.hit(t, pEn, vEn, vo, 'lever', NIGHT[I_LAST][1], ev=I_LAST); STRUCK.setdefault(('night', I_LAST), []).append(NIGHT[I_LAST][1])
NIGHTB.fly(t, t + D_CR, pEn, vo)
NIGHTB.hits.append(dict(t=float(t + D_CR), p=pcr.tolist(), vin=vcin.tolist(), vout=[0, 0, 0], n=[0, 0, 1], e=0.0, kind='cradle', pitch=None, ev=None))
SFX.append(['cradle', float(t + D_CR)])
a_ = 5 / 7 * G * math.sin(al) - CRR * G * math.cos(al)
dirv = np.array([hz[0] * math.cos(al), hz[1] * math.cos(al), -math.sin(al)])
t0r = t_open + FLAP_DELAY
NIGHTB.rest(t + D_CR, t0r, pcr)
tt = np.arange(t0r, t0r + Tr + 1e-9, 1 / 240.0)
NIGHTB.path(tt, np.array([pcr + 0.5 * a_ * (u - t0r) ** 2 * dirv for u in tt]))
lip = pcr + L * dirv; vlip = vl * dirv
NIGHTB.fly(t0r + Tr, FINAL, lip, vlip)
pEn2 = lip + vlip * dl + np.array([0, 0, -0.5 * G * dl * dl]); vEn2 = vlip + np.array([0, 0, -G * dl])
EV['endchute'] = dict(head=pcr.tolist(), lip=lip.tolist(), dir=dirv.tolist(), alpha=al, L=L, t_open=t_open, t_roll=t0r, t_lip=t0r + Tr, v_lip=vl, a=a_)
SFX.append(['flap', float(t_open)])
UP_S, UP_V = trapezoid(Z_H - Z_END, T_UP1 - T_UP0, 1.6)
EV['lift'] = dict(z_h=Z_H, sink=[T_SINK0, T_SINK1], up=[T_UP0, T_UP1], z_end=Z_END, vmax_up=UP_V, r_c=R_C, psi=PSI)
NIGHTB.hit(FINAL, pEn2, vEn2, [0, 0, 0], 'final', NIGHT[-1][1], ev=len(NIGHT) - 1)
STRUCK.setdefault(('night', len(NIGHT) - 1), []).append(NIGHT[-1][1])
# pearl's last jump into its lift cup
pcp = cup_pos('pearl', FINAL)
vo = (pcp - P28) / D_F + np.array([0, 0, 0.5 * G * D_F])
assert legal(V28, vo), ('pearl final jump', bounce_e(V28, vo))
i28 = len(LEAD) - 2
PEARL.hit(TP[i28], P28, V28, vo, 'chord', LEAD[i28][1], ev=i28); STRUCK.setdefault(('pearl', i28), []).append(LEAD[i28][1])
PEARL.fly(TP[i28], FINAL, P28, vo)
PEARL.hit(FINAL, pcp, vo + np.array([0, 0, -G * D_F]), [0, 0, 0], 'final', LEAD[-1][1], ev=len(LEAD) - 1)
STRUCK.setdefault(('pearl', len(LEAD) - 1), []).append(LEAD[-1][1])

# ================================================================ the final: both marbles in the lift, lift goes up
for nme, b in BALLS.items():
    pc = cup_pos(nme, FINAL)
    b.rest(FINAL, T_UP0, pc)
    tt = np.arange(T_UP0, T_UP1 + 1e-9, 1 / 240.0)
    b.path(tt, np.array([cup_pos(nme, t) for t in tt]))
    b.rest(T_UP1, T_UP1 + 30, cup_pos(nme, T_UP1))
SFX += [['final', float(FINAL)], ['lift_go', float(T_UP0)], ['lift_stop', float(T_UP1)]]
EV['t_end'] = float(T_UP1 + 1.3)


def check():
    info = {}
    hs = [(n, h) for n, b in BALLS.items() for h in b.hits if h['kind'] not in ('saddle', 'final', 'cradle')]
    ee = [h['e'] for n, h in hs]
    info['e_range'] = (round(min(ee), 3), round(max(ee), 3))
    info['speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for n, h in hs), 3)
    info['tilt_max_deg'] = round(max(math.degrees(math.acos(min(1, h['n'][2]))) for n, h in hs), 1)
    info['uncovered_pearl'] = sorted(set(range(len(LEAD))) - set(i for n, i in STRUCK if n == 'pearl'))
    info['uncovered_night'] = sorted(set(range(len(NIGHT))) - set(i for n, i in STRUCK if n == 'night'))
    info['hammers'] = len(HAM)
    for n, b in BALLS.items():
        zz = [h['p'][2] for h in b.hits]; rr = [np.hypot(*h['p'][:2]) for h in b.hits]
        info[n] = dict(hits=len(b.hits), z=(round(min(zz), 2), round(max(zz), 2)), r=(round(min(rr), 2), round(max(rr), 2)))
    info['ramps'] = [(r['ball'], round(r['L'], 2), round(r['v_lip'], 2)) for r in EV['ramps']]
    info['ride_vmax'] = [round(g['vmax'], 2) for g in EV['rides']]
    info['lift'] = dict(z_end=round(Z_END, 2), vmax_up=round(UP_V, 2))
    return info


if __name__ == '__main__':
    info = check()
    for k, v in info.items(): print(k, v)
    out = dict(G=G, R=dict(pearl=RP_, night=RN_), geo=dict(r_n=R_N, r_p=R_P, pitch=PITCH, z_off=Z_OFF, t0=T0, dz_p=DZ_P, th_end=TH_END, w_c=W_C, tm=TM),
               ext=dict(pearl=EXT_P, night=EXT_N),
               balls={n: dict(r=b.r, seg=b.seg, hits=b.hits) for n, b in BALLS.items()},
               struck={f'{k[0]}:{k[1]}': v for k, v in STRUCK.items()}, ev=EV, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
