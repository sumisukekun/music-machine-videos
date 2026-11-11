"""Morning Mood - '夜明けの山の村' (a mountain village before dawn). Physics + story planner.
Run after arrange.py; writes layout.json (read by scene.py, render_audio.py, verify.py).

Cast: the FLUTE ball (pale glass, falls from the sky as the morning star) and the OBOE ball (amber wood) call to each other:
  each plays its phrase on glass / wooden keys while the other waits at the top of a little ramp behind a closed gate.
  The last note of every phrase is a LEVER key: striking it pulls the gate open, the waiting ball rolls down the ramp and
  lands on its first note exactly on the downbeat. The ball that has finished bounces into a basket, and a ROPEWAY gondola
  (driven by the windmill) carries it ahead to the next ramp. Behind them the DAWN ball (copper) rolls along the back rail
  at a steady speed and pushes through BELL GATES: every gate rings one accompaniment chord.
  bars 17-20 the calls get shorter - the flute plays on alone down the hillside, the oboe rides the last gondola,
  bar 21 the gondola drops the oboe right next to the flute: the theme in octaves, both together (tutti),
  final chord: both land in the two cups of the sunrise lever, the dawn ball rolls off the rail into the latch funnel, the
  counterweight boulder drops and the SUN - a giant gold marble - is hauled up over the mountains.
Physics: constant gravity G; free flight = parabola; bounces along the key normal with EMIN <= e <= EMAX (never faster);
ramps roll with a = (5/7) g sin(a) - c_rr g cos(a); the dawn rail is inclined so that rolling resistance balances gravity
(constant speed); the gondola moves with a trapezoid speed profile; the sun rises on a counterweight (constant acceleration,
then a brake)."""
import json, os, math, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from planlib import *
import planlib
RB = 0.2                     # flute / oboe
RD = 0.26                    # dawn ball
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
TN = np.array([l[0] for l in LEAD]); NN = len(TN)
T0, E8, BAR, FINAL = E['t0'], E['e8'], E['bar'], E['final']
V = 1.4
SLOPE = 0.049                # course descends 4.9 cm per second of music (2 deg at V)
def XT(t): return V * (t - T0)
def ZT(t): return -SLOPE * (t - T0)
SFX = []; EV = dict(levers=[], ramps=[], baskets=[], gondolas=[], wake={})
CT = math.cos(math.radians(TILT))


def idx_bars(b0, b1): return [i for i, l in enumerate(LEAD) if b0 <= l[5] <= b1]
def tbar(b, off=0.0): return T0 + (b - 1) * BAR + off * E8


def legal(vin, vout, emin=EMIN, emax=EMAX):
    e, n = bounce_e(np.array(vin, float), np.array(vout, float))
    return emin <= e <= emax and n[2] >= CT


def exit_bounce(p, vin, d, want, vy=None, dzr=(-1.6, 0.4), vxr=(0.2, 2.2)):
    """legal outgoing velocity from contact p (incoming vin) landing after d seconds as close as possible to p + want"""
    best = None
    vy = want[1] / d if vy is None else vy
    for dz in np.linspace(dzr[0], dzr[1], 81):
        for vx in np.linspace(vxr[0], vxr[1], 61):
            vo = np.array([vx, vy, dz / d + 0.5 * G * d])
            if not legal(vin, vo): continue
            c = (vx * d - want[0]) ** 2 + (dz - want[2]) ** 2
            if best is None or c < best[0]: best = (c, vo)
    assert best is not None, ('no legal exit', p, vin, d, want)
    vo = best[1]; pl = np.array(p) + vo * d + np.array([0, 0, -0.5 * G * d * d])
    return vo, pl, vo + np.array([0, 0, -G * d])


FL = Ball('flute', RB); OB = Ball('oboe', RB)
BALLS = {'flute': FL, 'oboe': OB}
STRUCK = {}


def hop_phrase(ball, ids, p0, vin, final=None, yt=lambda t: 0.0, kind='key', **kw):
    """plan the hops of one ball over lead events ids; returns (p_last, vin_last) at the last event (not yet bounced)"""
    ts = TN[ids]
    hits, pE, vE = plan_retry(ts, p0, [vin], lambda t: XT(t), lambda t: ZT(t), yt, R=RB, final=final, **kw)
    for k, h in enumerate(hits):
        i = ids[k]; p = LEAD[i][1][0]
        ball.hit(h['t'], h['p'], h['vins'][0], h['vout'], kind, p, ev=i)
        STRUCK.setdefault(i, []).append((ball.name, p, kind))
        t1 = hits[k + 1]['t'] if k + 1 < len(hits) else ts[-1]
        ball.fly(h['t'], t1, h['p'], h['vout'])
    return pE, vE


# ramp: alpha, rolling resistance; the waiting ball rolls from rest when the gate opens
ALPHA = math.radians(24.0); CRR = 0.04
A_RAMP = 5 / 7 * G * math.sin(ALPHA) - CRR * G * math.cos(ALPHA)
GATE_DELAY = 0.06            # gate lifts 0.06 s after the lever key is struck
D_LIP = 0.3                  # flight from the ramp lip to the first key


def make_ramp(ball, p_lever, t_open, t_first, side=+1):
    """ramp next to the lever key; returns (p_first, vin_first) of the ball's landing on its first key at t_first"""
    Tr = t_first - t_open - GATE_DELAY - D_LIP
    L = 0.5 * A_RAMP * Tr * Tr; vl = A_RAMP * Tr
    ca, sa = math.cos(ALPHA), math.sin(ALPHA)
    head = np.array([p_lever[0] + 0.42, p_lever[1], p_lever[2] + 1.05])
    lip = head + L * np.array([ca, 0, -sa])
    v_lip = vl * np.array([ca, 0, -sa])
    t0 = t_open + GATE_DELAY
    tt = np.arange(t0, t0 + Tr + 1e-9, 1 / 240.0)
    ps = np.array([head + 0.5 * A_RAMP * (t - t0) ** 2 * np.array([ca, 0, -sa]) for t in tt])
    ball.path(tt, ps)
    ball.fly(t0 + Tr, t_first, lip, v_lip)
    d = t_first - t0 - Tr
    p1 = lip + v_lip * d + np.array([0, 0, -0.5 * G * d * d]); v1 = v_lip + np.array([0, 0, -G * d])
    EV['ramps'].append(dict(ball=ball.name, head=head.tolist(), lip=lip.tolist(), alpha=ALPHA, L=L, t_open=t_open, t_roll=t0,
                            t_lip=t0 + Tr, v_lip=vl, a=A_RAMP))
    SFX.append(['gate', t_open + 0.02])
    return head, p1, v1


def lever_and_basket(ball, i_lever, pL, vL):
    """the last note of a phrase: lever key -> the ball bounces into the basket on the near side"""
    t = TN[i_lever]; p = LEAD[i_lever][1][0]
    d = 0.48
    vo, pb, vbin = exit_bounce(pL, vL, d, (0.55, -0.62, -0.45), dzr=(-1.0, 0.0))
    ball.hit(t, pL, vL, vo, 'lever', p, ev=i_lever); STRUCK.setdefault(i_lever, []).append((ball.name, p, 'lever'))
    ball.fly(t, t + d, pL, vo)
    ball.hits.append(dict(t=float(t + d), p=pb.tolist(), vin=vbin.tolist(), vout=[0, 0, 0], n=[0, 0, 1], e=0.0, kind='basket', pitch=None, ev=None))
    EV['levers'].append(dict(t=float(t), p=pL.tolist(), n=ball.hits[-2]['n'], ball=ball.name, pitch=p))
    EV['baskets'].append(dict(ball=ball.name, p=pb.tolist(), t=float(t + d)))
    SFX.append(['basket', float(t + d)])
    return pb, t + d


def trapezoid(L, T, ta):
    vc = L / (T - ta)
    def s(u):
        if u <= 0: return 0.0
        if u >= T: return L
        if u < ta: return 0.5 * vc / ta * u * u
        if u > T - ta: w = T - u; return L - 0.5 * vc / ta * w * w
        return 0.5 * vc * ta + vc * (u - ta)
    return s, vc


def gondola(ball, p_from, t_from, p_to, t_to, ycab=1.35, hz=2.0, drop=False):
    """the ball rides a gondola bucket from p_from to p_to; route: up, back to the cable lane, along, forward, down"""
    zc = max(p_from[2], p_to[2]) + hz
    W = [np.array(p_from), np.array(p_from) + [0.0, 0.0, 0.45], np.array([p_from[0] + 0.5, ycab, zc]),
         np.array([p_to[0] - 0.5, ycab, zc]), np.array(p_to) + [0.0, 0.0, 0.45 if not drop else 0.0], np.array(p_to)]
    if drop: W = W[:-1]
    # dense polyline with rounded corners (Catmull-Rom through the waypoints)
    P = [W[0]] + W + [W[-1]]
    pts = []
    for k in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[k - 1], P[k], P[k + 1], P[k + 2]
        for u in np.linspace(0, 1, 60, endpoint=False):
            pts.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    pts.append(W[-1]); pts = np.array(pts)
    sl = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    T = t_to - t_from; s, vc = trapezoid(sl[-1], T, min(1.2, T / 3))
    tt = np.arange(t_from, t_to + 1e-9, 1 / 240.0)
    ps = np.array([[np.interp(s(t - t_from), sl, pts[:, c]) for c in range(3)] for t in tt])
    ball.rest(t_from - 0.6 if False else ball.seg[-1]['t1'], t_from, p_from)
    ball.path(tt, ps)
    EV['gondolas'].append(dict(ball=ball.name, t0=float(t_from), t1=float(t_to), ts=tt[::8].tolist(), ps=ps[::8].tolist(),
                               route=pts[::3].tolist(), vmax=float(vc), drop=drop))
    SFX.append(['gond_go', float(t_from)]); SFX.append(['gond_end', float(t_to)])


# ================================================== P1: the morning star falls, flute bars 1-4
H_STAR = 3.6
ids = idx_bars(1, 4)
p_star = np.array([0.0, 0.0, H_STAR])
T_FALL = math.sqrt(2 * (H_STAR) / G)
FL.rest(0.0, TN[0] - T_FALL, p_star); FL.fly(TN[0] - T_FALL, TN[0], p_star, [0, 0, 0])
EV['star'] = dict(p=p_star.tolist(), t_fall=float(TN[0] - T_FALL))
SFX.append(['star', float(TN[0] - T_FALL)])
pL, vL = hop_phrase(FL, ids, (0, 0, 0), np.array([0, 0, -G * T_FALL]))
iL = ids[-1]
pb1, tb1 = lever_and_basket(FL, iL, pL, vL)
head1, p_first, v_first = make_ramp(OB, pL, TN[iL], tbar(5))
OB.seg.insert(0, dict(k='rest', t0=0.0, t1=float(TN[iL] + GATE_DELAY), p=head1.tolist()))

# ================================================== P2: oboe bars 5-8
ids = idx_bars(5, 8)
pL, vL = hop_phrase(OB, ids, p_first, v_first)
iL = ids[-1]
pb2, tb2 = lever_and_basket(OB, iL, pL, vL)
# the flute rode the gondola from basket 1 to the head of ramp 2
head2 = np.array([pL[0] + 0.42, pL[1], pL[2] + 1.05])
gondola(FL, pb1, tb1 + 0.7, head2, TN[iL] - 0.9)
FL.rest(TN[iL] - 0.9, TN[iL] + GATE_DELAY, head2)
h_, p_first, v_first = make_ramp(FL, pL, TN[iL], tbar(9))
assert np.allclose(h_, head2)
EV['wake']['windmill'] = float(tb1 + 0.7)          # the windmill starts with the first gondola

# ================================================== P3: flute bars 9-12
ids = idx_bars(9, 12)
pL, vL = hop_phrase(FL, ids, p_first, v_first)
iL = ids[-1]
pb3, tb3 = lever_and_basket(FL, iL, pL, vL)
head3 = np.array([pL[0] + 0.42, pL[1], pL[2] + 1.05])
gondola(OB, pb2, tb2 + 0.7, head3, TN[iL] - 0.9)
OB.rest(TN[iL] - 0.9, TN[iL] + GATE_DELAY, head3)
h_, p_first, v_first = make_ramp(OB, pL, TN[iL], tbar(13))
EV['wake']['rooster'] = float(tbar(9))

# ================================================== P4: oboe bars 13-16 (each note lights a window)
ids = idx_bars(13, 16)
pL, vL = hop_phrase(OB, ids, p_first, v_first)
EV['wake']['windows'] = [dict(t=float(h['t']), x=float(h['p'][0]), z=float(h['p'][2])) for h in OB.hits if h['ev'] in ids]
iL = ids[-1]
EV['wake']['windows'].append(dict(t=float(TN[iL]), x=float(pL[0]), z=float(pL[2])))
pb4, tb4 = lever_and_basket(OB, iL, pL, vL)
head4 = np.array([pL[0] + 0.42, pL[1], pL[2] + 1.05])
gondola(FL, pb3, tb3 + 0.7, head4, TN[iL] - 0.9)
FL.rest(TN[iL] - 0.9, TN[iL] + GATE_DELAY, head4)
h_, p_first, v_first = make_ramp(FL, pL, TN[iL], tbar(17))

# ================================================== P5: flute alone bars 17-20, down the hillside; lands at the reunion
OFF = {'flute': np.array([+0.25, -0.22, 0.0]), 'oboe': np.array([-0.25, +0.22, 0.0])}
ids = idx_bars(17, 20)
t21 = tbar(21)
pc = np.array([XT(t21), 0.0, ZT(t21) - 0.9])                      # reunion contact centre of the pair
# bars 19-20: the falling arpeggio -> the course drops faster (down the slope to the village square)
zt5 = lambda t: ZT(t) - 0.9 * min(1, max(0, (t - tbar(19)) / (2 * BAR)))
ts = np.append(TN[ids], t21)
hits, pE, vE = plan_retry(ts, p_first, [v_first], XT, zt5, lambda t: OFF['flute'][1] * min(1, max(0, (t - tbar(20)) / BAR)),
                          R=RB, final=(pc[0] + OFF['flute'][0], pc[2]), vxlim=(0.5, 3.6))
for k, h in enumerate(hits):
    i = ids[k]; p = LEAD[i][1][0]
    FL.hit(h['t'], h['p'], h['vins'][0], h['vout'], 'key', p, ev=i); STRUCK.setdefault(i, []).append(('flute', p, 'key'))
    FL.fly(h['t'], hits[k + 1]['t'] if k + 1 < len(hits) else t21, h['p'], h['vout'])
EV['wake']['lanterns'] = [dict(t=float(h['t']), x=float(h['p'][0]), z=float(h['p'][2])) for h in FL.hits if h['ev'] in ids]
assert np.allclose(pE, pc + OFF['flute'], atol=1e-6), (pE, pc + OFF['flute'])
v_fl21 = vE
# the oboe's last gondola: basket 4 -> above the reunion key, the bucket tips and the oboe drops onto it on the downbeat
H_DROP = 1.25; T_DROP = math.sqrt(2 * H_DROP / G)
p_hover = pc + OFF['oboe'] + [0, 0, H_DROP]
gondola(OB, pb4, tb4 + 0.7, p_hover, t21 - T_DROP - 0.5, drop=True)
OB.rest(t21 - T_DROP - 0.5, t21 - T_DROP, p_hover)
OB.fly(t21 - T_DROP, t21, p_hover, [0, 0, 0])
v_ob21 = np.array([0, 0, -G * T_DROP])
EV['drop'] = dict(t=float(t21 - T_DROP), p=p_hover.tolist())
SFX.append(['tip', float(t21 - T_DROP)])

# ================================================== P6: tutti bars 21-24, the pair together; final: the sunrise cups
ids = idx_bars(21, 24)
ts = TN[ids]
hits, pE, vEs = plan_retry(ts, pc, [v_fl21, v_ob21], lambda t: pc[0] + V * (t - t21), lambda t: pc[2] - 0.02 * (t - t21), lambda t: 0.0, R=RB)
for k, h in enumerate(hits):
    i = ids[k]
    for j, (nme, b) in enumerate(BALLS.items()):
        p = LEAD[i][1][0 if nme == 'flute' else 1]
        vin = h['vins'][j] if len(h['vins']) > 1 else h['vins'][0]
        b.hit(h['t'], h['p'] + OFF[nme], vin, h['vout'], 'pkey' if k else 'reunion', p, ev=i)
        STRUCK.setdefault(i, []).append((nme, p, 'pkey'))
        b.fly(h['t'], hits[k + 1]['t'] if k + 1 < len(hits) else FINAL, h['p'] + OFF[nme], h['vout'])
iF = ids[-1]
CUPS = {}
for nme, b in BALLS.items():
    p = LEAD[iF][1][0 if nme == 'flute' else 1]
    pcup = pE + OFF[nme]
    b.hit(FINAL, pcup, vEs, [0, 0, 0], 'cup', p, ev=iF); STRUCK.setdefault(iF, []).append((nme, p, 'cup'))
    CUPS[nme] = pcup
EV['reunion'] = dict(t=float(t21), p=pc.tolist())
SFX.append(['reunion', float(t21)])
# the sunrise lever: the two cups sit on one arm; under the weight it sinks DROP_L in 0.35 s (pivot behind), then rests
DROP_L = 0.2; T_SINK = 0.6
for nme, b in BALLS.items():
    pcup = CUPS[nme]; tt = np.arange(FINAL, FINAL + T_SINK + 1e-9, 1 / 240.0)
    u = (tt - FINAL) / T_SINK; sm = u * u * (3 - 2 * u)
    b.path(tt, np.array([pcup - [0, 0, DROP_L * s_] for s_ in sm]))
    b.rest(FINAL + T_SINK, FINAL + 30, pcup - [0, 0, DROP_L])
EV['cups'] = {k: v.tolist() for k, v in CUPS.items()}; EV['cup_drop'] = DROP_L; EV['cup_sink'] = T_SINK

# ================================================== the dawn ball: back rail, steady speed, bell gates
Y_R = -1.5; DZ_R = -2.1; XO_R = 0.6
gates = E['gates']
ALPHA_R = math.atan(SLOPE / V)                      # rail slope = course slope
CRR_R = 5 / 7 * math.tan(ALPHA_R)                   # rolling resistance that balances it (constant speed)
def xr(t): return XO_R + XT(t)
def zr(t): return DZ_R + ZT(t)
DB = Ball('dawn', RD); BALLS['dawn'] = DB
# start: released at T_REL from rest on a steeper first section, reaches V on the main rail at T0 - 0.3
T_REL = 0.25; T_JOIN = T0 - 0.35
A_ST = V / (T_JOIN - T_REL); BETA = math.asin((A_ST + CRR_R * G) / (5 / 7 * G))
L_ST = 0.5 * A_ST * (T_JOIN - T_REL) ** 2
p_join = np.array([xr(T_JOIN), Y_R, zr(T_JOIN)])
p_top = p_join - L_ST * np.array([math.cos(BETA), 0, -math.sin(BETA)])
DB.rest(0.0, T_REL, p_top)
tt = np.arange(T_REL, T_JOIN + 1e-9, 1 / 240.0)
DB.path(tt, np.array([p_top + 0.5 * A_ST * (t - T_REL) ** 2 * np.array([math.cos(BETA), 0, -math.sin(BETA)]) for t in tt]))
T_END = FINAL + 0.32                                # rail end
tt = np.arange(T_JOIN, T_END + 1e-9, 1 / 240.0)
DB.path(tt, np.array([[xr(t), Y_R, zr(t)] for t in tt]))
GATES = []
for t, ps, d, bar, off, kind in gates:
    DB.hits.append(dict(t=float(t), p=[xr(t), Y_R, zr(t)], vin=[V, 0, -SLOPE], vout=[V, 0, -SLOPE], n=[-1, 0, 0], e=1.0, kind='gate',
                        pitch=ps, ev=None))
    GATES.append(dict(t=float(t), x=float(xr(t) + RD), z=float(zr(t)), y=Y_R, pitches=ps, kind=kind, bar=bar, off=off))
# off the rail end: free fall into the latch funnel
p_end = np.array([xr(T_END), Y_R, zr(T_END)]); v_end = np.array([V, 0, -SLOPE])
T_FUN = T_END + 0.42
DB.fly(T_END, T_FUN, p_end, v_end)
p_fun = p_end + v_end * (T_FUN - T_END) + np.array([0, 0, -0.5 * G * (T_FUN - T_END) ** 2])
DB.hits.append(dict(t=float(T_FUN), p=p_fun.tolist(), vin=(v_end + [0, 0, -G * (T_FUN - T_END)]).tolist(), vout=[0, 0, 0], n=[0, 0, 1],
                    e=0.0, kind='funnel', pitch=None, ev=None))
F_SINK, F_DROP = 0.6, 0.2
tt = np.arange(T_FUN, T_FUN + F_SINK + 1e-9, 1 / 240.0)
u = (tt - T_FUN) / F_SINK; sm = u * u * (3 - 2 * u)
DB.path(tt, np.array([p_fun - [0, 0, F_DROP * s_] for s_ in sm]))
DB.rest(T_FUN + F_SINK, FINAL + 30, p_fun - [0, 0, F_DROP])
EV['rail'] = dict(y=Y_R, dz=DZ_R, xo=XO_R, alpha=ALPHA_R, crr=CRR_R, t_rel=T_REL, t_join=T_JOIN, p_top=p_top.tolist(), beta=BETA,
                  a_start=A_ST, t_end=T_END, p_end=p_end.tolist(), t_fun=T_FUN, p_fun=p_fun.tolist())
EV['gates'] = GATES
SFX.append(['funnel', float(T_FUN)])

# ================================================== the sun: counterweight elevator behind the mountains
T_LATCH = T_FUN + 0.35
T_RISE = T_LATCH + 0.6                          # the pull runs along the latch line to the crane
A_SUN = 3.5; T_ACC = 1.8; T_BRAKE = 1.5          # rises with constant acceleration, then a brake stops it
VS = A_SUN * T_ACC; H1 = 0.5 * A_SUN * T_ACC ** 2; H2 = 0.5 * VS * T_BRAKE
EV['sun'] = dict(t_latch=float(T_LATCH), t_rise=float(T_RISE), a=A_SUN, t_acc=T_ACC, t_brake=T_BRAKE, h=H1 + H2, vmax=VS)
SFX.append(['latch', float(T_LATCH)]); SFX.append(['sun', float(T_RISE + T_ACC + T_BRAKE)])


def check():
    info = {}
    hs = [(n, h) for n, b in BALLS.items() for h in b.hits if h['kind'] not in ('basket', 'cup', 'gate', 'funnel')]
    ee = [h['e'] for n, h in hs]
    info['e_range'] = (round(min(ee), 3), round(max(ee), 3))
    info['speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for n, h in hs), 3)
    info['tilt_max_deg'] = round(max(math.degrees(math.acos(min(1, h['n'][2]))) for n, h in hs), 1)
    info['uncovered'] = sorted(set(range(NN)) - set(STRUCK))
    for n, b in BALLS.items():
        zz = [h['p'][2] for h in b.hits]; xx = [h['p'][0] for h in b.hits]
        info[n] = dict(hits=len(b.hits), z=(round(min(zz), 2), round(max(zz), 2)), x=(round(min(xx), 1), round(max(xx), 1)))
    info['ramps'] = [(r['ball'], round(r['L'], 2), round(r['v_lip'], 2)) for r in EV['ramps']]
    info['gondola_vmax'] = [round(g['vmax'], 2) for g in EV['gondolas']]
    info['rail'] = dict(alpha_deg=round(math.degrees(ALPHA_R), 2), crr=round(CRR_R, 4), beta_deg=round(math.degrees(BETA), 1))
    return info


if __name__ == '__main__':
    info = check()
    for k, v in info.items(): print(k, v)
    out = dict(G=G, R=dict(mel=RB, dawn=RD), off={k: v.tolist() for k, v in OFF.items()},
               balls={n: dict(r=b.r, seg=b.seg, hits=b.hits) for n, b in BALLS.items()},
               struck={str(k): v for k, v in STRUCK.items()}, ev=EV, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
