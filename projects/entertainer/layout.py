"""The Entertainer - 'Saloon player piano' (西部の酒場の自動ピアノ). Physics + story planner.
Run after arrange.py; writes layout.json (read by scene.py, render_audio.py, verify.py).

Story (every right-hand onset = a ball striking a glass bell or a chime rail, every left-hand eighth = a ball landing on
one end of a see-saw on the floor rail):
  hook   a gold coin rolls out of the coin chute of the big player piano (the piano wakes up)
  intro  the coin tumbles down a bell staircase (the octave run D-E-C-A-B-G, one tier lower per bar) and lands on the
         start lever (the accented G chord) -> the lever flings the ivory piano ball P onto the bar counter and pulls
         the two hopper gates: the bass balls ZUN (walnut) and CHA (brass) drop onto the see-saw rail
  A      P plays the tune along the counter: short notes = bounces on glass bells, held notes = P lands on a long
         chime rail and rolls along it while the note rings, then drops off its end onto the next note.
         Below, ZUN lands on the left end of each see-saw on the beat, CHA on the right end on the off-beat
         (every see-saw rocks left-right = oom-pah)
  A->B   P rolls off its last chime rail into a passing beer mug of the mug ropeway (splash); up on the balcony the
         swing doors bang open and the cowboy-hat ball C rolls out, drops onto the pick-up bells and plays strain B
         down the next counter while P rides the mug overhead
  B->A'  C's last held note is a long chime rail; at the very instant C drops off it, the mug tips and P drops too:
         both land on the D bell at the same moment (the 'quick draw' - two bells at once) and then play the tune
         together side by side
  end    both ring the jukebox crown bells on the final chord and drop inside; ZUN and CHA dive into its bin; the lid
         snaps shut, the jukebox lights up
  rests  (the left hand has 4 rests): CHA's note before the rest lands in a cup bell on a little catapult, sits
         through the rest, and ZUN's next landing on the catapult's other end throws CHA on to its next note
Physics: one constant gravity G; between contacts pure parabolas; rolling on a level rail = constant speed;
at a bounce the surface normal is chosen so the impulse is along it (tangential velocity kept) with restitution
e <= EMAX (never gains speed); landing on a rail kills the normal velocity (e = 0).  The bounce heights come from a small
linear program (launch speeds >= 0, every bounce e <= EPS) -> the counters descend only ~1 m per strain.
Speed-ups only come from visible mechanisms (start lever, catapults, hopper drop, driven mug ropeway)."""
import json, os, math
import numpy as np
from scipy.optimize import linprog
HERE = os.path.dirname(os.path.abspath(__file__))
G = 7.0; R = 0.15
EMAX = 0.97; EMIN = 0.08; EPS = 0.94
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json')); BASS = json.load(open(f'{HERE}/bass.json'))
T0 = E['t0']; BEAT = E['beat']
def tb(b): return T0 + b * BEAT
def beat_of(t): return (t - T0) / BEAT
V = 2.0                                   # mean course speed (m/s) of every lane -> the camera follows all of them
S = V * BEAT                              # see-saw spacing
Y = {'coin': 0.0, 'P': 0.22, 'C': -0.22, 'bass': -1.05, 'mug': 0.22}
ROLL_THR = 0.5 * BEAT - 1e-6              # notes this long or longer: land on a chime rail and roll
TF_MIN = 0.10


class Ball:
    def __init__(s, name):
        s.name = name; s.seg = []; s.hits = []
    def rest(s, t0, t1, p): s.seg.append(dict(k='rest', t0=float(t0), t1=float(t1), p=list(map(float, p))))
    def fly(s, t0, t1, p0, v0, a=None, k='fly'):
        s.seg.append(dict(k=k, t0=float(t0), t1=float(t1), p=list(map(float, p0)), v=list(map(float, v0)),
                          a=list(map(float, a if a is not None else (0, 0, -G)))))
    def lin(s, t0, t1, p0, p1):           # driven (carried by a mechanism at constant velocity)
        s.seg.append(dict(k='lin', t0=float(t0), t1=float(t1), p=list(map(float, p0)), q=list(map(float, p1))))
    def at(s, t):
        for g in s.seg:
            if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9: return seg_at(g, t)
        if t < s.seg[0]['t0']: return seg_at(s.seg[0], s.seg[0]['t0'])
        return seg_at(s.seg[-1], s.seg[-1]['t1'])


def seg_at(g, t):
    p = np.array(g['p'])
    if g['k'] == 'rest': return p
    u = t - g['t0']
    if g['k'] == 'lin': return p + (np.array(g['q']) - p) * (u / max(1e-9, g['t1'] - g['t0']))
    return p + np.array(g['v']) * u + 0.5 * np.array(g['a']) * u * u


def vout_for(dx, dz, dt, dy=0.0): return np.array([dx / dt, dy / dt, dz / dt + 0.5 * G * dt])


def bounce_e(vin, vout):
    d = vout - vin; n = d / max(1e-9, np.linalg.norm(d)); a = vin @ n
    return (-(vout @ n) / a if a < -1e-9 else 9.0), n


def endp(p, v, dt): return p + v * dt + np.array([0, 0, -0.5 * G * dt * dt]), v + np.array([0, 0, -G * dt])


def add_hit(ball, t, p, vin, vout, kind, **meta):
    vin = np.array(vin, float); vout = np.array(vout, float)
    if np.linalg.norm(vout) < 1e-9: n = -vin / max(1e-9, np.linalg.norm(vin)); e = 0.0
    else: e, n = bounce_e(vin, vout)
    ball.hits.append(dict(t=float(t), p=list(map(float, p)), n=list(map(float, n)), e=float(e), kind=kind,
                          vin=list(map(float, vin)), vout=list(map(float, vout)), **meta))


def lp_bounce(dt, eps=EPS, emin=0.12, ubounds=None, w_in=None, smooth=0.03):
    """maximise the launch speeds (= minimise the drop) of a run of bounces: u_{k+1} <= eps (G dt_k - u_k),
    u_{k+1} >= emin (G dt_k - u_k), 0.05 <= u_k <= G dt_k - 0.15 (still falling when it reaches the next bell);
    a small penalty on |u_{k+1} - u_k| picks the smooth solution among the equally good ones."""
    dt = np.array(dt, float); K = len(dt)
    if K == 0: return np.array([])
    N = K + max(0, K - 1)                         # u_0..u_{K-1}, s_0..s_{K-2}
    A = []; bb = []
    def row(): return np.zeros(N)
    for k in range(K - 1):
        r = row(); r[k + 1] = 1; r[k] = eps; A.append(r); bb.append(eps * G * dt[k])
        r = row(); r[k + 1] = -1; r[k] = -emin; A.append(r); bb.append(-emin * G * dt[k])
        r = row(); r[k + 1] = 1; r[k] = -1; r[K + k] = -1; A.append(r); bb.append(0.0)
        r = row(); r[k + 1] = -1; r[k] = 1; r[K + k] = -1; A.append(r); bb.append(0.0)
    bounds = [(0.05, G * d - 0.15) for d in dt] + [(0, None)] * (K - 1)
    if ubounds:
        for k, (lo, hi) in ubounds.items():
            if 0 <= k < K: bounds[k] = (max(bounds[k][0], lo), min(bounds[k][1], hi))
    if w_in is not None: bounds[0] = (bounds[0][0], min(bounds[0][1], eps * w_in))
    c = np.concatenate([-dt, np.full(K - 1, smooth * float(dt.mean()))])
    if w_in is None: c[0] += 0.25 / (eps * eps * G)          # the first launch has to come from a drop
    r = linprog(c, A_ub=np.array(A) if A else None, b_ub=np.array(bb) if bb else None, bounds=bounds, method='highs')
    assert r.success, r.message
    return r.x[:K]


def plan_lane(times, roll_thr=ROLL_THR, eps=EPS, ub=None):
    """per hop: ('b', u) bounce with launch speed u, or ('r', t_fall) roll along a chime rail then fall t_fall."""
    dt = np.diff(np.array(times)); K = len(dt)
    roll = [dt[k] >= roll_thr for k in range(K)]
    spec = [None] * K; k = 0
    while k < K:
        if roll[k]: k += 1; continue
        j = k
        while j < K and not roll[j]: j += 1
        sub = {kk - k: v for kk, v in (ub or {}).items() if k <= kk < j}
        u = lp_bounce(dt[k:j], eps, ubounds=sub)
        for m in range(k, j): spec[m] = ('b', float(u[m - k]))
        k = j
    for k in range(K):
        if roll[k]:
            nxt = spec[k + 1][1] if k + 1 < K and spec[k + 1] is not None and spec[k + 1][0] == 'b' else 0.0
            tfall = max(TF_MIN, nxt / (eps * G))
            spec[k] = ('r', float(min(tfall, dt[k] - 0.05)))
    return spec


def lane_drop(times, **kw):
    spec = plan_lane(times, **kw); dt = np.diff(np.array(times)); d = 0.0
    for (kind, v), h in zip(spec, dt):
        d += (0.5 * G * h * h - v * h) if kind == 'b' else 0.5 * G * v * v
    return d


def run(ball, p, vin, times, metas, vx_of, spec=None, emax=EMAX, emin=0.06, tilt_max=36.0, dvx=(-0.3, 0.3), kind='bell',
        eps=EPS, vy0=0.0):
    """ball contacts at p at times[0] with incoming vin; plays every times[k] (k < last) following the lane plan.
    Records a hit at every times[k] except the last; returns (p, vin) on arrival at times[-1]."""
    spec = spec or plan_lane(times, eps=eps)
    for k in range(len(times) - 1):
        ti, tj = times[k], times[k + 1]; dt = tj - ti
        if spec[k][0] == 'r':
            tf = spec[k][1]; vr = np.array([vin[0], 0.0, 0.0])
            add_hit(ball, ti, p, vin, vr, 'rail', tfall=tf, roll=float(vin[0] * (dt - tf)), **metas[k])
            ball.hits[-1]['e'] = 0.0; ball.hits[-1]['n'] = [0.0, 0.0, 1.0]
            ball.fly(ti, tj - tf, p, vr, a=(0, 0, 0), k='roll')
            pe = p + vr * (dt - tf)
            ball.fly(tj - tf, tj, pe, vr)
            p, vin = endp(pe, vr, tf)
            continue
        u = spec[k][1]
        dzt = u * dt - 0.5 * G * dt * dt
        vx0 = vin[0] + float(np.clip(vx_of(ti, p) - vin[0], dvx[0], dvx[1]))
        best = None
        for dvx_ in sorted(np.linspace(-0.35, 0.35, 29), key=abs):
            for ddz in np.concatenate([[0], -np.linspace(0.002, 0.12, 30)]):
                vo = vout_for((vx0 + dvx_) * dt, dzt + ddz, dt)
                e, n = bounce_e(vin, vo)
                tilt = math.degrees(math.acos(max(-1, min(1, n[2]))))
                if emin <= e <= emax and tilt <= tilt_max and vo[2] - G * dt < -0.1:
                    best = (vo, e, n); break
            if best: break
        if best is None:
            vo = vout_for(vin[0] * dt, dzt, dt); e, n = bounce_e(vin, vo); best = (vo, e, n)
            print('WARN', ball.name, round(beat_of(ti), 2), 'e', round(e, 3))
        vo, e, n = best
        ball.hits.append(dict(t=float(ti), p=list(map(float, p)), n=list(map(float, n)), e=float(e), kind=kind,
                              vin=list(map(float, vin)), vout=list(map(float, vo)), **metas[k]))
        ball.fly(ti, tj, p, vo)
        p, vin = endp(p, vo, dt)
    return p, vin


def track(xref, tref, gain=0.9, lo=-0.6, hi=0.6):
    """target speed: V plus a correction toward the lane's schedule x = xref + V (t - tref)"""
    def f(t, p): return V + float(np.clip(gain * (xref + V * (t - tref) - p[0]), lo, hi))
    return f


B = {n: Ball(n) for n in ('coin', 'P', 'C', 'Z', 'Ch')}
EV = {}; SFX = []
LT = np.array([l[0] for l in LEAD])
def lead_meta(i): return dict(pitch=int(LEAD[i][1]), low=int(LEAD[i][4]), nn=int(LEAD[i][2]), sec=LEAD[i][3], li=int(i))
def li_at(b): return int(np.argmin(np.abs(LT - tb(b))))

I_ACC = li_at(7.0); I_PK = li_at(7.5); I_A_END = li_at(38.0); I_C0 = li_at(39.25); I_B_END = li_at(69.75)
I_D = li_at(71.5); FINAL_I = len(LEAD) - 1
idxP = list(range(I_PK, I_A_END + 1)); idxC = list(range(I_C0, I_B_END + 1)); idxA2 = list(range(I_D, len(LEAD)))
tP = [LT[i] for i in idxP]; tC = [LT[i] for i in idxC] + [tb(71.5)]; tA2 = [LT[i] for i in idxA2]
D_A = lane_drop(tP + [tP[-1] + 1.25 * BEAT]); D_B = lane_drop(tC); D_A2 = lane_drop(tA2, ub={0: (0.45, 9)})
Z_FLOOR = 1.55                           # lowest bell contact (counter level)
z_A2 = Z_FLOOR + D_A2 + 0.05             # top of the last counter (both balls start A' here)
z_B = z_A2 + D_B + 0.12                  # C's first bell
z_A = Z_FLOOR + D_A + 0.15               # P's first bell
print('drops A/B/A2', round(D_A, 2), round(D_B, 2), round(D_A2, 2), 'start heights', round(z_A, 2), round(z_B, 2), round(z_A2, 2))

# ------------------------------------------------------------------ hook: the coin rolls out of the piano's coin chute
coin = B['coin']
intro = list(range(0, I_ACC))                        # 20 intro onsets on the bell staircase
tI = [LT[i] for i in intro] + [tb(7.0)]
# one tier per bar: the hop from each bar's last note (G) to the next bar's first (D) is a drop (low launch)
ubI = {}
for k in range(len(tI) - 1):
    if beat_of(tI[k]) % 2 >= 1.4: ubI[k] = (0.05, 0.25)
specI = [('b', u) for u in lp_bounce(np.diff(tI), 0.93, ubounds=ubI)]
dI = sum(0.5 * G * h * h - u * h for (_, u), h in zip(specI, np.diff(tI)))
z_lever = z_A - 0.1
UC = 1.45; TD = 0.20
z_coin0 = z_lever + dI
p_first = np.array([1.35, Y['coin'], z_coin0])
p_mouth = p_first + np.array([-UC * TD, 0, 0.5 * G * TD * TD])
T_ROLL = 0.6
coin.rest(0.0, T0 - TD - T_ROLL, p_mouth + [-UC * T_ROLL, 0, 0])
coin.fly(T0 - TD - T_ROLL, T0 - TD, p_mouth + [-UC * T_ROLL, 0, 0], [UC, 0, 0], a=(0, 0, 0), k='roll')   # level chute
coin.fly(T0 - TD, T0, p_mouth, [UC, 0, 0])
EV['chute'] = dict(start=(p_mouth + [-UC * T_ROLL, 0, 0]).tolist(), mouth=p_mouth.tolist(), t_roll=T0 - TD - T_ROLL, t_mouth=T0 - TD)
SFX.append(['crank', 0.1]); SFX.append(['coin', T0 - TD - T_ROLL])
p, vin = run(coin, p_first, np.array([UC, 0, -G * TD]), tI, [lead_meta(i) for i in intro], lambda t, p: 1.45, spec=specI,
             emax=0.95, dvx=(-0.2, 0.2), kind='cbell')
# ------------------------------------------------------------------ start lever: coin end <-> P's cup end
lev_c = p.copy()
VO_COIN = np.array([0.2, 0.0, 0.8])                  # the coin's own little bounce off the dropping lever end
add_hit(coin, tb(7.0), lev_c, vin, VO_COIN, 'lever', **lead_meta(I_ACC))
TCR = 0.3
pc_rest = lev_c + np.array([0.2 * TCR, 0, 0.8 * TCR - 0.5 * G * TCR * TCR])
coin.fly(tb(7.0), tb(7.0) + TCR, lev_c, VO_COIN)
coin.rest(tb(7.0) + TCR, 200, pc_rest)               # comes to rest on the lever's drop plate (it stays down)
p_cup = np.array([lev_c[0] + 0.95, Y['P'], z_lever])
P = B['P']
P.rest(0.0, tb(7.0), p_cup)
specP = plan_lane(tP + [tP[-1] + 1.25 * BEAT])
p_bell0 = np.array([p_cup[0] + 0.62, Y['P'], z_A])
tf = tb(7.5) - tb(7.0)
w0 = specP[0][1] / EPS if specP[0][0] == 'b' else 0.5
vz0 = min((p_bell0[2] - p_cup[2] + 0.5 * G * tf * tf) / tf, G * tf - w0)
p_bell0[2] = p_cup[2] + vz0 * tf - 0.5 * G * tf * tf
vP = np.array([(p_bell0[0] - p_cup[0]) / tf, 0, vz0])
P.fly(tb(7.0), tb(7.5), p_cup, vP)
EV['lever1'] = dict(t=tb(7.0), coin=lev_c.tolist(), cup=p_cup.tolist(), vP=vP.tolist(), coin_rest=pc_rest.tolist())
SFX.append(['lever', tb(7.0)])
# ------------------------------------------------------------------ P plays the pick-up + strain A (back lane)
_, vin = endp(p_cup, vP, tf)
T_LOAD = tb(38.0) + 1.25 * BEAT                       # P rolls off its last chime rail into the passing mug
p, vin = run(P, p_bell0, vin, tP + [T_LOAD], [lead_meta(i) for i in idxP], track(p_bell0[0], tb(7.5)), spec=specP)
# run() treated the mug as the 'next note': the last rail's fall lands P in the mug instead
mug_load = p + np.array([0, 0, 0])
vPin = vin
SFX.append(['splash', T_LOAD])

# ------------------------------------------------------------------ C: balcony -> swing doors -> plays B along the counter
C = B['C']
specC = plan_lane(tC)
xC0 = mug_load[0] + V * (tb(39.25) - T_LOAD) + 0.6
p_c0 = np.array([xC0, Y['C'], z_B])
wC = specC[0][1] / EPS if specC[0][0] == 'b' else 0.6
TFC = max(0.25, wC / G)                                # rolls off the level balcony edge and falls onto the pick-up bell
vzc = 0.0
U_BAL = 1.8
edge = p_c0 - np.array([U_BAL * TFC, 0, vzc * TFC - 0.5 * G * TFC * TFC])
T_EDGE = tb(39.25) - TFC
T_DOOR = tb(38.0)                                      # the swing doors bang open on the bar-21 down-beat
door_x = edge[0] - U_BAL * (T_EDGE - T_DOOR)
start = np.array([door_x - 0.9, Y['C'], edge[2]])
T_START = T_DOOR - 0.9 / U_BAL
C.rest(0.0, T_START, start)
C.fly(T_START, T_EDGE, start, [U_BAL, 0, 0], a=(0, 0, 0), k='roll')
v_edge = np.array([U_BAL, 0, vzc])
C.fly(T_EDGE, tb(39.25), edge, v_edge)
_, vin = endp(edge, v_edge, TFC)
EV['balcony'] = dict(start=start.tolist(), edge=edge.tolist(), door=[door_x, Y['C'], edge[2]], t_door=T_DOOR, t_edge=T_EDGE,
                     u=U_BAL, vz_lip=vzc, t_start=T_START)
SFX.append(['door', T_DOOR])
pC_draw, vC_draw = run(C, p_c0, vin, tC, [lead_meta(i) for i in idxC], track(xC0, tb(39.25)), spec=specC)
T_DRAW = tb(71.5)
# ---- P rides the mug through B; the unloading cam tips the mug; P drops onto the D bell together with C
specA2 = plan_lane(tA2, ub={0: (0.45, 9)})
specA2[-1] = ('r', 0.3)                                 # the final chord rail ends with a 0.3 s drop onto the crown bell
wA2 = specA2[0][1] / EPS
p_draw = np.array([pC_draw[0] + 0.75, Y['P'], pC_draw[2] + 0.06])
TFD = max(0.2, wA2 / G)
H_DROP = 0.5 * G * TFD * TFD
T_TIP = T_DRAW - TFD
mug_out = np.array([p_draw[0] - V * TFD, Y['mug'], p_draw[2] + H_DROP])
vy_out = (Y['P'] - Y['mug']) / TFD                     # the tipped mug rolls P out sideways, back over its lane
# ropeway path: comes down to the loading dip, climbs steeply, runs high over strain B, comes down to the unloading cam
ZH = 5.15; DXR = 2.6
VXM = (mug_out[0] - mug_load[0]) / (T_TIP - T_LOAD)
pts = [mug_load, np.array([mug_load[0] + DXR, Y['P'], ZH]), np.array([mug_out[0] - DXR, Y['P'], ZH]), mug_out]
MUG = [[float(T_LOAD + (q[0] - mug_load[0]) / VXM), q.tolist()] for q in pts]
pre = np.array([mug_load[0] - DXR, Y['P'], ZH]); post = np.array([mug_out[0] + DXR, Y['P'], ZH - 0.6])
MUG = [[float(T_LOAD - DXR / VXM), pre.tolist()]] + MUG + [[float(T_TIP + DXR / VXM), post.tolist()]]
for (ta, pa), (tb_, pb) in zip(MUG[1:-2], MUG[2:-1]): P.lin(ta, tb_, pa, pb)
VMUG = (np.array(MUG[2][1]) - mug_load) / (MUG[2][0] - T_LOAD)
add_hit(P, T_LOAD, mug_load, vPin, VMUG, 'mug')       # caught by the beer, carried
P.fly(T_TIP, T_DRAW, mug_out, [V, vy_out, 0])
_, vPd = endp(mug_out, np.array([V, vy_out, 0]), TFD)
EV['mug'] = dict(path=MUG, t_tip=T_TIP, t_load=T_LOAD, vy_out=vy_out, vxm=VXM)
SFX.append(['tip', T_TIP - 0.06])

# ------------------------------------------------------------------ A': P (back lane) and C (front lane) together
JUKE = {}
for name, p0, v0 in (('P', p_draw, vPd), ('C', pC_draw, vC_draw)):
    b = B[name]
    p, vin = run(b, p0, v0, tA2, [lead_meta(i) for i in idxA2[:-1]], track(p0[0], T_DRAW), spec=specA2)
    JUKE[name] = dict(p=p.tolist(), vin=vin.tolist())
T_IN = LT[-1] + 0.62
for name in ('P', 'C'):
    b = B[name]; p = np.array(JUKE[name]['p']); vin = np.array(JUKE[name]['vin'])
    vo = np.array([vin[0], 0.0, 0.85 * -vin[2]])      # bounces off the crown bell, over the rim, into the jukebox top
    add_hit(b, LT[-1], p, vin, vo, 'juke', **lead_meta(FINAL_I))
    b.fly(LT[-1], T_IN, p, vo)
    pin, vv = endp(p, vo, T_IN - LT[-1])
    JUKE[name]['hole'] = pin.tolist()
    b.fly(T_IN, T_IN + 0.6, pin, vv)                 # falls inside the jukebox (hidden)
xj = 0.5 * (JUKE['P']['hole'][0] + JUKE['C']['hole'][0]); zj = min(JUKE['P']['hole'][2], JUKE['C']['hole'][2])
EV['juke'] = dict(x=xj, z=zj, t_in=T_IN, p=JUKE)

# ------------------------------------------------------------------ bass: ZUN (beats) and CHA (off-beats) on the see-saw rail
BT = [b[0] for b in BASS]
ZI = [k for k, b in enumerate(BASS) if b[3] in (0, 2)]
CI = [k for k, b in enumerate(BASS) if b[3] in (1, 3)]
def bmeta(k): return dict(pitch=int(BASS[k][1]), low=int(BASS[k][2]), e8=int(BASS[k][3]), sec=BASS[k][4], bi=int(k))
TFB = 0.333                               # fall from the chute mouth to the first see-saw
THC = math.radians(40); ACH = 5.0 / 7.0 * G * math.sin(THC); TRC = V / ACH; LCH = V * V / (2 * ACH)
DCH = np.array([math.cos(THC), 0, -math.sin(THC)])
z_b0 = 1.45
xZ0 = p_cup[0] + 1.3
X_Z = lambda t: xZ0 + V * (t - BT[ZI[0]])
X_C = lambda t: X_Z(t) - S
rests = [k for k in CI if BASS[k][3] == 1 and not any(abs(BT[j] - (BT[k] + BEAT)) < 1e-4 for j in CI) and beat_of(BT[k]) < 102]
EV['hopper'] = {}; CUPS = []
BEPS = 0.99
for name, idx, X in (('Z', ZI, X_Z), ('Ch', CI, X_C)):
    b = B[name]
    t0 = BT[idx[0]]; p0 = np.array([X(t0), Y['bass'], z_b0])
    # the gate opens, the ball rolls down a 40-degree chute whose curved end sends it out level at the course speed
    mouth = p0 - np.array([V * TFB, 0, -0.5 * G * TFB * TFB])
    gate = mouth - DCH * LCH
    t_open = t0 - TFB - TRC
    b.rest(0.0, t_open, gate)
    b.fly(t_open, t0 - TFB, gate, [0, 0, 0], a=DCH * ACH, k='roll')
    b.fly(t0 - TFB, t0, mouth, [V, 0, 0])
    EV['hopper'][name] = dict(gate=gate.tolist(), mouth=mouth.tolist(), t_open=t_open, theta=THC)
    SFX.append(['gate', t_open])
    vin = np.array([V, 0, -G * TFB]); p = p0
    segs = [[]]
    for k in idx:
        segs[-1].append(k)
        if name == 'Ch' and k in rests: segs.append([])
    vx_f = lambda t, p, X=X: V + float(np.clip(0.9 * (X(t) - p[0]), -0.6, 0.6))
    for si, sg in enumerate(segs):
        times = [BT[k] for k in sg]
        last_seg = si == len(segs) - 1
        tt = times + [times[-1] + BEAT] if last_seg else times
        dtt = np.diff(tt)
        sp = [('b', u) for u in lp_bounce(dtt, BEPS, emin=0.1, w_in=-vin[2], ubounds={len(dtt) - 1: (0.05, 0.5 * G * dtt[-1])})]
        p, vin = run(b, p, vin, tt, [bmeta(k) for k in (sg if last_seg else sg[:-1])], vx_f, spec=sp, emax=0.995,
                     tilt_max=30, kind='saw', dvx=(-0.5, 0.5))
        if last_seg:
            # after its last note the ball makes one more ordinary hop and lands in the felt trough in front of the jukebox
            last = b.hits[-1]
            EV.setdefault('trough', {})[name] = dict(p=p.tolist(), t=float(tt[-1]))
            b.rest(tt[-1], tt[-1] + 3.0, p)
            continue
        # the note before the rest lands in the cup bell of a catapult (sounds, ball stops)
        tc = times[-1]
        add_hit(b, tc, p, vin, np.zeros(3), 'chacup', **bmeta(sg[-1]))
        tl = tc + 0.5 * BEAT                 # ZUN lands on the catapult's far end -> CHA is thrown
        t2 = tl + 1.5 * BEAT
        b.rest(tc, tl, p)
        nxt = [BT[k] for k in segs[si + 1][:3]]
        u_n = lp_bounce(np.diff(nxt), BEPS, emin=0.1, w_in=0.5 * G * BEAT / BEPS)[0]
        w_need = u_n / BEPS                  # land on the next see-saw fast enough for the next bounce
        tfl = t2 - tl
        vz = G * tfl - w_need
        dz = min(0.0, vz * tfl - 0.5 * G * tfl * tfl)
        q = np.array([X(t2), Y['bass'], p[2] + dz])
        vo = vout_for(q[0] - p[0], q[2] - p[2], tfl)
        b.fly(tl, t2, p, vo)
        CUPS.append(dict(t_cup=tc, t_throw=tl, p=p.tolist(), v=vo.tolist(), t_land=t2))
        p, vin = endp(p, vo, tfl)
        SFX.append(['thud', tl])
EV['cups'] = CUPS
for c in CUPS:
    for h in B['Z'].hits:
        if abs(h['t'] - c['t_throw']) < 1e-4: h['kind'] = 'cat'; c['z'] = h['p']
EV['rests_beats'] = [round(beat_of(BT[k]) + 1, 3) for k in rests]


# ------------------------------------------------------------------ checks
def check():
    info = {}
    hs = [(n, h) for n, b in B.items() for h in b.hits]
    bo = [h for n, h in hs if np.linalg.norm(h['vout']) > 0 and h['kind'] not in ('mug', 'lever', 'rail')]
    info['e_max'] = round(max(h['e'] for h in bo), 3); info['e_min'] = round(min(h['e'] for h in bo), 3)
    allb = [h for n, h in hs if np.linalg.norm(h['vout']) > 0 and h['kind'] not in ('mug', 'lever')]
    info['speed_out_in_max'] = round(max(np.linalg.norm(h['vout']) / max(1e-6, np.linalg.norm(h['vin'])) for h in allb), 3)
    worst = max(allb, key=lambda h: np.linalg.norm(h['vout']) / max(1e-6, np.linalg.norm(h['vin'])))
    info['speed_worst_at'] = (worst['kind'], round(beat_of(worst['t']), 2))
    info['tilt_max_deg'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for h in bo), 1)
    cov = {}
    for n, h in hs:
        if 'li' in h: cov.setdefault(h['li'], []).append(n)
    info['lead_uncovered'] = [i for i in range(len(LEAD)) if i not in cov]
    bcov = {}
    for n, h in hs:
        if 'bi' in h: bcov.setdefault(h['bi'], []).append(n)
    info['bass_uncovered'] = [k for k in range(len(BASS)) if k not in bcov]
    for n, b in B.items():
        zz = [h['p'][2] for h in b.hits]
        if zz: info[f'z_{n}'] = (round(min(zz), 2), round(max(zz), 2), len(zz))
        xx = [h['p'][0] for h in b.hits]
        if xx: info[f'x_{n}'] = (round(min(xx), 1), round(max(xx), 1))
    info['rails'] = sum(1 for n, h in hs if h['kind'] == 'rail')
    info['rests'] = EV['rests_beats']
    low = sorted(bo, key=lambda h: h['e'])[:6]
    info['lowest_e'] = [(h['kind'], round(beat_of(h['t']), 2), round(h['e'], 3)) for h in low]
    return info


if __name__ == '__main__':
    info = check()
    for k, v in info.items(): print(k, v)
    out = dict(G=G, R=R, V=V, S=S, Y=Y, balls={n: dict(seg=b.seg, hits=b.hits) for n, b in B.items()}, ev=EV, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
