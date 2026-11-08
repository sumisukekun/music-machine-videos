"""William Tell galop - 'Red vs White sports-day relay' (紅白リレー). Physics + story planner.
Run after arrange.py; writes layout.json (read by scene.py and render_audio.py).

Story (every melody note = a ball landing on a bar that sounds it):
  fanfare  gold starter ball rings the bell tower, drops on the trigger -> starting pistol, plungers kick the runners off
  A        1st runners (red / white) gallop side by side on their own lanes, red slowly pulls ahead
  A->A2    baton zone: each runner knocks the waiting 2nd runner off its tee (equal-mass collision: the 1st stops dead,
           the 2nd carries on with exactly its velocity), the 1st drops into its team basket
  Am       minor key: a trapdoor gives way under white, it tumbles down into the pit (dark lower track) and gets slowed
  bridge   white rolls onto a see-saw; the cheer squad's big weight ball drops on the other end and catapults white back
           up to its lane (red plays the bridge alone meanwhile)
  A3, C    white gains (hurdle section in C)      A4  neck and neck, white edges ahead
  coda     white breaks the finish tape a nose ahead; both hop to the podium, ping-pong 'c-B-c-B' between 1st and 2nd
           place, final chord -> kusudama opens
Physics: one constant gravity G; between hits pure parabolas; at a hit the bar's surface normal is chosen so the impulse
is along the normal (tangential velocity kept) with restitution e <= EMAX (never gains speed). Lane heights come out of
that rule (the course gently descends where the rhythm needs energy). Speed-ups only come from mechanisms (plunger,
see-saw)."""
import json, os, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = 7.0; R = 0.15
EMAX = 0.97
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
SECS = dict((k, v) for k, v in E['secs'])
TN = np.array([l[0] for l in LEAD]); NN = len(TN)
def idx(t): return int(np.argmin(np.abs(TN - t)))
SEC_I = {k: int(np.searchsorted(TN, v - 1e-6)) for k, v in SECS.items()}
LANE_Y = {'R': 0.75, 'W': -0.75, 'gold': 1.75}


class Ball:
    def __init__(s, name, team):
        s.name = name; s.team = team; s.seg = []; s.hits = []
    def rest(s, t0, t1, p): s.seg.append(dict(k='rest', t0=t0, t1=t1, p=list(map(float, p))))
    def fly(s, t0, t1, p0, v0): s.seg.append(dict(k='fly', t0=t0, t1=t1, p=list(map(float, p0)), v=list(map(float, v0))))
    def at(s, t):
        for g in s.seg:
            if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9:
                p = np.array(g['p'])
                if g['k'] == 'rest': return p
                u = t - g['t0']; v = np.array(g['v'])
                return p + v * u + np.array([0, 0, -0.5 * G * u * u])
        if t < s.seg[0]['t0']: return s.at(s.seg[0]['t0'])
        return s.at(s.seg[-1]['t1'])
    def vel(s, t):
        for g in s.seg:
            if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9:
                if g['k'] == 'rest': return np.zeros(3)
                return np.array(g['v']) + np.array([0, 0, -G * (t - g['t0'])])
        return np.zeros(3)


def vout_for(dx, dz, dt, dy=0.0): return np.array([dx / dt, dy / dt, dz / dt + 0.5 * G * dt])


def bounce_e(vin, vout):
    """surface normal n ~ vout - vin (impulse along n, tangential kept); restitution e"""
    d = vout - vin; n = d / max(1e-9, np.linalg.norm(d))
    a = vin @ n
    return (-(vout @ n) / a if a < -1e-9 else 9.0), n


EMIN = 0.2


def run(ball, p, vin, t, notes, vx_of, dz_of, emax=EMAX, kind_of=lambda i: 'bar', tilt_max=40.0, vy_of=lambda i: 0.0, emin=EMIN, dvx=(-0.12, 0.08)):
    """ball sits at contact centre p at time t (= note time of notes[0]) with incoming velocity vin.
    hops through notes[1:]; vx_of(i) target horizontal speed for the hop starting at note i, dz_of(i) target height change.
    The landing height is lowered (never raised) until the bounce is physical: emin <= e <= emax, bar tilt <= tilt_max,
    and the ball arrives at the next bar fast enough (vertically) to make that bar's own hop with e <= emax.
    returns state on arrival at the last note (contact centre p, incoming velocity)."""
    for k in range(len(notes) - 1):
        i = notes[k]; j = notes[k + 1]; dt = TN[j] - TN[i]
        dzt = dz_of(i) * dt / 0.205          # dz_of = height change per quarter-note of flight
        if k + 1 < len(notes) - 1:
            dt2 = TN[notes[k + 2]] - TN[j]
            need = (dz_of(j) * dt2 / 0.205 / dt2 + 0.5 * G * dt2) / (emax * 0.99)
        else: need = 0.0
        tx = vx_of(i); cx = vin[0]
        vx0 = cx + float(np.clip(tx - cx, dvx[0], dvx[1]))
        vxs = sorted(set(np.round(np.concatenate([[vx0], vx0 + np.linspace(-0.3, 0.1, 21)]), 4)), key=lambda v: abs(v - vx0))
        dzs = sorted(np.concatenate([dzt - np.linspace(0, 0.5 * G * dt * dt + 0.8, 300), dzt + np.linspace(0.003, 0.25, 40)]),
                     key=lambda d: (0 if d <= dzt else 1, abs(d - dzt)))
        best = None; cands = []
        for vx in vxs[:6]:
            for dz in dzs:
                vo = vout_for(vx * dt, dz, dt, vy_of(i) * dt)
                e, n = bounce_e(vin, vo)
                if not (emin <= e <= emax) or vo[2] - G * dt > -0.12: continue
                tilt = math.degrees(math.acos(max(-1, min(1, n[2]))))
                if tilt > tilt_max: continue
                cands.append((dz, vo, e, n))
                if -(vo[2] - G * dt) >= need: best = (dz, vo, e, n); break
            if best is not None: break
        if best is None and cands: best = cands[0]; print('WARN need', ball.name, i, round(need, 2))
        if best is None:
            for vx in vxs:
                for dz in dzs:
                    vo = vout_for(vx * dt, dz, dt, vy_of(i) * dt); e, n = bounce_e(vin, vo)
                    if emin <= e <= emax and vo[2] - G * dt < -0.05: best = (dz, vo, e, n); print('WARN tilt', ball.name, i); break
                if best is not None: break
        if best is None:
            allc = []
            for vx in vxs:
                for dz in dzs[::5]:
                    vo = vout_for(vx * dt, dz, dt, vy_of(i) * dt); e, n = bounce_e(vin, vo)
                    allc.append((abs(min(0, e - emin)) + max(0, e - emax), dz, vo, e, n))
            _, dz, vo, e, n = min(allc, key=lambda c: c[0]); best = (dz, vo, e, n)
            print('WARN e', ball.name, i, round(e, 3))
        dz, vo, e, n = best
        ball.hits.append(dict(i=int(i), t=float(TN[i]), p=list(map(float, p)), n=list(map(float, n)), e=float(e), kind=kind_of(i),
                              vin=list(map(float, vin)), vout=list(map(float, vo))))
        ball.fly(TN[i], TN[j], p, vo)
        p = p + vo * dt + np.array([0, 0, -0.5 * G * dt * dt]); vin = vo + np.array([0, 0, -G * dt])
    return p, vin


def add_hit(ball, i, p, vin, vout, kind, t=None):
    e, n = bounce_e(np.array(vin), np.array(vout))
    ball.hits.append(dict(i=int(i), t=float(TN[i] if t is None else t), p=list(map(float, p)), n=list(map(float, n)), e=float(e), kind=kind,
                          vin=list(map(float, vin)), vout=list(map(float, vout))))


B = {n: Ball(n, n[0] if n != 'gold' else 'gold') for n in ('R1', 'W1', 'R2', 'W2', 'gold')}
EV = {}
SFX = []
I_A, I_A2, I_AM, I_BR, I_A3, I_C, I_A4, I_CODA = (SEC_I[k] for k in ('A', 'A2', 'Am', 'br', 'A3', 'C', 'A4', 'coda'))

# ------------------------------------------------------------------ fanfare: gold starter ball on the bell tower
g = B['gold']
fan = list(range(0, I_A - 2))                # 19 fanfare notes (the last 2 notes before A are the runners' pick-up)
p = np.array([-7.0, LANE_Y['gold'], 4.6 + R]); TD = 0.32
p_ch = p + np.array([-2.1 * TD, 0, 0.5 * G * TD * TD])          # rolls out of the chute mouth horizontally
g.rest(0.0, TN[0] - TD - 0.25, p_ch + [-2.1 * 0.25, 0, 0.0])
g.fly(TN[0] - TD - 0.25, TN[0] - TD, p_ch + [-2.1 * 0.25, 0, 0.0], [2.1, 0, 0])
g.fly(TN[0] - TD, TN[0], p_ch, [2.1, 0, 0])
vin = np.array([2.1, 0, -G * TD])
EV['chute'] = dict(start=(p_ch + [-2.1 * 0.25, 0, 0.0]).tolist(), mouth=p_ch.tolist())
p, vin = run(g, p, vin, TN[0], fan, lambda i: 2.1, lambda i: -0.16, emax=0.95, dvx=(-0.3, 0.3))
# last fanfare note -> trigger lever at the bar-7 downbeat (the runners' gun)
T_GUN = TN[fan[-1]] + 0.205
dt = T_GUN - TN[fan[-1]]; trig = p + np.array([2.0 * dt, 0, -0.08])
vo = vout_for(trig[0] - p[0], trig[2] - p[2], dt)
add_hit(g, fan[-1], p, vin, vo, 'bell'); g.fly(TN[fan[-1]], T_GUN, p, vo)
g.rest(T_GUN, 80.0, trig)
EV['trigger'] = dict(t=T_GUN, p=trig.tolist(), vin=(vo + [0, 0, -G * dt]).tolist())
SFX.append(['gun', T_GUN])

# ------------------------------------------------------------------ runners
LANE_Z0 = {'R': 0.0, 'W': -1.25}
def start(ball, team):
    p0 = np.array([0.0, LANE_Y[team], LANE_Z0[team] + R])
    ball.rest(0.0, T_GUN, p0)
    i0 = I_A - 2; dt = TN[i0] - T_GUN
    p1 = p0 + np.array([0.62, 0, -0.02])
    v0 = vout_for(p1[0] - p0[0], p1[2] - p0[2], dt)
    ball.fly(T_GUN, TN[i0], p0, v0)
    EV.setdefault('start', {})[team] = dict(p=p0.tolist(), v=v0.tolist())
    return p1, v0 + np.array([0, 0, -G * dt])

# gap targets x_W - x_R (story) and red speeds
VR = {'A': 3.35, 'A2': 3.45, 'Am': 3.45, 'br': 3.45, 'A3': 3.45, 'C': 3.4, 'A4': 3.5}
def sec_of(i):
    for k in ('coda', 'A4', 'C', 'A3', 'br', 'Am', 'A2', 'A'):
        if i >= SEC_I[k]: return k
    return 'fan'

pR, vR = start(B['R1'], 'R'); pW, vW = start(B['W1'], 'W')
T_HO = {'R': (I_A2 - 3, 0.42), 'W': (I_A2, 0.42)}          # hand-off inside hop (note index, fraction of hop)


def handoff(b1, b2, team, p, vin, i):
    """b1 hops off note i, and at fraction f of the hop knocks b2 (resting on a tee) forward."""
    ii, f = T_HO[team]; assert ii == i
    j = i + 1; dt = TN[j] - TN[i]
    vx = b1.hits[-1]['vout'][0] if b1.hits else 3.3
    vo = vout_for(vx * dt, 0.0, dt)
    e, n = bounce_e(vin, vo)
    if e > EMAX:
        for dz in np.linspace(0, -1, 200):
            vo = vout_for(vx * dt, dz, dt); e, n = bounce_e(vin, vo)
            if e <= EMAX: break
    add_hit(b1, i, p, vin, vo, 'bar')
    tc = TN[i] + f * dt; u = tc - TN[i]
    pc = p + vo * u + np.array([0, 0, -0.5 * G * u * u]); vc = vo + np.array([0, 0, -G * u])
    b1.fly(TN[i], tc, p, vo)
    p2 = pc + 2 * R * vc / np.linalg.norm(vc)              # 2nd runner centre at contact (line of centres along velocity)
    b2.rest(0.0, tc, p2)
    # 1st runner: stops dead, falls into the team basket below
    hb = 0.62; tfall = math.sqrt(2 * hb / G)
    b1.fly(tc, tc + tfall, pc, [0, 0, 0]); pb = pc + np.array([0, 0, -hb])
    b1.rest(tc + tfall, 80.0, pb)
    EV.setdefault('handoff', {})[team] = dict(t=tc, p1=pc.tolist(), p2=p2.tolist(), v=vc.tolist(), basket=pb.tolist(), tland=tc + tfall)
    SFX.append(['clack', tc]); SFX.append(['basket', tc + tfall])
    # 2nd runner flies the rest of the hop from p2 with the same velocity, lands at note j
    u2 = TN[j] - tc
    b2.fly(tc, TN[j], p2, vc)
    pj = p2 + vc * u2 + np.array([0, 0, -0.5 * G * u2 * u2]); vj = vc + np.array([0, 0, -G * u2])
    return pj, vj


def gapv(team, i, vxR):
    """white speed from the story's gap plan"""
    s = sec_of(i)
    return vxR + {'A': -0.12, 'A2': -0.1}.get(s, 0)

flat = lambda i: -0.012
# --- red: A (R1), hand-off near the end of A, then R2 to the finish line
pR, vR = run(B['R1'], pR, vR, TN[I_A - 2], list(range(I_A - 2, I_A2 - 2)), lambda i: VR[sec_of(i)] if i >= I_A else 3.0, flat)
pR, vR = handoff(B['R1'], B['R2'], 'R', pR, vR, I_A2 - 3)
HURDLE = lambda i: 'hurdle' if I_C + 2 <= i < I_A4 - 2 and TN[i + 1] - TN[i] > 0.15 and (i - I_C) % 5 == 0 else 'bar'
pR, vR = run(B['R2'], pR, vR, TN[I_A2 - 2], list(range(I_A2 - 2, I_CODA + 1)), lambda i: VR.get(sec_of(i), 3.4), flat, kind_of=HURDLE)
RF = (pR.copy(), vR.copy())
# --- white: A (W1), hand-off on the A2 downbeat hop, A2 (W2) up to the trapdoor
pW, vW = run(B['W1'], pW, vW, TN[I_A - 2], list(range(I_A - 2, I_A2 + 1)), lambda i: gapv('W', i, VR[sec_of(i)]) if i >= I_A else 2.95, flat)
pW, vW = handoff(B['W1'], B['W2'], 'W', pW, vW, I_A2)
pW, vW = run(B['W2'], pW, vW, TN[I_A2 + 1], list(range(I_A2 + 1, I_AM + 1)), lambda i: gapv('W', i, VR[sec_of(i)]), flat)
# --- Am: trapdoor gives way (first minor-key note), white tumbles into the pit and is slowed down there
z_trap = pW[2]; EV['trap'] = dict(i=I_AM, t=float(TN[I_AM]), p=pW.tolist())
# W2 lands on the trapdoor (note I_AM); the door drops away under it, W2 falls freely into the mud pit
# (red alone plays the notes during the fall) and lands on the mud-pit floor on note I_AM + 5
i = I_AM; I_MUD = I_AM + 5; TFp = TN[I_MUD] - TN[i]
vo = np.array([2.9, 0.0, 0.0])
B['W2'].hits.append(dict(i=int(i), t=float(TN[i]), p=pW.tolist(), n=[0, 0, 1], e=0.0, kind='trap', vin=vW.tolist(), vout=vo.tolist()))
B['W2'].fly(TN[i], TN[I_MUD], pW, vo)
pW = pW + vo * TFp + np.array([0, 0, -0.5 * G * TFp ** 2]); vW = vo + np.array([0, 0, -G * TFp])
EV['trap']['fall'] = 0.5 * G * TFp ** 2; EV['trap']['mud_t'] = float(TN[I_MUD])
SFX.append(['trap', float(TN[I_AM]) + 0.02]); SFX.append(['splat', float(TN[I_MUD])])
pW, vW = run(B['W2'], pW, vW, TN[I_MUD], list(range(I_MUD, I_BR)), lambda i: 2.95, flat, emin=0.06,
             kind_of=lambda i: 'pit')
# last Am note -> soft landing in the see-saw cup (no note: the cup is padded)
i = I_BR - 1; T_CATCH = TN[i] + 0.31; dt = T_CATCH - TN[i]
cup = pW + np.array([2.9 * dt, 0, -0.08])
vo = vout_for(cup[0] - pW[0], cup[2] - pW[2], dt)
add_hit(B['W2'], i, pW, vW, vo, 'pit'); B['W2'].fly(TN[i], T_CATCH, pW, vo)
T_LAUNCH = SECS['br'] + 2 * E['bar']                      # the weight lands on the 3rd bridge bar downbeat
B['W2'].rest(T_CATCH, T_LAUNCH, cup)
TFALL = 0.5; RW = 0.34
EV['seesaw'] = dict(cup=cup.tolist(), t_catch=T_CATCH, t_launch=T_LAUNCH, arm=0.95, pivot=(cup + np.array([0.95, 0, -0.12])).tolist(),
                    weight_r=RW, t_release=T_LAUNCH - TFALL, fall=0.5 * G * TFALL ** 2)
SFX.append(['thud', T_CATCH]); SFX.append(['weight', T_LAUNCH]); SFX.append(['release', T_LAUNCH - TFALL])
# catapult flight back up onto the white lane, landing on a bridge note ~2.6 behind red
I_LAND = int(np.searchsorted(TN, SECS['br'] + 5 * E['bar'] - 1e-6))
tl = TN[I_LAND]; T_FL = tl - T_LAUNCH
xr = B['R2'].at(tl)[0]
land = np.array([xr - 3.4, LANE_Y['W'], z_trap - 0.25 + R])
vl = vout_for(land[0] - cup[0], land[2] - cup[2], T_FL)
B['W2'].fly(T_LAUNCH, tl, cup, vl)
EV['catapult'] = dict(t=T_LAUNCH, v=vl.tolist(), land=land.tolist(), t_land=tl, i_land=I_LAND)
vin = vl + np.array([0, 0, -G * T_FL])
# landing mat: flat, friction takes the extra speed off (tangential impulse <= 0.6 x normal impulse)
j = I_LAND + 1; dt = TN[j] - tl
vz_after = vout_for(0, -0.03, dt)[2]
vx_after = vin[0] - 0.6 * (vz_after - vin[2])          # rubber mat: friction impulse = 0.6 x normal impulse
vo = vout_for(vx_after * dt, -0.03, dt)
mu = abs(vo[0] - vin[0]) / max(1e-6, abs(vo[2] - vin[2]))
EV['catapult']['mat_mu'] = mu
B['W2'].hits.append(dict(i=I_LAND, t=float(tl), p=land.tolist(), n=[0, 0, 1], e=float(vo[2] / -vin[2]), kind='mat', vin=vin.tolist(), vout=vo.tolist()))
B['W2'].fly(tl, TN[j], land, vo)
pW = land + vo * dt + np.array([0, 0, -0.5 * G * dt * dt]); vW = vo + np.array([0, 0, -G * dt])
def w_vx(i):
    s = sec_of(i); xr_ = B['R2'].at(TN[i])[0]
    return VR.get(s, 3.4) + {'br': 0.1, 'A3': 0.1, 'C': 0.08, 'A4': 0.15}.get(s, 0)
jb = j + 14                                              # braking hops after the landing (bars tilted back a little)
pW, vW = run(B["W2"], pW, vW, TN[j], list(range(j, jb + 1)), w_vx, flat, dvx=(-0.25, 0.05))
pW, vW = run(B['W2'], pW, vW, TN[jb], list(range(jb, I_CODA + 1)), w_vx, flat, kind_of=HURDLE)
WF = (pW.copy(), vW.copy())

# ------------------------------------------------------------------ coda: finish tape, victory steps, podium ping-pong
tf = TN[I_CODA]
EV['finish'] = dict(t=tf, xW=float(WF[0][0]), xR=float(RF[0][0]))
CO = list(range(I_CODA, NN))
# who plays which coda note (searched so that each ball's hop sequence is physical, see notes in the docstring):
# k0 both cross the line, k1-k9 they take turns down the victory steps, k10-k18 ping-pong on the podium
# (white = the held B and every B on the 1st-place block, red = the c's on the 2nd-place block), last hit sticks in the podium cup.
CPLAN = 'BBWBWBWWRWWRWRWRWRW'
K_PING = 10
def coda_vx(team):
    def f(i):
        k = i - I_CODA
        if k < K_PING - 1: return max(0.6, 3.3 - 0.42 * k)
        return 0.0
    return f
def coda_dz(team):
    def f(i):
        k = i - I_CODA
        if k < K_PING - 1: return -0.04 if team == 'W' else -0.12
        return 0.0
    return f
def coda_vy(team):
    tgt = -0.3 if team == 'W' else 0.3
    def f(i):
        k = i - I_CODA
        if k >= K_PING - 2: return 0.0
        return (tgt - LANE_Y[team]) / (TN[I_CODA + K_PING - 2] - tf)
    return f
for team, (p, v) in (('W', WF), ('R', RF)):
    b = B[team + '2']
    seq = [I_CODA + k for k, o in enumerate(CPLAN) if o in ('B', team)]
    pre = [i for i in seq if i - I_CODA < K_PING + (0 if team == 'W' else 1)]
    ping = [i for i in seq if i not in pre]
    p, v = run(b, p, v, tf, pre, coda_vx(team), coda_dz(team), vy_of=coda_vy(team), tilt_max=40.0,
               emin=0.1, dvx=(-0.5, 0.05), kind_of=lambda i: 'finish' if i == I_CODA else 'step')
    # podium: in-place bounces (superball on the podium top, e <= 1), the last one sticks in the cup
    pod = [pre[-1]] + ping
    p, v = run(b, p, v, TN[pod[0]], pod, lambda i: 0.0, lambda i: 0.0, emax=1.0, emin=0.1, kind_of=lambda i: 'podium', tilt_max=40.0, dvx=(-3, 0))
    add_hit(b, pod[-1], p, v, np.zeros(3), 'podium')
    b.rest(TN[pod[-1]], 80.0, p)
    EV.setdefault('podium', {})[team] = dict(p=p.tolist(), t_first=float(TN[pod[0]]), t_last=float(TN[pod[-1]]))
# final tutti (after the ping-pong): the kusudama's gold balls drop one by one onto the victory bells, last = final chord
K_GOLD = len(CPLAN)
GB = []
for k in range(K_GOLD, len(CO)):
    i = I_CODA + k; GB.append(i)
EV['gold_notes'] = GB
SFX.append(['tape', tf - 0.05]); SFX.append(['kusudama', TN[-1]])

# ------------------------------------------------------------------ checks
def all_hits():
    out = []
    for n, b in B.items():
        for h in b.hits: out.append((n, h))
    return out
def check():
    info = {}
    hs = all_hits()
    ee = [h['e'] for n, h in hs if h['kind'] not in ('podium',) and h['kind'] != 'mat' and np.linalg.norm(h['vout']) > 0]
    info['e_max'] = round(max(ee), 3); info['e_min'] = round(min(ee), 3)
    sp = [np.linalg.norm(h['vout']) / max(1e-6, np.linalg.norm(h['vin'])) for n, h in hs if np.linalg.norm(h['vout']) > 0 and h['kind'] != 'mat']
    info['speed_out_in_max'] = round(max(sp), 3)
    til = [math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for n, h in hs if h['kind'] != 'podium']
    info['tilt_max_deg'] = round(max(til), 1)
    # every note covered
    cov = set(h['i'] for n, h in hs) | set(EV['gold_notes'])
    info['notes_uncovered'] = sorted(set(range(NN)) - cov)
    # z spans
    for n, b in B.items():
        zz = [h['p'][2] for h in b.hits]
        if zz: info[f'z_{n}'] = (round(min(zz), 2), round(max(zz), 2))
    info['mat_mu'] = round(EV['catapult']['mat_mu'], 2)
    # gap timeline
    gaps = []
    for t in (5, 10, 12, 17, 20, 23.5, 26, 30, 36.7, 43, 49.8, 53, 55.9):
        w = 'W1' if t < EV['handoff']['W']['t'] else 'W2'; r = 'R1' if t < EV['handoff']['R']['t'] else 'R2'
        gaps.append((t, round(B[w].at(t)[0] - B[r].at(t)[0], 2)))
    info['gap_W_minus_R'] = gaps
    return info

if __name__ == '__main__':
    info = check()
    for k, v in info.items(): print(k, v)
    out = dict(G=G, R=R, lane_y=LANE_Y, balls={n: dict(team=b.team, seg=b.seg, hits=b.hits) for n, b in B.items()}, ev=EV, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
