"""Radetzky March - 'New Year's concert of marbles' (ビー玉のニューイヤーコンサート). Physics + story planner.
Run after arrange.py; writes layout.json (read by scene.py, render_audio.py, verify.py).

Picture: a gold concert hall seen from the stage (vertical cut-away). Bottom: the basement corridor. Stage: two golden
spiral glockenspiel towers (left / right) and the conductor's drum podium in the middle. Above: 7 rows of marble audience,
one empty seat (top row, centre). Right: the service elevator shaft. Top: the ceiling catwalk, a see-saw and a glass pipe
that winds down the back wall to a golden funnel above the empty seat.

Story
  0-1 s    the conductor (big black marble) drops from a ceiling hatch onto the drum podium: first note
  intro    gold player spirals down the LEFT tower (keys slide out of the column just before the ball needs them)
  A 1st    audience claps softly on every beat. The LATE marble (red) appears at the basement door and hops down the stairs
           - on the OFF-beats: the only one out of time
  A 2nd    forte: a clapping wave rolls across the hall. The late one rolls along the corridor into the service elevator
  B        silver player on the RIGHT tower / gold again on the left. Elevator up, ding, rolls onto the see-saw, a weight
           drops on the other end and flings it into the glass pipe; it takes a wrong turn (uphill spur), rolls back
  return   forte wave again; the late one circles in the golden funnel above its seat ... drops through on the very last
           chord, landing in its seat together with the whole hall's final clap. Standing ovation, confetti.
Physics: one constant gravity G. Free flight = parabolas. Bounces on keys: the key's normal is along the impulse,
restitution e <= EMAX (never speeds up). Rolling = (5/7) g sin(slope) minus rolling resistance. Speed-ups only from
slopes or visible mechanisms (see-saw weight). Marbles in their seats jump by themselves (they are the audience)."""
import json, os, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = 7.0; GR = 5 / 7 * G           # rolling ball on an incline
EMAX = 0.97; EMIN = 0.2
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
TN = np.array([l[0] for l in LEAD]); NN = len(TN)
T0, Q, HALF, BAR, FINAL = E['t0'], E['q'], E['half'], E['bar'], E['final']
KB = lambda k: T0 + k * BAR                       # bar k start
STAB = np.array(E['stab']); BEATS = [b[0] for b in E['beats']]
R_P = 0.25; R_A = 0.22; R_L = 0.25; R_C = 0.45     # player, audience, late marble, conductor radii
SFX = []
EV = {}


# ------------------------------------------------------------------ ball paths
class Ball:
    def __init__(s, name, r): s.name = name; s.r = r; s.seg = []; s.hits = []
    def rest(s, t0, t1, p): s.seg.append(dict(k='rest', t0=float(t0), t1=float(t1), p=list(map(float, p))))
    def fly(s, t0, t1, p0, v0): s.seg.append(dict(k='fly', t0=float(t0), t1=float(t1), p=list(map(float, p0)), v=list(map(float, v0))))
    def samp(s, t0, P, dt=1 / 240):   # sampled kinematic path (rolling in pipes, lifts, funnel)
        s.seg.append(dict(k='samp', t0=float(t0), t1=float(t0 + dt * (len(P) - 1)), dt=dt, P=np.round(np.array(P), 5).tolist()))
    def at(s, t): return seg_at(s.seg, t)
    def end(s): return s.at(s.seg[-1]['t1'])


def seg_at(seg, t):
    for g in seg:
        if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9:
            if g['k'] == 'rest': return np.array(g['p'])
            if g['k'] == 'fly':
                u = t - g['t0']; return np.array(g['p']) + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
            P = g['P']; x = (t - g['t0']) / g['dt']; i = int(min(len(P) - 2, max(0, math.floor(x)))); f = x - i
            return np.array(P[i]) * (1 - f) + np.array(P[i + 1]) * f
    if t < seg[0]['t0']: return seg_at(seg, seg[0]['t0'])
    return seg_at(seg, seg[-1]['t1'])


def vout_for(d, dt): return np.array([d[0] / dt, d[1] / dt, d[2] / dt + 0.5 * G * dt])


def bounce_e(vin, vout):
    d = vout - vin; n = d / max(1e-9, np.linalg.norm(d)); a = vin @ n
    return (-(vout @ n) / a if a < -1e-9 else 9.0), n


def add_hit(ball, i, t, p, vin, vout, kind, **kw):
    e, n = bounce_e(np.array(vin, float), np.array(vout, float))
    h = dict(i=int(i), t=float(t), p=list(map(float, p)), n=list(map(float, n)), e=float(e), kind=kind,
             vin=list(map(float, vin)), vout=list(map(float, vout)))
    h.update(kw); ball.hits.append(h); return h


# ------------------------------------------------------------------ helix planner (spiral glockenspiel towers)
def run_helix(ball, p, vin, notes, c, r_t, dirn, vt_t, dzq, tower, emax=EMAX, tilt_max=40.0):
    """ball contact centre p at TN[notes[0]] with incoming velocity vin. Each hop lands on the helix around centre c
    (radius drifting to r_t), going round at tangential speed ~vt_t (dirn=+1 counter-clockwise seen from above).
    Landing height: target dzq per quarter of flight, lowered (never raised by more than a hair) until the bounce is
    physical: EMIN <= e <= emax, key tilt <= tilt_max, and the ball arrives fast enough for its next hop."""
    c = np.array(c, float)
    for k in range(len(notes) - 1):
        i, j = notes[k], notes[k + 1]; dt = TN[j] - TN[i]
        dzt = dzq * dt / Q
        if k + 1 < len(notes) - 1:
            dt2 = TN[notes[k + 2]] - TN[j]; need = (dzq / Q + 0.5 * G * dt2) / (emax * 0.985)
        else: need = 0.0
        rel = p[:2] - c; ri = np.linalg.norm(rel); th = math.atan2(rel[1], rel[0])
        rj = ri + float(np.clip(r_t - ri, -0.1, 0.1))
        vh = np.linalg.norm(vin[:2]); v0 = vh + float(np.clip(vt_t - vh, -0.15, 0.1))
        vts = sorted(set(np.round(np.concatenate([[v0], v0 + np.linspace(-0.4, 0.15, 23)]), 4)), key=lambda v: abs(v - v0))
        dzs = sorted(np.concatenate([dzt - np.linspace(0, 0.5 * G * dt * dt + 0.9, 320), dzt + np.linspace(0.003, 0.2, 30)]),
                     key=lambda d: (0 if d <= dzt else 1, abs(d - dzt)))
        best = None; cands = []
        for vt in [v for v in vts if v > 0.3][:8]:
            thj = th + dirn * vt * dt / (0.5 * (ri + rj))
            land_xy = c + rj * np.array([math.cos(thj), math.sin(thj)])
            for dz in dzs:
                d = np.array([land_xy[0] - p[0], land_xy[1] - p[1], dz]); vo = vout_for(d, dt)
                e, n = bounce_e(vin, vo)
                if not (EMIN <= e <= emax) or vo[2] - G * dt > -0.12: continue
                if math.degrees(math.acos(max(-1, min(1, n[2])))) > tilt_max: continue
                cands.append((d, vo))
                if -(vo[2] - G * dt) >= need: best = (d, vo); break
            if best is not None: break
        if best is None and cands: best = cands[0]; print('WARN need', ball.name, i)
        if best is None: raise RuntimeError(f'no hop {ball.name} {i}')
        d, vo = best
        add_hit(ball, i, TN[i], p, vin, vo, 'key', tower=tower)
        ball.fly(TN[i], TN[j], p, vo)
        p = p + vo * dt + np.array([0, 0, -0.5 * G * dt * dt]); vin = vo + np.array([0, 0, -G * dt])
    return p, vin


# ------------------------------------------------------------------ 1-D rolling along a polyline (pipes, rails)
def poly(pts, step=0.02):
    pts = [np.array(q, float) for q in pts]; out = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        for k in range(1, n + 1): out.append(a + (b - a) * k / n)
    P = np.array(out); s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return P, s


def roll(P, s, v0, t0, crr=0.18, stop_at_end=True, dt=1 / 240, max_t=60, S0=0.0):
    """integrate a rolling ball along the polyline (arc length s): dv/dt = -(5/7) g dz/ds - crr*sign(v).
    returns times, positions, final speed. Stops when it reaches the end (or settles)."""
    z = P[:, 2]; dzds = np.gradient(z, s)
    S = S0; v = v0; T = [t0]; X = [np.array([np.interp(S, s, P[:, k]) for k in range(3)])]; t = t0
    while t - t0 < max_t:
        sl = float(np.interp(S, s, dzds))
        a = -GR * sl - crr * (1 if v > 0 else -1 if v < 0 else 0)
        v += a * dt
        if S + v * dt >= s[-1]:
            T.append(t + (s[-1] - S) / max(1e-6, v)); X.append(P[-1]); break
        S += v * dt; t += dt
        if S <= 0 and v < 0: S = 0; T.append(t); X.append(P[0]); break
        T.append(t); X.append(np.array([np.interp(S, s, P[:, k]) for k in range(3)]))
    return np.array(T), np.array(X), v


def resample(T, X, dt=1 / 240):
    tt = np.arange(T[0], T[-1] + 1e-9, dt)
    return tt, np.stack([np.interp(tt, T, X[:, k]) for k in range(3)], 1)


def smoothstep(u): u = min(1, max(0, u)); return u * u * (3 - 2 * u)


# ==================================================================== geometry of the hall
XC = -0.15                                  # conductor / hall centre line
TOWER = {'L': np.array([XC - 2.75, 0.0]), 'R': np.array([XC + 2.75, 0.0])}
DIRN = {'L': +1, 'R': -1}                   # ball passes the front of each tower moving towards the centre
Z_TOP = 4.35                                # contact height of a ball resting in the lift cup at the top
R_HEL = 1.0; VT = 1.95
STAGE_Z = 0.0
SEAT_ROWS, SEAT_COLS = 7, 11; SEAT_DX = 0.62; ROW_DY = 0.72; ROW_DZ = 0.86
SEAT_Y0, SEAT_Z0 = 2.6, 4.75               # row 0 seat surface
def seat_pos(r, c): return np.array([XC + (c - (SEAT_COLS - 1) / 2) * SEAT_DX, SEAT_Y0 + r * ROW_DY, SEAT_Z0 + r * ROW_DZ + R_A])
EMPTY = (SEAT_ROWS - 1, SEAT_COLS // 2)     # top row, centre
PODIUM = np.array([XC, -1.25, 0.95])        # drum-head top centre
SHAFT_X = 4.5; SHAFT_Y = 0.6
BASE_Z = -2.25                              # basement floor
CAT_Z = 12.15                               # catwalk at the top of the shaft

# ==================================================================== players on the spiral towers
B = {}
def lift_ride(ball, t0, t1, p0, p1):
    """ball sits in the lift cup while the cup travels p0 -> p1 (eased)"""
    tt = np.arange(t0, t1 + 1e-9, 1 / 240); P = [p0 + (p1 - p0) * smoothstep((t - t0) / (t1 - t0)) for t in tt]
    ball.samp(t0, P)


def release(ball, tower, t_first, ang):
    """gate opens: the ball rolls down the short chute of the lift cup at the top and drops onto the first key.
    returns contact point & velocity at the first key"""
    c = TOWER[tower]; dirn = DIRN[tower]
    beta = math.radians(30); a = GR * math.sin(beta); TR = 0.6; TF = 0.26
    tang = np.array([-math.sin(ang), math.cos(ang)]) * dirn
    d = np.array([tang[0] * math.cos(beta), tang[1] * math.cos(beta), -math.sin(beta)])
    p_top = np.array([c[0] + 0.86 * math.cos(ang), c[1] + 0.86 * math.sin(ang), Z_TOP])
    t_rel = t_first - TF - TR
    tt = np.arange(t_rel, t_rel + TR + 1e-9, 1 / 240)
    P = [p_top + d * 0.5 * a * (t - t_rel) ** 2 for t in tt]
    ball.samp(t_rel, P)
    pe = p_top + d * 0.5 * a * TR ** 2; ve = d * a * TR
    ball.fly(t_rel + TR, t_first, pe, ve)
    p1 = pe + ve * TF + np.array([0, 0, -0.5 * G * TF ** 2]); v1 = ve + np.array([0, 0, -G * TF])
    return p_top, t_rel, p1, v1


def catch(ball, tower, p, vin, i_last, t_after):
    """after the last key of a run: fly into the lift cup (parked lower down, at the angle the ball is heading to)"""
    c = TOWER[tower]; dirn = DIRN[tower]
    best = None
    for tc in [s for s in STAB if s > TN[i_last] + 0.3][:4] + [TN[i_last] + 0.5, TN[i_last] + 0.6]:
        dt = tc - TN[i_last]
        rel = p[:2] - c; th = math.atan2(rel[1], rel[0]); thc = th + dirn * 0.9
        cup_xy = c + 0.86 * np.array([math.cos(thc), math.sin(thc)])
        for dz in np.linspace(-0.6, -2.2, 60):
            d = np.array([cup_xy[0] - p[0], cup_xy[1] - p[1], dz]); vo = vout_for(d, dt); e, n = bounce_e(vin, vo)
            if EMIN <= e <= EMAX and math.degrees(math.acos(min(1, n[2]))) < 40 and p[2] + dz > 0.45:
                best = (tc, d, vo, thc); break
        if best: break
    if best is None: print('catch fail', ball.name, p, vin); raise SystemExit
    tc, d, vo, thc = best
    add_hit(ball, i_last, TN[i_last], p, vin, vo, 'key', tower=tower)
    ball.fly(TN[i_last], tc, p, vo)
    pc = p + d; ball_in = vo + np.array([0, 0, -G * (tc - TN[i_last])])
    SFX.append(['cup', float(tc)])
    return pc, float(tc), thc


def note_range(t0, t1): return [i for i in range(NN) if t0 - 1e-4 <= TN[i] < t1 - 1e-4]


RUNS = []       # (ball, tower, first note time, notes)
I_IN = note_range(T0, KB(3) + 0.1); I_A1 = note_range(KB(3) + 0.1, KB(12)); I_A2 = note_range(KB(12), KB(20)); I_B1 = note_range(KB(20), KB(28)); I_B2 = note_range(KB(28), KB(36))
I_R1 = note_range(KB(36), KB(44)); I_R2 = note_range(KB(44), FINAL + 1)
PLAN = [('G1', 'L', I_IN, 0), ('G2', 'R', I_A1, 0), ('G1', 'L', I_A2, 0), ('G2', 'R', I_B1, 0), ('G1', 'L', I_B2, 0),
        ('G2', 'R', I_R1, 0), ('G1', 'L', I_R2, 0)]
for n in ('G1', 'G2'): B[n] = Ball(n, R_P)
state = {'G1': dict(ang=math.radians(130), p=None, t=0.0), 'G2': dict(ang=math.radians(50), p=None, t=0.0)}
# both players wait in their lift cups at the tower tops from the start
LIFT = {'L': [], 'R': []}       # lift-cup keyframes (t, angle, z, carrying)
for k_, (n, tw, notes, _) in enumerate(PLAN):
    b = B[n]; st = state[n]; ang = st['ang']
    t1 = TN[notes[0]]
    if st['p'] is None:                                        # first run of this ball: waits at the top from t=0
        c = TOWER[tw]; p_top = np.array([c[0] + 0.86 * math.cos(ang), c[1] + 0.86 * math.sin(ang), Z_TOP])
        b.rest(0.0, t1 - 0.86, p_top); LIFT[tw].append((0.0, ang, Z_TOP, 1))
    else:                                                      # ride the lift up from the last catch
        pc, tc = st['p'], st['t']
        c = TOWER[tw]; p_top = np.array([c[0] + 0.86 * math.cos(ang), c[1] + 0.86 * math.sin(ang), Z_TOP])
        t_up0 = tc + 0.35; t_up1 = t1 - 0.86 - 0.4
        b.rest(tc, t_up0, pc); lift_ride(b, t_up0, t_up1, pc, p_top); b.rest(t_up1, t1 - 0.86, p_top)
        LIFT[tw] += [(tc, ang, pc[2], 1), (t_up0, ang, pc[2], 1), (t_up1, ang, Z_TOP, 1)]
    p_top, t_rel, p, v = release(b, tw, t1, ang)
    LIFT[tw].append((t_rel, ang, Z_TOP, 1)); LIFT[tw].append((t_rel + 0.5, ang, Z_TOP, 0))
    SFX.append(['gate', float(t_rel)])
    p, v = run_helix(b, p, v, notes, TOWER[tw], R_HEL, DIRN[tw], VT, -0.012, tw, emax=0.985)
    if k_ < len(PLAN) - 2 or n == 'G2' and k_ == len(PLAN) - 1 or True:
        pc, tc, thc = catch(b, tw, p, v, notes[-1], None)
        st['p'], st['t'], st['ang'] = pc, tc, thc
        # empty cup goes down and round to the catch point right after the release
        LIFT[tw] += [(t_rel + 0.9, ang, Z_TOP, 0), (t_rel + 2.4, ang, pc[2] - 0.0, 0), (t_rel + 3.2, thc, pc[2], 0), (tc, thc, pc[2], 1)]
    RUNS.append(dict(ball=n, tower=tw, t_rel=float(t_rel), t_first=float(t1), t_catch=tc, notes=[int(x) for x in notes]))
for n in ('G1', 'G2'):
    st = state[n]; B[n].rest(st['t'], 80.0, st['p'])
for tw in LIFT: LIFT[tw].sort(key=lambda x: x[0])

# ==================================================================== conductor: drop from the ceiling, then a hop on every beat
C = Ball('cond', R_C); B['cond'] = C
pc0 = PODIUM + np.array([0, 0, R_C])
TD = T0 - 0.02                                             # falls from the ceiling hatch (z ~ 14.4) from frame 1
C.rest(0.0, T0 - TD, pc0 + np.array([0, 0, 0.5 * G * TD ** 2]))
C.fly(T0 - TD, T0, pc0 + np.array([0, 0, 0.5 * G * TD ** 2]), [0, 0, 0])
EV['cond_drop'] = dict(t0=T0 - TD, h=0.5 * G * TD ** 2)
def cond_tau(t):
    if t < KB(4): return 0.40
    if t < KB(12): return 0.26
    if t < KB(20): return 0.52
    if t < KB(32): return 0.32
    if t < KB(36): return 0.32 + 0.2 * (t - KB(32)) / (4 * BAR)
    return 0.53
CHOPS = []
tprev = T0
for bt in BEATS[1:]:
    if bt > FINAL + 1e-6: break
    tau = min(cond_tau(bt), bt - tprev - 0.02)
    C.rest(tprev, bt - tau, pc0); C.fly(bt - tau, bt, pc0, [0, 0, 0.5 * G * tau]); CHOPS.append([float(bt), float(tau)])
    tprev = bt
# after the final chord: two joyful hops, then rest
for k in range(2):
    tl = FINAL + 0.9 + 0.8 * k; tau = 0.6
    C.rest(tprev, tl - tau, pc0); C.fly(tl - tau, tl, pc0, [0, 0, 0.5 * G * tau]); CHOPS.append([float(tl), tau]); tprev = tl
C.rest(tprev, 80.0, pc0)
EV['cond_hops'] = CHOPS

# ==================================================================== audience: claps (hop & land on the beat), waves
def clap_tau(t, c):
    """hop duration for column c landing on beat t (0 = no clap)"""
    if t < KB(4) - 1e-6: return 0.0
    if t < KB(12) - 1e-6: return 0.24                                        # piano: small polite claps
    if t < KB(20) - 1e-6:                                                    # forte: a wave left -> right
        ph = (t - KB(12)) / (2 * BAR) - c / SEAT_COLS
        return 0.27 + 0.27 * (0.5 + 0.5 * math.cos(2 * math.pi * ph))
    if t < KB(32) - 1e-6: return 0.24
    if t < KB(36) - 1e-6: return 0.24 + 0.22 * (t - KB(32)) / (4 * BAR)
    if t < FINAL - 1e-6:                                                     # return: wave right -> left, bigger
        ph = (t - KB(36)) / (2 * BAR) + c / SEAT_COLS
        return 0.30 + 0.25 * (0.5 + 0.5 * math.cos(2 * math.pi * ph))
    if abs(t - FINAL) < 1e-6: return 0.54
    return 0.0
AUD = {}
rng = np.random.default_rng(7)
CLAPS = []                                                                    # (t, loudness 0..1) for the audio
for bt in BEATS + [FINAL]:
    if abs(bt - FINAL) < 1e-6 and bt in BEATS and bt != BEATS[-1]: pass
for r in range(SEAT_ROWS):
    for c in range(SEAT_COLS):
        if (r, c) == EMPTY: continue
        hops = []
        tprev = 0.0
        for bt in sorted(set([round(b, 5) for b in BEATS if b <= FINAL + 1e-6])):
            tau = clap_tau(bt, c)
            if tau <= 0: continue
            tau = min(tau, bt - tprev - 0.03); hops.append([bt, tau]); tprev = bt
        # standing ovation after the last chord: free applause hops
        t = FINAL + 0.25 + rng.uniform(0, 0.25)
        while t < FINAL + 4.0:
            tau = rng.uniform(0.28, 0.5); tl = t + tau; hops.append([float(tl), float(tau)]); t = tl + rng.uniform(0.02, 0.12)
        AUD[f'{r}_{c}'] = hops
CL = {}
for name, hops in AUD.items():
    for tl, tau in hops: CL.setdefault(round(tl, 3), []).append(tau)
CLAPS = sorted([[t, len(v), float(np.mean(v))] for t, v in CL.items()])

# ==================================================================== the late marble
L = Ball('late', R_L); B['late'] = L
# (a) basement stairs, hopping on the OFF-beats (the chord stabs)
door = np.array([-4.85, -0.55, -0.62])
st_t = [s for s in STAB if s >= KB(5) + 0.1][:13]
t_first = st_t[0]
# rolls out of the lobby door down a short ramp, drops onto the first step on an off-beat
beta = math.radians(18); a = GR * math.sin(beta); TR = 0.7; TF = 0.22
d = np.array([math.cos(beta), 0, -math.sin(beta)])
t_rel = t_first - TF - TR
L.rest(0.0, t_rel, door)
tt = np.arange(t_rel, t_rel + TR + 1e-9, 1 / 240); L.samp(t_rel, [door + d * 0.5 * a * (t - t_rel) ** 2 for t in tt])
pe = door + d * 0.5 * a * TR ** 2; ve = d * a * TR
L.fly(t_rel + TR, t_first, pe, ve)
p = pe + ve * TF + np.array([0, 0, -0.5 * G * TF ** 2]); v = ve + np.array([0, 0, -G * TF])
EV['late_door'] = dict(p=door.tolist(), t_rel=float(t_rel))
STEPS = []
for k in range(len(st_t) - 1):
    dt = st_t[k + 1] - st_t[k]
    vx = 0.55
    dzt = -0.11
    best = None
    for dz in np.linspace(dzt, dzt - 0.6, 200):
        vo = vout_for(np.array([vx * dt, 0, dz]), dt); e, n = bounce_e(v, vo)
        if EMIN <= e <= 0.9: best = vo; break
    vo = best
    h = add_hit(L, -1, st_t[k], p, v, vo, 'step'); STEPS.append(h); SFX.append(['step', float(st_t[k])])
    L.fly(st_t[k], st_t[k + 1], p, vo)
    p = p + vo * dt + np.array([0, 0, -0.5 * G * dt * dt]); v = vo + np.array([0, 0, -G * dt])
# last landing: onto the carpet of the corridor (dead bounce, keeps rolling right)
t_land = st_t[-1]
z_floor = p[2]
add_hit(L, -1, t_land, p, v, np.array([v[0], 0, 0]), 'carpet'); SFX.append(['step', float(t_land)])
EV['stairs'] = dict(t0=float(st_t[0]), t1=float(t_land), floor=float(z_floor))
# (b) corridor: gentle downhill to the elevator (visible slope), rolling resistance
cab0 = np.array([SHAFT_X, -0.55, None], dtype=object)
x_cab = SHAFT_X - 0.05
corr = [p.copy(), np.array([x_cab - 0.6, p[1], p[2] - 0.3]), np.array([x_cab - 0.3, p[1], p[2] - 0.3])]
Pp, ss = poly(corr)
Tt, X, vend = roll(Pp, ss, v[0], t_land, crr=0.1)
tt, XX = resample(Tt, X); L.samp(tt[0], XX)
# into the cabin: rolls up a short cushioned dip and settles (kinematic, decelerating)
t_in = tt[-1]; p_in = XX[-1]
z_cab_floor = p_in[2]
TS = 0.55
ts_ = np.arange(t_in, t_in + TS + 1e-9, 1 / 240)
P = [p_in + np.array([vend * TS / 2 * (1 - (1 - (t - t_in) / TS) ** 2), 0, 0]) for t in ts_]
L.samp(t_in, P); p_cab = P[-1]
EV['corridor'] = dict(pts=[q.tolist() for q in corr], t_in=float(t_in))
# (c) elevator: departs on a beat, arrives at the top on a beat (ding)
t_dep = min(b for b in BEATS if b > t_in + TS + 0.4)
z_top_cab = CAT_Z
t_arr = min(b for b in BEATS if b > t_dep + 5.2)
L.rest(t_in + TS, t_dep, p_cab)
dz_lift = z_top_cab - p_cab[2]
tt = np.arange(t_dep, t_arr + 1e-9, 1 / 240)
L.samp(t_dep, [p_cab + np.array([0, 0, dz_lift * smoothstep((t - t_dep) / (t_arr - t_dep))]) for t in tt])
p_up = p_cab + np.array([0, 0, dz_lift])
EV['elevator'] = dict(x=float(p_cab[0]), y=float(p_cab[1]), z0=float(p_cab[2]), z1=float(p_up[2]), t_dep=float(t_dep), t_arr=float(t_arr),
                      t_in=float(t_in))
SFX.append(['lift_go', float(t_dep)]); SFX.append(['ding', float(t_arr)])
# (d) the cabin floor tilts, the ball rolls out left along the catwalk (downhill) and settles in the see-saw cup
t_tilt = t_arr + 0.35
ss_piv = np.array([1.9, -0.55, CAT_Z - 0.55]); ARM = 1.0; ARM_W = 0.5; TILT = 14.0
cup_r = ss_piv + np.array([ARM * math.cos(math.radians(-TILT)), 0, ARM * math.sin(math.radians(-TILT))]) + np.array([0, 0, 0.05 + R_L])
path = [p_up, p_up + np.array([-0.45, 0, -0.03]), np.array([3.25, p_up[1], p_up[2] - 0.12]), np.array([cup_r[0] + 0.12, p_up[1], cup_r[2] + 0.05]), cup_r]
Pp, ss = poly(path[:-1])
L.rest(t_arr, t_tilt, p_up)
Tt, X, vend = roll(Pp, ss, 0.0, t_tilt, crr=0.05)
tt, XX = resample(Tt, X); L.samp(tt[0], XX)
d_in = np.linalg.norm(cup_r - XX[-1]); TC = 2 * d_in / vend                 # into the cup: even deceleration to rest
ts_ = np.arange(tt[-1], tt[-1] + TC + 1e-9, 1 / 240)
L.samp(tt[-1], [XX[-1] + (cup_r - XX[-1]) * (1 - (1 - (t - tt[-1]) / TC) ** 2) for t in ts_]); t_cup = ts_[-1]
SFX.append(['thud', float(t_cup)])
EV['catwalk'] = dict(pts=[q.tolist() for q in path], t_tilt=float(t_tilt), t_cup=float(t_cup))
# (e) the weight drops on the high end on a beat -> the ball is flung up-left into the mouth of the glass pipe
t_launch = min(b for b in BEATS if b > t_cup + 0.9)
TFALL = 0.64
mouth = np.array([1.6, 7.45, 13.45])
T_FL = 1.15
vl = vout_for(mouth - cup_r, T_FL)
L.rest(t_cup, t_launch, cup_r)
L.fly(t_launch, t_launch + T_FL, cup_r, vl)
EV['seesaw'] = dict(pivot=ss_piv.tolist(), arm=ARM, arm_w=ARM_W, tilt=TILT, cup=cup_r.tolist(), t_launch=float(t_launch), t_release=float(t_launch - TFALL),
                    fall=0.5 * G * TFALL ** 2, v=vl.tolist(), mouth=mouth.tolist(), t_mouth=float(t_launch + T_FL))
SFX.append(['weight', float(t_launch)]); SFX.append(['release', float(t_launch - TFALL)]); SFX.append(['clink', float(t_launch + T_FL)])
# (f) the glass pipe: funnel mouth catches it, it rolls down along the back wall: left, back right, a WRONG turn up the
#     spur (rolls up, slows, rolls back), then down the last leg to the golden funnel above its seat
v_m = vl + np.array([0, 0, -G * T_FL])
YP = 7.75
seat = seat_pos(*EMPTY)
FUN_C = np.array([seat[0], seat[1], seat[2] + 1.22])        # funnel hole centre height
FUN_R0 = 0.82; DEP = 0.62
z_rim = FUN_C[2] + DEP + R_L * 0.6
def smooth_pts(pts, n=8, rmax=0.3):
    """corner-rounded polyline (quadratic bezier at each corner) - the same path is drawn as the glass pipe"""
    pts = [np.array(p, float) for p in pts]; out = [pts[0]]
    for a_, b_, c_ in zip(pts, pts[1:], pts[2:]):
        la, lc = np.linalg.norm(b_ - a_), np.linalg.norm(c_ - b_); rr_ = min(rmax, la / 2, lc / 2)
        p0 = b_ + (a_ - b_) / la * rr_; p2 = b_ + (c_ - b_) / lc * rr_
        for k in range(n + 1):
            u = k / n; out.append((1 - u) ** 2 * p0 + 2 * u * (1 - u) * b_ + u * u * p2)
    out.append(pts[-1]); return out
def uturn(c, r, n=24):          # left end U-turn: from going -x on top to going +x below (vertical half circle)
    return [np.array([c[0] - r * math.sin(math.pi * k / n), c[1], c[2] + r * math.cos(math.pi * k / n)]) for k in range(n + 1)]
UR = 0.34
J = np.array([3.0, YP, 12.27])                               # junction at the right end of leg 2
YP3 = 7.13                                                   # the last leg runs one pipe in front of leg 2
leg1 = smooth_pts([mouth, np.array([1.25, YP, 13.4]), np.array([-4.0, YP, 13.12 + 0.0])]) + uturn(np.array([-4.0, YP, 13.12 - UR]), UR)[1:]
leg2 = [leg1[-1], np.array([-3.7, YP, 13.12 - 2 * UR - 0.01]), J]
spur = smooth_pts([J, np.array([3.45, YP, J[2] + 0.02]), np.array([4.3, YP, J[2] + 0.42])])
xf = FUN_C[0] + FUN_R0
leg3 = smooth_pts([J, np.array([2.75, YP3, J[2] - 0.08]), np.array([xf + 0.4, YP3, z_rim + 0.05]), np.array([xf, YP3, z_rim + 0.03]),
                   np.array([xf, FUN_C[1] + 0.02, z_rim])], rmax=0.22)
t = t_launch + T_FL
V_IN = 1.05                                                  # the bristle mouth of the pipe soaks up the impact
T_FUN_TARGET = KB(45) + 0.2
def pipe_run(crr, store=False):
    tt_all = []; t = t_launch + T_FL
    Pp, ss = poly(leg1 + smooth_pts(leg2)[1:])
    vm = float(np.linalg.norm(v_m)); TB = 0.3; SB = (vm + V_IN) / 2 * TB          # the bristles brake it evenly
    tb_ = np.arange(t, t + TB + 1e-9, 1 / 240); sb_ = [vm * (u - t) - 0.5 * (vm - V_IN) / TB * (u - t) ** 2 for u in tb_]
    tt_all.append((tb_, np.array([[np.interp(x, ss, Pp[:, k]) for k in range(3)] for x in sb_]))); t = tb_[-1]
    Tt, X, v_e = roll(Pp, ss, V_IN, t, crr=crr, S0=SB); tt_all.append((Tt, X)); t = Tt[-1]; tj = t
    Pp, ss = poly(spur); Tt, X, v_b = roll(Pp, ss, v_e, t, crr=crr); tt_all.append((Tt, X)); t = Tt[-1]; tb = t
    turn = float(Tt[int(np.argmax([np.linalg.norm(x - spur[0]) for x in X]))])
    Pp, ss = poly(leg3); Tt, X, v_f = roll(Pp, ss, abs(v_b), t, crr=crr); tt_all.append((Tt, X)); t = Tt[-1]
    if v_e <= 0 or v_f <= 0: return None
    return t, tt_all, tj, tb, turn, v_f
lo, hi = 0.05, 0.6
for _ in range(40):
    mid = 0.5 * (lo + hi); r_ = pipe_run(mid)
    if r_ is None or r_[0] > T_FUN_TARGET: hi = mid
    else: lo = mid
CRR = lo; t_fun, parts, tj, tb, turn, v_f = pipe_run(CRR)
for Tt, X in parts:
    tt, XX = resample(Tt, X); L.samp(tt[0], XX)
SFX.append(['flap', float(tb)])
EV['pipe'] = dict(main=[q.tolist() for q in leg1 + smooth_pts(leg2)[1:]], spur=[q.tolist() for q in spur],
                  leg3=[q.tolist() for q in leg3], mouth=mouth.tolist(), t_mouth=float(t_launch + T_FL), t_junction=float(tj),
                  t_back=float(tb), t_turn=turn, t_funnel=float(t_fun), y=YP, crr=CRR, v_exit=float(v_f))
# (g) the golden funnel: circles round and round, spiralling in (circular orbit speed on a 1/r funnel), drops through
#     the spout so that it lands in its seat exactly on the final chord
TFALL2 = 0.42
spout_bot = FUN_C[2] - 0.32
v_sp = 0.35
z_land = seat[2]
# free fall from the spout bottom: z_land = spout_bot - v_sp*T - G T^2 / 2  ->  solve T
A_, B_, C_ = 0.5 * G, v_sp, z_land - spout_bot
TFALL2 = (-B_ + math.sqrt(B_ * B_ - 4 * A_ * C_)) / (2 * A_)
t_spout = FINAL - TFALL2
T_SP = 0.3                                                   # down the spout
t_hole = t_spout - T_SP
FUN_RMIN = 0.13                                   # funnel depth rim -> hole
Bk = DEP / (1 / FUN_RMIN - 1 / FUN_R0)
zf = lambda r: FUN_C[2] + 0.62 - Bk * (1 / FUN_R0 - 1 / r) * -1 if False else FUN_C[2] + DEP - Bk * (1 / r - 1 / FUN_R0)
TFUN = t_hole - t_fun
# spiral: r(u) with 1/r linear in u^p (slow at first, faster near the hole); angle from orbital speed
uu = np.linspace(0, 1, int(TFUN * 240) + 1)
rr = 1 / (1 / FUN_R0 + (1 / FUN_RMIN - 1 / FUN_R0) * uu ** 2.2)
v_orb = np.sqrt(GR * Bk / rr)
vv = v_orb + (v_f - v_orb[0]) * np.exp(-uu * 7.0)        # enters fast, friction settles it onto the orbit speed
ang0 = math.atan2((leg3[-1] - FUN_C)[1], (leg3[-1] - FUN_C)[0])
dt_ = TFUN / (len(uu) - 1)
angs = ang0 - np.concatenate([[0], np.cumsum(vv[1:] / rr[1:] * dt_)])     # clockwise seen from above
P = np.stack([FUN_C[0] + rr * np.cos(angs), FUN_C[1] + rr * np.sin(angs), [zf(r) + R_L * 0.6 for r in rr]], 1)
# blend from the pipe exit into the orbit over the first 0.3 s
L.samp(t_fun, P, dt=dt_)
EV['funnel'] = dict(c=FUN_C.tolist(), r0=FUN_R0, rmin=FUN_RMIN, depth=DEP, t0=float(t_fun), t_hole=float(t_hole), turns=float((ang0 - angs[-1]) / (2 * math.pi)))
p_hole = P[-1]
tt = np.arange(t_hole, t_spout + 1e-9, 1 / 240)
p_sp = np.array([FUN_C[0], FUN_C[1], spout_bot])
om0 = vv[-1] / rr[-1]; a_end = angs[-1]; z0_ = p_hole[2]; vz0 = (P[-1][2] - P[-2][2]) / dt_
SP = []
for t in tt:
    u = min(1.0, max(0.0, (t - t_hole) / T_SP))
    an = a_end - om0 * T_SP * (u - u * u / 2); r_ = FUN_RMIN * (1 - u) ** 1.5
    h00, h10, h01, h11 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u, -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
    z = h00 * z0_ + h10 * vz0 * T_SP + h01 * spout_bot + h11 * (-v_sp) * T_SP
    SP.append([FUN_C[0] + r_ * math.cos(an), FUN_C[1] + r_ * math.sin(an), z])
L.samp(t_hole, SP)
L.fly(t_spout, FINAL, p_sp, [0, 0, -v_sp])
add_hit(L, -1, FINAL, seat, [0, 0, -v_sp - G * TFALL2], [0, 0, 0], 'seat')
EV['spout'] = dict(t=float(t_spout), p=p_sp.tolist(), fall=float(TFALL2))
# then it joins the standing ovation (its own applause hops)
hops = []; t = FINAL + 0.35
while t < FINAL + 4.0:
    tau = rng.uniform(0.3, 0.5); hops.append([float(t + tau), float(tau)]); t = t + tau + rng.uniform(0.02, 0.1)
L.rest(FINAL, hops[0][0] - hops[0][1], seat)
for k, (tl, tau) in enumerate(hops):
    L.fly(tl - tau, tl, seat, [0, 0, 0.5 * G * tau])
    nxt = hops[k + 1][0] - hops[k + 1][1] if k + 1 < len(hops) else 80.0
    L.rest(tl, nxt, seat)
AUD['late'] = hops
EV['empty'] = list(EMPTY); EV['seat'] = seat.tolist()


# ==================================================================== key sizes and visibility windows (used by scene.py & verify.py)
for n in ('G1', 'G2'):
    hs = [h for h in B[n].hits if h['kind'] == 'key']
    last = set(r['notes'][-1] for r in RUNS if r['ball'] == n)
    for k, h in enumerate(hs):
        nb = [np.linalg.norm(np.array(h['p']) - np.array(hs[j]['p'])) for j in (k - 1, k + 1) if 0 <= j < len(hs) and abs(hs[j]['t'] - h['t']) < 1.0]
        h['kw'] = float(min(0.2, (min(nb) if nb else 1) - 0.05))
        h['win'] = [1.5, 1.15, 0.12, 0.4] if h['i'] in last else [1.5, 1.15, 0.8, 1.15]

# ==================================================================== checks
def check():
    info = {}
    hs = [(n, h) for n, b in B.items() for h in b.hits]
    keys = [h for n, h in hs if h['kind'] == 'key']
    info['keys'] = len(keys)
    info['e_range'] = (round(min(h['e'] for h in keys), 3), round(max(h['e'] for h in keys), 3))
    info['speed_out_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for n, h in hs if np.linalg.norm(h['vout']) > 0), 3)
    info['tilt_max'] = round(max(math.degrees(math.acos(min(1, h['n'][2]))) for h in keys), 1)
    cov = set(h['i'] for h in keys); info['notes_uncovered'] = sorted(set(range(NN)) - cov)
    for n in ('G1', 'G2'):
        zz = [h['p'][2] for h in B[n].hits]; info[f'z_{n}'] = (round(min(zz), 2), round(max(zz), 2))
    info['runs'] = [(r['ball'], r['tower'], round(r['t_first'], 2), round(r['t_catch'], 2)) for r in RUNS]
    info['late'] = dict(stairs=EV['stairs'], cab_in=round(EV['elevator']['t_in'], 2), dep=round(t_dep, 2), arr=round(t_arr, 2),
                        cup=round(t_cup, 2), launch=round(t_launch, 2), junction=round(EV['pipe']['t_junction'], 2), crr=round(CRR, 3), vexit=round(v_f, 2),
                        back=round(EV['pipe']['t_back'], 2), funnel=round(t_fun, 2), hole=round(t_hole, 2), turns=round(EV['funnel']['turns'], 1))
    return info


if __name__ == '__main__':
    for k, v in check().items(): print(k, v)
    out = dict(G=G, radii=dict(G1=R_P, G2=R_P, cond=R_C, late=R_L, aud=R_A),
               balls={n: dict(r=b.r, seg=b.seg, hits=b.hits) for n, b in B.items()}, runs=RUNS, lift={k: v for k, v in LIFT.items()},
               aud=AUD, claps=CLAPS, ev=EV, sfx=SFX,
               geo=dict(XC=XC, tower={k: v.tolist() for k, v in TOWER.items()}, dirn=DIRN, z_top=Z_TOP, r_hel=R_HEL,
                        rows=SEAT_ROWS, cols=SEAT_COLS, seat_dx=SEAT_DX, row_dy=ROW_DY, row_dz=ROW_DZ, seat_y0=SEAT_Y0, seat_z0=SEAT_Z0,
                        podium=PODIUM.tolist(), shaft_x=SHAFT_X, base_z=BASE_Z, cat_z=CAT_Z))
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
