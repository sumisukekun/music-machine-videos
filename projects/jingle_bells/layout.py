"""Jingle Bells - 'Who wins the sleigh race?' (red vs green jingle-bell marbles racing down a snowy mountain at night).
Physics + story planner. Run after arrange.py; writes layout.json (read by scene.py, render_audio.py, verify.py).

Story (side view, the course runs along +x and descends; every melody note = a racer landing on an ice key of that pitch):
  0-2.4 s   summit start gate, countdown lamps, the gate drops: both bells fall onto their first keys on the pickup note
  V1        neck and neck on two lanes (green: upper / back lane, red: lower / front lane)
  C1        red pulls ahead (snowmen beat the bass on timpani, candy-cane hammers ring the off-beat sleigh bells)
  V2        the green lane's track switch throws green down into the ICE CAVE (green plays the tune on glowing ice crystals,
            red alone on the surface); at bar 40 green drops into the bucket of the ICE ELEVATOR (red alone plays on), the
            elevator (gear wheel + chain) lifts it to the top of a giant SKI JUMP, green rolls down the in-run, flies over red
  C2        lands in the snow bank ahead of red on the chorus downbeat (friction takes the extra speed off), red chases,
            green breaks the finish tape a nose ahead, final chord: green lands in the star on top of the Christmas tree,
            red in an ornament below; the tree lights up from the star down (chain reaction)
Physics: one constant gravity G; between contacts pure parabolas; every bounce is an impulse along the key's surface normal
(tangential velocity kept, or reduced by friction on the snow bank) with restitution EMIN <= e <= EMAX, so a bounce never
adds speed. Rolling on the in-run: a = (5/7) g sin(slope) (numerically integrated along the ramp profile). The elevator
lifts the ball (visible motor, gear and chain): the only energy input, and it is where the jump's speed comes from."""
import json, os, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = 7.0
R = 0.2                                    # racer radius
EMAX, EMIN, TILT = 0.95, 0.12, 32.0
LANE_Y = {'G': 0.45, 'R': -0.45}           # green: back lane, red: front lane (camera at -y)
DZL = 1.9                                  # green lane above red lane
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
TN = np.array([l[0] for l in LEAD]); NN = len(TN)
T0, Q, E8, BAR, FINAL = E['t0'], E['q'], E['e8'], E['bar'], E['final']
BT = {int(k): v for k, v in E['bar_t'].items()}
SECS = dict(E['secs'])
SFX = []; EV = {}


def ev_idx(bar, pos):
    for i, l in enumerate(LEAD):
        if l[4] == bar and abs(l[5] - pos) < 1e-6: return i
    raise KeyError((bar, pos))


class Ball:
    def __init__(s, name, r, team): s.name = name; s.r = r; s.team = team; s.seg = []; s.hits = []
    def rest(s, t0, t1, p): s.seg.append(dict(k='rest', t0=float(t0), t1=float(t1), p=[float(c) for c in p]))
    def fly(s, t0, t1, p0, v0): s.seg.append(dict(k='fly', t0=float(t0), t1=float(t1), p=[float(c) for c in p0], v=[float(c) for c in v0]))
    def path(s, ts, ps): s.seg.append(dict(k='path', t0=float(ts[0]), t1=float(ts[-1]), ts=[float(t) for t in ts], ps=[[float(c) for c in p] for p in ps]))
    def at(s, t):
        for g in s.seg:
            if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9: return seg_at(g, t)
        if t < s.seg[0]['t0']: return seg_at(s.seg[0], s.seg[0]['t0'])
        return seg_at(s.seg[-1], s.seg[-1]['t1'])
    def hit(s, t, p, vin, vout, kind, pitch=None, ev=None, n=None, **kw):
        e, n_ = bounce_e(np.array(vin, float), np.array(vout, float))
        if n is not None: n_ = np.array(n, float)
        if 'e_n' in kw: e = kw['e_n']
        d = dict(t=float(t), p=[float(c) for c in p], vin=[float(c) for c in vin], vout=[float(c) for c in vout], n=[float(c) for c in n_],
                 e=float(e), kind=kind, pitch=pitch, ev=ev); d.update(kw); s.hits.append(d); return d


def seg_at(g, t):
    if g['k'] == 'rest': return np.array(g['p'])
    if g['k'] == 'fly':
        u = t - g['t0']; return np.array(g['p']) + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
    ts = np.array(g['ts']); ps = np.array(g['ps'])
    return np.array([np.interp(t, ts, ps[:, c]) for c in range(3)])


def bounce_e(vin, vout):
    """impulse along n ~ vout - vin (tangential velocity kept); restitution e"""
    d = vout - vin; nn = np.linalg.norm(d)
    if nn < 1e-9: return 1.0, np.array([0, 0, 1.0])
    n = d / nn; a = vin @ n
    return (-(vout @ n) / a if a < -1e-9 else 9.0), n


def tilt_of(n): return math.degrees(math.acos(max(-1.0, min(1.0, n[2]))))


def hop_ok(vin, vout, emin=EMIN, emax=EMAX, tilt=TILT):
    e, n = bounce_e(vin, vout)
    return emin <= e <= emax and tilt_of(n) <= tilt, e, n


def run(ball, p, vin, idxs, vx_of, z_of, y, kind_of=lambda i: 'key', emin=EMIN, emax=EMAX, dvx=(-0.12, 0.06), vz_copy=None,
        first=None):
    """ball at contact centre p at TN[idxs[0]] with incoming velocity vin; hops through idxs.
    vx_of(t): target x speed; z_of(t, x): target contact height; vz_copy: {note index: vz_out} copy another ball's vertical
    bounce (then only vx is chosen); first: (emin, emax, mu) override for the first bounce (snow bank with friction).
    Records the hits and flights; returns (p, vin) on arrival at the last note."""
    for k in range(len(idxs) - 1):
        i, j = idxs[k], idxs[k + 1]; dt = TN[j] - TN[i]
        e_lo, e_hi, mu = (first if (k == 0 and first) else (emin, emax, 0.0))
        tx = vx_of(TN[i])
        if mu > 0:                                     # friction: tangential impulse <= mu x normal impulse
            vxs = [None]
        else:
            vxs = sorted(np.round(vin[0] + np.linspace(dvx[0], dvx[1], 13), 4), key=lambda v: abs(v - tx))
        vy = (y - p[1]) / dt
        if vz_copy is not None and i in vz_copy: vzs = [vz_copy[i]]
        else:
            vzs = list(np.linspace(e_lo * abs(vin[2]), e_hi * abs(vin[2]), 60))
        best = None
        j2 = idxs[k + 2] if k + 2 < len(idxs) else None
        for vz in vzs:
            for vx in vxs:
                if vx is None:                                # snow bank (flat, normal = z): friction impulse <= mu x normal impulse
                    jn = vz - vin[2]; mu_eff = float(np.clip((vin[0] - tx) / jn, 0.0, mu)); vx = vin[0] - mu_eff * jn
                    vout = np.array([vx, vy, vz]); e = vz / abs(vin[2]); n = np.array([0, 0, 1.0])
                    if not (e_lo <= e <= e_hi): continue
                else:
                    vout = np.array([vx, vy, vz])
                    ok, e, n = hop_ok(vin, vout, e_lo, e_hi)
                    if not ok: continue
                vland = vz - G * dt
                if vland > -min(0.8, 0.45 * G * dt): continue
                z1 = p[2] + vz * dt - 0.5 * G * dt * dt; x1 = p[0] + vx * dt
                cost = (z1 - z_of(TN[j], x1)) ** 2 + 0.3 * (vx - tx) ** 2
                if j2 is not None and vz_copy is None:
                    dt2 = TN[j2] - TN[j]
                    need = (z_of(TN[j2], x1 + vx * dt2) - z1) / dt2 + 0.5 * G * dt2
                    cost += 6.0 * max(0.0, need - emax * 0.98 * abs(vland)) ** 2
                if best is None or cost < best[0]: best = (cost, vout, e, n, mu_eff if mu > 0 else None)
            if vz_copy is not None and best is not None: break
        if best is None: raise RuntimeError(f'{ball.name}: no legal bounce at note {i} t={TN[i]:.2f}')
        _, vout, e, n, mu_used = best
        if mu_used is not None: ball.hit(TN[i], p, vin, vout, kind_of(i), LEAD[i][1], ev=int(i), n=[0, 0, 1], mu=mu_used, e_n=float(e))
        else: ball.hit(TN[i], p, vin, vout, kind_of(i), LEAD[i][1], ev=int(i))
        ball.fly(TN[i], TN[j], p, vout)
        p = p + vout * dt + np.array([0, 0, -0.5 * G * dt * dt]); vin = vout + np.array([0, 0, -G * dt])
    return p, vin


BALLS = {'R': Ball('red', R, 'R'), 'G': Ball('green', R, 'G')}
RB, GB = BALLS['R'], BALLS['G']
I_V2 = ev_idx(33, 0); I_C2 = ev_idx(49, 0); I_C1 = ev_idx(17, 0)
# ============================================================ start: gates at the summit, both drop onto their first keys
X0 = 0.0; ZR0 = 0.0
# the gate holds each bell at the top of a short steep start ramp (30 deg); the gate drops, the bell rolls down the ramp
# (rolling: a = 5/7 g sin 30) and flies off its lip onto the first key on the pickup note
A0 = math.radians(30.0); V_START = 2.9
a0 = 5 / 7 * G * math.sin(A0); T_RAMP0 = V_START / a0; L_RAMP0 = 0.5 * a0 * T_RAMP0 ** 2
T_F0 = 0.22
T_GATE = float(TN[0] - T_F0 - T_RAMP0)
EV['gate'] = dict(t=T_GATE, alpha=A0, L=L_RAMP0, t_ramp=T_RAMP0, t_fly=T_F0)
SFX.append(['gate', T_GATE])
u0 = np.array([math.cos(A0), 0, -math.sin(A0)])
v_lip0 = V_START * u0
p0 = {}
for team, z0 in (('R', ZR0), ('G', ZR0 + DZL)):
    b = BALLS[team]; y = LANE_Y[team]
    p_key = np.array([X0, y, z0])                                        # contact centre on the first key
    lip = p_key - v_lip0 * T_F0 + np.array([0, 0, 0.5 * G * T_F0 ** 2])
    top = lip - u0 * L_RAMP0
    b.rest(0.0, T_GATE, top)
    tt = np.arange(T_GATE, T_GATE + T_RAMP0 + 1e-9, 1 / 240.0)
    b.path(tt, [top + u0 * (0.5 * a0 * (t - T_GATE) ** 2) for t in tt])
    b.fly(T_GATE + T_RAMP0, TN[0], lip, v_lip0)
    EV.setdefault('start', {})[team] = dict(top=top.tolist(), lip=lip.tolist(), u=u0.tolist())
    p0[team] = p_key
v0 = v_lip0 + np.array([0, 0, -G * T_F0])

# ============================================================ red: the whole tune
SLOPE_T = 0.42                                       # target descent (m per s) of the course
def z_red(t, x=None): return ZR0 - SLOPE_T * (t - TN[0])
def vx_red(t):
    if t < SECS['C1']: return 2.85
    if t < SECS['V2']: return 3.05                         # C1: red pulls ahead
    if t < SECS['C2']: return 3.0
    return RED_C2(t)
RED_C2 = None                                        # set after green's C2 run (red chases green)
# plan red up to C2 first (green's C2 depends on red's position)
pR, vR = run(RB, p0['R'], v0.copy(), list(range(0, I_C2 + 1)), vx_red, z_red, LANE_Y['R'], dvx=(-0.12, 0.08))
VZ_RED = {h['ev']: h['vout'][2] for h in RB.hits}
RED_STATE_C2 = (pR.copy(), vR.copy())

# ============================================================ green V1, C1: same vertical bounce as red (lane 1.9 m higher), own x speed
def vx_green(t):
    if t < SECS['C1']: return 2.92
    return 2.85
I_SW = ev_idx(32, 0)                                  # the long A-flat at the end of C1: the track switch
pG, vG = run(GB, p0['G'], v0.copy(), list(range(0, I_SW + 1)), vx_green, lambda t, x: z_red(t) + DZL, LANE_Y['G'],
             vz_copy=VZ_RED, dvx=(-0.12, 0.08))

# ============================================================ the track switch: green bounces off the tilted switch plate and
# drops through the open trapdoor into the ice cave (red alone plays the next notes), lands on the first ice crystal
I_CAVE0 = ev_idx(33, 1)
DZC = 5.5                                             # cave crystals below green's lane
dt = TN[I_CAVE0] - TN[I_SW]
best = None
for th in np.radians(np.arange(5.0, 56.0, 0.5)):          # the switch plate is tilted forward-down by th
    n = np.array([math.sin(th), 0.0, math.cos(th)])
    for e in np.arange(0.2, 0.81, 0.02):
        vout = vG - (1 + e) * (vG @ n) * n
        if not (1.5 <= vout[0] <= 3.4): continue
        z1 = pG[2] + vout[2] * dt - 0.5 * G * dt * dt
        c = (z1 - (pG[2] - DZC)) ** 2 + 0.05 * (vout[0] - 2.4) ** 2
        if best is None or c < best[0]: best = (c, vout, th, e)
vout = best[1]
assert abs(best[0]) < 0.05, best
GB.hit(TN[I_SW], pG, vG, vout, 'switch', LEAD[I_SW][1], ev=int(I_SW))
GB.fly(TN[I_SW], TN[I_CAVE0], pG, vout)
EV['switch'] = dict(t=float(TN[I_SW]), p=pG.tolist(), n=GB.hits[-1]['n'], vout=vout.tolist())
SFX.append(['trap', float(TN[I_SW]) + 0.05])
pG = pG + vout * dt + np.array([0, 0, -0.5 * G * dt * dt]); vG = vout + np.array([0, 0, -G * dt])
EV['cave'] = dict(z_top=float(pG[2]), x0=float(pG[0]), t0=float(TN[I_CAVE0]))
# ---- the ice cave: bars 33-40 (green's own bounce plan, along a gently descending cave floor)
I_CAVE1 = ev_idx(40, 0)                               # the long C: green's last cave note
zc0 = pG[2]; tc0 = TN[I_CAVE0]
pG, vG = run(GB, pG, vG, list(range(I_CAVE0, I_CAVE1 + 1)), lambda t: 2.9, lambda t, x: zc0 - 0.3 * (t - tc0), LANE_Y['G'],
             kind_of=lambda i: 'crystal', dvx=(-0.15, 0.08), first=(0.04, 0.9, 0.0))
# ---- into the elevator bucket (no note: the bucket is padded): dead landing
T_CATCH = TN[I_CAVE1] + 0.42
dt = T_CATCH - TN[I_CAVE1]
vo = np.array([vG[0], 0.0, 0.55 * abs(vG[2])])
cup = pG + vo * dt + np.array([0, 0, -0.5 * G * dt * dt])
ok, e, n = hop_ok(vG, vo, EMIN, EMAX, TILT); assert ok, ('bucket hop', e, tilt_of(n))
GB.hit(TN[I_CAVE1], pG, vG, vo, 'crystal', LEAD[I_CAVE1][1], ev=int(I_CAVE1))
GB.fly(TN[I_CAVE1], T_CATCH, pG, vo)
SFX.append(['bucket', float(T_CATCH)])

# ============================================================ the elevator + the ski jump: solve the geometry so that green
# lands in the snow bank 2.4 m ahead of red on the first note of C2
t_L = TN[I_C2]
xr_L = float(RED_STATE_C2[0][0])
x_L = xr_L + 2.4
z_L = float(RED_STATE_C2[0][2]) + DZL
ALPHA = math.radians(36.0)                            # in-run slope
BETA = math.radians(10.0)                             # take-off angle (lip points up)
RK = 3.0                                              # radius of the transition curve at the bottom of the in-run
X_E = float(cup[0])                                   # elevator axis
def inrun(L):
    """profile of the in-run: straight slope (length L) from the top, then a circular transition up to +BETA.
    Returns arrays (s, x, z, slope_angle) relative to the top point (0, 0)."""
    ss = []; xs = []; zs = []; th = []
    for s in np.linspace(0, L, 400):
        ss.append(s); xs.append(s * math.cos(ALPHA)); zs.append(-s * math.sin(ALPHA)); th.append(-ALPHA)
    x1, z1 = xs[-1], zs[-1]
    cx, cz = x1 - RK * math.sin(ALPHA), z1 + RK * math.cos(ALPHA) - 0.0
    # centre above the slope (normal (sin a, cos a) points away from the surface, the ball rolls on the inside of the curve)
    cx, cz = x1 + RK * math.sin(-ALPHA) * -1 * 0 + RK * math.sin(ALPHA) * 0, 0
    nx, nz = math.sin(ALPHA), math.cos(ALPHA)                      # unit normal of the slope (up and forward)
    cx, cz = x1 + RK * nx, z1 + RK * nz
    for a in np.linspace(-ALPHA, BETA, 200)[1:]:
        ss.append(L + RK * (a + ALPHA)); xs.append(cx + RK * math.sin(a)); zs.append(cz - RK * math.cos(a)); th.append(a)
    return np.array(ss), np.array(xs), np.array(zs), np.array(th)


def roll(L, dt_=1 / 600):
    """roll from rest along the in-run (rolling ball: a = 5/7 g sin(slope)), integrated along the path of the ball's centre
    (the surface profile offset by R along its normal); returns times, centre positions, exit speed"""
    s_, x_, z_, th_ = inrun(L)
    cxs = x_ - R * np.sin(th_); czs = z_ + R * np.cos(th_)
    sc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(cxs), np.diff(czs)))])
    s = 0.0; v = 0.0; t = 0.0; T = [0.0]; S = [0.0]
    while s < sc[-1]:
        th = float(np.interp(s, sc, th_))
        a = -5 / 7 * G * math.sin(th)
        v += a * dt_; s += v * dt_; t += dt_; T.append(t); S.append(min(s, sc[-1]))
        if v <= 0: raise RuntimeError('stalls')
    S = np.array(S); T = np.array(T)
    return T, np.interp(S, sc, cxs), np.interp(S, sc, czs), v


sol = None
BUDGET = t_L - T_CATCH - 0.3 - 0.8
for L in np.arange(5.0, 14.0, 0.25):
    T_r, cx, cz, v_ex = roll(L)
    vx_, vz_ = v_ex * math.cos(BETA), v_ex * math.sin(BETA)
    for Tf in np.arange(1.0, 2.21, 0.05):
        T_lift = BUDGET - T_r[-1] - Tf
        if T_lift < 2.4: continue
        x_lip = x_L - vx_ * Tf
        z_lip = z_L - vz_ * Tf + 0.5 * G * Tf * Tf                      # ball centre at the lip
        x_top = x_lip - cx[-1]; z_top = z_lip - cz[-1]                  # ball centre at the top of the in-run
        dx = (x_top - 0.6) - X_E; rise = z_top - cup[2]
        if dx < 2.0 or rise < 3.0: continue
        spd = math.hypot(dx, rise) / T_lift
        c = spd + 0.3 * abs(Tf - 1.5)
        if sol is None or c < sol[0]: sol = (c, L, T_r, cx, cz, v_ex, Tf, x_top, z_top, T_lift, dx, rise, spd)
assert sol, 'ski jump geometry'
_, L_IN, T_r, cx, cz, V_EX, T_F, X_TOP, Z_TOP, T_LIFT, DX_LIFT, RISE, SPD_LIFT = sol
# chairlift: the bucket hangs from a cable running diagonally up from the cave exit to the top of the in-run tower;
# it moves with a smooth start / stop (motor + bull wheel), then tips the ball 0.6 m onto the start of the in-run
T_UP0 = T_CATCH + 0.3; T_UP1 = T_UP0 + T_LIFT; T_REL = T_UP1 + 0.8
tt = np.arange(T_UP0, T_UP1 + 1e-9, 1 / 240.0)
def ease(u): return u * u * (3 - 2 * u)
B0 = cup.copy(); B1 = np.array([X_TOP - 0.6, LANE_Y['G'], Z_TOP])
P_UP = [B0 + (B1 - B0) * ease((t - T_UP0) / T_LIFT) for t in tt]
GB.rest(T_CATCH, T_UP0, cup); GB.path(tt, P_UP)
top_c = np.array([X_TOP, LANE_Y['G'], Z_TOP])
tt2 = np.linspace(T_UP1, T_REL, 60)
GB.path(list(tt2), [B1 + (top_c - B1) * ease((t - T_UP1) / (T_REL - T_UP1)) for t in tt2])
tt3 = T_REL + T_r
GB.path(list(tt3), [np.array([X_TOP + x, LANE_Y['G'], Z_TOP + z]) for x, z in zip(cx, cz)])
t_lip = T_REL + T_r[-1]
p_lip = np.array([X_TOP + cx[-1], LANE_Y['G'], Z_TOP + cz[-1]])
v_lip = np.array([V_EX * math.cos(BETA), 0, V_EX * math.sin(BETA)])
GB.fly(t_lip, t_L, p_lip, v_lip)
p_land = p_lip + v_lip * (t_L - t_lip) + np.array([0, 0, -0.5 * G * (t_L - t_lip) ** 2])
v_land = v_lip + np.array([0, 0, -G * (t_L - t_lip)])
assert abs(p_land[0] - x_L) < 0.05 and abs(p_land[2] - z_L) < 0.05, (p_land, x_L, z_L)
EV['elevator'] = dict(p0=B0.tolist(), p1=B1.tolist(), t_catch=float(T_CATCH), t0=float(T_UP0), t1=float(T_UP1), t_tip=float(T_REL),
                      speed=float(SPD_LIFT))
EV['jump'] = dict(x_top=float(X_TOP), z_top=float(Z_TOP), L=float(L_IN), alpha=ALPHA, beta=BETA, rk=RK, t_release=float(T_REL),
                  t_lip=float(t_lip), p_lip=p_lip.tolist(), v_exit=float(V_EX), t_flight=float(T_F), p_land=p_land.tolist(),
                  v_land=v_land.tolist())
SFX.append(['lift', float(T_UP0)]); SFX.append(['tip', float(T_UP1)]); SFX.append(['whoosh', float(t_lip)])

# ============================================================ C2: green from the snow bank (friction) to the finish
I_FIN = ev_idx(64, 0)
I_TAPE = ev_idx(62, 0)                                # the finish line is crossed on bar 62's downbeat
def z_c2(t, x=None):
    i = int(np.searchsorted(TN, t - 1e-6)); return z_red(t) + DZL
def vx_g2(t): return 3.0
pG, vG = run(GB, p_land, v_land, list(range(I_C2, I_FIN)), vx_g2, lambda t, x: z_red(t) + DZL, LANE_Y['G'],
             first=(0.05, 0.4, 0.8), dvx=(-0.3, 0.06),
             kind_of=lambda i: 'snowbank' if i == I_C2 else 'key')
G_C2 = {h['ev']: h for h in GB.hits if h['ev'] is not None and h['ev'] >= I_C2}
xg_tape = G_C2[I_TAPE]['p'][0]

# ============================================================ red C2: chase, cross the line 0.4 m behind green
GAP_END = 0.6
def x_green_at(t):
    hs = sorted(G_C2.values(), key=lambda h: h['t'])
    return float(np.interp(t, [h['t'] for h in hs], [h['p'][0] for h in hs]))
def RED_C2(t):
    xr_now = RB.at(t)[0] if RB.seg and t <= RB.seg[-1]['t1'] + 1e-6 else None
    return 2.95
# iterate red's constant C2 target speed so that it crosses bar 62's downbeat GAP_END behind green
saved = (list(RB.seg), list(RB.hits))
lo, hi = 2.6, 4.2
for it in range(30):
    vt = 0.5 * (lo + hi)
    RB.seg, RB.hits = list(saved[0]), list(saved[1])
    RB.hits = RB.hits[:-0] if False else RB.hits
    pr, vr = run(RB, RED_STATE_C2[0], RED_STATE_C2[1], list(range(I_C2, I_TAPE + 1)), lambda t: vt, z_red, LANE_Y['R'], dvx=(-0.08, 0.08))
    gap = xg_tape - pr[0]
    if gap > GAP_END: lo = vt
    else: hi = vt
VT_RED = vt
RB.seg, RB.hits = list(saved[0]), list(saved[1])
pR, vR = run(RB, RED_STATE_C2[0], RED_STATE_C2[1], list(range(I_C2, I_FIN)), lambda t: VT_RED if t <= TN[I_TAPE] + 0.01 else 2.7,
             z_red, LANE_Y['R'], dvx=(-0.08, 0.08))
xr_tape = [h for h in RB.hits if h['ev'] == I_TAPE][0]['p'][0]
EV['finish'] = dict(t=float(TN[I_TAPE]), x_green=float(xg_tape), x_red=float(xr_tape), vt_red=VT_RED)

# ============================================================ the final note: into the Christmas tree (dead landings)
# last key -> green into the star cup on the tree top, red into an ornament cup on the tree's front side
I_LAST = I_FIN - 1
hR = RB.hits[-1]; hG = GB.hits[-1]
dt = TN[I_FIN] - TN[I_LAST]
# a gentle last hop (e = 0.6, x speed kept) carries each bell into its cup: green into the star cup on the tree top, red into an
# ornament cup on the tree's front side (dead landings: the cups are padded). The tree is placed where green's hop ends.
LANDS = {}
for b, (p, vin), kind in ((GB, (pG, vG), 'star'), (RB, (pR, vR), 'ornament')):
    vo = np.array([vin[0], 0.0, 0.6 * abs(vin[2])])
    ok, e, n = hop_ok(vin, vo, EMIN, EMAX, TILT); assert ok, (b.name, 'final hop', e, tilt_of(n))
    b.hit(TN[I_LAST], p, vin, vo, 'key', LEAD[I_LAST][1], ev=int(I_LAST))
    b.fly(TN[I_LAST], TN[I_FIN], p, vo)
    tgt = p + vo * dt + np.array([0, 0, -0.5 * G * dt * dt]); vland = vo + np.array([0, 0, -G * dt])
    b.hit(TN[I_FIN], tgt, vland, np.zeros(3), kind, LEAD[I_FIN][1], ev=int(I_FIN), n=[0, 0, 1], final=True)
    b.rest(TN[I_FIN], FINAL + 30, tgt)
    LANDS[kind] = tgt
p_star, p_orn = LANDS['star'], LANDS['ornament']
X_TREE = float(p_star[0])
EV['tree'] = dict(x=X_TREE, star=p_star.tolist(), ornament=p_orn.tolist(), t=float(TN[I_FIN]))
SFX.append(['tape', float(TN[I_TAPE]) - 0.02]); SFX.append(['star', float(TN[I_FIN])])

# ============================================================ checks
def check():
    info = {}
    for team, b in BALLS.items():
        hs = [h for h in b.hits if np.linalg.norm(h['vout']) > 0]
        info[f'{b.name}_hits'] = len(b.hits)
        info[f'{b.name}_e(min,max)'] = (round(min(h['e'] for h in hs), 3), round(max(h['e'] for h in hs), 3))
        info[f'{b.name}_speed_out/in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in hs), 3)
        info[f'{b.name}_tilt_max'] = round(max(tilt_of(h['n']) for h in hs), 1)
        zz = [h['p'][2] for h in b.hits]; info[f'{b.name}_z'] = (round(min(zz), 2), round(max(zz), 2))
    cov = set(h['ev'] for b in BALLS.values() for h in b.hits if h['ev'] is not None)
    info['notes_uncovered'] = sorted(set(range(NN)) - cov)
    info['green_only_red_notes'] = len([i for i in range(NN) if i not in set(h['ev'] for h in GB.hits)])
    info['jump'] = dict(L=round(L_IN, 2), v_exit=round(V_EX, 2), flight=round(T_F, 2), rise=round(RISE, 2), lift_dx=round(DX_LIFT, 2),
                        lift_speed=round(SPD_LIFT, 2), lift_time=round(T_LIFT, 2), top=round(Z_TOP, 2))
    gaps = []
    for t in (5, 10, 17, 22, 28, 31, 36, 42, 47, 50, 54, 58, TN[I_TAPE]):
        gaps.append((round(float(t), 2), round(float(GB.at(t)[0] - RB.at(t)[0]), 2)))
    info['gap_green_minus_red'] = gaps
    sb = [h for h in GB.hits if h['kind'] == 'snowbank'][0]
    info['snowbank'] = dict(e=round(sb['e_n'], 3), mu=sb.get('mu'), vin=np.round(sb['vin'], 2).tolist(), vout=np.round(sb['vout'], 2).tolist())
    return info


if __name__ == '__main__':
    info = check()
    for k, v in info.items(): print(k, v)
    out = dict(G=G, R=R, lane_y=LANE_Y, dzl=DZL, balls={b.name: dict(team=b.team, r=b.r, seg=b.seg, hits=b.hits) for b in BALLS.values()},
               ev=EV, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
