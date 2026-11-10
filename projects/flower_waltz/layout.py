"""Waltz of the Flowers - 「一晩で咲く魔法の庭」 (a magic greenhouse that blooms in one night). Physics + story planner.
Run after arrange.py; writes layout.json (read by scene.py, render_audio.py, verify.py).

Story (every melody onset = a ball landing on a flower part that sounds it, and that flower opens):
  vamp    night greenhouse; the moon lamp opens and a green hummingbird ball (H) drops out
  horn1   H hops bud to bud (horn theme); on the long notes it settles in a tulip cup that blooms around it,
          then the tulip's spring stamen flicks it on. Its last tulip pulls a string: watering can 1 tips
  dew1    clarinet answers: a water drop from each can runs down the hanging leaves; each drop ends in the next
          can's lever cup and tips it (chain reaction); the 3rd drop splashes into the pond
  horn2   H again (horn theme), further along the beds
  cold    F#-minor: a gust bursts the roof window open, frost and snow blow in, the flowers droop;
          H races over frozen buds and takes shelter in a tulip
  gold    the gust spins the windmill on the roof; its cam opens the seed rack gate once per phrase and a golden
          seed ball drops and plays the violin theme down its own bell-flower cascade into the counterweight bucket.
          Each ball = one ratchet notch: the bucket sinks, the rope pulls the window a quarter closed
  climb   4th ball -> window shut, wind stops; H is flicked up and plays the rising climb back to the horn theme
  horn3   warm again: H on new beds; dew2 = three more drops (new cans further along)
  coda    sunrise; H hops the last turns, final chords on the giant sunflower that opens as the sun comes up
Physics: one constant gravity G; between hits pure parabolas; at a hit the flower's surface normal is chosen so the
impulse is along the normal (tangential velocity kept) with restitution e <= EMAX (never gains speed).
Speed-ups only come from mechanisms (spring stamens, can tipping, rack gate)."""
import json, os, math
import numpy as np
from phys import G, R, at as seg_at_list
HERE = os.path.dirname(os.path.abspath(__file__))
EMAX = 0.95
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
TN = np.array([l['t'] for l in LEAD]); NN = len(TN); Q = E['q']; BAR = E['bar']
SEC = {}
for i, l in enumerate(LEAD): SEC.setdefault(l['sec'], []).append(i)
BT = {int(k): v for k, v in E['bar_t'].items()}


class Ball:
    def __init__(s, name, kind):
        s.name = name; s.kind = kind; s.seg = []; s.hits = []
    def rest(s, t0, t1, p): s.seg.append(dict(k='rest', t0=float(t0), t1=float(t1), p=list(map(float, p))))
    def fly(s, t0, t1, p0, v0): s.seg.append(dict(k='fly', t0=float(t0), t1=float(t1), p=list(map(float, p0)), v=list(map(float, v0))))
    def lin(s, t0, t1, p0, v0): s.seg.append(dict(k='lin', t0=float(t0), t1=float(t1), p=list(map(float, p0)), v=list(map(float, v0))))
    def at(s, t): return seg_at_list(s.seg, t)


def vout_for(dx, dz, dt, dy=0.0): return np.array([dx / dt, dy / dt, dz / dt + 0.5 * G * dt])
def after(p, v, dt): return p + v * dt + np.array([0, 0, -0.5 * G * dt * dt]), v + np.array([0, 0, -G * dt])


def bounce_e(vin, vout):
    d = vout - vin; n = d / max(1e-9, np.linalg.norm(d)); a = vin @ n
    return (-(vout @ n) / a if a < -1e-9 else 9.0), n


def hit(ball, i, p, vin, vout, kind, t=None, part=None):
    e, n = bounce_e(np.array(vin, float), np.array(vout, float))
    if np.linalg.norm(vout) < 1e-9: n = np.array([0, 0, 1.0]); e = 0.0
    ball.hits.append(dict(i=int(i), t=float(TN[i] if t is None else t), p=list(map(float, p)), n=list(map(float, n)), e=float(e),
                          kind=kind, vin=list(map(float, vin)), vout=list(map(float, vout)), ball=ball.name))


def run(ball, p, vin, notes, vx_of, dz_of=lambda i: -0.02, emax=EMAX, emin=0.12, tilt_max=38.0, vy_of=lambda i: 0.0,
        dvx=(-0.15, 0.15), kind_of=lambda i: 'bud'):
    """ball at contact centre p at time TN[notes[0]] with incoming velocity vin; hops through notes[1:].
    The landing height is lowered (never raised) until the bounce is physical (emin <= e <= emax, tilt <= tilt_max) and
    the ball arrives at the next flower fast enough to make that flower's own hop with e <= emax.
    returns (contact centre, incoming velocity) at the last note."""
    for k in range(len(notes) - 1):
        i = notes[k]; j = notes[k + 1]; dt = TN[j] - TN[i]
        dzt = dz_of(i) * dt / Q
        if k + 1 < len(notes) - 1:
            dt2 = TN[notes[k + 2]] - TN[j]
            need = (dz_of(j) / Q + 0.5 * G * dt2) / (emax * 0.99)
        else: need = 0.0
        tx = vx_of(i); cx = vin[0]
        vx0 = cx + float(np.clip(tx - cx, dvx[0], dvx[1]))
        vxs = sorted(set(np.round(np.concatenate([[vx0], vx0 + np.linspace(-0.3, 0.3, 25)]), 4)), key=lambda v: abs(v - vx0))
        dzs = sorted(np.concatenate([dzt - np.linspace(0, 0.5 * G * dt * dt + 1.0, 300), dzt + np.linspace(0.003, 0.25, 40)]),
                     key=lambda d: (0 if d <= dzt else 1, abs(d - dzt)))
        best = None; cands = []
        for vx in vxs[:8]:
            for dz in dzs:
                vo = vout_for(vx * dt, dz, dt, vy_of(i) * dt)
                e, n = bounce_e(vin, vo)
                if not (emin <= e <= emax) or vo[2] - G * dt > -0.15: continue
                tilt = math.degrees(math.acos(max(-1, min(1, n[2]))))
                if tilt > tilt_max: continue
                cands.append((dz, vo, e, n))
                if -(vo[2] - G * dt) >= need: best = (dz, vo, e, n); break
            if best is not None: break
        if best is None and cands: best = cands[0]; print('WARN need', ball.name, i, round(need, 2))
        if best is None:
            allc = []
            for vx in vxs:
                for dz in dzs[::3]:
                    vo = vout_for(vx * dt, dz, dt, vy_of(i) * dt); e, n = bounce_e(vin, vo)
                    tilt = math.degrees(math.acos(max(-1, min(1, n[2]))))
                    allc.append((abs(min(0, e - emin)) + max(0, e - emax) + max(0, tilt - tilt_max) / 30, dz, vo, e, n))
            _, dz, vo, e, n = min(allc, key=lambda c: c[0]); best = (dz, vo, e, n)
            print('WARN e/tilt', ball.name, i, round(e, 3))
        dz, vo, e, n = best
        hit(ball, i, p, vin, vo, kind_of(i))
        ball.fly(TN[i], TN[j], p, vo)
        p, vin = after(p, vo, dt)
    return p, vin


def bounce_to(ball, i, p, vin, t1, target, kind='bud', tilt_max=40.0):
    """hop from note i (contact p) to time t1, landing as close as physics allows to target: the horizontal speed may only
    shrink (rigid bounce keeps the tangential part), and the landing is lowered until e <= EMAX and tilt <= tilt_max."""
    dt = t1 - TN[i]; tg = np.array(target, float); d = tg - p
    best = None
    for f in np.linspace(1.0, 0.3, 36):
        vx = d[0] / dt * f
        if abs(vx) > abs(vin[0]) * 1.0 + 0.4: vx = math.copysign(abs(vin[0]) + 0.4, vx)
        for dz in sorted(d[2] + np.linspace(-2.5, 0.4, 291), key=lambda z: abs(z - d[2])):
            vo = np.array([vx, d[1] / dt, dz / dt + 0.5 * G * dt])
            e, n = bounce_e(vin, vo)
            tilt = math.degrees(math.acos(max(-1, min(1, n[2]))))
            if 0.0 <= e <= EMAX and tilt <= tilt_max:
                best = vo; break
        if best is not None: break
    if best is None: raise RuntimeError(f'bounce_to failed {ball.name} {i}')
    hit(ball, i, p, vin, best, kind); ball.fly(TN[i], t1, p, best)
    return after(p, best, dt)


def hop_onto(ball, i, p, vin, zfun, kind, dts=np.arange(0.3, 0.95, 0.01), tilt_max=40.0):
    """hop from note i onto a surface z = zfun(x) (e.g. a gutter): searches the flight time and horizontal speed
    (never faster than it came in) for a physical bounce. returns (landing time, point, arrival velocity)."""
    for dt in dts:
        for f in np.linspace(1.0, 0.5, 11):
            vx = vin[0] * f; x1 = p[0] + vx * dt; z1 = zfun(x1)
            vo = np.array([vx, 0.0, (z1 - p[2]) / dt + 0.5 * G * dt])
            e, n = bounce_e(vin, vo); tilt = math.degrees(math.acos(max(-1, min(1, n[2]))))
            if 0.1 <= e <= EMAX and tilt <= tilt_max:
                hit(ball, i, p, vin, vo, kind); ball.fly(TN[i], TN[i] + dt, p, vo)
                pp, vv = after(p, vo, dt)
                return TN[i] + dt, pp, vv
    raise RuntimeError(f'hop_onto failed {ball.name} {i}')


def flight(ball, t0, t1, p0, p1):
    """free flight from p0 (t0) to p1 (t1); returns arrival velocity"""
    dt = t1 - t0; d = np.array(p1, float) - np.array(p0, float)
    v0 = np.array([d[0] / dt, d[1] / dt, d[2] / dt + 0.5 * G * dt])
    ball.fly(t0, t1, p0, v0)
    va = v0 + np.array([0, 0, -G * dt])
    assert va[2] < -0.3, (ball.name, t0, t1, va)
    return v0, va


B = {}
def ball(name, kind):
    B[name] = Ball(name, kind); return B[name]
EV = {'cups': [], 'cans': [], 'springs': [], 'ponds': []}
SFX = []
LY = {'H': 0.0, 'dew': -0.35, 'gold': 0.35}

# ===================================================================== H: horn theme 1
H = ball('H', 'bird')
h1 = SEC['horn1']                                           # notes 0..13
t_rel = 0.55; p_first = np.array([0.0, LY['H'], 2.4 + R])
p_moon = p_first + np.array([0, 0, 0.5 * G * (TN[0] - t_rel) ** 2])
H.rest(0.0, t_rel, p_moon); vin = flight(H, t_rel, TN[0], p_moon, p_first)[1]
EV['moon'] = dict(p=p_moon.tolist(), t_open=t_rel - 0.25, t_rel=t_rel)
SFX.append(['latch', t_rel - 0.05])
p, vin = run(H, p_first, vin, h1[:5], lambda i: 1.4, dvx=(0, 1.5))       # A D F# G (F#) : last = the petal of the tulip


def tulip(ball, i_pet, p, vin, i_cup, t_launch, i_next, dx_next, dz_next, vy=0.0):
    """petal tap (note i_pet) -> drop into the tulip cup (note i_cup, e = 0), rest, spring stamen flicks it to note i_next."""
    tc = TN[i_cup]; cupp = p + np.array([0.18, 0, -0.06])
    p, vin = bounce_to(ball, i_pet, p, vin, tc, cupp, kind='petal')
    hit(ball, i_cup, p, vin, np.zeros(3), 'cup')
    ball.rest(tc, t_launch, p)
    tgt = p + np.array([dx_next, vy, dz_next])
    v0, vin = flight(ball, t_launch, TN[i_next], p, tgt)
    EV['cups'].append(dict(ball=ball.name, p=p.tolist(), t_in=float(tc), t_launch=float(t_launch), i=int(i_cup), v0=v0.tolist()))
    SFX.append(['spring', float(t_launch)])
    return tgt, vin, p


def cup_only(ball, i_cup, p, vin, t_launch, i_next, dx_next, dz_next, vy=0.0):
    """land in a tulip cup on note i_cup (sounds it), rest, spring flick to note i_next"""
    hit(ball, i_cup, p, vin, np.zeros(3), 'cup'); tc = TN[i_cup]
    ball.rest(tc, t_launch, p)
    tgt = p + np.array([dx_next, vy, dz_next])
    v0, vin = flight(ball, t_launch, TN[i_next], p, tgt)
    EV['cups'].append(dict(ball=ball.name, p=p.tolist(), t_in=float(tc), t_launch=float(t_launch), i=int(i_cup), v0=v0.tolist()))
    SFX.append(['spring', float(t_launch)])
    return tgt, vin, p


p, vin, cup1 = tulip(H, h1[4], p, vin, h1[5], TN[h1[6]] - 0.62, h1[6], 1.0, 0.5)
p, vin = run(H, p, vin, h1[6:], lambda i: 1.5)                            # A D F# | G F# E | A D
p_last_h1 = p.copy(); vin_last_h1 = vin.copy()

# H lands in the 2nd tulip (D4, end of bar 11); its stem pulls the string of watering can 1
CANS = []
def can_drop(name, t_tip, spout, i_first, lever=None):
    """watering can tips at t_tip (string pull or a drop landing in its lever cup); the drop hanging at the spout
    falls from rest onto the first leaf at note i_first."""
    d = ball(name, 'drop')
    leaf = np.array(spout, float) - np.array([0, 0, 0.5 * G * (TN[i_first] - t_tip - 0.08) ** 2])
    d.rest(0.0, t_tip + 0.08, spout)
    vin = flight(d, t_tip + 0.08, TN[i_first], spout, leaf)[1]
    EV['cans'].append(dict(drop=name, spout=list(map(float, spout)), t_tip=float(t_tip), lever=lever))
    return d, leaf, vin


h1_cup2_land = p
# the run from note 12 to 13 ended at p (contact of the tulip cup, D4)

# ----- dew1: three drops, chain reaction of watering cans (leaves hang from the vines on the upper level)
VXD = 1.7
dew = SEC['dew1']; d1 = [i for i in dew if i <= 22]; d2 = [i for i in dew if 23 <= i <= 31]; d3 = [i for i in dew if i >= 32]
spout1 = h1_cup2_land + np.array([0.9, LY['dew'] - LY['H'], 4.4])
D1, p, vin = can_drop('D1', TN[13] + 0.02, spout1, d1[0], lever=None)
EV['cans'][-1]['string_from'] = h1_cup2_land.tolist()
SFX.append(['tip', TN[13] + 0.02])
p, vin = run(D1, p, vin, d1, lambda i: VXD, lambda i: -0.03, dvx=(0, 1.8), kind_of=lambda i: 'leaf')
def into_lever(d, i_last, p, vin, dt, dx, dz):
    tgt = p + np.array([dx, 0, dz])
    pp, vv = bounce_to(d, i_last, p, vin, TN[i_last] + dt, tgt, kind='leaf')
    hit(d, i_last, pp, vv, np.zeros(3), 'lever', t=TN[i_last] + dt); d.rest(TN[i_last] + dt, 80, pp)
    SFX.append(['tick', TN[i_last] + dt])
    return pp
lev2 = into_lever(D1, d1[-1], p, vin, 0.2, 0.4, -0.1)
spout2 = lev2 + np.array([0.75, 0, 0.32])
D2, p, vin = can_drop('D2', TN[d1[-1]] + 0.17, spout2, d2[0], lever=lev2.tolist())
p, vin = run(D2, p, vin, d2, lambda i: VXD, lambda i: -0.03, dvx=(0, 1.8), kind_of=lambda i: 'leaf')
lev3 = into_lever(D2, d2[-1], p, vin, 0.2, 0.4, -0.1)
spout3 = lev3 + np.array([0.75, 0, 0.32])
D3, p, vin = can_drop('D3', TN[d2[-1]] + 0.17, spout3, d3[0], lever=lev3.tolist())
p, vin = run(D3, p, vin, d3, lambda i: VXD, lambda i: -0.035, dvx=(0, 1.8), kind_of=lambda i: 'leaf')
# last drop -> splashes into the lily pond
tp = TN[d3[-1]] + 0.42; pond = p + np.array([0.7, 0, -0.3])
pp, vv = bounce_to(D3, d3[-1], p, vin, tp, pond, kind='leaf')
D3.rest(tp, 80, pp + np.array([0, 0, -0.4]))
EV['ponds'].append(dict(p=pp.tolist(), t=float(tp))); SFX.append(['splash', float(tp)])

# ===================================================================== H: horn theme 2 + the cold F#-minor run
h2 = SEC['horn2']; cold = SEC['cold']
p, vin, cup2 = cup_only(H, 13, h1_cup2_land, vin_last_h1, TN[57] - 0.8, 57, 1.3, 1.0)
p, vin = run(H, p, vin, h2[:5], lambda i: 1.45)
p, vin, cup3 = tulip(H, h2[4], p, vin, h2[5], TN[h2[6]] - 0.85, h2[6], 1.3, 1.3)
p, vin = run(H, p, vin, h2[6:] + cold, lambda i: 1.5 if i < 71 else 1.75,
             kind_of=lambda i: 'bud' if i < 71 else 'ice')
# note 116 (E5) is played by the shelter tulip itself: H lands in it and waits out the storm
H.hits[-1]['kind'] = H.hits[-1]['kind']
cold_cup = p.copy(); vin_cold = vin.copy()
T_GUST = BT[28]                                              # bar 28: the F#-minor turn = the gust
EV['gust'] = dict(t=T_GUST)
# ===================================================================== gold seed balls (violin theme)
# The golden bell tree: the seed rack sits at the top of its trunk; G1/G3 run down the right branches, G2/G4 the left
# ones (G4 on the back branch); each branch ends on a gutter that rolls the ball into that side's counterweight bucket.
gold = SEC['gold']
GPH = [[i for i in gold if BT[37 + 4 * k] - 1e-6 <= TN[i] < BT[41 + 4 * k] - 1e-6] for k in range(4)]
X_TREE = cold_cup[0] + 2.6; Z_RACK = cold_cup[2] + 8.6
GDIR = [1, -1, 1, -1]; GLANE = [0.25, 0.25, 0.25, -0.4]; VXG = 2.2
G_END = []
for k, ph in enumerate(GPH):
    g = ball(f'G{k + 1}', 'seed')
    tf = 0.42; slot = np.array([X_TREE + 0.32 * GDIR[k] + (0.0 if k < 2 else 0.0), GLANE[k], Z_RACK]); t_gate = TN[ph[0]] - tf
    first = slot - np.array([0, 0, 0.5 * G * tf * tf])
    # the gate kicks the seed sideways out of the rack (visible flap); it then falls onto its first bell
    v0 = np.array([GDIR[k] * 0.9, 0, 0]); first = slot + v0 * tf + np.array([0, 0, -0.5 * G * tf * tf])
    if k == 2:      # G3 waits in the magazine behind G1's slot and rolls forward into it after G1 has left
        back = slot + np.array([0, 0.34, 0.03]); tr0 = EV['rack'][0]['t_gate'] + 0.3
        a = 5 / 7 * G * math.sin(math.atan2(0.03, 0.34)); trr = math.sqrt(2 * 0.3416 / a)
        u = (slot - back) / np.linalg.norm(slot - back)
        g.rest(0.0, tr0, back); g.seg.append(dict(k='acc', t0=float(tr0), t1=float(tr0 + trr), p=back.tolist(), v=[0, 0, 0], a=(a * u).tolist()))
        g.rest(tr0 + trr, t_gate, slot); SFX.append(['tick', float(tr0 + trr)])
    else: g.rest(0.0, t_gate, slot)
    g.fly(t_gate, TN[ph[0]], slot, v0); vin = v0 + np.array([0, 0, -G * tf])
    EV.setdefault('rack', []).append(dict(ball=g.name, slot=slot.tolist(), t_gate=float(t_gate), dir=GDIR[k]))
    SFX.append(['gate', float(t_gate)])
    p, vin = run(g, first, vin, ph, lambda i, d=GDIR[k]: d * VXG, lambda i: -0.05, dvx=(-1.4, 1.4), kind_of=lambda i: 'bell', tilt_max=40)
    G_END.append((p, vin, ph[-1]))
# gutters: each side slopes down and outwards, the bucket hangs past its lip
TH_CH = math.radians(8.0); ACH = 5 / 7 * G * math.sin(TH_CH)
GUT = {}
for side in (1, -1):
    ends = [(p, v, i) for (p, v, i), d in zip(G_END, GDIR) if d == side]
    x0 = X_TREE + side * 0.8
    x1 = (max if side > 0 else min)(p[0] for p, v, i in ends) + side * 1.3
    z0 = min(p[2] + abs(p[0] - x0) * math.tan(TH_CH) for p, v, i in ends) - 0.75
    GUT[side] = dict(x0=x0, x1=x1, z0=z0, u=[side * math.cos(TH_CH), 0, -math.sin(TH_CH)])
def ch_z(x, side): g_ = GUT[side]; return g_['z0'] - abs(x - g_['x0']) * math.tan(TH_CH)
HR = 2.1
BKS = {}
for side in (1, -1):
    g_ = GUT[side]; xb = g_['x1'] + side * 0.55; zh = ch_z(g_['x1'], side) - 0.75
    BKS[side] = dict(x=xb, z_high=zh, z_low=zh - HR, keys=[(0.0, zh - HR), (T_GUST, zh - HR), (T_GUST + 0.35, zh)])
SPOTS = [(-0.15, 0.15), (0.15, -0.15), (0.15, 0.15), (-0.15, -0.15)]
T_OPEN = T_GUST + 0.35
for k, (pe, ve, ie) in enumerate(G_END):
    g = B[f'G{k + 1}']; side = GDIR[k]; U = np.array(GUT[side]['u']); bk = BKS[side]
    t1, pl, vl = hop_onto(g, ie, pe, ve, lambda x, s_=side: ch_z(x, s_), 'bell')
    vt = (vl @ U) * U
    hit(g, ie, pl, vl, vt, 'gutter', t=t1)
    L_ = abs(GUT[side]['x1'] - pl[0]) / math.cos(TH_CH); v0 = vl @ U
    tr = (-v0 + math.sqrt(v0 * v0 + 2 * ACH * L_)) / ACH
    g.seg.append(dict(k='acc', t0=float(t1), t1=float(t1 + tr), p=pl.tolist(), v=vt.tolist(), a=(ACH * U).tolist()))
    plip = pl + vt * tr + 0.5 * ACH * U * tr * tr; vlip = vt + ACH * U * tr
    zb = bk['keys'][-1][1]; nin = sum(1 for j in range(k) if GDIR[j] == side)
    spot = np.array([bk['x'] + SPOTS[nin][0] * side, plip[1], zb + R + 0.04])
    dzf = plip[2] - spot[2]; vz0 = vlip[2]
    tfl = (vz0 + math.sqrt(vz0 * vz0 + 2 * G * dzf)) / G
    # horizontal: from the lip with the rolling speed; the bucket spot is where it comes down
    spot[0] = plip[0] + vlip[0] * tfl; spot[1] = plip[1]
    tl = t1 + tr + tfl
    g.fly(t1 + tr, tl, plip, vlip); v_in = vlip + np.array([0, 0, -G * tfl])
    hit(g, ie, spot, v_in, np.zeros(3), 'bucket', t=tl)
    SFX.append(['clunk', float(tl)])
    g.rest(tl, tl + 0.15, spot)
    tn0, tn1 = tl + 0.15, tl + 0.5
    bk['keys'] += [(tn0, zb), (tn1, zb - HR / 2)]
    SFX.append(['ratchet', float(tn0)])
    EV.setdefault('gold_in', []).append(dict(ball=g.name, t=float(tl), notch=[float(tn0), float(tn1)], side=side))
for k in range(4):
    g = B[f'G{k + 1}']; p0 = np.array(g.seg[-1]['p']); t = g.seg[-1]['t1']; BK = BKS[GDIR[k]]['keys']
    for (ta, za), (tb, zb_) in zip(BK, BK[1:]):
        if ta < t - 1e-9 or abs(zb_ - za) < 1e-9: continue
        if ta > t + 1e-9: g.rest(t, ta, p0); t = ta
        v = np.array([0, 0, (zb_ - za) / (tb - ta)]); g.lin(ta, tb, p0, v); p0 = p0 + v * (tb - ta); t = tb
    g.rest(t, 80.0, p0)
T_SHUT = max(BKS[1]['keys'][-1][0], BKS[-1]['keys'][-1][0])
EV['buckets'] = {str(s_): v for s_, v in BKS.items()}; EV['t_shut'] = T_SHUT; EV['hr'] = HR
EV['gutters'] = {str(s_): v for s_, v in GUT.items()}; EV['tree'] = dict(x=X_TREE, z_rack=Z_RACK)

# ===================================================================== H: climb (after the window is shut) + horn theme 3
climb = SEC['climb']; h3 = SEC['horn3']
p, vin, cupc = cup_only(H, 116, cold_cup, vin_cold, TN[162] - 1.0, 162, 1.5, 2.5)
p, vin = run(H, p, vin, climb + h3[:5], lambda i: 1.5, lambda i: -0.03)
p, vin, cup4 = tulip(H, h3[4], p, vin, h3[5], TN[h3[6]] - 0.85, h3[6], 1.3, 1.2)
p, vin = run(H, p, vin, h3[6:], lambda i: 1.5)
h3_cup = p.copy(); vin_h3 = vin.copy()

# ----- dew2: three more cans further along (same chain reaction)
dew2 = SEC['dew2']; e1 = [i for i in dew2 if i <= 199]; e2 = [i for i in dew2 if 200 <= i <= 208]; e3 = [i for i in dew2 if i >= 209]
spout4 = h3_cup + np.array([0.9, LY['dew'] - LY['H'], 4.4])
D4, p, vin = can_drop('D4', TN[190] + 0.02, spout4, e1[0])
EV['cans'][-1]['string_from'] = h3_cup.tolist(); SFX.append(['tip', TN[190] + 0.02])
p, vin = run(D4, p, vin, e1, lambda i: VXD, lambda i: -0.03, dvx=(0, 1.8), kind_of=lambda i: 'leaf')
lev5 = into_lever(D4, e1[-1], p, vin, 0.2, 0.4, -0.1)
D5, p, vin = can_drop('D5', TN[e1[-1]] + 0.17, lev5 + np.array([0.75, 0, 0.32]), e2[0], lever=lev5.tolist())
p, vin = run(D5, p, vin, e2, lambda i: VXD, lambda i: -0.03, dvx=(0, 1.8), kind_of=lambda i: 'leaf')
lev6 = into_lever(D5, e2[-1], p, vin, 0.2, 0.4, -0.1)
D6, p, vin = can_drop('D6', TN[e2[-1]] + 0.17, lev6 + np.array([0.75, 0, 0.32]), e3[0], lever=lev6.tolist())
p, vin = run(D6, p, vin, e3, lambda i: VXD, lambda i: -0.035, dvx=(0, 1.8), kind_of=lambda i: 'leaf')
tp = TN[e3[-1]] + 0.42; pond = p + np.array([0.7, 0, -0.3])
pp, vv = bounce_to(D6, e3[-1], p, vin, tp, pond, kind='leaf')
D6.rest(tp, 80, pp + np.array([0, 0, -0.4]))
EV['ponds'].append(dict(p=pp.tolist(), t=float(tp))); SFX.append(['splash', float(tp)])

# ===================================================================== coda: H, sunrise, giant sunflower
coda = SEC['coda']
p, vin, cup5 = cup_only(H, 190, h3_cup, vin_h3, TN[229] - 0.9, 229, 1.3, 1.9)
p, vin = run(H, p, vin, coda, lambda i: 1.3, lambda i: -0.02)
hit(H, coda[-1], p, vin, np.zeros(3), 'sunflower', t=TN[coda[-1]]) if False else None
H.hits[-1]['kind'] = H.hits[-1]['kind']
# the last chord: H lands in the heart of the giant sunflower and stays
H.hits.append(dict(i=int(coda[-1]), t=float(TN[coda[-1]]), p=p.tolist(), n=[0, 0, 1.0], e=0.0, kind='sunflower',
                   vin=vin.tolist(), vout=[0, 0, 0], ball='H'))
H.rest(TN[coda[-1]], 90.0, p)
EV['sunflower'] = dict(p=p.tolist(), t=float(TN[coda[-1]]))
EV['sunrise'] = dict(t0=float(BT[72]), t1=float(TN[coda[-1]]))


# ------------------------------------------------------------------ checks
def check():
    info = {}
    hs = [h for b in B.values() for h in b.hits]
    bb = [h for h in hs if np.linalg.norm(h['vout']) > 0 and h['kind'] not in ('gutter',)]
    info['e_max'] = round(max(h['e'] for h in bb), 3)
    info['speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in hs if np.linalg.norm(h['vout']) > 0), 3)
    info['tilt_max_deg'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for h in hs), 1)
    played = {}
    for h in hs:
        if h['kind'] in ('gutter', 'bucket', 'lever'): continue
        played.setdefault(h['i'], []).append(h)
    info['notes_uncovered'] = sorted(set(range(NN)) - set(played))
    info['notes_double'] = sorted(i for i, v in played.items() if len(v) > 1)
    for n, b in B.items():
        zz = [h['p'][2] for h in b.hits]; xx = [h['p'][0] for h in b.hits]
        info[f'{n}'] = f"x {min(xx):.1f}..{max(xx):.1f} z {min(zz):.2f}..{max(zz):.2f}"
    return info


if __name__ == '__main__':
    for k, v in check().items(): print(k, v)
    out = dict(G=G, R=R, lanes=LY, balls={n: dict(kind=b.kind, seg=b.seg, hits=b.hits) for n, b in B.items()}, ev=EV, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
