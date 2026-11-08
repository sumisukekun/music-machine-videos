"""New World Symphony IV - 'the ball train crosses the continent' (玉の大陸横断鉄道). Physics + story planner.
Run after arrange.py; writes layout.json (read by scene.py, render_audio.py, verify.py).

Cast: the LOCOMOTIVE (black ball, R 0.17) ROLLS on its rail (front lane): the rail's sleepers are xylophone keys, it
clacks over one on every melody note (so the rhythm is the spacing of the sleepers).  Four PASSENGER CARS (R 0.14)
BOUNCE behind it, each on its own row of bells: red = bass on beats 1 and 3, yellow/green/blue = the chord on beats
2 and 4 (red goes up while the others come down - the pistons of the train); in bars 26-33 all four take the cello line.
Story (bars of the score):
  1-7   departure: the loco rolls out of the engine shed down the 4-degree station ramp, faster and faster (the
        accelerating B-C motif); its last note is a lever that opens the loading gantry above the car rows
  8-9   the four cars drop out of the gantry behind it (free fall onto the 1st chord), three coupling jolts, big chord
  10-17 full steam across the prairie; bar 17 rattles over the iron river bridge
  18-25 mesa country, then the summit tunnel (22-25, dark: only the struck keys and the headlamp light it),
        bar 25 rattles out of the tunnel onto the girder bridge at the canyon rim
  26-33 the long wooden trestle over the canyon (quiet); from bar 30 it collapses behind the train and the wave catches
        up; on bar 33's long B the loco leaps off the last span, the cars bounce down after it
  34    everybody lands on the far rim on the tutti chord (the loco on a brass cymbal plate)
  34-43 downhill into the evening, crossing gates, city lights ahead (climbing sequence)
  44    arrival: the loco rolls into the station bell, the cars into their cups; clock tower, fireworks
Physics: one constant gravity G.  Rolling: a = 5/7 G (sin(slope) - CR) along the rail (CR = rolling resistance), so the
loco cruises at constant speed on the CR grade and speeds up only on the steeper station ramp.  Flight: pure
parabolas.  Bounces: the bar normal is chosen so the impulse is along it, restitution e <= EMAX (never faster out)."""
import json, os, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = 7.0
RL, RC = 0.17, 0.14
YL = -0.32
YC = [0.12, 0.42, 0.72, 1.02]                  # each car its own row of bells
EMAX, EMIN = 0.97, 0.2
CR = 0.028                                     # rolling resistance (grade of constant speed)
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json')); ACC = json.load(open(f'{HERE}/acc.json'))
Q = E['q']; BAR = E['bar']; T0 = E['t0']; FINAL = E['final']
def tb(b): return T0 + (b - 1) * BAR
TL = np.array([l[0] for l in LEAD]); NL = len(TL)
BARL = [l[4] for l in LEAD]
def li(b, k=0): return [i for i in range(NL) if BARL[i] == b][k]


class Ball:
    def __init__(s, name, R, y):
        s.name = name; s.R = R; s.y = y; s.seg = []; s.hits = []
    def rest(s, t0, t1, p): s.seg.append(dict(k='rest', t0=float(t0), t1=float(t1), p=list(map(float, p))))
    def fly(s, t0, t1, p0, v0): s.seg.append(dict(k='fly', t0=float(t0), t1=float(t1), p=list(map(float, p0)), v=list(map(float, v0))))
    def roll(s, t0, t1, p0, d, v0, a):
        s.seg.append(dict(k='roll', t0=float(t0), t1=float(t1), p=list(map(float, p0)), d=list(map(float, d)), v0=float(v0), a=float(a)))
    def at(s, t):
        for g in s.seg:
            if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9: return seg_at(g, t)
        if t < s.seg[0]['t0']: return seg_at(s.seg[0], s.seg[0]['t0'])
        return seg_at(s.seg[-1], s.seg[-1]['t1'])
    def vel(s, t):
        for g in s.seg:
            if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9: return seg_vel(g, t)
        return np.zeros(3)


def seg_at(g, t):
    p = np.array(g['p']); u = t - g['t0']
    if g['k'] == 'rest': return p
    if g['k'] == 'roll': return p + np.array(g['d']) * (g['v0'] * u + 0.5 * g['a'] * u * u)
    return p + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])


def seg_vel(g, t):
    u = t - g['t0']
    if g['k'] == 'rest': return np.zeros(3)
    if g['k'] == 'roll': return np.array(g['d']) * (g['v0'] + g['a'] * u)
    return np.array(g['v']) + np.array([0, 0, -G * u])


def vout_for(dx, dz, dt, dy=0.0): return np.array([dx / dt, dy / dt, dz / dt + 0.5 * G * dt])


def bounce_e(vin, vout):
    d = vout - vin; n = d / max(1e-9, np.linalg.norm(d)); a = vin @ n
    return (-(vout @ n) / a if a < -1e-9 else 9.0), n


def add_hit(ball, t, p, vin, vout, kind, pitch, n=None, e=None):
    if n is None: e, n = bounce_e(np.array(vin, float), np.array(vout, float))
    ball.hits.append(dict(t=float(t), p=list(map(float, p)), n=list(map(float, n)), e=None if e is None else float(e), kind=kind,
                          pitch=int(pitch), vin=list(map(float, vin)), vout=list(map(float, vout))))


def hops(ball, p, vin, times, pitches, vx_of, dz_of, kind_of, emax=EMAX, emin=EMIN, tilt_max=40.0, dvx=(-0.15, 0.1)):
    """bouncing ball at contact centre p at times[0] (incoming velocity vin) plays times[0..-2], arrives at times[-1]"""
    for k in range(len(times) - 1):
        ti, tj = times[k], times[k + 1]; dt = tj - ti
        dzt = dz_of(k, ti, p, dt)
        need = 0.0
        if k + 2 < len(times):
            dt2 = times[k + 2] - tj
            pj = p + np.array([vx_of(k, ti, p, dt) * dt, 0, dzt])
            need = (dz_of(k + 1, tj, pj, dt2) / dt2 + 0.5 * G * dt2) / (emax * 0.99)
        tx = vx_of(k, ti, p, dt); cx = vin[0]
        vx0 = cx + float(np.clip(tx - cx, dvx[0], dvx[1]))
        vxs = sorted(set(np.round(np.concatenate([[vx0], vx0 + np.linspace(-0.3, 0.1, 21)]), 4)), key=lambda v: abs(v - vx0))
        dzs = sorted(np.concatenate([dzt - np.linspace(0, 0.5 * G * dt * dt + 2.2, 400), dzt + np.linspace(0.003, 0.25, 40)]),
                     key=lambda d: (0 if d <= dzt else 1, abs(d - dzt)))
        best = None; cands = []
        for vx in vxs[:6]:
            for dz in dzs:
                vo = vout_for(vx * dt, dz, dt)
                e, n = bounce_e(vin, vo); arr = vo[2] - G * dt
                if not (emin <= e <= emax) or arr > -0.12: continue
                if math.degrees(math.acos(max(-1, min(1, n[2])))) > tilt_max: continue
                cands.append((dz, vo))
                if -arr >= need: best = (dz, vo); break
            if best is not None: break
        if best is None and cands: best = cands[0]; print('WARN need', ball.name, round(ti, 2), round(need, 2))
        if best is None:
            allc = []
            for vx in vxs:
                for dz in dzs[::3]:
                    vo = vout_for(vx * dt, dz, dt); e, n = bounce_e(vin, vo)
                    tilt = math.degrees(math.acos(max(-1, min(1, n[2]))))
                    allc.append((abs(min(0, e - emin)) + max(0, e - emax) + 0.01 * max(0, tilt - tilt_max), dz, vo))
            _, dz, vo = min(allc, key=lambda c: c[0]); best = (dz, vo); print('WARN e', ball.name, round(ti, 2))
        dz, vo = best
        add_hit(ball, ti, p, vin, vo, kind_of(k, ti), pitches[k])
        ball.fly(ti, tj, p, vo)
        p = p + vo * dt + np.array([0, 0, -0.5 * G * dt * dt]); vin = vo + np.array([0, 0, -G * dt])
    return p, vin


L = Ball('loco', RL, YL)
CARS = [Ball(n, RC, y) for n, y in zip(('red', 'yellow', 'green', 'blue'), YC)]
EV = {}; SFX = []

# ------------------------------------------------------------------ the locomotive's rail and motion
VCR = 2.9
A_RAMP = (VCR - 0.6) / (tb(8) - T0)              # 0.6 at the first note -> cruising speed at bar 8
SIN_RAMP = CR + A_RAMP / (5 / 7 * G)
T_LEAP = TL[li(33)]; T_LAND = TL[li(33, 1)]      # bar 33: B (dotted half) = the leap off the trestle, B (quarter) lands
def unit(sn):
    c = math.sqrt(1 - sn * sn); return np.array([c, 0, -sn])
def normal(sn): return np.array([sn, 0, math.sqrt(1 - sn * sn)])
d_ramp, d_cr = unit(SIN_RAMP), unit(CR)
a_ramp = 5 / 7 * G * (SIN_RAMP - CR)
v_start = 0.6 - a_ramp * T0
p = np.array([0.0, YL, 0.0 + RL])
L.roll(0.0, tb(8), p, d_ramp, v_start, a_ramp)
EV['shed'] = dict(p=p.tolist(), v=v_start, sin=SIN_RAMP)
p1 = seg_at(L.seg[-1], tb(8))
L.roll(tb(8), T_LEAP, p1, d_cr, VCR, 0.0)
p_lip = seg_at(L.seg[-1], T_LEAP)
KICK = math.radians(30)                                     # the last span ends in an up-turned lip (ski jump)
v_lip = VCR * np.array([math.cos(KICK), 0, math.sin(KICK)])
L.fly(T_LEAP, T_LAND, p_lip, v_lip)
dtl = T_LAND - T_LEAP
p_land = p_lip + v_lip * dtl + np.array([0, 0, -0.5 * G * dtl * dtl]); v_air = v_lip + np.array([0, 0, -G * dtl])
v_after = np.dot(v_air, d_cr) * d_cr                        # the landing key soaks up the normal part (e = 0)
L.roll(T_LAND, FINAL, p_land, d_cr, float(np.linalg.norm(v_after)), 0.0)
p_end = seg_at(L.seg[-1], FINAL)
L.rest(FINAL, 200.0, p_end)
EV['leap'] = dict(t0=float(T_LEAP), t1=float(T_LAND), lip=p_lip.tolist(), land=p_land.tolist(), drop=float(p_lip[2] - p_land[2]))
I_LEV = li(7, 3)
def kind_lead(i):
    b = BARL[i]
    if i == I_LEV: return 'lever'
    if i == NL - 1: return 'bell'
    if b == 17: return 'truss'
    if b == 25: return 'girder'
    if i == li(33): return 'lip'
    if i == li(33, 1): return 'land'
    if 26 <= b <= 33: return 'trestle'
    if i == li(34): return 'cymbal'
    if 22 <= b <= 24: return 'tunnel'
    return 'sleeper'
for i in range(NL):
    t = TL[i]; pp = L.at(t)
    vv = L.vel(t) if i != NL - 1 else seg_vel(L.seg[-2], t)
    n = normal(SIN_RAMP) if t < tb(8) else normal(CR)
    vo = vv; e = None
    if i == li(33, 1): vv = v_air; vo = v_after; e = 0.0
    if i == NL - 1: n = np.array([-1.0, 0, 0]); vo = np.zeros(3); e = 0.0   # the bell hangs at the end of the line
    add_hit(L, t, pp, vv, vo, kind_lead(i), LEAD[i][1], n=n, e=e)
_lx = np.array([h['p'][0] for h in L.hits]); _lz = np.array([h['p'][2] for h in L.hits]) - RL
def rail_z(x):
    """rail-top height under the loco at x"""
    return float(np.interp(x, _lx, _lz))
def loco_x(t): return float(L.at(t)[0])
EV['lever'] = dict(t=float(TL[I_LEV]), p=L.hits[I_LEV]['p'])

# ------------------------------------------------------------------ cars
OFFS = [0.62, 1.14, 1.66, 2.18]
X_C26 = loco_x(tb(26)) - 1.4; X_RIM = float(p_lip[0]) + 1.9; Z_RIM = float(p_land[2]) - RL
def car_rail(x):
    """the cars' deck: follows the loco's rail, except on the trestle where the lower car deck slopes down to the far rim"""
    if x < X_C26: return rail_z(x)
    if x < X_RIM: return rail_z(X_C26) + (Z_RIM - rail_z(X_C26)) * (x - X_C26) / (X_RIM - X_C26)
    if x < float(p_land[0]): return Z_RIM - CR * (x - X_RIM) * 0
    return rail_z(x)
EV['rim'] = dict(x=X_RIM, z=Z_RIM)
def car_hits(c): return [a for a in ACC if str(c) in a['p']]
def car_kind(c, a):
    b = a['bar']
    if a['kind'] == 'final': return 'cup'
    if a['kind'] == 'cello': return 'trestle'
    if b == 34 and a['kind'] == 'tutti': return 'rim'
    if 22 <= b <= 24: return 'tunnel'
    return 'drum' if c == 0 else 'bell'
T_REL = float(TL[I_LEV])
for c, b in enumerate(CARS):
    hs = car_hits(c); tt = [a['t'] for a in hs]; pitches = [a['p'][str(c)] for a in hs]
    # gantry: rest above the first landing, the lever opens the floor -> free fall
    t1 = tt[0]; tf = t1 - T_REL; xl = loco_x(t1) - OFFS[c]
    pl = np.array([xl, b.y, rail_z(xl) + RC]); ps = pl + np.array([0, 0, 0.5 * G * tf * tf])
    b.rest(0.0, T_REL, ps); b.fly(T_REL, t1, ps, [0, 0, 0])
    if c == 0: EV['gantry'] = dict(t=T_REL, fall=0.5 * G * tf * tf, z=float(ps[2]))
    EV['gantry']['x%d' % c] = float(xl)
    p, vin = pl, np.array([0, 0, -G * tf])
    def vx_of(k, t, p, dt, c=c, tt=tt):
        return (loco_x(tt[k + 1]) - OFFS[c] - p[0]) / dt
    def dz_of(k, t, p, dt, c=c, tt=tt):
        tj = tt[min(k + 1, len(tt) - 1)]; xj = loco_x(tj) - OFFS[c]
        return car_rail(xj) + RC - p[2]
    seg_bounds = sorted(set([0] + [k for k in range(1, len(tt)) if min(abs(tt[k] - tb(x)) for x in (10, 26, 34)) < 1e-4] + [len(tt) - 1]))
    for s0, s1 in zip(seg_bounds, seg_bounds[1:]):
        t_s = tt[s0]
        if s0 == 0: dv, em = (-0.4, 3.2), EMIN
        elif t_s < tb(10) - 1e-4: dv, em = (-0.4, 0.6), EMIN
        elif tb(26) - 1e-4 <= t_s < tb(34) - 1e-4: dv, em = (-0.6, 0.6), 0.12
        else: dv, em = (-0.3, 0.3), EMIN
        sub = list(range(s0, s1 + 1))
        p, vin = hops(b, p, vin, [tt[k] for k in sub], [pitches[k] for k in sub],
                      lambda k, t, p, dt, sub=sub: vx_of(sub[k], t, p, dt), lambda k, t, p, dt, sub=sub: dz_of(sub[k], t, p, dt),
                      lambda k, t, sub=sub, c=c, hs=hs: car_kind(c, hs[sub[k]]), dvx=dv, emin=em)
    add_hit(b, tt[-1], p, vin, np.zeros(3), 'cup', pitches[-1]); b.rest(tt[-1], 200.0, p)
SFX.append(['gate', T_REL + 0.01])

# ------------------------------------------------------------------ trestle collapse: a wave from bar 30 on
tr = [(b, h) for b in [L] + CARS for h in b.hits if h['kind'] in ('trestle', 'lip')]
X_TR0 = min(h['p'][0] for _, h in tr) - 0.9; X_TR1 = max(h['p'][0] for _, h in tr) + 0.25
T_W0 = tb(30)
lh = max((h for b in CARS for h in b.hits if h['kind'] == 'trestle'), key=lambda h: (h['t'], -h['p'][0]))
lb = [b.name for b in CARS if lh in b.hits][0]
VW = (lh['p'][0] - X_TR0) / (lh['t'] + 0.06 - T_W0)
def wave_t(x): return T_W0 + (x - X_TR0) / VW
for b, h in tr: h['t_fall'] = float(max(h['t'] + 0.06, wave_t(h['p'][0])))
EV['trestle'] = dict(x0=X_TR0, x1=X_TR1, t_w0=T_W0, vw=VW, last=lb, z=float(np.mean([h['p'][2] for _, h in tr])))
EV['rim_t'] = tb(34)


def check():
    info = {}
    hs = [(b.name, h) for b in CARS for h in b.hits if h['kind'] != 'cup']
    info['car_hits'] = len(hs); info['loco_notes'] = (len(L.hits), NL)
    info['e_range'] = (round(min(h['e'] for _, h in hs), 3), round(max(h['e'] for _, h in hs), 3))
    sp = [np.linalg.norm(h['vout']) / max(1e-6, np.linalg.norm(h['vin'])) for _, h in hs]
    info['bounce_speed_out_over_in_max'] = round(max(sp), 3)
    info['tilt_max'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for _, h in hs), 1)
    for b in [L] + CARS:
        zz = [h['p'][2] for h in b.hits]; info['z_' + b.name] = (round(min(zz), 2), round(max(zz), 2))
    info['x_end'] = round(L.at(FINAL)[0], 1); info['leap_drop'] = round(EV['leap']['drop'], 2)
    dev = []; dz = []
    for t in np.arange(12.5, FINAL, 0.1):
        for c, b in enumerate(CARS):
            q = b.at(t); dev.append(loco_x(t) - q[0] - OFFS[c]); dz.append(q[2] - RC - rail_z(q[0]))
    info['formation_dx'] = (round(min(dev), 2), round(max(dev), 2)); info['car_height_over_rail'] = (round(min(dz), 2), round(max(dz), 2))
    info['loco_speed'] = [round(float(np.linalg.norm(L.vel(t))), 2) for t in (0.5, 4, 8, 11.5, 30, 60)]
    return info


if __name__ == '__main__':
    for k, v in check().items(): print(k, v)
    out = dict(G=G, RL=RL, RC=RC, YL=YL, YC=YC, CR=CR, balls={b.name: dict(R=b.R, y=b.y, seg=b.seg, hits=b.hits) for b in [L] + CARS},
               ev=EV, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
