"""Beethoven 5 - 「運命の扉」 (The Door of Fate). Physics + story planner. Run after arrange.py; writes layout.json.

Side view of a cut-away stone castle (x to the right, z up, the camera looks along +y). The great door stands at x = 0..0.5:
outside (x < 0) a stormy night, inside the sleeping castle machine.

Story (every melody note = a ball striking something that sounds it):
  bars 1-5    a visitor ball rolls down the castle road, hops on 3 iron steps and knocks on the door (G G G Eb).
              Inside, a ball on a shelf is shaken loose and knocks back from the inside (F F F D).
  bars 6-13   the knock pulls a cord -> a ball rolls out of a chute; from then on every ball's first hit pulls the cord
              of the next hatch: 6 sleeping balls drop one after another, each plays its entry and lands in a bowl bell,
              lighting a window - the castle wakes up floor by floor (strings in imitation).
  bars 14-21  two balls answer each other across the top floor; the last one strikes the great gong (fermata):
              the camera pulls back - the castle is huge.
  bars 22-24  the gong ball falls into a basket on a cord -> the inside ball is released and knocks on the door again.
  bars 25-32  the awakening runs down the second tower: 8 hatches, 8 balls, 8 windows.
  bars 33-58  one ball zig-zags down the stair-well tower (rising sequence, then the fortissimo cascades), lands in the
              bell-rope bowl:
  bars 59-62  the bell rope opens the bell tower: horn call on bronze bells, dawn begins (E-flat major).
  bars 63-94  second theme: four balls walk home along the top gallery, the last one falls down the well (swell).
  bars 95-109 twin balls mirror each other down the door tower (the violins in unison).
  bars 110-122 the final balls hammer the door lever three times (ダダダ・ダン x3): the counterweight drops a notch each
              time, the door rises - light floods out and the castle's balls pour out.
Physics: one constant gravity G; free flight = parabolas; a bounce keeps the tangential velocity, the normal part is
reversed with restitution e <= EMAX (never gains speed). Rolling on ramps: a = 5/7 g sin(slope). Balls only gain
energy by falling. Bowls / baskets / lever catch the ball (it stays)."""
import json, os, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = 7.0; R = 0.15
EMAX = 0.95; EMIN = 0.12
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
TN = np.array([l[0] for l in LEAD]); NN = len(TN)
BAR = E['bar']


class Ball:
    def __init__(s, name, kind='plain', y=0.0):
        s.name = name; s.kind = kind; s.y = y; s.seg = []; s.hits = []
    def rest(s, t0, t1, p): s.seg.append(dict(k='rest', t0=float(t0), t1=float(t1), p=list(map(float, p))))
    def fly(s, t0, t1, p0, v0): s.seg.append(dict(k='fly', t0=float(t0), t1=float(t1), p=list(map(float, p0)), v=list(map(float, v0))))
    def acc(s, t0, t1, p0, v0, a): s.seg.append(dict(k='acc', t0=float(t0), t1=float(t1), p=list(map(float, p0)), v=list(map(float, v0)), a=list(map(float, a))))
    def at(s, t):
        for g in s.seg:
            if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9: return seg_at(g, t)
        if t < s.seg[0]['t0']: return seg_at(s.seg[0], s.seg[0]['t0'])
        return seg_at(s.seg[-1], s.seg[-1]['t1'])
    def end(s): return s.seg[-1]['t1']


def seg_at(g, t):
    p = np.array(g['p'])
    if g['k'] == 'rest': return p
    u = t - g['t0']; v = np.array(g['v'])
    a = np.array(g['a']) if g['k'] == 'acc' else np.array([0, 0, -G])
    return p + v * u + 0.5 * a * u * u


def vout_for(dx, dz, dt, dy=0.0): return np.array([dx / dt, dy / dt, dz / dt + 0.5 * G * dt])
def land(p, v, dt): return p + v * dt + np.array([0, 0, -0.5 * G * dt * dt]), v + np.array([0, 0, -G * dt])


def bounce_e(vin, vout):
    d = vout - vin; n = d / max(1e-9, np.linalg.norm(d))
    a = vin @ n
    return (-(vout @ n) / a if a < -1e-9 else 9.0), n


def tilt_of(n): return math.degrees(math.acos(max(-1, min(1, n[2]))))


def add_hit(ball, i, p, vin, vout, kind, t=None, n=None):
    e, n_ = bounce_e(np.array(vin, float), np.array(vout, float))
    if n is None: n = n_
    ball.hits.append(dict(i=int(i), t=float(TN[i] if t is None else t), p=list(map(float, p)), n=list(map(float, n)), e=float(e),
                          kind=kind, vin=list(map(float, vin)), vout=list(map(float, vout))))


def run(ball, p, vin, notes, vx_of, dz_of=lambda i: -0.02, kind_of=lambda i: 'key', rev=lambda i: False, emax=EMAX, emin=EMIN,
        tilt_max=38.0, dvx=(-0.25, 0.15), dz_lo=1.2):
    """ball is at contact point p (centre) at note notes[0] with incoming velocity vin; hops through notes[1:].
    vx_of(i): target horizontal speed of the hop leaving note i (sign = direction); rev(i): the hit at note i reverses
    the direction (a steep 'wall' key). dz_of(i): target height change of that hop. The landing is lowered (never raised
    much) until the bounce is physical. Returns (contact point, incoming velocity) at the last note (no hit recorded)."""
    p = np.array(p, float); vin = np.array(vin, float)
    for k in range(len(notes) - 1):
        i = notes[k]; j = notes[k + 1]; dt = TN[j] - TN[i]
        dzt = dz_of(i); tx = vx_of(i); cx = vin[0]; rv = rev(i)
        if rv: vx0 = tx; tm = 82.0
        elif abs(cx) < 0.3: vx0 = tx; tm = tilt_max                        # falling straight down: the first key sets the direction
        elif np.sign(tx) == np.sign(cx): vx0 = cx + float(np.clip(tx - cx, dvx[0], dvx[1])); tm = tilt_max
        else: vx0 = tx; tm = tilt_max
        if k + 1 < len(notes) - 1 and not rev(j):
            dt2 = TN[notes[k + 2]] - TN[j]
            need = (dz_of(j) / dt2 + 0.5 * G * dt2) / (emax * 0.995)
        else: need = 0.0
        vxs = sorted(set(np.round(np.concatenate([[vx0], vx0 + np.sign(vx0 or 1) * np.linspace(-0.4, 0.15, 23)]), 4)), key=lambda v: abs(v - vx0))
        dzs = sorted(np.concatenate([dzt - np.linspace(0, dz_lo, 400), dzt + np.linspace(0.003, 0.12, 20)]),
                     key=lambda d: (0 if d <= dzt else 1, abs(d - dzt)))
        best = None; cands = []
        for vx in vxs[:8]:
            for dz in dzs:
                vo = vout_for(vx * dt, dz, dt)
                e, n = bounce_e(vin, vo)
                if not (emin <= e <= emax) or vo[2] - G * dt > -0.15: continue
                if tilt_of(n) > tm: continue
                cands.append((dz, vo, e, n))
                if -(vo[2] - G * dt) >= need: best = (dz, vo, e, n); break
            if best is not None: break
        if best is None and cands: best = cands[0]; print('WARN need', ball.name, i, round(need, 2))
        if best is None:
            allc = []
            for vx in vxs:
                for dz in dzs[::4]:
                    vo = vout_for(vx * dt, dz, dt); e, n = bounce_e(vin, vo)
                    pen = abs(min(0, e - emin)) + max(0, e - emax) + max(0, tilt_of(n) - tm) / 30
                    allc.append((pen, dz, vo, e, n))
            pen, dz, vo, e, n = min(allc, key=lambda c: c[0])
            best = (dz, vo, e, n); print('WARN e', ball.name, i, round(e, 3), round(tilt_of(n), 1), 'pen', round(pen, 3))
        dz, vo, e, n = best
        ball.hits.append(dict(i=int(i), t=float(TN[i]), p=list(map(float, p)), n=list(map(float, n)), e=float(e), kind=kind_of(i),
                              vin=list(map(float, vin)), vout=list(map(float, vo))))
        ball.fly(TN[i], TN[j], p, vo)
        p, vin = land(p, vo, dt)
    return p, vin


def to_bowl(ball, i, p, vin, kind='bowl'):
    """the ball lands in a bowl bell at note i and stays"""
    add_hit(ball, i, p, vin, [0, 0, 0], kind, n=[0, 0, 1])
    ball.rest(TN[i], 200.0, p)


def drop(ball, p_hit, t_open, t_hit):
    """ball waits in a hatch straight above p_hit; the hatch opens at t_open, it falls freely and hits at t_hit"""
    tf = t_hit - t_open; h = 0.5 * G * tf * tf
    ph = np.array(p_hit, float) + np.array([0, 0, h])
    ball.rest(0.0, t_open, ph); ball.fly(t_open, t_hit, ph, [0, 0, 0])
    return ph, np.array([0, 0, -G * tf])


def ramp_roll(ball, t_rel, t_hit, p_hit, slope_deg, direction, t_fly=0.16):
    """ball waits at the top of a ramp, released at t_rel, rolls down (a = 5/7 g sin), flies t_fly, hits p_hit at t_hit.
    direction +1 / -1 along x. returns ramp geometry and incoming velocity at the hit."""
    th = math.radians(slope_deg); a = 5 / 7 * G * math.sin(th)
    tr = t_hit - t_fly - t_rel; vend = a * tr; L = 0.5 * a * tr * tr
    u = np.array([direction * math.cos(th), 0, -math.sin(th)])
    v_end = vend * u
    # flight back-computed from the hit point
    p_end = np.array(p_hit, float) - v_end * t_fly - np.array([0, 0, -0.5 * G * t_fly * t_fly])
    p_top = p_end - u * L
    ball.rest(0.0, t_rel, p_top); ball.acc(t_rel, t_hit - t_fly, p_top, [0, 0, 0], a * u)
    ball.fly(t_hit - t_fly, t_hit, p_end, v_end)
    return dict(top=p_top.tolist(), end=p_end.tolist(), slope=slope_deg, a=a, u=u.tolist()), v_end + np.array([0, 0, -G * t_fly])


def fly_until_z(p, v, z):
    """time until the centre comes down to height z"""
    a = -0.5 * G; b = v[2]; c = p[2] - z
    disc = b * b - 4 * a * c
    return (-b - math.sqrt(disc)) / (2 * a)


def hop_to_x(ball, i, j, xp, kind, dz_hi=0.3, dz_lo=-1.2):
    """replace the planned hop i->j by one that arrives at x = xp (a plate / gong face) at note j"""
    h = ball.hits.pop(); ball.seg.pop(); assert h['i'] == i
    p2 = np.array(h['p']); vin2 = np.array(h['vin']); dt = TN[j] - TN[i]
    best = None; bb = None
    for dz in np.linspace(dz_hi, dz_lo, 600):
        vo = vout_for(xp - p2[0], dz, dt); e, n = bounce_e(vin2, vo)
        sc = abs(min(0, e - EMIN)) + max(0, e - EMAX) + max(0, tilt_of(n) - 38) / 30
        if bb is None or sc < bb[0]: bb = (sc, vo, e, tilt_of(n), dz)
        if sc == 0: best = vo; break
    if best is None: print('WARN hop_to_x', ball.name, i, 'e', round(bb[2], 3), 'tilt', round(bb[3], 1)); best = bb[1]
    add_hit(ball, i, p2, vin2, best, kind); ball.fly(TN[i], TN[j], p2, best)
    return land(p2, best, dt)



B = {}
EV = dict(hatches=[], ramps=[], bowls=[], baskets=[], cords=[], plates=[], windows=[], sections={}, rows={})
SFX = []
def ball(name, kind='plain', y=0.0):
    assert name not in B, name
    B[name] = Ball(name, kind, y); return B[name]


def shift_ball(b, dx):
    for g in b.seg: g['p'][0] += dx
    for h in b.hits: h['p'][0] += dx


def hitc(name, i):
    """contact point (key surface) of ball name at note i"""
    for h in B[name].hits:
        if h['i'] == i: return np.array(h['p']) - R * np.array(h['n'])
    raise KeyError((name, i))


def trigger(t_first, prefer, lo=0.45, hi=0.75):
    """a visible cause for a hatch opening: the latest hit of the preferred balls 0.45-0.75 s before t_first"""
    for nm in prefer:
        c = [h for h in B[nm].hits if t_first - hi <= h['t'] <= t_first - lo]
        if c: h = c[-1]; return h['t'], np.array(h['p']) - R * np.array(h['n'])
    c = [(h['t'], n, h) for n, b in B.items() for h in b.hits if t_first - hi <= h['t'] <= t_first - lo]
    if c:
        t, n, h = max(c, key=lambda x: x[0]); return t, np.array(h['p']) - R * np.array(h['n'])
    return t_first - 0.6, None


def entry(name, notes, p1, start, vx, dz=-0.03, end='bowl', kind='sleeper', y=0.0, xp=None, kind_of=None, dvx=(-3, 3),
          tilt_max=38.0, last_dz=None, last_tilt=38.0, end_kind='bowl', window=True, row=None):
    """one ball: starts (hatch drop / ramp roll / continuing state), hops on keys through notes, ends in a bowl bell
    (stays), on a plate / gong face at x = xp (knock), or keeps going (end=None -> returns state)."""
    bl = B[name] if name in B else ball(name, kind, y)
    p1 = np.array(p1, float); p1[1] = bl.y
    if start[0] == 'drop':
        t_open, cause = start[1], start[2]; ph, vin = drop(bl, p1, t_open, TN[notes[0]])
    elif start[0] == 'ramp':
        t_rel, slope, dr, cause = start[1:]; rinfo, vin = ramp_roll(bl, t_rel, TN[notes[0]], p1, slope, dr, t_fly=0.2)
    else:
        vin = np.array(start[1], float); cause = None
    vxf = vx if callable(vx) else (lambda i: vx)
    dzf = dz if callable(dz) else (lambda i: dz)
    if last_dz is not None:
        dz0 = dzf; dzf = lambda i: last_dz if i == notes[-2] else dz0(i)
    kof = kind_of or (lambda i: 'key')
    p, vin = run(bl, p1, vin, notes, vxf, dzf, kind_of=kof, dvx=dvx,
                 tilt_max=tilt_max if last_dz is None else 38.0)
    shift = 0.0
    if end in ('plate', 'gong'):
        if start[0] in ('drop', 'ramp'):
            shift = xp - p[0]; shift_ball(bl, shift); p = p + [shift, 0, 0]; p1 = p1 + [shift, 0, 0]
            if start[0] == 'drop': ph = ph + [shift, 0, 0]
            else:
                for k_ in ('top', 'end'): rinfo[k_][0] += shift
        pk, vk = hop_to_x(bl, notes[-2], notes[-1], xp, kof(notes[-2]))
        p = pk; vin = vk
    if start[0] == 'drop':
        EV['hatches'].append(dict(ball=name, p=ph.tolist(), t=float(start[1]))); SFX.append(['hatch', float(start[1])])
        if cause is not None: EV['cords'].append(dict(src=list(map(float, cause)), dst=(ph + [0, 0, R + 0.1]).tolist(), t=float(start[1])))
    elif start[0] == 'ramp':
        EV['ramps'].append(dict(ball=name, **rinfo)); SFX.append(['latch', float(start[1])])
        if cause is not None: EV['cords'].append(dict(src=list(map(float, cause)), dst=list(map(float, np.array(rinfo['top']) + [0, 0, R + 0.1])), t=float(start[1])))
    if row is not None: EV['rows'].setdefault(row, []).append(name)
    if end == 'bowl':
        to_bowl(bl, notes[-1], p, vin, kind=end_kind)
        EV['bowls'].append(dict(ball=name, p=p.tolist(), t=float(TN[notes[-1]]), i=int(notes[-1]), kind=end_kind))
        if window: EV['windows'].append(dict(p=p.tolist(), t=float(TN[notes[-1]])))
        return p, None
    if end in ('plate', 'gong'):
        n = [-1, 0, 0] if vin[0] > 0 else [1, 0, 0]
        e_n = 0.12 if end == 'plate' else 0.35
        vb = np.array([-e_n * vin[0], 0, vin[2]])
        add_hit(bl, notes[-1], p, vin, vb, end, n=n)
        rec = dict(p=(p - R * np.array(n)).tolist(), n=n, t=float(TN[notes[-1]]), ball=name)
        (EV['plates'] if end == 'plate' else EV.setdefault('gongs', [])).append(rec)
        return p, vb
    return p, vin


def exit_to_basket(name, i, p, vin, e=0.5, drop_=0.35, keep=0.9):
    """last key: a soft bounce (normal = up, restitution e, a little friction) and the ball drops into a basket"""
    vo = np.array([vin[0] * keep, vin[1] * keep, -e * vin[2]])
    add_hit(B[name], i, p, vin, vo, 'key', n=[0, 0, 1]); B[name].hits[-1]['e'] = float(e); B[name].hits[-1]['friction'] = True
    return fall_to(name, TN[i], p, vo, p[2] - drop_)


def fall_to(name, t0, p, v, z, kind='basket'):
    bl = B[name]; tl = fly_until_z(p, v, z)
    bl.fly(t0, t0 + tl, p, v); q = land(p, v, tl)[0]; bl.rest(t0 + tl, 200.0, q)
    EV['baskets'].append(dict(p=q.tolist(), t=float(t0 + tl), ball=name, kind=kind)); return q, t0 + tl


DOOR_X0, DOOR_X1, DOOR_H = 0.0, 0.5, 3.4
PLATE = 0.05
XO = DOOR_X0 - PLATE - R          # ball centre touching the outer knocker plate
XI = DOOR_X1 + PLATE + R          # ... the inner one

# ================================================================== S1: the door (bars 1-5)
ball('K', 'visitor')
p, vb = entry('K', [0, 1, 2, 3], [-1.2, 0, 0.8], ('ramp', -1.1, 19.0, +1, None), 3.2, end='plate', xp=XO, dvx=(-0.2, 0.25),
              kind_of=lambda i: 'step')
q, tq = fall_to('K', TN[3], p, vb, R + 0.02, kind='puddle'); SFX.append(['door', TN[3]]); SFX.append(['splash', tq])
EV['ramps'][-1]['road'] = True
ball('I', 'inner')
p, vb = entry('I', [4, 5, 6, 7], [1.5, 0, 1.0], ('ramp', TN[3] + 0.04, 27.0, -1, None), lambda i: -2.3, end='plate', xp=XI,
              dvx=(-0.12, 0.25))
EV['ramps'][-1]['shelf'] = True; EV['ramps'][-1]['t_shake'] = float(TN[3])
q, tq = fall_to('I', TN[7], p, vb, 0.42, kind='cradle'); SFX.append(['door', TN[7]])
EV['sections']['door'] = dict(t0=0.0, t1=TN[7] + 0.5)

# ================================================================== S2: the castle wakes up (bars 6-13), column B
ENT2 = [(8, 11, -1), (12, 15, +1), (16, 19, -1), (20, 23, +1), (24, 27, -1), (28, 31, +1)]
X2L, X2R = 8.6, 11.9
Z2 = [1.55, 2.5, 3.45, 4.4, 5.35, 6.3]
prev = None
for k, (a, b, dr) in enumerate(ENT2):
    nm = f'E{k + 1}'
    p1 = [X2L if dr > 0 else X2R, 0, Z2[k]]
    if k == 0: st = ('ramp', TN[7] + 0.04, 15.0, -1, np.array(EV['plates'][-1]['p']))
    else:
        t_o, cause = trigger(TN[a], [prev]); st = ('drop', t_o, cause)
    if k < 5: entry(nm, list(range(a, b + 1)), p1, st, 2.7 * dr, dvx=(-0.2, 0.2) if k == 0 else (-3, 3), y=-0.75 if k == 4 else 0.0)
    else: A_state = entry(nm, list(range(a, b + 1)), p1, st, 2.7 * dr, end=None)
    prev = nm
EV['sections']['wake'] = dict(t0=TN[8] - 0.6, t1=TN[31])

# ================================================================== S3: call and response on the top floor (bars 14-21) + gong
A_NOTES = [31, 32, 33, 34, 35, 39, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 53]
pA, vA = A_state
GONG_X = None
p, vin = run(B['E6'], pA, vA, A_NOTES, lambda i: 1.55, lambda i: -0.02, dvx=(-3.5, 3.5))
GONG_X = float(p[0])
hop_to_x(B['E6'], 52, 53, GONG_X, 'key')            # (re-solved exactly, same x)
pg, vg = land(np.array(B['E6'].hits[-1]['p']), np.array(B['E6'].hits[-1]['vout']), TN[53] - TN[52])
vb = np.array([-0.35 * vg[0], 0, vg[2]])
add_hit(B['E6'], 53, pg, vg, vb, 'gong', n=[-1, 0, 0])
EV['gong'] = dict(p=(pg + [R, 0, 0]).tolist(), t=float(TN[53]))
T_CRADLE = TN[54] - 0.53
pbk, _ = land(pg, vb, T_CRADLE - TN[53])
B['E6'].fly(TN[53], T_CRADLE, pg, vb); B['E6'].rest(T_CRADLE, 200.0, pbk)
EV['baskets'].append(dict(p=pbk.tolist(), t=float(T_CRADLE), ball='E6', kind='weight')); SFX.append(['basket', T_CRADLE])
# B: the answering ball on the back lane, travelling the other way
ball('B', 'sleeper', y=0.75)
xa = float(B['E6'].at(TN[36])[0]); za = float(B['E6'].at(TN[36])[2])
t_o, cause = trigger(TN[36], ['E6'])
p, vin = entry('B', [36, 37, 38, 44, 45, 46], [xa + 1.9, 0.75, za + 0.2], ('drop', t_o, cause), -2.2, end=None)
exit_to_basket('B', 46, p, vin)
EV['sections']['top'] = dict(t0=TN[32], t1=TN[53])

# ================================================================== S4: the second knock from inside (bars 22-24)
ball('J', 'inner')
p, vb = entry('J', [54, 55, 56, 57], [1.4, 0, 1.75], ('drop', T_CRADLE, pbk - [0, 0, R]), -2.3, end='plate', xp=XI)
q, tq = fall_to('J', TN[57], p, vb, p[2] - 0.6); SFX.append(['door', TN[57]])
EV['sections']['knock2'] = dict(t0=TN[54] - 0.8, t1=TN[57] + 1.0)

# ================================================================== S5: down the second tower (bars 25-32), column C
ENT5 = [(58, 61), (62, 65), (66, 69), (70, 73), (74, 77), (78, 81), (82, 85), (86, 88)]
X5L = GONG_X + 1.1; X5R = X5L + 3.25
prev = None
for k, (a, b) in enumerate(ENT5):
    nm = f'F{k + 1}'; dr = +1 if k % 2 == 0 else -1
    p1 = [X5L if dr > 0 else X5R, 0, 6.6 - 0.72 * k]
    if k == 0: st = ('ramp', TN[a] - 1.7, 14.0, +1, np.array(EV['plates'][-1]['p']))
    else:
        t_o, cause = trigger(TN[a], [prev]); st = ('drop', t_o, cause)
    if b != 88: entry(nm, list(range(a, b + 1)), p1, st, 2.6 * dr)
    else:
        p, vin = entry(nm, list(range(a, b + 1)), p1, st, 2.6 * dr, end=None)
        exit_to_basket(nm, 88, p, vin)
    prev = nm
EV['sections']['down'] = dict(t0=TN[58] - 0.6, t1=TN[88])

# ================================================================== S6: relay down the stair-well (bars 33-58), column D
ROWS6 = [(89, 100), (101, 112), (113, 124), (125, 132), (133, 149), (150, 165), (166, 182)]
XD0 = X5R + 1.1; XD1 = XD0 + 4.6
prev = 'F8'
zr = 7.9; xstart = XD0
for k, (a, b) in enumerate(ROWS6):
    nm = f'C{k + 1}'; dr = +1 if k % 2 == 0 else -1
    p1 = [xstart, 0, zr]
    t_o, cause = trigger(TN[a], [prev]); st = ('drop', t_o, cause)
    if b == 182:
        p, vin = entry(nm, list(range(a, 182)), p1, st, lambda i: 2.2 * dr if i < 178 else 1.6 * dr, dz=-0.012, end=None, row='well', dvx=(-0.2, 0.12))
        # 181 -> 182: a kicker key throws it up and it falls into the bell-rope bowl
        best = None
        for tilt_ok in (38, 55, 66):
            for dz in np.linspace(-0.3, -4.0, 500):
                vo = vout_for(1.25 * dr * (TN[182] - TN[181]), dz, TN[182] - TN[181]); e, nn = bounce_e(vin, vo)
                if EMIN <= e <= EMAX and tilt_of(nn) < tilt_ok: best = vo; break
            if best is not None: break
        add_hit(B[nm], 181, p, vin, best, 'key'); B[nm].fly(TN[181], TN[182], p, best); pq, vq = land(p, best, TN[182] - TN[181])
        to_bowl(B[nm], 182, pq, vq, kind='ropebowl')
        EV['bowls'].append(dict(ball=nm, p=pq.tolist(), t=float(TN[182]), i=182, kind='ropebowl')); EV['windows'].append(dict(p=pq.tolist(), t=float(TN[182])))
        ROPE_BOWL = pq
    else:
        pe, _ = entry(nm, list(range(a, b + 1)), p1, st, 2.2 * dr, dz=-0.012, row='well', dvx=(-0.12, 0.1))
        xstart = pe[0] + 0.78 * dr; zr = pe[2] - 0.62
    prev = nm
XD1 = max(h['p'][0] for k in range(7) for h in B[f'C{k + 1}'].hits if h['i'] < 182)
EV['sections']['well'] = dict(t0=TN[89] - 0.6, t1=TN[182])

# ================================================================== S7: the bell tower (bars 59-62)
XB = XD1 + 1.2
p, vin = entry('H', [183, 184, 185, 186, 187, 188], [XB, 0, 6.3], ('drop', TN[182], ROPE_BOWL - [0, 0, R]), lambda i: 1.95,
               lambda i: -0.03 if i < 186 else -0.55, end=None, kind='bell', kind_of=lambda i: 'bell')
EV['bellrope'] = dict(p0=(ROPE_BOWL - [0, 0, R]).tolist(), p1=EV['hatches'][-1]['p'], t=float(TN[182]))
exit_to_basket('H', 188, p, vin, e=0.35, drop_=0.5); B['H'].hits[-1]['kind'] = 'bell'
EV['sections']['bells'] = dict(t0=TN[182], t1=TN[188] + 0.4)
EV['dawn'] = float(TN[186])

# ================================================================== S8: second theme - home along the top gallery (bars 63-94)
ZG = 9.9
XG = XB + 3.4
XT = 4.2                                      # mirror line of the twins (door tower)
TWV = 1.3
XP4 = XT + 2.85 + 16 * 0.265 * 1.95
PH = [('P1', 189, 196), ('P2', 197, 204), ('P3', 205, 212), ('P4', 213, 228)]
XS_ = np.linspace(XG, XP4 + 0.6, 4); XS_[-1] = XP4
prev = 'H'; xs = XG
for k, (nm, a, b) in enumerate(PH):
    t_o, cause = trigger(TN[a], [prev])
    if nm != 'P4':
        p, _ = entry(nm, list(range(a, b + 1)), [XS_[k], 0, ZG], ('drop', t_o, cause), -1.95, dz=-0.01, kind='walker')
    else:
        P4_state = entry(nm, list(range(a, b + 1)), [XS_[k], 0, ZG], ('drop', t_o, cause), -1.95, dz=-0.01, end=None, kind='walker')
    prev = nm
# the well: P4 bounces from wall to wall down the shaft (bars 83-94), lands on the bottom bowl at the fortissimo B-flat
p, vin = P4_state
WELL = list(range(228, 251))
p, vin = run(B['P4'], p, vin, WELL, lambda i: (-1 if (i - 228) % 2 == 0 else 1) * 2.2, lambda i: -0.4 if i == 228 else -0.14,
             rev=lambda i: i > 228, kind_of=lambda i: 'wall' if i > 228 else 'key', dvx=(-3, 3))
hl = B['P4'].hits[-1]
# last wall (250) -> bowl at the bottom (251): fall
dt = TN[251] - TN[250]
best = None
for dz in np.linspace(-0.2, -3.0, 500):
    vo = vout_for(-0.15 * np.sign(vin[0]), dz, dt); e, nn = bounce_e(vin, vo)
    if EMIN <= e <= EMAX: best = vo; break
add_hit(B['P4'], 250, p, vin, best, 'wall'); B['P4'].fly(TN[250], TN[251], p, best); pq, vq = land(p, best, dt)
to_bowl(B['P4'], 251, pq, vq, kind='bowl')
EV['bowls'].append(dict(ball='P4', p=pq.tolist(), t=float(TN[251]), i=251, kind='bowl')); EV['windows'].append(dict(p=pq.tolist(), t=float(TN[251])))
WELL_BOTTOM = pq
EV['well'] = dict(x=float(np.mean([h['p'][0] for h in B['P4'].hits if h['kind'] == 'wall'])), z0=float(B['P4'].hits[[h['i'] for h in B['P4'].hits].index(229)]['p'][2]),
                  z1=float(pq[2]))
EV['sections']['gallery'] = dict(t0=TN[189] - 0.6, t1=TN[228])
EV['sections']['shaft'] = dict(t0=TN[228], t1=TN[251])

# ================================================================== S9: twins (bars 95-109): mirrored pairs, centre -> outwards
TROWS = [(252, 261), (262, 271), (272, 281), (282, 291), (292, 301), (302, 310)]
ZQ1 = 3.35
zt = ZQ1 + 0.85 + 5 * 0.72; prev_c = (TN[251], WELL_BOTTOM - [0, 0, R])
TW = {i: LEAD[i][7] for i in range(NN) if LEAD[i][7] is not None}
for k, (a, b) in enumerate(TROWS):
    t_o, cause = trigger(TN[a], [f'T{k}a'] if k else ['P4'])
    for side, sgn in (('a', +1), ('b', -1)):
        nm = f'T{k + 1}{side}'
        entry(nm, list(range(a, b + 1)), [XT + sgn * 0.3, 0, zt], ('drop', t_o, cause), TWV * sgn, dz=-0.02, kind='twin', y=0.45 * (k % 2),
              kind_of=(lambda i: 'key') if side == 'a' else (lambda i: 'twinkey'), end_kind='bowl' if side == 'a' else 'twinbowl',
              row='twins', dvx=(-0.12, 0.1))
    zt -= 0.72
# the b-twins strike vln II's note: mark it on their hits
for n, bl in B.items():
    if n.startswith('T') and n.endswith('b'):
        for h in bl.hits: h['twin_pitch'] = TW.get(h['i'])
EV['twins'] = dict(xc=XT, t0=float(TN[252]), t1=float(TN[310]))
EV['sections']['twins'] = dict(t0=TN[251], t1=TN[310])

# ================================================================== S10: down to the door (bars 110-116)
t_o, cause = trigger(TN[311], ['T6a'])
entry('Q1', list(range(311, 324)), [XT + 2.45, 0, ZQ1], ('drop', t_o, cause), -2.1, dvx=(-0.12, 0.1), kind='runner')
t_o, cause = trigger(TN[324], ['Q1'])
entry('V', [324, 325, 326, 327], [2.4, -0.75, 2.85], ('drop', t_o, cause), 2.3, y=-0.75)
t_o, cause = trigger(TN[328], ['Q1', 'V'])
entry('Q2', list(range(328, 339)), [2.05, 0, 2.55], ('drop', t_o, cause), 2.1, dvx=(-0.12, 0.1), kind='runner')
EV['sections']['descent'] = dict(t0=TN[311], t1=TN[338])

# ================================================================== S11: three blows on the door lever (bars 117-122)
HN = [('H1', (339, 342), -0.45), ('H2', (343, 346), 0.0), ('H3', (347, 350), 0.45)]
prev = 'Q2'
LEV = []
for k, (nm, (a, b), yy) in enumerate(HN):
    t_o, cause = trigger(TN[a], [prev], hi=0.7)
    p, _ = entry(nm, [a, a + 1, a + 2, b], [XT + 3.2, yy, 1.65], ('drop', t_o, cause), -2.2, dz=-0.05, kind='hammer', y=yy,
                 end_kind='lever', window=False)
    LEV.append(dict(ball=nm, p=p.tolist(), t=float(TN[b]), step=k)); SFX.append(['ratchet', TN[b]])
    prev = nm
EV['lever'] = LEV
EV['door_open'] = float(TN[350])
EV['sections']['finale'] = dict(t0=TN[339] - 0.6, t1=TN[350])

# ================================================================== the pour: the castle's balls stream out of the open door
rng = np.random.default_rng(11)
POUR = []
TO = float(TN[350]) + 0.18
for k in range(60):
    y = float(rng.choice([-1, 1]) * rng.uniform(0.5, 1.05))
    t0 = TO + k * 0.04 + rng.uniform(0, 0.03)
    p0 = np.array([rng.uniform(0.65, 1.05), y, rng.uniform(0.4, 2.8)])
    v0 = np.array([-rng.uniform(2.0, 3.4), rng.uniform(-0.25, 0.25) * np.sign(y), rng.uniform(-0.2, 0.8)])
    segs = [dict(k='rest', t0=0.0, t1=t0, p=p0.tolist())]
    t, pp, vv = t0, p0, v0
    for _ in range(8):
        tl = fly_until_z(pp, vv, R)
        segs.append(dict(k='fly', t0=t, t1=t + tl, p=pp.tolist(), v=vv.tolist()))
        pp, vv = land(pp, vv, tl); t += tl
        vv = np.array([vv[0] * 0.85, vv[1] * 0.85, -vv[2] * 0.45])
        if vv[2] < 0.35: break
    dec = 0.45; sp = np.linalg.norm(vv[:2]); tr = sp / dec
    a = -dec * np.array([vv[0], vv[1], 0]) / max(1e-6, sp)
    vv = np.array([vv[0], vv[1], 0.0]); pp = np.array([pp[0], pp[1], R])
    segs.append(dict(k='acc', t0=t, t1=t + tr, p=pp.tolist(), v=vv.tolist(), a=a.tolist()))
    pe = pp + vv * tr + 0.5 * a * tr * tr
    segs.append(dict(k='rest', t0=t + tr, t1=200.0, p=pe.tolist()))
    POUR.append(dict(seg=segs, t0=t0, col=int(k % 6)))
EV['pour'] = POUR


# key sizes (width along the surface, depth along y, thickness) - shared by scene.py and verify.py
KEYDIM = {'key': (0.17, 0.42, 0.07), 'step': (0.24, 0.6, 0.09), 'wall': (0.17, 0.42, 0.07), 'twinkey': (0.17, 0.3, 0.07),
          'bell': (0.24, 0.24, 0.12)}

# ------------------------------------------------------------------ checks
def check():
    info = {}
    hs = [(n, h) for n, b in B.items() for h in b.hits]
    real = [h for n, h in hs if np.linalg.norm(h['vout']) > 0]
    info['e_max'] = round(max(h['e'] for h in real), 3); info['e_min'] = round(min(h['e'] for h in real), 3)
    info['speed_out_in_max'] = round(max(np.linalg.norm(h['vout']) / max(1e-6, np.linalg.norm(h['vin'])) for h in real), 3)
    info['tilt_max_keys'] = round(max(tilt_of(h['n']) for n, h in hs if h['kind'] in ('key', 'step', 'bell', 'twinkey')), 1)
    cov = {}
    for n, h in hs: cov.setdefault(h['i'], []).append(n)
    info['notes_uncovered'] = sorted(set(range(NN)) - set(cov))
    info['balls'] = len(B); info['bowls'] = len(EV['bowls']); info['hatches'] = len(EV['hatches'])
    xs = [h['p'][0] for n, h in hs]; zs = [h['p'][2] for n, h in hs]
    info['x_range'] = (round(min(xs), 2), round(max(xs), 2)); info['z_range'] = (round(min(zs), 2), round(max(zs), 2))
    return info


if __name__ == '__main__':
    for k, v in check().items(): print(k, v)
    out = dict(G=G, R=R, keydim=KEYDIM, door=dict(x0=DOOR_X0, x1=DOOR_X1, h=DOOR_H, plate=PLATE),
               balls={n: dict(kind=b.kind, y=b.y, seg=b.seg, hits=b.hits) for n, b in B.items()}, ev=EV, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
