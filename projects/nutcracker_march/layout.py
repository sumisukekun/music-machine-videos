"""The Nutcracker - March: 'Toy Soldiers vs the Mouse King - who wins?'.  Physics + story planner.
Run after arrange.py; writes layout.json (read by scene.py, render_audio.py, verify.py).

Story (side view of a toy battlefield under the Christmas tree, the stage scrolls to the right):
  fanfare (bars 9-12)  three tin-soldier marbles drop off the toy fort and play the fanfare chords side by side, one note
                       each per chord, on drum pads; the drummer marble adds the two bassoon B3s
  tune (13-16)         the Nutcracker captain plays the violin tune, the drummer the walking bass.  Then all four step
                       into cable-car gondolas and ride up to the cable above the battlefield: they watch from there
  mice (16-24)         the Mouse King drops from his cheese tower; with two mouse guards he answers with the same
                       fanfare in E minor; the rat drummer takes over the bass; mouse marbles drop from the cheese blimp
                       (flute grace-note figures); the King charges down his grand staircase (20 notes) into a cage
  army (41-48)         the mouse army: 125 mouse marbles drop from the blimp, one per 16th note, the rat drums the pedal
  finale (81-88)       the gondolas come down, the soldiers play the fanfare again; golden shells rise and fall from the
                       soldiers' balloon (the violin runs and the flute cascade); captain + drummer play the tune; on
                       the final chord all four land at once and the score flips: SOLDIERS 318 - 317 MICE
Physics: one constant gravity G; between contacts pure parabolas; bounces are impulses along the contact normal with
restitution emin <= e <= emax (never adds speed); cups / gondolas / troughs absorb (e = 0).  Driven mechanisms: the
gondolas and carts (smooth, cable visible)."""
import json, os, math, pickle, hashlib
import numpy as np
from phys import Ball, seg_at, bounce_e, G, EMAX, EMIN, TILT, pen_box
import phys
HERE = os.path.dirname(os.path.abspath(__file__))
_CF = '/tmp/nutcracker_march_plan_cache.pkl'
_C = pickle.load(open(_CF, 'rb')) if os.path.exists(_CF) else {}


def plan(times, p0, vins, xt, zt, yt, **kw):
    times = np.asarray(times, float)
    key = hashlib.md5(repr((np.round(times, 6).tolist(), np.round(np.asarray(p0, float), 6).tolist(), [np.round(np.asarray(v, float), 6).tolist() for v in vins],
                            [(round(xt(t), 6), round(zt(t), 6), round(yt(t), 6)) for t in times], sorted((k, repr(v)) for k, v in kw.items()))).encode()).hexdigest()
    if key not in _C:
        r = phys.plan(times, p0, vins, xt, zt, yt, **kw); _C[key] = (r, phys.plan.last); pickle.dump(_C, open(_CF, 'wb'))
    r, plan.last = _C[key]
    return r


def plan_retry(*a, **kw):
    err = None
    for W, NX, NZ in ((220, 17, 31), (400, 21, 41), (700, 25, 51)):
        try: return plan(*a, **dict(kw, W=W, NX=NX, NZ=NZ))
        except RuntimeError as e_: err = e_; print('retry', e_)
    raise err


E = json.load(open(f'{HERE}/events.json')); VO = json.load(open(f'{HERE}/voices.json'))
T0, U, FINAL = E['t0'], E['u'], E['final']
BAR_T = {int(k): v for k, v in E['bar_t'].items()}
def bt(bar, o=0.0): return BAR_T[bar] + o * U
V = 1.5                                            # the stage moves right at V
SLOPE = -0.085                                     # and gently down (the marbles' energy)
def xs(t): return V * (t - T0)
def zs(t): return SLOPE * (t - T0)
R = dict(cap=0.24, s2=0.2, s3=0.2, drum=0.24, king=0.25, m2=0.19, m3=0.19, rat=0.22)
RA_, RS_ = 0.1, 0.1                             # mouse marble, golden shell
LANE = dict(cap=(1.7, 3.75, 0.55), king=(2.3, 3.75, 0.55), s2=(1.1, 2.5, 0.0), m2=(1.1, 2.5, 0.0), s3=(0.5, 1.3, -0.55), m3=(0.5, 1.3, -0.55),
            drum=(0.0, 0.0, -1.1), rat=(-0.62, 0.0, -1.1))          # y (depth), z above the stage line, x offset
CREF = dict(cap=74, king=74, s2=71, m2=60, s3=65, m3=55, drum=45, rat=45)
CK = dict(cap=0.07, king=0.07, s2=0.05, m2=0.05, s3=0.05, m3=0.05, drum=0.025, rat=0.025)
TEAM = E['team']
SFX = []; EV = {}; CUPS = []; RAMPS = []
TDROP = 0.6
LONG = 1.2                                          # a gap at least this long: the marble waits in a cup
GONDOLA = 8.0                                       # ... at least this long: the cup is a gondola that rides up to the cable
TF_C = 0.4                                          # the cup opens and the marble falls TF_C to its next note
Z_CABLE = 5.5                                       # cable height above the stage line


def smoother(u): u = min(1.0, max(0.0, u)); return u * u * u * (u * (6 * u - 15) + 10)


def hermite(p0, v0, p1, v1, T_, tau):
    s = tau / T_; h00 = 2 * s ** 3 - 3 * s ** 2 + 1; h10 = s ** 3 - 2 * s ** 2 + s; h01 = -2 * s ** 3 + 3 * s ** 2; h11 = s ** 3 - s ** 2
    return h00 * p0 + h10 * T_ * v0 + h01 * p1 + h11 * T_ * v1


def ramp_from(t_a, p_a, vin, timesn, t_next, xt, zt, R_):
    """a run of quick notes: the marble lands on a little inclined xylophone ramp (keeps the velocity along the ramp, the
    normal part is absorbed), rolls down over the bars (rolling: a = 5/7 g sin th) and leaves the low end in free flight."""
    best = None
    for th in [math.radians(a) for a in (8, 10, 13, 16, 19, 22, 25, 28)]:
        u = np.array([math.cos(th), 0, -math.sin(th)]); n = np.array([math.sin(th), 0, math.cos(th)])
        v0 = float(vin @ u)
        if v0 < 0.25: continue
        acc = 5 / 7 * G * math.sin(th)
        tau = np.array(timesn) - t_a
        s_k = v0 * tau + 0.5 * acc * tau ** 2
        t_end = timesn[-1] + 0.05; te = t_end - t_a
        p_end = p_a + u * (v0 * te + 0.5 * acc * te * te); v_end = u * (v0 + acc * te)
        dt = t_next - t_end
        p_n = p_end + v_end * dt + np.array([0, 0, -0.5 * G * dt * dt]); v_n = v_end + np.array([0, 0, -G * dt])
        if v_n[2] > -0.6: continue
        cost = (p_n[0] - xt(t_next)) ** 2 + 2 * (p_n[2] - zt(t_next)) ** 2 + 0.02 * np.linalg.norm(v_n) ** 2
        if best is None or cost < best['cost']:
            best = dict(th=th, u=u, n=n, v0=v0, acc=acc, s_keys=s_k.tolist(), t_a=t_a, t_end=t_end, p_a=p_a, p_end=p_end, v_end=v_end,
                        p_next=p_n, v_next=v_n, cost=cost, vin=vin, nn=len(timesn))
    assert best is not None, ('no ramp', t_a)
    return best


def plan_voice(name, TIMES, p0, vin0, xt, zt, yt, R_, **kw):
    """segments: hops (planned bounces) | ramp (a run of quick notes rolled over a little xylophone ramp) |
    cup / gondola (a long gap: the marble lands in a cup - that landing is the note - which carries it to above the next
    note and opens TF_C before it)"""
    n = len(TIMES); i = 0; p = np.array(p0, float); vins = [np.array(vin0, float)]; out = []
    run_start = {}
    a = 0
    while a < n - 3:
        k = a
        while k + 1 < n and TIMES[k + 1] - TIMES[k] < 0.15: k += 1
        if k - a >= 3: run_start[a] = k - a + 1; a = k + 1
        else: a += 1

    def add_cup(c, t_land, pc, vc, t_next):
        gap = t_next - t_land
        tgt = np.array([xt(t_next), yt(t_next), zt(t_next)])
        if gap >= GONDOLA:                            # the gondola: up to the cable, along, down; level for the last 0.6 s
            kind = 'gondola'; t_rel = t_next - TF_C
            vx = (tgt[0] - pc[0]) / gap
            v_rel = np.array([vx, 0.0, 0.0])
            p_rel = tgt - v_rel * TF_C + np.array([0, 0, 0.5 * G * TF_C ** 2])
            zc = zs(t_land) + Z_CABLE
            ts = np.arange(t_land, t_rel + 1e-9, 1 / 120.0); ts[-1] = t_rel
            up_t, dn_t = 2.6, 2.6
            P_ = []
            for t in ts:
                x_ = hermite(pc[0], vc[0], p_rel[0], vx, t_rel - t_land, t - t_land)
                y_ = pc[1] + (p_rel[1] - pc[1]) * smoother((t - t_land) / (t_rel - t_land))
                zlo = pc[2] + (p_rel[2] - pc[2]) * smoother((t - (t_rel - 0.6 - dn_t)) / dn_t)
                h = smoother((t - t_land) / up_t) * (1 - smoother((t - (t_rel - 0.6 - dn_t)) / dn_t))
                P_.append([x_, y_, zlo + h * (zc - zlo)])
            path = (ts, np.array(P_))
        else:
            kind = 'cup'; t_rel = t_next - TF_C
            vcup = (tgt - np.array([0, 0, -0.5 * G * TF_C ** 2]) - pc) / gap
            vcup[1] = 0.0
            v_rel = vcup; p_rel = pc + vcup * (t_rel - t_land)
            ts = np.arange(t_land, t_rel + 1e-9, 1 / 120.0); ts[-1] = t_rel
            v0 = np.array([vc[0], 0.0, 0.0])                # the cart takes the marble's own horizontal speed and eases to its own
            path = (ts, np.array([hermite(pc, v0, p_rel, v_rel, t_rel - t_land, t - t_land) for t in ts]))
        out.append((kind, c, t_land, pc, vc, path, t_rel, p_rel, v_rel, t_next))
        CUPS.append(dict(who=name, kind=kind, t=float(t_land), p=pc.tolist(), t_rel=float(t_rel), p_rel=p_rel.tolist(), v_rel=v_rel.tolist()))
        SFX.append(['cup', float(t_land)]); SFX.append(['cup_open', float(t_rel - 0.1)])
        return p_rel + v_rel * TF_C + np.array([0, 0, -0.5 * G * TF_C ** 2]), v_rel + np.array([0, 0, -G * TF_C])

    while True:
        c = i
        while c < n - 1 and TIMES[c + 1] - TIMES[c] < LONG and c not in run_start: c += 1
        if c > i:
            kw2 = dict(kw)
            if c in run_start: kw2['dn_last'] = 0.12
            hits, pc, vc = plan_retry(TIMES[i:c + 1], p, vins, xt, zt, yt, R=R_, **kw2)
            out.append(('hops', i, hits, TIMES[c]))
        else:
            pc, vc = p, vins[0]
        if c == n - 1:
            out.append(('end', c, pc, vc)); return out
        if c in run_start:
            m = run_start[c]
            rp = ramp_from(TIMES[c], pc, vc, TIMES[c:c + m], TIMES[c + m], xt, zt, R_)
            out.append(('ramp', c, rp)); RAMPS.append(dict(who=name, i=c, **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in rp.items()}))
            p = rp['p_next']; vins = [rp['v_next']]; i = c + m
            continue
        p, vq = add_cup(c, TIMES[c], pc, vc, TIMES[c + 1]); vins = [vq]
        i = c + 1


def write_voice(ball, segs, TIMES, pitch_of):
    for sg in segs:
        if sg[0] == 'hops':
            _, i0, hits, t_next = sg
            for k, h in enumerate(hits):
                ball.hit(h['t'], h['p'], h['vins'][0], h['vout'], 'key', pitch_of(i0 + k), ev=i0 + k)
                t1 = hits[k + 1]['t'] if k + 1 < len(hits) else t_next
                ball.fly(h['t'], t1, h['p'], h['vout'])
        elif sg[0] in ('cup', 'gondola'):
            kind, c, t_land, pc, vc, path, t_rel, p_rel, v_rel, t_next = sg
            ball.hit(t_land, pc, vc, path[1][1] * 0 + (path[1][1] - path[1][0]) / (path[0][1] - path[0][0]), kind, pitch_of(c), ev=c)
            ball.path(path[0], path[1])
            ball.fly(t_rel, t_next, p_rel, v_rel)
        elif sg[0] == 'ramp':
            _, c, rp = sg
            u = rp['u']; v_land = u * rp['v0']; m = rp['nn']
            ball.hit(TIMES[c], rp['p_a'], rp['vin'], v_land, 'rampland', pitch_of(c), ev=c)
            for j in range(1, m):
                pj = rp['p_a'] + u * rp['s_keys'][j]; vj = u * (rp['v0'] + rp['acc'] * (TIMES[c + j] - TIMES[c]))
                ball.hits.append(dict(t=float(TIMES[c + j]), p=pj.tolist(), vin=vj.tolist(), vout=vj.tolist(), n=rp['n'].tolist(), e=1.0, kind='roll',
                                      pitch=pitch_of(c + j), ev=c + j))
            ts = np.arange(TIMES[c], rp['t_end'] + 1e-9, 1 / 240.0); ts[-1] = rp['t_end']
            ball.path(ts, np.array([rp['p_a'] + u * (rp['v0'] * (t - TIMES[c]) + 0.5 * rp['acc'] * (t - TIMES[c]) ** 2) for t in ts]))
            ball.fly(rp['t_end'], TIMES[c + m], rp['p_end'], rp['v_end'])
    return segs[-1]


# =================================================================== the marbles that bounce
BALLS = {}
START = {}
for name in ('cap', 's2', 's3', 'drum', 'king', 'm2', 'm3', 'rat'):
    ev = VO[name]; TT = np.array([e['t'] for e in ev]); PP = [e['p'] for e in ev]
    y0, z0, x0 = LANE[name]
    def pt(t, TT=TT, PP=PP): return max(PP[int(np.argmin(np.abs(TT - t)))])
    xt = lambda t, x0=x0: xs(t) + x0
    zt = lambda t, z0=z0, name=name, pt=pt: zs(t) + z0 + CK[name] * (pt(t) - CREF[name])
    yt = lambda t, y0=y0: y0
    b = Ball(name, R[name]); BALLS[name] = b
    # the start: the marble waits on its shelf (the fort for soldiers, the cheese tower for mice) and drops TDROP
    pc0 = np.array([xt(TT[0]), y0, zt(TT[0])]); p_rel = pc0 + np.array([0, 0, 0.5 * G * TDROP ** 2])
    b.rest(0.0, TT[0] - TDROP, p_rel); b.fly(TT[0] - TDROP, TT[0], p_rel, [0, 0, 0])
    START[name] = dict(p=p_rel.tolist(), t_release=float(TT[0] - TDROP)); SFX.append(['release', float(TT[0] - TDROP)])
    kw = dict(vxlim=(0.5, 3.0))
    if name in ('cap', 'king'): kw.update(ext=(0.2, 0.2, 0.05))
    segs = plan_voice(name, TT[:-1].tolist() + [TT[-1]], pc0, [0, 0, -G * TDROP], xt, zt, yt, R[name], **kw)
    _, _, pe, ve = write_voice(b, segs, TT, lambda k, PP=PP: PP[k])
    # the last note: the marble lands in its cup / cage / podium and stays there
    b.hit(TT[-1], pe, ve, [0, 0, 0], 'cupfinal', PP[-1], ev=len(PP) - 1)
    b.rest(TT[-1], FINAL + 30, pe)
    print(name, 'hits', len(b.hits), 'x', round(min(h['p'][0] for h in b.hits), 1), round(max(h['p'][0] for h in b.hits), 1))
EV['start'] = START
EV['final_cups'] = {n: dict(t=b.hits[-1]['t'], p=b.hits[-1]['p']) for n, b in BALLS.items()}

# =================================================================== mouse marbles and golden shells (one marble = one note)
# They are released from hatches under a blimp that flies with the stage (so they keep its speed V), fall, strike their
# key, bounce off (e along the key normal, tilted away from the other lanes) and drop into a soft trough (absorbed).
PELLETS = []
def ballistic_land(c, v, z_land):
    a_, b_, c_ = -0.5 * G, v[2], c[2] - z_land
    return (-b_ - math.sqrt(b_ * b_ - 4 * a_ * c_)) / (2 * a_)


def pellet_track(t_hit, top, n, r, z_rel, vx, e, z_land):
    c = np.array(top) + np.array(n) * r
    tf = math.sqrt(2 * (z_rel - c[2]) / G)
    rel = c + np.array([-vx * tf, 0, 0.5 * G * tf * tf]); vin = np.array([vx, 0, -G * tf])
    nn = np.array(n, float); vout = vin - (1 + e) * (vin @ nn) * nn
    tl = ballistic_land(c, vout, z_land)
    return dict(c=c, tf=tf, rel=rel, vin=vin, vout=vout, tl=tl, t_rel=t_hit - tf, t_land=t_hit + tl)


def pos_at(tr, t):
    if t < tr['t_rel'] - 0.25 or t > tr['t_land']: return None
    th = tr['t_rel'] + tr['tf']
    if t <= tr['t_rel']: return tr['rel'] + np.array([V * (t - tr['t_rel']), 0, 0])
    if t <= th: u = t - tr['t_rel']; return tr['rel'] + np.array([tr['vin'][0], 0, 0]) * u + np.array([0, 0, -0.5 * G * u * u])
    u = t - th; return tr['c'] + tr['vout'] * u + np.array([0, 0, -0.5 * G * u * u])


TG = np.arange(0, FINAL + 2, 1 / 120.0)
def track_pts(tr):
    m = (TG >= tr['t_rel'] - 0.25) & (TG <= tr['t_land'])
    return TG[m], np.array([pos_at(tr, t) for t in TG[m]])


MAINP = {n: np.array([b.at(t) for t in TG]) for n, b in BALLS.items()}
MAINK = []                                         # keys of the bouncing marbles: (t, top, n)
for n, b in BALLS.items():
    for h in b.hits:
        if h['kind'] in ('key',): MAINK.append((h['t'], np.array(h['p']) - b.r * np.array(h['n']), np.array(h['n'])))
KLEAD, KSTAY = 0.45, 0.55
KALIVE = (0.45, 0.55 + 0.25)
PLACED_ALL = []                          # pellet keys pop up KLEAD before and vanish KSTAY after their note
EXT_P = (0.075 * 1.18, 0.2 * 1.18, 0.035 * 1.18)


def place_pellets(tag, evs, r, lane_y, z0, kp, pref, tilt_dir, z_hatch, dx_opts, e=0.5):
    placed = list(PLACED_ALL)
    for j, ev in enumerate(evs):
        t = ev['t']; p = ev['p'][0]; best = None
        for dx in dx_opts:
            for row in (0, 1):
                y = lane_y + (0.42 * row * (-tilt_dir))
                top = np.array([xs(t) + dx, y, zs(t) + z0 + kp * (p - pref)])
                th = math.radians(30.0); n = np.array([0, tilt_dir * math.sin(th), math.cos(th)])
                tr = pellet_track(t, top, n, r, zs(t) + z_hatch, V, e, zs(t) + z0 - 1.5)
                ts_, P_ = track_pts(tr)
                bad = 0.0
                for q in placed[-40:]:                  # other pellets' keys while they stand, other pellets
                    alive = (ts_ > q['t'] - KALIVE[0]) & (ts_ < q['t'] + KALIVE[1])
                    if alive.any():
                        pen = pen_box(P_[alive], q['top'], q['n'], EXT_P, r).max()
                        bad += max(0, pen + 0.01) * 50
                    qa = (q['ts'] >= ts_[0]) & (q['ts'] <= ts_[-1])
                    if qa.any():
                        i0 = int(np.searchsorted(ts_, q['ts'][qa][0]))
                        k = min(len(ts_) - i0, int(qa.sum()))
                        if k > 0:
                            d = np.linalg.norm(P_[i0:i0 + k] - q['P'][qa][:k], axis=1).min()
                            bad += max(0, 2 * r + 0.04 - d) * 50
                    # this key vs their tracks
                    qal = (q['ts'] > t - KALIVE[0]) & (q['ts'] < t + KALIVE[1]) & ((q['ts'] < q['t'] - 0.03) | (q['ts'] > q['t'] + 0.03))
                    if qal.any():
                        pen = pen_box(q['P'][qal], top, n, EXT_P, r).max(); bad += max(0, pen + 0.01) * 50
                for (tk, ktop, kn) in MAINK:            # the bouncing marbles' keys and the marbles themselves
                    if abs(tk - t) < 2.5:
                        al = (ts_ > tk - 1.2) & (ts_ < tk + 1.25)
                        if al.any(): bad += max(0, pen_box(P_[al], ktop, kn, (0.24, 0.23, 0.07), r).max() + 0.02) * 50
                i0 = int(round(ts_[0] * 120)); i1 = i0 + len(ts_)
                for nm_, Pm in MAINP.items():
                    d = np.linalg.norm(Pm[i0:i1] - P_, axis=1).min(); bad += max(0, R[nm_] + r + 0.08 - d) * 50
                cost = bad + 0.05 * abs(dx) + 0.02 * row
                if best is None or cost < best[0]: best = (cost, dict(t=t, p=p, top=top, n=n, tr=tr, ts=ts_, P=P_, dx=dx, row=row, grace=ev['grace']))
        placed.append(best[1])
        if best[0] > 0.0901: print('WARN pellet', tag, j, round(t, 2), round(best[0], 2))
    out_ = placed[len(PLACED_ALL):]; PLACED_ALL.extend(out_)
    return out_


T_ARMY = E['secs'][[s_[0] for s_ in E['secs']].index('army')][1]
LANES_P = dict(armyB=(-1, 3.3, 3.4, 0.055, 64, 1, 6.3, 1.3), armyC=(-1, -1.15, 3.0, 0.055, 64, -1, 5.9, 1.3), shell=(1, 3.0, 3.0, 0.05, 52, 1, 6.9, 1.3))
DXO = (0.0, -0.2, 0.2, -0.4, 0.4, -0.6, 0.6, -0.8, 0.8)
ARMY = place_pellets('armyB', [e for e in VO['army'] if e['t'] < T_ARMY - 0.5], RA_, 3.3, 3.4, 0.055, 64, 1, 6.3, [1.3 + d for d in DXO]) + \
       place_pellets('armyC', [e for e in VO['army'] if e['t'] >= T_ARMY - 0.5], RA_, -0.6, 3.0, 0.055, 64, 1, 5.9, [1.3 + d for d in DXO])
SHELL = place_pellets('shell', VO['shell'], RS_, 3.0, 3.0, 0.05, 52, 1, 6.9, [1.3 + d for d in DXO])
for tag, L_, r in (('a', ARMY, RA_), ('s', SHELL, RS_)):
    for j, q in enumerate(L_):
        b = Ball(f'{tag}{j:03d}', r); tr = q['tr']
        ts_h = np.linspace(tr['t_rel'] - 0.25, tr['t_rel'], 31)              # held in its hatch, moving with the blimp
        b.path(ts_h, np.array([tr['rel'] + np.array([V * (t - tr['t_rel']), 0, 0]) for t in ts_h])); b.fly(tr['t_rel'], q['t'], tr['rel'], [V, 0, 0])
        b.hit(q['t'], tr['c'], tr['vin'], tr['vout'], 'pellet', q['p'], ev=j)
        b.fly(q['t'], tr['t_land'], tr['c'], tr['vout'])
        pl = tr['c'] + tr['vout'] * tr['tl'] + np.array([0, 0, -0.5 * G * tr['tl'] ** 2])
        vl = tr['vout'] + np.array([0, 0, -G * tr['tl']])
        b.hits.append(dict(t=float(tr['t_land']), p=pl.tolist(), vin=vl.tolist(), vout=[0, 0, 0], n=[0, 0, 1], e=0.0, kind='trough', pitch=None, ev=None))
        b.rest(tr['t_land'], FINAL + 30, pl)
        b.t_show = tr['t_rel'] - 0.25; b.t_land = tr['t_land']
        BALLS[b.name] = b; PELLETS.append(b)
EV['pellet_keys'] = [dict(name=f'{tag}{j:03d}', t=q['t'], p=q['p'], top=q['top'].tolist(), n=q['n'].tolist(), grace=q['grace'], team=('M' if tag == 'a' else 'S'))
                     for tag, L_ in (('a', ARMY), ('s', SHELL)) for j, q in enumerate(L_)]
EV['pellet_show'] = {b.name: float(b.t_show) for b in PELLETS}
EV['blimps'] = dict(armyB=dict(t0=float(min(q['tr']['t_rel'] for q in ARMY if q['t'] < T_ARMY - 0.5)), t1=float(max(q['tr']['t_rel'] for q in ARMY if q['t'] < T_ARMY - 0.5)), y=3.3, z=6.3, dx=1.3),
                    armyC=dict(t0=float(min(q['tr']['t_rel'] for q in ARMY if q['t'] >= T_ARMY - 0.5)), t1=float(max(q['tr']['t_rel'] for q in ARMY if q['t'] >= T_ARMY - 0.5)), y=-0.6, z=5.9, dx=1.3),
                    shell=dict(t0=float(min(q['tr']['t_rel'] for q in SHELL)), t1=float(max(q['tr']['t_rel'] for q in SHELL)), y=3.0, z=6.9, dx=1.3))
EV['troughs'] = dict(armyB=dict(y=3.3, z=3.4 - 1.5, dir=1), armyC=dict(y=-0.6, z=3.0 - 1.5, dir=1), shell=dict(y=3.0, z=3.0 - 1.5, dir=1))
EV['cups'] = CUPS; EV['ramps'] = RAMPS
EV['v'] = V; EV['slope'] = SLOPE; EV['z_cable'] = Z_CABLE
# the score: every key struck = one point for its team
SCORE = sorted([(h['t'], TEAM[n if n in TEAM else ('army' if n[0] == 'a' else 'shell')]) for n, b in BALLS.items() for h in b.hits if h['pitch'] is not None])
EV['score_events'] = SCORE
print('score', {k: sum(1 for t, s in SCORE if s == k) for k in 'SM'}, 'pellets', len(ARMY), len(SHELL))


def check():
    info = {}
    hs = [h for b in BALLS.values() for h in b.hits if h['kind'] in ('key', 'pellet')]
    info['e_range'] = (round(min(h['e'] for h in hs), 3), round(max(h['e'] for h in hs), 3))
    allh = [h for b in BALLS.values() for h in b.hits if np.linalg.norm(h['vout']) > 0 and h['kind'] not in ('cup', 'gondola', 'rampland')]
    info['speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in allh), 3)
    info['tilt_max_deg'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for h in hs), 1)
    for n in ('cap', 's2', 's3', 'drum', 'king', 'm2', 'm3', 'rat'):
        b = BALLS[n]; zz = [h['p'][2] - zs(h['t']) for h in b.hits]
        info[n] = dict(hits=len(b.hits), zrel=(round(min(zz), 2), round(max(zz), 2)))
    info['cups'] = [(c['who'], c['kind'], round(c['t'], 2), round(c['t_rel'], 2)) for c in CUPS]
    return info


if __name__ == '__main__':
    for k, v in check().items(): print(k, v)
    out = dict(G=G, R=dict(R, army=RA_, shell=RS_), lanes=LANE,
               balls={n: dict(r=b.r, seg=b.seg, hits=b.hits) for n, b in BALLS.items()}, ev=EV, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
