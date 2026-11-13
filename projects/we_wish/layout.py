"""We Wish You a Merry Christmas - 'Can ONE caroler light up the whole village?'  Physics + story planner.
Run after arrange.py; writes layout.json (read by scene.py, render_audio.py, verify.py).

Story (side view of a snowy village street on Christmas Eve, the camera travels along +x):
  0-2 s    a dark street, every house dark. The carolers' hut: its door pops, a spring plunger pushes Carol (the red marble with a
           scarf) out onto the first stepping-stone key.
  verses   Carol plays every 'We wish you a merry Christmas' phrase on stepping-stone keys; the last note of each phrase is the
           doorbell of a house: its windows, door lamp and wreath light up (4 houses per verse, 12 in all).
  refrains each 'Good tidings ...' (22 notes) is played by snowballs: they sit on the snowy canopy edge of the next big building
           (1 bakery, 2 church, 3 town hall), drop one by one onto bells (bell height = pitch) and bounce onto the window ledge.
           Meanwhile Carol rides on:  1 ropeway gondola   2 sleigh on rails   3 Christmas-market wheel up to the tree top.
  chain    refrain 1: snowball 15 lands in the bucket of the bakery's lever; the lever pulls the latch of the pudding chute: the
           figgy-pudding ball rolls out and plays the bass in verses 2 and 3.
  end      the wheel lifts Carol above the great bell on top of the town-square tree; she drops onto it on the final 'Year',
           the pudding lands on the big drum: every house flashes. (scene.py: the camera pulls back - the village sits on top of a
           giant Christmas pudding on a dinner table, and it is set alight.)
Physics: one constant gravity G; between contacts pure parabolas; every bounce is an impulse along the surface normal (tangential
velocity kept), restitution emin <= e <= emax, so no bounce adds speed; rolling from rest on a ramp: a = (5/7) g sin(alpha); speed-ups
only from visible mechanisms (spring plunger, gondola, sleigh, wheel)."""
import json, os, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = 7.0
RC, RS, RP = 0.16, 0.12, 0.21                   # Carol, snowballs, pudding (bass)
Y_SNOW, Y_BASS = 0.78, -0.9                     # depth planes: Carol y = 0, refrain bells behind, bass lane in front
EMAX, EMIN = 0.97, 0.2
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
TN = np.array([l[0] for l in LEAD]); NN = len(TN); FINAL = E['final']
FPS = 30


class Ball:
    def __init__(s, name, kind, r):
        s.name = name; s.kind = kind; s.r = r; s.seg = []; s.hits = []
    def rest(s, t0, t1, p): s.seg.append(dict(k='rest', t0=float(t0), t1=float(t1), p=list(map(float, p))))
    def fly(s, t0, t1, p0, v0): s.seg.append(dict(k='fly', t0=float(t0), t1=float(t1), p=list(map(float, p0)), v=list(map(float, v0))))
    def path(s, t0, t1, pts): s.seg.append(dict(k='path', t0=float(t0), t1=float(t1), pts=[list(map(float, q)) for q in pts]))
    def at(s, t): return seg_at(s.seg, t)


def seg_at(seg, t):
    for g in seg:
        if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9:
            if g['k'] == 'rest': return np.array(g['p'])
            if g['k'] == 'fly':
                u = t - g['t0']; return np.array(g['p']) + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
            P = np.array(g['pts']); u = (t - g['t0']) / max(1e-9, g['t1'] - g['t0']) * (len(P) - 1)
            i = min(len(P) - 2, int(u)); f = u - i; return P[i] * (1 - f) + P[i + 1] * f
    return np.array(seg_at(seg, seg[0]['t0'])) if t < seg[0]['t0'] else seg_at(seg, seg[-1]['t1'])


def vout_for(dx, dz, dt, dy=0.0): return np.array([dx / dt, dy / dt, dz / dt + 0.5 * G * dt])


def bounce_e(vin, vout):
    d = vout - vin; n = d / max(1e-9, np.linalg.norm(d)); a = vin @ n
    return (-(vout @ n) / a if a < -1e-9 else 9.0), n


def add_hit(ball, i, p, vin, vout, kind, t=None, **kw):
    e, n = bounce_e(np.array(vin, float), np.array(vout, float))
    if np.linalg.norm(vout) < 1e-9: n = np.array([0, 0, 1.0]); e = 0.0
    h = dict(i=int(i), t=float(TN[i] if t is None else t), p=list(map(float, p)), n=list(map(float, n)), e=float(e), kind=kind,
             vin=list(map(float, vin)), vout=list(map(float, vout)), pitch=int(kw.pop('pitch', LEAD[i][1] if i >= 0 else 0)))
    h.update(kw); ball.hits.append(h); return h


def run(ball, p, vin, ts, vx_of, dz_of, ids, kind='key', emax=EMAX, emin=EMIN, tilt_max=36.0, dvx=(-0.25, 0.15), pitches=None):
    """ball centre p (contact) at time ts[0] with incoming velocity vin; hops through ts[1:].
    vx_of(k) target horizontal speed of hop k, dz_of(k) target height change per second of flight.
    Landing height is only lowered (never raised) until the bounce is physical, looking one hop ahead."""
    for k in range(len(ts) - 1):
        dt = ts[k + 1] - ts[k]; dzt = dz_of(k) * dt
        if k + 1 < len(ts) - 1:
            dt2 = ts[k + 2] - ts[k + 1]
            need = (dz_of(k + 1) + 0.5 * G * dt2) / (emax * 0.985)
        else: need = 0.0
        tx = vx_of(k); cx = vin[0]
        vx0 = cx + float(np.clip(tx - cx, dvx[0], dvx[1]))
        vxs = sorted(set(np.round(np.concatenate([[vx0], vx0 + np.linspace(-0.3, 0.1, 21)]), 4)), key=lambda v: abs(v - vx0))
        dzs = sorted(np.concatenate([dzt - np.linspace(0, 0.5 * G * dt * dt + 1.0, 400), dzt + np.linspace(0.003, 0.2, 30)]),
                     key=lambda d: (0 if d <= dzt else 1, abs(d - dzt)))
        best = None; cands = []
        for vx in vxs[:6]:
            for dz in dzs:
                vo = vout_for(vx * dt, dz, dt); e, n = bounce_e(vin, vo)
                if not (emin <= e <= emax) or vo[2] - G * dt > -0.3: continue
                if math.degrees(math.acos(max(-1, min(1, n[2])))) > tilt_max: continue
                cands.append((dz, vo, e, n))
                if -(vo[2] - G * dt) >= need: best = (dz, vo, e, n); break
            if best is not None: break
        if best is None and cands: best = cands[0]; print('WARN need', ball.name, ids[k], round(need, 2))
        assert best is not None, ('no physical hop', ball.name, ids[k])
        dz, vo, e, n = best
        add_hit(ball, ids[k], p, vin, vo, kind, t=ts[k], **({'pitch': pitches[k]} if pitches else {}))
        ball.fly(ts[k], ts[k + 1], p, vo)
        p = p + vo * dt + np.array([0, 0, -0.5 * G * dt * dt]); vin = vo + np.array([0, 0, -G * dt])
    return p, vin


def smooth01(u): u = min(1.0, max(0.0, u)); return u * u * (3 - 2 * u)


def sample(fn, t0, t1, step=1 / 120):
    n = max(2, int(round((t1 - t0) / step)) + 1)
    return [fn(t0 + (t1 - t0) * k / (n - 1)) for k in range(n)]


B = {}; EV = {'dev': [], 'bld': [], 'snow': [], 'houses': []}; SFX = []
C = B['carol'] = Ball('carol', 'carol', RC)
VERSE = [[i for i in range(NN) if LEAD[i][6] == 'carol' and LEAD[i][7] == p and LEAD[i][4] < 48] for p in range(3)]
REFR = [[i for i in range(NN) if LEAD[i][6] == 'snow' and LEAD[i][8] == k] for k in range(3)]
I_FINAL = NN - 1
assert [len(v) for v in VERSE] == [30, 30, 30] and [len(r) for r in REFR] == [22, 22, 21]

# ------------------------------------------------------------------ start: the carolers' hut, spring plunger
X0, Z0 = 0.0, 0.0
TF0 = 0.42; VX0 = 0.95
t1 = TN[VERSE[0][0]]
p_box = np.array([X0 - VX0 * TF0, 0.0, Z0 + 0.5 * G * TF0 ** 2])
T_PUSH = t1 - TF0
C.rest(0.0, T_PUSH, p_box)
C.fly(T_PUSH, t1, p_box, [VX0, 0, 0])
EV['hut'] = dict(p=p_box.tolist(), t_push=T_PUSH, t_lid=T_PUSH - 0.75)
SFX.append(['door', T_PUSH - 0.75]); SFX.append(['spring', T_PUSH])
p = np.array([X0, 0, Z0]); vin = np.array([VX0, 0, -G * TF0])

VX_LINE = 1.0
DEVS = ['gondola', 'sleigh', 'wheel']
HDROP = 0.75                                     # trapdoor drop onto the first key of the next verse
for k in range(3):
    ids = VERSE[k]; ts = [TN[i] for i in ids]
    def dzf(j, ts=ts):                           # after an eighth, the next longer hop goes down a step (energy for the bigger arc)
        if j > 0 and ts[j] - ts[j - 1] < 0.3 and ts[j + 1] - ts[j] > 0.3: return -0.6
        return -0.03
    p, vin = run(C, p, vin, ts, lambda j: VX_LINE, dzf, ids, dvx=(-0.6, 0.6))
    for h in C.hits[-(len(ids) - 1):]:           # doorbells = the last note of each phrase
        i = h['i']
        if i + 1 < NN and LEAD[i + 1][8] != LEAD[i][8] or i == ids[-1]: h['kind'] = 'door'; h['house'] = LEAD[i][8]
    # last note of the verse ('Year', a half note): the doorbell of the 4th house, then a hop into the machine's cup on the refrain's
    # first beat
    il = ids[-1]; tl = TN[il]; tc = TN[REFR[k][0]]; dt = tc - tl
    dev = DEVS[k]; D = dict(kind=dev, k=k, t_catch=float(tc))
    best = None
    for dz in np.linspace(-0.12, -2.4, 200):
        vo = vout_for(VX_LINE * 0.8 * dt, dz, dt); e, n = bounce_e(vin, vo)
        if EMIN <= e <= EMAX and math.degrees(math.acos(n[2])) < 36: best = (dz, vo); break
    assert best, ('catch', k)
    dz, vo = best
    add_hit(C, il, p, vin, vo, 'door', house=LEAD[il][8])
    C.fly(tl, tc, p, vo)
    pc = p + vo * dt + np.array([0, 0, -0.5 * G * dt * dt]); D['catch'] = pc.tolist(); D['vin'] = (vo + [0, 0, -G * dt]).tolist()
    SFX.append(['catch', float(tc)])
    tn = TN[VERSE[k + 1][0]] if k < 2 else FINAL
    tdrop = math.sqrt(2 * HDROP / G)
    if dev in ('gondola', 'sleigh'):
        DX = 6.4; DZ = -0.35 if dev == 'gondola' else 0.0
        tr = tn - tdrop; m0, m1 = tc + 0.4, tr - 0.4
        f = lambda t: pc + np.array([DX, 0, DZ]) * smooth01((t - m0) / (m1 - m0))
        C.rest(tc, m0, pc); C.path(m0, m1, sample(f, m0, m1)); pr = f(m1); C.rest(m1, tr, pr)
        D.update(m0=m0, m1=m1, dx=DX, dz=DZ, rel=pr.tolist(), t_rel=tr)
        SFX += [['motor' if dev == 'gondola' else 'bells', m0], ['clunk', m1], ['trap', tr]]
        C.fly(tr, tn, pr, [0, 0, 0]); p = pr + np.array([0, 0, -HDROP]); vin = np.array([0, 0, -G * tdrop])
    else:
        # Christmas-market wheel: the cup enters at the left (180 deg) and is carried over the top to the right (0 deg), where its
        # trapdoor opens over the great bell on top of the town-square tree
        RW_ = 1.45; HANG = 0.36
        Cw = pc + np.array([RW_, 0, HANG])
        HD = 1.1; tdropF = math.sqrt(2 * HD / G)
        tr = FINAL - tdropF; m0, m1 = tc + 0.4, tr - 0.5
        def f(t):
            th = math.pi * (1 - smooth01((t - m0) / (m1 - m0)))
            return Cw + np.array([RW_ * math.cos(th), 0, RW_ * math.sin(th) - HANG])
        C.rest(tc, m0, pc); C.path(m0, m1, sample(f, m0, m1)); pr = f(m1); C.rest(m1, tr, pr)
        D.update(m0=m0, m1=m1, centre=Cw.tolist(), r=RW_, hang=HANG, rel=pr.tolist(), t_rel=tr)
        SFX += [['ratchet', m0], ['clunk', m1], ['trap', tr]]
        C.fly(tr, FINAL, pr, [0, 0, 0]); pf = pr + np.array([0, 0, -HD]); vf = np.array([0, 0, -G * tdropF])
        add_hit(C, I_FINAL, pf, vf, [0, 0, 0], 'great')
        C.rest(FINAL, FINAL + 10, pf)
        EV['great'] = dict(p=pf.tolist(), t=FINAL)
    EV['dev'].append(D)


def band(t0, t1, ax):
    v = [C.at(t)[ax] for t in np.arange(t0, t1, 0.05)]; return min(v), max(v)


# ------------------------------------------------------------------ houses: one per verse phrase, door behind its doorbell key
YH = 2.1                                         # house front wall
for h in C.hits:
    if h['kind'] != 'door': continue
    hs = [g for g in C.hits if g['i'] >= 0 and LEAD[g['i']][6] == 'carol' and LEAD[g['i']][8] == h['house'] and g['kind'] in ('key', 'door')]
    zlo = min(g['p'][2] for g in hs)
    EV['houses'].append(dict(n=h['house'], x=float(h['p'][0]), door=h['p'], t_on=h['t'], zb=float(zlo - 1.05),
                             x0=float(hs[0]['p'][0]), x1=float(hs[-1]['p'][0])))

# ------------------------------------------------------------------ refrain buildings + snowballs
BNAMES = ['bakery', 'church', 'townhall']
S_I = 0
DBGF = False
for k in range(3):
    ids = REFR[k]; ts = [TN[i] for i in ids]; ps = [LEAD[i][1] for i in ids]
    D = EV['dev'][k]
    t0 = TN[ids[0]] - 0.5; t1_ = TN[ids[-1]] + 0.6
    zlo, zhi = band(t0, t1_, 2); xlo, xhi = band(t0, t1_, 0)
    n = len(ids); W = 0.27 * (n - 1)
    xB = (xlo + xhi) / 2 + (0.3 if k < 2 else -0.6)
    zb = zlo - 1.3
    bl = dict(k=k, name=BNAMES[k], x=float(xB), zb=float(zb), w=float(W + 0.9), bells=[])
    EV['bld'].append(bl)
    zbell0 = zhi + 0.95
    if k == 2: zbell0 = max(zbell0, zhi + 0.95)
    pmin = min(ps)
    bells = [(xB - W / 2 + W * j / (n - 1), zbell0 + 0.09 * (pt - pmin)) for j, pt in enumerate(ps)]
    zrail = max(b[1] for b in bells) + 0.7
    zledge = min(b[1] for b in bells) - 0.55       # top of the window ledge the snowballs land on
    bl['rail'] = float(zrail); bl['ledge'] = float(zledge); bl['ybar'] = Y_SNOW
    targets = []
    for j, (i, t, pt) in enumerate(zip(ids, ts, ps)):
        xb, zc = bells[j]
        o = B[f'snow{S_I}'] = Ball(f'snow{S_I}', 'snow', RS); S_I += 1
        h = zrail - RS - 0.05 - zc; tfall = math.sqrt(2 * h / G)
        p0 = np.array([xb, Y_SNOW, zc + h]); pc_ = np.array([xb, Y_SNOW, zc]); vin = np.array([0, 0, -G * tfall])
        o.rest(0.0, t - tfall, p0); o.fly(t - tfall, t, p0, [0, 0, 0])
        lever = (k == 0 and j == 14); DBGF = (k == 0 and j == 2)
        best = None
        if lever:
            cand = [np.array([xb + dx, 2.14, zledge + 0.06 + RS]) for dx in (0.35, 0.5, 0.2, 0.65, 0.8)]
        else:
            cand = []
            for dyy in (1.3, 1.58, 1.86):
                for dx in (0.0, 0.2, -0.2, 0.4, -0.4, 0.6, -0.6, 0.8, -0.8, 1.0, -1.0, 1.2, -1.2):
                    cand.append(np.array([xb + dx, dyy, zledge + RS]))
        others = [B[f'snow{q}'] for q in range(S_I - 1) if B[f'snow{q}'].hits and B[f'snow{q}'].hits[0].get('bld') == k]
        for tgt in cand:
            if any(np.linalg.norm(tgt - q) < 2 * RS + 0.06 for q in targets): continue
            if abs(tgt[0] - xB) > W / 2 + 0.25 and not lever: continue
            for tf in np.linspace(0.25, 0.9, 66):
                vo = vout_for(tgt[0] - xb, tgt[2] - zc, tf, tgt[1] - Y_SNOW); e, n_ = bounce_e(vin, vo)
                if not (0.12 <= e <= 0.8) or vo[2] - G * tf > -0.4: continue
                if math.degrees(math.acos(n_[2])) > 40: continue
                sc_ = abs(e - 0.4) + 0.3 * abs(tf - 0.5)
                if best is not None and sc_ >= best[0]: continue
                ok = True
                for u in np.linspace(0, tf + 0.3, 24):
                    q = pc_ + vo * min(u, tf) + np.array([0, 0, -0.5 * G * min(u, tf) ** 2])
                    for ob in others:
                        if np.linalg.norm(q - ob.at(t + u)) < 2 * RS + 0.03: ok = False; break
                    if not ok: break
                    if u > 0.03 and u <= tf:
                        for jb, (xb2, zc2) in enumerate(bells):   # clear of every bell of the row (crown + body below it)
                            ztop = (0.04 + RS) if jb != j else -0.02
                            if abs(q[0] - xb2) < 0.09 + RS and abs(q[1] - Y_SNOW) < 0.09 + RS and -0.3 < q[2] - zc2 < ztop: ok = False; break
                    if not ok: break
                    for j2 in range(j + 1, n):                  # later snowballs: hanging, falling, and their bell at the strike
                        xb2, zc2 = bells[j2]; t2 = ts[j2]
                        if t + u > t2 + 0.02: continue
                        h2 = zrail - RS - 0.05 - zc2; tf2 = math.sqrt(2 * h2 / G); s2 = max(0.0, t + u - (t2 - tf2))
                        q2 = np.array([xb2, Y_SNOW, zc2 + h2 - 0.5 * G * s2 * s2])
                        if np.linalg.norm(q - q2) < 2 * RS + 0.04: ok = False; break
                    if not ok: break
                if ok: best = (sc_, tgt, tf, vo)
            if best is not None: break
        if best is None:
            print('DBG', k, j, xb, zc, zledge, vin, len(targets))
            for tgt in cand[:5]:
                for tf in (0.4, 0.6):
                    vo = vout_for(tgt[0] - xb, tgt[2] - zc, tf, tgt[1] - Y_SNOW); e, n_ = bounce_e(vin, vo); print(tgt, tf, e, math.degrees(math.acos(n_[2])), vo[2]-G*tf)
        assert best is not None, ('snowball', k, j)
        _, tgt, tf, vo = best
        targets.append(tgt)
        add_hit(o, i, pc_, vin, vo, 'bell', t=t, bld=k)
        o.fly(t, t + tf, pc_, vo); o.rest(t + tf, FINAL + 10, tgt)
        o.hits[-1]['land'] = tgt.tolist(); o.hits[-1]['t_land'] = float(t + tf)
        bl['bells'].append(dict(i=int(i), x=float(xb), z=float(zc), ball=o.name))
        EV['snow'].append(dict(ball=o.name, bld=k, t_rel=float(t - tfall), t_hit=float(t), t_land=float(t + tf), land=tgt.tolist(), lever=lever))
        SFX.append(['poof', float(t + tf)])
        if lever:                                   # the bucket tips 0.3 rad about a pivot 0.45 to its right: the snowball rides down in it
            ANG = 0.3; piv = tgt + np.array([0.45, 0, -RS - 0.03]); t_l = t + tf
            def fl_(tt, piv=piv, t_l=t_l):
                a_ = ANG * smooth01((tt - t_l) / 0.25); v_ = tgt - piv
                return piv + np.array([v_[0] * math.cos(a_) - v_[2] * math.sin(a_), v_[1], v_[0] * math.sin(a_) + v_[2] * math.cos(a_)])
            o.seg[-1]['t1'] = t_l
            o.path(t_l, t_l + 0.25, sample(fl_, t_l, t_l + 0.25)); o.rest(t_l + 0.25, FINAL + 10, fl_(t_l + 0.25))
            EV['lever'] = dict(t=float(t_l), p=tgt.tolist(), ball=o.name, ang=ANG)

# ------------------------------------------------------------------ the figgy pudding: chute (released by the lever), bass in verses 2-3
BASS = E['bass']; BT = [b[0] for b in BASS]
P_ = B['pudding'] = Ball('pudding', 'bass', RP)
t_latch = EV['lever']['t'] + 0.12
_XB = EV['great']['p'][0] + 2.0
def _cs(t, ax):                                  # Carol's position averaged over +-1 s
    return float(np.mean([C.at(t + u)[ax] for u in np.linspace(-1.0, 1.0, 21)]))
def lane_target(t):
    """the pudding keeps a little behind and below Carol (so both stay in the picture); it ends on the drum left of the tree"""
    tw = EV['dev'][2]['t_catch']
    if t <= tw: return _cs(t, 0) - 0.7, _cs(t, 2) - 1.75
    x0_ = _cs(tw, 0) - 0.7
    return x0_ + (_XB - x0_) * (t - tw) / (BT[-1] - tw), _cs(t, 2) - 1.75
x1, z1 = lane_target(BT[0])
alc = math.radians(3.0); ac = 5.0 / 7.0 * G * math.sin(alc); TFB = 0.42
t_r0 = t_latch + 0.1
t_roll = BT[0] - TFB - t_r0
assert t_roll > 1.0, t_roll
Lc = 0.5 * ac * t_roll ** 2; vc = ac * t_roll
uc = np.array([math.cos(alc), 0, -math.sin(alc)])
pend = np.array([x1, Y_BASS, z1]) - (uc * vc * TFB + np.array([0, 0, -0.5 * G * TFB ** 2]))
pstart = pend - uc * Lc
P_.rest(0.0, t_r0, pstart)
P_.path(t_r0, t_r0 + t_roll, sample(lambda t: pstart + uc * 0.5 * ac * (t - t_r0) ** 2, t_r0, t_r0 + t_roll, 1 / 60))
ve = uc * vc; P_.fly(t_r0 + t_roll, BT[0], pend, ve)
EV['pudding'] = dict(start=pstart.tolist(), end=pend.tolist(), t_release=t_r0, t_edge=t_r0 + t_roll, alpha=math.degrees(alc), v_edge=vc,
                     t_latch=t_latch)
SFX += [['latch', t_latch], ['roll', t_r0]]
p = np.array([x1, Y_BASS, z1]); vin = ve + np.array([0, 0, -G * TFB])
NB = len(BT)
pp, vv = p, vin
for j in range(NB - 1):
    xt, zt = lane_target(BT[j + 1]); dtj = BT[j + 1] - BT[j]
    vx_t = float(np.clip((xt - pp[0]) / dtj, max(0.5, 0.62 / dtj), 1.6))
    if j >= NB - 4: vx_t = max(0.3, (_XB - pp[0]) / (BT[-1] - BT[j]))
    dz_t = min(-0.06, (zt - pp[2])) / dtj
    pp, vv = run(P_, pp, vv, [BT[j], BT[j + 1]], lambda q: vx_t, lambda q: dz_t, [-1, -1], kind='bass', pitches=[BASS[j][1], BASS[j + 1][1]],
                 dvx=(-0.9, 0.9))
    P_.hits[-1]['bass_j'] = j
add_hit(P_, -1, pp, vv, [0, 0, 0], 'drum', t=BT[-1], pitch=BASS[-1][1], bass_j=NB - 1)
P_.rest(BT[-1], FINAL + 10, pp)
EV['drum'] = dict(p=pp.tolist(), t=BT[-1])
SFX += [['final', FINAL]]


def check():
    info = {}
    hs = [(n, h) for n, b in B.items() for h in b.hits if np.linalg.norm(h['vout']) > 0]
    info['e_range'] = (round(min(h['e'] for n, h in hs), 3), round(max(h['e'] for n, h in hs), 3))
    info['speed_out_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for n, h in hs), 3)
    info['tilt_max'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for n, h in hs), 1)
    cov = set(h['i'] for b in B.values() for h in b.hits if h['i'] >= 0)
    info['melody_uncovered'] = sorted(set(range(NN)) - cov)
    info['doors'] = sum(1 for h in C.hits if h['kind'] == 'door')
    info['bass_hits'] = len(P_.hits)
    info['carol_z'] = (round(min(h['p'][2] for h in C.hits), 2), round(max(h['p'][2] for h in C.hits), 2))
    info['carol_x'] = (round(C.hits[0]['p'][0], 2), round(C.hits[-1]['p'][0], 2))
    info['pud_z'] = (round(min(h['p'][2] for h in P_.hits), 2), round(max(h['p'][2] for h in P_.hits), 2))
    info['bld'] = [(b['name'], round(b['x'], 2), round(b['zb'], 2), round(b['rail'], 2)) for b in EV['bld']]
    info['houses'] = [(round(h['x'], 1), round(h['zb'], 1)) for h in EV['houses']]
    info['great'] = [round(v, 2) for v in EV['great']['p']]
    info['t_roll'] = round(t_roll, 2)
    return info


if __name__ == '__main__':
    for k_, v_ in check().items(): print(k_, v_)
    out = dict(G=G, R={'carol': RC, 'snow': RS, 'bass': RP}, y_snow=Y_SNOW, y_bass=Y_BASS, yh=YH,
               balls={n: dict(kind=b.kind, r=b.r, seg=b.seg, hits=b.hits) for n, b in B.items()}, ev=EV, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
