"""Dance of the Sugar Plum Fairy - 'Can one marble light up the Candy Castle?'.  Physics + story planner.
Run after arrange.py; writes layout.json (read by scene.py, render_audio.py, verify.py).

Cast (one role per voice of the score):
  fairy  (celesta melody)   the glowing sugar-plum marble. It drops out of the crescent moon; every note = it lands on a
                            glass key hanging in front of the castle (the key lights up and stays lit = one more light).
                            The key heights follow the pitch, so the lit keys draw the melody.
  mint   (pizz. chords)     a peppermint marble; every chord = one peppermint disc that pops up from the snow.
  cocoa  (pizz. bass, runs) a chocolate truffle marble on chocolate drums (runs = four quick drums like a staircase).
  pearls (celesta cadenza   sugar pearls: one pearl = one note.  In the cadenza they drop from the sieve under the giant
          and Presto)       sugar bowl onto the crystal harp; in the Presto they drop from the sugar cloud onto the tree's
                            ornaments.
Story:  bar 20 beat 1 the cocoa lands on the lever drum: the lever knocks the latch, the sugar bowl tips into the sieve
  and the cadenza rains down on the crystal harp that has risen out of the snow (124 pearls) while the three marbles
  wait in their glass cups.  The cups open on the reprise.  Bar 52: the fairy lands in the cup of the candy-cane lift
  that leans against the Christmas tree; the Presto pearls light the tree ornament by ornament; the lift stops at the
  top and throws the fairy up into the star bell = the final chord, together with mint, cocoa and the last three pearls.
Physics: one constant gravity G; between contacts pure parabolas; bounces are impulses along the contact normal with
restitution emin <= e <= emax (never adds speed); cups / trough / snow absorb (e = 0).  The only driven mechanism is the
lift (smooth acceleration, visible chain)."""
import json, os, math
import numpy as np
from phys import Ball, seg_at, bounce_e, G, EMAX, EMIN, TILT
import phys, pickle, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
_CF = '/tmp/sugar_plum_plan_cache.pkl'
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


RF, RM, RC, RP = 0.22, 0.2, 0.24, 0.12            # fairy, mint, cocoa, sugar pearl
YF, YM, YC = 2.4, 1.2, 0.0                         # lanes (depth): fairy at the back, mint, cocoa in front
ZF, ZM = 3.9, 1.85                                 # lane heights above the cocoa drums
KP = 0.085                                         # fairy key height per semitone (the melody contour), around B5
V = 1.5                                            # the stage moves right at V
SLOPE = -0.10
E = json.load(open(f'{HERE}/events.json'))
FA = json.load(open(f'{HERE}/fairy.json')); MI = json.load(open(f'{HERE}/mint.json'))
CO = json.load(open(f'{HERE}/cocoa.json')); RA = json.load(open(f'{HERE}/rain.json'))
T0, QA, FINAL = E['t0'], E['qa'], E['final']
GRID = E['grid']
def bar_t(k, sec): return [g['t'] for g in GRID if g['k'] == str(k) and g['sec'] == sec][0]
SFX = []; EV = {}


def zs0(t): return SLOPE * (t - T0)
def xs0(t): return V * (t - T0)
XST = ZST = None
def xs(t): return float(np.interp(t, XST[0], XST[1]))
def zs(t): return float(np.interp(t, ZST[0], ZST[1]))
def blend(f, g, t, t0, t1):
    u = min(1.0, max(0.0, (t - t0) / (t1 - t0))); return (1 - u) * f(t) + u * g


def write_hops(ball, hits, t_next, kinds, pitch_of, ev_of):
    for k, h in enumerate(hits):
        ball.hit(h['t'], h['p'], h['vins'][0], h['vout'], kinds(k), pitch_of(k), ev=ev_of(k))
        t1 = hits[k + 1]['t'] if k + 1 < len(hits) else t_next
        ball.fly(h['t'], t1, h['p'], h['vout'])


LONG = 2.0                                          # a gap at least this long: the ball waits in a glass cup
TF_C = 0.4                                          # the cup opens and the ball falls straight down to the next note
CUPS = []


RAMPS = []
TH_CAND = [math.radians(a) for a in (10, 13, 16, 19, 22, 25, 28, 31)]


def ramp_from(t_a, p_a, vin, times4, t_next, xt, zt, R, need_v=0.0):
    """a run of quick notes: the ball lands on a little inclined xylophone ramp (keeps its velocity along the ramp, the
    normal part is absorbed), rolls down over the bars (rolling: a = 5/7 g sin th) and leaves the low end in free flight.
    Returns the best (by the next note's target) dict(th, u, n, v0, acc, s_keys, t_end, p_end, v_end, p_next, v_next)."""
    best = None
    for th in TH_CAND:
        u = np.array([math.cos(th), 0, -math.sin(th)]); n = np.array([math.sin(th), 0, math.cos(th)])
        v0 = float(vin @ u)
        if v0 < 0.25: continue
        acc = 5 / 7 * G * math.sin(th)
        tau = np.array(times4) - t_a
        s_k = v0 * tau + 0.5 * acc * tau ** 2
        t_end = times4[-1] + 0.05; te = t_end - t_a
        p_end = p_a + u * (v0 * te + 0.5 * acc * te * te); v_end = u * (v0 + acc * te)
        dt = t_next - t_end
        p_n = p_end + v_end * dt + np.array([0, 0, -0.5 * G * dt * dt]); v_n = v_end + np.array([0, 0, -G * dt])
        if v_n[2] > -0.6: continue
        cost = (p_n[0] - xt(t_next)) ** 2 + 2 * (p_n[2] - zt(t_next)) ** 2 + 0.02 * np.linalg.norm(v_n) ** 2 + 50 * max(0.0, need_v - np.linalg.norm(v_n)) ** 2
        if best is None or cost < best['cost']:
            best = dict(th=th, u=u, n=n, v0=v0, acc=acc, s_keys=s_k.tolist(), t_a=t_a, t_end=t_end, p_a=p_a, p_end=p_end, v_end=v_end,
                        p_next=p_n, v_next=v_n, cost=cost, vin=vin)
    assert best is not None, ('no ramp', t_a)
    return best


def plan_voice(TIMES, p0, vin0, xt, zt, yt, R, name, ramps=False, long_gap=LONG, cup_v=lambda t, gap: 0.0, **kw):
    """segments: hops (planned bounces) | ramp (run of quick notes rolled over a little xylophone ramp) |
    cup (a long gap: the ball lands in a glass cup - that landing is the note - which rides along at cup_v and opens
    TF_C before the next note) | catch (after a ramp that is followed by a long gap: a silent catch cup)"""
    n = len(TIMES); i = 0; p = np.array(p0, float); vins = [np.array(vin0, float)]; out = []
    run_start = set()
    if ramps:
        for a in range(n - 3):
            if all(TIMES[a + j + 1] - TIMES[a + j] < 0.15 for j in range(3)) and (a == 0 or TIMES[a] - TIMES[a - 1] >= 0.15): run_start.add(a)

    def add_cup(kind, c, t_land, pc, vc, t_next):
        gap = t_next - t_land; vx = cup_v(t_land, gap)
        tfc = 0.75 if ((c + 1) in run_start and gap > 3.0) else TF_C   # before a run after the long wait: a longer drop
        if vx:                                                # a moving cup on its ropeway: it carries the ball to the next note
            T_ = t_next - t_land; extra = 0.35 if (c + 1) in run_start else 0.0
            vcup = np.array([(xt(t_next) - pc[0]) / T_, 0.0, (zt(t_next) + extra + 0.5 * G * tfc ** 2 - pc[2]) / T_])
        else: vcup = np.zeros(3)
        t_rel = t_next - tfc
        p_rel = pc + vcup * (t_rel - t_land)
        out.append((kind, c, t_land, pc, vc, vcup, t_rel, p_rel, t_next))
        CUPS.append(dict(who=name, kind=kind, t=float(t_land), p=pc.tolist(), t_rel=float(t_rel), p_rel=p_rel.tolist(), v=vcup.tolist()))
        SFX.append(['cup', float(t_land)]); SFX.append(['cup_open', float(t_rel - 0.12)])
        return p_rel + vcup * tfc + np.array([0, 0, -0.5 * G * tfc ** 2]), vcup + np.array([0, 0, -G * tfc])

    while True:
        c = i
        while c < n - 1 and TIMES[c + 1] - TIMES[c] < long_gap and c not in run_start: c += 1
        if c > i:
            kw2 = dict(kw)
            if c < n - 1: kw2.pop('final', None)
            if c in run_start: kw2['dn_last'] = 0.12
            hits, pc, vc = plan_retry(TIMES[i:c + 1], p, vins, xt, zt, yt, R=R, **kw2)
            out.append(('hops', i, hits, TIMES[c]))
        else:
            pc, vc = p, vins[0]
        if c == n - 1:
            out.append(('end', c, pc, vc)); return out
        if c in run_start:
            last_long = c + 4 < n and TIMES[c + 4] - TIMES[c + 3] >= min(long_gap, 0.5)
            need_v = 0.5 * G * (TIMES[c + 5] - TIMES[c + 4]) if (not last_long and c + 5 < n) else 0.0
            rp = ramp_from(TIMES[c], pc, vc, TIMES[c:c + 4], TIMES[c + 4] if not last_long else TIMES[c + 3] + 0.25, xt, zt, R, need_v)
            out.append(('ramp', c, rp)); RAMPS.append(dict(who=name, i=c, **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in rp.items()}))
            if last_long:
                pq, vq = add_cup('catch', c + 3, TIMES[c + 3] + 0.25, rp['p_next'], rp['v_next'], TIMES[c + 4])
                p = pq; vins = [vq]
            else:
                p = rp['p_next']; vins = [rp['v_next']]
            i = c + 4
            continue
        pq, vq = add_cup('cup', c, TIMES[c], pc, vc, TIMES[c + 1])
        p = pq; vins = [vq]
        i = c + 1


def write_voice(ball, segs, TIMES, kind_of, pitch_of):
    for sg in segs:
        if sg[0] == 'hops':
            _, i0, hits, t_next = sg
            write_hops(ball, hits, t_next, lambda k: kind_of(i0 + k), lambda k: pitch_of(i0 + k), lambda k: i0 + k)
        elif sg[0] in ('cup', 'catch'):
            _, c, t_land, pc, vc, vcup, t_rel, p_rel, t_next = sg
            ball.hit(t_land, pc, vc, vcup, sg[0], pitch_of(c) if sg[0] == 'cup' else None, ev=c if sg[0] == 'cup' else None)
            if np.linalg.norm(vcup) > 0:
                ts = np.arange(t_land, t_rel + 1e-9, 1 / 240.0); ts[-1] = t_rel
                ball.path(ts, np.array([pc + vcup * (t - t_land) for t in ts]))
            else: ball.rest(t_land, t_rel, pc)
            ball.fly(t_rel, t_next, p_rel, vcup)
        elif sg[0] == 'ramp':
            _, c, rp = sg
            u = rp['u']; v_land = u * rp['v0']
            ball.hit(TIMES[c], rp['p_a'], rp['vin'], v_land, 'rampland', pitch_of(c), ev=c)
            for j in range(1, 4):
                pj = rp['p_a'] + u * rp['s_keys'][j]; vj = u * (rp['v0'] + rp['acc'] * (TIMES[c + j] - TIMES[c]))
                ball.hits.append(dict(t=float(TIMES[c + j]), p=pj.tolist(), vin=vj.tolist(), vout=vj.tolist(), n=rp['n'].tolist(), e=1.0, kind='roll',
                                      pitch=pitch_of(c + j), ev=c + j))
            ts = np.arange(TIMES[c], rp['t_end'] + 1e-9, 1 / 240.0); ts[-1] = rp['t_end']
            ball.path(ts, np.array([rp['p_a'] + u * (rp['v0'] * (t - TIMES[c]) + 0.5 * rp['acc'] * (t - TIMES[c]) ** 2) for t in ts]))
            nxt = [x for x in segs if x[0] == 'catch' and x[1] == c + 3]
            ball.fly(rp['t_end'], nxt[0][2] if nxt else TIMES[c + 4], rp['p_end'], rp['v_end'])
    return segs[-1]


def pitch_top(pp): return pp if not isinstance(pp, list) else pp


# =================================================================== the fairy (celesta melody)
FAIRY = Ball('fairy', RF)
XF0 = 0.45
TDROP_F = 0.62
TF = np.array([f[0] for f in FA[:-1]])            # the final note (the star bell) is the lift's throw, below
PF = [f[1] for f in FA[:-1]]
def pitch_at(t): return PF[int(np.argmin(np.abs(TF - t)))]
def zf_t(t): return zs0(t) + ZF + KP * (pitch_at(t) - 83)
z_first = zf_t(TF[0])
p_moon = np.array([xs0(TF[0]) + XF0, YF, z_first + 0.5 * G * TDROP_F ** 2])
# the crescent moon carries her across the sky during the introduction (smooth start and stop), then opens
T_MOON1 = TF[0] - TDROP_F - 0.45
p_moon0 = p_moon + np.array([-(xs0(TF[0]) - xs0(0.0)) + 1.3, 0, 0])
ts_ = np.arange(0.0, T_MOON1 + 1e-9, 1 / 240.0); ts_[-1] = T_MOON1
FAIRY.path(ts_, np.array([p_moon0 + (p_moon - p_moon0) * (lambda u: u * u * (3 - 2 * u))(t / T_MOON1) for t in ts_]))
FAIRY.rest(T_MOON1, TF[0] - TDROP_F, p_moon); FAIRY.fly(TF[0] - TDROP_F, TF[0], p_moon, [0, 0, 0])
EV['moon'] = dict(p=p_moon.tolist(), p0=p_moon0.tolist(), t_stop=float(T_MOON1), t_release=float(TF[0] - TDROP_F))
SFX.append(['release', float(TF[0] - TDROP_F)])
FK = dict(stay=1e9, nkeep=12, ext=(0.2, 0.2, 0.05), vxlim=(0.55, 3.2))
# the cup of the lift (bar 52, B5) sits a little lower than the stage line so the lift has its run up the tree
segF = plan_voice(TF, p_moon - [0, 0, 0.5 * G * TDROP_F ** 2], [0, 0, -G * TDROP_F], lambda t: xs0(t) + XF0, zf_t, lambda t: YF, RF, 'fairy', **FK)
_, _, pF, vF = write_voice(FAIRY, segF, TF, lambda k: 'key', lambda k: PF[k])
# stage height = the fairy's contact heights minus the pitch contour, smoothed, never rising; stage x = its contact x
_hs = sorted([h for h in FAIRY.hits if h['ev'] is not None], key=lambda h: h['t'])
_t = np.array([h['t'] for h in _hs] + [TF[-1]])
_z = np.array([h['p'][2] for h in _hs] + [pF[2]]) - ZF - KP * (np.array([PF[h['ev']] for h in _hs] + [PF[-1]]) - 83)
_tt = np.arange(0.0, FINAL + 31, 0.1); _zz = np.interp(_tt, _t, _z)
_k = np.ones(41) / 41; _zz = np.convolve(np.pad(_zz, 20, mode='edge'), _k, 'valid')
ZST = (_tt, np.minimum.accumulate(_zz))
_x = np.array([h['p'][0] for h in _hs] + [pF[0]]) - XF0
_xx = np.interp(_tt, _t, _x) + np.where(_tt < _t[0], V * (_tt - _t[0]), 0.0)
_k = np.ones(31) / 31
for _ in range(3): _xx = np.convolve(np.pad(_xx, 15, mode='edge'), _k, 'valid')
_xx[_tt < _t[0]] = (_x[0] + V * (_tt - _t[0]))[_tt < _t[0]]
XST = (_tt, np.maximum.accumulate(_xx))
print('fairy: hits', len(FAIRY.hits), 'x', round(_x.min(), 1), round(_x.max(), 1), 'z', round(_z.min(), 2), round(_z.max(), 2))

# =================================================================== mint (chords) and cocoa (bass)
t_lever = bar_t('20s', 'A2')
MINT = Ball('mint', RM); COCOA = Ball('cocoa', RC)
XM0, XC0 = -0.35, -0.95
TDROP = 0.6
TM = np.array([m[0] for m in MI]); TC = np.array([c[0] for c in CO])
# their final notes (the chord at FINAL) are cups at the foot of the tree: plan up to the last note before it
XT_FIN = pF[0]                                     # the tree stands behind the fairy's lift cup
for ball, TT, X0, ZL, Y in ((MINT, TM, XM0, ZM, YM), (COCOA, TC, XC0, 0.0, YC)):
    p_ = np.array([xs(TT[0]) + X0, Y, zs(TT[0]) + ZL + 0.5 * G * TDROP ** 2])
    ball.rest(0.0, TT[0] - TDROP, p_); ball.fly(TT[0] - TDROP, TT[0], p_, [0, 0, 0])
    EV[f'{ball.name}_start'] = dict(p=p_.tolist(), t_release=float(TT[0] - TDROP))
    SFX.append(['release', float(TT[0] - TDROP)])
FINX = {'mint': XT_FIN - 0.9, 'cocoa': XT_FIN - 1.9}
segM = plan_voice(TM, (xs(TM[0]) + XM0, YM, zs(TM[0]) + ZM), [0, 0, -G * TDROP],
                  lambda t: blend(lambda s: xs(s) + XM0, FINX['mint'], t, FINAL - 3.0, FINAL), lambda t: zs(t) + ZM, lambda t: YM, RM, 'mint',
                  vxlim=(0.3, 3.0))
_, _, pM, vM = write_voice(MINT, segM, TM, lambda k: 'chord', lambda k: MI[k][1])
segC = plan_voice(TC, (xs(TC[0]) + XC0, YC, zs(TC[0])), [0, 0, -G * TDROP],
                  lambda t: blend(lambda s: xs(s) + XC0, FINX['cocoa'], t, FINAL - 3.0, FINAL), lambda t: zs(t), lambda t: YC, RC, 'cocoa',
                  ramps=True, long_gap=1.2, cup_v=lambda t, gap: V if (gap < 3.0 and abs(t - t_lever) > 1e-3) else 0.0, vxlim=(0.3, 3.0))
_, _, pC, vC = write_voice(COCOA, segC, TC, lambda k: 'drum', lambda k: CO[k][1])
# the final chord: both land in their cups (= the last keys) and stay
for ball, pe, ve, L_ in ((MINT, pM, vM, MI), (COCOA, pC, vC, CO)):
    ball.hits[-1]['kind'] = ball.hits[-1]['kind']
    # replace the last bounce (planned at FINAL) by a landing in the final cup
    ball.hit(FINAL, pe, ve, [0, 0, 0], 'cupfinal', L_[-1][1], ev=len(L_) - 1)
    ball.rest(FINAL, FINAL + 30, pe)
EV['final_cups'] = dict(mint=pM.tolist(), cocoa=pC.tolist())
# the lever drum (cocoa, bar 20 beat 1 = G3): knocks the sugar bowl latch
lv = [h for h in COCOA.hits if abs(h['t'] - t_lever) < 1e-6][0]; lv['kind'] = 'lever'
EV['lever'] = dict(t=float(t_lever), p=lv['p'], n=lv['n'])
T_LATCH = t_lever + 0.14; T_TIP0, T_TIP1 = T_LATCH + 0.05, T_LATCH + 0.45
SFX += [['latch', float(T_LATCH)], ['bowl_tip', float(T_TIP0)]]

# =================================================================== the crystal harp + cadenza pearls
CAD = [r for r in RA if r[4] in ('A2', 'cad')]
PRE = [r for r in RA if r[4] in ('presto', 'final')]
cup_f = [c for c in CUPS if c['who'] == 'fairy'][0]
XC_ = cup_f['p'][0] - XF0                           # stage x where the three wait
pitches = sorted(set(r[1] for r in CAD))
HSP = 0.34
HX0 = XC_ - 1.55
ZH0 = zs(cup_f['t']) + 2.35
YH = -1.7
TFALL = 0.6
HARP = {p: dict(x=HX0 + HSP * j, z=ZH0 + 0.055 * (p - pitches[0])) for j, p in enumerate(pitches)}
TILT_H = math.radians(28.0); NH = np.array([0, -math.sin(TILT_H), math.cos(TILT_H)])
ZTROUGH = ZH0 - 1.25; YTROUGH = YH - 1.2
PEARLS = []
E_PEARL = 0.5


def drop_pearl(name, t, p, top, n, e=E_PEARL, z_land=None, tfall=TFALL, land_kind='trough', r=RP):
    """a pearl released from rest straight above its key, falls tfall, bounces (e along n), lands at z_land (absorbed)"""
    b = Ball(name, r)
    c = np.array(top) + np.array(n) * r                # ball centre at contact
    rel = c + np.array([0, 0, 0.5 * G * tfall ** 2])
    b.rest(t - tfall - 0.25, t - tfall, rel); b.fly(t - tfall, t, rel, [0, 0, 0])
    vin = np.array([0, 0, -G * tfall]); nn = np.array(n, float); vout = vin - (1 + e) * (vin @ nn) * nn
    b.hit(t, c, vin, vout, 'pearl', p, ev=None)
    a_, b_, c_ = -0.5 * G, vout[2], c[2] - z_land
    tf = (-b_ - math.sqrt(b_ * b_ - 4 * a_ * c_)) / (2 * a_)
    b.fly(t, t + tf, c, vout)
    pl = c + vout * tf + np.array([0, 0, -0.5 * G * tf * tf]); vl = vout + np.array([0, 0, -G * tf])
    b.hits.append(dict(t=float(t + tf), p=pl.tolist(), vin=vl.tolist(), vout=[0, 0, 0], n=[0, 0, 1], e=0.0, kind=land_kind, pitch=None, ev=None))
    b.rest(t + tf, FINAL + 30, pl)
    b.t_release = t - tfall; b.t_land = t + tf
    return b


YA_H = np.array([0, math.cos(TILT_H), math.sin(TILT_H)])     # along the key's depth
RPC = 0.09                                                        # cadenza pearls (sugar sprinkles) are a little smaller
SLOTS = [(sd, xo) for sd in (-1, 1) for xo in (-0.095, 0.095)]      # front/back half x left/right of each key
_used = {}
for j, (t, p, v, d, sec, k, o) in enumerate(CAD):
    hk = HARP[p]; prev = [(tq, sq) for tq, sq in _used.get(p, []) if t - tq < 0.6]
    def cost(sl):
        c_ = 0.0
        for tq, sq in prev:
            dt_ = t - tq
            if sq == sl: c_ += 100 * max(0, 0.6 - dt_)
            elif sq[0] == sl[0]: c_ += 60 * max(0, 0.32 - dt_)
            elif sq[0] == 1 and sl[0] == -1: c_ += (40 if sq[1] == sl[1] else 15) * max(0, 0.35 - dt_)
        return c_ + 0.01 * (sl[0] > 0)
    sl = min(SLOTS, key=cost); _used.setdefault(p, []).append((t, sl))
    top = np.array([hk['x'] + sl[1], YH, hk['z']]) + YA_H * 0.145 * sl[0]
    b = drop_pearl(f'pc{j:03d}', t, p, top, NH, z_land=ZTROUGH + RPC, r=RPC)
    PEARLS.append(b)
EV['harp'] = dict(keys={str(p): v for p, v in HARP.items()}, y=YH, n=NH.tolist(), z_trough=ZTROUGH, y_trough=YTROUGH, tfall=TFALL,
                  t_up=float(T_TIP0 - 0.6), t_down=float(bar_t(5, 'A3') - 0.3), xc=XC_, sieve_z=ZH0 + 0.5 * G * TFALL ** 2 + RP)
print('harp', len(pitches), 'keys, x', round(HX0, 2), '..', round(HX0 + HSP * (len(pitches) - 1), 2), 'pearls', len(CAD))

# =================================================================== the lift and the tree
cup_l = pF.copy()                                   # the fairy lands in the lift cup on bar 52 (B5)
FAIRY.hit(TF[-1], cup_l, vF, [0, 0, 0], 'liftcup', PF[-1], ev=len(PF) - 1)
YT_, R0T = YF + 2.5, 3.2                           # tree axis depth, foliage radius at the bottom
ZT0 = zs(TF[-1]) - 0.3                              # the tree stands on the snow
T_FL = 0.42; VLIFT = 4.0
RISE_FL = None
th_rail = math.radians(25.0)                       # the lift rail leans back against the tree (towards +y)
ud = np.array([0, math.sin(th_rail), math.cos(th_rail)])
vl = VLIFT * ud
STAR_Y = YT_
y_launch = STAR_Y - vl[1] * T_FL
L_rail = (y_launch - cup_l[1]) / ud[1]
p_launch = cup_l + ud * L_rail
p_bell = p_launch + vl * T_FL + np.array([0, 0, -0.5 * G * T_FL ** 2])
vb_in = vl + np.array([0, 0, -G * T_FL])
assert vb_in[2] > 0.6, vb_in
HT = p_bell[2] + RF + 0.25 - ZT0                    # the tree's tip is just above the star bell
T_LAUNCH = FINAL - T_FL
T_LIFT0 = TF[-1] + 0.35
ACC = VLIFT ** 2 / (2 * L_rail)                     # constant acceleration from rest over the whole rail
T_ACC = VLIFT / ACC
T_LIFT0 = T_LAUNCH - T_ACC
assert T_LIFT0 > TF[-1] + 0.3, (T_LIFT0, TF[-1], L_rail)
FAIRY.rest(TF[-1], T_LIFT0, cup_l)
ts_ = np.arange(T_LIFT0, T_LAUNCH + 1e-9, 1 / 240.0); ts_[-1] = T_LAUNCH
FAIRY.path(ts_, np.array([cup_l + ud * (0.5 * ACC * (t - T_LIFT0) ** 2) for t in ts_]))
FAIRY.fly(T_LAUNCH, FINAL, p_launch, vl)
# it strikes the inside wall of the bell and drops back into the lift cup that stopped at the top
for tau in np.arange(0.3, 0.9, 0.01):
    vb_out = (p_launch - p_bell) / tau + np.array([0, 0, 0.5 * G * tau])
    e_b, n_b = bounce_e(vb_in, vb_out)
    if 0.3 <= e_b <= 0.92 and np.linalg.norm(vb_out) < np.linalg.norm(vb_in): break
else: raise RuntimeError('no bell bounce')
FAIRY.hit(FINAL, p_bell, vb_in, vb_out, 'starbell', FA[-1][1], ev=len(FA) - 1)
p_back = p_launch.copy(); tf2 = tau
FAIRY.fly(FINAL, FINAL + tf2, p_bell, vb_out)
FAIRY.hits.append(dict(t=float(FINAL + tf2), p=p_back.tolist(), vin=(vb_out + np.array([0, 0, -G * tf2])).tolist(), vout=[0, 0, 0], n=[0, 0, 1],
                       e=0.0, kind='cupland', pitch=None, ev=None))
FAIRY.rest(FINAL + tf2, FINAL + 30, p_back)
print('star bell bounce: e', round(e_b, 3), 'normal', n_b.round(2), 'back in the cup after', round(tau, 2), 's')
EV['lift'] = dict(cup=cup_l.tolist(), ud=ud.tolist(), L=float(L_rail), acc=float(ACC), t0=float(T_LIFT0), t_launch=float(T_LAUNCH), v=VLIFT,
                  p_launch=p_launch.tolist(), back=p_back.tolist(), t_back=float(FINAL + tf2))
EV['star'] = dict(p=(p_bell + np.array([0, 0, RF])).tolist(), t=float(FINAL))
EV['tree'] = dict(x=float(cup_l[0]), y=YT_, z0=ZT0, r0=R0T, h=HT)
SFX += [['lift_start', float(T_LIFT0)], ['lift_stop', float(T_LAUNCH)]]
print('lift: rail', round(L_rail, 2), 'm, acc', round(ACC, 2), 'start', round(T_LIFT0, 2), 'launch', round(T_LAUNCH, 2), 'bell z', round(p_bell[2], 2),
      'tree top z', round(ZT0 + HT, 2))

# ---- Presto pearls: ornaments on the front of the tree; they drop from the sugar cloud above the tree
rng = np.random.default_rng(11)
# sugar clouds float in three rings around the tree; each pearl drops 1.2-2.4 m from the ring right above its ornament
BANDS = [ZT0 + f * HT for f in (0.36, 0.58, 0.8)]
ORN = []
def cone_r(z): return R0T * max(0.0, 1 - (z - ZT0) / HT)
FKEYS = [np.array(h['p']) - RF * np.array(h['n']) for h in FAIRY.hits if h['kind'] == 'key']
RAIL = (cup_l, p_launch, p_bell)
def seg_dist(p, a, b):
    ab = b - a; u = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1); return np.linalg.norm(p - (a + u * ab))


def pearl_track(top, n, z_rel, e=0.45):
    c = top + n * RP; tfall = math.sqrt(2 * (z_rel - c[2]) / G)
    vin = np.array([0, 0, -G * tfall]); vout = vin - (1 + e) * (vin @ n) * n
    pts = [c + np.array([0, 0, z_rel - c[2] - 0.5 * G * u * u]) * 0 + np.array([0, 0, (z_rel - c[2]) * (1 - (u / tfall) ** 2)]) for u in np.linspace(0, tfall, 14)]
    a_, b_, c_ = -0.5 * G, vout[2], c[2] - (ZT0 + RP)
    tl = (-b_ - math.sqrt(b_ * b_ - 4 * a_ * c_)) / (2 * a_)
    pts += [c + vout * u + np.array([0, 0, -0.5 * G * u * u]) for u in np.linspace(0.04, min(tl, 0.45), 10)]   # it bursts into sparkles 0.45 s after
    return np.array(pts), tfall


TRACKS = []
TGRID = np.arange(T_LIFT0 - 4.0, FINAL + 1.0, 0.02)
def pearl_tpos(top, n, z_rel, t_hit, e=0.45):
    c = top + n * RP; tfall = math.sqrt(2 * (z_rel - c[2]) / G); vin = np.array([0, 0, -G * tfall]); vout = vin - (1 + e) * (vin @ n) * n
    P_ = np.full((len(TGRID), 3), np.nan); t0 = t_hit - tfall - 0.25
    for i_, t in enumerate(TGRID):
        if t < t0 or t > t_hit + 0.45: continue
        if t < t_hit - tfall: P_[i_] = c + np.array([0, 0, 0.5 * G * tfall ** 2])
        elif t <= t_hit: u = t_hit - t; P_[i_] = c + np.array([0, 0, 0.5 * G * (tfall ** 2 - (tfall - u) ** 2)]) * 0 + np.array([0, 0, -vin[2] * u - 0.5 * G * u * u])
        else: u = t - t_hit; P_[i_] = c + vout * u + np.array([0, 0, -0.5 * G * u * u])
    return P_
MCP = []
for ball in (MINT, COCOA):
    for t in np.arange(T_LIFT0 - 2.5, FINAL + 0.6, 0.04):
        p_ = ball.at(t); MCP += [p_, p_ - np.array([0, 0, ball.r + 0.12])]
MCP = np.array(MCP)
for j, (t, p, v, d, sec, k, o) in enumerate(PRE):
    fin = sec == 'final'
    best = None
    for tries in range(30000):
        if fin:                                    # the last three: big ornaments near the top
            band = int(rng.choice([1, 2])); zz = BANDS[band] - rng.uniform(1.2, 2.4); az = math.radians(rng.uniform(-86, 86))
        else:
            band = int(rng.choice(3, p=[0.42, 0.36, 0.22])); zz = BANDS[band] - rng.uniform(1.25, 2.4); az = math.radians(rng.uniform(-86, 86))
        if zz < ZT0 + 0.6: continue
        rr = cone_r(zz) + 0.06
        pos = np.array([cup_l[0] + rr * math.sin(az), YT_ - rr * math.cos(az), zz])
        radial = np.array([math.sin(az), -math.cos(az), 0.0])
        nrm = 0.55 * radial + np.array([0, 0, 0.84]); nrm /= np.linalg.norm(nrm)
        trk, tf_ = pearl_track(pos, nrm, BANDS[band])
        r_o = 0.25 if fin else 0.18
        ctr = pos - nrm * r_o
        if any(np.linalg.norm(ctr - (q['p'] - q['n'] * q['r'])) < r_o + q['r'] + 0.04 for q in ORN): continue
        bad = False
        for q in ORN:                              # its pearl must miss every other ornament, and their pearls must miss it
            qc = q['p'] - q['n'] * q['r']
            if (np.linalg.norm(trk[1:] - qc, axis=1) < q['r'] + RP + 0.05).any() or (np.linalg.norm(q['trk'][1:] - ctr, axis=1) < r_o + RP + 0.05).any(): bad = True; break
        if bad: continue
        if any((np.linalg.norm(trk - fk, axis=1) < 0.36).any() for fk in FKEYS): continue
        if min(seg_dist(x, RAIL[0], RAIL[1]) for x in trk) < 0.75 or min(seg_dist(x, RAIL[1], RAIL[2]) for x in trk) < 0.75: continue
        if seg_dist(ctr, RAIL[0], RAIL[1]) < 0.9: continue
        if (np.linalg.norm(trk[:, None, :] - MCP[None, :, :], axis=2) < 0.7).any(): continue
        tp_ = pearl_tpos(pos, nrm, BANDS[band], t)
        if any(np.nanmin(np.r_[np.linalg.norm(tp_ - q['tp'], axis=1), 9]) < 2 * RP + 0.04 for q in ORN): continue
        best = (pos, nrm, band, trk, r_o, tp_); break
    assert best is not None, ('no ornament place', j)
    pos, nrm, band, trk, r_o, tp_ = best
    ORN.append(dict(p=pos, n=nrm, pitch=p, t=t, fin=fin, band=band, trk=trk, r=r_o, tp=tp_))
for j, o_ in enumerate(ORN):
    top = o_['p']
    tfall = math.sqrt(2 * (BANDS[o_['band']] - (top[2] + o_['n'][2] * RP)) / G)
    b = drop_pearl(f'pt{j:03d}', o_['t'], o_['pitch'], top, o_['n'], e=0.45, z_land=ZT0 + RP, tfall=tfall, land_kind='snow')
    PEARLS.append(b)
EV['ornaments'] = [dict(p=o['p'].tolist(), n=o['n'].tolist(), pitch=o['pitch'], t=o['t'], fin=o['fin'], r=o['r'], band=o['band']) for o in ORN]
EV['cloud'] = dict(bands=BANDS, t_open=float(min(o['t'] - math.sqrt(2 * (BANDS[o['band']] - o['p'][2]) / G) for o in ORN) - 0.3))
EV['cups'] = CUPS
cupt = [c['t'] for c in CUPS if c['who'] == 'fairy'] + [TF[-1]]
EV['fairy_temp_keys'] = [h['t'] for h in FAIRY.hits if h['kind'] == 'key' and any(0 < ct - h['t'] < 1.0 for ct in cupt)]
# a pearl shows up (in the sieve / cloud) at most 0.25 s before it drops, never while the previous pearl is still there
SHOW = {}; _rel = []
for b in sorted(PEARLS, key=lambda b: b.t_release):
    p0_ = np.array(b.seg[0]['p']); t_show = b.t_release - 0.25
    for tq, pq in _rel:                          # wait until a pearl released from (nearly) the same hole has fallen clear
        if np.linalg.norm(pq - p0_) < 0.32 and tq < b.t_release: t_show = max(t_show, tq + 0.27)
    SHOW[b.name] = float(min(t_show, b.t_release - 0.02)); _rel.append((b.t_release, p0_))
EV['pearl_show'] = SHOW
EV['ramps'] = RAMPS
EV['stage'] = [ZST[0][::5].tolist(), ZST[1][::5].tolist()]
EV['stage_x'] = [XST[0][::5].tolist(), XST[1][::5].tolist()]
EV['bowl'] = dict(t_latch=float(T_LATCH), t_tip0=float(T_TIP0), t_tip1=float(T_TIP1))
for b in PEARLS: SFX.append(['pearl_bowl', float(b.t_land)])
BALLS = {'fairy': FAIRY, 'mint': MINT, 'cocoa': COCOA}
for b in PEARLS: BALLS[b.name] = b


def check():
    info = {}
    hs = [(n, h) for n, b in BALLS.items() for h in b.hits if h['kind'] in ('key', 'chord', 'drum', 'lever', 'pearl')]
    info['e_range'] = (round(min(h['e'] for n, h in hs), 3), round(max(h['e'] for n, h in hs), 3))
    allh = [h for b in BALLS.values() for h in b.hits if np.linalg.norm(h['vout']) > 0]
    info['speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in allh), 3)
    info['tilt_max_deg'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for n, h in hs), 1)
    for n in ('fairy', 'mint', 'cocoa'):
        b = BALLS[n]; zz = [h['p'][2] for h in b.hits]; xx = [h['p'][0] for h in b.hits]
        info[n] = dict(hits=len(b.hits), z=(round(min(zz), 2), round(max(zz), 2)), x=(round(min(xx), 1), round(max(xx), 1)))
    info['cups'] = [(c['who'], round(c['t'], 2), round(c['t_rel'], 2)) for c in CUPS]
    info['pearls'] = len(PEARLS)
    return info


if __name__ == '__main__':
    for k, v in check().items(): print(k, v)
    out = dict(G=G, R=dict(fairy=RF, mint=RM, cocoa=RC, pearl=RP), lanes=dict(fairy=YF, mint=YM, cocoa=YC), zoff=dict(fairy=ZF, mint=ZM), kp=KP, v=V,
               balls={n: dict(r=b.r, seg=b.seg, hits=b.hits) for n, b in BALLS.items()}, ev=EV, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
