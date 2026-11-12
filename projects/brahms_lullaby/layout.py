"""Brahms' Lullaby - 'Sleepy marbles / How many sheep until the baby sleeps?'. Physics + story planner.
Run after arrange.py; writes layout.json (read by scene.py, render_audio.py, verify.py).

Cast (one role per voice of the score):
  star    (voice)            the glowing star marble. Every note = it lands on the chime bar of a hanging star ornament
                             (the star lights up and stays lit). The heights of the chimes follow the pitch, so the lit
                             stars draw the melody across the nursery sky.
  twins   (piano RH)         two pastel marbles (pink + blue) moving as a pair: a dyad = each lands on its own toy-block
                             key, a single note = both land on one wide key. The blocks pop up just before and vanish.
  sheep   (piano LH)         one woolly sheep per bar: it drops out of its cloud on beat 1 onto the big Eb2 drum, hops to the
                             beat-2 drum and the beat-3 drum (their pitch = the score), then jumps the fence and lands in its
                             pillow and falls asleep (the counter counts it).  Bar 18: the 25th sheep is a GIANT; it lands on
                             the far end of a see-saw (the LH chord) and throws the star and the twins up into the baby mobile
                             (final chord: moon bell Eb5, star bells G5 Bb5). Then the room light goes out -> starry sky.
Physics: one constant gravity G; between contacts pure parabolas; bounces are impulses along the contact normal with
restitution emin <= e <= emax (never adds speed); pillows / cups absorb (e = 0); the only energy source is the falling giant
sheep on the see-saw lever."""
import json, os, math
import numpy as np
from phys import Ball, seg_at, bounce_e, G, EMAX, EMIN, TILT
import phys, pickle, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
_CF = '/tmp/brahms_lullaby_plan_cache.pkl'
_C = pickle.load(open(_CF, 'rb')) if os.path.exists(_CF) else {}


def plan(times, p0, vins, xt, zt, yt, **kw):
    """phys.plan with a disk cache (key = all inputs, target functions sampled at the note times)"""
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


RS, RT, RH, RG = 0.22, 0.19, 0.27, 0.6           # star, twins, sheep, giant sheep
DXT, DYT = 0.22, 0.2                              # twin offsets from the pair centre
YS, YT, YH = 2.5, 1.25, 0.0                       # lanes (depth): star at the back, twins, sheep in front
ZS, ZT = 4.5, 2.55                                # lane heights above the sheep drums
KP = 0.11                                         # star chime height per semitone (the melody contour)
V = 1.6                                           # the stage (and the camera) moves right at V
E = json.load(open(f'{HERE}/events.json'))
LEAD = json.load(open(f'{HERE}/lead.json')); DUO = json.load(open(f'{HERE}/duo.json')); BASS = json.load(open(f'{HERE}/bass.json'))
T0, Q, BAR, FINAL, T_LAUNCH, T_END = E['t0'], E['q'], E['bar'], E['final'], E['t_launch'], E['t_end']
GRID = E['grid']
TL = np.array([l[0] for l in LEAD]); TD = np.array([l[0] for l in DUO])
SFX = []; EV = {}


SLOPE = -0.13                                     # the dream land drifts downhill (the long notes cost the star height)
def zs0(t): return SLOPE * (t - T0)
def xs0(t): return V * (t - T0)
XST = None                                        # stage x (set from the star's real path, below)
def xs(t): return float(np.interp(t, XST[0], XST[1]))
ZST = None                                        # stage height (set from the star's real path, below)
def zs(t): return float(np.interp(t, ZST[0], ZST[1]))


def pitch_at(t):
    k = int(np.argmin(np.abs(TL - t))); return LEAD[k][1]


def blend(f, g, t, t0, t1):
    u = min(1.0, max(0.0, (t - t0) / (t1 - t0))); return (1 - u) * f(t) + u * g


def write_hops(ball, hits, t_next, kinds, pitch_of, ev_of, off=np.zeros(3), vin_pick=0):
    for k, h in enumerate(hits):
        vin = h['vins'][vin_pick if len(h['vins']) > 1 else 0]
        ball.hit(h['t'], h['p'] + off, vin, h['vout'], kinds(k), pitch_of(k), ev=ev_of(k))
        t1 = hits[k + 1]['t'] if k + 1 < len(hits) else t_next
        ball.fly(h['t'], t1, h['p'] + off, h['vout'])


def free_to(ball, t0, p0, v0, t1):
    ball.fly(t0, t1, p0, v0); u = t1 - t0
    return np.array(p0) + np.array(v0) * u + np.array([0, 0, -0.5 * G * u * u]), np.array(v0) + np.array([0, 0, -G * u])


LONG = 1.2                                         # a note at least this long: the ball rests in a moon cradle
TF_C = 0.4                                        # cradle: it opens (two halves) and the ball falls straight down to the next note
CRADLES = []


def plan_voice(ball_writer, TIMES, p0, vin0, xt, zt, yt, R, name, **kw):
    """plan a voice in segments split at long notes. Returns list of events:
    ('hops', i0, hits, t_next) | ('cradle', i, p, vin, t_rel, p_lip, v_rel, t_next) | ('end', i, p, vin)"""
    n = len(TIMES); i = 0; p = np.array(p0, float); vins = [np.array(vin0, float)]; out = []
    while True:
        c = i
        while c < n - 1 and TIMES[c + 1] - TIMES[c] < LONG: c += 1
        if c > i:
            kw2 = dict(kw)
            if c < n - 1: kw2.pop('final', None)
            hits, pc, vc = plan_retry(TIMES[i:c + 1], p, vins, xt, zt, yt, R=R, **kw2)
            out.append(('hops', i, hits, TIMES[c]))
        else:
            pc, vc = p, vins[0]
        if c == n - 1:
            out.append(('end', c, pc, vc)); return out
        t_next = TIMES[c + 1]; t_rel = t_next - TF_C
        p_lip = pc.copy(); v_rel = np.zeros(3)
        out.append(('cradle', c, pc, vc, t_rel, p_lip, v_rel, t_next))
        CRADLES.append(dict(who=name, t=float(TIMES[c]), p=pc.tolist(), t_roll=float(t_rel), t_rel=float(t_rel), lip=p_lip.tolist()))
        p = p_lip + v_rel * TF_C + np.array([0, 0, -0.5 * G * TF_C ** 2]); vins = [v_rel + np.array([0, 0, -G * TF_C])]
        i = c + 1


def write_voice(ball, segs, kind_of, pitch_of, off=np.zeros(3)):
    for sg in segs:
        if sg[0] == 'hops':
            _, i0, hits, t_next = sg
            write_hops(ball, hits, t_next, lambda k: kind_of(i0 + k), lambda k: pitch_of(i0 + k), lambda k: i0 + k, off=off)
        elif sg[0] == 'cradle':
            _, c, pc, vc, t_rel, p_lip, v_rel, t_next = sg
            pc = pc + off; p_lip = p_lip + off
            ball.hit(TIMES_OF[ball.name][c], pc, vc, [0, 0, 0], 'cradle', pitch_of(c), ev=c)
            ball.rest(TIMES_OF[ball.name][c], t_rel, pc)
            ball.fly(t_rel, t_next, p_lip, v_rel)
    return segs[-1]


TIMES_OF = {}
# =================================================================== the star marble (voice)
STAR = Ball('star', RS)
XS0 = 0.55                                         # star targets sit a little right of the frame centre
TDROP_S = 0.8
z_first = zs0(TL[0]) + ZS + KP * (LEAD[0][1] - 70)
p_moon = np.array([xs0(TL[0]) + XS0, YS, z_first + 0.5 * G * TDROP_S ** 2])
STAR.rest(0.0, TL[0] - TDROP_S, p_moon); STAR.fly(TL[0] - TDROP_S, TL[0], p_moon, [0, 0, 0])
EV['moon0'] = dict(x=float(p_moon[0]), y=YS, z=float(p_moon[2]), t_release=float(TL[0] - TDROP_S))
SFX.append(['release_star', float(TL[0] - TDROP_S)])
SK = dict(stay=1e9, nkeep=10, ext=(0.2, 0.2, 0.05), vxlim=(0.6, 3.2))
Z_SC = -13.4                                       # the star's see-saw cup: just under where the twins come down
TIMES_OF['star'] = TL
# the last note (Eb4, bar 18) is the cup on the see-saw platform
segS = plan_voice(None, TL, p_moon - [0, 0, 0.5 * G * TDROP_S ** 2], [0, 0, -G * TDROP_S],
                  lambda t: xs0(t) + XS0, lambda t: blend(lambda s: zs0(s) + ZS + KP * (pitch_at(s) - 70), Z_SC, t, TL[-1] - 5.0, TL[-1]), lambda t: YS, RS, 'star', **SK)
_, _, pS, vS = write_voice(STAR, segS, lambda k: 'chime', lambda k: LEAD[k][1])
# stage height = the star's contact heights minus the pitch contour, smoothed, never rising
_hs = sorted([h for h in STAR.hits if h['ev'] is not None], key=lambda h: h['t'])
_t = np.array([h['t'] for h in _hs] + [TL[-1]]); _z = np.array([h['p'][2] for h in _hs] + [pS[2]]) - ZS - KP * (np.array([LEAD[h['ev']][1] for h in _hs] + [LEAD[-1][1]]) - 70)
_tt = np.arange(0.0, FINAL + 31, 0.1); _zz = np.interp(_tt, _t, _z)
_k = np.ones(41) / 41; _zz = np.convolve(np.pad(_zz, 20, mode='edge'), _k, 'valid')
ZST = (_tt, np.minimum.accumulate(_zz))
# stage x = the star's contact x minus its offset, smoothed over 3 s, never going back (before the first note: speed V)
_x = np.array([h['p'][0] for h in _hs] + [pS[0]]) - XS0
_xx = np.interp(_tt, _t, _x) + np.where(_tt < _t[0], V * (_tt - _t[0]), 0.0)
_k = np.ones(31) / 31; _xx = np.convolve(np.pad(_xx, 15, mode='edge'), _k, 'valid')
for _ in range(2): _xx = np.convolve(np.pad(_xx, 15, mode='edge'), _k, 'valid')
_xx[_tt < _t[0]] = (_x[0] + V * (_tt - _t[0]))[_tt < _t[0]]
XST = (_tt, np.maximum.accumulate(_xx))
_lag = [h['p'][0] - XS0 - xs(h['t']) for h in _hs]
print('star x relative to the stage: min', round(min(_lag), 2), 'max', round(max(_lag), 2))
cupc = pS.copy()
STAR.hit(TL[-1], cupc, vS, [0, 0, 0], 'cup', LEAD[-1][1], ev=len(LEAD) - 1)
print('star cup', cupc.round(2), 'v in', vS.round(2))

# ---- finale geometry: see-saw plank along -y from under the platform (star lane) to the sheep lane, the mobile above
T_FL = FINAL - T_LAUNCH
H_UP = 2.4                                         # rise of the balls from their cups to the mobile's bells
VL = (H_UP + 0.5 * G * T_FL ** 2) / T_FL
assert VL - G * T_FL > 0.5, 'the balls must still be rising when they ring the bells'
TH0 = math.radians(14.0); ARM = (YS - YH) / (2 * math.cos(TH0)); PLANK_TOP = 0.06
pivot = np.array([cupc[0], YS - ARM * math.cos(TH0), cupc[2] - RS - PLANK_TOP - 0.06 + ARM * math.sin(TH0)])
far_up = pivot + np.array([0, -ARM * math.cos(TH0), ARM * math.sin(TH0) + PLANK_TOP + RG])
rise = 2 * ARM * math.sin(TH0)
STAR.rest(TL[-1], T_LAUNCH, cupc)
STAR.fly(T_LAUNCH, FINAL, cupc, [0, 0, VL])
pb = cupc + [0, 0, H_UP]; vb_in = np.array([0, 0, VL - G * T_FL]); vb_out = np.array([0, 0, -0.45 * vb_in[2]])
STAR.hit(FINAL, pb, vb_in, vb_out, 'moonbell', 75, ev=None)
zc = cupc[2] + rise; a_, b_, c_ = -0.5 * G, vb_out[2], pb[2] - zc
tf = (-b_ - math.sqrt(b_ * b_ - 4 * a_ * c_)) / (2 * a_)
STAR.fly(FINAL, FINAL + tf, pb, vb_out); STAR.rest(FINAL + tf, FINAL + 30, [cupc[0], cupc[1], zc])
STAR.hits.append(dict(t=float(FINAL + tf), p=[float(cupc[0]), float(cupc[1]), float(zc)], vin=[0, 0, float(vb_out[2] - G * tf)], vout=[0, 0, 0], n=[0, 0, 1],
                      e=0.0, kind='cupland', pitch=None, ev=None))
EV['seesaw'] = dict(cup=cupc.tolist(), pivot=pivot.tolist(), arm=ARM, th0=TH0, t_launch=float(T_LAUNCH), vl=VL, h_up=H_UP,
                    t_rest=float(FINAL + tf), far=far_up.tolist(), plank_top=PLANK_TOP, rise=rise)
EV['moonbell'] = dict(x=float(cupc[0]), y=YS, z=float(pb[2] + RS), t=float(FINAL))
SFX += [['launch', float(T_LAUNCH)], ['land_cup', float(FINAL + tf)], ['cup', float(TL[-1])]]

# =================================================================== the twins (piano right hand)
TW = {'pink': Ball('pink', RT), 'blue': Ball('blue', RT)}
OFF = {'pink': np.array([+DXT, -DYT, 0.0]), 'blue': np.array([-DXT, +DYT, 0.0])}
TDROP_T = 0.6
XT0 = -0.25
for n, b in TW.items():
    p_ = np.array([xs(TD[0]) + XT0, YT, zs(TD[0]) + ZT + 0.5 * G * TDROP_T ** 2]) + OFF[n]
    b.rest(0.0, TD[0] - TDROP_T, p_); b.fly(TD[0] - TDROP_T, TD[0], p_, [0, 0, 0])
EV['twin_drop'] = dict(x=float(xs(TD[0]) + XT0), y=YT, z=float(zs(TD[0]) + ZT + 0.5 * G * TDROP_T ** 2), t=float(TD[0] - TDROP_T))
SFX.append(['release_twins', float(TD[0] - TDROP_T)])
XP, ZP = cupc[0] - 1.1, -12.6                             # the twins' cups stand on posts on the platform (left of the star)
TE = TD[-1]
TIMES_OF['pink'] = TD; TIMES_OF['blue'] = TD
segT = plan_voice(None, TD, (xs(TD[0]) + XT0, YT, zs(TD[0]) + ZT), [0, 0, -G * TDROP_T],
                  lambda t: blend(lambda s: xs(s) + XT0, XP, t, TE - 5.0, TE), lambda t: blend(lambda s: zs(s) + ZT, ZP, t, TE - 5.0, TE),
                  lambda t: blend(lambda s: YT, YS, t, TE - 2.6, TE), RT, 'twins', vxlim=(0.3, 3.2))
pT, vT = segT[-1][2], segT[-1][3]
assert abs(pT[0] - cupc[0]) > 0.6 and pT[2] - RT > cupc[2] - RS - 0.06 + 0.25, ('twins cups', pT, cupc)
print('twins land at', pT.round(3), 'target', round(XP, 3), round(ZP, 3), 'v', vT.round(2))
PX = 0.5 * (cupc[0] + pT[0])                       # the plank pushes the middle of the platform
pivot[0] = PX; far_up[0] = PX
EV['seesaw'].update(pivot=pivot.tolist(), far=far_up.tolist(), px=float(PX))
FINB = []
for n, b in TW.items():
    def pk(k, n=n):
        ps_ = DUO[k][1]; return max(ps_) if (len(ps_) == 1 or n == 'pink') else min(ps_)
    write_voice(b, segT, lambda k: 'uni' if len(DUO[k][1]) == 1 else 'key', pk, off=OFF[n])
    pc = pT + OFF[n]
    b.hit(TE, pc, vT, [0, 0, 0], 'cup', pk(len(DUO) - 1), ev=len(DUO) - 1)
    b.rest(TE, T_LAUNCH, pc); b.fly(T_LAUNCH, FINAL, pc, [0, 0, VL])
    pbl = pc + [0, 0, H_UP]; pch = 82 if n == 'pink' else 79
    b.hit(FINAL, pbl, vb_in, vb_out, 'starbell', pch, ev=None)
    zc_ = pc[2] + rise; c2 = pbl[2] - zc_
    tf2 = (-b_ - math.sqrt(b_ * b_ - 4 * a_ * c2)) / (2 * a_)
    b.fly(FINAL, FINAL + tf2, pbl, vb_out); b.rest(FINAL + tf2, FINAL + 30, [pc[0], pc[1], zc_])
    b.hits.append(dict(t=float(FINAL + tf2), p=[float(pc[0]), float(pc[1]), float(zc_)], vin=[0, 0, float(vb_out[2] - G * tf2)], vout=[0, 0, 0], n=[0, 0, 1],
                       e=0.0, kind='cupland', pitch=None, ev=None))
    FINB.append(dict(ball=n, cup=pc.tolist(), bell=pbl.tolist(), pitch=pch))
EV['final_cups'] = FINB

# =================================================================== the sheep (piano left hand): one per bar
SHEEP = []; SHEEP_EV = []
XH0 = -1.0                                        # the sheep's cloud sits a little left of the frame centre
DXH = 0.75                                        # drum spacing along x
TF_DROP = 0.6; TF_FENCE = 0.95
bars = sorted(set((l[4], l[5]) for l in BASS), key=lambda pk_: [g[2] for g in GRID if g[0] == pk_[0] and g[1] == str(pk_[1])][0])


def bar_events(ps_, k): return [i for i, l in enumerate(BASS) if l[4] == ps_ and str(l[5]) == str(k)]


def zpad2(p): return 0.07 * (p - 55)               # beat-2 drums: higher for higher notes
def zpad3(p): return -0.24 + 0.05 * (p - 45)       # beat-3 drums (Ab2 / Bb2)


def check_bounce(vin, vout, what):
    e, n = bounce_e(np.array(vin, float), np.array(vout, float))
    tilt = math.degrees(math.acos(max(-1, min(1, n[2]))))
    assert EMIN <= e <= EMAX and tilt <= TILT and np.linalg.norm(vout) <= np.linalg.norm(vin) + 1e-9, (what, e, tilt)
    return e, tilt


for j, (ps_, k) in enumerate(bars[:-1]):
    ev = bar_events(ps_, k); assert len(ev) == 3, (ps_, k, ev)
    b0, b1, b2 = (BASS[i][0] for i in ev)
    sh = Ball(f'sheep{j:02d}', RH)
    x1 = xs(b0) + XH0; z1 = zs(b0)
    if SHEEP_EV: x1 = max(x1, SHEEP_EV[-1]['drums'][0][0] + 3.45)     # room for the previous pillow
    P1 = np.array([x1, YH, z1])
    P2 = np.array([x1 + DXH, YH, z1 + zpad2(BASS[ev[1]][1][0])])
    P3 = np.array([x1 + 2 * DXH, YH, z1 + zpad3(BASS[ev[2]][1][0])])
    P4 = np.array([P3[0] + 1.2, YH, z1 - 0.85])    # its pillow behind the fence
    pc_ = P1 + [0, 0, 0.5 * G * TF_DROP ** 2]
    sh.rest(0.0, b0 - TF_DROP, pc_); sh.fly(b0 - TF_DROP, b0, pc_, [0, 0, 0])
    vin = np.array([0, 0, -G * TF_DROP])
    seq = [(b0, P1, ev[0]), (b1, P2, ev[1]), (b2, P3, ev[2])]
    for m, (t, P, i) in enumerate(seq):
        tn, Pn = (seq[m + 1][0], seq[m + 1][1]) if m < 2 else (b2 + TF_FENCE, P4)
        d = tn - t; vout = (Pn - P) / d + np.array([0, 0, 0.5 * G * d])
        check_bounce(vin, vout, (j, m))
        sh.hit(t, P, vin, vout, 'drum', BASS[i][1][0], ev=i); sh.fly(t, tn, P, vout)
        vin = vout + np.array([0, 0, -G * d])
    sh.hit(b2 + TF_FENCE, P4, vin, [0, 0, 0], 'pillow', None, ev=None)
    sh.rest(b2 + TF_FENCE, FINAL + 30, P4)
    # the fence: under the flight's highest point, its top 0.12 below the sheep's wool
    vo = sh.hits[-2]['vout']; ta = vo[2] / G; xa = P3[0] + vo[0] * ta; za = P3[2] + vo[2] * ta - 0.5 * G * ta * ta
    SHEEP.append(sh)
    SHEEP_EV.append(dict(name=sh.name, ps=ps_, bar=str(k), cloud=pc_.tolist(), t_release=float(b0 - TF_DROP), drums=[P1.tolist(), P2.tolist(), P3.tolist()],
                         pitches=[BASS[i][1][0] for i in ev], times=[b0, b1, b2], pillow=P4.tolist(), t_sleep=float(b2 + TF_FENCE),
                         fence=dict(x=float(xa), z_top=float(za - RH - 0.3), t=float(b2 + ta))))
    SFX += [['cloud', float(b0 - TF_DROP)], ['pillow', float(b2 + TF_FENCE)]]
# ---- the 25th sheep: a giant. Eb2 drum on beat 1, the see-saw's far end (the LH chord bells) on beat 2
ps_, k = bars[-1]; ev = bar_events(ps_, k); assert len(ev) == 2
b0 = BASS[ev[0]][0]; assert abs(BASS[ev[1]][0] - T_LAUNCH) < 1e-6
GS = Ball('giant', RG)
P1 = np.array([PX - 1.25, YH, far_up[2] + 0.35])
pc_ = P1 + [0, 0, 0.5 * G * 0.75 ** 2]
GS.rest(0.0, b0 - 0.75, pc_); GS.fly(b0 - 0.75, b0, pc_, [0, 0, 0])
vin = np.array([0, 0, -G * 0.75]); d = T_LAUNCH - b0
vout = (far_up - P1) / d + np.array([0, 0, 0.5 * G * d]); check_bounce(vin, vout, 'giant')
GS.hit(b0, P1, vin, vout, 'drum', BASS[ev[0]][1][0], ev=ev[0]); vF = vout + np.array([0, 0, -G * d])
GS.fly(b0, T_LAUNCH, P1, vout)
GS.hits.append(dict(t=float(T_LAUNCH), p=far_up.tolist(), vin=vF.tolist(), vout=[0, 0, 0], n=[0, 0, 1], e=0.0, kind='seesaw', pitch=BASS[ev[1]][1], ev=ev[1]))
end_down = pivot + np.array([0, -ARM * math.cos(TH0), -ARM * math.sin(TH0) + PLANK_TOP + RG])
T_SW = 0.07
tt = np.arange(T_LAUNCH, T_LAUNCH + T_SW + 1e-9, 1 / 240.0)
GS.path(tt, np.array([far_up + (end_down - far_up) * min(1, (t - T_LAUNCH) / T_SW) for t in tt]))
GS.rest(T_LAUNCH + T_SW, FINAL + 30, end_down)
EV['seesaw']['t_swing'] = T_SW
EV['giant'] = dict(cloud=pc_.tolist(), t_release=float(b0 - 0.75), drum=P1.tolist(), t=float(b0), pitch=BASS[ev[0]][1][0], chord=BASS[ev[1]][1])
SFX += [['cloud_big', float(b0 - 0.75)], ['seesaw', float(T_LAUNCH)]]
SHEEP.append(GS)
EV['sheep'] = SHEEP_EV
EV['cradles'] = CRADLES
EV['lights_out'] = float(FINAL + 0.5)
EV['stage'] = [ZST[0][::5].tolist(), ZST[1][::5].tolist()]
EV['stage_x'] = [XST[0][::5].tolist(), XST[1][::5].tolist()]
BALLS = {'star': STAR, 'pink': TW['pink'], 'blue': TW['blue']}
for s_ in SHEEP: BALLS[s_.name] = s_


def check():
    info = {}
    hs = [(n, h) for n, b in BALLS.items() for h in b.hits if h['kind'] in ('chime', 'uni', 'key', 'drum')]
    info['cradles'] = len(CRADLES)
    info['e_range'] = (round(min(h['e'] for n, h in hs), 3), round(max(h['e'] for n, h in hs), 3))
    allh = [h for b in BALLS.values() for h in b.hits if np.linalg.norm(h['vout']) > 0]
    info['speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in allh), 3)
    info['tilt_max_deg'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for n, h in hs), 1)
    for n in ('star', 'pink', 'blue', 'sheep00', 'giant'):
        b = BALLS[n]; zz = [h['p'][2] for h in b.hits]; xx = [h['p'][0] for h in b.hits]
        info[n] = dict(hits=len(b.hits), z=(round(min(zz), 2), round(max(zz), 2)), x=(round(min(xx), 1), round(max(xx), 1)))
    info['launch_v'] = round(VL, 2); info['sheep'] = len(SHEEP)
    return info


if __name__ == '__main__':
    for k, v in check().items(): print(k, v)
    out = dict(G=G, R=dict(star=RS, twins=RT, sheep=RH, giant=RG), lanes=dict(star=YS, twins=YT, sheep=YH), zoff=dict(star=ZS, twins=ZT), kp=KP, v=V,
               dxt=DXT, dyt=DYT, balls={n: dict(r=b.r, seg=b.seg, hits=b.hits) for n, b in BALLS.items()}, ev=EV, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
