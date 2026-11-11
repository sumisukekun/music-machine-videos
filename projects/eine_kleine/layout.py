"""Eine kleine Nachtmusik - '夜の宮殿の灯り' (lights of the night palace). Physics + story planner.
Run after arrange.py; writes layout.json (read by scene.py, render_audio.py, verify.py).

Cast (one ball per voice):
  lamp   (V1, violin I)  a glowing ember ball. Every melody note = it lands on a candlestick key; in the first half each key's
                         candle lights. Bars 9-10 (sixteenths) it rolls down a xylophone ramp.
  duo    (V2, violin II) the dancing couple (rose + pearl). One shared wide key for a single note, two keys for a dyad.
  bass   (cello)         the big gold ball on gold bells; bars 5-10 (eighths) it rolls along a bass-bar track;
                         during its two rests (bars 11, 15) a bucket lift carries it.
Story
  0-2.4 s   pitch dark palace; three balls drop from a dark chandelier
  bars 1-10 every lamp note lights a candle (the counter rises), the palace slowly brightens
  bar 10/4  the window bursts open: a gust blows every candle (and the lamp ball's glow) out -> the counter falls to 0
  bars 11-17 piano: they play on in the moonlight
  da capo   the lamp ball lands in the last glowing brazier: it re-ignites, a fuse races along the walls and relights the palace
  finale    the lamp ball rolls into the cup of a see-saw; the bass ball drops on the far end and throws it up into the
            grand chandelier -> final chord, every light in the palace comes on (counter 525 = K.525)
Physics: one constant gravity G; between contacts pure parabolas; bounces are impulses along the key normal with
restitution emin <= e <= emax (never adds speed); ramps/tracks roll with a = (5/7) g sin(a) - c_rr g cos(a); a landing on a
ramp keeps at most the along-track part of the velocity. Visible energy sources only: the bucket lifts, the see-saw."""
import json, os, math
import numpy as np
from phys import Ball, seg_at, bounce_e, G, EMAX, EMIN, TILT
import phys, pickle, hashlib
_CF = '/tmp/eine_kleine_plan_cache.pkl'
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
    for W, NX, NZ in ((220, 17, 31), (400, 21, 41)):
        try: return plan(*a, **dict(kw, W=W, NX=NX, NZ=NZ))
        except RuntimeError as e_: err = e_; print('retry', e_)
    raise err
HERE = os.path.dirname(os.path.abspath(__file__))
RL, RP, RB = 0.22, 0.19, 0.3
DXP, DYP = 0.24, 0.2
YL, YD, YB = 0.0, 1.15, 2.3
ZD, ZB = 1.9, 3.7
V = 1.5
E = json.load(open(f'{HERE}/events.json'))
LEAD = json.load(open(f'{HERE}/lead.json')); DUO = json.load(open(f'{HERE}/duo.json')); BASS = json.load(open(f'{HERE}/bass.json'))
T0, Q, BAR, FINAL, GUST, DACAPO = E['t0'], E['q'], E['bar'], E['final'], E['gust'], E['dacapo']
TL = np.array([l[0] for l in LEAD]); TD = np.array([l[0] for l in DUO]); TB = np.array([l[0] for l in BASS])
SFX = []; EV = {}
X0, Z0 = 0.0, 0.0
T_DROP_STAGE = (T0 + 9 * BAR + 2 * Q, T0 + 10 * BAR)      # the stage steps down from the D6 cadence to bar 11
DROP = 1.25


def zs(t):
    u = min(1.0, max(0.0, (t - T_DROP_STAGE[0]) / (T_DROP_STAGE[1] - T_DROP_STAGE[0])))
    return Z0 - 0.02 * (t - T0) - DROP * u


def xs(t): return X0 + V * (t - T0)


def idx(lst, ps, k, b):
    for i, l in enumerate(lst):
        if l[4] == ps and l[5] == k and abs(l[6] - b) < 1e-6: return i
    raise KeyError((ps, k, b))


def ramp(ball, times, pitches, evs, p_land, v_in, alpha, crr, kind, k_land=0.85, tail=0.03, R=RL):
    """the ball lands on an inclined xylophone track at times[0] and rolls down it (a = 5/7 g sin a - c_rr g cos a);
    each note = the bar under the ball at that moment. Returns lip point / velocity / time and the bar list."""
    d = np.array([math.cos(alpha), 0, -math.sin(alpha)]); nrm = np.array([math.sin(alpha), 0, math.cos(alpha)])
    v0 = k_land * float(v_in @ d); assert v0 > 0.5, v0
    a = 5 / 7 * G * math.sin(alpha) - crr * G * math.cos(alpha)
    t0 = times[0]; t_end = times[-1] + tail
    def s(t): u = t - t0; return v0 * u + 0.5 * a * u * u
    tt = np.arange(t0, t_end + 1e-9, 1 / 240.0)
    if tt[-1] < t_end - 1e-9: tt = np.append(tt, t_end)
    ball.path(tt, np.array([p_land + d * s(t) for t in tt]))
    bars = []
    for j, (t, p, ev) in enumerate(zip(times, pitches, evs)):
        pc = p_land + d * s(t); vv = d * (v0 + a * (t - t0))
        if j == 0: ball.hit(t, pc, v_in, vv, kind + 'land', p, ev=ev)
        else: ball.hits.append(dict(t=float(t), p=pc.tolist(), vin=vv.tolist(), vout=vv.tolist(), n=nrm.tolist(), e=1.0, kind=kind, pitch=p, ev=ev))
        bars.append(dict(t=float(t), p=pc.tolist(), pitch=p, ev=ev))
    v_lip = d * (v0 + a * (t_end - t0)); p_lip = p_land + d * s(t_end)
    info = dict(t0=float(t0), t1=float(t_end), p0=p_land.tolist(), alpha=alpha, crr=crr, a=a, v0=v0, v1=float(v0 + a * (t_end - t0)),
                L=float(s(t_end)), lip=p_lip.tolist(), bars=bars, r=R, k_land=k_land, n=nrm.tolist())
    return p_lip, v_lip, t_end, info


def write_hops(ball, hits, t_next, kinds, pitch_of, ev_of, off=np.zeros(3), vin_pick=0):
    for k, h in enumerate(hits):
        vin = h['vins'][vin_pick if len(h['vins']) > 1 else 0]
        ball.hit(h['t'], h['p'] + off, vin, h['vout'], kinds(k), pitch_of(k), ev=ev_of(k))
        t1 = hits[k + 1]['t'] if k + 1 < len(hits) else t_next
        ball.fly(h['t'], t1, h['p'] + off, h['vout'])


def free_to(ball, t0, p0, v0, t1):
    ball.fly(t0, t1, p0, v0); u = t1 - t0
    return np.array(p0) + np.array(v0) * u + np.array([0, 0, -0.5 * G * u * u]), np.array(v0) + np.array([0, 0, -G * u])


def bounce_search(vin, d, dzs, vxs, vy=0.0, emin=EMIN, emax=EMAX, tilt=TILT):
    for dz in dzs:
        for vx in vxs:
            vo = np.array([vx, vy, dz / d + 0.5 * G * d]); e, n = bounce_e(vin, vo)
            if emin <= e <= emax and n[2] >= math.cos(math.radians(tilt)) and vo[2] - G * d < -0.3: return vo
    raise RuntimeError('no bounce')


# =================================================================== the lamp ball (violin I)
LAMP = Ball('lamp', RL)
H_DROP = 0.5 * G * 1.0 ** 2
p_ch = np.array([X0, YL, Z0 + H_DROP])
LAMP.rest(0.0, T0 - 1.0, p_ch); LAMP.fly(T0 - 1.0, T0, p_ch, [0, 0, 0])
EV['chandelier0'] = dict(x=X0, y=YL, z=float(Z0 + H_DROP), t_release=T0 - 1.0)
SFX.append(['release', T0 - 1.0])
LKIND = {}                                    # lamp event -> kind
def lamp_kind(i):
    ps, k = LEAD[i][4], LEAD[i][5]
    if ps == 1 and k <= 10: return 'candle'
    if ps == 1: return 'dark'
    if k == 1 and LEAD[i][6] == 0: return 'brazier'
    return 'candle2'
iR1 = idx(LEAD, 1, 9, 0); iD61 = idx(LEAD, 1, 10, 2); iR2 = idx(LEAD, 2, 9, 0); iCUP = idx(LEAD, 2, 10, 2)
LK = dict(stay=1e9, nkeep=12, R=RL, ext=(0.16, 0.26, 0.06))
hA, pA, vA = plan(TL[0:iR1 + 1], (X0, YL, Z0), [np.array([0, 0, -G])], xs, zs, lambda t: YL, vxlim=(0.6, 3.2), **LK)
keysA = plan.last[1]
write_hops(LAMP, hA, TL[iR1], lambda k: lamp_kind(k), lambda k: LEAD[k][1], lambda k: k)
RAMPS = []
ALPHA_L, CRR_L = math.radians(5.0), 0.03
ev1 = list(range(iR1, iD61))
pl, vl, tl, info = ramp(LAMP, TL[iR1:iD61], [LEAD[i][1] for i in ev1], ev1, pA, vA, ALPHA_L, CRR_L, 'xylo')
info['who'] = 'lamp'; info['pass'] = 1; RAMPS.append(info)
pD, vD = free_to(LAMP, tl, pl, vl, TL[iD61])
# B: D6 cadence -> the piano phrase (stage steps down) -> da capo (brazier) -> ramp 2
xoff = pD[0] - xs(TL[iD61])
xtB = lambda t: xs(t) + xoff * max(0.0, 1 - (t - TL[iD61]) / 8.0)
hB, pB, vB = plan(TL[iD61:iR2 + 1], pD, [vD], xtB, zs, lambda t: YL, vxlim=(0.5, 3.2), **LK)
write_hops(LAMP, hB, TL[iR2], lambda k: lamp_kind(iD61 + k), lambda k: LEAD[iD61 + k][1], lambda k: iD61 + k)
ev2 = list(range(iR2, iCUP))
pl, vl, tl, info = ramp(LAMP, TL[iR2:iCUP], [LEAD[i][1] for i in ev2], ev2, pB, vB, ALPHA_L, CRR_L, 'xylo2')
info['who'] = 'lamp'; info['pass'] = 2; RAMPS.append(info)
pC, vC = free_to(LAMP, tl, pl, vl, TL[iCUP])
cupc = pC.copy()
LAMP.hit(TL[iCUP], cupc, vC, [0, 0, 0], 'cup', LEAD[iCUP][1], ev=iCUP)
EV['brazier'] = [h['p'] for h in LAMP.hits if h['kind'] == 'brazier'][0]

# ---- finale geometry: see-saw (plank along +y from the cup to the bass lane), the grand chandelier above the cup
T_LAUNCH = TL[iCUP] + 0.62
T_FL = FINAL - T_LAUNCH
H_UP = 2.6                                     # the lamp ball's top touches the chandelier bell crown
VL = (H_UP + 0.5 * G * T_FL ** 2) / T_FL
TH0 = math.radians(14.0); ARM = (YB - YL) / (2 * math.cos(TH0)); PLANK_TOP = 0.06
pivot = np.array([cupc[0], YL + ARM * math.cos(TH0), cupc[2] - RL - PLANK_TOP + ARM * math.sin(TH0)])
far_up = pivot + np.array([0, ARM * math.cos(TH0), ARM * math.sin(TH0) + PLANK_TOP + RB])      # bass centre on the raised far end
LAMP.rest(TL[iCUP], T_LAUNCH, cupc)
rise = 2 * ARM * math.sin(TH0)
LAMP.fly(T_LAUNCH, FINAL, cupc, [0, 0, VL])
pb = cupc + [0, 0, H_UP]; vb_in = np.array([0, 0, VL - G * T_FL]); vb_out = np.array([0, 0, -0.45 * vb_in[2]])
LAMP.hit(FINAL, pb, vb_in, vb_out, 'bigbell', 79, ev=None)
zc = cupc[2] + rise; a_, b_, c_ = -0.5 * G, vb_out[2], pb[2] - zc
tf = (-b_ - math.sqrt(b_ * b_ - 4 * a_ * c_)) / (2 * a_)
LAMP.fly(FINAL, FINAL + tf, pb, vb_out); LAMP.rest(FINAL + tf, FINAL + 20, [cupc[0], cupc[1], zc])
EV['seesaw'] = dict(cup=cupc.tolist(), pivot=pivot.tolist(), arm=ARM, th0=TH0, t_launch=float(T_LAUNCH), vl=VL, h_up=H_UP,
                    t_rest=float(FINAL + tf), far=far_up.tolist(), plank_top=PLANK_TOP)
EV['bigbell'] = dict(x=float(cupc[0]), y=YL, z=float(cupc[2] + H_UP + RL), t=float(FINAL))
SFX += [['launch', float(T_LAUNCH)], ['land_cup', float(FINAL + tf)], ['cup', float(TL[iCUP])]]

# =================================================================== the couple (violin II)
DUOB = {'rose': Ball('rose', RP), 'pearl': Ball('pearl', RP)}
OFF = {'rose': np.array([+DXP, -DYP, 0.0]), 'pearl': np.array([-DXP, +DYP, 0.0])}
TDROP_D = 0.9
for n, b in DUOB.items():
    p_ = np.array([X0 + 0.2, YD, Z0 + ZD + 0.5 * G * TDROP_D ** 2]) + OFF[n]
    b.rest(0.0, T0 - TDROP_D, p_); b.fly(T0 - TDROP_D, T0, p_, [0, 0, 0])
EV['duo_drop'] = dict(x=X0 + 0.2, y=YD, z=float(Z0 + ZD + 0.5 * G * TDROP_D ** 2), t=T0 - TDROP_D)
# the couple's last note (G4) lands in two cups on the see-saw platform beside the lamp's cup; the see-saw throws all three
# up into the grand chandelier: the couple rings its two small bells (B4, D5) with the lamp's big bell (final chord)
XP, ZP = cupc[0] + 0.85, cupc[2] - RL + RP + 0.8          # the couple's cups stand on 0.8 m posts on the platform
def blend(f, g, t, t0, t1):
    u = min(1.0, max(0.0, (t - t0) / (t1 - t0))); return (1 - u) * f(t) + u * g
TE = TD[-1]
hDu, pDu, vDu = plan_retry(TD, (X0 + 0.2, YD, Z0 + ZD), [np.array([0, 0, -G * TDROP_D])],
                           lambda t: blend(lambda s: xs(s) + 0.2, XP, t, TE - 6.0, TE), lambda t: blend(lambda s: zs(s) + ZD, ZP, t, TE - 6.0, TE),
                           lambda t: blend(lambda s: YD, 0.0, t, TE - 3.0, TE), vxlim=(0.3, 3.2), R=RP)
print('couple lands at', pDu.round(3), 'target', round(XP, 3), round(ZP, 3), 'v', vDu.round(2))
PX = 0.5 * (cupc[0] + pDu[0])                      # the plank pushes the middle of the long platform
pivot[0] = PX; far_up[0] = PX
EV['seesaw'].update(pivot=pivot.tolist(), far=far_up.tolist(), px=float(PX))
FINB = []
for n, b in DUOB.items():
    def pk(k, n=n):
        ps_ = DUO[k][1]; return max(ps_) if (len(ps_) == 1 or n == 'rose') else min(ps_)
    write_hops(b, hDu, TE, lambda k: 'uni' if len(DUO[k][1]) == 1 else 'key', pk, lambda k: k, off=OFF[n])
    pc = pDu + OFF[n]
    b.hit(TE, pc, vDu, [0, 0, 0], 'cup', DUO[-1][1][0], ev=len(DUO) - 1)
    b.rest(TE, T_LAUNCH, pc); b.fly(T_LAUNCH, FINAL, pc, [0, 0, VL])
    pbl = pc + [0, 0, H_UP]; pch = 71 if n == 'pearl' else 74
    b.hit(FINAL, pbl, vb_in, vb_out, 'smallbell', pch, ev=None)
    zc_ = pc[2] + rise; c2 = pbl[2] - zc_
    tf2 = (-b_ - math.sqrt(b_ * b_ - 4 * a_ * c2)) / (2 * a_)
    b.fly(FINAL, FINAL + tf2, pbl, vb_out); b.rest(FINAL + tf2, FINAL + 20, [pc[0], pc[1], zc_])
    FINB.append(dict(ball=n, cup=pc.tolist(), bell=pbl.tolist(), pitch=pch))
EV['final_cups'] = FINB

# =================================================================== the bass ball (cello)
BB = Ball('bass', RB)
TDROP_B = 0.9
pst = np.array([X0 + 0.5, YB, Z0 + ZB + 0.5 * G * TDROP_B ** 2])
BB.rest(0.0, T0 - TDROP_B, pst); BB.fly(T0 - TDROP_B, T0, pst, [0, 0, 0])
EV['bass_drop'] = dict(x=X0 + 0.5, y=YB, z=float(pst[2]), t=T0 - TDROP_B)
BK = dict(R=RB, emax=0.975, emin=0.3, dzr=(-1.0, 0.6), ext=(0.26, 0.26, 0.2), vxlim=(0.4, 3.0))
bkind = lambda i: 'bell'
iT1 = idx(BASS, 1, 5, 0); iB10 = idx(BASS, 1, 10, 2); iD12 = idx(BASS, 1, 12, 0); iG14 = idx(BASS, 1, 14, 2)
iD16 = idx(BASS, 1, 16, 0); iT2 = idx(BASS, 2, 5, 0); iB20 = idx(BASS, 2, 10, 2)
zb = lambda t: zs(t) + ZB
h1, p1, v1 = plan(TB[0:iT1 + 1], (X0 + 0.5, YB, Z0 + ZB), [np.array([0, 0, -G * TDROP_B])], lambda t: xs(t) + 0.5, zb, lambda t: YB, **dict(BK, vxlim=(1.45, 3.0)))
write_hops(BB, h1, TB[iT1], bkind, lambda k: BASS[k][1], lambda k: k)
ALPHA_B = math.radians(1.5)


def track(i0, i1, p_land, v_in, alpha, crr, kind, k_land):
    evs = list(range(i0, i1 + 1))
    return ramp(BB, TB[i0:i1 + 1], [BASS[i][1] for i in evs], evs, p_land, v_in, alpha, crr, kind, k_land=k_land, tail=0.14, R=RB)


d_ = np.array([math.cos(ALPHA_B), 0, -math.sin(ALPHA_B)])
k1 = min(0.95, 1.5 / float(v1 @ d_))
pl, vl, tl, info = track(iT1, iB10, p1, v1, ALPHA_B, 5 / 7 * math.tan(ALPHA_B), 'track', k1)
info['who'] = 'bass'; info['pass'] = 1; RAMPS.append(info)
ELEV = []


def elevator(t_in, p_in, v_in, t_bell, z_bell_target, name, lip_info=None):
    """the ball drops into a padded bucket (stops), the bucket is wound up, tips, the ball rolls out and falls onto the bell"""
    BB.hit(t_in, p_in, v_in, [0, 0, 0], 'bucket', None, ev=None)
    tf = 0.55; vrel = np.array([0.45, 0, 0])
    z_rel = z_bell_target + 0.5 * G * tf * tf
    H = max(0.35, z_rel - p_in[2]); p_rel = p_in + np.array([0, 0, H])
    t_rel = t_bell - tf; t_up0 = t_in + 0.25; t_up1 = t_rel - 0.35
    tt = np.arange(t_up0, t_up1 + 1e-9, 1 / 240.0)
    def lift(t):
        u = (t - t_up0) / (t_up1 - t_up0); u = u * u * (3 - 2 * u); return p_in + np.array([0, 0, H * u])
    BB.rest(t_in, t_up0, p_in); BB.path(tt, np.array([lift(t) for t in tt])); BB.rest(t_up1, t_rel, p_rel)
    pb_, vb_ = free_to(BB, t_rel, p_rel, vrel, t_bell)
    ELEV.append(dict(name=name, t_in=float(t_in), p_in=p_in.tolist(), t_up0=float(t_up0), t_up1=float(t_up1), H=float(H),
                     t_rel=float(t_rel), p_rel=p_rel.tolist()))
    SFX.extend([['bucket', float(t_in)], ['winch', float(t_up0)], ['tip', float(t_rel - 0.1)]])
    return pb_, vb_


p_in, v_in = free_to(BB, tl, pl, vl, tl + 0.3)
pD3, vD3 = elevator(tl + 0.3, p_in, v_in, TB[iD12], zb(TB[iD12]), 'lift1')
h2, p2, v2 = plan(TB[iD12:iG14 + 1], pD3, [vD3], lambda t: xs(t) + 0.5, zb, lambda t: YB, **BK)
write_hops(BB, h2, TB[iG14], bkind, lambda k: BASS[iD12 + k][1], lambda k: iD12 + k)
for DB in (0.5, 0.6, 0.7, 0.8):
    try: vo = bounce_search(v2, DB, np.linspace(-0.05, -2.5, 120), np.linspace(1.4, 0.3, 12), emin=0.3, emax=0.975); break
    except RuntimeError: vo = None
assert vo is not None, ('bucket2', v2)
BB.hit(TB[iG14], p2, v2, vo, 'bell', BASS[iG14][1], ev=iG14)
p_in, v_in = free_to(BB, TB[iG14], p2, vo, TB[iG14] + DB)
pD16, vD16 = elevator(TB[iG14] + DB, p_in, v_in, TB[iD16], zb(TB[iD16]), 'lift2')
# B3: bars 16-17, da capo bars 1-4, onto the head of track 2, whose end must drop the ball on the see-saw's raised far end
TF_B = 0.55
T_LIP2 = T_LAUNCH - TF_B
zfar = far_up[2]
dur = T_LIP2 - TB[iT2] + TF_B
vxp = 1.5 * math.cos(math.radians(2.5)); vzp = 1.5 * math.sin(math.radians(2.5))
x_start = PX - vxp * dur
z_start = zfar + 0.5 * G * TF_B ** 2 + vzp * dur
h3, p3, v3 = plan(TB[iD16:iT2 + 1], pD16, [vD16], lambda t: xs(t) + 0.5 + (x_start - xs(TB[iT2]) - 0.5) * max(0, (t - TB[iD16]) / (TB[iT2] - TB[iD16])),
                  lambda t: zb(t) + (z_start - zb(TB[iT2])) * max(0, (t - TB[iD16]) / (TB[iT2] - TB[iD16])), lambda t: YB,
                  **dict(BK, vxlim=(1.55, 3.0)))
print('bass track2 head', p3.round(2), 'wanted', round(x_start, 2), round(z_start, 2), 'v', v3.round(2))
write_hops(BB, h3, TB[iT2], bkind, lambda k: BASS[iD16 + k][1], lambda k: iD16 + k)
# track 2: solve slope + speed so that the ball leaves the lip at T_LIP2 and lands on the far end at T_LAUNCH
dur = T_LIP2 - TB[iT2] + TF_B
vxp = (PX - p3[0]) / dur; vzp = (p3[2] - zfar - 0.5 * G * TF_B ** 2) / dur
ALPHA_B2 = math.atan2(vzp, vxp); v0_2 = math.hypot(vxp, vzp)
d2 = np.array([math.cos(ALPHA_B2), 0, -math.sin(ALPHA_B2)])
k2 = v0_2 / float(v3 @ d2)
print('track2 alpha deg', round(math.degrees(ALPHA_B2), 2), 'v0', round(v0_2, 2), 'k', round(k2, 3))
assert 0.3 < k2 <= 1.0, k2
pl, vl, tl, info = ramp(BB, TB[iT2:iB20 + 1], [BASS[i][1] for i in range(iT2, iB20 + 1)], list(range(iT2, iB20 + 1)),
                        p3, v3, ALPHA_B2, 5 / 7 * math.tan(ALPHA_B2), 'track2', k_land=k2, tail=T_LIP2 - TB[iB20], R=RB)
info['who'] = 'bass'; info['pass'] = 2; RAMPS.append(info)
pF, vF = free_to(BB, tl, pl, vl, T_LAUNCH)
BB.hits.append(dict(t=float(T_LAUNCH), p=pF.tolist(), vin=vF.tolist(), vout=[0, 0, 0], n=[0, 0, 1], e=0.0, kind='seesaw', pitch=None, ev=None))
end_down = pivot + np.array([0, ARM * math.cos(TH0), -ARM * math.sin(TH0) + PLANK_TOP + RB])
T_SW = 0.07
tt = np.arange(T_LAUNCH, T_LAUNCH + T_SW + 1e-9, 1 / 240.0)
BB.path(tt, np.array([pF + (end_down + [pF[0] - PX, 0, 0] - pF) * min(1, (t - T_LAUNCH) / T_SW) for t in tt]))
BB.rest(T_LAUNCH + T_SW, FINAL + 20, end_down + [pF[0] - PX, 0, 0])
EV['seesaw']['t_swing'] = T_SW; EV['seesaw']['bass_land'] = pF.tolist(); EV['seesaw']['bass_miss'] = float(np.linalg.norm(pF - far_up))
SFX.append(['seesaw', float(T_LAUNCH)])
EV['ramps'] = RAMPS; EV['elev'] = ELEV
SFX += [['gust', GUST], ['brazier', DACAPO]]
BALLS = {'lamp': LAMP, 'rose': DUOB['rose'], 'pearl': DUOB['pearl'], 'bass': BB}


def check():
    info = {}
    hs = [(n, h) for n, b in BALLS.items() for h in b.hits if h['kind'] in ('candle', 'dark', 'brazier', 'candle2', 'uni', 'key', 'bell')]
    info['e_range'] = (round(min(h['e'] for n, h in hs), 3), round(max(h['e'] for n, h in hs), 3))
    allh = [h for b in BALLS.values() for h in b.hits if np.linalg.norm(h['vout']) > 0]
    info['speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in allh), 3)
    info['tilt_max_deg'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for n, h in hs), 1)
    for n, b in BALLS.items():
        zz = [h['p'][2] for h in b.hits]; xx = [h['p'][0] for h in b.hits]
        info[n] = dict(hits=len(b.hits), z=(round(min(zz), 2), round(max(zz), 2)), x=(round(min(xx), 1), round(max(xx), 1)))
    info['ramps'] = [(r['who'], r['pass'], round(math.degrees(r['alpha']), 2), round(r['crr'], 3), round(r['v0'], 2), round(r['v1'], 2), round(r['L'], 1), round(r['k_land'], 2)) for r in RAMPS]
    info['lifts'] = [(e['name'], round(e['H'], 2)) for e in ELEV]
    info['launch_v'] = round(VL, 2); info['bass_miss_far_end'] = round(EV['seesaw']['bass_miss'], 4)
    return info


if __name__ == '__main__':
    for k, v in check().items(): print(k, v)
    out = dict(G=G, R=dict(lamp=RL, duo=RP, bass=RB), lanes=dict(lamp=YL, duo=YD, bass=YB), zoff=dict(duo=ZD, bass=ZB), dxp=DXP, dyp=DYP,
               balls={n: dict(r=b.r, seg=b.seg, hits=b.hits) for n, b in BALLS.items()}, ev=EV, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
