"""Clair de Lune - 'Can you hear the moonlight?'  Physics + story planner for a moonlit Japanese water garden (side view).
Run after arrange.py; writes layout.json (read by scene.py, render_audio.py, verify.py).

Cast
  twins       silver + gold glass marbles (the melody: silver = upper note, gold = lower note of the thirds), always as a pair.
              They bounce on lotus-leaf keys standing on stems in a chain of terraced ponds; every strike = a ripple below.
  drops       water drops (the harmony): each forms at the tip of a bamboo spout and falls onto a glass bowl bell.
  moon-stone  a big pale stone ball (the bass), waiting on a rock from the start.
Story (one constant gravity G; bounces never add speed; every lift / launch has a visible cause)
  0-3.5 s     two dew drops hang from a willow tip in front of a thin crescent moon; they let go
  bars 1-2    lotus keys; then a lantern gondola on a rope (zip line) carries the pair down a terrace
  bars 3-4    lotus keys; the pair drops into a well bucket: a spout fills the counter-bucket, which sinks and hauls it up
  bar 5       the bucket tips them onto the bamboo xylophone slide (one rail of bars per ball)
  bars 6-8    lotus keys, a second slide (bar 7); the last key of bar 8 is a lever: it pulls the latch of the moon-stone
  bar 8-9     the moon-stone falls, rings the A-flat slab, lands on the D-flat plate at the far end of a stone see-saw and
              catapults the pair (waiting in the near cup) high into the sky for the return of the theme
  bars 9-14   the moon-stone rolls along the stone channel at the water's edge, its weight sounding each bass bar;
              bar 10->11 the pair rides a shishi-odoshi (bamboo water see-saw): water fills the tube, it tips and throws them
  final       the moon-stone rolls off the channel into the middle of the last pond (the great gong), the pair lands in two lotus
              flower bells, the last drops fall: D-flat major, the full moon on the water.
Rolling: a = 5/7 g sin(a) - c_rr g cos(a); a landing on a track keeps at most the along-track part of the velocity."""
import json, os, math
import numpy as np
from phys import Ball, seg_at, bounce_e, G, EMAX, EMIN, TILT
import phys, pickle, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
_CF = '/tmp/clair_de_lune_plan_cache.pkl'
_C = pickle.load(open(_CF, 'rb')) if os.path.exists(_CF) else {}


def plan(times, p0, vins, xt, zt, yt, **kw):
    """phys.plan with a disk cache (key = all inputs, target functions sampled at the note times)"""
    times = np.asarray(times, float)
    key = hashlib.md5(repr((np.round(times, 6).tolist(), np.round(np.asarray(p0, float), 6).tolist(), [np.round(np.asarray(v, float), 6).tolist() for v in vins],
                            [(round(xt(t), 6), round(zt(t), 6), round(yt(t), 6)) for t in times], sorted((k, repr(v)) for k, v in kw.items()))).encode()).hexdigest()
    if key not in _C:
        err = None
        for W, NX, NZ in ((220, 17, 31), (450, 21, 41), (800, 25, 51)):
            try: r = phys.plan(times, p0, vins, xt, zt, yt, **dict(kw, W=W, NX=NX, NZ=NZ)); break
            except RuntimeError as e_: err = e_; print('retry', e_); r = None
        if r is None: raise err
        _C[key] = (r, phys.plan.last); pickle.dump(_C, open(_CF, 'wb'))
    r, plan.last = _C[key]
    return r


E = json.load(open(f'{HERE}/events.json'))
TW = json.load(open(f'{HERE}/twins.json')); DR = json.load(open(f'{HERE}/drops.json')); BS = json.load(open(f'{HERE}/bass.json'))
T0, E8, BAR, FINAL = E['t0'], E['e8'], E['bar'], E['final']
TT = np.array([w['t'] for w in TW])
RT, RB, RD = 0.2, 0.42, 0.13
OFF = {'silver': np.array([0.21, -0.26, 0.0]), 'gold': np.array([-0.21, 0.26, 0.0])}
YT, YC, YD = 0.0, 1.55, 3.1          # lanes: twins, moon-stone channel, drop bowls (back bank)
V = 1.45
SFX = []; EV = {}; RAMPS = []


def ti(bar, off):
    for i, w in enumerate(TW):
        if w['bar'] == bar and abs(w['off'] - off) < 1e-6: return i
    raise KeyError((bar, off))


def free_to(ball, t0, p0, v0, t1):
    ball.fly(t0, t1, p0, v0); u = t1 - t0
    return np.array(p0) + np.array(v0) * u + np.array([0, 0, -0.5 * G * u * u]), np.array(v0) + np.array([0, 0, -G * u])


def bounce_search(vin, d, dzs, vxs, vy=0.0, emin=EMIN, emax=EMAX, tilt=TILT):
    for dz in dzs:
        for vx in vxs:
            vo = np.array([vx, vy, dz / d + 0.5 * G * d]); e, n = bounce_e(vin, vo)
            if emin <= e <= emax and n[2] >= math.cos(math.radians(tilt)) and vo[2] - G * d < -0.3: return vo
    raise RuntimeError('no bounce')


class Pair:
    """the twins: a rigid pair (centre path + fixed offsets), one key per ball"""
    def __init__(s): s.b = {n: Ball(n, RT) for n in OFF}
    def rest(s, t0, t1, p):
        for n, b in s.b.items(): b.rest(t0, t1, np.array(p) + OFF[n])
    def fly(s, t0, t1, p, v):
        for n, b in s.b.items(): b.fly(t0, t1, np.array(p) + OFF[n], v)
    def path(s, ts, ps):
        for n, b in s.b.items(): b.path(ts, np.array(ps) + OFF[n])
    def hit(s, t, p, vin, vout, kind, w, **kw):
        for n, b in s.b.items():
            pitch = w['hi'] if (n == 'silver' or w['lo'] is None) else w['lo']
            b.hit(t, np.array(p) + OFF[n], vin, vout, kind, pitch, ev=TW.index(w), **kw)
    def sfx(s, t, p, vin, vout, kind):
        for n, b in s.b.items(): b.hit(t, np.array(p) + OFF[n], vin, vout, kind, None, ev=None)


PAIR = Pair()


def write_hops(hits, t_next, kind='lotus', i0=0):
    for k, h in enumerate(hits):
        PAIR.hit(h['t'], h['p'], h['vins'][0], h['vout'], kind, TW[i0 + k])
        t1 = hits[k + 1]['t'] if k + 1 < len(hits) else t_next
        PAIR.fly(h['t'], t1, h['p'], h['vout'])


LK = dict(stay=1e9, nkeep=14, R=RT, ext=(0.27, 0.27, 0.05), emin=0.22, emax=0.95)


def lin(x0, z0, t0, vx=V, vz=0.0):
    return (lambda t: x0 + vx * (t - t0)), (lambda t: z0 + vz * (t - t0))


# ============================================================ A: willow drop, lotus keys, into the gondola
X0, Z0 = 0.0, 0.0
H_DROP = 3.4; TD0 = math.sqrt(2 * H_DROP / G)
t1 = TT[0]
PAIR.rest(0.0, t1 - TD0, (X0, YT, Z0 + H_DROP)); PAIR.fly(t1 - TD0, t1, (X0, YT, Z0 + H_DROP), (0, 0, 0))
EV['willow'] = dict(x=X0, y=YT, z=Z0 + H_DROP, t_release=t1 - TD0)
SFX.append(['release', t1 - TD0])
iG = ti(2, 3.0)                                                  # C5/Eb5 at 8.0 s, then the gondola
T_GC = TT[iG] + 0.5                                              # catch in the gondola basket
xa, za = lin(X0, Z0, t1, vz=-0.12)
# gondola top (basket centre of the pair) - chosen ahead of the last key
XG, ZG = xa(T_GC) + 0.5, za(T_GC) - 0.15
hA, pA, vA = plan(list(TT[:iG + 1]) + [T_GC], (X0, YT, Z0), [np.array([0, 0, -G * TD0])], xa, za, lambda t: YT,
                  vxlim=(0.4, 2.6), **LK)
write_hops(hA, T_GC)
PAIR.sfx(T_GC, pA, vA, [0, 0, 0], 'basket')
# zip line: rope slope BETA; the basket accelerates from rest (gravity along the rope, pulley drag) and stops at a buffer
BETA = math.radians(17.0); VT, TAU = 2.3, 0.55
RD_ = np.array([math.cos(BETA), 0, -math.sin(BETA)])
T_G0 = T_GC + 0.2
iK3 = ti(3, 1.0)
T_GS = TT[iK3] - 0.72                                            # buffer stop; the pair flies on for 0.72 s
def gs(u): return VT * (u - TAU * (1 - math.exp(-u / TAU))) if u > 0 else 0.0
def gv(u): return VT * (1 - math.exp(-u / TAU)) if u > 0 else 0.0
tt = np.arange(T_G0, T_GS + 1e-9, 1 / 240.0)
pg0 = np.array(pA)
PAIR.rest(T_GC, T_G0, pg0); PAIR.path(tt, np.array([pg0 + RD_ * gs(t - T_G0) for t in tt]))
L_ROPE = gs(T_GS - T_G0); v_out = RD_ * gv(T_GS - T_G0)
p_stop = pg0 + RD_ * L_ROPE
EV['gondola'] = dict(top=pg0.tolist(), dir=RD_.tolist(), L=L_ROPE, t_catch=T_GC, t0=T_G0, t_stop=T_GS, vt=VT, tau=TAU, beta=BETA, v_stop=float(gv(T_GS - T_G0)))
SFX += [['basket', T_GC], ['zip', T_G0], ['buffer', T_GS]]
pB0, vB0 = free_to(PAIR, T_GS, p_stop, v_out, TT[iK3])

# ============================================================ B: lotus keys of bars 3-4, into the well bucket (tsurube lift)
iW = ti(4, 3.0)                                                  # C5/Ab4 at 16.1, then the bucket
T_WC = TT[iW] + 0.5
xb, zb = lin(pB0[0], pB0[2], TT[iK3], vz=-0.15)
hB, pB, vB = plan(list(TT[iK3:iW + 1]) + [T_WC], pB0, [vB0], xb, zb, lambda t: YT, vxlim=(0.4, 2.6), **LK)
write_hops(hB, T_WC, i0=iK3)
PAIR.sfx(T_WC, pB, vB, [0, 0, 0], 'bucket')
iR1 = ti(5, 1.0)
# the pulley lift: a bamboo spout fills the counter-bucket at the top; when it is heavier it sinks and hauls the pair up
H_LIFT = 2.0; T_FALLW = 0.7; V_OUT = np.array([1.25, 0, 0.12])
T_TIP = TT[iR1] - T_FALLW
t_up0 = T_WC + 0.3; t_up1 = T_TIP - 0.3
def lift(t):
    u = min(1.0, max(0.0, (t - t_up0) / (t_up1 - t_up0))); u = u * u * (3 - 2 * u); return np.array(pB) + np.array([0, 0, H_LIFT * u])
PAIR.rest(T_WC, t_up0, pB)
tt = np.arange(t_up0, t_up1 + 1e-9, 1 / 240.0)
PAIR.path(tt, np.array([lift(t) for t in tt])); PAIR.rest(t_up1, T_TIP, lift(t_up1))
p_tip = lift(t_up1)
EV['lift'] = dict(p_in=np.array(pB).tolist(), t_in=T_WC, t_up0=t_up0, t_up1=t_up1, H=H_LIFT, t_tip=T_TIP)
SFX += [['bucket', T_WC], ['pulley', t_up0], ['tip', T_TIP - 0.08]]
pR1, vR1 = free_to(PAIR, T_TIP, p_tip, V_OUT, TT[iR1])


def slide(times, idxs, p_land, v_in, alpha, crr, kind, k_land=0.85, tail=0.03, who='pair'):
    """the pair lands on an inclined bamboo xylophone slide and rolls down it; each note = the bar under each ball"""
    d = np.array([math.cos(alpha), 0, -math.sin(alpha)]); nrm = np.array([math.sin(alpha), 0, math.cos(alpha)])
    v0 = k_land * float(v_in @ d); assert v0 > 0.3, v0
    a = 5 / 7 * G * math.sin(alpha) - crr * G * math.cos(alpha)
    t0 = times[0]; t_end = times[-1] + tail
    def s(t): u = t - t0; return v0 * u + 0.5 * a * u * u
    tt = np.arange(t0, t_end + 1e-9, 1 / 240.0)
    if tt[-1] < t_end - 1e-9: tt = np.append(tt, t_end)
    PAIR.path(tt, np.array([p_land + d * s(t) for t in tt]))
    bars = []
    for j, (t, i) in enumerate(zip(times, idxs)):
        pc = p_land + d * s(t); vv = d * (v0 + a * (t - t0))
        if j == 0: PAIR.hit(t, pc, v_in, vv, kind + 'land', TW[i])
        else: PAIR.hit(t, pc, vv, vv, kind, TW[i])
        bars.append(dict(t=float(t), p=pc.tolist(), ev=i))
    v_lip = d * (v0 + a * (t_end - t0)); p_lip = p_land + d * s(t_end)
    RAMPS.append(dict(t0=float(t0), t1=float(t_end), p0=p_land.tolist(), alpha=alpha, crr=crr, a=a, v0=v0, v1=float(v0 + a * (t_end - t0)),
                      L=float(s(t_end)), lip=p_lip.tolist(), bars=bars, r=RT, k_land=k_land, n=nrm.tolist(), kind=kind))
    return p_lip, v_lip, t_end


i6 = ti(6, 1.0)
ALPHA_S, CRR_S = math.radians(4.0), 0.02
pl, vl, tl = slide(TT[iR1:i6], list(range(iR1, i6)), pR1, vR1, ALPHA_S, CRR_S, 'slide1')
pC0, vC0 = free_to(PAIR, tl, pl, vl, TT[i6])

# ============================================================ C: bar 6 keys, onto slide 2 (bar 7)
iR2 = ti(7, 1.0)
xc, zc = lin(pC0[0], pC0[2], TT[i6], vz=-0.45)
hC, pC, vC = plan(TT[i6:iR2 + 1], pC0, [vC0], xc, zc, lambda t: YT, vxlim=(0.4, 2.6), **LK)
write_hops(hC, TT[iR2], i0=i6)
i8 = ti(8, 1.0)
pl, vl, tl = slide(TT[iR2:i8], list(range(iR2, i8)), pC, vC, ALPHA_S, CRR_S, 'slide2')
pD0, vD0 = free_to(PAIR, tl, pl, vl, TT[i8])

# ============================================================ D: bar 8 keys; the last one (C4) is the latch lever; into cup A
iL = ti(8, 6.0); i9 = ti(9, 3.0)
T_CA = TT[iL] + 0.55
xd, zd = lin(pD0[0], pD0[2], TT[i8], vz=-0.3)
# see-saw: plank along +x, near end A (the pair's cup) down at TH0 at first; the far end B up, waiting for the moon-stone
TH0 = math.radians(15.0); LA, LB = 1.45, 1.1; CUP_D = 0.12
XA = xd(T_CA) + 0.4; ZA = zd(T_CA) - 0.35                         # pair centre resting in cup A (down position)
hD, pD, vD = plan(list(TT[i8:iL + 1]) + [T_CA], pD0, [vD0], xd, zd, lambda t: YT, vxlim=(0.4, 2.6), **LK)
write_hops(hD, T_CA, i0=i8)
for h in PAIR.b['silver'].hits:
    if h['ev'] == iL and h['kind'] == 'lotus': h['kind'] = 'latch'
for h in PAIR.b['gold'].hits:
    if h['ev'] == iL and h['kind'] == 'lotus': h['kind'] = 'latch'
PAIR.sfx(T_CA, pD, vD, [0, 0, 0], 'cupA')
SFX.append(['cupA', T_CA])
cupA_down = np.array(pD)
plankA_top = cupA_down - [0, 0, RT + CUP_D]                      # plank top under the cup centre
PIV = plankA_top + np.array([LA * math.cos(TH0), 0, LA * math.sin(TH0)])   # A is down: the pivot is up-right of it
def plank_pt(arm, th):
    """point on the plank top at signed distance arm from the pivot (+ = towards B) for plank angle th (+ = A down, B up)"""
    return PIV + np.array([arm * math.cos(th), 0, arm * math.sin(th)])
# with th = +TH0 the B end (arm +LB) is up and A (arm -LA) is down
B_up = plank_pt(LB, TH0); B_dn = plank_pt(LB, -TH0); A_dn = plank_pt(-LA, TH0); A_up = plank_pt(-LA, -TH0)
assert np.allclose(A_dn, plankA_top, atol=1e-6), (A_dn, plankA_top)

# ============================================================ the moon-stone: perch, latch, A-flat slab, D-flat plate on end B
MS = Ball('moon', RB)
iBa = 0; iBd = 1
T_LATCH = TT[iL] + 0.05
PLATE = 0.08
pB_land = B_up + np.array([0, 0, PLATE + RB])                   # moon-stone centre on the D-flat plate (end B up)
T_SLAB, T_PLATE = BS[iBa]['t'], BS[iBd]['t']
assert abs(T_PLATE - (BS[1]['t'])) < 1e-9
D_SB = T_PLATE - T_SLAB
# the slab stands on the channel lane, behind and a little upstream of end B; choose its height so the bounce is legal
best = None
for dxs in np.linspace(-1.2, 0.2, 15):
    for dzs in np.linspace(0.4, 2.2, 19):
        S = pB_land + np.array([dxs, YC - YT, dzs])
        vo = (pB_land - S + np.array([0, 0, 0.5 * G * D_SB ** 2])) / D_SB
        T_FALL = T_SLAB - T_LATCH - 0.04
        vin = np.array([0, 0, -G * T_FALL])
        e, n = bounce_e(vin, vo)
        tilt = math.degrees(math.acos(n[2]))
        if 0.3 <= e <= 0.9 and tilt <= 30 and np.linalg.norm(vo) < np.linalg.norm(vin):
            c = abs(e - 0.55) + 0.02 * tilt
            if best is None or c < best[0]: best = (c, S, vo, vin, e, tilt)
assert best is not None, 'no slab'
_, P_SLAB, VO_SLAB, VIN_SLAB, E_SLAB, TILT_SLAB = best
T_REL = T_LATCH + 0.04
P_PERCH = P_SLAB + np.array([0, 0, 0.5 * G * (T_SLAB - T_REL) ** 2])
MS.rest(0.0, T_REL, P_PERCH); MS.fly(T_REL, T_SLAB, P_PERCH, [0, 0, 0])
MS.hit(T_SLAB, P_SLAB, VIN_SLAB, VO_SLAB, 'slab', BS[iBa]['p'], ev=iBa)
MS.fly(T_SLAB, T_PLATE, P_SLAB, VO_SLAB)
vB_in = VO_SLAB + np.array([0, 0, -G * D_SB])
EV['perch'] = dict(p=P_PERCH.tolist(), t_rel=T_REL, t_latch=T_LATCH); EV['slab'] = dict(p=P_SLAB.tolist(), e=E_SLAB, tilt=TILT_SLAB)
SFX += [['latch', T_LATCH], ['gate', T_REL]]
# the heavy stone drives end B down: the plank turns from +TH0 to -TH0 in T_SW (end B moves at about the stone's speed)
vB_speed = float(np.linalg.norm(vB_in))
T_SW = 2 * TH0 * LB / (0.8 * vB_speed)
OMG = 2 * TH0 / T_SW
v_launch = OMG * LA * np.array([math.sin(TH0), 0, math.cos(TH0)])      # end A at the end of the swing (A up): up and forward
T_LA = T_PLATE + T_SW
MS.hit(T_PLATE, pB_land, vB_in, [0, 0, -0.8 * vB_speed * math.cos(TH0)], 'plate', BS[iBd]['p'], ev=iBd)
def th_at(t): return TH0 - OMG * min(max(0.0, t - T_PLATE), T_SW)
tt = np.arange(T_PLATE, T_LA + 1e-9, 1 / 240.0)
MS.path(tt, np.array([plank_pt(LB, th_at(t)) + np.array([0, 0, PLATE + RB]) for t in tt]))
# the pair rides cup A up and is thrown
cupA_up = A_up + np.array([0, 0, RT + CUP_D])
PAIR.rest(T_CA, T_PLATE, cupA_down)
tt = np.arange(T_PLATE, T_LA + 1e-9, 1 / 240.0)
PAIR.path(tt, np.array([plank_pt(-LA, th_at(t)) + np.array([0, 0, RT + CUP_D]) for t in tt]))
SFX += [['seesaw', T_PLATE], ['launch', T_LA]]
EV['seesaw'] = dict(piv=PIV.tolist(), la=LA, lb=LB, th0=TH0, t_hit=T_PLATE, t_sw=T_SW, cup_d=CUP_D, plate=PLATE, v_launch=v_launch.tolist())
pE0, vE0 = free_to(PAIR, T_LA, cupA_up, v_launch, TT[i9])
print('launch v', np.round(v_launch, 2), '|v|', round(float(np.linalg.norm(v_launch)), 2), 'stone in', round(vB_speed, 2), 'slab e', round(E_SLAB, 2), 'tilt', round(TILT_SLAB, 1))

# ---- the moon-stone rolls down end B, drops into the stone channel and rolls (constant speed) over the bass bars
p_now = plank_pt(LB, -TH0) + np.array([0, 0, PLATE + RB])
dB = np.array([math.cos(TH0), 0, -math.sin(TH0)])               # down the tilted plank towards the B tip
aB = 5 / 7 * G * math.sin(TH0)
s_tip = 0.32                                                    # rolls 0.32 m to the end of the plank
t_tip = math.sqrt(2 * s_tip / aB)
T_OFF = T_LA + 0.15 + t_tip
MS.rest(T_LA, T_LA + 0.15, p_now)
tt = np.arange(T_LA + 0.15, T_OFF + 1e-9, 1 / 240.0)
MS.path(tt, np.array([p_now + dB * 0.5 * aB * (t - T_LA - 0.15) ** 2 for t in tt]))
p_off = p_now + dB * s_tip; v_off = dB * aB * t_tip
# a stone chute catches it: inclined towards +x and +y (diagonal), then the long channel along +x (y = YC)
T_CH = T_OFF + 0.32
pch0, vch0 = free_to(MS, T_OFF, p_off, v_off, T_CH)


def roll_poly(ball, t0, p0, v_in, legs, t_end, k_land=0.9):
    """roll along a polyline of straight legs [(unit horizontal dir u, slope alpha, crr, length)], speed from the along-track
    part of v_in; returns the sample times/positions/speeds"""
    u0 = np.array([legs[0][0][0] * math.cos(legs[0][1]), legs[0][0][1] * math.cos(legs[0][1]), -math.sin(legs[0][1])])
    v = k_land * float(np.asarray(v_in) @ u0); assert v > 0.2, v
    ts = [t0]; ps = [np.array(p0, float)]; vs = [v]; t = t0; p = np.array(p0, float); leg = 0; s_leg = 0.0; dt = 1 / 240.0
    while t < t_end - 1e-9:
        u, al, cr, Ln = legs[min(leg, len(legs) - 1)]
        dvec = np.array([u[0] * math.cos(al), u[1] * math.cos(al), -math.sin(al)])
        a = 5 / 7 * G * math.sin(al) - cr * G * math.cos(al)
        h = min(dt, t_end - t)
        ds = v * h + 0.5 * a * h * h; v += a * h; p = p + dvec * ds; s_leg += ds; t += h
        if s_leg >= Ln and leg < len(legs) - 1: leg += 1; s_leg = 0.0
        ts.append(t); ps.append(p.copy()); vs.append(v)
    ball.path(ts, ps)
    return np.array(ts), np.array(ps), np.array(vs)


iBe = 2
T_LAST_B = BS[-2]['t']                                           # B-flat 2 (bar 14)
ALC = math.radians(2.0); CRC = 5 / 7 * math.tan(ALC)             # balanced slope: constant speed on the straight channel
u_diag = np.array([math.cos(math.radians(40)), math.sin(math.radians(40))])
L_DIAG = (YC - pch0[1]) / u_diag[1]
T_FALLF = 0.62                                                   # the last fall from the channel lip into the gong
T_LIPC = FINAL - T_FALLF
legs = [(u_diag, math.radians(3.0), 0.0, L_DIAG), (np.array([1.0, 0.0]), ALC, CRC, 1e9)]
CT, CP, CV = roll_poly(MS, T_CH, pch0, vch0, legs, T_LIPC, k_land=0.95)
RAMPS.append(dict(kind='channel', t0=T_CH, t1=T_LIPC, legs=[[l[0].tolist(), l[1], l[2], min(l[3], 999)] for l in legs], r=RB))
CBARS = []
for i in range(iBe, len(BS) - 1):
    t = BS[i]['t']; j = int(np.argmin(np.abs(CT - t))); assert abs(CT[j] - t) < 1e-3, (CT[j], t)
    pc = np.interp(t, CT, CP[:, 0]), np.interp(t, CT, CP[:, 1]), np.interp(t, CT, CP[:, 2])
    vv = float(np.interp(t, CT, CV)) * np.array([math.cos(ALC), 0, -math.sin(ALC)])
    MS.hits.append(dict(t=float(t), p=list(map(float, pc)), vin=vv.tolist(), vout=vv.tolist(), n=[math.sin(ALC), 0, math.cos(ALC)], e=1.0,
                        kind='cbar', pitch=BS[i]['p'], ev=i))
    CBARS.append(dict(t=float(t), p=list(map(float, pc)), pitch=BS[i]['p']))
p_lipc = CP[-1]; v_lipc = CV[-1] * np.array([math.cos(ALC), 0, -math.sin(ALC)])
pGONG, vGONG = free_to(MS, T_LIPC, p_lipc, v_lipc, FINAL)
MS.hit(FINAL, pGONG, vGONG, [0, 0, 0], 'gong', BS[-1]['p'], ev=len(BS) - 1)
MS.rest(FINAL, FINAL + 30, pGONG)
EV['channel'] = dict(t0=T_CH, p0=pch0.tolist(), lip=p_lipc.tolist(), t_lip=T_LIPC, v=float(CV[-1]), v0=float(CV[0]), bars=CBARS, diag=u_diag.tolist(), Ldiag=L_DIAG, path=CP[::12].tolist() + [CP[-1].tolist()])
EV['gong'] = dict(p=pGONG.tolist(), t=FINAL)
SFX += [['chute', T_CH], ['splash_big', FINAL]]
print('channel v0', round(CV[0], 2), 'v', round(CV[-1], 2), 'gong at', np.round(pGONG, 2))

# ============================================================ E: theme return (bars 9-10), into the shishi-odoshi
iS = ti(10, 3.0); i11 = ti(11, 1.0)
T_SC = TT[iS] + 0.5
xe, ze = lin(pE0[0], pE0[2], TT[i9], vz=-0.3)
# shishi-odoshi: bamboo tube on a pivot; the pair sits on the pad at the closed (short) end, which rests on a stone (down);
# a spout fills the long open end with water; when full it tips: the open end drops, the pad end swings up and throws the pair
SH_TH0 = math.radians(22.0); SH_L1, SH_L2 = 1.0, 1.6              # pad arm, open arm
XS_ = xe(T_SC) + 0.3; ZS_ = ze(T_SC) - 0.3
hE, pE, vE = plan(list(TT[i9:iS + 1]) + [T_SC], pE0, [vE0], xe, ze, lambda t: YT, vxlim=(0.4, 2.6), **LK)
write_hops(hE, T_SC, i0=i9)
PAIR.sfx(T_SC, pE, vE, [0, 0, 0], 'shishipad')
SFX.append(['pad', T_SC])
pad_c = np.array(pE)
SH_PAD = 0.1
pad_top = pad_c - [0, 0, RT + SH_PAD]
SH_PIV = pad_top + np.array([SH_L1 * math.cos(SH_TH0), 0, SH_L1 * math.sin(SH_TH0)])     # pad end down-left of the pivot
def sh_pt(arm, th): return SH_PIV + np.array([arm * math.cos(th), 0, arm * math.sin(th)])
T_SH = E['grid'][9] + 7 * E8                                       # tips on bar 10, eighth 8
T_SHS = 0.16
SH_OM = 2 * SH_TH0 / T_SHS
def sh_th(t): return SH_TH0 - SH_OM * min(max(0.0, t - T_SH), T_SHS)
T_SHL = T_SH + T_SHS
v_sh = SH_OM * SH_L1 * np.array([math.sin(SH_TH0), 0, math.cos(SH_TH0)])
PAIR.rest(T_SC, T_SH, pad_c)
tt = np.arange(T_SH, T_SHL + 1e-9, 1 / 240.0)
PAIR.path(tt, np.array([sh_pt(-SH_L1, sh_th(t)) + np.array([0, 0, RT + SH_PAD]) for t in tt]))
p_shl = sh_pt(-SH_L1, -SH_TH0) + np.array([0, 0, RT + SH_PAD])
EV['shishi'] = dict(piv=SH_PIV.tolist(), l1=SH_L1, l2=SH_L2, th0=SH_TH0, t_tip=T_SH, t_sw=T_SHS, pad=SH_PAD, v=v_sh.tolist(), t_catch=T_SC)
SFX += [['shishi', T_SHL]]
pF0, vF0 = free_to(PAIR, T_SHL, p_shl, v_sh, TT[i11])
print('shishi throw v', np.round(v_sh, 2), 'pair lands bar 11 at', np.round(pF0, 2))

# ============================================================ F: bars 11-14 and the final lotus-flower bells beside the gong
XF = pGONG[0] - 1.3; ZF = pGONG[2] + 0.55
YF = YT
TFb = TT[-1]
def blend(f, g, t, t0, t1):
    u = min(1.0, max(0.0, (t - t0) / (t1 - t0))); return (1 - u) * f(t) + u * g
xf0, zf0 = lin(pF0[0], pF0[2], TT[i11], vz=-0.12)
hF, pF, vF = plan(TT[i11:], pF0, [vF0], lambda t: blend(xf0, XF, t, TFb - 7.0, TFb), lambda t: blend(zf0, ZF, t, TFb - 7.0, TFb), lambda t: YF,
                  vxlim=(0.3, 2.6), **LK)
write_hops(hF, TFb, i0=i11)
PAIR.hit(TFb, pF, vF, [0, 0, 0], 'flower', TW[-1])
PAIR.rest(TFb, TFb + 30, pF)
EV['flowers'] = dict(c=np.array(pF).tolist())
print('pair final', np.round(pF, 2), 'gong', np.round(pGONG, 2))

# ============================================================ drops: spouts above glass bowls on the back bank
BALLS = {'silver': PAIR.b['silver'], 'gold': PAIR.b['gold'], 'moon': MS}
FPS = 30
tg = np.arange(0, FINAL + 4.5, 1 / FPS)
cx = np.array([0.5 * (BALLS['silver'].at(t)[0] + BALLS['gold'].at(t)[0]) for t in tg])
cz = np.array([0.5 * (BALLS['silver'].at(t)[2] + BALLS['gold'].at(t)[2]) for t in tg])
def smooth(Xa, k):
    Y = Xa.copy(); a = 1 - math.exp(-1 / FPS / k)
    for f in range(1, len(Xa)): Y[f] = Y[f - 1] + (Xa[f] - Y[f - 1]) * a
    for f in range(len(Xa) - 2, -1, -1): Y[f] = Y[f + 1] + (Y[f] - Y[f + 1]) * a
    return Y
CXs = smooth(cx, 0.6); CZs = smooth(cz, 1.2)
H_SP = 1.35; TF_D = math.sqrt(2 * H_SP / G)
BOWL_DX = 0.66
DROPS = []; cur = -1e9
for k, dr in enumerate(DR):
    t = dr['t']; n = len(dr['notes']); w = (n - 1) * BOWL_DX
    xc0 = float(np.interp(t, tg, CXs)) + 0.9
    x0 = max(xc0 - w / 2, cur + 0.75)
    if t >= FINAL - 1e-6: x0 = max(pGONG[0] + 0.9 - w / 2, cur + 0.75)
    zb_ = float(np.interp(t, tg, CZs)) + 1.25
    for j, nt in enumerate(dr['notes']):
        x = x0 + j * BOWL_DX; y = YD + (0.22 if j % 2 else -0.22) * (n > 3)
        top = np.array([x, y, zb_ - 0.012 * (nt['p'] - 60)])         # lower notes: bigger, lower bowls
        b = Ball(f'drop{k}_{j}', RD)
        pc = top + np.array([0, 0, RD]); p_sp = pc + np.array([0, 0, H_SP])
        b.rest(0.0, t - TF_D, p_sp); b.fly(t - TF_D, t, p_sp, [0, 0, 0])
        b.hit(t, pc, [0, 0, -G * TF_D], [0, 0, 0], 'bowl', nt['p'], ev=k)
        b.rest(t, t + 30, pc)
        DROPS.append(dict(name=b.name, t=float(t), top=top.tolist(), spout=p_sp.tolist(), pitch=nt['p'], k=k, j=j))
        BALLS[b.name] = b
    cur = x0 + w
EV['drops'] = DROPS; EV['h_sp'] = H_SP; EV['tf_drop'] = TF_D
EV['ramps'] = RAMPS
print('drops', len(DROPS), 'bowl x range', round(DROPS[0]['top'][0], 1), round(DROPS[-1]['top'][0], 1), 'pair x', round(cx[0], 1), round(cx[-1], 1))


def check():
    info = {}
    hs = [h for n in ('silver', 'gold', 'moon') for h in BALLS[n].hits if h['kind'] in ('lotus', 'latch', 'slab')]
    info['e_range'] = (round(min(h['e'] for h in hs), 3), round(max(h['e'] for h in hs), 3))
    allh = [h for n in ('silver', 'gold', 'moon') for h in BALLS[n].hits if np.linalg.norm(h['vout']) > 0]
    info['speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in allh), 3)
    info['tilt_max_deg'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for h in hs), 1)
    for n in ('silver', 'moon'):
        zz = [h['p'][2] for h in BALLS[n].hits]; xx = [h['p'][0] for h in BALLS[n].hits]
        info[n] = dict(hits=len(BALLS[n].hits), z=(round(min(zz), 2), round(max(zz), 2)), x=(round(min(xx), 1), round(max(xx), 1)))
    info['slides'] = [(r['kind'], round(math.degrees(r['alpha']), 2), round(r['v0'], 2), round(r['v1'], 2), round(r['L'], 2)) for r in RAMPS if 'alpha' in r]
    return info


if __name__ == '__main__':
    for k, v in check().items(): print(k, v)
    out = dict(G=G, R=dict(twin=RT, moon=RB, drop=RD), lanes=dict(twins=YT, channel=YC, drops=YD), off={k: v.tolist() for k, v in OFF.items()},
               balls={n: dict(r=b.r, seg=b.seg, hits=b.hits) for n, b in BALLS.items()}, ev=EV, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
