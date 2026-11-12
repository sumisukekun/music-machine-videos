"""Mozart Symphony No. 40 - 'Night city chase'. Physics + story planner. Run after arrange.py; writes layout.json.

Cast
  A (red)    and B (blue): the two chasing marbles. They take turns playing violin I (phrase by phrase, later note by note).
             Each note = a landing on a glowing neon key standing on a rooftop. A long rest = the ball lands on the edge tile
             of a roof (that is the note), rolls along the roof and drops off its far edge onto the next key.
             Scale runs (bars 34-37) = rolling down a long xylophone fire-escape (B first, then A).
  train      the night train (violas): one ball rolling along the elevated rail behind the rooftops; every viola eighth is a
             rail bar under it (split bars for the two notes of a dyad). It starts from the station incline before bar 1.
  bass       the street ball (cellos + basses): bounces on bronze bells on the street-lamp posts in front of the buildings;
             bars 16-21 and 28-43 it rolls along a tram rail of bass bars.
Story: 0 s two marbles wait on two rooftops ("Will they meet?"); B drops first, A chases; bar 28 (forte) the whole skyline
lights up; bar 43 (general pause) blackout - both roll in the dark; the coda: they pass the notes back and forth and at the
last chord land together in the bowl bell on the final rooftop: fireworks.
Physics: one constant gravity G; free flight = parabolas; bounces = impulses along the key normal with e <= 0.97 (never
adds speed); rolling a = 5/7 g sin(a) - c_rr g cos(a); a landing on a roof / track keeps at most the along-track part."""
import json, os, math, pickle, hashlib
import numpy as np
from phys import Ball, seg_at, bounce_e, G, EMAX, EMIN, TILT
import phys
HERE = os.path.dirname(os.path.abspath(__file__))
_CF = '/tmp/mozart40_plan_cache.pkl'
_C = pickle.load(open(_CF, 'rb')) if os.path.exists(_CF) else {}


def plan(times, p0, vins, xt, zt, yt, **kw):
    times = np.asarray(times, float)
    key = hashlib.md5(repr((np.round(times, 6).tolist(), np.round(np.asarray(p0, float), 6).tolist(), [np.round(np.asarray(v, float), 6).tolist() for v in vins],
                            [(round(xt(t), 6), round(zt(t), 6), round(yt(t), 6)) for t in times], sorted((k, repr(v)) for k, v in kw.items()))).encode()).hexdigest()
    if key not in _C:
        r = None; err = None
        for W, NX, NZ in ((220, 17, 31), (400, 21, 41), (700, 25, 51)):
            try: r = phys.plan(times, p0, vins, xt, zt, yt, **dict(kw, W=W, NX=NX, NZ=NZ)); break
            except RuntimeError as e_: err = e_; print('retry', e_)
        if r is None: raise err
        _C[key] = (r, phys.plan.last); pickle.dump(_C, open(_CF, 'wb'))
    r, plan.last = _C[key]
    return r


E = json.load(open(f'{HERE}/events.json'))
MEL = json.load(open(f'{HERE}/mel.json')); ACC = json.load(open(f'{HERE}/acc.json')); BASS = json.load(open(f'{HERE}/bass.json'))
T0, Q, BAR, FINAL, GP, FORTE = E['t0'], E['q'], E['bar'], E['final'], E['gp'], E['forte']
BT = {int(k): v for k, v in E['bt'].items()}
RM, RT, RS = 0.2, 0.17, 0.28
YA, YB, YT, YS = 0.0, 0.85, 2.3, -1.25
ZT_OFF, ZS_OFF = 2.4, -2.7
V = 1.5
SLOPE = 0.14                                   # the whole city slowly steps down (m/s)
def xs(t): return V * (t - T0)
def zs(t): return -SLOPE * (t - T0)
SFX = []; EV = {}
TF = 0.32                                      # fall time from a roof lip to the next key
GAPMAX = 1.25
X_BOWL = xs(FINAL) + 0.6
Z_BOWL = zs(FINAL) - 1.0
Y_BOWL = 0.42


def ramp(ball, times, pitches, evs, p_land, v_in, alpha, crr, kind, k_land=0.9, tail=0.03, R=RM, sound_land=True):
    """the ball lands on an inclined surface at times[0] and rolls down it (a = 5/7 g sin a - c_rr g cos a); notes = bars under
    the ball at times[...]. Returns lip point / velocity / time and the info dict."""
    d = np.array([math.cos(alpha), 0, -math.sin(alpha)]); nrm = np.array([math.sin(alpha), 0, math.cos(alpha)])
    v0 = k_land * float(v_in @ d); assert v0 > 0.35, (ball.name, kind, v0, v_in)
    a = 5 / 7 * G * math.sin(alpha) - crr * G * math.cos(alpha)
    t0 = times[0]; t_end = times[-1] + tail
    assert v0 + a * (t_end - t0) > 0.3, (ball.name, kind, 'stops', v0, a)
    def s(t): u = t - t0; return v0 * u + 0.5 * a * u * u
    tt = np.arange(t0, t_end + 1e-9, 1 / 240.0)
    if tt[-1] < t_end - 1e-9: tt = np.append(tt, t_end)
    ball.path(tt, np.array([p_land + d * s(t) for t in tt]))
    bars = []
    for j, (t, p, ev) in enumerate(zip(times, pitches, evs)):
        pc = p_land + d * s(t); vv = d * (v0 + a * (t - t0))
        if j == 0: ball.hit(t, pc, v_in, vv, kind + 'land', p if sound_land else None, ev=ev)
        else: ball.hits.append(dict(t=float(t), p=pc.tolist(), vin=vv.tolist(), vout=vv.tolist(), n=nrm.tolist(), e=1.0, kind=kind, pitch=p, ev=ev))
        bars.append(dict(t=float(t), p=pc.tolist(), pitch=p, ev=ev))
    v_lip = d * (v0 + a * (t_end - t0)); p_lip = p_land + d * s(t_end)
    info = dict(who=ball.name, kind=kind, t0=float(t0), t1=float(t_end), p0=p_land.tolist(), alpha=alpha, crr=crr, a=a, v0=v0,
                v1=float(v0 + a * (t_end - t0)), L=float(s(t_end)), lip=p_lip.tolist(), bars=bars, r=R, k_land=k_land, n=nrm.tolist())
    return p_lip, v_lip, t_end, info


def free_to(ball, t0, p0, v0, t1):
    ball.fly(t0, t1, p0, v0); u = t1 - t0
    return np.array(p0) + np.array(v0) * u + np.array([0, 0, -0.5 * G * u * u]), np.array(v0) + np.array([0, 0, -G * u])


def roll_params(v0, T, vt=V, a_lim=(-0.12, 0.3), crr=0.025):
    """slope for a roof roll: aim for speed vt at the lip after T seconds"""
    a = min(a_lim[1], max(a_lim[0], (vt - v0) / max(T, 0.3)))
    sa = (a + crr * G) / (5 / 7 * G)
    return math.asin(max(0.0, min(0.3, sa))), crr


RAMPS = []


def sequence(ball, notes, start, xoff, zoff, ylane, xylo=(), final=None, yfun=None, keykw=None, vxl=(0.5, 3.2)):
    """notes: list of (t, pitch, ev).  start = (p_contact, v_in) at the first note.  xylo: list of (i0, i1) note-index ranges
    played rolling down a xylophone track.  Rolls are inserted in rests longer than GAPMAX."""
    keykw = keykw or {}
    T = [n[0] for n in notes]; n = len(notes)
    kind = ['b'] * n
    xr = {}
    for i0, i1 in xylo:
        for i in range(i0, i1 + 1): kind[i] = 'x'
        xr[i0] = i1
    for i in range(n - 1):
        if kind[i] == 'b' and T[i + 1] - T[i] > GAPMAX: kind[i] = 'r'
    yf = yfun or (lambda t: ylane)
    xt = lambda t: xs(t) + xoff
    zt = lambda t: zs(t) + zoff
    p, v = np.array(start[0], float), np.array(start[1], float)
    i = 0
    while i < n:
        if kind[i] == 'b':
            j = i
            while j < n and kind[j] == 'b': j += 1
            # j = first non-b note (or n): plan hops from i .. j-1, landing at T[j] (or final)
            if j == n:
                times = T[i:n]; fx_, fz_ = final
                w_ = lambda t: min(1.0, max(0.0, (t - (T[-1] - 3.0)) / 3.0))
                xt2 = lambda t: (1 - w_(t)) * xt(t) + w_(t) * fx_
                zt2 = lambda t: (1 - w_(t)) * zt(t) + w_(t) * fz_
                hits, pe, ve = plan(times, p, [v], xt2, zt2, yf, soft=300.0, **keykw)
                for k, h in enumerate(hits):
                    ball.hit(h['t'], h['p'], h['vins'][0], h['vout'], 'key', notes[i + k][1], ev=notes[i + k][2])
                    ball.fly(h['t'], T[i + k + 1], h['p'], h['vout'])
                p, v = pe, ve; i = n - 1
                ball.hit(T[-1], p, v, [0, 0, 0], 'bowl', notes[-1][1], ev=notes[-1][2])
                return p
            times = T[i:j + 1]
            # the landing on a roof edge / track: aim slightly ahead and a bit lower
            hits, pe, ve = plan(times, p, [v], xt, zt, yf, **keykw)
            for k, h in enumerate(hits):
                ball.hit(h['t'], h['p'], h['vins'][0], h['vout'], 'key', notes[i + k][1], ev=notes[i + k][2])
                ball.fly(h['t'], T[i + k + 1], h['p'], h['vout'])
            p, v = pe, ve; i = j
            continue
        if kind[i] == 'r':
            t_next = T[i + 1]; dur = t_next - T[i] - TF
            d0 = np.array([1.0, 0, 0]); v0 = 0.9 * max(0.4, float(v @ d0))
            al, crr = roll_params(v0, dur)
            k_land = 0.9
            if float(v @ np.array([math.cos(al), 0, -math.sin(al)])) * k_land < 0.4: k_land = 0.4 / float(v @ np.array([math.cos(al), 0, -math.sin(al)]))
            pl, vl, tl, info = ramp(ball, [T[i]], [notes[i][1]], [notes[i][2]], p, v, al, crr, 'roof', k_land=min(0.95, k_land), tail=dur)
            RAMPS.append(info)
            # the drop from the lip: pull the next landing towards the lane target by choosing nothing (pure physics)
            p, v = free_to(ball, tl, pl, vl, t_next)
            i += 1; continue
        if kind[i] == 'x':
            i1 = xr[i]; t_next = T[i1 + 1] if i1 + 1 < n else None
            gap = (t_next - T[i1]) if t_next is not None else 0.5
            tail = 0.03 if gap < 0.3 else gap - TF
            al = math.atan(SLOPE / V); crr = 5 / 7 * math.tan(al)
            evs = list(range(i, i1 + 1))
            pl, vl, tl, info = ramp(ball, T[i:i1 + 1], [notes[k][1] for k in evs], [notes[k][2] for k in evs], p, v, al, crr, 'xylo',
                                    k_land=min(0.95, 1.55 / max(0.5, float(v @ np.array([math.cos(al), 0, -math.sin(al)])))), tail=tail)
            RAMPS.append(info)
            p, v = free_to(ball, tl, pl, vl, t_next)
            i = i1 + 1; continue
    return p


# =================================================================== the two marbles
NA = [(m[0], m[1], k) for k, m in enumerate(MEL) if m[4] in ('A', 'AB')]
NB = [(m[0], m[1], k) for k, m in enumerate(MEL) if m[4] in ('B', 'AB')]
def find(lst, bar, beat):
    for i, (t, p, k) in enumerate(lst):
        if MEL[k][5] == bar and abs(MEL[k][6] - beat) < 1e-6: return i
    raise KeyError((bar, beat))
KEYKW = dict(stay=1e9, nkeep=14, R=RM, ext=(0.16, 0.24, 0.06), vxlim=(0.5, 3.2))
BALLS = {}
starts = {}
for name, NN, xoff, zoff, y0, xyl in (('A', NA, -0.7, 0.0, YA, [(find(NA, 36, 0), find(NA, 37, 3.5))]),
                                      ('B', NB, 0.7, 0.35, YB, [(find(NB, 33, 3.5), find(NB, 35, 3.5))])):
    b = Ball(name, RM)
    t1 = NN[0][0]; tdrop = 0.75
    x1 = xs(t1) + xoff; z1 = zs(t1) + zoff
    ptop = np.array([x1, y0, z1 + 0.5 * G * tdrop ** 2])
    b.rest(0.0, t1 - tdrop, ptop); b.fly(t1 - tdrop, t1, ptop, [0, 0, 0])
    EV['drop_' + name] = dict(p=ptop.tolist(), t=float(t1 - tdrop))
    SFX.append(['release', float(t1 - tdrop)])
    i_end = len(NN) - 1
    tfin0 = NN[find(NN, 298, 2)][0]
    yfun = (lambda t, y0=y0, tf0=tfin0: y0 + (Y_BOWL - y0) * min(1.0, max(0.0, (t - tf0) / (FINAL - tf0))))
    fx = X_BOWL + (-0.215 if name == 'A' else 0.215)
    pend = sequence(b, NN, (np.array([x1, y0, z1]), np.array([0, 0, -G * tdrop])), xoff, zoff, y0, xylo=xyl,
                    final=(fx, Z_BOWL), yfun=yfun, keykw=KEYKW)
    print(name, 'end at', np.round(pend, 3))
    BALLS[name] = b; starts[name] = pend
# the bowl bell: both land in it at the last chord, roll to its centre and come to rest touching each other
pa, pb = starts['A'], starts['B']
mid = 0.5 * (pa + pb); ctr = mid.copy(); ctr[2] = min(pa[2], pb[2])
u = (pb - pa); u[2] = 0; u /= np.linalg.norm(u)
BOWL_R = max(0.75, 0.5 * float(np.linalg.norm((pb - pa)[:2])) + RM + 0.25)
for name, sgn in (('A', -1), ('B', 1)):
    b = BALLS[name]; p0 = starts[name]; p1 = ctr + sgn * u * (RM + 0.004)
    tt = np.arange(FINAL, FINAL + 1.2 + 1e-9, 1 / 240.0)
    w = lambda t: 1 - (1 - min(1.0, (t - FINAL) / 1.2)) ** 2
    b.path(tt, np.array([p0 + (p1 - p0) * w(t) for t in tt]))
    b.rest(FINAL + 1.2, FINAL + 20, p1)
EV['bowl'] = dict(x=float(ctr[0]), y=float(ctr[1]), z=float(ctr[2] - RM), r=BOWL_R, pa=pa.tolist(), pb=pb.tolist())

# =================================================================== the night train (violas)
TR = Ball('train', RT)
AL_T = math.atan(SLOPE / V); CRR_T = 5 / 7 * math.tan(AL_T)
VT = V / math.cos(AL_T)
AL_E = math.radians(12.0); CRR_E = 0.02
A_E = 5 / 7 * G * math.sin(AL_E) - CRR_E * G * math.cos(AL_E)
T_E = VT / A_E; L_E = VT * VT / (2 * A_E)
tA = ACC[0][0]
d_t = np.array([math.cos(AL_T), 0, -math.sin(AL_T)]); n_t = np.array([math.sin(AL_T), 0, math.cos(AL_T)])
p_rail0 = np.array([xs(tA) + 0.3, YT, zs(tA) + ZT_OFF])
d_e = np.array([math.cos(AL_E), 0, -math.sin(AL_E)])
p_st = p_rail0 - d_e * L_E
t_rel = tA - T_E
TR.rest(0.0, t_rel, p_st)
tt = np.arange(t_rel, tA + 1e-9, 1 / 240.0); tt = np.append(tt, tA) if tt[-1] < tA - 1e-9 else tt
TR.path(tt, np.array([p_st + d_e * 0.5 * A_E * (t - t_rel) ** 2 for t in tt]))
T_END = FINAL + 4.6
tt = np.arange(tA, T_END + 1e-9, 1 / 120.0)
TR.path(tt, np.array([p_rail0 + d_t * VT * (t - tA) for t in tt]))
for k, (t, ps, v, d, m, b) in enumerate(ACC):
    pc = p_rail0 + d_t * VT * (t - tA)
    for pp in ps:
        TR.hits.append(dict(t=float(t), p=pc.tolist(), vin=(d_t * VT).tolist(), vout=(d_t * VT).tolist(), n=n_t.tolist(), e=1.0, kind='rail', pitch=pp, ev=k))
EV['train'] = dict(station=p_st.tolist(), t_rel=float(t_rel), rail0=p_rail0.tolist(), alpha=AL_T, crr=CRR_T, v=VT, alpha_e=AL_E, crr_e=CRR_E,
                   a_e=A_E, L_e=L_E, t_end=T_END)
SFX.append(['train_go', float(t_rel)])

# =================================================================== the street ball (cellos + basses)
SB = Ball('bass', RS)
NS = [(r[0], r[1], k) for k, r in enumerate(BASS)]
def fb(bar, beat):
    for i, r in enumerate(BASS):
        if r[4] == bar and abs(r[5] - beat) < 1e-6: return i
    raise KeyError((bar, beat))
tdrop = 0.8; t1 = NS[0][0]
pS = np.array([xs(t1) + 0.2, YS, zs(t1) + ZS_OFF + 0.5 * G * tdrop ** 2])
SB.rest(0.0, t1 - tdrop, pS); SB.fly(t1 - tdrop, t1, pS, [0, 0, 0])
EV['drop_bass'] = dict(p=pS.tolist(), t=float(t1 - tdrop))
SKW = dict(stay=1e9, nkeep=8, R=RS, ext=(0.24, 0.24, 0.12), emax=0.97, emin=0.3, dzr=(-1.2, 0.6), vxlim=(0.6, 3.0))
pend = sequence(SB, NS, (pS - [0, 0, 0.5 * G * tdrop ** 2], np.array([0, 0, -G * tdrop])), 0.2, ZS_OFF, YS,
                xylo=[(fb(16, 0), fb(20, 0)), (fb(28, 0), fb(42, 2))], final=(X_BOWL + 0.3, zs(FINAL) + ZS_OFF + 0.2), keykw=SKW)
SB.rest(FINAL, FINAL + 20, pend)
EV['bass_end'] = pend.tolist()
BALLS['train'] = TR; BALLS['bass'] = SB
EV['ramps'] = RAMPS
SFX += [['forte', FORTE], ['blackout', GP + 2 * Q], ['final', FINAL]]


def check():
    info = {}
    hs = [(n, h) for n, b in BALLS.items() for h in b.hits if h['kind'] == 'key']
    info['e_range'] = (round(min(h['e'] for n, h in hs), 3), round(max(h['e'] for n, h in hs), 3))
    allh = [h for b in BALLS.values() for h in b.hits if np.linalg.norm(h['vout']) > 0]
    info['speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in allh), 3)
    info['tilt_max_deg'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for n, h in hs), 1)
    for n, b in BALLS.items():
        zz = [h['p'][2] for h in b.hits]; xx = [h['p'][0] for h in b.hits]
        info[n] = dict(hits=len(b.hits), z=(round(min(zz), 2), round(max(zz), 2)), x=(round(min(xx), 1), round(max(xx), 1)))
    info['ramps'] = [(r['who'], r['kind'], round(r['t0'], 2), round(math.degrees(r['alpha']), 2), round(r['crr'], 3), round(r['v0'], 2), round(r['v1'], 2), round(r['L'], 1), round(r['k_land'], 2)) for r in RAMPS]
    return info


if __name__ == '__main__':
    for k, v in check().items(): print(k, v)
    out = dict(G=G, R=dict(A=RM, B=RM, train=RT, bass=RS), lanes=dict(A=YA, B=YB, train=YT, bass=YS), V=V, slope=SLOPE,
               balls={n: dict(r=b.r, seg=b.seg, hits=b.hits) for n, b in BALLS.items()}, ev=EV, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
