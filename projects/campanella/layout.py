"""La Campanella - 'The bell tower' (鐘の塔). Physics + story planner. Run after arrange.py; writes layout.json.

Story (every note = one ball striking one bell of that pitch; nothing else in the scene is a bell):
  intro   F0  the silver hero S and its copper twin K roll out of the tower gate and hop side by side round the foot of
              the tower, striking the octave tolls (S the upper note, K the lower). On the fermata both hit the two gate
              bells; S hops into the glass lift, K drops onto the start pedal -> the lift starts to climb.
  theme   F1, F2  the lift's trip arm opens one release gate after another as it passes: each gate lets out a bell
              ringer (silver = right hand, gold = left hand) that hops round its own spiral lane, every landing a bell.
  climb   F3  chromatic climb under the high D# bells (silver + gold ringer)
  fall    F4  'bell waterfall': the outermost ringer runs down the famous D#5-D#6-D#7-D#6 figure, 96 bells,
              while three gold ringers play the theme in the left hand; the lift climbs past them (they cross).
  finale  the lift reaches the belfry deck and starts the clock hand; it sweeps round the belfry, opening trapdoors
              one after another: a spoke of bells rings per chord. The lift floor tips, S rolls into the launcher,
              is fired at the giant bell and strikes it on the final chord -> every bell in the tower lights, fireworks.
Physics: one constant gravity G. Between contacts pure parabolas. Each bounce is an impulse along the bell's surface
normal (tangential velocity kept) with restitution emin <= e <= emax, so a bounce never adds speed. Lift motion and
the launcher are the only energy sources and both are visible mechanisms. A ball that has to wait (long rest in the
music) parks in a tipping cup (silent, padded) and rolls off it."""
import json, os, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = 7.0; R = 0.17
EMAX = 0.95; EMIN = 0.12; TILT = 38.0
NOTES = json.load(open(f'{HERE}/notes.json')); EVT = json.load(open(f'{HERE}/events.json'))
T0 = EVT['t0']; BEAT = EVT['beat']; FINAL = EVT['final']
NAMES = 'C C# D D# E F F# G G# A A# B'.split()
def M(t): return (t - T0) / BEAT
def nm(p): return f'{NAMES[p % 12]}{p // 12 - 1}'


# ------------------------------------------------------------------ global spiral angle (the camera follows it)
def omega(t):
    a = 0.6; b = 0.36                       # rad/s: floors / waterfall
    u = np.clip((t - 37.6) / 1.4, 0, 1); return a + (b - a) * (3 * u * u - 2 * u ** 3)
_tg = np.arange(0, 70, 0.002); _ph = np.concatenate([[0], np.cumsum(omega(_tg[:-1]) * 0.002)]) - 0.6 * 2.0 - 1.2
def PHI(t): return float(np.interp(t, _tg, _ph))


# ------------------------------------------------------------------ voices: who plays which note
FLOORS = [  # name, M from, M to, RH pool, LH pool
    ('F0', 0, 10.9, ['S', 'K'], ['S', 'K']),
    ('F1', 10.9, 23.0, ['R1a', 'R1b'], ['L1a', 'L1b', 'L1c']),
    ('F2', 23.0, 35.5, ['R2a', 'R2b'], ['L2a', 'L2b', 'L2c']),
    ('F3', 35.5, 49.0, ['R3a', 'R3b'], ['L3a', 'L3b']),
    ('F4', 49.0, 73.0, ['R4a', 'R4b'], ['L4a', 'L4b', 'L4c'])]
# lane radius, angle offset, height offset, descent rate (u/s) for the voices of the floors
LANE = {'a_R': (3.0, 0.13, 0.55, -0.10), 'b_R': (2.6, 0.34, 1.05, -0.10),
        'a_L': (2.15, -0.16, 0.0, -0.10), 'b_L': (1.65, -0.02, -0.25, -0.10), 'c_L': (2.55, -0.40, -0.55, -0.10)}
LANE_F4 = {'R4a': (3.85, 0.10, 0.6, -0.30), 'R4b': (3.3, 0.30, 1.1, -0.10)}
voice_notes = {}; floor_of = {}; hand_of = {}
last = {}
for idx, n in enumerate(NOTES):
    t, p, v, d, hand, sb, sg = n
    m = M(t)
    fl = [f for f in FLOORS if f[1] - 1e-6 <= m < f[2] - 1e-6]
    if not fl: continue                              # finale chords: spokes (below)
    f = fl[0]
    if f[0] == 'F0':
        same = [x for x in NOTES if abs(x[0] - t) < 1e-6]
        b = 'S' if p == max(x[1] for x in same) else 'K'
    else:
        pool = f[3] if hand == 'RH' else f[4]; b = None
        for c in pool:
            if last.get(c, -9) <= t - (0.17 if hand == 'RH' else 0.3): b = c; break
        assert b, ('no free ball', f[0], n)
    last[b] = t; voice_notes.setdefault(b, []).append(idx); floor_of[b] = f[0]; hand_of[b] = hand
SPOKE_NOTES = [i for i, n in enumerate(NOTES) if M(n[0]) >= 73.0 - 1e-6]


# ------------------------------------------------------------------ ball paths
class Ball:
    def __init__(s, name, kind): s.name = name; s.kind = kind; s.seg = []; s.hits = []; s.cups = []; s.gate = None; s.basket = None
    def rest(s, t0, t1, p): s.seg.append(dict(k='rest', t0=float(t0), t1=float(t1), p=list(map(float, p))))
    def fly(s, t0, t1, p0, v0): s.seg.append(dict(k='fly', t0=float(t0), t1=float(t1), p=list(map(float, p0)), v=list(map(float, v0))))
    def ride(s, t0, t1, ref, off): s.seg.append(dict(k='ride', t0=float(t0), t1=float(t1), ref=ref, off=list(map(float, off))))
    def roll(s, t0, t1, p0, d, a):  # rolling from rest along unit direction d with constant acceleration a
        s.seg.append(dict(k='roll', t0=float(t0), t1=float(t1), p=list(map(float, p0)), d=list(map(float, d)), a=float(a)))
    def shift(s, dz):
        for g in s.seg:
            if 'p' in g: g['p'][2] += dz
        for h in s.hits: h['p'][2] += dz
        for c in s.cups: c['p'][2] += dz
        if s.gate: s.gate['p'][2] += dz
        if s.basket: s.basket['p'][2] += dz


def bounce_e(vin, vout):
    d = vout - vin; n = d / max(1e-9, np.linalg.norm(d)); a = vin @ n
    return (-(vout @ n) / a if a < -1e-9 else 9.0), n


def lane_xy(r, ph): return np.array([r * math.cos(ph), r * math.sin(ph)])


def bell_r(p): return float(np.clip(0.22 * 2 ** (-(p - 60) / 24), 0.075, 0.42))


def bell_clear(P, c, ax, rb):
    """distance from ball centres P (n,3) to a bell (crown point c, axis ax pointing up out of the crown) minus R"""
    hgt = 1.5 * rb; d = P - c; s = -(d @ ax); sc = np.clip(s, 0, hgt)
    rad = np.linalg.norm(d + np.outer(s, ax), axis=1); rloc = rb * (0.45 + 0.55 * sc / hgt)
    out = np.where((s >= 0) & (s <= hgt), rad - rloc, np.hypot(np.maximum(0, rad - rloc), np.where(s < 0, -s, s - hgt)))
    return out - R


def solve_voice(ball, idxs, r, dphi, zoff, rate, z_base, t_start, emin=EMIN, emax=EMAX, gapmax=1.3, bias=0.22, K=70, marg=None):
    """beam search over landing heights for the hop sequence of one ball (see module doc)."""
    ts = [NOTES[i][0] for i in idxs]; ps = [NOTES[i][1] for i in idxs]
    pmean = np.mean(ps)
    def ztgt(t, p): return z_base + zoff + rate * (t - t_start) + bias * (p - pmean) / 12
    # angle schedule with lag for parked waits
    lag = 0.0; seq = []                              # entries: ('hit', k) / ('cup', k)
    angle = {}
    for k in range(len(ts)):
        angle[k] = PHI(ts[k]) + dphi - lag
        if k + 1 < len(ts) and ts[k + 1] - ts[k] > gapmax:
            lag += omega(ts[k]) * (ts[k + 1] - ts[k] - 1.0)
    # gate: roll off a release flap and fall onto the first bell
    vg = 0.55
    g0 = (ts[1] - ts[0]) if len(ts) > 1 else 0.5
    TAU = float(np.clip(0.5 * G * min(g0, gapmax) / 0.85 / G, 0.32, 0.7))
    p1 = np.array([*lane_xy(r, angle[0]), ztgt(ts[0], ps[0])])
    tang = np.array([-math.sin(angle[0]), math.cos(angle[0]), 0.0])
    gp = p1 - tang * vg * TAU + np.array([0, 0, 0.5 * G * TAU * TAU])
    vin0 = tang * vg + np.array([0, 0, -G * TAU])
    ball.gate = dict(p=gp.tolist(), t=float(ts[0] - TAU), tang=tang.tolist())
    ball.rest(0.0, ts[0] - TAU, gp); ball.fly(ts[0] - TAU, ts[0], gp, tang * vg)
    # minimum downward speed needed on arrival at each note so that the following hops stay physical
    def dzmin(g): return -0.3 - (1.4 if g < 0.5 else 2.4) * g
    U = [0.6] * len(ts)
    for k in range(len(ts) - 2, -1, -1):
        g = ts[k + 1] - ts[k]
        if g > gapmax: U[k] = 0.9; continue
        vlo = (dzmin(g) + 0.5 * G * g * g) / g
        U[k] = max(0.6, vlo / (emax * 0.97))
    Umax = [99.0] * len(ts)
    for k in range(len(ts) - 1):
        g = ts[k + 1] - ts[k]
        if g <= gapmax: Umax[k] = (G * g - 0.55) / (emin * 1.02) + 0.8
    if len(ts) > 1 and ts[1] - ts[0] <= gapmax: TAU = float(np.clip(max(TAU, U[0] * 1.08 / G), 0.32, 0.9))
    gp = p1 - tang * vg * TAU + np.array([0, 0, 0.5 * G * TAU * TAU])
    vin0 = tang * vg + np.array([0, 0, -G * TAU])
    ball.gate = dict(p=gp.tolist(), t=float(ts[0] - TAU), tang=tang.tolist())
    ball.seg = []
    ball.rest(0.0, ts[0] - TAU, gp); ball.fly(ts[0] - TAU, ts[0], gp, tang * vg)
    # beam: each state = (cost, p, vin, history)
    beam = [(0.0, p1, vin0, [])]
    k = 0
    while k < len(ts) - 1:
        g = ts[k + 1] - ts[k]
        if g > gapmax:                               # park in a cup, roll off, fall onto the next bell
            new = []
            g1 = 0.5; TC = max(0.42, U[k + 1] * 1.05 / G); vc = 0.5
            for cost, p, vin, hist in beam:
                ph0 = angle[k]; ph1 = ph0 + omega(ts[k]) * g1 * 0.8
                cxy = lane_xy(r, ph1); dxy = cxy - p[:2]
                for ee in np.linspace(0.35, 0.8, 8):
                    vz = ee * -vin[2]; dz = vz * g1 - 0.5 * G * g1 * g1
                    vo = np.array([dxy[0] / g1, dxy[1] / g1, vz]); e, nrm = bounce_e(vin, vo)
                    if not (emin <= e <= emax) or math.degrees(math.acos(nrm[2])) > TILT or vz - G * g1 > -0.8: continue
                    cp = np.array([*cxy, p[2] + dz])
                    tg = np.array([-math.sin(ph1), math.cos(ph1), 0.0])
                    pn = cp + tg * vc * TC + np.array([0, 0, -0.5 * G * TC * TC])
                    vn = tg * vc + np.array([0, 0, -G * TC])
                    c2 = cost + (pn[2] - ztgt(ts[k + 1], ps[k + 1])) ** 2 * 2
                    h2 = hist + [('hit', k, p, vin, vo, e, nrm), ('cup', k, cp, ts[k] + g1, ts[k + 1] - TC, tg * vc)]
                    new.append((c2, pn, vn, h2))
            assert new, ('cup fail', ball.name, k)
            new.sort(key=lambda s: s[0]); beam = new[:K]
            # the angle after the cup continues from the cup
            for j in range(k + 1, len(ts)): angle[j] = angle[j] - (angle[k + 1] - (ph1 + vc * TC / r))
            k += 1; continue
        new = []
        nxy = lane_xy(r, angle[k + 1])
        zt = ztgt(ts[k + 1], ps[k + 1])
        for cost, p, vin, hist in beam:
            dxy = nxy - p[:2]
            dzs = np.linspace(dzmin(g), 0.15 + 0.5 * g, 48)
            for dz in dzs:
                vz = (dz + 0.5 * G * g * g) / g
                vo = np.array([dxy[0] / g, dxy[1] / g, vz])
                e, nrm = bounce_e(vin, vo)
                if not (emin <= e <= emax): continue
                tilt = math.degrees(math.acos(max(-1, min(1, nrm[2]))))
                if tilt > TILT: continue
                va = vz - G * g
                if va > -max(0.55, U[k + 1]): continue     # must come down onto the next bell fast enough for what follows
                if -va > Umax[k + 1]: continue
                pn = np.array([*nxy, p[2] + dz])
                # clearance: leaving bell k (axis = its normal) and arriving at bell k+1 (axis ~ vertical)
                uu = np.array([0.04, 0.08, 0.13, 0.2]); uu = uu[uu < g - 0.04]
                if len(uu):
                    Pd = p + np.outer(uu, vo) + np.outer(0.5 * -G * uu * uu, [0, 0, 1])
                    if bell_clear(Pd, p - R * nrm, nrm, bell_r(ps[k])).min() < -0.004 + (marg or {}).get(-k, 0.0): continue
                    ua = g - uu
                    Pa = p + np.outer(ua, vo) + np.outer(0.5 * -G * ua * ua, [0, 0, 1])
                    if bell_clear(Pa, pn - np.array([0, 0, R]), np.array([0, 0, 1.0]), bell_r(ps[k + 1])).min() < (0.0 if g < 0.3 else 0.045) + (marg or {}).get(k + 1, 0.0): continue
                c2 = cost + 3.0 * (pn[2] - zt) ** 2 + 0.4 * (e - 0.8) ** 2 + 0.0004 * tilt ** 2 + 0.02 * max(0, vz - 4.5) ** 2
                new.append((c2, pn, vo + np.array([0, 0, -G * g]), hist + [('hit', k, p, vin, vo, e, nrm)]))
        assert new, ('hop fail', ball.name, k, round(g, 3), [round(ts[j+1]-ts[j],3) for j in range(max(0,k-3), min(len(ts)-1,k+3))], [round(u,2) for u in U[max(0,k-3):k+4]], [np.round(s_[2],2).tolist() for s_ in beam[:4]])
        new.sort(key=lambda s: s[0])
        # keep diversity: unique by rounded height
        seen = set(); bb = []
        for s_ in new:
            key = round(s_[1][2], 2)
            if key in seen: continue
            seen.add(key); bb.append(s_)
            if len(bb) >= K: break
        beam = bb; k += 1
    cost, p, vin, hist = beam[0]
    # write out
    for h in hist:
        if h[0] == 'hit':
            _, kk, pp, vi, vo, e, nrm = h
            ball.hits.append(dict(i=int(idxs[kk]), t=float(ts[kk]), p=pp.tolist(), n=nrm.tolist(), e=float(e), vin=vi.tolist(), vout=vo.tolist(), kind='bell'))
        else:
            _, kk, cp, ta, tr, vr = h
            ball.cups.append(dict(p=cp.tolist(), t_in=float(ta), t_out=float(tr), v=vr.tolist()))
    # segments: hit k -> (cup) -> note k+1
    for j, h in enumerate(hist):
        if h[0] != 'hit': continue
        _, kk, pp, vi, vo, e, nrm = h
        nxt = hist[j + 1] if j + 1 < len(hist) else None
        if nxt is not None and nxt[0] == 'cup':
            _, _, cp, ta, tr, vr = nxt
            ball.fly(ts[kk], ta, pp, vo); ball.rest(ta, tr, cp); ball.fly(tr, ts[kk + 1], cp, vr)
        else:
            ball.fly(ts[kk], ts[kk + 1], pp, vo)
    return p, vin, ts[-1], angle[len(ts) - 1]


def last_hop_to_basket(ball, p, vin, t, ang, r, out=0.5, dur=0.5):
    """final bounce (last note) -> silent landing in a wire basket just beside the lane"""
    best = None
    for dur in np.linspace(0.3, 0.9, 13):
        for o in (out, out * 0.6, out * 1.3):
            for adv in (0.6, 0.3, 0.9):
                bxy = lane_xy(r + o, ang + adv * float(np.linalg.norm(vin[:2])) * dur / r)
                for dz in np.linspace(0.0, -1.6, 33):
                    vo = np.array([(bxy[0] - p[0]) / dur, (bxy[1] - p[1]) / dur, (dz + 0.5 * G * dur * dur) / dur])
                    e, nrm = bounce_e(vin, vo)
                    if not (0.3 <= e <= EMAX) or vo[2] - G * dur > -0.8: continue
                    tilt = math.degrees(math.acos(max(-1, min(1, nrm[2]))))
                    if tilt > TILT: continue
                    c = tilt * 0.02 + abs(e - 0.6) + abs(dz) * 0.3
                    if best is None or c < best[0]: best = (c, dur, dz, vo, e, nrm, bxy)
    assert best, ('basket', ball.name)
    _, dur, dz, vo, e, nrm, bxy = best
    ball.hits.append(dict(i=int(ball._last_i), t=float(t), p=p.tolist(), n=nrm.tolist(), e=float(e), vin=vin.tolist(), vout=vo.tolist(), kind='bell'))
    ball.fly(t, t + dur, p, vo)
    bp = np.array([*bxy, p[2] + dz])
    ball.rest(t + dur, 80.0, bp); ball.basket = dict(p=bp.tolist(), t=float(t + dur))


def own_collisions(ball, idxs):
    """hops where the ball clips its own bells (exact tilted axes): {k: worst penetration}"""
    out = {}
    hs = ball.hits
    for j, h in enumerate(hs):
        nrm = np.array(h['n']); c = np.array(h['p']) - R * nrm; rb = bell_r(NOTES[h['i']][1])
        for g in ball.seg:
            if g['k'] != 'fly': continue
            if g['t1'] < h['t'] - 1.4 or g['t0'] > h['t'] + 1.4: continue
            uu = np.linspace(0, g['t1'] - g['t0'], 40); tt = g['t0'] + uu
            m = np.abs(tt - h['t']) > 0.035
            if not m.any(): continue
            P = np.array(g['p']) + np.outer(uu[m], g['v']) + np.outer(-0.5 * G * uu[m] ** 2, [0, 0, 1])
            d = bell_clear(P, c, nrm, rb).min()
            if d < -0.002:
                key = j if g['t0'] < h['t'] else -j
                out[key] = max(out.get(key, 0), -d)
    return out


def solve_voice_iter(name, kind, idxs, *a, **kw):
    marg = {}; good = None
    for it in range(6):
        ball = Ball(name, kind)
        try: res = solve_voice(ball, idxs, *a, marg=marg, **kw)
        except AssertionError as ex:
            if good is None: raise
            break
        col = own_collisions(ball, idxs)
        if good is None or sum(col.values()) < sum(good[2].values()): good = (ball, res, col)
        if not col: break
        for k_, v in col.items(): marg[k_] = marg.get(k_, 0.0) + v + 0.03
    ball, res, col = good
    if col: print('WARN own-bell clip', name, {k_: round(v, 3) for k_, v in col.items()})
    return ball, res


BALLS = {}
FLOOR_INFO = {}
# ---------------- intro twins (F0) -------------------------------------------------------------------
def solve_voice_r(ball, idxs, rfun, dphi, zoff, rate, z_base, t_start, **kw):
    """solve_voice for a lane whose radius changes with time (r = rfun(t))."""
    global lane_xy
    old = lane_xy
    ts = {round(NOTES[i][0], 5): NOTES[i][0] for i in idxs}
    def lx(r, ph, _t=[None]): return old(r, ph)
    # map the angle to the time to pick the radius: angles are monotonic in time for this lane
    tl = np.array(sorted(ts.values())); al = np.array([PHI(t) + dphi for t in tl])
    def lane2(r, ph):
        t = float(np.interp(ph, al, tl)); return old(rfun(t), ph)
    lane_xy = lane2
    try: out = solve_voice(ball, idxs, rfun(tl[0]), dphi, zoff, rate, z_base, t_start, **kw)
    finally: lane_xy = old
    return out


S = Ball('S', 'hero'); KB = Ball('K', 'twin')
for b, dphi, rst, rend in ((S, 0.07, 2.35, 0.5), (KB, -0.07, 3.0, 1.3)):
    idxs = voice_notes[b.name]; ts = [NOTES[i][0] for i in idxs]
    rf = (lambda a, b_, rs_, re_: (lambda t: rs_ - (rs_ - re_) * float(np.clip((t - a) / (b_ - a), 0, 1)) ** 1.6))(ts[0], ts[-1], rst, rend)
    p, vin, tl, ang = solve_voice_r(b, idxs, rf, dphi, 0.0, -0.03, 1.05, ts[0], bias=0.0, emax=0.97)
    b._end = (p, vin, tl, ang, rf(tl))
    BALLS[b.name] = b
# fermata: both land in the padded cradle on their bell's crown and stay. S's bell stands on the lift platform:
# its weight on the platform starts the lift.
T_FERM = S._end[2]
for B_ in (S, KB):
    p, vin, tl, ang, rr = B_._end
    B_.hits.append(dict(i=int(voice_notes[B_.name][-1]), t=float(tl), p=p.tolist(), n=[0.0, 0.0, 1.0], e=0.0, vin=vin.tolist(), vout=[0.0, 0.0, 0.0], kind='cradle'))
KB.rest(KB._end[2], 80.0, KB._end[0])
P_S0 = np.array(S._end[0]); CAR_PHI = S._end[3]
BELL_H = 0.42                                    # height of S's gate bell standing on the platform
CAR_Z0 = P_S0[2] - R - BELL_H                     # platform top
T_LIFT = T_FERM + 0.7

# ---------------- floors F1..F4 --------------------------------------------------------------------
def lane_of(name):
    if name in LANE_F4: return LANE_F4[name]
    return LANE[name[-1] + '_' + name[0]]
floor_balls = {}
for fname in ('F1', 'F2', 'F3', 'F4'):
    names = [b for b in voice_notes if floor_of[b] == fname]
    t_start = min(NOTES[voice_notes[b][0]][0] for b in names)
    floor_balls[fname] = names
    for b in names:
        r, dphi, zoff, rate = lane_of(b)
        kind = 'rh' if b[0] == 'R' else 'lh'
        idxs = voice_notes[b]
        ball, (p, vin, tl, ang) = solve_voice_iter(b, kind, idxs, r, dphi, zoff, rate, 0.0, t_start)
        ball._last_i = idxs[-1]
        last_hop_to_basket(ball, p, vin, tl, ang, r, out=0.42 if kind == 'rh' else -0.42)
        BALLS[b] = ball


def extent(names, lo_only=False):
    zs = []
    for b in names:
        B = BALLS[b]
        for g in B.seg:
            if g['k'] == 'fly':
                p = np.array(g['p']); v = np.array(g['v']); u = np.linspace(0, g['t1'] - g['t0'], 12)
                zs += list(p[2] + v[2] * u - 0.5 * G * u * u)
            elif g['k'] == 'rest': zs.append(g['p'][2])
        zs += [h['p'][2] - R - 0.55 for h in B.hits]          # bell body below the contact
    return min(zs), max(zs)


# stack the floors: each floor's lowest point above the one below + margin
z_top_prev = extent(['S', 'K'])[1]
FLOOR_Z = {}
for fname in ('F1', 'F2', 'F3', 'F4'):
    names = [b for b in floor_balls[fname] if b != 'R4a']
    lo, hi = extent(names)
    dz = z_top_prev + 0.35 - lo
    for b in floor_balls[fname]: BALLS[b].shift(dz)
    FLOOR_Z[fname] = dz
    z_top_prev = extent(names)[1]
    print(fname, 'base', round(dz, 2), 'span', round(extent(names)[0], 2), round(extent(names)[1], 2))

# ---------------- lift --------------------------------------------------------------------------------
Z_DECK = z_top_prev + 0.55
T_TOP = 55.85
LIFT = dict(z0=CAR_Z0, z1=Z_DECK, t0=T_LIFT, t1=T_TOP, phi0=CAR_PHI)
def lift_z(t):
    t0, t1, z0, z1 = LIFT['t0'], LIFT['t1'], LIFT['z0'], LIFT['z1']
    if t <= t0: return z0
    if t >= t1: return z1
    A = 0.8; L = t1 - t0; v = (z1 - z0) / (L - A)       # ease-in / ease-out over A seconds each end
    u = t - t0
    if u < A: return z0 + 0.5 * v / A * u * u
    if u > L - A: w = L - u; return z1 - 0.5 * v / A * w * w
    return z0 + 0.5 * v * A + v * (u - A)
LIFT['v'] = (Z_DECK - CAR_Z0) / (T_TOP - T_LIFT - 0.8)
PHF = PHI(56.6)                    # camera-facing angle at the finale
ROT = (PHF - CAR_PHI)                # the platform turns with the camera: the hero ends up facing it at the top
LIFT['rot'] = ROT
def lift_phi(t):
    u = (lift_z(t) - LIFT['z0']) / (LIFT['z1'] - LIFT['z0']); return u * ROT
S.rest(T_FERM, T_LIFT, P_S0)
S.ride(T_LIFT, T_TOP + 0.3, 'lift', (P_S0 - np.array([0, 0, CAR_Z0])).tolist())
# trip levers: each gate opens when the lift passes its lever
TRIPS = []
for b, B in BALLS.items():
    if B.gate and b not in ('S', 'K'):
        tr = B.gate['t']
        if tr < T_LIFT + 0.05:
            print('WARN gate before lift', b, round(tr, 2)); tr = T_LIFT + 0.05
        TRIPS.append(dict(ball=b, t=float(B.gate['t']), z_lever=float(lift_z(B.gate['t'])), gate=B.gate['p']))

# ---------------- finale: deck, clock hand, spokes, launcher, giant bell ---------------------------------
chords = sorted(set(round(NOTES[i][0], 4) for i in SPOKE_NOTES))
SPOKE_ANG = [-1.45, -1.08, -0.72, -0.38, 0.38, 0.72, 1.08, 1.45]
assert len(chords) == len(SPOKE_ANG), chords
TAU_D = 0.5
RING_RH = dict(z=Z_DECK + 2.05, r=[1.62, 2.04, 2.46, 2.88]); RING_LH = dict(z=Z_DECK + 1.15, r=[3.15, 3.55, 3.95, 4.35])
GB = dict(c=[0.0, 0.0, Z_DECK + 3.1], h=1.75, r_mouth=1.12, z_mouth=Z_DECK + 1.35)   # giant bell: crown at c, mouth below
SPOKES = []
SPB = []                                   # spoke bells: (k, note, crown point, rb, ball name)
for k, tc in enumerate(chords):
    ids = [i for i in SPOKE_NOTES if abs(NOTES[i][0] - tc) < 1e-3]
    rh = sorted([i for i in ids if NOTES[i][4] == 'RH'], key=lambda i: NOTES[i][1])
    lh = sorted([i for i in ids if NOTES[i][4] == 'LH'], key=lambda i: NOTES[i][1])
    if k == len(chords) - 1:          # final chord: the hero takes G#4 on the giant bell
        hero_i = [i for i in lh + rh if NOTES[i][1] == 68][0]
        if hero_i in lh: lh.remove(hero_i)
        else: rh.remove(hero_i)
    ang = PHF + SPOKE_ANG[k]
    names = []
    for ring, lst, sgn in ((RING_RH, rh[::-1], -1), (RING_LH, lh[::-1], 1)):
        for j, i in enumerate(lst):
            r_ = ring['r'][j]; z_ = ring['z'] - 0.12 * j * (1 if sgn > 0 else -1)
            c = np.array([*lane_xy(r_, ang), z_ + R])           # ball centre at contact
            name = f'D{k}_{len(names)}'; names.append(name)
            SPB.append(dict(k=k, i=i, p=c, rb=bell_r(NOTES[i][1]), name=name, ang=ang, t=tc))
    SPOKES.append(dict(k=k, t=float(tc), ang=float(ang), t_open=float(tc - TAU_D), balls=names))
# every falling drop ball: search the bounce (impulse along the crown normal) whose whole fall to the deck gutter
# keeps clear of all other bells, the hero's chute and the balls already lying in the gutter
landed = []; flights = []
crowns = [(np.array(sb['p']) - np.array([0, 0, R]), sb['rb'], sb['name']) for sb in SPB]
for sb in SPB:
    c = sb['p']; tc = sb['t']; ang = sb['ang']
    tg = np.array([-math.sin(ang), math.cos(ang), 0.0]); rad = np.array([math.cos(ang), math.sin(ang), 0.0])
    b = Ball(sb['name'], 'drop')
    p_tr = c + np.array([0, 0, 0.5 * G * TAU_D ** 2])
    b.rest(0.0, tc - TAU_D, p_tr); b.fly(tc - TAU_D, tc, p_tr, [0, 0, 0])
    vin = np.array([0, 0, -G * TAU_D])
    zg = Z_DECK + 0.12 + R
    best = None
    for vt in np.linspace(-2.2, 2.2, 23):
        for vr in np.linspace(-1.6, 1.6, 17):
            for vzo in (0.3, 0.7, 1.1, 1.6, 2.1):
                vo = tg * vt + rad * vr + np.array([0, 0, vzo])
                e, nrm = bounce_e(vin, vo)
                if not (0.15 <= e <= 0.9) or math.degrees(math.acos(nrm[2])) > TILT: continue
                a_ = 0.5 * G; tf = (vzo + math.sqrt(vzo * vzo + 2 * G * (c[2] - zg))) / G
                uu = np.linspace(0.03, tf, 40)
                P = c + np.outer(uu, vo) + np.outer(-0.5 * G * uu * uu, [0, 0, 1])
                m = 9.0
                for cp, rb_, nm_ in crowns:
                    if np.abs(P[:, :2] - cp[:2]).max(1).min() > 1.2 and nm_ != sb['name']: continue
                    m = min(m, bell_clear(P, cp, np.array([0, 0, 1.0]), rb_).min())
                pe = P[-1]; re_ = math.hypot(pe[0], pe[1]); ae = math.atan2(pe[1], pe[0])
                dang = (ae - PHF + math.pi) % (2 * math.pi) - math.pi
                if abs(dang) < 0.3 or re_ < 1.35 or re_ > 4.75: continue        # chute / giant bell / deck edge
                if any(np.linalg.norm(pe - q) < 2.3 * R for q in landed): continue
                bad = False
                for (t0_, c0_, v0_, tf0_) in flights:
                    tt = tc + uu; u2 = tt - t0_; mm = (u2 >= 0) & (u2 <= tf0_)
                    if mm.any():
                        Q = c0_ + np.outer(u2[mm], v0_) + np.outer(-0.5 * G * u2[mm] ** 2, [0, 0, 1])
                        if np.linalg.norm(Q - P[mm], axis=1).min() < 2 * R + 0.03: bad = True; break
                    # balls still waiting/falling onto their bell at these times are checked via their trapdoor column
                if bad: continue
                # the giant bell body (r < 1.2 below its crown)
                rr = np.hypot(P[:, 0], P[:, 1])
                if ((rr < GB['r_mouth'] + R + 0.05) & (P[:, 2] < GB['c'][2] + 0.3)).any(): continue
                sc = m - 0.02 * abs(e - 0.4) - 0.01 * abs(vt)
                if best is None or sc > best[0]: best = (sc, vo, e, nrm, tf, pe)
    assert best and best[0] > 0.0, ('drop', sb['name'], best and best[0])
    _, vo, e, nrm, tf, pe = best
    b.hits.append(dict(i=int(sb['i']), t=float(tc), p=c.tolist(), n=nrm.tolist(), e=float(e), vin=vin.tolist(), vout=vo.tolist(), kind='bell'))
    b.fly(tc, tc + tf, c, vo); b.rest(tc + tf, 80.0, pe); landed.append(pe); flights.append((tc, c, vo, tf))
    b.gate = dict(p=p_tr.tolist(), t=float(tc - TAU_D), tang=tg.tolist())
    BALLS[sb['name']] = b
# hero: lift top -> floor tips -> rolls out along the chute -> launcher -> giant bell
T_TIP = T_TOP + 0.3
CH_R0, CH_R1 = 0.5, 2.45
ph = PHF
d = np.array([math.cos(ph), math.sin(ph), 0.0])
z_floor_top = Z_DECK + (P_S0[2] - CAR_Z0)
drop = z_floor_top - (Z_DECK + 0.18 + R); Lc = CH_R1 - CH_R0
dirv = np.array([d[0] * Lc, d[1] * Lc, -drop]); Ls = np.linalg.norm(dirv); dirv /= Ls
acc = 5.0 / 7.0 * G * drop / Ls                     # rolling without slipping down the chute
t_roll = math.sqrt(2 * Ls / acc)
P_TOP = np.array([*lane_xy(0.5, CAR_PHI + ROT), Z_DECK + (P_S0[2] - CAR_Z0)])
S.roll(T_TIP, T_TIP + t_roll, P_TOP, dirv, acc)
P_LAUNCH = np.array([d[0] * CH_R1, d[1] * CH_R1, z_floor_top - drop])
T_LAUNCHER = T_TIP + t_roll
# strike point on the giant bell's sound bow (camera side), arriving at the top of the arc moving inward
T_HIT = chords[-1]
rb = GB['r_mouth'] + 0.02; zc = GB['z_mouth'] + 0.22
alpha = math.radians(18)                              # flank normal points outward and a little down
BETA = 0.42                                           # struck a little off the chute line, so it bounces off to the side
dB = np.array([math.cos(PHF + BETA), math.sin(PHF + BETA), 0.0])
nrm = np.array([dB[0] * math.cos(alpha), dB[1] * math.cos(alpha), -math.sin(alpha)])
contact = np.array([dB[0] * rb, dB[1] * rb, zc]); P_HIT = contact + R * nrm
dzL = P_HIT[2] - P_LAUNCH[2]; tauL = math.sqrt(2 * dzL / G)
vL = (P_HIT - P_LAUNCH) / tauL + np.array([0, 0, 0.5 * G * tauL])
T_FIRE = T_HIT - tauL
assert T_FIRE > T_LAUNCHER + 0.3, (T_FIRE, T_LAUNCHER)
S.rest(T_LAUNCHER, T_FIRE, P_LAUNCH); S.fly(T_FIRE, T_HIT, P_LAUNCH, vL)
vin = vL + np.array([0, 0, -G * tauL])
eH = 0.55; vo = vin - (1 + eH) * (vin @ nrm) * nrm
hero_i = [i for i in SPOKE_NOTES if NOTES[i][1] == 68 and abs(NOTES[i][0] - T_HIT) < 1e-3][0]
S.hits.append(dict(i=int(hero_i), t=float(T_HIT), p=P_HIT.tolist(), n=nrm.tolist(), e=eH, vin=vin.tolist(), vout=vo.tolist(), kind='giant'))
# fall into the goal cup on the deck
zg = Z_DECK + 0.10 + R; a = 0.5 * G; bq = -vo[2]; cq = zg - P_HIT[2]
tf = (-bq + math.sqrt(bq * bq - 4 * a * cq)) / (2 * a)
S.fly(T_HIT, T_HIT + tf, P_HIT, vo); GOAL = P_HIT + vo * tf + np.array([0, 0, -0.5 * G * tf * tf])
S.rest(T_HIT + tf, 80.0, GOAL)
print('hero: roll', round(t_roll, 2), 'wait in launcher', round(T_FIRE - T_LAUNCHER, 2), 'launch speed', round(np.linalg.norm(vL), 2),
      'hit speed', round(np.linalg.norm(vin), 2), 'out', round(np.linalg.norm(vo), 2))

SFX = [['liftgo', T_LIFT], ['liftstop', T_TOP], ['tip', T_TIP], ['fire', T_FIRE], ['goal', T_HIT + tf]]
for tr in TRIPS: SFX.append(['trip', tr['t']])
for sp in SPOKES: SFX.append(['door', sp['t_open']])
for b, B in BALLS.items():
    if B.basket: SFX.append(['basket', B.basket['t']])
    for c in B.cups: SFX.append(['cup', c['t_in']]); SFX.append(['cuptip', c['t_out']])
for B in BALLS.values():
    if B.kind == 'drop': SFX.append(['gutter', B.seg[-1]['t0']])


# ------------------------------------------------------------------ checks
def check():
    info = {}
    hs = [(n, h) for n, B in BALLS.items() for h in B.hits]
    cov = sorted(h['i'] for n, h in hs)
    from collections import Counter
    c = Counter(cov)
    info['notes'] = len(NOTES); info['hits'] = len(hs)
    info['notes_uncovered'] = sorted(set(range(len(NOTES))) - set(cov))
    info['notes_double'] = [i for i, k in c.items() if k > 1]
    info['dt_max'] = max(abs(h['t'] - NOTES[h['i']][0]) for n, h in hs)
    ee = [h['e'] for n, h in hs if h['kind'] != 'cradle']; info['e_range'] = (round(min(ee), 3), round(max(ee), 3))
    info['speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for n, h in hs), 3)
    info['cradle_hits'] = [(n, round(h['t'], 2)) for n, h in hs if h['kind'] == 'cradle']
    til = [math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for n, h in hs if h['kind'] == 'bell']
    info['tilt_max'] = round(max(til), 1)
    info['balls'] = len(BALLS); info['cups'] = sum(len(B.cups) for B in BALLS.values())
    info['deck'] = round(Z_DECK, 2); info['lift_v'] = round(LIFT['v'], 3)
    return info


if __name__ == '__main__':
    for k, v in check().items(): print(k, v)
    out = dict(G=G, R=R, balls={n: dict(kind=B.kind, seg=B.seg, hits=B.hits, cups=B.cups, gate=B.gate, basket=B.basket) for n, B in BALLS.items()},
               lift=LIFT, trips=TRIPS, spokes=SPOKES, ring_rh=RING_RH, ring_lh=RING_LH, giant=GB, deck=Z_DECK, bell_h=BELL_H, p_s0=P_S0.tolist(),
               launcher=P_LAUNCH.tolist(), goal=GOAL.tolist(), chute=dict(r0=CH_R0, r1=CH_R1, phi=PHF, drop=drop), phf=PHF,
               t=dict(ferm=T_FERM, lift=T_LIFT, top=T_TOP, tip=T_TIP, launcher=T_LAUNCHER, fire=T_FIRE, hit=T_HIT),
               floor_z=FLOOR_Z, floors={k: v for k, v in floor_balls.items()}, sfx=SFX, fireworks=[61.25, 61.85, 62.5, 63.1, 63.75, 64.3],
               phi=dict(t=_tg[::25].tolist(), v=_ph[::25].tolist()))
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
