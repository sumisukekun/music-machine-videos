"""Grandfather's Clock - '100年時計の一生'. Physics + story planner. Run after arrange.py; writes layout.json.

Inside a tall cut-away grandfather clock, every melody note = one marble landing on one brass chime bar of that pitch.
The marbles are a family; the clock's own weights (descending at one constant speed = time) are their elevators.
  v1  1876  the escapement's first tick opens the hatch under the dial: the gold marble drops and plays verse 1 alone
  v2  1900  a rose marble drops out of her hopper; the two play verse 2 as a duet (left half / right half of the clock),
            end on the same G together: their two bars are the arms of a lever -> a small blue marble pops out of the
            cradle cup between them (birth) into the nest
  br  1925  the blue child plays the bridge alone; the parents ride down on the two clock weights, watching
  st  1950  the parents play again (stopped-short phrase); the child rides the left weight down for good
  ch  1976  chorus (tick-tock ping-pong between the parents); the clock gets old: dust, dark wood, dim lamps;
            the pendulum's swing dies away; the last G lands on the lever in the base: the child is catapulted out of
            the side hatch, rolls across the floor into a small new clock, which starts and plays the opening bar.
Physics: one constant gravity G; between contacts pure parabolas; at a chime bar the bar's normal is chosen so the
impulse is along the normal (tangential velocity kept) with restitution e <= EMAX (never gains speed). Catches (cups,
nest, weight cups) absorb the relative velocity; releases are trapdoors (start from the cup's own velocity).
Rolling on the floor / ramps: constant acceleration. The weights move at one constant speed VW."""
import json, os, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = 9.8; R = 0.13; RK = 0.095
EMAX = 0.96; EMIN = 0.3; TILT = 36.0
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
TN = np.array([l[0] for l in LEAD]); PN = [l[1] for l in LEAD]; NN = len(TN)
Q = E['q']
KEY_W = 0.17; KEY_T = 0.06          # chime bar width (x, along the run) and thickness
def key_w(p): return KEY_W
def key_d(p): return float(np.clip(0.40 - (p - 71) * 0.012, 0.30, 0.48))   # bar length (y): longer bar = lower pitch


class Ball:
    def __init__(s, name, r): s.name = name; s.r = r; s.seg = []; s.hits = []; s.events = []
    def rest(s, t0, t1, p): s.seg.append(dict(k='rest', t0=float(t0), t1=float(t1), p=list(map(float, p))))
    def fly(s, t0, t1, p0, v0): s.seg.append(dict(k='fly', t0=float(t0), t1=float(t1), p=list(map(float, p0)), v=list(map(float, v0))))
    def lin(s, t0, t1, p0, v0, a=(0, 0, 0)):   # constant-acceleration motion along a surface (rolling) or riding
        s.seg.append(dict(k='lin', t0=float(t0), t1=float(t1), p=list(map(float, p0)), v=list(map(float, v0)), a=list(map(float, a))))
    def end_t(s): return s.seg[-1]['t1'] if s.seg else 0.0


def seg_at(g, t):
    if g['k'] == 'pend':
        th = g['sgn'] * g['amp'] * math.sin(g['w'] * (t - g['t0']))
        pv = np.array(g['piv']); return pv + g['L'] * np.array([math.sin(th), 0, -math.cos(th)])
    p = np.array(g['p'])
    if g['k'] == 'rest': return p
    u = t - g['t0']
    if g['k'] == 'fly': return p + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
    return p + np.array(g['v']) * u + 0.5 * np.array(g['a']) * u * u


def seg_vel(g, t):
    if g['k'] == 'rest': return np.zeros(3)
    u = t - g['t0']
    if g['k'] == 'fly': return np.array(g['v']) + np.array([0, 0, -G * u])
    return np.array(g['v']) + np.array(g['a']) * u


def pos(ball, t):
    for g in ball.seg:
        if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9: return seg_at(g, t)
    if not ball.seg: return None
    if t < ball.seg[0]['t0']: return seg_at(ball.seg[0], ball.seg[0]['t0'])
    return seg_at(ball.seg[-1], ball.seg[-1]['t1'])


def bounce_e(vin, vout):
    d = vout - vin; n = d / max(1e-9, np.linalg.norm(d)); a = vin @ n
    return (-(vout @ n) / a if a < -1e-9 else 9.0), n


def frame_from_n(n):
    n = np.array(n, float); n /= np.linalg.norm(n)
    xa = np.cross([0, 1, 0], n); xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return np.stack([xa, ya, n], 1)


# ------------------------------------------------------------------ world registry for collision avoidance
KEYS = []          # dict(c, M, hw, hd, th, ball, t, i)
BOXES = []         # static obstacles: dict(c, M, h) oriented boxes (cups, hoppers ...)
_PATH = {}         # ball -> list of (t array, P array) chunks
_PCACHE = {}


def path_arrays(name):
    if name not in _PCACHE:
        ch = _PATH.get(name, [])
        if not ch: _PCACHE[name] = (np.zeros(0), np.zeros((0, 3)))
        else:
            t = np.concatenate([c[0] for c in ch]); P = np.concatenate([c[1] for c in ch]); o = np.argsort(t)
            _PCACHE[name] = (t[o], P[o])
    return _PCACHE[name]


def box_dist(P, c, M, h):
    rel = (P - c) @ M
    d = np.maximum(0, np.abs(rel) - np.array(h))
    return np.linalg.norm(d, axis=-1)


def keys_near(lo, hi, pad=0.8):
    out = []
    for K in KEYS:
        c = K['c']
        if lo[0] - pad <= c[0] <= hi[0] + pad and lo[2] - pad <= c[2] <= hi[2] + pad: out.append(K)
    return out


def arc_clear(ball, P, tt, extra_keys=(), margin=0.02):
    """sampled arc P at times tt: clear of bars, obstacles and the other balls"""
    lo, hi = P.min(0), P.max(0)
    for K in keys_near(lo, hi) + list(extra_keys):
        if box_dist(P, K['c'], K['M'], (K['hw'], K['hd'], K['th'])).min() < ball.r + margin: return 'keys'
    for Bx in BOXES:
        if box_dist(P, Bx['c'], Bx['M'], Bx['h']).min() < ball.r + margin: return 'box'
    for bn in B:
        if bn == ball.name: continue
        ts, PP = path_arrays(bn)
        if len(ts) == 0: continue
        a_, b_ = np.searchsorted(ts, tt[0] - 1e-6), np.searchsorted(ts, tt[-1] + 1e-6)
        if b_ <= a_: continue
        Pi = np.stack([np.interp(ts[a_:b_], tt, P[:, k]) for k in range(3)], 1)
        if np.linalg.norm(Pi - PP[a_:b_], axis=1).min() < ball.r + B[bn].r + 0.04: return 'ball'
    return None


def key_clear(c, M, hw, hd, th, ball, t, margin=0.015, extra=None):
    """a new bar must not sit in any already planned path (except its own ball right at its hit)"""
    h = (hw, hd, th)
    if extra is not None:     # the bar the ball is leaving: the boxes must be apart, and the ball resting on the new
        K, p_new = extra       # bar must clear the old one
        if box_dist(p_new[None, :], K['c'], K['M'], (K['hw'], K['hd'], K['th']))[0] < ball.r + 0.04: return False
        corners = np.array([[sx * hw, sy * hd, sz * th] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) @ M.T + c
        if box_dist(corners, K['c'], K['M'], (K['hw'], K['hd'], K['th'])).min() < 0.03 and abs(K['c'][2] - c[2]) < 0.15: return False
    for bn in B:
        ts, PP = path_arrays(bn)
        if len(ts) == 0: continue
        m = (np.abs(PP[:, 0] - c[0]) < 0.8) & (np.abs(PP[:, 2] - c[2]) < 0.8)
        if bn == ball.name: m &= np.abs(ts - t) > 0.04
        if not m.any(): continue
        if box_dist(PP[m], c, M, h).min() < B[bn].r + margin: return False
    for K in keys_near(c, c, 0.6):
        if abs(K['c'][2] - c[2]) < 0.12 and abs(K['c'][0] - c[0]) < K['hw'] + hw + 0.02 and abs(K['c'][1] - c[1]) < K['hd'] + hd: return False
    return True


def commit_path(ball, t0, t1):
    ts = np.arange(t0, t1, 1 / 120.0)
    if len(ts) == 0: return
    _PATH.setdefault(ball.name, []).append((ts, np.array([pos(ball, x) for x in ts])))
    _PCACHE.pop(ball.name, None)


def add_key(ball, i, p, n, vin, vout, kind='bar', e=None):
    n = np.array(n, float); n /= np.linalg.norm(n)
    w = key_w(PN[i]); c = p - ball.r * n - n * KEY_T / 2
    KEYS.append(dict(c=c, M=frame_from_n(n), hw=w / 2, hd=key_d(PN[i]) / 2, th=KEY_T / 2, ball=ball.name, t=float(TN[i]), i=int(i)))
    if e is None: e, _ = bounce_e(np.array(vin), np.array(vout))
    ball.hits.append(dict(i=int(i), t=float(TN[i]), p=list(map(float, p)), n=list(map(float, n)), e=float(e), kind=kind,
                          vin=list(map(float, vin)), vout=list(map(float, vout)), pitch=int(PN[i])))
    return len(KEYS) - 1


def vout_for(dx, dz, dt, dy=0.0): return np.array([dx / dt, dy / dt, dz / dt + 0.5 * G * dt])


def _parab(p, vo, u): return p[None, :] + vo[None, :] * u[:, None] + np.array([0, 0, -0.5 * G])[None, :] * (u * u)[:, None]


def run(ball, p, vin, notes, lane, vh=0.8, rate=-0.3, emax=EMAX, emin=EMIN, tilt=TILT, dirn=None, kind='bar',
        y_of=None, budget=1500, end_x=None, end_dir=None, final_vz=0.0):
    """ball is at contact centre p at the time of notes[0] with incoming velocity vin; hops through notes.
    Depth-first search with backtracking over (direction, horizontal step, landing height) per hop:
    the bounce must be physical (EMIN <= e <= EMAX, bar tilt <= TILT), the arc must miss every bar / obstacle / other
    ball, the new bar must not sit in any planned path. Preference: steady horizontal speed vh, heights near the line
    z0 + rate*(t - t0). Returns (p, vin, dirn) at the last note (its bar is NOT added)."""
    p = np.array(p, float); vin = np.array(vin, float)
    if dirn is None: dirn = 1 if vin[0] >= 0 else -1
    z0_, t0_ = p[2], TN[notes[0]]
    def z_line(t): return z0_ + rate * (t - t0_)
    own_keys = []; own_path = []; count = [0]; REJ = {}
    def rej(r): REJ[r] = REJ.get(r, 0) + 1

    def cands(k, p, vin, dirn):
        i, j = notes[k], notes[k + 1]; dt = TN[j] - TN[i]
        step0 = vh * dt
        zt = z_line(TN[j]); dzt = zt - p[2]
        if end_x is not None and k == len(notes) - 2:
            dirs = [1 if end_x > 0 else -1]
            steps0 = [x for x in np.arange(0.24, 1.2, 0.06) if (p[0] + dirs[0] * x) * dirs[0] >= abs(end_x)]
        elif end_dir is not None and k == len(notes) - 2:
            dirs = [end_dir]; steps0 = sorted(set(round(max(0.24, x), 3) for x in (step0, step0 * 0.7, step0 * 1.35, step0 * 0.45, step0 * 1.8)))
        else:
            dirs = [dirn, -dirn]; steps0 = sorted(set(round(max(0.24, x), 3) for x in (step0, step0 * 0.7, step0 * 1.35, step0 * 0.45, step0 * 1.8)))
        dzs = np.concatenate([dzt + np.linspace(0.05, 1.2, 24), dzt - np.linspace(0, 0.5 * G * dt * dt + 1.6, 120)])
        out = []
        dy = (y_of(j) - p[1]) if y_of else 0.0
        for sd in dirs:
            for st in steps0:
                x1 = p[0] + sd * st
                if not (lane[0] - 0.01 <= x1 <= lane[1] + 0.01): continue
                # vectorised cheap checks
                VZ = dzs / dt + 0.5 * G * dt
                VO = np.stack([np.full_like(dzs, sd * st / dt), np.full_like(dzs, dy / dt), VZ], 1)
                D = VO - vin[None, :]; Nn = D / np.linalg.norm(D, axis=1)[:, None]
                a_ = Nn @ vin; E_ = np.where(a_ < -1e-9, -(np.sum(VO * Nn, 1)) / np.where(a_ < -1e-9, a_, -1), 9)
                TL = np.degrees(np.arccos(np.clip(Nn[:, 2], -1, 1)))
                ok = (E_ >= emin) & (E_ <= emax) & (TL <= tilt) & (VZ - G * dt < -0.3)
                idx = np.where(ok)[0]
                idx = idx[np.argsort(np.abs(dzs[idx] - dzt))]
                chosen = []
                for q in idx:
                    dz = dzs[q]
                    if any(abs(dz - c_) < 0.12 for c_ in chosen): continue
                    if len(chosen) >= 4: break
                    vo = VO[q]; e = E_[q]; n = Nn[q]; tl = TL[q]
                    u = np.linspace(0.02, dt - 0.02, max(6, int(dt * 90)))
                    Pa = _parab(p, vo, u); tt = TN[i] + u
                    Kd = dict(c=p - ball.r * n - n * KEY_T / 2, M=frame_from_n(n), hw=key_w(PN[i]) / 2, hd=key_d(PN[i]) / 2, th=KEY_T / 2)
                    if box_dist(Pa[u > 0.06], Kd['c'], Kd['M'], (Kd['hw'], Kd['hd'], Kd['th'])).min() < ball.r + 0.01: rej('dep'); continue
                    if box_dist(Pa[u > 0.004], Kd['c'], Kd['M'], (Kd['hw'], Kd['hd'], Kd['th'])).min() < ball.r - 0.004: rej('dep'); continue
                    # exact check of the departing bar (its tilt is known now) against this run's earlier arcs
                    badd = False
                    for (ts0, PP0) in own_path:
                        mk = ts0 < TN[i] - 0.03
                        if mk.any() and box_dist(PP0[mk], Kd['c'], Kd['M'], (Kd['hw'], Kd['hd'], Kd['th'])).min() < ball.r + 0.003: badd = True; break
                    if badd: rej('exact'); continue
                    p1_ = p + vo * dt + np.array([0, 0, -0.5 * G * dt * dt])
                    cl_ = p1_ - np.array([0, 0, ball.r + KEY_T / 2])
                    if box_dist(Pa[u < dt - 0.07], cl_, np.eye(3), (KEY_W / 2 + 0.02, key_d(PN[j]) / 2 + 0.01, KEY_T / 2 + 0.05)).min() < ball.r + 0.005: rej('land'); continue
                    r_ = arc_clear(ball, Pa, tt, extra_keys=own_keys)
                    if r_: rej(r_); continue
                    p1 = p + vo * dt + np.array([0, 0, -0.5 * G * dt * dt])
                    c1 = p1 - np.array([0, 0, ball.r + KEY_T / 2])
                    hb = (key_w(PN[j]) / 2 + 0.03, key_d(PN[j]) / 2 + 0.02, KEY_T / 2 + 0.05)
                    if not key_clear(c1, np.eye(3), *hb, ball, TN[j], extra=(Kd, p1)): rej('keyclear'); continue
                    if box_dist(p[None, :], c1, np.eye(3), hb)[0] < ball.r + 0.01: rej('keyclear'); continue
                    bad = False
                    for (ts, PP) in own_path:
                        if box_dist(PP, c1, np.eye(3), hb).min() < ball.r + 0.015: bad = True; break
                    if bad: rej('ownpath'); continue
                    for K in own_keys:
                        if abs(K['c'][0] - c1[0]) < 0.3 and abs(K['c'][2] - c1[2]) < 0.15: bad = True; break
                    if bad: rej('ownkey'); continue
                    score = abs(dz - dzt) * 1.0 + abs(st - step0) * 0.8 + (0.6 if sd != dirn else 0) + 0.15 * tl / 30
                    out.append((score, sd, st, dz, vo, e, n, Kd, Pa, tt, p1, vo + np.array([0, 0, -G * dt])))
                    chosen.append(dz)
        out.sort(key=lambda c: c[0])
        return out[:8]

    sol = []

    deep = [0]
    def dfs(k, p, vin, dirn):
        if k == len(notes) - 1: return (p, vin, dirn)
        count[0] += 1; deep[0] = max(deep[0], k)
        if count[0] > budget: return None
        for c in cands(k, p, vin, dirn):
            score, sd, st, dz, vo, e, n, Kd, Pa, tt, p1, va = c
            if k == len(notes) - 2 and -va[2] < final_vz: continue
            own_keys.append(Kd); own_path.append((tt, Pa)); sol.append((notes[k], p, vin, vo, n, e, notes[k + 1]))
            r = dfs(k + 1, p1, va, sd)
            if r is not None: return r
            own_keys.pop(); own_path.pop(); sol.pop()
        return None

    r = dfs(0, p, vin, dirn)
    if r is None: print(REJ, 'deepest hop', deep[0], notes[deep[0]]); raise RuntimeError(f'run failed {ball.name} notes {notes[0]}..{notes[-1]} (nodes {count[0]})')
    for (i, p_, vin_, vo, n, e, j) in sol:
        add_key(ball, i, p_, n, vin_, vo, kind)
        ball.fly(TN[i], TN[j], p_, vo); commit_path(ball, TN[i], TN[j])
    print(f'  run {ball.name} {notes[0]}..{notes[-1]} nodes {count[0]}')
    return r


def hop_to(ball, i, p, vin, t1, p1, kind='bar'):
    """bounce off the bar of note i (at p, time TN[i]) so as to arrive at p1 at time t1; returns arrival velocity"""
    dt = t1 - TN[i]; vo = vout_for(*(p1 - p)[[0, 2]], dt, (p1 - p)[1])
    e, n = bounce_e(vin, vo)
    if not (EMIN * 0.5 <= e <= EMAX) or math.degrees(math.acos(max(-1, min(1, n[2])))) > 50:
        print('WARN hop_to', ball.name, i, round(e, 3), round(math.degrees(math.acos(n[2])), 1))
    add_key(ball, i, p, n, vin, vo, kind)
    ball.fly(TN[i], t1, p, vo); commit_path(ball, TN[i], t1)
    return vo + np.array([0, 0, -G * dt]), e


def best_hop_to(ball, i, p, vin, t1, target, search=0.4):
    """like hop_to but searches the catch point near target for a physical bounce (e in range, small tilt)"""
    best = None
    for dx in np.linspace(-search, search, 17):
        for dz in np.linspace(-search, search, 17):
            p1 = target + np.array([dx, 0, dz]); dt = t1 - TN[i]
            vo = vout_for(p1[0] - p[0], p1[2] - p[2], dt, p1[1] - p[1]); e, n = bounce_e(vin, vo)
            if not (EMIN <= e <= EMAX): continue
            tl = math.degrees(math.acos(max(-1, min(1, n[2]))))
            if tl > TILT: continue
            sc = np.linalg.norm([dx, dz]) + 0.01 * tl
            if best is None or sc < best[0]: best = (sc, p1)
    if best is None: print('WARN best_hop_to', ball.name, i); p1 = target
    else: p1 = best[1]
    va, e = hop_to(ball, i, p, vin, t1, p1)
    return p1, va


# ------------------------------------------------------------------ cups (catch + trapdoor release)
CUPS = []
def cup_catch_release(ball, t_in, p_in, v_in, t_out, ref=None, kind='cup'):
    """ball arrives at p_in (cup seat, ball centre) at t_in; rests (or rides with ref = weight) until t_out;
    trapdoor opens: ball falls from the seat with the seat's velocity."""
    cup = dict(ball=ball.name, t_in=float(t_in), t_out=float(t_out), p=list(map(float, p_in)), kind=kind, ref=ref,
               vin=list(map(float, v_in)))
    CUPS.append(cup)
    if ref is None:
        ball.rest(t_in, t_out, p_in); commit_path(ball, t_in, t_out)
        return np.array(p_in, float), np.zeros(3)
    w = WEIGHTS[ref]
    off = np.array(p_in) - w_pos(ref, t_in)
    ball.lin(t_in, t_out, p_in, [0, 0, -w['v']]); commit_path(ball, t_in, t_out)
    cup['off'] = list(map(float, off))
    return np.array(p_in) + np.array([0, 0, -w['v'] * (t_out - t_in)]), np.array([0, 0, -w['v']])


def drop_to(ball, t0, p0, v0, t1):
    """free fall from (t0, p0, v0) to t1; returns contact centre and incoming velocity"""
    dt = t1 - t0; ball.fly(t0, t1, p0, v0); commit_path(ball, t0, t1)
    return p0 + v0 * dt + np.array([0, 0, -0.5 * G * dt * dt]), v0 + np.array([0, 0, -G * dt])


def fall_time(h, v0z=0.0):
    # solve h = -v0z t + G t^2 / 2  (v0z <= 0 downward)
    a = 0.5 * G; b = -v0z; c = -h
    return (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)


# ------------------------------------------------------------------ the clock weights (one constant speed)
VW = 0.42
WEIGHTS = {}
def w_pos(name, t):
    w = WEIGHTS[name]; z = w['z0'] - w['v'] * t
    return np.array([w['x'], 0.0, max(z, w['zstop'])])


# ======================================================================================================
from assign import assign
B = {'gold': Ball('gold', R), 'rose': Ball('rose', R), 'kid': Ball('kid', RK)}
EVT = {}
SFX = []
LANE_FULL = (-1.0, 1.0); LANE_L = (-1.05, -0.22); LANE_R = (0.22, 1.05)
Z_TOP = 30.0
WX = 1.5                     # the two clock weights hang at x = -WX (left) and +WX (right)
CUP_SEAT = 0.05              # ball centre above the cup's seat plane
YG, YR = -0.3, 0.3           # depth lanes of the duets (gold in front, rose behind)


def hop_ok(ball, i, p, vin, t1, p1, extra_keys=()):
    """physical + collision-free bounce off the bar of note i (ball at p) arriving at p1 at t1? -> (vo, e, n) or None"""
    dt = t1 - TN[i]; vo = vout_for(p1[0] - p[0], p1[2] - p[2], dt, p1[1] - p[1]); e, n = bounce_e(vin, vo)
    if not (EMIN <= e <= EMAX): return None
    if math.degrees(math.acos(max(-1, min(1, n[2])))) > TILT: return None
    u = np.linspace(0.02, dt - 0.03, max(6, int(dt * 90))); Pa = _parab(p, vo, u)
    Kd = dict(c=p - ball.r * n - n * KEY_T / 2, M=frame_from_n(n), hw=key_w(PN[i]) / 2, hd=key_d(PN[i]) / 2, th=KEY_T / 2)
    if box_dist(Pa[u > 0.06], Kd['c'], Kd['M'], (Kd['hw'], Kd['hd'], Kd['th'])).min() < ball.r + 0.01: return None
    if box_dist(Pa[u > 0.004], Kd['c'], Kd['M'], (Kd['hw'], Kd['hd'], Kd['th'])).min() < ball.r - 0.004: return None
    if arc_clear(ball, Pa, TN[i] + u, extra_keys): return None
    return vo, e, n


def fall_clear(ball, t0, p0, v0, t1, skip_box=None):
    u = np.linspace(0.03, t1 - t0 - 0.02, max(5, int((t1 - t0) * 90))); P = _parab(p0, v0, u)
    lo, hi = P.min(0), P.max(0)
    for K in keys_near(lo, hi):
        if box_dist(P, K['c'], K['M'], (K['hw'], K['hd'], K['th'])).min() < ball.r + 0.02: return False
    for Bx in BOXES:
        if Bx is skip_box: continue
        if box_dist(P, Bx['c'], Bx['M'], Bx['h']).min() < ball.r + 0.02: return False
    for bn in B:
        if bn == ball.name: continue
        ts, PP = path_arrays(bn)
        a_, b_ = np.searchsorted(ts, t0 + 0.03), np.searchsorted(ts, t1 - 0.02)
        if b_ > a_:
            Pi = np.stack([np.interp(ts[a_:b_], t0 + u, P[:, k]) for k in range(3)], 1)
            if np.linalg.norm(Pi - PP[a_:b_], axis=1).min() < ball.r + B[bn].r + 0.04: return False
    return True


CUP_H = (0.21, 0.21, 0.16)       # cup half extents (around the seat centre, slightly below the ball centre)
def cup_box(seat): return dict(c=np.array(seat) + np.array([0, 0, -0.04]), M=np.eye(3), h=CUP_H, cup=True)


def cup_transfer(ball, i, p, vin, j, side=0, lane=(-1.08, 1.08), xpref=None, tfs=(0.4, 0.5, 0.32, 0.6, 0.7), y_key=None, pick=0, zpref=None):
    """note i -> bounce into a cup beside -> trapdoor -> falls straight onto the bar of note j (searched, collision-free)"""
    best = None; allo = []
    yk = p[1] if y_key is None else y_key
    for t1 in (0.42, 0.5, 0.36, 0.6, 0.3, 0.7, 0.25, 0.8):
        t_in = TN[i] + t1
        if t_in > TN[j] - 0.3: continue
        if len(allo) >= 60: break
        for dx in (0.4, 0.55, 0.3, 0.7, 0.85, 1.0, 0.25, 1.15):
            for sgn in ((side,) if side else (1, -1)):
                for dz in (-0.25, -0.1, -0.4, 0.05, -0.6, 0.2, -0.8, 0.35, -1.0, 0.5, -1.25, -1.5):
                    seat = p + np.array([sgn * dx, 0, dz]); seat[1] = yk
                    if not (lane[0] + 0.1 <= seat[0] <= lane[1] - 0.1): continue
                    r_ = hop_ok(ball, i, p, vin, t_in, seat)
                    if r_ is None: continue
                    cb = cup_box(seat)
                    if not key_clear(cb['c'], cb['M'], *cb['h'], ball, t_in, extra=None): continue
                    for tf in tfs:
                        t_out = TN[j] - tf
                        if t_out - t_in < 0.1: continue
                        pk = seat + np.array([0, 0, -0.5 * G * tf * tf])
                        if not fall_clear(ball, t_out, seat, np.zeros(3), TN[j]): continue
                        ck = pk - np.array([0, 0, ball.r + KEY_T / 2])
                        if not key_clear(ck, np.eye(3), KEY_W / 2 + 0.03, key_d(PN[j]) / 2 + 0.02, KEY_T / 2 + 0.02, ball, TN[j]): continue
                        sc = abs(dz + 0.2) + 0.3 * abs(tf - 0.42) + 0.2 * abs(t1 - 0.45) + (3 * abs(pk[0] - xpref) if xpref is not None else 0) \
                             + (3 * abs(pk[2] - zpref) if zpref is not None else 0)
                        allo.append((sc, t_in, tuple(seat), tf))
    if not allo: raise RuntimeError(f'no cup for {ball.name} {i}->{j}')
    allo.sort(); _, t_in, seat, tf = allo[min(pick * 7, len(allo) - 1)]; seat = np.array(seat)
    va, e = hop_to(ball, i, p, vin, t_in, seat)
    t_out = TN[j] - tf
    cup_catch_release(ball, t_in, seat, va, t_out)
    SFX.append(['cup', float(t_in)]); SFX.append(['latch', float(t_out)])
    BOXES.append(cup_box(seat))
    return drop_to(ball, t_out, seat, np.zeros(3), TN[j])


TIP_A = 5 / 7 * G * math.sin(math.radians(30)); TIP_T = 0.26     # rolling off a tipped cup (30 deg)
def tip_landing(seat, v_seat, dirx, tau):
    d = np.array([dirx * math.cos(math.radians(30)), 0, -math.sin(math.radians(30))]); a = TIP_A * d
    lip = np.array(seat) + np.array(v_seat) * TIP_T + 0.5 * a * TIP_T ** 2; vl = np.array(v_seat) + a * TIP_T
    return lip + vl * tau + np.array([0, 0, -0.5 * G * tau * tau]), vl + np.array([0, 0, -G * tau])


def tip_release(ball, t_tip, seat, v_seat, dirx):
    """the cup tips by 30 deg toward dirx: the ball rolls out over the lip (constant acceleration) then flies"""
    d = np.array([dirx * math.cos(math.radians(30)), 0, -math.sin(math.radians(30))])
    a = TIP_A * d
    ball.lin(t_tip, t_tip + TIP_T, seat, v_seat, a); commit_path(ball, t_tip, t_tip + TIP_T)
    p_lip = np.array(seat) + np.array(v_seat) * TIP_T + 0.5 * a * TIP_T ** 2
    v_lip = np.array(v_seat) + a * TIP_T
    return t_tip + TIP_T, p_lip, v_lip


# ---- v1 (1876): gold drops from the hatch under the dial (opened by the escapement's first tick)
g = B['gold']
p0 = np.array([-0.75, 0.0, Z_TOP])
T_REL = E["tb1"] - 2 * Q              # the escapement tick (pendulum extreme) 1.25 s before bar 1
TAU0 = TN[0] - T_REL - TIP_T
land_rel = tip_landing(np.zeros(3), np.zeros(3), 1, TAU0)[0]
p_hatch = p0 - land_rel
g.rest(0.0, T_REL, p_hatch); commit_path(g, 0.0, T_REL)
BOXES.append(cup_box(p_hatch))
t_l, lip, vl = tip_release(g, T_REL, p_hatch, np.zeros(3), 1)
pc, vin = drop_to(g, t_l, lip, vl, TN[0])
EVT['hatch'] = dict(t=float(T_REL), p=p_hatch.tolist())
SFX.append(['latch', float(T_REL)])
pc, vin, d = run(g, pc, vin, list(range(0, 20)), (-0.85, 0.85), vh=0.8, rate=-0.5, dirn=1, final_vz=3.0)
EVT['v1_end'] = pc.tolist()

# ---- v2 (1900): the rose marble's hopper opens; duet (gold left, rose right), unison G on the lever bars
_, hv2 = assign(20, 40, 1, fixed={20: 1, 21: 0, 39: 1}, unison=(40,))
own2 = dict(zip(range(20, 41), hv2))
NOTES_G2 = [k for k in range(21, 41) if own2[k] == 0 or k == 40]
NOTES_R2 = [k for k in range(20, 41) if own2[k] == 1 or k == 40]
print('v2 gold', NOTES_G2, 'rose', NOTES_R2)
# gold: note 19 -> cup (left) -> trapdoor -> note 21
pg, vg = cup_transfer(g, 19, pc, vin, NOTES_G2[0], tfs=(0.62, 0.55, 0.7), xpref=-0.6)
r_ = B['rose']
pr0 = np.array([0.85, YR, pg[2] - 0.1]); tf = 0.5
r_hop = pr0 + np.array([0, 0, 0.5 * G * tf * tf])
r_.rest(0.0, TN[20] - tf, r_hop); commit_path(r_, 0.0, TN[20] - tf)
BOXES.append(cup_box(r_hop))
EVT['rose_hopper'] = dict(t=float(TN[20] - tf), p=r_hop.tolist()); SFX.append(['latch', float(TN[20] - tf)])
prr, vrr = drop_to(r_, TN[20] - tf, r_hop, np.zeros(3), TN[20])
prr, vrr, _ = run(r_, prr, vrr, NOTES_R2, LANE_FULL, vh=0.3, rate=-0.4, dirn=-1, y_of=lambda j: YR)
pg, vg, _ = run(g, pg, vg, NOTES_G2[:-1], LANE_FULL, vh=0.3, rate=-0.4, dirn=1, y_of=lambda j: YG)
# unison: gold's last hop lands as close to rose's lever bar height as physics allows, 0.5-0.9 m beside her
best = None
i_g = NOTES_G2[-2]
for dz in np.arange(0.0, 2.5, 0.05):
    for dzs in (1, -1):
        z = prr[2] + dzs * dz
        for xg in np.arange(-0.95, 0.96, 0.1):
            dxg = abs(xg - prr[0])
            if dxg < 0.45: continue
            if True:
                tg = np.array([xg, YG, z])
                if hop_ok(g, i_g, pg, vg, TN[40], tg) is None: continue
                cbar = tg - np.array([0, 0, g.r + KEY_T / 2])
                if not key_clear(cbar, np.eye(3), KEY_W / 2 + 0.04, key_d(PN[40]) / 2 + 0.02, KEY_T / 2 + 0.05, g, TN[40]): continue
                # the lever rail toward gold's weight must be free of bars
                sgw = -1 if tg[0] <= prr[0] else 1
                line = np.linspace(tg, np.array([sgw * WX, 0, tg[2] - 0.6]), 30)
                if any(box_dist(line, K['c'], K['M'], (K['hw'], K['hd'], K['th'])).min() < g.r + 0.03 for K in keys_near(line.min(0), line.max(0))): continue
                liner = np.linspace(prr, np.array([-sgw * WX, 0, prr[2] - 0.6]), 30)
                if any(box_dist(liner, K['c'], K['M'], (K['hw'], K['hd'], K['th'])).min() < g.r + 0.03 for K in keys_near(liner.min(0), liner.max(0)) if not (K['ball'] == 'rose' and K['i'] == NOTES_R2[-2])): continue
                sc = dz + 0.3 * abs(dxg - 0.6)
                if best is None or sc < best[0]: best = (sc, tg)
    if best: break
assert best, 'no unison'
tg = best[1]
vg, _ = hop_to(g, i_g, pg, vg, TN[40], tg); pg = tg
print('unison: rose', np.round(prr, 2), 'gold', np.round(pg, 2), 'vin z', round(vg[2], 2), round(vrr[2], 2))
EVT['unison40'] = dict(g=pg.tolist(), r=prr.tolist())

# ---- the cradle: a small blue marble has been waiting in a cup between the two lever bars since the start;
#      the parents land together on the two lever bars (unison G) -> the inner ends kick the cradle up: birth
k_ = B['kid']
x_mid = 0.5 * (pg[0] + prr[0]); z_kc0 = min(pg[2], prr[2]) - 0.08
T40 = TN[40]
best = None
for dxc in (0.0, 0.12, -0.12, 0.25, -0.25):
    for dzc in (0.0, -0.2, -0.4, 0.15, -0.6):
        kc = np.array([x_mid + dxc, 0.0, z_kc0 + dzc])
        cbx = dict(c=kc + np.array([0, 0, -0.07]), M=np.eye(3), h=(0.13, 0.13, 0.05))
        if not key_clear(cbx['c'], cbx['M'], *cbx['h'], k_, T40): continue
        if box_dist(np.array([pg, prr]), cbx['c'], cbx['M'], cbx['h']).min() < R + 0.05: continue
        sgn_n = 1 if kc[0] < 0 else -1
        for tau in (0.5, 0.55, 0.45, 0.6, 0.4, 0.65, 0.7, 0.35):
            for dxn, sg_ in [(d_, s_) for d_ in (0.3, 0.4, 0.22, 0.5, 0.6) for s_ in (sgn_n, -sgn_n)]:
                for dzn in (0.45, 0.3, 0.6, 0.2, 0.75, 0.9, 0.1):
                    nest = kc + np.array([sg_ * dxn, 0, dzn])
                    if abs(nest[0]) > 1.05: continue
                    vk = np.array([(nest[0] - kc[0]) / tau, 0, (nest[2] - kc[2]) / tau + 0.5 * G * tau])
                    if not fall_clear(k_, T40 + 0.01, kc, vk, T40 + tau): continue
                    cb = cup_box(nest)
                    if not key_clear(cb['c'], cb['M'], *cb['h'], k_, T40 + tau): continue
                    # the drop from the nest to the first bridge bar must be clear too
                    if not fall_clear(k_, TN[41] - 0.5, nest, np.zeros(3), TN[41]): continue
                    best = (tau, nest, vk, kc); break
                if best: break
            if best: break
        if best: break
    if best: break
assert best, 'no cradle/nest'
tau, nest, vk, kc = best
k_.rest(0.0, T40, kc); commit_path(k_, 0.0, T40)
BOXES.append(dict(c=kc + np.array([0, 0, -0.07]), M=np.eye(3), h=(0.13, 0.13, 0.05), cradle=True))
best = (tau, nest, vk)
tau, nest, vk = best
k_.fly(T40, T40 + tau, kc, vk); commit_path(k_, T40, T40 + tau)
EVT['cradle'] = dict(p=kc.tolist(), t=float(T40), v=vk.tolist(), nest=nest.tolist(), t_nest=float(T40 + tau))
SFX.append(['pop', float(T40)]); SFX.append(['cup', float(T40 + tau)])
BOXES.append(cup_box(nest))

# ---- the parents' lever bars tip: each parent rolls down its lever rail and drops into the cup on a clock weight
W_OFF = 0.42                       # weight centre below the cup seat
RAIL_TH = math.radians(14)
A_RAIL = 5 / 7 * G * math.sin(RAIL_TH)
CATCH = {}; RAILS = []
side_g = -1 if pg[0] <= prr[0] else 1; side_r = -side_g
for ball, pp, vv, side in ((g, pg, vv_ := vg, side_g), (r_, prr, vrr, side_r)):
    tgt = np.array([side * WX, 0.0]); tf_ = 0.32
    lipxy = tgt.copy()
    for _ in range(30):
        hor = lipxy - pp[:2]; Lh = np.linalg.norm(hor)
        d = np.array([hor[0] / Lh * math.cos(RAIL_TH), hor[1] / Lh * math.cos(RAIL_TH), -math.sin(RAIL_TH)])
        L = Lh / math.cos(RAIL_TH); T_r = math.sqrt(2 * L / A_RAIL); vl = d * A_RAIL * T_r
        lipxy = tgt - vl[:2] * tf_
    t0 = TN[40] + 0.08                  # the lever settles, the ball starts to roll from rest
    ball.rest(TN[40], t0, pp); commit_path(ball, TN[40], t0)
    ball.lin(t0, t0 + T_r, pp, [0, 0, 0], A_RAIL * d); commit_path(ball, t0, t0 + T_r)
    lip = pp + d * L; vlip = d * A_RAIL * T_r
    seat = lip + vlip * tf_ + np.array([0, 0, -0.5 * G * tf_ * tf_])
    ball.fly(t0 + T_r, t0 + T_r + tf_, lip, vlip); commit_path(ball, t0 + T_r, t0 + T_r + tf_)
    va = vlip + np.array([0, 0, -G * tf_])
    ball.hits.append(dict(i=40, t=float(TN[40]), p=list(map(float, pp)), n=[0, 0, 1], e=0.0, kind='lever', vin=list(map(float, vv)),
                          vout=[0.0, 0.0, 0.0], pitch=int(PN[40])))
    KEYS.append(dict(c=pp - np.array([0, 0, ball.r + KEY_T / 2]), M=np.eye(3), hw=KEY_W / 2, hd=key_d(PN[40]) / 2, th=KEY_T / 2, ball=ball.name, t=float(TN[40]), i=40))
    RAILS.append(dict(ball=ball.name, p0=pp.tolist(), p1=lip.tolist(), t0=float(t0), t1=float(t0 + T_r), kind='lever'))
    CATCH[ball.name] = (t0 + T_r + tf_, seat, va, side)
    SFX.append(['wcup', float(t0 + T_r + tf_)])
    mid = 0.5 * (pp + lip); dd = lip - pp
    BOXES.append(dict(c=mid - np.array([0, 0, ball.r + 0.03]), M=frame_from_n(np.cross(np.cross(dd, [0, 0, 1]), dd) if abs(dd[2]) > 1e-6 else [0, 0, 1]),
                      h=(np.linalg.norm(dd) / 2, 0.08, 0.025), rail=True))
print('weights: gold', side_g, 'rose', side_r, {k: (round(v[0], 2), np.round(v[1], 2)) for k, v in CATCH.items()})

# ---- br (1925): the child drops from the nest and plays the bridge alone
TFK = 0.5
pk_, vk_ = None, None
k_.rest(T40 + tau, TN[41] - TFK, nest); commit_path(k_, T40 + tau, TN[41] - TFK)
SFX.append(['latch', float(TN[41] - TFK)])
pk_, vk_ = drop_to(k_, TN[41] - TFK, nest, np.zeros(3), TN[41])
EVT['nest_open'] = float(TN[41] - TFK)
_sn = (len(KEYS), len(k_.seg), len(k_.hits), len(_PATH.get('kid', [])))
for ex in (-0.5, -0.35, 0.5, 0.35, None):
    try:
        pk2, vk2, _ = run(k_, pk_, vk_, list(range(41, 63)), (-0.98, 0.98), vh=0.75, rate=-0.45, dirn=1 if pk_[0] < 0 else -1, end_x=ex)
        pk_, vk_ = pk2, vk2; break
    except RuntimeError as er:
        del KEYS[_sn[0]:]; del k_.seg[_sn[1]:]; del k_.hits[_sn[2]:]; del _PATH['kid'][_sn[3]:]; _PCACHE.clear()
        print('  kid end_x', ex, 'failed')
EVT['br_end'] = pk_.tolist()

# ---- the weights: each descends at its own constant speed; its cup carries a parent from the lever rail down to the
#      start of the 1950 duet, where the cup tips inward and the parent rolls out onto the first bar
z_s3_min = min(h['p'][2] for h in k_.hits)
Z_S4 = z_s3_min - 1.0





VL_FIX = json.load(open(f'{HERE}/vl.json'))['vl'] if os.path.exists(f'{HERE}/vl.json') else 0.0
WEIGHTS = {}
REL = {}
for ball, side, j, zoff in ((r_, side_r, 63, 0.0), (g, side_g, 64, -1.0)):
    tc, seat_c, va_c, _ = CATCH[ball.name]
    tau = 0.42; t_tip = TN[j] - tau - TIP_T
    z_land_rel = tip_landing(np.zeros(3), np.zeros(3), -side, tau)[0][2]
    # landing z = seat_c.z - v*(t_tip - tc + TIP_T + tau) + z_land_rel  ==  Z_S4 + zoff
    v = (seat_c[2] + z_land_rel - (Z_S4 + zoff)) / (t_tip - tc + TIP_T + tau)
    name = 'L' if side < 0 else 'R'
    if name == 'L' and VL_FIX: v = max(v, VL_FIX)       # the left weight also has to bring the child down in time
    WEIGHTS[name] = dict(x=side * WX, v=float(v), z0=float(seat_c[2] + v * tc), zstop=-99.0)
    REL[ball.name] = (name, t_tip, j, tau)
print('weight speeds', {k: round(w['v'], 3) for k, w in WEIGHTS.items()})


def w_seat(name, t):
    w = WEIGHTS[name]; return np.array([w['x'], 0.0, max(w['z0'] - w['v'] * t, w['zstop'])])


S4_START = {}
for ball in (r_, g):
    name, t_tip, j, tau = REL[ball.name]
    tc, seat_c, va_c, side = CATCH[ball.name]
    seat_c = w_seat(name, tc)
    ball.lin(tc, t_tip, seat_c, [0, 0, -WEIGHTS[name]['v']]); commit_path(ball, tc, t_tip)
    CUPS.append(dict(ball=ball.name, t_in=float(tc), t_out=float(t_tip), p=seat_c.tolist(), kind='weight', ref=name, vin=list(map(float, va_c))))
    t_l, lip, vl = tip_release(ball, t_tip, w_seat(name, t_tip), [0, 0, -WEIGHTS[name]['v']], -side)
    pj, vj = drop_to(ball, t_l, lip, vl, TN[j])
    SFX.append(['tip', float(t_tip)])
    S4_START[ball.name] = (pj, vj)
print('S4 start', {k: np.round(v[0], 2) for k, v in S4_START.items()})


# ---- the child: A (note 62) -> cup by the left weight column -> tips into the left weight's cup (gold has left it)
pk62 = np.array(EVT['br_end']); vk62 = k_.seg[-1]['v']
vk62 = np.array(vk62) + np.array([0, 0, -G * (k_.seg[-1]['t1'] - k_.seg[-1]['t0'])])
YK_W = 0.3                      # the child sits in the back half of the weight's dish
KCUP_H = (0.15, 0.15, 0.13)
best = None
for KW in ('L', 'R'):
    sk = 1 if KW == 'R' else -1
    t_free = REL['rose' if KW == 'R' else 'gold'][1] + TIP_T + 0.3      # this weight's cup is empty from here
    for tau in np.linspace(0.2, 0.75, 12):
        xk = WEIGHTS[KW]['x'] - tip_landing(np.zeros(3), np.zeros(3), sk, tau)[0][0]
        for t1 in np.arange(0.3, 1.0, 0.05):
            for dz in np.arange(0.2, -2.0, -0.1):
                seat = np.array([xk, YK_W, pk62[2] + dz])
                if hop_ok(k_, 62, pk62, vk62, TN[62] + t1, seat) is None: continue
                cb = dict(c=seat + np.array([0, 0, -0.03]), M=np.eye(3), h=KCUP_H)
                if not key_clear(cb['c'], cb['M'], *cb['h'], k_, TN[62] + t1): continue
                lip_l = tip_landing(seat, np.zeros(3), sk, tau)[0]
                t_land = (WEIGHTS[KW]['z0'] - lip_l[2]) / WEIGHTS[KW]['v']
                t_tip = t_land - TIP_T - tau
                if t_tip < max(t_free, TN[62] + t1 + 0.3): continue
                best = (t1, seat, tau, t_tip, t_land); break
            if best: break
        if best: break
    if best: break
print('child boards weight', KW)
if best is None:
    # move the cup sideways so that the tip lands in the weight column exactly
    raise RuntimeError('child cannot reach the weight')
t1, seatk, tauk, t_tipk, t_landk = best
va, e = hop_to(k_, 62, pk62, vk62, TN[62] + t1, seatk)
k_.rest(TN[62] + t1, t_tipk, seatk); commit_path(k_, TN[62] + t1, t_tipk)
CUPS.append(dict(ball='kid', t_in=float(TN[62] + t1), t_out=float(t_tipk), p=seatk.tolist(), kind='tipcup', ref=None, vin=list(map(float, va)), tip=sk))
BOXES.append(dict(c=seatk + np.array([0, 0, -0.03]), M=np.eye(3), h=KCUP_H, cup=True)); SFX.append(['cup', float(TN[62] + t1)]); SFX.append(['tip', float(t_tipk)])
t_l, lip, vl = tip_release(k_, t_tipk, seatk, np.zeros(3), sk)
pkw, vkw = drop_to(k_, t_l, lip, vl, t_landk)
SFX.append(['wcup', float(t_landk)])

# ---- st / ch (1950 -> 1976): the parents' duet to the end (who plays what: dynamic programming, see assign.py)
_, h45 = assign(63, 114, 1, fixed={63: 1, 64: 0, 114: 0}, unison=(77,))
OWN = dict(zip(range(63, 115), h45))
SEQ = {'gold': [k for k in range(63, 115) if OWN[k] == 0 or k == 77], 'rose': [k for k in range(63, 115) if OWN[k] == 1 or k == 77]}
print('duet gold', SEQ['gold']); print('duet rose', SEQ['rose'])
YL = {'gold': YG, 'rose': YR}


def split_runs(seq):
    """consecutive notes with gaps <= 1.3 s form a run; between runs: a cup transfer"""
    runs = [[seq[0]]]
    for a, b in zip(seq, seq[1:]):
        if TN[b] - TN[a] < 1.2: runs[-1].append(b)
        else: runs.append([b])
    return runs


def chunk(runs, n=5):
    out = []
    for rn in runs:
        if len(rn) <= n + 2: out.append((rn, True)); continue
        k = 0
        while k < len(rn) - 1:
            part = rn[k:k + n + 1]
            out.append((part, k == 0)); k += n
    return out


RUNS = {n: chunk(split_runs(SEQ[n])) for n in SEQ}
STATE = {n: (S4_START[n][0], S4_START[n][1], 1 if n == 'gold' else -1) for n in SEQ}
PIECES = []
for n in SEQ:
    for k, (rn, fresh) in enumerate(RUNS[n]): PIECES.append((TN[rn[0]], n, k))
PIECES.sort()


def snapshot():
    return dict(K=len(KEYS), BX=len(BOXES), S=len(SFX), C=len(CUPS),
                b={n: (len(b.seg), len(b.hits), len(_PATH.get(n, []))) for n, b in B.items()}, st=dict(STATE))


def restore(sn):
    del KEYS[sn['K']:]; del BOXES[sn['BX']:]; del SFX[sn['S']:]; del CUPS[sn['C']:]
    for n, (a_, b_, c_) in sn['b'].items():
        del B[n].seg[a_:]; del B[n].hits[b_:]
        if n in _PATH: del _PATH[n][c_:]
    _PCACHE.clear(); STATE.clear(); STATE.update(sn['st'])


ZFIN = {}
VARS = [(0.42, -0.35), (0.55, -0.35), (0.32, -0.45), (0.5, -0.25), (0.38, -0.55), (0.6, -0.45)]
SNAP = [None] * len(PIECES); TRY = [0] * len(PIECES)
q = 0
while q < len(PIECES):
    t0, n, k = PIECES[q]
    if SNAP[q] is None: SNAP[q] = snapshot()
    else: restore(SNAP[q])
    ball = B[n]; rn, fresh = RUNS[n][k]; p, v, d = STATE[n]
    vh, rate = VARS[TRY[q] % len(VARS)]
    try:
        if k > 0 and fresh:                          # cup transfer from the end of the previous run
            prev = RUNS[n][k - 1][0][-1]
            zp = None
            if rn[0] == 113: zp = STATE['gold'][0][2] - 0.9
            if rn[0] == 114: zp = STATE['rose'][0][2] + 0.15
            p, v = cup_transfer(ball, prev, p, v, rn[0], y_key=YL[n], pick=TRY[q], xpref=(-sk * 0.25 if rn[0] == 114 else (0.75 * -sk if rn[0] == 113 else None)), zpref=zp,
                                tfs=((0.4, 0.5, 0.32, 0.6, 0.7, 0.8, 0.9) if rn[0] == 114 else (0.4, 0.5, 0.32, 0.6, 0.7)))
        if len(rn) > 1:
            p, v, d = run(ball, p, v, rn, LANE_FULL, vh=vh, rate=rate, dirn=d, y_of=lambda j, n=n: YL[n], budget=600)
        STATE[n] = (p, v, d)
        print('  piece', n, rn[0], '..', rn[-1], 'z', round(p[2], 2), 'try', TRY[q])
        q += 1
        if q < len(PIECES): SNAP[q] = None
    except RuntimeError as ex:
        print('  piece failed', n, rn[0], ex)
        TRY[q] += 1
        if TRY[q] >= 3:                              # back up to this ball's previous piece and try it differently
            TRY[q] = 0; SNAP[q] = None
            q2 = max([x for x in range(q) if PIECES[x][1] == n] or [q - 1])
            if TRY[q2] >= 8: q2 = q - 1
            for x in range(q2 + 1, q + 1): TRY[x] = 0
            q = q2
            assert q >= 0, 'duet planning failed'
            TRY[q] += 1
            if TRY[q] > 14: raise
EVT['duet_end'] = {n: STATE[n][0].tolist() for n in STATE}


# ---- rose: after her last note she bounces into a cup by the base and stays there
def final_cup(ball, i, p, vin):
    for t1 in (0.45, 0.55, 0.38, 0.65):
        for dx in (0.45, 0.6, 0.35, 0.75):
            for sgn in (1, -1):
                for dz in (-0.3, -0.15, -0.45, 0.0):
                    seat = p + np.array([sgn * dx, 0, dz])
                    if abs(seat[0]) > 1.0: continue
                    if hop_ok(ball, i, p, vin, TN[i] + t1, seat) is None: continue
                    cb = cup_box(seat)
                    if not key_clear(cb['c'], cb['M'], *cb['h'], ball, TN[i] + t1): continue
                    va, e = hop_to(ball, i, p, vin, TN[i] + t1, seat)
                    ball.rest(TN[i] + t1, 80.0, seat); commit_path(ball, TN[i] + t1, 80.0)
                    CUPS.append(dict(ball=ball.name, t_in=float(TN[i] + t1), t_out=80.0, p=seat.tolist(), kind='cup', ref=None, vin=list(map(float, va))))
                    BOXES.append(cup_box(seat)); SFX.append(['cup', float(TN[i] + t1)])
                    return seat
    raise RuntimeError('no final cup')


pr_end = final_cup(r_, SEQ['rose'][-1], *STATE['rose'][:2])


# ---- gold's last note lands on the lever (see-saw) in the base; the other end holds the child's cup
p114, v114 = STATE['gold'][0], STATE['gold'][1]
T_END = TN[114]
PHI0 = math.radians(14)
tau2 = 0.3
for _ in range(20):
    off = tip_landing(np.zeros(3), np.zeros(3), -sk, tau2)[0]
    x_kend = WEIGHTS[KW]['x'] + off[0]
    sgn_l = 1 if x_kend > p114[0] else -1
    ARM = abs(x_kend - p114[0]) / 2 / math.cos(PHI0)
    z_piv = p114[2] - g.r - KEY_T - ARM * math.sin(PHI0) - 0.02
    kz = z_piv - ARM * math.sin(PHI0) + 0.06 + RK
need = -off[2]
z_wstop = kz + need
T_WSTOP = (WEIGHTS[KW]['z0'] - z_wstop) / WEIGHTS[KW]['v']        # the child's weight reaches its stop
print('T_WSTOP', T_WSTOP, 'VL', WEIGHTS[KW]['v'], 'z_wstop', z_wstop, 'z0', WEIGHTS[KW]['z0'])
assert T_WSTOP < T_END - 1.3, ('weight too late', T_WSTOP)
assert sgn_l == sk and abs(x_kend - p114[0]) > 0.5 and need > 0.1, (x_kend, p114, need)
x_piv = 0.5 * (x_kend + p114[0])
kb_seat = np.array([x_kend, YK_W, kz])
print('lever: tau2', round(tau2, 2), 'drop', round(need, 2), 'arm', round(ARM, 2))
T_FLIP = 0.075                                   # the lever swings from +PHI0 to -PHI0
g.hits.append(dict(i=114, t=float(T_END), p=list(map(float, p114)), n=[0, 0, 1], e=0.0, kind='seesaw', vin=list(map(float, v114)),
                   vout=[0.0, 0.0, 0.0], pitch=int(PN[114])))
KEYS.append(dict(c=p114 - np.array([0, 0, g.r + KEY_T / 2]), M=np.eye(3), hw=KEY_W / 2, hd=key_d(PN[114]) / 2, th=KEY_T / 2, ball='gold', t=float(T_END), i=114))
# gold rides its lever end down (lever angle phi(t) = PHI0 -> -PHI0 linearly in T_FLIP) and stays
dz_end = -2 * ARM * math.sin(PHI0)
g.lin(T_END, T_END + T_FLIP, p114, [0, 0, dz_end / T_FLIP]); g.rest(T_END + T_FLIP, 80.0, p114 + np.array([0, 0, dz_end]))
commit_path(g, T_END, 80.0)
EVT['seesaw'] = dict(t=float(T_END), pivot=[float(x_piv), 0.0, float(z_piv)], arm=ARM, phi0=PHI0, t_flip=T_FLIP, side=sgn_l,
                     gold_end=p114.tolist(), kid_seat=kb_seat.tolist())
SFX.append(['seesaw', float(T_END)])

# rides the weight down to its stop above the lever's child end, then the weight cup tips inward
lip_rel = tip_landing(np.zeros(3), np.zeros(3), -sk, tau2)[0]
W_STOP_SEAT = z_wstop
x_need = kb_seat[0] - lip_rel[0]
WEIGHTS[KW]['zstop'] = float(W_STOP_SEAT)
t_wstop = T_WSTOP
print('child on weight', KW, round(t_landk, 2), '-> stop', round(t_wstop, 2), 'seat x', WEIGHTS[KW]['x'], 'need', round(x_need, 2))
assert t_wstop < T_END - 1.2, 'the weight arrives too late'
SEAT_K = np.array([0, YK_W, 0])
k_.lin(t_landk, t_wstop, pkw, [0, 0, -WEIGHTS[KW]['v']]); k_.rest(t_wstop, t_wstop + 0.5, w_seat(KW, t_wstop) + SEAT_K)
commit_path(k_, t_landk, t_wstop + 0.5)
CUPS.append(dict(ball='kid', t_in=float(t_landk), t_out=float(t_wstop + 0.5), p=list(map(float, pkw)), kind='weight', ref=KW, vin=list(map(float, vkw))))
SFX.append(['wstop', float(t_wstop)])
t_l2, lip2, vl2 = tip_release(k_, t_wstop + 0.5, w_seat(KW, t_wstop) + SEAT_K, np.zeros(3), -sk)
pkb, vkb = drop_to(k_, t_l2, lip2, vl2, t_l2 + tau2)
kb_seat = pkb.copy(); EVT['seesaw']['kid_seat'] = kb_seat.tolist()
k_.rest(t_l2 + tau2, T_END, kb_seat); commit_path(k_, t_l2 + tau2, T_END)
SFX.append(['tip', float(t_wstop + 0.5)]); SFX.append(['cup', float(t_l2 + tau2)])
# the other weight also stops at the same height
WEIGHTS['L' if KW == 'R' else 'R']['zstop'] = float(W_STOP_SEAT)

# ---- launch: the lever flips, the child flies out through the hatch in the left wall onto the room floor
Z_BASE = min(p114[2] + dz_end, kb_seat[2], pr_end[2], z_piv) - 0.75     # inside floor of the clock base
WALL_X = 1.8 * sk
Z_ROOM = Z_BASE - 1.1                                                   # room floor (the clock's plinth is 1.1 high)
VL = np.array([2.3 * sk, 0.0, 3.5])
k_.lin(T_END, T_END + T_FLIP, kb_seat, [0, 0, 2 * ARM * math.sin(PHI0) / T_FLIP])
p_l = kb_seat + np.array([0, 0, 2 * ARM * math.sin(PHI0)])
commit_path(k_, T_END, T_END + T_FLIP)
t0 = T_END + T_FLIP
zf = Z_ROOM + RK
tland = fall_time(p_l[2] - zf, VL[2])                   # (fall_time solves h = -v0 t + G t^2/2 with v0 = vz)
tland = (VL[2] + math.sqrt(VL[2] ** 2 + 2 * G * (p_l[2] - zf))) / G
k_.fly(t0, t0 + tland, p_l, VL); commit_path(k_, t0, t0 + tland)
t_cross = (WALL_X - p_l[0]) / VL[0]
z_cross = p_l[2] + VL[2] * t_cross - 0.5 * G * t_cross ** 2
EVT['hatch_out'] = dict(t=float(t0 + t_cross), z=float(z_cross), x=WALL_X)
SFX.append(['launch', float(T_END)])
# floor bounces (wooden floor, e = 0.4), then rolling with constant rolling resistance
p = p_l + VL * tland + np.array([0, 0, -0.5 * G * tland * tland]); v = VL + np.array([0, 0, -G * tland]); t = t0 + tland
FLOOR_HITS = []
for k in range(3):
    FLOOR_HITS.append(dict(t=float(t), p=p.tolist(), v=float(-v[2])))
    SFX.append(['floor', float(t)])
    vz = -v[2] * 0.4
    if vz < 0.5: break
    dt = 2 * vz / G; vo = np.array([v[0] * 0.92, 0, vz])
    k_.fly(t, t + dt, p, vo); commit_path(k_, t, t + dt)
    p = p + vo * dt + np.array([0, 0, -0.5 * G * dt * dt]); p[2] = zf; v = vo + np.array([0, 0, -G * dt]); t = t + dt
# roll: v_x decreasing linearly (a = 0.35 m/s^2) until the pendulum cup of the little clock at T_ARR
kq = 0
while E['t_tag'] - Q / 2 - (kq + 1) * Q >= t + 0.9: kq += 1
T_ARR = E['t_tag'] - Q / 2 - kq * Q             # the little pendulum's extremes then fall on the tag's beats
print('floor roll from', round(t, 2), 'arrive', round(T_ARR, 2))
vx0 = v[0]; troll = T_ARR - t
assert troll > 0.3, troll
A_ROLL = (0.62 * sk - vx0) / troll              # rolling on the rug: constant rolling resistance
v_arr = vx0 + A_ROLL * troll                   # vx0 < 0: slowing down
dist = vx0 * troll + 0.5 * A_ROLL * troll ** 2
k_.lin(t, T_ARR, [p[0], p[1], zf], [vx0, 0, 0], [A_ROLL, 0, 0]); commit_path(k_, t, T_ARR)
x_arr = p[0] + dist
print('child rolls from', round(p[0], 2), 'to', round(x_arr, 2), 'arrives', round(T_ARR, 2), 'speed', round(v_arr, 2))
# ---- the little clock: the child settles in the cup of its pendulum bob and becomes its heart
LP = G * (Q / math.pi) ** 2                     # pendulum length for one swing per beat
AMP = abs(v_arr) / math.sqrt(G * LP)            # small-angle amplitude from the arrival speed
piv_s = np.array([x_arr, p[1], zf + LP])
EVT['little'] = dict(t_arr=float(T_ARR), pivot=piv_s.tolist(), L=float(LP), amp=float(AMP), dir=sk)
k_.seg.append(dict(k='pend', t0=float(T_ARR), t1=80.0, piv=piv_s.tolist(), L=float(LP), amp=float(AMP), w=math.pi / Q, sgn=float(sk)))
EVT['floor_hits'] = FLOOR_HITS
EVT['base'] = dict(z_base=float(Z_BASE), z_room=float(Z_ROOM), wall_x=WALL_X)
SFX.append(['settle', float(T_ARR)])


# ---- everything the scene needs
def jl(x): return [float(v) for v in x]
OUT = dict(G=G, R=R, RK=RK, key_w=KEY_W, key_t=KEY_T, wx=WX, w_off=W_OFF,
           balls={n: dict(r=b.r, seg=b.seg, hits=b.hits) for n, b in B.items()},
           keys=[dict(c=jl(K['c']), M=[jl(r) for r in K['M']], hw=float(K['hw']), hd=float(K['hd']), th=float(K['th']), ball=K['ball'], t=K['t'], i=K['i'],
                      pitch=int(PN[K['i']])) for K in KEYS],
           cups=CUPS, rails=RAILS, weights=WEIGHTS, ev=EVT, sfx=sorted(SFX, key=lambda x: x[1]),
           boxes=[dict(c=jl(b['c']), M=[jl(r) for r in b['M']], h=jl(b['h']), kind=('rail' if b.get('rail') else 'cradle' if b.get('cradle') else 'cup')) for b in BOXES])
if __name__ == '__main__':
    for n, b in B.items():
        zz = [h['p'][2] for h in b.hits]
        if zz: print(n, len(b.hits), 'z', round(min(zz), 2), round(max(zz), 2), 'e', round(min(h['e'] for h in b.hits), 2), round(max(h['e'] for h in b.hits), 2))
    json.dump(OUT, open(f'{HERE}/layout.json', 'w'), default=float)
    import sys
    if 'hits' in sys.argv:
        for n, b in B.items():
            for h in b.hits: print(n, h['i'], h['kind'], [round(x, 2) for x in h['p']], np.round(h['vout'], 2), round(h['e'], 2), round(math.degrees(math.acos(h['n'][2])), 1))
