"""Checks for hungarian5 (no Blender needed): python3 verify.py [table]
 - every melody note is struck by the thief and every bass note by the police ball, at its exact time (pitch table)
 - contact distance between ball and tile on the rendered frame nearest each hit
 - collisions: ball surface entering any tile other than its own at its own hit (240 Hz)
 - free flight: per-frame acceleration == gravity, frame-to-frame speed ratio; rolling: constant acceleration
 - bounces never add speed; ball-ball distance"""
import json, os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(f'{HERE}/layout.json')); LEAD = json.load(open(f'{HERE}/lead.json')); BASS = json.load(open(f'{HERE}/bass.json'))
E = json.load(open(f'{HERE}/events.json'))
G = L['G']; R = L['R']; FPS = 30; NF = int((E['final'] + 4.2) * FPS)
TILE_W = {'tile': 0.17, 'plate': 0.2, 'gut': 0.1, 'slat': 0.16, 'bell': 0.2, 'cbell': 0.2}


def seg_at(g, t):
    p = np.array(g['p']); u = t - g['t0']
    if g['k'] == 'rest': return p
    if g['k'] == 'fly': return p + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
    return p + np.array(g['u']) * (g['v0'] * u + 0.5 * g['a'] * u * u)


def at(seg, t):
    for g in seg:
        if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9: return seg_at(g, t)
    return seg_at(seg[0], seg[0]['t0']) if t < seg[0]['t0'] else seg_at(seg[-1], seg[-1]['t1'])


def frame(n):
    n = np.array(n, float); n /= np.linalg.norm(n)
    xa = np.cross([0, 1, 0], n); xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return np.stack([xa, ya, n], 1)


bars = []
for name, b in L['balls'].items():
    for h in b['hits']:
        if h['kind'] == 'gbell': continue
        n = np.array(h['n'], float); n /= np.linalg.norm(n); c = np.array(h['p']) - R * n
        if h['kind'] in ('bell', 'cbell'): hw, hd, th = 0.1, 0.1, 0.08
        else:
            w = h.get('w', TILE_W[h['kind']]); d = 0.8 if h['kind'] in ('tile', 'slat') else (0.85 if h['kind'] == 'plate' else 0.22)
            hw, hd, th = w / 2, d / 2, 0.03
        bars.append(dict(ball=name, i=h['i'], t=h['t'], kind=h['kind'], c=c - n * th, M=frame(n), hw=hw, hd=hd, th=th, contact=c, n=n,
                         roll=h.get('roll', False)))
info = {}
# 1) note coverage + pitch/time table
NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
nm = lambda p: f'{NAMES[p % 12]}{p // 12 - 1}'
for who, notes in (('thief', LEAD), ('cop', BASS)):
    hm = {h['i']: h for h in L['balls'][who]['hits']}
    miss = [i for i in range(len(notes)) if i not in hm]
    err = max(abs(hm[i]['t'] - notes[i]['t']) for i in hm)
    info[f'{who}_notes'] = len(notes); info[f'{who}_missing'] = miss; info[f'{who}_time_err_max_s'] = err
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    ev = sorted([(l['t'], 'MEL', nm(l['p']), l['sec'], l['bar'], l['pos'], L['balls']['thief']['hits'][[h['i'] for h in L['balls']['thief']['hits']].index(k)]['kind'])
                 for k, l in enumerate(LEAD)] +
                [(b['t'], 'BASS', '+'.join(nm(p) for p in b['p']), b['sec'], b['bar'], b['pos'], 'plate') for b in BASS])
    for t, w, p, s, bar, pos, k in ev: print(f'{t:8.3f}s f{round(t * FPS):5d} {s:5s} bar{bar + 1:3d} pos{pos:4.1f} {w:4s} {p:16s} {k}')
# 2) contact distance on the nearest frame
gaps = []
for B in bars:
    f = round(B['t'] * FPS); p = at(L['balls'][B['ball']]['seg'], f / FPS)
    gaps.append(np.linalg.norm(p - (B['contact'] + R * B['n'])))
info['contact_gap_nearest_frame_max'] = round(float(max(gaps)), 3); info['contact_gap_mean'] = round(float(np.mean(gaps)), 3)
exact = max(np.linalg.norm(at(L['balls'][B['ball']]['seg'], B['t']) - (B['contact'] + R * B['n'])) for B in bars)
info['contact_gap_at_exact_hit_time_max'] = float(exact)
# 3) collisions (240 Hz)
ts = np.arange(0.0, E['final'] + 1.0, 1 / 240)
P = {n: np.array([at(b['seg'], t) for t in ts]) for n, b in L['balls'].items()}
bad = {}; worst = 0.0
for B in bars:
    for n, Pn in P.items():
        m = (np.abs(Pn[:, 0] - B['c'][0]) < 0.6) & (np.abs(Pn[:, 2] - B['c'][2]) < 0.6) & (np.abs(Pn[:, 1] - B['c'][1]) < 0.9)
        if B['ball'] == n: m &= np.abs(ts - B['t']) > (0.03 if not B['roll'] else 0.0)
        if not m.any(): continue
        rel = (Pn[m] - B['c']) @ B['M']
        d = np.stack([np.maximum(0, np.abs(rel[:, 0]) - B['hw']), np.maximum(0, np.abs(rel[:, 1]) - B['hd']), np.maximum(0, np.abs(rel[:, 2]) - B['th'])], 1)
        pen = R - np.linalg.norm(d, axis=1)
        k = pen > 0.01
        if k.any():
            key = (n, B['ball'], B['kind']); bad.setdefault(key, []).append((B['i'], round(float(ts[m][k][0]), 3), round(float(pen[k].max()), 3)))
            worst = max(worst, float(pen[k].max()))
info['collisions'] = {f'{a}->{b}:{c}': len(v) for (a, b, c), v in bad.items()}
info['collision_examples'] = {f'{a}->{b}:{c}': v[:4] for (a, b, c), v in bad.items()}
info['collision_worst_pen'] = round(worst, 3)
# 4) free flight: acceleration per frame equals gravity; speed ratio (frames not touching a contact / segment change)
tf = np.arange(NF) / FPS
evt = np.array(sorted([h['t'] for b in L['balls'].values() for h in b['hits']] + [g['t0'] for b in L['balls'].values() for g in b['seg']]))
amax = 0.0; rat = 1.0; rat_fast = 1.0; roll_dev = 0.0
for n, b in L['balls'].items():
    Q = np.array([at(b['seg'], t) for t in tf])
    for f in range(1, NF - 1):
        t0, t1 = tf[f - 1], tf[f + 1]
        lo, hi = np.searchsorted(evt, t0 - 1e-6), np.searchsorted(evt, t1 + 1e-6)
        if hi > lo: continue
        g = [g for g in b['seg'] if g['t0'] - 1e-9 <= tf[f] <= g['t1'] + 1e-9][0]
        a = (Q[f + 1] - 2 * Q[f] + Q[f - 1]) * FPS * FPS
        if g['k'] == 'rest': continue
        if g['k'] == 'fly': amax = max(amax, float(np.linalg.norm(a - np.array([0, 0, -G]))))
        else: roll_dev = max(roll_dev, float(np.linalg.norm(a - np.array(g['u']) * g['a'])))
        s0 = np.linalg.norm(Q[f] - Q[f - 1]) * FPS; s1 = np.linalg.norm(Q[f + 1] - Q[f]) * FPS
        if min(s0, s1) > 0.5: rat = max(rat, max(s0, s1) / min(s0, s1))
        if min(s0, s1) > 2.0: rat_fast = max(rat_fast, max(s0, s1) / min(s0, s1))
info['flight_accel_error_max'] = round(amax, 6); info['rolling_accel_error_max'] = round(roll_dev, 6)
info['frame_speed_ratio_max(v>0.5)'] = round(rat, 3); info['frame_speed_ratio_max(v>2)'] = round(rat_fast, 3)
# 5) bounces
hs = [h for b in L['balls'].values() for h in b['hits']]
info['speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / max(1e-9, np.linalg.norm(h['vin'])) for h in hs), 4)
bb = [h for h in hs if not h.get('roll') and h['kind'] != 'gbell']
info['restitution_range'] = (round(min(h['e'] for h in bb), 3), round(max(h['e'] for h in bb), 3))
info['tile_tilt_max_deg'] = round(max(math.degrees(math.acos(min(1, h['n'][2]))) for h in hs), 1)
rolls = [g for b in L['balls'].values() for g in b['seg'] if g['k'] == 'roll']
info['rolling_parts'] = [(round(g['t0'], 2), round(g['t1'], 2), round(g['a'], 3), 'gutter (speeds up)' if g['a'] > 0 else 'deck friction (slows)') for g in rolls]
# 6) ball-ball distance
d = np.linalg.norm(P['thief'] - P['cop'], axis=1)
info['ball_distance_min'] = round(float(d.min()), 4); info['ball_distance_min_t'] = round(float(ts[np.argmin(d)]), 3)
for k, v in info.items(): print(k, v)
