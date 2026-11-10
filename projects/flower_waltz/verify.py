"""Checks for flower_waltz (no Blender needed): python3 verify.py [table]
 - every melody onset is struck at its exact time by a ball, pitch/time table against the score (bar.beat)
 - contact distance between ball and flower part on the rendered frame nearest each hit
 - collisions: ball surface entering any flower part other than at its own hit (240 Hz)
 - free flight: per-frame acceleration == gravity, frame-to-frame speed ratio; bounce speed out/in"""
import json, os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from phys import at, KEY, key_frame
L = json.load(open(f'{HERE}/layout.json')); LEAD = json.load(open(f'{HERE}/lead.json')); E = json.load(open(f'{HERE}/events.json'))
G = L['G']; R = L['R']; FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.0) * FPS)
NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
keys = []
for name, b in L['balls'].items():
    for h in b['hits']:
        if h['kind'] not in KEY: continue
        hw, hd, th = KEY[h['kind']]; n = np.array(h['n'], float); n /= np.linalg.norm(n)
        c = np.array(h['p']) - R * n
        keys.append(dict(ball=name, i=h['i'], t=h['t'], kind=h['kind'], c=c - n * th, M=key_frame(n), hw=hw, hd=hd, th=th, contact=c, n=n))
info = {}
hitmap = {}
for name, b in L['balls'].items():
    for h in b['hits']:
        if h['kind'] in ('gutter', 'bucket', 'lever', 'stop'): continue
        hitmap.setdefault(h['i'], []).append((name, h['t'], h['kind']))
miss = [i for i in range(len(LEAD)) if i not in hitmap]
info['onsets'] = len(LEAD); info['onsets_without_visible_hit'] = miss
info['hit_time_error_max_s'] = max(abs(t - LEAD[i]['t']) for i, v in hitmap.items() for _, t, _ in v)
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    for i, l in enumerate(LEAD):
        who = ','.join(f'{n}:{k}' for n, _, k in hitmap[i]); hm = ' '.join(NAMES[q % 12] + str(q // 12 - 1) for q in l['harm'])
        print(f"{i:3d} {l['t']:7.3f}s bar {l['bar']:2d}.{l['beat'] + 1:<6g} {NAMES[l['p'] % 12]:>2s}{l['p'] // 12 - 1:<2d} [{hm:12s}] {l['sec']:5s} frame {round(l['t'] * FPS):5d}  {who}")
gaps = []
for K in keys:
    f = round(K['t'] * FPS); p = at(L['balls'][K['ball']]['seg'], f / FPS)
    gaps.append(np.linalg.norm(p - (K['contact'] + R * K['n'])))
info['contact_gap_nearest_frame_max'] = round(float(max(gaps)), 3); info['contact_gap_mean'] = round(float(np.mean(gaps)), 3)
ts = np.arange(0.0, FINAL + 1.0, 1 / 240)
P = {n: np.array([at(b['seg'], t) for t in ts]) for n, b in L['balls'].items()}
bad = {}; worst = 0.0
for K in keys:
    for n, Pn in P.items():
        m = (np.abs(Pn[:, 0] - K['c'][0]) < 0.8) & (np.abs(Pn[:, 2] - K['c'][2]) < 0.8) & (np.abs(Pn[:, 1] - K['c'][1]) < 0.8)
        if K['ball'] == n: m &= np.abs(ts - K['t']) > 0.03
        if K['kind'] == 'cup' and K['ball'] == n: m &= ~((ts > K['t'] - 0.03) & (ts < K['t'] + 30))   # resting in its own cup
        if not m.any(): continue
        rel = (Pn[m] - K['c']) @ K['M']
        d = np.stack([np.maximum(0, np.abs(rel[:, 0]) - K['hw']), np.maximum(0, np.abs(rel[:, 1]) - K['hd']), np.maximum(0, np.abs(rel[:, 2]) - K['th'])], 1)
        pen = R - np.linalg.norm(d, axis=1); k = pen > 0.01
        if k.any():
            key = (n, K['ball'], K['kind']); bad.setdefault(key, []).append((K['i'], round(float(ts[m][k][0]), 3), round(float(pen[k].max()), 3)))
            worst = max(worst, float(pen[k].max()))
info['collisions'] = {f'{a}->{b}:{c}': len(v) for (a, b, c), v in bad.items()}
info['collision_examples'] = {f'{a}->{b}:{c}': v[:4] for (a, b, c), v in bad.items()}
info['collision_worst_pen'] = round(worst, 3)
names = list(P); mind = 9.0; pair = None
for a in range(len(names)):
    for c in range(a + 1, len(names)):
        d = np.linalg.norm(P[names[a]] - P[names[c]], axis=1); k = int(np.argmin(d))
        if d[k] < mind: mind = float(d[k]); pair = (names[a], names[c], round(float(ts[k]), 2))
info['min_ball_distance'] = (round(mind, 3), pair)
tf = np.arange(NF) / FPS
evt = np.array(sorted([h['t'] for b in L['balls'].values() for h in b['hits']] + [g['t0'] for b in L['balls'].values() for g in b['seg']]))
amax = 0.0; rat = 1.0; rat_fast = 1.0
for n, b in L['balls'].items():
    Q = np.array([at(b['seg'], t) for t in tf])
    for f in range(1, NF - 1):
        t0, t1 = tf[f - 1], tf[f + 1]
        k0 = np.searchsorted(evt, t0 - 1e-6); k1 = np.searchsorted(evt, t1 + 1e-6)
        if k1 > k0: continue
        if np.linalg.norm(Q[f + 1] - Q[f - 1]) < 1e-9: continue
        seg = [g for g in b['seg'] if g['t0'] - 1e-9 <= tf[f] <= g['t1'] + 1e-9][0]
        if seg['k'] != 'fly': continue
        a = (Q[f + 1] - 2 * Q[f] + Q[f - 1]) * FPS * FPS
        amax = max(amax, float(np.linalg.norm(a - np.array([0, 0, -G]))))
        s0 = np.linalg.norm(Q[f] - Q[f - 1]) * FPS; s1 = np.linalg.norm(Q[f + 1] - Q[f]) * FPS
        if min(s0, s1) > 0.5: rat = max(rat, max(s0, s1) / min(s0, s1))
        if min(s0, s1) > 2.0: rat_fast = max(rat_fast, max(s0, s1) / min(s0, s1))
info['flight_accel_error_max'] = round(amax, 6); info['flight_speed_ratio_max(v>0.5)'] = round(rat, 3); info['flight_speed_ratio_max(v>2)'] = round(rat_fast, 3)
hs = [h for b in L['balls'].values() for h in b['hits'] if np.linalg.norm(h['vout']) > 0]
info['bounce_speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in hs), 3)
info['restitution_max'] = round(max(h['e'] for h in hs if h['kind'] not in ('gutter', 'stop')), 3)
info['tilt_max_deg(flower parts)'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for h in hs if h['kind'] != 'stop'), 1)
for k, v in info.items(): print(k, v)
