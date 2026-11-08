"""Checks for tell_gallop (no Blender needed): python3 verify.py [table]
 - every melody note is struck at its exact time by a ball (or a gold drop ball in the final tutti), pitch per note
 - contact distance between ball and bar on the rendered frame nearest each hit
 - collisions: ball surface entering any bar other than its own at its own hit
 - free flight: per-frame acceleration == gravity, frame-to-frame speed ratio"""
import json, os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(f'{HERE}/layout.json')); LEAD = json.load(open(f'{HERE}/lead.json')); E = json.load(open(f'{HERE}/events.json'))
G = L['G']; R = L['R']; FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.5) * FPS)
BAR_W = {'bar': 0.16, 'trap': 0.5, 'pit': 0.18, 'mat': 0.9, 'hurdle': 0.16, 'finish': 0.3, 'step': 0.16, 'podium': 0.5, 'bell': 0.2}
BAR_TH = 0.07


def at(seg, t):
    for g in seg:
        if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9:
            p = np.array(g['p'])
            if g['k'] == 'rest': return p
            u = t - g['t0']; return p + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
    return np.array(seg[0]['p']) if t < seg[0]['t0'] else at(seg, seg[-1]['t1'])


def frame(n):
    n = np.array(n, float); n /= np.linalg.norm(n)
    xa = np.cross([0, 1, 0], n); xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return np.stack([xa, ya, n], 1)


bars = []
for name, b in L['balls'].items():
    for h in b['hits']:
        if h['kind'] == 'podium': continue
        n = np.array(h['n'], float); n /= np.linalg.norm(n); c = np.array(h['p']) - R * n
        if name == 'gold': hw, hd, th = 0.07, 0.1, 0.07; ctr = c - n * 0.07
        else:
            w = BAR_W.get(h['kind'], 0.16); d = 0.95 if h['kind'] in ('bar', 'hurdle', 'pit', 'trap', 'mat', 'finish') else 0.5
            hw, hd, th = w / 2, d / 2, BAR_TH / 2; ctr = c - n * th
        bars.append(dict(ball=name, i=h['i'], t=h['t'], kind=h['kind'], c=ctr, M=frame(n), hw=hw, hd=hd, th=th, contact=c, n=n))
info = {}
# 1) note coverage + pitch/time table
hitmap = {}
for name, b in L['balls'].items():
    for h in b['hits']: hitmap.setdefault(h['i'], []).append((name, h['t'], h['kind']))
for i in L['ev']['gold_notes']: hitmap.setdefault(i, []).append(('goldbell', LEAD[i][0], 'bell'))
miss = [i for i in range(len(LEAD)) if i not in hitmap]
dt_err = max(abs(t - LEAD[i][0]) for i, v in hitmap.items() for _, t, _ in v)
info['notes'] = len(LEAD); info['notes_without_visible_hit'] = miss; info['hit_time_error_max_s'] = dt_err
NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    secs = E['secs']
    for i, (t, p, v, d) in enumerate(LEAD):
        s = [n for n, ts in secs if ts <= t + 1e-6][-1]
        who = ','.join(f'{n}:{k}' for n, _, k in hitmap[i])
        print(f'{i:3d} {t:7.3f}s {s:5s} {NAMES[p % 12]}{p // 12 - 1:<2d} frame {round(t * FPS):5d}  {who}')
# 2) contact distance on the nearest frame
gaps = []
for B in bars:
    f = round(B['t'] * FPS); p = at(L['balls'][B['ball']]['seg'], f / FPS)
    gaps.append(np.linalg.norm(p - (B['contact'] + R * B['n'])))
info['contact_gap_nearest_frame_max'] = round(float(max(gaps)), 3); info['contact_gap_mean'] = round(float(np.mean(gaps)), 3)
# 3) collisions (sampled at 240 Hz)
ts = np.arange(0.2, FINAL + 1.0, 1 / 240)
bad = {}; worst = 0.0
P = {n: np.array([at(b['seg'], t) for t in ts]) for n, b in L['balls'].items()}
for B in bars:
    for n, Pn in P.items():
        m = (np.abs(Pn[:, 0] - B['c'][0]) < 0.7) & (np.abs(Pn[:, 2] - B['c'][2]) < 0.7) & (np.abs(Pn[:, 1] - B['c'][1]) < 1.0)
        if B['ball'] == n: m &= np.abs(ts - B['t']) > 0.03
        if B['kind'] == 'trap': m &= ts < B['t']          # the trapdoor swings away right after the hit
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
# 4) free flight: acceleration per frame equals gravity; speed ratio
tf = np.arange(NF) / FPS
evt = sorted([h['t'] for b in L['balls'].values() for h in b['hits']] + [g['t0'] for b in L['balls'].values() for g in b['seg']])
amax = 0.0; rat = 1.0; rat_fast = 1.0
for n, b in L['balls'].items():
    Q = np.array([at(b['seg'], t) for t in tf])
    for f in range(1, NF - 1):
        t0, t1 = tf[f - 1], tf[f + 1]
        if any(t0 - 1e-6 <= e <= t1 + 1e-6 for e in evt[np.searchsorted(evt, t0 - 0.1):np.searchsorted(evt, t1 + 0.1)]): continue
        a = (Q[f + 1] - 2 * Q[f] + Q[f - 1]) * FPS * FPS
        if np.linalg.norm(Q[f + 1] - Q[f - 1]) < 1e-9: continue           # at rest
        amax = max(amax, float(np.linalg.norm(a - np.array([0, 0, -G]))))
        s0 = np.linalg.norm(Q[f] - Q[f - 1]) * FPS; s1 = np.linalg.norm(Q[f + 1] - Q[f]) * FPS
        if min(s0, s1) > 0.5: rat = max(rat, max(s0, s1) / min(s0, s1))
        if min(s0, s1) > 2.0: rat_fast = max(rat_fast, max(s0, s1) / min(s0, s1))
info['flight_accel_error_max'] = round(amax, 6); info['flight_speed_ratio_max(v>0.5)'] = round(rat, 3); info['flight_speed_ratio_max(v>2)'] = round(rat_fast, 3)
# restitution / speed at every bounce
hs = [h for b in L['balls'].values() for h in b['hits'] if np.linalg.norm(h['vout']) > 0 and h['kind'] != 'mat']
info['bounce_speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in hs), 3)
info['restitution_range'] = (round(min(h['e'] for h in hs if h['kind'] != 'trap'), 2), round(max(h['e'] for h in hs), 3))
for k, v in info.items(): print(k, v)
