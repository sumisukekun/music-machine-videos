"""Checks for beethoven5 (no Blender needed): python3 verify.py [table]
 - every melody note is struck at its exact time by a ball; pitch / time table (with 'table')
 - contact distance between ball and key on the rendered frame nearest each hit
 - collisions: a ball entering any key / bowl / hatch box / bell other than its own at its own hit
 - free flight: per-frame acceleration == gravity; frame-to-frame speed ratio; bounce restitution and speed out/in
 - ball-ball overlap"""
import json, os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(f'{HERE}/layout.json')); LEAD = json.load(open(f'{HERE}/lead.json')); E = json.load(open(f'{HERE}/events.json'))
G = L['G']; R = L['R']; FPS = 30; FINAL = E['final']; NF = int((FINAL + 3.8) * FPS)
KD = L['keydim']; EV = L['ev']


def at(seg, t):
    for g in seg:
        if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9:
            p = np.array(g['p'])
            if g['k'] == 'rest': return p
            u = t - g['t0']; a = np.array(g['a']) if g['k'] == 'acc' else np.array([0, 0, -G])
            return p + np.array(g['v']) * u + 0.5 * a * u * u
    return np.array(seg[0]['p']) if t < seg[0]['t0'] else at(seg, seg[-1]['t1'])


def frame(n):
    n = np.array(n, float); n /= np.linalg.norm(n)
    xa = np.cross([0, 1, 0], n); xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return np.stack([xa, ya, n], 1)


# obstacles: keys (oriented boxes), bowls, hatch boxes, bells
obs = []
for name, b in L['balls'].items():
    for h in b['hits']:
        k = h['kind']; n = np.array(h['n'], float); n /= np.linalg.norm(n); c = np.array(h['p']) - R * n
        if k in ('plate', 'gong'): continue                                   # on the door / gong face (checked separately)
        if k in ('bowl', 'twinbowl', 'ropebowl', 'lever'):
            obs.append(dict(ball=name, t=h['t'], i=h['i'], kind=k, c=c - [0, 0, 0.06], M=np.eye(3), hw=0.22, hd=0.22, th=0.08)); continue
        dk = 'twinkey' if b['kind'] == 'twin' else ('bell' if k == 'bell' else ('step' if k == 'step' else 'key'))
        w, d, th = KD[dk]
        obs.append(dict(ball=name, t=h['t'], i=h['i'], kind=k, c=c - n * th / 2, M=frame(n), hw=w / 2, hd=d / 2, th=th / 2))
for hc in EV['hatches']:
    obs.append(dict(ball=hc['ball'], t=-1, i=-1, kind='hatch', c=np.array(hc['p']) + [0, 0, 0.0], M=np.eye(3), hw=0.19, hd=0.19, th=0.18))
info = {}
hitmap = {}
for name, b in L['balls'].items():
    for h in b['hits']: hitmap.setdefault(h['i'], []).append((name, h['t'], h['kind']))
miss = [i for i in range(len(LEAD)) if i not in hitmap]
info['notes'] = len(LEAD); info['notes_without_visible_hit'] = miss
info['hit_time_error_max_s'] = max(abs(t - LEAD[i][0]) for i, v in hitmap.items() for _, t, _ in v)
NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    for i, l in enumerate(LEAD):
        t, p = l[0], l[1]; tw = l[7]
        who = ','.join(f'{n}:{k}' for n, _, k in hitmap[i])
        print(f'{i:3d} {t:7.3f}s bar {l[5] + 1:7.3f} {NAMES[p % 12]}{p // 12 - 1:<2d}' + (f' +{NAMES[tw % 12]}{tw // 12 - 1}' if tw else '    ') +
              f' frame {round(t * FPS):5d}  {who}')
# contact distance on the nearest frame
gaps = []
for name, b in L['balls'].items():
    for h in b['hits']:
        f = round(h['t'] * FPS); p = at(b['seg'], f / FPS); gaps.append(np.linalg.norm(p - np.array(h['p'])))
info['contact_gap_nearest_frame_max'] = round(float(max(gaps)), 3); info['contact_gap_mean'] = round(float(np.mean(gaps)), 3)
# collisions (sampled at 240 Hz)
ts = np.arange(0.0, FINAL + 3.8, 1 / 240)
P = {n: np.array([at(b['seg'], t) for t in ts]) for n, b in L['balls'].items()}
bad = {}; worst = 0.0
for O in obs:
    for n, Pn in P.items():
        m = (np.abs(Pn[:, 0] - O['c'][0]) < 0.6) & (np.abs(Pn[:, 2] - O['c'][2]) < 0.6) & (np.abs(Pn[:, 1] - O['c'][1]) < 0.7)
        if O['ball'] == n:
            if O['kind'] == 'hatch': continue
            if O['kind'] in ('bowl', 'twinbowl', 'ropebowl', 'lever'): m &= ts < O['t'] - 0.03
            else: m &= np.abs(ts - O['t']) > 0.03
        if not m.any(): continue
        rel = (Pn[m] - O['c']) @ O['M']
        d = np.stack([np.maximum(0, np.abs(rel[:, 0]) - O['hw']), np.maximum(0, np.abs(rel[:, 1]) - O['hd']), np.maximum(0, np.abs(rel[:, 2]) - O['th'])], 1)
        pen = R - np.linalg.norm(d, axis=1); k = pen > 0.01
        if k.any():
            key = (n, O['ball'], O['kind']); bad.setdefault(key, []).append((O['i'], round(float(ts[m][k][0]), 3), round(float(pen[k].max()), 3)))
            worst = max(worst, float(pen[k].max()))
info['collisions'] = {f'{a}->{b}:{c}': v[:3] for (a, b, c), v in bad.items()}
info['collision_worst_pen'] = round(worst, 3)
# ball-ball overlap
names = list(P); mind = 9.0; pair = None
for a in range(len(names)):
    for c in range(a + 1, len(names)):
        d = np.linalg.norm(P[names[a]] - P[names[c]], axis=1).min()
        if d < mind: mind = d; pair = (names[a], names[c])
info['min_ball_distance'] = (round(float(mind), 3), pair)
# free flight: acceleration per frame == gravity; speed ratio between frames
tf = np.arange(NF) / FPS
amax = 0.0; rat = 1.0; where = None
for n, b in L['balls'].items():
    evt = sorted([h['t'] for h in b['hits']] + [g['t0'] for g in b['seg']])
    Q = np.array([at(b['seg'], t) for t in tf])
    for f in range(1, NF - 1):
        t0, t1 = tf[f - 1], tf[f + 1]
        if any(t0 - 1e-6 <= e <= t1 + 1e-6 for e in evt): continue
        seg = [g for g in b['seg'] if g['t0'] <= tf[f] <= g['t1']]
        if not seg or seg[0]['k'] != 'fly': continue
        a = (Q[f + 1] - 2 * Q[f] + Q[f - 1]) * FPS * FPS
        amax = max(amax, float(np.linalg.norm(a - np.array([0, 0, -G]))))
        s0 = np.linalg.norm(Q[f] - Q[f - 1]) * FPS; s1 = np.linalg.norm(Q[f + 1] - Q[f]) * FPS
        if min(s0, s1) > 0.5 and max(s0, s1) / min(s0, s1) > rat: rat = max(s0, s1) / min(s0, s1); where = (n, round(tf[f], 2))
info['flight_accel_error_max'] = round(amax, 6); info['flight_speed_ratio_max(v>0.5)'] = (round(rat, 3), where)
hs = [h for b in L['balls'].values() for h in b['hits'] if np.linalg.norm(h['vout']) > 0]
info['bounce_speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in hs), 3)
info['restitution_range'] = (round(min(h['e'] for h in hs), 3), round(max(h['e'] for h in hs), 3))
# ramps: rolling acceleration = 5/7 g sin(slope)
info['ramp_accel'] = [round(r['a'], 3) for r in EV['ramps']]
for k, v in info.items(): print(k, v)
