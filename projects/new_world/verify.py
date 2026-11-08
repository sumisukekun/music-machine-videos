"""Checks for new_world (no Blender needed): python3 verify.py [table]
 - every melody note is a sleeper the loco rolls over at that exact time (pitch per note), every accompaniment note a
   car landing on its bell at that exact time (pitch per car)
 - contact distance between ball and bar on the rendered frame nearest each hit
 - collisions: ball surface entering any bar other than at its own hit (fallen trestle pieces excluded after they fall)
 - free flight: per-frame acceleration == gravity, frame-to-frame speed ratio; rolling: speed changes only on the ramp
 - bounces: restitution, speed out / speed in;  ball-ball minimum distance"""
import json, os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(f'{HERE}/layout.json')); LEAD = json.load(open(f'{HERE}/lead.json')); ACC = json.load(open(f'{HERE}/acc.json'))
E = json.load(open(f'{HERE}/events.json'))
G = L['G']; FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.5) * FPS)
NAMES = ['loco', 'red', 'yellow', 'green', 'blue']
B = L['balls']


def seg_at(g, t):
    p = np.array(g['p']); u = t - g['t0']
    if g['k'] == 'rest': return p
    if g['k'] == 'roll': return p + np.array(g['d']) * (g['v0'] * u + 0.5 * g['a'] * u * u)
    return p + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
def at(seg, t):
    for g in seg:
        if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9: return seg_at(g, t)
    return seg_at(seg[0], seg[0]['t0']) if t < seg[0]['t0'] else seg_at(seg[-1], seg[-1]['t1'])
def frame(n):
    n = np.array(n, float); n /= np.linalg.norm(n)
    up = np.array([0, 1.0, 0]) if abs(n[1]) < 0.9 else np.array([1.0, 0, 0])
    xa = np.cross(up, n); xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return np.stack([xa, ya, n], 1)


bars = []
for name in NAMES:
    R = B[name]['R']
    for h in B[name]['hits']:
        k = h['kind']
        if k in ('bell', 'cup'): continue
        n = np.array(h['n'], float); n /= np.linalg.norm(n); c = np.array(h['p']) - R * n
        if name == 'loco': hw, hd, th = (0.26, 0.26, 0.02) if k == 'cymbal' else (0.08, 0.23, 0.03)
        elif k == 'trestle': hw, hd, th = 0.1, 0.13, 0.025
        elif name == 'red' and k in ('drum', 'tunnel', 'rim'): hw, hd, th = 0.15, 0.15, 0.05
        else: hw, hd, th = 0.13, 0.13, 0.02
        bars.append(dict(ball=name, t=h['t'], kind=k, c=c - n * th + n * 0.004, M=frame(n), hw=hw, hd=hd, th=th, contact=c, n=n,
                         t_fall=h.get('t_fall'), pitch=h['pitch']))
info = {}
# 1) notes: loco
lh = B['loco']['hits']
info['melody_notes'] = len(LEAD)
info['melody_time_err_max'] = max(abs(h['t'] - l[0]) for h, l in zip(lh, LEAD))
info['melody_pitch_mismatch'] = sum(h['pitch'] != l[1] for h, l in zip(lh, LEAD)) + abs(len(lh) - len(LEAD))
miss = 0; terr = 0.0; nacc = 0
for a in ACC:
    for c, p in a['p'].items():
        nacc += 1; hs = [h for h in B[NAMES[int(c) + 1]]['hits'] if abs(h['t'] - a['t']) < 1e-3]
        if not hs or hs[0]['pitch'] != p: miss += 1
        else: terr = max(terr, abs(hs[0]['t'] - a['t']))
info['car_notes'] = nacc; info['car_notes_missing_or_wrong'] = miss; info['car_time_err_max'] = terr
N_ = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    for i, (t, p, d, s, b) in enumerate(LEAD):
        cars = [a for a in ACC if abs(a['t'] - t) < 1e-3]
        cs = ' '.join(f"{NAMES[int(c) + 1]}:{N_[q % 12]}{q // 12 - 1}" for a in cars for c, q in a['p'].items())
        print(f'{i:3d} bar{b:3d} {t:7.3f}s fr{round(t * FPS):5d} {s:6s} loco:{lh[i]["kind"]:8s} {N_[p % 12]}{p // 12 - 1:<2d} {cs}')
# 2) contact distance on nearest frame
gaps = []
for b in bars:
    f = round(b['t'] * FPS); p = at(B[b['ball']]['seg'], f / FPS)
    gaps.append(np.linalg.norm(p - (b['contact'] + B[b['ball']]['R'] * b['n'])))
info['contact_gap_nearest_frame_max'] = round(float(max(gaps)), 3); info['contact_gap_mean'] = round(float(np.mean(gaps)), 3)
# 3) collisions at 240 Hz
ts = np.arange(0.0, FINAL + 0.5, 1 / 240)
P = {n: np.array([at(B[n]['seg'], t) for t in ts]) for n in NAMES}
bad = {}; worst = 0.0
for b in bars:
    for n in NAMES:
        Pn = P[n]; R = B[n]['R']
        m = (np.abs(Pn[:, 0] - b['c'][0]) < 0.8) & (np.abs(Pn[:, 2] - b['c'][2]) < 0.8) & (np.abs(Pn[:, 1] - b['c'][1]) < 0.8)
        if b['ball'] == n: m &= np.abs(ts - b['t']) > 0.03
        if b['t_fall'] is not None: m &= ts < b['t_fall']
        if not m.any(): continue
        rel = (Pn[m] - b['c']) @ b['M']
        d = np.stack([np.maximum(0, np.abs(rel[:, 0]) - b['hw']), np.maximum(0, np.abs(rel[:, 1]) - b['hd']), np.maximum(0, np.abs(rel[:, 2]) - b['th'])], 1)
        pen = R - np.linalg.norm(d, axis=1); k = pen > 0.01
        if k.any():
            key = f"{n}->{b['ball']}:{b['kind']}"; bad.setdefault(key, []).append((round(b['t'], 2), round(float(ts[m][k][0]), 3), round(float(pen[k].max()), 3)))
            worst = max(worst, float(pen[k].max()))
info['collisions'] = {k: len(v) for k, v in bad.items()}; info['collision_examples'] = {k: v[:3] for k, v in bad.items()}
info['collision_worst_pen'] = round(worst, 3)
# 4) per-frame motion: free flight == gravity, speed ratio between frames
tf = np.arange(NF) / FPS
evt = sorted([h['t'] for n in NAMES for h in B[n]['hits']] + [g['t0'] for n in NAMES for g in B[n]['seg']])
amax = 0.0; rat = 1.0; rat2 = 1.0; rat_roll = 1.0
for n in NAMES:
    Q_ = np.array([at(B[n]['seg'], t) for t in tf]); segs = B[n]['seg']
    for f in range(1, NF - 1):
        t0, t1 = tf[f - 1], tf[f + 1]
        lo = np.searchsorted(evt, t0 - 1e-6); hi = np.searchsorted(evt, t1 + 1e-6)
        if hi > lo: continue
        g = [s for s in segs if s['t0'] - 1e-9 <= tf[f] <= s['t1'] + 1e-9][0]
        if g['k'] == 'rest': continue
        s0 = np.linalg.norm(Q_[f] - Q_[f - 1]) * FPS; s1 = np.linalg.norm(Q_[f + 1] - Q_[f]) * FPS
        if g['k'] == 'fly':
            a = (Q_[f + 1] - 2 * Q_[f] + Q_[f - 1]) * FPS * FPS
            amax = max(amax, float(np.linalg.norm(a - np.array([0, 0, -G]))))
            if min(s0, s1) > 0.5: rat = max(rat, max(s0, s1) / min(s0, s1))
            if min(s0, s1) > 2.0: rat2 = max(rat2, max(s0, s1) / min(s0, s1))
        else: rat_roll = max(rat_roll, max(s0, s1) / max(1e-9, min(s0, s1)))
info['flight_accel_error_max'] = round(amax, 6); info['flight_speed_ratio_max(v>0.5)'] = round(rat, 3)
info['flight_speed_ratio_max(v>2)'] = round(rat2, 3); info['rolling_speed_ratio_per_frame_max'] = round(rat_roll, 4)
hs = [h for n in NAMES[1:] for h in B[n]['hits'] if h['kind'] != 'cup']
info['car_restitution_range'] = (round(min(h['e'] for h in hs), 3), round(max(h['e'] for h in hs), 3))
info['car_bounce_speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in hs), 3)
lj = [h for h in lh if h['kind'] in ('land', 'bell')]
info['loco_impacts_speed_out_over_in'] = [round(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']), 3) for h in lj]
# 5) ball-ball distance
mind = 9.0
for i in range(5):
    for j in range(i + 1, 5):
        d = np.linalg.norm(P[NAMES[i]] - P[NAMES[j]], axis=1) - B[NAMES[i]]['R'] - B[NAMES[j]]['R']; mind = min(mind, float(d[ts > 11.2].min()))
info['ball_ball_min_gap_after_drop'] = round(mind, 3)
for k, v in info.items(): print(k, v)
