"""Checks for deck_halls (no Blender needed): python3 verify.py [table]
 - every melody note is struck at its exact time by a ball, with the score's pitch; every bass note too (verse 2)
 - pitch/time table against the source (score_src.json, Wood, transposed to F)
 - contact distance between ball and struck surface on the frame nearest each hit
 - collisions: a ball's surface entering any struck surface other than its own hit, ball-ball overlaps (incl. resting ornaments)
 - free flight: per-frame acceleration == gravity, frame-to-frame speed ratio; bounces: speed out / speed in <= 1
 - mechanisms (path segments): no jump in speed between consecutive frames larger than the device's smooth motion allows"""
import json, os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(f'{HERE}/layout.json')); LEAD = json.load(open(f'{HERE}/lead.json')); E = json.load(open(f'{HERE}/events.json'))
SRC = json.load(open(f'{HERE}/score_src.json'))
G = L['G']; FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.3) * FPS)
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def at(seg, t):
    for g in seg:
        if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9:
            if g['k'] == 'rest': return np.array(g['p'])
            if g['k'] == 'fly':
                u = t - g['t0']; return np.array(g['p']) + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
            P = np.array(g['pts']); u = (t - g['t0']) / max(1e-9, g['t1'] - g['t0']) * (len(P) - 1)
            i = min(len(P) - 2, int(u)); f = u - i; return P[i] * (1 - f) + P[i + 1] * f
    return at(seg, seg[0]['t0']) if t < seg[0]['t0'] else at(seg, seg[-1]['t1'])


def frame(n):
    n = np.array(n, float); n /= np.linalg.norm(n)
    xa = np.cross([0, 1, 0], n); xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return np.stack([xa, ya, n], 1)


DIMS = {'key': (0.11, 0.21, 0.03), 'bell': (0.07, 0.07, 0.03), 'bass': (0.14, 0.17, 0.04), 'trigger': (0.16, 0.17, 0.04),
        'bassfinal': (0.2, 0.2, 0.04)}
BALLS = L['balls']
surf = []
for name, b in BALLS.items():
    for h in b['hits']:
        if h['kind'] not in DIMS: continue
        n = np.array(h['n'], float); n /= np.linalg.norm(n); c = np.array(h['p']) - b['r'] * n
        hw, hd, th = DIMS[h['kind']]
        surf.append(dict(ball=name, t=h['t'], kind=h['kind'], c=c - n * th, M=frame(n), hw=hw, hd=hd, th=th, contact=c, n=n, r=b['r']))
info = {}
# 1) notes: every melody note struck once at its time with its pitch; bass notes likewise
mel = {}
for name, b in BALLS.items():
    for h in b['hits']:
        if h['i'] >= 0: mel.setdefault(h['i'], []).append((name, h['t'], h['pitch'], h['kind']))
info['melody_notes'] = len(LEAD)
info['melody_missing_or_wrong'] = [i for i in range(len(LEAD)) if i not in mel or any(abs(t - LEAD[i][0]) > 1e-6 or p != LEAD[i][1] for _, t, p, _ in mel[i])]
info['melody_struck_twice'] = [i for i, v in mel.items() if len(v) > 1]
bh = [h for h in BALLS['yule']['hits']]
info['bass_notes'] = len(E['bass'])
info['bass_missing_or_wrong'] = [j for j, (t, p, gb, pos) in enumerate(E['bass']) if not any(abs(h['t'] - t) < 1e-6 and h['pitch'] == p for h in bh)]
# melody against the source (Wood, E-flat +2 = F), both verses
SOP = [e for e in SRC['wood_eb'] if e[4] == 'S']
src_err = 0
for i, l in enumerate(LEAD):
    e = SOP[i % 68]
    if not (e[0] == (l[4] - 1) % 16 + 1 and abs(e[1] - l[5]) < 1e-6 and e[3] + 2 == l[1]): src_err += 1
info['melody_vs_source_mismatch'] = src_err
BASSV = [e for e in SRC['wood_eb'] if e[4] == 'B']
berr = 0
for t, p, gb, pos in E['bass']:
    bar = gb - 16
    s_ = [e for e in BASSV if e[0] == bar and e[1] <= pos + 1e-6 < e[1] + e[2]]
    if not s_ or s_[0][3] + 2 != p: berr += 1
info['bass_vs_source_mismatch'] = berr
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    for i, l in enumerate(LEAD):
        who = ','.join(f'{n}:{k}' for n, _, _, k in mel.get(i, []))
        print(f'{i:3d} {l[0]:7.3f}s bar{l[4]:3d} beat {l[5] + 1:<4g} {NM[l[1] % 12]}{l[1] // 12 - 1:<2d} frame {round(l[0] * FPS):5d}  {l[6]:5s} {who}')
    for t, p, gb, pos in E['bass']:
        print(f'bass {t:7.3f}s bar{gb:3d} beat {pos + 1:g} {NM[p % 12]}{p // 12 - 1}')
# 2) contact distance on the nearest frame
gaps = {}
for S in surf:
    f = round(S['t'] * FPS); p = at(BALLS[S['ball']]['seg'], f / FPS)
    gaps.setdefault(S['kind'], []).append(np.linalg.norm(p - (S['contact'] + S['r'] * S['n'])))
info['contact_gap_nearest_frame(max,mean)'] = {k: (round(float(max(v)), 3), round(float(np.mean(v)), 3)) for k, v in gaps.items()}
# 3) collisions with struck surfaces (240 Hz)
ts = np.arange(0.0, FINAL + 1.5, 1 / 240)
P = {n: np.array([at(b['seg'], t) for t in ts]) for n, b in BALLS.items()}
bad = {}; worst = 0.0
for S in surf:
    for n, Pn in P.items():
        r = BALLS[n]['r']
        m = (np.abs(Pn[:, 0] - S['c'][0]) < 0.6) & (np.abs(Pn[:, 2] - S['c'][2]) < 0.6) & (np.abs(Pn[:, 1] - S['c'][1]) < 0.6)
        if S['ball'] == n: m &= np.abs(ts - S['t']) > 0.03
        if S['kind'] == 'bassfinal' and n == 'yule': m &= ts < S['t']
        if not m.any(): continue
        rel = (Pn[m] - S['c']) @ S['M']
        d = np.stack([np.maximum(0, np.abs(rel[:, 0]) - S['hw']), np.maximum(0, np.abs(rel[:, 1]) - S['hd']), np.maximum(0, np.abs(rel[:, 2]) - S['th'])], 1)
        pen = r - np.linalg.norm(d, axis=1); k = pen > 0.005
        if k.any():
            key = f"{n}->{S['ball']}:{S['kind']}"; bad.setdefault(key, []).append((round(float(ts[m][k][0]), 3), round(float(pen[k].max()), 3)))
            worst = max(worst, float(pen[k].max()))
info['collisions'] = {k: len(v) for k, v in bad.items()}
info['collision_examples'] = {k: v[:3] for k, v in bad.items()}
info['collision_worst_pen'] = round(worst, 3)
# ball-ball (surface gap), all pairs, 240 Hz
names = list(BALLS); mind = (9, None)
X = np.stack([P[n] for n in names]); RR = np.array([BALLS[n]['r'] for n in names])
for a in range(len(names)):
    d = np.linalg.norm(X[a + 1:] - X[a][None], axis=2) - RR[a + 1:, None] - RR[a]
    if d.size and d.min() < mind[0]:
        j, f = np.unravel_index(np.argmin(d), d.shape); mind = (float(d.min()), (names[a], names[a + 1 + j], round(float(ts[f]), 2)))
info['min_ball_surface_gap'] = (round(mind[0], 3), mind[1])
# 4) free flight + device motion per frame
tf = np.arange(NF) / FPS
evt = sorted(set([h['t'] for b in BALLS.values() for h in b['hits']] + [g['t0'] for b in BALLS.values() for g in b['seg']]))
amax = 0.0; rat = (1.0, None); rat2 = (1.0, None); devjerk = (0.0, None)
for n, b in BALLS.items():
    Q = np.array([at(b['seg'], t) for t in tf])
    for f in range(1, NF - 1):
        t0, t1 = tf[f - 1], tf[f + 1]
        near = any(t0 - 1e-6 <= e <= t1 + 1e-6 for e in evt[np.searchsorted(evt, t0 - 0.1):np.searchsorted(evt, t1 + 0.1)])
        if near: continue
        if np.linalg.norm(Q[f + 1] - Q[f - 1]) < 1e-9: continue
        kind = [g['k'] for g in b['seg'] if g['t0'] <= tf[f] <= g['t1']]
        a = (Q[f + 1] - 2 * Q[f] + Q[f - 1]) * FPS * FPS
        if kind and kind[0] == 'fly': amax = max(amax, float(np.linalg.norm(a - np.array([0, 0, -G]))))
        elif float(np.linalg.norm(a)) > devjerk[0]: devjerk = (float(np.linalg.norm(a)), (n, round(tf[f], 2)))
        s0 = np.linalg.norm(Q[f] - Q[f - 1]) * FPS; s1 = np.linalg.norm(Q[f + 1] - Q[f]) * FPS
        if kind and kind[0] == 'fly' and min(s0, s1) > 0.5 and max(s0, s1) / min(s0, s1) > rat[0]: rat = (max(s0, s1) / min(s0, s1), (n, round(tf[f], 2), round(s0, 2), round(s1, 2)))
        if kind and kind[0] == 'fly' and min(s0, s1) > 1.5 and max(s0, s1) / min(s0, s1) > rat2[0]: rat2 = (max(s0, s1) / min(s0, s1), (n, round(tf[f], 2), round(s0, 2), round(s1, 2)))
info['flight_accel_error_max'] = round(amax, 6)
info['flight_speed_ratio_max(v>0.5)'] = (round(rat[0], 3), rat[1])
info['flight_speed_ratio_max(v>1.5)'] = (round(rat2[0], 3), rat2[1])
info['mechanism_accel_max'] = (round(devjerk[0], 2), devjerk[1])
hs = [h for b in BALLS.values() for h in b['hits'] if np.linalg.norm(h['vout']) > 0]
info['bounce_speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in hs), 4)
info['restitution_range'] = (round(min(h['e'] for h in hs), 3), round(max(h['e'] for h in hs), 3))
info['key_tilt_max_deg'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for h in hs), 1)
info['video_length_s'] = round(NF / FPS, 2)
for k, v in info.items(): print(k, v)
