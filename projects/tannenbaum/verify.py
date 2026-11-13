"""Checks for tannenbaum (no Blender needed): python3 verify.py [table]
 - every melody note is struck at its exact time by a ball, with the score's pitch; every bass note too (verses 2, 3)
 - pitch/time table against the source (score_src.json, SATB setting in F)
 - contact distance between ball and struck surface on the frame nearest each hit
 - collisions: a ball's surface entering any struck surface other than its own hit, ball-ball overlaps (incl. resting balls)
 - free flight: per-frame acceleration == gravity, frame-to-frame speed ratio; bounces: speed out / speed in <= 1
 - mechanisms (path segments): the largest per-frame acceleration (smooth machines; the catapult launch is an impulse)"""
import json, os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(f'{HERE}/layout.json')); LEAD = json.load(open(f'{HERE}/lead.json')); E = json.load(open(f'{HERE}/events.json'))
SRC = json.load(open(f'{HERE}/score_src.json'))
G = L['G']; FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.6) * FPS)
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


# struck surfaces: half sizes (x, y, thickness/2) as built in scene.py
DIMS = {'key': (0.06, 0.21, 0.03), 'bell': (0.07, 0.07, 0.03), 'bass': (0.2, 0.2, 0.15), 'trigger': (0.2, 0.2, 0.15),
        'bassfinal': (0.25, 0.25, 0.15)}
BALLS = L['balls']
surf = []
for name, b in BALLS.items():
    for h in b['hits']:
        n = np.array(h['n'], float); n /= np.linalg.norm(n); c = np.array(h['p']) - b['r'] * n
        if h['kind'] == 'rollkey':          # a long key: from 0.07 behind the contact to 0.07 past the roll-off point
            d = 1 if h['roll'] >= 0 else -1; hw = (abs(h['roll']) + 0.07 - 0.05) / 2; cc = c + np.array([d * (hw - 0.07), 0, 0])
            surf.append(dict(ball=name, t=h['t'], t1=h['t'] + h['t_roll'], kind='rollkey', c=cc - n * 0.03, M=frame(n), hw=hw, hd=0.21,
                             th=0.03, contact=c, n=n, r=b['r']))
            continue
        if h['kind'] not in DIMS: continue
        hw, hd, th = DIMS[h['kind']]
        surf.append(dict(ball=name, t=h['t'], t1=h['t'], kind=h['kind'], c=c - n * th, M=frame(n), hw=hw, hd=hd, th=th, contact=c, n=n, r=b['r']))
info = {}
# 1) notes
mel = {}
for name, b in BALLS.items():
    for h in b['hits']:
        if h['i'] >= 0: mel.setdefault(h['i'], []).append((name, h['t'], h['pitch'], h['kind']))
info['melody_notes'] = len(LEAD)
info['melody_missing_or_wrong'] = [i for i in range(len(LEAD)) if i not in mel or any(abs(t - LEAD[i][0]) > 1e-6 or p != LEAD[i][1] for _, t, p, _ in mel[i])]
info['melody_struck_twice'] = [i for i, v in mel.items() if len(v) > 1]
bh = [h for n, b in BALLS.items() if b['kind'] == 'pine' for h in b['hits']]
info['bass_notes'] = len(E['bass'])
info['bass_missing_or_wrong'] = [j for j, (t, p, *_r) in enumerate(E['bass']) if sum(abs(h['t'] - t) < 1e-6 and h['pitch'] == p for h in bh) != 1]
# melody against the source (SATB soprano, F major), all three verses
SOP = [e for e in SRC['satb_f'] if e[4] == 'S']
src_err = 0
for l in LEAD:
    v, m, pos, p = l[4], l[5], l[6], l[1]
    if not any(e[0] == m and abs(e[1] - pos) < 1e-6 and e[3] == p for e in SOP): src_err += 1
info['melody_vs_source_mismatch'] = src_err
info['melody_per_verse'] = [sum(1 for l in LEAD if l[4] == v) for v in range(3)]
BASSV = [e for e in SRC['satb_f'] if e[4] == 'B']
berr = 0
for t, p, v, m, pos, k in E['bass']:
    s_ = [e for e in BASSV if e[0] == m and e[1] <= pos + 1e-6 < e[1] + e[2]]
    if not s_ or s_[0][3] != p: berr += 1
info['bass_vs_source_mismatch'] = berr
# timing grid: every note time = T0 + Q * (beats from the first pickup) (+ the final ritardando)
q = E['q']; t0 = E['t0']
grid = [abs(((l[0] - t0) / q) * 4 - round(((l[0] - t0) / q) * 4)) for l in LEAD[:-1]]
info['melody_off_16th_grid_max_s'] = round(max(grid) * q / 4, 6)
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    for i, l in enumerate(LEAD):
        who = ','.join(f'{n}:{k}' for n, _, _, k in mel.get(i, []))
        print(f'{i:3d} {l[0]:7.3f}s verse{l[4] + 1} bar{l[5]:3d} beat {l[6] + 1:<5g} {NM[l[1] % 12]}{l[1] // 12 - 1:<2d} frame {round(l[0] * FPS):5d}  {l[7]:5s} {who}')
    for t, p, v, m, pos, k in E['bass']:
        print(f'bass {t:7.3f}s verse{v + 1} bar{m:3d} beat {pos + 1:g} {NM[p % 12]}{p // 12 - 1} frame {round(t * FPS):5d}')
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
        m = (np.abs(Pn[:, 0] - S['c'][0]) < S['hw'] + 0.6) & (np.abs(Pn[:, 2] - S['c'][2]) < 0.6) & (np.abs(Pn[:, 1] - S['c'][1]) < 0.6)
        if S['ball'] == n: m &= (ts < S['t'] - 0.03) | (ts > S['t1'] + 0.03)
        if S['kind'] == 'bassfinal' and S['ball'] == n: m &= ts < S['t']
        if not m.any(): continue
        rel = (Pn[m] - S['c']) @ S['M']
        dd = np.stack([np.maximum(0, np.abs(rel[:, 0]) - S['hw']), np.maximum(0, np.abs(rel[:, 1]) - S['hd']), np.maximum(0, np.abs(rel[:, 2]) - S['th'])], 1)
        pen = r - np.linalg.norm(dd, axis=1); k = pen > 0.005
        if k.any():
            key = f"{n}->{S['ball']}:{S['kind']}"; bad.setdefault(key, []).append((round(float(ts[m][k][0]), 3), round(float(pen[k].max()), 3)))
            worst = max(worst, float(pen[k].max()))
info['collisions'] = {k: len(v) for k, v in bad.items()}
info['collision_examples'] = {k: v[:3] for k, v in bad.items()}
info['collision_worst_pen'] = round(worst, 3)
names = list(BALLS); mind = (9, None)
X = np.stack([P[n] for n in names]); RR = np.array([BALLS[n]['r'] for n in names])
for a in range(len(names)):
    dd = np.linalg.norm(X[a + 1:] - X[a][None], axis=2) - RR[a + 1:, None] - RR[a]
    if dd.size and dd.min() < mind[0]:
        j, f = np.unravel_index(np.argmin(dd), dd.shape); mind = (float(dd.min()), (names[a], names[a + 1 + j], round(float(ts[f]), 2)))
info['min_ball_surface_gap'] = (round(mind[0], 3), mind[1])
# 4) free flight + device motion per frame
tf = np.arange(NF) / FPS
evt = sorted(set([h['t'] for b in BALLS.values() for h in b['hits']] + [g['t0'] for b in BALLS.values() for g in b['seg']]))
devk = {}; amax = 0.0; rat = (1.0, None); rat2 = (1.0, None); devacc = (0.0, None)
for n, b in BALLS.items():
    Q = np.array([at(b['seg'], t) for t in tf])
    for f in range(1, NF - 1):
        t0_, t1_ = tf[f - 1], tf[f + 1]
        near = any(t0_ - 1e-6 <= e <= t1_ + 1e-6 for e in evt[np.searchsorted(evt, t0_ - 0.1):np.searchsorted(evt, t1_ + 0.1)])
        if near: continue
        if np.linalg.norm(Q[f + 1] - Q[f - 1]) < 1e-9: continue
        kind = [g['k'] for g in b['seg'] if g['t0'] <= tf[f] <= g['t1']]
        a = (Q[f + 1] - 2 * Q[f] + Q[f - 1]) * FPS * FPS
        if kind and kind[0] == 'fly': amax = max(amax, float(np.linalg.norm(a - np.array([0, 0, -G]))))
        else:
            kk = b['kind']
            if float(np.linalg.norm(a)) > devk.get(kk, (0.0,))[0]: devk[kk] = (round(float(np.linalg.norm(a)), 2), round(tf[f], 2))
            if float(np.linalg.norm(a)) > devacc[0]: devacc = (float(np.linalg.norm(a)), (n, round(tf[f], 2)))
        s0 = np.linalg.norm(Q[f] - Q[f - 1]) * FPS; s1 = np.linalg.norm(Q[f + 1] - Q[f]) * FPS
        if kind and kind[0] == 'fly' and min(s0, s1) > 0.5 and max(s0, s1) / min(s0, s1) > rat[0]: rat = (max(s0, s1) / min(s0, s1), (n, round(tf[f], 2), round(s0, 2), round(s1, 2)))
        if kind and kind[0] == 'fly' and min(s0, s1) > 1.5 and max(s0, s1) / min(s0, s1) > rat2[0]: rat2 = (max(s0, s1) / min(s0, s1), (n, round(tf[f], 2), round(s0, 2), round(s1, 2)))
info['flight_accel_error_max'] = round(amax, 6)
info['flight_speed_ratio_max(v>0.5)'] = (round(rat[0], 3), rat[1])
info['flight_speed_ratio_max(v>1.5)'] = (round(rat2[0], 3), rat2[1])
info['mechanism_accel_max'] = (round(devacc[0], 2), devacc[1])
info['mechanism_accel_max_by_ball(machines move Stella; snowball = plank after the slam)'] = devk
hs = [h for b in BALLS.values() for h in b['hits'] if np.linalg.norm(h['vout']) > 0]
info['bounce_speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in hs), 4)
info['restitution_range'] = (round(min(h['e'] for h in hs if h['kind'] != 'rollkey'), 3), round(max(h['e'] for h in hs), 3))
info['key_tilt_max_deg'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for h in hs), 1)
lv = [np.linalg.norm(d['v0']) for d in L['ev']['dev'] if d['kind'] == 'catapult']
info['catapult_launch_speed'] = [round(float(v), 2) for v in lv]
info['video_length_s'] = round(NF / FPS, 2)
for k, v in info.items(): print(k, v)
