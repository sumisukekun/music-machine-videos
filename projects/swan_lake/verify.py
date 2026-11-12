"""Checks for swan_lake (no Blender needed): python3 verify.py [table]
 - every melody note is struck at its exact time by the ball(s) that own it (both balls in the tutti), pitch per note
 - contact distance between ball and key on the rendered frame nearest each hit
 - collisions: a ball entering any key other than its own at its own hit; the moon ring; the other ball
 - free flight: per-frame acceleration == gravity, frame-to-frame speed ratio; bounce speed out/in (wing keys excluded)"""
import json, os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(f'{HERE}/layout.json')); LEAD = json.load(open(f'{HERE}/lead.json')); E = json.load(open(f'{HERE}/events.json'))
SRC = json.load(open(f'{HERE}/score_src.json'))
G = L['G']; R = L['R']; FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.5) * FPS)
RG = L['ring']; RT = np.array(L['ev']['ring']['t']); RTH = np.array(L['ev']['ring']['th'])
def theta(g, t): return float(np.interp(t, RT, RTH)) + (0 if g == 'A' else math.pi)
def gball(g, t):
    th = theta(g, t); return np.array([RG['r'] * math.cos(th), 0.0, RG['zc'] + RG['r'] * math.sin(th) - RG['hang']])
KEY = {'plate': (0.13, 0.36), 'key': (0.17, 0.5), 'wing': (0.2, 0.62), 'lever': (0.22, 0.44), 'winglever': (0.22, 0.5)}
TH_KEY = 0.07


def at(seg, t):
    for g in seg:
        if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9:
            if g['k'] == 'rest': return np.array(g['p'])
            if g['k'] == 'ride': return gball(g['g'], t)
            u = t - g['t0']; return np.array(g['p']) + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
    return at(seg, seg[0]['t0']) if t < seg[0]['t0'] else at(seg, seg[-1]['t1'])


def key_frame(h):
    n = np.array(h['n'], float); n /= np.linalg.norm(n)
    d = np.array(h['vin'], float) + np.array(h['vout'], float); d[2] = 0
    if np.linalg.norm(d) < 1e-6: d = np.array([1.0, 0, 0])
    xa = d - (d @ n) * n; xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return np.stack([xa, ya, n], 1)


keys = []
for name, b in L['balls'].items():
    for h in b['hits']:
        if h['kind'] == 'bell': continue
        n = np.array(h['n'], float); n /= np.linalg.norm(n); c = np.array(h['p']) - R * n
        w, d = KEY[h['kind']]
        keys.append(dict(ball=name, i=h['i'], t=h['t'], kind=h['kind'], c=c - n * TH_KEY / 2, M=key_frame(h), hw=w / 2, hd=d / 2,
                         th=TH_KEY / 2, contact=c, n=n))
info = {}
# 1) note coverage, owners, pitch / time table
hitmap = {}
for name, b in L['balls'].items():
    for h in b['hits']:
        if h['i'] is not None: hitmap.setdefault(h['i'], []).append((name, h['t'], h['kind']))
miss = [i for i in range(len(LEAD)) if i not in hitmap]
wrong_owner = [i for i in hitmap if sorted(n for n, _, _ in hitmap[i]) != sorted(LEAD[i][5])]
dt_err = max(abs(t - LEAD[i][0]) for i, v in hitmap.items() for _, t, _ in v)
info['notes'] = len(LEAD); info['notes_without_visible_hit'] = miss; info['owner_mismatch'] = wrong_owner
info['hit_time_error_max_s'] = round(dt_err, 6)
# melody vs the source score (oboe bars 2-18, horns bars 19-25 +12, harp bar 1 +12)
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
ref = [r[3] + 12 for r in SRC['harp_intro']] + [r[3] for r in SRC['oboe'] if 2 <= r[0] <= 18] + \
      [p + 12 for (_, _, _, p) in sorted(set(tuple(r) for r in SRC['horn'] if 19 <= r[0] <= 25))] + [71]
info['melody_vs_source_mismatch'] = [i for i, (l, p) in enumerate(zip(LEAD, ref)) if l[1] != p] + ([] if len(ref) == len(LEAD) else ['len'])
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    for i, l in enumerate(LEAD):
        who = ','.join(f'{n}:{k}' for n, _, k in hitmap[i])
        print(f'{i:3d} {l[0]:7.3f}s {l[4]:6s} {NM[l[1] % 12]}{l[1] // 12 - 1:<2d} frame {round(l[0] * FPS):5d}  {who}')
# 2) contact distance on the nearest frame
gaps = []
for K in keys:
    f = round(K['t'] * FPS); p = at(L['balls'][K['ball']]['seg'], f / FPS)
    gaps.append(np.linalg.norm(p - (K['contact'] + R * K['n'])))
info['contact_gap_nearest_frame_max_m'] = round(float(max(gaps)), 3); info['contact_gap_mean_m'] = round(float(np.mean(gaps)), 3)
# 3) collisions (sampled at 240 Hz) with keys, the ring, the other ball
ts = np.arange(0.0, FINAL + 1.0, 1 / 240)
P = {n: np.array([at(b['seg'], t) for t in ts]) for n, b in L['balls'].items()}
riding = {n: np.array([any(g['k'] == 'ride' and g['t0'] <= t <= g['t1'] for g in b['seg']) for t in ts]) for n, b in L['balls'].items()}
bad = {}; worst = 0.0
for K in keys:
    for n, Pn in P.items():
        m = (np.abs(Pn[:, 0] - K['c'][0]) < 0.8) & (np.abs(Pn[:, 1] - K['c'][1]) < 0.8) & (np.abs(Pn[:, 2] - K['c'][2]) < 0.8)
        if K['ball'] == n: m &= np.abs(ts - K['t']) > 0.03
        if not m.any(): continue
        rel = (Pn[m] - K['c']) @ K['M']
        d = np.stack([np.maximum(0, np.abs(rel[:, 0]) - K['hw']), np.maximum(0, np.abs(rel[:, 1]) - K['hd']), np.maximum(0, np.abs(rel[:, 2]) - K['th'])], 1)
        pen = R - np.linalg.norm(d, axis=1)
        k = pen > 0.005
        if k.any():
            key = (n, K['ball'], K['kind']); bad.setdefault(key, []).append((K['i'], round(float(ts[m][k][0]), 3), round(float(pen[k].max()), 3)))
            worst = max(worst, float(pen[k].max()))
info['key_collisions'] = {f'{a}->{b}:{c}': len(v) for (a, b, c), v in bad.items()}
info['key_collision_examples'] = {f'{a}->{b}:{c}': v[:4] for (a, b, c), v in bad.items()}
info['key_collision_worst_pen_m'] = round(worst, 3)
rp = {}
for n, Pn in P.items():
    rr = np.sqrt(Pn[:, 0] ** 2 + (Pn[:, 2] - RG['zc']) ** 2)
    d = np.sqrt((rr - RG['r']) ** 2 + Pn[:, 1] ** 2) - (0.1 + R)
    d[riding[n]] = 9
    rp[n] = (round(float(d.min()), 3), round(float(ts[np.argmin(d)]), 2))
info['ring_clearance_min_m(ball surface to ring tube)'] = rp
dd = np.linalg.norm(P['W'] - P['K'], axis=1) - 2 * R
info['ball_ball_surface_gap_min_m'] = (round(float(dd.min()), 3), round(float(ts[np.argmin(dd)]), 2))
# 4) free flight: acceleration per frame equals gravity; speed ratio between frames
tf = np.arange(NF) / FPS
evt = np.array(sorted([h['t'] for b in L['balls'].values() for h in b['hits']] + [g['t0'] for b in L['balls'].values() for g in b['seg']]))
amax = 0.0; rat = 1.0; where = None
for n, b in L['balls'].items():
    Q = np.array([at(b['seg'], t) for t in tf])
    fly = [(g['t0'], g['t1']) for g in b['seg'] if g['k'] == 'fly']
    for f in range(1, NF - 1):
        t0, t1 = tf[f - 1], tf[f + 1]
        if not any(a - 1e-9 <= t0 and t1 <= c + 1e-9 for a, c in fly): continue
        a = (Q[f + 1] - 2 * Q[f] + Q[f - 1]) * FPS * FPS
        amax = max(amax, float(np.linalg.norm(a - np.array([0, 0, -G]))))
        s0 = np.linalg.norm(Q[f] - Q[f - 1]) * FPS; s1 = np.linalg.norm(Q[f + 1] - Q[f]) * FPS
        if min(s0, s1) > 1.0 and max(s0, s1) / min(s0, s1) > rat: rat = max(s0, s1) / min(s0, s1); where = (n, round(tf[f], 2), round(s0, 2), round(s1, 2))
info['flight_accel_error_max'] = round(amax, 6); info['flight_speed_ratio_max(v>1)'] = round(rat, 3); info['worst_ratio_at'] = where
hs = [h for b in L['balls'].values() for h in b['hits'] if h['kind'] in ('key', 'plate', 'lever')]
info['bounce_speed_out_over_in_max(keys, plates, levers)'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in hs), 3)
info['restitution_range(keys)'] = (round(min(h['e'] for h in hs if h['kind'] == 'key'), 3), round(max(h['e'] for h in hs), 3))
wg = [h for b in L['balls'].values() for h in b['hits'] if 'wing' in h['kind']]
info['wing_keys(spring launch)'] = (len(wg), round(max(h['e'] for h in wg), 2))
info['key_tilt_max_deg'] = round(max(math.degrees(math.acos(min(1, h['n'][2]))) for b in L['balls'].values() for h in b['hits']), 1)
for k, v in info.items(): print(k, v)
