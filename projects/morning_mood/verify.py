"""Checks for morning_mood (no Blender needed): python3 verify.py [table]
 - every melody note is struck at its exact time by the right ball with the right pitch; every accompaniment onset
   (score_src.json harmony, reduced as in arrange.py) is rung by a bell gate at its exact time
 - contact distance between ball and key on the rendered frame nearest each hit, and at the exact time
 - collisions: ball surface entering any key other than its own hit (sampled at 240 Hz), ball-ball overlap
 - free flight: per-frame acceleration == gravity, frame-to-frame speed ratio; every bounce: e <= emax, |v_out| <= |v_in|
 - mechanisms (ramps, gondola, rail, cups): max speed change per frame"""
import json, os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(f'{HERE}/layout.json')); LEAD = json.load(open(f'{HERE}/lead.json')); E = json.load(open(f'{HERE}/events.json'))
S = json.load(open(f'{HERE}/score_src.json'))
G = L['G']; FPS = 30; FINAL = E['final']; NF = int((FINAL + 6.0) * FPS)
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'


def seg_at(g, t):
    if g['k'] == 'rest': return np.array(g['p'])
    if g['k'] == 'fly':
        u = t - g['t0']; return np.array(g['p']) + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
    ts = np.array(g['ts']); ps = np.array(g['ps'])
    return np.array([np.interp(t, ts, ps[:, c]) for c in range(3)])


def at(b, t):
    for g in b['seg']:
        if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9: return seg_at(g, t)
    return seg_at(b['seg'][0], b['seg'][0]['t0']) if t < b['seg'][0]['t0'] else seg_at(b['seg'][-1], b['seg'][-1]['t1'])


def frame(n):
    n = np.array(n, float); n /= np.linalg.norm(n)
    xa = np.cross([0, 1, 0], n); xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return np.stack([xa, ya, n], 1)


info = {}
# 1) melody vs the score (score_src melody bars 1-23 + the bar-24 ending), voice -> ball
src = [e for e in S['melody'] if e[0] <= 23] + [[24, 0, 1, 83], [24, 1, 1, 80], [24, 2, 1, 78], [24, 3, 3, 76]]
src.sort(key=lambda e: (e[0], e[1]))
T0, E8 = E['t0'], E['e8']
st = {int(k): v for k, v in L['struck'].items()}
bad = []; table = []; t_err = 0.0
for i, (bar, off, dur, p) in enumerate(src):
    t = T0 + (bar - 1) * 6 * E8 + off * E8
    l = LEAD[i]
    want = [p] if bar < 21 else [p, p - 12]
    who = st.get(i, [])
    got = sorted(set(q for _, q, _ in who))
    if abs(l[0] - t) > 1e-6 or sorted(want) != got: bad.append(i)
    for n in ('flute', 'oboe'):
        for h in L['balls'][n]['hits']:
            if h.get('ev') == i: t_err = max(t_err, abs(h['t'] - t))
    vo = 'flute' if l[4] == 'fl' else ('oboe' if l[4] == 'ob' else 'both')
    if vo != 'both' and any(n != vo for n, _, _ in who): bad.append(('wrong ball', i))
    table.append((i, t, bar, off, want, got, who))
info['melody_notes'] = len(src); info['melody_wrong_or_missing'] = bad; info['melody_time_error_max_s'] = round(t_err, 9)
# 2) accompaniment gates vs the score harmony onsets
from collections import defaultdict
on = defaultdict(set)
for bar, off, dur, p in S['harmony']: on[(bar, off)].add(p)
on[(15, 3.0)] = {49, 56, 64}; on[(24, 0.0)] = {40, 64, 68, 71}; on[(24, 3.0)] = {40, 52, 64, 68, 71}
gb = []
G_ = {(g['bar'], g['off']): g for g in L['ev']['gates']}
for k, ps in on.items():
    ps = sorted(ps); red = [ps[0]] + ps[-3:] if len(ps) > 4 else ps
    g = G_.get(k)
    t = T0 + (k[0] - 1) * 6 * E8 + k[1] * E8
    hit = [h for h in L['balls']['dawn']['hits'] if h['kind'] == 'gate' and abs(h['t'] - t) < 1e-6]
    if g is None or sorted(g['pitches']) != red or not hit: gb.append((k, red))
info['accomp_onsets'] = len(on); info['accomp_wrong_or_missing'] = gb
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    for i, t, bar, off, want, got, who in table:
        print(f'{i:3d} {t:7.3f}s bar{bar:2d} 8th{off + 1:4.1f}  score {"+".join(nm(p) for p in want):8s} struck {"+".join(nm(p) for p in got):8s} '
              f'frame {round(t * FPS):5d}  ' + ' '.join(f'{n}:{nm(p)}:{kd}' for n, p, kd in who))
    for g in sorted(L['ev']['gates'], key=lambda g: g['t']):
        print(f'gate {g["t"]:7.3f}s bar{g["bar"]:2d} 8th{g["off"] + 1:4.1f} {g["kind"]:6s} {"+".join(nm(p) for p in g["pitches"])}  frame {round(g["t"] * FPS)}')
# 3) contact distance on the nearest frame
gaps = []
for n, b in L['balls'].items():
    for h in b['hits']:
        f = round(h['t'] * FPS); p = at(b, f / FPS)
        gaps.append((float(np.linalg.norm(p - np.array(h['p']))), n, h['kind'], round(h['t'], 3)))
gaps.sort(reverse=True)
info['contact_gap_nearest_frame_max_m'] = round(gaps[0][0], 3); info['contact_gap_mean_m'] = round(float(np.mean([g[0] for g in gaps])), 3)
info['contact_gap_worst'] = gaps[:3]
info['contact_gap_exact_time_max_m'] = round(max(float(np.linalg.norm(at(b, h['t']) - np.array(h['p']))) for b in L['balls'].values() for h in b['hits']), 6)
# 4) collisions with keys other than the ball's own hit (keys exist t-1.2 .. t+1.25 s)
R = {'flute': L['R']['mel'], 'oboe': L['R']['mel'], 'dawn': L['R']['dawn']}
pieces = []
for n in ('flute', 'oboe'):
    for h in L['balls'][n]['hits']:
        if h['kind'] in ('key', 'lever', 'pkey', 'reunion'):
            nv = np.array(h['n']); top = np.array(h['p']) - R[n] * nv
            pieces.append(dict(ball=n, t=h['t'], c=top - nv * 0.06, M=frame(nv), hw=0.21, hd=0.28 if h['kind'] in ('key', 'lever') else 0.2, th=0.06, kind=h['kind']))
ts = np.arange(0.2, FINAL + 1.0, 1 / 240)
P = {n: np.array([at(b, t) for t in ts]) for n, b in L['balls'].items()}
badc = {}; worst = 0.0
for B in pieces:
    for n, Pn in P.items():
        r = R[n]
        m = (np.abs(Pn[:, 0] - B['c'][0]) < 0.8) & (np.abs(Pn[:, 2] - B['c'][2]) < 0.8) & (np.abs(Pn[:, 1] - B['c'][1]) < 0.8)
        if B['ball'] == n or (B['kind'] in ('pkey', 'reunion') and n in ('flute', 'oboe')): m &= np.abs(ts - B['t']) > 0.03
        m &= (ts > B['t'] - 1.2) & (ts < B['t'] + 1.25)
        if not m.any(): continue
        rel = (Pn[m] - B['c']) @ B['M']
        d = np.stack([np.maximum(0, np.abs(rel[:, 0]) - B['hw']), np.maximum(0, np.abs(rel[:, 1]) - B['hd']), np.maximum(0, np.abs(rel[:, 2]) - B['th'])], 1)
        pen = r - np.linalg.norm(d, axis=1); k = pen > 0.01
        if k.any():
            key = f"{n}->{B['ball']}:{B['kind']}"; badc.setdefault(key, []).append((round(float(ts[m][k][0]), 3), round(float(pen[k].max()), 3), 'key_t', round(B['t'], 3)))
            worst = max(worst, float(pen[k].max()))
info['collisions'] = {k: len(v) for k, v in badc.items()}; info['collision_examples'] = {k: v[:3] for k, v in badc.items()}
info['collision_worst_pen_m'] = round(worst, 3)
names = list(P); mind = (9, None)
for a in range(len(names)):
    for c in range(a + 1, len(names)):
        d = np.linalg.norm(P[names[a]] - P[names[c]], axis=1) - R[names[a]] - R[names[c]]
        j = int(np.argmin(d))
        if d[j] < mind[0]: mind = (round(float(d[j]), 3), names[a], names[c], round(float(ts[j]), 2))
info['min_ball_surface_gap'] = mind
# 5) free flight per frame
tf = np.arange(NF) / FPS
amax = 0.0; rat = 1.0; worst_r = None; rat_fast = 1.0
for n, b in L['balls'].items():
    for g in b['seg']:
        if g['k'] != 'fly' or g['t1'] - g['t0'] <= 3 / FPS: continue
        for f in range(1, NF - 1):
            if not (g['t0'] + 1e-6 < tf[f - 1] and tf[f + 1] < g['t1'] - 1e-6): continue
            Q = [seg_at(g, tf[f + k]) for k in (-1, 0, 1)]
            a = (Q[2] - 2 * Q[1] + Q[0]) * FPS * FPS
            amax = max(amax, float(np.linalg.norm(a - np.array([0, 0, -G]))))
            s0 = np.linalg.norm(Q[1] - Q[0]) * FPS; s1 = np.linalg.norm(Q[2] - Q[1]) * FPS
            if min(s0, s1) > 0.5:
                r_ = max(s0, s1) / min(s0, s1)
                if r_ > rat: rat = r_; worst_r = (n, round(tf[f], 2), round(s0, 2), round(s1, 2))
            if min(s0, s1) > 2.0: rat_fast = max(rat_fast, max(s0, s1) / min(s0, s1))
info['flight_accel_error_max'] = round(amax, 6); info['flight_speed_ratio_max(v>0.5)'] = round(rat, 3); info['worst_ratio_at'] = worst_r
info['flight_speed_ratio_max(v>2)'] = round(rat_fast, 3)
dv = 0.0; dvw = None
for n, b in L['balls'].items():
    for g in b['seg']:
        if g['k'] != 'path' or g['t1'] - g['t0'] < 0.2: continue
        tt = np.arange(g['t0'], g['t1'] + 1e-9, 1 / FPS)
        Q = np.array([seg_at(g, t) for t in tt]); sp = np.linalg.norm(np.diff(Q, axis=0), axis=1) * FPS
        if len(sp) > 2:
            j = int(np.argmax(np.abs(np.diff(sp))))
            if np.abs(np.diff(sp)).max() > dv: dv = float(np.abs(np.diff(sp)).max()); dvw = (n, round(tt[j], 2), round(sp[j], 2), round(sp[j + 1], 2))
info['mechanism_speed_change_per_frame_max_mps'] = round(dv, 3); info['mechanism_worst'] = dvw
hs = [h for b in L['balls'].values() for h in b['hits'] if np.linalg.norm(h['vout']) > 0 and h['kind'] not in ('gate',)]
info['bounce_speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in hs), 4)
info['restitution_range'] = (round(min(h['e'] for h in hs), 3), round(max(h['e'] for h in hs), 3))
info['key_tilt_max_deg'] = round(max(math.degrees(math.acos(min(1, h['n'][2]))) for h in hs), 1)
# launches from the ramps: speed at the lip <= free-roll speed (no hidden push)
info['ramp_lip_speeds'] = [round(r['v_lip'], 3) for r in L['ev']['ramps']]
for k, v in info.items(): print(k, v)
