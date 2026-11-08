"""Checks for campanella (no Blender needed): python3 verify.py [table]
 1 every note of notes.json is struck exactly once at its exact time by a ball, on a bell of that pitch; pitch table
 2 notes.json == score_excerpt.json (pitches, rhythm) after the bar-aligned joins
 3 contact distance between ball and bell crown on the rendered frame nearest each hit
 4 collisions: ball surface entering any bell other than its own at its own hit; ball-ball overlap; lane rails
 5 free flight: per-frame acceleration == gravity, frame-to-frame speed ratio; path continuity
 6 every bounce: restitution e <= 0.97 and |v_out| <= |v_in|; tilt of the bell"""
import json, os, sys, math
import numpy as np
from motion import L, PATHS, G, R, bell_r, lift_z
HERE = os.path.dirname(os.path.abspath(__file__))
NOTES = json.load(open(f'{HERE}/notes.json')); E = json.load(open(f'{HERE}/events.json')); SRC = json.load(open(f'{HERE}/score_excerpt.json'))
FPS = 30; NF = int(E['dur'] * FPS)
NAMES = 'C C# D D# E F F# G G# A A# B'.split()
def nm(p): return f'{NAMES[p % 12]}{p // 12 - 1}'
info = {}

# 1 coverage
hits = [(n, h) for n, b in L['balls'].items() for h in b['hits']]
by_i = {}
for n, h in hits: by_i.setdefault(h['i'], []).append((n, h))
info['notes'] = len(NOTES); info['hits'] = len(hits)
info['notes_without_hit'] = [i for i in range(len(NOTES)) if i not in by_i]
info['notes_hit_twice'] = [i for i, v in by_i.items() if len(v) > 1]
info['hit_time_err_max_s'] = max(abs(h['t'] - NOTES[h['i']][0]) for n, h in hits)
# 2 notes vs source score (independent re-derivation of the mapping)
def mid(p): return NAMES.index(p[:-1]) + 12 * (int(p[-1]) + 1)
exp = []
for e in SRC['events']:
    b = e['beat']; m = b if b < 35.5 else (b - 75.0 if b < 148 else b - 354.562)
    for p in e['pitch']: exp.append((round(E['t0'] + m * E['beat'], 4), mid(p)))
got = sorted((round(n[0], 4), n[1]) for n in NOTES)
info['notes_equal_score'] = sorted(exp) == got
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    for i, n in enumerate(NOTES):
        who = ','.join(f'{b}:{h["kind"]}' for b, h in by_i.get(i, []))
        print(f'{i:3d} {n[0]:7.3f}s frame {round(n[0] * FPS):5d} {n[6]} src-beat {n[5]:8.3f} {n[4]} {nm(n[1]):4s} {who}')
# bells: one per hit (kind bell/cradle), giant bell for the hero's last
BELLS = []
for n, h in hits:
    nr = np.array(h['n'], float); nr /= np.linalg.norm(nr)
    if h['kind'] == 'giant': continue
    p = NOTES[h['i']][1]; rb = bell_r(p)
    c = np.array(h['p']) - R * nr          # crown contact point
    ax = nr if h['kind'] != 'cradle' else np.array([0, 0, 1.0])
    BELLS.append(dict(ball=n, t=h['t'], i=h['i'], c=c, ax=ax, rb=rb))
# 3 contact distance on the nearest frame
gaps = []
for n, h in hits:
    f = round(h['t'] * FPS); gaps.append(np.linalg.norm(PATHS[n].at(f / FPS) - np.array(h['p'])))
info['contact_gap_nearest_frame_max'] = round(float(max(gaps)), 3); info['contact_gap_mean'] = round(float(np.mean(gaps)), 3)
# 4 collisions with bells (bell = cone from crown down along -axis: radius 0.45 rb at the crown to rb at the mouth, height 1.5 rb)
ts = np.arange(0.0, E['dur'], 1 / 240)
POS = {n: np.array([p.at(t) for t in ts]) for n, p in PATHS.items()}
bad = {}; worst = 0.0
for B in BELLS:
    hgt = 1.5 * B['rb']
    for n, P in POS.items():
        d = P - B['c']
        m = np.abs(d).max(1) < 0.75
        if B['ball'] == n: m &= np.abs(ts - B['t']) > 0.035
        if not m.any(): continue
        dd = d[m]; s = -(dd @ B['ax'])                      # depth along the bell below the crown
        sc = np.clip(s, 0, hgt); rad = np.linalg.norm(dd + np.outer(s, B['ax']), axis=1)
        rloc = B['rb'] * (0.45 + 0.55 * sc / hgt)
        # distance from ball centre to the cone body
        out = np.where((s >= 0) & (s <= hgt), rad - rloc, np.hypot(np.maximum(0, rad - rloc), np.where(s < 0, -s, s - hgt)))
        pen = R - out
        k = pen > 0.012
        if k.any():
            key = f'{n}->bell_of_{B["ball"]}'; bad.setdefault(key, []).append((B['i'], round(float(ts[m][k][0]), 3), round(float(pen[k].max()), 3)))
            worst = max(worst, float(pen[k].max()))
info['bell_collisions'] = {k: len(v) for k, v in bad.items()}
info['bell_collision_examples'] = {k: v[:3] for k, v in list(bad.items())[:12]}
info['bell_collision_worst_pen'] = round(worst, 3)
# ball-ball
names = list(POS); mind = 9; pair = None
for a in range(len(names)):
    for b in range(a + 1, len(names)):
        dd = np.linalg.norm(POS[names[a]] - POS[names[b]], axis=1).min()
        if dd < mind: mind = dd; pair = (names[a], names[b])
info['min_ball_ball_distance'] = (round(float(mind), 3), pair)
# 5 free flight & continuity
tf = np.arange(NF) / FPS; amax = 0.0; rat = 1.0; rat2 = 1.0; jump = 0.0
for n, b in L['balls'].items():
    segs = b['seg']
    for g0, g1 in zip(segs, segs[1:]):
        from motion import seg_at
        jump = max(jump, float(np.linalg.norm(seg_at(g0, g0['t1']) - seg_at(g1, g1['t0']))), abs(g0['t1'] - g1['t0']))
    for g in segs:
        if g['k'] != 'fly': continue
        fs = [f for f in range(NF) if g['t0'] + 1e-6 < tf[f] - 1 / FPS and tf[f] + 1 / FPS < g['t1'] - 1e-6]
        for f in fs:
            Q = [PATHS[n].at(tf[f] + k / FPS) for k in (-1, 0, 1)]
            a = (Q[2] - 2 * Q[1] + Q[0]) * FPS * FPS; amax = max(amax, float(np.linalg.norm(a - np.array([0, 0, -G]))))
            s0 = np.linalg.norm(Q[1] - Q[0]) * FPS; s1 = np.linalg.norm(Q[2] - Q[1]) * FPS
            if min(s0, s1) > 0.5: rat = max(rat, max(s0, s1) / min(s0, s1))
            if min(s0, s1) > 2.0: rat2 = max(rat2, max(s0, s1) / min(s0, s1))
info['flight_accel_error_max'] = round(amax, 6); info['flight_speed_ratio_max(v>0.5)'] = round(rat, 3); info['flight_speed_ratio_max(v>2)'] = round(rat2, 3)
info['path_discontinuity_max'] = round(jump, 6)
# 6 bounces
hs = [h for n, h in hits if np.linalg.norm(h['vout']) > 0]
info['bounce_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in hs), 3)
info['restitution_range'] = (round(min(h['e'] for h in hs), 3), round(max(h['e'] for h in hs), 3))
info['bell_tilt_max_deg'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for n, h in hits if h['kind'] == 'bell'), 1)
info['distinct_bells'] = len(BELLS) + 1
for k, v in info.items(): print(k, v)
