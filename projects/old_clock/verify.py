"""Checks for old_clock (no Blender needed): python3 verify.py [table]
 1 every melody note is struck at its exact time by a marble landing on a bar of that pitch (table: pitch/time/frame)
 2 notes.json melody == the score in arrange.py (independent re-parse of the bar strings), bass = chord roots
 3 contact distance ball <-> bar on the rendered frame nearest each hit
 4 path continuity (no jumps between segments), free flight: per-frame acceleration == gravity,
   frame-to-frame speed ratio in flight, every bounce |v_out| <= |v_in| and e <= 0.96, bar tilt
 5 collisions: a ball entering any bar other than its own at its own hit, ball-ball overlap, cups/rails"""
import json, os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(f'{HERE}/layout.json')); LEAD = json.load(open(f'{HERE}/lead.json')); E = json.load(open(f'{HERE}/events.json'))
NOTES = json.load(open(f'{HERE}/notes.json'))
G = L['G']; FPS = 30; DUR = E['dur']; NF = int(DUR * FPS)
NAMES = 'C C# D D# E F F# G G# A A# B'.split()
def nm(p): return f'{NAMES[p % 12]}{p // 12 - 1}'


def seg_at(g, t):
    if g['k'] == 'pend':
        th = g['sgn'] * g['amp'] * math.sin(g['w'] * (t - g['t0'])); pv = np.array(g['piv'])
        return pv + g['L'] * np.array([math.sin(th), 0, -math.cos(th)])
    p = np.array(g['p'])
    if g['k'] == 'rest': return p
    u = t - g['t0']
    if g['k'] == 'fly': return p + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
    return p + np.array(g['v']) * u + 0.5 * np.array(g['a']) * u * u


def at(b, t):
    S = b['seg']
    if t <= S[0]['t0']: return seg_at(S[0], S[0]['t0'])
    for g in S:
        if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9: return seg_at(g, t)
    return seg_at(S[-1], min(t, S[-1]['t1']))


info = {}
# 1 coverage
hits = [(n, h) for n, b in L['balls'].items() for h in b['hits']]
by_i = {}
for n, h in hits: by_i.setdefault(h['i'], []).append((n, h))
info['melody_notes'] = len(LEAD)
info['notes_without_hit'] = [i for i in range(len(LEAD)) if i not in by_i]
info['hit_time_err_max_s'] = max(abs(h['t'] - LEAD[h['i']][0]) for n, h in hits)
info['hit_pitch_mismatch'] = [h['i'] for n, h in hits if h['pitch'] != LEAD[h['i']][1]]
keys_by = {(K['ball'], K['i']): K for K in L['keys']}
info['hits_without_bar'] = [(n, h['i']) for n, h in hits if (n, h['i']) not in keys_by]
info['bars_never_hit'] = [(K['ball'], K['i']) for K in L['keys'] if not any(n == K['ball'] and h['i'] == K['i'] for n, h in hits)]
# 2 score re-parse
src = open(f'{HERE}/arrange.py').read()
ns = {"__file__": f"{HERE}/arrange.py"}
exec(src.split("lead = [[T_PICK")[0], ns)
exp = [(round(ns['T_PICK'], 4), 74)]
for b, (sec, bar) in enumerate(ns['BARS']):
    pos = 0
    for p, d in ns['parse'](bar):
        if p is not None: exp.append((round(ns['TB1'] + b * ns['BAR'] + pos * ns['E8'], 4), p))
        pos += d
got = [(round(n[0], 4), n[3]) for n in NOTES if n[2] == 'mel']
info['melody_equals_score'] = exp == got
bass = [(n[0], n[3]) for n in NOTES if n[2] == 'bass']
info['bass_notes'] = len(bass); info['bass_pitches'] = sorted(set(nm(p) for t, p in bass))
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    for i, l in enumerate(LEAD):
        who = ','.join(f'{n}:{h["kind"]}' for n, h in by_i.get(i, []))
        print(f'{i:3d} {l[0]:7.3f}s frame {round(l[0] * FPS):5d} bar {l[5]:2d} {nm(l[1]):4s} {who}')
# 3 contact distance on the nearest frame
gaps = []
for n, h in hits:
    f = round(h['t'] * FPS); gaps.append(np.linalg.norm(at(L['balls'][n], f / FPS) - np.array(h['p'])))
info['contact_gap_nearest_frame_max_m'] = round(float(max(gaps)), 3); info['contact_gap_mean_m'] = round(float(np.mean(gaps)), 3)
# 4 continuity + flight
jumps = []
for n, b in L['balls'].items():
    for g0, g1 in zip(b['seg'], b['seg'][1:]):
        if abs(g0['t1'] - g1['t0']) > 1e-6: jumps.append((n, 'time', g0['t1'], g1['t0']))
        d = np.linalg.norm(seg_at(g0, g0['t1']) - seg_at(g1, g1['t0']))
        if d > 1e-4: jumps.append((n, round(g0['t1'], 3), round(float(d), 4)))
info['path_jumps'] = jumps
tf = np.arange(NF) / FPS
evt = sorted(set([h['t'] for n, h in hits] + [g['t0'] for b in L['balls'].values() for g in b['seg']]))
evt = np.array(evt)
amax = 0.0; rat = 1.0
for n, b in L['balls'].items():
    Q = np.array([at(b, t) for t in tf])
    for f in range(1, NF - 1):
        t0, t1 = tf[f - 1], tf[f + 1]
        k = np.searchsorted(evt, t0 - 1e-6)
        if k < len(evt) and evt[k] <= t1 + 1e-6: continue
        seg = [g for g in b['seg'] if g['t0'] - 1e-9 <= tf[f] <= g['t1'] + 1e-9]
        if not seg or seg[0]['k'] != 'fly': continue
        a = (Q[f + 1] - 2 * Q[f] + Q[f - 1]) * FPS * FPS
        amax = max(amax, float(np.linalg.norm(a - np.array([0, 0, -G]))))
        s0 = np.linalg.norm(Q[f] - Q[f - 1]) * FPS; s1 = np.linalg.norm(Q[f + 1] - Q[f]) * FPS
        if min(s0, s1) > 0.5: rat = max(rat, max(s0, s1) / min(s0, s1))
info['flight_accel_error_max'] = round(amax, 6); info['flight_frame_speed_ratio_max(v>0.5)'] = round(rat, 3)
bn = [h for n, h in hits if np.linalg.norm(h['vout']) > 0]
info['bounce_speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in bn), 3)
info['restitution_range'] = (round(min(h['e'] for h in bn), 2), round(max(h['e'] for h in bn), 3))
info['bar_tilt_max_deg'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for n, h in hits), 1)
info['landings_on_levers_e0'] = sum(1 for n, h in hits if h['e'] == 0.0)
# 5 collisions (240 Hz)
ts = np.arange(0.0, DUR, 1 / 240)
P = {n: np.array([at(b, t) for t in ts]) for n, b in L['balls'].items()}
Rb = {n: b['r'] for n, b in L['balls'].items()}
worst = 0.0; bad = {}
LEVER_I = {h['i'] for n, h in hits if h['kind'] in ('lever', 'seesaw')}
for K in L['keys']:
    c = np.array(K['c']); M = np.array(K['M']); h = np.array([K['hw'], K['hd'], K['th']])
    for n, Pn in P.items():
        m = (np.abs(Pn[:, 0] - c[0]) < 0.8) & (np.abs(Pn[:, 2] - c[2]) < 0.8)
        if K['ball'] == n: m &= np.abs(ts - K['t']) > 0.035
        if K['i'] in LEVER_I: m &= ts < K['t'] - 0.035       # lever bars swing away under the ball after their hit
        if not m.any(): continue
        rel = (Pn[m] - c) @ M; d = np.linalg.norm(np.maximum(0, np.abs(rel) - h), axis=1)
        pen = Rb[n] - d
        if (pen > 0.01).any():
            bad.setdefault(f'{n}->bar({K["ball"]}:{K["i"]})', round(float(pen.max()), 3)); worst = max(worst, float(pen.max()))
info['bar_collisions'] = bad; info['bar_collision_worst_pen_m'] = round(worst, 3)
bb = {}
names = list(P)
for a in range(len(names)):
    for b_ in range(a + 1, len(names)):
        d = np.linalg.norm(P[names[a]] - P[names[b_]], axis=1) - Rb[names[a]] - Rb[names[b_]]
        bb[f'{names[a]}-{names[b_]}'] = round(float(d.min()), 3)
info['ball_ball_min_clearance_m'] = bb
for k, v in info.items(): print(k, v)
