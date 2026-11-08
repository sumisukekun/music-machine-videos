"""Checks for radetzky (no Blender needed): python3 verify.py [table]
 - every melody note is struck at its exact time by a player marble on a key of that pitch; pitch/time table vs the score
 - contact distance between marble and key on the rendered frame nearest each hit
 - collisions: marble surface entering any key that is out (visible) at that moment, other than its own hit
 - free flight: per-frame acceleration == gravity, frame-to-frame speed ratio; bounces never speed up
 - rolling (pipe, rails, funnel): no speed jumps between frames
 - claps: every audience landing is on a beat; the conductor lands on every beat"""
import json, os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(f'{HERE}/layout.json')); LEAD = json.load(open(f'{HERE}/lead.json')); E = json.load(open(f'{HERE}/events.json'))
G = L['G']; FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.5) * FPS)
NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
KEY_W, KEY_D, KEY_T = 0.2, 0.46, 0.065


def at(seg, t):
    for g in seg:
        if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9:
            if g['k'] == 'rest': return np.array(g['p'])
            if g['k'] == 'fly':
                u = t - g['t0']; return np.array(g['p']) + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
            P = g['P']; x = (t - g['t0']) / g['dt']; i = int(min(len(P) - 2, max(0, math.floor(x)))); f = x - i
            return np.array(P[i]) * (1 - f) + np.array(P[i + 1]) * f
    return np.array(seg[0].get('p', seg[0].get('P', [[0, 0, 0]])[0])) if t < seg[0]['t0'] else at(seg, seg[-1]['t1'])


info = {}
B = L['balls']; TW = {k: np.array(v) for k, v in L['geo']['tower'].items()}
keys = []
for n in ('G1', 'G2'):
    R = B[n]['r']
    for h in B[n]['hits']:
        if h['kind'] != 'key': continue
        nrm = np.array(h['n']); nrm /= np.linalg.norm(nrm); ct = np.array(h['p']) - R * nrm
        rad = np.array([ct[0] - TW[h['tower']][0], ct[1] - TW[h['tower']][1], 0]); rad /= np.linalg.norm(rad)
        ya = rad - nrm * (rad @ nrm); ya /= np.linalg.norm(ya); xa = np.cross(ya, nrm)
        keys.append(dict(ball=n, i=h['i'], t=h['t'], c=ct - nrm * KEY_T / 2, M=np.stack([xa, ya, nrm], 1), contact=ct, n=nrm, R=R, w=h['kw'], win=h['win']))
# 1) notes
hit_by = {}
for k in keys: hit_by.setdefault(k['i'], []).append(k)
info['notes'] = len(LEAD); info['notes_without_key'] = [i for i in range(len(LEAD)) if i not in hit_by]
info['notes_with_two_keys'] = [i for i, v in hit_by.items() if len(v) > 1]
info['hit_time_error_max_s'] = max(abs(k['t'] - LEAD[k['i']][0]) for k in keys)
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    for i, l in enumerate(LEAD):
        k = hit_by[i][0]
        print(f"{i:3d} {l[0]:7.3f}s {l[4]:5s} bar{l[5]:3d} {NAMES[l[1] % 12]}{l[1] // 12 - 1:<2d} frame {round(l[0] * FPS):5d}  {k['ball']} key@({k['contact'][0]:.2f},{k['contact'][1]:.2f},{k['contact'][2]:.2f})")
# 2) contact distance at the nearest frame
gaps = [np.linalg.norm(at(B[k['ball']]['seg'], round(k['t'] * FPS) / FPS) - (k['contact'] + k['R'] * k['n'])) for k in keys]
info['key_contact_gap_nearest_frame_max'] = round(float(max(gaps)), 3); info['key_contact_gap_mean'] = round(float(np.mean(gaps)), 3)
# 3) collisions with keys while they are out (240 Hz)
ts = np.arange(0.0, FINAL + 3.0, 1 / 240)
PP = {n: np.array([at(B[n]['seg'], t) for t in ts]) for n in ('G1', 'G2', 'late', 'cond')}
bad = []; worst = 0
for k in keys:
    m_t = (ts > k['t'] - k['win'][0]) & (ts < k['t'] + k['win'][3])
    for n, Pn in PP.items():
        R = B[n]['r']; m = m_t & (np.abs(Pn[:, 0] - k['c'][0]) < 0.9) & (np.abs(Pn[:, 1] - k['c'][1]) < 0.9) & (np.abs(Pn[:, 2] - k['c'][2]) < 0.9)
        if n == k['ball']: m &= np.abs(ts - k['t']) > 0.02
        if not m.any(): continue
        rel = (Pn[m] - k['c']) @ k['M']
        d = np.stack([np.maximum(0, np.abs(rel[:, 0]) - k['w'] / 2), np.maximum(0, np.abs(rel[:, 1]) - KEY_D / 2), np.maximum(0, np.abs(rel[:, 2]) - KEY_T / 2)], 1)
        pen = R - np.linalg.norm(d, axis=1)
        if (pen > 0.005).any(): bad.append((n, k['ball'], k['i'], round(float(ts[m][pen > 0.005][0]), 3), round(float(pen.max()), 3))); worst = max(worst, float(pen.max()))
info['key_collisions'] = len(bad); info['key_collision_examples'] = bad[:6]; info['key_collision_worst'] = round(worst, 3)
# player vs player, player vs conductor
dmin = min(np.linalg.norm(PP['G1'] - PP['G2'], axis=1).min(), np.linalg.norm(PP['G1'] - PP['cond'], axis=1).min(), np.linalg.norm(PP['G2'] - PP['cond'], axis=1).min())
info['min_distance_between_players_and_conductor'] = round(float(dmin), 3)
# 4) free flight
tf = np.arange(NF) / FPS
amax = 0.0; rat = 1.0; rat2 = 1.0
for n, b in B.items():
    for g in b['seg']:
        if g['k'] != 'fly': continue
        fs = [f for f in range(NF) if g['t0'] + 1e-6 < tf[f] - 1 / FPS and tf[f] + 1 / FPS < g['t1'] - 1e-6]
        for f in fs:
            Q = [at([g], tf[f + d]) for d in (-1, 0, 1)]
            a = (Q[2] - 2 * Q[1] + Q[0]) * FPS * FPS; amax = max(amax, float(np.linalg.norm(a - np.array([0, 0, -G]))))
            s0 = np.linalg.norm(Q[1] - Q[0]) * FPS; s1 = np.linalg.norm(Q[2] - Q[1]) * FPS
            if min(s0, s1) > 0.5: rat = max(rat, max(s0, s1) / min(s0, s1))
            if min(s0, s1) > 2.0: rat2 = max(rat2, max(s0, s1) / min(s0, s1))
info['flight_accel_error_max'] = round(amax, 6); info['flight_speed_ratio_max(v>0.5)'] = round(rat, 3); info['flight_speed_ratio_max(v>2)'] = round(rat2, 3)
hs = [h for b in B.values() for h in b['hits'] if np.linalg.norm(h['vout']) > 0]
info['bounce_speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in hs), 4)
info['restitution_range'] = (round(min(h['e'] for h in hs), 3), round(max(h['e'] for h in hs), 3))
info['key_tilt_max_deg'] = round(max(math.degrees(math.acos(min(1, k['n'][2]))) for k in keys), 1)
# 5) rolling / lifts: speed change per frame (acceleration) inside sampled segments
acc = {}
for n, b in B.items():
    for g in b['seg']:
        if g['k'] != 'samp': continue
        P = np.array(g['P']); dt = g['dt']; step = int(round(1 / FPS / dt))
        Pf = P[::step]
        if len(Pf) < 4: continue
        v = np.linalg.norm(np.diff(Pf, axis=0), axis=1) * FPS
        a = np.abs(np.diff(v)) * FPS
        acc[n] = max(acc.get(n, 0), float(a.max()))
info['sampled_motion_max_speed_change_per_s'] = {k: round(v, 2) for k, v in acc.items()}
# speed continuity at segment joins (late marble): speed before/after each join
late = B['late']['seg']; joins = []
for g0, g1 in zip(late, late[1:]):
    t = g1['t0']; v0 = np.linalg.norm(at([g0], t) - at([g0], t - 1 / 120)) * 120; v1 = np.linalg.norm(at([g1], t + 1 / 120) - at([g1], t)) * 120
    joins.append((round(t, 2), g0['k'], g1['k'], round(v0, 2), round(v1, 2)))
info['late_segment_joins(t,from,to,v_before,v_after)'] = joins
# 6) claps & conductor on the beat
beats = np.array([b[0] for b in E['beats']] + [FINAL])
land = [tl for name, hops in L['aud'].items() for tl, tau in hops if tl <= FINAL + 1e-6]
info['audience_landings_before_end'] = len(land); info['audience_landing_off_beat_max_s'] = round(float(max(np.abs(beats[np.argmin(np.abs(beats[None, :] - np.array(land)[:, None]), axis=1)] - land))), 6)
ch = [b for b, tau in L['ev']['cond_hops'] if b <= FINAL + 1e-6]
info['conductor_landings'] = len(ch); info['conductor_off_beat_max_s'] = round(float(max(np.min(np.abs(beats[None, :] - np.array(ch)[:, None]), axis=1))), 6)
info['late_lands_in_seat_at'] = [h['t'] for h in B['late']['hits'] if h['kind'] == 'seat']; info['final_chord_at'] = FINAL
stab = np.array(E['stab'])
steps = [h['t'] for h in B['late']['hits'] if h['kind'] in ('step', 'carpet')]
info['late_stair_hits_on_offbeats_max_err'] = round(float(max(np.min(np.abs(stab[None, :] - np.array(steps)[:, None]), axis=1))), 6)
for k, v in info.items(): print(k, v)
