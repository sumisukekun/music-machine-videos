"""Checks for gymnopedie (no Blender needed): python3 verify.py [table]
 - every melody note (score_src.json top voice, cut as in arrange.py) is struck by the pearl at its exact time and pitch;
   every bass note by the night ball; every chord (middle voice + inner bass voice) by a hammer stroke at its exact time
 - contact distance between ball and key on the rendered frame nearest each hit, and at the exact time
 - collisions: each ball vs every key / bass bar of both balls (keys stand for the whole video), hammers, ball-ball gap
 - free flight: per-frame acceleration == gravity, frame-to-frame speed ratio; every bounce: e <= emax, |v_out| <= |v_in|
 - mechanisms (chutes, carousel horses, lift): max speed change per frame"""
import json, os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from helix import pen_box, key_axes
L = json.load(open(f'{HERE}/layout.json')); LEAD = json.load(open(f'{HERE}/lead.json')); E = json.load(open(f'{HERE}/events.json'))
S = json.load(open(f'{HERE}/score_src.json'))
G = L['G']; FPS = 30; FINAL = E['final']; NF = int(round(L['ev']['t_end'] * FPS))
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'
T0, BEAT, BAR = E['t0'], E['beat'], E['bar']
def orig(v): return v + 2 if v <= 2 else v + 10
VOF = {orig(v): v for v in range(1, 30)}
def tq(v, b): return T0 + (v - 1) * BAR + b * BEAT


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


info = {}
B = L['balls']
# 1) score -> expected strikes, straight from score_src.json
from collections import defaultdict
want_top = defaultdict(set); want_bass = defaultdict(set); want_ch = defaultdict(set)
for bar, b, d, p in S['top']:
    if bar in VOF: want_top[round(tq(VOF[bar], b), 5)].add(p)
for bar, b, d, p in S['bottom']:
    if bar in VOF:
        (want_bass if b == 0 else want_ch)[round(tq(VOF[bar], b), 5)].add(p)
for bar, b, d, p in S['middle']:
    if bar in VOF: want_ch[round(tq(VOF[bar], b), 5)].add(p)
got_top = defaultdict(set); got_bass = defaultdict(set); got_ch = defaultdict(set)
for h in B['pearl']['hits']:
    if h['pitch']: got_top[round(h['t'], 5)] |= set(h['pitch'])
for h in B['night']['hits']:
    if h['pitch']: got_bass[round(h['t'], 5)] |= set(h['pitch'])
for hm in L['ev']['hammers']: got_ch[round(hm['t'], 5)] |= set(hm['pitches'])
def cmp(w, g): return [(t, sorted(w.get(t, [])), sorted(g.get(t, []))) for t in sorted(set(w) | set(g)) if w.get(t) != g.get(t)]
info['melody_onsets'] = len(want_top); info['melody_mismatch'] = cmp(want_top, got_top)
info['bass_onsets'] = len(want_bass); info['bass_mismatch'] = cmp(want_bass, got_bass)
info['chord_onsets'] = len(want_ch); info['chord_mismatch'] = cmp(want_ch, got_ch)
# hammer flight = exact ballistic return after dt
info['hammer_dt'] = sorted(set(round(h['dt'], 3) for h in L['ev']['hammers']))
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    rows = []
    for t, ps in want_top.items(): rows.append((t, 'melody', ps, got_top.get(t, set()), 'pearl'))
    for t, ps in want_bass.items(): rows.append((t, 'bass', ps, got_bass.get(t, set()), 'night'))
    for t, ps in want_ch.items(): rows.append((t, 'chord', ps, got_ch.get(t, set()), 'hammer'))
    for t, k, w, g, who in sorted(rows):
        print(f'{t:7.3f}s frame {round(t * FPS):5d} {k:6s} score {"+".join(nm(p) for p in sorted(w)):18s} struck {"+".join(nm(p) for p in sorted(g)):18s} by {who}')
# 2) contact distance on the nearest frame
gaps = []
for n, b in B.items():
    for h in b['hits']:
        f = round(h['t'] * FPS); p = at(b, f / FPS)
        gaps.append((float(np.linalg.norm(p - np.array(h['p']))), n, h['kind'], round(h['t'], 3)))
gaps.sort(reverse=True)
info['contact_gap_nearest_frame_max_m'] = round(gaps[0][0], 3); info['contact_gap_mean_m'] = round(float(np.mean([g[0] for g in gaps])), 3)
info['contact_gap_worst'] = gaps[:3]
info['contact_gap_exact_time_max_m'] = round(max(float(np.linalg.norm(at(b, h['t']) - np.array(h['p']))) for b in B.values() for h in b['hits']), 6)
# 3) collisions: all keys stand for the whole video
R = {n: b['r'] for n, b in B.items()}
EXT = {n: L['ext'][n] for n in B}
keys = []
for n, b in B.items():
    for h in b['hits']:
        if h['kind'] in ('key', 'lever', 'chord', 'bass'):
            nv = np.array(h['n']); keys.append(dict(ball=n, t=h['t'], top=np.array(h['p']) - R[n] * nv, n=nv, ext=EXT[n], kind=h['kind']))
ts = np.arange(0.0, L['ev']['t_end'], 1 / 240)
P = {n: np.array([at(b, t) for t in ts]) for n, b in B.items()}
badc = {}; worst = 0.0
for K in keys:
    for n, Pn in P.items():
        m = np.linalg.norm(Pn - K['top'], axis=1) < 1.2
        if K['ball'] == n: m &= np.abs(ts - K['t']) > 0.03
        if not m.any(): continue
        pen = pen_box(Pn[m], K['top'], K['n'], K['ext'], R[n]); k = pen > 0.005
        if k.any():
            key = f"{n}->{K['ball']}:{K['kind']}"; badc.setdefault(key, []).append((round(float(ts[m][k][0]), 3), round(float(pen[k].max()), 3), 'key_t', round(K['t'], 3)))
            worst = max(worst, float(pen[k].max()))
info['collisions'] = {k: len(v) for k, v in badc.items()}; info['collision_examples'] = {k: v[:3] for k, v in badc.items()}
info['collision_worst_pen_m'] = round(worst, 3)
# hammers (plunger guide = vertical cylinder r 0.08 from bar top - 0.3 to + h + 0.5)
hmin = 9.0
for hm in L['ev']['hammers']:
    hb = B['night']['hits'][[i for i, h in enumerate(B['night']['hits']) if h.get('ev') == hm['ib']][0]]
    top = np.array(hb['p']) - R['night'] * np.array(hb['n']); th = math.atan2(top[1], top[0]); rr = math.hypot(top[0], top[1]) - 0.62
    c = np.array([rr * math.cos(th), rr * math.sin(th)])
    hmax = G * hm['dt'] ** 2 / 8
    for n, Pn in P.items():
        m = (Pn[:, 2] > top[2] - 0.5) & (Pn[:, 2] < top[2] + hmax + 0.8)
        if m.any(): hmin = min(hmin, float((np.linalg.norm(Pn[m, :2] - c, axis=1) - 0.09 - R[n]).min()))
info['hammer_guide_min_gap_m'] = round(hmin, 3)
d = np.linalg.norm(P['pearl'] - P['night'], axis=1) - R['pearl'] - R['night']
mm = ts < FINAL - 0.05
j = int(np.argmin(d[mm])); info['min_ball_surface_gap'] = (round(float(d[mm][j]), 3), round(float(ts[mm][j]), 2))
# core: balls stay outside the glass core (r 1.6, top rim r 1.65 at z 0.05) except in the lift
for n, Pn in P.items():
    t_out = [r_['t_lip'] for r_ in L['ev']['ramps'] if r_['ball'] == n][0]
    m = (ts > t_out) & (ts < FINAL - 1.2) & (P[n][:, 2] > L['ev']['lift']['z_end'] + 2.65)   # below that: open arcade
    r = np.hypot(Pn[m, 0], Pn[m, 1]); z = Pn[m, 2]
    below = z < 0.05
    gap_wall = np.where(below, r - 1.6, np.hypot(np.maximum(0, r - 1.65), z - 0.05)) - R[n]
    info[f'{n}_core_min_gap_m'] = round(float(gap_wall.min()), 3)
# 4) free flight per frame
tf = np.arange(NF) / FPS
amax = 0.0; rat = 1.0; worst_r = None
for n, b in B.items():
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
info['flight_accel_error_max'] = round(amax, 6); info['flight_speed_ratio_max(v>0.5)'] = round(rat, 3); info['worst_ratio_at'] = worst_r
dv = 0.0; dvw = None
for n, b in B.items():
    for g in b['seg']:
        if g['k'] != 'path' or g['t1'] - g['t0'] < 0.2: continue
        tt = np.arange(g['t0'], g['t1'] + 1e-9, 1 / FPS)
        Q = np.array([seg_at(g, t) for t in tt]); sp = np.linalg.norm(np.diff(Q, axis=0), axis=1) * FPS
        if len(sp) > 2 and np.abs(np.diff(sp)).max() > dv:
            j = int(np.argmax(np.abs(np.diff(sp)))); dv = float(np.abs(np.diff(sp)).max()); dvw = (n, round(tt[j], 2), round(sp[j], 2), round(sp[j + 1], 2))
info['mechanism_speed_change_per_frame_max_mps'] = round(dv, 3); info['mechanism_worst'] = dvw
hs = [h for b in B.values() for h in b['hits'] if np.linalg.norm(h['vout']) > 0]
info['bounce_speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in hs), 4)
info['restitution_range'] = (round(min(h['e'] for h in hs), 3), round(max(h['e'] for h in hs), 3))
info['key_tilt_max_deg'] = round(max(math.degrees(math.acos(min(1, h['n'][2]))) for h in hs), 1)
info['chute_lip_speeds'] = [round(r['v_lip'], 3) for r in L['ev']['ramps']] + [round(L['ev']['endchute']['v_lip'], 3)]
info['duration_s'] = round(L['ev']['t_end'], 2)
for k, v in info.items(): print(k, v)
