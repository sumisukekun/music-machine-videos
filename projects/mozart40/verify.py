"""Checks for mozart40 (no Blender needed; run scene.py -- pieces first for pieces.json): python3 verify.py [table]
 - every violin I / flute / oboe event (A, B), viola event (train) and cello event (street ball) is struck at its exact time
   by its ball with the score pitch(es); no pitched hit outside the score
 - contact distance between ball and key on the rendered frame nearest each hit, and at the exact time
 - collisions: ball surface entering any key / tile / bar / bell other than its own at its own hit (sampled at 240 Hz)
 - free flight: per-frame acceleration == gravity, frame-to-frame speed ratio; every bounce: e <= emax, |v_out| <= |v_in|
 - rolling (roofs, tracks, rail): frame-to-frame speed change; minimum distance between the balls"""
import json, os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(f'{HERE}/layout.json')); E = json.load(open(f'{HERE}/events.json'))
MEL = json.load(open(f'{HERE}/mel.json')); ACC = json.load(open(f'{HERE}/acc.json')); BASS = json.load(open(f'{HERE}/bass.json'))
PC = json.load(open(f'{HERE}/pieces.json'))
G = L['G']; FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.3) * FPS)
NM = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
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


B = L['balls']; info = {}
def struck(names, t):
    return sorted(h['pitch'] for n in names for h in B[n]['hits'] if abs(h['t'] - t) < 1e-6 and h['pitch'] is not None)
bad = []; table = []
for t, p, v, d, who, m, b, dbl, src in MEL:
    names = ['A', 'B'] if who == 'AB' else [who]
    got = struck(names, t)
    want = [p] * len(names)
    if got != sorted(want): bad.append(('MEL', who, round(t, 3), nm(p), [nm(x) for x in got]))
    table.append((f'{src:3s}{who:2s}', t, m, b, want, got))
# A and B together at the same time: the union must equal the score's set
for t, ps, v, d, m, b in ACC:
    got = struck(['train'], t)
    if got != sorted(ps): bad.append(('ACC', round(t, 3), [nm(x) for x in ps], [nm(x) for x in got]))
    table.append(('Vla  ', t, m, b, ps, got))
for t, p, v, d, m, b, cb in BASS:
    got = struck(['bass'], t)
    if got != [p]: bad.append(('BASS', round(t, 3), nm(p), [nm(x) for x in got]))
    table.append(('Vc   ', t, m, b, [p], got))
info['events'] = dict(mel=len(MEL), acc=len(ACC), bass=len(BASS)); info['events_wrong_or_missing'] = bad
sched = set([round(r[0], 5) for r in MEL] + [round(r[0], 5) for r in ACC] + [round(r[0], 5) for r in BASS])
extra = [(n, round(h['t'], 3), h['kind']) for n, b in B.items() for h in b['hits'] if h['pitch'] is not None and round(h['t'], 5) not in sched]
info['pitched_hits_not_in_score'] = extra
# every voice-time pair: each ball only hits what its voice plays
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    for v_, t, m, b, pp, got in sorted(table, key=lambda r: (r[1], r[0])):
        print(f'{v_} {t:7.3f}s bar{m:3d} beat{b + 1:<5g} score {"+".join(nm(p) for p in pp):8s} struck {"+".join(nm(p) for p in got):8s} frame {t * FPS:7.2f}')
# 2) contact distance at the nearest frame and at the exact time
gaps = []; ex = 0.0
for n, b in B.items():
    for h in b['hits']:
        f = round(h['t'] * FPS); p = at(b, f / FPS)
        gaps.append((round(float(np.linalg.norm(p - np.array(h['p']))), 3), n, h['kind'], round(h['t'], 3)))
        ex = max(ex, float(np.linalg.norm(at(b, h['t']) - np.array(h['p']))))
gaps.sort(reverse=True)
info['contact_gap_nearest_frame_max_m'] = gaps[0][0]; info['contact_gap_mean_m'] = round(float(np.mean([g[0] for g in gaps])), 3)
info['contact_gap_worst'] = gaps[:3]; info['contact_gap_exact_time_max_m'] = round(ex, 6)
# 3) collisions with every piece (all pieces stay for good)
R = L['R']
ts = np.arange(0.0, FINAL + 1.5, 1 / 240)
P = {n: np.array([at(b, t) for t in ts]) for n, b in B.items()}
badc = {}; worst = 0.0
ROLL = ('rooftile', 'xbar', 'tbar', 'rbar')
for Bx in PC['pieces']:
    M = frame(Bx['n']); c = np.array(Bx['top']) - M[:, 2] * Bx['th']
    for n, Pn in P.items():
        r = R[n]
        m = (np.abs(Pn[:, 0] - c[0]) < 1.0) & (np.abs(Pn[:, 2] - c[2]) < 1.0) & (np.abs(Pn[:, 1] - c[1]) < 1.0)
        own = Bx['ball'] == n
        if Bx['kind'] == 'bowl' and n in ('A', 'B'): own = True
        if own and Bx['kind'] not in ROLL: m &= np.abs(ts - Bx['t']) > 0.03
        if own and Bx['kind'] == 'bowl': m &= ts < Bx['t'] - 0.03
        if not m.any(): continue
        rel = (Pn[m] - c[None, :]) @ M
        d = np.stack([np.maximum(0, np.abs(rel[:, 0]) - Bx['hw']), np.maximum(0, np.abs(rel[:, 1]) - Bx['hd']), np.maximum(0, np.abs(rel[:, 2]) - Bx['th'])], 1)
        pen = r - np.linalg.norm(d, axis=1)
        if own and Bx['kind'] in ROLL: pen = pen - 0.012            # rolling over the bars of its own track = expected contact
        k = pen > 0.01
        if k.any():
            key = f"{n}->{Bx['ball']}:{Bx['kind']}"; badc.setdefault(key, []).append((round(float(ts[m][k][0]), 3), round(float(pen[k].max()), 3), 'piece_t', round(Bx['t'], 3)))
            worst = max(worst, float(pen[k].max()))
info['collisions'] = {k: len(v) for k, v in badc.items()}; info['collision_examples'] = {k: v[:3] for k, v in badc.items()}
info['collision_worst_pen_m'] = round(worst, 3)
names = list(P); mind = (9, None)
for a in range(len(names)):
    for c_ in range(a + 1, len(names)):
        d = np.linalg.norm(P[names[a]] - P[names[c_]], axis=1) - R[names[a]] - R[names[c_]]
        j = int(np.argmin(d))
        if d[j] < mind[0]: mind = (round(float(d[j]), 3), names[a], names[c_], round(float(ts[j]), 2))
info['min_ball_surface_gap'] = mind
# 4) free flight
tf = np.arange(NF) / FPS
amax = 0.0; rat = 1.0; worst_r = None
for n, b in B.items():
    for g in b['seg']:
        if g['k'] != 'fly' or g['t1'] - g['t0'] <= 3 / FPS: continue
        for f in range(1, NF - 1):
            if not (g['t0'] + 1e-6 < tf[f - 1] and tf[f + 1] < g['t1'] - 1e-6): continue
            Qp = [seg_at(g, tf[f + k]) for k in (-1, 0, 1)]
            a = (Qp[2] - 2 * Qp[1] + Qp[0]) * FPS * FPS
            amax = max(amax, float(np.linalg.norm(a - np.array([0, 0, -G]))))
            s0 = np.linalg.norm(Qp[1] - Qp[0]) * FPS; s1 = np.linalg.norm(Qp[2] - Qp[1]) * FPS
            if min(s0, s1) > 0.5:
                r_ = max(s0, s1) / min(s0, s1)
                if r_ > rat: rat = r_; worst_r = (n, round(tf[f], 2), round(s0, 2), round(s1, 2))
info['flight_accel_error_max'] = round(amax, 6); info['flight_speed_ratio_max(v>0.5)'] = round(rat, 3); info['worst_ratio_at'] = worst_r
dv = 0.0; dvw = None
for n, b in B.items():
    for g in b['seg']:
        if g['k'] != 'path' or g['t1'] - g['t0'] < 0.3: continue
        tt = np.arange(g['t0'] + 1 / FPS, g['t1'] - 1 / FPS, 1 / FPS)
        Qp = np.array([seg_at(g, t) for t in tt]); sp = np.linalg.norm(np.diff(Qp, axis=0), axis=1) * FPS
        if len(sp) > 2 and float(np.abs(np.diff(sp)).max()) > dv: dv = float(np.abs(np.diff(sp)).max()); dvw = (n, round(g['t0'], 2))
info['rolling_speed_change_per_frame_max_mps'] = (round(dv, 3), dvw)
hs = [h for b in B.values() for h in b['hits'] if np.linalg.norm(h['vout']) > 0]
info['bounce_speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in hs), 4)
bh = [h for h in hs if h['kind'] == 'key']
info['restitution_range'] = (round(min(h['e'] for h in bh), 3), round(max(h['e'] for h in bh), 3))
info['key_tilt_max_deg'] = round(max(math.degrees(math.acos(min(1, h['n'][2]))) for h in bh), 1)
ramps = L['ev']['ramps']
info['ramps(who,kind,t0,deg,crr,v0,v1,L,k_land)'] = [(r['who'], r['kind'], round(r['t0'], 2), round(math.degrees(r['alpha']), 2), round(r['crr'], 3), round(r['v0'], 2), round(r['v1'], 2), round(r['L'], 1), round(r['k_land'], 2)) for r in ramps]
for k, v in info.items(): print(k, v)
