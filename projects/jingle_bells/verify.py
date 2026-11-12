"""Checks for jingle_bells (no Blender needed): python3 verify.py [table]
 - every melody note is struck at its exact time with the right pitch (by one or both racers); bass / off-beat times on grid
 - contact distance between ball and key on the rendered frame nearest each hit (and at the exact time)
 - free flight: per-frame acceleration == gravity; every bounce: e <= 0.95, |v_out| <= |v_in| (snow bank: friction <= mu)
 - collisions: a racer entering any key other than at its own hit (sampled at 240 Hz)
 - ramps (start, in-run): speed changes smoothly, no jumps"""
import json, os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(f'{HERE}/layout.json')); LEAD = json.load(open(f'{HERE}/lead.json')); E = json.load(open(f'{HERE}/events.json'))
G = L['G']; R = L['R']; FPS = 30; FINAL = E['final']
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'
B = L['balls']


def seg_at(g, t):
    if g['k'] == 'rest': return np.array(g['p'])
    if g['k'] == 'fly':
        u = t - g['t0']; return np.array(g['p']) + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
    ts = np.array(g['ts']); ps = np.array(g['ps'])
    return np.array([np.interp(t, ts, ps[:, c]) for c in range(3)])


def at_many(b, ts):
    out = np.zeros((len(ts), 3))
    for g in b['seg']:
        m = (ts >= g['t0'] - 1e-9) & (ts <= g['t1'] + 1e-9)
        if m.any(): out[m] = np.array([seg_at(g, t) for t in ts[m]])
    s0 = b['seg'][0]; s1 = b['seg'][-1]
    out[ts < s0['t0']] = seg_at(s0, s0['t0']); out[ts > s1['t1']] = seg_at(s1, s1['t1'])
    return out


info = {}
# 1) notes
miss = []; terr = 0.0; table = []
for i, (t, p, v, d, bar, pos, sec) in enumerate(LEAD):
    hs = [(n, h) for n, b in B.items() for h in b['hits'] if h['ev'] == i]
    ok = [h for n, h in hs if h['pitch'] == p and abs(h['t'] - t) < 1e-6]
    if not ok or len(ok) != len(hs): miss.append(i)
    table.append((i, t, bar, pos, p, '+'.join(sorted(n for n, h in hs))))
info['melody_notes'] = len(LEAD); info['melody_missing_or_wrong'] = miss
info['notes_by_both'] = sum(1 for r in table if '+' in r[5]); info['notes_red_only'] = sum(1 for r in table if r[5] == 'red')
grid = lambda t: abs(((t - E['t0']) / E['e8']) - round((t - E['t0']) / E['e8'])) < 1e-6 or t > E['bar_t']['63'] - 1e-6
info['bass_on_grid'] = all(grid(b[0]) for b in E['bass']); info['stabs_on_grid'] = all(grid(s[0]) for s in E['stabs'])
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    for i, t, bar, pos, p, who in table: print(f'mel {i:3d} {t:7.3f}s bar{bar:2d} 8th{pos:<4} {nm(p):4s} frame {t * FPS:7.1f} {who}')
    for t, p, bar, pos, ch in E['bass']: print(f'bass {t:7.3f}s bar{bar:2d} {nm(p)} {ch}')
# 2) contact gaps
gaps = []; ex = 0.0
for n, b in B.items():
    for h in b['hits']:
        f = round(h['t'] * FPS); p = at_many(b, np.array([f / FPS]))[0]
        gaps.append(float(np.linalg.norm(p - np.array(h['p'])))); ex = max(ex, float(np.linalg.norm(at_many(b, np.array([h['t']]))[0] - np.array(h['p']))))
info['contact_gap_nearest_frame_max_m'] = round(max(gaps), 3); info['contact_gap_mean_m'] = round(float(np.mean(gaps)), 3)
info['contact_gap_exact_time_max_m'] = round(ex, 6)
# 3) bounces
ee = []; sp = []; fr = []
for n, b in B.items():
    for h in b['hits']:
        vin, vout = np.array(h['vin']), np.array(h['vout'])
        if np.linalg.norm(vout) == 0: continue
        sp.append(np.linalg.norm(vout) / np.linalg.norm(vin))
        if h.get('mu') is not None:
            nrm = np.array(h['n']); dv = vout - vin; dn = dv @ nrm; dt_ = np.linalg.norm(dv - dn * nrm)
            fr.append(dt_ / dn); ee.append(h['e_n'])
        else: ee.append(h['e'])
info['e(min,max)'] = (round(min(ee), 3), round(max(ee), 3)); info['speed_out/in_max'] = round(max(sp), 4)
info['snowbank_friction_ratio'] = [round(x, 3) for x in fr]
# 4) flight: per-frame acceleration and speed ratio between frames (flights only)
acc_err = 0.0; ratio = 1.0
for n, b in B.items():
    for g in b['seg']:
        if g['k'] != 'fly' or g['t1'] - g['t0'] < 3 / FPS: continue
        ts = np.arange(g['t0'], g['t1'], 1 / FPS); P = np.array([seg_at(g, t) for t in ts])
        a = (P[2:] - 2 * P[1:-1] + P[:-2]) * FPS * FPS
        acc_err = max(acc_err, float(np.abs(a - np.array([0, 0, -G])).max()))
        v = np.linalg.norm(np.diff(P, axis=0), axis=1) * FPS
        if len(v) > 1: ratio = max(ratio, float((v[1:] / np.maximum(v[:-1], 1e-6)).max()))
info['flight_accel_err_max'] = round(acc_err, 6); info['flight_speed_ratio_per_frame_max'] = round(ratio, 3)
# ramps (path segments): max speed jump between consecutive frames
jump = 0.0
for n, b in B.items():
    for g in b['seg']:
        if g['k'] != 'path': continue
        ts = np.arange(g['t0'], g['t1'], 1 / FPS)
        if len(ts) < 4: continue
        P = np.array([seg_at(g, t) for t in ts]); v = np.linalg.norm(np.diff(P, axis=0), axis=1) * FPS
        jump = max(jump, float(np.abs(np.diff(v)).max()))
info['path_speed_change_per_frame_max_mps'] = round(jump, 3)
# 5) collisions with keys (box 0.16 x 0.62 x 0.07 at each hit, crystals 0.17 radius) other than at the ball's own hit
def frame(n):
    n = np.array(n, float); n /= np.linalg.norm(n)
    xa = np.cross([0, 1, 0], n); xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return np.stack([xa, ya, n], 1)
keys = []
for n, b in B.items():
    for h in b['hits']:
        if h['kind'] in ('star', 'ornament'): continue
        nv = np.array(h['n']) / np.linalg.norm(h['n']); top = np.array(h['p']) - R * nv
        hw = {'snowbank': 0.25, 'switch': 0.14, 'crystal': 0.13}.get(h['kind'], 0.08)
        keys.append((n, h['t'], top - nv * 0.035, frame(nv), np.array([hw, 0.31, 0.035])))
ts = np.arange(0.3, FINAL + 0.5, 1 / 240)
PB = {n: at_many(b, ts) for n, b in B.items()}
worst = 0.0; bad = []
for kn, kt, c, M, ext in keys:
    for n, P in PB.items():
        m = (np.abs(P[:, 0] - c[0]) < 0.6) & (np.abs(P[:, 2] - c[2]) < 0.6) & (np.abs(P[:, 1] - c[1]) < 0.7)
        if n == kn: m &= np.abs(ts - kt) > 0.03
        if not m.any(): continue
        rel = (P[m] - c) @ M
        q = np.maximum(np.abs(rel) - ext, 0); d = np.linalg.norm(q, axis=1)
        pen = R - d.min()
        if pen > 0.002: bad.append((n, kn, round(kt, 2), round(float(pen), 3)))
        worst = max(worst, pen)
info['key_collisions'] = bad[:10]; info['key_collision_count'] = len(bad)
for k, v in info.items(): print(k, v)
