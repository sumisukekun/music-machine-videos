"""Checks for valkyries (no Blender needed): python3 verify.py [table]
 - every theme note is struck at its exact time with the right pitch (bars 1-8 by the one rider of that bar, from bar 9 by all
   eight riders, upper riders at the written octave, lower riders an octave down); the final chord; every bass note
 - contact distance between ball and key / drum on the rendered frame nearest each hit (and at the exact time)
 - collisions: balls entering keys / drums other than at their own hit (sampled at 240 Hz), ball-ball overlaps
 - free flight: per-frame acceleration == gravity, frame-to-frame speed ratio; every bounce: e <= 0.975, |v_out| <= |v_in|
 - ramps: speed changes smoothly (rolling, (5/7) g sin a)"""
import json, os, sys, math
import numpy as np
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(f'{HERE}/layout.json')); LEAD = json.load(open(f'{HERE}/lead.json')); E = json.load(open(f'{HERE}/events.json'))
G = L['G']; FPS = 30; FINAL = E['final']; NF = int((FINAL + 6.2) * FPS)
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'
B = L['balls']; EV = L['ev']
RID = [f'v{j}' for j in range(8)]


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


def at_many(b, ts):
    out = np.zeros((len(ts), 3))
    for g in b['seg']:
        m = (ts >= g['t0'] - 1e-9) & (ts <= g['t1'] + 1e-9)
        if not m.any(): continue
        tt = ts[m]
        if g['k'] == 'rest': out[m] = g['p']
        elif g['k'] == 'fly':
            u = (tt - g['t0'])[:, None]; out[m] = np.array(g['p']) + np.array(g['v']) * u + np.array([0, 0, -0.5 * G]) * u * u
        else:
            gt = np.array(g['ts']); gp = np.array(g['ps'])
            out[m] = np.stack([np.interp(tt, gt, gp[:, c]) for c in range(3)], 1)
    s0 = b['seg'][0]; s1 = b['seg'][-1]
    out[ts < s0['t0']] = seg_at(s0, s0['t0']); out[ts > s1['t1']] = seg_at(s1, s1['t1'])
    return out


def frame(n):
    n = np.array(n, float); n /= np.linalg.norm(n)
    xa = np.cross([0, 1, 0], n); xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return np.stack([xa, ya, n], 1)


info = {}
# 1) notes
miss = []; terr = 0.0; table = []
for i, (t, p, v, d, bar, pos, cell) in enumerate(LEAD):
    who = [f'v{bar - 1}'] if bar <= 8 else RID
    got = 0
    for n in who:
        j = int(n[1:]); pe = p if bar <= 8 else p - 12 * (j % 2)
        hs = [h for h in B[n]['hits'] if abs(h['t'] - t) < 1e-6 and h['pitch'] == pe and h['kind'] == 'key']
        if hs: got += 1; terr = max(terr, abs(hs[0]['t'] - t))
        else: miss.append((i, n))
    # nobody else strikes a key at that time in act 1
    if bar <= 8:
        extra = [n for n in RID if n not in who and any(abs(h['t'] - t) < 1e-6 and h['kind'] == 'key' for h in B[n]['hits'])]
        if extra: miss.append(('extra', i, extra))
    table.append((i, t, bar, pos, cell, p, got))
nkeys = sum(1 for n in RID for h in B[n]['hits'] if h['kind'] == 'key')
info['melody_notes'] = len(LEAD); info['rider_key_strikes'] = nkeys; info['melody_missing_or_wrong'] = miss; info['melody_time_err'] = terr
fm = [(n, h['pitch'], round(h['t'], 4)) for n in RID for h in B[n]['hits'] if h['kind'] == 'final']
info['final_chord'] = (sorted(nm(p) for n, p, t in fm), all(abs(t - FINAL) < 1e-6 for n, p, t in fm),
                       all(p == E['fin_p'][int(n[1:])] for n, p, t in fm))
bm = []
for t, p, bar, pos, ch in E['bass']:
    if not any(abs(h['t'] - t) < 1e-6 and h['pitch'] == p for h in B['bass']['hits']): bm.append((t, p))
info['bass_notes'] = len(E['bass']); info['bass_missing'] = bm
info['bass_extra_hits'] = len(B['bass']['hits']) - len(E['bass'])
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    for i, t, bar, pos, cell, p, got in table:
        print(f'mel {i:3d} {t:7.3f}s frame {t * FPS:7.1f} bar{bar:2d} {cell:3s} pos{pos:<3d} {nm(p):4s} riders {got}')
    for t, p, bar, pos, ch in E['bass']: print(f'bass {t:7.3f}s bar{bar:2d} pos{pos:<3d} {nm(p):4s} chord {ch}')
# 2) contact distance
gaps = []; ex = 0.0; GR = {}
for n, b in B.items():
    for h in b['hits']:
        f = round(h['t'] * FPS); p = at(b, f / FPS)
        gaps.append(float(np.linalg.norm(p - np.array(h['p'])))); GR.setdefault(b['role'], []).append(gaps[-1])
        ex = max(ex, float(np.linalg.norm(at(b, h['t']) - np.array(h['p']))))
info['contact_gap_nearest_frame_max_m'] = round(max(gaps), 3); info['contact_gap_mean_m'] = round(float(np.mean(gaps)), 3)
info['contact_gap_exact_time_max_m'] = round(ex, 6)
info['contact_gap_nearest_frame_by_role(max, mean)'] = {k: (round(max(v), 3), round(float(np.mean(v)), 3)) for k, v in GR.items()}
# 3) collisions with keys / drums (boxes) of every ball, other than the ball's own hit
R = {n: b['r'] for n, b in B.items()}
pieces = []
for n, b in B.items():
    for h in b['hits']:
        if h['kind'] == 'cup': continue
        nv = np.array(h['n']) if np.linalg.norm(h['vout']) > 0 else np.array([0, 0, 1.0]); top = np.array(h['p']) - R[n] * nv
        if b['role'] == 'rider': hw, hd, th = 0.16, 0.25, 0.06
        else: hw, hd, th = 0.42, 0.42, 0.25
        keep = h['kind'] == 'final'
        pieces.append(dict(ball=n, t=h['t'], c=top - nv * th, M=frame(nv), hw=hw, hd=hd, th=th, keep=keep,
                           life=(h['t'] - (1.0 if not keep else 0.45), (h['t'] + 1.15) if not keep else 1e9)))
ts = np.arange(0.2, FINAL + 2.0, 1 / 240)
P = {n: at_many(b, ts) for n, b in B.items()}
bad = {}; worst = 0.0
for Bp in pieces:
    for n, Pn in P.items():
        r = R[n]
        m = (np.abs(Pn[:, 0] - Bp['c'][0]) < 1.0) & (np.abs(Pn[:, 1] - Bp['c'][1]) < 1.0) & (np.abs(Pn[:, 2] - Bp['c'][2]) < 1.0)
        m &= (ts > Bp['life'][0]) & (ts < Bp['life'][1])
        if Bp['ball'] == n:
            m &= np.abs(ts - Bp['t']) > 0.03
            if Bp['keep']: m &= ts < Bp['t']
        if not m.any(): continue
        rel = (Pn[m] - Bp['c']) @ Bp['M']
        d = np.stack([np.maximum(0, np.abs(rel[:, 0]) - Bp['hw']), np.maximum(0, np.abs(rel[:, 1]) - Bp['hd']), np.maximum(0, np.abs(rel[:, 2]) - Bp['th'])], 1)
        pen = r - np.linalg.norm(d, axis=1); k = pen > 0.01
        if k.any():
            key = f"{'rider' if n != 'bass' else 'bass'}->{'own' if n == Bp['ball'] else 'other'}"
            bad.setdefault(key, []).append((n, Bp['ball'], round(float(ts[m][k][0]), 3), round(float(pen[k].max()), 3), round(Bp['t'], 3)))
            worst = max(worst, float(pen[k].max()))
info['collisions'] = {k: len(v) for k, v in bad.items()}; info['collision_examples'] = {k: v[:4] for k, v in bad.items()}
info['collision_worst_pen_m'] = round(worst, 3)
# ball-ball overlaps
tb = np.arange(0.5, FINAL + 3.0, 1 / 120)
names = list(B); RA = np.array([R[n] for n in names])
ALL = np.stack([at_many(B[n], tb) for n in names], 1)
mind = (9.0, None)
for k in range(len(tb)):
    tr = cKDTree(ALL[k]); prs = tr.query_pairs(0.8, output_type='ndarray')
    if not len(prs): continue
    d = np.linalg.norm(ALL[k, prs[:, 0]] - ALL[k, prs[:, 1]], axis=1) - RA[prs[:, 0]] - RA[prs[:, 1]]
    j = int(np.argmin(d))
    if d[j] < mind[0]: mind = (round(float(d[j]), 4), names[prs[j, 0]], names[prs[j, 1]], round(float(tb[k]), 2))
info['min_ball_surface_gap_m'] = mind
# 4) free flight + bounces
tf = np.arange(NF) / FPS
amax = 0.0; rat = 1.0; worst_r = None
for n, b in B.items():
    for g in b['seg']:
        if g['k'] != 'fly' or g['t1'] - g['t0'] < 3 / FPS: continue
        fr = np.nonzero((tf[:-2] > g['t0'] + 1e-6) & (tf[2:] < g['t1'] - 1e-6))[0] + 1
        if not len(fr): continue
        Q3 = np.stack([np.array([seg_at(g, tf[f + k]) for f in fr]) for k in (-1, 0, 1)])
        a = (Q3[2] - 2 * Q3[1] + Q3[0]) * FPS * FPS
        amax = max(amax, float(np.abs(a - np.array([0, 0, -G])).max()))
        s0 = np.linalg.norm(Q3[1] - Q3[0], axis=1) * FPS; s1 = np.linalg.norm(Q3[2] - Q3[1], axis=1) * FPS
        mm = np.minimum(s0, s1) > 0.5
        if mm.any():
            r_ = (np.maximum(s0, s1)[mm] / np.minimum(s0, s1)[mm]); j = int(np.argmax(r_))
            if r_[j] > rat: rat = float(r_[j]); worst_r = (n, round(float(tf[fr[mm][j]]), 2), round(float(s0[mm][j]), 2), round(float(s1[mm][j]), 2))
info['flight_accel_error_max'] = round(amax, 6); info['flight_speed_ratio_max(v>0.5)'] = round(rat, 3); info['worst_ratio_at'] = worst_r
dv = 0.0
for n, b in B.items():
    for g in b['seg']:
        if g['k'] != 'path' or g['t1'] - g['t0'] < 0.15: continue
        tt = np.arange(g['t0'], g['t1'], 1 / FPS)
        Q3 = np.array([seg_at(g, t) for t in tt]); sp = np.linalg.norm(np.diff(Q3, axis=0), axis=1) * FPS
        if len(sp) > 2: dv = max(dv, float(np.abs(np.diff(sp)).max()))
info['ramp_speed_change_per_frame_max_mps'] = round(dv, 3)
hs = [h for b in B.values() for h in b['hits']]
info['bounce_speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in hs), 4)
info['restitution_range'] = (round(min(h['e'] for h in hs if np.linalg.norm(h['vout']) > 0), 3), round(max(h['e'] for h in hs), 3))
info['dead_landings(cups+final)'] = sum(1 for h in hs if np.linalg.norm(h['vout']) == 0)
info['key_tilt_max_deg'] = round(max(math.degrees(math.acos(min(1, h['n'][2]))) for h in hs if np.linalg.norm(h['vout']) > 0), 1)
# 5) the race: leader changes (largest x among the eight riders, sampled per frame from bar 9 to the finish)
tr_ = tf[(tf > E['bar_t']['9'] + 0.6) & (tf < FINAL - 0.05)]
X = np.stack([at_many(B[n], tr_)[:, 0] for n in RID])
lead_ = np.argmax(X, 0); ch = int((np.diff(lead_) != 0).sum())
info['leader_changes'] = ch; info['leaders'] = sorted(set(int(v) + 1 for v in lead_))
info['finish_x_spread_m'] = round(float(np.ptp([at(B[n], FINAL)[0] for n in RID])), 4)
for k, v in info.items(): print(k, v)
