"""Checks for ode_to_joy (no Blender needed): python3 verify.py [table]
 - every melody / bass / pair note is struck at its exact time with the right pitch; every rain beat rings exactly its chord
 - contact distance between ball and key / bell on the rendered frame nearest each hit (and at the exact time)
 - collisions: balls entering keys / bells / the riser other than at their own hit (sampled at 240 Hz), ball-ball overlaps
 - free flight: per-frame acceleration == gravity, frame-to-frame speed ratio; every bounce: e <= 0.975, |v_out| <= |v_in|
 - ramps: speed changes smoothly (rolling, (5/7) g sin a)"""
import json, os, sys, math
import numpy as np
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(f'{HERE}/layout.json')); LEAD = json.load(open(f'{HERE}/lead.json')); E = json.load(open(f'{HERE}/events.json'))
G = L['G']; FPS = 30; FINAL = E['final']; NF = int((FINAL + 5.5) * FPS)
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'
B = L['balls']; EV = L['ev']


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
mel = [n for n, b in B.items() if b['role'] == 'melody']
miss = []; terr = 0.0; table = []
for i, (t, p, v, d, bar, pos) in enumerate(LEAD):
    hs = [h for h in B['solo']['hits'] if h['ev'] == i]
    if not hs or hs[0]['pitch'] != p: miss.append(i)
    else: terr = max(terr, abs(hs[0]['t'] - t))
    lanes = sum(1 for n in mel if any(h['ev'] == i and abs(h['t'] - t) < 1e-6 and h['pitch'] == p for h in B[n]['hits']))
    if lanes != L['lanes_at'][i]: miss.append(('lanes', i))
    table.append((i, t, bar, pos, p, lanes))
info['melody_notes'] = len(LEAD); info['melody_missing_or_wrong'] = miss; info['melody_time_err'] = terr
bm = []
for t, p, bar, pos in E['bass']:
    if not any(abs(h['t'] - t) < 1e-6 and h['pitch'] == p for h in B['bass']['hits']): bm.append((t, p))
pm = []
for t, ps, bar, pos, ch in E['pair']:
    for n, p in zip(('pairlo', 'pairhi'), ps):
        if not any(abs(h['t'] - t) < 1e-6 and h['pitch'] == p for h in B[n]['hits']): pm.append((t, n, p))
info['bass_notes'] = len(E['bass']); info['bass_missing'] = bm; info['pair_notes'] = len(E['pair']); info['pair_missing'] = pm
rm = []; nrain = 0
for k, (t, ch, bells, bar, pos) in enumerate(E['rain']):
    got = sorted(set(p for tt, p in L['bellhits'] if abs(tt - t) < 1e-6))
    nrain += sum(1 for tt, p in L['bellhits'] if abs(tt - t) < 1e-6)
    if got != sorted(bells): rm.append((bar, pos, ch, got))
info['rain_beats'] = len(E['rain']); info['rain_balls'] = nrain; info['rain_chord_mismatch'] = rm
info['final_count'] = EV['count'][-1]
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    for i, t, bar, pos, p, lanes in table:
        print(f'mel {i:3d} {t:7.3f}s bar{bar:2d} beat{pos + 1:<4} {nm(p):4s} frame {t * FPS:7.1f} balls {lanes}')
    for t, p, bar, pos in E['bass']: print(f'bass {t:7.3f}s bar{bar:2d} beat{pos + 1} {nm(p)}  chord {E["chords"][f"{bar}-{pos}"]}')
    for t, ps, bar, pos, ch in E['pair']: print(f'pair {t:7.3f}s bar{bar:2d} beat{pos + 1} {"+".join(nm(p) for p in ps)}  chord {ch}')
    for t, ch, bells, bar, pos in E['rain']:
        c = sum(1 for tt, p in L['bellhits'] if abs(tt - t) < 1e-6)
        print(f'rain {t:7.3f}s bar{bar:2d} beat{pos + 1} {ch:6s} bells {"+".join(nm(p) for p in bells)}  balls {c}')
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
# 3) collisions with keys / bells (boxes and spheres) other than the ball's own hit
R = {n: b['r'] for n, b in B.items()}
pieces = []
LN = L['lane']
for h in B['solo']['hits']:
    nv = np.array(h['n']); top = np.array(h['p']) - B['solo']['r'] * nv
    lanes = h['lanes']; hd = (lanes - 1) * LN / 2 + 0.25
    c = top - nv * 0.06 + np.array([0, (lanes - 1) * LN / 2, 0])
    pieces.append(dict(kind='key', ball='melody', t=h['t'], c=c, M=frame(nv), hw=0.21, hd=hd, th=0.06, keep=h['kind'] == 'final'))
for n in ('bass', 'pairlo', 'pairhi'):
    for h in B[n]['hits']:
        nv = np.array(h['n']) if np.linalg.norm(h['vout']) > 0 else np.array([0, 0, 1.0]); top = np.array(h['p']) - R[n] * nv
        hw, th = ((0.24, 0.2) if n == 'bass' else (0.12, 0.12))
        pieces.append(dict(kind='bell', ball=n, t=h['t'], c=top - nv * th, M=frame(nv), hw=hw, hd=hw, th=th, keep=bool(h.get('final'))))
ts = np.arange(0.2, FINAL + 2.0, 1 / 240)
P = {n: at_many(b, ts) for n, b in B.items() if b['role'] != 'rain'}
bad = {}; worst = 0.0
for Bp in pieces:
    for n, Pn in P.items():
        r = R[n]
        m = (np.abs(Pn[:, 0] - Bp['c'][0]) < 0.9 + Bp['hd'] * 0) & (np.abs(Pn[:, 2] - Bp['c'][2]) < 0.9)
        own = (Bp['ball'] == n) or (Bp['ball'] == 'melody' and B[n]['role'] == 'melody')
        if own: m &= np.abs(ts - Bp['t']) > 0.03
        m &= (ts > Bp['t'] - 1.2) & ((ts < Bp['t'] + (1.25 if Bp['kind'] == 'key' else 1.1)) | Bp['keep'])
        if own and Bp['keep']: m &= ts < Bp['t']
        if not m.any(): continue
        rel = (Pn[m] - Bp['c']) @ Bp['M']
        d = np.stack([np.maximum(0, np.abs(rel[:, 0]) - Bp['hw']), np.maximum(0, np.abs(rel[:, 1]) - Bp['hd']), np.maximum(0, np.abs(rel[:, 2]) - Bp['th'])], 1)
        pen = r - np.linalg.norm(d, axis=1); k = pen > 0.01
        if k.any():
            key = f"{n}->{Bp['ball']}:{Bp['kind']}"; bad.setdefault(key, []).append((round(float(ts[m][k][0]), 3), round(float(pen[k].max()), 3), round(Bp['t'], 3)))
            worst = max(worst, float(pen[k].max()))
# rain bells (spheres, from the curtain on), the riser (solid tiers), for every ball
bell = EV['bells']; RBL = bell['r']; LT = bell['length']
def tube_d(P):
    d = []
    for x in bell['x']:
        q = np.stack([np.clip(P[:, 0], x - LT / 2, x + LT / 2), np.full(len(P), bell['y']), np.full(len(P), bell['z'])], 1)
        d.append(np.linalg.norm(P - q, axis=1) - RBL)
    return np.min(np.stack(d), axis=0)
ri = EV['riser']
def riser_pen(Pn, r):
    """penetration of balls into the stepped riser (tier k: y >= yr + k*depth, z <= zr + k*rise)"""
    x0, x1 = ri['x_col'][0] - ri['px'] / 2, ri['x_col'][-1] + ri['px'] / 2
    inx = (Pn[:, 0] > x0 - r) & (Pn[:, 0] < x1 + r)
    k = np.floor((Pn[:, 1] - ri['yr']) / ri['depth']); k = np.clip(k, -1, ri['nrow'] - 1)
    ztop = np.where(k >= 0, ri['zr'] + k * ri['rise'], -99)
    return np.where(inx & (Pn[:, 1] > ri['yr'] - r), r - (Pn[:, 2] - ztop), -1)
rain = [n for n, b in B.items() if b['role'] == 'rain']
tsr = np.arange(EV['curtain']['t0'], FINAL + 3.0, 1 / 240)
PR = {n: at_many(B[n], tsr) for n in rain}
bpen = 0.0; rpen = 0.0; bcount = 0
for n in rain:
    Pn = PR[n]; r = R[n]; h = B[n]['hits'][0]
    d = tube_d(Pn) - r
    m = np.abs(tsr - h['t']) > 0.02
    pen = -d[m].min()
    if pen > 0.01: bcount += 1; bpen = max(bpen, pen)
    t_seat = EV['rain'][int(n[1:])]['t_seat']
    mr = tsr < t_seat - 0.02
    rp = riser_pen(Pn[mr], r).max() if mr.any() else -1
    rpen = max(rpen, rp)
for n, Pn in P.items():
    d = tube_d(at_many(B[n], tsr)) - R[n]
    if -d.min() > 0.01: bad.setdefault(f'{n}->rainbell', []).append(round(float(-d.min()), 3))
    rp = riser_pen(at_many(B[n], tsr), R[n]).max()
    if rp > 0.01: bad.setdefault(f'{n}->riser', []).append(round(float(rp), 3))
info['collisions'] = {k: len(v) for k, v in bad.items()}; info['collision_examples'] = {k: v[:3] for k, v in bad.items()}
info['collision_worst_pen_m'] = round(worst, 3)
info['rain_vs_tubes_pen_max_m'] = (round(bpen, 3), bcount); info['rain_vs_riser_pen_max_m(before seating)'] = round(float(rpen), 3)
# ball-ball overlaps: every ball, 120 Hz, from the first note to the end
tb = np.arange(T := 0.5, FINAL + 3.0, 1 / 120)
names = list(B); RA = np.array([R[n] for n in names])
ALL = np.stack([at_many(B[n], tb) for n in names], 1)            # (T, N, 3)
for j, n in enumerate(names):                                     # rain balls are hidden inside the organ until they drop out
    if B[n]['role'] == 'rain': ALL[tb < EV['rain'][int(n[1:])]['t_out'], j] = 1e4 + 10 * j
mind = (9.0, None)
for k in range(len(tb)):
    tr = cKDTree(ALL[k]); prs = tr.query_pairs(0.62, output_type='ndarray')
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
info['dead_landings(final)'] = sum(1 for h in hs if np.linalg.norm(h['vout']) == 0)
info['key_tilt_max_deg'] = round(max(math.degrees(math.acos(min(1, h['n'][2]))) for h in hs if h['kind'] != 'rainbell' and np.linalg.norm(h['vout']) > 0), 1)
info['seesaw_energy'] = (round(EV['seesaw']['ke_bass_loss'], 2), round(EV['seesaw']['ke_pair'], 2))
for k, v in info.items(): print(k, v)
