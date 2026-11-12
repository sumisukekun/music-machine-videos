"""Checks for brahms_lullaby (needs pieces.json from `python3 scene.py -- pieces`): python3 verify.py [table]
 - every voice / right-hand / left-hand event is struck at its exact time by its ball(s) with the score pitch(es); the final chord
 - contact distance between ball and instrument on the rendered frame nearest each hit, and at the exact time
 - collisions: a ball entering any chime / block / cradle / drum / fence / pillow / cup / bell other than its own contact
   (sampled at 240 Hz, using each piece's pop-up / vanish scale)
 - free flight: per-frame acceleration == gravity, frame-to-frame speed ratio; every bounce: e <= emax, |v_out| <= |v_in|
 - mechanisms (cradle roll-outs, see-saw): frame-to-frame speed change"""
import json, os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(f'{HERE}/layout.json')); E = json.load(open(f'{HERE}/events.json'))
LEAD = json.load(open(f'{HERE}/lead.json')); DUO = json.load(open(f'{HERE}/duo.json')); BASS = json.load(open(f'{HERE}/bass.json'))
PC = json.load(open(f'{HERE}/pieces.json'))
G = L['G']; FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.6) * FPS)
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


B = L['balls']; info = {}
SHEEP = [n for n in B if n.startswith('sheep')] + ['giant']


def pl(p): return p if isinstance(p, list) else [p]
def struck(names, t):
    return sorted(set(q for n in names for h in B[n]['hits'] if abs(h['t'] - t) < 1e-6 and h['pitch'] is not None for q in pl(h['pitch'])))


bad = []; table = []
for i, (t, p, v, d, ps, k, b) in enumerate(LEAD):
    got = struck(['star'], t)
    if got != [p]: bad.append(('V', i, round(t, 3), nm(p), [nm(x) for x in got]))
    table.append(('V ', t, ps, k, b, [p], got))
for i, (t, pp, v, d, ps, k, b) in enumerate(DUO):
    got = struck(['pink', 'blue'], t)
    if got != sorted(pp): bad.append(('RH', i, round(t, 3), [nm(x) for x in pp], [nm(x) for x in got]))
    table.append(('RH', t, ps, k, b, pp, got))
for i, (t, pp, v, d, ps, k, b) in enumerate(BASS):
    got = struck(SHEEP, t)
    if got != sorted(pp): bad.append(('LH', i, round(t, 3), [nm(x) for x in pp], [nm(x) for x in got]))
    table.append(('LH', t, ps, k, b, pp, got))
info['events'] = dict(V=len(LEAD), RH=len(DUO), LH=len(BASS)); info['events_wrong_or_missing'] = bad
info['final_chord_struck'] = [nm(p) for p in struck(['star', 'pink', 'blue'], FINAL)]
sched = set([round(l[0], 5) for l in LEAD] + [round(l[0], 5) for l in DUO] + [round(l[0], 5) for l in BASS] + [round(FINAL, 5)])
info['pitched_hits_not_in_score'] = [(n, round(h['t'], 3), h['kind']) for n, b in B.items() for h in b['hits'] if h['pitch'] is not None and round(h['t'], 5) not in sched]
# one sheep per bar, each bar's three left-hand notes on that sheep
per = {}
for n in SHEEP:
    for h in B[n]['hits']:
        if h['pitch'] is not None: per.setdefault(n, []).append(h['t'])
info['sheep'] = (len(SHEEP), sorted(set(len(v) for v in per.values())))
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    for v_, t, ps, k, b, pp, got in sorted(table, key=lambda r: (r[1], r[0])):
        print(f'{v_} {t:7.3f}s P{ps} bar{k!s:>4} beat{b + 1:<5g} score {"+".join(nm(p) for p in pp):12s} struck {"+".join(nm(p) for p in got):12s} frame {t * FPS:7.2f}')
# contact distance at the nearest frame and at the exact time
gaps = []; ex = 0.0
for n, b in B.items():
    for h in b['hits']:
        f = round(h['t'] * FPS); p = at(b, f / FPS)
        gaps.append((round(float(np.linalg.norm(p - np.array(h['p']))), 3), n, h['kind'], round(h['t'], 3)))
        ex = max(ex, float(np.linalg.norm(at(b, h['t']) - np.array(h['p']))))
gaps.sort(reverse=True)
info['contact_gap_nearest_frame_max_m'] = gaps[0][0]; info['contact_gap_mean_m'] = round(float(np.mean([g[0] for g in gaps])), 3)
info['contact_gap_worst'] = gaps[:3]; info['contact_gap_exact_time_max_m'] = round(ex, 6)
# collisions
R = {n: b['r'] for n, b in B.items()}


def rscale(t, Bx):
    if not Bx.get('rs'): return np.ones_like(t)
    lead, stay, pop, shrink = Bx['rs']
    u1 = (t - (Bx['t'] - lead)) / pop; u2 = (t - (Bx['t'] + stay)) / shrink
    a = np.where(u1 <= 0, 0.0, np.where(u1 < 0.7, 1.18 * np.sin(np.clip(u1, 0, 1) * np.pi / 2), np.where(u1 < 1, 1.18 - 0.18 * (u1 - 0.7) / 0.3, 1.0)))
    if Bx['kind'] == 'cradle': a = np.clip(u1, 0, 1)
    return a * (1.0 - np.clip(u2, 0, 1) ** 1.5)


ts = np.arange(0.2, FINAL + 1.5, 1 / 240)
P = {n: np.array([at(b, t) for t in ts]) for n, b in B.items()}
badc = {}; worst = 0.0
for Bx in PC['pieces']:
    c0 = np.array(Bx['top']); M = frame(Bx['n'])
    for n, Pn in P.items():
        r = R[n]
        m = (np.abs(Pn[:, 0] - c0[0]) < 1.6) & (np.abs(Pn[:, 2] - c0[2]) < 1.6) & (np.abs(Pn[:, 1] - c0[1]) < 1.6)
        own = (Bx['ball'] == n) or (Bx['ball'] == 'pair' and n in ('pink', 'blue'))
        if own:
            o0, o1 = Bx['own'] if Bx.get('own') else (Bx['t'] - 0.03, Bx['t'] + 0.03)
            m &= (ts < o0) | (ts > o1)
        if Bx.get('life'): m &= (ts > Bx['life'][0]) & (ts < Bx['life'][1])
        if not m.any(): continue
        s_ = rscale(ts[m], Bx)
        ok_ = s_ > 1e-3
        if not ok_.any(): continue
        tt_ = ts[m][ok_]; s_ = s_[ok_]
        cc = np.repeat(c0[None, :], len(tt_), 0)
        if Bx.get('sway'):                                       # struck chimes are nudged back (scene: sway_dx)
            u = np.maximum(0, tt_ - Bx['t'])
            cc[:, 0] += np.where(tt_ > Bx['t'], -Bx['sway'] * (1 - np.exp(-u / 0.25)) + 0.03 * np.exp(-u / 1.2) * np.sin(5.0 * u), 0.0)
        rel = (Pn[m][ok_] - (cc - M[:, 2][None, :] * (Bx['th'] * s_)[:, None])) @ M
        d = np.stack([np.maximum(0, np.abs(rel[:, 0]) - Bx['hw'] * s_), np.maximum(0, np.abs(rel[:, 1]) - Bx['hd'] * s_), np.maximum(0, np.abs(rel[:, 2]) - Bx['th'] * s_)], 1)
        pen = r - np.linalg.norm(d, axis=1)
        k = pen > 0.01
        if k.any():
            key = f"{n}->{Bx['ball']}:{Bx['kind']}"; badc.setdefault(key, []).append((round(float(tt_[k][0]), 3), round(float(pen[k].max()), 3), 'piece_t', round(Bx['t'], 3)))
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
# free flight
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
        if g['k'] != 'path' or g['t1'] - g['t0'] < 0.2: continue
        tt = np.arange(g['t0'] + 1 / FPS, g['t1'] - 1 / FPS, 1 / FPS)
        Qp = np.array([seg_at(g, t) for t in tt]); sp = np.linalg.norm(np.diff(Qp, axis=0), axis=1) * FPS
        if len(sp) > 2 and float(np.abs(np.diff(sp)).max()) > dv: dv = float(np.abs(np.diff(sp)).max()); dvw = (n, round(g['t0'], 2))
info['mechanism_speed_change_per_frame_max_mps'] = (round(dv, 3), dvw)
hs = [h for b in B.values() for h in b['hits'] if np.linalg.norm(h['vout']) > 0]
info['bounce_speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in hs), 4)
bh = [h for h in hs if h['kind'] in ('chime', 'uni', 'key', 'drum')]
info['restitution_range'] = (round(min(h['e'] for h in bh), 3), round(max(h['e'] for h in bh), 3))
info['contact_tilt_max_deg'] = round(max(math.degrees(math.acos(min(1, h['n'][2]))) for h in bh), 1)
info['cradles'] = len(L['ev']['cradles']); info['sheep_counter_final'] = PC['counter']['final']
for k, v in info.items(): print(k, v)
