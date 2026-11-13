"""Checks for nutcracker_march (needs pieces.json from `python3 scene.py -- pieces`): python3 verify.py [table]
 - every note of every voice (voices.json, read from the score by arrange.py) is struck at its exact time by its
   marble(s) with exactly the score pitch(es); no pitched contact that is not in the score
 - contact distance between marble and key on the rendered frame nearest each hit, and at the exact time
 - collisions: a marble entering any key / drum / ramp bar / cup other than its own contact (sampled at 240 Hz, using
   each piece's pop-up / vanish scale)
 - marble-marble overlap (main marbles; mouse marbles / shells while visible)
 - free flight: per-frame acceleration == gravity, frame-to-frame speed ratio; every bounce: e <= emax, |v_out| <= |v_in|
 - mechanisms (cups, gondolas, ramps, hatches): frame-to-frame speed change
 - the score counter: SOLDIERS / MICE = number of keys struck by each team"""
import json, os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(f'{HERE}/layout.json')); E = json.load(open(f'{HERE}/events.json')); VO = json.load(open(f'{HERE}/voices.json'))
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
PEL = [n for n in B if len(n) == 4 and n[1:].isdigit()]
MAIN = ['cap', 's2', 's3', 'drum', 'king', 'm2', 'm3', 'rat']
def pl(p): return p if isinstance(p, list) else [p]
def struck(names, t):
    return sorted(q for n in names for h in B[n]['hits'] if abs(h['t'] - t) < 1e-4 and h['pitch'] is not None for q in pl(h['pitch']))


bad = []; table = []
for v in MAIN:
    for i, e in enumerate(VO[v]):
        got = struck([v], e['t'])
        if got != sorted(e['p']): bad.append((v, i, round(e['t'], 3), [nm(x) for x in e['p']], [nm(x) for x in got]))
        table.append((v, e['t'], e['bar'], e['o'], sorted(e['p']), got, e['grace']))
for v, pre in (('army', 'a'), ('shell', 's')):
    names = [n for n in PEL if n[0] == pre]
    byt = {}
    for e in VO[v]: byt.setdefault(round(e['t'], 5), []).append((e['p'][0], e))
    for t, lst in byt.items():
        got = struck(names, t); want = sorted(p for p, _ in lst)
        if got != want: bad.append((v, round(t, 3), [nm(x) for x in want], [nm(x) for x in got]))
        e = lst[0][1]; table.append((v, t, e['bar'], e['o'], want, got, e['grace']))
info['events'] = {v: len(VO[v]) for v in VO}; info['events_wrong_or_missing'] = bad
sched = set(round(e['t'], 5) for v in VO for e in VO[v])
info['pitched_hits_not_in_score'] = [(n, round(h['t'], 3), h['kind']) for n, b in B.items() for h in b['hits'] if h['pitch'] is not None and round(h['t'], 5) not in sched]
info['pellets_one_note_each'] = all(sum(1 for h in B[n]['hits'] if h['pitch'] is not None) == 1 for n in PEL)
info['final_chord_struck'] = [nm(p) for p in struck(MAIN + PEL, FINAL)]
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    for v_, t, bar, o, pp, got, gr in sorted(table, key=lambda r: (r[1], r[0])):
        print(f'{v_:6s} {t:7.3f}s  score bar {bar:3d} +{o:6.3f}/16{" grace" if gr else "      "}  score {"+".join(nm(p) for p in pp):14s} struck {"+".join(nm(p) for p in got):14s} frame {t * FPS:8.2f}')
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
R = {n: b['r'] for n, b in B.items()}


def rscale(t, Bx):
    if not Bx.get('rs'): return np.ones_like(t)
    lead, stay, pop, shrink = Bx['rs']
    u1 = (t - (Bx['t'] - lead)) / pop; u2 = (t - (Bx['t'] + stay)) / shrink
    a = np.where(u1 <= 0, 0.0, np.where(u1 < 0.7, 1.18 * np.sin(np.clip(u1, 0, 1) * np.pi / 2), np.where(u1 < 1, 1.18 - 0.18 * (u1 - 0.7) / 0.3, 1.0)))
    return a * (1.0 - np.clip(u2, 0, 1) ** 1.5)


ts = np.arange(0.0, FINAL + 2.5, 1 / 240)
P = {n: np.array([at(b, t) for t in ts]) for n, b in B.items()}
VIS = {}
SHOW = L['ev']['pellet_show']
for n in PEL: VIS[n] = (ts > SHOW[n]) & (ts < B[n]['hits'][-1]['t'] + 0.08)
for n in MAIN: VIS[n] = np.ones(len(ts), bool)
badc = {}; worst = 0.0
for Bx in PC['pieces']:
    c0 = np.array(Bx['top']); M = frame(Bx['n'])
    for n, Pn in P.items():
        r = R[n]
        m = VIS[n] & (np.abs(Pn[:, 0] - c0[0]) < 1.6) & (np.abs(Pn[:, 2] - c0[2]) < 1.6) & (np.abs(Pn[:, 1] - c0[1]) < 1.6)
        own = (Bx['ball'] == n)
        if own:
            o0, o1 = Bx['own'] if Bx.get('own') else (Bx['t'] - 0.03, Bx['t'] + 0.03)
            m &= (ts < o0) | (ts > o1)
        if Bx.get('life'): m &= (ts > Bx['life'][0]) & (ts < Bx['life'][1])
        if not m.any(): continue
        s_ = rscale(ts[m], Bx); ok_ = s_ > 1e-3
        if not ok_.any(): continue
        tt_ = ts[m][ok_]; s_ = s_[ok_]
        cc = np.repeat(c0[None, :], len(tt_), 0)
        rel = (Pn[m][ok_] - (cc - M[:, 2][None, :] * (Bx['th'] * s_)[:, None])) @ M
        d = np.stack([np.maximum(0, np.abs(rel[:, 0]) - Bx['hw'] * s_), np.maximum(0, np.abs(rel[:, 1]) - Bx['hd'] * s_), np.maximum(0, np.abs(rel[:, 2]) - Bx['th'] * s_)], 1)
        pen = r - np.linalg.norm(d, axis=1)
        k = pen > 0.01
        if k.any():
            key = f"{'pellet' if n in PEL else n}->{Bx['kind']}"; badc.setdefault(key, []).append((n, round(float(tt_[k][0]), 3), round(float(pen[k].max()), 3), 'piece_t', round(Bx['t'], 3), Bx['ball']))
            worst = max(worst, float(pen[k].max()))
info['collisions'] = {k: len(v) for k, v in badc.items()}; info['collision_examples'] = {k: v[:3] for k, v in badc.items()}
info['collision_worst_pen_m'] = round(worst, 3)
names = list(P); mind = (9, None); pmin = (9, None)
for a in range(len(names)):
    for c in range(a + 1, len(names)):
        A_, C_ = names[a], names[c]
        both = VIS[A_] & VIS[C_]
        if not both.any(): continue
        d = np.linalg.norm(P[A_][both] - P[C_][both], axis=1) - R[A_] - R[C_]
        j = int(np.argmin(d)); rec = (round(float(d[j]), 3), A_, C_, round(float(ts[both][j]), 2))
        if A_ in MAIN or C_ in MAIN:
            if d[j] < mind[0]: mind = rec
        elif d[j] < pmin[0]: pmin = rec
info['min_surface_gap_main_marbles'] = mind
info['min_surface_gap_pellets'] = pmin
# free flight
tf = np.arange(NF) / FPS
amax = 0.0; rat = 1.0; worst_r = None
for n, b in B.items():
    for g in b['seg']:
        if g['k'] != 'fly' or g['t1'] - g['t0'] <= 3 / FPS: continue
        for f in range(int(g['t0'] * FPS), min(NF - 1, int(g['t1'] * FPS) + 2)):
            if f < 1 or not (g['t0'] + 1e-6 < tf[f - 1] and tf[f + 1] < g['t1'] - 1e-6): continue
            Qp = [seg_at(g, tf[f + k]) for k in (-1, 0, 1)]
            a = (Qp[2] - 2 * Qp[1] + Qp[0]) * FPS * FPS
            amax = max(amax, float(np.linalg.norm(a - np.array([0, 0, -G]))))
            s0 = np.linalg.norm(Qp[1] - Qp[0]) * FPS; s1 = np.linalg.norm(Qp[2] - Qp[1]) * FPS
            if min(s0, s1) > 1.0:
                r_ = max(s0, s1) / min(s0, s1)
                if r_ > rat: rat = r_; worst_r = (n, round(tf[f], 2), round(s0, 2), round(s1, 2))
info['flight_accel_error_max'] = round(amax, 6); info['flight_speed_ratio_max(v>1.0)'] = round(rat, 3); info['worst_ratio_at'] = worst_r
dv = 0.0; dvw = None
for n, b in B.items():
    for g in b['seg']:
        if g['k'] != 'path' or g['t1'] - g['t0'] < 0.2: continue
        tt = np.arange(g['t0'] + 1 / FPS, g['t1'] - 1 / FPS, 1 / FPS)
        Qp = np.array([seg_at(g, t) for t in tt]); sp = np.linalg.norm(np.diff(Qp, axis=0), axis=1) * FPS
        if len(sp) > 2 and float(np.abs(np.diff(sp)).max()) > dv: dv = float(np.abs(np.diff(sp)).max()); dvw = (n, round(g['t0'], 2))
info['mechanism_speed_change_per_frame_max_mps'] = (round(dv, 3), dvw)
hs = [h for b in B.values() for h in b['hits'] if np.linalg.norm(h['vout']) > 0 and h['kind'] in ('key', 'pellet')]
info['bounce_speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in hs), 4)
info['restitution_range'] = (round(min(h['e'] for h in hs), 3), round(max(h['e'] for h in hs), 3))
info['contact_tilt_max_deg'] = round(max(math.degrees(math.acos(min(1, h['n'][2]))) for h in hs), 1)
info['ramps'] = (len(L['ev']['ramps']), 'slopes deg', sorted(set(round(math.degrees(r['th'])) for r in L['ev']['ramps'])))
cup_in = [h for b in B.values() for h in b['hits'] if h['kind'] in ('cup', 'gondola', 'cupfinal')]
info['cup_landings_speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in cup_in), 3)
info['cups'] = len(L['ev']['cups']); info['score_counter_final'] = PC['counter']
SC = L['ev']['score_events']
s_b = sum(1 for t, s in SC if s == 'S' and t < FINAL - 1e-3); m_b = sum(1 for t, s in SC if s == 'M' and t < FINAL - 1e-3)
info['score_before_final_chord'] = dict(S=s_b, M=m_b)
for k, v in info.items(): print(k, v)
