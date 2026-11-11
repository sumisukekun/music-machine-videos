"""Checks for eine_kleine (no Blender needed): python3 verify.py [table]
 - every violin I / violin II / cello event is struck at its exact time by its ball, with the right pitch(es)
 - contact distance between ball and key on the rendered frame nearest each hit, and at the exact time
 - collisions: ball surface entering any key / bell / bar other than its own at its own hit (sampled at 240 Hz)
 - free flight: per-frame acceleration == gravity, frame-to-frame speed ratio; every bounce: e <= emax, |v_out| <= |v_in|
 - mechanisms (ramps, tracks, lifts): frame-to-frame speed change"""
import json, os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(f'{HERE}/layout.json')); E = json.load(open(f'{HERE}/events.json'))
LEAD = json.load(open(f'{HERE}/lead.json')); DUO = json.load(open(f'{HERE}/duo.json')); BASS = json.load(open(f'{HERE}/bass.json'))
G = L['G']; FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.3) * FPS)
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
# 1) notes: each score event struck by its ball(s) at the exact time with the score pitch
def struck(names, t):
    return sorted(set(h['pitch'] for n in names for h in B[n]['hits'] if abs(h['t'] - t) < 1e-6 and h['pitch'] is not None))
bad = []; table = []
for i, (t, p, v, d, ps, k, b) in enumerate(LEAD):
    got = struck(['lamp'], t)
    if got != [p]: bad.append(('V1', i, round(t, 3), nm(p), [nm(x) for x in got]))
    table.append(('V1', t, ps, k, b, [p], got))
for i, (t, pp, v, d, ps, k, b) in enumerate(DUO):
    got = struck(['rose', 'pearl'], t)
    if got != sorted(pp): bad.append(('V2', i, round(t, 3), [nm(x) for x in pp], [nm(x) for x in got]))
    table.append(('V2', t, ps, k, b, pp, got))
for i, (t, p, v, d, ps, k, b) in enumerate(BASS):
    got = struck(['bass'], t)
    if got != [p]: bad.append(('B', i, round(t, 3), nm(p), [nm(x) for x in got]))
    table.append(('B ', t, ps, k, b, [p], got))
info['events'] = dict(V1=len(LEAD), V2=len(DUO), B=len(BASS)); info['events_wrong_or_missing'] = bad
fin = struck(['lamp', 'rose', 'pearl', 'bass'], FINAL)
info['final_chord_struck'] = [nm(p) for p in fin]
# every hit with a pitch belongs to a score event (no extra sounding hits)
sched = set([round(l[0], 5) for l in LEAD] + [round(l[0], 5) for l in DUO] + [round(l[0], 5) for l in BASS] + [round(FINAL, 5)])
extra = [(n, round(h['t'], 3), h['kind']) for n, b in B.items() for h in b['hits'] if h['pitch'] is not None and round(h['t'], 5) not in sched]
info['pitched_hits_not_in_score'] = extra
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    for v_, t, ps, k, b, pp, got in sorted(table, key=lambda r: (r[1], r[0])):
        print(f'{v_} {t:7.3f}s P{ps} bar{k:2d} beat{b + 1:<5g} score {"+".join(nm(p) for p in pp):8s} struck {"+".join(nm(p) for p in got):8s} frame {t * FPS:7.2f}')
# 2) contact distance at the nearest frame and at the exact time
gaps = []; ex = 0.0
for n, b in B.items():
    for h in b['hits']:
        if h['kind'] in ('seesaw',): continue
        f = round(h['t'] * FPS); p = at(b, f / FPS)
        gaps.append((round(float(np.linalg.norm(p - np.array(h['p']))), 3), n, h['kind'], round(h['t'], 3)))
        ex = max(ex, float(np.linalg.norm(at(b, h['t']) - np.array(h['p']))))
gaps.sort(reverse=True)
info['contact_gap_nearest_frame_max_m'] = gaps[0][0]; info['contact_gap_mean_m'] = round(float(np.mean([g[0] for g in gaps])), 3)
info['contact_gap_worst'] = gaps[:3]; info['contact_gap_exact_time_max_m'] = round(ex, 6)
# 3) collisions: key boxes (lamp candlesticks stay for good, couple keys / bass bells live t-1.2 .. t+1.25, ramp bars stay)
R = {'lamp': L['R']['lamp'], 'rose': L['R']['duo'], 'pearl': L['R']['duo'], 'bass': L['R']['bass']}
pieces = []
for n, b in B.items():
    r = R[n]
    for h in b['hits']:
        kd = h['kind']; nv = np.array(h['n'])
        if kd.endswith('land'): nv = np.array([rp['n'] for rp in L['ev']['ramps'] if abs(rp['t0'] - h['t']) < 1e-6][0])
        top = np.array(h['p']) - r * nv
        if kd in ('candle', 'dark', 'candle2'): hw, hd, th, life = 0.16, 0.26, 0.04, (h['t'] - 1.2, 1e9)
        elif kd == 'brazier': hw, hd, th, life = 0.42, 0.42, 0.03, (h['t'] - 1.2, 1e9)
        elif kd in ('xylo', 'xylo2', 'xyloland', 'xylo2land', 'track', 'track2', 'trackland', 'track2land'):
            hw, hd, th, life = 0.045, (0.2 if n == 'lamp' else 0.27), 0.03, (0, 1e9)
        elif kd in ('key', 'uni'):
            hw, hd, th = 0.19, 0.19, 0.05
            life = (h['t'] - 0.15, h['t'] + 0.2) if h['t'] > E['grid'][-2][2] - 0.2 else (h['t'] - 0.35, h['t'] + 1.25)
        elif kd == 'bell': hw, hd, th, life = 0.13, 0.13, 0.05, (h['t'] - 1.2, h['t'] + 1.25)
        else: continue
        pieces.append(dict(ball=n, t=h['t'], kind=kd, c=top - nv * th, M=frame(nv), hw=hw, hd=hd, th=th, life=life))
def rscale(t, Bx):
    """the scene's riser(): pops in over `pop` s from t_hit-lead (overshoot 1.18), shrinks after t_hit+stay over `shrink` s"""
    lead, stay, pop, shrink = Bx.get('rs', (1.2, 0.9, 0.3, 0.35))
    if Bx['t'] <= E['t0'] + 0.1 or Bx['kind'].startswith(('xylo', 'track')): return np.ones_like(t)
    u1 = (t - (Bx['t'] - lead)) / pop; u2 = (t - (Bx['t'] + stay)) / shrink
    a = np.where(u1 <= 0, 0.0, np.where(u1 < 0.7, 1.18 * np.sin(np.clip(u1, 0, 1) * np.pi / 2), np.where(u1 < 1, 1.18 - 0.18 * (u1 - 0.7) / 0.3, 1.0)))
    return a * (1.0 - np.clip(u2, 0, 1) ** 1.5)


for Bx in pieces:
    if Bx['kind'] in ('candle', 'dark', 'candle2', 'brazier'): Bx['rs'] = (1.2, 1e6, 0.3, 0.35)
    elif Bx['kind'] in ('key', 'uni'): Bx['rs'] = (0.15, 0.05, 0.1, 0.15) if Bx['t'] > E['grid'][-2][2] - 0.2 else (0.35, 0.9, 0.15, 0.35)
ts = np.arange(0.2, FINAL + 1.0, 1 / 240)
P = {n: np.array([at(b, t) for t in ts]) for n, b in B.items()}
badc = {}; worst = 0.0
for Bx in pieces:
    for n, Pn in P.items():
        r = R[n]
        m = (np.abs(Pn[:, 0] - Bx['c'][0]) < 0.9) & (np.abs(Pn[:, 2] - Bx['c'][2]) < 0.9) & (np.abs(Pn[:, 1] - Bx['c'][1]) < 0.9)
        own = (Bx['ball'] == n) or (Bx['ball'] == 'pair' and n in ('rose', 'pearl'))
        if own: m &= np.abs(ts - Bx['t']) > 0.03
        m &= (ts > Bx['life'][0]) & (ts < Bx['life'][1])
        if not m.any(): continue
        s_ = rscale(ts[m], Bx)                                   # pop-in / shrink-away scale of the piece (about its top)
        top = Bx['c'] + Bx['M'][:, 2] * Bx['th']
        rel = (Pn[m] - (top[None, :] - Bx['M'][:, 2][None, :] * (Bx['th'] * s_)[:, None])) @ Bx['M']
        d = np.stack([np.maximum(0, np.abs(rel[:, 0]) - Bx['hw'] * s_), np.maximum(0, np.abs(rel[:, 1]) - Bx['hd'] * s_), np.maximum(0, np.abs(rel[:, 2]) - Bx['th'] * s_)], 1)
        pen = r - np.linalg.norm(d, axis=1)
        # rolling over the bars of a ramp: the ball sits on the bar plane (expected contact, not a collision)
        if Bx['kind'] in ('xylo', 'xylo2', 'xyloland', 'xylo2land', 'track', 'track2', 'trackland', 'track2land') and own: pen = pen - 0.012
        k = pen > 0.01
        if k.any():
            key = f"{n}->{Bx['ball']}:{Bx['kind']}"; badc.setdefault(key, []).append((round(float(ts[m][k][0]), 3), round(float(pen[k].max()), 3), 'key_t', round(Bx['t'], 3)))
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
info['mechanism_speed_change_per_frame_max_mps'] = (round(dv, 3), dvw)
hs = [h for b in B.values() for h in b['hits'] if np.linalg.norm(h['vout']) > 0]
info['bounce_speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in hs), 4)
bh = [h for h in hs if h['kind'] in ('candle', 'dark', 'candle2', 'brazier', 'uni', 'key', 'bell', 'bigbell')]
info['restitution_range'] = (round(min(h['e'] for h in bh), 3), round(max(h['e'] for h in bh), 3))
info['key_tilt_max_deg'] = round(max(math.degrees(math.acos(min(1, h['n'][2]))) for h in bh if h['kind'] != 'bigbell'), 1)
ramps = L['ev']['ramps']
info['ramps(who,pass,deg,crr,v0,v1,L,k_land)'] = [(r['who'], r['pass'], round(math.degrees(r['alpha']), 2), round(r['crr'], 3), round(r['v0'], 2), round(r['v1'], 2), round(r['L'], 1), round(r['k_land'], 2)) for r in ramps]
for k, v in info.items(): print(k, v)
