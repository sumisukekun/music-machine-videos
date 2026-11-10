"""Checks for blue_danube (no Blender needed): python3 verify.py [table]
 - every melody event and every accompaniment event is struck at its exact time by a ball, with the right pitch(es)
 - contact distance between ball and key/bell on the rendered frame nearest each hit
 - collisions: ball surface entering any key / bell other than its own at its own hit (sampled at 240 Hz)
 - free flight: per-frame acceleration == gravity, frame-to-frame speed ratio; every bounce: e <= emax, |v_out| <= |v_in|"""
import json, os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(f'{HERE}/layout.json')); LEAD = json.load(open(f'{HERE}/lead.json')); E = json.load(open(f'{HERE}/events.json'))
G = L['G']; FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.5) * FPS)
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


info = {}
# 1) melody: pitches struck per event == score pitches (inner voice may be dropped only while one dancer plays alone)
st = {int(k): v for k, v in L['struck'].items()}
miss_top = []; t_err = 0.0; table = []
for i, (t, ps, v, d, pss, k, b) in enumerate(LEAD):
    got = sorted(set(p for _, p, _ in st.get(i, [])))
    if max(ps) not in got: miss_top.append(i)
    if any(p not in ps for p in got): miss_top.append(('wrong', i))
    for n, bb in L['balls'].items():
        for h in bb['hits']:
            if h.get('ev') == i and n in ('red', 'white') and abs(h['t'] - t) > 1e-6: t_err = max(t_err, abs(h['t'] - t))
    table.append((i, t, pss, k, b, ps, got, st.get(i, [])))
info['melody_events'] = len(LEAD); info['melody_top_note_missing_or_wrong'] = miss_top
info['melody_inner_voice_dropped(solo)'] = [i for i, t, pss, k, b, ps, got, w in table if len(ps) == 2 and len(got) == 1]
# 2) accompaniment: every bass / chord event struck by the right balls with the right pitches at the right time
acc = E['acc']; amiss = []
for j, (t, role, ps, pss, k, b) in enumerate(acc):
    names = ['bass'] if role == 'bass' else [role + 'lo', role + 'hi']
    for nme, p in zip(names, ps if role != 'bass' else ps[:1]):
        ok = any(abs(h['t'] - t) < 1e-6 and h['pitch'] == p for h in L['balls'][nme]['hits'])
        if not ok: amiss.append((round(t, 3), role, nme, p))
info['accomp_events'] = len(acc); info['accomp_missing'] = amiss
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    for i, t, pss, k, b, ps, got, who in table:
        w = ' '.join(f'{n}:{nm(p)}:{kd}' for n, p, kd in who)
        print(f'{i:3d} {t:7.3f}s P{pss} bar{k:2d} beat{b + 1}  score {"+".join(nm(p) for p in ps):8s} struck {"+".join(nm(p) for p in got):8s} frame {round(t * FPS):5d}  {w}')
    for t, role, ps, pss, k, b in acc:
        print(f'acc {t:7.3f}s P{pss} bar{k:2d} beat{b + 1} {role:4s} {"+".join(nm(p) for p in ps)}  chord {E["chords"].get(f"{pss}-{k}", "D")}')
# 3) contact distance on the nearest frame (ball centre vs the contact centre recorded in the hit)
gaps = []
for n, b in L['balls'].items():
    for h in b['hits']:
        if h['kind'] in ('seesaw',): continue
        f = round(h['t'] * FPS); p = at(b, f / FPS)
        gaps.append((float(np.linalg.norm(p - np.array(h['p']))), n, h['kind'], round(h['t'], 3)))
gaps.sort(reverse=True)
info['contact_gap_nearest_frame_max_m'] = round(gaps[0][0], 3); info['contact_gap_mean_m'] = round(float(np.mean([g[0] for g in gaps])), 3)
info['contact_gap_worst'] = gaps[:3]
# exact: position at the exact hit time
ex = max(float(np.linalg.norm(at(b, h['t']) - np.array(h['p']))) for n, b in L['balls'].items() for h in b['hits'] if h['kind'] != 'seesaw')
info['contact_gap_exact_time_max_m'] = round(ex, 6)
# 4) collisions with keys / bells (approximate boxes) other than the ball's own hit
R = {'red': L['R']['pair'], 'white': L['R']['pair'], 'bass': L['R']['bass']}
pieces = []
shared = {}
for n in ('red', 'white'):
    for h in L['balls'][n]['hits']:
        if h['kind'] in ('crack', 'reunion'): shared.setdefault(h['ev'], []).append(h)
for ev, hs in shared.items():
    nv = np.array(hs[0]['n']); top = np.mean([h['p'] for h in hs], axis=0) - L['R']['pair'] * nv
    pieces.append(dict(ball='pair', t=hs[0]['t'], kind=hs[0]['kind'], c=top - nv * 0.06, M=frame(nv), hw=0.46, hd=0.42, th=0.06, ev=ev))
for n, b in L['balls'].items():
    r = R.get(n, L['R']['chord'])
    for h in b['hits']:
        if h['kind'] in ('key', 'uni', 'half', 'floe', 'low', 'bell', 'tink'):
            nv = np.array(h['n']); top = np.array(h['p']) - r * nv
            if False: pass
            elif h['kind'] in ('key', 'uni', 'half', 'floe', 'low'): hw, hd, th = 0.21, 0.2, 0.06
            elif h['kind'] == 'bell': hw, hd, th = 0.26, 0.26, 0.2
            else: hw, hd, th = 0.16, 0.16, 0.17
            pieces.append(dict(ball=n, t=h['t'], kind=h['kind'], c=top - nv * th, M=frame(nv), hw=hw, hd=hd, th=th, ev=h.get('ev')))
ts = np.arange(0.2, FINAL + 1.0, 1 / 240)
P = {n: np.array([at(b, t) for t in ts]) for n, b in L['balls'].items()}
bad = {}; worst = 0.0
for B in pieces:
    for n, Pn in P.items():
        r = R.get(n, L['R']['chord'])
        m = (np.abs(Pn[:, 0] - B['c'][0]) < 0.8) & (np.abs(Pn[:, 2] - B['c'][2]) < 0.8) & (np.abs(Pn[:, 1] - B['c'][1]) < 0.8)
        own = (B['ball'] == n) or (B['ball'] == 'pair' and n in ('red', 'white'))
        if own: m &= np.abs(ts - B['t']) > 0.03
        # keys/bells only exist from t-1.2 s to t+0.9 s (+0.35 s shrinking); the cracked key breaks away 0.2 s after its hit
        m &= (ts > B['t'] - 1.2) & (ts < B['t'] + (0.2 if B['kind'] == 'crack' else 1.25))
        if not m.any(): continue
        rel = (Pn[m] - B['c']) @ B['M']
        d = np.stack([np.maximum(0, np.abs(rel[:, 0]) - B['hw']), np.maximum(0, np.abs(rel[:, 1]) - B['hd']), np.maximum(0, np.abs(rel[:, 2]) - B['th'])], 1)
        pen = r - np.linalg.norm(d, axis=1); k = pen > 0.01
        if k.any():
            key = f"{n}->{B['ball']}:{B['kind']}"; bad.setdefault(key, []).append((round(float(ts[m][k][0]), 3), round(float(pen[k].max()), 3), 'key_t', round(B['t'], 3)))
            worst = max(worst, float(pen[k].max()))
info['collisions'] = {k: len(v) for k, v in bad.items()}; info['collision_examples'] = {k: v[:3] for k, v in bad.items()}
info['collision_worst_pen_m'] = round(worst, 3)
# ball-ball overlap
names = list(P); mind = (9, None)
for a in range(len(names)):
    for c in range(a + 1, len(names)):
        d = np.linalg.norm(P[names[a]] - P[names[c]], axis=1) - R.get(names[a], L['R']['chord']) - R.get(names[c], L['R']['chord'])
        j = int(np.argmin(d))
        if d[j] < mind[0]: mind = (round(float(d[j]), 3), names[a], names[c], round(float(ts[j]), 2))
info['min_ball_surface_gap'] = mind
# 5) free flight: acceleration per frame equals gravity; frame-to-frame speed ratio (fly segments only)
tf = np.arange(NF) / FPS
amax = 0.0; rat = 1.0; rat_fast = 1.0; worst_r = None
for n, b in L['balls'].items():
    for g in b['seg']:
        if g['k'] != 'fly': continue
        fr = [f for f in range(1, NF - 1) if g['t0'] + 1e-6 < tf[f - 1] and tf[f + 1] < g['t1'] - 1e-6] if g['t1'] - g['t0'] > 3 / FPS else []
        for f in fr:
            Q = [seg_at(g, tf[f + k]) for k in (-1, 0, 1)]
            a = (Q[2] - 2 * Q[1] + Q[0]) * FPS * FPS
            amax = max(amax, float(np.linalg.norm(a - np.array([0, 0, -G]))))
            s0 = np.linalg.norm(Q[1] - Q[0]) * FPS; s1 = np.linalg.norm(Q[2] - Q[1]) * FPS
            if min(s0, s1) > 0.5:
                r_ = max(s0, s1) / min(s0, s1)
                if r_ > rat: rat = r_; worst_r = (n, round(tf[f], 2), round(s0, 2), round(s1, 2))
            if min(s0, s1) > 2.0: rat_fast = max(rat_fast, max(s0, s1) / min(s0, s1))
info['flight_accel_error_max'] = round(amax, 6); info['flight_speed_ratio_max(v>0.5)'] = round(rat, 3); info['worst_ratio_at'] = worst_r
info['flight_speed_ratio_max(v>2)'] = round(rat_fast, 3)
# path segments (sled, door, slide): max frame-to-frame speed change in m/s per frame
dv = 0.0
for n, b in L['balls'].items():
    for g in b['seg']:
        if g['k'] != 'path' or g['t1'] - g['t0'] < 0.3: continue
        tt = np.arange(g['t0'] + 1 / FPS, g['t1'] - 1 / FPS, 1 / FPS)
        Q = np.array([seg_at(g, t) for t in tt]); sp = np.linalg.norm(np.diff(Q, axis=0), axis=1) * FPS
        if len(sp) > 2: dv = max(dv, float(np.abs(np.diff(sp)).max()))
info['mechanism_speed_change_per_frame_max_mps'] = round(dv, 3)
hs = [h for b in L['balls'].values() for h in b['hits'] if np.linalg.norm(h['vout']) > 0 and h['kind'] not in ('chime',)]
info['bounce_speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in hs), 4)
info['restitution_range'] = (round(min(h['e'] for h in hs), 3), round(max(h['e'] for h in hs), 3))
info['key_tilt_max_deg'] = round(max(math.degrees(math.acos(min(1, h['n'][2]))) for h in hs if h['kind'] != 'bigbell'), 1)
for k, v in info.items(): print(k, v)
