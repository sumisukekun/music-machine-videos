"""Checks for moonlight_sonata (no Blender needed; pieces.json = python3 scene.py -- pieces):  python3 verify.py [table]
 - every note of notes.json (score) has a visible strike at its exact time with the same pitch(es), and every pitched
   strike is a note of the score (no extra / wrong pitches); pitch/time table with 'table'
 - contact distance between the ball and the struck surface on the rendered frame nearest each hit
 - collisions: a ball surface entering any struck piece other than at its own hit; ball-ball overlaps
 - free flight: per-frame acceleration == gravity, frame-to-frame speed ratio; bounces never gain speed
 - every struck-looking object (lotus, silver key, bowl, plate, piano key with a marble) is struck at least once"""
import json, os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(f'{HERE}/layout.json')); E = json.load(open(f'{HERE}/events.json')); NOTES = json.load(open(f'{HERE}/notes.json'))
PC = json.load(open(f'{HERE}/pieces.json'))['pieces']
G = L['G']; FPS = 30; FINAL = E['final']; NF = int((FINAL + 7.5) * FPS)
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def pn(m): return f'{NM[m % 12]}{m // 12 - 1}'


def at(seg, t):
    for g in seg:
        if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9:
            p = np.array(g['p']); u = t - g['t0']
            if g['k'] == 'rest': return p
            if g['k'] == 'lin': return p + np.array(g['v']) * u
            return p + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
    return np.array(seg[0]['p']) if t < seg[0]['t0'] else at(seg, seg[-1]['t1'])


info = {}
# ---- 1) notes <-> strikes
strikes = []          # (t, set of pitches, ball, kind)
for n, b in L['balls'].items():
    for h in b['hits']:
        if h['pitch'] is None: continue
        ps = h['pitch'] if isinstance(h['pitch'], list) else [h['pitch']]
        strikes.append((round(h['t'], 4), tuple(sorted(ps)), n, h['kind']))
score = {}
for t, d, role, p, v in NOTES: score.setdefault(round(t, 4), []).append((p, role))
struck = {}
for t, ps, n, k in strikes:
    for p in ps: struck.setdefault(round(t, 4), []).append((p, n, k))
missing = []; extra = []
for t, lst in score.items():
    have = sorted(p for p, n, k in struck.get(t, []))
    want = sorted(p for p, r in lst)
    if have != want: missing.append((t, [pn(p) for p in want], [pn(p) for p in have]))
for t in struck:
    if t not in score: extra.append((t, [pn(p) for p, n, k in struck[t]]))
info['score_notes'] = len(NOTES); info['note_onsets'] = len(score); info['strikes'] = len(strikes)
info['onsets_wrong_or_missing'] = missing; info['strikes_not_in_score'] = extra
roles = {}
for t, d, role, p, v in NOTES: roles[role] = roles.get(role, 0) + 1
info['notes_by_role'] = roles
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    for t in sorted(score):
        who = ', '.join(f'{pn(p)}:{n}/{k}' for p, n, k in sorted(struck.get(t, [])))
        print(f'{t:7.3f}s  frame {round(t * FPS):5d}  score {" ".join(pn(p) + "(" + r + ")" for p, r in sorted(score[t]))}  ->  {who}')
# ---- 2) contact distance on the frame nearest each hit (ball centre vs struck surface + R along the normal)
gaps = []; gaps0 = []
for n, b in L['balls'].items():
    for h in b['hits']:
        nn = np.array(h['n']); nn /= np.linalg.norm(nn); c = np.array(h['p']) - b['R'] * nn
        f = round(h['t'] * FPS); p = at(b['seg'], f / FPS)
        gaps.append((float(np.linalg.norm(p - (c + b['R'] * nn))), n, h['kind'], h['t']))
        gaps0.append(float(np.linalg.norm(at(b['seg'], h['t']) - np.array(h['p']))))
info['contact_gap_nearest_frame_max'] = round(max(g[0] for g in gaps), 3)
info['contact_gap_mean'] = round(float(np.mean([g[0] for g in gaps])), 4)
info['contact_gap_worst'] = [(round(g[0], 3), g[1], g[2], round(g[3], 2)) for g in sorted(gaps, reverse=True)[:3]]
info['contact_gap_exact_time_max'] = round(max(gaps0), 6)
# the struck surface of each piece = the ball contact point
pc_err = 0.0
for P in PC:
    if P['kind'] in ('pkey',): continue
    b = L['balls'][P['ball']]
    hs = [h for h in b['hits'] if abs(h['t'] - P['t']) < 1e-6]
    if not hs: continue
    h = hs[0]; nn = np.array(h['n']); c = np.array(h['p']) - b['R'] * nn / np.linalg.norm(nn)
    if P['kind'] in ('bowl', 'ssbowl', 'catbowl'): c = np.array(h['p']) - np.array([0, 0, b['R']])
    pc_err = max(pc_err, float(np.linalg.norm(c - np.array(P['top']))))
info['piece_surface_vs_contact_max'] = round(pc_err, 4)
# piano keys: marble resting on the key top at the hit
kerr = 0.0
for n, b in L['balls'].items():
    for h in b['hits']:
        if h['kind'] != 'pkey': continue
        P = [p for p in PC if p['kind'] == 'pkey' and p['pitch'] == h['pitch'] and abs(p['t'] - h['t']) < 1e-6][0]
        kerr = max(kerr, abs(h['p'][2] - b['R'] - P['top'][2]), max(0, abs(h['p'][1] - P['top'][1]) - P['hd']))
info['piano_key_contact_err'] = round(kerr, 4)
# ---- 3) collisions with pieces (box approx, sampled at 240 Hz) and ball-ball
ts = np.arange(0.0, FINAL + 7.0, 1 / 240)
POS = {n: np.array([at(b['seg'], t) for t in ts]) for n, b in L['balls'].items()}
SHOW = {n: b['show'] for n, b in L['balls'].items()}
def frame_n(n):
    n = np.array(n, float); n /= np.linalg.norm(n)
    xa = np.cross([0, 1, 0], n); xa /= np.linalg.norm(xa); ya = np.cross(n, xa); return np.stack([xa, ya, n], 1)
bad = {}; worst = 0.0
for P in PC:
    if P['kind'] in ('ssbowl', 'catbowl', 'bowl'): continue       # bowls are open: the marble is meant to sit inside
    if P['kind'] == 'sskey': continue                                # the see-saw end moves; checked via its own hit
    Mx = frame_n(P['n']); ctr = np.array(P['top']) - np.array(P['n']) * P['th'] / 2
    for n, Pn in POS.items():
        R_ = L['balls'][n]['R']
        m = (np.abs(Pn[:, 0] - ctr[0]) < 1.2) & (np.abs(Pn[:, 1] - ctr[1]) < 1.2) & (np.abs(Pn[:, 2] - ctr[2]) < 1.2) & (ts >= SHOW[n])
        own = [h['t'] for h in L['balls'][n]['hits'] if P['ball'] == n or (P['kind'] == 'pkey')]
        if P['kind'] == 'pkey':
            own = [h['t'] for h in L['balls'][n]['hits'] if h['kind'] == 'pkey' and h['pitch'] == P['pitch']]
            if own: m &= ~np.any([np.abs(ts - o) < 0.03 for o in own] + [ts > max(own) - 0.01], axis=0)
        elif P['ball'] == n: m &= np.abs(ts - P['t']) > 0.03
        if not m.any(): continue
        rel = (Pn[m] - ctr) @ Mx
        d = np.stack([np.maximum(0, np.abs(rel[:, 0]) - P['hw']), np.maximum(0, np.abs(rel[:, 1]) - P['hd']), np.maximum(0, np.abs(rel[:, 2]) - P['th'] / 2)], 1)
        pen = R_ - np.linalg.norm(d, axis=1)
        k = pen > 0.01
        if k.any():
            key = f"{n}->{P['kind']}({P['ball']})"; bad.setdefault(key, []).append((round(P['t'], 2), round(float(ts[m][k][0]), 3), round(float(pen[k].max()), 3)))
            worst = max(worst, float(pen[k].max()))
info['collisions'] = {k: len(v) for k, v in bad.items()}; info['collision_examples'] = {k: v[:3] for k, v in bad.items()}
info['collision_worst_pen'] = round(worst, 3)
names = [n for n in POS]; mind = (9.0, None)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, c = names[i], names[j]
        vis = (ts >= max(SHOW[a], SHOW[c]) + 0.05)
        if L['balls'][a]['role'] in ('bass', 'fin') and L['balls'][c]['role'] in ('bass', 'fin'):
            vis &= (ts >= max(SHOW[a], SHOW[c]) + 0.3)                 # both still inside the gondola hopper
        if not vis.any(): continue
        d = np.linalg.norm(POS[a][vis] - POS[c][vis], axis=1) - L['balls'][a]['R'] - L['balls'][c]['R']
        k = int(np.argmin(d))
        if d[k] < mind[0]: mind = (float(d[k]), a, c, round(float(ts[vis][k]), 2))
info['min_ball_surface_gap'] = (round(mind[0], 3),) + mind[1:]
# ---- 4) free flight: acceleration == gravity on every frame away from contacts; speed ratio between frames
tf = np.arange(NF) / FPS
evt = sorted([h['t'] for b in L['balls'].values() for h in b['hits']] + [g['t0'] for b in L['balls'].values() for g in b['seg']] +
             [g['t1'] for b in L['balls'].values() for g in b['seg']])
evt = np.array(evt)
amax = 0.0; rat = (1.0, None)
for n, b in L['balls'].items():
    Q = np.array([at(b['seg'], t) for t in tf])
    for f in range(1, NF - 1):
        if tf[f] < b['show']: continue
        t0, t1 = tf[f - 1], tf[f + 1]
        if np.any((evt >= t0 - 1e-6) & (evt <= t1 + 1e-6)): continue
        if np.linalg.norm(Q[f + 1] - Q[f - 1]) < 1e-9: continue
        seg = [g for g in b['seg'] if g['t0'] - 1e-9 <= tf[f] <= g['t1'] + 1e-9][0]
        if seg['k'] != 'fly': continue
        a = (Q[f + 1] - 2 * Q[f] + Q[f - 1]) * FPS * FPS
        amax = max(amax, float(np.linalg.norm(a - np.array([0, 0, -G]))))
        s0 = np.linalg.norm(Q[f] - Q[f - 1]) * FPS; s1 = np.linalg.norm(Q[f + 1] - Q[f]) * FPS
        if min(s0, s1) > 0.5 and max(s0, s1) / min(s0, s1) > rat[0]: rat = (max(s0, s1) / min(s0, s1), n, round(tf[f], 2), round(min(s0, s1), 2))
info['flight_accel_error_max'] = round(amax, 6)
info['flight_speed_ratio_max(v>0.5)'] = (round(rat[0], 3),) + rat[1:]
# bounces: free ones never gain speed; mechanisms listed separately
free = [h for b in L['balls'].values() for h in b['hits'] if not h['mech'] and np.linalg.norm(h['vout']) > 0]
info['free_bounces'] = len(free)
info['bounce_speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in free), 4)
info['restitution_range'] = (round(min(h['e'] for h in free), 3), round(max(h['e'] for h in free), 3))
info['contact_tilt_max_deg'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2] / np.linalg.norm(h['n']))))) for h in free), 1)
mech = [(n, h) for n, b in L['balls'].items() for h in b['hits'] if h['mech'] and h['kind'] in ('ss', 'ss0', 'tray')]
info['mechanism_launches'] = len(mech)
ok = all(any(abs(bb['t'] - h['t']) < 1e-6 and bb['kind'] in ('ssbowl', 'catbowl') for bb in L['ev']['bass']) for n, h in mech)
info['every_launch_has_a_bass_marble_landing_at_that_instant'] = ok
# ---- static overlap between struck pieces (footprint circles + height ranges)
FOOT = dict(pad=0.21, key=None, bowl=0.3, ssbowl=0.3, sskey=None, plate=0.24, catbowl=0.3)     # None = box (hw, hd)
def gap2(a, c):
    dx = abs(a['top'][0] - c['top'][0]); dy = abs(a['top'][1] - c['top'][1])
    ea = (FOOT[a['kind']], FOOT[a['kind']]) if FOOT[a['kind']] else (a['hw'], a['hd'])
    ec = (FOOT[c['kind']], FOOT[c['kind']]) if FOOT[c['kind']] else (c['hw'], c['hd'])
    return max(dx - ea[0] - ec[0], dy - ea[1] - ec[1])
st = [p for p in PC if p['kind'] in FOOT]
ov = []
for i in range(len(st)):
    for j in range(i + 1, len(st)):
        a, c = st[i], st[j]
        if a['kind'] in ('ssbowl', 'sskey') and c['kind'] in ('ssbowl', 'sskey') and abs(a['t'] - c['t']) < 1e-6: continue
        g = gap2(a, c)
        if g < -0.01 and abs(a['top'][2] - c['top'][2]) < 0.35: ov.append((a['kind'], round(a['t'], 2), c['kind'], round(c['t'], 2), round(g, 2)))
info['static_piece_overlaps'] = ov[:6]; info['static_piece_overlap_count'] = len(ov)
# ---- 5) every struck-looking piece is struck
info['pieces'] = len(PC); info['pieces_never_struck'] = [p['kind'] for p in PC if p['t'] is None]
info['duration_s'] = round(NF / FPS, 2)
for k, v in info.items(): print(k, v)
