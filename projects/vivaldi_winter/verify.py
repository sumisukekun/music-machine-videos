"""Checks for vivaldi_winter (no Blender needed): python3 verify.py [table]
 - score: the MIDI rendered by LilyPond from the Mutopia parts (score/winter_midi.midi) is read again independently and every
   note of the chosen parts / bars is compared with what the balls actually strike (time, pitch, which ball)
 - contact distance between ball and key on the rendered frame nearest each hit
 - collisions: a ball entering any key / drum / bud other than its own at its own hit (sampled at 240 Hz)
 - free flight: per-frame acceleration == gravity, frame-to-frame speed ratio; every bounce: e <= emax, |v_out| <= |v_in|
 - mechanisms (ropeway, zip line): frame-to-frame speed change"""
import json, os, sys, math
import numpy as np
import mido
HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(f'{HERE}/layout.json')); LEAD = json.load(open(f'{HERE}/lead.json')); E = json.load(open(f'{HERE}/events.json'))
G = L['G']; FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.5) * FPS); KL, KS = L['key_lead'], L['key_stay']
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
# 1) score check, independent of arrange.py: read the MIDI again
mf = mido.MidiFile(f'{HERE}/score/winter_midi.midi'); tpb = mf.ticks_per_beat
PART = {}
for ti, name in zip(range(1, 6), ('solo', 'vn1', 'vn2', 'vla', 'vc')):
    t = 0; on = {}; ns = []
    for m in mf.tracks[ti]:
        t += m.time
        if m.type == 'note_on' and m.velocity > 0: on[m.note] = t
        elif m.type in ('note_off', 'note_on') and m.note in on:
            s_ = on.pop(m.note); ns.append((s_ / tpb, m.note))
    PART[name] = ns
T0, Q = E['t0'], E['q']
def sel(part): return [(round(T0 + Q * b, 4), p) for b, p in PART[part]]
want = {'mel': sel('solo'), 'bas': sel('vc'), 'rain1': sel('vn1'), 'rain2': sel('vn2')}
got = {n: sorted((round(h['t'], 4), h['pitch']) for h in b['hits'] if h.get('ev') is not None) for n, b in L['balls'].items()}
for r in ('rain1', 'rain2'): got[r] = sorted((round(d['t'], 4), d['pitch']) for d in L['drops'] if d['role'] == r)
sc_bad = {}
for n in want:
    w = sorted(want[n]); g = got[n]
    if w != g: sc_bad[n] = dict(missing=sorted(set(w) - set(g))[:5], extra=sorted(set(g) - set(w))[:5])
info['score_notes_checked'] = {n: len(want[n]) for n in want}
info['score_mismatch'] = sc_bad
# every lead event exactly once
cnt = {}
for n, b in L['balls'].items():
    for h in b['hits']:
        if h.get('ev') is not None: cnt[h['ev']] = cnt.get(h['ev'], 0) + 1
for d in L['drops']: cnt[d['ev']] = cnt.get(d['ev'], 0) + 1
info['events_not_struck_once'] = [i for i in range(len(LEAD)) if cnt.get(i, 0) != 1]
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    for e in LEAD:
        h = ([(n, x) for n, b in L['balls'].items() for x in b['hits'] if x.get('ev') == e['i']] +
             [(d['role'], dict(d, kind='chime')) for d in L['drops'] if d['ev'] == e['i']])[0]
        print(f"{e['i']:3d} {e['t']:7.3f}s bar{e['bar']:2d} beat{e['beatno']:<5} {e['role']:5s} score {nm(e['pitch']):4s} struck {nm(h[1]['pitch']):4s} "
              f"by {h[0]:5s} on {h[1]['kind']:7s} frame {round(e['t'] * FPS):5d}")
# 2) contact distance on the nearest frame
gaps = []
for n, b in L['balls'].items():
    for h in b['hits']:
        f = round(h['t'] * FPS); p = at(b, f / FPS)
        gaps.append((float(np.linalg.norm(p - np.array(h['p']))), n, h['kind'], round(h['t'], 3)))
gaps.sort(reverse=True)
info['contact_gap_nearest_frame_max_m'] = round(gaps[0][0], 3); info['contact_gap_mean_m'] = round(float(np.mean([g[0] for g in gaps])), 3)
info['contact_gap_exact_time_max_m'] = round(max(float(np.linalg.norm(at(b, h['t']) - np.array(h['p']))) for b in L['balls'].values() for h in b['hits']), 6)
# 3) collisions with keys / drums / buds other than the ball's own hit (keys exist from t-KL to t+KS)
pieces = []
for n, b in L['balls'].items():
    r = b['r']; ext = L['lanes'][n]['ext']
    for h in b['hits']:
        if h['kind'] in ('key', 'drum', 'bellows'):
            nv = np.array(h['n']); top = np.array(h['p']) - r * nv
            pieces.append(dict(ball=n, t=h['t'], c=top - nv * ext[2], M=frame(nv), ext=ext))
for d in L['drops']:
    R_ = L['rain'][d['role']]; nv = np.array(d['n'])
    pieces.append(dict(ball='drop%d' % d['ev'], chime=d['chime'], t=d['t'], c=np.array(d['top']) - nv * R_['ext'][2], M=frame(nv), ext=R_['ext']))
DROP_CHIME = {'drop%d' % d['ev']: d['chime'] for d in L['drops']}; DROP_T = {'drop%d' % d['ev']: d['t'] for d in L['drops']}
ts = np.arange(0.2, FINAL + 1.0, 1 / 240)
P = {n: np.array([at(b, t) for t in ts]) for n, b in L['balls'].items()}
RADIUS = {n: b['r'] for n, b in L['balls'].items()}


def drop_at(d, t):
    if t < d['t']: u = t - d['t_rel']; return np.array(d['p_rel']) + np.array([0, 0, -0.5 * G * u * u])
    u = t - d['t']; return np.array(d['p']) + np.array(d['vout']) * u + np.array([0, 0, -0.5 * G * u * u])


def drop_alive(d, t): return (t >= d['t_rel']) & ((t <= d['t']) | (np.array([drop_at(d, x)[2] for x in np.atleast_1d(t)]) > d['ground']))
for d in L['drops']:
    tt_ = ts[(ts >= d['t_rel']) & (ts <= d['t'] + 1.5)]
    Pd = np.array([drop_at(d, x) for x in tt_]); al = (tt_ <= d['t']) | (Pd[:, 2] > d['ground'])
    Pf = np.full((len(ts), 3), 1e3); idx = np.searchsorted(ts, tt_); Pf[idx[al]] = Pd[al]
    P['drop%d' % d['ev']] = Pf; RADIUS['drop%d' % d['ev']] = L['rain'][d['role']]['r']
bad = {}; worst = 0.0
for B in pieces:
    for n, Pn in P.items():
        r = RADIUS[n]
        m = (np.abs(Pn[:, 0] - B['c'][0]) < 0.8) & (np.abs(Pn[:, 2] - B['c'][2]) < 0.8) & (np.abs(Pn[:, 1] - B['c'][1]) < 0.8)
        if B['ball'] == n: m &= np.abs(ts - B['t']) > (0.03 if not n.startswith('drop') else 0.012)
        if n.startswith('drop') and B.get('chime') is not None:
            own = DROP_CHIME[n]
            if own == B['chime']: m &= np.abs(ts - DROP_T[n]) > 0.012
        m &= (ts > B['t'] - KL) & (ts < B['t'] + KS)
        if not m.any(): continue
        rel = (Pn[m] - B['c']) @ B['M']
        d = np.stack([np.maximum(0, np.abs(rel[:, 0]) - B['ext'][0]), np.maximum(0, np.abs(rel[:, 1]) - B['ext'][1]), np.maximum(0, np.abs(rel[:, 2]) - B['ext'][2])], 1)
        pen = r - np.linalg.norm(d, axis=1); k = pen > 0.01
        if k.any():
            key = f"{n}->{B['ball']}"; bad.setdefault(key, []).append((round(float(ts[m][k][0]), 3), round(float(pen[k].max()), 3), round(B['t'], 3)))
            worst = max(worst, float(pen[k].max()))
info['collisions'] = {k: len(v) for k, v in bad.items()}; info['collision_examples'] = {k: v[:3] for k, v in bad.items()}
info['collision_worst_pen_m'] = round(worst, 3)
names = list(L['balls']); mind = (9, None)
for a in range(len(names)):
    for c in range(a + 1, len(names)):
        d = np.linalg.norm(P[names[a]] - P[names[c]], axis=1) - L['balls'][names[a]]['r'] - L['balls'][names[c]]['r']
        j = int(np.argmin(d))
        if d[j] < mind[0]: mind = (round(float(d[j]), 3), names[a], names[c], round(float(ts[j]), 2))
info['min_ball_surface_gap'] = mind
# raindrops: contact distance (exact by construction) and speed in / out
dr = [(np.linalg.norm(d['vout']) / np.linalg.norm(d['vin'])) for d in L['drops']]
info['raindrops'] = len(L['drops']); info['raindrop_speed_out_over_in_max'] = round(max(dr), 3)
gapd = max(float(np.linalg.norm(drop_at(d, d['t']) - np.array(d['p']))) for d in L['drops'])
info['raindrop_contact_gap_m'] = round(gapd, 6)
# 4) free flight
tf = np.arange(NF) / FPS
amax = 0.0; rat = 1.0; worst_r = None
for n, b in L['balls'].items():
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
# speed across a bounce, measured on frames: speed of the frame step after the hit / before the hit
sr = 1.0
for n, b in L['balls'].items():
    for h in b['hits']:
        if np.linalg.norm(h['vout']) == 0: continue
        t_ = h['t']; dd = 1 / FPS / 4
        v0 = np.linalg.norm(at(b, t_ - 1e-6) - at(b, t_ - dd)) / dd; v1 = np.linalg.norm(at(b, t_ + dd) - at(b, t_ + 1e-6)) / dd
        if v0 > 0.5: sr = max(sr, v1 / v0)
info['speed_after_over_before_bounce_sampled_max'] = round(sr, 3)
dv = 0.0
for n, b in L['balls'].items():
    for g in b['seg']:
        if g['k'] != 'path' or g['t1'] - g['t0'] < 0.3: continue
        tt = np.arange(g['t0'] + 1 / FPS, g['t1'] - 1 / FPS, 1 / FPS)
        Qp = np.array([seg_at(g, t) for t in tt]); sp = np.linalg.norm(np.diff(Qp, axis=0), axis=1) * FPS
        if len(sp) > 2: dv = max(dv, float(np.abs(np.diff(sp)).max()))
info['mechanism_speed_change_per_frame_max_mps'] = round(dv, 3)
hs = [h for b in L['balls'].values() for h in b['hits'] if np.linalg.norm(h['vout']) > 0]
info['bounce_speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in hs), 4)
info['restitution_range'] = (round(min(h['e'] for h in hs), 3), round(max(h['e'] for h in hs), 3))
info['key_tilt_max_deg'] = round(max(math.degrees(math.acos(min(1, h['n'][2]))) for h in hs), 1)
info['duration_s'] = round(NF / FPS, 2)
for k, v in info.items(): print(k, v)
