"""Checks for carmen_prelude (no Blender needed): python3 verify.py [table]
 - score check: every onset of the score (score/prelude.midi, bars 1-66) is struck at its time by the right ball(s) with
   the right pitch(es): right hand -> toreros (or the crowd in the tremolo), left hand -> bull / spectators
 - contact distance between ball and key/drum/bell on the rendered frame nearest each hit, and at the exact time
 - collisions: ball surface entering any key / drum / bell other than its own at its own hit (sampled at 240 Hz)
 - free flight: per-frame acceleration == gravity, frame-to-frame speed ratio; every bounce: e <= emax, |v_out| <= |v_in|;
   mechanisms (ramp, trebuchet): largest frame-to-frame speed change"""
import json, os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
L = json.load(open(f'{HERE}/layout.json')); LEAD = json.load(open(f'{HERE}/lead.json')); E = json.load(open(f'{HERE}/events.json'))
G = L['G']; FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.5) * FPS)
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'
B = L['balls']


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
# ---------------- 1) score check against the MIDI made from the Mutopia source ----------------
import arrange as A                                            # re-reads score/prelude.midi (also rewrites the json, same content)
hits_at = {}                                                    # (round(t,4)) -> list of (ball, pitch)
for n, b in B.items():
    for h in b['hits']:
        if h['pitch'] is None: continue
        hits_at.setdefault(round(h['t'], 4), []).append((n, h['pitch'], h['kind']))
miss = []; rh_n = lh_n = 0
for (bar, pos), ns in sorted(A.RH.items()):
    t = round(A.tb(bar, pos), 4); ps = sorted(set(p for p, d in ns)); s = A.sect(bar)
    tbar = {'A1': bar, 'A2': bar - 16, 'A3': bar - 50}.get(s)
    if tbar in A.TRILL_BARS and pos > 0: continue               # the printed turn is inside the written-out trill
    got = sorted(set(p for n, p, k in hits_at.get(t, []) if n in ('red', 'gold') or n.startswith('m')))
    rh_n += 1
    if max(ps) not in got: miss.append(('RH top missing', bar, pos, [nm(p) for p in ps], [nm(p) for p in got]))
    if any(p not in ps and not (tbar in A.TRILL_BARS) for p in got): miss.append(('RH wrong', bar, pos, [nm(p) for p in ps], [nm(p) for p in got]))
    if s == 'B3' and len(ps) > 1 and len(got) < 2: miss.append(('crowd chord', bar, pos))
for tbar_, (mn, up, turn) in A.TRILL_BARS.items():
    for base in (0, 16, 50):
        bar = tbar_ + base
        for k, p in enumerate((mn, up, mn, up, mn, up, turn, mn)):
            got = [pp for n, pp, kk in hits_at.get(round(A.tb(bar, k), 4), []) if n in ('red', 'gold')]
            if p not in got: miss.append(('trill', bar, k, nm(p), [nm(x) for x in got]))
for (bar, pos), ns in sorted(A.LH.items()):
    t = round(A.tb(bar, pos), 4); ps = sorted(set(p for p, d in ns)); lh_n += 1
    got = hits_at.get(t, [])
    bull = [p for n, p, k in got if n == 'bull']; spec = sorted(p for n, p, k in got if n.startswith('s'))
    if bull and bull[0] != ps[0]: miss.append(('bass wrong', bar, pos, nm(ps[0]), nm(bull[0])))
    if not bull and not spec: miss.append(('LH not struck', bar, pos, [nm(p) for p in ps]))
    if spec and any(p not in ps for p in spec): miss.append(('chord wrong', bar, pos, [nm(p) for p in ps], [nm(p) for p in spec]))
    if spec and len(set(spec)) == 1 and len(ps) > 1: miss.append(('chord thin', bar, pos))
info['score_rh_onsets'] = rh_n; info['score_lh_onsets'] = lh_n; info['score_mismatches'] = miss
# every struck pitch must be heard: the arrangement's notes.json at that time/pitch
notes = json.load(open(f'{HERE}/notes.json'))
heard = set((round(t, 4), p) for t, d, role, p, v in notes if role in ('key', 'drum', 'bell', 'trem'))
unheard = [(n, round(h['t'], 3), nm(h['pitch'])) for n, b in B.items() for h in b['hits']
           if h['pitch'] is not None and (round(h['t'], 4), h['pitch']) not in heard]
info['struck_but_not_in_audio'] = unheard[:5] + ([len(unheard)] if len(unheard) > 5 else [])
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    for t in sorted(hits_at):
        g = hits_at[t]
        print(f'{t:7.3f}s frame {round(t * FPS):5d}  ' + '  '.join(f'{n}:{nm(p)}' for n, p, k in sorted(g)))
# ---------------- 2) contact distance on the nearest frame / exact time ----------------
gaps = []; ex = 0.0
for n, b in B.items():
    for h in b['hits']:
        f = round(h['t'] * FPS); p = at(b, f / FPS)
        gaps.append((float(np.linalg.norm(p - np.array(h['p']))), n, h['kind'], round(h['t'], 3)))
        ex = max(ex, float(np.linalg.norm(at(b, h['t']) - np.array(h['p']))))
gaps.sort(reverse=True)
info['contact_gap_nearest_frame_max_m'] = round(gaps[0][0], 3); info['contact_gap_mean_m'] = round(float(np.mean([g[0] for g in gaps])), 3)
info['contact_gap_worst'] = gaps[:3]; info['contact_gap_exact_time_max_m'] = round(ex, 6)
# ---------------- 3) collisions with keys / drums / bells other than the ball's own hit ----------------
R = {n: b['r'] for n, b in B.items()}
pieces = []
shared = {}
for n in ('red', 'gold'):
    for h in B[n]['hits']:
        if h['kind'] == 'uni': shared.setdefault(h['ev'], []).append(h)
for ev, hs in shared.items():
    nv = np.array(hs[0]['n']); top = np.mean([h['p'] for h in hs], axis=0) - L['R']['pair'] * nv
    pieces.append(dict(ball='pair', t=hs[0]['t'], c=top - nv * 0.05, M=frame(nv), hw=L['kw'] / 2, hd=L['dyp'] + 0.22, th=0.05))
for n, b in B.items():
    for h in b['hits']:
        if h['kind'] in ('key', 'drum', 'trigger', 'bell'):
            nv = np.array(h['n']); top = np.array(h['p']) - b['r'] * nv
            hw, hd, th = {'key': (L['kw'] / 2, 0.22, 0.05), 'drum': (0.24, 0.24, 0.17), 'trigger': (0.24, 0.24, 0.17), 'bell': (0.14, 0.14, 0.14)}[h['kind']]
            pieces.append(dict(ball=n, t=h['t'], c=top - nv * th, M=frame(nv), hw=hw, hd=hd, th=th))
ts = np.arange(0.2, FINAL + 1.0, 1 / 240)
P = {n: np.array([at(b, t) for t in ts]) for n, b in B.items() if not n.startswith('m')}
bad = {}; worst = 0.0
for Bp in pieces:
    for n, Pn in P.items():
        r = R[n]
        m = (np.abs(Pn[:, 0] - Bp['c'][0]) < 0.9) & (np.abs(Pn[:, 2] - Bp['c'][2]) < 0.9) & (np.abs(Pn[:, 1] - Bp['c'][1]) < 0.9)
        own = (Bp['ball'] == n) or (Bp['ball'] == 'pair' and n in ('red', 'gold'))
        if own: m &= np.abs(ts - Bp['t']) > 0.03
        m &= (ts > Bp['t'] - 1.2) & (ts < Bp['t'] + 1.25)        # pieces exist from t-1.2 s to t+1.25 s
        if not m.any(): continue
        rel = (Pn[m] - Bp['c']) @ Bp['M']
        d = np.stack([np.maximum(0, np.abs(rel[:, 0]) - Bp['hw']), np.maximum(0, np.abs(rel[:, 1]) - Bp['hd']), np.maximum(0, np.abs(rel[:, 2]) - Bp['th'])], 1)
        pen = r - np.linalg.norm(d, axis=1); k = pen > 0.01
        if k.any():
            key = f"{n}->{Bp['ball']}"; bad.setdefault(key, []).append((round(float(ts[m][k][0]), 3), round(float(pen[k].max()), 3), 'piece_t', round(Bp['t'], 3)))
            worst = max(worst, float(pen[k].max()))
info['collisions'] = {k: len(v) for k, v in bad.items()}; info['collision_examples'] = {k: v[:3] for k, v in bad.items()}
info['collision_worst_pen_m'] = round(worst, 3)
names = list(P); mind = (9, None)
for a in range(len(names)):
    for c in range(a + 1, len(names)):
        d = np.linalg.norm(P[names[a]] - P[names[c]], axis=1) - R[names[a]] - R[names[c]]
        j = int(np.argmin(d))
        if d[j] < mind[0]: mind = (round(float(d[j]), 3), names[a], names[c], round(float(ts[j]), 2))
info['min_ball_surface_gap'] = mind
# ---------------- 4) free flight / bounces / mechanisms ----------------
tf = np.arange(NF) / FPS
amax = 0.0; rat = 1.0; worst_r = None; rat_fast = 1.0
for n, b in B.items():
    for g in b['seg']:
        if g['k'] != 'fly' or g['t1'] - g['t0'] <= 3 / FPS: continue
        for f in range(1, NF - 1):
            if not (g['t0'] + 1e-6 < tf[f - 1] and tf[f + 1] < g['t1'] - 1e-6): continue
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
dv = 0.0
for n, b in B.items():
    for g in b['seg']:
        if g['k'] != 'path' or g['t1'] - g['t0'] < 0.2: continue
        tt = np.arange(g['t0'] + 1 / FPS, g['t1'] - 1 / FPS, 1 / FPS)
        Q = np.array([seg_at(g, t) for t in tt]); sp = np.linalg.norm(np.diff(Q, axis=0), axis=1) * FPS
        if len(sp) > 2: dv = max(dv, float(np.abs(np.diff(sp)).max()))
info['mechanism_speed_change_per_frame_max_mps'] = round(dv, 3)
# speed continuity where a ball leaves a mechanism (ramp lip, trebuchet release)
tr = L['ev']['treb']; rp = L['ev']['ramp']
for nm_, t_ in (('ramp_lip', rp['t1']), ('trebuchet_release', tr['t_rel'])):
    v_before = (at(B['red'], t_ - 1e-3) - at(B['red'], t_ - 2e-3)) / 1e-3; v_after = (at(B['red'], t_ + 2e-3) - at(B['red'], t_ + 1e-3)) / 1e-3
    info[f'{nm_}_speed_before_after'] = (round(float(np.linalg.norm(v_before)), 3), round(float(np.linalg.norm(v_after)), 3))
hs = [h for n, b in B.items() for h in b['hits'] if np.linalg.norm(h['vout']) > 0 and not n.startswith('m')]
info['bounce_speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in hs), 4)
info['restitution_range'] = (round(min(h['e'] for h in hs), 3), round(max(h['e'] for h in hs), 3))
info['key_tilt_max_deg'] = round(max(math.degrees(math.acos(min(1, h['n'][2]))) for h in hs), 1)
hm = [h for n, b in B.items() for h in b['hits'] if n.startswith('m')]
info['crowd_marbles'] = len(hm); info['crowd_bounce_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in hm), 3)
for k, v in info.items(): print(k, v)
