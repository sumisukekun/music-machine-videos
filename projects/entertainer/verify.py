"""Checks for entertainer (no Blender needed): python3 verify.py [table]
 - notes.json == the score excerpt re-read independently from the LilyPond MIDI (pitch, onset, duration)
 - every right-hand onset and every left-hand eighth is struck by a ball at its exact time
 - contact distance between ball and bell / rail / pad on the rendered frame nearest each strike
 - collisions: a ball entering any bell dome or chime rail other than its own strike (sampled at 240 Hz)
 - free flight: per-frame acceleration == gravity, frame-to-frame speed ratio; bounce restitution / speed out over in"""
import json, os, sys, math
import numpy as np
import mido
HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(f'{HERE}/layout.json')); LEAD = json.load(open(f'{HERE}/lead.json')); BASS = json.load(open(f'{HERE}/bass.json'))
E = json.load(open(f'{HERE}/events.json')); NOTES = json.load(open(f'{HERE}/notes.json'))
G = L['G']; R = L['R']; FPS = 30; FINAL = E['final']; NF = int((FINAL + 4.5) * FPS); T0 = E['t0']; BEAT = E['beat']
RB = 0.1
NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
nm = lambda p: f'{NAMES[p % 12]}{p // 12 - 1}'
info = {}
# 0) the score: re-read the MIDI and compare
mf = mido.MidiFile(f'{HERE}/score/entertainer_midi.midi'); tpb = mf.ticks_per_beat
SEL = [0, 1, 2, 3] + list(range(20, 36)) + list(range(52, 68)) + list(range(68, 84))
ref = []
for tr in mf.tracks[1:]:
    t = 0; on = {}
    for msg in tr:
        t += msg.time
        if msg.type == 'note_on' and msg.velocity > 0: on[msg.note] = t
        elif msg.type in ('note_off', 'note_on') and msg.note in on:
            s = on.pop(msg.note) / tpb; m = int(s // 2)
            if m in SEL: ref.append((round(T0 + (2 * SEL.index(m) + s - 2 * m) * BEAT, 4), msg.note, round((t / tpb - s) * BEAT, 4)))
mine = sorted((round(n[0], 4), n[2], round(n[1], 4)) for n in NOTES)
ref = sorted(ref)
info['score_notes'] = len(ref)
info['notes_json_matches_score'] = len(ref) == len(mine) and all(a[1] == b[1] and abs(a[0] - b[0]) < 1e-3 and abs(a[2] - b[2]) < 1e-3 for a, b in zip(ref, mine))
# melody of the first A bars as names (should read D D# | E C E C E C~ | C C D D# E C D E~ | E B D C ...)
a0 = [l for l in LEAD if 7.4 <= (l[0] - T0) / BEAT < 16]
info['A_first_bars_top_notes'] = ' '.join(nm(l[1]) for l in a0)
# 1) coverage + exact times
hits = [(n, h) for n, b in L['balls'].items() for h in b['hits']]
lt = {}; bt = {}
for n, h in hits:
    if 'li' in h: lt.setdefault(h['li'], []).append((n, h['t'], h['kind']))
    if 'bi' in h: bt.setdefault(h['bi'], []).append((n, h['t'], h['kind']))
info['lead_onsets'] = len(LEAD); info['lead_without_strike'] = [i for i in range(len(LEAD)) if i not in lt]
info['bass_onsets'] = len(BASS); info['bass_without_strike'] = [k for k in range(len(BASS)) if k not in bt]
info['strike_time_error_max_s'] = max([abs(t - LEAD[i][0]) for i, v in lt.items() for _, t, _ in v] +
                                      [abs(t - BASS[k][0]) for k, v in bt.items() for _, t, _ in v])
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    rows = [(l[0], 'RH', l[1], l[4], ','.join(f'{n}:{k}' for n, _, k in lt[i])) for i, l in enumerate(LEAD)]
    rows += [(b[0], 'LH', b[1], b[2], ','.join(f'{n}:{k}' for n, _, k in bt[k])) for k, b in enumerate(BASS)]
    for t, hand, top, low, who in sorted(rows):
        bb = (t - T0) / BEAT
        print(f'{t:7.3f}s bar {int(bb // 2) + 1:2d} beat {bb % 2 + 1:5.2f} {hand} {nm(top):4s}{"(" + nm(low) + ")" if low != top else "":7s} frame {round(t * FPS):5d}  {who}')


def at(seg, t):
    for g in seg:
        if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9: break
    else: g = seg[0] if t < seg[0]['t0'] else seg[-1]; t = min(max(t, g['t0']), g['t1'])
    p = np.array(g['p'])
    if g['k'] == 'rest': return p
    u = t - g['t0']
    if g['k'] == 'lin': return p + (np.array(g['q']) - p) * (u / max(1e-9, g['t1'] - g['t0']))
    return p + np.array(g['v']) * u + 0.5 * np.array(g['a']) * u * u


# 2) contact distance on the nearest frame
gb, gr = [], []; GBK = []
DOMES = []; RAILS = []
for n, h in hits:
    p = np.array(h['p']); nv = np.array(h['n']); nv /= np.linalg.norm(nv)
    if h['kind'] in ('bell', 'cbell', 'juke'):
        c = p - (R + RB) * nv; DOMES.append((n, h['t'], c))
        q = at(L['balls'][n]['seg'], round(h['t'] * FPS) / FPS); gb.append(abs(np.linalg.norm(q - c) - (R + RB)))
    if h['kind'] == 'rail':
        RAILS.append((n, h['t'], p[0] - 0.12, p[0] + h['roll'], p[1], p[2] - R))
        q = at(L['balls'][n]['seg'], round(h['t'] * FPS) / FPS); gr.append(abs(q[2] - R - (p[2] - R)))
    if h['kind'] in ('saw', 'cat', 'chacup', 'sawend'):
        q = at(L['balls'][n]['seg'], round(h['t'] * FPS) / FPS); gb.append(np.linalg.norm(q - p))
    if h['kind'] in ('bell', 'cbell', 'juke', 'saw', 'cat', 'chacup'):
        GBK.append((gb[-1], n, h['kind'], round(h['t'], 3), round(float(np.linalg.norm(h['vin'])), 2)))
info['contact_gap_nearest_frame_max_m'] = round(float(max(gb)), 3); info['contact_gap_mean_m'] = round(float(np.mean(gb)), 4)
info['rail_gap_nearest_frame_max_m'] = round(float(max(gr)), 3)
info['contact_gap_p95_m'] = round(float(np.percentile(gb, 95)), 3)
info['contact_gap_worst'] = [(round(g, 3), n, k, t, v) for g, n, k, t, v in sorted(GBK, reverse=True)[:3]]
# 3) collisions with bell domes and chime rails (240 Hz)
ts = np.arange(0.0, FINAL + 1.5, 1 / 240)
PB = {n: np.array([at(b['seg'], t) for t in ts]) for n, b in L['balls'].items()}
bad = {}; worst = 0.0
for owner, th, c in DOMES:
    for n, Pn in PB.items():
        m = np.linalg.norm(Pn - c, axis=1) < R + RB - 0.01
        if owner == n: m &= np.abs(ts - th) > 0.03
        if m.any():
            pen = R + RB - np.linalg.norm(Pn[m] - c, axis=1).min(); worst = max(worst, pen)
            bad.setdefault(f'{n}->bell_of_{owner}', []).append((round(float(ts[m][0]), 3), round(float(pen), 3)))
for owner, th, x0, x1, y, zt in RAILS:
    for n, Pn in PB.items():
        dx = np.maximum(0, np.maximum(x0 - Pn[:, 0], Pn[:, 0] - x1)); dy = np.maximum(0, np.abs(Pn[:, 1] - y) - 0.06)
        dz = np.maximum(0, np.maximum((zt - 0.05) - Pn[:, 2], Pn[:, 2] - zt))
        d = np.sqrt(dx * dx + dy * dy + dz * dz); m = d < R - 0.01
        if owner == n: m &= ~((ts >= th - 0.03) & (ts <= th + (x1 - x0) / 1.5 + 0.1))
        if m.any():
            pen = R - d[m].min(); worst = max(worst, pen)
            bad.setdefault(f'{n}->rail_of_{owner}', []).append((round(float(ts[m][0]), 3), round(float(pen), 3)))
info['collisions'] = {k: len(v) for k, v in bad.items()}; info['collision_examples'] = {k: v[:3] for k, v in bad.items()}
info['collision_worst_pen_m'] = round(worst, 3)
names = list(PB); mind = 9
for a in range(len(names)):
    for b in range(a + 1, len(names)):
        d = np.linalg.norm(PB[names[a]] - PB[names[b]], axis=1); mind = min(mind, float(d[(ts > 0.5)].min()))
info['min_ball_ball_distance_m'] = round(mind, 3)
# 4) free flight: acceleration per frame == gravity (or 0 while rolling), speed ratio
tf = np.arange(NF) / FPS
amax = 0.0; rat = 1.0; rat_fast = 1.0
for n, b in L['balls'].items():
    ev = sorted(set([g['t0'] for g in b['seg']] + [g['t1'] for g in b['seg']]))
    Q = np.array([at(b['seg'], t) for t in tf])
    for f in range(1, NF - 1):
        t0_, t1_ = tf[f - 1], tf[f + 1]
        k = np.searchsorted(ev, t0_ - 1e-6)
        if k < len(ev) and ev[k] <= t1_ + 1e-6: continue
        g = [s for s in b['seg'] if s['t0'] <= tf[f] <= s['t1']]
        if not g or g[0]['k'] in ('rest', 'lin'): continue
        acc = (Q[f + 1] - 2 * Q[f] + Q[f - 1]) * FPS * FPS
        amax = max(amax, float(np.linalg.norm(acc - np.array(g[0]['a']))))
        if g[0]['k'] == 'fly' and abs(g[0]['a'][2] + G) > 1e-9: amax = 99
        s0 = np.linalg.norm(Q[f] - Q[f - 1]) * FPS; s1 = np.linalg.norm(Q[f + 1] - Q[f]) * FPS
        if min(s0, s1) > 0.5: rat = max(rat, max(s0, s1) / min(s0, s1))
        if min(s0, s1) > 2.0: rat_fast = max(rat_fast, max(s0, s1) / min(s0, s1))
info['flight_accel_error_max'] = round(amax, 6)
info['flight_speed_ratio_max(v>0.5)'] = round(rat, 3); info['flight_speed_ratio_max(v>2)'] = round(rat_fast, 3)
bo = [h for n, h in hits if np.linalg.norm(h['vout']) > 0 and h['kind'] not in ('mug', 'lever', 'rail')]
info['bounce_speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for h in bo), 3)
info['restitution_range'] = (round(min(h['e'] for h in bo), 3), round(max(h['e'] for h in bo), 3))
info['tilt_max_deg'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2] / np.linalg.norm(h['n']))))) for h in bo), 1)
rl = [h for n, h in hits if h['kind'] == 'rail']
info['rail_landings_keep_horizontal_speed'] = all(abs(h['vout'][0] - h['vin'][0]) < 1e-9 and h['vout'][2] == 0 for h in rl)
for k, v in info.items(): print(k, v)
