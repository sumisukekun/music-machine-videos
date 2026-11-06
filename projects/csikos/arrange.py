"""Csikos Post (H. Necke, c.1900, public domain) - own arrangement for a marble horse-race.
Melody/harmony read from src.abc (simplified piano transcription). 2/4, quarter = 168.
Lead (top note of the right hand) = the xylophone hurdles the marbles jump."""
import json
D = json.load(open('parsed.json'))
V1, V2 = D['1']['bars'], D['2']['bars']
Q = 60 / 168; BAR = 2 * Q; T0 = 1.0
FORM = ([('intro', b) for b in range(0, 4)] + [('A', b) for b in range(4, 20)] + [('B', b) for b in range(20, 36)]
        + [('B2', b) for b in range(20, 36)] + [('trans', b) for b in range(36, 40)] + [('C', b) for b in range(40, 56)]
        + [('A2', b) for b in range(4, 18)] + [('coda', 72), ('coda', 73)])
notes = []; lead = []; perc = []
def dyn(sec, k, b):
    if sec == 'intro': return 78 + 8 * k
    if sec in ('A', 'A2'): return 74
    if sec in ('B', 'B2'): return 108
    if sec == 'trans': return 95 - 5 * k
    if sec == 'C': return 92 if (b - 40) % 8 in (4, 5, 6, 7) else 72
    if sec == 'coda': return 120
secstart = {}
for k, (sec, b) in enumerate(FORM):
    tb = T0 + k * BAR; secstart.setdefault(sec, tb)
    kk = k - [i for i, (s, _) in enumerate(FORM) if s == sec][0]
    v0 = dyn(sec, kk, b)
    for i, (t, d, ps) in enumerate(V1[b]['notes']):
        tt = tb + t * Q; top = max(ps); v = v0
        if sec in ('A', 'A2') and t == 0 and (b - 4) % 4 == 0: v = 108   # sfz downbeats of each phrase
        lead.append([tt, top, v])
        notes.append([tt, max(0.15, d * Q), 'xylo', top, v])
        if sec in ('B', 'B2', 'coda'): notes.append([tt, max(0.12, d * Q * 0.8), 'tpt', top - 12 if top > 84 else top, v - 10])
        if sec in ('C',) and v > 80: notes.append([tt, d * Q, 'glock', top, v - 25])
        for p in ps:
            if p != top: notes.append([tt, d * Q, 'rh', p, v - 22])
    for t, d, ps in V2[b]['notes']:
        tt = tb + t * Q; lo = min(ps)
        onbeat = abs(t - round(t)) < 1e-6
        if len(ps) == 1 or (len(ps) == 2 and abs(ps[0] - ps[1]) == 12):
            notes.append([tt, max(0.2, d * Q), 'bass', lo, v0 - 5])
            if sec in ('B', 'B2', 'coda'): notes.append([tt, 0.2, 'tuba', lo, v0 - 15])
            if onbeat and t == 0: perc.append([tt, 'bd', v0 - 15 if sec not in ('B', 'B2', 'coda') else v0])
        else:
            for p in ps: notes.append([tt, max(0.15, d * Q * 0.7), 'lh', p, v0 - 25])
            if sec in ('B', 'B2'): 
                for p in ps: notes.append([tt, 0.18, 'horn', p, v0 - 30])
    # snare on the off-beats in the loud parts, light in C
    if sec in ('B', 'B2'):
        for e in (0.5, 1.5): perc.append([tb + e * Q, 'sn', 95])
        perc.append([tb + Q, 'bd', 95])
    if sec == 'C':
        for e in (0.5, 1.5): perc.append([tb + e * Q, 'sn', 55])
    if sec == 'intro' and kk >= 2:
        perc.append([tb, 'roll', 70 + 15 * kk])
    if sec in ('B', 'B2') and b in (20, 28): perc.append([tb, 'crash', 105])
FINAL = T0 + (len(FORM) - 1) * BAR
perc += [[FINAL, 'crash', 120], [FINAL, 'bd', 122], [FINAL, 'timp', 120], [secstart['A'], 'gate', 100]]
for p in (40, 52): notes.append([FINAL, 3.5, 'bass', p, 115])
for p in (64, 67, 71, 76): notes.append([FINAL, 3.0, 'lh', p, 100])
notes.sort(key=lambda n: n[0]); lead.sort(); perc.sort()
secs = []
for i, (s, b) in enumerate(FORM):
    if i == 0 or FORM[i - 1][0] != s: secs.append([s, T0 + i * BAR])
json.dump(notes, open('notes.json', 'w')); json.dump(lead, open('lead.json', 'w'))
json.dump({'t0': T0, 'bar': BAR, 'q': Q, 'final': FINAL, 'secs': secs, 'perc': perc, 'nbars': len(FORM)}, open('events.json', 'w'))
import numpy as np
lt = np.array([l[0] for l in lead]); print(len(notes), len(lead), round(FINAL, 2), 'min gap', round(np.diff(lt).min(), 3), secs)
