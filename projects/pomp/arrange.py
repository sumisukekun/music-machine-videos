"""Pomp and Circumstance March No.1, Trio (E. Elgar, 1901 - public domain in Japan; d.1934).
Melody checked against two independent transcriptions (organ Trio in G, lead sheet in C). Own arrangement in C major.
Structure: original 3-bar fanfare intro -> Trio bars 1-16 (p, strings) -> Trio bars 25-40 (ff, full brass, allargando)."""
import json
BEAT = 0.5
# (label, beats-duration) ; bar time map
BARS = [('I1', 2.0), ('I2', 2.0), ('I3', 2.0)] + [(f'A{b}', 2.0) for b in range(1, 17)] + \
       [(f'B{b}', 2.0) for b in range(25, 37)] + [('B37', 2.5), ('B38', 2.5), ('B39', 2.0), ('B40', 2.0)]
T0 = 1.0
BT = {}; t = T0
for lab, d in BARS: BT[lab] = (t, d); t += d
def tb(lab, beat): t0, d = BT[lab]; return t0 + beat * d / 4
def dur(lab, beats): return beats * BT[lab][1] / 4
MEL = {  # bar: [(beat, midi, beats)]
 1: [(0, 72, 4)], 2: [(0, 71, 1), (1, 72, 1), (2, 74, 2)], 3: [(0, 69, 4)], 4: [(0, 67, 4)], 5: [(0, 65, 4)],
 6: [(0, 64, 1), (1, 65, 1), (2, 67, 2)], 7: [(0, 62, 8)], 9: [(0, 64, 4)], 10: [(0, 66, 1), (1, 67, 2), (3, 69, 1)],
 11: [(0, 74, 4)], 12: [(0, 67, 4)], 13: [(0, 72, 5)], 14: [(1, 71, 2), (3, 69, 1)], 15: [(0, 67, 6)],
 25: [(0, 64, 4)], 26: [(0, 66, 1), (1, 67, 2), (3, 69, 1)], 27: [(0, 74, 4)], 28: [(0, 67, 4)], 29: [(0, 77, 4)],
 30: [(0, 77, 1), (1, 76, 2), (3, 74, 1)], 31: [(0, 76, 6)], 33: [(0, 69, 4)], 34: [(0, 71, 1), (1, 72, 2), (3, 74, 1)],
 35: [(0, 67, 4)], 36: [(0, 72, 4)], 37: [(0, 60, 4)], 38: [(0, 65, 1), (1, 64, 2), (3, 62, 1)], 39: [(0, 60, 8)]}
CH = {1: ([60, 64, 67], 48), 2: ([59, 62, 65], 50), 3: ([57, 60, 64], 52), 4: ([55, 60, 64], 52), 5: ([57, 60, 65], 53),
      6: ([55, 60, 64], 55), 7: ([54, 57, 60], 45), 8: ([55, 59, 62], 43), 9: ([55, 60, 64], 48), 10: ([54, 57, 62], 48),
      11: ([55, 59, 62], 47), 12: ([55, 59, 64], 52), 13: ([55, 57, 64], 45), 14: ([54, 57, 60], 50), 15: ([55, 59, 62], 43),
      16: ([53, 59, 62], 43), 25: ([55, 60, 64], 48), 26: ([54, 57, 62], 48), 27: ([55, 59, 62], 47), 28: ([55, 59, 64], 52),
      29: ([57, 60, 65], 45), 30: ([53, 59, 62], 43), 31: ([55, 60, 64], 48), 32: ([55, 58, 64], 48), 33: ([57, 60, 65], 53),
      34: ([55, 59, 62], 53), 35: ([55, 60, 64], 52), 36: ([55, 60, 64], 45), 37: ([53, 57, 60], 50), 38: ([53, 59, 62], 43),
      39: ([55, 60, 64], 36), 40: ([55, 60, 64], 36)}
notes = []; lead = []; fan = []
def N(t, d, role, p, v): notes.append([round(t, 4), round(d, 4), role, p, int(v)])
# ---- intro fanfare (original): drum roll, trumpets G G G C | E - G - | C
N(tb('I1', 0), 2.0, 'sroll', 0, 60); N(tb('I1', 0), 2.0, 'troll', 43, 70)
FAN = [('I1', 2, 67, 1 / 3), ('I1', 2 + 1 / 3, 67, 1 / 3), ('I1', 2 + 2 / 3, 67, 1 / 3), ('I1', 3, 72, 1), ('I2', 0, 76, 2), ('I2', 2, 79, 2), ('I3', 0, 72, 1.2)]
for lab, bt, p, bl in FAN:
    t_ = tb(lab, bt); d = dur(lab, bl); v = 105 if lab != 'I1' else 95
    N(t_, d * 0.9, 'tpt', p, v); N(t_, d * 0.9, 'tpt', p - 12 if p > 70 else p - 5, v - 15); fan.append([t_, p, d])
for lab, ch, bs in (('I2', [60, 64, 67], 36), ('I3', [60, 64, 67], 36)):
    t_ = tb(lab, 0); N(t_, 1.8, 'tbn', ch[0] - 12, 100); N(t_, 1.8, 'tbn', ch[1] - 12, 95); N(t_, 1.8, 'hrn', ch[2], 95)
    N(t_, 1.0, 'tuba', bs, 105); N(t_, 0.6, 'timp', 48, 115); N(t_, 1, 'bd', 0, 100)
N(tb('I2', 0), 2, 'crash', 0, 105); N(tb('I3', 0), 1, 'crash', 0, 90)
N(tb('I2', 2), 0.5, 'timp', 43, 95)
# ---- Trio
def bar_lab(b): return f'A{b}' if b <= 16 else f'B{b}'
for b in list(range(1, 17)) + list(range(25, 41)):
    lab = bar_lab(b); grand = b >= 25; t_ = tb(lab, 0); bd = BT[lab][1]
    k = (b - 1) / 15 if not grand else 1.0
    for bt, p, bl in MEL.get(b, []):
        tt = tb(lab, bt); d = bl * bd / 4 if bl <= 4 else (4 - bt) * bd / 4 + (bl - (4 - bt)) * BT[bar_lab(b + 1)][1] / 4
        if not grand:
            v = 62 + 30 * k; lead.append([tt, p, int(v + 10)])
            N(tt, d, 'bell', p, v + 10); N(tt, d * 0.97, 'vln', p, v); N(tt, d * 0.97, 'vln', p + 12, v - 25)
        else:
            v = 112; lead.append([tt, p, 120])
            N(tt, d, 'bell', p, 118); N(tt, d * 0.95, 'tpt', p if p >= 60 else p + 12, v); N(tt, d * 0.95, 'hrn', p - 12, v - 8)
            N(tt, d * 0.97, 'vln', p + 12, v - 5); N(tt, d * 0.97, 'vln', p, v - 15)
    ch, bs = CH[b]
    if not grand:
        v = 48 + 32 * k
        for p in ch: N(t_, bd * 0.98, 'hrn', p, v)
        N(t_, bd * 0.98, 'cello', bs, v + 5); N(t_, 0.4, 'cbp', bs - 12, v + 15); N(tb(lab, 2), 0.4, 'cbp', bs - 12 + 7 if b in (8, 15, 16) else bs - 12, v + 5)
        if b in (1, 5, 9, 13): N(t_, 0.5, 'timp', 48 if b != 13 else 45, 55 + 30 * k)
        if b == 16:
            N(tb(lab, 1), dur(lab, 3), 'sroll', 0, 85); N(tb(lab, 2), dur(lab, 2), 'troll', 43, 90)
            for i, p in enumerate((55, 59, 62, 65)): N(tb(lab, i), dur(lab, 1), 'tpt', p + 12, 70 + 10 * i)
    else:
        last = b >= 39
        vv = 108
        if b == 32:
            for p in (55, 60, 64): N(t_, bd / 2, 'tbn', p - 12, vv)
            for p in (55, 58, 64): N(tb(lab, 2), bd / 2, 'tbn', p - 12, vv)
            N(tb(lab, 0), bd / 2, 'hrn', 71, 100); N(tb(lab, 2), bd / 2, 'hrn', 70, 100)
            N(tb(lab, 2), dur(lab, 2), 'sroll', 0, 95)
        elif not last:
            for p in ch: N(t_, bd * 0.98, 'tbn', p - 12, vv); N(t_, bd * 0.98, 'vsus', p + 12, 80)
        N(t_, bd * 0.98 if not last else 4.0, 'tuba', bs if bs >= 40 else bs, 112); N(t_, bd * 0.98 if not last else 4.0, 'cello', bs if bs >= 43 else bs + 12, 100)
        N(t_, 0.4, 'cbp', bs - 12 if bs >= 40 else bs, 115); N(tb(lab, 2), 0.4, 'cbp', bs - 12 + 7 if bs >= 40 else bs + 7, 100)
        if not last:
            N(t_, 0.5, 'timp', 48 if bs % 12 == 0 else 43, 118); N(tb(lab, 2), 0.5, 'timp', 43, 100); N(t_, 0.6, 'bd', 0, 105)
        if b in (25, 29, 33): N(t_, 2, 'crash', 0, 110)
        if b == 38: N(tb(lab, 2), dur(lab, 2), 'sroll', 0, 105); N(tb(lab, 2), dur(lab, 2), 'troll', 43, 110)
        if b == 39:
            for p in (48, 55, 60, 64, 67, 72): N(t_, 4.0, 'tbn' if p < 60 else 'hrn', p, 118)
            for p in (72, 76, 79, 84): N(t_, 4.0, 'vsus', p, 105)
            for p in (67, 72, 76): N(t_, 4.0, 'tpt', p, 112)
            N(t_, 1.2, 'timp', 36 + 12, 125); N(t_, 4.0, 'troll', 48, 100); N(t_, 2.5, 'crash', 0, 120); N(t_, 1, 'bd', 0, 120)
            N(t_, 1, 'bt', 0, 100); N(t_, 4.0, 'pbass', 36, 110); N(t_, 4.0, 'pbass', 24, 110)
FINAL = tb('B39', 0)
notes.sort(key=lambda n: n[0]); lead.sort()
SECS = [['intro', tb('I1', 0), tb('A1', 0)], ['A', tb('A1', 0), tb('B25', 0)], ['B', tb('B25', 0), FINAL], ['final', FINAL, FINAL + 4.5]]
# trumpeters: blow during fanfare notes, raised through the whole grand section
blow = [[f[0], f[2]] for f in fan] + [[tb('A16', i), dur('A16', 1)] for i in range(4)]
json.dump(notes, open('notes.json', 'w')); json.dump(lead, open('lead.json', 'w'))
json.dump({'t0': T0, 'final': FINAL, 'secs': SECS, 'fan': fan, 'blow': blow, 'grand': [tb('B25', 0) - 0.6, FINAL + 3.5],
           'bars': {k: v for k, v in BT.items()}, 'crash': [n[0] for n in notes if n[2] == 'crash'],
           'timp': [n[0] for n in notes if n[2] == 'timp'], 'bass': [n[0] for n in notes if n[2] == 'cbp']}, open('events.json', 'w'))
print('notes', len(notes), 'lead', len(lead), 'final', FINAL, 'end', FINAL + 4.5)
