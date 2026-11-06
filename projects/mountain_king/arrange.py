"""In the Hall of the Mountain King (E. Grieg, Peer Gynt op.23, 1875 - public domain) - own arrangement.
B minor, 4/4, accelerando 92 -> 212 bpm over three AABA cycles (48 bars) + coda.
Lead = three marbles on a triple-helix crystal track: lane0 marimba (cycle 1-3), lane1 xylophone (+8va, cycle 2-3),
lane2 glockenspiel (+15ma, cycle 3). Drum-stones / timpani are struck by swinging stone hammers."""
import json, math
# ---- the theme (beat offsets in quarter notes, MIDI pitch, duration in beats) ----
def bar(*ev): return list(ev)
A = [  # B minor phrase
    [(0, 59, .5), (.5, 61, .5), (1, 62, .5), (1.5, 64, .5), (2, 66, .5), (2.5, 62, .5), (3, 66, 1)],
    [(0, 65, .5), (.5, 61, .5), (1, 65, 1), (2, 64, .5), (2.5, 60, .5), (3, 64, 1)],
    [(0, 59, .5), (.5, 61, .5), (1, 62, .5), (1.5, 64, .5), (2, 66, .5), (2.5, 62, .5), (3, 66, .5), (3.5, 71, .5)],
    [(0, 69, .5), (.5, 66, .5), (1, 62, .5), (1.5, 66, .5), (2, 69, 2)],
]
B = [  # F# major phrase (up a fifth)
    [(0, 66, .5), (.5, 68, .5), (1, 70, .5), (1.5, 71, .5), (2, 73, .5), (2.5, 70, .5), (3, 73, 1)],
    [(0, 74, .5), (.5, 70, .5), (1, 74, 1), (2, 73, .5), (2.5, 70, .5), (3, 73, 1)],
    [(0, 66, .5), (.5, 68, .5), (1, 70, .5), (1.5, 71, .5), (2, 73, .5), (2.5, 70, .5), (3, 73, 1)],
    [(0, 74, .5), (.5, 70, .5), (1, 74, 1), (2, 73, 2)],
]
PLAN = [(0, 'A'), (0, 'A'), (0, 'B'), (0, 'A'), (1, 'A'), (1, 'A'), (1, 'B'), (1, 'A'), (2, 'A'), (2, 'B'), (2, 'A')]
FORM, META = [], []              # META: (cycle, section, bar-in-phrase, phrase index)
for pi, (c, sname) in enumerate(PLAN):
    for j, ev in enumerate(A if sname == 'A' else B): FORM.append(ev); META.append((c, sname, j, pi))
NB = len(FORM); NC = 4           # 44 bars + 4 coda bars
T0 = 1.6                         # gate opens / first note
BPM = [118 * (250 / 118) ** (b / (NB - 1)) for b in range(NB)] + [250] * NC
bar_t = [T0]
for b in range(NB + NC): bar_t.append(bar_t[-1] + 4 * 60 / BPM[b])
def T(b, beat):   # time of (bar, beat)
    return bar_t[b] + beat * 60 / BPM[b]
notes, lead = [], []      # notes: [t, dur, role, midi, vel] ; lead: [t, lane, midi, vel, gbeat]
hits = []                 # mechanical percussion: [t, stone, role, midi, vel]
def N(t, d, role, p, v): notes.append([round(t, 4), round(d, 4), role, p, int(min(127, v))])
def LEAD(b, beat, lane, p, v, d):
    t = T(b, beat); lead.append([round(t, 4), lane, p, int(min(127, v)), b * 4 + beat])
    N(t, d, ['mar', 'xylo', 'glock'][lane], p, v)
def HIT(b, beat, stone, role, p, v):
    t = T(b, beat); hits.append([round(t, 4), stone, role, p, int(min(127, v))]); N(t, 0.5, role, p, v)
for b, ev in enumerate(FORM):
    cyc, sec, j, pi = META[b]; prog = b / (NB - 1)
    k = j  # bar within phrase
    v = 46 + 70 * prog
    sp = 60 / BPM[b]
    root = 54 if sec == 'B' else 47           # F#3 / B2 pedal
    for beat, p, d in ev:
        LEAD(b, beat, 0, p, v, d * sp)
        if cyc >= 1: LEAD(b, beat, 1, p + 12, v - 6, d * sp)
        if cyc >= 2: LEAD(b, beat, 2, p + 24, v - 10, d * sp)
    # pizzicato pedal: root on 1 & 3, fifth on 2 & 4 (Grieg's bassoons/pizz)
    for i in range(4):
        q = root if i % 2 == 0 else root + 7
        N(T(b, i), 0.4 * sp, 'cpizz', q, v - 4)
        if cyc >= 1: N(T(b, i), 0.4 * sp, 'cb', q - 12, v)
        if cyc >= 2: N(T(b, i), 0.5 * sp, 'tuba', q - 12, v - 10)
    # drum stones (struck by hammers): stone 0 = bass drum, 1/2 = timpani B / F#
    if cyc == 0:
        if k % 2 == 0: HIT(b, 0, 0, 'bd', 0, 40 + 40 * prog)
    else:
        HIT(b, 0, 0, 'bd', 0, v)
        HIT(b, 2, 1 if sec == 'A' else 2, 'timp', 47 if sec == 'A' else 54, v - 5)
        if cyc == 2:
            HIT(b, 1, 2 if sec == 'A' else 1, 'timp', 54 if sec == 'A' else 47, v - 15)
            HIT(b, 3, 2 if sec == 'A' else 1, 'timp', 54 if sec == 'A' else 47, v - 15)
    # stalactite chimes at phrase starts (struck by a pendulum)
    if j == 0 and (sec == 'B' or PLAN[pi - 1][0] != cyc if pi else True): HIT(b, 0, 3, 'chime', 0, 70 + 40 * prog)
    if cyc == 2 and j == 0: HIT(b, 0, 4, 'crash', 0, 95 + 20 * (sec == 'B'))
# ---- coda: hammered B-minor blocks, then the final chord ----
cb = NB
CH = [(71, 74, 78), (71, 74, 78), (73, 76, 79), (71, 74, 78)]   # Bm Bm Em(add) Bm  (tops played by lanes)
for i in range(3):      # three bars of hammered quarter-note chords
    b = cb + i
    for beat in range(4):
        tops = CH[beat if i < 2 else (beat % 2) * 2]
        for lane in range(3 if i < 2 else 0): LEAD(b, beat, lane, tops[lane] - 12 + 12 * lane if lane else tops[0] - 12, 120, 0.5 * 60 / BPM[b])
        N(T(b, beat), 0.3, 'cspic', 47, 120); N(T(b, beat), 0.3, 'tuba', 35, 120); N(T(b, beat), 0.4, 'cb', 35, 120)
        HIT(b, beat, 0 if beat % 2 == 0 else (1 if beat == 1 else 2), 'bd' if beat % 2 == 0 else 'timp', 47 if beat == 1 else 54, 120)
    HIT(b, 0, 4, 'crash', 0, 112)
fb = cb + 3; tf = T(fb, 0)        # final chord (marbles drop into the giant crystal)
for p in (35, 47, 54, 59, 62, 66, 71, 74, 78, 83): N(tf, 3.8, 'final', p, 118)
N(tf, 3.8, 'troll', 47, 120); N(tf, 1, 'crash', 0, 125); N(tf, 1, 'bd', 0, 125); N(tf, 2, 'chime', 0, 120)
notes.sort(key=lambda n: n[0]); lead.sort(); hits.sort()
json.dump(notes, open('notes.json', 'w')); json.dump(lead, open('lead.json', 'w'))
gb = []   # global beat -> time map (for the marble's distance along the track)
for b in range(NB + NC):
    for i in range(4): gb.append([b * 4 + i, round(T(b, i), 4)])
gb.append([(NB + NC) * 4, round(bar_t[-1], 4)])
json.dump({'t0': T0, 'final': tf, 'bars': bar_t, 'bpm': BPM, 'beats': gb, 'hits': hits, 'nb': NB, 'coda': cb,
           'cycles': [T(META.index(next(m for m in META if m[0] == c)), 0) for c in range(3)]}, open('events.json', 'w'))
print(len(notes), len(lead), 'final', round(tf, 2), 'end', round(tf + 4.2, 2))
