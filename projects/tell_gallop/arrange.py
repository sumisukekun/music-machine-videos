"""William Tell Overture - finale galop (G. Rossini, 1829, public domain). Own arrangement.
Melody follows the standard 2/4 theme (trumpet call, galop A theme, A-minor episode, B-section bridge, return, coda).
Written in C (sixteenth-note units), transposed to the original E major. Quarter = 168.
Outputs notes.json (all sounds), lead.json (melody = what the train strikes), events.json."""
import json

TR = 4                       # C -> E major
Q = 0.205                    # notated quarter (= one galop step, ~146 'beats' of the real 2/4 half-bar)
S16 = Q / 4                  # sixteenth
BAR = 2 * Q
T0 = 0.6                     # silence before the fanfare
NAMES = {'C': 60, 'D': 62, 'E': 64, 'F': 65, 'G': 67, 'A': 69, 'B': 71,
         'c': 72, 'd': 74, 'e': 76, 'f': 77, 'g': 79, 'a': 81, 'b': 83}


def P(tok):
    """'G4' 'c2' '^f4' 'z4' -> (midi or None, sixteenths)"""
    acc = 0
    if tok[0] == '^': acc = 1; tok = tok[1:]
    if tok[0] == 'z': return None, int(tok[1:])
    return NAMES[tok[0]] + acc, int(tok[1:])


# ---- melody, bar by bar (2/4 = 8 sixteenths) -------------------------------------------------
FAN = [  # trumpet call (in C: G = dominant)
    'G2 G2 G4', 'G2 G2 G4', 'G2 G2 G4', 'c4 e4',
    'G2 G2 G4', 'G2 G2 G4', 'e4 g4', 'z8']
A = [  # galop theme (the bar before = pickup G2 G2)
    'G4 G2 G2', 'G4 G2 G2', 'c4 d4', 'e4 G2 G2', 'G4 G2 G2', 'c4 e2 e2', 'd4 B4', 'G4 G2 G2',
    'G4 G2 G2', 'G4 G2 G2', 'c4 d4', 'e4 c2 e2', 'g8', 'g2 f2 e2 d2', 'c4 e4', 'c4 z4']
CC = [  # C section: the running passage (first time, with 1st ending)
    'c4 c2 c2', 'c4 c4', 'e8', 'e4 d4', 'c4 B4', 'c4 A4', 'G2 A2 G2 A2', 'G2 A2 B2 c2',
    'F2 G2 F2 G2', 'F2 G2 A2 B2', 'E2 F2 E2 F2', 'E2 F2 G2 A2']
C1 = ['D2 E2 D2 E2', 'D2 E2 D2 E2', 'D4 G2 G2', 'G4 G4']
C2 = ['D4 E4', 'G2 F2 E2 D2', 'C8', 'C4 z4']
CH_CC = ['C', 'C', 'E', 'E7', 'F', 'F', 'C', 'C', 'Dm', 'G7', 'C', 'C']
AM = [  # A-minor episode
    'e4 e2 e2', 'e4 e2 e2', 'e4 a4', 'e4 a4', 'e4 a4', 'e4 d4', 'c4 B4', 'A4 e2 e2',
    'e4 e2 e2', 'e4 e2 e2', 'e4 a4', 'e4 a4', 'e4 a4', 'g4 ^f4', 'g8', 'g4 d4']
BR = [  # B-section bridge
    'd4 d2 d2', 'd4 e4', 'f4 d4', 'd4 f4', 'e4 c4', 'c4 e4', 'd4 G4', 'G4 d4',
    'd4 d2 d2', 'd4 e4', 'f4 d4', 'd4 f4', 'e4 c4', 'c4 e4', 'd4 G4', 'G4 G2 G2']
CODA = [
    'c4 z4', 'e4 f4', 'g4 z4', 'c4 d4', 'e4 z4', 'E4 F4', 'G4 z4', 'z4 B4',
    'c4 B4', 'c4 B4', 'c4 B4', 'c4 B4', 'c8', 'z4 c2 c2', 'c4 z4', 'c4 z4', 'c8', 'z4 z2 e2', 'e8', 'z4 z2 c2', 'c8']
# chords per bar (C-major names, transposed later)
CH_A = ['C', 'C', 'C', 'C', 'C', 'C', 'G7', 'G7', 'C', 'C', 'C', 'C', 'G7', 'G7', 'C', 'C']
CH_AM = ['Am', 'Am', 'Am', 'Am', 'Am', 'E7', 'Am', 'Am', 'Am', 'Am', 'Am', 'Am', 'Am', 'D7', 'G', 'G7']
CH_BR = ['G7', 'G7', 'G7', 'G7', 'C', 'C', 'G7', 'G7', 'G7', 'G7', 'G7', 'G7', 'C', 'C', 'G7', 'G7']
CH_CODA = ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'G7', 'CG', 'CG', 'CG', 'CG', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C']
CHORD = {'F': (5, [5, 9, 0]), 'Dm': (2, [2, 5, 9]), 'E': (4, [4, 8, 11]), 'C': (0, [0, 4, 7]), 'G7': (7, [7, 11, 2, 5]), 'Am': (9, [9, 0, 4]), 'E7': (4, [4, 8, 11, 2]),
         'D7': (2, [2, 6, 9, 0]), 'G': (7, [7, 11, 2])}

# sections: name, bars, chords, the last bar of the previous section carries this section's pickup
SECT = [('fan', FAN, ['C'] * 8, 'z4 G2 G2'), ('A', A, CH_A, 'c4 G2 G2'), ('A2', A, CH_A, 'c4 e2 e2'),
        ('Am', AM, CH_AM, None), ('br', BR, CH_BR, None), ('A3', A, CH_A, 'c4 z4'),
        ('C', CC + C1 + CC + C2, CH_CC + ['D7', 'D7', 'G7', 'G7'] + CH_CC + ['Dm', 'G7', 'C', 'C'], 'C4 G2 G2'),
        ('A4', A[:15], CH_A[:15], None), ('coda', CODA, CH_CODA, None)]
bars = []          # [section, bar_tokens, chord]
for name, B, C, last in SECT:
    for i, (b, c) in enumerate(zip(B, C)):
        bars.append([name, b if not (last and i == len(B) - 1) else last, c])
notes, lead, sect_t = [], [], []
cur = None
for bi, (name, toks, ch) in enumerate(bars):
    t_bar = T0 + bi * BAR
    if name != cur: sect_t.append([name, t_bar]); cur = name
    pos = 0
    for tok in toks.split():
        p, d = P(tok)
        if p is not None:
            t = t_bar + pos * S16; dur = d * S16; pm = p + TR
            v = {'fan': 100, 'A': 94, 'A2': 104, 'Am': 92, 'br': 96, 'A3': 104, 'C': 90, 'A4': 114, 'coda': 116}[name]
            if name == 'coda' and d >= 8: v = 122
            lead.append([round(t, 5), pm, v, round(dur, 5), name])
        pos += d
    assert pos == 8, (name, toks)
FINAL = lead[-1][0]
tail_bars = [b for b in range(len(bars)) if bars[b][0] == 'coda']


def N(t, d, role, p, v): notes.append([round(t, 5), round(d, 5), role, int(p), int(max(1, min(127, v)))])


# ---- lead voices ---------------------------------------------------------------------------
for t, p, v, d, name in lead:
    N(t, d, 'xylo', p, v)                                   # the struck rail-key itself
    if name in ('fan',): N(t, d * 0.9, 'tpt', p, v); N(t, d * 0.9, 'hn', p - 12, v - 10)
    elif name in ('A', 'A2', 'A3', 'A4'): N(t, d * 0.85, 'tpt', p, v - 5)
    elif name in ('Am', 'br'): N(t, d * 0.9, 'vspic', p, v - 5)
    elif name == 'C': N(t, d * 0.9, 'vspic', p + 12, v - 8); N(t, d, 'glock', p + 12, v - 35)
    if name in ('A2', 'A4'): N(t, d * 0.9, 'vspic', p + 12, v - 25); N(t, d, 'glock', p + 12, v - 30)
    if name == 'A4': N(t, d * 0.9, 'tbn', p - 12, v - 20)
    if name == 'coda': N(t, d * 0.9, 'tpt', p, v); N(t, d * 0.9, 'vspic', p + 12, v - 15); N(t, d * 0.9, 'tbn', p - 12, v - 10)

# ---- galop accompaniment -------------------------------------------------------------------
for bi, (name, toks, ch) in enumerate(bars):
    t_bar = T0 + bi * BAR
    if name == 'fan':
        if bi >= 4:   # timpani roll crescendo + snare roll
            for k in range(16): N(t_bar + k * S16 * 2, 0.15, 'timp', 40, 60 + 10 * (bi - 4) + k * 2)
        if bi in (0, 4): N(t_bar, 0.6, 'tuba', 40, 100); N(t_bar, 0.6, 'timp', 40, 110)
        if bi == 7:
            for k in range(8): N(t_bar + k * S16, 0.1, 'snare', 0, 60 + k * 7)
        continue
    if name == 'coda' and bi >= tail_bars[0] + 8:
        continue   # homophonic tutti handled below
    names = ch if ch != 'CG' else 'C'
    r, tones = CHORD[names]
    root = 40 + ((r + TR) % 12)
    if root > 47: root -= 12
    fifth = root + 7
    lo = {'A': 1.0, 'A2': 1.06, 'Am': 0.92, 'br': 0.95, 'A3': 1.04, 'C': 0.88, 'A4': 1.12, 'coda': 1.15}[name]
    for beat, bp in ((0, root), (1, fifth if ch not in ('E7', 'D7') else root + 7)):
        tb = t_bar + beat * Q
        N(tb, Q * 0.45, 'tuba', bp, 96 * lo); N(tb, Q * 0.4, 'cb', bp + 12, 92 * lo); N(tb, Q * 0.4, 'cspic', bp + 12, 80 * lo)
        N(tb, 0.2, 'bd', 0, (82 if beat == 0 else 66) * lo)
        # off-beat chord stabs: horns + violas (as violins low) + snare
        to = tb + Q / 2
        for pc in tones[:3]:
            pp = 60 + ((pc + TR) % 12)
            if pp > 69: pp -= 12
            N(to, Q * 0.35, 'hnst', pp, 86 * lo)
        N(to, 0.1, 'snare', 0, 72 * lo)
    if bi % 2 == 0 or name in ('A4', 'coda'):
        N(t_bar, 0.5, 'timp', root if root >= 40 else root + 12, 95 * lo)
# cymbal crashes at section starts
for name, t in sect_t[1:]:
    N(t, 2.0, 'crash', 0, 112 if name in ('A4', 'coda') else 92)
N(sect_t[1][1] - 2 * S16, 0.1, 'snare', 0, 110)
# coda tutti: hit on every melody note of the last 13 bars
t_tut = T0 + (tail_bars[0] + 8) * BAR
for t, p, v, d, name in lead:
    if t >= t_tut - 1e-6:
        cg = (p - TR) % 12 in (11, 2, 5, 7) and abs(p - TR - 71) < 1   # B -> dominant
        r = (7 if cg else 0) + TR
        root = 40 + r % 12
        if root > 47: root -= 12
        tones = [7, 11, 2] if cg else [0, 4, 7]
        N(t, d * 0.9, 'tuba', root, 118); N(t, d * 0.9, 'cb', root + 12, 110); N(t, 0.5, 'timp', root if root >= 40 else root + 12, 120)
        for pc in tones: N(t, d * 0.9, 'hnst', 60 + (pc + TR) % 12 - (12 if 60 + (pc + TR) % 12 > 69 else 0), 108)
        N(t, 0.2, 'bd', 0, 118)
# final chord: E major tutti + timpani roll + crash
for k in range(14): N(FINAL - 1.0 + k * 0.07, 0.1, 'timp', 40, 70 + 3 * k)
for p in (40, 52): N(FINAL, 3.0, 'tubasus', p, 120)
for p in (64, 68, 71, 76): N(FINAL, 3.0, 'hnsus', p, 110)
for p in (76, 80, 83, 88): N(FINAL, 3.0, 'vsus', p, 105)
for p in (76, 83, 88): N(FINAL, 2.6, 'tptsus', p, 112)
N(FINAL, 3, 'crash', 0, 125); N(FINAL, 0.5, 'bd', 0, 125); N(FINAL, 0.6, 'timp', 40, 125)
N(FINAL, 1.0, 'bt', 0, 100)

notes.sort(key=lambda n: n[0])
json.dump(notes, open('notes.json', 'w'))
json.dump([[t, p, v, d] for t, p, v, d, n in lead], open('lead.json', 'w'))
json.dump({'t0': T0, 'q': Q, 'bar': BAR, 'final': FINAL, 'secs': [[n, t] for n, t in sect_t],
           'beats': [round(T0 + bi * BAR + k * Q, 4) for bi in range(len(bars)) for k in range(2)]},
          open('events.json', 'w'), indent=0)
print('bars', len(bars), 'lead', len(lead), 'final', round(FINAL, 2), 'secs', [(n, round(t, 2)) for n, t in sect_t])
