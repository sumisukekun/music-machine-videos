"""Radetzky March, Op. 228 (Johann Strauss I, 1848, public domain). Own arrangement of the march (without the trio).
Melody checked against two independent lead sheets (Wikifonia lead sheet in D, PDMX/MuseScore CC0 transcription):
intro (4 bars, tutti unison) | A theme 1st time (8 bars with 1st ending) | A theme repeat (2nd ending) | B section (16 bars)
| A theme return (with repeat, short 2nd ending to the final D).  Key D major, alla breve, half note = 108.
Melody bars 5-43 are written an octave lower than they sound (as in the lead sheet; the PDMX transcription has the real
octave: theme on F#5, intro on D5): +12 on output, the intro stays as written.
Outputs notes.json (all sounds), lead.json (melody = what the balls strike), events.json (bars, beats, sections)."""
import json

HALF = 60 / 108; Q = HALF / 2; E8 = Q / 2; BAR = 2 * HALF
T0 = 1.95
PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def P(tok):
    """'F#4:2' (pitch:eighths), 'r:1' rest. E#4 = F4."""
    n, d = tok.split(':'); d = int(d)
    if n == 'r': return None, d
    o = int(n[-1]); s = n[:-1]; p = PC[s[0]] + (1 if '#' in s else 0) - (1 if s.endswith('b') else 0)
    return 12 * (o + 1) + p, d


M = {
    1: 'D5:1 D5:1 D5:1 r:1 D5:1 D5:1 D5:1 r:1', 2: 'F#5:2 E5:2 D5:3 C#5:1', 3: 'B4:1 A#4:1 B4:1 C#5:1 D5:2 E5:2',
    4: 'A4:2 r:2 r:2 F#4:1 E#4:1',
    5: 'F#4:2 F#4:1 E#4:1 F#4:2 F#4:1 E#4:1', 6: 'F#4:1 r:1 E4:1 r:1 D4:1 r:1 F#4:1 E#4:1',
    7: 'F#4:2 F#4:1 E#4:1 F#4:2 F#4:1 E#4:1', 8: 'F#4:1 r:1 B4:1 r:1 A4:1 r:1 A4:1 F#4:1',
    9: 'G4:1 r:1 B4:1 r:1 A4:1 r:1 A4:1 E4:1', 10: 'F#4:1 r:1 B4:1 r:1 A4:1 r:1 A4:1 F#4:1',
    11: 'E4:1 r:1 C#5:2 D4:1 r:1 B4:2', 12: 'A4:1 r:1 A4:1 G#4:1 A4:1 B4:1 A4:1 G4:1',
    13: 'G#4:1 r:1 F#5:1 r:1 E5:4', 14: 'F#4:1 r:1 E5:1 r:1 D5:4', 15: 'C#5:3 B4:1 A4:1 G4:1 F#4:1 E4:1',
    16: 'D4:2 D4:2 D4:2 D5:1 C#5:1',
    17: 'D5:1 r:1 D5:1 C#5:1 D5:1 r:1 D5:1 C#5:1', 18: 'D5:1 r:1 G5:1 r:1 B4:1 r:1 D5:1 C#5:1',
    19: 'D5:1 r:1 D5:1 C#5:1 D5:1 r:1 D5:1 C#5:1', 20: 'D5:1 r:1 F#5:1 r:1 A4:1 r:1 A4:1 r:1',
    21: 'A4:4 G4:1 r:1 B4:1 r:1', 22: 'B4:4 A4:1 r:1 D5:1 r:1', 23: 'C#5:1 r:1 C#5:1 B4:1 A4:1 r:1 G4:1 r:1',
    24: 'B4:1 r:1 B4:1 A4:1 G4:1 r:1 F#4:1 r:1', 25: 'A4:4 G4:1 r:1 B4:1 r:1', 26: 'B4:4 A4:1 r:1 D5:1 r:1',
    27: 'C#5:1 r:1 C#5:1 A#4:1 B4:1 r:1 D5:1 r:1', 28: 'C#5:1 r:1 C#5:1 A#4:1 B4:1 r:1 D5:1 r:1',
    29: 'D5:2 D5:1 D5:1 D5:2 D5:1 D5:1', 30: 'D5:2 F#4:1 F#4:1 F#4:2 F#4:1 F#4:1',
    31: 'F#4:2 F#4:1 E#4:1 F#4:2 F#4:1 E#4:1', 32: 'F#4:1 r:1 F#4:1 E#4:1 F#4:2 F#4:1 E#4:1',
    33: 'F#4:1 r:1 F#4:1 E#4:1 F#4:1 r:1 F#4:1 E#4:1', 34: 'F#4:1 r:1 E4:1 r:1 D4:1 r:1 F#4:1 E#4:1',
    35: 'F#4:1 r:1 F#4:1 E#4:1 F#4:2 F#4:1 E#4:1', 36: 'F#4:1 r:1 B4:1 r:1 A4:1 r:1 A4:1 F#4:1',
    43: 'D4:2 D4:2 D4:2 r:2',
}
M[37], M[38], M[39], M[40] = M[9], M[10], M[11], M[12]
M[41], M[42] = M[13], M[15]
# chord per half bar ('U' = tutti unison, no chord)
CH = {1: 'U U', 2: 'U U', 3: 'U U', 4: 'A -', 5: 'D D', 6: 'D D', 7: 'D D', 8: 'D D', 9: 'A7 A7', 10: 'D D', 11: 'A/E E7', 12: 'A A7',
      13: 'E7 E7', 14: 'D/A D/A', 15: 'A7 A7', 16: 'D D', 17: 'G/D G/D', 18: 'G/D G/D', 19: 'D D', 20: 'D D', 21: 'A7 A7', 22: 'D/A D/A',
      23: 'A7 A7', 24: 'D D', 25: 'A7 A7', 26: 'D D', 27: 'F#7 Bm', 28: 'F#7 Bm', 29: 'D Bm', 30: 'D Bm', 31: 'D D', 32: 'D D',
      33: 'D D', 34: 'D D', 35: 'D D', 36: 'D D', 41: 'E7 E7', 42: 'A7 A7', 43: 'D D'}
CH[37], CH[38], CH[39], CH[40] = CH[9], CH[10], CH[11], CH[12]
# chord: (root pc, tones pcs)
CHORD = {'D': (2, [2, 6, 9]), 'A': (9, [9, 1, 4]), 'A7': (9, [9, 1, 4, 7]), 'E7': (4, [4, 8, 11, 2]), 'G': (7, [7, 11, 2]),
         'F#7': (6, [6, 10, 1, 4]), 'Bm': (11, [11, 2, 6])}
ORDER = ([('intro', b) for b in (1, 2, 3, 4)] + [('A1', b) for b in (5, 6, 7, 8, 9, 10, 11, 12)] +
         [('A2', b) for b in (5, 6, 7, 8, 13, 14, 15, 16)] + [('B', b) for b in range(17, 33)] +
         [('R', b) for b in (33, 34, 35, 36, 37, 38, 39, 40, 33, 34, 35, 36, 41, 42, 43)])
DYN = {'intro': 112, 'A1': 70, 'A2': 112, 'B': 84, 'R': 112}

notes, lead, secs, cur = [], [], [], None


def N(t, d, role, p, v): notes.append([round(t, 5), round(d, 5), role, int(p), int(max(1, min(127, v)))])


for k, (sec, b) in enumerate(ORDER):
    tb = T0 + k * BAR
    if sec != cur: secs.append([sec, round(tb, 5)]); cur = sec
    pos = 0
    for tok in M[b].split():
        p, d = P(tok)
        if p is not None:
            v = DYN[sec] + (8 if sec == 'B' and b >= 29 else 0)
            lead.append([round(tb + pos * E8, 5), p + (0 if sec == 'intro' and not (b == 4 and pos >= 6) else 12), v, round(d * E8, 5), sec, k, b])
        pos += d
    assert pos == 8, (b, M[b])
FINAL = lead[-1][0]

# ---- melody voices: the struck glockenspiel bar + doubling ------------------------------------------
for t, p, v, d, sec, k, b in lead:
    N(t, d, 'glk', p, v)                                       # golden bar the ball strikes
    if sec == 'intro':
        N(t, d * 0.9, 'vs', p, v); N(t, d * 0.9, 'tpt', p - 12, v - 10); N(t, d * 0.9, 'vc', p - 24, v - 5)
        N(t, d * 0.9, 'tuba', p - 36, v - 5)                   # tutti unison over three octaves
    elif sec == 'A1': N(t, d * 0.9, 'vs', p, v)
    elif sec in ('A2', 'R'): N(t, d * 0.9, 'vs', p, v); N(t, d * 0.85, 'tpt', p - 12 if p >= 70 else p, v - 12)
    elif sec == 'B': N(t, d * 0.9, 'vs', p, v); N(t, d * 0.9, 'fl', p + 12, v - 25)

# ---- march accompaniment: bass on beats 1 & 3, chords on 2 & 4 ---------------------------------------
BASSN, STAB, BEATS = [], [], []
for k, (sec, b) in enumerate(ORDER):
    tb = T0 + k * BAR
    halves = CH[b].split()
    lo = {'intro': 1.0, 'A1': 0.62, 'A2': 1.0, 'B': 0.78, 'R': 1.05}[sec]
    if sec == 'B' and b >= 29: lo = 0.95
    for hb in range(2): BEATS.append([round(tb + hb * HALF, 5), sec, k])
    for hb, ch in enumerate(halves):
        th = tb + hb * HALF
        if ch in ('U', '-'): continue
        if ch == 'A' and b == 4:                               # intro bar 4: A major chord on the downbeat only
            for p in (45, 57): N(th, Q * 1.6, 'tuba', p, 110)
            for p in (61, 64, 69): N(th, Q * 1.6, 'hn', p, 100)
            N(th, 0.6, 'timp', 45, 110); BASSN.append([th, 45]); continue
    # bass: quarter 1 = root (or slash bass) of the first half's chord, quarter 3 = fifth if the chord holds, else the new root
    def low(pc): p = 36 + pc; return p if p <= 45 else p - 12
    def bass_of(ch):
        name, sl = (ch.split('/') + [None])[:2]
        return low(PC[sl[0]] + (1 if '#' in sl else 0)) if sl else low(CHORD[name][0])
    if halves[0] not in ('U', '-') and b != 4:
        b1 = bass_of(halves[0])
        if halves[1] == halves[0]:
            b3 = b1 if '/' in halves[0] else low((CHORD[halves[0]][0] + 7) % 12)
        else: b3 = bass_of(halves[1])
        for tq, bp in ((tb, b1), (tb + 2 * Q, b3)):
            N(tq, Q * 0.6, 'tuba', bp, 98 * lo); N(tq, Q * 0.5, 'cb', bp + 12, 92 * lo); BASSN.append([round(tq, 5), bp])
    # off-beat chords: quarters 2 and 4
    for qi in (1, 3):
        hb = qi // 2; ch = halves[hb]
        if ch in ('U', '-'): continue
        to = tb + qi * Q
        name = ch.split('/')[0]
        r, tones = CHORD[name]
        for pc in tones[:4]:
            pp = 60 + pc
            if pp > 69: pp -= 12
            N(to, Q * 0.45, 'hn', pp, 84 * lo); N(to, Q * 0.4, 'va', pp + 12, 70 * lo)
        N(to, 0.08, 'snare', 0, 64 * lo); STAB.append(round(to, 5))
    if sec in ('A2', 'R', 'intro') and halves[0] not in ('U',):
        N(tb, 0.4, 'timp', 38 if (CHORD[halves[0].split('/')[0]][0] in (2, 7, 11, 6)) else 45, 96 * lo)
# intro: snare roll leading in, cymbal at section starts
for k in range(12): N(T0 - 0.62 + k * 0.05, 0.06, 'snare', 0, 50 + 4 * k)
for name, t in secs[1:]: N(t, 2.0, 'crash', 0, 100 if name in ('A2', 'R') else 74)
# final: D major tutti on the last D (3rd quarter), held
for p in (38, 50): N(FINAL, 3.0, 'tubasus', p, 122)
for p in (62, 66, 69, 74): N(FINAL, 3.0, 'hnsus', p, 112)
for p in (74, 78, 81, 86): N(FINAL, 3.0, 'vsus', p, 108)
for p in (74, 81): N(FINAL, 2.6, 'tptsus', p, 114)
N(FINAL, 3, 'crash', 0, 125); N(FINAL, 0.8, 'timp', 38, 125)

notes.sort(key=lambda n: n[0])
json.dump(notes, open('notes.json', 'w'))
json.dump([l[:4] + [l[4], l[5]] for l in lead], open('lead.json', 'w'))
json.dump({'t0': T0, 'q': Q, 'half': HALF, 'bar': BAR, 'final': FINAL, 'secs': secs, 'beats': BEATS, 'stab': STAB,
           'bass': BASSN, 'order': ORDER}, open('events.json', 'w'))
print('bars', len(ORDER), 'lead', len(lead), 'final', round(FINAL, 2), secs)
