"""Dvorak - Symphony No. 9 'From the New World', IV. Allegro con fuoco (1893, public domain). Own reduction.
Bars 1-43 (intro, main theme x2, the quiet violin answer, tutti theme + the climbing sequence) and a final E-major
chord on bar 44. 4/4, quarter = 152. Pitches / rhythms / chords follow the score (checked note by note against an
orchestral MIDI reduction of the movement, see verify.py 'score').
Who strikes what (= what is seen):
  lead  -> the locomotive (black ball) on its own rail of xylophone sleepers
  acc   -> the four passenger cars: red = bass on beats 1 and 3 (drum bars), yellow/green/blue = the 3 chord tones on
           beats 2 and 4 (brass bells) - an 'oom-pah' that repeats the score's chord of that half bar (bars 8, 9, 34 and
           the end: all four together as in the score); bars 26-33 have no chords in the score, only the cello line:
           all four cars play it in octaves
  the excerpt ends on the downbeat of bar 44 with an E-major chord (the key the movement ends in)
Outputs notes.json (all sounds), lead.json, acc.json, events.json."""
import json
from fractions import Fraction as F

Q = 60 / 152                 # quarter
S16 = Q / 4
BAR = 4 * Q
T0 = 0.5
PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def M(n):
    """'F#5' -> midi"""
    p = PC[n[0]]; k = 1
    while n[k] in '#b': p += 1 if n[k] == '#' else -1; k += 1
    return 12 * (int(n[k:]) + 1) + p


def bar(s):
    out = []
    for tok in s.split():
        n, d = tok.split(':'); out.append((None if n == 'z' else M(n), F(d)))
    assert sum(d for _, d in out) == 16, s
    return out


TH1 = ['E5:8 F#5:4 G5:4', 'F#5:6 E5:2 E5:8', 'E5:8 D5:4 B4:2 D5:2', 'E5:12 E5:4',
       'E5:8 F#5:4 G5:4', 'F#5:6 E5:2 E5:8', 'E5:4 G5:4/3 E5:4/3 G5:4/3 B5:4 B4:4',
       'E5:1 E5:1 B4:1 B4:1 G4:1 G4:1 F#4:1 F#4:1 E4:1 E4:1 B3:1 B3:1 E4:1 E4:1 G4:1 G4:1']
VA = ['B4:8 C5:4 D5:4', 'C5:6 B4:2 B4:8', 'B4:8 A4:4 G4:2 A4:2', 'B4:12 B4:4']
LEAD = (['B3:6 C4:10', 'B3:6 C4:10', 'B3:4 C4:4 B3:4 C4:4',
         'B3:2 C4:2 B3:2 C#4:2 B3:2 D#4:2 B3:2 E4:2', 'B3:2 F#4:2 B3:2 G4:2 B3:2 A4:2 B3:2 A#4:2',
         'B4:4 C5:4 B4:4 C5:4', 'B4:4 C#5:4 B4:4 D#5:4',
         'z:4 D#5:4 D#5:4 D#5:4', 'C5:2 B4:2 C5:2 B4:2 C5:2 B4:2 C5:2 B4:2']
        + TH1 + TH1 + VA + VA + TH1[:4] +
        ['E5:8 F#5:4 G5:4', 'A5:6 G5:1 A5:1 B5:8', 'G5:1 F#5:1 E5:6 F#5:4 D#5:4', 'G5:1 F#5:1 E5:6 F#5:4 D#5:4',
         'G5:1 F#5:1 E5:2 F#5:2 D#5:2 G5:1 F#5:1 E5:2 F#5:2 D#5:2',
         'G5:1 F#5:1 E5:2 F#5:2 D#5:2 F#5:2 D#5:2 F#5:2 D#5:2'])
NB = len(LEAD)                     # 43 bars, the final chord is the downbeat of bar 44
SEC = [('intro', 1), ('couple', 8), ('th1', 10), ('th2', 18), ('va', 26), ('tutti', 34), ('climb', 38), ('final', 44)]
def sec_of_bar(b): return [n for n, b0 in SEC if b0 <= b][-1]

# chords: bar -> [(pos16, bass, [3 tones])]   (all from the score)
Em, Am = ['G3', 'B3', 'E4'], ['A3', 'C4', 'E4']
G_, B7, Fo = ['G3', 'B3', 'D4'], ['A3', 'B3', 'D#4'], ['F#3', 'A3', 'C4']
BM = ['F#3', 'B3', 'D#4']
TH_CH = {1: [(0, 'E2', Em)], 2: [(0, 'A2', Am)], 3: [(0, 'E2', Em), (8, 'B2', G_)], 4: [(0, 'E2', Em)],
         5: [(0, 'E2', Em)], 6: [(0, 'A2', Am)], 7: [(0, 'G2', Em), (8, 'F#2', B7)], 8: [(0, 'E2', Em)]}
CH = {8: [(4, 'D#2', ['G3', 'A#3', 'D#4']), (8, 'C#2', ['G3', 'A#3', 'C#4']), (12, 'C#2', ['G3', 'A#3', 'C#4'])],
      9: [(0, 'C2', ['F#3', 'A3', 'D#4'])]}
for k, v in TH_CH.items():
    CH[9 + k] = v; CH[17 + k] = v
CH[20] = [(0, 'E2', Em), (8, 'G2', G_)]            # 2nd time: bass G under the G chord (score)
CH.update({34: [(0, 'E2', Em)], 35: [(0, 'A2', Am)], 36: [(0, 'E2', Em), (8, 'G2', G_)], 37: [(0, 'E2', Em)],
           38: [(0, 'E2', Em)], 39: [(0, 'C3', Am), (8, 'G2', Em)],
           40: [(0, 'A2', Am), (8, 'A2', Fo), (12, 'B2', BM)], 41: [(0, 'A2', Am), (8, 'A2', Fo), (12, 'B2', BM)],
           42: [(0, 'A2', Am), (4, 'A2', Fo), (6, 'B2', BM), (8, 'A2', Am), (12, 'A2', Fo), (14, 'B2', BM)],
           43: [(0, 'A2', Am), (4, 'A2', Fo), (6, 'B2', BM), (8, 'A2', Fo), (10, 'B2', BM), (12, 'A2', Fo), (14, 'B2', BM)],
           44: [(0, 'E2', ['G#3', 'B3', 'E4'])]})
# bars 26-33: the cello line under the violins (no chords in the score) -> all four cars in octaves
CEL = {26: 'B3:4 A3:4 G3:4 F#3:4', 27: 'G3:8 A3:4 B3:4', 28: 'B3:4 C4:4 D4:4 E4:2 C4:2', 29: 'B3:4 A3:2 F#3:2 B3:4 B2:4',
       30: 'B3:4 A3:4 G3:4 F#3:4', 31: 'G3:8 A3:6 B3:2', 32: 'B3:4 C4:4 D4:4 E4:2 C4:2', 33: 'B3:4 A3:2 F#3:2 B3:4 B2:4'}

def tb(b): return T0 + (b - 1) * BAR
lead = []
for b, s in enumerate(LEAD, 1):
    pos = F(0)
    for p, d in bar(s):
        if p is not None: lead.append([round(tb(b) + float(pos) * S16, 5), p, float(d) * S16, sec_of_bar(b), b])
        pos += d
FINAL = tb(44)
lead.append([FINAL, M('E5'), 2.5, 'final', 44])
acc = []
def chord_at(b, pos):
    return [c for c in CH[b] if c[0] <= pos][-1]
def add_acc(t, b, kind, per):
    acc.append(dict(t=round(t, 5), bar=b, kind=kind, p={str(c): int(p) for c, p in per.items()}))
for b in range(8, 44):
    if b in (8,):
        for pos, bs, tones in CH[b]: add_acc(tb(b) + pos * S16, b, 'tutti', {0: M(bs), 1: M(tones[0]), 2: M(tones[1]), 3: M(tones[2])})
        continue
    if 26 <= b <= 33:                     # the cello line, all four cars in unison (octaves)
        pos = F(0)
        for p, d in bar(CEL[b]):
            add_acc(tb(b) + float(pos) * S16, b, 'cello', {0: p - 12, 1: p, 2: p, 3: p + 12}); pos += d
        continue
    for beat in range(4):                  # 'oom-pah': red = bass on beats 1 and 3, the three chord cars on 2 and 4
        pos = 4 * beat
        _, bs, tones = chord_at(b, pos)
        if b in (42, 43) and beat in (1, 3): tones = BM                  # the dominant of the climbing sequence
        t = tb(b) + pos * S16
        if beat == 0 and b in (9, 34):
            add_acc(t, b, 'tutti', {0: M(bs), 1: M(tones[0]), 2: M(tones[1]), 3: M(tones[2])})
        elif beat in (0, 2): add_acc(t, b, 'bass', {0: M(bs)})
        else: add_acc(t, b, 'chord', {1: M(tones[0]), 2: M(tones[1]), 3: M(tones[2])})
_, bs, tones = CH[44][0]
add_acc(FINAL, 44, 'final', {0: M(bs), 1: M(tones[0]), 2: M(tones[1]), 3: M(tones[2])})
acc.sort(key=lambda a: a['t'])

notes = []
def N(t, d, role, p, v): notes.append([round(t, 5), round(d, 5), role, int(p), int(max(1, min(127, v)))])
DYN = {'intro': 70, 'couple': 112, 'th1': 100, 'th2': 106, 'va': 78, 'tutti': 116, 'climb': 112, 'final': 124}
for i, (t, p, d, s, b) in enumerate(lead):
    v = DYN[s]
    if s == 'intro': v = 56 + 4 * b + (8 if b >= 6 else 0)
    if s == 'climb': v = 104 + 3 * (b - 38)
    N(t, d, 'loco', p, v)                                   # the struck sleeper itself (xylophone + marimba)
    if s == 'intro':
        N(t, d * 0.9, 'cspic', p - 12, v); N(t, d * 0.9, 'vspic', p, v - 10)
    elif s == 'couple':
        N(t, d * 0.9, 'vspic', p, v); N(t, d * 0.9, 'tpt', p, v - 10)
    elif s in ('th1', 'th2', 'tutti', 'climb'):
        fig = d < S16 * 1.5 and b in (17, 25)                 # the string run under the held horn note
        if fig: N(t, d, 'vspic', p, v - 6)
        else:
            N(t, d * 0.9, 'tpt' if s != 'th1' else 'hn', p if s != 'th1' else p - 12, v)
            if s == 'th1': N(t, d * 0.9, 'tpt', p, v - 18)
            else: N(t, d * 0.9, 'hn', p - 12, v - 8); N(t, d * 0.9, 'vspic', p, v - 14)
    elif s == 'va':
        N(t, d, 'vsus', p, v); N(t, d, 'vsus', p - 12, v - 20)
for a in acc:
    t = a['t']; b = a['bar']; s = sec_of_bar(b); k = a['kind']; P = {int(c): p for c, p in a['p'].items()}
    v = DYN[s]
    if k == 'cello':
        d = [x for x in acc if x['t'] > t + 1e-6]; d = (d[0]['t'] - t) if d else Q
        for c, p in P.items(): N(t, d, 'car', p, 80 if c else 86)
        N(t, d, 'csus', P[1], 82); N(t, d * 0.8, 'cb', P[0], 74); continue
    if 0 in P:
        bs = P[0]
        N(t, 0.4, 'carbass', bs, v); N(t, 0.4, 'cb', bs + 12, v - 6); N(t, 0.4, 'tuba', bs + 12 if bs < 40 else bs, v - 4)
        if bs % 12 in (4, 11): N(t, 0.5, 'timp', bs, v)     # timpani on E and B (as in the score)
    for c in (1, 2, 3):
        if c in P:
            p = P[c]; N(t, 0.4, 'car', p, v - 8); N(t, 0.3, 'hnst', p + 12 if p < 57 else p, v - 12); N(t, 0.3, 'vlaspic', p + 12, v - 20)
    if b == 9 and k == 'tutti':                                # the held diminished chord of bar 9
        for c in (1, 2, 3): N(t, BAR * 0.95, 'hnsus', P[c] + 12, 92); N(t, BAR * 0.95, 'vsus', P[c] + 24, 86)
        N(t, BAR * 0.95, 'tubasus', P[0] + 12, 92)
    if k == 'final':
        N(t, 3.4, 'tubasus', 40, 122); N(t, 3.4, 'tubasus', 52, 116)
        for p in (56, 59, 64, 68): N(t, 3.4, 'hnsus', p, 112)
        for p in (76, 80, 83, 88): N(t, 3.4, 'vsus', p, 108)
        for p in (76, 83): N(t, 3.0, 'tptsus', p, 114)
N(tb(34), 2.2, 'crash', 0, 118)        # bar 34: the locomotive lands on the brass plate at the far cliff
N(FINAL, 3.0, 'crash', 0, 124)
notes.sort(key=lambda n: n[0])
json.dump(notes, open('notes.json', 'w'))
json.dump(lead, open('lead.json', 'w'))
json.dump(acc, open('acc.json', 'w'), indent=0)
json.dump({'t0': T0, 'q': Q, 'bar': BAR, 'final': FINAL, 'secs': [[n, tb(b0)] for n, b0 in SEC]}, open('events.json', 'w'), indent=0)
print('lead', len(lead), 'acc', len(acc), 'final', round(FINAL, 2), 'secs', [(n, round(tb(b0), 2)) for n, b0 in SEC])
