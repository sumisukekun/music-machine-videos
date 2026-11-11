"""Eine kleine Nachtmusik K.525, 1st movement (Allegro) - W. A. Mozart, 1787, public domain. Own reduction.
Score source: Mutopia Project, MozartWA/KV525/eine-kleine-nachtmusik-mvt1 (typeset from the IMSLP edition, placed in the
public domain by the typesetter; copies of the four part files are in score/). The pitches and rhythms below were read
from those parts (LilyPond -> MIDI dump of bars 1-17). Only Mozart's notes are used; the 3-voice reduction, the ending
chord and everything else are written here.
  form   : bars 1-17 (opening theme, 'rocket' theme, the piano phrase), then bars 1-10 again (da capo), final G chord
  voices : V1 = violin I (the lamp ball), V2 = violin II (the dancing couple), B = cello (the gold bass ball)
  reduction rules: the violin I trill in bars 6/8 is played as its main note (g8); the appoggiaturas of bars 9/10 are
  written out as sixteenths (g16 fis e fis, b16 a g a, d16 c b c); the acciaccaturas of bars 12/13 are left out;
  the violin II tremolo of bars 5-8 is played as half-note dyads (its harmony); the viola (doubling) is left out.
Outputs notes.json (every sound), lead.json (V1), duo.json (V2), bass.json (B), events.json (grid, sections)."""
import json

Q = 0.5                      # quarter = 120
BAR = 4 * Q
T0 = 2.4                     # bar 1 downbeat (the balls drop from the dark chandelier before it)
FERMATA = 0.3
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def P(s):
    n, o = (s[:-1], int(s[-1])); return 12 * (o + 1) + NM.index(n)
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'

# (beat, pitch, beats) per bar; beats counted from 0
V1 = {
    1: [(0, 'G5', 1), (1.5, 'D5', .5), (2, 'G5', 1), (3.5, 'D5', .5)],
    2: [(0, 'G5', .5), (.5, 'D5', .5), (1, 'G5', .5), (1.5, 'B5', .5), (2, 'D6', 1)],
    3: [(0, 'C6', 1), (1.5, 'A5', .5), (2, 'C6', 1), (3.5, 'A5', .5)],
    4: [(0, 'C6', .5), (.5, 'A5', .5), (1, 'F#5', .5), (1.5, 'A5', .5), (2, 'D5', 1)],
    5: [(0, 'G5', .5), (1, 'G5', 1.5), (2.5, 'B5', .5), (3, 'A5', .5), (3.5, 'G5', .5)],
    6: [(0, 'G5', .5), (.5, 'F#5', .5), (1, 'F#5', 1.5), (2.5, 'A5', .5), (3, 'C6', .5), (3.5, 'F#5', .5)],
    7: [(0, 'A5', .5), (.5, 'G5', .5), (1, 'G5', 1.5), (2.5, 'B5', .5), (3, 'A5', .5), (3.5, 'G5', .5)],
    9: [(0, 'G5', .5), (.5, 'G5', .5), (1, 'G5', .25), (1.25, 'F#5', .25), (1.5, 'E5', .25), (1.75, 'F#5', .25),
        (2, 'G5', .5), (2.5, 'G5', .5), (3, 'B5', .25), (3.25, 'A5', .25), (3.5, 'G5', .25), (3.75, 'A5', .25)],
    10: [(0, 'B5', .5), (.5, 'B5', .5), (1, 'D6', .25), (1.25, 'C6', .25), (1.5, 'B5', .25), (1.75, 'C6', .25), (2, 'D6', 1)],
    11: [(0, 'D5', 2), (2, 'E5', 2)],
    12: [(0, 'C5', 1), (1, 'C5', 1), (2, 'B4', 1), (3, 'B4', 1)],
    13: [(0, 'A4', 1), (1, 'A4', 1), (2, 'G4', .5), (2.5, 'F#4', .5), (3, 'E4', .5), (3.5, 'F#4', .5)],
    14: [(0, 'G4', .5), (1, 'A4', .5), (2, 'B4', .5)],
    15: [(0, 'D5', 2), (2, 'E5', 2)],
    16: [(0, 'D5', .5), (.5, 'C5', .5), (1, 'C5', .5), (1.5, 'C5', .5), (2, 'C5', .5), (2.5, 'B4', .5), (3, 'B4', .5), (3.5, 'B4', .5)],
    17: [(0, 'B4', .5), (.5, 'A4', .5), (1, 'A4', .5), (1.5, 'A4', .5), (2, 'G4', .5), (2.5, 'F#4', .5), (3, 'E4', .5), (3.5, 'F#4', .5)],
}
V1[8] = V1[6]
# violin II: bars 1-4 in unison with violin I (bar 1 beat 1 = the chord's lower notes D4 + B4)
V2 = {k: [(b, [p], d) for b, p, d in V1[k]] for k in (1, 2, 3, 4)}
V2[1][0] = (0, ['D4', 'B4'], 1)
V2[5] = [(0, ['B3', 'D4'], 2), (2, ['B3', 'D4'], 2)]
V2[6] = [(0, ['C4', 'D4'], 2), (2, ['C4', 'D4'], 2)]
V2[7] = V2[5]; V2[8] = V2[6]
V2[9] = [(0, ['B3', 'D4'], 1), (1, ['C5'], .5), (1.5, ['C5'], .5), (2, ['D5'], .5), (2.5, ['D5'], .5), (3, ['C5'], .5), (3.5, ['B4'], .25), (3.75, ['A4'], .25)]
V2[10] = [(0, ['G4'], .5), (.5, ['G4'], .5), (1, ['F#4'], .5), (1.5, ['F#4'], .5), (2, ['G4'], 1)]
V2[11] = [(0, ['B4'], 2), (2, ['C5'], 2)]
V2[12] = [(0, ['A4'], 1), (1, ['A4'], 1), (2, ['G4'], 1), (3, ['G4'], 1)]
V2[13] = [(0, ['E4'], 1), (1, ['E4'], 1), (2, ['C4'], 1), (3, ['A3'], 1)]
V2[14] = [(0, ['D4'], .5), (1, ['F#4'], .5), (2, ['G4'], .5)]
V2[15] = [(0, ['B4'], 2), (2, ['C5'], 2)]
V2[16] = [(0, ['B4'], .5), (.5, ['A4'], .5), (1, ['A4'], .5), (1.5, ['A4'], .5), (2, ['A4'], .5), (2.5, ['G4'], .5), (3, ['G4'], .5), (3.5, ['G4'], .5)]
V2[17] = [(0, ['E4'], 1), (1, ['E4'], .5), (1.5, ['E4'], .5), (2, ['C4'], 1), (3, ['C4'], .5), (3.5, ['C4'], .5)]
# cello
B = {
    1: [(0, 'G3', 1), (1.5, 'D3', .5), (2, 'G3', 1), (3.5, 'D3', .5)],
    2: [(0, 'G3', .5), (.5, 'D3', .5), (1, 'G3', .5), (1.5, 'B3', .5), (2, 'D4', 1)],
    3: [(0, 'C4', 1), (1.5, 'A3', .5), (2, 'C4', 1), (3.5, 'A3', .5)],
    4: [(0, 'C4', .5), (.5, 'A3', .5), (1, 'F#3', .5), (1.5, 'A3', .5), (2, 'D3', 1)],
    9: [(k * .5, p, .5) for k, p in enumerate(['G3', 'G3', 'A3', 'A3', 'B3', 'B3', 'F#3', 'F#3'])],
    10: [(0, 'G3', .5), (.5, 'G3', .5), (1, 'A3', .5), (1.5, 'A3', .5), (2, 'B3', 1)],
    11: [],
    12: [(0, 'D3', 2), (2, 'E3', 2)],
    13: [(0, 'C3', 1), (1, 'C3', 1), (2, 'D3', 1), (3, 'D3', 1)],
    14: [(0, 'B2', .5), (1, 'D3', .5), (2, 'G3', 1)],
    15: [],
    16: [(0, 'D3', 2), (2, 'E3', 2)],
    17: [(0, 'C3', 1), (1, 'C3', 1), (2, 'D3', 1), (3, 'D3', 1)],
}
for k in (5, 6, 7, 8): B[k] = [(j * .5, 'G3', .5) for j in range(8)]

FORM = [(1, k) for k in range(1, 18)] + [(2, k) for k in range(1, 11)]
grid = []; t = T0
for ps, k in FORM: grid.append((ps, k, round(t, 5))); t += BAR
FINAL = round(t + FERMATA, 5)            # bar 11 of the da capo: the final chord (the grand chandelier)


def vel(ps, k):
    if ps == 1 and 11 <= k <= 17: return 58
    return 100 if ps == 1 else 108


lead, duo, bass = [], [], []
for ps, k, tb in grid:
    for b, p, d in V1[k]: lead.append([round(tb + b * Q, 5), P(p), vel(ps, k), round(d * Q, 5), ps, k, b])
    for b, pp, d in V2[k]: duo.append([round(tb + b * Q, 5), [P(p) for p in pp], vel(ps, k) - 8, round(d * Q, 5), ps, k, b])
    for b, p, d in B[k]: bass.append([round(tb + b * Q, 5), P(p), vel(ps, k) - 4, round(d * Q, 5), ps, k, b])

notes = []
def N(t, d, role, p, v): notes.append([round(t, 5), round(d, 5), role, int(p), int(max(1, min(127, v)))])
for t, p, v, d, ps, k, b in lead: N(t, d, 'lamp', p, v)
for t, pp, v, d, ps, k, b in duo:
    for p in pp: N(t, d, 'duo', p, v)
for t, p, v, d, ps, k, b in bass: N(t, d, 'bass', p, v)
# final chord (G major): the lamp ball rings the chandelier bell (G5), the couple lands in the two cup bells (B4, D5),
# the great chandelier bell sounds the low G; strings hold the chord
N(FINAL, 3.0, 'bigbell', 43, 124)
N(FINAL, 3.0, 'lampfin', 79, 118)
for p in (71, 74): N(FINAL, 3.0, 'duofin', p, 108)
for p in (43, 55, 62, 71, 79): N(FINAL, 3.2, 'tutti', p, 110)
notes.sort(key=lambda n: n[0])
GUST = round(grid[9][2] + 3 * Q, 5)      # bar 10 beat 4 (a rest): the window bursts open
DACAPO = grid[17][2]                      # bar 1 again: the brazier
json.dump(notes, open('notes.json', 'w'))
json.dump(lead, open('lead.json', 'w')); json.dump(duo, open('duo.json', 'w')); json.dump(bass, open('bass.json', 'w'))
json.dump({'t0': T0, 'q': Q, 'bar': BAR, 'final': FINAL, 'grid': grid, 'gust': GUST, 'dacapo': DACAPO}, open('events.json', 'w'))
if __name__ == '__main__':
    print('bars', len(grid), 'V1', len(lead), 'V2', len(duo), 'B', len(bass), 'gust', GUST, 'dacapo', DACAPO, 'final', FINAL)
    print('V1 pass1 bars1-10:', sum(1 for l in lead if l[4] == 1 and l[5] <= 10), 'pass2:', sum(1 for l in lead if l[4] == 2))
