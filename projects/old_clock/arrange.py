"""Grandfather's Clock (Henry Clay Work, 1876 - public domain). Melody only (no lyrics anywhere).
Melody: Work's 1876 tune (first edition New York: C.M. Cady 1876, IMSLP #674507, public domain), melody line only,
in G major, 4/4: verse x2, bridge, 'stopped short' phrase, chorus (8 bars). Pitches/rhythm taken from a melody-only
traditional transcription (abc, folkinfo.org) of the PD tune - no modern harmony/arrangement was copied; every other
voice here is our own. Played one octave up (music-box register).
Own accompaniment: bass chime rods struck on beats 1 and 3 (chord roots/fifths below).
Tempo q = 96. After the last note: epilogue tag (pickup + bar 1 + first note of bar 2) in a small new clock.
Outputs lead.json [t, midi, vel, dur, section, bar], notes.json [t, dur, role, midi, vel], events.json."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
Q = 0.625; E8 = Q / 2; BAR = 4 * Q
T_PICK = 2.0                       # the first (pickup) note
TB1 = T_PICK + Q                   # downbeat of bar 1
NM = {'B,': 59, 'C': 60, 'D': 62, 'E': 64, 'F': 66, 'G': 67, 'A': 69, 'B': 71, 'c': 72, 'd': 74, 'e': 76}   # key G: F = F#


def parse(bar):
    out = []
    for tok in bar.split():
        if tok.startswith('B,'): p, d = 'B,', tok[2:]
        else: p, d = tok[0], tok[1:]
        out.append((None if p == 'z' else NM[p] + 12, float(eval(d)) if d else 1.0))
    return out


V1 = ['G2 F G A2 G A', 'B2 c3/2 B1/2 E2 A A', 'G2 G G F2 E F', 'G6 D D']
V2 = ['G2 F G A2 G A', 'B2 c3/2 B1/2 E2 A A', 'G2 G G F2 E F', 'G6 G B']
BR = ['d2 B3/2 A1/2 G2 F G', 'A G F E D2 G B', 'd2 B3/2 A1/2 G2 F G', 'A6 D D']
ST = ['G2 z2 A2 z2', 'B2/3 B2/3 B2/3 c3/2 B1/2 E2 A A', 'G4 F4', 'G6 D D']
CH = ['G2 D D E D D2', 'B,2 D2 B,2 D D', 'G2 D D E D D2', 'B,2 D2 B,2 D D',
      'G2 z2 A2 z2', 'B2/3 B2/3 B2/3 c3/2 B1/2 E2 A A', 'G4 F4', 'G6 z2']
BARS = [('v1', b) for b in V1] + [('v2', b) for b in V2] + [('br', b) for b in BR] + [('st', b) for b in ST] + [('ch', b) for b in CH]
# chords per half bar (own harmonisation, checked against every melody note)
CHORD = ['G D7', 'G Am', 'G D7', 'G G'] * 2 + ['G Em', 'D7 G', 'G Em', 'D7 D7'] + ['G D7', 'G Am', 'G D7', 'G G'] + \
        ['G G', 'G Em', 'G D7', 'G G', 'G D7', 'G Am', 'G D7', 'G -']
TONES = {'G': (67, 71, 74), 'Am': (69, 72, 76), 'D7': (74, 78, 81, 72), 'Em': (76, 79, 71)}
ROOT = {'G': 43, 'Am': 45, 'D7': 38, 'Em': 40}
FIFTH = {'G': 38, 'Am': 40, 'D7': 45, 'Em': 47}

lead = [[T_PICK, NM['D'] + 12, 84, Q, 'v1', 0]]
for b, (sec, bar) in enumerate(BARS):
    pos = 0.0
    for p, d in parse(bar):
        t = TB1 + b * BAR + pos * E8
        if p is not None:
            v = {'v1': 86, 'v2': 90, 'br': 92, 'st': 96, 'ch': 94}[sec]
            if d >= 4: v += 8
            lead.append([round(t, 5), p, v, round(d * E8, 5), sec, b + 1])
        pos += d
    assert abs(pos - 8) < 1e-6, (b, bar, pos)
FINAL = TB1 + 23 * BAR
assert abs(lead[-1][0] - FINAL) < 1e-6
# epilogue tag in the little clock: pickup + bar 1 + first note of bar 2, two octaves up from the written tune
T_TAG = 64.4
tag = [[T_TAG, 74 + 12, 80, Q]]
pos = 0.0
for p, d in parse('G2 F G A2 G A') + [(71 + 12, 4.0)]:
    tag.append([round(T_TAG + Q + pos * E8, 5), p + 12, 84 if d < 4 else 92, d * E8]); pos += d

notes = []
for t, p, v, d, sec, b in lead: notes.append([t, d, 'mel', p, v])
for t, p, v, d in tag: notes.append([t, d, 'tag', p, v])
bass = []
for b, ch in enumerate(CHORD):
    a, c = ch.split()
    tb = TB1 + b * BAR
    bass.append([tb, ROOT[a], b + 1, 1])
    if c == '-': continue
    bass.append([tb + 2 * Q, FIFTH[a] if c == a else ROOT[c], b + 1, 3])
for t, p, b, beat in bass: notes.append([round(t, 5), 2 * Q, 'bass', p, 96 if beat == 1 else 84])
notes.sort(key=lambda n: (n[0], n[2]))
# harmony check: every melody note on a beat must be a chord tone or a passing/neighbour tone a step from one
names = 'C C# D D# E F F# G G# A A# B'.split()
bad = []
for t, p, v, d, sec, b in lead[1:]:
    half = int(round((t - TB1) / (2 * Q) - 0.5 + 1e-6)) if False else int((t - TB1 + 1e-6) // (2 * Q))
    ch = CHORD[b - 1].split()[half % 2]
    if ch == '-': ch = CHORD[b - 1].split()[0]
    pcs = {x % 12 for x in TONES[ch]}
    if p % 12 not in pcs:
        on_beat = abs(((t - TB1) / Q) - round((t - TB1) / Q)) < 1e-6
        bad.append((b, names[p % 12], ch, 'beat' if on_beat else 'off'))
json.dump(lead, open(f'{HERE}/lead.json', 'w'))
json.dump(notes, open(f'{HERE}/notes.json', 'w'))
secs = {'v1': T_PICK, 'v2': TB1 + 3 * BAR + 6 * E8, 'br': TB1 + 7 * BAR + 6 * E8, 'st': TB1 + 11 * BAR + 6 * E8,
        'ch': TB1 + 15 * BAR + 6 * E8, 'end': TB1 + 20 * BAR}
json.dump({'q': Q, 'bar': BAR, 'tb1': TB1, 't_pick': T_PICK, 'final': FINAL, 'secs': secs, 'tag': tag, 't_tag': T_TAG,
           'bass': bass, 'dur': 71.5}, open(f'{HERE}/events.json', 'w'), indent=0)
if __name__ == '__main__':
    print('melody notes', len(lead), 'final', FINAL, 'tag', [round(x[0], 3) for x in tag])
    print('non-chord tones', bad)
