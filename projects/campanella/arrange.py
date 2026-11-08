"""La Campanella (F. Liszt, Grandes etudes de Paganini S.141 No.3, 1851 - public domain). ~63 s excerpt.
Notes come unchanged from score_excerpt.json (JingleScript transcription by Satoshi Nakajima, CC BY 4.0, checked bar by
bar against a public-domain edition). Three passages are joined on matching beats of the 6/8 bar:
  A  bars 1-12   intro (octave bell tolls, fermata) + the theme             source beats [0.5, 35.5)
  B  bars 36-49  chromatic climb under the high D# bells, then the theme in the left hand under the
                 famous D#5-D#6-D#7-D#6 bell figure in the right hand          source beats [110.5, 148)
  C  last 3 bars G#-minor chords and the final chord                           source beats [427.5, 437)
A ends on the tonic arpeggio, B starts on the D# octave of the same beat position; B ends on the dominant (D#7) and C
resolves on the next downbeat. Tempo q = 80 (6/8, dotted quarter = 53).
Outputs notes.json  [t, midi, vel, dur, hand, src_beat, seg]  (one entry per struck pitch) and events.json."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(f'{HERE}/score_excerpt.json'))
BEAT = 0.75                 # q = 80
T0 = 2.0                    # hook before the first toll
NAMES = 'C C# D D# E F F# G G# A A# B'.split()


def midi(p):
    return NAMES.index(p[:-1]) + 12 * (int(p[-1]) + 1)


def seg_of(b):
    if b < 35.5: return 'A', b
    if b < 148: return 'B', b - 75.0
    return 'C', b - 354.562


notes = []
for e in S['events']:
    sg, m = seg_of(e['beat'])
    for p in e['pitch']:
        notes.append([round(T0 + m * BEAT, 5), midi(p), int(round(127 * e['vel'])), round(e['len'] * BEAT, 5), e['hand'], e['beat'], sg])
notes.sort(key=lambda n: (n[0], -n[1]))
FINAL = max(n[0] for n in notes)
json.dump(notes, open(f'{HERE}/notes.json', 'w'))
secs = {'intro': T0 + 0.5 * BEAT, 'theme': T0 + 11.0 * BEAT, 'climb': T0 + 35.5 * BEAT, 'fall': T0 + 49.0 * BEAT, 'finale': T0 + 73.0 * BEAT}
json.dump({'t0': T0, 'beat': BEAT, 'final': FINAL, 'secs': secs, 'dur': FINAL + 4.5}, open(f'{HERE}/events.json', 'w'), indent=0)
if __name__ == '__main__':
    print('notes', len(notes), 'final', FINAL, 'dur', FINAL + 4.5)
