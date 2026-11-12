"""Deck the Halls (Nos Galan) - traditional Welsh melody (public domain). Own arrangement for marbles.
Score source: score_src.json = Charles Wood's SATB setting (Cambridge Carol Book 1910, Wood d. 1926: public domain) as transcribed
in PDMX (Public Domain Mark), E-flat major; transposed here up a whole tone to F major. The melody is cross-checked note by note
against a second public-domain SATB edition in PDMX (F major).
Taken from the source: the melody (pitch / rhythm, all 16 bars) and the bass voice (the note sounding on beats 1 and 3).
Written here: who plays what (Holly the red marble = the verse lines, the ornaments = every 'Fa-la-la' refrain, the Yule-log ball =
the bass in verse 2), two verses, quarter = 120, a small ritardando in the very last bar.

Form (4/4): verse = 16 bars; lines (Holly) = bars 1-2, 5-6, 9-10, 13-14; refrains (9 ornament notes each) = 3-4, 7-8, 11-12, 15-16.
Outputs lead.json (melody notes) and events.json (bass, bar times, chords)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = json.load(open(f'{HERE}/score_src.json'))
Q = 0.5                     # quarter note (s)
T0 = 2.0                    # bar 1, beat 1
TR = 2                      # E-flat (Wood) -> F
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'
W = [[b, pos, d, p + TR, v] for b, pos, d, p, v in SRC['wood_eb']]
A = SRC['check_anon_f']
SOP = [e for e in W if e[4] == 'S']
SOPA = [e for e in A if e[4] == 'S']
assert len(SOP) == 68 and len(SOPA) == 68
for a, b in zip(SOP, SOPA):
    assert a[:4] == b[:4], ('melody differs from the cross-check edition', a, b)
LINE_BARS = (1, 2, 5, 6, 9, 10, 13, 14)


def t_of(gb, pos):
    """global bar gb = 1..32, pos in quarters; the last bar slows down a little (A, G, F)"""
    if gb < 32: return T0 + ((gb - 1) * 4 + pos) * Q
    tb = T0 + 31 * 4 * Q
    return tb + {0: 0.0, 1: 0.56, 2: 1.2}.get(pos, 1.2 + (pos - 2) * 0.65)


def sounding(bar, pos, voice):
    n = [e for e in W if e[4] == voice and e[0] == bar and e[1] <= pos + 1e-6 and e[1] + e[2] > pos + 1e-6]
    assert len(n) == 1, (bar, pos, voice, n)
    return n[0][3]


lead = []        # [t, pitch, vel, dur_s, global bar, pos, role, verse, refrain/line index]
for v in range(2):
    for bar, pos, d, p, _ in SOP:
        gb = 16 * v + bar
        role = 'holly' if bar in LINE_BARS else 'orn'
        k = 4 * v + (bar - 1) // 4          # line / refrain number 0..7
        vel = 74 + (8 if v else 0) + (6 if role == 'orn' else 0)
        t = t_of(gb, pos); dur = t_of(gb, pos + d) - t if gb < 32 or pos < 2 else 3.5
        lead.append([round(t, 5), p, vel, round(dur, 4), gb, pos, role, v, k])
FINAL = lead[-1][0]
bass = []; chords = {}
for gb in range(1, 33):
    bar = (gb - 1) % 16 + 1
    for pos in (0, 1, 2, 3):
        chords[f'{gb}-{pos}'] = sorted(set(NM[sounding(bar, pos, vv) % 12] for vv in 'SATB'))
    if gb >= 17:
        for pos in (0, 2):
            bass.append([round(t_of(gb, pos), 5), sounding(bar, pos, 'B'), gb, pos])
# the bass voice sounds the root or the 3rd of the source's chord at every struck beat
for t, p, gb, pos in bass:
    assert NM[p % 12] in chords[f'{gb}-{pos}']
json.dump(lead, open(f'{HERE}/lead.json', 'w'))
json.dump({'t0': T0, 'q': Q, 'final': FINAL, 'bass': bass, 'chords': chords,
           'bar_t': {gb: round(t_of(gb, 0), 5) for gb in range(1, 33)}}, open(f'{HERE}/events.json', 'w'))
if __name__ == '__main__':
    print('melody notes', len(lead), 'holly', sum(l[6] == 'holly' for l in lead), 'orn', sum(l[6] == 'orn' for l in lead),
          'bass', len(bass), 'final', round(FINAL, 3))
    for b in range(1, 17):
        print(b, ' '.join(f"{nm(l[1])}@{l[5]:g}" for l in lead if l[4] == b), '| bass(v2)',
              ' '.join(nm(x[1]) for x in bass if x[2] == b + 16), '|', chords[f'{b}-0'], chords[f'{b}-2'])
