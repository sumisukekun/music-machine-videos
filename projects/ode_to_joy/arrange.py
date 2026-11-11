"""Ode to Joy - L. van Beethoven, Symphony No. 9 Op. 125, IV (1824, public domain). Own arrangement for marbles.
Score source: score_src.json (PDMX, CC0 1.0: MuseScore score 192793 'Beethoven Symphony No. 9 (4th movement) Piano solo with
chorus'; see its 'source' field). 'tutti' = the piano part of score bars 164-187, the full-orchestra statement of the theme;
'baritone' = the baritone solo statement (score bars 241-256). Cross-checked with Mutopia 'Ode to Joy' (BeethovenLv/ode, PD).
Taken from the source: the melody (pitches / rhythm, incl. the syncopated F# tied over bar 12-13), and per beat the bass note and
the chord that the tutti statement sounds. Everything else (who plays what, voicings, the order of entries, the extra repeat of
the last 8 bars, the ritardando and the final chord) is written here.

Form (theme bars, D major, 4/4):  A 1-4 | A' 5-8 | B 9-12 | A'' 13-16 | B 17-20 | A'' 21-24 | B 25-28 | A'' 29-32 (= 17-24 again)
Entries (Beethoven also starts the theme with one voice - cellos alone - and lets the orchestra join statement by statement):
  bar 1   the soloist alone: melody                                   (1 ball)
  bar 5   + the bass ball: bass note on beats 1 and 3                  (2)
  bar 9   + the harmony pair: a two-note chord on beats 2 and 4         (4)
  12-13   + 4 balls join the melody in unison (one per note)            (8)
  17-18   + 8 more join the melody (one per note)                      (16)
  25-32   the rain: 16 balls per beat ring the chord of that beat on the great bells and take their seats (+496 = 512 = 2^9)
Outputs lead.json (melody notes), events.json (grid, bass, pair, rain, chords)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = json.load(open(f'{HERE}/score_src.json'))
Q = 0.45                      # quarter (= 133 bpm)
BAR = 4 * Q
T0 = 2.3                      # bar 1 beat 1
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'

# ---- melody per theme bar: (pos, dur, midi)
M = {}
M[1] = [(0, 1, 66), (1, 1, 66), (2, 1, 67), (3, 1, 69)]
M[2] = [(0, 1, 69), (1, 1, 67), (2, 1, 66), (3, 1, 64)]
M[3] = [(0, 1, 62), (1, 1, 62), (2, 1, 64), (3, 1, 66)]
M[4] = [(0, 1.5, 66), (1.5, 0.5, 64), (2, 2, 64)]
M[5], M[6], M[7] = M[1], M[2], M[3]
M[8] = [(0, 1.5, 64), (1.5, 0.5, 62), (2, 2, 62)]
M[9] = [(0, 1, 64), (1, 1, 64), (2, 1, 66), (3, 1, 62)]
M[10] = [(0, 1, 64), (1, 0.5, 66), (1.5, 0.5, 67), (2, 1, 66), (3, 1, 62)]
M[11] = [(0, 1, 64), (1, 0.5, 66), (1.5, 0.5, 67), (2, 1, 66), (3, 1, 64)]
M[12] = [(0, 1, 62), (1, 1, 64), (2, 1, 57), (3, 2, 66)]          # F# tied over into bar 13 (the famous syncopation)
M[13] = [(1, 1, 66), (2, 1, 67), (3, 1, 69)]
M[14], M[15], M[16] = M[2], M[3], M[8]
for k in range(17, 25): M[k] = M[k - 8]
# ---- chord per beat (theme bars), read from the tutti statement (bass + all notes of that beat)
CH = {1: ['D', 'D', 'Em/D', 'D'], 2: ['D/A', 'A7', 'D/A', 'A'], 3: ['D', 'D', 'A', 'D'], 4: ['D/A', 'D/A', 'A', 'A'],
      8: ['A', 'A', 'D', 'D'], 9: ['A', 'A', 'D/A', 'D'], 10: ['A', 'D/A', 'D/A', 'D'], 11: ['A', 'D/A', 'F#/A#', 'F#7'],
      12: ['Bm', 'E', 'A', 'D/A'], 13: ['D', 'D', 'Em/D', 'D'], 16: ['A', 'A', 'D', 'A7'], 24: ['A', 'A', 'D', 'D']}
CH[5], CH[6], CH[7] = CH[1], CH[2], CH[3]
CH[14], CH[15] = CH[2], CH[3]
for k in range(17, 24): CH[k] = CH[k - 8]
PCS = {'D': [2, 6, 9], 'D/A': [2, 6, 9], 'A': [9, 1, 4], 'A7': [9, 1, 4, 7], 'F#/A#': [6, 10, 1], 'F#7': [6, 10, 1, 4],
       'Bm': [11, 2, 6], 'E': [4, 8, 11], 'Em/D': [2, 4, 7]}   # Em/D: the passing chord under the G of bars 1/13
# ---- check the melody and the chords against the source
tut = SRC['tutti']; bari = SRC['baritone']
for k in range(1, 25):
    for pos, d, p in M[k]:
        rh = [e for e in tut if e[0] == k and abs(e[1] - pos) < 1e-6 and e[4] == 'RH']
        top = max(e[3] for e in rh) if rh else None
        # this source writes two repeated quarters of the tune as one half note in the piano part: accept a held top note
        held = [e for e in tut if e[0] == k and e[4] == 'RH' and e[1] < pos - 1e-6 and e[1] + e[2] > pos + 1e-6 and e[3] % 12 == p % 12]
        assert (top is not None and top % 12 == p % 12) or held, ('melody vs tutti', k, pos, nm(p))
        if k <= 16 and not (k == 14 and pos == 3):        # baritone sings the vocal variant G-E in bar 14 beat 4
            b = [e for e in bari if e[0] == k and abs(e[1] - pos) < 1e-6]
            assert b and b[0][3] % 12 == p % 12, ('melody vs baritone', k, pos, nm(p))
    for pos in range(4):
        notes = [e for e in tut if e[0] == k and e[1] <= pos + 1e-6 and e[1] + e[2] > pos + 1e-6]
        pcs = set(e[3] % 12 for e in notes)
        mel = [p % 12 for q, d, p in M[k] if q <= pos + 1e-6 and q + d > pos + 1e-6]
        ch = set(PCS[CH[k][pos]])
        extra = pcs - ch - set(mel)
        assert not extra, ('chord', k, pos, CH[k][pos], [NM[c] for c in pcs])
# bass: lowest note of the left hand sounding on beats 1 and 3, put into the bass register A1..G#2
BASSP = {}
for k in range(1, 25):
    for pos in (0, 2):
        lh = [e for e in tut if e[0] == k and e[4] == 'LH' and e[1] <= pos + 1e-6 and e[1] + e[2] > pos + 1e-6]
        pc = min(lh, key=lambda e: e[3])[3] % 12
        BASSP[(k, pos)] = 33 + (pc - 9) % 12
assert BASSP[(3, 2)] == 33           # bar 3 beat 3: the left hand has A3/E4 (no low note): the bass is A
# harmony pair voicings (two chord tones below / around the tune) and rain bells (chord tones A4..G#5)
VO = {'D': (57, 62), 'D/A': (57, 62), 'A': (57, 61), 'A7': (55, 61), 'F#/A#': (58, 61), 'F#7': (58, 64), 'Bm': (59, 62), 'E': (56, 59), 'Em/D': (55, 64)}
BELLP = {9: 69, 10: 70, 11: 71, 1: 73, 2: 74, 4: 76, 6: 78, 7: 79, 8: 80}


def theme_bar(k): return k if k <= 24 else k - 8


def t_of(k, pos):
    """bar 1..32, pos in quarters; the last bar slows down (rit.)"""
    tb = T0 + (k - 1) * BAR
    if k < 32: return tb + pos * Q
    # bar 32: beat 1 = 0.47 s, beat 2 = 0.56 s, then the final chord on beat 3
    if pos <= 1: return tb + pos * 0.47
    return tb + 0.47 + (pos - 1) * 0.56


lead = []                       # [t, pitch, vel, dur_s, bar, pos]
for k in range(1, 33):
    tb_ = theme_bar(k)
    for pos, d, p in M[tb_]:
        v = 70 + (8 if k >= 9 else 0) + (8 if k >= 17 else 0) + (10 if k >= 25 else 0)
        dur = t_of(k, pos + d) - t_of(k, pos) if k < 32 else (t_of(k, pos + d) - t_of(k, pos) if pos < 2 else 3.0)
        lead.append([round(t_of(k, pos), 5), p, v, round(dur, 4), k, pos])
FINAL = lead[-1][0]
assert lead[-1][4] == 32 and lead[-1][5] == 2
bass = []; pair = []; rain = []; chords = {}
for k in range(1, 33):
    tb_ = theme_bar(k)
    for pos in range(4):
        if k == 32 and pos == 3: continue
        ch = CH[tb_][pos]; t = round(t_of(k, pos), 5); chords[f'{k}-{pos}'] = ch
        if k >= 5 and pos in (0, 2): bass.append([t, BASSP[(tb_, pos)], k, pos])
        if k >= 9 and pos in (1, 3): pair.append([t, list(VO[ch]) if not (tb_ == 16 and pos == 3) else [52, 55], k, pos, ch])
        if k >= 25: rain.append([t, ch, sorted(BELLP[c] for c in PCS[ch]), k, pos])
# final chord (bar 32 beat 3): the pair rings D (A3 D4) too
pair.append([round(FINAL, 5), [57, 62], 32, 2, 'D'])
json.dump(lead, open(f'{HERE}/lead.json', 'w'))
json.dump({'t0': T0, 'q': Q, 'bar': BAR, 'final': FINAL, 'bass': bass, 'pair': pair, 'rain': rain, 'chords': chords,
           'bar_t': {k: round(t_of(k, 0), 5) for k in range(1, 33)}}, open(f'{HERE}/events.json', 'w'))
if __name__ == '__main__':
    print('melody notes', len(lead), 'bass', len(bass), 'pair', len(pair), 'rain beats', len(rain), 'final', round(FINAL, 3))
