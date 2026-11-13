"""We Wish You a Merry Christmas - traditional English (West Country) carol, public domain. Own arrangement for marbles.
Melody source: score_src.json = PDMX transcription 'Traditional music - We Wish You a Merry Christmas' (Public Domain Mark 1.0,
musescore 4812363), G major 3/4, cross-checked note for note with two more PDMX Public Domain Mark transcriptions (4676124, 4800948).
The source is the melody only. Written here: the harmony/bass (plain functional harmony, one bass note per bar, two in the cadence
bars), the form (three times through), who plays what, quarter = 0.42 s (~143 bpm) and a small ritardando in the last bar.

Form (3/4, one pass = pickup + 16 bars):
  verse   'We wish you ...' x3 + 'and a happy New Year'  = pickup .. bar 9 beat 1; four phrases, each ends on a doorbell
  refrain 'Good tidings ...'                              = bar 9 beat 3 .. bar 17 beat 1 (22 notes)
Roles: the caroler (red marble) plays every verse phrase on stepping-stone keys, the last note of each phrase is a house's doorbell
(4 houses per pass, 12 in all); the refrain is played by 22 snowballs dropping off a roof onto bells; the figgy-pudding ball plays the
bass in passes 2 and 3; the very last note (G) is the caroler landing on the great bell on top of the town-square tree.
Outputs lead.json (melody notes) and events.json (bass, bar times, chords)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = json.load(open(f'{HERE}/score_src.json'))
Q = 0.42                    # quarter note (s)
T0 = 2.42                   # global bar 1 (= source bar 2), beat 1; the first pickup sounds at T0 - Q
NPASS = 3
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'
MEL = SRC['melody_g']
assert MEL == SRC['check_4676124'] == SRC['check_4800948'] and len(MEL) == 52
LAST_GB = 16 * NPASS          # 48: the final 'Year'

# harmony (my own): chord per source bar (and beat 2 in the two cadence bars); bass note sounding the chord root
CHORD = {2: 'G', 3: 'C', 4: 'A', 5: 'D', 6: 'B7', 7: 'Em', 8: ('C', 'D'), 9: 'G',
         10: 'G', 11: 'D', 12: 'Em', 13: 'D', 14: 'G', 15: 'G/B', 16: ('C', 'D'), 17: 'G'}
TONES = {'G': 'G B D', 'C': 'C E G', 'A': 'A C# E', 'D': 'D F# A', 'B7': 'B D# F# A', 'Em': 'E G B', 'G/B': 'G B D'}
BASSP = {'G': 43, 'C': 48, 'A': 45, 'D': 50, 'B7': 47, 'Em': 52, 'G/B': 47}     # G2 C3 A2 D3 B2 E3 B2


def t_of(gb, pos):
    """global bar gb (1..48), pos in quarters; the last two bars slow down a little"""
    if gb < LAST_GB - 1: return T0 + ((gb - 1) * 3 + pos) * Q
    tb = T0 + (LAST_GB - 2) * 3 * Q
    if gb == LAST_GB - 1: return tb + {0: 0.0, 1: 0.45, 2: 0.95}.get(pos, 0.95 + (pos - 2) * 0.52)
    return tb + 1.5 + pos * 0.6


def gbar(p, b):
    return (16 * p, 2.0) if b == 1 else (16 * p + b - 1, None)


PHRASE_START = [(1, 0.0), (3, 2.0), (5, 2.0), (7, 2.0), (9, 2.0)]       # (source bar, pos): phrase 0..3 of the verse, 4 = refrain
def phrase_of(b, pos):
    k = 0
    for j, (bb, pp) in enumerate(PHRASE_START):
        if (b, pos) >= (bb, pp): k = j
    return k


lead = []        # [t, pitch, vel, dur_s, global bar, pos, role, pass, house (carol) / refrain (snow), source bar]
for p in range(NPASS):
    for b, pos, d, pitch in MEL:
        gb, ppos = gbar(p, b)
        pos_ = ppos if ppos is not None else pos
        if gb > LAST_GB: continue
        ph = phrase_of(b, pos)
        role = 'carol' if ph < 4 else 'snow'
        if gb == LAST_GB: role = 'carol'                    # the final 'Year' = the caroler on the great bell
        vel = 76 + 6 * p + (4 if role == 'snow' else 0)
        t = t_of(gb, pos_)
        dur = (t_of(gb, pos_ + d) - t) if gb < LAST_GB else 3.6
        idx = 4 * p + ph if ph < 4 else p
        lead.append([round(t, 5), pitch, vel, round(dur, 4), gb, pos_, role, p, idx, b])
lead.sort(key=lambda l: l[0])
FINAL = lead[-1][0]
assert lead[-1][4] == LAST_GB and lead[-1][1] == 67

bass = []; chords = {}
for gb in range(1, LAST_GB + 1):
    b = (gb - 1) % 16 + 2
    ch = CHORD[b]
    chords[gb] = ch if isinstance(ch, str) else list(ch)
    if gb < 17: continue                                      # the pudding joins in pass 2
    if isinstance(ch, str): bass.append([round(t_of(gb, 0), 5), BASSP[ch], gb, 0, ch])
    else:
        bass.append([round(t_of(gb, 0), 5), BASSP[ch[0]], gb, 0, ch[0]]); bass.append([round(t_of(gb, 1), 5), BASSP[ch[1]], gb, 1, ch[1]])
# harmony check: every melody note on beat 1 (and on the chord-change beat 2 of the cadence bars) is a tone of its chord;
# every bass note is the chord's root (or the 3rd for G/B)
for l in lead:
    if l[4] < 1: continue
    ch = chords[l[4]]
    if l[5] == 0: c_ = ch if isinstance(ch, str) else ch[0]
    elif l[5] == 1 and not isinstance(ch, str): c_ = ch[1]
    else: continue
    assert NM[l[1] % 12] in TONES[c_].split(), ('melody vs chord', l, c_)
for t, pb, gb, pos, c_ in bass: assert NM[pb % 12] == TONES[c_].split()[0] or c_ == 'G/B'
json.dump(lead, open(f'{HERE}/lead.json', 'w'))
json.dump({'t0': T0, 'q': Q, 'final': FINAL, 'bass': bass, 'chords': chords,
           'bar_t': {gb: round(t_of(gb, 0), 5) for gb in range(1, LAST_GB + 1)}}, open(f'{HERE}/events.json', 'w'))
if __name__ == '__main__':
    print('melody notes', len(lead), 'carol', sum(l[6] == 'carol' for l in lead), 'snow', sum(l[6] == 'snow' for l in lead),
          'bass', len(bass), 'final', round(FINAL, 3))
    for gb in range(0, 17):
        print(gb, ' '.join(f"{nm(l[1])}@{l[5]:g}{'*' if l[6] == 'snow' else ''}" for l in lead if l[4] == gb), '|', chords.get(gb, ''))
