"""Silent Night (Stille Nacht, heilige Nacht) - Franz Xaver Gruber, 1818 (public domain). Own arrangement for marbles.
Score source: score_src.json (PDMX 'Silent Night (SATB)', Public Domain Mark; see its 'source' field), melody cross-checked
against a second public-domain edition in PDMX (F. Vidotto). Taken from the source: the melody (pitch / rhythm), the bass note
on every dotted-quarter beat, and the two inner voices (alto + baritone) that sound on the 2nd and 5th eighth of every bar.
Written here: who plays what, the order of entries, the second verse with the joining balls and the ritardando.

Form: 6/8, C major. Verse 1 = bars 1-12, verse 2 = bars 13-24 (= the same 12 bars again), final chord on bar 24 beat 1.
  bar 1     the lantern ball alone: the melody                                              (1 ball)
  bar 5     + the snowball: the bass on both dotted-quarter beats (eighths 1 and 4)          (2)
  bar 9     + the two ornaments: the inner voices on eighths 2 and 5 (bass - chord, bass - chord)  (4)
  13-15     five children come out of their houses and sing the melody along, one per dotted beat (9)
  bar 23    ritardando; bar 24: the final chord, the great church bell
Outputs lead.json (melody notes) and events.json (bass, pair, chords, bar times)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = json.load(open(f'{HERE}/score_src.json'))
EI = 0.42                        # eighth note (dotted quarter = 47.6 bpm)
T0 = 1.7                         # bar 1, first eighth
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'
S = SRC['satb']; V = SRC['check_vidotto']
SOP = [e for e in S if e[4] == 'S']
# bar 12: this edition splits the last C into C(3/8) + C(2/8) for the words; Gruber's tune (and the Vidotto edition) holds one C
SOP = [e for e in SOP if not (e[0] == 12 and e[1] > 0)]; SOP[-1] = [12, 0.0, 6.0, 60, 'S']
# ---- the melody = the soprano of the source (both editions must agree, except the last bar's length)
for e in SOP:
    w = [x for x in V if x[4] == 'S' and x[0] == e[0] and abs(x[1] - e[1]) < 1e-6]
    assert w and w[0][3] == e[3] and (abs(w[0][2] - e[2]) < 1e-6 or e[0] == 12), ('melody differs', e, w)
assert len(SOP) == 46


def sounding(bar, pos, voice):
    n = [e for e in S if e[4] == voice and e[0] == bar and e[1] <= pos + 1e-6 and e[1] + e[2] > pos + 1e-6]
    assert len(n) == 1, (bar, pos, voice, n)
    return n[0][3]


def t_of(k, pos):
    """bar k = 1..24, pos in eighths. Bar 23 slows down (the eighth grows from 0.42 to 0.56 s); bar 24 starts after it."""
    if k <= 22:
        return T0 + ((k - 1) * 6 + pos) * EI
    tb = T0 + 22 * 6 * EI
    if k == 23:
        # eighth length grows linearly: e(u) = EI + (0.56 - EI) * u / 6 over u = 0..6
        return tb + EI * pos + (0.56 - EI) * pos * pos / 12
    return t_of(23, 6) + pos * 0.6


def vbar(k): return k if k <= 12 else k - 12


lead = []                         # [t, pitch, vel, dur_s, bar, pos]
for k in range(1, 25):
    for bar, pos, d, p, v in [e for e in SOP if e[0] == vbar(k)]:
        if k == 24: d = 6
        vel = 72 + (6 if vbar(k) >= 9 else 0) + (10 if k >= 13 else 0)
        dur = t_of(k, pos + d) - t_of(k, pos) if k < 24 else 3.5
        lead.append([round(t_of(k, pos), 5), p, vel, round(dur, 4), k, pos])
FINAL = lead[-1][0]
assert lead[-1][4] == 24 and lead[-1][5] == 0 and len(lead) == 92
bass = []; pair = []; chords = {}
def fold_bass(p): return 43 + (p % 12 - 7) % 12          # G2 .. F#3
for k in range(1, 25):
    for pos in (0, 3):
        if k == 24 and pos == 3: continue
        vb = vbar(k); t = round(t_of(k, pos), 5)
        pcs = sorted(set(sounding(vb, pos, vv) % 12 for vv in 'SATB'))
        chords[f'{k}-{pos}'] = [NM[c] for c in pcs]
        if k >= 5: bass.append([t, fold_bass(sounding(vb, pos, 'B')), k, pos])
    for pos in (1, 4):
        if k == 24: continue
        if k >= 9:
            vb = vbar(k)
            pair.append([round(t_of(k, pos), 5), sorted([sounding(vb, pos, 'T'), sounding(vb, pos, 'A')]), k, pos])
# the final chord: the ornaments ring the inner voices of bar 12 beat 1 with everybody else
pair.append([round(FINAL, 5), sorted([sounding(12, 0, 'T'), sounding(12, 0, 'A')]), 24, 0])
# sanity: the bass is the root of I / IV / V (or B = the 3rd of G7 in bar 9) and the pair sounds with the source's chord
for t, p, k, pos in bass:
    assert NM[p % 12] in chords[f'{k}-{pos}'], (k, pos, nm(p))
for t, ps, k, pos in pair:
    vb = vbar(k) if k < 24 else 12
    assert all(p in [sounding(vb, pos, vv) for vv in 'SATB'] for p in ps)       # exactly the source's inner voices
json.dump(lead, open(f'{HERE}/lead.json', 'w'))
json.dump({'t0': T0, 'eighth': EI, 'final': FINAL, 'bass': bass, 'pair': pair, 'chords': chords,
           'bar_t': {k: round(t_of(k, 0), 5) for k in range(1, 25)}}, open(f'{HERE}/events.json', 'w'))
if __name__ == '__main__':
    print('melody notes', len(lead), 'bass', len(bass), 'pair', len(pair), 'final', round(FINAL, 3))
    for k in range(1, 13):
        print(k, ' '.join(f"{nm(l[1])}@{l[5]:g}" for l in lead if l[4] == k), '| bass', ' '.join(nm(b[1]) for b in bass if b[2] == k),
              '| pair', ' '.join('+'.join(nm(p) for p in q[1]) + f'@{q[3]}' for q in pair if q[2] == k), '|', chords[f'{k}-0'], chords[f'{k}-3'])
