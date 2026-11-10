"""Waltz of the Flowers (P. I. Tchaikovsky, The Nutcracker Op.71, 1892, public domain) - main waltz, D major, 3/4.
Source notes: score_src.json (CC0, see its 'source' field), original bar numbers 37-106:
  37 waltz vamp | 38-44 horn theme | 45-53 clarinet answer | 54-60 horn theme | 61-69 F#-minor clarinet episode |
  70-81 violin theme + 82-85 its 1st ending | 86-89 2nd ending (the repeat of 70-81 is skipped) | 90-96 horn theme |
  97-105 clarinet answer | 106 downbeat: the D major chord the answer resolves to = our final chord.
Own reduction (not copied from any arrangement):
  melody  = the top note of every source onset that belongs to the tune (accompaniment-only onsets listed in ACC)
  harmony = the other tune notes of the same onset (A3 and up) -> sound with that strike
  bass    = lowest note of the downbeat onsets (pizz. basses + cellos an octave up)
  chords  = pitch classes of the remaining accompaniment notes, re-voiced in one close position F#3..F4 (horns + violas)
Editorial choices:
  - bars 53/105 end of the clarinet run: E (tied) D# D C# B A# -> A, the reading of 5 of the 6 CC0 sources
    (score 6159042 alone has ... C B)
  - bars 73/77/81: the flute trill on beat 1 is reduced to its principal note (one strike per beat)
  - final: bar 106 beat 1 as a sustained D major tutti chord (D2 D3 A3 D4 F#4 A4 D5) with timpani roll.
Outputs notes.json (every sound), lead.json (one entry per strike), events.json."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(f'{HERE}/score_src.json'))['events']
Q = 0.30                       # quarter = 200 (dotted half = 66.7, a usual concert tempo for this waltz)
BAR = 3 * Q
T0 = 0.5                       # start of bar 37
B0 = 37
def tq(bar, off): return T0 + (bar - B0) * BAR + off * Q

def sect(b):
    if b <= 37: return 'vamp'
    if b <= 44: return 'horn1'
    if b <= 53: return 'dew1'
    if b <= 60: return 'horn2'
    if b <= 69: return 'cold'
    if b <= 85: return 'gold'
    if b <= 89: return 'climb'
    if b <= 96: return 'horn3'
    if b <= 105: return 'dew2'
    return 'coda'

ACC = {(61, 0), (63, 0), (70, 1), (71, 1), (75, 1), (76, 1), (78, 1), (79, 1), (86, 1), (87, 1), (53, 0), (105, 0)}
FIX = {(53, 2.0): 59, (53, 2.5): 58, (105, 2.0): 59, (105, 2.5): 58}        # majority reading (B3, A#3)
on = {}
for b, o, d, p in S:
    if b > 106 or (b == 106 and o > 0): continue
    if abs(o * 5 - round(o * 5)) < 1e-6 and abs(o - round(o * 2) / 2) > 1e-6: continue   # trill notes (fifths of a beat)
    on.setdefault((b, round(o, 4)), []).append((p, d))

lead, acc = [], []
for (b, o) in sorted(on):
    ps = on[(b, o)]; top = max(p for p, d in ps)
    s = sect(b)
    if b == 106:
        lead.append(dict(t=round(tq(b, o), 5), p=74, harm=[50, 57, 62, 66, 69], d=3 * Q, bar=b, beat=o, sec='coda'))
        acc.append((b, o, [38], 3.0)); continue
    tune = s != 'vamp' and top >= 57 and (b, o) not in ACC
    if tune:
        p = FIX.get((b, o), top)
        d = max(dd for pp, dd in ps if pp == top)
        if b in (73, 77, 81) and o == 0: d = 1.0                 # trill reduced to its principal note
        harm = sorted(pp for pp, dd in ps if 57 <= pp < p and p - pp <= 12 and pp % 12 != p % 12)[-2:]
        lead.append(dict(t=round(tq(b, o), 5), p=p, harm=harm, d=round(d * Q, 5), bar=b, beat=o, sec=s))
        rest = [pp for pp, dd in ps if pp < 57]
    else:
        rest = [pp for pp, dd in ps]
    if rest: acc.append((b, o, sorted(rest), max(dd for pp, dd in ps)))
FINAL = lead[-1]['t']

notes = []
def N(t, d, role, p, v): notes.append([round(t, 5), round(d, 5), role, int(p), int(max(1, min(127, v)))])

for l in lead:
    t, p, d, s = l['t'], l['p'], l['d'], l['sec']
    v = 100
    if s.startswith('horn'):
        N(t, d, 'key', p + 12, v - 20)                       # the struck flower bell (glock, an octave up)
        for q in [p] + l['harm']: N(t, d, 'hn', q, v)          # horn section
    elif s.startswith('dew'):
        N(t, d, 'leaf', p + 12, v)                           # struck leaf (marimba)
        N(t, d, 'key', p + 24, v - 45)
    elif s == 'cold':
        N(t, d, 'ice', p + 12, v - 5)                        # frosted bud (xylophone)
        N(t, d, 'vspic', p, v - 20)
    elif s in ('gold', 'climb'):
        N(t, d, 'key', p, v)                                 # golden bell flowers (glock)
        for q in [p] + l['harm']: N(t, d, 'vsus', q, v - 10)  # violins
    else:                                                    # final chord
        N(t, 1.0, 'key', p, 115)
        for q in [p] + l['harm']: N(t, 3.0, 'vsus', q + 12 if q < 60 else q, 100); N(t, 3.0, 'hnsus', q if q < 72 else q - 12, 105)

# waltz accompaniment: downbeat bass, after-beat chords re-voiced in close position
beats = []
for b, o, ps, d in acc:
    t = tq(b, o); s = sect(b); lo = 0.9 if s == 'cold' else 1.0
    if b == 106: continue
    if o == 0 or min(ps) < 45:
        bp = min(ps)
        N(t, max(d, 1) * Q, 'cb', bp, 104 * lo); N(t, Q, 'cpz', bp + 12, 84 * lo); beats.append([round(t, 4), 'bass'])
        others = sorted(set(p % 12 for p in ps if p % 12 != bp % 12))
    else:
        others = sorted(set(p % 12 for p in ps)); beats.append([round(t, 4), 'chord'])
    for pc in others:
        pp = 54 + (pc - 54) % 12                              # F#3..F4
        N(t, d * Q * 0.8, 'hnst', pp, 70 * lo); N(t, d * Q * 0.6, 'vla', pp + 12, 62 * lo)
# final chord: D major tutti + timpani roll + cymbal
for k in range(12): N(FINAL - 0.9 + k * 0.075, 0.1, 'timp', 38, 60 + 4 * k)
N(FINAL, 0.6, 'timp', 38, 122)
for p in (38, 50): N(FINAL, 3.0, 'cbsus', p, 110)
N(FINAL, 3.0, 'crash', 0, 105)

notes.sort(key=lambda n: n[0])
json.dump(notes, open(f'{HERE}/notes.json', 'w'))
json.dump(lead, open(f'{HERE}/lead.json', 'w'))
secs = []
for i, l in enumerate(lead):
    if not secs or secs[-1][0] != l['sec']: secs.append([l['sec'], l['t'], i])
json.dump({'t0': T0, 'q': Q, 'bar': BAR, 'final': FINAL, 'secs': secs, 'beats': sorted(beats),
           'bar_t': {str(b): round(T0 + (b - B0) * BAR, 4) for b in range(B0, 110)}}, open(f'{HERE}/events.json', 'w'), indent=0)
if __name__ == '__main__':
    print('lead', len(lead), 'final', round(FINAL, 2), 'secs', [(n, round(t, 2), i) for n, t, i in secs])
