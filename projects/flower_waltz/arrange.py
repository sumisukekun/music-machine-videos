"""Waltz of the Flowers (P. I. Tchaikovsky, The Nutcracker Op.71, 1892, public domain) - main waltz, D major, 3/4.
score_src.json = the notes of a standard piano reduction (bars 4-80: one bar of waltz vamp, horn theme, clarinet answers,
F#-minor episode, violin theme, horn theme again, short cadence).  Own orchestration.
Outputs notes.json (every sound), lead.json (one entry per onset = what a ball strikes), events.json."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(f'{HERE}/score_src.json'))
Q = 0.30                       # quarter = 200 (dotted half = 66.7, a usual concert tempo for this waltz)
BAR = 3 * Q
T0 = 0.5                       # start of bar 4
B0 = S['first_bar']
def tq(o): return T0 + o * Q
def bar_of(o): return B0 + int(o // 3 + 1e-6)

# sections (bar ranges) -> who plays the melody and with what sound
def sect(b):
    if b <= 4: return 'vamp'
    if b <= 11: return 'horn1'
    if b <= 20: return 'dew1'
    if b <= 27: return 'horn2'
    if b <= 36: return 'cold'
    if b <= 52: return 'gold'
    if b <= 56: return 'climb'
    if b <= 63: return 'horn3'
    if b <= 71: return 'dew2'
    return 'coda'

# ---- lead = one entry per RH onset (top pitch); lower RH notes of the same onset are the harmony of that strike
ons = {}
for o, d, ps in S['RH']:
    ons.setdefault(round(o, 4), []).append((d, ps))
lead = []
for o in sorted(ons):
    ps = sorted(p for d, pp in ons[o] for p in pp); d = max(d for d, pp in ons[o])
    b = bar_of(o); s = sect(b)
    if s == 'dew2' and False: pass
    lead.append(dict(t=round(tq(o), 5), p=ps[-1], harm=ps[:-1], d=round(d * Q, 5), bar=b, beat=round(o % 3, 4), sec=s))
# the coda's first note (bar 72 beat 1, the D that resolves the clarinet answer) belongs to the last water drop
for l in lead:
    if l['bar'] == 72 and l['beat'] == 0: l['sec'] = 'dew2'
FINAL = lead[-1]['t']

notes = []
def N(t, d, role, p, v): notes.append([round(t, 5), round(d, 5), role, int(p), int(max(1, min(127, v)))])

VOICE = {'horn1': 'hn', 'horn2': 'hn', 'horn3': 'hn', 'dew1': 'dew', 'dew2': 'dew', 'cold': 'ice', 'gold': 'vln', 'climb': 'vln', 'coda': 'tutti'}
for l in lead:
    t, p, d, s = l['t'], l['p'], l['d'], l['sec']
    v = 100 if s != 'coda' else 108
    if s.startswith('horn'):
        N(t, d, 'key', p + 12, v - 20)                       # the struck flower bell (glock, an octave up)
        for q in [p] + l['harm']: N(t, d, 'hn', q, v)          # horn section (melody + 2nd horn)
    elif s.startswith('dew'):
        N(t, d, 'leaf', p + 12, v)                           # struck leaf (marimba)
        N(t, d, 'key', p + 24, v - 45)
        for q in l['harm']: N(t, d, 'leaf', q + 12, v - 25)
    elif s == 'cold':
        N(t, d, 'ice', p + 12, v - 5)                        # frosted bud (xylophone)
        N(t, d, 'vspic', p, v - 20)
    elif s in ('gold', 'climb'):
        N(t, d, 'key', p, v)                                 # golden bell flowers (glock)
        for q in [p] + l['harm']: N(t, d, 'vsus', q, v - 10)  # violins
    else:
        N(t, d, 'key', p + 12 if p < 76 else p, v)
        for q in [p] + l['harm']: N(t, d, 'vsus', q, v - 5); N(t, d * 0.9, 'hnst', q - 12, v - 15)

# ---- waltz accompaniment (left hand): bass = pizz basses + cellos; after-beats = horns + violas
beats = []
for o, d, ps in S['LH']:
    t = tq(o); b = bar_of(o); s = sect(b); beat = round(o % 3, 4)
    lo = 1.0 if s not in ('cold',) else 0.9
    if s == 'coda': lo = 1.1
    for k, p in enumerate(sorted(ps)):
        if beat == 0 and k == 0 and p < 52:
            N(t, max(d, 1) * Q, 'cb', p, 104 * lo); N(t, Q, 'cpz', p + 12, 84 * lo); beats.append([round(t, 4), 'bass'])
        else:
            N(t, d * Q * 0.8, 'hnst', p, 70 * lo); N(t, d * Q * 0.6, 'vla', p + 12, 62 * lo)
    if not (beat == 0 and min(ps) < 52): beats.append([round(t, 4), 'chord'])
# final chord: D major tutti + timpani roll + cymbal
for k in range(12): N(FINAL - 0.9 + k * 0.075, 0.1, 'timp', 38, 60 + 4 * k)
N(FINAL, 0.6, 'timp', 38, 122)
for p in (38, 50): N(FINAL, 3.0, 'cbsus', p, 110)
for p in (62, 66, 69, 74): N(FINAL, 3.0, 'hnsus', p, 105)
for p in (74, 78, 81, 86): N(FINAL, 3.0, 'vsus', p, 100)
N(FINAL, 3.0, 'crash', 0, 110)
N(lead[-2]['t'], 0.6, 'timp', 33, 110)
for t in (tq((72 - B0) * 3 + 6), tq((73 - B0) * 3 + 6), tq((74 - B0) * 3 + 6), tq((75 - B0) * 3 + 6)): N(t, 0.4, 'tri', 0, 80)

notes.sort(key=lambda n: n[0])
json.dump(notes, open(f'{HERE}/notes.json', 'w'))
json.dump(lead, open(f'{HERE}/lead.json', 'w'))
secs = []
for i, l in enumerate(lead):
    if not secs or secs[-1][0] != l['sec']: secs.append([l['sec'], l['t'], i])
json.dump({'t0': T0, 'q': Q, 'bar': BAR, 'final': FINAL, 'secs': secs, 'beats': sorted(beats),
           'bar_t': {str(b): round(T0 + (b - B0) * BAR, 4) for b in range(B0, 82)}}, open(f'{HERE}/events.json', 'w'), indent=0)
if __name__ == '__main__':
    print('lead', len(lead), 'final', round(FINAL, 2), 'secs', [(n, round(t, 2), i) for n, t, i in secs])
