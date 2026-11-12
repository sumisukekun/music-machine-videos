"""Moonlight Sonata, 1st movement (L. van Beethoven, Piano Sonata No.14 in C# minor Op.27 No.2, 1801, first edition
Cappi, Vienna 1802 - public domain).  Own abridgement / arrangement for the marble machine.
Notes come from score_src.json (extract_src.py: Mutopia LilyPond transcription of a pre-1929 edition), copied note
for note - triplet arpeggios, melody and bass octaves exactly as printed; only whole bars are selected:
  bars 1-4 (introduction), 5-8 (= 42-45, the theme), 46-50 (the recapitulation's climb to the dominant),
  60 (tonic arrival, melody c#', the dotted g# figure in the bass), 67-69 (descending arpeggio and the two final chords).
Joins: 8 -> 46 is the score's own 45 -> 46 (bars 5-8 = 42-45 note for note); 50 -> 60 is the score's own 59 -> 60
(both end on the G#7 dominant); 60 -> 67 joins two tonic bars.  Bar 67's held low C# (tied over from 66) is re-struck.
Adagio sostenuto, quarter = 60 (alla breve half = 30), no rubato so every hit sits on the beat grid.
Outputs notes.json [t, dur, role, midi, vel], lead.json [t, midi, vel, dur] (melody), events.json."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(f'{HERE}/score_src.json'))
SEL = [1, 2, 3, 4, 5, 6, 7, 8, 46, 47, 48, 49, 50, 60, 67, 68, 69]
Q = 1.0; BAR = 4 * Q; T0 = 0.8
BT = {b: T0 + k * BAR for k, b in enumerate(SEL)}          # start time of each selected bar


def tm(bar, beat):
    # onsets on the 30 fps frame grid: beats and triplets fall on it exactly; the dotted figure's 16th moves by +1/60 s
    t = BT[bar] + beat * Q
    return round(round(t * 30 + 1e-6) / 30 if abs(t * 30 - round(t * 30)) > 1e-6 else t, 5)


# dynamics: pp throughout, cresc. bars 48-49 (score), decresc. 50, pp 60-69
VEL = {1: 40, 2: 40, 3: 41, 4: 42, 5: 42, 6: 42, 7: 43, 8: 43, 46: 44, 47: 45, 48: 49, 49: 54, 50: 50, 60: 42, 67: 38, 68: 36, 69: 36}
notes = []; lead = []; tri = []; bass = []; chord = []


def N(t, d, role, p, v): notes.append([round(t, 5), round(d, 5), role, int(p), int(v)])


# bass events (each one = one struck bowl / plate / key group), with pedal-change times
for b, beat, dur, p in S['bass']:
    if b not in BT: continue
    if b == 67 and beat >= 2 - 1e-6: tri.append([tm(b, beat), p]); continue    # bar 67 LH triplets join the arpeggio
    bass.append([tm(b, beat), p, dur])
bass.append([tm(67, 0), 37, 4.0])            # bar 67: the low C# pedal (tied over from bar 66 in the score), re-struck
for b, beat, dur, p in S['triplets']:
    if b in BT: tri.append([tm(b, beat), p])
for b, beat, dur, p in S['melody']:
    if b not in BT: continue
    if b in (68, 69): chord.append([tm(b, beat), p, dur]); continue          # RH final chords
    lead.append([tm(b, beat), p, dur])
tri.sort(); bass.sort(); lead.sort(); chord.sort()
# bars 68/69 bass chord notes go with the final chords
fin_t = sorted(set(t for t, p, d in chord))
for t, p, d in list(bass):
    if any(abs(t - f) < 1e-6 for f in fin_t): chord.append([t, p, d]); bass.remove([t, p, d])
chord.sort()
# pedal: sounds ring until the next change of bass (senza sordini, the pedal changes with the harmony)
ptimes = sorted(set([t for t, p, d in bass] + fin_t + [tm(69, 0) + 6.0]))


def ring(t):
    nxt = [x for x in ptimes if x > t + 1e-6]
    return min(nxt[0] - t + 0.15, 4.5) if nxt else 4.0


def barof(t): return max(b for b in SEL if BT[b] <= t + 1e-6)


for t, p in tri: N(t, ring(t), 'tri', p, VEL[barof(t)] + (4 if t == tri[0][0] else 0))
for t, p, d in lead: N(t, max(ring(t), d * Q), 'mel', p, VEL[barof(t)] + 22)
for t, p, d in bass: N(t, max(ring(t), d * Q), 'bass', p, VEL[barof(t)] + 12)
for t, p, d in chord: N(t, 6.5 if t >= tm(69, 0) - 1e-6 else 2.05, 'chord', p, VEL[barof(t)] + 10)
notes.sort(key=lambda n: (n[0], n[3]))
FINAL = tm(69, 0)
json.dump(notes, open(f'{HERE}/notes.json', 'w'))
json.dump([[t, p, VEL[barof(t)] + 22, d * Q] for t, p, d in lead], open(f'{HERE}/lead.json', 'w'))
bev = {}
for t, p, d in bass: bev.setdefault(t, []).append(p)
cev = {}
for t, p, d in chord: cev.setdefault(t, []).append(p)
json.dump({'t0': T0, 'q': Q, 'bar': BAR, 'final': FINAL, 'bars': {str(b): BT[b] for b in SEL}, 'sel': SEL,
           'tri': tri, 'bass_ev': [[t, sorted(ps)] for t, ps in sorted(bev.items())],
           'chords': [[t, sorted(ps)] for t, ps in sorted(cev.items())],
           'secs': [['intro', BT[1]], ['theme', BT[5]], ['climb', BT[46]], ['home', BT[60]], ['coda', BT[67]]]},
          open(f'{HERE}/events.json', 'w'), indent=0)
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
nm = lambda m: NM[m % 12] + str(m // 12 - 1)
print('notes', len(notes), 'tri', len(tri), 'lead', len(lead), 'bass events', len(bev), 'chords', len(cev), 'final', FINAL)
print('lead', [(round(t, 2), nm(p), d) for t, p, d in lead])
print('bass', [(round(t, 2), [nm(p) for p in ps]) for t, ps in sorted(bev.items())])
print('chords', [(round(t, 2), [nm(p) for p in ps]) for t, ps in sorted(cev.items())])
