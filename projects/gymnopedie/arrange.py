"""Gymnopedie No.1 - Erik Satie (1888), public domain.  Own arrangement for two marbles.
Source notes: score_src.json (made by extract_src.py from the Mutopia LilyPond file src_gymnopedie_1.ly, public domain,
typeset from the Dover reprint of the original edition).  Nothing is taken from any modern arrangement.
Cut (29 bars, 3/4, 'Lent et douloureux', quarter = 0.76 s):
  video bars  1- 2 = original bars  3- 4   introduction (bass + chord, G / D)
  video bars  3-29 = original bars 13-39   the theme (13-21), the second part (22-31), the first-ending close (32-39)
  (= the piece's first section without its first eight-bar statement 5-12, which bars 13-16 repeat note for note)
Roles:
  PEARL (melody, upper voice)   strikes one glass key per melody note; bars 38/39 its chord (4 notes) = chord plate / final cup
  NIGHT (bass ball)             beat 1 = the bass note on a bass bar, beat 2 = the chord (middle voice) on a chime plate;
                                original bar 37: chords on beats 2 and 3 (middle voice + the bass staff's inner voice B2, E3)
Outputs notes.json (every sound: [t, dur, role, midi, vel]), lead.json (pearl strikes), events.json (night strikes, grid)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(f'{HERE}/score_src.json'))
BEAT = 0.76
BAR = 3 * BEAT
T0 = 1.8                                   # video bar 1 downbeat (the night ball rolls out of the crown before it)
NB = 29
def orig(v): return v + 2 if v <= 2 else v + 10
VOF = {orig(v): v for v in range(1, NB + 1)}
def tq(v, beat): return round(T0 + (v - 1) * BAR + beat * BEAT, 5)

def dyn(v):                                # pp at the start, a little more in the second part, p at the close
    return 52 if v <= 2 else (60 if v < 12 else (66 if v < 22 else 62))

from collections import defaultdict
top = defaultdict(list); acc = defaultdict(list); bass = defaultdict(list)
for nm, dst in (('top', top), ('middle', acc), ('bottom', bass)):
    for bar, b, d, p in S[nm]:
        if bar in VOF: dst[(VOF[bar], b)].append((p, d))

# ---- pearl: melody strikes
lead = []                                  # [t, [pitches], vel, dur, kind, vbar, beat]
for (v, b) in sorted(top):
    ps = sorted(p for p, d in top[(v, b)]); d = max(d for p, d in top[(v, b)])
    kind = 'key' if len(ps) == 1 else ('chord' if v < NB else 'final')
    lead.append([tq(v, b), ps, dyn(v) + (4 if kind != 'key' else 0), round(d * BEAT, 5), kind, v, b])
lead.sort(key=lambda l: l[0])
FINAL = tq(NB, 0)
assert lead[-1][0] == FINAL

# ---- night: bass bar on beat 1, chord plate on the other onsets
night = []                                 # [t, [pitches], vel, dur, kind, vbar, beat]
on = defaultdict(list)
for k, v_ in bass.items(): on[k] += v_
for k, v_ in acc.items(): on[k] += v_
for (v, b) in sorted(on):
    ps = sorted(set(p for p, d in on[(v, b)])); d = max(d for p, d in on[(v, b)])
    if v == NB: kind = 'final'
    elif b == 0: kind = 'bass'
    else: kind = 'chord'
    night.append([tq(v, b), ps, dyn(v) - (6 if kind == 'chord' else 0), round(d * BEAT, 5), kind, v, b])
night.sort(key=lambda l: l[0])
assert night[-1][0] == FINAL

notes = []
def N(t, d, role, p, v): notes.append([round(t, 5), round(d, 5), role, int(p), int(max(1, min(127, v)))])
for t, ps, v, d, kind, vb, b in lead:
    for p in ps: N(t, d, 'mel' if kind == 'key' else ('melchord' if kind == 'chord' else 'melfinal'), p, v)
for t, ps, v, d, kind, vb, b in night:
    for p in ps: N(t, d, {'bass': 'bass', 'chord': 'chord', 'final': 'bassfinal'}[kind], p, v)
notes.sort(key=lambda n: (n[0], n[3]))
json.dump(notes, open(f'{HERE}/notes.json', 'w'))
json.dump(lead, open(f'{HERE}/lead.json', 'w'))
json.dump({'t0': T0, 'beat': BEAT, 'bar': BAR, 'nbars': NB, 'final': FINAL, 'night': night}, open(f'{HERE}/events.json', 'w'))
if __name__ == '__main__':
    NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    nm = lambda p: f'{NM[p % 12]}{p // 12 - 1}'
    for v in range(1, NB + 1):
        a = '  '.join(f'b{l[6]:g}:{"+".join(nm(p) for p in l[1])}' for l in lead if l[5] == v)
        c = '  '.join(f'b{l[6]:g}:{l[4][0]}:{"+".join(nm(p) for p in l[1])}' for l in night if l[5] == v)
        print(f'v{v:2d} (orig {orig(v):2d})  {a:40s} | {c}')
    print('pearl', len(lead), 'night', len(night), 'final', FINAL, 'dur ~', round(FINAL + 7.5, 2))
