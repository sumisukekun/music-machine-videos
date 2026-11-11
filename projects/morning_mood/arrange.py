"""Morning Mood (Morgenstemning) - Edvard Grieg, Peer Gynt Suite No.1 Op.46 No.1 (1875/1888), public domain. Own arrangement.
Source notes: score_src.json (CC0, see its 'source' field; made by mm_tools/extract.py from three PDMX CC0 scores).
Bars 1-23 of the original, E major 6/8, then an own 4-bar-cadence ending (bar 24 = the melody of bar 4 in the tutti register,
B G# F# E, the E held as the final chord).
  bars  1- 4  flute    theme in E                 bars  5- 8  oboe     the answer an octave lower, turning to G# (= Ab)
  bars  9-12  flute    theme in G#                bars 13-16  oboe     answer, turning to B
  bars 17-20  flute    the calls get shorter (17 high, 18 the oboe's low octave, 19-20 the falling B-major arpeggio)
  bars 21-24  tutti    the theme in octaves (flute ball = upper octave, oboe ball = lower octave) -> sunrise
Melody = score 5886802 (graces dropped). Harmony = score 5842106 piano part (+1 semitone, bar n+1 -> n): every onset of the
accompaniment becomes one 'chime gate': chords = bass + the top three notes (2-note onsets kept as they are), the moving
inner line of bars 4/8/12/16 = single chimes. Editorial: bar 15 beat 4 is C#m (C#3 G#3 E4, as the CC0 piano score 4812602;
5842106 has the bare fifth C#3 G#3 C#4); bar 24 = E major on beat 1 and the final E major chord with the last melody note.
Outputs notes.json (every sound), lead.json (melody strikes), events.json (bar grid, chime gates)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(f'{HERE}/score_src.json'))
E8 = 0.42                     # eighth note (dotted quarter = 47.6, Allegretto pastorale)
BAR = 6 * E8
T0 = 1.8                      # bar 1 downbeat (the morning star falls 1 s before)
def tq(bar, off): return round(T0 + (bar - 1) * BAR + off * E8, 5)
VOICE = {}
for b in range(1, 25):
    VOICE[b] = 'fl' if b in (1, 2, 3, 4, 9, 10, 11, 12, 17, 18, 19, 20) else ('ob' if b <= 16 else 'tutti')

mel = [e for e in S['melody'] if e[0] <= 23]
mel += [[24, 0, 1, 83], [24, 1, 1, 80], [24, 2, 1, 78], [24, 3, 3, 76]]          # own ending: B G# F# E (bar 4 tune)
lead = []                     # [t, [pitches], vel, dur, voice, bar, offset]
for bar, off, dur, p in mel:
    v = VOICE[bar]
    vel = 64 + 2 * bar if bar < 21 else 104
    if bar >= 17 and bar < 21: vel = 80 + 4 * (bar - 17)
    ps = [p] if v != 'tutti' else [p, p - 12]
    lead.append([tq(bar, off), ps, vel, round(dur * E8, 5), v, bar, off])
lead.sort(key=lambda l: l[0])
FINAL = lead[-1][0]

# accompaniment onsets -> chime gates
from collections import defaultdict
on = defaultdict(list)
for bar, off, dur, p in S['harmony']:
    on[(bar, off)].append((p, dur))
on[(15, 3.0)] = [(49, 3), (56, 3), (64, 3)]                                    # C#m (see docstring)
on[(24, 0.0)] = [(40, 3), (64, 3), (68, 3), (71, 3)]
on[(24, 3.0)] = [(40, 6), (52, 6), (64, 6), (68, 6), (71, 6)]                     # final chord
gates = []                    # [t, [pitches low..high], dur, bar, off, kind]
keys = sorted(on)
for j, k in enumerate(keys):
    ps = sorted(set(p for p, d in on[k]))
    t = tq(*k)
    nxt = tq(*keys[j + 1]) if j + 1 < len(keys) else t + 3.0
    if len(ps) >= 3: kind = 'chord'; ps = [ps[0]] + ps[-3:] if len(ps) > 4 else ps
    elif len(ps) == 2: kind = 'chord'
    else: kind = 'single'
    if k == (24, 3.0): kind = 'final'
    gates.append([t, ps, round(max(0.4, nxt - t), 5), k[0], k[1], kind])

notes = []
def N(t, d, role, p, v): notes.append([round(t, 5), round(d, 5), role, int(p), int(max(1, min(127, v)))])
for t, ps, v, d, vo, bar, off in lead:
    if vo == 'fl': N(t, d, 'fl', ps[0], v)
    elif vo == 'ob': N(t, d, 'ob', ps[0], v)
    else: N(t, d, 'fl', ps[0], v); N(t, d, 'ob', ps[1], v - 8)
for t, ps, d, bar, off, kind in gates:
    loud = 70 + 2 * bar if bar < 21 else 100
    if kind == 'final': loud = 116
    for p in ps: N(t, d, 'chime' if kind != 'final' else 'finchime', p, loud if kind != 'single' else loud - 10)
notes.sort(key=lambda n: n[0])
json.dump(notes, open(f'{HERE}/notes.json', 'w'))
json.dump(lead, open(f'{HERE}/lead.json', 'w'))
json.dump({'t0': T0, 'e8': E8, 'bar': BAR, 'final': FINAL, 'gates': gates,
           'phrases': {k: [tq(b, 0) for b in range(1, 25) if VOICE[b] == k] for k in ('fl', 'ob', 'tutti')}},
          open(f'{HERE}/events.json', 'w'))
if __name__ == '__main__':
    print('lead', len(lead), 'gates', len(gates), 'final', FINAL, 'dur', round(FINAL + 4.5, 2))
