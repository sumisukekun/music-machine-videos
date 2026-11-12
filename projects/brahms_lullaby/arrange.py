"""Brahms - Wiegenlied (Lullaby) Op.49 No.4 (1868), public domain. Own arrangement for three marble voices.
Score source: OpenScore Lieder corpus (CC0), score lc5701612 'Wiegenlied' (copy in score/), a transcription of
IMSLP #81909 = Johannes Brahms, Sämtliche Werke Bd. 24, Breitkopf & Härtel 1926-27 (public domain). Only Brahms' notes are
used: the voice line, the piano right hand (the rocking off-beat figure) and the piano left hand (E-flat pedal bass).
  key / metre : E-flat major, 3/4, 'Zart bewegt', played at quarter = 72
  form        : piano intro (bar 2 with its tied pick-up), the whole strophe (bars 3-18), then the second half once more
                (bars 11-18) as the lights go down; a final music-box chord (Eb5 G5 Bb5) is added after the last bar.
  seam        : the first bar 18 keeps the voice's Eb4 and takes the voice pick-up (Eb4 Eb4) and the piano of bar 14
                (identical harmony) so the music flows on into bar 11.
  reduction   : three-note right-hand chords lose one octave doubling ([Eb4 Db5 Eb5] -> [Db5 Eb5], final [G4 Bb4 Eb5] ->
                [Bb4 Eb5]); the grace note of bar 14 is left out; the grace pair of bar 17 (Bb4 Ab4) is written out as two
                sixteenths taken from the Ab4; tied right-hand notes are held (no new strike).
Voices -> balls:  V = voice (the star marble)   RH = piano right hand (the twins)   LH = piano left hand (one sheep per bar)
Outputs notes.json (every sound), lead.json (V), duo.json (RH), bass.json (LH), events.json (grid), score_src.json."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = f'{HERE}/score_src.json'


def extract():
    """music21 dump of the OpenScore file -> {part: {bar: [(beat, [midi...], beats, tie_stop)]}} (run once, needs music21)"""
    import music21 as m
    s = m.converter.parse(f'{HERE}/score/lc5701612.mxl')
    out = {}
    for name, p in zip(('V', 'RH', 'LH'), s.parts):
        bars = {}
        for meas in p.getElementsByClass('Measure'):
            ev = []
            for v in (meas.voices or [meas]):
                for n in v.notesAndRests:
                    if n.isRest: continue
                    ps = [x.midi for x in (n.pitches if n.isChord else [n.pitch])]
                    stop = bool(n.tie and n.tie.type in ('stop', 'continue'))
                    grace = float(n.quarterLength) == 0
                    ev.append([float(n.offset), ps, float(n.quarterLength), stop, grace])
            bars[int(meas.number)] = ev
        out[name] = bars
    json.dump(out, open(SRC, 'w'), indent=0)


if not os.path.exists(SRC): extract()
S = {k: {int(b): v for b, v in d.items()} for k, d in json.load(open(SRC)).items()}

Q = 60 / 72.0                 # quarter = 72
BAR = 3 * Q
T0 = 2.2                      # downbeat of the intro bar (bar 2); the right-hand pick-up sounds half a beat before it
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'


def voice_bar(k):
    ev = []
    for b, ps, d, stop, grace in S['V'][k]:
        if grace: continue
        ev.append((b, ps[0], d))
    if k == 17:                                   # Ab4 (Bb4 Ab4 graces) G4 F4 -> Ab4 8th, Bb4 16th, Ab4 16th, G4, F4
        ev = [(0.0, 68, 0.5), (0.5, 70, 0.25), (0.75, 68, 0.25)] + [e for e in ev if e[0] >= 1.0]
    return ev


def rh_bar(k):
    ev = []
    for b, ps, d, stop, grace in S['RH'][k]:
        if stop or grace: continue                # tied continuation: held, not struck again
        ps = sorted(ps)
        if len(ps) == 3: ps = ps[1:]              # drop the lowest octave doubling (Eb4 under Db5 Eb5 / G4 under Bb4 Eb5)
        ev.append((b, ps, d))
    return ev


def lh_bar(k): return [(b, sorted(ps), d) for b, ps, d, stop, grace in S['LH'][k] if not stop and not grace]


# ---- form: (pass, bar number, source bars for V / RH / LH)
FORM = [(1, 2)] + [(1, k) for k in range(3, 18)] + [(1, '18a')] + [(2, k) for k in range(11, 18)] + [(2, 18)]
def content(k):
    if k == '18a':                                # seam: voice Eb4 + pick-up Eb4 Eb4, piano of bar 14
        return [(0.0, 63, 2.0), (2.0, 63, 0.5), (2.5, 63, 0.5)], rh_bar(14), lh_bar(14)
    return voice_bar(k), rh_bar(k), lh_bar(k)


grid = []; t = T0
for ps_, k in FORM: grid.append((ps_, k, round(t, 5))); t += BAR
T_END = grid[-1][2]                               # last bar downbeat


def vel(ps_, k, b):
    base = 72 if ps_ == 1 else 64                 # the reprise is softer (the lights go down)
    if k in (11, 15): base += 8                   # 'morgen frueh' high point
    return base + (6 if b == 0 else 0)


lead, duo, bass = [], [], []
# right-hand pick-up of the intro ([G4 Bb4] on the last eighth of bar 1, tied into bar 2)
pk = [e for e in S['RH'][1] if abs(e[0] - 2.5) < 1e-6][0]
duo.append([round(T0 - 0.5 * Q, 5), sorted(pk[1]), 66, round(0.5 * Q, 5), 1, 1, 2.5])
for ps_, k, tb in grid:
    V, RH, LH = content(k)
    if k == 2: V = [e for e in V if e[0] >= 2.0]  # voice pick-up of the intro bar (G4 G4)
    for b, p, d in V: lead.append([round(tb + b * Q, 5), p, vel(ps_, k, b) + 10, round(d * Q, 5), ps_, k, b])
    for b, pp, d in RH: duo.append([round(tb + b * Q, 5), pp, vel(ps_, k, b) - 10, round(d * Q, 5), ps_, k, b])
    for b, pp, d in LH: bass.append([round(tb + b * Q, 5), pp, vel(ps_, k, b) - 2, round(d * Q, 5), ps_, k, b])

notes = []
def N(t, d, role, p, v): notes.append([round(t, 5), round(d, 5), role, int(p), int(max(1, min(127, v)))])
for t, p, v, d, *_ in lead: N(t, d, 'star', p, v)
for t, pp, v, d, *_ in duo:
    for p in pp: N(t, d, 'twins', p, v)
for t, pp, v, d, *_ in bass:
    for p in pp: N(t, d, 'sheep', p, v)
T_LAUNCH = round(T_END + Q, 5)                    # last bar beat 2: the giant sheep lands on the see-saw (LH chord)
FINAL = round(T_LAUNCH + 0.62, 5)                 # the star and the twins ring the mobile's moon and star bells
N(FINAL, 3.2, 'moon', 75, 112)                    # Eb5: the crescent moon bell
for p in (79, 82): N(FINAL, 3.0, 'twinbell', p, 100)   # G5, Bb5: the two star bells of the mobile
notes.sort(key=lambda n: n[0])
json.dump(notes, open(f'{HERE}/notes.json', 'w'))
json.dump(lead, open(f'{HERE}/lead.json', 'w')); json.dump(duo, open(f'{HERE}/duo.json', 'w')); json.dump(bass, open(f'{HERE}/bass.json', 'w'))
json.dump({'t0': T0, 'q': Q, 'bar': BAR, 'final': FINAL, 't_launch': T_LAUNCH, 't_end': T_END,
           'grid': [[a, str(b), c] for a, b, c in grid]}, open(f'{HERE}/events.json', 'w'))
if __name__ == '__main__':
    print('bars', len(grid), 'V', len(lead), 'RH', len(duo), 'LH', len(bass), 'final', FINAL, 'end of video ~', round(FINAL + 4.8, 1))
    for ps_, k, tb in grid:
        V, RH, LH = content(k)
        print(f'P{ps_} bar {k!s:>3} {tb:6.2f}s  V:', ' '.join(f'{b:g}:{nm(p)}' for b, p, d in V), ' | RH:', ' '.join(f'{b:g}:' + '+'.join(nm(p) for p in pp) for b, pp, d in RH),
              ' | LH:', ' '.join(f'{b:g}:' + '+'.join(nm(p) for p in pp) for b, pp, d in LH))
