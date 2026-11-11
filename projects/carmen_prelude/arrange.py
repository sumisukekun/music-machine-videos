"""Carmen - Prelude, opening Allegro giocoso (Georges Bizet, 1875, public domain). Own arrangement.
Score source: Mutopia Project, BizetG/prelude (Mutopia-2005/11/22-635, 'Carmen - Prelude', piano reduction typeset by
Alex O'S from the Schirmer 1895 edition; Mutopia edition CC BY-SA 2.5). score/make_midi.py turns that file into
score/prelude.midi (repeats unfolded); this script reads ONLY the pitches and rhythms of the first part from it
(A theme 16 bars + its repeat, the F#-minor middle part 18 bars, the A theme again 16 bars = 66 bars, 2/4) and
writes its own orchestration, roles and ending.
Changes from the piano score (performance practice / arrangement):
  * the trills on B5 / G#5 (theme bars 4, 8, 12) are written out as measured 16ths main-upper-main-... + the printed
    turn (A#5 B5 / F#5 G#5), so that every note is a visible strike;
  * where the right hand plays a chord the melody balls strike two of its notes: the top note and its lower octave if
    the chord has it, else the next note below (the orchestra adds the other chord notes at the same instant);
  * the left hand is split into the bass (lowest note of single notes, octaves and root-position downbeats -> the bull)
    and the off-beat chords (top two notes, or the single off-beat note twice -> the two spectator balls in the stands);
  * the last chord (A major) is held as a final tutti.
Outputs notes.json (every sound), lead.json (melody events), events.json (bar grid, bass / chord / crowd events)."""
import json, os
import mido

HERE = os.path.dirname(os.path.abspath(__file__))
Q = 0.48                     # quarter note (= 125 / min, Allegro giocoso)
S16 = Q / 4; BAR = 2 * Q
T0 = 1.8                     # first downbeat (the toreros drop from the president's box before it)
NBARS = 66
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'
def sect(bar):
    if bar <= 16: return 'A1'
    if bar <= 32: return 'A2'
    if bar <= 40: return 'B1'
    if bar <= 46: return 'B2'
    if bar <= 50: return 'B3'
    return 'A3'
def tb(bar, pos): return T0 + (bar - 1) * BAR + pos * S16


def read_midi():
    m = mido.MidiFile(f'{HERE}/score/prelude.midi'); tpb = m.ticks_per_beat; tracks = []
    for tr in m.tracks:
        t = 0; on = {}; ns = []
        for msg in tr:
            t += msg.time
            if msg.type == 'note_on' and msg.velocity > 0: on[msg.note] = t
            elif msg.type in ('note_off', 'note_on') and msg.note in on: ns.append((on.pop(msg.note), t, msg.note))
        if ns: tracks.append(sorted(ns))
    out = []
    for ns in tracks:
        by = {}
        for a, b, p in ns:
            q = a / tpb * 4                                   # in sixteenths from the start
            bar = int(q // 8) + 1; pos = q - (bar - 1) * 8
            if bar > NBARS: continue
            by.setdefault((bar, round(pos, 3)), []).append((p, (b - a) / tpb * 4))
        out.append(by)
    return out                                                # [right hand, left hand]: {(bar, pos16): [(pitch, dur16)]}


RH, LH = read_midi()
assert max(b for b, _ in RH) == NBARS and max(b for b, _ in LH) == NBARS

# ---------------- melody (the two toreros) ----------------
TRILL_BARS = {4: (83, 85, 82), 8: (80, 81, 78), 12: (83, 85, 82)}       # theme bar: main, upper, turn
lead = []                    # [t, [pitches], vel, dur, bar, pos16, section]
crowd = []                   # tremolo marbles: [t, pitch, bar, pos16]
for (bar, pos), ns in sorted(RH.items()):
    s = sect(bar); tbar = {'A1': bar, 'A2': bar - 16, 'A3': bar - 50}.get(s)
    if tbar in TRILL_BARS:
        if pos > 0: continue                                  # the printed turn is part of the written-out trill
        mn, up, turn = TRILL_BARS[tbar]
        assert ns[0][0] == mn, (bar, ns)
        for k, p in enumerate((mn, up, mn, up, mn, up, turn, mn)):
            lead.append([round(tb(bar, k), 5), [p], 0, S16, bar, k, s])
        continue
    ps = sorted(set(p for p, d in ns)); dur = max(d for p, d in ns) * S16
    if s == 'B3':                                             # tremolo + the last two chords: the crowd wave
        if len(ps) == 1: crowd.append([round(tb(bar, pos), 5), ps[0], bar, pos])
        else:
            for p in ps[-1:-3:-1]: crowd.append([round(tb(bar, pos), 5), p, bar, pos])   # top two (G#5 + E5)
        continue
    top = ps[-1]
    if len(ps) == 1: pair = [top]
    elif top - 12 in ps: pair = [top, top - 12]
    else: pair = [top, ps[-2]]
    lead.append([round(tb(bar, pos), 5), pair, 0, round(dur, 5), bar, pos, s, ps])
for l in lead:
    if len(l) == 7: l.append(l[1])
VEL = {'A1': 92, 'A2': 100, 'B1': 84, 'B2': 98, 'A3': 108}
for l in lead:
    l[2] = VEL[l[6]] + (10 if l[5] == 0 else 0) + (6 if len(l[7]) > 2 else 0)
FINAL = tb(66, 4)
assert abs(lead[-1][0] - FINAL) < 1e-6

# ---------------- left hand: bass (the bull) and off-beat chords (spectators) ----------------
bass = []                    # [t, pitch, bar, pos16, [all lh pitches]]
chords = []                  # [t, [lo, hi], bar, pos16]
for (bar, pos), ns in sorted(LH.items()):
    ps = sorted(set(p for p, d in ns)); t = round(tb(bar, pos), 5)
    octave = len(ps) == 2 and ps[1] - ps[0] == 12
    if sect(bar) in ('A1', 'A2', 'A3') and len(ps) == 1 and pos in (2, 6) and bar not in (16, 32, 66):
        chords.append([t, [ps[0], ps[0]], bar, pos])                       # off-beat single notes (theme bars 13-14): 'pah'
    elif len(ps) == 1 or octave or ps[0] <= 45 or sect(bar) in ('B1', 'B2', 'B3'):
        bass.append([t, ps[0], bar, pos, ps])
        if bar in (16, 32, 66) and pos == 4: chords.append([t, ps[-2:], bar, pos])   # phrase ends / final: the spectators join in
    else:
        chords.append([t, ps[-2:], bar, pos])

notes = []
def N(t, d, role, p, v): notes.append([round(t, 5), round(d, 5), role, int(p), int(max(1, min(127, v)))])
for t, ps, v, d, bar, pos, s, full in lead:
    for p in ps: N(t, d, 'key', p, v if p == max(ps) else v - 12)            # the struck keys
    N(t, d * 0.9, 'vln', max(ps), v - 6)                                    # violins + flutes sing the tune
    if s in ('A1', 'A2', 'A3', 'B2'): N(t, d * 0.85, 'tpt', max(ps) - 12, v - 18)
    if s == 'A3': N(t, d * 0.85, 'vln8', max(ps) - 12, v - 22)
    if s in ('B1',): N(t, d * 0.95, 'vla', min(ps), v - 14)
    for p in full:
        if p not in ps and pos == 0: N(t, d, 'hnch', p - 12, v - 25)       # other chord notes: horns, same instant
for t, p, bar, pos, full in bass:
    s = sect(bar); v = {'A1': 96, 'A2': 102, 'B1': 92, 'B2': 104, 'B3': 112, 'A3': 110}[s]
    N(t, 0.4, 'drum', p, v)
    N(t, 0.35, 'cb', p if p >= 40 else p + 12, v - 4); N(t, 0.3, 'tuba', p if p >= 36 else p + 12, v - 14)
    if len(full) == 2 and full[1] - full[0] == 12: N(t, 0.3, 'cello', full[1], v - 10)
for t, ps, bar, pos in chords:
    s = sect(bar); v = {'A1': 88, 'A2': 96, 'A3': 104}[s]
    for p in ps: N(t, 0.25, 'bell', p, v); N(t, 0.22, 'hn', p, v - 12); N(t, 0.2, 'vlasp', p + 12, v - 16)
for t, p, bar, pos in crowd:
    N(t, 0.1, 'trem', p, 70 + 12 * (bar - 47) + 3 * pos)
# final chord: A major tutti, held
for p in (33, 45): N(FINAL, 3.0, 'tubasus', p + 12, 118)
for p in (57, 61, 64, 69): N(FINAL, 3.0, 'hnsus', p, 108)
for p in (69, 73, 76, 81): N(FINAL, 3.0, 'vsus', p, 106)
notes.sort(key=lambda n: n[0])
json.dump(notes, open(f'{HERE}/notes.json', 'w'))
json.dump(lead, open(f'{HERE}/lead.json', 'w'))
json.dump({'t0': T0, 'q': Q, 's16': S16, 'bar': BAR, 'final': FINAL, 'nbars': NBARS,
           'secs': {k: tb(b, 0) for k, b in (('A1', 1), ('A2', 17), ('B1', 33), ('B2', 41), ('B3', 47), ('A3', 51))},
           'bass': bass, 'chords': chords, 'crowd': crowd}, open(f'{HERE}/events.json', 'w'))
if __name__ == '__main__':
    from collections import Counter
    print('lead', len(lead), Counter(l[6] for l in lead), 'bass', len(bass), 'chords', len(chords), 'crowd', len(crowd),
          'final', round(FINAL, 3), 'dur', round(FINAL + 4.5, 2))
    print('min melody gap', min(b[0] - a[0] for a, b in zip(lead, lead[1:])), 'min bass gap', min(b[0] - a[0] for a, b in zip(bass, bass[1:])))
