"""Symphony No. 40 in G minor K.550, 1st movement (Molto allegro) - W. A. Mozart, 1788, public domain. Own reduction.
Score source: PDMX (CC0) QmNsmhXwgGBZiCRTBFVa4VdAbiwuHQcY6SZxUT3WZaBnq3 'Mozart: Symphony No. 40 in G Minor K. 550: I. Molto
Allegro' (an engraving of the full orchestral score), cross-checked against a second, independent CC0 PDMX engraving
(QmebfX6jV6E3r8pFKgNenTBxffNFJ6Z2Lriwy12MqfkK4A, the original version without clarinets): violin I, violas, cellos agree in
bars 1-59 and in the coda. score_src.json = the bars used, extracted from the first file. Only Mozart's notes are used.
  form   : bars 1-42, bar 43 (the general pause, silence), then the last 7 bars of the movement (293-299, the final cadence)
  voices : MEL = violin I (violin II doubles it an octave lower where it does), played by the two chasing balls:
                 red 'A' and blue 'B' take turns phrase by phrase (and note by note in the fast exchanges);
                 bars 14-16 the flute's answer is played by B while A holds violin I's long notes;
                 bars 16-20 (violin I only repeats the viola's D) both balls play the oboes' thirds (G-Bb / F#-A);
                 double stops: bar 28 D5+Bb5, bar 42 A3+F4, bar 298 the two upper notes of each triple stop.
           ACC = violas (the night train on the elevated rail: every eighth a rail bar, dyads on split bars)
           BASS = cellos with the contrabass octave below (the street ball)
Outputs notes.json (every sound), mel.json, acc.json, bass.json, events.json."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(f'{HERE}/score_src.json'))['rows']
Q = 0.29                     # quarter (half = 103)
BAR = 4 * Q
T0 = 1.7                     # bar 1 downbeat
SEQ = list(range(1, 44)) + list(range(293, 300))       # bar 43 = general pause
BT = {m: round(T0 + i * BAR, 5) for i, m in enumerate(SEQ)}
FINAL = BT[299]
def tb(m, b): return round(BT[m] + b * Q, 5)
def part(nm): return [r for r in S if r[4] == nm]
V1, V2, VLA, VC, CB, FL, OB = (part(x) for x in ('V1', 'V2', 'Vla', 'Vc', 'Cb', 'Fl', 'Ob1'))
def pos(m, b): return m * 100 + b                       # comparable bar/beat position


def owner(m, b):
    """which ball plays the violin I note at bar m beat b (quarters from 0)"""
    p = pos(m, b)
    rngB = [(1, 3, 5, 2), (9, 3, 11, 2), (20, 3, 24, 2)]
    rngA = [(5, 3, 9, 2), (11, 3, 16, 0.5), (24, 3, 28, -1), (29, 0, 29, 3), (31, 0, 31, 3), (33, 0, 33, 3), (36, 0, 38, 4)]
    for a, ab, c, cb in rngB:
        if pos(a, ab) <= p < pos(c, cb): return 'B'
    for a, ab, c, cb in rngA:
        if pos(a, ab) <= p < pos(c, cb): return 'A'
    if (m, b) in ((29, 3.5), (31, 3.5)) or m in (30, 32) or (m == 33 and b == 3.5) or m in (34, 35): return 'B'
    if m == 39: return 'B'
    if m in (40, 41) or m in (293, 294, 295, 296):
        if m == 40 and b < 2: return 'A'
        return 'A' if abs(b - round(b)) < 1e-6 else 'B'
    if m in (42, 297): return 'A'
    raise KeyError((m, b))


mel = []          # [t, pitch, vel, dur, ball, bar, beat, doubles(list of extra pitches sounding with it), src]
def vel_of(m):
    if 28 <= m <= 33: return 112
    if m in (34, 35, 36, 37, 38, 39, 40, 41, 42): return 104
    if m >= 293: return 116
    return 84
for m, b, d, p, nm in V1:
    if 16 <= m <= 20 and not (m == 20 and b >= 3) and not (m == 16 and b == 0): continue     # violin I doubles the viola's pedal D: the oboes' thirds instead
    if m == 43: continue
    on = [r for r in V1 if r[0] == m and abs(r[1] - b) < 1e-6]
    if len(on) >= 2:                                            # double / triple stops
        ps = sorted(r[3] for r in on)
        if p != ps[-1]: continue
        top, sec = ps[-1], ps[-2]
        dbl = [x[3] for x in V2 if x[0] == m and abs(x[1] - b) < 1e-6 and x[3] not in ps]
        mel.append([tb(m, b), top, vel_of(m), d * Q, 'A', m, b, [], 'V1'])
        mel.append([tb(m, b), sec, vel_of(m), d * Q, 'B', m, b, [], 'V1'])
        continue
    dbl = [x[3] for x in V2 if x[0] == m and abs(x[1] - b) < 1e-6 and x[3] == p - 12]
    if m == 299:
        mel.append([tb(m, b), p, 120, 3.0, 'AB', m, b, [], 'V1']); continue
    mel.append([tb(m, b), p, vel_of(m), d * Q, owner(m, b), m, b, dbl, 'V1'])
# flute answer (bars 14-16) by B
for m, b, d, p, nm in FL:
    if (m == 14 and b >= 1) or m == 15 or (m == 16 and b == 0):
        mel.append([tb(m, b), p, 84, d * Q, 'B', m, b, [], 'Fl'])
# oboes' thirds, bars 16-20 (A upper, B lower)
for m, b, d, p, nm in OB:
    if 16 <= m <= 20 and not (m == 16 and b < 2):
        on = sorted(r[3] for r in OB if r[0] == m and abs(r[1] - b) < 1e-6)
        mel.append([tb(m, b), p, 80, d * Q, 'A' if p == on[-1] else 'B', m, b, [], 'Ob'])
mel.sort(key=lambda r: (r[0], r[4]))

acc = []          # viola: [t, [pitches], vel, dur, bar, beat]
grp = {}
for m, b, d, p, nm in VLA:
    if m == 43: continue
    grp.setdefault((m, b), []).append((p, d))
for (m, b), ps in sorted(grp.items(), key=lambda kv: BT[kv[0][0]] + kv[0][1] * Q):
    acc.append([tb(m, b), sorted(p for p, d in ps), vel_of(m) - 14, max(d for p, d in ps) * Q, m, b])
bass = []         # cello (+ contrabass octave): [t, pitch, vel, dur, bar, beat, cb_pitch]
for m, b, d, p, nm in VC:
    if m == 43: continue
    cb = [x[3] for x in CB if x[0] == m and abs(x[1] - b) < 1e-6]
    bass.append([tb(m, b), p, vel_of(m) - 4, d * Q, m, b, cb[0] if cb else None])
bass.sort(key=lambda r: r[0])

notes = []
def N(t, d, role, p, v): notes.append([round(t, 5), round(d, 5), role, int(p), int(max(1, min(127, v)))])
for t, p, v, d, who, m, b, dbl, src in mel:
    N(t, d, 'mel' + ('' if src == 'V1' else '_' + src.lower()), p, v)
    for q in dbl: N(t, d, 'mel2', q, v - 10)
for t, ps, v, d, m, b in acc:
    for p in ps: N(t, d, 'acc', p, v)
for t, p, v, d, m, b, cb in bass:
    N(t, d, 'bass', p, v)
    if cb is not None: N(t, d, 'cb', cb, v)
# final chord (bar 299): the two balls ring the great rooftop bell (G3 = violin I's last note); the orchestra holds G minor
for p in (31, 43, 55, 62, 67, 70, 74, 79): N(FINAL, 3.2, 'tutti', p, 112)
notes.sort(key=lambda n: n[0])
GP = BT[43]
FORTE = BT[28]
json.dump(notes, open(f'{HERE}/notes.json', 'w'))
json.dump(mel, open(f'{HERE}/mel.json', 'w')); json.dump(acc, open(f'{HERE}/acc.json', 'w')); json.dump(bass, open(f'{HERE}/bass.json', 'w'))
json.dump({'t0': T0, 'q': Q, 'bar': BAR, 'final': FINAL, 'bt': {str(k): v for k, v in BT.items()}, 'gp': GP, 'forte': FORTE, 'seq': SEQ},
          open(f'{HERE}/events.json', 'w'))
if __name__ == '__main__':
    NM = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    nm = lambda p: f'{NM[p % 12]}{p // 12 - 1}'
    print('mel', len(mel), 'A', sum(1 for r in mel if r[4] == 'A'), 'B', sum(1 for r in mel if r[4] == 'B'), 'acc', len(acc), 'bass', len(bass))
    print('final', FINAL, 'gp', GP, 'forte', FORTE, 'end', FINAL + 4.3)
    import sys
    if len(sys.argv) > 1:
        for r in mel: print(f"{r[0]:7.3f} bar{r[5]:3d} b{r[6]:<5g} {r[4]:2s} {nm(r[1]):4s} {r[8]} {'+' + ','.join(nm(x) for x in r[7]) if r[7] else ''}")
