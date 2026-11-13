"""Tchaikovsky - Dance of the Sugar Plum Fairy (Danse de la Fee-Dragee), The Nutcracker Op.71 No.14c / Suite Op.71a No.3
(composed 1891-92, first published by P. Jurgenson, Moscow 1892 - public domain).  Own reduction for four marble voices.

Score sources (copies in score/):
  score/pdmx_5606908_orch_CC0.json      PDMX (CC0 1.0) MuseScore transcription of the original full orchestral score
                                         (Jurgenson 1892).  Its celesta part is stored one octave too high (written+24):
                                         everything taken from it is lowered by an octave (checked against the two below).
  score/pdmx_36758_orch_PDmark.json     PDMX (Public Domain Mark) second, independent transcription of the full score
                                         (celesta and pizzicato voicings, octave of the celesta).
  score/pdmx_1635221_esipoff_piano.json PDMX (CC0) transcription of Stepan Esipoff's piano solo reduction (Augener, c.1900,
                                         public domain) - harmony cross-check; bars 17-20 left hand follow it.
  Only Tchaikovsky's notes are used; nothing is copied from any modern arrangement.  score_check.py compares every note
  written below with the sources.

Key E minor, 2/4.  Andante ma non troppo, played at quarter = 70; Presto at quarter = 170.
Form (score bar numbers):  1-4 pizzicato intro | 5-12 theme | 17-20 second half of the repeat | bar 32 beat 2 pick-up +
  33-36 celesta cadenza (64th-note arpeggios) | 5-12 theme again (= bars 37-44, the reprise) | 49-52 | Presto: bars 81-84
  (the last four bars, bar 84 = two bars + the final E-minor chord).
Voices -> marbles:
  F  fairy  (celesta melody, top line)                     glass keys that light up and stay lit
  M  mint   (the off-beat pizzicato chords / wind chords)   peppermint chord discs
  C  cocoa  (the pizzicato bass, bass-clarinet runs)       chocolate drums
  R  rain   (celesta cadenza 64ths and the Presto 16ths)   sugar pearls dropping onto crystal keys (one pearl = one note)
Outputs notes.json (every sound), fairy.json, mint.json, cocoa.json, rain.json, events.json."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))

QA = 60 / 70.0                 # andante quarter
QP = 60 / 170.0                # presto quarter
T0 = 1.2                       # downbeat of bar 1
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'
def P(s):
    """'F#6' -> midi"""
    n = s[:-1]; o = int(s[-1]); return 12 * (o + 1) + NM.index(n)
def chord(s): return sorted(P(x) for x in s.split())

# ---------------------------------------------------------------- the bars (time unit = 32nd note, a 2/4 bar = 16)
# F: melody tokens 'G6:2' (pitch:length) or 'r:4';  M / C: [(offset, 'pitches', length)];  R: [(offset, 'pitches')]
B = {}
def bar(k, F=None, M=(), C=(), R=(), L=16): B[k] = dict(F=F, M=list(M), C=list(C), R=list(R), L=L)

INTRO1 = dict(M=[(4, 'G3 B3 E4', 4), (12, 'A3 C4 F#4', 4)], C=[(0, 'E2 E3', 4), (8, 'E2 E3', 4)])
INTRO2 = dict(M=[(4, 'A#3 C#4 G4', 4), (12, 'A3 C4 D#4', 4)], C=[(0, 'E2 E3', 4), (8, 'E2 E3', 4)])
bar(1, **INTRO1); bar(2, **INTRO2); bar(3, **INTRO1); bar(4, **INTRO2)
# --- the theme (bars 5-12)
bar(5, F='r:4 G6:2 E6:2 G6:4 F#6:4', **INTRO1)
bar(6, F='D#6:4 E6:4 D6:2 D6:2 D6:4', M=[(4, 'A#3 C#4 G4', 4), (12, 'B3 D4 G#4', 4)], C=[(0, 'E2 E3', 4), (8, 'E2 E3', 4)])
bar(7, F='C#6:2 C#6:2 C#6:4 C6:2 C6:2 C6:4', M=[(4, 'C#4 E4 A#4', 4), (12, 'D#4 F#4 C5', 4)], C=[(0, 'E2 E3', 4), (8, 'E2 E3', 4)])
bar(8, F='B5:2 E6:2 C6:2 E6:2 B5:4 r:4', M=[(0, 'B3 G4 B4', 4), (4, 'A3 F#4 C5', 4), (8, 'B3 G4 B4', 4)],
    C=[(0, 'E3 E4', 4), (4, 'E3 E4', 4), (8, 'E3 E4', 4), (12, 'E3', 1), (13, 'D3', 1), (14, 'C3', 1), (15, 'B2', 1)])
bar(9, F='r:4 G5:2 E5:2 G5:4 F#5:4', M=[(4, 'E3 G3 C4', 4), (12, 'F#3 C4 D#4', 4)], C=[(0, 'A#1 A#2', 4), (8, 'A1 A2', 4)])
bar(10, F='C6:4 B5:4 G6:2 G6:2 G6:4', M=[(4, 'E3 G3 B3', 4), (12, 'F#3 A#3 C#4 E4', 4)], C=[(0, 'G1 G2', 4), (8, 'F#1 F#2', 4)])
bar(11, F='F#6:2 F#6:2 F#6:4 E6:2 E6:2 E6:4', M=[(4, 'G#3 B3 D4 F#4', 4), (12, 'F#3 A#3 C#4 E4', 4)], C=[(0, 'F#2 F#3', 4), (8, 'F#1 F#2', 4)])
bar(12, F='D#6:2 F#6:2 E6:2 F#6:2 D#6:4 r:4', M=[(0, 'F#3 B3 D#4', 4), (4, 'F#3 A#3 E4', 4), (8, 'F#3 B3 D#4', 4)],
    C=[(0, 'B1 B2', 4), (4, 'C2 C3', 4), (8, 'B1 B2', 4), (12, 'B2', 1), (13, 'A2', 1), (14, 'G2', 1), (15, 'F#2', 1)])
# --- second half of the repeat (bars 17-20): the descending sequence F#7-B7 / E7-A7 / D7-G / B7-Em
bar(17, F='r:4 E6:2 C#6:2 E6:4 D#6:4', M=[(4, 'F#3 C#4', 4), (8, 'F#3 C#4', 4), (12, 'A4 B4 D#5', 4)],
    C=[(0, 'C#4', 4), (12, 'F#4', 1), (13, 'E4', 1), (14, 'D#4', 1), (15, 'C#4', 1)])
bar(18, F='r:4 D6:2 B5:2 D6:4 C#6:4', M=[(4, 'E3 B3', 4), (8, 'E3 B3', 4), (12, 'G4 A4 C#5', 4)],
    C=[(0, 'B3', 4), (12, 'E4', 1), (13, 'D4', 1), (14, 'C#4', 1), (15, 'B3', 1)])
bar(19, F='r:4 C6:2 A5:2 C6:4 B5:4', M=[(4, 'D3 A3', 4), (8, 'D3 A3', 4), (12, 'G4 B4', 4)],
    C=[(0, 'A3', 4), (12, 'D4', 1), (13, 'C4', 1), (14, 'B3', 1), (15, 'A3', 1)])
# bar 20 + the cadenza pick-up of bar 32 (beat 2, 64ths over the B pedal)
PICK = ['D4', 'F#4', 'A4', 'C5', 'D5', 'F#5', 'A5', 'C6']
bar('20s', F='r:4 B5:1 D#6:1 F#6:1 B6:1 E6:4 r:4', M=[(4, 'F#3 A3', 4), (8, 'E3 G3', 4)], C=[(0, 'G3', 4), (12, 'B1 B2', 4)],
    R=[(12 + 0.5 * j, PICK[j]) for j in range(8)])
# --- the cadenza (bars 33-36): celesta alone, rising arpeggios (64ths = 0.5 here; sextuplets 2/3)
def arp(o, notes, step): return [(o + step * j, n) for j, n in enumerate(notes)]
A_DS = 'D#4 F#4 A4 B4 D#5 F#5 A5 B5'.split(); A_D = 'D4 F#4 A4 C5 D5 F#5 A5 C6'.split()
bar(33, R=arp(0, A_DS, 0.5) + arp(4, A_D, 0.5) + arp(8, A_DS, 0.5) + arp(12, A_D, 0.5))
bar(34, R=arp(0, 'F#4 A4 B4 D#5 F#5 A5 B5 D#6'.split(), 0.5) + arp(4, 'F4 A4 C5 E5 F5 A5 C6 E6'.split(), 0.5)
    + arp(8, 'F#4 A4 B4 D#5 F#5 A5 B5 D#6'.split(), 0.5) + arp(12, 'A4 C5 D#5 G5 A5 C6 D#6 G6'.split(), 0.5))
bar(35, R=arp(0, 'B4 D#5 F#5 B4 D#5 F#5'.split(), 2 / 3) + arp(4, 'A4 C5 D#5 G5 A4 C5 D#5 G5'.split(), 0.5)
    + arp(8, 'B4 D#5 F#5 B4 D#5 F#5'.split(), 2 / 3) + arp(12, 'D5 F#5 A5 C6 D5 F#5 A5 C6'.split(), 0.5))
bar(36, R=arp(0, 'D#5 F#5 A5 B5 D#5 F#5 A5 B5'.split(), 0.5) + arp(4, 'D5 F#5 A5 C6 D5 F#5 A5 C6'.split(), 0.5)
    + arp(8, 'D#5 F#5 A5 B5 D#5 F#5 A5 B5'.split(), 0.5), C=[(12, 'B3', 1), (13, 'A3', 1), (14, 'G3', 1), (15, 'F#3', 1)])
# --- bar 52: rising scale to the dominant, then Presto
bar(52, F='r:4 E5:1 F#5:1 G5:1 A5:1 B5:4 r:4', M=[(0, 'D#5 A5 B5', 4), (4, 'E5 G5 B5', 4), (8, 'F#5 B5', 4)])
# --- Presto, bars 81-84 (unit = presto 32nd). Celesta 16ths (both voices) as rain; violins = mint; bass clarinet = cocoa
def pres(line, extra):
    r = [(2 * j, n) for j, n in enumerate(line.split())]
    for o, n in extra: r.append((2 * o, n))
    return r
bar('p81', R=pres('G5 B5 G5 B5 E5 B5 E5 B5', [(2, 'B5'), (6, 'B5')]), M=[(0, 'E4 B4', 4), (8, 'E4 B4', 4)], C=[(0, 'E3', 8), (8, 'G3', 8)])
bar('p82', R=pres('D#5 B5 D#5 B5 A5 B5 A5 B5', [(2, 'B5'), (6, 'B5')]), M=[(0, 'D#4 B4', 4), (8, 'D#4 B4', 4)], C=[(0, 'A3', 8), (8, 'F#3', 8)])
P83 = dict(R=pres('G5 B5 E5 B5 G5 B5 E5 B5', [(0, 'E5'), (2, 'B5'), (4, 'E5'), (6, 'B5')]), M=[(0, 'E4 B4', 4), (8, 'E4 B4', 4)], C=[(0, 'E3', 8), (8, 'E3', 8)])
bar('p83', **P83); bar('p84a', **P83)
bar('p84b', R=pres('G5 B5 E5 B5 E5 E5 G5 B5', [(0, 'E5'), (2, 'B5'), (4, 'G5'), (4, 'B5'), (6, 'E6')]), M=P83['M'], C=P83['C'])

FORM = ([(k, 'intro') for k in (1, 2, 3, 4)] + [(k, 'A') for k in range(5, 13)] + [(k, 'A2') for k in (17, 18, 19, '20s')]
        + [(k, 'cad') for k in (33, 34, 35, 36)] + [(k, 'A3') for k in range(5, 13)] + [(k, 'B') for k in (17, 18, 19, 52)]
        + [(k, 'presto') for k in ('p81', 'p82', 'p83', 'p84a', 'p84b')])
SCORE_BAR = {('A2', '20s'): 20}         # A3 = bars 5-12 played again, B = bars 17-19 again (= 49-51, same melody) + bar 52

grid = []; t = T0
for k, sec in FORM:
    u = (QP if sec == 'presto' else QA) / 8
    grid.append(dict(k=str(k), sec=sec, t=round(t, 5), u=u, score=SCORE_BAR.get((sec, k), k if isinstance(k, int) else int(''.join(c for c in str(k) if c.isdigit())))))
    t += B[k]['L'] * u
FINAL = round(t, 5)                                    # downbeat after bar 84: the final chord

VEL = {'intro': 70, 'A': 78, 'A2': 82, 'cad': 76, 'A3': 86, 'B': 92, 'presto': 104}
fairy, mint, cocoa, rain = [], [], [], []
for g in grid:
    b = B[int(g['k']) if g['k'].isdigit() else g['k']]; tb = g['t']; u = g['u']; v = VEL[g['sec']]
    if b['F']:
        o = 0
        for tok in b['F'].split():
            n, d = tok.split(':'); d = float(d)
            if n != 'r': fairy.append([round(tb + o * u, 5), P(n), v + (6 if o == 0 else 0), round(d * u, 5), g['sec'], g['k'], o])
            o += d
        assert abs(o - b['L']) < 1e-6, (g['k'], o)
    for o, s, d in b['M']: mint.append([round(tb + o * u, 5), chord(s), v - 12, round(d * u, 5), g['sec'], g['k'], o])
    for o, s, d in b['C']: cocoa.append([round(tb + o * u, 5), chord(s), v - 4 + (6 if o == 0 else 0), round(d * u, 5), g['sec'], g['k'], o])
    rr = {}
    for o, s in b['R']: rr.setdefault(round(o, 4), []).append(P(s))
    for o in sorted(rr):
        for p in sorted(rr[o]): rain.append([round(tb + o * u, 5), p, v - 6, round(2 * u, 5), g['sec'], g['k'], o])
# the final chord (score bar 84, last beat): fairy E6 on the star, rain E5 G5 B5, mint (violins) E4 B4, cocoa E3 E4
fairy.append([FINAL, P('E6'), 118, 3.0, 'final', 'fin', 0])
for n in ('E5', 'G5', 'B5'): rain.append([FINAL, P(n), 112, 3.0, 'final', 'fin', 0])
mint.append([FINAL, chord('E4 B4'), 110, 3.0, 'final', 'fin', 0])
cocoa.append([FINAL, chord('E3 E4'), 116, 3.0, 'final', 'fin', 0])
for L_ in (fairy, mint, cocoa, rain): L_.sort(key=lambda r: (r[0], r[1] if not isinstance(r[1], list) else r[1][0]))

notes = []
def N(t, d, role, p, v): notes.append([round(t, 5), round(d, 5), role, int(p), int(max(1, min(127, v)))])
for t, p, v, d, *_ in fairy: N(t, d, 'fairy', p, v)
for t, pp, v, d, *_ in mint:
    for p in pp: N(t, d, 'mint', p, v)
for t, pp, v, d, sec, k, o in cocoa:
    for p in pp: N(t, d, 'cocoa_run' if (len(pp) == 1 and d < 0.12) else 'cocoa', p, v)
for t, p, v, d, *_ in rain: N(t, d, 'rain', p, v)
notes.sort(key=lambda n: n[0])
json.dump(notes, open(f'{HERE}/notes.json', 'w'))
for name, L_ in (('fairy', fairy), ('mint', mint), ('cocoa', cocoa), ('rain', rain)): json.dump(L_, open(f'{HERE}/{name}.json', 'w'))
secs = []
for g in grid:
    if not secs or secs[-1][0] != g['sec']: secs.append([g['sec'], g['t']])
json.dump(dict(t0=T0, qa=QA, qp=QP, final=FINAL, grid=grid, secs=secs), open(f'{HERE}/events.json', 'w'))
if __name__ == '__main__':
    print('bars', len(grid), 'fairy', len(fairy), 'mint', len(mint), 'cocoa', len(cocoa), 'rain', len(rain), 'final', FINAL, 'video ~', round(FINAL + 4.6, 1))
    print('sections', [(s, round(t, 2)) for s, t in secs])
    gaps = sorted((round(b[0] - a[0], 3), round(a[0], 2)) for a, b in zip(fairy, fairy[1:]))[-6:]
    print('fairy longest gaps', gaps)
