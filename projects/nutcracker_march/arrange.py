"""Tchaikovsky - March (Marche), The Nutcracker Op.71 No.2 / Suite Op.71a No.2 (1891-92, P. Jurgenson, Moscow 1892 -
public domain).  Own reduction for marble voices, every note taken programmatically from the source below.

Score source (copy in score/):
  score/pdmx_5904891_march_orch_CC0.json   PDMX (CC0 1.0) MuseScore transcription "Studio KonKon (CC0) 2018" of the full
      orchestral score IMSLP #231707 = first edition, Moscow: P. Jurgenson [1892], plate 17668 (early reprint, Leipzig:
      D. Rahter, plate 319).  Concert pitch, 88 bars, G major, 4/4, "Tempo di marcia viva".
  Only Tchaikovsky's notes are used (each one is read out of a named track / bar / offset of that file below); nothing
  is copied from any modern arrangement.  score_check.py re-reads the source and checks every note.

Abridgement (score bars):  9-24 (= 1-16 without the first statement of the opening fanfare) | 41-48 (the scurrying
  16th-note episode) | 81-88 (the last fanfare and the end).  Played at quarter = 130.
Voices -> marbles (team SOLDIERS vs team MICE; every key struck = one point for that team):
  cap   Nutcracker captain   fanfare top (clarinet I), the violin tune 13-16 and 85-88, final G4      SOLDIERS
  s2    soldier              fanfare middle (horn I)                                                    SOLDIERS
  s3    soldier              fanfare bottom (clarinet II)                                              SOLDIERS
  drum  drummer              bass: bassoon B3 (10, 12), cellos+basses 13-16 and 85-88, the bass run and
                             tuba in 81-84                                                               SOLDIERS
  shell golden shells        81-84: the rising runs (cello -> violin I -> viola -> violin II -> violin I)
                             and the falling flute cascade (one shell = one note)                       SOLDIERS
  king  Mouse King           trumpet I: B4 (16), the minor fanfare 17 and 21, C4 (18, 22), then the long
                             descending scale 22-24 (clarinet I -> trumpet I/II -> bassoon)              MICE
  m2/m3 mouse guards         trombones I / II in 17-18, 21-22                                            MICE
  rat   rat drummer          bass 17-24 (tuba, cellos, basses, violas) and 41-48 (basses+cellos)        MICE
  army  mouse army           flute grace-note figures 16, 18-20, the 16th-note stream 41-48 and the flute E5s of 48 (one mouse
                             marble = one note)                                                          MICE
Outputs notes.json (every sound), voices.json, events.json."""
import json, os, bisect
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = f'{HERE}/score/pdmx_5904891_march_orch_CC0.json'
D = json.load(open(SRC))
BT = sorted(set(x['time'] for x in D['barlines'])); Q16 = D['resolution'] / 4
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'
TR = {'fl1': 0, 'cl1': 5, 'cl2': 6, 'bn1': 7, 'hn1': 9, 'tp1': 13, 'tp2': 14, 'tb1': 15, 'tb2': 16, 'tu1': 17, 'tu2': 18,
      'vn1': 20, 'vn2': 21, 'va': 22, 'vc': 23, 'cb': 24}
SRCN = {}                                                   # (bar, track name) -> [(offset16, pitch, dur16, grace)]
for name, ti in TR.items():
    for n in D['tracks'][ti]['notes']:
        i = bisect.bisect_right(BT, n['time']) - 1
        SRCN.setdefault((i + 1, name), []).append((round((n['time'] - BT[i]) / Q16, 3), n['pitch'], n['duration'] / Q16, bool(n.get('is_grace'))))


def take(bar, tr, lo=0, hi=15.99, grace=False, pick=None):
    """notes of track `tr` in score bar `bar` with lo <= offset <= hi (grace notes only if grace=True).
    pick: 'top' / 'low' keeps one pitch per onset."""
    r = [x for x in SRCN.get((bar, tr), []) if lo - 1e-6 <= x[0] <= hi + 1e-6 and (grace or not x[3])]
    if pick:
        by = {}
        for x in r: by.setdefault(x[0], []).append(x)
        r = [max(v, key=lambda x: x[1]) if pick == 'top' else min(v, key=lambda x: x[1]) for k, v in sorted(by.items())]
    return [(bar, o, p, d, g, tr) for o, p, d, g in r]


QB = 60 / 130.0                     # quarter note
U = QB / 4                          # 16th note
T0 = 1.25                           # downbeat of score bar 9
PLAY = list(range(9, 25)) + list(range(41, 49)) + list(range(81, 89))
BAR_T = {b: T0 + 16 * U * k for k, b in enumerate(PLAY)}
SEC = {}
for b in PLAY: SEC[b] = 'fanfare' if b <= 12 else 'tune' if b <= 16 else 'mice' if b <= 24 else 'army' if b <= 48 else 'finale' if b <= 84 else 'end'
GRACE_DT = 0.045                    # a grace note sounds this long before its main note (several: this far apart)

V = {k: [] for k in ('cap', 's2', 's3', 'drum', 'shell', 'king', 'm2', 'm3', 'rat', 'army')}
def add(v, notes): V[v].extend(notes)
# ---------------- SOLDIERS
for b in (9, 10, 11, 12, 81, 82, 83, 84):
    add('cap', take(b, 'cl1')); add('s2', take(b, 'hn1')); add('s3', take(b, 'cl2'))
add('cap', take(12, 'vn1', 15, 15))
for b in (13, 14, 15): add('cap', take(b, 'vn1'))
add('cap', take(16, 'vn1', 0, 8))
add('cap', take(84, 'vn1', 15, 15))
for b in (85, 86, 87): add('cap', take(b, 'vn1'))
add('cap', take(88, 'vn1', 0, 8)); add('cap', take(88, 'vn1', 12, 12, pick='top'))      # final: G4 (violins I: G3+G4)
add('s2', take(88, 'vn2', 12, 12, pick='top')); add('s3', take(88, 'vn2', 12, 12, pick='low'))   # D4 / B3
add('drum', take(10, 'bn1', 0, 0)); add('drum', take(12, 'bn1', 0, 0))
for b in (13, 14, 15): add('drum', take(b, 'vc') + take(b, 'cb'))
add('drum', take(16, 'vc') + take(16, 'cb'))
for b in (81, 83):
    add('drum', take(b, 'cb', 0, 4)); add('drum', take(b, 'tu1', 8, 12))
for b in (82, 84): add('drum', take(b, 'tu1'))
for b in (85, 86, 87): add('drum', take(b, 'vc') + take(b, 'cb'))
add('drum', take(88, 'vc', 0, 8) + take(88, 'cb', 0, 8)); add('drum', take(88, 'vc', 12, 12, pick='low') + take(88, 'cb', 12, 12))
for b in (81, 83):
    add('shell', take(b, 'vc', 4, 8) + take(b, 'vn1', 8, 12) + take(b, 'va', 12, 15))
for b in (82, 84):
    add('shell', take(b, 'va', 0, 0) + take(b, 'vn2', 0, 4) + take(b, 'vn1', 4, 8) + take(b, 'fl1', 9, 15))
add('shell', take(83, 'fl1', 0, 0)); add('shell', take(85, 'fl1', 0, 0))
# ---------------- MICE
add('king', take(16, 'tp1', 12, 12))
for b in (17, 21): add('king', take(b, 'tp1'))
for b in (18, 22): add('king', take(b, 'tp1', 0, 0))
add('king', take(22, 'cl1', 10, 14) + take(23, 'cl1', 0, 0) + take(23, 'cl2', 2, 8) + take(23, 'tp1', 10, 12) + take(23, 'tp2', 14, 14) + take(24, 'bn1'))
for b in (17, 21): add('m2', take(b, 'tb1')); add('m3', take(b, 'tb2'))
for b in (18, 22): add('m2', take(b, 'tb1', 0, 0)); add('m3', take(b, 'tb2', 0, 0))
for b in (17, 21): add('rat', take(b, 'tu1', 12, 12))
for b in (18, 22): add('rat', take(b, 'tu1', 0, 8) + take(b, 'tu2', 4, 8))
add('rat', take(18, 'vc', 10, 14)); add('rat', take(19, 'vc')); add('rat', take(20, 'vc'))
add('rat', take(22, 'tu1', 12, 12) + take(22, 'cb', 12, 12))
add('rat', take(23, 'cb', 0, 0) + take(23, 'va', 4, 12))
add('rat', take(24, 'va', 0, 0) + take(24, 'tb2', 2, 8) + take(24, 'tu1', 10, 14))
for b in range(41, 49): add('rat', take(b, 'cb') + take(b, 'vc'))
add('army', take(16, 'fl1', 12, 12, grace=True)); add('army', take(18, 'fl1', 12, 12, grace=True))
for b in (19, 20): add('army', take(b, 'fl1', grace=True))
for b in (41, 42, 45, 46): add('army', take(b, 'fl1'))
for b in (43, 47): add('army', take(b, 'vn1'))
add('army', take(44, 'vn1', 0, 12) + take(44, 'fl1', 12, 15)); add('army', take(48, 'vn1') + take(48, 'fl1', 8, 12))

TEAM = {v: ('S' if v in ('cap', 's2', 's3', 'drum', 'shell') else 'M') for v in V}
# ---------------- events: one per (voice, onset) with all its pitches; grace notes get their own (earlier) event
EVT = {}
for v, L in V.items():
    rows = {}; graces = {}
    for bar, o, p, d, g, tr in L:
        if g: graces.setdefault((bar, o), []).append((p, tr)); continue
        rows.setdefault((bar, o), dict(p=[], d=d, tr=[]))
        if p not in rows[(bar, o)]['p']: rows[(bar, o)]['p'].append(p); rows[(bar, o)]['tr'].append(tr)
        rows[(bar, o)]['d'] = max(rows[(bar, o)]['d'], d)
    ev = []
    for (bar, o), r in rows.items():
        t = BAR_T[bar] + o * U
        pt = sorted(zip(r['p'], r['tr']))
        ev.append(dict(t=round(t, 5), p=[a for a, _ in pt], d=round(r['d'] * U, 5), bar=bar, o=o, tr=[b for _, b in pt], sec=SEC[bar], grace=False))
        gs = sorted(graces.get((bar, o), []))                       # grace notes in rising order (they lead up to the main note)
        for j, (gp, gtr) in enumerate(gs):
            ev.append(dict(t=round(t - GRACE_DT * (len(gs) - j), 5), p=[gp], d=0.05, bar=bar, o=o, tr=[gtr], sec=SEC[bar], grace=True))
    if v in ('shell', 'army'):                                      # one marble per note: split chords
        ev = [dict(e, p=[q], tr=[tq]) for e in ev for q, tq in zip(e['p'], e['tr'])]
    EVT[v] = sorted(ev, key=lambda e: (e['t'], e['p']))
FINAL = BAR_T[88] + 12 * U
notes = []
for v, ev in EVT.items():
    for e in ev:
        for p in e['p']: notes.append([e['t'], max(0.05, e['d']), v, int(p), 100 if e['t'] >= FINAL - 1e-6 else 84])
notes.sort(key=lambda n: (n[0], n[3]))
json.dump(notes, open(f'{HERE}/notes.json', 'w'))
json.dump(EVT, open(f'{HERE}/voices.json', 'w'))
secs = []
for b in PLAY:
    if not secs or secs[-1][0] != SEC[b]: secs.append([SEC[b], BAR_T[b]])
json.dump(dict(t0=T0, q=QB, u=U, final=FINAL, bar_t={str(k): v for k, v in BAR_T.items()}, secs=secs, team=TEAM), open(f'{HERE}/events.json', 'w'))
if __name__ == '__main__':
    hs = {'S': 0, 'M': 0}
    for v, ev in EVT.items():
        k = len(ev) if v in ('shell', 'army') else len(ev)
        hs[TEAM[v]] += k
        print(f'{v:6s} {TEAM[v]} events {len(ev):4d}  pitches {min(min(e["p"]) for e in ev)}-{max(max(e["p"]) for e in ev)}  first {ev[0]["t"]:.2f} last {ev[-1]["t"]:.2f}')
    print('hits per team', hs, 'final', round(FINAL, 3), 'video', round(FINAL + 4.6, 2), 's')
    print('sections', [(s, round(t, 2)) for s, t in secs])
