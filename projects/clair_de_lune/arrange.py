"""Clair de Lune (Suite bergamasque L.75 No.3) - Claude Debussy, published 1905 (E. Fromont), public domain. Own arrangement.
Source notes: score_src.json (made by extract_src.py from the public-domain Mutopia edition of the 1905 Fromont print, score/).
Bars 1-14 exactly as printed (D-flat major, 9/8, every pitch / rhythm / tie of the four engraved voices), then an own final
D-flat major chord (one beat after bar 14).  Eighth = 0.45 s (dotted quarter = 44, Andante tres expressif).
Who plays what (every note in the score is struck by exactly one ball; same pitch at the same onset in two voices = one strike):
  twins (melody)  the right hand's top voice (rhUp). Thirds / octaves: the silver ball strikes the upper note, the gold ball the
                  lower one (a middle note of a 3-note chord goes to the drops). Single notes: both strike one shared key.
                  rhUp's two cross-staff accompaniment dyads (bar 1 and bar 9, eighth 2) belong to the drops.
  moon-stone      the lowest left-hand notes below C3 (bass, bars 8-14 and the final D-flat).
  drops           everything else (left hand chords, the inner voice of the right hand, the D-flat 3 above the bass):
                  water drops falling from bamboo spouts onto glass bowl bells.
Outputs notes.json (every sound: [t, dur, role, midi, vel]), twins.json, drops.json, bass.json, events.json."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(f'{HERE}/score_src.json'))['notes']
E8 = 0.45
BAR = 9 * E8
T0 = 2.6                      # bar 1 downbeat (the dew drops let go from the willow before it)
def tq(bar, off): return round(T0 + (bar - 1) * BAR + off * E8, 5)
FINAL = round(T0 + 14 * BAR + 0.35, 5)
NM = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'
CROSS = {(1, 1.0), (9, 1.0)}

def vel(bar, off):
    """pp, swelling a little to bar 4 and in bars 11-13, softest in bars 7-8 (the low phrase)"""
    base = {1: 46, 2: 50, 3: 54, 4: 56, 5: 50, 6: 48, 7: 44, 8: 44, 9: 52, 10: 56, 11: 62, 12: 58, 13: 66, 14: 56}[bar]
    return base + (6 if off < 0.5 else 0)

onsets = {}
for bar, off, dur, p, v in S:
    onsets.setdefault((bar, off), []).append((p, dur, v))
twins, drops, bass = [], [], []
for (bar, off), ns in sorted(onsets.items()):
    t = tq(bar, off); vv = vel(bar, off)
    used = set()
    rh = sorted([n for n in ns if n[2] == 'rhUp'], key=lambda n: -n[0]) if (bar, off) not in CROSS else []
    if rh:
        hi = rh[0]; lo = rh[-1] if len(rh) > 1 else None
        twins.append(dict(t=t, hi=hi[0], lo=lo[0] if lo else None, d=round(max(hi[1], lo[1] if lo else 0) * E8, 5), v=vv + 8, bar=bar, off=off))
        used |= {hi[0]} | ({lo[0]} if lo else set())
    for p, dur, v in ns:
        if v == 'lhDown' and p < 48:
            bass.append(dict(t=t, p=p, d=round(dur * E8, 5), v=vv + 12, bar=bar, off=off)); used.add(p)
    ds = []
    for p, dur, v in sorted(ns, key=lambda n: n[0]):
        if p in used: continue
        used.add(p); ds.append(dict(p=p, d=round(dur * E8, 5)))
    if ds: drops.append(dict(t=t, notes=ds, v=vv, bar=bar, off=off))
# own ending: D-flat major, one beat after bar 14 (bass Db2, twins F5 + Ab4, drops Db3 Ab3 Db4 F4)
twins.append(dict(t=FINAL, hi=77, lo=68, d=4.0, v=72, bar=15, off=0)); bass.append(dict(t=FINAL, p=37, d=4.5, v=84, bar=15, off=0))
drops.append(dict(t=FINAL, notes=[dict(p=p, d=4.2) for p in (49, 56, 61, 65)], v=62, bar=15, off=0))
notes = []
def N(t, d, role, p, v): notes.append([round(t, 5), round(d, 5), role, int(p), int(max(1, min(127, v)))])
for w in twins:
    if w['lo'] is None: N(w['t'], w['d'], 'uni', w['hi'], w['v'])
    else: N(w['t'], w['d'], 'hi', w['hi'], w['v']); N(w['t'], w['d'], 'lo', w['lo'], w['v'] - 6)
for dr in drops:
    for n in dr['notes']: N(dr['t'], n['d'], 'drop', n['p'], dr['v'])
for b in bass: N(b['t'], b['d'], 'bass', b['p'], b['v'])
notes.sort(key=lambda n: (n[0], n[3]))
json.dump(notes, open(f'{HERE}/notes.json', 'w')); json.dump(twins, open(f'{HERE}/twins.json', 'w'))
json.dump(drops, open(f'{HERE}/drops.json', 'w')); json.dump(bass, open(f'{HERE}/bass.json', 'w'))
json.dump({'t0': T0, 'e8': E8, 'bar': BAR, 'final': FINAL, 'grid': [tq(b, 0) for b in range(1, 16)]}, open(f'{HERE}/events.json', 'w'))
if __name__ == '__main__':
    print('twins', len(twins), 'drop onsets', len(drops), 'drop notes', sum(len(d['notes']) for d in drops), 'bass', len(bass), 'final', FINAL)
    allsrc = sorted(set((tq(b, o), p) for b, o, d, p, v in S))
    got = sorted(set((n[0], n[3]) for n in notes if n[0] < FINAL - 1e-6))
    print('score (onset,pitch) pairs', len(allsrc), 'arranged', len(got), 'equal', allsrc == got)
    for w in twins: print('T', w['bar'], w['off'], nm(w['hi']), nm(w['lo']) if w['lo'] else '-', round(w['t'], 2))
    for d in drops: print('D', d['bar'], d['off'], [nm(n['p']) for n in d['notes']], round(d['t'], 2))
    for b in bass: print('B', b['bar'], b['off'], nm(b['p']), round(b['t'], 2))
