"""Note-by-note check of the arrangement against the public-domain source (python3 score_check.py [-v]).
Re-reads score/pdmx_5904891_march_orch_CC0.json independently of arrange.py and checks, for every event in voices.json:
  - its score bar / offset maps to the event time (quarter = 130, bars 9-24, 41-48, 81-88 played in that order)
  - every pitch of the event is present, at exactly that bar and offset, in the source part it is credited to
  - grace notes are grace notes of that part at that offset
Also prints, per voice, how many notes were checked and the source parts used."""
import json, os, sys, bisect
HERE = os.path.dirname(os.path.abspath(__file__))
V = '-v' in sys.argv
D = json.load(open(f'{HERE}/score/pdmx_5904891_march_orch_CC0.json'))
BT = sorted(set(x['time'] for x in D['barlines'])); Q16 = D['resolution'] / 4
TR = {'fl1': 0, 'cl1': 5, 'cl2': 6, 'bn1': 7, 'hn1': 9, 'tp1': 13, 'tp2': 14, 'tb1': 15, 'tb2': 16, 'tu1': 17, 'tu2': 18,
      'vn1': 20, 'vn2': 21, 'va': 22, 'vc': 23, 'cb': 24}
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'
SRC = {}
for name, ti in TR.items():
    assert D['tracks'][ti]['notes'], name
    for n in D['tracks'][ti]['notes']:
        i = bisect.bisect_right(BT, n['time']) - 1
        SRC.setdefault((name, i + 1, round((n['time'] - BT[i]) / Q16, 3)), []).append((n['pitch'], bool(n.get('is_grace'))))
VO = json.load(open(f'{HERE}/voices.json')); E = json.load(open(f'{HERE}/events.json'))
PLAY = list(range(9, 25)) + list(range(41, 49)) + list(range(81, 89)); U = 60 / 130 / 4
bad = []; n_ok = 0; used = {}
for v, evs in VO.items():
    for e in evs:
        bar, o = e['bar'], e['o']
        t_exp = E['t0'] + 16 * U * PLAY.index(bar) + o * U
        if not e['grace'] and abs(t_exp - e['t']) > 1e-4: bad.append((v, bar, o, 'time', e['t'], t_exp))
        trs = e['tr'] if len(e['tr']) == len(e['p']) else e['tr'] * len(e['p'])
        for p, tr in zip(e['p'], trs):
            got = SRC.get((tr, bar, round(o, 3)), [])
            ok = any(gp == p and gg == e['grace'] for gp, gg in got)
            used.setdefault(v, set()).add(tr)
            if ok: n_ok += 1
            else: bad.append((v, bar, o, nm(p), tr, [(nm(a), g) for a, g in got]))
            if V: print(f'{v:6s} bar {bar:3d} +{o:6.3f}/16 {nm(p):4s} {"grace" if e["grace"] else "     "} <- {tr:4s} {"OK" if ok else "MISSING"}')
print('notes checked', n_ok + len(bad), 'ok', n_ok, 'not in source', len(bad))
for v in VO: print(f'  {v:6s} {len(VO[v]):4d} events, parts {sorted(used.get(v, []))}')
for b in bad[:20]: print('BAD', b)
