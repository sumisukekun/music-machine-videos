"""Note-by-note comparison of arrange.py with the public-domain sources in score/ (python3 score_check.py [-v]).
 fairy  : top note of every celesta onset, orchestral transcription PDMX 5606908 (celesta lowered by its octave error)
 rain   : every celesta note of the cadenza (bar 32 pick-up, 33-36) and of Presto bars 81-84 (same source)
 cocoa  : pizzicato bass = cellos / basses of 5606908 (bars 1-12), bass-clarinet runs (bars 8, 12) and bassoon run (36),
          left hand of Esipoff's piano reduction (bars 17-20)
 mint   : off-beat chords: pitch classes compared with violins+violas of 5606908 and with Esipoff's left hand"""
import json, os, sys, bisect
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv += [] ; V = '-v' in sys.argv
import importlib.util
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'


def load(fn, tracks, shift=0):
    d = json.load(open(f'{HERE}/score/{fn}')); BT = sorted(set(x['time'] for x in d['barlines']))
    out = {}
    for ti in tracks:
        for n in d['tracks'][ti]['notes']:
            i = bisect.bisect_right(BT, n['time']) - 1; L = BT[i + 1] - BT[i] if i + 1 < len(BT) else 960
            off = round((n['time'] - BT[i]) / (L / 16), 3)          # 32nds
            out.setdefault((i + 1, off), []).append(n['pitch'] + shift)
    return out


ORCH = 'pdmx_5606908_orch_CC0.json'; ESI = 'pdmx_1635221_esipoff_piano.json'
CEL = load(ORCH, [10], -12)
STR = load(ORCH, [11, 12, 13])                 # violins I, II, violas
BAS = load(ORCH, [14, 15])                     # cellos, basses
BCL = load(ORCH, [5]); BSN = load(ORCH, [6, 7])
ESIP = load(ESI, [0])
E = json.load(open(f'{HERE}/events.json'))
SB = {g['k'] + '|' + g['sec']: g['score'] for g in E['grid']}
rows = []; bad = []; FIX = []


def score_bar(sec, k): return SB[f'{k}|{sec}']
def chk(voice, sec, k, o, got, ref, src, mode='eq'):
    ok = (sorted(got) == sorted(ref)) if mode == 'eq' else ({p % 12 for p in got} <= {p % 12 for p in ref}) if mode == 'pc' else (got[0] == max(ref))
    rows.append((voice, sec, k, o, [nm(p) for p in got], [nm(p) for p in ref], src, ok))
    if not ok: bad.append(rows[-1])


for name in ('fairy', 'rain', 'cocoa', 'mint'):
    for t, p, v, d, sec, k, o in json.load(open(f'{HERE}/{name}.json')):
        if sec == 'final': continue
        sb = score_bar(sec, k); pp = p if isinstance(p, list) else [p]
        if k == '20s' and o >= 12 and name == 'rain': sb, o = 32, o           # the cadenza pick-up is bar 32 beat 2
        if name == 'fairy':
            ref = CEL.get((sb, o)); e_ = ESIP.get((sb, o)) or [-1]
            if ref and pp[0] != max(ref) and pp[0] % 12 == max(e_) % 12:
                chk('F', sec, k, o, pp, [max(e_)], f'Esipoff top (orch. copy has {nm(max(ref))})', 'pc'); FIX.append(rows[-1])
            else: chk('F', sec, k, o, pp, ref or [-1], 'orch celesta top', 'top')
        elif name == 'rain':
            src = sb if sec != 'presto' else {'p81': 81, 'p82': 82, 'p83': 83, 'p84a': 84, 'p84b': 84}[k]
            oo = o + (16 if k == 'p84b' else 0)
            ref = CEL.get((src, round(oo, 3)))
            e_ = ESIP.get((35, 14.0)) if (k == '35' and o == 8.0) else None      # orch. copy E5: Esipoff's B4 (the B-major arpeggio)
            if e_: chk('R', sec, k, o, pp, e_, 'Esipoff (orch. copy has E5)', 'pc'); FIX.append(rows[-1])
            else: chk('R', sec, k, o, pp, sorted(set(ref)) if ref else [-1], 'orch celesta', 'pc' if ref else 'eq')
        elif name == 'cocoa':
            if len(pp) == 1 and d < 0.12:                                     # runs
                srcd = BCL if sb in (8, 12) else BSN if sb == 36 else ESIP
                ref = srcd.get((sb, o)) or ESIP.get((sb, o)) or [-1]
                chk('C', sec, k, o, pp, ref, 'bass clar.' if sb in (8, 12) else 'bassoon' if sb == 36 else 'Esipoff', 'pc')
            elif sec == 'presto':
                src = {'p81': 81, 'p82': 82, 'p83': 83, 'p84a': 84, 'p84b': 84}[k]; ref = BCL.get((src, o + (16 if k == 'p84b' else 0))) or [-1]
                chk('C', sec, k, o, pp, ref, 'bass clar.', 'pc')
            elif sb in (17, 18, 19, 20, 49, 50, 51) and o == 0:
                ref = ESIP.get((sb, o)) or [-1]; chk('C', sec, k, o, pp, ref, 'Esipoff LH', 'pc')
            elif sb == 20:
                ref = BAS.get((20, o)) or [-1]; chk('C', sec, k, o, pp, ref, 'vc/cb', 'pc')
            else:
                ref = BAS.get((sb, o)) or [-1]; chk('C', sec, k, o, pp, ref, 'vc/cb', 'eq')
        else:
            if sec == 'presto':
                src = {'p81': 81, 'p82': 82, 'p83': 83, 'p84a': 84, 'p84b': 84}[k]; ref = STR.get((src, o + (16 if k == 'p84b' else 0))) or [-1]
                chk('M', sec, k, o, pp, ref, 'violins', 'pc')
            elif sb in (17, 18, 19, 20, 49, 50, 51):
                ref = ESIP.get((sb, o)) or [-1]; chk('M', sec, k, o, pp, ref, 'Esipoff LH', 'pc')
            elif sb == 52:
                ref = (STR.get((52, o)) or []) + (load(ORCH, [0]).get((52, o)) or []); chk('M', sec, k, o, pp, ref or [-1], 'vln+fl', 'pc')
            else:
                s0 = sb
                ref = STR.get((s0, o)) or [-1]; ref2 = ESIP.get((s0, o)) or []
                chk('M', sec, k, o, pp, sorted(set(ref + ref2)), 'strings+Esipoff', 'pc')
n = {}
for r in rows: n.setdefault(r[0], [0, 0]); n[r[0]][0] += 1; n[r[0]][1] += r[7]
print('checked (voice: events, matching):', n)
print('mismatches:', len(bad)); print('orchestral-copy errors resolved by the second source:', len(FIX))
for r in FIX: print('  ', r)
for r in bad: print('  ', r)
if V:
    for r in rows: print(r)
