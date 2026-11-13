"""O Tannenbaum (O Christmas Tree) - German folk melody, words Ernst Anschuetz 1824 (public domain). Own arrangement for marbles.
Score source: score_src.json = an SATB setting in F major (PDMX, Public Domain Mark; see 'source' in the file): pickup + 12 bars, A-B-A.
The melody is cross-checked note by note against two more public-domain / CC0 melody editions in PDMX (G major, the A section
repeated there): same pitches everywhere (46 of 46); rhythm variants only in source bars 2 / 10 (F dotted quarter + G eighth in the SATB setting,
two quarters there), 7 / 9 (dotted eighth + sixteenth vs two eighths) and the held last note - the SATB setting's rhythm is used.
Taken from the source: the melody (pitch / rhythm, all 12 bars + pickup) and the bass voice (the note sounding on beat 1 of each bar;
in the cadence bars 5 and 13 also on beat 2, where it resolves C -> F).
Written here: three verses (as in Anschuetz's three stanzas, no words are shown or sung), quarter = 103.4, who plays what,
a small ritardando on the very last note.

Phrases (k = 0..8, three per verse: A, B, A). Each phrase = half 1 (pickup + 2 bars, 9 notes) played by Stella, the fallen-star
marble, on ice keys; half 2 (2 bars) played by glass ornaments on the bells of the newly grown tree tier while Stella rides up.
Bass (verses 2 and 3): pinecones dropping onto log drums. The very last melody note is Stella landing on the tree top.
Outputs lead.json (melody notes) and events.json (bass, bar times, chords)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = json.load(open(f'{HERE}/score_src.json'))
Q = 0.58                    # quarter note (s)
T0 = 2.6                    # the first pickup (verse 1)
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'
W = SRC['satb_f']
SOP = [e for e in W if e[4] == 'S']
assert len(SOP) == 46

# ---- cross-check of the melody against the two melody editions (G major -> F major: -2), first A, B, last A
def check_mel(rows):
    res = 480; ns = [(t / res, d / res, p - 2) for t, d, p in rows]
    # their form: pickup + A (4 bars) twice, then B, A: take notes of the 1st pass (pickup + 4 bars), then from the B pickup on
    a1 = [n for n in ns if n[0] < 12 - 1e-6]
    b_start = [i for i, n in enumerate(ns) if n[0] >= 24 - 1e-6][0]
    return a1 + ns[b_start:]
for name in ('check_pd_g', 'check_cc0_g'):
    rows = SRC[name]
    if name == 'check_cc0_g':        # that edition is written a fifth lower (B-flat): bring it to G first
        rows = [[t, d, p + 9] for t, d, p in rows]
        rows = [[t - 960, d, p] for t, d, p in rows]
    cm = check_mel(rows)
    sp = [e[3] for e in SOP]; cp = [n[2] for n in cm]
    assert len(cp) == len(sp) and all(a % 12 == b % 12 and abs(a - b) < 12 for a, b in zip(sp, cp)), (name, [nm(x) for x in cp])
    so = [(e[0], e[1], e[2]) for e in SOP]
    rdiff = sorted(set(so[i][0] for i in range(len(so)) if abs(cm[i][1] - so[i][2]) > 1e-6))
    assert set(rdiff) <= {2, 7, 9, 10, 13}, (name, rdiff)


def t_of(v, m, pos):
    """verse v = 0..2, bar m = 1..13 of the source (1 = pickup), pos in quarters. The pickup of verses 2, 3 = beat 3 of bar 13."""
    if m == 1: m, pos, v = 13, 2.0 + pos, v - 1
    tb = T0 + Q + ((12 * v + m - 2) * 3 + pos) * Q
    return tb


FINAL_EXTRA = 0.12                      # ritardando: the final F comes a little late
def sounding(m, pos, voice, rest_ok=False):
    n = [e for e in W if e[4] == voice and e[0] == m and e[1] <= pos + 1e-6 and e[1] + e[2] > pos + 1e-6]
    if rest_ok and not n: return None
    assert len(n) == 1, (m, pos, voice, n)
    return n[0][3]


HALF1 = {0: lambda m, pos: (m == 1) or m in (2, 3), 1: lambda m, pos: (m == 5 and pos >= 2.5) or m in (6, 7),
         2: lambda m, pos: (m == 9 and pos >= 2) or m in (10, 11)}
HALF2 = {0: lambda m, pos: m == 4 or (m == 5 and pos < 2), 1: lambda m, pos: m == 8 or (m == 9 and pos < 2),
         2: lambda m, pos: m in (12, 13)}
lead = []        # [t, pitch, vel, dur_s, verse, bar, pos, role, phrase k]
for v in range(3):
    for m, pos, d, p, _ in SOP:
        if m == 1 and v > 0: pass
        for ph in range(3):
            if HALF1[ph](m, pos): role = 'star'; break
            if HALF2[ph](m, pos): role = 'orn'; break
        else: raise AssertionError((m, pos))
        k = 3 * v + ph
        t = t_of(v, m, pos)
        last = v == 2 and m == 13 and pos == 1
        if last: t += FINAL_EXTRA; role = 'top'
        vel = 72 + 6 * v + (6 if role == 'orn' else 0)
        dur = 4.0 if last else d * Q
        lead.append([round(t, 5), p, vel, round(dur, 4), v, m, pos, role, k])
lead.sort(key=lambda l: l[0])
assert len(lead) == 3 * 46
FINAL = lead[-1][0]
bass = []; chords = {}
for v in range(3):
    for m in range(2, 14):
        for pos in (0, 1, 2):
            chords[f'{v}-{m}-{pos}'] = sorted(set(NM[q % 12] for vv in 'SATB' for q in [sounding(m, pos, vv, True)] if q is not None))
        if v == 0: continue
        for pos in ((0, 1) if m in (5, 13) else (0,)):
            t = t_of(v, m, pos) + (FINAL_EXTRA if (v == 2 and m == 13 and pos == 1) else 0.0)
            k = 3 * v + (0 if m <= 5 else 1 if m <= 9 else 2)
            bass.append([round(t, 5), sounding(m, pos, 'B'), v, m, pos, k])
for t, p, v, m, pos, k in bass:
    assert NM[p % 12] in chords[f'{v}-{m}-{pos}']
json.dump(lead, open(f'{HERE}/lead.json', 'w'))
json.dump({'t0': T0, 'q': Q, 'final': FINAL, 'bass': bass, 'chords': chords,
           'bar_t': {f'{v}-{m}': round(t_of(v, m, 0), 5) for v in range(3) for m in range(2, 14)}}, open(f'{HERE}/events.json', 'w'))
if __name__ == '__main__':
    print('melody notes', len(lead), {r: sum(l[7] == r for l in lead) for r in ('star', 'orn', 'top')}, 'bass', len(bass), 'final', round(FINAL, 3))
    for k in range(9):
        print(k, 'star', ' '.join(nm(l[1]) for l in lead if l[8] == k and l[7] == 'star'), '| orn', ' '.join(nm(l[1]) for l in lead if l[8] == k and l[7] != 'star'),
              '| bass', ' '.join(nm(b[1]) for b in bass if b[5] == k))
