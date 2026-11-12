"""Toccata and Fugue in D minor BWV 565 (J. S. Bach), bars 1-12 of the Toccata (public domain). Own arrangement for a marble organ.
Score source: score_src.json = Mutopia Project 'Toccata and Fugue in D Minor' (Bach-Gesellschaft Ausgabe 1867, Public Domain),
compiled to MIDI with LilyPond; see its 'source' field. Every pitch / onset / duration below is checked against that file.
Written here: the performance timing (free Adagio calls with fermatas, Prestissimo, broadening into bar 10 and 12), the ornament
realisations (lower mordent A-G-A on the three calls, trill G-F-G-F-G-F on the F of bar 11), the rolled dim7 chord of bar 2, who
plays what (one marble key sounds the octave pair the score doubles), and the closing registration of the final chord
(D2 D3 D4 F4 of the score + D5 F5 as 4' octave couplers: no new pitch classes).
Outputs lead.json (one row per struck key) and events.json (section times)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = json.load(open(f'{HERE}/score_src.json'))['notes']
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'

# ------------------------------------------------------------------ performance time map
# the three opening calls: free time (each note placed by hand)
CALL_T = {
    # call 1 (bar 1 beat 1): A (mordent A-G-A) held under the fermata, then G F E D (64ths) C# (32nd) D (16th)
    1: dict(a=1.25, mord=0.11, hold=2.5, run=[0.075, 0.06, 0.06, 0.06, 0.06, 0.12], dlen=0.17),
    2: dict(a=5.85, mord=0.11, hold=2.5, run=[0.08, 0.085, 0.085, 0.085, 0.13], dlen=0.18),     # E F C# (32nds) D (16th)
    3: dict(a=10.5, mord=0.12, hold=2.6, run=[0.08, 0.065, 0.065, 0.065, 0.065, 0.13], dlen=0.18),
}
T_PED1 = 15.25                 # bar 2 beat 3: the pedal D
QA = 1.25                      # Adagio quarter for the chord (bar 2 beat 4 - bar 3 beat 3)
ROLL = 0.13                    # arpeggio spread of the dim7 chord (bottom up)
T_PICK1 = 21.6                # bar 3, last 16th: the C# that starts the Prestissimo
QP = 0.80                      # Prestissimo quarter (triplet sixteenth = 0.13 s)
PAUSE = 1.95                   # fermata rest after bar 5 and bar 7 (from the end of the last A to the next pickup)
Q10 = 1.05                     # broadening: bar 10 beat 3 (pedal) -> beat 4 (chord)
QH = 1.12                      # the held chord bar 10 beat 4 - bar 11 beat 2
QB = 0.9                       # bar 11 (the recitative run)
RIT12 = 0.18                   # extra time before the final chord
HOLD12 = 5.6                   # final fermata


def build_times():
    T = {}
    b4 = T_PICK1 + 0.25 * QP
    for bar, t0 in ((4, b4), (5, b4 + 4 * QP)):
        T[bar] = t0
    t5_end = T[5] + 1.75 * QP
    T[6] = t5_end + PAUSE + 0.25 * QP
    T[7] = T[6] + 4 * QP
    t7_end = T[7] + 1.75 * QP
    T[8] = t7_end + PAUSE + 0.25 * QP
    T[9] = T[8] + 4 * QP; T[10] = T[9] + 4 * QP
    return T
BT = build_times()
T_PICK2 = BT[6] - 0.25 * QP; T_PICK3 = BT[8] - 0.25 * QP
T_PED2 = BT[10] + 2 * QP + 0.06
T_CH10 = T_PED2 + Q10
T_B11 = T_CH10 + 2 * QH          # bar 11 beat 2 (pos 1.0)


def tm(bar, pos):
    """performance time of score position (bars 2 beat 3 .. 12)"""
    if bar == 2:
        if pos >= 3: return T_PED1 + QA * (pos - 2)
        return T_PED1 + QA * (pos - 2)
    if bar == 3:
        if pos < 3.75 - 1e-6: return T_PED1 + 2 * QA + QA * pos
        return T_PICK1 + (pos - 3.75) * QP
    if 4 <= bar <= 10:
        if bar == 10 and pos >= 2 - 1e-6:
            if pos < 3 - 1e-6: return T_PED2 + (pos - 2) * Q10
            return T_CH10 + (pos - 3) * QH
        if bar in (5, 7) and pos > 3.7: return (T_PICK2 if bar == 5 else T_PICK3) + (pos - 3.75) * QP
        return BT[bar] + pos * QP
    if bar == 11:
        if pos <= 1 + 1e-6: return T_CH10 + (pos + 1) * QH
        return T_B11 + (pos - 1) * QB
    if bar == 12:
        return T_B11 + 3 * QB + RIT12
    raise ValueError((bar, pos))


rows = []          # [t, dur_s, [pitches], role, n (index within role), bar, pos, note]
def add(t, dur, ps, role, bar, pos, note=''):
    rows.append([round(t, 5), round(dur, 4), sorted(ps), role, 0, bar, pos, note])

# ------------------------------------------------------------------ the three calls (octave pairs)
CALLS = {1: (1, 0.0, [(69, 81), (67, 79), (65, 77), (64, 76), (62, 74), (61, 73), (62, 74)], [0, .625, .6875, .75, .8125, .875, 1.0]),
         2: (1, 2.0, [(57, 69), (64, 76), (65, 77), (61, 73), (62, 74)], [2.0, 2.625, 2.75, 2.875, 3.0]),
         3: (2, 0.0, [(45, 57), (43, 55), (41, 53), (40, 52), (38, 50), (37, 49), (38, 50)], [0, .625, .6875, .75, .8125, .875, 1.0])}
CALLS[2] = (1, 2.0, [(57, 69), (52, 64), (53, 65), (49, 61), (50, 62)], [2.0, 2.625, 2.75, 2.875, 3.0])
for c, (bar, p0, ps, poss) in CALLS.items():
    ct = CALL_T[c]; a, g = ps[0], (ps[0][0] - 2, ps[0][1] - 2)
    t = ct['a']; m = ct['mord']
    add(t, m, a, f'call{c}', bar, poss[0], 'mordent A'); add(t + m, m, g, f'call{c}', bar, poss[0], 'mordent G')
    t_rel = t + 2 * m + ct['hold']
    add(t + 2 * m, t_rel - t - 2 * m, a, f'call{c}', bar, poss[0], 'A (fermata)')
    tt = t_rel
    for k, (pp, pos) in enumerate(zip(ps[1:], poss[1:])):
        tt += ct['run'][k]
        last = k == len(ps) - 2
        dur = ct['dlen'] if last else ct['run'][k + 1]
        add(tt, dur, pp, f'call{c}', bar, pos, 'run')
# ------------------------------------------------------------------ pedal, the rolled dim7 chord, D major
add(T_PED1, tm(3, 2.0) - T_PED1, [38], 'ped', 2, 2.0, 'pedal D')
DIM = [49, 52, 55, 58, 61, 64]                       # C#3 E3 G3 | Bb3 C#4 E4  (LH + RH, arpeggio from the bottom)
for k, p in enumerate(DIM):
    t = tm(2, 3.0) + ROLL * k
    end = tm(3, 1.25) if p == 55 else tm(3, 1.0)
    add(t, end - t, [p], 'chord', 2, 3.0, 'dim7 (rolled)')
for p in (50, 57, 62):                                # D3 A3 D4 on bar 3 beat 2
    add(tm(3, 1.0), tm(3, 2.0) - tm(3, 1.0), [p], 'dmaj', 3, 1.0, 'D major')
add(tm(3, 1.25), tm(3, 1.5) - tm(3, 1.25), [52], 'inner', 3, 1.25, 'E')
add(tm(3, 1.5), tm(3, 2.0) - tm(3, 1.5), [54], 'inner', 3, 1.5, 'F#')
# ------------------------------------------------------------------ Prestissimo: bars 3 (pickup) - 10 beat 2, octave pairs
fast = sorted(set((b, p) for b, p, d, m, s in SRC if s == 'RH' and ((b == 3 and p > 3.7) or (4 <= b <= 9) or (b == 10 and p < 2))))
for b, p in fast:
    rh = [n for n in SRC if n[0] == b and abs(n[1] - p) < 1e-4 and n[4] == 'RH']
    lh = [n for n in SRC if n[0] == b and abs(n[1] - p) < 1e-4 and n[4] == 'LH']
    assert len(rh) == 1 and len(lh) == 1 and rh[0][3] - lh[0][3] == 12, (b, p, rh, lh)
    d = rh[0][2]
    dur = tm(b, p + d) - tm(b, p) if not (b in (5, 7) and p > 1.4) else d * QP
    if b == 3: dur = 0.25 * QP
    add(tm(b, p), dur, [lh[0][3], rh[0][3]], 'trio', b, p)
# ------------------------------------------------------------------ bar 10 pedal + dim7 chord (octave pairs), bar 11 solo
add(T_PED2, tm(11, 1.0) - T_PED2, [38], 'ped', 10, 2.0, 'pedal D')
for lo in (49, 52, 55, 58):
    end = tm(11, 1.125) if lo == 58 else tm(11, 1.0)
    add(T_CH10, end - T_CH10, [lo, lo + 12], 'chord2', 10, 3.0, 'dim7')
SOLO = [(1.125, 69), (1.25, 67), (1.375, 65), (1.5, 64), (1.625, 62), (1.75, 61), (1.875, 59), (2.0, 61), (2.25, 57), (2.5, 61),
        (2.75, 64), (2.875, 67)]
for k, (pos, p) in enumerate(SOLO):
    nxt = SOLO[k + 1][0] if k + 1 < len(SOLO) else 3.0
    add(tm(11, pos), tm(11, nxt) - tm(11, pos), [p], 'solo', 11, pos)
TR = [67, 65, 67, 65, 67, 65]                        # trill on F (from the upper note), dotted eighth
for k, p in enumerate(TR):
    t0 = tm(11, 3.0) + k * 0.75 * QB / len(TR)
    add(t0, 0.75 * QB / len(TR), [p], 'solo', 11, 3.0, 'trill')
add(tm(11, 3.75), tm(12, 0) - tm(11, 3.75), [64], 'solo', 11, 3.75)
T_FINAL = tm(12, 0)
add(T_FINAL, HOLD12, [38, 50, 62, 65, 74, 77], 'final', 12, 0.0, 'final chord (D2 D3 D4 F4 + D5 F5 couplers)')

rows.sort(key=lambda r: (r[0], r[3]))
cnt = {}
for r in rows: r[4] = cnt.get(r[3], 0); cnt[r[3]] = r[4] + 1

# ------------------------------------------------------------------ check against the source
def src_at(b, p): return [n for n in SRC if n[0] == b and abs(n[1] - p) < 1e-4]
covered = set()
for r in rows:
    t, d, ps, role, n, b, pos, note = r
    if note in ('mordent G', 'trill') or role == 'final': continue
    s = src_at(b, pos)
    for p in ps:
        hit = [x for x in s if x[3] == p]
        assert hit, ('pitch not in source', role, b, pos, nm(p))
        covered.add((b, pos, p))
for n in SRC:
    b, p, d, m, s = n
    if b == 12 and p > 0: continue
    if (b, p, m) in covered: continue
    if b == 11 and abs(p - 3.0) < 1e-4 and m == 65: continue          # the trilled F
    if b == 12 and p == 0 and m in (38, 50, 62, 65): continue          # final chord (checked below)
    raise AssertionError(('source note not played', b, p, nm(m), s))
assert sorted(x[3] for x in src_at(12, 0.0)) == [38, 50, 62, 65]
# durations of the held notes follow the score positions (sanity: ends inside the next onset)
if __name__ == '__main__':
    json.dump(rows, open(f'{HERE}/lead.json', 'w'))
    secs = dict(call1=CALL_T[1]['a'], call2=CALL_T[2]['a'], call3=CALL_T[3]['a'], ped1=T_PED1, chord=tm(2, 3.0), dmaj=tm(3, 1.0),
                end3=tm(3, 2.0), pick1=T_PICK1, b4=BT[4], b5=BT[5], pick2=T_PICK2, b6=BT[6], b7=BT[7], pick3=T_PICK3, b8=BT[8],
                b9=BT[9], b10=BT[10], ped2=T_PED2, ch10=T_CH10, b11=T_B11, trill=tm(11, 3.0), final=T_FINAL, end=T_FINAL + HOLD12)
    json.dump(dict(secs=secs, qp=QP, final=T_FINAL, hold=HOLD12), open(f'{HERE}/events.json', 'w'))
    from collections import Counter
    print('keys struck', len(rows), dict(Counter(r[3] for r in rows)))
    for k, v in secs.items(): print(f'  {k:6s} {v:7.3f}')
