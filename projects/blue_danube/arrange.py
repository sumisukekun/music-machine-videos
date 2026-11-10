"""The Blue Danube (An der schoenen blauen Donau, Op. 314) - Johann Strauss II, 1867, public domain. Own arrangement.
Score source: Mutopia Project, StraussJJ/O314/blue_danube (main theme; typeset by N. Kouremenos from C. Mondrup's
four-hand piano reduction of the original, Werner Icking Music Archive; Mutopia edition CC BY-SA 4.0).
Only Strauss's melody (pitches/rhythm) and the harmony per bar are taken from it; the accompaniment voicing,
orchestration, ending and everything else are written here.
Waltz No. 1 main theme (32 bars + pickup), played twice, in the original D major, 3/4.
Melody / inner voice / harmony follow the standard piano reduction of the theme (written here in C, transposed +2):
  pickup c | c e g | g2 <e'g'> | <e'g'>2 <c'e'> | <c'e'>2 c | c e g | g2 <f'g'> | <f'g'>2 <b f'> | <b f'>2 b, | ...
Accompaniment: waltz 'oom-pah-pah' - bass on beat 1, a two-note chord on beats 2 and 3 (Viennese: beat 2 a touch early).
Outputs notes.json (every sound), lead.json (melody events), events.json (bar grid, accompaniment events)."""
import json

TR = 2                       # C -> D major (original key)
Q = 0.33                     # quarter note (dotted half ~ 60)
BAR = 3 * Q
LILT = 0.94                  # beat 2 comes a little early (Viennese waltz)
T0 = 1.45                    # time of the pickup note (the pair drops from the chandelier before it)

# melody per theme bar: list of (beats, pitches); one pitch = both dancers on one key, two = inner + top voice
B = {}
def bar(k, *ev): B[k] = ev
bar(2, (1, [60]), (1, [64]), (1, [67]))
bar(3, (2, [67]), (1, [76, 79])); bar(4, (2, [76, 79]), (1, [72, 76])); bar(5, (2, [72, 76]), (1, [60]))
bar(6, (1, [60]), (1, [64]), (1, [67]))
bar(7, (2, [67]), (1, [77, 79])); bar(8, (2, [77, 79]), (1, [71, 77])); bar(9, (2, [71, 77]), (1, [59]))
bar(10, (1, [59]), (1, [62]), (1, [69]))
bar(11, (2, [69]), (1, [77, 81])); bar(12, (2, [77, 81]), (1, [71, 77])); bar(13, (2, [71, 77]), (1, [59]))
bar(14, (1, [59]), (1, [62]), (1, [69]))
bar(15, (2, [69]), (1, [76, 81])); bar(16, (2, [76, 81]), (1, [72, 76])); bar(17, (2, [72, 76]), (1, [60]))
bar(18, (1, [60]), (1, [64]), (1, [67]))
bar(19, (2, [72]), (1, [79, 84])); bar(20, (2, [79, 84]), (1, [76, 79])); bar(21, (2, [76, 79]), (1, [60]))
bar(22, (1, [60]), (1, [64]), (1, [67]))
bar(23, (2, [72]), (1, [81, 84])); bar(24, (2, [81, 84]), (1, [77, 81])); bar(25, (2, [77, 81]), (1, [62]))
bar(26, (1, [62]), (1, [65]), (1, [69]))
bar(27, (3, [69]))          # tied over into bar 28 (held 4 beats)
bar(28, (1, None), (1, [66]), (1, [67]))
bar(29, (3, [76]))          # tied over into bar 30
bar(30, (1, None), (1, [72]), (1, [64]))
bar(31, (2, [64]), (1, [62])); bar(32, (2, [69]), (1, [67]))
FIRST33 = ((2, [72]), (1, [60]))      # 1st time: cadence note + pickup into the repeat
LAST33 = ((3, [72]),)                 # 2nd time: cadence note held (the launch happens on beat 3)
FINAL_MEL = [72]                      # bar 34 downbeat: final chord

# harmony per bar (C names): bass, inner dyad
H = {}
for k in range(2, 7): H[k] = ('C', 48, [52, 55])
for k in range(7, 15): H[k] = ('G7', 50, [53, 55])
for k in range(15, 19): H[k] = ('C', 48, [52, 55])
for k in range(19, 23): H[k] = ('C/E', 52, [55, 60])
for k in range(23, 26): H[k] = ('Dm/F', 53, [57, 62])
H[26] = ('Dm', 50, [53, 57]); H[27] = ('G7/D', 50, [53, 55]); H[28] = ('G7/B', 47, [53, 55])
H[29] = ('C', 48, [52, 55]); H[30] = ('C/E', 52, [55, 60]); H[31] = ('F', 53, [57, 60]); H[32] = ('G', 43, [55, 59])
H[33] = ('C', 48, [52, 55])
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'
def chord_name(c):
    """C-based chord name -> D-based"""
    m = {'C': 'D', 'G7': 'A7', 'C/E': 'D/F#', 'Dm/F': 'Em/G', 'Dm': 'Em', 'G7/D': 'A7/E', 'G7/B': 'A7/C#', 'F': 'G', 'G': 'A'}
    return m[c]

# ---- bar grid: P1 = theme bars 2..33, P2 = theme bars 2..34 (34 = final chord only)
grid = []                    # (pass, theme bar, start time)
t = T0 + Q
for k in range(2, 34): grid.append((1, k, t)); t += BAR
for k in range(2, 35): grid.append((2, k, t)); t += BAR
def beat_t(t_bar, b):
    return t_bar + (0, LILT * Q, 2 * Q)[b] if b < 3 else t_bar + b * Q

lead = []                    # [t, [pitches], vel, dur, pass, bar, beat]
lead.append([T0, [60 + TR], 70, Q, 1, 1, 2])
FERMATA = 0.35                        # the last downbeat comes a little late (the launch to the chandelier)
for ps, k, tb in grid:
    if k == 34: lead.append([tb + FERMATA, [p + TR for p in FINAL_MEL], 118, 3 * Q, 2, 34, 0]); continue
    evs = B[k] if k < 33 else (FIRST33 if ps == 1 else LAST33)
    pos = 0
    # durations of tied notes extend into the next bar
    for d, ps_ in evs:
        if ps_ is not None:
            dur = d
            if k in (27, 29): dur = 4
            v = 74 if ps == 1 else 96
            if k >= 19 and k <= 26: v += 8
            if ps == 2 and k >= 29: v += 10
            lead.append([round(beat_t(tb, pos), 5), [p + TR for p in ps_], v, round(dur * Q, 5), ps, k, pos])
        pos += d
    assert pos == 3, (ps, k)
FINAL = lead[-1][0]

acc = []                     # accompaniment events: [t, role, [pitches], pass, bar, beat]
for ps, k, tb in grid:
    if k == 34: break
    nmC, bass, dy = H[k]
    acc.append([round(tb, 5), 'bass', [bass + TR], ps, k, 0])
    acc.append([round(beat_t(tb, 1), 5), 'ch2', [p + TR for p in dy], ps, k, 1])
    acc.append([round(beat_t(tb, 2), 5), 'ch3', [p + TR for p in dy], ps, k, 2])

notes = []
def N(t, d, role, p, v): notes.append([round(t, 5), round(d, 5), role, int(p), int(max(1, min(127, v)))])
for t, ps_, v, d, pss, k, b in lead:
    hi = max(ps_)
    for p in ps_:
        N(t, d, 'ice', p, v if p == hi else v - 12)                  # the struck ice key
    N(t, d * 0.97, 'vln', hi, v - 4)                                  # violins sing the tune
    if len(ps_) == 2: N(t, d * 0.97, 'vla', min(ps_), v - 22)
    if pss == 2 and k >= 18: N(t, d * 0.95, 'vln8', hi + 12, v - 30)
for t, role, ps_, pss, k, b in acc:
    lo = 1.0 if pss == 1 else 1.12
    if role == 'bass':
        p = ps_[0]
        N(t, 0.6, 'bell', p, 84 * lo); N(t, 0.5, 'cb', p, 92 * lo); N(t, 0.5, 'cb', p - 12, 70 * lo)
        if pss == 2: N(t, 0.4, 'tuba', p - 12, 64)
    else:
        for p in ps_:
            N(t, 0.3, 'tink', p, 74 * lo); N(t, 0.25, 'spic', p, 74 * lo)
        if pss == 2 and k >= 18: N(t, 0.2, 'hnst', ps_[1], 62)
# final chord: the two chord couples land on their last bells (D major), see layout
acc.append([round(FINAL, 5), 'ch2', [62, 66], 2, 34, 0]); acc.append([round(FINAL, 5), 'ch3', [69, 74], 2, 34, 0])
for p in (62, 66, 69, 74): N(FINAL, 2.5, 'tink', p, 100)
# final chord (bar 34 downbeat): the chandelier bell + every chord ball + tutti
for p in (38, 50): N(FINAL, 3.2, 'cbsus', p, 116)
for p in (62, 66, 69, 74): N(FINAL, 3.2, 'vsus', p, 108)
for p in (62, 66, 69): N(FINAL, 3.0, 'hnsus', p, 100)
N(FINAL, 3.5, 'bigbell', 50, 124)
notes.sort(key=lambda n: n[0])
json.dump(notes, open('notes.json', 'w'))
json.dump(lead, open('lead.json', 'w'))
json.dump({'t0': T0, 'q': Q, 'bar': BAR, 'lilt': LILT, 'final': FINAL, 'grid': grid, 'acc': acc,
           'chords': {f'{ps}-{k}': chord_name(H[k][0]) for ps, k, tb in grid if k < 34}},
          open('events.json', 'w'))
if __name__ == '__main__':
    print('bars', len(grid), 'lead events', len(lead), 'acc events', len(acc), 'final', round(FINAL, 3), 'P2 start', round(grid[32][2], 3))
