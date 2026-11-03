"""しあわせなら手をたたこう (If You're Happy and You Know It, traditional) - instrumental arrangement.
4 verses: clap / stomp / shoulder-tap / all together (+2 key change). Swing eighths.
-> notes.json [t, dur, role, midi, vel], events.json (gestures, sections, timing)"""
import json
Q = 116; B = 60 / Q; SW = 0.6          # swing: off-beat eighth at 0.6 of the beat
INTRO = 8                              # beats of intro (music-box wind-up + vamp)
C, D, E, F, G, A, Bn = 60, 62, 64, 65, 67, 69, 71
G3, B3 = 55, 59
# (beat-in-line, pitch) for the 4 lines; line = 8 beats starting at "hap"; pickups belong to previous line (beats 7,7.5)
L1 = [C, C, C, C, C, C, B3, C, D]
L2 = [D, D, D, D, D, D, C, D, E]
L3 = [F, F, F, F, A, A, F, F, E, E, E, E, C, C]
L4 = [E, D, C, C, B3, B3, A - 12, B3, C]
PICK = {0: (G3, G3), 1: (G3, G3), 2: (E, E), 3: (D, D)}   # pickup into line k
# chords per line: (beat, root, quality)
CH = [[(0, 0, ''), (4, 7, '')], [(0, 7, '7'), (4, 0, '')], [(0, 5, ''), (4, 0, '/E'), (6, 2, '7')], [(0, 7, '7'), (4, 0, '')]]
GEST = {0: True, 1: True, 2: False, 3: True}
KINDS = ['clap', 'stomp', 'tap', 'all']


def bt(b):  # beat -> seconds with swing on off-beat eighths
    i = int(b // 1); fr = b - i
    if abs(fr - 0.5) < 1e-6: fr = SW
    return (i + fr) * B


notes = []; gest = []; secs = []
def N(beat, dur, role, p, v): notes.append([round(bt(beat), 4), round(dur * B, 4), role, p, v])

# intro: 2 bars of oom-pah in C with a little marimba run, pickups at beat 7
for b in range(INTRO):
    root = 48 if b % 2 == 0 else 43
    N(b, 0.9, 'pbass', root if b < 4 else (48 if b % 2 == 0 else 43), 70 + 8 * (b % 2 == 0))
    if b % 2 == 1: [N(b, 0.5, 'pchord', p, 58) for p in (64, 67, 72)]
for i, p in enumerate((72, 76, 79, 84)): N(4 + i * 0.5, 0.5, 'bell', p, 70)
NV = 4; LINE = 8
for v in range(NV):
    tr = 2 if v == 3 else 0
    v0 = INTRO + v * 4 * LINE
    secs.append({'t': round(bt(v0 - 1), 3), 'kind': KINDS[v], 'v': v})
    for k, mel in enumerate((L1, L2, L3, L4)):
        l0 = v0 + k * LINE
        # pickup (on beats 7, 7.5 of previous line)
        for j, p in enumerate(PICK[k]): N(l0 - 1 + 0.5 * j, 0.5, 'mel', p + tr, 92)
        if k < 3 or True:
            nn = len(mel)
            for j, p in enumerate(mel):
                if k == 2:   # line 3: 14 eighths straight through to beat 6.5
                    b = l0 + 0.5 * j; d = 0.5
                else:
                    b = l0 + 0.5 * j if j < 8 else l0 + 4; d = 0.5 if j < 8 else 1.0
                if v == 1 and d >= 1.0:          # verse 2: the long note becomes four 16ths
                    for q, (off, vel) in enumerate(((0, 104), (0.3, 80), (0.6, 92), (0.8, 80))):
                        N(b + off, 0.2, 'mel', p + tr, vel)
                elif v == 1 and 2 <= j <= 5:     # verse 2: middle 8ths split into 16th pairs
                    sub = (0, 0.3) if j % 2 == 0 else (0.1, 0.3)
                    base_b = l0 + 0.5 * j
                    N(base_b, 0.25, 'mel', p + tr, 90)
                    N(base_b + (0.3 if j % 2 == 0 else 0.3), 0.2, 'mel', p + tr, 74)
                else:
                    N(b, d, 'mel', p + tr, 100 if j in (0, nn - 1) else 88)
        if v == 1 and k != 2:                    # verse 2: chord-tone 8ths over the two stomps
            fill = {0: (55, 59, 62, 59), 1: (60, 64, 67, 64), 3: (60, 64, 67, 64)}[k]
            for q, p in enumerate(fill): N(l0 + 5 + 0.5 * q, 0.5, 'mel', p, 72)
        if GEST[k]:
            for g in (5, 6): gest.append({'t': round(bt(l0 + g), 4), 'kind': KINDS[v], 'v': v, 'n': g - 5, 'line': k})
        # accompaniment
        chs = CH[k]
        for ci, (cb, root, q) in enumerate(chs):
            ce = chs[ci + 1][0] if ci + 1 < len(chs) else LINE
            r = 48 + root + tr; r = r - 12 if r > 54 else r
            if q == '/E': bassn = 40 + 4 + tr
            elif q == '7' and root == 2: bassn = 42 + tr   # D7/F#
            else: bassn = r
            third = 4; fifth = 7
            tones = [root + 12 * 5, root + third + 60, root + fifth + 60] + ([root + 70] if q == '7' else [])
            tones = [t_ + tr - (12 if t_ + tr > 74 else 0) for t_ in tones]
            for bb in range(cb, ce):
                if bb % 2 == 0: N(l0 + bb, 0.9, 'pbass', bassn if bb == cb else (bassn + 7 if bassn + 7 < 55 else bassn - 5), 78)
                else: [N(l0 + bb, 0.45, 'pchord', p, 52) for p in tones]
                if v >= 1 and bb % 2 == 1: N(l0 + bb, 0.2, 'tamb', 0, 50 + 10 * (v == 3))
                if v >= 1 and k == 2: N(l0 + bb + 0.5, 0.2, 'shaker', 0, 45)
            if v >= 2: N(l0 + cb, (ce - cb) * 0.9, 'cbass', bassn, 80)
        # glock doubles the melody in the last verse
    if v == 3:
        for n in [x for x in notes if x[2] == 'mel' and x[0] >= bt(v0 - 1) - 1e-3]: notes.append([n[0], n[1], 'glock', n[3] + 12, 80])
END = INTRO + NV * 4 * LINE
# ending: "hands" of the last line then a big final chord + all gestures + confetti
fin = bt(END)
for p in (62 - 12, 62, 66, 69, 74): N(END, 4, 'pchord', p, 90)
N(END, 4, 'pbass', 38, 100); N(END, 4, 'mel', 74, 110); N(END, 4, 'glock', 86, 100); N(END, 4, 'cbass', 38, 100)
gest.append({'t': round(fin, 4), 'kind': 'final', 'v': 4, 'n': 0, 'line': 4})
notes.sort(key=lambda x: x[0])
json.dump(notes, open('notes.json', 'w'))
json.dump({'beat': B, 'swing': SW, 'intro': INTRO, 'line': LINE, 'verse_beats': 4 * LINE, 'secs': secs, 'gest': gest,
           'final': round(fin, 3), 'mel': [[n[0], n[3]] for n in notes if n[2] == 'mel']}, open('events.json', 'w'), indent=0)
print(len(notes), 'notes; final', round(fin, 2), 's; gestures', len(gest))
