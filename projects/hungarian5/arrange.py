"""Brahms - Hungarian Dance No. 5 (1869, public domain). Own arrangement in G minor (the familiar orchestral key).
Melody / bass follow the standard piano score (Allegro A theme, the B strain with its poco rit., Vivace trio in G major with
the dolce/vivace alternation, return of A in octaves, final Cm6 - D7 - G). Shortened form:
  A (bars 1-16) | B (33-48, rit. on 41-44) | trio (49-60, 61-68 once) | A (69-84) | coda (85-86)
The slow places are not a tempo curve drawn by hand: they come out of the physics planned in layout.py
  rit. (bars 41-44) and the two dolce bars: both balls roll over a slatted rooftop deck (物干し台) and are slowed by the
  slats' rolling friction (constant deceleration), the slats sit at equal spacing per eighth, so each eighth takes longer;
  they creep to the edge of the deck (fermata) and tip over; the next a-tempo note is the landing after the free fall.
Outputs notes.json (every sound), lead.json (thief = melody), bass.json (cop = bass notes), events.json."""
import json, math

G = 7.0                                   # gravity of the whole film (layout.py uses the same)
Q144 = 60 / 144; S16 = Q144 / 4           # Allegro
S16V = 60 / 160 / 4                       # Vivace
T0 = 1.25                                 # hook (searchlight + whistle) before the first note
PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def pitch(s):
    n = PC[s[0]]; k = 1
    while k < len(s) and s[k] in 'b#':
        n += 1 if s[k] == '#' else -1; k += 1
    return 12 * (int(s[k:]) + 1) + n


def mel(bar):
    """'D4:6 G4:2' -> [(pos16, dur16, midi or None)]"""
    out = []; pos = 0
    for tok in bar.split():
        p, d = tok.split(':'); d = float(d)
        out.append((pos, d, None if p == 'r' else pitch(p))); pos += d
    assert abs(pos - 8) < 1e-9, bar
    return out


# ---------------------------------------------------------------- melody (written pitch) ----------------------------
A1 = ['D4:6 G4:2', 'Bb4:6 G4:2', 'F#4:6 G4:1 A4:1', 'G4:8', 'Eb4:6 F4:1 G4:1', 'D4:8', 'C4:1 Bb3:1 Bb3:1 A3:1 A3:3 D4:1', 'G3:8']
A2 = ['D4:6 G4:1 Bb4:1', 'D5:6 Bb4:2', 'A4:6 Bb4:1 C5:1', 'Bb4:8', 'Eb5:1 F5:1 G5:1 Eb5:1 D5:1 Eb5:1 F5:1 D5:1',
      'C5:1 D5:1 Eb5:1 C5:1 Bb4:1 C5:1 D5:1 Bb4:1', 'C5:1 Bb4:1 Bb4:1 A4:1 A4:3 D5:1', 'G4:4 G5:2 r:2']
B1 = ['D5:4 D5:4', 'Eb5:6 D5:2', 'r:2 C5:4 B4:1 C5:1', 'D5:1 C5:1 B4:1 D5:1 C5:2 r:2',
      'C5:4 C5:4', 'D5:6 C5:2', 'r:2 Bb4:4 A4:1 Bb4:1', 'C5:1 Bb4:1 A4:1 C5:1 Bb4:2 r:2']
BR = ['A4:4 A4:4', 'C5:2 Bb4:4 A4:2', 'A4:2 G4:4 F#4:1 G4:1', 'A4:1 G4:1 F#4:1 A4:1 G4:4']
B2 = ['D5:2 r:5 D4:1', 'E4:2 r:2 F#4:2 r:2', 'G4:2 G4:4 F#4:1 G4:1', 'A4:1 G4:1 F#4:1 A4:1 G4:2 r:2']
TV = ['G5:2 D5:2 D5:2 D5:2', 'D5:2 C5:2 B4:2 C5:2', 'D5:2 D5:2 D5:2 D5:2', 'G5:2 D5:2 D5:2 D5:2', 'D5:2 C5:2 B4:2 A4:2', 'G4:2 G4:2 G4:2 G4:2']
TV_H = [['D5', 'B4', 'B4', 'B4'], ['B4', 'A4', 'G4', 'A4'], [['G4', 'B4']] * 4, ['D5', 'B4', 'B4', 'B4'], ['B4', 'A4', 'G4', 'F#4'],
        [['B3', 'D4']] * 4]                    # harmony under the trio melody (violins II), per eighth
D1 = ['B4:4 C5:4', 'B4:4 A4:4']
V1 = ['B5:2 B5:2 C6:2 C6:2', 'D6:2 G6:2 D6:4']
V1_H = [[['B4', 'G5'], ['B4', 'G5'], ['C5', 'A5'], ['C5', 'A5']], [['D5', 'B5'], ['G5', 'B5'], ['D5', 'B5']]]
D2 = ['B4:4 C5:4', 'D5:4 D5:4']
V2 = ['D6:2 C6:2 B5:2 A5:2', 'D5:2 D6:2 r:4']
V2_H = [[['D5', 'B5'], ['C5', 'A5'], ['B4', 'G5'], ['A4', 'F#5']], [['G4', 'B4'], ['G5', 'B5']]]
A3b = ['D4:3 G4:1 Bb4:1 D5:1 G5:1 Bb5:1', 'D6:6 Bb5:2', 'A5:6 Bb5:1 C6:1', 'Bb5:8', 'Eb6:1 F6:1 G6:1 Eb6:1 D6:1 Eb6:1 F6:1 D6:1',
       'C6:1 D6:1 Eb6:1 C6:1 Bb5:1 C6:1 D6:1 Bb5:1', 'C6:1 Bb5:1 Bb5:1 A5:1 A5:3 D6:1', 'G5:4 G6:2 r:2']
CODA = ['Eb6:2 r:2 D6:2 r:2', 'G5:8']

# ---------------------------------------------------------------- bass (cop) + off-beat chords ----------------------
# per bar: list of (pos16, dur16, kind, pitches); kind 'b' = bass note (the cop strikes it), 'c' = off-beat chord ('pah')
def ump(b1, c1, b2, c2):
    return [(0, 2, 'b', b1), (2, 2, 'c', c1), (4, 2, 'b', b2), (6, 2, 'c', c2)]
Gm = ump(['G2'], ['Bb2', 'D3'], ['D2'], ['Bb2', 'D3']); Cm = ump(['G2'], ['C3', 'Eb3'], ['C2'], ['C3', 'Eb3'])
D7 = ump(['F#2'], ['C3', 'D3'], ['D2'], ['C3', 'D3'])
BA1 = [Gm, Gm, Cm, Gm, Cm, Gm, D7, Gm]
BA2 = [Gm, Gm, Cm, Gm, ump(['G3'], ['Bb3', 'Eb4'], ['F3'], ['A3', 'D4']), ump(['Eb3'], ['G3', 'C4'], ['D3'], ['G3', 'Bb3']),
       [(0, 2, 'b', ['D2']), (2, 2, 'c', ['F#3', 'D4']), (6, 2, 'c', ['D3', 'F3', 'C4'])],
       [(0, 4, 'b', ['G3']), (4, 2, 'b', ['G2', 'D3', 'G3'])]]
BB1 = [ump(['B2'], ['F3', 'G3', 'B3'], ['G2'], ['F3', 'G3', 'B3']), ump(['B2'], ['G3', 'B3'], ['G2'], ['F3', 'G3', 'B3']),
       ump(['C3'], ['Eb3', 'G3', 'C4'], ['G2'], ['Eb3', 'G3', 'C4']), ump(['C3'], ['Eb3', 'G3', 'C4'], ['Bb2'], ['C3', 'Eb3', 'G3']),
       ump(['A2'], ['Eb3', 'F3', 'A3'], ['F2'], ['Eb3', 'F3', 'A3']), ump(['A2'], ['F3', 'A3'], ['F2'], ['Eb3', 'F3', 'A3']),
       ump(['Bb2'], ['D3', 'F3', 'Bb3'], ['F2'], ['D3', 'F3', 'Bb3']), ump(['Bb2'], ['D3', 'F3', 'Bb3'], ['F2'], ['D3', 'F3', 'Bb3'])]
BBR = [ump(['F#3'], ['A3', 'C4', 'D4'], ['D3'], ['A3', 'C4', 'D4']), ump(['F#3'], ['A3', 'D4'], ['D3'], ['A3', 'C4', 'D4']),
       ump(['Eb3'], ['G3', 'Bb3', 'Eb4'], ['Bb2'], ['G3', 'Bb3', 'Eb4']), ump(['Eb3'], ['G3', 'Bb3', 'Eb4'], ['Bb2'], ['G3', 'Bb3', 'D4'])]
BB2 = [[(0, 4, 'b', ['F#2', 'F#3']), (7, 1, 'b', ['D2', 'D3'])], [(0, 2, 'b', ['E2', 'E3']), (4, 2, 'b', ['F#2', 'F#3'])],
       [(0, 2, 'b', ['G1', 'G2']), (2, 4, 'c', ['D3', 'G3', 'Bb3'])], [(0, 4, 'b', ['C3', 'Eb3', 'G3']), (4, 4, 'b', ['G2', 'Bb2', 'D3', 'G3'])]]
BTV = [[(0, 2, 'b', ['G2', 'G3']), (2, 2, 'b', ['D3']), (4, 2, 'b', ['G2', 'G3']), (6, 2, 'b', ['D3'])]] * 6
def dol(c1, c2): return [(0, 2, 'b', ['G2']), (2, 2, 'b', ['G2', 'D3']), (4, 2, 'b', ['G2']), (6, 2, 'b', ['G2', 'D3']),
                         (2, 2, 'c', c1), (6, 2, 'c', c2)]
BD1 = [dol(['D4', 'G4'], ['E4', 'G4']), dol(['D4', 'G4'], ['D4', 'F#4'])]
BV1 = [[(0, 2, 'b', ['G2']), (2, 2, 'b', ['G3']), (4, 2, 'b', ['E3']), (6, 2, 'b', ['C3'])],
       [(0, 2, 'b', ['G2', 'G3']), (2, 2, 'c', ['G3', 'B3', 'D4']), (6, 2, 'c', ['G3', 'B3', 'D4'])]]
BD2 = [dol(['D4', 'G4'], ['E4', 'G4']), dol(['A4', 'C5'], ['G4', 'B4'])]
BV2 = [[(0, 2, 'b', ['G2']), (2, 2, 'b', ['C3']), (4, 2, 'b', ['D3']), (6, 2, 'b', ['D3'])],
       [(0, 4, 'b', ['G2', 'D3', 'G3']), (4, 2, 'c', ['G3', 'B3', 'D4'])]]
BA3b = [Gm, Gm, Cm, Gm, BA2[4], BA2[5], BA2[6], [(0, 4, 'b', ['G2', 'G3']), (4, 4, 'b', ['G1', 'G2'])]]
BCODA = [[(0, 2, 'b', ['C2', 'A2', 'Eb3']), (4, 2, 'b', ['D2', 'A2', 'D3'])], [(0, 8, 'b', ['G2', 'Bb2', 'D3', 'Bb3'])]]
CODA_CH = [[['G5', 'A5'], ['F#5', 'A5']], [['G4', 'Bb4', 'D5']]]       # right-hand chord tones under the coda melody

# ---------------------------------------------------------------- sections + time map ---------------------------------
# deck = rolling over equally spaced slats (spacing d per eighth) with constant rolling-friction deceleration,
# landing speed v0, creeping over the deck's edge with vE, then a free fall of height hfall to the next note.
DECKS = {'BR': dict(v0=2.0, tau1=2 * S16, n8=16, vE=0.13, hfall=0.95),
         'D1': dict(v0=2.3, tau1=0.30, n8=8, vE=0.13, hfall=0.95),
         'D2': dict(v0=2.3, tau1=0.30, n8=8, vE=0.13, hfall=0.95)}
for k, d in DECKS.items():
    d['d'] = d['v0'] * d['tau1']; d['L'] = d['n8'] * d['d']
    d['a'] = (d['v0'] ** 2 - d['vE'] ** 2) / (2 * d['L']); d['T'] = (d['v0'] - d['vE']) / d['a']
    d['tfall'] = math.sqrt(2 * d['hfall'] / G)
SECT = [  # name, melody bars, bass bars, timing, extra
    ('A', A1 + A2, BA1 + BA2, ('c', S16), 108),
    ('B', B1, BB1, ('c', S16), 104),
    ('BR', BR, BBR, ('deck',), 84),
    ('B2', B2, BB2, ('c', S16), 108),
    ('TV', TV + TV, BTV + BTV, ('c', S16V), 112),
    ('D1', D1, BD1, ('deck',), 76),
    ('V1', V1, BV1, ('c', S16V), 116),
    ('D2', D2, BD2, ('deck',), 76),
    ('V2', V2, BV2, ('c', S16V), 116),
    ('A3', A1 + A3b, BA1 + BA3b, ('c', S16), 116),
    ('coda', CODA, BCODA, ('c', S16), 124)]
GAP_BEFORE = {'D1': 0.45}                 # last vivace eighth before the first deck: the hop down onto the deck (Q=40 in the score)

secs = []; t_cur = T0; bar_t = []         # bar_t: list of (section, bar index in section, function pos16 -> time)
for name, M, Bs, tim, vel in SECT:
    t_start = t_cur + GAP_BEFORE.get(name, 0.0)
    if tim[0] == 'c':
        s16 = tim[1]
        def tf(sec_pos, t_start=t_start, s16=s16): return t_start + sec_pos * s16
        t_end = t_start + 8 * len(M) * s16
    else:
        dk = DECKS[name]
        def tf(sec_pos, t_start=t_start, dk=dk):
            x = sec_pos / 2 * dk['d']             # position along the deck (pos in 16ths -> eighths)
            disc = max(0.0, dk['v0'] ** 2 - 2 * dk['a'] * x)
            return t_start + (dk['v0'] - math.sqrt(disc)) / dk['a']
        t_edge = t_start + dk['T']; dk['t_start'] = t_start; dk['t_edge'] = t_edge
        t_end = t_edge + dk['tfall']               # next note = landing after tipping over the edge
    secs.append(dict(name=name, t=t_start, t_end=t_end, bars=len(M), vel=vel, tf=tf))
    t_cur = t_end

lead, bass, notes, pahs = [], [], [], []


def N(t, d, role, p, v): notes.append([round(t, 5), round(max(0.03, d), 5), role, int(p), int(max(1, min(127, v)))])


for si, (name, M, Bs, tim, vel) in enumerate(SECT):
    S = secs[si]; tf = S['tf']
    for bi, bar in enumerate(M):
        for pos, d, p in mel(bar):
            if p is None: continue
            sp = bi * 8 + pos
            t = tf(sp); te = tf(min(sp + d, 8 * len(M))) if not (tim[0] == 'deck' and sp + d >= 8 * len(M)) else S['t_end']
            lead.append(dict(t=round(t, 5), p=p, d=round(te - t, 5), sec=name, bar=bi, pos=pos, dur16=d, v=vel))
        for pos, d, kind, ps in Bs[bi]:
            sp = bi * 8 + pos; t = tf(sp); te = tf(min(sp + d, 8 * len(M)))
            ev = dict(t=round(t, 5), p=[pitch(x) for x in ps], d=round(max(0.08, te - t), 5), sec=name, bar=bi, pos=pos, v=vel)
            (bass if kind == 'b' else pahs).append(ev)
lead.sort(key=lambda e: e['t']); bass.sort(key=lambda e: e['t']); pahs.sort(key=lambda e: e['t'])

# harmony tones under the melody (sounded by the same tile strike: the violins play the double-stops)
HARM = {}
def put_h(sec, bars_h, per='e'):
    S = [s for s in secs if s['name'] == sec][0]
    for bi, hb in enumerate(bars_h):
        for k, h in enumerate(hb):
            t = round(S['tf'](bi * 8 + 2 * k), 5); HARM.setdefault(t, []).extend([pitch(x) for x in (h if isinstance(h, list) else [h])])
put_h('TV', TV_H + TV_H); put_h('V1', [V1_H[0], [V1_H[1][0], V1_H[1][1], V1_H[1][2]]])
put_h('V2', V2_H)
S = [s for s in secs if s['name'] == 'coda'][0]
for t_, hs in ((S['tf'](0), CODA_CH[0][0]), (S['tf'](4), CODA_CH[0][1]), (S['tf'](8), CODA_CH[1][0])):
    HARM.setdefault(round(t_, 5), []).extend(pitch(x) for x in hs)

# ---------------------------------------------------------------- sound roles ---------------------------------------
FF = ('A3', 'coda')
for e in lead:
    t, p, d, v, sec = e['t'], e['p'], e['d'], e['v'], e['sec']
    kind = 'bell' if (sec == 'B' and e['bar'] < 4) or sec == 'coda' else 'tile'
    e['kind'] = kind
    if kind == 'tile': N(t, d, 'xylo', p + 12 if p < 79 else p, v)
    else: N(t, d, 'bell', p + 12 if p < 72 else p, v)
    role = 'vsus' if d >= 0.36 else 'vspic'
    N(t, d * 0.95, role, p, v - 6)
    if sec in FF and e['bar'] >= 8 or sec == 'coda': N(t, d * 0.9, 'vspic', p - 12, v - 18)
    for h in HARM.get(t, []): N(t, d * 0.95, 'vspic' if d < 0.36 else 'vsus', h, v - 22)
for e in bass:
    t, ps, d, v = e['t'], e['p'], e['d'], e['v']
    lo = min(ps)
    N(t, d, 'mar', lo + 12, v - 10)                 # the struck copper plate (bass marimba)
    for p in ps:
        N(t, d, 'cb', p, v); N(t, d * 0.8, 'cpizz', p + 12 if p < 48 else p, v - 12)
    if e['sec'] in FF or e['sec'] in ('TV', 'V1', 'V2'): N(t, min(d, 0.3), 'tuba', lo, v - 10)
for e in pahs:
    for p in e['p']: N(e['t'], min(e['d'], 0.25), 'pah', p, e['v'] - 24)

notes.sort(key=lambda n: n[0])
FINAL = lead[-1]['t']
json.dump(notes, open('notes.json', 'w'))
json.dump(lead, open('lead.json', 'w'), indent=0)
json.dump(bass, open('bass.json', 'w'), indent=0)
json.dump(dict(G=G, t0=T0, s16=S16, s16v=S16V, final=FINAL, pahs=[[e['t'], e['p']] for e in pahs],
               secs=[{k: v for k, v in s.items() if k != 'tf'} for s in secs], decks=DECKS), open('events.json', 'w'), indent=1)
print('lead', len(lead), 'bass', len(bass), 'pah', len(pahs), 'final', round(FINAL, 2))
for s in secs: print(f"{s['name']:5s} {s['t']:7.2f} -> {s['t_end']:7.2f}")
for k, d in DECKS.items(): print(k, {kk: round(vv, 3) for kk, vv in d.items()})
