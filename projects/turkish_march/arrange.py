"""Rondo alla Turca (W.A. Mozart, Sonata K.331, 1783 - public domain) - own arrangement for an Ottoman clockwork band.
A minor / A major, 2/4, quarter = 126. Lead (brass bell-keys struck by marbles) = Mozart's right-hand melody."""
import json
Q = 60 / 126; S16 = Q / 4; BAR = 2 * Q; T0 = 1.0
notes, lead, drum, crash = [], [], [], []
# melody: lists of (midi or None, length in 16ths) per bar
PICK = [(71, 1), (69, 1), (68, 1), (69, 1)]
A1 = [[(72, 2), (None, 2), (74, 1), (72, 1), (71, 1), (72, 1)],
      [(76, 2), (None, 2), (77, 1), (76, 1), (75, 1), (76, 1)],
      [(83, 1), (81, 1), (80, 1), (81, 1), (83, 1), (81, 1), (80, 1), (81, 1)],
      [(84, 4), (81, 2), (84, 2)],
      [(83, 2), (81, 2), (80, 2), (81, 2)],
      [(83, 2), (81, 2), (80, 2), (81, 2)],
      [(83, 2), (81, 2), (80, 2), (78, 2)],
      [(76, 4), (76, 2), (77, 2)]]
B1 = [[(79, 2), (79, 2), (81, 1), (79, 1), (77, 1), (76, 1)],
      [(74, 4), (76, 2), (77, 2)],
      [(79, 2), (79, 2), (81, 1), (79, 1), (77, 1), (76, 1)],
      [(74, 4), (72, 2), (74, 2)],
      [(76, 2), (76, 2), (77, 1), (76, 1), (74, 1), (72, 1)],
      [(71, 4), (72, 2), (74, 2)],
      [(76, 2), (76, 2), (77, 1), (76, 1), (74, 1), (72, 1)],
      [(71, 4)] + PICK]
A2 = [A1[0], A1[1], A1[2],
      [(84, 4), (81, 2), (83, 2)],
      [(84, 2), (83, 2), (81, 2), (80, 2)],
      [(81, 2), (76, 2), (77, 2), (74, 2)],
      [(72, 4), (71, 2), (69, 1), (71, 1)],
      [(69, 4), (81, 2), (83, 2)]]
CH = [[(85, 4), (81, 2), (83, 2)],
      [(85, 2), (83, 2), (81, 2), (80, 2)],
      [(78, 2), (80, 2), (81, 2), (83, 2)],
      [(80, 2), (76, 2), (81, 2), (83, 2)],
      [(85, 4), (81, 2), (83, 2)],
      [(85, 2), (83, 2), (81, 2), (80, 2)],
      [(78, 2), (83, 2), (80, 2), (76, 2)],
      [(81, 8)]]
CODA = [[(76, 1), (81, 1), (85, 1), (81, 1), (76, 1), (81, 1), (85, 1), (81, 1)],
        [(83, 1), (80, 1), (76, 1), (80, 1), (83, 1), (80, 1), (76, 1), (80, 1)],
        [(81, 2), (85, 2), (83, 2), (80, 2)],
        [(81, 8)]]
# harmony per bar: (bass root midi, chord tones) for the left hand
Am, E, C, G, Dm, A, F, E7 = (45, (57, 60, 64)), (40, (56, 59, 64)), (48, (55, 60, 64)), (43, (55, 59, 62)), (50, (57, 62, 65)), (45, (57, 61, 64)), (41, (57, 60, 65)), (40, (56, 59, 62))
H_A1 = [Am, Am, Am, Am, E, E, E, E]
H_B1 = [C, G, C, G, Am, E, Am, E]
H_A2 = [Am, Am, Am, F, Am, Dm, E7, Am]
H_CH = [A, A, E, E, A, A, E, A]
H_CO = [A, E, A, A]
SEC = []
bar = 0
def put_bar(b, mel, harm, part, var):
    t0 = T0 + b * BAR; pos = 0
    for p, n in mel:
        if p is not None:
            v = {'A': 86, 'B': 80, 'CH': 108, 'CO': 112}[part] + (8 if var else 0)
            lead.append([t0 + pos * S16, p, min(122, v)])
            notes.append([t0 + pos * S16, n * S16, 'lead', p, min(122, v)])
            if part in ('CH', 'CO') or (var and part == 'A'):
                notes.append([t0 + pos * S16, n * S16, 'pdouble', p - 12, min(118, v - 10)])
        pos += n
    root, ch = harm
    vv = 70 + (25 if part in ('CH', 'CO') else 0) + (8 if var else 0)
    for e in range(4):                       # alla-turca left hand: bass on beats, chord on off-eighths
        te = t0 + e * 2 * S16
        if e % 2 == 0: notes.append([te, 2 * S16, 'pbass', root, vv]); notes.append([te, 2 * S16, 'pbass', root - 12, vv - 10])
        else:
            for c in ch: notes.append([te, 1.6 * S16, 'pchord', c, vv - 12])
    if part == 'CH' or part == 'CO':
        notes.append([t0, 2 * S16, 'cb', root - 12, 100])
for i, (p, n) in enumerate(PICK):     # pickup before bar 0 of the first A
    pass
def section(name, bars, harm, part, var):
    global bar
    SEC.append([name, T0 + bar * BAR])
    for k, mel in enumerate(bars): put_bar(bar, mel, harm[k], part, var); bar += 1
# intro: 2 bars of janissary drums alone, pickup at the end of bar 2
bar = 2
SEC.append(['intro', T0])
for k, (p, n) in enumerate(PICK):
    t = T0 + 2 * BAR - (4 - k) * S16; lead.append([t, p, 84]); notes.append([t, S16, 'lead', p, 84])
section('A', A1, H_A1, 'A', False); section('B', B1, H_B1, 'B', False); section('A2', A2, H_A2, 'A', False)
section('CH', CH, H_CH, 'CH', False)
for k, (p, n) in enumerate(PICK):
    t = T0 + bar * BAR - (4 - k) * S16; lead.append([t, p, 92]); notes.append([t, S16, 'lead', p, 92])
section('Av', A1, H_A1, 'A', True); section('Bv', B1, H_B1, 'B', True); section('A2v', A2, H_A2, 'A', True)
section('CH2', CH, H_CH, 'CH', True)
section('CO', CODA, H_CO, 'CO', True)
FINAL = T0 + (bar - 1) * BAR
# the final chord
for p in (57, 61, 64, 69, 73): notes.append([FINAL, 3.5, 'pchord', p, 112])
for p in (33, 45): notes.append([FINAL, 3.5, 'pbass', p, 118])
# drums (davul) and cymbals (zil)
secs = dict(SEC); nb = bar
def tb(b, beat=0): return T0 + b * BAR + beat * Q
for b in range(nb):
    t = tb(b); name = [n for n, s in SEC if s <= t + 1e-6][-1]
    if name == 'intro': drum += [[tb(b, 0), 100], [tb(b, 1), 90], [tb(b, 1.5), 80]]
    elif name in ('A', 'A2'): drum.append([tb(b, 0), 78])
    elif name == 'B': pass
    elif name in ('Av', 'A2v'): drum += [[tb(b, 0), 92], [tb(b, 1), 70]]
    elif name == 'Bv': drum += [[tb(b, 0), 85], [tb(b, 1), 75]]
    elif name in ('CH', 'CH2', 'CO'):
        if t >= FINAL - 1e-6: continue
        drum += [[tb(b, 0), 112], [tb(b, 1), 100]]
        if name != 'CH': drum.append([tb(b, 1.5), 90])
for name in ('CH', 'CH2'):
    s = secs[name]; crash += [[s, 105], [s + 4 * BAR, 100]]
crash.append([secs['Av'], 90]); crash.append([secs['CO'], 105])
drum.append([FINAL, 125]); crash.append([FINAL, 125])
notes.sort(key=lambda n: n[0]); lead.sort(); drum.sort(); crash.sort()
json.dump(notes, open('notes.json', 'w')); json.dump(lead, open('lead.json', 'w'))
json.dump({'t0': T0, 'bar': BAR, 'q': Q, 'final': FINAL, 'secs': SEC, 'drum': drum, 'crash': crash}, open('events.json', 'w'))
print(len(lead), 'lead notes; final', round(FINAL, 2), 'pitches', sorted(set(p for _, p, _ in lead)))
