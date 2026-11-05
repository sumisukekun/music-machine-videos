"""Carol of the Bells (M. Leontovych, 'Shchedryk', 1916 - public domain) - own virtuoso arrangement.
G minor, 3/4, bar = 1.0 s. Lead line (glockenspiel) = what the ball rolls over."""
import json
BEAT = 1 / 3; BAR = 1.0; T0 = 1.2
OST = [(0, 70), (1, 69), (1.5, 70), (2, 67)]                 # Bb A Bb G  (the bell motif)
UP3 = [(0, 74), (1, 72), (1.5, 74), (2, 70)]                 # in thirds above
BASS = [43, 41, 39, 38]                                      # G F Eb D (one per bar, repeating)
CH = [(67, 70, 74), (65, 69, 72), (63, 67, 70), (62, 66, 69)]  # Gm F Eb D
notes = []      # [t, dur, role, midi, vel]
lead = []       # [t, midi, vel]  (ball track)
def t_(b, beat): return T0 + b * BAR + beat * BEAT
def L(b, beat, p, v, d=0.3): lead.append([t_(b, beat), p, v]); notes.append([t_(b, beat), d, 'glock', p, v])
def N(b, beat, d, role, p, v): notes.append([t_(b, beat), d, role, p, v])
SEC = [('intro', 0, 4), ('A', 4, 12), ('B', 12, 20), ('run1', 20, 28), ('calm', 28, 36),
       ('run2', 36, 48), ('final', 48, 60), ('outro', 60, 64)]
for name, b0, b1 in SEC:
    for b in range(b0, b1):
        k = b % 4; r, th, fi = CH[k]; vb = 70 + 50 * (b - b0) / max(1, b1 - b0 - 1)
        if name == 'intro':
            for bt, p in OST: L(b, bt, p + 12, 55 + 8 * b)
        elif name == 'A':
            for bt, p in OST: L(b, bt, p + 12, 80); N(b, bt, 0.25, 'vspic', p, 75)
            N(b, 0, 2.9 * BEAT, 'cello', BASS[k] + 12, 70)
        elif name == 'B':
            for bt, p in UP3: L(b, bt, p + 12, 88)
            for bt, p in OST: N(b, bt, 0.25, 'vspic', p + 12, 80)
            N(b, 0, 2.9 * BEAT, 'cello', BASS[k] + 12, 80); N(b, 0, 0.5, 'pbass', BASS[k], 85)
            for i in range(3): N(b, i, 0.2, 'sleigh', 0, 70 + 20 * (i == 0))
        elif name in ('run1', 'run2'):
            seq = [82, 81, 82, 79] + [r + 24, fi + 12, th + 12, r + 12] + [th + 12, fi + 12, r + 24, th + 24]
            if name == 'run2' and (b - b0) % 2: seq = [r + 12, th + 12, fi + 12, r + 24, th + 24, fi + 24, r + 36, fi + 24, th + 24, r + 24, fi + 12, th + 12]
            for i, p in enumerate(seq): L(b, i * 0.25, min(p, 98), int(min(118, vb)), 0.15)
            for bt, p in OST: N(b, bt, 0.25, 'vspic', p + 12, int(vb))
            N(b, 0, 2.9 * BEAT, 'cello', BASS[k] + 12, int(vb)); N(b, 0, 0.5, 'pbass', BASS[k], int(vb))
            N(b, 0, 0.5, 'timp', BASS[k], int(vb))
            if name == 'run2':
                for i in range(6): N(b, i * 0.5, 0.2, 'sleigh', 0, 60 + 30 * (i == 0))
                for p in (r, th, fi): N(b, 0, 2.9 * BEAT, 'vsus', p + 12, int(vb * 0.7))
            if name == 'run2' and b >= b1 - 2: N(b, 0, 1.0, 'troll', 43, int(vb))
        elif name == 'calm':
            for bt, p in OST: L(b, bt, p + 24, 62)
            for p in (r, th, fi): N(b, 0, 2.95 * BEAT, 'pchord', p, 55)
            N(b, 0, 0.5, 'cpizz', BASS[k] + 12, 75)
        elif name == 'final':
            for bt, p in UP3: L(b, bt, p + 12, 115); N(b, bt, 0.3, 'glock2', p, 100)
            for bt, p in OST: N(b, bt, 0.25, 'vspic', p + 12, 112); N(b, bt, 0.25, 'cspic', p - 12, 100)
            for p in (r, th, fi): N(b, 0, 2.9 * BEAT, 'vsus', p + 12, 95)
            N(b, 0, 2.9 * BEAT, 'cello', BASS[k], 110); N(b, 0, 0.6, 'pbass', BASS[k], 115); N(b, 0, 0.6, 'timp', BASS[k], 118)
            N(b, 2, 0.3, 'timp', BASS[k], 90)
            for i in range(6): N(b, i * 0.5, 0.2, 'sleigh', 0, 70 + 35 * (i == 0))
            if b == b0: N(b, 0, 2, 'crash', 0, 110)
        elif name == 'outro':
            v = 90 - 15 * (b - b0)
            for bt, p in OST: L(b, bt, p + 12, v); N(b, bt, 0.25, 'vspic', p, v - 10)
            N(b, 0, 2.9 * BEAT, 'cello', BASS[k] + 12, v - 10)
fb = 64; tf = t_(fb, 0)
for p in (67, 74, 79, 82, 86, 91): notes.append([tf, 3.5, 'glock2' if p > 78 else 'pchord', p, 100])
lead.append([tf, 91, 120]); notes.append([tf, 3.5, 'glock', 91, 120])
for p in (31, 43): notes.append([tf, 3.5, 'pbass', p, 110])
notes.append([tf, 3.5, 'cello', 43, 100]); notes.append([tf, 1, 'timp', 43, 120]); notes.append([tf, 1, 'bt', 0, 100])
notes.append([tf, 1, 'crash', 0, 90])
notes.sort(key=lambda n: n[0]); lead.sort()
json.dump(notes, open('notes.json', 'w')); json.dump(lead, open('lead.json', 'w'))
json.dump({'t0': T0, 'bar': BAR, 'beat': BEAT, 'final': tf, 'secs': [[n, t_(a, 0), t_(b, 0)] for n, a, b in SEC], 'bass': BASS}, open('events.json', 'w'))
print(len(notes), len(lead), tf)
