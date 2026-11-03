"""Chip Fab - original bright pop tune (D major), 120 BPM, 32 bars ~ 66 s.
Writes notes.json: [t, dur, role, pitch, vel] and events.json (timeline info for the scene)."""
import json
BPM = 120; B = 60 / BPM; BAR = 4 * B; T0 = 0.4
sec = lambda bar, beat=0.0: round(T0 + bar * BAR + beat * B, 4)
ev = []
def add(bar, beat, dur, role, p, v): ev.append([sec(bar, beat), round(dur * B, 3), role, int(p), int(v)])

def line(bar0, bars, role, vel=100, shift=0):
    """bars: lists of 8 eighth-slots: pitch | '-' hold | '.' rest"""
    for i, slots in enumerate(bars):
        for k, s in enumerate(slots):
            if isinstance(s, int):
                d = 1
                while k + d < 8 and slots[k + d] == '-': d += 1
                acc = vel + (8 if k % 2 == 0 else -6) + (6 if k == 0 else 0)
                add(bar0 + i, k * 0.5, d * 0.5, role, s + shift, min(127, acc))

_ = '-'; o = '.'
A = [[74, o, 74, 76, 78, _, 74, o],
     [76, o, 76, 78, 81, _, 76, o],
     [78, _, 79, 78, 76, _, 74, 71],
     [74, _, _, _, 71, o, 74, 76],
     [78, o, 78, 79, 81, _, 86, _],
     [85, _, 81, 78, 81, _, 76, o],
     [79, _, 78, 76, 76, 78, 81, 85],
     [86, _, _, o, 81, 78, 76, 73]]
Bm = [[79, o, 79, 81, 83, _, 79, o],
      [81, o, 81, 83, 85, _, 81, o],
      [86, _, 85, 83, 81, _, 78, _],
      [83, _, 81, 78, 74, _, 78, 81],
      [83, _, 86, _, 83, 81, 79, 78],
      [76, _, 81, _, 85, _, 88, _]]
CH_A = ['D', 'A', 'Bm', 'G', 'D', 'A', 'GA', 'D']
CH_B = ['Em', 'A', 'D', 'Bm', 'G', 'A']
ROOT = {'D': 38, 'C': 36, 'G': 43, 'A': 45, 'Bm': 47, 'Em': 40}
VOX = {'D': [62, 66, 69], 'C': [60, 64, 67], 'G': [62, 67, 71], 'A': [61, 64, 69], 'Bm': [62, 66, 71], 'Em': [64, 67, 71]}
ARP = {'D': [62, 66, 69, 74], 'C': [60, 64, 67, 72], 'G': [62, 67, 71, 74], 'A': [61, 64, 69, 73], 'Bm': [62, 66, 71, 74], 'Em': [64, 67, 71, 76]}

def chords_of(sym): return [sym] if sym in ROOT else [sym[0], sym[1]]

def bass(bar, sym, vel=104):
    cs = chords_of(sym)
    pat = [(0, 0, .6), (0.75, 0, .3), (1.5, 7, .4), (2.5, 0, .4), (3, 10, .3), (3.5, 7, .4)]
    for b, iv, d in pat:
        c = cs[0] if b < 2 or len(cs) == 1 else cs[1]
        r = ROOT[c]; iv = 12 if iv == 10 else iv
        add(bar, b, d, 'bass', r + iv, vel + (10 if b == 0 else 0))

def stabs(bar, sym, vel=92):
    cs = chords_of(sym)
    for b in (1.5, 3.5):
        c = cs[0] if b < 2 or len(cs) == 1 else cs[1]
        for p in VOX[c]: add(bar, b, 0.4, 'stab', p, vel)

def arp(bar, sym, vel=84, sub=0.5):
    cs = chords_of(sym); n = int(4 / sub)
    for k in range(n):
        b = k * sub; c = cs[0] if b < 2 or len(cs) == 1 else cs[1]
        seq = ARP[c] + ARP[c][2:0:-1]
        add(bar, b, sub, 'arp', seq[k % len(seq)], vel + (8 if k % 4 == 0 else 0))

# ---------- form ----------
# bars 0-1 intro | 2-9 A | 10-17 A' | 18-25 B (24-25 drum break) | 26-30 A'' | 31 final
arp(0, 'D', 76); arp(1, 'A', 80)
line(2, A, 'mel', 100)
line(10, A, 'mel', 104); line(10, A, 'xmel', 88)
line(18, Bm[:6], 'xlead', 104)
for i, c in enumerate(CH_B[:6]): arp(18 + i, c, 80)
line(26, A[:4] + [A[6]], 'mel', 110); line(26, A[:4] + [A[6]], 'xmel', 96)
for i, c in enumerate(CH_A): bass(2 + i, c); bass(10 + i, c)
for i, c in enumerate(CH_B[:6]): bass(18 + i, c)
for i, c in enumerate(['D', 'A', 'Bm', 'G', 'GA']): bass(26 + i, c, 110)
for i, c in enumerate(CH_A): stabs(10 + i, c)
for i, c in enumerate(['D', 'A', 'Bm', 'G', 'GA']): stabs(26 + i, c, 100)
add(0, 0, 1, 'bass', 38, 110); add(1, 0, 1, 'bass', 45, 110)
# drum break bars 24-25: bass hits with the toms, strings glissando up into A''
for b in (0, 1.5, 3): add(24, b, 0.5, 'bass', 38, 112)
for b in (0, 1.5): add(25, b, 0.5, 'bass', 45, 112)
run = [62, 64, 66, 67, 69, 71, 73, 74, 76, 78, 79, 81]
for k, p in enumerate(run): add(25, 1 + k * 0.25, 0.25, 'arp', p, 90 + k * 2)

# ---------- drums ----------  roles: kick snare hat ohat tom1 tom2 tom3 crash
def D(bar, beat, role, v): ev.append([sec(bar, beat), 0.1, role, 0, int(v)])
def groove(bar, full=True):
    for b in (0, 1.5, 2) if full else (0, 2.5): D(bar, b, 'kick', 118 if b == 0 else 104)
    for b in ((1, 3) if full else (2,)): D(bar, b, 'snare', 112)
    for k in range(8):
        b = k * 0.5
        if full and k == 7: D(bar, b, 'ohat', 96)
        else: D(bar, b, 'hat', 92 if k % 2 == 0 else 70)
def fill(bar, beat0, seq, step=0.25):
    for i, r in enumerate(seq): D(bar, beat0 + i * step, r, 108 + (i % 2) * 6)

D(0, 0, 'kick', 120); D(0, 0, 'crash', 110)
for k in range(8): D(0, k * 0.5, 'hat', 70 + k * 3)
D(1, 0, 'kick', 110); D(1, 1, 'snare', 96)
fill(1, 2, ['snare', 'snare', 'tom1', 'tom1', 'tom2', 'tom2', 'tom3', 'tom3'])
for bar in range(2, 24):
    groove(bar, full=not (18 <= bar < 22))
    if bar in (9, 17): fill(bar, 3, ['snare', 'tom1', 'tom2', 'tom3'])
    if bar == 21: fill(bar, 2, ['tom1', 'tom1', 'tom2', 'tom2', 'tom3', 'tom3', 'snare', 'snare'])
for b in (2, 10, 18, 22): D(b, 0, 'crash', 120)
# drum break
for bar in (24, 25):
    seq = ['kick', 'tom3', 'tom3', 'tom2', 'snare', 'tom2', 'tom1', 'tom1', 'kick', 'tom3', 'snare', 'tom2', 'snare', 'tom1', 'snare', 'snare']
    for i, r in enumerate(seq): D(bar, i * 0.25, r, 104 + (12 if i % 4 == 0 else 0))
D(24, 0, 'crash', 124)
for bar in range(26, 31):
    groove(bar)
D(26, 0, 'crash', 127); D(28, 0, 'crash', 120); D(30, 0, 'crash', 116)
fill(30, 2, ['snare', 'snare', 'tom1', 'tom1', 'tom2', 'tom2', 'tom3', 'tom3'])
# final
FIN = 31
D(FIN, 0, 'kick', 127); D(FIN, 0, 'crash', 127)
for p in (38, 50): add(FIN, 0, 6, 'final_bass', p, 124)
for p in (62, 66, 69, 74, 78, 81, 86): add(FIN, 0, 6, 'final_str', p, 120)
for p in (66, 69, 74, 78): add(FIN, 0, 6, 'final_sun', p, 118)

ev.sort(key=lambda e: e[0])
json.dump(ev, open('notes.json', 'w'))
roles = {}
for e in ev: roles[e[2]] = roles.get(e[2], 0) + 1
info = dict(beat=B, t0=T0, bar=BAR, final=sec(FIN),
            sections=[[sec(0), 'intro'], [sec(2), 'A'], [sec(10), 'A2'], [sec(18), 'B'], [sec(24), 'break'], [sec(26), 'A3'], [sec(FIN), 'final']])
json.dump(info, open('events.json', 'w'))
print(roles, info['final'])
for r in ('mel', 'xmel', 'xlead', 'arp', 'stab', 'bass'):
    print(r, sorted({e[3] for e in ev if e[2] == r}))
