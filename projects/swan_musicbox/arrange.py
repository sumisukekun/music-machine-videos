"""Swan Lake - Act II 'Scene' (Tchaikovsky 1876, public domain): the full oboe theme (A A' B B' A'')
in the original key B minor, own music-box + orchestra arrangement.
Melody transcribed from the theme (ABC source, Am) and transposed +2 to B minor.
The music box comb plays melody + bass (every comb note = one pin plucking one tooth)."""
import json, re
BPM = 80; BEAT = 60 / BPM; BAR = 4 * BEAT; T0 = 1.6
PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
def parse(s):
    """tiny ABC subset, L=1/16, one bar per '|' ; returns list of bars [(beat, dur_beats, midi)]"""
    bars = []
    for b in s.split('|'):
        pos = 0; acc = {}; out = []
        for m in re.finditer(r"(\^|_|=)?([A-Ga-g])(\d+)(-?)", b):
            a, l, n, tie = m.groups(); n = int(n)
            base = 60 + PC[l.upper()] + (12 if l.islower() else 0)
            if a: acc[l] = {'^': 1, '_': -1, '=': 0}[a]
            p = base + acc.get(l, 0) + 2                      # -> B minor
            out.append([pos / 4, n / 4, p]); pos += n
        if out: bars.append(out)
    return bars
A1 = parse("e8 A2B2 c2d2 | e6 c2 e6 c2 | e6 A2 c2A2 F2c2 | A10 d2 c2B2")
A2 = parse("e8 A2B2 c2d2 | e6 c2 e6 c2 | e6 A2 c2A2 F2c2 | A16")
B1 = parse("B4 c4 d4 e2f2 | g6 f2 e4 f2g2 | a6 g2 f4 g2a2 | b6 a2 e2c2 B2A2")
B2 = parse("B4 c4 d4 e2f2 | g6 f2 e4 f2g2 | a6 g2 f4 g2a2 | _b6 f2 d4 f2_b2 | b6 ^f2 b6 e2")
A3 = parse("e8 A2B2 c2d2 | e6 c2 e6 c2 | e6 A2 c2^G2 F2c2 | A16")
# harmony per half bar (already in B minor)
Bm, G, F7, A, Fm, Em, C = [11, 2, 6], [7, 11, 2], [6, 10, 1, 4], [9, 1, 4], [6, 9, 1], [4, 7, 11], [0, 4, 7]
H_A1 = [[Bm, Bm], [Bm, Bm], [Bm, G], [Bm, F7]]
H_A2 = [[Bm, Bm], [Bm, Bm], [Bm, G], [Bm, Bm]]
H_B1 = [[A, A], [Fm, Fm], [Em, G], [Bm, Bm]]
H_B2 = [[A, A], [Fm, Fm], [Em, G], [C, C], [F7, F7]]
H_A3 = [[Bm, Bm], [Bm, Bm], [Bm, F7], [Bm, Bm]]
SECS = [('A1', A1, H_A1, 1), ('A2', A2, H_A2, 2), ('B1', B1, H_B1, 3), ('B2', B2, H_B2, 4), ('A3', A3, H_A3, 5)]
def tb(bar, beat): return T0 + bar * BAR + beat * BEAT
notes = []    # [t, dur, role, midi, vel]
comb = []     # [t, midi, vel]   pins / teeth
def N(t, d, role, p, v): notes.append([round(t, 4), round(d, 4), role, int(p), int(max(1, min(127, v)))])
def CB(t, p, v, d=1.5): comb.append([round(t, 4), int(p), int(v)]); N(t, d, 'box', p, v)
def bass_of(ch): return 47 + (ch[0] - 11) % 12 - (12 if (ch[0] - 11) % 12 > 6 else 0)   # around B2..F#3
# intro bar 0: music box arpeggio on B minor (pins only)
for k, p in enumerate([59, 62, 66, 71, 74, 78, 74, 71]):
    CB(tb(0, k * 0.5), p, 60 + 3 * k)
bar = 1; secs = []; final_bar = None
for name, mel, harm, dyn in SECS:
    secs.append([name, tb(bar, 0)])
    for i, (mb, hb) in enumerate(zip(mel, harm)):
        last = (name == 'A3' and i == len(mel) - 1)
        if last: final_bar = bar; break
        for beat, d, p in mb:
            v = [0, 78, 84, 90, 98, 108][dyn] + (6 if beat == 0 else 0)
            CB(tb(bar, beat), p, v)
            if dyn >= 2: N(tb(bar, beat), d * BEAT * 0.97, 'vsus', p - 12, 40 + dyn * 10)          # strings double an octave below
            if dyn >= 5: N(tb(bar, beat), d * BEAT * 0.97, 'horn', p - 12, 92)
        for h, ch in enumerate(hb):
            t = tb(bar, h * 2); b = bass_of(ch)
            CB(t, b, 72 + dyn * 6); 
            if dyn == 1: CB(tb(bar, h * 2 + 1), b + 7 if (ch[1] - ch[0]) % 12 in (3, 4) else b + 7, 52)   # music-box bass fifth
            if dyn >= 2:
                N(t, 2 * BEAT * 0.98, 'cello', b - 12 if dyn >= 4 else b, 50 + dyn * 9)
                for pc in ch[:3]: N(t, 2 * BEAT * 0.98, 'vpad', 62 + (pc - 2) % 12, 30 + dyn * 8)
            if dyn in (2, 3):
                for k in range(4): N(t + k * BEAT * 0.5, 0.3, 'pizz', b + (0, 7, 12, 7)[k], 50 + dyn * 5)
            if dyn >= 4:
                N(t, 0.4, 'timp', b, 70 + dyn * 8)
                if dyn == 5: N(t + BEAT, 0.3, 'timp', b, 70)
        bar += 1
# swell into A3 (last bar of B2): timpani roll
for k in range(16): N(tb(bar - 1, 2) + k * BEAT / 8, 0.2, 'timp', 42, 40 + k * 5)
N(tb(bar, 0), 2, 'crash', 0, 90)
# final chord: B minor, everything
tf = tb(final_bar, 0)
for p in (47, 59, 66, 71, 74, 78, 83): CB(tf, p, 112, 4.5)
for p in (59, 62, 66, 71): N(tf, 4.0, 'vpad', p, 100)
for p in (54, 59, 62): N(tf, 4.0, 'horn', p, 100)
N(tf, 4.0, 'cello', 35, 110); N(tf, 1, 'timp', 47, 120); N(tf, 3, 'crash', 0, 105); N(tf, 1, 'bt', 0, 100)
for k in range(10): CB(tf - 0.9 + k * 0.09, (59, 62, 66, 71, 74, 78, 83, 86, 90, 95)[k], 70 + 3 * k, 1.2) if False else None
notes.sort(key=lambda n: n[0]); comb.sort()
json.dump(notes, open('notes.json', 'w')); json.dump(comb, open('comb.json', 'w'))
json.dump({'t0': T0, 'bar': BAR, 'beat': BEAT, 'final': tf, 'secs': secs,
           'bass': sorted({n[0] for n in notes if n[2] == 'cello'})}, open('events.json', 'w'))
print('notes', len(notes), 'comb', len(comb), 'final', round(tf, 2), 'end', round(tf + 4.5, 2), 'teeth', sorted({c[1] for c in comb}))
