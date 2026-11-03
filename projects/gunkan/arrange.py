"""Warship March (Gunkan koushinkyoku, Setoguchi Tokichi 1897 / trio 'Umi yukaba' Tougi Sueyoshi) - public domain.
Melody + chord symbols re-encoded from a plain lead sheet; arrangement for a naval-gun ball machine
(military band: trumpets, horns, trombones, tuba, glockenspiel, snare, bass drum, cymbals, timpani).
Outputs events.json (every ball / hit) and notes.json (audio, rendered by render_audio.py)."""
import json

TR = 3                                     # D -> F major (the march's Taisho key)
MEL0 = json.load(open('gunkan_melody.json'))   # [beat, dur, pitch] in D, 32 bars (A strain 16 + trio 16)
CH = ('d d d d a:7 a:7 a:7 a:7 a:7 a:7 a:7 a:7 d d d d d d d d d d d d d g d d a:7 a:7 d d '
      'd d d d d d d d d d g g d d d d d d d d g g g d d d a:7 a:7 d d d d').split()
ROOT = {'d': 2, 'g': 7, 'a:7': 9}
PCS = {'d': [2, 6, 9], 'g': [7, 11, 2], 'a:7': [1, 4, 7, 9]}
BPM = 150.0; B = 60 / BPM; T0 = 1.6
off = T0 / B
sec = lambda b: round((b + off) * B, 4)

ev = []
def add(b, d, ins, p, v, role):
    ev.append((round(b, 4), d, ins, int(p), int(max(1, min(127, v))), role))
H = {k: [] for k in ('mel', 'harm', 'chord', 'bass', 'kick', 'snare', 'tom', 'crash', 'hat', 'gliss')}

INTRO = 8                                  # 2 bars of fanfare
A0 = INTRO                                 # A strain (16 bars)
TRIO0 = A0 + 64                            # trio (16 bars)
FIN0 = TRIO0 + 64                          # last 4 bars of A
FINAL = FIN0 + 16                          # final chord
def chord_at(b):
    """chord symbol at absolute beat b (None in intro)"""
    if b < A0: return 'd' if b < 4 else ('a:7' if b >= 6 else 'd')
    if b >= FINAL: return 'd'
    if b >= FIN0: rel = 48 + (b - FIN0)          # bars 12-15 of A
    else: rel = b - A0
    return CH[int(rel // 2)]

# ---------- melody ----------
mel = []
for b, d, p in MEL0:
    if b < 64: mel.append((A0 + b, d, p + TR, 'A'))
    else: mel.append((TRIO0 + b - 64, d, p + TR, 'T'))
for b, d, p in MEL0:
    if 48 <= b < 60: mel.append((FIN0 + b - 48, d, p + TR, 'F'))
mel.append((FIN0 + 12, 1, 2 + 60 + TR, 'F'))       # resolve on D (F) - bar 15 beat 1
# intro fanfare (bugle call on the tonic triad)
for b, d, p in ((0, 0.75, 57), (0.75, 0.25, 57), (1, 1, 62), (2, 0.75, 57), (2.75, 0.25, 62), (3, 1, 66),
                (4, 1, 69), (5, 0.75, 66), (5.75, 0.25, 62), (6, 1, 64), (7, 0.5, 57), (7.5, 0.5, 57)):
    mel.append((b, d, p + TR, 'I'))
mel.sort()
mel = [m for m in mel if not (FIN0 + 13 <= m[0] < FINAL)]
for b, d, p, sect in mel:
    role = 'mel_trio' if sect == 'T' else 'mel'
    add(b, d * B, 'mel', p, 112 if sect != 'T' else 100, role); H['mel'].append((b, p))
mel_p = sorted({p for _, _, p, _ in mel})

# ---------- glissandi (bell lyre runs) ----------
def gliss(b0, b1, lo, hi):
    ps = [p for p in mel_p if lo <= p <= hi]
    for i, p in enumerate(ps):
        b = b0 + (b1 - b0) * i / (len(ps) - 1)
        add(b, 0.2, 'gliss', p, 80 + 40 * i / len(ps), 'gliss'); H['gliss'].append((b, p))
gliss(TRIO0 - 1.0, TRIO0 - 0.05, 60, 80)
gliss(FINAL - 1.0, FINAL - 0.05, 62, 82)

# ---------- chords: march 'pah' on beats 2 and 4 (horns) ----------
def voicing(sym):
    pcs = [(pc + TR) % 12 for pc in PCS[sym]]
    if sym == 'a:7': pcs = [pcs[0], pcs[1], pcs[2]]      # 3rd, 5th, 7th
    vs = sorted(57 + (pc - 57) % 12 for pc in pcs)
    return vs
for bar in range(FINAL // 4):
    for k in (1, 3):
        b = bar * 4 + k
        sym = chord_at(b); ps = voicing(sym)
        v = 92 if b < TRIO0 or b >= FIN0 else 78
        for p in ps: add(b, 0.5 * B, 'chord', p, v, 'chord')
        H['chord'].append((b, ps))
# intro: stabs on every beat of bar 1
for b in (0, 2, 4):
    ps = voicing(chord_at(b)); [add(b, 0.5 * B, 'chord', p, 118, 'chord') for p in ps]; H['chord'].append((b, ps))

# ---------- harmony: 3rd/6th below the melody (A strains); arpeggios in the trio ----------
for b, d, p, sect in mel:
    if sect not in ('A', 'F') or b < A0 + 16: continue
    pcs = {(pc + TR) % 12 for pc in PCS[chord_at(b)]}
    cand = [q for q in range(p - 9, p - 2) if q % 12 in pcs]
    if not cand: continue
    hp = max(cand)
    add(b, d * B, 'harm', hp, 92, 'harm'); H['harm'].append((b, hp))
for bar in range(16):                         # trio: bell-lyre broken chords in 8ths
    for k in range(8):
        b = TRIO0 + bar * 4 + k * 0.5
        sym = chord_at(b); pcs = [(pc + TR) % 12 for pc in PCS[sym]][:3]
        ch = sorted(72 + (pc - 72) % 12 for pc in pcs)
        seq = [ch[0], ch[1], ch[2], ch[1] + 12 if ch[1] + 12 <= 86 else ch[1]][k % 4]
        add(b, 0.4 * B, 'harm', seq, 62 + (10 if k % 2 == 0 else 0), 'harm_arp'); H['harm'].append((b, seq))

# ---------- bass: tuba root / fifth on beats 1 and 3 ----------
for bar in range(FINAL // 4):
    for k in (0, 2):
        b = bar * 4 + k; sym = chord_at(b)
        r = (ROOT[sym] + TR) % 12
        p = 36 + (r - 36) % 12 if k == 0 else 36 + (r + 7 - 36) % 12
        if b < 4: p = 36 + (r - 36) % 12
        add(b, 0.9 * B, 'bass', p, 112, 'bass'); H['bass'].append((b, p))
# walk-ups into the trio and the finale
for b, p in ((TRIO0 - 2, 41 + 0), (TRIO0 - 1.5, 43), (TRIO0 - 1, 45), (FIN0 - 2, 45), (FIN0 - 1.5, 47), (FIN0 - 1, 48)):
    add(b, 0.4 * B, 'bass', p + 0, 104, 'bass'); H['bass'].append((b, p))

# ---------- percussion ----------
def drum(b, k, v, key=None, role='drum'):
    ev.append((round(b, 4), 0.25 * B, 'drum', k, v, role))
    if key: H[key].append(b)
for bar in range(FINAL // 4):
    for k in (0, 2):
        drum(bar * 4 + k, 36, 110 if k == 0 else 96, 'kick')
    for k in (1, 3):
        if bar * 4 + k < FINAL - 1: drum(bar * 4 + k, 38, 100 if k == 3 else 88, 'snare')
# field drum cadence (cam-driven sticks): 8ths, 16th fills at phrase ends
for bar in range(FINAL // 4):
    sub = 4 if (bar % 4 == 3) else 2
    for s in range(4 * sub):
        b = bar * 4 + s / sub
        acc = 92 if s % sub == 0 else 58
        ev.append((round(b, 4), 0.1 * B, 'drum', 37, acc, 'hat')); H['hat'].append(b)
# timpani (pots -> ventilators) : intro, into trio, into finale
def fill(b0, seq, step=0.5):
    for i, k in enumerate(seq): drum(b0 + i * step, k, 112); H['tom'].append((b0 + i * step, k))
fill(4, [45, 47, 50, 50], 0.5)
fill(TRIO0 - 4, [50, 47, 45, 45], 1.0)
fill(FIN0 - 4, [45, 47, 50, 50, 45, 45], 0.5)
fill(FINAL - 3, [50, 47, 45], 1.0)
# cymbal crashes (cannon salvos) at the big downbeats
for i, b in enumerate((0, A0, A0 + 32, TRIO0, TRIO0 + 32, FIN0, FINAL)):
    drum(b, 49 if i % 2 == 0 else 57, 124); H['crash'].append((b, i % 2))
# final chord
for p in (41, 53): add(FINAL, 3, 'bass', p, 124, 'final')
for p in (65, 69, 72, 77): add(FINAL, 3, 'chord', p, 118, 'final')
H['bass'].append((FINAL, 41)); H['chord'].append((FINAL, [65, 69, 72])); H['mel'].append((FINAL, 77))
add(FINAL, 3, 'mel', 77, 120, 'final')
drum(FINAL, 36, 127, 'kick')

json.dump([[sec(b), d, ins, p, v, r, b] for b, d, ins, p, v, r in sorted(ev)], open('notes.json', 'w'))
out = {}
for k, v in H.items():
    out[k] = [[sec(x[0])] + list(x[1:]) if isinstance(x, tuple) else sec(x) for x in sorted(v, key=lambda z: z[0] if isinstance(z, tuple) else z)]
out['final'] = sec(FINAL); out['beat'] = B; out['t0'] = sec(0)
out['sections'] = [(sec(0), 'A'), (sec(A0), 'A+'), (sec(TRIO0), 'B'), (sec(FIN0 - 4), 'roll'), (sec(FIN0), 'A2'), (sec(FINAL - 4), 'finale')]
json.dump(out, open('events.json', 'w'))
print({k: len(v) for k, v in H.items()}, 'final', out['final'])
print('mel', mel_p); print('harm', sorted({p for _, p in H['harm']})); print('chord', sorted({p for _, ps in H['chord'] for p in ps})); print('bass', sorted({p for _, p in H['bass']}))
