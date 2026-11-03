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

INTRO = 16                                 # 4 bars: cannon salvos + drum roll, then the bugle call
A0 = INTRO                                 # A strain (16 bars)
A1 = A0 + 64                               # A strain again, full band ('tutti')
TAG = A1 + 64                              # 1-bar tag: three salvos
FINAL = TAG + 4                            # final chord
def chord_at(b):
    if b < A0: return 'd' if b < 12 else ('a:7' if b >= 14 else 'd')
    if b >= TAG: return 'd'
    rel = (b - A0) % 64
    return CH[int(rel // 2)]

# ---------- melody ----------
mel = []
for rep, base in enumerate((A0, A1)):
    for b, d, p in MEL0:
        if b < 64: mel.append((base + b, d, p + TR, 'A' if rep == 0 else 'T'))
for b, d, p in ((0, 0.75, 57), (0.75, 0.25, 57), (1, 1, 62), (2, 0.75, 57), (2.75, 0.25, 62), (3, 1, 66),
                (4, 1, 69), (5, 0.75, 66), (5.75, 0.25, 62), (6, 1, 64), (7, 0.5, 57), (7.5, 0.5, 57)):
    mel.append((8 + b, d, p + TR, 'I'))
mel.sort()
mel = [m for m in mel if not (A1 + 61 <= m[0] < TAG)]          # drop the pick-up at the very end
mel.append((A1 + 60, 2.0, 62 + TR, 'T'))
for b, d, p, sect in mel:
    role = 'mel_tutti' if sect == 'T' else 'mel'
    add(b, d * B, 'mel', p, 118 if sect == 'T' else 110, role); H['mel'].append((b, p))
mel_p = sorted({p for _, _, p, _ in mel})

# ---------- glissando (bell lyre run) into the tutti ----------
def gliss(b0, b1, lo, hi):
    ps = [p for p in mel_p if lo <= p <= hi]
    for i, p in enumerate(ps):
        b = b0 + (b1 - b0) * i / (len(ps) - 1)
        add(b, 0.2, 'gliss', p, 80 + 40 * i / len(ps), 'gliss'); H['gliss'].append((b, p))
gliss(A1 - 1.0, A1 - 0.05, 60, 80)

# ---------- chords: horn 'pah' on beats 2 and 4, stabs in intro and tag ----------
def voicing(sym):
    pcs = [(pc + TR) % 12 for pc in PCS[sym]]
    if sym == 'a:7': pcs = [pcs[0], pcs[1], pcs[2]]
    return sorted(57 + (pc - 57) % 12 for pc in pcs)
for bar in range(A0 // 4, TAG // 4):
    for k in (1, 3):
        b = bar * 4 + k; ps = voicing(chord_at(b))
        for p in ps: add(b, 0.5 * B, 'chord', p, 96 if b >= A1 else 88, 'chord')
        H['chord'].append((b, ps))
for b in (0, 4, 12, 13.5, 14, TAG, TAG + 1, TAG + 2):
    ps = voicing(chord_at(b)); [add(b, 0.6 * B, 'chord', p, 120, 'chord') for p in ps]; H['chord'].append((b, ps))

# ---------- harmony: 3rd/6th below the melody (from bar 2 of the A strain) ----------
for b, d, p, sect in mel:
    if sect not in ('A', 'T') or b < A0 + 8: continue
    pcs = {(pc + TR) % 12 for pc in PCS[chord_at(b)]}
    cand = [q for q in range(p - 9, p - 2) if q % 12 in pcs]
    if not cand: continue
    hp = max(cand)
    add(b, d * B, 'harm', hp, 100 if sect == 'T' else 90, 'harm'); H['harm'].append((b, hp))

# ---------- bass: tuba root / fifth on beats 1 and 3, walking 8ths at phrase ends ----------
for bar in range(TAG // 4):
    if bar < 2: continue
    for k in (0, 2):
        b = bar * 4 + k; sym = chord_at(b)
        r = (ROOT[sym] + TR) % 12
        p = 36 + (r - 36) % 12 if (k == 0 or bar < 4) else 36 + (r + 7 - 36) % 12
        add(b, 0.9 * B, 'bass', p, 116, 'bass'); H['bass'].append((b, p))
for base in (A0 + 28, A0 + 60, A1 + 28):
    for i, p in enumerate((41, 43, 45, 46)):
        add(base + 2 + i * 0.5, 0.45 * B, 'bass', p, 108, 'bass'); H['bass'].append((base + 2 + i * 0.5, p))

# ---------- percussion ----------
def drum(b, k, v, key=None, role='drum'):
    ev.append((round(b, 4), 0.25 * B, 'drum', k, v, role))
    if key: H[key].append(b)
for bar in range(2, TAG // 4):
    for k in (0, 2): drum(bar * 4 + k, 36, 116 if k == 0 else 104, 'kick', 'kick_cym' if k == 0 else 'drum')
    for k in (1, 3): drum(bar * 4 + k, 38, 110 if k == 3 else 98, 'snare')
for b in (0, 4): drum(b, 36, 124, 'kick')
# field drum cadence (cam sticks): roll through the intro, 8ths + 16th fills
for bar in range(TAG // 4 + 1):
    sub = 4 if (bar < 2 or bar % 4 == 3 or bar * 4 >= TAG) else 2
    for s_ in range(4 * sub):
        b = bar * 4 + s_ / sub
        if b >= FINAL: break
        acc = 100 if s_ % sub == 0 else (66 + (s_ * 3 if bar < 2 else 0))
        ev.append((round(b, 4), 0.1 * B, 'drum', 37, min(120, acc), 'hat')); H['hat'].append(b)
# timpani at the phrase ends and into the tutti / tag
def fill(b0, seq, step=0.5):
    for i, k in enumerate(seq): drum(b0 + i * step, k, 116); H['tom'].append((b0 + i * step, k))
fill(14, [45, 45, 47, 50])
for base in (A0 + 30, A0 + 62, A1 + 30, A1 + 62): fill(base, [50, 47, 45, 45])
fill(FINAL - 1, [45, 45], 0.5)
# cymbal crash balls (searchlights) at the big downbeats
for i, b in enumerate((A0, A0 + 32, A1, A1 + 32, TAG, FINAL)):
    drum(b, 49 if i % 2 == 0 else 57, 124); H['crash'].append((b, i % 2))
# main battery salvos (big turrets behind the mast fire out to sea; the boom is heard on firing)
SALVO = [0, 4, 8, A0, A0 + 16, A0 + 32, A0 + 48, A1, A1 + 16, A1 + 32, A1 + 48, TAG, TAG + 1, TAG + 2, FINAL]
for i, b in enumerate(SALVO): ev.append((b, 0.5, 'drum', 35, 127, 'salvo'))
# final chord
for p in (41, 53): add(FINAL, 3, 'bass', p, 124, 'final')
for p in (65, 69, 72, 77): add(FINAL, 3, 'chord', p, 120, 'final')
H['bass'].append((FINAL, 41)); H['chord'].append((FINAL, [65, 69, 72])); H['mel'].append((FINAL, 77))
add(FINAL, 3, 'mel', 77, 124, 'final')
drum(FINAL, 36, 127, 'kick')

json.dump([[sec(b), d, ins, p, v, r, b] for b, d, ins, p, v, r in sorted(ev)], open('notes.json', 'w'))
out = {}
for k, v in H.items():
    out[k] = [[sec(x[0])] + list(x[1:]) if isinstance(x, tuple) else sec(x) for x in sorted(v, key=lambda z: z[0] if isinstance(z, tuple) else z)]
out['final'] = sec(FINAL); out['beat'] = B; out['t0'] = sec(0)
out['sections'] = [(sec(0), 'A'), (sec(A0), 'A+'), (sec(A0 + 20), 'B'), (sec(A1 - 8), 'roll'), (sec(A1), 'A2'), (sec(TAG - 4), 'finale')]
out['salvo'] = [sec(b) for b in SALVO]
json.dump(out, open('events.json', 'w'))
print({k: len(v) for k, v in H.items()}, 'final', out['final'])
print('mel', mel_p); print('harm', sorted({p for _, p in H['harm']})); print('chord', sorted({p for _, ps in H['chord'] for p in ps})); print('bass', sorted({p for _, p in H['bass']}))
