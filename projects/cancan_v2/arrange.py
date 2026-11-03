"""Can-can v2 (Offenbach, Galop infernal 1858, PD) - detailed arrangement for the precision machine.
Notes taken from the reference MIDI (melody, real piano voicings, walking bass), plus
computed 3rd/6th harmony, octave bass in the reprise, full drum kit, tom fills and glissandi.
Output: cancan.mid (audio) and events.json (every ball / hit for the scene)."""
import json
import mido

SRC = mido.MidiFile('src/cancan_offenbach.mid'); TPB = SRC.ticks_per_beat


def notes(ti):
    t = 0; on = {}; out = []
    for x in SRC.tracks[ti]:
        t += x.time
        if x.type == 'note_on' and x.velocity > 0: on[x.note] = t
        elif x.type in ('note_off', 'note_on') and x.note in on:
            s = on.pop(x.note); out.append((s / TPB, (t - s) / TPB, x.note))
    return sorted(out)


SEGS = [(0, 78), (142, 190), (266, 281)]
def remap(b):
    off = 0
    for a, e in SEGS:
        if a <= b + 1e-6 < e: return round(off + (b - a), 3)
        off += e - a
    return None

def keep(tr): return [(remap(b), d, p) for b, d, p in notes(tr) if remap(b) is not None]

MEL = keep(10); PIANO = keep(8); BASS = keep(9)
FINAL = 140
BPM = 140.0; B = 60 / BPM; T0 = 1.6
off = T0 / B
sec = lambda b: round((b + off) * B, 4)

ev = []          # audio events (beat, dur, inst, pitch, vel)
def add(b, d, ins, p, v): ev.append((b, d, ins, int(p), int(max(1, min(127, v)))))
H = {k: [] for k in ('mel', 'harm', 'chord', 'bass', 'kick', 'snare', 'tom', 'crash', 'hat', 'gliss')}

# ---- melody (xylophone + soft glockenspiel octave) ----
mel = [m for m in MEL if m[0] <= FINAL and not (m[0] >= 139 and m[2] < 60)]
# remove melody under the two glissandi
mel = [m for m in mel if not (77.0 <= m[0] < 78.0) and not (138.5 <= m[0] < 139.0)]
for b, d, p in mel:
    add(b, max(d, 0.3), 'xylo', p, 112); add(b, max(d, 0.3), 'glock', p + 12, 52)
    H['mel'].append((b, p))
mel_p = sorted({p for _, _, p in mel})

# ---- glissandi: balls sweep up every xylophone bar ----
def gliss(b0, b1, lo, hi):
    ps = [p for p in mel_p if lo <= p <= hi]
    for i, p in enumerate(ps):
        b = b0 + (b1 - b0) * i / (len(ps) - 1)
        add(b, 0.25, 'xylo', p, 80 + 40 * i / len(ps)); H['gliss'].append((b, p))
gliss(77.0, 77.9, 60, 84)
gliss(138.5, 139.0, 72, 84)

# ---- chords: real piano voicings (offbeats) -> vibraphone, 3 balls at once ----
chords = {}
for b, d, p in PIANO:
    if b < FINAL: chords.setdefault(b, set()).add(p)
chord_list = []
for b in sorted(chords):
    ps = sorted(chords[b])[-3:]
    for p in ps: add(b, 0.32, 'vibes', p, 84)
    H['chord'].append((b, ps)); chord_list.append((b, ps))

def chord_at(b):
    best = None
    for cb, ps in chord_list:
        if cb <= b + 0.5: best = ps
        else: break
    return best

# ---- harmony: chord tone a 3rd/6th below the melody (celesta) from beat 16 ----
for b, p in H['mel']:
    if b < 16 or b >= FINAL: continue
    ch = chord_at(b)
    if not ch: continue
    pcs = {q % 12 for q in ch}
    cand = [q for q in range(p - 9, p - 2) if q % 12 in pcs]
    if not cand: continue
    hp = max(cand)
    while hp < 60: hp += 12
    if hp >= p: continue
    add(b, 0.35, 'celesta', hp, 96); H['harm'].append((b, hp))

# ---- bass: walking bass from the score, octave pops in the reprise ----
for b, d, p in BASS:
    if b > FINAL: continue
    add(b, 0.45, 'marimba', p, 118); H['bass'].append((b, p))
    if 78 <= b < 126 and abs(b - round(b)) < 0.01:
        add(b + 0.5, 0.25, 'marimba', p + 12, 96); H['bass'].append((b + 0.5, p + 12))

# ---- drums ----
def drum(b, k, v, key=None):
    ev.append((b, 0.25, 'drum', k, v))
    if key: H[key].append(b)
for beat in range(FINAL):
    if not (74 <= beat < 78):
        drum(beat, 36, 108, 'kick')
        drum(beat + 0.5, 38, 96, 'snare')
    # hi-hat (cam driven, no balls): 8ths, 16ths in the reprise / finale
    sub = 4 if beat >= 78 else 2
    for s in range(sub):
        bb = beat + s / sub
        acc = 92 if s == 0 else (78 if s == sub // 2 else 58)
        ev.append((bb, 0.1, 'drum', 46 if (sub == 2 and s == 1 and 32 <= beat < 74) else 42, acc)); H['hat'].append(bb)
# snare roll crescendo into the reprise (repeated C's)
for k in range(16):
    drum(74 + k * 0.25, 38, 64 + k * 3, 'snare')
for k in range(4):
    drum(74 + k, 36, 112, 'kick')
# tom fills (hi/mid/low toms 50/47/45)
def fill(b0, seq):
    for i, k in enumerate(seq): drum(b0 + i * 0.5, k, 112); H['tom'].append((b0 + i * 0.5, k))
fill(30, [50, 50, 47, 47])
fill(124, [50, 47, 45, 45])
fill(136, [50, 50, 47, 47, 45, 45])
# crashes at section starts (alternating L/R cymbal)
for i, b in enumerate((0, 16, 32, 54, 78, 94, 110, 126, FINAL)):
    drum(b, 49 if i % 2 == 0 else 57, 122); H['crash'].append((b, i % 2))
# finale chord
for p in (36, 48):
    add(FINAL, 3, 'marimba', p, 124)
for p in (60, 64, 67, 72):
    add(FINAL, 3, 'vibes', p, 110)
H['bass'].append((FINAL, 48)) if (FINAL, 48) not in H['bass'] else None
H['chord'].append((FINAL, [64, 67, 72])) if all(c[0] != FINAL for c in H['chord']) else None
drum(FINAL, 36, 127, 'kick')

# ---- midi ----
P = {'xylo': (0, 13), 'marimba': (1, 12), 'vibes': (2, 11), 'glock': (3, 9), 'celesta': (4, 8)}
mid = mido.MidiFile(ticks_per_beat=480); tr = mido.MidiTrack(); mid.tracks.append(tr)
tr.append(mido.MetaMessage('set_tempo', tempo=int(B * 1e6)))
msgs = []
for name, (ch, prog) in P.items():
    msgs.append((0, mido.Message('program_change', channel=ch, program=prog)))
    msgs.append((0, mido.Message('control_change', channel=ch, control=91, value=40)))
    msgs.append((0, mido.Message('control_change', channel=ch, control=10, value={'xylo': 64, 'marimba': 44, 'vibes': 80, 'glock': 64, 'celesta': 70}[name])))
for b, d, ins, p, v in ev:
    ch = 9 if ins == 'drum' else P[ins][0]
    msgs.append((int(round((b + off) * 480)), mido.Message('note_on', channel=ch, note=p, velocity=v)))
    msgs.append((int(round((b + off + d) * 480)), mido.Message('note_off', channel=ch, note=p, velocity=0)))
msgs.sort(key=lambda m: (m[0], 0 if m[1].type == 'note_off' else 1))
last = 0
for t, m in msgs: tr.append(m.copy(time=t - last)); last = t
mid.save('cancan.mid')

out = {}
for k, v in H.items():
    out[k] = [[sec(x[0])] + list(x[1:]) if isinstance(x, tuple) else sec(x) for x in sorted(v, key=lambda z: z[0] if isinstance(z, tuple) else z)]
out['final'] = sec(FINAL); out['beat'] = B; out['t0'] = sec(0)
out['sections'] = [(sec(0), 'A'), (sec(16), 'A+'), (sec(32), 'B'), (sec(74), 'roll'), (sec(78), 'A2'), (sec(126), 'finale')]
json.dump(out, open('events.json', 'w'))
print({k: len(v) for k, v in H.items()}, 'final', out['final'])
print('mel', mel_p); print('harm', sorted({p for _, p in H['harm']})); print('chord', sorted({p for _, ps in H['chord'] for p in ps})); print('bass', sorted({p for _, p in H['bass']}))
