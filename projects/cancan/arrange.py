"""Can-can (Offenbach, Galop infernal 1858, PD): melody + harmony taken from the reference MIDI,
re-arranged for a ball-launcher machine: xylophone melody, bass marimba, chord bells, drums."""
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

MEL = notes(10); BASS = notes(9)
SEGS = [(0, 78), (142, 190), (266, 281)]          # source beats kept
def remap(b):
    off = 0
    for a, e in SEGS:
        if a <= b < e: return off + (b - a)
        off += e - a
    return None

mel = [(remap(b), d, p) for b, d, p in MEL if remap(b) is not None]
# drop the doubled octave in the last chord
mel = [m for m in mel if not (m[0] >= 137 and m[2] == 84)]
# harmony: bass root on each beat
root = {}
for b, d, p in BASS:
    r = remap(b)
    if r is not None and abs(r - round(r)) < 0.01: root.setdefault(int(round(r)), p)
TOTAL = 141
BPM = 140.0; B = 60 / BPM
T0 = 1.6                                           # time for the first balls to fly
MAJ = {0: [0, 4, 7], 5: [5, 9, 0], 7: [7, 11, 2, 5], 2: [2, 6, 9, 0], 10: [10, 2, 5], 9: [9, 1, 4, 7], 4: [4, 8, 11, 2]}

P = {'xylo': (0, 13), 'marimba': (1, 12), 'vibes': (2, 11), 'glock': (3, 9)}
ev = []
def add(b, d, ins, p, v): ev.append((b, d, ins, int(p), int(max(1, min(127, v)))))
def drum(b, k, v): ev.append((b, 0.25, 'drum', k, v))

hits = {'mel': [], 'bass': [], 'chord': [], 'kick': [], 'snare': [], 'crash': [], 'hat': []}
for b, d, p in mel:
    add(b, max(d, 0.3), 'xylo', p, 112); add(b, max(d, 0.3), 'glock', p + 12, 60)
    hits['mel'].append((b, p))
last_root = 48
for beat in range(TOTAL):
    r = root.get(beat, last_root); last_root = r
    rr = 36 + (r - 36) % 12
    final = beat >= 137
    if final: continue
    add(beat, 0.45, 'marimba', rr, 118); hits['bass'].append((beat, rr))
    pcs = MAJ.get(r % 12, [r % 12, (r + 4) % 12, (r + 7) % 12])
    ch = sorted(60 + (pc - 0) % 12 for pc in pcs[:3])
    for p in ch: add(beat + 0.5, 0.3, 'vibes', p, 90)
    hits['chord'].append((beat + 0.5, ch))
    drum(beat, 36, 105); hits['kick'].append(beat)
    drum(beat + 0.5, 38, 100); hits['snare'].append(beat + 0.5)
for b in (0, 32, 78, 94, 110, 126):
    drum(b, 49, 120); hits['crash'].append(b)
# build-up snare roll before the reprise (beats 74-78)
for k in range(16):
    drum(74 + k * 0.25, 38, 70 + k * 3); hits['snare'].append(74 + k * 0.25)
# finale: big chord + crash
for p in (36, 48, 60, 64, 67, 72): add(137, 3, 'marimba' if p < 60 else 'vibes', p, 120)
drum(137, 49, 127); drum(137, 57, 120); hits['crash'].append(137)
hits['bass'].append((137, 36))

mid = mido.MidiFile(ticks_per_beat=480); tr = mido.MidiTrack(); mid.tracks.append(tr)
tr.append(mido.MetaMessage('set_tempo', tempo=int(B * 1e6)))
msgs = []
for name, (ch, prog) in P.items():
    msgs.append((0, mido.Message('program_change', channel=ch, program=prog)))
    msgs.append((0, mido.Message('control_change', channel=ch, control=91, value=45)))
off = T0 / B
for b, d, ins, p, v in ev:
    ch = 9 if ins == 'drum' else P[ins][0]
    msgs.append((int((b + off) * 480), mido.Message('note_on', channel=ch, note=p, velocity=v)))
    msgs.append((int((b + off + d) * 480), mido.Message('note_off', channel=ch, note=p, velocity=0)))
msgs.sort(key=lambda m: (m[0], 0 if m[1].type == 'note_off' else 1))
last = 0
for t, m in msgs: tr.append(m.copy(time=t - last)); last = t
mid.save('cancan.mid')
sec = lambda b: (b + off) * B
out = {k: [((sec(x[0]),) + tuple(x[1:])) if isinstance(x, tuple) else sec(x) for x in v] for k, v in hits.items()}
out['final'] = sec(137); out['beat'] = B; out['sections'] = [(sec(0), 'A'), (sec(32), 'B'), (sec(78), 'A2'), (sec(126), 'finale')]
json.dump(out, open('events.json', 'w'))
print('melody', len(hits['mel']), 'final', round(out['final'], 2), 'pitches', sorted({p for _, p in hits['mel']}), 'bass', sorted({p for _, p in hits['bass']}))
