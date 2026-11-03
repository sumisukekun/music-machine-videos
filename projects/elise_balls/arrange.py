"""Beethoven - Fur Elise WoO 59 (public domain; score: Mutopia Project). Excerpt beats 0..60 (repeats unfolded) at q=58.
Balls join one by one: RH melody, LH arpeggio, glock octave, bass roots, xylo arpeggio, violin long notes.
notes: [t, dur, ball, pitch, vel]"""
import json, mido
m = mido.MidiFile('src_elise.midi'); tpb = m.ticks_per_beat
tr = []
for t_ in m.tracks:
    t = 0; on = {}; ns = []
    for msg in t_:
        t += msg.time
        if msg.type == 'note_on' and msg.velocity > 0: on[msg.note] = t
        elif msg.type in ('note_off', 'note_on') and msg.note in on:
            s = on.pop(msg.note); ns.append((s / tpb, (t - s) / tpb, msg.note))
    if ns: tr.append(sorted(ns))
RH, LH = tr
BPM = 58; B = 60 / BPM; T0 = 0.6; S1 = 59.75
sec = lambda b: round(T0 + b * B, 4)
def mono(src, pick=max, grid=0.06):
    out = {}
    for st, d, p in src:
        if st >= S1: continue
        k = round(st / grid)
        if k not in out or pick(out[k][2], p) == p: out[k] = (st, d, p)
    return [out[k] for k in sorted(out)]
rh = mono(RH, max); lh = mono(LH, min)
roots = []
for st, d, p in lh:                               # bar downbeats (3/8 = 1.5 beats)
    if abs(((st - 0.5) / 1.5) - round((st - 0.5) / 1.5)) < 1e-3: roots.append((st, 1.0, p - 12))
long_rh = [(st, d, p) for st, d, p in rh if abs(st * 2 - round(st * 2)) < 1e-3]   # melody notes on the 8th grid
ev = []
def put(ball, notes, e0, tr_=0, vel=96):
    for st, d, p in notes:
        if st < e0: continue
        v = vel + (8 if abs(((st - 0.5) / 1.5) - round((st - 0.5) / 1.5)) < 1e-3 else 0)
        ev.append([sec(st), round(d * B, 3), ball, p + tr_, v])
ENTRY = [0, 2, 12, 24, 39, 45]
put(0, rh, ENTRY[0]); put(1, lh, ENTRY[1], vel=84); put(2, rh, ENTRY[2], 12, 86)
put(3, roots, ENTRY[3], 0, 100); put(4, lh, ENTRY[4], 24, 80); put(5, long_rh, ENTRY[5], 0, 92)
FIN = sec(60.0)
for bi, p in {0: 76, 1: 45, 2: 88, 3: 33, 4: 81, 5: 69}.items(): ev.append([FIN, 3.0, bi, p, 116])
for p in (57, 64, 72): ev.append([FIN, 3.0, 1, p, 104])          # sound-only chord tones on the LH ball
ev.sort()
json.dump(ev, open('notes.json', 'w'))
json.dump(dict(beat=B, t0=T0, final=FIN, entries=[sec(e) for e in ENTRY]), open('events.json', 'w'))
from collections import Counter
print(Counter(e[2] for e in ev), 'final', FIN, [sec(e) for e in ENTRY])
