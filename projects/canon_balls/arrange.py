"""Pachelbel - Canon in D (public domain; score: Mutopia Project, 'Canon per 3 Violini e Basso').
Excerpt beats 8..96 at q=84; voices enter one by one = balls join one by one. Writes notes.json + events.json.
notes: [t, dur, ball, pitch, vel]"""
import json, mido
m = mido.MidiFile('src_canon.midi'); tpb = m.ticks_per_beat
tracks = []
for tr in m.tracks:
    t = 0; on = {}; ns = []
    for msg in tr:
        t += msg.time
        if msg.type == 'note_on' and msg.velocity > 0: on[msg.note] = t
        elif msg.type in ('note_off', 'note_on') and msg.note in on:
            s = on.pop(msg.note); ns.append((s / tpb, (t - s) / tpb, msg.note))
    if ns: tracks.append(sorted(ns))
V1, V2, V3, BASS = tracks
BPM = 84; B = 60 / BPM; T0 = 0.5; S0 = 8; S1 = 96
sec = lambda beat: round(T0 + (beat - S0) * B, 4)
# ball: (source, entry beat, transpose, name)
BALLS = [(V1, 8, 0, 'v1'), (BASS, 16, 0, 'bass'), (V2, 24, 0, 'v2'), (V3, 40, 0, 'v3'), (V1, 56, 12, 'glock'), (V2, 72, 12, 'xylo')]
ev = []
for bi, (src, e0, tr, name) in enumerate(BALLS):
    last = -1
    for st, d, p in src:
        if st < e0 or st >= S1: continue
        if abs(st - last) < 1e-6: continue          # one ball = one note at a time (keep the first)
        last = st
        v = 92 + (10 if abs(st % 1) < 1e-6 else 0) + (6 if abs(st % 4) < 1e-6 else 0)
        ev.append([sec(st), round(d * B, 3), bi, p + tr, v])
FIN = sec(S1)
final = {0: 74, 1: 38, 2: 78, 3: 81, 4: 86, 5: 90}
for bi, p in final.items(): ev.append([FIN, 3.0, bi, p, 118])
ev.append([FIN, 3.0, 1, 50, 112])  # (sound only: octave in the bass)
ev.sort()
json.dump(ev, open('notes.json', 'w'))
entries = [sec(max(e0, S0)) for _, e0, _, _ in BALLS]
json.dump(dict(beat=B, t0=T0, final=FIN, entries=entries, names=[b[3] for b in BALLS]), open('events.json', 'w'))
from collections import Counter
print(Counter(e[2] for e in ev), 'final', FIN, 'entries', entries)
for bi in range(6):
    ts = [e[0] for e in ev if e[2] == bi]; dts = [b - a for a, b in zip(ts, ts[1:]) if b > a]
    print(bi, 'min dt', round(min(dts), 3), 'pitch', min(e[3] for e in ev if e[2] == bi), max(e[3] for e in ev if e[2] == bi))
