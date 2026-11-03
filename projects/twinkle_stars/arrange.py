"""Mozart - 12 Variations on "Ah vous dirai-je, Maman" K.265 (Twinkle Twinkle), complete (theme + 12 variations,
no repeats). Score: Mutopia Project, arr. for two guitars, ed. J. J. Olson - Public Domain. Reordered to Mozart's order.
Balls: 0 = upper voice (top notes), 1 = lower voice (bass notes), 2 = inner notes.
notes: [t, dur, ball, pitch, vel]"""
import json, mido
m = mido.MidiFile('src_k265.midi'); tpb = m.ticks_per_beat
tmap = []
for tr in m.tracks:
    t = 0
    for msg in tr:
        t += msg.time
        if msg.type == 'set_tempo': tmap.append((t, msg.tempo))
tmap.sort()
def tsec(tick):
    s = 0.0; last_t, last_tempo = 0, 500000
    for tt, tp in tmap:
        if tt >= tick: break
        s += (tt - last_t) / tpb * last_tempo / 1e6; last_t, last_tempo = tt, tp
    return s + (tick - last_t) / tpb * last_tempo / 1e6
tracks = []
for tr in m.tracks:
    t = 0; on = {}; ns = []
    for msg in tr:
        t += msg.time
        if msg.type == 'note_on' and msg.velocity > 0: on[msg.note] = t
        elif msg.type in ('note_off', 'note_on') and msg.note in on:
            s = on.pop(msg.note); ns.append((s, t, msg.note))
    if ns: tracks.append(ns)
assert len(tracks) == 26, len(tracks)
T0 = 1.0; UP = 12                      # guitar sounds an octave low: lift into the mallet range
sections = []; ev = []
names = ['Theme'] + [f'Var. {r}' for r in 'I II III IV V VI VII VIII IX X XI XII'.split()]
for k in range(13):
    hi, lo = tracks[2 * k], tracks[2 * k + 1]
    allns = [(a, b, p, 0) for a, b, p in hi] + [(a, b, p, 1) for a, b, p in lo]
    st = min(a for a, _, _, _ in allns); en = max(b for _, b, _, _ in allns)
    sections.append([round(T0 + tsec(st), 3), round(T0 + tsec(en), 3), names[k]])
    by = {}
    for a, b, p, g in allns: by.setdefault(a, []).append((b, p, g))
    for a in sorted(by):
        ch = by[a]; t = round(T0 + tsec(a), 4)
        top = max(ch, key=lambda x: x[1]); bot = min(ch, key=lambda x: x[1])
        v = 98
        ev.append([t, round(tsec(top[0]) - tsec(a), 3), 0, top[1] + UP, v])
        if bot is not top: ev.append([t, round(tsec(bot[0]) - tsec(a), 3), 1, bot[1] + UP, v - 6])
        mids = [x for x in ch if x is not top and x is not bot]
        if mids:
            md = max(mids, key=lambda x: x[1]); ev.append([t, round(tsec(md[0]) - tsec(a), 3), 2, md[1] + UP, v - 12])
# a ball needs some time between hits: drop inner notes closer than 0.09 s to the previous inner note
ev.sort(key=lambda e: (e[2], e[0])); out = []; last = {}
for e in ev:
    if e[0] - last.get(e[2], -9) < 0.085: continue
    last[e[2]] = e[0]; out.append(e)
out.sort()
END = sections[-1][1]
json.dump(out, open('notes.json', 'w'))
json.dump(dict(t0=T0, final=END, sections=sections, beat=0.5), open('events.json', 'w'))
from collections import Counter
print(Counter(e[2] for e in out), 'end', END)
for s in sections: print(s)
