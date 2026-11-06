"""Flight of the Bumblebee (N. Rimsky-Korsakov, 1900 - public domain) - own arrangement.
Melody notes read from a public-domain MusicXML transcription (src/fb.mxl, MuseTrainer PD library);
orchestration / dynamics / ending are ours. 2/4, q = 176.
lead = the buzzing semiquaver line (what the bee-ball bounces over), one per hex cell."""
import json, music21 as m21
Q = 176; BEAT = 60 / Q; T0 = 1.3
s = m21.converter.parse('src/fb.mxl')
def evs(part):
    out = []
    for n in part.flatten().notes:
        ps = [p.midi for p in n.pitches]
        out.append((float(n.offset), float(n.quarterLength), ps))
    return out
RH, LH = evs(s.parts[0]), evs(s.parts[1])
NBEATS = 202
def t_(q): return T0 + q * BEAT
def in_beat(ev, b): return [e for e in ev if b <= e[0] < b + 1 - 1e-6]
lead, notes, counter, chords = [], [], [], []
def dyn(b):            # beat -> base velocity (our dynamics plan)
    bar = b // 2 + 1
    if bar <= 6: return 70 + bar * 4
    if bar <= 22: return 78 + (bar - 7)
    if bar <= 42: return 85
    if bar <= 58: return 80
    if bar <= 73: return 88 + (bar - 59) * 0.8
    return 104
for b in range(NBEATS):
    r16 = [e for e in in_beat(RH, b) if abs(e[1] - 0.25) < 1e-6 and len(e[2]) == 1]
    l16 = [e for e in in_beat(LH, b) if abs(e[1] - 0.25) < 1e-6 and len(e[2]) == 1]
    if r16: src, other = r16, 'L'
    elif l16: src, other = l16, 'R'
    else:
        src, other = [e for e in in_beat(RH, b) if len(e[2]) == 1], 'L'
        if not src: src = [e for e in in_beat(RH, b)]          # dyad/chord: the ball strikes its top note
    used = set()
    for e in src:
        v = dyn(b) + (10 if abs(e[0] - b) < 1e-6 else 0)
        top = max(e[2])
        lead.append([t_(e[0]), top, int(min(124, v)), b, round((e[0] - b) * 4)])
        for p in e[2]:
            if p != top: notes.append([t_(e[0]), max(0.12, e[1] * BEAT), 'lead2', p, int(min(124, v)) - 10])
        notes.append([t_(e[0]), max(0.12, e[1] * BEAT), 'lead', top, int(min(124, v))]); used.add(id(e))
    # everything else in this beat = accompaniment / counter-melody
    for e in in_beat(RH, b) + in_beat(LH, b):
        if e in src: continue
        if abs(e[1] - 0.25) < 1e-6 and len(e[2]) == 1:      # second 16th voice (rare) -> soft pizz
            notes.append([t_(e[0]), 0.1, 'pizz', e[2][0], 60]); continue
        d = e[1] * BEAT; v = int(dyn(b) * 0.9)
        if len(e[2]) == 1 and e[1] >= 0.5 and e in in_beat(RH, b) and other == 'R':
            counter.append([t_(e[0]), d, e[2][0], v]); notes.append([t_(e[0]), d, 'horn', e[2][0], v])
        else:
            for p in e[2]: notes.append([t_(e[0]), min(d, 0.22), 'chord', p, v])
            notes.append([t_(e[0]), 0.3, 'bass', min(e[2]) - (12 if min(e[2]) >= 52 else 0), v])
            chords.append([t_(e[0]), min(e[2]), max(e[2]), v])
lead.sort(); notes.sort(key=lambda n: n[0]); chords.sort()
FINAL = t_(200)       # bar 101 low A
for t, p in ((FINAL, 33), (FINAL, 45)): notes.append([t, 2.5, 'bass', p, 115])
notes.append([FINAL, 1.2, 'timp', 45, 120]); notes.append([FINAL, 2, 'crash', 0, 105])
notes.append([t_(198), 1.0, 'timp', 45, 110])
notes.sort(key=lambda n: n[0])
json.dump(notes, open('notes.json', 'w')); json.dump(lead, open('lead.json', 'w'))
json.dump({'t0': T0, 'beat': BEAT, 'final': FINAL, 'chords': chords, 'counter': counter, 'nbeats': NBEATS}, open('events.json', 'w'))
print('notes', len(notes), 'lead', len(lead), 'counter', len(counter), 'chords', len(chords), 'final', round(FINAL, 2),
      'range', min(l[1] for l in lead), max(l[1] for l in lead))
