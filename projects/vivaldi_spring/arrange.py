"""Vivaldi - Le quattro stagioni, 'La Primavera' (Spring), RV 269, 1st movement (Allegro). Antonio Vivaldi, publ. 1725 - public domain.
Score source: Mutopia Project, VivaldiA/O8/spring (source: Performers' Facsimiles reprint of the 1725 edition, typeset by
John Williams, Mutopia-2010/02/08-301, CC BY-SA 3.0). The part files are copied unchanged into score/ (spring1*.ly; only
convert-ly syntax updates) and score/spring_midi.ly renders them to score/spring_midi.midi with LilyPond (quarter = 60, so
one MIDI second = one beat). Only Vivaldi's notes (pitches / rhythm) are taken; the reduction, the choice of voices, the
instrumentation and the ending are made here. Ornaments (trills, mordents) are not played - every sound is one visible hit.

Excerpt: pickup + bars 1-30 + the downbeat of bar 31 (E major), in the original key E major, 4/4, quarter = 107.
  A  (pickup - bar 13 beat 3)   tutti ritornello:  melody = solo violin (= violin 1), harmony = violin 2, bass = cello
  B  (bar 13 beat 3 - bar 27)   'Canto de gl'Uccelli' (birdsong): bird A = solo violin, bird B = violin 1 (its echo, 2 bars
                                 later); the other parts rest in the score
  A' (bar 27 last 8th - bar 31) tutti ritornello again; final chord bar 31 beat 1 = solo E5, viola G#4 (taken by the
                                 harmony ball, violin 2 only doubles the E5 there), cello E3
Outputs lead.json (every struck event: role, t, pitch, dur, vel, bar, beat), notes.json (every sound), events.json."""
import json, os
import mido
HERE = os.path.dirname(os.path.abspath(__file__))
Q = 0.56                      # quarter note (107 bpm)
T0 = 1.7                      # time of the pickup (beat 0 of the MIDI)
FINAL_BEAT = 120.5            # bar 31 beat 1
B_START, B_END = 50.5, 108.0  # bird section (beats)
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'
def tb(b): return round(T0 + Q * b, 5)
def bar_of(b): return int((b - 0.5) // 4) + 1, round((b - 0.5) % 4 + 1, 3)

mf = mido.MidiFile(f'{HERE}/score/spring_midi.midi'); tpb = mf.ticks_per_beat
PART = {}
for ti, name in zip(range(1, 6), ('solo', 'vn1', 'vn2', 'vla', 'vc')):
    t = 0; on = {}; ns = []
    for m in mf.tracks[ti]:
        t += m.time
        if m.type == 'note_on' and m.velocity > 0: on[m.note] = t
        elif m.type in ('note_off', 'note_on') and m.note in on:
            s = on.pop(m.note); ns.append((round(s / tpb, 4), round((t - s) / tpb, 4), m.note))
    PART[name] = sorted(ns)


def dyn(b):
    """dynamics from the score: f bars 1-3, p 4-6 (echo), f 7-9, p 10-13, solo birdsong mf, tutti f"""
    bar, _ = bar_of(b)
    if b >= B_END - 0.01: return 100
    if b >= B_START - 0.01: return 88
    if bar <= 3 or 7 <= bar <= 9: return 98
    return 72


def pick(part, lo, hi):
    return [n for n in PART[part] if lo - 1e-6 <= n[0] < hi - 1e-6]


EV = []
def add(role, ns):
    for b, d, p in ns:
        bar, beat = bar_of(b)
        EV.append(dict(role=role, beat=b, t=tb(b), pitch=p, dur=round(d * Q, 4), vel=dyn(b), bar=bar, beatno=beat))

# melody: solo violin (tutti = violin 1); the cadence E5 of bar 13 beat 3 comes from violin 1 (the solo starts the birdsong there)
add('mel', pick('solo', 0, B_START) + [n for n in PART['vn1'] if abs(n[0] - B_START) < 1e-6] + pick('solo', B_END, FINAL_BEAT + 0.01))
add('har', pick('vn2', 0, B_START + 0.01) + pick('vn2', B_END, FINAL_BEAT))
add('har', [n for n in PART['vla'] if abs(n[0] - FINAL_BEAT) < 1e-6])
add('bas', pick('vc', 0, B_START + 0.01) + pick('vc', B_END, FINAL_BEAT + 0.01))
add('birdA', pick('solo', B_START, B_END))
add('birdB', pick('vn1', B_START + 0.01, B_END))
EV.sort(key=lambda e: (e['t'], e['role']))
for i, e in enumerate(EV): e['i'] = i
FINAL = tb(FINAL_BEAT)

# sounds (render_audio.py turns roles into instruments; every sound starts at a visible hit)
notes = []
for e in EV:
    fin = abs(e['beat'] - FINAL_BEAT) < 1e-6
    d = 3.2 if fin else max(0.12, e['dur'])
    notes.append([e['t'], round(d, 4), e['role'] + ('_fin' if fin else ''), e['pitch'], e['vel'] + (18 if fin else 0)])
notes.sort(key=lambda n: n[0])
json.dump(EV, open(f'{HERE}/lead.json', 'w'), indent=0)
json.dump(notes, open(f'{HERE}/notes.json', 'w'))
json.dump(dict(t0=T0, q=Q, bar=4 * Q, final=FINAL, b_start=tb(B_START), b_end=tb(B_END), final_beat=FINAL_BEAT,
               bar1=tb(0.5)), open(f'{HERE}/events.json', 'w'))
if __name__ == '__main__':
    from collections import Counter
    print('events', len(EV), Counter(e['role'] for e in EV), 'final', FINAL, 'bird', tb(B_START), tb(B_END))
    for r in ('mel', 'har', 'bas', 'birdA', 'birdB'):
        ts = sorted(set(e['t'] for e in EV if e['role'] == r))
        dmin = min(b - a for a, b in zip(ts, ts[1:]))
        print(r, len(ts), 'min gap', round(dmin, 3), 'first', ts[0], 'last', ts[-1], 'dup times', len([e for e in EV if e['role'] == r]) - len(ts))
