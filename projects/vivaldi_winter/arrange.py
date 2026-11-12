"""Vivaldi - Le quattro stagioni, 'L'Inverno' (Winter), Op. 8 No. 4, RV 297, 2nd movement (Largo). Antonio Vivaldi, publ. 1725 -
public domain. Score source: Mutopia Project, VivaldiA/O8/winter (source: Performers' Facsimiles reprint of the 1725 Le Cene
edition, typeset by John Williams, Mutopia-2010/02/08-351, CC BY-SA 3.0). The part files winter2*.ly are copied unchanged
into score/ (only convert-ly syntax updates); score/winter_midi.ly renders them to score/winter_midi.midi with LilyPond
(quarter = 60, so one MIDI second = one beat). Only Vivaldi's notes (pitches / rhythm) are taken; the reduction, the
instrumentation and the staging are made here. Ornaments (the trills) are not played - every sound is one visible hit.

The whole movement, bars 1-18, E-flat major, 4/4, quarter = 60:
  mel    Violino principale (solo)         -> the ember ball on copper keys (front lane)
  rain1  Violino primo, pizzicato 16ths     -> raindrops on the upper row of glass chimes outside ('La Pioggia')
  rain2  Violino secondo, pizzicato 16ths   -> raindrops on the lower row of glass chimes outside
  bas    Violoncello, repeated 8ths          -> the iron ball on firewood log drums (back lane)
  (the viola's pianissimo held notes only double the harmony of the two pizzicato parts and are left out of the reduction)
Final chord bar 18 (fermata): solo Eb5, violin 1 G4, violin 2 Eb4, cello Eb2.
Outputs lead.json (every struck event), notes.json (every sound), events.json."""
import json, os
import mido
HERE = os.path.dirname(os.path.abspath(__file__))
Q = 1.0                       # quarter note (60 bpm)
T0 = 2.0                      # bar 1 downbeat
FINAL_BEAT = 68.0             # bar 18 beat 1
NM = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'
def tb(b): return round(T0 + Q * b, 5)
def bar_of(b): return int(b // 4) + 1, round(b % 4 + 1, 3)


def read_parts():
    mf = mido.MidiFile(f'{HERE}/score/winter_midi.midi'); tpb = mf.ticks_per_beat
    PART = {}
    for ti, name in zip(range(1, 6), ('solo', 'vn1', 'vn2', 'vla', 'vc')):
        t = 0; on = {}; ns = []
        for m in mf.tracks[ti]:
            t += m.time
            if m.type == 'note_on' and m.velocity > 0: on[m.note] = t
            elif m.type in ('note_off', 'note_on') and m.note in on:
                s = on.pop(m.note); ns.append((round(s / tpb, 4), round((t - s) / tpb, 4), m.note))
        PART[name] = sorted(ns)
    return PART


PART = read_parts()
ROLE = {'solo': 'mel', 'vn1': 'rain1', 'vn2': 'rain2', 'vc': 'bas'}
VEL = {'mel': 92, 'rain1': 70, 'rain2': 64, 'bas': 80}
EV = []
for part, role in ROLE.items():
    for b, d, p in PART[part]:
        bar, beat = bar_of(b)
        fin = abs(b - FINAL_BEAT) < 1e-6
        v = VEL[role] + (16 if fin else 0)
        if role == 'mel' and bar in (14, 15): v -= 10                 # the echo bars, softer
        EV.append(dict(role=role, beat=b, t=tb(b), pitch=p, dur=round(d * Q, 4), vel=v, bar=bar, beatno=beat, fin=fin))
EV.sort(key=lambda e: (e['t'], e['role'], e['pitch']))
for i, e in enumerate(EV): e['i'] = i
FINAL = tb(FINAL_BEAT)

notes = []
for e in EV:
    d = 4.2 if e['fin'] else max(0.12, e['dur'])
    notes.append([e['t'], round(d, 4), e['role'] + ('_fin' if e['fin'] else ''), e['pitch'], e['vel']])
notes.sort(key=lambda n: n[0])
json.dump(EV, open(f'{HERE}/lead.json', 'w'), indent=0)
json.dump(notes, open(f'{HERE}/notes.json', 'w'))
json.dump(dict(t0=T0, q=Q, bar=4 * Q, final=FINAL, final_beat=FINAL_BEAT,
               gust=tb(44.0), bellows=tb(52.0), lift=tb(62.0)), open(f'{HERE}/events.json', 'w'))
if __name__ == '__main__':
    from collections import Counter
    print('events', len(EV), Counter(e['role'] for e in EV), 'final', FINAL, 'dur', FINAL + 4.5)
    for r in ('mel', 'rain1', 'rain2', 'bas'):
        ts = sorted(set(e['t'] for e in EV if e['role'] == r))
        dmin = min(b - a for a, b in zip(ts, ts[1:])); dmax = max(b - a for a, b in zip(ts, ts[1:]))
        print(r, len(ts), 'gap', round(dmin, 3), '..', round(dmax, 3), 'first', ts[0], 'last', ts[-1],
              'pitches', nm(min(e['pitch'] for e in EV if e['role'] == r)), '-', nm(max(e['pitch'] for e in EV if e['role'] == r)))
    same = sum(1 for a in EV if a['role'] == 'rain1' for b in EV if b['role'] == 'rain2' and a['t'] == b['t'] and a['pitch'] == b['pitch'])
    print('rain unisons', same)
    big = [(e['bar'], e['beatno'], round(n['t'] - e['t'], 2)) for e, n in zip([x for x in EV if x['role'] == 'mel'], [x for x in EV if x['role'] == 'mel'][1:]) if n['t'] - e['t'] > 1.2]
    print('mel gaps > 1.2 s', big)
