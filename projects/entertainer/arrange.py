"""The Entertainer (Scott Joplin, 1902, public domain). ~69 s excerpt for the saloon player-piano marble machine.
Notes come unchanged from the Mutopia Project engraving of the 1902 edition (score/entertainer.ly, public domain,
typeset by Chris Sawer / Simon Albrecht).  score/entertainer_midi.ly = the same music with only the MIDI score
(\\unfoldRepeats), rendered by LilyPond to score/entertainer_midi.midi (quarter = 60 so 1 MIDI second = 1 beat).
Excerpt (unfolded 2/4 measures):
  intro   m0-3    bars 1-4   (octave run D-E-C-A-B-G three times, G chord, pick-up D-D#)
  A       m20-35  bars 5-19 + 2nd ending bar 21   (the famous tune, 2nd time through)
  B       m52-67  bars 22-36 + 2nd ending bar 38
  A'      m68-83  bars 39-54 (da capo of A, ends on the C chord - the natural end of the first part)
Tempo quarter = 92.  Outputs notes.json [t, dur, midi, vel, hand] (every sounding note),
lead.json [t, top_midi, n_notes, section]  (one entry per right-hand onset = one visible bell strike),
bass.json [t, top_midi, low_midi, eighth_pos, section] (left-hand onsets from bar 5 on), events.json."""
import json, os
import mido
HERE = os.path.dirname(os.path.abspath(__file__))
BPM = 92.0; BEAT = 60.0 / BPM; BAR = 2 * BEAT
T0 = 1.6                                  # coin drop before the first note
SEL = [0, 1, 2, 3] + list(range(20, 36)) + list(range(52, 68)) + list(range(68, 84))
SECT = ['intro'] * 4 + ['A'] * 16 + ['B'] * 16 + ['A2'] * 16

mf = mido.MidiFile(f'{HERE}/score/entertainer_midi.midi'); tpb = mf.ticks_per_beat
raw = []
for ti, tr in enumerate(mf.tracks[1:], 1):
    t = 0; on = {}
    for msg in tr:
        t += msg.time
        if msg.type == 'note_on' and msg.velocity > 0: on[msg.note] = t
        elif msg.type in ('note_off', 'note_on') and msg.note in on:
            s = on.pop(msg.note); raw.append((s / tpb, (t - s) / tpb, msg.note, 'R' if ti == 1 else 'L'))


def dyn(sec, k, pos):
    """loudness shape from the 1902 dynamics: A strains p / cresc / f / dim per 4 bars, B f (bars 9-12 p)"""
    if sec == 'intro': return 1.0
    if sec == 'B': return 0.82 if 8 <= k < 12 else 1.0
    ph = k % 4
    return [0.8, 0.86 + 0.06 * pos / 2, 1.0, 0.95 - 0.06 * pos / 2][ph]


notes = []
for ei, m in enumerate(SEL):
    sec = SECT[ei]; k = ei - {'intro': 0, 'A': 4, 'B': 20, 'A2': 36}[sec]
    for s, d, p, h in raw:
        if 2 * m <= s < 2 * m + 2 - 1e-9:
            pos = s - 2 * m; t = T0 + (2 * ei + pos) * BEAT
            g = dyn(sec, k, pos)
            v = (88 if h == 'R' else 74) * g
            notes.append([round(t, 5), round(d * BEAT, 5), p, v, h, ei, pos])
# voice balance: the top note of each right-hand onset sings, inner notes softer; left hand: bass (oom) > chord (pah)
by_t = {}
for n in notes: by_t.setdefault((n[4], n[0]), []).append(n)
for (h, t), g in by_t.items():
    top = max(x[2] for x in g); low = min(x[2] for x in g)
    for x in g:
        if h == 'R': x[3] *= 1.0 if x[2] == top else 0.72
        else:
            oom = abs(x[6] * 2 % 2) < 1e-6 and x[5] >= 4          # on the beat (from bar 5)
            x[3] *= (1.05 if x[2] == low else 0.8) if oom else 0.7
for x in notes:
    if x[5] == 3 and abs(x[6] - 1.0) < 1e-6: x[3] = min(122, x[3] * 1.3)     # intro accent chord (^^)
    x[3] = int(max(20, min(127, round(x[3]))))
notes.sort(key=lambda n: (n[0], n[2]))
FINAL = max(n[0] for n in notes)

# ---- visible strikes ----
lead, bass = [], []
for (h, t), g in sorted(by_t.items(), key=lambda kv: kv[0][1]):
    ei = g[0][5]; pos = g[0][6]; sec = SECT[ei]
    ps = sorted(x[2] for x in g)
    if sec == 'intro' and not (ei == 3 and pos >= 1.5):   # intro: one ball plays every onset (both hands in unison)
        if any(abs(l[0] - t) < 1e-6 for l in lead): continue
        allp = sorted(x[2] for hh in 'RL' for x in by_t.get((hh, t), []))
        lead.append([t, allp[-1], len(allp), 'intro', allp[0]])
        continue
    if h == 'R':
        lead.append([t, ps[-1], len(ps), sec if ei >= 4 else 'pick', ps[0]])
    elif ei >= 4:
        e8 = pos * 2; assert abs(e8 - round(e8)) < 1e-6, ('left hand off the eighth grid', ei, pos)
        bass.append([t, ps[-1], ps[0], int(round(e8)), sec])
    elif ei == 3:   # bar 4 left hand at 1.5 sounds together with the right-hand pick-up D
        pass
lead.sort(); bass.sort()
json.dump([n[:5] for n in notes], open(f'{HERE}/notes.json', 'w'))
json.dump(lead, open(f'{HERE}/lead.json', 'w'))
json.dump(bass, open(f'{HERE}/bass.json', 'w'))
secs = [['intro', T0], ['A', T0 + 8 * BEAT], ['B', T0 + 40 * BEAT], ['A2', T0 + 72 * BEAT]]
json.dump({'t0': T0, 'beat': BEAT, 'bar': BAR, 'final': FINAL, 'secs': secs, 'dur': FINAL + 4.5}, open(f'{HERE}/events.json', 'w'), indent=0)
if __name__ == '__main__':
    from collections import Counter
    print('notes', len(notes), 'lead', len(lead), Counter(l[3] for l in lead), 'bass', len(bass), 'final', round(FINAL, 3), 'dur', round(FINAL + 4.5, 2))
    print('lead pitch range', min(l[1] for l in lead), max(l[1] for l in lead), 'distinct', len(set(l[1] for l in lead)))
    rests = [b for b in range(4, 52) if len([x for x in bass if abs(x[0] - (T0 + (2 * b + 1.5) * BEAT)) < 1e-4]) == 0]
    print('bars with a left-hand rest on the last eighth', rests)
