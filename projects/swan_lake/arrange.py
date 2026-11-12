"""Swan Lake - P. I. Tchaikovsky, Op. 20 (1875-76, public domain), Act II No. 10 'Scene' (Moderato), the oboe theme.
Own arrangement for marbles.  Score source: score_src.json (PDMX, CC0 1.0 - see its 'source' field): melody bars 2-18 = the
oboe part, bars 19-26 = the horns' fortissimo statement of the theme, the harp arpeggio of bar 1, and the bass line; the
harmony per beat was read off all parts of that score (listed below as CH).  Original key B minor, 4/4.
Everything else (who plays what, the octave of the struck keys, voicings, the ritardando and the final chord) is written here.

Form (theme bars):  1 harp arpeggio | A 2-5 | A' 6-9 | B 10-13 | B' 14-18 | A tutti 19-25 | 26 final chord
Players:  bar 1 notes 1-6 white ball (rising harp), 7-12 black ball (falling harp); A = white swan alone; A' = black swan
(copies it); B = white; B' = black; tutti = both in unison on mirrored keys; bar 26 = both land on the final bell.
Outputs notes.json (all sounds), lead.json (melody = what the balls strike), events.json."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = json.load(open(f'{HERE}/score_src.json'))
Q = 60 / 90                   # quarter = 90
T0 = 1.0                      # bar 1 beat 1
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'
RIT = [1.0, 1.05, 1.12, 1.22]   # beat lengths in bar 25 (ritardando into the final chord)


def tb(bar, beat):
    """seconds of (bar, beat-in-quarters)"""
    t = T0 + (min(bar, 25) - 1) * 4 * Q
    if bar < 25: return t + beat * Q
    if bar == 25:
        k = int(beat); f = beat - k
        return t + sum(RIT[:k]) * Q + (f * RIT[k] * Q if k < 4 else 0)
    return t + sum(RIT) * Q + (bar - 26) * 4 * Q * 1.25 + beat * Q * 1.25


# ---- melody ----------------------------------------------------------------------------------------------------------------
lead = []          # [t, midi (struck key), vel, dur, section, owner]
def sec_of(bar):
    return {1: 'intro'}.get(bar) or ('A' if bar <= 5 else 'A2' if bar <= 9 else 'B' if bar <= 13 else 'B2' if bar <= 18 else 'tutti' if bar <= 25 else 'final')
OWN = {'intro': None, 'A': 'W', 'A2': 'K', 'B': 'W', 'B2': 'K', 'tutti': 'WK', 'final': 'WK'}
VEL = {'intro': 70, 'A': 74, 'A2': 78, 'B': 84, 'B2': 92, 'tutti': 112, 'final': 118}
harp = [r for r in SRC['harp_intro']]
for k, (bar, off, dur, p) in enumerate(harp):
    lead.append([tb(1, off), p + 12, 66 + 3 * (k if k < 6 else 11 - k), dur * Q, 'intro', 'W' if k < 6 else 'K'])
mel = [r for r in SRC['oboe'] if 2 <= r[0] <= 18]
hn = sorted(set(tuple(r) for r in SRC['horn'] if 19 <= r[0] <= 26))
for bar, off, dur, p in mel:
    s = sec_of(bar); v = VEL[s] + (6 if off == 0 else 0)
    if bar in (17, 18): v += 6
    lead.append([tb(bar, off), p, v, dur * Q, s, OWN[s]])
for bar, off, dur, p in hn:
    if bar == 26: continue
    s = sec_of(bar)
    lead.append([tb(bar, off), p + 12, VEL[s] + (6 if off == 0 else 0), dur * Q, s, 'WK'])     # struck an octave up (oboe octave)
FINAL = tb(26, 0)
lead.append([FINAL, 71, 120, 3.0, 'final', 'WK'])
lead.sort(key=lambda r: r[0])
for r in lead: r[0] = round(r[0], 5); r[3] = round(r[3], 5)

# ---- harmony per beat (read off all parts of the source score; B minor) -------------------------------------------------
# chord name per quarter beat, bars 1-26
CH = {1: 'Bm Bm Bm Bm', 2: 'Bm Bm Em Em', 3: 'Bm Bm Bm Bm', 4: 'Bm Bm E#o7 F#7', 5: 'Bm Em/G Bm/D Em',
      6: 'Bm Bm7/A G#o7 Em/G', 7: 'Bm/F# Bm/F# F#7/E Bm/D', 8: 'F#7/C# Bm/A E#o7/G# F#7', 9: 'Bm Bm Bm Bm/D',
      10: 'A A/C# A7/G A/C#', 11: 'F#m F#m/A F#m/C# F#m/C#', 12: 'Em Em Em7/D Em', 13: 'C# C#/E# F#7 Bm',
      14: 'A A/C# A7/G A/C#', 15: 'F#m F#m/A F#m/C# F#m/C#', 16: 'Em Em Em7/D G/D', 17: 'C C C7/Bb C',
      18: 'C#7/G# C#7/B F#7 F#7/A#', 19: 'Bm Bm C#o7/B Em/B', 20: 'Bm Bm Bm Bm', 21: 'Bm Bm G/B A#o7/B',
      22: 'Bm Bm Em/B Em/B', 23: 'Bm Bm Em/B Em/B', 24: 'Bm Bm Bm Bm', 25: 'Bm Bm G/B A#o7/B', 26: 'Bm'}
PCN = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'E#': 5, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
QUAL = {'': [0, 4, 7], 'm': [0, 3, 7], '7': [0, 4, 7, 10], 'm7': [0, 3, 7, 10], 'o7': [0, 3, 6, 9]}


def chord(name):
    """'F#7/C#' -> (bass pc, [pcs])"""
    base, _, bass = name.partition('/')
    r = base[:2] if len(base) > 1 and base[1] in '#b' else base[:1]
    q = base[len(r):]
    pcs = [(PCN[r] + i) % 12 for i in QUAL[q]]
    return (PCN[bass] if bass else pcs[0]), pcs


notes = []
def N(t, d, role, p, v): notes.append([round(t, 5), round(d, 5), role, int(p), int(max(1, min(127, v)))])


# struck keys (+ the oboe / horns doubling the theme at the written octave)
for t, p, v, d, s, own in lead:
    if s == 'intro':
        N(t, 0.9, 'plate', p, v); N(t, 1.6, 'harp', p - 12, v + 10)
    elif s in ('A', 'A2', 'B', 'B2'):
        N(t, d + 0.4, 'key', p, v)
        N(t, d * 0.98, 'oboe', p, v - 8)
    elif s == 'tutti':
        N(t, d + 0.4, 'key', p, v - 10); N(t, d + 0.4, 'key2', p, v - 14)          # two balls, two mirrored keys
        N(t, d * 0.97, 'horn', p - 12, v); N(t, d * 0.97, 'oboe', p, v - 22)
        N(t, d * 0.97, 'vln', p, v - 30)
DYN = {s: VEL[s] for s in VEL}
# string pad + bass per beat
bass_notes = []   # [t, midi] struck on the bass buoys
for bar in range(1, 27):
    names = CH[bar].split()
    for k, name in enumerate(names):
        t = tb(bar, k); t1 = tb(bar, k + 1) if bar < 26 else t + 3.5
        s = sec_of(bar); dyn = DYN[s]
        bpc, pcs = chord(name)
        prev = names[k - 1] if k > 0 else (CH[bar - 1].split()[-1] if bar > 1 else None)
        # pad: re-voiced close chord in F#3-F#4 (violas), sustained over the beat
        if bar > 1:
            for pc in pcs[:4]:
                pp = 54 + (pc - 6) % 12
                N(t, (t1 - t) * 1.02, 'pad' if s in ('A', 'A2') else 'trem', pp, dyn * 0.42 if s in ('A', 'A2') else dyn * 0.55)
        # bass: on beats 1 and 3, and when the bass changes
        b = 35 + (bpc - 11) % 12
        if b > 42: b -= 12                       # F#1..F2 -> B1..F#2 region
        if (k in (0, 2) or (prev and chord(prev)[0] != bpc)) and bar < 26:
            bass_notes.append([t, b + 12, dyn])
for t, p, v in bass_notes:
    N(t, 1.2, 'pizz', p, v * 0.85); N(t, 1.8, 'cello', p + 12 if p < 47 else p, v * 0.5)
# tutti extras: horns hold the chord roots, timpani on the downbeats
for bar in range(19, 26):
    N(tb(bar, 0), 4 * Q, 'tuba', 35, 100); N(tb(bar, 0), 0.6, 'timp', 47, 96)
# final chord: B minor tutti
for p in (35, 47): N(FINAL, 3.4, 'tuba', p, 118)
for p in (59, 62, 66): N(FINAL, 3.4, 'hornsus', p, 112)
for p in (71, 74, 78, 83): N(FINAL, 3.4, 'vsus', p, 108)
N(FINAL, 3.0, 'harp', 59, 110); N(FINAL, 3.0, 'harp', 66, 110); N(FINAL, 3.0, 'harp', 71, 110)
N(FINAL, 0.6, 'timp', 47, 124); N(FINAL, 3, 'crash', 0, 110)
for k in range(14): N(FINAL - 1.0 + k * 0.07, 0.1, 'timp', 47, 60 + 3 * k)
notes.sort(key=lambda n: n[0])
json.dump(notes, open(f'{HERE}/notes.json', 'w'))
json.dump([[t, p, v, d, s, o] for t, p, v, d, s, o in lead], open(f'{HERE}/lead.json', 'w'))
secs = []
for t, p, v, d, s, o in lead:
    if not secs or secs[-1][0] != s: secs.append([s, t])
json.dump({'t0': T0, 'q': Q, 'final': FINAL, 'secs': secs, 'bass': bass_notes,
           'bars': [round(tb(b, 0), 4) for b in range(1, 27)],
           'chords': [[round(tb(b, k), 4), CH[b].split()[k]] for b in range(1, 27) for k in range(len(CH[b].split()))]},
          open(f'{HERE}/events.json', 'w'), indent=0)
if __name__ == '__main__':
    print('lead', len(lead), 'final', round(FINAL, 2), [(s, round(t, 2)) for s, t in secs])
    for b in range(1, 27):
        print(b, ' '.join(f'{nm(p)}' for t, p, v, d, s, o in lead if tb(b, 0) - 1e-6 <= t < tb(b + 1, 0) - 1e-6))
