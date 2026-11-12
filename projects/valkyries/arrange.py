"""Ride of the Valkyries - R. Wagner, Die Walkuere WWV 86B, Act III (1856, public domain). Own arrangement for marbles.
Score source: score/pdmx_944826_notes.json = PDMX (CC0 1.0), MuseScore score 944826 'Wagner Ride of the Valkyries (Piano solo)',
license cc-zero. The left hand of that score carries the Valkyrie theme in octaves, one cell per 9/8 bar:
    dotted 8th - 16th - 8th - dotted quarter - dotted quarter   (sixteenth positions 0, 3, 4, 6, 12)
Taken from the source (checked below, note by note): the pitches and the rhythm of every theme cell of the B-minor statement
(cells T1-T8, source bars 14-24) and of the B-major statement with its cadence (cells M1-M8, E1, E2, C, source bars 133-148),
the harmony of each cell (all chord tones sounding in the source within that cell), the final B-major chord.
Written here: the form (B-minor theme twice, then the B-major theme), who plays what (8 riders in octaves, the thunder ball doubles
the theme two octaves down on beats 1 and 3, on a held note it strikes that note again on beat 3), the B-major cells set one per bar as in the B-minor statement (the source spaces
them over two bars there), the long notes at the end of each half held for the rest of their bar only.
Bar = 2.0 s (9/8, dotted quarter = 90).  Outputs lead.json and events.json."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = json.load(open(f'{HERE}/score/pdmx_944826_notes.json'))['notes']
BAR = 2.0
S16 = BAR / 18
T0 = 1.8                                   # bar 1 beat 1
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'
PCN = {n: i for i, n in enumerate(NM)}

# ---- theme cells, upper octave (source left hand + 2 octaves): [(pos16, dur16, midi)], chord of beat 1-2 and of beat 3
def cell(ps, held=False):
    pos = [0, 3, 4, 6] if held else [0, 3, 4, 6, 12]
    dur = [3, 1, 2, 12] if held else [3, 1, 2, 6, 6]
    return list(zip(pos, dur, ps))
CELLS = {
    'T1': (cell([59, 54, 59, 62, 59]), 'Bm', 'Bm'), 'T2': (cell([62, 59, 62, 66, 62]), 'Bm', 'Bm'),
    'T3': (cell([66, 62, 66, 69, 57]), 'D', 'A'), 'T4': (cell([62, 57, 62, 66], True), 'D', 'D'),
    'T5': (cell([62, 57, 62, 66, 62]), 'D', 'D'), 'T6': (cell([66, 62, 66, 69, 66]), 'D', 'D'),
    'T7': (cell([69, 66, 69, 73, 61]), 'F#m', 'C#'), 'T8': (cell([66, 61, 66, 70], True), 'F#', 'F#'),
    'M1': (cell([59, 54, 59, 63, 59]), 'B', 'B'), 'M2': (cell([63, 59, 63, 66, 63]), 'B', 'B'),
    'M3': (cell([66, 63, 66, 70, 58]), 'D#m', 'D#m'), 'M4': (cell([63, 58, 63, 66], True), 'B', 'B'),
    'M5': (cell([63, 58, 63, 66, 63]), 'D#m', 'D#m'), 'M6': (cell([66, 63, 66, 70, 66]), 'D#m', 'D#m'),
    'M7': (cell([70, 66, 70, 73, 61]), 'F#', 'F#'), 'M8': (cell([66, 61, 66, 70], True), 'F#', 'F#'),
    'E1': (cell([56, 52, 56, 59], True), 'E', 'E'), 'E2': (cell([56, 52, 56, 59], True), 'E', 'E'),
    'C': (cell([56, 52, 56, 61, 54]), 'C#m', 'F#'),
}
PCS = {'Bm': 'B D F#', 'D': 'D F# A', 'A': 'A C# E', 'F#m': 'F# A C#', 'C#': 'C# F G#', 'F#': 'F# A# C#', 'B': 'B D# F#',
       'D#m': 'D# F# A#', 'E': 'E G# B', 'C#m': 'C# E G#'}
PCS = {k: [PCN[x] for x in v.split()] for k, v in PCS.items()}
# where each cell is in the source: onset of its first note, and the parse's unit per sixteenth there
ANCH = {'T1': 309.5, 'T2': 327.5, 'T3': 345.5, 'T4': 363.5, 'T5': 466, 'T6': 484, 'T7': 502, 'T8': 520,
        'M1': 3740.2, 'M2': 3785.8, 'M3': 3831.4, 'M4': 3876.7, 'M5': 3921.1, 'M6': 3966.7, 'M7': 4012.3, 'M8': 4057.6,
        'E1': 4101.7, 'E2': 4145.8, 'C': 4190.2, 'FIN': 4235.5}
UNIT = {k: (1.0 if k[0] == 'T' else 1.2) for k in ANCH}

# ---- check every theme note and every chord against the source
def src_at(t, tol=0.3): return [n for n in SRC if abs(n[0] - t) <= tol]
for k, (notes, c1, c3) in CELLS.items():
    a, u = ANCH[k], UNIT[k]
    for pos, d, p in notes:
        cand = [n for n in src_at(a + pos * u) if n[2] % 12 == p % 12 and n[2] < 60]
        assert cand, ('theme note not in source', k, pos, nm(p))
        assert any(abs(n[1] - d * u) < 0.25 for n in cand) or d == 12, ('duration', k, pos, nm(p), [n[1] for n in cand])
    # the parse lets the right hand (the arpeggio figures) drift against the left hand (up to a bar late, a beat early):
    # look from one beat before the cell to the end of the bar after it
    win = set(n[2] % 12 for n in SRC if a - 6 * u <= n[0] < a + 36 * u + 0.5)
    for ch in (c1, c3):
        miss = set(PCS[ch]) - win
        assert not miss, ('chord tone not sounding in the source cell', k, ch, [NM[x] for x in miss])
fin = set(n[2] % 12 for n in SRC if abs(n[0] - ANCH['FIN']) < 0.3)
assert fin == set(PCS['B']), fin
last = max(n[0] for n in SRC); assert set(n[2] % 12 for n in SRC if n[0] > last - 3.5 and n[1] >= 2) <= set(PCS['B'])

# ---- form: bars 1-8 the riders one by one (relay), 9-16 all eight race, 17-27 B major, bar 28 beat 1 the final chord
FORM = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8'] * 2 + ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'E1', 'E2', 'C']
NBAR = len(FORM)
FINAL = T0 + NBAR * BAR
def t_of(bar, pos16): return round(T0 + (bar - 1) * BAR + pos16 * S16, 5)

lead = []                                   # [t, midi (upper octave), vel, dur_s, bar, pos16, cell]
bass = []                                   # [t, midi, bar, pos16, chord]
chords = {}
for b, k in enumerate(FORM, start=1):
    notes, c1, c3 = CELLS[k]
    for pos, d, p in notes:
        v = 78 if b <= 8 else (92 if b <= 16 else 104)
        lead.append([t_of(b, pos), p, v, round(d * S16, 4), b, pos, k])
        if pos in (0, 12): bass.append([t_of(b, pos), p - 24, b, pos, c1 if pos == 0 else c3])
    if notes[-1][1] == 12:                  # held long note: the thunder ball still strikes beat 3 (same pitch, the held note)
        bass.append([t_of(b, 12), notes[-1][2] - 24, b, 12, c3])
    chords[f'{b}-0'] = c1; chords[f'{b}-12'] = c3
# the final chord (the eight riders land together): B major, B2 .. D#5; the thunder ball B1
FIN_P = [47, 51, 54, 59, 63, 66, 71, 75]
bass.append([round(FINAL, 5), 35, NBAR + 1, 0, 'B'])
for b_ in bass: assert b_[1] % 12 in PCS[b_[4]], b_
for l in lead:
    ch = chords[f'{l[4]}-{12 if l[5] >= 12 else 0}']
    if l[5] in (0, 6, 12): assert l[1] % 12 in PCS[ch] or l[6] == 'M4', l          # every long theme note is a chord tone
json.dump(lead, open(f'{HERE}/lead.json', 'w'))
json.dump({'t0': T0, 'bar': BAR, 's16': S16, 'final': FINAL, 'form': FORM, 'bass': bass, 'chords': chords, 'fin_p': FIN_P,
           'pcs': PCS, 'bar_t': {b: t_of(b, 0) for b in range(1, NBAR + 2)}}, open(f'{HERE}/events.json', 'w'))
if __name__ == '__main__':
    print('bars', NBAR, 'melody notes', len(lead), 'bass', len(bass), 'final', round(FINAL, 3))
