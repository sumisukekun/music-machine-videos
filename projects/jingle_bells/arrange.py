"""Jingle Bells - J. L. Pierpont, 'The One Horse Open Sleigh' (1857, public domain). Own arrangement for marbles.
Score source: score_src.json (PDMX, CC0 1.0; see its 'source' field):
  verse  = the 1857 edition (A-flat major, 2/4): voice line (pickup + 16 bars) and, per beat, the bass note and the chord of
           the 1857 piano part.
  chorus = the familiar chorus (the 1857 print has a different refrain; the one everybody sings is in print / on record since
           at least 1898, public domain), from the public-domain lead sheet in score_src.json, transposed to A-flat.
Written here: the form (verse - chorus - verse - chorus), tempo, who plays what, the chorus harmony (I / IV / II7 / V7, the
standard harmonisation), the chord voicings, the final ritardando.

Form (2/4, bar = 2 quarters):  pickup | V1 bars 1-16 | C1 17-32 | V2 33-48 | C2 49-64 (final chord on bar 64)
Outputs lead.json (melody notes = what the two racing bells strike), events.json (grid, bass, off-beat chords, sections)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = json.load(open(f'{HERE}/score_src.json'))
Q = 0.46                      # quarter (130 bpm)
E8 = Q / 2
BAR = 2 * Q
T0 = 2.6                      # bar 1 beat 1 (the pickup is one eighth earlier)
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'

# ---- melody per bar: (pos in eighths, dur in eighths, midi). Eb4=63 F4=65 G4=67 Ab4=68 Bb4=70 C5=72 Db5=73 D5=74 Eb5=75 F5=77
V = {1: [(0, 1, 63), (1, 1, 72), (2, 1, 70), (3, 1, 68)], 2: [(0, 2, 63), (3, .5, 63), (3.5, .5, 63)],
     4: [(0, 3, 65), (3, 1, 65)], 5: [(0, 1, 65), (1, 1, 73), (2, 1, 72), (3, 1, 70)], 6: [(0, 3, 67), (3, 1, 67)],
     7: [(0, 1, 67), (1, 1, 77), (2, 1, 75), (3, 1, 73)], 8: [(0, 3, 72), (3, 1, 63)],
     10: [(0, 3, 63), (3, .5, 63), (3.5, .5, 63)], 14: [(0, 1, 67), (1, 1, 75), (2, 1, 74), (3, 1, 75)],
     15: [(0, 1, 77), (1, 1, 75), (2, 1, 73), (3, 1, 70)], 16: [(0, 4, 68)]}
V[3] = V[9] = V[11] = V[1]; V[12] = V[4]; V[13] = V[5]
C = {1: [(0, 1, 72), (1, 1, 72), (2, 2, 72)], 3: [(0, 1, 72), (1, 1, 75), (2, 1, 68), (3, 1, 70)], 4: [(0, 4, 72)],
     5: [(0, 1, 73), (1, 1, 73), (2, 1.5, 73), (3.5, .5, 73)], 6: [(0, 1, 73), (1, 1, 72), (2, 1.5, 72), (3.5, .5, 72)],
     7: [(0, 1, 72), (1, 1, 70), (2, 1, 70), (3, 1, 72)], 8: [(0, 2, 70), (2, 2, 75)],
     15: [(0, 1, 75), (1, 1, 75), (2, 1, 73), (3, 1, 70)]}
C[2] = C[1]
for k in range(9, 15): C[k] = C[k - 8]
C[16] = [(0, 3, 68), (3, 1, 63)]             # chorus 1 ends: A-flat + the pickup of verse 2
C_END = [(0, 4, 68)]                         # chorus 2 ends: the final A-flat

# ---- check the melody against the sources
vv = SRC['verse_voice']
src_v = sorted((e[0] - 8, e[1], e[3]) for e in vv if e[0] >= 9)
mine_v = sorted((k, p, m) for k in range(1, 17) for p, d, m in V[k])
assert src_v == mine_v, 'verse vs 1857'
assert [e for e in vv if e[0] == 8] == [[8, 3.0, 1.0, 63]]              # the pickup
cs = sorted(set((e[0], e[1], e[3]) for e in SRC['chorus_src']))
src_bar = {}
for b, p, m in cs: src_bar.setdefault(b, []).append((p, m))
order = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 9, 10]                  # 1st time + 1st ending, 2nd time + 2nd ending
for k in range(1, 16):
    assert sorted((p, m) for p, d, m in C[k]) == sorted(src_bar[order[k - 1]]), ('chorus', k)
assert C[16][0][::2] == (0, 68) and src_bar[10][0] == (0.0, 68)

# ---- verse harmony per beat (eighth pos 0 and 2) from the 1857 piano part: bass = lowest sounding note, chord = all pcs
PN = SRC['verse_piano']
def sounding(bar, pos): return [e for e in PN if e[0] == bar and e[1] <= pos + 1e-6 and e[1] + e[2] > pos + 1e-6]
VBASS = {}; VPCS = {}
for k in range(1, 17):
    for pos in (0, 2):
        s = sounding(k + 8, pos)
        if not s:                                                    # (bar 15/16/24 beat 2: nothing new: keep beat 1)
            VBASS[(k, pos)] = VBASS[(k, 0)]; VPCS[(k, pos)] = VPCS[(k, 0)]; continue
        on = [e[3] for e in s if abs(e[1] - pos) < 1e-6 and e[3] < 60]            # left-hand note struck on this beat
        VBASS[(k, pos)] = min(on) if on else VBASS[(k, 0)] + 7; VPCS[(k, pos)] = sorted(set(e[3] % 12 for e in s))
# name the verse chords (and check that every sounding pc belongs to the named chord or is the melody note)
PC = {'Ab': [8, 0, 3], 'Db': [1, 5, 8], 'Bbm': [10, 1, 5], 'Eb': [3, 7, 10], 'Eb7': [3, 7, 10, 1], 'Bb7': [10, 2, 5, 8]}
VCH = {1: 'Ab', 2: 'Ab', 3: 'Ab', 4: 'Db', 5: 'Bbm', 6: 'Eb', 7: 'Eb', 8: 'Ab', 9: 'Ab', 10: 'Ab', 11: 'Ab', 12: 'Db', 13: 'Bbm',
       14: 'Eb', 15: 'Eb7', 16: 'Ab'}
for (k, pos), pcs in VPCS.items():
    mel = set(m % 12 for p, d, m in V[k] if p <= pos + 1e-6 < p + d)
    extra = set(pcs) - set(PC[VCH[k]]) - mel
    assert not extra, ('verse chord', k, pos, VCH[k], [NM[c] for c in pcs])
# chorus harmony (own): I I I I | IV I II7 V7 | I I I I | IV I V7 I
CCH = {1: 'Ab', 2: 'Ab', 3: 'Ab', 4: 'Ab', 5: 'Db', 6: ('Db', 'Ab'), 7: 'Bb7', 8: 'Eb7', 9: 'Ab', 10: 'Ab', 11: 'Ab', 12: 'Ab',
       13: 'Db', 14: ('Db', 'Ab'), 15: 'Eb7', 16: 'Ab'}
def cch(k, pos): c = CCH[k]; return c if isinstance(c, str) else c[0 if pos < 2 else 1]
ROOT = {'Ab': 44, 'Db': 49, 'Bbm': 46, 'Eb': 51, 'Eb7': 51, 'Bb7': 46}          # Ab2 Db3 Bb2 Eb3
FIFTH = {'Ab': 51, 'Db': 44, 'Bbm': 41, 'Eb': 46, 'Eb7': 46, 'Bb7': 41}        # alternating bass (oom-pah)
for k in CCH:
    for p, d, m in (C[k] if k < 16 else C_END):
        if p not in (0, 2): continue                                    # off-beat passing tones
        ch = cch(k, p)
        assert m % 12 in PC[ch] or (ch == 'Bb7' and m % 12 == 0) or (ch == 'Eb7' and m % 12 == 5) or (ch == 'Db' and m % 12 == 3), ('chorus chord', k, nm(m))
# chord voicing for the off-beat stabs (three notes A3..A4)
def voicing(ch):
    out = []
    for pc in PC[ch][:4] if ch.endswith('7') else PC[ch]:
        p = 57 + (pc - 9) % 12
        out.append(p)
    return sorted(out)

SECT = [('V1', 1), ('C1', 17), ('V2', 33), ('C2', 49)]
def sect_of(bar): return [n for n, b in SECT if bar >= b][-1]
def bar_mel(bar):
    s = sect_of(bar); k = bar - dict(SECT)[s] + 1
    if s[0] == 'V': return V[k], VCH[k], k, s
    return (C[k] if not (s == 'C2' and k == 16) else C_END), k, k, s


NB = 64
def t_of(bar, pos):
    """bar 1..64, pos in eighths. The last two bars slow down (rit.)."""
    if bar < 63: return T0 + (bar - 1) * BAR + pos * E8
    tb = T0 + 62 * BAR
    sl = [1.0, 1.04, 1.08, 1.13, 1.2, 1.28, 1.36, 1.45]               # eighth lengths of bars 63-64 (x E8)
    q = (bar - 63) * 4 + pos
    full = int(q); frac = q - full
    return tb + E8 * (sum(sl[:full]) + (sl[full] * frac if full < len(sl) else 0))


lead = []        # [t, midi, vel, dur_s, bar, pos, section]
lead.append([round(T0 - E8, 5), 63, 84, round(E8, 4), 0, 3.0, 'V1'])
for bar in range(1, NB + 1):
    mel, ch, k, s = bar_mel(bar)
    for p, d, m in mel:
        v = {'V1': 84, 'C1': 96, 'V2': 88, 'C2': 104}[s]
        t = t_of(bar, p); dur = t_of(bar, p + d) - t if not (bar == NB) else 3.0
        lead.append([round(t, 5), m, v, round(dur, 4), bar, p, s])
FINAL = lead[-1][0]
bass = []; stabs = []; chords = {}
for bar in range(1, NB + 1):
    mel, ch, k, s = bar_mel(bar)
    for pos in (0, 2):
        if bar == NB and pos == 2: continue
        if s[0] == 'V':
            ch_b = VCH[k]; bp = VBASS[(k, pos)]
            while bp > 51: bp -= 12
            while bp < 39: bp += 12
        else:
            ch_b = cch(k, pos); bp = ROOT[ch_b] if pos == 0 else FIFTH[ch_b]
        chords[f'{bar}-{pos}'] = ch_b
        bass.append([round(t_of(bar, pos), 5), bp, bar, pos, ch_b])
        if bar < NB: stabs.append([round(t_of(bar, pos + 1), 5), voicing(ch_b), bar, pos + 1, ch_b])
bass.append([round(FINAL, 5), 44, NB, 0, 'Ab']) if not any(abs(b[0] - FINAL) < 1e-6 for b in bass) else None
json.dump(lead, open(f'{HERE}/lead.json', 'w'))
json.dump({'t0': T0, 'q': Q, 'e8': E8, 'bar': BAR, 'final': FINAL, 'bass': bass, 'stabs': stabs, 'chords': chords,
           'secs': [[n, round(t_of(b, 0), 5)] for n, b in SECT],
           'bar_t': {k: round(t_of(k, 0), 5) for k in range(1, NB + 1)}}, open(f'{HERE}/events.json', 'w'))
if __name__ == '__main__':
    print('melody notes', len(lead), 'bass', len(bass), 'stabs', len(stabs), 'final', round(FINAL, 3))
    for (k, pos) in sorted(VBASS): print('verse', k, pos, VCH[k], nm(VBASS[(k, pos)]), [NM[c] for c in VPCS[(k, pos)]])
