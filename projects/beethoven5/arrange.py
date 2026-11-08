"""Beethoven - Symphony No.5 in C minor, Op.67, 1st movement (1808, public domain): the whole exposition, bars 1-122.
Note data: score.json = the string parts (vln I, vln II, vla, vc, cb) and the flute part of the Mutopia Project edition
(ftp/BeethovenLv/O67/Symphony5_1), compiled to MIDI with LilyPond and checked bar by bar against the score
(the other wind parts of that MIDI are not used: their timing drifts). Rows: [track, start_bar, dur_bar, midi] with
track 1 = flute, 8 = vln I, 9 = vln II, 10 = viola, 11 = cello, 12 = bass; bar 0.0 = downbeat of bar 1.
The horn call (bars 59-62) and the clarinet answer (bars 67-70) are written in by hand from the score.

Arrangement for balls: the melody line (whoever has the tune: strings in imitation, horn call, clarinet / flute answers,
flute in the codetta) is what the balls strike - marimba / xylophone / glockenspiel bars and bronze bells.
Everything else in the strings is played by a piano (harmony) and pizzicato bass (bass line), timpani on the tutti blows.
Tempo: 2/4, one bar = 0.53 s (half note = 113), fermatas on bars 2, 5, 21, 24.
Outputs notes.json (all sounds), lead.json (melody = what the balls strike), events.json."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
BAR = 0.53
T0 = 0.7
FERM = {1: 0.55, 4: 0.75, 20: 0.6, 23: 0.8}          # bar index (0-based) -> extra seconds added at the end of that bar
NBARS = 122


def tb(b):
    """bar position (float, 0 = downbeat of bar 1) -> seconds"""
    t = T0 + b * BAR
    for k, x in FERM.items():
        if b >= k + 1 - 1e-9: t += x
        elif b > k: t += x * (b - k)            # inside a fermata bar the bar is stretched
    return t


S = json.load(open(f'{HERE}/score.json'))
S = [s for s in S if s[1] < NBARS - 1e-6]
# ---- melody (lead) sources, by bar range: instrument priority (first that has an onset at that moment wins)
SRC = [((0, 58), (8, 9, 10, 11)),       # opening, imitation, tutti: the motif passes vln II -> vla -> vln I -> vc
       ((62, 66), (8,)), ((70, 74), (1,)), ((74, 94), (8,)),
       ((94, 109), (8,)), ((109, 116), (1, 8)), ((116, NBARS), (8,))]
MAN = []                                 # hand-written lead: horn call 59-62 and clarinet answer 67-70
for b, p, d in ((58.25, 70, 0.25), (58.5, 70, 0.25), (58.75, 70, 0.25), (59.0, 63, 1.0), (60.0, 65, 1.0), (61.0, 58, 1.5)):
    MAN.append([5, b, d, p])
for k, p in enumerate((70, 75, 74, 75, 77, 72, 72, 70)):
    MAN.append([3, 66 + 0.5 * k, 0.5 if k < 7 else 1.0, p])
lead = []
ons = {}
for tr, b, d, p in S:
    ons.setdefault(round(b * 8) / 8, []).append((tr, p, d))
for (b0, b1), pri in SRC:
    for b in sorted(ons):
        if not (b0 <= b < b1): continue
        for tr in pri:
            c = [(p, d) for t_, p, d in ons[b] if t_ == tr]
            if c:
                p, d = max(c); lead.append([b, p, d, tr]); break
for tr, b, d, p in MAN: lead.append([b, p, d, tr])
lead.sort()
LEADSET = set((round(b, 4), p) for b, p, d, tr in lead)

notes = []
def N(t, d, role, p, v): notes.append([round(t, 5), round(d, 5), role, int(p), int(max(1, min(127, v)))])

DYN = [(0, 118), (5, 70), (13, 82), (17, 104), (21, 118), (24, 66), (32, 84), (37, 100), (43, 120), (56, 112),
       (62, 66), (82, 84), (90, 104), (93, 122)]
def dyn(b):
    v = DYN[0][1]
    for k, x in DYN:
        if b >= k: v = x
    return v


def mallet(p):
    if p >= 79: return 'glock'
    if p >= 62: return 'xylo'
    return 'mar'


# ---- codetta (bars 95-109): vln I and vln II run together; a twin ball strikes the vln II note (unison or octave below)
TWIN = (94.2, 109.9)
v2on = {}
for tr, b, d, p in S:
    if tr == 9 and TWIN[0] <= b <= TWIN[1]: v2on.setdefault(round(b, 4), []).append((p, d))
TWINSET = set()
# ---- lead: struck by the balls
LEAD = []
for b, p, d, tr in lead:
    t = tb(b); v = dyn(b) + (8 if d >= 0.5 else 0)
    role = 'bell' if tr == 5 else mallet(p)
    tw = None
    if TWIN[0] <= b <= TWIN[1] and tr == 8 and round(b, 4) in v2on:
        tw = max(v2on[round(b, 4)])[0]; TWINSET.add((round(b, 4), tw))
        N(t, tb(b + d) - t, mallet(tw), tw, v - 12)
    LEAD.append([round(t, 5), p, int(min(127, v)), round(tb(b + d) - t, 5), role, round(b, 4), tr, tw])
    N(t, tb(b + d) - t, role, p, v)
    if tr == 5: N(t, tb(b + d) - t, 'hn', p, v - 5)
    if tr == 3: N(t, tb(b + d) - t, 'clar', p, v - 10)
# ---- accompaniment: the rest of the strings (piano + pizzicato bass + low strings), flute doubled softly on piano
for tr, b, d, p in S:
    if (round(b, 4), p) in LEADSET and tr in (1, 8, 9, 10, 11): continue
    if tr == 9 and (round(b, 4), p) in TWINSET: continue
    t = tb(b); dur = tb(b + d) - t; v = dyn(b)
    if tr == 12: N(t, dur, 'pizz', p, v - 5)
    elif tr == 11: N(t, dur, 'pbass', p, v - 8); N(t, dur, 'vcsus', p, v - 30) if d >= 0.5 else None
    elif tr == 1: N(t, dur, 'pchord', p, v - 30)
    else: N(t, dur, 'pchord', p, v - 14)
# ---- timpani: on the tutti blows (bass downbeats in the fortissimo bars), from the bass line
TUTTI = [(0, 5), (17, 25), (43, 58), (93, 100), (109, 122)]
for tr, b, d, p in S:
    if tr != 12: continue
    if any(a <= b < c for a, c in TUTTI) and abs(b * 8 - round(b * 8)) < 1e-6 and (d >= 0.45 or (b % 1) == 0):
        pc = p % 12
        if pc not in (0, 7): continue                     # Beethoven's timpani are tuned to C and G
        N(tb(b), 0.6, 'timp', 48 if pc == 0 else 43, dyn(b))
FINAL = tb(121.0)
N(FINAL, 3.2, 'crash', 0, 118)
for p in (39, 51): N(FINAL, 3.0, 'pizz', p, 118)
for p in (55, 58, 63, 67, 70, 75): N(FINAL, 3.0, 'pchord', p, 112)
for k in range(12): N(FINAL - 0.62 + k * 0.05, 0.1, 'timp', 39 + 12, 70 + 4 * k)
notes.sort(key=lambda n: n[0])
json.dump(notes, open(f'{HERE}/notes.json', 'w'))
json.dump(LEAD, open(f'{HERE}/lead.json', 'w'))
secs = [['open', tb(0)], ['imit', tb(5)], ['knock2', tb(21)], ['imit2', tb(24)], ['climb', tb(32)], ['tutti', tb(43)],
        ['horn', tb(56)], ['theme2', tb(62)], ['swell', tb(82)], ['codetta', tb(93)], ['final', tb(109)]]
json.dump({'t0': T0, 'bar': BAR, 'final': FINAL, 'secs': secs, 'nbars': NBARS,
           'bars': [round(tb(b), 5) for b in range(NBARS + 3)]}, open(f'{HERE}/events.json', 'w'), indent=0)
print('lead', len(LEAD), 'notes', len(notes), 'final', round(FINAL, 2), 'end', round(FINAL + 3.8, 2))
