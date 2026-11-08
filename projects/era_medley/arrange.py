"""Era medley: travel back in time 1901 -> 1680 with six public-domain classics.
Melodies are taken from this repo's own checked arrangements (pomp, swan_musicbox, mountain_king,
tell_gallop, turkish_march, canon_balls), cut to ~10 s each and joined by short 'time-tunnel' glissandi.
One lead timbre (marimba + glockenspiel) like a single keyboard, light bass and chords on piano."""
import json, math
R = '../'
def load(p): return json.load(open(R + p + '/notes.json'))
notes, lead = [], []           # notes [t,dur,role,midi,vel]; lead [t,midi,vel,sec]
secs = []                      # [year, title_jp, t_start, t_end]
def N(t, d, role, p, v): notes.append([round(t, 4), round(max(0.05, d), 4), role, int(p), int(max(1, min(127, v)))])
def L(t, p, v, s, d=0.3): lead.append([round(t, 4), int(p), int(v), s]); N(t, d, 'lead', p, v)

def take(src, t0, t1, sc, roles, at, s, lead_filter=lambda n: True, vel=None):
    """copy notes of src in [t0,t1) to start at 'at', time-scaled by sc. roles: {src_role: dst_role}"""
    for t, d, r, p, v in src:
        if not (t0 - 1e-6 <= t < t1 - 1e-6) or r not in roles: continue
        tt = at + (t - t0) * sc; dd = d * sc; dst = roles[r]
        if dst == 'lead':
            if not lead_filter((t, d, r, p, v)): continue
            L(tt, p, vel if vel else max(70, v), s, dd)
        else: N(tt, dd, dst, p, min(v, 95))
    return at + (t1 - t0) * sc

def tunnel(t, p0, p1, s, n=8, span=0.85):
    """time-tunnel run: the ball skids over a row of small chimes from p0 to p1"""
    lo = p1 - 12 if abs(p1 - p0) < 7 else p0
    for i in range(n):
        f = i / (n - 1); p = round(lo + (p1 - 1 - lo) * f)
        L(t + span * f * 0.9, p, 70 + 4 * i, s, 0.15)
    N(t, span, 'swoosh', 0, 90)
    return t + span

T = 1.0
# 1) 1901 Pomp and Circumstance No.1 - Trio bars 1-8 (C major), x0.66 time
pomp = load('pomp'); a = T
T = take(pomp, 7.0, 31.0, 0.55, {'bell': 'lead', 'cbp': 'bass', 'hrn': 'chord'}, T, 0)
secs.append([1901, '威風堂々', a, T]); last = [l for l in lead if l[3] == 0][-1][1]
# 2) 1877 Swan Lake - theme A1 (B minor), x0.85
swan = load('swan_musicbox'); swan_lead = [n for n in swan if n[2] == 'box' and n[3] >= 60 and 4.6 <= n[0] < 16.6]
T = tunnel(T, last, swan_lead[0][3], 1); a = T
T = take(swan, 4.6, 16.6, 0.85, {'box': 'lead', 'pizz': 'bass', 'vpad': 'chord'}, T, 1, lambda n: n[3] >= 60)
# box bass notes (comb) -> bass
for t, d, r, p, v in swan:
    if r == 'box' and p < 60 and 4.6 <= t < 16.6: N(a + (t - 4.6) * 0.85, d * 0.85, 'bass', p, v)
secs.append([1877, '白鳥の湖', a, T]); last = lead[-1][1]
# 3) 1875 In the Hall of the Mountain King - 8 bars A B with accelerando (B minor)
mk = load('mountain_king'); ev = json.load(open(R + 'mountain_king/events.json')); bars = ev['bars']
b0 = 16; b1 = b0 + 8          # cycle-2 A + A (accelerating), phrase-aligned
mt0, mt1 = bars[b0], bars[b1]
first = min((n for n in mk if n[2] == 'mar' and mt0 <= n[0] < mt1), key=lambda n: n[0])
T = tunnel(T, last, first[3], 2); a = T
sc = 10.4 / (mt1 - mt0)
T = take(mk, mt0, mt1, sc, {'mar': 'lead', 'cpizz': 'bass', 'tuba': 'bass'}, T, 2)
secs.append([1875, '山の魔王の宮殿にて', a, T]); last = lead[-1][1]
# 4) 1829 William Tell Overture - galop theme A (+ start of A2)
tell = load('tell_gallop'); tev = json.load(open(R + 'tell_gallop/events.json'))
ta = dict((k, v) for k, v in tev['secs'])['A']; tb_ = ta + 24 * tev['bar']
first = min((n for n in tell if n[2] == 'xylo' and ta <= n[0] < tb_), key=lambda n: n[0])
T = tunnel(T, last, first[3], 3); a = T
nb0 = len(notes)
T = take(tell, ta, tb_, 1.0, {'xylo': 'lead', 'cb': 'bass', 'hnst': 'chord'}, T, 3)
# tell_gallop's bass is built from root = 40 + pc, i.e. a major third too high (G#/D# under E/B7 instead of E/B):
# put the galop bass back on the chord roots/fifths
for n in notes[nb0:]:
    if n[2] == 'bass': n[3] -= 4
secs.append([1829, 'ウィリアム・テル序曲', a, T]); last = lead[-1][1]
# 5) 1783 Rondo alla Turca - pick-up + theme A (A minor)
tk = load('turkish_march'); kev = json.load(open(R + 'turkish_march/events.json')); ks = dict((k, v) for k, v in kev['secs'])
k0, k1 = ks['A'] - 2 * kev['q'], ks['B']
first = min((n for n in tk if n[2] == 'lead' and k0 <= n[0] < k1), key=lambda n: n[0])
T = tunnel(T, last, first[3], 4); a = T
nb0 = len(notes); kq = kev['q']; kbar = kev['bar']; ka = T
# Mozart bar 8 is just the long E (the E-F after it is the pick-up into the B section, which the medley skips)
T = take(tk, k0, k1, 1.0, {'lead': 'lead', 'pbass': 'bass', 'pchord': 'chord'}, T, 4, lambda n: n[0] < k1 - kq - 1e-6)
# Mozart bars 5-7: the melody is B A G(natural) A over a bare E-B fifth (LH 'e <h e'>'); bar 7 ends on low B.
# turkish_march had G# there (and a G#-B-E chord), which clashes by one note against the original.
mz = lambda t: round(((t - ka) / kbar - 1) * 8) / 8                       # bar index in the Mozart A section (0 = bar 1)
for n in notes[nb0:]:
    b = mz(n[0])
    if 4 - 1e-6 <= b < 7 - 1e-6:
        if n[2] == 'lead' and n[3] == 80 and n[1] > kq / 2 * 0.9: n[3] = 79
        if n[2] == 'chord' and n[3] == 56: n[1] = -1           # drop the G#
        if n[2] == 'bass' and 6.5 - 1e-6 <= b and n[3] in (40, 28): n[3] += 7   # E -> B on the 2nd beat of bar 7
for l in lead:
    if l[3] == 4 and 4 - 1e-6 <= mz(l[0]) < 7 - 1e-6 and l[1] == 80:
        src = [n for n in notes[nb0:] if n[2] == 'lead' and abs(n[0] - l[0]) < 1e-4]
        if src and src[0][3] == 79: l[1] = 79
notes[:] = [n for n in notes if n[1] > 0]
secs.append([1783, 'トルコ行進曲', a, T]); last = lead[-1][1]
# 6) c.1680 Pachelbel Canon in D - first violin, 4 bars + own ground bass & chords
cn = load('canon_balls'); beat = 0.7142857
c0, c1 = 0.5, 0.5 + 16 * beat
first = min((n for n in cn if n[2] == 0 and c0 <= n[0] < c1), key=lambda n: n[0])
T = tunnel(T, last, first[3], 5); a = T
T = take(cn, c0, c1, 1.0, {0: 'lead'}, T, 5)
GB = [(50, [62, 66, 69]), (45, [61, 64, 69]), (47, [62, 66, 71]), (42, [61, 66, 69]),
      (43, [62, 67, 71]), (38, [62, 66, 69]), (43, [62, 67, 71]), (45, [61, 64, 69])]
for rep in range(1):
    for i, (bs, ch) in enumerate(GB):
        t = a + (rep * 8 + i) * 2 * beat
        N(t, 2 * beat * 0.95, 'bass', bs, 80); [N(t, 2 * beat * 0.95, 'chord', p, 50 + 10 * rep) for p in ch]
secs.append([1680, 'パッヘルベルのカノン', a, T])
# final D major chord: the ball lands in the big bell at the bottom
FINAL = T
L(FINAL, 86, 120, 6, 3.5); L(FINAL, 74, 110, 6, 3.5)
for p in (62, 66, 69, 74, 78): N(FINAL, 3.8, 'chord', p, 100)
for p in (38, 50): N(FINAL, 3.8, 'bass', p, 110)
N(FINAL, 3, 'crash', 0, 105); N(FINAL, 3, 'bt', 0, 100)
for y, n, s, e in secs: N(s - 0.05, 1.5, 'bt', 0, 70)   # year reveal shimmer
notes.sort(key=lambda n: n[0]); lead.sort()
json.dump(notes, open('notes.json', 'w')); json.dump(lead, open('lead.json', 'w'))
json.dump({'t0': 1.0, 'final': FINAL, 'secs': secs}, open('events.json', 'w'), ensure_ascii=False)
print(len(notes), len(lead), round(FINAL, 2), [(y, round(s, 2), round(e - s, 2)) for y, n, s, e in secs])
# min gap check
g = min(b[0] - a_[0] for a_, b in zip(lead, lead[1:]) if b[0] > a_[0]); print('min lead gap', round(g, 3))
