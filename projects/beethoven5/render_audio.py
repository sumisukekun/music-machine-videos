"""Beethoven 5 - acoustic render with VSCO-2 CE (CC0 recordings only).
python3 render_audio.py /path/to/VSCO-2-CE -> music.wav
Lead (what the balls strike): xylophone / glockenspiel / marimba keys by register, bowl bells and bronze bells =
glockenspiel + marimba, door plates = low marimba + bass drum + log drum, the gong = timpani + crash + glockenspiel.
Accompaniment (the rest of the string parts): upright piano, pizzicato bass, sustained cellos; horn call on horns,
clarinet answer on violins (soft), timpani on C / G in the tutti. Machine sounds: hatches, latches, baskets, ratchet, splash."""
import sys, os, json
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
src = src.replace("DUR = EV['final'] + 4.5", "DUR = EV['final'] + 3.8")
exec(src)
I['hnsus'] = Inst('Brass/F Horn/sus'); I['vsus'] = Inst('Strings/Violin Section/susVib'); I['csus'] = Inst('Strings/Cello Section/susvib')
TIMP = [Hit(sorted(glob.glob(P(f'Percussion/Timpani/Timpani{k}_Hit_*')))) for k in range(1, 6)]
LOGHI = Hit(sorted(glob.glob(P('Percussion/LogDrumHi_*'))))
LAYJ = json.load(open('layout.json')); LEADJ = json.load(open('lead.json'))
KIND = {}
for n, b in LAYJ['balls'].items():
    for h in b['hits']: KIND.setdefault(h['i'], h['kind'])


def mallet(t, p, v, g=1.0, pan=0.0):
    if p >= 79: play('glock', t, p, v, 0.30 * g, pan + 0.15, dur=0.9, rel=0.5); play('xylo', t, p, v, 0.10 * g, pan)
    elif p >= 62: play('xylo', t, p, v, 0.42 * g, pan); play('glock', t, p + 12 if p + 12 <= 96 else p, v, 0.05 * g, pan + 0.2, dur=0.4, rel=0.3)
    else: play('mar', t, p, v, 0.85 * g, pan); play('xylo', t, p + 12, v, 0.06 * g, pan)


def bell(t, p, v, d, g=1.0, pan=0.0):
    q = p
    while q < 67: q += 12
    play('glock', t, q, v, 0.26 * g, pan + 0.1, dur=max(0.8, d + 0.6), rel=0.8)
    play('mar', t, p if p >= 41 else p + 12, v, 0.55 * g, pan - 0.1)


for i, l in enumerate(LEADJ):
    t, p, v, d = l[0], l[1], l[2], l[3]; k = KIND.get(i, 'key')
    if k in ('bowl', 'ropebowl', 'lever', 'bell'): bell(t, p, v, d, 1.15 if k != 'bowl' else 1.0)
    elif k == 'plate':
        play('mar', t, p if p < 60 else p - 12, v, 0.9, -0.1); hit(BD, t, 118, 0.55, 0.0); hit(LOGLO, t + 0.003, 115, 0.45, -0.1)
        mallet(t, p, v, 0.6)
    elif k == 'gong':
        hit(TIMP[1], t, 125, 0.5, 0.0); hit(CRASH, t, 110, 0.4, 0.2); bell(t, p, v, 1.5, 1.2)
    else: mallet(t, p, v, 1.0, -0.05)
    if l[7] is not None: mallet(t, l[7], v - 10, 0.75, 0.3)          # the mirror twin strikes vln II's note
for t, d, role, p, v in json.load(open('notes.json')):
    if role == 'pchord': play('pno', t, p, v, 0.20, 0.15, dur=d, rel=0.25)
    elif role == 'pbass': play('pno', t, p, v, 0.32, -0.15, dur=d, rel=0.25)
    elif role == 'pizz': play('cb', t, p, v, 0.55, -0.05)
    elif role == 'vcsus': play('csus', t, p, v, 0.10, -0.2, dur=d, rel=0.4)
    elif role == 'hn': play('hnsus', t, p, v, 0.28, -0.25, dur=d, rel=0.5)
    elif role == 'clar': play('vsus', t, p, v, 0.10, 0.25, dur=d, rel=0.35)
    elif role == 'timp': hit(TIMP[1 if p >= 46 else 0], t, v, 0.42, -0.05)
    elif role == 'crash': hit(CRASH, t, v, 0.3, 0.25)
# the castle machine's own sounds (times from layout.json)
for kind, t in LAYJ['sfx']:
    if kind == 'hatch': hit(WOOD, t, 70, 0.10, 0.3)
    elif kind == 'latch': hit(CLAV, t, 80, 0.12, 0.2)
    elif kind == 'basket': hit(LOGHI, t, 70, 0.18, 0.2)
    elif kind == 'splash': hit(SHK, t, 90, 0.18, -0.4)
    elif kind == 'ratchet': hit(RAT, t + 0.02, 100, 0.25, -0.2); hit(BD, t, 110, 0.4, 0.0)
EVL = LAYJ['ev']
for t in [EVL['door_open']]:
    hit(RAT, t + 0.05, 110, 0.3, -0.2); hit(BT, t + 0.25, 110, 0.35, -0.3); hit(SLEIGH, t + 0.4, 100, 0.25, 0.3)
    for k in range(10): hit(SHK, t + 0.3 + k * 0.17, 80 - 3 * k, 0.12, -0.3 + 0.06 * k)     # the balls rolling out
# thunder at the lightning flashes (timpani rolls far away)
ROLL = sorted(glob.glob(P('Percussion/Timpani/Rolls/Timpani1_Roll_*')))
for t in [EVL['plates'][0]['t'] + 0.25, EVL['plates'][1]['t'] + 0.35, EVL['gong']['t'] + 0.3, EVL['plates'][2]['t'] + 0.3]:
    x = load(ROLL[-1]); place(x, 0.8, t, 0.22, -0.4, dur=1.3, rel=0.6)
L = int(2.4 * SR); tt = np.arange(L) / SR
ir = np.stack([rng.normal(0, 1, L), rng.normal(0, 1, L)], 1) * np.exp(-tt / 0.55)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.6], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.26 * wet
fade = int(1.8 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
