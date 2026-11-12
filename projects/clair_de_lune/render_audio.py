"""Clair de Lune - acoustic render with VSCO-2 CE (CC0 samples only).  python3 render_audio.py /path/to/VSCO-2-CE -> music.wav
What you hear is what you see (every pitched sound is a ball striking something at that moment):
  lotus keys (twins)          glockenspiel (the metallophone 'ting') + the piano (the piece is a piano piece)
  bamboo xylophone slides     marimba + piano
  glass bowls (drops)         piano (the harmony), a soft glockenspiel shimmer on the upper bowls
  bronze slab / plate / bars  piano bass + contrabass pizzicato an octave below; the see-saw and the gong add a soft timpani
  final                       D-flat major: piano, glockenspiel on the flower bells, timpani + bell tree on the gong
  mechanisms                  release (bell tree), basket / buffer / bucket / tip (wood), pulley (ratchet), latch (claves),
                              trapdoor, see-saw thump, shishi-odoshi 'kon' (log drum), water: drops, spill, the big splash
A soft running-water bed (filtered noise) and a long night-garden reverb."""
import sys, os, json, glob
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
LAY = json.load(open('layout.json')); E = json.load(open('events.json'))
I['vcsus'] = Inst('Strings/Cello Section/susvib')
TIMP = Hit(sorted(glob.glob(P('Percussion/Timpani/Timpani1_Hit_*'))))
FINAL = E['final']
KIND = {}
for n in ('silver', 'gold'):
    for h in LAY['balls'][n]['hits']:
        if h['pitch'] is not None: KIND[(round(h['t'], 5), h['pitch'])] = h['kind']
for t, d, role, p, v in json.load(open('notes.json')):
    k = KIND.get((round(t, 5), p), '')
    if role in ('hi', 'lo', 'uni'):
        g = 1.0 if role != 'lo' else 0.8
        if k.startswith('slide'): play('mar', t, p, v, 0.32 * g, -0.1, dur=0.8, rel=0.4)
        elif k == 'flower': play('glock', t, p, v, 0.3, 0.1, dur=3.0, rel=1.2)
        else: play('glock', t, p, v, 0.17 * g, 0.12, dur=1.2, rel=0.6)
        play('pno', t, p, v, 0.34 * g, -0.05, dur=d + 0.2, rel=0.7)
    elif role == 'drop':
        play('pno', t, p, v, 0.27, -0.2, dur=d + 0.3, rel=0.8)
        if p >= 62: play('glock', t, p + 12, v, 0.035, 0.35, dur=0.8, rel=0.5)
    elif role == 'bass':
        play('pno', t, p, v, 0.42, 0.0, dur=d + 0.4, rel=1.0)
        play('pno', t, p + 12, v, 0.12, 0.0, dur=d + 0.4, rel=1.0)
        play('cb', t, p - 12 if p - 12 >= 28 else p, v, 0.22, 0.05)
        if abs(t - LAY['ev']['seesaw']['t_hit']) < 1e-4: hit(TIMP, t, 90, 0.22, 0.0)
        if abs(t - FINAL) < 1e-4:
            hit(TIMP, t, 110, 0.32, 0.0); hit(BT, t + 0.05, 100, 0.25, -0.3); play('vcsus', t, p + 12, 90, 0.15, 0.0, dur=3.5, rel=1.5)


def noise_swell(t, dur, gain, lp=0.97, attack=0.15, pan=0.0, decay=0.45):
    n = int(dur * SR); x = rng.normal(0, 1, (n, 2)).astype('float32')
    for c in range(2): x[:, c] = lfilter([1 - lp], [1, -lp], x[:, c])
    env = np.minimum(1, np.arange(n) / (attack * SR)) * np.exp(-np.arange(n) / (decay * dur * SR))
    x *= (env * gain / (np.abs(x).max() + 1e-9))[:, None]
    i = int(t * SR); e = min(N, i + n); out[i:e] += x[:e - i] * np.array([1 - 0.3 * pan, 1 + 0.3 * pan], 'float32')


for kind, t in LAY['sfx']:
    if kind == 'release': hit(BT, t, 70, 0.14, -0.3)
    elif kind in ('basket', 'cupA', 'pad'): hit(WOOD, t, 70, 0.12, 0.0)
    elif kind == 'zip': noise_swell(t, 1.8, 0.05, 0.9, 0.4, 0.2, 0.8)
    elif kind == 'buffer': hit(WOOD, t, 90, 0.18, 0.2); hit(LOGLO, t, 70, 0.1, 0.2)
    elif kind == 'bucket': hit(LOGLO, t, 70, 0.14, -0.2)
    elif kind == 'pulley': [hit(RAT, t + 0.35 * k, 45, 0.05, -0.2) for k in range(4)]
    elif kind == 'tip': hit(WOOD, t, 80, 0.14, 0.1)
    elif kind == 'latch': hit(CLAV, t, 70, 0.12, 0.2)
    elif kind == 'gate': hit(WOOD, t, 80, 0.16, 0.2)
    elif kind == 'seesaw': hit(LOGLO, t + 0.003, 90, 0.18, 0.0)
    elif kind == 'launch': noise_swell(t, 0.6, 0.06, 0.8, 0.05, 0.0, 0.5)
    elif kind == 'chute': hit(LOGLO, t, 60, 0.1, 0.2)
    elif kind == 'shishi':
        noise_swell(t, 0.9, 0.09, 0.93, 0.02, 0.3, 0.5)                     # the water spills
        hit(LOGLO, t + 0.75, 105, 0.32, 0.2); hit(WOOD, t + 0.752, 100, 0.2, 0.2)   # 'kon': the tube knocks back on its stone
    elif kind == 'splash_big': noise_swell(t, 2.6, 0.22, 0.95, 0.01, 0.0, 0.35)
for d in LAY['ev']['drops']: noise_swell(d['t'] + 0.01, 0.12, 0.012, 0.6, 0.002, 0.5, 0.4)   # a tiny splash in the bowl
# running water bed (very soft)
bed = rng.normal(0, 1, (N, 2)).astype('float32')
for c in range(2): bed[:, c] = lfilter([0.02], [1, -0.98], bed[:, c])
bed *= 0.012 / (np.abs(bed).max() + 1e-9) * 3
fade_in = np.minimum(1, np.arange(N) / (2.0 * SR))[:, None]
out += bed * fade_in
# night garden: long, soft reverb
L_ = int(3.2 * SR); tt = np.arange(L_) / SR
ir = np.stack([rng.normal(0, 1, L_), rng.normal(0, 1, L_)], 1) * np.exp(-tt / 0.9)[:, None]
for c in range(2): ir[:, c] = lfilter([0.25], [1, -0.7], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.3 * wet
fade = int(2.5 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-20dB:ratio=2.2:attack=10:release=150,loudnorm=I=-15:TP=-1.2:LRA=10" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
