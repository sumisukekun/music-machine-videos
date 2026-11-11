"""Eine kleine Nachtmusik - acoustic render with VSCO-2 CE (CC0 samples only).  python3 render_audio.py /path/to/VSCO-2-CE -> music.wav
What you hear is what you see (every pitched sound is a ball striking a key / bell / bar at that moment):
  candlestick keys (lamp)    glockenspiel + soft piano + the violin section (spiccato, or sustained for long notes)
  xylophone ramp (lamp)      xylophone + violins
  porcelain keys (couple)    piano + second violins (quieter)
  gold bells / bass bars     piano bass + cello section + contrabass pizzicato an octave below
  final chord                grand chandelier bell (low G), cup bells, string tutti
  mechanisms                 chandelier trapdoor, the gust (wind + window bang), the brazier, the fuse crackle, bucket lifts,
                             see-saw, launch"""
import sys, os, json, glob, re
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
LAY = json.load(open('layout.json')); LEAD = json.load(open('lead.json')); E = json.load(open('events.json'))
KIND = {}
for h in LAY['balls']['lamp']['hits']:
    if h.get('ev') is not None: KIND[h['ev']] = h['kind']
TN = [l[0] for l in LEAD]
def lamp_ev(t):
    k = min(range(len(TN)), key=lambda i: abs(TN[i] - t)); return k if abs(TN[k] - t) < 1e-4 else None
I['vln'] = Inst('Strings/Violin Section/susVib'); I['vlnsp'] = Inst('Strings/Violin Section/Spic')
I['vcsus'] = Inst('Strings/Cello Section/susvib'); I['vcsp'] = Inst('Strings/Cello Section/spic')
TRI = Hit(sorted(glob.glob(P('Percussion/Triangle3-Hit_*')))) if glob.glob(P('Percussion/Triangle3-Hit_*')) else BT
TIMP = Hit(sorted(glob.glob(P('Percussion/Timpani/Timpani1_Hit_*'))))
Q = E['q']
for t, d, role, p, v in json.load(open('notes.json')):
    if role == 'lamp':
        i = lamp_ev(t); kd = KIND.get(i, 'candle')
        if kd in ('xylo', 'xylo2', 'xyloland', 'xylo2land'): play('xylo', t, p, v, 0.2, 0.15, dur=0.5, rel=0.3)
        else: play('glock', t, p, v, 0.2, 0.15, dur=1.0, rel=0.5)
        play('pno', t, p, v, 0.12, 0.1, dur=d, rel=0.25)
        if d > 1.6 * Q: play('vln', t, p, v, 0.22, -0.15, dur=d, rel=0.25)
        else: play('vlnsp', t, p, v, 0.26, -0.15)
    elif role == 'duo':
        play('pno', t, p, v, 0.16, 0.3, dur=d, rel=0.25)
        if d > 1.6 * Q: play('vln', t, p, v, 0.11, 0.35, dur=d, rel=0.25)
        else: play('vlnsp', t, p, v, 0.13, 0.35)
    elif role == 'bass':
        play('pno', t, p, v, 0.2, -0.1, dur=min(d, 0.9), rel=0.3)
        if d > 1.6 * Q: play('vcsus', t, p, v, 0.24, -0.25, dur=d, rel=0.3)
        else: play('vcsp', t, p, v, 0.3, -0.25)
        play('cb', t, p - 12, v, 0.22, -0.05)
    elif role == 'tutti':
        if p < 50: play('vcsus', t, p, v, 0.3, -0.2, dur=d, rel=1.0); play('cb', t, p - 12, v, 0.3, 0.0)
        elif p < 66: play('vcsus', t, p, v, 0.18, 0.1, dur=d, rel=1.0)
        else: play('vln', t, p, v, 0.18, 0.2, dur=d, rel=1.0)
    elif role == 'lampfin': play('glock', t, p, v, 0.3, 0.0, dur=2.5, rel=1.0); play('pno', t, p, v, 0.18, 0.0, dur=d, rel=0.8)
    elif role == 'duofin': play('glock', t, p, v, 0.2, 0.3, dur=2.0, rel=1.0); play('pno', t, p, v, 0.15, 0.3, dur=d, rel=0.8)
    elif role == 'bigbell':
        for pp, g in ((p - 12, 0.32), (p, 0.3), (p + 12, 0.18), (p + 19, 0.08)): play('pno', t, pp, v, g, 0.0, dur=d, rel=1.0)
        hit(TIMP, t, 120, 0.35, 0.0); hit(CRASH, t, 110, 0.22, 0.2); hit(BT, t + 0.05, 110, 0.3, -0.25)


def noise_swell(t, dur, gain, lp=0.97, attack=0.15, pan=0.0):
    n = int(dur * SR); x = rng.normal(0, 1, (n, 2)).astype('float32')
    for c in range(2): x[:, c] = lfilter([1 - lp], [1, -lp], x[:, c])
    env = np.minimum(1, np.arange(n) / (attack * SR)) * np.exp(-np.arange(n) / (0.45 * dur * SR))
    x *= (env * gain / (np.abs(x).max() + 1e-9))[:, None]
    i = int(t * SR); e = min(N, i + n); out[i:e] += x[:e - i] * np.array([1 - 0.3 * pan, 1 + 0.3 * pan], 'float32')


for kind, t in LAY['sfx']:
    if kind == 'release': hit(BT, t, 80, 0.2, 0.2); hit(WOOD, t, 80, 0.15, 0.0)
    elif kind == 'gust':
        hit(SLAP, t, 120, 0.4, 0.3); hit(LOGLO, t + 0.01, 110, 0.3, 0.3); noise_swell(t, 2.6, 0.5, 0.985, 0.12, -0.6)
    elif kind == 'brazier':
        noise_swell(t - 0.01, 1.4, 0.45, 0.93, 0.01, 0.0); hit(BD, t, 110, 0.35, 0.0); hit(CRASH, t + 0.02, 100, 0.18, -0.2)
        for k in range(26): hit(WOOD, t + 0.35 + 0.15 * k + rng.uniform(0, 0.05), 50, 0.05, rng.uniform(-0.8, 0.8))   # the fuse crackles
    elif kind == 'bucket': hit(LOGLO, t, 90, 0.28, -0.3)
    elif kind == 'winch': [hit(RAT, t + 0.3 * k, 60, 0.07, -0.3) for k in range(3)]
    elif kind == 'tip': hit(WOOD, t, 85, 0.18, -0.3)
    elif kind == 'seesaw': hit(BD, t, 120, 0.45, 0.0); hit(LOGLO, t + 0.003, 110, 0.35, 0.1)
    elif kind == 'launch': hit(SLAP, t + 0.02, 100, 0.25, 0.1)
    elif kind == 'land_cup': hit(WOOD, t, 70, 0.15, 0.0)
# palace hall
L_ = int(2.2 * SR); tt = np.arange(L_) / SR
ir = np.stack([rng.normal(0, 1, L_), rng.normal(0, 1, L_)], 1) * np.exp(-tt / 0.55)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.55], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.24 * wet
fade = int(2.5 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
