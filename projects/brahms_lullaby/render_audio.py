"""Brahms' Lullaby - acoustic render with VSCO-2 CE (CC0 samples only).  python3 render_audio.py /path/to/VSCO-2-CE -> music.wav
What you hear is what you see (every pitched sound is a ball striking a chime / block / drum / bell at that moment):
  star chimes (star marble)       glockenspiel (music box) + soft piano + a quiet violin line for the long notes
  toy-block keys (twins)          soft piano + a touch of marimba
  sheep drums (sheep)             piano bass + soft cello; the Eb2 drum adds a pizzicato contrabass an octave below
  see-saw chord bells (giant)     piano + cello chord
  the mobile (final chord)        moon bell Eb5 + star bells G5 Bb5 (glockenspiel + piano + bell tree)
  mechanisms                      cloud puffs, pillow landings, the see-saw thump, the light switch"""
import sys, os, json, glob, re
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
LAY = json.load(open('layout.json')); E = json.load(open('events.json'))
I['vln'] = Inst('Strings/Violin Section/susVib'); I['vcsus'] = Inst('Strings/Cello Section/susvib')
TRI = Hit(sorted(glob.glob(P('Percussion/Triangle3-Hit_*'))))
Q = E['q']
for t, d, role, p, v in json.load(open('notes.json')):
    if role == 'star':
        play('glock', t, p, v, 0.17, 0.12, dur=1.3, rel=0.6)
        play('pno', t, p, v, 0.2, 0.08, dur=d, rel=0.35)
        if d > 1.6 * Q: play('vln', t, p, v, 0.07, -0.1, dur=d, rel=0.4)
    elif role == 'twins':
        play('pno', t, p, v, 0.15, 0.3, dur=d, rel=0.3)
        play('mar', t, p, v, 0.05, 0.35)
    elif role == 'sheep':
        play('pno', t, p, v, 0.22, -0.2, dur=min(d, 1.2), rel=0.35)
        play('vcsus', t, p + (12 if p < 40 else 0), v, 0.06, -0.3, dur=d, rel=0.3)
        if p < 40: play('cb', t, p, v, 0.14, -0.1)
    elif role == 'moon':
        play('glock', t, p, v, 0.32, 0.0, dur=3.0, rel=1.2)
        for pp, g in ((p - 24, 0.22), (p - 12, 0.2), (p, 0.22)): play('pno', t, pp, v, g, 0.0, dur=d, rel=1.2)
        hit(BT, t + 0.03, 100, 0.28, -0.2)
    elif role == 'twinbell':
        play('glock', t, p, v, 0.24, 0.3 if p > 80 else -0.3, dur=2.6, rel=1.0); play('pno', t, p, v, 0.1, 0.0, dur=d, rel=1.0)


def noise_swell(t, dur, gain, lp=0.97, attack=0.15, pan=0.0):
    n = int(dur * SR); x = rng.normal(0, 1, (n, 2)).astype('float32')
    for c in range(2): x[:, c] = lfilter([1 - lp], [1, -lp], x[:, c])
    env = np.minimum(1, np.arange(n) / (attack * SR)) * np.exp(-np.arange(n) / (0.45 * dur * SR))
    x *= (env * gain / (np.abs(x).max() + 1e-9))[:, None]
    i = int(t * SR); e = min(N, i + n); out[i:e] += x[:e - i] * np.array([1 - 0.3 * pan, 1 + 0.3 * pan], 'float32')


for kind, t in LAY['sfx']:
    if kind in ('release_star', 'release_twins'): hit(WOOD, t, 60, 0.08, 0.2)
    elif kind == 'cloud': noise_swell(t, 0.35, 0.05, 0.9, 0.02, -0.3)
    elif kind == 'cloud_big': noise_swell(t, 0.6, 0.12, 0.92, 0.02, 0.0)
    elif kind == 'pillow': hit(TAP, t, 40, 0.07, -0.4)
    elif kind == 'seesaw': hit(BD, t, 105, 0.32, 0.0); hit(LOGLO, t + 0.003, 95, 0.25, 0.1)
    elif kind == 'launch': hit(SLAP, t + 0.02, 70, 0.08, 0.1)
    elif kind == 'land_cup': hit(WOOD, t, 50, 0.07, 0.0)
    elif kind == 'switch': hit(CLAV, t, 70, 0.12, 0.3)
# the light switch after the final chord (the room goes dark)
hit(CLAV, LAY['ev']['lights_out'], 70, 0.1, 0.3)
# nursery room
L_ = int(1.8 * SR); tt = np.arange(L_) / SR
ir = np.stack([rng.normal(0, 1, L_), rng.normal(0, 1, L_)], 1) * np.exp(-tt / 0.42)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.6], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.22 * wet
fade = int(2.5 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-20dB:ratio=2.2:attack=10:release=150,loudnorm=I=-15:TP=-1.5:LRA=10" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
