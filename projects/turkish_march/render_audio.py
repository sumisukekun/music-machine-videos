"""Rondo alla Turca - acoustic render with VSCO-2 CE (CC0): glockenspiel+xylophone bell-keys, upright piano,
contrabass pizz, bass drum (davul), crash cymbals (zil), ratchet (the clockwork being wound)."""
import sys, os, json, glob
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
for t, d, role, p, v in json.load(open('notes.json')):
    if role == 'lead':
        play('glock', t, p + 12, v, 0.26, 0.1, dur=max(d, 0.45), rel=0.5); play('xylo', t, p + 12, v, 0.08, 0.25)
        play('mar', t, p, v, 0.42, -0.05)
    elif role == 'pdouble': play('pno', t, p, v, 0.22, 0.05, dur=d, rel=0.15)
    elif role == 'pbass': play('pno', t, p, v, 0.42, -0.15, dur=d, rel=0.2)
    elif role == 'pchord': play('pno', t, p, v, 0.20, 0.15, dur=d, rel=0.15)
    elif role == 'cb': play('cb', t, p, v, 0.55, 0.0)
for t, v in EV['drum']: hit(BD, t, v, 0.55, 0.0); hit(TAP, t + 0.004, v, 0.12, -0.2)
for t, v in EV['crash']: hit(CRASH, t, v, 0.32, 0.25)
hit(RAT, 0.15, 90, 0.3, -0.4); hit(RAT, 0.9, 90, 0.25, -0.4)
hit(BT, EV['final'] + 0.05, 100, 0.25, 0.3)
L = int(1.8 * SR); tt = np.arange(L) / SR
ir = np.stack([rng.normal(0, 1, L), rng.normal(0, 1, L)], 1) * np.exp(-tt / 0.4)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.6], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.22 * wet
fade = int(2.0 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
