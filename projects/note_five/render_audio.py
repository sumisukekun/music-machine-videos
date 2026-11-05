"""Music-box style render: RH = glockenspiel + soft piano, LH = upright piano. VSCO-2 CE samples (CC0)."""
import sys, os, json, math
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
for t, d, voice, p, v in json.load(open('notes.json')):
    fin = d >= 2.5
    if voice == 0:
        play('glock', t, p + 12 if p < 72 else p, v, 0.32 if not fin else 0.4, 0.15)
        play('pno', t, p, v, 0.22, 0.1, dur=max(d, 0.4), rel=0.5)
    else:
        play('pno', t, p, v, 0.5, -0.15, dur=d * 0.95, rel=0.4)
hit(BT, EV['final'] - 0.05, 90, 0.25, 0.2)
L = int(2.2 * SR); tt = np.arange(L) / SR
ir = np.stack([rng.normal(0, 1, L), rng.normal(0, 1, L)], 1) * np.exp(-tt / 0.55)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.6], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.3 * wet
fade = int(2.0 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2:attack=10:release=150,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
