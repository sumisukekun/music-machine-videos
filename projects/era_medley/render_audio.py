"""Era medley - acoustic render with VSCO-2 CE (CC0): marimba+glockenspiel lead, upright piano bass/chords."""
import sys, os, json, math, glob
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
for t, d, role, p, v in json.load(open('notes.json')):
    if role == 'lead':
        play('mar', t, p, v, 0.62, -0.05, dur=max(d, 0.5), rel=0.4)
        play('glock', t, p + 12, v, 0.10, 0.25, dur=max(d, 0.5), rel=0.5)
    elif role == 'bass': play('pno', t, p if p < 60 else p - 12, v, 0.42, -0.15, dur=d, rel=0.25)
    elif role == 'chord': play('pno', t, p, v, 0.16, 0.18, dur=d, rel=0.3)
    elif role == 'swoosh': hit(SHK, t, 90, 0.18, 0.3); hit(BT, t + 0.1, 80, 0.15, -0.3)
    elif role == 'crash': hit(CRASH, t, v, 0.3, 0.2)
    elif role == 'bt': hit(BT, t, v, 0.2, -0.2)
L = int(2.0 * SR); tt = np.arange(L) / SR
ir = np.stack([rng.normal(0, 1, L), rng.normal(0, 1, L)], 1) * np.exp(-tt / 0.45)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.6], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.22 * wet
fade = int(2.0 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
