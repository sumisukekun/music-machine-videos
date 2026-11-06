"""Csikos Post - acoustic render with VSCO-2 CE (CC0): xylophone lead, trumpet doubling, horns/tuba oom-pah, piano, pizz bass, drums."""
import sys, os, json, math, glob, re
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
I['tpt'] = Inst('Brass/Trumpet/stac'); I['horn'] = Inst('Brass/F Horn/stac'); I['tuba'] = Inst('Brass/Tuba/stac')
SN = Hit(sorted(glob.glob(P('Percussion/Snare2-HitNS_*')), key=lambda f: int(re.search(r'_v(\d)', f).group(1))))
ROLL = Hit(sorted(glob.glob(P('Percussion/Snare2-rollNS_*'))))
TIMP = Hit(sorted(glob.glob(P('Percussion/Timpani/Timpani1_Hit_*'))))
for t, d, role, p, v in json.load(open('notes.json')):
    if role == 'xylo':
        play('xylo', t, p, v, 0.55, 0.05); play('mar', t, p, v, 0.22, -0.1, dur=max(d, 0.25), rel=0.2)
    elif role == 'tpt': play('tpt', t, p, v, 0.20, 0.25, dur=d, rel=0.08)
    elif role == 'glock': play('glock', t, p, v, 0.16, 0.3)
    elif role == 'rh': play('pno', t, p, v, 0.20, 0.15, dur=d, rel=0.15)
    elif role == 'lh': play('pno', t, p, v, 0.22, -0.15, dur=d, rel=0.15)
    elif role == 'bass': play('cb', t, p, v, 0.55, -0.05); play('pno', t, p, v, 0.25, -0.2, dur=d, rel=0.2)
    elif role == 'tuba': play('tuba', t, p, v, 0.30, -0.1, dur=d, rel=0.08)
    elif role == 'horn': play('horn', t, p, v, 0.10, -0.3, dur=d, rel=0.08)
for t, k, v in EV['perc']:
    if k == 'bd': hit(BD, t, v, 0.45, 0.0)
    elif k == 'sn': hit(SN, t, v, 0.22, 0.15)
    elif k == 'roll': hit(ROLL, t, v, 0.2, 0.15)
    elif k == 'crash': hit(CRASH, t, v, 0.28, -0.25)
    elif k == 'timp': hit(TIMP, t, v, 0.45, -0.1)
    elif k == 'gate': hit(SLAP, t - 0.02, v, 0.35, 0.3); hit(RAT, t - 0.4, 80, 0.18, 0.3)
L = int(1.8 * SR); tt = np.arange(L) / SR
ir = np.stack([rng.normal(0, 1, L), rng.normal(0, 1, L)], 1) * np.exp(-tt / 0.35)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.6], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.2 * wet
fade = int(2.0 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
