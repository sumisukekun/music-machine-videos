"""Mountain King - acoustic render with VSCO-2 CE (CC0): marimba / xylophone / glockenspiel lead (the marbles),
cello & contrabass pizzicato pedal, tuba, timpani + bass drum (stone hammers), triangle/bell-tree chimes."""
import sys, os, json, glob
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
I['cpizz'] = Inst('Strings/Cello Section/pizzT'); I['cspic'] = Inst('Strings/Cello Section/spic')
I['tuba'] = Inst('Brass/Tuba/stac'); I['tbn'] = Inst('Brass/Tenor Trombone/sus'); I['hn'] = Inst('Brass/F Horn/sus')
I['vsus'] = Inst('Strings/Violin Section/susVib'); I['csus'] = Inst('Strings/Cello Section/susvib')
TIMP = [Hit(sorted(glob.glob(P(f'Percussion/Timpani/Timpani{k}_Hit_*')))) for k in range(1, 6)]
TROLL = Hit(sorted(glob.glob(P('Percussion/Timpani/Rolls/*'))) or sorted(glob.glob(P('Percussion/Timpani/Timpani1_Hit_*'))))
TRI = Hit(sorted(glob.glob(P('Percussion/Triangle3-Hit_*'))))
for t, d, role, p, v in json.load(open('notes.json')):
    if role == 'mar': play('mar', t, p, v, 0.78, -0.12, dur=max(d, 0.35), rel=0.3)
    elif role == 'xylo': play('xylo', t, p, v, 0.36, 0.22)
    elif role == 'glock': play('glock', t, p, v, 0.2, 0.05, dur=max(d, 0.5), rel=0.5)
    elif role == 'cpizz': play('cpizz', t, p, v, 0.5, -0.25)
    elif role == 'cb': play('cb', t, p, v, 0.55, 0.0)
    elif role == 'tuba': play('tuba', t, p, v, 0.32, 0.1)
    elif role == 'cspic': play('cspic', t, p, v, 0.35, -0.2)
    elif role == 'bd': hit(BD, t, v, 0.62, 0.0)
    elif role == 'timp': hit(TIMP[0] if p == 47 else TIMP[4], t, v, 0.55, -0.1 if p == 47 else 0.15, ratio=2 ** (5.5 / 12) if p == 47 else 2 ** (-0.6 / 12))
    elif role == 'chime': hit(TRI, t, v, 0.16, 0.35); hit(BT, t + 0.01, v, 0.14, -0.3)
    elif role == 'crash': hit(CRASH, t, v, 0.3, 0.25)
    elif role == 'troll': hit(TROLL, t, v, 0.4, 0.0)
    elif role == 'final':
        if p < 50: play('csus', t, p, v, 0.4, -0.2, dur=d, rel=0.8); play('tbn', t, p + 12, v, 0.22, 0.1, dur=d, rel=0.8)
        elif p < 70: play('hn', t, p, v, 0.18, 0.15, dur=d, rel=0.8); play('pno', t, p, v, 0.3, 0.0, dur=d, rel=1.0)
        else: play('vsus', t, p, v, 0.14, 0.25, dur=d, rel=0.8); play('glock', t, p + 12, v, 0.12, 0.1, dur=d, rel=1.0)
        if p == 35: play('pno', t, p, v, 0.45, -0.1, dur=d, rel=1.0)
# cave reverb (longer, darker)
L = int(3.2 * SR); tt = np.arange(L) / SR
ir = np.stack([rng.normal(0, 1, L), rng.normal(0, 1, L)], 1) * np.exp(-tt / 0.9)[:, None]
for c in range(2): ir[:, c] = lfilter([0.25], [1, -0.72], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.3 * wet
fade = int(1.5 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
