"""Carol of the Bells - acoustic render with VSCO-2 CE (CC0): glockenspiel, violin & cello sections, piano, timpani, sleigh bells."""
import sys, os, json, math, glob
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
I['vspic'] = Inst('Strings/Violin Section/Spic'); I['vsus'] = Inst('Strings/Violin Section/susVib')
I['csus'] = Inst('Strings/Cello Section/susvib'); I['cspic'] = Inst('Strings/Cello Section/spic')
I['cpizz'] = Inst('Strings/Cello Section/pizzT')
TIMP = [Hit(sorted(glob.glob(P(f'Percussion/Timpani/Timpani{k}_Hit_*')))) for k in range(1, 6)]
TROLL = Hit(sorted(glob.glob(P('Percussion/Timpani/Rolls/*'))) or sorted(glob.glob(P('Percussion/Timpani/Timpani1_Hit_*'))))
for t, d, role, p, v in json.load(open('notes.json')):
    if role == 'glock': play('glock', t, p, v, 0.34, 0.12, dur=max(d, 0.6), rel=0.6)
    elif role == 'glock2': play('glock', t, p, v, 0.22, -0.2, dur=max(d, 0.6), rel=0.6)
    elif role == 'vspic': play('vspic', t, p, v, 0.32, 0.25)
    elif role == 'cspic': play('cspic', t, p, v, 0.3, -0.2)
    elif role == 'vsus': play('vsus', t, p, v, 0.13, 0.3, dur=d, rel=0.3)
    elif role == 'cello': play('csus', t, p, v, 0.38, -0.25, dur=d, rel=0.35)
    elif role == 'cpizz': play('cpizz', t, p, v, 0.45, -0.2)
    elif role == 'pbass': play('pno', t, p, v, 0.45, -0.1, dur=d, rel=0.5)
    elif role == 'pchord': play('pno', t, p, v, 0.25, 0.1, dur=d, rel=0.6)
    elif role == 'timp': hit(TIMP[{43: 1, 41: 1, 39: 0, 38: 0}.get(p, 0)], t, v, 0.5, -0.05)
    elif role == 'troll': hit(TROLL, t, v, 0.35, 0.0)
    elif role == 'sleigh': hit(SLEIGH, t, v, 0.16, 0.4)
    elif role == 'crash': hit(CRASH, t, v, 0.3, 0.2)
    elif role == 'bt': hit(BT, t, v, 0.3, -0.2)
L = int(2.5 * SR); tt = np.arange(L) / SR
ir = np.stack([rng.normal(0, 1, L), rng.normal(0, 1, L)], 1) * np.exp(-tt / 0.6)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.55], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.28 * wet
fade = int(2.0 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
