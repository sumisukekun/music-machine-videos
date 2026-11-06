"""William Tell galop - acoustic render with VSCO-2 CE (CC0): xylophone keys (what the train strikes),
trumpets, horns, trombone, tuba, violin & cello sections, contrabass, timpani, snare, bass drum, cymbal."""
import sys, os, json, math, glob
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
I['tpt'] = Inst('Brass/Trumpet/stac'); I['tptsus'] = Inst('Brass/Trumpet/sus')
I['hnst'] = Inst('Brass/F Horn/stac'); I['hnsus'] = Inst('Brass/F Horn/sus')
I['tbn'] = Inst('Brass/Tenor Trombone/stac'); I['tuba'] = Inst('Brass/Tuba/stac'); I['tubasus'] = Inst('Brass/Tuba/sus')
I['vspic'] = Inst('Strings/Violin Section/Spic'); I['vsus'] = Inst('Strings/Violin Section/susVib')
I['cspic'] = Inst('Strings/Cello Section/spic')
TIMP = [Hit(sorted(glob.glob(P(f'Percussion/Timpani/Timpani{k}_Hit_*')))) for k in range(1, 6)]
SNARE = Hit(sorted(glob.glob(P('Percussion/Snare2-HitNS_*')), key=lambda f: int(re.search(r'_v(\d)', f).group(1))))
for t, d, role, p, v in json.load(open('notes.json')):
    if role == 'xylo': play('xylo', t, p, v, 0.42, 0.15); play('glock', t, p, v, 0.07, 0.25, dur=0.5, rel=0.3)
    elif role == 'glock': play('glock', t, p, v, 0.12, 0.3, dur=0.6, rel=0.4)
    elif role == 'tpt': play('tpt', t, p, v, 0.30, -0.1)
    elif role == 'tptsus': play('tptsus', t, p, v, 0.22, -0.1, dur=d, rel=0.5)
    elif role == 'hn': play('hnst', t, p, v, 0.25, -0.3)
    elif role == 'hnst': play('hnst', t, p, v, 0.13, -0.3)
    elif role == 'hnsus': play('hnsus', t, p, v, 0.18, -0.3, dur=d, rel=0.5)
    elif role == 'tbn': play('tbn', t, p, v, 0.22, 0.2)
    elif role == 'tuba': play('tuba', t, p, v, 0.42, 0.0)
    elif role == 'tubasus': play('tubasus', t, p, v, 0.35, 0.0, dur=d, rel=0.6)
    elif role == 'vspic': play('vspic', t, p, v, 0.24, 0.3)
    elif role == 'vsus': play('vsus', t, p, v, 0.12, 0.3, dur=d, rel=0.5)
    elif role == 'cspic': play('cspic', t, p, v, 0.22, -0.2)
    elif role == 'cb': play('cb', t, p, v, 0.35, -0.05)
    elif role == 'timp': hit(TIMP[1 if p >= 43 else 0], t, v, 0.42, -0.05)
    elif role == 'snare': hit(SNARE, t, v, 0.16, 0.2)
    elif role == 'bd': hit(BD, t, v, 0.32, 0.0)
    elif role == 'crash': hit(CRASH, t, v, 0.26, 0.25)
    elif role == 'bt': hit(BT, t, v, 0.25, -0.2)
# the railway's own sounds: start whistle-bell and the buffer stop
hit(WOOD, 0.25, 90, 0.2, -0.3)
hit(SLAP, EV['final'] + 0.35, 100, 0.25, 0.3)
L = int(2.2 * SR); tt = np.arange(L) / SR
ir = np.stack([rng.normal(0, 1, L), rng.normal(0, 1, L)], 1) * np.exp(-tt / 0.5)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.6], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.22 * wet
fade = int(2.0 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
