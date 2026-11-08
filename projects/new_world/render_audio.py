"""New World IV - acoustic render with VSCO-2 CE (CC0).  python3 render_audio.py /path/to/VSCO-2-CE -> music.wav
Struck sounds: xylophone + marimba (loco sleepers), marimba + soft glock (car bells), low marimba + timpani (red drum),
with orchestral doubling at the same instants (horns, trumpets, strings, tuba, basses).  Extra sounds, each tied to
something seen: gantry trapdoors opening, fireworks bursts.
import sys, os, json, glob, re
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
I['tpt'] = Inst('Brass/Trumpet/stac'); I['tptsus'] = Inst('Brass/Trumpet/sus')
I['hnst'] = Inst('Brass/F Horn/stac'); I['hnsus'] = Inst('Brass/F Horn/sus')
I['tuba'] = Inst('Brass/Tuba/stac'); I['tubasus'] = Inst('Brass/Tuba/sus')
I['vspic'] = Inst('Strings/Violin Section/Spic'); I['vsus'] = Inst('Strings/Violin Section/susVib')
I['vlaspic'] = Inst('Strings/Viola Section/spic')
I['cspic'] = Inst('Strings/Cello Section/spic'); I['csus'] = Inst('Strings/Cello Section/susvib')
TIMP = Hit(sorted(glob.glob(P('Percussion/Timpani/Timpani1_Hit_*'))))
TRI = Hit(sorted(glob.glob(P('Percussion/Triangle3-Hit_*'))))
for t, d, role, p, v in json.load(open('notes.json')):
    if role == 'loco': play('xylo', t, p, v, 0.30, 0.1); play('mar', t, p, v, 0.38, 0.0)
    elif role == 'car': play('mar', t, p, v, 0.42, -0.15); play('glock', t, p + 12, v, 0.05, 0.3, dur=0.4, rel=0.3)
    elif role == 'carbass': play('mar', t, p if p >= 41 else p + 12, v, 0.5, -0.05)
    elif role == 'tpt': play('tpt', t, p, v, 0.28, -0.1)
    elif role == 'tptsus': play('tptsus', t, p, v, 0.2, -0.1, dur=d, rel=0.6)
    elif role == 'hn': play('hnst', t, p, v, 0.32, -0.3)
    elif role == 'hnst': play('hnst', t, p, v, 0.12, -0.3)
    elif role == 'hnsus': play('hnsus', t, p, v, 0.16, -0.3, dur=d, rel=0.6)
    elif role == 'tuba': play('tuba', t, p, v, 0.36, 0.0)
    elif role == 'tubasus': play('tubasus', t, p, v, 0.3, 0.0, dur=d, rel=0.7)
    elif role == 'vspic': play('vspic', t, p, v, 0.22, 0.3)
    elif role == 'vlaspic': play('vlaspic', t, p, v, 0.12, 0.2)
    elif role == 'vsus': play('vsus', t, p, v, 0.16, 0.3, dur=d, rel=0.4)
    elif role == 'cspic': play('cspic', t, p, v, 0.3, -0.2)
    elif role == 'csus': play('csus', t, p, v, 0.22, -0.2, dur=d, rel=0.3)
    elif role == 'cb': play('cb', t, p, v, 0.32, -0.05)
    elif role == 'timp': place(TIMP.get(v), 2 ** (((p if p >= 40 else p + 12) - 41.5) / 12), t, 0.4 * (0.55 + 0.45 * v / 127), -0.05)
    elif role == 'crash': hit(CRASH, t, v, 0.24, 0.25)
LAY = json.load(open('layout.json'))
for kind, t in LAY['sfx']:
    if kind == 'gate': hit(WOOD, t, 90, 0.2, 0.3); hit(CLAV, t + 0.02, 70, 0.12, 0.3)
SS = json.load(open('sfx_scene.json'))      # written by scene.py -- check (times of the dust clouds / firework bursts)
# (the trestle wreckage lands far below the frame: no sound for it)
for t in SS['fw']: hit(TRI, t, 90, 0.16, 0.4); hit(BT, t + 0.03, 80, 0.12, -0.3)
L = int(2.2 * SR); tt = np.arange(L) / SR
ir = np.stack([rng.normal(0, 1, L), rng.normal(0, 1, L)], 1) * np.exp(-tt / 0.55)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.6], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.24 * wet
fade = int(1.5 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
