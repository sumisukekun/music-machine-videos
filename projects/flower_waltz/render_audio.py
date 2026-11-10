"""Waltz of the Flowers - acoustic render with VSCO-2 CE (CC0 recordings): glockenspiel = the struck flower bells,
marimba = water drops on leaves, xylophone = frozen buds, horn section (melody), violins, violas, cello & contrabass
pizzicato (waltz bass), horn after-beats, timpani, triangle, cymbal + the machine's own little sounds.
python3 render_audio.py /path/to/VSCO-2-CE -> music.wav"""
import sys, os, json, glob, re
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
I['hn'] = Inst('Brass/F Horn/sus'); I['hnst'] = Inst('Brass/F Horn/stac'); I['tubasus'] = Inst('Brass/Tuba/sus')
I['vsus'] = Inst('Strings/Violin Section/susVib'); I['vspic'] = Inst('Strings/Violin Section/Spic')
I['vla'] = Inst('Strings/Viola Section/spic'); I['cpz'] = Inst('Strings/Cello Section/pizzT')
TIMP = [Hit(sorted(glob.glob(P(f'Percussion/Timpani/Timpani{k}_Hit_*')))) for k in range(1, 6)]
TRI = Hit(sorted(glob.glob(P('Percussion/Triangle3-Hit_*'))))
for t, d, role, p, v in json.load(open('notes.json')):
    if role == 'key': play('glock', t, p, v, 0.30, 0.25, dur=max(0.5, d), rel=0.5)
    elif role == 'leaf': play('mar', t, p, v, 0.55, -0.15)
    elif role == 'ice': play('xylo', t, p, v, 0.40, 0.2)
    elif role == 'hn': play('hn', t, p, v, 0.34, -0.25, dur=d, rel=0.35)
    elif role == 'hnst': play('hnst', t, p, v, 0.12, -0.3)
    elif role == 'hnsus': play('hn', t, p, v, 0.22, -0.3, dur=d, rel=0.6)
    elif role == 'vsus': play('vsus', t, p, v, 0.20, 0.3, dur=d, rel=0.35)
    elif role == 'vspic': play('vspic', t, p, v, 0.22, 0.3)
    elif role == 'vla': play('vla', t, p, v, 0.11, 0.15)
    elif role == 'cb': play('cb', t, p, v, 0.55, -0.05)
    elif role == 'cpz': play('cpz', t, p, v, 0.30, -0.15)
    elif role == 'cbsus': play('tubasus', t, p, v, 0.35, 0.0, dur=d, rel=0.8)
    elif role == 'timp': hit(TIMP[0], t, v, 0.35, -0.05, ratio=2 ** ((p - 40) / 12))
    elif role == 'crash': hit(CRASH, t, v, 0.22, 0.25)
    elif role == 'tri': hit(TRI, t, v, 0.12, 0.35)
# the garden machine's own sounds (times from layout.json)
for kind, t in json.load(open('layout.json'))['sfx']:
    if kind == 'latch': hit(BT, t, 80, 0.18, -0.2)
    elif kind == 'spring': hit(WOOD, t, 85, 0.16, 0.1)
    elif kind == 'tip': hit(CLAV, t, 70, 0.12, 0.2)
    elif kind == 'tick': hit(WOOD, t, 70, 0.12, 0.25)
    elif kind == 'splash': hit(SHK, t, 90, 0.2, 0.2)
    elif kind == 'gate': hit(CLAV, t, 80, 0.12, -0.1)
    elif kind == 'clunk': hit(LOGLO, t, 90, 0.22, 0.0)
    elif kind == 'ratchet': hit(RAT, t, 80, 0.14, 0.3)
L = int(2.4 * SR); tt = np.arange(L) / SR
ir = np.stack([rng.normal(0, 1, L), rng.normal(0, 1, L)], 1) * np.exp(-tt / 0.6)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.6], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.28 * wet
fade = int(2.0 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
