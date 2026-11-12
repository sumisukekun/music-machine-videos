"""Swan Lake scene - acoustic render with VSCO-2 CE (CC0 recordings): glockenspiel + celesta-like bells for the struck keys,
oboe (the theme), harp, violas / violins (pad, tremolo), contrabass + cello pizzicato (bass buoys), horns, tuba, timpani, cymbal.
python3 render_audio.py /path/to/VSCO-2-CE -> music.wav"""
import sys, os, json, glob, re
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
I['oboe'] = Inst('Woodwinds/Oboe/Vib'); I['harp'] = Inst('Strings/Harp', offset=0, rx=r'_([A-G]#?)(\d)_')
I['vla'] = Inst('Strings/Viola Section/susvib'); I['vtrem'] = Inst('Strings/Viola Section/trem')
I['vln'] = Inst('Strings/Violin Section/susVib'); I['ctrem'] = Inst('Strings/Cello Section/trem')
I['cpizz'] = Inst('Strings/Cello Section/pizzT'); I['hnsus'] = Inst('Brass/F Horn/sus'); I['tubasus'] = Inst('Brass/Tuba/sus')
TIMP = [Hit(sorted(glob.glob(P(f'Percussion/Timpani/Timpani{k}_Hit_*')))) for k in range(1, 6)]
for t, d, role, p, v in json.load(open('notes.json')):
    if role == 'plate': play('glock', t, p, v, 0.20, 0.0, dur=0.9, rel=0.5); play('harp', t, p, v, 0.30, 0.1)
    elif role == 'harp': play('harp', t, p, v, 0.45, -0.15)
    elif role == 'key': play('glock', t, p, v, 0.22, -0.2, dur=1.2, rel=0.6); play('mar', t, p, v, 0.18, -0.2)
    elif role == 'key2': play('glock', t, p, v, 0.16, 0.25, dur=1.2, rel=0.6); play('mar', t, p, v, 0.14, 0.25)
    elif role == 'oboe': play('oboe', t, p, v, 0.55, 0.05, dur=d, rel=0.25)
    elif role == 'pad': play('vla', t, p, v, 0.11, 0.25, dur=d, rel=0.5)
    elif role == 'trem': play('vtrem', t, p, v, 0.10, 0.25, dur=d, rel=0.4)
    elif role == 'vln': play('vln', t, p, v, 0.16, 0.35, dur=d, rel=0.5)
    elif role == 'pizz': play('cb', t, p, v, 0.55, -0.05)
    elif role == 'cello': play('cpizz', t, p, v, 0.30, -0.25)
    elif role == 'horn': play('hnsus', t, p, v, 0.30, -0.3, dur=d, rel=0.35)
    elif role == 'hornsus': play('hnsus', t, p, v, 0.22, -0.3, dur=d, rel=0.8)
    elif role == 'tuba': play('tubasus', t, p, v, 0.30, 0.0, dur=d, rel=0.5)
    elif role == 'vsus': play('vln', t, p, v, 0.14, 0.3, dur=d, rel=0.9)
    elif role == 'timp': hit(TIMP[1], t, v, 0.40, -0.05)
    elif role == 'crash': hit(CRASH, t, v, 0.20, 0.25)
# the machine's own sounds (times from layout.json): moon cradle creak, gates, wheel latch, nest landings
LAYF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'layout.json')
if os.path.exists(LAYF):
    for kind, t in json.load(open(LAYF))['sfx']:
        if kind == 'gate': hit(WOOD, t, 85, 0.16, 0.2)
        elif kind == 'nest': hit(LOGLO, t, 60, 0.12, -0.1)
        elif kind == 'latch': hit(CLAV, t, 70, 0.10, 0.1)
        elif kind == 'tip': hit(RAT, t, 60, 0.10, -0.3)
        elif kind == 'chime': hit(BT, t, 70, 0.16, 0.3)
L = int(3.0 * SR); tt = np.arange(L) / SR
ir = np.stack([rng.normal(0, 1, L), rng.normal(0, 1, L)], 1) * np.exp(-tt / 0.9)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.55], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.30 * wet
fade = int(2.5 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
