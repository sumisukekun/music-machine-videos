"""Carmen Prelude - acoustic render with VSCO-2 CE (CC0 samples only).  python3 render_audio.py /path/to/VSCO-2-CE -> music.wav
What you hear is what you see:
  xylophone keys (toreros)     xylophone + glockenspiel at the struck pitch(es); violins (and trumpets in the theme) play
                               the tune, horns add the rest of a printed chord at the same instant
  drums (bull)                 timpani + piano bass + contrabass / tuba at the bass pitch, bass-drum thump
  brass bells (spectators)     glockenspiel + piano + horns / viola spiccato at the chord pitches, a tambourine tap
  chime rails (crowd wave)     glockenspiel + violin spiccato at G#4 / G#5 (the tremolo), the two last bells
  mechanisms                   ramp gate, toril trapdoor, trebuchet cups, trigger, counterweight, fling, stop, landing"""
import sys, os, json, glob, re
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
LAY = json.load(open('layout.json'))
I['vln'] = Inst('Strings/Violin Section/Spic'); I['vlnsus'] = Inst('Strings/Violin Section/susVib'); I['vla'] = Inst('Strings/Viola Section/spic')
I['vcsp'] = Inst('Strings/Cello Section/spic'); I['vcsus'] = Inst('Strings/Cello Section/susvib')
I['tpt'] = Inst('Brass/Trumpet/stac'); I['tptsus'] = Inst('Brass/Trumpet/sus')
I['hnst'] = Inst('Brass/F Horn/stac'); I['hnsus'] = Inst('Brass/F Horn/sus'); I['tuba'] = Inst('Brass/Tuba/stac'); I['tubasus'] = Inst('Brass/Tuba/sus')
TIMP = [Hit(sorted(glob.glob(P(f'Percussion/Timpani/Timpani{k}_Hit_*')))) for k in range(1, 6)]
TRI = Hit(sorted(glob.glob(P('Percussion/Triangle3-Hit_*'))))
TAMBV = Hit(sorted(glob.glob(P('VSCO 1 Percussion/varWood/tambourine_[Dd]own*.wav'))))
SNARE = Hit(sorted(glob.glob(P('Percussion/Snare2-HitNS_*')), key=lambda f: int(re.search(r'_v(\d)', f).group(1))))
for t, d, role, p, v in json.load(open('notes.json')):
    if role == 'key':
        play('xylo', t, p, v, 0.34, 0.12); play('glock', t, p, v, 0.10, 0.2, dur=0.5, rel=0.3)
    elif role == 'vln': play('vln', t, p, v, 0.26, -0.15)
    elif role == 'vln8': play('vln', t, p, v, 0.12, -0.25)
    elif role == 'tpt': play('tpt', t, p, v, 0.20, -0.05)
    elif role == 'vla': play('vla', t, p, v, 0.18, 0.2)
    elif role == 'hnch': play('hnst', t, p, v, 0.10, -0.3)
    elif role == 'drum':
        k = 0 if p < 43 else (1 if p < 48 else 2)
        hit(TIMP[k], t, v, 0.30, -0.05); play('pno', t, p if p >= 33 else p + 12, v, 0.26, -0.1, dur=0.5, rel=0.25)
        hit(BD, t, v - 10, 0.16, 0.0)
    elif role == 'cb': play('cb', t, p, v, 0.30, -0.05)
    elif role == 'tuba': play('tuba', t, p, v, 0.20, 0.05)
    elif role == 'cello': play('vcsp', t, p, v, 0.14, -0.2)
    elif role == 'bell':
        play('glock', t, p + 24, v, 0.06, 0.3, dur=0.45, rel=0.25); play('pno', t, p + 12, v, 0.09, 0.25, dur=0.25, rel=0.12)
    elif role == 'hn': play('hnst', t, p, v, 0.10, -0.3)
    elif role == 'vlasp': play('vla', t, p, v, 0.08, 0.3)
    elif role == 'trem':
        play('vln', t, p, v, 0.16, 0.2); play('glock', t, p + 12, v, 0.05, 0.3, dur=0.2, rel=0.1)
    elif role == 'tubasus': play('tubasus', t, p, v, 0.30, 0.0, dur=d, rel=0.8); play('vcsus', t, p + 12, v, 0.22, -0.1, dur=d, rel=0.8)
    elif role == 'hnsus': play('hnsus', t, p, v, 0.16, -0.3, dur=d, rel=0.8)
    elif role == 'vsus': play('vlnsus', t, p, v, 0.14, 0.2, dur=d, rel=0.8); play('tptsus', t, p, v, 0.06, -0.1, dur=d, rel=0.6)
# spectators' bells: a tambourine tap with every landing (the stands are festive)
for n, b in LAY['balls'].items():
    if n.endswith('lo'):
        for h in b['hits']:
            if h['kind'] == 'bell': hit(TAMBV, h['t'], 70, 0.07, 0.4)
# the bull's drums: a little more weight in the charge
for h in LAY['balls']['bull']['hits']:
    if EV['secs']['B3'] - 0.01 < h['t'] < EV['secs']['A3'] - 0.01: hit(SNARE, h['t'] + 0.003, 85, 0.07, 0.1)
for kind, t in LAY['sfx']:
    if kind == 'gate': hit(WOOD, max(0.02, t), 90, 0.2, -0.3); hit(BT, max(0.05, t + 0.05), 80, 0.18, 0.3)
    elif kind == 'bull_drop': hit(LOGLO, t, 80, 0.25, 0.2)
    elif kind == 'cups': hit(WOOD, t, 90, 0.2, 0.1)
    elif kind == 'trigger': hit(SLAP, t + 0.01, 110, 0.35, 0.0); hit(CRASH, t, 100, 0.2, 0.25)
    elif kind == 'release': hit(RAT, t, 100, 0.22, -0.2)
    elif kind == 'fling': hit(BT, t, 110, 0.3, 0.2); hit(TRI, t + 0.02, 110, 0.2, -0.2)
    elif kind == 'stop': hit(LOGLO, t, 110, 0.35, 0.0); hit(BD, t + 0.005, 100, 0.25, 0.0)
    elif kind == 'land': hit(CRASH, t, 120, 0.32, 0.15); hit(BD, t, 120, 0.3, 0.0)
    elif kind == 'seat': hit(WOOD, t, 60, 0.12, 0.4)
# final chord: crash, timpani roll into the final A, triangle, bell tree
FT = EV['final']
for k in range(12): hit(TIMP[0], FT - 0.85 + k * 0.07, 70 + 4 * k, 0.2, -0.05)
hit(CRASH, FT, 125, 0.35, 0.2); hit(BD, FT, 125, 0.4, 0.0); hit(TRI, FT + 0.01, 110, 0.25, -0.2); hit(BT, FT + 0.12, 110, 0.3, 0.3)
# hall
L_ = int(2.2 * SR); tt = np.arange(L_) / SR
ir = np.stack([rng.normal(0, 1, L_), rng.normal(0, 1, L_)], 1) * np.exp(-tt / 0.5)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.55], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.22 * wet
fade = int(2.5 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
