"""Mozart Symphony No. 40 - acoustic render with VSCO-2 CE (CC0 samples only).  python3 render_audio.py /path/to/VSCO-2-CE -> music.wav
What you hear is what you see (every pitched sound is a ball striking a key / rail bar / bell at that moment):
  neon keys (A, B)        glockenspiel + violin section (spiccato, sustained for long notes) + soft piano;
                          violin II's octave below where it doubles; bars 14-16 B's keys = flute, bars 16-20 = oboes
  xylophone fire escape   xylophone + violins
  rail bars (train)       marimba + viola section
  bronze bells / tram     cello section + contrabass pizzicato an octave below + piano bass
  final chord             the bowl bell (both balls) + string / horn / timpani tutti; fireworks"""
import sys, os, json, glob, re
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
LAY = json.load(open('layout.json')); MEL = json.load(open('mel.json')); E = json.load(open('events.json'))
XY = set()
for n in ('A', 'B'):
    for h in LAY['balls'][n]['hits']:
        if h['kind'].startswith('xylo'): XY.add(round(h['t'], 4))
I['vln'] = Inst('Strings/Violin Section/susVib'); I['vlnsp'] = Inst('Strings/Violin Section/Spic')
I['vla'] = Inst('Strings/Viola Section/spic'); I['vlasus'] = Inst('Strings/Viola Section/susvib')
I['vcsus'] = Inst('Strings/Cello Section/susvib'); I['vcsp'] = Inst('Strings/Cello Section/spic')
I['fl'] = Inst('Woodwinds/Flute/susvib'); I['ob'] = Inst('Woodwinds/Oboe/Sus')
TIMP = Hit(sorted(glob.glob(P('Percussion/Timpani/Timpani1_Hit_*'))))
Q = E['q']
for t, d, role, p, v in json.load(open('notes.json')):
    if role == 'mel':
        if round(t, 4) in XY: play('xylo', t, p, v, 0.2, -0.1, dur=0.5, rel=0.3)
        else: play('glock', t, p, v, 0.18, -0.1, dur=1.0, rel=0.5)
        play('pno', t, p, v, 0.1, 0.0, dur=d, rel=0.25)
        if d > 1.6 * Q: play('vln', t, p, v, 0.24, -0.2, dur=d, rel=0.25)
        else: play('vlnsp', t, p, v, 0.28, -0.2)
    elif role == 'mel2':
        if d > 1.6 * Q: play('vln', t, p, v, 0.12, 0.2, dur=d, rel=0.25)
        else: play('vlnsp', t, p, v, 0.14, 0.2)
    elif role == 'mel_fl':
        play('glock', t, p, v, 0.16, 0.25, dur=1.0, rel=0.5); play('fl', t, p, v, 0.3, 0.3, dur=max(d, 0.25), rel=0.2)
    elif role == 'mel_ob':
        play('glock', t, p, v, 0.12, 0.2, dur=1.0, rel=0.5); play('ob', t, p, v, 0.2, 0.25, dur=d, rel=0.2)
    elif role == 'acc':
        play('mar', t, p + 12, v, 0.18, 0.35)
        if d > 1.6 * Q: play('vlasus', t, p, v, 0.2, 0.3, dur=d, rel=0.2)
        else: play('vla', t, p, v, 0.22, 0.3)
    elif role == 'bass':
        play('pno', t, p, v, 0.16, -0.1, dur=min(d, 0.9), rel=0.3)
        if d > 1.6 * Q: play('vcsus', t, p, v, 0.26, -0.25, dur=d, rel=0.3)
        else: play('vcsp', t, p, v, 0.3, -0.25)
    elif role == 'cb': play('cb', t, p, v, 0.26, -0.05)
    elif role == 'tutti':
        if p < 40: play('cb', t, p, v, 0.3, 0.0); hit(TIMP, t, 120, 0.4, 0.0)
        elif p < 50: play('vcsus', t, p, v, 0.3, -0.2, dur=d, rel=1.0)
        elif p < 66: play('vlasus', t, p, v, 0.2, 0.1, dur=d, rel=1.0)
        else: play('vln', t, p, v, 0.18, 0.2, dur=d, rel=1.0)
        if p in (43, 55): play('pno', t, p, v, 0.22, 0.0, dur=d, rel=1.0)
        if p == 67: play('glock', t, p, v, 0.25, -0.2, dur=2.0, rel=1.0); play('glock', t, 70, v, 0.25, 0.2, dur=2.0, rel=1.0)

for kind, t in LAY['sfx']:
    if kind == 'release': hit(WOOD, t, 80, 0.15, 0.0)
    elif kind == 'train_go': hit(RAT, t, 70, 0.12, 0.4); hit(WOOD, t + 0.02, 80, 0.12, 0.4)
    elif kind == 'final':
        hit(CRASH, t, 110, 0.25, 0.2); hit(BT, t + 0.05, 110, 0.3, -0.25)
for t in json.load(open('pieces.json'))['fireworks']: hit(BD, t, 70, 0.12, rng.uniform(-0.6, 0.6)); hit(SHK, t + 0.05, 80, 0.08, 0.0)
# night city: a wide hall-like reverb
L_ = int(2.0 * SR); tt = np.arange(L_) / SR
ir = np.stack([rng.normal(0, 1, L_), rng.normal(0, 1, L_)], 1) * np.exp(-tt / 0.5)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.55], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.22 * wet
fade = int(2.5 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
