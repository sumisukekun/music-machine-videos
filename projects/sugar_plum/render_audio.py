"""Dance of the Sugar Plum Fairy - acoustic render with VSCO-2 CE (CC0 samples only).
python3 render_audio.py /path/to/VSCO-2-CE -> music.wav
What you hear is what you see (every pitched sound = a marble striking a key / drum / disc / chime at that moment):
  glass keys (fairy marble)        celesta = glockenspiel + soft high piano
  sugar pearls on crystal keys     celesta (cadenza and Presto)  = glockenspiel + piano, a touch of harp
  peppermint chord discs (mint)    pizzicato chords = viola / cello section pizzicato + soft harp
  chocolate drums (cocoa)          pizzicato bass = cello section + solo contrabass pizzicato; the runs = marimba + harp
  the machine                      sugar bowl latch, pearls dropping into the bowls, the lift, the bell at the top"""
import sys, os, json, glob, re
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
I['vcp'] = Inst('Strings/Cello Section/pizzT'); I['vlap'] = Inst('Strings/Viola Section/pizz'); I['harp'] = Inst('Strings/Harp', offset=0)
I['vcsus'] = Inst('Strings/Cello Section/susvib'); I['vsus'] = Inst('Strings/Violin Section/susVib')
TRI = Hit(sorted(glob.glob(P('Percussion/Triangle3-Hit_*'))))
LAY = json.load(open('layout.json')) if os.path.exists('layout.json') else dict(sfx=[])
for t, d, role, p, v in json.load(open('notes.json')):
    if role in ('fairy', 'rain'):
        g = 1.0 if role == 'fairy' else 0.75
        play('glock', t, p, v, 0.20 * g, 0.15 if role == 'fairy' else -0.15, dur=1.2, rel=0.6)
        play('pno', t, p, v, 0.16 * g, 0.1, dur=min(d, 0.6), rel=0.35)
        if role == 'rain': play('harp', t, p, v, 0.05, -0.3)
        if d > 2.0: play('glock', t, p - 12, v, 0.10, 0.0, dur=2.5, rel=1.2)
    elif role == 'mint':
        play('vlap' if p >= 48 else 'vcp', t, p, v, 0.22, 0.35)
        play('harp', t, p, v, 0.06, 0.3)
        if d > 2.0: play('vsus', t, p, v, 0.07, 0.3, dur=d, rel=1.0)
    elif role == 'cocoa':
        if p < 40: play('cb', t, p, v, 0.42, -0.1)
        else: play('vcp', t, p, v, 0.32, -0.2)
        if d > 2.0: play('vcsus', t, p, v, 0.08, -0.2, dur=d, rel=1.0)
    elif role == 'cocoa_run':
        play('mar', t, p, v, 0.34, -0.25); play('vcp', t, p, v, 0.18, -0.2); play('harp', t, p, v, 0.06, -0.2)


def noise_swell(t, dur, gain, lp=0.97, attack=0.15, pan=0.0):
    n = int(dur * SR); x = rng.normal(0, 1, (n, 2)).astype('float32')
    for c in range(2): x[:, c] = lfilter([1 - lp], [1, -lp], x[:, c])
    env = np.minimum(1, np.arange(n) / (attack * SR)) * np.exp(-np.arange(n) / (0.45 * dur * SR))
    x *= (env * gain / (np.abs(x).max() + 1e-9))[:, None]
    i = int(t * SR); e = min(N, i + n); out[i:e] += x[:e - i] * np.array([1 - 0.3 * pan, 1 + 0.3 * pan], 'float32')


for kind, t in LAY['sfx']:
    if kind == 'release': hit(WOOD, t, 70, 0.10, 0.2)
    elif kind == 'latch': hit(CLAV, t, 85, 0.16, -0.2); hit(WOOD, t + 0.01, 70, 0.1, -0.2)
    elif kind == 'bowl_tip': noise_swell(t, 0.9, 0.05, 0.93, 0.2, -0.2)
    elif kind == 'cup': hit(WOOD, t, 55, 0.07, 0.0)
    elif kind == 'cup_open': hit(WOOD, t, 45, 0.05, 0.1)
    elif kind == 'pearl_bowl': hit(TRI, t, 25, 0.012, 0.3)
    elif kind == 'lift_start': hit(RAT, t, 60, 0.10, 0.3)
    elif kind == 'lift_stop': hit(CLAV, t, 90, 0.14, 0.3)
    elif kind == 'gate': hit(WOOD, t, 75, 0.1, -0.1)
fin = json.load(open('events.json'))['final']
hit(BT, fin + 0.03, 105, 0.30, -0.2); hit(SLEIGH, fin + 0.05, 95, 0.22, 0.3); hit(CRASH, fin, 70, 0.10, 0.0)
# candy-castle hall
L_ = int(2.0 * SR); tt = np.arange(L_) / SR
ir = np.stack([rng.normal(0, 1, L_), rng.normal(0, 1, L_)], 1) * np.exp(-tt / 0.45)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.6], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.24 * wet
fade = int(2.0 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-20dB:ratio=2.2:attack=8:release=140,loudnorm=I=-14:TP=-1.3:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
