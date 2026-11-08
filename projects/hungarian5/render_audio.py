"""Hungarian Dance No. 5 - acoustic render with VSCO-2 CE (CC0 recordings) + synthesized bells / whistle.
Thief strikes: xylophone (roof tiles, gutter brackets, deck slats) or bells (chimney bells, chord bells), doubled by the
violin section. Police strikes: bass marimba plates doubled by contrabass + cello pizzicato (+ tuba in the loud parts).
Off-beat chords (siren flashes): violas / cellos spiccato + horns.
python3 render_audio.py /path/to/VSCO-2-CE -> music.wav"""
import sys, os, json, math, glob
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
LAY = json.load(open('layout.json'))
I['vspic'] = Inst('Strings/Violin Section/Spic'); I['vsus'] = Inst('Strings/Violin Section/susVib')
I['vla'] = Inst('Strings/Viola Section/spic'); I['vcs'] = Inst('Strings/Cello Section/spic'); I['cpizz'] = Inst('Strings/Cello Section/pizzT')
I['tuba'] = Inst('Brass/Tuba/stac'); I['hnst'] = Inst('Brass/F Horn/stac')
CRASH2 = CRASH


def bell(t, midi, gain, decay=1.6, pan=0.2, big=False):
    """church-bell like partials (hum, prime, tierce, quint, nominal...) - synthesized"""
    f = 440 * 2 ** ((midi - 69) / 12)
    ratios = [(0.5, 0.5, 1.6), (1.0, 1.0, 1.0), (1.19, 0.45, 0.8), (1.5, 0.3, 0.7), (2.0, 0.55, 0.55), (2.52, 0.25, 0.4), (3.01, 0.18, 0.3), (4.1, 0.1, 0.2)]
    L_ = int((decay * 3.2) * SR); tt = np.arange(L_) / SR; x = np.zeros(L_)
    for r, a, dmul in ratios:
        if f * r > 16000: continue
        x += a * np.sin(2 * math.pi * f * r * tt + rng.uniform(0, 6.28)) * np.exp(-tt / (decay * dmul))
    att = np.minimum(1, tt / 0.002); x *= att
    if big: x += 0.6 * rng.normal(0, 1, L_) * np.exp(-tt / 0.03)          # the clang of the strike
    x = x / max(1e-9, np.abs(x).max()); x2 = np.stack([x, x], 1).astype('float32')
    place(x2, 1.0, t, gain, pan)


def whistle(t0, t1, gain):
    """police pea whistle: ~3 kHz tone, warbled by the pea (~28 Hz)"""
    L_ = int((t1 - t0) * SR); tt = np.arange(L_) / SR
    warble = 1 + 0.035 * np.sign(np.sin(2 * math.pi * 28 * tt)) * 0.5 + 0.02 * np.sin(2 * math.pi * 28 * tt)
    ph = 2 * math.pi * np.cumsum(2950 * warble) / SR
    x = np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.15 * rng.normal(0, 1, L_)
    env = np.minimum(1, tt / 0.02) * np.minimum(1, (tt[-1] - tt) / 0.05) * (0.75 + 0.25 * np.sin(2 * math.pi * 28 * tt))
    x = (x * env).astype('float32'); place(np.stack([x, x], 1), 1.0, t0, gain, -0.3)


for t, d, role, p, v in json.load(open('notes.json')):
    if role == 'xylo': play('xylo', t, p, v, 0.46, 0.12); play('glock', t, p, v, 0.04, 0.25, dur=0.4, rel=0.3)
    elif role == 'bell': bell(t, p, 0.2 * (0.6 + 0.4 * v / 127), decay=1.2, pan=0.15); play('glock', t, p, v, 0.18, 0.2, dur=1.0, rel=0.5)
    elif role == 'vspic': play('vspic', t, p, v, 0.21, 0.25)
    elif role == 'vsus': play('vsus', t, p, v, 0.13, 0.25, dur=d, rel=0.35)
    elif role == 'mar': play('mar', t, p, v, 0.42, -0.15)
    elif role == 'cb': play('cb', t, p, v, 0.42, -0.1)
    elif role == 'cpizz': play('cpizz', t, p, v, 0.22, -0.2)
    elif role == 'tuba': play('tuba', t, p, v, 0.2, 0.0)
    elif role == 'pah':
        if p >= 55: play('vla', t, p, v, 0.12, 0.3)
        else: play('vcs', t, p, v, 0.12, -0.3)
        play('hnst', t, p, v, 0.05, -0.35)
# section accents on the struck downbeats (coincide with a visible strike)
E_ = json.load(open('events.json'))
for s in E_['secs']:
    if s['name'] in ('A', 'TV', 'A3'): hit(CRASH, s['t'], 90, 0.14, 0.3)
# the chase's own sounds
whistle(0.1, 0.42, 0.16); whistle(0.5, 0.95, 0.18)
ev = LAY['ev']
hit(WOOD, ev['thief_start']['t_rel'], 90, 0.25, -0.1); hit(WOOD, ev['cop_start']['t_rel'], 80, 0.22, 0.2)
for kind, t in LAY['sfx']:
    if kind == 'creak': hit(RAT, t - 0.25, 60, 0.07, -0.2)
    elif kind == 'clonk': hit(CLAV, t, 120, 0.35, 0.0); hit(SLAP, t + 0.005, 90, 0.15, 0.0)
    elif kind == 'gong':
        bell(t, 43 + 12, 0.45, decay=3.2, pan=0.0, big=True); bell(t, 43, 0.3, decay=4.0, pan=0.0)
        hit(BD, t, 120, 0.4, 0.0); hit(CRASH, t + 0.02, 110, 0.22, 0.25); hit(BT, t + 0.15, 100, 0.22, -0.3)
L = int(2.2 * SR); tt = np.arange(L) / SR
ir = np.stack([rng.normal(0, 1, L), rng.normal(0, 1, L)], 1) * np.exp(-tt / 0.45)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.6], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.2 * wet
fade = int(2.0 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
