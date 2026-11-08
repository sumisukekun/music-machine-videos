"""Radetzky March - acoustic render with VSCO-2 CE / VSCO 1 (CC0 recordings only) + synthesized hand claps.
Glockenspiel = the keys the player marbles strike; violins, trumpets, horns, tuba, contrabass, timpani, snare, cymbal;
bass drum = the conductor landing on his drum podium; claps = the audience marbles landing in their seats.
python3 render_audio.py /path/to/VSCO-2-CE -> music.wav"""
import sys, os, json, math, glob
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
src = src.replace("EV = json.load(open('events.json')); DUR = EV['final'] + 4.5", "EV = json.load(open('events.json')); DUR = EV['final'] + 4.5")
exec(src)
from scipy.signal import butter, sosfilt
LAY = json.load(open('layout.json'))
I['tpt'] = Inst('Brass/Trumpet/stac'); I['tptsus'] = Inst('Brass/Trumpet/sus')
I['hnst'] = Inst('Brass/F Horn/stac'); I['hnsus'] = Inst('Brass/F Horn/sus')
I['tuba'] = Inst('Brass/Tuba/stac'); I['tubasus'] = Inst('Brass/Tuba/sus')
I['vspic'] = Inst('Strings/Violin Section/Spic'); I['vsus'] = Inst('Strings/Violin Section/susVib')
I['vaspic'] = Inst('Strings/Viola Section/spic'); I['cspic'] = Inst('Strings/Cello Section/spic')
TIMP = [Hit(sorted(glob.glob(P(f'Percussion/Timpani/Timpani{k}_Hit_*')))) for k in range(1, 6)]
SNARE = Hit(sorted(glob.glob(P('Percussion/Snare2-HitNS_*')), key=lambda f: int(re.search(r'_v(\d)', f).group(1))))
TRI = Hit(sorted(glob.glob(P('Percussion/Triangle3-Hit_*'))))
for t, d, role, p, v in json.load(open('notes.json')):
    if role == 'glk': play('glock', t, p, v, 0.30, 0.1, dur=max(0.35, d * 1.5), rel=0.35); play('xylo', t, p, v, 0.06, 0.1)
    elif role == 'vs':
        if d > 0.4: play('vsus', t, p, v, 0.16, 0.25, dur=d, rel=0.15)
        else: play('vspic', t, p, v, 0.24, 0.25)
    elif role == 'fl': play('glock', t, p, v, 0.07, 0.35, dur=0.4, rel=0.3)
    elif role == 'tpt': play('tpt', t, p, v, 0.22, -0.15)
    elif role == 'tptsus': play('tptsus', t, p, v, 0.2, -0.15, dur=d, rel=0.5)
    elif role == 'vc': play('cspic', t, p, v, 0.28, -0.2)
    elif role == 'hn': play('hnst', t, p, v, 0.12, -0.3); 
    elif role == 'va': play('vaspic', t, p, v, 0.08, 0.3)
    elif role == 'hnsus': play('hnsus', t, p, v, 0.18, -0.3, dur=d, rel=0.5)
    elif role == 'tuba': play('tuba', t, p, v, 0.40, 0.0)
    elif role == 'tubasus': play('tubasus', t, p, v, 0.35, 0.0, dur=d, rel=0.6)
    elif role == 'vsus': play('vsus', t, p, v, 0.14, 0.3, dur=d, rel=0.5)
    elif role == 'cb': play('cb', t, p, v, 0.32, -0.05)
    elif role == 'timp': hit(TIMP[1 if p >= 43 else 0], t, v, 0.38, -0.05)
    elif role == 'snare': hit(SNARE, t, v, 0.13, 0.2)
    elif role == 'crash': hit(CRASH, t, v, 0.22, 0.25)

# ---- conductor: bass drum on every landing (the drum podium)
hit(BD, EV['t0'], 125, 0.65, 0.0); hit(TIMP[0], EV['t0'] + 0.002, 120, 0.3)
for tl, tau in LAY['ev']['cond_hops']:
    hit(BD, tl, int(70 + 100 * tau), 0.25 + 0.5 * tau, 0.0)

# ---- hand claps (synthesized): one clap = three short band-passed noise bursts + a short room tail
sos = butter(2, [800, 3200], btype='band', fs=SR, output='sos')
def one_clap(seed):
    r = np.random.default_rng(seed); n = int(0.16 * SR); x = r.normal(0, 1, n); env = np.zeros(n); tt = np.arange(n) / SR
    for k, dt in enumerate((0.0, 0.008 + r.uniform(0, 0.003), 0.017 + r.uniform(0, 0.004))):
        env += np.where(tt >= dt, np.exp(-(tt - dt) / 0.0035), 0) * (0.7 if k < 2 else 1.0)
    env += np.where(tt >= 0.017, 0.25 * np.exp(-(tt - 0.017) / 0.045), 0)
    y = sosfilt(sos, x * env); y /= np.abs(y).max()
    return np.stack([y, y], 1).astype('float32')
CLAPS1 = [one_clap(s) for s in range(24)]
def crowd(seed, n=26):
    r = np.random.default_rng(100 + seed); L = int(0.2 * SR); out_ = np.zeros((L, 2), 'float32')
    for k in range(n):
        c = CLAPS1[r.integers(len(CLAPS1))]; o = int(abs(r.normal(0, 0.0025)) * SR); pan = r.uniform(-0.8, 0.8)
        g = np.array([math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)]) * 1.414 * r.uniform(0.6, 1.0)
        e = min(L, o + len(c)); out_[o:e] += c[:e - o] * g
    return out_ / n * 3.0
CROWD = [crowd(s) for s in range(8)]
fin = LAY['ev'].get('final', EV['final'])
for k, (t, n, tau) in enumerate(LAY['claps']):
    if t > EV['final'] + 0.05: continue
    loud = (tau - 0.2) / 0.35
    g = 0.16 + 0.5 * max(0.0, min(1.0, loud))
    if abs(t - EV['final']) < 1e-3: g = 0.6
    place(CROWD[k % 8], 1.0, t, g, 0.0)
# applause after the last chord: every marble's own landings
r_ = np.random.default_rng(9)
for name, hops in LAY['aud'].items():
    for tl, tau in hops:
        if tl > EV['final'] + 0.05:
            pan = (int(name.split('_')[1]) - 5) / 6 if '_' in name else 0.0
            place(CLAPS1[r_.integers(24)], 1.0, tl, 0.05 + 0.03 * r_.uniform(), pan)
# ---- the machines and the late marble
for kind, t in LAY['sfx']:
    if kind == 'step': hit(WOOD, t, 95, 0.32, -0.5)
    elif kind == 'cup': hit(WOOD, t, 60, 0.10, 0.4)
    elif kind == 'gate': hit(CLAV, t, 60, 0.05, 0.3)
    elif kind == 'lift_go': hit(RAT, t, 70, 0.07, 0.6)
    elif kind == 'ding': play('glock', t, 88, 110, 0.35, 0.6, dur=1.2, rel=0.6); play('glock', t + 0.004, 84, 90, 0.2, 0.6, dur=1.2, rel=0.6)
    elif kind == 'thud': hit(LOGLO, t, 70, 0.18, 0.5)
    elif kind == 'release': hit(WOOD, t, 90, 0.18, 0.4)
    elif kind == 'weight': hit(BD, t, 120, 0.45, 0.4); hit(LOGLO, t + 0.003, 120, 0.35, 0.4)
    elif kind == 'clink': hit(TRI, t, 90, 0.18, 0.3)
    elif kind == 'flap': hit(CLAV, t, 90, 0.15, 0.4)
# the late marble lands in its seat on the final chord: bell tree sparkle
hit(BT, EV['final'] + 0.02, 110, 0.3, 0.0); hit(SLEIGH, EV['final'] + 0.05, 100, 0.22, 0.3)
L = int(2.0 * SR); tt = np.arange(L) / SR
ir = np.stack([rng.normal(0, 1, L), rng.normal(0, 1, L)], 1) * np.exp(-tt / 0.55)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.6], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.24 * wet
fade = int(1.5 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
