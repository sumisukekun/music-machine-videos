"""Acoustic render (VSCO-2 CE / VSCO 1, CC0 recordings only).
python3 render_audio.py /path/to/VSCO-2-CE -> music.wav"""
import sys, os, re, json, glob, math, zipfile, io
import numpy as np, soundfile as sf
from scipy.signal import fftconvolve, lfilter

ROOT = sys.argv[1]; SR = 44100
EV = json.load(open('events.json')); DUR = EV['final'] + 6.2
PC = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
rng = np.random.default_rng(5); _cache = {}


def norm(x, sr):
    if x.ndim == 1: x = np.stack([x, x], 1)
    x = x[:, :2].astype('float32')
    if sr != SR:
        n = int(len(x) * SR / sr); t = np.arange(n) * sr / SR
        x = np.stack([np.interp(t, np.arange(len(x)), x[:, c]) for c in range(2)], 1).astype('float32')
    a = np.abs(x).max(1); on = int(np.argmax(a > a.max() * 0.06))
    return x[max(0, on - int(0.003 * SR)):]


def load(fn):
    if fn not in _cache:
        if '::' in fn:
            z, m = fn.split('::'); d = zipfile.ZipFile(z).read(m); x, sr = sf.read(io.BytesIO(d), dtype='float32')
        else: x, sr = sf.read(fn, dtype='float32')
        _cache[fn] = norm(x, sr)
    return _cache[fn]


class Inst:
    def __init__(self, folder, offset=12, rx=r'_([A-G]#?)(-?\d)[_.]'):
        self.s = {}
        for f in glob.glob(os.path.join(ROOT, folder, '*.wav')):
            b = os.path.basename(f)
            if rx == 'piano':
                m = re.search(r'dyn(\d)_rr\d_(\d+)', b); p = 21 + 2 * int(m.group(2)); v = int(m.group(1))
            else:
                m = re.search(rx, b)
                if not m: continue
                p = 12 * (int(m.group(2)) + 1) + PC[m.group(1)] + offset
                v = re.search(r'_v(\d)', b); v = int(v.group(1)) if v else 1
            self.s.setdefault(p, {}).setdefault(v, []).append(f)
        self.keys = np.array(sorted(self.s)); self.rr = 0
        assert len(self.keys), folder

    def get(self, p, vel):
        k = int(self.keys[np.argmin(np.abs(self.keys - p))])
        lay = sorted(self.s[k]); li = lay[min(len(lay) - 1, int(vel / 128 * len(lay)))]
        fs = self.s[k][li]; self.rr += 1
        return load(fs[self.rr % len(fs)]), 2 ** ((p - k) / 12)


class Hit:
    def __init__(self, files): self.f = list(files); self.rr = 0; assert self.f

    def get(self, vel, layered=True):
        n = len(self.f); i = min(n - 1, int(vel / 128 * n)); self.rr += 1
        return load(self.f[i] if not layered else self.f[(i + self.rr) % n if n > 1 and self.rr % 2 else i])


P = lambda *a: os.path.join(ROOT, *a)
I = {'mar': Inst('Percussion/Marimba', 0 + 12), 'glock': Inst('Percussion/Glock'), 'xylo': Inst('Percussion/Xylo'),
     'pno': Inst('Keys/Upright Piano', rx='piano'), 'cb': Inst('Strings/Solo Contrabass/Pizz')}
SZ = os.path.join(ROOT, 'VSCO 1 Percussion/varWood/Slapstick.zip')
SLAP = Hit([SZ + '::slapstick_quiet1.wav', SZ + '::slapstick2.wav', SZ + '::slapstick3.wav'])
CLAV = Hit(sorted(glob.glob(P('Percussion/Claves1_Hit_*'))))
LOGLO = Hit(sorted(glob.glob(P('Percussion/LogDrumLo_*'))))
BD = Hit(sorted(glob.glob(P('Percussion/BDrumNewhit_*')), key=lambda f: int(re.search(r'_v(\d)', f).group(1))))
TAP = Hit(sorted(glob.glob(P('Percussion/Tumba-HitN_*'))) + sorted(glob.glob(P('Percussion/Conga-HitN_*'))))
TAMB = Hit(sorted(glob.glob(P('Percussion/Tamb1-Hit_*'))))
SHK = Hit(sorted(glob.glob(P("VSCO 1 Percussion/varWood/Camo's Shaker/shake*.wav"))))
RAT = Hit(sorted(glob.glob(P('Percussion/Ratchet1-Crank_*'))))
BT = Hit(sorted(glob.glob(P('Percussion/BellTree_Stroke*'))))
CRASH = Hit(sorted(glob.glob(P('Percussion/cymbal-crash1_*'))))
SLEIGH = Hit(sorted(glob.glob(P('Percussion/Sleighbells_Hit_*'))))
WOOD = Hit(sorted(glob.glob(P('VSCO 1 Percussion/varWood/wood_click*'))))

N = int(SR * DUR); out = np.zeros((N, 2), 'float32')


def place(x, ratio, t, gain, pan, dur=None, rel=0.15):
    if abs(ratio - 1) > 1e-4:
        n = int(len(x) / ratio); x = np.stack([np.interp(np.arange(n) * ratio, np.arange(len(x)), x[:, c]) for c in range(2)], 1)
    else: x = x.copy()
    if dur is not None:
        L = int((dur + rel) * SR)
        if L < len(x): x = x[:L]; r = int(rel * SR); x[-r:] *= np.linspace(1, 0, r)[:, None]
    l, r = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
    x = x * gain * np.array([l * 1.414, r * 1.414], 'float32')
    i = max(0, int(t * SR)); e = min(N, i + len(x)); out[i:e] += x[:e - i]


def play(n, t, p, v, g, pan, dur=None, rel=0.15):
    x, ra = I[n].get(p, v); place(x, ra, t, g * (0.5 + 0.5 * v / 127), pan, dur, rel)


def hit(h, t, v, g, pan=0.0, ratio=1.0):
    place(h.get(v), ratio, t + rng.normal(0, 0.003), g * (0.55 + 0.45 * v / 127), pan)


for t, d, role, p, v in json.load(open('notes.json')):
    if role == 'mel': play('mar', t, p + 12, v, 0.75, -0.05); play('xylo', t, p + 12, v, 0.07, 0.25)
    elif role == 'glock': play('glock', t, p, v, 0.22, 0.2)
    elif role == 'bell': play('glock', t, p, v, 0.3, 0.3)
    elif role == 'pbass': play('pno', t, p, v, 0.55, -0.15, dur=d, rel=0.2)
    elif role == 'pchord': play('pno', t, p, v, 0.28, 0.15, dur=d, rel=0.2)
    elif role == 'cbass': play('cb', t, p, v, 0.7, 0.0)
    elif role == 'tamb': hit(TAMB, t, v, 0.22, 0.35)
    elif role == 'shaker': hit(SHK, t, v, 0.2, -0.3)

# the machine's own sounds: wind-up crank, then each gesture
hit(RAT, 0.15, 90, 0.35, -0.4); hit(RAT, 1.25, 90, 0.3, -0.4)
hit(BT, EV['secs'][0]['t'] - 1.0, 80, 0.25, 0.3)
for g in EV['gest']:
    t = g['t']; k = g['kind']
    if k in ('clap', 'all'):
        hit(SLAP, t, 115, 0.55, 0.0); hit(CLAV, t + 0.004, 100, 0.25, 0.1)
    if k in ('stomp', 'all'):
        hit(BD, t, 105, 0.6 if k == 'stomp' else 0.45, -0.05); hit(LOGLO, t + 0.003, 100, 0.4, -0.2 if g['n'] == 0 else 0.2)
    if k == 'tap':
        hit(TAP, t, 105, 0.6, -0.25 if g['n'] == 0 else 0.25)
    if k == 'final':
        hit(SLAP, t, 125, 0.6); hit(BD, t, 120, 0.6); hit(CRASH, t, 110, 0.35); hit(SLEIGH, t + 0.02, 100, 0.25, 0.3)
        hit(BT, t + 0.1, 100, 0.3, -0.3)
for s in EV['secs'][1:]: hit(WOOD, s['t'] - 0.15, 90, 0.18, 0.4)   # verse-change "click" of the cam barrel

# room
L = int(1.4 * SR); tt = np.arange(L) / SR
ir = np.stack([rng.normal(0, 1, L), rng.normal(0, 1, L)], 1) * np.exp(-tt / 0.28)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.7], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.16 * wet
fade = int(2.5 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
