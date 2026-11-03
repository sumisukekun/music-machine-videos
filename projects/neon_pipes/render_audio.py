"""Orchestral render of notes.json with real recorded samples (VSCO-2 Community Edition, CC0).
Usage: python3 render_audio.py /path/to/VSCO-2-CE  -> music.wav"""
import sys, os, re, json, glob, math
import numpy as np, soundfile as sf
from scipy.signal import fftconvolve

ROOT = sys.argv[1]; SR = 44100; DUR = 66.0
PC = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
rng = np.random.default_rng(3)
_cache = {}


def load(fn):
    if fn not in _cache:
        x, sr = sf.read(fn, dtype='float32')
        if x.ndim == 1: x = np.stack([x, x], 1)
        x = x[:, :2]
        if sr != SR:
            n = int(len(x) * SR / sr); t = np.arange(n) * sr / SR
            x = np.stack([np.interp(t, np.arange(len(x)), x[:, c]) for c in range(2)], 1).astype('float32')
        a = np.abs(x).max(1); pk = a.max(); on = int(np.argmax(a > pk * 0.06))
        x = x[max(0, on - int(0.004 * SR)):]
        _cache[fn] = x
    return _cache[fn]


class Inst:
    """pitched sample set: files named ..._<Note><oct>_v<layer>..., sounding pitch = name + 12"""
    def __init__(self, folder, offset=12, pattern='*.wav'):
        self.s = {}
        for f in glob.glob(os.path.join(ROOT, folder, pattern)):
            m = re.search(r'_([A-G]#?)(-?\d)[_.]', os.path.basename(f))
            if not m: continue
            p = 12 * (int(m.group(2)) + 1) + PC[m.group(1)] + offset
            v = re.search(r'_v(\d)', os.path.basename(f)); v = int(v.group(1)) if v else 1
            self.s.setdefault(p, {}).setdefault(v, []).append(f)
        self.keys = np.array(sorted(self.s)); self.rr = 0
        assert len(self.keys), folder

    def get(self, p, vel):
        k = int(self.keys[np.argmin(np.abs(self.keys - p))])
        lay = sorted(self.s[k]); li = lay[min(len(lay) - 1, int(vel / 128 * len(lay)))]
        fs = self.s[k][li]; self.rr += 1
        return load(fs[self.rr % len(fs)]), 2 ** ((p - k) / 12)


class Hit:
    def __init__(self, files, pitch=None):
        self.f = sorted(files); self.rr = 0; self.pitch = pitch
        assert self.f, files

    def get(self, p, vel):
        n = len(self.f); grp = self.f
        # files sorted by velocity name: pick by velocity band
        i = min(n - 1, int(vel / 128 * n)); i = max(0, min(n - 1, i + (self.rr % 2) * (1 if i + 1 < n else -1) * 0)); self.rr += 1
        ratio = 1.0 if self.pitch is None or p is None else 2 ** ((p - self.pitch) / 12)
        return load(grp[i]), ratio


P = lambda *a: os.path.join(*a)
V = ROOT
I = {
    'xylo': Inst('Percussion/Xylo'),
    'glock': Inst('Percussion/Glock'),
    'vln_sus': Inst('Strings/Violin Section/susVib'),
    'vln': Inst('Strings/Violin Section/Spic'),
    'vla': Inst('Strings/Viola Section/spic'),
    'vc': Inst('Strings/Cello Section/pizzT'),
    'cb': Inst('Strings/Solo Contrabass/Pizz'),
    'hn_sus': Inst('Brass/F Horn/sus'),
    'tpt_sus': Inst('Brass/Trumpet/sus'),
    'tba_sus': Inst('Brass/Tuba/sus'),
}
def vsort(files):
    return sorted(files, key=lambda f: (int(re.search(r'_v(\d)', f).group(1)) if re.search(r'_v(\d)', f) else 0, f))
BD = Hit(vsort(glob.glob(P(V, 'Percussion/BDrumNewhit_*'))))
SN = Hit(vsort(glob.glob(P(V, 'Percussion/Snare2-HitNS_*'))))
CR = Hit(sorted(glob.glob(P(V, 'Percussion/cymbal-crash1_*')), key=lambda f: ['pp', 'mp', 'mf', 'ff'].index(re.search(r'crash1_(\w\w)', f).group(1))))
TB = Hit(vsort(glob.glob(P(V, 'Percussion/Tamb1-Hit_*'))))
TR = Hit(vsort(glob.glob(P(V, 'Percussion/Triangle3-Hit_*'))))
SROLL = Hit(vsort(glob.glob(P(V, 'Percussion/Snare2-rollNS_*'))))
TIMP = {k: Hit(vsort(glob.glob(P(V, f'Percussion/Timpani/Timpani{k}_Hit_*'))), pitch=pp)
        for k, pp in ((1, 41.6), (2, 46.7), (3, 37.4), (4, 39.9), (5, 42.4))}

N = int(SR * DUR) + SR
bus = {k: np.zeros((N, 2), 'float32') for k in ('str', 'mallet', 'perc', 'synth')}
PAN = {'vln': -0.3, 'xylo': 0.15, 'glock': 0.1, 'vln_sus': -0.35, 'vla': 0.2, 'vc': 0.3, 'cb': 0.0, 'hn_sus': -0.2, 'tpt_sus': 0.1}
BUS = {'vln': 'str', 'xylo': 'mallet', 'glock': 'mallet', 'vln_sus': 'str', 'vla': 'str', 'vc': 'str', 'cb': 'str', 'hn_sus': 'str', 'tpt_sus': 'str'}
from scipy.signal import lfilter


def place(b, x, ratio, t, gain, pan, dur=None, rel=0.12):
    if abs(ratio - 1) > 1e-4:
        n = int(len(x) / ratio); src = np.arange(n) * ratio
        x = np.stack([np.interp(src, np.arange(len(x)), x[:, c]) for c in range(2)], 1)
    else: x = x.copy()
    if dur is not None:
        L = int((dur + rel) * SR)
        if L < len(x):
            x = x[:L]; r = int(rel * SR); x[-r:] *= np.linspace(1, 0, r)[:, None]
    l, r = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
    x = x * gain * np.array([l * 1.414, r * 1.414], 'float32')
    i = max(0, int(t * SR)); e = min(N, i + len(x)); bus[b][i:e] += x[:e - i]


def play(name, t, p, vel, gain, dur=None, rel=0.12):
    x, ratio = I[name].get(p, vel); place(BUS[name], x, ratio, t, gain * (0.55 + 0.45 * vel / 127), PAN[name], dur, rel)


def hit(h, t, vel, gain, pan=0.0, p=None):
    x, ratio = h.get(p, vel); place('perc', x, ratio, t, gain * (0.5 + 0.5 * vel / 127), pan)


_ks = {}
def pluck(p, dur, bright=0.5, decay=0.996, body=1.0):
    """Karplus-Strong plucked string (electric-guitar-ish), cached per pitch/length"""
    key = (p, round(dur, 2), bright, decay)
    if key in _ks: return _ks[key]
    f = 440 * 2 ** ((p - 69) / 12); n = int((dur + 0.6) * SR); L = SR / f; Li = int(L); fr = L - Li
    buf = rng.uniform(-1, 1, Li + 2); buf = lfilter([bright], [1, -(1 - bright)], buf)
    out = np.zeros(n); y = np.zeros(n + Li + 2); y[:Li + 2] = buf
    # vectorised in blocks of one period
    pos = Li + 2
    while pos < n + Li + 2:
        e = min(n + Li + 2, pos + Li)
        a = y[pos - Li: e - Li]; b_ = y[pos - Li - 1: e - Li - 1]
        y[pos:e] = decay * ((1 - fr) * (0.5 * a + 0.5 * b_) + fr * b_ if False else 0.5 * (a + b_))
        pos = e
    out = y[Li + 2: Li + 2 + n]
    ratio = f / (SR / (Li + 0.5))           # exact tuning: resample the integer-delay loop
    m = int(n / ratio); out = np.interp(np.arange(m) * ratio, np.arange(n), out); n = m
    env = np.ones(n); r0 = int(dur * SR); env[r0:] = np.exp(-np.arange(n - r0) / (0.06 * SR))
    out = out * env
    # pickup/drive colour
    out = np.tanh(out * 1.8 * body) * 0.7
    x = np.stack([out, out], 1).astype('float32'); _ks[key] = x
    return x


def chorus(x, depth=0.0025, rate=0.8):
    n = len(x); t = np.arange(n) / SR
    d = (depth * SR) * (1 + np.sin(2 * np.pi * rate * t)) / 2 + 0.006 * SR
    idx = np.clip(np.arange(n) - d, 0, n - 1)
    w = np.interp(idx, np.arange(n), x[:, 0])
    return np.stack([x[:, 0] * 0.8 + w * 0.5, x[:, 0] * 0.8 - w * 0.3], 1).astype('float32')


def synth_hat(t, vel, open_=False):
    n = int((0.45 if open_ else 0.07) * SR); tt = np.arange(n) / SR
    nz = rng.normal(0, 1, n); nz = lfilter([1, -1], [1, -0.2], nz)
    nz = nz * np.exp(-tt / (0.15 if open_ else 0.018))
    x = np.stack([nz, nz], 1).astype('float32') * 0.16 * (vel / 127)
    place('perc', x, 1, t, 1.0, 0.25)


def tom(t, f0, vel):
    n = int(0.6 * SR); tt = np.arange(n) / SR
    ph = 2 * np.pi * np.cumsum(f0 * (1 + 0.6 * np.exp(-tt / 0.03))) / SR
    s = np.sin(ph) * np.exp(-tt / 0.22) + rng.normal(0, 1, n) * np.exp(-tt / 0.01) * 0.2
    x = np.stack([s, s], 1).astype('float32') * 0.75 * (vel / 127)
    place('perc', x, 1, t, 1.0, {150: -0.3, 115: 0.05, 85: 0.35}[f0])


def kick808(t, vel):
    n = int(0.5 * SR); tt = np.arange(n) / SR
    ph = 2 * np.pi * np.cumsum(48 + 110 * np.exp(-tt / 0.035)) / SR
    s = np.sin(ph) * np.exp(-tt / 0.28)
    x = np.stack([s, s], 1).astype('float32') * 0.7 * (vel / 127)
    place('perc', x, 1, t, 1.0, 0)


def gtr(t, p, d, v, g, pan=0.0, bright=0.45):
    x = pluck(p, max(0.12, d * 0.95), bright)
    place('synth', chorus(x), 1, t, g * (0.55 + 0.45 * v / 127), pan)


notes = json.load(open('notes.json'))
# all acoustic: mallets, strings, real drums (VSCO-2 CE, CC0) - no synthesised sounds
TOMP = {'tom1': 53, 'tom2': 48, 'tom3': 43}
hr = 0
for t, d, role, p, v in notes:
    if role == 'mel':
        play('glock', t, p, v, 0.5); play('xylo', t, p, v, 0.28)
        if d >= 0.45: play('vln_sus', t, p, 96, 0.3, dur=d, rel=0.25)
        else: play('vln', t, p, v, 0.32)
    elif role == 'xmel':
        play('xylo', t, p, v, 0.4)
    elif role == 'xlead':
        play('xylo', t, p, v, 0.6); play('glock', t, p + 12, v, 0.1)
        play('vln_sus', t, p, 90, 0.25, dur=d, rel=0.3)
    elif role == 'arp':
        play('vln', t, p, v, 0.42)
    elif role == 'stab':
        play('xylo', t, p, v, 0.3); play('vla', t, p, v, 0.28)
    elif role == 'bass':
        play('cb', t, p, v, 0.95); play('vc', t, p + 12, v, 0.35)
    elif role == 'final_bass':
        play('cb', t, p, 124, 0.9); play('vc', t, p + 12, 120, 0.4)
    elif role == 'final_str':
        play('vln_sus', t, p, 110, 0.28, dur=3.2, rel=1.2); play('glock', t, p, 120, 0.25)
    elif role == 'final_sun':
        play('xylo', t, p, 120, 0.4); play('glock', t, p + 12, 120, 0.2)
    elif role == 'kick':
        hit(BD, t, v, 1.1)
    elif role == 'snare':
        hit(SN, t, v, 0.75, -0.05)
    elif role == 'hat':
        hit(TB, t, v * 0.85, 0.32, 0.3)
    elif role == 'ohat':
        hit(TR, t, v, 0.3, 0.35)
    elif role in TOMP:
        tp = TOMP[role]; k = min(TIMP, key=lambda k: abs(TIMP[k].pitch - tp)); hit(TIMP[k], t, v, 1.0, {'tom1': -0.3, 'tom2': 0.05, 'tom3': 0.35}[role], tp)
    elif role == 'crash':
        hit(CR, t, v, 0.6, 0.3)

L = int(1.6 * SR); tt = np.arange(L) / SR
ir = np.stack([rng.normal(0, 1, L), rng.normal(0, 1, L)], 1) * np.exp(-tt / 0.32)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.7], ir[:, c])
ir[: int(0.01 * SR)] = 0; ir /= np.sqrt((ir ** 2).sum(0))
WET = {'str': 0.3, 'mallet': 0.22, 'perc': 0.14, 'synth': 0.16}
mix = np.zeros((N, 2), 'float32')
for k, x in bus.items():
    wet = np.stack([fftconvolve(x[:, c], ir[:, c])[:N] for c in range(2)], 1)
    mix += x + WET[k] * wet * 2.2
mix = mix[: int(SR * DUR)]
mix /= np.abs(mix).max() * 1.05
sf.write('music_raw.wav', mix, SR)
print('ok')
