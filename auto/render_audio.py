"""Orchestral render of notes.json with real recorded samples (VSCO-2 Community Edition, CC0).
Usage: python3 render_audio.py VSCO_DIR PALETTE(orchestra|band) OUTDIR  -> OUTDIR/music_raw.wav"""
import sys, os, re, json, glob, math
import numpy as np, soundfile as sf
from scipy.signal import fftconvolve

ROOT = sys.argv[1]; PALETTE = sys.argv[2]; OUT = sys.argv[3]; SR = 44100; DUR = 66.0
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
    'vln': Inst('Strings/Violin Section/Spic'),
    'vln_sus': Inst('Strings/Violin Section/susVib'),
    'vla': Inst('Strings/Viola Section/spic'),
    'vc': Inst('Strings/Cello Section/pizzT'),
    'cb': Inst('Strings/Solo Contrabass/Pizz'),
    'hn': Inst('Brass/F Horn/stac'),
    'hn_sus': Inst('Brass/F Horn/sus'),
    'tpt': Inst('Brass/Trumpet/stac'),
    'tbn': Inst('Brass/Tenor Trombone/stac'),
    'tba': Inst('Brass/Tuba/stac'),
    'tpt_sus': Inst('Brass/Trumpet/sus'),
    'tbn_sus': Inst('Brass/Tenor Trombone/sus'),
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
bus = {k: np.zeros((N, 2), 'float32') for k in ('str', 'brass', 'mallet', 'perc')}
PAN = {'xylo': 0.0, 'glock': 0.1, 'vln': -0.4, 'vln_sus': -0.4, 'vla': 0.15, 'vc': 0.35, 'cb': 0.45, 'hn': -0.2, 'hn_sus': -0.2,
       'tpt': 0.05, 'tbn': 0.3, 'tba': 0.4, 'tpt_sus': 0.05, 'tbn_sus': 0.3, 'tba_sus': 0.4}
BUS = {'xylo': 'mallet', 'glock': 'mallet', 'vln': 'str', 'vln_sus': 'str', 'vla': 'str', 'vc': 'str', 'cb': 'str',
       'hn': 'brass', 'hn_sus': 'brass', 'tpt': 'brass', 'tbn': 'brass', 'tba': 'brass', 'tpt_sus': 'brass', 'tbn_sus': 'brass', 'tba_sus': 'brass'}


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
    i = int(t * SR) + int(rng.normal(0, 0.002) * SR * 0)   # exact on the hit
    i = max(0, i); e = min(N, i + len(x)); bus[b][i:e] += x[:e - i]


def play(name, t, p, vel, gain, dur=None, rel=0.12):
    x, ratio = I[name].get(p, vel); place(BUS[name], x, ratio, t, gain * (0.55 + 0.45 * vel / 127), PAN[name], dur, rel)


def hit(h, t, vel, gain, pan=0.0, p=None):
    x, ratio = h.get(p, vel); place('perc', x, ratio, t, gain, pan)




def cannon(t, gain):
    """synthesised naval-gun boom: low thump + filtered noise blast"""
    n = int(1.6 * SR); tt = np.arange(n) / SR
    thump = np.sin(2 * np.pi * (48 * np.exp(-tt / 0.25) + 28) * tt) * np.exp(-tt / 0.35)
    from scipy.signal import lfilter
    nz = rng.normal(0, 1, n); nz = lfilter([0.05], [1, -0.95], nz) * np.exp(-tt / 0.18) * 3
    crack = rng.normal(0, 1, n) * np.exp(-tt / 0.012) * 0.4
    x = (thump + nz + crack)[:, None] * np.array([[1, 1]], 'float32')
    place('perc', x.astype('float32'), 1.0, t, gain, 0.0)


def brass(name_s, name_l, t, p, v, g, dur, rel=0.15):
    if dur < 0.45: play(name_s, t, p, v, g)
    else: play(name_l, t, p, v, g * 0.85, dur=dur, rel=rel)


def cannon(t, gain):
    """synthesised naval-gun boom: low thump + filtered noise blast"""
    n = int(1.6 * SR); tt = np.arange(n) / SR
    thump = np.sin(2 * np.pi * (48 * np.exp(-tt / 0.25) + 28) * tt) * np.exp(-tt / 0.35)
    from scipy.signal import lfilter
    nz = rng.normal(0, 1, n); nz = lfilter([0.05], [1, -0.95], nz) * np.exp(-tt / 0.18) * 3
    crack = rng.normal(0, 1, n) * np.exp(-tt / 0.012) * 0.4
    x = (thump + nz + crack)[:, None] * np.array([[1, 1]], 'float32')
    place('perc', x.astype('float32'), 1.0, t, gain, 0.0)


def brass(name_s, name_l, t, p, v, g, dur, rel=0.15):
    if dur < 0.45: play(name_s, t, p, v, g)
    else: play(name_l, t, p, v, g * 0.85, dur=dur, rel=rel)



notes = json.load(open(os.path.join(OUT, 'notes.json')))
ORCH = PALETTE == 'orchestra'
for t, d, ins, p, v, role, b in notes:
    if role == 'mel':
        if ORCH:
            play('xylo', t, p, v, 0.5); q = p
            while q > 86: q -= 12
            brass('vln', 'vln_sus', t, q, v, 0.85, d)
            if t > 35: play('tpt', t, p - 12 if p > 82 else p, v, 0.35)
        else:
            q = p
            while q > 82: q -= 12
            brass('tpt', 'tpt_sus', t, q, v, 0.8, d); play('glock', t, p + 12, v, 0.2); brass('hn', 'hn_sus', t, q - 12, v, 0.3, d)
    elif role == 'gliss':
        play('xylo', t, p, v, 0.45); play('glock', t, p + 12, v, 0.25)
    elif role == 'harm':
        if ORCH: brass('vla', 'vln_sus', t, p, v, 0.65, d)
        else: brass('hn', 'hn_sus', t, p, v, 0.45, d)
    elif role == 'chord':
        if p < 57: play('tbn', t, p, v, 0.45)
        if p <= 65: play('hn', t, p, v, 0.5)
        else: play('tpt', t, p, v, 0.35); play('hn', t, p - 12, v, 0.3)
    elif role == 'bass':
        play('tba', t, p, v, 0.8)
        if ORCH: play('cb', t, p, v, 0.65)
        else: play('tbn', t, p + 12, v, 0.25)
    elif role == 'final':
        if p < 50: play('tba_sus', t, p, 120, 0.8, dur=2.6, rel=0.9); play('tbn_sus', t, p + 12, 120, 0.5, dur=2.6, rel=0.9)
        elif p < 75:
            play('hn_sus', t, p, 120, 0.6, dur=2.6, rel=0.9); play('tpt_sus', t, p, 115, 0.3, dur=2.6, rel=0.9)
            if ORCH: play('vln_sus', t, p + 12, 110, 0.5, dur=2.6, rel=0.9)
        else: play('tpt_sus', t, p if p <= 82 else p - 12, 120, 0.6, dur=2.6, rel=0.9); play('glock', t, p + 12, 120, 0.3)
    elif role == 'drum':
        if p == 36: hit(BD, t, v, 0.9)
        elif p == 38: hit(SN, t, v, 0.5)
        elif p in (49, 57):
            hit(CR, t, 127, 0.75, -0.3 if p == 49 else 0.3)
            if not ORCH: cannon(t, 0.5)
        elif p in (50, 47, 45):
            tp = {50: 53, 47: 48, 45: 41}[p]; k = min(TIMP, key=lambda k: abs(TIMP[k].pitch - tp))
            hit(TIMP[k], t, v, 1.1, 0.0, tp)
    elif role == 'hat':
        if ORCH: hit(TB, t, v * 0.8, 0.2 if v > 80 else 0.1, 0.2)
        else: hit(SN, t, v * 0.75, 0.28 if v > 80 else 0.14, 0.15)

# reverb: synthetic hall IR (decorrelated stereo exponential noise, darker tail)
L = int(2.0 * SR); tt = np.arange(L) / SR
ir = np.stack([rng.normal(0, 1, L), rng.normal(0, 1, L)], 1) * np.exp(-tt / 0.42)[:, None]
from scipy.signal import lfilter
for c in range(2):
    ir[:, c] = lfilter([0.25], [1, -0.75], ir[:, c])
ir[: int(0.012 * SR)] = 0; ir /= np.sqrt((ir ** 2).sum(0))
WET = {'str': 0.32, 'brass': 0.26, 'mallet': 0.22, 'perc': 0.2}
mix = np.zeros((N, 2), 'float32')
for k, x in bus.items():
    wet = np.stack([fftconvolve(x[:, c], ir[:, c])[:N] for c in range(2)], 1)
    mix += x + WET[k] * wet * 2.2
mix = mix[: int(SR * DUR)]
mix /= np.abs(mix).max() * 1.05
sf.write(os.path.join(OUT, 'music_raw.wav'), mix, SR)
print('ok', {k: len(v.s) if hasattr(v, 's') else 0 for k, v in I.items()})
