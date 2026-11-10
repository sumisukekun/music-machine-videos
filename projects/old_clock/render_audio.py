"""Grandfather's Clock - audio: every melody note = the brass chime bar a marble lands on (music-box / celesta tone with
a glockenspiel shimmer, VSCO-2 CE glock samples are CC0), bass = the clock's chime rods struck by hammers (upright piano
VSCO-2 CE + struck-tube partials), the hammer's wooden 'tick', the little clock's music box (comb tines), and the
mechanism's own sounds from layout.json (hatch latch, cups, trapdoors, lever, weights, floor, ...).
python3 render_audio.py [/path/to/VSCO-2-CE] -> music.wav"""
import sys, os, re, json, glob, math
import numpy as np, soundfile as sf
from scipy.signal import fftconvolve, lfilter
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/samples/vsco'
SR = 44100
NOTES = json.load(open(f'{HERE}/notes.json')); EV = json.load(open(f'{HERE}/events.json')); LAY = json.load(open(f'{HERE}/layout.json'))
DUR = EV['dur']; N = int(SR * DUR); out = np.zeros((N, 2), 'float32')
rng = np.random.default_rng(11)
PC = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}


def load(fn):
    x, sr = sf.read(fn, dtype='float32')
    if x.ndim == 1: x = np.stack([x, x], 1)
    x = x[:, :2]
    if sr != SR:
        n = int(len(x) * SR / sr); t = np.arange(n) * sr / SR
        x = np.stack([np.interp(t, np.arange(len(x)), x[:, c]) for c in range(2)], 1).astype('float32')
    a = np.abs(x).max(1); on = int(np.argmax(a > a.max() * 0.06)); return x[max(0, on - 100):]


def resample(x, ratio):
    n = int(len(x) / ratio); t = np.arange(n) * ratio
    return np.stack([np.interp(t, np.arange(len(x)), x[:, c]) for c in range(2)], 1).astype('float32')


GLOCK = {}
for f in glob.glob(os.path.join(ROOT, 'Percussion/Glock/*.wav')):
    m = re.search(r'_([A-G]#?)(\d)\.wav', os.path.basename(f)); p = 12 * (int(m.group(2)) + 1) + PC[m.group(1)] + 12
    GLOCK[p] = load(f)
GK = np.array(sorted(GLOCK))
PNO = {}
for f in glob.glob(os.path.join(ROOT, 'Keys/Upright Piano/*.wav')):
    m = re.search(r'dyn(\d)_rr1_(\d+)', os.path.basename(f))
    if not m: continue
    p = 21 + 2 * int(m.group(2)); v = int(m.group(1))
    PNO.setdefault(p, {})[v] = f
PK = np.array(sorted(PNO))
_rs = {}


def glock(p):
    k = int(GK[np.argmin(np.abs(GK - p))])
    if ('g', p) not in _rs: _rs[('g', p)] = resample(GLOCK[k], 2 ** ((p - k) / 12))
    return _rs[('g', p)]


def piano(p, dyn=3):
    k = int(PK[np.argmin(np.abs(PK - p))]); vs = sorted(PNO[k]); v = min(vs, key=lambda x: abs(x - dyn))
    key = ('p', p, v)
    if key not in _rs: _rs[key] = resample(load(PNO[k][v]), 2 ** ((p - k) / 12))
    return _rs[key]


def add(sig, t, g, pan=0.0):
    i = int(round(t * SR)); j = min(N, i + len(sig))
    if i >= N or j <= i or i < 0: return
    a = (pan + 1) * math.pi / 4
    if sig.ndim == 1: sig = np.stack([sig, sig], 1)
    out[i:j, 0] += sig[:j - i, 0] * g * math.cos(a) * 1.414; out[i:j, 1] += sig[:j - i, 1] * g * math.sin(a) * 1.414


def hz(p): return 440.0 * 2 ** ((p - 69) / 12)


def partials(f, dur, parts, click=0.1, att=0.0015):
    t = np.arange(int(dur * SR)) / SR; y = np.zeros_like(t)
    for r_, a, tau in parts:
        if f * r_ > 18000: continue
        y += a * np.sin(2 * np.pi * f * r_ * t + rng.uniform(0, 6)) * np.exp(-t / tau)
    y *= np.minimum(t / att, 1)
    n = int(0.005 * SR); y[:n] += click * rng.normal(0, 1, n) * np.exp(-np.arange(n) / (0.0012 * SR))
    return y


def chime(p):
    """brass chime bar: celesta-like, warm"""
    f = hz(p); k = (f / 523.0) ** -0.3
    return partials(f, 2.8, [(1, 1.0, 1.4 * k), (2, 0.22, 0.5 * k), (3, 0.09, 0.25 * k), (4.07, 0.05, 0.12 * k), (0.5, 0.04, 0.8)], 0.05)


def tine(p):
    """music-box comb tine (little clock)"""
    f = hz(p)
    return partials(f, 2.2, [(1, 1.0, 0.9), (2.0, 0.12, 0.25), (3.0, 0.06, 0.12), (5.4, 0.05, 0.05)], 0.08, att=0.0008)


def tube(p):
    """struck chime rod (bass): near-harmonic partials, long ring"""
    f = hz(p)
    return partials(f, 4.0, [(1, 1.0, 2.2), (2.0, 0.5, 1.4), (3.0, 0.25, 0.8), (4.2, 0.1, 0.5), (5.4, 0.05, 0.3)], 0.05)


def thud(f0, dec, amp=1.0, noise=0.3, L=0.4):
    t = np.arange(int(L * SR)) / SR
    return amp * (np.sin(2 * np.pi * f0 * t * (1 - 0.3 * t)) * np.exp(-t / dec) + noise * rng.normal(0, 1, len(t)) * np.exp(-t / (dec * 0.25)))


def click(f0=2500, dec=0.006, amp=1.0):
    t = np.arange(int(0.06 * SR)) / SR
    return amp * (np.sin(2 * np.pi * f0 * t) * np.exp(-t / dec) + 0.4 * rng.normal(0, 1, len(t)) * np.exp(-t / (dec * 0.5)))


def wood(f0=1100, dec=0.012, amp=1.0):
    """tick of a wooden hammer / escapement"""
    t = np.arange(int(0.09 * SR)) / SR
    y = np.sin(2 * np.pi * f0 * t) * np.exp(-t / dec) + 0.5 * np.sin(2 * np.pi * f0 * 2.6 * t) * np.exp(-t / (dec * 0.6))
    y += 0.3 * rng.normal(0, 1, len(t)) * np.exp(-t / 0.002)
    return amp * y


# ---------------------------------------------------------------- melody: one chime per landing (pitch per note)
FINAL = EV['final']
for t, d, role, p, v in NOTES:
    g = (v / 127) ** 1.2
    pan = float(np.clip((p - 76) / 30, -0.5, 0.5))
    if role == 'mel':
        add(chime(p), t, 0.42 * g, pan)
        add(glock(p + 12) if p + 12 <= GK.max() + 2 else glock(p), t, 0.10 * g, pan + 0.1)
        if t >= FINAL - 1e-6: add(tube(p - 12), t, 0.25, 0.0)          # the last note rings on
    elif role == 'tag':
        add(tine(p), t, 0.36 * g, -0.35)
        add(glock(p), t, 0.06 * g, -0.35)
    elif role == 'bass':
        add(piano(p, 2), t, 0.46 * g, -0.1 if p < 42 else 0.1)
        add(tube(p + 12), t, 0.10 * g, 0.0)
        add(wood(900 if (round((t - EV['tb1']) / 1.25)) % 2 == 0 else 1250, 0.012), t - 0.004, 0.22 * g, 0.0)   # tick / tock
# the escapement ticks before the music (the clock is running)
for t in (EV['tb1'] - 2.5, EV['tb1'] - 1.25):
    add(wood(1250 if t < EV['tb1'] - 2 else 900, 0.012), t, 0.3, 0.0)
# ---------------------------------------------------------------- the mechanism's own sounds
for kind, t in LAY['sfx']:
    if kind == 'latch': add(click(1700, 0.006, 0.7), t, 0.12, 0.2)
    elif kind == 'cup': add(thud(260, 0.035, 0.6, 0.4, 0.2), t, 0.10, -0.1)
    elif kind == 'wcup': add(thud(150, 0.05, 0.7, 0.4), t, 0.16, 0.0)
    elif kind == 'tip': add(click(900, 0.012, 0.6), t, 0.10, 0.0)
    elif kind == 'pop':
        tt = np.arange(int(0.3 * SR)) / SR
        add(np.sin(2 * np.pi * (300 + 700 * np.exp(-tt / 0.04)) * tt) * np.exp(-tt / 0.07), t, 0.18, 0.0)
    elif kind == 'seesaw': add(thud(110, 0.06, 0.9, 0.3), t, 0.25, 0.0); add(wood(700, 0.02), t + 0.07, 0.25, 0.0)
    elif kind == 'launch':
        tt = np.arange(int(0.35 * SR)) / SR
        add(np.sin(2 * np.pi * (220 + 600 * np.exp(-tt / 0.05)) * tt) * np.exp(-tt / 0.08), t + 0.02, 0.2, -0.3)
    elif kind == 'floor': add(wood(520, 0.03, 1.0), t, 0.30, -0.5); add(thud(180, 0.03, 0.5, 0.2, 0.15), t, 0.15, -0.5)
    elif kind == 'wstop': add(thud(90, 0.08, 0.8, 0.3), t, 0.2, -0.3)
    elif kind == 'settle': add(click(1300, 0.01, 0.8), t, 0.14, -0.5)
# lever bars (unison / see-saw) also creak a little
# ---------------------------------------------------------------- the little clock: tick of its escapement at every swing
lt = LAY['ev']['little']
t = lt['t_arr'] + EV['q'] / 2; k = 0
while t < DUR - 0.2:
    add(wood(1500 if k % 2 == 0 else 1150, 0.008), t, 0.16, -0.4); t += EV['q']; k += 1
# ---------------------------------------------------------------- room
L_ = int(2.4 * SR); tt = np.arange(L_) / SR
ir = np.stack([rng.normal(0, 1, L_), rng.normal(0, 1, L_)], 1) * np.exp(-tt / 0.55)[:, None]
for c in range(2): ir[:, c] = lfilter([0.25], [1, -0.65], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.26 * wet
fade = int(1.8 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write(f'{HERE}/music_raw.wav', mix, SR)
os.system(f'ffmpeg -v error -y -i {HERE}/music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.2:attack=6:release=150,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 {HERE}/music.wav && rm {HERE}/music_raw.wav')
print('ok', DUR)
