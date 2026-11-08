"""La Campanella for bells: glockenspiel (VSCO-2 CE samples, CC0) + celesta and tower-bell tones (additive synthesis),
the giant bell (hero's final strike) and the machine's own sounds (lift, gates, cups, launcher, fireworks).
python3 render_audio.py /path/to/VSCO-2-CE -> music.wav (reads notes.json, events.json, layout.json)"""
import sys, os, re, json, glob, math
import numpy as np, soundfile as sf
from scipy.signal import fftconvolve, lfilter
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/samples/vsco'
SR = 44100
NOTES = json.load(open(f'{HERE}/notes.json')); EV = json.load(open(f'{HERE}/events.json')); LAY = json.load(open(f'{HERE}/layout.json'))
DUR = EV['dur']; N = int(SR * DUR); out = np.zeros((N, 2), 'float32')
rng = np.random.default_rng(7)
PC = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}


def load(fn):
    x, sr = sf.read(fn, dtype='float32')
    if x.ndim == 1: x = np.stack([x, x], 1)
    x = x[:, :2]
    if sr != SR:
        n = int(len(x) * SR / sr); t = np.arange(n) * sr / SR
        x = np.stack([np.interp(t, np.arange(len(x)), x[:, c]) for c in range(2)], 1).astype('float32')
    a = np.abs(x).max(1); on = int(np.argmax(a > a.max() * 0.06)); return x[max(0, on - 100):]


GLOCK = {}
for f in glob.glob(os.path.join(ROOT, 'Percussion/Glock/*.wav')):
    m = re.search(r'_([A-G]#?)(\d)\.wav', os.path.basename(f)); p = 12 * (int(m.group(2)) + 1) + PC[m.group(1)] + 12
    GLOCK[p] = load(f)
GK = np.array(sorted(GLOCK))
_rs = {}


def resample(x, ratio):
    n = int(len(x) / ratio); t = np.arange(n) * ratio
    return np.stack([np.interp(t, np.arange(len(x)), x[:, c]) for c in range(2)], 1).astype('float32')


def glock(p):
    k = int(GK[np.argmin(np.abs(GK - p))])
    if (p, k) not in _rs: _rs[(p, k)] = resample(GLOCK[k], 2 ** ((p - k) / 12))
    return _rs[(p, k)]


def add(sig, t, g, pan=0.0):
    i = int(round(t * SR)); j = min(N, i + len(sig))
    if i >= N or j <= i: return
    a = (pan + 1) * math.pi / 4
    if sig.ndim == 1: sig = np.stack([sig, sig], 1)
    out[i:j, 0] += sig[:j - i, 0] * g * math.cos(a); out[i:j, 1] += sig[:j - i, 1] * g * math.sin(a)


def hz(p): return 440.0 * 2 ** ((p - 69) / 12)


def partials(f, dur, parts, click=0.1):
    t = np.arange(int(dur * SR)) / SR; y = np.zeros_like(t)
    for r_, a, tau in parts:
        if f * r_ > 18000: continue
        y += a * np.sin(2 * np.pi * f * r_ * t + rng.uniform(0, 6)) * np.exp(-t / tau)
    y *= np.minimum(t / 0.0015, 1)
    n = int(0.005 * SR); y[:n] += click * rng.normal(0, 1, n) * np.exp(-np.arange(n) / (0.0012 * SR))
    return y


def celesta(p):
    f = hz(p); k = (f / 523.0) ** -0.35
    return partials(f, 2.6, [(1, 1.0, 1.3 * k), (2, 0.28, 0.5 * k), (3, 0.10, 0.25 * k), (4.0, 0.05, 0.15 * k)], 0.06)


def towerbell(p):
    """struck tube bell: near-harmonic partials so the pitch reads clearly, long ring"""
    f = hz(p)
    return partials(f, 4.5, [(1, 1.0, 2.6), (2.0, 0.45, 1.6), (3.0, 0.22, 0.9), (4.16, 0.12, 0.6), (5.43, 0.06, 0.4), (0.5, 0.10, 3.0)], 0.12)


def giantbell(p):
    f = hz(p)     # minor-third tierce and fifth: hum G#3, prime G#4, tierce B4, quint D#5, nominal G#5 (= the chord)
    t = np.arange(int(7.0 * SR)) / SR
    parts = [(0.5, 0.55, 5.5), (1.0, 0.8, 4.0), (1.189, 0.5, 3.0), (1.498, 0.32, 2.4), (2.0, 0.65, 2.0), (2.52, 0.18, 1.0), (3.0, 0.12, 0.7)]
    y = sum(a * np.sin(2 * np.pi * f * r_ * t * (1 + 0.0006 * np.sin(2 * np.pi * 2.2 * t)) + rng.uniform(0, 6)) * np.exp(-t / d) for r_, a, d in parts)
    n = int(0.012 * SR); y[:n] += 0.5 * rng.normal(0, 1, n) * np.exp(-np.arange(n) / (0.002 * SR))
    return y * np.minimum(t / 0.002, 1)


def thud(f0, dec, amp=1.0, noise=0.3):
    t = np.arange(int(0.4 * SR)) / SR
    y = np.sin(2 * np.pi * f0 * t * (1 - 0.3 * t)) * np.exp(-t / dec) + noise * rng.normal(0, 1, len(t)) * np.exp(-t / (dec * 0.25))
    return amp * y


def click(f0=2500, dec=0.006, amp=1.0):
    t = np.arange(int(0.06 * SR)) / SR
    return amp * (np.sin(2 * np.pi * f0 * t) * np.exp(-t / dec) + 0.4 * rng.normal(0, 1, len(t)) * np.exp(-t / (dec * 0.5)))


# ---------------------------------------------------------------- the notes
hero_t = LAY['t']['hit']
for t, p, v, d, hand, sb, sg in NOTES:
    g = (v / 127) ** 1.3
    pan = float(np.clip((p - 66) / 40, -0.6, 0.6))
    if abs(t - hero_t) < 1e-4 and p == 68:
        add(giantbell(p), t, 0.62 * g + 0.25, 0.0); continue
    if p >= 75: add(glock(p), t, 0.55 * g, pan)
    if p >= 58: add(celesta(p), t, (0.20 if p >= 75 else 0.34) * g, pan)
    if p < 70: add(towerbell(p), t, (0.42 if p < 58 else 0.22) * g, pan * 0.5)
# ---------------------------------------------------------------- machine sounds (times from layout.json)
for kind, t in LAY['sfx']:
    if kind == 'liftgo': add(thud(70, 0.12, 0.5), t, 0.5, -0.1); add(click(900, 0.02, 0.6), t + 0.05, 0.25)
    elif kind == 'liftstop': add(thud(80, 0.1, 0.6), t, 0.5); add(click(1200, 0.01), t + 0.02, 0.2)
    elif kind == 'trip': add(click(1800, 0.008, 0.7), t, 0.12, 0.2)
    elif kind == 'door': add(click(1500, 0.01, 0.8), t, 0.10, 0.3)
    elif kind == 'basket': add(click(700, 0.02, 0.6), t, 0.10, -0.2)
    elif kind == 'cup': add(thud(180, 0.04, 0.6, 0.5), t, 0.12)
    elif kind == 'cuptip': add(click(1100, 0.01, 0.6), t, 0.10)
    elif kind == 'gutter': add(click(600, 0.02, 0.5), t, 0.06, rng.uniform(-0.5, 0.5))
    elif kind == 'tip': add(click(900, 0.015, 0.8), t, 0.2)
    elif kind == 'fire':
        tt = np.arange(int(0.35 * SR)) / SR
        boing = np.sin(2 * np.pi * (180 + 900 * np.exp(-tt / 0.05)) * tt) * np.exp(-tt / 0.09)
        add(boing, t, 0.35); add(thud(110, 0.05, 0.8), t, 0.3)
    elif kind == 'goal': add(thud(220, 0.05, 0.5, 0.2), t, 0.15)
# fireworks after the final chord
for k, t in enumerate(LAY.get('fireworks', [])):
    tb = t; add(thud(55, 0.25, 1.0, 0.5), tb, 0.22, rng.uniform(-0.5, 0.5))
    for j in range(40):
        add(click(rng.uniform(2500, 6000), 0.003, 0.5), tb + 0.1 + rng.exponential(0.25), 0.03, rng.uniform(-0.8, 0.8))
# ---------------------------------------------------------------- hall
L_ = int(3.0 * SR); tt = np.arange(L_) / SR
ir = np.stack([rng.normal(0, 1, L_), rng.normal(0, 1, L_)], 1) * np.exp(-tt / 0.75)[:, None]
for c in range(2): ir[:, c] = lfilter([0.25], [1, -0.7], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.3 * wet
fade = int(1.5 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write(f'{HERE}/music_raw.wav', mix, SR)
os.system(f'ffmpeg -v error -y -i {HERE}/music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.2:attack=6:release=150,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 {HERE}/music.wav && rm {HERE}/music_raw.wav')
print('ok', DUR)
