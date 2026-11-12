"""Toccata and Fugue in D minor - audio.  python3 render_audio.py /path/to/VSCO-2-CE -> music.wav
The organ is synthesised here (additive pipe-organ model: principal 8' + octave 4' + twelfth + fifteenth, a 16' under the low
notes, wind chiff on every attack; nothing sampled). Mechanism and marble sounds use CC0 recordings (VSCO-2 CE / VSCO 1).
What you hear is what you see: every organ note starts on a visible strike (lead.json rows = layout.json hits), a key sounds
as long as the score holds the note (a marble resting on it for the held notes); the final chord's pipes speak when the first
golden marbles land and every later landing adds a small chime of its key's pitch."""
import sys, os, json, glob, math
import numpy as np, soundfile as sf
from scipy.signal import fftconvolve, lfilter
sys.argv = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else '/home/claude/samples/vsco']
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
src = open(os.path.join(HERE, 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
src = src.replace("EV = json.load(open('events.json')); DUR = EV['final'] + 5.5", "EV = json.load(open('events.json')); DUR = EV['secs']['end'] + 3.4")
exec(src)
LEAD = json.load(open('lead.json')); LAY = json.load(open('layout.json')); P = LAY['parts']
CRANK = Hit(sorted(glob.glob(os.path.join(ROOT, 'Percussion/Ratchet1-Slow*'))) or sorted(glob.glob(os.path.join(ROOT, 'Percussion/Ratchet1-Crank*'))))
LOGHI = Hit(sorted(glob.glob(os.path.join(ROOT, 'Percussion/LogDrumHi_*'))))
GUIRO = Hit(sorted(glob.glob(os.path.join(ROOT, 'VSCO 1 Percussion/varWood/guiro_long*.wav'))))
CLV1 = Hit(sorted(glob.glob(os.path.join(ROOT, 'VSCO 1 Percussion/varWood/claves_*.wav'))))
rngA = np.random.default_rng(565)


# ------------------------------------------------------------------ the organ
def organ(p, dur, level=1.0, full=False, att=None, rel=None, chiff=1.0):
    """one pipe-organ note (all speaking ranks of one key), stereo"""
    f0 = 440.0 * 2 ** ((p - 69) / 12)
    att = att or (0.035 if p >= 48 else 0.07); rel = rel or (0.14 if p >= 48 else 0.3)
    n = int((dur + rel + 0.05) * SR); t = np.arange(n) / SR
    A = {}
    def rank(mult, gain, slope=1.25):
        for k in range(1, 40):
            h = mult * k
            if f0 * h > 9000: break
            A[h] = A.get(h, 0) + gain / k ** slope
    rank(1, 1.0)                                    # principal 8'
    rank(2, 0.45)                                   # octave 4'
    rank(3, 0.12 if not full else 0.3)              # twelfth
    rank(4, 0.16 if not full else 0.32)             # fifteenth
    if full: rank(6, 0.12); rank(8, 0.1)            # mixture
    if p < 52 or full: rank(0.5, 0.55 if p < 52 else 0.35, 1.6)   # bourdon 16'
    y = np.zeros(n)
    for h, a in A.items():
        ph = rngA.uniform(0, 2 * np.pi)
        y += a * np.sin(2 * np.pi * f0 * h * t * (1 + 0.0004 * rngA.normal()) + ph)
    # second, slightly detuned principal (two pipes on the key: warmth)
    y += 0.25 * np.sin(2 * np.pi * f0 * 1.0007 * t) + 0.08 * np.sin(2 * np.pi * f0 * 2.0011 * t)
    y /= math.sqrt(sum(a * a for a in A.values()))
    env = np.ones(n)
    ka = int(att * SR); env[:ka] = np.linspace(0, 1, ka) ** 1.5 * (1 + 0.15 * np.sin(np.linspace(0, np.pi, ka)))
    kr = int(dur * SR)
    if kr < n: env[kr:] = np.exp(-(t[kr:] - dur) / (rel / 4))
    # wind: a slow tremble + chiff (a short noise burst around the 4th-8th harmonic on the attack)
    env *= 1 + 0.01 * np.sin(2 * np.pi * 5.3 * t)
    nz = rngA.normal(0, 1, min(n, int(0.06 * SR)))
    w0 = 2 * np.pi * min(7000, f0 * 5) / SR; r_ = 0.97
    nz = lfilter([1 - r_], [1, -2 * r_ * np.cos(w0), r_ * r_], nz) * np.exp(-np.arange(len(nz)) / SR / 0.015)
    y[:len(nz)] += chiff * 0.6 * nz / max(1e-9, np.abs(nz).max())
    y = y * env * level
    pan = (((p * 7) % 12) / 11 - 0.5) * 0.7
    l, r = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
    return np.stack([y * l * 1.41, y * r * 1.41], 1).astype('float32')


def put(x, t):
    i = max(0, int(t * SR)); e = min(N, i + len(x)); out[i:e] += x[:e - i]


final_row = [r for r in LEAD if r[3] == 'final'][0]
for t, d, ps, role, n_, bar, pos, note in LEAD:
    if role == 'final': continue
    for p in ps:
        lev = 0.22
        if role == 'ped': lev = 0.42
        elif role in ('chord', 'dmaj', 'inner', 'chord2'): lev = 0.2
        elif role == 'trio': lev = 0.2
        elif role == 'solo': lev = 0.3
        elif role.startswith('call'): lev = 0.3
        full = role in ('ped', 'chord2') or (role == 'chord')
        put(organ(p, max(0.05, d), lev, full=full), t)
# final chord: full organ, plus a 32'-like bass under the pedal D, swelling as the golden pipes empty
tF = final_row[0]; dF = final_row[1]
for p in final_row[2]:
    put(organ(p, dF, 0.3 if p > 40 else 0.5, full=True, rel=1.2), tF)
put(organ(26, dF, 0.35, full=False, att=0.25, rel=1.5, chiff=0.0), tF)
# every marble strike: a small click of the marble on the key (so the hits read in the sound too)
hits = []
for n, b in LAY['balls'].items():
    for h in b['hits']:
        hits.append((h['t'], h['kind'], b['kind'], np.linalg.norm(h['vin'])))
for t, kind, bk, v in hits:
    if bk == 'rain': continue
    if kind in ('slat',): hit(WOOD, t, 60, 0.05, 0.2); continue
    if bk == 'giant': hit(BD, t, 110, 0.5, -0.1); hit(LOGLO, t + 0.004, 100, 0.25, 0.0); continue
    g = 0.06 + 0.02 * min(4, v)
    hit(CLAV if kind in ('wall', 'bar', 'console', 'chordkey') else WOOD, t, 70, g, rngA.uniform(-0.4, 0.4))
# the flood: every golden marble clicks, plus a little chime of its key (D or F, high octave), a rising shaker bed
rain = P['final']['rain_hits']
for k, (t, p, name) in enumerate(rain):
    if k % 2 == 0: hit(WOOD, t, 55 + rngA.integers(0, 30), 0.035, rngA.uniform(-0.6, 0.6))
    if k % 3 == 0: play('glock', t, p + 24 if p > 40 else 74, 70, 0.035, rngA.uniform(-0.5, 0.5), dur=0.25, rel=0.3)
for tt in np.arange(tF, P['final']['t_last'], 0.11): hit(SHK, tt, 60, 0.05, rngA.uniform(-0.5, 0.5))
# mechanisms
for kind, t in LAY['sfx']:
    if kind == 'flick': hit(CLAV, t, 100, 0.18, 0.1)
    elif kind == 'lever': hit(LOGHI, t, 90, 0.2, 0.0)
    elif kind == 'unlock': hit(CLV1, t, 110, 0.3, 0.0); hit(BT, t + 0.03, 90, 0.18, 0.2); hit(SLAP, t + 0.35, 90, 0.15, 0.0)
    elif kind in ('shelf', 'pin', 'gate', 'tip'): hit(WOOD, t, 90, 0.15, 0.0)
    elif kind == 'door':
        for k in range(10): hit(CRANK, t + 0.33 * k, 70, 0.07, -0.2)
    elif kind in ('socket', 'bucket'): hit(BD, t, 90, 0.3, 0.0)
    elif kind in ('lift', 'winch'):
        for k in range(5): hit(CRANK, t + 0.2 * k, 70, 0.07, 0.3)
    elif kind == 'liftstop': hit(LOGLO, t, 80, 0.15, 0.3)
    elif kind == 'ledge': hit(GUIRO, t, 70, 0.12, -0.3)
    elif kind == 'spout': hit(BT, t, 60, 0.12, 0.3)
    elif kind == 'gate_open': hit(CRASH, t + 0.7, 80, 0.15, 0.0); hit(BT, t, 100, 0.2, 0.0)
hit(CRASH, P['final']['t_last'] + 0.05, 100, 0.25, 0.0); hit(BT, P['final']['t_last'] + 0.1, 110, 0.25, 0.0)
# cathedral: long dense reverb (about 3.5 s), a little pre-delay
L_ = int(4.5 * SR); tt = np.arange(L_) / SR
ir = np.stack([rngA.normal(0, 1, L_), rngA.normal(0, 1, L_)], 1) * np.exp(-tt / 0.55)[:, None]
for c in range(2): ir[:, c] = lfilter([0.25], [1, -0.72], ir[:, c])
ir = np.concatenate([np.zeros((int(0.03 * SR), 2)), ir])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.42 * wet
fade = int(1.8 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.2:attack=10:release=150,loudnorm=I=-14:TP=-1.2:LRA=11" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
