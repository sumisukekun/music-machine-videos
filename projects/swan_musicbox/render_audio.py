"""Swan Lake music box - own synthesized steel-comb music box + VSCO-2 CE (CC0) strings, horns, timpani."""
import sys, os, json, math, glob
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
I['vsus'] = Inst('Strings/Violin Section/susVib'); I['csus'] = Inst('Strings/Cello Section/susvib')
I['cpizz'] = Inst('Strings/Cello Section/pizzT'); I['horn'] = Inst('Brass/F Horn/sus')
TIMP = [Hit(sorted(glob.glob(P(f'Percussion/Timpani/Timpani{k}_Hit_*')))) for k in range(1, 6)]

def mbox(p, v, dur):
    """steel comb tooth: fundamental + inharmonic cantilever partials + a tiny pluck click"""
    f = 440 * 2 ** ((p - 69) / 12); n = int((dur + 0.2) * SR); t = np.arange(n) / SR
    tau = 1.6 * (440 / f) ** 0.45
    x = np.sin(2 * np.pi * f * t) * np.exp(-t / tau)
    x += 0.16 * np.sin(2 * np.pi * f * 2.0 * t + 0.3) * np.exp(-t / (tau * 0.35))
    x += 0.22 * np.sin(2 * np.pi * f * 5.4 * t) * np.exp(-t / 0.09)
    x += 0.07 * np.sin(2 * np.pi * f * 9.3 * t) * np.exp(-t / 0.03)
    if p < 56: x += 0.35 * np.sin(2 * np.pi * f * 3.0 * t) * np.exp(-t / (tau * 0.4))     # bass teeth: more body
    cl = rng.normal(0, 1, int(0.004 * SR)) * np.exp(-np.arange(int(0.004 * SR)) / (0.0008 * SR)); x[:len(cl)] += 0.25 * cl
    x[:int(0.002 * SR)] *= np.linspace(0, 1, int(0.002 * SR))
    end = int(dur * SR)
    if end < n: x[end:] *= np.linspace(1, 0, n - end)
    return np.stack([x, x], 1).astype('float32')
for t, d, role, p, v in json.load(open('notes.json')):
    if role == 'box':
        pan = max(-0.6, min(0.6, (p - 66) / 30))
        place(mbox(p, v, max(d, 1.2)), 1.0, t, 0.30 * (0.35 + 0.65 * v / 127) * (1.25 if p < 56 else 1.0), pan)
    elif role == 'vsus': play('vsus', t, p, v, 0.22, 0.25, dur=d, rel=0.35)
    elif role == 'vpad': play('vsus', t, p, v, 0.10, -0.1, dur=d, rel=0.4)
    elif role == 'cello': play('csus', t, p, v, 0.30, -0.25, dur=d, rel=0.4)
    elif role == 'pizz': play('cpizz', t, p, v, 0.30, -0.3)
    elif role == 'horn': play('horn', t, p, v, 0.20, 0.1, dur=d, rel=0.4)
    elif role == 'timp': hit(TIMP[0 if p < 45 else 1], t, v, 0.45, -0.05)
    elif role == 'crash': hit(CRASH, t, v, 0.28, 0.2)
    elif role == 'bt': hit(BT, t, v, 0.3, -0.2)
hit(RAT, 0.2, 80, 0.30, -0.35); hit(RAT, 0.75, 80, 0.26, -0.35)       # winding the key
L = int(3.0 * SR); tt = np.arange(L) / SR
ir = np.stack([rng.normal(0, 1, L), rng.normal(0, 1, L)], 1) * np.exp(-tt / 0.75)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.6], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.32 * wet
fade = int(2.5 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
