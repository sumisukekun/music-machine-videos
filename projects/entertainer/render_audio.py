"""The Entertainer - acoustic render with VSCO-2 CE / VSCO 1 (CC0 recordings only).
Upright piano plays every note of the score (notes.json, unchanged from the 1902 edition).  Each bell / chime-rail strike
of the melody balls is doubled very softly by a glockenspiel, each see-saw landing by a soft wood knock (log drum for
ZUN, claves for CHA), so every sound has its visible strike.  Machine sounds (coin, crank, lever, gates, swing doors,
splash, catapults, mug tip, jukebox) come from layout.json.
python3 render_audio.py /path/to/VSCO -> music.wav"""
import sys, os, json, math, glob, re
sys.argv = [sys.argv[0], sys.argv[1]]
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
src = open(os.path.join(HERE, 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
TRI = Hit(sorted(glob.glob(P('Percussion/Triangle3-Hit_*'))))
LOGHI = Hit(sorted(glob.glob(P('Percussion/LogDrumHi_*'))))
LAY = json.load(open('layout.json'))
# 1) the piano: every note of the score
for t, d, p, v, hand in json.load(open('notes.json')):
    play('pno', t, p, v, 0.62 if hand == 'R' else 0.5, 0.12 if hand == 'R' else -0.12, dur=max(d, 0.12), rel=0.16)
# 2) the strikes themselves (very soft): glock on the melody bells, wood knocks on the see-saws
for n in ('coin', 'P', 'C'):
    for h in LAY['balls'][n]['hits']:
        if 'pitch' not in h: continue
        if h['kind'] in ('bell', 'cbell', 'juke', 'rail', 'lever'):
            if n == 'C' and abs(h['t'] - LAY['balls']['P']['hits'][-1]['t']) < 0.01 and False: pass
            p = h['pitch']
            while p < 72: p += 12
            g = 0.06 if h['kind'] != 'rail' else 0.075
            if n in ('P', 'C') and h['sec'] == 'A2': g *= 0.75          # two balls strike together in A'
            play('glock', h['t'], p, 80, g, 0.3 if n == 'C' else 0.18, dur=0.5, rel=0.3)
for n, H, g in (('Z', LOGLO, 0.13), ('Ch', CLAV, 0.06)):
    for h in LAY['balls'][n]['hits']:
        if 'pitch' in h: hit(H, h['t'], 70, g, -0.25 if n == 'Z' else 0.2)
# 3) machine sounds
EVL = LAY['ev']
for kind, t in LAY['sfx']:
    if kind == 'crank': hit(RAT, t, 80, 0.22, -0.4); hit(RAT, t + 0.45, 80, 0.18, -0.4)
    elif kind == 'coin': play('glock', t, 96, 90, 0.16, 0.3, dur=0.3, rel=0.4); hit(TRI, t + 0.03, 70, 0.08, 0.3)
    elif kind == 'lever': hit(SLAP, t, 95, 0.22, 0.1); hit(LOGLO, t + 0.004, 100, 0.25, 0.1)
    elif kind == 'gate': hit(WOOD, t, 85, 0.16, -0.3)
    elif kind == 'door': hit(SLAP, t, 125, 0.45, 0.25); hit(LOGLO, t + 0.01, 110, 0.3, 0.25); hit(SLAP, t + 0.32, 90, 0.2, 0.3)
    elif kind == 'splash': hit(SHK, t, 110, 0.3, 0.2); hit(TAMB, t + 0.02, 70, 0.08, 0.2)
    elif kind == 'thud': hit(LOGLO, t, 85, 0.12, -0.25)
    elif kind == 'tip': hit(RAT, t, 70, 0.14, 0.3)
# jukebox: final chord sparkle, the lid snaps shut
FINAL = EV['final']; TIN = EVL['juke']['t_in']
hit(BT, FINAL + 0.02, 100, 0.26, -0.3); hit(SLEIGH, FINAL + 0.05, 90, 0.16, 0.3); hit(CRASH, FINAL, 70, 0.12, 0.2)
hit(SLAP, TIN + 0.35 + 0.22, 110, 0.32, 0.0); hit(LOGHI, TIN + 0.36 + 0.22, 90, 0.18, 0.0)
# saloon room
L = int(1.6 * SR); tt = np.arange(L) / SR
ir = np.stack([rng.normal(0, 1, L), rng.normal(0, 1, L)], 1) * np.exp(-tt / 0.32)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.65], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.17 * wet
fade = int(2.0 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.2:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
