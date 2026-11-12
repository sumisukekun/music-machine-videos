"""Deck the Halls - acoustic render with VSCO-2 CE (CC0 samples only).  python3 render_audio.py /path/to/VSCO-2-CE -> music.wav
What you hear is what you see (every pitched sound starts on a visible strike, times from layout.json):
  Holly's keys      marimba + soft glockenspiel at the struck pitch
  ornament bells    glockenspiel + harp at the struck pitch (verse 2: + violins, started by the same strike)
  Yule bass bars    contrabass pizzicato + piano at the struck bass note (verse 2)
  star (final)      glockenspiel + bell tree; strings and horns hold the final F major chord from that strike
  mechanisms        gift-box lid / spring, cup catches, lift motor, wheel ratchet, gates, see-saw, latch, train, cannon"""
import sys, os, json, glob
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
LAY = json.load(open('layout.json')); LEAD = json.load(open('lead.json')); B = LAY['balls']
I['vln'] = Inst('Strings/Violin Section/susVib'); I['vcsus'] = Inst('Strings/Cello Section/susvib')
I['hnsus'] = Inst('Brass/F Horn/sus'); I['harp'] = Inst('Strings/Harp', 0)
TIMP = Hit(sorted(glob.glob(P('Percussion/Timpani/Timpani1_Hit_*'))))
TRI = Hit(sorted(glob.glob(P('Percussion/Triangle3-Hit_*'))))
for h in B['holly']['hits']:
    if h['kind'] != 'key': continue
    t, p, v, d = LEAD[h['i']][:4]; v2 = LEAD[h['i']][7] == 1
    play('mar', t, p, v, 0.5, -0.1)
    play('glock', t, p + 12, v, 0.07, 0.15, dur=0.6, rel=0.3)
    if v2: play('vln', t, p, v, 0.07, 0.2, dur=d * 0.95, rel=0.25)
for n, b in B.items():
    if b['kind'] != 'orn': continue
    h = b['hits'][0]; t, p, v, d = LEAD[h['i']][:4]; v2 = LEAD[h['i']][7] == 1
    fin = abs(t - EV['final']) < 1e-6
    play('glock', t, p + 12, v, 0.17, 0.25, dur=1.0 if not fin else 3.0, rel=0.5)
    play('harp', t, p, v, 0.2, 0.2, dur=1.0 if not fin else 3.5, rel=0.4)
    if v2: play('vln', t, p, v, 0.07, 0.2, dur=d * 0.95 if not fin else 3.5, rel=0.25 if not fin else 1.4)
for h in B['yule']['hits']:
    t, p = h['t'], h['pitch']; fin = h['kind'] == 'bassfinal'
    play('cb', t, p, 96, 0.4, -0.05)
    play('pno', t, p, 90, 0.3, -0.15, dur=0.9 if not fin else 4.0, rel=0.3 if not fin else 1.5)
    if h['kind'] == 'trigger': hit(TIMP, t, 100, 0.3, 0.0)
    if fin:
        hit(TIMP, t, 118, 0.45, 0.0); play('pno', t, p - 12, 110, 0.35, 0.0, dur=4.0, rel=1.5)
        for q in (41, 48, 53, 57, 60, 65, 69, 72):
            play('vln' if q >= 60 else 'vcsus', t, q, 100, 0.08, 0.2 if q >= 60 else -0.2, dur=3.6, rel=1.6)
            if 53 <= q <= 69: play('hnsus', t, q, 100, 0.06, -0.25, dur=3.6, rel=1.6)
h = B['star']['hits'][0]
play('glock', h['t'], 89, 120, 0.3, 0.0, dur=3.0, rel=1.0); hit(BT, h['t'] + 0.02, 110, 0.32, 0.3); hit(CRASH, h['t'], 90, 0.15, 0.2)
for kind, t in LAY['sfx']:
    if kind == 'lid': hit(WOOD, t, 90, 0.25, -0.3)
    elif kind == 'spring': hit(SLAP, t, 80, 0.2, -0.3)
    elif kind == 'catch': hit(LOGLO, t, 70, 0.14, 0.0)
    elif kind == 'motor': hit(RAT, t, 60, 0.1, 0.2)
    elif kind == 'ratchet': hit(RAT, t, 75, 0.14, 0.2)
    elif kind == 'clunk': hit(WOOD, t, 70, 0.12, 0.1)
    elif kind == 'trap': hit(CLAV, t, 70, 0.1, 0.0)
    elif kind == 'gate': hit(WOOD, t, 85, 0.18, 0.2)
    elif kind == 'knock': hit(WOOD, t, 65, 0.12, 0.2)
    elif kind == 'seesaw': hit(LOGLO, t, 100, 0.25, -0.1)
    elif kind == 'latch': hit(CLAV, t, 95, 0.2, -0.3)
    elif kind == 'roll': hit(SHK, t, 60, 0.08, -0.3)
    elif kind == 'chug':
        for k in range(10): hit(SHK, t + 0.3 * k, 70 - 3 * k, 0.07, 0.2)
    elif kind == 'plink': hit(TRI, t, 45, 0.035, 0.35)
    elif kind == 'cannon': hit(BD, t, 110, 0.4, 0.2); hit(SLAP, t + 0.004, 110, 0.3, 0.2)
    elif kind == 'giftpop':
        hit(SLEIGH, t + 0.03, 100, 0.22, -0.3)
        for k in range(6): hit(SLEIGH, t + 0.3 + 0.25 * k, 85 - 6 * k, 0.1, -0.4 + 0.15 * k)
L_ = int(2.4 * SR); tt = np.arange(L_) / SR
ir = np.stack([rng.normal(0, 1, L_), rng.normal(0, 1, L_)], 1) * np.exp(-tt / 0.6)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.55], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.25 * wet
fade = int(1.5 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
