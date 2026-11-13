"""O Tannenbaum - acoustic render with VSCO-2 CE (CC0 samples only).  python3 render_audio.py /path/to/VSCO-2-CE -> music.wav
What you hear is what you see (every pitched sound starts on a visible strike, times from layout.json):
  Stella's ice keys   glockenspiel (octave up) + marimba at the struck pitch; long keys (she rolls along them) + harp;
                      verse 3: + violins, started by the same strike
  ornament bells      glockenspiel + harp at the struck pitch (verse 3: + violins)
  pinecone drums      contrabass pizzicato + piano at the struck bass note, a soft log-drum thump (verses 2, 3)
  tree top (final)    glockenspiel + bell tree; strings and horns hold the final F major chord from that strike
  mechanisms          snow landing, gates, cup catches, lift motor, chairlift ratchet, balloon burner, latch, snowball, cannon"""
import sys, os, json, glob
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
src = src.replace("DUR = EV['final'] + 4.5", "DUR = int((EV['final'] + 4.6) * 30) / 30")
exec(src)
LAY = json.load(open('layout.json')); LEAD = json.load(open('lead.json')); B = LAY['balls']
I['vln'] = Inst('Strings/Violin Section/susVib'); I['vcsus'] = Inst('Strings/Cello Section/susvib')
I['hnsus'] = Inst('Brass/F Horn/sus'); I['harp'] = Inst('Strings/Harp', 0)
TRI = Hit(sorted(glob.glob(P('Percussion/Triangle3-Hit_*'))))
GUIRO = Hit(sorted(glob.glob(P('VSCO 1 Percussion/varWood/guiro_long*'))))
for h in B['stella']['hits']:
    if h['kind'] not in ('key', 'rollkey'): continue
    t, p, v, d = LEAD[h['i']][:4]; verse = LEAD[h['i']][4]
    play('glock', t, p + 12, v, 0.2, 0.15, dur=0.9, rel=0.4)
    play('mar', t, p, v, 0.42, -0.1)
    if h['kind'] == 'rollkey': play('harp', t, p, v, 0.22, 0.25, dur=d, rel=0.4)
    if verse == 1: play('harp', t, p - 12, v, 0.1, -0.2, dur=d, rel=0.3)
    if verse == 2: play('vln', t, p, v, 0.07, 0.2, dur=d * 0.95, rel=0.25)
for n, b in B.items():
    if b['kind'] != 'orn': continue
    h = b['hits'][0]; t, p, v, d = LEAD[h['i']][:4]; verse = LEAD[h['i']][4]
    play('glock', t, p + 12, v, 0.17, 0.25, dur=1.0, rel=0.5)
    play('harp', t, p, v, 0.22, 0.2, dur=1.0, rel=0.4)
    if verse == 2: play('vln', t, p, v, 0.07, 0.2, dur=d * 0.95, rel=0.25)
for n, b in B.items():
    if b['kind'] != 'pine': continue
    h = b['hits'][0]; t, p = h['t'], h['pitch']; fin = h['kind'] == 'bassfinal'
    play('cb', t, p, 96, 0.42, -0.05)
    play('pno', t, p, 88, 0.3, -0.15, dur=0.9 if not fin else 4.0, rel=0.3 if not fin else 1.5)
    hit(LOGLO, t, 70, 0.12, -0.1)
    if fin: play('pno', t, p - 12, 108, 0.35, 0.0, dur=4.0, rel=1.5)
h = [h for h in B['stella']['hits'] if h['kind'] == 'top'][0]
t, p, v, d = LEAD[h['i']][:4]
play('glock', t, p + 12, 120, 0.3, 0.0, dur=3.0, rel=1.0); play('glock', t, p + 24, 110, 0.18, 0.1, dur=3.0, rel=1.0)
play('harp', t, p, 110, 0.25, 0.2, dur=3.5, rel=1.2); play('mar', t, p, 110, 0.4, -0.1)
hit(BT, t + 0.02, 110, 0.32, 0.3); hit(CRASH, t, 85, 0.13, 0.2)
for q in (41, 48, 53, 57, 60, 65, 69, 72):
    play('vln' if q >= 60 else 'vcsus', t, q, 100, 0.08, 0.2 if q >= 60 else -0.2, dur=3.6, rel=1.6)
    if 53 <= q <= 69: play('hnsus', t, q, 100, 0.06, -0.25, dur=3.6, rel=1.6)
for kind, t in LAY['sfx']:
    if kind == 'whoosh': hit(BT, t, 70, 0.18, -0.4)
    elif kind == 'snowland': hit(BD, t, 60, 0.2, -0.3); hit(SHK, t + 0.01, 90, 0.2, -0.3)
    elif kind == 'gate': hit(WOOD, t, 80, 0.16, 0.2)
    elif kind == 'catch': hit(LOGLO, t, 60, 0.12, 0.0)
    elif kind == 'motor': hit(RAT, t, 60, 0.1, 0.2)
    elif kind == 'ratchet': hit(RAT, t, 70, 0.13, 0.2)
    elif kind == 'clunk': hit(WOOD, t, 70, 0.12, 0.1)
    elif kind == 'burner': hit(SHK, t, 95, 0.16, -0.2); hit(SHK, t + 0.12, 80, 0.12, -0.2)
    elif kind == 'latch': hit(CLAV, t, 90, 0.18, -0.3)
    elif kind == 'roll': hit(GUIRO, t, 60, 0.08, -0.3)
    elif kind == 'slam': hit(BD, t, 100, 0.35, 0.2); hit(SLAP, t + 0.004, 100, 0.25, 0.2)
    elif kind == 'plink': hit(TRI, t, 45, 0.035, 0.35)
    elif kind == 'thud': hit(LOGLO, t, 55, 0.08, -0.2)
    elif kind == 'cannon': hit(BD, t, 110, 0.4, 0.2); hit(SLAP, t + 0.004, 110, 0.3, 0.2)
    elif kind == 'star':
        hit(SLEIGH, t + 0.03, 100, 0.22, -0.3)
        for k in range(6): hit(SLEIGH, t + 0.3 + 0.25 * k, 85 - 6 * k, 0.1, -0.4 + 0.15 * k)
L_ = int(2.4 * SR); tt = np.arange(L_) / SR
ir = np.stack([rng.normal(0, 1, L_), rng.normal(0, 1, L_)], 1) * np.exp(-tt / 0.65)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.55], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.27 * wet
fade = int(1.5 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
