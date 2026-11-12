"""Gymnopedie - acoustic render with VSCO-2 CE (CC0 samples only).  python3 render_audio.py /path/to/VSCO-2-CE -> music.wav
What you hear is what you see:
  pearl on a glass key        soft piano at the struck pitch + a quiet glockenspiel (the music-box tone)
  night ball on a bass bar    piano bass note (+ a very soft contrabass) for its written length
  plunger on a chime crown    the chord on the piano (+ glockenspiel two octaves up, very quiet)
  final cups                  both chords together, piano + strings swell
  mechanisms                  wind-up key, flaps, saddles, carousel start, the lift (quiet)"""
import sys, os, json, glob, re
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
src = src.replace("DUR = EV['final'] + 4.5", "DUR = json.load(open('layout.json'))['ev']['t_end']")
exec(src)
LAY = json.load(open('layout.json'))
I['vln'] = Inst('Strings/Violin Section/susVib'); I['vcsus'] = Inst('Strings/Cello Section/susvib')
TRI = Hit(sorted(glob.glob(P('Percussion/Triangle3-Hit_*'))))
for t, d, role, p, v in json.load(open('notes.json')):
    if role == 'mel':
        play('pno', t, p, v, 0.42, 0.1, dur=d, rel=0.6); play('glock', t, p, v, 0.07, 0.25, dur=0.8, rel=0.4)
    elif role in ('melchord', 'melfinal'):
        play('pno', t, p, v, 0.3, 0.1, dur=d + (2.5 if role == 'melfinal' else 0), rel=1.5 if role == 'melfinal' else 0.6)
        play('glock', t, p, v, 0.04, 0.25, dur=0.8, rel=0.4)
        if role == 'melfinal': play('vln', t, p, v, 0.08, 0.15, dur=4.0, rel=2.0)
    elif role in ('bass', 'bassfinal'):
        play('pno', t, p, v, 0.36, -0.15, dur=d + (2.5 if role == 'bassfinal' else 0), rel=0.8 if role == 'bass' else 1.5)
        play('cb', t, p if p >= 40 else p + 12, v, 0.05, -0.1)
        if role == 'bassfinal': play('vcsus', t, p + 12, v, 0.1, -0.2, dur=4.0, rel=2.0)
    elif role == 'chord':
        play('pno', t, p, v, 0.2, 0.0, dur=d, rel=0.6); play('glock', t, p + 24, v, 0.015, 0.3, dur=0.5, rel=0.3)
for kind, t in LAY['sfx']:
    if kind == 'flap': hit(WOOD, t, 70, 0.08, 0.2)
    elif kind == 'saddle': hit(SHK, t, 60, 0.06, -0.2)
    elif kind == 'ride_go': hit(RAT, t, 50, 0.04, 0.3)
    elif kind == 'tip': hit(WOOD, t, 70, 0.07, -0.2)
    elif kind == 'cradle': hit(SHK, t, 60, 0.05, 0.2)
    elif kind == 'final': hit(BT, t + 0.05, 80, 0.12, 0.2); hit(TRI, t + 0.3, 70, 0.07, -0.2)
    elif kind == 'lift_go': hit(RAT, t, 60, 0.05, -0.3)
    elif kind == 'lift_stop': hit(WOOD, t, 70, 0.06, 0.0)
hit(RAT, 0.02, 70, 0.12, -0.3)                     # the wind-up key at the start
# hall
L_ = int(3.2 * SR); tt = np.arange(L_) / SR
ir = np.stack([rng.normal(0, 1, L_), rng.normal(0, 1, L_)], 1) * np.exp(-tt / 0.9)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.55], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.32 * wet
fade = int(2.5 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-20dB:ratio=2.2:attack=10:release=150,loudnorm=I=-15:TP=-1.2:LRA=10" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
