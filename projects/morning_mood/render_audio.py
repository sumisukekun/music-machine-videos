"""Morning Mood - acoustic render with VSCO-2 CE (CC0 samples only).  python3 render_audio.py /path/to/VSCO-2-CE -> music.wav
What you hear is what you see:
  flute ball on a glass key   glockenspiel + soft piano at the struck pitch; violins sing the note for its written length
  oboe ball on a wooden key   marimba + soft piano at the struck pitch; a horn sings the note for its written length
  dawn ball through a gate    one bell per chord note (piano + glockenspiel two octaves up), the chord sustained by strings
                              (cellos for the bass, violins above) until the next gate
  mechanisms                  morning-star release, gates, baskets, ropeway, the drop, the latch, the sunrise (quiet)"""
import sys, os, json, glob, re
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")].replace("DUR = EV['final'] + 4.5", "DUR = EV['final'] + 6.0")
exec(src)
LAY = json.load(open('layout.json'))
I['vln'] = Inst('Strings/Violin Section/susVib'); I['vcsus'] = Inst('Strings/Cello Section/susvib')
I['hnsus'] = Inst('Brass/F Horn/sus')
TRI = Hit(sorted(glob.glob(P('Percussion/Triangle3-Hit_*'))))
TIMP = Hit(sorted(glob.glob(P('Percussion/Timpani/Timpani1_Hit_*'))))
for t, d, role, p, v in json.load(open('notes.json')):
    if role == 'fl':
        play('glock', t, p, v, 0.2, 0.15, dur=1.0, rel=0.5); play('pno', t, p, v, 0.16, 0.1, dur=d, rel=0.3)
        play('vln', t, p, v, 0.2, 0.2, dur=d * 0.98, rel=0.22)
    elif role == 'ob':
        play('mar', t, p, v, 0.42, -0.15); play('pno', t, p, v, 0.1, -0.1, dur=d, rel=0.3)
        play('hnsus', t, p, v, 0.16, -0.25, dur=d * 0.98, rel=0.22)
    elif role in ('chime', 'finchime'):
        big = role == 'finchime'
        play('pno', t, p, v, 0.17 if not big else 0.3, 0.0, dur=min(d, 2.5), rel=0.5)
        play('glock', t, p + 24, v, 0.025 if not big else 0.06, 0.3, dur=0.5, rel=0.3)
        if p < 52: play('vcsus', t, p, v, 0.16 if not big else 0.3, -0.1, dur=d, rel=0.5); play('cb', t, p - 12 if p >= 40 else p, v, 0.12, 0.0)
        else: play('vln', t, p, v, 0.07 if not big else 0.16, 0.1, dur=d, rel=0.5)
        if big: play('hnsus', t, p, v, 0.16, -0.2, dur=d, rel=0.8)
EVN = LAY['ev']
for kind, t in LAY['sfx']:
    if kind == 'star': hit(BT, t, 80, 0.2, 0.2)
    elif kind == 'gate': hit(WOOD, t, 80, 0.12, 0.25)
    elif kind == 'basket': hit(SHK, t, 70, 0.1, -0.3)
    elif kind == 'gond_go': hit(RAT, t, 60, 0.06, 0.4)
    elif kind == 'tip': hit(WOOD, t, 80, 0.12, -0.2)
    elif kind == 'reunion': hit(TRI, t, 90, 0.14, 0.2)
    elif kind == 'funnel': hit(LOGLO, t, 90, 0.22, 0.2)
    elif kind == 'latch': hit(SLAP, t, 90, 0.18, 0.1)
    elif kind == 'sun': hit(TIMP, t, 100, 0.3, 0.0); hit(BT, t + 0.05, 100, 0.25, -0.2); hit(TRI, t + 0.1, 100, 0.15, 0.3)
# hall
L_ = int(2.6 * SR); tt = np.arange(L_) / SR
ir = np.stack([rng.normal(0, 1, L_), rng.normal(0, 1, L_)], 1) * np.exp(-tt / 0.7)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.55], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.28 * wet
fade = int(2.0 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
