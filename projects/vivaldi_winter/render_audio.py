"""Vivaldi 'Winter' II - acoustic render with VSCO-2 CE (CC0 samples only).  python3 render_audio.py /path/to/VSCO-2-CE -> music.wav
What you hear is what you see (every sound starts on a visible hit, layout.json):
  mel    copper keys (ember ball)     violins sing the solo line + soft marimba / glockenspiel at the struck pitch
  rain1/2 glass chimes (raindrops)    marimba (soft) + glockenspiel an octave up, quiet: the pizzicato rain
  bas    log drums (iron ball)        cello pizzicato + contrabass pizzicato + a soft log-drum knock
  final  lantern bell / big drum      the chord sustained by violins / cellos, bell tree, triangle
  mechanisms                          stove pop, shelf flap, gust, bellows, chain lift, roof chute"""
import sys, os, json, glob
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
LAY = json.load(open('layout.json'))
STRUCK = {}
for n, b in LAY['balls'].items():
    for h in b['hits']:
        if h.get('ev') is not None: STRUCK[(round(h['t'], 4), h['pitch'], n)] = 1
for d in LAY['drops']: STRUCK[(round(d['t'], 4), d['pitch'], d['role'])] = 1
I['vln'] = Inst('Strings/Violin Section/susVib'); I['vsp'] = Inst('Strings/Violin Section/Spic')
I['vcsus'] = Inst('Strings/Cello Section/susvib'); I['vcp'] = Inst('Strings/Cello Section/pizzT')
TRI = Hit(sorted(glob.glob(P('Percussion/Triangle3-Hit_*'))))
skipped = 0
for t, d, role, p, v in json.load(open('notes.json')):
    r = role[:-4] if role.endswith('_fin') else role
    if (round(t, 4), p, r) not in STRUCK: skipped += 1; continue
    fin = role.endswith('_fin')
    if r == 'mel':
        play('vln', t, p, v, 0.36 if not fin else 0.4, -0.1, dur=d, rel=0.35 if not fin else 1.5)
        play('mar', t, p, v, 0.3, -0.1); play('glock', t, p + 12, v, 0.05, -0.2, dur=0.5, rel=0.3)
    elif r in ('rain1', 'rain2'):
        pan = 0.35 if r == 'rain1' else 0.2
        play('mar', t, p, v, 0.13, pan); play('glock', t, p + 12, v, 0.035, pan, dur=0.35, rel=0.2)
        if fin: play('vln', t, p, v, 0.18, pan, dur=d, rel=1.5)
    elif r == 'bas':
        play('vcp', t, p, v, 0.3, -0.05); play('cb', t, p, v, 0.32, 0.0); hit(LOGLO, t, v - 30, 0.04, 0.05)
        if fin: play('vcsus', t, p, v, 0.3, 0.0, dur=d, rel=1.5)
print('notes not struck (dropped from the audio):', skipped)
for kind, t in LAY['sfx']:
    if kind == 'pop': hit(SLAP, t, 90, 0.18, -0.3); hit(BT, t, 70, 0.12, -0.3)
    elif kind == 'flap': hit(WOOD, t, 80, 0.1, 0.1)
    elif kind == 'gust': hit(SHK, t, 100, 0.25, 0.4); hit(BT, t + 0.1, 90, 0.15, 0.4)
    elif kind == 'bellows': hit(SHK, t + 0.05, 110, 0.25, 0.0); hit(BT, t + 0.1, 100, 0.2, 0.2)
    elif kind == 'lift_go': hit(RAT, t, 70, 0.12, -0.2)
    elif kind == 'tip': hit(WOOD, t, 80, 0.08, 0.1)
    elif kind == 'final':
        hit(BT, t + 0.03, 110, 0.3, -0.25); hit(TRI, t + 0.02, 100, 0.15, 0.3); hit(SLEIGH, t + 0.3, 90, 0.12, 0.2)
# hall
L_ = int(2.4 * SR); tt = np.arange(L_) / SR
ir = np.stack([rng.normal(0, 1, L_), rng.normal(0, 1, L_)], 1) * np.exp(-tt / 0.6)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.55], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.26 * wet
fade = int(2.5 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
