"""Vivaldi 'Spring' - acoustic render with VSCO-2 CE (CC0 samples only).  python3 render_audio.py /path/to/VSCO-2-CE -> music.wav
What you hear is what you see (every sound starts on a visible hit, layout.json):
  mel ice keys (spring ball)     marimba + soft glockenspiel at the struck pitch, violins sing the tune
  har ice keys (blossom ball)    marimba (softer) + violin spiccato
  bas stump drums (earth ball)   cello pizzicato + contrabass pizzicato + a soft log-drum knock
  bud bells (birds)              glockenspiel an octave up + xylophone + violin spiccato
  final golden bells             the chord sustained by violins / cellos, the great bell, cymbal, bell tree
  mechanisms                     icicles cracking, nest doors, ropeway motor, trapdoors"""
import sys, os, json, glob
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
LAY = json.load(open('layout.json'))
STRUCK = {}
for n, b in LAY['balls'].items():
    for h in b['hits']:
        if h.get('ev') is not None: STRUCK[(round(h['t'], 4), h['pitch'])] = n
I['vln'] = Inst('Strings/Violin Section/susVib'); I['vsp'] = Inst('Strings/Violin Section/Spic')
I['vcsus'] = Inst('Strings/Cello Section/susvib'); I['vcp'] = Inst('Strings/Cello Section/pizzT')
TRI = Hit(sorted(glob.glob(P('Percussion/Triangle3-Hit_*'))))
TIMP = Hit(sorted(glob.glob(P('Percussion/Timpani/Timpani1_Hit_*'))))
skipped = 0
for t, d, role, p, v in json.load(open('notes.json')):
    if (round(t, 4), p) not in STRUCK: skipped += 1; continue
    if role == 'mel':
        play('mar', t, p, v, 0.42, -0.1); play('glock', t, p + 12, v, 0.07, -0.2, dur=0.5, rel=0.3)
        play('vln', t, p, v, 0.2, -0.15, dur=d, rel=0.2)
    elif role == 'har':
        play('mar', t, p, v, 0.26, 0.15); play('vsp', t, p, v, 0.16, 0.25)
    elif role == 'bas':
        play('vcp', t, p, v, 0.34, 0.0); play('cb', t, p - 12, v, 0.3, 0.0); hit(LOGLO, t, v - 20, 0.07, 0.05)
    elif role in ('birdA', 'birdB'):
        pan = -0.3 if role == 'birdA' else 0.3
        play('glock', t, p + 12, v, 0.2, pan, dur=0.6, rel=0.3); play('xylo', t, p, v, 0.1, pan)
        play('vsp', t, p, v, 0.12, pan)
    elif role.endswith('_fin'):
        r = role[:-4]
        if r == 'bas':
            play('vcsus', t, p, v, 0.3, 0.0, dur=d, rel=1.0); play('cb', t, p - 12, v, 0.35, 0.0); play('vcp', t, p, v, 0.3, 0.0)
        else:
            play('vln', t, p, v, 0.2, -0.2 if r == 'mel' else 0.2, dur=d, rel=1.0); play('mar', t, p, v, 0.4, 0.0)
        play('glock', t, p + 12, v, 0.12, 0.0, dur=2.5, rel=1.0)
print('notes not struck (dropped from the audio):', skipped)
for kind, t in LAY['sfx']:
    if kind == 'crack':
        hit(SLAP, t, 100, 0.25, 0.0); hit(BT, t - 0.02, 90, 0.2, 0.3)
    elif kind == 'nest': hit(WOOD, t, 90, 0.2, 0.3)
    elif kind == 'rope_go': hit(RAT, t, 70, 0.1, -0.4)
    elif kind == 'door': hit(WOOD, t, 80, 0.07, 0.1)
    elif kind == 'final':
        for pp, g in ((40, 0.3), (52, 0.26), (64, 0.16), (71, 0.07)): play('pno', t, pp, 120, g, 0.0, dur=3.0, rel=1.0)
        hit(TIMP, t, 120, 0.35, 0.0); hit(CRASH, t, 110, 0.22, 0.2); hit(BT, t + 0.05, 110, 0.32, -0.25); hit(TRI, t + 0.03, 100, 0.15, 0.3)
# hall
L_ = int(2.2 * SR); tt = np.arange(L_) / SR
ir = np.stack([rng.normal(0, 1, L_), rng.normal(0, 1, L_)], 1) * np.exp(-tt / 0.5)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.55], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.22 * wet
fade = int(2.5 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
