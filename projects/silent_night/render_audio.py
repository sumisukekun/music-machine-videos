"""Silent Night - acoustic render with VSCO-2 CE (CC0 samples only).  python3 render_audio.py /path/to/VSCO-2-CE -> music.wav
What you hear is what you see (every pitched sound starts on a visible strike, from layout.json):
  melody keys     glockenspiel + piano at the struck pitch (a music box); the children 'sing' it along: strings grow with the
                  number of balls on the key (2+: violins, 4+: cellos an octave below)
  bass bells      contrabass pizzicato + piano (the snowball)
  ornament bells  harp + soft glockenspiel, the two inner voices struck
  church bell     the last bass note: piano + contrabass + timpani, and strings / horns hold the final C major chord
  mechanisms      lantern trapdoor, gates, see-saw, lift crank, doors; sleigh bells and a bell tree for the wave of lights"""
import sys, os, json, glob
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
LAY = json.load(open('layout.json')); LEAD = json.load(open('lead.json')); B = LAY['balls']
I['vln'] = Inst('Strings/Violin Section/susVib'); I['vcsus'] = Inst('Strings/Cello Section/susvib')
I['hnsus'] = Inst('Brass/F Horn/sus'); I['harp'] = Inst('Strings/Harp', 0)
TIMP = Hit(sorted(glob.glob(P('Percussion/Timpani/Timpani1_Hit_*'))))
for h in B['solo']['hits']:
    i = h['ev']; t, p, v, d, bar, pos = LEAD[i]; n = h['lanes']
    g = 1.0 + 0.3 * np.log2(n); fin = h['kind'] == 'final'
    play('glock', t, p + 12, v, 0.2 * g, 0.1, dur=1.2 if not fin else 3.0, rel=0.6)
    play('pno', t, p, v, 0.22 * g, -0.05, dur=d if not fin else 3.5, rel=0.3 if not fin else 1.2)
    if n >= 2: play('vln', t, p, v, 0.06 + 0.02 * n, 0.15, dur=d * 0.97, rel=0.3 if not fin else 1.5)
    if n >= 4: play('vcsus', t, p - 12, v, 0.12, -0.2, dur=d * 0.97, rel=0.3 if not fin else 1.5)
for h in B['bass']['hits']:
    t, p = h['t'], h['pitch']; fin = h['kind'] == 'churchbell'
    play('pno', t, p, 92, 0.28 if not fin else 0.45, -0.1, dur=1.0 if not fin else 4.0, rel=0.4 if not fin else 1.5)
    play('cb', t, p, 92, 0.32, -0.05)
    if fin:
        play('pno', t, p - 12, 110, 0.35, 0.0, dur=4.0, rel=1.5); hit(TIMP, t, 120, 0.45, 0.0)
        for q in (48, 55, 60, 64, 67, 72):
            play('vln' if q >= 60 else 'vcsus', t, q, 100, 0.09, 0.0 if q < 60 else 0.2, dur=3.6, rel=1.6)
            if 55 <= q <= 67: play('hnsus', t, q, 100, 0.07, -0.2, dur=3.6, rel=1.6)
for n in ('pairlo', 'pairhi'):
    for h in B[n]['hits']:
        t, p = h['t'], h['pitch']; fin = bool(h.get('final'))
        play('harp', t, p, 90, 0.2, 0.25 if n == 'pairhi' else -0.25, dur=1.5 if not fin else 3.5, rel=0.5)
        play('glock', t, p + 24, 70, 0.04, 0.3, dur=0.5, rel=0.3)
for kind, t in LAY['sfx']:
    if kind == 'release': hit(WOOD, t, 90, 0.25, 0.0); hit(BT, t + 0.05, 70, 0.15, 0.3)
    elif kind == 'gate': hit(WOOD, t, 100, 0.3, -0.2); hit(CLAV, t + 0.01, 80, 0.12, -0.2)
    elif kind in ('gate_small', 'door'): hit(WOOD, t, 75, 0.15, 0.3)
    elif kind == 'seesaw': hit(LOGLO, t + 0.003, 110, 0.3, -0.1)
    elif kind == 'launch': hit(SLAP, t, 85, 0.15, -0.1)
    elif kind == 'lift': hit(RAT, t, 70, 0.12, -0.3)
    elif kind == 'wave':
        hit(BT, t, 110, 0.3, 0.0)
        for k in range(10): hit(SLEIGH, t + 0.28 * k, 90 - 4 * k, 0.12, -0.5 + 0.1 * k)
L_ = int(3.0 * SR); tt = np.arange(L_) / SR
ir = np.stack([rng.normal(0, 1, L_), rng.normal(0, 1, L_)], 1) * np.exp(-tt / 0.75)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.55], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.32 * wet
fade = int(2.0 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
