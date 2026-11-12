"""Jingle Bells - acoustic render with VSCO-2 CE (CC0 samples only).  python3 render_audio.py /path/to/VSCO-2-CE -> music.wav
What you hear is what you see (every sound starts on a visible strike, times from layout.json / events.json):
  ice keys / crystals   glockenspiel + piano at the struck pitch (a little louder when both racers strike the note)
  snowmen's timpani     timpani + contrabass pizzicato + piano at the bass pitch, on every bass note
  golden sleigh bells   the off-beat chord (piano + glockenspiel) + sleigh-bell shake (louder in the choruses)
  mechanisms            start gates, countdown lamps, trapdoor, bucket, ski-lift motor, take-off, snow bank, finish tape,
                        the star (final chord: the tree lights up)"""
import sys, os, json, glob
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
LAY = json.load(open('layout.json')); LEAD = json.load(open('lead.json')); B = LAY['balls']
I['vln'] = Inst('Strings/Violin Section/susVib'); I['vcp'] = Inst('Strings/Cello Section/pizzT')
I['hnsus'] = Inst('Brass/F Horn/sus')
TIMP = Hit(sorted(glob.glob(P('Percussion/Timpani/Timpani1_Hit_*'))))
from collections import Counter
cnt = Counter(h['ev'] for b in B.values() for h in b['hits'] if h['ev'] is not None)
for i, (t, p, v, d, bar, pos, sec) in enumerate(LEAD):
    n = cnt.get(i, 0); assert n >= 1, i
    g = 1.0 if n == 1 else 1.3
    fin = i == len(LEAD) - 1
    play('glock', t, p + 12, v, 0.2 * g, 0.1, dur=1.2 if not fin else 3.5, rel=0.5)
    play('pno', t, p, v, 0.2 * g, -0.05, dur=max(0.25, d) if not fin else 3.5, rel=0.3)
    play('xylo', t, p + 12, v, 0.05 * g, 0.2)
    if sec[0] == 'C': play('vln', t, p, v, 0.07, 0.2, dur=d * 0.95 if not fin else 3.5, rel=0.25)
for t, p, bar, pos, ch in EV['bass']:
    fin = abs(t - EV['final']) < 1e-3
    hit(TIMP, t, 96, 0.22, -0.1)
    play('cb', t, p, 96, 0.3, -0.05); play('pno', t, p, 92, 0.22, -0.15, dur=0.45 if not fin else 3.5, rel=0.3)
    play('vcp', t, p + 12, 88, 0.1, -0.2)
for t, ps, bar, pos, ch in EV['stabs']:
    ch_sec = bar >= 17 and bar < 33 or bar >= 49
    for p in ps:
        play('pno', t, p, 80, 0.07, 0.25, dur=0.2, rel=0.12); play('glock', t, p + 24, 70, 0.035, 0.35, dur=0.35, rel=0.2)
    hit(SLEIGH, t, 100 if ch_sec else 70, 0.2 if ch_sec else 0.1, 0.3)
FINAL = EV['final']
for p in (56, 60, 63, 68):
    play('vln', FINAL, p + 12, 100, 0.09, 0.2, dur=3.5, rel=1.2); play('hnsus', FINAL, p, 100, 0.08, -0.2, dur=3.2, rel=1.2)
    play('glock', FINAL, p + 24, 100, 0.06, 0.3, dur=3.5, rel=1.0)
for kind, t in LAY['sfx']:
    if kind == 'gate': hit(WOOD, t, 100, 0.3, -0.2); hit(SLAP, t + 0.01, 90, 0.2, 0.0)
    elif kind == 'trap': hit(WOOD, t, 95, 0.25, 0.2); hit(RAT, t + 0.02, 80, 0.15, 0.2)
    elif kind == 'bucket': hit(LOGLO, t, 90, 0.3, 0.1)
    elif kind == 'lift':
        el = LAY['ev']['elevator']; k = 0.0
        while el['t0'] + k < el['t1']: hit(RAT, el['t0'] + k, 70, 0.08, 0.3); k += 0.42
    elif kind == 'tip': hit(WOOD, t, 90, 0.2, 0.3)
    elif kind == 'whoosh': hit(SHK, t, 100, 0.3, 0.0); hit(SHK, t + 0.25, 90, 0.25, 0.2)
    elif kind == 'tape': hit(SLAP, t, 90, 0.22, 0.25)
    elif kind == 'star': hit(CRASH, t, 110, 0.22, 0.2); hit(BT, t + 0.08, 110, 0.35, -0.2); hit(SLEIGH, t + 0.02, 120, 0.3, 0.3)
# countdown lamps: a click as each lamp switches on (the gate's relay)
g0 = LAY['ev']['gate']['t']
for dt in (-0.75, -0.5, -0.25): hit(CLAV, g0 + dt, 90, 0.18, 0.0)
# tree lights: a soft bell-tree sweep while the chain lights from the top down
hit(BT, FINAL + 0.6, 90, 0.25, 0.3); hit(BT, FINAL + 1.3, 80, 0.2, -0.3)
L_ = int(2.4 * SR); tt = np.arange(L_) / SR
ir = np.stack([rng.normal(0, 1, L_), rng.normal(0, 1, L_)], 1) * np.exp(-tt / 0.55)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.55], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.25 * wet
fade = int(2.0 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
