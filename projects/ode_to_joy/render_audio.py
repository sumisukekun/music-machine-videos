"""Ode to Joy - acoustic render with VSCO-2 CE (CC0 samples only).  python3 render_audio.py /path/to/VSCO-2-CE -> music.wav
What you hear is what you see (every sound starts on a visible strike, from layout.json):
  melody keys       glockenspiel + piano at the struck pitch; the choir 'sings' it with strings that grow with the number of
                    balls on the key (1: cellos, from 5 balls: + violins, from 9: + violins 8va, from bar 25: + horns)
  bass bells        piano + contrabass / cello pizzicato (+ tuba from bar 17)
  pair bells        glockenspiel two octaves up + viola spiccato, the two chord tones struck
  golden tubes      piano + glockenspiel bell sound of each struck tube, louder with the number of balls; brass and strings
                    hold the chord of that beat (the tubes' pitches)
  mechanisms        podium trapdoor, gates, see-saw, curtain, fireworks"""
import sys, os, json, glob
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
LAY = json.load(open('layout.json')); LEAD = json.load(open('lead.json')); B = LAY['balls']
I['vln'] = Inst('Strings/Violin Section/susVib'); I['vla'] = Inst('Strings/Viola Section/spic')
I['vcsus'] = Inst('Strings/Cello Section/susvib'); I['vcp'] = Inst('Strings/Cello Section/pizzT')
I['hnsus'] = Inst('Brass/F Horn/sus'); I['tbsus'] = Inst('Brass/Tenor Trombone/sus'); I['tuba'] = Inst('Brass/Tuba/stac')
I['tpsus'] = Inst('Brass/Trumpet/sus')
TRI = Hit(sorted(glob.glob(P('Percussion/Triangle3-Hit_*'))))
TIMP = Hit(sorted(glob.glob(P('Percussion/Timpani/Timpani1_Hit_*'))))
# melody: one sound per struck key (the soloist's hit carries the number of balls on it)
for h in B['solo']['hits']:
    i = h['ev']; t, p, v, d, bar, pos = LEAD[i]; n = h['lanes']
    g = 1.0 + 0.35 * np.log2(n)
    play('glock', t, p + 12, v, 0.22 * g, 0.1, dur=1.0, rel=0.5); play('pno', t, p, v, 0.2 * g, -0.05, dur=d, rel=0.25)
    play('vcsus', t, p - 12, v, 0.2, -0.2, dur=d * 0.97, rel=0.25)
    if n >= 5: play('vln', t, p, v, 0.15 + 0.012 * n, 0.15, dur=d * 0.97, rel=0.25)
    if n >= 9: play('vln', t, p + 12, v, 0.08, 0.3, dur=d * 0.97, rel=0.25)
    if bar >= 25: play('hnsus', t, p, v, 0.12, -0.3, dur=d * 0.97, rel=0.25)
for h in B['bass']['hits']:
    t, p = h['t'], h['pitch']; bar = int(np.interp(t, [l[0] for l in LEAD], [l[4] for l in LEAD]))
    fin = np.linalg.norm(h['vout']) == 0
    play('pno', t, p, 96, 0.3, -0.1, dur=2.5 if fin else 0.8, rel=0.4); play('cb', t, p, 96, 0.3, -0.05); play('vcp', t, p + 12, 90, 0.12, -0.2)
    if bar >= 17: play('tuba', t, p, 90, 0.16, 0.0)
    if fin: play('vcsus', t, p + 12, 100, 0.25, -0.1, dur=3.2, rel=1.0)
for n in ('pairlo', 'pairhi'):
    for h in B[n]['hits']:
        t, p = h['t'], h['pitch']
        play('glock', t, p + 24, 80, 0.07, 0.3, dur=0.45, rel=0.25); play('pno', t, p, 80, 0.08, 0.25, dur=0.3, rel=0.15)
        play('vla', t, p, 84, 0.1, 0.3)
# golden tubes: one sound per (beat, pitch), louder with the number of balls; brass/strings hold that beat's chord
from collections import Counter
cnt = Counter((round(t, 4), p) for t, p in LAY['bellhits'])
beats = sorted(set(t for t, p in cnt))
fin_t = round(LEAD[-1][0], 4)
for (t, p), c in cnt.items():
    g = 0.6 + 0.12 * c; fin = abs(t - fin_t) < 1e-3
    play('pno', t, p, 100, 0.13 * g, 0.0, dur=2.5 if fin else 0.7, rel=0.5); play('pno', t, p - 12, 96, 0.08 * g, 0.0, dur=0.7, rel=0.4)
    play('glock', t, p, 100, 0.07 * g, 0.2, dur=0.9, rel=0.4)
for k, t in enumerate(beats):
    ps = sorted(p for (tt, p) in cnt if tt == t); fin = abs(t - fin_t) < 1e-3
    d = (beats[k + 1] - t) if k + 1 < len(beats) else 3.5
    for p in ps:
        play('tbsus', t, p - 24, 96, 0.09, -0.25, dur=d * 0.95, rel=0.3 if not fin else 1.2)
        play('vln', t, p, 90, 0.07, 0.25, dur=d * 0.95, rel=0.3 if not fin else 1.2)
        if fin: play('tpsus', t, p - 12, 110, 0.12, 0.1, dur=3.0, rel=1.2); play('hnsus', t, p - 12, 110, 0.12, -0.2, dur=3.0, rel=1.2)
    if fin: hit(TIMP, t, 120, 0.4, 0.0); hit(CRASH, t, 110, 0.25, 0.2)
EV = LAY['ev']
for kind, t in LAY['sfx']:
    if kind == 'release': hit(WOOD, t, 90, 0.25, 0.0); hit(BT, t + 0.05, 80, 0.2, 0.3)
    elif kind == 'gate': hit(WOOD, t, 100, 0.3, -0.2); hit(CLAV, t + 0.01, 80, 0.12, -0.2)
    elif kind == 'gate_small': hit(WOOD, t, 75, 0.15, 0.3)
    elif kind == 'seesaw': hit(LOGLO, t + 0.003, 110, 0.3, -0.1)
    elif kind == 'launch': hit(SLAP, t, 90, 0.2, -0.1)
    elif kind == 'curtain': hit(SHK, t, 80, 0.2, 0.0); hit(BT, t + 0.3, 100, 0.25, 0.0)
FINAL = LEAD[-1][0]
for b in range(7):
    tb = FINAL + 0.05 + 0.35 * b
    if b: hit(BD, tb, 110, 0.35, -0.6 + 0.2 * b); hit(CRASH, tb + 0.02, 90, 0.12, -0.6 + 0.2 * b)
    hit(SLEIGH, tb + 0.05, 70, 0.08, 0.3)
# hall
L_ = int(2.6 * SR); tt = np.arange(L_) / SR
ir = np.stack([rng.normal(0, 1, L_), rng.normal(0, 1, L_)], 1) * np.exp(-tt / 0.65)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.55], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.28 * wet
fade = int(2.0 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
