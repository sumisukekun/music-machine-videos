"""The Blue Danube - acoustic render with VSCO-2 CE (CC0 samples only).  python3 render_audio.py /path/to/VSCO-2-CE -> music.wav
What you hear is what you see:
  ice keys (the couple)      glockenspiel + soft piano at the struck pitch(es); violins sing the top line, violas the inner
                             voice - only the pitches the balls actually strike (layout.json 'struck')
  gold bells (beat 1)        piano bass + contrabass / cello pizzicato (+ tuba in the second pass)
  ice bells (beats 2 and 3)  glockenspiel two octaves up + viola spiccato at the chord pitch (+ horns in the second pass)
  mechanisms                 chandelier release, the crack, sled bells, snowbank, revolving door, slide, reunion, see-saw,
                             the great chandelier bell"""
import sys, os, json, glob, re
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
LAY = json.load(open('layout.json')); LEAD = json.load(open('lead.json'))
STRUCK = {int(k): sorted(set(p for _, p, _ in v)) for k, v in LAY['struck'].items()}
TN = [l[0] for l in LEAD]
def ev_at(t):
    k = min(range(len(TN)), key=lambda i: abs(TN[i] - t)); return k if abs(TN[k] - t) < 1e-4 else None
I['vln'] = Inst('Strings/Violin Section/susVib'); I['vla'] = Inst('Strings/Viola Section/spic')
I['vcsus'] = Inst('Strings/Cello Section/susvib'); I['vcp'] = Inst('Strings/Cello Section/pizzT')
I['hnst'] = Inst('Brass/F Horn/stac'); I['hnsus'] = Inst('Brass/F Horn/sus'); I['tuba'] = Inst('Brass/Tuba/stac')
TRI = Hit(sorted(glob.glob(P('Percussion/Triangle3-Hit_*'))))
TIMP = Hit(sorted(glob.glob(P('Percussion/Timpani/Timpani1_Hit_*'))))
skipped = 0
for t, d, role, p, v in json.load(open('notes.json')):
    if role in ('ice', 'vla'):
        i = ev_at(t)
        if i is None or p not in STRUCK.get(i, []): skipped += 1; continue
    if role == 'ice':
        play('glock', t, p + 12, v, 0.26, 0.15, dur=1.2, rel=0.6); play('pno', t, p + 12, v, 0.12, 0.1, dur=d, rel=0.3)
        play('pno', t, p, v, 0.22, 0.05, dur=d, rel=0.3)
    elif role == 'vln': play('vln', t, p, v, 0.24, -0.15, dur=d, rel=0.25)
    elif role == 'vln8': play('vln', t, p, v, 0.07, -0.25, dur=d, rel=0.25)
    elif role == 'vla': play('vln', t, p, v, 0.10, 0.2, dur=d, rel=0.25)
    elif role == 'bell': play('pno', t, p, v, 0.24, -0.1, dur=0.8, rel=0.35); play('glock', t, p + 36, v, 0.03, -0.2, dur=0.6, rel=0.3)
    elif role == 'cb': play('cb', t, p, v, 0.3, -0.05); play('vcp', t, p, v, 0.13, -0.2)
    elif role == 'tuba': play('tuba', t, p, v, 0.22, 0.0)
    elif role == 'tink': play('glock', t, p + 24, v, 0.06, 0.3, dur=0.45, rel=0.25); play('pno', t, p, v, 0.06, 0.25, dur=0.22, rel=0.12)
    elif role == 'spic': play('vla', t, p, v, 0.10, 0.25)
    elif role == 'hnst': play('hnst', t, p, v, 0.10, -0.3)
    elif role == 'cbsus': play('vcsus', t, p, v, 0.28, -0.1, dur=d, rel=0.8); play('cb', t, p, v, 0.3, 0.0)
    elif role == 'vsus': play('vln', t, p, v, 0.14, 0.2, dur=d, rel=0.8)
    elif role == 'hnsus': play('hnsus', t, p, v, 0.16, -0.3, dur=d, rel=0.8)
    elif role == 'bigbell':
        for pp, g in ((p - 12, 0.35), (p, 0.3), (p + 12, 0.18), (p + 19, 0.08)): play('pno', t, pp, v, g, 0.0, dur=d, rel=1.0)
        play('glock', t, p + 36, v, 0.12, 0.1, dur=2.0, rel=1.0); hit(TIMP, t, 120, 0.35, 0.0); hit(CRASH, t, 110, 0.22, 0.2)
        hit(BT, t + 0.05, 110, 0.3, -0.25)
print('melody notes not struck (dropped from the audio):', skipped)
for kind, t in LAY['sfx']:
    if kind == 'release': hit(BT, t, 90, 0.22, 0.2)
    elif kind == 'crack': hit(RAT, t, 120, 0.35, -0.2); hit(SLAP, t + 0.01, 110, 0.3, 0.0); hit(LOGLO, t + 0.02, 100, 0.3, -0.1)
    elif kind == 'sled_land': hit(LOGLO, t, 90, 0.3, -0.3)
    elif kind == 'jingle': hit(SLEIGH, t, 70, 0.12, -0.35)
    elif kind == 'stop': hit(BD, t, 100, 0.35, -0.2); hit(SHK, t + 0.01, 100, 0.25, -0.2)
    elif kind == 'door_in': hit(WOOD, t, 90, 0.2, 0.3)
    elif kind == 'door_out': hit(WOOD, t, 90, 0.2, 0.35)
    elif kind == 'slide_in': hit(SHK, t, 80, 0.15, 0.3)
    elif kind == 'reunion': hit(BT, t, 110, 0.28, 0.0); hit(TRI, t + 0.02, 100, 0.2, 0.2)
    elif kind == 'seesaw': hit(BD, t, 120, 0.45, 0.0); hit(LOGLO, t + 0.003, 110, 0.35, 0.1)
    elif kind == 'launch': hit(SLAP, t + 0.02, 100, 0.25, 0.1)
    elif kind == 'land_cups': hit(WOOD, t, 70, 0.15, 0.0)
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
