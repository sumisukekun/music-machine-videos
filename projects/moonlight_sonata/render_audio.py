"""Moonlight Sonata - acoustic render with VSCO-2 CE (CC0) / Ivy Audio upright piano (redistributable, see Info.txt):
every struck glass lotus, silver key, bronze bowl and piano key sounds as a soft piano note (the machine plays on a piano);
a whisper of glockenspiel two octaves up on the silver melody keys, a cello-section pad under the bass bowls.
python3 render_audio.py /path/to/VSCO-2-CE -> music.wav"""
import sys, os, json, glob, re
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
src = src.replace("DUR = EV['final'] + 4.5", "DUR = EV['final'] + 7.6")
exec(src)
I['csus'] = Inst('Strings/Cello Section/susvib')
I['harp'] = Inst('Strings/Harp', offset=0)          # KS Harp files are named at sounding pitch
for t, d, role, p, v in json.load(open('notes.json')):
    if role == 'tri':
        play('pno', t, p, v, 0.40, 0.05 - 0.004 * (p - 60), dur=d, rel=0.5)
        play('harp', t, p + 12, v, 0.035, 0.25, dur=min(d, 1.5), rel=0.4)
    elif role == 'mel':
        play('pno', t, p, v, 0.62, 0.12, dur=d, rel=0.6)
        play('glock', t, p + 24, v, 0.028, 0.3, dur=1.2, rel=0.5)
    elif role == 'bass':
        play('pno', t, p, v, 0.50, -0.2, dur=d, rel=0.7)
        play('csus', t, p + 12, v, 0.05, -0.25, dur=d, rel=0.8)
    elif role == 'chord':
        play('pno', t, p, v, 0.48, -0.1 + 0.01 * (p - 50), dur=d, rel=1.2)
        if p < 50: play('csus', t, p + 12, v, 0.05, -0.25, dur=d, rel=1.5)
# the machine's own (unpitched, very soft) sounds: cradle hatches, see-saw clacks, catapult, lighthouse lamp
LAYF = 'layout.json'
if os.path.exists(LAYF):
    for kind, t in json.load(open(LAYF))['sfx']:
        if kind == 'hatch': hit(WOOD, t, 50, 0.05, 0.2)
        elif kind == 'seesaw': hit(WOOD, t + 0.01, 55, 0.05, -0.1)
        elif kind == 'catapult': hit(CLAV, t + 0.01, 60, 0.06, 0.0)
        elif kind == 'lamp': hit(BT, t + 0.05, 55, 0.08, 0.2)
        elif kind == 'settle': hit(WOOD, t, 35, 0.03, 0.0)
# a large, soft room
L = int(3.2 * SR); tt = np.arange(L) / SR
ir = np.stack([rng.normal(0, 1, L), rng.normal(0, 1, L)], 1) * np.exp(-tt / 0.75)[:, None]
for c in range(2): ir[:, c] = lfilter([0.25], [1, -0.75], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.30 * wet
fade = int(2.4 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 1.5
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-20dB:ratio=2:attack=10:release=200,loudnorm=I=-15:TP=-1.5:LRA=11" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
