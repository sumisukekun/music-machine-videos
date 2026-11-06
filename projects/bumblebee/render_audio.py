"""Flight of the Bumblebee - acoustic render with VSCO-2 CE (CC0): marimba (the ball on the wax cells) doubled by
violin-section spiccato (the buzz), piano + pizzicato on the chords, horn on the long counter-melody notes."""
import sys, os, json, glob
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
I['vspic'] = Inst('Strings/Violin Section/Spic'); I['vlspic'] = Inst('Strings/Viola Section/spic')
I['cpizz'] = Inst('Strings/Cello Section/pizzT'); I['hn'] = Inst('Brass/F Horn/sus'); I['vsus'] = Inst('Strings/Violin Section/susVib')
TIMP = [Hit(sorted(glob.glob(P(f'Percussion/Timpani/Timpani{k}_Hit_*')))) for k in range(1, 6)]
for t, d, role, p, v in json.load(open('notes.json')):
    if role in ('lead', 'lead2'):
        g = 1.0 if role == 'lead' else 0.6
        play('mar', t, p, v, 0.55 * g, -0.1)
        if p >= 55: play('vspic', t, p, v, 0.30 * g, 0.25)
        else: play('vlspic', t, p + 12, v, 0.18 * g, 0.2); play('cpizz', t, p, v, 0.25 * g, -0.2)
        if p >= 76: play('xylo', t, p, v, 0.06 * g, 0.3)
    elif role == 'pizz': play('cpizz', t, p, v, 0.3, -0.2)
    elif role == 'chord': play('pno', t, p, v, 0.20, 0.1, dur=d, rel=0.12)
    elif role == 'bass': play('cpizz', t, p, v, 0.38, -0.2); play('cb', t, p, v, 0.45, -0.05)
    elif role == 'horn': play('hn', t, p, v, 0.42, -0.3, dur=d, rel=0.2); play('vsus', t, p, v, 0.10, 0.3, dur=d, rel=0.2)
    elif role == 'timp': hit(TIMP[1], t, v, 0.55, -0.05)
    elif role == 'crash': hit(CRASH, t, v, 0.32, 0.2)
L = int(1.8 * SR); tt = np.arange(L) / SR
ir = np.stack([rng.normal(0, 1, L), rng.normal(0, 1, L)], 1) * np.exp(-tt / 0.4)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.6], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.2 * wet
fade = int(2.0 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
