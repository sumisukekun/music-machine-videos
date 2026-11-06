"""Pomp and Circumstance Trio - acoustic render with VSCO-2 CE (CC0): brass, strings, piano/glock 'pipes', timpani, snare."""
import sys, os, json, math, glob
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
I['tpt'] = Inst('Brass/Trumpet/sus'); I['hrn'] = Inst('Brass/F Horn/sus'); I['tbn'] = Inst('Brass/Tenor Trombone/sus')
I['tuba'] = Inst('Brass/Tuba/sus'); I['vln'] = Inst('Strings/Violin Section/susVib'); I['csus'] = Inst('Strings/Cello Section/susvib')
TIMP = [Hit(sorted(glob.glob(P(f'Percussion/Timpani/Timpani{k}_Hit_*')))) for k in range(1, 6)]
TROLL = Hit(sorted(glob.glob(P('Percussion/Timpani/Rolls/*'))) or sorted(glob.glob(P('Percussion/Timpani/Timpani1_Hit_*'))))
SROLL = Hit(sorted(glob.glob(P('Percussion/Snare2-rollNS_*'))))
SN = Hit(sorted(glob.glob(P('Percussion/Snare2-HitNS_*'))))
def roll(h, t, d, v, g, pan):
    x = h.get(v); L = int(min(len(x), (d + 0.25) * SR)); x = x[:L].copy()
    r = min(L, int(0.25 * SR)); x[-r:] *= np.linspace(1, 0, r)[:, None]
    ramp = np.linspace(0.45, 1.0, L)[:, None]; place(x * ramp, 1.0, t, g * (0.55 + 0.45 * v / 127), pan)
TB = EV['secs'][2][1] - 1.0
_play = play
def play(n, t, p, v, g, pan, dur=None, rel=0.15):
    # quiet Trio: stay on the soft sample layer, crescendo by gain (avoids sudden layer jumps)
    if EV['secs'][1][1] - 0.3 <= t < TB and n not in ('glock', 'pno', 'cb'):
        vl = min(v, 58); return _play(n, t, p, vl, g * (v / vl) ** 1.3, pan, dur, rel)
    return _play(n, t, p, v, g, pan, dur, rel)
for t, d, role, p, v in json.load(open('notes.json')):
    if role == 'bell':
        play('glock', t, p + 24, v, 0.20, 0.15, dur=max(d, 0.8), rel=0.8); play('pno', t, p + 12, v, 0.20, 0.05, dur=max(d, 0.6), rel=0.8)
    elif role == 'tpt': play('tpt', t, p, v, 0.30, 0.15, dur=d, rel=0.18)
    elif role == 'hrn': play('hrn', t, p, v, 0.30, -0.25, dur=d, rel=0.3)
    elif role == 'tbn': play('tbn', t, p, v, 0.26, 0.25, dur=d, rel=0.25)
    elif role == 'tuba': play('tuba', t, p, v, 0.40, -0.05, dur=d, rel=0.3)
    elif role == 'vln': play('vln', t, p, v, 0.30, 0.3, dur=d, rel=0.35)
    elif role == 'vsus': play('vln', t, p, v, 0.12, -0.3, dur=d, rel=0.4)
    elif role == 'cello': play('csus', t, p, v, 0.36, -0.2, dur=d, rel=0.35)
    elif role == 'cbp': play('cb', t, p, v, 0.55, 0.0)
    elif role == 'pbass': play('pno', t, p, v, 0.35, -0.1, dur=d, rel=1.0)
    elif role == 'timp': hit(TIMP[{48: 2, 45: 1, 43: 1, 36: 0}.get(p, 1)], t, v, 0.5, -0.05)
    elif role == 'troll': roll(TROLL, t, d, v, 0.35, 0.0)
    elif role == 'sroll': roll(SROLL, t, d, v, 0.30, 0.2)
    elif role == 'bd': hit(BD, t, v, 0.5, 0.0)
    elif role == 'crash': hit(CRASH, t, v, 0.32, 0.25)
    elif role == 'bt': hit(BT, t, v, 0.3, -0.3)
# the machine: crank wind-up at start, gate click when the marble is released
hit(RAT, 0.1, 85, 0.25, -0.4); hit(WOOD, EV['secs'][1][1] - 0.8, 90, 0.2, -0.3)
L = int(2.8 * SR); tt = np.arange(L) / SR
ir = np.stack([rng.normal(0, 1, L), rng.normal(0, 1, L)], 1) * np.exp(-tt / 0.75)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.5], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.30 * wet
fade = int(2.0 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
import subprocess
AC = 'acompressor=threshold=-16dB:ratio=2:attack=10:release=150'
r = subprocess.run(['ffmpeg', '-hide_banner', '-i', 'music_raw.wav', '-af', AC + ',loudnorm=I=-14:TP=-1.2:LRA=20:print_format=json', '-f', 'null', '-'], capture_output=True, text=True).stderr
j = json.loads(r[r.rindex('{'):r.rindex('}') + 1])
ln = f"loudnorm=I=-14:TP=-1.2:LRA=20:linear=true:measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}:measured_thresh={j['input_thresh']}:offset={j['target_offset']}"
os.system(f'ffmpeg -v error -y -i music_raw.wav -af "{AC},{ln}" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
