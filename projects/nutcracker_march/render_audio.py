"""The Nutcracker - March: acoustic render with VSCO-2 CE (CC0 samples only).
python3 render_audio.py /path/to/VSCO-2-CE -> music.wav
What you hear is what you see: every pitched sound = a marble striking a key at that moment, played by the instrument
of the score part the note comes from (voices.json keeps the source part of every note):
  clarinets / trumpets / horns (fanfare)       trumpet + horn staccato (+ a glockenspiel tick on the captain's bell plates)
  violins / violas / cellos (tune, runs)       string spiccato;  pizzicato cellos + basses (bass line)
  bassoon / trombones / tuba                   trombone / tuba staccato
  flute                                        flute (short) + glockenspiel / xylophone tick (mouse marbles, shells)
  cheese keys add a soft marimba tick, red drums a soft drum-skin tick (the marble's impact)."""
import sys, os, json, glob, re
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
I['trp'] = Inst('Brass/Trumpet/stac'); I['hrn'] = Inst('Brass/F Horn/stac'); I['tbn'] = Inst('Brass/Tenor Trombone/stac'); I['tba'] = Inst('Brass/Tuba/stac')
I['vln'] = Inst('Strings/Violin Section/Spic'); I['vla'] = Inst('Strings/Viola Section/spic'); I['vcs'] = Inst('Strings/Cello Section/spic')
I['vcp'] = Inst('Strings/Cello Section/pizzT'); I['flu'] = Inst('Woodwinds/Flute/susvib'); I['obo'] = Inst('Woodwinds/Oboe/Sus')
I['vsus'] = Inst('Strings/Violin Section/susVib'); I['tbs'] = Inst('Brass/Tenor Trombone/sus'); I['trs'] = Inst('Brass/Trumpet/sus')
VOI = json.load(open('voices.json')); FIN = EV['final']
_ATT = {}


def attack(x):
    """time (s, at the sample's own rate) for the sample to reach 35 % of its peak level: the sample is started that much
    earlier (at most 60 ms) so that its audible onset falls on the visible strike"""
    k = id(x)
    if k not in _ATT:
        e = np.convolve(np.abs(x).max(1), np.ones(64) / 64, 'same'); _ATT[k] = max(0.0, np.argmax(e > 0.35 * e.max()) / SR - 0.004)
    return _ATT[k]


def play(n, t, p, v, g, pan, dur=None, rel=0.15):
    x, ra = I[n].get(p, v); a = min(0.06, attack(x) / ra)
    place(x, ra, t - a, g * (0.5 + 0.5 * v / 127), pan, None if dur is None else dur + a, rel)
SNARE = Hit(sorted(glob.glob(P('Percussion/Snare2-HitNS_*'))))
PAN = dict(cap=0.1, s2=-0.15, s3=0.25, drum=-0.1, shell=0.35, king=0.15, m2=-0.25, m3=0.3, rat=-0.2, army=-0.35)


def sound(v, tr, t, p, d, vel, grace):
    pan = PAN[v]; last = t >= FIN - 1e-3
    dd = max(0.12, min(d, 0.5)) if not last else 3.0
    if tr in ('cl1', 'cl2', 'tp1', 'tp2'):
        if v in ('king',) and tr in ('cl1', 'cl2'):
            play('obo', t, p, vel, 0.30, pan, dur=min(0.22, dd), rel=0.08); play('trp', t, p, vel, 0.16, pan)
        else:
            play('trp', t, p, vel, 0.42, pan); play('hrn', t, p, vel, 0.12, pan)
            if last: play('trs', t, p, vel, 0.25, pan, dur=dd, rel=1.0)
    elif tr == 'hn1':
        play('hrn', t, p, vel, 0.45, pan)
    elif tr in ('tb1', 'tb2', 'bn1'):
        play('tbn', t, p, vel, 0.40 if tr != 'bn1' else 0.3, pan)
        if tr == 'bn1': play('vcs', t, p, vel, 0.18, pan)
    elif tr in ('tu1', 'tu2'):
        play('tba', t, p, vel, 0.5, pan)
    elif tr in ('vn1', 'vn2'):
        play('vln', t, p, vel, 0.42, pan)
        if last: play('vsus', t, p, vel, 0.25, pan, dur=dd, rel=1.2)
    elif tr == 'va':
        play('vla', t, p, vel, 0.4, pan)
    elif tr == 'vc':
        if v in ('shell',): play('vcs', t, p, vel, 0.45, pan)
        else: play('vcp', t, p, vel, 0.5, pan); play('vcs', t, p, vel, 0.12, pan)
    elif tr == 'cb':
        play('cb', t, p, vel, 0.55, pan)
    elif tr == 'fl1':
        play('flu', t, p, vel, 0.32 if not grace else 0.2, pan, dur=min(0.18, dd) if not grace else 0.05, rel=0.06)
    # the struck material (soft)
    if v == 'cap': play('glock', t, p, vel, 0.05, pan, dur=0.4, rel=0.3)
    elif v in ('king', 'm2', 'm3'): play('mar', t, p, vel, 0.10, pan)
    elif v == 'army': play('xylo', t, p, vel, 0.05 if not grace else 0.03, pan)
    elif v == 'shell': play('glock', t, p, vel, 0.06, pan, dur=0.5, rel=0.3)
    elif v in ('s2', 's3'): hit(SNARE, t, 25, 0.035, pan)


for v, evs in VOI.items():
    for e in evs:
        vel = 104 if e['t'] >= FIN - 1e-3 else (88 if e['sec'] in ('fanfare', 'finale', 'end') else 80)
        for p, tr in zip(e['p'], e['tr'] if len(e['tr']) == len(e['p']) else e['tr'] * len(e['p'])):
            sound(v, tr, e['t'], p, e['d'], vel, e['grace'])
LAY = json.load(open('layout.json'))
for kind, t in LAY['sfx']:
    if kind == 'release': hit(WOOD, t, 70, 0.08, 0.2)
    elif kind == 'cup': hit(WOOD, t + 0.005, 50, 0.05, 0.0)
    elif kind == 'cup_open': hit(WOOD, t, 45, 0.04, 0.1)
hit(BT, FIN + 0.05, 100, 0.3, -0.2); hit(CRASH, FIN, 80, 0.16, 0.0); hit(SLEIGH, FIN + 0.08, 95, 0.2, 0.3)
for k in range(3): hit(CRASH, FIN + 0.25 * k + 0.2, 55, 0.05, -0.4 + 0.4 * k)       # the fireworks
# hall
L_ = int(1.6 * SR); tt = np.arange(L_) / SR
ir = np.stack([rng.normal(0, 1, L_), rng.normal(0, 1, L_)], 1) * np.exp(-tt / 0.32)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.65], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.2 * wet
fade = int(1.8 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-20dB:ratio=2.2:attack=8:release=140,loudnorm=I=-14:TP=-1.3:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
