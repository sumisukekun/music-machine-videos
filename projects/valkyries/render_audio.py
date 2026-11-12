"""Ride of the Valkyries - acoustic render with VSCO-2 CE (CC0 samples only).  python3 render_audio.py /path/to/VSCO-2-CE -> music.wav
What you hear is what you see (every sound starts on a visible strike, from layout.json):
  rider keys       glockenspiel at the struck pitch (the strike) + the theme in the brass: horns for the upper riders, trombones
                   for the lower ones (trumpets join the upper riders in B major); one sound per struck key
  final bells      each rider's bell: its pitch in brass + glockenspiel + piano, held
  thunder drums    timpani (tuned to the struck pitch) + contrabass pizzicato + piano an octave up (+ tuba from bar 9); the
                   string section (tremolo) holds the chord of that beat (source harmony, events.json) until the next drum
  mechanisms       gates, start bar, lever, drawbridge, castle gate; thunder with every lightning bolt"""
import sys, os, json, glob, re
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ra_base.py')).read()
src = src[:src.index("for t, d, role, p, v in json.load(open('notes.json')):")]
exec(src)
LAY = json.load(open('layout.json')); LEAD = json.load(open('lead.json')); B = LAY['balls']
I['hn'] = Inst('Brass/F Horn/sus'); I['tb'] = Inst('Brass/Tenor Trombone/sus'); I['tp'] = Inst('Brass/Trumpet/sus')
I['tuba'] = Inst('Brass/Tuba/stac'); I['vlat'] = Inst('Strings/Viola Section/trem'); I['vct'] = Inst('Strings/Cello Section/trem')
I['vln'] = Inst('Strings/Violin Section/susVib')
# timpani: five drums, their pitch measured from the samples (fundamental, midi); tuned by resampling to the struck pitch
TIMP = [(41.4, sorted(glob.glob(P('Percussion/Timpani/Timpani1_Hit_*')))), (49.3, sorted(glob.glob(P('Percussion/Timpani/Timpani5_Hit_*')))),
        (52.4, sorted(glob.glob(P('Percussion/Timpani/Timpani4_Hit_*')))), (53.6, sorted(glob.glob(P('Percussion/Timpani/Timpani2_Hit_*')))),
        (56.4, sorted(glob.glob(P('Percussion/Timpani/Timpani3_Hit_*'))))]
TH = [Hit(f) for _, f in TIMP]
ROLL = Hit(sorted(glob.glob(P('Percussion/Timpani/Rolls/Timpani1_Roll_*'))))
def timp(t, p, v, g, pan):
    q = p + 12 if p < 40 else p
    k = int(np.argmin([abs(q - m) for m, _ in TIMP])); ratio = 2 ** ((q - TIMP[k][0]) / 12)
    place(TH[k].get(v), ratio, t, g * (0.55 + 0.45 * v / 127), pan)
EVJ = json.load(open('events.json')); PCS = EVJ['pcs']; LDUR = {round(l[0], 4): l[3] for l in LEAD}
T9 = EVJ['bar_t']['9']; T17 = EVJ['bar_t']['17']; FINAL = EVJ['final']
# ---- riders
for n, b in B.items():
    if b['role'] != 'rider': continue
    j = int(n[1:]); pan = -0.5 + j / 7
    for h in b['hits']:
        if h['kind'] == 'cup': continue
        t, p = h['t'], h['pitch']
        if h['kind'] == 'final':
            play('glock', t, p + 12, 100, 0.12, pan, dur=3.0, rel=1.2); play('pno', t, p, 100, 0.12, pan, dur=3.5, rel=1.2)
            play('hn' if p >= 55 else 'tb', t, p, 112, 0.2, pan, dur=3.4, rel=1.4)
            if p >= 62: play('tp', t, p, 110, 0.12, pan, dur=3.4, rel=1.4)
            continue
        d = LDUR.get(round(t, 4), 0.3); solo = t < T9 - 0.5
        g = 1.0 if solo else 0.5
        play('glock', t, p + 12, 90, 0.16 * g, pan, dur=0.8, rel=0.4)
        if p >= 52 and j % 2 == 0 or solo:
            play('hn', t, p, 104, 0.3 * g, pan, dur=d * 0.97, rel=0.12)
            if t > T17 - 0.1: play('tp', t, p, 104, 0.14 * g, pan, dur=d * 0.95, rel=0.12)
        else:
            play('tb', t, p, 104, 0.3 * g, pan, dur=d * 0.97, rel=0.12)
# ---- thunder ball: drums + the harmony of the beat in the strings
hb = B['bass']['hits']; BE = EVJ['bass']
for k, h in enumerate(hb):
    t, p = h['t'], h['pitch']; fin = h['kind'] == 'final'
    timp(t, p, 120 if fin else 105, 0.55, -0.1)
    play('cb', t, p, 100, 0.35, -0.05); play('pno', t, p + 12, 96, 0.18, -0.1, dur=2.5 if fin else 0.7, rel=0.4)
    if t > T9 - 0.1: play('tuba', t, p, 100, 0.22, 0.0)
    ch = BE[k][4]; pcs = PCS[ch]
    d = (hb[k + 1]['t'] - t) if k + 1 < len(hb) else 4.0
    for pc in pcs:
        hi = 60 + (pc - 60) % 12          # C4..B4
        lo = 48 + (pc - 48) % 12          # C3..B3
        play('vlat', t, hi, 96, 0.07, 0.3, dur=d * 0.97, rel=0.3 if not fin else 1.4)
        play('vct', t, lo, 96, 0.08, -0.3, dur=d * 0.97, rel=0.3 if not fin else 1.4)
        if fin: play('vln', t, hi + 12, 110, 0.08, 0.2, dur=3.5, rel=1.5)
    if fin: hit(CRASH, t, 115, 0.35, 0.2); hit(BD, t, 120, 0.4, 0.0)
# ---- lightning: thunder with every bolt (the cloud drop and every downbeat drum of the storm)
leap0 = LAY['ev']['leap']['t0']
bolts = [LAY['ev']['cloud_drop']['t']] + [h['t'] for h in hb if h['t'] < leap0 - 0.1 and abs(((h['t'] - EVJ['t0']) / EVJ['bar']) % 1) < 1e-3]
for k, t in enumerate(bolts):
    hit(ROLL, t + 0.02, 70, 0.22 if k else 0.4, 0.4 if k % 2 else -0.4); hit(BD, t, 90, 0.2, 0.0)
    if k == 0: hit(CRASH, t, 80, 0.18, 0.3)
for kind, t in LAY['sfx']:
    if kind == 'gate': hit(WOOD, t, 90, 0.22, 0.2); hit(CLAV, t + 0.01, 70, 0.1, 0.2)
    elif kind == 'start': hit(CLAV, t, 110, 0.3, 0.0); hit(WOOD, t + 0.02, 110, 0.3, 0.0); hit(BT, t + 0.05, 90, 0.25, 0.3)
    elif kind == 'lever': hit(LOGLO, t + 0.01, 110, 0.3, -0.2); hit(RAT, t + 0.1, 90, 0.25, 0.3)
    elif kind == 'bridge': hit(BD, t, 120, 0.45, 0.0); hit(LOGLO, t + 0.01, 120, 0.35, 0.0); hit(CRASH, t + 0.02, 90, 0.2, 0.0)
    elif kind == 'gateopen':
        hit(RAT, t, 90, 0.25, -0.3); hit(BT, t + 0.3, 110, 0.35, 0.0); hit(SLEIGH, t + 0.6, 100, 0.25, 0.3)
        for k in range(5): hit(BT, t + 0.9 + 0.35 * k, 90, 0.15, -0.5 + 0.25 * k)
# hall
L_ = int(2.4 * SR); tt = np.arange(L_) / SR
ir = np.stack([rng.normal(0, 1, L_), rng.normal(0, 1, L_)], 1) * np.exp(-tt / 0.6)[:, None]
for c in range(2): ir[:, c] = lfilter([0.3], [1, -0.55], ir[:, c])
ir /= np.sqrt((ir ** 2).sum(0)); wet = np.stack([fftconvolve(out[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = out + 0.26 * wet
fade = int(2.0 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix /= max(1e-6, np.abs(mix).max()) / 0.89
sf.write('music_raw.wav', mix, SR)
os.system('ffmpeg -v error -y -i music_raw.wav -af "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=9" -ar 44100 music.wav && rm music_raw.wav')
print('ok', DUR)
