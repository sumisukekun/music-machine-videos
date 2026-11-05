"""Note Bounce: chord-arpeggio arrangement supplied by the user (five_orgel.pdf, user's own arpeggio score from chord progression).
Intro (bars 1-8) + chorus (bars 33-50) + final C. RH = 8th-note arpeggio, LH = half-note root/fifth.
-> notes.json [t, dur, voice(0=RH,1=LH), midi, vel], events.json"""
import json, re
Q = 96; B = 60 / Q; BAR = 4 * B; T0 = 1.2
BARS = {1:'C',2:'Csus4',3:'C',4:'Csus4',5:'C',6:'Gsus4',7:'Am7',8:'F',
        33:'Esus4',34:'C',35:'Gsus4',36:'Am7',37:'Em7',38:'Fadd9',39:'Cmaj7 C',40:'Dm7',41:'Gsus4',
        42:'Am',43:'G#aug',44:'C/G',45:'F#m7b5',46:'Dm7',47:'C/E Am',48:'Bbmaj7',49:'Gsus4 G',50:'Fadd9'}
ORDER = list(range(1, 9)) + list(range(33, 51))
PC = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}
QUAL = {'':[0,4,7],'m':[0,3,7],'7':[0,4,7,10],'m7':[0,3,7,10],'maj7':[0,4,7,11],'sus4':[0,5,7],'add9':[0,4,7,14],
        'dim':[0,3,6],'aug':[0,4,8],'m7b5':[0,3,6,10]}
def parse(c):
    m = re.match(r'([A-G])([#b]?)([a-z0-9]*)(?:/([A-G][#b]?))?$', c)
    r = PC[m.group(1)] + {'#':1,'b':-1,'':0}[m.group(2)]
    ivs = QUAL[m.group(3)]
    bass = None
    if m.group(4): bass = PC[m.group(4)[0]] + {'#':1,'b':-1,'':0}[m.group(4)[1:]]
    return r % 12, ivs, (bass if bass is not None else r) % 12
notes = []
def N(t, d, v, p, vel): notes.append([round(t, 4), round(d, 4), v, p, vel])
for bi, bn in enumerate(ORDER):
    t_bar = T0 + bi * BAR
    chords = BARS[bn].split()
    half = len(chords) == 2
    for ci, ch in enumerate(chords):
        r, ivs, bass = parse(ch)
        root = 60 + r if 60 + r <= 71 else 48 + r
        if root < 60: root += 12
        tones = sorted(root + i for i in ivs[:4])
        if len(tones) == 3: pat = [tones[0], tones[1], tones[2], tones[1], tones[0] + 12, tones[1], tones[2], tones[1]]
        else: pat = [tones[0], tones[1], tones[2], tones[3], tones[0] + 12, tones[3], tones[2], tones[1]]
        if half: pat = pat[:4]
        t0 = t_bar + ci * 2 * B
        for k, p in enumerate(pat):
            N(t0 + k * B / 2, B / 2, 0, p, 92 if k % 4 == 0 else 74)
        lb = 48 + bass if 48 + bass <= 59 else 36 + bass
        fifth = 48 + (r + 7) % 12
        while fifth <= lb: fifth += 12
        if half: N(t0, 2 * B, 1, lb, 90)
        else: N(t0, 2 * B, 1, lb, 90); N(t0 + 2 * B, 2 * B, 1, fifth, 80)
tf = T0 + len(ORDER) * BAR
for p in (60, 64, 67, 72): N(tf, 3.0, 0, p, 100)
N(tf, 3.0, 1, 36, 100); N(tf, 3.0, 1, 48, 90)
notes.sort(key=lambda x: (x[0], x[2]))
json.dump(notes, open('notes.json', 'w'))
json.dump({'beat': B, 'bar': BAR, 't0': T0, 'final': round(tf, 3), 'nbars': len(ORDER)}, open('events.json', 'w'))
print(len(notes), 'notes, final', round(tf, 2))
