"""Jingle Bells (J. L. Pierpont, 1857, public domain) - festive arrangement + light-show event list."""
import json
import mido

BPM = 138.0
B = 60 / BPM
T0 = 0.6
D4, E4, Fs4, G4, A4, B4, C5, D5, E5, Fs5, G5 = 62, 64, 66, 67, 69, 71, 72, 74, 76, 78, 79
# (beat_in_bar, dur, pitch) per bar
VERSE = [
    [(0, 1, D4), (1, 1, B4), (2, 1, A4), (3, 1, G4)], [(0, 3, D4), (3, .5, D4), (3.5, .5, D4)],
    [(0, 1, D4), (1, 1, B4), (2, 1, A4), (3, 1, G4)], [(0, 3, E4), (3, 1, E4)],
    [(0, 1, E4), (1, 1, C5), (2, 1, B4), (3, 1, A4)], [(0, 4, Fs4)],
    [(0, 1, D5), (1, 1, D5), (2, 1, C5), (3, 1, A4)], [(0, 4, B4)],
    [(0, 1, D4), (1, 1, B4), (2, 1, A4), (3, 1, G4)], [(0, 3, D4), (3, 1, D4)],
    [(0, 1, D4), (1, 1, B4), (2, 1, A4), (3, 1, G4)], [(0, 3, E4), (3, 1, E4)],
    [(0, 1, E4), (1, 1, C5), (2, 1, B4), (3, 1, A4)], [(0, 1, D5), (1, 1, D5), (2, 1, D5), (3, 1, D5)],
    [(0, 1, E5), (1, 1, D5), (2, 1, C5), (3, 1, A4)], [(0, 2, G4), (2, 2, D5)],
]
VERSE_CH = ['G', 'G', 'G', 'C', 'Am', 'D', 'D7', 'G', 'G', 'G', 'G', 'C', 'Am', 'D', 'D7', 'G']
CHORUS = [
    [(0, 1, B4), (1, 1, B4), (2, 2, B4)], [(0, 1, B4), (1, 1, B4), (2, 2, B4)],
    [(0, 1, B4), (1, 1, D5), (2, 1.5, G4), (3.5, .5, A4)], [(0, 4, B4)],
    [(0, 1, C5), (1, 1, C5), (2, 1.5, C5), (3.5, .5, C5)], [(0, 1, C5), (1, 1, B4), (2, 1, B4), (3, .5, B4), (3.5, .5, B4)],
    [(0, 1, B4), (1, 1, A4), (2, 1, A4), (3, 1, B4)], [(0, 2, A4), (2, 2, D5)],
    [(0, 1, B4), (1, 1, B4), (2, 2, B4)], [(0, 1, B4), (1, 1, B4), (2, 2, B4)],
    [(0, 1, B4), (1, 1, D5), (2, 1.5, G4), (3.5, .5, A4)], [(0, 4, B4)],
    [(0, 1, C5), (1, 1, C5), (2, 1.5, C5), (3.5, .5, C5)], [(0, 1, C5), (1, 1, B4), (2, 1, B4), (3, .5, B4), (3.5, .5, B4)],
    [(0, 1, D5), (1, 1, D5), (2, 1, C5), (3, 1, A4)], [(0, 4, G4)],
]
CHORUS_CH = ['G', 'G', 'G', 'G', 'C', 'G', 'A7', 'D7', 'G', 'G', 'G', 'G', 'C', 'G', 'D7', 'G']
TAG = [[(0, 1, D5), (1, 1, D5), (2, 1, C5), (3, 1, A4)], [(0, 4, G4)]]
TAG_CH = ['D7', 'G']
CH = {'G': [7, 11, 2], 'C': [0, 4, 7], 'Am': [9, 0, 4], 'D': [2, 6, 9], 'D7': [2, 6, 9, 0], 'A7': [9, 1, 4, 7]}

bars = []   # (section, notes, chord)
bars += [('intro', [], 'G'), ('intro', [], 'D7')]
bars += [('verse', n, c) for n, c in zip(VERSE, VERSE_CH)]
bars += [('chorus', n, c) for n, c in zip(CHORUS, CHORUS_CH)]
bars += [('tag', n, c) for n, c in zip(TAG, TAG_CH)]
bars += [('final', [(0, 4, G5)], 'G')]
NB = len(bars)

P = {'glock': (0, 9), 'piano': (1, 0), 'bass': (2, 32), 'str': (3, 48), 'brass': (4, 61), 'flute': (5, 73), 'tubular': (6, 14), 'celesta': (7, 8)}
ev = []
def add(beat, dur, ins, p, v): ev.append((beat, dur, ins, int(p), int(max(1, min(127, v)))))
def drum(beat, k, v): ev.append((beat, 0.3, 'drum', k, v))

melody, bassev, kicks, bells, chords = [], [], [], [], []
for bi, (sec, notes, ch) in enumerate(bars):
    b0 = bi * 4
    big = sec in ('chorus', 'tag', 'final')
    for x, d, p in notes:
        melody.append((b0 + x, d, p, sec))
        add(b0 + x, d * 0.9, 'glock', p + 12, 95)
        add(b0 + x, d * 0.9, 'piano', p, 90 if big else 80)
        if big: add(b0 + x, d * 0.9, 'flute', p + 12, 70)
    pcs = CH[ch]; root = 43 + (pcs[0] - 7) % 12
    if root > 50: root -= 12
    chords.append((b0, ch))
    if sec == 'final':
        for p in (43, 55, 59, 62, 67, 71, 74, 79):
            add(b0, 4, 'str', p, 105); add(b0, 4, 'brass', p, 90)
        add(b0, 4, 'tubular', 79, 110); add(b0, 4, 'tubular', 67, 100)
        drum(b0, 49, 115); drum(b0, 57, 105)
        for k in range(8): drum(b0 + k * 0.5, 83, 100 - k * 8)
        continue
    # bass: root / fifth on beats 1 & 3 (walking in the chorus)
    for k, off in enumerate((0, 2)):
        bp = root if k == 0 else root + 7
        add(b0 + off, 1.8, 'bass', bp, 100); bassev.append(b0 + off)
    if big:
        add(b0 + 1, 0.9, 'bass', root + 4 if ch in ('G', 'C', 'D', 'D7', 'A7') else root + 3, 85)
        add(b0 + 3, 0.9, 'bass', root + 12, 85)
    # strings pad + brass stabs on 2 & 4 in the chorus
    for pc in pcs[:3]:
        add(b0, 3.9, 'str', 60 + (pc - 7) % 12 - 5, 60 if not big else 75)
    if big:
        for off in (1, 3):
            for pc in pcs[:3]: add(b0 + off, 0.4, 'brass', 60 + (pc - 7) % 12 - 5, 80)
    # celesta arpeggio in the verse
    if sec == 'verse':
        arp = sorted(72 + (pc - 7) % 12 for pc in pcs[:3])
        for k in range(8): add(b0 + k * 0.5, 0.45, 'celesta', arp[k % 3] + (12 if k % 6 >= 3 else 0), 55)
    # sleigh bells on eighths, kick/snare
    for k in range(8):
        v = 95 if k % 2 == 0 else 70
        drum(b0 + k * 0.5, 83, v if sec != 'intro' or bi > 0 else v - 20); bells.append(b0 + k * 0.5)
    if sec != 'intro':
        drum(b0, 36, 95); drum(b0 + 2, 36, 90); kicks += [b0, b0 + 2]
        drum(b0 + 1, 38, 80 if big else 55); drum(b0 + 3, 38, 80 if big else 55)
    if sec == 'chorus' and bi == 18:
        drum(b0, 49, 110)
FINAL_BEAT = (NB - 1) * 4

TPB = 480
mid = mido.MidiFile(ticks_per_beat=TPB); tr = mido.MidiTrack(); mid.tracks.append(tr)
tr.append(mido.MetaMessage('set_tempo', tempo=int(B * 1e6)))
msgs = []
for name, (ch, prog) in P.items():
    msgs.append((0, mido.Message('program_change', channel=ch, program=prog)))
    msgs.append((0, mido.Message('control_change', channel=ch, control=91, value=60)))
off = T0 / B
for b0, d, ins, p, v in ev:
    ch = 9 if ins == 'drum' else P[ins][0]
    msgs.append((int((b0 + off) * TPB), mido.Message('note_on', channel=ch, note=p, velocity=v)))
    msgs.append((int((b0 + off + d) * TPB), mido.Message('note_off', channel=ch, note=p, velocity=0)))
msgs.sort(key=lambda m: (m[0], 0 if m[1].type == 'note_off' else 1))
last = 0
for t, m in msgs: tr.append(m.copy(time=t - last)); last = t
mid.save('jingle.mid')
sec = lambda b: (b + off) * B
json.dump({'melody': [{'t': sec(b), 'd': d * B, 'p': p, 'sect': s} for b, d, p, s in melody],
           'bass': [sec(b) for b in bassev], 'kick': [sec(b) for b in kicks], 'bells': [sec(b) for b in bells],
           'chords': [(sec(b), c) for b, c in chords], 'final': sec(FINAL_BEAT), 'beat': B,
           'sections': [(sec(i * 4), s) for i, (s, _, _) in enumerate(bars) if i == 0 or bars[i - 1][0] != s]},
          open('events.json', 'w'))
print('bars', NB, 'final', round(sec(FINAL_BEAT), 2), 'melody notes', len(melody))
