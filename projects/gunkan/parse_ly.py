import re, json
src = open('gunkan_src.ly').read()
mel = src[src.index('\\numericTimeSignature') + 21: src.index('\\bar "|."')]
toks = re.findall(r"([a-gr](?:is|es)?)([',]*)(\d*)(\.?)(~?)", mel)
STEP = {'c': 0, 'd': 2, 'e': 4, 'f': 5, 'g': 7, 'a': 9, 'b': 11}
LET = 'cdefgab'
prev_letter, prev_oct = 'c', 4      # c' = middle C (octave 4)
dur = 4; t = 0.0; notes = []; tie = False
for name, octs, d, dot, tl in toks:
    if d: dur = int(d)
    q = 4.0 / dur * (1.5 if dot else 1.0)          # in quarter-note beats
    if name == 'r':
        t += q; tie = False; continue
    letter = name[0]; acc = 1 if name.endswith('is') else (-1 if name.endswith('es') and len(name) > 1 else 0)
    # relative: choose octave so that letter distance <= 3 steps
    di = LET.index(letter) - LET.index(prev_letter); o = prev_oct
    if di > 3: o -= 1
    elif di < -3: o += 1
    o += octs.count("'") - octs.count(',')
    prev_letter, prev_oct = letter, o
    pitch = 12 * (o + 1) + STEP[letter] + acc
    # key of D: f and c are sharp
    if letter in 'fc' and acc == 0: pitch += 1
    if tie and notes and notes[-1][2] == pitch:
        notes[-1][1] += q
    else:
        notes.append([t, q, pitch])
    tie = bool(tl); t += q
json.dump(notes, open('gunkan_melody.json', 'w'))
print('beats', t, 'bars', t / 4, 'notes', len(notes))
names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
for bar in range(int(t // 4)):
    print(bar, ' '.join(f"{names[p%12]}{p//12-1}:{q:g}" for b, q, p in notes if bar*4 <= b < bar*4+4))
