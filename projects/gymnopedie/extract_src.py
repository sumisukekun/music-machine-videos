"""Extract the source notes of Gymnopedie No.1 from the Mutopia LilyPond file (public domain).
src_gymnopedie_1.ly = Mutopia Project SatieE/gymnopedie_1 (typeset by Evin Robertson from the Dover reprint of the original
1888 edition, 'Placed in the public domain by the typesetter').  The three voices (top = melody, middle = chords,
bottom = bass) are written to separate MIDI tracks with the repeat unfolded (first time through = bars 1-39).
Writes score_src.json rows [bar, beat offset (quarters), duration (quarters), midi] per voice.  Needs lilypond + mido."""
import json, os, re, subprocess, tempfile
import mido
HERE = os.path.dirname(os.path.abspath(__file__))
ly = open(f'{HERE}/src_gymnopedie_1.ly', encoding='utf-8').read()
ly = ly[:ly.index('\\score')]
ly = re.sub(r'\\change Staff\s*=\s*\w+', '', ly)
ly += r'''
\score { \unfoldRepeats <<
  \new Staff = "t" << \global \top >>
  \new Staff = "m" << \global \new Voice \middle >>
  \new Staff = "b" << \global \bottom >>
>> \midi { \tempo 4 = 60 } }
'''
d = tempfile.mkdtemp()
open(f'{d}/g.ly', 'w').write(ly)
subprocess.run(['lilypond', '-s', '-o', f'{d}/g', f'{d}/g.ly'], check=True)
mid = mido.MidiFile(f'{d}/g.midi')
tpb = mid.ticks_per_beat
out = {}
names = ['top', 'middle', 'bottom']
tracks = [t for t in mid.tracks if any(m.type == 'note_on' for m in t)]
assert len(tracks) == 3, len(tracks)
for nm, tr in zip(names, tracks):
    t = 0; on = {}; ev = []
    for m in tr:
        t += m.time
        if m.type == 'note_on' and m.velocity > 0: on.setdefault(m.note, []).append(t)
        elif m.type in ('note_off', 'note_on'):
            t0 = on[m.note].pop(0); q0 = t0 / tpb; dq = (t - t0) / tpb
            ev.append([int(q0 // 3) + 1, round(q0 % 3, 3), round(dq, 3), m.note])
    out[nm] = sorted(ev, key=lambda e: (e[0], e[1], e[3]))
out['source'] = ('Erik Satie, Gymnopedie No.1 (1888), public domain. Notes from the Mutopia Project LilyPond source '
                 'SatieE/gymnopedie_1 (Evin Robertson, from the Dover reprint of the original edition; public domain), '
                 'voices top/middle/bottom exported with lilypond (repeat unfolded). Rows: [bar, beat, quarters, midi].')
json.dump(out, open(f'{HERE}/score_src.json', 'w'))
for nm in names: print(nm, len(out[nm]), 'bars', out[nm][0][0], '-', out[nm][-1][0])
