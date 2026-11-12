"""Extract the source notes of the Moonlight Sonata, 1st movement (Beethoven, Op.27 No.2, 1801/1802, public domain)
from the Mutopia Project LilyPond file score/mutopia_moonlight1-a4.ly (Mutopia-2007/02/11-276, typeset by Stewart Holmes
from the Berners edition 1908 ed. A. Winterberger - a pre-1929 public-domain print; transcription CC BY-SA 2.5: we only
read the note facts from it, the arrangement is our own).  The three voices (topmain = melody, topsecondary = triplet
accompaniment, bottom = bass octaves) are written to separate MIDI tracks.
Writes score_src.json rows [bar, beat offset (quarters), duration (quarters), midi] per voice.  Needs lilypond + mido."""
import json, os, re, subprocess, tempfile, shutil
import mido
HERE = os.path.dirname(os.path.abspath(__file__))
d = tempfile.mkdtemp()
shutil.copy(f'{HERE}/score/mutopia_moonlight1-a4.ly', f'{d}/m.ly')
subprocess.run(['convert-ly', '-e', f'{d}/m.ly'], check=True, capture_output=True)
ly = open(f'{d}/m.ly', encoding='utf-8').read()
ly = ly[:ly.index('\nbreaking =')]
ly = re.sub(r'#\(override-auto-beam-setting[^\n]*\)', '', ly)
ly = re.sub(r'\\change Staff\s*=\s*"?\w+"?', '', ly)
ly += r'''
\score { <<
  \new Staff = "a" { \time 2/2 \topmain }
  \new Staff = "b" { \time 2/2 \topsecondary }
  \new Staff = "c" { \time 2/2 \bottom }
>> \midi { \tempo 4 = 60 } }
'''
open(f'{d}/g.ly', 'w').write(ly)
r = subprocess.run(['lilypond', '-s', '-o', f'{d}/g', f'{d}/g.ly'], capture_output=True, text=True)
if r.returncode: print(r.stderr[:3000]); print(d); raise SystemExit(1)
mid = mido.MidiFile(f'{d}/g.midi')
tpb = mid.ticks_per_beat
out = {}
names = ['melody', 'triplets', 'bass']
tracks = [t for t in mid.tracks if any(m.type == 'note_on' for m in t)]
assert len(tracks) == 3, len(tracks)
for nm, tr in zip(names, tracks):
    t = 0; on = {}; ev = []
    for m in tr:
        t += m.time
        if m.type == 'note_on' and m.velocity > 0: on.setdefault(m.note, []).append(t)
        elif m.type in ('note_off', 'note_on'):
            t0 = on[m.note].pop(0); q0 = t0 / tpb; dq = (t - t0) / tpb
            ev.append([int(q0 // 4) + 1, round(q0 % 4, 4), round(dq, 4), m.note])
    out[nm] = sorted(ev, key=lambda e: (e[0], e[1], e[3]))
out['source'] = ('Ludwig van Beethoven, Piano Sonata No.14 in C-sharp minor Op.27 No.2 "Sonata quasi una fantasia" '
                 '(composed 1801, first edition Giovanni Cappi, Vienna 1802), 1st movement Adagio sostenuto - public domain. '
                 'Notes read from the Mutopia Project LilyPond source BeethovenLv/O27/moonlight (Mutopia-2007/02/11-276, '
                 'Stewart Holmes, source: Berners edition 1908, ed. A. Winterberger). Rows: [bar, beat, quarters, midi].')
json.dump(out, open(f'{HERE}/score_src.json', 'w'))
for nm in names: print(nm, len(out[nm]), 'bars', out[nm][0][0], '-', out[nm][-1][0])
