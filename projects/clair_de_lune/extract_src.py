"""Extract the source notes of Clair de Lune bars 1-14 from the Mutopia Project LilyPond file (score/, public domain).
The file is compiled with LilyPond 2.24 to MIDI with one MIDI track per voice (Staff_performer moved into Voice), so the four
voices of the engraving (rhUp, rhDown, lhUp, lhDown) stay separate.  python3 extract_src.py -> score_src.json"""
import json, os, subprocess, tempfile
import mido
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = f'{HERE}/score/debussy_Ste_Bergamesq_Clair.ly'
s = open(SRC).read()
s = s.replace('\\remove "Dynamic_performer"', '\\remove "Dynamic_performer"\n      \\consists "Staff_performer"\n    }\n    \\context { \\Staff \\remove "Staff_performer"')
d = tempfile.mkdtemp()
open(f'{d}/v.ly', 'w').write(s)
subprocess.run(['lilypond', '-s', '-dno-print-pages', '-o', f'{d}/v', f'{d}/v.ly'], check=True)
m = mido.MidiFile(f'{d}/v.midi'); E8 = m.ticks_per_beat / 2; BAR = 9 * E8
VOICE = {1: 'lhUp', 2: 'rhUp', 3: 'rhDown', 4: 'lhDown'}
rows = []
for ti, tr in enumerate(m.tracks):
    t = 0; on = {}
    for msg in tr:
        t += msg.time
        if msg.type == 'note_on' and msg.velocity > 0: on[msg.note] = t
        elif msg.type in ('note_off', 'note_on') and msg.note in on:
            t0 = on.pop(msg.note); bar = int(t0 // BAR) + 1
            if bar <= 14: rows.append([bar, round((t0 % BAR) / E8, 3), round((t - t0) / E8, 3), msg.note, VOICE[ti]])
rows.sort(key=lambda r: (r[0], r[1], -r[3]))
json.dump({'source': 'C. Debussy, Suite bergamasque L.75 (E. Fromont, Paris 1905, plate E. 1404 F., IMSLP #02907), III. Clair de lune, '
           'bars 1-14 (Andante tres expressif), D-flat major 9/8. Notes from the Mutopia Project edition '
           'DebussyC/L75/debussy_Ste_Bergamesq_Clair (typeset by Keith OHara from the 1905 Fromont print, placed in the public '
           'domain by the typesetter; copy in score/), compiled with LilyPond 2.24 to MIDI, one track per engraved voice. '
           'Rows: [bar, offset in eighths, duration in eighths (ties merged), midi, voice].', 'notes': rows}, open(f'{HERE}/score_src.json', 'w'))
print(len(rows), 'notes')
