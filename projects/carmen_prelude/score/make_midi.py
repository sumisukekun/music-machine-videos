"""Builds score/prelude.midi from the Mutopia source (Mutopia-2005/11/22-635, Bizet 'Carmen - Prelude', piano reduction
typeset from the Schirmer 1895 edition; CC BY-SA 2.5). Only the notes are read from it (arrange.py); repeats unfolded.
python3 score/make_midi.py   (needs lilypond 2.24)"""
import os, re, subprocess
H = os.path.dirname(os.path.abspath(__file__))
s = open(f'{H}/prelude_mutopia_BizetG.ly', encoding='latin-1').read()
body = s[s.index('global ='):s.index('dynamics =')]
body = ''.join(c if ord(c) < 128 else 'u' for c in body)
body = re.sub(r'_\\markup \{"" \\raise #-5 \\bigger \\italic \{piu+\}\}', '', body)
open('/tmp/carmen_mid.ly', 'w').write('\\version "2.24.0"\n' + body + '\n\\score { \\unfoldRepeats { << \\new Staff { \\global \\upper } '
                                     '\\new Staff { \\global \\lower } >> } \\midi { \\tempo 4=116 } }\n')
subprocess.run(['lilypond', '-s', '-o', '/tmp/carmen_mid', '/tmp/carmen_mid.ly'], check=True)
os.replace('/tmp/carmen_mid.midi', f'{H}/prelude.midi'); print('ok')
