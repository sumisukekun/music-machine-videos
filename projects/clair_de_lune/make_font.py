"""jp_sub.otf / jp_serif.otf = Noto Sans CJK JP Black / Noto Serif CJK JP Bold (SIL Open Font License), subset to the
characters used in scene.py (+ ASCII)."""
import os, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(f'{HERE}/scene.py', encoding='utf-8').read()
chars = sorted(set(c for c in src if ord(c) > 127) | set(chr(c) for c in range(32, 127)))
for fn, out in (('NotoSansCJK-Black.ttc', 'jp_sub.otf'), ('NotoSerifCJK-Bold.ttc', 'jp_serif.otf')):
    subprocess.run(['pyftsubset', f'/usr/share/fonts/opentype/noto/{fn}', '--font-number=0', '--text=' + ''.join(chars),
                    f'--output-file={HERE}/{out}'], check=True)
    print(out, len(chars), 'chars', os.path.getsize(f'{HERE}/{out}'), 'bytes')
