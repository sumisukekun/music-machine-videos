"""jp_sub.otf = Noto Sans CJK JP Bold (SIL Open Font License), subset to the characters used in scene.py (+ ASCII).
en_bold.ttf = Poppins Bold (SIL Open Font License), copied unchanged."""
import os, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(f'{HERE}/scene.py', encoding='utf-8').read()
chars = sorted(set(c for c in src if ord(c) > 127) | set(chr(c) for c in range(32, 127)))
subprocess.run(['pyftsubset', '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', '--font-number=0', '--text=' + ''.join(chars),
                f'--output-file={HERE}/jp_sub.otf'], check=True)
print('jp_sub.otf', len(chars), 'chars', os.path.getsize(f'{HERE}/jp_sub.otf'), 'bytes')
