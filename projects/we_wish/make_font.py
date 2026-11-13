"""jp_sub.otf = Noto Sans CJK JP Black (SIL Open Font License) subset to the characters used in scene.py."""
import os, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(f'{HERE}/scene.py', encoding='utf-8').read()
chars = sorted(set(c for c in src if ord(c) > 127) | set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ()-.,!?「」（）×+:'))
subprocess.run(['pyftsubset', '/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc', '--font-number=0', '--text=' + ''.join(chars),
                f'--output-file={HERE}/jp_sub.otf'], check=True)
print(len(chars), 'chars', os.path.getsize(f'{HERE}/jp_sub.otf'), 'bytes')
