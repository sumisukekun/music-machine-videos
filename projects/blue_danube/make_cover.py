"""Cover: a frame of the video + hook (top) and title/tagline (bottom).  python3 make_cover.py video.mp4 time out.png"""
import sys, subprocess
from PIL import Image, ImageDraw, ImageFont, ImageFilter

vid, tsec, out = sys.argv[1], sys.argv[2], sys.argv[3]
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', tsec, '-i', vid, '-frames:v', '1', '/tmp/bd_cover_bg.png'], check=True)
im = Image.open('/tmp/bd_cover_bg.png').convert('RGB').resize((1080, 1920))
F = '/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc'
ov = Image.new('RGBA', im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
for y0, y1 in ((230, 560), (1380, 1740)):
    for y in range(y0, y1):
        a = int(130 * min(1, (y - y0) / 60, (y1 - y) / 60)); d.line([(0, y), (1080, y)], fill=(5, 8, 25, a))
im = Image.alpha_composite(im.convert('RGBA'), ov.filter(ImageFilter.GaussianBlur(8)))
d = ImageDraw.Draw(im)


def text(y, s, size, fill, stroke=10):
    f = ImageFont.truetype(F, size, index=0)
    w = d.textlength(s, font=f)
    while w > 1000: size -= 4; f = ImageFont.truetype(F, size, index=0); w = d.textlength(s, font=f)
    d.text(((1080 - w) / 2, y), s, font=f, fill=fill, stroke_width=stroke, stroke_fill=(20, 15, 40))


text(290, '2つの玉、また出会える？', 104, (255, 255, 255), 12)
text(450, 'ワルツで踊るビー玉からくり', 58, (255, 214, 120), 8)
text(1420, '美しく青きドナウ', 112, (255, 255, 255), 12)
text(1575, 'ヨハン・シュトラウス2世｜氷上の舞踏会', 52, (190, 225, 255), 8)
im.convert('RGB').save(out)
print('ok', out)
