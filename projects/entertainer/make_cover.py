"""Cover: a frame of the video + hook (top) and title/tagline (bottom), same style as the other covers.
python3 make_cover.py video.mp4 time out.png"""
import sys, subprocess
from PIL import Image, ImageDraw, ImageFont, ImageFilter

vid, tsec, out = sys.argv[1], sys.argv[2], sys.argv[3]
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', tsec, '-i', vid, '-frames:v', '1', '/tmp/ent_cover_bg.png'], check=True)
im = Image.open('/tmp/ent_cover_bg.png').convert('RGB').resize((1080, 1920))
F = '/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc'
ov = Image.new('RGBA', im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
for y0, y1 in ((230, 560), (1380, 1720)):
    for y in range(y0, y1):
        a = int(130 * min(1, (y - y0) / 60, (y1 - y) / 60)); d.line([(0, y), (1080, y)], fill=(12, 6, 2, a))
im = Image.alpha_composite(im.convert('RGBA'), ov.filter(ImageFilter.GaussianBlur(8)))
d = ImageDraw.Draw(im)


def text(y, s, size, fill, stroke=10):
    f = ImageFont.truetype(F, size, index=0)
    w = d.textlength(s, font=f)
    while w > 1000: size -= 4; f = ImageFont.truetype(F, size, index=0); w = d.textlength(s, font=f)
    d.text(((1080 - w) / 2, y), s, font=f, fill=fill, stroke_width=stroke, stroke_fill=(30, 14, 6))


text(290, 'この曲、どこで聴いた？', 104, (255, 255, 255), 12)
text(450, '西部の酒場の自動ピアノ', 62, (255, 200, 90), 8)
text(1420, 'ジ・エンターテイナー', 100, (255, 255, 255), 11)
text(1565, 'ビー玉が弾くラグタイム（ジョプリン）', 54, (255, 200, 90), 8)
im.convert('RGB').save(out)
print('ok', out)
