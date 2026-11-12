"""Cover (English first): a frame of the video + English hook (top), English title / tagline and a small Japanese title
(bottom).  python3 make_cover.py video.mp4 time out.png"""
import sys, subprocess
from PIL import Image, ImageDraw, ImageFont, ImageFilter

vid, tsec, out = sys.argv[1], sys.argv[2], sys.argv[3]
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', tsec, '-i', vid, '-frames:v', '1', '/tmp/gy_cover_bg.png'], check=True)
im = Image.open('/tmp/gy_cover_bg.png').convert('RGB').resize((1080, 1920))
EN = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
JP = '/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc'
ov = Image.new('RGBA', im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
for y0, y1 in ((230, 560), (1360, 1760)):
    for y in range(y0, y1):
        a = int(130 * min(1, (y - y0) / 60, (y1 - y) / 60)); d.line([(0, y), (1080, y)], fill=(5, 8, 30, a))
im = Image.alpha_composite(im.convert('RGBA'), ov.filter(ImageFilter.GaussianBlur(8)))
d = ImageDraw.Draw(im)
STROKE = (15, 15, 45)


def text(y, s, size, fill, stroke=10, font=EN):
    f = ImageFont.truetype(font, size, index=0)
    w = d.textlength(s, font=f)
    while w > 1000: size -= 4; f = ImageFont.truetype(font, size, index=0); w = d.textlength(s, font=f)
    d.text(((1080 - w) / 2, y), s, font=f, fill=fill, stroke_width=stroke, stroke_fill=STROKE)


HOOK, HOOK2 = 'This tower is a music box', 'Can 2 marbles light it up?'
TITLE, TAG, JPT = 'Gymnopédie No.1', 'Satie · Marble machine music box', 'サティ「ジムノペディ第1番」'
text(285, HOOK, 84, (255, 255, 255), 12)
text(420, HOOK2, 60, (255, 214, 120), 8)
text(1395, TITLE, 104, (255, 255, 255), 12)
text(1545, TAG, 52, (255, 225, 190), 8)
text(1640, JPT, 46, (255, 255, 255), 7, font=JP)
im.convert('RGB').save(out)
print('ok', out)
