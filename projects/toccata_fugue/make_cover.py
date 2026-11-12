"""Cover (English first): a frame of the video + English hook (top), English title / tagline and a small Japanese title
(bottom).  python3 make_cover.py video.mp4 time out.png"""
import sys, subprocess
from PIL import Image, ImageDraw, ImageFont, ImageFilter

vid, tsec, out = sys.argv[1], sys.argv[2], sys.argv[3]
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', tsec, '-i', vid, '-frames:v', '1', '/tmp/tf_cover_bg.png'], check=True)
im = Image.open('/tmp/tf_cover_bg.png').convert('RGB').resize((1080, 1920))
EN = '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf'; ENS = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
JP = '/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc'
ov = Image.new('RGBA', im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
for y0, y1 in ((200, 560), (1360, 1780)):
    for y in range(y0, y1):
        a = int(150 * min(1, (y - y0) / 60, (y1 - y) / 60)); d.line([(0, y), (1080, y)], fill=(4, 4, 14, a))
im = Image.alpha_composite(im.convert('RGBA'), ov.filter(ImageFilter.GaussianBlur(8)))
d = ImageDraw.Draw(im)
STROKE = (12, 8, 24)


def text(y, s, size, fill, stroke=10, font=ENS):
    f = ImageFont.truetype(font, size, index=0)
    w = d.textlength(s, font=f)
    while w > 1000: size -= 4; f = ImageFont.truetype(font, size, index=0); w = d.textlength(s, font=f)
    d.text(((1080 - w) / 2, y), s, font=f, fill=fill, stroke_width=stroke, stroke_fill=STROKE)


HOOK, HOOK2 = 'Can 3 marbles wake the giant organ?', 'Midnight cathedral · marble machine'
TITLE, TAG, JPT = 'Toccata and Fugue in D minor', 'Bach BWV 565 · Pipe organ of marbles', 'トッカータとフーガ ニ短調（バッハ）'
text(265, HOOK, 84, (255, 255, 255), 12)
text(420, HOOK2, 54, (255, 205, 120), 8)
text(1395, TITLE, 92, (255, 255, 255), 12, font=EN)
text(1540, TAG, 52, (255, 200, 150), 8)
text(1640, JPT, 46, (255, 255, 255), 7, font=JP)
im.convert('RGB').save(out)
print('ok', out)
