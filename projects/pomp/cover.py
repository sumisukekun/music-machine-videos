"""Text cover in the Ball Music style: この曲わかる？ + hook (top), title + tagline (bottom)."""
import sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter
src, dst = sys.argv[1], sys.argv[2]
im = Image.open(src).convert('RGB').resize((1080, 1920), Image.LANCZOS)
F = '/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc'
def font(sz): return ImageFont.truetype(F, sz, index=0)
# gentle darkening behind the text bands
ov = Image.new('L', im.size, 0); d = ImageDraw.Draw(ov)
for y in range(1920):
    a = 0
    if y < 640: a = int(150 * (1 - y / 640) ** 1.2)
    if y > 1300: a = int(170 * ((y - 1300) / 620) ** 1.1)
    d.line([(0, y), (1080, y)], fill=a)
im = Image.composite(Image.new('RGB', im.size, (5, 5, 15)), im, ov)
dr = ImageDraw.Draw(im)
def text(y, s, sz, col, stroke=10):
    f = font(sz); w = dr.textlength(s, font=f)
    while w > 1000: sz -= 4; f = font(sz); w = dr.textlength(s, font=f)
    dr.text(((1080 - w) / 2, y), s, font=f, fill=col, stroke_width=stroke, stroke_fill=(20, 12, 25))
text(300, 'この曲わかる？', 150, (255, 255, 255), 12)
text(470, '卒業式で聴いたあの曲…', 66, (255, 214, 92), 8)
text(1450, '威風堂々', 140, (255, 255, 255), 12)
text(1630, 'ビー玉×宮殿のパイプオルガン', 66, (255, 214, 92), 8)
im.save(dst); print('ok', dst)
