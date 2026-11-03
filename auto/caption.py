import json, sys
m = json.load(open(sys.argv[1]))
TH = {'kitchen': ('台所の鍋やグラスが', '#台所 #キッチン', '🍳'), 'warship': ('戦艦の大砲が', '#戦艦 #大砲', '⚓'), 'concert': ('16台の砲塔マシンが', '#精密 #機械', '🎯')}
a, tags, em = TH[m['theme']]
title = m['title_ja']
cap = (f"{a}「{title}」を演奏{em} 玉が当たるたびに音が鳴る！（{m['composer_ja']}）"
       f" #{title.replace(' ', '').replace('（', '').replace('）', '')} #{m['composer_ja'].split('（')[0].replace('・', '')} #クラシック #音楽 {tags}"
       " #からくり #ピタゴラ #blender #3dアニメ #ballmachine #satisfying")
print(cap)
