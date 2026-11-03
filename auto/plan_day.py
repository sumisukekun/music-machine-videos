"""Pick today's songs+themes, write the matrix for the daily workflow and update auto/state.json.
Skips (prints empty) when the posting queue already holds >= MAX_QUEUE items."""
import json, os, sys, glob, datetime, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
N = int(os.environ.get('PER_DAY', 4)); MAX_QUEUE = int(os.environ.get('MAX_QUEUE', 12))
cat = [c for c in json.load(open(os.path.join(HERE, 'catalog.json'))) if not c.get('skip')]
st_path = os.path.join(HERE, 'state.json')
st = json.load(open(st_path)) if os.path.exists(st_path) else {'used': [], 'theme_i': 0}
THEMES = ['kitchen', 'warship', 'concert']
queued = len(glob.glob(os.path.join(ROOT, 'queue', '*.json')))
items = []
if queued < MAX_QUEUE:
    day = datetime.datetime.now(datetime.UTC).replace(tzinfo=None) + datetime.timedelta(hours=9)
    used = set(st['used']); picked = set()
    for idx in range(N):
        th = THEMES[(st['theme_i'] + idx) % 3]
        cands = [c['id'] for c in cat if c['id'] not in picked and f"{c['id']}|{th}" not in used]
        if not cands: cands = [c['id'] for c in cat if c['id'] not in picked]
        sid = min(cands, key=lambda s: (sum(u.startswith(s + '|') for u in st['used']), [c['id'] for c in cat].index(s)))
        picked.add(sid)
        items.append({'name': f"{day:%Y%m%d}_{idx + 1}_{sid}_{th}", 'song': sid, 'theme': th})
        st['used'].append(f'{sid}|{th}')
    st['theme_i'] = (st['theme_i'] + 1) % 3
    if len(set(st['used'])) >= len(cat) * 3: st['used'] = []          # all combinations done: start over
    json.dump(st, open(st_path, 'w'), indent=1)
print(json.dumps(items))
