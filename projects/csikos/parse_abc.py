"""Minimal ABC parser for src.abc (2 voices, L:1/8). Output: bars per voice = list of [ (beat_offset_in_quarters, dur_q, [midi...]) ]
plus repeat/section markers."""
import re, json
PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
lines = open('src.abc').read().splitlines()
key_sharps = {'Gmaj': {'F': 1}, 'Cmaj': {}}
voices = {'1': [], '2': []}
key = 'Gmaj'
for ln in lines:
    if ln.startswith('K:'): key = ln[2:].strip(); voices['1'].append(('K', key)); voices['2'].append(('K', key)); continue
    m = re.match(r'\[V:(\d)\]\s*(.*)', ln)
    if m: voices[m.group(1)].append(('M', m.group(2)))
tok = re.compile(r"!.*?!|\{[^}]*\}|\[[^\]]*\][\d/]*|[\^=_]*[A-Ga-gz][',]*[\d/]*|\|:|:\||\|\]|\||[.LT()]|\s+|.")
note_re = re.compile(r"([\^=_]*)([A-Ga-g])([',]*)")

def dur(s):
    if not s: return 1.0
    if s == '/': return 0.5
    if '/' in s:
        a, b = s.split('/'); a = float(a) if a else 1; b = float(b) if b else 2; return a / b
    return float(s)

def parse(seq):
    bars = []; cur = []; t = 0.0; key = 'Gmaj'; acc = {}; marks = []
    for kind, txt in seq:
        if kind == 'K': key = txt; continue
        for tk in tok.findall(txt):
            if tk.startswith('!'):
                if tk in ('!D.C.!', '!coda!'): marks.append((len(bars), tk))
                continue
            if tk in ('|', '|:', ':|', '|]'):
                if cur or t > 0: bars.append({'notes': cur, 'len': t, 'key': key})
                if tk in ('|:', ':|'): marks.append((len(bars), tk))
                cur = []; t = 0.0; acc = {}; continue
            if tk.startswith('{') or tk.strip() == '' or tk in '.LT()': continue
            if tk.startswith('['):
                inner, tail = tk[1:tk.index(']')], tk[tk.index(']') + 1:]; ps = []; d = None
                for mm in re.finditer(r"([\^=_]*)([A-Ga-g])([',]*)([\d/]*)", inner):
                    ps.append(pitch(mm.group(1), mm.group(2), mm.group(3), key, acc)); d = dur(mm.group(4))
                if tail: d = d * dur(tail)
                cur.append((t, d * 0.5, ps)); t += d * 0.5; continue
            mm = re.match(r"([\^=_]*)([A-Ga-gz])([',]*)([\d/]*)", tk)
            if mm:
                d = dur(mm.group(4)) * 0.5
                if mm.group(2) != 'z': cur.append((t, d, [pitch(mm.group(1), mm.group(2), mm.group(3), key, acc)]))
                t += d
    if cur: bars.append({'notes': cur, 'len': t, 'key': key})
    return bars, marks

def pitch(a, l, o, key, acc):
    base = PC[l.upper()]; octv = 5 if l.islower() else 4
    octv += o.count("'") - o.count(',')
    L = l.upper()
    if a: acc[(L, octv)] = {'^': 1, '_': -1, '=': 0}[a[-1]] * (2 if len(a) == 2 and a[0] == a[1] else 1)
    s = acc.get((L, octv), key_sharps[key].get(L, 0))
    return 12 * (octv + 1) + base + s

out = {}
for v in ('1', '2'):
    b, mk = parse(voices[v])
    if v == '2':
        for bb in b: bb['notes'] = [(t, d, [p - 12 for p in ps]) for t, d, ps in bb['notes']]
    out[v] = {'bars': b, 'marks': mk}
    print(v, len(b), mk)
json.dump(out, open('parsed.json', 'w'))
