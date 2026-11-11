"""Extract the source notes for morning_mood from the PDMX (CC0) JSON files."""
import json, sys
SRC = '/home/claude/mm_ref/pdmx/'
OUT = sys.argv[1]
def notes(fn, ti, tr=0, bar_shift=0, m0=1, m1=23):
    d = json.load(open(SRC + fn)); res = d['resolution']; bl = sorted(b['time'] for b in d['barlines'])
    ev = []
    for n in d['tracks'][ti]['notes']:
        if n['is_grace']: continue
        m = n['measure'] + bar_shift
        if not (m0 <= m <= m1): continue
        b0 = max(b for b in bl if b <= n['time'])
        ev.append([m, round((n['time'] - b0) / (res / 2), 3), round(n['duration'] / (res / 2), 3), n['pitch'] + tr])
    return sorted(ev, key=lambda e: (e[0], e[1], -e[3]))
mel = notes('QmaoTg2HDUBohMRFjjDrmGJGmo9HzEa76VoKEx4oZnYNH3.json', 0)
acc = notes('QmVHzhRfP7v4UWWHcMw9z6K13ixmZXFFdV26SkBNpXNdVz.json', 2, tr=1, bar_shift=-1)
fl = notes('QmVHzhRfP7v4UWWHcMw9z6K13ixmZXFFdV26SkBNpXNdVz.json', 0, tr=1, bar_shift=-1)
ob = notes('QmVHzhRfP7v4UWWHcMw9z6K13ixmZXFFdV26SkBNpXNdVz.json', 1, tr=1, bar_shift=-1)
pno = notes('QmPUNHoPbosz4QLCQuAYoU1NmsCxVQGScfv9xKvq5sTYhV.json', 0, tr=4)
json.dump({'source': 'E. Grieg, Peer Gynt Suite No.1 Op.46 (1888), No.1 Morgenstemning (Morning Mood), bars 1-23, E major 6/8. '
           'Notes taken from PDMX (CC0 1.0 dataset of public-domain / CC0 MuseScore scores, huggingface.co/datasets/openmusic/pdmx): '
           'melody = MuseScore score 5886802 "Morning Mood - Violin Adaptation" (cc-zero, original key E); '
           'harmony = score 5842106 "Peer Gynt - Morning Mood" (cc-zero; written in E-flat, transposed +1 semitone, its bar n+1 = our bar n), '
           'piano part (track 2); its flute / oboe lines (tracks 0, 1) used to decide which instrument has which bar. '
           'Cross-check: score 4812602 "Morgenstimmung" (cc-zero, piano, in C, transposed +4). Grace notes dropped. '
           'Rows: [bar, offset in eighths, duration in eighths, midi].',
           'melody': mel, 'harmony': acc, 'flute': fl, 'oboe': ob, 'check_piano': pno}, open(OUT, 'w'))
print(len(mel), len(acc), len(fl), len(ob), len(pno))
