"""Turn a public-domain score (Mutopia LilyPond source) into a ball-machine arrangement.
  python3 auto/make_song.py CATALOG_ID OUTDIR THEME
Writes OUTDIR/events.json (balls/hits, same schema as the hand-made projects) and OUTDIR/notes.json (audio).
Pipeline: lilypond -> MIDI -> skyline melody, lowest-voice bass, chord per half bar -> ~60 s excerpt at the
catalog tempo -> machine arrangement (melody, 3rd-below harmony, chord stabs, bass, drums by style)."""
import sys, os, json, glob, subprocess, re, math, collections
import mido

HERE = os.path.dirname(os.path.abspath(__file__))
CAT = {c['id']: c for c in json.load(open(os.path.join(HERE, 'catalog.json')))}
MUTOPIA = os.environ.get('MUTOPIA', '/home/claude/mutopiaproject/mutopiaproject')
T0 = 1.6; TARGET_FINAL = 61.6


def score_to_midi(entry, work):
    d = os.path.join(MUTOPIA, 'ftp', entry['ly'])
    main = os.path.join(d, entry['main']) if entry.get('main') else None
    if not main:
        cands = sorted(glob.glob(os.path.join(d, '**', '*.ly'), recursive=True), key=len)
        main = next(f for f in cands if '\\score' in open(f, errors='ignore').read())
    os.makedirs(work, exist_ok=True)
    subprocess.run(['convert-ly', '-e', main], capture_output=True)
    r = subprocess.run(['lilypond', '-dno-print-pages', '-dmidi-extension=mid', '-o', os.path.join(work, 'score'), main],
                       capture_output=True, text=True, cwd=os.path.dirname(main))
    mids = sorted(glob.glob(os.path.join(work, 'score*.mid')))
    if not mids: print(r.stderr[-2000:]); sys.exit('no midi')
    return mids[0]


def read_notes(path):
    m = mido.MidiFile(path); tpb = m.ticks_per_beat; notes = []; ts = (4, 4)
    for ti, tr in enumerate(m.tracks):
        t = 0; on = {}
        for x in tr:
            t += x.time
            if x.type == 'time_signature' and t == 0: ts = (x.numerator, x.denominator)
            if x.type == 'note_on' and x.velocity > 0 and getattr(x, 'channel', 0) != 9: on[(x.channel, x.note)] = t
            elif x.type in ('note_off', 'note_on') and (getattr(x, 'channel', 0), x.note) in on:
                s = on.pop((x.channel, x.note)); notes.append((s / tpb, max(0.05, (t - s) / tpb), x.note, ti))
    notes.sort()
    return notes, ts


TEMPL = {'maj': [0, 4, 7], 'min': [0, 3, 7], 'dim': [0, 3, 6], 'dom7': [0, 4, 7, 10]}
def best_chord(hist):
    best = None
    for root in range(12):
        for name, iv in TEMPL.items():
            sc = sum(hist[(root + i) % 12] for i in iv) - 0.3 * sum(hist[p] for p in range(12) if (p - root) % 12 not in iv)
            if name == 'dom7': sc -= 0.15 * sum(hist)
            if best is None or sc > best[0]: best = (sc, root, name)
    return best[1], best[2]


def main(cid, out, theme):
    out = os.path.abspath(out)
    e = CAT[cid]; os.makedirs(out, exist_ok=True)
    notes, ts = read_notes(score_to_midi(e, os.path.join(out, 'work')))
    num, den = ts; bar = num * 4 / den                          # bar length in quarter beats
    compound = num in (6, 9, 12) and den == 8
    unit = 4 / den * (3 if compound else 1)                      # beat length in quarters
    nbeats = int(round(bar / unit))
    bpm = e['bpm']; B = 60 / bpm
    start = e.get('start', 0.0)
    nb = int((TARGET_FINAL - T0) / B / bar)                     # whole bars that fit
    nb -= nb % 2
    end = start + nb * bar
    plen = math.ceil(max(b + d for b, d, p, tr in notes) / bar - 0.05) * bar      # repeat short pieces
    base = list(notes); k = 1
    while plen * k < end:
        notes += [(b + plen * k, d, p, tr) for b, d, p, tr in base]; k += 1
    sel = [(b - start, d, p, tr) for b, d, p, tr in notes if start <= b < end]
    if not sel: sys.exit('empty excerpt')
    tr_mel = e.get('mel_track')
    # ---- skyline melody (highest pitch per onset, notes covered by a sounding higher note dropped) ----
    on = collections.defaultdict(list)
    for b, d, p, tr in sel:
        if tr_mel is None or tr == tr_mel: on[round(b, 3)].append((p, d))
    mel = []; last_end = -1; last_p = 0
    for b in sorted(on):
        p, d = max(on[b])
        if mel and b < last_end - 1e-6 and p < last_p: continue
        if mel and b - mel[-1][0] < 0.12: continue
        mel.append((b, d, p)); last_end = b + d; last_p = p
    # ---- bass: lowest note starting within each beat ----
    bass = []
    for k in range(int(nb * bar / unit)):
        cand = [(p, b) for b, d, p, tr in sel if k * unit - 0.01 <= b < (k + 0.5) * unit]
        if cand:
            p, b = min(cand)
            if p < 60: bass.append((k * unit, p))
    # ---- chord per half bar ----
    hw = bar / 2 if num % 2 == 0 else bar
    chords = []
    for k in range(int(nb * bar / hw)):
        h = [0.0] * 12
        for b, d, p, tr in sel:
            ov = min(b + d, (k + 1) * hw) - max(b, k * hw)
            if ov > 0: h[p % 12] += ov * (1.5 if p < 55 else 1.0)
        chords.append(best_chord(h) if sum(h) else (chords[-1] if chords else (0, 'maj')))
    # key = chord of the first bar; final chord = tonic of that key
    tonic, tq = chords[0]
    # ---- transpose so the melody sits in 60..84 ----
    pmin, pmax = min(p for _, _, p in mel), max(p for _, _, p in mel)
    tr = 0
    while pmax + tr > 86: tr -= 12
    while pmin + tr < 58: tr += 12
    tr += e.get('transpose', 0)
    sec = lambda b: round(T0 + b * B, 4)
    H = {k: [] for k in ('mel', 'harm', 'chord', 'bass', 'kick', 'snare', 'tom', 'crash', 'hat', 'gliss')}
    N = []
    def note(b, d, ins, p, v, role): N.append([sec(b), d * B, ins, int(p), int(max(1, min(127, v))), role, b])
    for b, d, p in mel:
        note(b, max(d, 0.25), 'mel', p + tr, 108, 'mel'); H['mel'].append((b, p + tr))
    mel_p = sorted({p + tr for _, _, p in mel})
    # harmony: chord tone a 3rd/6th below (only notes >= an eighth long, from bar 4)
    def chord_at(b): return chords[min(len(chords) - 1, int(b // hw))]
    for b, d, p in mel:
        if b < 4 * bar or d < 0.45: continue
        root, q = chord_at(b); pcs = {(root + i + tr) % 12 for i in TEMPL[q]}
        cand = [x for x in range(p + tr - 9, p + tr - 2) if x % 12 in pcs]
        if cand:
            hp = max(cand); note(b, d, 'harm', hp, 84, 'harm'); H['harm'].append((b, hp))
    # chord stabs
    style = e.get('style') or ('waltz' if num == 3 else ('lyrical' if bpm < 90 else ('galop' if bpm >= 150 and num in (2, 4) else 'march')))
    beats_in_bar = nbeats
    half = (2 / 3) if compound else 0.5
    if style == 'waltz': offs = [1, 2]
    elif style == 'galop': offs = [x + half for x in range(beats_in_bar)]
    elif style == 'lyrical': offs = [0]
    else: offs = [1, 3] if beats_in_bar == 4 else [x + half for x in range(beats_in_bar)]
    offs = [o * unit for o in offs]
    for k in range(nb):
        for o in offs:
            b = k * bar + o
            root, q = chord_at(b); pcs = [(root + i + tr) % 12 for i in TEMPL[q]][:4]
            if q == 'dom7': pcs = pcs[1:]
            ps = sorted(57 + (pc - 57) % 12 for pc in pcs)[:3]
            for p in ps: note(b, 0.4, 'chord', p, 80, 'chord')
            H['chord'].append((b, ps))
    # bass on the strong beats, folded into one octave and a fifth (<= 12 bars)
    strong = [0, 2] if beats_in_bar == 4 else ([0] if beats_in_bar == 3 else list(range(beats_in_bar)))
    strong = [o * unit for o in strong]
    for k in range(nb):
        for o in strong:
            b = k * bar + o
            cand = [p for bb, p in bass if abs(bb - b) < 0.6 * unit]
            root, q = chord_at(b)
            p = cand[0] + tr if cand else 36 + (root + tr - 36) % 12
            p = 36 + (p - 36) % 12
            note(b, 0.8, 'bass', p, 112, 'bass'); H['bass'].append((b, p))
    # drums
    def drum(b, k, v, key=None, role='drum'):
        N.append([sec(b), 0.25 * B, 'drum', k, v, role, b])
        if key: H[key].append(b)
    for k in range(nb):
        b0 = k * bar
        if style == 'waltz':
            drum(b0, 36, 104, 'kick')
            for o in (1, 2): drum(b0 + o * unit, 38, 70, 'snare')
        elif style == 'lyrical':
            if k % 2 == 0: drum(b0, 36, 90, 'kick')
        elif style == 'galop':
            for o in range(beats_in_bar): drum(b0 + o * unit, 36, 106, 'kick'); drum(b0 + (o + half) * unit, 38, 94, 'snare')
        else:
            for o in strong: drum(b0 + o, 36, 108, 'kick')
            for o in range(beats_in_bar):
                if o * unit not in strong: drum(b0 + o * unit, 38, 96, 'snare')
            if beats_in_bar <= 2:
                for o in range(beats_in_bar): drum(b0 + (o + half) * unit, 38, 90, 'snare')
        if style != 'lyrical':
            sub = (3 if compound else 2) * (2 if style == 'galop' and not compound else 1)
            for s_ in range(int(round(beats_in_bar * sub))):
                bb = b0 + s_ / sub * unit; acc = 90 if s_ % sub == 0 else 56
                N.append([sec(bb), 0.1 * B, 'drum', 37, acc, 'hat', bb]); H['hat'].append(bb)
    # sections, crashes, fills, glissandi
    total = nb * bar
    def at_sec(s): return max(0, round(((s - T0) / B) / bar) * bar)
    marks = {'A': 0.0, 'A+': at_sec(8.0), 'B': at_sec(15.3), 'roll': at_sec(31.0), 'A2': at_sec(35.0), 'finale': at_sec(55.6)}
    for i, b in enumerate(sorted(set([0.0, marks['B'], marks['A2'], marks['finale'], total]))):
        drum(b, 49 if i % 2 == 0 else 57, 122); H['crash'].append((b, i % 2))
    for fb in (marks['A2'], marks['finale'], total):
        for i, (o, kk) in enumerate(((-2, 50), (-1.5, 47), (-1, 45), (-0.5, 45))):
            if fb + o > 0: drum(fb + o, kk, 110); H['tom'].append((fb + o, kk))
    gl = [p for p in mel_p]
    for fb in (marks['A2'], total):
        ps = gl[-min(len(gl), 10):]
        for i, p in enumerate(ps):
            b = fb - 1 + i / max(1, len(ps) - 1) * 0.95
            note(b, 0.2, 'gliss', p, 80 + 40 * i / len(ps), 'gliss'); H['gliss'].append((b, p))
    # remove melody notes under the glissandi
    gl_w = [(fb - 1.0, fb) for fb in (marks['A2'], total)]
    keep = lambda b: not any(a <= b < z for a, z in gl_w)
    H['mel'] = [x for x in H['mel'] if keep(x[0])]; N = [n for n in N if not (n[5] == 'mel' and not keep(n[6]))]
    # final chord on the tonic
    tp = (tonic + tr) % 12; third = 3 if tq == 'min' else 4
    fin_mel = max([p for p in mel_p if p % 12 == tp] or [mel_p[-1]])
    fin_ch = sorted(57 + ((tp + i) - 57) % 12 for i in (0, third, 7))
    fin_bass = 36 + (tp - 36) % 12
    for p in fin_ch: note(total, 3, 'chord', p, 118, 'final')
    note(total, 3, 'bass', fin_bass, 124, 'final'); note(total, 3, 'mel', fin_mel, 120, 'final')
    H['mel'].append((total, fin_mel)); H['chord'].append((total, fin_ch)); H['bass'].append((total, fin_bass))
    drum(total, 36, 127, 'kick')
    ev = {}
    for k, v in H.items():
        ev[k] = [[sec(x[0])] + list(x[1:]) if isinstance(x, tuple) else sec(x) for x in sorted(v, key=lambda z: z[0] if isinstance(z, tuple) else z)]
    ev['final'] = sec(total); ev['beat'] = B; ev['t0'] = T0
    ev['sections'] = [(sec(marks[k]), k) for k in ('A', 'A+', 'B', 'roll', 'A2', 'finale')]
    json.dump(ev, open(os.path.join(out, 'events.json'), 'w'))
    json.dump(sorted(N), open(os.path.join(out, 'notes.json'), 'w'))
    meta = dict(e, theme=theme, style=style, bars=nb, final=ev['final'], mel_pitches=len(mel_p),
                counts={k: len(v) for k, v in H.items()})
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
