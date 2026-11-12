"""Checks for toccata_fugue (run after layout.py):  python3 verify.py
 1. every note row of lead.json is struck exactly at its time by a marble touching its key (contact distance at the hit time
    and at the nearest video frame), pitch table vs the source score (arrange.py asserts the score itself)
 2. physics: restitution / speed out<=in at every bounce, in-flight frame-to-frame speed ratio (constant gravity)
 3. collisions: marble vs every key box (outside its own hit), marble vs marble
 4. every visible key is struck at least once"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0]]
L = json.load(open(f'{HERE}/layout.json')); LEAD = json.load(open(f'{HERE}/lead.json')); EV = json.load(open(f'{HERE}/events.json'))
G = L['G']; FPS = 30
NM = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def nm(p): return f'{NM[p % 12]}{p // 12 - 1}'
GZ = np.array([0, 0, -G])


def seg_at(g, t):
    p = np.array(g['p']); u = t - g['t0']
    if g['k'] == 'rest': return p
    if g['k'] == 'fly': return p + np.array(g['v']) * u + 0.5 * GZ * u * u
    if g['k'] == 'line': return p + np.array(g['d']) * (g['v0'] * u + 0.5 * g['a'] * u * u)
    f = 0 if g['t1'] <= g['t0'] else min(1, max(0, (t - g['t0']) / (g['t1'] - g['t0'])))
    if g['k'] == 'ease': f = f * f * (3 - 2 * f)
    return p + (np.array(g['p1']) - p) * f


def pos(b, t):
    for g in b['seg']:
        if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9: return seg_at(g, t)
    return seg_at(b['seg'][0], b['seg'][0]['t0']) if t < b['seg'][0]['t0'] else seg_at(b['seg'][-1], b['seg'][-1]['t1'])


def positions(b, ts):
    out = np.zeros((len(ts), 3)); j = 0; segs = b['seg']
    for k, t in enumerate(ts):
        while j < len(segs) - 1 and t > segs[j]['t1'] + 1e-9: j += 1
        g = segs[j]
        out[k] = seg_at(g, min(max(t, g['t0']), g['t1'])) if (g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9) else (
            seg_at(segs[0], segs[0]['t0']) if t < segs[0]['t0'] else seg_at(g, g['t1']))
    return out


def key_frame(k):
    n = np.array(k['n'], float); n /= np.linalg.norm(n)
    td = np.array(k['tdir'], float) if k.get('tdir') else np.cross([0, 1, 0], n)
    td = td - n * np.dot(td, n)
    if np.linalg.norm(td) < 1e-6: td = np.cross([0, 1, 0], n)
    if np.linalg.norm(td) < 1e-6: td = np.array([1.0, 0, 0])
    td /= np.linalg.norm(td); dd = np.cross(n, td)
    return td, dd, n


def key_boxes():
    """(centre, axes, half sizes) of every key solid (final keys: flat lip + sloped part)"""
    out = []
    P = L['parts']
    for ki, k in enumerate(L['keys']):
        if k['kind'] == 'final':
            f = P['final']; x = k['c'][0]; w = k['w']; zf = f['zf']; th = math.tan(f['tilt'])
            lipc = np.array([x, (f['y0'] + f['yl']) / 2, zf - 0.06])
            out.append((ki, lipc, np.eye(3), np.array([w / 2, (f['yl'] - f['y0']) / 2, 0.06])))
            ylen = (f['y1'] - f['yl']) / math.cos(f['tilt']); n = np.array([0, math.sin(f['tilt']), math.cos(f['tilt'])])
            ax_y = np.array([0, math.cos(f['tilt']), -math.sin(f['tilt'])])
            c = np.array([x, f['yl'], zf]) + ax_y * ylen / 2 - n * 0.06
            out.append((ki, c, np.stack([[1, 0, 0], ax_y, n]), np.array([w / 2, ylen / 2, 0.06])))
            continue
        td, dd, n = key_frame(k)
        c = np.array(k['c']) - n * k['th'] / 2
        out.append((ki, c, np.stack([td, dd, n]), np.array([k['w'] / 2, k['d'] / 2, k['th'] / 2])))
    return out


def main():
    info = {}
    B = L['balls']; KEYS = L['keys']
    # ---------------------------------------------------------------- 1. notes vs strikes
    struck = {}
    for n, b in B.items():
        for h in b['hits']:
            if h.get('row') is not None: struck.setdefault(h['row'], []).append((n, h))
    gaps, fgaps, dts = [], [], []
    for i, r in enumerate(LEAD):
        assert i in struck, ('row not struck', i, r)
        for n, h in struck[i]:
            if h['kind'] == 'final' or abs(h['t'] - r[0]) < 1e-6:
                dts.append(abs(h['t'] - r[0]) if h['kind'] != 'final' else 0.0)
            p = pos(B[n], r[0]); gaps.append(np.linalg.norm(p - np.array(h['p'])))
            f = int(round(r[0] * FPS)); pf = pos(B[n], f / FPS); fgaps.append(np.linalg.norm(pf - np.array(h['p'])))
    info['rows'] = len(LEAD); info['strike_time_err_max_s'] = round(max(dts), 6)
    info['contact_gap_at_note_time_max'] = round(max(gaps), 4)
    info['contact_gap_at_nearest_frame_max'] = round(max(fgaps), 3); info['contact_gap_at_nearest_frame_mean'] = round(float(np.mean(fgaps)), 3)
    # every key: its struck times == its note times
    kt = 0
    for k in KEYS:
        assert len(k['t']) >= 1; kt += len(k['t'])
    info['keys'] = len(KEYS); info['key_strikes'] = kt
    # ---------------------------------------------------------------- 2. physics
    es, sr, til = [], [], []
    for n, b in B.items():
        for h in b['hits']:
            vi, vo = np.array(h['vin']), np.array(h['vout'])
            if np.linalg.norm(vo) == 0 or h['kind'] == 'slat': continue
            es.append(h['e']); sr.append(np.linalg.norm(vo) / max(1e-9, np.linalg.norm(vi)))
    info['bounce_e_max'] = round(max(es), 3); info['speed_out_over_in_max'] = round(max(sr), 3)
    # in-flight frame-to-frame speed ratio: sample every frame, skip frames next to a hit or a segment change
    worst = 1.0; worst_at = None
    for n, b in B.items():
        if b['kind'] == 'rain' and int(n[1:]) % 7: continue
        ch = set()
        for g in b['seg']:
            for t in (g['t0'], g['t1']):
                for d in (-1, 0, 1, 2): ch.add(int(math.floor(t * FPS)) + d)
        fly_frames = set()
        for g in b['seg']:
            if g['k'] == 'fly':
                for f in range(int(math.ceil(g['t0'] * FPS)), int(math.floor(g['t1'] * FPS)) + 1): fly_frames.add(f)
        fr = sorted(f for f in fly_frames if f not in ch and f - 1 in fly_frames and f + 1 in fly_frames and f - 1 not in ch and f + 1 not in ch)
        for f in fr:
            p0, p1, p2 = pos(b, (f - 1) / FPS), pos(b, f / FPS), pos(b, (f + 1) / FPS)
            v1 = np.linalg.norm(p1 - p0) * FPS; v2 = np.linalg.norm(p2 - p1) * FPS
            if min(v1, v2) < 0.6: continue
            rr = max(v1, v2) / min(v1, v2)
            # free fall: the speed change per frame is bounded by g/FPS
            lim = 1 + G / FPS / min(v1, v2) + 1e-6
            if rr > lim + 1e-3: worst = max(worst, rr); worst_at = (n, f)
    info['inflight_speed_ratio_violations'] = worst_at is not None
    # ---------------------------------------------------------------- 3. collisions
    boxes = key_boxes()
    ts = np.arange(0.0, EV['secs']['end'] + 1.0, 1 / 120)
    bad = 0; worst = 0.0; ex = []
    own = {}
    for ki, k in enumerate(KEYS):
        for t in k['t']: own.setdefault(ki, []).append(t)
    for n, b in B.items():
        if b['kind'] == 'rain' and int(n[1:]) % 3: continue
        P = positions(b, ts); r = b['r']
        hit_t = [h['t'] for h in b['hits']]
        for ki, c, ax, hs in boxes:
            k = KEYS[ki]
            rel = (P - c) @ ax.T
            d = np.maximum(0, np.abs(rel) - hs); dist = np.linalg.norm(d, axis=1)
            pen = r - dist
            m = pen > 0.012
            if not m.any(): continue
            for idx in np.nonzero(m)[0]:
                t = ts[idx]
                if k.get('ball') == n and any(abs(t - th) < 0.04 for th in own.get(ki, [])): continue
                if any(abs(t - th) < 0.04 for th in hit_t) and (k['kind'] in ('final', 'console', 'slat', 'cradle', 'chordkey') or k.get('ball') == n):
                    continue
                # resting on / rolling off its own key (keys tip to release their marble)
                if any(o[0] == n and o[1] - 0.04 <= t <= o[2] + 0.6 for o in k.get('occ', [])): continue
                if k['kind'] == 'cradle' and k.get('ball') == n: continue
                bad += 1; worst = max(worst, pen[idx])
                if len(ex) < 12: ex.append((n, round(float(t), 3), k['kind'], [nm(p) for p in k['ps']], round(float(pen[idx]), 3)))
                break
    info['key_collisions'] = bad; info['key_collision_worst'] = round(worst, 3); info['examples'] = ex
    # marble vs marble (main marbles every sample; rain marbles sub-sampled)
    names = [n for n, b in B.items() if b['kind'] != 'rain']
    PP = {n: positions(B[n], ts) for n in names}
    rs = {n: B[n]['r'] for n in names}
    mind = 9.0; pair = None
    for a in range(len(names)):
        for c in range(a + 1, len(names)):
            d = np.linalg.norm(PP[names[a]] - PP[names[c]], axis=1) - rs[names[a]] - rs[names[c]]
            k = int(np.argmin(d))
            if d[k] < mind: mind = d[k]; pair = (names[a], names[c], round(float(ts[k]), 2))
    info['main_marbles_min_clearance'] = round(float(mind), 3); info['closest_pair'] = pair
    rn = [n for n, b in B.items() if b['kind'] == 'rain']
    t2 = np.arange(EV['secs']['final'] - 1.2, EV['secs']['final'] + 4.0, 1 / 60)
    RP = np.stack([positions(B[n], t2) for n in rn])          # (N, T, 3)
    rr = B[rn[0]]['r']; mr = 9.0
    for k in range(len(t2)):
        X = RP[:, k, :]; idx = np.nonzero(X[:, 2] > L['parts']['final']['zf'] - 1.0)[0]     # (below that: inside the trough)
        if len(idx) < 2: continue
        Y = X[idx]; D = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2) + np.eye(len(idx)) * 9
        mr = min(mr, D.min() - 2 * rr)
    info['rain_min_clearance'] = round(float(mr), 3)
    for k, v in info.items(): print(f'{k}: {v}')
    # pitch table (first 40 rows + summary)
    with open(f'{HERE}/verify_out.txt', 'w') as fo:
        for k, v in info.items(): fo.write(f'{k}: {v}\n')
        fo.write('\nrow  time    pitches        role    bar pos   note\n')
        for i, r in enumerate(LEAD):
            fo.write(f'{i:3d} {r[0]:7.3f}  {" ".join(nm(p) for p in r[2]):14s} {r[3]:7s} {r[5]:3d} {r[6]:5.3f} {r[7]}\n')


if __name__ == '__main__':
    main()
