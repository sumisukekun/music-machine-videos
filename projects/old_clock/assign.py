"""Who plays which note in the parents' duets (dynamic programming): each ball's consecutive notes are either a hop
(0.14 s <= gap <= 1.3 s; long hops cost more) or a cup transfer (gap >= 1.3 s, fixed cost); alternating is preferred."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
TN = [l[0] for l in json.load(open(f'{HERE}/lead.json'))]
INF = 1e9


def gcost(g):
    if g < 0.14: return INF
    if g < 1.2: return max(0, g - 1.0) * 6 + (0.3 if g < 0.25 else 0)
    return 4.0


def assign(lo, hi, first_owner, fixed={}, unison=(), same_pen=1.2):
    st = {(first_owner, -1): (0.0, [first_owner])}
    for k in range(lo + 1, hi + 1):
        ns = {}
        for (own, lo_other), (c, hist) in st.items():
            for who in (0, 1):
                if k in fixed and fixed[k] != who: continue
                if who == own: cc = gcost(TN[k] - TN[k - 1]) + same_pen; last_o = lo_other
                else:
                    cc = gcost(TN[k] - TN[lo_other]) if lo_other >= 0 else 0.0
                    last_o = k - 1
                if k in unison:
                    cc += gcost(TN[k] - TN[last_o]) if last_o >= 0 else 0.0
                    last_o = k
                key = (who, last_o); tot = c + cc
                if key not in ns or ns[key][0] > tot: ns[key] = (tot, hist + [who])
        st = ns
    return min(st.values(), key=lambda v: v[0])


if __name__ == '__main__':
    c, h = assign(63, 114, 1, fixed={63: 1, 64: 0, 114: 0}, unison=(77,))
    print(c); print(' '.join(f'{k}{"GR"[w]}' for k, w in zip(range(63, 115), h)))
