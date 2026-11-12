"""Physics helpers for a course that spirals around a tower (axis = world z through the origin).
One constant gravity G; free flight = straight parabola in world space; a bounce is an impulse along the key normal with
EMIN <= e <= EMAX (never adds speed).  The beam-search planner chooses landing points along a helix target
(angle theta(t), height z(t), radius r(t)); every flight is solved in world coordinates, so no curved 'unrolled' flight."""
import math
import numpy as np
G = 5.0
EMAX, EMIN, TILT = 0.95, 0.22, 34.0


class Ball:
    def __init__(s, name, r): s.name = name; s.r = r; s.seg = []; s.hits = []
    def rest(s, t0, t1, p): s.seg.append(dict(k='rest', t0=float(t0), t1=float(t1), p=[float(c) for c in p]))
    def fly(s, t0, t1, p0, v0): s.seg.append(dict(k='fly', t0=float(t0), t1=float(t1), p=[float(c) for c in p0], v=[float(c) for c in v0]))
    def path(s, ts, ps): s.seg.append(dict(k='path', t0=float(ts[0]), t1=float(ts[-1]), ts=[float(t) for t in ts], ps=[[float(c) for c in p] for p in ps]))
    def at(s, t):
        for g in s.seg:
            if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9: return seg_at(g, t)
        if t < s.seg[0]['t0']: return seg_at(s.seg[0], s.seg[0]['t0'])
        return seg_at(s.seg[-1], s.seg[-1]['t1'])
    def hit(s, t, p, vin, vout, kind, pitch=None, ev=None, **kw):
        e, n = bounce_e(np.array(vin, float), np.array(vout, float))
        d = dict(t=float(t), p=[float(c) for c in p], vin=[float(c) for c in vin], vout=[float(c) for c in vout], n=[float(c) for c in n],
                 e=float(e), kind=kind, pitch=pitch, ev=ev); d.update(kw); s.hits.append(d); return d


def seg_at(g, t):
    if g['k'] == 'rest': return np.array(g['p'])
    if g['k'] == 'fly':
        u = t - g['t0']; return np.array(g['p']) + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
    ts = np.array(g['ts']); ps = np.array(g['ps'])
    return np.array([np.interp(t, ts, ps[:, c]) for c in range(3)])


def bounce_e(vin, vout):
    d = vout - vin; nn = np.linalg.norm(d)
    if nn < 1e-9: return 1.0, np.array([0, 0, 1.0])
    n = d / nn; a = vin @ n
    return (-(vout @ n) / a if a < -1e-9 else 9.0), n


def cyl(th, r, z): return np.array([r * math.cos(th), r * math.sin(th), z])


def key_axes(top, n):
    """key frame: xa along the course (tangent of the circle through the key, made normal to n), ya = n x xa"""
    top = np.asarray(top, float); n = np.asarray(n, float)
    th = np.arctan2(top[..., 1], top[..., 0])
    tg = np.stack([-np.sin(th), np.cos(th), np.zeros_like(th)], -1)
    xa = tg - (tg * n).sum(-1, keepdims=True) * n; xa /= np.linalg.norm(xa, axis=-1, keepdims=True) + 1e-12
    return xa, np.cross(n, xa)


def pen_box(P, top, n, ext, R):
    """penetration of balls (centres P, radius R) into key boxes (top centre `top`, normal n, half extents ext).
    P: (C, S, 3) with top, n: (C, 3), or P: (S, 3) with top, n: (3,). >0 = overlap"""
    top = np.asarray(top, float); n = np.asarray(n, float)
    xa, ya = key_axes(top, n)
    c = top - n * ext[2]
    if P.ndim == 3 and c.ndim == 2: c, xa, ya, nn = c[:, None], xa[:, None], ya[:, None], n[:, None]
    else: nn = n
    rel = P - c
    lx = np.abs((rel * xa).sum(-1)) - ext[0]; ly = np.abs((rel * ya).sum(-1)) - ext[1]; lz = np.abs((rel * nn).sum(-1)) - ext[2]
    d = np.sqrt(np.maximum(lx, 0) ** 2 + np.maximum(ly, 0) ** 2 + np.maximum(lz, 0) ** 2)
    inside = (lx < 0) & (ly < 0) & (lz < 0)
    return np.where(inside, R + 0.1, R - d)


def fly_pts(p, v, taus):
    return p + v * taus[..., None] + np.array([0, 0, -0.5 * G]) * (taus[..., None] ** 2)


def plan(times, p0, vins, tht, zt, rt, R, ext, emax=EMAX, emin=EMIN, tilt=TILT, W=220, NX=17, NZ=31, wlim=(0.12, 0.75),
         land_min=0.3, final=None, dzr=(-2.0, 1.0), hist0=(), keys0=(), r_ref=3.5, smooth=2.0):
    """Beam search over landing points for a ball that strikes at `times` (contact at times[0] = p0 with incoming vins).
    Targets: angle tht(t), height zt(t), radius rt(t) (radius is not searched: it is taken from rt).  The angular step of
    each hop is searched in wlim (rad/s), the height step in dzr (scaled by hop length).  Legal bounce: emin <= e <= emax,
    key tilt <= tilt, ball still falling >= land_min at the next contact, no flight through a key (all keys stay forever:
    the last few keys and flights of this ball are checked).  final = (theta, z): the last landing is exactly there."""
    ct = math.cos(math.radians(tilt))
    th0 = math.atan2(p0[1], p0[0]); th0 += 2 * math.pi * round((tht(times[0]) - th0) / (2 * math.pi))
    S = [(0.0, np.array(p0, float), th0, [np.array(v, float) for v in vins], (), tuple(hist0), tuple(keys0))]
    n = len(times)
    for k in range(n - 1):
        d = times[k + 1] - times[k]; t1 = times[k + 1]; tk = times[k]
        dn = times[k + 2] - times[k + 1] if k + 2 < n else d
        need = (0.5 * G * dn) ** 2 + (wlim[0] * r_ref) ** 2
        r1 = rt(t1); new = []
        last = (k == n - 2) and final is not None
        tau_out = np.append(np.arange(0.03, d - 0.02, 0.025), d - 0.004)
        for c, p, th, vl, h, fl, keys in S:
            if last:
                TH = np.array([final[0]]); Z1 = np.array([final[1]])
            else:
                ws = np.linspace(wlim[0], wlim[1], NX); dzs = np.linspace(dzr[0], dzr[1], NZ) * max(0.4, d / 0.66)
                WW, DZ = np.meshgrid(ws, dzs); TH = th + WW.ravel() * d; Z1 = p[2] + DZ.ravel()
            P1 = np.stack([r1 * np.cos(TH), r1 * np.sin(TH), Z1], 1)
            VO = (P1 - p) / d; VO[:, 2] += 0.5 * G * d
            ok = np.ones(len(TH), bool); NRM = None
            for v in vl:
                dv = VO - v; nn = np.linalg.norm(dv, axis=1) + 1e-9; nr = dv / nn[:, None]
                if NRM is None: NRM = nr
                a = nr @ v; b = (VO * nr).sum(1)
                e = np.where(a < -1e-9, -b / np.minimum(a, -1e-9), 9)
                ok &= (e >= emin) & (e <= emax) & (nr[:, 2] >= ct)
            land = VO[:, 2] - G * d
            ok &= land <= -land_min
            if not ok.any(): continue
            idx = np.nonzero(ok)[0]
            VOk = VO[idx]; NR = NRM[idx]; TOP = p - R * NR
            if len(tau_out):
                P_out = p + VOk[:, None, :] * tau_out[None, :, None] + np.array([0, 0, -0.5 * G]) * (tau_out[None, :, None] ** 2)
                bad = (pen_box(P_out, TOP, NR, ext, R) > 0.004).any(1)
                for (ktop, kn, kext) in keys:
                    bad |= (pen_box(P_out, np.broadcast_to(ktop, TOP.shape), np.broadcast_to(kn, NR.shape), kext, R) > 0.004).any(1)
            else: bad = np.zeros(len(idx), bool)
            for (f0, fp, fv, f1) in fl:
                tt = np.arange(f0 + 0.03, f1 - 0.03, 0.025)
                if not len(tt): continue
                P_in = fly_pts(np.array(fp), np.array(fv), tt - f0)
                bad |= (pen_box(np.broadcast_to(P_in, (len(idx),) + P_in.shape), TOP, NR, ext, R) > 0.004).any(1)
            THt = tht(t1); Zt = zt(t1)
            vh_in = math.hypot(vl[0][0], vl[0][1])
            for jj in np.nonzero(~bad)[0]:
                j = idx[jj]
                p1 = P1[j]
                ke = (VO[j, 0] ** 2 + VO[j, 1] ** 2 + land[j] ** 2) * emax ** 2
                wgt = 1.0 + (8.0 * (k / max(1, n - 2)) ** 2 if final is not None else 0.0)
                cc = c + wgt * (((TH[j] - THt) * r1) ** 2 + (p1[2] - Zt) ** 2) + 3.0 * max(0.0, need - ke) + smooth * (math.hypot(VO[j, 0], VO[j, 1]) - vh_in) ** 2
                vo = VO[j]; vland = np.array([VO[j, 0], VO[j, 1], land[j]])
                fl2 = tuple(fl)[-1:] + ((tk, p, vo, t1),)
                ks2 = tuple(keys)[-4:] + ((TOP[jj], NR[jj], ext),)
                new.append((cc, p1, float(TH[j]), [vland], h + ((float(tk), p, vl, vo),), fl2, ks2))
        if not new:
            if last:
                for c, p, th, vl, h, fl, keys in S[:3]:
                    P1 = np.array([r1 * math.cos(final[0]), r1 * math.sin(final[0]), final[1]])
                    vo = (P1 - p) / d; vo[2] += 0.5 * G * d
                    e, nr = bounce_e(vl[0], vo)
                    print('  dead final: p', np.round(p, 2), 'vin', np.round(vl[0], 2), 'vout', np.round(vo, 2), 'e', round(e, 3),
                          'tilt', round(math.degrees(math.acos(min(1, nr[2]))), 1))
            raise RuntimeError(f'plan dead at hop {k} t={times[k]:.2f}')
        new.sort(key=lambda s: s[0]); keep = []; seen = set()
        for s_ in new:
            key = (round(s_[3][0][2] / 0.15), round(np.hypot(s_[3][0][0], s_[3][0][1]) / 0.08), round(s_[1][2] / 0.06), round(s_[2] / 0.015))
            if key in seen: continue
            seen.add(key); keep.append(s_)
            if len(keep) >= W: break
        S = keep
    best = S[0]
    hits = [dict(t=t, p=p, vins=vl, vout=vo) for t, p, vl, vo in best[4]]
    return hits, best[1], best[3][0], best[5], best[6]


def plan_retry(*a, **kw):
    err = None
    for W, NX, NZ in ((90, 15, 27), (220, 19, 35), (500, 25, 45)):
        kw.update(W=W, NX=NX, NZ=NZ)
        try: return plan(*a, **kw)
        except RuntimeError as e_: err = e_; print('retry', e_)
    raise err


def legal(vin, vout, emin=EMIN, emax=EMAX, tilt=TILT):
    e, n = bounce_e(np.array(vin, float), np.array(vout, float))
    return emin <= e <= emax and n[2] >= math.cos(math.radians(tilt))


def exit_to(p, vin, d, target, spread=0.35, nz=41):
    """legal outgoing velocity from contact p (incoming vin) that lands after d seconds as near as possible to target"""
    best = None
    for dx in np.linspace(-spread, spread, nz):
        for dy in np.linspace(-spread, spread, nz):
            for dz in np.linspace(-spread, spread, 15):
                q = np.array(target) + [dx, dy, dz]
                vo = (q - p) / d + np.array([0, 0, 0.5 * G * d])
                if not legal(vin, vo): continue
                c = dx * dx + dy * dy + dz * dz
                if best is None or c < best[0]: best = (c, vo, q)
    assert best is not None, ('no legal exit', p, vin, d, target)
    vo, q = best[1], best[2]
    return vo, q, vo + np.array([0, 0, -G * d])


def trapezoid(L, T, ta):
    vc = L / (T - ta)
    def s(u):
        if u <= 0: return 0.0
        if u >= T: return L
        if u < ta: return 0.5 * vc / ta * u * u
        if u > T - ta: w = T - u; return L - 0.5 * vc / ta * w * w
        return 0.5 * vc * ta + vc * (u - ta)
    return s, vc
