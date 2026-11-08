"""Physical layout for the era medley marble (run after arrange.py; writes layout.json, read by scene.py).

Why: the old path keyed the marble between strikers with an ad-hoc curve (stay up, then fall with u^1.6) and
placed strikers by a fixed descent rate, so every gap had its own speed and the marble visibly lurched
(slow drift, then a sudden dash).  Here the marble obeys one rule for the whole video:
  * between notes it flies a pure gravity parabola (one constant G),
  * at each note it hits a striker whose surface normal n (tilt theta) gives a rigid-impact bounce:
    the normal part is returned with restitution e, the tangential part kept with factor f (<= 1),
    so it never gains speed at a hit,
  * the next striker is placed exactly where that parabola is at the next note time (timing comes from
    geometry, never from changing speed).
A beam search picks (theta, e, f) per note so the strikers stay separated, the marble always lands from
above, never passes through any striker, and the wall keeps descending."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))

G = 7.0                      # gravity (wall units / s^2), identical for the whole video
R = 0.14                     # marble radius
XW = 60.0                    # soft: XPULL keeps it near the middle
DXMIN, DXMAX = 0.43, 2.4
HWC, TKC = 0.2, 0.3         # collision half-width / thickness used while planning
VCRUISE, VMAX = 3.5, 8.5
VHIT, WHIT = 4.2, 60.0          # keep impact speeds low so every hit is visible on a 30 fps frame
H0 = 1.6                     # first free-fall height (from rest)
BEAM = 64
XPULL = 0.3
DZW = 3
TW = 2
NB, BX0, BW = 1500, -60.0, 0.08
_th = np.radians(np.linspace(-85, 85, 35)); _e = np.linspace(0.3, 0.9, 5); _f = np.array([0.6, 0.8, 1.0])
CT, CE, CF = [a.ravel() for a in np.meshgrid(_th, _e, _f, indexing='ij')]
CNX, CNZ = np.sin(CT), np.cos(CT)
NS = 16
def _bins(x): return np.clip(((x - BX0) / BW).astype(int), 0, NB - 1)
def _raster(xs, zs, half):
    """per-bin min z of points xs/zs (K,P) widened by +-half -> (K,NB)"""
    K = xs.shape[0]; out = np.full((K, NB), np.inf)
    for off in np.linspace(-half, half, max(2, int(2 * half / BW) + 2)):
        b = _bins(xs + off)
        np.minimum.at(out, (np.repeat(np.arange(K), xs.shape[1]), b.ravel()), zs.ravel())
    return out
def plan(ht):
    """Rigid impact at every note: the striker surface (normal n, tilt theta) keeps a fraction f of the tangential
    velocity and returns e of the normal one, so the marble never gains speed at a hit. Between notes it flies a
    pure gravity parabola; the next striker is placed exactly where that parabola is at the next note time.
    A beam search picks (theta, e, f) per note; everything new must stay below what is already on the wall."""
    n = len(ht); dt = np.diff(ht)
    x = np.array([-0.8]); z = np.array([0.0]); vx = np.array([0.0]); vz = np.array([-np.sqrt(2 * G * H0)]); cost = np.zeros(1)
    SS = np.full((1, NB), np.inf); SP = np.full((1, NB), np.inf); PEND = np.full((1, NB), np.inf)
    vxmin = DXMIN / dt; vxmin[-1] = 0.0; need = np.zeros(n)
    for j in range(n - 2, -1, -1): need[j] = max(vxmin[j] * 1.08, need[j + 1] - G * dt[j] * 0.6) / 0.93
    hist = []; sfr = (np.arange(NS) + 0.5) / NS
    for i in range(n - 1):
        d = dt[i]; last = (i + 1 == n - 1); K = len(x)
        dot = vx[:, None] * CNX + vz[:, None] * CNZ
        tx, tz = vx[:, None] - dot * CNX, vz[:, None] - dot * CNZ
        ox = CF * tx - CE * dot * CNX; oz = CF * tz - CE * dot * CNZ
        dx = ox * d; dz = oz * d - 0.5 * G * d * d; nxv = ox; nzv = oz - G * d
        vo = np.hypot(ox, oz); vn = np.hypot(nxv, nzv); X1 = x[:, None] + dx; Z1 = z[:, None] + dz
        c = cost[:, None] + np.where(dot < -1e-3, 0, 1e9)
        c += np.where(nzv < -0.3, 0, 1e6 + 1e5 * (nzv + 0.3))
        c += 300 * np.maximum(0, need[i + 1] - vn) ** 2 + WHIT * np.maximum(0, vn - VHIT) ** 2
        c += 2 * ((vo - VCRUISE) / VCRUISE) ** 2 + 50 * np.maximum(0, vn - VMAX) ** 2
        c += TW * (CT / np.radians(45)) ** 4 + 0.3 * (1 - CF) + 0.4 * (CE - 0.65) ** 2
        c += 40 * np.maximum(0, np.maximum(0, oz) ** 2 / (2 * G) - 1.2) ** 2
        c += 2000 * np.maximum(0, dz + 0.04) ** 2 + DZW * dz
        # path samples (ball centre)
        s = d * sfr
        px = x[:, None, None] + ox[:, :, None] * s; pz = (z[:, None, None] + R) + oz[:, :, None] * s - 0.5 * G * s * s
        kk = np.arange(K)[:, None, None]
        top = pz + R
        lim = np.minimum(np.minimum(SS[kk, _bins(px - R)], SS[kk, _bins(px + R)]), SS[kk, _bins(px)])
        c += 1e4 * np.maximum(0, top - lim + 0.04).sum(-1)
        nxc, nzc = CNX[None, :, None], CNZ[None, :, None]           # striker i's own (tilted) surface
        rx = px - (x[:, None, None] - R * nxc); rz = pz - (z[:, None, None] + R - R * nzc)
        uu = rx * nzc - rz * nxc; ww = rx * nxc + rz * nzc
        own = (np.abs(uu) < HWC + R) & (ww < R - 0.02) & (ww > -TKC - R) & (s > 0.03)
        c += 1e4 * own.sum(-1)
        # new striker vs everything older
        SPall = np.minimum(SP, PEND); kk2 = np.arange(K)[:, None]
        lim2 = np.inf
        for off in (-HWC, 0, HWC):
            b = _bins(X1 + off); lim2 = np.minimum(lim2, np.minimum(SPall[kk2, b], SS[kk2, b]))
        c += 1e4 * np.maximum(0, Z1 + 0.05 - lim2 + 0.04)
        c += 1e4 * ((np.abs(dx) < 2 * HWC + 0.02) & (dz > -(TKC + 0.08)))
        if last:
            c += 500 * X1 ** 2 + 200 * np.maximum(0, dz + 1.2) ** 2
        else:
            c += 3000 * np.maximum(0, DXMIN - np.abs(dx)) ** 2 + 1000 * np.maximum(0, np.abs(X1) - XW) ** 2
            c += 300 * np.maximum(0, np.abs(dx) - DXMAX) ** 2 + XPULL * np.maximum(0, np.abs(X1) - 3.0) ** 2
        flat = c.ravel(); Kn = min(BEAM, flat.size)
        idx = np.argpartition(flat, Kn - 1)[:Kn]; idx = idx[np.argsort(flat[idx])]
        pi, ci = np.unravel_index(idx, c.shape)
        hist.append((pi, ci, ox[pi, ci], oz[pi, ci], dx[pi, ci], dz[pi, ci], vx[pi], vz[pi]))
        # update skylines: striker i joins SS; path i-1 (PEND) joins SP; path i becomes PEND
        SS_old = SS[pi]; SP = np.minimum(SP[pi], PEND[pi])
        sx = x[pi][:, None] + np.array([[-HWC, 0, HWC]]); sz = np.repeat((z[pi] - TKC)[:, None], 3, 1)
        SS = np.minimum(SS_old, _raster(sx, sz, 0.02))
        ss = d * np.linspace(0, 1, 16)
        ppx = x[pi][:, None] + ox[pi, ci][:, None] * ss; ppz = z[pi][:, None] + R + oz[pi, ci][:, None] * ss - 0.5 * G * ss * ss - R
        PEND = _raster(ppx, ppz, R)
        x = X1[pi, ci]; z = Z1[pi, ci]; vx = nxv[pi, ci]; vz = nzv[pi, ci]; cost = c[pi, ci]
    k = 0; Xs = np.zeros(n); Zs = np.zeros(n); V0 = np.zeros((n, 2)); VIN = np.zeros((n, 2)); TH = np.zeros(n)
    VIN[-1] = (vx[0], vz[0]); steps = []
    for i in range(n - 2, -1, -1):
        pi, ci, OX, OZ, DX, DZ, IX, IZ = hist[i]
        steps.append((i, OX[k], OZ[k], DX[k], DZ[k], IX[k], IZ[k], CT[ci[k]])); k = pi[k]
    steps.reverse(); Xs[0] = -0.8
    for i, ox_, oz_, dx_, dz_, ix, iz, th in steps:
        V0[i] = (ox_, oz_); VIN[i] = (ix, iz); TH[i] = th; Xs[i + 1] = Xs[i] + dx_; Zs[i + 1] = Zs[i] + dz_
    sp_in = np.hypot(VIN[:, 0], VIN[:, 1]); sp_out = np.hypot(V0[:, 0], V0[:, 1]); r = sp_out[:-1] / sp_in[:-1]
    adx = np.abs(np.diff(Xs))[:-1]
    return Xs, Zs, V0, TH, VIN, dict(cost=round(float(cost[0]), 1), out_in_max=round(r.max(), 3), out_in_min=round(r.min(), 2),
        tilt_max=round(np.degrees(np.abs(TH)).max(), 1), dx_min=round(adx.min(), 2), dx_max=round(adx.max(), 2),
        xmax=round(np.abs(Xs).max(), 2), drop=round(Zs[0] - Zs[-1], 1), vmax=round(sp_in.max(), 1),
        land_vz_max=round(VIN[1:, 1].max(), 2), apex=round((np.maximum(0, V0[:, 1]) ** 2 / (2 * G)).max(), 2))


def hits_from_lead():
    E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json')); SECS = E['secs']
    def era_of(t, s):
        if s >= 6: return 6
        return s if t >= SECS[s][2] - 1e-6 else -1
    hits = []
    for t, p, v, s in LEAD:                                   # merge simultaneous notes (final chord)
        if hits and abs(t - hits[-1][0]) < 1e-3: continue
        hits.append([t, p, v, s, era_of(t, s)])
    return hits


def check(X, Z, TH, V0, ht, HW=HWC, TK=TKC, step=1 / 240):
    """sample the analytic path; count samples where the marble overlaps a striker other than at its hit"""
    nx, nz = np.sin(TH), np.cos(TH); cx = X - R * nx; czp = Z + R - R * nz; bad = 0
    for t in np.arange(ht[0], ht[-1], step):
        i = int(np.searchsorted(ht, t, 'right') - 1); s = t - ht[i]
        b = (X[i] + V0[i][0] * s, Z[i] + R + V0[i][1] * s - 0.5 * G * s * s)
        rx, rz = b[0] - cx, b[1] - czp
        u = rx * nz - rz * nx; w = rx * nx + rz * nz
        du = np.maximum(0, np.abs(u) - HW); dw = np.where(w > 0, w, np.maximum(0, -w - TK))
        hit = (np.hypot(du, dw) < R - 0.015) & (np.abs(ht - t) > 0.03); hit[-1] = False
        bad += int(hit.sum())
    return bad


if __name__ == '__main__':
    hits = hits_from_lead(); ht = np.array([h[0] for h in hits])
    X, Z, V0, TH, VIN, info = plan(ht)
    info['collisions'] = check(X, Z, TH, V0, ht)
    gap = []                                  # marble centre at the rendered frame nearest each hit vs. at the hit
    for i in range(len(ht) - 1):
        s_ = round(ht[i] * 30) / 30 - ht[i]; j = i if s_ >= 0 else i - 1; s_ = round(ht[i] * 30) / 30 - ht[j]
        b = (X[j] + V0[j][0] * s_, Z[j] + R + V0[j][1] * s_ - 0.5 * G * s_ * s_) if j >= 0 else (X[0], Z[0] + R)
        gap.append(np.hypot(b[0] - X[i], b[1] - Z[i] - R))
    SHUT = 0.5                                # render motion-blur shutter (fraction of a frame, centred)
    blur = [max(0.0, g * (1 - SHUT / 2 * 30 / 30 / max(1e-9, abs(round(t * 30) / 30 - t) * 30) )) if abs(round(t * 30) / 30 - t) * 30 > SHUT / 2 else 0.0
            for g, t in zip(gap, ht)]
    info['blur_gap_max'] = round(max(blur), 3)
    info['frame_gap_max'] = round(max(gap), 3); info['frame_gap_mean'] = round(float(np.mean(gap)), 3)
    print(info)
    json.dump({'G': G, 'R': R, 'H0': H0, 'HW': HWC, 't': ht.round(5).tolist(), 'x': X.round(6).tolist(), 'z': Z.round(6).tolist(),
               'tilt': TH.round(6).tolist(), 'v0': V0.round(6).tolist(), 'vin': VIN.round(6).tolist()}, open(f'{HERE}/layout.json', 'w'))
