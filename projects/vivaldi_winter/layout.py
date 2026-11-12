"""Vivaldi 'Winter' II Largo - "Will the snowmen smile?" (雪の夜の暖炉). Physics + story planner.
Run after arrange.py; writes layout.json (read by scene.py, render_audio.py, verify.py).

Cast (side view of a long log cabin on a blizzard night, cut open towards the camera):
  mel    the ember ball (glowing orange, solo violin)   copper keys along the hearth (front lane)
  bas    the iron ball (cello)                          firewood log drums on the bench (back lane)
  rain1 / rain2  raindrops (violins 1 / 2, pizzicato)   glass chimes on two rows outside the windows; every 16th note one
                 drop falls from the sky onto the next chime of each row (the 'La Pioggia' of Vivaldi's score)
  snowmen        sad snowmen in the yard behind the cabin, a hero snowman outside the end of the cabin holding a lantern
Story
  0-2 s     blizzard, a sad snowman outside the first window. The stove door pops (log crackle, sparks) and throws the ember
            ball onto its first key; the iron ball's shelf flap opens; the rain begins
  bars 1-11 the Largo. every struck key / drum / chime lights up
  bar 12    a gust blows the shutters open: snow storms in, every fire dims (the music goes on)
  bar 14    the iron ball's drum is the board of a big bellows: the puff relights the fireplace, the shutters slam shut
  bar 16 b3 the ember ball's long E-flat lands in the bucket bell of the chimney elevator. The roaring fire turns the smoke-jack
            fan at the top of the chimney, which drives the chain: the bucket rises, tips the ember onto the roof chute; it rolls
            down (constant acceleration), leaves the eave and lands at bar 18 (final chord) in the hero snowman's lantern
  final     the lantern lights, the hero smiles, the warmth runs back along the row: every snowman lights its lantern and smiles
Physics: one constant gravity G; between contacts pure parabolas; every bounce is an impulse along the key's surface normal
(tangential velocity kept) with restitution EMIN <= e <= EMAX, so a bounce never adds speed (fixed note times force the keys
to tilt). Energy sources are visible: the stove pop, the shelf flap (drop), the bellows-free chain elevator driven by the fan."""
import json, os, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = 7.0
EMAX, EMIN, TILT = 0.97, 0.22, 34.0
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
T0, Q, FINAL = E['t0'], E['q'], E['final']
V = 1.0                                             # course speed (m/s) shared by every lane
T_SLOW, V_SLOW = 99.0, 1.0                          # (no slow-down: the rain rows and the bench run on to the end)
GZ = 0.03                                           # gentle descent of the course per second (the cabin steps down the hill)
LANES = {   # y, start z, radius, key half sizes (hw along x, hd along y, half thickness)
    'mel': dict(y=0.0, z=0.6, r=0.2, ext=(0.1, 0.28, 0.05)),
    'bas': dict(y=1.7, z=1.7, r=0.24, ext=(0.21, 0.21, 0.12)),
}
RAIN = {    # glass chime rows outside: y, top height above the course, x offset, tilt (deg, top faces away from the camera)
    'rain1': dict(y=4.1, dz=3.25, dx=0.75, tilt=8.0, r=0.06, ext=(0.1, 0.09, 0.03)),
    'rain2': dict(y=4.7, dz=2.85, dx=0.875, tilt=8.0, r=0.06, ext=(0.1, 0.09, 0.03)),
}
RAIN_H = 3.5; RAIN_E = 0.25
SFX = []; EV = {}
KEY_LEAD, KEY_STAY = 0.7, 0.6       # keys exist from t-0.7 s (rising) to t+0.6 s (sinking away)


def XC(t):
    """course reference x"""
    return V * (min(t, T_SLOW) - T0) + V_SLOW * max(0.0, t - T_SLOW)


def ZC(t): return -GZ * (t - T0)


GZB = 0.088                                         # the bench and the yard step down with the ember's hearth
def ZB(t): return -GZB * (t - T0)


def evs(role): return [e for e in LEAD if e['role'] == role]


class Ball:
    def __init__(s, name, r): s.name = name; s.r = r; s.seg = []; s.hits = []
    def rest(s, t0, t1, p): s.seg.append(dict(k='rest', t0=float(t0), t1=float(t1), p=[float(c) for c in p]))
    def fly(s, t0, t1, p0, v0): s.seg.append(dict(k='fly', t0=float(t0), t1=float(t1), p=[float(c) for c in p0], v=[float(c) for c in v0]))
    def path(s, ts, ps): s.seg.append(dict(k='path', t0=float(ts[0]), t1=float(ts[-1]), ts=[float(t) for t in ts], ps=[[float(c) for c in p] for p in ps]))
    def hit(s, t, p, vin, vout, kind, pitch=None, ev=None, **kw):
        e, n = bounce_e(np.array(vin, float), np.array(vout, float))
        d = dict(t=float(t), p=[float(c) for c in p], vin=[float(c) for c in vin], vout=[float(c) for c in vout], n=[float(c) for c in n],
                 e=float(e), kind=kind, pitch=pitch, ev=ev); d.update(kw); s.hits.append(d); return d


def bounce_e(vin, vout):
    d = vout - vin; nn = np.linalg.norm(d)
    if nn < 1e-9: return 1.0, np.array([0, 0, 1.0])
    n = d / nn; a = vin @ n
    return (-(vout @ n) / a if a < -1e-9 else 9.0), n


def axes_of(n):
    up = np.zeros_like(n); up[..., 1] = 1.0
    xa = np.cross(up, n); xa /= np.linalg.norm(xa, axis=-1, keepdims=True) + 1e-12
    return xa, np.cross(n, xa)


def pen_box(P, top, n, ext, R):
    """penetration of balls (centres P, radius R) into key boxes whose top centre is `top` with normal n"""
    xa, ya = axes_of(np.asarray(n, float))
    c = np.asarray(top) - np.asarray(n) * ext[2]
    if P.ndim == 3 and np.ndim(c) == 2: c, xa, ya, nn = c[:, None], xa[:, None], ya[:, None], np.asarray(n)[:, None]
    else: nn = np.asarray(n)
    rel = P - c
    lx = np.abs((rel * xa).sum(-1)) - ext[0]; ly = np.abs((rel * ya).sum(-1)) - ext[1]; lz = np.abs((rel * nn).sum(-1)) - ext[2]
    d = np.sqrt(np.maximum(lx, 0) ** 2 + np.maximum(ly, 0) ** 2 + np.maximum(lz, 0) ** 2)
    inside = (lx < 0) & (ly < 0) & (lz < 0)
    return np.where(inside, R + 0.1, R - d)


def fly_pts(p, v, taus):
    return p + v * taus[..., None] + np.array([0, 0, -0.5 * G]) * (taus[..., None] ** 2)


def plan(times, p0, vins, xt, zt, yt, emax=EMAX, emin=EMIN, tilt=TILT, W=200, NX=17, NZ=29, vxlim=(0.5, 3.4),
         land_min=0.3, final=None, dzr=(-2.0, 1.0), R=0.19, ext=(0.13, 0.25, 0.05)):
    """beam search over landing points (x, z) for a ball that strikes at `times` (from blue_danube, own copy).
    Legal bounce: emin <= e <= emax, key tilt <= `tilt`, still falling >= land_min m/s at the next contact, and no flight
    passes through a key that exists at that moment. Returns hits [dict(t, p, vins, vout)] for times[:-1], p_end, vin_end."""
    ct = math.cos(math.radians(tilt))
    S = [(0.0, np.array(p0, float), [np.array(v, float) for v in vins], (), (), ())]
    n = len(times)
    for k in range(n - 1):
        d = times[k + 1] - times[k]; t1 = times[k + 1]; tk = times[k]
        dn = times[k + 2] - times[k + 1] if k + 2 < n else d
        need = (0.5 * G * dn) ** 2 + vxlim[0] ** 2
        y1 = yt(t1); new = []
        last = (k == n - 2) and final is not None
        tau_out = np.append(np.arange(0.03, d - 0.02, 0.02), d - 0.004)
        for c, p, vl, h, fl, keys in S:
            dy = y1 - p[1]
            if last:
                VX = np.array([(final[0] - p[0]) / d]); DZ = np.array([final[1] - p[2]])
            else:
                vxo = np.linspace(max(vxlim[0], (2 * ext[0] + 0.04) / d), max(vxlim[1], (2 * ext[0] + 0.04) / d + 1.0), NX); dzs = np.linspace(dzr[0], dzr[1], NZ) * max(0.4, d / 0.66)
                VX, DZ = np.meshgrid(vxo, dzs); VX = VX.ravel(); DZ = DZ.ravel()
            VY = np.full_like(VX, dy / d); VZ = DZ / d + 0.5 * G * d
            ok = np.ones(len(VX), bool); NRM = None
            for v in vl:
                dvx, dvy, dvz = VX - v[0], VY - v[1], VZ - v[2]
                nn = np.sqrt(dvx ** 2 + dvy ** 2 + dvz ** 2) + 1e-9
                nx, ny, nz = dvx / nn, dvy / nn, dvz / nn
                if NRM is None: NRM = np.stack([nx, ny, nz], 1)
                a = v[0] * nx + v[1] * ny + v[2] * nz; b = VX * nx + VY * ny + VZ * nz
                e = np.where(a < -1e-9, -b / np.minimum(a, -1e-9), 9)
                ok &= (e >= emin) & (e <= emax) & (nz >= ct)
            land = VZ - G * d
            ok &= land <= -land_min
            if not ok.any(): continue
            idx = np.nonzero(ok)[0]
            VO = np.stack([VX[idx], VY[idx], VZ[idx]], 1); NR = NRM[idx]; TOP = p - R * NR
            if len(tau_out):
                P_out = p + VO[:, None, :] * tau_out[None, :, None] + np.array([0, 0, -0.5 * G]) * (tau_out[None, :, None] ** 2)
                bad = (pen_box(P_out, TOP, NR, ext, R) > 0.004).any(1)
                for (tkey, ktop, kn) in keys:
                    alive = (tk + tau_out) < tkey + KEY_STAY
                    if alive.any(): bad |= (pen_box(P_out[:, alive], ktop, kn, ext, R) > 0.004).any(1)
            else: bad = np.zeros(len(idx), bool)
            for (f0, fp, fv, f1) in fl:
                lo_, hi_ = max(f0, tk - KEY_LEAD), min(f1, tk - 0.03)
                if hi_ <= lo_: continue
                tt = np.arange(lo_, hi_, 0.02); P_in = fly_pts(np.array(fp), np.array(fv), tt - f0)
                bad |= (pen_box(np.broadcast_to(P_in, (len(idx),) + P_in.shape), TOP, NR, ext, R) > 0.004).any(1)
            X = xt(t1); Z = zt(t1)
            for jj in np.nonzero(~bad)[0]:
                j = idx[jj]
                p1 = np.array([p[0] + VX[j] * d, y1, p[2] + DZ[j]])
                ke = (VX[j] ** 2 + VY[j] ** 2 + land[j] ** 2) * emax ** 2
                wgt = 1.0 + (8.0 * (k / max(1, n - 2)) ** 2 if final is not None else 0.0)
                cc = c + wgt * ((p1[0] - X) ** 2 + (p1[2] - Z) ** 2) + 3.0 * max(0.0, need - ke)
                vo = VO[jj]; vland = np.array([VX[j], VY[j], land[j]])
                fl2 = tuple(f for f in fl if f[3] > tk - KEY_LEAD - 1.0)[-6:] + ((tk, p, vo, t1),)
                ks2 = tuple(kk for kk in keys if kk[0] + KEY_STAY > tk)[-6:] + ((tk, TOP[jj], NR[jj]),)
                new.append((cc, p1, [vland], h + ((float(tk), p, vl, vo),), fl2, ks2))
        if not new: raise RuntimeError(f'plan dead at hop {k} t={times[k]:.2f}')
        new.sort(key=lambda s: s[0]); keep = []; seen = set()
        for s_ in new:
            key = (round(s_[2][0][0] / 0.08), round(s_[2][0][2] / 0.15), round(s_[1][2] / 0.06), round(s_[1][0] / 0.06))
            if key in seen: continue
            seen.add(key); keep.append(s_)
            if len(keep) >= W: break
        S = keep
    best = S[0]
    hits = [dict(t=t, p=p, vins=vl, vout=vo) for t, p, vl, vo in best[3]]
    return hits, best[1], best[2][0]


def plan_retry(*a, **kw):
    err = None
    for W, NX, NZ in ((200, 17, 29), (400, 21, 41), (700, 25, 51)):
        kw.update(W=W, NX=NX, NZ=NZ)
        try: return plan(*a, **kw)
        except RuntimeError as e_: err = e_; print('retry', e_)
    raise err


def write_run(ball, hits, events, t_end, kinds):
    """hits (from plan) for events[:-1]; flights between them up to t_end"""
    for k, h in enumerate(hits):
        e = events[k]
        ball.hit(h['t'], h['p'], h['vins'][0], h['vout'], kinds(e), e['pitch'], ev=e['i'])
        t1 = hits[k + 1]['t'] if k + 1 < len(hits) else t_end
        ball.fly(h['t'], t1, h['p'], h['vout'])


def find_hop(p, vin, dts, dz_lo, dz_hi, vx_hi, vx_lo=0.3, n=300, land_min=0.3, R=None, ext=None):
    """a legal bounce from p (incoming vin) that lands dt later (first dt of dts that works); scans landing height
    (high first) and vx. returns (vout, dt)"""
    for dt in dts:
        try: return find_hop1(p, vin, dt, dz_lo, dz_hi, vx_hi, vx_lo, n, land_min, R, ext), dt
        except RuntimeError: pass
    raise RuntimeError('no hop')


def find_hop1(p, vin, dt, dz_lo, dz_hi, vx_hi, vx_lo=0.3, n=300, land_min=0.3, R=None, ext=None):
    taus = np.arange(0.02, dt, 0.01)
    for dz in np.linspace(dz_hi, dz_lo, n):
        for vx in np.linspace(vx_hi, vx_lo, 25):
            vo = np.array([vx, 0.0, dz / dt + 0.5 * G * dt]); e, nn = bounce_e(np.array(vin), vo)
            if EMIN <= e <= EMAX and nn[2] >= math.cos(math.radians(TILT)) and vo[2] - G * dt < -land_min:
                if R is not None and (pen_box(fly_pts(np.array(p), vo, taus), np.array(p) - R * nn, nn, ext, R) > 0.004).any(): continue
                return vo
    raise RuntimeError('no hop')




BALLS = {n: Ball(n, L['r']) for n, L in LANES.items()}

# ============================================================ opening: the stove pops the ember out, the shelf flap drops the iron ball
L = LANES['mel']; b = BALLS['mel']
P1 = np.array([0.0, L['y'], L['z']])
T_POP = T0 - 0.85
PS = np.array([-1.25, L['y'] + 0.1, L['z'] + 0.45])           # inside the stove door
VP = np.array([(P1[0] - PS[0]) / 0.85, (P1[1] - PS[1]) / 0.85, (P1[2] - PS[2] + 0.5 * G * 0.85 ** 2) / 0.85])
b.rest(0.0, T_POP, PS); b.fly(T_POP, T0, PS, VP)
V_IN_MEL = VP + np.array([0, 0, -G * 0.85])
EV['stove'] = dict(p=PS.tolist(), t=T_POP)
SFX.append(['pop', T_POP])
L = LANES['bas']; b = BALLS['bas']
T_FLAP = T0 - 0.5
PB1 = np.array([0.15, L['y'], L['z']]); PB = PB1 + [0, 0, 0.5 * G * 0.5 ** 2]
b.rest(0.0, T_FLAP, PB); b.fly(T_FLAP, T0, PB, [0, 0, 0])
EV['shelf'] = dict(p=PB.tolist(), t=T_FLAP)
SFX.append(['flap', T_FLAP])

# ============================================================ the two balls' runs (beam search, cached)
import pickle, hashlib
CK = hashlib.md5((open(f'{HERE}/lead.json').read() + repr((G, EMAX, EMIN, TILT, V, T_SLOW, V_SLOW, GZ, LANES, KEY_LEAD, KEY_STAY, GZB, 6, 0.27))).encode()).hexdigest()
CF = f'/tmp/vivaldi_winter_{CK}.pkl'
END = {}
if os.path.exists(CF):
    END, segs_, hits_ = pickle.load(open(CF, 'rb'))
    for r in segs_: BALLS[r].seg = segs_[r]; BALLS[r].hits = hits_[r]
else:
    # iron ball: one run over the whole movement
    L = LANES['bas']; b = BALLS['bas']; ev_ = evs('bas'); ts = np.array([e['t'] for e in ev_]); z0 = L['z']; x0 = PB1[0]  # GZB %s
    hits, pE, vE = plan_retry(ts, PB1, [np.array([0, 0, -G * 0.5])], lambda t: x0 + XC(t), lambda t: z0 + ZB(t),
                              lambda t, y=L['y']: y, R=L['r'], ext=L['ext'])
    write_run(b, hits, ev_, ts[-1], lambda e: 'drum')
    e = ev_[-1]; b.hit(e['t'], pE, vE, [0, 0, 0], 'finbell', e['pitch'], ev=e['i'])
    END['bas'] = dict(p=pE.tolist(), vin=vE.tolist(), t=e['t'])
    print('bas done', len(hits) + 1, 'hits, end at', np.round(pE, 2))
    # ember ball: phrases between the long notes; on every long note (>= 1 s) it hops onto a level copper rail, rolls at
    # constant speed and drops off the rail end onto the next key (TD s of free fall)
    L = LANES['mel']; b = BALLS['mel']; R_ = L['r']
    ev_all = [e for e in evs('mel') if e['t'] <= E['lift'] + 1e-6]
    segs = [[ev_all[0]]]
    for a, c in zip(ev_all, ev_all[1:]):
        if c['t'] - a['t'] > 0.99 and c is not ev_all[-1]: segs.append([c])
        else: segs[-1].append(c)
    TD = 0.27
    p0 = P1; v0 = V_IN_MEL; RAILS = []
    for si, ev_ in enumerate(segs):
        ts = np.array([e['t'] for e in ev_]); xr, zr, tr = p0[0], p0[2], ts[0]
        last = si == len(segs) - 1
        if len(ts) > 1:
            hits, pE, vE = plan_retry(ts, p0, [v0], lambda t, xr=xr, tr=tr: xr + XC(t) - XC(tr),
                                      lambda t, zr=zr, tr=tr: zr + ZC(t) - ZC(tr), lambda t, y=L['y']: y, R=R_, ext=L['ext'])
            write_run(b, hits, ev_, ts[-1], lambda e: 'key')
        else: pE, vE = p0, v0
        e = ev_[-1]
        if last:
            b.hit(e['t'], pE, vE, [0, 0, 0], 'bucket', e['pitch'], ev=e['i'])
            END['mel'] = dict(p=pE.tolist(), vin=vE.tolist(), t=e['t']); break
        tn = segs[si + 1][0]['t']
        vo, dt = find_hop(pE, vE, (0.3, 0.26, 0.22, 0.18), -0.45, 0.12, 1.5, vx_lo=0.6, R=R_, ext=L['ext'])
        b.hit(ts[-1], pE, vE, vo, 'key', e['pitch'], ev=e['i'])
        b.fly(ts[-1], ts[-1] + dt, pE, vo)
        pl = pE + vo * dt + np.array([0, 0, -0.5 * G * dt * dt]); vl = vo + np.array([0, 0, -G * dt]); t_land = ts[-1] + dt
        b.hits.append(dict(t=float(t_land), p=pl.tolist(), vin=vl.tolist(), vout=[float(vl[0]), 0.0, 0.0], n=[0, 0, 1], e=0.0,
                           kind='rail', pitch=None, ev=None))
        t_off = tn - TD; vr = float(vl[0])
        tt = np.arange(t_land, t_off + 1e-9, 1 / 240.0)
        b.path(tt, np.array([pl + [vr * (t - t_land), 0, 0] for t in tt]))
        p_off = pl + [vr * (t_off - t_land), 0, 0]
        b.fly(t_off, tn, p_off, [vr, 0, 0])
        p0 = p_off + np.array([vr * TD, 0, -0.5 * G * TD * TD]); v0 = np.array([vr, 0, -G * TD])
        RAILS.append(dict(p0=pl.tolist(), p1=p_off.tolist(), t_land=float(t_land), t_off=float(t_off), v=vr))
    END['rails'] = RAILS
    print('mel done', len(b.hits), 'hits, end at', np.round(pE, 2), 'rails', len(RAILS))
    pickle.dump((END, {r: BALLS[r].seg for r in BALLS}, {r: BALLS[r].hits for r in BALLS}), open(CF, 'wb'))
BALLS['bas'].rest(FINAL, FINAL + 30, END['bas']['p'])
# the bellows: the iron ball's drum at bar 14 beat 1
bel = [h for h in BALLS['bas'].hits if abs(h['t'] - E['bellows']) < 1e-6][0]
bel['kind'] = 'bellows'
EV['rails'] = END['rails']
EV['bellows'] = dict(t=bel['t'], p=bel['p'], n=bel['n'])
SFX.append(['gust', E['gust']]); SFX.append(['bellows', bel['t']])

# ============================================================ finale: chimney elevator, roof chute, the hero's lantern
b = BALLS['mel']; pc = np.array(END['mel']['p']); t_c = END['mel']['t']
RISE = 4.3
# inclined chain elevator: its top is far enough along that the roof chute ends beyond the iron ball's last drum (end wall)
DXL = max(2.0, END['bas']['p'][0] + 0.9 - pc[0] - 1.6)
t_go, t_top = t_c + 0.35, t_c + 3.25
ptop = pc + [DXL, 0, RISE]; LL = float(np.hypot(DXL, RISE)); ud = (ptop - pc) / LL
T_ = t_top - t_go; ta = T_ / 4; vmax = LL / (T_ - ta); acc = vmax / ta
def s_of(t):
    u = t - t_go
    if u <= 0: return 0.0
    if u < ta: return 0.5 * acc * u * u
    if u < T_ - ta: return 0.5 * acc * ta * ta + vmax * (u - ta)
    if u < T_: w = T_ - u; return LL - 0.5 * acc * w * w
    return LL
b.rest(t_c, t_go, pc)
tt = np.arange(t_go, t_top + 1e-9, 1 / 240.0)
b.path(tt, np.array([pc + ud * s_of(t) for t in tt]))
# the bucket tips: the ember rolls out onto the roof chute (slope TH, rolling sphere a = 5/7 G sin TH)
t_tip = t_top + 0.25
TH = math.radians(14.0); dirv = np.array([math.cos(TH), 0, -math.sin(TH)]); a_r = 5 / 7 * G * math.sin(TH)
p_r0 = ptop + np.array([0.32, 0, -0.08])
b.rest(t_top, t_tip, ptop)
# the tip: a short quarter-turn of the bucket hands the ball to the chute start (slow, 0.2 s)
tt = np.arange(t_tip, t_tip + 0.2 + 1e-9, 1 / 240.0)
b.path(tt, np.array([ptop + (p_r0 - ptop) * (0.5 - 0.5 * math.cos(math.pi * (t - t_tip) / 0.2)) for t in tt]))
t_r0 = t_tip + 0.2; T_FLY = 0.8; t_r1 = FINAL - T_FLY
tt = np.arange(t_r0, t_r1 + 1e-9, 1 / 240.0)
b.path(tt, np.array([p_r0 + dirv * 0.5 * a_r * (t - t_r0) ** 2 for t in tt]))
Lr = 0.5 * a_r * (t_r1 - t_r0) ** 2; v_e = dirv * a_r * (t_r1 - t_r0); p_e = p_r0 + dirv * Lr
b.fly(t_r1, FINAL, p_e, v_e)
p_l = p_e + v_e * T_FLY + np.array([0, 0, -0.5 * G * T_FLY ** 2]); v_l = v_e + np.array([0, 0, -G * T_FLY])
evf = [e for e in evs('mel') if e['fin']][0]
b.hit(FINAL, p_l, v_l, [0, 0, 0], 'lantern', evf['pitch'], ev=evf['i'])
b.rest(FINAL, FINAL + 30, p_l)
EV['lift'] = dict(p0=pc.tolist(), top=ptop.tolist(), t_c=t_c, t_go=t_go, t_top=t_top, t_tip=t_tip, vmax=vmax, acc=acc, ta=ta,
                  rise=RISE, dx=DXL, L=LL)
EV['chute'] = dict(p0=p_r0.tolist(), p1=p_e.tolist(), t0=t_r0, t1=t_r1, th=TH, v_end=float(np.linalg.norm(v_e)))
EV['lantern'] = dict(p=p_l.tolist(), t=FINAL)
SFX += [['lift_go', t_go], ['tip', t_tip], ['roll', t_r0], ['final', FINAL]]
print('lift', np.round(pc, 2), '-> top', np.round(ptop, 2), 'chute', round(Lr, 2), 'm, v_end', round(float(np.linalg.norm(v_e)), 2),
      'lantern', np.round(p_l, 2))

# ============================================================ the rain: one drop per pizzicato note, straight down from the sky
RAINDROPS = []
CHIMES = {}                     # chime id -> dict(top, n, pitch, role, times): one glass chime per drop on the moving rows;
                                # from bar 16 (T_SLOW) the rain plays the fixed chime tree next to the hero snowman (one chime per pitch)
for role, R_ in RAIN.items():
    th = math.radians(R_['tilt']); n = np.array([0.0, math.sin(th), math.cos(th)])
    tree = sorted(set(e['pitch'] for e in evs(role) if e['t'] >= T_SLOW - 1e-6))
    x_tree0 = XC(T_SLOW) + R_['dx'] + 0.35
    for e in evs(role):
        t = e['t']; fin = e['fin']
        if t < T_SLOW - 1e-6:
            cid = f"{role}_{e['i']}"; x = XC(t) + R_['dx']; zt = ZB(t) + R_['dz']
        else:
            k = tree.index(e['pitch']); cid = f"{role}_tree_{e['pitch']}"; x = x_tree0 + 0.25 * k; zt = ZB(T_SLOW) + R_['dz']
        top = np.array([x, R_['y'], zt])
        p_hit = top + n * R_['r']
        tf = math.sqrt(2 * RAIN_H / G)
        vin = np.array([0, 0, -G * tf])
        p_rel = p_hit + [0, 0, RAIN_H]
        vn = vin @ n; vout = vin - (1 + RAIN_E) * vn * n
        CHIMES.setdefault(cid, dict(top=top.tolist(), n=n.tolist(), pitch=e['pitch'], role=role, times=[], tree=t >= T_SLOW - 1e-6))
        CHIMES[cid]['times'].append(t)
        RAINDROPS.append(dict(role=role, ev=e['i'], t=t, pitch=e['pitch'], t_rel=t - tf, p_rel=p_rel.tolist(), p=p_hit.tolist(),
                              top=top.tolist(), n=n.tolist(), vin=vin.tolist(), vout=vout.tolist(), fin=fin, chime=cid,
                              ground=ZB(min(t, T_SLOW)) - 0.05))
print('chimes', len(CHIMES), 'tree', sum(1 for c in CHIMES.values() if c['tree']))


# ============================================================ checks
def check():
    info = {}
    hs = [(n, h) for n, b in BALLS.items() for h in b.hits if np.linalg.norm(h['vout']) > 0]
    info['e_range'] = (round(min(h['e'] for n, h in hs), 3), round(max(h['e'] for n, h in hs), 3))
    info['speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / max(1e-6, np.linalg.norm(h['vin'])) for n, h in hs), 3)
    info['tilt_max_deg'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for n, h in hs), 1)
    struck = set(h['ev'] for b in BALLS.values() for h in b.hits if h['ev'] is not None) | set(d['ev'] for d in RAINDROPS)
    info['events_uncovered'] = sorted(set(range(len(LEAD))) - struck)
    for n, b in BALLS.items():
        zz = [h['p'][2] for h in b.hits]; xx = [h['p'][0] for h in b.hits]
        info[n] = dict(hits=len(b.hits), z=(round(min(zz), 2), round(max(zz), 2)), x=(round(min(xx), 1), round(max(xx), 1)))
    info['lift_vmax'] = round(vmax, 2)
    return info


if __name__ == '__main__':
    info = check()
    for k, v in info.items(): print(k, v)
    out = dict(G=G, lanes=LANES, rain=RAIN, rain_h=RAIN_H, key_lead=KEY_LEAD, key_stay=KEY_STAY, v=V, t_slow=T_SLOW, v_slow=V_SLOW,
               gz=GZ, gzb=GZB, balls={n: dict(r=b.r, seg=b.seg, hits=b.hits) for n, b in BALLS.items()}, drops=RAINDROPS, chimes=CHIMES, ev=EV, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
