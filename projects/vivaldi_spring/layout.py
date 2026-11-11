"""Vivaldi 'Spring' - 「春をはこぶビー玉」 (the marbles that carry spring). Physics + story planner.
Run after arrange.py; writes layout.json (read by scene.py, render_audio.py, verify.py).

Cast (side view of a snowy terraced hillside, three terraces = three lanes, birds on the branches above):
  mel   'spring' ball (green, solo violin)     strikes ice keys on the front terrace; snow melts behind it
  har   'blossom' ball (pink, violin 2)        strikes ice keys on the middle terrace
  bas   'earth' ball (gold, cello)             strikes tree-stump drums on the back terrace
  birdA / birdB  two small balls (yellow / blue) asleep in nests on the bare tree: the birdsong (solo violin, then violin 1
                 echoing it two bars later)
Story
  0-1.7 s   winter. the three balls hang frozen in icicles under a snow cloud; the icicles crack, they drop onto their first
            keys (the pickup) and wherever a key is struck the snow melts and a flower opens
  A         tutti ritornello. bar 13 beat 2: the earth ball's drum is the end of a lever that tips bird A's nest
  cadence   bar 13 beat 3: the three balls land in the cups (bells) of three little ropeway gondolas; bird A drops out of its
            nest onto its first branch-bell at the same instant: the birdsong starts
  B         the birds sing; bird A's bell in bar 15 tips bird B's nest - bird B wakes and imitates bird A two bars later.
            every bud bell a bird strikes opens into a blossom; meanwhile the ropeway carries the three balls along
  A'        bar 27: the gondola trapdoors open, the three drop onto their keys for the last ritornello
  final     bar 31 beat 1: they land in the three golden bells of the giant cherry tree -> full bloom, petal storm, sunrise
Physics: one constant gravity G; between contacts pure parabolas; every bounce is an impulse along the key's surface normal
(tangential velocity kept) with restitution EMIN <= e <= EMAX, so a bounce never adds speed (fixed note times force the keys
to tilt). The only energy sources are visible: the drops (icicles, nests, trapdoors) and the motor-driven ropeway."""
import json, os, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = 7.0
EMAX, EMIN, TILT = 0.97, 0.22, 34.0
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
T0, Q, FINAL = E['t0'], E['q'], E['final']
TB0, TB1 = E['b_start'], E['b_end']                 # cadence / birdsong start, ritornello pickup
V = 1.35                                            # mean course speed (m/s) shared by every lane
X0 = 0.0
LANES = {   # y, start z, radius, key half sizes (hw along x, hd along y, half thickness)
    'mel': dict(y=0.0, z=0.0, r=0.19, ext=(0.17, 0.3, 0.05)),
    'har': dict(y=1.7, z=2.0, r=0.17, ext=(0.16, 0.28, 0.05)),
    'bas': dict(y=3.4, z=4.0, r=0.23, ext=(0.22, 0.22, 0.12)),
    'birdA': dict(y=4.7, z=5.6, r=0.16, ext=(0.12, 0.14, 0.05)),
    'birdB': dict(y=5.9, z=6.9, r=0.16, ext=(0.12, 0.14, 0.05)),
}
SFX = []; EV = {}
KEY_LEAD, KEY_STAY = 0.7, 0.6       # keys exist from t-0.7 s (popping up) to t+0.6 s (sinking away)


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
                vxo = np.linspace(vxlim[0], vxlim[1], NX); dzs = np.linspace(dzr[0], dzr[1], NZ) * max(0.4, d / 0.66)
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
def xt_of(t): return X0 + V * (t - T0)
GZ = 0.03                                            # gentle descent of the course per second (terraces step down)
PLANS = {}

# ============================================================ opening: icicles under the snow cloud release the three balls
T_REL = T0 - 0.5
EV['icicles'] = {}
for role in ('mel', 'har', 'bas'):
    L = LANES[role]; b = BALLS[role]; p1 = np.array([X0, L['y'], L['z']])
    h = 0.5 * G * 0.5 ** 2; pc = p1 + [0, 0, h]
    b.rest(0.0, T_REL, pc); b.fly(T_REL, T0, pc, [0, 0, 0])
    EV['icicles'][role] = pc.tolist()
SFX.append(['crack', T_REL])

# ============================================================ A: tutti ritornello up to the cadence (gondola cups)
import pickle, hashlib
CK = hashlib.md5((open(f'{HERE}/lead.json').read() + repr((G, EMAX, EMIN, TILT, V, GZ, LANES, KEY_LEAD, KEY_STAY))).encode()).hexdigest()
CF = f'/tmp/vivaldi_spring_A_{CK}.pkl'
CAD = {}
if os.path.exists(CF):
    CAD, segs_, hits_ = pickle.load(open(CF, 'rb'))
    for r in segs_: BALLS[r].seg = segs_[r]; BALLS[r].hits = hits_[r]
for role in (('mel', 'har', 'bas') if not CAD else ()):
    L = LANES[role]; b = BALLS[role]
    ev_ = [e for e in evs(role) if e['t'] <= TB0 + 1e-6]
    ts = np.array([e['t'] for e in ev_])
    z0 = L['z']
    hits, pE, vE = plan_retry(ts, (X0, L['y'], z0), [np.array([0, 0, -G * 0.5])], xt_of, lambda t, z0=z0: z0 - GZ * (t - T0),
                              lambda t, y=L['y']: y, R=L['r'], ext=L['ext'])
    write_run(b, hits, ev_, ts[-1], lambda e: 'drum' if role == 'bas' else 'key')
    # the cadence (last event, bar 13 beat 3): the ball lands in the cup bell of its gondola and stays
    e = ev_[-1]
    b.hit(e['t'], pE, vE, [0, 0, 0], 'cup', e['pitch'], ev=e['i'])
    CAD[role] = dict(p=pE.tolist(), vin=vE.tolist(), t=e['t'])
    print(role, 'A done', len(hits) + 1, 'hits, cadence at', np.round(pE, 2))
if not os.path.exists(CF):
    pickle.dump((CAD, {r: BALLS[r].seg for r in CAD}, {r: BALLS[r].hits for r in CAD}), open(CF, 'wb'))
# the lever: the earth ball's drum at bar 13 beat 2 (t = TB0 - Q) tips bird A's nest
lev = [h for h in BALLS['bas'].hits if abs(h['t'] - (TB0 - Q)) < 1e-6][0]
lev['kind'] = 'lever'

# ============================================================ B: the birds
def bird_run(name, t_drop, x_nest, z_nest):
    """bird drops out of its nest onto its first bud bell; a long rest (> 2 s) is spent riding a zip-line basket"""
    L = LANES[name]; b = BALLS[name]
    ev_all = evs(name)
    segs = [[ev_all[0]]]
    for a, c in zip(ev_all, ev_all[1:]):
        if c['t'] - a['t'] > 2.0: segs.append([c])
        else: segs[-1].append(c)
    tf = ev_all[0]['t'] - t_drop; h = 0.5 * G * tf * tf
    pn = np.array([x_nest, L['y'], z_nest + h])
    b.rest(0.0, t_drop, pn); b.fly(t_drop, ev_all[0]['t'], pn, [0, 0, 0])
    p0 = np.array([x_nest, L['y'], z_nest]); v0 = np.array([0, 0, -G * tf]); x_ref = x_nest; t_ref = ev_all[0]['t']; z_ref = z_nest
    leaves = []
    for si, ev_ in enumerate(segs):
        ts = np.array([e['t'] for e in ev_])
        hits, pE, vE = plan_retry(ts, p0, [v0], lambda t, x_ref=x_ref, t_ref=t_ref: x_ref + V * (t - t_ref),
                                  lambda t, z_ref=z_ref, t_ref=t_ref: z_ref - GZ * (t - t_ref), lambda t, y=L['y']: y, R=L['r'], ext=L['ext'])
        write_run(b, hits, ev_, ts[-1], lambda e: 'bud')
        e = ev_[-1]
        if si + 1 < len(segs):
            # hop into a little basket on a zip line (soft catch). The bird's push sets the light basket going (momentum
            # shared with the basket, mass 0.15 x bird); on the gently sloped line rolling resistance balances gravity, so
            # it glides at constant speed, hits the stopper at the end and the bird flies on onto the next phrase's bell
            vo, dt = find_hop(pE, vE, (0.6, 0.5, 0.4, 0.32, 0.26, 0.2), -0.5, 0.6, 1.6, vx_lo=0.9, R=L['r'], ext=L['ext'])
            b.hit(ts[-1], pE, vE, vo, 'bud', e['pitch'], ev=e['i'])
            b.fly(ts[-1], ts[-1] + dt, pE, vo)
            pl = pE + vo * dt + np.array([0, 0, -0.5 * G * dt * dt]); vl = vo + np.array([0, 0, -G * dt])
            t_land = ts[-1] + dt
            b.hits.append(dict(t=float(t_land), p=pl.tolist(), vin=vl.tolist(), vout=[0, 0, 0], n=[0, 0, 1], e=0.0, kind='zip', pitch=None, ev=None))
            tn = segs[si + 1][0]['t']; tfl = 0.4; t_stop = tn - tfl
            TH = math.radians(7.0); dirv = np.array([math.cos(TH), 0, -math.sin(TH)])
            v_s = vl[0] / 1.15 / math.cos(TH)
            tt = np.arange(t_land, t_stop + 1e-9, 1 / 240.0)
            b.path(tt, np.array([pl + dirv * v_s * (t - t_land) for t in tt]))
            p_stop = pl + dirv * v_s * (t_stop - t_land); v_th = dirv * v_s
            b.fly(t_stop, tn, p_stop, v_th)
            p0 = p_stop + v_th * tfl + np.array([0, 0, -0.5 * G * tfl * tfl]); v0 = v_th + np.array([0, 0, -G * tfl])
            leaves.append(dict(p=pl.tolist(), t_land=float(t_land), t_stop=float(t_stop), dir=dirv.tolist(), v=float(v_s), p_stop=p_stop.tolist()))
            SFX.append(['zip', float(t_land)]); SFX.append(['stop', float(t_stop)])
            x_ref, t_ref, z_ref = p0[0], tn, p0[2]
        else:
            # last note -> one silent hop into its home nest, where it stays
            vo, dt = find_hop(pE, vE, (0.7, 0.6, 0.5, 0.4, 0.32, 0.26), -0.8, 0.3, 1.6, R=L['r'], ext=L['ext'])
            b.hit(ts[-1], pE, vE, vo, 'bud', e['pitch'], ev=e['i'])
            b.fly(ts[-1], ts[-1] + dt, pE, vo)
            pH = pE + vo * dt + np.array([0, 0, -0.5 * G * dt * dt]); vH = vo + np.array([0, 0, -G * dt])
            b.hits.append(dict(t=float(ts[-1] + dt), p=pH.tolist(), vin=vH.tolist(), vout=[0, 0, 0], n=[0, 0, 1], e=0.0, kind='home', pitch=None, ev=None))
            b.rest(ts[-1] + dt, FINAL + 30, pH)
    print(name, 'done', len(b.hits), 'hits, home', np.round(pH, 2))
    return dict(nest=pn.tolist(), t_drop=float(t_drop), first=[x_nest, L['y'], z_nest], home=pH.tolist(), t_home=float(ts[-1] + dt), leaves=leaves)

xA = xt_of(TB0) + 0.4
EV['nestA'] = bird_run('birdA', TB0 - Q, xA, CAD['bas']['p'][2] + 1.5)
# bird A's bell at bar 15 beat 2 is the lever under bird B's nest
evB0 = evs('birdB')[0]
levB = [h for h in BALLS['birdA'].hits if abs(h['t'] - (evB0['t'] - Q)) < 1e-6][0]
levB['kind'] = 'lever'
xB = levB['p'][0] + 0.9
EV['nestB'] = bird_run('birdB', evB0['t'] - Q, xB, levB['p'][2] + 1.4)
EV['leverA'] = dict(t=lev['t'], p=lev['p'], nest=EV['nestA']['nest'])
EV['leverB'] = dict(t=levB['t'], p=levB['p'], nest=EV['nestB']['nest'])
SFX.append(['nest', EV['nestA']['t_drop']]); SFX.append(['nest', EV['nestB']['t_drop']])

# ============================================================ ropeway: cups carried from the cadence to above the ritornello keys
T_DROP = 0.38                                       # trapdoor -> first ritornello key
RISE = 1.1                                          # the gondolas ride this much above their start, clear of everything
DESC = 3.4                                          # the hillside (and the ropeway) descends with the birds
ROPE = {}
for role in ('mel', 'har', 'bas'):
    L = LANES[role]; b = BALLS[role]
    pc = np.array(CAD[role]['p'])
    x_r = xt_of(TB1) - 0.15; z_r = pc[2] - 0.25 - DESC
    pd = np.array([x_r, L['y'], z_r + 0.5 * G * T_DROP ** 2])
    # cable path (ball-centre path): up and forward, along, down onto the drop point
    pts = [pc, pc + [1.6, 0, RISE], np.array([pd[0] - 1.6, L['y'], pd[2] + RISE]), pd]
    seglen = [np.linalg.norm(pts[k + 1] - pts[k]) for k in range(3)]; Ltot = sum(seglen)
    t_go = TB0 + 0.45; t_stop = TB1 - T_DROP - 0.35
    T_ = t_stop - t_go
    A_ = 4 * Ltot / T_ ** 2 * 1.0                   # trapezoid: accelerate for T/4, cruise, decelerate for T/4
    ta = T_ / 4; vmax = Ltot / (T_ - ta)
    acc = vmax / ta
    def s_of(t):
        u = t - t_go
        if u <= 0: return 0.0
        if u < ta: return 0.5 * acc * u * u
        if u < T_ - ta: return 0.5 * acc * ta * ta + vmax * (u - ta)
        if u < T_: w = T_ - u; return Ltot - 0.5 * acc * w * w
        return Ltot
    def p_of(s):
        for k in range(3):
            if s <= seglen[k] + 1e-9 or k == 2:
                return pts[k] + (pts[k + 1] - pts[k]) * min(1.0, s / seglen[k])
            s -= seglen[k]
    b.rest(CAD[role]['t'], t_go, pc)
    tt = np.arange(t_go, t_stop + 1e-9, 1 / 240.0)
    b.path(tt, np.array([p_of(s_of(t)) for t in tt]))
    t_open = TB1 - T_DROP
    b.rest(t_stop, t_open, pd)
    b.fly(t_open, TB1, pd, [0, 0, 0])
    ROPE[role] = dict(pts=[p.tolist() for p in pts], t_go=t_go, t_stop=t_stop, t_open=t_open, vmax=vmax, acc=acc, L=Ltot, ta=ta)
    SFX.append(['door', t_open])
SFX.append(['rope_go', TB0 + 0.45])
EV['rope'] = ROPE

# ============================================================ A': last ritornello and the three golden bells
FINB = {}
for role in ('mel', 'har', 'bas'):
    L = LANES[role]; b = BALLS[role]
    ev_ = [e for e in evs(role) if e['t'] >= TB1 - 1e-6]
    ts = np.array([e['t'] for e in ev_])
    p1 = np.array(ROPE[role]['pts'][-1]) - [0, 0, 0.5 * G * T_DROP ** 2]
    z1 = p1[2]; x1 = p1[0]
    hits, pE, vE = plan_retry(ts, p1, [np.array([0, 0, -G * T_DROP])], lambda t, x1=x1: x1 + V * (t - TB1),
                              lambda t, z1=z1: z1 - GZ * (t - TB1), lambda t, y=L['y']: y, R=L['r'], ext=L['ext'])
    write_run(b, hits, ev_, ts[-1], lambda e: 'drum' if role == 'bas' else 'key')
    e = ev_[-1]
    b.hit(e['t'], pE, vE, [0, 0, 0], 'finbell', e['pitch'], ev=e['i'])
    b.rest(e['t'], FINAL + 30, pE)
    FINB[role] = dict(p=pE.tolist(), pitch=e['pitch'])
    print(role, "A' done, final bell at", np.round(pE, 2))
EV['final_bells'] = FINB
SFX.append(['final', FINAL])


# ============================================================ checks
def check():
    info = {}
    hs = [(n, h) for n, b in BALLS.items() for h in b.hits if np.linalg.norm(h['vout']) > 0]
    info['e_range'] = (round(min(h['e'] for n, h in hs), 3), round(max(h['e'] for n, h in hs), 3))
    info['speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / max(1e-6, np.linalg.norm(h['vin'])) for n, h in hs), 3)
    info['tilt_max_deg'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for n, h in hs), 1)
    struck = set(h['ev'] for b in BALLS.values() for h in b.hits if h['ev'] is not None)
    info['events_uncovered'] = sorted(set(range(len(LEAD))) - struck)
    for n, b in BALLS.items():
        zz = [h['p'][2] for h in b.hits]; xx = [h['p'][0] for h in b.hits]
        info[n] = dict(hits=len(b.hits), z=(round(min(zz), 2), round(max(zz), 2)), x=(round(min(xx), 1), round(max(xx), 1)))
    info['rope_vmax'] = {r: round(v['vmax'], 2) for r, v in ROPE.items()}
    return info


if __name__ == '__main__':
    info = check()
    for k, v in info.items(): print(k, v)
    out = dict(G=G, lanes=LANES, key_lead=KEY_LEAD, key_stay=KEY_STAY, t_drop=T_DROP,
               balls={n: dict(r=b.r, seg=b.seg, hits=b.hits) for n, b in BALLS.items()}, ev=EV, sfx=SFX)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
