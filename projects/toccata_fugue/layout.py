"""Toccata and Fugue in D minor - 'The Midnight Cathedral' (真夜中の大聖堂). Physics + story planner (run after arrange.py).
Writes layout.json (read by scene.py, render_audio.py, verify.py).

Story (every struck key = one organ note; a key sounds while a marble presses it, a struck key for the note's length):
  calls    three silver marbles roll out of three galleries above the locked cathedral door (high, middle, low). Each one
           rattles down a narrow shaft (mordent A-G-A on the shaft walls), lands in the A cradle (fermata = it rests there),
           is flicked by a little hammer down a keyboard ramp (the run G-F-E-D-C#-D, one key per note) and flies onto
           the lever of one of the three padlocks on the door.  LOCK 1/3, 2/3, 3/3 open.
  giant    the third lever also tips the shelf of the bronze GIANT: it drops onto the pedal key (low D) whose chain
           pulls the great doors open, then it rolls off into its floor socket (doors latched open).
  chord    ten ivory marbles fall from the dark vault onto the console in front of the door: the rolled dim7 chord
           (they stay on their keys = the chord is held), then D major (three more) and the inner voice E-F#.
  trio     the doors reveal three gold marbles inside: Prestissimo. They roll out and bounce down three stair lanes
           (one key per note, each takes every third note of the triplets), ride the lift platform up for the octave-higher
           repeat, wait at the landing gate during the fermata, then race down the long cascade (three descending scales).
  bar 10   the trio drops together onto the dim7 keys; GIANT II falls from its ledge onto the second pedal key; one gold
           marble leaks from the reservoir onto the Bb key (the 'first drop').
  bar 11   the first drop is flicked down a keyboard ramp (A..B), hops C#-A-C#-E-G, zig-zags down the trill shaft (G-F x3)
           and hops E -> D.  GIANT II rolls onto the floodgate's counterweight: the reservoir opens.
  bar 12   the flood: 546 gold marbles pour over the lip onto the six D / F keys of the final chord. Counter: 565 = BWV 565.
Physics: one constant gravity G; free flight = parabolas; bounces keep the tangential velocity and reduce the normal one
(e <= EMAX, never faster out than in); rolling on ramps = constant acceleration 5/7 g sin(slope); speed-ups only from
visible causes (falls, the flick hammers)."""
import json, os, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = 7.0; R = 0.15; RG = 0.32; RR = 0.1
EMAX = 0.96; EMIN = 0.05
LEAD = json.load(open(f'{HERE}/lead.json')); EVJ = json.load(open(f'{HERE}/events.json')); S = EVJ['secs']
rng = np.random.default_rng(565)
ROWS = {}
for k, r in enumerate(LEAD): ROWS.setdefault(r[3], []).append(dict(k=k, t=r[0], d=r[1], ps=r[2], bar=r[5], pos=r[6], note=r[7]))
GZ = np.array([0, 0, -G])


def flyp(p, v, u): return np.asarray(p, float) + np.asarray(v, float) * u + 0.5 * GZ * u * u
def flyv(v, u): return np.asarray(v, float) + GZ * u


class Ball:
    def __init__(s, name, kind, r=R):
        s.name = name; s.kind = kind; s.r = r; s.seg = []; s.hits = []
    def rest(s, t0, t1, p): s.seg.append(dict(k='rest', t0=float(t0), t1=float(t1), p=list(map(float, p))))
    def fly(s, t0, t1, p0, v0): s.seg.append(dict(k='fly', t0=float(t0), t1=float(t1), p=list(map(float, p0)), v=list(map(float, v0))))
    def line(s, t0, t1, p0, d, v0, a):
        s.seg.append(dict(k='line', t0=float(t0), t1=float(t1), p=list(map(float, p0)), d=list(map(float, d)), v0=float(v0), a=float(a)))
    def carry(s, t0, t1, p0, p1): s.seg.append(dict(k='carry', t0=float(t0), t1=float(t1), p=list(map(float, p0)), p1=list(map(float, p1))))
    def at(s, t):
        for g in s.seg:
            if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9: return seg_at(g, t)
        if t < s.seg[0]['t0']: return seg_at(s.seg[0], s.seg[0]['t0'])
        return seg_at(s.seg[-1], s.seg[-1]['t1'])
    def end(s): return seg_at(s.seg[-1], s.seg[-1]['t1'])


def seg_at(g, t):
    p = np.array(g['p']); u = t - g['t0']
    if g['k'] == 'rest': return p
    if g['k'] == 'fly': return flyp(p, g['v'], u)
    if g['k'] == 'line': return p + np.array(g['d']) * (g['v0'] * u + 0.5 * g['a'] * u * u)
    if g['k'] == 'ease':
        f = 0 if g['t1'] <= g['t0'] else min(1, max(0, (t - g['t0']) / (g['t1'] - g['t0'])))
        f = f * f * (3 - 2 * f); return p + (np.array(g['p1']) - p) * f
    if g['k'] == 'carry':
        f = 0 if g['t1'] <= g['t0'] else (t - g['t0']) / (g['t1'] - g['t0'])
        return p + (np.array(g['p1']) - p) * f
    raise ValueError(g['k'])


BALLS = {}
KEYS = []          # every struck surface that sounds
PARTS = {}         # mechanism data for scene.py
SFX = []
def ball(name, kind, r=R): BALLS[name] = Ball(name, kind, r); return BALLS[name]


def key(kind, ps, contact, n, t_on, t_off, w, d=0.5, th=0.07, ball_=None, region='', row=None, tdir=None):
    """contact = point of the key surface touched by the ball (ball centre - r n); w = length along the travel direction"""
    n = np.asarray(n, float); n = n / np.linalg.norm(n)
    KEYS.append(dict(kind=kind, ps=list(ps), c=list(map(float, contact)), n=list(map(float, n)), t=[float(x) for x in np.atleast_1d(t_on)],
                     toff=[float(x) for x in np.atleast_1d(t_off)], w=float(w), d=float(d), th=float(th), ball=ball_, region=region,
                     row=row, tdir=None if tdir is None else list(map(float, tdir))))
    return KEYS[-1]


def bounce_e(vin, vout):
    d = np.asarray(vout) - np.asarray(vin); n = d / max(1e-9, np.linalg.norm(d)); a = np.dot(vin, n)
    return (-(np.dot(vout, n)) / a if a < -1e-9 else 9.0), n


def add_hit(b, t, p, vin, vout, n=None, kind='bar', row=None):
    e, nn = bounce_e(vin, vout)
    if n is None: n = nn
    b.hits.append(dict(t=float(t), p=list(map(float, p)), n=list(map(float, n)), e=float(e), kind=kind, row=row,
                       vin=list(map(float, vin)), vout=list(map(float, vout))))


def vout_for(d, dt):
    """velocity that covers displacement d (3-vector) in dt under gravity"""
    d = np.asarray(d, float); return np.array([d[0] / dt, d[1] / dt, d[2] / dt + 0.5 * G * dt])


def hops(b, p, vin, rows_, vx_of, dzps, y_of=None, emax=EMAX, emin=EMIN, tilt_max=38.0, dvx=(-0.15, 0.1), kind='bar', region='',
         w=0.26, d=0.42):
    """the ball sits at its contact centre p at rows_[0]['t'] with incoming velocity vin; it hops through rows_[1:].
    vx_of(k): target horizontal speed of hop k; dzps: target height change per second of flight; y_of(k): target y at row k.
    The landing height is lowered (never raised) until the bounce is physical (emin <= e <= emax, key tilt <= tilt_max) and
    the ball arrives fast enough to make the next hop. Creates one key per row. Returns (p, vin) on arrival at the last row."""
    for k in range(len(rows_) - 1):
        r0, r1 = rows_[k], rows_[k + 1]; dt = r1['t'] - r0['t']
        dzt = dzps(k, p, dt) if callable(dzps) else dzps * dt
        if k + 1 < len(rows_) - 1:
            dt2 = rows_[k + 2]['t'] - r1['t']
            pn = np.array([p[0] + vx_of(k) * dt, p[1], p[2] + dzt])
            need = ((dzps(k + 1, pn, dt2) / dt2 if callable(dzps) else dzps) + 0.5 * G * dt2) / (emax * 0.99)
        else: need = 0.0
        cx = vin[0]; tx = vx_of(k)
        vx0 = cx + float(np.clip(tx - cx, dvx[0], dvx[1]))
        vxs = sorted(set(np.round(np.concatenate([[vx0], vx0 + np.linspace(-0.3, 0.1, 21)]), 4)), key=lambda v: abs(v - vx0))
        dzs = sorted(np.concatenate([dzt - np.linspace(0, 0.5 * G * dt * dt + 0.8, 300), dzt + np.linspace(0.003, 0.2, 30)]),
                     key=lambda q: (0 if q <= dzt else 1, abs(q - dzt)))
        dy = (y_of(k + 1) - p[1]) if y_of else 0.0
        best = None; cands = []
        for vx in vxs[:6]:
            for dz in dzs:
                vo = vout_for((vx * dt, dy, dz), dt); e, n = bounce_e(vin, vo)
                if not (emin <= e <= emax) or vo[2] - G * dt > -0.12: continue
                tilt = math.degrees(math.acos(max(-1, min(1, n[2]))))
                if tilt > tilt_max: continue
                cands.append((dz, vo, e, n))
                if -(vo[2] - G * dt) >= need: best = (dz, vo, e, n); break
            if best is not None: break
        if best is None and cands: best = cands[0]; print('WARN need', b.name, k, round(need, 2))
        if best is None:
            allc = []
            for vx in vxs:
                for dz in dzs[::5]:
                    vo = vout_for((vx * dt, dy, dz), dt); e, n = bounce_e(vin, vo)
                    allc.append((abs(min(0, e - emin)) + max(0, e - emax), dz, vo, e, n))
            _, dz, vo, e, n = min(allc, key=lambda c: c[0]); best = (dz, vo, e, n); print('WARN e', b.name, k, round(e, 3))
        dz, vo, e, n = best
        add_hit(b, r0['t'], p, vin, vo, n, kind, r0['k'])
        tdir = np.array([vo[0], vo[1], 0.0]); tdir = tdir / max(1e-9, np.linalg.norm(tdir))
        key(kind, r0['ps'], p - b.r * n, n, r0['t'], r0['t'] + r0['d'], w, d, ball_=b.name, region=region, row=r0['k'], tdir=tdir)
        b.fly(r0['t'], r1['t'], p, vo)
        p = flyp(p, vo, dt); vin = flyv(vo, dt)
    return p, vin


# =====================================================================================================================
# 1. the three calls
# =====================================================================================================================
CALLS = {}
MORD_E = 0.9


def call(c, chute_end, s_c, s_r, th_r=26.0, v_kick=1.55, t_fl=0.36, lever_dz=0.0):
    """chute_end: ball centre where it leaves the gallery chute; s_c: direction of the chute (+1 right / -1 left);
    s_r: direction of the keyboard ramp. Returns the landing point on the lock lever."""
    rows_ = ROWS[f'call{c}']; b = ball(f'S{c}', 'silver')
    tA, tG, tA2 = rows_[0]['t'], rows_[1]['t'], rows_[2]['t']
    # gallery chute: 30 deg incline, rolls from rest (released by a gate), leaves at chute_end
    th_c = math.radians(30); a_c = 5 / 7 * G * math.sin(th_c); dch = np.array([s_c * math.cos(th_c), 0, -math.sin(th_c)])
    v_ex = 2.2; T_roll = v_ex / a_c; L = 0.5 * a_c * T_roll ** 2
    tf0 = 0.16
    t_ex = tA - tf0; t_go = t_ex - T_roll
    p_start = np.array(chute_end) - dch * L
    b.rest(-1.0, t_go, p_start); b.line(t_go, t_ex, p_start, dch, 0.0, a_c)
    v = dch * v_ex
    # shaft: wall A, wall G, cradle A
    b.fly(t_ex, tA, chute_end, v); p1 = flyp(chute_end, v, tf0); v1 = flyv(v, tf0)
    vo1 = np.array([-MORD_E * v1[0], 0, v1[2]]); n1 = np.array([-s_c, 0, 0.0])
    add_hit(b, tA, p1, v1, vo1, n1, 'wall', rows_[0]['k'])
    key('wall', rows_[0]['ps'], p1 - R * n1, n1, tA, tA + rows_[0]['d'], 0.34, 0.4, ball_=b.name, region='calls', row=rows_[0]['k'], tdir=(0, 0, -1))
    dt = tG - tA; b.fly(tA, tG, p1, vo1); p2 = flyp(p1, vo1, dt); v2 = flyv(vo1, dt)
    vo2 = np.array([-MORD_E * v2[0], 0, v2[2]]); n2 = -n1
    add_hit(b, tG, p2, v2, vo2, n2, 'wall', rows_[1]['k'])
    key('wall', rows_[1]['ps'], p2 - R * n2, n2, tG, tG + rows_[1]['d'], 0.34, 0.4, ball_=b.name, region='calls', row=rows_[1]['k'], tdir=(0, 0, -1))
    dt = tA2 - tG; b.fly(tG, tA2, p2, vo2); p3 = flyp(p2, vo2, dt); v3 = flyv(vo2, dt)
    th = math.radians(th_r); a = 5 / 7 * G * math.sin(th); dr = np.array([s_r * math.cos(th), 0, -math.sin(th)])
    nr = np.array([s_r * math.sin(th), 0, math.cos(th)])
    # the A cradle is the top of the keyboard ramp (same slope); a stop pin holds the marble under the fermata
    add_hit(b, tA2, p3, v3, np.zeros(3), nr, 'cradle', rows_[2]['k'])
    key('cradle', rows_[2]['ps'], p3 - R * nr - dr * 0.05, nr, tA2, tA2 + rows_[2]['d'], 0.3, 0.42, ball_=b.name,
        region='calls', row=rows_[2]['k'], tdir=dr)
    t_rel = tA2 + rows_[2]['d']
    b.rest(tA2, t_rel, p3)
    run_rows = rows_[3:]
    sp = lambda t: v_kick * (t - t_rel) + 0.5 * a * (t - t_rel) ** 2
    t_end = run_rows[-1]['t'] + run_rows[-1]['d']
    # the ramp starts at the cradle lip: the ball centre line is p3 + dr*s, the surface is r below it (along nr)
    slats = []
    for k, r in enumerate(run_rows):
        s0 = sp(r['t']); s1 = sp(run_rows[k + 1]['t']) if k + 1 < len(run_rows) else sp(t_end)
        cpt = p3 + dr * s0 - R * nr
        add_hit(b, r['t'], p3 + dr * s0, dr * (v_kick + a * (r['t'] - t_rel)), dr * (v_kick + a * (r['t'] - t_rel)), nr, 'slat', r['k'])
        kk = key('slat', r['ps'], p3 + dr * (s0 + s1) / 2 - R * nr, nr, r['t'], r['t'] + r['d'], s1 - s0 - 0.012, 0.42, 0.06,
                 ball_=b.name, region='calls', row=r['k'], tdir=dr)
        kk['s0'] = float(s0); slats.append(kk)
    b.line(t_rel, t_end, p3, dr, v_kick, a)
    pe = p3 + dr * sp(t_end); ve = dr * (v_kick + a * (t_end - t_rel))
    t_lock = t_end + t_fl
    b.fly(t_end, t_lock, pe, ve); pl = flyp(pe, ve, t_fl); vl = flyv(ve, t_fl)
    add_hit(b, t_lock, pl, vl, np.zeros(3), (0, 0, 1), 'lever', None)
    # lever: tips down 0.14 under the marble over 0.3 s (the marble rides it), then stays
    pl2 = pl + np.array([0, 0, -0.14])
    b.carry(t_lock, t_lock + 0.3, pl, pl2); b.rest(t_lock + 0.3, 200.0, pl2)
    SFX.append(['flick', t_rel]); SFX.append(['lever', t_lock]); SFX.append(['unlock', t_lock + 0.25])
    CALLS[c] = dict(t_go=t_go, p_start=p_start.tolist(), chute_end=list(map(float, chute_end)), dch=dch.tolist(), L=L,
                    wall1=(p1 - R * n1).tolist(), wall2=(p2 - R * n2).tolist(), cradle=(p3 - R * nr).tolist(),
                    ball_cradle=p3.tolist(), t_rel=t_rel, ramp_p0=p3.tolist(), ramp_dir=dr.tolist(), ramp_n=nr.tolist(),
                    ramp_len=float(sp(t_end)), t_lock=t_lock, lever=pl.tolist(), lever_dn=pl2.tolist(), s_c=s_c, s_r=s_r)
    return pl


call(1, (0.55, 0, 9.15), -1, +1)
call(2, (-0.55, 0, 7.45), +1, -1)
call(3, (0.55, 0, 5.75), -1, +1)
PARTS['calls'] = CALLS

# =====================================================================================================================
# 2. GIANT I: shelf tips when lock 3 opens, falls onto the pedal key (low D), the key's chain opens the doors,
#    then it rolls off into its floor socket.
# =====================================================================================================================
ped1 = ROWS['ped'][0]; gi = ball('GIANT1', 'giant', RG)
X_G1 = 3.0; Y_G1 = -0.3; Z_PED = 0.22
t_fall = 0.72; t_rel_g = ped1['t'] - t_fall
p_g0 = np.array([X_G1, Y_G1, Z_PED + RG + 0.5 * G * t_fall ** 2])
assert t_rel_g > CALLS[3]['t_lock'] + 0.1, (t_rel_g, CALLS[3]['t_lock'])
gi.rest(-1.0, t_rel_g, p_g0); gi.fly(t_rel_g, ped1['t'], p_g0, (0, 0, 0))
pg = np.array([X_G1, Y_G1, Z_PED + RG]); vg = flyv((0, 0, 0), t_fall)
add_hit(gi, ped1['t'], pg, vg, np.zeros(3), (0, 0, 1), 'pedal', ped1['k'])
kp = key('pedal', ped1['ps'], pg - RG * np.array([0, 0, 1]), (0, 0, 1), ped1['t'], ped1['t'] + ped1['d'], 0.9, 0.9, 0.16, ball_='GIANT1', region='door', row=ped1['k'])
kp['occ'] = [['GIANT1', ped1['t'], ped1['t'] + ped1['d']]]
t_off = ped1['t'] + ped1['d']
# the pedal key sinks 0.1 (rides), at the release it tilts and the giant rolls 0.75 left into the socket (rolling 10 deg)
pg2 = pg.copy(); gi.rest(ped1['t'], t_off, pg2)
th = math.radians(10); a = 5 / 7 * G * math.sin(th); dr = np.array([math.cos(th), 0, -math.sin(th)])
T_r = math.sqrt(2 * 0.75 / a); gi.line(t_off, t_off + T_r, pg2, dr, 0.0, a); psock = pg2 + dr * 0.75
gi.rest(t_off + T_r, 200.0, psock)
PARTS['giant1'] = dict(shelf=p_g0.tolist(), t_rel=t_rel_g, pedal=(pg - RG * np.array([0, 0, 1])).tolist(), t_on=ped1['t'], t_off=t_off,
                       socket=psock.tolist(), t_sock=t_off + T_r)
DOOR = dict(x0=-1.45, x1=1.45, z1=4.9, y=1.4, t_open0=ped1['t'] + 0.25, t_open1=S['dmaj'])
PARTS['door'] = DOOR
SFX.append(['shelf', t_rel_g]); SFX.append(['door', DOOR['t_open0']]); SFX.append(['socket', t_off + T_r])

# =====================================================================================================================
# 3. the console: ten ivory marbles fall from the vault onto the chord keys
# =====================================================================================================================
CONS_P = [49, 50, 52, 55, 57, 58, 61, 62, 64]
CONS_Y = -0.62; CONS_Z = 0.78; CONS_DX = 0.31
CONS_X = {p: -1.16 + CONS_DX * i for i, p in enumerate(CONS_P)}
INNER_X = {52: 1.7, 54: 2.02}; INNER_Z = CONS_Z + 0.32
Z_VAULT = 14.0
cons_keys = {p: dict(x=CONS_X[p], t=[], toff=[]) for p in CONS_P}
def drop_onto(name, kind, x, y, ztop, t_hit, r=R, z_from=Z_VAULT, dz=0.0):
    z_from = z_from + dz
    b = ball(name, kind, r); h = z_from - (ztop + r); tf = math.sqrt(2 * h / G)
    p0 = np.array([x, y, z_from]); b.rest(-1.0, t_hit - tf, p0); b.fly(t_hit - tf, t_hit, p0, (0, 0, 0))
    return b, np.array([x, y, ztop + r]), flyv((0, 0, 0), tf), t_hit - tf
CHB = []
for k, r in enumerate(ROWS['chord'] + ROWS['dmaj']):
    p = r['ps'][0]; b, pc, vin, t0 = drop_onto(f'C{k}', 'ivory', CONS_X[p], CONS_Y, CONS_Z, r['t'], dz=0.45 * (k % 2))
    add_hit(b, r['t'], pc, vin, np.zeros(3), (0, 0, 1), 'console', r['k'])
    t_off = r['t'] + r['d']
    b.rest(r['t'], t_off, pc)
    # the key tips back: the marble rolls off backwards (+y) into the trough behind the console
    th = math.radians(24); a = 5 / 7 * G * math.sin(th); dr = np.array([0, math.cos(th), -math.sin(th)])
    T_r = 0.32; b.line(t_off, t_off + T_r, pc, dr, 0.0, a); pe = pc + dr * 0.5 * a * T_r ** 2; ve = dr * a * T_r
    b.fly(t_off + T_r, t_off + T_r + 0.3, pe, ve); pt = flyp(pe, ve, 0.3)
    pt[2] = max(pt[2], 0.2); b.rest(t_off + T_r + 0.3, 200.0, pt)
    cons_keys[p]['t'].append(r['t']); cons_keys[p]['toff'].append(t_off); cons_keys[p].setdefault('occ', []).append([b.name, r['t'], t_off])
    CHB.append(dict(name=b.name, t0=t0, t=r['t'], toff=t_off))
# inner voice: one more marble lands on the small E3 key at the right end, hops to F#3 and rests
rE, rF = ROWS['inner']
b, pc, vin, t0 = drop_onto('C10', 'ivory', INNER_X[52], CONS_Y, INNER_Z, rE['t'])
dtE = rF['t'] - rE['t']; vo = vout_for((INNER_X[54] - INNER_X[52], 0, 0), dtE)
add_hit(b, rE['t'], pc, vin, vo, None, 'console', rE['k']); b.fly(rE['t'], rF['t'], pc, vo)
pF = flyp(pc, vo, dtE); vF = flyv(vo, dtE)
add_hit(b, rF['t'], pF, vF, np.zeros(3), (0, 0, 1), 'console', rF['k'])
t_off = rF['t'] + rF['d']; b.rest(rF['t'], t_off, pF)
th = math.radians(24); a = 5 / 7 * G * math.sin(th); dr = np.array([0, math.cos(th), -math.sin(th)])
b.line(t_off, t_off + 0.32, pF, dr, 0.0, a); pe = pF + dr * 0.5 * a * 0.32 ** 2; ve = dr * a * 0.32
b.fly(t_off + 0.32, t_off + 0.62, pe, ve); pt = flyp(pe, ve, 0.3); pt[2] = max(pt[2], 0.2); b.rest(t_off + 0.62, 200.0, pt)
CHB.append(dict(name='C10', t0=t0, t=rE['t'], toff=t_off))
for p_, x_, r_ in ((52, INNER_X[52], rE), (54, INNER_X[54], rF)):
    kk = key('console', [p_], (x_, CONS_Y, INNER_Z), (0, 0, 1), r_['t'], (r_['t'] + r_['d']) if p_ == 52 else t_off, 0.25, 0.62, 0.1, region='inner')
    kk['occ'] = [['C10', r_['t'], (r_['t'] + r_['d']) if p_ == 52 else t_off]]
for p, ck in cons_keys.items():
    o = np.argsort(ck['t'])
    kk = key('console', [p], (ck['x'], CONS_Y, CONS_Z), (0, 0, 1), np.array(ck['t'])[o], np.array(ck['toff'])[o], 0.25, 0.62, 0.1, region='console')
    kk['occ'] = ck['occ']
PARTS['console'] = dict(y=CONS_Y, z=CONS_Z, x={str(p): x for p, x in CONS_X.items()}, inner={str(p): x for p, x in INNER_X.items()}, inner_z=INNER_Z,
                       vault=Z_VAULT, balls=CHB)

# =====================================================================================================================
# 4. the trio (Prestissimo): three gold marbles, one per triplet slot, three stair lanes
# =====================================================================================================================
QP = EVJ['qp']
LANE_Y = [0.28, 0.68, 1.08]; LANE_DZ = [0.0, 0.38, 0.76]
def phase(r):
    if r['t'] <= S['b5'] + 1.8 * QP: return 1
    if r['t'] <= S['b7'] + 1.8 * QP: return 2
    return 3
# who plays which note: always the marble that has been waiting longest (round robin by availability) -> hops of
# 0.4-0.55 s; in the pure-triplet cascade each marble keeps its slot (three descending scales)
TRIO = {0: [], 1: [], 2: []}; last = {0: -9.0, 1: -9.5, 2: -10.0}
first_of = {1: [2, 0, 1], 2: [2, 0, 1], 3: [2, 0, 1]}
for ph in (1, 2, 3):
    rr = [r for r in ROWS['trio'] if phase(r) == ph]
    order = first_of[ph]
    for k, r in enumerate(rr):
        if k < 3: i = order[k]
        else: i = min(last, key=lambda j: last[j])
        TRIO[i].append(r); last[i] = r['t']
def phase(r):
    if r['t'] <= S['b5'] + 1.8 * QP: return 1
    if r['t'] <= S['b7'] + 1.8 * QP: return 2
    return 3
TP = {(i, ph): [r for r in TRIO[i] if phase(r) == ph] for i in range(3) for ph in (1, 2, 3)}
for i in range(3):
    ts = [r['t'] for r in TRIO[i]]; assert min(np.diff(ts)) > 0.3, (i, min(np.diff(ts)))
trio = [ball(f'T{i}', 'gold') for i in range(3)]
TR = {}
# --- start: the marbles sit inside the door (revealed when it opens), roll forward down a chute through the doorway,
#     fly off and land on their first key in front of the door, which turns them to the right (keys turned in plan view)
X_IN = [0.0, 0.48, 0.96]; Y_K1 = 1.18; Y_CH = 1.55; Z_KEY1 = 0.62
starts = {}
for i in range(3):
    b = trio[i]; r0 = TP[(i, 1)][0]
    p1 = np.array([X_IN[i], Y_K1, Z_KEY1 + LANE_DZ[i] + R])                   # ball centre on the first (turning) key
    tf = 0.2; th = math.radians(16); a = 5 / 7 * G * math.sin(th)
    dch = np.array([0, -math.cos(th), -math.sin(th)])
    v_ex = (Y_CH - Y_K1) / tf / math.cos(th); ve = dch * v_ex
    pe = p1 - ve * tf - 0.5 * GZ * tf * tf                                    # chute end (inside the doorway)
    T_roll = v_ex / a; L = 0.5 * a * T_roll ** 2; ps = pe - dch * L
    t_ex = r0['t'] - tf; t_go = t_ex - T_roll
    b.rest(-1.0, t_go, ps); b.line(t_go, t_ex, ps, dch, 0.0, a); b.fly(t_ex, r0['t'], pe, ve)
    starts[i] = dict(rest=ps.tolist(), chute_end=pe.tolist(), t_go=t_go, t_ex=t_ex, L=L, dir=dch.tolist())
    TR[i] = (p1, flyv(ve, tf))
PARTS['trio_start'] = starts
assert min(s['t_go'] for s in starts.values()) > S['dmaj'] + 0.3, [s['t_go'] for s in starts.values()]
PARTS['trio_start'] = starts


def to_cup(b, p, vin, r_last, target, dt, kind='bar', region='', zmin_drop=0.0):
    """last note of a phrase: bounce off the key at p into a cup / cradle at target (xy fixed); the cup height is lowered
    until the bounce is physical. Creates the key. Returns landing point and velocity."""
    tgt = np.array(target, float)
    for dz in np.linspace(-zmin_drop, -2.5, 400):
        q = tgt.copy(); q[2] = p[2] + dz
        vo = vout_for(q - p, dt); e, n = bounce_e(vin, vo)
        tilt = math.degrees(math.acos(max(-1, min(1, n[2]))))
        if EMIN <= e <= EMAX and tilt <= 38 and vo[2] - G * dt < -0.3: break
    add_hit(b, r_last['t'], p, vin, vo, n, kind, r_last['k'])
    tdir = np.array([vo[0], vo[1], 0]); tdir /= max(1e-9, np.linalg.norm(tdir))
    key(kind, r_last['ps'], p - b.r * n, n, r_last['t'], r_last['t'] + r_last['d'], 0.26, 0.4, ball_=b.name, region=region, row=r_last['k'], tdir=tdir)
    b.fly(r_last['t'], r_last['t'] + dt, p, vo)
    return q, flyv(vo, dt)


def prof(i, x0, z0, slope, vx=1.12):
    """target height change of the hop starting at p: follow the lane floor z0 + slope (x - x0) (+ lane offset)"""
    def f(k, p, dt): return (z0 + LANE_DZ[i] + slope * (p[0] + vx * dt - x0)) - p[2]
    return f


def ytrans(i, y0, n=3):
    def f(k): return LANE_Y[i] if k >= n else y0 + (LANE_Y[i] - y0) * k / n
    return f


# --- phrase 1: bars 3 (pickup) - 5 beat 2
P1END = {}
for i in range(3):
    p, vin = TR[i]; rr = TP[(i, 1)]
    p, vin = hops(trio[i], p, vin, rr, lambda k: 1.12, prof(i, X_IN[0], Z_KEY1 + R, -0.1), y_of=ytrans(i, Y_K1), region='stairs1', dvx=(-0.2, 1.2), w=0.2)
    P1END[i] = (p, vin, rr[-1])
# --- the lift: one rack with three cups (one per lane) rises between the phrases
X_L = max(P1END[i][0][0] for i in range(3)) + 0.45
CUPS = {}
for i in range(3):
    p, vin, rl = P1END[i]
    dt = 0.36
    q, vq = to_cup(trio[i], p, vin, rl, (p[0] + 1.1 * dt, LANE_Y[i], 0), dt, region='stairs1', zmin_drop=0.15)
    CUPS[i] = dict(p=q, t=rl['t'] + dt, vin=vq)
T_UP0 = max(c['t'] for c in CUPS.values()) + 0.1
H_LIFT = 3.4; T_LIFT = 1.15; T_UP1 = T_UP0 + T_LIFT
for i in range(3):
    b = trio[i]; c = CUPS[i]
    add_hit(b, c['t'], c['p'], c['vin'], np.zeros(3), (0, 0, 1), 'cup', None)
    b.rest(c['t'], T_UP0, c['p'])
    b.seg.append(dict(k='ease', t0=float(T_UP0), t1=float(T_UP1), p=list(map(float, c['p'])), p1=list(map(float, c['p'] + [0, 0, H_LIFT]))))
PARTS['lift'] = dict(x=X_L, cups={i: (CUPS[i]['p'] - [0, 0, R]).tolist() for i in range(3)}, t_in={i: CUPS[i]['t'] for i in range(3)},
                     t0=T_UP0, t1=T_UP1, h=H_LIFT)
SFX.append(['lift', T_UP0]); SFX.append(['liftstop', T_UP1])


def release(b, p_rest, t_rest, r_first, v_rel=(0.95, 0, 0.0), tf=0.3, kind='cuptip'):
    """the marble rests at p_rest from t_rest; tipped out at r_first.t - tf with v_rel, reaches its first key after tf"""
    t_rel = r_first['t'] - tf
    assert t_rel > t_rest, (b.name, t_rel, t_rest)
    b.rest(t_rest, t_rel, p_rest); b.fly(t_rel, r_first['t'], p_rest, v_rel)
    return flyp(p_rest, v_rel, tf), flyv(v_rel, tf), t_rel


# --- phrase 2: bar 5 pickup - bar 7 beat 2 (octave higher), from the top of the lift
P2END = {}; REL2 = {}
for i in range(3):
    rr = TP[(i, 2)]; ptop = CUPS[i]['p'] + [0, 0, H_LIFT]
    p, vin, tr_ = release(trio[i], ptop, T_UP1, rr[0])
    REL2[i] = tr_
    if i == 0: X2, Z2 = p[0], p[2]
    p, vin = hops(trio[i], p, vin, rr, lambda k: 1.12, prof(i, X2, Z2 - LANE_DZ[0], -0.1), region='stairs2', dvx=(-0.2, 0.15), w=0.2)
    P2END[i] = (p, vin, rr[-1])
PARTS['lift']['t_rel'] = REL2
for i in range(3): SFX.append(['tip', REL2[i]])
# --- landing gate (fermata): three padded cradles, a gate bar releases them one by one for the cascade
X_LD = max(P2END[i][0][0] for i in range(3)) + 0.45
LAND = {}
for i in range(3):
    p, vin, rl = P2END[i]
    dt = 0.36
    q, vq = to_cup(trio[i], p, vin, rl, (p[0] + 1.1 * dt, LANE_Y[i], 0), dt, region='stairs2', zmin_drop=0.15)
    add_hit(trio[i], rl['t'] + dt, q, vq, np.zeros(3), (0, 0, 1), 'cradle', None)
    LAND[i] = dict(p=q, t=rl['t'] + dt)
P3END = {}; REL3 = {}
for i in range(3):
    rr = TP[(i, 3)]
    p, vin, tr_ = release(trio[i], LAND[i]['p'], LAND[i]['t'], rr[0])
    REL3[i] = tr_
    if i == 0: X3, Z3 = p[0], p[2]
    p, vin = hops(trio[i], p, vin, rr, lambda k: 1.12, prof(i, X3, Z3 - LANE_DZ[0], -0.38), region='cascade', dvx=(-0.2, 0.15), w=0.2)
    P3END[i] = (p, vin, rr[-1])
PARTS['landing'] = dict(x=X_LD, cradles={i: (LAND[i]['p'] - [0, 0, R]).tolist() for i in range(3)}, t_in={i: LAND[i]['t'] for i in range(3)},
                        t_rel=REL3)
for i in range(3): SFX.append(['gate', REL3[i]])

# =====================================================================================================================
# 5. bar 10: the trio waits in catch cradles, drops together onto the dim7 keys (octave pairs); bar 11 they roll away
# =====================================================================================================================
C2 = {r['ps'][0]: r for r in ROWS['chord2']}
CH_OF = {0: 49, 1: 52, 2: 55}
T_CH10 = S['ch10']; T_B11 = S['b11']
CATCH = {}
for i in range(3):
    p, vin, rl = P3END[i]
    q, vq = to_cup(trio[i], p, vin, rl, (p[0] + 1.1 * 0.36, LANE_Y[i], 0), 0.36, region='cascade', zmin_drop=0.15)
    t_in = rl['t'] + 0.36
    add_hit(trio[i], t_in, q, vq, np.zeros(3), (0, 0, 1), 'cradle', None)
    rch = C2[CH_OF[i]]
    pk, vk, t_rel = release(trio[i], q, t_in, rch, v_rel=(0.7, 0, 0.0), tf=0.32)
    add_hit(trio[i], rch['t'], pk, vk, np.zeros(3), (0, 0, 1), 'chordkey', rch['k'])
    kk = key('chordkey', rch['ps'], pk - [0, 0, R], (0, 0, 1), rch['t'], rch['t'] + rch['d'], 0.36, 0.42, 0.08, ball_=trio[i].name, region='bar10', row=rch['k'])
    kk['occ'] = [[trio[i].name, rch['t'], rch['t'] + rch['d']]]
    t_off = rch['t'] + rch['d']
    trio[i].rest(rch['t'], t_off, pk)
    th = math.radians(28); a = 5 / 7 * G * math.sin(th); dr = np.array([0, math.cos(th), -math.sin(th)])
    trio[i].line(t_off, t_off + 0.3, pk, dr, 0, a); pe = pk + dr * 0.5 * a * 0.09; ve = dr * a * 0.3
    trio[i].fly(t_off + 0.3, t_off + 1.0, pe, ve); pt = flyp(pe, ve, 0.7); trio[i].rest(t_off + 1.0, 200.0, pt)
    CATCH[i] = dict(cradle=(q - [0, 0, R]).tolist(), t_in=t_in, t_rel=t_rel, key=(pk - [0, 0, R]).tolist(), t_off=t_off, exit=pt.tolist())
    SFX.append(['gate', t_rel])
PARTS['catch'] = CATCH
X_CH = max(c['key'][0] for c in CATCH.values()); Z_CH = min(c['key'][2] for c in CATCH.values())

# =====================================================================================================================
# 6. the first drop: one gold marble leaks from the reservoir onto the Bb key (bar 10 beat 4), bar 11 solo
# =====================================================================================================================
Y_S = -0.38
rBb = C2[58]; sol = ball('DROP', 'gold')
X_BB = X_CH + 0.75; Z_BB = Z_CH + 0.15
pb = np.array([X_BB, Y_S, Z_BB + R])
TF_SP = 0.55; z_sp = pb[2] + 0.5 * G * TF_SP ** 2
p_sp = np.array([X_BB, Y_S, z_sp])
sol.rest(-1.0, rBb['t'] - TF_SP, p_sp); sol.fly(rBb['t'] - TF_SP, rBb['t'], p_sp, (0, 0, 0))
th = math.radians(13); a = 5 / 7 * G * math.sin(th); dr = np.array([math.cos(th), 0, -math.sin(th)]); nr = np.array([math.sin(th), 0, math.cos(th)])
add_hit(sol, rBb['t'], pb, flyv((0, 0, 0), TF_SP), np.zeros(3), nr, 'cradle', rBb['k'])
key('cradle', rBb['ps'], pb - R * nr - dr * 0.05, nr, rBb['t'], rBb['t'] + rBb['d'], 0.3, 0.42, ball_='DROP', region='bar10', row=rBb['k'], tdir=dr)
SFX.append(['spout', rBb['t'] - TF_SP])
SO = ROWS['solo']
run7 = SO[:7]; hop5 = SO[7:12]; tri = SO[12:18]; rE = SO[18]
assert [r['ps'][0] for r in run7] == [69, 67, 65, 64, 62, 61, 59] and [r['note'] for r in tri] == ['trill'] * 6
t_kick = run7[0]['t'] - 0.03
sol.rest(rBb['t'], t_kick, pb)
VK = 1.25
sp = lambda t: VK * (t - t_kick) + 0.5 * a * (t - t_kick) ** 2
t_rend = hop5[0]['t'] - 0.1
for k, r in enumerate(run7):
    s0 = sp(r['t']); s1 = sp(run7[k + 1]['t']) if k + 1 < 7 else sp(t_rend)
    vv = dr * (VK + a * (r['t'] - t_kick))
    add_hit(sol, r['t'], pb + dr * s0, vv, vv, nr, 'slat', r['k'])
    key('slat', r['ps'], pb + dr * (s0 + s1) / 2 - R * nr, nr, r['t'], r['t'] + r['d'], s1 - s0 - 0.012, 0.42, 0.06, ball_='DROP', region='solo', row=r['k'], tdir=dr)
sol.line(t_kick, t_rend, pb, dr, VK, a)
pe = pb + dr * sp(t_rend); ve = dr * (VK + a * (t_rend - t_kick))
sol.fly(t_rend, hop5[0]['t'], pe, ve); p = flyp(pe, ve, 0.1); vin = flyv(ve, 0.1)
SFX.append(['flick', t_kick])
PARTS['solo'] = dict(spout=p_sp.tolist(), bb=(pb - [0, 0, R]).tolist(), t_kick=t_kick, ramp_p0=pb.tolist(), ramp_dir=dr.tolist(), ramp_n=nr.tolist(),
                     ramp_len=float(sp(t_rend)))
# hops C# A C# E G, then into the trill shaft
hrows = hop5 + [tri[0]]
p, vin = hops(sol, p, vin, hrows, lambda k: 1.15, lambda k, p_, dt: -0.12, region='solo', dvx=(-0.35, 0.1), w=0.17, d=0.38)
# the trill shaft: walls G (right) / F (left), one key plate per hit; the ball zig-zags down
SH = []
vx = vin[0]
for k, r in enumerate(tri):
    s_ = 1 if k % 2 == 0 else -1                   # G = right wall (normal -x), F = left wall (normal +x)
    n = np.array([-s_, 0, 0.0]); vo = vin.copy(); vo[0] = -0.93 * vin[0]
    add_hit(sol, r['t'], p, vin, vo, n, 'wall', r['k'])
    key('wall', r['ps'], p - R * n, n, r['t'], r['t'] + r['d'], 0.24, 0.38, 0.06, ball_='DROP', region='solo', row=r['k'], tdir=(0, 0, -1))
    SH.append((p - R * n).tolist())
    nxt = tri[k + 1]['t'] if k + 1 < len(tri) else rE['t']
    dt = nxt - r['t']; sol.fly(r['t'], nxt, p, vo); p = flyp(p, vo, dt); vin = flyv(vo, dt)
# E: floor of the shaft, then the hop onto D4 of the final chord
T_FINAL = S['final']; dt = T_FINAL - rE['t']
vo = vout_for((0.75 * dt, 0, -0.32), dt); e, n = bounce_e(vin, vo)
add_hit(sol, rE['t'], p, vin, vo, n, 'bar', rE['k'])
key('bar', rE['ps'], p - R * n, n, rE['t'], rE['t'] + rE['d'], 0.28, 0.38, ball_='DROP', region='solo', row=rE['k'], tdir=(1, 0, 0))
sol.fly(rE['t'], T_FINAL, p, vo); P_D4 = flyp(p, vo, dt); V_D4 = flyv(vo, dt)
PARTS['solo']['shaft'] = SH
Z_F = P_D4[2] - R                                    # top of the final keys
print('solo e at E', round(e, 3), 'D4 at', np.round(P_D4, 2), 'Z_F', round(Z_F, 2), 'Bb key', np.round(pb, 2))

# =====================================================================================================================
# 7. GIANT II, the reservoir, the flood (final chord)
# =====================================================================================================================
TILT_F = math.radians(20.0); N_F = np.array([0, math.sin(TILT_F), math.cos(TILT_F)])
FK_P = [38, 50, 62, 65, 74, 77]                                      # D2 (pedal) D3 D4 F4 D5 F5
FK_X = {62: P_D4[0]}; FK_X[50] = P_D4[0] - 0.68; FK_X[38] = FK_X[50] - 0.72
for p_, k in ((65, 1), (74, 2), (77, 3)): FK_X[p_] = P_D4[0] + 0.68 * k
FK_W = {p_: (0.66 if p_ == 38 else 0.62) for p_ in FK_P}
FK_Y0, FK_YL, FK_Y1 = -0.62, -0.12, 0.95                             # flat front lip y0..yl, then the slope down to y1
def zf(y): return Z_F if y <= FK_YL else Z_F - (y - FK_YL) * math.tan(TILT_F)
Y_COL = 0.42                                                           # the rain columns (one row of golden pipes)
# --- GIANT II rolls off a sloped ledge at the left and lands on the pedal key (bar 10 beat 3), against its stop pin
rP2 = ROWS['ped'][1]; g2 = ball('GIANT2', 'giant', RG)
Y_G2 = Y_COL
pland = np.array([FK_X[38], Y_G2, zf(Y_G2) + RG / math.cos(TILT_F)])
thl = math.radians(20); al = 5 / 7 * G * math.sin(thl); dl = np.array([math.cos(thl), 0, -math.sin(thl)])
tfl = 0.55; vx = 2.1; vl = dl * (vx / math.cos(thl))
pedge = pland - vl * tfl - 0.5 * GZ * tfl * tfl
T_roll = np.linalg.norm(vl) / al; Ll = 0.5 * al * T_roll ** 2; pst = pedge - dl * Ll
t_go2 = rP2['t'] - tfl - T_roll
g2.rest(-1.0, t_go2, pst); g2.line(t_go2, rP2['t'] - tfl, pst, dl, 0, al); g2.fly(rP2['t'] - tfl, rP2['t'], pedge, vl)
vin = flyv(vl, tfl)
add_hit(g2, rP2['t'], pland, vin, np.zeros(3), N_F, 'pedal', rP2['k'])
t_off2 = rP2['t'] + rP2['d']
g2.rest(rP2['t'], t_off2, pland)
# the stop pin drops: GIANT II rolls down the pedal's slope into the counterweight bucket behind, which sinks and
# winches the floodgates of the golden pipes open
dback = np.array([0, math.cos(TILT_F), -math.sin(TILT_F)]); ab = 5 / 7 * G * math.sin(TILT_F)
Lb = (FK_Y1 - Y_G2) / math.cos(TILT_F); T_b = math.sqrt(2 * Lb / ab)
g2.line(t_off2, t_off2 + T_b, pland, dback, 0, ab); pe = pland + dback * Lb; ve = dback * ab * T_b
T_FINAL = S['final']; TFALL_R = 0.75
T_GATE = T_FINAL - TFALL_R
tfb = 0.32; pbk = flyp(pe, ve, tfb); g2.fly(t_off2 + T_b, t_off2 + T_b + tfb, pe, ve)
add_hit(g2, t_off2 + T_b + tfb, pbk, flyv(ve, tfb), np.zeros(3), (0, 0, 1), 'bucket', None)
CW_DROP = 1.3; t_cw0 = t_off2 + T_b + tfb + 0.15; t_cw1 = T_GATE
g2.rest(t_off2 + T_b + tfb, t_cw0, pbk); g2.seg.append(dict(k='ease', t0=t_cw0, t1=t_cw1, p=pbk.tolist(), p1=(pbk - [0, 0, CW_DROP]).tolist()))
g2.rest(t_cw1, 200.0, pbk - [0, 0, CW_DROP])
PARTS['giant2'] = dict(start=pst.tolist(), edge=pedge.tolist(), t_go=t_go2, pedal=(pland - RG * N_F).tolist(), t_on=rP2['t'], t_off=t_off2,
                       bucket=(pbk - [0, 0, RG]).tolist(), t_cw0=t_cw0, t_cw1=t_cw1, drop=CW_DROP)
SFX.append(['ledge', t_go2]); SFX.append(['pin', t_off2]); SFX.append(['bucket', t_off2 + T_b + tfb]); SFX.append(['winch', t_cw0])
# --- the golden pipes: 18 glass tubes (3 over each final key) packed with gold marbles; at T_GATE their floodgates open,
#     each column drains at a constant rate (hopper flow), every marble free-falls onto its key and bounces back into
#     the trough behind (the keys slope back 20 deg, so a bounced marble leaves its stream at once)
NR = 565 - len(BALLS)
U_DRAIN = 2.1
cols = [(FK_X[p_] + dx, p_) for p_ in FK_P for dx in (-0.205, 0.0, 0.205)]
NL = int(math.ceil(NR / len(cols)))
slots = sorted(rng.permutation(len(cols) * NL)[:NR], key=lambda q: (q // len(cols), q % len(cols)))
RAIN_HITS = []; PIPES = []
ztop = zf(Y_COL) + RR / math.cos(TILT_F); ZB = ztop + U_DRAIN * TFALL_R + 0.5 * G * TFALL_R ** 2
for j, q in enumerate(slots):
    L = q // len(cols); cx, p_ = cols[q % len(cols)]
    p0 = np.array([cx, Y_COL, ZB + L * 2 * RR * 1.03])
    b = ball(f'R{j:03d}', 'rain', RR)
    t_exit = T_GATE + (p0[2] - ZB) / U_DRAIN
    pex = np.array([cx, Y_COL, ZB]); tf = TFALL_R
    b.rest(-1.0, T_GATE, p0); b.carry(T_GATE, t_exit, p0, pex); b.fly(t_exit, t_exit + tf, pex, (0, 0, -U_DRAIN))
    th_ = t_exit + tf; vin = np.array([0, 0, -U_DRAIN - G * tf]); ph = np.array([cx, Y_COL, ztop])
    e = 0.42 + rng.uniform(-0.04, 0.04)
    vo = vin - (1 + e) * np.dot(vin, N_F) * N_F
    add_hit(b, th_, ph, vin, vo, N_F, 'rain', None)
    t_tr = 0.8; pt = flyp(ph, vo, t_tr)
    b.fly(th_, th_ + t_tr, ph, vo); b.carry(th_ + t_tr, th_ + t_tr + 0.4, pt, pt - [0, 0, 2.5]); b.rest(th_ + t_tr + 0.4, 200.0, pt - [0, 0, 2.5])
    RAIN_HITS.append([float(th_), int(p_), b.name])
RAIN_HITS.sort()
for p_ in FK_P:
    ts = sorted([h[0] for h in RAIN_HITS if h[1] == p_] + ([T_FINAL] if p_ == 62 else []))
    t_end = T_FINAL + EVJ['hold']
    if p_ == 38: ts = [rP2['t']] + ts; toff = [t_off2] + [t_end] * (len(ts) - 1)
    else: toff = [t_end] * len(ts)
    kk = key('final', [p_], (FK_X[p_], (FK_Y0 + FK_Y1) / 2, Z_F), (0, 0, 1), ts, toff, FK_W[p_], FK_Y1 - FK_Y0, 0.14, region='final',
             row=ROWS['final'][0]['k'])
    kk['occ'] = [[h[2], h[0], h[0] + 0.1] for h in RAIN_HITS if h[1] == p_] + ([['DROP', T_FINAL, 200.0]] if p_ == 62 else []) + \
        ([['GIANT2', rP2['t'], t_off2 + 0.6]] if p_ == 38 else [])
add_hit(sol, T_FINAL, P_D4, V_D4, np.zeros(3), (0, 0, 1), 'final', ROWS['final'][0]['k'])
sol.rest(T_FINAL, 200.0, P_D4)
assert abs(RAIN_HITS[0][0] - T_FINAL) < 1e-6, RAIN_HITS[0]
PIPE = dict(xs=[c[0] for c in cols], y=Y_COL, r=RR + 0.022, z0=ZB - RR - 0.05, z1=ZB + NL * 2 * RR * 1.03 + 0.25, t_gate=T_GATE, u=U_DRAIN,
            layers=NL)
PARTS['final'] = dict(keys={str(p_): dict(x=FK_X[p_], w=FK_W[p_]) for p_ in FK_P}, y0=FK_Y0, yl=FK_YL, y1=FK_Y1, zf=Z_F, ys=Y_S,
                      tilt=TILT_F, pipes=PIPE, rain_hits=RAIN_HITS, t_final=T_FINAL, t_last=RAIN_HITS[-1][0])
SFX.append(['gate_open', T_GATE])
# counter: every marble that has played a key (first strike)
first = []
for n, b in BALLS.items():
    ts = [h['t'] for h in b.hits if h['kind'] not in ('lever', 'cup', 'cradle', 'bucket') or h.get('row') is not None]
    if ts: first.append(min(ts))
first.sort()
assert len(first) == 565, len(first)
PARTS['counter'] = first


# =====================================================================================================================
def check():
    info = {}
    hs = [(n, h) for n, b in BALLS.items() for h in b.hits]
    bo = [(n, h) for n, h in hs if np.linalg.norm(h['vout']) > 0 and h['kind'] not in ('slat',)]
    info['bounces'] = len(bo)
    info['e_max'] = round(max(h['e'] for n, h in bo), 3); info['e_min'] = round(min(h['e'] for n, h in bo), 3)
    info['speed_out_over_in_max'] = round(max(np.linalg.norm(h['vout']) / max(1e-9, np.linalg.norm(h['vin'])) for n, h in bo), 3)
    info['tilt_max_deg'] = round(max(math.degrees(math.acos(max(-1, min(1, h['n'][2])))) for n, h in hs if h['kind'] not in ('wall',)), 1)
    rows_struck = set(h['row'] for n, h in hs if h.get('row') is not None)
    info['rows_unstruck'] = sorted(set(range(len(LEAD))) - rows_struck)
    info['keys'] = len(KEYS); info['balls'] = len(BALLS)
    return info


if __name__ == '__main__':
    info = check()
    for k, v in info.items(): print(k, v)
    out = dict(G=G, R=R, RG=RG, RR=RR, balls={n: dict(kind=b.kind, r=b.r, seg=b.seg, hits=b.hits) for n, b in BALLS.items()},
               keys=KEYS, parts=PARTS, sfx=sorted(SFX, key=lambda s: s[1]), lane_y=LANE_Y)
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
    print('layout.json', os.path.getsize(f'{HERE}/layout.json') // 1024, 'KB')
