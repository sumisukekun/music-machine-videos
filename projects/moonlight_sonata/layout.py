"""Moonlight Sonata - 'Where is this lake?'  Physics + story planner (run after arrange.py; writes layout.json).

Stage: a black mirror 'lake' at night under a full moon (it is the closed lid of a grand piano in a moonlit room,
revealed at the end).  x = travel direction (tail of the piano -> keyboard), y = depth (camera at -y), z = up, lid top z = 0.
Voices / balls (every note = one visible strike at its exact time):
  trio A/B/C   three glass marbles, one per voice of the triplet arpeggio (1st/2nd/3rd note of every triplet), each
               bounces once per beat on its own lane of glass lotus chimes (staggered by 1/3 beat) that descend
               gently (energy loss: restitution 0.97).  They start in the crescent cradle and drop out one by one.
  melody       the silver marble rides in the moon gondola (a cable car), is dropped onto its first key, plays the theme
               on silver keys; every long note is a see-saw: the bass marble lands in the bowl on the far end at the
               same instant (melody long notes always start together with a bass note) and the near end throws the
               silver marble up into a long slow arc.  Its last arc ends in the lantern of the toy lighthouse (c#').
  bass         one dark 'night' marble per bass note, dropped from the gondola (free fall from rest relative to the
               gondola, which runs at the procession speed) into a bronze bowl (plain or see-saw).
  ending       bar 67: trio A strikes the bass plate (the low C# pedal) then all three play the descending arpeggio;
               bar 68: they land together in the launch tray at the lid's front edge while the bass marble drops into
               the catapult bowl (low C#) -> the tray kicks them over the edge onto the piano keys (e3 g#3 c#4) while
               three marbles from the gondola land on C#2 G#2 C#3: the final chord, struck again by the same six
               marbles bouncing in place (bar 69).
Physics: one constant gravity G; between contacts pure parabolas; at a free bounce the surface normal is along the
impulse (tangential velocity kept) with restitution emin <= e <= emax (never gains speed); speed gains only from the
see-saws / tray (struck by a falling bass marble at that instant)."""
import json, os, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = 5.0; VX = 0.5
R = dict(trio=0.2, mel=0.22, bass=0.22, fin=0.15)
E = json.load(open(f'{HERE}/events.json')); LEAD = json.load(open(f'{HERE}/lead.json'))
T0, Q = E['t0'], E['q']; BARS = {int(k): v for k, v in E['bars'].items()}
Y = dict(A=-3.0, B=-2.26, C=-1.37, G=-4.33, M=-5.4)
EMAX, EMIN = 0.97, 0.2
XOFF = -0.35
def X(t): return VX * (t - T0) + XOFF


# ---- piano geometry (keys, lid edge) ----
LID_FRONT = 30.3; KEY_BACK = 30.5; WHITE_FRONT = 32.25; BLACK_FRONT = 31.45
Z_WHITE = -3.25; Z_BLACK = Z_WHITE + 0.12; WK = 0.296
WHITE_PC = [0, 2, 4, 5, 7, 9, 11]
def white_index(m):                      # index of a white key counted from A0 = 0
    o, pc = divmod(m - 12, 12)            # C-based octave (C0 = 12)
    return o * 7 + WHITE_PC.index(pc) - 5
Y_A0 = Y['A'] - (white_index(52) + 0.5) * WK    # E3 (52) centred on lane A
def key_geom(m):
    """(y centre, half width, top z, x back, x front, black?)"""
    if m % 12 in WHITE_PC:
        return Y_A0 + (white_index(m) + 0.5) * WK, WK / 2 - 0.006, Z_WHITE, KEY_BACK, WHITE_FRONT, False
    return Y_A0 + white_index(m + 1) * WK, 0.08, Z_BLACK, KEY_BACK, BLACK_FRONT, True


class Ball:
    def __init__(s, name, role, r):
        s.name = name; s.role = role; s.R = r; s.seg = []; s.hits = []; s.show = -1.0
    def rest(s, t0, t1, p): s.seg.append(dict(k='rest', t0=float(t0), t1=float(t1), p=list(map(float, p))))
    def lin(s, t0, t1, p, v): s.seg.append(dict(k='lin', t0=float(t0), t1=float(t1), p=list(map(float, p)), v=list(map(float, v))))
    def fly(s, t0, t1, p, v): s.seg.append(dict(k='fly', t0=float(t0), t1=float(t1), p=list(map(float, p)), v=list(map(float, v))))
    def at(s, t):
        for g in s.seg:
            if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9:
                p = np.array(g['p']); u = t - g['t0']
                if g['k'] == 'rest': return p
                if g['k'] == 'lin': return p + np.array(g['v']) * u
                return p + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
        return s.at(s.seg[0]['t0']) if t < s.seg[0]['t0'] else s.at(s.seg[-1]['t1'])


def vout_for(d, dt): return np.array([d[0] / dt, d[1] / dt, d[2] / dt + 0.5 * G * dt])
def arrive(vo, dt): return vo + np.array([0, 0, -G * dt])
def bounce_e(vin, vout):
    d = vout - vin; n = d / max(1e-9, np.linalg.norm(d)); a = vin @ n
    return (-(vout @ n) / a if a < -1e-9 else 9.0), n
def tilt(n): return math.degrees(math.acos(max(-1, min(1, n[2]))))


def add_hit(b, t, p, vin, vout, kind, pitch=None, mech=False, **kw):
    e, n = bounce_e(np.array(vin, float), np.array(vout, float))
    if mech or np.linalg.norm(vout) < 1e-9: n = np.array([0, 0, 1.0])
    h = dict(t=float(t), p=list(map(float, p)), n=list(map(float, n)), e=float(e), kind=kind, vin=list(map(float, vin)),
             vout=list(map(float, vout)), mech=bool(mech), pitch=pitch)
    h.update(kw); b.hits.append(h); return h


B = {}; EV = {}; SFX = []
PNAME = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def pn(m): return PNAME[m % 12] + str(m // 12 - 1)

# ------------------------------------------------------------------ trio
TRI = E['tri']
T_TRAY = BARS[68]                      # 60.8: tray launch (with the catapult bowl's low C#)
T_FIN1, T_FIN2 = BARS[68] + 2 * Q, BARS[69]
K_DESC = 0.04; Z_END = 0.44
def z_trio(t): return Z_END + K_DESC * (T_TRAY - t)
lanes = {'A': [], 'B': [], 'C': []}
for t, p in TRI:
    k = int(math.floor((t - T0) / Q + 1e-6)); pos = int(round(((t - T0) / Q - k) * 3))
    lanes['ABC'[pos]].append([t, p, 'pad'])
bass_plate = [ev for ev in E['bass_ev'] if abs(ev[0] - BARS[67]) < 1e-6][0]
lanes['A'].append([bass_plate[0], bass_plate[1], 'plate'])
for k in lanes: lanes[k].sort()
FINKEY = {'A': 52, 'B': 56, 'C': 61}           # e3, g#3, c#4 (RH of the final chord)
CRADLE = {}
X_TRAY = 30.0                       # the launch tray at the lid's front edge
TRAY_DZ = 0.12                      # ... a little lower than the last lotus row
for k in 'ABC':
    b = B['trio' + k] = Ball('trio' + k, 'trio', R['trio'])
    seq = lanes[k]
    t1 = seq[0][0]; p1 = np.array([X(t1), Y[k], z_trio(t1)])
    HF = 0.9; tf = math.sqrt(2 * HF / G)
    pc = p1 + np.array([0, 0, HF])                       # nest in the crescent cradle right above the first lotus
    b.rest(-1.0, t1 - tf, pc); b.fly(t1 - tf, t1, pc, [0, 0, 0])
    CRADLE[k] = dict(p=pc.tolist(), t_release=t1 - tf); SFX.append(['hatch', t1 - tf])
    vin = np.array([0, 0, -G * tf]); p = p1
    for j in range(len(seq)):
        t, pitch, kind = seq[j]
        tn = seq[j + 1][0] if j + 1 < len(seq) else T_TRAY
        pn_ = np.array([X(tn) if j + 1 < len(seq) else X_TRAY, Y[k], z_trio(tn) - (0 if j + 1 < len(seq) else TRAY_DZ)])
        dt = tn - t; vo = vout_for(pn_ - p, dt)
        add_hit(b, t, p, vin, vo, kind, pitch, lane=k)
        b.fly(t, tn, p, vo); vin = arrive(vo, dt); p = pn_
    # tray: lands at T_TRAY and is kicked over the lid edge onto its key
    m = FINKEY[k]; ky, hw, ztop, xb, xf, blk = key_geom(m)
    kx = 31.0 if blk else 31.8
    pk = np.array([kx, ky, ztop + R['trio']])
    vo = vout_for(pk - p, T_FIN1 - T_TRAY)
    add_hit(b, T_TRAY, p, vin, vo, 'tray', None, mech=True, lane=k)
    b.fly(T_TRAY, T_FIN1, p, vo); vin = arrive(vo, T_FIN1 - T_TRAY)
    vo = vout_for(np.zeros(3), T_FIN2 - T_FIN1)
    add_hit(b, T_FIN1, pk, vin, vo, 'pkey', m, lane=k)
    b.fly(T_FIN1, T_FIN2, pk, vo); vin = arrive(vo, T_FIN2 - T_FIN1)
    add_hit(b, T_FIN2, pk, vin, np.zeros(3), 'pkey', m, lane=k)
    b.rest(T_FIN2, 99, pk); SFX.append(['settle', T_FIN2 + 0.02])
EV['cradle'] = CRADLE
EV['tray'] = dict(t=T_TRAY, x=X_TRAY, z=z_trio(T_TRAY) - TRAY_DZ - R['trio'], y0=Y['A'] - 0.3, y1=Y['C'] + 0.3)

# ------------------------------------------------------------------ gondola + bass
Z_REL = 7.05                                  # centre of a marble in the gondola's release hatch
XG_D = 0.5                                    # the gondola rides half a unit ahead of the trio
def XG(t): return X(t) + XG_D
EV['gondola'] = dict(z_rel=Z_REL, y=Y['G'], vx=VX)
LEADT = [l[0] for l in LEAD]
SEESAW_T = [t for t, p, v, d in LEAD if d >= 2 - 1e-6 or abs(t - BARS[46]) < 1e-6]      # long notes (+ the e' before the rest)
SEESAW_T = [t for t in SEESAW_T if any(abs(t - ev[0]) < 1e-6 for ev in E['bass_ev']) and t < BARS[60] - 1e-6]
T_S0 = BARS[5]; SEESAW_T = sorted(SEESAW_T + [T_S0])     # S0: the bar-5 bass throws the waiting silver marble up
Z_BOWL = 0.45                # contact centre of a bass marble in a plain bowl
Z_SSB = 0.95                 # ... in the raised bowl of a see-saw
Z_SSA = 0.62                 # melody marble on the lowered key end of a see-saw
Z_CAT = 0.75                 # catapult bowl (raised)
KEY_TH = 0.07; BOWL_BOT = 0.05; T_TIP = 0.08; CAT_DROP = 0.22; TRAY_UP = 0.12
SS_L = (Y['G'] - Y['M']) / 2
_zA = Z_SSA - R['mel'] - KEY_TH; _zB = Z_SSB - R['bass'] - BOWL_BOT
SS_ZP = (_zA + _zB) / 2; SS_TH = math.atan2(_zB - _zA, 2 * SS_L)      # beam tilt (A end down) before the bass lands
def SS_PIV(x): return np.array([x, (Y['G'] + Y['M']) / 2, SS_ZP])
EV['seesaw_geom'] = dict(L=SS_L, zp=SS_ZP, th=SS_TH, key_th=KEY_TH, bowl_bot=BOWL_BOT, t_tip=T_TIP, beam_zA=_zA, beam_zB=_zB)
EV['catapult'] = dict(drop=CAT_DROP, tray_up=TRAY_UP, t_tip=T_TIP)
def drop(b, t, p_land, vin_note=None, show=None):
    """b falls from the gondola hatch (moving with the gondola) and lands at p_land at time t"""
    h = Z_REL - p_land[2]; tf = math.sqrt(2 * h / G)
    pr = np.array([p_land[0] - VX * tf, p_land[1], Z_REL])
    b.show = t - tf if show is None else show
    b.lin(-2.0, t - tf, pr - np.array([VX * (t - tf + 2.0), 0, 0]), [VX, 0, 0])      # rides inside the gondola
    b.fly(t - tf, t, pr, [VX, 0, 0])
    return np.array([VX, 0, -G * tf]), t - tf
BASS = []
for i, (t, ps) in enumerate(E['bass_ev']):
    if abs(t - BARS[67]) < 1e-6: continue      # struck by trio A on the bass plate
    b = B[f'bass{i:02d}'] = Ball(f'bass{i:02d}', 'bass', R['bass'])
    ss = any(abs(t - s) < 1e-6 for s in SEESAW_T); cat = abs(t - T_TRAY) < 1e-6
    z = Z_SSB if ss else (Z_CAT if cat else Z_BOWL)
    pl = np.array([XG(t), Y['G'], z])
    if i > 0 and t - E['bass_ev'][i - 1][0] < 1.1 and not (ss or cat): pl[1] += 0.62     # close pair: the gondola's side hatch
    vin, trel = drop(b, t, pl)
    kind = 'ssbowl' if ss else ('catbowl' if cat else 'bowl')
    add_hit(b, t, pl, vin, np.zeros(3), kind, ps, mech=True)
    # settles in the bowl (a see-saw / catapult bowl goes down with its beam in T_TIP seconds)
    if ss:
        rel = pl - SS_PIV(pl[0]); c, s_ = math.cos(-2 * SS_TH), math.sin(-2 * SS_TH)
        after = SS_PIV(pl[0]) + np.array([0, rel[1] * c - rel[2] * s_, rel[1] * s_ + rel[2] * c])
    elif cat: after = pl + np.array([0, 0, -CAT_DROP])
    else: after = pl
    if ss or cat:
        b.lin(t, t + T_TIP, pl, (after - pl) / T_TIP); b.rest(t + T_TIP, 99, after)
    else: b.rest(t, 99, pl)
    BASS.append(dict(t=t, pitches=ps, kind=kind, p=pl.tolist(), t_release=trel, ball=b.name))
    if ss: SFX.append(['seesaw', t])
    if cat: SFX.append(['catapult', t])
# final chord, left hand: three marbles from the gondola onto C#2 G#2 C#3, bounce in place, rest
FIN_L = [37, 44, 49]
for j, m in enumerate(FIN_L):
    b = B[f'fin{j}'] = Ball(f'fin{j}', 'fin', R['fin'])
    ky, hw, ztop, xb, xf, blk = key_geom(m)
    pk = np.array([XG(T_FIN1), ky, ztop + R['fin']])
    h = Z_REL - pk[2]; tf = math.sqrt(2 * h / G)
    pr = np.array([pk[0] - VX * tf, ky, Z_REL]); b.show = T_FIN1 - tf
    b.lin(-2.0, T_FIN1 - tf, pr - np.array([VX * (T_FIN1 - tf + 2.0), 0, 0]), [VX, 0, 0])
    b.fly(T_FIN1 - tf, T_FIN1, pr, [VX, 0, 0]); vin = np.array([VX, 0, -G * tf])
    vo = vout_for(np.zeros(3), T_FIN2 - T_FIN1)
    add_hit(b, T_FIN1, pk, vin, vo, 'pkey', m)
    b.fly(T_FIN1, T_FIN2, pk, vo); vin = arrive(vo, T_FIN2 - T_FIN1)
    add_hit(b, T_FIN2, pk, vin, np.zeros(3), 'pkey', m)
    b.rest(T_FIN2, 99, pk)
EV['fin_release'] = T_FIN1 - math.sqrt(2 * (Z_REL - (Z_BLACK + R['fin'])) / G)
EV['bass'] = BASS

# ------------------------------------------------------------------ melody
# the silver marble waits from the start on the key end of see-saw S0; the bass marble of bar 5 (c#) lands in S0's
# bowl and throws it up: the theme starts with its landing (silent launch: S0's key is not struck, it is pushed).
M = B['mel'] = Ball('mel', 'mel', R['mel'])
TL = np.array([T_S0] + LEADT); PL = [None] + [l[1] for l in LEAD]; NL = len(TL)
MECH = set(); fixed = {}
for i, t in enumerate(TL):
    if any(abs(t - s) < 1e-6 for s in SEESAW_T) and t < BARS[60] - 1e-6:
        fixed[i] = np.array([XG(t), Y['M'], Z_SSA]); MECH.add(i)
LAMP = np.array([26.3, Y['M'], 4.3])
fixed[NL - 1] = LAMP
EV['lamp'] = dict(p=LAMP.tolist(), t=float(TL[-1]))
def zpitch(m): return 0.95 + (m - 66) * 0.11
KHW, KHD, KTH = 0.1, 0.26, 0.07                 # silver key half sizes (x, y) and thickness
KGAP = 0.36
def keys_clear(pts, vs, ts):
    """the marble must not touch the key it just left nor the one it is about to hit (except at the contacts)"""
    r = R['mel']
    for k in range(len(pts) - 1):
        dt = ts[k + 1] - ts[k]
        for kk in (k, k + 1):
            c = pts[kk] - np.array([0, 0, r]); ctr = c - np.array([0, 0, KTH / 2])
            for u in np.linspace(0.01, dt - 0.01, 24):
                p = pts[k] + vs[k] * u + np.array([0, 0, -0.5 * G * u * u])
                d = np.maximum(0, np.abs(p - ctr) - np.array([KHW, KHD, KTH / 2]))
                if np.linalg.norm(d) < r - 0.005: return False
    return True
def plan_group(i0, i1, p0, v0):
    """free hits i0+1 .. i1-1 between hit i0 (position p0, incoming v0) and fixed hit i1.
    a see-saw at i0 may launch with any upward speed; every other bounce must keep emin <= e <= emax."""
    import itertools
    free = list(range(i0 + 1, i1)); p1 = fixed[i1]
    best = None
    zs = np.arange(0.45, 3.41, 0.05)
    for zc in (itertools.product(zs, repeat=len(free)) if free else [()]):
        pts = [p0]
        xs_ = [p0[0] + (p1[0] - p0[0]) * (TL[i] - TL[i0]) / (TL[i1] - TL[i0]) for i in free]
        for k in range(len(xs_) - 1, -1, -1):          # keep neighbouring keys apart (>= KGAP between centres)
            xs_[k] = min(xs_[k], (xs_[k + 1] if k + 1 < len(xs_) else p1[0]) - KGAP)
        for k, i in enumerate(free):
            pts.append(np.array([xs_[k], Y['M'], zc[k]]))
        pts.append(p1)
        vs = [vout_for(pts[k + 1] - pts[k], TL[i0 + k + 1] - TL[i0 + k]) for k in range(len(pts) - 1)]
        ok = True; cost = 0.0
        chk = ([(v0, vs[0])] if i0 not in MECH else []) + [(arrive(vs[k], TL[i0 + k + 1] - TL[i0 + k]), vs[k + 1]) for k in range(len(free))]
        for vin, vo in chk:
            e, n = bounce_e(vin, vo)
            if not (EMIN <= e <= EMAX) or tilt(n) > 35: ok = False; break
            cost += (0.55 - e) ** 2 * 0.5
        if not ok: continue
        if i0 in MECH and not (0.5 <= vs[0][2] <= 9.5): continue
        if not keys_clear(pts, vs, [TL[i0 + k] for k in range(len(pts))]): continue
        for k, i in enumerate(free): cost += 0.4 * (zc[k] - zpitch(PL[i])) ** 2
        if best is None or cost < best[0]: best = (cost, pts, vs)
    assert best is not None, ('no plan', i0, i1)
    return best
M.show = -1.0
M.rest(-2.0, T_S0, fixed[0])
PTS = {0: fixed[0]}; VIN = {0: np.zeros(3)}
fx = sorted(fixed)
for a, c in zip(fx, fx[1:]):
    cost, pts, vs = plan_group(a, c, PTS[a], VIN[a])
    for k in range(len(pts) - 1):
        i = a + k; j = a + k + 1; dt = TL[j] - TL[i]
        mech = i in MECH
        kind = ('ss0' if i == 0 else 'ss') if mech else 'key'
        add_hit(M, TL[i], pts[k], VIN[i], vs[k], kind, PL[i], mech=mech)
        M.fly(TL[i], TL[j], pts[k], vs[k])
        PTS[j] = pts[k + 1]; VIN[j] = arrive(vs[k], dt)
add_hit(M, TL[-1], LAMP, VIN[NL - 1], np.zeros(3), 'lamp', PL[-1], mech=True)
M.rest(TL[-1], 99, LAMP)
SFX.append(['lamp', float(TL[-1])])
EV['seesaws'] = [dict(t=float(TL[i]), x=float(fixed[i][0]), launch=M.hits[i]['vout'], vin=M.hits[i]['vin'], silent=(i == 0))
                 for i in sorted(MECH)]


# ------------------------------------------------------------------ checks
def check():
    info = {}
    hs = [(n, h) for n, b in B.items() for h in b.hits]
    free = [(n, h) for n, h in hs if not h['mech'] and np.linalg.norm(h['vout']) > 0]
    info['free_bounces'] = len(free)
    info['e_range'] = (round(min(h['e'] for n, h in free), 3), round(max(h['e'] for n, h in free), 3))
    info['speed_out_over_in_max(free)'] = round(max(np.linalg.norm(h['vout']) / np.linalg.norm(h['vin']) for n, h in free), 3)
    info['tilt_max_deg'] = round(max(tilt(h['n']) for n, h in free), 1)
    info['launch_vz_range'] = (round(min(h['vout'][2] for n, h in hs if h['mech'] and h['kind'] in ('ss', 'tray')), 2),
                               round(max(h['vout'][2] for n, h in hs if h['mech'] and h['kind'] in ('ss', 'tray')), 2))
    info['mel_z_range'] = (round(min(h['p'][2] for h in M.hits), 2), round(max(h['p'][2] for h in M.hits), 2))
    zmax = 0
    for t in np.arange(19.8, 52.8, 0.02): zmax = max(zmax, M.at(t)[2])
    info['mel_apex_max'] = round(zmax, 2)
    info['trio_e'] = round(B['trioA'].hits[5]['e'], 4)
    info['notes_lead'] = (len([h for h in M.hits if h['pitch'] is not None]), len(LEAD)); info['n_bass_balls'] = len([b for b in B.values() if b.role == 'bass'])
    return info


if __name__ == '__main__':
    for k, v in check().items(): print(k, v)
    out = dict(G=G, VX=VX, R=R, Y=Y, balls={n: dict(role=b.role, R=b.R, seg=b.seg, hits=b.hits, show=b.show) for n, b in B.items()},
               ev=EV, sfx=SFX, piano=dict(lid_front=LID_FRONT, key_back=KEY_BACK, white_front=WHITE_FRONT, black_front=BLACK_FRONT,
                                          z_white=Z_WHITE, z_black=Z_BLACK, wk=WK, y_a0=Y_A0))
    json.dump(out, open(f'{HERE}/layout.json', 'w'))
    for i, h in enumerate(M.hits):
        print(i, round(h['t'], 2), pn(h['pitch']) if h['pitch'] else '-', h['kind'], np.round(h['p'], 2), 'e', round(h['e'], 2), 'vout', np.round(h['vout'], 2))
