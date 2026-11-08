"""Shared motion evaluation for layout.json (scene.py and verify.py): parabolas, rests, lift rides, rolls."""
import json, math, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(f'{HERE}/layout.json'))
G = L['G']; R = L['R']; LIFT = L['lift']


def lift_z(t):
    t0, t1, z0, z1 = LIFT['t0'], LIFT['t1'], LIFT['z0'], LIFT['z1']
    if t <= t0: return z0
    if t >= t1: return z1
    A = 0.8; Ln = t1 - t0; v = (z1 - z0) / (Ln - A); u = t - t0
    if u < A: return z0 + 0.5 * v / A * u * u
    if u > Ln - A: w = Ln - u; return z1 - 0.5 * v / A * w * w
    return z0 + 0.5 * v * A + v * (u - A)


def lift_phi(t): return (lift_z(t) - LIFT['z0']) / (LIFT['z1'] - LIFT['z0']) * LIFT['rot']


def bell_r(p): return float(np.clip(0.22 * 2 ** (-(p - 60) / 24), 0.075, 0.42))


def seg_at(g, t):
    k = g['k']
    if k == 'rest': return np.array(g['p'])
    if k == 'fly':
        u = t - g['t0']; return np.array(g['p']) + np.array(g['v']) * u + np.array([0, 0, -0.5 * G * u * u])
    if k == 'roll':
        u = t - g['t0']; return np.array(g['p']) + np.array(g['d']) * 0.5 * g['a'] * u * u
    if k == 'ride':
        a = lift_phi(t); o = g['off']
        return np.array([o[0] * math.cos(a) - o[1] * math.sin(a), o[0] * math.sin(a) + o[1] * math.cos(a), lift_z(t) + o[2]])


class Path:
    def __init__(s, b):
        s.seg = b['seg']; s.t0 = np.array([g['t0'] for g in s.seg]); s.t1 = np.array([g['t1'] for g in s.seg])
    def at(s, t):
        if t < s.t0[0]: return seg_at(s.seg[0], s.t0[0])
        j = int(np.searchsorted(s.t0, t, side='right')) - 1
        j = max(0, min(j, len(s.seg) - 1))
        g = s.seg[j]
        if t > g['t1'] + 1e-9: return seg_at(g, g['t1'])
        return seg_at(g, t)


PATHS = {n: Path(b) for n, b in L['balls'].items()}
