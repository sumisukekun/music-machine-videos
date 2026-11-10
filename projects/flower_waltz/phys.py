"""Shared motion model (layout.py, scene.py, verify.py).
Segments: rest (fixed point), fly (gravity parabola, one constant G), acc (rolling down a straight gutter:
constant acceleration 5/7 G sin(slope) along it), lin (carried by a mechanism at constant velocity:
a ball sitting in the counterweight bucket while a ratchet notch lowers it)."""
import numpy as np
G = 7.0
R = 0.15


def seg_at(g, t):
    p = np.array(g['p'], float)
    if g['k'] == 'rest': return p
    u = t - g['t0']; v = np.array(g['v'], float)
    if g['k'] == 'lin': return p + v * u
    if g['k'] == 'acc': return p + v * u + 0.5 * np.array(g['a'], float) * u * u
    return p + v * u + np.array([0, 0, -0.5 * G * u * u])


def seg_vel(g, t):
    if g['k'] == 'rest': return np.zeros(3)
    v = np.array(g['v'], float)
    if g['k'] == 'lin': return v
    if g['k'] == 'acc': return v + np.array(g['a'], float) * (t - g['t0'])
    return v + np.array([0, 0, -G * (t - g['t0'])])


def at(segs, t):
    for g in segs:
        if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9: return seg_at(g, t)
    if t < segs[0]['t0']: return seg_at(segs[0], segs[0]['t0'])
    return seg_at(segs[-1], segs[-1]['t1'])


def vel(segs, t):
    for g in segs:
        if g['t0'] - 1e-9 <= t <= g['t1'] + 1e-9: return seg_vel(g, t)
    return np.zeros(3)


# struck parts: half width (along the key's x axis), half depth (y), half thickness. Key centre = contact - n (R + th)
KEY = {'bud': (0.1, 0.1, 0.03), 'petal': (0.08, 0.08, 0.02), 'cup': (0.11, 0.11, 0.025), 'leaf': (0.1, 0.13, 0.02),
       'ice': (0.11, 0.11, 0.035), 'bell': (0.1, 0.1, 0.035), 'sunflower': (0.5, 0.5, 0.05)}


def key_frame(n):
    n = np.array(n, float); n /= np.linalg.norm(n)
    xa = np.cross([0, 1, 0], n); xa /= np.linalg.norm(xa); ya = np.cross(n, xa)
    return np.stack([xa, ya, n], 1)
