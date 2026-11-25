"""Speed-limit time law for recorded demonstrations.

The demonstrations move faster than the drive allows (fastest joint above 34 deg/s in ~20 % of the
moving time, accelerations up to 300+ deg/s^2). `time_law` returns, for every recorded frame, the time
at which an arm that follows the SAME joint path but keeps --vlim / --alim would reach it. The path is
never sped up. Same forward-backward pass as dataset_gen_curobo.retime, on a 4 ms upsampling of the
10 Hz recording."""
import numpy as np
from scipy.interpolate import CubicSpline


def time_law(q, rec_dt=0.1, vlim=32.0, alim=140.0, dt=0.004):
    q = np.asarray(q, float)
    T = len(q)
    tf = np.arange(0.0, (T - 1) * rec_dt + 1e-9, dt)
    qf = CubicSpline(np.arange(T) * rec_dt, q, axis=0)(tf)
    vl, al = np.radians(vlim), np.radians(alim)
    q1 = np.gradient(qf, dt, axis=0)
    q2 = np.gradient(q1, dt, axis=0)
    aq1, aq2 = np.abs(q1).max(1), np.abs(q2).max(1)
    x = np.minimum(1.0, np.minimum((vl / np.maximum(aq1, 1e-9)) ** 2, 0.5 * al / np.maximum(aq2, 1e-9)))
    s1 = np.sign(q1)
    den = np.maximum(np.abs(q1), 1e-9)
    for i in range(len(x) - 1):
        u = np.min(np.maximum(al - s1[i] * q2[i] * x[i], 0.0) / den[i])
        x[i + 1] = min(x[i + 1], x[i] + 2 * dt * u)
    for i in range(len(x) - 1, 0, -1):
        u = np.min(np.maximum(al + s1[i] * q2[i] * x[i], 0.0) / den[i])
        x[i - 1] = min(x[i - 1], x[i] + 2 * dt * u)
    sd = np.sqrt(np.maximum(x, 1e-4))
    t_new = np.concatenate([[0.0], np.cumsum(2 * dt / (sd[1:] + sd[:-1]))])
    return np.interp(np.arange(T) * rec_dt, tf, t_new)          # new time of every recorded frame


def retime_path(q, vlim=32.0, alim=140.0, dt=0.004):
    """Slow a dt-sampled joint path down (never speed it up) to the limits; same path, resampled at dt."""
    q = np.asarray(q, float)
    if vlim <= 0 or len(q) < 4:
        return q
    vl, al = np.radians(vlim), np.radians(alim)
    q1 = np.gradient(q, dt, axis=0)
    q2 = np.gradient(q1, dt, axis=0)
    aq1, aq2 = np.abs(q1).max(1), np.abs(q2).max(1)
    x = np.minimum(1.0, np.minimum((vl / np.maximum(aq1, 1e-9)) ** 2, 0.5 * al / np.maximum(aq2, 1e-9)))
    if x.min() >= 1.0:
        return q
    s1 = np.sign(q1)
    den = np.maximum(np.abs(q1), 1e-9)
    for i in range(len(x) - 1):
        x[i + 1] = min(x[i + 1], x[i] + 2 * dt * np.min(np.maximum(al - s1[i] * q2[i] * x[i], 0.0) / den[i]))
    for i in range(len(x) - 1, 0, -1):
        x[i - 1] = min(x[i - 1], x[i] + 2 * dt * np.min(np.maximum(al + s1[i] * q2[i] * x[i], 0.0) / den[i]))
    sd = np.sqrt(np.maximum(x, 1e-4))
    t_new = np.concatenate([[0.0], np.cumsum(2 * dt / (sd[1:] + sd[:-1]))])
    idx = np.interp(np.arange(0.0, t_new[-1] + 1e-9, dt), t_new, np.arange(len(q)))
    return CubicSpline(np.arange(len(q)), q, axis=0)(idx)
