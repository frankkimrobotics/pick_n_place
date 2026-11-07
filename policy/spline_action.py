"""spline_action :: B-spline action parameterisation for the ZED/lingbot policies.

The network shape is UNCHANGED ([TA, ADIM]); only the MEANING of the 16 rows changes:
waypoints -> clamped-cubic B-spline control points. Fitting uses the same knots/basis as
policy/convert_litdata.py so this matches the earlier chunk policies.

ABPolicy (github.com/teee000/ABPolicy-code) idea: a chunk must start where the robot
actually IS and with the velocity it actually HAS. In this RELATIVE-pose action space
position continuity is free (the chunk is expressed about the current pose, so c0 = 0);
velocity continuity is one control point: c1 = v0 * span / degree. `ab_refit` pins c0/c1
and least-squares refits the remaining control points to the predicted curve, so the
executed trajectory is C1-continuous across re-plans instead of kinking at the splice.
"""
import numpy as np

HORIZON = 16
N_CTRL = 16
DEGREE = 3
DT = 0.1
T_SPAN = (HORIZON - 1) * DT
KNOTS = np.array([0.0] * (DEGREE + 1)
                 + [T_SPAN * j / (N_CTRL - DEGREE) for j in range(1, N_CTRL - DEGREE)]
                 + [T_SPAN] * (DEGREE + 1))


def bspline_basis(s, deriv=0):
    """Cox-de Boor basis row (N_CTRL,) at parameter s; deriv=1 gives d/ds."""
    if deriv == 1:
        h = 1e-4
        a = bspline_basis(min(s + h, T_SPAN - 1e-9))
        b = bspline_basis(max(s - h, 0.0))
        return (a - b) / (2 * h)
    s = min(max(s, 0.0), T_SPAN - 1e-9)
    nk = len(KNOTS)
    b = np.zeros((nk - 1,))
    for i in range(nk - 1):
        b[i] = 1.0 if KNOTS[i] <= s < KNOTS[i + 1] else 0.0
    for p in range(1, DEGREE + 1):
        nb = np.zeros(nk - 1 - p)
        for i in range(len(nb)):
            left = right = 0.0
            if KNOTS[i + p] > KNOTS[i]:
                left = (s - KNOTS[i]) / (KNOTS[i + p] - KNOTS[i]) * b[i]
            if KNOTS[i + p + 1] > KNOTS[i + 1]:
                right = (KNOTS[i + p + 1] - s) / (KNOTS[i + p + 1] - KNOTS[i + 1]) * b[i + 1]
            nb[i] = left + right
        b = nb
    return b


FIT_S = np.array([T_SPAN * k / (HORIZON - 1) for k in range(HORIZON)])
BASIS = np.stack([bspline_basis(s) for s in FIT_S])             # (16,16) eval matrix
_D2 = np.diff(np.eye(N_CTRL), 2, axis=0)                        # smoothness prior
LAMBDA = 1e-3
FIT_MAT = np.linalg.solve(BASIS.T @ BASIS + LAMBDA * _D2.T @ _D2, BASIS.T)   # (16,16)


def waypoints_to_ctrl(A):
    """A (16, D) sampled waypoints -> (16, D) control points."""
    return FIT_MAT @ A


def ctrl_to_waypoints(C):
    """(16, D) control points -> (16, D) sampled waypoints."""
    return BASIS @ C


def ab_refit(C, v0=None, n_free=None, c0=None):
    """ABPolicy continuity: pin c0 (default 0 = the obs-time pose in the relative frame;
    pass the ACTUAL handover state for latency matching) and c1 (current velocity),
    then least-squares refit the rest so the curve still tracks the prediction."""
    C = np.asarray(C, float).copy()
    target = BASIS @ C                      # the curve the policy asked for
    C[0] = 0.0 if c0 is None else np.asarray(c0, float)   # curve starts at the robot's real state
    if v0 is not None:
        # dB/ds at s=0 is non-zero only for c0,c1 -> c1 sets the initial velocity
        C[1] = np.asarray(v0, float) * (T_SPAN / (N_CTRL - DEGREE)) / DEGREE
    k = 2
    free = N_CTRL - k if n_free is None else min(n_free, N_CTRL - k)
    Bf = BASIS[:, k:k + free]
    resid = target - BASIS[:, :k] @ C[:k]
    if free < N_CTRL - k:                   # keep the tail as predicted
        resid = resid - BASIS[:, k + free:] @ C[k + free:]
    C[k:k + free] = np.linalg.lstsq(Bf, resid, rcond=None)[0]
    return C
