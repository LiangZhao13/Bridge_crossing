"""Speed-ups that are numerically identical to bridge_sim.core (verified bit-for-bit
in tests/test_regression.py). They only remove Python/NumPy call overhead."""
import math
import numpy as np
from . import core
from .core import KPOS, KNEG, ARM, FMIN, FMAX


def _rpm_scalar(F):
    F = min(max(float(F), FMIN), FMAX)
    if F >= 0:
        return math.sqrt(abs(F)/KPOS)
    return -math.sqrt(abs(F)/KNEG)


def derivative(self, z, F, current, current_dot=None):
    u, v, r, _, _, psi, nl, nr = z
    co, si = np.cos(psi), np.sin(psi)
    uc = co*current[0] + si*current[1]
    vc = -si*current[0] + co*current[1]
    ur, vr = u - uc, v - vc
    p1, p2 = self.A[0, 0]*ur, self.A[1, 1]*vr
    coriolis = np.array([-self.m*r*vr - self.m*r*self.xg*r - p2*r,
                         self.m*r*ur + p1*r,
                         self.m*r*self.xg*ur + p2*ur - p1*vr])
    dragv = vr + self.strips*r
    strips = -self.drag_k*dragv*np.abs(dragv)
    drag = np.array([0., strips.sum(), self.strips @ strips])
    damping = -self.D*np.array([ur, vr, r])
    damping[2] *= 1 + 10*abs(r)
    if self.mismatch:
        damping *= np.array([1.015, 1.025, 1.02])
    n = np.array([nl, nr])
    thrust = np.where(n >= 0, KPOS, KNEG)*n*np.abs(n)
    tau = np.array([thrust.sum(), 0., ARM*(thrust[0] - thrust[1])])
    dc = np.array([r*vc, -r*uc, 0.])
    if current_dot is not None:
        dc[:2] += np.array([co*current_dot[0] + si*current_dot[1], -si*current_dot[0] + co*current_dot[1]])
    dv = dc + self.invM @ (tau + damping + drag - coriolis)
    out = np.empty(8)
    out[:3] = dv
    out[3] = co*u - si*v
    out[4] = si*u + co*v
    out[5] = r
    out[6] = (_rpm_scalar(F[0]) - nl)/self.actuator_T
    out[7] = (_rpm_scalar(F[1]) - nr)/self.actuator_T
    return out


def install():
    core.OtterPlant.derivative = derivative
