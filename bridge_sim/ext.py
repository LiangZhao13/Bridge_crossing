"""Experiment extensions around the frozen method in bridge_sim/core.py.

Nothing here changes the method. It adds
  * a closed loop `run_case_ext` that is bit-for-bit identical to core.run_case with
    default arguments (checked in tests/test_regression.py) but lets an experiment
    swap the gyro bias, the navigator (model-aided / pure INS) and the error-bound
    law used by the safety filter (deterministic box / Gaussian 3-sigma / INS box);
  * random scenarios (entry state, currents, sensor seed) inside the declared bounds;
  * a hindsight "can this bridge be passed" label;
  * per-run summaries used by the notebooks.
"""
import math
from dataclasses import dataclass, replace, asdict, field

import numpy as np

from . import core, fast
from .core import (Config, TruthSpec, bridge_case, cfg as BASE_CFG, OtterPlant, CurrentField,
                   Sensors, Navigator, WaypointILOS, PredictiveSafetyFilter, ErrorBound,
                   allocate, force_to_rpm, FMIN, propagate, wrap, earth_velocity, body_velocity,
                   tube_margins, hull_clearance, retreat_command, jet_shape, thrust_metrics)

fast.install()


# --------------------------------------------------------------------------- sensors
@dataclass(frozen=True)
class ImuSpec:
    """Accelerometer used only by the pure-INS baseline.
    accel_bias_bound: declared |bias| per body axis (m/s^2); the truth draws inside it.
    accel_noise_bound: declared |noise| per axis (m/s^2), uniform.
    accel_max: declared bound on |horizontal acceleration| (couples heading error)."""
    accel_bias_bound: float = 0.0196      # 2 mg
    accel_noise_bound: float = 0.002
    accel_max: float = 0.8               # max |a| seen after the gate in the main run is 0.73
    velocity0: float = 0.02               # initial velocity error bound (m/s)


class SensorsExt(Sensors):
    """core.Sensors with a configurable gyro bias and an optional accelerometer
    (delta-velocity output). With gyro_bias=0.00055 and imu=None the random
    streams and outputs are identical to core.Sensors."""
    def __init__(self, cfg, gnss_always=False, gyro_bias=0.00055, imu=None, accel_bias=(0., 0.)):
        super().__init__(cfg, gnss_always)
        self.gyro_bias = gyro_bias
        self.imu = imu
        self.accel_bias = np.asarray(accel_bias, dtype=float)
        if imu is not None:
            self.accel_rng = np.random.default_rng(np.random.SeedSequence(cfg.seed).spawn(4)[3])
        self.prev_velocity = None
        self.prev_heading = None

    def sample(self, truth, gnss_tick):
        c = self.cfg
        packet = {'gyro': truth[2] + self.gyro_bias + self.gyro_rng.uniform(-.00025, .00025),
                  'rpm': truth[6:8] + self.rpm_rng.uniform(-.04, .04, 2),
                  'fix': None, 'quality': None}
        if self.imu is not None:
            v = earth_velocity(truth)
            if self.prev_velocity is None:
                dv_body = np.zeros(2)
            else:
                heading = self.prev_heading + 0.5*wrap(truth[5] - self.prev_heading)
                dv_body = body_velocity(v - self.prev_velocity, heading)
                dv_body = dv_body + c.dt*(self.accel_bias + self.accel_rng.uniform(
                    -self.imu.accel_noise_bound, self.imu.accel_noise_bound, 2))
            packet['dv'] = dv_body
            self.prev_velocity, self.prev_heading = v, truth[5]
        if not gnss_tick:
            return packet
        noise = self.gnss_rng.uniform(-1., 1., 5)
        north = truth[3]
        if not self.gnss_always and c.blackout_start <= north <= c.blackout_end:
            packet['quality'] = 'outage'
        elif not self.gnss_always and c.degraded_start <= north <= c.degraded_end:
            packet['quality'] = 'degraded'
        else:
            packet['quality'] = 'good'
            packet['fix'] = (truth[3:5] + .025*noise[:2],
                             earth_velocity(truth) + .008*noise[2:4],
                             wrap(truth[5] + .003*noise[4]))
        return packet


# --------------------------------------------------------------------------- pure INS
class InsNavigator(Navigator):
    """Pure inertial dead reckoning: gyro heading + integrated accelerometer delta-velocity.
    No ship model, no propeller rpm and no current map in the position solution.
    GNSS fusion uses the same gains as the model-aided navigator."""
    def __init__(self, cfg, field):
        super().__init__(cfg, field)
        self.v_earth = earth_velocity(self.z)

    def update(self, packet, t, previous_force, initial=False):
        c = self.cfg
        heading = self.z[5]
        if not initial:
            mean_gyro = packet['gyro'] if self.previous_gyro is None else .5*(self.previous_gyro + packet['gyro'])
            mid = heading + .5*c.dt*mean_gyro
            v_new = self.v_earth + earth_velocity(np.r_[packet['dv'], 0., 0., 0., mid])
            self.z[3:5] += c.dt*.5*(self.v_earth + v_new)
            self.v_earth = v_new
            self.z[5] = wrap(heading + c.dt*mean_gyro)
        self.previous_gyro = packet['gyro']
        self.z[2] = packet['gyro']
        self.z[6:8] = packet['rpm']
        self.z[:2] = body_velocity(self.v_earth, self.z[5])
        self.predicted_z = self.z.copy()
        self.fusion_gain = None
        if packet['quality'] == 'good':
            position, velocity, gheading = packet['fix']
            gain = 1.0 if self.last_fix is None else (.12 if t - self.last_fix > 1. else .70)
            self.z[3:5] += gain*(position - self.z[3:5])
            self.z[5] = wrap(self.z[5] + gain*wrap(gheading - self.z[5]))
            self.v_earth = self.v_earth + gain*(velocity - self.v_earth)
            self.z[:2] = body_velocity(self.v_earth, self.z[5])
            self.last_fix = t
            self.fusion_gain = gain
        return self.z.copy()


class InsBound(ErrorBound):
    """Deterministic box for the pure-INS navigator (current-independent):
        d(vel)/dt <= accel bias (rotated) + accel noise + accel_max * heading error
        d(pos)/dt <= vel
    fused as b+ = (1-K) b- + K * GNSS noise bound for position, velocity, heading."""
    def __init__(self, c, speed=None, imu=ImuSpec()):
        self.imu = imu
        super().__init__(c, speed)

    def reset(self):
        super().reset()
        self.vel = self.imu.velocity0

    def copy(self):
        b = InsBound(self.c, self.speed, self.imu)
        b.north, b.east, b.age, b.heading, b.vel = self.north, self.east, self.age, self.heading, self.vel
        return b

    def fuse(self, gain):
        super().fuse(gain)
        self.vel = (1-gain)*self.vel + gain*.008

    def step(self, north_est, dt, heading_est=0.):
        imu = self.imu
        rot = min(math.sqrt(2.), abs(math.cos(heading_est)) + abs(math.sin(heading_est)) + self.heading)
        rate = imu.accel_bias_bound*rot + imu.accel_noise_bound + imu.accel_max*self.heading
        grow = dt*(self.vel + .5*dt*rate)
        self.north += grow
        self.east += grow
        self.vel += dt*rate
        self.age += dt
        self.heading += self.c.gyro_bound*dt


# --------------------------------------------------------------------------- EKF 3-sigma
class GaussianBound(ErrorBound):
    """EKF-style covariance of the model-aided navigator's error, used as a 3-sigma box.

    Error state x = [dN, dE, dpsi, b_u, b_v, b_cN, b_cE, b_jet, b_g]:
      dN'   = cos(psi) b_u - sin(psi) b_v - U sin(psi) dpsi + b_cN
      dE'   = sin(psi) b_u + cos(psi) b_v + U cos(psi) dpsi + b_cE + shape(N) b_jet + grad(N) dN
      dpsi' = b_g ;  all b_* are random constants.
    Each source gets the SAME declared magnitude as the deterministic box, read with the
    common datasheet convention "declared bound = 3 sigma". GNSS fusion applies the
    navigator's actual fixed gain in Joseph form, so the covariance is consistent with the
    navigator that is really running. The safety filter then uses 3*sigma in place of the
    worst-case box. Differences to the box: sources add in quadrature (RSS) instead of
    linearly, the jet shape is taken at the estimate instead of its max over the North
    interval, and the gradient term uses the local gradient."""
    NX = 9

    def __init__(self, c, speed=None):
        super().__init__(c, speed)

    def reset(self):
        c = self.c
        U = self.speed
        # position/heading errors in metres/radians; the six bias states are normalised
        # (unit variance) and scaled by their 1-sigma magnitude inside step().
        self.P = np.diag(np.r_[(np.array([c.delta0, c.delta0, c.heading0])/3.)**2, np.ones(6)])
        self.age = 0.

    # the filter and run loop read these as plain attributes
    @property
    def north(self):
        return 3.*math.sqrt(self.P[0, 0])

    @north.setter
    def north(self, value):
        pass

    @property
    def east(self):
        return 3.*math.sqrt(self.P[1, 1])

    @east.setter
    def east(self, value):
        pass

    def heading_error(self):
        return 3.*math.sqrt(self.P[2, 2])

    def copy(self):
        b = GaussianBound.__new__(GaussianBound)
        b.c, b.speed, b.P, b.age = self.c, self.speed, self.P.copy(), self.age
        return b

    def fuse(self, gain):
        A = np.eye(self.NX)
        A[0, 0] = A[1, 1] = A[2, 2] = 1 - gain
        R = np.zeros(self.NX)
        R[0] = R[1] = (gain*.025/3)**2
        R[2] = (gain*.003/3)**2
        self.P = A @ self.P @ A.T + np.diag(R)
        self.age = 0.

    def step(self, north_est, dt, heading_est=0.):
        c = self.c
        U = self.speed
        co, si = math.cos(heading_est), math.sin(heading_est)
        shape, grad = jet_shape(north_est, c)
        su = (c.speed_error_floor + c.speed_error_ratio*U)/3.     # 1-sigma = declared bound / 3
        sv = c.speed_error_lateral/3.
        scn, sce = c.current_north_error_bound/3., c.current_east_error_bound/3.
        sj, sg = c.jet_amplitude_error_bound/3., c.gyro_bound/3.
        Phi = np.eye(self.NX)
        Phi[0, 2] = -U*si*dt; Phi[0, 3] = su*co*dt; Phi[0, 4] = -sv*si*dt; Phi[0, 5] = scn*dt
        Phi[1, 0] = c.jet_amplitude*grad*dt
        Phi[1, 2] = U*co*dt; Phi[1, 3] = su*si*dt; Phi[1, 4] = sv*co*dt; Phi[1, 6] = sce*dt
        Phi[1, 7] = sj*shape*dt
        Phi[2, 8] = sg*dt
        self.P = Phi @ self.P @ Phi.T
        self.age += dt


# --------------------------------------------------------------------------- closed loop
class TightBound(ErrorBound):
    """Tightened deterministic box (variant "tight"). Same terms as core.ErrorBound, still a
    worst-case bound under A1-A2; one term is evaluated more tightly:
      jet-gradient coupling: core uses the GLOBAL max |d(jet)/dN| everywhere; here the max
      over the current North uncertainty interval [N - bN, N + bN] (outside the jet it is 0).
    The water-speed margin in the heading term is a Config value (water_speed_margin) and is
    set by the variant, not here."""
    def copy(self):
        b = TightBound(self.c, self.speed)
        b.north, b.east, b.age, b.heading = self.north, self.east, self.age, self.heading
        return b

    def local_gradient(self, lo, hi):
        c = self.c
        a, b = (lo - c.jet_center)/c.jet_half_length, (hi - c.jet_center)/c.jet_half_length
        a, b = max(a, -1.), min(b, 1.)
        if a >= b:
            return 0.
        k = abs(c.jet_amplitude)*math.pi/(2*c.jet_half_length)
        if a <= -.5 <= b or a <= .5 <= b:
            return k
        return k*max(abs(math.sin(math.pi*a)), abs(math.sin(math.pi*b)))

    def step(self, north_est, dt, heading_est=0.):
        from .core import jet_shape_max
        c = self.c
        surge_err = c.speed_error_floor + c.speed_error_ratio*self.speed
        heading_part = (self.speed + c.water_speed_margin)*self.heading_error()
        s_max = min(1., abs(math.sin(heading_est)) + self.heading_error())
        d_north = surge_err + c.speed_error_lateral*s_max + heading_part + c.current_north_error_bound
        lo, hi = north_est - self.north, north_est + self.north
        jet = c.jet_amplitude_error_bound*jet_shape_max(lo, hi, c)
        d_east = (surge_err*s_max + c.speed_error_lateral + heading_part + c.current_east_error_bound
                  + jet + self.local_gradient(lo, hi)*self.north)
        self.north += dt*d_north
        self.east += dt*d_east
        self.age += dt
        self.heading += c.gyro_bound*dt


# Experiment variants: margins removed and/or tightened bound. Applied to ALL methods alike.
#   margins0: map_bound = sample_guard = safety_margin = 0, and the truth chart error
#             (wall_shift) is set to 0 so that A1-A2 still hold without a chart margin.
#   tight:    TightBound replaces ErrorBound for the proposed filter; water-speed margin
#             0.20 -> 0.05 m/s (max water speed minus transit speed seen in the worst
#             worlds is 0.012 m/s; checked again by coverage in notebook 04).
VARIANTS = {
    'base':          dict(cfg={}, tight=False, zero_wall=False),
    'margins0':      dict(cfg=dict(map_bound=0., sample_guard=0., safety_margin=0.), tight=False, zero_wall=True),
    'tight':         dict(cfg=dict(water_speed_margin=0.05), tight=True, zero_wall=False),
    'tight_margins0': dict(cfg=dict(map_bound=0., sample_guard=0., safety_margin=0., water_speed_margin=0.05),
                           tight=True, zero_wall=True),
}


def make_bound(kind, c, imu=None):
    if kind == 'box':
        return ErrorBound(c)
    if kind == 'tight':
        return TightBound(c)
    if kind == 'ekf':
        return GaussianBound(c)
    if kind == 'ins':
        return InsBound(c, imu=imu or ImuSpec())
    raise ValueError(kind)


def run_case_ext(c, truth_spec, strategy="ilos", gnss_always=False, fixed_gate_speed=None,
                 bypass_filter=False, gyro_bias=0.00055, nav="model", bound_kind="box",
                 imu=None, accel_bias=(0., 0.), stop_after_gate=False):
    """core.run_case with swappable gyro bias, navigator and bound law.
    nav: "model" (core.Navigator) or "ins" (InsNavigator, needs imu)
    bound_kind: "box" (core.ErrorBound), "ekf" (GaussianBound 3-sigma), "ins" (InsBound)."""
    steps_per_control = int(round(c.control_dt/c.dt))
    wp = c.waypoints()
    a = math.radians(c.approach_heading_deg)
    start = wp[0] + truth_spec.initial_cross_track*np.array([-math.sin(a), math.cos(a)])
    truth = np.r_[c.speed, 0., 0., start, a + truth_spec.initial_heading_error, force_to_rpm([46., 46.])]
    plant = OtterPlant(c, mismatch=True)
    true_current = CurrentField(c, truth=truth_spec)
    mapped_current = CurrentField(c)
    if nav == "ins":
        imu = imu or ImuSpec()
        sensors = SensorsExt(c, gnss_always, gyro_bias, imu=imu, accel_bias=accel_bias)
        nav_ = InsNavigator(c, mapped_current)
    else:
        sensors = SensorsExt(c, gnss_always, gyro_bias)
        nav_ = Navigator(c, mapped_current)
    nav = nav_
    guide = WaypointILOS(c, mapped_current)
    guard = PredictiveSafetyFilter(c, mapped_current)
    bound = make_bound(bound_kind, c, imu)
    force, quality, phase, mode, margin = allocate(2*46., 0.), 'good', 'approach', 'ILOS', np.nan
    ev = dict(impact_time=np.nan, completed=False, gate_time=np.nan, admitted=None,
              entry_margin=np.nan, transit_speed=c.speed, speed_table={}, retreated=False, failure='')
    rows, settle_ticks = [], 0

    for i in range(round(c.duration/c.dt) + 1):
        t = i*c.dt
        tick = i % steps_per_control == 0
        packet = sensors.sample(truth, gnss_tick=tick)
        if packet['quality'] is not None:
            quality = packet['quality']
        before = nav.last_fix
        z = nav.update(packet, t, force, initial=(i == 0))
        if i > 0:
            bound.step(nav.predicted_z[3], c.dt, nav.predicted_z[5])
        if nav.last_fix != before:
            bound.fuse(nav.fusion_gain)

        if tick:
            guard.previous_force = force.copy()
            margin = np.nan
            if (strategy == "filtered" or bypass_filter) and phase == 'approach' and z[3] >= c.gate_north:
                ev['gate_time'] = t
                if bypass_filter:
                    speed_g = c.speed if fixed_gate_speed is None else fixed_gate_speed
                    ev['transit_speed'] = speed_g
                    guide.speed = speed_g
                    bound.speed = speed_g
                    phase = 'unfiltered_transit'
                else:
                    if fixed_gate_speed is None:
                        ok, margin_g, speed_g, table = guard.admit(z, bound, t, nav.last_fix)
                    else:
                        guard.deadline = nav.last_fix + c.deadline_seconds
                        guard.speed = fixed_gate_speed
                        margin_g, progress = guard.rollout(z, bound, t)
                        ok = bool(progress and margin_g >= 0.)
                        speed_g = fixed_gate_speed if ok else None
                        table = {fixed_gate_speed: margin_g if progress else -np.inf}
                    ev.update(admitted=ok, entry_margin=margin_g, speed_table=table)
                    if ok:
                        ev['transit_speed'] = speed_g
                        guide.speed = speed_g
                        bound.speed = speed_g
                    phase = 'transit' if ok else 'abort'
                    if stop_after_gate:
                        break

            if phase == 'transit':
                _, exit_slack = tube_margins(z, bound, c)
                if exit_slack >= 0.:
                    phase = 'exit'
            if phase == 'abort':
                if earth_velocity(z)[0] > .08 and not ev['retreated']:
                    force, mode = allocate(2*FMIN, 0.), 'brake'
                else:
                    ev['retreated'] = True
                    force, mode = retreat_command(z, t, c, mapped_current), 'retreat'
                    settled = abs(z[3] - c.retreat_north) <= .25 and abs(earth_velocity(z)[0]) <= .12
                    settle_ticks = settle_ticks + 1 if settled else 0
            else:
                nominal, _ = guide.command(z, t)
                force, mode = nominal, 'ILOS'
                if phase == 'transit' and not bypass_filter:
                    selected, mode, margin = guard.apply(z, bound, t, nominal)
                    if selected is None:
                        ev['failure'] = 'no verified action'
                        break
                    force = selected
                elif phase == 'exit' and c.smooth_filter and not bypass_filter:
                    force = guard.limit_force(force, guard.previous_force)

        clearance = hull_clearance(truth, c, truth_spec.wall_shift)
        rows.append(dict(t=t, truth=truth.copy(), estimate=z.copy(), F=force.copy(), quality=quality,
                         clearance=clearance, mode=mode, tick=tick, margin=margin,
                         bound=np.array([bound.north, bound.east]), heading_bound=bound.heading_error()))
        if np.isfinite(clearance) and clearance <= 0.:
            ev['impact_time'] = t
            break
        if truth[3] - c.length/2 > c.degraded_end + 3.:
            ev['completed'] = True
            break
        if phase == 'abort' and settle_ticks*c.control_dt >= 1.0:
            break
        truth = propagate(plant, true_current, truth, force, t, c.dt)

    data = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    data.update(ev)
    data["filter_trace"] = guard.trace
    return data


# --------------------------------------------------------------------------- scenarios
@dataclass(frozen=True)
class Scenario:
    """One random world. Everything stays inside the declared bounds (A1-A2) unless
    a field says otherwise."""
    seed: int
    length: float = 19.0
    width: float = 8.0
    # mapped (known) currents
    mapped_current_north: float = 0.025
    mapped_current_east: float = 0.090
    jet_amplitude: float = 0.25
    # unknown truth
    jet_amplitude_error: float = 0.20
    current_north_bias: float = 0.005
    current_east_bias: float = 0.008
    tide_amplitude: float = 0.004
    tide_frequency: float = 0.050
    wall_shift: float = 0.04
    initial_cross_track: float = 0.30
    initial_heading_error: float = 0.0
    gyro_bias: float = 0.00055
    accel_bias_n: float = 0.0     # used only by the INS navigator (fraction of declared bound)
    accel_bias_e: float = 0.0
    variant: str = 'base'         # key of VARIANTS


DEFAULT_SCENARIO = Scenario(seed=42)


def sample_scenario(rng, seed, length=19.0, width=8.0):
    """Random entry state and currents, all inside the declared error bounds:
      truth jet error |.| <= 0.20 (< 0.22), East bias + tide <= 0.014 (< 0.015),
      North bias <= 0.008 (< 0.010), gyro |bias| + 0.00025 <= 0.0008, wall shift <= 0.06 (< 0.08)."""
    u = rng.uniform
    return Scenario(seed=int(seed), length=float(length), width=float(width),
                    mapped_current_north=u(0.0, 0.05), mapped_current_east=u(0.03, 0.12),
                    jet_amplitude=u(0.15, 0.35),
                    jet_amplitude_error=u(-0.20, 0.20),
                    current_north_bias=u(-0.008, 0.008), current_east_bias=u(-0.010, 0.010),
                    tide_amplitude=u(0.0, 0.004), tide_frequency=u(0.02, 0.08),
                    wall_shift=u(-0.06, 0.06),
                    initial_cross_track=u(-0.5, 0.5), initial_heading_error=u(-0.10, 0.10),
                    gyro_bias=u(-0.00055, 0.00055),
                    accel_bias_n=u(-1., 1.), accel_bias_e=u(-1., 1.))


def make_case(s, base=BASE_CFG):
    v = VARIANTS[getattr(s, 'variant', 'base')]
    base = replace(base, **v['cfg'])
    c = replace(base, seed=s.seed, mapped_current_north=s.mapped_current_north,
                mapped_current_east=s.mapped_current_east, jet_amplitude=s.jet_amplitude)
    c = bridge_case(c, s.length, s.width)
    # simulation horizon only (not a method parameter): long bridges need more than 90 s
    c = replace(c, duration=max(base.duration, c.deadline_seconds + 30.0))
    ts = TruthSpec(jet_amplitude_error=s.jet_amplitude_error, current_north_bias=s.current_north_bias,
                   current_east_bias=s.current_east_bias, tide_amplitude=s.tide_amplitude,
                   tide_frequency=s.tide_frequency, wall_shift=0. if v['zero_wall'] else s.wall_shift,
                   initial_cross_track=s.initial_cross_track, initial_heading_error=s.initial_heading_error)
    return c, ts


# --------------------------------------------------------------------------- summaries
def summarize(d, c, prefix=''):
    clear = d['clearance']
    finite = np.isfinite(clear)
    err = np.abs(d['truth'][:, 3:5] - d['estimate'][:, 3:5])
    b = d['bound']
    ratio = err/np.maximum(b, 1e-12)
    blind = d['quality'] != 'good'
    hd_err = np.abs((d['truth'][:, 5] - d['estimate'][:, 5] + np.pi) % (2*np.pi) - np.pi)
    out = dict(
        admitted=(None if d['admitted'] is None else bool(d['admitted'])),
        entry_margin=float(d['entry_margin']),
        transit_speed=float(d['transit_speed']),
        completed=bool(d['completed']),
        collided=bool(np.isfinite(d['impact_time'])),
        impact_time=float(d['impact_time']),
        cert_lost=(d['failure'] == 'no verified action'),
        retreated=bool(d['retreated']),
        entered_bridge=bool(finite.any()),
        min_clearance=float(np.nanmin(clear)) if finite.any() else np.nan,
        cov_ratio_N=float(ratio[:, 0].max()),
        cov_ratio_E=float(ratio[:, 1].max()),
        cov_ratio_heading=float((hd_err/np.maximum(d['heading_bound'], 1e-12)).max()),
        cov_ratio_E_blind=float(ratio[blind, 1].max()) if blind.any() else np.nan,
        max_err_E=float(err[:, 1].max()),
        max_bound_E_blind=float(b[blind, 1].max()) if blind.any() else np.nan,
        thrust_tv=thrust_metrics(d)['total_variation_N'] if np.isfinite(d['gate_time']) else np.nan,
        t_end=float(d['t'][-1]),
    )
    if d['speed_table']:
        for U, m in d['speed_table'].items():
            out[f'gate_margin_{U:.1f}'] = float(m)
    return {prefix + k: v for k, v in out.items()}


def oracle(c, ts, gyro_bias, speeds=None):
    """Hindsight label. Plain ILOS (no filter), same approach and same seed, switched at
    the gate to each candidate speed. Returns the best true minimum clearance and the
    per-speed clearances (negative/0 means wall contact)."""
    speeds = c.speed_candidates if speeds is None else speeds
    res = {}
    for U in speeds:
        d = run_case_ext(c, ts, "filtered", fixed_gate_speed=U, bypass_filter=True, gyro_bias=gyro_bias)
        ok = bool(d['completed']) and not np.isfinite(d['impact_time'])
        mc = float(np.nanmin(d['clearance'])) if np.isfinite(d['clearance']).any() else np.nan
        res[U] = mc if ok else min(0., mc if np.isfinite(mc) else 0.)
    best_U = max(res, key=res.get)
    out = dict(oracle_best_clearance=res[best_U], oracle_best_speed=best_U,
               oracle_feasible=res[best_U] > 0.)
    for U, m in res.items():
        out[f'oracle_clear_{U:.1f}'] = m
    return out


METHODS = {
    # name: kwargs for run_case_ext
    'filter':   dict(strategy='filtered'),                                  # proposed method
    'ilos07':   dict(strategy='ilos'),                                      # nominal ILOS 0.7 m/s
    'ekf3s':    dict(strategy='filtered', bound_kind='ekf'),                # EKF 3-sigma filter
    'ins':      dict(strategy='filtered', nav='ins', bound_kind='ins'),     # pure INS + filter, 2 mg
    'ins_1mg':  dict(strategy='filtered', nav='ins', bound_kind='ins', imu=ImuSpec(accel_bias_bound=0.0098)),
}


def evaluate(s, methods=('filter', 'ilos07', 'oracle'), imu=None):
    """Run the requested methods on one scenario. Returns a flat dict (one CSV row)."""
    c, ts = make_case(s)
    row = asdict(s)
    for m in methods:
        if m == 'oracle':
            row.update(oracle(c, ts, s.gyro_bias))
            continue
        kw = dict(METHODS[m])
        if VARIANTS[s.variant]['tight'] and kw.get('strategy') == 'filtered' and kw.get('bound_kind', 'box') == 'box':
            kw['bound_kind'] = 'tight'
        if kw.get('nav') == 'ins':
            imu_ = kw.pop('imu', None) or imu or ImuSpec()
            kw['imu'] = imu_
            kw['accel_bias'] = (s.accel_bias_n*imu_.accel_bias_bound, s.accel_bias_e*imu_.accel_bias_bound)
        d = run_case_ext(c, ts, gyro_bias=s.gyro_bias, **kw)
        row.update(summarize(d, c, prefix=m + '_'))
        if kw.get('nav') == 'ins':
            v = np.array([earth_velocity(z) for z in d['truth']])
            acc = np.linalg.norm(np.diff(v, axis=0)/c.dt, axis=1)
            row[m + '_max_true_accel'] = float(acc.max())
            # the declared accel_max is only needed outside the brake/retreat manoeuvre
            normal = ~np.isin(d['mode'][1:], ('brake', 'retreat'))
            row[m + '_max_true_accel_nominal'] = float(acc[normal].max()) if normal.any() else np.nan
    return row


def _evaluate_star(args):
    s, methods, imu = args
    try:
        return evaluate(s, methods, imu)
    except Exception as exc:          # keep the batch alive, record the failure
        row = asdict(s)
        row['error'] = repr(exc)
        return row


def run_batch(scenarios, methods, imu=None, workers=4, progress_every=10, log_path=None):
    """Parallel map over scenarios (process pool). Order of rows = order of scenarios."""
    import concurrent.futures as cf
    import time
    t0 = time.time()
    rows = [None]*len(scenarios)
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_evaluate_star, (s, methods, imu)): i for i, s in enumerate(scenarios)}
        for k, f in enumerate(cf.as_completed(futs), 1):
            rows[futs[f]] = f.result()
            if progress_every and k % progress_every == 0:
                msg = f"  {k}/{len(scenarios)} done, {time.time()-t0:.0f} s"
                print(msg, flush=True)
                if log_path:
                    with open(log_path, 'a') as fh:
                        fh.write(msg + '\n')
    return rows


def gate_check(s, bound_kind='box', nav='model', imu=None):
    """Admission decision only (stops right after the gate). Cheap; used for maps."""
    c, ts = make_case(s)
    if bound_kind == 'box' and VARIANTS[s.variant]['tight']:
        bound_kind = 'tight'
    kw = dict(strategy='filtered', bound_kind=bound_kind, nav=nav, gyro_bias=s.gyro_bias, stop_after_gate=True)
    if nav == 'ins':
        imu = imu or ImuSpec()
        kw.update(imu=imu, accel_bias=(s.accel_bias_n*imu.accel_bias_bound, s.accel_bias_e*imu.accel_bias_bound))
    d = run_case_ext(c, ts, **kw)
    return dict(admitted=bool(d['admitted']), entry_margin=float(d['entry_margin']),
                speed=float(d['transit_speed']) if d['admitted'] else np.nan)


def _call(args):
    func, kw = args
    try:
        return func(**kw)
    except Exception as exc:
        return dict(error=repr(exc))


def pmap(func, kwargs_list, workers=4):
    """Parallel func(**kw) over a list of keyword dicts (func must be importable)."""
    import concurrent.futures as cf
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_call, [(func, kw) for kw in kwargs_list]))


def gate_and_oracle(s, bound_kinds=('box', 'ekf'), with_oracle=True):
    """Gate decisions for several bound laws plus the hindsight label, for one scenario."""
    row = dict(length=s.length, width=s.width)
    for b in bound_kinds:
        g = gate_check(s, bound_kind=b)
        row.update({f'{b}_{k}': v for k, v in g.items()})
    if with_oracle:
        c, ts = make_case(s)
        row.update(oracle(c, ts, s.gyro_bias))
    return row


def ins_gate(s, accel_mg):
    g = gate_check(s, nav='ins', bound_kind='ins', imu=ImuSpec(accel_bias_bound=accel_mg*0.0098))
    return dict(length=s.length, width=s.width, accel_mg=accel_mg, **g)
