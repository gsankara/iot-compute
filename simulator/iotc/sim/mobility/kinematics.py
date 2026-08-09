"""How a body actually moves.

Three platforms, and the differences between them are the whole point of the
flotation study:

``drone``
    Real thrust on all three axes.  Fast, responsive, isotropic, and expensive
    -- station-keeping costs watts continuously, and a browned-out drone falls.

``buoyant``
    A ship or submarine in a liquid.  The xy-plane is driven by thrusters and
    responds immediately, but z is controlled by taking on or expelling water,
    which is **rate limited** and therefore has tens of seconds of actuation
    lag.  Station-keeping is nearly free, because a neutrally buoyant body just
    stays put.

``tethered``
    Only a small optical head is immersed; the compute box, the store and the
    harvester sit outside the medium on a gantry, and the head hangs beneath on
    a wire or waveguide.  Position is commanded from outside, so all three axes
    are fast and precise and the node pays no motion energy at all -- but every
    head trails a fibre, and those fibres occlude the beams.

So on a drone the compiler may treat all three axes alike; on a tethered rig it
may too, and the binding constraint becomes occlusion instead.

On a float the axes are anisotropic, but *not* in the obvious direction, and it
is worth being precise because the naive reading points the wrong way.  Depth is
slow -- tens of seconds of pump lag -- yet once trimmed it costs nothing to hold
or to change, because buoyancy does the work.  Translation is quick but the
thrusters draw around 500 mW against a photovoltaic income of a few tens of
milliwatts at depth, so a float simply cannot afford to move laterally for long.
Run ``examples/03_custom_plugin.py`` and the swarm covers the full 16 m of its
helix in z while stalling at about 2 m in x and y.

The honest summary is therefore: **z is slow but free, xy is fast but rationed.**
Over any horizon longer than a battery's worth of thrust, z is the cheap axis.
A placement pass that assumes the opposite -- that depth is expensive and lateral
travel is not -- will produce formations the swarm cannot afford to reach.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...registry import register
from ...units import G_EARTH


@dataclass
class MotionState:
    """Mutable per-device motion state."""

    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    #: Water currently taken on, in kg.  Zero for a drone.
    ballast: float = 0.0
    #: Where the ballast controller is driving toward.
    ballast_target: float = 0.0
    #: Set while the body is unable to actuate (fault, brown-out).
    frozen: bool = False

    def copy(self) -> "MotionState":
        return MotionState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            ballast=self.ballast,
            ballast_target=self.ballast_target,
            frozen=self.frozen,
        )


@dataclass
class MotionCost:
    """What one integration step consumed."""

    energy: float = 0.0
    thrusting: bool = False
    pumping: bool = False


def drag_force(velocity: np.ndarray, medium, body) -> np.ndarray:
    """Quadratic drag with a Stokes term for the low-Reynolds regime.

    The two regimes matter: a 6 cm float in water at 0.2 m/s sits around
    Re ~ 10^4 where the quadratic term dominates, but during the slow start of
    a ballast manoeuvre it passes through the linear regime, and using only the
    quadratic law there would let it accelerate unphysically fast.
    """
    speed = float(np.linalg.norm(velocity))
    if speed <= 0.0 or medium.density <= 0.0:
        return np.zeros(3)
    quad = 0.5 * medium.density * body.drag_coefficient * body.frontal_area * speed
    stokes = 6.0 * math.pi * medium.viscosity * body.radius
    return -(quad + stokes) * velocity


class Kinematics:
    """Base class for a motion law."""

    #: Whether the z axis is rate limited rather than force limited.
    slow_z: bool = False

    def step(self, state: MotionState, target: np.ndarray, dt: float,
             medium, body, power, tether=None) -> MotionCost:
        """Advance ``state`` one tick toward ``target``, returning what it cost."""
        raise NotImplementedError

    def reachable_speed(self, medium, body, axis: str = "xy") -> float:
        """Steady-state speed [m/s] along an axis -- the compiler's travel estimate."""
        raise NotImplementedError


# --------------------------------------------------------------------------
@register("kinematics", "static", "moored", "fixed")
class StaticKinematics(Kinematics):
    """Does not move.  Anchors and any study that isolates the channel."""

    def step(self, state, target, dt, medium, body, power, tether=None) -> MotionCost:
        state.velocity[:] = 0.0
        return MotionCost()

    def reachable_speed(self, medium, body, axis: str = "xy") -> float:
        return 0.0


# --------------------------------------------------------------------------
@register("kinematics", "drone", "aerial", "flight")
class DroneKinematics(Kinematics):
    """Thrust-limited flight with a critically damped position controller."""

    def __init__(self, gain: float = 1.5, damping: float = 2.2,
                 max_speed: float = 6.0, arrival_radius: float = 0.05) -> None:
        self.gain = float(gain)
        self.damping = float(damping)
        self.max_speed = float(max_speed)
        self.arrival_radius = float(arrival_radius)

    def step(self, state, target, dt, medium, body, power, tether=None) -> MotionCost:
        if state.frozen:
            # No thrust means no lift.  A browned-out drone falls, and saying so
            # is the point: the same energy fault that merely parks a float is
            # unrecoverable in the air, which is a real argument for flotation.
            accel = np.array([0.0, 0.0, -G_EARTH])
            accel += drag_force(state.velocity, medium, body) / body.mass
            state.velocity += accel * dt
            state.position += state.velocity * dt
            return MotionCost()

        error = np.asarray(target, dtype=float) - state.position
        desired = self.gain * error - self.damping * state.velocity
        thrust = desired * body.mass

        limit = np.array([body.max_thrust_xy, body.max_thrust_xy, body.max_thrust_z])
        # Weight has to be carried before anything is left over for manoeuvring.
        thrust[2] += body.mass * G_EARTH
        magnitude = np.abs(thrust)
        scale = np.where(magnitude > limit, limit / np.maximum(magnitude, 1e-12), 1.0)
        thrust = thrust * scale

        accel = (thrust - np.array([0.0, 0.0, body.mass * G_EARTH])) / body.mass
        accel += drag_force(state.velocity, medium, body) / body.mass
        state.velocity += accel * dt
        speed = float(np.linalg.norm(state.velocity))
        if speed > self.max_speed:
            state.velocity *= self.max_speed / speed
        state.position += state.velocity * dt

        duty = float(np.linalg.norm(thrust) / max(np.linalg.norm(limit), 1e-12))
        moving = float(np.linalg.norm(error)) > self.arrival_radius
        # Hovering is not free: a rotorcraft draws most of its power just to
        # stay in the air, whether or not it is going anywhere.
        cost = power.thruster_power * dt * (0.6 + 0.4 * min(duty, 1.0))
        return MotionCost(energy=cost, thrusting=moving)

    def reachable_speed(self, medium, body, axis: str = "xy") -> float:
        return self.max_speed


# --------------------------------------------------------------------------
@register("kinematics", "buoyant", "float", "submarine", "ballast")
class BuoyantKinematics(Kinematics):
    """Thrusters in the plane, ballast for depth.

    The z channel is a first-order actuator: the controller asks for a ballast
    mass, the pump moves toward it at ``body.ballast_rate``, and only then does
    the net force change.  With the stock float that is a 25 s lag from full up
    to full down, so any plan that treats a depth change as instantaneous will
    fail in simulation -- which is the point.
    """

    slow_z = True

    def __init__(self, gain_xy: float = 1.2, damping_xy: float = 2.0,
                 gain_z: float = 0.25, damping_z: float = 1.5,
                 lead_fraction: float = 0.3, arrival_radius: float = 0.05) -> None:
        self.gain_xy = float(gain_xy)
        self.damping_xy = float(damping_xy)
        self.gain_z = float(gain_z)
        self.damping_z = float(damping_z)
        #: Lead time as a fraction of the full ballast swing, for anticipation.
        self.lead_fraction = float(lead_fraction)
        self.arrival_radius = float(arrival_radius)

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def neutral_ballast(medium, body) -> float:
        """Ballast mass [kg] at which the body neither rises nor sinks."""
        return medium.density * body.volume - body.mass

    @staticmethod
    def _drag_k(medium, body) -> float:
        return 0.5 * medium.density * body.drag_coefficient * body.frontal_area

    def _ballast_command(self, state, target_z: float, medium, body) -> float:
        """Ballast setting [kg] that should produce the depth we want.

        A plain PD loop cannot fly this actuator.  Full ballast travel takes
        tens of seconds, so by the time the body reaches its target the pump is
        still saturated in the wrong direction and it sails straight past --
        a limit cycle, not a controller.

        Instead: pick a target *velocity*, invert the steady-state force balance
        to get the ballast that sustains it, and add a lead term so the command
        starts unwinding while the body is still approaching.  The feedforward
        does the work and the lag becomes lag rather than oscillation.
        """
        neutral = self.neutral_ballast(medium, body)
        k = self._drag_k(medium, body)
        if k <= 0.0:
            return neutral

        # Terminal velocity the ballast range can actually sustain.
        force_up = max(neutral, 0.0) * G_EARTH
        force_down = max(body.ballast_capacity - neutral, 0.0) * G_EARTH
        v_up = math.sqrt(force_up / k)
        v_down = -math.sqrt(force_down / k)

        lag = (body.ballast_capacity / body.ballast_rate) if body.ballast_rate > 0 else 0.0
        tau = self.lead_fraction * lag
        error = (target_z - state.position[2]) - tau * state.velocity[2]

        v_des = float(np.clip(self.gain_z * error, v_down, v_up))
        # Steady-state force for that velocity, plus damping on the shortfall.
        force = k * v_des * abs(v_des)
        force += self.damping_z * k * abs(v_des) * (v_des - state.velocity[2])
        return float(np.clip(neutral - force / G_EARTH, 0.0, body.ballast_capacity))

    def step(self, state, target, dt, medium, body, power, tether=None) -> MotionCost:
        target = np.asarray(target, dtype=float)
        error = target - state.position
        cost = MotionCost()

        if not medium.supports_buoyancy:
            # Nothing to float in: the body just sits where it is.
            state.velocity[:] = 0.0
            return cost

        # --- z: ask the pump for a ballast setting, then let it get there ---
        if not state.frozen and body.ballast_capacity > 0.0:
            state.ballast_target = self._ballast_command(state, target[2], medium, body)
        delta = state.ballast_target - state.ballast
        max_delta = body.ballast_rate * dt
        if abs(delta) > 1e-12 and not state.frozen:
            state.ballast += math.copysign(min(abs(delta), max_delta), delta)
            cost.pumping = True
            cost.energy += power.ballast_power * dt

        # --- forces ----------------------------------------------------
        total_mass = body.mass + state.ballast
        buoyancy = medium.density * body.volume * G_EARTH
        weight = total_mass * G_EARTH
        force = np.array([0.0, 0.0, buoyancy - weight])

        if not state.frozen and body.max_thrust_xy > 0.0:
            desired = self.gain_xy * error[:2] - self.damping_xy * state.velocity[:2]
            thrust_xy = desired * total_mass
            mag = float(np.linalg.norm(thrust_xy))
            if mag > body.max_thrust_xy:
                thrust_xy *= body.max_thrust_xy / mag
            force[:2] += thrust_xy
            if float(np.linalg.norm(error[:2])) > self.arrival_radius:
                cost.thrusting = True
                cost.energy += power.thruster_power * dt * min(
                    mag / body.max_thrust_xy, 1.0
                )

        force += drag_force(state.velocity, medium, body)
        state.velocity += (force / total_mass) * dt
        state.position += state.velocity * dt
        return cost

    def reachable_speed(self, medium, body, axis: str = "xy") -> float:
        """Terminal velocity, which is what actually bounds travel time."""
        if medium.density <= 0.0:
            return 0.0
        k = 0.5 * medium.density * body.drag_coefficient * body.frontal_area
        if axis == "z":
            force = abs(body.ballast_capacity - self.neutral_ballast(medium, body))
            force = max(force, self.neutral_ballast(medium, body)) * G_EARTH
        else:
            force = body.max_thrust_xy
        return math.sqrt(max(force, 0.0) / max(k, 1e-12))

    def actuation_lag(self, body) -> float:
        """Seconds to swing the ballast across its full range."""
        if body.ballast_rate <= 0.0:
            return math.inf
        return body.ballast_capacity / body.ballast_rate


# --------------------------------------------------------------------------
@register("kinematics", "tethered", "remote_head", "gantry", "suspended")
class TetheredKinematics(Kinematics):
    """A remote optical head positioned from outside the medium.

    Only the head is immersed; the compute box, the store and the harvester sit
    topside on a gantry, and the head hangs beneath on a wire or waveguide.  The
    consequences are worth being explicit about, because they invert most of the
    trade-offs the free-swimming platforms impose:

    * **z stops being the expensive axis.**  A winch pays out at a commanded
      rate, so depth is as fast and as precise as translation -- no ballast, no
      25 second lag, no overshoot.  The anisotropic projection in the placement
      pass should be switched off for this platform.
    * **Station-keeping is free to the node.**  The gantry holds position, so
      none of the motion energy is drawn from the harvested budget.
    * **The workspace is a cylinder,** not open space: the head cannot leave the
      column under its gantry by more than the tether allows.
    * **The tether occludes.**  That cost is charged in the line-of-sight test,
      not here, but it is the price of the density this architecture buys.
    """

    slow_z = False

    def __init__(self, settle_gain: float = 6.0, drag_lag: bool = True) -> None:
        self.settle_gain = float(settle_gain)
        #: Let the medium's drag round off the corners of a commanded move.
        self.drag_lag = bool(drag_lag)

    @staticmethod
    def _confine(position: np.ndarray, anchor_xy: np.ndarray, tether) -> np.ndarray:
        """Clamp a commanded head position into the tether's workspace."""
        out = position.copy()
        if tether is None:
            return out
        if tether.xy_limits is not None:
            x0, x1, y0, y1 = tether.xy_limits
            out[0] = min(max(out[0], x0), x1)
            out[1] = min(max(out[1], y0), y1)
        # Depth is limited by how much wire there is to pay out.
        deepest = tether.anchor_z - tether.length
        out[2] = min(max(out[2], deepest), tether.anchor_z)
        return out

    def step(self, state, target, dt, medium, body, power, tether=None) -> MotionCost:
        cost = MotionCost()
        if state.frozen:
            # The rig keeps holding the wire, so a dead node simply stops here
            # rather than sinking or falling.  That is a real robustness gain.
            state.velocity[:] = 0.0
            return cost

        target = self._confine(np.asarray(target, dtype=float),
                               state.position[:2], tether)
        error = target - state.position

        v_xy = tether.max_speed_xy if tether is not None else 0.8
        v_z = tether.max_speed_z if tether is not None else 0.5

        desired = self.settle_gain * error
        horiz = float(np.linalg.norm(desired[:2]))
        if horiz > v_xy:
            desired[:2] *= v_xy / horiz
        desired[2] = float(np.clip(desired[2], -v_z, v_z))

        if self.drag_lag and medium.density > 0.0:
            # First-order approach to the commanded velocity, so the head does
            # not step discontinuously through the fluid.
            tau = max(dt, 0.05)
            state.velocity += (desired - state.velocity) * min(dt / tau, 1.0)
        else:
            state.velocity[:] = desired

        state.position += state.velocity * dt
        state.position[:] = self._confine(state.position, state.position[:2], tether)

        moving = bool(np.any(np.abs(desired) > 1e-6))
        cost.thrusting = moving
        # Energy is spent by the topside rig, not out of the harvested budget.
        cost.energy = 0.0
        return cost

    def reachable_speed(self, medium, body, axis: str = "xy", tether=None) -> float:
        if tether is None:
            return 0.8 if axis == "xy" else 0.5
        return tether.max_speed_xy if axis == "xy" else tether.max_speed_z
