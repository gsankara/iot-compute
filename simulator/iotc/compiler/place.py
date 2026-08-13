"""B3 -- Place: turn the embedding into physical coordinates.

B2 left one task per device and an embedding in ``R^kappa`` where distance means
"how much these two talk".  B3 turns that into metres.  Two constraints decide
the answer, and neither is geometric.

**The axes are not interchangeable.**  This is where Phase A's most
counter-intuitive measurement earns its keep.  For a photovoltaic float,
depth is changed by ballast: slow to actuate, but once the trim is set the body
drifts to its new depth at terminal velocity for *no further energy*.  Lateral
motion is the opposite -- thrusters are quick but burn stored charge every
second they run.  So

    z is slow but free;  xy is fast but rationed.

The naive reading of "z is slow" is to compress the z axis and spread the
formation out laterally.  That is exactly backwards: it spends the rationed axis
and leaves the free one idle.  :class:`AnisotropicPlacer` scales each axis by
what the **energy budget** allows rather than by how fast the body moves, and
the ``velocity`` placer is kept alongside it to measure what the naive reading
would have cost.

**Placement cannot outrun the light.**  Two devices that must exchange bytes have
to be inside optical reach, which the link budget already knows how to compute
(``FSOChannel.max_range``, written for this purpose).  A placement that ignores
it is not a worse placement, it is a disconnected machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..registry import register
from .fastmap import embed_graph
from .ir import TaskGraph


@dataclass(frozen=True)
class MotionBudget:
    """How far a device of a given class can actually move, per axis.

    Derived from the ``DeviceClass`` and the medium rather than configured, so
    the compiler and the simulator cannot disagree about what a body can do.
    """

    xy_reach: float
    z_reach: float
    max_link: float
    separation: float
    #: Joules to traverse one metre laterally; zero when a gantry does the work.
    xy_cost_per_m: float = 0.0
    #: Joules to traverse one metre vertically.  Zero for a ballast platform:
    #: the pump cost is paid once to set the trim, not per metre travelled.
    z_cost_per_m: float = 0.0
    platform: str = "buoyant"

    @property
    def anisotropy(self) -> float:
        """How much further the cheap axis reaches than the rationed one."""
        return self.z_reach / self.xy_reach if self.xy_reach > 0 else math.inf

    def summary(self) -> str:
        return (
            f"{self.platform:8s} xy={self.xy_reach:8.2f} m  z={self.z_reach:8.2f} m  "
            f"anisotropy={self.anisotropy:6.2f}  link={self.max_link:7.2f} m  "
            f"sep={self.separation:.3f} m"
        )


def terminal_velocity(body, medium, force: float) -> float:
    """Steady speed where drag balances ``force`` [m/s]."""
    rho = max(float(getattr(medium, "density", 0.0)), 1e-6)
    area = max(body.frontal_area, 1e-12)
    cd = max(body.drag_coefficient, 1e-6)
    return math.sqrt(2.0 * max(force, 0.0) / (rho * cd * area))


def derive_budget(
    device_class,
    medium,
    horizon: float = 600.0,
    energy_fraction: float = 0.25,
    line_rate: float = 1e9,
) -> MotionBudget:
    """What ``device_class`` can reach in ``horizon`` seconds on its own charge.

    ``energy_fraction`` is the share of usable stored energy the formation
    manoeuvre is allowed to spend; the rest has to remain for computing and for
    keeping the beams pointed, which is what the device is actually for.
    """
    store = device_class.store
    usable = max(store.capacity * (store.initial_fraction - store.brownout_fraction), 0.0)
    budget_j = usable * energy_fraction
    body = device_class.body

    if device_class.tether is not None:
        # The box, the battery and the winch are all topside.  Both axes are
        # driven from outside the medium, so motion costs the node nothing and
        # the limit is time and tether length.
        t = device_class.tether
        return MotionBudget(
            xy_reach=t.max_speed_xy * horizon,
            z_reach=min(t.max_speed_z * horizon, t.length),
            max_link=_optical_reach(device_class, medium, line_rate),
            separation=2.0 * body.radius + t.radius,
            platform="tethered",
        )

    v_xy = terminal_velocity(body, medium, body.max_thrust_xy)
    xy_cost = device_class.power.thruster_power / v_xy if v_xy > 0 else math.inf
    xy_reach = min(budget_j / xy_cost if xy_cost > 0 else math.inf, v_xy * horizon)

    if body.ballast_capacity > 0 and body.max_thrust_z <= 0:
        # Buoyancy-driven: the pump runs once to set the trim, then the body
        # coasts to depth.  Energy is a constant, not a rate, so the reach is
        # bounded by *time*, not by charge -- the free axis.
        net_force = body.ballast_capacity * 9.80665
        v_z = terminal_velocity(body, medium, net_force)
        return MotionBudget(
            xy_reach=xy_reach,
            z_reach=v_z * horizon,
            max_link=_optical_reach(device_class, medium, line_rate),
            separation=2.0 * body.radius,
            xy_cost_per_m=xy_cost,
            z_cost_per_m=0.0,
            platform="buoyant",
        )

    # Thrust-driven in every axis -- a drone.  Vertical travel fights gravity,
    # so it is the *expensive* axis and the anisotropy inverts.
    v_z = terminal_velocity(body, medium, body.max_thrust_z)
    z_cost = device_class.power.thruster_power / v_z if v_z > 0 else math.inf
    return MotionBudget(
        xy_reach=xy_reach,
        z_reach=min(budget_j / z_cost if z_cost > 0 else math.inf, v_z * horizon),
        max_link=_optical_reach(device_class, medium, line_rate),
        separation=2.0 * body.radius,
        xy_cost_per_m=xy_cost,
        z_cost_per_m=z_cost,
        platform="aerial",
    )


def _optical_reach(device_class, medium, line_rate: float) -> float:
    """Longest link this class can close in this medium, from the link budget."""
    from ..registry import create
    from ..sim.links import LinkGeometry

    o = device_class.optical
    channel = create("channel", "fso")
    geo = LinkGeometry(
        distance=1.0,
        wavelength=o.wavelength,
        tx_power=o.tx_power,
        divergence=o.divergence,
        tx_aperture_radius=o.tx_aperture_radius,
        rx_aperture_radius=o.rx_aperture_radius,
        rx_fov=o.rx_fov,
    )
    return float(channel.max_range(geo, medium, line_rate=line_rate))


# --------------------------------------------------------------------------
def principal_axes(coords: np.ndarray) -> np.ndarray:
    """Project ``R^kappa`` onto its three principal directions.

    The embedding's axes are FastMap pivots, which are arbitrary directions in
    a space of arbitrary dimension.  Physical space has exactly three, and which
    three matter is a property of the data -- so take the directions carrying
    the most spread rather than the first three columns.
    """
    if coords.shape[0] == 0:
        return np.zeros((0, 3))
    centred = coords - coords.mean(axis=0)
    out = np.zeros((centred.shape[0], 3))
    if centred.shape[1] <= 3:
        out[:, : centred.shape[1]] = centred
        return out
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    # SVD returns min(n, kappa) components, so a graph coarsened to fewer than
    # three devices yields fewer than three directions.  Physical space always
    # has three axes; the missing ones are genuinely flat, not absent.
    projected = centred @ vt[:3].T
    out[:, : projected.shape[1]] = projected
    return out


def _scale_to(values: np.ndarray, extent: float) -> np.ndarray:
    """Rescale one axis so its full spread is ``extent`` metres."""
    spread = float(values.max() - values.min()) if values.size else 0.0
    if spread <= 0 or not math.isfinite(extent):
        return np.zeros_like(values)
    return (values - values.mean()) * (extent / spread)


class Placer:
    """The ``placer`` plugin slot."""

    def project(self, axes: np.ndarray, budget: MotionBudget) -> np.ndarray:
        raise NotImplementedError  # pragma: no cover


@register("placer", "anisotropic", "energy")
class AnisotropicPlacer(Placer):
    """Scale each axis by what the energy budget allows.  The pass.

    The widest direction of the embedding is assigned to the axis with the most
    reach -- for a float that is depth, because depth is free.  The result is a
    formation that is tall and laterally compact, which is the opposite of what
    the naive reading produces and is why this is a pass rather than a rescale.
    """

    def project(self, axes: np.ndarray, budget: MotionBudget) -> np.ndarray:
        spreads = axes.max(axis=0) - axes.min(axis=0) if axes.size else np.zeros(3)
        # Widest embedded direction onto the roomiest physical axis.
        order = np.argsort(-spreads)
        reaches = np.array([budget.xy_reach, budget.xy_reach, budget.z_reach])
        target = np.argsort(-reaches)

        out = np.zeros_like(axes)
        for embedded, physical in zip(order, target):
            out[:, physical] = _scale_to(axes[:, embedded], reaches[physical])
        return out


@register("placer", "velocity")
class VelocityPlacer(Placer):
    """Scale by how fast the body moves, not by what the motion costs.

    The pre-correction design, kept as the comparator.  Ballast is slow, so this
    compresses depth and spreads the swarm laterally -- spending the rationed
    axis to save the free one.  The bench measures the bill.
    """

    def project(self, axes: np.ndarray, budget: MotionBudget) -> np.ndarray:
        # Reach in proportion to speed: z is the slow axis, so it gets squeezed.
        z_share = budget.z_reach / max(budget.z_reach + budget.xy_reach, 1e-12)
        scale = np.array([
            budget.xy_reach,
            budget.xy_reach,
            budget.xy_reach * z_share,
        ])
        spreads = axes.max(axis=0) - axes.min(axis=0) if axes.size else np.zeros(3)
        order = np.argsort(-spreads)
        out = np.zeros_like(axes)
        for embedded, physical in zip(order, np.argsort(-scale)):
            out[:, physical] = _scale_to(axes[:, embedded], scale[physical])
        return out


@register("placer", "isotropic")
class IsotropicPlacer(Placer):
    """One scale for all three axes -- geometry with no physics in it."""

    def project(self, axes: np.ndarray, budget: MotionBudget) -> np.ndarray:
        extent = min(budget.xy_reach, budget.z_reach)
        out = np.zeros_like(axes)
        for axis in range(3):
            out[:, axis] = _scale_to(axes[:, axis], extent)
        return out


def enforce_separation(
    positions: np.ndarray, separation: float, rounds: int = 24
) -> np.ndarray:
    """Push overlapping bodies apart.

    Devices have volume.  An embedding does not know that, and two tasks that
    talk constantly will happily be placed on top of each other -- which is not
    a near-optimal placement but an impossible one.
    """
    if positions.shape[0] < 2 or separation <= 0:
        return positions
    p = positions.astype(float).copy()

    # Two tasks with identical connectivity embed to identical coordinates, and
    # a pair at zero distance has no separation *direction* -- the push is zero
    # and they stay stacked however many rounds run.  Break the tie first, with
    # a deterministic spiral offset rather than a random one so the placement
    # stays reproducible.
    n = p.shape[0]
    golden = math.pi * (3.0 - math.sqrt(5.0))
    idx = np.arange(n)
    lat = np.arccos(np.clip(1.0 - 2.0 * (idx + 0.5) / n, -1.0, 1.0))
    lon = golden * idx
    spiral = np.column_stack([
        np.sin(lat) * np.cos(lon), np.sin(lat) * np.sin(lon), np.cos(lat)
    ])
    p += spiral * (separation * 1e-3)
    for _ in range(rounds):
        delta = p[:, None, :] - p[None, :, :]
        dist = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(dist, np.inf)
        close = dist < separation
        if not close.any():
            break
        # Work only on the overlapping pairs.  Computing the push everywhere
        # multiplies the infinite self-distances by a zero direction and yields
        # NaN, which then leaks into every coordinate.
        safe = np.where(close, dist, 1.0)
        unit = np.where(close[:, :, None] & (safe[:, :, None] > 0),
                        delta / safe[:, :, None], 0.0)
        push = np.where(close[:, :, None], unit * (separation - safe[:, :, None]) * 0.5, 0.0)
        p += push.sum(axis=1)
    return p


def fit_to_reach(
    positions: np.ndarray, graph: TaskGraph, max_link: float, margin: float = 0.95
) -> np.ndarray:
    """Shrink the whole formation until its longest link closes.

    The energy budget and the optical budget are separate limits and they do not
    agree: a photovoltaic float may move 161 m vertically for free, but 470 nm
    light only crosses about 42 m of water.  Spending the whole free axis
    therefore buys a formation the beams cannot span.

    Scaling uniformly is the right correction because it preserves the *shape*
    the projection chose -- a tall, laterally compact swarm stays tall and
    laterally compact -- while bringing every link inside reach.  The anisotropy
    is the decision; the absolute size is not.

    ``margin`` leaves headroom below the theoretical maximum.  A link placed at
    exactly its closing distance has none, and Phase A measured that a 0.04 m/s
    drift is enough to destroy an open-loop link -- so the last few percent of
    reach is not usable range, it is the tolerance budget.
    """
    e = graph.edges
    if e.size == 0 or not math.isfinite(max_link) or max_link <= 0:
        return positions
    target = max_link * margin
    lengths = np.linalg.norm(positions[e[:, 0]] - positions[e[:, 1]], axis=1)
    longest = float(lengths.max()) if lengths.size else 0.0
    if longest <= target or longest <= 0:
        return positions
    return positions * (target / longest)


@dataclass
class PlaceResult:
    """Physical coordinates plus the evidence they are usable."""

    positions: np.ndarray
    graph: TaskGraph
    budget: MotionBudget
    method: str = ""

    @property
    def link_lengths(self) -> np.ndarray:
        e = self.graph.edges
        if e.size == 0:
            return np.zeros(0)
        return np.linalg.norm(self.positions[e[:, 0]] - self.positions[e[:, 1]], axis=1)

    @property
    def unreachable(self) -> int:
        """Edges longer than the optical budget can close -- broken links."""
        return int((self.link_lengths > self.budget.max_link).sum())

    @property
    def unreachable_fraction(self) -> float:
        return self.unreachable / self.graph.n_edges if self.graph.n_edges else 0.0

    @property
    def weighted_link_length(self) -> float:
        """Mean link length weighted by payload -- what the beams actually cost."""
        lengths = self.link_lengths
        if lengths.size == 0:
            return 0.0
        w = self.graph.payload
        return float((lengths * w).sum() / max(w.sum(), 1e-12))

    @property
    def motion_energy(self) -> float:
        """Joules to fly this formation from the origin, summed over devices."""
        p = self.positions
        xy = np.linalg.norm(p[:, :2], axis=1)
        z = np.abs(p[:, 2])
        return float((xy * self.budget.xy_cost_per_m + z * self.budget.z_cost_per_m).sum())

    @property
    def min_separation(self) -> float:
        if self.positions.shape[0] < 2:
            return math.inf
        d = np.linalg.norm(self.positions[:, None, :] - self.positions[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        return float(d.min())

    @property
    def extent(self) -> np.ndarray:
        if self.positions.size == 0:
            return np.zeros(3)
        return self.positions.max(axis=0) - self.positions.min(axis=0)

    def summary(self) -> str:
        ext = self.extent
        return (
            f"{self.method:12s} extent {ext[0]:7.1f} x {ext[1]:7.1f} x {ext[2]:7.1f} m  "
            f"link {self.weighted_link_length:7.2f} m  "
            f"unreachable {self.unreachable_fraction:5.1%}  "
            f"energy {self.motion_energy:8.2f} J"
        )


def place(
    graph: TaskGraph,
    device_class,
    medium,
    model: Placer | str = "anisotropic",
    kappa: int = 6,
    seed: int = 0,
    budget: MotionBudget | None = None,
    separate: bool = True,
) -> PlaceResult:
    """Embed, project onto physical axes, and separate the bodies."""
    if isinstance(model, str):
        from ..registry import create

        name, model = model, create("placer", model)
    else:
        name = type(model).plugin_name

    if budget is None:
        budget = derive_budget(device_class, medium)

    coords = embed_graph(graph, kappa=kappa, seed=seed)
    axes = principal_axes(coords)
    positions = model.project(axes, budget)

    # The two hard constraints pull against each other: shrinking to optical
    # reach can push bodies into one another, and pushing them apart can stretch
    # a link back out of reach.  Alternate rather than pick one -- the
    # separation nudges are centimetres against links of metres, so a couple of
    # rounds settles it.
    for _ in range(3):
        positions = fit_to_reach(positions, graph, budget.max_link)
        if not separate:
            break
        moved = enforce_separation(positions, budget.separation)
        if np.allclose(moved, positions):
            break
        positions = moved

    return PlaceResult(
        positions=positions, graph=graph, budget=budget, method=name
    )
