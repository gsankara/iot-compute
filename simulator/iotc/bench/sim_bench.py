"""Design-space sweeps over the simulator.

A sweep is a base configuration plus a list of parameters to vary, run across
several seeds, producing one tidy CSV row per (configuration, seed).  That is
deliberately boring: the point is that "should this deployment float, fly or
hang off a wire" becomes a table rather than an argument.

Every row carries the full summary, so the same output answers questions the
sweep was not designed for.
"""

from __future__ import annotations

import copy
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np

from .. import config as cfgmod
from ..sim.trace import Trace


# --------------------------------------------------------------------------
def set_path(cfg: dict, path: str, value) -> None:
    """Set ``cfg`` at a dotted path, creating nothing and indexing lists.

    ``swarm.0.variant`` addresses the first swarm group's variant.  Failing
    loudly on a missing key is deliberate -- a typo in a sweep axis would
    otherwise silently produce a column of identical results.
    """
    parts = path.split(".")
    node = cfg
    for part in parts[:-1]:
        if isinstance(node, list):
            node = node[int(part)]
        else:
            if part not in node:
                raise KeyError(f"sweep path {path!r}: no key {part!r}")
            node = node[part]
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        if last not in node:
            raise KeyError(f"sweep path {path!r}: no key {last!r}")
        node[last] = value


def get_path(cfg: dict, path: str):
    node = cfg
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


# --------------------------------------------------------------------------
def run_one(cfg: dict, seed: int, until: float, connect: bool = True,
            traffic: int = 0) -> dict:
    """Run a single configuration and return its summary row."""
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = seed
    trace = Trace(sample_every=max(until / 500.0, 1e-3))
    world = cfgmod.build_world(cfg, trace=trace)
    sim = cfgmod.build_simulation(cfg, world)

    started = time.perf_counter()
    links = 0
    if connect:
        links = _connect(sim, world)
    sim.run(until)

    if traffic:
        _offer_traffic(sim, world, traffic)
    sim.drain(1.0)

    row = sim.summary()
    row["seed"] = seed
    row["links_established"] = links
    row["wall_seconds"] = time.perf_counter() - started
    row["retarget_energy"] = trace.sums.get("retarget_energy", 0.0)
    row["mean_gimbal_share"] = float(np.mean([
        d.power.breakdown().get("gimbal", 0.0) for d in world.devices
    ]))
    row["mean_motion_share"] = float(np.mean([
        d.power.breakdown().get("motion", 0.0) for d in world.devices
    ]))
    if world.goals is not None:
        err = np.linalg.norm(world.positions - world.goals, axis=1)
        row["formation_rms"] = float(np.sqrt(np.mean(err**2)))
    return row


def _connect(sim, world) -> int:
    """Aim every device at its nearest reachable peers."""
    from ..sim.links.base import LinkGeometry

    count = 0
    for i, dev in enumerate(world.devices):
        o = dev.cls.optical
        geo = LinkGeometry(
            distance=1.0, wavelength=o.wavelength, tx_power=dev.tx_power,
            divergence=o.divergence, tx_aperture_radius=o.tx_aperture_radius,
            rx_aperture_radius=o.rx_aperture_radius, rx_fov=o.rx_fov,
        )
        reach = world.channel.max_range(geo, world.medium, line_rate=o.line_rate)
        for j in world.neighbours(i, reach):
            if dev.free_transceiver(world.time) is None:
                break
            if dev.transceiver_for(j) is not None:
                continue
            if world.devices[j].free_transceiver(world.time) is None:
                continue
            if sim.establish(i, j).get("ok"):
                count += 1
    return count


def _offer_traffic(sim, world, packets: int) -> None:
    """Push traffic across every established link, to measure goodput."""
    links = world.established_links()
    if not links:
        return
    for _ in range(packets):
        for i, j in links:
            sim.send(i, j, world.payload_bits)
        sim.run(world.time + sim.dt)


# --------------------------------------------------------------------------
def expand(cfg: dict) -> list[tuple[dict, dict]]:
    """Cartesian product of the sweep axes -> (labels, configuration)."""
    base = cfg.get("base")
    if isinstance(base, str):
        base_cfg = cfgmod.load_file(Path(base))
    elif isinstance(base, dict):
        base_cfg = base
    else:
        base_cfg = {k: v for k, v in cfg.items()
                    if k not in ("sweep", "seeds", "run", "name", "base")}

    axes = cfg.get("sweep") or []
    if not axes:
        return [({}, copy.deepcopy(base_cfg))]

    names = [a["path"] for a in axes]
    values = [a["values"] for a in axes]
    # An axis may name its values.  Two entries can flatten to the same string
    # -- two position_drift rates, say -- and would then be silently averaged
    # together in the summary, so an explicit label is the way to keep them apart.
    tags = []
    for axis in axes:
        given = axis.get("labels")
        if given is not None and len(given) != len(axis["values"]):
            raise ValueError(
                f"sweep axis {axis['path']!r}: {len(given)} labels for "
                f"{len(axis['values'])} values"
            )
        tags.append(given)

    out = []
    for indices in itertools.product(*(range(len(v)) for v in values)):
        variant = copy.deepcopy(base_cfg)
        labels = {}
        for path, value_list, tag_list, k in zip(names, values, tags, indices):
            set_path(variant, path, value_list[k])
            labels[path] = tag_list[k] if tag_list else value_list[k]
        out.append((labels, variant))
    return out


def run_sweep(args) -> int:
    """Entry point for ``iotc bench sweep``."""
    cfg = cfgmod.load_file(args.config)
    run_cfg = cfg.get("run", {}) or {}
    until = float(run_cfg.get("until", 30.0))
    connect = bool(run_cfg.get("connect", True))
    traffic = int(run_cfg.get("traffic", 0))
    seeds = int(args.seeds if args.seeds is not None else cfg.get("seeds", 3))

    combos = expand(cfg)
    rows = []
    total = len(combos) * seeds
    print(f"{len(combos)} configurations x {seeds} seeds = {total} runs")

    for index, (labels, variant) in enumerate(combos):
        for seed in range(seeds):
            row = run_one(variant, seed, until, connect=connect, traffic=traffic)
            row.update({f"cfg.{k}": _flatten(v) for k, v in labels.items()})
            row["config_index"] = index
            rows.append(row)
            label = ", ".join(f"{k}={_flatten(v)}" for k, v in labels.items()) or "base"
            print(f"  [{len(rows):4d}/{total}] {label} seed={seed} "
                  f"links={row['links_established']} "
                  f"delivered={row['delivered']} "
                  f"({row['wall_seconds']:.2f} s)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out / "sweep.csv")
    summary = aggregate(rows, [f"cfg.{a['path']}" for a in (cfg.get("sweep") or [])])
    write_csv(summary, out / "sweep_summary.csv")
    (out / "sweep.json").write_text(json.dumps(rows, indent=2, default=float),
                                    encoding="utf-8")
    print(f"\nwrote {out}/sweep.csv and sweep_summary.csv ({len(rows)} rows)")

    if getattr(args, "plot", False):
        try:
            plot_sweep(rows, cfg, out)
            print(f"wrote figures to {out}")
        except ImportError:
            print("matplotlib not installed; skipping figures")
    return 0


def _flatten(value):
    """Reduce a sweep value to something hashable and readable in a CSV cell.

    Sweep axes routinely carry lists (composed device variants) and dicts (a
    plugin spec), and both have to survive into a grouping key.
    """
    if isinstance(value, dict):
        return value.get("type", json.dumps(value, sort_keys=True))
    if isinstance(value, (list, tuple)):
        if not value:
            return "none"
        return "+".join(str(_flatten(v)) for v in value)
    return value


# --------------------------------------------------------------------------
def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def aggregate(rows: list[dict], group_keys: list[str]) -> list[dict]:
    """Mean and standard deviation across seeds, per configuration."""
    if not rows:
        return []
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        groups.setdefault(key, []).append(row)

    numeric = [
        k for k, v in rows[0].items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    out = []
    for key, members in groups.items():
        entry = dict(zip(group_keys, key))
        entry["runs"] = len(members)
        for k in numeric:
            vals = np.array([m.get(k, np.nan) for m in members], dtype=float)
            entry[f"{k}.mean"] = float(np.nanmean(vals))
            entry[f"{k}.std"] = float(np.nanstd(vals))
        out.append(entry)
    return out


def plot_sweep(rows: list[dict], cfg: dict, out: Path) -> None:
    """Bar charts of the headline metrics against the first sweep axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes = cfg.get("sweep") or []
    if not axes:
        return
    key = f"cfg.{axes[0]['path']}"
    metrics = ["links_established", "delivery_rate", "optical_share",
               "retarget_overhead", "mean_energy_fraction"]

    labels = sorted({str(r.get(key)) for r in rows})
    fig, ax_list = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.4))
    for ax, metric in zip(np.atleast_1d(ax_list), metrics):
        means, errs = [], []
        for label in labels:
            vals = [r[metric] for r in rows if str(r.get(key)) == label and metric in r]
            means.append(float(np.mean(vals)) if vals else 0.0)
            errs.append(float(np.std(vals)) if vals else 0.0)
        ax.bar(labels, means, yerr=errs, capsize=3, color="#4C72B0")
        ax.set_title(metric.replace("_", " "))
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(cfg.get("name", "sweep"))
    fig.tight_layout()
    fig.savefig(out / "sweep.png", dpi=140)
    plt.close(fig)
