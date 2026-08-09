"""Config loading, the CLI, and the sweep harness."""

from __future__ import annotations

import json

import numpy as np
import pytest

from iotc import cli, config as cfgmod
from iotc.bench import aggregate, expand, run_one, set_path
from iotc.config import ConfigError, build_layout, build_world, resolve_class
from iotc.rng import RandomSource

BASE = {
    "seed": 0,
    "medium": {"type": "water", "turbidity": 0.02},
    "channel": {"type": "fso", "turbulence": "none"},
    "topology": "static",
    "world": {"dt": 0.05},
    "swarm": [
        {"class": "mote-c", "variant": ["aquatic"], "count": 6,
         "layout": {"type": "ring", "radius": 8.0, "centre": [0, 0, -5]}}
    ],
}


# --------------------------------------------------------------------------
# device resolution
# --------------------------------------------------------------------------
def test_variants_compose_in_order():
    """``tethered`` alone would leave near-infrared optics under water."""
    both = resolve_class({"class": "mote-c", "variant": ["aquatic", "tethered"]})
    assert both.optical.wavelength == pytest.approx(470e-9)
    assert both.kinematics == "tethered"
    assert both.tether is not None

    platform_only = resolve_class({"class": "mote-c", "variant": ["tethered"]})
    assert platform_only.optical.wavelength == pytest.approx(850e-9)


def test_per_variant_arguments():
    cls = resolve_class({
        "class": "mote-c",
        "variant": ["aquatic", "tethered"],
        "variant_args": {"aquatic": {"wavelength": 520e-9},
                         "tethered": {"length": 12.0}},
    })
    assert cls.optical.wavelength == pytest.approx(520e-9)
    assert cls.tether.length == pytest.approx(12.0)


def test_single_variant_takes_plain_kwargs():
    cls = resolve_class({"class": "mote-c", "variant": "aquatic",
                         "variant_args": {"wavelength": 500e-9}})
    assert cls.optical.wavelength == pytest.approx(500e-9)


def test_unknown_variant_is_rejected():
    with pytest.raises(ConfigError):
        resolve_class({"class": "mote-c", "variant": "submarine"})


def test_missing_class_is_rejected():
    with pytest.raises(ConfigError):
        resolve_class({"count": 3})


# --------------------------------------------------------------------------
# layouts
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["lattice", "sphere", "ring", "line", "random"])
def test_layouts_produce_the_right_shape(kind):
    rng = RandomSource(0).stream("layout")
    pts = build_layout({"type": kind, "centre": [1.0, 2.0, -3.0]}, 7, rng)
    assert pts.shape == (7, 3)
    assert np.isfinite(pts).all()


def test_explicit_layout_must_match_the_count():
    rng = RandomSource(0).stream("layout")
    assert build_layout([[0, 0, 0], [1, 1, 1]], 2, rng).shape == (2, 3)
    with pytest.raises(ConfigError):
        build_layout([[0, 0, 0]], 2, rng)


def test_unknown_layout_is_rejected():
    with pytest.raises(ConfigError):
        build_layout({"type": "fractal"}, 3, RandomSource(0).stream("layout"))


# --------------------------------------------------------------------------
# world construction
# --------------------------------------------------------------------------
def test_build_world_from_a_mapping():
    world = build_world(BASE)
    assert len(world.devices) == 6
    assert world.medium.plugin_name == "water"
    assert all(d.cls.optical.wavelength == pytest.approx(470e-9) for d in world.devices)


def test_a_config_without_a_swarm_is_rejected():
    with pytest.raises(ConfigError):
        build_world({"medium": "water"})


def test_faults_are_built_from_the_config():
    cfg = dict(BASE, faults=[{"type": "node_crash", "rate": 0.01}])
    world = build_world(cfg)
    assert len(world.faults) == 1
    assert world.faults[0].rate == pytest.approx(0.01)


def test_reach_report_names_every_class():
    world = build_world(BASE)
    report = cfgmod.reach_report(world)
    assert "mote-c-aq" in report
    assert "optical reach" in report


def test_summary_lists_the_device_mix():
    world = build_world(BASE)
    text = cfgmod.dump_summary(world, {"name": "unit-test"})
    assert "unit-test" in text
    assert "6 x mote-c-aq" in text


def test_shipped_configs_all_build(tmp_path):
    """A config that no longer parses is a broken example, not a stale one."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "configs"
    for path in sorted(root.glob("*.yaml")):
        cfg = cfgmod.load_file(path)
        if "swarm" in cfg:
            world = build_world(cfg)
            assert len(world.devices) > 0, path.name
        else:  # a sweep config
            combos = expand(cfg)
            assert combos, path.name
            for _, variant in combos[:2]:
                assert len(build_world(variant).devices) > 0, path.name


def test_shipped_liquid_config_can_actually_link():
    """The reach must exceed the layout spacing, or the config is useless."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "configs"
    world = build_world(cfgmod.load_file(root / "liquid_swarm.yaml"))
    from iotc.sim.links.base import LinkGeometry

    dev = world.devices[2]
    o = dev.cls.optical
    geo = LinkGeometry(
        distance=1.0, wavelength=o.wavelength, tx_power=dev.tx_power,
        divergence=o.divergence, tx_aperture_radius=o.tx_aperture_radius,
        rx_aperture_radius=o.rx_aperture_radius, rx_fov=o.rx_fov,
    )
    assert world.channel.max_range(geo, world.medium, line_rate=o.line_rate) > 10.0


# --------------------------------------------------------------------------
# sweeps
# --------------------------------------------------------------------------
def test_set_path_indexes_lists_and_rejects_typos():
    cfg = {"swarm": [{"variant": "aquatic", "layout": {"spacing": 5.0}}]}
    set_path(cfg, "swarm.0.layout.spacing", 9.0)
    assert cfg["swarm"][0]["layout"]["spacing"] == 9.0
    # A typo would otherwise produce a column of identical results.
    with pytest.raises(KeyError):
        set_path(cfg, "swarm.0.layuot.spacing", 9.0)


def test_expand_is_the_cartesian_product():
    cfg = {"base": BASE, "sweep": [
        {"path": "swarm.0.count", "values": [2, 4]},
        {"path": "topology", "values": ["static", "gradient"]},
    ]}
    combos = expand(cfg)
    assert len(combos) == 4
    assert {c[0]["swarm.0.count"] for c in combos} == {2, 4}


def test_sweep_labels_keep_distinct_values_apart():
    """Two entries can flatten to the same string and be silently averaged."""
    cfg = {"base": BASE, "sweep": [{
        "path": "swarm.0.count",
        "labels": ["small", "large"],
        "values": [2, 4],
    }]}
    combos = expand(cfg)
    assert [c[0]["swarm.0.count"] for c in combos] == ["small", "large"]


def test_mismatched_label_count_is_rejected():
    cfg = {"base": BASE, "sweep": [{
        "path": "swarm.0.count", "labels": ["only-one"], "values": [2, 4],
    }]}
    with pytest.raises(ValueError):
        expand(cfg)


def test_run_one_produces_a_summary_row():
    row = run_one(BASE, seed=0, until=2.0, connect=True, traffic=2)
    for key in ("delivered", "links", "seed", "links_established", "wall_seconds"):
        assert key in row
    assert row["seed"] == 0


def test_aggregate_reports_mean_and_spread():
    rows = [
        {"cfg.x": "a", "delivered": 10.0}, {"cfg.x": "a", "delivered": 20.0},
        {"cfg.x": "b", "delivered": 5.0},
    ]
    out = aggregate(rows, ["cfg.x"])
    by = {r["cfg.x"]: r for r in out}
    assert by["a"]["delivered.mean"] == pytest.approx(15.0)
    assert by["a"]["delivered.std"] == pytest.approx(5.0)
    assert by["b"]["runs"] == 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def test_cli_plugins_and_devices(capsys):
    assert cli.main(["plugins"]) == 0
    out = capsys.readouterr().out
    assert "turbulence" in out and "gamma_gamma" in out

    assert cli.main(["devices"]) == 0
    assert "mote-c" in capsys.readouterr().out


def test_cli_link_calculator(capsys):
    assert cli.main(["link", "--medium", "water", "--wavelength", "470",
                     "--distance", "10"]) == 0
    out = capsys.readouterr().out
    assert "max range" in out and "extinction" in out


def test_cli_sim_check_mode(tmp_path, capsys):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(BASE), encoding="utf-8")
    assert cli.main(["sim", str(path), "--check"]) == 0
    out = capsys.readouterr().out
    assert "optical reach" in out


def test_cli_sim_writes_output(tmp_path, capsys):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(BASE), encoding="utf-8")
    out_dir = tmp_path / "run"
    assert cli.main(["sim", str(path), "--until", "1.0", "--connect",
                     "--traffic", "2", "--out", str(out_dir)]) == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["devices"] == 6
    assert (out_dir / "events.csv").exists()
    assert (out_dir / "devices.json").exists()
