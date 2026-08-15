"""B9's evaluation harness."""

from __future__ import annotations

import json

from iotc.bench import codegen_bench as cb


def test_looping_dominates_at_realistic_run_lengths():
    rows = cb.run(["stencil3d"], repeats=(100_000,))
    by = {r["encoding"]: r["total_kib"] for r in rows}
    assert by["packed"] < by["flat"]
    assert by["looped"] < by["packed"] / 1000


def test_unrolled_images_overflow_flash_and_looped_ones_do_not():
    rows = cb.run(["stencil3d"], repeats=(100, 100_000))
    long_flat = [r for r in rows if r["encoding"] == "flat" and r["repeat"] == 100_000][0]
    long_loop = [r for r in rows if r["encoding"] == "looped" and r["repeat"] == 100_000][0]
    assert not long_flat["fits_flash"]
    assert long_loop["fits_flash"]
    assert "do not fit flash" in cb.verdict(rows)


def test_the_looped_image_does_not_grow_with_the_run():
    rows = [r for r in cb.run(["stencil3d"], encodings=("looped",),
                              repeats=(100, 100_000))]
    sizes = sorted(r["total_kib"] for r in rows)
    assert sizes[-1] < sizes[0] * 1.2


def test_the_vm_round_trip_is_part_of_the_bench():
    rows = cb.run(["stencil3d"], encodings=("looped",), repeats=(100,))
    assert all(r["delivers_plan"] for r in rows)
    assert all(r["vm_tx_bytes"] == r["vm_rx_bytes"] for r in rows)
    assert "deliver exactly the bytes B6 planned" in cb.verdict(rows)


def test_instructions_stored_is_independent_of_instructions_executed():
    """The distinction the looped encoding exists to exploit."""
    rows = cb.run(["stencil3d"], encodings=("looped",), repeats=(100, 10_000))
    for r in rows:
        assert r["instructions_executed"] > r["instructions_stored"]
    stored = {r["instructions_stored"] for r in rows}
    assert len(stored) == 1


def test_run_writes_tidy_output(tmp_path):
    rows = cb.run(["stencil3d"], encodings=("looped",), repeats=(100,), out=tmp_path)
    assert (tmp_path / "codegen.csv").exists()
    assert len(json.loads((tmp_path / "codegen.json").read_text())) == len(rows)


def test_evaluation_is_reproducible():
    kw = dict(encodings=("looped",), repeats=(100,))
    assert cb.run(["stencil3d"], **kw) == cb.run(["stencil3d"], **kw)
