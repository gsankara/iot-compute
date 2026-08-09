"""Structured workloads: stencils, GEMM, FFT, all-reduce.

These four span the interesting range of arithmetic intensity, which on this
fabric is the axis that decides feasibility:

* a **stencil** has surface-to-volume scaling, so intensity *rises* with tile
  size -- bigger tiles are the cheapest thing a compiler can do here;
* **SUMMA** is compute-bound at scale and the friendliest of the four;
* an **FFT** butterfly has fixed all-to-all-ish structure whose degree grows as
  log p, and it re-partners every stage, which is exactly what an aim-dominated
  fabric is worst at;
* an **all-reduce** is nearly pure communication and exists here as the
  pathological case the scheduler has to be measured against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...registry import register
from ..ir import TaskGraph
from .base import Workload, _split


@register("workload", "stencil2d", "jacobi2d")
@dataclass(frozen=True)
class Stencil2D(Workload):
    """Five-point Jacobi on a 2D grid, decomposed into a ``tx`` by ``ty`` mesh.

    Each task owns a rectangular subdomain and exchanges one row or column of
    halo with each face neighbour per iteration.
    """

    nx: int = 512
    ny: int = 512
    tx: int = 4
    ty: int = 4
    iterations: int = 100
    #: FLOPs per cell per sweep: four adds, one scale, one residual update.
    flop_per_cell: float = 6.0
    #: Arrays held per cell -- u and u_new for a Jacobi (not Gauss-Seidel) sweep.
    fields: int = 2

    def build(self) -> TaskGraph:
        cols, rows = _split(self.nx, self.tx), _split(self.ny, self.ty)
        flop, ws, label = [], [], []
        for j, h in enumerate(rows):
            for i, w in enumerate(cols):
                cells = w * h
                halo = 2 * (w + h)  # one ghost ring
                flop.append(cells * self.flop_per_cell)
                ws.append((cells * self.fields + halo) * self.dtype_bytes)
                label.append(f"tile[{i},{j}]")

        def tid(i: int, j: int) -> int:
            return j * self.tx + i

        src, dst, pay = [], [], []
        for j, h in enumerate(rows):
            for i, w in enumerate(cols):
                if i + 1 < self.tx:  # east face: exchange a column of height h
                    src.append(tid(i, j))
                    dst.append(tid(i + 1, j))
                    pay.append(h * self.dtype_bytes)
                if j + 1 < self.ty:  # north face: exchange a row of width w
                    src.append(tid(i, j))
                    dst.append(tid(i, j + 1))
                    pay.append(w * self.dtype_bytes)

        return TaskGraph.from_arrays(
            flop, ws, src, dst, pay,
            name=f"stencil2d[{self.nx}x{self.ny}/{self.tx}x{self.ty}]",
            iterations=self.iterations,
            kind="stencil",
            label=label,
        )


@register("workload", "stencil3d", "jacobi3d")
@dataclass(frozen=True)
class Stencil3D(Workload):
    """Seven-point Jacobi on a 3D grid over a ``tx`` by ``ty`` by ``tz`` mesh.

    The 3D case is the one that motivates the whole substrate: the task mesh is
    already a 3D lattice, so a placement that maps it onto physical space has a
    natural, low-distortion target -- which is precisely what B3 exploits.
    """

    nx: int = 64
    ny: int = 64
    nz: int = 64
    tx: int = 2
    ty: int = 2
    tz: int = 2
    iterations: int = 100
    flop_per_cell: float = 8.0
    fields: int = 2

    def build(self) -> TaskGraph:
        xs, ys, zs = _split(self.nx, self.tx), _split(self.ny, self.ty), _split(self.nz, self.tz)
        flop, ws, label = [], [], []
        for k, d in enumerate(zs):
            for j, h in enumerate(ys):
                for i, w in enumerate(xs):
                    cells = w * h * d
                    halo = 2 * (w * h + w * d + h * d)
                    flop.append(cells * self.flop_per_cell)
                    ws.append((cells * self.fields + halo) * self.dtype_bytes)
                    label.append(f"tile[{i},{j},{k}]")

        def tid(i: int, j: int, k: int) -> int:
            return (k * self.ty + j) * self.tx + i

        src, dst, pay = [], [], []
        for k, d in enumerate(zs):
            for j, h in enumerate(ys):
                for i, w in enumerate(xs):
                    if i + 1 < self.tx:
                        src.append(tid(i, j, k)); dst.append(tid(i + 1, j, k))
                        pay.append(h * d * self.dtype_bytes)
                    if j + 1 < self.ty:
                        src.append(tid(i, j, k)); dst.append(tid(i, j + 1, k))
                        pay.append(w * d * self.dtype_bytes)
                    if k + 1 < self.tz:
                        src.append(tid(i, j, k)); dst.append(tid(i, j, k + 1))
                        pay.append(w * h * self.dtype_bytes)

        return TaskGraph.from_arrays(
            flop, ws, src, dst, pay,
            name=f"stencil3d[{self.nx}x{self.ny}x{self.nz}/{self.tx}x{self.ty}x{self.tz}]",
            iterations=self.iterations,
            kind="stencil",
            label=label,
        )


@register("workload", "gemm", "summa")
@dataclass(frozen=True)
class Summa(Workload):
    """Dense ``n`` by ``n`` GEMM by SUMMA on a ``p`` by ``p`` process grid.

    Each task owns one C block and, per step, receives an A panel from its row
    and a B panel from its column.  Communication therefore lives on the rows
    and columns of the process mesh, never on its diagonals -- a structure B5's
    b-matching can exploit, since a task needs at most two distinct beam groups.
    """

    n: int = 1024
    p: int = 4

    def __post_init__(self) -> None:
        if self.p <= 0:
            raise ValueError(f"process grid p must be positive, got {self.p}")

    def build(self) -> TaskGraph:
        rows, cols = _split(self.n, self.p), _split(self.n, self.p)
        flop, ws, label = [], [], []
        for j, h in enumerate(rows):
            for i, w in enumerate(cols):
                block = h * w
                # 2n FLOP per output element, accumulated over the full k range.
                flop.append(2.0 * block * self.n)
                # A block, B block, C block.
                ws.append(3.0 * block * self.dtype_bytes)
                label.append(f"C[{i},{j}]")

        def tid(i: int, j: int) -> int:
            return j * self.p + i

        src, dst, pay = [], [], []
        for j, h in enumerate(rows):
            for i, w in enumerate(cols):
                # Row peers carry A panels, column peers carry B panels.
                for i2 in range(i + 1, self.p):
                    src.append(tid(i, j)); dst.append(tid(i2, j))
                    pay.append(h * cols[i] * self.dtype_bytes)
                for j2 in range(j + 1, self.p):
                    src.append(tid(i, j)); dst.append(tid(i, j2))
                    pay.append(w * rows[j] * self.dtype_bytes)

        return TaskGraph.from_arrays(
            flop, ws, src, dst, pay,
            name=f"summa[n={self.n},p={self.p}]",
            iterations=1,
            kind="gemm",
            label=label,
        )


@register("workload", "fft", "butterfly")
@dataclass(frozen=True)
class FFTButterfly(Workload):
    """Radix-2 FFT of ``n`` points over ``p`` tasks (both powers of two).

    Only the ``log2 p`` outermost stages cross task boundaries; the rest are
    local.  Each crossing stage pairs task ``i`` with ``i XOR 2^k``, so the
    communication graph is a hypercube -- degree ``log2 p``, and *every* stage
    needs a different partner.  On a fabric where re-aiming costs ~0.5 s that
    is the hard case, and it is here to keep B6 honest.
    """

    n: int = 1 << 20
    p: int = 8
    #: Complex float32 pair.
    bytes_per_point: int = 8

    def __post_init__(self) -> None:
        for name, v in (("n", self.n), ("p", self.p)):
            if v <= 0 or v & (v - 1):
                raise ValueError(f"{name} must be a positive power of two, got {v}")
        if self.p > self.n:
            raise ValueError(f"p={self.p} exceeds n={self.n}")

    def build(self) -> TaskGraph:
        per = self.n // self.p
        stages = int(math.log2(self.n))
        # 5 FLOP per point per stage is the standard radix-2 butterfly count.
        flop = [5.0 * per * stages] * self.p
        ws = [float(per * self.bytes_per_point * 2)] * self.p  # in and out buffers
        label = [f"rank{i}" for i in range(self.p)]

        src, dst, pay = [], [], []
        crossing = int(math.log2(self.p))
        for k in range(crossing):
            partner_stride = 1 << k
            for i in range(self.p):
                j = i ^ partner_stride
                if i < j:
                    src.append(i); dst.append(j)
                    # Half the local points move each crossing stage.
                    pay.append(per // 2 * self.bytes_per_point)

        return TaskGraph.from_arrays(
            flop, ws, src, dst, pay,
            name=f"fft[n={self.n},p={self.p}]",
            iterations=1,
            kind="fft",
            label=label,
        )


@register("workload", "allreduce")
@dataclass(frozen=True)
class AllReduce(Workload):
    """All-reduce of a ``count``-element vector over ``p`` tasks.

    ``topology`` picks ``ring`` (p edges, 2(p-1) steps, bandwidth-optimal) or
    ``tree`` (p-1 edges, 2 log2 p steps, latency-optimal).  On this substrate
    the trade is not the textbook one: steps cost *aiming*, so the tree's
    shallower depth matters far more than its worse bandwidth term.
    """

    count: int = 1 << 16
    p: int = 16
    topology: str = "ring"

    def build(self) -> TaskGraph:
        payload_bytes = self.count * self.dtype_bytes
        if self.topology == "ring":
            chunk = payload_bytes / max(self.p, 1)
            steps = 2 * (self.p - 1)
            src = list(range(self.p))
            dst = [(i + 1) % self.p for i in range(self.p)]
            pay = [chunk * steps / 2] * self.p
        elif self.topology == "tree":
            src = [(i - 1) // 2 for i in range(1, self.p)]
            dst = list(range(1, self.p))
            pay = [2.0 * payload_bytes] * (self.p - 1)  # up then down
        else:
            raise ValueError(f"unknown all-reduce topology {self.topology!r}")

        # The reduction itself is one op per element per participant.
        flop = [float(self.count)] * self.p
        ws = [float(payload_bytes * 2)] * self.p
        label = [f"rank{i}" for i in range(self.p)]
        # A 2-rank ring degenerates to a single edge; drop the duplicate.
        if self.topology == "ring" and self.p == 2:
            src, dst, pay = [0], [1], [pay[0] * 2]

        return TaskGraph.from_arrays(
            flop, ws, src, dst, pay,
            name=f"allreduce[{self.topology},p={self.p}]",
            iterations=1,
            kind="collective",
            label=label,
        )
