"""Workload frontends -- the ``workload`` plugin slot.

A frontend turns problem parameters into a :class:`~iotc.compiler.ir.TaskGraph`.
It is the third pluggable axis of the package, and it follows the same rules as
the other two: subclass, decorate, and the workload becomes selectable by name
from a config file or the CLI.

    from iotc.registry import register
    from iotc.compiler.frontends.base import Workload

    @register("workload", "my_problem")
    class MyProblem(Workload):
        def build(self) -> TaskGraph:
            ...

Every frontend must be **seed-deterministic**: two builds with the same
parameters produce identical graphs, edge order included.  The simulator holds
determinism as a correctness requirement rather than a convenience, and a
compiler that placed the same problem differently on two runs would make every
downstream fault campaign unreproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from ..ir import TaskGraph


@dataclass(frozen=True)
class Workload:
    """Base class for problem frontends.

    Subclasses are frozen dataclasses of parameters plus a :meth:`build`.
    Keeping them pure data means a workload can be written into a config,
    round-tripped, and compared between runs.
    """

    #: Bytes per scalar.  float32 throughout; these devices have no FPU to
    #: spare and doubling the payload doubles the aiming, not just the wire.
    dtype_bytes: int = 4

    def build(self) -> TaskGraph:  # pragma: no cover - interface
        raise NotImplementedError

    def params(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def describe(self) -> str:
        inner = ", ".join(f"{k}={v}" for k, v in self.params().items())
        return f"{type(self).plugin_name}({inner})"


def _split(total: int, parts: int) -> list[int]:
    """Split ``total`` items into ``parts`` as evenly as possible.

    Deterministic and exact: the first ``total % parts`` chunks get one extra,
    and the sizes always sum back to ``total``.  Load imbalance on this fabric
    is not a rounding detail -- an over-full task is one that fails
    :meth:`TaskGraph.fits` and forces another tiling round in B1.
    """
    if parts <= 0:
        raise ValueError(f"parts must be positive, got {parts}")
    base, extra = divmod(total, parts)
    return [base + 1] * extra + [base] * (parts - extra)
