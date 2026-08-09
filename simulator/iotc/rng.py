"""Reproducible randomness.

A fault campaign is only useful if a failing run can be replayed exactly, so
every stochastic subsystem draws from its own named substream rather than from
one shared generator.  That way adding turbulence to a run does not shift the
packet-loss draws, and two configurations differing in one plugin stay
comparable.

    src = RandomSource(seed=7)
    src.stream("turbulence").normal(size=100)
    src.stream("faults/node/17").random()

Labels are hashed with blake2b rather than :func:`hash`, which is salted per
process and would destroy reproducibility across runs.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _label_entropy(label: str) -> int:
    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


class RandomSource:
    """A tree of independent, named, reproducible generators."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self._streams: dict[str, np.random.Generator] = {}

    def stream(self, label: str) -> np.random.Generator:
        """The generator for ``label``, created on first use and then cached."""
        gen = self._streams.get(label)
        if gen is None:
            seq = np.random.SeedSequence(entropy=[self.seed, _label_entropy(label)])
            gen = np.random.default_rng(seq)
            self._streams[label] = gen
        return gen

    def child(self, prefix: str) -> "NamedSource":
        """A view whose labels are all prefixed -- handy inside a subsystem."""
        return NamedSource(self, prefix)

    def reset(self) -> None:
        """Drop every cached generator so the next draw restarts the sequence."""
        self._streams.clear()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"RandomSource(seed={self.seed}, streams={len(self._streams)})"


class NamedSource:
    """A prefixed view onto a :class:`RandomSource`."""

    __slots__ = ("_root", "_prefix")

    def __init__(self, root: RandomSource, prefix: str) -> None:
        self._root = root
        self._prefix = prefix.rstrip("/")

    def stream(self, label: str = "") -> np.random.Generator:
        full = f"{self._prefix}/{label}" if label else self._prefix
        return self._root.stream(full)

    def child(self, prefix: str) -> "NamedSource":
        return NamedSource(self._root, f"{self._prefix}/{prefix}")
