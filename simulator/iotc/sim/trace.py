"""Event log and metrics.

The trace is deliberately cheap to write and structured enough to answer the
questions the testbench asks: what fraction of traffic went optical, how much
time went into retargeting rather than sending, when did a node brown out, how
far did the formation get from where the compiler wanted it.

Sampling is on by default because a 10 000-device run over minutes of simulated
time produces far more events than anyone needs; counters are always exact, only
the per-event records are subsampled.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class Event:
    time: float
    kind: str
    device: int | None = None
    peer: int | None = None
    detail: dict = field(default_factory=dict)


class Trace:
    """Counters, time series and a subsampled event log."""

    def __init__(self, max_events: int = 200_000, sample_every: float = 0.0) -> None:
        self.events: list[Event] = []
        self.counters: Counter = Counter()
        self.sums: defaultdict[str, float] = defaultdict(float)
        self.series: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)
        self.max_events = int(max_events)
        self.sample_every = float(sample_every)
        self._last_sample: dict[str, float] = {}
        self.dropped_events = 0

    # ------------------------------------------------------------------
    def event(self, time: float, kind: str, device=None, peer=None, **detail) -> None:
        self.counters[kind] += 1
        if len(self.events) >= self.max_events:
            self.dropped_events += 1
            return
        self.events.append(Event(time, kind, device, peer, detail))

    def count(self, key: str, n: int = 1) -> None:
        self.counters[key] += n

    def add(self, key: str, value: float) -> None:
        self.sums[key] += value

    def sample(self, time: float, key: str, value: float) -> None:
        """Record a time series point, honouring ``sample_every``."""
        if self.sample_every > 0.0:
            last = self._last_sample.get(key, -math.inf)
            if time - last < self.sample_every:
                return
            self._last_sample[key] = time
        self.series[key].append((time, float(value)))

    # ------------------------------------------------------------------
    def of_kind(self, kind: str) -> list[Event]:
        return [e for e in self.events if e.kind == kind]

    def summary(self) -> dict:
        out = {f"count.{k}": v for k, v in sorted(self.counters.items())}
        out.update({f"sum.{k}": v for k, v in sorted(self.sums.items())})
        if self.dropped_events:
            out["count.trace_dropped"] = self.dropped_events
        return out

    def report(self) -> str:  # pragma: no cover - reporting aid
        lines = []
        for k, v in sorted(self.counters.items()):
            lines.append(f"  {k:34s} {v:12d}")
        for k, v in sorted(self.sums.items()):
            lines.append(f"  {k:34s} {v:12.6g}")
        return "\n".join(lines)

    def to_csv(self, path) -> None:
        """Write the event log as tidy CSV."""
        import csv

        keys: list[str] = []
        for e in self.events:
            for k in e.detail:
                if k not in keys:
                    keys.append(k)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["time", "kind", "device", "peer", *keys])
            for e in self.events:
                w.writerow(
                    [f"{e.time:.9g}", e.kind, e.device, e.peer]
                    + [e.detail.get(k, "") for k in keys]
                )

    def series_to_csv(self, path) -> None:
        """Write every time series as one long-format CSV."""
        import csv

        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["series", "time", "value"])
            for name, points in sorted(self.series.items()):
                for t, v in points:
                    w.writerow([name, f"{t:.9g}", f"{v:.9g}"])
