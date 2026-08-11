from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    t: float
    value: float


@dataclass(frozen=True)
class SignalSeries:
    """Time series of one scalar signal; t is seconds from file start."""

    name: str
    samples: list[Sample]

    @property
    def t0(self) -> float:
        return self.samples[0].t

    @property
    def t1(self) -> float:
        return self.samples[-1].t

    def value_at(self, t: float) -> float | None:
        if not self.samples or t < self.t0 or t > self.t1:
            return None
        lo, hi = 0, len(self.samples) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.samples[mid].t <= t:
                lo = mid
            else:
                hi = mid
        a, b = self.samples[lo], self.samples[hi]
        if b.t == a.t:
            return a.value
        k = (t - a.t) / (b.t - a.t)
        return a.value + k * (b.value - a.value)

    def resample(self, step: float) -> SignalSeries:
        out: list[Sample] = []
        t = self.t0
        while t <= self.t1 + 1e-9:
            v = self.value_at(min(t, self.t1))
            assert v is not None
            out.append(Sample(round(t, 6), v))
            t += step
        return SignalSeries(self.name, out)
