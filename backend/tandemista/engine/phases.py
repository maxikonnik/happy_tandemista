from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .signals import SignalSeries

FREEFALL_VSPEED = -35.0   # m/s
CANOPY_VSPEED = -15.0     # deployment complete when slower than this
LANDED_VSPEED = 2.5       # |v| below this means on the ground
MIN_FREEFALL_S = 5.0
DEPLOY_SPIKE_G = 2.5
SMOOTH_WINDOW = 5         # seconds, centered moving average


class PhaseName(StrEnum):
    INTERVIEW = "interview"
    BOARDING = "boarding"
    CLIMB = "climb"
    EXIT = "exit"
    FREEFALL = "freefall"
    DEPLOYMENT = "deployment"
    CANOPY = "canopy"
    LANDING = "landing"
    AFTER = "after"


@dataclass(frozen=True)
class Phase:
    name: PhaseName
    start: float
    end: float
    confidence: float
    source: str  # "telemetry" | "audio" | "cv" | "role"


def _smooth(samples: list, window: int = SMOOTH_WINDOW) -> list:
    """Centered moving average; keeps Sample type and timestamps."""
    from .signals import Sample

    half = window // 2
    out = []
    for i in range(len(samples)):
        lo, hi = max(0, i - half), min(len(samples), i + half + 1)
        mean = sum(p.value for p in samples[lo:hi]) / (hi - lo)
        out.append(Sample(samples[i].t, mean))
    return out


def detect_phases_from_telemetry(
    vspeed: SignalSeries, accel: SignalSeries | None = None
) -> list[Phase]:
    s = _smooth(vspeed.resample(1.0).samples)
    # freefall: first sustained run of v <= FREEFALL_VSPEED
    ff_start = ff_end = None
    run_start = None
    for p in s:
        if p.value <= FREEFALL_VSPEED:
            run_start = p.t if run_start is None else run_start
            if p.t - run_start >= MIN_FREEFALL_S:
                ff_start = run_start
        elif run_start is not None and ff_start is not None:
            ff_end = p.t
            break
        else:
            run_start = None
    if ff_start is None:
        return []
    if ff_end is None:
        ff_end = s[-1].t

    # deployment confirmed by accel spike near freefall end
    conf = 0.8
    if accel is not None:
        window = [
            p.value for p in accel.resample(1.0).samples if ff_end - 3 <= p.t <= ff_end + 5
        ]
        if window and max(window) >= DEPLOY_SPIKE_G:
            conf = 0.95

    # landing: first sustained |v| < LANDED_VSPEED after deployment
    land_t = None
    calm_start = None
    for p in s:
        if p.t <= ff_end + 3:
            continue
        if abs(p.value) < LANDED_VSPEED:
            calm_start = p.t if calm_start is None else calm_start
            if p.t - calm_start >= 5.0:
                land_t = calm_start
                break
        else:
            calm_start = None
    canopy_end = land_t if land_t is not None else s[-1].t

    phases = [
        Phase(PhaseName.CLIMB, s[0].t, ff_start, conf, "telemetry"),
        Phase(PhaseName.EXIT, ff_start, ff_start, conf, "telemetry"),
        Phase(PhaseName.FREEFALL, ff_start, ff_end, conf, "telemetry"),
        Phase(PhaseName.DEPLOYMENT, ff_end, ff_end, conf, "telemetry"),
        Phase(PhaseName.CANOPY, ff_end, canopy_end, conf, "telemetry"),
    ]
    if land_t is not None:
        phases.append(Phase(PhaseName.LANDING, land_t, min(land_t + 15.0, s[-1].t), conf, "telemetry"))
    return phases
