from __future__ import annotations

import random

from .signals import Sample, SignalSeries

CLIMB_RATE = 8.0        # m/s up
TERMINAL = -50.0        # m/s freefall
CANOPY_RATE = -6.0      # m/s under canopy
EXIT_ALT = 4000.0


def make_jump_signals(
    exit_t: float = 300.0,
    freefall_s: float = 60.0,
    canopy_s: float = 240.0,
    noise: float = 0.0,
    seed: int = 0,
) -> dict[str, SignalSeries]:
    rng = random.Random(seed)
    deploy_t = exit_t + freefall_s
    land_t = deploy_t + canopy_s
    total = land_t + 30.0

    vs: list[Sample] = []
    acc: list[Sample] = []
    alt: list[Sample] = []
    altitude = EXIT_ALT - CLIMB_RATE * exit_t
    t = 0.0
    while t <= total:
        if t < exit_t:
            v = CLIMB_RATE
            a = 1.0
        elif t < exit_t + 5.0:            # accelerating after exit
            v = TERMINAL * (t - exit_t) / 5.0
            a = 0.3
        elif t < deploy_t:
            v = TERMINAL
            a = 1.0                        # drag at terminal reads ~1g
        elif t < deploy_t + 3.0:           # opening shock and deceleration
            v = TERMINAL + (CANOPY_RATE - TERMINAL) * (t - deploy_t) / 3.0
            a = 3.5
        elif t < land_t:
            v = CANOPY_RATE
            a = 1.0
        else:
            v = 0.0
            a = 1.0
        v += rng.gauss(0.0, noise)
        altitude = max(0.0, altitude + v)
        vs.append(Sample(t, v))
        acc.append(Sample(t, a))
        alt.append(Sample(t, altitude))
        t += 1.0

    return {
        "vspeed_ms": SignalSeries("vspeed_ms", vs),
        "altitude_m": SignalSeries("altitude_m", alt),
        "accel_g": SignalSeries("accel_g", acc),
    }
