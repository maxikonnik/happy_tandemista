from __future__ import annotations

from dataclasses import dataclass

from .phases import PhaseName


@dataclass(frozen=True)
class Slot:
    phase: PhaseName
    min_s: float
    max_s: float
    required: bool = False
    prefer_roles: tuple[str, ...] = ()
    lead_in: float = 0.0
    lead_out: float = 0.0


@dataclass(frozen=True)
class Template:
    variant: str
    aspect: str  # "16:9" | "9:16"
    slots: tuple[Slot, ...]


TEMPLATES: dict[str, Template] = {
    "full_16x9": Template(
        "full_16x9", "16:9",
        (
            Slot(PhaseName.INTERVIEW, 15, 25, prefer_roles=("ground_interview",)),
            Slot(PhaseName.CLIMB, 5, 10, prefer_roles=("handcam",)),
            Slot(PhaseName.EXIT, 5, 13, required=True, lead_in=3, lead_out=10,
                 prefer_roles=("outside", "handcam")),
            Slot(PhaseName.FREEFALL, 30, 60, prefer_roles=("outside", "handcam")),
            Slot(PhaseName.CANOPY, 5, 15, prefer_roles=("handcam",)),
            Slot(PhaseName.LANDING, 5, 15, prefer_roles=("ground_landing", "outside")),
        ),
    ),
    "emotions_16x9": Template(
        "emotions_16x9", "16:9",
        (
            Slot(PhaseName.INTERVIEW, 5, 10, prefer_roles=("ground_interview",)),
            Slot(PhaseName.EXIT, 5, 13, required=True, lead_in=3, lead_out=10,
                 prefer_roles=("outside", "handcam")),
            Slot(PhaseName.FREEFALL, 15, 25, prefer_roles=("handcam", "outside")),
            Slot(PhaseName.DEPLOYMENT, 3, 6, lead_out=4, prefer_roles=("handcam",)),
            Slot(PhaseName.LANDING, 4, 8, prefer_roles=("ground_landing",)),
        ),
    ),
    # Same beats and lengths as emotions_16x9, reframed vertically. The exit and
    # freefall slots reach for the handcam first: a 9:16 crop of the outside
    # operator's wide shot throws away the sides, and the faces are the point here.
    "emotions_9x16": Template(
        "emotions_9x16", "9:16",
        (
            Slot(PhaseName.INTERVIEW, 5, 10, prefer_roles=("ground_interview",)),
            Slot(PhaseName.EXIT, 5, 13, required=True, lead_in=3, lead_out=10,
                 prefer_roles=("handcam", "outside")),
            Slot(PhaseName.FREEFALL, 15, 25, prefer_roles=("handcam", "outside")),
            Slot(PhaseName.DEPLOYMENT, 3, 6, lead_out=4, prefer_roles=("handcam",)),
            Slot(PhaseName.LANDING, 4, 8, prefer_roles=("ground_landing",)),
        ),
    ),
    "highlights_9x16": Template(
        "highlights_9x16", "9:16",
        (
            Slot(PhaseName.EXIT, 3, 5, required=True, lead_in=1, lead_out=4,
                 prefer_roles=("outside", "handcam")),
            Slot(PhaseName.FREEFALL, 6, 8, prefer_roles=("handcam", "outside")),
            Slot(PhaseName.LANDING, 2, 4, prefer_roles=("ground_landing",)),
        ),
    ),
}
