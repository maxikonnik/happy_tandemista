from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engine.edl import SlotUnfillableError, generate_edl
from .engine.gpmf import telemetry_from_gopro
from .engine.media import extract_audio_rms, probe_duration
from .engine.phases import detect_phases_from_audio, detect_phases_from_telemetry
from .engine.render import render_edl
from .engine.templates import TEMPLATES
from .engine.timeline import SourceFile, build_timeline

ROLE_PREFIX = {
    "interview": "ground_interview",
    "handcam": "handcam",
    "outside": "outside",
    "landing": "ground_landing",
}
# MVP ordering offsets until real matching arrives (plan 3)
ROLE_OFFSET = {"ground_interview": -3600.0, "handcam": 0.0, "outside": 0.0,
               "ground_landing": 3600.0}


def analyze_file(path: Path, role: str, use_cv: bool = True) -> SourceFile:
    duration = probe_duration(path)
    phases = []
    if role in ("handcam", "outside"):
        telemetry = telemetry_from_gopro(path)
        if "vspeed_ms" in telemetry:
            phases = detect_phases_from_telemetry(telemetry["vspeed_ms"])
        if not phases:
            phases = detect_phases_from_audio(extract_audio_rms(path))
    moments = []
    if use_cv:
        from .engine.cv import LocalCVAnnotator

        ann = LocalCVAnnotator().annotate(path)
        moments = ann.moments
        if not phases:
            phases = ann.phases
    return SourceFile(path, role, duration, ROLE_OFFSET[role], phases, moments)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tandemista")
    parser.add_argument("jump_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-cv", action="store_true", help="skip local CV analysis (debug)")
    args = parser.parse_args(argv)

    files: list[SourceFile] = []
    for p in sorted(args.jump_dir.glob("*.mp4")):
        role = next(
            (r for pref, r in ROLE_PREFIX.items() if p.name.startswith(pref)), None
        )
        if role is None:
            print(f"skip (unknown role): {p.name}", file=sys.stderr)
            continue
        files.append(analyze_file(p, role, use_cv=not args.no_cv))
    if not files:
        print("no recognizable files in jump_dir", file=sys.stderr)
        return 1

    timeline = build_timeline(files)
    ok = 0
    for name, template in TEMPLATES.items():
        try:
            edl = generate_edl(timeline, template)
        except SlotUnfillableError as e:
            print(f"warn: {name} skipped: {e}", file=sys.stderr)
            continue
        out = render_edl(edl, args.out / f"{name}.mp4", height=args.height)
        print(f"rendered {out}")
        ok += 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
