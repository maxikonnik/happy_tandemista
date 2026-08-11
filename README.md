# happy_tandemista

Automatic collection, matching, editing and delivery of tandem skydive videos.

- Spec: docs/superpowers/specs/2026-08-11-happy-tandemista-design.md
- Engine (this stage): backend/tandemista/engine — headless analyze+cut pipeline.

## Usage (engine CLI)
    tandemista /path/to/jump_dir --out /tmp/cuts
    # file naming: interview_*.mp4, handcam_*.mp4, outside_*.mp4, landing_*.mp4
    # (also matches uppercase .MP4 extensions from GoPro cameras)
    # local CV runs always; --no-cv skips it for debugging
    # Note: existing .mp4 files in --out directory will be overwritten

## Known limitations of this stage

The engine is proven on synthetic footage only. Before pointing it at a real jump day,
know what it does not yet handle — each item is a task for a follow-up plan:

- **GoPro telemetry timing is approximated.** Sample timestamps are derived from a fixed
  packet rate rather than the `STMP`/`TSMP` pacing GoPro records, so on a long recording the
  drift grows into tens of seconds. There is also no GPS-fix check, so garbage altitude
  during the climb can fabricate a freefall, and HERO11+ cameras write `GPS9` instead of
  `GPS5`, for which telemetry extraction currently returns nothing.
- **One jump per file, one file per role.** Clock offsets are placeholders until the
  matching stage lands. A card holding several jumps renders an arbitrary one, and a
  recording split into chapters by the camera (any file over 4 GB) can cut footage from the
  wrong chapter. Trim and rename per jump before running.
- **Not tuned for large files.** Frame analysis decodes every frame at full resolution, and
  the renderer trims with a filter rather than seeking, so an hour of 4K60 takes far longer
  than a jump day allows.
- **The CV fallback assumes visible sky.** Phase heuristics lean on blue, bright pixels;
  overcast conditions weaken them. Telemetry and audio carry the load when present.

## Dev setup
    cd backend
    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    pytest
Requires ffmpeg/ffprobe in PATH.
