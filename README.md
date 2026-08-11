# happy_tandemista

Automatic collection, matching, editing and delivery of tandem skydive videos.

- Spec: docs/superpowers/specs/2026-08-11-happy-tandemista-design.md
- Engine (this stage): backend/tandemista/engine — headless analyze+cut pipeline.

## Usage (engine CLI)
    tandemista /path/to/jump_dir --out /tmp/cuts
    # file naming: interview_*.mp4, handcam_*.mp4, outside_*.mp4, landing_*.mp4
    # local CV runs always; --no-cv skips it for debugging

## Dev setup
    cd backend
    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    pytest
Requires ffmpeg/ffprobe in PATH.
