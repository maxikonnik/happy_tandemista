from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SKY_MIN_BRIGHTNESS = 0.45
FLOW_PYR_SCALE = 0.5
FLOW_LEVELS = 3
FLOW_WINSIZE = 15
FLOW_ITERATIONS = 3
FLOW_POLY_N = 5
FLOW_POLY_SIGMA = 1.2


@dataclass(frozen=True)
class FrameFeatures:
    """Weight-free visual measurements of one sampled frame."""

    t: float
    sharpness: float
    motion: float
    brightness: float
    sky_ratio: float


def _sky_ratio(bgr: np.ndarray) -> float:
    b = bgr[:, :, 0].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    value = bgr.max(axis=2).astype(np.float64) / 255.0
    sky = (b >= r) & (value > SKY_MIN_BRIGHTNESS)
    return float(sky.mean())


def extract_frame_features(
    video: Path, fps: float = 0.5, width: int = 320
) -> list[FrameFeatures]:
    """Sample the video at `fps` and measure each frame. No model weights involved."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if src_fps <= 0:
            src_fps = 30.0
        stride = max(1, int(round(src_fps / fps)))
        out: list[FrameFeatures] = []
        prev_gray: np.ndarray | None = None
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride == 0:
                scale = width / frame.shape[1]
                small = cv2.resize(frame, (width, max(1, int(frame.shape[0] * scale))))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                brightness = float(gray.mean()) / 255.0
                motion = 0.0
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None,
                        FLOW_PYR_SCALE, FLOW_LEVELS, FLOW_WINSIZE,
                        FLOW_ITERATIONS, FLOW_POLY_N, FLOW_POLY_SIGMA, 0,
                    )
                    motion = float(np.linalg.norm(flow, axis=2).mean())
                out.append(
                    FrameFeatures(
                        t=index / src_fps,
                        sharpness=sharpness,
                        motion=motion,
                        brightness=brightness,
                        sky_ratio=_sky_ratio(small),
                    )
                )
                prev_gray = gray
            index += 1
        return out
    finally:
        cap.release()
