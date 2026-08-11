# Движок анализа и автомонтажа (headless) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Headless-ядро happy_tandemista: из папки с файлами прыжка (handcam/outside/интервью/приземление) построить таймлайн фаз по телеметрии+аудио+CV и отрендерить три варианта монтажа (full 16:9, emotions 16:9, highlights 9:16) через CLI.

**Architecture:** Чистый Python-пакет `tandemista.engine` без БД и без сети — весь анализ, включая CV, идёт локально. Конвейер: извлечение сигналов (GPMF-телеметрия, аудио-RMS, CV-разметка) → детект фаз → фьюжн в общий таймлайн → EDL по декларативным шаблонам → ffmpeg-рендер. CLI связывает всё. Облачный каркас (FastAPI/Celery) подключит этот пакет в следующем плане.

**Tech Stack:** Python 3.12+, numpy, opencv-python-headless (свой CV), pytest; системные ffmpeg/ffprobe. Полная спека: `docs/superpowers/specs/2026-08-11-happy-tandemista-design.md`.

## Global Constraints

- Python ≥ 3.12; зависимости backend только: `numpy`, `opencv-python-headless`, dev: `pytest` (новые — только через обсуждение).
- `ffmpeg` и `ffprobe` должны быть в PATH; проверять через `shutil.which` с понятной ошибкой.
- Код, имена и комментарии — на английском; тексты для пользователей — позже, не в этом плане.
- Все публичные функции — с type hints; dataclasses — `frozen=True` где нет мутаций.
- Коммиты — conventional commits (`feat:`, `test:`, `chore:`).
- Времена в сигналах и фазах — секунды `float` от начала файла; общий таймлайн — секунды от общего нуля прыжка.
- **CV — свой локальный конвейер, БЕЗ вызовов LLM на инференсе** (требование пользователя). Никаких обращений к vision-API из кода анализа; `anthropic` из зависимостей уходит.
- CV-инференс CPU-first: модели лёгкие, GPU не требуется.
- Веса моделей НЕ хранятся в git: качаются скриптом `backend/scripts/fetch_models.py` в `backend/models/` (папка в .gitignore). Тесты, требующие весов, скипаются при их отсутствии.
- Тесты, требующие реальных съёмок GoPro, ищут файлы в `backend/tests/samples/` и скипаются, если их нет (`pytest.mark.skipif`).

---

## File Structure

```
backend/
  pyproject.toml
  tandemista/
    __init__.py
    engine/
      __init__.py
      signals.py     # Sample, SignalSeries, resample
      phases.py      # PhaseName, Phase, детекторы (телеметрия, аудио)
      synthetic.py   # генератор синтетических сигналов прыжка (для тестов и демо)
      media.py       # обёртки ffmpeg/ffprobe: длительность, извлечение аудио, кадров, gpmd-потока
      gpmf.py        # минимальный KLV-парсер GPMF, GPS5 → SignalSeries
      frames.py      # FrameFeatures, extract_frame_features (OpenCV, без весов)
      cv.py          # CVAnnotator (Protocol), LocalCVAnnotator, эвристики фаз/моментов, типы
      timeline.py    # SourceFile, JumpTimeline, build_timeline (фьюжн)
      templates.py   # Slot, Template, три встроенных шаблона
      edl.py         # Clip, EDL, generate_edl
      render.py      # render_edl через ffmpeg
    cli.py           # tandemista analyze-and-cut
  tests/
    engine/ (зеркалит модули)
    samples/         # (в .gitignore, кладёт пользователь) реальные GoPro-файлы
```

Каждый модуль — одна ответственность; порядок задач следует зависимостям.

---

### Task 1: Скелет репозитория и backend-пакета

**Files:**
- Create: `backend/pyproject.toml`, `backend/tandemista/__init__.py`, `backend/tandemista/engine/__init__.py`, `backend/tests/engine/test_sanity.py`, `.gitignore`, `README.md`

**Interfaces:**
- Produces: устанавливаемый пакет `tandemista`, работающий `pytest`.

- [ ] **Step 1: Создать файлы**

`backend/pyproject.toml`:
```toml
[project]
name = "tandemista"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["numpy>=1.26", "opencv-python-headless>=4.10"]

[project.optional-dependencies]
dev = ["pytest>=8"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["tandemista*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore`:
```
__pycache__/
*.egg-info/
.venv/
backend/tests/samples/
.pytest_cache/
node_modules/
.next/
dist/
```

`README.md`:
```markdown
# happy_tandemista

Automatic collection, matching, editing and delivery of tandem skydive videos.

- Spec: docs/superpowers/specs/2026-08-11-happy-tandemista-design.md
- Engine (this stage): backend/tandemista/engine — headless analyze+cut pipeline.

## Dev setup
    cd backend
    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    pytest
Requires ffmpeg/ffprobe in PATH.
```

`backend/tests/engine/test_sanity.py`:
```python
import tandemista


def test_package_imports():
    assert tandemista is not None
```

Оба `__init__.py` — пустые.

- [ ] **Step 2: Установить и прогнать тесты**

Run: `cd backend && python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]" && pytest -q`
Expected: `1 passed`

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "chore: repo skeleton, backend package, spec and plan docs"
```

---

### Task 2: Модель сигналов (SignalSeries)

**Files:**
- Create: `backend/tandemista/engine/signals.py`
- Test: `backend/tests/engine/test_signals.py`

**Interfaces:**
- Produces: `Sample(t: float, value: float)`; `SignalSeries(name: str, samples: list[Sample])` с методами `resample(step: float) -> SignalSeries`, `value_at(t: float) -> float | None`, свойствами `t0`/`t1`. Все детекторы дальше принимают `SignalSeries`.

- [ ] **Step 1: Написать падающий тест**

```python
# backend/tests/engine/test_signals.py
from tandemista.engine.signals import Sample, SignalSeries


def make(name, pairs):
    return SignalSeries(name, [Sample(t, v) for t, v in pairs])


def test_value_at_interpolates_and_none_outside():
    s = make("v", [(0.0, 0.0), (10.0, 10.0)])
    assert s.value_at(5.0) == 5.0
    assert s.value_at(-1.0) is None
    assert s.value_at(11.0) is None


def test_resample_uniform_grid():
    s = make("v", [(0.0, 0.0), (4.0, 8.0)])
    r = s.resample(2.0)
    assert [(p.t, p.value) for p in r.samples] == [(0.0, 0.0), (2.0, 4.0), (4.0, 8.0)]


def test_bounds():
    s = make("v", [(1.5, 0.0), (9.0, 1.0)])
    assert (s.t0, s.t1) == (1.5, 9.0)
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `cd backend && pytest tests/engine/test_signals.py -q`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Реализация**

```python
# backend/tandemista/engine/signals.py
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
```

- [ ] **Step 4: Тесты зелёные**

Run: `cd backend && pytest tests/engine/test_signals.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/tandemista/engine/signals.py backend/tests/engine/test_signals.py
git commit -m "feat: SignalSeries time-series model"
```

---

### Task 3: Синтетический генератор сигналов прыжка

**Files:**
- Create: `backend/tandemista/engine/synthetic.py`
- Test: `backend/tests/engine/test_synthetic.py`

**Interfaces:**
- Consumes: `SignalSeries`, `Sample` из Task 2.
- Produces: `make_jump_signals(exit_t: float = 300.0, freefall_s: float = 60.0, canopy_s: float = 240.0, noise: float = 0.0, seed: int = 0) -> dict[str, SignalSeries]` с ключами `"vspeed_ms"`, `"altitude_m"`, `"accel_g"` (шаг 1 с). Используется тестами детектора и демо.

- [ ] **Step 1: Падающий тест**

```python
# backend/tests/engine/test_synthetic.py
from tandemista.engine.synthetic import make_jump_signals


def test_shape_and_phases_present():
    sig = make_jump_signals(exit_t=300, freefall_s=60, canopy_s=240)
    v = sig["vspeed_ms"]
    assert v.value_at(100.0) > -5.0            # climb: почти нет вертикальной скорости вниз
    assert v.value_at(330.0) < -45.0           # terminal freefall
    assert -15.0 < v.value_at(450.0) < -3.0    # canopy
    assert abs(v.value_at(620.0)) < 1.0        # landed
    alt = sig["altitude_m"]
    assert alt.value_at(299.0) > 3500.0
    assert alt.value_at(620.0) < 50.0
    acc = sig["accel_g"]
    assert max(p.value for p in acc.samples if 360 <= p.t <= 366) > 2.5  # deployment spike
```

- [ ] **Step 2: Убедиться, что падает**

Run: `cd backend && pytest tests/engine/test_synthetic.py -q` — FAIL (ModuleNotFoundError)

- [ ] **Step 3: Реализация**

```python
# backend/tandemista/engine/synthetic.py
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
```

- [ ] **Step 4: Тесты зелёные**

Run: `cd backend && pytest tests/engine/test_synthetic.py -q` — `1 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/tandemista/engine/synthetic.py backend/tests/engine/test_synthetic.py
git commit -m "feat: synthetic jump signal generator for tests and demos"
```

---

### Task 4: Детектор фаз по телеметрии

**Files:**
- Create: `backend/tandemista/engine/phases.py`
- Test: `backend/tests/engine/test_phases_telemetry.py`

**Interfaces:**
- Consumes: `SignalSeries` (Task 2), `make_jump_signals` (Task 3).
- Produces: `PhaseName` (StrEnum: `INTERVIEW, BOARDING, CLIMB, EXIT, FREEFALL, DEPLOYMENT, CANOPY, LANDING, AFTER`), `Phase(name: PhaseName, start: float, end: float, confidence: float, source: str)`, `detect_phases_from_telemetry(vspeed: SignalSeries, accel: SignalSeries | None = None) -> list[Phase]`. EXIT — точечная фаза (start==end), DEPLOYMENT — тоже.

- [ ] **Step 1: Падающие тесты**

```python
# backend/tests/engine/test_phases_telemetry.py
import pytest

from tandemista.engine.phases import PhaseName, detect_phases_from_telemetry
from tandemista.engine.synthetic import make_jump_signals


def by_name(phases, name):
    return next(p for p in phases if p.name == name)


def test_clean_jump_phases():
    sig = make_jump_signals(exit_t=300, freefall_s=60, canopy_s=240)
    phases = detect_phases_from_telemetry(sig["vspeed_ms"], sig["accel_g"])
    exit_p = by_name(phases, PhaseName.EXIT)
    assert exit_p.start == pytest.approx(300, abs=6)
    ff = by_name(phases, PhaseName.FREEFALL)
    assert ff.start == pytest.approx(300, abs=6)
    assert ff.end == pytest.approx(360, abs=6)
    dep = by_name(phases, PhaseName.DEPLOYMENT)
    assert dep.start == pytest.approx(360, abs=6)
    canopy = by_name(phases, PhaseName.CANOPY)
    assert canopy.end == pytest.approx(600, abs=10)
    landing = by_name(phases, PhaseName.LANDING)
    assert landing.start == pytest.approx(600, abs=10)
    assert ff.confidence >= 0.9  # accel spike confirms deployment


def test_noisy_jump_still_detected():
    sig = make_jump_signals(noise=3.0, seed=42)
    phases = detect_phases_from_telemetry(sig["vspeed_ms"], sig["accel_g"])
    assert {p.name for p in phases} >= {PhaseName.FREEFALL, PhaseName.CANOPY, PhaseName.LANDING}


def test_no_freefall_returns_empty():
    sig = make_jump_signals()
    ground_only = sig["vspeed_ms"].__class__(
        "vspeed_ms", [s for s in sig["vspeed_ms"].samples if s.t < 200]
    )
    assert detect_phases_from_telemetry(ground_only) == []
```

- [ ] **Step 2: Убедиться, что падает** — `pytest tests/engine/test_phases_telemetry.py -q` → FAIL

- [ ] **Step 3: Реализация**

```python
# backend/tandemista/engine/phases.py
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
```

- [ ] **Step 4: Тесты зелёные** — `pytest tests/engine/test_phases_telemetry.py -q` → `3 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/tandemista/engine/phases.py backend/tests/engine/test_phases_telemetry.py
git commit -m "feat: telemetry-based jump phase detector"
```

---

### Task 5: Обёртки ffmpeg и аудио-профиль файла

**Files:**
- Create: `backend/tandemista/engine/media.py`
- Test: `backend/tests/engine/test_media.py`

**Interfaces:**
- Produces: `require_ffmpeg() -> None` (RuntimeError с внятным текстом, если нет ffmpeg/ffprobe); `probe_duration(path: Path) -> float`; `extract_audio_rms(path: Path, step: float = 1.0) -> SignalSeries` (имя `"audio_rms"`, значения 0..1 — RMS, нормированный на пик файла); `extract_frames(path: Path, out_dir: Path, fps: float = 1.0) -> list[Path]` (jpg-кадры, имена `frame_%06d.jpg`, индекс = секунда). Внутри — `subprocess.run` ffmpeg с decode в `s16le` 8kHz mono через pipe + numpy.

- [ ] **Step 1: Падающий тест** (тестовое медиа генерим самим ffmpeg — тон 440 Гц в первой половине, тишина во второй)

```python
# backend/tests/engine/test_media.py
import subprocess
from pathlib import Path

import pytest

from tandemista.engine.media import extract_audio_rms, extract_frames, probe_duration


@pytest.fixture(scope="module")
def tone_then_silence(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("media") / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=6:size=320x240:rate=10",
            "-f", "lavfi", "-i",
            "aevalsrc=if(lt(t\\,3)\\,sin(2*PI*440*t)\\,0):d=6",
            "-shortest", "-c:v", "libx264", "-c:a", "aac", str(out),
        ],
        check=True, capture_output=True,
    )
    return out


def test_probe_duration(tone_then_silence):
    assert probe_duration(tone_then_silence) == pytest.approx(6.0, abs=0.5)


def test_audio_rms_loud_then_quiet(tone_then_silence):
    rms = extract_audio_rms(tone_then_silence, step=1.0)
    assert rms.value_at(1.0) > 0.5
    assert rms.value_at(5.0) < 0.1


def test_extract_frames(tone_then_silence, tmp_path):
    frames = extract_frames(tone_then_silence, tmp_path, fps=1.0)
    assert len(frames) == pytest.approx(6, abs=1)
    assert frames[0].exists()
```

- [ ] **Step 2: Убедиться, что падает** — FAIL (ModuleNotFoundError)

- [ ] **Step 3: Реализация**

```python
# backend/tandemista/engine/media.py
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .signals import Sample, SignalSeries


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not found in PATH; install ffmpeg to use tandemista")


def probe_duration(path: Path) -> float:
    require_ffmpeg()
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def extract_audio_rms(path: Path, step: float = 1.0) -> SignalSeries:
    require_ffmpeg()
    rate = 8000
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn",
         "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"],
        check=True, capture_output=True,
    ).stdout
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    win = int(rate * step)
    samples: list[Sample] = []
    for i in range(0, len(pcm) - win + 1, win):
        rms = float(np.sqrt(np.mean(pcm[i : i + win] ** 2)))
        samples.append(Sample(i / rate, rms))
    peak = max((s.value for s in samples), default=1.0) or 1.0
    return SignalSeries("audio_rms", [Sample(s.t, s.value / peak) for s in samples])


def extract_frames(path: Path, out_dir: Path, fps: float = 1.0) -> list[Path]:
    require_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%06d.jpg"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-vf", f"fps={fps}", "-q:v", "4", str(pattern)],
        check=True, capture_output=True,
    )
    return sorted(out_dir.glob("frame_*.jpg"))
```

- [ ] **Step 4: Тесты зелёные** — `pytest tests/engine/test_media.py -q` → `3 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/tandemista/engine/media.py backend/tests/engine/test_media.py
git commit -m "feat: ffmpeg wrappers - duration, audio RMS profile, frame extraction"
```

---

### Task 6: Аудио-детектор фаз (fallback без телеметрии)

**Files:**
- Modify: `backend/tandemista/engine/phases.py` (добавить функцию в конец файла)
- Test: `backend/tests/engine/test_phases_audio.py`

**Interfaces:**
- Consumes: `SignalSeries` c именем `"audio_rms"` (Task 5).
- Produces: `detect_phases_from_audio(rms: SignalSeries) -> list[Phase]` — находит FREEFALL как самый длинный непрерывный интервал 20–120 с с rms выше порога, ставит EXIT в его начало, DEPLOYMENT/CANOPY после; confidence 0.6, source `"audio"`. Пусто, если такого интервала нет.

- [ ] **Step 1: Падающий тест**

```python
# backend/tests/engine/test_phases_audio.py
from tandemista.engine.phases import PhaseName, detect_phases_from_audio
from tandemista.engine.signals import Sample, SignalSeries


def rms_series(profile: list[tuple[int, int, float]], total: int) -> SignalSeries:
    values = [0.05] * total
    for start, end, level in profile:
        for t in range(start, end):
            values[t] = level
    return SignalSeries("audio_rms", [Sample(float(t), v) for t, v in enumerate(values)])


def test_wind_roar_detected_as_freefall():
    # speech at 10..40, wind roar 300..360, quiet canopy after
    rms = rms_series([(10, 40, 0.4), (300, 360, 0.95)], total=650)
    phases = detect_phases_from_audio(rms)
    ff = next(p for p in phases if p.name == PhaseName.FREEFALL)
    assert abs(ff.start - 300) <= 3
    assert abs(ff.end - 360) <= 3
    assert ff.source == "audio"
    assert any(p.name == PhaseName.CANOPY for p in phases)


def test_no_roar_no_phases():
    rms = rms_series([(10, 40, 0.4)], total=200)
    assert detect_phases_from_audio(rms) == []
```

- [ ] **Step 2: Убедиться, что падает** — FAIL (ImportError)

- [ ] **Step 3: Реализация** (добавить в `phases.py`)

```python
WIND_RMS = 0.7
MIN_FF_AUDIO_S = 20.0
MAX_FF_AUDIO_S = 120.0


def detect_phases_from_audio(rms: SignalSeries) -> list[Phase]:
    s = rms.resample(1.0).samples
    runs: list[tuple[float, float]] = []
    start = None
    for p in s:
        if p.value >= WIND_RMS:
            start = p.t if start is None else start
        elif start is not None:
            runs.append((start, p.t))
            start = None
    if start is not None:
        runs.append((start, s[-1].t))
    runs = [r for r in runs if MIN_FF_AUDIO_S <= r[1] - r[0] <= MAX_FF_AUDIO_S]
    if not runs:
        return []
    ff_start, ff_end = max(runs, key=lambda r: r[1] - r[0])
    conf = 0.6
    return [
        Phase(PhaseName.CLIMB, s[0].t, ff_start, conf, "audio"),
        Phase(PhaseName.EXIT, ff_start, ff_start, conf, "audio"),
        Phase(PhaseName.FREEFALL, ff_start, ff_end, conf, "audio"),
        Phase(PhaseName.DEPLOYMENT, ff_end, ff_end, conf, "audio"),
        Phase(PhaseName.CANOPY, ff_end, s[-1].t, conf, "audio"),
    ]
```

- [ ] **Step 4: Тесты зелёные** — `pytest tests/engine/test_phases_audio.py -q` → `2 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/tandemista/engine/phases.py backend/tests/engine/test_phases_audio.py
git commit -m "feat: audio-based freefall phase detector (no-telemetry fallback)"
```

---

### Task 7: GPMF-парсер и извлечение телеметрии GoPro

**Files:**
- Create: `backend/tandemista/engine/gpmf.py`
- Test: `backend/tests/engine/test_gpmf.py`

**Interfaces:**
- Consumes: `SignalSeries`; ffmpeg-извлечение через `subprocess` (по образцу `media.py`).
- Produces: `parse_gpmf(data: bytes) -> dict[bytes, list]` — минимальный KLV-парсер (структура GPMF: key 4 байта, type 1 байт, size 1 байт, repeat 2 байта BE; type `\x00` — вложенный контейнер; payload выровнен до 4 байт); `telemetry_from_gopro(path: Path) -> dict[str, SignalSeries]` — находит ffprobe'ом поток `gpmd`, извлекает `ffmpeg -codec copy -f data`, парсит `GPS5` (int32 ×5: lat, lon, alt, speed2d, speed3d) с делителями `SCAL`, возвращает `{"altitude_m": ..., "vspeed_ms": ...}` (vspeed — численная производная altitude, сглаженная окном 3 с); пустой dict, если потока нет.

- [ ] **Step 1: Падающий тест** (GPMF-байты собираем вручную `struct.pack` — формат открытый)

```python
# backend/tests/engine/test_gpmf.py
import struct
from pathlib import Path

import pytest

from tandemista.engine.gpmf import gps5_series, parse_gpmf


def klv(key: bytes, type_: bytes, size: int, repeat: int, payload: bytes) -> bytes:
    pad = (-len(payload)) % 4
    return key + type_ + bytes([size]) + struct.pack(">H", repeat) + payload + b"\x00" * pad


def make_stream(alts_m: list[float]) -> bytes:
    # SCAL for GPS5: divisors [1e7, 1e7, 1000, 1000, 100]
    scal = klv(b"SCAL", b"l", 4, 5, struct.pack(">5l", 10000000, 10000000, 1000, 1000, 100))
    gps_payload = b"".join(
        struct.pack(">5l", 0, 0, int(alt * 1000), 0, 0) for alt in alts_m
    )
    gps5 = klv(b"GPS5", b"l", 20, len(alts_m), gps_payload)
    strm = klv(b"STRM", b"\x00", 1, len(scal) + len(gps5), scal + gps5)
    return klv(b"DEVC", b"\x00", 1, len(strm), strm)


def test_parse_nested_and_scaled():
    data = make_stream([4000.0, 3950.0, 3900.0])
    tree = parse_gpmf(data)
    series = gps5_series(tree, packet_rate_hz=1.0)
    alt = series["altitude_m"]
    assert [round(s.value) for s in alt.samples] == [4000, 3950, 3900]


def test_vspeed_derivative():
    data = make_stream([4000.0, 3950.0, 3900.0, 3850.0])
    series = gps5_series(parse_gpmf(data), packet_rate_hz=1.0)
    v = series["vspeed_ms"]
    assert v.value_at(1.5) == pytest.approx(-50.0, abs=5.0)


SAMPLE = Path(__file__).parent.parent / "samples" / "gopro.mp4"


@pytest.mark.skipif(not SAMPLE.exists(), reason="no real GoPro sample provided")
def test_real_gopro_file():
    from tandemista.engine.gpmf import telemetry_from_gopro

    series = telemetry_from_gopro(SAMPLE)
    assert "altitude_m" in series
```

- [ ] **Step 2: Убедиться, что падает** — FAIL (ModuleNotFoundError)

- [ ] **Step 3: Реализация**

```python
# backend/tandemista/engine/gpmf.py
from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

from .media import require_ffmpeg
from .signals import Sample, SignalSeries

_INT_TYPES = {b"l": (">l", 4), b"L": (">L", 4), b"s": (">h", 2), b"S": (">H", 2)}


def parse_gpmf(data: bytes) -> dict[bytes, list]:
    """Parse GPMF KLV into {key: [values...]}; nested containers are flattened."""
    out: dict[bytes, list] = {}
    pos = 0
    while pos + 8 <= len(data):
        key = data[pos : pos + 4]
        type_ = data[pos + 4 : pos + 5]
        size = data[pos + 5]
        repeat = struct.unpack(">H", data[pos + 6 : pos + 8])[0]
        payload_len = size * repeat
        payload = data[pos + 8 : pos + 8 + payload_len]
        if type_ == b"\x00":
            nested = parse_gpmf(payload)
            for k, v in nested.items():
                out.setdefault(k, []).extend(v)
        elif type_ in _INT_TYPES:
            fmt, width = _INT_TYPES[type_]
            n = size // width
            for r in range(repeat):
                chunk = payload[r * size : (r + 1) * size]
                values = [struct.unpack(fmt, chunk[i * width : (i + 1) * width])[0] for i in range(n)]
                out.setdefault(key, []).append(values if n > 1 else values[0])
        # other types (strings, floats) are irrelevant for altitude and skipped
        pos += 8 + payload_len + ((-payload_len) % 4)
    return out


def gps5_series(tree: dict[bytes, list], packet_rate_hz: float = 18.0) -> dict[str, SignalSeries]:
    gps = tree.get(b"GPS5", [])
    scal = tree.get(b"SCAL", [])
    if not gps or len(scal) < 5:
        return {}
    alt_div = float(scal[2] if not isinstance(scal[2], list) else scal[2][0])
    step = 1.0 / packet_rate_hz
    alt = [Sample(i * step, row[2] / alt_div) for i, row in enumerate(gps)]
    alt_series = SignalSeries("altitude_m", alt).resample(1.0)
    vs: list[Sample] = []
    window = 3
    pts = alt_series.samples
    for i in range(1, len(pts)):
        lo = max(0, i - window)
        dt = pts[i].t - pts[lo].t
        vs.append(Sample(pts[i].t, (pts[i].value - pts[lo].value) / dt if dt else 0.0))
    return {"altitude_m": alt_series, "vspeed_ms": SignalSeries("vspeed_ms", vs)}


def telemetry_from_gopro(path: Path) -> dict[str, SignalSeries]:
    require_ffmpeg()
    probe = json.loads(
        subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
            check=True, capture_output=True, text=True,
        ).stdout
    )
    idx = next(
        (s["index"] for s in probe["streams"] if s.get("codec_tag_string") == "gpmd"), None
    )
    if idx is None:
        return {}
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-codec", "copy",
         "-map", f"0:{idx}", "-f", "data", "-"],
        check=True, capture_output=True,
    ).stdout
    return gps5_series(parse_gpmf(raw))
```

- [ ] **Step 4: Тесты зелёные** — `pytest tests/engine/test_gpmf.py -q` → `2 passed, 1 skipped`

- [ ] **Step 5: Commit**

```bash
git add backend/tandemista/engine/gpmf.py backend/tests/engine/test_gpmf.py
git commit -m "feat: minimal GPMF parser and GoPro telemetry extraction"
```

---

### Task 8: Кадровые признаки без весов (frames.py)

Заменяет прежний Task 8 (адаптер Claude Vision), отменённый решением пользователя «своя CV-модель, без LLM на инференсе». Это фундамент собственного CV: числовые признаки каждого кадра, из которых дальше строятся фазы и моменты. Работает на голом OpenCV, без единого файла весов, поэтому полностью тестируется в CI.

**Files:**
- Create: `backend/tandemista/engine/frames.py`
- Test: `backend/tests/engine/test_frames.py`
- Modify: `backend/pyproject.toml` (в `dependencies` добавить `opencv-python-headless>=4.10`, удалить `anthropic`)

**Interfaces:**
- Consumes: ничего из движка (только OpenCV и numpy).
- Produces:
  - `FrameFeatures(t: float, sharpness: float, motion: float, brightness: float, sky_ratio: float)` — frozen dataclass. `t` — секунды от начала файла; `sharpness` — дисперсия лапласиана (безразмерная, больше = резче); `motion` — средняя магнитуда оптического потока к предыдущему сэмплированному кадру в пикселях уменьшенного кадра (у первого кадра 0.0); `brightness` — средняя яркость 0..1; `sky_ratio` — доля пикселей, похожих на небо, 0..1.
  - `extract_frame_features(video: Path, fps: float = 0.5, width: int = 320) -> list[FrameFeatures]` — читает видео через OpenCV, сэмплирует с шагом `1/fps`, уменьшает кадр до ширины `width` (сохраняя пропорции) и считает признаки. Пустой список, если видео не открывается или пусто.

- [ ] **Step 1: Написать падающий тест**

```python
# backend/tests/engine/test_frames.py
import subprocess
from pathlib import Path

import pytest

from tandemista.engine.frames import extract_frame_features


def lavfi_clip(path: Path, source: str, seconds: int) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"{source}:d={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path


@pytest.fixture(scope="module")
def clips(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("frames")
    return {
        "sky": lavfi_clip(d / "sky.mp4", "color=c=0x3399FF:s=640x360:r=10", 8),
        "busy": lavfi_clip(d / "busy.mp4", "testsrc=s=640x360:r=10", 8),
    }


def test_static_sky_clip_is_calm_flat_and_skylike(clips):
    feats = extract_frame_features(clips["sky"], fps=1.0)
    assert len(feats) >= 6
    assert all(f.motion < 0.5 for f in feats[1:])       # nothing moves
    assert all(f.sharpness < 5.0 for f in feats)        # flat colour has no edges
    assert all(f.sky_ratio > 0.8 for f in feats)        # blue and bright
    assert feats[0].motion == 0.0                       # no previous frame


def test_moving_pattern_has_motion_and_detail(clips):
    feats = extract_frame_features(clips["busy"], fps=1.0)
    assert max(f.motion for f in feats) > 1.0
    assert max(f.sharpness for f in feats) > 50.0
    assert max(f.sky_ratio for f in feats) < 0.8        # test pattern is not sky


def test_timestamps_follow_sampling_rate(clips):
    feats = extract_frame_features(clips["sky"], fps=2.0)
    assert feats[0].t == pytest.approx(0.0, abs=0.1)
    assert feats[1].t == pytest.approx(0.5, abs=0.1)
    assert feats[2].t == pytest.approx(1.0, abs=0.1)


def test_missing_file_returns_empty(tmp_path):
    assert extract_frame_features(tmp_path / "nope.mp4") == []
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `cd backend && pytest tests/engine/test_frames.py -q`
Expected: FAIL (ModuleNotFoundError: tandemista.engine.frames)

- [ ] **Step 3: Реализация**

```python
# backend/tandemista/engine/frames.py
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
```

Также в `backend/pyproject.toml`: в `dependencies` заменить `"anthropic>=0.40"` на `"opencv-python-headless>=4.10"` (anthropic больше не используется нигде — CV перешёл на локальный конвейер).

- [ ] **Step 4: Тесты зелёные**

Run: `cd backend && pip install -e ".[dev]" -q && pytest tests/engine/test_frames.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/tandemista/engine/frames.py backend/tests/engine/test_frames.py backend/pyproject.toml
git commit -m "feat: weight-free frame features (sharpness, optical flow, sky ratio)"
```

---

### Task 8b: Локальный CV-аннотатор (замена адаптера Claude Vision)

**Files:**
- Rewrite: `backend/tandemista/engine/cv.py` (сохранить `Moment`, `CVAnnotation`, `CVAnnotator`; удалить `ClaudeVisionAnnotator` и весь код обращения к API; добавить `LocalCVAnnotator` и эвристики)
- Rewrite: `backend/tests/engine/test_cv.py` (удалить fake-клиент Claude; тестировать эвристики на синтетических `FrameFeatures`)

**Interfaces:**
- Consumes: `FrameFeatures`, `extract_frame_features` (Task 8); `Phase`, `PhaseName` (Task 4).
- Produces (имена сохраняются — Tasks 9, 10, 12 на них завязаны):
  - `Moment(t: float, score: float, kind: str)` — kind: `"emotion" | "exit" | "deployment" | "scenic"`.
  - `CVAnnotation(phases: list[Phase], moments: list[Moment])`.
  - `CVAnnotator` Protocol с `annotate(video: Path) -> CVAnnotation`.
  - `phases_from_features(feats: list[FrameFeatures]) -> list[Phase]` — фазы-эвристики, source `"cv"`, confidence 0.5 (ниже телеметрии и аудио: это догадка по картинке).
  - `moments_from_features(feats: list[FrameFeatures]) -> list[Moment]` — момент `"exit"` в точке максимального прироста движения, `"deployment"` в точке максимального спада после него, `"scenic"` — до трёх самых резких кадров с большой долей неба вне пиков движения.
  - `LocalCVAnnotator(fps: float = 0.5)` с `annotate(video)` — извлекает признаки и прогоняет обе эвристики. Task 12 конструирует его без аргументов.

Пороговые значения — относительные (перцентили внутри файла), а не абсолютные: камеры и условия съёмки слишком разные, чтобы зашивать константы яркости и потока.

- [ ] **Step 1: Написать падающий тест**

```python
# backend/tests/engine/test_cv.py
from pathlib import Path

from tandemista.engine.cv import (
    CVAnnotation,
    LocalCVAnnotator,
    Moment,
    moments_from_features,
    phases_from_features,
)
from tandemista.engine.frames import FrameFeatures
from tandemista.engine.phases import PhaseName


def jump_features() -> list[FrameFeatures]:
    """Ground interview, then boarding, then a violent freefall, canopy, landing."""
    feats: list[FrameFeatures] = []

    def add(n: int, motion: float, sky: float, sharp: float, bright: float = 0.5) -> None:
        for _ in range(n):
            t = len(feats) * 2.0
            feats.append(FrameFeatures(t, sharp, motion, bright, sky))

    add(10, motion=0.3, sky=0.05, sharp=60.0)    # interview on the ground
    add(10, motion=0.6, sky=0.10, sharp=40.0)    # inside the plane
    add(20, motion=9.0, sky=0.75, sharp=90.0)    # freefall: violent flow, lots of sky
    add(20, motion=1.5, sky=0.60, sharp=80.0)    # under canopy: calm, still sky
    add(10, motion=0.8, sky=0.10, sharp=70.0)    # landed
    return feats


def test_freefall_detected_from_flow_and_sky():
    phases = phases_from_features(jump_features())
    ff = next(p for p in phases if p.name == PhaseName.FREEFALL)
    assert ff.source == "cv"
    assert ff.confidence == 0.5
    assert ff.start == 40.0            # frame 20 * 2s
    assert ff.end == 80.0              # frame 40 * 2s
    assert any(p.name == PhaseName.CANOPY for p in phases)


def test_no_freefall_when_nothing_moves():
    flat = [FrameFeatures(i * 2.0, 50.0, 0.2, 0.5, 0.1) for i in range(30)]
    assert phases_from_features(flat) == []


def test_exit_and_deployment_moments_bracket_the_freefall():
    moments = moments_from_features(jump_features())
    exit_m = next(m for m in moments if m.kind == "exit")
    dep_m = next(m for m in moments if m.kind == "deployment")
    assert exit_m.t == 40.0            # flow jumps here
    assert dep_m.t == 80.0             # flow collapses here
    assert dep_m.t > exit_m.t
    assert any(m.kind == "scenic" for m in moments)


def test_annotator_returns_annotation_for_a_real_clip(tmp_path):
    import subprocess

    clip = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc=s=320x180:r=10:d=6", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(clip)],
        check=True, capture_output=True,
    )
    ann = LocalCVAnnotator(fps=1.0).annotate(clip)
    assert isinstance(ann, CVAnnotation)
    assert all(isinstance(m, Moment) for m in ann.moments)


def test_annotator_satisfies_the_protocol():
    from tandemista.engine.cv import CVAnnotator

    annotator: CVAnnotator = LocalCVAnnotator()
    assert callable(annotator.annotate)
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `cd backend && pytest tests/engine/test_cv.py -q`
Expected: FAIL (ImportError: LocalCVAnnotator / phases_from_features не существуют)

- [ ] **Step 3: Реализация — полностью переписать `cv.py`**

```python
# backend/tandemista/engine/cv.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .frames import FrameFeatures, extract_frame_features
from .phases import Phase, PhaseName

CV_CONFIDENCE = 0.5          # below telemetry (0.8-0.95) and audio (0.6): this is a guess from pixels
FREEFALL_FLOW_PERCENTILE = 0.75
FREEFALL_MIN_SKY = 0.35
MIN_FREEFALL_FRAMES = 3
CANOPY_MIN_SKY = 0.25
MAX_SCENIC = 3


@dataclass(frozen=True)
class Moment:
    t: float
    score: float
    kind: str


@dataclass(frozen=True)
class CVAnnotation:
    phases: list[Phase]
    moments: list[Moment]


class CVAnnotator(Protocol):
    def annotate(self, video: Path) -> CVAnnotation: ...


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def _step(feats: list[FrameFeatures]) -> float:
    return feats[1].t - feats[0].t if len(feats) > 1 else 1.0


def phases_from_features(feats: list[FrameFeatures]) -> list[Phase]:
    """Guess phases from motion and sky alone. Last-resort fallback: no telemetry, no audio."""
    if len(feats) < MIN_FREEFALL_FRAMES + 1:
        return []
    flows = [f.motion for f in feats]
    threshold = _percentile(flows, FREEFALL_FLOW_PERCENTILE)
    if threshold <= 0.0:
        return []
    step = _step(feats)

    # freefall: the longest run of high flow over open sky
    best: tuple[int, int] | None = None
    start: int | None = None
    for i, f in enumerate(feats):
        if f.motion >= threshold and f.sky_ratio >= FREEFALL_MIN_SKY:
            start = i if start is None else start
        elif start is not None:
            if best is None or (i - start) > (best[1] - best[0]):
                best = (start, i)
            start = None
    if start is not None and (best is None or (len(feats) - start) > (best[1] - best[0])):
        best = (start, len(feats))
    if best is None or (best[1] - best[0]) < MIN_FREEFALL_FRAMES:
        return []

    ff_start, ff_end = feats[best[0]].t, feats[best[1] - 1].t + step
    phases = [
        Phase(PhaseName.EXIT, ff_start, ff_start, CV_CONFIDENCE, "cv"),
        Phase(PhaseName.FREEFALL, ff_start, ff_end, CV_CONFIDENCE, "cv"),
        Phase(PhaseName.DEPLOYMENT, ff_end, ff_end, CV_CONFIDENCE, "cv"),
    ]
    if best[0] > 0:
        phases.insert(0, Phase(PhaseName.CLIMB, feats[0].t, ff_start, CV_CONFIDENCE, "cv"))

    # canopy: frames after freefall that still show sky
    tail = [f for f in feats[best[1]:] if f.sky_ratio >= CANOPY_MIN_SKY]
    if tail:
        phases.append(
            Phase(PhaseName.CANOPY, ff_end, tail[-1].t + step, CV_CONFIDENCE, "cv")
        )
        ground = [f for f in feats if f.t > tail[-1].t and f.sky_ratio < CANOPY_MIN_SKY]
        if ground:
            phases.append(
                Phase(PhaseName.LANDING, ground[0].t, ground[-1].t + step, CV_CONFIDENCE, "cv")
            )
    return phases


def moments_from_features(feats: list[FrameFeatures]) -> list[Moment]:
    """Highlights that need no face model: the flow spike, the flow collapse, the pretty frames."""
    if len(feats) < 3:
        return []
    moments: list[Moment] = []
    deltas = [feats[i].motion - feats[i - 1].motion for i in range(1, len(feats))]
    rise = max(range(len(deltas)), key=lambda i: deltas[i])
    if deltas[rise] > 0:
        moments.append(Moment(feats[rise + 1].t, 1.0, "exit"))
        after = deltas[rise + 1:]
        if after:
            fall = rise + 1 + min(range(len(after)), key=lambda i: after[i])
            if deltas[fall] < 0:
                moments.append(Moment(feats[fall + 1].t, 1.0, "deployment"))

    peak_flow = _percentile([f.motion for f in feats], FREEFALL_FLOW_PERCENTILE)
    calm_and_pretty = [
        f for f in feats if f.motion < peak_flow and f.sky_ratio >= CANOPY_MIN_SKY
    ]
    for f in sorted(calm_and_pretty, key=lambda f: -f.sharpness)[:MAX_SCENIC]:
        moments.append(Moment(f.t, min(1.0, f.sharpness / 100.0), "scenic"))
    return sorted(moments, key=lambda m: m.t)


class LocalCVAnnotator:
    """Own CV pipeline: OpenCV only, no model weights, no network, no LLM."""

    def __init__(self, fps: float = 0.5):
        self.fps = fps

    def annotate(self, video: Path) -> CVAnnotation:
        feats = extract_frame_features(video, fps=self.fps)
        return CVAnnotation(phases_from_features(feats), moments_from_features(feats))
```

- [ ] **Step 4: Тесты зелёные**

Run: `cd backend && pytest tests/engine/test_cv.py -q && pytest -q`
Expected: `5 passed` для файла; вся сюита зелёная (тестов Claude-адаптера больше нет).

- [ ] **Step 5: Commit**

```bash
git add backend/tandemista/engine/cv.py backend/tests/engine/test_cv.py
git commit -m "feat: replace Claude vision adapter with local CV annotator"
```

---

### Task 9: Фьюжн таймлайна прыжка

**Files:**
- Create: `backend/tandemista/engine/timeline.py`
- Test: `backend/tests/engine/test_timeline.py`

**Interfaces:**
- Consumes: `Phase`, `PhaseName` (Task 4), `Moment` (Task 8).
- Produces:
  - `SourceFile(path: Path, role: str, duration: float, clock_offset: float, phases: list[Phase], moments: list[Moment])` — role: `"handcam" | "outside" | "ground_interview" | "ground_landing"`; `phases`/`moments` — в локальном времени файла.
  - `JumpTimeline(files: list[SourceFile], phases: list[Phase])` — фазы в общем времени (`local + clock_offset`).
  - `build_timeline(files: list[SourceFile]) -> JumpTimeline`: для каждой фазы берёт вариант с максимальным приоритетом источника (`telemetry` > `audio` > `cv`), а при равном источнике — с большей confidence; ground_interview-файл целиком даёт INTERVIEW, ground_landing — LANDING (source `"role"`, confidence 0.9), если файлы прыжковых камер эту фазу не дали.

- [ ] **Step 1: Падающий тест**

```python
# backend/tests/engine/test_timeline.py
from pathlib import Path

from tandemista.engine.phases import Phase, PhaseName
from tandemista.engine.timeline import SourceFile, build_timeline


def sf(role, offset, phases, duration=600.0):
    return SourceFile(Path(f"/{role}.mp4"), role, duration, offset, phases, [])


def test_telemetry_beats_cv_and_offsets_applied():
    handcam = sf(
        "handcam", 100.0,
        [Phase(PhaseName.FREEFALL, 200.0, 260.0, 0.9, "telemetry")],
    )
    outside = sf(
        "outside", 0.0,
        [Phase(PhaseName.FREEFALL, 310.0, 365.0, 0.7, "cv")],
    )
    tl = build_timeline([handcam, outside])
    ff = next(p for p in tl.phases if p.name == PhaseName.FREEFALL)
    assert ff.source == "telemetry"
    assert ff.start == 300.0  # 200 local + 100 offset


def test_ground_roles_fill_missing_phases():
    interview = sf("ground_interview", -1800.0, [], duration=120.0)
    handcam = sf("handcam", 0.0, [Phase(PhaseName.FREEFALL, 300.0, 360.0, 0.9, "telemetry")])
    tl = build_timeline([interview, handcam])
    iv = next(p for p in tl.phases if p.name == PhaseName.INTERVIEW)
    assert iv.source == "role"
    assert iv.start == -1800.0 and iv.end == -1680.0
```

- [ ] **Step 2: Убедиться, что падает** — FAIL

- [ ] **Step 3: Реализация**

```python
# backend/tandemista/engine/timeline.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cv import Moment
from .phases import Phase, PhaseName

SOURCE_RANK = {"telemetry": 3, "role": 2, "audio": 1, "cv": 0}
ROLE_PHASE = {
    "ground_interview": PhaseName.INTERVIEW,
    "ground_landing": PhaseName.LANDING,
}


@dataclass(frozen=True)
class SourceFile:
    path: Path
    role: str
    duration: float
    clock_offset: float
    phases: list[Phase]
    moments: list[Moment]


@dataclass(frozen=True)
class JumpTimeline:
    files: list[SourceFile]
    phases: list[Phase]


def build_timeline(files: list[SourceFile]) -> JumpTimeline:
    candidates: dict[PhaseName, Phase] = {}

    def offer(phase: Phase) -> None:
        cur = candidates.get(phase.name)
        if cur is None or (SOURCE_RANK[phase.source], phase.confidence) > (
            SOURCE_RANK[cur.source], cur.confidence
        ):
            candidates[phase.name] = phase

    for f in files:
        for p in f.phases:
            offer(
                Phase(p.name, p.start + f.clock_offset, p.end + f.clock_offset,
                      p.confidence, p.source)
            )
    for f in files:
        name = ROLE_PHASE.get(f.role)
        if name is not None and name not in candidates:
            offer(
                Phase(name, f.clock_offset, f.clock_offset + f.duration, 0.9, "role")
            )
    phases = sorted(candidates.values(), key=lambda p: p.start)
    return JumpTimeline(files, phases)
```

- [ ] **Step 4: Тесты зелёные** — `pytest tests/engine/test_timeline.py -q` → `2 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/tandemista/engine/timeline.py backend/tests/engine/test_timeline.py
git commit -m "feat: jump timeline fusion across sources"
```

---

### Task 10: Шаблоны монтажа и генератор EDL

**Files:**
- Create: `backend/tandemista/engine/templates.py`, `backend/tandemista/engine/edl.py`
- Test: `backend/tests/engine/test_edl.py`

**Interfaces:**
- Consumes: `JumpTimeline`, `SourceFile` (Task 9), `PhaseName` (Task 4), `Moment` (Task 8).
- Produces:
  - `Slot(phase: PhaseName, min_s: float, max_s: float, required: bool = False, prefer_roles: tuple[str, ...] = (), lead_in: float = 0.0, lead_out: float = 0.0)`.
  - `Template(variant: str, aspect: str, slots: tuple[Slot, ...])`; `TEMPLATES: dict[str, Template]` с ключами `"full_16x9"`, `"emotions_16x9"`, `"highlights_9x16"`.
  - `Clip(source: Path, src_in: float, src_out: float)`; `EDL(variant: str, aspect: str, clips: list[Clip])`.
  - `generate_edl(timeline: JumpTimeline, template: Template) -> EDL`: на каждый слот выбирает файл (первый из `prefer_roles`, чья запись покрывает фазу в общем времени; иначе любой покрывающий; у ground-ролей их фаза считается покрытой всегда), окно = фаза ± lead_in/lead_out, длительность клипа зажимается в `[min_s, max_s]`; центр окна — у лучшего `Moment` внутри фазы, если есть. Слот пропускается, если нечем заполнить и `required=False`; если `required=True` и нечем — `raise SlotUnfillableError(slot)`.

- [ ] **Step 1: Падающий тест**

```python
# backend/tests/engine/test_edl.py
from pathlib import Path

import pytest

from tandemista.engine.cv import Moment
from tandemista.engine.edl import SlotUnfillableError, generate_edl
from tandemista.engine.phases import Phase, PhaseName
from tandemista.engine.templates import TEMPLATES
from tandemista.engine.timeline import SourceFile, build_timeline


def full_jump_files():
    handcam_phases = [
        Phase(PhaseName.CLIMB, 0.0, 300.0, 0.9, "telemetry"),
        Phase(PhaseName.EXIT, 300.0, 300.0, 0.9, "telemetry"),
        Phase(PhaseName.FREEFALL, 300.0, 360.0, 0.9, "telemetry"),
        Phase(PhaseName.DEPLOYMENT, 360.0, 360.0, 0.9, "telemetry"),
        Phase(PhaseName.CANOPY, 360.0, 600.0, 0.9, "telemetry"),
        Phase(PhaseName.LANDING, 600.0, 615.0, 0.9, "telemetry"),
    ]
    moments = [Moment(320.0, 0.95, "emotion")]
    return [
        SourceFile(Path("/interview.mp4"), "ground_interview", 90.0, -2000.0, [], []),
        SourceFile(Path("/handcam.mp4"), "handcam", 640.0, 0.0, handcam_phases, moments),
        SourceFile(Path("/landing.mp4"), "ground_landing", 60.0, 590.0, [], []),
    ]


def test_full_template_covers_slots_in_order():
    edl = generate_edl(build_timeline(full_jump_files()), TEMPLATES["full_16x9"])
    assert edl.aspect == "16:9"
    assert edl.clips[0].source == Path("/interview.mp4")
    assert any(c.source == Path("/handcam.mp4") for c in edl.clips)
    for c in edl.clips:
        assert c.src_out > c.src_in >= 0.0


def test_freefall_clip_centered_on_best_moment():
    edl = generate_edl(build_timeline(full_jump_files()), TEMPLATES["emotions_16x9"])
    ff = [c for c in edl.clips if c.source == Path("/handcam.mp4")]
    assert any(c.src_in <= 320.0 <= c.src_out for c in ff)


def test_required_slot_without_footage_raises():
    files = [SourceFile(Path("/interview.mp4"), "ground_interview", 90.0, 0.0, [], [])]
    with pytest.raises(SlotUnfillableError):
        generate_edl(build_timeline(files), TEMPLATES["full_16x9"])
```

- [ ] **Step 2: Убедиться, что падает** — FAIL

- [ ] **Step 3: Реализация**

```python
# backend/tandemista/engine/templates.py
from __future__ import annotations

from dataclasses import dataclass, field

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
```

```python
# backend/tandemista/engine/edl.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .phases import Phase
from .templates import Slot, Template
from .timeline import JumpTimeline, ROLE_PHASE, SourceFile


class SlotUnfillableError(Exception):
    def __init__(self, slot: Slot):
        super().__init__(f"no footage for required slot {slot.phase}")
        self.slot = slot


@dataclass(frozen=True)
class Clip:
    source: Path
    src_in: float
    src_out: float


@dataclass(frozen=True)
class EDL:
    variant: str
    aspect: str
    clips: list[Clip]


def _covers(f: SourceFile, start: float, end: float) -> bool:
    return f.clock_offset <= start and end <= f.clock_offset + f.duration


def _pick_file(timeline: JumpTimeline, slot: Slot, phase: Phase) -> SourceFile | None:
    def candidates():
        for role in slot.prefer_roles:
            yield from (f for f in timeline.files if f.role == role)
        yield from (f for f in timeline.files if f.role not in slot.prefer_roles)

    for f in candidates():
        if ROLE_PHASE.get(f.role) == phase.name or _covers(f, phase.start, phase.end):
            return f
    return None


def generate_edl(timeline: JumpTimeline, template: Template) -> EDL:
    by_name = {p.name: p for p in timeline.phases}
    clips: list[Clip] = []
    for slot in template.slots:
        phase = by_name.get(slot.phase)
        f = _pick_file(timeline, slot, phase) if phase else None
        if f is None:
            if slot.required:
                raise SlotUnfillableError(slot)
            continue
        start = phase.start - slot.lead_in
        end = phase.end + slot.lead_out
        length = min(max(end - start, slot.min_s), slot.max_s)
        best = max(
            (m for m in f.moments
             if phase.start <= m.t + f.clock_offset <= phase.end),
            key=lambda m: m.score, default=None,
        )
        if best is not None:
            start = best.t + f.clock_offset - length / 2
        # convert to file-local time and clamp to the file bounds
        local_in = max(0.0, start - f.clock_offset)
        local_out = min(f.duration, local_in + length)
        if local_out - local_in < 1.0:
            if slot.required:
                raise SlotUnfillableError(slot)
            continue
        clips.append(Clip(f.path, round(local_in, 3), round(local_out, 3)))
    return EDL(template.variant, template.aspect, clips)
```

- [ ] **Step 4: Тесты зелёные** — `pytest tests/engine/test_edl.py -q` → `3 passed`. Если строка с `center` вызывает вопросы линтера — убрать переменную, оставить только ветку со `start`.

- [ ] **Step 5: Commit**

```bash
git add backend/tandemista/engine/templates.py backend/tandemista/engine/edl.py backend/tests/engine/test_edl.py
git commit -m "feat: cut templates and EDL generator"
```

---

### Task 11: Рендерер EDL (ffmpeg)

**Files:**
- Create: `backend/tandemista/engine/render.py`
- Test: `backend/tests/engine/test_render.py`

**Interfaces:**
- Consumes: `EDL`, `Clip` (Task 10); `probe_duration`, `require_ffmpeg` (Task 5).
- Produces: `render_edl(edl: EDL, out_path: Path, height: int = 720) -> Path` — один вызов ffmpeg с filter_complex: на каждый клип `trim` + `setpts` + масштабирование (`16:9` → `scale=-2:height`, pad до кратности; `9:16` → `crop=ih*9/16:ih` центр, затем `scale`), `concat` видео и аудио (`atrim`/`asetpts`), кодеки `libx264` + `aac`.

- [ ] **Step 1: Падающий тест** (исходники — три цветных клипа из lavfi)

```python
# backend/tests/engine/test_render.py
import subprocess
from pathlib import Path

import pytest

from tandemista.engine.edl import EDL, Clip
from tandemista.engine.media import probe_duration
from tandemista.engine.render import render_edl


def make_clip(path: Path, color: str, seconds: int) -> Path:
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", f"color=c={color}:s=640x360:d={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}",
         "-shortest", "-c:v", "libx264", "-c:a", "aac", str(path)],
        check=True, capture_output=True,
    )
    return path


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    d = tmp_path_factory.mktemp("render")
    return [
        make_clip(d / "a.mp4", "red", 5),
        make_clip(d / "b.mp4", "green", 5),
    ]


def test_render_16x9_duration(sources, tmp_path):
    edl = EDL("full_16x9", "16:9", [
        Clip(sources[0], 0.0, 3.0),
        Clip(sources[1], 1.0, 4.0),
    ])
    out = render_edl(edl, tmp_path / "out.mp4", height=360)
    assert probe_duration(out) == pytest.approx(6.0, abs=0.5)


def test_render_9x16_aspect(sources, tmp_path):
    edl = EDL("highlights_9x16", "9:16", [Clip(sources[0], 0.0, 2.0)])
    out = render_edl(edl, tmp_path / "v.mp4", height=640)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(out)],
        check=True, capture_output=True, text=True,
    ).stdout.strip().split(",")
    w, h = int(probe[0]), int(probe[1])
    assert abs(w / h - 9 / 16) < 0.02
```

- [ ] **Step 2: Убедиться, что падает** — FAIL

- [ ] **Step 3: Реализация**

```python
# backend/tandemista/engine/render.py
from __future__ import annotations

import subprocess
from pathlib import Path

from .edl import EDL
from .media import require_ffmpeg


def render_edl(edl: EDL, out_path: Path, height: int = 720) -> Path:
    require_ffmpeg()
    if not edl.clips:
        raise ValueError("EDL has no clips")
    inputs: list[str] = []
    filters: list[str] = []
    if edl.aspect == "9:16":
        width = int(height * 9 / 16 / 2) * 2
        vscale = f"crop=ih*9/16:ih,scale={width}:{height}"
    else:
        width = int(height * 16 / 9 / 2) * 2
        vscale = f"scale={width}:{height}"
    for i, c in enumerate(edl.clips):
        inputs += ["-i", str(c.source)]
        filters.append(
            f"[{i}:v]trim=start={c.src_in}:end={c.src_out},setpts=PTS-STARTPTS,"
            f"{vscale},setsar=1[v{i}];"
            f"[{i}:a]atrim=start={c.src_in}:end={c.src_out},asetpts=PTS-STARTPTS[a{i}]"
        )
    pairs = "".join(f"[v{i}][a{i}]" for i in range(len(edl.clips)))
    filters.append(f"{pairs}concat=n={len(edl.clips)}:v=1:a=1[vo][ao]")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", *inputs,
         "-filter_complex", ";".join(filters),
         "-map", "[vo]", "-map", "[ao]",
         "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(out_path)],
        check=True, capture_output=True,
    )
    return out_path
```

- [ ] **Step 4: Тесты зелёные** — `pytest tests/engine/test_render.py -q` → `2 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/tandemista/engine/render.py backend/tests/engine/test_render.py
git commit -m "feat: ffmpeg EDL renderer with 16:9 and 9:16 output"
```

---

### Task 12: CLI `analyze-and-cut` — сквозная сборка

**Files:**
- Create: `backend/tandemista/cli.py`
- Test: `backend/tests/test_cli.py`
- Modify: `backend/pyproject.toml` (добавить `[project.scripts]`), `README.md` (раздел Usage)

**Interfaces:**
- Consumes: всё выше — `telemetry_from_gopro`, `extract_audio_rms`, `detect_phases_from_telemetry`, `detect_phases_from_audio`, `probe_duration`, `SourceFile`, `build_timeline`, `TEMPLATES`, `generate_edl`, `render_edl`, `LocalCVAnnotator`.
- Produces: консольная команда `tandemista <jump_dir> --out <dir> [--height N] [--no-cv]`. Роли файлов — по префиксу имени: `interview_*` → ground_interview, `handcam_*` → handcam, `outside_*` → outside, `landing_*` → ground_landing (конвенция MVP; в облачном каркасе роли придут из Device). Смещения часов: interview ставится до прыжковых файлов, landing — после, по порядку (MVP: `interview=-3600`, прыжковые=0, `landing=+3600` — реальная синхронизация придёт с метчингом). Для каждого шаблона: если генерация упала (`SlotUnfillableError`) — пропустить вариант с warning, остальные рендерить. CV локальный и бесплатный, поэтому включён ВСЕГДА: он даёт моменты (`exit`/`deployment`/`scenic`) для наполнения слотов и фазы, когда нет ни телеметрии, ни аудио. Флаг `--no-cv` отключает его для отладки.

- [ ] **Step 1: Падающий тест** (синтетические клипы; телеметрии нет, аудио сделаем «ветер» на нужном интервале — сработает audio-fallback)

```python
# backend/tests/test_cli.py
import subprocess
from pathlib import Path

import pytest

from tandemista.cli import main


def lavfi_clip(path: Path, seconds: int, audio_expr: str) -> Path:
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=640x360:rate=10",
         "-f", "lavfi", "-i", f"aevalsrc={audio_expr}:d={seconds}",
         "-shortest", "-c:v", "libx264", "-c:a", "aac", str(path)],
        check=True, capture_output=True,
    )
    return path


@pytest.fixture(scope="module")
def jump_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("jump")
    lavfi_clip(d / "interview_01.mp4", 30, "0.2*sin(2*PI*300*t)")
    # 120s handcam: quiet 0..40, wind roar 40..100 (freefall), quiet after
    roar = "if(between(t\\,40\\,100)\\,0.9*random(0)\\,0.05*sin(2*PI*200*t))"
    lavfi_clip(d / "handcam_01.mp4", 120, roar)
    lavfi_clip(d / "landing_01.mp4", 20, "0.2*sin(2*PI*300*t)")
    return d


def test_cli_renders_variants(jump_dir, tmp_path):
    code = main([str(jump_dir), "--out", str(tmp_path), "--height", "240"])
    assert code == 0
    rendered = sorted(p.name for p in tmp_path.glob("*.mp4"))
    assert "full_16x9.mp4" in rendered
    assert "highlights_9x16.mp4" in rendered
```

- [ ] **Step 2: Убедиться, что падает** — FAIL

- [ ] **Step 3: Реализация**

```python
# backend/tandemista/cli.py
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
```

Добавить в `backend/pyproject.toml`:
```toml
[project.scripts]
tandemista = "tandemista.cli:main"
```

Добавить в `README.md` раздел:
```markdown
## Usage (engine CLI)
    tandemista /path/to/jump_dir --out /tmp/cuts
    # file naming: interview_*.mp4, handcam_*.mp4, outside_*.mp4, landing_*.mp4
    # local CV runs always; --no-cv skips it for debugging
```

- [ ] **Step 4: Все тесты зелёные**

Run: `cd backend && pip install -e ".[dev]" -q && pytest -q`
Expected: все тесты (Task 1–12) PASS, кроме skipped-теста реального GoPro.

- [ ] **Step 5: Commit**

```bash
git add backend/tandemista/cli.py backend/tests/test_cli.py backend/pyproject.toml README.md
git commit -m "feat: analyze-and-cut CLI wiring the full engine pipeline"
```

---

## Верификация плана целиком

1. `cd backend && pytest -q` — всё зелёное.
2. Прогнать CLI на синтетике из `tests/test_cli.py` (fixture-папку можно собрать теми же ffmpeg-командами руками) — получить `full_16x9.mp4`, `emotions_16x9.mp4`, `highlights_9x16.mp4`, открыть глазами.
3. Положить реальные файлы прыжка в папку с префиксами ролей, прогнать `tandemista <dir> --out cuts` — проверить фазы и монтаж на настоящем материале (появление `backend/tests/samples/gopro.mp4` включит интеграционный тест GPMF).

## Отложено в следующие планы

- Облачный каркас: FastAPI, PostgreSQL, Celery, StorageBackend, админка ревью, клиентская страница (план 2).
- Аплоадер Tauri, автометчинг, clock_offset-калибровка (план 3).
- Доставка, вотермарки, PaymentProvider (ЮKassa, Stripe) (план 4).
- Музыка с даккингом, титры, кроп 9:16 по лицам (после пилотной обратной связи; заготовки — `Moment`, `aspect` в EDL).
- **CV, слой с весами**: детекция лиц (YuNet ONNX, ~230 КБ) и мимика (MediaPipe Face Landmarker blendshapes, ~3.7 МБ) → моменты `"emotion"` и bbox лица для автокропа 9:16. Веса качает `backend/scripts/fetch_models.py` в `backend/models/` (в .gitignore), тесты — `skipif` при отсутствии. Отложено сознательно: сквозной конвейер должен сначала заработать целиком; эмоции — это качество выборки, а не блокер.
- **CV, своя обученная модель**: разметку копим из правок оператора на экране ревью (подвинутая граница клипа = метка «здесь фаза начинается на самом деле», заменённый отрезок = метка «этот момент лучше»), затем обучаем классификатор фаз и скорер моментов и подставляем через тот же Protocol `CVAnnotator`. Отдельный план после пилота.
