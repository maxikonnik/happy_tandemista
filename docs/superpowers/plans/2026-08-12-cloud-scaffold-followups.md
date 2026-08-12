# Cloud scaffold — follow-ups for the next (pipeline) plan

These are the items deliberately deferred during the `feat/cloud-scaffold` backend build
(plan: `2026-08-12-cloud-scaffold-backend.md`). Each was found by review, judged non-blocking
for the scaffold, and recorded here so the analysis-pipeline plan starts with them in view.
The three cheap correctness fixes I1/I3/M1 were already applied on the branch (commit `13de478`).

## Carry into the pipeline plan (matter once real analysis replaces the stub)

- **I2 — Streaming uploads / size cap.** `api/routes/media.py` reads the whole upload into
  memory (`file.file.read()`), then copies it into a `BytesIO` and hashes it. Skydive footage is
  multi-GB; a few concurrent uploads will OOM the API worker. Change `StorageBackend.put` to accept
  a stream (chunked write + incremental sha256) and enforce a max size / content-type allowlist.
  Touches `storage/base.py`, `storage/local.py`, `storage/s3.py`, `api/routes/media.py`.

- **M2 — Enum columns round-trip as plain `str`.** Enums are stored in `String` columns, so a
  freshly loaded row's `media.status` is `"registered"` (a `str`), not `MediaStatus.REGISTERED`;
  equality against an enum member still works, but `.value`/`.name` on a loaded row raises
  `AttributeError`. The pipeline code will read these back constantly — either add a
  `values_callable`/validator to coerce on load, or standardise on comparing to enum members and
  never calling `.value` on a loaded attribute. Document the chosen convention.

- **Worker session in a real process.** The worker binds its DB session via the Celery
  `worker_process_init` signal (`worker/celery_app.py`). Confirm this holds when the real
  `analyze_media` runs under `celery worker` (not just eager tests), including per-task session
  lifecycle and retries.

## Hardening (not tenancy-blocking, do when the area is touched)

- **S3 `_ensure_bucket` swallows all `ClientError`.** `storage/s3.py` catches any `ClientError`
  from `head_bucket` and then creates the bucket, which masks non-404 failures (auth/permission)
  behind a less diagnostic secondary error. Narrow the catch to 404 / `NoSuchBucket`. S3 is not
  the default backend, so this is latent today.

- **`create_app` mutates the process-global Celery singleton.** `api/app.py` sets
  `celery.conf.task_always_eager` from its settings (mirrors the existing global `configure_session`
  pattern). Inert for a single-process API, but two apps with different settings in one process would
  share the flag. Scope the Celery config per-app (or late-bind settings) if that scenario appears.

- **`MANUAL_WEB` upload provenance.** The upload handler accepts `device_id`/`jump_id` but never
  creates an `UploadSource` row or sets `MediaFile.upload_source_id`. Wire provenance when matching
  (stage 2 of the product plan) needs the upload source as a signal.

## Multi-tenancy (revisit when SaaS tenancy lands)

- **`GET /dropzones/` is globally unscoped** (`api/routes/dropzones.py`) — the one endpoint with no
  `dropzone_id` filter. Fine single-tenant; add auth/tenant scoping with the tenancy work.

## Cosmetic (trivial one-pass cleanup, anytime)

- Nine leftover `# backend/...` path-header comments at the top of new modules
  (`api/app.py`, `api/deps.py`, `api/routes/media.py`, `db/base.py`, `db/models.py`, `db/types.py`,
  `worker/celery_app.py`, `worker/tasks.py`, `migrations/env.py`).
- Unused `from sqlalchemy import select` in `tests/cloud/test_models.py`.
