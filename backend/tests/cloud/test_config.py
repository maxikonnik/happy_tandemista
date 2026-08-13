from tandemista.config import Settings, get_settings


def test_defaults_are_offline_friendly():
    s = Settings()
    assert s.database_url.startswith("sqlite")
    assert s.storage_backend == "local"
    assert s.s3_bucket == "tandemista"
    assert s.celery_task_always_eager is False


def test_env_prefix_overrides(monkeypatch):
    monkeypatch.setenv("TANDEMISTA_STORAGE_BACKEND", "s3")
    monkeypatch.setenv("TANDEMISTA_S3_BUCKET", "jumps")
    s = Settings()
    assert s.storage_backend == "s3"
    assert s.s3_bucket == "jumps"


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
