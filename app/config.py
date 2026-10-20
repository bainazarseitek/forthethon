import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None or value == "" else value.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default_factory=lambda: os.environ.get("API_KEY", ""))
    detector: str = field(default_factory=lambda: os.environ.get("DETECTOR", "stub"))
    threshold: float = field(default_factory=lambda: _env_float("THRESHOLD", 0.5))
    threshold_other: float = field(default_factory=lambda: _env_float("THRESHOLD_OTHER", 0.7))
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("DATA_DIR", ROOT / "var")))
    rules_path: Path = field(default_factory=lambda: Path(os.environ.get("RULES_PATH", ROOT / "data" / "rules.json")))
    job_timeout_sec: float = field(default_factory=lambda: _env_float("JOB_TIMEOUT_SEC", 270))
    job_deadline_sec: float = field(default_factory=lambda: _env_float("JOB_DEADLINE_SEC", 285))
    post_budget_sec: float = field(default_factory=lambda: _env_float("POST_BUDGET_SEC", 15))
    min_budget_sec: float = field(default_factory=lambda: _env_float("MIN_BUDGET_SEC", 20))
    app_uploads_enabled: bool = field(default_factory=lambda: _env_bool("APP_UPLOADS_ENABLED", True))
    app_queue_cap: int = field(default_factory=lambda: int(_env_float("APP_QUEUE_CAP", 3)))
    retention_sec: float = field(default_factory=lambda: _env_float("RETENTION_SEC", 24 * 3600))
    max_upload_bytes: int = field(default_factory=lambda: int(_env_float("MAX_UPLOAD_MB", 2048) * 1024 * 1024))
    cameras_enabled: bool = field(default_factory=lambda: _env_bool("CAMERAS_ENABLED", True))
    cameras_config: Path = field(default_factory=lambda: Path(os.environ.get("CAMERAS_CONFIG", ROOT / "data" / "cameras.json")))
    cameras_idle_sec: float = field(default_factory=lambda: _env_float("CAMERAS_IDLE_SEC", 120))
    timezone: str = field(default_factory=lambda: os.environ.get("TIMEZONE", "Asia/Almaty"))
    jpeg_quality: int = 85
    max_frames_per_event: int = 3
