from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _env_float(key: str) -> float | None:
    v = os.environ.get(key, "").strip()
    return float(v) if v else None


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


@dataclass
class Config:
    base_url: str
    api_key: str
    stream: bool
    traces_dir: Path
    max_usd_per_task: float | None
    max_usd_per_model: float | None
    max_usd_per_run: float | None
    request_timeout_s: float
    max_retries: int
    app_url: str | None
    app_name: str | None
    hf_token: str | None
    hf_dataset_repo: str | None


def load_config(dotenv_path: str | os.PathLike | None = None) -> Config:
    load_dotenv(dotenv_path=dotenv_path, override=False)

    base_url = os.environ.get("OPENMESH_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    api_key = os.environ.get("OPENMESH_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENMESH_API_KEY is not set (see .env.example)")

    return Config(
        base_url=base_url,
        api_key=api_key,
        stream=_env_bool("OPENMESH_STREAM", False),
        traces_dir=Path(os.environ.get("OPENMESH_TRACES_DIR", "./traces")).resolve(),
        max_usd_per_task=_env_float("OPENMESH_MAX_USD_PER_TASK"),
        max_usd_per_model=_env_float("OPENMESH_MAX_USD_PER_MODEL"),
        max_usd_per_run=_env_float("OPENMESH_MAX_USD_PER_RUN"),
        request_timeout_s=float(os.environ.get("OPENMESH_REQUEST_TIMEOUT_S", "120")),
        max_retries=int(os.environ.get("OPENMESH_MAX_RETRIES", "3")),
        app_url=os.environ.get("OPENMESH_APP_URL") or None,
        app_name=os.environ.get("OPENMESH_APP_NAME") or None,
        hf_token=os.environ.get("HF_TOKEN") or None,
        hf_dataset_repo=os.environ.get("HF_DATASET_REPO") or None,
    )
