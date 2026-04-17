"""Disk caching for slow / rate-limited fetchers.

Pickle-based, namespaced by function name and a stable hash of arguments.
Vendor responses are big and the same bar history is requested over and over
during research, so a simple file cache keeps iteration loops fast.
"""

from __future__ import annotations

import hashlib
import pickle
from functools import wraps
from pathlib import Path
from typing import Callable


def _hash_args(args: tuple, kwargs: dict) -> str:
    h = hashlib.sha256()
    h.update(repr(args).encode())
    h.update(repr(sorted(kwargs.items())).encode())
    return h.hexdigest()[:16]


def disk_cache(cache_dir: str | Path, namespace: str | None = None) -> Callable:
    """Decorator that pickles return values to disk.

    Use only with deterministic-input functions (data fetchers keyed by
    ticker/date). Skips caching when env var TREND_ENGINE_NO_CACHE=1.
    """
    cache_root = Path(cache_dir)

    def decorator(fn: Callable) -> Callable:
        ns = namespace or fn.__name__
        ns_dir = cache_root / ns
        ns_dir.mkdir(parents=True, exist_ok=True)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            import os
            if os.environ.get("TREND_ENGINE_NO_CACHE") == "1":
                return fn(*args, **kwargs)
            key = _hash_args(args, kwargs)
            path = ns_dir / f"{key}.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    return pickle.load(f)
            value = fn(*args, **kwargs)
            tmp = path.with_suffix(".pkl.tmp")
            with open(tmp, "wb") as f:
                pickle.dump(value, f)
            tmp.replace(path)
            return value

        return wrapper

    return decorator
