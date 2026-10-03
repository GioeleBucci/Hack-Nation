"""Shared helpers for tools: error handling, JSON conversion, cached HTTP, the record handle."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import re
from functools import lru_cache
from typing import Any, Callable

import numpy as np
import requests

from firelab import config
from firelab.record import Record

log = logging.getLogger("firelab")

SAFE_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


class ToolError(Exception):
    """An expected failure whose message is shown to the agent as-is."""


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def tool(fn: Callable) -> Callable:
    """Wrap a tool so it always returns a JSON-serializable dict and never raises."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            result = fn(*args, **kwargs)
            if isinstance(result, dict):
                result.setdefault("ok", True)
            return to_jsonable(result)
        except ToolError as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:  # noqa: BLE001 - agents get the message instead of a crashed session
            log.exception("tool %s failed", fn.__name__)
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return wrapper


def check_id(value: str, what: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.match(value):
        raise ToolError(f"{what} '{value}' must be 1-64 characters of letters, digits, '_', '-', '.'")
    return value


@lru_cache(maxsize=1)
def get_record() -> Record:
    return Record()


def get_json(url: str, params: dict | None = None, use_cache: bool = True, timeout: int = 30) -> dict:
    """GET a JSON API with an on-disk cache, so reruns are reproducible and offline-friendly."""
    key = hashlib.sha1((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
    path = config.CACHE_DIR / f"{key}.json"
    if use_cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    agent = "firelab/0.1" + (f" (mailto:{config.OPENALEX_MAILTO})" if config.OPENALEX_MAILTO else "")
    resp = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": agent})
    resp.raise_for_status()
    data = resp.json()
    config.ensure_dirs()
    path.write_text(json.dumps(data), encoding="utf-8")
    return data
