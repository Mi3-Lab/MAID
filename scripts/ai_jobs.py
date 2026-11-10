#!/usr/bin/env python3
"""Shared live-progress + log writer for WheelView AI jobs.

Scripts write var/ai_jobs/current.json and append var/ai_jobs/log.txt.
The /ai review page polls these (via /api/ai/status). Never a human label.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
JOBS = Path(os.environ.get("WV_AI_JOBS", ROOT / "var" / "ai_jobs"))
CURRENT = JOBS / "current.json"
LOG = JOBS / "log.txt"
TZ_NAME = "America/Los_Angeles"


def now_iso() -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(TZ_NAME)).isoformat(timespec="seconds")
    except Exception:
        return datetime.now().astimezone().isoformat(timespec="seconds")


def _ensure() -> None:
    JOBS.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    _ensure()
    line = f"[{now_iso()}] {msg}"
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def read_current() -> dict:
    try:
        return json.loads(CURRENT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def log_tail(n: int = 40) -> list:
    try:
        lines = LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-max(1, n):]
    except OSError:
        return []


def update(
    job: str,
    stage: str,
    n_done: int = 0,
    n_total: int = 0,
    status: str = "running",
    message: str = "",
    extra: Optional[dict] = None,
) -> dict:
    """Write current.json. Preserves `started` across updates of the same job."""
    _ensure()
    prev = read_current()
    started = prev.get("started") if prev.get("job") == job and prev.get("status") == "running" else None
    if not started:
        started = now_iso()
    now = now_iso()
    eta = None
    elapsed = None
    try:
        # started may be ISO with offset
        t0 = datetime.fromisoformat(started)
        t1 = datetime.fromisoformat(now)
        elapsed = max(0.0, (t1 - t0).total_seconds())
        if n_done > 0 and n_total > n_done and elapsed > 0:
            rate = n_done / elapsed
            remain = (n_total - n_done) / rate if rate > 0 else None
            if remain is not None:
                eta = now_iso()  # placeholder; also numeric seconds
    except Exception:
        remain = None
    else:
        remain = remain if n_done > 0 and n_total > n_done else None
    rec: dict[str, Any] = {
        "job": job,
        "stage": stage,
        "status": status,
        "n_done": int(n_done),
        "n_total": int(n_total),
        "started": started,
        "last_update": now,
        "elapsed_s": None if elapsed is None else round(elapsed, 1),
        "eta_s": None if remain is None else round(remain, 1),
        "message": message,
        "timezone": TZ_NAME,
        "log_path": str(LOG),
    }
    if extra:
        rec.update(extra)
    tmp = CURRENT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    tmp.replace(CURRENT)
    return rec


def fail(job: str, message: str, **kwargs) -> dict:
    rec = update(job, kwargs.get("stage") or "error", kwargs.get("n_done") or 0,
                 kwargs.get("n_total") or 0, status="error", message=message)
    log(f"ERROR {job}: {message}")
    return rec
