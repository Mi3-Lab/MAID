"""Read-only helpers for the /ai review page.

Never writes events, never treats AI labels as human labels.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = Path(os.environ.get("WV_AI_JOBS", ROOT / "var" / "ai_jobs"))
CURRENT = JOBS / "current.json"
LOG = JOBS / "log.txt"
VERIFY_CSV = ROOT / "derived" / "ai_label_verification.csv"
VERIFY_JSON = ROOT / "derived" / "ai_label_verification_summary.json"
CONSENSUS_JSON = ROOT / "derived" / "ai_labels_3model_summary.json"

# columns the table needs; extra CSV columns are kept on the row
TABLE_FIELDS = (
    "image_id", "field", "value", "source", "source_file", "verdict",
    "recommended_action", "reason", "human_value_folded", "human_votes",
    "ai_vote", "voters", "majority_outcome", "majority_value", "majority_basis",
    "majority_agreeing", "n_agree", "n_votes", "n_human_voters",
    "model_a_value", "model_a_confidence", "model_b_value", "model_b_confidence",
    "model_c_value", "model_c_confidence", "dino_oof", "threshold",
    "in_training_split", "split", "run_timestamp",
)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def log_tail(n: int = 40) -> list:
    try:
        lines = LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-max(1, int(n)):]
    except OSError:
        return []


def status(log_n: int = 40) -> dict:
    job = _read_json(CURRENT) or {}
    return {
        "ok": True,
        "job": job,
        "log_tail": log_tail(log_n),
        "has_verification": VERIFY_CSV.is_file(),
        "has_consensus_summary": CONSENSUS_JSON.is_file(),
    }


def _slim_consensus(doc):
    if not doc:
        return None
    out = {
        "run_timestamp": doc.get("run_timestamp"),
        "models": doc.get("models"),
        "device": doc.get("device"),
        "inference": doc.get("inference"),
    }
    for field in ("device_type", "occupied"):
        m = doc.get(field) or {}
        ch = m.get("chosen") or {}
        out[field] = {
            "emit": m.get("emit"),
            "threshold": m.get("chosen_threshold"),
            "precision": ch.get("precision"),
            "precision_ci95_lo": ch.get("precision_ci95_lo"),
            "precision_ci95_hi": ch.get("precision_ci95_hi"),
            "n_agreed": ch.get("n_agreed"),
            "coverage": ch.get("coverage"),
            "per_model_accuracy": {
                k: (m.get("per_model_all") or {}).get(k, {}) and (m["per_model_all"][k] or {}).get("accuracy")
                for k in ("siglip", "clip", "dinov2_oof")
            },
        }
    return out


def summary() -> dict:
    ver = _read_json(VERIFY_JSON)
    cons = _slim_consensus(_read_json(CONSENSUS_JSON))
    return {
        "ok": True,
        "verification": ver,
        "consensus_3model": cons,
        "paths": {
            "verification_csv": str(VERIFY_CSV) if VERIFY_CSV.is_file() else None,
            "verification_summary": str(VERIFY_JSON) if VERIFY_JSON.is_file() else None,
            "consensus_summary": str(CONSENSUS_JSON) if CONSENSUS_JSON.is_file() else None,
            "current_job": str(CURRENT) if CURRENT.is_file() else None,
        },
    }


def _parse_maybe_json(v):
    if not v:
        return v
    if isinstance(v, (list, dict)):
        return v
    s = str(v).strip()
    if s.startswith("[") or s.startswith("{"):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return v
    return v


def load_rows() -> list:
    if not VERIFY_CSV.is_file():
        return []
    rows = []
    with VERIFY_CSV.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            r["voters"] = _parse_maybe_json(r.get("voters"))
            r["human_votes"] = _parse_maybe_json(r.get("human_votes"))
            for k in ("model_a_confidence", "model_b_confidence", "model_c_confidence", "threshold"):
                if r.get(k) not in (None, ""):
                    try:
                        r[k] = float(r[k])
                    except ValueError:
                        pass
            r["in_training_split"] = str(r.get("in_training_split") or "").lower() in ("1", "true", "yes")
            r["dino_oof"] = str(r.get("dino_oof") or "").lower() in ("1", "true", "yes")
            rows.append(r)
    return rows


def query_labels(
    *,
    verdict: str = "",
    field: str = "",
    value: str = "",
    source: str = "",
    q: str = "",
    action: str = "",
    basis: str = "",
    offset: int = 0,
    limit: int = 50,
) -> dict:
    rows = load_rows()
    facets = {"verdict": {}, "field": {}, "value": {}, "source": {}, "recommended_action": {}, "majority_basis": {}}
    for r in rows:
        for key in facets:
            v = r.get(key) or ""
            if v == "":
                continue
            facets[key][v] = facets[key].get(v, 0) + 1

    def ok(r):
        if verdict and r.get("verdict") != verdict:
            return False
        if field and r.get("field") != field:
            return False
        if value and r.get("value") != value:
            return False
        if source and r.get("source") != source:
            return False
        if action and r.get("recommended_action") != action:
            return False
        if basis and r.get("majority_basis") != basis:
            return False
        if q:
            needle = q.lower()
            blob = " ".join(str(r.get(k) or "") for k in ("image_id", "field", "value", "source", "verdict", "majority_basis", "reason", "human_value_folded"))
            if needle not in blob.lower():
                return False
        return True

    filtered = [r for r in rows if ok(r)]
    offset = max(0, int(offset or 0))
    limit = max(1, min(int(limit or 50), 200))
    page = filtered[offset: offset + limit]
    # only send table fields plus any already-parsed structures
    slim = []
    for r in page:
        slim.append({k: r.get(k) for k in TABLE_FIELDS if k in r})
    return {
        "ok": True,
        "total": len(filtered),
        "n_all": len(rows),
        "offset": offset,
        "limit": limit,
        "rows": slim,
        "facets": facets,
        "available": bool(rows),
    }
