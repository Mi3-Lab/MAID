"""Assistant + SigLIP consensus -> status=auto_approved (never a human label).

Agreement on device_type (manual/powered/other_aid) with SigLIP top-1 confidence >= threshold
writes var/assist/auto_approved.jsonl. Disagreements append var/adjudication/queue.jsonl with
reason=assistant_siglip_disagreement (deduped against existing device_type_disagreement /
assistant_siglip_disagreement / assistant_review_conflict rows).

Exclusions (never auto-approved): child flags (assistant / human exclude / safety hold /
removed_model_child_stroller / SigLIP child flag), assistant flagged.ndjson, bad license,
assistant device_type ambiguous/unclear/lookalike. Existing human labels are never overridden:
matching humans are noted; differing humans are queued as conflicts.
"""
from __future__ import annotations
import csv, json, time
from collections import Counter
from pathlib import Path

import wvcore

ASSIST_TAG = "assistant_visual_review_attr-v1"
SIGLIP_TAG = "siglip2-so400m-224_zs-v1"
STATUS = "auto_approved"
REASON = "assistant_siglip_disagreement"
# Existing queue reasons that already cover a device-type conflict for an image (dedupe).
DEDUPE_REASONS = frozenset({
    "device_type_disagreement", "assistant_siglip_disagreement", "assistant_review_conflict",
})
# Conservative default; calibrated on val human labels vs SigLIP alone (~0.73 acc @ 0.6).
DEFAULT_THRESHOLD = 0.6
APPROVABLE = frozenset({"manual", "powered", "other_aid"})
AMBIGUOUS = frozenset({"ambiguous", "lookalike", "unclear"})
BAD_LICENSE = frozenset({"needs_ross", "exclude"})
CHILD_EXCLUDE = frozenset(getattr(wvcore, "ALWAYS_OUT_REASONS", ("child", "identifiable_person")))


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict):
            out.append(r)
    return out


def write_jsonl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


def append_jsonl(path, rec):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")


def load_assist(path):
    """{image_id: {device_type, device_conf, subtype, subtype_conf, raw}}"""
    out = {}
    for r in _jsonl(path):
        img = r.get("image_id")
        sug = r.get("suggestions") or {}
        if not isinstance(img, str) or not isinstance(sug, dict):
            continue
        dt = sug.get("device_type") or {}
        if not isinstance(dt, dict) or dt.get("value") in (None, "", []):
            continue
        st = sug.get("chair_subtype") or {}
        out[img] = {
            "device_type": wvcore.hval(dt.get("value")),
            "device_conf": float(dt["confidence"]) if isinstance(dt.get("confidence"), (int, float)) else None,
            "subtype": (st.get("value") if isinstance(st, dict) else None),
            "subtype_conf": (float(st["confidence"]) if isinstance(st, dict) and isinstance(st.get("confidence"), (int, float)) else None),
            "model_tag": r.get("model_tag") or ASSIST_TAG,
            "raw": r,
        }
    return out


def load_siglip(path, tag=SIGLIP_TAG):
    """{image_id: {device_type, conf, probs, flags, screening, child}} from the newest matching tag file."""
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("siglip*_zs-*.jsonl"), key=lambda x: x.stat().st_mtime)
        if tag:
            prefer = [f for f in files if f.name.startswith(tag)]
            files = prefer or files
        if not files:
            return {}, None
        p = files[-1]
    out = {}
    for r in _jsonl(p):
        img = r.get("image_id")
        sug = r.get("suggestions") or {}
        dt = sug.get("device_type") if isinstance(sug, dict) else None
        if not isinstance(img, str) or not isinstance(dt, dict) or dt.get("value") in (None, "", []):
            continue
        flags = list(r.get("flags") or [])
        screening = r.get("screening") or {}
        child = ("child" in flags) or (isinstance(screening.get("child"), (int, float)) and screening["child"] >= 0.5)
        out[img] = {
            "device_type": wvcore.hval(dt.get("value")),
            "conf": float(dt["confidence"]) if isinstance(dt.get("confidence"), (int, float)) else 0.0,
            "probs": dt.get("probs") or {},
            "flags": flags,
            "screening": screening,
            "child": bool(child),
            "model_tag": r.get("model_tag") or p.stem,
            "subtype": ((sug.get("chair_subtype") or {}) if isinstance(sug, dict) else {}).get("value"),
        }
    return out, p


def load_id_list(path):
    p = Path(path)
    if not p.exists():
        return set()
    return {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.strip().startswith("#")}


def load_license_status(path):
    p = Path(path)
    if not p.exists():
        return {}
    return {r["image_id"]: r.get("status") for r in csv.DictReader(open(p, encoding="utf-8")) if r.get("image_id")}


def load_flags(path):
    out = {}
    for r in _jsonl(path):
        if isinstance(r.get("image_id"), str):
            out.setdefault(r["image_id"], []).append(str(r.get("reason", "")))
    return out


def load_auto_approved(path):
    """Latest auto_approved record per image_id (file order)."""
    out = {}
    for r in _jsonl(path):
        if r.get("status") == STATUS and isinstance(r.get("image_id"), str):
            out[r["image_id"]] = r
    return out


def human_device_types(img, per, adjv):
    """{annotator: canonical device_type} including adj."""
    h = {}
    for a, fv in (per.get(img) or {}).items():
        if fv.get("device_type") not in (None, "", []):
            h[a] = wvcore.hval(fv["device_type"])
    a = adjv.get((img, "device_type"))
    if a and a[0] not in (None, "", []):
        h["adj"] = wvcore.hval(a[0])
    return h


def human_child_or_hold(img, per, adjv, hold):
    if img in hold:
        return True
    for a, fv in (per.get(img) or {}).items():
        if fv.get("exclude") in CHILD_EXCLUDE:
            return True
    a = adjv.get((img, "exclude"))
    if a and a[0] in CHILD_EXCLUDE:
        return True
    return False


def queue_device_ids(queue_path):
    """image_ids already queued for a device-type conflict reason."""
    ids = set()
    for r in _jsonl(queue_path):
        if r.get("reason") in DEDUPE_REASONS and isinstance(r.get("image_id"), str):
            ids.add(r["image_id"])
    return ids


def compare(assist, siglip, *, flags, child_ids, hold, per, adjv, license_status, threshold=DEFAULT_THRESHOLD):
    """Return (auto_rows, adj_rows, notes) without writing."""
    auto_rows, adj_rows, notes = [], [], []
    counts = Counter()
    by_class = Counter()
    subtype_agree = subtype_both = 0

    for img, a in sorted(assist.items()):
        av = a["device_type"]
        s = siglip.get(img)
        reasons = []

        if img in flags:
            reasons.append("flagged:" + ",".join(flags[img][:3]))
        if img in child_ids or (s and s.get("child")) or human_child_or_hold(img, per, adjv, hold):
            reasons.append("child")
        lic = license_status.get(img)
        if lic in BAD_LICENSE:
            reasons.append(f"license_{lic}")
        if av in AMBIGUOUS:
            reasons.append("assistant_ambiguous")

        if reasons:
            if "child" in reasons:
                counts["excluded_child"] += 1
            if any(x.startswith("flagged") for x in reasons):
                counts["excluded_flagged"] += 1
            if any(x.startswith("license_") for x in reasons):
                counts["excluded_license"] += 1
            if "assistant_ambiguous" in reasons:
                counts["excluded_ambiguous"] += 1
            # child / ambiguous always go to adjudication; flagged or bad-license alone are skipped
            if "child" in reasons or "assistant_ambiguous" in reasons:
                adj_rows.append(_adj(img, a, s, "excluded:" + "+".join(reasons), human_device_types(img, per, adjv)))
                counts["queued_exclusion"] += 1
            else:
                counts["skipped_exclusion"] += 1
                notes.append({"image_id": img, "note": "excluded", "reasons": reasons})
            continue

        if not s:
            counts["no_siglip"] += 1
            adj_rows.append(_adj(img, a, None, "no_siglip", human_device_types(img, per, adjv)))
            continue

        sv, sc = s["device_type"], s["conf"]
        # subtype agreement (informational only)
        if a.get("subtype") not in (None, "", "unknown") and s.get("subtype") not in (None, "", "unknown"):
            subtype_both += 1
            if a["subtype"] == s["subtype"]:
                subtype_agree += 1

        humans = human_device_types(img, per, adjv)
        agree = (av == sv) and (sc >= threshold) and (av in APPROVABLE)

        if not agree:
            if av != sv:
                counts["disagree"] += 1
                detail = f"assistant={av} siglip={sv} conf={sc:.4f}"
            else:
                counts["agree_low_conf"] += 1
                detail = f"agree={av} but siglip_conf={sc:.4f} < {threshold}"
                # low-conf agreement: do not auto-approve; leave for human review (no adj spam)
                notes.append({"image_id": img, "note": "agree_low_conf", "assistant": av, "siglip_conf": sc})
                continue
            adj_rows.append(_adj(img, a, s, detail, humans))
            continue

        # agreement above threshold
        if humans:
            if all(v == av for v in humans.values()):
                counts["agree_matches_human"] += 1
                notes.append({"image_id": img, "note": "matches_existing_human", "device_type": av, "humans": humans})
            else:
                counts["agree_conflicts_human"] += 1
                adj_rows.append(_adj(img, a, s, f"models agree on {av} but human labels differ: {humans}", humans,
                                     reason_override="assistant_siglip_disagreement"))
            continue

        counts["auto_approved"] += 1
        by_class[av] += 1
        auto_rows.append({
            "image_id": img,
            "device_type": av,
            "subtype": a.get("subtype"),
            "sources": [a.get("model_tag") or ASSIST_TAG, s.get("model_tag") or SIGLIP_TAG],
            "confidences": {"assistant": a.get("device_conf"), "siglip": sc},
            "threshold": threshold,
            "timestamp": now(),
            "status": STATUS,
        })

    summary = {
        "threshold": threshold,
        "assist_n": len(assist),
        "siglip_n": len(siglip),
        "counts": dict(counts),
        "auto_approved_by_class": dict(by_class),
        "subtype_agree": subtype_agree,
        "subtype_both": subtype_both,
        "n_auto_approved": len(auto_rows),
        "n_adjudication_candidates": len(adj_rows),
        "n_notes": len(notes),
    }
    return auto_rows, adj_rows, notes, summary


def _adj(img, a, s, detail, humans, reason_override=None):
    return {
        "image_id": img,
        "reason": reason_override or REASON,
        "category": "review",
        "source": "auto_consensus",
        "queued_at": now(),
        "field": "device_type",
        "assistant": {"device_type": a.get("device_type"), "confidence": a.get("device_conf"), "subtype": a.get("subtype")},
        "siglip": ({"device_type": s["device_type"], "confidence": s["conf"]} if s else None),
        "human_at_queue": humans or {},
        "detail": detail,
    }


def apply(auto_rows, adj_rows, *, auto_path, queue_path, replace_auto=True):
    """Write auto_approved store and append new adjudication rows (deduped)."""
    existing = queue_device_ids(queue_path)
    written_adj = 0
    skipped_dup = 0
    for r in adj_rows:
        if r["image_id"] in existing:
            skipped_dup += 1
            continue
        append_jsonl(queue_path, r)
        existing.add(r["image_id"])
        written_adj += 1
    if replace_auto:
        write_jsonl(auto_path, auto_rows)
    else:
        # merge: keep previous auto rows for images not revisited
        prev = load_auto_approved(auto_path)
        for r in auto_rows:
            prev[r["image_id"]] = r
        # drop autos that are no longer in auto_rows set if replace_auto - already handled
        write_jsonl(auto_path, [prev[k] for k in sorted(prev)])
    return {"auto_written": len(auto_rows), "adj_appended": written_adj, "adj_deduped": skipped_dup}


def review_status_with_auto(item, outcomes, auto_approved):
    """Like assistant_review.review_status but surfaces auto_approved when no human outcome yet."""
    n = sum((item["image_id"], x["field"]) in outcomes for x in item["fields"])
    if n == 0:
        if auto_approved and item["image_id"] in auto_approved:
            return STATUS
        return "pending"
    return "reviewed" if n == len(item["fields"]) else "partial"
