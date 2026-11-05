#!/usr/bin/env python3
"""Tests for the adjudication review queue (var/adjudication/queue.jsonl -> /api/adjudication_queue).
Scratch DB/queue in a temp dir; no live data.   python tests/test_adjudication_queue.py
"""
import csv, json, os, subprocess, sys, tempfile, uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="wv_adjq_"))
sys.path.insert(0, str(ROOT / "server"))
results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond))); print(("PASS " if cond else "FAIL ") + name + (f" -- {detail}" if detail else ""), flush=True)

ids = ["wv_000001", "wv_000002", "wv_000003"]
with open(TMP / "candidates.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, ["candidate_id", "source", "creator", "license", "needs_visual_screen"]); w.writeheader()
    for k, i in enumerate(ids): w.writerow({"candidate_id": i, "source": "commons", "creator": f"p{k}", "license": "CC-BY-4.0", "needs_visual_screen": "no"})
for d in ("auto", "block", "images", "images_blurred"): (TMP / d).mkdir()
(TMP / "blur.jsonl").write_text("")
(TMP / "block" / "removed_test.txt").write_text("wv_000003\n")
Q = TMP / "queue.jsonl"
rows = [{"image_id": "wv_000001", "reason": "device_type_disagreement", "source": "assistant_visual_review", "category": "a", "assistant": {"device_type": "powered"}},
        {"image_id": "wv_000002", "reason": "child_visible", "source": "assistant_visual_review", "category": "b"},
        {"image_id": "wv_000003", "reason": "child_visible", "source": "assistant_visual_review", "category": "c"},
        {"image_id": "wv_000002", "reason": "child_visible", "source": "assistant_visual_review", "category": "b", "detail": "updated"},
        {"image_id": "wv_000001", "reason": "", "source": "x"}]
Q.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n\n", encoding="utf-8")
env = {"WV_DB": str(TMP / "scratch.db"), "WV_CANDIDATES": str(TMP / "candidates.csv"), "WV_CONTRIBUTED": str(TMP / "contributed.csv"),
       "WV_IMAGE_DIR": str(TMP / "images"), "WV_BLUR_DIR": str(TMP / "images_blurred"), "WV_EXPORT_DIR": str(TMP / "exports"),
       "WV_AUTO_DIR": str(TMP / "auto"), "WV_REVIEW_AUTO_DIR": str(TMP / "auto"), "WV_BLOCKLIST_DIR": str(TMP / "block"), "WV_IMAGE_PROXY": "0",
       "WV_ACCESS_CODE": "", "WV_BLUR_MANIFESTS": str(TMP / "blur.jsonl"), "WV_CACHE_DIR": str(TMP / "cache"),
       "WV_ASSIGN_WITHOUT_LOCAL_IMAGE": "1", "WV_ADJ_QUEUE": str(Q)}
os.environ.pop("WV_HF_DATASET", None); os.environ.update(env)
from fastapi.testclient import TestClient
import main
def ev(img, ann, value):
    return {"uuid": str(uuid.uuid4()), "image_id": img, "annotator": ann, "field": "device_type", "value": value, "kind": "label",
            "ts": "2026-11-05T10:00:00.000Z", "device_id": "t-" + ann}
with TestClient(main.app) as cl:
    check("events accepted", cl.post("/api/events", json={"events": [ev("wv_000001", "alice", "manual")]}).status_code == 200)
    n_before = len(cl.get("/api/events").json().get("events", []))
    q = cl.get("/api/adjudication_queue").json()
    check("invalid/blank lines skipped, duplicates collapse", q["total"] == 3, json.dumps(q)[:300])
    check("counts by reason", q["counts"] == {"device_type_disagreement": 1, "child_visible": 2}, json.dumps(q["counts"]))
    it = {(r["image_id"], r["reason"]): r for r in q["items"]}
    check("later line wins", it[("wv_000002", "child_visible")].get("detail") == "updated")
    a = it[("wv_000001", "device_type_disagreement")]
    check("joins current human labels", a["human_device_type"] == {"alice": "manual"} and a["assistant"] == {"device_type": "powered"})
    check("source + reason recorded", all(r["source"] == "assistant_visual_review" and r["reason"] for r in q["items"]))
    b = it[("wv_000003", "child_visible")]
    check("blocked image listed without image url", b["blocked"] and b["img"] == "")
    check("unblocked image has image url", a["img"].startswith("/img/"))
    f = cl.get("/api/adjudication_queue", params={"reason": "child_visible"}).json()
    check("reason filter", len(f["items"]) == 2 and f["total"] == 3)
    check("queue read does not write events", len(cl.get("/api/events").json().get("events", [])) == n_before)
    h = cl.get("/adjudicate.html").text
    check("adjudicate page has review queue section", 'id="queue"' in h and 'id="qreason"' in h)
os.environ["WV_ADJ_QUEUE"] = str(TMP / "missing.jsonl")
check("missing queue file = empty", main.load_adj_queue(TMP / "missing.jsonl") == [])
check("default queue path is gitignored", subprocess.run(["git", "check-ignore", "-q", "--no-index", "var/adjudication/queue.jsonl"], cwd=ROOT).returncode == 0)
n = sum(ok for _, ok in results)
print(f"\n{n}/{len(results)} passed; temp dir {TMP}")
sys.exit(0 if n == len(results) else 1)
