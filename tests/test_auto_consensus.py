#!/usr/bin/env python3
"""Tests for assistant+SigLIP auto_consensus (status=auto_approved, never a human label)."""
import csv, json, os, subprocess, sys, tempfile, uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="wv_ac_"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "server"))  # server wins over scripts/auto_consensus.py CLI
results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f" -- {detail}" if detail else ""), flush=True)

import wvcore, auto_consensus as AC
import assistant_review as AR

# --- unit: compare ---
assist = {
    "wv_a1": {"device_type": "manual", "device_conf": 0.7, "subtype": "folding", "model_tag": AC.ASSIST_TAG},
    "wv_a2": {"device_type": "powered", "device_conf": 0.7, "subtype": None, "model_tag": AC.ASSIST_TAG},
    "wv_a3": {"device_type": "manual", "device_conf": 0.7, "subtype": None, "model_tag": AC.ASSIST_TAG},  # disagree
    "wv_a4": {"device_type": "ambiguous", "device_conf": 0.5, "subtype": None, "model_tag": AC.ASSIST_TAG},
    "wv_a5": {"device_type": "manual", "device_conf": 0.7, "subtype": None, "model_tag": AC.ASSIST_TAG},  # child
    "wv_a6": {"device_type": "manual", "device_conf": 0.7, "subtype": None, "model_tag": AC.ASSIST_TAG},  # matches human
    "wv_a7": {"device_type": "powered", "device_conf": 0.7, "subtype": None, "model_tag": AC.ASSIST_TAG},  # conflicts human
    "wv_a8": {"device_type": "manual", "device_conf": 0.7, "subtype": None, "model_tag": AC.ASSIST_TAG},  # low conf
    "wv_a9": {"device_type": "other_aid", "device_conf": 0.7, "subtype": None, "model_tag": AC.ASSIST_TAG},  # flagged
}
siglip = {
    "wv_a1": {"device_type": "manual", "conf": 0.9, "child": False, "model_tag": AC.SIGLIP_TAG, "subtype": "folding"},
    "wv_a2": {"device_type": "powered", "conf": 0.85, "child": False, "model_tag": AC.SIGLIP_TAG, "subtype": None},
    "wv_a3": {"device_type": "powered", "conf": 0.9, "child": False, "model_tag": AC.SIGLIP_TAG, "subtype": None},
    "wv_a4": {"device_type": "manual", "conf": 0.9, "child": False, "model_tag": AC.SIGLIP_TAG, "subtype": None},
    "wv_a5": {"device_type": "manual", "conf": 0.9, "child": True, "model_tag": AC.SIGLIP_TAG, "subtype": None},
    "wv_a6": {"device_type": "manual", "conf": 0.9, "child": False, "model_tag": AC.SIGLIP_TAG, "subtype": None},
    "wv_a7": {"device_type": "powered", "conf": 0.9, "child": False, "model_tag": AC.SIGLIP_TAG, "subtype": None},
    "wv_a8": {"device_type": "manual", "conf": 0.4, "child": False, "model_tag": AC.SIGLIP_TAG, "subtype": None},
    "wv_a9": {"device_type": "other_aid", "conf": 0.9, "child": False, "model_tag": AC.SIGLIP_TAG, "subtype": None},
}
per = {
    "wv_a6": {"alice": {"device_type": "manual"}},
    "wv_a7": {"alice": {"device_type": "manual"}},
}
adjv = {}
flags = {"wv_a9": ["no_device"]}
auto_rows, adj_rows, notes, summ = AC.compare(
    assist, siglip, flags=flags, child_ids=set(), hold=set(), per=per, adjv=adjv,
    license_status={}, threshold=0.6,
)
auto_ids = {r["image_id"] for r in auto_rows}
check("auto-approves agreeing high-conf without human", auto_ids == {"wv_a1", "wv_a2"}, str(auto_ids))
check("auto rows have status=auto_approved", all(r["status"] == "auto_approved" for r in auto_rows))
check("auto rows never look like human labels", all("annotator" not in r and r["status"] == AC.STATUS for r in auto_rows))
check("disagree goes to adjudication", any(r["image_id"] == "wv_a3" and r["reason"] == AC.REASON for r in adj_rows))
check("ambiguous queued", any(r["image_id"] == "wv_a4" for r in adj_rows))
check("child queued not auto", "wv_a5" not in auto_ids and any(r["image_id"] == "wv_a5" for r in adj_rows))
check("matching human noted not auto", "wv_a6" not in auto_ids and any(n.get("note") == "matches_existing_human" and n["image_id"] == "wv_a6" for n in notes))
check("human conflict queued", any(r["image_id"] == "wv_a7" for r in adj_rows))
check("low conf not auto", "wv_a8" not in auto_ids)
check("flagged skipped", "wv_a9" not in auto_ids and summ["counts"].get("excluded_flagged") == 1)
check("subtype agreement counted", summ["subtype_agree"] == 1 and summ["subtype_both"] == 1)

# apply + dedupe
auto_path = TMP / "auto_approved.jsonl"
q_path = TMP / "queue.jsonl"
# pre-seed existing device_type_disagreement
AC.append_jsonl(q_path, {"image_id": "wv_a3", "reason": "device_type_disagreement", "queued_at": "2026-11-01T00:00:00Z"})
w = AC.apply(auto_rows, adj_rows, auto_path=auto_path, queue_path=q_path)
loaded = AC.load_auto_approved(auto_path)
check("wrote auto_approved store", set(loaded) == auto_ids)
check("deduped existing device disagreement", w["adj_deduped"] >= 1)
q = AC._jsonl(q_path)
check("no duplicate wv_a3", sum(1 for r in q if r["image_id"] == "wv_a3") == 1)

# review_status override
item = {"image_id": "wv_a1", "fields": [{"field": "device_type"}, {"field": "occupied"}]}
check("status auto_approved when no outcomes", AR.review_status(item, {}, loaded) == "auto_approved")
outcomes = {("wv_a1", "device_type"): {"action": "accept"}}
check("human outcome overrides auto_approved", AR.review_status(item, outcomes, loaded) == "partial")
outcomes2 = {("wv_a1", "device_type"): {"action": "accept"}, ("wv_a1", "occupied"): {"action": "accept"}}
check("full human review -> reviewed", AR.review_status(item, outcomes2, loaded) == "reviewed")

# --- API integration (scratch) ---
ids = [f"wv_{k:06d}" for k in range(1, 21)]
with open(TMP / "candidates.csv", "w", newline="", encoding="utf-8") as fh:
    wcsv = csv.DictWriter(fh, ["candidate_id", "source", "creator", "license", "needs_visual_screen"]); wcsv.writeheader()
    for k, i in enumerate(ids):
        wcsv.writerow({"candidate_id": i, "source": "commons", "creator": f"p{k}", "license": "CC-BY-4.0", "needs_visual_screen": "no"})
for d in ("auto", "assist", "block", "images", "images_blurred", "exports", "cache"):
    (TMP / d).mkdir(exist_ok=True)
(TMP / "blur.jsonl").write_text("")
(TMP / "block" / "removed_test.txt").write_text("")
(TMP / "block" / "safety_hold.csv").write_text("image_id,flagged,flagged_by,reason\n")
# assist suggestions
rows = []
for i in ids:
    rows.append({"image_id": i, "model_tag": AC.ASSIST_TAG, "source": "assistant_visual_review",
                 "suggestions": {"device_type": {"value": "manual", "confidence": 0.8}}})
(TMP / "assist" / f"{AC.ASSIST_TAG}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
(TMP / "assist" / "assistant_visual_review_flagged.ndjson").write_text("")
# siglip agrees on first 5, disagrees on next 5
sig_rows = []
for i in ids[:5]:
    sig_rows.append({"image_id": i, "model_tag": AC.SIGLIP_TAG, "flags": [], "screening": {"child": 0.01},
                     "suggestions": {"device_type": {"value": "manual", "confidence": 0.9, "probs": {"manual": 0.9}}}})
for i in ids[5:10]:
    sig_rows.append({"image_id": i, "model_tag": AC.SIGLIP_TAG, "flags": [], "screening": {"child": 0.01},
                     "suggestions": {"device_type": {"value": "powered", "confidence": 0.9, "probs": {"powered": 0.9}}}})
for i in ids[10:]:
    sig_rows.append({"image_id": i, "model_tag": AC.SIGLIP_TAG, "flags": [], "screening": {"child": 0.01},
                     "suggestions": {"device_type": {"value": "manual", "confidence": 0.9, "probs": {"manual": 0.9}}}})
(TMP / "auto" / f"{AC.SIGLIP_TAG}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sig_rows))
# pre-write auto_approved for first 3
AC.write_jsonl(TMP / "assist" / "auto_approved.jsonl", [
    {"image_id": ids[0], "device_type": "manual", "sources": [AC.ASSIST_TAG, AC.SIGLIP_TAG],
     "confidences": {"assistant": 0.8, "siglip": 0.9}, "threshold": 0.6, "timestamp": "2026-11-09T00:00:00Z", "status": "auto_approved"},
    {"image_id": ids[1], "device_type": "manual", "sources": [AC.ASSIST_TAG, AC.SIGLIP_TAG],
     "confidences": {"assistant": 0.8, "siglip": 0.9}, "threshold": 0.6, "timestamp": "2026-11-09T00:00:00Z", "status": "auto_approved"},
])
Q = TMP / "queue.jsonl"
env = {"WV_DB": str(TMP / "scratch.db"), "WV_CANDIDATES": str(TMP / "candidates.csv"), "WV_CONTRIBUTED": str(TMP / "contributed.csv"),
       "WV_IMAGE_DIR": str(TMP / "images"), "WV_BLUR_DIR": str(TMP / "images_blurred"), "WV_EXPORT_DIR": str(TMP / "exports"),
       "WV_AUTO_DIR": str(TMP / "auto"), "WV_REVIEW_AUTO_DIR": str(TMP / "assist"), "WV_BLOCKLIST_DIR": str(TMP / "block"), "WV_IMAGE_PROXY": "0",
       "WV_ACCESS_CODE": "", "WV_BLUR_MANIFESTS": str(TMP / "blur.jsonl"), "WV_CACHE_DIR": str(TMP / "cache"),
       "WV_ASSIGN_WITHOUT_LOCAL_IMAGE": "1", "WV_ADJ_QUEUE": str(Q), "WV_AUTO_APPROVED": str(TMP / "assist" / "auto_approved.jsonl")}
for k in ("WV_HF_DATASET", "WV_ASSIST_REVIEW_FILE", "WV_ASSIST_REVIEW_FLAGS", "WV_REVIEW_OUTCOMES"):
    os.environ.pop(k, None)
os.environ.update(env)
from fastapi.testclient import TestClient
import main
with TestClient(main.app) as cl:
    keys = cl.get("/api/filter/keys").json()
    check("filter keys include auto_approved", "auto_approved" in keys["keys"]["review_status"])
    check("filter keys include model_consensus label_source", "model_consensus" in keys["keys"]["label_source"])
    q = cl.get("/api/assistant_review/queue", params={"status": "auto_approved"}).json()
    got = {it["image_id"] for it in q["items"]}
    check("queue status=auto_approved returns consensus images", got == {ids[0], ids[1]}, str(got))
    check("UI tag text available via status field", all(it["review_status"] == "auto_approved" for it in q["items"]))
    f = cl.get("/api/filter", params={"q": "review_status:auto_approved", "limit": 1000}).json()
    check("filter review_status:auto_approved", set(f["ids"]) == {ids[0], ids[1]}, str(f["ids"]))
    # human accept overrides
    r = cl.post("/api/assistant_review/decide", json={"reviewer": "carol", "image_id": ids[0],
          "decisions": [{"field": "device_type", "action": "accept"}]}).json()
    check("human accept writes label not auto status", any(x.get("result") == "label_written" for x in r["results"]))
    q2 = cl.get("/api/assistant_review/queue", params={"status": "auto_approved"}).json()
    check("after human review image leaves auto_approved", ids[0] not in {it["image_id"] for it in q2["items"]})
    html = cl.get("/review_assistant.html").text
    check("UI offers auto-approved filter option", "auto-approved (model consensus)" in html)

# make_splits --include-auto-approved
import make_splits as MS
check("frozen outs listed", "splits_v1.csv" in MS.FROZEN_SPLIT_OUTS)
# refuse frozen
import subprocess as sp
r = sp.run([sys.executable, str(ROOT / "scripts" / "make_splits.py"), "--include-auto-approved",
            "--out", str(ROOT / "data" / "splits_v1.csv"), "--db", str(TMP / "scratch.db"),
            "--events", str(TMP / "noevents")], capture_output=True, text=True)
check("refuses frozen splits_v1 with --include-auto-approved", r.returncode != 0 and "refusing" in (r.stderr + r.stdout))

failed = [n for n, ok in results if not ok]
print(f"\n{sum(ok for _, ok in results)}/{len(results)} passed")
sys.exit(1 if failed else 0)
