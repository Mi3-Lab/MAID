#!/usr/bin/env python3
"""Corrections: any human can edit any image via kind=correction; latest wins; child-exclude clearance refused."""
import csv, json, os, sys, tempfile, uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="wv_corr_"))
sys.path.insert(0, str(ROOT / "server"))
results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond))); print(("PASS " if cond else "FAIL ") + name + (f" -- {detail}" if detail else ""), flush=True)

import wvcore
# unit: validate + fold
def ev(img, ann, field, value, kind="label", **kw):
    e = {"uuid": str(uuid.uuid4()), "image_id": img, "annotator": ann, "field": field, "value": value, "kind": kind,
         "ts": "2026-11-08T12:00:00.000Z", "device_id": "t"}; e.update(kw); return e
ok, why = wvcore.validate(ev("wv_000001", "samir", "device_type", "manual", kind="correction", previous="powered", previous_by="shaaz"))
check("validate correction with previous", ok is not None, why)
check("previous_by may list consensus annotators with +", wvcore.validate({"uuid": str(__import__("uuid").uuid4()), "image_id": "wv_000001", "annotator": "carol", "field": "device_type", "value": "powered", "kind": "correction", "previous": "manual", "previous_by": "devin+samir+shaaz", "ts": "2026-11-09T20:00:00.000Z", "device_id": "t"})[0] is not None)
ok, why = wvcore.validate(ev("wv_000001", "samir", "device_type", "manual", kind="correction", source="auto_review", model_tag="x"))
check("correction refuses model provenance", ok is None)
per, adj = wvcore.fold([
    ev("wv_000001", "shaaz", "device_type", "powered"),
    ev("wv_000001", "shaaz", "_status", "done"),
    ev("wv_000001", "samir", "device_type", "manual", kind="correction", previous="powered", previous_by="shaaz",
       ts="2026-11-08T13:00:00.000Z"),
])
check("correction supersedes resolved device_type", adj[("wv_000001", "device_type")][0] == "manual" and adj[("wv_000001", "device_type")][2] == "corrected")
st = wvcore.image_status("wv_000001", per, adj)
check("image_status agreement=corrected", st["fields"]["device_type"][1] == "corrected" and st["fields"]["device_type"][0] == "manual")

# server
ids = ["wv_000001", "wv_000002", "wv_000003"]
with open(TMP / "candidates.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, ["candidate_id", "source", "creator", "license", "needs_visual_screen"]); w.writeheader()
    for k, i in enumerate(ids): w.writerow({"candidate_id": i, "source": "commons", "creator": f"p{k}", "license": "CC-BY-4.0", "needs_visual_screen": "no"})
for d in ("auto", "block", "images", "images_blurred", "assist"): (TMP / d).mkdir()
(TMP / "blur.jsonl").write_text("")
(TMP / "queue.jsonl").write_text("")
# release split stub so in_v10_release works
rel = TMP / "release" / "splits"; rel.mkdir(parents=True)
(rel / "test.txt").write_text("wv_000002\n")
env = {"WV_DB": str(TMP / "scratch.db"), "WV_CANDIDATES": str(TMP / "candidates.csv"), "WV_CONTRIBUTED": str(TMP / "contributed.csv"),
       "WV_IMAGE_DIR": str(TMP / "images"), "WV_BLUR_DIR": str(TMP / "images_blurred"), "WV_EXPORT_DIR": str(TMP / "exports"),
       "WV_AUTO_DIR": str(TMP / "auto"), "WV_REVIEW_AUTO_DIR": str(TMP / "assist"), "WV_BLOCKLIST_DIR": str(TMP / "block"),
       "WV_IMAGE_PROXY": "0", "WV_ACCESS_CODE": "", "WV_BLUR_MANIFESTS": str(TMP / "blur.jsonl"), "WV_CACHE_DIR": str(TMP / "cache"),
       "WV_ASSIGN_WITHOUT_LOCAL_IMAGE": "1", "WV_ADJ_QUEUE": str(TMP / "queue.jsonl")}
os.environ.pop("WV_HF_DATASET", None); os.environ.update(env)
# point ROOT release to TMP by monkeypatching after import is hard; instead write release under real ROOT? 
# Patch: create release under TMP and set via symlink - on Windows use copy into wheelview_v11/release/splits only if missing test ids
# Safer: monkeypatch main.ROOT after import - actually get_image uses CANDIDATES; history uses v10_split_ids from ROOT/release
# Write a tiny release under the real repo? Don't. Monkeypatch after import:
from fastapi.testclient import TestClient
import main
main.ROOT = TMP  # so v10_split_ids and corrections log use TMP
main.V10_SPLIT_IDS = None
main.ADJ_QUEUE = TMP / "queue.jsonl"
# also need CANDIDATES from env - already loaded at import from env... actually CANDIDATES loaded at import from WV_CANDIDATES
# Re-import is messy; CANDIDATES should already be from TMP if env was set before import. Good.

def E(img, ann, field, value, kind="label", **kw):
    d = {"uuid": str(uuid.uuid4()), "image_id": img, "annotator": ann, "field": field, "value": value, "kind": kind,
         "ts": "2026-11-08T12:00:00.000Z", "device_id": "t-" + ann}; d.update(kw); return d

with TestClient(main.app) as cl:
    # shaaz labels an image
    r = cl.post("/api/events", json={"events": [
        E("wv_000001", "shaaz", "device_type", "powered"), E("wv_000001", "shaaz", "exclude", ""),
        E("wv_000001", "shaaz", "_status", "done")]})
    check("shaaz labels accepted", r.status_code == 200 and len(r.json()["accepted"]) == 3, r.text[:200])
    # samir corrects device_type
    r = cl.post("/api/events", json={"events": [
        E("wv_000001", "samir", "device_type", "manual", kind="correction", previous="powered", previous_by="shaaz",
          ts="2026-11-08T14:00:00.000Z")]}).json()
    check("correction accepted", len(r["accepted"]) == 1 and not r["rejected"], json.dumps(r))
    h = cl.get("/api/images/wv_000001/history").json()
    check("history lists both annotators", "shaaz" in h["annotators"] and h["status"]["fields"]["device_type"][0] == "manual")
    check("history agreement corrected", h["status"]["fields"]["device_type"][1] == "corrected")
    check("history includes correction event", any(e.get("kind") == "correction" for e in h["events"]))
    # child exclude clearance refused
    cl.post("/api/events", json={"events": [
        E("wv_000003", "shaaz", "device_type", "manual"), E("wv_000003", "shaaz", "exclude", "child"),
        E("wv_000003", "shaaz", "_status", "done")]})
    r = cl.post("/api/events", json={"events": [
        E("wv_000003", "samir", "exclude", "", kind="correction", previous="child", previous_by="shaaz",
          ts="2026-11-08T15:00:00.000Z")]}).json()
    check("clearing child exclude refused", r["rejected"] and "adjudication" in r["rejected"][0]["reason"])
    q = [json.loads(l) for l in (TMP / "queue.jsonl").read_text().splitlines() if l.strip()]
    check("queued for adjudication", any(x.get("reason") == "child_exclude_clearance_requested" for x in q))
    h3 = cl.get("/api/images/wv_000003/history").json()
    check("child exclude still present", h3["status"]["fields"]["exclude"][0] == "child")
    # v1.0 release image correction logged
    cl.post("/api/events", json={"events": [
        E("wv_000002", "shaaz", "device_type", "manual"), E("wv_000002", "shaaz", "_status", "done")]})
    cl.post("/api/events", json={"events": [
        E("wv_000002", "samir", "device_type", "powered", kind="correction", previous="manual", previous_by="shaaz",
          ts="2026-11-08T16:00:00.000Z")]})
    log = TMP / "var" / "corrections_on_v10.jsonl"
    check("v10 correction logged (not altering release files)", log.exists() and "wv_000002" in log.read_text())
    check("history flags in_v10_release", cl.get("/api/images/wv_000002/history").json().get("in_v10_release") is True)
    # page has jump UI
    html = cl.get("/").text
    check("annotator page has Go-to UI", 'id="jump"' in html and 'id="btnJump"' in html)

n = sum(ok for _, ok in results)
print(f"\n{n}/{len(results)} passed; temp {TMP}")
sys.exit(0 if n == len(results) else 1)
