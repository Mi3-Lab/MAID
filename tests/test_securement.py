#!/usr/bin/env python3
"""securement_hardware schema v5 + attr-v4 low-conf -> unclear + vehicle API guidance."""
import csv, json, os, sys, tempfile, uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="wv_sec_"))
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "scripts"))
results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond))); print(("PASS " if cond else "FAIL ") + name + (f" -- {detail}" if detail else ""), flush=True)

import wvcore
check("schema v5", wvcore.SCHEMA["version"] == 5)
check("field in COMPARE_V11", "securement_hardware" in wvcore.COMPARE_V11)
check("options", wvcore.FIELD_OPTS["securement_hardware"] == ["docking_bracket", "tiedown_points", "not_visible", "unclear"])
check("definitions guidance present", "NOT a safety certification" in (wvcore.SCHEMA["definitions"]["securement_hardware"].get("_vehicle_guidance") or ""))
e, why = wvcore.validate({"uuid": str(uuid.uuid4()), "image_id": "wv_000001", "annotator": "alice", "field": "securement_hardware",
                          "value": "docking_bracket", "kind": "label", "ts": "2026-11-09T00:00:00.000Z", "device_id": "t"})
check("validate docking_bracket", e is not None, why)
e2, why2 = wvcore.validate({"uuid": str(uuid.uuid4()), "image_id": "wv_000001", "annotator": "alice", "field": "securement_hardware",
                            "value": "ez_lock", "kind": "label", "ts": "2026-11-09T00:00:00.000Z", "device_id": "t"})
check("reject unknown option", e2 is None)

import auto_annotate as AA
import numpy as np
check("attr-v4", AA.ATTR_PROMPT_VERSION == "attr-v4")
# Fake bank: securement only, force low-confidence by nearly equal embeddings
# Build a tiny bank_v11 with 3 options (no unclear) and nearly uniform scores via equal text vectors
d = 8
bank = {"securement_hardware": {
    "docking_bracket": np.ones(d, dtype="float32") / np.sqrt(d),
    "tiedown_points": np.ones(d, dtype="float32") / np.sqrt(d),
    "not_visible": np.ones(d, dtype="float32") / np.sqrt(d),
}}
E = np.ones((2, d), dtype="float32") / np.sqrt(d)
# Make first image slightly prefer docking but still below thr by flattening with TEMP... 
# With equal vectors softmax is uniform ~0.333 < 0.45 -> unclear
out = AA.score_attr(E, bank)
check("low-conf maps to unclear", out[0]["suggestions"]["securement_hardware"]["value"] == "unclear", str(out[0]))
# High-conf: one option much closer
bank2 = {"securement_hardware": {
    "docking_bracket": np.array([1,0,0,0,0,0,0,0], dtype="float32"),
    "tiedown_points": np.array([0,1,0,0,0,0,0,0], dtype="float32"),
    "not_visible": np.array([0,0,1,0,0,0,0,0], dtype="float32"),
}}
E2 = np.array([[1,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0]], dtype="float32")
out2 = AA.score_attr(E2, bank2)
check("high-conf docking kept", out2[0]["suggestions"]["securement_hardware"]["value"] == "docking_bracket"
      and out2[0]["suggestions"]["securement_hardware"]["confidence"] >= 0.45, str(out2[0]))
check("high-conf not_visible kept", out2[1]["suggestions"]["securement_hardware"]["value"] == "not_visible")

# API
ids = ["wv_000001"]
with open(TMP / "candidates.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, ["candidate_id", "source", "creator", "license", "needs_visual_screen"]); w.writeheader()
    w.writerow({"candidate_id": "wv_000001", "source": "commons", "creator": "p", "license": "CC-BY-4.0", "needs_visual_screen": "no"})
for dname in ("auto", "assist", "block", "images", "images_blurred", "exports", "cache"):
    (TMP / dname).mkdir(exist_ok=True)
(TMP / "blur.jsonl").write_text("")
(TMP / "block" / "removed_test.txt").write_text("")
(TMP / "block" / "safety_hold.csv").write_text("image_id,flagged,flagged_by,reason\n")
(TMP / "auto" / "siglip2-so400m-224_attr-v4.jsonl").write_text(json.dumps({
    "image_id": "wv_000001", "model_tag": "siglip2-so400m-224_attr-v4",
    "suggestions": {"securement_hardware": {"value": "docking_bracket", "confidence": 0.82}}
}) + "\n")
env = {"WV_DB": str(TMP / "scratch.db"), "WV_CANDIDATES": str(TMP / "candidates.csv"), "WV_CONTRIBUTED": str(TMP / "contributed.csv"),
       "WV_IMAGE_DIR": str(TMP / "images"), "WV_BLUR_DIR": str(TMP / "images_blurred"), "WV_EXPORT_DIR": str(TMP / "exports"),
       "WV_AUTO_DIR": str(TMP / "auto"), "WV_REVIEW_AUTO_DIR": str(TMP / "assist"), "WV_BLOCKLIST_DIR": str(TMP / "block"),
       "WV_IMAGE_PROXY": "0", "WV_ACCESS_CODE": "", "WV_BLUR_MANIFESTS": str(TMP / "blur.jsonl"), "WV_CACHE_DIR": str(TMP / "cache"),
       "WV_ASSIGN_WITHOUT_LOCAL_IMAGE": "1", "WV_AUTO_ATTR_MODEL": "siglip2-so400m-224_attr-v4"}
for k in list(os.environ):
    if k.startswith("WV_"): os.environ.pop(k, None)
os.environ.update(env)
from fastapi.testclient import TestClient
import main
with TestClient(main.app) as cl:
    keys = cl.get("/api/filter/keys").json()
    check("filter key securement_hardware", "securement_hardware" in keys["keys"])
    r = cl.get("/api/vehicle/securement/wv_000001").json()
    check("vehicle API returns label+confidence", r["label"] == "docking_bracket" and r["confidence"] == 0.82, str(r))
    check("vehicle API guidance warns not certification", "NOT a safety certification" in (r.get("guidance") or "").upper() or "not a safety" in (r.get("guidance") or "").lower(), str(r.get("guidance"))[:120])
    check("high conf docking does not force defer", r["defer_to_rider"] is False)
    # low conf -> defer
    (TMP / "auto" / "siglip2-so400m-224_attr-v4.jsonl").write_text(json.dumps({
        "image_id": "wv_000001", "model_tag": "siglip2-so400m-224_attr-v4",
        "suggestions": {"securement_hardware": {"value": "docking_bracket", "confidence": 0.4}}
    }) + "\n")
    # force reload
    main._AUTO["sig"] = None
    r2 = cl.get("/api/vehicle/securement/wv_000001").json()
    check("low conf defers to rider", r2["defer_to_rider"] is True, str(r2))
    html = cl.get("/").text
    check("annotator help mentions securement", "Securement hardware" in html or "securement" in html.lower())

failed = [n for n, ok in results if not ok]
print(f"{sum(ok for _, ok in results)}/{len(results)} passed")
sys.exit(1 if failed else 0)
