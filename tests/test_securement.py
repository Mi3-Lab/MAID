#!/usr/bin/env python3
"""securement_hardware retired in schema v7; attr-v4 low-conf and vehicle API still work."""
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
check("schema v7: securement retired from live fields", wvcore.SCHEMA["version"] >= 7 and "securement_hardware" in wvcore.RETIRED and "securement_hardware" not in wvcore.FIELD_OPTS)
check("field not in COMPARE_V11", "securement_hardware" not in wvcore.COMPARE_V11)
check("retired options kept", wvcore.RETIRED_OPTS["securement_hardware"] == ["docking_bracket", "tiedown_points", "not_visible", "unclear"])
check("definitions guidance present", "NOT a safety certification" in (wvcore.SCHEMA["definitions"]["securement_hardware"].get("_vehicle_guidance") or ""))
e, why = wvcore.validate({"uuid": str(uuid.uuid4()), "image_id": "wv_000001", "annotator": "alice", "field": "securement_hardware",
                          "value": "docking_bracket", "kind": "label", "ts": "2026-11-09T00:00:00.000Z", "device_id": "t"})
check("new write of securement_hardware rejected", e is None and "retired" in (why or ""), why)
e2, why2 = wvcore.validate({"uuid": str(uuid.uuid4()), "image_id": "wv_000001", "annotator": "alice", "field": "securement_hardware",
                            "value": "ez_lock", "kind": "label", "ts": "2026-11-09T00:00:00.000Z", "device_id": "t"})
check("reject unknown option (still invalid even if retired)", e2 is None)
e3, why3 = wvcore.validate({"uuid": str(uuid.uuid4()), "image_id": "wv_000001", "annotator": "alice", "field": "securement_hardware",
                            "value": "docking_bracket", "kind": "label", "ts": "2026-11-09T00:00:00.000Z", "device_id": "t"}, allow_retired=True)
check("historical securement accepted with allow_retired", e3 is not None, why3)

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
check("score_attr does not emit retired securement_hardware", "securement_hardware" not in out[0]["suggestions"], str(out[0]))
# Existing attr-v4 jsonl files and the vehicle API still use the low-conf->unclear rule; prompts stay in P_V11_RETIRED.
check("retired prompts and low-conf threshold kept for existing model files",
      "securement_hardware" in AA.P_V11_RETIRED and AA.ATTR_LOW_CONF.get("securement_hardware") == 0.45)

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
    check("filter has no live securement_hardware key", "securement_hardware" not in keys["keys"])
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
    check("annotator help does not offer a live securement row", "docking bracket / tiedown" not in html.lower() and "securement hardware were removed" in html.lower())

failed = [n for n, ok in results if not ok]
print(f"{sum(ok for _, ok in results)}/{len(results)} passed")
sys.exit(1 if failed else 0)
