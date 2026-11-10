#!/usr/bin/env python3
"""Schema v6: retired fields ignored on export, device_use_case, and editable latest labels (append-only corrections)."""
import csv, json, os, sys, tempfile, uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="wv_v6_"))
sys.path.insert(0, str(ROOT / "server"))
results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond))); print(("PASS " if cond else "FAIL ") + name + (f" -- {detail}" if detail else ""), flush=True)

import wvcore

def ev(img, ann, field, value, kind="label", **kw):
    e = {"uuid": str(uuid.uuid4()), "image_id": img, "annotator": ann, "field": field, "value": value, "kind": kind,
         "ts": kw.pop("ts", "2026-11-09T18:00:00.000Z"), "device_id": "t-" + ann}
    e.update(kw); return e

check("schema version 7", wvcore.SCHEMA.get("version") == 7)
check("securement_hardware is retired, not live", "securement_hardware" in wvcore.RETIRED and "securement_hardware" not in wvcore.FIELD_OPTS and "securement_hardware" not in wvcore.COMPARE_V11)
check("device_use_case sits above device_type",
      [f["name"] for f in wvcore.SCHEMA["fields"][:2]] == ["device_use_case", "device_type"])
check("device_use_case option labels present",
      wvcore.SCHEMA["fields"][0]["option_labels"]["specialized"].startswith("Specialized"))

# historical lighting in the log must not appear in labels.csv
v10_fields = [f for f in wvcore.SCHEMA["fields"] if not f.get("since")]
events = []
for ann, dev in (("alice", "manual"), ("bob", "manual")):
    for f in v10_fields:
        events.append(ev("wv_000001", ann, f["name"], dev if f["name"] == "device_type" else f["options"][0]))
    events += [ev("wv_000001", ann, "lighting", "daylight"), ev("wv_000001", ann, "exclude", ""), ev("wv_000001", ann, "_status", "done")]
cands = {"wv_000001": {"source": "commons"}}
outdir = TMP / "der"; outdir.mkdir()
# lighting events must be readable then ignored
ok, why = wvcore.validate(ev("wv_000001", "alice", "lighting", "daylight"), allow_retired=True)
check("allow_retired reads lighting", ok is not None, why)
validated = [wvcore.validate(e, allow_retired=True)[0] for e in events]
validated = [e for e in validated if e]
wvcore.materialize(validated, cands, str(outdir))
cols = next(csv.reader(open(outdir / "labels.csv", encoding="utf-8")))
check("labels.csv has no lighting column", "lighting" not in cols, str(cols))
row = next(csv.DictReader(open(outdir / "labels.csv", encoding="utf-8")))
check("labels.csv still has device_type", row["device_type"] == "manual")

# correction of someone else's device_type is latest-wins
per, adj = wvcore.fold(validated + [
    ev("wv_000001", "carol", "device_type", "powered", kind="correction", previous="manual", previous_by="alice",
       ts="2026-11-09T19:00:00.000Z")
])
st = wvcore.image_status("wv_000001", per, adj)
check("correction supersedes resolved device_type", st["fields"]["device_type"][0] == "powered" and st["fields"]["device_type"][1] == "corrected")

# live API
ids = ["wv_000001"]
with open(TMP / "candidates.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, ["candidate_id", "source", "creator", "license", "needs_visual_screen"]); w.writeheader()
    w.writerow({"candidate_id": "wv_000001", "source": "commons", "creator": "p", "license": "CC-BY-4.0", "needs_visual_screen": "no"})
for d in ("auto", "block", "images", "images_blurred", "assist"): (TMP / d).mkdir()
(TMP / "blur.jsonl").write_text("")
from PIL import Image
Image.new("RGB", (64, 64), (10, 20, 30)).save(TMP / "images" / "wv_000001.jpg")
Image.new("RGB", (64, 64), (10, 20, 30)).save(TMP / "images_blurred" / "wv_000001.jpg")
env = {"WV_DB": str(TMP / "scratch.db"), "WV_CANDIDATES": str(TMP / "candidates.csv"), "WV_CONTRIBUTED": str(TMP / "contributed.csv"),
       "WV_IMAGE_DIR": str(TMP / "images"), "WV_BLUR_DIR": str(TMP / "images_blurred"), "WV_EXPORT_DIR": str(TMP / "exports"),
       "WV_AUTO_DIR": str(TMP / "auto"), "WV_BLOCKLIST_DIR": str(TMP / "block"), "WV_IMAGE_PROXY": "0", "WV_ACCESS_CODE": "",
       "WV_BLUR_MANIFESTS": str(TMP / "blur.jsonl"), "WV_CACHE_DIR": str(TMP / "cache"), "WV_ASSIGN_WITHOUT_LOCAL_IMAGE": "1"}
os.environ.pop("WV_HF_DATASET", None); os.environ.update(env)
(TMP / "contributed.csv").write_text("image_id,status\n", encoding="utf-8")

from fastapi.testclient import TestClient
import importlib
import main
importlib.reload(main)
with TestClient(main.app) as cl:
    sch = cl.get("/api/schema").json()
    names = [f["name"] for f in sch["fields"]]
    check("API schema has no retired live fields", all(x not in names for x in ("weather", "lighting", "environment", "ground_wet", "size_class", "ramp_deploy", "ramp_reasons")))
    check("API schema has device_use_case first", names[0] == "device_use_case")
    keys = cl.get("/api/filter/keys").json()["keys"]
    check("filter has device_use_case, not weather/lighting/size_class", "device_use_case" in keys and "weather" not in keys and "lighting" not in keys and "size_class" not in keys and "size_basis" not in keys)
    # two annotators finish v1.0 (+ use case later)
    evs = []
    def E(ann, field, value, kind="label", **kw):
        d = ev("wv_000001", ann, field, value, kind, **kw); return d
    for ann in ("alice", "bob"):
        evs += [E(ann, f["name"], "manual" if f["name"]=="device_type" else f["options"][0]) for f in v10_fields]
        evs += [E(ann, "exclude", ""), E(ann, "_status", "done")]
    r = cl.post("/api/events", json={"events": evs}).json()
    check("v1.0 events accepted", len(r.get("accepted") or []) == len(evs), json.dumps({k: (len(v) if isinstance(v, list) else v) for k,v in r.items()}))
    # new write of lighting rejected
    r = cl.post("/api/events", json={"events": [E("alice", "lighting", "daylight")]}).json()
    check("API rejects new lighting event", r.get("rejected"), json.dumps(r))
    b = cl.get("/api/attr/assign", params={"annotator": "carol", "n": 5}).json()
    got = next((m for m in b["images"] if m["image_id"]=="wv_000001"), None)
    check("attr batch includes image", got is not None, json.dumps(b)[:200])
    if got:
        check("attr context preloads device_type=manual", got.get("context", {}).get("device_type") == "manual", json.dumps(got.get("context")))
        check("attr context has occupied", got.get("context", {}).get("occupied") in ("yes", "no"), json.dumps(got.get("context")))
    # carol changes device_type via correction; reload history
    r = cl.post("/api/events", json={"events": [
        E("carol", "device_type", "powered", kind="correction", previous="manual", previous_by="alice",
          ts="2026-11-09T20:00:00.000Z")]}).json()
    check("correction accepted", (r.get("accepted") and not r.get("rejected")), json.dumps(r))
    h = cl.get("/api/images/wv_000001/history").json()
    check("reload: device_type is powered", h["status"]["fields"]["device_type"][0] == "powered", json.dumps(h["status"]["fields"].get("device_type")))
    html = cl.get("/").text
    check("UI help no longer offers weather/ramp as current labels in the Attributes shortcut title",
          "wet ground" not in html and "Device use case" in html)
    js = (ROOT / "app" / "app.js").read_text(encoding="utf-8")
    check("attr mode MODEF uses every live schema field (editable rows, not a read-only summary)",
          "S.mode === 'attr' ? S.schema.fields.slice()" in js and "v1.0 labels:" not in js)

    check("app.js has no 'for you only' latest-wins wording", "for you only" not in js and "wins globally" in js)
    check("app.js does not render a lighting= or read-only core-label summary line", "lighting=" not in js and "v1.0 labels:" not in js)
    sw = (ROOT / "app" / "sw.js").read_text(encoding="utf-8")
    check("service worker is network-first and not ignoreSearch",
          "network-first" in sw.lower() and "ignoreSearch" not in sw and "wv-shell-v10" in sw)
    check("index.html cache-busts app.js", "?v=8" in html or "?v=8" in (ROOT / "app" / "index.html").read_text(encoding="utf-8"))
    check("no live securement row in UI help or field list",
          "docking bracket / tiedown" not in html.lower() and "securement_hardware" not in [f["name"] for f in cl.get("/api/schema").json()["fields"]])
    sw = (ROOT / "app" / "sw.js").read_text(encoding="utf-8")

n = sum(ok for _, ok in results)
print(f"\n{n}/{len(results)} passed; temp {TMP}")
sys.exit(0 if n == len(results) else 1)
