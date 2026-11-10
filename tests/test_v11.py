#!/usr/bin/env python3
"""v1.1 tests on synthetic data in a temp dir (never the live DB, events, images or derived/).

  python tests/test_v11.py            # core + size + server (FastAPI TestClient) + manifest/trainer dry run
  python tests/test_v11.py --import   # also scripts/import_contributed.py on synthetic images (needs cv2 + YuNet model)

Checks: v1.0 derived files are byte-identical with and without v1.1 events; v1.1 consensus/conflict statuses; the
attribute pool serves only clean v1.0-labelled images (+ active contributed, never pending ones); v1.0 labels/status
untouched by the attribute pass; rough size estimates are flagged unvalidated and never fill measured columns;
build_manifest_v11 keeps frozen v1.0 splits; train_multitask exits cleanly without enough labels."""
import csv, filecmp, io, json, os, shutil, subprocess, sys, tempfile, uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="wv_v11_"))
sys.path.insert(0, str(ROOT / "server")); sys.path.insert(0, str(ROOT / "scripts"))
results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond))); print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail else ""), flush=True)

import wvcore
V10 = [f for f in wvcore.SCHEMA["fields"] if not f.get("since")]
V11 = [f for f in wvcore.SCHEMA["fields"] if f.get("since") == "1.1"]
N = [0]
def ev(img, ann, field, value, kind="label"):
    N[0] += 1
    return {"uuid": str(uuid.uuid4()), "image_id": img, "annotator": ann, "field": field, "value": value, "kind": kind,
            "ts": f"2026-11-04T10:{N[0] // 60:02d}:{N[0] % 60:02d}.000Z", "device_id": "test-" + ann}
def v10_done(img, ann, dev="manual", bbox=None):
    out = [ev(img, ann, f["name"], dev if f["name"] == "device_type" else f["options"][0]) for f in V10]
    out += [ev(img, ann, "exclude", ""), ev(img, ann, "bbox", bbox or [[0.2, 0.1, 0.4, 0.6]]), ev(img, ann, "_status", "done")]
    return out
def v11_done(img, ann, over=None):
    return [ev(img, ann, f["name"], (over or {}).get(f["name"], f["options"][0])) for f in V11] + [ev(img, ann, "_status_v11", "done")]

# ---------------------------------------------------------------- synthetic dataset
CAND_COLS = "candidate_id,source,also_in,proposed_class,label_strength,title,creator,license,license_url,source_page_url,image_url,width,height,access_date,query_or_category,group_hint,needs_visual_screen,screen_reasons,cutout_hint,thumb_path,thumb_status,phash,flickr_id,src_key".split(",")
IDS = ["wv_000001", "wv_000002", "wv_000003", "wv_000004"]
with open(TMP / "candidates.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, CAND_COLS); w.writeheader()
    for k, i in enumerate(IDS): w.writerow({"candidate_id": i, "source": "commons", "creator": f"person{k}", "license": "CC-BY-4.0", "needs_visual_screen": "no"})
import import_contributed as IC
with open(TMP / "contributed.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, IC.COLS); w.writeheader()
    w.writerow({"image_id": "wv_900001", "status": "active", "source": "contributed", "creator": "Test Person", "license": "CC0-1.0", "people": "none", "identifiable_people": "no"})
    w.writerow({"image_id": "wv_900002", "status": "pending_approval", "source": "contributed", "creator": "Test Person", "license": "CC0-1.0", "people": "none", "identifiable_people": "no"})
from PIL import Image
for d in ("images", "images_blurred"):
    (TMP / d).mkdir()
    for k, i in enumerate(IDS + ["wv_900001", "wv_900002"]):
        Image.new("RGB", (640, 480), (40 * k, 120, 200 - 30 * k)).save(TMP / d / f"{i}.jpg")

# ---------------------------------------------------------------- 1. core: v1.0 outputs unchanged by v1.1 events
cands = wvcore.load_candidates(TMP / "candidates.csv") | wvcore.load_contributed(TMP / "contributed.csv")
check("pending contributed rows are not candidates", "wv_900001" in cands and "wv_900002" not in cands)
E10 = v10_done("wv_000001", "alice") + v10_done("wv_000001", "bob") + v10_done("wv_000002", "alice") + v10_done("wv_000003", "alice", "powered") \
    + v10_done("wv_000003", "bob", "manual") + v10_done("wv_900001", "alice") + v10_done("wv_900001", "bob")
A, B = TMP / "derived_a", TMP / "derived_b"; A.mkdir(); B.mkdir()
wvcore.materialize(E10, cands, str(A))
check("no v1.1 files without v1.1 events", not any((A / f).exists() for f in ("labels_v11.csv", "kappa_v11.json")))
E11 = v11_done("wv_000001", "alice") + v11_done("wv_000001", "bob") + v11_done("wv_900001", "alice", {"control": "joystick"}) \
    + v11_done("wv_900001", "bob", {"control": "push_rims"}) + v11_done("wv_000002", "alice")
wvcore.materialize(E10 + E11, cands, str(B))
same = [f for f in ("labels.csv", "per_annotator_labels.csv", "adjudication_needed.csv", "kappa.json") if filecmp.cmp(A / f, B / f, shallow=False)]
check("v1.0 derived files byte-identical with v1.1 events", len(same) == 4, f"identical: {same}")
L11 = {r["image_id"]: r for r in csv.DictReader(open(B / "labels_v11.csv", encoding="utf-8"))}
check("v1.1 consensus -> final", L11["wv_000001"]["status_v11"] == "final" and L11["wv_000001"]["control"] == "joystick")
check("v1.1 disagreement -> needs_third_annotator", L11["wv_900001"]["status_v11"] == "needs_third_annotator" and L11["wv_900001"]["control"] == "")
check("v1.1 single annotator -> in_progress", L11["wv_000002"]["status_v11"] == "in_progress")
check("contributed source kept in labels_v11", L11["wv_900001"]["source"] == "contributed")
bad = [wvcore.validate(ev("wv_000001", "alice", "_status_v11", "maybe"))[1], wvcore.validate(ev("wv_000001", "alice", "control", "hail"))[1],
       wvcore.validate(dict(ev("wv_000001", "alice", "control", "joystick"), source="auto_attr", model_tag="x_attr-v1"))[1]]
check("validation: bad _status_v11 / bad option rejected, auto_attr provenance accepted", bad[0] and bad[1] and bad[2] is None, str(bad))
retired_why = wvcore.validate(ev("wv_000001", "alice", "weather", "rain"))[1]
check("validation: new write of retired weather rejected", bool(retired_why), str(retired_why))
hist, hist_why = wvcore.validate(ev("wv_000001", "alice", "weather", "rain"), allow_retired=True)
check("validation: historical weather accepted with allow_retired", hist is not None, hist_why)

# ---------------------------------------------------------------- 2. size estimators
from size_estimators import ChairContext, best_estimate, proxies
e_front = best_estimate(ChairContext("x", [0.2, 0.1, 0.4, 0.6], 640, 480, "manual", "front", "no", "none"))
e_occ = best_estimate(ChairContext("x", [0.2, 0.1, 0.4, 0.6], 640, 480, "manual", "front", "yes", "none"))
e_amb = best_estimate(ChairContext("x", [0.2, 0.1, 0.4, 0.6], 640, 480, "ambiguous", "front", "no", "none"))
e_tr = best_estimate(ChairContext("x", [0.0, 0.1, 0.4, 0.6], 640, 480, "manual", "front", "no", "none"))
check("rough estimate is labelled rough_estimate + unvalidated", e_front.measurement_source == "rough_estimate" and "unvalidated" in e_front.flags)
check("estimate ranges contain the point value", e_front.width_low <= e_front.width_cm <= e_front.width_high and e_front.height_low <= e_front.height_cm <= e_front.height_high)
check("occupied box widens the width range", (e_occ.width_high - e_occ.width_low) > (e_front.width_high - e_front.width_low))
check("no estimate for ambiguous devices", e_amb is None)
check("truncated box -> prior only", "width=prior" in e_tr.basis)
check("proxies are non-metric", proxies(ChairContext("x", [0.2, 0.1, 0.4, 0.6], 640, 480))["non_metric"] == 1)

# ---------------------------------------------------------------- 3. server (scratch env only)
env = {"WV_DB": str(TMP / "scratch.db"), "WV_CANDIDATES": str(TMP / "candidates.csv"), "WV_CONTRIBUTED": str(TMP / "contributed.csv"),
       "WV_IMAGE_DIR": str(TMP / "images"), "WV_BLUR_DIR": str(TMP / "images_blurred"), "WV_EXPORT_DIR": str(TMP / "exports"),
       "WV_AUTO_DIR": str(TMP / "auto"), "WV_BLOCKLIST_DIR": str(TMP / "block"), "WV_IMAGE_PROXY": "0", "WV_ACCESS_CODE": "",
       "WV_BLUR_MANIFESTS": str(TMP / "blur.jsonl"), "WV_CACHE_DIR": str(TMP / "cache")}
(TMP / "auto").mkdir(); (TMP / "block").mkdir(); (TMP / "blur.jsonl").write_text("")
os.environ.pop("WV_HF_DATASET", None); os.environ.update(env)
try:
    from fastapi.testclient import TestClient
    sys.path.insert(0, str(ROOT / "server")); import main
    with TestClient(main.app) as cl:
        h = cl.get("/api/health").json()
        check("server: health lists attr_v11 + 1 active contributed", "attr_v11" in h["features"] and h["contributed"] == 1, str(h.get("contributed")))
        r = cl.post("/api/events", json={"events": E10}).json()
        check("server: v1.0 events accepted", len(r.get("accepted", [])) == len(E10) if isinstance(r.get("accepted"), list) else r.get("accepted") == len(E10), str({k: (v if not isinstance(v, list) else len(v)) for k, v in r.items()}))
        lab_before = cl.get("/export/labels.csv").content
        check("server: no labels_v11 export before v1.1 labels", cl.get("/export/labels_v11.csv").status_code == 404)
        b = cl.get("/api/attr/assign", params={"annotator": "carol", "n": 10}).json()
        got = [m["image_id"] for m in b["images"]]
        check("server: attr pool = clean v1.0-labelled images (+active contributed)", set(got) <= {"wv_000001", "wv_000002", "wv_000003", "wv_900001"}
              and "wv_000004" not in got and "wv_900002" not in got and "wv_900001" in got, str(got))
        ctx = next(m for m in b["images"] if m["image_id"] == "wv_000001").get("context", {})
        check("server: v1.0 labels shown as context", ctx.get("device_type") == "manual", str(ctx))
        r = cl.post("/api/events", json={"events": v11_done("wv_000001", "carol") + v11_done("wv_000001", "dave")}).json()
        lab_after = cl.get("/export/labels.csv").content
        check("server: v1.0 labels.csv unchanged by the attribute pass", lab_before == lab_after)
        l11 = cl.get("/export/labels_v11.csv")
        check("server: labels_v11.csv exported", l11.status_code == 200 and "wv_000001,final" in l11.text)
        check("server: stats include kappa_v11", "kappa_v11" in cl.get("/api/stats").json())
except ImportError as e:
    check("server tests skipped (fastapi/httpx missing)", True, str(e))

# ---------------------------------------------------------------- 4. manifest v1.1 + trainer guard
MAN = TMP / "manifest_v2.csv"
mcols = ["image_id", "split", "train_eligible", "exclusion_reason", "device_type", "setting", "occupied", "occlusion", "propulsion", "viewpoint",
         "lighting", "license", "creator", "leak_group", "path", "phash_blurred"]
with open(MAN, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, mcols); w.writeheader()
    for k, (i, s) in enumerate(zip(IDS[:3], ("train", "val", "test"))):
        w.writerow({"image_id": i, "split": s, "train_eligible": 1, "device_type": "manual", "creator": f"person{k}", "leak_group": f"g{k:04d}",
                    "path": f"images_blurred/{i}.jpg", "phash_blurred": format((0x5a5a5a5a5a5a5a5a >> k) & (2**63 - 1), "016x")})
ev_dir = TMP / "events"; ev_dir.mkdir()
by = {}
for e in E10 + E11: by.setdefault(e["annotator"], []).append(e)
for a_, es in by.items(): (ev_dir / f"{a_}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in es))
py = sys.executable
mo = TMP / "out" / "manifest_v11.csv"
r = subprocess.run([py, str(ROOT / "scripts" / "build_manifest_v11.py"), "--db", str(TMP / "none.db"), "--events", str(ev_dir), "--manifest-v10", str(MAN),
                    "--contributed", str(TMP / "contributed.csv"), "--images", str(TMP / "images_blurred"), "--size", str(TMP / "nosize.csv"),
                    "--measurements", str(TMP / "nomeas.csv"), "--out", str(mo)], capture_output=True, text=True)
check("build_manifest_v11 runs", r.returncode == 0, (r.stderr or "")[-300:])
if r.returncode == 0:
    R = {x["image_id"]: x for x in csv.DictReader(open(mo, encoding="utf-8"))}
    check("v1.0 splits frozen in v1.1 manifest", all(R[i]["split"] == s for i, s in zip(IDS[:3], ("train", "val", "test"))))
    check("v1.1-final image is v11_eligible", R["wv_000001"]["v11_eligible"] == "1" and R["wv_000001"]["control"] != "")
    check("contributed image present with its own leak group, not eligible while v1.1 disagrees", R["wv_900001"]["leak_group"].startswith("c")
          and R["wv_900001"]["v11_eligible"] == "0", R["wv_900001"]["v11_exclusion_reason"])
    check("pending contributed image absent", "wv_900002" not in R)
    check("no measured dims invented", all(not x.get("height_cm") for x in R.values()))
    t = subprocess.run([py, str(ROOT / "scripts" / "train_multitask.py"), "--manifest", str(mo), "--run", "v11_test_" + TMP.name, "--dry-run"], capture_output=True, text=True)
    check("train_multitask: exits 0 and trains nothing without enough labels", t.returncode == 0 and "nothing trained" in t.stdout, (t.stdout + t.stderr)[-300:])
    shutil.rmtree(ROOT / "var" / "runs" / ("v11_test_" + TMP.name), ignore_errors=True)

# ---------------------------------------------------------------- 5. optional: real importer on synthetic images
if "--import" in sys.argv:
    root = TMP / "root"; (root / "data").mkdir(parents=True); (root / "images").mkdir(); (root / "images_blurred").mkdir()
    shutil.copy(TMP / "candidates.csv", root / "data" / "candidates.csv")
    src = TMP / "src"; src.mkdir()
    import numpy as np
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (900, 1200, 3), dtype=np.uint8)).save(src / "a.jpg")
    Image.fromarray(np.tile(np.linspace(0, 255, 1000, dtype=np.uint8), (800, 1)).astype(np.uint8)).convert("RGB").save(src / "b.jpg")
    envi = {k: v for k, v in os.environ.items() if k not in ("WV_BLUR_DIR", "WV_CONTRIBUTED", "WV_CANDIDATES")}
    base = [py, str(ROOT / "scripts" / "import_contributed.py"), "--root", str(root), "--creator", "Test Person", "--license", "CC0-1.0", "--people", "none",
            "--attest-own-photo", "--attest-no-children"]
    r0 = subprocess.run(base + [str(src), "--dry-run"], capture_output=True, text=True, env=envi)
    check("import: --dry-run writes nothing", r0.returncode == 0 and not (root / "data" / "contributed.csv").exists() and not any((root / "images").iterdir()), (r0.stdout + r0.stderr)[-300:])
    r1 = subprocess.run(base + [str(src)], capture_output=True, text=True, env=envi)
    rows = list(csv.DictReader(open(root / "data" / "contributed.csv", encoding="utf-8"))) if (root / "data" / "contributed.csv").exists() else []
    check("import: 2 images, pending_approval without approval", r1.returncode == 0 and len(rows) == 2 and all(x["status"] == "pending_approval" for x in rows), (r1.stdout + r1.stderr)[-400:])
    check("import: ids in contributed range", all(int(x["image_id"][3:]) >= wvcore.CONTRIB_ID_MIN for x in rows))
    ok_files = all((root / d / f"{x['image_id']}.jpg").exists() for x in rows for d in ("images", "images_blurred"))
    sizes = [Image.open(root / "images_blurred" / f"{x['image_id']}.jpg").size for x in rows] if ok_files else []
    check("import: files written, long side <= 640", ok_files and all(max(s) <= 640 for s in sizes), str(sizes))
    exif = [bool(Image.open(root / "images" / f"{x['image_id']}.jpg").getexif()) for x in rows] if ok_files else [True]
    check("import: no EXIF in stored copies", not any(exif))
    Image.open(src / "a.jpg").transpose(Image.FLIP_LEFT_RIGHT).save(TMP / "a_mirror.jpg")
    r2 = subprocess.run(base + [str(src / "a.jpg"), str(TMP / "a_mirror.jpg")], capture_output=True, text=True, env=envi)
    rows2 = list(csv.DictReader(open(root / "data" / "contributed.csv", encoding="utf-8")))
    check("import: exact + mirrored duplicates rejected", len(rows2) == 2, (r2.stdout + r2.stderr)[-300:])
    r3 = subprocess.run([py, str(ROOT / "scripts" / "import_contributed.py"), "--root", str(root), "--approve-pending", "--approved-by", "tester", "--approval-ref", "test"], capture_output=True, text=True, env=envi)
    rows3 = list(csv.DictReader(open(root / "data" / "contributed.csv", encoding="utf-8")))
    check("import: --approve-pending activates", all(x["status"] == "active" for x in rows3), r3.stdout[-200:])
    r4 = subprocess.run(base + [str(src / "b.jpg"), "--height-cm", "300", "--measurement-source", "user_measured", "--chair-ref", "c1"], capture_output=True, text=True, env=envi)
    check("import: implausible measurement rejected", r4.returncode != 0)

print(f"\n{sum(ok for _, ok in results)}/{len(results)} passed; temp dir {TMP}")
sys.exit(0 if all(ok for _, ok in results) else 1)
