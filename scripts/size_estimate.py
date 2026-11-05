"""v1.1 size features: non-metric box proxies + ROUGH metric estimates (unvalidated) + validation against real measurements.

  python scripts/size_estimate.py                                   # read-only: DB (ro) + annotations/events + derived/labels.csv
  python scripts/size_estimate.py --db var/v11_scratch/wheelview.db --out var/v11_scratch/size_estimates.csv

Output (one row per human-drawn device box; never written into derived/ or data/measurements.csv):
  var/v11/size_estimates.csv          proxies (non_metric=1) + est_* columns with low/high, measurement_source=rough_estimate
  var/v11/size_validation.json        error vs data/measurements.csv IF any real measurement matches; else status "unvalidated"

Boxes: human bboxes only (latest per annotator; the first annotator who finished the image, alphabetically). Machine
OWLv2 boxes are used only with --machine-boxes and are flagged box_source=machine:<tag>. Device type / viewpoint /
occupied / occlusion come from the resolved v1.0 labels (derived/labels.csv). Read-only: nothing here writes events.
"""
import argparse, csv, json, os, sqlite3, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server")); sys.path.insert(0, str(ROOT / "scripts"))
import wvcore
from size_estimators import ChairContext, proxies, best_estimate, VERSION, PRIORS

REAL_SOURCES = ("user_measured", "manufacturer_spec", "depth_sensor", "reference_object")
COLS = ["image_id", "box_idx", "n_boxes", "box_source", "device_type", "viewpoint", "occupied", "occlusion", "img_w", "img_h",
        "box_w_norm", "box_h_norm", "box_area_norm", "aspect_hw_px", "box_w_reflects", "truncated", "non_metric",
        "est_height_cm", "est_height_low_cm", "est_height_high_cm", "est_width_cm", "est_width_low_cm", "est_width_high_cm",
        "est_length_cm", "est_length_low_cm", "est_length_high_cm", "measurement_source", "method", "basis", "flags",
        "estimator_version", "validated"]

def load_events(db, evdir):
    evs, seen = [], set()
    def add(raw):
        e, _ = wvcore.validate(raw)
        if e and e["uuid"] not in seen: seen.add(e["uuid"]); evs.append(e)
    if db and os.path.exists(db):
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True); c.row_factory = sqlite3.Row
        for r in c.execute("SELECT * FROM events ORDER BY seq"):
            raw = {k: r[k] for k in ("uuid", "image_id", "annotator", "field", "ts", "device_id", "kind")}
            raw["value"] = json.loads(r["value"]) if r["value"] is not None else None
            if r["prov"]: raw.update(json.loads(r["prov"]))
            add(raw)
        c.close()
    if evdir and os.path.isdir(evdir):
        for fn in sorted(Path(evdir).glob("*.jsonl")):
            for line in open(fn, encoding="utf-8"):
                if line.strip():
                    try: add(json.loads(line))
                    except json.JSONDecodeError: pass
    return evs

def human_boxes(per, img):
    for ann in sorted(per.get(img, {})):
        lab = per[img][ann]
        if lab.get("_status") == "done" and lab.get("bbox"): return lab["bbox"], f"human:{ann}"
    return [], ""

def image_dims(paths, imgdir):
    dims = {}
    for p in paths:
        if p and os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    if r.get("width") and r.get("height"): dims[r["image_id"]] = (r["width"], r["height"])
    def get(img):
        if img in dims: return dims[img]
        f = Path(imgdir) / f"{img}.jpg"
        if f.exists():
            from PIL import Image
            with Image.open(f) as im: dims[img] = im.size
            return dims[img]
        return None
    return get

def validate(rows, meas_path):
    real = {}
    if meas_path and os.path.exists(meas_path):
        for m in csv.DictReader(open(meas_path, encoding="utf-8")):
            if m.get("measurement_source") in REAL_SOURCES and m.get("image_id"): real[m["image_id"]] = m
    res = {"estimator_version": VERSION, "n_measurements_real": len(real), "matched_single_box_images": 0, "dims": {}}
    single = {r["image_id"]: r for r in rows if r["n_boxes"] == 1}
    pairs = {d: [] for d in ("height", "width", "length")}
    for img, m in real.items():
        r = single.get(img)
        if not r: continue
        res["matched_single_box_images"] += 1
        for d in pairs:
            gt, est = m.get(f"{d}_cm"), r.get(f"est_{d}_cm")
            if gt not in (None, "") and est not in (None, ""):
                pairs[d].append((float(gt), float(est), float(r[f"est_{d}_low_cm"]), float(r[f"est_{d}_high_cm"]), r["basis"]))
    if not any(pairs.values()):
        res["status"] = "unvalidated"
        res["note"] = "no real measurements (data/measurements.csv) match a single-box estimated image; no error can be reported"
        return res
    res["status"] = "validated_on_small_sample"
    for d, p in pairs.items():
        if not p: continue
        err = [abs(e - g) for g, e, *_ in p]
        res["dims"][d] = {"n": len(p), "mae_cm": round(sum(err) / len(err), 1), "max_abs_err_cm": round(max(err), 1),
                          "range_coverage": round(sum(lo <= g <= hi for g, _, lo, hi, _ in p) / len(p), 3),
                          "note": "n is tiny; treat as a sanity check, not an accuracy figure" if len(p) < 30 else ""}
    return res

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.environ.get("WV_DB", str(ROOT / "var" / "wheelview.db")))
    ap.add_argument("--events", default=str(ROOT / "annotations" / "events"))
    ap.add_argument("--labels", default=str(ROOT / "derived" / "labels.csv"))
    ap.add_argument("--blur", nargs="*", default=[str(ROOT / "data" / "blur_manifest.jsonl"), str(ROOT / "var" / "blur_manifest_local.jsonl")])
    ap.add_argument("--images", default=os.environ.get("WV_BLUR_DIR", str(ROOT / "images_blurred")))
    ap.add_argument("--measurements", default=str(ROOT / "data" / "measurements.csv"))
    ap.add_argument("--machine-boxes", default=None, help="optional annotations/auto/<tag>.jsonl: use OWLv2 boxes where no human box exists (flagged)")
    ap.add_argument("--box-covers", default="auto", choices=["auto", "device", "device_plus_occupant"])
    ap.add_argument("--out", default=str(ROOT / "var" / "v11" / "size_estimates.csv"))
    ap.add_argument("--validation-out", default=None)
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.resolve().parent == (ROOT / "derived").resolve(): sys.exit("refusing to write into derived/ (CI-rebuilt, v1.0 frozen)")
    per, _ = wvcore.fold(load_events(a.db, a.events))
    labels = {r["image_id"]: r for r in csv.DictReader(open(a.labels, encoding="utf-8"))} if os.path.exists(a.labels) else {}
    mach = {}
    if a.machine_boxes and os.path.exists(a.machine_boxes):
        for line in open(a.machine_boxes, encoding="utf-8"):
            r = json.loads(line)
            if r.get("boxes"): mach[r["image_id"]] = ([b["box"] for b in r["boxes"]], "machine:" + r.get("model_tag", "?"))
    dims = image_dims(a.blur, a.images)
    rows, skipped = [], {"no_label": 0, "no_prior_device_type": 0, "no_box": 0, "no_dims": 0, "excluded": 0}
    for img in sorted(set(per) | set(labels)):
        L = labels.get(img)
        if not L or L.get("status") != "final": skipped["no_label"] += 1; continue
        if L.get("exclude") not in ("", "none", None): skipped["excluded"] += 1; continue
        if L.get("device_type") not in PRIORS: skipped["no_prior_device_type"] += 1; continue
        boxes, src = human_boxes(per, img)
        if not boxes and img in mach: boxes, src = mach[img]
        if not boxes: skipped["no_box"] += 1; continue
        d = dims(img)
        if not d: skipped["no_dims"] += 1; continue
        for i, b in enumerate(boxes):
            ctx = ChairContext(img, b, d[0], d[1], L["device_type"], L.get("viewpoint", ""), L.get("occupied", ""),
                               L.get("occlusion", ""), src, a.box_covers)
            r = {"image_id": img, "box_idx": i, "n_boxes": len(boxes), "box_source": src, "device_type": ctx.device_type,
                 "viewpoint": ctx.viewpoint, "occupied": ctx.occupied, "occlusion": ctx.occlusion, "img_w": d[0], "img_h": d[1]}
            r.update(proxies(ctx))
            e = best_estimate(ctx)
            if e:
                for k in ("height", "width", "length"):
                    r[f"est_{k}_cm"], r[f"est_{k}_low_cm"], r[f"est_{k}_high_cm"] = getattr(e, f"{k}_cm"), getattr(e, f"{k}_low"), getattr(e, f"{k}_high")
                r.update(measurement_source=e.measurement_source, method=e.method, basis=e.basis, flags=";".join(e.flags),
                         estimator_version=VERSION, validated="no")
            rows.append(r)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    v = validate(rows, a.measurements)
    vp = Path(a.validation_out) if a.validation_out else out.with_name("size_validation.json")
    v["skipped"] = skipped; v["n_rows"] = len(rows)
    vp.write_text(json.dumps(v, indent=1), encoding="utf-8")
    print(f"{len(rows)} box rows -> {out}; skipped {skipped}; validation: {v['status']} -> {vp}")

if __name__ == "__main__": main()
