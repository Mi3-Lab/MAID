"""MAID v1.1 manifest: frozen v1.0 splits + v1.1 attribute labels + contributed images + size features.

  python scripts/build_manifest_v11.py --out var/v11_scratch/manifest_v11.csv          # test (scratch)
  python scripts/build_manifest_v11.py                                                  # -> data/manifest_v11.csv

Read-only over labels (DB opened mode=ro, annotations/events read, nothing appended). Never touches data/manifest_v2.csv,
data/splits_v3.csv, release/ or derived/.

Rules
  * v1.0 images keep their frozen manifest_v2 split / train_eligible / leak_group (the v1.0 release is not re-split).
  * v1.1 attribute columns are filled only where the v1.1 pass is FINAL (>=2 annotators agree, majority of 3, or
    adjudicated); otherwise empty and v11_status says why. 'unknown' is a real label value (kept, the trainer masks it).
  * Anyone's open exclude / face-blur flag keeps an image out of v1.1 training (v11_eligible=0) until removal review.
  * Contributed images (data/contributed.csv, status active only) need a FINAL v1.0 label in the clean set. Leak group =
    one group per creator (default) -- or per series/chair_ref with --contrib-group series. A new group whose creator
    already exists in v1.0 inherits that split; otherwise a deterministic hash split (70/15/15).
  * A contributed image whose pHash (incl. mirror) is within --phash-max of an image in ANOTHER split is excluded
    (near_dup_bridges_split): "when unsure, leave it out".
  * Size: proxies (non_metric) and rough estimates (est_*, measurement_source=rough_estimate) from
    scripts/size_estimate.py for single-box images only; measured dims (data/measurements.csv, real sources) separately.
"""
import argparse, csv, hashlib, json, os, re, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server")); sys.path.insert(0, str(ROOT / "scripts"))
import wvcore
from size_estimate import load_events

DESC = ["setting", "occupied", "occlusion", "propulsion", "viewpoint", "lighting"]
V11 = list(wvcore.COMPARE_V11)
SIZE_COLS = ["box_w_norm", "box_h_norm", "aspect_hw_px", "box_w_reflects", "est_height_cm", "est_height_low_cm", "est_height_high_cm",
             "est_width_cm", "est_width_low_cm", "est_width_high_cm", "est_length_cm", "est_length_low_cm", "est_length_high_cm",
             "est_basis", "est_flags"]
MEAS_COLS = ["height_cm", "width_cm", "length_cm", "measurement_source"]
COLS = (["image_id", "source", "split", "split_origin", "train_eligible_v10", "v11_eligible", "v11_exclusion_reason", "device_type", *DESC,
         "v11_status", "v11_annotators", *V11, "license", "creator", "leak_group", "path", "phash_blurred"] + SIZE_COLS + MEAS_COLS)
FRACS = (("train", 70), ("val", 85), ("test", 100))
REAL_SOURCES = ("user_measured", "manufacturer_spec", "depth_sensor", "reference_object")

def norm_name(s): return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()
def hash_split(key):
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 100
    return next(s for s, t in FRACS if h < t)

def phash_hex(path):
    """Same 63-bit pHash as build_manifest.phash_bits; returns (hex, mirrored hex)."""
    from PIL import Image, ImageOps
    sys.path.insert(0, str(ROOT / "scripts")); import build_manifest as BM
    with Image.open(path) as im:
        im = im.convert("RGB")
        to = lambda bits: format(int("".join("1" if b else "0" for b in bits), 2), "016x")
        return to(BM.phash_bits(im)), to(BM.phash_bits(ImageOps.mirror(im)))

def ham(a, b): return bin(int(a, 16) ^ int(b, 16)).count("1")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.environ.get("WV_DB", str(ROOT / "var" / "wheelview.db")))
    ap.add_argument("--events", default=str(ROOT / "annotations" / "events"))
    ap.add_argument("--manifest-v10", default=str(ROOT / "data" / "manifest_v2.csv"))
    ap.add_argument("--contributed", default=os.environ.get("WV_CONTRIBUTED", str(ROOT / "data" / "contributed.csv")))
    ap.add_argument("--images", default=os.environ.get("WV_BLUR_DIR", str(ROOT / "images_blurred")))
    ap.add_argument("--size", default=str(ROOT / "var" / "v11" / "size_estimates.csv"), help="scripts/size_estimate.py output (optional)")
    ap.add_argument("--measurements", default=str(ROOT / "data" / "measurements.csv"))
    ap.add_argument("--contrib-group", choices=["creator", "series"], default="creator")
    ap.add_argument("--phash-max", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "data" / "manifest_v11.csv"))
    a = ap.parse_args()
    out = Path(a.out).resolve()
    for frozen in (ROOT / "data" / "manifest_v2.csv", ROOT / "data" / "splits_v3.csv"):
        if out == frozen.resolve(): sys.exit(f"refusing to overwrite frozen {frozen.name}")
    if (ROOT / "derived").resolve() in out.parents or (ROOT / "release").resolve() in out.parents: sys.exit("refusing to write into derived/ or release/")
    events = load_events(a.db, a.events); per, adjv = wvcore.fold(events)
    hold = wvcore.load_safety_hold(str(ROOT / "data")); hist = wvcore.exclude_history(events)
    M = list(csv.DictReader(open(a.manifest_v10, encoding="utf-8")))
    C = wvcore.load_contributed(a.contributed)
    size = {}
    if a.size and os.path.exists(a.size):
        for r in csv.DictReader(open(a.size, encoding="utf-8")):
            if r["n_boxes"] == "1": size[r["image_id"]] = r
    meas = {}
    if os.path.exists(a.measurements):
        for r in csv.DictReader(open(a.measurements, encoding="utf-8")):
            if r.get("measurement_source") in REAL_SOURCES and r.get("image_id"): meas[r["image_id"]] = r
    def v11_cols(img):
        s = wvcore.image_status_v11(img, per, adjv)
        d = {"v11_status": s["status"], "v11_annotators": ";".join(s["done"])}
        for f in V11: d[f] = s["fields"][f][0] if s["status"] == "final" else ""
        return d
    def flagged(img):
        return any(fv.get("exclude") not in (None, "", "none") or fv.get("face_blur_missed") is True for fv in per.get(img, {}).values())
    def extras(img):
        d = {}
        s = size.get(img)
        if s:
            for k in SIZE_COLS:
                d[k] = s.get({"est_basis": "basis", "est_flags": "flags"}.get(k, k), "")
        m = meas.get(img)
        if m: d.update({k: m.get(k, "") for k in MEAS_COLS})
        return d
    rows, creator_split = [], {}
    for r in M:
        i = r["image_id"]; el = r["train_eligible"] == "1"
        if el and norm_name(r.get("creator")): creator_split.setdefault(norm_name(r["creator"]), set()).add(r["split"])
        d = {"image_id": i, "source": "web_v1.0", "split": r["split"], "split_origin": "frozen_v1.0", "train_eligible_v10": int(el),
             "device_type": r["device_type"], **{k: r.get(k, "") for k in DESC}, "license": r.get("license", ""), "creator": r.get("creator", ""),
             "leak_group": r["leak_group"], "path": r["path"], "phash_blurred": r.get("phash_blurred", "")}
        d.update(v11_cols(i)); d.update(extras(i))
        why = "" if el else "excluded_in_v1.0"
        if not why and flagged(i): why = "open_flag_pending_review"   # same rule as the server's attribute pool (main.v11_flagged)
        if not why and d["v11_status"] != "final": why = "v11_not_final"
        d["v11_eligible"] = int(not why); d["v11_exclusion_reason"] = why
        rows.append(d)
    v10_hash = [(r["phash_blurred"], r["split"], r["image_id"]) for r in rows if r["train_eligible_v10"] and r["phash_blurred"]]
    groups, n_bridge = {}, 0
    for i, c in sorted(C.items()):
        st = wvcore.image_status(i, per, adjv); ok, cwhy = wvcore.clean_decision(i, per, adjv, history=hist, hold=hold)
        dev = st["fields"]["device_type"][0] if st["status"] == "final" else ""
        key = "creator:" + norm_name(c.get("creator")) if a.contrib_group == "creator" else "series:" + norm_name(c.get("series") or c.get("chair_ref") or c.get("creator"))
        if key not in groups: groups[key] = f"c{len(groups) + 1:04d}"
        inherit = creator_split.get(norm_name(c.get("creator")), set())
        if len(inherit) == 1: split, origin = next(iter(inherit)), "inherited_creator_split"
        elif len(inherit) > 1: split, origin = "excluded", "creator_spans_v1.0_splits"
        else: split, origin = hash_split(key), f"hash_{a.contrib_group}"
        d = {"image_id": i, "source": "contributed", "split": split, "split_origin": origin, "train_eligible_v10": 0, "device_type": dev,
             **{k: (st["fields"][k][0] if st["status"] == "final" and k in st["fields"] else "") for k in DESC},
             "license": c.get("license", ""), "creator": c.get("creator", ""), "leak_group": groups[key], "path": f"images_blurred/{i}.jpg"}
        d.update(v11_cols(i)); d.update(extras(i))
        why = ""
        if split == "excluded": why = origin
        elif not ok: why = "v10_" + cwhy
        elif st["status"] != "final": why = "v10_" + st["status"]
        elif dev not in wvcore.DEVICE_CLASSES: why = "device_unclear"
        elif flagged(i): why = "open_flag_pending_review"
        p = Path(a.images) / f"{i}.jpg"
        if p.exists():
            h, hm = phash_hex(p); d["phash_blurred"] = h
            near = [(vi, vs) for vh, vs, vi in v10_hash if vs != split and min(ham(h, vh), ham(hm, vh)) <= a.phash_max]
            if near and not why: why = "near_dup_bridges_split"; n_bridge += 1
        elif not why: why = "image_missing_locally"
        if not why and d["v11_status"] != "final": why = "v11_not_final"
        d["v11_eligible"] = int(not why); d["v11_exclusion_reason"] = why
        rows.append(d)
    # contributed vs contributed across splits (only possible with --contrib-group series)
    ch = [(r["phash_blurred"], r["split"], r["image_id"]) for r in rows if r["source"] == "contributed" and r["phash_blurred"] and r["v11_eligible"]]
    bad = {y[2] for k, x in enumerate(ch) for y in ch[k + 1:] if x[1] != y[1] and ham(x[0], y[0]) <= a.phash_max}
    for r in rows:
        if r["image_id"] in bad: r["v11_eligible"] = 0; r["v11_exclusion_reason"] = "near_dup_bridges_split"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, COLS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    rep = {"rows": len(rows), "v10_rows": len(M), "contributed_rows": len(C), "v11_eligible": sum(r["v11_eligible"] for r in rows),
           "v11_eligible_by_split": dict(Counter(r["split"] for r in rows if r["v11_eligible"])),
           "exclusion_reasons": dict(Counter(r["v11_exclusion_reason"] for r in rows if not r["v11_eligible"])),
           "contrib_groups": len(groups), "near_dup_bridges": n_bridge + len(bad), "size_rows_joined": sum(1 for r in rows if r.get("est_height_cm")),
           "measured_rows_joined": sum(1 for r in rows if r.get("measurement_source"))}
    out.with_suffix(".report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))

if __name__ == "__main__": main()
