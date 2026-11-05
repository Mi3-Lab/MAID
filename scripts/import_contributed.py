#!/usr/bin/env python3
"""Import your OWN open-licensed wheelchair photos into MAID (v1.1) so they can be labelled like any other image.

Everything stays on this machine: images go to images/ (resized original, no EXIF) and images_blurred/ (face-blurred copy,
the one annotators see), metadata to data/contributed.csv, the blur record to var/blur_manifest_local.jsonl, free-text
consent notes to var/contrib/consent_notes.jsonl. Nothing is uploaded; images/ and images_blurred/ are git-ignored.

Per image:
  1. EXIF orientation applied, then ALL metadata dropped (re-encoded with OpenCV; no EXIF/GPS/XMP/ICC survives).
  2. Resized so the long side is <= 640 px (same as the web images).
  3. Faces detected with the YuNet detector used for v1.0 (scripts/auto_blur.py, blur-v3) and blurred in the copy served
     to annotators; the record goes to the LOCAL blur manifest.
  4. Near-duplicate check (64-bit DCT pHash, also vs. the mirrored image, Hamming <= --phash-max) against every image in
     images/, images_blurred/ and earlier contributions in this batch. Duplicates are skipped and reported.
  5. New id in the contributed range (wv_900001, wv_900002, ...), source=contributed, queued for normal v1.0 labelling
     (and the v1.1 Attributes pass afterwards). Only the SHA-256 of the original file is stored, never its file name/path.

Lab rule: "Do not take any photos or videos yourselves." Importing your own photos needs an explicit, recorded approval.
Without --approved-by/--approval-ref the images are imported with status=pending_approval: stored locally, NOT served
for annotation and NOT in any split until you re-run with --approve-pending once approval exists.

Example (PowerShell, from the repo root):
  .venv\\Scripts\\python.exe scripts\\import_contributed.py C:\\Users\\kamoo\\Pictures\\chair1 `
      --creator "Samir Shah Ahmed" --license CC-BY-4.0 --people none `
      --attest-own-photo --attest-no-children --approved-by "Ross" --approval-ref "email 2026-11-05" `
      --height-cm <tape cm> --width-cm <tape cm> --measurement-source user_measured --measured-by samir --measured-on 2026-11-04 `
      --chair-ref chair-A --dry-run
"""
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server")); sys.path.insert(0, str(ROOT / "scripts"))
import wvcore

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".heic"}
COLS = ["image_id", "status", "source", "creator", "license", "license_url", "title", "people", "identifiable_people", "consent",
        "own_photo_attested", "no_children_attested", "approved_by", "approval_ref", "imported_at", "imported_by", "orig_sha256",
        "sha256", "width", "height", "n_faces_blurred", "chair_ref", "series", "notes"]
MCOLS = ["image_id", "chair_ref", "height_cm", "width_cm", "length_cm", "measurement_source", "measured_by", "measured_on",
         "configuration", "notes"]
LICENSE_URL = {"CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/", "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
               "CC-BY-SA-4.0": "https://creativecommons.org/licenses/by-sa/4.0/", "CC-BY-2.0": "https://creativecommons.org/licenses/by/2.0/",
               "CC-BY-3.0": "https://creativecommons.org/licenses/by/3.0/", "CC-BY-SA-2.0": "https://creativecommons.org/licenses/by-sa/2.0/",
               "CC-BY-SA-3.0": "https://creativecommons.org/licenses/by-sa/3.0/", "PDM": "https://creativecommons.org/publicdomain/mark/1.0/"}
PEOPLE = {"none": "no people in the photo",
          "not_identifiable": "people present but not identifiable (from behind / far / cropped)",
          "consented": "identifiable people who gave written consent for open release (keep the form; note where in --consent-note)"}

def phash_bits(im):
    """Identical to scripts/build_manifest.py::phash_bits (63-bit DCT pHash, DC dropped)."""
    import scipy.fft
    from PIL import Image
    g = np.asarray(im.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    d = scipy.fft.dctn(g, norm="ortho")[:8, :8].flatten()
    return d[1:] > np.median(d[1:])

def hashes(im):
    from PIL import ImageOps
    return phash_bits(im), phash_bits(ImageOps.mirror(im))

def load_rgb(path):
    from PIL import Image, ImageOps
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)          # honour camera orientation, then the EXIF is gone for good
    im = im.convert("RGB"); im.thumbnail((640, 640), Image.LANCZOS)
    return im

def read_csv(p):
    if not p.exists(): return []
    with open(p, encoding="utf-8", newline="") as fh: return list(csv.DictReader(fh))

def write_csv(p, cols, rows):
    p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    os.replace(tmp, p)

def next_id(used):
    n = wvcore.CONTRIB_ID_MIN
    nums = [int(u[3:]) for u in used if wvcore.ID_RE.match(u) and int(u[3:]) >= wvcore.CONTRIB_ID_MIN]
    if nums: n = max(nums) + 1
    if n > 999999: sys.exit("contributed id range exhausted")
    return f"wv_{n:06d}"

def existing_hashes(dirs, cache_path):
    """pHash of every existing image (cached by file size+mtime in var/contrib/phash_cache.json)."""
    from PIL import Image
    cache = {}
    if cache_path.exists():
        try: cache = json.load(open(cache_path, encoding="utf-8"))
        except Exception: cache = {}
    out, changed = {}, False
    for d in dirs:
        if not d.exists(): continue
        for p in d.glob("*.jpg"):
            st = p.stat(); key = f"{d.name}/{p.name}"; sig = f"{st.st_size}:{st.st_mtime_ns}"
            if cache.get(key, {}).get("sig") != sig:
                try:
                    with Image.open(p) as im: h = phash_bits(im.convert("RGB"))
                except Exception: continue
                cache[key] = {"sig": sig, "h": "".join("1" if b else "0" for b in h)}; changed = True
            out[key] = np.array([c == "1" for c in cache[key]["h"]])
    if changed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(cache, open(cache_path, "w", encoding="utf-8"))
    return out

def main():
    schema = wvcore.SCHEMA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="image files and/or folders")
    ap.add_argument("--creator", help="name for attribution (the photographer)")
    ap.add_argument("--license", choices=schema["contributed_licenses"])
    ap.add_argument("--people", choices=list(PEOPLE), help="; ".join(f"{k}: {v}" for k, v in PEOPLE.items()))
    ap.add_argument("--consent-note", default="", help="where the consent form is kept etc. (stored locally only, var/contrib/)")
    ap.add_argument("--attest-own-photo", action="store_true", help="I took this photo / hold the rights and release it under --license")
    ap.add_argument("--attest-no-children", action="store_true", help="no children appear in the photo (child images are excluded)")
    ap.add_argument("--approved-by", default="", help="who approved using own photos (lab rule: no own photos without approval)")
    ap.add_argument("--approval-ref", default="", help="reference for that approval (e.g. 'email 2026-11-05')")
    ap.add_argument("--approve-pending", action="store_true", help="activate earlier pending_approval imports (needs --approved-by/--approval-ref)")
    ap.add_argument("--title", default=""); ap.add_argument("--notes", default=""); ap.add_argument("--series", default="",
                    help="free label for a photo session (all contributed images share one leak group anyway)")
    ap.add_argument("--chair-ref", default="", help="your label for the physical chair (links photos of the same chair and its measurements)")
    g = ap.add_argument_group("measured dimensions (ONLY real measurements; leave empty otherwise)")
    g.add_argument("--height-cm", type=float); g.add_argument("--width-cm", type=float); g.add_argument("--length-cm", type=float)
    g.add_argument("--measurement-source", choices=[m for m in schema["measurement_sources"] if m != "none"])
    g.add_argument("--measured-by", default=""); g.add_argument("--measured-on", default="", help="YYYY-MM-DD")
    g.add_argument("--configuration", default="", help="e.g. 'unfolded, footrests on, no occupant'")
    ap.add_argument("--imported-by", default=os.environ.get("USERNAME") or os.environ.get("USER") or "")
    ap.add_argument("--phash-max", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="check everything, write nothing")
    ap.add_argument("--root", default=str(ROOT), help="dataset root (point at a scratch copy for tests)")
    a = ap.parse_args()
    R = Path(a.root)
    images_dir, blur_dir = R / "images", Path(os.environ.get("WV_BLUR_DIR", R / "images_blurred"))
    contrib_csv = Path(os.environ.get("WV_CONTRIBUTED", R / "data" / "contributed.csv"))
    meas_csv, local_blur = R / "data" / "measurements.csv", R / "var" / "blur_manifest_local.jsonl"
    notes_path, cache_path = R / "var" / "contrib" / "consent_notes.jsonl", R / "var" / "contrib" / "phash_cache.json"
    rows = read_csv(contrib_csv)
    approved = bool(a.approved_by.strip() and a.approval_ref.strip())

    if a.approve_pending:
        if not approved: sys.exit("--approve-pending needs --approved-by and --approval-ref")
        n = 0
        for r in rows:
            if r["status"] == "pending_approval": r.update(status="active", approved_by=a.approved_by, approval_ref=a.approval_ref); n += 1
        if not a.dry_run: write_csv(contrib_csv, COLS, rows)
        print(f"activated {n} pending image(s)" + (" (dry run)" if a.dry_run else "")); return

    miss = [k for k in ("creator", "license", "people") if not getattr(a, k)]
    if miss or not a.paths: sys.exit("required: paths, --" + ", --".join(miss or ["(paths)"]))
    if not a.attest_own_photo: sys.exit("--attest-own-photo is required (you hold the rights and release under --license)")
    if not a.attest_no_children: sys.exit("--attest-no-children is required (child images are excluded from MAID); leave such photos out")
    if a.people == "consented" and not a.consent_note.strip(): sys.exit("--people consented needs --consent-note (where the signed form is kept)")
    dims = {k: getattr(a, k) for k in ("height_cm", "width_cm", "length_cm") if getattr(a, k) is not None}
    if dims and not a.measurement_source: sys.exit("measured dimensions need --measurement-source")
    if a.measurement_source and not dims: sys.exit("--measurement-source given without any --*-cm value")
    for k, v in dims.items():
        if not 10 <= v <= 250: sys.exit(f"--{k.replace('_', '-')} {v} is outside 10..250 cm; check the value")
    if dims and not a.chair_ref: sys.exit("measured dimensions need --chair-ref (which physical chair they belong to)")

    files = []
    for p in map(Path, a.paths):
        if p.is_dir(): files += sorted(q for q in p.iterdir() if q.suffix.lower() in EXTS)
        elif p.is_file(): files.append(p)
        else: print(f"skip (not found): {p}", file=sys.stderr)
    if not files: sys.exit("no images found")

    import cv2, auto_blur
    used = set(wvcore.load_candidates(os.environ.get("WV_CANDIDATES", R / "data" / "candidates.csv"))) | {r["image_id"] for r in rows}
    seen_orig = {r["orig_sha256"]: r["image_id"] for r in rows}
    ex = existing_hashes([images_dir, blur_dir], cache_path)
    new_rows, new_meas, blur_recs, report = [], [], {}, []
    for f in files:
        raw = f.read_bytes(); osha = hashlib.sha256(raw).hexdigest()
        if osha in seen_orig: report.append((f.name, "skip", f"already imported as {seen_orig[osha]}")); continue
        try: im = load_rgb(f)
        except Exception as e: report.append((f.name, "skip", f"unreadable ({e.__class__.__name__}; HEIC needs pillow-heif)")); continue
        h, hm = hashes(im)
        near = [(int(min((h != v).sum(), (hm != v).sum())), k) for k, v in ex.items()]
        near = sorted(x for x in near if x[0] <= a.phash_max)
        if near: report.append((f.name, "skip", f"near-duplicate of {near[0][1]} (pHash distance {near[0][0]})")); continue
        iid = next_id(used); used.add(iid)
        bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR); H, W = bgr.shape[:2]
        px = auto_blur.detect(bgr); blurred = auto_blur.blur(bgr, px)
        if not a.dry_run:
            images_dir.mkdir(parents=True, exist_ok=True); blur_dir.mkdir(parents=True, exist_ok=True)
            src_sha1 = auto_blur.write_jpg(images_dir / f"{iid}.jpg", bgr)       # resized, metadata-free original (local only)
            out_sha1 = auto_blur.write_jpg(blur_dir / f"{iid}.jpg", blurred)
            out_sha256 = hashlib.sha256((blur_dir / f"{iid}.jpg").read_bytes()).hexdigest()
        else: src_sha1 = out_sha1 = out_sha256 = ""
        blur_recs[iid] = {"image_id": iid, "src_sha1": src_sha1, "out_sha1": out_sha1, "out_sha256": out_sha256, "width": W, "height": H,
                          "n_faces": len(px), "faces": [{"box": [round(b[0] / W, 4), round(b[1] / H, 4), round(b[2] / W, 4), round(b[3] / H, 4)],
                          "pad_box": [round(float(v), 4) for v in np.array(auto_blur.pad_box(b, W, H)) / [W, H, W, H]], "score": round(b[4], 3)} for b in px],
                          "detector": auto_blur.DETECTOR, "model_sha256": auto_blur.MODEL_SHA256, "score_threshold": auto_blur.SCORE_THR,
                          "score_threshold_large": auto_blur.SCORE_LARGE, "small_frac": auto_blur.SMALL_FRAC, "aspect": list(auto_blur.ASPECT),
                          "nms_threshold": auto_blur.NMS_THR, "scales": auto_blur.scales_for(H, W), "mirrored": True, "blur": auto_blur.BLUR,
                          "version": auto_blur.VERSION, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source": "contributed"}
        ex[f"new/{iid}"] = h
        new_rows.append({"image_id": iid, "status": "active" if approved else "pending_approval", "source": "contributed",
                         "creator": a.creator.strip(), "license": a.license, "license_url": LICENSE_URL.get(a.license, ""),
                         "title": a.title or f"Contributed photo {iid}", "people": a.people,
                         "identifiable_people": "yes" if a.people == "consented" else "no", "consent": a.people,
                         "own_photo_attested": "yes", "no_children_attested": "yes", "approved_by": a.approved_by.strip(),
                         "approval_ref": a.approval_ref.strip(), "imported_at": time.strftime("%Y-%m-%d"), "imported_by": a.imported_by,
                         "orig_sha256": osha, "sha256": out_sha256, "width": W, "height": H, "n_faces_blurred": len(px),
                         "chair_ref": a.chair_ref, "series": a.series, "notes": a.notes})
        if dims:
            new_meas.append({"image_id": iid, "chair_ref": a.chair_ref, **{k: f"{v:g}" for k, v in dims.items()},
                             "measurement_source": a.measurement_source, "measured_by": a.measured_by, "measured_on": a.measured_on,
                             "configuration": a.configuration, "notes": ""})
        report.append((f.name, "import", f"{iid}  {W}x{H}  faces blurred: {len(px)}"))
    for name, act, msg in report: print(f"{act:6s} {name}: {msg}")
    if a.dry_run: print(f"dry run: {len(new_rows)} would be imported; nothing written"); return
    if new_rows:
        local_blur.parent.mkdir(parents=True, exist_ok=True)
        auto_blur.flush(local_blur, True, {}, blur_recs)          # blur records first, then register the images
        if new_meas: write_csv(meas_csv, MCOLS, read_csv(meas_csv) + new_meas)
        write_csv(contrib_csv, COLS, rows + new_rows)
        if a.consent_note.strip():
            notes_path.parent.mkdir(parents=True, exist_ok=True)
            with open(notes_path, "a", encoding="utf-8") as fh:
                for r in new_rows: fh.write(json.dumps({"image_id": r["image_id"], "note": a.consent_note.strip(), "at": r["imported_at"]}) + "\n")
    st = "queued for annotation (the server picks them up without a restart)" if approved else \
         "PENDING APPROVAL: stored locally, not served for annotation (re-run with --approve-pending --approved-by ... --approval-ref ...)"
    print(f"imported {len(new_rows)} image(s): {st}")

if __name__ == "__main__": main()
