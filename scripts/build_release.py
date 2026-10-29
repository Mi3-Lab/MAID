#!/usr/bin/env python3
"""Build the WheelView v1.0 public release.

  python scripts/build_release.py meta                 # metadata only -> release/ (no images; safe anywhere, committed)
  python scripts/build_release.py images [--zip]       # LAB LAPTOP ONLY: images -> var/release/WheelView_v1.0/ (+ zip)
  python scripts/build_release.py sync [--zip]         # LAB LAPTOP ONLY: refresh metadata in the built folder, re-verify (+ zip)

Inputs (read-only): data/manifest_v2.csv (the 464 released images, splits v3), data/license_audit_v2.csv,
derived/attributions.csv + data/candidates.csv (titles), data/blur_manifest.jsonl (face counts, output size),
data/safety_hold.csv. The annotation event log, the DB and the blocklists are never read or written here.

meta writes release/{annotations/labels.csv, annotations/boxes_per_annotator.csv, annotations/coco_{train,val,test}.json,
splits/{train,val,test}.txt, LICENSES.csv, SHA256SUMS.txt, code/...}. README/DATASHEET/CHANGELOG are hand-written.

images copies images_blurred/<id>.jpg -> images/<split>/<id>.jpg byte-for-byte after checking: sha256 equals the
manifest's sha256_blurred (the copy the face-blur verification ran on), JPEG has no EXIF/XMP/ICC/IPTC/comment segments
(stripped at byte level if present, pixels untouched), longer side <= 1280 px. Safety-hold images and anything not in
the manifest are never copied. Then release/ metadata is copied in, SHA256SUMS is verified and (with --zip) the folder is
zipped to var/WheelView_v1.0_release.zip with its sha256 written next to it."""
import argparse, csv, hashlib, json, os, shutil, sys, zipfile
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server")); import wvcore
REL = ROOT / "release"; NAME = "WheelView_v1.0"
CLASSES = ["manual", "powered", "other_aid", "ambiguous"]
ANNOTATOR_IDS = {"samir": "annotator_1", "shaaz": "annotator_2"}   # pseudonyms used in the public files
LABEL_COLS = ["image_id", "file", "split", "train_eligible", "exclusion_reason", "device_type", "occupied", "propulsion",
              "occlusion", "viewpoint", "lighting", "setting", "label_how", "labelers", "adjudicated_by", "labeled_via",
              "n_boxes", "width", "height", "license", "license_url", "license_status", "creator", "source_site", "source_url",
              "uploader_group", "dup_cluster", "leak_group", "sha256"]
CODE_FILES = [("scripts/predict.py", "scripts/predict.py"), ("scripts/check_leakage.py", "scripts/check_leakage.py"),
              ("scripts/train_baseline.py", "scripts/train_baseline.py"), ("scripts/train_siglip.py", "scripts/train_siglip.py"),
              ("scripts/train_yolo.py", "scripts/train_yolo.py"), ("scripts/auto_annotate.py", "scripts/auto_annotate.py"),
              ("scripts/bench_latency.py", "scripts/bench_latency.py"), ("server/wvcore.py", "server/wvcore.py"),
              ("server/schema.json", "server/schema.json")]
LIC_NAMES = {"CC-BY-2.0": "CC BY 2.0", "CC-BY-SA-2.0": "CC BY-SA 2.0", "CC0-1.0": "CC0 1.0", "LicenseRef-PDM-1.0": "Public Domain Mark 1.0"}

def pseud(s):
    out = s
    for k, v in ANNOTATOR_IDS.items(): out = out.replace(k, v)
    return out

def rows():
    M = list(csv.DictReader(open(ROOT / "data" / "manifest_v2.csv", encoding="utf-8")))
    hold = {r["image_id"] for r in csv.DictReader(open(ROOT / "data" / "safety_hold.csv", encoding="utf-8"))}
    assert len(M) == len({r["image_id"] for r in M}) == 464, "manifest must have 464 unique images"
    assert not hold & {r["image_id"] for r in M}, "safety-hold image in manifest"
    assert all(r["license_status"] in ("ok", "ok_with_note") for r in M)
    for r in M: r["file"] = f"images/{r['split']}/{r['image_id']}.jpg"
    return M, hold

def blur_info():
    B = {}
    for l in open(ROOT / "data" / "blur_manifest.jsonl", encoding="utf-8"):
        d = json.loads(l); B[d["image_id"]] = d
    return B

def iou(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax1, bx1) - max(a[0], b[0])); ih = max(0, min(ay1, by1) - max(a[1], b[1])); i = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.0

def consensus(per):
    """per = {annotator: [[x,y,w,h], ...]} (normalised). Greedy IoU>=0.5 matching between two annotators -> averaged box
    (source consensus_2); unmatched boxes and boxes from a single annotator are kept as-is (single_annotator/unmatched)."""
    anns = sorted(per)
    if len(anns) == 1: return [(b, "single_annotator", anns) for b in per[anns[0]]]
    if len(anns) != 2: raise ValueError("more than 2 annotators")
    A, Bx = per[anns[0]], per[anns[1]]
    pairs = sorted(((iou(a, b), i, j) for i, a in enumerate(A) for j, b in enumerate(Bx)), reverse=True)
    ua, ub, out = set(), set(), []
    for v, i, j in pairs:
        if v < 0.5 or i in ua or j in ub: continue
        ua.add(i); ub.add(j)
        a, b = A[i], Bx[j]; x0 = (a[0] + b[0]) / 2; y0 = (a[1] + b[1]) / 2
        x1 = (a[0] + a[2] + b[0] + b[2]) / 2; y1 = (a[1] + a[3] + b[1] + b[3]) / 2
        out.append(([x0, y0, x1 - x0, y1 - y0], "consensus_2", anns, round(v, 3)))
    out = [(b, s, an, {"iou": v}) for b, s, an, v in out]
    out += [(A[i], "unmatched", [anns[0]]) for i in range(len(A)) if i not in ua]
    out += [(Bx[j], "unmatched", [anns[1]]) for j in range(len(Bx)) if j not in ub]
    return out

def write_csv(path, cols, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n"); w.writeheader(); w.writerows(data)

def meta(a):
    M, _ = rows(); B = blur_info()
    audit = {r["image_id"]: r for r in csv.DictReader(open(ROOT / "data" / "license_audit_v2.csv", encoding="utf-8"))}
    attr = {r["image_id"]: r for r in csv.DictReader(open(ROOT / "derived" / "attributions.csv", encoding="utf-8"))}
    cand = {r["candidate_id"]: r for r in csv.DictReader(open(ROOT / "data" / "candidates.csv", encoding="utf-8"))}
    # labels.csv
    L = []
    for r in M:
        b = B[r["image_id"]]
        d = {k: r.get(k, "") for k in LABEL_COLS}
        d.update(file=r["file"], width=b["width"], height=b["height"], sha256=r["sha256_blurred"], labelers=pseud(r["labelers"]),
                 adjudicated_by=pseud(r["adjudicated_by"]), labeled_via=pseud(r["labeled_via"]), device_type=wvcore.canon_class(r["device_type"]))
        L.append(d)
    write_csv(REL / "annotations" / "labels.csv", LABEL_COLS, L)
    for s in ("train", "val", "test"):
        (REL / "splits").mkdir(parents=True, exist_ok=True)
        (REL / "splits" / f"{s}.txt").write_text("".join(r["image_id"] + "\n" for r in M if r["split"] == s), encoding="utf-8")
    # boxes
    BX, coco = [], {s: {"images": [], "annotations": []} for s in ("train", "val", "test")}
    stats = Counter(); aid = 1
    for r in M:
        per = {pseud(k): v for k, v in json.loads(r["bboxes"]).items()} if r["bboxes"] else {}
        for an, bs in sorted(per.items()):
            for k, bb in enumerate(bs):
                BX.append({"image_id": r["image_id"], "annotator": an, "box_index": k, "x": bb[0], "y": bb[1], "w": bb[2], "h": bb[3]})
        if r["split"] not in coco: continue
        W, H = B[r["image_id"]]["width"], B[r["image_id"]]["height"]; iid = int(r["image_id"][3:])
        coco[r["split"]]["images"].append({"id": iid, "file_name": r["file"], "width": W, "height": H, "wv_image_id": r["image_id"],
                                           "n_annotators_with_boxes": len(per)})
        if not per: stats["images_without_boxes"] += 1; continue
        for item in consensus(per):
            bb, src, an = item[0], item[1], item[2]; extra = item[3] if len(item) > 3 else {}
            x, y, w, h = bb[0] * W, bb[1] * H, bb[2] * W, bb[3] * H
            coco[r["split"]]["annotations"].append({"id": aid, "image_id": iid, "category_id": CLASSES.index(r["device_type"]) + 1,
                "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)], "area": round(w * h, 2), "iscrowd": 0,
                "attributes": {"box_source": src, "annotators": an, "preliminary": True, "category_from": "image-level device_type", **extra}})
            aid += 1; stats[src] += 1
    write_csv(REL / "annotations" / "boxes_per_annotator.csv", ["image_id", "annotator", "box_index", "x", "y", "w", "h"], BX)
    cats = [{"id": i + 1, "name": c, "supercategory": "mobility_device" if c != "ambiguous" else "lookalike_object"} for i, c in enumerate(CLASSES)]
    for s, d in coco.items():
        d.update(info={"description": f"WheelView v1.0 device boxes, {s} split. PRELIMINARY: boxes were drawn by one annotator on most images; "
                       "where two annotators drew boxes, boxes matched at IoU>=0.5 were averaged (box_source=consensus_2) and the rest kept "
                       "(unmatched). Box categories are copied from the image-level device_type, not labelled per box.", "version": "1.0",
                       "year": 2026, "url": "", "date_created": "2026-10-29"}, licenses=[], categories=cats)
        (REL / "annotations" / f"coco_{s}.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
    # LICENSES.csv
    LC = ["image_id", "file", "title", "creator", "source_site", "source_url", "source_status", "license", "license_url",
          "license_recorded_by", "license_recorded_on", "license_rechecked_on", "license_on_source_at_recheck", "license_note",
          "modifications", "attribution"]
    out = []
    for r in M:
        i = r["image_id"]; au = audit[i]; b = B[i]; lc = au["live_check"]
        title = (attr.get(i) or {}).get("title") or cand[i]["title"]
        via = "Open Images V7" if r["source_site"].startswith("openimages") else "Openverse API"
        st = "live" if lc.startswith("match") else ("deleted at source" if lc == "gone" else ("login required" if lc == "login_required" else "live"))
        if "moved" in lc: st = "live (moved URL)"
        live = {"match": r["license"], "match;moved": r["license"], "gone": "unknown (photo deleted)", "login_required": "unknown (login required)"}.get(lc, lc.replace("changed:", ""))
        note = ""
        if lc.startswith("changed:") and lc != "changed:CC0-1.0":
            note = (f"Recorded as {r['license']} by {via} at collection; the source page now shows {live}. Distributed under the license "
                    "recorded at collection by team decision (2026-10-28); not confirmed with the photographer.")
        elif lc == "login_required":
            note = (f"Recorded as {r['license']} by {via} at collection; source page now requires a login, license could not be re-checked. "
                    "Distributed under the recorded license by team decision (2026-10-28).")
        elif lc == "gone": note = "Source photo deleted after collection; license and creator as recorded at collection."
        elif lc == "changed:CC0-1.0": note = f"Source page now shows CC0 1.0 (less restrictive); distributed under the recorded {r['license']}."
        elif "moved" in lc: note = au["reason"].split(";")[0]
        elif au["status"] == "ok_with_note": note = "An annotator raised a license question during labeling; the 2026-10-28 re-check found the recorded license unchanged on the source page."
        if r["license"] == "CC-BY-SA-2.0":
            note = (note + " " if note else "") + "ShareAlike: this adapted image is distributed under CC BY-SA 2.0, and adaptations of it must keep that license."
        nf = b["n_faces"]
        mods = ("resized to at most 640 px on the longer side and re-encoded as JPEG (all EXIF/XMP/ICC metadata removed); "
                + (f"{nf} face region{'s' if nf != 1 else ''} pixelated and blurred" if nf else "no face detected, no blur applied"))
        ln = LIC_NAMES.get(r["license"], r["license"])
        verb = "is marked with" if "PDM" in r["license"] else ("is dedicated to the public domain under" if "CC0" in r["license"] else "is licensed under")
        att = f"\"{title}\" by {r['creator']} ({r['source_url']}) {verb} {ln} ({r['license_url']}). Modified for WheelView: {mods}."
        out.append({"image_id": i, "file": r["file"], "title": title, "creator": r["creator"], "source_site": "Flickr (via " + via + ")",
                    "source_url": r["source_url"], "source_status": st, "license": r["license"], "license_url": r["license_url"],
                    "license_recorded_by": via, "license_recorded_on": (attr.get(i) or {}).get("date_accessed") or cand[i]["access_date"],
                    "license_rechecked_on": au["checked"], "license_on_source_at_recheck": live, "license_note": note,
                    "modifications": mods, "attribution": att})
    write_csv(REL / "LICENSES.csv", LC, out)
    (REL / "SHA256SUMS.txt").write_text("".join(f"{r['sha256_blurred']}  {r['file']}\n" for r in M), encoding="utf-8")
    # code snapshot
    for src, dst in CODE_FILES:
        (REL / "code" / dst).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / src, REL / "code" / dst)
    print(json.dumps({"labels": len(L), "splits": dict(Counter(r["split"] for r in M)), "boxes_rows": len(BX), "coco": dict(stats),
                      "coco_counts": {s: [len(d["images"]), len(d["annotations"])] for s, d in coco.items()},
                      "licenses": dict(Counter(r["license"] for r in M)), "license_notes": dict(Counter(bool(o["license_note"]) for o in out))}, indent=1))

SKIP_MARKERS = set(range(0xE1, 0xF0)) | {0xFE}   # APP1..APP15 (EXIF, XMP, ICC, IPTC, ...) and COM

def jpeg_strip(data):
    """Drop APP1-APP15 and COM segments before the scan; keep everything else byte-identical. Returns (bytes, dropped)."""
    assert data[:2] == b"\xff\xd8", "not a JPEG"
    out, i, dropped = bytearray(b"\xff\xd8"), 2, []
    while i < len(data):
        assert data[i] == 0xFF, f"bad marker at {i}"
        m = data[i + 1]
        if m == 0xDA: out += data[i:]; break                       # start of scan: rest is entropy-coded data
        if m in (0x01,) or 0xD0 <= m <= 0xD7: out += data[i:i + 2]; i += 2; continue
        n = int.from_bytes(data[i + 2:i + 4], "big")
        if m in SKIP_MARKERS: dropped.append(hex(m))
        else: out += data[i:i + 2 + n]
        i += 2 + n
    return bytes(out), dropped

def images(a):
    from PIL import Image
    M, hold = rows(); out = ROOT / "var" / "release" / NAME
    if out.exists() and any(out.rglob("*.jpg")):
        sys.exit(f"{out} already has images; move it aside first (this script never deletes)")
    rep = Counter(); problems = []
    for r in M:
        src = ROOT / r["path"]; data = src.read_bytes()
        if hashlib.sha256(data).hexdigest() != r["sha256_blurred"]: problems.append(f"{r['image_id']}: sha256 differs from manifest"); continue
        clean, dropped = jpeg_strip(data)
        if dropped: rep["stripped_segments"] += 1; problems.append(f"{r['image_id']}: metadata segments {dropped} found (stripped)")
        im = Image.open(src); w, h = im.size
        if max(w, h) > 1280: problems.append(f"{r['image_id']}: {w}x{h} > 1280"); continue
        dst = out / r["file"]; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(clean); rep[r["split"]] += 1
    finish(a, M, hold, out, dict(rep), problems)

def finish(a, M, hold, out, copied, problems):
    """Copy release/ metadata into the built folder (overwrites metadata files only; images untouched), verify, optionally zip."""
    for p in REL.rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts:
            d = out / p.relative_to(REL); d.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, d)
    bad = []
    for line in (out / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        h, f = line.split("  ", 1)
        if not (out / f).exists() or hashlib.sha256((out / f).read_bytes()).hexdigest() != h: bad.append(f)
    files = {r["file"] for r in M}
    extra = [p for p in out.rglob("*.jpg") if p.stem in hold or p.relative_to(out).as_posix() not in files]
    n_img = sum(1 for p in out.rglob("*.jpg") if p.relative_to(out).as_posix() in files)
    res = {"out": str(out), "copied": copied, "images_present": n_img, "problems": problems, "sha256sums_mismatch": bad, "unexpected_images": [str(p) for p in extra]}
    print(json.dumps(res, indent=1))
    if problems or bad or extra or n_img != len(M): sys.exit("release build FAILED")
    if a.zip:
        z = ROOT / "var" / f"{NAME}_release.zip"
        if z.exists(): sys.exit(f"{z} exists; move it aside first")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for p in sorted(out.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    zf.write(p, (Path(NAME) / p.relative_to(out)).as_posix(), compress_type=zipfile.ZIP_STORED if p.suffix == ".jpg" else zipfile.ZIP_DEFLATED)
        hz = hashlib.sha256(z.read_bytes()).hexdigest()
        Path(str(z) + ".sha256").write_text(f"{hz}  {z.name}\n", encoding="ascii")
        print(json.dumps({"zip": str(z), "bytes": z.stat().st_size, "sha256": hz, "files": len(zipfile.ZipFile(z).namelist())}))

def sync(a):
    """Refresh metadata in an already-built folder (after release/ changed) and re-verify; images are not touched."""
    M, hold = rows(); out = ROOT / "var" / "release" / NAME
    if not out.exists(): sys.exit(f"{out} missing; run the images mode first")
    finish(a, M, hold, out, {}, [])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["meta", "images", "sync"]); ap.add_argument("--zip", action="store_true")
    a = ap.parse_args(); {"meta": meta, "images": images, "sync": sync}[a.mode](a)
