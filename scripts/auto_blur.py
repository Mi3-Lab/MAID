#!/usr/bin/env python3
"""Automatic face blurring for the release copies of the images. Runs fully locally (no network, no hosted API).

Detector: OpenCV YuNet (face_detection_yunet_2023mar.onnx, MIT licence, models/), run for HIGH RECALL:
  low score threshold (0.35 for small faces; 0.72 for boxes wider than 6% of the image), the image at 1x + 2x/3x upscale (small faces in 640px thumbnails), plus the mirrored image
  (catches extra profile faces), merged with NMS. False positives only blur a little extra background; a missed face
  is the expensive error, so humans still flag misses with 'f' (face_blur_missed) in the app.
Blur: padded box (hair/ears/chin), pixelated to ~5 cells across the face, then a strong Gaussian blur -> not recoverable.

NEVER touches the originals (images/, var/imgcache/). Writes re-encoded copies WITHOUT EXIF/metadata to
images_blurred/<id>.jpg (git-ignored, like images/) for every processed image (also those with 0 faces, so
"processed" == "file exists"), and logs one JSON line per image to the manifest (committed; ids + boxes only):
  data/blur_manifest.jsonl  {image_id, src_sha1, out_sha1, n_faces, faces:[{box:[x,y,w,h] normalized, pad_box, score}],
                             detector, model_sha256, score_threshold, scales, blur, version, at}
Incremental + resumable: images whose src_sha1 + version already appear in the manifest are not re-detected; their
recorded boxes are re-applied if the blurred file is missing (so another machine reproduces the same release copies
without re-deciding). Use --local on the laptop so the committed manifest is never modified there.

  python scripts/auto_blur.py                  # all local images, wheelchair rows first
  python scripts/auto_blur.py --limit 200      # a sample
  python scripts/auto_blur.py --local          # laptop: new detections go to var/blur_manifest_local.jsonl
Requires only: pip install opencv-python-headless numpy   (no torch)"""
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np, cv2

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
DETECTOR = "opencv-yunet-2023mar"; DETECTOR_LICENSE = "MIT (opencv_zoo face_detection_yunet; Shiqi Yu)"
VERSION = "blur-v3"
SCORE_THR = 0.35          # YuNet default is 0.9; low for recall: any box >= SCORE_THR that is SMALL is blurred
SCORE_LARGE = 0.72        # boxes wider than SMALL_FRAC of the image need this score (big low-score boxes were hands/balls/wheels)
SMALL_FRAC = 0.06
ASPECT = (0.6, 2.3)       # face box h/w
NMS_THR = 0.3
MIN_FACE_PX = 6
PAD = {"x": 0.30, "top": 0.40, "bottom": 0.25}
BLUR = "pixelate(5 cells/face) + gaussian(sigma=cell); padded box x±30% top+40% bottom+25%"
OUT_DIR = Path(os.environ.get("WV_BLUR_DIR", ROOT / "images_blurred"))
MANIFEST = ROOT / "data" / "blur_manifest.jsonl"
LOCAL_MANIFEST = ROOT / "var" / "blur_manifest_local.jsonl"
IMG_DIRS = [ROOT / "images", ROOT / "var" / "imgcache"]

def sha1(p): return hashlib.sha1(Path(p).read_bytes()).hexdigest()

def src_path(i):
    for d in IMG_DIRS:
        p = d / f"{i}.jpg"
        if p.exists(): return p
    return None

_det = None
def detector():
    global _det
    if _det is None:
        if not MODEL.exists(): sys.exit(f"missing {MODEL} (git pull)")
        if hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA256: sys.exit(f"{MODEL} checksum mismatch")
        _det = cv2.FaceDetectorYN.create(str(MODEL), "", (320, 320), SCORE_THR, NMS_THR, 5000)
    return _det

def scales_for(h, w):
    m = max(h, w)
    return [1.0, 2.0] + ([3.0] if m <= 700 else [])

def nms(boxes, thr=0.3):
    boxes = sorted(boxes, key=lambda b: -b[4]); keep = []
    for b in boxes:
        ok = True
        for k in keep:
            ix = max(0, min(b[0] + b[2], k[0] + k[2]) - max(b[0], k[0])); iy = max(0, min(b[1] + b[3], k[1] + k[3]) - max(b[1], k[1]))
            inter = ix * iy; u = b[2] * b[3] + k[2] * k[3] - inter
            if u > 0 and (inter / u > thr or inter / max(1e-9, min(b[2] * b[3], k[2] * k[3])) > 0.7): ok = False; break
        if ok: keep.append(b)
    return keep

def detect(img):
    """-> list of (x, y, w, h, score) in original pixel coords."""
    h, w = img.shape[:2]; d = detector(); out = []
    for flip in (False, True):
        base = cv2.flip(img, 1) if flip else img
        for s in scales_for(h, w):
            im = base if s == 1.0 else cv2.resize(base, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
            d.setInputSize((im.shape[1], im.shape[0]))
            _, faces = d.detect(im)
            for f in (faces if faces is not None else []):
                x, y, bw, bh = (float(v) / s for v in f[:4]); sc = float(f[-1])
                if flip: x = w - x - bw
                x, y = max(0.0, x), max(0.0, y); bw, bh = min(bw, w - x), min(bh, h - y)
                if bw >= MIN_FACE_PX and bh >= MIN_FACE_PX: out.append((x, y, bw, bh, sc))
    keep = [b for b in out if ASPECT[0] <= b[3] / b[2] <= ASPECT[1] and (b[4] >= SCORE_LARGE or b[2] <= SMALL_FRAC * w)]
    return nms(keep, NMS_THR)

def pad_box(b, w, h):
    x, y, bw, bh = b[:4]
    x0 = max(0, int(x - PAD["x"] * bw)); x1 = min(w, int(np.ceil(x + bw + PAD["x"] * bw)))
    y0 = max(0, int(y - PAD["top"] * bh)); y1 = min(h, int(np.ceil(y + bh + PAD["bottom"] * bh)))
    return x0, y0, x1, y1

def blur(img, faces_px):
    out = img.copy(); h, w = img.shape[:2]
    for b in faces_px:
        x0, y0, x1, y1 = pad_box(b, w, h)
        if x1 - x0 < 2 or y1 - y0 < 2: continue
        roi = out[y0:y1, x0:x1]
        cells = 5; cw = max(1, int(np.ceil(max(x1 - x0, y1 - y0) / cells)))
        small = cv2.resize(roi, (max(1, (x1 - x0) // cw), max(1, (y1 - y0) // cw)), interpolation=cv2.INTER_AREA)
        pix = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        k = 2 * int(cw) + 1
        out[y0:y1, x0:x1] = cv2.GaussianBlur(pix, (k, k), cw)
    return out

def write_jpg(path, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])   # cv2 writes no EXIF/ICC/XMP
    if not ok: raise RuntimeError("encode failed")
    tmp = path.with_suffix(".tmp"); tmp.write_bytes(buf.tobytes()); os.replace(tmp, path)
    return hashlib.sha1(buf.tobytes()).hexdigest()

def load_manifest(*paths):
    m = {}
    for p in paths:
        if p and Path(p).exists():
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line); m[r["image_id"]] = r
    return m

def priority(cands):
    def k(i):
        pc = cands[i].get("proposed_class", "")
        return (0 if pc == "wheelchair_unspecified" else 1 if "wheelchair" in pc else 2, i)
    return sorted(cands, key=k)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--ids", default="", help="comma-separated image ids")
    ap.add_argument("--local", action="store_true", help="write new manifest lines to var/blur_manifest_local.jsonl (laptop)")
    ap.add_argument("--force", action="store_true", help="re-detect even if already in the manifest")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads: cv2.setNumThreads(a.threads)
    cands = {r["candidate_id"]: r for r in csv.DictReader(open(ROOT / "data" / "candidates.csv", encoding="utf-8"))}
    ids = [i for i in (a.ids.split(",") if a.ids else priority(cands)) if i in cands and src_path(i)]
    if a.limit: ids = ids[:a.limit]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    man_path = LOCAL_MANIFEST if a.local else MANIFEST; man_path.parent.mkdir(parents=True, exist_ok=True)
    known = load_manifest(MANIFEST, LOCAL_MANIFEST if a.local else None)
    t0 = time.time(); n_new = n_reuse = n_skip = faces = 0; new = {}
    for n, i in enumerate(ids):
        sp = src_path(i); s1 = sha1(sp); rec = known.get(i); outp = OUT_DIR / f"{i}.jpg"
        same = rec and rec.get("src_sha1") == s1 and rec.get("version") == VERSION and not a.force
        if same and outp.exists(): n_skip += 1; faces += rec["n_faces"]; continue
        img = cv2.imread(str(sp), cv2.IMREAD_COLOR)
        if img is None: print(f"unreadable {sp}", file=sys.stderr); continue
        h, w = img.shape[:2]
        if same:      # reproduce recorded decision (no re-detection)
            px = [(f["box"][0] * w, f["box"][1] * h, f["box"][2] * w, f["box"][3] * h, f["score"]) for f in rec["faces"]]
            write_jpg(outp, blur(img, px)); n_reuse += 1; faces += rec["n_faces"]; continue
        px = detect(img); oh = write_jpg(outp, blur(img, px))
        r = {"image_id": i, "src_sha1": s1, "out_sha1": oh, "width": w, "height": h, "n_faces": len(px),
             "faces": [{"box": [round(b[0] / w, 4), round(b[1] / h, 4), round(b[2] / w, 4), round(b[3] / h, 4)],
                        "pad_box": [round(v, 4) for v in np.array(pad_box(b, w, h)) / [w, h, w, h]], "score": round(b[4], 3)} for b in px],
             "detector": DETECTOR, "model_sha256": MODEL_SHA256, "score_threshold": SCORE_THR, "score_threshold_large": SCORE_LARGE, "small_frac": SMALL_FRAC, "aspect": list(ASPECT), "nms_threshold": NMS_THR,
             "scales": scales_for(h, w), "mirrored": True, "blur": BLUR, "version": VERSION, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        new[i] = r; n_new += 1; faces += len(px)
        if len(new) % 200 == 0: flush(man_path, a.local, known, new)
        if (n + 1) % 100 == 0: print(f"{n + 1}/{len(ids)}  {(time.time() - t0) / max(1, n_new + n_reuse):.2f} s/img  faces so far {faces}", flush=True)
    flush(man_path, a.local, known, new)
    print(json.dumps({"images": len(ids), "detected_now": n_new, "reapplied": n_reuse, "already_done": n_skip, "faces": faces,
                      "manifest": str(man_path.relative_to(ROOT)), "out_dir": str(OUT_DIR), "seconds": round(time.time() - t0, 1)}))

def flush(man_path, local, known, new):
    """Rewrite the manifest sorted by image_id (stable diffs); local mode only holds the laptop's own lines."""
    base = load_manifest(man_path) if man_path.exists() else {}
    base.update(new)
    tmp = man_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        for i in sorted(base): fh.write(json.dumps(base[i], sort_keys=True) + "\n")
    os.replace(tmp, man_path)

if __name__ == "__main__": main()
