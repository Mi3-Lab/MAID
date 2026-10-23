#!/usr/bin/env python3
"""Populate images/<image_id>.jpg (git-ignored) for every candidate.
Sources, in order: --from-dir (a local thumbs cache, e.g. the WheelView build's thumbs/ folder, matched by candidates.csv thumb_path),
then (with --download) the row's image_url, resized to <=640px and re-encoded (drops EXIF).
Examples:
  python scripts/fetch_images.py --from-dir /workspace/wheelview --link        # symlink existing thumbs (no copy)
  python scripts/fetch_images.py --download --source openverse --limit 200      # fetch from source URLs, politely
"""
import argparse, csv, io, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
UA = "WheelViewAnnotate/0.1 (UC Merced Mi3 Lab; research dataset annotation)"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=ROOT / "data" / "candidates.csv")
    ap.add_argument("--out", default=os.environ.get("WV_IMAGE_DIR", ROOT / "images"))
    ap.add_argument("--from-dir", help="directory that candidates.csv thumb_path values are relative to")
    ap.add_argument("--link", action="store_true", help="symlink instead of copying from --from-dir")
    ap.add_argument("--download", action="store_true", help="download missing images from image_url")
    ap.add_argument("--source", help="only rows of this source (openimages/openverse/commons)")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--delay", type=float, default=1.0)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.candidates, newline="", encoding="utf-8")))
    if a.source: rows = [r for r in rows if r["source"] == a.source]
    stats = dict(present=0, linked=0, copied=0, downloaded=0, failed=0, skipped=0); n_dl = 0
    for r in rows:
        dst = out / f"{r['candidate_id']}.jpg"
        if dst.exists(): stats["present"] += 1; continue
        if a.from_dir and r.get("thumb_path"):
            src = Path(a.from_dir) / r["thumb_path"]
            if src.exists():
                if a.link: dst.symlink_to(src.resolve()); stats["linked"] += 1
                else: dst.write_bytes(src.read_bytes()); stats["copied"] += 1
                continue
        if not a.download or not r.get("image_url", "").startswith("https://") or (a.limit and n_dl >= a.limit):
            stats["skipped"] += 1; continue
        import httpx; from PIL import Image
        n_dl += 1
        for attempt in range(4):
            try:
                resp = httpx.get(r["image_url"], headers={"User-Agent": UA}, timeout=60, follow_redirects=True)
                if resp.status_code == 429: time.sleep(30 * (attempt + 1)); continue
                resp.raise_for_status()
                im = Image.open(io.BytesIO(resp.content)).convert("RGB"); im.thumbnail((640, 640)); im.save(dst, "JPEG", quality=88)
                stats["downloaded"] += 1; break
            except Exception as e:
                print("fail", r["candidate_id"], e, file=sys.stderr); time.sleep(5)
        else: stats["failed"] += 1
        time.sleep(a.delay)
    print(stats)

if __name__ == "__main__": main()
