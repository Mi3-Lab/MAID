#!/usr/bin/env python3
"""Reproduce PMMA frames used in the MAID v1.1-with-auto training split.

Dataset: Polytechnique Montreal Mobility Aids Dataset (PMMA)
  DOI: 10.5683/SP3/XJPQUG
  Landing: https://doi.org/10.5683/SP3/XJPQUG
  License: CC0 1.0 (https://creativecommons.org/publicdomain/zero/1.0/)

This repo does not ship video or frames. Download the Borealis files yourself,
or run: python scripts/fetch_images.py --pmma

Supported inputs (match data/pmma_frames.jsonl "file" field):
  - video_2_1.zip   (PNG frames inside)
  - video_3_part1.mp4  (0-based frame index)

Optional: --blur runs scripts/auto_blur.py on the extracted JPEGs.

PowerShell 5.1 (separate lines, not &&):
  python scripts/extract_pmma.py --video2-zip var/pmma/video_2_1.zip --video3-mp4 var/pmma/video_3_part1.mp4
  python scripts/extract_pmma.py ... --blur

Requires: pip install opencv-python-headless pillow numpy
"""
from __future__ import annotations
import argparse, io, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default=str(ROOT / "data" / "pmma_frames.jsonl"))
    ap.add_argument("--out", default=os.environ.get("WV_IMAGE_DIR", str(ROOT / "images")))
    ap.add_argument("--video2-zip", default=None)
    ap.add_argument("--video3-mp4", default=None)
    ap.add_argument("--blur", action="store_true")
    ap.add_argument("--max-edge", type=int, default=1280)
    a = ap.parse_args()
    try:
        import cv2
        from PIL import Image
        import numpy as np  # noqa: F401
        import zipfile, json, subprocess
    except ImportError:
        sys.exit("Need: pip install opencv-python-headless pillow numpy")

    if not a.video2_zip and not a.video3_mp4:
        sys.exit(
            "PMMA extract needs local Borealis files.\n"
            "  DOI 10.5683/SP3/XJPQUG  https://doi.org/10.5683/SP3/XJPQUG\n"
            "  Pass --video2-zip path/to/video_2_1.zip and/or --video3-mp4 path/to/video_3_part1.mp4\n"
            "  Or run: python scripts/fetch_images.py --pmma"
        )
    rows = [json.loads(l) for l in open(a.frames, encoding="utf-8") if l.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(a.video2_zip) if a.video2_zip else None
    cap = None
    if a.video3_mp4:
        cap = cv2.VideoCapture(a.video3_mp4)
        if not cap.isOpened():
            sys.exit("cannot open %s" % a.video3_mp4)

    ok = miss = 0
    written = []
    for r in rows:
        iid = r["image_id"]
        dest = out / (iid + ".jpg")
        lg = r.get("leak_group") or ""
        try:
            if lg == "pmma_video_2" or str(r.get("file") or "").endswith(".zip"):
                if not zf:
                    print("skip (no --video2-zip)", iid); miss += 1; continue
                name = r.get("orig") or r.get("frame")
                base = Path(str(name)).name
                matches = [n for n in zf.namelist() if n.endswith(base)]
                if not matches:
                    print("missing in zip", iid, base); miss += 1; continue
                raw = zf.read(matches[0])
                im = Image.open(io.BytesIO(raw)).convert("RGB")
            elif lg == "pmma_video_3" or str(r.get("file") or "").endswith(".mp4"):
                if not cap:
                    print("skip (no --video3-mp4)", iid); miss += 1; continue
                fi = int(r["frame"])
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                good, frame = cap.read()
                if not good:
                    print("bad frame", iid, fi); miss += 1; continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(rgb)
            else:
                print("unknown pmma row", iid); miss += 1; continue
            w, h = im.size
            scale = a.max_edge / max(w, h)
            if scale < 1:
                im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            im.save(dest, quality=90)
            ok += 1
            written.append(iid)
        except Exception as e:
            print("fail", iid, e); miss += 1
    if zf:
        zf.close()
    if cap:
        cap.release()
    print({"extracted": ok, "missed": miss, "out": str(out)})
    if a.blur and written:
        blur = ROOT / "scripts" / "auto_blur.py"
        if not blur.exists():
            sys.exit("scripts/auto_blur.py missing; extracted files are in %s" % out)
        env = os.environ.copy()
        env["WV_IMAGE_DIR"] = str(out)
        subprocess.check_call([sys.executable, str(blur), "--ids", ",".join(written)], env=env)


if __name__ == "__main__":
    main()