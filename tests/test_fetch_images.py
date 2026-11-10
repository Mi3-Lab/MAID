#!/usr/bin/env python3
"""Unit tests for scripts/fetch_images.py and the missing-image placeholder.
No live downloads of the main pool; scratch dirs only.
  python tests/test_fetch_images.py
"""
import csv, io, json, os, sys, tempfile, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_images as FI

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f" -- {detail}" if detail else ""), flush=True)

# Flickr fallback
u = "https://farm4.staticflickr.com/123/abc_z.jpg"
got = FI.candidate_urls(u)
check("flickr farm falls back to live.staticflickr.com",
      got[0] == u and got[1] == "https://live.staticflickr.com/123/abc_z.jpg", str(got))
u2 = "https://c1.staticflickr.com/9/x.jpg"
check("flickr c* host also falls back",
      FI.candidate_urls(u2)[1] == "https://live.staticflickr.com/9/x.jpg", str(FI.candidate_urls(u2)))
u3 = "https://upload.wikimedia.org/wikipedia/commons/a/ab/x.jpg"
check("non-flickr URL has no extra candidate", FI.candidate_urls(u3) == [u3])

# process_bytes re-encodes, drops EXIF, caps at 640
from PIL import Image
im = Image.new("RGB", (1200, 800), (10, 20, 30))
buf = io.BytesIO(); im.save(buf, "JPEG", quality=95)
out = FI.process_bytes(buf.getvalue())
im2 = Image.open(io.BytesIO(out))
check("process_bytes is JPEG <=640", im2.format == "JPEG" and max(im2.size) == 640, str(im2.size))

# skip lists
TMP = Path(tempfile.mkdtemp(prefix="wv_fetch_"))
data = TMP / "data"; data.mkdir()
(data / "safety_hold.csv").write_text("image_id,flagged,flagged_by,reason\nwv_000250,yes,t,child\n", encoding="utf-8")
(data / "removed_model_child_stroller.txt").write_text("wv_000003\n# comment\nwv_000004  # inline\n", encoding="utf-8")
(data / "license_exclusions.csv").write_text("image_id,reason,decided\nwv_000354,license,x\n", encoding="utf-8")
skip = FI.load_skip_ids(data)
check("safety_hold skipped", skip.get("wv_000250") == "safety_hold", str(skip))
check("removed_* skipped", skip.get("wv_000003", "").startswith("excluded:"), str(skip))
check("license exclusion skipped", skip.get("wv_000354") == "license_exclusion", str(skip))

# load_web_rows drops skipped and pmma, keeps http urls
cands = TMP / "candidates.csv"
with cands.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, ["candidate_id", "source", "image_url"])
    w.writeheader()
    w.writerow({"candidate_id": "wv_000001", "source": "commons", "image_url": "https://example.org/a.jpg"})
    w.writerow({"candidate_id": "wv_000250", "source": "commons", "image_url": "https://example.org/b.jpg"})
    w.writerow({"candidate_id": "wv_000003", "source": "openverse", "image_url": "https://example.org/c.jpg"})
    w.writerow({"candidate_id": "pmma_000001", "source": "pmma", "image_url": "https://example.org/p.jpg"})
    w.writerow({"candidate_id": "wv_000009", "source": "openimages", "image_url": ""})
rows, skipped = FI.load_web_rows(cands, [], skip)
ids = [r["image_id"] for r in rows]
check("only downloadable non-skipped web row remains", ids == ["wv_000001"], str(ids))
check("skipped includes hold and block", {s["image_id"] for s in skipped} >= {"wv_000250", "wv_000003"})

# --check on empty out dir
out = TMP / "images"; out.mkdir()
(data / "pmma_frames.jsonl").write_text(json.dumps({"image_id": "pmma_000001", "file": "video_2_1.zip", "frame": "x"}) + "\n", encoding="utf-8")
st = FI.check_status(out, rows, data / "pmma_frames.jsonl")
check("--check counts missing web + pmma", st["web_missing"] == 1 and st["pmma_missing"] == 1, str(st))
code = FI.main(["--candidates", str(cands), "--data-dir", str(data), "--out", str(out),
                "--failures", str(TMP / "fail.csv"), "--check"])
check("--check exits 2 when web images are missing", code == 2, str(code))
# pretend the web image is present
(out / "wv_000001.jpg").write_bytes(FI.process_bytes(buf.getvalue()))
code = FI.main(["--candidates", str(cands), "--data-dir", str(data), "--out", str(out),
                "--failures", str(TMP / "fail.csv"), "--check"])
check("--check exits 0 when web images present (pmma still noted)", code == 0, str(code))

# placeholder JPEG from the server (temp env, proxy off, no local file)
import uuid as _uuid
from fastapi.testclient import TestClient
ENVTMP = Path(tempfile.mkdtemp(prefix="wv_fetch_srv_"))
for d in ("auto", "block", "images", "images_blurred"):
    (ENVTMP / d).mkdir()
(ENVTMP / "blur.jsonl").write_text("")
(ENVTMP / "block" / "removed_test.txt").write_text("wv_000099\n")
with (ENVTMP / "candidates.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, ["candidate_id", "source", "creator", "license", "needs_visual_screen", "image_url"])
    w.writeheader()
    w.writerow({"candidate_id": "wv_000001", "source": "commons", "creator": "p", "license": "CC-BY-4.0",
                "needs_visual_screen": "no", "image_url": "https://example.org/a.jpg"})
    w.writerow({"candidate_id": "wv_000099", "source": "commons", "creator": "p", "license": "CC-BY-4.0",
                "needs_visual_screen": "no", "image_url": "https://example.org/b.jpg"})
env = {"WV_DB": str(ENVTMP / "scratch.db"), "WV_CANDIDATES": str(ENVTMP / "candidates.csv"),
       "WV_CONTRIBUTED": str(ENVTMP / "contributed.csv"), "WV_IMAGE_DIR": str(ENVTMP / "images"),
       "WV_BLUR_DIR": str(ENVTMP / "images_blurred"), "WV_EXPORT_DIR": str(ENVTMP / "exports"),
       "WV_AUTO_DIR": str(ENVTMP / "auto"), "WV_BLOCKLIST_DIR": str(ENVTMP / "block"),
       "WV_IMAGE_PROXY": "0", "WV_ACCESS_CODE": "", "WV_BLUR_MANIFESTS": str(ENVTMP / "blur.jsonl"),
       "WV_CACHE_DIR": str(ENVTMP / "cache"), "WV_ASSIGN_WITHOUT_LOCAL_IMAGE": "1"}
for k in ("WV_HF_DATASET",):
    os.environ.pop(k, None)
os.environ.update(env)
sys.path.insert(0, str(ROOT / "server"))
import main
with TestClient(main.app) as cl:
    r = cl.get("/api/images/wv_000001")
    check("api image_present is false when file missing", r.status_code == 200 and r.json().get("image_present") is False, r.text[:200])
    check("api image_note tells you to run fetch_images", "fetch_images" in (r.json().get("image_note") or ""), str(r.json()))
    img = cl.get("/img/wv_000001")
    check("missing image returns a JPEG placeholder, not 404", img.status_code == 200 and img.headers.get("content-type", "").startswith("image/jpeg"), str(img.status_code))
    check("placeholder header X-WV-Image-Missing", img.headers.get("X-WV-Image-Missing") == "1")
    body = Image.open(io.BytesIO(img.content))
    check("placeholder is a real image", body.size[0] >= 320, str(body.size))
    blocked = cl.get("/img/wv_000099")
    check("blocked image is still 404", blocked.status_code == 404)

# Dataverse listing (network, public metadata only -- no multi-GB download)
try:
    import httpx
    with httpx.Client(headers={"User-Agent": FI.UA}, timeout=30.0, follow_redirects=True) as client:
        listing = FI.list_dataverse_files(client)
    check("Borealis public API lists video_2_1.zip and video_3_part1.mp4",
          "video_2_1.zip" in listing and "video_3_part1.mp4" in listing and listing["video_2_1.zip"].get("id"),
          str({k: listing[k] for k in FI.PMMA_NEEDED if k in listing}))
except Exception as e:
    check("Borealis public API lists video_2_1.zip and video_3_part1.mp4", False, str(e))

failed = [n for n, ok in results if not ok]
print(f"\n{sum(ok for _, ok in results)}/{len(results)} passed; temp {TMP}")
sys.exit(1 if failed else 0)