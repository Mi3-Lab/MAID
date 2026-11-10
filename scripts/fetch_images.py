#!/usr/bin/env python3
"""Download the MAID image pool onto this machine. Images are not in git.

Anyone who clones Mi3-Lab/MAID (v1.1-schema) can run this once and get every
web image the annotator or training split references, in the directory the
server already expects (images/, or WV_IMAGE_DIR).

Idempotent and resumable: existing files are skipped. Reads data/candidates.csv
and optional manifest CSVs for open-license source URLs. Skips child /
safety_hold / license-excluded images. Re-encodes to RGB JPEG (long edge <= 640,
quality 88), then applies the same face-blur step as the lab pipeline
(scripts/auto_blur.py).

Download logic matches maid-training-manifest/download.py: soft sha256 when a
hash column is present, 429/5xx retries with backoff, resume, and a
live.staticflickr.com fallback for farm/c*.staticflickr.com URLs.

PMMA frames (pmma_000001..) in the v1.1-with-auto training split are not in
candidates.csv. They come from Borealis DOI 10.5683/SP3/XJPQUG (CC0 1.0).
The public Dataverse API allows GET of the source files (large: video_2_1.zip
~3.1 GB, video_3_part1.mp4 ~2.1 GB). Pass --pmma to download + extract, or
download those two files yourself and run scripts/extract_pmma.py.

PowerShell 5.1: run each command on its own line (do not use &&):
  python scripts/fetch_images.py --limit 20
  python scripts/fetch_images.py
  python scripts/fetch_images.py --pmma
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UA = "MAID-annotate/1.1 (research; +https://github.com/Mi3-Lab/MAID)"
FLICKR_HOST_RE = re.compile(r"^https?://(?:farm\d+|c\d+)\.staticflickr\.com/", re.I)
PMMA_DOI = "10.5683/SP3/XJPQUG"
DATAVERSE = "https://borealisdata.ca"
DATASET_API = f"{DATAVERSE}/api/datasets/:persistentId/?persistentId=doi:{PMMA_DOI}"
PMMA_NEEDED = ("video_2_1.zip", "video_3_part1.mp4")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def process_bytes(raw: bytes) -> bytes:
    from PIL import Image
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im.thumbnail((640, 640))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def candidate_urls(url: str) -> list[str]:
    """Original URL first; for legacy Flickr farm/c hosts also try live.staticflickr.com."""
    out = [url]
    if FLICKR_HOST_RE.match(url):
        live = FLICKR_HOST_RE.sub("https://live.staticflickr.com/", url)
        if live not in out:
            out.append(live)
    return out


def fetch_with_retries(client, url: str, retries: int, delay: float):
    last_err = None
    for attempt in range(retries + 1):
        try:
            for u in candidate_urls(url):
                resp = client.get(u)
                if resp.status_code == 429:
                    last_err = f"HTTP 429 for {u}"
                    break
                if resp.status_code in (404, 410) and u != url:
                    continue
                if resp.status_code >= 500:
                    last_err = f"HTTP {resp.status_code} for {u}"
                    break
                resp.raise_for_status()
                ctype = (resp.headers.get("content-type") or "").lower()
                if not ctype.startswith("image") and len(resp.content) < 1000:
                    last_err = f"not an image ({ctype or 'unknown'}) from {u}"
                    continue
                return resp.content, u
            else:
                if last_err is None:
                    last_err = f"HTTP failure for {url}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        sleep_s = delay * (2 ** attempt) if attempt < retries else 0
        if "429" in str(last_err):
            sleep_s = max(sleep_s, 30 * (attempt + 1))
        if sleep_s:
            time.sleep(sleep_s)
    raise RuntimeError(last_err or "download failed")


def load_skip_ids(data_dir: Path) -> dict:
    """image_id -> reason for child / safety_hold / license-excluded / removed_* lists."""
    skip = {}

    def add(iid, reason):
        iid = (iid or "").strip()
        if iid and iid not in skip:
            skip[iid] = reason

    hold = data_dir / "safety_hold.csv"
    if hold.exists():
        with hold.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                add(r.get("image_id") or r.get("candidate_id") or "", "safety_hold")
    for p in sorted(data_dir.glob("removed_*.txt")):
        for line in p.read_text(encoding="utf-8").splitlines():
            t = line.split("#", 1)[0].strip()
            if t:
                add(t, "excluded:" + p.name)
    lic = data_dir / "license_exclusions.csv"
    if lic.exists():
        with lic.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                add(r.get("image_id") or r.get("candidate_id") or "", "license_exclusion")
    return skip


def load_web_rows(candidates: Path, manifests, skip):
    """Rows with a downloadable http URL, minus skipped ids.

    Accepts candidates.csv (candidate_id + image_url) and optional manifest
    CSVs (image_id + download_url or image_url, optional sha256_original).
    """
    by_id = {}

    def eat(path: Path):
        if not path or not path.exists():
            return
        with path.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                iid = (r.get("candidate_id") or r.get("image_id") or "").strip()
                if not iid:
                    continue
                url = (r.get("image_url") or r.get("download_url") or "").strip()
                sha = (r.get("sha256_original") or r.get("sha256") or "").strip()
                kind = (r.get("source_kind") or r.get("source") or "").strip()
                if kind == "pmma_video_frame" or iid.startswith("pmma_"):
                    continue
                rec = by_id.setdefault(iid, {"image_id": iid, "url": "", "sha256": "", "source": kind})
                if url.startswith("http"):
                    rec["url"] = url
                if sha:
                    rec["sha256"] = sha
                if kind:
                    rec["source"] = kind

    eat(candidates)
    for m in manifests:
        eat(m)
    rows, skipped = [], []
    for iid, rec in by_id.items():
        if iid in skip:
            skipped.append({"image_id": iid, "reason": skip[iid]})
            continue
        if rec.get("url", "").startswith("http"):
            rows.append(rec)
    return rows, skipped


def load_thumb_paths(candidates: Path):
    if not candidates.exists():
        return {}
    out = {}
    with candidates.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            iid = (r.get("candidate_id") or "").strip()
            tp = (r.get("thumb_path") or "").strip()
            if iid and tp:
                out[iid] = tp
    return out


def copy_from_dir(rows, from_dir: Path, out: Path, link: bool, thumb_by_id):
    stats = {"linked": 0, "copied": 0}
    copied_ids = []
    for rec in rows:
        dest = out / (rec["image_id"] + ".jpg")
        if dest.exists():
            continue
        rel = thumb_by_id.get(rec["image_id"]) or ""
        src = (from_dir / rel) if rel else None
        if not src or not src.exists():
            for cand in (from_dir / (rec["image_id"] + ".jpg"), from_dir / "thumbs" / (rec["image_id"] + ".jpg")):
                if cand.exists():
                    src = cand
                    break
            else:
                continue
        if link:
            dest.symlink_to(src.resolve())
            stats["linked"] += 1
        else:
            dest.write_bytes(src.read_bytes())
            stats["copied"] += 1
        copied_ids.append(rec["image_id"])
    stats["ids"] = copied_ids
    return stats


def run_blur(ids, image_dir: Path, blur_dir: Path) -> str:
    if not ids:
        return "blur skipped (no new images)"
    blur = ROOT / "scripts" / "auto_blur.py"
    if not blur.exists():
        return "blur skipped (scripts/auto_blur.py missing)"
    import subprocess
    env = os.environ.copy()
    env["WV_IMAGE_DIR"] = str(image_dir)
    env["WV_BLUR_DIR"] = str(blur_dir)
    cmd = [sys.executable, str(blur), "--ids", ",".join(ids)]
    try:
        subprocess.check_call(cmd, env=env)
        return "blurred %d image(s) -> %s" % (len(ids), blur_dir)
    except subprocess.CalledProcessError as e:
        return ("blur failed (exit %s). Install opencv if needed: "
                "pip install opencv-python-headless. Then: python scripts/auto_blur.py" % e.returncode)
    except Exception as e:
        return "blur failed: %s" % e


def pmma_instructions() -> str:
    return (
        "PMMA frames (266 images, pmma_000001..) are in the v1.1-with-auto training split "
        "but not in candidates.csv. Source: Borealis DOI %s "
        "(https://doi.org/%s), license CC0 1.0.\n"
        "  Needed files: %s\n"
        "  Automated (large, ~5.3 GB):  python scripts/fetch_images.py --pmma\n"
        "  Or download those two files from the DOI page into var/pmma/ and run:\n"
        "    python scripts/extract_pmma.py --video2-zip var/pmma/video_2_1.zip "
        "--video3-mp4 var/pmma/video_3_part1.mp4"
    ) % (PMMA_DOI, PMMA_DOI, ", ".join(PMMA_NEEDED))


def list_dataverse_files(client):
    """filename -> {id, size, md5} for the published PMMA dataset. Public API, no auth."""
    r = client.get(DATASET_API)
    r.raise_for_status()
    payload = r.json()
    files = (payload.get("data") or {}).get("latestVersion", {}).get("files") or []
    out = {}
    for rec in files:
        df = rec.get("dataFile") or {}
        name = df.get("filename") or rec.get("label") or ""
        if not name:
            continue
        checksum = df.get("checksum") or {}
        md5 = checksum.get("value") if checksum.get("type") == "MD5" else df.get("md5")
        out[name] = {"id": df.get("id"), "size": df.get("filesize"), "md5": md5, "restricted": rec.get("restricted")}
    return out


def download_dataverse_file(client, file_id, dest: Path, expected_size=None):
    """GET /api/access/datafile/:id with resume. HEAD is 403 on Borealis; GET works."""
    url = "%s/api/access/datafile/%s" % (DATAVERSE, file_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    if expected_size and existing >= expected_size and expected_size > 0:
        return "already"
    headers = {"User-Agent": UA}
    if existing:
        headers["Range"] = "bytes=%d-" % existing
    with client.stream("GET", url, headers=headers) as r:
        if r.status_code == 416:
            return "already"
        if r.status_code in (401, 403):
            raise RuntimeError(
                "Dataverse returned %s for file id %s. "
                "The file may require a request on https://doi.org/%s. "
                "Download video_2_1.zip and video_3_part1.mp4 by hand, then run extract_pmma.py."
                % (r.status_code, file_id, PMMA_DOI)
            )
        r.raise_for_status()
        mode = "ab" if (r.status_code == 206 and existing) else "wb"
        with dest.open(mode) as f:
            for chunk in r.iter_bytes(1024 * 1024):
                if chunk:
                    f.write(chunk)
    return "downloaded"


def fetch_pmma(out: Path, stash: Path, delay: float):
    import httpx
    import subprocess
    headers = {"User-Agent": UA, "Accept": "*/*"}
    stash.mkdir(parents=True, exist_ok=True)
    result = {"doi": PMMA_DOI, "files": {}, "extracted": None, "error": None}
    try:
        with httpx.Client(headers=headers, timeout=60.0, follow_redirects=True) as client:
            listing = list_dataverse_files(client)
        missing_names = [n for n in PMMA_NEEDED if n not in listing]
        if missing_names:
            result["error"] = "dataset listing is missing %s; see https://doi.org/%s" % (missing_names, PMMA_DOI)
            return result
        timeout = httpx.Timeout(60.0, read=600.0)
        with httpx.Client(headers=headers, timeout=timeout, follow_redirects=True) as big:
            for name in PMMA_NEEDED:
                meta = listing[name]
                dest = stash / name
                print("PMMA: fetching %s (id=%s, ~%s bytes) -> %s" % (name, meta["id"], meta.get("size") or "?", dest), flush=True)
                status = download_dataverse_file(big, int(meta["id"]), dest, expected_size=meta.get("size"))
                result["files"][name] = {"path": str(dest), "status": status, "size": dest.stat().st_size if dest.exists() else 0}
                time.sleep(delay)
    except Exception as e:
        result["error"] = str(e)
        return result
    extract = ROOT / "scripts" / "extract_pmma.py"
    v2 = stash / "video_2_1.zip"
    v3 = stash / "video_3_part1.mp4"
    if not extract.exists():
        result["error"] = "scripts/extract_pmma.py missing"
        return result
    cmd = [sys.executable, str(extract), "--frames", str(ROOT / "data" / "pmma_frames.jsonl"), "--out", str(out)]
    if v2.exists():
        cmd += ["--video2-zip", str(v2)]
    if v3.exists():
        cmd += ["--video3-mp4", str(v3)]
    try:
        subprocess.check_call(cmd)
        result["extracted"] = "ok"
    except subprocess.CalledProcessError as e:
        result["error"] = "extract_pmma failed (exit %s)" % e.returncode
    return result


def check_status(out: Path, rows, pmma_frames: Path):
    missing = [r["image_id"] for r in rows if not (out / (r["image_id"] + ".jpg")).exists()]
    pmma_missing = []
    pmma_n = 0
    if pmma_frames.exists():
        for line in pmma_frames.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            pmma_n += 1
            if not (out / (rec["image_id"] + ".jpg")).exists():
                pmma_missing.append(rec["image_id"])
    return {
        "web_needed": len(rows),
        "web_missing": len(missing),
        "web_present": len(rows) - len(missing),
        "pmma_needed": pmma_n,
        "pmma_missing": len(pmma_missing),
        "out": str(out),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default=str(ROOT / "data" / "candidates.csv"))
    ap.add_argument("--manifest", action="append", default=[], help="optional extra CSV with image_id + download_url and optional sha256_original")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=os.environ.get("WV_IMAGE_DIR", str(ROOT / "images")))
    ap.add_argument("--blur-dir", default=os.environ.get("WV_BLUR_DIR", str(ROOT / "images_blurred")))
    ap.add_argument("--from-dir", help="optional local thumbs cache (copy or --link) before downloading")
    ap.add_argument("--link", action="store_true", help="symlink instead of copying from --from-dir")
    ap.add_argument("--download", action="store_true", help="accepted for compatibility; downloading missing web images is the default")
    ap.add_argument("--source", help="only rows of this source (openimages/openverse/commons)")
    ap.add_argument("--limit", type=int, default=0, help="max NEW web downloads this run (0=all)")
    ap.add_argument("--delay", type=float, default=1.0, help="polite delay between requests (seconds)")
    ap.add_argument("--retries", type=int, default=4, help="retries on 429/5xx/network errors")
    ap.add_argument("--failures", default=str(ROOT / "var" / "download_failures.csv"))
    ap.add_argument("--strict-hash", action="store_true", help="do not keep images whose re-encoded sha256 != manifest hash")
    ap.add_argument("--skip-hash", action="store_true", help="do not compare sha256 even when a hash is present")
    ap.add_argument("--skip-blur", action="store_true", help="do not run auto_blur after downloads")
    ap.add_argument("--pmma", action="store_true", help="download PMMA source files from Borealis and extract frames")
    ap.add_argument("--pmma-dir", default=str(ROOT / "var" / "pmma"), help="where to store Borealis source files")
    ap.add_argument("--check", action="store_true", help="print how many images are missing and exit (2 if any web image is missing)")
    a = ap.parse_args(argv)

    try:
        import httpx
        from PIL import Image  # noqa: F401
    except ImportError:
        sys.exit("Need: pip install -r requirements.txt  (pillow httpx)")

    data_dir = Path(a.data_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    blur_dir = Path(a.blur_dir)
    skip = load_skip_ids(data_dir)
    manifests = [Path(p) for p in a.manifest]
    rows, skipped = load_web_rows(Path(a.candidates), manifests, skip)
    if a.source:
        rows = [r for r in rows if r.get("source") == a.source]

    pmma_frames = data_dir / "pmma_frames.jsonl"
    status = check_status(out, rows, pmma_frames)
    print(
        "web_downloadable=%s present=%s missing=%s skipped_hold_or_excluded=%s out=%s"
        % (status["web_needed"], status["web_present"], status["web_missing"], len(skipped), out),
        flush=True,
    )
    if status["pmma_needed"]:
        print(
            "pmma_frames=%s present=%s missing=%s"
            % (status["pmma_needed"], status["pmma_needed"] - status["pmma_missing"], status["pmma_missing"]),
            flush=True,
        )

    if a.check:
        print(pmma_instructions(), flush=True)
        if status["web_missing"]:
            print(
                "%s of %s web images are not on disk yet. Run: python scripts/fetch_images.py"
                % (status["web_missing"], status["web_needed"]),
                flush=True,
            )
            return 2
        print("All downloadable web images are present.", flush=True)
        if status["pmma_missing"]:
            print("%s PMMA frames are still missing. Run: python scripts/fetch_images.py --pmma" % status["pmma_missing"], flush=True)
        return 0

    stats = {
        "downloaded": 0, "skipped_existing": 0, "failed": 0,
        "sha256_mismatch_warnings_kept": 0, "copied": 0, "linked": 0,
    }
    failures = []
    new_ids = []

    if a.from_dir:
        thumbs = load_thumb_paths(Path(a.candidates))
        cstats = copy_from_dir(rows, Path(a.from_dir), out, a.link, thumbs)
        stats["copied"] += cstats["copied"]
        stats["linked"] += cstats["linked"]
        new_ids.extend(cstats["ids"])

    headers = {"User-Agent": UA, "Accept": "image/*,*/*;q=0.8"}
    n = 0
    with httpx.Client(headers=headers, timeout=60.0, follow_redirects=True) as client:
        for rec in rows:
            dest = out / (rec["image_id"] + ".jpg")
            expected = rec.get("sha256") or ""
            if dest.exists():
                if a.skip_hash or not expected or sha256_bytes(dest.read_bytes()) == expected:
                    stats["skipped_existing"] += 1
                    continue
            if a.limit and n >= a.limit:
                break
            n += 1
            url = rec["url"]
            try:
                raw, used = fetch_with_retries(client, url, a.retries, a.delay)
                processed = process_bytes(raw)
                got = sha256_bytes(processed)
                hash_ok = (not expected) or (got == expected)
                if not hash_ok and a.strict_hash:
                    failures.append({"image_id": rec["image_id"], "reason": "sha256_mismatch", "expected": expected, "got": got, "url": used, "source": rec.get("source", "")})
                    stats["failed"] += 1
                else:
                    dest.write_bytes(processed)
                    new_ids.append(rec["image_id"])
                    if not hash_ok and not a.skip_hash:
                        stats["sha256_mismatch_warnings_kept"] += 1
                        failures.append({"image_id": rec["image_id"], "reason": "sha256_mismatch_warning_kept", "expected": expected, "got": got, "url": used, "source": rec.get("source", "")})
                    stats["downloaded"] += 1
            except Exception as e:
                failures.append({"image_id": rec["image_id"], "reason": str(e)[:300], "expected": expected, "got": "", "url": url, "source": rec.get("source", "")})
                stats["failed"] += 1
            time.sleep(a.delay)

    if failures:
        fp = Path(a.failures)
        fp.parent.mkdir(parents=True, exist_ok=True)
        with fp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["image_id", "reason", "expected", "got", "url", "source"])
            w.writeheader()
            w.writerows(failures)
        stats["failures_file"] = str(fp)

    if not a.skip_blur:
        stats["blur"] = run_blur(new_ids, out, blur_dir)
    else:
        stats["blur"] = "skipped (--skip-blur)"

    print(pmma_instructions(), flush=True)
    if a.pmma:
        result = fetch_pmma(out, Path(a.pmma_dir), a.delay)
        stats["pmma"] = result
        if result.get("extracted") == "ok" and not a.skip_blur:
            pmma_ids = []
            if pmma_frames.exists():
                for line in pmma_frames.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        iid = json.loads(line)["image_id"]
                        if (out / (iid + ".jpg")).exists() and not (blur_dir / (iid + ".jpg")).exists():
                            pmma_ids.append(iid)
            if pmma_ids:
                stats["pmma_blur"] = run_blur(pmma_ids, out, blur_dir)
        if result.get("error"):
            print("PMMA error:", result["error"], flush=True)

    print(stats, flush=True)
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main() or 0)