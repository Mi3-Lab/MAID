#!/usr/bin/env python3
"""Fetch source-specific attribution metadata into data/attribution_cache.json (incremental; NETWORK).
Run by hand when candidates are added. CI never runs this; build_attributions.py reads only the cache.

  commons    -> Commons API imageinfo extmetadata (batches of 50 pageids, polite UA, backs off on 429)
  openverse  -> 'attribution' / creator_url from the original crawl dump (or Openverse API with --openverse-api)
  openimages -> Author / AuthorProfileURL / OriginalLandingURL from Open Images V7 image info CSV

Usage: python scripts/fetch_attribution_meta.py [--crawl-dir DIR] [--refresh] [--openverse-api]
"""
import argparse, csv, json, os, sys, time
from pathlib import Path
from urllib.parse import unquote
import urllib.request, urllib.parse, urllib.error

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "attribution_cache.json"
UA = "WheelViewDatasetBuilder/0.2 (UC Merced Mi3 Lab research dataset; https://github.com/samir-shah-ahmed; contact rossgreer@ucmerced.edu)"
API = "https://commons.wikimedia.org/w/api.php"
FIELDS = ["Attribution", "AttributionRequired", "Credit", "Artist", "LicenseShortName", "LicenseUrl", "License",
          "UsageTerms", "Restrictions", "Permission", "Copyrighted", "ObjectName"]

def load_cache():
    if CACHE.exists(): return json.load(open(CACHE, encoding="utf-8"))
    return {"_about": "attribution metadata cache; written by scripts/fetch_attribution_meta.py", "items": {}}

def save_cache(c):
    tmp = str(CACHE) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh: json.dump(c, fh, ensure_ascii=False, indent=0, sort_keys=True)
    os.replace(tmp, CACHE)

def post(data, tries=8):
    body = urllib.parse.urlencode(data).encode()
    wait = 5
    for _ in range(tries):
        req = urllib.request.Request(API, data=body, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=60) as r: return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                ra = int(e.headers.get("retry-after") or wait); print(f"HTTP {e.code}, sleeping {ra}s", flush=True)
                time.sleep(ra + 1); wait = min(wait * 2, 300); continue
            raise
        except Exception as e:
            print("ERR", e, flush=True); time.sleep(wait); wait = min(wait * 2, 300)
    raise SystemExit("giving up after repeated errors; rerun later (cache is incremental)")

def commons(cands, cache, refresh):
    todo = []
    for c in cands:
        if c["source"] != "commons": continue
        key = c["candidate_id"]
        if not refresh and cache["items"].get(key, {}).get("fetched"): continue
        todo.append(c)
    print("commons to fetch:", len(todo), flush=True)
    for i in range(0, len(todo), 50):
        batch = todo[i:i + 50]
        byid = {c["src_key"].split("_", 1)[1]: c for c in batch}
        j = post({"action": "query", "pageids": "|".join(byid), "prop": "imageinfo", "iiprop": "extmetadata",
                  "iiextmetadatafilter": "|".join(FIELDS), "iiextmetadatalanguage": "en",
                  "format": "json", "formatversion": 2, "maxlag": 5})
        if "error" in j and j["error"].get("code") == "maxlag":
            print("maxlag, sleeping 10s"); time.sleep(10); continue
        now = time.strftime("%Y-%m-%d")
        for p in j.get("query", {}).get("pages", []):
            c = byid.get(str(p.get("pageid")))
            if not c: continue
            em = ((p.get("imageinfo") or [{}])[0]).get("extmetadata", {})
            cache["items"][c["candidate_id"]] = {"source": "commons", "fetched": now, "title": p.get("title"),
                "missing": bool(p.get("missing")),
                "extmetadata": {k: (v.get("value") if isinstance(v, dict) else v) for k, v in em.items()}}
        save_cache(cache); print(f"commons {min(i + 50, len(todo))}/{len(todo)}", flush=True)
        time.sleep(1.5)

def openverse(cands, cache, crawl, use_api):
    raw = {}
    p = crawl / "raw_openverse.json"
    if p.exists():
        for r in json.load(open(p, encoding="utf-8")): raw[r["id"]] = r
    n = 0
    for c in cands:
        if c["source"] != "openverse": continue
        ovid = c["src_key"].split("_", 1)[1]
        r = raw.get(ovid)
        if r is None and use_api:
            req = urllib.request.Request(f"https://api.openverse.org/v1/images/{ovid}/", headers={"User-Agent": UA})
            try:
                with urllib.request.urlopen(req, timeout=30) as resp: r = json.load(resp)
            except Exception as e: print("openverse", ovid, e); r = None
            time.sleep(3)
        if r is None: continue
        cache["items"][c["candidate_id"]] = {"source": "openverse", "fetched": c.get("access_date") or "", "openverse_id": ovid,
            "attribution": r.get("attribution"), "creator": r.get("creator"), "creator_url": r.get("creator_url"),
            "license": r.get("license"), "license_version": r.get("license_version"), "license_url": r.get("license_url"),
            "provider": r.get("provider"), "foreign_landing_url": r.get("foreign_landing_url")}
        n += 1
    print("openverse cached:", n)

def openimages(cands, cache, crawl):
    p = crawl / "oi" / "info.csv"
    if not p.exists(): print("no", p); return
    info = {r["ImageID"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}
    n = 0
    for c in cands:
        if c["source"] != "openimages": continue
        r = info.get(c["src_key"].split("_", 1)[1])
        if not r: continue
        cache["items"][c["candidate_id"]] = {"source": "openimages", "fetched": c.get("access_date") or "",
            "oi_image_id": r["ImageID"], "subset": r["Subset"], "author": r["Author"], "author_profile_url": r["AuthorProfileURL"],
            "landing_url": r["OriginalLandingURL"], "license_url": r["License"], "title": r["Title"]}
        n += 1
    print("openimages cached:", n)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crawl-dir", default=os.environ.get("WV_CRAWL_DIR", "/workspace/wheelview"))
    ap.add_argument("--refresh", action="store_true"); ap.add_argument("--openverse-api", action="store_true")
    ap.add_argument("--skip-commons", action="store_true")
    a = ap.parse_args()
    cands = list(csv.DictReader(open(ROOT / "data" / "candidates.csv", encoding="utf-8")))
    cache = load_cache(); crawl = Path(a.crawl_dir)
    openverse(cands, cache, crawl, a.openverse_api); openimages(cands, cache, crawl); save_cache(cache)
    if not a.skip_commons: commons(cands, cache, a.refresh)
    save_cache(cache); print("cache items:", len(cache["items"]))
