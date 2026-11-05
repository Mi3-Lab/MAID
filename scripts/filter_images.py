"""Filter MAID images by any labelled field (same logic as the app's filter panel and the server's /api/filter).

  python scripts/filter_images.py --filter "device_type:manual;chair_subtype:_unlabeled"            # ids, one per line
  python scripts/filter_images.py --filter "weather:rain,snow;ground_wet:yes" --format csv --out var/rain.csv
  python scripts/filter_images.py --filter "ramp_deploy:unsafe;ramp_reasons:obstruction" --format manifest --out var/subset.csv
  python scripts/filter_images.py --filter "environment:curbside;basis:model" --counts              # per-value counts (JSON)
  python scripts/filter_images.py --keys                                                             # every key and value

Filter text: key:value[,value];key:value...  OR within a key, AND across keys. Keys: every label field in
server/schema.json plus split, approval, status_v11, synthetic, size_basis, label_source, excluded. Special field
values: _unlabeled (no human value) and _model_only (no human value, but a machine suggestion exists).
basis:human (default) | model | either picks whether field values come from human labels or machine suggestions.

Read-only: the DB is opened mode=ro, annotations/events and annotations/auto are only read, nothing is written except
--out. Synthetic images (var/synthetic/synthetic_manifest.csv) are included only with --include-synthetic.
"""
import argparse, csv, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server")); sys.path.insert(0, str(ROOT / "scripts"))
import wvcore
from size_estimate import load_events

def build(a):
    evs = load_events(a.db, a.events)
    per, adjv = wvcore.fold(evs)
    cands = wvcore.load_candidates(a.candidates) if os.path.exists(a.candidates) else {}
    if a.contributed and os.path.exists(a.contributed): cands.update(wvcore.load_contributed(a.contributed))
    auto = wvcore.load_auto(a.auto_dir); auto.update({k: v for k, v in wvcore.load_auto(str(ROOT / "var" / "assist")).items() if k not in auto})
    ctx = wvcore.load_filter_context(a.manifest_v11, a.manifest_v2, a.synthetic, a.measurements, a.size_estimates)
    blocked = wvcore.effective_blocklist(a.blocklist_dir, evs) if a.blocklist_dir and os.path.isdir(a.blocklist_dir) else set()
    ids = sorted(i for i in cands if i not in blocked)
    if a.include_synthetic: ids += sorted(i for i in ctx["synthetic"] if i not in cands)
    prefer = sorted(auto, key=lambda t: (wvcore.is_attr_tag(t), t)) if not a.model_tag else [a.model_tag]
    facts = {i: wvcore.image_facts(i, per, adjv, wvcore.merged_suggestions(i, auto, prefer), ctx) for i in ids}
    return ids, facts, cands, ctx

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filter", default="", help="e.g. 'device_type:manual,powered;weather:rain'")
    ap.add_argument("--format", choices=("ids", "csv", "manifest", "json"), default="ids")
    ap.add_argument("--counts", action="store_true", help="print per-value counts (all images and matches) as JSON")
    ap.add_argument("--keys", action="store_true", help="list filter keys and values, then exit")
    ap.add_argument("--out", help="write here instead of stdout")
    ap.add_argument("--db", default=os.environ.get("WV_DB", str(ROOT / "var" / "wheelview.db")))
    ap.add_argument("--events", default=str(ROOT / "annotations" / "events"))
    ap.add_argument("--candidates", default=os.environ.get("WV_CANDIDATES", str(ROOT / "data" / "candidates.csv")))
    ap.add_argument("--contributed", default=os.environ.get("WV_CONTRIBUTED", str(ROOT / "data" / "contributed.csv")))
    ap.add_argument("--auto-dir", default=os.environ.get("WV_AUTO_DIR", str(ROOT / "annotations" / "auto")))
    ap.add_argument("--model-tag", help="use only this machine-suggestion tag for model values")
    ap.add_argument("--blocklist-dir", default=os.environ.get("WV_BLOCKLIST_DIR", str(ROOT / "data")))
    ap.add_argument("--manifest-v11", default=os.environ.get("WV_MANIFEST_V11", str(ROOT / "data" / "manifest_v11.csv")))
    ap.add_argument("--manifest-v2", default=os.environ.get("WV_MANIFEST_V2", str(ROOT / "data" / "manifest_v2.csv")))
    ap.add_argument("--synthetic", default=os.environ.get("WV_SYNTHETIC_MANIFEST", str(ROOT / "var" / "synthetic" / "synthetic_manifest.csv")))
    ap.add_argument("--include-synthetic", action="store_true")
    ap.add_argument("--measurements", default=os.environ.get("WV_MEASUREMENTS", str(ROOT / "data" / "measurements.csv")))
    ap.add_argument("--size-estimates", default=os.environ.get("WV_SIZE_ESTIMATES", str(ROOT / "var" / "v11" / "size_estimates.csv")))
    a = ap.parse_args(argv)
    if a.keys:
        print(json.dumps({k: wvcore.filter_values(k) for k in wvcore.FILTER_KEYS}, indent=1)); return 0
    try: flt, basis = wvcore.parse_filter(a.filter)
    except ValueError as e: ap.error(str(e))
    out_p = Path(a.out).resolve() if a.out else None
    if out_p and any(str(out_p).startswith(str((ROOT / d).resolve())) for d in ("derived", "release", "results", "annotations")):
        ap.error("refusing to write into derived/, release/, results/ or annotations/")
    ids, facts, cands, ctx = build(a)
    res = wvcore.run_filter(ids, facts, flt, basis, counts=a.counts or a.format == "json")
    fh = open(out_p, "w", newline="", encoding="utf-8") if out_p else sys.stdout
    try:
        if a.counts or a.format == "json":
            json.dump({k: res[k] for k in ("filter", "total", "count", "counts", "match_counts")} | ({"ids": res["ids"]} if a.format == "json" else {}), fh, indent=1)
            fh.write("\n")
        elif a.format == "ids":
            fh.writelines(i + "\n" for i in res["ids"])
        elif a.format == "csv":
            w = csv.writer(fh); keys = wvcore.FILTER_FIELDS
            w.writerow(["image_id"] + keys + [k + "_model" for k in keys] + list(wvcore.PSEUDO_KEYS))
            for i in res["ids"]:
                f = facts[i]
                w.writerow([i] + ["|".join(sorted(f["human"].get(k, ()))) for k in keys] + ["|".join(sorted(f["model"].get(k, ()))) for k in keys]
                           + [f["pseudo"][k] for k in wvcore.PSEUDO_KEYS])
        else:   # manifest: the matching rows of the v1.1 manifest (else v2), synthetic rows from the synthetic manifest
            src = a.manifest_v11 if os.path.exists(a.manifest_v11) else a.manifest_v2
            if not os.path.exists(src): ap.error("no manifest found (data/manifest_v11.csv or data/manifest_v2.csv); use --format csv")
            with open(src, newline="", encoding="utf-8") as mf: rows = list(csv.DictReader(mf))
            want = set(res["ids"]); cols = list(rows[0].keys()) if rows else ["image_id"]
            if a.include_synthetic and ctx["synthetic"]:
                cols += [c for c in next(iter(ctx["synthetic"].values())) if c not in cols]
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader()
            for r in rows:
                if r.get("image_id") in want: w.writerow(r)
            for i, r in ctx["synthetic"].items():
                if a.include_synthetic and i in want: w.writerow(r)
    finally:
        if out_p: fh.close()
    print(f"{res['count']} of {res['total']} images match {res['filter'] or '(no filter)'}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    sys.exit(main())
