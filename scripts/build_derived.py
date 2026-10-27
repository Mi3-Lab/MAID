#!/usr/bin/env python3
"""Rebuild derived/ (labels.csv, per_annotator_labels.csv, adjudication_needed.csv, kappa.json, progress.json)
from the append-only per-annotator event files annotations/events/*.jsonl. Never hand-edit derived/.
Machine suggestions (annotations/auto/*.jsonl) are NOT events: they never count as an annotation, never enter
labels/kappa, and are only compared against human labels in derived/auto_vs_human.json.
Run locally or by .github/workflows/rebuild-derived.yml on push."""
import argparse, glob, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server")); import wvcore

def load_events(evdir, reserved=(), drop_blocked=True):
    blocked = wvcore.load_blocklist(os.environ.get("WV_BLOCKLIST_DIR", ROOT / "data")) if drop_blocked else set()
    evs, seen, bad = [], set(), 0
    for fn in sorted(glob.glob(os.path.join(evdir, "*.jsonl"))):
        owner = Path(fn).stem
        for ln, line in enumerate(open(fn, encoding="utf-8"), 1):
            if not line.strip(): continue
            try: e, why = wvcore.validate(json.loads(line))
            except json.JSONDecodeError: e, why = None, "invalid json"
            if e and e["annotator"] != owner: e, why = None, f"annotator {e['annotator']} in {owner}.jsonl"
            if e and e["annotator"] in reserved: e, why = None, f"{e['annotator']} is a machine model_tag, not a human annotator"
            if not e: bad += 1; print(f"WARN {fn}:{ln}: {why}", file=sys.stderr); continue
            if e["uuid"] in seen or e["image_id"] in blocked: continue
            seen.add(e["uuid"]); evs.append(e)
    return evs, bad

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=ROOT / "annotations" / "events"); ap.add_argument("--out", default=ROOT / "derived")
    ap.add_argument("--candidates", default=ROOT / "data" / "candidates.csv"); ap.add_argument("--strict", action="store_true")
    ap.add_argument("--auto", default=ROOT / "annotations" / "auto", help="machine suggestion files (compared, never counted)")
    a = ap.parse_args()
    auto = wvcore.load_auto(a.auto)
    evs, bad = load_events(a.events, reserved=set(auto))
    stat, k, prog = wvcore.materialize(evs, wvcore.load_candidates(a.candidates), a.out, auto=auto, blocked=wvcore.load_blocklist(os.environ.get("WV_BLOCKLIST_DIR", ROOT / "data")))
    print(json.dumps({"events": len(evs), "invalid_lines": bad, "status_counts": prog["status_counts"]}))
    if a.strict and bad: sys.exit(1)
