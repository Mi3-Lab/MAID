#!/usr/bin/env python3
"""Compare assistant visual-review device_type with SigLIP; auto-approve agreements.

  python scripts/auto_consensus.py              # dry-run summary
  python scripts/auto_consensus.py --apply       # write var/assist/auto_approved.jsonl + append adjudication queue

Never writes human label events. Threshold default 0.6 (SigLIP top-1); override with --threshold.
"""
import argparse, glob, importlib.util, json, os, sqlite3, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
import wvcore

def _load_ac():
    path = ROOT / "server" / "auto_consensus.py"
    spec = importlib.util.spec_from_file_location("wv_auto_consensus", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
AC = _load_ac()

HUMANS = ("samir", "shaaz", "devin", "wesley")

def load_events(db, evdir):
    evs, seen = [], set()
    def add(raw):
        e, _ = wvcore.validate(raw, allow_retired=True)
        if e and e["annotator"] in HUMANS and e["uuid"] not in seen:
            seen.add(e["uuid"]); evs.append(e)
    if db and os.path.exists(db):
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True); c.row_factory = sqlite3.Row
        for r in c.execute("SELECT * FROM events ORDER BY seq"):
            raw = {k: r[k] for k in ("uuid", "image_id", "annotator", "field", "ts", "device_id", "kind")}
            raw["value"] = json.loads(r["value"]) if r["value"] is not None else None
            if r["prov"]: raw.update(json.loads(r["prov"]))
            add(raw)
        c.close()
    for fn in sorted(glob.glob(os.path.join(str(evdir), "*.jsonl"))):
        for line in open(fn, encoding="utf-8"):
            if not line.strip(): continue
            try: raw = json.loads(line)
            except json.JSONDecodeError: continue
            add(raw)
    return evs

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assist", default=str(ROOT / "var" / "assist" / (AC.ASSIST_TAG + ".jsonl")))
    ap.add_argument("--flags", default=str(ROOT / "var" / "assist" / "assistant_visual_review_flagged.ndjson"))
    ap.add_argument("--siglip-dir", default=str(ROOT / "annotations" / "auto"))
    ap.add_argument("--siglip-tag", default=AC.SIGLIP_TAG)
    ap.add_argument("--child-list", default=str(ROOT / "data" / "removed_model_child_stroller.txt"))
    ap.add_argument("--license", default=None, help="default: data/license_audit_v2.csv if present else v1")
    ap.add_argument("--db", default=os.environ.get("WV_DB", str(ROOT / "var" / "wheelview.db")))
    ap.add_argument("--events", default=str(ROOT / "annotations" / "events"))
    ap.add_argument("--hold-dir", default=str(ROOT / "data"))
    ap.add_argument("--auto-out", default=str(ROOT / "var" / "assist" / "auto_approved.jsonl"))
    ap.add_argument("--queue", default=os.environ.get("WV_ADJ_QUEUE", str(ROOT / "var" / "adjudication" / "queue.jsonl")))
    ap.add_argument("--threshold", type=float, default=AC.DEFAULT_THRESHOLD)
    ap.add_argument("--apply", action="store_true", help="write outputs (default: dry-run)")
    ap.add_argument("--summary-out", default=None, help="optional JSON summary path")
    a = ap.parse_args()

    lic = a.license
    if not lic:
        v2 = ROOT / "data" / "license_audit_v2.csv"
        lic = str(v2 if v2.exists() else ROOT / "data" / "license_audit_v1.csv")

    assist = AC.load_assist(a.assist)
    siglip, sig_path = AC.load_siglip(a.siglip_dir, a.siglip_tag)
    flags = AC.load_flags(a.flags)
    child_ids = AC.load_id_list(a.child_list) | {i for i, rs in flags.items() if any("child" in r for r in rs)}
    hold = wvcore.load_safety_hold(a.hold_dir)
    license_status = AC.load_license_status(lic)
    evs = load_events(a.db, a.events)
    per, adjv = wvcore.fold(evs)

    auto_rows, adj_rows, notes, summary = AC.compare(
        assist, siglip, flags=flags, child_ids=child_ids, hold=hold, per=per, adjv=adjv,
        license_status=license_status, threshold=a.threshold,
    )
    summary["siglip_file"] = str(sig_path) if sig_path else None
    summary["license_file"] = lic
    summary["dry_run"] = not a.apply

    if a.apply:
        w = AC.apply(auto_rows, adj_rows, auto_path=a.auto_out, queue_path=a.queue, replace_auto=True)
        summary["write"] = w
    else:
        summary["write"] = {"auto_would_write": len(auto_rows), "adj_would_append_before_dedupe": len(adj_rows)}

    text = json.dumps(summary, indent=2)
    print(text)
    if a.summary_out:
        Path(a.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.summary_out).write_text(text, encoding="utf-8")
    sample = [n for n in notes if n.get("note") == "matches_existing_human"][:5]
    if sample:
        print("sample matches_existing_human:", json.dumps(sample, indent=2))

if __name__ == "__main__":
    main()
