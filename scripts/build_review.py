"""Build review/review_data.js for the offline review viewer (review/index.html).

Clean set = images marked done by samir or shaaz where no human labeler who marked it done set an exclude
(latest value per image/annotator/field). face_blur_missed flags are ignored (they predate the face blur).
Inputs (read-only): var/wheelview.db (default if present) or annotations/events/*.jsonl, data/license_audit_v2.csv,
data/v1_triage.csv, data/removed_*.txt (model reasons only), derived/attributions.csv, data/candidates.csv.
Output: review/review_data.js (window.WV_REVIEW = {...}); no images are written or referenced except by relative path.

  python scripts/build_review.py [--db var/wheelview.db | --events annotations/events | --jsonl dump.jsonl]
"""
import argparse, csv, json, os, re, sqlite3, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server")); sys.path.insert(0, str(ROOT / "scripts"))
import wvcore  # noqa: E402

HUMANS = ("samir", "shaaz")
FIELDS = ["device_type", "propulsion", "occupied", "setting", "viewpoint", "lighting", "occlusion"]

def canon_text(t):
    """Free-text warnings from older data files may name renamed classes (e.g. 'single label lookalike');
    show the canonical name (read-side only, the source files are not rewritten)."""
    for old, new in wvcore.VALUE_ALIASES.get("device_type", {}).items():
        t = re.sub(r"\b%s\b" % re.escape(old), new, t)
    return t

def events_from_db(path):
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True); c.row_factory = sqlite3.Row
    out = []
    for r in c.execute("select seq,uuid,image_id,annotator,field,value,ts,device_id,kind,prov from events order by seq"):
        e = dict(r); e["value"] = json.loads(e["value"]) if e["value"] is not None else None
        p = json.loads(e.pop("prov")) if e.get("prov") else None
        if p: e.update(p)
        out.append(e)
    return out

def events_from_jsonl(path):
    out = []
    for l in open(path, encoding="utf-8"):
        if not l.strip(): continue
        e = json.loads(l)
        if isinstance(e.get("value"), str) and e["value"][:1] in '"[{tf' and e["value"] not in ("", ):
            try: e["value"] = json.loads(e["value"])
            except json.JSONDecodeError: pass
        if isinstance(e.get("prov"), dict): e.update(e["prov"])
        out.append(e)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db"); ap.add_argument("--events"); ap.add_argument("--jsonl")
    ap.add_argument("--out", default=str(ROOT / "review" / "review_data.js"))
    a = ap.parse_args()
    if a.jsonl: evs, src = events_from_jsonl(a.jsonl), "events dump"
    elif a.events or not (a.db or (ROOT / "var" / "wheelview.db").exists()):
        import build_derived
        evs, _ = build_derived.load_events(a.events or str(ROOT / "annotations" / "events"), drop_blocked=False); src = "annotations/events"
    else: evs, src = events_from_db(a.db or str(ROOT / "var" / "wheelview.db")), "server DB"
    evs = [e for e in evs if e.get("annotator") in HUMANS or e.get("kind") == "adjudication"]
    per, adjv = wvcore.fold(evs)

    D = ROOT / "data"
    lic = {r["image_id"]: r for r in csv.DictReader(open(D / "license_audit_v2.csv", encoding="utf-8"))}
    hist = wvcore.exclude_history(evs, annotators=HUMANS)
    lic_status = {k: r.get("status") for k, r in lic.items()}
    clean = sorted(i for i in per if wvcore.clean_decision(i, per, adjv, annotators=HUMANS, history=hist,
                                                           license_status=lic_status,
                                                           hold=wvcore.load_safety_hold(str(D)))[0])   # shared clean-set rule
    tri = {r["image_id"]: r for r in csv.DictReader(open(D / "v1_triage.csv", encoding="utf-8"))}
    blk = wvcore.load_blocklist_entries(str(D))
    attr = {r["image_id"]: r for r in csv.DictReader(open(ROOT / "derived" / "attributions.csv", encoding="utf-8"))}
    cand = {r["candidate_id"]: r for r in csv.DictReader(open(D / "candidates.csv", encoding="utf-8"))}
    auto_review = {(e["image_id"], e["annotator"]) for e in evs if e.get("source") == "auto_review"}

    items = []
    for i in clean:
        st = wvcore.image_status(i, per, adjv)
        done = [h for h in HUMANS if h in st["done"]]
        L = lic.get(i, {}); A = attr.get(i) or {}; C = cand.get(i, {})
        warn = []
        t = tri.get(i)
        if t and t.get("model_warning"): warn += [w.strip() for w in t["model_warning"].split(";") if w.strip()]
        for fn, why in blk.get(i, []):
            if fn.startswith("removed_model_") and why and why not in warn: warn.append(why + " (model removal, restored by a human)")
        warn = [canon_text(w) for w in warn]
        kinds = sorted({w.split(":")[0].split(" ")[0] for w in warn})
        items.append({
            "id": i,
            "labeled_by": "double" if len(done) >= 2 else "single",
            "annotators": done,
            "status": st["status"],
            "consensus": {f: {"value": st["fields"][f][0], "how": st["fields"][f][1]} for f in FIELDS},
            "labels": {h: {f: per[i][h].get(f) for f in FIELDS} for h in done},
            "model_assisted": sorted(h for h in done if (i, h) in auto_review),
            "title": A.get("title") or C.get("title", ""),
            "creator": L.get("creator") or A.get("creator") or C.get("creator", ""),
            "license": L.get("license") or A.get("license") or C.get("license", ""),
            "license_url": L.get("license_url") or A.get("license_url") or C.get("license_url", ""),
            "source_site": L.get("source_site") or A.get("source", ""),
            "source_url": L.get("source_url") or A.get("source_url") or C.get("source_page_url", ""),
            "credit": A.get("required_credit_text", ""),
            "license_status": L.get("status") or "unchecked",
            "license_reason": L.get("reason", "" if L else "not in data/license_audit_v2.csv"),
            "live_check": L.get("live_check", ""),
            "license_checked": L.get("checked", ""),
            "warnings": warn, "warning_kinds": kinds,
        })
    summary = {
        "total": len(items),
        "labeled_by": dict(Counter(x["labeled_by"] for x in items)),
        "license_status": dict(Counter(x["license_status"] for x in items)),
        "device_type": dict(Counter(x["consensus"]["device_type"]["value"] or "(conflict/none)" for x in items)),
        "with_warnings": sum(1 for x in items if x["warnings"]),
    }
    data = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "events_source": src,
            "definition": "done by samir or shaaz; no labeler who marked it done excluded it, or the exclude was adjudicated to keep (never for child / identifiable_person; license_concern only if the license audit cleared it); face_blur_missed ignored",
            "fields": FIELDS, "summary": summary, "items": items}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("// Generated by scripts/build_review.py. Do not edit by hand. No images in here.\n")
        fh.write("window.WV_REVIEW = " + json.dumps(data, ensure_ascii=False, indent=0) + ";\n")
    print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
