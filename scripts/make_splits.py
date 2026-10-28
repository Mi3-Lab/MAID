#!/usr/bin/env python3
"""Build data/splits_v1.csv for the WheelView v1 baseline (image_id, split, label, creator).

Clean set: images marked _status=done by a human labeler (samir/shaaz) where no labeler who marked it done set a
non-empty `exclude`. Label = device_type (manual / powered / other_aid / lookalike) resolved with wvcore's
consensus/adjudication rules; images with an unresolved conflict, `unclear`, or no device class are dropped.
Images whose license audit status is needs_ross or exclude are dropped. face_blur_missed is ignored.
Events = union (by uuid) of the server DB (var/wheelview.db) and annotations/events/*.jsonl. Read-only: nothing
here writes to events, labels or blocklists.

Split: ~70/15/15 grouped by creator (Flickr owner id from the source URL when available, else creator name), so a
creator never spans two splits; stratified by class as well as a grouped split allows; fixed seed."""
import argparse, csv, glob, json, os, random, re, sqlite3, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server")); import wvcore

HUMANS = ("samir", "shaaz")
CLASSES = ["manual", "powered", "other_aid", "lookalike"]
BAD_LICENSE = ("needs_ross", "exclude")

def load_events(db, evdir):
    evs, seen = [], set()
    def add(raw):
        e, _ = wvcore.validate(raw)
        if e and e["annotator"] in HUMANS and e["uuid"] not in seen: seen.add(e["uuid"]); evs.append(e)
    n_db = n_js = 0
    if db and os.path.exists(db):
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True); c.row_factory = sqlite3.Row
        for r in c.execute("SELECT * FROM events ORDER BY seq"):
            raw = {k: r[k] for k in ("uuid", "image_id", "annotator", "field", "ts", "device_id", "kind")}
            raw["value"] = json.loads(r["value"]) if r["value"] is not None else None
            if r["prov"]: raw.update(json.loads(r["prov"]))
            before = len(evs); add(raw); n_db += len(evs) - before
        c.close()
    for fn in sorted(glob.glob(os.path.join(evdir, "*.jsonl"))):
        for line in open(fn, encoding="utf-8"):
            if line.strip():
                try: raw = json.loads(line)
                except json.JSONDecodeError: continue
                before = len(evs); add(raw); n_js += len(evs) - before
    return evs, n_db, n_js

def group_key(row):
    url = (row.get("source_url") or row.get("source_page_url") or "")
    m = re.search(r"flickr\.com/photos/([^/]+)/", url)
    if m: return "flickr:" + m.group(1).lower()
    cr = (row.get("creator") or "").strip().lower()
    return ("creator:" + cr) if cr else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=ROOT / "var" / "wheelview.db")
    ap.add_argument("--events", default=ROOT / "annotations" / "events")
    ap.add_argument("--license", default=None, help="default: data/license_audit_v2.csv if present else v1")
    ap.add_argument("--out", default=ROOT / "data" / "splits_v1.csv")
    ap.add_argument("--summary", default=ROOT / "results" / "splits_v1_summary.json")
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--fracs", default="0.70,0.15,0.15")
    a = ap.parse_args()
    lic_path = Path(a.license) if a.license else (ROOT / "data" / "license_audit_v2.csv" if (ROOT / "data" / "license_audit_v2.csv").exists() else ROOT / "data" / "license_audit_v1.csv")
    lic = {r["image_id"]: r for r in csv.DictReader(open(lic_path, encoding="utf-8"))}
    attr = {}
    ap_ = ROOT / "derived" / "attributions.csv"
    if ap_.exists(): attr = {r["image_id"]: r for r in csv.DictReader(open(ap_, encoding="utf-8"))}

    evs, n_db, n_js = load_events(str(a.db), str(a.events))
    per, adj = wvcore.fold(evs)
    drops = Counter(); rows = []
    for img in sorted(per):
        st = wvcore.image_status(img, per, adj)
        if not st["done"]: continue
        if any(per[img][ann].get("exclude") for ann in st["done"]): continue   # labeler-excluded: not in clean set
        drops["clean_set"] += 1
        val, how, _ = st["fields"].get("device_type", ("", "missing", {}))
        if how == "conflict": drops["drop_unresolved_device_conflict"] += 1; continue
        if val not in CLASSES: drops[f"drop_device_{val or 'missing'}"] += 1; continue
        L = lic.get(img)
        if L is None: drops["license_unchecked_kept"] += 1
        elif L.get("status") in BAD_LICENSE: drops[f"drop_license_{L['status']}"] += 1; continue
        src = L or attr.get(img) or {}
        g = group_key(src) or group_key(attr.get(img, {})) or ("image:" + img)
        creator = (src.get("creator") or attr.get(img, {}).get("creator") or "").strip()
        rows.append({"image_id": img, "label": val, "label_how": how, "group": g, "creator": creator,
                     "license_status": (L or {}).get("status", "unchecked")})

    fr = [float(x) for x in a.fracs.split(",")]; names = ["train", "val", "test"]
    groups = defaultdict(list)
    for r in rows: groups[r["group"]].append(r)
    tot = Counter(r["label"] for r in rows)
    target = {s: {c: tot[c] * f for c in CLASSES} for s, f in zip(names, fr)}

    def score(assign):
        cnt = {s: Counter() for s in names}
        for g, s in assign.items():
            for r in groups[g]: cnt[s][r["label"]] += 1
        sc = sum(((cnt[s][c] - target[s][c]) ** 2) / max(tot[c], 1) for s in names for c in CLASSES)
        sc += sum(((sum(cnt[s].values()) - len(rows) * f) ** 2) / len(rows) for s, f in zip(names, fr))
        sc += sum(50 for s in ("test", "val") for c in CLASSES if tot[c] >= 3 and cnt[s][c] == 0)
        return sc
    rng = random.Random(a.seed); best = None
    keys = sorted(groups)
    for trial in range(3000):
        order = keys[:]; rng.shuffle(order); order.sort(key=lambda g: -len(groups[g]) * (0.5 + rng.random()))
        assign, cnt = {}, {s: Counter() for s in names}
        for g in order:
            need = {}
            for s in names:
                gain = sum(min(target[s][c] - cnt[s][c], len([r for r in groups[g] if r["label"] == c])) for c in CLASSES)
                need[s] = gain + rng.random() * 0.5
            s = max(names, key=lambda s: need[s]); assign[g] = s
            for r in groups[g]: cnt[s][r["label"]] += 1
        sc = score(assign)
        if best is None or sc < best[0]: best = (sc, assign)
    assign = best[1]
    for r in rows: r["split"] = assign[r["group"]]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["image_id", "split", "label", "creator", "group", "label_how", "license_status"])
        for r in sorted(rows, key=lambda r: r["image_id"]):
            w.writerow([r["image_id"], r["split"], r["label"], r["creator"], r["group"], r["label_how"], r["license_status"]])
    counts = {s: {c: sum(1 for r in rows if r["split"] == s and r["label"] == c) for c in CLASSES} for s in names}
    summ = {"n": len(rows), "class_counts": dict(tot), "split_counts": counts,
            "split_sizes": {s: sum(counts[s].values()) for s in names}, "n_groups": len(groups),
            "largest_groups": sorted(((g, len(v)) for g, v in groups.items()), key=lambda x: -x[1])[:8],
            "seed": a.seed, "fracs": fr, "license_file": lic_path.name, "filters": dict(drops),
            "events_used": len(evs), "events_from_db": n_db, "events_only_in_jsonl": n_js,
            "label_resolution": dict(Counter(r["label_how"] for r in rows)),
            "license_status": dict(Counter(r["license_status"] for r in rows))}
    os.makedirs(os.path.dirname(a.summary), exist_ok=True)
    json.dump(summ, open(a.summary, "w", encoding="utf-8"), indent=2)
    print(json.dumps(summ, indent=2))

if __name__ == "__main__":
    main()
