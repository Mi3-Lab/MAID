"""Shared, dependency-free core: schema, event validation, latest-value folding, consensus, kappa.
Used by the server (live views) and scripts/build_derived.py (CI rebuild from annotations/events/*.jsonl)."""
import csv, json, os, re
from collections import Counter, defaultdict
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = json.load(open(os.path.join(HERE, "schema.json")))
FIELD_OPTS = {f["name"]: f["options"] for f in SCHEMA["fields"]}
COMPARE = [f["name"] for f in SCHEMA["fields"] if f.get("compare")] + ["exclude"]
ALL_FIELDS = list(FIELD_OPTS) + ["exclude", "face_blur_missed", "bbox", "_status"]
EVENT_KEYS = ["uuid", "image_id", "annotator", "field", "value", "ts", "device_id", "kind"]
UUID_RE = re.compile(r"^[0-9a-fA-F-]{8,64}$")
ID_RE = re.compile(r"^wv_\d{6}$")

def slug(name):
    s = re.sub(r"[^a-z0-9._-]+", "-", (name or "").strip().lower()).strip("-.")
    return s[:40]

def validate(ev):
    """Return (normalised_event, None) or (None, reason)."""
    try:
        e = {k: ev.get(k) for k in EVENT_KEYS}
    except AttributeError:
        return None, "not an object"
    e["kind"] = e["kind"] or "label"
    if not isinstance(e["uuid"], str) or not UUID_RE.match(e["uuid"]): return None, "bad uuid"
    if not isinstance(e["image_id"], str) or not ID_RE.match(e["image_id"]): return None, "bad image_id"
    if not e["annotator"] or slug(e["annotator"]) != e["annotator"]: return None, "annotator must be a slug"
    if e["kind"] not in ("label", "adjudication"): return None, "bad kind"
    if not isinstance(e["ts"], str) or len(e["ts"]) < 19: return None, "bad ts"
    if not isinstance(e["device_id"], str) or not e["device_id"]: return None, "bad device_id"
    f, v = e["field"], e["value"]
    if f in FIELD_OPTS:
        if v is not None and v not in FIELD_OPTS[f]: return None, f"bad value for {f}"
    elif f == "exclude":
        if v not in ("", None) and v not in SCHEMA["exclude_reasons"]: return None, "bad exclude reason"
    elif f == "face_blur_missed":
        if not isinstance(v, bool): return None, "face_blur_missed must be bool"
    elif f == "bbox":
        if not (isinstance(v, list) and all(isinstance(b, list) and len(b) == 4 and all(isinstance(x, (int, float)) and -0.01 <= x <= 1.01 for x in b) for b in v)):
            return None, "bad bbox"
    elif f == "_status":
        if v not in ("done", "todo"): return None, "bad _status"
    else:
        return None, "unknown field"
    if e["kind"] == "adjudication" and f not in COMPARE + ["face_blur_missed"]: return None, "field not adjudicable"
    return e, None

def fold(events):
    """Latest value per (image, annotator, field) for labels and per (image, field) for adjudications.
    Ordering: client ts, then uuid (deterministic tie-break). Nothing is ever overwritten across annotators."""
    lab, adj = {}, {}
    for e in events:
        key = (e["ts"], e["uuid"])
        if e["kind"] == "adjudication":
            k = (e["image_id"], e["field"])
            if k not in adj or key > adj[k][0]: adj[k] = (key, e["value"], e["annotator"])
        else:
            k = (e["image_id"], e["annotator"], e["field"])
            if k not in lab or key > lab[k][0]: lab[k] = (key, e["value"])
    per = defaultdict(lambda: defaultdict(dict))   # image -> annotator -> field -> value
    for (img, ann, f), (_, v) in lab.items(): per[img][ann][f] = v
    adjv = {k: (v, who) for k, (_, v, who) in adj.items()}
    return per, adjv

def norm(f, v):
    if f == "exclude": return v or "none"
    return v

def image_status(img, per, adjv):
    anns = per.get(img, {})
    done = sorted(a for a, fv in anns.items() if fv.get("_status") == "done")
    out = {"done": done, "fields": {}, "conflicts": []}
    for f in COMPARE:
        vals = {a: norm(f, anns[a].get(f)) for a in done if anns[a].get(f) is not None or f == "exclude"}
        if (img, f) in adjv:
            out["fields"][f] = (norm(f, adjv[(img, f)][0]), "adjudicated", vals); continue
        if len(vals) < 2:
            out["fields"][f] = (next(iter(vals.values())) if vals else "", "single" if vals else "missing", vals); continue
        c = Counter(vals.values()); top, n = c.most_common(1)[0]
        if n == len(vals): out["fields"][f] = (top, "consensus", vals)
        elif len(vals) >= 3 and n > len(vals) / 2: out["fields"][f] = (top, "majority", vals)
        else: out["fields"][f] = ("", "conflict", vals); out["conflicts"].append(f)
    ex = out["fields"].get("exclude")
    if ex and ex[1] in ("consensus", "majority", "adjudicated") and ex[0] != "none":
        # image agreed-excluded: disagreements on descriptive fields no longer matter
        for f in list(out["conflicts"]):
            if f != "exclude": out["fields"][f] = ("", "moot_excluded", out["fields"][f][2]); out["conflicts"].remove(f)
    fb = [bool(anns[a].get("face_blur_missed")) for a in done]
    out["face_blur_missed"] = adjv[(img, "face_blur_missed")][0] if (img, "face_blur_missed") in adjv else any(fb)
    out["bboxes"] = {a: anns[a].get("bbox") or [] for a in done}
    nd = len(done)
    if nd < SCHEMA["annotators_per_image"]: st = "in_progress" if anns else "unlabelled"
    elif out["conflicts"]: st = "needs_third_annotator" if nd < SCHEMA["max_annotators_per_image"] else "needs_adjudication"
    else: st = "final"
    out["status"] = st
    return out

def cohen(pairs):
    n = len(pairs)
    if n == 0: return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return 1.0 if pe == 1 else round((po - pe) / (1 - pe), 4)

def kappa(per):
    """Per field: pooled Cohen's kappa over the first two 'done' annotators of each image (alphabetical),
    plus pairwise Cohen's kappa for every annotator pair with >= 5 shared images."""
    res = {}
    for f in COMPARE:
        pooled, pair = [], defaultdict(list)
        for img, anns in per.items():
            done = sorted(a for a, fv in anns.items() if fv.get("_status") == "done")
            vals = {a: norm(f, anns[a].get(f)) for a in done if anns[a].get(f) is not None or f == "exclude"}
            ks = sorted(vals)
            if len(ks) >= 2: pooled.append((vals[ks[0]], vals[ks[1]]))
            for a, b in combinations(ks, 2): pair[(a, b)].append((vals[a], vals[b]))
        res[f] = {"pooled_kappa": cohen(pooled), "n_items": len(pooled),
                  "pairwise": {f"{a}|{b}": {"kappa": cohen(p), "n": len(p)} for (a, b), p in sorted(pair.items()) if len(p) >= 5}}
    return res

def load_candidates(path):
    with open(path, newline="") as fh:
        return {r["candidate_id"]: r for r in csv.DictReader(fh)}

def materialize(events, cands, outdir):
    """Write labels.csv, per_annotator_labels.csv, adjudication_needed.csv, kappa.json, progress.json."""
    os.makedirs(outdir, exist_ok=True)
    per, adjv = fold(events)
    imgs = sorted(set(per) | {k[0] for k in adjv})
    stat = {i: image_status(i, per, adjv) for i in imgs}
    with open(os.path.join(outdir, "labels.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "status", "n_done", "annotators"] + [x for f in COMPARE for x in (f, f + "_agreement")] +
                   ["face_blur_missed", "proposed_class", "source", "license", "source_page_url"])
        for i in imgs:
            s = stat[i]; c = cands.get(i, {})
            w.writerow([i, s["status"], len(s["done"]), ";".join(s["done"])] + [x for f in COMPARE for x in s["fields"][f][:2]] +
                       [s["face_blur_missed"], c.get("proposed_class", ""), c.get("source", ""), c.get("license", ""), c.get("source_page_url", "")])
    with open(os.path.join(outdir, "per_annotator_labels.csv"), "w", newline="") as fh:
        w = csv.writer(fh); fl = list(FIELD_OPTS) + ["exclude", "face_blur_missed", "bbox", "_status"]
        w.writerow(["image_id", "annotator"] + fl)
        for i in imgs:
            for a, fv in sorted(per.get(i, {}).items()):
                w.writerow([i, a] + [json.dumps(fv[f]) if f == "bbox" and f in fv else fv.get(f, "") for f in fl])
    with open(os.path.join(outdir, "adjudication_needed.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["image_id", "status", "field", "values_by_annotator"])
        for i in imgs:
            for f in stat[i]["conflicts"]:
                w.writerow([i, stat[i]["status"], f, json.dumps(stat[i]["fields"][f][2], sort_keys=True)])
    k = kappa(per)
    json.dump(k, open(os.path.join(outdir, "kappa.json"), "w"), indent=1, sort_keys=True)
    prog = progress(events, cands, stat)
    json.dump(prog, open(os.path.join(outdir, "progress.json"), "w"), indent=1, sort_keys=True)
    return stat, k, prog

def progress(events, cands, stat):
    by_ann = Counter(); done_by_ann = Counter()
    for e in events:
        if e["kind"] == "label": by_ann[e["annotator"]] += 1
    for i, s in stat.items():
        for a in s["done"]: done_by_ann[a] += 1
    return {"candidates": len(cands), "images_touched": len(stat),
            "status_counts": dict(Counter(s["status"] for s in stat.values())),
            "events": len(events), "events_by_annotator": dict(by_ann), "images_done_by_annotator": dict(done_by_ann)}
