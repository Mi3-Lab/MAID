"""Shared, dependency-free core: schema, event validation, latest-value folding, consensus, kappa.
Used by the server (live views) and scripts/build_derived.py (CI rebuild from annotations/events/*.jsonl)."""
import csv, json, os, re
from collections import Counter, defaultdict
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = json.load(open(os.path.join(HERE, "schema.json"), encoding="utf-8"))
FIELD_OPTS = {f["name"]: f["options"] for f in SCHEMA["fields"]}
COMPARE = [f["name"] for f in SCHEMA["fields"] if f.get("compare")] + ["exclude"]
# Renamed values: {field: {historical value: canonical value}} (schema.json "value_aliases"; e.g. device_type
# 'lookalike' -> 'ambiguous', renamed 2026-10-28). The event log is append-only and is NEVER rewritten: old events
# keep the value they were recorded with, and every reader maps it through canon() (fold() does this for all
# label/adjudication folding, load_auto() for machine suggestions). This is the ONE place the mapping lives.
VALUE_ALIASES = SCHEMA.get("value_aliases", {})
# Device classes used for classification (device_type minus 'unclear'), canonical names, fixed order.
DEVICE_CLASSES = [o for o in FIELD_OPTS["device_type"] if o != "unclear"]

def canon(field, value):
    """Canonical value for (field, value): maps renamed/historical option values (see VALUE_ALIASES)."""
    if isinstance(value, str):
        return VALUE_ALIASES.get(field, {}).get(value, value)
    return value

def canon_keys(obj, field="device_type"):
    """Copy of a JSON-like object with renamed option values mapped in dict keys, also as a '_<old>' key suffix
    (e.g. results JSON 'lookalike' -> 'ambiguous', 'test_f1_lookalike' -> 'test_f1_ambiguous'). For reading old results."""
    al = VALUE_ALIASES.get(field, {})
    def k2(k):
        if not isinstance(k, str): return k
        for old, new in al.items():
            if k == old: return new
            if k.endswith("_" + old): return k[: -len(old)] + new
        return k
    if isinstance(obj, dict): return {k2(k): canon_keys(v, field) for k, v in obj.items()}
    if isinstance(obj, list): return [canon_keys(v, field) for v in obj]
    return canon(field, obj) if isinstance(obj, str) else obj

def canon_class(value):
    """Canonical device class name (e.g. 'lookalike' -> 'ambiguous'); other values unchanged."""
    return canon("device_type", value)
ALL_FIELDS = list(FIELD_OPTS) + ["exclude", "face_blur_missed", "bbox", "_status"]
EVENT_KEYS = ["uuid", "image_id", "annotator", "field", "value", "ts", "device_id", "kind"]
# Optional provenance on HUMAN events made in review mode ("model-suggested, human-confirmed/corrected").
# source: "auto_review"; model_tag: which machine suggestion file; suggested: the model's value for this field (or null);
# confidence: the model's confidence for that value. Absent on ordinary events (old lines stay byte-identical).
PROV_KEYS = ["source", "model_tag", "suggested", "confidence"]
PROV_SOURCES = ("auto_review",)
TAG_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
UUID_RE = re.compile(r"^[0-9a-fA-F-]{8,64}$")
ID_RE = re.compile(r"^wv_\d{6}$")
REMOVAL_ACTIONS = ("restore", "confirm_removed")   # kind=removal events (never folded into labels)

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
    if e["kind"] not in ("label", "adjudication", "removal"): return None, "bad kind"
    if not isinstance(e["ts"], str) or len(e["ts"]) < 19: return None, "bad ts"
    if not isinstance(e["device_id"], str) or not e["device_id"]: return None, "bad device_id"
    f, v = e["field"], e["value"]
    if e["kind"] == "removal" or f == "removal":   # 'Removed' review: a human restores / confirms a blocklisted image
        if e["kind"] != "removal" or f != "removal": return None, "removal events need kind=removal and field=removal"
        if not (isinstance(v, dict) and set(v) <= {"action", "list", "reason"} and v.get("action") in REMOVAL_ACTIONS
                and all(isinstance(v.get(k, ""), str) and len(v.get(k, "")) <= 400 for k in ("list", "reason"))): return None, "bad removal value"
        if ev.get("source") is not None or ev.get("model_tag") is not None: return None, "provenance only on label events"
        return e, None
    if f in FIELD_OPTS:
        # historical aliases stay valid (offline clients may still queue them); stored as sent, mapped on read
        if v is not None and canon(f, v) not in FIELD_OPTS[f]: return None, f"bad value for {f}"
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
    if ev.get("source") is not None or ev.get("model_tag") is not None:
        if ev.get("source") not in PROV_SOURCES: return None, "bad source"
        if not isinstance(ev.get("model_tag"), str) or not TAG_RE.match(ev["model_tag"]): return None, "bad model_tag"
        sv, cf = ev.get("suggested"), ev.get("confidence")
        if sv is not None and not isinstance(sv, (str, bool)): return None, "bad suggested"
        if isinstance(sv, str) and len(sv) > 40: return None, "bad suggested"
        if cf is not None and (isinstance(cf, bool) or not isinstance(cf, (int, float)) or not 0 <= cf <= 1): return None, "bad confidence"
        if e["kind"] != "label": return None, "provenance only on label events"
        e.update({"source": ev["source"], "model_tag": ev["model_tag"], "suggested": sv, "confidence": cf})
    return e, None

def prov_of(e):
    """Provenance dict of an event (or None) — what the DB stores in its `prov` column."""
    return {k: e[k] for k in PROV_KEYS if k in e} if e.get("source") else None

def fold(events):
    """Latest value per (image, annotator, field) for labels and per (image, field) for adjudications.
    Ordering: client ts, then uuid (deterministic tie-break). Nothing is ever overwritten across annotators."""
    lab, adj = {}, {}
    for e in events:
        if e["kind"] == "removal": continue        # removal-review decisions are not labels (see removal_decisions)
        key = (e["ts"], e["uuid"]); v = canon(e["field"], e["value"])   # historical aliases -> canonical (read-side only)
        if e["kind"] == "adjudication":
            k = (e["image_id"], e["field"])
            if k not in adj or key > adj[k][0]: adj[k] = (key, v, e["annotator"])
        else:
            k = (e["image_id"], e["annotator"], e["field"])
            if k not in lab or key > lab[k][0]: lab[k] = (key, v)
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

# ---------------------------------------------------------------- clean-set rule (the ONE definition)
# Used by make_splits.collect (-> build_manifest), build_review and anything else that needs "the clean labeled set".
# An image is in the clean set when at least one annotator marked it done and no done annotator excluded it, OR when
# its exclude was adjudicated to keep ('' / none): the adjudication then overrides the labelers' excludes, except
# (safety guard) when any labeler ever gave child or identifiable_person (always out), or license_concern and the
# license audit has not cleared the image (status ok / ok_with_note). Images in data/safety_hold.csv are never in the
# clean set (e.g. a visual check found a person who may be a child; they wait for the normal removal review).
ALWAYS_OUT_REASONS = ("child", "identifiable_person")
LICENSE_CLEARED = ("ok", "ok_with_note")

def exclude_history(events, annotators=None):
    """{image: set of every non-empty exclude reason ever recorded} (label and adjudication events, any annotator
    unless `annotators` is given). Lets the safety guard see reasons a labeler later changed."""
    h = defaultdict(set)
    for e in events:
        if e.get("kind") not in ("label", "adjudication") or e.get("field") != "exclude": continue
        if annotators is not None and e.get("kind") == "label" and e.get("annotator") not in annotators: continue
        v = canon("exclude", e.get("value"))
        if isinstance(v, str) and v not in ("", "none"): h[e["image_id"]].add(v)
    return h

def load_safety_hold(datadir):
    """{image_id: reason} from data/safety_hold.csv (image_id, flagged, flagged_by, reason); missing file = no holds."""
    p = os.path.join(datadir, "safety_hold.csv")
    if not os.path.exists(p): return {}
    with open(p, encoding="utf-8", newline="") as fh:
        return {r["image_id"].strip(): r.get("reason", "") for r in csv.DictReader(fh) if r.get("image_id", "").strip()}

def clean_decision(img, per, adjv, annotators=None, history=None, license_status=None, hold=None):
    """(in_clean_set, why). `annotators` limits who counts (default: everyone in `per`); `history` is exclude_history();
    `license_status` maps image -> license-audit status (only consulted for the license_concern guard);
    `hold` is load_safety_hold() (always out)."""
    if hold and img in hold: return False, "safety_hold"
    anns = per.get(img, {})
    done = sorted(a for a, fv in anns.items() if fv.get("_status") == "done" and (annotators is None or a in annotators))
    if not done: return False, "not_done"
    reasons = {anns[a]["exclude"] for a in done if anns[a].get("exclude") not in (None, "", "none")}
    if not reasons: return True, "clean"
    adj = adjv.get((img, "exclude"))
    if adj is None or (adj[0] or "none") != "none": return False, "labeler_excluded"
    reasons |= (history or {}).get(img, set())
    hard = sorted(reasons & set(ALWAYS_OUT_REASONS))
    if hard: return False, "kept_out_" + "+".join(hard)
    if "license_concern" in reasons and (license_status or {}).get(img) not in LICENSE_CLEARED:
        return False, "kept_out_license_concern_not_cleared"
    return True, "adjudicated_keep"

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

BLOCKLIST_GLOB = "removed_*.txt"   # data/removed_*.txt: image ids removed entirely (never served, assigned or exported)

def load_blocklist_entries(datadir):
    """{image_id: [(file name, inline '# reason' comment), ...]} from data/removed_*.txt (one id per line)."""
    out = {}
    if not datadir or not os.path.isdir(datadir): return out
    for fn in sorted(os.listdir(datadir)):
        if fn.startswith("removed_") and fn.endswith(".txt"):
            for line in open(os.path.join(datadir, fn), encoding="utf-8"):
                t, _, c = line.partition("#"); t = t.strip()
                if ID_RE.match(t): out.setdefault(t, []).append((fn, c.strip()))
    return out

def load_blocklist(datadir):
    """Union of image ids in data/removed_*.txt (one id per line; '#' comments). Files are lists, images stay on disk.
    Use effective_blocklist() to honour human 'restore' decisions from the Removed review."""
    return set(load_blocklist_entries(datadir))

def removal_decisions(events):
    """Latest human decision per image from the Removed review (kind=removal events): {image_id: (action, annotator, ts)}."""
    best = {}
    for e in events:
        if e.get("kind") != "removal" or not isinstance(e.get("value"), dict): continue
        k = (e["ts"], e["uuid"])
        if e["image_id"] not in best or k > best[e["image_id"]][0]: best[e["image_id"]] = (k, e["value"].get("action"), e["annotator"])
    return {i: (a, who, k[0]) for i, (k, a, who) in best.items()}

def effective_blocklist(datadir, events):
    """Blocklist files minus images a human restored (latest removal decision == 'restore')."""
    return load_blocklist(datadir) - {i for i, d in removal_decisions(events).items() if d[0] == "restore"}

def load_candidates(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["candidate_id"]: r for r in csv.DictReader(fh)}

def materialize(events, cands, outdir, auto=None, blocked=None):
    """Blocked ids (data/removed_*.txt) are dropped from events, candidates and machine suggestions first. Write labels.csv, per_annotator_labels.csv, adjudication_needed.csv, kappa.json, progress.json
    (+ auto_vs_human.json when machine suggestions are given). Machine suggestions NEVER enter labels/kappa:
    they are not events, only humans' events are folded."""
    os.makedirs(outdir, exist_ok=True)
    if blocked:
        events = [e for e in events if e["image_id"] not in blocked]
        cands = {k: v for k, v in cands.items() if k not in blocked}
        if auto is not None: auto = {t: {i: r for i, r in recs.items() if i not in blocked} for t, recs in auto.items()}
    per, adjv = fold(events)
    imgs = sorted(set(per) | {k[0] for k in adjv})
    stat = {i: image_status(i, per, adjv) for i in imgs}
    with open(os.path.join(outdir, "labels.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "status", "n_done", "annotators"] + [x for f in COMPARE for x in (f, f + "_agreement")] +
                   ["face_blur_missed", "proposed_class", "source", "license", "source_page_url"])
        for i in imgs:
            s = stat[i]; c = cands.get(i, {})
            w.writerow([i, s["status"], len(s["done"]), ";".join(s["done"])] + [x for f in COMPARE for x in s["fields"][f][:2]] +
                       [s["face_blur_missed"], c.get("proposed_class", ""), c.get("source", ""), c.get("license", ""), c.get("source_page_url", "")])
    with open(os.path.join(outdir, "per_annotator_labels.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); fl = list(FIELD_OPTS) + ["exclude", "face_blur_missed", "bbox", "_status"]
        w.writerow(["image_id", "annotator"] + fl)
        for i in imgs:
            for a, fv in sorted(per.get(i, {}).items()):
                w.writerow([i, a] + [json.dumps(fv[f]) if f == "bbox" and f in fv else fv.get(f, "") for f in fl])
    with open(os.path.join(outdir, "adjudication_needed.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["image_id", "status", "field", "values_by_annotator"])
        for i in imgs:
            for f in stat[i]["conflicts"]:
                w.writerow([i, stat[i]["status"], f, json.dumps(stat[i]["fields"][f][2], sort_keys=True)])
    k = kappa(per)
    json.dump(k, open(os.path.join(outdir, "kappa.json"), "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    prog = progress(events, cands, stat)
    if auto is not None:
        json.dump(auto_agreement(per, auto, events, stat), open(os.path.join(outdir, "auto_vs_human.json"), "w", encoding="utf-8", newline="\n"),
                  indent=1, sort_keys=True)
    json.dump(prog, open(os.path.join(outdir, "progress.json"), "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    return stat, k, prog

def progress(events, cands, stat):
    by_ann = Counter(); done_by_ann = Counter(); rev_by_ann = Counter()
    for e in events:
        if e["kind"] == "label": by_ann[e["annotator"]] += 1
        if e.get("source") == "auto_review": rev_by_ann[e["annotator"]] += 1
    for i, s in stat.items():
        for a in s["done"]: done_by_ann[a] += 1
    return {"candidates": len(cands), "images_touched": len(stat),
            "status_counts": dict(Counter(s["status"] for s in stat.values())),
            "events": len(events), "events_by_annotator": dict(by_ann), "images_done_by_annotator": dict(done_by_ann),
            "auto_review_events_by_annotator": dict(rev_by_ann)}

# ---------- machine suggestions (annotations/auto/<model_tag>.jsonl) ----------
SCREEN_TO_HUMAN = {"child": "child", "clinical_setting": "clinical_setting", "not_a_photo": "not_a_photo"}

def load_auto(autodir):
    """{model_tag: {image_id: record}} from annotations/auto/*.jsonl (machine source; never events)."""
    out = {}
    if not autodir or not os.path.isdir(autodir): return out
    for fn in sorted(os.listdir(autodir)):
        if not fn.endswith(".jsonl"): continue
        tag = fn[:-6]; recs = {}
        for line in open(os.path.join(autodir, fn), encoding="utf-8"):
            if not line.strip(): continue
            try: r = json.loads(line)
            except json.JSONDecodeError: continue
            if isinstance(r, dict) and isinstance(r.get("image_id"), str) and isinstance(r.get("suggestions"), dict): recs[r["image_id"]] = canon_auto(r)
        out[tag] = recs
    return out

def canon_auto(r):
    """Map renamed option values inside one machine-suggestion record on read (the file itself is never rewritten):
    suggestions[field].value and the keys of suggestions[field].dist / .distribution / .probs."""
    for f, sg in (r.get("suggestions") or {}).items():
        if not isinstance(sg, dict) or f not in VALUE_ALIASES: continue
        if "value" in sg: sg["value"] = canon(f, sg["value"])
        for dk in ("dist", "distribution", "probs", "scores"):
            if isinstance(sg.get(dk), dict):
                d = {}
                for k, v in sg[dk].items():
                    ck = canon(f, k)
                    d[ck] = d[ck] + v if ck in d and isinstance(v, (int, float)) else v
                sg[dk] = d
    return r

def _agree(pairs):
    n = len(pairs)
    return {"n": n, "accuracy": round(sum(a == b for a, b in pairs) / n, 4) if n else None, "kappa": cohen(pairs),
            "confusion": {f"{h}|{m}": c for (h, m), c in sorted(Counter(pairs).items())}}

def auto_agreement(per, auto, events=None, stat=None):
    """Model-vs-human agreement per field, kept SEPARATE from human inter-annotator kappa.
    pairs = (human value, model value) over every 'done' human label that is not excluded by that human.
    `independent` uses only human labels made without seeing the suggestions (no auto_review provenance on that
    image), `reviewed` only those made in review mode (anchored, so expect higher agreement)."""
    reviewed = set()
    for e in events or []:
        if e.get("source") == "auto_review":
            reviewed.add((e["image_id"], e["annotator"], e.get("model_tag")))
    fl = {}
    for e in sorted(events or [], key=lambda e: (e["ts"], e["uuid"])):   # latest prov per (img, ann, field)
        if e.get("source") == "auto_review" and e["field"] in FIELD_OPTS: fl[(e["image_id"], e["annotator"], e["field"])] = e
    res = {}
    for tag, recs in sorted(auto.items()):
        r = {"images_with_suggestions": len(recs), "fields": {}, "screening": {}}
        for f in FIELD_OPTS:
            allp, ind, rev, fin = [], [], [], []
            for img, anns in per.items():
                if img not in recs or f not in recs[img]["suggestions"]: continue
                mv = recs[img]["suggestions"][f]["value"]
                for a, fv in anns.items():
                    if fv.get("_status") != "done" or fv.get("exclude") or fv.get(f) is None: continue
                    p = (fv[f], mv); allp.append(p)
                    (rev if (img, a, tag) in reviewed else ind).append(p)
                if stat and img in stat and stat[img]["fields"].get(f, ("", ""))[1] in ("consensus", "majority", "adjudicated"):
                    fin.append((stat[img]["fields"][f][0], mv))
            r["fields"][f] = {"all_done_human_labels": _agree(allp), "independent": _agree(ind), "reviewed": _agree(rev),
                              "vs_final_labels": _agree(fin)}
        for k in ("child", "clinical_setting", "not_a_photo", "visible_face"):
            tp = fp = fn = tn = 0
            for img, anns in per.items():
                if img not in recs: continue
                flag = k in (recs[img].get("flags") or [])
                for a, fv in anns.items():
                    if fv.get("_status") != "done": continue
                    h = bool(fv.get("face_blur_missed")) if k == "visible_face" else fv.get("exclude") == SCREEN_TO_HUMAN[k]
                    tp += flag and h; fp += flag and not h; fn += (not flag) and h; tn += (not flag) and not h
            r["screening"][k] = {"human_proxy": "face_blur_missed" if k == "visible_face" else f"exclude={SCREEN_TO_HUMAN[k]}",
                                 "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                                 "recall": round(tp / (tp + fn), 4) if tp + fn else None,
                                 "precision": round(tp / (tp + fp), 4) if tp + fp else None}
        acc = defaultdict(lambda: [0, 0])
        for (img, a, f), e in fl.items():
            if e.get("model_tag") != tag or e.get("suggested") is None: continue
            fv = per.get(img, {}).get(a, {})
            if fv.get("_status") != "done": continue
            acc[f][1] += 1; acc[f][0] += fv.get(f) == canon(f, e["suggested"])
        r["review_acceptance"] = {f: {"n": t, "accepted": k, "rate": round(k / t, 4) if t else None} for f, (k, t) in sorted(acc.items())}
        res[tag] = r
    return res
