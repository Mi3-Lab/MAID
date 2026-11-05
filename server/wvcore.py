"""Shared, dependency-free core: schema, event validation, latest-value folding, consensus, kappa.
Used by the server (live views) and scripts/build_derived.py (CI rebuild from annotations/events/*.jsonl)."""
import csv, json, os, re
from collections import Counter, defaultdict
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = json.load(open(os.path.join(HERE, "schema.json"), encoding="utf-8"))
FIELD_OPTS = {f["name"]: f["options"] for f in SCHEMA["fields"]}   # every field (v1.0 + v1.1): used for validation
# v1.0 fields (no "since") keep their exact v1.0 behaviour: status, consensus, kappa, labels.csv, per_annotator_labels.csv.
# v1.1 attribute fields ("since": "1.1") are resolved separately (image_status_v11, labels_v11.csv) so labelling them can
# never re-open, re-assign or change a v1.0 image or any v1.0 derived file.
V10_FIELDS = [f["name"] for f in SCHEMA["fields"] if not f.get("since")]
V11_FIELDS = [f["name"] for f in SCHEMA["fields"] if f.get("since") == "1.1"]
COMPARE = [f["name"] for f in SCHEMA["fields"] if f.get("compare") is True and not f.get("since")] + ["exclude"]
COMPARE_V11 = [f["name"] for f in SCHEMA["fields"] if f.get("since") == "1.1" and f.get("compare")]
# Renamed values: {field: {historical value: canonical value}} (schema.json "value_aliases"; e.g. device_type
# 'lookalike' -> 'ambiguous', renamed 2026-10-28). The event log is append-only and is NEVER rewritten: old events
# keep the value they were recorded with, and every reader maps it through canon() (fold() does this for all
# label/adjudication folding, load_auto() for machine suggestions). This is the ONE place the mapping lives.
VALUE_ALIASES = SCHEMA.get("value_aliases", {})
# Device classes used for classification (device_type minus 'unclear'), canonical names, fixed order.
DEVICE_CLASSES = [o for o in FIELD_OPTS["device_type"] if o != "unclear"]
# Multi-select fields ("multi": true, e.g. ramp_reasons): the event value is a JSON list of options (or null). Derived
# files / consensus compare the sorted values joined with '+' (hval), so ['a','b'] == ['b','a'].
MULTI_FIELDS = [f["name"] for f in SCHEMA["fields"] if f.get("multi")]
# Dependent fields ("show_when": {parent: [values]}, e.g. chair_subtype only for manual/powered, ramp_reasons only for
# ramp_deploy unsafe/not_needed). A value recorded while the parent did not apply is ignored by consensus (see dep_ok).
SHOW_WHEN = {f["name"]: f["show_when"] for f in SCHEMA["fields"] if f.get("show_when")}
OPTIONS_BY_PARENT = {f["name"]: f["options_by_parent"] for f in SCHEMA["fields"] if f.get("options_by_parent")}
MULTI_MAX = 12

def hval(v):
    """Hashable / CSV form of a label value: lists (multi-select) -> sorted values joined with '+'; others unchanged."""
    if isinstance(v, (list, tuple)): return "+".join(sorted(str(x) for x in v))
    return v

def dep_ok(field, values, ctx=None):
    """False when `field` depends on a parent whose (known) value says the field does not apply.
    values: this annotator's {field: value}; ctx: fallback {field: value} (e.g. the resolved v1.0 device_type).
    Unknown parent (no value anywhere) -> True (keep the value)."""
    sw = SHOW_WHEN.get(field)
    if not sw: return True
    for parent, allowed in sw.items():
        pv = values.get(parent)
        if pv is None and ctx: pv = ctx.get(parent)
        if pv is None or pv == "": continue
        if canon(parent, pv) not in allowed: return False
    return True

def options_for(field, parent_value=None):
    """Options offered for a field; dependent fields with options_by_parent narrow to the parent's list."""
    obp = OPTIONS_BY_PARENT.get(field)
    if obp and parent_value in obp: return list(obp[parent_value])
    return list(FIELD_OPTS.get(field, []))

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
ALL_FIELDS = list(FIELD_OPTS) + ["exclude", "face_blur_missed", "bbox", "_status", "_status_v11"]
EVENT_KEYS = ["uuid", "image_id", "annotator", "field", "value", "ts", "device_id", "kind"]
# Optional provenance on HUMAN events made in review mode ("model-suggested, human-confirmed/corrected").
# source: "auto_review"; model_tag: which machine suggestion file; suggested: the model's value for this field (or null);
# confidence: the model's confidence for that value. Absent on ordinary events (old lines stay byte-identical).
PROV_KEYS = ["source", "model_tag", "suggested", "confidence"]
PROV_SOURCES = ("auto_review", "auto_attr")   # auto_attr: v1.1 attribute pass, human clicked a field while its suggestion was shown
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
        if f in MULTI_FIELDS:
            if v is not None:
                if not isinstance(v, list) or not v or len(v) > MULTI_MAX or len(set(map(str, v))) != len(v): return None, f"bad value for {f}"
                if any(not isinstance(x, str) or canon(f, x) not in FIELD_OPTS[f] for x in v): return None, f"bad value for {f}"
                if "unknown" in v and len(v) > 1: return None, f"{f}: unknown cannot be combined"
        elif v is not None and (not isinstance(v, str) or canon(f, v) not in FIELD_OPTS[f]): return None, f"bad value for {f}"
    elif f == "exclude":
        if v not in ("", None) and v not in SCHEMA["exclude_reasons"]: return None, "bad exclude reason"
    elif f == "face_blur_missed":
        if not isinstance(v, bool): return None, "face_blur_missed must be bool"
    elif f == "bbox":
        if not (isinstance(v, list) and all(isinstance(b, list) and len(b) == 4 and all(isinstance(x, (int, float)) and -0.01 <= x <= 1.01 for x in b) for b in v)):
            return None, "bad bbox"
    elif f in ("_status", "_status_v11"):
        if v not in ("done", "todo"): return None, f"bad {f}"
    else:
        return None, "unknown field"
    if e["kind"] == "adjudication" and f not in COMPARE + COMPARE_V11 + ["face_blur_missed"]: return None, "field not adjudicable"
    if ev.get("source") is not None or ev.get("model_tag") is not None:
        if ev.get("source") not in PROV_SOURCES: return None, "bad source"
        if not isinstance(ev.get("model_tag"), str) or not TAG_RE.match(ev["model_tag"]): return None, "bad model_tag"
        sv, cf = ev.get("suggested"), ev.get("confidence")
        if isinstance(sv, list):   # multi-select suggestion (e.g. ramp_reasons): short strings only
            if f not in MULTI_FIELDS or len(sv) > MULTI_MAX or any(not isinstance(x, str) or len(x) > 40 for x in sv): return None, "bad suggested"
        elif sv is not None and not isinstance(sv, (str, bool)): return None, "bad suggested"
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
    for (img, ann, f), (_, v) in lab.items(): per[img][ann][f] = [canon(f, x) for x in v] if isinstance(v, list) and f in MULTI_FIELDS else v
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

def v11_values(fv, ctx=None):
    """One annotator's v1.1 values in comparable form: lists -> 'a+b' (hval); dependent fields whose parent says they
    do not apply (dep_ok) are dropped, so a stale ramp_reasons after switching ramp_deploy to safe never conflicts."""
    return {f: hval(fv.get(f)) for f in COMPARE_V11 if fv.get(f) is not None and dep_ok(f, fv, ctx)}

def image_status_v11(img, per, adjv, ctx=None):
    """v1.1 attribute-pass resolution, same consensus rules as image_status but over COMPARE_V11 and the annotators whose
    _status_v11 is 'done'. Independent of the v1.0 status (an image can be v1.0-final and v1.1-unlabelled).
    ctx: optional fallback parent values (e.g. {"device_type": resolved v1.0 value}) for dependent fields."""
    anns = per.get(img, {})
    done = sorted(a for a, fv in anns.items() if fv.get("_status_v11") == "done")
    out = {"done": done, "fields": {}, "conflicts": []}
    cmp = {a: v11_values(anns[a], ctx) for a in done}
    for f in COMPARE_V11:
        vals = {a: cmp[a][f] for a in done if f in cmp[a]}
        if (img, f) in adjv:
            out["fields"][f] = (hval(adjv[(img, f)][0]), "adjudicated", vals); continue
        if len(vals) < 2:
            out["fields"][f] = (next(iter(vals.values())) if vals else "", "single" if vals else "missing", vals); continue
        c = Counter(vals.values()); top, n = c.most_common(1)[0]
        if n == len(vals): out["fields"][f] = (top, "consensus", vals)
        elif len(vals) >= 3 and n > len(vals) / 2: out["fields"][f] = (top, "majority", vals)
        else: out["fields"][f] = ("", "conflict", vals); out["conflicts"].append(f)
    nd = len(done); base = SCHEMA.get("annotators_per_image_v11", SCHEMA["annotators_per_image"])
    touched = any(f in anns[a] for a in anns for f in COMPARE_V11 + ["_status_v11"])
    if nd < base: st = "in_progress" if touched else "unlabelled"
    elif out["conflicts"]: st = "needs_third_annotator" if nd < SCHEMA["max_annotators_per_image"] else "needs_adjudication"
    else: st = "final"
    out["status"] = st
    return out

def has_v11(events):
    return any(e.get("field") in COMPARE_V11 + ["_status_v11"] for e in events)

def kappa_fields(per, fields, status_key):
    """kappa() for an arbitrary field list / done flag (used for v1.1 fields)."""
    res = {}
    for f in fields:
        pooled, pair = [], defaultdict(list)
        for img, anns in per.items():
            done = sorted(a for a, fv in anns.items() if fv.get(status_key) == "done")
            vals = {a: hval(anns[a].get(f)) for a in done if anns[a].get(f) is not None and dep_ok(f, anns[a])}
            ks = sorted(vals)
            if len(ks) >= 2: pooled.append((vals[ks[0]], vals[ks[1]]))
            for a, b in combinations(ks, 2): pair[(a, b)].append((vals[a], vals[b]))
        res[f] = {"pooled_kappa": cohen(pooled), "n_items": len(pooled),
                  "pairwise": {f"{a}|{b}": {"kappa": cohen(p), "n": len(p)} for (a, b), p in sorted(pair.items()) if len(p) >= 5}}
    return res

ATTR_TAG_MARK = "_attr-"   # annotations/auto/<model>_attr-vN.jsonl: v1.1 attribute suggestions (never the review-mode default)
def is_attr_tag(tag): return ATTR_TAG_MARK in (tag or "")

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

CONTRIB_ID_MIN = 900001   # contributed (own) images: wv_900001 ... (separate range, never reused by the web candidates)
def load_contributed(path):
    """data/contributed.csv (scripts/import_contributed.py) -> candidate-like rows keyed by image_id. Missing file = {}."""
    if not path or not os.path.exists(path): return {}
    with open(path, newline="", encoding="utf-8") as fh:
        out = {}
        for r in csv.DictReader(fh):
            if not ID_RE.match(r.get("image_id", "")) or r.get("status", "active") != "active": continue
            out[r["image_id"]] = r | {"candidate_id": r["image_id"], "source": "contributed", "proposed_class": r.get("proposed_class", ""),
                                      "needs_visual_screen": "yes" if r.get("identifiable_people") == "yes" else "no",
                                      "screen_reasons": "contributed: identifiable people, consent " + r.get("consent", "") if r.get("identifiable_people") == "yes" else ""}
        return out

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
        w = csv.writer(fh); fl = list(V10_FIELDS) + ["exclude", "face_blur_missed", "bbox", "_status"]
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
        a10 = {t: r for t, r in auto.items() if not is_attr_tag(t)}
        json.dump(auto_agreement(per, a10, events, stat), open(os.path.join(outdir, "auto_vs_human.json"), "w", encoding="utf-8", newline="\n"),
                  indent=1, sort_keys=True)
    if has_v11(events):   # v1.1 outputs only once v1.1 labelling has started (v1.0 derived files stay byte-identical)
        materialize_v11(events, per, adjv, cands, outdir, auto)
    json.dump(prog, open(os.path.join(outdir, "progress.json"), "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    return stat, k, prog

def materialize_v11(events, per, adjv, cands, outdir, auto=None):
    """derived/labels_v11.csv, per_annotator_labels_v11.csv, adjudication_needed_v11.csv, kappa_v11.json (+ auto_vs_human_v11.json)."""
    imgs = sorted(i for i in set(per) | {k[0] for k in adjv} if any(f in fv for fv in per.get(i, {}).values() for f in COMPARE_V11 + ["_status_v11"]))
    st = {i: image_status_v11(i, per, adjv) for i in imgs}
    with open(os.path.join(outdir, "labels_v11.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "status_v11", "n_done_v11", "annotators_v11"] + [x for f in COMPARE_V11 for x in (f, f + "_agreement")] + ["source"])
        for i in imgs:
            s = st[i]
            w.writerow([i, s["status"], len(s["done"]), ";".join(s["done"])] + [x for f in COMPARE_V11 for x in s["fields"][f][:2]] + [cands.get(i, {}).get("source", "")])
    with open(os.path.join(outdir, "per_annotator_labels_v11.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); fl = list(V11_FIELDS) + ["_status_v11"]; w.writerow(["image_id", "annotator"] + fl)
        for i in imgs:
            for a, fv in sorted(per.get(i, {}).items()):
                if any(f in fv for f in fl): w.writerow([i, a] + [hval(fv.get(f, "")) for f in fl])
    with open(os.path.join(outdir, "adjudication_needed_v11.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["image_id", "status_v11", "field", "values_by_annotator"])
        for i in imgs:
            for f in st[i]["conflicts"]: w.writerow([i, st[i]["status"], f, json.dumps(st[i]["fields"][f][2], sort_keys=True)])
    json.dump(kappa_fields(per, COMPARE_V11, "_status_v11"), open(os.path.join(outdir, "kappa_v11.json"), "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    if auto:
        a11 = {t: r for t, r in auto.items() if is_attr_tag(t)}
        if a11:
            res = {}
            for tag, recs in sorted(a11.items()):
                r = {"images_with_suggestions": len(recs), "fields": {}}
                for f in V11_FIELDS:
                    allp, fin = [], []
                    for img, anns in per.items():
                        sg = (recs.get(img) or {}).get("suggestions", {}).get(f)
                        if not sg: continue
                        for a, fv in anns.items():
                            if fv.get("_status_v11") == "done" and fv.get(f) is not None: allp.append((hval(fv[f]), hval(sg.get("value"))))
                        if img in st and st[img]["fields"][f][1] in ("consensus", "majority", "adjudicated"): fin.append((st[img]["fields"][f][0], hval(sg.get("value"))))
                    r["fields"][f] = {"all_done_human_labels": _agree(allp), "vs_final_labels": _agree(fin)}
                res[tag] = r
            json.dump(res, open(os.path.join(outdir, "auto_vs_human_v11.json"), "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    return st

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
        if e.get("source") == "auto_review" and e["field"] in V10_FIELDS: fl[(e["image_id"], e["annotator"], e["field"])] = e
    res = {}
    for tag, recs in sorted(auto.items()):
        r = {"images_with_suggestions": len(recs), "fields": {}, "screening": {}}
        for f in V10_FIELDS:
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

# ---------------------------------------------------------------- filtering (the ONE definition)
# Used by the server (/api/filter), scripts/filter_images.py and (as a cached server answer) the app's filter panel.
# A filter is {key: set(values)}: OR within a key, AND across keys. Keys are label fields (v1.0 + v1.1) or the
# pseudo keys below. Text form (URL hash / CLI): "device_type:manual,powered;weather:rain;synthetic:no" (+ ";basis:model").
# Per-field values come from HUMAN labels by default (basis=human): the resolved value when the image is final for that
# pass, else every annotator's latest value. Two special values per field:
#   _unlabeled  - no human value for the field
#   _model_only - no human value, but a machine suggestion exists (suggestions are never labels)
# basis=model matches the machine suggestion instead; basis=either matches human OR model values.
UNLABELED, MODEL_ONLY = "_unlabeled", "_model_only"
FILTER_BASES = ("human", "model", "either")
PSEUDO_KEYS = {
    "split": ["train", "val", "test", "none"],
    "approval": ["final", "in_progress", "needs_third_annotator", "needs_adjudication", "unlabelled"],
    "status_v11": ["final", "in_progress", "needs_third_annotator", "needs_adjudication", "unlabelled"],
    "synthetic": ["yes", "no"],
    "size_basis": ["measured", "estimate_only", "none"],
    "label_source": ["human", "model_only", "unlabeled"],
    "excluded": ["yes", "no"],
    "review_status": ["pending", "partial", "reviewed", "none"],   # human review of the assistant visual review suggestions
}
FILTER_FIELDS = list(FIELD_OPTS)
FILTER_KEYS = FILTER_FIELDS + list(PSEUDO_KEYS)
_FTOK = re.compile(r"^[A-Za-z0-9_.+-]{1,60}$")

def filter_values(key):
    """Selectable values of a filter key (special values included for label fields)."""
    if key in PSEUDO_KEYS: return list(PSEUDO_KEYS[key])
    return list(FIELD_OPTS[key]) + [UNLABELED, MODEL_ONLY]

def parse_filter(text):
    """'f:a,b;g:c;basis:model' -> ({f: {a, b}, g: {c}}, basis). Raises ValueError on unknown keys/values."""
    flt, basis = {}, "human"
    for part in (text or "").replace("&", ";").split(";"):
        part = part.strip()
        if not part: continue
        if ":" not in part: raise ValueError(f"bad filter term {part!r} (want key:value[,value])")
        k, vs = part.split(":", 1); k = k.strip()
        vals = [v.strip() for v in vs.split(",") if v.strip()]
        if k == "basis":
            if len(vals) != 1 or vals[0] not in FILTER_BASES: raise ValueError(f"basis must be one of {FILTER_BASES}")
            basis = vals[0]; continue
        if k not in FILTER_KEYS: raise ValueError(f"unknown filter key {k!r}")
        ok = set(filter_values(k))
        for v in vals:
            if not _FTOK.match(v): raise ValueError(f"bad value {v!r}")
            if canon(k, v) not in ok: raise ValueError(f"unknown value {v!r} for {k}")
        if vals: flt.setdefault(k, set()).update(canon(k, v) for v in vals)
    return flt, basis

def format_filter(flt, basis="human"):
    parts = [f"{k}:{','.join(sorted(flt[k]))}" for k in FILTER_KEYS if flt.get(k)]
    if basis != "human": parts.append(f"basis:{basis}")
    return ";".join(parts)

def _vals(f, v):
    """Value set of one label value: lists (and 'a+b' strings of multi fields) -> their members; canonical names."""
    if v is None or v == "" or v == []: return set()
    if isinstance(v, (list, tuple)): return {canon(f, str(x)) for x in v}
    if f in MULTI_FIELDS and isinstance(v, str): return {canon(f, x) for x in v.split("+") if x}
    return {canon(f, str(v)) if not isinstance(v, bool) else str(v).lower()}

def merged_suggestions(img, auto, prefer=None):
    """{field: suggestion dict (+ 'model_tag')} for one image over every machine-suggestion file. For each field the
    first tag (in `prefer` order; default: v1.0 tags then attr tags, by name) that has a value wins."""
    out = {}
    tags = list(prefer) if prefer else sorted(auto, key=lambda t: (is_attr_tag(t), t))
    for tag in tags:
        r = (auto.get(tag) or {}).get(img)
        if not r: continue
        for f, sg in (r.get("suggestions") or {}).items():
            if f in FIELD_OPTS and f not in out and isinstance(sg, dict) and sg.get("value") not in (None, "", []):
                out[f] = {**sg, "model_tag": tag}
    return out

def image_facts(img, per, adjv, sugg=None, extra=None, st10=None, st11=None):
    """Per-image facts for filtering: {"human": {field: set}, "model": {field: set}, "pseudo": {key: value}}.
    sugg: merged_suggestions(); extra: load_filter_context() result; st10/st11: precomputed statuses (optional)."""
    extra = extra or {}
    anns = per.get(img, {})
    st10 = st10 or image_status(img, per, adjv); st11 = st11 or image_status_v11(img, per, adjv)
    human = {}
    for f in FILTER_FIELDS:
        st = st10 if f in COMPARE else st11 if f in COMPARE_V11 else None
        res = st["fields"].get(f) if st else None
        if res and st["status"] == "final" and res[1] in ("consensus", "majority", "adjudicated", "single") and res[0] != "":
            vs = _vals(f, res[0])
        else:
            vs = set()
            for a, fv in anns.items():
                if fv.get(f) is not None and dep_ok(f, fv): vs |= _vals(f, fv[f])
            if (img, f) in adjv and adjv[(img, f)][0] is not None: vs = _vals(f, adjv[(img, f)][0])
        if vs: human[f] = vs
    model = {f: _vals(f, sg.get("value")) for f, sg in (sugg or {}).items()}
    model = {f: v for f, v in model.items() if v}
    syn = img in extra.get("synthetic", {})
    split = extra.get("split", {}).get(img) or (extra["synthetic"][img].get("split") if syn else "") or "none"
    size = "measured" if img in extra.get("measured", set()) else "estimate_only" if img in extra.get("estimated", set()) else "none"
    any_h = any(f in fv for fv in anns.values() for f in FILTER_FIELDS + ["_status", "_status_v11"])
    ex = st10["fields"].get("exclude", ("", ""))
    excluded = "yes" if (ex[1] in ("consensus", "majority", "adjudicated", "single") and ex[0] not in ("", "none")) or \
        any(fv.get("exclude") for fv in anns.values()) else "no"
    pseudo = {"split": split if split in PSEUDO_KEYS["split"] else "none", "approval": st10["status"], "status_v11": st11["status"],
              "synthetic": "yes" if syn else "no", "size_basis": size,
              "label_source": "human" if any_h else "model_only" if model else "unlabeled", "excluded": excluded,
              "review_status": extra.get("review_status", {}).get(img, "none")}
    return {"human": human, "model": model, "pseudo": pseudo}

def field_values(facts, f, basis="human"):
    """The value set a field contributes under a basis (with _unlabeled / _model_only)."""
    h, m = facts["human"].get(f, set()), facts["model"].get(f, set())
    if basis == "model": return set(m) or {UNLABELED}
    vs = set(h) | (set(m) if basis == "either" else set())
    if not h: vs |= {UNLABELED} | ({MODEL_ONLY} if m else set())
    return vs

def match(facts, flt, basis="human"):
    """OR within a key, AND across keys."""
    for k, want in flt.items():
        if not want: continue
        have = {facts["pseudo"].get(k)} if k in PSEUDO_KEYS else field_values(facts, k, basis)
        if not (have & want): return False
    return True

def facet_counts(facts_list, basis="human", keys=None):
    """{key: {value: n_images}} over a list of facts (a multi-valued image counts once per value)."""
    out = {}
    for k in keys or FILTER_KEYS:
        c = Counter()
        for fa in facts_list:
            for v in ({fa["pseudo"].get(k)} if k in PSEUDO_KEYS else field_values(fa, k, basis)): c[v] += 1
        out[k] = dict(sorted(c.items()))
    return out

def run_filter(ids, facts, flt, basis="human", counts=True):
    """-> {"total": n, "count": k, "ids": [...matching, in input order], "counts": facet counts over ALL ids,
    "match_counts": facet counts over the matches}."""
    m = [i for i in ids if match(facts[i], flt, basis)]
    out = {"total": len(ids), "count": len(m), "ids": m, "filter": format_filter(flt, basis)}
    if counts:
        out["counts"] = facet_counts([facts[i] for i in ids], basis)
        out["match_counts"] = facet_counts([facts[i] for i in m], basis)
    return out

def load_filter_context(manifest_v11=None, manifest_v2=None, synthetic=None, measurements=None, size_estimates=None):
    """Read-only side files for the pseudo keys. Missing files are fine (keys then read 'none'/'no')."""
    def rows(p):
        if not p or not os.path.exists(p): return []
        with open(p, newline="", encoding="utf-8") as fh: return list(csv.DictReader(fh))
    split = {}
    for p in (manifest_v2, manifest_v11):   # v1.1 manifest wins
        for r in rows(p):
            if r.get("image_id") and r.get("split"): split[r["image_id"]] = r["split"]
    syn = {r["image_id"]: r for r in rows(synthetic) if r.get("image_id")}
    measured = {r["image_id"] for r in rows(measurements) if r.get("image_id") and (r.get("measurement_source") or "none") != "none"}
    estimated = {r["image_id"] for r in rows(size_estimates) if r.get("image_id")}
    return {"split": split, "synthetic": syn, "measured": measured, "estimated": estimated}
