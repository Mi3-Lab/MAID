"""Human review of the assistant visual review suggestions (var/assist/assistant_visual_review_attr-v1.jsonl).

Pure helpers; server/main.py wires them to /api/assistant_review/*. Rules:
- suggestions are never labels. A label is written only when a human reviewer accepts/edits a field, through the
  same insert path as POST /api/events, attributed to the reviewer (provenance source=auto_review).
- an existing human label (any annotator or an adjudication) is never overridden: matching values are only recorded,
  differing values are routed to the adjudication queue.
- reject writes only a review outcome (gitignored JSONL), never a label.
- child-flagged, safety-held and blocked images are never served.
"""
import hashlib, json, math, re, time
from pathlib import Path

SOURCE = "assistant_visual_review"
TAG = "assistant_visual_review_attr-v1"
HIGH_CONF = 0.7          # the assistant's highest confidence level; only these images are audit-sampled
AUDIT_FRAC = 0.05
ACTIONS = ("accept", "edit", "reject")
TIERS = ("disagreement", "low_confidence", "audit", "rest")
NON_HUMAN = re.compile(r"(grok|assistant|bot|auto|model|siglip|machine)", re.I)

def _h(s): return hashlib.sha1(s.encode()).hexdigest()

def _jsonl(path):
    p = Path(path)
    if not p.exists(): return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except ValueError: continue
        if isinstance(r, dict): out.append(r)
    return out

def load_suggestions(path):
    """{image_id: {field: {"value", "confidence"}}}"""
    out = {}
    for r in _jsonl(path):
        if isinstance(r.get("image_id"), str) and isinstance(r.get("suggestions"), dict):
            out[r["image_id"]] = {f: {"value": s.get("value"), "confidence": s.get("confidence")} for f, s in r["suggestions"].items()
                                  if isinstance(s, dict) and s.get("value") not in (None, "", [])}
    return out

def load_flags(path):
    """{image_id: [reasons]} from the assistant's flagged file."""
    out = {}
    for r in _jsonl(path):
        if isinstance(r.get("image_id"), str): out.setdefault(r["image_id"], []).append(str(r.get("reason", "")))
    return out

def child_ids(flags):
    return {i for i, rs in flags.items() if any("child" in r for r in rs)}

def load_outcomes(path):
    """Latest outcome per (image_id, field) (file order)."""
    out = {}
    for r in _jsonl(path):
        if isinstance(r.get("image_id"), str) and isinstance(r.get("field"), str) and r.get("action") in ACTIONS:
            out[(r["image_id"], r["field"])] = r
    return out

def append_jsonl(path, rec):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh: fh.write(json.dumps(rec, sort_keys=True) + "\n")

def human_values(img, field, per, adjv, hval):
    """{annotator: value} of existing human labels for one field (adjudication as 'adj')."""
    h = {a: hval(fv[field]) for a, fv in (per.get(img) or {}).items() if fv.get(field) not in (None, "", [])}
    a = adjv.get((img, field))
    if a and a[0] not in (None, "", []): h["adj"] = hval(a[0])
    return h

def build_items(sugg, per, adjv, other, excluded, hval):
    """Ordered review items. other: {img: {field: suggestion}} from other models (SigLIP etc.)."""
    items = []
    for img, fs in sugg.items():
        if img in excluded: continue
        fields = []
        for f, s in sorted(fs.items()):
            v = hval(s["value"]); h = human_values(img, f, per, adjv, hval)
            o = (other.get(img) or {}).get(f); ov = hval(o["value"]) if o and o.get("value") not in (None, "", []) else None
            fields.append({"field": f, "value": s["value"], "confidence": s["confidence"], "other": ov,
                           "other_tag": (o or {}).get("model_tag"), "human": h,
                           "disagree_human": any(x != v for x in h.values()), "disagree_model": ov is not None and ov != v})
        if not fields: continue
        nd = sum(x["disagree_human"] or x["disagree_model"] for x in fields)
        mc = min((x["confidence"] if isinstance(x["confidence"], (int, float)) else 0) for x in fields)
        nh = sum(x["disagree_human"] for x in fields)
        items.append({"image_id": img, "fields": fields, "n_disagree": nd, "n_disagree_human": nh, "min_conf": mc,
                      "has_human": any(x["human"] for x in fields),
                      "tier": "disagreement" if nd else "low_confidence" if mc < HIGH_CONF else "high"})
    # stratified (by assistant device_type) deterministic ~5% audit sample of the high-confidence, agreeing images
    strata = {}
    for it in items:
        if it["tier"] == "high":
            dt = next((x["value"] for x in it["fields"] if x["field"] == "device_type"), "none")
            strata.setdefault(str(dt), []).append(it)
    for _, g in sorted(strata.items()):
        g.sort(key=lambda it: _h("audit:" + it["image_id"]))
        for it in g[:max(1, math.ceil(AUDIT_FRAC * len(g)))]: it["tier"] = "audit"; it["audit"] = True
    for it in items:
        if it["tier"] == "high": it["tier"] = "rest"
        it.setdefault("audit", False)
    rank = {t: k for k, t in enumerate(TIERS)}
    items.sort(key=lambda it: (rank[it["tier"]], (-it["n_disagree_human"], -it["n_disagree"]) if it["tier"] == "disagreement" else (0, 0),
                               it["min_conf"] if it["tier"] in ("disagreement", "low_confidence") else 0, _h(it["image_id"])))
    for k, it in enumerate(items): it["rank"] = k + 1
    return items

def review_status(item, outcomes, auto_approved=None):
    """pending | partial | reviewed | auto_approved.

    auto_approved is a set/dict of image_ids from var/assist/auto_approved.jsonl (model consensus).
    Human review outcomes always win: once any field has an outcome, status is partial/reviewed
    and never auto_approved (so Accept/Edit/Reject override model consensus).
    """
    n = sum((item["image_id"], x["field"]) in outcomes for x in item["fields"])
    if n == 0:
        if auto_approved is not None and item["image_id"] in auto_approved:
            return "auto_approved"
        return "pending"
    return "reviewed" if n == len(item["fields"]) else "partial"

def summary(outcomes, items=None):
    """Accept/edit/reject rates per field (an estimate of the assistant's error rate), overall and on the audit sample."""
    def agg(rs):
        c = {a: 0 for a in ACTIONS}
        for r in rs: c[r["action"]] += 1
        n = sum(c.values())
        return {"n": n, **c, **({f"{a}_rate": round(c[a] / n, 4) for a in ACTIONS} if n else {}),
                "error_rate": round((c["edit"] + c["reject"]) / n, 4) if n else None,
                "routed_to_adjudication": sum(r.get("result") == "conflict_routed" for r in rs)}
    rs = list(outcomes.values()); fields = sorted({r["field"] for r in rs})
    out = {"overall": agg(rs), "per_field": {f: agg([r for r in rs if r["field"] == f]) for f in fields},
           "by_tier": {t: agg([r for r in rs if r.get("tier") == t]) for t in TIERS},
           "audit_sample": {"overall": agg([r for r in rs if r.get("audit")]),
                            "per_field": {f: agg([r for r in rs if r.get("audit") and r["field"] == f]) for f in fields},
                            "note": "unbiased-ish estimate for high-confidence suggestions; other tiers are deliberately biased toward likely errors"}}
    if items is not None:
        nf = sum(len(it["fields"]) for it in items)
        st = [review_status(it, outcomes) for it in items]
        out["progress"] = {"images": len(items), "reviewed": st.count("reviewed"), "partial": st.count("partial"),
                           "pending": st.count("pending"), "fields": nf,
                           "fields_reviewed": sum((it["image_id"], x["field"]) in outcomes for it in items for x in it["fields"])}
    return out

def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
