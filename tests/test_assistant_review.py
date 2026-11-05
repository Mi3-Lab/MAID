#!/usr/bin/env python3
"""Tests for the human review of assistant visual review suggestions (/api/assistant_review/*, review_assistant page).
Scratch DB / suggestion / outcome files in a temp dir; no live data.   python tests/test_assistant_review.py
"""
import csv, json, os, subprocess, sys, tempfile, uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="wv_arev_"))
sys.path.insert(0, str(ROOT / "server"))
results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond))); print(("PASS " if cond else "FAIL ") + name + (f" -- {detail}" if detail else ""), flush=True)

ids = [f"wv_{k:06d}" for k in range(1, 61)]
with open(TMP / "candidates.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, ["candidate_id", "source", "creator", "license", "needs_visual_screen"]); w.writeheader()
    for k, i in enumerate(ids): w.writerow({"candidate_id": i, "source": "commons", "creator": f"p{k}", "license": "CC-BY-4.0", "needs_visual_screen": "no"})
for d in ("auto", "assist", "block", "images", "images_blurred"): (TMP / d).mkdir()
(TMP / "blur.jsonl").write_text("")
(TMP / "block" / "removed_test.txt").write_text("wv_000004\n")
with open(TMP / "block" / "safety_hold.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["image_id", "flagged", "flagged_by", "reason"]); w.writerow(["wv_000005", "yes", "t", "hold"])
def sg(dt, c=0.7, env="indoor"):
    return {"device_type": {"value": dt, "confidence": c}, "environment": {"value": env, "confidence": c}}
S = {"wv_000001": sg("manual"),                    # human says powered -> disagreement
     "wv_000002": sg("powered", 0.4),               # low confidence
     "wv_000003": sg("manual", 0.7),                # child flagged -> hidden
     "wv_000004": sg("manual"), "wv_000005": sg("manual"),   # blocked / safety held -> hidden
     "wv_000006": sg("manual", 0.5)}
for i in ids[6:]: S[i] = sg("manual" if int(i[3:]) % 2 else "powered")
(TMP / "assist" / "assistant_visual_review_attr-v1.jsonl").write_text(
    "".join(json.dumps({"image_id": i, "model_tag": "assistant_visual_review_attr-v1", "source": "assistant_visual_review", "suggestions": s}) + "\n" for i, s in S.items()))
(TMP / "assist" / "assistant_visual_review_flagged.ndjson").write_text(json.dumps({"image_id": "wv_000003", "reason": "child_visible"}) + "\n")
(TMP / "auto" / "siglip-test_zs-v1.jsonl").write_text(json.dumps({"image_id": "wv_000006", "suggestions": {"device_type": {"value": "powered", "confidence": 0.9}}}) + "\n")
OUT = TMP / "assist" / "review_outcomes.jsonl"; Q = TMP / "queue.jsonl"
env = {"WV_DB": str(TMP / "scratch.db"), "WV_CANDIDATES": str(TMP / "candidates.csv"), "WV_CONTRIBUTED": str(TMP / "contributed.csv"),
       "WV_IMAGE_DIR": str(TMP / "images"), "WV_BLUR_DIR": str(TMP / "images_blurred"), "WV_EXPORT_DIR": str(TMP / "exports"),
       "WV_AUTO_DIR": str(TMP / "auto"), "WV_REVIEW_AUTO_DIR": str(TMP / "assist"), "WV_BLOCKLIST_DIR": str(TMP / "block"), "WV_IMAGE_PROXY": "0",
       "WV_ACCESS_CODE": "", "WV_BLUR_MANIFESTS": str(TMP / "blur.jsonl"), "WV_CACHE_DIR": str(TMP / "cache"),
       "WV_ASSIGN_WITHOUT_LOCAL_IMAGE": "1", "WV_ADJ_QUEUE": str(Q)}
for k in ("WV_HF_DATASET", "WV_ASSIST_REVIEW_FILE", "WV_ASSIST_REVIEW_FLAGS", "WV_REVIEW_OUTCOMES"): os.environ.pop(k, None)
os.environ.update(env)
from fastapi.testclient import TestClient
import main
def ev(img, ann, field, value):
    return {"uuid": str(uuid.uuid4()), "image_id": img, "annotator": ann, "field": field, "value": value, "kind": "label",
            "ts": "2026-11-05T10:00:00.000Z", "device_id": "t-" + ann}
def labels(cl, img):
    return [e for e in cl.get("/api/events", params={"image_id": img}).json()["events"] if e["kind"] == "label"]
with TestClient(main.app) as cl:
    cl.post("/api/events", json={"events": [ev("wv_000001", "alice", "device_type", "powered"), ev("wv_000009", "alice", "device_type", "manual")]})
    q = cl.get("/api/assistant_review/queue", params={"status": "all"}).json()
    got = [it["image_id"] for it in q["items"]]
    check("child-flagged, blocked and safety-held images never served", not {"wv_000003", "wv_000004", "wv_000005"} & set(got), str(got[:8]))
    check("all other suggestions queued", len(got) == 57 and q["progress"]["images"] == 57, str(len(got)))
    tiers = [it["tier"] for it in q["items"]]
    check("disagreements first (human + other model)", set(got[:2]) == {"wv_000001", "wv_000006"} and tiers[:2] == ["disagreement"] * 2, str(got[:3]))
    check("then lowest confidence", got[2] == "wv_000002" and tiers[2] == "low_confidence")
    aud = [it for it in q["items"] if it["audit"]]
    hi = {}
    for it in q["items"]:
        if it["tier"] in ("audit", "rest"): hi[it["fields"][0]["value"]] = hi.get(it["fields"][0]["value"], 0) + 1
    import math
    check("stratified ~5% audit sample of high-confidence (ceil(5%) per device_type stratum)",
          len(aud) == sum(math.ceil(0.05 * n) for n in hi.values()) and {it["fields"][0]["value"] for it in aud} == {"manual", "powered"} and all(it["min_conf"] >= 0.7 for it in aud))
    check("audit sample comes before the rest", tiers.index("rest") > max(k for k, t in enumerate(tiers) if t == "audit"))
    one = q["items"][0] if got[0] == "wv_000001" else q["items"][1]
    check("existing human labels shown side by side", next(x for x in one["fields"] if x["field"] == "device_type")["human"] == {"alice": "powered"})
    o6 = next(it for it in q["items"] if it["image_id"] == "wv_000006")
    check("other model value shown", next(x for x in o6["fields"] if x["field"] == "device_type")["other"] == "powered")

    for bad in ("grok-bot", "assistant", "Bob"):
        r = cl.post("/api/assistant_review/decide", json={"reviewer": bad, "image_id": "wv_000002", "decisions": [{"field": "device_type", "action": "accept"}]})
        check(f"reviewer {bad!r} refused", r.status_code == 400)
    check("hidden image cannot be reviewed", cl.post("/api/assistant_review/decide", json={"reviewer": "carol", "image_id": "wv_000003",
          "decisions": [{"field": "device_type", "action": "accept"}]}).status_code == 404)
    check("no labels before a human acts", labels(cl, "wv_000002") == [])
    r = cl.post("/api/assistant_review/decide", json={"reviewer": "carol", "image_id": "wv_000002", "decisions": [
        {"field": "device_type", "action": "accept"}, {"field": "environment", "action": "edit", "value": "curbside"}]}).json()
    L = {e["field"]: e for e in labels(cl, "wv_000002")}
    check("accept + edit write human labels attributed to the reviewer", L["device_type"]["annotator"] == "carol" and L["device_type"]["value"] == "powered"
          and L["environment"]["value"] == "curbside" and L["environment"]["annotator"] == "carol", json.dumps(r))
    check("labels carry review provenance", L["device_type"].get("source") == "auto_review" and L["device_type"].get("model_tag") == "assistant_visual_review_attr-v1"
          and L["environment"].get("suggested") == "indoor")
    r = cl.post("/api/assistant_review/decide", json={"reviewer": "carol", "image_id": "wv_000007", "decisions": [{"field": "device_type", "action": "reject"}]}).json()
    check("reject writes no label", labels(cl, "wv_000007") == [] and r["results"][0]["result"] == "rejected_no_label")
    r = cl.post("/api/assistant_review/decide", json={"reviewer": "carol", "image_id": "wv_000001", "decisions": [{"field": "device_type", "action": "accept"}]}).json()
    L1 = labels(cl, "wv_000001")
    check("conflict with existing human label: no label written, human label untouched", len(L1) == 1 and L1[0]["annotator"] == "alice"
          and L1[0]["value"] == "powered" and r["results"][0]["result"] == "conflict_routed")
    qq = [json.loads(l) for l in Q.read_text().splitlines()]
    check("conflict routed to adjudication queue", len(qq) == 1 and qq[0]["reason"] == "assistant_review_conflict" and qq[0]["image_id"] == "wv_000001")
    check("adjudication queue endpoint lists it", cl.get("/api/adjudication_queue").json()["counts"].get("assistant_review_conflict") == 1)
    r = cl.post("/api/assistant_review/decide", json={"reviewer": "dave", "image_id": "wv_000009", "decisions": [{"field": "device_type", "action": "accept"}]}).json()
    check("matching existing human label: recorded, no duplicate label", r["results"][0]["result"] == "matches_existing_human" and len(labels(cl, "wv_000009")) == 1)
    r = cl.post("/api/assistant_review/decide", json={"reviewer": "carol", "image_id": "wv_000008", "decisions": [{"field": "device_type", "action": "edit", "value": "spaceship"}]}).json()
    check("invalid edit value refused, nothing written", "error" in r["results"][0] and labels(cl, "wv_000008") == [])
    check("outcomes file holds every decision", len(OUT.read_text().splitlines()) == 5)

    sm = cl.get("/api/assistant_review/summary").json()
    dt = sm["per_field"]["device_type"]
    check("summary: per-field accept/edit/reject rates", dt["n"] == 4 and dt["accept"] == 3 and dt["edit"] == 0 and dt["reject"] == 1
          and dt["error_rate"] == 0.25 and sm["per_field"]["environment"]["edit"] == 1 and dt["routed_to_adjudication"] == 1, json.dumps(dt))
    check("summary: progress counter", sm["progress"]["reviewed"] == 1 and sm["progress"]["partial"] == 3 and sm["progress"]["images"] == 57, json.dumps(sm["progress"]))
    check("summary: audit-sample block present", "audit_sample" in sm and "per_field" in sm["audit_sample"])
    pend = cl.get("/api/assistant_review/queue", params={"status": "pending"}).json()
    check("queue status=pending excludes reviewed/partial", not {"wv_000001", "wv_000002", "wv_000007", "wv_000009"} & {it["image_id"] for it in pend["items"]})
    f = cl.get("/api/filter", params={"q": "review_status:reviewed", "limit": 1000}).json()
    check("filter review_status:reviewed", sorted(f["ids"]) == ["wv_000002"], json.dumps(f["ids"]))
    f = cl.get("/api/filter", params={"q": "review_status:pending", "limit": 1000}).json()
    check("filter review_status:pending", len(f["ids"]) == 53 and "wv_000008" in f["ids"] and "wv_000003" not in f["ids"], str(len(f["ids"])))
    f = cl.get("/api/filter", params={"q": "review_status:none", "limit": 1000}).json()
    check("filter review_status:none = no reviewable suggestion", "wv_000003" in f["ids"] and "wv_000002" not in f["ids"])
    check("filter keys list review_status", "review_status" in cl.get("/api/filter/keys").json()["pseudo"])
    h = cl.get("/review_assistant.html").text
    check("review page served", 'id="acceptAll"' in h and "review_assistant.js" in h and cl.get("/review_assistant.js").status_code == 200)
ign = all(subprocess.run(["git", "check-ignore", "-q", "--no-index", p], cwd=ROOT).returncode == 0
          for p in ("var/assist/review_outcomes.jsonl", "var/assist/assistant_visual_review_attr-v1.jsonl", "var/adjudication/queue.jsonl"))
check("outcomes, suggestions and queue are gitignored", ign)
n = sum(ok for _, ok in results)
print(f"\n{n}/{len(results)} passed; temp dir {TMP}")
sys.exit(0 if n == len(results) else 1)
