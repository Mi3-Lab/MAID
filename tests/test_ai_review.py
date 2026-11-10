#!/usr/bin/env python3
"""Tests for majority-vote verification and the read-only /ai review API."""
import csv, json, os, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="wv_ai_rev_"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "server"))

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f" -- {detail}" if detail else ""), flush=True)

from ai_majority import majority_outcome, ai_consensus_value, decide_verdict

# --- majority rule ---
m = majority_outcome([("samir", "manual")], "manual")
check("ai+human same value is majority_pass", m["passes"] and m["basis"] == "ai+human" and m["majority_value"] == "manual", str(m))

m = majority_outcome([("samir", "manual")], "powered")
check("ai vs single human is a tie", m["outcome"] == "tie" and not m["passes"], str(m))

m = majority_outcome([("samir", "manual"), ("shaaz", "manual")], "powered")
check("human+human majority beats AI", m["passes"] and m["basis"] == "human+human" and m["majority_value"] == "manual", str(m))

m = majority_outcome([("samir", "manual"), ("shaaz", "manual")], "manual")
check("unanimous humans+AI is human+human+ai", m["passes"] and m["basis"] == "human+human+ai", str(m))

m = majority_outcome([("samir", "manual"), ("shaaz", "powered")], "manual")
check("AI breaks a human split: ai+human majority", m["passes"] and m["basis"] == "ai+human" and m["majority_value"] == "manual", str(m))

m = majority_outcome([("samir", "manual"), ("shaaz", "powered")], None)
check("two humans disagree, AI abstains: tie", m["outcome"] == "tie" and not m["passes"], str(m))

m = majority_outcome([], "manual")
check("AI-only is not a majority (need two voters)", m["outcome"] == "no_majority" and not m["passes"], str(m))

m = majority_outcome([("samir", "manual"), ("shaaz", "powered"), ("devin", "powered")], "manual")
check("2 humans vs (1 human + AI) is a 2-2 tie", m["outcome"] == "tie" and not m["passes"], str(m))

check("3-model consensus requires agreement and threshold",
      ai_consensus_value("manual", 0.9, "manual", 0.8, "manual", 0.7, 0.6, ["manual", "powered"]) == "manual"
      and ai_consensus_value("manual", 0.9, "powered", 0.8, "manual", 0.7, 0.6, ["manual", "powered"]) is None
      and ai_consensus_value("manual", 0.5, "manual", 0.9, "manual", 0.9, 0.6, ["manual", "powered"]) is None)

v, a, _ = decide_verdict(ai_label="manual", ai_vote="manual", models_complete=True,
                         human_votes=[("samir", "manual")],
                         majority=majority_outcome([("samir", "manual")], "manual"))
check("verdict: ai+human -> majority_pass keep", v == "majority_pass" and a == "keep", f"{v} {a}")

v, a, _ = decide_verdict(ai_label="powered", ai_vote="powered", models_complete=True,
                         human_votes=[("samir", "manual")],
                         majority=majority_outcome([("samir", "manual")], "powered"))
check("verdict: tie -> send to human review, not drop", v == "tie" and a == "send to human review", f"{v} {a}")

v, a, _ = decide_verdict(ai_label="powered", ai_vote="powered", models_complete=True,
                         human_votes=[("samir", "manual"), ("shaaz", "manual")],
                         majority=majority_outcome([("samir", "manual"), ("shaaz", "manual")], "powered"))
check("verdict: human+human against AI label -> drop from split", v == "majority_pass" and a == "drop from split", f"{v} {a}")

v, a, _ = decide_verdict(ai_label="manual", ai_vote="manual", models_complete=True,
                         human_votes=[], majority=majority_outcome([], "manual"))
check("verdict: AI-only confirmed keep", v == "confirmed" and a == "keep", f"{v} {a}")

v, a, _ = decide_verdict(ai_label="manual", ai_vote=None, models_complete=True,
                         human_votes=[], majority=majority_outcome([], None))
check("verdict: AI-only disputed", v == "model_disputed" and a == "send to human review", f"{v} {a}")

# --- server API (scratch env, no live DB / no events written) ---
(TMP / "derived").mkdir()
(TMP / "var" / "ai_jobs").mkdir(parents=True)
(TMP / "app").mkdir()
# minimal shell files so /ai can be served if we copy html; we only test APIs here
(TMP / "app" / "ai.html").write_text("<!doctype html><title>AI review</title><h1>AI review</h1>", encoding="utf-8")
(TMP / "app" / "index.html").write_text("<!doctype html><title>x</title>", encoding="utf-8")
(TMP / "app" / "app.js").write_text("", encoding="utf-8")
(TMP / "app" / "sw.js").write_text("", encoding="utf-8")
(TMP / "app" / "style.css").write_text("", encoding="utf-8")
(TMP / "app" / "unlock.html").write_text("<!doctype html><title>u</title>", encoding="utf-8")
(TMP / "data").mkdir()
(TMP / "data" / "candidates.csv").write_text("candidate_id,source,creator,license,needs_visual_screen\nwv_000001,commons,p,CC-BY-4.0,no\n", encoding="utf-8")
for d in ("auto", "block", "images", "images_blurred", "exports", "cache"):
    (TMP / d).mkdir(exist_ok=True)
(TMP / "blur.jsonl").write_text("")

# progress + a tiny verification csv
job = {"job": "verify_ai_labels", "stage": "score", "status": "running", "n_done": 3, "n_total": 10,
       "started": "2026-11-10T02:50:00-08:00", "last_update": "2026-11-10T02:51:00-08:00", "message": "scoring"}
(TMP / "var" / "ai_jobs" / "current.json").write_text(json.dumps(job), encoding="utf-8")
(TMP / "var" / "ai_jobs" / "log.txt").write_text("[t] starting\n[t] scoring\n", encoding="utf-8")
ver_cols = ["image_id","field","value","source","source_file","verdict","recommended_action","reason",
            "human_value_folded","human_votes","ai_vote","voters","majority_outcome","majority_value",
            "majority_basis","majority_agreeing","n_agree","n_votes","n_human_voters",
            "model_a_value","model_a_confidence","model_b_value","model_b_confidence",
            "model_c_value","model_c_confidence","dino_oof","threshold","in_training_split","split","run_timestamp"]
rows = [
    {"image_id":"wv_000001","field":"device_type","value":"manual","source":"auto_approved","source_file":"var/assist/auto_approved.jsonl",
     "verdict":"majority_pass","recommended_action":"keep","reason":"majority ai+human agrees with the AI label",
     "human_value_folded":"manual","human_votes":json.dumps([{"id":"samir","value":"manual"}]),
     "ai_vote":"manual","voters":json.dumps([{"id":"samir","type":"human","value":"manual"},{"id":"ai","type":"ai","value":"manual"}]),
     "majority_outcome":"majority_pass","majority_value":"manual","majority_basis":"ai+human","majority_agreeing":"samir,ai",
     "n_agree":"2","n_votes":"2","n_human_voters":"1","model_a_value":"manual","model_a_confidence":"0.9",
     "model_b_value":"manual","model_b_confidence":"0.8","model_c_value":"manual","model_c_confidence":"0.7",
     "dino_oof":"True","threshold":"0.6","in_training_split":"True","split":"train","run_timestamp":"2026-11-10T02:51:00-08:00"},
    {"image_id":"wv_000002","field":"device_type","value":"powered","source":"3model","source_file":"derived/ai_labels_3model.csv",
     "verdict":"tie","recommended_action":"send to human review","reason":"tie",
     "human_value_folded":"manual","human_votes":json.dumps([{"id":"shaaz","value":"manual"}]),
     "ai_vote":"powered","voters":json.dumps([{"id":"shaaz","type":"human","value":"manual"},{"id":"ai","type":"ai","value":"powered"}]),
     "majority_outcome":"tie","majority_value":"","majority_basis":"","majority_agreeing":"",
     "n_agree":"1","n_votes":"2","n_human_voters":"1","model_a_value":"powered","model_a_confidence":"0.9",
     "model_b_value":"powered","model_b_confidence":"0.8","model_c_value":"powered","model_c_confidence":"0.7",
     "dino_oof":"False","threshold":"0.6","in_training_split":"False","split":"","run_timestamp":"2026-11-10T02:51:00-08:00"},
]
with (TMP / "derived" / "ai_label_verification.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=ver_cols); w.writeheader(); w.writerows(rows)
(TMP / "derived" / "ai_label_verification_summary.json").write_text(json.dumps({
    "n_labels": 2, "verdicts_overall": {"majority_pass": 1, "tie": 1},
    "majority_rule": {"passes": 1, "ties": 1, "description": "majority"},
}), encoding="utf-8")

env = {
    "WV_DB": str(TMP / "scratch.db"),
    "WV_CANDIDATES": str(TMP / "data" / "candidates.csv"),
    "WV_CONTRIBUTED": str(TMP / "contributed.csv"),
    "WV_IMAGE_DIR": str(TMP / "images"),
    "WV_BLUR_DIR": str(TMP / "images_blurred"),
    "WV_EXPORT_DIR": str(TMP / "exports"),
    "WV_AUTO_DIR": str(TMP / "auto"),
    "WV_BLOCKLIST_DIR": str(TMP / "block"),
    "WV_IMAGE_PROXY": "0",
    "WV_ACCESS_CODE": "",
    "WV_BLUR_MANIFESTS": str(TMP / "blur.jsonl"),
    "WV_CACHE_DIR": str(TMP / "cache"),
    "WV_ASSIGN_WITHOUT_LOCAL_IMAGE": "1",
    "WV_AI_JOBS": str(TMP / "var" / "ai_jobs"),
}
os.environ.pop("WV_HF_DATASET", None)
os.environ.update(env)

# Point the ai_review module paths at TMP by chdir? The module uses ROOT = parents[1]
# of server/ai_review.py which is the real repo. We'll monkeypatch after import via env
# is not enough. Test the functions by patching attributes, and TestClient against
# a copied... Actually the live server module will read repo derived/, which may be
# fine for unit tests of routing, but we want isolation.
#
# We'll import ai_review and patch its paths, then import main similarly... main
# imports ai_review at route-call time if we add the import in main. For tests of
# query_labels we patch ai_review constants.

import ai_review as AR
AR.ROOT = TMP
AR.JOBS = TMP / "var" / "ai_jobs"
AR.CURRENT = AR.JOBS / "current.json"
AR.LOG = AR.JOBS / "log.txt"
AR.VERIFY_CSV = TMP / "derived" / "ai_label_verification.csv"
AR.VERIFY_JSON = TMP / "derived" / "ai_label_verification_summary.json"
AR.CONSENSUS_JSON = TMP / "derived" / "ai_labels_3model_summary.json"

st = AR.status()
check("status reads live job", st["ok"] and st["job"]["job"] == "verify_ai_labels" and st["job"]["n_done"] == 3, json.dumps(st["job"]))
check("status log tail nonempty", len(st["log_tail"]) >= 2)
sm = AR.summary()
check("summary reads verification json", (sm.get("verification") or {}).get("n_labels") == 2)

q = AR.query_labels(verdict="tie")
check("query filters by verdict", q["total"] == 1 and q["rows"][0]["image_id"] == "wv_000002", json.dumps(q["total"]))
q = AR.query_labels(source="auto_approved")
check("query filters by source", q["total"] == 1 and q["rows"][0]["majority_basis"] == "ai+human")
q = AR.query_labels(q="wv_000001")
check("query text search", q["total"] == 1)
check("facets include majority_basis and verdict", "majority_pass" in q["facets"]["verdict"] and "ai+human" in q["facets"]["majority_basis"])
# voters parsed as json
check("voters are structured", isinstance(q["rows"][0]["voters"], list) and q["rows"][0]["voters"][1]["id"] == "ai")

# FastAPI routes: import main after env is set. Patch AR paths used by main.
# main will import server.ai_review (or ai_review). We'll patch after import.
try:
    from fastapi.testclient import TestClient
    import main
    # If main imported its own ai_review, patch that module too
    import importlib
    for mod in (main,):
        pass
    # main.py will do `import ai_review as AI` — after tests started from server/.
    # Re-patch whatever main bound.
    if hasattr(main, "AI"):
        main.AI.ROOT = TMP
        main.AI.JOBS = AR.JOBS
        main.AI.CURRENT = AR.CURRENT
        main.AI.LOG = AR.LOG
        main.AI.VERIFY_CSV = AR.VERIFY_CSV
        main.AI.VERIFY_JSON = AR.VERIFY_JSON
        main.AI.CONSENSUS_JSON = AR.CONSENSUS_JSON
    ev_before = None
    with TestClient(main.app) as cl:
        # events count before
        ev_before = cl.get("/api/health").json().get("events")
        r = cl.get("/api/ai/status")
        check("GET /api/ai/status 200", r.status_code == 200, str(r.status_code))
        body = r.json()
        check("GET /api/ai/status has job", body.get("job", {}).get("stage") == "score", json.dumps(body.get("job")))
        r = cl.get("/api/ai/summary")
        check("GET /api/ai/summary 200", r.status_code == 200)
        r = cl.get("/api/ai/labels", params={"verdict": "majority_pass"})
        check("GET /api/ai/labels filter", r.status_code == 200 and r.json()["total"] == 1, str(r.text)[:200])
        r = cl.get("/ai")
        check("GET /ai serves html", r.status_code == 200 and "text/html" in r.headers.get("content-type", ""), r.headers.get("content-type"))
        ev_after = cl.get("/api/health").json().get("events")
        check("AI review APIs do not write events", ev_before == ev_after, f"{ev_before} -> {ev_after}")
        # no POST endpoints should exist that write AI as human
        check("no /api/ai write routes on the app", not any(
            getattr(rt, "path", "").startswith("/api/ai") and (getattr(rt, "methods", None) or set()) & {"POST", "PUT", "DELETE", "PATCH"}
            for rt in main.app.routes
        ))
        h = cl.get("/api/health").json()
        check("health advertises ai_review", "ai_review" in (h.get("features") or []), json.dumps(h.get("features")))
except ImportError as e:
    check("server tests (fastapi missing)", False, str(e))
except Exception as e:
    check("server tests ran", False, repr(e))

failed = [n for n, ok in results if not ok]
print(f"\n{len(results)-len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
