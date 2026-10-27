#!/usr/bin/env python3
"""End-to-end test: real server (temp DB) + headless Chromium.
alice and bob annotate the same images independently; bob works OFFLINE (Playwright context offline, page reloaded
offline from the service worker), then reconnects; events must sync, conflicts must be flagged, a third annotator
must be assigned, adjudication must resolve, labels.csv must update, and the JSONL export/import/rebuild must round-trip.
Review mode: machine suggestions (synthetic, in a temp WV_AUTO_DIR) are confirmed with one key or fixed per field;
events carry provenance under the human's name; the model never counts as an annotator, in kappa or in labels.csv.
Run: python tests/test_e2e.py   (screenshots -> screenshots/)"""
import csv, io, json, os, shutil, subprocess, sys, tempfile, time, uuid
from pathlib import Path
import httpx
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("E2E_PORT", 8812)); BASE = f"http://127.0.0.1:{PORT}"
SHOTS = ROOT / "screenshots"; SHOTS.mkdir(exist_ok=True)
TMP = Path(tempfile.mkdtemp(prefix="wv_e2e_"))
results = []

def check(name, cond, detail=""):
    results.append((name, bool(cond), detail)); print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail else ""), flush=True)

def api(path, **kw): return httpx.get(BASE + path, timeout=30, **kw)
def labels():
    return {r["image_id"]: r for r in csv.DictReader(io.StringIO(api("/export/labels.csv").text))}
def wait(fn, timeout=30, step=0.3):
    t = time.time()
    while time.time() - t < timeout:
        v = fn()
        if v: return v
        time.sleep(step)
    return fn()

# ---- synthetic machine suggestions (the real model is not needed for the test) ----
sys.path.insert(0, str(ROOT / "server")); import wvcore
TAG = "testmodel_zs-v0"; AUTO = TMP / "auto"; AUTO.mkdir()
_ids = sorted(r["candidate_id"] for r in csv.DictReader(open(ROOT / "data/candidates.csv", encoding="utf-8")) if (ROOT / "images" / f"{r['candidate_id']}.jpg").exists())
CHILD = None
with open(AUTO / f"{TAG}.jsonl", "w") as fh:
    for n, i in enumerate(_ids):
        sug = {f["name"]: {"value": ("manual" if f["name"] == "device_type" else f["options"][0]), "confidence": round(0.3 + (n % 60) / 100, 2)} for f in wvcore.SCHEMA["fields"]}
        flags = []
        if n < 5: sug["device_type"]["confidence"] = round(0.10 + n / 100, 2)                  # the 5 least confident come first
        fh.write(json.dumps({"image_id": i, "model_tag": TAG, "suggestions": sug, "screening": {"child": 0.9 if flags else 0.1, "visible_face": 0.2},
                             "flags": flags, "boxes": [{"box": [0.1, 0.1, 0.5, 0.6], "score": 0.8, "label": "wheelchair"}]}) + "\n")

BLUR = TMP / "blurred"; BLUR.mkdir(); BLUR_MAN = TMP / "blur_manifest.jsonl"; BLUR_MAN.write_text("")
env = dict(os.environ, WV_DB=str(TMP / "e2e.db"), WV_EXPORT_DIR=str(TMP / "exports"), WV_IMAGE_PROXY="0", WV_AUTO_DIR=str(AUTO),
           WV_BLUR_DIR=str(BLUR), WV_BLUR_MANIFESTS=str(BLUR_MAN), WV_BLOCKLIST_DIR=str(TMP / "block"))
(TMP / "block").mkdir(); BLOCKED_ID = _ids[-1]; (TMP / "block" / "removed_test.txt").write_text("# test\n" + BLOCKED_ID + "\n")
srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--port", str(PORT), "--log-level", "warning"],
                       cwd=ROOT / "server", env=env, stdout=open(TMP / "server.log", "w"), stderr=subprocess.STDOUT)
def _try_health():
    try: return api("/api/health").status_code == 200
    except Exception: return False

try:
    check("server starts", wait(_try_health, 20))
    check("blocklist counted in health", api("/api/health").json().get("blocked") == 1, str(api("/api/health").json().get("blocked")))
    check("blocked image never served", api(f"/api/images/{BLOCKED_ID}").status_code == 404 and api(f"/img/{BLOCKED_ID}").status_code == 404)
    with sync_playwright() as p:
        br = p.chromium.launch()
        def login(name):
            ctx = br.new_context(viewport={"width": 1280, "height": 800}); pg = ctx.new_page()
            pg.on("console", lambda m: m.type == "error" and print(f"[{name} console] {m.text}"))
            pg.goto(BASE + "/"); pg.fill("#name", name); pg.press("#name", "Enter")
            pg.wait_for_function("document.querySelector('#pos').textContent.includes(' / 30')", timeout=20000)
            pg.evaluate("navigator.serviceWorker.ready")
            # all 30 batch images cached for offline use
            pg.wait_for_function("caches.open('wv-img-v2').then(c=>c.keys()).then(k=>window.__n=k.length) && window.__n>=30", timeout=60000)
            return ctx, pg
        def ids(pg): return pg.evaluate("S.images.map(i=>i.image_id)")
        def press(pg, *keys):
            for k in keys: pg.keyboard.press(k); pg.wait_for_timeout(60)
        def done_and_wait(pg, nxt):
            pg.keyboard.press("Enter"); pg.wait_for_function(f"document.querySelector('#pos').textContent.startsWith('{nxt} /')", timeout=10000)

        actx, A = login("alice"); bctx, B = login("bob")
        ia, ib = ids(A), ids(B)
        check("bob gets the same images as alice (2nd independent annotator)", ia == ib, f"{len(ia)} images")
        check("alice batch cached for offline", A.evaluate("window.__n") >= 30)
        im1, im2, im3, im4 = ia[:4]

        # ---- alice (online) ----
        press(A, "1", "1", "1", "1", "2", "1", "3")            # manual, yes, self, none, side, daylight, street
        done_and_wait(A, 2)
        done_and_wait(A, 3)                                       # accept suggestions only
        press(A, "1", "1", "1", "1", "1", "1", "1"); done_and_wait(A, 4)
        press(A, "4", "2", "6", "1", "1", "1", "1"); done_and_wait(A, 5)
        wait(lambda: A.evaluate("document.querySelector('#queue').textContent") == "0 queued", 15)
        A.screenshot(path=str(SHOTS / "01_alice_annotating.png"))
        n_alice = len(api("/api/events?annotator=alice").json()["events"])
        check("alice's events reach the server", n_alice >= 20, f"{n_alice} events")

        # ---- bob goes offline ----
        bctx.set_offline(True); B.wait_for_timeout(300)
        B.reload(); B.wait_for_function("document.querySelector('#pos').textContent.includes(' / 30')", timeout=15000)
        check("app shell reloads while offline (service worker)", "offline" in B.inner_text("#net"))
        B.wait_for_function("document.querySelector('#img').complete && document.querySelector('#img').naturalWidth>0", timeout=10000)
        check("cached image displays offline", B.evaluate("document.querySelector('#img').naturalWidth") > 0)
        press(B, "2", "1", "3", "2", "2", "1", "3")              # powered (alice: manual) -> conflict
        press(B, "f", "b")                                         # face-blur flag + bbox mode
        bb = B.locator("#cv").bounding_box()
        B.mouse.move(bb["x"] + bb["width"] * .2, bb["y"] + bb["height"] * .2); B.mouse.down()
        B.mouse.move(bb["x"] + bb["width"] * .7, bb["y"] + bb["height"] * .8, steps=5); B.mouse.up(); press(B, "b")
        B.wait_for_timeout(200)
        B.screenshot(path=str(SHOTS / "02_bob_offline_bbox_conflict.png"))
        done_and_wait(B, 2)
        done_and_wait(B, 3)
        press(B, "1", "1", "1", "1", "1", "1", "1"); done_and_wait(B, 4)   # same as alice -> consensus
        press(B, "e", "3"); done_and_wait(B, 5)                           # exclude identifiable_person (alice did not)
        q = int(B.inner_text("#queue").split()[0])
        check("bob's edits queue while offline", q > 0, f"{q} queued")
        check("server has none of bob's events yet", len(api("/api/events?annotator=bob").json()["events"]) == 0)
        B.screenshot(path=str(SHOTS / "03_bob_offline_queue.png"))

        # ---- bob back online: 'online' event triggers push ----
        bctx.set_offline(False)
        ok = wait(lambda: B.inner_text("#queue") == "0 queued" and len(api("/api/events?annotator=bob").json()["events"]) >= q, 30)
        nb = len(api("/api/events?annotator=bob").json()["events"])
        check("bob's queued events sync after reconnect", ok, f"{nb} events on server")
        B.screenshot(path=str(SHOTS / "04_bob_back_online_synced.png"))

        # idempotency: re-push everything bob has
        evs = [{k: v for k, v in e.items() if k != "seq"} for e in api("/api/events?annotator=bob").json()["events"]]
        r = httpx.post(BASE + "/api/events", json={"events": evs}).json()
        check("re-push is idempotent", not r["accepted"] and len(r["duplicate"]) == len(evs), f"{len(r['duplicate'])} duplicates")
        check("server never overwrote alice's labels", len(api("/api/events?annotator=alice").json()["events"]) == n_alice)

        conf = {c["image_id"]: c for c in api("/api/conflicts").json()["conflicts"]}
        check("device_type conflict flagged (manual vs powered)", im1 in conf and conf[im1]["conflicts"].get("device_type") == {"alice": "manual", "bob": "powered"},
              json.dumps(conf.get(im1, {}).get("conflicts")))
        check("exclude conflict flagged", im4 in conf and "exclude" in conf[im4]["conflicts"])
        L = labels()
        check("labels.csv: conflict image needs third annotator", L[im1]["status"] == "needs_third_annotator", L[im1]["status"])
        check("labels.csv: agreed image is final consensus", L[im3]["status"] == "final" and L[im3]["device_type_agreement"] == "consensus",
              f"{L[im3]['status']} {L[im3]['device_type']}")
        check("face_blur_missed OR-combined", L[im1]["face_blur_missed"] == "True")
        pa = list(csv.DictReader(io.StringIO(api("/export/per_annotator_labels.csv").text)))
        bobbox = [json.loads(r["bbox"]) for r in pa if r["image_id"] == im1 and r["annotator"] == "bob"][0]
        check("bob's bbox stored", len(bobbox) == 1 and 0.15 < bobbox[0][0] < 0.25, str(bobbox))

        # ---- correction of a prior label by alice (does not touch bob) ----
        A.keyboard.press("Home"); A.evaluate("S.idx=0; render()")
        press(A, "ArrowUp", "ArrowUp", "ArrowUp", "ArrowUp", "ArrowUp", "ArrowUp", "ArrowUp", "5")  # device_type -> unclear
        wait(lambda: A.inner_text("#queue") == "0 queued", 15)
        conf = {c["image_id"]: c for c in api("/api/conflicts").json()["conflicts"]}
        check("correction: alice's latest value wins for alice only", conf[im1]["conflicts"]["device_type"] == {"alice": "unclear", "bob": "powered"})
        press(A, "ArrowUp", "1"); wait(lambda: A.inner_text("#queue") == "0 queued", 15)   # back to manual (row auto-advanced)

        # ---- third annotator ----
        cctx = br.new_context(); C = cctx.new_page(); C.goto(BASE + "/"); C.fill("#name", "carol"); C.press("#name", "Enter")
        C.wait_for_function("S.images && S.images.length > 0", timeout=20000)
        ic = C.evaluate("S.images.map(i=>i.image_id)")
        check("third annotator gets the disagreement images first", set(ic[:2]) == {im1, im4}, str(ic[:3]))
        C.evaluate(f"S.idx = S.images.findIndex(i=>i.image_id=='{im1}'); render()")
        press(C, "1"); C.keyboard.press("Enter"); C.wait_for_timeout(500)
        wait(lambda: C.inner_text("#queue") == "0 queued", 15)
        L = labels()
        check("2-of-3 majority resolves device_type", L[im1]["device_type"] == "manual" and L[im1]["device_type_agreement"] == "majority",
              f"{L[im1]['device_type']} / {L[im1]['device_type_agreement']} / {L[im1]['status']}")

        # ---- adjudicator view ----
        D = br.new_context(viewport={"width": 1280, "height": 900}).new_page()
        D.goto(BASE + "/adjudicate.html"); D.fill("#adj", "samir"); D.dispatch_event("#adj", "change")
        D.wait_for_selector(f".card[data-id='{im4}']"); D.screenshot(path=str(SHOTS / "05_adjudicator_view.png"), full_page=True)
        D.click(f".card[data-id='{im4}'] .opt[data-f='exclude'][data-v='identifiable_person']")
        D.wait_for_timeout(800)
        L = labels()
        check("adjudication resolves exclude conflict in labels.csv", L[im4]["exclude"] == "identifiable_person" and L[im4]["exclude_agreement"] == "adjudicated",
              f"{L[im4]['exclude']} / {L[im4]['exclude_agreement']} / {L[im4]['status']}")
        D.screenshot(path=str(SHOTS / "06_adjudicator_after_resolve.png"), full_page=True)
        A.click("#btnList"); A.wait_for_timeout(300); A.screenshot(path=str(SHOTS / "07_alice_my_images.png"))
        A.click("#btnList"); A.keyboard.press("?"); A.wait_for_timeout(200); A.screenshot(path=str(SHOTS / "08_keyboard_help.png"))

        # ---- review auto labels: dave confirms / fixes machine suggestions ----
        check("server serves machine suggestions", api("/api/auto").json()["models"].get(TAG, 0) == len(_ids), json.dumps(api("/api/auto").json()["models"]))
        dctx, Dv = login("dave")
        Dv.click("#btnReview")
        Dv.wait_for_function("S.mode==='review' && S.images.length>0 && S.images.every(i=>S.auto[i.image_id]) && document.querySelector('#pos').textContent.includes('REVIEW')", timeout=20000)
        Dv.evaluate("S.row=0; render()")
        rq = Dv.evaluate("S.images.map(i=>i.image_id)")
        confs = Dv.evaluate("S.images.filter(i=>!S.mine[i.image_id]).map(i=>S.auto[i.image_id].suggestions.device_type.confidence)")
        check("review queue: lowest confidence first", confs == sorted(confs) and len(confs) > 5, str(confs[:6]))
        check("review mode shows suggestions with confidence", Dv.locator(".opt.ai").count() >= 5 and "%" in Dv.inner_text(".aiv"), Dv.inner_text(".aiv"))
        Dv.wait_for_function("document.querySelector('#img').complete && document.querySelector('#img').naturalWidth>0", timeout=10000)
        Dv.screenshot(path=str(SHOTS / "09_review_mode_suggestions.png"))
        r1 = Dv.evaluate("cur().image_id"); press(Dv, "a")
        Dv.wait_for_function(f"cur().image_id !== '{r1}'", timeout=10000)
        r2 = Dv.evaluate("cur().image_id"); Dv.evaluate("S.row=0; render()")
        press(Dv, "2")                                                       # fix device_type: powered (model said manual)
        Dv.wait_for_timeout(200); Dv.screenshot(path=str(SHOTS / "10_review_fixed_field.png"))
        press(Dv, "Enter"); Dv.wait_for_function(f"cur().image_id !== '{r2}'", timeout=10000)
        wait(lambda: Dv.inner_text("#queue") == "0 queued", 15)
        ev = api("/api/events?annotator=dave").json()["events"]
        e1 = {e["field"]: e for e in ev if e["image_id"] == r1}; e2 = {e["field"]: e for e in ev if e["image_id"] == r2}
        check("confirm-all: every suggested field saved as dave's label + done",
              all(e1.get(f["name"], {}).get("value") == (f["options"][0] if f["name"] != "device_type" else "manual") for f in wvcore.SCHEMA["fields"]) and e1["_status"]["value"] == "done",
              json.dumps({k: v["value"] for k, v in e1.items()}))
        check("provenance recorded on review events", all(e1[f].get("source") == "auto_review" and e1[f].get("model_tag") == TAG for f in e1)
              and e1["device_type"]["suggested"] == "manual" and isinstance(e1["device_type"]["confidence"], float), json.dumps(e1["device_type"]))
        check("fix one field: human value wins, suggestion kept as provenance", e2["device_type"]["value"] == "powered" and e2["device_type"]["suggested"] == "manual"
              and e2["occupied"]["value"] == e2["occupied"]["suggested"], json.dumps(e2["device_type"]))
        check("review events are written under the human's name only", {e["annotator"] for e in ev} == {"dave"})
        # screening flag: mark one queued image as "likely child" (server reloads annotations/auto when it changes)
        CHILD = [i for i in rq if i not in (r1, r2)][3]
        recs = [json.loads(l) for l in open(AUTO / f"{TAG}.jsonl")]
        for rec in recs:
            if rec["image_id"] == CHILD: rec["flags"] = ["child"]; rec["screening"]["child"] = 0.9
        (AUTO / f"{TAG}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        Dv.evaluate("getReviewBatch()"); Dv.wait_for_function(f"(S.auto['{CHILD}'].flags||[]).includes('child')", timeout=10000)
        if CHILD not in rq:
            check("child-flagged image is in dave's review queue", False, CHILD)
        else:
            Dv.evaluate(f"S.idx = S.images.findIndex(i=>i.image_id=='{CHILD}'); render()"); press(Dv, "a"); Dv.wait_for_timeout(300)
            warned = "Press a again" in Dv.inner_text("#aibanner") and Dv.evaluate(f"val('{CHILD}','_status')") is None
            Dv.screenshot(path=str(SHOTS / "11_review_screening_warning.png"))
            check("screening flag never auto-excludes; needs a 2nd press", warned and Dv.evaluate(f"val('{CHILD}','exclude')") is None)
            press(Dv, "a"); Dv.wait_for_timeout(300)
            check("2nd press confirms the flagged image (not excluded)", Dv.evaluate(f"val('{CHILD}','_status')") == "done" and Dv.evaluate(f"val('{CHILD}','exclude')") == "")
        # face blur: a blurred copy appears (as after scripts/auto_blur.py); the app must show it + the face boxes
        from PIL import Image, ImageFilter
        BL = [i for i in rq if i not in (r1, r2, CHILD)][0]
        im0 = Image.open(ROOT / "images" / f"{BL}.jpg").convert("RGB"); W0, H0 = im0.size; fb = (0.4, 0.2, 0.2, 0.25)
        box = tuple(int(v) for v in (fb[0] * W0, fb[1] * H0, (fb[0] + fb[2]) * W0, (fb[1] + fb[3]) * H0))
        im0.paste(im0.crop(box).filter(ImageFilter.GaussianBlur(12)), box[:2]); im0.save(BLUR / f"{BL}.jpg", quality=90)
        BLUR_MAN.write_text(json.dumps({"image_id": BL, "n_faces": 1, "faces": [{"box": list(fb), "score": 0.91}], "detector": "test", "version": "t1"}) + "\n")
        check("server serves the blurred copy, original untouched", api(f"/img/{BL}").content == (BLUR / f"{BL}.jpg").read_bytes()
              and (ROOT / "images" / f"{BL}.jpg").read_bytes() != (BLUR / f"{BL}.jpg").read_bytes())
        Dv.evaluate("getReviewBatch()"); Dv.wait_for_function(f"(S.images.find(i=>i.image_id=='{BL}')||{{}}).blurred===true", timeout=10000)
        Dv.evaluate(f"S.idx = S.images.findIndex(i=>i.image_id=='{BL}'); render()")
        Dv.wait_for_function("document.querySelector('#img').src.includes('?b=') && document.querySelector('#img').complete && document.querySelector('#img').naturalWidth>0", timeout=10000)
        Dv.wait_for_timeout(200); Dv.screenshot(path=str(SHOTS / "13_review_blurred_face_overlay.png"))
        check("app shows blurred image with face-box overlay hint", "face-blurred" in Dv.inner_text("#aibanner") and Dv.evaluate("cur().faces.length") == 1)
        press(Dv, "f"); wait(lambda: Dv.inner_text("#queue") == "0 queued", 15)
        fm = [e for e in api("/api/events?annotator=dave").json()["events"] if e["image_id"] == BL and e["field"] == "face_blur_missed"]
        check("'f' flags a missed face in review mode (with provenance)", fm and fm[-1]["value"] is True and fm[-1].get("source") == "auto_review", json.dumps(fm[-1:]))
        Dv.keyboard.press("?"); Dv.wait_for_timeout(200); Dv.screenshot(path=str(SHOTS / "12_review_help.png")); Dv.keyboard.press("Escape")
        wait(lambda: Dv.inner_text("#queue") == "0 queued", 15)
        L = labels()
        check("model is not an annotator: dave-only image is not final", L[r1]["status"] != "final" and L[r1]["annotators"] == "dave", f"{L[r1]['status']} {L[r1]['annotators']}")
        # erin reviews: pair formation offers dave's images first
        ectx, E = login("erin"); E.click("#btnReview")
        E.wait_for_function("S.mode==='review' && S.images.length>0 && S.images.every(i=>S.auto[i.image_id]) && document.querySelector('#pos').textContent.includes('REVIEW')", timeout=20000)
        ie = E.evaluate("S.images.map(i=>i.image_id)")
        check("review queue prefers images with one human label (pairs form)", r1 in ie[:6] and r2 in ie[:6], str(ie[:6]))
        for rid in (r1, r2):
            E.evaluate(f"S.idx = S.images.findIndex(i=>i.image_id=='{rid}'); render()"); press(E, "a"); E.wait_for_timeout(300)
        wait(lambda: E.inner_text("#queue") == "0 queued", 15)
        L = labels()
        check("two humans confirming -> final consensus", L[r1]["status"] == "final" and L[r1]["device_type"] == "manual" and L[r1]["annotators"] == "dave;erin",
              f"{L[r1]['status']} {L[r1]['device_type']} {L[r1]['annotators']}")
        check("human fix vs human confirm -> disagreement (model's vote ignored)", L[r2]["status"] == "needs_third_annotator", L[r2]["status"])
        br.close()

    st = api("/api/stats").json()
    anns = {a for f in st["kappa"].values() for pair in f.get("pairwise", {}) for a in str(pair).replace("|", ",").replace("__", ",").split(",")}
    pa_anns = {r["annotator"] for r in csv.DictReader(io.StringIO(api("/export/per_annotator_labels.csv").text))}
    check("auto labels are excluded from kappa and per-annotator labels", TAG not in json.dumps(st["kappa"]) and TAG not in pa_anns and TAG not in api("/export/labels.csv").text,
          f"annotators {sorted(pa_anns)}")
    avh = api("/export/auto_vs_human.json").json()
    dt = avh.get(TAG, {}).get("fields", {}).get("device_type", {})
    check("model-vs-human agreement reported separately", dt.get("all_done_human_labels", {}).get("n", 0) >= 6 and dt.get("reviewed", {}).get("n", 0) >= 4
          and dt.get("independent", {}).get("n", 0) >= 2, json.dumps({k: (dt.get(k) or {}).get("n") for k in ("all_done_human_labels", "independent", "reviewed")}))
    check("kappa computed per field", "device_type" in st["kappa"] and st["kappa"]["device_type"]["n_items"] >= 3,
          json.dumps({f: v["pooled_kappa"] for f, v in st["kappa"].items()}))

    # ---- git bridge: export -> JSONL -> rebuild == server; import into empty DB round-trips ----
    evdir = TMP / "events"
    subprocess.run([sys.executable, ROOT / "scripts/sync_events.py", "export", "--db", env["WV_DB"], "--events", evdir], check=True)
    files = sorted(p.name for p in evdir.glob("*.jsonl"))
    check("export writes one append-only file per annotator", files == ["alice.jsonl", "bob.jsonl", "carol.jsonl", "dave.jsonl", "erin.jsonl", "samir.jsonl"], str(files))
    dl = [json.loads(l) for l in open(evdir / "dave.jsonl")]
    check("provenance survives JSONL export", sum(e.get("source") == "auto_review" for e in dl) >= 10 and all(e.get("model_tag") == TAG for e in dl if e.get("source")))
    n_lines = sum(1 for p in evdir.glob("*.jsonl") for _ in open(p))
    subprocess.run([sys.executable, ROOT / "scripts/sync_events.py", "export", "--db", env["WV_DB"], "--events", evdir], check=True)
    check("re-export appends nothing (dedup by uuid)", sum(1 for p in evdir.glob("*.jsonl") for _ in open(p)) == n_lines, f"{n_lines} lines")
    subprocess.run([sys.executable, ROOT / "scripts/build_derived.py", "--events", evdir, "--out", TMP / "derived", "--auto", AUTO], check=True)
    check("CI rebuild writes derived/auto_vs_human.json", json.load(open(TMP / "derived/auto_vs_human.json"))[TAG]["fields"]["device_type"] == dt)
    same = (TMP / "derived/labels.csv").read_bytes() == api("/export/labels.csv").content
    check("CI rebuild from JSONL reproduces server labels.csv byte-for-byte", same)
    out = subprocess.run([sys.executable, ROOT / "scripts/sync_events.py", "import", "--db", TMP / "fresh.db", "--events", evdir], capture_output=True, text=True).stdout
    check("import into empty DB restores every event", f"imported {n_lines} new" in out, out.strip())
    out2 = subprocess.run([sys.executable, ROOT / "scripts/sync_events.py", "import", "--db", TMP / "fresh.db", "--events", evdir], capture_output=True, text=True).stdout
    check("re-import is idempotent", "imported 0 new" in out2, out2.strip())
    ev2 = TMP / "events2"
    subprocess.run([sys.executable, ROOT / "scripts/sync_events.py", "export", "--db", TMP / "fresh.db", "--events", ev2], check=True)
    check("provenance round-trips through import", (ev2 / "dave.jsonl").read_text().count('"auto_review"') == (evdir / "dave.jsonl").read_text().count('"auto_review"'))
    shutil.copy(TMP / "derived/labels.csv", SHOTS / "e2e_labels.csv")
except Exception:
    import traceback; traceback.print_exc(); check("test ran to completion", False)
finally:
    srv.terminate()
    print(f"\n{sum(r[1] for r in results)}/{len(results)} checks passed; temp dir {TMP}")
    json.dump([{"check": n, "pass": ok, "detail": d} for n, ok, d in results], open(SHOTS / "e2e_results.json", "w"), indent=1)
    sys.exit(0 if results and all(r[1] for r in results) else 1)
