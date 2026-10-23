#!/usr/bin/env python3
"""End-to-end test: real server (temp DB) + headless Chromium.
alice and bob annotate the same images independently; bob works OFFLINE (Playwright context offline, page reloaded
offline from the service worker), then reconnects; events must sync, conflicts must be flagged, a third annotator
must be assigned, adjudication must resolve, labels.csv must update, and the JSONL export/import/rebuild must round-trip.
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

env = dict(os.environ, WV_DB=str(TMP / "e2e.db"), WV_EXPORT_DIR=str(TMP / "exports"), WV_IMAGE_PROXY="0")
srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--port", str(PORT), "--log-level", "warning"],
                       cwd=ROOT / "server", env=env, stdout=open(TMP / "server.log", "w"), stderr=subprocess.STDOUT)
def _try_health():
    try: return api("/api/health").status_code == 200
    except Exception: return False

try:
    check("server starts", wait(_try_health, 20))
    with sync_playwright() as p:
        br = p.chromium.launch()
        def login(name):
            ctx = br.new_context(viewport={"width": 1280, "height": 800}); pg = ctx.new_page()
            pg.on("console", lambda m: m.type == "error" and print(f"[{name} console] {m.text}"))
            pg.goto(BASE + "/"); pg.fill("#name", name); pg.press("#name", "Enter")
            pg.wait_for_function("document.querySelector('#pos').textContent.includes(' / 30')", timeout=20000)
            pg.evaluate("navigator.serviceWorker.ready")
            # all 30 batch images cached for offline use
            pg.wait_for_function("caches.open('wv-img-v1').then(c=>c.keys()).then(k=>window.__n=k.length) && window.__n>=30", timeout=60000)
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
        br.close()

    st = api("/api/stats").json()
    check("kappa computed per field", "device_type" in st["kappa"] and st["kappa"]["device_type"]["n_items"] >= 3,
          json.dumps({f: v["pooled_kappa"] for f, v in st["kappa"].items()}))

    # ---- git bridge: export -> JSONL -> rebuild == server; import into empty DB round-trips ----
    evdir = TMP / "events"
    subprocess.run([sys.executable, ROOT / "scripts/sync_events.py", "export", "--db", env["WV_DB"], "--events", evdir], check=True)
    files = sorted(p.name for p in evdir.glob("*.jsonl"))
    check("export writes one append-only file per annotator", files == ["alice.jsonl", "bob.jsonl", "carol.jsonl", "samir.jsonl"], str(files))
    n_lines = sum(1 for p in evdir.glob("*.jsonl") for _ in open(p))
    subprocess.run([sys.executable, ROOT / "scripts/sync_events.py", "export", "--db", env["WV_DB"], "--events", evdir], check=True)
    check("re-export appends nothing (dedup by uuid)", sum(1 for p in evdir.glob("*.jsonl") for _ in open(p)) == n_lines, f"{n_lines} lines")
    subprocess.run([sys.executable, ROOT / "scripts/build_derived.py", "--events", evdir, "--out", TMP / "derived"], check=True)
    same = (TMP / "derived/labels.csv").read_bytes() == api("/export/labels.csv").content
    check("CI rebuild from JSONL reproduces server labels.csv byte-for-byte", same)
    out = subprocess.run([sys.executable, ROOT / "scripts/sync_events.py", "import", "--db", TMP / "fresh.db", "--events", evdir], capture_output=True, text=True).stdout
    check("import into empty DB restores every event", f"imported {n_lines} new" in out, out.strip())
    out2 = subprocess.run([sys.executable, ROOT / "scripts/sync_events.py", "import", "--db", TMP / "fresh.db", "--events", evdir], capture_output=True, text=True).stdout
    check("re-import is idempotent", "imported 0 new" in out2, out2.strip())
    shutil.copy(TMP / "derived/labels.csv", SHOTS / "e2e_labels.csv")
finally:
    srv.terminate()
    print(f"\n{sum(r[1] for r in results)}/{len(results)} checks passed; temp dir {TMP}")
    json.dump([{"check": n, "pass": ok, "detail": d} for n, ok, d in results], open(SHOTS / "e2e_results.json", "w"), indent=1)
    sys.exit(0 if results and all(r[1] for r in results) else 1)
