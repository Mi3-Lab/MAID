"""WheelView annotation server: FastAPI + SQLite. Serves the PWA, images, the append-only event log,
assignment (2 independent annotators, a 3rd on disagreement), adjudication and derived exports."""
import csv, hashlib, hmac, io, json, os, sqlite3, threading, time, uuid
from contextlib import asynccontextmanager
from pathlib import Path
import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
import wvcore

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("WV_DB", ROOT / "var" / "wheelview.db"))
CANDS = Path(os.environ.get("WV_CANDIDATES", ROOT / "data" / "candidates.csv"))
IMAGE_DIR = Path(os.environ.get("WV_IMAGE_DIR", ROOT / "images"))          # <image_id>.jpg, filled by scripts/fetch_images.py
CACHE_DIR = Path(os.environ.get("WV_CACHE_DIR", ROOT / "var" / "imgcache"))
EXPORT_DIR = Path(os.environ.get("WV_EXPORT_DIR", ROOT / "var" / "exports"))
AUTO_DIR = Path(os.environ.get("WV_AUTO_DIR", ROOT / "annotations" / "auto"))     # machine suggestions (never human events)
REVIEW_AUTO_DIR = Path(os.environ.get("WV_REVIEW_AUTO_DIR", ROOT / "var" / "assist"))  # gitignored extra suggestion files (e.g. assistant_visual_review_attr-v1.jsonl); never labels
ADJ_QUEUE = Path(os.environ.get("WV_ADJ_QUEUE", ROOT / "var" / "adjudication" / "queue.jsonl"))  # gitignored review-request queue (concerns, not decisions)
AUTO_MODEL = os.environ.get("WV_AUTO_MODEL", "").strip()                          # default: newest annotations/auto/*.jsonl
BLUR_DIR = Path(os.environ.get("WV_BLUR_DIR", ROOT / "images_blurred"))            # scripts/auto_blur.py output (originals untouched)
BLUR_MANIFESTS = [Path(p) for p in os.environ.get("WV_BLUR_MANIFESTS", "").split(os.pathsep) if p] or \
                 [ROOT / "data" / "blur_manifest.jsonl", ROOT / "var" / "blur_manifest_local.jsonl"]
PROXY = os.environ.get("WV_IMAGE_PROXY", "1") == "1"          # fetch missing images (e.g. Commons) from image_url and cache them
ASSIGN_NO_IMAGE = os.environ.get("WV_ASSIGN_WITHOUT_LOCAL_IMAGE", "0") == "1"  # default: only assign images we can already serve
UA = {"User-Agent": "WheelViewAnnotate/0.1 (UC Merced Mi3 Lab; research dataset annotation)"}
for d in (DB_PATH.parent, CACHE_DIR, EXPORT_DIR): d.mkdir(parents=True, exist_ok=True)

CONTRIB = Path(os.environ.get("WV_CONTRIBUTED", ROOT / "data" / "contributed.csv"))   # v1.1: own open-licensed images (scripts/import_contributed.py)
CANDIDATES = wvcore.load_candidates(CANDS)
_CONTRIB_SIG = [None]
def refresh_candidates():
    """Merge data/contributed.csv into CANDIDATES (in place) whenever it changes, so imported images need no restart.
    Contributed ids are a separate range (wv_9xxxxx) and never collide with candidates.csv."""
    try: st = CONTRIB.stat(); sig = (st.st_mtime_ns, st.st_size)
    except OSError: sig = None
    if sig == _CONTRIB_SIG[0]: return
    _CONTRIB_SIG[0] = sig
    if sig is None: return
    for k, r in wvcore.load_contributed(CONTRIB).items():
        if k not in CANDIDATES or CANDIDATES[k].get("source") == "contributed": CANDIDATES[k] = r
refresh_candidates()
BLOCK_DIR = Path(os.environ.get("WV_BLOCKLIST_DIR", ROOT / "data"))   # data/removed_*.txt: removed entirely (files stay on disk)
_BLOCK = {"sig": None, "entries": {}, "ids": set(), "dec_ver": -1, "dec": {}, "eff": set()}
_DEC_VER = [0]       # bumped whenever a kind=removal event is stored
def block_entries():
    """{id: [(file, reason comment)]} from data/removed_*.txt; reloaded whenever those files change (e.g. after git pull)."""
    try: sig = sorted((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in BLOCK_DIR.glob(wvcore.BLOCKLIST_GLOB))
    except OSError: sig = []
    if sig != _BLOCK["sig"]:
        ent = wvcore.load_blocklist_entries(BLOCK_DIR); _BLOCK.update(sig=sig, entries=ent, ids=set(ent), dec_ver=-1)
    return _BLOCK["entries"]
def removal_decisions():
    block_entries()
    if _BLOCK["dec_ver"] != _DEC_VER[0]:
        ver = _DEC_VER[0]
        with db() as c: evs = [row_event(r) for r in c.execute("SELECT * FROM events WHERE kind='removal' ORDER BY seq")]
        dec = wvcore.removal_decisions(evs)
        _BLOCK.update(dec=dec, eff=_BLOCK["ids"] - {i for i, d in dec.items() if d[0] == "restore"}, dec_ver=ver)
    return _BLOCK["dec"]
def blocked():
    """Blocklist files minus images a human restored in the Removed review (latest decision wins)."""
    removal_decisions(); return _BLOCK["eff"]
LOCK = threading.Lock()
BACKOFF = {}  # host -> unix time until which the image proxy will not contact it (after a 429)

def db():
    c = sqlite3.connect(DB_PATH, timeout=30); c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL"); return c

with db() as c:
    c.executescript("""
    CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, uuid TEXT UNIQUE NOT NULL, image_id TEXT NOT NULL,
        annotator TEXT NOT NULL, field TEXT NOT NULL, value TEXT, ts TEXT NOT NULL, device_id TEXT NOT NULL, kind TEXT NOT NULL,
        received_at TEXT NOT NULL, prov TEXT);
    CREATE INDEX IF NOT EXISTS ev_ann ON events(annotator, seq);
    CREATE INDEX IF NOT EXISTS ev_img ON events(image_id);
    CREATE TABLE IF NOT EXISTS assignments(image_id TEXT NOT NULL, annotator TEXT NOT NULL, assigned_at TEXT NOT NULL, round INTEGER NOT NULL,
        PRIMARY KEY(image_id, annotator));
    CREATE TABLE IF NOT EXISTS assignments_v11(image_id TEXT NOT NULL, annotator TEXT NOT NULL, assigned_at TEXT NOT NULL, round INTEGER NOT NULL,
        PRIMARY KEY(image_id, annotator));
    """)
    if "prov" not in [r[1] for r in c.execute("PRAGMA table_info(events)")]:
        c.execute("ALTER TABLE events ADD COLUMN prov TEXT")      # migration: review-mode provenance

def row_event(r):
    e = {"seq": r["seq"], "uuid": r["uuid"], "image_id": r["image_id"], "annotator": r["annotator"], "field": r["field"],
         "value": json.loads(r["value"]), "ts": r["ts"], "device_id": r["device_id"], "kind": r["kind"]}
    if r["prov"]: e.update(json.loads(r["prov"]))
    return e

# ---------- machine suggestions: annotations/auto/*.jsonl, reloaded whenever the files change (e.g. after git pull) ----------
_AUTO = {"sig": None, "data": {}, "active": "", "attr_active": "", "attr_order": []}
AUTO_ATTR_MODEL = os.environ.get("WV_AUTO_ATTR_MODEL", "").strip()                # v1.1 attribute suggestions (default: newest *_attr-*.jsonl)
def auto_state():
    try: files = sorted((p.name, p.stat().st_mtime_ns, p.stat().st_size) for d in (AUTO_DIR, REVIEW_AUTO_DIR) for p in d.glob("*.jsonl"))
    except OSError: files = []
    if files != _AUTO["sig"]:
        data = wvcore.load_auto(AUTO_DIR); data.update({k: v for k, v in wvcore.load_auto(REVIEW_AUTO_DIR).items() if k not in data})
        main = [f for f in files if not wvcore.is_attr_tag(f[0][:-6])]; attr = [f for f in files if wvcore.is_attr_tag(f[0][:-6])]
        newest = max(main, key=lambda f: f[1])[0][:-6] if main else ""
        newest_attr = max(attr, key=lambda f: f[1])[0][:-6] if attr else ""
        aa = AUTO_ATTR_MODEL if AUTO_ATTR_MODEL in data else newest_attr
        # every attr tag, active first then newest first: other assist models (VLM, ...) fill fields the active one lacks
        order = [aa] * bool(aa) + [f[0][:-6] for f in sorted(attr, key=lambda f: -f[1]) if f[0][:-6] != aa]
        _AUTO.update(sig=files, data=data, active=AUTO_MODEL if AUTO_MODEL in data else newest, attr_active=aa, attr_order=order)
    return _AUTO

def auto_rec(i, tag=None):
    st = auto_state(); return st["data"].get(tag or st["active"], {}).get(i)

def all_events(c):
    return [row_event(r) for r in c.execute("SELECT * FROM events ORDER BY seq")]

def insert_events(evs):
    refresh_candidates()
    acc, dup, rej = [], [], []
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with LOCK, db() as c:
        for ev in evs:
            e, why = wvcore.validate(ev)
            if not e: rej.append({"uuid": ev.get("uuid") if isinstance(ev, dict) else None, "reason": why}); continue
            if e["image_id"] not in CANDIDATES: rej.append({"uuid": e["uuid"], "reason": "unknown image_id"}); continue
            pv = wvcore.prov_of(e)
            cur = c.execute("INSERT OR IGNORE INTO events(uuid,image_id,annotator,field,value,ts,device_id,kind,received_at,prov) VALUES(?,?,?,?,?,?,?,?,?,?)",
                            (e["uuid"], e["image_id"], e["annotator"], e["field"], json.dumps(e["value"]), e["ts"], e["device_id"], e["kind"], now,
                             json.dumps(pv, sort_keys=True) if pv else None))
            (acc if cur.rowcount else dup).append(e["uuid"])
            if cur.rowcount and e["kind"] == "removal": _DEC_VER[0] += 1
    return {"accepted": acc, "duplicate": dup, "rejected": rej}

_BLUR = {"sig": None, "data": {}}
def blur_state():
    """Face boxes per image from the blur manifest(s); reloaded when the files change (git pull / a new auto_blur run)."""
    sig = []
    for p in BLUR_MANIFESTS:
        try: st = p.stat(); sig.append((str(p), st.st_mtime_ns, st.st_size))
        except OSError: pass
    if sig != _BLUR["sig"]:
        d = {}
        for p in BLUR_MANIFESTS:
            if p.exists():
                for line in open(p, encoding="utf-8"):
                    if line.strip():
                        try: r = json.loads(line); d[r["image_id"]] = r
                        except Exception: pass
        _BLUR.update(sig=sig, data=d)
    return _BLUR["data"]

def blurred_image(i):
    p = BLUR_DIR / f"{i}.jpg"
    return p if p.exists() else None

def image_meta(i):
    r = CANDIDATES[i]; b = blurred_image(i); rec = blur_state().get(i) if b else None
    m = {k: r.get(k, "") for k in ("candidate_id", "source", "proposed_class", "title", "creator", "license", "license_url",
                                   "source_page_url", "needs_visual_screen", "screen_reasons")} | {"image_id": i, "img": f"/img/{i}"}
    if b:   # distinct URL so clients that cached the original fetch the blurred copy
        m |= {"img": f"/img/{i}?b={(rec or {}).get('version', '1')}", "blurred": True,
              "faces": [f["box"] + [f.get("score", 0)] for f in (rec or {}).get("faces", [])], "blur_detector": (rec or {}).get("detector", "")}
    return m

def local_image(i):
    for p in (IMAGE_DIR / f"{i}.jpg", CACHE_DIR / f"{i}.jpg"):
        if p.exists(): return p
    return None

# ---------- optional Hugging Face persistence (ephemeral hosts) ----------
HF_DATASET = os.environ.get("WV_HF_DATASET", "").strip()
HF = None
if HF_DATASET:
    from hf_persist import HFPersistence
    if not os.environ.get("HF_TOKEN"): raise SystemExit("WV_HF_DATASET is set but HF_TOKEN is missing")
    HF = HFPersistence(HF_DATASET, os.environ["HF_TOKEN"], DB_PATH, DB_PATH.parent / "hfdata", IMAGE_DIR)

def images_ready():
    return HF is None or HF.state["images_ready"]

@asynccontextmanager
async def lifespan(app):
    if HF:
        HF.restore()          # raises -> server does not start (never push an empty copy over the remote)
        HF.start()
    yield
    if HF: HF.stop()

# ---------- access-code gate (one unlock per device, cookie-based) ----------
ACCESS_CODE = os.environ.get("WV_ACCESS_CODE", "").strip()
ACCESS_TOKEN = hmac.new(ACCESS_CODE.encode(), b"wheelview-access-v1", hashlib.sha256).hexdigest() if ACCESS_CODE else ""
OPEN_PATHS = {"/unlock", "/unlock.html", "/api/unlock", "/api/health", "/manifest.webmanifest", "/icon-192.png", "/icon-512.png", "/style.css", "/favicon.ico"}
FAILS = {}  # ip -> [window_start, count]

def has_access(req: Request):
    if not ACCESS_CODE: return True
    c = req.cookies.get("wv_access", ""); h = req.headers.get("x-access-code", "")
    return (c and hmac.compare_digest(c, ACCESS_TOKEN)) or (h and hmac.compare_digest(h, ACCESS_CODE))

app = FastAPI(title="WheelView Annotate", lifespan=lifespan)

@app.middleware("http")
async def gate(req: Request, call_next):
    if req.url.path in OPEN_PATHS or has_access(req): return await call_next(req)
    if req.method == "GET" and "text/html" in req.headers.get("accept", ""):
        return RedirectResponse("/unlock.html", status_code=303)
    return JSONResponse({"detail": "access code required", "unlock": "/unlock.html"}, status_code=401)

@app.post("/api/unlock")
async def unlock(req: Request):
    ip = (req.headers.get("x-forwarded-for") or (req.client.host if req.client else "?")).split(",")[0].strip()
    w = FAILS.get(ip)
    if w and time.time() - w[0] < 600 and w[1] >= 10: raise HTTPException(429, "too many attempts; try again in 10 minutes")
    body = await req.json() if req.headers.get("content-type", "").startswith("application/json") else {}
    code = str(body.get("code", "")).strip().upper()
    if not ACCESS_CODE: return {"ok": True}
    if not hmac.compare_digest(code, ACCESS_CODE.upper()):
        if not w or time.time() - w[0] >= 600: FAILS[ip] = [time.time(), 1]
        else: w[1] += 1
        raise HTTPException(403, "wrong access code")
    FAILS.pop(ip, None)
    secure = req.headers.get("x-forwarded-proto", req.url.scheme) == "https"
    r = JSONResponse({"ok": True})
    # SameSite=None so it also works inside the huggingface.co Space iframe; Secure is required for that.
    r.set_cookie("wv_access", ACCESS_TOKEN, max_age=400 * 86400, httponly=True, secure=secure, samesite="none" if secure else "lax", path="/")
    return r

@app.get("/api/health")
def health():
    with db() as c: n = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    a = auto_state()
    out = {"ok": True, "events": n, "candidates": len(CANDIDATES), "blocked": len(blocked() & CANDIDATES.keys()),
           "restored": sum(1 for i, d in removal_decisions().items() if d[0] == "restore" and i in block_entries()), "images_ready": images_ready(),
           "features": ["review", "blur", "attr_v11", "contributed", "filter", "multi_fields", "assist_models"], "blurred_images": sum(1 for _ in BLUR_DIR.glob("*.jpg")) if BLUR_DIR.exists() else 0, "auto": {"active": a["active"], "attr_active": a["attr_active"], "models": {t: len(r) for t, r in a["data"].items()}},
           "contributed": sum(1 for r in CANDIDATES.values() if r.get("source") == "contributed")}
    if HF: out["hf"] = {k: v for k, v in HF.state.items()}
    return out

@app.post("/api/admin/flush")
def flush():
    """Export + commit events to the HF dataset now (access-gated like every other endpoint)."""
    if not HF: return {"ok": False, "detail": "HF persistence not enabled"}
    HF.push_now(); return {"ok": True, "state": HF.state}

@app.get("/api/schema")
def schema(): return wvcore.SCHEMA

@app.post("/api/events")
async def post_events(req: Request):
    body = await req.json()
    evs = body.get("events") if isinstance(body, dict) else None
    if not isinstance(evs, list) or len(evs) > 2000: raise HTTPException(400, "expected {events:[...]} (max 2000)")
    return insert_events(evs)

@app.get("/api/events")
def get_events(since: int = 0, annotator: str | None = None, image_id: str | None = None, limit: int = Query(5000, le=20000)):
    q, a = "SELECT * FROM events WHERE seq > ?", [since]
    if annotator: q += " AND annotator = ?"; a.append(annotator)
    if image_id: q += " AND image_id = ?"; a.append(image_id)
    with db() as c: rows = [row_event(r) for r in c.execute(q + " ORDER BY seq LIMIT ?", a + [limit])]
    return {"events": rows, "max_seq": rows[-1]["seq"] if rows else since}

@app.get("/api/images/{image_id}")
def get_image(image_id: str):
    if image_id not in CANDIDATES or image_id in blocked(): raise HTTPException(404)
    return image_meta(image_id)

def statuses11(c):
    per, adjv = wvcore.fold(all_events(c))
    return per, adjv, {i: wvcore.image_status_v11(i, per, adjv) for i in set(per) | {k[0] for k in adjv}}

def statuses(c):
    per, adjv = wvcore.fold(all_events(c))
    return {i: wvcore.image_status(i, per, adjv) for i in set(per) | {k[0] for k in adjv}}

def order_key(i):  # deterministic shuffle so batches mix classes/sources
    return hashlib.sha1(i.encode()).hexdigest()

@app.get("/api/assign")
def assign(annotator: str, n: int = Query(30, ge=1, le=200)):
    refresh_candidates()
    ann = wvcore.slug(annotator)
    if not ann or ann != annotator: raise HTTPException(400, "annotator must be a slug (lowercase, a-z0-9._-)")
    if not images_ready(): raise HTTPException(503, "server is still downloading images; try again in a minute")
    maxn = wvcore.SCHEMA["max_annotators_per_image"]; base = wvcore.SCHEMA["annotators_per_image"]
    with LOCK, db() as c:
        st = statuses(c)
        asg = {}
        for r in c.execute("SELECT image_id, annotator FROM assignments"): asg.setdefault(r[0], set()).add(r[1])
        B = blocked()
        mine_open = [i for i, s in asg.items() if ann in s and ann not in st.get(i, {}).get("done", []) and i not in B]
        out = list(mine_open)
        eligible = (lambda i: i not in B and (ASSIGN_NO_IMAGE or local_image(i) is not None))
        # 1) third annotator for disagreements; 2) second annotator; 3) fresh images
        third = [i for i, s in st.items() if s["status"] == "needs_third_annotator" and ann not in asg.get(i, set())
                 and ann not in s["done"] and len(asg.get(i, set())) < maxn]
        second = [i for i, s in asg.items() if len(s) < base and ann not in s]
        fresh = [i for i in CANDIDATES if i not in asg]
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for pool, rnd in ((sorted(third, key=order_key), 3), (sorted(second, key=order_key), 2), (sorted(fresh, key=order_key), 1)):
            for i in pool:
                if len(out) >= n: break
                if i in out or not eligible(i): continue
                c.execute("INSERT OR IGNORE INTO assignments VALUES(?,?,?,?)", (i, ann, now, rnd)); out.append(i)
    return {"annotator": ann, "images": [image_meta(i) for i in out[:max(n, len(mine_open))]]}

@app.get("/api/auto")
def auto_info():
    a = auto_state(); meta = {}
    for t in a["data"]:
        mf = AUTO_DIR / f"{t}.meta.json"
        if mf.exists():
            try: m = json.load(open(mf, encoding="utf-8")); meta[t] = {k: m.get(k) for k in ("classifier", "detector", "prompt_version", "images", "generated_at", "screening_thresholds")}
            except Exception: pass
    return {"active": a["active"], "models": {t: len(r) for t, r in a["data"].items()}, "meta": meta}

@app.get("/api/auto/{image_id}")
def auto_one(image_id: str, model_tag: str | None = None):
    r = auto_rec(image_id, model_tag)
    if not r: raise HTTPException(404, "no machine suggestion for this image")
    return r

def conf_key(rec):   # "lowest confidence first" uses the device_type confidence (the field that matters most)
    return rec["suggestions"].get("device_type", {}).get("confidence", 1.0)

REVIEW_ORDERS = ("lowconf", "wheelchair", "random")

@app.get("/api/review/assign")
def review_assign(annotator: str, n: int = Query(30, ge=1, le=200), order: str = "lowconf", model_tag: str | None = None):
    """A batch for 'Review auto labels': images WITH machine suggestions, skipping images this annotator has done.
    Pair formation first (needs-third, then needs-a-second-human), then fresh images, each tier in the chosen order.
    Uses the same assignments table as /api/assign, so review and normal batches never hand out a 3rd/4th label
    by accident. Every image comes with its suggestion record (cached by the client for offline review)."""
    ann = wvcore.slug(annotator)
    if not ann or ann != annotator: raise HTTPException(400, "annotator must be a slug (lowercase, a-z0-9._-)")
    if order not in REVIEW_ORDERS: raise HTTPException(400, f"order must be one of {REVIEW_ORDERS}")
    if not images_ready(): raise HTTPException(503, "server is still downloading images; try again in a minute")
    a = auto_state(); tag = model_tag or a["active"]; recs = a["data"].get(tag, {})
    if not recs: raise HTTPException(404, "no machine suggestions available on this server (annotations/auto is empty)")
    maxn = wvcore.SCHEMA["max_annotators_per_image"]; base = wvcore.SCHEMA["annotators_per_image"]
    def key(i):
        if order == "lowconf": return (conf_key(recs[i]), order_key(i))
        if order == "wheelchair": return (CANDIDATES[i].get("proposed_class") != "wheelchair_unspecified", conf_key(recs[i]), order_key(i))
        return (order_key(i),)
    with LOCK, db() as c:
        st = statuses(c)
        asg = {}
        for r in c.execute("SELECT image_id, annotator FROM assignments"): asg.setdefault(r[0], set()).add(r[1])
        mine_done = {i for i, s in st.items() if ann in s["done"]}; B = blocked()
        ok = lambda i: i in recs and i not in mine_done and i not in B and (ASSIGN_NO_IMAGE or local_image(i) is not None)
        third = [i for i, s in st.items() if s["status"] == "needs_third_annotator" and ann not in asg.get(i, set()) and len(asg.get(i, set())) < maxn]
        # a second human: someone else already finished (or holds) it, it still needs another label, and it is not mine
        second = [i for i in recs if ann not in asg.get(i, set()) and 0 < len(asg.get(i, set()) | set(st.get(i, {}).get("done", []))) < base]
        # finished-by-someone first (a pair completes as soon as I am done), then merely held by someone
        # (an image someone finished still counts even if another person holds it but never finished: up to maxn holders)
        dn = lambda i: set(st.get(i, {}).get("done", []))
        done_one = [i for i in recs if ann not in asg.get(i, set()) and 0 < len(dn(i)) < base and len(asg.get(i, set()) | dn(i)) < maxn]
        held = [i for i in second if not dn(i)]
        mine_open = [i for i, s in asg.items() if ann in s]
        fresh = [i for i in recs if i not in asg and not st.get(i, {}).get("done")]
        out, now = [], time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for pool, rnd in ((third, 3), (done_one, 2), (held, 2), (mine_open, 0), (fresh, 1)):
            for i in sorted(pool, key=key):
                if len(out) >= n: break
                if i in out or not ok(i): continue
                if rnd: c.execute("INSERT OR IGNORE INTO assignments VALUES(?,?,?,?)", (i, ann, now, rnd))
                out.append(i)
    return {"annotator": ann, "model_tag": tag, "order": order,
            "images": [image_meta(i) | {"auto": recs[i]} for i in out]}

# ---------- v1.1 attribute pass: control / environment / weather / ground_wet / size_class ----------
CONTEXT_FIELDS = ("device_type", "occupied", "propulsion", "setting", "lighting", "viewpoint")
def _release_ids():
    p = ROOT / "data" / "manifest_v2.csv"
    if not p.exists(): return set()
    with open(p, encoding="utf-8", newline="") as fh:
        return {r["image_id"] for r in csv.DictReader(fh) if r.get("train_eligible") == "1" or r.get("split") == "excluded"}

def v11_flagged(img, per):
    """Exclude / face-blur flags raised by anyone (v1.0 or v1.1 pass) that the v1.0 rules would not see: keep the image
    out of the attribute pass until the normal removal review has handled it."""
    for a, fv in per.get(img, {}).items():
        if fv.get("exclude") not in (None, "", "none") or fv.get("face_blur_missed") is True: return True
    return False

@app.get("/api/attr/assign")
def attr_assign(annotator: str, n: int = Query(30, ge=1, le=200)):
    """A batch for the v1.1 attribute pass. Pool: images in the v1.0 clean set (>=1 human done, not labeler-excluded,
    not on a safety hold / blocklist) incl. contributed images once they have their v1.0 labels. Order: third v1.1
    annotator for disagreements, then a second v1.1 annotator, then fresh -- release (manifest_v2) images first.
    Each image comes with the v1.0 resolved labels as context and the v1.1 machine suggestion (shown, never saved)."""
    refresh_candidates()
    ann = wvcore.slug(annotator)
    if not ann or ann != annotator: raise HTTPException(400, "annotator must be a slug (lowercase, a-z0-9._-)")
    if not images_ready(): raise HTTPException(503, "server is still downloading images; try again in a minute")
    a = auto_state(); order = a.get("attr_order") or []
    maxn = wvcore.SCHEMA["max_annotators_per_image"]; base = wvcore.SCHEMA.get("annotators_per_image_v11", 2)
    hold = wvcore.load_safety_hold(str(BLOCK_DIR)); rel = _release_ids()
    with LOCK, db() as c:
        per, adjv, st11 = statuses11(c)
        asg = {}
        for r in c.execute("SELECT image_id, annotator FROM assignments_v11"): asg.setdefault(r[0], set()).add(r[1])
        B = blocked()
        def clean(i):
            if i not in CANDIDATES or i in B or v11_flagged(i, per): return False
            ok, _ = wvcore.clean_decision(i, per, adjv, hold=hold); return ok
        ok = lambda i: clean(i) and (ASSIGN_NO_IMAGE or local_image(i) is not None)
        done11 = lambda i: set(st11.get(i, {}).get("done", []))
        mine_open = [i for i, s in asg.items() if ann in s and ann not in done11(i)]
        third = [i for i, s in st11.items() if s["status"] == "needs_third_annotator" and ann not in asg.get(i, set()) and ann not in s["done"] and len(asg.get(i, set())) < maxn]
        second = [i for i, s in asg.items() if len(s | done11(i)) < base and ann not in s and ann not in done11(i)]
        fresh = [i for i in per if i not in asg and not done11(i)]
        key = lambda i: (i not in rel, order_key(i))
        out, now = [], time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for pool, rnd in ((mine_open, 0), (sorted(third, key=key), 3), (sorted(second, key=key), 2), (sorted(fresh, key=key), 1)):
            for i in pool:
                if len(out) >= n: break
                if i in out or not ok(i): continue
                if rnd: c.execute("INSERT OR IGNORE INTO assignments_v11 VALUES(?,?,?,?)", (i, ann, now, rnd))
                out.append(i)
    imgs = []
    for i in out:
        s10 = wvcore.image_status(i, per, adjv)
        ctx = {f: s10["fields"][f][0] for f in CONTEXT_FIELDS if f in s10["fields"] and s10["fields"][f][0]}
        m = image_meta(i) | {"context": ctx}
        sg = wvcore.merged_suggestions(i, a["data"], order)   # per field: active attr model first, then other attr models
        if sg: m["auto11"] = {"image_id": i, "model_tag": a["attr_active"] or order[0], "suggestions": sg}
        dv = (auto_rec(i) or {}).get("suggestions", {}).get("device_type")   # v1.0 model's device guess: offers subtypes
        if dv and dv.get("value"): m["auto_device"] = {"value": wvcore.canon("device_type", dv["value"]), "confidence": dv.get("confidence"), "model_tag": a["active"]}
        imgs.append(m)
    return {"annotator": ann, "model_tag": a["attr_active"], "images": imgs}

# ---------- filtering (logic: wvcore.parse_filter / image_facts / run_filter; same as scripts/filter_images.py) ----------
FILTER_CTX = {"manifest_v11": Path(os.environ.get("WV_MANIFEST_V11", ROOT / "data" / "manifest_v11.csv")),
              "manifest_v2": Path(os.environ.get("WV_MANIFEST_V2", ROOT / "data" / "manifest_v2.csv")),
              "synthetic": Path(os.environ.get("WV_SYNTHETIC_MANIFEST", ROOT / "var" / "synthetic" / "synthetic_manifest.csv")),
              "measurements": Path(os.environ.get("WV_MEASUREMENTS", ROOT / "data" / "measurements.csv")),
              "size_estimates": Path(os.environ.get("WV_SIZE_ESTIMATES", ROOT / "var" / "v11" / "size_estimates.csv"))}
_FCTX = {"sig": None, "data": None}
def filter_context():
    sig = tuple((str(p), p.stat().st_mtime_ns if p.exists() else 0) for p in FILTER_CTX.values())
    if sig != _FCTX["sig"]: _FCTX.update(sig=sig, data=wvcore.load_filter_context(**{k: str(v) for k, v in FILTER_CTX.items()}))
    return _FCTX["data"]

def _filter(q: str, ids: list | None, limit: int, counts: bool):
    try: flt, basis = wvcore.parse_filter(q)
    except ValueError as e: raise HTTPException(400, str(e))
    refresh_candidates(); B = blocked()
    with db() as c: per, adjv = wvcore.fold(all_events(c))
    a = auto_state(); prefer = [a["active"]] * bool(a["active"]) + (a.get("attr_order") or [])
    pool = [i for i in (ids if ids is not None else sorted(CANDIDATES, key=order_key)) if i in CANDIDATES and i not in B]
    ctx = filter_context()
    facts = {i: wvcore.image_facts(i, per, adjv, wvcore.merged_suggestions(i, a["data"], prefer), ctx) for i in pool}
    res = wvcore.run_filter(pool, facts, flt, basis, counts=counts)
    res["truncated"] = len(res["ids"]) > limit; res["ids"] = res["ids"][:limit]
    return res

@app.get("/api/filter")
def filter_get(q: str = "", limit: int = Query(500, ge=0, le=20000), counts: bool = True):
    """Images matching a filter over every served image: {total, count, ids (first `limit`), counts, match_counts}.
    q: 'device_type:manual,powered;weather:rain;basis:human' -- OR within a field, AND across fields."""
    return _filter(q, None, limit, counts)

@app.post("/api/filter")
async def filter_post(req: Request):
    """{q, ids?: [...]} -> same as GET, restricted to `ids` (the app sends its cached queue)."""
    b = await req.json(); ids = b.get("ids")
    if ids is not None and (not isinstance(ids, list) or len(ids) > 20000 or not all(isinstance(i, str) for i in ids)): raise HTTPException(400, "ids must be a list of image ids")
    return _filter(str(b.get("q") or ""), ids, int(b.get("limit") or 20000), bool(b.get("counts", True)))

@app.get("/api/filter/keys")
def filter_keys():
    """Filter keys and their selectable values (for UIs)."""
    return {"keys": {k: wvcore.filter_values(k) for k in wvcore.FILTER_KEYS}, "pseudo": list(wvcore.PSEUDO_KEYS),
            "bases": list(wvcore.FILTER_BASES), "special": [wvcore.UNLABELED, wvcore.MODEL_ONLY]}

@app.get("/api/assist/status")
def assist_status():
    """Which annotation-assist models (scripts/assist_models.py) are configured and usable; never loads weights."""
    import importlib, sys as _sys
    sp = str(ROOT / "scripts")
    if sp not in _sys.path: _sys.path.insert(0, sp)
    try: return importlib.import_module("assist_models").status()
    except Exception as e: return {"error": type(e).__name__}

@app.get("/api/conflicts")
def conflicts():
    with db() as c: per, adjv, st11 = statuses11(c)
    st = {i: wvcore.image_status(i, per, adjv) for i in set(per) | {k[0] for k in adjv}}
    rows = []; B = blocked()
    for i in sorted(set(st) | set(st11)):
        s = st.get(i); s11 = st11.get(i)
        cf = {f: s["fields"][f][2] for f in s["conflicts"]} if s else {}
        if s11 and s11["status"] == "needs_adjudication":   # v1.1 conflicts only once 3 people disagree (or 2 at max)
            cf |= {f: s11["fields"][f][2] for f in s11["conflicts"]}
        if cf and i not in B and i in CANDIDATES:
            rows.append(image_meta(i) | {"status": s["status"] if s and s["conflicts"] else "v1.1:" + s11["status"],
                        "done": sorted(set((s or {}).get("done", [])) | set((s11 or {}).get("done", []))), "conflicts": cf})
    return {"conflicts": rows}

def load_adj_queue(path=None):
    """Read the review-request queue (JSONL). Items are concerns for an adjudicator, never labels or decisions.
    Each line needs image_id, reason, source; later lines with the same (image_id, reason, source) replace earlier ones."""
    path = Path(path or ADJ_QUEUE); out = {}
    if not path.exists(): return []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except ValueError: continue
        if not isinstance(r, dict) or not all(isinstance(r.get(k), str) and r.get(k) for k in ("image_id", "reason", "source")): continue
        out[(r["image_id"], r["reason"], r["source"])] = r | {"line": n}
    return list(out.values())

@app.get("/api/adjudication_queue")
def adjudication_queue(reason: str | None = None, source: str | None = None):
    """Queued concerns joined with the current human device_type labels. Read-only: resolving is done with normal
    adjudication/label events in the UI; this endpoint never writes events."""
    items = load_adj_queue(); B = blocked()
    with db() as c: evs = all_events(c)
    human = {}
    for e in evs:
        if e.get("field") == "device_type" and e.get("kind", "label") in ("label", "adjudication"):
            human.setdefault(e["image_id"], {})[("adj:" if e.get("kind") == "adjudication" else "") + e["annotator"]] = e["value"]
    counts = {}
    for r in items: counts[r["reason"]] = counts.get(r["reason"], 0) + 1
    rows = []
    for r in items:
        if (reason and r["reason"] != reason) or (source and r["source"] != source): continue
        i = r["image_id"]
        meta = image_meta(i) if i in CANDIDATES and i not in B else {"image_id": i, "img": ""}
        rows.append(meta | {k: v for k, v in r.items() if k != "img"} | {"blocked": i in B, "human_device_type": human.get(i, {})})
    return {"items": rows, "counts": counts, "total": len(items)}

@app.get("/api/stats")
def stats():
    with db() as c: evs = all_events(c)
    stat, k, prog = wvcore.materialize(evs, CANDIDATES, EXPORT_DIR, auto=auto_state()["data"], blocked=blocked())
    with db() as c: prog["assignments"] = c.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
    out = {"progress": prog, "kappa": k}
    kp = EXPORT_DIR / "kappa_v11.json"
    if wvcore.has_v11(evs) and kp.exists(): out["kappa_v11"] = json.load(open(kp, encoding="utf-8"))
    return out

@app.get("/export/{name}")
def export(name: str):
    allowed = {"labels.csv", "per_annotator_labels.csv", "adjudication_needed.csv", "kappa.json", "progress.json", "auto_vs_human.json",
               "labels_v11.csv", "per_annotator_labels_v11.csv", "adjudication_needed_v11.csv", "kappa_v11.json", "auto_vs_human_v11.json"}
    if name not in allowed: raise HTTPException(404)
    with db() as c: evs = all_events(c)
    wvcore.materialize(evs, CANDIDATES, EXPORT_DIR, auto=auto_state()["data"], blocked=blocked())
    if not (EXPORT_DIR / name).exists(): raise HTTPException(404, "not generated yet (no v1.1 labels)")
    return FileResponse(EXPORT_DIR / name, media_type="text/csv" if name.endswith(".csv") else "application/json",
                        headers={"Cache-Control": "no-store"})

# ---------- Removed review: a human goes through blocklisted images and restores or confirms each one ----------
RULES = ("child", "stroller", "no_device", "human_exclude")
def entry_rules(srcs):
    """Rule keys from the inline comments ('child: ...; stroller: ...'); falls back to the list file name."""
    out = set()
    for fn, c in srcs:
        keys = {seg.split(":", 1)[0].strip() for seg in c.split(";") if ":" in seg} & set(RULES)
        out |= keys or {fn[len("removed_"):-4]}
    return sorted(out)

@app.get("/api/removed")
def removed_list(rule: str | None = None, status: str | None = None):
    """Every blocklisted image with its list(s), reason/scores and the latest human decision.
    status: undecided | restore | confirm_removed (default: all)."""
    ent = block_entries(); dec = removal_decisions(); rows = []; counts = {"rule": {}, "status": {}}
    for i, srcs in sorted(ent.items(), key=lambda kv: order_key(kv[0])):
        if i not in CANDIDATES: continue
        rules = entry_rules(srcs); d = dec.get(i); stv = d[0] if d else "undecided"
        for r in rules: counts["rule"][r] = counts["rule"].get(r, 0) + 1
        counts["status"][stv] = counts["status"].get(stv, 0) + 1
        if rule and rule not in rules: continue
        if status and status != stv: continue
        c = CANDIDATES[i]
        rows.append({"image_id": i, "rules": rules, "lists": [fn for fn, _ in srcs], "reason": " | ".join(x for _, x in srcs if x),
                     "status": stv, "decided_by": d[1] if d else "", "decided_at": d[2] if d else "",
                     "has_blurred": blurred_image(i) is not None, "proposed_class": c.get("proposed_class", ""), "source": c.get("source", ""),
                     "title": c.get("title", ""), "img": f"/api/removed/img/{i}"})
    return {"total": len(rows), "counts": counts, "images": rows}

@app.post("/api/removed/decision")
async def removed_decision(req: Request):
    """{annotator, image_id, action: restore|confirm_removed} -> append-only kind=removal event (exported to
    annotations/events/<annotator>.jsonl by scripts/sync_events.py like every other event). Latest decision wins."""
    b = await req.json(); ann = b.get("annotator", ""); i = b.get("image_id", ""); act = b.get("action", "")
    if not ann or wvcore.slug(ann) != ann: raise HTTPException(400, "annotator must be a slug (lowercase, a-z0-9._-)")
    ent = block_entries()
    if i not in ent: raise HTTPException(404, "not on a removal list")
    if act not in wvcore.REMOVAL_ACTIONS: raise HTTPException(400, f"action must be one of {wvcore.REMOVAL_ACTIONS}")
    srcs = ent[i]
    ev = {"uuid": str(uuid.uuid4()), "image_id": i, "annotator": ann, "field": "removal", "kind": "removal",
          "value": {"action": act, "list": ",".join(fn for fn, _ in srcs)[:400], "reason": " | ".join(x for _, x in srcs if x)[:400]},
          "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int(time.time() * 1000) % 1000:03d}Z", "device_id": "removed-review-web"}
    r = insert_events([ev])
    if r["rejected"]: raise HTTPException(400, r["rejected"][0]["reason"])
    d = removal_decisions().get(i)
    return {"ok": True, "image_id": i, "status": d[0] if d else "undecided", "blocked": i in blocked(), "event": ev["uuid"]}

@app.get("/api/removed/img/{image_id}")
def removed_img(image_id: str):
    """Face-BLURRED copy of a blocklisted image for the Removed review (access-gated). Originals are never served here."""
    if image_id not in block_entries(): raise HTTPException(404)
    p = blurred_image(image_id)
    if not p: raise HTTPException(404, "no blurred copy (the unblurred original is never shown in the Removed review)")
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "private, no-store"})

@app.get("/img/{image_id}")
def img(image_id: str):
    if image_id not in CANDIDATES or image_id in blocked(): raise HTTPException(404)
    p = blurred_image(image_id) or local_image(image_id)      # annotators see the face-blurred release copy when one exists
    if p: return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=604800"})
    if not PROXY: raise HTTPException(404, "image not available locally (proxy disabled)")
    url = CANDIDATES[image_id].get("image_url") or ""
    if not url.startswith("https://"): raise HTTPException(404)
    host = url.split("/")[2]
    if BACKOFF.get(host, 0) > time.time(): raise HTTPException(503, f"{host} rate-limited us; backing off")
    try:
        from PIL import Image
        r = httpx.get(url, headers=UA, timeout=30, follow_redirects=True)
        if r.status_code == 429: BACKOFF[host] = time.time() + 600
        if r.status_code != 200: raise HTTPException(502, f"upstream {r.status_code}")
        im = Image.open(io.BytesIO(r.content)).convert("RGB"); im.thumbnail((640, 640))
        out = CACHE_DIR / f"{image_id}.jpg"; im.save(out, "JPEG", quality=88)   # re-encode: drops EXIF
        return FileResponse(out, media_type="image/jpeg")
    except HTTPException: raise
    except Exception as e: raise HTTPException(502, f"proxy error: {e}")

# PWA (served last so /api, /img, /export win). sw.js must be served from / for full scope.
app.mount("/", StaticFiles(directory=ROOT / "app", html=True), name="app")
