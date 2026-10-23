"""WheelView annotation server: FastAPI + SQLite. Serves the PWA, images, the append-only event log,
assignment (2 independent annotators, a 3rd on disagreement), adjudication and derived exports."""
import csv, hashlib, io, json, os, sqlite3, threading, time
from pathlib import Path
import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
import wvcore

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("WV_DB", ROOT / "var" / "wheelview.db"))
CANDS = Path(os.environ.get("WV_CANDIDATES", ROOT / "data" / "candidates.csv"))
IMAGE_DIR = Path(os.environ.get("WV_IMAGE_DIR", ROOT / "images"))          # <image_id>.jpg, filled by scripts/fetch_images.py
CACHE_DIR = Path(os.environ.get("WV_CACHE_DIR", ROOT / "var" / "imgcache"))
EXPORT_DIR = Path(os.environ.get("WV_EXPORT_DIR", ROOT / "var" / "exports"))
PROXY = os.environ.get("WV_IMAGE_PROXY", "1") == "1"          # fetch missing images (e.g. Commons) from image_url and cache them
ASSIGN_NO_IMAGE = os.environ.get("WV_ASSIGN_WITHOUT_LOCAL_IMAGE", "0") == "1"  # default: only assign images we can already serve
UA = {"User-Agent": "WheelViewAnnotate/0.1 (UC Merced Mi3 Lab; research dataset annotation)"}
for d in (DB_PATH.parent, CACHE_DIR, EXPORT_DIR): d.mkdir(parents=True, exist_ok=True)

CANDIDATES = wvcore.load_candidates(CANDS)
LOCK = threading.Lock()
BACKOFF = {}  # host -> unix time until which the image proxy will not contact it (after a 429)

def db():
    c = sqlite3.connect(DB_PATH, timeout=30); c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL"); return c

with db() as c:
    c.executescript("""
    CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, uuid TEXT UNIQUE NOT NULL, image_id TEXT NOT NULL,
        annotator TEXT NOT NULL, field TEXT NOT NULL, value TEXT, ts TEXT NOT NULL, device_id TEXT NOT NULL, kind TEXT NOT NULL,
        received_at TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS ev_ann ON events(annotator, seq);
    CREATE INDEX IF NOT EXISTS ev_img ON events(image_id);
    CREATE TABLE IF NOT EXISTS assignments(image_id TEXT NOT NULL, annotator TEXT NOT NULL, assigned_at TEXT NOT NULL, round INTEGER NOT NULL,
        PRIMARY KEY(image_id, annotator));
    """)

def row_event(r):
    return {"seq": r["seq"], "uuid": r["uuid"], "image_id": r["image_id"], "annotator": r["annotator"], "field": r["field"],
            "value": json.loads(r["value"]), "ts": r["ts"], "device_id": r["device_id"], "kind": r["kind"]}

def all_events(c):
    return [row_event(r) for r in c.execute("SELECT * FROM events ORDER BY seq")]

def insert_events(evs):
    acc, dup, rej = [], [], []
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with LOCK, db() as c:
        for ev in evs:
            e, why = wvcore.validate(ev)
            if not e: rej.append({"uuid": ev.get("uuid") if isinstance(ev, dict) else None, "reason": why}); continue
            if e["image_id"] not in CANDIDATES: rej.append({"uuid": e["uuid"], "reason": "unknown image_id"}); continue
            cur = c.execute("INSERT OR IGNORE INTO events(uuid,image_id,annotator,field,value,ts,device_id,kind,received_at) VALUES(?,?,?,?,?,?,?,?,?)",
                            (e["uuid"], e["image_id"], e["annotator"], e["field"], json.dumps(e["value"]), e["ts"], e["device_id"], e["kind"], now))
            (acc if cur.rowcount else dup).append(e["uuid"])
    return {"accepted": acc, "duplicate": dup, "rejected": rej}

def image_meta(i):
    r = CANDIDATES[i]
    return {k: r.get(k, "") for k in ("candidate_id", "source", "proposed_class", "title", "creator", "license", "license_url",
                                      "source_page_url", "needs_visual_screen", "screen_reasons")} | {"image_id": i, "img": f"/img/{i}"}

def local_image(i):
    for p in (IMAGE_DIR / f"{i}.jpg", CACHE_DIR / f"{i}.jpg"):
        if p.exists(): return p
    return None

app = FastAPI(title="WheelView Annotate")

@app.get("/api/health")
def health():
    with db() as c: n = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    return {"ok": True, "events": n, "candidates": len(CANDIDATES)}

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
    if image_id not in CANDIDATES: raise HTTPException(404)
    return image_meta(image_id)

def statuses(c):
    per, adjv = wvcore.fold(all_events(c))
    return {i: wvcore.image_status(i, per, adjv) for i in set(per) | {k[0] for k in adjv}}

def order_key(i):  # deterministic shuffle so batches mix classes/sources
    return hashlib.sha1(i.encode()).hexdigest()

@app.get("/api/assign")
def assign(annotator: str, n: int = Query(30, ge=1, le=200)):
    ann = wvcore.slug(annotator)
    if not ann or ann != annotator: raise HTTPException(400, "annotator must be a slug (lowercase, a-z0-9._-)")
    maxn = wvcore.SCHEMA["max_annotators_per_image"]; base = wvcore.SCHEMA["annotators_per_image"]
    with LOCK, db() as c:
        st = statuses(c)
        asg = {}
        for r in c.execute("SELECT image_id, annotator FROM assignments"): asg.setdefault(r[0], set()).add(r[1])
        mine_open = [i for i, s in asg.items() if ann in s and ann not in st.get(i, {}).get("done", [])]
        out = list(mine_open)
        eligible = (lambda i: ASSIGN_NO_IMAGE or local_image(i) is not None)
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

@app.get("/api/conflicts")
def conflicts():
    with db() as c: st = statuses(c)
    rows = []
    for i, s in sorted(st.items()):
        if s["conflicts"]:
            rows.append(image_meta(i) | {"status": s["status"], "done": s["done"],
                        "conflicts": {f: s["fields"][f][2] for f in s["conflicts"]}})
    return {"conflicts": rows}

@app.get("/api/stats")
def stats():
    with db() as c: evs = all_events(c)
    stat, k, prog = wvcore.materialize(evs, CANDIDATES, EXPORT_DIR)
    with db() as c: prog["assignments"] = c.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
    return {"progress": prog, "kappa": k}

@app.get("/export/{name}")
def export(name: str):
    allowed = {"labels.csv", "per_annotator_labels.csv", "adjudication_needed.csv", "kappa.json", "progress.json"}
    if name not in allowed: raise HTTPException(404)
    with db() as c: evs = all_events(c)
    wvcore.materialize(evs, CANDIDATES, EXPORT_DIR)
    return FileResponse(EXPORT_DIR / name, media_type="text/csv" if name.endswith(".csv") else "application/json",
                        headers={"Cache-Control": "no-store"})

@app.get("/img/{image_id}")
def img(image_id: str):
    if image_id not in CANDIDATES: raise HTTPException(404)
    p = local_image(image_id)
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
