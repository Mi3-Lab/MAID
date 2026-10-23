#!/usr/bin/env python3
"""Bridge between the server's SQLite event log and git.
  export : append server events to annotations/events/<annotator>.jsonl (append-only, dedup by uuid, server order)
  import : load every annotations/events/*.jsonl into the server DB (INSERT OR IGNORE by uuid -> idempotent)
  sync   : import, export, rebuild derived/ locally (not committed), git add + commit annotations/events,
           and `git push` only with --push AND a configured remote.
Each annotator only ever appends to their own file, so concurrent work on different branches merges without conflicts.
Adjudication events live in the adjudicator's own file (kind = "adjudication")."""
import argparse, glob, json, os, sqlite3, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server")); import wvcore
KEYS = ["uuid", "image_id", "annotator", "field", "value", "ts", "device_id", "kind"]

def con(db):
    c = sqlite3.connect(db); c.row_factory = sqlite3.Row; return c

def existing_uuids(fn):
    s = set()
    if os.path.exists(fn):
        for line in open(fn):
            if line.strip():
                try: s.add(json.loads(line)["uuid"])
                except Exception: pass
    return s

def export(db, evdir, only=None):
    os.makedirs(evdir, exist_ok=True); n = 0
    with con(db) as c:
        anns = [r[0] for r in c.execute("SELECT DISTINCT annotator FROM events ORDER BY annotator")]
        for a in anns:
            if only and a != only: continue
            fn = os.path.join(evdir, f"{a}.jsonl"); have = existing_uuids(fn)
            new = [r for r in c.execute("SELECT * FROM events WHERE annotator=? ORDER BY seq", (a,)) if r["uuid"] not in have]
            if not new: continue
            with open(fn, "a") as fh:
                for r in new:
                    e = {k: r[k] for k in KEYS}; e["value"] = json.loads(e["value"])
                    fh.write(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n"); n += 1
    return n

def import_(db, evdir):
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); added = bad = 0
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    with con(db) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, uuid TEXT UNIQUE NOT NULL, image_id TEXT NOT NULL,
            annotator TEXT NOT NULL, field TEXT NOT NULL, value TEXT, ts TEXT NOT NULL, device_id TEXT NOT NULL, kind TEXT NOT NULL, received_at TEXT NOT NULL)""")
        for fn in sorted(glob.glob(os.path.join(evdir, "*.jsonl"))):
            for line in open(fn):
                if not line.strip(): continue
                e, why = wvcore.validate(json.loads(line))
                if not e or e["annotator"] != Path(fn).stem: bad += 1; continue
                cur = c.execute("INSERT OR IGNORE INTO events(uuid,image_id,annotator,field,value,ts,device_id,kind,received_at) VALUES(?,?,?,?,?,?,?,?,?)",
                                (e["uuid"], e["image_id"], e["annotator"], e["field"], json.dumps(e["value"]), e["ts"], e["device_id"], e["kind"], now))
                added += cur.rowcount
    return added, bad

def git(*args, check=True):
    return subprocess.run(["git", "-C", str(ROOT), *args], check=check, capture_output=True, text=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "import", "sync"])
    ap.add_argument("--db", default=os.environ.get("WV_DB", ROOT / "var" / "wheelview.db"))
    ap.add_argument("--events", default=ROOT / "annotations" / "events")
    ap.add_argument("--annotator", help="export only this annotator (e.g. on a personal branch)")
    ap.add_argument("--push", action="store_true", help="git push after committing (only if a remote is configured)")
    a = ap.parse_args()
    if a.cmd in ("import", "sync"): print("imported %d new events (%d invalid lines skipped)" % import_(a.db, a.events))
    if a.cmd in ("export", "sync"): print("exported %d new events" % export(a.db, a.events, a.annotator))
    if a.cmd == "sync":
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build_derived.py")], check=True)
        git("add", "annotations/events")
        if git("diff", "--cached", "--quiet", check=False).returncode == 0: print("nothing to commit")
        else:
            git("commit", "-m", f"annotations: sync events {time.strftime('%Y-%m-%d %H:%M')}", "--", "annotations/events"); print("committed annotations/events")
        if a.push:
            remotes = git("remote").stdout.split()
            if not remotes: print("no git remote configured -> not pushing")
            else: print(git("push", "origin", "HEAD").stderr)
