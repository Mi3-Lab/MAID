"""Create/refresh the local Label Studio 'WheelView' project (evaluation only; local, 127.0.0.1).

Mirrors server/schema.json as a Label Studio labeling config, registers images_blurred/ as local-files
storage (blurred copies ONLY - the unblurred images/ folder is never referenced), imports one task per
blurred image, and attaches the SigLIP2 zero-shot suggestions (+ OWLv2 boxes where present) as
predictions so Label Studio's pre-labeling shows them.

Stdlib only. Usage (Label Studio must be running with LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true and
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<repo root>):
    python scripts/labelstudio_import.py [--url http://127.0.0.1:8080] [--credentials ~/labelstudio/CREDENTIALS.txt] [--limit N] [--recreate]
Label Studio data (its SQLite DB, exports) lives outside the repo and must never be committed.
"""
import argparse, json, os, sys, urllib.request, urllib.error
from xml.sax.saxutils import quoteattr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = json.load(open(os.path.join(ROOT, "server", "schema.json"), encoding="utf-8"))
AUTO = os.path.join(ROOT, "annotations", "auto", "siglip2-so400m-224_zs-v1.jsonl")
BLUR_DIR = os.path.join(ROOT, "images_blurred")
TITLE = "WheelView"

# hotkeys mirror the spirit of the WheelView app: digits for device type; letters elsewhere avoid LS defaults
HOTKEYS = {"device_type": ["1", "2", "3", "4", "5"]}


def labeling_config():
    x = ['<View>', '  <Header value="WheelView - faces are pre-blurred; tick the face flag if any face is still recognisable"/>',
         '  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>',
         '  <RectangleLabels name="bbox" toName="image" strokeWidth="2">',
         '    <Label value="device" background="#ff7f0e"/>', '  </RectangleLabels>',
         '  <View style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px">']
    for f in SCHEMA["fields"]:
        hk = HOTKEYS.get(f["name"], [])
        x.append(f'    <View><Header value={quoteattr(f["label"])} size="6"/>')
        x.append(f'    <Choices name="{f["name"]}" toName="image" choice="single-radio" showInline="false" required="false">')
        for i, o in enumerate(f["options"]):
            h = f' hotkey="{hk[i]}"' if i < len(hk) else ""
            x.append(f'      <Choice value={quoteattr(o)}{h}/>')
        x.append('    </Choices></View>')
    x.append('    <View><Header value="Exclude (reason)" size="6"/>')
    x.append('    <Choices name="exclude" toName="image" choice="single" showInline="false">')
    x += [f'      <Choice value={quoteattr(r)}/>' for r in SCHEMA["exclude_reasons"]]
    x.append('    </Choices></View>')
    x.append('  </View>')
    x.append('  <Choices name="face_blur_missed" toName="image" choice="multiple">')
    x.append('    <Choice value="face blur missed (face still recognisable)" hotkey="f"/>')
    x.append('  </Choices>')
    x.append('</View>')
    return "\n".join(x)


class API:
    """Session (cookie + CSRF) login with the local admin's email/password. Label Studio >=1.20 disables legacy
    'Token' auth for new organizations by default, so this avoids needing a personal access token."""
    def __init__(s, url, email, password):
        import http.cookiejar, urllib.parse, re
        s.url = url.rstrip("/"); s.cj = http.cookiejar.CookieJar()
        s.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(s.cj))
        html = s.op.open(s.url + "/user/login/", timeout=60).read().decode("utf-8", "replace")
        m = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', html)
        form = urllib.parse.urlencode({"email": email, "password": password, "csrfmiddlewaretoken": m.group(1) if m else s.csrf()}).encode()
        r = s.op.open(urllib.request.Request(s.url + "/user/login/", data=form, headers={"Referer": s.url + "/user/login/"}), timeout=60)
        if "/user/login" in r.geturl(): sys.exit("login failed (check credentials file)")

    def csrf(s): return next((c.value for c in s.cj if c.name == "csrftoken"), "")

    def __call__(s, method, path, body=None):
        h = {"Content-Type": "application/json", "X-CSRFToken": s.csrf(), "Referer": s.url + "/"}
        req = urllib.request.Request(s.url + path, method=method, headers=h, data=None if body is None else json.dumps(body).encode())
        try:
            with s.op.open(req, timeout=600) as r:
                t = r.read(); return json.loads(t) if t else None
        except urllib.error.HTTPError as e:
            sys.exit(f"{method} {path} -> {e.code}: {e.read()[:500]!r}")


def read_credentials(path):
    kv = dict(l.split(":", 1) for l in open(path, encoding="utf-8-sig") if ":" in l and not l.startswith("Label Studio"))
    return kv["username"].strip(), kv["password"].strip()


def prediction(a, size=None):
    res = []
    for f in SCHEMA["fields"]:
        s = (a.get("suggestions") or {}).get(f["name"])
        if s and s.get("value") in f["options"]:
            res.append({"from_name": f["name"], "to_name": "image", "type": "choices",
                        "value": {"choices": [s["value"]]}, "score": s.get("confidence")})
    for i, b in enumerate(a.get("boxes") or []):
        x, y, w, h = b["box"]
        r = {"id": f"owl{i}", "from_name": "bbox", "to_name": "image", "type": "rectanglelabels", "score": b.get("score"),
                    "value": {"x": 100 * x, "y": 100 * y, "width": 100 * w, "height": 100 * h, "rotation": 0,
                              "rectanglelabels": ["device"]}}
        if size: r.update(original_width=size[0], original_height=size[1], image_rotation=0)  # COCO/YOLO export needs these
        res.append(r)
    dt = (a.get("suggestions") or {}).get("device_type") or {}
    return {"model_version": a.get("model_tag", "siglip2"), "score": dt.get("confidence"), "result": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--credentials", default=os.path.join(os.path.expanduser("~"), "labelstudio", "CREDENTIALS.txt"))
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--recreate", action="store_true")
    a = ap.parse_args()
    api = API(a.url, *read_credentials(a.credentials))
    projs = api("GET", "/api/projects?page_size=100")
    projs = projs.get("results", projs) if isinstance(projs, dict) else projs
    p = next((q for q in projs if q["title"] == TITLE), None)
    if p and a.recreate: api("DELETE", f"/api/projects/{p['id']}"); p = None
    cfg = labeling_config()
    if p is None:
        p = api("POST", "/api/projects", {"title": TITLE, "description": "WheelView evaluation copy (face-blurred images only, local)",
                                          "label_config": cfg, "show_collab_predictions": True, "maximum_annotations": 2,
                                          "model_version": "siglip2-so400m-224_zs-v1"})
        print("created project", p["id"])
    else:
        api("PATCH", f"/api/projects/{p['id']}", {"label_config": cfg}); print("updated project", p["id"])
    pid = p["id"]
    st = api("GET", f"/api/storages/localfiles?project={pid}")
    if not st:
        api("POST", "/api/storages/localfiles", {"project": pid, "title": "images_blurred (face-blurred only)", "path": BLUR_DIR,
                                                  "regex_filter": r".*\.jpg$", "use_blob_urls": True})
        print("registered local storage", BLUR_DIR)
    if (api("GET", f"/api/projects/{pid}").get("task_number") or 0) > 0 and not a.recreate:
        print("project already has tasks; use --recreate to rebuild"); return
    auto = {}
    for line in open(AUTO, encoding="utf-8"):
        d = json.loads(line); auto[d["image_id"]] = d
    files = sorted(f for f in os.listdir(BLUR_DIR) if f.lower().endswith(".jpg"))
    if a.limit: files = files[:a.limit]
    tasks, npred = [], 0
    for fn in files:
        iid = os.path.splitext(fn)[0]
        t = {"data": {"image": f"/data/local-files/?d=images_blurred/{fn}", "image_id": iid}}
        if iid in auto:
            size = None
            if auto[iid].get("boxes"):
                try:
                    from PIL import Image
                    with Image.open(os.path.join(BLUR_DIR, fn)) as im: size = im.size
                except Exception: pass
            t["predictions"] = [prediction(auto[iid], size)]; npred += 1
        tasks.append(t)
    for i in range(0, len(tasks), 1000):
        r = api("POST", f"/api/projects/{pid}/import?commit_to_project=true", tasks[i:i + 1000])
        print("imported batch", i // 1000 + 1, {k: r.get(k) for k in ("task_count", "prediction_count")} if isinstance(r, dict) else r)
    print(f"tasks={len(tasks)} with_predictions={npred} project_id={pid}")


if __name__ == "__main__":
    main()
