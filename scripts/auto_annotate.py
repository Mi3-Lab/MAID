#!/usr/bin/env python3
"""WheelView auto-annotator: machine PRE-LABELS for human review. Never a human annotation.

Runs open-weights models LOCALLY (CPU is fine). Images never leave the machine; only model weights are downloaded from
the Hugging Face hub (once, then cached).

  * Classifier: SigLIP 2 (google/siglip2-so400m-patch14-224, Apache-2.0), zero-shot with a prompt ensemble per option of
    every Table-1 field. Output per image and field: suggested value + confidence (softmax probability of the top option)
    + the full distribution.
  * Screening (suggestions only, NEVER auto-exclusion): likely child, clinical setting, visible face, not a photo.
  * Optional device boxes: OWLv2 (google/owlv2-base-patch16-ensemble, Apache-2.0) open-vocabulary detector (`--detect`).

Incremental + resumable: image embeddings and detections are cached in var/auto_cache/<model_tag>.sqlite keyed by
image id + file SHA-1, so re-runs only process new/changed images; scoring (prompts -> suggestions) is cheap and the
output file is fully regenerated from the cache on every run.

Output (a separate MACHINE source, never under annotations/events/):
  annotations/auto/<model_tag>.jsonl      one JSON object per image (sorted by image_id)
  annotations/auto/<model_tag>.meta.json  model names, revisions, licenses, prompts, thresholds, runtime

Usage:
  pip install -r requirements-auto.txt
  python scripts/auto_annotate.py                          # all images available locally, untyped wheelchairs first
  python scripts/auto_annotate.py --limit 500              # stop after 500 newly embedded images (resume later)
  python scripts/auto_annotate.py --detect                 # also run OWLv2 boxes (≈2 s/image on CPU)
  python scripts/auto_annotate.py --score-only             # re-score from cache (e.g. after editing prompts)
  python scripts/auto_annotate.py --eval                   # model-vs-human agreement on derived/per_annotator_labels.csv
"""
import argparse, csv, hashlib, io, json, os, sqlite3, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server")); import wvcore

CLS_MODEL = "google/siglip2-so400m-patch14-224"
CLS_LICENSE = "Apache-2.0"
DET_MODEL = "google/owlv2-base-patch16-ensemble"
DET_LICENSE = "Apache-2.0"
PROMPT_VERSION = "zs-v1"
MODEL_TAG = "siglip2-so400m-224_" + PROMPT_VERSION     # changes whenever prompts/model change -> new output file

# ---------- prompt ensembles (option -> descriptions); averaged normalised text embeddings ----------
P = {
 "device_type": {
  "manual": ["a manual wheelchair with large rear wheels and push rims", "a person sitting in a manual wheelchair",
             "a folding manual wheelchair", "a lightweight sports wheelchair pushed by hand",
             "a hospital-style manual wheelchair with push handles"],
  "powered": ["an electric powered wheelchair with a joystick control", "a person in a power wheelchair",
              "a motorized wheelchair with a battery box and small wheels", "an electric wheelchair"],
  "other_aid": ["a mobility scooter", "a person riding a mobility scooter", "a rollator walker with wheels and a seat",
                "a walking frame", "a person using a walker", "a walking cane or crutches"],
  "lookalike": ["a baby stroller", "a pram or pushchair", "a shopping cart", "a supermarket trolley",
                "an office chair on wheels", "a luggage trolley", "a baggage cart at an airport"],
 },
 "occupied": {
  "yes": ["a person sitting in a wheelchair", "a person seated on a mobility device", "someone riding a scooter",
          "a wheelchair user"],
  "no": ["an empty wheelchair with nobody in it", "an empty mobility device with no one sitting on it",
         "an unoccupied chair", "an empty stroller"],
 },
 "_propulsion_manual": {     # conditional on a manual wheelchair; marginalised with P(device_type)
  "self": ["a person pushing the wheels of their own wheelchair", "a wheelchair user propelling themselves",
           "a person wheeling themselves in a manual wheelchair"],
  "attendant": ["a person pushing someone else in a wheelchair from behind", "a caregiver pushing a wheelchair",
                "a helper standing behind a wheelchair holding the handles"],
  "not_applicable": ["an empty wheelchair with nobody in it", "a parked wheelchair"],
 },
 "occlusion": {
  "none": ["a fully visible wheelchair, nothing blocking the view", "the whole mobility device is clearly visible"],
  "partial": ["a wheelchair partly hidden behind other objects or people", "a partially occluded wheelchair",
              "a wheelchair cut off by the edge of the photo"],
  "heavy": ["a wheelchair mostly hidden, only a small part visible", "a heavily occluded wheelchair in a crowd"],
 },
 "viewpoint": {
  "front": ["a wheelchair seen from the front", "a front view of a person in a wheelchair facing the camera"],
  "rear": ["a wheelchair seen from behind", "a rear view of a wheelchair, back of the seat facing the camera"],
  "side": ["a side view of a wheelchair in profile", "a wheelchair seen from the side"],
  "three-quarter": ["a three-quarter view of a wheelchair, seen diagonally from the front corner",
                    "a wheelchair photographed at an angle"],
 },
 "lighting": {
  "daylight": ["a photo taken outdoors in bright sunlight", "an outdoor scene in daylight", "a sunny day outside"],
  "shade": ["an outdoor photo in the shade", "an overcast cloudy day outside", "an outdoor scene in shadow"],
  "indoor": ["a photo taken indoors under artificial light", "an indoor room", "inside a building"],
  "low_light": ["a dark photo taken at night", "a dimly lit scene in low light", "a night-time street"],
 },
 "setting": {
  "curb": ["a wheelchair at a street curb", "a curb ramp at a crosswalk", "the edge of a sidewalk by the road"],
  "vehicle": ["a wheelchair getting into a van or car", "a wheelchair ramp on a bus", "a wheelchair inside a vehicle",
              "a wheelchair lift on a vehicle"],
  "sidewalk": ["a wheelchair on a sidewalk", "a pavement or footpath in a city street", "a pedestrian walkway"],
  "indoor": ["inside a building", "an indoor room or hallway", "inside a shop or office"],
  "other": ["in a park or garden", "on a beach or trail", "at a sports court", "on a stage or at an event",
            "a plain studio background"],
 },
}
SCREEN = {   # flag -> (positive prompts, negative prompts); score = P(positive) under a 2-way softmax
 "child": (["a photo of a child", "a young kid", "a toddler", "a baby", "children playing"],
           ["a photo of an adult", "an elderly person", "a photo with no people", "an adult man or woman"]),
 "clinical_setting": (["a hospital room", "a hospital ward with beds", "a clinic or doctor's office",
                       "a patient in a hospital", "a nurse in a medical setting", "a rehabilitation clinic"],
                      ["a street or public place", "a home living room", "a shop", "a park", "an office"]),
 "visible_face": (["a photo where a person's face is clearly visible", "a close-up portrait of a person's face",
                   "a person looking at the camera"],
                  ["a photo with no people", "a person seen from behind", "an object with no face visible",
                   "a person far away, face too small to see"]),
 "not_a_photo": (["a drawing", "a cartoon illustration", "an icon or pictogram", "a diagram or technical drawing",
                  "clip art", "a logo or sign symbol", "a 3D render", "a scanned document with text"],
                 ["a real photograph", "a photo taken with a camera", "a snapshot of a real scene"]),
}
# Screening flags are shown to reviewers when score >= threshold (chosen for recall; see --eval); never auto-exclude.
SCREEN_THRESH = {"child": 0.5, "clinical_setting": 0.5, "visible_face": 0.5, "not_a_photo": 0.5}
DET_QUERIES = ["a wheelchair", "a power wheelchair", "a mobility scooter", "a rollator walker", "a baby stroller",
               "a shopping cart", "an office chair", "a luggage trolley"]
DET_MIN_SCORE = 0.25

def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def image_path(cid, row, build_dir):
    for p in (ROOT / "images" / f"{cid}.jpg", ROOT / "var" / "imgcache" / f"{cid}.jpg"):
        if p.exists(): return p
    tp = row.get("thumb_path") or ""
    if tp and build_dir:
        p = Path(build_dir) / "thumbs" / Path(tp).name
        if p.exists(): return p
    return None

def priority(row):
    pc = row.get("proposed_class", "")
    if pc == "wheelchair_unspecified": return 0
    if "wheelchair" in pc or "other_aid" in pc: return 1
    return 2

def cache(model_tag):
    d = ROOT / "var" / "auto_cache"; d.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(d / f"{model_tag.split('_')[0]}.sqlite")
    c.execute("CREATE TABLE IF NOT EXISTS emb(image_id TEXT PRIMARY KEY, sha1 TEXT, emb BLOB, at TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS det(image_id TEXT PRIMARY KEY, sha1 TEXT, boxes TEXT, at TEXT)")
    return c

def sha1(p): return hashlib.sha1(Path(p).read_bytes()).hexdigest()

def load_cls():
    import torch
    from transformers import AutoModel, AutoProcessor
    m = AutoModel.from_pretrained(CLS_MODEL).eval(); p = AutoProcessor.from_pretrained(CLS_MODEL)
    return torch, m, p

def feats(out):
    return out if hasattr(out, "norm") else (getattr(out, "pooler_output", None) if getattr(out, "pooler_output", None) is not None else out[0])

def embed(todo, c, batch, limit, threads):
    import numpy as np
    from PIL import Image
    torch, m, p = load_cls(); torch.set_num_threads(threads)
    t0 = time.time(); n = 0
    for i in range(0, len(todo), batch):
        if limit and n >= limit: break
        chunk = todo[i:i + batch]; ims, ok = [], []
        for cid, path, h in chunk:
            try: ims.append(Image.open(path).convert("RGB")); ok.append((cid, h))
            except Exception as e: print(f"WARN cannot open {cid}: {e}", file=sys.stderr)
        if not ims: continue
        with torch.inference_mode():
            e = feats(m.get_image_features(**p(images=ims, return_tensors="pt")))
            e = torch.nn.functional.normalize(e.float(), dim=-1).numpy().astype("float32")
        c.executemany("INSERT OR REPLACE INTO emb VALUES(?,?,?,?)", [(cid, h, e[j].tobytes(), now()) for j, (cid, h) in enumerate(ok)])
        c.commit(); n += len(ok)
        el = time.time() - t0
        print(f"embedded {n}/{len(todo)}  {el/n:.2f} s/img  eta {(len(todo)-n)*el/n/60:.1f} min", flush=True)
    return n, time.time() - t0

def text_bank(torch, m, p):
    """option -> normalised mean text embedding, for every field (and screening pos/neg)."""
    import numpy as np
    def enc(texts):
        with torch.inference_mode():
            x = p(text=[t.lower() for t in texts], padding="max_length", max_length=64, return_tensors="pt", truncation=True)
            t = feats(m.get_text_features(**x)).float()
            t = torch.nn.functional.normalize(t, dim=-1); t = torch.nn.functional.normalize(t.mean(0), dim=-1)
        return t.numpy()
    bank = {f: {o: enc(ds) for o, ds in opts.items()} for f, opts in P.items()}
    scr = {k: (enc(pos), enc(neg)) for k, (pos, neg) in SCREEN.items()}
    scale = float(m.logit_scale.exp()) if hasattr(m, "logit_scale") else 100.0
    return bank, scr, scale

def softmax(z):
    import numpy as np
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)

# Label-free calibration: subtract CALIB x the corpus-mean similarity of each option (computed over all local images,
# no human labels used) so options whose prompts are simply "closer to everything" (e.g. 'vehicle') stop winning.
CALIB = float(os.environ.get("WV_AUTO_CALIB", "0"))   # off: on the 73 labelled images it helped device_type/occupied but hurt setting/lighting
TEMP = 0.01   # softmax temperature on cosine similarity (≈ CLIP's 1/100); SigLIP's own logit scale is far sharper

def score(E, bank, scr):
    """E: (n, d) normalised image embeddings -> list of per-image dicts."""
    import numpy as np
    out = [dict() for _ in range(len(E))]; sug = [dict() for _ in range(len(E))]
    prob = {}
    for f, opts in bank.items():
        names = list(opts); T = np.stack([opts[o] for o in names])
        z = E @ T.T
        if CALIB and len(E) >= 200: z = z - CALIB * z.mean(0, keepdims=True)   # label-free prior correction
        prob[f] = (names, softmax(z / TEMP))
    dn, dp = prob["device_type"]; pman = dp[:, dn.index("manual")]
    pn, pp = prob.pop("_propulsion_manual")
    names = ["self", "attendant", "not_applicable", "unclear"]
    prop = np.zeros((len(E), 4))
    for j, o in enumerate(pn): prop[:, names.index(o)] += pman * pp[:, j]
    prop[:, names.index("not_applicable")] += 1 - pman
    prob["propulsion"] = (names, prop)
    for f in wvcore.FIELD_OPTS:
        names, pr = prob[f]
        for i in range(len(E)):
            k = int(pr[i].argmax())
            sug[i][f] = {"value": names[k], "confidence": round(float(pr[i, k]), 4),
                         "probs": {n: round(float(pr[i, j]), 4) for j, n in enumerate(names)}}
    scrs = {}
    for k, (pos, neg) in scr.items():
        z = np.stack([E @ pos, E @ neg], 1) / TEMP; scrs[k] = softmax(z)[:, 0]
    for i in range(len(E)):
        s = {k: round(float(v[i]), 4) for k, v in scrs.items()}
        out[i] = {"suggestions": sug[i], "screening": s, "flags": [k for k, v in s.items() if v >= SCREEN_THRESH[k]]}
    return out

def detect(todo, c, limit, threads):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Owlv2ForObjectDetection
    torch.set_num_threads(threads)
    m = Owlv2ForObjectDetection.from_pretrained(DET_MODEL).eval(); p = AutoProcessor.from_pretrained(DET_MODEL)
    t0 = time.time(); n = 0
    for cid, path, h in todo:
        if limit and n >= limit: break
        try: im = Image.open(path).convert("RGB")
        except Exception: continue
        W, H = im.size; S = max(W, H)   # OWLv2 pads to a square at the bottom/right
        with torch.inference_mode():
            x = p(text=[DET_QUERIES], images=im, return_tensors="pt"); o = m(**x)
            pp = getattr(p, "post_process_object_detection", None) or p.post_process_grounded_object_detection  # transformers 4 / 5
            r = pp(o, threshold=DET_MIN_SCORE, target_sizes=torch.tensor([[S, S]]))[0]
        boxes = []
        for b, s, l in zip(r["boxes"].tolist(), r["scores"].tolist(), r["labels"].tolist()):
            x0, y0, x1, y1 = [max(0.0, v) for v in b]; x1, y1 = min(x1, W), min(y1, H)
            if x1 - x0 < 4 or y1 - y0 < 4: continue
            boxes.append({"box": [round(x0 / W, 4), round(y0 / H, 4), round((x1 - x0) / W, 4), round((y1 - y0) / H, 4)],
                          "score": round(s, 4), "label": DET_QUERIES[l]})
        boxes = nms(sorted(boxes, key=lambda d: -d["score"]))[:5]
        c.execute("INSERT OR REPLACE INTO det VALUES(?,?,?,?)", (cid, h, json.dumps(boxes), now())); c.commit(); n += 1
        if n % 20 == 0: el = time.time() - t0; print(f"detected {n}/{len(todo)}  {el/n:.2f} s/img  eta {(len(todo)-n)*el/n/60:.1f} min", flush=True)
    return n, time.time() - t0

def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    u = aw * ah + bw * bh - ix * iy; return ix * iy / u if u else 0

def nms(ds, thr=0.5):
    keep = []
    for d in ds:
        if all(iou(d["box"], k["box"]) < thr for k in keep): keep.append(d)
    return keep

def write_output(c, cands, meta_extra):
    import numpy as np
    from transformers import AutoModel, AutoProcessor
    import torch
    rows = c.execute("SELECT image_id, sha1, emb FROM emb ORDER BY image_id").fetchall()
    rows = [r for r in rows if r[0] in cands]
    dets = {r[0]: (r[1], json.loads(r[2])) for r in c.execute("SELECT image_id, sha1, boxes FROM det")}
    m = AutoModel.from_pretrained(CLS_MODEL).eval(); p = AutoProcessor.from_pretrained(CLS_MODEL)
    bank, scr, _ = text_bank(torch, m, p)
    E = np.stack([np.frombuffer(r[2], dtype="float32") for r in rows]) if rows else np.zeros((0, 1))
    res = score(E, bank, scr) if rows else []
    outdir = ROOT / "annotations" / "auto"; outdir.mkdir(parents=True, exist_ok=True)
    fn = outdir / f"{MODEL_TAG}.jsonl"; nbox = 0
    with open(fn, "w", encoding="utf-8", newline="\n") as fh:
        for (cid, h, _), r in zip(rows, res):
            rec = {"image_id": cid, "model_tag": MODEL_TAG, "image_sha1": h, **r}
            if cid in dets and dets[cid][0] == h:
                rec["boxes"] = dets[cid][1]; nbox += 1
            fh.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
    meta = {"model_tag": MODEL_TAG, "kind": "machine_suggestions", "never_counts_as_human_annotation": True,
            "classifier": {"model": CLS_MODEL, "revision": rev(CLS_MODEL), "license": CLS_LICENSE,
                           "method": "zero-shot, prompt ensemble (mean of normalised text embeddings per option), softmax over options at temperature %.3g on cosine similarity; propulsion = P(manual)*P(self/attendant/empty | manual) + (1-P(manual)) -> not_applicable" % TEMP},
            "detector": {"model": DET_MODEL, "revision": rev(DET_MODEL), "license": DET_LICENSE, "queries": DET_QUERIES,
                         "min_score": DET_MIN_SCORE, "images_with_boxes": nbox},
            "prompt_version": PROMPT_VERSION, "prompts": P, "screening_prompts": SCREEN, "screening_thresholds": SCREEN_THRESH,
            "screening_policy": "flags are suggestions for the human reviewer; the model never excludes an image",
            "images": len(rows), "generated_at": now(), "runs_locally": "images are processed on the lab machine; nothing is uploaded",
            **meta_extra}
    old = {}
    mf = outdir / f"{MODEL_TAG}.meta.json"
    if mf.exists():
        try: old = json.load(open(mf, encoding="utf-8")).get("runs", [])
        except Exception: old = []
    meta["runs"] = (old if isinstance(old, list) else []) + ([meta_extra["run"]] if meta_extra.get("run") else [])
    meta.pop("run", None)
    json.dump(meta, open(mf, "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    print(f"wrote {fn.relative_to(ROOT)}: {len(rows)} images ({nbox} with boxes)")
    return fn

def rev(name):
    try:
        from huggingface_hub import model_info; return model_info(name).sha
    except Exception:
        return None

def evaluate():
    """Model vs human (every 'done' human label, events/*.jsonl) per field; uses wvcore.auto_agreement."""
    sys.path.insert(0, str(ROOT / "scripts")); import build_derived
    evs, _ = build_derived.load_events(ROOT / "annotations" / "events")
    auto = wvcore.load_auto(ROOT / "annotations" / "auto")
    per, _ = wvcore.fold(evs)
    res = wvcore.auto_agreement(per, auto)
    print(json.dumps(res, indent=1)); return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default=ROOT / "data" / "candidates.csv")
    ap.add_argument("--build-dir", default=os.environ.get("WHEELVIEW_BUILD_DIR", "/workspace/wheelview"))
    ap.add_argument("--limit", type=int, default=0, help="max NEW images to embed/detect this run (0 = all)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--detect", action="store_true", help="also run OWLv2 device boxes")
    ap.add_argument("--detect-only", action="store_true")
    ap.add_argument("--detect-classes", default="wheelchair,other_aid", help="proposed_class substrings to run the detector on ('' = all)")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.eval: evaluate(); sys.exit(0)
    cands = wvcore.load_candidates(a.candidates)
    c = cache(MODEL_TAG)
    have = {r[0]: r[1] for r in c.execute("SELECT image_id, sha1 FROM emb")}
    hdet = {r[0]: r[1] for r in c.execute("SELECT image_id, sha1 FROM det")}
    avail = []
    for cid, row in sorted(cands.items(), key=lambda kv: (priority(kv[1]), kv[0])):
        pth = image_path(cid, row, a.build_dir)
        if pth: avail.append((cid, pth, sha1(pth)))
    print(f"{len(avail)} of {len(cands)} candidates have a local image; {len(have)} already embedded")
    run = {"started": now(), "host_cpus": os.cpu_count(), "threads": a.threads}
    if not a.score_only and not a.detect_only:
        todo = [t for t in avail if have.get(t[0]) != t[2]]
        n, el = embed(todo, c, a.batch, a.limit, a.threads) if todo else (0, 0.0)
        run.update(embedded=n, embed_seconds=round(el, 1))
    if a.detect or a.detect_only:
        keys = [k for k in a.detect_classes.split(",") if k]
        todo = [t for t in avail if hdet.get(t[0]) != t[2] and (not keys or any(k in cands[t[0]].get("proposed_class", "") for k in keys))]
        n, el = detect(todo, c, a.limit, a.threads) if todo else (0, 0.0)
        run.update(detected=n, detect_seconds=round(el, 1))
    run["finished"] = now()
    remaining = len([t for t in avail if have.get(t[0]) != t[2]]) if a.score_only else None
    write_output(c, cands, {"run": run})
    left = sum(1 for t in avail if (c.execute("SELECT sha1 FROM emb WHERE image_id=?", (t[0],)).fetchone() or [None])[0] != t[2])
    print(f"{left} locally available images still need embedding" + (" -> re-run the same command to resume" if left else ""))
