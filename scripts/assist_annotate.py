"""Run the annotation-assist models (scripts/assist_models.py) over images and store MACHINE SUGGESTIONS.

  python scripts/assist_models.py                                   # which models are available, and why not
  python scripts/assist_annotate.py --tasks vlm --limit 50          # VLM attribute hints (Attributes pass, orange)
  python scripts/assist_annotate.py --tasks detect,segment          # boxes (+ SAM 2 masks) for synth_augment.py
  python scripts/assist_annotate.py --ids my_ids.txt --tasks detect # only these images

Outputs (never annotation events, never human labels):
  * VLM attribute hints -> annotations/auto/<model>_attr-vlm-v1.jsonl (+ .meta.json). Tags containing "_attr-" are shown
    only in the Attributes v1.1 pass as orange hints with the model name; they are saved only if a human picks them,
    and then with provenance source=auto_attr + model_tag.
  * Detector boxes -> var/assist/<detector>/boxes.jsonl; SAM 2 masks -> var/assist/<segmenter>/masks/<id>.png +
    masks.jsonl (var/ is gitignored). Boxes/masks deliberately do NOT go to annotations/auto/ (that folder's
    non-attribute files drive review mode).
Each record carries the model name, backend, revision and run time.

No labels are read: the image list comes from the image folder (or --ids), never from events, manifests or labels,
so inference cannot see training labels (checked by scripts/check_leakage.py). Models that are not installed or whose
weights are not cached are skipped with a reason; nothing is downloaded unless config allow_download is true.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server")); sys.path.insert(0, str(ROOT / "scripts"))
import wvcore, assist_models as AM

VLM_PROMPT_VERSION = "vlm-v1"
CAPTION_SKIP = {"chair_subtype", "drive_position", "ramp_deploy"}   # a caption cannot answer these reliably (need VQA)
VLM_FIELDS = ["environment", "weather", "ground_wet", "control", "chair_subtype", "drive_position", "ramp_deploy"]

def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def sha1(p): return hashlib.sha1(open(p, "rb").read()).hexdigest()

def image_list(imgdir, ids_file=None, limit=0):
    ids = None
    if ids_file: ids = [x.strip() for x in open(ids_file, encoding="utf-8") if x.strip() and not x.startswith("#")]
    files = sorted(Path(imgdir).glob("*.jpg")) if ids is None else [Path(imgdir) / f"{i}.jpg" for i in ids]
    files = [f for f in files if f.exists()]
    return files[:limit] if limit else files

def vlm_suggest(vlm, image):
    """{field: {value, confidence: None, raw}} -- options from the schema; dependent fields asked only after their parent
    answer (chair_subtype: manual/powered list from a 'manual or powered?' question)."""
    out = {}
    if not getattr(vlm, "free_text", True):   # caption models: one caption, options matched in it
        cap = vlm.caption(image)
        for f in VLM_FIELDS:
            if f in CAPTION_SKIP: continue
            v = AM.parse_choice(cap, wvcore.FIELD_OPTS[f])
            if v: out[f] = {"value": v, "confidence": None, "raw": cap[:300], "method": "caption-match"}
        return out
    parent = AM.parse_choice(vlm.ask(image, "Is the wheelchair manual or powered? Answer manual, powered, or none."), ["manual", "powered"])
    for f in VLM_FIELDS:
        opts = wvcore.FIELD_OPTS[f]
        if f in wvcore.OPTIONS_BY_PARENT:
            if parent not in wvcore.OPTIONS_BY_PARENT[f]: continue
            opts = wvcore.OPTIONS_BY_PARENT[f][parent]
        if f == "drive_position" and parent != "powered": continue
        raw = vlm.ask(image, AM.prompt_for(f, opts)); v = AM.parse_choice(raw, opts)
        if v: out[f] = {"value": v, "confidence": None, "raw": raw[:200], "method": "vqa"} | ({"parent": parent} if f in wvcore.OPTIONS_BY_PARENT else {})
    return out

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=os.environ.get("WV_BLUR_DIR", str(ROOT / "images_blurred")), help="face-blurred images (default)")
    ap.add_argument("--ids", help="text file with one image id per line (default: every .jpg in --images)")
    ap.add_argument("--tasks", default="vlm", help="comma list of: vlm, detect, segment")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--auto-dir", default=str(ROOT / "annotations" / "auto"))
    ap.add_argument("--assist-dir", default=str(ROOT / "var" / "assist"))
    ap.add_argument("--dry-run", action="store_true", help="report model availability and the image count; load nothing")
    a = ap.parse_args(argv)
    cfg = AM.load_config(a.config); tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    files = image_list(a.images, a.ids, a.limit)
    st = AM.status(cfg=cfg); print(json.dumps(st, indent=1)); print(f"{len(files)} images in {a.images}")
    if a.dry_run: return 0
    from PIL import Image
    kinds = {"vlm": "vlm", "detect": "detector", "segment": "segmenter"}
    M = {}
    for t in tasks:
        k = kinds.get(t)
        if not k: print(f"unknown task {t!r}"); continue
        m = AM.get(k, cfg)
        if m is None: print(f"skip {t}: {AM.status(k, cfg)['reason']}"); continue
        M[t] = m
    if "segment" in M and "detect" not in M: print("segment needs boxes: running only where var/assist/<detector>/boxes.jsonl exists")
    if not M: print("no assist model available -> nothing written"); return 0
    fh = {}; t0 = time.time(); n = 0
    boxes_cache = {}
    if "segment" in M and "detect" not in M:
        for p in Path(a.assist_dir).glob("*/boxes.jsonl"):
            for line in open(p, encoding="utf-8"):
                r = json.loads(line); boxes_cache.setdefault(r["image_id"], r.get("boxes") or [])
    try:
        for f in files:
            iid = f.stem
            try: im = Image.open(f).convert("RGB")
            except Exception as e: print(f"skip {iid}: {e}"); continue
            h = sha1(f); boxes = None
            if "vlm" in M:
                v = M["vlm"]; tag = f"{v.tag}_attr-{VLM_PROMPT_VERSION}"
                if tag not in fh:
                    Path(a.auto_dir).mkdir(parents=True, exist_ok=True); fh[tag] = open(Path(a.auto_dir) / f"{tag}.jsonl", "w", encoding="utf-8", newline="\n")
                    json.dump({"model_tag": tag, "kind": "machine_suggestions_v11_attributes", "never_counts_as_human_annotation": True,
                               "shown_in": "Attributes pass only, as orange hints; never bulk-accepted", "vlm": v.info(),
                               "prompt_version": VLM_PROMPT_VERSION, "questions": AM.QUESTIONS, "fields": VLM_FIELDS,
                               "inputs": "pixels of the face-blurred image only (no labels)", "generated_at": now()},
                              open(Path(a.auto_dir) / f"{tag}.meta.json", "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
                sg = vlm_suggest(v, im)
                fh[tag].write(json.dumps({"image_id": iid, "model_tag": tag, "image_sha1": h, "suggestions": sg}, sort_keys=True, separators=(",", ":")) + "\n")
            if "detect" in M:
                d = M["detect"]; key = ("det", d.tag)
                if key not in fh:
                    (Path(a.assist_dir) / d.tag).mkdir(parents=True, exist_ok=True); fh[key] = open(Path(a.assist_dir) / d.tag / "boxes.jsonl", "w", encoding="utf-8", newline="\n")
                boxes = d.detect(im)
                fh[key].write(json.dumps({"image_id": iid, "image_sha1": h, "model": d.info(), "boxes": boxes, "generated_at": now()}, sort_keys=True) + "\n")
            if "segment" in M:
                s = M["segment"]; key = ("seg", s.tag); bx = boxes if boxes is not None else boxes_cache.get(iid, [])
                if key not in fh:
                    (Path(a.assist_dir) / s.tag / "masks").mkdir(parents=True, exist_ok=True); fh[key] = open(Path(a.assist_dir) / s.tag / "masks.jsonl", "w", encoding="utf-8", newline="\n")
                for j, b in enumerate(sorted(bx, key=lambda b: -b["score"])[:3]):
                    mk = s.segment(im, b["box"]); fn = f"{iid}_{j}.png"
                    Image.fromarray((mk * 255).astype("uint8")).save(Path(a.assist_dir) / s.tag / "masks" / fn)
                    fh[key].write(json.dumps({"image_id": iid, "image_sha1": h, "model": s.info(), "box": b["box"], "box_label": b.get("label"),
                                              "mask_file": f"masks/{fn}", "area_px": int(mk.sum()), "generated_at": now()}, sort_keys=True) + "\n")
            n += 1
            if n % 25 == 0: print(f"{n}/{len(files)} images, {time.time() - t0:.0f}s")
    finally:
        for x in fh.values(): x.close()
    print(f"done: {n} images in {time.time() - t0:.0f}s; wrote " + ", ".join(str(k) for k in fh))
    return 0

if __name__ == "__main__":
    sys.exit(main())
