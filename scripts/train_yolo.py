#!/usr/bin/env python3
"""YOLO baselines for WheelView (Ultralytics), same splits and face-BLURRED images as the ResNet/SigLIP baselines.

  * cls   : YOLO11 classification (ImageNet-pretrained yolo11s-cls), fine-tuned on train, best-val checkpoint on test,
            one run per seed (Ultralytics defaults for augmentation; no class weighting -- Ultralytics cls has none).
  * world : YOLO-World v2 (yolov8s-worldv2) zero-shot / untrained. Text classes = the project's SigLIP prompt
            ensemble (auto_annotate.P['device_type']); each prompt maps back to its device class. Image class =
            class of the highest-confidence box; an image with no box counts as wrong (rate reported).
  * det   : (--det) PRELIMINARY YOLO11 detector trained on the human boxes (bbox field), every box labelled with the
            image's device class; images with no human box are left out of the detector train/val/test sets.
            Reports box mAP plus image-level accuracy (top box) on the full test split.

Datasets are built under var/ as hard links (or copies) of images_blurred/ files; Ultralytics project = var/runs/.
Nothing here writes images or weights outside var/. Metrics only -> results/baseline_yolo_v2.json.

  python scripts/train_yolo.py --splits data/splits_v2.csv --extra_test data/splits_v1.csv --seeds 0,1,2 --det"""
import argparse, csv, glob, json, os, shutil, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from train_baseline import CLASSES, metrics
import auto_annotate as AA

NONE = len(CLASSES)   # "no box" pseudo-prediction index for YOLO-World / detector

def boot(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed); y, p = np.array(y), np.array(p); acc, mf = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); m = metrics(y[i], p[i], k=NONE + 1); acc.append(m["accuracy"]); mf.append(m["macro_f1"])
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"accuracy_95ci": q(acc), "macro_f1_95ci": q(mf), "n_boot": n}

def met(y, p):
    m = metrics(y, p, k=NONE + 1); m["confusion_matrix"] = [r[:NONE] + [r[NONE]] for r in m["confusion_matrix"][:NONE]]; m["no_box"] = int(sum(1 for v in p if v == NONE)); return m

def agg(v): return {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else 0.0}
def spread(runs):
    o = {"test_accuracy": agg([r["test"]["accuracy"] for r in runs]), "test_macro_f1": agg([r["test"]["macro_f1"] for r in runs])}
    for c in CLASSES: o[f"test_f1_{c}"] = agg([r["test"]["per_class"][c]["f1"] for r in runs])
    if all("extra_test" in r for r in runs):
        o["extra_test_accuracy"] = agg([r["extra_test"]["accuracy"] for r in runs]); o["extra_test_macro_f1"] = agg([r["extra_test"]["macro_f1"] for r in runs])
    return o

def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists(): return
    try: os.link(src, dst)
    except OSError: shutil.copy2(src, dst)

def latest_boxes(ids):
    last = {}
    for f in sorted(glob.glob(str(ROOT / "annotations" / "events" / "*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            try: e = json.loads(line)
            except Exception: continue
            if e.get("field") != "bbox" or e.get("image_id") not in ids: continue
            k = (e["image_id"], e["annotator"])
            if k not in last or (e["ts"], e["uuid"]) > (last[k]["ts"], last[k]["uuid"]): last[k] = e
    out = {}
    for (i, a), e in sorted(last.items(), key=lambda kv: (kv[1]["ts"], kv[1]["uuid"])):   # most recent non-empty labeler wins
        if e.get("value"): out[i] = e["value"]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=ROOT / "data" / "splits_v2.csv"); ap.add_argument("--extra_test", default=None)
    ap.add_argument("--images", default=ROOT / "images_blurred"); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--cls_model", default="yolo11s-cls.pt"); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--world_model", default="yolov8s-worldv2.pt"); ap.add_argument("--world_conf", type=float, default=0.01)
    ap.add_argument("--det", action="store_true"); ap.add_argument("--det_model", default="yolo11s.pt")
    ap.add_argument("--det_epochs", type=int, default=60); ap.add_argument("--det_bs", type=int, default=8); ap.add_argument("--det_seeds", default="0")
    ap.add_argument("--min_det_train", type=int, default=100)
    ap.add_argument("--skip_cls", action="store_true"); ap.add_argument("--skip_world", action="store_true")
    ap.add_argument("--pause", type=int, default=0); ap.add_argument("--tag", default="v2")
    ap.add_argument("--results", default=ROOT / "results")
    a = ap.parse_args()
    from ultralytics import YOLO, settings
    import ultralytics, torch
    try: settings.update({"sync": False})
    except Exception: pass
    var = ROOT / "var"; runs_dir = var / "runs"; work = var / "yolo_work"; work.mkdir(parents=True, exist_ok=True)
    a.splits, a.images, a.results = Path(a.splits).resolve(), Path(a.images).resolve(), Path(a.results).resolve()
    if a.extra_test: a.extra_test = Path(a.extra_test).resolve()
    os.chdir(work)   # pretrained .pt downloads land in var/yolo_work/
    rows = list(csv.DictReader(open(a.splits, encoding="utf-8"))); img = Path(a.images)
    sp = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    if a.extra_test:
        seen = {r["image_id"] for r in rows if r["split"] in ("train", "val")}
        old = [r for r in csv.DictReader(open(a.extra_test, encoding="utf-8")) if r["split"] == "test"]
        sp["extra"] = [r for r in old if r["image_id"] not in seen and r["label"] in CLASSES]
    y = {s: [CLASSES.index(r["label"]) for r in v] for s, v in sp.items()}
    paths = {s: [str(img / f"{r['image_id']}.jpg") for r in v] for s, v in sp.items()}
    out = {"ultralytics": ultralytics.__version__, "torch": torch.__version__, "splits_file": "data/" + Path(a.splits).name,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "extra_test": {"file": Path(a.extra_test).name, "n_evaluated": len(sp["extra"])} if a.extra_test else None}
    seeds = [int(s) for s in a.seeds.split(",")]
    def chunks(lst, n=32):
        for i in range(0, len(lst), n): yield lst[i:i + n]

    # ---------- YOLO11 classification ----------
    if not a.skip_cls:
        ds = var / f"yolo_cls_{a.tag}"
        for s in ("train", "val", "test"):
            for r in sp[s]: link(img / f"{r['image_id']}.jpg", ds / s / r["label"] / f"{r['image_id']}.jpg")
        cruns = []
        for si, s in enumerate(seeds):
            if si and a.pause: time.sleep(a.pause)
            name = f"yolo11s_cls_{a.tag}_seed{s}"; t0 = time.time()
            m = YOLO(a.cls_model)
            m.train(data=str(ds), epochs=a.epochs, patience=a.patience, imgsz=a.imgsz, batch=a.bs, seed=s, deterministic=True,
                    project=str(runs_dir), name=name, exist_ok=True, workers=0, plots=False, verbose=False)
            best = YOLO(str(runs_dir / name / "weights" / "best.pt"))
            def pred(lst):
                ps = []
                for ch in chunks(lst):
                    for r in best.predict(ch, imgsz=a.imgsz, verbose=False): ps.append(CLASSES.index(r.names[int(r.probs.top1)]))
                return ps
            p = pred(paths["test"]); r = {"seed": s, "test": metrics(y["test"], p), "test_bootstrap": boot(y["test"], p, seed=s), "train_seconds": round(time.time() - t0, 1)}
            if "extra" in sp: r["extra_test"] = metrics(y["extra"], pred(paths["extra"]))
            json.dump([{"image_id": rr["image_id"], "label": rr["label"], "pred": CLASSES[k]} for rr, k in zip(sp["test"], p)], open(runs_dir / name / "test_preds.json", "w"), indent=1)
            cruns.append(r); print(f"[yolo-cls seed {s}] TEST acc {r['test']['accuracy']:.3f} mF1 {r['test']['macro_f1']:.3f}", flush=True)
        out["cls"] = {"model": a.cls_model + " (ImageNet-pretrained)", "config": {"epochs": a.epochs, "patience": a.patience, "imgsz": a.imgsz, "batch": a.bs,
                      "notes": "Ultralytics classify defaults (SGD/AdamW auto, RandAugment, erasing); early stop / best.pt on val top-1; no class weighting"},
                      "runs": cruns, "spread": spread(cruns)}

    # ---------- YOLO-World zero-shot ----------
    if not a.skip_world:
        prom = [(t.lower(), c) for c in CLASSES for t in AA.P["device_type"][c]]
        w = YOLO(a.world_model); w.set_classes([t for t, _ in prom])
        def wpred(lst):
            best_conf = []
            for ch in chunks(lst, 16):
                for r in w.predict(ch, conf=a.world_conf, verbose=False):
                    if r.boxes is None or len(r.boxes) == 0: best_conf.append((NONE, 0.0)); continue
                    j = int(r.boxes.conf.argmax()); best_conf.append((CLASSES.index(prom[int(r.boxes.cls[j])][1]), float(r.boxes.conf[j])))
            return best_conf
        bc = wpred(paths["test"]); p = [k for k, _ in bc]
        p25 = [k if c >= 0.25 else NONE for k, c in bc]
        wr = {"seed": None, "test": met(y["test"], p), "test_bootstrap": boot(y["test"], p)}
        if "extra" in sp: wr["extra_test"] = met(y["extra"], [k for k, _ in wpred(paths["extra"])])
        out["world"] = {"model": a.world_model, "prompts": "auto_annotate.P['device_type'] (" + AA.PROMPT_VERSION + "), " + str(len(prom)) + " text classes",
                        "conf": a.world_conf, "no_box_rate": round(wr["test"]["no_box"] / len(p), 4), "runs": [wr], "spread": spread([wr]),
                        "at_conf_0.25": {"accuracy": met(y["test"], p25)["accuracy"], "macro_f1": met(y["test"], p25)["macro_f1"], "no_box_rate": round(sum(v == NONE for v in p25) / len(p25), 4)}}
        print("[yolo-world] TEST", round(wr["test"]["accuracy"], 3), round(wr["test"]["macro_f1"], 3), "no-box", out["world"]["no_box_rate"], flush=True)
        del w

    # ---------- preliminary detector ----------
    ids = {r["image_id"] for r in rows}; bx = latest_boxes(ids)
    have = {s: [r for r in sp[s] if r["image_id"] in bx] for s in ("train", "val", "test")}
    out["det_data"] = {"images_with_boxes": {s: len(v) for s, v in have.items()}, "boxes": {s: sum(len(bx[r["image_id"]]) for r in v) for s, v in have.items()}}
    if a.det and len(have["train"]) >= a.min_det_train and have["val"] and have["test"]:
        dd = var / f"yolo_det_{a.tag}"
        for s, v in have.items():
            for r in v:
                link(img / f"{r['image_id']}.jpg", dd / "images" / s / f"{r['image_id']}.jpg")
                lp = dd / "labels" / s / f"{r['image_id']}.txt"; lp.parent.mkdir(parents=True, exist_ok=True)
                k = CLASSES.index(r["label"]); L = []
                for x0, y0, bw, bh in bx[r["image_id"]]:
                    x0, y0 = max(0.0, x0), max(0.0, y0); bw, bh = min(bw, 1 - x0), min(bh, 1 - y0)
                    if bw > 0 and bh > 0: L.append(f"{k} {x0 + bw/2:.6f} {y0 + bh/2:.6f} {bw:.6f} {bh:.6f}")
                lp.write_text("\n".join(L) + "\n")
        yaml = dd / "data.yaml"
        yaml.write_text(f"path: {dd.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
        druns = []
        for si, s in enumerate(int(t) for t in a.det_seeds.split(",")):
            if a.pause: time.sleep(a.pause)
            name = f"yolo11s_det_{a.tag}_seed{s}"; t0 = time.time()
            m = YOLO(a.det_model)
            m.train(data=str(yaml), epochs=a.det_epochs, patience=15, imgsz=640, batch=a.det_bs, seed=s, deterministic=True,
                    project=str(runs_dir), name=name, exist_ok=True, workers=0, plots=False, verbose=False)
            best = YOLO(str(runs_dir / name / "weights" / "best.pt"))
            v = best.val(data=str(yaml), split="test", imgsz=640, batch=a.det_bs, project=str(runs_dir), name=name + "_test", exist_ok=True, plots=False, verbose=False)
            def dpred(lst):
                ps = []
                for ch in chunks(lst, 16):
                    for r in best.predict(ch, conf=0.01, imgsz=640, verbose=False):
                        ps.append(NONE if r.boxes is None or len(r.boxes) == 0 else int(r.boxes.cls[int(r.boxes.conf.argmax())]))
                return ps
            p = dpred(paths["test"])
            r = {"seed": s, "box_test": {"mAP50": round(float(v.box.map50), 4), "mAP50_95": round(float(v.box.map), 4), "n_images": len(have["test"])},
                 "test": met(y["test"], p), "test_bootstrap": boot(y["test"], p, seed=s), "train_seconds": round(time.time() - t0, 1)}
            if "extra" in sp: r["extra_test"] = met(y["extra"], dpred(paths["extra"]))
            druns.append(r); print(f"[yolo-det seed {s}] mAP50 {r['box_test']['mAP50']} img-acc {r['test']['accuracy']:.3f}", flush=True)
        out["det"] = {"PRELIMINARY": True, "model": a.det_model + " (COCO-pretrained)", "epochs": a.det_epochs, "imgsz": 640,
                      "notes": "every human box is labelled with the image's device class (one class per image); images without human boxes excluded from detector sets; image-level class = top box, conf>=0.01; no box = wrong",
                      "runs": druns, "spread": spread(druns)}
    else:
        out["det"] = {"skipped": True, "reason": f"--det not set or too few images with human boxes (train {len(have['train'])} < {a.min_det_train})"}
    res = Path(a.results); res.mkdir(parents=True, exist_ok=True)
    prev = res / f"baseline_yolo_{a.tag}.json"
    if prev.exists() and (a.skip_cls or a.skip_world):   # partial re-run: keep the other sections
        old = json.load(open(prev)); old.update({k: v for k, v in out.items()}); out = old
    json.dump(out, open(prev, "w"), indent=2)
    print(json.dumps({k: out[k].get("spread", out[k]) for k in ("cls", "world", "det") if k in out}, indent=1)[:3000])

if __name__ == "__main__":
    main()
