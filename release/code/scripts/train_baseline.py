#!/usr/bin/env python3
"""ResNet-50 baseline for WheelView v1: 4-way device classification (manual / powered / other_aid / ambiguous;
`ambiguous` was called `lookalike` before 2026-10-28, old split files are mapped on read).

Reads split assignments + labels from the dataset manifest data/manifest_v2.csv (scripts/build_manifest.py; only
train_eligible rows) or, with --splits, an older splits CSV (scripts/make_splits.py), and the face-BLURRED copies in images_blurred/<image_id>.jpg.
Fine-tunes torchvision ResNet-50 (ImageNet weights), 224 px, light augmentation, class-weighted cross-entropy,
AdamW + cosine schedule, early stopping on validation macro-F1, fixed seeds. Evaluates the best-val checkpoint on test.

Local only: weights + logs go to var/runs/<run>/ (gitignored). Only metrics (JSON/MD) and the confusion-matrix PNG
are written to results/. Images never leave the machine.

  python scripts/train_baseline.py --seeds 0,1,2"""
import argparse, csv, json, math, os, platform, random, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torchvision
from torchvision import transforms as T
from PIL import Image
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server")); import wvcore
CLASSES = list(wvcore.DEVICE_CLASSES)   # manual, powered, other_aid, ambiguous (canonical names, fixed order)
DEFAULT_SPLITS = ROOT / "data" / "manifest_v2.csv"

def read_split_rows(path):
    """Rows {image_id, split, label, ...} for train/val/test from the dataset manifest (scripts/build_manifest.py:
    rows with train_eligible=1, label = device_type) or an older splits CSV (label column). Labels are the only
    supervision read and come from this file only; historical class names are canonicalized (lookalike -> ambiguous).
    Test labels are used only to score predictions after training (never for fitting, weighting or model selection)."""
    out = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if "label" not in r:   # manifest
            if str(r.get("train_eligible", "")).strip() not in ("1", "true", "True"): continue
            r["label"] = r["device_type"]
        r["label"] = wvcore.canon_class(r["label"])
        if r.get("split") in ("train", "val", "test") and r["label"] in CLASSES: out.append(r)
    return out

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def metrics(y, p, k=len(CLASSES)):
    cm = np.zeros((k, k), dtype=int)
    for a, b in zip(y, p): cm[a, b] += 1
    per = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per[c] = {"precision": round(float(pr), 4), "recall": round(float(rc), 4), "f1": round(float(f1), 4), "n": int(cm[i].sum())}
    present = [c for c in CLASSES if per[c]["n"] > 0]
    return {"accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
            "macro_f1": float(np.mean([per[c]["f1"] for c in present])) if present else 0.0,
            "per_class": per, "confusion_matrix": cm.tolist()}

def bootstrap(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed); y, p = np.array(y), np.array(p); acc, mf = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); m = metrics(y[i], p[i]); acc.append(m["accuracy"]); mf.append(m["macro_f1"])
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"accuracy_95ci": q(acc), "macro_f1_95ci": q(mf), "n_boot": n}

class DS(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, tf): self.imgs, self.labels, self.tf = imgs, labels, tf
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i): return self.tf(self.imgs[i]), self.labels[i]

def load_split(rows, imgdir, short=256):
    out, lab, ids = [], [], []
    for r in rows:
        fp = imgdir / f"{r['image_id']}.jpg"
        im = Image.open(fp).convert("RGB"); w, h = im.size; s = short / min(w, h)
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
        out.append(im); lab.append(CLASSES.index(r["label"])); ids.append(r["image_id"])
    return out, lab, ids

def run(seed, data, a, dev, outdir):
    set_seed(seed)
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf_tr = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.8, 1.25)), T.RandomHorizontalFlip(),
                       T.ColorJitter(0.2, 0.2, 0.1), T.ToTensor(), norm])
    tf_ev = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), norm])
    g = torch.Generator(); g.manual_seed(seed)
    dl = lambda s, tf, sh: torch.utils.data.DataLoader(DS(data[s][0], data[s][1], tf), batch_size=a.bs, shuffle=sh, generator=g if sh else None, num_workers=0)
    tr, va, te = dl("train", tf_tr, True), dl("val", tf_ev, False), dl("test", tf_ev, False)
    xt = dl("extra", tf_ev, False) if "extra" in data else None
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES)); model.to(dev)
    cnt = np.bincount(data["train"][1], minlength=len(CLASSES)).astype(float)
    w = cnt.sum() / (len(CLASSES) * np.maximum(cnt, 1)); cw = torch.tensor(w, dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=a.label_smoothing)
    head = list(model.fc.parameters()); hid = {id(p) for p in head}
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in hid], "lr": a.lr},
                             {"params": head, "lr": a.lr * a.head_lr_mult}], weight_decay=a.wd)
    steps = a.epochs * len(tr); warm = len(tr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    use_amp = dev.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    def predict(loader):
        model.eval(); ys, ps = [], []
        with torch.no_grad(), torch.autocast(dev.type, enabled=use_amp):
            for x, y in loader:
                ps += model(x.to(dev)).argmax(1).cpu().tolist(); ys += y.tolist()
        return ys, ps
    best, bad, hist, ck = -1, 0, [], outdir / f"seed{seed}_best.pt"
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); tl, n = 0.0, 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev); opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, enabled=use_amp): loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(y); n += len(y)
        vm = metrics(*predict(va))
        hist.append({"epoch": ep, "train_loss": round(tl / n, 4), "val_acc": round(vm["accuracy"], 4), "val_macro_f1": round(vm["macro_f1"], 4)})
        print(f"[seed {seed}] ep {ep:02d} loss {tl/n:.4f} val_acc {vm['accuracy']:.3f} val_mF1 {vm['macro_f1']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if vm["macro_f1"] > best + 1e-6: best, bad, best_ep = vm["macro_f1"], 0, ep; torch.save(model.state_dict(), ck)
        else:
            bad += 1
            if bad >= a.patience: print(f"[seed {seed}] early stop at ep {ep}", flush=True); break
    model.load_state_dict(torch.load(ck, map_location=dev))
    ys, ps = predict(te); tm = metrics(ys, ps)
    preds = [{"image_id": i, "label": CLASSES[y], "pred": CLASSES[p]} for i, y, p in zip(data["test"][2], ys, ps)]
    json.dump(preds, open(outdir / f"seed{seed}_test_preds.json", "w"), indent=1)   # local only (var/)
    out = {"seed": seed, "best_epoch": best_ep, "epochs_run": len(hist), "best_val_macro_f1": round(best, 4),
           "test": tm, "test_bootstrap": bootstrap(ys, ps, seed=seed), "train_seconds": round(time.time() - t0, 1), "history": hist}
    if xt is not None:
        ey, ep_ = predict(xt); out["extra_test"] = metrics(ey, ep_)
        json.dump([{"image_id": i, "label": CLASSES[y], "pred": CLASSES[p]} for i, y, p in zip(data["extra"][2], ey, ep_)],
                  open(outdir / f"seed{seed}_extra_test_preds.json", "w"), indent=1)
    return out

def plot_cm(cm, path, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cm = np.array(cm); fig, ax = plt.subplots(figsize=(5, 4.4), dpi=150)
    rn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if rn[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30); ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=9); fig.tight_layout(); fig.savefig(path); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", "--manifest", dest="splits", default=DEFAULT_SPLITS, help="dataset manifest (default) or an older splits CSV")
    ap.add_argument("--images", default=ROOT / "images_blurred")
    ap.add_argument("--run", default="resnet50_v1"); ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head_lr_mult", type=float, default=10.0); ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--results", default=ROOT / "results")
    ap.add_argument("--extra_test", default=None, help="older splits CSV; its test rows not in this run's train/val are also evaluated")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = ROOT / "var" / "runs" / a.run; outdir.mkdir(parents=True, exist_ok=True)
    rows = read_split_rows(a.splits)
    imgdir = Path(a.images)
    missing = [r["image_id"] for r in rows if not (imgdir / f"{r['image_id']}.jpg").exists()]
    if missing: sys.exit(f"{len(missing)} blurred images missing from {imgdir}, e.g. {missing[:5]}")
    data = {s: load_split([r for r in rows if r["split"] == s], imgdir) for s in ("train", "val", "test")}
    extra_info = None
    if a.extra_test:
        seen = {r["image_id"] for r in rows if r["split"] in ("train", "val")}
        old = [r for r in read_split_rows(a.extra_test) if r["split"] == "test"]
        keep = [r for r in old if r["image_id"] not in seen and r["label"] in CLASSES]
        data["extra"] = load_split(keep, imgdir)
        extra_info = {"file": Path(a.extra_test).name, "n_old_test": len(old), "n_evaluated": len(keep),
                      "n_dropped_now_in_train_or_val": len(old) - len(keep)}
        print("extra test:", extra_info, flush=True)
    n_per = {s: {c: data[s][1].count(i) for i, c in enumerate(CLASSES)} for s in ("train", "val", "test")}
    print("n per split/class:", n_per, "device:", dev, flush=True)
    hw = {"device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
          "cpu": platform.processor(), "os": platform.platform(), "python": platform.python_version(),
          "torch": torch.__version__, "torchvision": torchvision.__version__}
    seeds = [int(s) for s in a.seeds.split(",")]; runs = []; t0 = time.time()
    for s in seeds:
        r = run(s, data, a, dev, outdir); runs.append(r)
        print(f"[seed {s}] TEST acc {r['test']['accuracy']:.3f} macro-F1 {r['test']['macro_f1']:.3f} best_ep {r['best_epoch']}", flush=True)
    agg = lambda f: {"mean": round(float(np.mean([f(r) for r in runs])), 4), "std": round(float(np.std([f(r) for r in runs], ddof=1)), 4) if len(runs) > 1 else 0.0}
    summary = {"model": "torchvision resnet50, IMAGENET1K_V2 weights, full fine-tune", "task": "4-way device class",
               "classes": CLASSES, "splits_file": "data/" + Path(a.splits).name, "extra_test": extra_info, "images": "images_blurred/ (face-blurred copies)",
               "n_per_split": n_per,
               "hyperparams": {k: v for k, v in vars(a).items() if k in ("epochs", "patience", "bs", "lr", "head_lr_mult", "wd", "label_smoothing")} |
                   {"optimizer": "AdamW", "schedule": "1-epoch linear warmup + cosine", "loss": "class-weighted CE (inverse train frequency)",
                    "input": "224px; train RandomResizedCrop(0.6-1.0)+HFlip+ColorJitter(0.2,0.2,0.1); eval Resize(224)+CenterCrop(224)",
                    "early_stopping": "best val macro-F1", "amp": dev.type == "cuda"},
               "seeds": seeds, "primary_seed": seeds[0], "primary": {k: v for k, v in runs[0].items() if k != "history"},
               "seed_spread": {"test_accuracy": agg(lambda r: r["test"]["accuracy"]), "test_macro_f1": agg(lambda r: r["test"]["macro_f1"]),
                               **{f"test_f1_{c}": agg(lambda r, c=c: r["test"]["per_class"][c]["f1"]) for c in CLASSES},
                               **({"extra_test_accuracy": agg(lambda r: r["extra_test"]["accuracy"]),
                                   "extra_test_macro_f1": agg(lambda r: r["extra_test"]["macro_f1"])} if extra_info else {})},
               "runs": runs, "hardware": hw, "total_seconds": round(time.time() - t0, 1)}
    res = Path(a.results); res.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(res / f"baseline_{a.run}.json", "w"), indent=2)
    p = runs[0]["test"]
    plot_cm(p["confusion_matrix"], res / f"baseline_{a.run}_confusion.png", f"ResNet-50 baseline, test (n={sum(map(sum, p['confusion_matrix']))}), seed {seeds[0]}")
    ss = summary["seed_spread"]; bs = runs[0]["test_bootstrap"]
    md = [f"# ResNet-50 baseline ({a.run})", "", f"4-way device classification on WheelView, splits `{Path(a.splits).name}` (face-blurred images). "
          "Generated by `scripts/train_baseline.py`; splits from `scripts/make_splits.py`.", "",
          "| split | " + " | ".join(CLASSES) + " | total |", "|---|" + "---|" * (len(CLASSES) + 1)]
    for s in ("train", "val", "test"): md.append(f"| {s} | " + " | ".join(str(n_per[s][c]) for c in CLASSES) + f" | {sum(n_per[s].values())} |")
    md += ["", f"## Test results, primary seed {seeds[0]}", "",
           f"- Accuracy: **{p['accuracy']:.3f}** (bootstrap 95% CI {bs['accuracy_95ci'][0]:.2f}-{bs['accuracy_95ci'][1]:.2f})",
           f"- Macro-F1: **{p['macro_f1']:.3f}** (bootstrap 95% CI {bs['macro_f1_95ci'][0]:.2f}-{bs['macro_f1_95ci'][1]:.2f})", "",
           "| class | precision | recall | F1 | n (test) |", "|---|---|---|---|---|"]
    for c in CLASSES: q = p["per_class"][c]; md.append(f"| {c} | {q['precision']:.3f} | {q['recall']:.3f} | {q['f1']:.3f} | {q['n']} |")
    md += ["", "Confusion matrix (rows = true, columns = predicted): `" + json.dumps(p["confusion_matrix"]) + "`; see "
           f"`baseline_{a.run}_confusion.png`.", "", f"## Seed spread ({len(seeds)} seeds: {', '.join(map(str, seeds))}; same split, mean ± sd)", "",
           f"- Test accuracy: {ss['test_accuracy']['mean']:.3f} ± {ss['test_accuracy']['std']:.3f}",
           f"- Test macro-F1: {ss['test_macro_f1']['mean']:.3f} ± {ss['test_macro_f1']['std']:.3f}"]
    md += [f"- F1 {c}: {ss['test_f1_'+c]['mean']:.3f} ± {ss['test_f1_'+c]['std']:.3f}" for c in CLASSES]
    if extra_info:
        md += ["", f"## Older test set ({extra_info['file']})", "",
               f"{extra_info['n_evaluated']} of its {extra_info['n_old_test']} test images (the {extra_info['n_dropped_now_in_train_or_val']} now in this run's train/val are dropped to avoid leakage):",
               f"accuracy {ss['extra_test_accuracy']['mean']:.3f} ± {ss['extra_test_accuracy']['std']:.3f}, macro-F1 {ss['extra_test_macro_f1']['mean']:.3f} ± {ss['extra_test_macro_f1']['std']:.3f}."]
    md += ["", "## Setup", "", "```json", json.dumps(summary["hyperparams"], indent=1), "```", "",
           f"Hardware: {hw['gpu'] or hw['cpu']} ({hw['device']}), torch {hw['torch']}, torchvision {hw['torchvision']}. "
           f"Total wall time {summary['total_seconds']:.0f}s.", "",
           "## Caveats", "", "- The test set is small, so the bootstrap intervals are wide; rare classes (ambiguous, other_aid) "
           "have only a handful of test images and their per-class F1 can swing a lot with one prediction.",
           "- Seed spread only reflects training randomness on one fixed split, not split-to-split variance.",
           "- Splits are grouped by Flickr uploader/creator; one uploader contributes a large share of images and sits entirely in one split.",
           "- Most labels are single-annotator; see `results/" + Path(a.splits).stem + "_summary.json` for label resolution counts."]
    open(res / f"baseline_{a.run}.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(json.dumps({"primary_test": {"acc": p["accuracy"], "macro_f1": p["macro_f1"]}, "seed_spread": ss}, indent=1))

if __name__ == "__main__":
    main()
