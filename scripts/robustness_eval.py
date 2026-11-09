#!/usr/bin/env python3
"""Robustness of the fine-tuned SigLIP 2 (last 4 blocks) baseline to image degradations on the frozen v1.0 test split.

Inference only: saved checkpoints + test images; corruptions are applied to TEST images in memory (nothing written back,
no retraining). Preprocessing is identical to scripts/predict.py / train_siglip.py eval (short side 256 -> Resize 224x224,
mean/std 0.5). Corruptions are applied to the loaded release image before that preprocessing.

  python scripts/robustness_eval.py --root C:/Users/kamoo/wheelview --out C:/Users/kamoo/wheelview/var/robustness_v1

Outputs (untracked): results.csv, per_image.csv, quality.csv, summary.json, robustness_macro_f1.{pdf,png}, latex_draft.tex
"""
import argparse, csv, io, json, math, sys, time
from pathlib import Path
import numpy as np

SEED = 20261107
CORR = {   # 5 severities each (ImageNet-C style ordering, mild -> severe)
    "gaussian_blur": [1.0, 2.0, 3.0, 4.5, 6.0],               # sigma (px, on the ~640px image)
    "motion_blur":   [7, 11, 17, 25, 35],                     # kernel length (px), random angle per image (seeded)
    "gaussian_noise":[0.04, 0.08, 0.12, 0.18, 0.26],          # sigma on [0,1] intensities
    "low_light":     [(0.80, 1.4, 0.01), (0.65, 1.8, 0.02), (0.50, 2.2, 0.03), (0.35, 2.6, 0.04), (0.25, 3.0, 0.05)],  # (gain, gamma, noise sigma)
    "jpeg":          [25, 18, 15, 10, 7],                     # JPEG quality (ImageNet-C levels)
}

def load_labels(root):
    lab = {}
    with open(root / "release/annotations/labels.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh): lab[r["image_id"]] = r
    return lab

def child_flagged(root):
    """Human exclude=child, adjudicated child, safety_hold: these stay excluded from the robustness experiment."""
    ids = set()
    for f in (root / "annotations/events").glob("*.jsonl"):
        if "deploytest" in f.name: continue
        for line in open(f, encoding="utf-8"):
            if not line.strip(): continue
            try: e = json.loads(line)
            except ValueError: continue
            if e.get("field") == "exclude" and e.get("value") == "child": ids.add(e["image_id"])
    sh = root / "data/safety_hold.csv"
    if sh.exists():
        with open(sh, encoding="utf-8") as fh:
            ids |= {r["image_id"] for r in csv.DictReader(fh) if r.get("image_id")}
    return ids

# ---------------------------------------------------------------- corruptions (float32 RGB in [0,1])
def to_f(im): return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
def to_im(a):
    from PIL import Image
    return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))

def corrupt(im, kind, p, rng):
    import cv2
    if kind == "clean": return im
    a = to_f(im)
    if kind == "gaussian_blur":
        a = cv2.GaussianBlur(a, (0, 0), sigmaX=p, sigmaY=p, borderType=cv2.BORDER_REFLECT)
    elif kind == "motion_blur":
        L = int(p); k = np.zeros((L, L), np.float32); k[L // 2, :] = 1.0
        ang = float(rng.uniform(0, 180)); M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), ang, 1.0)
        k = cv2.warpAffine(k, M, (L, L)); k /= max(k.sum(), 1e-8)
        a = cv2.filter2D(a, -1, k, borderType=cv2.BORDER_REFLECT)
    elif kind == "gaussian_noise":
        a = a + rng.normal(0, p, a.shape).astype(np.float32)
    elif kind == "low_light":
        gain, gamma, s = p
        a = gain * np.power(np.clip(a, 0, 1), gamma) + rng.normal(0, s, a.shape).astype(np.float32)
    elif kind == "jpeg":
        buf = io.BytesIO(); to_im(a).save(buf, "JPEG", quality=int(p)); buf.seek(0)
        from PIL import Image
        return Image.open(buf).convert("RGB")
    else:
        raise ValueError(kind)
    return to_im(a)

# ---------------------------------------------------------------- quality metrics (clean images, descriptive)
STD_LUMA_Q = np.array([16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,24,40,57,69,56,14,17,22,29,51,87,80,62,
                       18,22,37,56,68,109,103,77,24,35,55,64,81,104,113,92,49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99], float)
def jpeg_quality(path):
    from PIL import Image
    im = Image.open(path); q = getattr(im, "quantization", None)
    if not q or 0 not in q: return float("nan")
    t = np.array(q[0], float)[:64]; s = 100.0 * t.mean() / STD_LUMA_Q.mean()   # libjpeg scale factor
    return float(np.clip((200 - s) / 2 if s <= 100 else 5000 / s, 1, 100))
def quality(path):
    import cv2
    from PIL import Image
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    lap = float(cv2.Laplacian(g, cv2.CV_64F).var())
    # Immerkaer (1996) fast noise sigma estimate
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], float)
    H, W = g.shape; conv = cv2.filter2D(g, -1, M)[1:-1, 1:-1]
    sigma = float(np.sqrt(math.pi / 2) * np.abs(conv).sum() / (6 * (W - 2) * (H - 2)))
    return {"laplacian_var": lap, "noise_sigma": sigma, "mean_luminance": float(g.mean()), "jpeg_quality_est": jpeg_quality(path)}

# ---------------------------------------------------------------- metrics
def macro_f1(y, p, k):
    f = []
    for c in range(k):
        n = int((y == c).sum())
        if n == 0: continue
        tp = int(((y == c) & (p == c)).sum()); fp = int(((y != c) & (p == c)).sum()); fn = n - tp
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / n
        f.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f))

def seed_mean_metrics(y, P, k, idx=None):
    if idx is not None: y = y[idx]; P = P[:, idx]
    acc = float(np.mean([(p == y).mean() for p in P])); mf = float(np.mean([macro_f1(y, p, k) for p in P]))
    return acc, mf

def boot(y, P, k, n=1000, seed=SEED):
    rng = np.random.default_rng(seed); A, F = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); a, f = seed_mean_metrics(y, P, k, i); A.append(a); F.append(f)
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return q(A), q(F)

# ---------------------------------------------------------------- model
def load_models(ckpts, base, dev):
    import torch
    from transformers import AutoModel
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import auto_annotate as AA
    models = []
    for ck in ckpts:
        m = AutoModel.from_pretrained(base).eval(); vm = m.vision_model.to(dev)
        sd = torch.load(ck, map_location="cpu", weights_only=True)
        n = len([kk for kk in sd if kk.startswith("m")]); has_head = getattr(vm, "head", None) is not None
        kb = n - 1 - int(has_head)
        mods = list(vm.encoder.layers[-kb:]) + [vm.post_layernorm] + ([vm.head] if has_head else [])
        for j, mm in enumerate(mods): mm.load_state_dict(sd[f"m{j}"])
        hw = sd["head"]["weight"]; head = torch.nn.Linear(hw.shape[1], hw.shape[0]); head.load_state_dict(sd["head"]); head.eval().to(dev)
        vm.eval(); models.append((vm, head, kb)); del m
    return models, AA.feats

def predict_batch(models, feats, ims, dev, bs=16):
    import torch, torchvision.transforms as T
    from PIL import Image
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    def short_side(im, short=256):
        w, h = im.size; s = short / min(w, h)
        return im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    X = torch.stack([tf(short_side(im)) for im in ims])
    out = []
    with torch.inference_mode():
        for vm, head, _ in models:
            ps = []
            for i in range(0, len(X), bs):
                x = X[i:i + bs].to(dev)
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    z = head(feats(vm(pixel_values=x)).float())
                ps += z.argmax(1).cpu().tolist()
            out.append(ps)
    return np.array(out)   # seeds x n

# ---------------------------------------------------------------- figure + latex
def figure(rows, clean, outdir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.5, 2.6), dpi=300)
    names = {"gaussian_blur": "Gaussian blur", "motion_blur": "Motion blur", "gaussian_noise": "Gaussian noise",
             "low_light": "Low light", "jpeg": "JPEG"}
    marks = {"gaussian_blur": "o", "motion_blur": "s", "gaussian_noise": "^", "low_light": "D", "jpeg": "v"}
    for kind in CORR:
        rs = [clean] + sorted([r for r in rows if r["corruption"] == kind], key=lambda r: r["severity"])
        x = [r["severity"] for r in rs]; y = [r["macro_f1"] for r in rs]
        lo = [r["macro_f1_lo"] for r in rs]; hi = [r["macro_f1_hi"] for r in rs]
        l, = ax.plot(x, y, marker=marks[kind], ms=3, lw=1.1, label=names[kind]); ax.fill_between(x, lo, hi, color=l.get_color(), alpha=0.12, lw=0)
    ax.set_xlabel("Severity (0 = clean)"); ax.set_ylabel("Macro-F1 (4 classes)"); ax.set_xticks(range(6)); ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False, loc="lower left"); fig.tight_layout()
    fig.savefig(outdir / "robustness_macro_f1.pdf"); fig.savefig(outdir / "robustness_macro_f1.png"); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:/Users/kamoo/wheelview")
    ap.add_argument("--run", default="var/runs/siglip2_v3")
    ap.add_argument("--base", default="google/siglip2-so400m-patch14-224")
    ap.add_argument("--out", default=r"C:/Users/kamoo/wheelview/var/robustness_v1")
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()
    import torch
    from PIL import Image
    root = Path(a.root); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "server")); import wvcore
    CLASSES = list(wvcore.DEVICE_CLASSES); K = len(CLASSES)
    lab = load_labels(root)
    test = [r for r in lab.values() if r["split"] == "test"]
    test.sort(key=lambda r: r["image_id"])
    excl = child_flagged(root)
    ckpts = sorted((root / a.run).glob("ft_seed*_best.pt"))
    assert len(ckpts) == 3, ckpts
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, feats = load_models(ckpts, a.base, dev)
    paths = {r["image_id"]: root / "var/release/WheelView_v1.0" / r["file"] for r in test}
    if not all(p.exists() for p in paths.values()):
        paths = {r["image_id"]: root / "release" / r["file"] for r in test}
    clean_ims = {i: Image.open(p).convert("RGB") for i, p in paths.items()}
    ids_all = [r["image_id"] for r in test]; y_all = np.array([CLASSES.index(wvcore.canon_class(r["device_type"])) for r in test])

    # 1) clean reproduction on the full frozen n=64 test split
    P64 = predict_batch(models, feats, [clean_ims[i] for i in ids_all], dev)
    per_seed = [{"seed": s, "accuracy": float((P64[s] == y_all).mean()), "macro_f1": macro_f1(y_all, P64[s], K)} for s in range(len(ckpts))]
    repro = {"n": len(ids_all), "per_seed": per_seed,
             "accuracy_mean": float(np.mean([r["accuracy"] for r in per_seed])), "accuracy_sd": float(np.std([r["accuracy"] for r in per_seed], ddof=1)),
             "macro_f1_mean": float(np.mean([r["macro_f1"] for r in per_seed])), "macro_f1_sd": float(np.std([r["macro_f1"] for r in per_seed], ddof=1)),
             "reported": {"accuracy": "0.896 +- .018", "macro_f1": "0.836 +- .074"}}
    # agreement with the stored release predictions
    agree = []
    for s, ck in enumerate(ckpts):
        f = root / "results/preds_v3" / f"siglip2_v3__ft_seed{s}_test_preds.csv"
        if f.exists():
            st = {r["image_id"]: r["device_type"] for r in csv.DictReader(open(f, encoding="utf-8"))}
            agree.append(float(np.mean([CLASSES[P64[s][j]] == wvcore.canon_class(st.get(i, "")) for j, i in enumerate(ids_all) if i in st])))
    repro["agreement_with_stored_preds_per_seed"] = agree
    print("CLEAN n=64", json.dumps(repro, indent=1), flush=True)

    # 2) robustness subset: drop human/adjudicated child + safety-held images
    keep = [j for j, i in enumerate(ids_all) if i not in excl]
    ids = [ids_all[j] for j in keep]; y = y_all[keep]
    dropped = [i for i in ids_all if i in excl]
    print("robustness n", len(ids), "dropped child/safety", dropped, flush=True)

    rows, per_img = [], []
    def run(kind, sev, param, ims_fn):
        P = predict_batch(models, feats, ims_fn(), dev)
        acc, mf = seed_mean_metrics(y, P, K); (alo, ahi), (flo, fhi) = boot(y, P, K, a.n_boot)
        r = {"corruption": kind, "severity": sev, "param": json.dumps(param), "n": len(y), "accuracy": acc, "accuracy_lo": alo, "accuracy_hi": ahi,
             "macro_f1": mf, "macro_f1_lo": flo, "macro_f1_hi": fhi,
             "accuracy_seeds": json.dumps([round(float((p == y).mean()), 4) for p in P]),
             "macro_f1_seeds": json.dumps([round(macro_f1(y, p, K), 4) for p in P])}
        rows.append(r); print(kind, sev, f"acc {acc:.3f} [{alo:.3f},{ahi:.3f}] mF1 {mf:.3f} [{flo:.3f},{fhi:.3f}]", flush=True)
        for j, i in enumerate(ids):
            per_img.append({"corruption": kind, "severity": sev, "image_id": i, "label": CLASSES[y[j]],
                            **{f"pred_seed{s}": CLASSES[P[s][j]] for s in range(len(P))}})
        return r, P
    clean_row, Pc = run("clean", 0, None, lambda: [clean_ims[i] for i in ids])
    for kind, levels in CORR.items():
        for sev, p in enumerate(levels, 1):
            rng = np.random.default_rng(SEED + 1000 * list(CORR).index(kind) + sev)
            run(kind, sev, p, lambda: [corrupt(clean_ims[i], kind, p, rng) for i in ids])
    # face-blurred release copy vs the original (unblurred) thumbnail
    orig = {i: root / "images" / f"{i}.jpg" for i in ids}
    if all(p.exists() for p in orig.values()):
        run("unblurred_original", 0, "images/<id>.jpg (not face-blurred)", lambda: [Image.open(orig[i]).convert("RGB") for i in ids])

    # 3) quality metrics on the clean (release) test images vs correctness (mean over seeds)
    from scipy import stats
    corr_rate = (Pc == y[None, :]).mean(0)
    q = []
    for j, i in enumerate(ids):
        m = quality(paths[i]); m.update({"image_id": i, "label": CLASSES[y[j]], "correct_rate": float(corr_rate[j])}); q.append(m)
    qc = {}
    for k in ("laplacian_var", "noise_sigma", "mean_luminance", "jpeg_quality_est"):
        v = np.array([r[k] for r in q], float); ok = ~np.isnan(v)
        if ok.sum() < 5 or np.std(v[ok]) == 0 or np.std(corr_rate[ok]) == 0:
            qc[k] = {"n": int(ok.sum()), "note": "constant or too few values"}; continue
        rs, ps = stats.spearmanr(v[ok], corr_rate[ok]); rp, pp = stats.pearsonr(np.log1p(v[ok]) if k == "laplacian_var" else v[ok], corr_rate[ok])
        qc[k] = {"n": int(ok.sum()), "spearman_rho": float(rs), "spearman_p": float(ps), "pearson_r": float(rp), "pearson_p": float(pp),
                 "median": float(np.median(v[ok])), "iqr": [float(np.percentile(v[ok], 25)), float(np.percentile(v[ok], 75))]}

    # write outputs
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(per_img[0])); w.writeheader(); w.writerows(per_img)
    with open(out / "quality.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(q[0])); w.writeheader(); w.writerows(q)
    figure([r for r in rows if r["corruption"] in CORR], clean_row, out)
    summ = {"seed": SEED, "n_boot": a.n_boot, "checkpoints": [str(c.relative_to(root)) for c in ckpts], "base": a.base,
            "test_images": "var/release/WheelView_v1.0 images/test (face-blurred release copies)",
            "clean_reproduction_n64": repro, "robustness_n": len(ids), "robustness_dropped_child_or_safety": dropped,
            "severities": {k: v for k, v in CORR.items()}, "quality_correlations": qc,
            "ci_method": "bootstrap over test images (1000 resamples); metric = mean over the 3 fine-tune seeds per resample"}
    json.dump(summ, open(out / "summary.json", "w", encoding="utf-8"), indent=1)
    print("QUALITY", json.dumps(qc, indent=1))

if __name__ == "__main__":
    main()
