#!/usr/bin/env python3
"""MAID v1.1 multi-task heads on a frozen SigLIP 2 backbone + per-attribute evaluation.

  python scripts/train_multitask.py --dry-run                     # label counts per attribute/split, no model loaded
  python scripts/train_multitask.py --manifest var/v11_scratch/manifest_v11.csv --run mt_smoke

  python scripts/train_multitask.py --synthetic var/synthetic/synthetic_manifest.csv --run mt_syn   # + synthetic TRAIN rows

Tasks: device_type (v1.0 label) + every single-choice v1.1 attribute (control, environment, weather, ground_wet,
size_class, chair_subtype, drive_position, ramp_deploy) + one binary yes/no task per option of each multi-select field
(ramp_reasons=obstruction, ramp_reasons=steep_slope, ...). chair_subtype / drive_position are dependent fields: rows
whose device_type is not manual / powered carry no value and contribute nothing to those heads.
Model: one shared frozen image encoder (google/siglip2-so400m-patch14-224, as auto_annotate) and one head per task:
  * probe (default): per-task logistic regression on the frozen embedding, C chosen on val macro-F1.
  * joint (--joint): shared 1-hidden-layer adapter + one linear head per task, masked cross-entropy: rows whose label
    for a task is empty (not final) or 'unknown' contribute no loss for that task.
Guards: a task is trained only if it has >= --min-train labelled train rows and >= 2 classes with >= --min-class rows
in train; it is evaluated only if val and test each have >= --min-eval rows. Otherwise the task is reported as
insufficient_labels and skipped. If no task qualifies the script exits 0 without loading the model.

Leakage: splits come from the v1.1 manifest only (frozen v1.0 splits; contributed images grouped per creator);
fit() sees train only; C and early stopping use val only; test is scored once at the end. Embeddings are computed
in memory per run and NOT cached or read back; the saved heads are the models (inference = encoder + head).
Outputs (local only): var/runs/<run>/mt_<task>_head.npz (or mt_joint.pt) and var/runs/<run>/metrics_v11.json.
Nothing is written to results/ (v1.0 results stay frozen).
Synthetic (--synthetic): rows of scripts/synth_augment.py's manifest are added to TRAIN only, after checking that each
is split=train and derived from a train image of this manifest; val/test (and every reported metric) stay real-only."""
import argparse, csv, json, os, random, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server")); sys.path.insert(0, str(ROOT / "scripts"))
import wvcore

DEFAULT_MANIFEST = ROOT / "data" / "manifest_v11.csv"
MULTI_TASKS = {f"{f}={o}": (f, o) for f in wvcore.COMPARE_V11 if f in wvcore.MULTI_FIELDS for o in wvcore.FIELD_OPTS[f] if o != "unknown"}
TASKS = ["device_type"] + [f for f in wvcore.COMPARE_V11 if f not in wvcore.MULTI_FIELDS] + list(MULTI_TASKS)
IGNORE = ("", "unknown")
SPLITS = ("train", "val", "test")

def read_rows(path):
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r.get("split") in SPLITS]
    for r in rows: r["device_type"] = wvcore.canon_class(r.get("device_type", "")); derive_multi(r)
    return rows

def derive_multi(r):
    """Multi-select value 'a+b' -> one yes/no column per option ('' when the field is empty or 'unknown')."""
    for name, (f, o) in MULTI_TASKS.items():
        v = [x for x in (r.get(f) or "").split("+") if x]
        r[name] = "" if not v or v == ["unknown"] else ("yes" if o in v else "no")

SYNTH_SCENE_FIELDS = ("environment", "weather", "ground_wet", "ramp_deploy", "ramp_reasons", "setting", "lighting")

def read_synthetic(path, rows):
    """Synthetic rows (train only). Refuses rows that are not split=train or whose source is not a train image here."""
    train = {r["image_id"]: r for r in rows if r["split"] == "train"}
    real = {r["image_id"] for r in rows}; out = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("synthetic") not in ("1", "true", "True"): continue
        assert r.get("split") == "train", f"synthetic {r['image_id']} has split {r.get('split')!r}: synthetic images are train-only"
        assert r.get("source_image_id") in train, f"synthetic {r['image_id']} derives from {r.get('source_image_id')}, not a train image of the manifest"
        assert r["image_id"] not in real, f"synthetic id {r['image_id']} collides with a real image"
        src = train[r["source_image_id"]]
        row = {**src, **{k: v for k, v in r.items() if v != ""}, "split": "train", "v11_eligible": src.get("v11_eligible", ""),
               "train_eligible_v10": src.get("train_eligible_v10", ""), "_path": str((Path(path).parent / r.get("file", "")).resolve()) if r.get("file") else ""}
        for k in SYNTH_SCENE_FIELDS: row[k] = r.get(k, "")   # scene-dependent labels never inherit from the source photo
        row["device_type"] = wvcore.canon_class(row.get("device_type", "")); derive_multi(row); out.append(row)
    return out

def task_rows(rows, task):
    """Rows usable for one task: v1.0 device_type needs train_eligible_v10 or v11_eligible; v1.1 tasks need v11_eligible."""
    ok = (lambda r: r.get("train_eligible_v10") == "1" or r.get("v11_eligible") == "1") if task == "device_type" else (lambda r: r.get("v11_eligible") == "1")
    return {s: [r for r in rows if r["split"] == s and ok(r) and r.get(task, "") not in IGNORE] for s in SPLITS}

def plan(rows, a):
    out = {}
    for t in TASKS:
        tr = task_rows(rows, t); cnt = {s: dict(Counter(r[t] for r in tr[s])) for s in SPLITS}
        classes = sorted(c for c, n in cnt["train"].items() if n >= a.min_class)
        why = ""
        if len(tr["train"]) < a.min_train: why = f"train has {len(tr['train'])} labelled rows (< {a.min_train})"
        elif len(classes) < 2: why = f"fewer than 2 classes with >= {a.min_class} train rows"
        elif min(len(tr["val"]), len(tr["test"])) < a.min_eval: why = f"val/test have {len(tr['val'])}/{len(tr['test'])} rows (< {a.min_eval})"
        out[t] = {"counts": cnt, "classes": classes, "status": "train" if not why else "insufficient_labels", "why": why}
    return out

def scores(y, p, classes):
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    lab = list(range(len(classes)))
    per = f1_score(y, p, labels=lab, average=None, zero_division=0)
    return {"n": int(len(y)), "accuracy": round(float(accuracy_score(y, p)), 4),
            "macro_f1": round(float(f1_score(y, p, labels=lab, average="macro", zero_division=0)), 4),
            "per_class_f1": {c: round(float(f), 4) for c, f in zip(classes, per)},
            "support": {c: int((np.asarray(y) == k).sum()) for k, c in enumerate(classes)},
            "confusion": confusion_matrix(y, p, labels=lab).tolist()}

def boot(y, p, classes, n=1000, seed=0):
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed); y, p = np.asarray(y), np.asarray(p); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); v.append(f1_score(y[i], p[i], labels=list(range(len(classes))), average="macro", zero_division=0))
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

def embed(ids, imgdir, model, bs, paths=None):
    import torch
    from transformers import AutoModel, AutoProcessor
    import auto_annotate as AA
    from PIL import Image
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = AutoModel.from_pretrained(model).eval().to(dev); proc = AutoProcessor.from_pretrained(model); out = []
    for k in range(0, len(ids), bs):
        ims = [Image.open((paths or {}).get(i) or Path(imgdir) / f"{i}.jpg").convert("RGB") for i in ids[k:k + bs]]
        x = proc(images=ims, return_tensors="pt").to(dev)
        with torch.inference_mode():
            e = AA.feats(m.get_image_features(**x))
        out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--images", default=os.environ.get("WV_BLUR_DIR", str(ROOT / "images_blurred")))
    ap.add_argument("--model", default="google/siglip2-so400m-patch14-224")
    ap.add_argument("--run", default="mt_v11"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--min-train", type=int, default=30); ap.add_argument("--min-class", type=int, default=5)
    ap.add_argument("--min-eval", type=int, default=10)
    ap.add_argument("--joint", action="store_true"); ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--synthetic", help="synthetic manifest (scripts/synth_augment.py); its rows are added to TRAIN only")
    a = ap.parse_args()
    if not os.path.exists(a.manifest): sys.exit(f"{a.manifest} missing: run scripts/build_manifest_v11.py first")
    rows = read_rows(a.manifest); want = [t for t in a.tasks.split(",") if t in TASKS]
    syn = read_synthetic(a.synthetic, rows) if a.synthetic else []
    if syn: print(f"+ {len(syn)} synthetic train rows from {a.synthetic} (val/test stay real-only)"); rows = rows + syn
    assert all(r["split"] == "train" for r in rows if r.get("_path")), "synthetic rows outside train"
    P = {t: v for t, v in plan(rows, a).items() if t in want}
    for t, v in P.items(): print(f"{t:12s} {v['status']:20s} train={sum(v['counts']['train'].values())} val={sum(v['counts']['val'].values())} test={sum(v['counts']['test'].values())} {v['why']}")
    go = [t for t, v in P.items() if v["status"] == "train"]
    outdir = ROOT / "var" / "runs" / a.run
    res = {"manifest": os.path.basename(a.manifest), "synthetic_train_rows": len(syn), "metrics_on": "real val/test images only", "model": a.model, "seed": a.seed, "mode": "joint" if a.joint else "probe",
           "guards": {"min_train": a.min_train, "min_class": a.min_class, "min_eval": a.min_eval}, "tasks": P}
    if a.dry_run or not go:
        if not go: print("no task has enough final labels yet -> nothing trained (label more in the Attributes v1.1 pass)")
        if not a.dry_run:
            outdir.mkdir(parents=True, exist_ok=True); res["trained"] = False
            json.dump(res, open(outdir / "metrics_v11.json", "w"), indent=1)
        return
    random.seed(a.seed); np.random.seed(a.seed)
    split_rows = {s: [r for r in rows if r["split"] == s] for s in SPLITS}
    paths = {r["image_id"]: r["_path"] for r in rows if r.get("_path")}
    t0 = time.time(); E = {s: embed([r["image_id"] for r in split_rows[s]], a.images, a.model, a.bs, paths) for s in SPLITS}
    idx = {s: {r["image_id"]: k for k, r in enumerate(split_rows[s])} for s in SPLITS}
    res["embed_seconds"] = round(time.time() - t0, 1); outdir.mkdir(parents=True, exist_ok=True)
    def xy(task, s):
        cl = P[task]["classes"]; rs = [r for r in task_rows(rows, task)[s] if r[task] in cl]
        return E[s][[idx[s][r["image_id"]] for r in rs]], np.array([cl.index(r[task]) for r in rs], dtype=int)
    if not a.joint:
        from sklearn.linear_model import LogisticRegression
        for t in go:
            cl = P[t]["classes"]; Xtr, ytr = xy(t, "train"); Xva, yva = xy(t, "val"); Xte, yte = xy(t, "test")
            grid = []
            for C in [float(c) for c in np.logspace(-2, 3, 11)]:
                clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(Xtr, ytr)
                grid.append((scores(yva, clf.predict(Xva), cl)["macro_f1"], -C, C))
            C = max(grid)[2]; clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=a.seed).fit(Xtr, ytr)
            p = clf.predict(Xte)
            np.savez(outdir / f"mt_{t}_head.npz", coef=clf.coef_, intercept=clf.intercept_, classes=np.array(cl), C=C, model=a.model)
            P[t].update(C=C, val=scores(yva, clf.predict(Xva), cl), test=scores(yte, p, cl), test_macro_f1_95ci=boot(yte, p, cl, seed=a.seed))
            print(f"{t}: C={C} test macro-F1 {P[t]['test']['macro_f1']} (n={len(yte)})")
    else:
        import torch, torch.nn as nn
        torch.manual_seed(a.seed); dim = E["train"].shape[1]
        heads = nn.ModuleDict({t: nn.Linear(256, len(P[t]["classes"])) for t in go}); adapter = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.2))
        def targets(s):   # -1 = masked (no final label / unknown / class below min-class)
            Y = np.full((len(split_rows[s]), len(go)), -1, dtype=np.int64)
            for j, t in enumerate(go):
                cl = P[t]["classes"]; ok = {r["image_id"] for r in task_rows(rows, t)[s]}
                for k, r in enumerate(split_rows[s]):
                    if r["image_id"] in ok and r[t] in cl: Y[k, j] = cl.index(r[t])
            return torch.tensor(Y)
        Y = {s: targets(s) for s in SPLITS}; X = {s: torch.tensor(E[s]) for s in SPLITS}
        ytr = Y["train"]
        cw = {t: torch.tensor([len(ytr) / max(1, int((ytr[:, j] == c).sum())) for c in range(len(P[t]["classes"]))], dtype=torch.float32) for j, t in enumerate(go)}
        opt = torch.optim.AdamW(list(adapter.parameters()) + list(heads.parameters()), lr=1e-3, weight_decay=0.01)
        def loss_on(Xs, Ys):
            h = adapter(Xs); L = 0.0
            for j, t in enumerate(go):
                m = Ys[:, j] >= 0
                if m.any(): L = L + nn.functional.cross_entropy(heads[t](h[m]), Ys[m, j], weight=cw[t])
            return L
        def predict(s):
            adapter.eval(); heads.eval()
            with torch.no_grad(): h = adapter(X[s]); return {t: heads[t](h).argmax(1).numpy() for t in go}
        def val_f1():
            pr = predict("val"); v = []
            for j, t in enumerate(go):
                m = (Y["val"][:, j] >= 0).numpy(); v.append(scores(Y["val"][m, j].numpy(), pr[t][m], P[t]["classes"])["macro_f1"])
            return float(np.mean(v))
        best, state, bad = -1, None, 0
        g = torch.Generator(); g.manual_seed(a.seed)
        for ep in range(a.epochs):
            adapter.train(); heads.train()
            perm = torch.randperm(len(X["train"]), generator=g)
            for k in range(0, len(perm), 64):
                b = perm[k:k + 64]; opt.zero_grad(); L = loss_on(X["train"][b], Y["train"][b])
                if torch.is_tensor(L): L.backward(); opt.step()
            f = val_f1()
            if f > best: best, bad = f, 0; state = ({k: v.clone() for k, v in adapter.state_dict().items()}, {k: v.clone() for k, v in heads.state_dict().items()})
            else:
                bad += 1
                if bad >= 20: break
        adapter.load_state_dict(state[0]); heads.load_state_dict(state[1])
        torch.save({"adapter": adapter.state_dict(), "heads": heads.state_dict(), "tasks": go, "classes": {t: P[t]["classes"] for t in go}, "model": a.model}, outdir / "mt_joint.pt")
        pr = predict("test"); pv = predict("val")
        for j, t in enumerate(go):
            cl = P[t]["classes"]; mt = (Y["test"][:, j] >= 0).numpy(); mv = (Y["val"][:, j] >= 0).numpy()
            yte = Y["test"][mt, j].numpy()
            P[t].update(val=scores(Y["val"][mv, j].numpy(), pv[t][mv], cl), test=scores(yte, pr[t][mt], cl), test_macro_f1_95ci=boot(yte, pr[t][mt], cl, seed=a.seed))
            print(f"{t}: test macro-F1 {P[t]['test']['macro_f1']} (n={len(yte)})")
        res["joint"] = {"best_val_mean_macro_f1": round(best, 4), "epochs_run": ep + 1}
    res["trained"] = True
    json.dump(res, open(outdir / "metrics_v11.json", "w"), indent=1)
    print("->", outdir / "metrics_v11.json")

if __name__ == "__main__": main()
