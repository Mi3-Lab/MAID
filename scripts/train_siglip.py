#!/usr/bin/env python3
"""SigLIP 2 baselines for WheelView: 4-way device classification (manual / powered / other_aid / lookalike).

Model: google/siglip2-so400m-patch14-224 (Apache-2.0) -- the same checkpoint scripts/auto_annotate.py uses.
Rows produced (all on the same face-BLURRED images and the same splits file):
  * zero-shot  : untrained; the project's own prompt ensemble (auto_annotate.P['device_type']), mean of normalised text
                 embeddings per option, softmax at auto_annotate.TEMP, calibration auto_annotate.CALIB (0 = off by default).
  * probe      : frozen image embeddings (pooler output, L2-normalised) -> logistic regression (class_weight='balanced'),
                 C chosen on val macro-F1. Deterministic given the split; uncertainty = bootstrap CI on test.
  * finetune   : last --ft_blocks encoder blocks + post-layernorm + attention-pool head + new linear head trained,
                 class-weighted CE, AdamW, warmup+cosine, AMP, early stop on val macro-F1, one run per seed.
Also reports how often the final human label equals the stored zero-shot suggestion
(annotations/auto/siglip2-so400m-224_zs-v1.jsonl) -- labels were confirmed from those suggestions, so zero-shot (and,
to a lesser degree, every SigLIP row) is biased upward.

Local only: embeddings, weights and per-image predictions go to var/runs/<run>/ (gitignored); only metrics go to results/.

  python scripts/train_siglip.py --splits data/splits_v2.csv --extra_test data/splits_v1.csv --seeds 0,1,2"""
import argparse, csv, glob, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); import auto_annotate as AA   # prompts / temperature / feats: single source of truth
sys.path.insert(0, str(ROOT / "scripts")); from train_baseline import CLASSES, metrics, bootstrap, set_seed, load_split, DS, plot_cm
import torch, torch.nn as nn
from torchvision import transforms as T
from PIL import Image

def agg(vals):
    return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0}

def summarize(runs):
    """runs: list of {'test': metrics, ...} -> spread dict."""
    out = {"test_accuracy": agg([r["test"]["accuracy"] for r in runs]), "test_macro_f1": agg([r["test"]["macro_f1"] for r in runs])}
    for c in CLASSES: out[f"test_f1_{c}"] = agg([r["test"]["per_class"][c]["f1"] for r in runs])
    if all("extra_test" in r for r in runs):
        out["extra_test_accuracy"] = agg([r["extra_test"]["accuracy"] for r in runs]); out["extra_test_macro_f1"] = agg([r["extra_test"]["macro_f1"] for r in runs])
    return out

def human_events():
    ev = []
    for f in sorted(glob.glob(str(ROOT / "annotations" / "events" / "*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if line:
                try: ev.append(json.loads(line))
                except Exception: pass
    return ev

def agreement(rows, sugg):
    """final label vs stored zero-shot suggestion; and how often the deciding device_type event accepted the suggestion as-is."""
    ev = human_events(); last = {}
    for e in ev:
        if e.get("field") != "device_type" or e.get("kind", "label") != "label": continue
        k = (e["image_id"], e["annotator"])
        if k not in last or (e["ts"], e["uuid"]) > (last[k]["ts"], last[k]["uuid"]): last[k] = e
    out = {}
    for name, rs in (("all_clean", rows), *((s, [r for r in rows if r["split"] == s]) for s in ("train", "val", "test"))):
        have = [r for r in rs if r["image_id"] in sugg]
        eq = sum(sugg[r["image_id"]] == r["label"] for r in have)
        acc = 0; n_ev = 0
        for r in rs:
            es = [e for (i, a), e in last.items() if i == r["image_id"] and e.get("value") == r["label"]]
            if es:
                n_ev += 1; acc += any(e.get("source") == "auto_review" and e.get("suggested") == e.get("value") for e in es)
        out[name] = {"n": len(rs), "n_with_suggestion": len(have), "label_equals_suggestion": eq,
                     "rate": round(eq / max(len(have), 1), 4),
                     "accepted_suggestion_unchanged": acc, "accepted_rate": round(acc / max(n_ev, 1), 4)}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=ROOT / "data" / "splits_v2.csv")
    ap.add_argument("--extra_test", default=None)
    ap.add_argument("--images", default=ROOT / "images_blurred")
    ap.add_argument("--model", default=AA.CLS_MODEL)
    ap.add_argument("--run", default="siglip2_v2"); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--emb_bs", type=int, default=16)
    ap.add_argument("--no_ft", action="store_true")
    ap.add_argument("--ft_blocks", type=int, default=4); ap.add_argument("--ft_epochs", type=int, default=15)
    ap.add_argument("--ft_patience", type=int, default=5); ap.add_argument("--ft_bs", type=int, default=16)
    ap.add_argument("--ft_lr", type=float, default=2e-5); ap.add_argument("--ft_head_lr", type=float, default=1e-3)
    ap.add_argument("--ft_wd", type=float, default=0.05); ap.add_argument("--pause", type=int, default=0, help="seconds to idle between fine-tune seeds (thermal)")
    ap.add_argument("--results", default=ROOT / "results")
    a = ap.parse_args()
    from transformers import AutoModel, AutoProcessor
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = dev.type == "cuda"
    outdir = ROOT / "var" / "runs" / a.run; outdir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.splits, encoding="utf-8"))); imgdir = Path(a.images)
    sp = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    extra_info = None
    if a.extra_test:
        seen = {r["image_id"] for r in rows if r["split"] in ("train", "val")}
        old = [r for r in csv.DictReader(open(a.extra_test, encoding="utf-8")) if r["split"] == "test"]
        sp["extra"] = [r for r in old if r["image_id"] not in seen and r["label"] in CLASSES]
        extra_info = {"file": Path(a.extra_test).name, "n_old_test": len(old), "n_evaluated": len(sp["extra"]), "n_dropped_now_in_train_or_val": len(old) - len(sp["extra"])}
    y = {s: np.array([CLASSES.index(r["label"]) for r in v]) for s, v in sp.items()}
    n_per = {s: {c: int((y[s] == i).sum()) for i, c in enumerate(CLASSES)} for s in ("train", "val", "test")}
    print("n per split/class:", n_per, "extra:", extra_info, "device:", dev, flush=True)
    t_all = time.time()
    m = AutoModel.from_pretrained(a.model).eval().to(dev); proc = AutoProcessor.from_pretrained(a.model)

    # ---------- embeddings (processor preprocessing, exactly as auto_annotate) ----------
    E = {}
    t0 = time.time()
    for s, rs in sp.items():
        out = []
        for i in range(0, len(rs), a.emb_bs):
            ims = [Image.open(imgdir / f"{r['image_id']}.jpg").convert("RGB") for r in rs[i:i + a.emb_bs]]
            x = proc(images=ims, return_tensors="pt").to(dev)
            with torch.inference_mode(), torch.autocast(dev.type, enabled=amp):
                e = AA.feats(m.get_image_features(**x))
            out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
        E[s] = np.concatenate(out); print(f"embedded {s}: {E[s].shape}", flush=True)
    emb_s = time.time() - t0
    np.savez(outdir / "emb.npz", **{s: E[s] for s in E}, **{f"ids_{s}": np.array([r["image_id"] for r in sp[s]]) for s in sp})

    # ---------- zero-shot (untrained) ----------
    opts = AA.P["device_type"]; names = list(opts)
    with torch.inference_mode():
        Tm = []
        for o in names:
            x = proc(text=[t.lower() for t in opts[o]], padding="max_length", max_length=64, return_tensors="pt", truncation=True).to(dev)
            t = torch.nn.functional.normalize(AA.feats(m.get_text_features(**x)).float(), dim=-1)
            Tm.append(torch.nn.functional.normalize(t.mean(0), dim=-1).cpu().numpy())
    Tm = np.stack(Tm); order = [names.index(c) for c in CLASSES]
    def zs(Es):
        z = Es @ Tm.T
        if AA.CALIB and len(Es) >= 200: z = z - AA.CALIB * z.mean(0, keepdims=True)
        return AA.softmax(z / AA.TEMP)[:, order].argmax(1)
    zres = {"seed": None, "test": metrics(y["test"], zs(E["test"])), "test_bootstrap": bootstrap(y["test"], zs(E["test"]))}
    if "extra" in E: zres["extra_test"] = metrics(y["extra"], zs(E["extra"]))
    # stored pipeline suggestions (made at labeling time, from the files the labelers saw)
    sugg = {}
    sf = ROOT / "annotations" / "auto" / f"{AA.MODEL_TAG}.jsonl"
    for line in open(sf, encoding="utf-8"):
        d = json.loads(line); v = d.get("suggestions", {}).get("device_type", {}).get("value")
        if v: sugg[d["image_id"]] = v
    st = [(CLASSES.index(r["label"]), CLASSES.index(sugg[r["image_id"]])) for r in sp["test"] if sugg.get(r["image_id"]) in CLASSES]
    stored = {"n": len(st), "test": metrics([u for u, _ in st], [v for _, v in st]), "test_bootstrap": bootstrap([u for u, _ in st], [v for _, v in st])}
    agree = agreement(rows, sugg)
    print("zero-shot test:", round(zres["test"]["accuracy"], 3), round(zres["test"]["macro_f1"], 3), "| stored:", round(stored["test"]["accuracy"], 3), "| agreement:", agree["all_clean"], flush=True)

    # ---------- linear probe ----------
    from sklearn.linear_model import LogisticRegression
    Cs = [float(c) for c in np.logspace(-2, 3, 11)]; grid = []
    for C in Cs:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(E["train"], y["train"])
        grid.append({"C": C, "val_macro_f1": round(metrics(y["val"], clf.predict(E["val"]))["macro_f1"], 4)})
    bestC = max(grid, key=lambda g: (g["val_macro_f1"], -g["C"]))["C"]
    seeds = [int(s) for s in a.seeds.split(",")]; probe_runs = []
    for s in seeds:   # lbfgs is deterministic; seeds kept for a like-for-like table (std is 0 by construction)
        clf = LogisticRegression(C=bestC, class_weight="balanced", max_iter=5000, random_state=s).fit(E["train"], y["train"])
        p = clf.predict(E["test"]); r = {"seed": s, "C": bestC, "test": metrics(y["test"], p), "test_bootstrap": bootstrap(y["test"], p, seed=s)}
        if "extra" in E: r["extra_test"] = metrics(y["extra"], clf.predict(E["extra"]))
        probe_runs.append(r)
        json.dump([{"image_id": rr["image_id"], "label": rr["label"], "pred": CLASSES[k]} for rr, k in zip(sp["test"], p)], open(outdir / f"probe_seed{s}_test_preds.json", "w"), indent=1)
    print("probe C", bestC, "test", round(probe_runs[0]["test"]["accuracy"], 3), round(probe_runs[0]["test"]["macro_f1"], 3), flush=True)

    # ---------- light fine-tune ----------
    ft_runs, ft_cfg = [], None
    if not a.no_ft:
        m.text_model.to("cpu") if hasattr(m, "text_model") else None
        torch.cuda.empty_cache() if amp else None
        vm = m.vision_model
        layers = vm.encoder.layers; k = min(a.ft_blocks, len(layers))
        train_mods = list(layers[-k:]) + [vm.post_layernorm] + ([vm.head] if getattr(vm, "head", None) is not None else [])
        init = {id(mm): {n: t.detach().clone() for n, t in mm.state_dict().items()} for mm in train_mods}
        dim = E["train"].shape[1]
        ft_cfg = {"trainable": f"last {k}/{len(layers)} encoder blocks + post_layernorm + attention-pool head + linear head",
                  "epochs": a.ft_epochs, "patience": a.ft_patience, "bs": a.ft_bs, "lr_backbone": a.ft_lr, "lr_head": a.ft_head_lr,
                  "wd": a.ft_wd, "optimizer": "AdamW", "schedule": "1-epoch warmup + cosine", "loss": "class-weighted CE (inverse train frequency)",
                  "input": "224px; train RandomResizedCrop(0.6-1.0)+HFlip+ColorJitter(0.2,0.2,0.1); eval Resize((224,224)); mean/std 0.5", "amp": amp}
        norm = T.Normalize([0.5] * 3, [0.5] * 3)
        tf_tr = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.8, 1.25)), T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.1), T.ToTensor(), norm])
        tf_ev = T.Compose([T.Resize((224, 224)), T.ToTensor(), norm])
        data = {s: load_split(sp[s], imgdir) for s in sp}
        cnt = np.bincount(data["train"][1], minlength=len(CLASSES)).astype(float)
        cw = torch.tensor(cnt.sum() / (len(CLASSES) * np.maximum(cnt, 1)), dtype=torch.float32, device=dev)
        for si, s in enumerate(seeds):
            if si and a.pause: print(f"pause {a.pause}s", flush=True); time.sleep(a.pause)
            set_seed(s)
            for mm in train_mods: mm.load_state_dict(init[id(mm)])
            for prm in vm.parameters(): prm.requires_grad_(False)
            for mm in train_mods:
                for prm in mm.parameters(): prm.requires_grad_(True)
            head = nn.Linear(dim, len(CLASSES)).to(dev)
            fwd = lambda x: head(AA.feats(vm(pixel_values=x)).float())
            g = torch.Generator(); g.manual_seed(s)
            dl = lambda k_, tf, sh: torch.utils.data.DataLoader(DS(data[k_][0], data[k_][1], tf), batch_size=a.ft_bs, shuffle=sh, generator=g if sh else None, num_workers=0)
            tr = dl("train", tf_tr, True)
            bb = [p_ for mm in train_mods for p_ in mm.parameters()]
            opt = torch.optim.AdamW([{"params": bb, "lr": a.ft_lr}, {"params": head.parameters(), "lr": a.ft_head_lr}], weight_decay=a.ft_wd)
            steps = a.ft_epochs * len(tr); warm = len(tr)
            sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps)))
            scaler = torch.amp.GradScaler("cuda", enabled=amp); crit = nn.CrossEntropyLoss(weight=cw)
            def predict(k_):
                vm.eval(); head.eval(); ps = []
                with torch.no_grad(), torch.autocast(dev.type, enabled=amp):
                    for x, _ in dl(k_, tf_ev, False): ps += fwd(x.to(dev)).argmax(1).cpu().tolist()
                return np.array(ps)
            best, bad, hist, t0 = -1, 0, [], time.time(); ck = outdir / f"ft_seed{s}_best.pt"
            for ep in range(1, a.ft_epochs + 1):
                vm.train(); head.train(); tl = n = 0
                for x, yy in tr:
                    x, yy = x.to(dev), yy.to(dev); opt.zero_grad(set_to_none=True)
                    with torch.autocast(dev.type, enabled=amp): loss = crit(fwd(x), yy)
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step(); tl += loss.item() * len(yy); n += len(yy)
                vmf = metrics(y["val"], predict("val"))["macro_f1"]
                hist.append({"epoch": ep, "train_loss": round(tl / n, 4), "val_macro_f1": round(vmf, 4)})
                print(f"[ft seed {s}] ep {ep:02d} loss {tl/n:.4f} val_mF1 {vmf:.3f} ({time.time()-t0:.0f}s)", flush=True)
                if vmf > best + 1e-6:
                    best, bad, best_ep = vmf, 0, ep
                    torch.save({"head": head.state_dict(), **{f"m{j}": mm.state_dict() for j, mm in enumerate(train_mods)}}, ck)
                else:
                    bad += 1
                    if bad >= a.ft_patience: print(f"[ft seed {s}] early stop ep {ep}", flush=True); break
            sd = torch.load(ck, map_location=dev); head.load_state_dict(sd["head"])
            for j, mm in enumerate(train_mods): mm.load_state_dict(sd[f"m{j}"])
            p = predict("test"); r = {"seed": s, "best_epoch": best_ep, "epochs_run": len(hist), "best_val_macro_f1": round(best, 4),
                                     "test": metrics(y["test"], p), "test_bootstrap": bootstrap(y["test"], p, seed=s), "train_seconds": round(time.time() - t0, 1), "history": hist}
            if "extra" in sp: r["extra_test"] = metrics(y["extra"], predict("extra"))
            json.dump([{"image_id": rr["image_id"], "label": rr["label"], "pred": CLASSES[k_]} for rr, k_ in zip(sp["test"], p)], open(outdir / f"ft_seed{s}_test_preds.json", "w"), indent=1)
            ft_runs.append(r); print(f"[ft seed {s}] TEST acc {r['test']['accuracy']:.3f} mF1 {r['test']['macro_f1']:.3f}", flush=True)
            if amp: print("peak GPU mem GB", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)

    res = Path(a.results); res.mkdir(parents=True, exist_ok=True)
    summary = {"model": a.model, "model_tag": AA.MODEL_TAG, "classes": CLASSES, "splits_file": "data/" + Path(a.splits).name,
               "images": "images_blurred/ (face-blurred copies)", "n_per_split": n_per, "extra_test": extra_info,
               "zero_shot": {"method": f"auto_annotate prompt ensemble ({AA.PROMPT_VERSION}), softmax T={AA.TEMP} on cosine, CALIB={AA.CALIB}; recomputed on blurred images",
                             "n_prompts": {c: len(opts[c]) for c in CLASSES}, "runs": [zres], "spread": summarize([zres])},
               "zero_shot_stored_suggestions": {"note": "the suggestions shown to labelers (annotations/auto/" + AA.MODEL_TAG + ".jsonl), scored against final labels on the same test images", **stored},
               "label_suggestion_agreement": agree,
               "probe": {"method": "LogisticRegression(class_weight='balanced', lbfgs) on L2-normalised frozen image embeddings; C by val macro-F1",
                         "C_grid": grid, "C": bestC, "runs": probe_runs, "spread": summarize(probe_runs),
                         "note": "deterministic given the split: seed std is 0 by construction; use the bootstrap CI"},
               "finetune": ({"config": ft_cfg, "runs": ft_runs, "spread": summarize(ft_runs)} if ft_runs else None),
               "embed_seconds": round(emb_s, 1), "total_seconds": round(time.time() - t_all, 1),
               "hardware": {"device": str(dev), "gpu": torch.cuda.get_device_name(0) if amp else None, "torch": torch.__version__}}
    json.dump(summary, open(res / f"baseline_{a.run}.json", "w"), indent=2)
    best_row = ft_runs[0] if ft_runs else probe_runs[0]
    plot_cm(best_row["test"]["confusion_matrix"], res / f"baseline_{a.run}_confusion.png",
            f"SigLIP2 {'fine-tune' if ft_runs else 'linear probe'}, test (n={len(y['test'])}), seed {best_row['seed']}")
    print(json.dumps({"zero_shot": summary["zero_shot"]["spread"], "probe": summary["probe"]["spread"],
                      "finetune": summary["finetune"]["spread"] if ft_runs else None, "agreement": agree["test"]}, indent=1))

if __name__ == "__main__":
    main()
