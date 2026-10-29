#!/usr/bin/env python3
"""Untrained-vs-trained comparison on the WheelView v3 test split (data/manifest_v2.csv = data/splits_v3.csv).

Reads results/baseline_{siglip2_v3,resnet50_v3,yolo_v3}.json, results/val_scores_v3.json (val-only model selection,
from scripts/score_val.py), the v2 results for the v2-vs-v3 note, and the local GPU log var/runs/gpu_log_v3.csv
(+ phases_v3.json, thermal_pauses_v3.json). Writes results/baseline_v3.{md,json}, results/baseline_v3_comparison.png,
results/baseline_v3_confusion.png and results/gpu_log_v3_summary.json.

3-class macro-F1 ("no ambiguous"): test images labelled ambiguous are dropped; the 4-way predictions are kept, so an
image predicted ambiguous counts as a miss; F1 is averaged over manual, powered, other_aid.

  python scripts/compare_v3.py"""
import csv, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; RES = ROOT / "results"; RUNS = ROOT / "var" / "runs"
sys.path.insert(0, str(ROOT / "scripts"))
from train_baseline import CLASSES, read_split_rows, wvcore
import compare_v2 as C2
K3 = CLASSES[:3]

def L(n):
    p = RES / f"baseline_{n}.json"; return wvcore.canon_keys(json.load(open(p))) if p.exists() else None

def f1_3(cm):
    cm = np.asarray(cm); out = {}
    for k, c in enumerate(K3):
        tp = cm[k, k]; pred = cm[:3, k].sum(); act = cm[k, :].sum()
        p = tp / pred if pred else 0.0; r = tp / act if act else 0.0
        out[c] = (p, r, 2 * p * r / (p + r) if p + r else 0.0)
    return out

def mf3(cm): return float(np.mean([v[2] for v in f1_3(cm).values()]))

def boot3(cm, n=2000, seed=0):
    cm = np.asarray(cm)[:3]; pairs = np.array([(i, j) for i in range(cm.shape[0]) for j in range(cm.shape[1]) for _ in range(cm[i, j])])
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        s = pairs[rng.integers(0, len(pairs), len(pairs))]; m = np.zeros(cm.shape, int)
        np.add.at(m, (s[:, 0], s[:, 1]), 1); v.append(mf3(m))
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

def agg(v): return {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else 0.0}
def fmt(d, nd=3): return f"{d['mean']:.{nd}f}" + (f" ± {d['std']:.{nd}f}" if d["std"] else "")
def cis(v): return f" [{v[0]:.2f}-{v[1]:.2f}]" if v else ""

def gpu_summary():
    C2.ROOT = ROOT  # reuse the v2 parser on the v3 files
    p, ph = RUNS / "gpu_log_v3.csv", RUNS / "phases_v3.json"
    if not p.exists(): return None
    import shutil, tempfile
    # compare_v2.gpu_summary reads fixed names; point it at the v3 files through a temporary root
    tmp = Path(tempfile.mkdtemp()); (tmp / "var" / "runs").mkdir(parents=True)
    shutil.copy2(p, tmp / "var" / "runs" / "gpu_log.csv")
    if ph.exists():
        json.dump([x for x in json.load(open(ph)) if "start" in x], open(tmp / "var" / "runs" / "phases.json", "w"))
    C2.ROOT = tmp; g = C2.gpu_summary(); C2.ROOT = ROOT
    tp = RUNS / "thermal_pauses_v3.json"
    g["thermal_pauses_over_85c"] = json.load(open(tp)) if tp.exists() else []
    pw = ROOT / "var" / "power_pre_v3.json"
    g["power_plan"] = {"during_run_ac": {"SYSCOOLPOL": "1 (active cooling; already set)", "PERFBOOSTMODE": "0 (CPU turbo off)", "PROCTHROTTLEMAX": 99,
                                         "STANDBYIDLE": "0 (never sleep)", "runner": "SetThreadExecutionState(ES_SYSTEM_REQUIRED)"},
                       "restored_after_from": "var/power_pre_v3.json" if pw.exists() else None}
    return g

def main():
    sg, rn, yo = L("siglip2_v3"), L("resnet50_v3"), L("yolo_v3")
    vs = json.load(open(RES / "val_scores_v3.json")) if (RES / "val_scores_v3.json").exists() else {"models": {}}
    rows = read_split_rows(ROOT / "data" / "manifest_v2.csv"); test = [r for r in rows if r["split"] == "test"]
    ntest = {c: sum(1 for r in test if r["label"] == c) for c in CLASSES}; N = len(test)
    M = []   # dict(name, kind, key, runs, note, val)
    ag = sg["label_suggestion_agreement"]
    M.append(dict(name="SigLIP2 zero-shot", kind="untrained", key="siglip_zs", runs=sg["zero_shot"]["runs"],
                  note=f"BIASED UP: final labels equal the SigLIP2 suggestion for {ag['test']['rate']:.0%} of test images ({ag['all_clean']['rate']:.0%} of all clean)"))
    if yo and yo.get("world"):
        w = yo["world"]; M.append(dict(name="YOLO-World v2-s zero-shot", kind="untrained", key="yolo_world", runs=w["runs"], note=f"no box (counted wrong) on {w['no_box_rate']:.0%} of test images at conf {w['conf']}"))
    M.append(dict(name="SigLIP2 linear probe", kind="trained (head only)", key="siglip-probe", runs=sg["probe"]["runs"], note=f"C={sg['probe']['C']:g} chosen on val; deterministic (seeds identical)"))
    if sg.get("finetune"): M.append(dict(name="SigLIP2 fine-tune (last 4 blocks)", kind="trained", key="siglip-ft", runs=sg["finetune"]["runs"], note=sg["finetune"]["config"]["trainable"]))
    if rn: M.append(dict(name="ResNet-50 fine-tune", kind="trained", key="resnet", runs=rn["runs"], note="same recipe as v1/v2"))
    if yo and yo.get("cls"): M.append(dict(name="YOLO11s-cls fine-tune", kind="trained", key="yolo-cls", runs=yo["cls"]["runs"], note="Ultralytics defaults, no class weights"))
    if yo and yo.get("det") and not yo["det"].get("skipped"):
        M.append(dict(name="YOLO11s detector (PRELIMINARY)", kind="trained", key="yolo_det", runs=yo["det"]["runs"],
                      note="image class = top box; box mAP50 " + ", ".join(str(r["box_test"]["mAP50"]) for r in yo["det"]["runs"])))
    for m in M:
        R = m["runs"]; t = [r["test"] for r in R]
        m["acc"] = agg([x["accuracy"] for x in t]); m["mf1"] = agg([x["macro_f1"] for x in t])
        m["mf1_3"] = agg([mf3(x["confusion_matrix"]) for x in t]); m["mf1_3_ci"] = boot3(t[0]["confusion_matrix"])
        m["acc_ci"] = R[0]["test_bootstrap"]["accuracy_95ci"]; m["mf1_ci"] = R[0]["test_bootstrap"]["macro_f1_95ci"]
        m["per_class"] = {c: {k: agg([x["per_class"][c][k] for x in t]) for k in ("precision", "recall", "f1")} for c in CLASSES}
        m["cm_sum"] = np.sum([np.asarray(x["confusion_matrix"]) for x in t], axis=0).tolist(); m["n_seeds"] = len(R)
        m["val"] = vs["models"].get(m["key"])
    trained = [m for m in M if m["val"]]
    best = max(trained, key=lambda m: (m["val"]["mean_val_macro_f1"], m["val"]["mean_val_accuracy"])) if trained else None
    v2 = {"SigLIP2 zero-shot": (0.661, 0.440), "YOLO-World v2-s zero-shot": (0.431, 0.345), "SigLIP2 linear probe": (0.785, 0.557),
          "SigLIP2 fine-tune (last 4 blocks)": (0.877, 0.633), "ResNet-50 fine-tune": (0.764, 0.544), "YOLO11s-cls fine-tune": (0.769, 0.578),
          "YOLO11s detector (PRELIMINARY)": (0.677, 0.431)}
    a2 = {r["image_id"]: r["split"] for r in csv.DictReader(open(ROOT / "data" / "splits_v2.csv", encoding="utf-8"))}
    ov = {"v3_test_in_v2_test": sum(1 for r in test if a2.get(r["image_id"]) == "test"),
          "v3_test_in_v2_train_val": sum(1 for r in test if a2.get(r["image_id"]) in ("train", "val")),
          "v3_test_not_in_v2": sum(1 for r in test if r["image_id"] not in a2)}
    lab = {}
    for r in rows: lab[r.get("label_how", "")] = lab.get(r.get("label_how", ""), 0) + 1
    md = ["# WheelView v3 baselines: untrained vs trained", "",
          f"All rows: v3 test split (`data/manifest_v2.csv`, same as `data/splits_v3.csv`; n = {N}: " + ", ".join(f"{c} {ntest[c]}" for c in CLASSES) +
          f"), face-blurred images, 4-way device class. 458 trainable images: train {sum(1 for r in rows if r['split']=='train')} / val {sum(1 for r in rows if r['split']=='val')} / test {N}. "
          "Mean ± sd over seeds 0,1,2 where training is stochastic; [95% bootstrap CI] is for seed 0. Models and settings were picked on val only "
          "(early stopping / best checkpoint on val, probe C on val, best-model choice on mean val macro-F1 from `results/val_scores_v3.json`); test was scored once per trained model. "
          "Generated by `scripts/compare_v3.py`.", "",
          "| model | type | accuracy | macro-F1 (4 cls) | macro-F1 (3 cls, no ambiguous) | val macro-F1 | note |", "|---|---|---|---|---|---|---|"]
    for m in M:
        md.append(f"| {m['name']}{' **(best on val)**' if m is best else ''} | {m['kind']} | {fmt(m['acc'])}{cis(m['acc_ci'])} | {fmt(m['mf1'])}{cis(m['mf1_ci'])} | "
                  f"{fmt(m['mf1_3'])}{cis(m['mf1_3_ci'])} | {m['val']['mean_val_macro_f1']:.3f}" if m["val"] else
                  f"| {m['name']} | {m['kind']} | {fmt(m['acc'])}{cis(m['acc_ci'])} | {fmt(m['mf1'])}{cis(m['mf1_ci'])} | {fmt(m['mf1_3'])}{cis(m['mf1_3_ci'])} | n/a")
        md[-1] += f" | {m['note']} |"
    maj = max(ntest.values()) / N
    md += ["", f"Majority-class (always `manual`) accuracy: {maj:.3f}; 4-class macro-F1 {(2*maj/(1+maj))/4:.3f}; 3-class {(2*maj/(1+maj))/3:.3f}.", "",
           "## Per-class precision / recall / F1 (test, mean ± sd over seeds)", "",
           "| model | " + " | ".join(f"{c} (n={ntest[c]}) P / R / F1" for c in CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for m in M:
        md.append(f"| {m['name']} | " + " | ".join(" / ".join(fmt(m["per_class"][c][k], 2) for k in ("precision", "recall", "f1")) for c in CLASSES) + " |")
    md += ["", f"With only {ntest['ambiguous']} ambiguous test images, that class's numbers move in steps of 0.5; the 3-class column leaves it out.", "",
           "## Label / suggestion agreement (zero-shot bias)", "",
           "Labelers reviewed SigLIP2 zero-shot suggestions while labeling, so the zero-shot row is scored against labels it influenced, "
           "and every SigLIP-based row (probe, fine-tune) shares that backbone, so it is likely biased upward too, though less.", "",
           "| subset | n | final label = suggestion | deciding label event accepted the suggestion unchanged |", "|---|---|---|---|"]
    for k in ("all_clean", "train", "val", "test"):
        a = ag[k]; md.append(f"| {k} | {a['n']} | {a['label_equals_suggestion']}/{a['n_with_suggestion']} ({a['rate']:.1%}) | {a['accepted_suggestion_unchanged']} ({a['accepted_rate']:.1%}) |")
    md += ["", "## v2 vs v3 (not directly comparable)", "",
           f"v3 is a fresh grouped split of a different clean set (458 vs 433 trainable images), so the test sets differ: of the {N} v3 test images, "
           f"{ov['v3_test_in_v2_test']} were in v2 test, {ov['v3_test_in_v2_train_val']} were in v2 train/val and {ov['v3_test_not_in_v2']} are new. "
           "Differences below mix model change, data change and test-set change; read them as indicative only.", "",
           "| model | v2 acc (n=65) | v3 acc (n=69) | v2 macro-F1 | v3 macro-F1 |", "|---|---|---|---|---|"]
    for m in M:
        if m["name"] in v2: md.append(f"| {m['name']} | {v2[m['name']][0]:.3f} | {m['acc']['mean']:.3f} | {v2[m['name']][1]:.3f} | {m['mf1']['mean']:.3f} |")
    g = gpu_summary()
    if g:
        json.dump(g, open(RES / "gpu_log_v3_summary.json", "w"), indent=1)
        md += ["", "## GPU during training", "", f"nvidia-smi every 5 s ({g['samples']} samples, {g['busy_samples']} busy) from {g['first_ts']} to {g['last_ts']} (laptop local time, PDT): "
               f"max temp {g['max_temp_c']} °C (busy mean {g['mean_temp_busy_c']} °C), max power {g['max_power_w']} W, busy mean SM clock {g['mean_sm_clock_busy_mhz']:.0f} of {g['max_sm_clock_mhz']:.0f} MHz; "
               f"busy time thermally throttled {g['pct_busy_time_thermal_throttled']}%, power-capped {g['pct_busy_time_power_capped']}%. "
               f"Pauses for > 85 °C: {len(g['thermal_pauses_over_85c'])}. Power plan during the run: CPU turbo off, active cooling, never sleep; restored afterwards. "
               "Details in `results/gpu_log_v3_summary.json`.", ""]
    md += ["## Caveats", "", f"- Small test set (n = {N}; ambiguous {ntest['ambiguous']}, other_aid {ntest['other_aid']}): CIs are wide; rare-class F1 swings with single images.",
           "- SigLIP2 zero-shot is biased upward (labels were confirmed from its suggestions); SigLIP-based trained rows inherit some of that.",
           f"- Label resolution over the clean set: " + ", ".join(f"{k} {v}" for k, v in sorted(lab.items()) if k) + ".",
           "- Seed spread is training randomness on one fixed grouped split, not split-to-split variance.",
           "- YOLO11-cls uses Ultralytics' own recipe, so it is not identical to the ResNet/SigLIP recipe. The detector row is preliminary.",
           "- Inference uses only saved weights + the image (`scripts/predict.py`; `check_leakage.py --runtime` audits resnet, siglip-probe, siglip-ft and yolo-cls).", ""]
    (RES / "baseline_v3.md").write_text("\n".join(md), encoding="utf-8")
    keep = ("name", "kind", "key", "n_seeds", "acc", "acc_ci", "mf1", "mf1_ci", "mf1_3", "mf1_3_ci", "per_class", "cm_sum", "val", "note")
    json.dump({"test_n": N, "test_counts": ntest, "best_on_val": best["name"] if best else None, "models": [{k: m[k] for k in keep} for m in M],
               "v2_reference": v2, "v2_v3_test_overlap": ov, "gpu": g}, open(RES / "baseline_v3.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    mets = [("acc", "accuracy"), ("mf1", "macro-F1"), ("mf1_3", "macro-F1\n(3 cls)")] + [(c, f"F1 {c}") for c in CLASSES]
    fig, ax = plt.subplots(figsize=(13, 6), dpi=150); W = 0.8 / len(M); x = np.arange(len(mets))
    for j, m in enumerate(M):
        g_ = lambda k: m[k] if k in m else m["per_class"][k]["f1"]
        ax.bar(x + (j - (len(M) - 1) / 2) * W, [g_(k)["mean"] for k, _ in mets], W, yerr=[g_(k)["std"] for k, _ in mets], capsize=2,
               label=f"{m['name']} ({m['kind']})" + (" *" if m["key"] == "siglip_zs" else ""), hatch="//" if m["kind"] == "untrained" else None, edgecolor="black", linewidth=0.4)
    ax.axhline(maj, ls="--", c="gray", lw=0.8); ax.text(len(mets) - 0.5, maj + 0.01, "majority-class acc", fontsize=7, color="gray", ha="right")
    ax.set_xticks(x, [l for _, l in mets]); ax.set_ylim(0, 1.05); ax.set_ylabel("score on v3 test")
    ax.set_title(f"WheelView v3 test (n={N}): untrained (hatched) vs trained; error bars = sd over seeds\n* SigLIP2 zero-shot biased upward: labels were confirmed from its suggestions", fontsize=9)
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1)); fig.tight_layout(); fig.savefig(RES / "baseline_v3_comparison.png"); plt.close(fig)
    nc = 4; nr = int(np.ceil(len(M) / nc)); fig, axs = plt.subplots(nr, nc, figsize=(4 * nc, 3.8 * nr), dpi=130)
    for a_, m in zip(axs.flat, M):
        cm = np.asarray(m["cm_sum"]); cols = CLASSES + (["no box"] if cm.shape[1] > 4 else [])
        a_.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for k in range(cm.shape[1]): a_.text(k, i, cm[i, k], ha="center", va="center", fontsize=8)
        a_.set_xticks(range(len(cols)), cols, rotation=40, fontsize=7); a_.set_yticks(range(4), CLASSES, fontsize=7)
        a_.set_title(f"{m['name']}\n(summed over {m['n_seeds']} seed{'s' if m['n_seeds'] > 1 else ''})", fontsize=8); a_.set_xlabel("predicted", fontsize=7); a_.set_ylabel("true", fontsize=7)
    for a_ in list(axs.flat)[len(M):]: a_.axis("off")
    fig.suptitle(f"WheelView v3 test confusion matrices (n={N} per seed; colour = row-normalised)", fontsize=10); fig.tight_layout(); fig.savefig(RES / "baseline_v3_confusion.png"); plt.close(fig)
    print("\n".join(md))

if __name__ == "__main__":
    main()
