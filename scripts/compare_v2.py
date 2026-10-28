#!/usr/bin/env python3
"""Untrained-vs-trained comparison on the WheelView v2 test split -> results/baseline_siglip2_v2.md + _comparison.png.

Reads results/baseline_{resnet50_v2,siglip2_v2,yolo_v2}.json (+ resnet50_v1 for the v1 reference), the v1 ResNet
per-image predictions in var/runs/resnet50_v1/ (local, to score v1 on the same old-test subset), and the GPU log
var/runs/gpu_log.csv (+ phases.json) -> results/gpu_log_v2_summary.json.

  python scripts/compare_v2.py"""
import csv, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
from train_baseline import CLASSES, metrics

def L(n):
    p = RES / f"baseline_{n}.json"; return json.load(open(p)) if p.exists() else None

def ms(d, k, nd=3):
    if not d or k not in d: return "n/a"
    return f"{d[k]['mean']:.{nd}f}" + (f" ± {d[k]['std']:.{nd}f}" if d[k]["std"] else "")

def ci(run, k):
    b = run.get("test_bootstrap") or {}; v = b.get(k)
    return f" [{v[0]:.2f}-{v[1]:.2f}]" if v else ""

BITS = {0x1: "GpuIdle", 0x2: "ApplicationsClocksSetting", 0x4: "SwPowerCap", 0x8: "HwSlowdown", 0x10: "SyncBoost",
        0x20: "SwThermalSlowdown", 0x40: "HwThermalSlowdown", 0x80: "HwPowerBrakeSlowdown", 0x100: "DisplayClockSetting"}
def gpu_summary():
    p = ROOT / "var" / "runs" / "gpu_log.csv"
    if not p.exists(): return None
    rows = []
    for line in open(p, encoding="utf-8", errors="ignore"):
        f = [x.strip() for x in line.split(",")]
        if len(f) < 6 or not f[1].isdigit(): continue
        try: rows.append({"ts": f[0], "temp": int(f[1]), "sm": float(f[2].split()[0]), "smmax": float(f[3].split()[0]),
                          "w": float(f[4].split()[0]) if f[4][0].isdigit() else None, "r": int(f[5], 16)})
        except Exception: continue
    busy = [r for r in rows if not (r["r"] & 0x1) or r["w"] and r["w"] > 15]
    def frac(rs, mask): return round(sum(1 for r in rs if r["r"] & mask) / max(len(rs), 1), 4)
    out = {"samples": len(rows), "interval_s": 5, "busy_samples": len(busy),
           "max_temp_c": max((r["temp"] for r in rows), default=None),
           "mean_temp_busy_c": round(float(np.mean([r["temp"] for r in busy])), 1) if busy else None,
           "max_power_w": max((r["w"] for r in rows if r["w"] is not None), default=None),
           "mean_sm_clock_busy_mhz": round(float(np.mean([r["sm"] for r in busy])), 0) if busy else None,
           "max_sm_clock_mhz": max((r["smmax"] for r in rows), default=None),
           "pct_busy_time_thermal_throttled": round(100 * frac(busy, 0x20 | 0x40), 2),
           "pct_busy_time_power_capped": round(100 * frac(busy, 0x4 | 0x80), 2),
           "pct_busy_time_hw_slowdown": round(100 * frac(busy, 0x8 | 0x40 | 0x80), 2),
           "pct_busy_time_any_throttle": round(100 * frac(busy, 0x4 | 0x8 | 0x20 | 0x40 | 0x80), 2),
           "reasons_seen_busy": {name: sum(1 for r in busy if r["r"] & b) for b, name in BITS.items() if any(r["r"] & b for r in busy)},
           "first_ts": rows[0]["ts"] if rows else None, "last_ts": rows[-1]["ts"] if rows else None}
    ph = ROOT / "var" / "runs" / "phases.json"
    if ph.exists():
        out["phases"] = json.load(open(ph))
        for P in out["phases"]:
            rs = [r for r in busy if P["start"] <= r["ts"][:19] <= P["end"]]
            P.update({"busy_samples": len(rs), "max_temp_c": max((r["temp"] for r in rs), default=None),
                      "pct_thermal": round(100 * frac(rs, 0x20 | 0x40), 2), "pct_power_cap": round(100 * frac(rs, 0x4 | 0x80), 2)})
    return out

def main():
    rn, sg, yo, v1 = L("resnet50_v2"), L("siglip2_v2"), L("yolo_v2"), L("resnet50_v1")
    split = json.load(open(RES / "splits_v2_summary.json"))
    rows = []   # (name, kind, runs, spread, note)
    if sg:
        ag = sg["label_suggestion_agreement"]
        rows.append(("SigLIP2 zero-shot", "untrained", sg["zero_shot"]["runs"], sg["zero_shot"]["spread"],
                     f"BIASED UP: final labels equal the SigLIP2 suggestion for {ag['test']['rate']:.0%} of test images ({ag['all_clean']['rate']:.0%} of all clean); labels were confirmed from these suggestions"))
    if yo and yo.get("world"):
        w = yo["world"]; rows.append(("YOLO-World v2-s zero-shot", "untrained", w["runs"], w["spread"], f"no box (counted wrong) on {w['no_box_rate']:.0%} of test images at conf {w['conf']}"))
    if sg:
        rows.append(("SigLIP2 linear probe", "trained (head only)", sg["probe"]["runs"], sg["probe"]["spread"], f"C={sg['probe']['C']:g}; deterministic, so no seed spread: see bootstrap CI"))
        if sg.get("finetune"): rows.append(("SigLIP2 fine-tune (last blocks)", "trained", sg["finetune"]["runs"], sg["finetune"]["spread"], sg["finetune"]["config"]["trainable"]))
    if rn: rows.append(("ResNet-50 fine-tune", "trained", rn["runs"], rn["seed_spread"], "same recipe as v1"))
    if yo and yo.get("cls"): rows.append(("YOLO11s-cls fine-tune", "trained", yo["cls"]["runs"], yo["cls"]["spread"], "Ultralytics defaults, no class weights"))
    if yo and yo.get("det") and not yo["det"].get("skipped"):
        rows.append(("YOLO11s detector (PRELIMINARY)", "trained", yo["det"]["runs"], yo["det"]["spread"],
                     "image class = top box; mAP50 " + ", ".join(str(r["box_test"]["mAP50"]) for r in yo["det"]["runs"])))
    n_test = sum(split["split_counts"]["test"].values())
    md = ["# WheelView v2 baselines: untrained vs trained", "",
          f"All rows: same v2 test split (`data/splits_v2.csv`, n = {n_test}: " + ", ".join(f"{c} {split['split_counts']['test'][c]}" for c in CLASSES) +
          "), face-blurred images, 4-way device class. Mean ± sd over seeds (0,1,2) where training is stochastic; "
          "[95% bootstrap CI] is for the first seed. Generated by `scripts/compare_v2.py` from "
          "`results/baseline_{siglip2_v2,resnet50_v2,yolo_v2}.json`.", "",
          f"Clean v2 set: {split['n']} images (" + ", ".join(f"{c} {split['class_counts'][c]}" for c in CLASSES) +
          f"); train {split['split_sizes']['train']} / val {split['split_sizes']['val']} / test {split['split_sizes']['test']}; {split['n_groups']} uploader groups.", "",
          "| model | type | accuracy | macro-F1 | F1 manual | F1 powered | F1 other_aid | F1 lookalike | note |", "|---|---|---|---|---|---|---|---|---|"]
    for name, kind, runs, s, note in rows:
        md.append(f"| {name} | {kind} | {ms(s,'test_accuracy')}{ci(runs[0],'accuracy_95ci')} | {ms(s,'test_macro_f1')}{ci(runs[0],'macro_f1_95ci')} | " +
                  " | ".join(ms(s, f"test_f1_{c}") for c in CLASSES) + f" | {note} |")
    maj = max(split["split_counts"]["test"].values()) / n_test
    md += ["", f"Majority-class (always `manual`) accuracy on this test set: {maj:.3f}; macro-F1 {(2*maj/(1+maj))/4:.3f}.", ""]
    if sg:
        ag = sg["label_suggestion_agreement"]; st = sg["zero_shot_stored_suggestions"]
        md += ["## Label / suggestion agreement (zero-shot bias)", "",
               "Labelers reviewed SigLIP2 zero-shot suggestions (`annotations/auto/siglip2-so400m-224_zs-v1.jsonl`) while labeling, so the "
               "zero-shot row is evaluated against labels it influenced. Every SigLIP-based row shares that backbone, so it is biased upward too, but less.", "",
               "| subset | n | final label = suggestion | deciding label event accepted the suggestion unchanged (auto_review) |", "|---|---|---|---|"]
        for k in ("all_clean", "train", "val", "test"):
            a = ag[k]; md.append(f"| {k} | {a['n']} | {a['label_equals_suggestion']}/{a['n_with_suggestion']} ({a['rate']:.1%}) | {a['accepted_suggestion_unchanged']} ({a['accepted_rate']:.1%}) |")
        md += ["", f"The stored suggestions themselves (made at labeling time) score accuracy {st['test']['accuracy']:.3f}, macro-F1 {st['test']['macro_f1']:.3f} on the same {st['n']} test images; "
               "the zero-shot row above is recomputed on the blurred copies with the same prompts.", ""]
    # v1 comparison
    md += ["## Comparison with v1", ""]
    if v1:
        s1 = v1["seed_spread"]; md.append(f"- v1 ResNet-50 on the v1 test split (n = {sum(v1['n_per_split']['test'].values())}): accuracy {ms(s1,'test_accuracy')}, macro-F1 {ms(s1,'test_macro_f1')}.")
    ex = (rn or {}).get("extra_test")
    if ex:
        v2rows = list(csv.DictReader(open(ROOT / "data" / "splits_v2.csv", encoding="utf-8")))
        seen = {r["image_id"] for r in v2rows if r["split"] in ("train", "val")}
        old = [r for r in csv.DictReader(open(ROOT / "data" / "splits_v1.csv", encoding="utf-8")) if r["split"] == "test"]
        keep = {r["image_id"] for r in old if r["image_id"] not in seen}
        v1sub = []
        for s in (v1 or {}).get("seeds", []):
            pp = ROOT / "var" / "runs" / "resnet50_v1" / f"seed{s}_test_preds.json"
            if pp.exists():
                pr = [d for d in json.load(open(pp)) if d["image_id"] in keep]
                v1sub.append(metrics([CLASSES.index(d["label"]) for d in pr], [CLASSES.index(d["pred"]) for d in pr]))
        md += [f"- Old v1 test images not in v2 train/val: {ex['n_evaluated']} of {ex['n_old_test']} ({ex['n_dropped_now_in_train_or_val']} moved into v2 train/val by the fresh split and are dropped). Scores on that subset:", "",
               "| model | accuracy | macro-F1 |", "|---|---|---|"]
        if v1sub:
            A = lambda k: f"{np.mean([m[k] for m in v1sub]):.3f}" + (f" ± {np.std([m[k] for m in v1sub], ddof=1):.3f}" if len(v1sub) > 1 else "")
            md.append(f"| ResNet-50 v1 (trained on v1) | {A('accuracy')} | {A('macro_f1')} |")
        for name, kind, runs, s, note in rows:
            if "extra_test_accuracy" in s: md.append(f"| {name} | {ms(s,'extra_test_accuracy')} | {ms(s,'extra_test_macro_f1')} |")
        md += ["", "The subset is small, so treat differences as indicative only."]
    md.append("")
    # GPU
    g = gpu_summary()
    if g:
        json.dump(g, open(RES / "gpu_log_v2_summary.json", "w"), indent=1)
        md += ["## GPU during training", "", f"nvidia-smi every 5 s ({g['samples']} samples, {g['busy_samples']} with the GPU busy) from {g['first_ts']} to {g['last_ts']} (laptop local time): "
               f"max temp {g['max_temp_c']} °C (busy mean {g['mean_temp_busy_c']} °C), max power {g['max_power_w']} W, busy mean SM clock {g['mean_sm_clock_busy_mhz']:.0f} of {g['max_sm_clock_mhz']:.0f} MHz max; "
               f"busy time thermally throttled {g['pct_busy_time_thermal_throttled']}%, power-capped {g['pct_busy_time_power_capped']}%, HW slowdown {g['pct_busy_time_hw_slowdown']}%. "
               f"Reasons seen while busy (sample counts): {g['reasons_seen_busy']}. Details in `results/gpu_log_v2_summary.json`.", ""]
    md += ["## Caveats", "", f"- Small test set (n = {n_test}; lookalike {split['split_counts']['test']['lookalike']}, other_aid {split['split_counts']['test']['other_aid']}): CIs are wide and per-class F1 for rare classes swings with single images.",
           "- Zero-shot SigLIP2 is biased upward (labels were confirmed from its suggestions); SigLIP-based trained rows inherit some of that.",
           f"- Most labels are single-annotator ({split['label_resolution']['single']} of {split['n']}; {split['label_resolution']['consensus']} consensus, {split['label_resolution']['adjudicated']} adjudicated).",
           "- Seed spread reflects training randomness on one fixed grouped split, not split-to-split variance. One uploader group "
           f"({split['largest_groups'][0][0]}, {split['largest_groups'][0][1]} images) sits entirely in one split.",
           "- YOLO11-cls uses Ultralytics' own recipe (augmentation, optimizer, no class weights), so it is not identical to the ResNet/SigLIP recipe.", ""]
    (RES / "baseline_siglip2_v2.md").write_text("\n".join(md), encoding="utf-8")
    # chart
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    mets = [("test_accuracy", "accuracy"), ("test_macro_f1", "macro-F1")] + [(f"test_f1_{c}", f"F1 {c}") for c in CLASSES]
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150); W = 0.8 / max(len(rows), 1); x = np.arange(len(mets))
    for j, (name, kind, runs, s, note) in enumerate(rows):
        mu = [s[k]["mean"] if k in s else 0 for k, _ in mets]; sd = [s[k]["std"] if k in s else 0 for k, _ in mets]
        ax.bar(x + (j - (len(rows) - 1) / 2) * W, mu, W, yerr=sd, capsize=2, label=f"{name} ({kind})" + (" *" if "zero-shot" in name and "SigLIP" in name else ""),
               hatch="//" if kind == "untrained" else None, edgecolor="black", linewidth=0.4)
    ax.axhline(maj, ls="--", c="gray", lw=0.8); ax.text(len(mets) - 0.5, maj + 0.01, "majority-class acc", fontsize=7, color="gray", ha="right")
    ax.set_xticks(x, [m for _, m in mets]); ax.set_ylim(0, 1.05); ax.set_ylabel("score on v2 test")
    ax.set_title(f"WheelView v2 test (n={n_test}): untrained (hatched) vs trained; error bars = sd over seeds\n* SigLIP2 zero-shot biased upward: labels were confirmed from its suggestions", fontsize=9)
    ax.legend(fontsize=7, ncol=2, loc="upper right"); fig.tight_layout(); fig.savefig(RES / "baseline_siglip2_v2_comparison.png"); plt.close(fig)
    print("\n".join(md))

if __name__ == "__main__":
    main()
