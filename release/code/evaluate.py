#!/usr/bin/env python3
"""WheelView v1.0 official scorer: T1 (device type), T2 (robustness by condition), T4 (ambiguous-object rejection).

  python code/evaluate.py --pred my_test_predictions.csv                  # scores the test split
  python code/evaluate.py --pred preds.csv --split val --json scores.json
  python code/evaluate.py --pred seed0.csv --pred seed1.csv --pred seed2.csv   # mean +- s.d. over runs

--pred: CSV with header `image_id,device_type` (one row per image of the split; predicted class is one of
manual, powered, other_aid, ambiguous). Every image of the split must be predicted exactly once; extra IDs are an error.
--labels defaults to annotations/labels.csv next to this folder. Standard library only (Python >= 3.8).

T1  accuracy; macro-F1 over the 4 classes; per-class precision/recall/F1; "3-class macro-F1": images labelled
    ambiguous are dropped, the 4-way predictions are kept (an image predicted ambiguous counts as a miss), F1 averaged
    over manual/powered/other_aid. 95% bootstrap CIs (percentile, 2000 resamples of images, seed 0) for one run.
T2  T1 accuracy and macro-F1 (over the classes present in the subset) on condition subsets: occupied yes/no,
    occlusion none/partial/heavy, viewpoint front/side/rear/three-quarter, lighting daylight/indoor/other. n is printed;
    subsets with fewer than 10 images are reported but should not be interpreted.
T4  ambiguous-object rejection. Ambiguous false-positive rate (Amb-FPR) = share of ambiguous-labelled images predicted
    as a mobility device (manual/powered/other_aid); lower is better. Also reported: false-ambiguous rate = share of
    real devices predicted ambiguous. (T3, propulsion, has no baseline in v1.0 and is not scored here.)"""
import argparse, csv, json, random, statistics, sys
from collections import Counter
from pathlib import Path
CLASSES = ["manual", "powered", "other_aid", "ambiguous"]
DEVICES = CLASSES[:3]
HERE = Path(__file__).resolve().parent

def prf(y, p, classes):
    out = {}
    for c in classes:
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c); fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        out[c] = {"precision": pr, "recall": rc, "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0, "n": tp + fn}
    return out

def t1(y, p):
    if not y: return None
    pc = prf(y, p, CLASSES)
    keep = [(a, b) for a, b in zip(y, p) if a != "ambiguous"]
    pc3 = prf([a for a, _ in keep], [b for _, b in keep], DEVICES)
    present = [c for c in CLASSES if pc[c]["n"]]
    return {"n": len(y), "accuracy": sum(a == b for a, b in zip(y, p)) / len(y),
            "macro_f1": statistics.mean(pc[c]["f1"] for c in present) if present else 0.0,
            "macro_f1_3": statistics.mean(pc3[c]["f1"] for c in DEVICES if pc3[c]["n"]) if keep else None, "per_class": pc}

def boot(y, p, n=2000, seed=0):
    rng = random.Random(seed); idx = range(len(y)); acc, mf = [], []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]; r = t1([y[i] for i in s], [p[i] for i in s]); acc.append(r["accuracy"]); mf.append(r["macro_f1"])
    q = lambda v: [round(sorted(v)[int(0.025 * n)], 3), round(sorted(v)[int(0.975 * n) - 1], 3)]
    return {"accuracy_95ci": q(acc), "macro_f1_95ci": q(mf)}

SUBSETS = [("occupied", ["yes", "no"]), ("occlusion", ["none", "partial", "heavy"]),
           ("viewpoint", ["front", "side", "rear", "three-quarter"]), ("lighting", ["daylight", "indoor", "other"])]

def t2(rows, pred):
    out = {}
    for f, vals in SUBSETS:
        for v in vals:
            sel = [r for r in rows if (r.get(f) or "") == v or (v == "other" and f == "lighting" and (r.get(f) or "") not in ("daylight", "indoor"))]
            res = t1([r["device_type"] for r in sel], [pred[r["image_id"]] for r in sel])
            out[f"{f}={v}"] = None if res is None else {"n": res["n"], "accuracy": res["accuracy"], "macro_f1": res["macro_f1"]}
    return out

def t4(rows, pred):
    amb = [r for r in rows if r["device_type"] == "ambiguous"]; dev = [r for r in rows if r["device_type"] in DEVICES]
    return {"n_ambiguous": len(amb), "amb_fpr": (sum(pred[r["image_id"]] in DEVICES for r in amb) / len(amb)) if amb else None,
            "n_devices": len(dev), "false_ambiguous_rate": (sum(pred[r["image_id"]] == "ambiguous" for r in dev) / len(dev)) if dev else None}

def load_pred(path, ids):
    P = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rd = csv.DictReader(fh)
        if not {"image_id", "device_type"} <= set(rd.fieldnames or []): sys.exit(f"{path}: header must contain image_id,device_type")
        for r in rd:
            i, c = r["image_id"].strip(), r["device_type"].strip().lower()
            if c == "lookalike": c = "ambiguous"
            if c not in CLASSES: sys.exit(f"{path}: {i}: unknown class {c!r}")
            if i in P: sys.exit(f"{path}: {i} predicted twice")
            P[i] = c
    miss, extra = sorted(set(ids) - set(P)), sorted(set(P) - set(ids))
    if miss or extra: sys.exit(f"{path}: {len(miss)} split images missing {miss[:5]}, {len(extra)} IDs not in the split {extra[:5]}")
    return P

def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return None
    return {"mean": round(statistics.mean(vals), 4), "sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0}

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pred", action="append", required=True, help="predictions CSV (repeat for several runs/seeds)")
    ap.add_argument("--labels", default=str(HERE.parent / "annotations" / "labels.csv"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"]); ap.add_argument("--json", default=None)
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.labels, encoding="utf-8")) if r["split"] == a.split]
    if not rows: sys.exit(f"no rows with split={a.split} in {a.labels}")
    ids = [r["image_id"] for r in rows]; y = [r["device_type"] for r in rows]
    runs = []
    for pth in a.pred:
        P = load_pred(pth, ids); p = [P[i] for i in ids]
        r = {"pred_file": pth, "T1": t1(y, p), "T2": t2(rows, P), "T4": t4(rows, P)}
        if a.bootstrap: r["T1"]["bootstrap"] = boot(y, p, a.bootstrap)
        runs.append(r)
    summ = {"split": a.split, "n": len(rows), "class_counts": dict(Counter(y)), "n_runs": len(runs),
            "T1": {k: agg([r["T1"][k] for r in runs]) for k in ("accuracy", "macro_f1", "macro_f1_3")},
            "T1_per_class_f1": {c: agg([r["T1"]["per_class"][c]["f1"] for r in runs]) for c in CLASSES},
            "T2": {k: {"n": runs[0]["T2"][k]["n"], "accuracy": agg([r["T2"][k]["accuracy"] for r in runs]),
                       "macro_f1": agg([r["T2"][k]["macro_f1"] for r in runs])} for k in runs[0]["T2"] if runs[0]["T2"][k]},
            "T4": {"n_ambiguous": runs[0]["T4"]["n_ambiguous"], "amb_fpr": agg([r["T4"]["amb_fpr"] for r in runs]),
                   "false_ambiguous_rate": agg([r["T4"]["false_ambiguous_rate"] for r in runs])}, "runs": runs}
    f = lambda d: "n/a" if d is None else (f"{d['mean']:.3f}" + (f" ± {d['sd']:.3f}" if d["sd"] else ""))
    print(f"split={a.split} n={len(rows)} {dict(Counter(y))} runs={len(runs)}")
    print(f"T1 accuracy {f(summ['T1']['accuracy'])}  macro-F1 {f(summ['T1']['macro_f1'])}  3-class macro-F1 {f(summ['T1']['macro_f1_3'])}")
    if len(runs) == 1 and a.bootstrap: print(f"   bootstrap 95% CI: {runs[0]['T1']['bootstrap']}")
    print("   per-class F1: " + ", ".join(f"{c} {f(v)}" for c, v in summ["T1_per_class_f1"].items()))
    print("T2 condition                n   accuracy        macro-F1")
    for k, v in summ["T2"].items(): print(f"   {k:<24}{v['n']:>4}   {f(v['accuracy']):<15} {f(v['macro_f1'])}" + ("   (n<10)" if v["n"] < 10 else ""))
    print(f"T4 Amb-FPR {f(summ['T4']['amb_fpr'])} (n_ambiguous={summ['T4']['n_ambiguous']}); false-ambiguous rate {f(summ['T4']['false_ambiguous_rate'])}")
    if a.json: Path(a.json).write_text(json.dumps(summ, indent=1)); print("wrote", a.json)

if __name__ == "__main__":
    main()
