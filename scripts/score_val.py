#!/usr/bin/env python3
"""Score every saved trained model of a run tag on the VAL split only (for model selection), via scripts/predict.py.

predict.py gets only the weights and the val image paths (no labels); labels are joined here afterwards.
Writes results/val_scores_<tag>.json. The test split is never touched.

  python scripts/score_val.py --tag v3"""
import argparse, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from train_baseline import CLASSES, DEFAULT_SPLITS, metrics, read_split_rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="v3"); ap.add_argument("--splits", default=DEFAULT_SPLITS)
    ap.add_argument("--seeds", default="0,1,2"); a = ap.parse_args()
    val = [r for r in read_split_rows(a.splits) if r["split"] == "val"]
    paths = [str(ROOT / "images_blurred" / f"{r['image_id']}.jpg") for r in val]
    R = ROOT / "var" / "runs"; t = a.tag
    kinds = {"resnet": lambda s: R / f"resnet50_{t}" / f"seed{s}_best.pt",
             "siglip-probe": lambda s: R / f"siglip2_{t}" / f"probe_seed{s}_head.npz",
             "siglip-ft": lambda s: R / f"siglip2_{t}" / f"ft_seed{s}_best.pt",
             "yolo-cls": lambda s: R / f"yolo11s_cls_{t}_seed{s}" / "weights" / "best.pt"}
    out = {"tag": t, "split": "val", "n": len(val), "models": {}}
    for kind, wf in kinds.items():
        runs = []
        for s in [int(x) for x in a.seeds.split(",")]:
            w = wf(s)
            if not w.exists(): continue
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "predict.py"), "--kind", kind, "--weights", str(w)] + paths,
                               capture_output=True, text=True, cwd=ROOT)
            pred = {}
            for line in r.stdout.splitlines():
                try: d = json.loads(line)
                except Exception: continue
                if "image" in d: pred[Path(d["image"]).stem] = d["pred"]
            if len(pred) != len(val): print(f"{kind} seed{s}: {len(pred)}/{len(val)} predictions; stderr tail: {r.stderr[-500:]}"); continue
            m = metrics([CLASSES.index(x["label"]) for x in val], [CLASSES.index(pred[x["image_id"]]) for x in val])
            runs.append({"seed": s, "weights": w.relative_to(ROOT).as_posix(), "val_accuracy": round(m["accuracy"], 4), "val_macro_f1": round(m["macro_f1"], 4)})
            print(kind, s, runs[-1]["val_accuracy"], runs[-1]["val_macro_f1"], flush=True)
        if runs:
            import numpy as np
            out["models"][kind] = {"runs": runs, "mean_val_macro_f1": round(float(np.mean([x["val_macro_f1"] for x in runs])), 4),
                                   "mean_val_accuracy": round(float(np.mean([x["val_accuracy"] for x in runs])), 4)}
    json.dump(out, open(ROOT / "results" / f"val_scores_{t}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
