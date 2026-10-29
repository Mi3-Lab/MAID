# WheelView v1.0 code

Run everything from the dataset root (the folder that contains `images/`, `annotations/`, `splits/`).

| File | Purpose | Needs |
|---|---|---|
| `wheelview_loader.py` | load images + labels by split; `--summary`; `--flatten DIR` (images as `DIR/<image_id>.jpg` for the training scripts) | Python 3.9+, Pillow (torch optional) |
| `evaluate.py` | official scorer: T1 device type (accuracy, macro-F1 over 4 and 3 classes, per-class P/R/F1, bootstrap 95% CI), T2 robustness by occupancy / occlusion / viewpoint / lighting, T4 ambiguous-object rejection (Amb-FPR and false-ambiguous rate) | standard library only |
| `scripts/train_siglip.py` | SigLIP 2 zero-shot, linear probe and fine-tune (last 4 blocks) baselines | `requirements.txt` |
| `scripts/train_baseline.py` | ResNet-50 fine-tune baseline | `requirements.txt` |
| `scripts/train_yolo.py` | YOLO11s-cls fine-tune and YOLO-World v2-s zero-shot baselines | `requirements.txt` |
| `scripts/predict.py` | predicts single images from saved weights + the image only | `requirements.txt` |
| `scripts/bench_latency.py` | batch-1 latency benchmark | `requirements.txt` |
| `scripts/check_leakage.py` | split / feature / file / runtime leakage checks; `--release DIR` audits a built release folder | standard library, scipy + Pillow for `--release` |
| `scripts/auto_annotate.py` | SigLIP 2 zero-shot prompt ensemble (used by the zero-shot baseline and the labeling suggestions) | `requirements.txt` |
| `server/schema.json`, `server/wvcore.py` | label schema (class order) and shared helpers imported by the scripts | |
| `requirements.txt`, `requirements-lock.txt` | direct dependencies pinned to the versions used; full `pip freeze` of that environment | |

## Score your model

Write one CSV per run with a header `image_id,device_type` (one of `manual`, `powered`, `other_aid`, `ambiguous`)
covering every image in the split, then:

```bash
python code/evaluate.py --labels annotations/labels.csv --split test --pred my_preds.csv [--pred seed1.csv ...] --json out.json
```

Several `--pred` files are summarized as mean ± s.d. T2 subsets with fewer than 10 images are flagged.
There is no T3 (propulsion) baseline in v1.0 because the propulsion labels have known inconsistencies.

## Reproduce the baselines

```bash
pip install -r code/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128   # CPU: drop "+cu128"
python code/wheelview_loader.py --root . --flatten flat                                      # images as flat/<id>.jpg
python code/scripts/train_siglip.py   --splits annotations/labels.csv --images flat --run siglip2_v3 --seeds 0,1,2
python code/scripts/train_baseline.py --splits annotations/labels.csv --images flat --run resnet50_v3 --seeds 0,1,2
python code/scripts/train_yolo.py     --splits annotations/labels.csv --images flat --tag v3 --seeds 0,1,2
python code/scripts/predict.py --kind siglip-ft --weights <run dir>/ft_seed0_best.pt flat/wv_000235.jpg
```

Models are selected on `val` only (early stopping, best checkpoint, probe regularization); `test` is scored once.
SigLIP 2 weights (`google/siglip2-so400m-patch14-224`), ImageNet ResNet-50 and Ultralytics YOLO weights are
downloaded from their public sources on first use. Reported results were produced on Windows 11, Python 3.13.5,
NVIDIA RTX 3000 Ada Generation Laptop GPU; GPU nondeterminism means re-runs can differ slightly from the report.

## Leakage check

```bash
python code/scripts/check_leakage.py --release .
```
checks that labels, splits, files and sha256 sums agree, that no image is outside the expected set, that image
files carry no metadata segments and are at most 1280 px, and that no near-duplicate (perceptual hash, including
mirror images) crosses splits.
