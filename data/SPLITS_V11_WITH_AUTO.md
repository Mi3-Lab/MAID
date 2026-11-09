# `splits_v11_with_auto.csv` (v1.1 training)

Official v1.1 training split for MAID/WheelView on branch `v1.1-schema`.

- **n = 1080** = 460 human clean-set labels + 620 `auto_approved` model-consensus labels
- Built by `scripts/make_splits.py --include-auto-approved --seed 11 --out data/splits_v11_with_auto.csv`
- Column `label_source`: `human` | `model_consensus` (never rewrite model-consensus as human / Samir)
- Frozen v1.0 files untouched: `data/splits_v3.csv`, `data/manifest_v2.csv`, `data/splits_v1.csv`, `data/splits_v2.csv`
- Excludes: non-consensus assist suggestions, flagged, child, adjudication-only items (only `var/assist/auto_approved.jsonl` consensus rows are added)
- Leak groups intact (0 cross-split groups)
- Default for `scripts/train_baseline.py` / SigLIP / YOLO on this branch; pass `--splits data/manifest_v2.csv` for frozen v1.0

Summary: `results/splits_v11_with_auto_summary.json`
