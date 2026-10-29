# Changelog

## v1.0 (2026-10-29)

First public release, prepared for the IEEE AgeTech Global Dataset Competition 2026.

* 464 face-blurred images (458 in train/val/test splits 321/68/69, 6 `excluded` with device type `unclear`).
* `annotations/labels.csv` (device type plus occupied, propulsion, occlusion, viewpoint, lighting, setting; labeling
  provenance; license and grouping keys), split files, preliminary COCO boxes, per-annotator boxes.
* `LICENSES.csv` from the 26-28 October 2026 license re-check of every image.
* Splits v3 (seed 20261027), grouped by leak group; checked by `code/scripts/check_leakage.py`.
* The class `lookalike` used during labeling is called `ambiguous` in this release (rename only; same images).
* Final face re-check of the release copies: 7 faces missed by the first blur pass (mostly side profiles) were blurred
  in 7 training images (`face_recheck.csv`). The baselines in the report were trained on the copies before this fix.
* 4 images from one street scene with a possible child are held back (safety hold) and are not included.
* Code: loader, evaluation (T1/T2/T4), training and prediction scripts, leakage checker, pinned requirements.

Known issues carried into v1.0 (not edited, documented): `propulsion` inconsistencies (57 manual chairs marked
`not_applicable`, 7 non-manual devices marked `self`/`attendant`); `occlusion` low agreement; boxes preliminary.

Removal requests and label corrections received after release will be listed here.
