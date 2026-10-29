# WheelView v1.0

An annotated, face-blurred image dataset for recognizing **wheelchair type** (manual, powered, other mobility aid, or
an ambiguous wheelchair-like object) for accessible curbside transportation. Submitted to the IEEE AgeTech Global
Dataset Competition 2026 (theme 3, Mobility & Assistive Technologies).

Authors: Samir Ahmed, Shaaz Saleem, Wesley Maia, Ross Greer (Mi³ Lab, UC Merced).
Contact (questions, label corrections, **removal requests**): Ross Greer, rossgreer@ucmerced.edu

> Labels describe the device and the scene only. Do **not** use WheelView to infer anyone's disability, medical
> condition, age, gender, race, abilities, or identity.

## Contents

| Path | What it is |
|---|---|
| `images/{train,val,test}/<image_id>.jpg` | 449 images with a determinable device type (319 / 66 / 64) |
| `images/excluded/<image_id>.jpg` | 6 images whose device type is `unclear`; in no split, not for training or scoring |
| `annotations/labels.csv` | one row per image (455 rows): split, labels, provenance, license, grouping keys, sha256 |
| `annotations/coco_{train,val,test}.json` | device bounding boxes in COCO format, **preliminary** (see below) |
| `annotations/boxes_per_annotator.csv` | the raw boxes as each annotator drew them, before merging |
| `splits/{train,val,test}.txt` | image IDs per split (fixed; do not re-split) |
| `LICENSES.csv` | per-image license, creator, title, source URL, license as recorded and as re-checked, modifications, ready-to-use attribution line |
| `SHA256SUMS.txt` | sha256 of every image file |
| `face_recheck.csv` | 7 images whose release copy got extra face regions blurred at the final check (box, before/after sha256) |
| `DATASHEET.md` | datasheet (Gebru et al. style): motivation, collection, labeling, uses, limitations |
| `CHANGELOG.md` | version history |
| `code/` | loader, evaluation, training/prediction scripts, leakage checker, pinned requirements (see `code/README.md`) |

Images are JPEG, at most 640 px on the longer side, with no EXIF/XMP/ICC metadata (no GPS).

### Class counts (images in splits)

| Class (`device_type`) | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| `manual` | 200 | 41 | 39 | 280 |
| `powered` | 66 | 14 | 14 | 94 |
| `other_aid` (mobility scooter, walker, rollator) | 42 | 9 | 9 | 60 |
| `ambiguous` (stroller, shopping cart, office chair, ... not clearly a wheelchair) | 11 | 2 | 2 | 15 |
| **Total** | 319 | 66 | 64 | 449 |

## How to load

```python
import csv
rows = list(csv.DictReader(open("annotations/labels.csv", encoding="utf-8")))
train = [r for r in rows if r["split"] == "train"]          # r["file"] is the image path relative to this folder
```

or with the bundled loader (standard library + Pillow; `torch_dataset()` needs PyTorch):

```python
import sys; sys.path.insert(0, "code")
from wheelview_loader import WheelView
wv = WheelView(".")
for img, row in wv.iter("train"): ...                    # PIL image + labels.csv row
ds = wv.torch_dataset("train", transform=my_transform)        # (image, class_index) pairs
```

`python code/wheelview_loader.py --root . --summary` prints split and class counts.

### Columns of `annotations/labels.csv`

| Column | Meaning |
|---|---|
| `image_id`, `file`, `split` | `wv_NNNNNN`; path of the image; `train` / `val` / `test` / `excluded` |
| `train_eligible`, `exclusion_reason` | 1/0; `device_unclear` for the 6 excluded images |
| `device_type` | target: `manual`, `powered`, `other_aid`, `ambiguous` (`unclear` only in `excluded`) |
| `occupied` | `yes` / `no`: a person is seated in the device |
| `propulsion` | `self` / `attendant` / `not_applicable` / `unclear`: whether a second person visibly pushes a manual chair (scene, not ability). **Known noisy, see Limitations** |
| `occlusion` | `none` (<10% hidden) / `partial` (10-50%) / `heavy` (>50%). **Low agreement (kappa 0.09)** |
| `viewpoint` | `front` / `rear` / `side` / `three-quarter` |
| `lighting` | `daylight` / `shade` / `indoor` / `low_light` |
| `setting` | `curb` / `vehicle` / `sidewalk` / `indoor` / `other` |
| `label_how`, `labelers`, `adjudicated_by` | `single` / `consensus` / `adjudicated`; who labeled; who adjudicated |
| `labeled_via` | per annotator: `manual` or `confirmed_suggestion` (picked a shown SigLIP 2 zero-shot suggestion). For analysis only; never use as a feature |
| `n_boxes` | number of device boxes (max over annotators) |
| `width`, `height` | image size in pixels |
| `license`, `license_url`, `license_status`, `creator`, `source_site`, `source_url` | attribution (full details in `LICENSES.csv`) |
| `uploader_group`, `dup_cluster`, `leak_group` | grouping keys; `leak_group` is the unit the splits were made on |
| `sha256` | sha256 of the image file |

Empty string = value missing or annotators disagreed and it was not resolved (1 `setting`, 1 `lighting`).

### Bounding boxes (preliminary)

Boxes are `[x, y, w, h]` in pixels in the COCO files (normalized 0-1 `[x, y, w, h]` in `boxes_per_annotator.csv`).
Most images were boxed by one annotator; where two annotators boxed an image, boxes matching at IoU >= 0.5 were
averaged and the rest kept. Box categories are copied from the image-level `device_type`, not labeled per box.
They have not been reviewed as a detection benchmark; treat them as preliminary.

## Rules for fair results

* Use the provided splits. Re-splitting breaks the leak groups (near-duplicates, same uploader/creator, photo bursts
  and series are kept in one split) and inflates scores.
* Train on `train`, choose models and hyperparameters on `val`, report `test` once.
* **Do not sort, bin or batch by `image_id`**: IDs follow collection order and correlate with class.
* Do not use `labeled_via`, `source_site`, `uploader_group`, `dup_cluster`, `leak_group` or `license*` as inputs.
* Report macro-F1 (classes are imbalanced) and per-condition results; `code/evaluate.py` computes the WheelView tasks
  (T1 type, T2 robustness by condition, T4 ambiguous-object rejection) from a CSV of `image_id,device_type`.

## License summary

* **Images keep their own licenses**, listed per image in `LICENSES.csv`: CC BY 2.0 (235), Public Domain Mark 1.0
  (205), CC BY-SA 2.0 (9), CC0 1.0 (6). All are Flickr photographs. Each image was modified for WheelView (resized,
  re-encoded, metadata removed, and detected faces pixelated and blurred); the `attribution` column gives the credit
  line to use. The per-image license in `LICENSES.csv` is the license of that image file.
* The **9 CC BY-SA 2.0 images** are adaptations distributed under CC BY-SA 2.0; adaptations of them must keep that
  license (ShareAlike).
* **9 images excluded for license reasons** (2026-10-29): Open Images recorded them as CC BY 2.0 at collection,
  but their Flickr pages later showed a different license or required a login. They are not distributed.
* **Annotations, documentation and code** written by the WheelView team are made available under the Creative Commons
  Attribution (CC BY) terms that IEEE DataPort applies to uploaded datasets.
* Please cite the WheelView dataset (DOI assigned by IEEE DataPort) and credit the photographers.

## Removal requests and corrections

If you appear in an image, or are the photographer, and want an image removed, email rossgreer@ucmerced.edu with the
`image_id`. Removals and label corrections are recorded in `CHANGELOG.md` and applied in the next versioned release;
the test split is otherwise kept fixed across versions.

## Limitations (short; details in DATASHEET.md)

* 449 split images, all from Flickr; one U.S. government Flickr account supplies 197 (all in train).
* Web photos, not vehicle or curb cameras: only 12 curb and 15 vehicle scenes; 4 low-light images.
* Test set is small (64 images; 2 ambiguous, 9 other aid), so confidence intervals are wide.
* Most images have one annotator; 132 were labeled by two. Annotators could confirm SigLIP 2 zero-shot suggestions,
  so labels are not independent of that model, and SigLIP-based scores are likely optimistic.
* `occlusion` has near-chance agreement; `propulsion` has known inconsistencies (57 manual chairs marked
  `not_applicable`, 7 non-manual devices marked `self`/`attendant`). No baseline is reported for propulsion in v1.0.
* Faces were blurred automatically (YuNet). A second YuNet pass over the final release copies flagged 235 detections
  outside the blurred regions; each was inspected visually on contact sheets, and 7 were visible faces the first pass had missed (mostly
  side profiles). Those 7 regions were blurred in the release copies of 7 training images (`face_recheck.csv`). The
  rest were tiny background faces a few pixels wide, already-blurred regions, or non-faces. Automated detection can
  still miss faces (e.g. strong profiles or partly hidden faces); report any you find (contact below) and the image will
  be fixed or removed. License plates, house numbers and signs were **not** blurred.
* Device types do not separate tilt-in-space, standing, sport, or power-assisted manual chairs.
