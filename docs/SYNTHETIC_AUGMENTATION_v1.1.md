# Synthetic augmentation (MAID v1.1): train-only

`scripts/synth_augment.py` cuts a wheelchair out of a **train** image and composites it into licensed, people-free scene
backgrounds (curbside, crossing, sidewalk, ramp/lift, vehicle door, parking lot), with weather and lighting variants.
The goal is more variety in exactly the scenes the ramp / environment heads care about, without new photos of people.

## Hard rules (enforced in code and by `check_leakage.py --synthetic`)
* **Sources:** train split only (frozen split from `data/manifest_v11.csv`, else `data/manifest_v2.csv`), trainable,
  never excluded or child-flagged, `device_type` manual or powered, always the **face-blurred** copy (`images_blurred/`).
  `--empty-only` restricts to unoccupied chairs.
* **Outputs** are `split=train`, `synthetic=true`, inherit the source's `leak_group`, get ids `wv_800001`+ (never a real
  id), and are **excluded from all val/test metrics** (`train_multitask.py` reports `metrics_on: real val/test images only`).
* **Backgrounds** only from `data/backgrounds/backgrounds.csv` (license, author, source_url required; `people=none`
  required) or, via `dataset_image_id`, an approved **train** image of this dataset. Val/test images are refused.
  **No photos of people are taken, generated or downloaded** by this pipeline; nothing is downloaded at all.
* `var/synthetic/` and the background photos are gitignored. Images never go to GitHub.

## Setup
1. Put licensed people-free scene photos in `data/backgrounds/` (e.g. CC0 street photos you already have the rights to)
   and add a row per photo to `data/backgrounds/backgrounds.csv`:
   `background_id,file,scene,source_url,license,author,people,ground_y,environment,dataset_image_id,notes`.
   `ground_y` = height fraction of the horizon line. `environment` optionally overrides the scene's default label.
2. `python scripts\synth_augment.py --check-backgrounds` lists accepted rows and why others were refused.
3. Optional better cutouts: `python scripts\assist_annotate.py --tasks detect,segment` (SAM 2 masks in `var/assist/`).

## Run
```powershell
.venv\Scripts\python.exe scripts\synth_augment.py --n 200 --seed 0
.venv\Scripts\python.exe scripts\synth_augment.py --n 100 --scenes curbside,vehicle_door --variants rain,wet,night --empty-only
.venv\Scripts\python.exe scripts\check_leakage.py --v11 data\manifest_v11.csv --synthetic var\synthetic\synthetic_manifest.csv
.venv\Scripts\python.exe scripts\train_multitask.py --synthetic var\synthetic\synthetic_manifest.csv --run mt_syn
```
Re-running appends (ids continue). Delete `var/synthetic/` to start over.

## How an image is made
1. **Box:** the largest human box of the source (manifest `bboxes`, else events read-only), or an assist mask's extent.
2. **Mask:** SAM 2 mask from `var/assist/*/masks.jsonl` (or `--sam` live), else GrabCut inside the box (if OpenCV is
   installed), else a feathered ellipse; edges feathered.
3. **Scale + ground placement:** a foot point is sampled below the horizon; object height in px =
   object height (source `est_height_cm`, else 95 cm manual / 110 cm powered) / camera height (150 cm) x
   (foot_y - horizon_y). So chairs further away are smaller and stand on the ground plane.
4. **Contact shadow:** soft ellipse under the chair (weaker for overcast/fog/night).
5. **Colour matching:** per-channel mean/std of the cutout moved 60% towards the background region it lands on.
6. **Variant:** `none, overcast, rain` (streaks, darker, wet reflection), `wet`, `fog` (depth haze), `night` (gamma,
   blue cast, vignette), `dusk`; lighting `normal, dusk, bright`.

## Manifest columns (`var/synthetic/synthetic_manifest.csv`)
`image_id, file, synthetic, split, leak_group, source_image_id, source_split, background_id, background_source,
background_license, background_author, background_url, scene, environment, weather, ground_wet, weather_variant,
lighting_variant, mask_source, box_source, transform_params (JSON: box, scale, foot/horizon, seed), generator,
generated_at, device_type, chair_subtype, drive_position, size_class, control, occupied, propulsion, ramp_deploy,
ramp_reasons`.
Chair attributes are inherited from the source; scene labels (`environment`, `weather`, `ground_wet`) come from the
background/variant; **`ramp_deploy`/`ramp_reasons` stay empty** (needs a human). Filter in the app/CLI with `synthetic:yes`.

## Limits
Compositing artefacts (edges, lighting direction, occupant cut at the box) are visible; treat synthetic data as a
regulariser and always report metrics on real val/test images only.
