# MAID v1.1: data dictionary (draft, not released)

v1.1 builds on the frozen v1.0 release (455 images, IEEE DataPort). The v1.0 fields, labels, splits and derived files
stay exactly as they were. Nothing here changes `release/`, the v1.0 zip or `results/`. v1.1 annotation is a separate pass
with its own done flag (`_status_v11`) and its own derived files (`*_v11.*`). Those files are written only after v1.1
events exist, so v1.0's `derived/` stays byte-identical (tested in `tests/test_v11.py`).

## Schema v6 (2026-11-09)

Retired from the annotator, new-event validation, label filter, derive/export, and training-split manifests:
`lighting`, `environment`, `weather`, `ground_wet`, `size_class`, `ramp_deploy`, `ramp_reasons`, and size / tape-measure
estimate columns (`est_*`, `height_cm` / `width_cm` / `length_cm`). Historical events in `annotations/events/*.jsonl`
are never rewritten; derive and export ignore those keys. Frozen v1.0 release artifacts are unchanged.

Added `device_use_case` (specialized / standard / unknown) **above** `device_type`. Unset means not yet labelled;
new saves require a value. Machine hints, if any, stay suggestions and are never stored as human labels.

The Attributes pass now pre-selects the latest saved core labels (device type, occupied, etc.) and lets any labeler
change them; saves append correction events (latest wins).


## 1. v1.0 fields (unchanged)
`device_type` (manual = non-powered / powered / other_aid / ambiguous / unclear), `setting`, `occupied`, `occlusion`,
`propulsion`, `viewpoint`, `bbox` (`lighting` was retired in schema v6), `exclude`, `face_blur_missed`. Definitions are in `server/schema.json`.
`device_type` itself is unchanged (manual = non-powered / powered / other_aid / ambiguous / unclear). v1.1 adds a
**dependent** `chair_subtype` (and `drive_position` for powered chairs) below it; see section 2a. Scooters stay under
`other_aid`; "pediatric" is a size descriptor (`size_class = pediatric_size`), never a subtype.

## 2. New human-labelled fields (since 1.1; 2 annotators, 3rd on disagreement)
| field | values | meaning |
|---|---|---|
| `device_use_case` | specialized, standard, unknown | Visual judgement of intended use, shown above device type. specialized = special use cases; standard = everyday use; unknown = cannot reasonably tell. Optional on already-labelled images; required on new saves. |
| `control` | joystick, attendant_control, push_rims, tiller, none_visible, unknown | Control interface you can **see** in the image. joystick = armrest joystick (user). attendant_control = rear/attendant joystick or push handles in use. push_rims = hand rims on the rear wheels. tiller = handlebar steering (scooters). none_visible = no control interface visible. |
| `environment` | indoor, sidewalk, curbside, crosswalk_road, vehicle_interior, ramp_lift, parking_lot, other_outdoor, unknown | Where the device is. curbside = at the kerb edge or kerb cut. crosswalk_road = on the roadway or a crossing. ramp_lift = on a vehicle or building ramp or lift. |
| `weather` | sunny_clear, overcast, rain, snow, fog, night, indoor_na, unknown | Visible conditions. indoor_na for indoor scenes. night when it is dark outdoors. |
| `ground_wet` | yes, no, unknown | Wet, snowy or icy ground visible. |
| `size_class` | standard, wide, pediatric_size, unknown | Visual impression only (e.g. a clearly extra-wide frame; pediatric_size = a visibly child-sized chair, judged from the chair alone). **Not a measurement.** Never machine-suggested. |
| `chair_subtype` | manual: standard_folding, lightweight_rigid, negative_camber, transport_attendant, tilt_recline, bariatric, unknown · powered: power_standard, power_tilt_recline, unknown | Dependent on `device_type` (section 2a). |
| `drive_position` | mid_wheel, front_wheel, rear_wheel, unknown | Powered chairs only: which wheels are the large drive wheels. |
| `ramp_deploy` | safe, unsafe, not_needed, unknown | Research judgement of the **scene**: could a vehicle wheelchair ramp be deployed here? See section 2b. |
| `securement_hardware` | docking_bracket, tiedown_points, not_visible, unclear | Visible securement fittings only. docking_bracket = under-seat docking pin/bracket (e.g. EZ Lock / Q'Straint). tiedown_points = marked 4-point / WC19 loops or brackets. not_visible = none in view. unclear = cannot tell. **Not a safety certification.** See §2c. |
| `ramp_reasons` | multi-select: insufficient_space, obstruction, uneven_surface, steep_slope, curb_height_issue, wet_icy_surface, traffic_hazard, no_wheelchair_user, user_not_waiting_unclear, unknown | Shown when `ramp_deploy` is unsafe or not_needed. Stored as a JSON list; in CSVs as sorted values joined with `+` (e.g. `insufficient_space+obstruction`). `unknown` cannot be combined. |

### 2a. Chair subtype (dependent field)
* Shown only when the device is `manual` or `powered`: when **the annotator** picked it in this pass, else the final v1.0
  label, else the model's v1.0 suggestion (the app then marks the row "(AI: manual)"). The option list follows that
  parent: the manual list for manual chairs, the powered list for powered chairs.
* `chair_subtype` and `drive_position` also appear in the normal v1.0 labelling pass, right after device type, marked
  "v1.1, optional". They're stored as v1.1 fields and never change v1.0 derived files.
* Manual definitions: standard_folding = cross-brace folding frame; lightweight_rigid = rigid/active-user frame, often no
  push handles; negative_camber = wheels angled in at the top (sports/active); transport_attendant = small rear wheels,
  pushed by an attendant; tilt_recline = tilting or reclining seat system (manual); bariatric = visibly heavy-duty /
  extra-wide frame. Powered: power_standard; power_tilt_recline = power seat functions visible. Choose `unknown` when
  you can't tell.
* A value whose parent no longer applies (e.g. the device was later set to other_aid) is ignored by consensus
  (`wvcore.dep_ok`).


### 2c. Securement hardware (visible fittings only; not a certification)
`securement_hardware` records what an annotator can **see** in one photo: under-seat docking hardware, tiedown loops/brackets, nothing visible, or unclear. It does **not** certify crashworthiness, WC19 compliance, or that a vehicle may secure the chair.

**Vehicle-system guidance (model/API):** when the model (or a human label) reports `docking_bracket` at **high confidence**, a boarding stack may *prepare* a docking station as a prior — still confirm with the rider. `not_visible`, `unclear`, or **low confidence** => defer to rider confirmation / additional sensing. Never use this field alone to authorize securement or boarding.

SigLIP attr-v4 scores this field; top-1 probability below the configured threshold (default 0.45) is rewritten to `unclear`.

### 2b. Ramp deployment labels: research judgement, not validated
`ramp_deploy` records what an annotator judges from **one photo**: is there room and a suitable surface to deploy a
vehicle ramp here, and is someone actually waiting? It is **not** a safety assessment and must not be used to decide a
real deployment. See `docs/RAMP_LABELS_v1.1.md`.

Rules for annotators: choose `unknown` when you can't tell. Machine suggestions are shown in orange and never saved,
so a human value is only recorded when a human sets it. v1.0 labels appear as read-only context and are never changed.

Consensus (`derived/labels_v11.csv`, `status_v11`): `final` when the done annotators agree on every field (or an
adjudication resolves it), `needs_third_annotator` on disagreement, `in_progress` with fewer than 2 done.
Agreement: `kappa_v11.json`.

**Who's in the attribute pool:** clean v1.0 images (≥1 human done, not labeler-excluded, not on a safety hold or
blocklist). Any image that has **ever** had an exclude or face-blur flag is kept out (conservative: even when it was
adjudicated as keep). Contributed images join once they're `active` and have v1.0 labels.

## 3. Machine suggestions (never labels)
`python scripts\auto_annotate.py --attr` writes `annotations/auto/siglip2-so400m-224_attr-v2.jsonl` (SigLIP2
zero-shot, separate `_attr-` tag; attr-v2/v3/v4 adds chair_subtype within the manual/powered list, drive_position, ramp_deploy and
per-reason ramp_reasons scores). Optional assist models (`scripts/assist_annotate.py`, `docs/ANNOTATION_MODELS_v1.1.md`)
write `annotations/auto/<vlm>_attr-vlm-v1.jsonl` the same way, plus boxes/masks in `var/assist/` (local only).
The app shows these as orange hints with the model name. They aren't validated and can't be bulk-accepted. When a
human picks a hint, the event carries provenance (`source=auto_attr`, `model_tag`), but it counts as that human's label.
`pediatric_size` and `unknown` are never suggested.

## 3a. Synthetic images (train-only augmentation)
`var/synthetic/synthetic_manifest.csv` (local, gitignored) from `scripts/synth_augment.py`: `image_id` (`wv_800001`+),
`synthetic=true`, `split=train`, `leak_group` (the source's), `source_image_id`, background provenance
(`background_id, background_source, background_license, background_author, background_url`), `scene, environment,
weather, ground_wet, weather_variant, lighting_variant, mask_source, box_source, transform_params, generator,
generated_at`, inherited chair labels, and empty `ramp_deploy/ramp_reasons`. They are never in val/test, never in
metrics, never in a release. See `docs/SYNTHETIC_AUGMENTATION_v1.1.md`.

## 3b. Filtering
Any field above (plus split, approval, status_v11, synthetic, size_basis, label_source, excluded) can be filtered in the
app, the server (`/api/filter`) and `scripts/filter_images.py`. See `docs/FILTERING_v1.1.md`.

## 4. Wheelchair size: three separate layers
For in-vehicle systems. Monocular web photos have **no metric scale**, so the three layers below are kept strictly apart.

### 4a. Measured dimensions (real sources only) — `data/measurements.csv`
Columns: `image_id, chair_ref, height_cm, width_cm, length_cm, measurement_source, measured_by, measured_on, configuration, notes`.
`measurement_source` ∈ user_measured, manufacturer_spec, depth_sensor, reference_object, none. Filled only from a real
measurement (e.g. a tape measure on the actual chair, or the spec sheet for that exact model). `chair_ref` links photos
of the same physical chair. `configuration` records e.g. "backrest up, footrests on". Currently **empty**: no real
measurements exist yet.

### 4b. Non-metric box proxies — `var/v11/size_estimates.csv`
`box_w_norm, box_h_norm, box_area_norm` (fraction of image), `aspect_hw_px` (box height/width in pixels),
`box_w_reflects` (front/rear → width, side → length, three-quarter → mixed), `truncated` (box touches the image edge),
`non_metric=1`. These are image-relative and **not** centimetres.

### 4c. Rough estimates (`measurement_source = rough_estimate`) — same CSV, `est_*` columns
`est_height_cm, est_width_cm, est_length_cm`, each with `_low_cm` / `_high_cm`, plus `method, basis, flags,
estimator_version (rough-prior-v1), validated`. Written by `scripts/size_estimate.py`. **Never** copied into the measured
columns. Status: **unvalidated** (no real measurements exist). See `docs/SIZE_ESTIMATION_v1.1.md` for the assumptions.

## 5. Contributed (own) images — `data/contributed.csv`
Columns: `image_id, status, source(=contributed), creator, license, license_url, title, people, identifiable_people,
consent, own_photo_attested, no_children_attested, approved_by, approval_ref, imported_at, imported_by, orig_sha256,
sha256, width, height, n_faces_blurred, chair_ref, series, notes`.

* IDs: `wv_900001` and up, so they never clash with web-sourced IDs.
* `status`: `pending_approval` (default: they're not served, labelled or exported) → `active` only via
  `--approve-pending --approved-by ... --approval-ref ...`. This exists because the project rule says "do not take any
  photos or videos yourselves", so the lab lead needs to approve own photos explicitly.
* Licenses: CC0-1.0, CC-BY-(SA-)2.0/3.0/4.0, PDM.
* On import: EXIF is stripped, faces are blurred with YuNet, the long side is resized to 640 px, exact + perceptual
  (incl. mirrored) near-duplicates are rejected against existing images, and the original file is never modified.
  The importer refuses unless `--attest-own-photo` and `--attest-no-children` are given. If `--people consented` is
  used, `--consent-note` must say where the signed form is kept.
* Images stay local (`images\`, `images_blurred\`). They never go to GitHub or shared drives.

## 6. `data/manifest_v11.csv` (built by `scripts/build_manifest_v11.py`)
`image_id, source, split, split_origin, train_eligible_v10, v11_eligible, v11_exclusion_reason, device_type,
setting, occupied, occlusion, propulsion, viewpoint, lighting, v11_status, v11_annotators, control, environment, weather,
ground_wet, size_class, chair_subtype, drive_position, ramp_deploy, ramp_reasons, securement_hardware` (final values only; multi as `a+b`), `license, creator, leak_group, path, phash_blurred`, size proxy + `est_*`
columns (single-box images only), and the measured columns (real sources only).

* v1.0 images keep their **frozen** v1.0 split (`split_origin=v1.0`).
* Contributed images: leak group `cNNNN` per creator (or `--contrib-group series`). The split is inherited from a v1.0
  image by the same creator. If that creator's v1.0 images span several splits, the image is excluded. Otherwise the
  split is a stable hash (70/15/15). Images that are pHash near-dups (≤8, incl. mirror) of an image in another split are
  excluded (`near_dup_bridges_split`).
* `v11_exclusion_reason`: excluded_in_v1.0, open_flag_pending_review, v11_not_final, creator_spans_v1.0_splits,
  near_dup_bridges_split, no_final_v10_label, image_missing_locally.
* Audit: `python scripts\check_leakage.py --v11 data\manifest_v11.csv`.

## 7. Workflows (PowerShell, repo root, venv)
```powershell
# import your own photos (they stay pending until approved)
.venv\Scripts\python.exe scripts\import_contributed.py C:\path\to\photos --creator "Samir ..." --license CC-BY-4.0 `
    --people none --attest-own-photo --attest-no-children --dry-run          # preview first, then drop --dry-run
# optional real measurements of the same chair (tape measure):
#   --chair-ref chair01 --height-cm <cm> --width-cm <cm> --measurement-source user_measured --configuration "footrests on"
.venv\Scripts\python.exe scripts\import_contributed.py --approve-pending --approved-by ross --approval-ref "email 2026-11-.."

# machine suggestions for the attribute pass (hints only)
.venv\Scripts\python.exe scripts\auto_annotate.py --attr

# label: start the server as usual, open the app, press v (or the "Attributes v1.1" header button)

# rough sizes, v1.1 manifest, leakage audit, heads (heads train only once enough final labels exist)
.venv\Scripts\python.exe scripts\size_estimate.py
.venv\Scripts\python.exe scripts\build_manifest_v11.py
.venv\Scripts\python.exe scripts\check_leakage.py --v11 data\manifest_v11.csv
.venv\Scripts\python.exe scripts\train_multitask.py            # add --joint for the shared-adapter model

# focus on one detail (same syntax as the app's filter panel and URL #filter=...)
.venv\Scripts\python.exe scripts\filter_images.py --filter "device_type:manual;chair_subtype:_unlabeled" --counts

# optional: assist models (see docs\ANNOTATION_MODELS_v1.1.md) and synthetic train-only augmentation
.venv\Scripts\python.exe scripts\assist_models.py
.venv\Scripts\python.exe scripts\assist_annotate.py --tasks vlm,detect,segment --limit 50
.venv\Scripts\python.exe scripts\synth_augment.py --n 200
.venv\Scripts\python.exe scripts\check_leakage.py --v11 data\manifest_v11.csv --synthetic var\synthetic\synthetic_manifest.csv
.venv\Scripts\python.exe scripts\train_multitask.py --synthetic var\synthetic\synthetic_manifest.csv
```
