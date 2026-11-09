# Filtering (MAID v1.1)

Focus on one detail ("only manual chairs without a subtype yet", "rainy curbside scenes", "ramp unsafe because of an
obstruction") in the app, through the server, or from the command line. All three use the same filter text and the same
code (`server/wvcore.py`: `parse_filter`, `image_facts`, `run_filter`), so they give the same answer.

## Filter text
```
key:value[,value];key:value[;basis:human|model|either]
```
* OR within a key, AND across keys: `device_type:manual,powered;weather:rain` = (manual or powered) and rain.
* Keys: every label field (`device_type, occupied, propulsion, occlusion, viewpoint, lighting, setting, control,
  environment, weather, ground_wet, size_class, chair_subtype, drive_position, ramp_deploy, ramp_reasons, securement_hardware`) plus:

| pseudo key | values |
|---|---|
| `split` | train, val, test, none (from data/manifest_v11.csv, else manifest_v2.csv) |
| `approval` | v1.0 status: final, in_progress, needs_third_annotator, needs_adjudication, unlabelled |
| `status_v11` | same, for the v1.1 attribute pass |
| `synthetic` | yes, no (var/synthetic/synthetic_manifest.csv) |
| `size_basis` | measured (data/measurements.csv), estimate_only (var/v11/size_estimates.csv), none |
| `label_source` | human, model_only, unlabeled |
| `excluded` | yes, no |

* Special values on any label field: `_unlabeled` (no human value) and `_model_only` (no human value, but a machine
  suggestion exists).
* `basis` (default `human`): which values a label field is matched against. `human` = the resolved value when the image
  is final for that pass, else every annotator's latest value. `model` = machine suggestions only (v1.0 review model,
  SigLIP attribute hints, VLM hints). `either` = human or model. Model values never become labels by being filtered.
* Multi-select fields (`ramp_reasons`) match when any selected reason matches.
* Unknown keys or values are refused (HTTP 400 / CLI error), so typos don't silently match nothing.

Examples
```
device_type:manual;chair_subtype:_unlabeled
device_type:powered;drive_position:_unlabeled
environment:curbside,crosswalk_road;weather:rain,snow;ground_wet:yes
ramp_deploy:unsafe;ramp_reasons:obstruction,insufficient_space
chair_subtype:tilt_recline;basis:model
status_v11:needs_third_annotator
```

## App (desktop + iPhone)
* **Filter** button in the header (or **Shift+F**) opens the panel: one chip per value with its count; tap chips to
  combine. "Basis" switches human / model / either. **Clear** removes it; **Copy link** copies a URL with
  `#filter=...`, so the same view opens on another device.
* While a filter is active the header shows `match k/N`; **prev/next (← → or the on-screen buttons) only step through matches**, and the
  next open image after "done" is the next match.
* The filter is kept in the URL hash and in localStorage (`wv_filter`), so reloading or reopening the PWA keeps it.
* On a phone the panel is a full-width bottom sheet with large chips.
* Offline the app filters the images it already has cached (approximation from local labels); online it asks the server.

## Server
* `GET /api/filter?q=device_type:manual&limit=500&counts=true` -> `{total, count, ids, counts, match_counts, filter}`
  over every served image.
* `POST /api/filter` `{"q": "...", "ids": [...], "limit": 20000, "counts": true}`: same, restricted to `ids`.
* `GET /api/filter/keys`: keys, values, bases and special values (for UIs).
Read-only: filters never write events.

## CLI: `scripts/filter_images.py`
```powershell
.venv\Scripts\python.exe scripts\filter_images.py --filter "device_type:manual;chair_subtype:_unlabeled"          # ids
.venv\Scripts\python.exe scripts\filter_images.py --filter "weather:rain" --format csv --out var\rain.csv
.venv\Scripts\python.exe scripts\filter_images.py --filter "ramp_deploy:unsafe" --format manifest --out var\ramp_unsafe.csv
.venv\Scripts\python.exe scripts\filter_images.py --filter "device_type:manual" --counts                          # facet counts
.venv\Scripts\python.exe scripts\filter_images.py --keys
```
`--format ids|csv|manifest|json`. Synthetic rows are excluded unless `--include-synthetic`. The DB is opened read-only;
`--out` refuses `derived/`, `release/`, `results/` and `annotations/`. Same environment variables as the server
(`WV_DB`, `WV_CANDIDATES`, `WV_AUTO_DIR`, ...).
