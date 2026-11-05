# Ramp deployment labels (MAID v1.1): research judgement, NOT validated for deployment

> These labels record what a human annotator judges from a **single photo**. They are a research signal for studying
> scene understanding around wheelchair-accessible vehicles. They are **not** a safety assessment, have not been
> validated against real deployments, and must **not** be used to decide whether a real ramp is deployed.

## Fields
| field | values |
|---|---|
| `ramp_deploy` | `safe`: from this view there appears to be room, a suitable surface and a waiting wheelchair user. `unsafe`: something visible would make deploying a ramp here a bad idea. `not_needed`: no reason to deploy (e.g. no wheelchair user waiting). `unknown`: can't tell from this photo. |
| `ramp_reasons` (multi-select, shown for unsafe / not_needed) | `insufficient_space` (no room for the ramp's footprint / turning space), `obstruction` (pole, bin, bike, parked car, people in the way), `uneven_surface`, `steep_slope`, `curb_height_issue`, `wet_icy_surface`, `traffic_hazard` (ramp would land in moving traffic), `no_wheelchair_user` (nobody using a wheelchair is there), `user_not_waiting_unclear` (a user is there but it's unclear they're waiting for a vehicle), `unknown` |

Rules
* Judge only what's visible. Imagine a side-deploying vehicle ramp at the kerb next to the chair, ~1.5 m deep.
* Pick every reason that applies (`unknown` alone if none fits). Reasons are stored as a JSON list; CSVs use sorted values
  joined with `+` (`insufficient_space+obstruction`).
* Indoor photos: usually `not_needed` + `no_wheelchair_user` / `user_not_waiting_unclear`, or `unknown`.
* Consensus works like the other v1.1 fields: 2 annotators, 3rd on disagreement; multi-select values must match exactly.

## Where it's wired in
* Schema: `server/schema.json` (`ramp_deploy`, `ramp_reasons` with `multi`, `show_when`).
* App: Attributes v1.1 pass (press `v`); number keys toggle reasons; filterable (`ramp_deploy:unsafe;ramp_reasons:obstruction`).
* Derived: `derived/labels_v11.csv`, `kappa_v11.json`, `data/manifest_v11.csv` (`ramp_deploy`, `ramp_reasons`).
* Machine hints (orange, never labels): SigLIP `attr-v2` scores `ramp_deploy` and each reason separately (positive vs
  negative prompt); a VLM can answer `ramp_deploy` (`docs/ANNOTATION_MODELS_v1.1.md`).
* Training: `scripts/train_multitask.py` heads `ramp_deploy` and one yes/no head per reason (`ramp_reasons=<reason>`);
  evaluated on real val/test images only.
* Synthetic images never inherit a ramp judgement (the scene changed), so their ramp fields are empty until a human
  labels them.

## Limits
Single images, no depth, no vehicle model, web photo bias, and annotators are not ramp operators. Treat any metric on
these labels as agreement with annotators, not safety performance.
