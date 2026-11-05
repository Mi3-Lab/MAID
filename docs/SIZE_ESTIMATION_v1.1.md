# Wheelchair size estimation (v1.1) — design and assumptions

**Status: unvalidated.** No real measurements of any pictured chair exist yet, so nobody knows how accurate these
estimates are. Don't quote any error figure until `scripts/size_estimate.py` reports one from real measurements.

## Why "rough"
A single web photo has no metric scale: the camera distance, focal length and chair angle are unknown. A box in pixels
can't become centimetres without some outside assumption. So the estimator combines **population priors** (typical
chair dimensions) with **ratios** from the human-drawn box. Its point values mostly reflect the prior and should be read
together with their `low`–`high` range.

## Priors (cm: central, low, high) — `scripts/size_estimators.py`, version `rough-prior-v1`
| device_type | height (floor → top of backrest/handles) | width (overall) | length (overall) |
|---|---|---|---|
| manual | 92 (80–102) | 63 (55–72) | 105 (90–120) |
| powered | 102 (88–120) | 64 (56–75) | 115 (100–130) |
| other_aid | 95 (75–125) | 58 (45–72) | 110 (60–145) |

Further assumptions: a manual rear wheel is about 61 cm (56–66) in diameter (24-inch-class wheels). A seated occupant's
head top is about 128 cm (115–145) above the floor. Plausible bounds are width 40–100 cm and length 55–170 cm.
These are typical-adult-chair ranges chosen by hand, not a measured survey. Bariatric, pediatric and sports chairs fall
outside them. (Child images are excluded from MAID anyway.)

## Method (viewpoint-aware)
1. **Height** = the prior. Exception: a manual chair in side or three-quarter view with a human-drawn rear-wheel box,
   where scale = 61 cm / wheel box height in px, applied to the chair box.
2. **Front/rear view:** width = prior height × (box width px / box height px). **Side view:** the same for length.
3. **Occupied chair, box_covers=auto:** the box may include the occupant, so two hypotheses are combined (box = chair
   only, or box = chair + occupant up to ~128 cm). This gives a wider range and flags `box_may_include_occupant`.
4. **Three-quarter or unknown view, box truncated by the image edge, or heavy occlusion:** prior only (flagged).
5. **Box-derived value outside the plausible bounds:** prior kept and `implausible_box_<dim>_<N>cm_prior_kept` flagged.
6. Every estimate is flagged `unvalidated`. No estimate is made for `ambiguous` / `unclear` devices.

## Validation
`size_estimate.py` compares estimates with `data/measurements.csv` rows whose `measurement_source` is a real source
(user_measured, manufacturer_spec, depth_sensor, reference_object), on single-box images only. It reports MAE, max error
and how often the true value falls inside low–high, with a small-n warning. It writes `size_validation.json`. With no
real rows the status is `unvalidated`. To validate: photograph your own chairs, measure them with a tape
(configuration noted), and import with `--chair-ref/--height-cm/--width-cm --measurement-source user_measured`.

## Pluggable estimators (for metric estimates later)
`Estimator.applicable(ctx)` / `.estimate(ctx) -> Estimate`, tried in order:
* `ReferenceObjectEstimator`: experimental. Uses an object of known size in the same plane (`ctx.reference`).
* `CalibratedCameraEstimator`: stub. Needs intrinsics + extrinsics (e.g. a fixed in-vehicle camera).
* `DepthEstimator`: stub. Needs a metric depth map (stereo / ToF / LiDAR).
* `RoughPriorEstimator`: the fallback described above.

Stubs raise `NotImplementedError` and are skipped. No accuracy is claimed for any of them.
