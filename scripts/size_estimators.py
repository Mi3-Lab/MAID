"""Wheelchair size from a single photo: non-metric proxies, a ROUGH prior-based metric estimate, and pluggable estimators.

A monocular web photo has no metric scale. Everything here is one of three clearly separated kinds:

  1. proxies (non_metric)   box size / aspect ratio in image units. No centimetres. Safe to use as model features.
  2. rough_estimate         centimetres derived from ASSUMED typical dimensions (priors below) + the box aspect ratio.
                            Every value comes with a [low, high] range and measurement_source='rough_estimate'.
                            UNVALIDATED until compared with real measurements (scripts/size_estimate.py --validate).
                            Never copied into the measured columns (height_cm / width_cm / length_cm).
  3. measured               data/measurements.csv, real sources only (user_measured, manufacturer_spec, depth_sensor,
                            reference_object). Not produced by this module.

Estimator interface: an Estimator gets a ChairContext (box, image size, v1.0 labels, optional extras such as wheel boxes,
a reference object, camera intrinsics or a depth map) and returns an Estimate or None. Add a new method by subclassing
Estimator and registering it in ESTIMATORS; scripts/size_estimate.py runs them in order of preference.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional

VERSION = "rough-prior-v1"

# ---------------------------------------------------------------- assumptions (all in cm; central, low, high)
# Typical adult equipment, overall dimensions, unfolded, footrests on. Ranges are deliberately wide: they are meant to
# contain most standard adult chairs, not to describe any one product. Sources of the orders of magnitude: common
# manufacturer spec sheets (manual chairs with 16-20 in seats are ~55-70 cm wide overall; 24 in rear wheels), and
# in-vehicle securement guidance (WC19 / ISO 7176-19 style "reference" chairs) -- NOT fitted to MAID data.
PRIORS = {
    "manual":    {"height": (92, 80, 102),  "width": (63, 55, 72),  "length": (105, 90, 120)},
    "powered":   {"height": (102, 88, 120), "width": (64, 56, 75),  "length": (115, 100, 130)},
    "other_aid": {"height": (95, 75, 125),  "width": (58, 45, 72),  "length": (110, 60, 145)},   # scooters + rollators: very broad
}
REAR_WHEEL_DIAMETER = (61.0, 56.0, 66.0)           # manual chair rear wheel incl. tyre (22-26 in), only if a wheel box exists
SEATED_OCCUPANT_TOP = (128, 115, 145)              # floor to top of an adult occupant's head when seated in a wheelchair
FIT_MARGIN = 0.10                                  # box-fit / perspective slack applied to image-derived ratios (+-10 %)
BOUNDS = {"width": (40, 100), "length": (55, 170)}  # outside these an image-derived value is treated as implausible (prior kept)

@dataclass
class ChairContext:
    image_id: str
    box: list                       # [x, y, w, h] in 0..1 image coordinates (device box)
    img_w: int
    img_h: int
    device_type: str = ""
    viewpoint: str = ""             # front / rear / side / three-quarter
    occupied: str = ""              # yes / no
    occlusion: str = ""
    box_source: str = ""            # human:<annotator> | machine:<model_tag>
    box_covers: str = "auto"        # device | device_plus_occupant | auto (unknown: both hypotheses)
    wheel_boxes: list = field(default_factory=list)   # optional [[x,y,w,h], ...] of rear wheels (none collected yet)
    reference: Optional[dict] = None  # optional {"box": [...], "size_cm": float, "axis": "h"|"w"} (ReferenceObjectEstimator)
    camera: Optional[dict] = None     # optional intrinsics / extrinsics (CalibratedCameraEstimator)
    depth: Optional[object] = None    # optional metric depth map (DepthEstimator)

    @property
    def box_px(self):
        x, y, w, h = self.box
        return w * self.img_w, h * self.img_h

    @property
    def truncated(self):
        x, y, w, h = self.box; e = 0.01
        return x <= e or y <= e or x + w >= 1 - e or y + h >= 1 - e

@dataclass
class Estimate:
    method: str
    measurement_source: str
    height_cm: Optional[float] = None; height_low: Optional[float] = None; height_high: Optional[float] = None
    width_cm: Optional[float] = None;  width_low: Optional[float] = None;  width_high: Optional[float] = None
    length_cm: Optional[float] = None; length_low: Optional[float] = None; length_high: Optional[float] = None
    basis: str = ""                 # how each value was obtained (prior_only / aspect_from_box / wheel_scale ...)
    flags: list = field(default_factory=list)
    validated: str = "no"
    def row(self):
        d = asdict(self); d["flags"] = ";".join(self.flags); return d

def proxies(ctx: ChairContext):
    """Non-metric box features (image units). viewpoint-aware: says which physical extent the box width mostly reflects."""
    x, y, w, h = ctx.box; pw, ph = ctx.box_px
    extent = {"front": "width", "rear": "width", "side": "length", "three-quarter": "mixed width+length"}.get(ctx.viewpoint, "unknown")
    return {"box_w_norm": round(w, 4), "box_h_norm": round(h, 4), "box_area_norm": round(w * h, 5),
            "aspect_hw_px": round(ph / pw, 4) if pw else None, "box_w_reflects": extent,
            "truncated": int(ctx.truncated), "non_metric": 1}

class Estimator:
    name = "base"; measurement_source = ""
    def applicable(self, ctx: ChairContext) -> bool: return False
    def estimate(self, ctx: ChairContext) -> Optional[Estimate]: raise NotImplementedError

class ReferenceObjectEstimator(Estimator):
    """STUB for future use: scale from an object of known size in the same plane (e.g. a 30 cm floor tile, a door,
    a ruler Samir places next to the chair). Needs ctx.reference; nothing in MAID provides one yet. No accuracy claim."""
    name = "reference_object"; measurement_source = "reference_object"
    def applicable(self, ctx): return bool(ctx.reference)
    def estimate(self, ctx):
        r = ctx.reference; rw, rh = r["box"][2] * ctx.img_w, r["box"][3] * ctx.img_h
        px = rh if r.get("axis", "h") == "h" else rw
        if px <= 0: return None
        s = r["size_cm"] / px; pw, ph = ctx.box_px
        e = Estimate(self.name, "reference_object", basis="reference_scale", flags=["experimental", "same_plane_assumed"])
        e.height_cm = round(ph * s, 1)
        if ctx.viewpoint in ("front", "rear"): e.width_cm = round(pw * s, 1)
        return e

class CalibratedCameraEstimator(Estimator):
    """STUB: metric size from known camera intrinsics + mounting height/tilt (e.g. a fixed in-vehicle or kerb-side
    camera) and the ground-contact line of the box. Not implemented: web photos have no calibration."""
    name = "calibrated_camera"; measurement_source = "depth_sensor"
    def applicable(self, ctx): return bool(ctx.camera)
    def estimate(self, ctx): raise NotImplementedError("needs camera intrinsics + extrinsics; see docs/SIZE_ESTIMATION_v1.1.md")

class DepthEstimator(Estimator):
    """STUB: metric size from a depth map (RGB-D sensor in a vehicle, or a metric monocular-depth model -- the latter
    would itself be an estimate). Not implemented."""
    name = "depth"; measurement_source = "depth_sensor"
    def applicable(self, ctx): return ctx.depth is not None
    def estimate(self, ctx): raise NotImplementedError("needs a metric depth map; see docs/SIZE_ESTIMATION_v1.1.md")

def _scale_range(c, lo, hi, ratio):
    return round(c * ratio, 1), round(lo * ratio * (1 - FIT_MARGIN), 1), round(hi * ratio * (1 + FIT_MARGIN), 1)

class RoughPriorEstimator(Estimator):
    """ROUGH metric estimate from typical dimensions (PRIORS) applied to the device box. UNVALIDATED.

    height : the device-type prior (a photo without scale cannot refine it), unless rear-wheel boxes are given for a
             manual chair (scale = 61 cm / wheel box height).
    width  : front/rear views -> box-height prior x box aspect ratio (w_px / h_px); if the box may include a seated
             occupant (occupied=yes, box_covers auto), the box-height prior spans BOTH hypotheses (chair top .. head top),
             which widens the range. side / three-quarter / unknown views -> prior only.
    length : side views -> same construction as width; otherwise prior only.
    No estimate for ambiguous/unclear device types. Truncated or heavily occluded boxes -> prior only (flagged)."""
    name = "rough_prior"; measurement_source = "rough_estimate"
    def applicable(self, ctx): return ctx.device_type in PRIORS
    def estimate(self, ctx):
        P = PRIORS[ctx.device_type]; e = Estimate(self.name, "rough_estimate", flags=["unvalidated"])
        e.height_cm, e.height_low, e.height_high = P["height"]
        e.width_cm, e.width_low, e.width_high = P["width"]
        e.length_cm, e.length_low, e.length_high = P["length"]
        basis = {"height": "prior", "width": "prior", "length": "prior"}
        pw, ph = ctx.box_px
        usable = pw > 0 and ph > 0 and not ctx.truncated and ctx.occlusion != "heavy"
        if not usable: e.flags.append("truncated_or_occluded_prior_only" if pw and ph else "no_box")
        # scale: cm per pixel
        s = None
        if usable and ctx.device_type == "manual" and ctx.wheel_boxes and ctx.viewpoint in ("side", "three-quarter"):
            wh = max(b[3] for b in ctx.wheel_boxes) * ctx.img_h
            if wh > 0:
                c, lo, hi = REAR_WHEEL_DIAMETER; s = (c / wh, lo / wh, hi / wh)
                e.height_cm, e.height_low, e.height_high = round(ph * s[0], 1), round(ph * s[1] * (1 - FIT_MARGIN), 1), round(ph * s[2] * (1 + FIT_MARGIN), 1)
                basis["height"] = "wheel_scale"
        if usable and s is None:
            covers = ctx.box_covers
            if covers == "auto": covers = "either" if ctx.occupied == "yes" else "device"
            hc, hlo, hhi = P["height"]
            if covers == "device_plus_occupant": hc, hlo, hhi = SEATED_OCCUPANT_TOP
            elif covers == "either":
                hc, hlo, hhi = (P["height"][0] + SEATED_OCCUPANT_TOP[0]) / 2, P["height"][1], SEATED_OCCUPANT_TOP[2]
                e.flags.append("box_may_include_occupant")
            s = (hc / ph, hlo / ph, hhi / ph)
        if usable and s is not None:
            target = {"front": "width", "rear": "width", "side": "length"}.get(ctx.viewpoint)
            if target:
                c, lo, hi = pw * s[0], pw * s[1] * (1 - FIT_MARGIN), pw * s[2] * (1 + FIT_MARGIN)
                if not (BOUNDS[target][0] <= c <= BOUNDS[target][1]):
                    e.flags.append(f"implausible_box_{target}_{round(c)}cm_prior_kept")
                else:
                    setattr(e, target + "_cm", round(c, 1)); setattr(e, target + "_low", round(lo, 1)); setattr(e, target + "_high", round(hi, 1))
                    basis[target] = "aspect_from_box" if basis["height"] == "prior" else "wheel_scale"
            elif ctx.viewpoint == "three-quarter": e.flags.append("three_quarter_view_width_prior_only")
            else: e.flags.append("unknown_viewpoint_prior_only")
        e.basis = ";".join(f"{k}={v}" for k, v in basis.items())
        return e

ESTIMATORS = [ReferenceObjectEstimator(), CalibratedCameraEstimator(), DepthEstimator(), RoughPriorEstimator()]

def best_estimate(ctx: ChairContext):
    """First applicable estimator that returns a value (stubs that are not implemented are skipped)."""
    for est in ESTIMATORS:
        if not est.applicable(ctx): continue
        try:
            r = est.estimate(ctx)
            if r is not None: return r
        except NotImplementedError:
            continue
    return None
