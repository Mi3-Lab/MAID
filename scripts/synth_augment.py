"""Synthetic augmentation (MAID v1.1): cut a wheelchair out of a TRAIN image and composite it into licensed scene
backgrounds (curbside, crosswalk, sidewalk, ramp/lift, vehicle door) with weather / lighting variants.

  python scripts/synth_augment.py --check-backgrounds                 # validate data/backgrounds/backgrounds.csv only
  python scripts/synth_augment.py --n 200 --seed 0                    # -> var/synthetic/images/*.jpg + synthetic_manifest.csv
  python scripts/synth_augment.py --n 50 --empty-only --variants rain,night --scenes curbside,vehicle_door
  python scripts/check_leakage.py --synthetic var/synthetic/synthetic_manifest.csv
  python scripts/train_multitask.py --synthetic var/synthetic/synthetic_manifest.csv --run mt_syn

Rules (enforced here and again by check_leakage.py --synthetic):
  * Sources are TRAIN images only (frozen split from data/manifest_v11.csv, else data/manifest_v2.csv), never
    excluded / child-flagged rows, always the face-blurred copy. Every output is split=train with its source's leak_group.
  * Backgrounds come only from data/backgrounds/backgrounds.csv (each row: background_id, file, scene, source_url,
    license, author, people=none, ground_y [, environment]) -- or, with dataset_image_id, an approved TRAIN image of this
    dataset. Rows without license info, with people != none, or pointing at a non-train image are refused.
    Nothing is downloaded or generated: no photos of people are taken, generated or fetched by this script.
  * Outputs are tagged synthetic=true with full provenance and live in var/synthetic/ (gitignored, like all images).
    Synthetic images are train-only and never enter val/test metrics.
Compositing: cutout from a SAM 2 mask (var/assist/*/masks.jsonl or --sam), else a GrabCut / feathered-ellipse mask
inside the human box; scale from a ground-plane model (object height in px = obj_h / camera_h x (foot_y - horizon_y),
horizon = ground_y); a soft contact shadow; mean/std colour matching to the background; then a weather variant.
Inherited labels: chair attributes of the source (device_type, chair_subtype, drive_position, size_class, control,
occupied, propulsion). Scene labels come from the background (environment) and variant (weather, ground_wet);
ramp_deploy / ramp_reasons are left empty -- that judgement needs a human.
"""
import argparse, csv, json, os, random, re, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server")); sys.path.insert(0, str(ROOT / "scripts"))
import wvcore

SCENES = {"curbside": "curbside", "crossing": "crosswalk_road", "crosswalk": "crosswalk_road", "sidewalk": "sidewalk",
          "ramp_lift": "ramp_lift", "vehicle_door": "curbside", "parking_lot": "parking_lot"}
VARIANTS = ["none", "overcast", "rain", "wet", "fog", "night", "dusk"]
WEATHER_OF = {"none": "", "overcast": "overcast", "rain": "rain", "wet": "overcast", "fog": "fog", "night": "night", "dusk": ""}
WET_OF = {"rain": "yes", "wet": "yes"}
CHAIR_COLS = ["device_type", "chair_subtype", "drive_position", "control", "occupied", "propulsion"]
CHAIRS = {"manual", "powered"}
OBJ_H_CM = {"manual": 95.0, "powered": 110.0}
BG_COLS = ["background_id", "file", "scene", "source_url", "license", "author", "people", "ground_y"]
OUT_COLS = ["image_id", "file", "synthetic", "split", "leak_group", "source_image_id", "source_split", "background_id",
            "background_source", "background_license", "background_author", "background_url", "scene",
            "weather_variant", "lighting_variant", "mask_source", "box_source", "transform_params",
            "generator", "generated_at"] + [c for c in CHAIR_COLS]
FIRST_ID = 800001

def default_manifest():
    for p in (ROOT / "data" / "manifest_v11.csv", ROOT / "data" / "manifest_v2.csv"):
        if p.exists(): return p
    return ROOT / "data" / "manifest_v11.csv"

def read_csv(p): return list(csv.DictReader(open(p, encoding="utf-8"))) if p and Path(p).exists() else []

def eligible_source(r, empty_only=False):
    """Train rows usable as cutout sources: train split, trainable, not excluded / child-flagged, a wheelchair class."""
    if r.get("split") != "train": return False
    if not ("1" in (r.get("train_eligible"), r.get("train_eligible_v10"), r.get("v11_eligible"))): return False
    why = " ".join(r.get(k) or "" for k in ("exclusion_reason", "v11_exclusion_reason"))
    if "child" in why or why.strip() not in ("", "none"): return False
    if wvcore.canon_class(r.get("device_type", "")) not in CHAIRS: return False
    return not empty_only or r.get("occupied") == "no"

def check_backgrounds(bg_csv, manifest_rows):
    """(usable rows, problems). Dataset backgrounds must be train images; files must exist in the backgrounds folder."""
    rows, probs = [], []
    split = {r["image_id"]: r for r in manifest_rows}
    if not Path(bg_csv).exists(): return [], [f"{bg_csv} missing (see docs/SYNTHETIC_AUGMENTATION_v1.1.md)"]
    for r in read_csv(bg_csv):
        bid = r.get("background_id") or "?"; did = (r.get("dataset_image_id") or "").strip()
        if did:
            m = split.get(did)
            if not m or m.get("split") != "train": probs.append(f"{bid}: dataset_image_id {did} is not a train image"); continue
            r = {**r, "license": r.get("license") or m.get("license", ""), "author": r.get("author") or m.get("creator", ""),
                 "source_url": r.get("source_url") or m.get("source_url", "") or f"dataset:{did}"}
        miss = [k for k in ("license", "author", "source_url") if not (r.get(k) or "").strip()]
        if miss: probs.append(f"{bid}: missing {', '.join(miss)}"); continue
        if (r.get("people") or "").strip().lower() != "none": probs.append(f"{bid}: people must be 'none' (no photos of people as backgrounds)"); continue
        if r.get("scene") not in SCENES: probs.append(f"{bid}: scene {r.get('scene')!r} not in {sorted(SCENES)}"); continue
        try: gy = float(r.get("ground_y") or 0.55); assert 0.05 < gy < 0.95
        except Exception: probs.append(f"{bid}: ground_y must be a fraction of the height (horizon line), e.g. 0.5"); continue
        p = (Path(bg_csv).parent / r["file"]) if r.get("file") else None
        if not did and (not p or not p.exists()): probs.append(f"{bid}: file {r.get('file')!r} not found"); continue
        rows.append({**r, "ground_y": gy, "_path": str(p) if p and p.exists() else "", "_dataset": did})
    return rows, probs

def human_box(r, per):
    """Largest human box [x0,y0,x1,y1] (normalised) from the manifest bboxes column or, failing that, events (read-only)."""
    bx = []
    try: bx = [b for v in json.loads(r.get("bboxes") or "{}").values() for b in v]
    except Exception: pass
    if not bx and per:
        from size_estimate import human_boxes
        bx, _ = human_boxes(per, r["image_id"])
    bx = [b for b in bx if len(b) == 4 and b[2] > b[0] and b[3] > b[1]]
    return max(bx, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if bx else None

def assist_masks(assist_dir):
    out = {}
    for p in sorted(Path(assist_dir).glob("*/masks.jsonl")):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line); out.setdefault(r["image_id"], (p.parent / r["mask_file"], r["model"].get("tag", p.parent.name)))
    return out

# ---------------------------------------------------------------- image ops (PIL + numpy)
def ellipse_mask(w, h):
    import numpy as np
    y, x = np.mgrid[0:h, 0:w]; return (((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2 <= 1.0).astype("float32")

def cut_mask(crop, mask_img=None):
    """Alpha in [0,1] for the crop: given mask, else GrabCut (if OpenCV is installed), else a feathered ellipse."""
    import numpy as np
    from PIL import ImageFilter, Image
    w, h = crop.size
    if mask_img is not None: m = np.asarray(mask_img.convert("L").resize((w, h))).astype("float32") / 255; src = "mask"
    else:
        try:
            import cv2
            a = np.asarray(crop)[:, :, ::-1].copy(); gm = np.zeros(a.shape[:2], np.uint8); bg, fg = np.zeros((1, 65)), np.zeros((1, 65))
            cv2.grabCut(a, gm, (2, 2, max(3, w - 4), max(3, h - 4)), bg, fg, 4, cv2.GC_INIT_WITH_RECT)
            m = np.isin(gm, (1, 3)).astype("float32"); src = "grabcut"
            if m.mean() < 0.05: raise ValueError("empty grabcut")
        except Exception: m = ellipse_mask(w, h); src = "ellipse"
    a = Image.fromarray((m * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(max(1, min(w, h) // 120)))
    return np.asarray(a).astype("float32") / 255, src

def color_match(obj, alpha, region, strength=0.6):
    import numpy as np
    o = obj.astype("float32"); r = region.reshape(-1, 3).astype("float32"); m = alpha.reshape(-1) > 0.5
    if m.sum() < 10: return obj
    om, os_ = o.reshape(-1, 3)[m].mean(0), o.reshape(-1, 3)[m].std(0) + 1e-3
    rm, rs = r.mean(0), r.std(0) + 1e-3
    tm, ts = om + strength * (rm - om), os_ + strength * (rs - os_)
    return np.clip((o - om) / os_ * ts + tm, 0, 255)

def variant(img, v, light, rng):
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter
    a = np.asarray(img).astype("float32"); H, W = a.shape[:2]
    if v == "overcast": a = a * 0.85 + a.mean(2, keepdims=True) * 0.15; a = (a - a.mean()) * 0.8 + a.mean()
    if v in ("rain", "wet"):
        a *= 0.8; a[..., 2] *= 1.06
        g = int(H * 0.6); a[g:] = a[g:] * 0.85 + np.flipud(a[max(0, 2 * g - H):g])[: H - g] * 0.15   # wet ground: faint reflection
    if v == "rain":
        st = np.zeros((H, W), "float32")
        for _ in range(int(W * H / 900)):
            x, y, L = rng.randrange(W), rng.randrange(H), rng.randrange(8, 24)
            st[y:min(H, y + L), x] = 1
        st = np.asarray(Image.fromarray((st * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(0.8))).astype("float32") / 255
        a = a * (1 - 0.25 * st[..., None]) + 220 * 0.25 * st[..., None]
    if v == "fog":
        depth = np.linspace(1.0, 0.35, H)[:, None, None]; a = a * (1 - 0.55 * depth) + 200 * 0.55 * depth
    if v == "night":
        a = 255 * (a / 255) ** 1.9 * 0.55; a[..., 2] *= 1.15
        y, x = np.mgrid[0:H, 0:W]; vig = 1 - 0.5 * (((x - W / 2) / W) ** 2 + ((y - H / 2) / H) ** 2) * 2; a *= vig[..., None]
    out = Image.fromarray(np.clip(a, 0, 255).astype("uint8"))
    if light == "dusk": out = ImageEnhance.Brightness(ImageEnhance.Color(out).enhance(1.2)).enhance(0.8)
    if light == "bright": out = ImageEnhance.Brightness(out).enhance(1.15)
    return out

def composite(bg, crop, alpha, box_h_cm, ground_y, rng, cam_h_cm=150.0, shadow=0.45):
    """Paste crop (RGB) with alpha onto bg at a sampled foot position; returns (image, params)."""
    import numpy as np
    from PIL import Image, ImageFilter
    W, H = bg.size; hz = ground_y * H
    foot = rng.uniform(hz + 0.35 * (H - hz), H * 0.97)
    h = max(16, int(box_h_cm / cam_h_cm * (foot - hz)))
    w = max(8, int(h * crop.size[0] / crop.size[1]))
    if w > 0.9 * W: w = int(0.9 * W); h = int(w * crop.size[1] / crop.size[0])
    x0 = int(rng.uniform(0.05 * W, max(0.05 * W + 1, 0.95 * W - w))); y0 = int(foot - h)
    c = crop.resize((w, h), Image.LANCZOS); al = np.asarray(Image.fromarray((alpha * 255).astype("uint8")).resize((w, h), Image.LANCZOS)).astype("float32") / 255
    B = np.asarray(bg).astype("float32")
    ry0, ry1, rx0, rx1 = max(0, y0), min(H, y0 + h), max(0, x0), min(W, x0 + w)
    obj = color_match(np.asarray(c), al, B[ry0:ry1, rx0:rx1])
    # contact shadow: soft ellipse under the object
    sw, sh = int(w * 0.95), max(4, int(w * 0.14)); sm = Image.new("L", (W, H), 0)
    sm.paste(Image.fromarray((ellipse_mask(sw, sh) * 255 * shadow).astype("uint8")), (x0 + (w - sw) // 2, int(foot - sh * 0.6)))
    sm = np.asarray(sm.filter(ImageFilter.GaussianBlur(max(2, sh // 2)))).astype("float32")[..., None] / 255
    B = B * (1 - sm)
    sub = B[ry0:ry1, rx0:rx1]; oa = al[ry0 - y0:ry1 - y0, rx0 - x0:rx1 - x0, None]; oo = obj[ry0 - y0:ry1 - y0, rx0 - x0:rx1 - x0]
    B[ry0:ry1, rx0:rx1] = sub * (1 - oa) + oo * oa
    return Image.fromarray(np.clip(B, 0, 255).astype("uint8")), {"x0": x0, "y0": y0, "w": w, "h": h, "foot_y": round(foot, 1),
                                                                  "horizon_y": round(hz, 1), "obj_h_cm": box_h_cm, "camera_h_cm": cam_h_cm, "shadow": shadow}

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(default_manifest()))
    ap.add_argument("--images", default=os.environ.get("WV_BLUR_DIR", str(ROOT / "images_blurred")), help="face-blurred source images")
    ap.add_argument("--backgrounds", default=str(ROOT / "data" / "backgrounds" / "backgrounds.csv"))
    ap.add_argument("--out", default=str(ROOT / "var" / "synthetic"))
    ap.add_argument("--assist-dir", default=str(ROOT / "var" / "assist"))
    ap.add_argument("--db", default=os.environ.get("WV_DB", str(ROOT / "var" / "wheelview.db")))
    ap.add_argument("--events", default=str(ROOT / "annotations" / "events"))
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenes", default=",".join(sorted(set(SCENES) - {"crosswalk"})))
    ap.add_argument("--variants", default=",".join(VARIANTS)); ap.add_argument("--lighting", default="normal,dusk,bright")
    ap.add_argument("--empty-only", action="store_true", help="only unoccupied chairs (occupied=no) as cutouts")
    ap.add_argument("--sam", action="store_true", help="compute SAM 2 masks live (scripts/assist_models.py) when none is cached")
    ap.add_argument("--camera-height-cm", type=float, default=150.0)
    ap.add_argument("--check-backgrounds", action="store_true")
    a = ap.parse_args(argv)
    M = read_csv(a.manifest)
    if not M: sys.exit(f"{a.manifest} missing: build it first (scripts/build_manifest_v11.py)")
    bgs, probs = check_backgrounds(a.backgrounds, M)
    for p in probs: print("background refused:", p)
    print(f"{len(bgs)} usable background(s)")
    if a.check_backgrounds: return 1 if probs else 0
    scenes = set(a.scenes.split(",")); bgs = [b for b in bgs if b["scene"] in scenes]
    if not bgs: sys.exit("no usable backgrounds for the chosen scenes -> nothing generated (see docs/SYNTHETIC_AUGMENTATION_v1.1.md)")
    out = Path(a.out); (out / "images").mkdir(parents=True, exist_ok=True)
    gi = out / ".gitignore"
    if not gi.exists(): gi.write_text("*\n")   # belt and braces: never commit synthetic images
    per = None
    if not any(r.get("bboxes") for r in M):
        from size_estimate import load_events
        per, _ = wvcore.fold(load_events(a.db, a.events))
    masks = assist_masks(a.assist_dir)
    src = []
    for r in M:
        if not eligible_source(r, a.empty_only): continue
        p = Path(a.images) / f"{r['image_id']}.jpg"
        if not p.exists(): continue                    # face-blurred copy required
        b = human_box(r, per)
        if b is None and r["image_id"] not in masks: continue
        src.append((r, p, b))
    print(f"{len(src)} eligible train source image(s) with a box or mask")
    if not src: sys.exit("no eligible sources (train split, wheelchair, blurred copy, human box or assist mask)")
    real = {r["image_id"] for r in M}
    mf = out / "synthetic_manifest.csv"; old = read_csv(mf)
    nums = [int(m.group(1)) for r in old for m in [re.match(r"wv_(\d+)$", r["image_id"])] if m]
    nxt = max(nums + [FIRST_ID - 1]) + 1
    seg = None
    if a.sam:
        import assist_models as AM
        seg = AM.get("segmenter")
        if seg is None: print("SAM 2 unavailable:", AM.status("segmenter")["reason"])
    from PIL import Image
    rng = random.Random(a.seed); variants = a.variants.split(","); lights = a.lighting.split(",")
    rows = list(old)
    for k in range(a.n):
        r, p, b = rng.choice(src); bg = rng.choice(bgs); v = rng.choice(variants); lt = rng.choice(lights)
        iid = f"wv_{nxt:06d}"; nxt += 1
        assert iid not in real, f"synthetic id {iid} collides with a real image id"
        im = Image.open(p).convert("RGB"); W, H = im.size
        mk, msrc = None, ""
        if r["image_id"] in masks and masks[r["image_id"]][0].exists():
            mp, tag = masks[r["image_id"]]; mk = Image.open(mp); msrc = f"assist:{tag}"
            if b is None:
                import numpy as np
                ys, xs = np.nonzero(np.asarray(mk.convert("L")) > 127)
                if not len(xs): continue
                b = [xs.min() / W, ys.min() / H, (xs.max() + 1) / W, (ys.max() + 1) / H]
        box = [int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H)]
        if box[2] - box[0] < 24 or box[3] - box[1] < 24: continue
        crop = im.crop(box)
        if mk is not None: mk = mk.convert("L").crop(box)
        elif seg is not None: mk = Image.fromarray((seg.segment(im, box) * 255).astype("uint8")).crop(box); msrc = f"sam:{seg.tag}"
        alpha, how = cut_mask(crop, mk); msrc = msrc or how
        bimg = Image.open(bg["_path"] or Path(a.images) / f"{bg['_dataset']}.jpg").convert("RGB")
        dt = wvcore.canon_class(r.get("device_type", ""))
        try: oh = float(r.get("est_height_cm") or r.get("height_cm") or OBJ_H_CM.get(dt, 100.0))
        except ValueError: oh = OBJ_H_CM.get(dt, 100.0)
        comp, prm = composite(bimg, crop, alpha, oh, bg["ground_y"], rng, a.camera_height_cm, shadow=0.25 if v in ("overcast", "fog", "night") else 0.45)
        comp = variant(comp, v, lt, rng)
        fn = f"images/{iid}.jpg"; comp.save(out / fn, quality=92)
        row = {"image_id": iid, "file": fn, "synthetic": "true", "split": "train", "leak_group": r.get("leak_group", ""),
               "source_image_id": r["image_id"], "source_split": r["split"], "background_id": bg["background_id"],
               "background_source": f"dataset:{bg['_dataset']}" if bg["_dataset"] else "backgrounds_folder",
               "background_license": bg["license"], "background_author": bg["author"], "background_url": bg["source_url"],
               "scene": bg["scene"],
               "weather_variant": v, "lighting_variant": lt, "mask_source": msrc, "box_source": "human" if b and not msrc.startswith("assist") else msrc,
               "transform_params": json.dumps({**prm, "src_box": box, "seed": a.seed, "k": k}, sort_keys=True),
               "generator": "synth_augment.py v1", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               **{c: r.get(c, "") for c in CHAIR_COLS}, "device_type": dt}
        rows.append(row)
    with open(mf, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, OUT_COLS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows) - len(old)} synthetic image(s) -> {out / 'images'}; manifest {mf} ({len(rows)} rows)")
    print("next: python scripts/check_leakage.py --synthetic", mf)
    return 0

if __name__ == "__main__":
    sys.exit(main())
