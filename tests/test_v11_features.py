#!/usr/bin/env python3
"""Tests for the v1.1 feature set: filters, chair subtypes, ramp labels, synthetic augmentation, annotation-assist models.
Heavy models are mocked; images are generated shapes/gradients in a temp dir (no photos, no people, no live data).

  python tests/test_v11_features.py
"""
import csv, json, os, sys, tempfile, uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="wv_v11f_"))
sys.path.insert(0, str(ROOT / "server")); sys.path.insert(0, str(ROOT / "scripts"))
results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond))); print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail else ""), flush=True)
def raises(fn):
    try: fn(); return False
    except (ValueError, AssertionError, SystemExit): return True

import wvcore
N = [0]
def ev(img, ann, field, value, kind="label"):
    N[0] += 1
    return {"uuid": str(uuid.uuid4()), "image_id": img, "annotator": ann, "field": field, "value": value, "kind": kind,
            "ts": f"2026-11-04T11:{N[0] // 60:02d}:{N[0] % 60:02d}.000Z", "device_id": "t-" + ann}
V10 = [f for f in wvcore.SCHEMA["fields"] if not f.get("since")]
def v10_done(img, ann, dev):
    return [ev(img, ann, f["name"], dev if f["name"] == "device_type" else f["options"][0]) for f in V10] + \
           [ev(img, ann, "exclude", ""), ev(img, ann, "bbox", [[0.2, 0.1, 0.6, 0.9]]), ev(img, ann, "_status", "done")]

# ---------------------------------------------------------------- 1. schema: subtypes, ramp, multi-select
F = {f["name"]: f for f in wvcore.SCHEMA["fields"]}
check("schema: chair_subtype, drive_position, ramp_deploy, ramp_reasons, securement_hardware are v1.1 fields",
      all(F.get(k, {}).get("since") == "1.1" for k in ("chair_subtype", "drive_position", "ramp_deploy", "ramp_reasons", "securement_hardware")))
check("schema v5 securement_hardware options",
      wvcore.FIELD_OPTS["securement_hardware"] == ["docking_bracket", "tiedown_points", "not_visible", "unclear"])
check("schema version is 5", wvcore.SCHEMA.get("version") == 5)
check("schema: scooters stay under other_aid (no scooter subtype)", not any("scooter" in o for o in wvcore.FIELD_OPTS["chair_subtype"]))
check("schema: pediatric is a size descriptor, not a subtype", "pediatric_size" in wvcore.FIELD_OPTS["size_class"]
      and not any("pediatric" in o for o in wvcore.FIELD_OPTS["chair_subtype"]))
check("schema: ramp_deploy options", wvcore.FIELD_OPTS["ramp_deploy"] == ["safe", "unsafe", "not_needed", "unknown"])
check("schema: ramp_reasons is multi-select", "ramp_reasons" in wvcore.MULTI_FIELDS)
check("options_for: manual list", "standard_folding" in wvcore.options_for("chair_subtype", "manual") and "power_standard" not in wvcore.options_for("chair_subtype", "manual"))
check("options_for: negative_camber is a manual subtype only",
      "negative_camber" in wvcore.options_for("chair_subtype", "manual")
      and "negative_camber" not in wvcore.options_for("chair_subtype", "powered")
      and "negative_camber" in wvcore.FIELD_OPTS["chair_subtype"])
check("validate: negative_camber accepted for chair_subtype",
      wvcore.validate(ev("wv_000001", "alice", "chair_subtype", "negative_camber"))[0] is not None)
check("options_for: powered list", "power_tilt_recline" in wvcore.options_for("chair_subtype", "powered") and "bariatric" not in wvcore.options_for("chair_subtype", "powered"))
check("dep_ok: subtype hidden for other_aid", not wvcore.dep_ok("chair_subtype", {"device_type": "other_aid"}))
check("dep_ok: subtype shown for manual (from context)", wvcore.dep_ok("chair_subtype", {}, {"device_type": "manual"}))
check("dep_ok: drive_position only for powered", not wvcore.dep_ok("drive_position", {"device_type": "manual"}) and wvcore.dep_ok("drive_position", {"device_type": "powered"}))
check("dep_ok: ramp reasons need unsafe/not_needed", not wvcore.dep_ok("ramp_reasons", {"ramp_deploy": "safe"}) and wvcore.dep_ok("ramp_reasons", {"ramp_deploy": "unsafe"}))
ok_e, _ = wvcore.validate(ev("wv_000001", "alice", "ramp_reasons", ["obstruction", "insufficient_space"]))
check("validate: multi list accepted", ok_e is not None)
check("validate: multi plain string refused", wvcore.validate(ev("wv_000001", "alice", "ramp_reasons", "obstruction"))[0] is None)
check("validate: 'unknown' not combinable", wvcore.validate(ev("wv_000001", "alice", "ramp_reasons", ["unknown", "obstruction"]))[0] is None)
check("validate: unknown reason refused", wvcore.validate(ev("wv_000001", "alice", "ramp_reasons", ["aliens"]))[0] is None)
check("hval: list -> sorted 'a+b'", wvcore.hval(["obstruction", "insufficient_space"]) == "insufficient_space+obstruction")

# ---------------------------------------------------------------- 2. filter parsing + matching
flt, basis = wvcore.parse_filter("device_type:manual,powered;weather:rain;basis:model")
check("parse_filter: OR within, AND across, basis", flt == {"device_type": {"manual", "powered"}, "weather": {"rain"}} and basis == "model")
check("format_filter round-trip", wvcore.parse_filter(wvcore.format_filter(flt, basis)) == (flt, basis))
check("parse_filter: unknown key refused", raises(lambda: wvcore.parse_filter("colour:red")))
check("parse_filter: unknown value refused", raises(lambda: wvcore.parse_filter("device_type:hovercraft")))
check("parse_filter: injection-looking value refused", raises(lambda: wvcore.parse_filter("device_type:<script>")))
E10 = v10_done("wv_000001", "alice", "manual") + v10_done("wv_000001", "bob", "manual") + v10_done("wv_000002", "alice", "powered") \
    + v10_done("wv_000002", "bob", "powered")
per, adjv = wvcore.fold([wvcore.validate(e)[0] for e in E10])
ids = ["wv_000001", "wv_000002", "wv_000003"]
sugg = {"wv_000003": {"device_type": {"value": "manual", "confidence": 0.9}}}
facts = {i: wvcore.image_facts(i, per, adjv, sugg=sugg.get(i)) for i in ids}
r = wvcore.run_filter(ids, facts, wvcore.parse_filter("device_type:manual")[0])
check("run_filter: human basis", r["ids"] == ["wv_000001"] and r["total"] == 3, json.dumps(r["ids"]))
r = wvcore.run_filter(ids, facts, wvcore.parse_filter("device_type:manual")[0], basis="model")
check("run_filter: model basis sees only suggestions", r["ids"] == ["wv_000003"], json.dumps(r["ids"]))
r = wvcore.run_filter(ids, facts, wvcore.parse_filter("device_type:manual")[0], basis="either")
check("run_filter: either basis", sorted(r["ids"]) == ["wv_000001", "wv_000003"], json.dumps(r["ids"]))
check("run_filter: facet counts", r["counts"] and "device_type" in r["counts"])

# ---------------------------------------------------------------- 3. server /api/filter (scratch env)
CC = "candidate_id,source,creator,license,needs_visual_screen".split(",")
with open(TMP / "candidates.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, CC, extrasaction="ignore"); w.writeheader()
    for k, i in enumerate(ids): w.writerow({"candidate_id": i, "source": "commons", "creator": f"p{k}", "license": "CC-BY-4.0", "needs_visual_screen": "no"})
for d in ("auto", "block", "images", "images_blurred"): (TMP / d).mkdir()
(TMP / "blur.jsonl").write_text("")
env = {"WV_DB": str(TMP / "scratch.db"), "WV_CANDIDATES": str(TMP / "candidates.csv"), "WV_CONTRIBUTED": str(TMP / "contributed.csv"),
       "WV_IMAGE_DIR": str(TMP / "images"), "WV_BLUR_DIR": str(TMP / "images_blurred"), "WV_EXPORT_DIR": str(TMP / "exports"),
       "WV_AUTO_DIR": str(TMP / "auto"), "WV_BLOCKLIST_DIR": str(TMP / "block"), "WV_IMAGE_PROXY": "0", "WV_ACCESS_CODE": "",
       "WV_BLUR_MANIFESTS": str(TMP / "blur.jsonl"), "WV_CACHE_DIR": str(TMP / "cache"), "WV_ASSIGN_WITHOUT_LOCAL_IMAGE": "1"}
os.environ.pop("WV_HF_DATASET", None); os.environ.update(env)
try:
    from fastapi.testclient import TestClient
    import main
    with TestClient(main.app) as cl:
        check("server: events accepted", cl.post("/api/events", json={"events": E10}).status_code == 200)
        k = cl.get("/api/filter/keys").json()
        check("server: /api/filter/keys lists subtype + ramp keys", "chair_subtype" in k["keys"] and "ramp_reasons" in k["keys"] and "synthetic" in k["pseudo"])
        g = cl.get("/api/filter", params={"q": "device_type:powered"}).json()
        check("server: GET /api/filter", "wv_000002" in g["ids"] and "wv_000001" not in g["ids"], json.dumps(g)[:200])
        p = cl.post("/api/filter", json={"q": "device_type:manual", "ids": ["wv_000002", "wv_000001"]}).json()
        check("server: POST /api/filter restricted to ids", p["ids"] == ["wv_000001"] and p["total"] == 2, json.dumps(p)[:200])
        check("server: bad filter -> 400", cl.get("/api/filter", params={"q": "nope:x"}).status_code == 400)
        check("server: bad ids -> 400", cl.post("/api/filter", json={"q": "", "ids": "x"}).status_code == 400)
        h = cl.get("/api/health").json()
        check("server: health advertises filter + multi fields", "filter" in h["features"] and "multi_fields" in h["features"])
        st = cl.get("/api/assist/status").json()
        check("server: /api/assist/status reports every kind without loading", all(x in st for x in ("detector", "segmenter", "vlm")), json.dumps(st)[:200])
except ImportError as e:
    check("server tests (fastapi missing)", False, str(e))

# ---------------------------------------------------------------- 4. CLI filter
import subprocess
cp = subprocess.run([sys.executable, str(ROOT / "scripts" / "filter_images.py"), "--filter", "device_type:manual", "--db", env["WV_DB"],
                     "--events", str(TMP / "noevents"), "--candidates", env["WV_CANDIDATES"], "--contributed", env["WV_CONTRIBUTED"],
                     "--auto-dir", env["WV_AUTO_DIR"], "--blocklist-dir", env["WV_BLOCKLIST_DIR"]], capture_output=True, text=True)
check("filter_images.py: ids", cp.returncode == 0 and cp.stdout.split() == ["wv_000001"], (cp.stdout + cp.stderr)[-300:])
cp = subprocess.run([sys.executable, str(ROOT / "scripts" / "filter_images.py"), "--filter", "device_type:manual", "--out", str(ROOT / "derived" / "x.txt"),
                     "--db", env["WV_DB"], "--candidates", env["WV_CANDIDATES"]], capture_output=True, text=True)
check("filter_images.py: refuses to write into derived/", cp.returncode != 0 and not (ROOT / "derived" / "x.txt").exists())

# ---------------------------------------------------------------- 5. SigLIP attribute scoring (mock text bank, no model)
import numpy as np
import auto_annotate as AA
rng = np.random.default_rng(0)
def unit(v): return v / np.linalg.norm(v)
D = 16; E = np.stack([unit(rng.normal(size=D)) for _ in range(4)]).astype("float32")
bank = {f: {o: unit(rng.normal(size=D)).astype("float32") for o in AA.P_V11[f]} for f in AA.P_V11}   # same options as the real prompt bank
bank["_parent"] = {"manual": E[0], "powered": E[1]}
bank["_multi_ramp_reasons"] = {o: (unit(rng.normal(size=D)).astype("float32"), unit(rng.normal(size=D)).astype("float32")) for o in wvcore.FIELD_OPTS["ramp_reasons"] if o != "unknown"}
S = AA.score_attr(E, bank)
cs = [s["suggestions"].get("chair_subtype") for s in S]
check("auto prompts include negative_camber (attr-v4)", "negative_camber" in AA.P_V11["chair_subtype"] and AA.ATTR_PROMPT_VERSION == "attr-v4")
check("attr-v4 securement_hardware prompts", "docking_bracket" in AA.P_V11["securement_hardware"] and "securement_hardware" in AA.ATTR_LOW_CONF)
check("score_attr: chair_subtype within the parent's list",
      cs[0] and cs[0]["parent"] == "manual" and cs[0]["value"] in wvcore.options_for("chair_subtype", "manual")
      and cs[1] and cs[1]["parent"] == "powered" and cs[1]["value"] in wvcore.options_for("chair_subtype", "powered"), json.dumps(cs[:2]))
ok_all = all(wvcore.validate(ev("wv_000001", "x", f, v["value"]))[0] is not None for s in S for f, v in s["suggestions"].items())
check("score_attr: every suggestion is a valid label value", ok_all)
check("score_attr: never suggests 'unknown' or pediatric_size", not any(v["value"] in ("unknown", "pediatric_size") or (isinstance(v["value"], list) and "unknown" in v["value"])
                                                                          for s in S for v in s["suggestions"].values()))

# ---------------------------------------------------------------- 6. assist models (mocked; nothing downloaded)
import assist_models as AM, assist_annotate as AS
cfg_off = {"detector": {"enabled": False}, "segmenter": {"enabled": False}, "vlm": {"enabled": False}}
check("assist: disabled -> get() is None with a reason", AM.get("vlm", cfg_off) is None and AM.status("vlm", cfg_off)["reason"] == "disabled in config")
st = AM.status("detector", {"detector": {"enabled": True, "backend": "nope"}, "segmenter": {}, "vlm": {}})
check("assist: unknown backend degrades", not st["available"] and "unknown backend" in st["reason"])
check("assist: model_tag is a valid auto tag", wvcore.TAG_RE.match(AM.model_tag({"model": "microsoft/Florence-2-base", "revision": "abcdef1234"}) + "_attr-vlm-v1") is not None)
check("assist: default config never downloads", AM.load_config().get("allow_download") is False)
check("assist: parse_choice", AM.parse_choice("It is a Power Tilt Recline chair", wvcore.FIELD_OPTS["chair_subtype"]) == "power_tilt_recline"
      and AM.parse_choice("unknown", ["a", "unknown"]) is None)
class FakeVLM:
    free_text, tag, name = True, "fake-vlm", "fake/vlm"
    def info(self): return {"model": self.name, "backend": "fake", "revision": "test", "tag": self.tag}
    def ask(self, image, q):
        return "manual" if "manual or powered" in q else "standard folding" if "kind of wheelchair" in q else "sidewalk" if "Where" in q else "no idea"
class FakeDet:
    tag = "fake-det"
    def info(self): return {"model": "fake/det", "backend": "fake", "revision": "test", "tag": self.tag}
    def detect(self, image, queries=None, min_score=None): return [{"box": [4, 4, 40, 40], "label": "wheelchair", "score": 0.9}]
class FakeSeg:
    tag = "fake-sam"
    def info(self): return {"model": "fake/sam", "backend": "fake", "revision": "test", "tag": self.tag}
    def segment(self, image, box):
        m = np.zeros((image.size[1], image.size[0]), bool); m[box[1]:box[3], box[0]:box[2]] = True; return m
from PIL import Image, ImageDraw
def chair_img(p, W=160, H=120):
    """A drawn wheel-and-frame shape (no photo, no person)."""
    im = Image.new("RGB", (W, H), (120, 140, 160)); d = ImageDraw.Draw(im)
    d.ellipse((40, 60, 90, 110), outline=(20, 20, 20), width=5); d.rectangle((60, 30, 110, 70), fill=(60, 60, 200)); d.line((110, 30, 110, 100), fill=(30, 30, 30), width=4)
    im.save(p)
(TMP / "ai").mkdir()
for i in ("wv_000001", "wv_000002"): chair_img(TMP / "ai" / f"{i}.jpg")
fakes = {"vlm": FakeVLM(), "detector": FakeDet(), "segmenter": FakeSeg()}
orig_get = AM.get; AM.get = lambda kind, cfg=None: fakes[kind]
try:
    rc = AS.main(["--images", str(TMP / "ai"), "--tasks", "vlm,detect,segment", "--auto-dir", str(TMP / "aauto"), "--assist-dir", str(TMP / "assist")])
finally:
    AM.get = orig_get
vf = TMP / "aauto" / "fake-vlm_attr-vlm-v1.jsonl"
recs = [json.loads(l) for l in open(vf, encoding="utf-8")] if vf.exists() else []
check("assist_annotate: VLM hints in an _attr- file", rc == 0 and len(recs) == 2 and wvcore.is_attr_tag(recs[0]["model_tag"]) and wvcore.TAG_RE.match(recs[0]["model_tag"]))
sg = recs[0]["suggestions"] if recs else {}
check("assist_annotate: subtype from the parent list, unanswerable fields omitted",
      sg.get("chair_subtype", {}).get("value") == "standard_folding" and sg.get("environment", {}).get("value") == "sidewalk" and "weather" not in sg, json.dumps(sg))
check("assist_annotate: drive_position not asked for a manual chair", "drive_position" not in sg)
meta = json.load(open(TMP / "aauto" / "fake-vlm_attr-vlm-v1.meta.json", encoding="utf-8")) if vf.exists() else {}
check("assist_annotate: meta says never human annotation + model name", meta.get("never_counts_as_human_annotation") is True and meta.get("vlm", {}).get("model") == "fake/vlm")
bx = [json.loads(l) for l in open(TMP / "assist" / "fake-det" / "boxes.jsonl", encoding="utf-8")]
mk = [json.loads(l) for l in open(TMP / "assist" / "fake-sam" / "masks.jsonl", encoding="utf-8")]
check("assist_annotate: boxes + masks with model provenance in var/assist", len(bx) == 2 and bx[0]["model"]["tag"] == "fake-det" and len(mk) == 2
      and (TMP / "assist" / "fake-sam" / mk[0]["mask_file"]).exists())
check("assist_annotate: boxes never land in annotations/auto (review mode)", not list((TMP / "aauto").glob("fake-det*")))
au = wvcore.load_auto(str(TMP / "aauto"))
check("assist_annotate: output loads as machine suggestions", "fake-vlm_attr-vlm-v1" in au and "wv_000001" in au["fake-vlm_attr-vlm-v1"])

# ---------------------------------------------------------------- 7. synthetic augmentation + leakage checks
import synth_augment as SY, check_leakage as CL, train_multitask as TM
SD = TMP / "syn"; (SD / "src").mkdir(parents=True); (SD / "bg").mkdir()
MC = ["image_id", "split", "train_eligible", "exclusion_reason", "device_type", "occupied", "leak_group", "bboxes", "license", "creator", "source_url"]
man = [{"image_id": "wv_000101", "split": "train", "train_eligible": "1", "device_type": "manual", "occupied": "no", "leak_group": "g1", "bboxes": json.dumps({"a": [[0.2, 0.2, 0.75, 0.95]]})},
       {"image_id": "wv_000102", "split": "train", "train_eligible": "1", "device_type": "powered", "occupied": "no", "leak_group": "g2", "bboxes": json.dumps({"a": [[0.2, 0.2, 0.75, 0.95]]})},
       {"image_id": "wv_000103", "split": "test", "train_eligible": "1", "device_type": "manual", "occupied": "no", "leak_group": "g3", "bboxes": json.dumps({"a": [[0.2, 0.2, 0.75, 0.95]]})},
       {"image_id": "wv_000104", "split": "train", "train_eligible": "0", "exclusion_reason": "child", "device_type": "manual", "occupied": "no", "leak_group": "g4", "bboxes": json.dumps({"a": [[0.2, 0.2, 0.75, 0.95]]})},
       {"image_id": "wv_000105", "split": "val", "train_eligible": "1", "device_type": "other_aid", "occupied": "no", "leak_group": "g5", "license": "CC0", "creator": "x", "source_url": "u"}]
with open(SD / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, MC, extrasaction="ignore"); w.writeheader(); w.writerows(man)
for r in man: chair_img(SD / "src" / f"{r['image_id']}.jpg")
g = np.zeros((240, 320, 3), "uint8"); g[:, :, 0] = np.linspace(80, 200, 320)[None, :]; g[:, :, 1] = np.linspace(150, 90, 240)[:, None]; g[:, :, 2] = 140
Image.fromarray(g).save(SD / "bg" / "curb1.jpg")
BC = ["background_id", "file", "scene", "source_url", "license", "author", "people", "ground_y", "dataset_image_id"]
bgr = [{"background_id": "bg1", "file": "curb1.jpg", "scene": "curbside", "source_url": "https://example.org/curb1", "license": "CC0-1.0", "author": "test", "people": "none", "ground_y": "0.45"},
       {"background_id": "bg_people", "file": "curb1.jpg", "scene": "curbside", "source_url": "https://example.org/x", "license": "CC0-1.0", "author": "t", "people": "some", "ground_y": "0.45"},
       {"background_id": "bg_nolic", "file": "curb1.jpg", "scene": "sidewalk", "source_url": "https://example.org/y", "license": "", "author": "t", "people": "none", "ground_y": "0.5"},
       {"background_id": "bg_val", "file": "", "scene": "sidewalk", "people": "none", "ground_y": "0.5", "dataset_image_id": "wv_000105"}]
with open(SD / "bg" / "backgrounds.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, BC); w.writeheader(); w.writerows(bgr)
usable, probs = SY.check_backgrounds(SD / "bg" / "backgrounds.csv", man)
check("synth: only the licensed people-free background is usable", [b["background_id"] for b in usable] == ["bg1"], json.dumps(probs))
check("synth: people / missing license / val dataset backgrounds refused", len(probs) == 3 and any("people" in p for p in probs) and any("license" in p for p in probs) and any("not a train" in p for p in probs))
check("synth: sources are train, non-child, wheelchair only", [r["image_id"] for r in man if SY.eligible_source(r)] == ["wv_000101", "wv_000102"])
OUT = SD / "out"
rc = SY.main(["--manifest", str(SD / "manifest.csv"), "--images", str(SD / "src"), "--backgrounds", str(SD / "bg" / "backgrounds.csv"), "--out", str(OUT),
              "--assist-dir", str(TMP / "none"), "--n", "6", "--seed", "1", "--db", str(TMP / "none.db"), "--events", str(TMP / "none")])
rows = list(csv.DictReader(open(OUT / "synthetic_manifest.csv", encoding="utf-8")))
check("synth: generated images + manifest", rc == 0 and len(rows) == 6 and all((OUT / r["file"]).exists() for r in rows))
check("synth: tagged synthetic, train-only, source train, leak group inherited",
      all(r["synthetic"] == "true" and r["split"] == "train" and r["source_image_id"] in ("wv_000101", "wv_000102")
          and r["leak_group"] == {"wv_000101": "g1", "wv_000102": "g2"}[r["source_image_id"]] for r in rows))
check("synth: provenance (background license/url, transform, variant)", all(r["background_license"] == "CC0-1.0" and r["background_url"] and json.loads(r["transform_params"]).get("foot_y") and r["weather_variant"] for r in rows))
check("synth: ramp judgement never inherited", all(r["ramp_deploy"] == "" and r["ramp_reasons"] == "" for r in rows))
check("synth: ids never collide with real ids", all(int(r["image_id"][3:]) >= SY.FIRST_ID for r in rows))
check("synth: no model-output column names", not [c for c in rows[0] if __import__("re").search(r"sugg|confid|prob|score|logit|emb|auto|pred", c.lower())])
im0 = Image.open(OUT / rows[0]["file"]); check("synth: output has background size", im0.size == (320, 240))
check("synth: gitignore guard in output dir", (OUT / ".gitignore").read_text().strip() == "*")
CL.FAIL.clear(); CL.check_synthetic(OUT / "synthetic_manifest.csv", SD / "manifest.csv")
check("check_leakage --synthetic: clean output passes", not CL.FAIL, json.dumps(CL.FAIL))
bad = [dict(rows[0], split="val"), dict(rows[1], source_image_id="wv_000103"), dict(rows[2], source_image_id="wv_000104"), dict(rows[3], image_id="wv_000101"),
       dict(rows[4], leak_group="g9"), dict(rows[5], background_license="")]
with open(SD / "bad.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(bad)
CL.FAIL.clear(); CL.check_synthetic(SD / "bad.csv", SD / "manifest.csv")
fl = " | ".join(CL.FAIL)
check("check_leakage --synthetic: catches val split, test source, child source, id collision, leak group, license",
      all(s in fl for s in ("split=train", "source is a train", "child", "collide", "leak_group", "license")), fl[:400])
CL.FAIL.clear(); CL.check_code()
check("check_leakage: assist scripts are label-free + model scripts pass", not [f for f in CL.FAIL if "assist" in f], json.dumps(CL.FAIL))
CL.FAIL.clear()
trows = [dict(r) for r in man]
syn = TM.read_synthetic(str(OUT / "synthetic_manifest.csv"), trows)
check("train_multitask.read_synthetic: rows load as train with image path + scene labels from the background",
      len(syn) == 6 and all(s["split"] == "train" and s["_path"].endswith(".jpg") and s["environment"] == "curbside" for s in syn))
check("train_multitask.read_synthetic: refuses val synthetic rows", raises(lambda: TM.read_synthetic(str(SD / "bad.csv"), trows)))

# ---------------------------------------------------------------- 8. docs + gitignore
for d in ("FILTERING_v1.1.md", "SYNTHETIC_AUGMENTATION_v1.1.md", "RAMP_LABELS_v1.1.md", "ANNOTATION_MODELS_v1.1.md"):
    check(f"docs/{d} exists", (ROOT / "docs" / d).exists())
dd = (ROOT / "docs" / "DATA_DICTIONARY_v1.1.md").read_text(encoding="utf-8")
check("data dictionary documents subtypes + ramp + securement + attr-v4",
      "chair_subtype" in dd and "ramp_deploy" in dd and "securement_hardware" in dd and "attr-v4" in dd and "adds **no** wheelchair subtype" not in dd)
check("data dictionary lists negative_camber in the manual subtypes", "negative_camber" in dd)
ign = [subprocess.run(["git", "check-ignore", "-q", "--no-index", p], cwd=ROOT).returncode == 0
       for p in ("var/synthetic/images/wv_800001.jpg", "var/assist/x/masks/a.png", "data/backgrounds/a.jpg", "x/model.safetensors", "y.pt", "weights/z.bin")]
kept = subprocess.run(["git", "check-ignore", "-q", "--no-index", "data/backgrounds/backgrounds.csv"], cwd=ROOT).returncode != 0
check("gitignore: synthetic output, masks, background photos and weights ignored; backgrounds.csv kept", all(ign) and kept, str(ign))

n = sum(ok for _, ok in results)
print(f"\n{n}/{len(results)} passed; temp dir {TMP}")
sys.exit(0 if n == len(results) else 1)
