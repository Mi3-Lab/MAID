#!/usr/bin/env python3
"""Fail (exit 1) if the WheelView dataset or the train/eval code can leak test information into a model.

  python scripts/check_leakage.py              # manifest + split checks + static code audit (works without images)
  python scripts/check_leakage.py --features   # + recompute near-duplicates from var/manifest_features.npz (lab laptop)
  python scripts/check_leakage.py --files      # + sha256 of images_blurred/ files must match the manifest
  python scripts/check_leakage.py --runtime    # + run scripts/predict.py on a test image under an audit hook per saved model
  python scripts/check_leakage.py --release var/release/WheelView_v1.0   # + audit a built release folder (see check_release)

Checks (see data/MANIFEST.md):
  data   no duplicate (sha256 / pHash / dup_cluster), uploader, creator, source page, Flickr photo, Flickr burst or
         leak_group spans two splits; manifest consistent with its split file; neutral paths/IDs; no model-output columns
  code   no kNN / retrieval / prototype code; no cached embeddings or manifest features read back; fit() never sees
         val/test/extra data; class weights from train only; YOLO-cls test images not in labelled folders; YOLO-det
         training yaml has no test split; zero-shot uses no batch statistics; scripts read splits from the manifest
  infer  (--runtime) predicting one image opens only the weights, that image, code and server/schema.json
Warnings (reported, not failures): ID order vs class correlation, historical artifacts under var/."""
import argparse, ast, csv, glob, json, os, re, subprocess, sys, tempfile
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server")); import wvcore
CLASSES = list(wvcore.DEVICE_CLASSES); SPLITS = ("train", "val", "test")
ALL_CLASS_WORDS = set(CLASSES) | {"lookalike", "unclear", "wheelchair", "stroller", "walker", "rollator", "scooter"}
MODEL_SCRIPTS = ["train_baseline.py", "train_siglip.py", "train_yolo.py", "compare_v2.py", "predict.py"]
FAIL, WARN = [], []
def fail(m): FAIL.append(m); print("FAIL", m)
def warn(m): WARN.append(m); print("warn", m)
def ok(m): print("ok  ", m)
norm_name = lambda s: re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def flickr_photo(url):
    m = re.search(r"flickr\.com/photos/[^/]+/(\d+)", url or ""); return int(m.group(1)) if m else None

def one_split(rows, key, label, skip=("",)):
    by = defaultdict(set)
    for r in rows:
        k = key(r)
        if k not in skip and k is not None: by[k].add(r["split"])
    bad = {k: s for k, s in by.items() if len(s) > 1}
    if bad: fail(f"{label}: {len(bad)} value(s) span splits, e.g. " + "; ".join(f"{k} -> {sorted(s)}" for k, s in list(bad.items())[:5]))
    else: ok(f"{label}: {len(by)} values, each in one split")

def check_data(a, rep):
    mf = ROOT / "data" / "manifest_v2.csv"
    if not mf.exists(): fail("data/manifest_v2.csv missing (run scripts/build_manifest.py)"); return []
    M = list(csv.DictReader(open(mf, encoding="utf-8")))
    ids = [r["image_id"] for r in M]
    if len(ids) != len(set(ids)): fail("duplicate image_id rows in manifest")
    T = [r for r in M if r["train_eligible"] == "1"]
    for r in M:
        if r["train_eligible"] == "1" and (r["split"] not in SPLITS or r["device_type"] not in CLASSES): fail(f"{r['image_id']}: trainable row with split={r['split']} label={r['device_type']}")
        if r["train_eligible"] != "1" and (r["split"] != "excluded" or not r["exclusion_reason"]): fail(f"{r['image_id']}: excluded row must have split=excluded and a reason")
    ok(f"manifest: {len(M)} rows, {len(T)} trainable, {len(M) - len(T)} excluded")
    # cross-split groupings
    one_split(T, lambda r: r["leak_group"], "leak_group")
    one_split(T, lambda r: r["dup_cluster"], "dup_cluster")
    one_split(T, lambda r: r["sha256_blurred"], "sha256 (exact duplicate)")
    one_split(T, lambda r: r["phash_blurred"], "identical pHash")
    one_split(T, lambda r: r["uploader_group"], "uploader")
    one_split(T, lambda r: norm_name(r["creator"]), "creator name", skip=("", "unknown", "anonymous"))
    one_split(T, lambda r: re.sub(r"[?#].*$", "", r["source_url"].lower()).rstrip("/"), "source page")
    one_split(T, lambda r: flickr_photo(r["source_url"]), "Flickr photo id")
    gap = (rep.get("thresholds") or {}).get("burst_gap", 3)
    fl = sorted((flickr_photo(r["source_url"]), r["split"], r["image_id"]) for r in T if flickr_photo(r["source_url"]))
    bursts = [(x, y) for x, y in zip(fl, fl[1:]) if y[0] - x[0] <= gap and x[1] != y[1]]
    (fail if bursts else ok)(f"Flickr id bursts (gap <= {gap}) across splits: {len(bursts)}")
    # pHash near-dups straight from the manifest column (no cache needed)
    tm = (rep.get("thresholds") or {}).get("phash_max", 8)
    hs = [(int(r["phash_blurred"], 16), r["split"], r["image_id"]) for r in T]
    near = [(x[2], y[2], bin(x[0] ^ y[0]).count("1")) for i, x in enumerate(hs) for y in hs[i + 1:] if x[1] != y[1] and bin(x[0] ^ y[0]).count("1") <= tm]
    (fail if near else ok)(f"pHash (non-mirrored) <= {tm} across splits: {len(near)} {near[:5]}")
    # split file agreement
    sf = ROOT / "data" / ("splits_v3.csv" if (ROOT / "data" / "splits_v3.csv").exists() else "splits_v2.csv")
    S = {r["image_id"]: r["split"] for r in csv.DictReader(open(sf, encoding="utf-8"))}
    dis = [r["image_id"] for r in T if S.get(r["image_id"]) != r["split"]] + [i for i in S if i not in {r["image_id"] for r in T}]
    (fail if dis else ok)(f"manifest splits == {sf.name}: {len(dis)} mismatches")
    # neutral identifiers
    bad = [r["image_id"] for r in M if not re.fullmatch(r"wv_\d{6}", r["image_id"]) or r["path"] != f"images_blurred/{r['image_id']}.jpg"
           or not re.fullmatch(r"g\d{4}", r["leak_group"]) or not re.fullmatch(r"d\d{4}", r["dup_cluster"])]
    (fail if bad else ok)(f"ids/paths/group ids are neutral (wv_NNNNNN, images_blurred/<id>.jpg, gNNNN, dNNNN): {len(bad)} bad")
    cols = [c for c in M[0] if re.search(r"sugg|confid|prob|score|logit|emb|auto|pred", c.lower())] if M else []
    (fail if cols else ok)(f"no model-output columns in manifest: {cols}")
    # ID order vs class (IDs follow collection order, which follows the search query): warning only
    if T:
        rank = {i: k for k, i in enumerate(sorted(r["image_id"] for r in T))}
        med = {c: sorted(rank[r["image_id"]] for r in T if r["device_type"] == c) for c in CLASSES}
        med = {c: (v[len(v) // 2] / len(T) if v else None) for c, v in med.items()}
        warn("image_id order correlates with class (median ID rank / n: " + ", ".join(f"{c} {v:.2f}" for c, v in med.items() if v is not None)
             + "); IDs are never a model input; never sort/bin/batch by ID (train loaders shuffle)")
    return T

def check_features(a, T, rep):
    import numpy as np
    cache = ROOT / "var" / "manifest_features.npz"
    if not cache.exists(): fail("--features: var/manifest_features.npz missing"); return
    z = np.load(cache, allow_pickle=False); idx = {s: k for k, s in enumerate(z["sha"])}
    miss = [r["image_id"] for r in T if r["sha256_blurred"] not in idx]
    if miss: fail(f"--features: {len(miss)} manifest images have no cached features"); return
    k = [idx[r["sha256_blurred"]] for r in T]; sp = np.array([r["split"] for r in T])
    E = z["emb"][k].astype("float32"); P = z["ph"][k].astype("uint8"); PM = z["phm"][k].astype("uint8")
    C = E @ E.T; H = np.minimum((P[:, None] != P[None]).sum(-1), (P[:, None] != PM[None]).sum(-1))
    th = rep.get("thresholds") or {}; cm, hm = th.get("cos_min", 0.95), th.get("phash_max", 8)
    diff = sp[:, None] != sp[None]
    bad = np.argwhere(np.triu(diff & ((C >= cm) | (H <= hm)), 1))
    (fail if len(bad) else ok)(f"--features: near-duplicates (cos >= {cm} or mirrored pHash <= {hm}) across splits: {len(bad)}"
                               + (" e.g. " + ", ".join(f"{T[i]['image_id']}/{T[j]['image_id']} cos={C[i, j]:.3f} ph={H[i, j]}" for i, j in bad[:5]) if len(bad) else ""))
    m = np.where(np.triu(diff, 1), C, -1); i, j = np.unravel_index(m.argmax(), m.shape)
    ok(f"--features: closest cross-split pair {T[i]['image_id']}/{T[j]['image_id']} cos={C[i, j]:.3f} pHash={H[i, j]}")

def check_files(T):
    import hashlib
    bad = [r["image_id"] for r in T if not (ROOT / r["path"]).exists() or hashlib.sha256((ROOT / r["path"]).read_bytes()).hexdigest() != r["sha256_blurred"]]
    (fail if bad else ok)(f"--files: sha256 of images_blurred files matches manifest: {len(bad)} mismatches {bad[:5]}")

# ---------------------------------------------------------------- static code audit
FORBIDDEN_NAMES = re.compile(r"(?i)^(NearestNeighbors|KNeighbors\w*|RadiusNeighbors\w*|faiss|annoy\w*|hnswlib|pynndescent|BallTree|KDTree|cdist|"
                             r"prototypes?|centroids?|class_means?|memory_bank|support_set|kmeans|KMeans)$")
FORBIDDEN_FILES = re.compile(r"emb\.npz|manifest_features|emb_blur|emb_orig")
LOADERS = {"load", "loadtxt", "genfromtxt", "load_npz", "read_csv", "read_parquet", "open", "fromfile", "memmap"}
def strs(node): return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]

def audit_script(p):
    src = p.read_text(encoding="utf-8"); tree = ast.parse(src); name = p.name; probs = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and FORBIDDEN_NAMES.match(n.id): probs.append(f"retrieval/prototype identifier `{n.id}` line {n.lineno}")
        if isinstance(n, ast.Attribute) and FORBIDDEN_NAMES.match(n.attr): probs.append(f"retrieval/prototype attribute `.{n.attr}` line {n.lineno}")
        if isinstance(n, ast.alias) and FORBIDDEN_NAMES.match(n.name.split(".")[0]): probs.append(f"import {n.name} line {n.lineno}")
        if isinstance(n, ast.Call):
            fn = n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
            if fn in LOADERS and any(FORBIDDEN_FILES.search(s) for s in strs(n)): probs.append(f"reads cached features ({fn}) line {n.lineno}")
            if fn in ("fit", "partial_fit", "fit_transform", "fit_predict"):
                bad = [s for s in strs(n) if s in ("val", "test", "extra")]
                if bad: probs.append(f"{fn}() on {bad} data line {n.lineno}")
            if fn == "bincount" and not any(s == "train" for s in strs(n)): probs.append(f"class counts not from train line {n.lineno}")
    # compare_v2.py only tabulates the archived v2 runs, so it intentionally reads data/splits_v2.csv
    if name in ("train_baseline.py", "train_siglip.py", "train_yolo.py") and "manifest_v2.csv" not in src and "DEFAULT_SPLITS" not in src:
        probs.append("does not default to data/manifest_v2.csv")
    if name == "train_siglip.py":
        zs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "zs"]
        if not zs: probs.append("zero-shot function zs() not found")
        for f in zs:
            for n in ast.walk(f):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("mean", "std", "fit", "median"):
                    probs.append(f"zero-shot uses batch statistic .{n.func.attr}() line {n.lineno} (transductive)")
        if "probe_seed" not in src or "_head.npz" not in src: probs.append("linear-probe head is not saved (inference would need train embeddings)")
    if name == "train_yolo.py":
        if not re.search(r'for s in \("train", "val"\):\s*\n\s*for r in sp\[s\]: link\(img / f"\{r\[\'image_id\'\]\}\.jpg", ds / s / r\["label"\]', src):
            probs.append("YOLO-cls dataset: could not verify that only train/val are linked into class folders")
        if re.search(r'ds\s*/\s*"test"\s*/', src): probs.append("YOLO-cls: test images linked into labelled folders")
        m = re.search(r'\byaml\.write_text\((.*)\)', src)
        if not m or "test:" in m.group(1): probs.append("YOLO-det training data.yaml declares a test split")
    if name == "train_baseline.py" and 'dl("train", tf_tr, True)' not in src: probs.append("train loader not shuffled")
    return probs

def check_code():
    for s in MODEL_SCRIPTS:
        p = ROOT / "scripts" / s
        if not p.exists(): fail(f"code: scripts/{s} missing"); continue
        pr = audit_script(p)
        for x in pr: fail(f"code: {s}: {x}")
        if not pr: ok(f"code: {s}: no retrieval/kNN/prototypes, no cached features read, fit/class weights on train only")
    for d in sorted(glob.glob(str(ROOT / "var" / "yolo_cls_*" / "test"))):
        warn(f"historical artifact: {Path(d).relative_to(ROOT)} has labelled test folders from an older run (not used by the current "
             "script, which refuses to reuse that dataset dir; left in place, nothing deleted)")

ALLOWED_REPO = re.compile(r"\.(py|pyc|pyd|so|dll)$")
def check_runtime(T, img=None, tag="--runtime"):
    test = [r for r in T if r["split"] == "test"]
    if not test: fail("--runtime: no test row"); return
    img = img or ROOT / test[0]["path"]
    if not img.exists(): fail(f"--runtime: {img} not on this machine"); return
    cands = []
    for kind, pat in [("resnet", "var/runs/resnet50_*/seed0_best.pt"), ("siglip-probe", "var/runs/siglip2_*/probe_seed0_head.npz"),
                      ("siglip-ft", "var/runs/siglip2_*/ft_seed0_best.pt"), ("yolo-cls", "var/runs/yolo11s_cls_*_seed0/weights/best.pt")]:
        g = sorted(glob.glob(str(ROOT / pat)))
        if g: cands.append((kind, Path(g[-1])))
        else: warn(f"--runtime: no {kind} weights ({pat}); not checked")
    for kind, w in cands:
        with tempfile.TemporaryDirectory() as td:
            au = Path(td) / "opened.json"
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "predict.py"), "--kind", kind, "--weights", str(w), "--audit", str(au), str(img)],
                               capture_output=True, text=True, cwd=str(ROOT))
            if r.returncode != 0: fail(f"{tag} {kind}: predict.py failed: {r.stderr.strip().splitlines()[-1:]}"); continue
            opened = json.loads(au.read_text())
        root = str(ROOT.resolve()).lower(); allowed = {str(w.resolve()).lower(), str(img.resolve()).lower(), str((ROOT / "server" / "schema.json").resolve()).lower()}
        bad = []
        for f in opened:
            fl = str(Path(f).resolve()).lower() if os.path.exists(f) else f.lower()
            if not fl.startswith(root) or fl in allowed or ALLOWED_REPO.search(fl) or os.path.isdir(f): continue
            if "__pycache__" in fl or os.sep + ".venv" + os.sep in fl or "/.venv/" in fl: continue
            bad.append(os.path.relpath(f, ROOT))
        (fail if bad else ok)(f"{tag} {kind}: predict on {os.path.relpath(img, ROOT)} opened {len(opened)} files; repo files other than weights/image/code/schema: {bad[:8]}")

# ---------------------------------------------------------------- release audit
META_MARKERS = set(range(0xE1, 0xF0)) | {0xFE}
def jpeg_meta_segments(data):
    out, i = [], 2
    while i + 4 <= len(data) and data[i] == 0xFF:
        m = data[i + 1]
        if m == 0xDA: break
        if m == 0x01 or 0xD0 <= m <= 0xD7: i += 2; continue
        if m in META_MARKERS: out.append(hex(m))
        i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    return out

def phash64(im):
    """Same 63-bit pHash as scripts/build_manifest.py (32x32 grey LANCZOS, orthonormal 2-D DCT, 8x8 block minus DC, > median)."""
    import numpy as np, scipy.fft
    from PIL import Image
    g = np.asarray(im.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    d = scipy.fft.dctn(g, norm="ortho")[:8, :8].flatten(); bits = d[1:] > np.median(d[1:])
    return int("".join("1" if b else "0" for b in bits), 2)

def check_release(a, T, rep, M=None, hold=None):
    """Audit a built release folder: metadata vs manifest, exact image set (no safety-hold, nothing extra), SHA256SUMS,
    no EXIF/XMP/ICC/comment segments, size <= 1280 px, cross-split near-duplicates recomputed from the released pixels
    (pHash incl. mirror), no model-output columns or weights shipped, and predict.py on a released test image."""
    import hashlib
    from PIL import Image, ImageOps
    R = Path(a.release).resolve(); tag = "--release"
    if not R.is_dir(): fail(f"{tag}: {R} not found"); return
    if M is None:
        M = {r["image_id"]: r for r in csv.DictReader(open(ROOT / "data" / "manifest_v2.csv", encoding="utf-8"))}
        hold = {r["image_id"] for r in csv.DictReader(open(ROOT / "data" / "safety_hold.csv", encoding="utf-8"))}
    L = list(csv.DictReader(open(R / "annotations" / "labels.csv", encoding="utf-8")))
    LD = {r["image_id"]: r for r in L}
    # face_recheck.csv: images whose release copy got extra face regions blurred; their released sha256 is sha256_after,
    # and sha256_before must equal the manifest's sha256_blurred.
    FX = {}
    if (R / "face_recheck.csv").exists():
        for r in csv.DictReader(open(R / "face_recheck.csv", encoding="utf-8")): FX.setdefault(r["image_id"], r)
    fxbad = [i for i, r in FX.items() if i not in M or r["sha256_before"] != M[i]["sha256_blurred"] or not r["sha256_after"]]
    if FX: (fail if fxbad else ok)(f"{tag}: face_recheck.csv lists {len(FX)} images with extra face blur; sha256_before == manifest, sha256_after set: {len(fxbad)} bad {fxbad[:5]}")
    want = {i: (FX[i]["sha256_after"] if i in FX else M[i]["sha256_blurred"]) for i in M}
    bad = [i for i in M if i not in LD or LD[i]["split"] != M[i]["split"] or wvcore.canon_class(LD[i]["device_type"]) != wvcore.canon_class(M[i]["device_type"])
           or LD[i]["sha256"] != want[i] or LD[i]["file"] != f"images/{M[i]['split']}/{i}.jpg"] + [i for i in LD if i not in M]
    (fail if bad or len(L) != len(LD) else ok)(f"{tag}: labels.csv has {len(L)} rows == manifest ({len(M)}), same split/label/sha256/file: {len(bad)} mismatches {bad[:5]}")
    cols = [c for c in (L[0] if L else {}) if re.search(r"sugg|confid|prob|score|logit|emb|auto|pred", c.lower())]
    (fail if cols else ok)(f"{tag}: no model-output columns in labels.csv: {cols}")
    seen = {}
    for s in SPLITS:
        f = R / "splits" / f"{s}.txt"; ids = f.read_text(encoding="utf-8").split() if f.exists() else None
        if ids is None: fail(f"{tag}: splits/{s}.txt missing"); continue
        exp = sorted(r["image_id"] for r in T if r["split"] == s)
        dup = [i for i in ids if i in seen]; seen.update({i: s for i in ids})
        (fail if sorted(ids) != exp or dup else ok)(f"{tag}: splits/{s}.txt {len(ids)} ids == manifest {s} ({len(exp)}); in another split file: {len(dup)}")
    jpgs = sorted(p for p in (R / "images").rglob("*") if p.is_file())
    exp_files = {f"images/{r['split']}/{i}.jpg" for i, r in M.items()}
    got = [p.relative_to(R).as_posix() for p in jpgs]
    extra = sorted(set(got) - exp_files); missing = sorted(exp_files - set(got))
    held = [g for g in got if Path(g).stem in hold]
    stems = Counter(Path(g).stem for g in got); dups = [k for k, v in stems.items() if v > 1]
    (fail if extra or missing or held or dups else ok)(f"{tag}: images/ holds exactly the {len(exp_files)} manifest images in their split folders "
        f"(extra {len(extra)} {extra[:3]}, missing {len(missing)} {missing[:3]}, safety-hold {len(held)}, same id twice {len(dups)})")
    other = [p.relative_to(R).as_posix() for p in R.rglob("*") if p.is_file() and not p.relative_to(R).as_posix().startswith("images/")
             and re.search(r"\.(jpe?g|png|webp|gif|bmp|tiff?|pt|pth|npz|npy|onnx|safetensors|bin|ckpt|db|jsonl)$", p.name.lower())]
    (fail if other else ok)(f"{tag}: no images, weights, embeddings, DBs or event logs outside images/: {other[:5]}")
    sums = {}
    sf = R / "SHA256SUMS.txt"
    if sf.exists():
        for line in sf.read_text(encoding="utf-8").splitlines():
            h, f = line.split("  ", 1); sums[f] = h
    badsum, meta, big, ph = [], [], [], []
    for p, g in zip(jpgs, got):
        data = p.read_bytes()
        if sums.get(g) != hashlib.sha256(data).hexdigest(): badsum.append(g)
        segs = jpeg_meta_segments(data)
        if segs: meta.append((g, segs))
        im = Image.open(p); im.load()
        if im.format != "JPEG" or max(im.size) > 1280 or im.info.get("exif") or im.info.get("icc_profile"): big.append((g, im.format, im.size))
        if g in exp_files and M[Path(g).stem]["split"] in SPLITS:
            ph.append((phash64(im), phash64(ImageOps.mirror(im)), M[Path(g).stem]["split"], Path(g).stem))
    (fail if badsum or set(sums) != set(got) else ok)(f"{tag}: SHA256SUMS.txt covers {len(sums)} files, all images match: {len(badsum)} mismatches {badsum[:3]}")
    (fail if meta else ok)(f"{tag}: no EXIF/XMP/ICC/IPTC/comment segments in any JPEG: {len(meta)} files with metadata {meta[:3]}")
    (fail if big else ok)(f"{tag}: all images JPEG, longer side <= 1280 px, no EXIF/ICC: {len(big)} bad {big[:3]}")
    tm = (rep.get("thresholds") or {}).get("phash_max", 8)
    near = [(x[3], y[3], min(bin(x[0] ^ y[0]).count("1"), bin(x[0] ^ y[1]).count("1"))) for k, x in enumerate(ph) for y in ph[k + 1:]
            if x[2] != y[2] and min(bin(x[0] ^ y[0]).count("1"), bin(x[0] ^ y[1]).count("1")) <= tm]
    (fail if near else ok)(f"{tag}: pHash recomputed from released pixels (incl. mirror) <= {tm} across splits: {len(near)} {near[:5]}")
    ag = sum(1 for h, _, _, i in ph if M[i].get("phash_blurred") and int(M[i]["phash_blurred"], 16) == h)
    if any(r.get("phash_blurred") for r in M.values()): ok(f"{tag}: recomputed pHash equals the manifest's phash_blurred for {ag}/{len(ph)} images (informational)")
    test = sorted(i for i, r in M.items() if r["split"] == "test")
    if a.runtime and test: check_runtime(T, img=R / "images" / "test" / f"{test[0]}.jpg", tag=tag)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", action="store_true"); ap.add_argument("--files", action="store_true"); ap.add_argument("--runtime", action="store_true")
    ap.add_argument("--release", default=None, help="audit a built release folder (e.g. var/release/WheelView_v1.0)")
    a = ap.parse_args()
    if a.release and not (ROOT / "data" / "manifest_v2.csv").exists():
        # Standalone use inside a downloaded release (code/scripts/check_leakage.py --release .): labels.csv is the reference.
        R = Path(a.release).resolve()
        L = list(csv.DictReader(open(R / "annotations" / "labels.csv", encoding="utf-8")))
        M = {r["image_id"]: dict(r, sha256_blurred=r["sha256"]) for r in L}
        if (R / "face_recheck.csv").exists():   # re-checked images: pre-fix hash is the reference for sha256_before
            for x in csv.DictReader(open(R / "face_recheck.csv", encoding="utf-8")):
                if x["image_id"] in M and x["sha256_after"] == M[x["image_id"]]["sha256"]: M[x["image_id"]]["sha256_blurred"] = x["sha256_before"]
        T = [dict(r, path=r["file"]) for r in L if r["split"] in SPLITS]
        print("standalone release check: labels.csv is the reference (no repo manifest found)")
        groups = {}
        for r in T: groups.setdefault(r["leak_group"], set()).add(r["split"])
        span = [g for g, v in groups.items() if len(v) > 1]
        (fail if span else ok)(f"--release: leak groups spanning two splits: {len(span)} {span[:5]}")
        a.runtime = a.runtime and any((ROOT / "var" / "runs").glob("*")) if (ROOT / "var" / "runs").exists() else False
        check_release(a, T, {}, M=M, hold=set())
        print(f"\n{len(FAIL)} failure(s), {len(WARN)} warning(s)")
        sys.exit(1 if FAIL else 0)
    rp = ROOT / "results" / "leakage_v2.json"; rep = json.load(open(rp, encoding="utf-8")) if rp.exists() else {}
    T = check_data(a, rep)
    if a.features and T: check_features(a, T, rep)
    if a.files and T: check_files(T)
    check_code()
    if a.runtime and T: check_runtime(T)
    if a.release and T: check_release(a, T, rep)
    print(f"\n{len(FAIL)} failure(s), {len(WARN)} warning(s)")
    sys.exit(1 if FAIL else 0)

if __name__ == "__main__":
    main()
