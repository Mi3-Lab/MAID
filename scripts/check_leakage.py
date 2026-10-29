#!/usr/bin/env python3
"""Fail (exit 1) if the WheelView dataset or the train/eval code can leak test information into a model.

  python scripts/check_leakage.py              # manifest + split checks + static code audit (works without images)
  python scripts/check_leakage.py --features   # + recompute near-duplicates from var/manifest_features.npz (lab laptop)
  python scripts/check_leakage.py --files      # + sha256 of images_blurred/ files must match the manifest
  python scripts/check_leakage.py --runtime    # + run scripts/predict.py on a test image under an audit hook per saved model

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
def check_runtime(T):
    test = [r for r in T if r["split"] == "test"]
    if not test: fail("--runtime: no test row"); return
    img = ROOT / test[0]["path"]
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
            if r.returncode != 0: fail(f"--runtime {kind}: predict.py failed: {r.stderr.strip().splitlines()[-1:]}"); continue
            opened = json.loads(au.read_text())
        root = str(ROOT.resolve()).lower(); allowed = {str(w.resolve()).lower(), str(img.resolve()).lower(), str((ROOT / "server" / "schema.json").resolve()).lower()}
        bad = []
        for f in opened:
            fl = str(Path(f).resolve()).lower() if os.path.exists(f) else f.lower()
            if not fl.startswith(root) or fl in allowed or ALLOWED_REPO.search(fl) or os.path.isdir(f): continue
            if "__pycache__" in fl or os.sep + ".venv" + os.sep in fl or "/.venv/" in fl: continue
            bad.append(os.path.relpath(f, ROOT))
        (fail if bad else ok)(f"--runtime {kind}: predict opened {len(opened)} files; repo files other than weights/image/code/schema: {bad[:8]}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", action="store_true"); ap.add_argument("--files", action="store_true"); ap.add_argument("--runtime", action="store_true")
    a = ap.parse_args()
    rp = ROOT / "results" / "leakage_v2.json"; rep = json.load(open(rp, encoding="utf-8")) if rp.exists() else {}
    T = check_data(a, rep)
    if a.features and T: check_features(a, T, rep)
    if a.files and T: check_files(T)
    check_code()
    if a.runtime and T: check_runtime(T)
    print(f"\n{len(FAIL)} failure(s), {len(WARN)} warning(s)")
    sys.exit(1 if FAIL else 0)

if __name__ == "__main__":
    main()
