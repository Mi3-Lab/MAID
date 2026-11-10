#!/usr/bin/env python3
"""Three-family model-consensus AI labels for WheelView / MAID.

Writes AI-only CSVs under derived/. Never writes annotations/events/*.jsonl,
never overwrites human labels, and never counts these as human annotations.

Models (distinct families):
  A  SigLIP 2 zero-shot   google/siglip2-so400m-patch14-224
                          (reuse annotations/auto/siglip2-so400m-224_zs-v1.jsonl)
  B  CLIP zero-shot       openai/clip-vit-large-patch14  (OpenAI CLIP, not SigLIP)
  C  DINOv2 + LR probe    facebook/dinov2-base features + LogisticRegression
                          trained on human labels with stratified k-fold OOF

A label is emitted only when all three models predict the same class and each
confidence is at or above a threshold chosen so 3-way-consensus precision on
held-out human labels meets the target (device_type >= 97%). A field that
cannot meet the bar is not written.

Usage:
  python scripts/ai_labels_3model.py --phase validate
  python scripts/ai_labels_3model.py --phase infer --limit 40
  python scripts/ai_labels_3model.py --phase infer
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import os
import sqlite3
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
try:
    import ai_jobs
except ImportError:
    ai_jobs = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

IMAGE_ROOT_DEFAULT = Path(r"C:\Users\kamoo\wheelview_all_images_new")
MANIFEST_DEFAULT = IMAGE_ROOT_DEFAULT / "manifest.csv"

MODEL_A_ID = "google/siglip2-so400m-patch14-224"
MODEL_A_TAG = "siglip2-so400m-224_zs-v1"
MODEL_A_REV = "78e403963a4f6a3640d07803284752326fdf4edf"
MODEL_A_JSONL = ROOT / "annotations" / "auto" / "siglip2-so400m-224_zs-v1.jsonl"

MODEL_B_ID = "openai/clip-vit-large-patch14"
MODEL_C_ID = "facebook/dinov2-base"

SAFETY_HOLD = ("wv_000250", "wv_000269", "wv_000277", "wv_000280")
ZS_TEMP = 0.01
DEVICE_CLASSES = ("manual", "powered", "other_aid", "ambiguous")  # never emit unclear
OCCUPIED_CLASSES = ("yes", "no")
N_FOLDS = 5
TARGET_PRECISION = 0.97
THRESH_GRID = tuple(round(x * 0.01, 2) for x in range(0, 100, 5))  # 0.00 .. 0.95
BATCH = 8

# Zero-shot prompts (same wording as scripts/auto_annotate.py P["device_type"] /
# P["occupied"] so model A and B are comparable; lookalike == ambiguous).
PROMPTS_DEVICE = {
    "manual": [
        "a manual wheelchair with large rear wheels and push rims",
        "a person sitting in a manual wheelchair",
        "a folding manual wheelchair",
        "a lightweight sports wheelchair pushed by hand",
        "a hospital-style manual wheelchair with push handles",
    ],
    "powered": [
        "an electric powered wheelchair with a joystick control",
        "a person in a power wheelchair",
        "a motorized wheelchair with a battery box and small wheels",
        "an electric wheelchair",
    ],
    "other_aid": [
        "a mobility scooter",
        "a person riding a mobility scooter",
        "a rollator walker with wheels and a seat",
        "a walking frame",
        "a person using a walker",
        "a walking cane or crutches",
    ],
    "ambiguous": [
        "a baby stroller",
        "a pram or pushchair",
        "a shopping cart",
        "a supermarket trolley",
        "an office chair on wheels",
        "a luggage trolley",
        "a baggage cart at an airport",
    ],
}
PROMPTS_OCCUPIED = {
    "yes": [
        "a person sitting in a wheelchair",
        "a person seated on a mobility device",
        "someone riding a scooter",
        "a wheelchair user",
    ],
    "no": [
        "an empty wheelchair with nobody in it",
        "an empty mobility device with no one sitting on it",
        "an unoccupied chair",
        "an empty stroller",
    ],
}

TZ_NAME = "America/Los_Angeles"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(TZ_NAME)).isoformat(timespec="seconds")
    except Exception:
        return datetime.now().astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{now_iso()}] {msg}", flush=True)

def _job(stage, n_done=0, n_total=0, status="running", message=""):
    if ai_jobs is None:
        return
    try:
        ai_jobs.update("ai_labels_3model", stage, n_done, n_total, status=status, message=message or stage)
    except Exception:
        pass



def ensure_hf_token() -> None:
    """Load HF_TOKEN from the huggingface cache file if the env var is unset.

    Never prints the token or the file contents.
    """
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.is_file():
        val = token_path.read_text(encoding="utf-8").strip()
        if val:
            os.environ["HF_TOKEN"] = val


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if n <= 0:
        return (None, None, None)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def canon_device(v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    v = str(v).strip().lower()
    if v in ("lookalike", "look-alike"):
        return "ambiguous"
    if v in DEVICE_CLASSES or v == "unclear":
        return v
    return v


def file_sha1(path: Path, n: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(n)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def pred_from_probs(probs: Dict[str, float]) -> Tuple[Optional[str], float]:
    if not probs:
        return (None, 0.0)
    lab, conf = max(probs.items(), key=lambda kv: kv[1])
    return (lab, float(conf))


# ---------------------------------------------------------------------------
# Skip lists + human ground truth (human events only)
# ---------------------------------------------------------------------------

def load_skip_ids(root: Path) -> set:
    skip = set(SAFETY_HOLD)
    data = root / "data"
    if data.is_dir():
        for pat in ("removed_*.txt", "*license*exclu*", "license_exclusions*.txt", "exclusions*.txt"):
            for p in data.glob(pat):
                try:
                    for line in p.read_text(encoding="utf-8").splitlines():
                        s = line.strip()
                        if s and not s.startswith("#") and s.startswith("wv_"):
                            skip.add(s.split(",")[0].split()[0])
                except OSError:
                    pass
    # human child / exclude flags from events
    ev_dir = root / "annotations" / "events"
    if ev_dir.is_dir():
        for p in ev_dir.glob("*.jsonl"):
            if p.name == "deploytest.jsonl":
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                field = e.get("field")
                val = str(e.get("value") or "").lower()
                iid = e.get("image_id")
                if not iid:
                    continue
                if field in ("exclude", "exclusion", "safety") and val in (
                    "child", "children", "safety_hold", "removed", "license", "license_blocked",
                ):
                    skip.add(iid)
                if field == "child" and val in ("yes", "true", "child"):
                    skip.add(iid)
    return skip


def load_human_gt(root: Path) -> Tuple[dict, dict, dict]:
    """Return (per_image fold, adj, gt) where gt[field][image_id] = {value, agreement, quality, status}.

    Only human events (shaaz/samir/devin etc.). Auto / model_consensus files are
    not in annotations/events/*.jsonl of this form — we also skip deploytest.
    """
    import wvcore

    events: List[dict] = []
    ev_dir = root / "annotations" / "events"
    for p in ev_dir.glob("*.jsonl"):
        if p.name == "deploytest.jsonl":
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    per, adj = wvcore.fold(events)
    gt: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for iid in per:
        st = wvcore.image_status(iid, per, adj)
        st11 = wvcore.image_status_v11(iid, per, adj)
        for field, st_use in (
            ("device_type", st),
            ("occupied", st),
            ("chair_subtype", st11),
            ("device_use_case", st11),
        ):
            fd = st_use["fields"].get(field)
            if not fd or fd[0] in (None, "", []):
                continue
            value, agreement = fd[0], fd[1]
            quality = agreement in ("consensus", "majority", "adjudicated") or (
                st_use.get("status") == "final" and agreement in ("consensus", "majority", "adjudicated")
            )
            # "quality" = multi-annotator agreed (or adjudicated). status==final alone
            # with a single label is still a human label, just not multi-agreed.
            multi = agreement in ("consensus", "majority", "adjudicated")
            gt[field][iid] = {
                "value": canon_device(value) if field == "device_type" else value,
                "agreement": agreement,
                "multi": multi,
                "status": st_use.get("status"),
            }
    return per, adj, gt


def resolve_image(image_id: str, image_root: Path, manifest_index: dict) -> Optional[Path]:
    recs = manifest_index.get(image_id) or []
    # Prefer unblurred candidates, then other non-excluded unblurred, then blurred.
    order = ("candidates", "review_misc", "splits_v1.1", "release_v1.0", "candidates_blurred")
    by_sub = {r["subfolder"]: r for r in recs}
    for sub in order:
        r = by_sub.get(sub)
        if not r:
            continue
        p = image_root / r["subfolder"] / r["filename"]
        if p.is_file():
            return p
    # filename fallbacks
    for sub in ("candidates", "candidates_blurred"):
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = image_root / sub / f"{image_id}{ext}"
            if p.is_file():
                return p
    return None


def load_manifest(path: Path) -> Tuple[List[dict], dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    idx: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        idx[r["image_id"]].append(r)
    return rows, idx


def candidate_pool_ids(manifest_rows: List[dict], skip: set, human_dt: set) -> List[str]:
    """Images we may label: in candidates/, not skipped, no human device_type."""
    ids = []
    seen = set()
    excluded_subs = {
        "needs_child_check", "needs_child_check_blurred",
        "excluded_license", "excluded_unclear",
    }
    in_candidates = set()
    in_excluded_folder = set()
    for r in manifest_rows:
        iid = r["image_id"]
        sub = r.get("subfolder") or ""
        if sub == "candidates":
            in_candidates.add(iid)
        if sub in excluded_subs:
            in_excluded_folder.add(iid)
    for iid in sorted(in_candidates):
        if iid in seen or iid in skip or iid in human_dt or iid in in_excluded_folder:
            continue
        seen.add(iid)
        ids.append(iid)
    return ids


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

class EmbCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS emb (image_id TEXT, sha1 TEXT, model TEXT, emb BLOB, PRIMARY KEY(image_id, model, sha1))"
        )
        self.conn.commit()

    def get(self, image_id: str, model: str, sha1: str) -> Optional["np.ndarray"]:
        import numpy as np
        row = self.conn.execute(
            "SELECT emb FROM emb WHERE image_id=? AND model=? AND sha1=?",
            (image_id, model, sha1),
        ).fetchone()
        if not row:
            return None
        return np.load(io.BytesIO(row[0]), allow_pickle=False)

    def put(self, image_id: str, model: str, sha1: str, arr: "np.ndarray") -> None:
        import numpy as np
        buf = io.BytesIO()
        np.save(buf, arr.astype(np.float32, copy=False))
        self.conn.execute(
            "INSERT OR REPLACE INTO emb(image_id, sha1, model, emb) VALUES (?,?,?,?)",
            (image_id, sha1, model, buf.getvalue()),
        )

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()


# ---------------------------------------------------------------------------
# Model A: existing SigLIP jsonl
# ---------------------------------------------------------------------------

def load_siglip_preds(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        iid = r.get("image_id")
        sug = r.get("suggestions") or {}
        rec: Dict[str, dict] = {}
        for field in ("device_type", "occupied"):
            sg = sug.get(field) or {}
            probs_raw = sg.get("probs") or {}
            probs = {}
            for k, v in probs_raw.items():
                ck = canon_device(k) if field == "device_type" else k
                if ck == "unclear":
                    continue
                if field == "device_type" and ck not in DEVICE_CLASSES:
                    continue
                if field == "occupied" and ck not in OCCUPIED_CLASSES:
                    continue
                probs[ck] = probs.get(ck, 0.0) + float(v)
            # renormalize if we dropped a class
            s = sum(probs.values())
            if s > 0:
                probs = {k: v / s for k, v in probs.items()}
            lab, conf = pred_from_probs(probs)
            rec[field] = {"value": lab, "confidence": conf, "probs": probs}
        out[iid] = rec
    return out


# ---------------------------------------------------------------------------
# Model B: CLIP ViT-L/14 zero-shot
# ---------------------------------------------------------------------------

def _revision_of(model) -> str:
    cfg = getattr(model, "config", None)
    for attr in ("_commit_hash", "commit_hash"):
        v = getattr(cfg, attr, None) if cfg is not None else None
        if v:
            return str(v)
    v = getattr(model, "name_or_path", None)
    return str(v) if v else "unknown"


def _softmax_from_cosine(sim, temp: float):
    import torch
    logits = sim / max(temp, 1e-6)
    return torch.softmax(logits, dim=-1)



def clip_feat_tensor(out):
    """transformers 5 CLIP get_*_features returns BaseModelOutputWithPooling."""
    import torch
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "image_embeds") and out.image_embeds is not None:
        return out.image_embeds
    if hasattr(out, "text_embeds") and out.text_embeds is not None:
        return out.text_embeds
    if torch.is_tensor(out):
        return out
    raise TypeError(f"unexpected CLIP feature type: {type(out)}")


def local_hf_revision(model_id: str, fallback: str = "unknown") -> str:
    """Read the cached snapshot commit without printing any tokens."""
    safe = "models--" + model_id.replace("/", "--")
    hub = Path.home() / ".cache" / "huggingface" / "hub" / safe
    ref = hub / "refs" / "main"
    try:
        if ref.is_file():
            v = ref.read_text(encoding="utf-8").strip()
            if v:
                return v
    except OSError:
        pass
    snaps = hub / "snapshots"
    if snaps.is_dir():
        dirs = [d.name for d in snaps.iterdir() if d.is_dir()]
        if len(dirs) == 1:
            return dirs[0]
        if dirs:
            return sorted(dirs)[-1]
    return fallback


def load_clip(device: str):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    log(f"loading CLIP {MODEL_B_ID} on {device}")
    model = CLIPModel.from_pretrained(MODEL_B_ID)
    proc = CLIPProcessor.from_pretrained(MODEL_B_ID)
    model.eval()
    model.to(device)
    rev = local_hf_revision(MODEL_B_ID, _revision_of(model))
    log(f"CLIP revision={rev}")
    return model, proc, rev


def clip_prototypes(model, processor, prompts: Dict[str, List[str]], device: str):
    import torch
    names = list(prompts.keys())
    vecs = []
    with torch.no_grad():
        for cls in names:
            texts = prompts[cls]
            inputs = processor(text=texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            emb = clip_feat_tensor(model.get_text_features(**inputs))
            emb = emb / emb.norm(dim=-1, keepdim=True)
            proto = emb.mean(dim=0)
            proto = proto / proto.norm()
            vecs.append(proto)
    return names, torch.stack(vecs, dim=0)


def extract_clip_image_embs(model, processor, paths: List[Path], device: str, batch: int = BATCH):
    """Return list of L2-normalised image embeddings (numpy)."""
    import numpy as np
    import torch
    from PIL import Image
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            chunk = paths[i:i + batch]
            images = []
            for p in chunk:
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            feat = clip_feat_tensor(model.get_image_features(**inputs))
            feat = feat / feat.norm(dim=-1, keepdim=True)
            out.append(feat.detach().cpu().numpy())
            for im in images:
                im.close()
    return np.concatenate(out, axis=0) if out else np.zeros((0, 768), dtype=np.float32)


def zs_probs_from_emb(img_emb, names: Sequence[str], proto_np, temp: float = ZS_TEMP) -> Dict[str, float]:
    import numpy as np
    sim = img_emb @ proto_np.T
    logits = sim / max(temp, 1e-6)
    logits = logits - logits.max()
    ex = np.exp(logits)
    pr = ex / ex.sum()
    return {names[i]: float(pr[i]) for i in range(len(names))}


# ---------------------------------------------------------------------------
# Model C: DINOv2 features
# ---------------------------------------------------------------------------

def load_dinov2(device: str):
    from transformers import AutoImageProcessor, AutoModel
    log(f"loading DINOv2 {MODEL_C_ID} on {device}")
    model = AutoModel.from_pretrained(MODEL_C_ID)
    proc = AutoImageProcessor.from_pretrained(MODEL_C_ID)
    model.eval()
    model.to(device)
    rev = local_hf_revision(MODEL_C_ID, _revision_of(model))
    log(f"DINOv2 revision={rev}")
    return model, proc, rev


def extract_dino_embs(model, processor, paths: List[Path], device: str, batch: int = BATCH):
    import numpy as np
    import torch
    from PIL import Image
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            chunk = paths[i:i + batch]
            images = []
            for p in chunk:
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            hidden = model(**inputs).last_hidden_state  # [B, T, D]
            feat = hidden[:, 0]  # CLS
            feat = feat / feat.norm(dim=-1, keepdim=True)
            out.append(feat.detach().cpu().numpy())
            for im in images:
                im.close()
    return np.concatenate(out, axis=0) if out else np.zeros((0, 768), dtype=np.float32)


def unload(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Cache-aware embedding extraction for a list of (image_id, path)
# ---------------------------------------------------------------------------

def embed_cached(
    items: List[Tuple[str, Path]],
    cache: EmbCache,
    model_key: str,
    extract_fn,
    model,
    processor,
    device: str,
    batch: int = BATCH,
) -> Dict[str, "np.ndarray"]:
    import numpy as np
    got: Dict[str, Any] = {}
    missing: List[Tuple[str, Path, str]] = []
    for iid, path in items:
        sha = file_sha1(path)
        arr = cache.get(iid, model_key, sha)
        if arr is not None:
            got[iid] = arr
        else:
            missing.append((iid, path, sha))
    log(f"{model_key}: cached {len(got)}, to compute {len(missing)}")
    for i in range(0, len(missing), batch):
        chunk = missing[i:i + batch]
        paths = [c[1] for c in chunk]
        embs = extract_fn(model, processor, paths, device, batch=len(paths))
        for (iid, path, sha), vec in zip(chunk, embs):
            cache.put(iid, model_key, sha, vec)
            got[iid] = vec
        if (i // batch) % 10 == 0:
            cache.commit()
            log(f"{model_key}: computed {min(i + batch, len(missing))}/{len(missing)}")
    cache.commit()
    return got


# ---------------------------------------------------------------------------
# DINOv2 logistic probe + metrics
# ---------------------------------------------------------------------------

def train_probe(X, y, classes: Sequence[str]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    C=1.0,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return pipe


def oof_predict(X, y, classes: Sequence[str], n_folds: int = N_FOLDS, seed: int = 0):
    """Return oof proba array [N, C] aligned to `classes`. Held-out folds only."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    # shrink folds if a class is smaller than n_folds
    counts = Counter(y)
    min_c = min(counts.values()) if counts else 1
    folds = max(2, min(n_folds, min_c))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), len(classes)), dtype=np.float64)
    class_index = {c: i for i, c in enumerate(classes)}
    for fold_i, (tr, te) in enumerate(skf.split(X, y)):
        pipe = train_probe(X[tr], y[tr], classes)
        proba = pipe.predict_proba(X[te])
        names = list(pipe.classes_)
        for j, name in enumerate(names):
            oof[te, class_index[name]] = proba[:, j]
        log(f"  DINO fold {fold_i + 1}/{folds} test={len(te)}")
    return oof, folds


def classification_report_dict(y_true: List[str], y_pred: List[str], classes: Sequence[str]) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    acc = float(accuracy_score(y_true, y_pred)) if y_true else None
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(classes), zero_division=0
    )
    per = {}
    for i, c in enumerate(classes):
        per[c] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
    return {"accuracy": acc, "n": len(y_true), "per_class": per}


def consensus_sweep(
    rows: List[dict],
    field: str,
    classes: Sequence[str],
    target: float = TARGET_PRECISION,
) -> dict:
    """rows: {iid, y_true, a_value, a_conf, b_value, b_conf, c_value, c_conf}.

    Returns sweep table + chosen threshold (max coverage among those with
    precision >= target). If none reach the target, chosen is None.
    """
    sweep = []
    chosen = None
    for t in THRESH_GRID:
        agreed = []
        for r in rows:
            av, bv, cv = r["a_value"], r["b_value"], r["c_value"]
            if av is None or bv is None or cv is None:
                continue
            if av != bv or av != cv:
                continue
            if av not in classes:
                continue
            if r["a_conf"] < t or r["b_conf"] < t or r["c_conf"] < t:
                continue
            agreed.append((av, r["y_true"]))
        n = len(agreed)
        k = sum(1 for pred, yt in agreed if pred == yt)
        p, lo, hi = wilson_ci(k, n)
        coverage = n / len(rows) if rows else 0.0
        rec = {
            "threshold": t,
            "n_agreed": n,
            "n_correct": k,
            "n_eval": len(rows),
            "precision": p,
            "precision_ci95_lo": lo,
            "precision_ci95_hi": hi,
            "coverage": coverage,
        }
        # per-class precision/recall on agreed set
        if n:
            yt = [x[1] for x in agreed]
            yp = [x[0] for x in agreed]
            rec["per_class"] = classification_report_dict(yt, yp, classes)["per_class"]
        sweep.append(rec)
    # Prefer a threshold whose 95% Wilson lower bound is also >= target
    # (when any such threshold exists); otherwise the highest-coverage
    # threshold whose point precision is >= target.
    qualified = [r for r in sweep if r.get("precision") is not None and r["precision"] + 1e-12 >= target]
    qualified_ci = [r for r in qualified if (r.get("precision_ci95_lo") or 0) + 1e-12 >= target]
    pool = qualified_ci or qualified
    rule = "wilson_lo>=target" if qualified_ci else ("point>=target" if qualified else "none")
    chosen = None
    if pool:
        chosen = max(pool, key=lambda r: (r["coverage"], r.get("precision_ci95_lo") or 0, r.get("precision") or 0))
    return {"sweep": sweep, "chosen": chosen, "target": target, "selection_rule": rule}


def per_model_metrics(rows: List[dict], who: str, classes: Sequence[str]) -> dict:
    yt = [r["y_true"] for r in rows]
    yp = [r[f"{who}_value"] for r in rows]
    # map missing to a dummy that will count as wrong
    yp2 = [v if v is not None else "__none__" for v in yp]
    return classification_report_dict(yt, yp2, classes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def pick_device(arg: Optional[str]) -> str:
    if arg:
        return arg
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def open_image_items(
    ids: Sequence[str],
    image_root: Path,
    manifest_index: dict,
) -> List[Tuple[str, Path]]:
    items = []
    missing = 0
    for iid in ids:
        p = resolve_image(iid, image_root, manifest_index)
        if p is None:
            missing += 1
            continue
        items.append((iid, p))
    if missing:
        log(f"missing image files: {missing}")
    return items


def build_rows_for_field(
    field: str,
    eval_ids: List[str],
    y_true: Dict[str, str],
    siglip: Dict[str, dict],
    clip_probs: Dict[str, Dict[str, dict]],
    dino_probs: Dict[str, Dict[str, float]],
) -> List[dict]:
    rows = []
    for iid in eval_ids:
        yt = y_true.get(iid)
        if yt is None:
            continue
        a = (siglip.get(iid) or {}).get(field) or {}
        b = (clip_probs.get(iid) or {}).get(field) or {}
        c_probs = dino_probs.get(iid) or {}
        cv, cc = pred_from_probs(c_probs)
        rows.append(
            {
                "image_id": iid,
                "y_true": yt,
                "a_value": a.get("value"),
                "a_conf": float(a.get("confidence") or 0.0),
                "a_probs": a.get("probs") or {},
                "b_value": b.get("value"),
                "b_conf": float(b.get("confidence") or 0.0),
                "b_probs": b.get("probs") or {},
                "c_value": cv,
                "c_conf": float(cc),
                "c_probs": c_probs,
            }
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("validate", "infer", "all"), default="all")
    ap.add_argument("--limit", type=int, default=0, help="cap unlabeled inference (0 = all)")
    ap.add_argument("--image-root", type=Path, default=IMAGE_ROOT_DEFAULT)
    ap.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "derived")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "var" / "ai_3model_cache")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--skip-occupied", action="store_true")
    args = ap.parse_args(argv)

    ensure_hf_token()
    device = pick_device(args.device)
    log(f"phase={args.phase} device={device} root={ROOT}")
    _job("start", 0, 0, message=f"phase={args.phase} device={device}")
    try:
        import torch
        if device == "cuda":
            log(f"cuda device={torch.cuda.get_device_name(0)} vram={torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GiB")
    except Exception as e:
        log(f"torch info: {e}")

    skip = load_skip_ids(ROOT)
    log(f"skip ids (holds/removed/license/child): {len(skip)}")
    _per, _adj, gt = load_human_gt(ROOT)
    human_dt_all = {
        iid: rec
        for iid, rec in gt.get("device_type", {}).items()
        if rec["value"] and rec["value"] != "unclear" and iid not in skip
    }
    human_dt_multi = {iid: rec for iid, rec in human_dt_all.items() if rec["multi"]}
    human_occ_all = {
        iid: rec
        for iid, rec in gt.get("occupied", {}).items()
        if rec["value"] in OCCUPIED_CLASSES and iid not in skip
    }
    human_occ_multi = {iid: rec for iid, rec in human_occ_all.items() if rec["multi"]}
    log(
        f"human GT device_type all={len(human_dt_all)} multi={len(human_dt_multi)} "
        f"occupied all={len(human_occ_all)} multi={len(human_occ_multi)} "
        f"device_use_case={len(gt.get('device_use_case') or {})} "
        f"chair_subtype={len(gt.get('chair_subtype') or {})}"
    )
    if not gt.get("device_use_case"):
        log("device_use_case: no human labels; will not emit")
    if len(gt.get("chair_subtype") or {}) < 30:
        log("chair_subtype: too few human labels; will not emit")

    manifest_rows, manifest_index = load_manifest(args.manifest)
    human_dt_ids = set(gt.get("device_type", {}))  # includes unclear; any human label
    pool = candidate_pool_ids(manifest_rows, skip, human_dt_ids)
    log(f"unlabeled candidate pool: {len(pool)}")

    # validation + (maybe) inference id set we need embeddings for
    need_ids = set(human_dt_all) | set(human_occ_all)
    infer_ids = list(pool)
    if args.limit and args.phase in ("infer", "all"):
        infer_ids = infer_ids[: args.limit]
    if args.phase in ("infer", "all"):
        need_ids |= set(infer_ids)
    # always embed human GT so we can validate even in infer-only reruns
    need_ids |= set(human_dt_all) | set(human_occ_all)

    items = open_image_items(sorted(need_ids), args.image_root, manifest_index)
    log(f"images resolved: {len(items)} / {len(need_ids)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache = EmbCache(args.cache_dir / "embeddings.sqlite")

    # ----- Model A: SigLIP (reuse jsonl; do not load the weights unless a few are missing) -----
    siglip = load_siglip_preds(MODEL_A_JSONL)
    log(f"SigLIP jsonl predictions: {len(siglip)}")
    missing_a = [iid for iid, _ in items if iid not in siglip]
    model_a_rev = MODEL_A_REV
    if missing_a:
        log(f"SigLIP missing for {len(missing_a)} needed images; will leave those without model A")

    # ----- Model B: CLIP embeddings + zero-shot -----
    clip_rev = "unloaded"
    clip_embs: Dict[str, Any] = {}
    try:
        import numpy as np
        model_b, proc_b, clip_rev = load_clip(device)
        clip_embs = embed_cached(
            items, cache, "clip-vit-large-patch14", extract_clip_image_embs,
            model_b, proc_b, device, batch=args.batch,
        )
        # text prototypes once
        dev_names, dev_proto = clip_prototypes(model_b, proc_b, PROMPTS_DEVICE, device)
        occ_names, occ_proto = clip_prototypes(model_b, proc_b, PROMPTS_OCCUPIED, device)
        dev_proto_np = dev_proto.detach().cpu().numpy()
        occ_proto_np = occ_proto.detach().cpu().numpy()
        unload(model_b, proc_b)
        model_b = proc_b = None
    except Exception:
        log("CLIP failed:\n" + traceback.format_exc())
        raise

    clip_probs: Dict[str, Dict[str, dict]] = {}
    for iid, vec in clip_embs.items():
        dprobs = zs_probs_from_emb(vec, dev_names, dev_proto_np, ZS_TEMP)
        oprobs = zs_probs_from_emb(vec, occ_names, occ_proto_np, ZS_TEMP)
        dv, dc = pred_from_probs(dprobs)
        ov, oc = pred_from_probs(oprobs)
        clip_probs[iid] = {
            "device_type": {"value": dv, "confidence": dc, "probs": dprobs},
            "occupied": {"value": ov, "confidence": oc, "probs": oprobs},
        }
    log(f"CLIP zero-shot scored {len(clip_probs)} images")

    # ----- Model C: DINOv2 embeddings + probe -----
    dino_rev = "unloaded"
    dino_embs: Dict[str, Any] = {}
    try:
        model_c, proc_c, dino_rev = load_dinov2(device)
        dino_embs = embed_cached(
            items, cache, "dinov2-base", extract_dino_embs,
            model_c, proc_c, device, batch=args.batch,
        )
        unload(model_c, proc_c)
        model_c = proc_c = None
    except Exception:
        log("DINOv2 failed:\n" + traceback.format_exc())
        raise
    cache.close()
    log(f"DINOv2 embeddings: {len(dino_embs)}")

    import numpy as np

    # Train / OOF for device_type
    def matrix_for(id_list: List[str], embs: Dict[str, Any]):
        xs, keep = [], []
        for iid in id_list:
            if iid in embs:
                xs.append(embs[iid])
                keep.append(iid)
        if not xs:
            return np.zeros((0, 1)), []
        return np.stack(xs, axis=0), keep

    def run_field(field: str, classes: Sequence[str], human_all: dict, human_multi: dict) -> dict:
        # IDs that have embeddings from all three (A may be missing)
        eval_all = [
            iid for iid in human_all
            if iid in siglip and iid in clip_probs and iid in dino_embs
        ]
        y_all = {iid: human_all[iid]["value"] for iid in eval_all if human_all[iid]["value"] in classes}
        eval_all = [iid for iid in eval_all if iid in y_all]
        # OOF DINO on these IDs
        X, keep = matrix_for(eval_all, dino_embs)
        y = np.array([y_all[iid] for iid in keep])
        log(f"{field}: training DINO probe on {len(keep)} human labels, class counts={dict(Counter(y.tolist()))}")
        oof, nfolds = oof_predict(X, y, classes)
        dino_oof = {iid: {classes[j]: float(oof[i, j]) for j in range(len(classes))} for i, iid in enumerate(keep)}

        rows_all = build_rows_for_field(field, keep, y_all, siglip, clip_probs, dino_oof)
        rows_multi = [r for r in rows_all if r["image_id"] in human_multi]
        log(f"{field}: eval rows all={len(rows_all)} multi={len(rows_multi)}")

        metrics = {
            "n_human_all": len(rows_all),
            "n_human_multi": len(rows_multi),
            "n_folds": nfolds,
            "class_counts_all": dict(Counter(r["y_true"] for r in rows_all)),
            "class_counts_multi": dict(Counter(r["y_true"] for r in rows_multi)),
            "per_model_all": {
                "siglip": per_model_metrics(rows_all, "a", classes),
                "clip": per_model_metrics(rows_all, "b", classes),
                "dinov2_oof": per_model_metrics(rows_all, "c", classes),
            },
            "per_model_multi": {
                "siglip": per_model_metrics(rows_multi, "a", classes) if rows_multi else None,
                "clip": per_model_metrics(rows_multi, "b", classes) if rows_multi else None,
                "dinov2_oof": per_model_metrics(rows_multi, "c", classes) if rows_multi else None,
            },
            "consensus_all": consensus_sweep(rows_all, field, classes, TARGET_PRECISION),
            "consensus_multi": consensus_sweep(rows_multi, field, classes, TARGET_PRECISION),
        }
        # Threshold is chosen on ALL human labels (more power). Also record
        # whether the same threshold meets the bar on the multi-agreed subset.
        chosen = metrics["consensus_all"]["chosen"]
        emit = chosen is not None
        metrics["emit"] = emit
        metrics["chosen_threshold"] = None if chosen is None else chosen["threshold"]
        metrics["chosen"] = chosen
        # Full-data probe for unlabeled inference (unlabeled IDs are disjoint
        # from this training set, so this is not leaking into validation).
        full_pipe = train_probe(X, y, classes)
        metrics["_full_pipe"] = full_pipe
        metrics["_classes"] = list(classes)
        return metrics

    val: Dict[str, Any] = {
        "run_timestamp": now_iso(),
        "timezone": TZ_NAME,
        "models": {
            "model_a": {
                "family": "SigLIP 2 (contrastive / zero-shot)",
                "id": MODEL_A_ID,
                "tag": MODEL_A_TAG,
                "revision": model_a_rev,
                "method": "zero-shot prompt ensemble, softmax temp=0.01 on cosine; predictions reused from existing jsonl",
            },
            "model_b": {
                "family": "OpenAI CLIP (contrastive / zero-shot, distinct from SigLIP)",
                "id": MODEL_B_ID,
                "revision": clip_rev,
                "method": "zero-shot prompt ensemble (mean of L2-normalised text embeddings per option), softmax temp=0.01 on cosine",
            },
            "model_c": {
                "family": "DINOv2 (self-supervised ViT) + logistic-regression probe",
                "id": MODEL_C_ID,
                "revision": dino_rev,
                "method": "CLS features, L2-normalised, StandardScaler + LogisticRegression (class_weight=balanced); validation uses stratified k-fold OOF so no image is scored with a probe trained on itself",
            },
        },
        "notes": {
            "device_use_case": "no human labels folded; field not emitted",
            "chair_subtype": "too few human labels for a 97% precision claim; field not emitted",
            "unclear": "never emitted; human=unclear images excluded from device_type validation",
            "ground_truth": "human events only (fold of annotations/events, excluding deploytest). auto_approved / model_consensus are not used as GT. 'all' = any human label (single or multi-agreed); 'multi' = consensus/majority/adjudicated.",
            "held_out": "DINOv2 metrics and 3-way consensus on the human set use out-of-fold probe predictions. SigLIP and CLIP are zero-shot (not trained on these labels).",
        },
        "skip_count": len(skip),
        "pool_unlabeled_candidates": len(pool),
        "device": device,
    }

    dt_metrics = run_field("device_type", DEVICE_CLASSES, human_dt_all, human_dt_multi)
    # strip unserialisable
    dt_pipe = dt_metrics.pop("_full_pipe")
    dt_classes = dt_metrics.pop("_classes")
    val["device_type"] = dt_metrics

    occ_pipe = None
    occ_classes = list(OCCUPIED_CLASSES)
    if not args.skip_occupied and human_occ_all:
        occ_metrics = run_field("occupied", OCCUPIED_CLASSES, human_occ_all, human_occ_multi)
        occ_pipe = occ_metrics.pop("_full_pipe")
        occ_classes = occ_metrics.pop("_classes")
        val["occupied"] = occ_metrics
    else:
        val["occupied"] = {"emit": False, "reason": "skipped"}

    def brief(m):
        if not m:
            return None
        ch = m.get("chosen")
        return {
            "emit": m.get("emit"),
            "threshold": m.get("chosen_threshold"),
            "precision": None if not ch else ch.get("precision"),
            "ci95": None if not ch else [ch.get("precision_ci95_lo"), ch.get("precision_ci95_hi")],
            "n_agreed": None if not ch else ch.get("n_agreed"),
            "coverage": None if not ch else ch.get("coverage"),
            "per_model_acc_all": {
                k: (m.get("per_model_all") or {}).get(k, {}) and (m["per_model_all"][k] or {}).get("accuracy")
                for k in ("siglip", "clip", "dinov2_oof")
            },
        }

    log("VALIDATION device_type: " + json.dumps(brief(dt_metrics), default=str))
    log("VALIDATION occupied: " + json.dumps(brief(val.get("occupied")), default=str))

    # Write summary early (so a crash during infer still leaves metrics)
    summary_path = args.out_dir / "ai_labels_3model_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(val, f, indent=2, default=str)
    log(f"wrote {summary_path}")

    if args.phase == "validate":
        log("validate-only done")
        return 0

    emit_dt = bool(dt_metrics.get("emit"))
    emit_occ = bool((val.get("occupied") or {}).get("emit"))
    if not emit_dt and not emit_occ:
        log("no field met the precision bar; will still write disagreements + empty labels CSV")

    t_dt = dt_metrics.get("chosen_threshold")
    t_occ = (val.get("occupied") or {}).get("chosen_threshold")

    # DINO full-data predictions for unlabeled
    infer_keep = [iid for iid in infer_ids if iid in dino_embs and iid in clip_probs]
    missing_models = [iid for iid in infer_ids if iid not in infer_keep]
    log(f"inference set: {len(infer_keep)} with all embeddings/preds ({len(missing_models)} missing a model)")

    dino_infer_dt: Dict[str, Dict[str, float]] = {}
    dino_infer_occ: Dict[str, Dict[str, float]] = {}
    if infer_keep:
        X_inf, keep_inf = matrix_for(infer_keep, dino_embs)
        proba = dt_pipe.predict_proba(X_inf)
        names = list(dt_pipe.classes_)
        for i, iid in enumerate(keep_inf):
            dino_infer_dt[iid] = {names[j]: float(proba[i, j]) for j in range(len(names))}
        if emit_occ and occ_pipe is not None:
            proba_o = occ_pipe.predict_proba(X_inf)
            names_o = list(occ_pipe.classes_)
            for i, iid in enumerate(keep_inf):
                dino_infer_occ[iid] = {names_o[j]: float(proba_o[i, j]) for j in range(len(names_o))}

    run_ts = val["run_timestamp"]
    label_rows = []
    disagree_rows = []

    def consider(field: str, iid: str, a, b, c_probs, thresh, classes) -> None:
        if a is None or b is None or not c_probs:
            disagree_rows.append({
                "image_id": iid, "field": field, "value": "",
                "label_source": "model_consensus", "agreed": False,
                "reason": "missing_model", "threshold": thresh,
                "model_a_id": MODEL_A_ID, "model_a_revision": model_a_rev,
                "model_a_value": None if not a else a.get("value"),
                "model_a_confidence": None if not a else a.get("confidence"),
                "model_b_id": MODEL_B_ID, "model_b_revision": clip_rev,
                "model_b_value": None if not b else b.get("value"),
                "model_b_confidence": None if not b else b.get("confidence"),
                "model_c_id": MODEL_C_ID, "model_c_revision": dino_rev,
                "model_c_value": None, "model_c_confidence": None,
                "run_timestamp": run_ts,
            })
            return
        cv, cc = pred_from_probs(c_probs)
        av, ac = a.get("value"), float(a.get("confidence") or 0)
        bv, bc = b.get("value"), float(b.get("confidence") or 0)
        same = av == bv == cv and av in classes
        above = same and ac >= thresh and bc >= thresh and cc >= thresh
        rec = {
            "image_id": iid, "field": field, "value": av if above else "",
            "label_source": "model_consensus", "agreed": bool(above),
            "reason": "consensus" if above else ("low_confidence" if same else "disagreement"),
            "threshold": thresh,
            "model_a_id": MODEL_A_ID, "model_a_revision": model_a_rev,
            "model_a_value": av, "model_a_confidence": round(ac, 6),
            "model_b_id": MODEL_B_ID, "model_b_revision": clip_rev,
            "model_b_value": bv, "model_b_confidence": round(bc, 6),
            "model_c_id": MODEL_C_ID, "model_c_revision": dino_rev,
            "model_c_value": cv, "model_c_confidence": round(float(cc), 6),
            "run_timestamp": run_ts,
        }
        if above:
            label_rows.append(rec)
        else:
            disagree_rows.append(rec)

    for iid in infer_keep:
        if emit_dt and t_dt is not None:
            consider(
                "device_type", iid,
                (siglip.get(iid) or {}).get("device_type"),
                (clip_probs.get(iid) or {}).get("device_type"),
                dino_infer_dt.get(iid) or {},
                t_dt, DEVICE_CLASSES,
            )
        if emit_occ and t_occ is not None:
            consider(
                "occupied", iid,
                (siglip.get(iid) or {}).get("occupied"),
                (clip_probs.get(iid) or {}).get("occupied"),
                dino_infer_occ.get(iid) or {},
                t_occ, occ_classes,
            )
    for iid in missing_models:
        if emit_dt:
            consider("device_type", iid, (siglip.get(iid) or {}).get("device_type"),
                     (clip_probs.get(iid) or {}).get("device_type"), {}, t_dt, DEVICE_CLASSES)

    cols = [
        "image_id", "field", "value", "label_source", "agreed", "reason", "threshold",
        "model_a_id", "model_a_revision", "model_a_value", "model_a_confidence",
        "model_b_id", "model_b_revision", "model_b_value", "model_b_confidence",
        "model_c_id", "model_c_revision", "model_c_value", "model_c_confidence",
        "run_timestamp",
    ]
    labels_path = args.out_dir / "ai_labels_3model.csv"
    disagree_path = args.out_dir / "ai_labels_3model_disagreements.csv"

    def write_csv(path: Path, rows: List[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    write_csv(labels_path, label_rows)
    write_csv(disagree_path, disagree_rows)

    # counts
    by_field_value = Counter((r["field"], r["value"]) for r in label_rows)
    val["inference"] = {
        "n_pool": len(pool),
        "n_attempted": len(infer_ids),
        "n_with_all_models": len(infer_keep),
        "n_labels_emitted": len(label_rows),
        "n_disagreements_or_abstain": len(disagree_rows),
        "emitted_by_field_value": {f"{k[0]}={k[1]}": v for k, v in sorted(by_field_value.items())},
        "labels_csv": str(labels_path),
        "disagreements_csv": str(disagree_path),
        "limit": args.limit or None,
        "finished": True,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(val, f, indent=2, default=str)

    log(f"wrote {labels_path} ({len(label_rows)} labels)")
    log(f"wrote {disagree_path} ({len(disagree_rows)} disagreements/abstain)")
    log(f"emitted {dict(by_field_value)}")
    log("done")
    _job("done", 1, 1, status="done", message="finished")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
