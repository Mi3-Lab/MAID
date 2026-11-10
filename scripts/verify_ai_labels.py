#!/usr/bin/env python3
"""Verify prior assistant/model labels against the 3-model panel and human votes.

Never writes annotations/events/*.jsonl, never stores the AI vote as a human
label, and never changes data/splits_v11_with_auto.csv. Outputs (uncommitted):

  derived/ai_label_verification.csv
  derived/ai_label_verification_summary.json

Majority rule (Samir): each human labeler has one vote; the 3-model consensus
counts as one vote ('ai'). If at least two voters agree and they form a
strict majority, the label proceeds (verdict=majority_pass, basis=ai+human
or human+human or human+human+ai). An AI-vs-single-human disagreement is a
tie and goes to human review (the human is not overridden). AI-only labels
(no human voters) still use confirmed / model_disputed / unverifiable.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "server"))

import ai_jobs
import ai_labels_3model as M
from ai_majority import majority_outcome, ai_consensus_value, decide_verdict

JOB = "verify_ai_labels"
OUT_CSV = ROOT / "derived" / "ai_label_verification.csv"
OUT_JSON = ROOT / "derived" / "ai_label_verification_summary.json"
SPLIT_PATH = ROOT / "data" / "splits_v11_with_auto.csv"
AUTO_APPROVED = ROOT / "var" / "assist" / "auto_approved.jsonl"
ASSIST_JSONL = ROOT / "var" / "assist" / "assistant_visual_review_attr-v1.jsonl"
CSV_3MODEL = ROOT / "derived" / "ai_labels_3model.csv"
SUMMARY_3MODEL = ROOT / "derived" / "ai_labels_3model_summary.json"

VERIFY_FIELDS = ("device_type", "occupied")
IMAGE_EXTRA_ROOTS = [
    Path(r"C:\Users\kamoo\wheelview_all_images_new"),
    Path(r"C:\Users\kamoo\wheelview\images"),
    Path(r"C:\Users\kamoo\wheelview\images_blurred"),
    ROOT / "images",
    ROOT / "images_blurred",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def human_votes_for(per, adj, iid: str, field: str) -> List[Tuple[str, str]]:
    votes: List[Tuple[str, str]] = []
    anns = per.get(iid) or {}
    if not isinstance(anns, dict):
        return votes
    for ann, fv in anns.items():
        if not isinstance(fv, dict):
            continue
        if str(ann).lower() in ("deploytest", "assistant", "auto", "model", "ai"):
            continue
        v = fv.get(field)
        if field == "device_type":
            v = M.canon_device(v) if v else v
        if v in (None, "", []):
            continue
        if isinstance(v, list):
            continue
        votes.append((str(ann), str(v)))
    adjrec = adj.get((iid, field)) if adj else None
    if adjrec:
        val, who = adjrec[0], adjrec[1] if len(adjrec) > 1 else None
        if field == "device_type":
            val = M.canon_device(val) if val else val
        if who and val not in (None, "", []):
            cv = str(val)
            key = str(who).strip().lower()
            replaced = False
            out = []
            for a, v in votes:
                if a.strip().lower() == key:
                    out.append((a, cv))
                    replaced = True
                else:
                    out.append((a, v))
            if not replaced:
                out.append((str(who), cv))
            votes = out
    return votes


def folded_human(gt: dict, iid: str, field: str) -> Optional[str]:
    rec = (gt.get(field) or {}).get(iid)
    if not rec:
        return None
    return rec.get("value")


def load_thresholds() -> Dict[str, float]:
    out = {"device_type": 0.60, "occupied": 0.60}
    if SUMMARY_3MODEL.is_file():
        try:
            s = json.loads(SUMMARY_3MODEL.read_text(encoding="utf-8"))
            for field in ("device_type", "occupied"):
                t = (s.get(field) or {}).get("chosen_threshold")
                if t is not None:
                    out[field] = float(t)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return out


def collect_prior_labels() -> List[dict]:
    """One row per (source, image_id, field) prior AI/assistant label."""
    items: List[dict] = []

    if AUTO_APPROVED.is_file():
        for line in AUTO_APPROVED.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            iid = r.get("image_id")
            val = M.canon_device(r.get("device_type"))
            if not iid or not val:
                continue
            items.append({
                "source": "auto_approved",
                "source_file": "var/assist/auto_approved.jsonl",
                "image_id": iid,
                "field": "device_type",
                "value": val,
                "prior_status": r.get("status") or "auto_approved",
                "prior_subtype": r.get("subtype"),
            })

    if ASSIST_JSONL.is_file():
        for line in ASSIST_JSONL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            iid = r.get("image_id")
            sug = r.get("suggestions") or {}
            if not iid:
                continue
            for field in VERIFY_FIELDS:
                sg = sug.get(field) or {}
                val = sg.get("value") if isinstance(sg, dict) else sg
                if field == "device_type":
                    val = M.canon_device(val)
                if val in (None, "", []):
                    continue
                items.append({
                    "source": "assist",
                    "source_file": "var/assist/assistant_visual_review_attr-v1.jsonl",
                    "image_id": iid,
                    "field": field,
                    "value": str(val),
                    "prior_status": "assistant_suggestion",
                    "prior_confidence": (sg.get("confidence") if isinstance(sg, dict) else None),
                })

    if CSV_3MODEL.is_file():
        with CSV_3MODEL.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if not r.get("image_id") or not r.get("field") or not r.get("value"):
                    continue
                items.append({
                    "source": "3model",
                    "source_file": "derived/ai_labels_3model.csv",
                    "image_id": r["image_id"],
                    "field": r["field"],
                    "value": r["value"],
                    "prior_status": r.get("label_source") or "model_consensus",
                    "cached_a_value": r.get("model_a_value"),
                    "cached_a_conf": r.get("model_a_confidence"),
                    "cached_b_value": r.get("model_b_value"),
                    "cached_b_conf": r.get("model_b_confidence"),
                    "cached_c_value": r.get("model_c_value"),
                    "cached_c_conf": r.get("model_c_confidence"),
                })
    return items


def load_split_index() -> Dict[str, dict]:
    out = {}
    if not SPLIT_PATH.is_file():
        return out
    with SPLIT_PATH.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["image_id"]] = r
    return out


def resolve_image_any(image_id: str, image_root: Path, manifest_index: dict) -> Optional[Path]:
    p = M.resolve_image(image_id, image_root, manifest_index)
    if p:
        return p
    for root in IMAGE_EXTRA_ROOTS:
        if not root.exists():
            continue
        # flat <id>.jpg (lab image dir)
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            cand = root / f"{image_id}{ext}"
            if cand.is_file():
                return cand
        # pack-style subfolders
        for sub in ("candidates", "candidates_blurred", "review_misc", "splits_v1.1", "release_v1.0"):
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                cand = root / sub / f"{image_id}{ext}"
                if cand.is_file():
                    return cand
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--image-root", type=Path, default=M.IMAGE_ROOT_DEFAULT)
    ap.add_argument("--manifest", type=Path, default=M.MANIFEST_DEFAULT)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=M.BATCH)
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "var" / "ai_3model_cache")
    args = ap.parse_args(argv)

    ai_jobs.log(f"starting {JOB}")
    ai_jobs.update(JOB, "collect", 0, 0, message="collecting prior AI/assistant labels")
    M.ensure_hf_token()
    device = M.pick_device(args.device)
    thresholds = load_thresholds()
    items = collect_prior_labels()
    if args.limit:
        items = items[: args.limit]
    ai_jobs.log(f"collected {len(items)} prior labels; thresholds={thresholds} device={device}")

    ai_jobs.update(JOB, "load_gt", 0, len(items), message="loading human ground truth")
    _per, adj, gt = M.load_human_gt(ROOT)
    skip = M.load_skip_ids(ROOT)
    split_idx = load_split_index()

    need_ids = sorted({it["image_id"] for it in items})
    manifest_rows, manifest_index = ([], {})
    if args.manifest.is_file():
        manifest_rows, manifest_index = M.load_manifest(args.manifest)

    paths: Dict[str, Path] = {}
    missing_img = []
    for iid in need_ids:
        p = resolve_image_any(iid, args.image_root, manifest_index)
        if p:
            paths[iid] = p
        else:
            missing_img.append(iid)
    ai_jobs.log(f"images resolved {len(paths)}/{len(need_ids)}; missing {len(missing_img)}")

    # Embeddings for images we can open (reuse cache)
    items_to_embed = [(iid, paths[iid]) for iid in need_ids if iid in paths]
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache = M.EmbCache(args.cache_dir / "embeddings.sqlite")

    clip_rev = "unloaded"
    dino_rev = "unloaded"
    clip_probs: Dict[str, Dict[str, dict]] = {}
    dino_embs: Dict[str, Any] = {}
    siglip = M.load_siglip_preds(M.MODEL_A_JSONL)
    model_a_rev = M.MODEL_A_REV

    # CLIP
    ai_jobs.update(JOB, "embed_clip", 0, len(items_to_embed), message=f"CLIP embeddings for {len(items_to_embed)} images")
    try:
        import numpy as np
        model_b, proc_b, clip_rev = M.load_clip(device)
        # wrap extract to report progress: embed_cached already logs; we pulse the job file
        clip_embs = M.embed_cached(
            items_to_embed, cache, "clip-vit-large-patch14", M.extract_clip_image_embs,
            model_b, proc_b, device, batch=args.batch,
        )
        dev_names, dev_proto = M.clip_prototypes(model_b, proc_b, M.PROMPTS_DEVICE, device)
        occ_names, occ_proto = M.clip_prototypes(model_b, proc_b, M.PROMPTS_OCCUPIED, device)
        dev_proto_np = dev_proto.detach().cpu().numpy()
        occ_proto_np = occ_proto.detach().cpu().numpy()
        M.unload(model_b, proc_b)
        for iid, vec in clip_embs.items():
            dprobs = M.zs_probs_from_emb(vec, dev_names, dev_proto_np, M.ZS_TEMP)
            oprobs = M.zs_probs_from_emb(vec, occ_names, occ_proto_np, M.ZS_TEMP)
            dv, dc = M.pred_from_probs(dprobs)
            ov, oc = M.pred_from_probs(oprobs)
            clip_probs[iid] = {
                "device_type": {"value": dv, "confidence": dc, "probs": dprobs},
                "occupied": {"value": ov, "confidence": oc, "probs": oprobs},
            }
        ai_jobs.log(f"CLIP scored {len(clip_probs)} images rev={clip_rev}")
        ai_jobs.update(JOB, "embed_clip", len(clip_probs), len(items_to_embed), message="CLIP done")
    except Exception:
        ai_jobs.fail(JOB, "CLIP failed: " + traceback.format_exc().splitlines()[-1], stage="embed_clip")
        raise

    # DINOv2
    ai_jobs.update(JOB, "embed_dino", 0, len(items_to_embed), message="DINOv2 embeddings")
    try:
        model_c, proc_c, dino_rev = M.load_dinov2(device)
        dino_embs = M.embed_cached(
            items_to_embed, cache, "dinov2-base", M.extract_dino_embs,
            model_c, proc_c, device, batch=args.batch,
        )
        M.unload(model_c, proc_c)
        ai_jobs.log(f"DINOv2 embeddings {len(dino_embs)} rev={dino_rev}")
        ai_jobs.update(JOB, "embed_dino", len(dino_embs), len(items_to_embed), message="DINOv2 done")
    except Exception:
        ai_jobs.fail(JOB, "DINOv2 failed: " + traceback.format_exc().splitlines()[-1], stage="embed_dino")
        raise
    cache.close()

    # DINO probes: OOF on human-labeled images, full-data probe for the rest
    ai_jobs.update(JOB, "probe", 0, 2, message="training DINOv2 probes (OOF on human set)")
    import numpy as np

    def matrix_for(id_list, embs):
        xs, keep = [], []
        for iid in id_list:
            if iid in embs:
                xs.append(embs[iid])
                keep.append(iid)
        if not xs:
            return np.zeros((0, 1)), []
        return np.stack(xs, axis=0), keep

    dino_probs: Dict[str, Dict[str, Dict[str, float]]] = {"device_type": {}, "occupied": {}}
    dino_used_oof: Dict[str, set] = {"device_type": set(), "occupied": set()}

    for field, classes, human_map in (
        ("device_type", M.DEVICE_CLASSES, {
            iid: rec["value"]
            for iid, rec in (gt.get("device_type") or {}).items()
            if rec.get("value") in M.DEVICE_CLASSES and iid not in skip and iid in dino_embs
        }),
        ("occupied", M.OCCUPIED_CLASSES, {
            iid: rec["value"]
            for iid, rec in (gt.get("occupied") or {}).items()
            if rec.get("value") in M.OCCUPIED_CLASSES and iid not in skip and iid in dino_embs
        }),
    ):
        train_ids = [iid for iid in human_map if iid in dino_embs]
        X, keep = matrix_for(train_ids, dino_embs)
        y = np.array([human_map[iid] for iid in keep])
        ai_jobs.log(f"{field}: DINO probe train n={len(keep)} counts={dict(Counter(y.tolist()))}")
        if len(keep) >= 4 and len(set(y.tolist())) >= 2:
            oof, nfolds = M.oof_predict(X, y, classes)
            for i, iid in enumerate(keep):
                dino_probs[field][iid] = {classes[j]: float(oof[i, j]) for j in range(len(classes))}
                dino_used_oof[field].add(iid)
            pipe = M.train_probe(X, y, classes)
        elif len(keep) >= 1:
            nfolds = 0
            pipe = M.train_probe(X, y, classes)
        else:
            pipe = None
            nfolds = 0
        # unlabeled / not-in-train: full-data probe
        rest = [iid for iid in dino_embs if iid not in dino_probs[field]]
        if pipe is not None and rest:
            Xr, kr = matrix_for(rest, dino_embs)
            if len(kr):
                proba = pipe.predict_proba(Xr)
                names = list(pipe.classes_)
                for i, iid in enumerate(kr):
                    dino_probs[field][iid] = {names[j]: float(proba[i, j]) for j in range(len(names))}
        ai_jobs.log(f"{field}: DINO scored {len(dino_probs[field])} (OOF {len(dino_used_oof[field])}, folds={nfolds})")

    ai_jobs.update(JOB, "score", 0, len(items), message="scoring labels and applying majority rule")

    def model_pack(iid: str, field: str, it: dict):
        a = (siglip.get(iid) or {}).get(field) or {}
        b = (clip_probs.get(iid) or {}).get(field) or {}
        c_probs = (dino_probs.get(field) or {}).get(iid) or {}
        cv, cc = M.pred_from_probs(c_probs) if c_probs else (None, 0.0)
        # fallback to 3model CSV cache if a live pred is missing
        if not a.get("value") and it.get("cached_a_value"):
            a = {"value": it.get("cached_a_value"), "confidence": float(it.get("cached_a_conf") or 0), "probs": {}}
        if not b.get("value") and it.get("cached_b_value"):
            b = {"value": it.get("cached_b_value"), "confidence": float(it.get("cached_b_conf") or 0), "probs": {}}
        if cv is None and it.get("cached_c_value"):
            cv, cc = it.get("cached_c_value"), float(it.get("cached_c_conf") or 0)
        return a, b, cv, cc, c_probs

    rows = []
    run_ts = ai_jobs.now_iso()
    classes_for = {"device_type": M.DEVICE_CLASSES, "occupied": M.OCCUPIED_CLASSES}

    for i, it in enumerate(items):
        iid, field, val = it["image_id"], it["field"], it["value"]
        if field == "device_type":
            val = M.canon_device(val) or val
        thresh = float(thresholds.get(field, 0.60))
        classes = classes_for.get(field) or ()
        a, b, cv, cc, c_probs = model_pack(iid, field, it)
        av, ac = a.get("value"), float(a.get("confidence") or 0)
        bv, bc = b.get("value"), float(b.get("confidence") or 0)
        models_complete = bool(av and bv and cv and iid in paths)
        # if we used CSV fallback without an image, still treat preds as present
        if not (av and bv and cv):
            models_complete = False
        elif iid not in paths and not (it.get("cached_a_value") and it.get("cached_b_value") and it.get("cached_c_value")):
            models_complete = False

        ai_vote = ai_consensus_value(av, ac, bv, bc, cv, cc, thresh, classes) if models_complete or (av and bv and cv) else None
        hvotes = human_votes_for(_per, adj, iid, field)
        maj = majority_outcome(hvotes, ai_vote)
        verdict, action, reason = decide_verdict(
            ai_label=val, ai_vote=ai_vote, models_complete=bool(av and bv and cv) and (iid in paths or it.get("cached_a_value")),
            human_votes=hvotes, majority=maj,
        )
        # If models truly missing and no complete cached preds:
        if not (av and bv and cv) and iid not in paths:
            verdict, action, reason = "unverifiable", "send to human review", "missing_image_or_model"

        folded = folded_human(gt, iid, field)
        sp = split_idx.get(iid) or {}
        in_split_auto = sp.get("label_source") == "model_consensus" and field == "device_type"
        rows.append({
            "image_id": iid,
            "field": field,
            "value": val,
            "source": it["source"],
            "source_file": it["source_file"],
            "label_source": it.get("prior_status") or it["source"],
            "in_training_split": bool(in_split_auto),
            "split": sp.get("split") or "",
            "model_a_id": M.MODEL_A_ID,
            "model_a_revision": model_a_rev,
            "model_a_value": av,
            "model_a_confidence": None if av is None else round(float(ac), 6),
            "model_b_id": M.MODEL_B_ID,
            "model_b_revision": clip_rev,
            "model_b_value": bv,
            "model_b_confidence": None if bv is None else round(float(bc), 6),
            "model_c_id": M.MODEL_C_ID,
            "model_c_revision": dino_rev,
            "model_c_value": cv,
            "model_c_confidence": None if cv is None else round(float(cc), 6),
            "dino_oof": iid in dino_used_oof.get(field, set()),
            "threshold": thresh,
            "ai_vote": ai_vote or "",
            "human_value_folded": folded or "",
            "human_votes": json.dumps([{"id": n, "value": v} for n, v in hvotes], separators=(",", ":")),
            "n_human_voters": len(hvotes),
            "voters": json.dumps(maj.get("voters") or [], separators=(",", ":")),
            "majority_outcome": maj.get("outcome"),
            "majority_value": maj.get("majority_value") or "",
            "majority_basis": maj.get("basis") or "",
            "majority_agreeing": ",".join(maj.get("agreeing_voters") or []),
            "n_agree": maj.get("n_agree"),
            "n_votes": maj.get("n_votes"),
            "verdict": verdict,
            "recommended_action": action,
            "reason": reason,
            "run_timestamp": run_ts,
        })
        if (i + 1) % 200 == 0:
            ai_jobs.update(JOB, "score", i + 1, len(items), message=f"scored {i+1}/{len(items)}")

    # write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys()) if rows else [
        "image_id", "field", "value", "source", "verdict", "recommended_action",
        "majority_basis", "majority_value", "voters",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def accuracy_vs_human(subset):
        """Among rows with a folded human label, fraction where the prior AI value matches."""
        hit = [r for r in subset if r.get("human_value_folded")]
        if not hit:
            return {"n_overlap": 0, "n_match": 0, "accuracy": None}
        n_match = sum(1 for r in hit if r["value"] == r["human_value_folded"])
        return {"n_overlap": len(hit), "n_match": n_match, "accuracy": n_match / len(hit)}

    def maj_accuracy(subset):
        """Where a majority formed, does majority_value match the folded human label?"""
        hit = [r for r in subset if r.get("human_value_folded") and r.get("majority_outcome") == "majority_pass"]
        if not hit:
            return {"n": 0, "n_match": 0, "accuracy": None}
        n_match = sum(1 for r in hit if r["majority_value"] == r["human_value_folded"])
        return {"n": len(hit), "n_match": n_match, "accuracy": n_match / len(hit)}

    by_src_verdict = defaultdict(Counter)
    by_src_action = defaultdict(Counter)
    by_src_basis = defaultdict(Counter)
    for r in rows:
        by_src_verdict[r["source"]][r["verdict"]] += 1
        by_src_action[r["source"]][r["recommended_action"]] += 1
        if r.get("majority_basis"):
            by_src_basis[r["source"]][r["majority_basis"]] += 1

    # accuracy overall and per source, per field
    acc = {
        "overall": accuracy_vs_human(rows),
        "by_source": {s: accuracy_vs_human([r for r in rows if r["source"] == s]) for s in {r["source"] for r in rows}},
        "by_field": {f: accuracy_vs_human([r for r in rows if r["field"] == f]) for f in {r["field"] for r in rows}},
    }
    acc_maj = {
        "overall": maj_accuracy(rows),
        "by_source": {s: maj_accuracy([r for r in rows if r["source"] == s]) for s in {r["source"] for r in rows}},
    }

    # problem areas: high dispute / conflict rates, pmma unverifiable, class-level
    problems = []
    for src, c in by_src_verdict.items():
        tot = sum(c.values()) or 1
        for v in ("tie", "human_conflict", "model_disputed", "unverifiable"):
            if c[v] / tot >= 0.1 and c[v] >= 10:
                problems.append({"area": f"{src}:{v}", "n": c[v], "share": c[v] / tot})
    # value-level confirmation rate for auto_approved / 3model device_type
    for src in ("auto_approved", "3model", "assist"):
        sub = [r for r in rows if r["source"] == src and r["field"] == "device_type"]
        for val, grp in _group(sub, "value"):
            n = len(grp)
            n_ok = sum(1 for r in grp if r["verdict"] in ("confirmed", "majority_pass") and r["recommended_action"] == "keep")
            if n >= 15 and (n_ok / n) < 0.5:
                problems.append({"area": f"{src} device_type={val} low keep-rate", "n": n, "n_keep": n_ok, "share": n_ok / n})
    pmma_unv = sum(1 for r in rows if r["image_id"].startswith("pmma_") and r["verdict"] == "unverifiable")
    if pmma_unv:
        problems.append({"area": "pmma frames not on disk (unverifiable)", "n": pmma_unv})

    # Training split recount (do not write the split file)
    split_rows = list(csv.DictReader(SPLIT_PATH.open(encoding="utf-8"))) if SPLIT_PATH.is_file() else []
    human_split = [r for r in split_rows if r.get("label_source") == "human"]
    auto_split = [r for r in split_rows if r.get("label_source") == "model_consensus"]
    # map auto split image -> verification row (device_type, auto_approved)
    auto_ver = {
        r["image_id"]: r
        for r in rows
        if r["source"] == "auto_approved" and r["field"] == "device_type"
    }

    def recount(keep_pred):
        kept_auto = [r for r in auto_split if keep_pred(auto_ver.get(r["image_id"]))]
        dropped = [r for r in auto_split if r not in kept_auto]
        all_kept = human_split + kept_auto
        return {
            "n_total": len(all_kept),
            "n_human": len(human_split),
            "n_auto_kept": len(kept_auto),
            "n_auto_dropped_or_review": len(dropped),
            "by_split": dict(Counter(r["split"] for r in all_kept)),
            "by_label": dict(Counter(r["label"] for r in all_kept)),
            "auto_kept_by_label": dict(Counter(r["label"] for r in kept_auto)),
            "auto_not_kept_by_reason": dict(Counter(
                (auto_ver[r["image_id"]]["verdict"] if r["image_id"] in auto_ver else "missing_verification")
                for r in dropped
            )),
        }

    def keep_confirmed_only(vr):
        return bool(vr) and vr["verdict"] == "confirmed" and vr["recommended_action"] == "keep"

    def keep_majority_rule(vr):
        if not vr:
            return False
        # Proceeds if confirmed (AI-only) or majority_pass that agrees with the auto label (keep).
        if vr["recommended_action"] != "keep":
            return False
        return vr["verdict"] in ("confirmed", "majority_pass")

    split_sim = {
        "note": "Counts only — data/splits_v11_with_auto.csv was not modified.",
        "original": {
            "n_total": len(split_rows),
            "n_human": len(human_split),
            "n_auto": len(auto_split),
            "by_split": dict(Counter(r["split"] for r in split_rows)),
            "by_label": dict(Counter(r["label"] for r in split_rows)),
            "by_source": dict(Counter(r["label_source"] for r in split_rows)),
        },
        "if_only_confirmed_auto_kept": recount(keep_confirmed_only),
        "if_majority_rule_auto_kept": recount(keep_majority_rule),
    }

    summary = {
        "run_timestamp": run_ts,
        "timezone": "America/Los_Angeles",
        "job": JOB,
        "n_labels": len(rows),
        "n_images": len({r["image_id"] for r in rows}),
        "n_images_missing": len(missing_img),
        "thresholds": thresholds,
        "models": {
            "model_a": {"id": M.MODEL_A_ID, "revision": model_a_rev},
            "model_b": {"id": M.MODEL_B_ID, "revision": clip_rev},
            "model_c": {"id": M.MODEL_C_ID, "revision": dino_rev},
        },
        "majority_rule": {
            "description": (
                "Each human labeler has one vote. The 3-model consensus (all three agree at "
                "the validated threshold) counts as one vote ('ai'). A label proceeds when at "
                "least two voters agree and they form a strict majority. basis=ai+human when "
                "the AI and at least one human are in that majority (and there are not already "
                "two humans in it); basis=human+human when at least two humans agree; "
                "basis=human+human+ai when two or more humans and the AI agree. An AI-vs-single-"
                "human disagreement with no third vote is a tie: the human is not overridden and "
                "the row is sent to human review. The AI vote is never stored as a human label."
            ),
            "counts_by_source": {s: dict(c) for s, c in by_src_basis.items()},
            "outcomes": dict(Counter(r["majority_outcome"] for r in rows)),
            "passes": sum(1 for r in rows if r["verdict"] == "majority_pass"),
            "ties": sum(1 for r in rows if r["verdict"] == "tie"),
        },
        "verdicts_by_source": {s: dict(c) for s, c in by_src_verdict.items()},
        "verdicts_overall": dict(Counter(r["verdict"] for r in rows)),
        "recommended_actions_by_source": {s: dict(c) for s, c in by_src_action.items()},
        "accuracy_vs_human_folded": acc,
        "majority_value_vs_human_folded": acc_maj,
        "problem_areas": problems,
        "split_simulation": split_sim,
        "paths": {
            "verification_csv": str(OUT_CSV),
            "summary_json": str(OUT_JSON),
            "split_file_unchanged": str(SPLIT_PATH),
        },
        "finished": True,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    ai_jobs.log(f"wrote {OUT_CSV} ({len(rows)} rows) and {OUT_JSON}")
    ai_jobs.log("verdicts " + json.dumps(summary["verdicts_overall"]))
    ai_jobs.log("majority " + json.dumps(summary["majority_rule"]["outcomes"]))
    ai_jobs.update(
        JOB, "done", len(rows), len(rows), status="done",
        message=f"finished {len(rows)} labels",
        extra={"summary_path": str(OUT_JSON), "csv_path": str(OUT_CSV)},
    )
    return 0


def _group(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[r.get(key)].append(r)
    return g.items()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        try:
            ai_jobs.fail(JOB, traceback.format_exc().splitlines()[-1])
        except Exception:
            pass
        sys.exit(1)
