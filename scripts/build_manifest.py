#!/usr/bin/env python3
"""Dataset manifest + leakage-safe grouping for the WheelView device-class dataset.

  python scripts/build_manifest.py                 # features -> duplicate/series groups -> split check -> manifest
  python scripts/build_manifest.py --pairs 40      # also print the 40 closest cross-group pairs (threshold tuning)

Reads (never writes) the events DB + annotations/events/*.jsonl, the license audit and derived/attributions.csv via
make_splits.collect(), and the face-blurred images in images_blurred/. Writes:
  data/manifest_v2.csv     one row per clean-set image (trainable rows: split train/val/test; others: split=excluded)
  data/MANIFEST.md         column reference, counts, leakage findings, the no-training-data-at-inference contract
  results/leakage_v2.json  duplicate pairs, groups, what crossed splits_v2
  data/splits_v3.csv (+ results/splits_v3_summary.json) when splits_v2 lets a duplicate/series group span two splits
Local cache (git-ignored var/, used ONLY here for deduplication, never by a model): var/manifest_features.npz

Duplicate detection (images_blurred/ files): sha256 (exact), DCT pHash incl. mirrored copy (near-dup), SigLIP2 image
embedding cosine (near-dup / re-crop / re-edit; GPU when available). Grouping (union-find): duplicate clusters, Flickr
owner (uploader), normalised creator name (catches one Flickr account seen under both its NSID and its path alias),
source page, Flickr photo id, near-consecutive Flickr photo ids (bursts), meaningful title stems (series / events).
Search queries / categories are NOT used: they encode the class (e.g. "openimages_v7_human_label:Motorized wheelchair")."""
import argparse, csv, hashlib, json, os, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "server"))
import make_splits as MS
import wvcore

CLASSES = list(wvcore.DEVICE_CLASSES)
SPLITS = ("train", "val", "test")
DESC = ["setting", "occupied", "occlusion", "propulsion", "viewpoint", "lighting"]
SIGLIP = "google/siglip2-so400m-patch14-224"
GENERIC_CREATORS = {"", "unknown", "anonymous", "flickr", "wikimedia commons", "public domain", "none"}
GENERIC_TITLE = re.compile(r"^(img|dsc|dscn|dscf|dcp|p|pict|pic|photo|image|untitled|sam|gopr|mvc|cimg|imag|wp|pxl|file|jpeg|jpg)?$")
COLS = ["image_id", "split", "train_eligible", "exclusion_reason", "device_type", "label_how", "labelers", "adjudicated_by",
        "labeled_via", *DESC, "n_boxes", "bboxes", "license", "license_url", "license_status", "creator", "source_site",
        "source_url", "uploader_group", "leak_group", "dup_cluster", "path", "sha256_blurred", "phash_blurred", "split_v2"]

# ---------------------------------------------------------------- features
def phash_bits(im):
    import scipy.fft
    from PIL import Image
    g = np.asarray(im.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    d = scipy.fft.dctn(g, norm="ortho")[:8, :8].flatten()
    return d[1:] > np.median(d[1:])          # 63 bits (DC term dropped)

def features(ids, imgdir, cache, device=None, bs=16):
    """sha256, pHash (and of the mirrored image), unit SigLIP2 embeddings; cached by sha256 in var/."""
    from PIL import Image, ImageOps
    old = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        old = {s: (p, q, e) for s, p, q, e in zip(z["sha"], z["ph"], z["phm"], z["emb"])}
    sha, ph, phm, emb, todo = {}, {}, {}, {}, []
    for i in ids:
        b = (imgdir / f"{i}.jpg").read_bytes(); s = hashlib.sha256(b).hexdigest(); sha[i] = s
        if s in old: ph[i], phm[i], emb[i] = old[s]
        else: todo.append(i)
    if todo:
        import torch
        from transformers import AutoModel, AutoProcessor
        import auto_annotate as AA
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        m = AutoModel.from_pretrained(SIGLIP).eval().to(dev); proc = AutoProcessor.from_pretrained(SIGLIP)
        print(f"embedding {len(todo)} images on {dev}", flush=True); t0 = time.time()
        for k in range(0, len(todo), bs):
            chunk = todo[k:k + bs]; ims = [Image.open(imgdir / f"{i}.jpg").convert("RGB") for i in chunk]
            for i, im in zip(chunk, ims): ph[i] = phash_bits(im); phm[i] = phash_bits(ImageOps.mirror(im))
            x = proc(images=ims, return_tensors="pt").to(dev)
            with torch.inference_mode(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
                e = AA.feats(m.get_image_features(**x))
            e = torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()
            for i, v in zip(chunk, e): emb[i] = v.astype(np.float32)
        print(f"embedded in {time.time() - t0:.0f}s", flush=True)
        allk = {sha[i]: (ph[i], phm[i], emb[i]) for i in ids}; allk.update({s: v for s, v in old.items() if s not in allk})
        ks = sorted(allk); cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, sha=np.array(ks), ph=np.stack([allk[s][0] for s in ks]), phm=np.stack([allk[s][1] for s in ks]),
                 emb=np.stack([allk[s][2] for s in ks]))
    return sha, ph, phm, emb

# ---------------------------------------------------------------- grouping helpers
class UF:
    def __init__(self, xs): self.p = {x: x for x in xs}
    def find(self, x):
        while self.p[x] != x: self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.p[max(a, b)] = min(a, b)

def norm_name(s): return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()
def title_stem(t):
    t = re.sub(r"\.(jpe?g|png|tiff?|gif)$", "", (t or "").lower())
    return re.sub(r"[^a-z]+", " ", re.sub(r"[\d_]+", " ", t)).strip()
def flickr_photo(url):
    m = re.search(r"flickr\.com/photos/[^/]+/(\d+)", url or ""); return int(m.group(1)) if m else None
def norm_url(u): return re.sub(r"[?#].*$", "", (u or "").strip().lower()).rstrip("/")

def labeled_via(img, done, evs):
    """Provenance only (never a feature): did each labeler pick device_type by hand, or confirm the shown suggestion?"""
    last = {}
    for e in evs:
        if e["image_id"] == img and e["kind"] == "label" and e["field"] == "device_type" and e["annotator"] in done:
            k = (e["ts"], e["uuid"])
            if e["annotator"] not in last or k > last[e["annotator"]][0]: last[e["annotator"]] = (k, e)
    via = []
    for a in done:
        e = last.get(a, (None, None))[1]
        if e is None: continue
        conf = e.get("source") == "auto_review" and e.get("suggested") is not None and \
            wvcore.canon_class(e.get("suggested")) == wvcore.canon_class(e.get("value"))
        via.append(f"{a}:{'confirmed_suggestion' if conf else 'manual'}")
    return ";".join(via)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=ROOT / "var" / "wheelview.db"); ap.add_argument("--events", default=ROOT / "annotations" / "events")
    ap.add_argument("--license", default=None); ap.add_argument("--images", default=ROOT / "images_blurred")
    ap.add_argument("--cache", default=ROOT / "var" / "manifest_features.npz"); ap.add_argument("--device", default=None)
    ap.add_argument("--phash_max", type=int, default=8, help="near-dup if pHash Hamming distance <= this (63 bits)")
    ap.add_argument("--cos_min", type=float, default=0.95, help="near-dup if SigLIP2 cosine >= this")
    ap.add_argument("--burst_gap", type=int, default=3, help="same owner not required; Flickr ids within this gap = one burst")
    ap.add_argument("--splits_v2", default=ROOT / "data" / "splits_v2.csv")
    ap.add_argument("--seed", type=int, default=20261027); ap.add_argument("--fracs", default="0.70,0.15,0.15")
    ap.add_argument("--pairs", type=int, default=0); ap.add_argument("--dry", action="store_true", help="report only, write nothing")
    a = ap.parse_args()
    imgdir = Path(a.images)
    rows, drops, info, excl = MS.collect(a.db, a.events, a.license, keep_excluded=True)
    evs = info["_events"]; allr = sorted(rows + excl, key=lambda r: r["image_id"]); ids = [r["image_id"] for r in allr]
    elig = {r["image_id"] for r in rows}
    sha, ph, phm, emb = features(ids, imgdir, Path(a.cache), a.device)

    # ---------- duplicate edges (over the whole clean set, so an excluded copy is still visible) ----------
    P = np.stack([ph[i] for i in ids]).astype(np.uint8); PM = np.stack([phm[i] for i in ids]).astype(np.uint8)
    H = np.minimum((P[:, None, :] != P[None, :, :]).sum(-1), (P[:, None, :] != PM[None, :, :]).sum(-1))
    E = np.stack([emb[i] for i in ids]); C = E @ E.T
    n = len(ids); iu = np.triu_indices(n, 1)
    edges = []   # (i, j, kind, detail)
    for x, y in zip(*iu):
        why = []
        if sha[ids[x]] == sha[ids[y]]: why.append("sha256")
        if H[x, y] <= a.phash_max: why.append(f"phash={int(H[x, y])}")
        if C[x, y] >= a.cos_min: why.append(f"cos={C[x, y]:.3f}")
        if why: edges.append((ids[x], ids[y], "dup", ",".join(why)))
    dup = UF(ids)
    for x, y, _, _ in edges: dup.union(x, y)
    # ---------- series / source edges ----------
    keyed = defaultdict(list); fid = {}
    cand = ROOT / "data" / "candidates.csv"
    titles = {r["candidate_id"]: r.get("title", "") for r in csv.DictReader(open(cand, encoding="utf-8"))} if cand.exists() else {}
    for r in allr:
        i = r["image_id"]; src = r["_lic"] or r["_attr"]; url = src.get("source_url") or src.get("source_page_url") or ""
        r["_url"] = url; keyed["uploader:" + r["group"]].append(i)
        nm = norm_name(r["creator"])
        if nm not in GENERIC_CREATORS: keyed["creator:" + nm].append(i)
        if url: keyed["page:" + norm_url(url)].append(i)
        f = flickr_photo(url)
        if f: keyed[f"flickr_photo:{f}"].append(i); fid[i] = f
        t = title_stem(titles.get(i) or (r["_attr"] or {}).get("title") or "")
        if not GENERIC_TITLE.match(t) and (len(t.split()) >= 2 or len(t) >= 8): keyed["title:" + t].append(i)
    byf = sorted(fid.items(), key=lambda kv: kv[1])
    for (i, f), (j, g) in zip(byf, byf[1:]):
        if g - f <= a.burst_gap: edges.append((i, j, "burst", f"flickr_ids {f},{g}"))
    for k, v in keyed.items():
        for j in v[1:]: edges.append((v[0], j, k.split(":")[0], k))
    leak = UF(ids)
    for x, y, _, _ in edges: leak.union(x, y)

    def numbered(uf, prefix):
        roots = sorted({uf.find(i) for i in ids}); return {rt: f"{prefix}{k:04d}" for k, rt in enumerate(roots)}
    dnum, lnum = numbered(dup, "d"), numbered(leak, "g")
    for r in allr:
        r["dup_cluster"] = dnum[dup.find(r["image_id"])]; r["leak_group"] = lnum[leak.find(r["image_id"])]

    # ---------- splits_v2 check ----------
    v2 = {r["image_id"]: r["split"] for r in csv.DictReader(open(a.splits_v2, encoding="utf-8"))} if Path(a.splits_v2).exists() else {}
    tr = [r for r in rows]
    crossing = defaultdict(set)
    for r in tr:
        if r["image_id"] in v2: crossing[r["leak_group"]].add(v2[r["image_id"]])
    crossing = {g: s for g, s in crossing.items() if len(s) > 1}
    cross_edges = [e for e in edges if e[0] in v2 and e[1] in v2 and v2[e[0]] != v2[e[1]]]
    cross_by_kind = Counter(k for _, _, k, _ in cross_edges)
    dup_cross = [e for e in cross_edges if e[2] == "dup"]
    v2_not_clean = sorted(set(v2) - elig); clean_not_v2 = sorted(elig - set(v2))
    leaks = bool(crossing)
    need_v3 = leaks or v2_not_clean or clean_not_v2
    fr = [float(x) for x in a.fracs.split(",")]
    if need_v3:
        asg = MS.assign(tr, a.seed, fr, group_field="leak_group", require_all=("train", "val", "test"))
        for r in tr: r["split"] = asg[r["leak_group"]]
        split_file = "data/splits_v3.csv"
    else:
        for r in tr: r["split"] = v2[r["image_id"]]
        split_file = "data/splits_v2.csv"
    moved = [r["image_id"] for r in tr if r["image_id"] in v2 and v2[r["image_id"]] != r["split"]]
    for r in excl: r["split"] = "excluded"

    # ---------- manifest rows ----------
    out = []
    for r in allr:
        i = r["image_id"]; st = r["_st"]; L = r["_lic"]; A = r["_attr"]
        f = st["fields"]; dv, how, _ = f.get("device_type", ("", "missing", {}))
        adj = [info["_adj"][(i, fl)][1] for fl in wvcore.COMPARE if (i, fl) in info["_adj"]]
        boxes = {k: v for k, v in st["bboxes"].items() if v}
        out.append({"image_id": i, "split": r["split"], "train_eligible": int(i in elig), "exclusion_reason": r.get("exclusion_reason", ""),
                    "device_type": dv, "label_how": how, "labelers": ";".join(st["done"]), "adjudicated_by": ";".join(sorted(set(adj))),
                    "labeled_via": labeled_via(i, st["done"], evs),
                    **{d: (f.get(d, ("",))[0] or "") for d in DESC},
                    "n_boxes": max((len(v) for v in boxes.values()), default=0), "bboxes": json.dumps(boxes, separators=(",", ":")) if boxes else "",
                    "license": L.get("license") or A.get("license", ""), "license_url": L.get("license_url") or A.get("license_url", ""),
                    "license_status": r["license_status"], "creator": r["creator"],
                    "source_site": L.get("source_site") or A.get("source_site", ""), "source_url": r["_url"],
                    "uploader_group": r["group"], "leak_group": r["leak_group"], "dup_cluster": r["dup_cluster"],
                    "path": f"images_blurred/{i}.jpg", "sha256_blurred": sha[i], "phash_blurred": "%016x" % int("".join("1" if b else "0" for b in ph[i]), 2),
                    "split_v2": v2.get(i, "")})

    # ---------- report ----------
    cnt = {s: Counter(r["label"] for r in tr if r["split"] == s) for s in SPLITS}
    gsz = Counter(r["leak_group"] for r in tr); usz = Counter(r["group"] for r in tr)
    big = usz.most_common(1)[0]; big_split = sorted({r["split"] for r in tr if r["group"] == big[0]})
    big_creator = Counter(r["creator"] for r in tr if r["group"] == big[0]).most_common(1)[0][0]
    near = []
    for x, y in zip(*iu):
        if leak.find(ids[x]) != leak.find(ids[y]): near.append((float(C[x, y]), int(H[x, y]), ids[x], ids[y]))
    near.sort(reverse=True)
    rep = {"generated_by": "scripts/build_manifest.py", "n_clean_set": len(allr), "n_trainable": len(tr), "n_excluded": len(excl),
           "excluded_reasons": dict(Counter(r["exclusion_reason"] for r in excl)),
           "thresholds": {"phash_max": a.phash_max, "cos_min": a.cos_min, "burst_gap": a.burst_gap},
           "edges_by_kind": dict(Counter(k for _, _, k, _ in edges)),
           "dup_pairs": [{"a": x, "b": y, "why": d} for x, y, k, d in edges if k == "dup"],
           "dup_clusters_multi": sum(1 for v in Counter(r["dup_cluster"] for r in allr).values() if v > 1),
           "leak_groups": len(gsz), "leak_groups_multi": sum(1 for v in gsz.values() if v > 1), "largest_leak_groups": gsz.most_common(5),
           "uploader_groups": len(usz),
           "splits_v2": {"n": len(v2), "leaking_groups": {g: sorted(s) for g, s in crossing.items()},
                         "crossing_edges_by_kind": dict(cross_by_kind), "crossing_edges": [{"a": x, "b": y, "kind": k, "why": d, "splits": [v2[x], v2[y]]} for x, y, k, d in cross_edges],
                         "dup_pairs_across_splits": len(dup_cross), "in_v2_not_trainable_now": v2_not_clean, "trainable_not_in_v2": clean_not_v2},
           "split_file": split_file, "moved_vs_v2": len(moved), "moved_ids": moved,
           "counts": {s: {c: cnt[s][c] for c in CLASSES} for s in SPLITS},
           "largest_uploader": {"group": big[0], "creator": big_creator, "n": big[1], "splits": big_split,
                                 "leak_group_size": max(gsz[r["leak_group"]] for r in tr if r["group"] == big[0])},
           "closest_cross_group_pairs": [{"cos": round(c, 4), "phash": h, "a": x, "b": y} for c, h, x, y in near[:max(a.pairs, 25)]]}
    print(json.dumps({k: v for k, v in rep.items() if k not in ("dup_pairs", "closest_cross_group_pairs", "moved_ids")}, indent=1, default=str))
    if a.pairs:
        for c, h, x, y in near[:a.pairs]: print(f"near cross-group: cos={c:.4f} phash={h} {x} {y}")
    if a.dry: return
    with open(ROOT / "data" / "manifest_v2.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, COLS); w.writeheader(); w.writerows(out)
    if need_v3:
        with open(ROOT / "data" / "splits_v3.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(["image_id", "split", "label", "creator", "group", "leak_group", "label_how", "license_status"])
            for r in sorted(tr, key=lambda r: r["image_id"]):
                w.writerow([r["image_id"], r["split"], r["label"], r["creator"], r["group"], r["leak_group"], r["label_how"], r["license_status"]])
        json.dump({"seed": a.seed, "fracs": fr, "group_by": "leak_group (duplicates + uploader + creator + page + burst + title series)",
                   "require_every_class_in": ["train", "val", "test"], "n": len(tr), "counts": rep["counts"],
                   "moved_vs_v2": len(moved), "license_file": info["license_file"], "events_used": info["events_used"]},
                  open(ROOT / "results" / "splits_v3_summary.json", "w"), indent=2)
    json.dump(rep, open(ROOT / "results" / "leakage_v2.json", "w"), indent=1, default=str)
    write_md(rep, out, a)
    print("wrote data/manifest_v2.csv", len(out), "rows;", split_file)

def write_md(rep, out, a):
    tr = [r for r in out if r["train_eligible"]]; ex = [r for r in out if not r["train_eligible"]]
    c = rep["counts"]; s2 = rep["splits_v2"]
    tab = ["| split | " + " | ".join(CLASSES) + " | total |", "|---|" + "---|" * (len(CLASSES) + 1)]
    for s in SPLITS: tab.append(f"| {s} | " + " | ".join(str(c[s][k]) for k in CLASSES) + f" | {sum(c[s].values())} |")
    tab.append("| all | " + " | ".join(str(sum(c[s][k] for s in SPLITS)) for k in CLASSES) + f" | {len(tr)} |")
    exl = "\n".join(f"| {r['image_id']} | {r['exclusion_reason']} | {r['device_type'] or '-'} | {r['license_status']} |" for r in ex) or "| - | - | - | - |"
    leakl = "\n".join(f"- `{e['a']}` ({e['splits'][0]}) -- `{e['b']}` ({e['splits'][1]}): {e['kind']} `{e['why']}`" for e in s2["crossing_edges"]) or "- none"
    big = rep["largest_uploader"]
    md = f"""# WheelView dataset manifest (`data/manifest_v2.csv`)

Generated by `scripts/build_manifest.py`; checked by `scripts/check_leakage.py` (exit code 1 on any leak). Do not edit by hand --
re-run both. Images are never committed; `path` is relative to the repo root on the lab laptop (face-blurred copies).

**{len(out)} rows** = every clean-set image (human `_status=done` by samir/shaaz, not labeler-excluded): **{len(tr)} trainable**
(`train_eligible=1`, split train/val/test) and **{len(ex)} excluded** (`train_eligible=0`, `split=excluded`, never used for
training or evaluation). Splits: `{rep['split_file']}` (seed {a.seed}, fractions {a.fracs}, grouped by `leak_group`, class-stratified).

## Class counts (trainable rows)

{chr(10).join(tab)}

`ambiguous` = the device class called `lookalike` before 2026-10-28 (wheelchair-like objects that are not clearly a wheelchair:
strollers, rollators, shopping carts, ...). Old events/suggestions say `lookalike`; they are mapped on read
(`server/schema.json` `value_aliases`, `wvcore.canon`). `unclear` is not a class: those images are excluded.

## Excluded rows

| image_id | reason | device_type | license |
|---|---|---|---|
{exl}

## Columns

| column | meaning |
|---|---|
| image_id | `wv_NNNNNN`; assigned in collection order (source, then search query), so **ID order correlates with class -- never sort, bin or batch by ID** |
| split | train / val / test (trainable) or `excluded` |
| train_eligible, exclusion_reason | 1/0; why a clean-set image is not trainable (`device_unclear`, `license_needs_ross`, `unresolved_device_conflict`, ...) |
| device_type | label (manual / powered / other_aid / ambiguous), human consensus or adjudication |
| label_how, labelers, adjudicated_by | single / consensus / majority / adjudicated; who marked the image done; who adjudicated any field |
| labeled_via | per labeler: `manual` or `confirmed_suggestion` (picked the shown SigLIP2 suggestion). Provenance for analysis only |
| setting, occupied, occlusion, propulsion, viewpoint, lighting | resolved descriptive fields ("" = missing or unresolved disagreement) |
| n_boxes, bboxes | max boxes per labeler; JSON `{{labeler: [[x0,y0,x1,y1], ...]}}` normalised 0-1 |
| license, license_url, license_status, creator, source_site, source_url | attribution from the license audit (fallback `derived/attributions.csv`) |
| uploader_group | Flickr owner (NSID or path alias) or creator, as used by splits_v1/v2 |
| dup_cluster | exact / near-duplicate cluster (sha256, pHash, SigLIP2 cosine) |
| leak_group | union of dup_cluster, uploader, creator name, source page, Flickr photo id, Flickr id bursts, title series: **the split unit** |
| path, sha256_blurred, phash_blurred | `images_blurred/<id>.jpg`, its sha256 and 64-bit DCT pHash (hex) |
| split_v2 | the image's split in `data/splits_v2.csv` (for comparing old results); "" if it was not in v2 |

No model output (suggestions, confidences, probabilities, embeddings) is a manifest column, and none may be used as a feature.

## Leakage checks

Duplicates (thresholds tuned by inspecting the closest pairs): pHash Hamming <= {a.phash_max}/63 (also against the mirrored
image) or SigLIP2 cosine >= {a.cos_min} or identical sha256 -> one `dup_cluster`. Found {len(rep['dup_pairs'])} duplicate pair(s),
{rep['dup_clusters_multi']} multi-image cluster(s). Edges by kind: `{json.dumps(rep['edges_by_kind'])}`.
Visual check (2026-10-28, defaults 8 / 0.95): no exact (sha256) duplicates; pairs at cosine 0.95-0.98 are the same
scene/event shot seconds apart (burst-like, kept together on purpose) and every one was within one uploader already;
the closest cross-group pairs (cosine <= 0.93, pHash >= 18) are different photos of similar wheelchair-sports scenes.
{rep['leak_groups']} leak groups ({rep['leak_groups_multi']} with more than one image); largest: {', '.join(f'{g} ({n})' for g, n in rep['largest_leak_groups'])}.

Series grouping: Flickr owner, normalised creator name (merges one account seen as NSID in one source and as path alias in
another), identical source page or Flickr photo id (the same photo found via two sources), Flickr photo ids within
{a.burst_gap} of each other (bursts), and non-generic title stems (digits stripped; camera defaults like IMG/DSC ignored).
Search queries / Open Images labels / Openverse queries are **not** used for anything: they encode the class.

**splits_v2 audit:** {len(s2['leaking_groups'])} leak group(s) spanned two splits; crossing edges by kind
`{json.dumps(s2['crossing_edges_by_kind'])}`; {s2['dup_pairs_across_splits']} duplicate pair(s) across splits.
{leakl}

In v2 but no longer trainable: {len(s2['in_v2_not_trainable_now'])}; trainable but not in v2 (labeled since): {len(s2['trainable_not_in_v2'])}.
Split used: `{rep['split_file']}`; {rep['moved_vs_v2']} image(s) are in a different split than in v2.

**Largest uploader:** `{big['group']}` ({big['n']} trainable images by {big['creator']}; its leak group has {big['leak_group_size']}
images) is one group and sits entirely in **{', '.join(big['splits'])}**. It can never inflate test numbers through
near-duplicates, but it is ~{round(100 * big['n'] / max(len(tr), 1))}% of the trainable set, so models see a lot of one photographer's
style and sports wheelchairs; per-source / per-uploader error analysis is advised.

## Rules for training and evaluation code

1. Splits come from this manifest (`split`, `train_eligible=1` only); `scripts/train_*.py` default to `--manifest data/manifest_v2.csv`.
2. Labels used for fitting, class weights, early stopping, C / threshold selection come from train (fit) and val (selection)
   rows only. Nothing -- normalisation statistics, calibration, prompt choice, thresholds -- is fit on test.
3. Image files are addressed by `path` (`images_blurred/<id>.jpg`): no class name in any file name, folder or ID. (YOLO-cls
   needs class folders: only train/val are linked into them; test is predicted from the neutral paths.)
4. Model suggestions (SigLIP2 zero-shot shown in the labeling app) are never inputs. **Bias caveat:** labelers saw the
   suggestion and could confirm it (`labeled_via`), so labels may lean toward SigLIP2's opinion; SigLIP2-based models may look
   better than they are, and "agreement with stored suggestions" is not an independent accuracy measure.
5. **No training data at inference.** Evaluation / inference loads only the saved weights plus the image being predicted:
   no kNN / retrieval / prototype lookups over train images, embeddings or labels; no cached train features
   (`var/runs/*/emb.npz` is an inspection cache that no script reads back) and no split/label files opened while predicting;
   no test-time adaptation or batch statistics (eval mode, zero-shot uses fixed text embeddings); the linear probe predicts
   with its fitted head only (saved as `probe_seed*_head.npz`). `scripts/predict.py` is the reference inference path;
   `python scripts/check_leakage.py --runtime` runs it under a Python audit hook and fails if it opens any repo file other
   than the weights, the image, code and `server/schema.json`.
6. `var/manifest_features.npz` (embeddings of every clean image) exists only for deduplication; no training/eval code may read it.
"""
    (ROOT / "data" / "MANIFEST.md").write_text(md, encoding="utf-8")

if __name__ == "__main__":
    main()
