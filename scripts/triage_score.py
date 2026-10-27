"""Local-only triage embeddings (nothing leaves this machine): SigLIP2 image embeddings of every local candidate
(blurred release copy + local original) and simple pixel heuristics. Output: var/triage/emb_*.npy, ids.json, heur.jsonl.
Scoring with text prompts is done separately by scripts/triage_decide.py (fast, re-runnable)."""
import json, time, os
from pathlib import Path
import numpy as np, torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
ROOT = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("WV_TRIAGE_MODEL", "google/siglip2-base-patch16-224")
OUT = ROOT / "var" / "triage"
def heur(im):
    a = np.asarray(im.convert("RGB").resize((256, 256)), dtype=np.int16)
    b = np.concatenate([a[:12].reshape(-1, 3), a[-12:].reshape(-1, 3), a[:, :12].reshape(-1, 3), a[:, -12:].reshape(-1, 3)])
    med = np.median(b, 0); near = (np.abs(b - med).max(1) <= 12).mean()
    white = (b.min(1) >= 235).mean()
    q = (a // 16).reshape(-1, 3); cols = len(np.unique(q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]))
    g = a.mean(2).astype(np.int32); h = np.bincount(g.ravel(), minlength=256) / g.size; ent = float(-(h[h > 0] * np.log2(h[h > 0])).sum())
    allw = (a.min(2) >= 240).mean()
    return {"border_uniform": round(float(near), 3), "border_white": round(float(white), 3), "colors16": int(cols),
            "gray_entropy": round(ent, 3), "white_frac": round(float(allw), 3)}
def feat(o): return o if torch.is_tensor(o) else (o.pooler_output if getattr(o, "pooler_output", None) is not None else o[0])
def main():
    ids = sorted(p.stem for p in (ROOT / "images_blurred").glob("wv_*.jpg"))
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    m = AutoModel.from_pretrained(MODEL).eval(); p = AutoProcessor.from_pretrained(MODEL)
    dim = None; EB, EO = None, None; H = open(OUT / "heur.jsonl", "w", encoding="utf-8"); t0 = time.time(); B = 16
    for i in range(0, len(ids), B):
        chunk = ids[i:i + B]
        for kind, d in (("blur", "images_blurred"), ("orig", "images")):
            ims = []
            for c in chunk:
                try: im = Image.open(ROOT / d / f"{c}.jpg"); im.load(); im = im.convert("RGB")
                except Exception as e: print("WARN", c, kind, e, flush=True); im = Image.new("RGB", (224, 224))
                ims.append(im)
                if kind == "blur": H.write(json.dumps({"image_id": c, "w": im.size[0], "h": im.size[1], **heur(im)}) + "\n")
            with torch.inference_mode():
                E = torch.nn.functional.normalize(feat(m.get_image_features(**p(images=ims, return_tensors="pt"))).float(), dim=-1).numpy()
            if EB is None: EB = np.zeros((len(ids), E.shape[1]), "float32"); EO = np.zeros_like(EB)
            (EB if kind == "blur" else EO)[i:i + len(chunk)] = E
        n = i + len(chunk); el = time.time() - t0
        if (i // B) % 10 == 0: print(f"{n}/{len(ids)} {el/n:.2f}s/img eta {(len(ids)-n)*el/n/60:.1f}min", flush=True)
    H.close(); np.save(OUT / "emb_blur.npy", EB); np.save(OUT / "emb_orig.npy", EO)
    json.dump({"ids": ids, "model": MODEL}, open(OUT / "ids.json", "w")); print("done", flush=True)
if __name__ == "__main__": main()
