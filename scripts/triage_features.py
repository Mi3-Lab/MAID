"""Local-only zero-shot features for triage (text prompts vs the saved SigLIP2 image embeddings of scripts/triage_score.py).
Writes var/triage/features.csv (one row per local candidate). Nothing leaves this machine."""
import json, csv, os
from pathlib import Path
import numpy as np, torch
from transformers import AutoModel, AutoProcessor
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "var" / "triage"
P = lambda *xs: ["a photo of " + x for x in xs]
GROUPS = {
 "child": {"child": P("a child", "a baby", "a toddler", "a young kid", "a little boy", "a little girl", "an infant", "a teenager", "schoolchildren", "a kid playing"),
           "adult": P("an adult", "an adult man", "an adult woman", "an elderly person", "an adult in a wheelchair", "an old man", "a young woman"),
           "nopeople": P("no people", "an empty wheelchair", "an object", "a building", "an empty street")},
 "device": {"wheelchair": P("a wheelchair", "a person in a wheelchair", "a manual wheelchair", "an empty wheelchair", "a wheelchair user"),
            "power": P("a power wheelchair", "an electric wheelchair"),
            "scooter": P("a mobility scooter", "an elderly person on a mobility scooter"),
            "walker": P("a rollator walker", "a walking frame", "a person using a walker"),
            "cane": P("a person with crutches", "a walking cane", "a person using a white cane"),
            "cart": P("a shopping cart", "a shopping trolley"),
            "office": P("an office chair", "a rolling desk chair", "a swivel chair"),
            "luggage": P("a luggage trolley", "a hand truck", "a baggage cart at an airport"),
            "stroller": P("a baby stroller", "a pram", "a pushchair", "a buggy for a baby", "a baby carriage", "a jogging stroller"),
            "x_scene": P("a street", "a building", "a city", "a bridge"),
            "x_people": P("people standing", "a portrait of a person", "a group of people at an event", "a crowd", "people walking"),
            "x_room": P("a room interior", "a store aisle", "a kitchen"),
            "x_nature": ["a landscape photo", "a photo of a park", "a photo of a beach"],
            "x_vehicle": P("a car", "a bus", "a bicycle", "a motorcycle", "a train"),
            "x_text": ["a document with text", "a sign", "a logo", "a map", "a chart", "a poster"],
            "x_misc": P("food", "an animal", "a sofa", "a bed", "a toy", "a sculpture", "a ramp", "stairs")},
 "photo": {"photo": ["a photo", "a real photograph", "a snapshot taken with a camera", "a candid photo of a scene"],
           "render": ["a 3D render", "a computer generated image", "a product render", "a CAD model"],
           "drawing": ["a drawing", "an illustration", "a cartoon", "clip art", "a diagram", "a pictogram", "a painting", "a vector graphic"],
           "cutout": ["a product photo on a white background", "a catalog cut-out on a plain white background", "an isolated object on a white background"]},
}
SIG = {"child_sig": GROUPS["child"]["child"], "stroller_sig": GROUPS["device"]["stroller"]}
def feat(o): return o if torch.is_tensor(o) else (o.pooler_output if getattr(o, "pooler_output", None) is not None else o[0])
def main():
    meta = json.load(open(OUT / "ids.json")); ids = meta["ids"]
    EB, EO = np.load(OUT / "emb_blur.npy"), np.load(OUT / "emb_orig.npy")
    m = AutoModel.from_pretrained(meta["model"]).eval(); pr = AutoProcessor.from_pretrained(meta["model"])
    def T(txts):
        with torch.inference_mode():
            t = pr(text=txts, padding="max_length", max_length=64, return_tensors="pt")
            return torch.nn.functional.normalize(feat(m.get_text_features(**t)).float(), dim=-1).numpy()
    scale, bias = float(m.logit_scale.exp()), float(m.logit_bias)
    cols = {}
    for g, cls in GROUPS.items():
        names = list(cls); C = np.stack([T(cls[n]).mean(0) for n in names]); C /= np.linalg.norm(C, axis=1, keepdims=True)
        for tag, E in (("b", EB), ("o", EO)):
            L = E @ C.T / 0.01; L -= L.max(1, keepdims=True); Pm = np.exp(L); Pm /= Pm.sum(1, keepdims=True)
            for j, n in enumerate(names): cols[f"{g}_{n}_{tag}"] = Pm[:, j]
    for k, txts in SIG.items():
        Tt = T(txts)
        for tag, E in (("b", EB), ("o", EO)):
            cols[f"{k}_{tag}"] = (1 / (1 + np.exp(-(E @ Tt.T * scale + bias)))).max(1)
    heur = {json.loads(l)["image_id"]: json.loads(l) for l in open(OUT / "heur.jsonl", encoding="utf-8")}
    hk = ["w", "h", "border_uniform", "border_white", "colors16", "gray_entropy", "white_frac"]
    with open(OUT / "features.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["image_id"] + list(cols) + hk)
        for i, c in enumerate(ids): w.writerow([c] + [f"{float(cols[k][i]):.4f}" for k in cols] + [heur[c][k] for k in hk])
    print("features", len(ids), len(cols))
if __name__ == "__main__": main()
