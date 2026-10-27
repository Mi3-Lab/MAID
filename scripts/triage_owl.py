"""Local OWLv2 confirm pass for no-device removal candidates (var/triage/owl_needed.txt -> var/triage/owl.jsonl). Local only."""
import json, os, sys, time
from pathlib import Path
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "var" / "triage"
Q = ["a wheelchair", "a power wheelchair", "a mobility scooter", "a rollator walker", "a walking cane", "crutches",
     "a shopping cart", "an office chair", "a luggage trolley", "a baby stroller"]
def main():
    ids = [l.strip() for l in open(OUT / "owl_needed.txt") if l.strip()]
    done = set()
    if (OUT / "owl.jsonl").exists(): done = {json.loads(l)["image_id"] for l in open(OUT / "owl.jsonl")}
    ids = [i for i in ids if i not in done]; torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    M = "google/owlv2-base-patch16-ensemble"; p = Owlv2Processor.from_pretrained(M); m = Owlv2ForObjectDetection.from_pretrained(M).eval()
    f = open(OUT / "owl.jsonl", "a"); t0 = time.time()
    for n, i in enumerate(ids, 1):
        im = Image.open(ROOT / "images" / f"{i}.jpg").convert("RGB"); W, H = im.size; S = max(W, H)
        with torch.inference_mode():
            inp = p(text=[Q], images=im, return_tensors="pt"); o = m(**inp)
            r = p.image_processor.post_process_object_detection(o, threshold=0.1, target_sizes=torch.tensor([[S, S]]))[0]
        boxes = [{"label": Q[int(l)][2:] if Q[int(l)].startswith("a ") else Q[int(l)], "score": round(float(s), 3),
                  "box": [round(float(v), 1) for v in b]} for s, l, b in zip(r["scores"], r["labels"], r["boxes"])]
        boxes = [b for b in boxes if b["box"][0] < W and b["box"][1] < H]
        f.write(json.dumps({"image_id": i, "boxes": sorted(boxes, key=lambda b: -b["score"])[:10]}) + "\n"); f.flush()
        if n % 10 == 0: print(f"{n}/{len(ids)} {(time.time()-t0)/n:.2f}s/img", flush=True)
    print("done", flush=True)
if __name__ == "__main__": main()
