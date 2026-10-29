#!/usr/bin/env python3
"""Predict the device class of single images from a SAVED model -- weights + the image, nothing else.

  python scripts/predict.py --kind resnet      --weights var/runs/resnet50_v2/seed0_best.pt     img1.jpg [img2.jpg ...]
  python scripts/predict.py --kind siglip-ft   --weights var/runs/siglip2_v2/ft_seed0_best.pt  img.jpg
  python scripts/predict.py --kind siglip-probe --weights var/runs/siglip2_v2/probe_seed0_head.npz img.jpg
  python scripts/predict.py --kind yolo-cls    --weights var/runs/yolo11s_cls_v2_seed0/weights/best.pt img.jpg

No-training-data-at-inference contract (checked by scripts/check_leakage.py --runtime, which runs this with --audit):
  * the only repo files opened are the weights file, the input image(s), code and server/schema.json (class names);
  * no split/manifest/label/annotation file, no cached embeddings (emb.npz), no other image is read;
  * no kNN / prototype / retrieval and no test-time adaptation: each image is predicted independently in eval mode.
Pretrained base models (torchvision ResNet-50 definition, SigLIP2 backbone for siglip-*) come from the library caches outside
the repo; for siglip-ft the saved checkpoint overrides every layer that was fine-tuned.

Each image is preprocessed exactly as at evaluation time in the training scripts. Prints one JSON line per image."""
import argparse, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
import wvcore
CLASSES = list(wvcore.DEVICE_CLASSES)
SIGLIP = "google/siglip2-so400m-patch14-224"

AUDIT = []
def _hook(ev, args):
    if ev == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        try: AUDIT.append(os.path.abspath(os.fsdecode(args[0])))
        except Exception: pass

def short_side(im, short=256):
    from PIL import Image
    w, h = im.size; s = short / min(w, h)
    return im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["resnet", "siglip-ft", "siglip-probe", "yolo-cls"])
    ap.add_argument("--weights", required=True); ap.add_argument("--base", default=SIGLIP, help="SigLIP2 base model id (siglip-*)")
    ap.add_argument("--device", default=None); ap.add_argument("--audit", default=None, help="write the list of opened files here (JSON)")
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    if a.audit: sys.addaudithook(_hook)
    import numpy as np, torch
    from PIL import Image
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = []
    if a.kind == "resnet":
        import torchvision, torchvision.transforms as T
        m = torchvision.models.resnet50(weights=None); m.fc = torch.nn.Linear(m.fc.in_features, len(CLASSES))
        m.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=True)); m.eval().to(dev)
        tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        for p in a.images:
            x = tf(short_side(Image.open(p).convert("RGB"))).unsqueeze(0).to(dev)
            with torch.inference_mode(): pr = torch.softmax(m(x).float(), 1)[0].cpu().numpy()
            out.append((p, pr, CLASSES))
    elif a.kind in ("siglip-ft", "siglip-probe"):
        sys.path.insert(0, str(ROOT / "scripts")); import auto_annotate as AA
        from transformers import AutoModel, AutoProcessor
        m = AutoModel.from_pretrained(a.base).eval()
        if a.kind == "siglip-probe":
            h = np.load(a.weights, allow_pickle=False); W, b, cls = h["coef"], h["intercept"], [str(c) for c in h["classes"]]
            proc = AutoProcessor.from_pretrained(a.base); m.to(dev)
            for p in a.images:
                x = proc(images=[Image.open(p).convert("RGB")], return_tensors="pt").to(dev)
                with torch.inference_mode(): e = torch.nn.functional.normalize(AA.feats(m.get_image_features(**x)).float(), dim=-1)[0].cpu().numpy()
                z = W @ e + b; z = np.exp(z - z.max()); out.append((p, z / z.sum(), cls))
        else:
            import torchvision.transforms as T
            vm = m.vision_model; vm.to(dev)
            sd = torch.load(a.weights, map_location="cpu", weights_only=True)
            n = len([k for k in sd if k.startswith("m")]); has_head = getattr(vm, "head", None) is not None
            k = n - 1 - int(has_head)
            mods = list(vm.encoder.layers[-k:]) + [vm.post_layernorm] + ([vm.head] if has_head else [])
            for j, mm in enumerate(mods): mm.load_state_dict(sd[f"m{j}"])
            hw = sd["head"]["weight"]; head = torch.nn.Linear(hw.shape[1], hw.shape[0]); head.load_state_dict(sd["head"]); head.eval().to(dev)
            vm.eval(); tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
            for p in a.images:
                x = tf(short_side(Image.open(p).convert("RGB"))).unsqueeze(0).to(dev)
                with torch.inference_mode(): pr = torch.softmax(head(AA.feats(vm(pixel_values=x)).float()), 1)[0].cpu().numpy()
                out.append((p, pr, CLASSES))
    else:
        from ultralytics import YOLO
        m = YOLO(a.weights)
        for p in a.images:
            r = m.predict(p, verbose=False, save=False)[0]; names = [r.names[i] for i in range(len(r.names))]
            out.append((p, r.probs.data.float().cpu().numpy(), names))
    for p, pr, names in out:
        i = int(np.argmax(pr))
        print(json.dumps({"image": str(p), "pred": wvcore.canon_class(names[i]),
                          "probs": {wvcore.canon_class(n): round(float(v), 4) for n, v in zip(names, pr)}}))
    if a.audit:
        json.dump(sorted(set(AUDIT)), open(a.audit, "w"), indent=0)

if __name__ == "__main__":
    main()
