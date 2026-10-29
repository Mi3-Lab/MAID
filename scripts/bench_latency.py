#!/usr/bin/env python3
"""Batch-size-1 inference latency of the saved WheelView v3 baselines (seed 0) on this machine's GPU and CPU.

  python scripts/bench_latency.py --out results/latency_v3.json

Models are loaded exactly as scripts/predict.py loads them (weights + one image; no training data). For each model and
device we time, on the first 20 test images (decoded once, held in memory):
  * model_ms     : forward pass only (tensor already on the device), torch.inference_mode, fp32
  * end2end_ms   : PIL image in memory -> preprocessing -> forward -> softmax -> class (what predict.py does per image)
after warm-up. Reports median and 90th percentile over all timed iterations. CUDA timings synchronise before reading
the clock. Disk read/JPEG decode and model loading are excluded. GPU temperature is logged before and after."""
import argparse, csv, json, platform, statistics, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server")); sys.path.insert(0, str(ROOT / "scripts"))
import wvcore
CLASSES = list(wvcore.DEVICE_CLASSES)
SIGLIP = "google/siglip2-so400m-patch14-224"

def gpu_temp():
    try: return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks_throttle_reasons.active", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception: return None

def short_side(im, short=256):
    from PIL import Image
    w, h = im.size; s = short / min(w, h)
    return im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)

def build(kind, w, dev):
    import torch
    if kind == "resnet":
        import torchvision, torchvision.transforms as T
        m = torchvision.models.resnet50(weights=None); m.fc = torch.nn.Linear(m.fc.in_features, len(CLASSES))
        m.load_state_dict(torch.load(w, map_location="cpu", weights_only=True)); m.eval().to(dev)
        tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        pre = lambda im: tf(short_side(im)).unsqueeze(0).to(dev)
        fwd = lambda x: torch.softmax(m(x).float(), 1)
        return pre, fwd
    if kind == "siglip-ft":
        import auto_annotate as AA, torchvision.transforms as T
        from transformers import AutoModel
        vm = AutoModel.from_pretrained(SIGLIP).eval().vision_model.to(dev)
        sd = torch.load(w, map_location="cpu", weights_only=True)
        n = len([k for k in sd if k.startswith("m")]); has_head = getattr(vm, "head", None) is not None; k = n - 1 - int(has_head)
        mods = list(vm.encoder.layers[-k:]) + [vm.post_layernorm] + ([vm.head] if has_head else [])
        for j, mm in enumerate(mods): mm.load_state_dict(sd[f"m{j}"])
        hw = sd["head"]["weight"]; head = torch.nn.Linear(hw.shape[1], hw.shape[0]); head.load_state_dict(sd["head"]); head.eval().to(dev)
        tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
        pre = lambda im: tf(short_side(im)).unsqueeze(0).to(dev)
        fwd = lambda x: torch.softmax(head(AA.feats(vm(pixel_values=x)).float()), 1)
        return pre, fwd
    if kind == "yolo-cls":
        from ultralytics import YOLO
        y = YOLO(w); y.to(dev)
        return None, (lambda im: y.predict(im, verbose=False, save=False, device=str(dev))[0].probs.data)
    raise ValueError(kind)

def timeit(fn, args, dev, warm, iters):
    import torch
    for i in range(warm): fn(args[i % len(args)])
    if dev.type == "cuda": torch.cuda.synchronize()
    ts = []
    for i in range(iters):
        t0 = time.perf_counter(); fn(args[i % len(args)])
        if dev.type == "cuda": torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {"median_ms": round(statistics.median(ts), 2), "p90_ms": round(ts[int(0.9 * (len(ts) - 1))], 2), "iters": iters}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/latency_v3.json")
    ap.add_argument("--n-images", type=int, default=20); ap.add_argument("--cpu-iters", type=int, default=40); ap.add_argument("--gpu-iters", type=int, default=200)
    a = ap.parse_args()
    import torch
    from PIL import Image
    M = [r for r in csv.DictReader(open(ROOT / "data" / "manifest_v2.csv", encoding="utf-8")) if r["split"] == "test"][: a.n_images]
    ims = [Image.open(ROOT / r["path"]).convert("RGB") for r in M]
    models = [("SigLIP 2 so400m fine-tune (last 4 blocks)", "siglip-ft", "var/runs/siglip2_v3/ft_seed0_best.pt"),
              ("ResNet-50 fine-tune", "resnet", "var/runs/resnet50_v3/seed0_best.pt"),
              ("YOLO11s-cls fine-tune", "yolo-cls", "var/runs/yolo11s_cls_v3_seed0/weights/best.pt")]
    devs = (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]
    res = {"batch_size": 1, "precision": "fp32", "n_images": len(ims), "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
           "cpu": platform.processor(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "gpu_state_before": gpu_temp(), "results": []}
    try:
        import cpuinfo; res["cpu"] = cpuinfo.get_cpu_info().get("brand_raw", res["cpu"])
    except Exception: pass
    for name, kind, w in models:
        for d in devs:
            dev = torch.device(d); pre, fwd = build(kind, str(ROOT / w), dev)
            iters = a.gpu_iters if d == "cuda" else a.cpu_iters; warm = 20 if d == "cuda" else 5
            row = {"model": name, "weights": w, "device": d}
            with torch.inference_mode():
                if pre is not None:
                    xs = [pre(im) for im in ims]
                    row["model_only"] = timeit(fwd, xs, dev, warm, iters)
                    row["end2end"] = timeit(lambda im: int(fwd(pre(im)).argmax()), ims, dev, warm, iters)
                else:
                    row["end2end"] = timeit(lambda im: int(fwd(im).argmax()), ims, dev, warm, iters)
            print(json.dumps(row)); res["results"].append(row)
            del pre, fwd
            if d == "cuda": torch.cuda.empty_cache()
    res["gpu_state_after"] = gpu_temp()
    Path(ROOT / a.out).write_text(json.dumps(res, indent=1)); print("wrote", a.out)

if __name__ == "__main__":
    main()
