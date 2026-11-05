"""Pluggable annotation-assist models: open-vocabulary detector, SAM 2 segmenter, small VLM (next to SigLIP in
scripts/auto_annotate.py). Everything here produces MACHINE SUGGESTIONS only; they are stored separately from human
labels (annotations/auto/*_attr-vlm-*.jsonl for attribute hints, var/assist/ for boxes and masks) and are never
written as annotation events.

Design
  * Config: config/assist_models.json (or env WV_ASSIST_CONFIG). Each model: backend, model id, optional revision/path.
  * Lazy: nothing heavy is imported until a model is first used; loaded models are cached (one instance per process).
  * Offline by default: from_pretrained(..., local_files_only=True) unless "allow_download": true. Weights are never
    committed; see docs/ANNOTATION_MODELS_v1.1.md for how to fetch them once.
  * Graceful: get(kind) returns None and status() explains why (disabled / package missing / weights not cached /
    load error); CPU is used when there is no GPU.
  * Inference only sees pixels and prompts -- never labels, events or manifests.

Backends: detector = owlv2 | grounding_dino | yolo_world; segmenter = sam2; vlm = florence2 | qwen2vl.
"""
import functools, importlib.util, json, os, re
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "assist_models.json"
KINDS = ("detector", "segmenter", "vlm")
BACKENDS = {"detector": ("owlv2", "grounding_dino", "yolo_world"), "segmenter": ("sam2",), "vlm": ("florence2", "qwen2vl")}
NEEDS = {"owlv2": ("torch", "transformers"), "grounding_dino": ("torch", "transformers"), "yolo_world": ("ultralytics",),
         "sam2": ("torch", "transformers"), "florence2": ("torch", "transformers"), "qwen2vl": ("torch", "transformers")}
_STATUS = {}

def load_config(path=None):
    p = Path(path or os.environ.get("WV_ASSIST_CONFIG") or DEFAULT_CONFIG)
    cfg = json.load(open(p, encoding="utf-8")) if p.exists() else {}
    for k in KINDS: cfg.setdefault(k, {"enabled": False})
    return cfg

def model_tag(spec):
    """Short, filesystem-safe tag that names model + revision, e.g. 'florence-2-base' or 'owlv2-base-patch16-ensemble-rabc12345'."""
    t = re.sub(r"[^A-Za-z0-9._-]+", "-", (spec.get("model") or spec.get("backend") or "model").split("/")[-1]).lower()
    return t + (f"-r{spec['revision'][:8]}" if spec.get("revision") else "")

def _device(cfg):
    want = cfg.get("device", "auto")
    if want != "auto": return want
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception: return "cpu"

def status(kind=None, cfg=None):
    """{kind: {available, reason, backend, model}} without loading anything heavy (plus load results seen so far)."""
    cfg = cfg or load_config(); out = {}
    for k in ([kind] if kind else KINDS):
        s = cfg.get(k) or {}; b = s.get("backend"); r = ""
        if not s.get("enabled"): r = "disabled in config"
        elif b not in BACKENDS[k]: r = f"unknown backend {b!r} (choose {BACKENDS[k]})"
        else:
            miss = [m for m in NEEDS[b] if importlib.util.find_spec(m) is None]
            if miss: r = "python package missing: " + ", ".join(miss) + " (pip install -r requirements-auto.txt)"
        r = r or _STATUS.get(k, "")
        out[k] = {"available": not r, "reason": r or "ok (loads on first use)", "backend": b, "model": s.get("model"), "tag": model_tag(s) if s else ""}
    return out if not kind else out[kind]

def get(kind, cfg=None):
    """The loaded model wrapper for `kind`, or None if unavailable (reason in status(kind))."""
    cfg = cfg or load_config()
    st = status(kind, cfg)
    if not st["available"]: return None
    return _load(kind, json.dumps(cfg, sort_keys=True))

@functools.lru_cache(maxsize=None)
def _load(kind, cfg_json):
    cfg = json.loads(cfg_json); s = cfg[kind]
    try:
        return {"owlv2": OwlDetector, "grounding_dino": GroundingDinoDetector, "yolo_world": YoloWorldDetector,
                "sam2": Sam2Segmenter, "florence2": Florence2VLM, "qwen2vl": Qwen2VLM}[s["backend"]](s, cfg)
    except Exception as e:   # weights not cached, no network, incompatible versions ... -> degrade, do not crash
        msg = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
        _STATUS[kind] = f"load failed: {msg} (fetch weights: docs/ANNOTATION_MODELS_v1.1.md)"
        return None

def _hf_kwargs(s, cfg):
    kw = {"local_files_only": not cfg.get("allow_download", False)}
    if s.get("revision"): kw["revision"] = s["revision"]
    return kw

def _src(s): return s.get("path") or s["model"]

class _Base:
    kind = ""
    def __init__(self, spec, cfg):
        self.spec, self.cfg, self.device = spec, cfg, _device(cfg)
        self.name, self.tag = spec.get("model"), model_tag(spec)
        self.version = spec.get("revision") or "local-cache"
        self.load()
    def load(self): raise NotImplementedError
    def info(self): return {"model": self.name, "backend": self.spec.get("backend"), "revision": self.version, "tag": self.tag}

# ---------------------------------------------------------------- detectors: boxes [x0, y0, x1, y1] in pixels
class OwlDetector(_Base):
    kind = "detector"
    def load(self):
        from transformers import AutoProcessor, Owlv2ForObjectDetection
        kw = _hf_kwargs(self.spec, self.cfg)
        self.p = AutoProcessor.from_pretrained(_src(self.spec), **kw)
        self.m = Owlv2ForObjectDetection.from_pretrained(_src(self.spec), **kw).eval().to(self.device)
    def detect(self, image, queries=None, min_score=None):
        import torch
        q = queries or self.spec.get("queries") or ["a wheelchair"]; thr = self.spec.get("min_score", 0.2) if min_score is None else min_score
        W, H = image.size; S = max(W, H)   # OWLv2 pads to a square
        with torch.inference_mode():
            x = self.p(text=[q], images=image, return_tensors="pt").to(self.device); o = self.m(**x)
            pp = getattr(self.p, "post_process_object_detection", None) or self.p.post_process_grounded_object_detection
            r = pp(o, threshold=thr, target_sizes=torch.tensor([[S, S]], device=self.device))[0]
        return [{"box": [round(max(0.0, b[0]), 1), round(max(0.0, b[1]), 1), round(min(b[2], W), 1), round(min(b[3], H), 1)],
                 "label": q[l], "score": round(float(s), 4)} for b, s, l in zip(r["boxes"].tolist(), r["scores"].tolist(), r["labels"].tolist())]

class GroundingDinoDetector(_Base):
    kind = "detector"
    def load(self):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        kw = _hf_kwargs(self.spec, self.cfg)
        self.p = AutoProcessor.from_pretrained(_src(self.spec), **kw)
        self.m = AutoModelForZeroShotObjectDetection.from_pretrained(_src(self.spec), **kw).eval().to(self.device)
    def detect(self, image, queries=None, min_score=None):
        import torch
        q = queries or self.spec.get("queries") or ["a wheelchair"]; thr = self.spec.get("min_score", 0.25) if min_score is None else min_score
        text = ". ".join(x.lower().strip(".") for x in q) + "."
        with torch.inference_mode():
            x = self.p(images=image, text=text, return_tensors="pt").to(self.device); o = self.m(**x)
            r = self.p.post_process_grounded_object_detection(o, x.input_ids, threshold=thr, target_sizes=[image.size[::-1]])[0]
        labs = r.get("text_labels") or r.get("labels")
        return [{"box": [round(v, 1) for v in b], "label": str(l), "score": round(float(s), 4)} for b, s, l in zip(r["boxes"].tolist(), r["scores"].tolist(), labs)]

class YoloWorldDetector(_Base):
    kind = "detector"
    def load(self):
        from ultralytics import YOLOWorld
        src = _src(self.spec)
        if not self.cfg.get("allow_download") and not Path(src).exists(): raise FileNotFoundError(f"{src} not found locally")
        self.m = YOLOWorld(src)
    def detect(self, image, queries=None, min_score=None):
        q = queries or self.spec.get("queries") or ["a wheelchair"]; self.m.set_classes(q)
        r = self.m.predict(image, conf=self.spec.get("min_score", 0.2) if min_score is None else min_score, verbose=False)[0]
        return [{"box": [round(v, 1) for v in b], "label": q[int(c)], "score": round(float(s), 4)}
                for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]

# ---------------------------------------------------------------- SAM 2: box prompt -> binary mask (H x W bool numpy)
class Sam2Segmenter(_Base):
    kind = "segmenter"
    def load(self):
        from transformers import Sam2Model, Sam2Processor
        kw = _hf_kwargs(self.spec, self.cfg)
        self.p = Sam2Processor.from_pretrained(_src(self.spec), **kw)
        self.m = Sam2Model.from_pretrained(_src(self.spec), **kw).eval().to(self.device)
    def segment(self, image, box):
        import torch
        with torch.inference_mode():
            x = self.p(images=image, input_boxes=[[[float(v) for v in box]]], return_tensors="pt").to(self.device)
            o = self.m(**x, multimask_output=False)
            masks = self.p.post_process_masks(o.pred_masks.cpu(), x["original_sizes"].cpu())[0]
        return masks.reshape(-1, *masks.shape[-2:])[0].numpy().astype(bool)

# ---------------------------------------------------------------- VLMs: question + options -> one option (or None)
def parse_choice(text, options):
    """Map free VLM text to one option: exact token match first (underscores and spaces equal), longest option wins;
    None when nothing (or 'unknown') matches -- 'unknown' is the annotator's choice, never a suggestion."""
    t = " " + re.sub(r"[^a-z0-9]+", " ", (text or "").lower()) + " "
    hits = [o for o in options if o != "unknown" and f" {o.replace('_', ' ')} " in t]
    return max(hits, key=len) if hits else None

QUESTIONS = {   # field -> question; options are appended from the schema at run time (only the parent's for dependent fields)
 "environment": "Where is this photo taken?", "weather": "What is the weather or lighting?",
 "ground_wet": "Is the ground wet?", "control": "What wheelchair control is visible?",
 "chair_subtype": "What kind of wheelchair is this?", "drive_position": "Where are the drive wheels of this power wheelchair?",
 "ramp_deploy": "Could a vehicle wheelchair ramp be deployed here, judging only from the scene?",
}
def prompt_for(field, options):
    return f"{QUESTIONS.get(field, field)} Answer with one of: {', '.join(o.replace('_', ' ') for o in options if o != 'unknown')}."

class Florence2VLM(_Base):
    kind = "vlm"
    def load(self):
        from transformers import AutoModelForCausalLM, AutoProcessor
        kw = _hf_kwargs(self.spec, self.cfg)
        self.p = AutoProcessor.from_pretrained(_src(self.spec), trust_remote_code=True, **kw)
        self.m = AutoModelForCausalLM.from_pretrained(_src(self.spec), trust_remote_code=True, **kw).eval().to(self.device)
    free_text = False   # Florence-2 answers task prompts, not free questions: we caption once and match options in it
    def caption(self, image):
        return self.ask(image, "<MORE_DETAILED_CAPTION>")
    def ask(self, image, question):
        import torch
        with torch.inference_mode():
            x = self.p(text=question, images=image, return_tensors="pt").to(self.device)
            ids = self.m.generate(input_ids=x["input_ids"], pixel_values=x["pixel_values"], max_new_tokens=self.spec.get("max_new_tokens", 24), do_sample=False, num_beams=1)
        return self.p.batch_decode(ids, skip_special_tokens=True)[0]

class Qwen2VLM(_Base):
    kind = "vlm"
    free_text = True
    def load(self):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        kw = _hf_kwargs(self.spec, self.cfg)
        self.p = AutoProcessor.from_pretrained(_src(self.spec), **kw)
        self.m = Qwen2VLForConditionalGeneration.from_pretrained(_src(self.spec), torch_dtype="auto", **kw).eval().to(self.device)
    def ask(self, image, question):
        import torch
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = self.p.apply_chat_template(msgs, add_generation_prompt=True)
        with torch.inference_mode():
            x = self.p(text=[text], images=[image], return_tensors="pt").to(self.device)
            ids = self.m.generate(**x, max_new_tokens=self.spec.get("max_new_tokens", 24), do_sample=False)
        return self.p.batch_decode(ids[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0]

if __name__ == "__main__":
    print(json.dumps(status(), indent=1))
