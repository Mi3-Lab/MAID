# Annotation assist models (MAID v1.1)

Optional models that **suggest** boxes, masks and attribute values to speed up annotation. Suggestions are shown in
orange with the model name, stored separately from human labels with model name / version, and are **never** human
labels. A suggestion becomes a label only when a human picks it, and that event records `source=auto_attr` + `model_tag`.

| kind | default | alternatives | output |
|---|---|---|---|
| detector | OWLv2 `google/owlv2-base-patch16-ensemble` (open-vocabulary) | Grounding DINO `IDEA-Research/grounding-dino-tiny`, YOLO-World `yolov8s-worldv2.pt` (ultralytics) | `var/assist/<tag>/boxes.jsonl` |
| segmenter | SAM 2 `facebook/sam2.1-hiera-small` (box prompt) | — | `var/assist/<tag>/masks/*.png` + `masks.jsonl` |
| vlm | Florence-2 `microsoft/Florence-2-base` (caption -> matched options) | Qwen2-VL `Qwen/Qwen2-VL-2B-Instruct` (VQA per field) | `annotations/auto/<tag>_attr-vlm-v1.jsonl` + `.meta.json` |

Config: `config/assist_models.json` (or `WV_ASSIST_CONFIG`). Each kind has `enabled, backend, model, revision`
(optional pin) and `path` (optional local folder). `device: auto|cpu|cuda`. **`allow_download: false`** by default:
models load from the local Hugging Face cache only, so nothing large is downloaded by accident.

## Design
* **Pluggable:** one wrapper class per backend in `scripts/assist_models.py`; switch in the config.
* **Lazy + cached:** nothing heavy is imported until first use; each model loads once per process.
* **Degrades gracefully:** missing packages, missing weights or load errors -> the model is skipped with a reason
  (`python scripts\assist_models.py`, `GET /api/assist/status`); the app works exactly as before.
* **Inference never sees labels:** `assist_annotate.py` lists images from the image folder (or `--ids`) and reads pixels
  of the face-blurred copy only: no events, folds, manifests, candidates or label files. `check_leakage.py` audits
  this statically.
* VLM questions and parsed answers are limited to schema options; dependent fields use the parent's list (subtype asked
  only after "manual or powered?", drive position only for powered). `unknown` and `pediatric_size` are never suggested.
  Florence-2 only captions, so subtype / drive position / ramp are left to Qwen2-VL or SigLIP.
* Boxes/masks go to `var/assist/` (gitignored), not `annotations/auto/` (whose non-attribute files drive review mode).

## Fetch the weights (once, manually; never committed)
```powershell
.venv\Scripts\pip.exe install -r requirements-auto.txt        # torch + transformers (as for auto_annotate.py)
.venv\Scripts\pip.exe install "transformers>=4.47" timm einops  # SAM 2 / Florence-2 / Qwen2-VL need a recent transformers
# then download into the HF cache (needs network; uses your HF login if the model requires one)
.venv\Scripts\huggingface-cli.exe download google/owlv2-base-patch16-ensemble
.venv\Scripts\huggingface-cli.exe download facebook/sam2.1-hiera-small
.venv\Scripts\huggingface-cli.exe download microsoft/Florence-2-base
# optional: IDEA-Research/grounding-dino-tiny, Qwen/Qwen2-VL-2B-Instruct (~4.5 GB); YOLO-World: pip install ultralytics
```
Sizes (approx.): OWLv2-base 0.6 GB, SAM 2.1 small 0.2 GB, Florence-2-base 0.5 GB, Qwen2-VL-2B 4.5 GB. Alternatively set
`allow_download: true` for one run. Weights, caches and `*.pt/*.safetensors/*.bin/*.onnx` are gitignored.

## Run
```powershell
.venv\Scripts\python.exe scripts\assist_models.py                                    # status of each model
.venv\Scripts\python.exe scripts\assist_annotate.py --dry-run
.venv\Scripts\python.exe scripts\assist_annotate.py --tasks vlm --limit 50            # attribute hints
.venv\Scripts\python.exe scripts\assist_annotate.py --tasks detect,segment            # boxes + masks (for synth_augment)
.venv\Scripts\python.exe scripts\assist_annotate.py --ids var\todo.txt --tasks vlm   # e.g. ids from filter_images.py
```
The server picks up new files in annotations/auto automatically; reload the app to see VLM hints in the Attributes v1.1 pass.

## SigLIP (existing) attr-v2
`scripts\auto_annotate.py --attr` (SigLIP2 zero-shot) now writes `..._attr-v2.jsonl`: chair_subtype is scored within the
manual or powered list chosen by a manual-vs-powered prompt pair, plus drive_position, ramp_deploy and per-reason
ramp_reasons scores. `train_multitask.py` trains subtype / drive / ramp heads on human labels only.
