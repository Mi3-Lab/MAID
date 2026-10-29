#!/usr/bin/env python3
"""Minimal loader for the WheelView v1.0 release (standard library + Pillow; torch optional).

    from wheelview_loader import WheelView
    wv = WheelView("path/to/WheelView_v1.0")
    wv.summary()                               # counts per split and class
    for img, row in wv.iter("train"):          # PIL image + labels.csv row (dict)
        ...
    ds = wv.torch_dataset("val", transform)    # torch.utils.data.Dataset of (tensor, class_index)

Command line:
    python code/wheelview_loader.py --root . --summary
    python code/wheelview_loader.py --root . --flatten flat_images   # hard-links (or copies) every image to flat_images/<id>.jpg,
                                                                     # the layout the training scripts in code/scripts expect
Labels: device_type is one of manual (manual wheelchair), powered (powered wheelchair), other_aid (other mobility aid such as
a mobility scooter, walker or rollator) and ambiguous (cannot be told apart from the image alone). Rows with split == "excluded" (train_eligible 0) are shipped with their exclusion_reason and are never used for training or
evaluation. Boxes: annotations/boxes_per_annotator.csv ([x, y, w, h], normalised 0-1, top-left origin) and
annotations/coco_<split>.json (pixel boxes, marked preliminary)."""
import argparse, csv, os, shutil, sys
from collections import Counter
from pathlib import Path

CLASSES = ["manual", "powered", "other_aid", "ambiguous"]   # same order as code/server/schema.json
SPLITS = ("train", "val", "test")

class WheelView:
    def __init__(self, root="."):
        self.root = Path(root)
        with open(self.root / "annotations" / "labels.csv", encoding="utf-8", newline="") as f:
            self.rows = list(csv.DictReader(f))
        self.by_id = {r["image_id"]: r for r in self.rows}
        self.splits = {}
        for s in SPLITS:
            p = self.root / "splits" / f"{s}.txt"
            self.splits[s] = p.read_text(encoding="utf-8").split() if p.exists() else []

    def ids(self, split):
        return list(self.splits[split]) if split in SPLITS else [r["image_id"] for r in self.rows if r["split"] == split]

    def path(self, image_id):
        return self.root / self.by_id[image_id]["file"]

    def label(self, image_id):
        return self.by_id[image_id]["device_type"]

    def iter(self, split):
        from PIL import Image
        for i in self.ids(split):
            with Image.open(self.path(i)) as im:
                yield im.convert("RGB"), self.by_id[i]

    def torch_dataset(self, split, transform=None):
        import torch.utils.data as tud
        from PIL import Image
        ids, wv = self.ids(split), self
        class _DS(tud.Dataset):
            def __len__(self): return len(ids)
            def __getitem__(self, k):
                im = Image.open(wv.path(ids[k])).convert("RGB")
                return (transform(im) if transform else im), CLASSES.index(wv.label(ids[k]))
        return _DS()

    def summary(self):
        for s in SPLITS + ("excluded",):
            ids = self.ids(s); c = Counter(self.by_id[i]["device_type"] or "-" for i in ids)
            print(f"{s:9s} {len(ids):4d}  " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))

    def flatten(self, out):
        out = Path(out); out.mkdir(parents=True, exist_ok=True); n = 0
        for r in self.rows:
            src, dst = self.root / r["file"], out / f"{r['image_id']}.jpg"
            if dst.exists(): continue
            try: os.link(src, dst)
            except OSError: shutil.copy2(src, dst)
            n += 1
        print(f"{n} images linked/copied into {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=".")
    ap.add_argument("--summary", action="store_true"); ap.add_argument("--flatten", default=None)
    a = ap.parse_args(); wv = WheelView(a.root)
    if a.flatten: wv.flatten(a.flatten)
    if a.summary or not a.flatten: wv.summary()
