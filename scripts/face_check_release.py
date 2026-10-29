#!/usr/bin/env python3
"""Second, independent face pass over the FINAL release copies (lab laptop; images never leave var/).

  python scripts/face_check_release.py var/release/WheelView_v1.0 [--score 0.5]

Runs OpenCV YuNet (models/face_detection_yunet_2023mar.onnx) on every released JPEG at 1x and 2x and on the mirrored
image. A detection whose centre lies inside a region that the blur step recorded (data/blur_manifest.jsonl pad_box) is
'inside_blur' (the detector firing on an already-blurred face); anything else is 'outside_blur' and is written, enlarged,
to var/face_check/sheet_NN.jpg for a person to inspect. Writes var/face_check/report.json. Read-only on the release."""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "face_detection_yunet_2023mar.onnx"

def detect(img, score):
    h, w = img.shape[:2]; out = []
    for s in (1.0, 2.0):
        im = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC) if s != 1 else img
        for mir in (False, True):
            x = cv2.flip(im, 1) if mir else im
            d = cv2.FaceDetectorYN.create(str(MODEL), "", (x.shape[1], x.shape[0]), score, 0.3, 5000)
            _, f = d.detect(x)
            for r in (f if f is not None else []):
                bx, by, bw, bh = r[:4] / s
                if mir: bx = w - bx - bw
                out.append((float(bx / w), float(by / h), float(bw / w), float(bh / h), float(r[-1])))
    keep = []   # merge overlapping hits
    for b in sorted(out, key=lambda b: -b[4]):
        if all(abs((b[0] + b[2] / 2) - (k[0] + k[2] / 2)) > k[2] / 2 or abs((b[1] + b[3] / 2) - (k[1] + k[3] / 2)) > k[3] / 2 for k in keep): keep.append(b)
    return keep

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("release"); ap.add_argument("--score", type=float, default=0.5)
    a = ap.parse_args(); R = Path(a.release)
    B = {}
    for l in open(ROOT / "data" / "blur_manifest.jsonl", encoding="utf-8"):
        d = json.loads(l); B[d["image_id"]] = [f["pad_box"] for f in d.get("faces", [])]
    out = ROOT / "var" / "face_check"; out.mkdir(parents=True, exist_ok=True)
    rep = {"score_threshold": a.score, "images": 0, "detections": 0, "inside_blur": 0, "outside_blur": []}; crops = []
    for p in sorted((R / "images").rglob("*.jpg")):
        img = cv2.imread(str(p)); h, w = img.shape[:2]; rep["images"] += 1
        for x, y, bw, bh, sc in detect(img, a.score):
            rep["detections"] += 1; cx, cy = x + bw / 2, y + bh / 2
            if any(px0 <= cx <= px1 and py0 <= cy <= py1 for px0, py0, px1, py1 in B.get(p.stem, [])): rep["inside_blur"] += 1; continue
            rep["outside_blur"].append({"image_id": p.stem, "box": [round(x, 4), round(y, 4), round(bw, 4), round(bh, 4)], "score": round(sc, 3), "crop": len(crops)})
            m = 0.6; X0, Y0 = int(max(0, (x - bw * m) * w)), int(max(0, (y - bh * m) * h)); X1, Y1 = int(min(w, (x + bw * (1 + m)) * w)), int(min(h, (y + bh * (1 + m)) * h))
            c = cv2.resize(img[Y0:Y1, X0:X1], (200, 200), interpolation=cv2.INTER_CUBIC)
            cv2.putText(c, f"{len(crops)} {p.stem[3:]} {sc:.2f}", (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1); crops.append(c)
    for k in range(0, len(crops), 30):
        ch = crops[k:k + 30] + [np.zeros((200, 200, 3), np.uint8)] * (-len(crops[k:k + 30]) % 6)
        sheet = np.vstack([np.hstack(ch[i:i + 6]) for i in range(0, len(ch), 6)]); cv2.imwrite(str(out / f"sheet_{k // 30:02d}.jpg"), sheet)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in rep.items()}), "sheets:", (len(crops) + 29) // 30)

if __name__ == "__main__":
    main()
