#!/usr/bin/env python3
"""make_splits includes v1.1 human annotators."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import make_splits as MS
assert "devin" in MS.HUMANS and "wesley" in MS.HUMANS and "samir" in MS.HUMANS
# assign still groups
rows = [{"image_id": f"wv_{i:06d}", "label": "manual" if i%2==0 else "powered", "group": f"g{i//3}"} for i in range(1,31)]
asg = MS.assign(rows, 20261106, (0.7,0.15,0.15))
assert set(asg.values()) <= {"train","val","test"}
print("PASS make_splits humans + assign")
