#!/usr/bin/env python3
"""Merge annotations/events/*.jsonl from the HF dataset repo into the git checkout, append-only (union by uuid).
Existing lines are never modified or reordered; remote lines with unseen uuids are appended in remote order.
Env: HF_TOKEN, WV_HF_DATASET (default samitesla/wheelview-data)."""
import json, os, sys, tempfile
from pathlib import Path
from huggingface_hub import snapshot_download
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server")); import wvcore

def main():
    repo = os.environ.get("WV_HF_DATASET", "samitesla/wheelview-data")
    local = ROOT / "annotations" / "events"; local.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    snapshot_download(repo, repo_type="dataset", local_dir=tmp, allow_patterns=["annotations/events/*.jsonl"], token=os.environ.get("HF_TOKEN"))
    added = skipped = 0
    for rf in sorted((tmp / "annotations" / "events").glob("*.jsonl")):
        lf = local / rf.name
        have = set()
        if lf.exists():
            for line in lf.open():
                if line.strip():
                    try: have.add(json.loads(line)["uuid"])
                    except Exception: pass
        new = []
        for line in rf.open():
            if not line.strip(): continue
            try: e, why = wvcore.validate(json.loads(line))
            except json.JSONDecodeError: e, why = None, "json"
            if not e or e["annotator"] != rf.stem: skipped += 1; continue
            if e["uuid"] in have: continue
            have.add(e["uuid"]); new.append(json.dumps({k: e[k] for k in sorted(e)}, sort_keys=True, separators=(",", ":")))
        if new:
            with lf.open("a") as fh:
                if lf.stat().st_size and not lf.read_bytes().endswith(b"\n"): fh.write("\n")
                fh.write("\n".join(new) + "\n")
        added += len(new); print(f"{rf.name}: +{len(new)}")
    print(f"merged {added} new events from {repo} ({skipped} invalid lines skipped)")
    if os.environ.get("GITHUB_OUTPUT"):
        open(os.environ["GITHUB_OUTPUT"], "a").write(f"added={added}\n")

if __name__ == "__main__": main()
