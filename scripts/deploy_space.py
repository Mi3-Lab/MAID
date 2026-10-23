#!/usr/bin/env python3
"""Push the git-tracked tree (HEAD) to the Hugging Face Docker Space, with a Space README front-matter.
Usage: HF_TOKEN=... python scripts/deploy_space.py [--space samitesla/wheelview-annotate] [--dataset samitesla/wheelview-data]
Secrets (HF_TOKEN, WV_ACCESS_CODE) are set separately with HfApi.add_space_secret, never stored in the repo."""
import argparse, os, shutil, subprocess, tempfile
from pathlib import Path
from huggingface_hub import HfApi
ROOT = Path(__file__).resolve().parent.parent
FRONT = """---
title: WheelView Annotate
emoji: ♿
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
Offline-first annotation tool for the WheelView dataset. Requires a team access code.
Source: private GitHub repo samir-shah-ahmed/wheelview. Data and annotation events live in the private dataset {dataset}.

"""
ap = argparse.ArgumentParser(); ap.add_argument("--space", default="samitesla/wheelview-annotate")
ap.add_argument("--dataset", default="samitesla/wheelview-data"); a = ap.parse_args()
stage = Path(tempfile.mkdtemp(prefix="space-"))
subprocess.run(f"git -C {ROOT} archive HEAD | tar -x -C {stage}", shell=True, check=True)
for p in (".github",): shutil.rmtree(stage / p, ignore_errors=True)
readme = stage / "README.md"; readme.write_text(FRONT.format(dataset=a.dataset) + readme.read_text())
sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
api = HfApi(token=os.environ["HF_TOKEN"])
r = api.upload_folder(repo_id=a.space, repo_type="space", folder_path=stage, delete_patterns=["*"], commit_message=f"Deploy {sha} from GitHub main")
print("deployed", sha, "->", r)
