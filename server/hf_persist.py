"""Optional persistence for ephemeral hosts (Hugging Face Spaces).
Enabled when WV_HF_DATASET (e.g. "user/wheelview-data") and HF_TOKEN are set.
 * startup : download annotations/events/*.jsonl from the private dataset repo and import them into SQLite
             (INSERT OR IGNORE by uuid -> idempotent). If this fails the server refuses to start, so an empty
             local copy can never be pushed over the remote files.
 * running : every WV_EXPORT_EVERY_S (30 s) append new DB events to the local per-annotator jsonl files (append-only,
             dedup by uuid); a huggingface_hub CommitScheduler commits changed files every WV_HF_COMMIT_MIN (2) minutes.
 * shutdown: final export + synchronous commit.
 * images  : snapshot_download of images/*.jpg into WV_IMAGE_DIR in a background thread."""
import logging, os, sys, threading, time
from pathlib import Path

log = logging.getLogger("wheelview.hf")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import sync_events  # noqa: E402  (export/import shared with the git workflow)

class HFPersistence:
    def __init__(self, repo_id, token, db_path, work_dir, image_dir):
        self.repo_id, self.token, self.db_path = repo_id, token, str(db_path)
        self.work_dir = Path(work_dir); self.events_dir = self.work_dir / "annotations" / "events"
        self.image_dir = Path(image_dir)
        self.scheduler = None; self._stop = threading.Event()
        self.state = {"restored_events": None, "images_ready": False, "images": 0, "last_export": None,
                      "last_export_new": 0, "image_error": None, "export_error": None}

    def restore(self):
        from huggingface_hub import snapshot_download
        self.events_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(self.repo_id, repo_type="dataset", local_dir=self.work_dir, token=self.token,
                          allow_patterns=["annotations/events/*.jsonl"])
        added, bad = sync_events.import_(self.db_path, self.events_dir)
        self.state["restored_events"] = added
        log.warning("restored %d events from %s (%d invalid lines skipped)", added, self.repo_id, bad)

    def start(self):
        from huggingface_hub import CommitScheduler
        self.scheduler = CommitScheduler(repo_id=self.repo_id, repo_type="dataset", folder_path=self.events_dir,
                                         path_in_repo="annotations/events", allow_patterns=["*.jsonl"], private=True,
                                         every=float(os.environ.get("WV_HF_COMMIT_MIN", 2)), token=self.token)
        threading.Thread(target=self._export_loop, daemon=True, name="hf-export").start()
        threading.Thread(target=self._images, daemon=True, name="hf-images").start()

    def export_now(self):
        lock = self.scheduler.lock if self.scheduler else threading.Lock()
        with lock:
            n = sync_events.export(self.db_path, self.events_dir)
        self.state.update(last_export=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), last_export_new=n)
        return n

    def _export_loop(self):
        every = float(os.environ.get("WV_EXPORT_EVERY_S", 30))
        while not self._stop.wait(every):
            try: self.export_now(); self.state["export_error"] = None
            except Exception as e: self.state["export_error"] = repr(e); log.exception("export failed")

    def _images(self):
        from huggingface_hub import snapshot_download
        for attempt in range(5):
            try:
                snapshot_download(self.repo_id, repo_type="dataset", local_dir=self.image_dir.parent, token=self.token,
                                  allow_patterns=[f"{self.image_dir.name}/*.jpg"], max_workers=16)
                self.state["images"] = len(list(self.image_dir.glob("*.jpg"))); self.state["images_ready"] = True
                self.state["image_error"] = None; log.warning("images ready: %d", self.state["images"]); return
            except Exception as e:
                self.state["image_error"] = repr(e); log.exception("image download failed (attempt %d)", attempt + 1)
                time.sleep(20 * (attempt + 1))

    def push_now(self):
        self.export_now()
        if self.scheduler: self.scheduler.trigger().result(timeout=120)

    def stop(self):
        self._stop.set()
        try: self.push_now(); log.warning("final export + commit done")
        except Exception: log.exception("final push failed")
        if self.scheduler: self.scheduler.stop()
