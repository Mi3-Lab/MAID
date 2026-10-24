# WheelView Annotate

An offline-first annotation and correction tool for the WheelView candidate pool (IEEE AgeTech Global Dataset
Competition 2026, Track 3). It consists of:

* **The PWA.** Uses a service worker and IndexedDB, and is installable.
* **A FastAPI + SQLite server.** It holds an append-only event log.
* **A git layout.** Each annotator appends to their own event file, so annotators never overwrite each other's work.

```
app/                    PWA: index.html, app.js, sw.js, manifest, adjudicate.html/js
server/                 main.py (FastAPI), wvcore.py (validation/consensus/kappa, shared with CI), schema.json
data/                   candidates.csv, LICENSES_draft.csv (from the WheelView candidate build)
annotations/events/     <annotator>.jsonl, append-only event logs: the source of truth in git
derived/                generated outputs (labels.csv, adjudication_needed.csv, per_annotator_labels.csv, kappa.json, progress.json). Do not edit.
scripts/                fetch_images.py, sync_events.py (export/import/sync), build_derived.py
tests/test_e2e.py       Playwright end-to-end test (two annotators, offline period, third annotator, adjudication, git round-trip)
.github/                rebuild-derived.yml workflow, CODEOWNERS
images/ var/            git-ignored: image files, SQLite DB, proxy cache, server-side exports
```

## Run locally (one command)
```bash
./run.sh            # http://127.0.0.1:8770  (or ./run.sh 9000; HOST=0.0.0.0 ./run.sh to expose on the LAN)
# or with Docker:
docker build -t wheelview-annotate . && docker run -p 8770:8000 -v $PWD/var:/srv/wheelview/var -v $PWD/images:/srv/wheelview/images wheelview-annotate
```
`run.sh` installs the Python dependencies if they are missing. If a WheelView build is present
(`WHEELVIEW_BUILD_DIR`, default `/workspace/wheelview`), it links the thumbnails into `images/`. It then imports any
committed `annotations/events/*.jsonl` (an idempotent step) and starts uvicorn.

Pages:
* `/` is the annotator app.
* `/adjudicate.html` is the adjudicator view. It also shows progress and kappa.
* `/export/labels.csv` and the other files under `/export/` are always rebuilt live.

## Annotating
* **Starting out.** Enter your name; there is no password. **Get batch** assigns about 30 images and caches them, with
  their metadata, for offline work. You can then label with no connection. Edits are queued in IndexedDB and pushed
  automatically on the browser's `online` event and every 15 s.
* **Fields.** These come from `server/schema.json`: device_type, occupied, propulsion, occlusion, viewpoint, lighting
  and setting.
  * There is also exclude-with-reason (child, clinical_setting, identifiable_person, not_a_photo, license_concern,
    duplicate), a *face blur missed* flag and optional bboxes.
  * **The option values are my working definition.** The report's Table 1 was not available when this was built.
    Edit `schema.json` to match Table 1 exactly; the UI, the server validation and CI all read it.
* **Prefill.** Weak labels from `proposed_class` appear as dashed suggestions. They are saved only when you click them
  or press **Done** (Enter).
* **Keyboard shortcuts.**

  | Key | Action |
  |---|---|
  | ↑ / ↓ | choose the field row |
  | 1–9 | pick a value (0 clears it) |
  | Enter | Done, then next image |
  | ← / → or p / n | previous / next image |
  | e, then 1–6 | exclude with that reason |
  | f | face-blur flag (a face is still recognisable) |
  | r | Review auto labels on/off |
  | a | (review) confirm all suggestions, then next |
  | o | (review) face-blur boxes overlay |
  | b | bbox mode |
  | Backspace | clear bboxes |
  | ? | help |
* **Corrections.** *My images* lists everything you have labelled; open any of them to change a field. Your newest
  event wins, but only for *you*.
* **Independence.** Annotators never see each other's labels.

## Review auto labels (machine pre-labels)
* **What it is.** `scripts/auto_annotate.py` pre-labels every local image with a local open-weights model (zero-shot
  SigLIP 2, `google/siglip2-so400m-patch14-224`, Apache-2.0; optional device boxes from OWLv2
  `google/owlv2-base-patch16-ensemble`, Apache-2.0). Everything runs on the machine itself; no image or candidate list
  is sent to any hosted API. Output: `annotations/auto/<model_tag>.jsonl` (per image and field: value, confidence, probs;
  screening scores; boxes) plus `<model_tag>.meta.json` (models, revisions, licences, prompts, thresholds, runs).
  It is incremental (embedding cache in `var/auto_cache/`), so re-running only processes new images:
  `pip install -r requirements-auto.txt` then `python scripts/auto_annotate.py [--limit N] [--detect] [--eval]`.
* **The model is not an annotator.** Its file is a separate source: it never appears in `annotations/events/`,
  never counts towards the 2-annotator consensus, kappa or `labels.csv`, and never excludes an image. Screening
  flags (likely child / clinical setting / not a photo / visible face) are suggestions for the human only.
  `derived/auto_vs_human.json` reports model-vs-human agreement per field as a separate metric (all done human
  labels, split into *independent* labels and labels made in review mode, plus acceptance rates of suggestions).
* **In the app.** Press **Review auto labels** (or `r`). Suggestions show as orange dashed options with the model's
  confidence (`AI: manual 87%`). `a` or Enter confirms every remaining suggestion as *your* label and moves on; fix a
  field first with ↑/↓ and 1–9. Images with a ⚠ screening flag need a second `a` (or exclude with `e`). `Shift+B`
  accepts the model's device boxes, `o` toggles the face-blur boxes. Queue order: lowest confidence first (default),
  untyped wheelchairs first, or random. Images you have already done are skipped and images that already have one
  human label come first, so pairs form. Review batches (with their suggestions) are cached for offline work.
* **Provenance.** Review-mode events are normal events under your name with extra keys
  `source: "auto_review"`, `model_tag`, `suggested` (the model's value) and `confidence`, so we can always tell
  confirmed-from-suggestion labels apart (anchoring bias) — and `auto_vs_human.json` does.
* **Server.** It loads `annotations/auto/*.jsonl` at startup and reloads whenever the files change (e.g. after the
  sync task's `git pull`). `WV_AUTO_DIR` / `WV_AUTO_MODEL` override the folder / active model. Endpoints:
  `GET /api/auto`, `GET /api/auto/{image_id}`, `GET /api/review/assign?annotator=&n=30&order=lowconf|wheelchair|random`.

## Face blurring
* `scripts/auto_blur.py` detects faces with **OpenCV YuNet** (`models/face_detection_yunet_2023mar.onnx`, MIT licence,
  committed; ~230 kB) tuned for recall: score ≥ 0.35 for small faces (≥ 0.72 for boxes wider than 6% of the image, which
  removed hands/balls/wheels), the image at 1×/2×/3× plus mirrored, merged with NMS. Each face is covered by a padded
  box (hair, ears, chin), pixelated to ~5 cells and Gaussian-blurred, so it is not recoverable.
* **Originals are never modified.** Blurred copies (re-encoded, no EXIF) go to `images_blurred/<id>.jpg`, which, like
  `images/`, is not in git. `data/blur_manifest.jsonl` (committed) logs per image: source checksum, number of faces,
  boxes and scores, detector, model checksum, thresholds, blur method and version. Resumable; re-running only does new
  or changed images. `pip install -r requirements-blur.txt` (OpenCV only), then `python scripts/auto_blur.py`.
* The server serves the blurred copy wherever one exists, so annotators see what will be released. If a face is still
  recognisable, press `f` (*face blur missed*); in review mode the detector's face boxes are drawn in magenta (`o`).
* `derived/attributions.csv` lists "faces blurred" as a modification only for images actually processed
  (column `face_blur`: done / no_faces_found / pending).
* On the laptop: `powershell -ExecutionPolicy Bypass -File scripts\windows\blur.ps1` (reuses the committed boxes when
  the image checksum matches; new detections go to the ignored `var\blur_manifest_local.jsonl`).

## Data model and consensus
* **Events.** Every edit is one event: `{uuid, image_id, annotator, field, value, ts, device_id, kind}`.
  * Inserts are `INSERT OR IGNORE` on uuid, so re-pushes are idempotent.
  * Nothing is ever updated or deleted.
  * The latest value per (image, annotator, field) is chosen by `ts`, with `uuid` as the tie-break.
* **Assignment.** Each image goes to 2 independent annotators. When their *done* labels disagree on a compared field,
  the image gets a third annotator, who is assigned first.
* **labels.csv status per image.** These are computed per field:
  * `consensus` when all done annotators agree
  * `majority` for 2 of 3
  * `adjudicated` when an adjudication event exists, which always wins
  * `conflict` otherwise
  * `face_blur_missed` is OR-combined across annotators, for safety.
  * If the image is excluded by consensus, disagreements on the other fields are marked `moot_excluded`.
* **Image status.** The overall status is one of `unlabelled`, `in_progress`, `needs_third_annotator`,
  `needs_adjudication` or `final`.
* **Kappa.** Cohen's kappa is computed per field in two ways:
  * pooled, using the first two annotators (alphabetical) of every image
  * pairwise, for every annotator pair with at least 5 shared images

  Kappa is meaningless until there are many items per field.

## Git workflow (see CONTRIBUTING.md)
* **The SQLite DB is the server's working copy.** The jsonl files in git are the durable, reviewable record.
  * `python scripts/sync_events.py export` appends the server's new events to `annotations/events/<annotator>.jsonl`.
    It deduplicates by uuid, so it never rewrites a line.
  * `python scripts/sync_events.py import` loads all jsonl files into the DB. It is idempotent, which also makes it the
    way to restore the DB.
  * `python scripts/sync_events.py sync [--push]` runs import, export and a local rebuild of `derived/`, then
    `git commit annotations/events`. It runs `git push` only with `--push` *and* a configured remote.
* **Rebuilding derived/.** `python scripts/build_derived.py` rebuilds `derived/` from the jsonl files alone, using the
  same code as the server, so the output is byte-identical (the e2e test checks this). The GitHub Actions workflow:
  * runs this on every push and PR
  * fails PRs that modify or remove existing event lines
  * commits `derived/` on `main`
* **Branches.** Use one branch per annotator (`annotate/<name>`) and PRs into `main`. Event files never conflict because
  each annotator writes only their own file; `.gitattributes` sets `merge=union` for the rare same-annotator case.

## Images are not in git
* **Fetching.** `scripts/fetch_images.py` fills `images/<image_id>.jpg`:
  * `--from-dir <wheelview build> [--link]` takes them from an existing thumbnail cache.
  * `--download [--source openverse] [--limit N]` takes them from `image_url`, politely: 1 request/s and backoff on 429.
    Images are resized to at most 640 px and re-encoded, which drops EXIF.
* **The server's /img/ proxy.** `/img/<id>` serves `images/`. Otherwise, by default (`WV_IMAGE_PROXY=1`), it fetches
  `image_url` once, caches the result in `var/imgcache/`, and backs off a host for 10 minutes after a 429.
* **Wikimedia Commons rows (2,528) have no local thumbnail yet.** `upload.wikimedia.org` was answering 429 to this build
  machine, so these rows are **not assigned by default**: only images that can already be served are assigned. To
  include them, either:
  * run `fetch_images.py --download --source commons` from a machine that is not rate-limited, or
  * set `WV_ASSIGN_WITHOUT_LOCAL_IMAGE=1` to rely on the proxy. In that case a batch only caches the images that the
    proxy managed to fetch.
* **Git LFS (optional).** If the team prefers images inside the repo, run `git lfs install && git lfs track "images/*.jpg"`,
  remove `images/` and `*.jpg` from `.gitignore`, and commit.
  * About 4,850 thumbnails are about 440 MB. That fits the GitHub LFS free quota only barely: 1 GB of storage and
    1 GB/month of bandwidth on free plans as of this writing, so check your org's plan.
  * Every clone that pulls LFS uses up bandwidth.
  * The license obligations (attribution in `data/LICENSES_draft.csv`) still apply to redistributed copies.

## Backing up SQLite
The jsonl files are a complete backup of all labels: `sync_events.py export`, then commit. To restore, run
`sync_events.py import` against an empty DB. Assignments are not exported; they are re-derived as people request new
batches. For a hot copy of the DB itself:
```bash
sqlite3 var/wheelview.db ".backup 'backup/wheelview-$(date +%F-%H%M).db'"     # safe while the server runs (WAL)
# cron example: */30 * * * *  cd /srv/wheelview-annotate && python3 scripts/sync_events.py sync --push && sqlite3 var/wheelview.db ".backup var/backup.db"
```
Litestream (continuous replication to S3-compatible storage) is a good fit for a hosted instance.

## Hosting publicly (not done; needs an account or a server)
Requirements:
* **HTTPS.** Service workers and `crypto.randomUUID` only work on HTTPS or localhost.
* **Persistent disk** for `var/`.
* **The images.** Either run `fetch_images.py --download` on the host or mount a volume.
* **Some access control.** Names are not authentication. At minimum, put the app behind your institution's SSO, an
  HTTP basic-auth reverse proxy, Cloudflare Access, or a Tailscale/VPN.

| Option | Notes |
|---|---|
| **UC Merced server / VM** | `HOST=0.0.0.0 ./run.sh 8770` under systemd, with nginx or Caddy in front for TLS, campus SSO or basic auth, cron backups. Simplest for data governance. |
| **Fly.io** | `fly launch` (uses the Dockerfile), `fly volumes create wv_data`, mounted at `/srv/wheelview/var`. Images go in a second volume or are fetched on boot. Single machine only (SQLite). |
| **Render** | Docker web service plus a persistent disk at `/srv/wheelview/var`. Persistent disks require a paid instance; the free tier's filesystem is ephemeral and sleeps when idle. |
| **Railway** | Docker deploy plus a volume at `/srv/wheelview/var`. |
| **Hugging Face Spaces** | Docker Space with `PORT=7860`. Storage is ephemeral unless you buy persistent storage, so run `sync_events.py sync --push` frequently (it needs a deploy key) or attach persistent storage. Make the Space private. |

Run only **one** server instance per SQLite file. For multiple sites, run separate servers and merge through git: each
server exports its events, and `import` on the others unions them.

## Hosted deployment (Hugging Face Space)

- **App:** Docker Space `samitesla/wheelview-annotate` at https://samitesla-wheelview-annotate.hf.space. It is built from this repo's `Dockerfile` on port 7860. Deploy with `python scripts/deploy_space.py`, which pushes `git archive HEAD` plus the Space front-matter.
- **Data:** private dataset `samitesla/wheelview-data`.
  - `images/` holds the thumbnails. The Space downloads them at startup.
  - `annotations/events/*.jsonl` is the durable copy of all events.
- **Persistence:** Space disk is ephemeral, so `server/hf_persist.py` takes over when `WV_HF_DATASET` is set.
  - At startup it restores the events into SQLite. The restore is idempotent, and the server refuses to start if it fails.
  - It exports new events to the per-annotator jsonl every 30 s.
  - A `CommitScheduler` commits them to the dataset every 2 minutes and again on shutdown.
  - `POST /api/admin/flush` forces a commit.
- **Access:** set `WV_ACCESS_CODE` (a Space secret) to require a team code once per browser.
  - The code is entered on `/unlock.html` and stored in an HttpOnly cookie that lasts 400 days.
  - The service worker serves the cached app offline after that.
  - Scripted clients can send the code in an `X-Access-Code` header instead.
- **GitHub sync:** `.github/workflows/sync-from-hf.yml` runs every 30 minutes and on demand.
  - It pulls `annotations/events/*.jsonl` from the dataset and appends only unseen uuids (`scripts/pull_hf_events.py`).
  - It rebuilds `derived/` and commits to `main`. It needs the repo secret `HF_TOKEN`.
- **Space settings:**
  - secrets: `HF_TOKEN`, `WV_ACCESS_CODE`
  - variable: `WV_HF_DATASET=samitesla/wheelview-data`

## Self-hosting on a Windows laptop (current deployment)

- `run.ps1` creates `.venv`, installs the requirements, and imports the committed events.
  - It then serves on `127.0.0.1:8770` and restarts uvicorn if it exits.
  - It reads `ACCESS_CODE.txt`, which is gitignored. HF persistence stays off.
- `scripts\windows\tunnel.ps1` runs a Cloudflare quick tunnel (no account).
  - It writes the current HTTPS URL to `var\public_url.txt`.
  - Single instance: a named mutex (`WheelViewTunnelLoop`) plus `var\tunnel.lock` (PID-checked) make a second loop exit immediately. On startup it kills stray `cloudflared` processes from `var\bin` (or pointing at this port) left by dead loops. `run.ps1` has the same mutex guard (`WheelViewServerLoop<port>`).
  - Health check: every 60 s it fetches `<public url>/api/health`. After 3 consecutive failures while the local server is up, it kills cloudflared and starts a new tunnel (new URL in `var\public_url.txt`, log in `var\tunnel.log`). A dead local server doesn't count against the tunnel.
  - `Stop-ScheduledTask` kills only the `wscript` wrapper and can orphan the loop. To stop it for real, kill the `powershell` process whose command line contains `tunnel.ps1`.
  - The URL changes whenever cloudflared restarts, and each new URL is a new origin, so annotators must reopen it and unlock again. A stable URL needs a free Cloudflare named tunnel or a Tailscale Funnel account.
- `scripts\windows\sync.ps1` runs every 30 minutes. It exports events, commits, does `git pull --rebase` and pushes. `rebuild-derived.yml` then rebuilds `derived/`.
- `scripts\windows\install-tasks.ps1` registers the scheduled tasks.
  - Run it elevated to also start the server and tunnel at boot without a logon.
  - Run it with `-Uninstall` to remove the tasks.
- `scripts\windows\power-apply.ps1` and `power-revert.ps1` change only the AC settings:
  - never sleep
  - never hibernate
  - lid close does nothing

  The original values are saved in `var\power_original.json`.
- Images: unpack the thumbnails into `images\<candidate_id>.jpg`. Commons images are proxied lazily.

## Image attributions / citations
`derived/ATTRIBUTIONS.md`, `derived/attributions.csv` and `derived/attributions.bib` are rebuilt by CI on every push
(`scripts/build_attributions.py`, offline). They list the exact credit line for each image, images whose authors ask for a
specific credit or carry restrictions, and the images currently in use. When candidates are added, refresh the metadata
cache by hand with `python scripts/fetch_attribution_meta.py` (network; incremental; polite to the Commons API) and push.
