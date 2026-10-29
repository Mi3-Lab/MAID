# Contributing to WheelView Annotate

## Golden rules
1. **Never edit `derived/` by hand.** It is rebuilt from `annotations/events/*.jsonl` by `scripts/build_derived.py`
   (locally) and by `.github/workflows/rebuild-derived.yml` (on every push to `main`).
2. **Event files are append-only.** `annotations/events/<you>.jsonl` gets new lines only; never edit or delete a line.
   To correct a label, make a new edit in the app (a newer event supersedes your older one). CI rejects PRs that change
   or remove existing lines.
3. **Only touch your own file.** Your annotator id (the name you type in the app, lower-cased) is your file name.
   Adjudicators write adjudication events (`"kind": "adjudication"`) into *their own* file too.
4. **No images in git.** Images are fetched by `scripts/fetch_images.py` into the ignored `images/` folder.
5. **Machine labels are not annotations.** `annotations/auto/*.jsonl` (from `scripts/auto_annotate.py`) is a separate
   machine source: never copy it into `annotations/events/`, and never use a model tag as an annotator name
   (`build_derived.py` rejects it). Confirming suggestions in *Review auto labels* mode writes *your* events with
   `source: "auto_review"` provenance; that is fine, but label honestly — fix anything the model got wrong, and
   prefer normal mode when you want an independent label. Screening flags never exclude an image by themselves.
6. **Faces.** Annotate on the face-blurred copies the app shows; press `f` whenever a face is still recognisable.
   Never commit images, blurred or not (`images/`, `images_blurred/` are ignored).
7. Don't look at other annotators' labels before you finish yours (the app never shows them to you); disagreements are
   resolved in the adjudicator view (`/adjudicate.html`).

## Workflow: one branch per annotator, PRs into `main`
```bash
git checkout main && git pull
git checkout -b annotate/<you>          # once; reuse it afterwards (rebase/merge main into it as needed)
# ... annotate in the app (offline is fine; it syncs to the team server when online) ...
python scripts/sync_events.py export --annotator <you>    # server DB -> annotations/events/<you>.jsonl (append-only)
git add annotations/events/<you>.jsonl && git commit -m "annotations: <you> <date>"
git push -u origin annotate/<you>       # then open a PR into main
```
`python scripts/sync_events.py sync --push` does import + export + local rebuild + commit (and pushes only if a remote
exists). On a shared server, a maintainer can run `sync` for everybody on a `sync/server` branch instead.

Because every annotator appends to a different file, PRs from different annotators never conflict. If you annotate the
same id on two machines/branches, `.gitattributes` sets `merge=union` for event files, and readers de-duplicate by uuid.

## Changing the label schema
`server/schema.json` is the single source of truth for the UI, server validation and derivation. Adding options is
safe. Removing options makes old events fail validation, so add a migration note and get a review from CODEOWNERS.
To rename an option, never rewrite events: change the option in `fields`, add `{"<field>": {"<old>": "<new>"}}` to
`value_aliases`, and bump `version`. `wvcore.validate` still accepts the old value (offline clients may queue it) and
every reader maps it through `wvcore.canon` (example: `device_type` `lookalike` -> `ambiguous`, 2026-10-28).

## Recommended GitHub settings (private repo)
* Branch protection on `main`: require PRs, 1 approving review (CODEOWNERS), the status check
  *validate events & rebuild derived outputs*, and linear history; block force-pushes and deletions.
* Allow the workflow's `GITHUB_TOKEN` to push `derived/` to `main`: under *Settings → Actions → Workflow permissions*
  choose "Read and write". If branch protection blocks the bot, either add it as a bypass actor or change the last
  workflow step to upload `derived/` as an artifact only, which the workflow already does.
* Replace the placeholder handles in `.github/CODEOWNERS`.

## Tests
```bash
pip install -r requirements-dev.txt && python -m playwright install chromium
python tests/test_e2e.py         # spins up its own server on :8812 with a temp DB (incl. review mode + blurred images)
```
