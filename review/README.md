# WheelView offline review (lab members only)

This folder is a review viewer for the current clean labeled set: every image Samir or Shaaz marked done and nobody excluded, plus images whose exclude was adjudicated to keep (never for child or identifiable_person; license_concern only after the license audit cleared it). It runs entirely on your own computer. You don't need a server, tunnel, access code, or internet connection, and nothing you do in it is uploaded anywhere.

## What you need

1. **The repo** (labels, license status, the viewer):

   ```
   git clone https://github.com/samir-shah-ahmed/wheelview.git   # first time (private repo; ask Samir for access)
   cd wheelview
   git pull                                                     # later, to get the newest labels
   ```

2. **The images folder** (face-blurred copies only). These are **not** in Git. Samir hands them over directly as `WheelView_review_images.zip`, either on a USB drive or in person.
   Unzip it so the files end up at `review/images/wv_000233.jpg`, `review/images/wv_000234.jpg`, and so on:

   ```
   wheelview/
     review/
       index.html
       review_data.js
       images/          <- unzip here (the zip already contains an images/ folder, so unzip it into review/)
         wv_000233.jpg
         ...
   ```

   `review/images/` is in `.gitignore`. Please **do not** commit, upload, or share these images (GitHub, Overleaf, Drive, Slack, email, etc.). The team rule is that images, blurred copies included, stay inside the lab.

## Open it

Double-click `review/index.html`, or open it from your browser with File > Open. It works from `file://`. If an image shows "not found", the images folder isn't in `review/images/`.

## What you see

- **Summary at the top:** counts by class, single vs. double labeled, license status, and model warnings.
- **Filters:** class (device_type), license status, single/double labeled, model warning type, your own review mark, and id/creator search.
- **Per image:**
  - the consensus label and each labeler's values for device_type, propulsion, occupied, setting, viewpoint, lighting, and occlusion (disagreements are shown in red)
  - license, creator, and a source link
  - license status with the reason and the live Flickr re-check result
  - model warnings (child, stroller, and so on), which are for **information only**. A model never removes an image a human labeled.
- **License status meanings** (from `data/license_audit_v2.csv`):
  - `ok`: allowed license (CC0, PDM, CC BY, or CC BY-SA), still matches on Flickr.
  - `ok_with_note`: usable, but read the note (for example, the Flickr photo was deleted after collection, or the license changed to another allowed one).
  - `needs_ross`: Ross must decide (for example, an NC license, a page that now needs a login, or a license changed to all-rights-reserved after we collected it).
  - `exclude`: don't use.

## Marking images (optional)

Type your name in **Reviewer**, then use the keyboard:

- <kbd>O</kbd>: OK
- <kbd>Q</kbd>: question
- <kbd>U</kbd>: clear
- <kbd>→</kbd>/<kbd>J</kbd>: next
- <kbd>←</kbd>/<kbd>K</kbd>: previous

You can also add a note. Marks are saved only in this browser (localStorage). Click **Export my marks (CSV)** and give the CSV to Samir. That CSV contains only ids, marks, and notes, never images.

## Rebuilding the data (Samir)

```
python scripts/build_review.py                 # reads var/wheelview.db if present, else annotations/events
```

This regenerates `review/review_data.js` from the labels, `data/license_audit_v2.csv`, and `data/v1_triage.csv`. It never changes labels, events, or blocklists.
