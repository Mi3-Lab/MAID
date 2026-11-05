# Scene backgrounds for scripts/synth_augment.py

Put licensed, **people-free** scene photos here (curbside, crossing, sidewalk, ramp_lift, vehicle_door, parking_lot)
and add one row per photo to `backgrounds.csv`. The photos themselves are gitignored; only this README and the
manifest are committed.

Columns: `background_id, file, scene, source_url, license, author, people, ground_y, environment, dataset_image_id, notes`

* `people` must be `none`; rows with anything else are refused. Never take, generate or download photos of people for this.
* `license`, `author`, `source_url` are required (e.g. CC0 / CC BY with attribution).
* `ground_y` = height fraction of the horizon line (where the ground plane meets the horizon), e.g. `0.45`.
* `dataset_image_id` (optional, leave `file` empty): use an approved **train** image of this dataset as the background.
  Its license/author/url are taken from the manifest. Val/test images are refused.

Check: `python scripts/synth_augment.py --check-backgrounds`
