# Datasheet for WheelView v1.0

Following the questions of Gebru et al., *Datasheets for Datasets* (CACM 2021). Numbers refer to v1.0 (29 October
2026). The dataset report (PDF) gives the full method and the baselines.

## Motivation

**Why was the dataset created?** Accessible vehicles prepare differently for a manual and a powered wheelchair (ramp
or lift, interior space, securement). Widely used vision datasets have no wheelchair class (COCO) or a single one
(nuScenes, Open Images boxes). WheelView provides images labeled for wheelchair *type* so that a perception system
can tell which device is approaching. It was prepared for the IEEE AgeTech Global Dataset Competition 2026, Track 3
(Mobility & Assistive Technologies).

**Who created it?** Samir Ahmed, Shaaz Saleem, Wesley Maia and Ross Greer, Mi3 Lab, Department of Computer Science and
Engineering, University of California, Merced. Contact: Ross Greer, rossgreer@ucmerced.edu.

**Funding.** No grant for this dataset is recorded in the project files. Ask the contact above.

## Composition

**What do the instances represent?** Each instance is one photograph (JPEG) showing a mobility device or a
wheelchair-like object, with one row of labels in `annotations/labels.csv`.

**How many instances?** 464 images. 458 have a determinable device type and form the splits; 6 are labeled
`unclear` and are in `images/excluded/` (in no split).

| Class | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| manual | 200 | 43 | 43 | 286 |
| powered | 67 | 14 | 15 | 96 |
| other_aid | 43 | 9 | 9 | 61 |
| ambiguous | 11 | 2 | 2 | 15 |
| **total** | 321 | 68 | 69 | 458 |

**Is it a sample?** Yes, a screened sample of openly licensed web photographs, not a representative sample of
wheelchair users or of street scenes. On 22 October 2026 the team assembled 7,377 candidates: Open Images V7 (3,488),
the Openverse API (1,361) and Wikimedia Commons (2,528; never downloaded because the site rate-limited requests, so no
Commons image is included). Models pre-flagged 1,581 candidates as a possible child or stroller and 131 as showing no
mobility device; these were not queued for annotation. Annotators reviewed 1,050 images and excluded 604 (433 not a
photograph, 118 showing a child, 26 with a person identifiable beyond the face, 22 license concerns, 4 clinical
settings, 1 duplicate). 22 exclusions were reversed on adjudication (never for a child or an identifiable person);
4 of those, from one street scene with a possible child, are on a safety hold and not released:
1,050 - 604 + 18 = 464. No minimum image size or single-subject rule was applied.

**What data does each instance consist of?** A face-blurred JPEG (longer side at most 640 px, no EXIF/ICC/XMP
metadata) and labels: `device_type`, `occupied`, `propulsion`, `occlusion`, `viewpoint`, `lighting`, `setting`;
labeling provenance (`label_how`, pseudonymous `labelers`, `adjudicated_by`, `labeled_via`); license and attribution;
grouping keys (`uploader_group`, `dup_cluster`, `leak_group`); sha256. Device boxes are provided as preliminary COCO
JSON (one box per annotator per image, not reconciled). See README.md for the column dictionary.

**Labels.** `device_type` is the target. The other fields describe the scene for robustness analysis. Labels describe
only the device and the image, never a person.

**Missing information.** One image has an empty `setting` and one an empty `lighting` (unresolved). `not_applicable`
and `unclear` are explicit values.

**Relationships between instances.** Near-duplicates, photos from the same Flickr uploader or creator, the same source
page or photo, Flickr bursts (IDs within 3) and shared title stems are linked into 185 leak groups (34 with more than
one image). `leak_group` is the split unit; no group spans two splits.

**Recommended splits.** Use the provided `splits/*.txt` (70/15/15 by leak group, stratified by device type, seed
20261027). Re-splitting breaks the leak groups.

**Errors, noise, redundancy.** 132 images were labeled by two annotators, 332 by one. Cohen's kappa on doubly labeled
images: device_type 0.75 (86% raw agreement, n = 124), occupied 0.74, lighting 0.71, propulsion 0.69, setting 0.59,
viewpoint 0.47, occlusion 0.09 (near chance; treat as unreliable), exclusion decision 0.56 (n = 205). `propulsion`
has known inconsistencies (57 manual chairs marked `not_applicable`, 7 non-manual devices marked `self`/`attendant`).
44 near/exact duplicate pairs (15 clusters) are kept but confined to one split.

**Is it self-contained?** Yes. Images and metadata are in the package. Source URLs are given for attribution; 30
source photos have since been deleted at the source.

**Confidential data?** No. All images were already published under an open license or marked public domain.

**Offensive content?** None known. Report anything to the contact.

**Does it identify subpopulations or individuals?** No labels describe age, gender, race, disability or diagnosis.
Faces were blurred; see the privacy section. People are visible in most images (420 of the 458 devices are occupied)
but are not counted or identified.

**Sensitive data?** The images show people using mobility devices. Wheelchair use can be health-related, so the
dataset must not be used to infer anyone's disability, medical condition or identity.

## Collection process

**How was the data acquired?** Images were downloaded from Flickr through Open Images V7 (215 released) and the
Openverse API (249 released). Openverse queries: wheelchair, manual wheelchair, power wheelchair, motorized
wheelchair, electric wheelchair, wheelchair bus, mobility scooter, rollator, walking frame, stroller, pushchair,
shopping cart, office chair, luggage trolley, baggage cart. Queries and source labels only proposed a class to the
annotator and are never used as labels, features or split keys. Only candidates recorded as CC0, Public Domain Mark,
CC BY or CC BY-SA were kept.

**Who collected it, and when?** The authors, on 22 October 2026. Labeling ran 22-28 October 2026; the license
re-check ran 26-28 October 2026. Photographs themselves were taken earlier by their creators (dates not recorded).

**Ethical review.** No participants were recruited and the team photographed no one, so no consent was collected. No
IRB review is recorded for this release. An open license covers the photographer's copyright, not the
consent of the people shown; see the safeguards below.

**Were the individuals notified?** No. Anyone shown may request removal (contact above).

## Preprocessing, cleaning, labeling

* **Face blurring.** Faces were detected with OpenCV YuNet (2023 model) at three scales and on the mirrored image,
  with a lower threshold for small faces, and each face region (padded) was pixelated and Gaussian-blurred:
  4,962 regions in 445 of the 464 images. A second YuNet pass over the final release copies flagged 235 detections
  outside the blurred regions; each was inspected visually on contact sheets. 7 were visible faces the first pass had missed (mostly
  side profiles); those regions were blurred in 7 training images (`face_recheck.csv`), for 4,969 blurred regions in total. License plates, house numbers
  and signs were not blurred.
* **Resizing and metadata.** Longer side at most 640 px; JPEG re-encoded with no EXIF/ICC/XMP (so no GPS).
* **Labeling.** Two team members labeled in the team's own web app, which writes every action to an append-only event
  log; annotators did not see each other's labels and labeled only blurred copies. Labels were entered by hand or by
  confirming or correcting a SigLIP 2 zero-shot suggestion; for 108 of the 132 doubly labeled images at least one
  annotator took the suggestion, and the final label equals the suggestion for 73% of the 458 images (68% of test).
  15 disagreements were adjudicated by S. Ahmed, who is also an annotator.
* **Raw data.** The unblurred downloads are kept on a lab computer only and are not distributed.

## Uses

**Intended uses.** Training and evaluating classifiers that recognize the type of mobility device (T1); robustness
by occupancy, occlusion, viewpoint and lighting (T2); confusion of wheelchair-like objects with wheelchairs (T4).
Example applications: boarding configuration for accessible ride-hailing, paratransit and automated vehicles;
assistive robots and smart doors that adapt to the device present.

**Uses to avoid.** Inferring anyone's disability, diagnosis, age, gender, race, abilities or identity; face
recognition or re-identification; surveillance of individuals; any decision about a person based on the device they
use other than providing accommodation. Do not treat scores on these web photos as in-vehicle performance; only 12
images are at a curb and 15 at a vehicle, and 4 are low-light.

**Known biases.** All images are from Flickr. One U.S. government Flickr account supplies 197 of the 458 split images
(all in train). Classes are imbalanced (62% manual, 15 ambiguous). Powered chairs are mostly indoors (64 of 96).
Geography and device brands were not measured. Annotators could confirm SigLIP 2 suggestions, so SigLIP-based scores
are likely optimistic. The test set is small (69 images; 2 ambiguous), so confidence intervals are wide.

## Distribution

**How is it distributed?** As a zip archive on IEEE DataPort (DOI assigned by DataPort on upload). Images are not
hosted on GitHub.

**License.** Each image keeps its own license, listed in `LICENSES.csv` with creator, title, source, recorded license
and modifications: CC BY 2.0 (244), Public Domain Mark 1.0 (205), CC BY-SA 2.0 (9), CC0 1.0 (6). For 9 images that
Open Images recorded as CC BY 2.0 at collection, the live Flickr page now shows a different license or cannot be
checked; the team decided to include them under the license recorded at collection, and `LICENSES.csv` lists both.
This is a team decision, not a confirmation from the photographers. Annotations, documentation and code are
distributed under the terms IEEE DataPort applies to the upload (see README.md). The 9 CC BY-SA images remain under
CC BY-SA 2.0.

**Export controls or other restrictions?** None known.

## Maintenance

**Who maintains it?** Ross Greer (rossgreer@ucmerced.edu), Mi3 Lab, UC Merced.

**Updates.** Versioned releases. Removal requests and label corrections are recorded in CHANGELOG.md and applied in
the next version; the test split stays fixed across versions.

**Removal.** Anyone shown in an image, or the photographer, may ask for removal by email to the contact above.

**Contributions.** Corrections are welcome by email. External contributions are reviewed by the maintainers before a
new version.
