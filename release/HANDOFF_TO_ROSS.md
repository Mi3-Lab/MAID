# WheelView v1.0: handoff to Ross Greer

Team-internal file. It is kept in the repo and is **not** shipped in the dataset folder or zip.
Prepared on Thu 2026-10-29 (PDT) for Samir Ahmed's handoff. Nothing has been uploaded, submitted or emailed.

**Competition:** IEEE AgeTech Global Dataset Competition 2026, Theme 3 (Mobility & Assistive Technologies).
**Deadline:** Oct 30, 2026, GMT+0, with no time of day stated. **Plan for Thu Oct 29, 5:00 PM PDT.**

## 1. What is where

| Item | Location |
|---|---|
| Final report (submit this) | `report/final/wheelview_report_final.pdf` (box). 14 PDF pages: body is pages 1–10 including the title page, references 11–12, Appendices A–C 13–14. Source: `wheelview_report_final.tex` and `references.bib` in the same folder. |
| Report .docx (convenience only) | `report/final/wheelview_report_final.docx`. A pandoc conversion with rough tables and figures; the PDF is the version of record. |
| Release zip (DataPort "Dataset Files") | Lab laptop only: `C:\Users\kamoo\wheelview\var\WheelView_v1.0_release.zip`. Size 43,477,473 bytes (43.5 MB), 492 entries. sha256 `11a102c05b90bd8f4fc9c5e39079018da480e9d796c8be9db98f95865f4bfc3e`, also in the `.sha256` file next to it. |
| Unzipped release | Lab laptop: `C:\Users\kamoo\wheelview\var\release\WheelView_v1.0\` |
| Release metadata and code (no images) | GitHub repo, `main`, folder `release/`. Box copy: `report/final/release/` |
| This memo, plain-language PDF version | `report/final/WheelView_handoff_memo.pdf` |
| Submission checklist | `release/SUBMISSION_CHECKLIST.md` |
| Competition rules with quotes and sources | `report/final/AGETECH_RULES.md`. Saved source pages are in `report/rules_src/` on the box. |

Zip layout: `WheelView_v1.0/` contains:
- `images/{train,val,test,excluded}/wv_XXXXXX.jpg` (464 JPEGs, longest side 640 px or less, metadata stripped, faces blurred)
- `annotations/labels.csv`, `coco_{train,val,test}.json`, `boxes_per_annotator.csv`
- `splits/{train,val,test}.txt`
- `LICENSES.csv`, `README.md`, `DATASHEET.md`, `CHANGELOG.md`, `SHA256SUMS.txt`, `face_recheck.csv`
- `code/`: loader, `evaluate.py`, `scripts/` (train, predict, leakage check, latency), `server/schema.json`, `wvcore.py`, and pinned requirements

The images are **not** on GitHub and must not go to GitHub, Overleaf or any shared drive outside the lab.
Move the zip only by lab-internal means, or upload it straight from the laptop to DataPort.

## 2. Headline numbers (all from v3 data; nothing from v1/v2 is in the report)

- **Images:** 464 released. 458 are in splits (train 321 / val 68 / test 69), plus 6 `excluded` (`unclear`).
  - Classes in train/val/test: manual 200/43/43, powered 67/14/15, other_aid 43/9/9, ambiguous 11/2/2.
  - 185 leak groups (34 multi-image), each kept within one split.
  - 4 images from one scene are held back because a child may be present.
- **Licenses:** CC BY 2.0 244, Public Domain Mark 205, CC BY-SA 2.0 9, CC0 6.
- **Face blur:** 4,969 regions in total. 4,962 came from the first YuNet pass on 445 images; 7 were added by the final re-check.
- **Agreement (Cohen's kappa):**

  | Attribute | kappa |
  |---|---|
  | device_type (n=124) | 0.75 |
  | occupied | 0.74 |
  | lighting | 0.71 |
  | propulsion | 0.69 |
  | setting | 0.59 |
  | exclusion (n=205) | 0.56 |
  | viewpoint | 0.47 |
  | occlusion | 0.09 |

  108 of the 132 relevant images had at least one annotator confirm the model suggestion.
- **T1 test results (n=69):**
  - SigLIP 2 fine-tune: accuracy 0.903±.017, macro-F1 0.841±.073 (3 seeds).
  - ResNet-50: 0.657 / 0.498. YOLO11s-cls: 0.570 / 0.362. Linear probe: 0.754 / 0.721.
  - Zero-shot: SigLIP 2 0.681 / 0.723; YOLO-World 0.449 / 0.364. Always-manual: 0.623 / 0.192.
  - SigLIP-based rows are likely optimistic, because 68% of test labels equal the SigLIP suggestion. The report says so.
- **T2 robustness, SigLIP 2 FT accuracy:**
  - occupied .923, unoccupied .750
  - occlusion none .867, partial 1.000, heavy 1.000 (small n)
  - rear .879, other lighting .867
- **T4 (2 ambiguous test images):** Amb-FPR, the share of ambiguous objects predicted as a device, is 0.33±0.29 for SigLIP FT, 0.5 for the probe, 0.83 for ResNet-50 and 1.0 for YOLO11s-cls. False-ambiguous rate is 0.005 or less.
- **Latency:** batch 1, fp32, median, image in memory, preprocessing plus forward pass.

  | Model | RTX 3000 Ada laptop GPU | Core Ultra 7 165H CPU |
  |---|---|---|
  | SigLIP 2 FT | 43.6 ms | 564 ms |
  | ResNet-50 | 11.1 ms | 38.7 ms |
  | YOLO11s-cls | 11.9 ms | 14.8 ms |

  These are laptop numbers, not in-vehicle hardware.
- **Leakage check:**
  - Full check (`--features --files --runtime --release`): 0 failures, 2 informational warnings.
  - Standalone check inside the release: 0 failures, 0 warnings.

## 3. Decisions for Ross, with recommendations

1. **DataPort access tier.** Options are Standard (free; downloading needs a DataPort subscription) or Open Access ($1,950). The rules don't say which tier competition entries need (unverified).
   *Recommendation:* upload as **Standard** and link it to the competition, unless the chair says judges need Open Access. Ask in the chair email (§5).
2. **The 9 CC BY-SA 2.0 images vs DataPort's blanket CC-BY.** They are included, and `LICENSES.csv`, the README and report §6.3 say their adaptations stay CC BY-SA.
   *Recommendation:* keep them, with a sentence in the DataPort abstract or instructions that per-image licenses in `LICENSES.csv` take precedence for images.
   If DataPort or the chair objects, drop them in a v1.0.1. That needs a rebuild, and the numbers would change slightly, so it is not something to do tonight.
3. **Annotation license.** DataPort applies CC-BY to uploads. The report says annotations, documentation and code are released "under the terms IEEE DataPort applies to uploads (Creative Commons Attribution)".
   The no-ability-inference rule is worded as a use restriction in the README and report; it is not enforceable under CC-BY.
   *Recommendation:* accept CC-BY (CC BY 4.0 if DataPort asks for a version).
4. **IRB / HIPAA.** The report says no health information, identities or location metadata are collected, no participants were recruited, and "No IRB determination is recorded for this release."
   *Recommendation:* leave as is for submission. If UC Merced's IRB office gives a "not human subjects research" determination later, add it in v1.0.1.
5. **passcal.** All mention of passcal2026 has been removed from the report.
   *Recommendation:* leave it out. It is not needed, and the page budget is full.
6. **The 9 Open Images photos whose Flickr licenses changed after collection** (3 now non-commercial, 3 all rights reserved, 3 behind a login and not re-checkable). Samir decided on 2026-10-28 to include them under the CC BY 2.0 recorded at collection. The report states this is a team decision, not photographer confirmation, and `LICENSES.csv` flags each one.
   *Recommendation:* keep them, and remove any image immediately if the photographer asks. The conservative alternative is to drop them in v1.0.1.
7. **Template and "brief introduction document".** No official REPORT template could be found ([D] mentions one), and it is unclear whether [D]'s "brief introduction document" is a separate file.
   *Recommendation:* email the chair (draft in §5). If there is no reply in time, submit the PDF as is; the title page and abstract serve as the introduction.
8. **Eligibility.** The FAQ allows existing datasets only if they were "not previously submitted to any competition". WheelView was built during the CivicProgress Challenge.
   *Action:* Ross or Samir must confirm the dataset itself was not submitted there or anywhere else. If it was, ask the chair before submitting.
9. **Deadline time.** The deadline is given as GMT+0 with no hour.
   *Recommendation:* treat it as **Thu Oct 29, 5:00 PM PDT**.
10. **The 15 device-type adjudications** were made by S. Ahmed, who is also an annotator. The report says so.
    *Recommendation:* optional independent re-adjudication by Ross for v1.0.1. It is not needed for submission.
11. **Propulsion label inconsistency.** 57 manual chairs are marked `not_applicable` (9 of them are unoccupied), and 7 non-manual devices are marked `self` or `attendant`. This is documented in Appendix C and CHANGELOG, and there is no T3 baseline. Human labels were not edited.
    *Recommendation:* fix it in v1.0.1 with a documented relabel pass.
12. **Literal Times New Roman.** The PDF uses a Times clone (TeX Gyre Termes, same metrics) because the build box has no TNR.
    *Optional:* recompile with `\usetnrtrue` under XeLaTeX on a machine that has TNR.

## 4. Wording log (conservative choices made where only Samir or Ross could decide)

| Topic | Wording now in the report | Why |
|---|---|---|
| DOI | "assigned by IEEE DataPort on upload" | No DOI exists yet. |
| Ageing statistic | UN: 761 M people aged 65+ in 2021, 1.6 B by 2050. Kaye et al. 2000: 55.6% of wheelchair/scooter users are 65+. | Checked against the sources. |
| SAE/RESNA | Limited to the tiedown and occupant-restraint statement | Broader claims could not be verified. |
| passcal | Removed (paragraph and bullet) | Samir's instruction; no backing. |
| Image filters | "No minimum image size or single-subject rule was applied." | That is what happened. |
| Boxes | "preliminary" COCO boxes plus a per-annotator CSV; some boxes were accepted from detector suggestions | True provenance. |
| Face re-check | 235 detections outside blur "inspected visually"; 7 missed faces were blurred; the baselines were trained on the pre-fix copies; residual risk stated | The visual inspection was done on contact sheets by the assistant, so "a person reviewed" was not claimed. |
| Plates and signs | "License plates, house numbers, and signs are not blurred." | True. |
| Geography and brands | "not recorded or measured" | Never measured. |
| Government account | 197 images "screened under the same rules as every other image" | True. |
| Licensing §6.3 | Annotations etc. under DataPort's CC-BY terms; images keep their own licenses; CC BY-SA adaptations stay CC BY-SA | Decisions 2–3 are still open. |
| Changed licenses | Described as a "team decision (S. Ahmed)" on 28 Oct, not photographer confirmation | Samir's decision. |
| IRB | "No IRB determination is recorded for this release." | No determination exists in our records. |
| DATASHEET | "The authors" (was "student team"); "No IRB review is recorded" (was "no ethics review was carried out") | Accurate and neutral. |
| Hard-case rules (App. B) | "No separate written rules were issued for hard cases"; the workflow is quoted from the app's help text | That is all that existed. |
| Tasks | T2 includes lighting; T3 defined with no baseline (noisy propulsion); T4 is Amb-FPR | Matches what was run. |
| Latency | Laptop figures, explicitly "not in-vehicle hardware" | Only laptop numbers exist. |
| Replication | "Trained weights are not included." | True. |
| Title, authors, acknowledgments, contact, structure, font, spacing | Unchanged | Samir's instruction. |

## 5. Draft email to the competition chair (NOT sent; Ross to send if wanted)

To: polatgoktas@ieee.org
Subject: AgeTech Dataset Competition 2026: questions before our submission (WheelView)

> Dear Dr. Göktaş,
>
> We are about to submit WheelView, an image dataset of wheelchairs and mobility aids for Theme 3, and have a few quick questions:
> 1. Is there an official REPORT template? The DataPort page mentions one, but we could not find a link. Is a PDF that follows the handbook's recommended structure acceptable?
> 2. Should competition datasets be uploaded as a Standard or an Open Access dataset?
> 3. Is the "brief introduction document" a separate file, or does the report's title page and abstract suffice?
> 4. What time on October 30 (GMT) is the deadline?
> 5. Nine of our images are CC BY-SA 2.0. Is it acceptable to state in the documentation that each image keeps its own license (listed per file) while our annotations are CC-BY?
>
> Thank you,
> Ross Greer (on behalf of the WheelView team)

## 6. DataPort upload steps (for Ross)

1. Sign in to IEEE DataPort and choose **Submit a Dataset**, then link it to the competition, "IEEE AgeTech Global Dataset Competition" (competition DOI 10.21227/tga3-sw58).
2. Fill in the required fields:
   - **Title and Authors:** as on the report title page.
   - **Category:** the competition's category, or Computer Vision.
   - **Abstract:** report §1. Add: "Image licenses are per file in LICENSES.csv."
   - **Instructions:** paste or attach `README.md`.
   - **Documentation:** the report PDF, plus optionally `DATASHEET.md`.
   - **Dataset Files:** `WheelView_v1.0_release.zip`, uploaded from the laptop.
3. Choose the access tier (decision 1).
4. After the DOI appears, it can optionally go on the title page:
   - In `wheelview_report_final.tex`, replace "assigned by IEEE DataPort on upload" with the DOI.
   - Recompile with `tectonic -k wheelview_report_final.tex`. The page count is unaffected.
5. Submit the report through the DataPort competition page as instructed there [H: "All submissions must be completed through IEEE DataPort"].
6. Download the zip back from DataPort and check its sha256 against the value in §1.

## 7. Known limitations (all stated in the report or README)

- Small test set (69). The ambiguous class has 2 test images, so T4 numbers are very noisy.
- Web photos only: no in-vehicle imagery and no night imagery. Geography and brands were not measured.
- 197 images come from one U.S. government Flickr account.
- SigLIP-based scores are likely optimistic, because labels were partly confirmed from SigLIP suggestions.
- Occlusion has low agreement (kappa 0.09), and propulsion labels are inconsistent. Boxes are preliminary.
- Plates and signs are not blurred. Automated face blur can miss faces.
- Baselines were trained on the copies from before the 7-image face fix.
- No trained weights are shipped.
- The `evaluate.py` bootstrap can differ in the third decimal from the training-time CI (different resampling); point estimates match.
