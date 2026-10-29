# WheelView v1.0: IEEE AgeTech Global Dataset Competition 2026 submission checklist

Team-internal file. It is kept in the repo and is **not** shipped in the dataset folder or zip.
Prepared 2026-10-29 (PDT). Nothing has been uploaded or submitted.

Rule sources: [H] Handbook PDF (2026-06-08), [W] agetech.ieee.org/competition, [D] DataPort competition page,
[S] DataPort "Submit a Dataset" page and FAQ, [F] DataPort licensing FAQ. All were read on 2026-10-28.
The quotes are in `AGETECH_RULES.md` (the box copy is in the report folder).

Status key: **MET** means the evidence is in the package. **NEEDS ROSS** means only Ross or Samir can do or decide it.
**MET (assumed)** means we met the rule under a conservative reading that Ross should confirm.

## 1. Deadline and team

| # | Rule | Status | Evidence / action |
|---|---|---|---|
| 1.1 | Submission deadline: Oct 30, 2026, "Key Dates (GMT+0)" [D, W]. No time of day is given. | **NEEDS ROSS** | Plan for **Thu Oct 29, 5:00 PM PDT** (start of Oct 30 GMT). The latest possible reading is Fri Oct 30, 4:59 PM PDT. Ask the chair if unsure. |
| 1.2 | Max 4 members per team; one submission per team; one team per person [H] | MET (assumed) | The report lists 4 authors (Ahmed, Saleem, Maia, Greer). Ross should confirm that none of them is on another AgeTech team. |
| 1.3 | Existing dataset allowed only if: you own the rights to share it; it complies with DataPort policies; it "has not previously been submitted to any competition" [H FAQ] | **NEEDS ROSS** | WheelView was built during the CivicProgress Challenge, which is named in the acknowledgments. Ross or Samir must confirm that the *dataset* was not submitted to that challenge or to any other competition. |

## 2. Report format

| # | Rule | Status | Evidence / action |
|---|---|---|---|
| 2.1 | Word (.docx) or PDF [H] | MET | `wheelview_report_final_r3.pdf`. A pandoc .docx is also provided as a convenience copy, but it is rough; the PDF is the version to submit. |
| 2.2 | Max 10 pages excluding references and appendices [H, W] | MET | The body is PDF pages 1–10, *counting the title page* (printed pages 1–9 follow it). References start on PDF page 11, and Appendices A–C are on pages 13–14. The PDF has 14 pages in total. |
| 2.3 | Single column [H] | MET | `article` class, one column. |
| 2.4 | Times New Roman 11 pt [H] | MET (assumed) | The PDF uses 11 pt Times via `newtxtext` (TeX Gyre Termes, a Times clone metrically compatible with Times New Roman). Literal TNR is available with `\usetnrtrue` under XeLaTeX on a machine that has the font installed (the box has no TNR). |
| 2.5 | 1.15 line spacing [H] | MET | `\setstretch{1.15}`. |
| 2.6 | "Use the official REPORT template provided through the competition website" [D] | **NEEDS ROSS** | No template is linked on [W], [D] or in [H] (checked 2026-10-28). We followed the handbook's "Recommended report structure". Ask the chair whether a template exists. |
| 2.7 | Title page: Dataset Title, Authors, Affiliations, Contact Information [H] | MET | Report title page; unchanged from Samir's draft. |
| 2.8 | Structure: Abstract; Motivation; Data Collection; Dataset Characteristics; Data Quality; Ethical Considerations; Potential Applications; Accessibility and Reproducibility; References; Appendix (optional) [H] | MET | Sections 1–8 follow this order, then References, then Appendix A (data dictionary), Appendix B (annotation guide), and Appendix C (label notes). |
| 2.9 | Data Collection covers sources, devices, sensors, recruitment, methodology [H] | MET | §3. It states that no sensors were used and no participants were recruited. |
| 2.10 | Characteristics covers samples, participants, features, labels, file formats [H] | MET | §4 and Tables 1–2; release layout; Appendix A. |
| 2.11 | Data Quality covers missing data, validation, QA [H] | MET | §5: kappa per attribute, adjudication, blur QA, face re-check, leakage checks. |
| 2.12 | Ethics covers privacy, consent, anonymization, regulatory compliance [H]; [D] requires compliance with "privacy, consent, and data governance requirements" | MET, with one open item | §6: face blur, the 7-image re-check fix, licenses, and the no-ability-inference statement. §6.4 says "No IRB determination is recorded for this release." Any IRB/HIPAA determination is **NEEDS ROSS** (see 5.3). |
| 2.13 | "A brief introduction document" [D] | **NEEDS ROSS** | It is unclear whether this means a separate file. For now the title page and abstract serve as the introduction. Ask the chair. |
| 2.14 | Recommended documentation (overview, sampling, data dictionary, label definitions, QC, ethics statement, limitations, intended use, reproducibility) [H] | MET | Report §§3–8 and Appendix A; `README.md`; `DATASHEET.md`. |
| 2.15 | No TODOs left in the PDF (our own rule) | MET | 0 TODO markers; `\showtodosfalse`. Text search of the PDF for "TODO" and "passcal" finds nothing. |
| 2.16 | No claims that cannot be backed up; no results from old data (Samir's rules) | MET | All numbers come from v3 data (`results/*_v3.json`, `robustness_v3.json`, `latency_v3.json`). Every wording decision is logged in `HANDOFF_TO_ROSS.md` §4. |

## 3. Optional content scored by the judges

| # | Rule | Status | Evidence |
|---|---|---|---|
| 3.1 | Optional baselines, benchmarks, use cases [H, D] | MET | Table 3 (T1, 7 models), Table 4 (T2 robustness by condition plus T4 Amb-FPR), latency paragraph. |
| 3.2 | Code is optional [H] | MET | `code/` in the release: loader, `evaluate.py`, training and prediction scripts, leakage checker, pinned requirements. Trained weights are not included (the report says so). |
| 3.3 | Evaluation weights: relevance 20, quality/diversity 25, documentation 15, ethics 15, accessibility 10, AI readiness 15 [H] | Covered | §2/§7; §4/§5; §3/§4/§8/App. A; §6; §4/§8; §7. |

## 4. IEEE DataPort hosting

| # | Rule | Status | Evidence / action |
|---|---|---|---|
| 4.1 | "All submissions must be completed through IEEE DataPort"; upload the dataset there [H] | **NEEDS ROSS** | The upload has not been done. Steps are in `HANDOFF_TO_ROSS.md` §6. |
| 4.2 | Required fields: Dataset Authors, Title, Category, Abstract, Documentation, Instructions, Dataset Files [S] | Ready, **NEEDS ROSS** to enter | Authors and title are as in the report. Category: suggest the competition's category or "Computer Vision". Abstract: report §1. Documentation: report PDF (plus `DATASHEET.md`). Instructions: `README.md`. Files: the zip. |
| 4.3 | More than 100 files → ZIP/GZIP [S] | MET | `WheelView_v1.0_release.zip` (492 entries, 43,477,473 bytes, sha256 `11a102c05b90bd8f4fc9c5e39079018da480e9d796c8be9db98f95865f4bfc3e`). |
| 4.4 | Package contents: dataset files, metadata, data dictionary, labels, collection protocols, ethics information, licensing information [H] | MET | images/; `labels.csv`; `README.md` (data dictionary and layout); COCO plus per-annotator boxes; `DATASHEET.md` (collection, ethics); `LICENSES.csv`; `CHANGELOG.md`; `SHA256SUMS.txt`; `face_recheck.csv`. |
| 4.5 | Accepted formats (CSV, JSON, image archives …) [H] | MET | CSV, JSON, JPEG in a ZIP, plus Markdown and Python. |
| 4.6 | DOI is assigned automatically [S] | **NEEDS ROSS** | The report says "assigned by IEEE DataPort on upload". After upload, Ross may put the actual DOI on the title page (one line, then recompile; see handoff §6). |
| 4.7 | Access tier: Standard (free; files need a DataPort subscription) or Open Access ($1,950) [S]. Which tier competition entries use is not stated. | **NEEDS ROSS** | Recommendation is in the handoff (§3, decision 1). |

## 5. Licensing, privacy, ethics

| # | Rule | Status | Evidence / action |
|---|---|---|---|
| 5.1 | Uploading makes the dataset available under CC-BY [F]; you must own the rights to share it [H] | MET, with decisions open | Per-image licenses are in `LICENSES.csv`: 244 CC-BY-2.0, 205 Public Domain Mark, 9 CC-BY-SA-2.0, 6 CC0. Our annotations go under DataPort's CC-BY terms. Images keep their own licenses (report §6.3, README). |
| 5.2 | Share-alike conflict for the 9 CC BY-SA images | **NEEDS ROSS** | They are included and marked CC BY-SA in `LICENSES.csv` and the README (report §6.3). Decision 2 in the handoff. |
| 5.3 | Privacy and consent compliance [D, H] | MET (blur, no recruitment); IRB/HIPAA is **NEEDS ROSS** | Faces are blurred with YuNet, with a second detector pass and a visual check of the 235 remaining detections; 7 more faces were blurred (`face_recheck.csv`). Plates and signs are **not** blurred, and the report says so. No sensitive attributes are labeled. Decision 4 in the handoff. |
| 5.4 | Licenses that changed at the source (9 Open Images photos: 3 now non-commercial, 3 all-rights-reserved, 3 behind a login) | MET: excluded 2026-10-29 | Listed in `data/license_exclusions.csv`; not in the release; report §6.3. Decision 6 resolved. |
| 5.5 | Images not posted to GitHub, Overleaf or shared drives outside the lab (Samir's rule) | MET | GitHub holds only metadata and code (`release/`). Images exist only on the lab laptop and in the zip. |

## 6. Leakage and integrity (our rules)

| # | Check | Status | Evidence |
|---|---|---|---|
| 6.1 | `check_leakage.py --features --files --runtime --release var/release/WheelView_v1.0` | MET | 0 failures, 2 informational warnings (image_id order correlates with class; old `var/yolo_cls_v2` test folders that the current scripts do not use). Laptop log: `var\leak_release_out2.txt`. |
| 6.2 | Standalone `code/scripts/check_leakage.py --release .` run inside the built folder | MET | 0 failures, 0 warnings (`var\leak_standalone_out.txt`). |
| 6.3 | Face re-check rerun after the fix | MET | Detections outside blur dropped from 235 to 228; the 7 fixed faces are no longer detected (`var\face_check_out2.txt`). |
| 6.4 | Zip integrity | MET | `testzip` OK. Get-FileHash equals the `.sha256` file. 455 images (319/66/64/6 excluded); the internal docs are not inside. |
| 6.5 | Loader and scorer smoke tests on the built folder | MET | `wheelview_loader.py --summary` gives the expected split counts. `evaluate.py` on the SigLIP 2 FT seed-0 predictions gives test accuracy 0.913 and macro-F1 0.834, consistent with Table 3. |

## 7. Administrative dates after submission [D] (tentative)
Administrative screening: Nov 2026. Technical evaluation: Nov–Dec 2026. Demo session (if required): Nov 2026. Winners: Dec 2026.
