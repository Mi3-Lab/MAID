#!/usr/bin/env python3
"""Build the attribution / citation note for every candidate image. OFFLINE: reads only
data/candidates.csv, data/LICENSES_draft.csv, data/attribution_cache.json and annotations/events/*.jsonl.
(Refresh the cache by hand with scripts/fetch_attribution_meta.py, which uses the network.)

Writes:
  derived/ATTRIBUTIONS.md   human-readable note: TASL rules, special requirements, images in use
  derived/attributions.csv  one row per candidate (in_use flag grows as people label)
  derived/attributions.bib  BibTeX: per-source dataset entries + special-requirement and in-use images
Run by .github/workflows/rebuild-derived.yml after build_derived.py. Never hand-edit derived/."""
import argparse, csv, html, json, os, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "server"))
IGNORE_ANNOTATORS = {"deploytest"}          # test data, never counts as "in use"

# ---------------------------------------------------------------- helpers
def strip_html(s):
    s = re.sub(r"(?s)<style.*?</style>", " ", s or "")
    s = re.sub(r"\.mw-parser-output[^{]*\{[^}]*\}", " ", s)        # inline CSS that leaks into extmetadata
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()

def spdx_from_short(short):
    """'CC BY-SA 3.0 de' -> 'CC-BY-SA-3.0-DE'; 'CC0' -> 'CC0-1.0'."""
    s = (short or "").strip()
    if s.upper() in ("CC0", "CC0 1.0"): return "CC0-1.0"
    m = re.match(r"(?i)^CC[ -](BY(?:-NC)?(?:-SA|-ND)?)[ -]([\d.]+)(?:[ -]([a-z]{2}))?$", s)
    if not m: return None
    return "CC-" + m.group(1).upper() + "-" + m.group(2) + ("-" + m.group(3).upper() if m.group(3) else "")

def human_license(spdx):
    if spdx == "CC0-1.0": return "CC0 1.0"
    if spdx == "LicenseRef-PDM-1.0": return "Public Domain Mark 1.0"
    m = re.match(r"CC-(BY(?:-SA)?(?:-NC)?(?:-ND)?)-([\d.]+)(?:-([A-Z]{2}))?$", spdx or "")
    if m: return f"CC {m.group(1)} {m.group(2)}" + (f" {m.group(3)}" if m.group(3) else "")
    return spdx or "UNKNOWN"

def needs_attribution(spdx): return bool(spdx) and spdx.startswith("CC-BY")

# Permission text that is just a standard licence template / upload-bot notice (not a custom requirement)
BOILER = [r"^This file is licensed under the Creative Commons", r"^I, the copyright holder of this work",
          r"copyright holder of this work, hereby publish", r"^You may select the license",
          r"^At the time of upload, the image license was automatically confirmed", r"originally posted to Flickr",
          r"^This file, which was originally posted to", r"^granted by licen[cs]e$", r"^CC[ -]BY", r"^GFDL",
          r"^English \|?$", r"^Personality rights", r"^cc-by-sa/GFDL$", r"^Own work, copyleft",
          r"This work is free and may be used by anyone for any purpose"]
CUSTOM_KW = re.compile(r"(?i)attribut|credit|mention|must|please|contact|send me|photo:|nutno|uvést|my name|"
                       r"only|not public domain|copyright ©|©|wikimedia projects|terms of use|requirements")

def custom_permission(perm):
    p = strip_html(perm)
    if not p: return ""
    if any(re.search(b, p) for b in BOILER) and not re.search(r"(?i)attribute as|attribution must|must be credited|"
                                                               r"please (mention|credit|attribute)|photo:", p):
        return ""
    return p if CUSTOM_KW.search(p) else ""

def bib_escape(s):
    s = (s or "").replace("\\", r"\textbackslash{}")
    for ch in "&%$#_{}": s = s.replace(ch, "\\" + ch)
    return s.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")

def md_escape(s): return (s or "").replace("|", "\\|").replace("\n", " ")

# ---------------------------------------------------------------- in-use set
def in_use_images(evdir):
    import build_derived, wvcore
    evs, _ = build_derived.load_events(str(evdir))
    evs = [e for e in evs if e["annotator"] not in IGNORE_ANNOTATORS]
    per, adjv = wvcore.fold(evs)
    use = {}
    for img, anns in per.items():
        if adjv.get((img, "exclude"), ("", ""))[0]: continue            # adjudicated as excluded
        ok = sorted(a for a, fv in anns.items() if not fv.get("exclude"))
        if ok: use[img] = ok
    return use

# ---------------------------------------------------------------- per-image record
def load_blur():
    """data/blur_manifest.jsonl (scripts/auto_blur.py): only images listed there were actually face-blurred + EXIF-stripped."""
    p = ROOT / "data" / "blur_manifest.jsonl"; m = {}
    if p.exists():
        for line in open(p, encoding="utf-8"):
            if line.strip(): r = json.loads(line); m[r["image_id"]] = r
    return m

def modifications(planned, b):
    """Claim only what was done. `planned` is LICENSES_draft.csv's text (e.g. 'resized; faces blurred; EXIF stripped (planned)')."""
    if b is None:
        return ("resized" if "resized" in planned else ""), "pending"
    det = f"automatic, {b.get('detector', '?')} {b.get('version', '')}".strip()
    if b["n_faces"]: return f"resized; faces blurred ({b['n_faces']} detected, {det}); EXIF stripped", "done"
    return f"resized; EXIF stripped (automatic face-blur pass found no faces, {det})", "no_faces_found"

def build_record(c, lic, cache, blur=None):
    iid = c["candidate_id"]; src = c["source"]; spdx = c["license"]
    title = lic.get("title") or c["title"]; creator = lic.get("creator") or c["creator"]
    url = c["source_page_url"]; lurl = c["license_url"]
    flags, notes, usable = [], [], "yes"
    exact = ""                                       # author-specified credit line, if any
    ce = cache.get(iid) or {}
    if src == "commons":
        em = ce.get("extmetadata") or {}
        attr = strip_html(em.get("Attribution"))
        artist = strip_html(em.get("Artist")) or creator
        if re.match(r"(?i)^unknown author", artist or ""): artist = "Unknown author"
        credit_field = strip_html(em.get("Credit"))
        if not artist or artist == "Unknown author":
            if re.search(r"(?i)wellcome", credit_field): artist = "Wellcome Collection"     # Wellcome asks for this credit
            elif credit_field and not re.search(r"(?i)own work", credit_field):
                notes.append(f"no named author; Commons credit/source line: {credit_field[:200]}")
        if artist: creator = artist
        if attr:
            exact = attr; flags.append("author_specified_credit")
            notes.append(f'author asks to be credited as: "{attr}"')
        perm = custom_permission(em.get("Permission"))
        if perm:
            flags.append("custom_permission_text"); notes.append("uploader's terms: " + perm[:600] + ("…" if len(perm) > 600 else ""))
            if re.search(r"(?i)wikimedia projects|for use in wikipedia", perm) and not re.search(r"(?i)outside", perm):
                usable = "review"; notes.append("permission text reads as Wikimedia-only; confirm the free licence on the file page")
        restr = [r for r in (em.get("Restrictions") or "").split("|") if r]
        for r in restr:
            flags.append("restriction:" + r)
            notes.append({"personality": "personality rights: identifiable people may need consent for non-editorial use; keep faces blurred or exclude",
                          "trademarked": "trademark: depicts a trademark; licence covers the photo, not the mark (do not imply endorsement)",
                          "costume": "costume/character rights may apply to what is depicted",
                          "2257": "tagged as sexually explicit content (US 18 USC 2257): do not use"}.get(r, r))
            if r == "2257": usable = "no"
        live = spdx_from_short(strip_html(em.get("LicenseShortName")))
        if live and live != spdx:
            flags.append("license_mismatch"); usable = "review"
            notes.append(f"Commons now shows {live}, our CSV says {spdx}")
        if re.search(r"(?i)\bNC\b|-NC|\bND\b|-ND", strip_html(em.get("LicenseShortName")) + " " + strip_html(em.get("UsageTerms"))):
            flags.append("nc_or_nd"); usable = "no"; notes.append("NonCommercial/NoDerivatives licence: not allowed without Ross's approval")
        src_name = "Wikimedia Commons"
    elif src == "openverse":
        src_name = f"Openverse / {(ce.get('provider') or 'flickr').title()}"
    else:
        src_name = "Flickr, via Open Images Dataset V7"
        creator = ce.get("author") or creator
    if needs_attribution(spdx) and not exact and (not creator or re.match(r"(?i)^(unknown|anonymous|unknown author)\b", creator)):
        flags.append("missing_author"); usable = "review"
        notes.append("no author name recorded: credit as given on the source page, or drop the image")
    if re.search(r"-NC|-ND", spdx or ""): flags.append("nc_or_nd"); usable = "no"
    who = exact or creator or "unknown author"
    ttl = re.sub(r"^File:", "", title or "untitled")
    lic_h = human_license(spdx)
    lic_part = f"{lic_h} ({lurl})" if spdx in ("CC0-1.0", "LicenseRef-PDM-1.0") else f"licensed under {lic_h} ({lurl})"
    if exact:   # author-specified credit line is used verbatim
        credit = f'"{ttl}". Credit: {exact}. {src_name}, {lic_part}, {url}'
    else:
        credit = f'"{ttl}" by {who}, {src_name}, {lic_part}, {url}'
    if src == "openverse" and ce.get("attribution"):                      # Openverse's own TASL string
        credit = ce["attribution"].rstrip(".") + f". Source: {url}"
    mods, blur_status = modifications(lic.get("modifications") or "", (blur or {}).get(iid))
    if needs_attribution(spdx) and mods: credit += f". Modified: {mods}"
    if "-SA-" in (spdx or ""): notes.append("ShareAlike: the modified image must be redistributed under the same licence")
    return {"image_id": iid, "source": src, "title": ttl, "creator": creator, "source_url": url, "license": spdx,
            "license_url": lurl, "required_credit_text": credit, "author_specified_credit": exact,
            "special_requirement": "yes" if flags else "no", "special_flags": ";".join(dict.fromkeys(flags)),
            "special_note": " | ".join(notes), "usable": usable, "modifications": mods, "face_blur": blur_status,
            "date_accessed": lic.get("date_accessed") or c.get("access_date", ""), "in_use": "no", "in_use_by": ""}

def is_special(r):   # ShareAlike note alone is not "special"
    return r["special_requirement"] == "yes"

# ---------------------------------------------------------------- outputs
SOURCE_BIB = r"""@misc{wikimediacommons,
  author       = {{Wikimedia Commons contributors}},
  title        = {Wikimedia Commons: the free media repository},
  howpublished = {\url{https://commons.wikimedia.org/}},
  note         = {Individual images used under the licences listed in attributions.csv; accessed %(date)s},
  year         = {2026}
}

@misc{openverse,
  author       = {{Openverse}},
  title        = {Openverse: openly licensed media search (WordPress Foundation)},
  howpublished = {\url{https://openverse.org/}},
  note         = {Search index; images hosted by their providers (mostly Flickr) under the licences listed in attributions.csv; accessed %(date)s},
  year         = {2026}
}

@article{kuznetsova2020openimages,
  author  = {Kuznetsova, Alina and Rom, Hassan and Alldrin, Neil and Uijlings, Jasper and Krasin, Ivan and Pont-Tuset, Jordi and Kamali, Shahab and Popov, Stefan and Malloci, Matteo and Kolesnikov, Alexander and Duerig, Tom and Ferrari, Vittorio},
  title   = {The Open Images Dataset V4: Unified Image Classification, Object Detection, and Visual Relationship Detection at Scale},
  journal = {International Journal of Computer Vision},
  volume  = {128},
  number  = {7},
  pages   = {1956--1981},
  year    = {2020},
  doi     = {10.1007/s11263-020-01316-z}
}

@misc{openimagesv7,
  author       = {{Google LLC}},
  title        = {Open Images Dataset V7},
  howpublished = {\url{https://storage.googleapis.com/openimages/web/index.html}},
  note         = {Annotations licensed by Google LLC under CC BY 4.0; images are listed as having a CC BY 2.0 licence (Google does not own their copyright; see attributions.csv for each image's author and Flickr page); accessed %(date)s},
  year         = {2026}
}
"""

def write_bib(path, rows, date):
    out = ["% Auto-generated by scripts/build_attributions.py. Do not edit; rerun the script.",
           "% Cite the source entries for the dataset as a whole; per-image entries below are for images with special",
           "% attribution requirements and images currently in use. Needs \\usepackage{url} (or hyperref).", "",
           SOURCE_BIB % {"date": date}]
    for r in rows:
        if not (is_special(r) or r["in_use"] == "yes"): continue
        note = f"{human_license(r['license'])}, \\url{{{r['license_url']}}}"
        if r["author_specified_credit"]: note += f"; required credit: {bib_escape(r['author_specified_credit'])}"
        if r["modifications"] and needs_attribution(r["license"]): note += f"; modified: {bib_escape(r['modifications'])}"
        out.append("@misc{%s,\n  author       = {{%s}},\n  title        = {%s},\n  howpublished = {\\url{%s}},\n  note         = {%s; accessed %s},\n  year         = {n.d.}\n}\n" % (
            r["image_id"], bib_escape(r["author_specified_credit"] or r["creator"] or "Unknown author"), bib_escape(r["title"]),
            r["source_url"], note, r["date_accessed"]))
    path.write_text("\n".join(out), encoding="utf-8")

def write_md(path, rows, date):
    special = [r for r in rows if is_special(r)]
    use = [r for r in rows if r["in_use"] == "yes"]
    bad = [r for r in rows if r["usable"] != "yes"]
    lc = Counter(r["license"] for r in rows); lcu = Counter(r["license"] for r in use)
    L = ["# WheelView image attributions", "",
         "_Auto-generated by `scripts/build_attributions.py` on every push (CI). Do not edit by hand. "
         f"Metadata cache last refreshed: {date}._", "",
         "## How to credit (TASL)", "",
         "Every image in the release and every image shown in the paper gets a credit line with **T**itle, **A**uthor, "
         "**S**ource (link to the original page) and **L**icense (name and link). The `required_credit_text` column in "
         "[`attributions.csv`](attributions.csv) has a ready-made line for each image.", "",
         "- **CC BY / CC BY-SA (all versions):** credit is required. We resize, blur faces and strip EXIF, so the credit must "
         "also say the image was modified (CC 4.0 requires indicating changes; 2.x/3.0 require crediting adaptations). "
         "Face blurring is listed only for images that scripts/auto_blur.py actually processed (column `face_blur`: "
         "done / no_faces_found / pending; boxes in data/blur_manifest.jsonl); pending images must be blurred before release. "
         "The credit lines below already include `Modified: ...`.",
         "- **CC BY-SA:** the modified image must be redistributed under the same licence (ShareAlike). Keep each image's own licence in the release.",
         "- **CC0 / Public Domain Mark:** no credit is legally required; we credit anyway.",
         "- **Author-specified credit:** when the author asked for a specific credit line, use it exactly as written (see next section).",
         "- **Paper:** cite the sources with `\\cite{wikimediacommons, openverse, kuznetsova2020openimages, openimagesv7}` from "
         "[`attributions.bib`](attributions.bib). Any single image reproduced in a figure needs its own credit line in the caption (cite its `wv_` key).",
         "- **Open Images V7:** annotations are CC BY 4.0 (Google LLC); images are listed as CC BY 2.0 and credited to their Flickr authors.", "",
         f"Candidates: **{len(rows)}**. Images in use (labelled and not excluded by at least one annotator): **{len(use)}**. "
         f"Images with special requirements: **{len(special)}**. Rows needing review or not usable: **{len(bad)}**.", "",
         "## Rows that need review or cannot be used", "",
         "| image_id | usable | flags | note | source |", "|---|---|---|---|---|"]
    for r in bad:
        L.append(f"| {r['image_id']} | {r['usable']} | {r['special_flags']} | {md_escape(r['special_note'])} | {r['source_url']} |")
    L += ["", "## Images with special attribution requirements", "",
          "Use the **exact** credit text below. `restriction:*` flags come from the Commons file page and limit what may be shown, even with credit.", ""]
    for r in special:
        L.append(f"### {r['image_id']}{' (in use)' if r['in_use'] == 'yes' else ''} — {md_escape(r['title'])}")
        L.append(f"- Source: {r['source_url']}  ·  License: [{human_license(r['license'])}]({r['license_url']})  ·  Flags: `{r['special_flags']}`")
        if r["author_specified_credit"]: L.append(f"- Author-specified credit: **{md_escape(r['author_specified_credit'])}**")
        L.append(f"- Required credit: {md_escape(r['required_credit_text'])}")
        L.append(f"- Notes: {md_escape(r['special_note'])}")
        L.append("")
    L += ["## Images currently in use", "",
          f"{len(use)} images. This list grows automatically as people label.", "",
          "| image_id | title | author | license | source | special | credit line |", "|---|---|---|---|---|---|---|"]
    for r in use:
        L.append(f"| {r['image_id']} | {md_escape(r['title'][:70])} | {md_escape(r['author_specified_credit'] or r['creator'])} | "
                 f"[{human_license(r['license'])}]({r['license_url']}) | [link]({r['source_url']}) | {r['special_requirement']} | {md_escape(r['required_credit_text'])} |")
    L += ["", "## All candidates", "",
          f"All {len(rows)} candidates, with credit lines, are in [`attributions.csv`](attributions.csv) (`in_use` column = yes/no).", "",
          "| license | all candidates | in use |", "|---|---|---|"]
    for k, n in lc.most_common(): L.append(f"| {k} | {n} | {lcu.get(k, 0)} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=ROOT / "derived"); ap.add_argument("--events", default=ROOT / "annotations" / "events")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import wvcore
    blocked = wvcore.load_blocklist(os.environ.get("WV_BLOCKLIST_DIR", ROOT / "data"))      # data/removed_*.txt: removed entirely, never credited/released
    cands = [c for c in csv.DictReader(open(ROOT / "data" / "candidates.csv", encoding="utf-8")) if c["candidate_id"] not in blocked]
    lics = {r["image_id"]: r for r in csv.DictReader(open(ROOT / "data" / "LICENSES_draft.csv", encoding="utf-8"))}
    cp = ROOT / "data" / "attribution_cache.json"
    cache = json.load(open(cp, encoding="utf-8"))["items"] if cp.exists() else {}
    if not cache: print("WARN: no attribution cache; Commons-specific requirements cannot be checked", file=sys.stderr)
    use = in_use_images(a.events); blur = load_blur()
    rows = []
    for c in cands:
        r = build_record(c, lics.get(c["candidate_id"], {}), cache, blur)
        if c["candidate_id"] in use: r["in_use"] = "yes"; r["in_use_by"] = ";".join(use[c["candidate_id"]])
        rows.append(r)
    dates = [v.get("fetched") for v in cache.values() if v.get("source") == "commons" and v.get("fetched")]
    date = max(dates) if dates else "never"
    with open(out / "attributions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write_md(out / "ATTRIBUTIONS.md", rows, date); write_bib(out / "attributions.bib", rows, date)
    print(json.dumps({"candidates": len(rows), "in_use": sum(r["in_use"] == "yes" for r in rows),
                      "special": sum(is_special(r) for r in rows), "face_blur": dict(Counter(r["face_blur"] for r in rows)),
                      "not_usable_or_review": dict(Counter(r["usable"] for r in rows if r["usable"] != "yes")),
                      "flags": dict(Counter(f.split(":")[0] if not f.startswith("restriction") else f for r in rows for f in r["special_flags"].split(";") if f))}))
