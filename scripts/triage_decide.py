"""Triage of the v1 candidate pool (local only; images never leave this machine).
Inputs: var/triage/features.csv (scripts/triage_features.py), var/wheelview.db (read-only) or annotations/events,
annotations/auto/*.jsonl, data/candidates.csv, data/license_audit_v1.csv.
Policy (samir, 2026-10-27): "Do not interfere with anything unless I had already flagged it."
  * LABELED images (any human label/adjudication event from samir or shaaz) are never blocked or dropped by a model rule.
    They are dropped / stay blocked only on a human exclude (typed, or a model flag the labeler accepted with a keypress).
    Model concerns go into the model_warning column; license problems into license_warning (decision keep_flagged_license).
  * UNLABELED images keep the model-based removals (child / stroller / no device), written to data/removed_model_*.txt.
Outputs: data/removed_model_child_stroller.txt, data/removed_model_no_device.txt, data/removed_human_exclude.txt,
data/no_device_uncertain.txt, data/v1_triage.csv,
var/triage/summary.json (+ var/triage/owl_needed.txt for the optional OWLv2 confirm pass)."""
import csv, json, os, sqlite3, sys, collections, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "server")); import wvcore
OUT = ROOT / "var" / "triage"; DATA = ROOT / "data"
LABELERS = {"samir", "shaaz"}
TH = dict(nopeople=0.60,   # weak auto child flags (0.1-0.3) are ignored only when SigLIP2 is >=60% sure there is no person
          child=0.40, child_sig=0.02, child_auto=0.10,   # calibrated: 109/114 human child excludes caught by the scores alone   # child: LOW thresholds, any one fires -> remove (err toward dropping)
          stroller=0.25,                               # stroller share of the device/scene softmax, max(blurred, original)
          dev_absent=0.10, dev_unsure=0.35,            # device presence = sum of device-class probs; < absent -> remove, < unsure -> uncertain
          owl=0.15,                                     # any OWLv2 device box >= this blocks a no-device removal
          photo_keep=0.80, photo_keep_human=0.90, photo_drop=0.30, recover_dev=0.60, small_px=64, model_conf=0.90)
DEV = ["wheelchair", "power", "scooter", "walker", "cane", "cart", "office", "luggage", "stroller"]
def fnum(r, k): return float(r[k])
def load_events():
    db = ROOT / "var" / "wheelview.db"
    if db.exists():
        c = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True); c.row_factory = sqlite3.Row; evs = []
        for r in c.execute("SELECT * FROM events ORDER BY seq"):
            e = {k: r[k] for k in ("uuid", "image_id", "annotator", "field", "ts", "device_id", "kind")}
            e["value"] = json.loads(r["value"]) if r["value"] is not None else None
            if r["prov"]: e.update(json.loads(r["prov"]))
            evs.append(e)
        return evs, "db"
    sys.path.insert(0, str(ROOT / "scripts")); import build_derived
    return build_derived.load_events(str(ROOT / "annotations" / "events"), drop_blocked=False)[0], "jsonl"
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--calib", action="store_true"); a = ap.parse_args()
    F = {r["image_id"]: r for r in csv.DictReader(open(OUT / "features.csv", encoding="utf-8"))}
    cands = {r["candidate_id"]: r for r in csv.DictReader(open(DATA / "candidates.csv", encoding="utf-8"))}
    auto = {}
    for l in open(ROOT / "annotations" / "auto" / "siglip2-so400m-224_zs-v1.jsonl", encoding="utf-8"):
        r = json.loads(l); auto[r["image_id"]] = r
    audit = {r["image_id"]: r for r in csv.DictReader(open(DATA / "license_audit_v1.csv", encoding="utf-8"))}
    owl_local = {}
    if (OUT / "owl.jsonl").exists():
        for l in open(OUT / "owl.jsonl", encoding="utf-8"): r = json.loads(l); owl_local[r["image_id"]] = r
    evs, evsrc = load_events(); evs = [e for e in evs if e["annotator"] in LABELERS and e["kind"] in ("label", "adjudication")]
    per, adjv = wvcore.fold(evs)
    done = {i for i, anns in per.items() if any(fv.get("_status") == "done" for fv in anns.values())}
    labeled = {e["image_id"] for e in evs} | set(audit)     # any human label/adjudication event
    st = {i: wvcore.image_status(i, per, adjv) for i in labeled}
    # exclude provenance: latest exclude event per (image, annotator)
    lastex = {}
    for e in evs:
        if e["field"] == "exclude":
            k = (e["image_id"], e["annotator"], e["kind"])
            if k not in lastex or (e["ts"], e["uuid"]) > (lastex[k]["ts"], lastex[k]["uuid"]): lastex[k] = e
    def ex_info(i):
        """-> (set of exclude reasons in force, how, source human|auto_review|none)"""
        s = st[i]; v, how, vals = s["fields"].get("exclude", ("", "missing", {}))
        if how == "adjudicated": return ({v} if v and v != "none" else set()), how, ("human" if v and v != "none" else "none")
        rs = {x for x in vals.values() if x and x != "none"} if how == "conflict" else ({v} if v and v != "none" else set())
        if not rs: return set(), how, "none"
        srcs = [lastex.get((i, an, "label"), {}).get("source") for an, x in vals.items() if x in rs]
        return rs, how, ("human" if any(sv != "auto_review" for sv in srcs) else "auto_review")
    def human_dev(i):
        if i not in st: return None
        v, how, vals = st[i]["fields"].get("device_type", ("", "missing", {}))
        if how == "conflict": return "unclear"
        return v or None
    # ---------------------------------------------------------------- per-candidate scores
    S = {}
    for i, r in F.items():
        g = lambda k: max(fnum(r, k + "_b"), fnum(r, k + "_o"))
        dev_b = sum(fnum(r, f"device_{d}_b") for d in DEV); dev_o = sum(fnum(r, f"device_{d}_o") for d in DEV)
        ph = (fnum(r, "photo_photo_b") + fnum(r, "photo_photo_o")) / 2
        bw, wf, c16, ent = fnum(r, "border_white"), fnum(r, "white_frac"), int(r["colors16"]), fnum(r, "gray_entropy")
        cut = bw >= 0.85 and wf >= 0.30; flat = c16 < 60 or ent < 4.5
        ps = min(ph, 0.15) if (cut or flat) else ph
        ar = auto.get(i, {}); boxes = ar.get("boxes") or []; ob = owl_local.get(i, {}).get("boxes") or []
        owl_dev = max([b["score"] for b in boxes + ob if "stroller" not in b["label"]] or [0.0])
        owl_str = max([b["score"] for b in boxes + ob if "stroller" in b["label"]] or [0.0])
        S[i] = dict(nopeople=min(fnum(r, "child_nopeople_b"), fnum(r, "child_nopeople_o")), child=g("child_child"), child_sig=g("child_sig"), child_auto=(ar.get("screening") or {}).get("child", 0.0),
                    stroller=g("device_stroller"), stroller_sig=g("stroller_sig"), dev=max(dev_b, dev_o), photo=ps, photo_raw=ph,
                    cut=cut, flat=flat, owl_dev=owl_dev, owl_str=owl_str, owl_checked=i in owl_local or bool(boxes),
                    top=max(DEV + [k[7:-2] for k in r if k.startswith("device_x_") and k.endswith("_b")], key=lambda d: fnum(r, f"device_{d}_b")))
    local = set(F)
    if a.calib: return calib(S, st, ex_info, human_dev, labeled, cands, local)
    # ---------------------------------------------------------------- rule 1: child / stroller (all local candidates)
    child, stroller, why = set(), set(), collections.defaultdict(list)
    for i in local:
        s = S[i]; pc = cands[i]["proposed_class"]
        if s["child"] >= TH["child"]: why[i].append(f"siglip child {s['child']:.2f}")
        if s["child_sig"] >= TH["child_sig"]: why[i].append(f"siglip child sigmoid {s['child_sig']:.2f}")
        if s["child_auto"] >= TH["child_auto"] and (s["nopeople"] < TH["nopeople"] or s["child_auto"] >= 0.3): why[i].append(f"auto child flag {s['child_auto']:.2f}")
        if i in st and "child" in ex_info(i)[0]: why[i].append("human child exclude")
        if why[i]: child.add(i)
        sw = []
        if "stroller" in pc: sw.append(f"proposed_class {pc}")
        if s["stroller"] >= TH["stroller"]: sw.append(f"siglip stroller {s['stroller']:.2f}")
        if s["owl_str"] >= 0.3: sw.append(f"owlv2 stroller box {s['owl_str']:.2f}")
        if sw: stroller.add(i); why[i] += sw
    nonlocal_str = sorted(c for c, r in cands.items() if c not in local and "stroller" in r["proposed_class"])
    cs = child | stroller
    # ---------------------------------------------------------------- rule 2: no device (keep when unsure)
    nodev, unsure, owl_needed = set(), set(), []
    for i in sorted(local):
        s = S[i]; hd = human_dev(i)
        if hd in wvcore.DEVICE_CLASSES: continue      # human saw a device: overrides the model
        if s["owl_dev"] >= TH["owl"]: continue
        verified = cands[i]["label_strength"].startswith("human_verified")
        if s["dev"] < TH["dev_absent"] and not verified and s["owl_checked"]: nodev.add(i); why[i].append(f"no device: siglip presence {s['dev']:.2f}, top={s['top']}, owl max {s['owl_dev']:.2f}")
        elif s["dev"] < TH["dev_unsure"]:
            if s["dev"] < TH["dev_absent"] and not s["owl_checked"] and not verified: owl_needed.append(i)
            unsure.add(i)
    (OUT / "owl_needed.txt").write_text("\n".join(owl_needed) + "\n")
    unsure -= nodev | cs
    # ---------------------------------------------------------------- labeled set: human labels only
    def human_ex(i):
        """(exclude reasons a labeler set, how). Latest value per annotator; review-mode keypress acceptances count. Adjudication overrides."""
        if (i, "exclude") in adjv:
            v = adjv[(i, "exclude")][0]; return ({v} if v and v != "none" else set()), "adjudicated"
        rs = {fv.get("exclude") for fv in per.get(i, {}).values() if fv.get("exclude") not in (None, "", "none")}
        if not rs: return set(), ""
        srcs = {lastex.get((i, an, "label"), {}).get("source") for an, fv in per.get(i, {}).items() if fv.get("exclude") in rs}
        return rs, ("typed" if any(x != "auto_review" for x in srcs) else "review_keypress")
    def livebad(live): return live.startswith("changed:") and any(x in live for x in ("ALL-RIGHTS", "-NC", "-ND")) or "login_required" in live
    def lic_issues(i):
        au = audit.get(i, {}); lic = au.get("status", "not_audited"); live = au.get("live_check", ""); rsn = au.get("reason", ""); out = []
        if lic == "needs_ross": out.append("needs_ross: " + rsn)
        if lic == "exclude" and not rsn.startswith("labeler-excluded"): out.append("audit exclude: " + rsn)
        if livebad(live): out.append("live license now " + live)
        if rsn.startswith("same Flickr uploader"): out.append("antefixus21 uploader (6 of their images license-excluded)")
        if lic == "not_audited": out.append("not license-audited (labeled after the audit)")
        return out
    def model_warn(i):
        if i not in S: return ["no local image (model checks not run)"]
        s = S[i]; w = []
        cm = [x for x in why.get(i, []) if "child" in x and not x.startswith("human")]
        if cm: w.append("child: " + ", ".join(cm))
        sm = [x for x in why.get(i, []) if "stroller" in x]
        if sm: w.append("stroller: " + ", ".join(sm))
        if i in nodev: w.append(f"no_device: presence {s['dev']:.2f}, top={s['top']}")
        elif i in unsure_all: w.append(f"device_unsure: presence {s['dev']:.2f}")
        if s["photo"] <= TH["photo_drop"]: w.append(f"low_photo_score {s['photo']:.2f}")
        boxes = [b for bl in st[i]["bboxes"].values() for b in bl]; W, H = int(cands[i]["width"] or 0), int(cands[i]["height"] or 0)
        if boxes and W and H:
            big = max(min(b[2] * W, b[3] * H) for b in boxes)
            if big < TH["small_px"]: w.append(f"small_box {big:.0f}px")
        hd = human_dev(i); sg = (auto.get(i, {}).get("suggestions") or {}).get("device_type") or {}
        if hd and len(st[i]["done"]) == 1 and sg.get("confidence", 0) >= TH["model_conf"] and sg.get("value") != hd:
            w.append(f"model_disagrees: {sg['value']} ({sg['confidence']:.2f}) vs single label {hd}")
        return w
    unsure_all = set(unsure)
    rows = []
    for i in sorted(labeled):
        rs, how = human_ex(i); hd = human_dev(i); au = audit.get(i, {}); lic = au.get("status", "not_audited")
        li = lic_issues(i); mw = model_warn(i); notes = []
        if lic == "ok_with_note": notes.append("license note: " + au.get("reason", ""))
        if st[i]["face_blur_missed"]: notes.append("labeler flagged a missed face blur")
        if lic == "exclude" and not rs and au.get("reason", "").startswith("labeler-excluded"): notes.append("audit listed as labeler-excluded but current labels have no exclude")
        if rs: d = ("drop", f"human exclude: {'/'.join(sorted(rs))} ({how})")
        elif not st[i]["done"]: d = ("needs_label", "labeling not finished (no annotator marked it done)")
        elif hd in (None, "", "unclear"): d = ("needs_label", f"device_type {'conflict/unclear' if hd else 'missing'} in the human labels")
        elif li: d = ("keep_flagged_license", "human-approved; license warning for samir")
        else: d = ("keep", "human-approved with device_type, no human exclude")
        s0 = S.get(i)
        rows.append(dict(image_id=i, decision=d[0], reason=d[1], device_type=hd or "", license_status=lic, exclude_source=how,
                         model_warning="; ".join(mw), license_warning="; ".join(li), photo_score=f"{s0['photo']:.3f}" if s0 else "", notes="; ".join(notes)))
    with open(DATA / "v1_triage.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # ---------------------------------------------------------------- blocklists
    def segs(i):
        out = []
        cm = [x for x in why.get(i, []) if "child" in x]; sm = [x for x in why.get(i, []) if "stroller" in x]
        if i in child: out.append("child: " + ", ".join(cm))
        if i in stroller or i in nonlocal_str: out.append("stroller: " + (", ".join(sm) or "proposed_class stroller"))
        if i in nodev and i in S: out.append(f"no_device: siglip presence {S[i]['dev']:.2f}, top={S[i]['top']}, owl max {S[i]['owl_dev']:.2f}")
        return out
    line = lambda i, extra=(): f"{i}  # " + "; ".join(list(extra) + segs(i))
    hdr = lambda t: f"# {t}\n# one image id per line ('# ...' = reason/scores); read by server/main.py and scripts/build_*.py.\n# Removed entirely (files stay on disk). Restores in the app's Removed review (/removed.html) override this list.\n"
    model_cs = sorted((cs | set(nonlocal_str)) - labeled); model_nd = sorted((nodev - cs) - labeled)
    prev_blocked_lab = (cs | nodev | set(nonlocal_str)) & labeled
    hum = sorted(i for i in prev_blocked_lab if human_ex(i)[0]); restored_lab = sorted(prev_blocked_lab - set(hum))
    (DATA / "removed_model_child_stroller.txt").write_text(hdr(f"MODEL-BASED (unlabeled images only): child (siglip child>={TH['child']} / sigmoid>={TH['child_sig']} / auto flag>={TH['child_auto']}) or stroller (proposed_class, siglip>={TH['stroller']} or OWLv2 stroller box>=0.3)")
        + "\n".join(line(i) for i in model_cs) + "\n", encoding="utf-8")
    (DATA / "removed_model_no_device.txt").write_text(hdr(f"MODEL-BASED (unlabeled images only): no wheelchair/aid/look-alike: siglip2 device presence <{TH['dev_absent']}, no OWLv2 box >={TH['owl']}, not an OpenImages-verified label")
        + "\n".join(line(i) for i in model_nd) + "\n", encoding="utf-8")
    (DATA / "removed_human_exclude.txt").write_text(hdr("HUMAN-FLAGGED: labeled images a labeler excluded (typed, or accepted the model flag with a keypress) that the child/stroller/no-device pass had also caught")
        + "\n".join(line(i, ["human_exclude: " + "/".join(sorted(human_ex(i)[0])) + f" ({human_ex(i)[1]})"]) for i in hum) + "\n", encoding="utf-8")
    for old in ("removed_child_stroller.txt", "removed_no_device.txt"):
        if (DATA / old).exists(): (DATA / old).unlink()
    (DATA / "no_device_uncertain.txt").write_text(f"# device presence unsure (siglip2 presence <{TH['dev_unsure']} but not removable); kept in the queue. Informational only.\n"
        + "\n".join(sorted(unsure)) + "\n", encoding="utf-8")
    # ---------------------------------------------------------------- summary
    removed = set(model_cs) | set(model_nd) | set(hum)
    kept = [r for r in rows if r["decision"] in ("keep", "keep_flagged_license")]
    pool = collections.Counter(cands[i]["proposed_class"] for i in local - removed)
    wcount = lambda key: sum(key in r["model_warning"] for r in kept)
    summ = dict(evsrc=evsrc, thresholds=TH, labeled=len(labeled), decisions=dict(collections.Counter(r["decision"] for r in rows)),
        restored_labeled=len(restored_lab), human_excluded_still_blocked=len(hum),
        model_child_stroller=len(model_cs), model_no_device=len(model_nd), blocked_total=len(removed),
        model_child=len((child - labeled)), model_stroller=len((stroller | set(nonlocal_str)) - labeled), model_child_and_stroller=len((child & stroller) - labeled),
        model_nodev_raw=len(nodev - labeled), uncertain=len(unsure),
        remaining_pool=len(local - removed), remaining_by_class=dict(pool.most_common()),
        lookalike_pool_after=sum(v for k, v in pool.items() if k.startswith("lookalike")),
        keep=sum(r["decision"] == "keep" for r in rows), keep_flagged_license=sum(r["decision"] == "keep_flagged_license" for r in rows),
        kept_by_class=dict(collections.Counter(r["device_type"] for r in kept)),
        keep_only_by_class=dict(collections.Counter(r["device_type"] for r in rows if r["decision"] == "keep")),
        kept_with_model_warning=sum(bool(r["model_warning"]) for r in kept),
        kept_warn_child=wcount("child:"), kept_warn_stroller=wcount("stroller:"), kept_warn_child_or_stroller=sum(("child:" in r["model_warning"]) or ("stroller:" in r["model_warning"]) for r in kept),
        kept_warn_no_device=wcount("no_device:"), kept_warn_device_unsure=wcount("device_unsure"), kept_warn_low_photo=wcount("low_photo"),
        kept_warn_small_box=wcount("small_box"), kept_warn_model_disagrees=wcount("model_disagrees"),
        kept_with_license_warning=sum(bool(r["license_warning"]) for r in kept), kept_face_blur_flag=sum("missed face blur" in r["notes"] for r in kept),
        drop_reasons=dict(collections.Counter(r["reason"] for r in rows if r["decision"] == "drop")),
        needs_label_reasons=dict(collections.Counter(r["reason"] for r in rows if r["decision"] == "needs_label")),
        license_warnings=dict(collections.Counter(x.split(":")[0].split(" (")[0][:40] for r in rows for x in r["license_warning"].split("; ") if x)))
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))
def calib(S, st, ex_info, human_dev, labeled, cands, local):
    import statistics
    lab = [i for i in labeled if i in local]
    pos = [i for i in lab if "child" in ex_info(i)[0]]
    neg = [i for i in lab if not ex_info(i)[0] and human_dev(i) in ("manual", "powered", "other_aid")]
    print("child excludes", len(pos), "clean device-labeled", len(neg))
    for k, ts in (("child", [.02, .05, .1, .2, .3, .5]), ("child_sig", [.01, .02, .05, .1, .2]), ("child_auto", [.1, .2, .3, .5])):
        for t in ts:
            print(f"  {k}>={t}: recall {sum(S[i][k] >= t for i in pos)}/{len(pos)}  neg_hit {sum(S[i][k] >= t for i in neg)}/{len(neg)}  all_local {sum(S[i][k] >= t for i in local)}")
    f = lambda i: S[i]["child"] >= TH["child"] or S[i]["child_sig"] >= TH["child_sig"] or (S[i]["child_auto"] >= TH["child_auto"] and (S[i]["nopeople"] < TH["nopeople"] or S[i]["child_auto"] >= .3))
    print("  FINAL rule recall", sum(map(f, pos)), "/", len(pos), "neg", sum(map(f, neg)), "/", len(neg), "all", sum(map(f, local)))
    for t in [(.5, .1, .1), (.5, .2, .1), (.6, .1, .1), (.4, .1, .1), (.5, .1, .05), (.7, .2, .1), (.5, .05, .1), (.3, .1, .1)]:
        f = lambda i: S[i]["child"] >= t[0] or S[i]["child_sig"] >= t[1] or S[i]["child_auto"] >= t[2]
        print("  combo", t, "recall", sum(map(f, pos)), "/", len(pos), "neg", sum(map(f, neg)), "/", len(neg), "all", sum(map(f, local)))
    strp = [i for i in local if "stroller" in cands[i]["proposed_class"]]
    nons = [i for i in local if "stroller" not in cands[i]["proposed_class"]]
    for t in [.1, .2, .25, .3, .4, .5]:
        print(f"  stroller>={t}: stroller-class {sum(S[i]['stroller'] >= t for i in strp)}/{len(strp)}  human-device {sum(S[i]['stroller'] >= t for i in neg)}/{len(neg)}  non-stroller-class {sum(S[i]['stroller'] >= t for i in nons)}/{len(nons)}")
    hd_all = [i for i in lab if human_dev(i) in wvcore.DEVICE_CLASSES]
    v = sorted(S[i]["dev"] for i in hd_all); print("device presence on human-device images: n", len(v), "p1/p5/p10", v[len(v)//100], v[len(v)//20], v[len(v)//10])
    for t in [.05, .1, .2, .35, .5]:
        print(f"  dev<{t}: human-device {sum(S[i]['dev'] < t for i in hd_all)}  all_local {sum(S[i]['dev'] < t for i in local)}  weak-label {sum(S[i]['dev'] < t and not cands[i]['label_strength'].startswith('human') for i in local)}")
    npos = [i for i in lab if "not_a_photo" in ex_info(i)[0]]; real = [i for i in lab if not ex_info(i)[0]]
    for t in [.3, .5, .8, .9]:
        print(f"  photo>={t}: not_a_photo-excl {sum(S[i]['photo'] >= t for i in npos)}/{len(npos)}  clean {sum(S[i]['photo'] >= t for i in real)}/{len(real)}")
    hu = [i for i in npos if ex_info(i)[2] == "human"]
    print("  human-typed not_a_photo", len(hu), "photo>=0.9", sum(S[i]["photo"] >= .9 for i in hu))
if __name__ == "__main__": main()
