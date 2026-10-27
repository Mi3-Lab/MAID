"""Triage of the v1 candidate pool (local only; images never leave this machine).
Inputs: var/triage/features.csv (scripts/triage_features.py), var/wheelview.db (read-only) or annotations/events,
annotations/auto/*.jsonl, data/candidates.csv, data/license_audit_v1.csv.
Outputs: data/removed_child_stroller.txt, data/removed_no_device.txt, data/no_device_uncertain.txt, data/v1_triage.csv,
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
    evs, evsrc = load_events(); evs = [e for e in evs if e["annotator"] in LABELERS]
    per, adjv = wvcore.fold(evs)
    done = {i for i, anns in per.items() if any(fv.get("_status") == "done" for fv in anns.values())}
    labeled = done | set(audit)
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
        if hd in ("manual", "powered", "other_aid", "lookalike"): continue      # human saw a device: overrides the model
        if s["owl_dev"] >= TH["owl"]: continue
        verified = cands[i]["label_strength"].startswith("human_verified")
        if s["dev"] < TH["dev_absent"] and not verified and s["owl_checked"]: nodev.add(i); why[i].append(f"no device: siglip presence {s['dev']:.2f}, top={s['top']}, owl max {s['owl_dev']:.2f}")
        elif s["dev"] < TH["dev_unsure"]:
            if s["dev"] < TH["dev_absent"] and not s["owl_checked"] and not verified: owl_needed.append(i)
            unsure.add(i)
    (OUT / "owl_needed.txt").write_text("\n".join(owl_needed) + "\n")
    unsure -= nodev | cs
    # ---------------------------------------------------------------- labeled set decisions
    rows = []
    for i in sorted(labeled):
        rs, how, src = ex_info(i); hd = human_dev(i); au = audit.get(i, {}); lic = au.get("status", "not_audited")
        s = S.get(i, {}); notes = []
        if lic == "ok_with_note": notes.append("license note: " + au.get("reason", ""))
        if st[i]["face_blur_missed"]: notes.append("labeler flagged a missed face blur")
        live = au.get("live_check", "")
        livebad = live.startswith("changed:") and any(x in live for x in ("ALL-RIGHTS", "-NC", "-ND")) or "login_required" in live
        d = None
        if i not in local: d = ("drop", "no local image")
        elif i in child: d = ("drop", "child (rule 1): " + "; ".join(x for x in why[i] if "stroller" not in x and "no device" not in x))
        elif i in stroller: d = ("drop", "stroller (rule 1): " + "; ".join(x for x in why[i] if "stroller" in x))
        elif i in nodev: d = ("drop", "no device (rule 2)")
        elif lic == "needs_ross": d = ("drop", "license needs_ross")
        elif livebad: d = ("drop", f"live license now {live}")
        elif "license_concern" in rs and src == "human": d = ("drop", "human license_concern exclude")
        elif "identifiable_person" in rs or "clinical_setting" in rs or "license_concern" in rs:
            r0 = "/".join(sorted(rs - {"not_a_photo"}))
            d = ("drop", f"human {r0} exclude ({how})") if src == "human" else ("user_decides", f"{r0} exclude accepted from model flag ({how}); borderline consent case")
        if d is None and "not_a_photo" in rs:
            ps = s["photo"]; kt = TH["photo_keep_human"] if src == "human" else TH["photo_keep"]
            real_dev = hd in ("manual", "powered", "other_aid", "lookalike")
            if ps <= TH["photo_drop"]: d = ("drop", f"not_a_photo confirmed (photo_score {ps:.2f}{', cut-out' if s['cut'] else ''}{', flat graphic' if s['flat'] else ''})")
            elif not real_dev and s["dev"] < TH["dev_unsure"]: d = ("drop", f"not_a_photo exclude upheld: real photo but no clear device (presence {s['dev']:.2f})")
            elif ps < kt: d = ("user_decides", f"not_a_photo exclude ({src}) but photo check ambiguous ({ps:.2f})")
            elif real_dev: notes.append(f"recovered_not_a_photo (exclude {src}, photo_score {ps:.2f})")
            elif s["dev"] >= TH["recover_dev"]: d = ("user_decides", f"not_a_photo exclude ({src}): real photo ({ps:.2f}) with likely {s['top']} (presence {s['dev']:.2f}), needs device_type"); notes.append("recovered_not_a_photo")
            else: d = ("drop", f"not_a_photo exclude upheld: real photo, device unclear (presence {s['dev']:.2f})")
        if d is None and lic == "exclude" and not rs and au.get("reason", "").startswith("labeler-excluded"): notes.append("audit listed as labeler-excluded but current labels have no exclude")
        if d is None and lic == "not_audited": d = ("user_decides", "labeled after the license audit; not audited")
        if d is None and hd in (None, "", "unclear"): d = ("user_decides", f"device_type {'conflict/unclear' if hd else 'missing'}")
        if d is None:
            boxes = [b for bl in st[i]["bboxes"].values() for b in bl]; W, H = int(cands[i]["width"] or 0), int(cands[i]["height"] or 0)
            if boxes and W and H:
                big = max(min(b[2] * W, b[3] * H) for b in boxes)
                if big < TH["small_px"]: d = ("drop", f"device too small ({big:.0f}px shorter side)")
            elif not boxes: notes.append("no bbox")
        if d is None and au.get("reason", "").startswith("same Flickr uploader"): d = ("user_decides", "antefixus21 uploader (6 of their images excluded for license)")
        if d is None and len(st[i]["done"]) == 1 and ("device_type" not in {f for (ii, f) in adjv if ii == i}):
            sg = (auto.get(i, {}).get("suggestions") or {}).get("device_type") or {}
            if sg.get("confidence", 0) >= TH["model_conf"] and sg.get("value") != hd: d = ("user_decides", f"single label {hd} vs model {sg['value']} ({sg['confidence']:.2f})")
        if d is None and st[i]["face_blur_missed"]: d = ("user_decides", "labeler flagged a missed face blur")
        if d is None: d = ("keep", "allowed license, no exclude, real photo, device labeled")
        rows.append(dict(image_id=i, decision=d[0], reason=d[1], device_type=hd or "", license_status=lic, exclude_source=src,
                         photo_score=f"{s['photo']:.3f}" if s else "", notes="; ".join(notes)))
    with open(DATA / "v1_triage.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    hdr = lambda t: f"# {t}\n# one image id per line; read by server/main.py and scripts/build_*.py (removed entirely; files stay on disk)\n"
    (DATA / "removed_child_stroller.txt").write_text(hdr(f"child (any score over child>={TH['child']} / sigmoid>={TH['child_sig']} / auto>={TH['child_auto']} / human child exclude) or stroller (proposed_class or siglip>={TH['stroller']} or OWLv2 stroller box>=0.3); +{len(nonlocal_str)} non-local stroller-class candidates")
        + "\n".join(sorted(cs | set(nonlocal_str))) + "\n", encoding="utf-8")
    (DATA / "removed_no_device.txt").write_text(hdr(f"no wheelchair/aid/look-alike: siglip2 device presence <{TH['dev_absent']}, no OWLv2 box >={TH['owl']}, no human device label, not an OpenImages-verified label")
        + "\n".join(sorted(nodev - cs)) + "\n", encoding="utf-8")
    (DATA / "no_device_uncertain.txt").write_text(f"# device presence unsure (siglip2 presence <{TH['dev_unsure']} but not removable); kept in the queue\n"
        + "\n".join(sorted(unsure)) + "\n", encoding="utf-8")
    # ---------------------------------------------------------------- summary
    removed = cs | nodev; keep = [r for r in rows if r["decision"] == "keep"]; ud = [r for r in rows if r["decision"] == "user_decides"]
    pool = collections.Counter(cands[i]["proposed_class"] for i in local - removed)
    summ = dict(evsrc=evsrc, thresholds=TH, labeled=len(labeled), decisions=dict(collections.Counter(r["decision"] for r in rows)),
        child=len(child), stroller=len(stroller), child_and_stroller=len(child & stroller), stroller_nonlocal=len(nonlocal_str),
        nodev_raw=len(nodev), nodev_overlap_child_stroller=len(nodev & cs), nodev_only=len(nodev - cs), uncertain=len(unsure), owl_needed=len(owl_needed),
        removed_local=len(removed), removed_labeled=len(removed & labeled), removed_unlabeled=len(removed - labeled),
        child_labeled=len(child & labeled), stroller_labeled=len(stroller & labeled), nodev_labeled=len((nodev - cs) & labeled),
        remaining_pool=len(local - removed), remaining_by_class=dict(pool.most_common()),
        lookalike_pool_after=sum(v for k, v in pool.items() if k.startswith("lookalike")),
        keep=len(keep), keep_by_class=dict(collections.Counter(r["device_type"] for r in keep)),
        keep_if_all_ud=len(keep) + len(ud), keep_if_all_ud_by_class=dict(collections.Counter((r["device_type"] or "?") for r in keep + ud)),
        recovered_not_a_photo=sum("recovered_not_a_photo" in r["notes"] for r in rows),
        recovered_kept=sum("recovered_not_a_photo" in r["notes"] and r["decision"] == "keep" for r in rows),
        not_a_photo_total=sum(1 for i in labeled if "not_a_photo" in ex_info(i)[0]),
        drop_reasons=dict(collections.Counter(r["reason"].split(":")[0].split(" (")[0] for r in rows if r["decision"] == "drop")),
        ud_reasons=dict(collections.Counter(r["reason"].split(" (")[0][:60] for r in ud)))
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
    hd_all = [i for i in lab if human_dev(i) in ("manual", "powered", "other_aid", "lookalike")]
    v = sorted(S[i]["dev"] for i in hd_all); print("device presence on human-device images: n", len(v), "p1/p5/p10", v[len(v)//100], v[len(v)//20], v[len(v)//10])
    for t in [.05, .1, .2, .35, .5]:
        print(f"  dev<{t}: human-device {sum(S[i]['dev'] < t for i in hd_all)}  all_local {sum(S[i]['dev'] < t for i in local)}  weak-label {sum(S[i]['dev'] < t and not cands[i]['label_strength'].startswith('human') for i in local)}")
    npos = [i for i in lab if "not_a_photo" in ex_info(i)[0]]; real = [i for i in lab if not ex_info(i)[0]]
    for t in [.3, .5, .8, .9]:
        print(f"  photo>={t}: not_a_photo-excl {sum(S[i]['photo'] >= t for i in npos)}/{len(npos)}  clean {sum(S[i]['photo'] >= t for i in real)}/{len(real)}")
    hu = [i for i in npos if ex_info(i)[2] == "human"]
    print("  human-typed not_a_photo", len(hu), "photo>=0.9", sum(S[i]["photo"] >= .9 for i in hu))
if __name__ == "__main__": main()
