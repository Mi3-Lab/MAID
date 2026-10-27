"""Local contact sheet for triage review: var/triage/contact_sheet.html (gitignored; links to ../../images_blurred, nothing is copied).
Sections: user_decides (grouped by reason), a sample of not_a_photo excludes recovered by the photo check, and samples of
the child/stroller and no-device removals plus the no-device-uncertain list for spot checks."""
import csv, html, random, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "var" / "triage"; DATA = ROOT / "data"
def ids(p): return [l.strip() for l in open(p, encoding="utf-8") if l.strip() and not l.startswith("#")]
def card(i, cap):
    return (f'<figure><a href="../../images_blurred/{i}.jpg" target="_blank"><img loading="lazy" src="../../images_blurred/{i}.jpg"></a>'
            f'<figcaption><b>{i}</b><br>{html.escape(cap)}</figcaption></figure>')
def main():
    random.seed(0); R = list(csv.DictReader(open(DATA / "v1_triage.csv", encoding="utf-8")))
    feats = {r["image_id"]: r for r in csv.DictReader(open(OUT / "features.csv", encoding="utf-8"))}
    sec = []
    ud = collections.defaultdict(list)
    for r in R:
        if r["decision"] == "user_decides": ud[r["reason"].split(" (")[0].split(":")[0][:50]].append(r)
    for k, rs in sorted(ud.items(), key=lambda kv: -len(kv[1])):
        sec.append((f"user_decides: {k} ({len(rs)})", [(r["image_id"], f"{r['reason']} | dev={r['device_type'] or '-'} lic={r['license_status']} {r['notes']}") for r in rs]))
    rec = [r for r in R if "recovered_not_a_photo" in r["notes"] and r["decision"] == "keep"]
    sec.insert(0, (f"not_a_photo excludes recovered to KEEP ({len(rec)})", [(r["image_id"], f"photo {r['photo_score']} | {r['device_type']} | {r['notes']}") for r in rec]))
    for name, f, n in (("removed: child/stroller (random 60)", "removed_child_stroller.txt", 60), ("removed: no device (random 60)", "removed_no_device.txt", 60),
                       ("kept in queue: no_device_uncertain (random 40)", "no_device_uncertain.txt", 40)):
        L = ids(DATA / f); s = random.sample(L, min(n, len(L)))
        sec.append((f"{name} of {len(L)}", [(i, "") for i in s if i in feats]))
    css = ("body{font:13px sans-serif;margin:16px}section{display:flex;flex-wrap:wrap;gap:8px}figure{width:220px;margin:0}"
           "img{width:220px;height:180px;object-fit:contain;background:#eee}figcaption{font-size:11px;word-break:break-word}")
    body = "".join(f"<h2>{html.escape(t)}</h2><section>{''.join(card(i, c) for i, c in items)}</section>" for t, items in sec)
    (OUT / "contact_sheet.html").write_text(f"<!doctype html><meta charset=utf-8><title>WheelView v1 triage</title><style>{css}</style>"
        "<h1>WheelView v1 triage (local only, blurred copies)</h1>" + body, encoding="utf-8")
    print(OUT / "contact_sheet.html", sum(len(x[1]) for x in sec))
if __name__ == "__main__": main()
