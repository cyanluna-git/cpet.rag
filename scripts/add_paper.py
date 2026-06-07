#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add a paper to the shared sinico corpus (dedup + name + index + OA fetch).

Usage:
  python3 add_paper.py <DOI> --by <이름> [--pdf <파일경로>] [--source <태그>]

- DOI가 이미 corpus_index.csv에 있으면 거부 (중복 방지)
- --pdf 주면 그 파일을 규칙대로 이름 바꿔 pdf/에 넣음
- --pdf 없으면 Unpaywall로 OA PDF 자동 시도 (유료면 메타만 기록)
"""
import os, re, csv, sys, json, argparse, datetime, shutil
import urllib.request, urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # corpus root
PDF_DIR = os.path.join(ROOT, "pdf")
INDEX = os.path.join(ROOT, "index", "corpus_index.csv")
EMAIL = "cte5301@gmail.com"
HEAD = ["doi","title","first_author","year","journal","source","file","oa_status","added_by","added_at"]
STOP = {"the","a","an","of","and","in","on","for","to","with","via","using","study"}

def norm(d):
    m = re.search(r'10\.\d{4,9}/\S+', (d or "").strip()); return m.group(0).lower().rstrip('.') if m else (d or "").lower()
def slug(s, n=40): return re.sub(r'[^A-Za-z0-9가-힣]+','-',(s or '')).strip('-')[:n]

def load_index():
    if not os.path.exists(INDEX): return []
    return list(csv.DictReader(open(INDEX)))

def crossref(doi):
    try:
        u = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}?mailto={EMAIL}"
        m = json.load(urllib.request.urlopen(u, timeout=20))["message"]
        au = (m.get("author") or [{}])[0]
        return {"title": (m.get("title") or [""])[0],
                "author": au.get("family") or au.get("name") or "",
                "year": (m.get("issued",{}).get("date-parts",[[None]])[0] or [None])[0],
                "journal": (m.get("container-title") or [""])[0]}
    except Exception: return {}

def unpaywall_pdf(doi):
    try:
        u = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={EMAIL}"
        loc = (json.load(urllib.request.urlopen(u, timeout=20)).get("best_oa_location") or {})
        return loc.get("url_for_pdf")
    except Exception: return None

def fetch(url, dst):
    try:
        req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0","Accept":"application/pdf"})
        data = urllib.request.urlopen(req, timeout=60).read()
        if data[:4] == b"%PDF" and len(data) > 5000:
            open(dst,"wb").write(data); return True
    except Exception: pass
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("doi"); ap.add_argument("--by", required=True)
    ap.add_argument("--pdf"); ap.add_argument("--source", default="manual")
    a = ap.parse_args()
    doi = norm(a.doi)
    if not doi.startswith("10."):
        print("❌ 올바른 DOI가 아닙니다:", a.doi); sys.exit(1)

    rows = load_index()
    for r in rows:
        if norm(r.get("doi")) == doi:
            print(f"⚠️ 이미 있음 (중복): {r.get('file')} — {r.get('added_by')} 추가")
            sys.exit(0)

    meta = crossref(doi)
    title = meta.get("title",""); author = meta.get("author","")
    year = meta.get("year","") or ""; journal = meta.get("journal","")
    kw = next((w for w in re.findall(r'[A-Za-z가-힣]+', title) if w.lower() not in STOP and len(w)>2), "paper")
    base = f"{year or 'nd'}_{slug(author,20) or 'unknown'}_{slug(kw,20)}"
    fname = base + ".pdf"

    placed = ""
    os.makedirs(PDF_DIR, exist_ok=True)
    if a.pdf:
        if not os.path.exists(a.pdf): print("❌ --pdf 파일 없음:", a.pdf); sys.exit(1)
        shutil.copy2(a.pdf, os.path.join(PDF_DIR, fname)); placed = fname
        print(f"✅ PDF 추가: {fname}")
    else:
        up = unpaywall_pdf(doi)
        if up and fetch(up, os.path.join(PDF_DIR, fname)):
            placed = fname; print(f"✅ OA PDF 다운로드: {fname}")
        else:
            print("ℹ️ OA PDF 없음 — 메타데이터만 기록 (PDF는 직접 받아 --pdf로 추가하세요)")

    row = {"doi": doi, "title": title[:200], "first_author": author, "year": year,
           "journal": journal, "source": a.source, "file": placed, "oa_status": "",
           "added_by": a.by, "added_at": datetime.date.today().isoformat()}
    new = not os.path.exists(INDEX)
    with open(INDEX, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEAD)
        if new: w.writeheader()
        w.writerow(row)
    print(f"📝 인덱스 기록 완료 — {title[:60]} ({journal} {year})")

if __name__ == "__main__":
    main()
