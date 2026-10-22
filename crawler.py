"""Interlinq crawler: collect pages from a sitemap, a URL list, or a local folder."""
import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

CONTACT = "hafsakoaml445@gmail.com"  # put your email here
HEADERS = {"User-Agent": f"InterlinqBot/0.1 (learning project; {CONTACT})"}
TIMEOUT = 15
MIN_CHARS = 200
MAX_CHARS = 6000

_robots_cache = {}


def allowed(url):
    """Return True if the site's robots.txt allows fetching this URL."""
    parts = urlparse(url)
    base = f"{parts.scheme}://{parts.netloc}"
    if base not in _robots_cache:
        rp = RobotFileParser()
        try:
            r = requests.get(base + "/robots.txt", headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 200:
                rp.parse(r.text.splitlines())
            else:
                rp = None  # no robots.txt means no restrictions
        except requests.RequestException:
            rp = None
        _robots_cache[base] = rp
    rp = _robots_cache[base]
    return True if rp is None else rp.can_fetch(HEADERS["User-Agent"], url)


def clean_html(html, source):
    """Extract title and main text from an HTML string."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else source
    main = soup.find("article") or soup.find("main") or soup.body
    text = main.get_text(" ", strip=True) if main else ""
    if len(text) < MIN_CHARS:
        return None
    return {"url": source, "title": title, "text": text[:MAX_CHARS]}


def fetch_page(url):
    if not allowed(url):
        print("  skipped (blocked by robots.txt)")
        return None
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  skipped: {e}")
        return None
    return clean_html(resp.text, url)


def get_sitemap_urls(sitemap_url, limit):
    """Return up to `limit` page URLs from a sitemap or sitemap index."""
    urls = []
    try:
        resp = requests.get(sitemap_url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Could not fetch sitemap {sitemap_url}: {e}")
        return urls
    soup = BeautifulSoup(resp.content, "xml")
    for loc in soup.select("sitemap > loc"):
        urls.extend(get_sitemap_urls(loc.text.strip(), limit - len(urls)))
        if len(urls) >= limit:
            return urls[:limit]
    for loc in soup.select("url > loc"):
        urls.append(loc.text.strip())
        if len(urls) >= limit:
            break
    return urls[:limit]


def read_url_file(path, limit):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    urls = [l.strip() for l in lines if l.strip() and not l.startswith("#")]
    return list(dict.fromkeys(urls))[:limit]  # remove duplicates, keep order


def crawl_urls(urls, delay):
    pages = []
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {url}")
        page = fetch_page(url)
        if page:
            pages.append(page)
        time.sleep(delay)  # be polite to the server
    return pages


def read_folder(folder):
    pages = []
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix.lower() not in {".txt", ".md", ".html", ".htm"}:
            continue
        raw = path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix.lower() in {".html", ".htm"}:
            page = clean_html(raw, str(path))
        else:
            first_line = raw.strip().splitlines()[0].lstrip("# ").strip() if raw.strip() else path.stem
            page = {"url": str(path), "title": first_line, "text": raw[:MAX_CHARS]} if len(raw) >= MIN_CHARS else None
        if page:
            pages.append(page)
    return pages


def main():
    parser = argparse.ArgumentParser(description="Interlinq crawler")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sitemap", help="URL of a sitemap.xml")
    source.add_argument("--urls", help="text file with one URL per line")
    source.add_argument("--folder", help="folder of .txt/.md/.html files")
    parser.add_argument("--limit", type=int, default=30, help="max pages (default 30)")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    parser.add_argument("--out", default="pages.json")
    args = parser.parse_args()

    if args.folder:
        pages = read_folder(args.folder)
    else:
        urls = get_sitemap_urls(args.sitemap, args.limit) if args.sitemap else read_url_file(args.urls, args.limit)
        print(f"Found {len(urls)} URLs")
        pages = crawl_urls(urls, args.delay)

    Path(args.out).write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(pages)} pages to {args.out}")


if __name__ == "__main__":
    main()