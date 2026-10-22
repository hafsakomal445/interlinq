"""Interlinq link suggester: FAISS finds related pages, Gemini picks anchor text.

Setup:
    pip install google-genai python-dotenv
    Create a .env file in this folder containing:
        GEMINI_API_KEY=your_key_here
    (optional) GEMINI_MODEL=gemini-2.5-flash

Run:
    python suggest_links.py              # all pages
    python suggest_links.py --max-pages 3   # quick test on 3 pages
"""
import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from indexer import build_index, load_pages, related_pages

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
SOURCE_CHARS = 3000      # how much of the source page Gemini sees
SNIPPET_CHARS = 300      # how much of each candidate page Gemini sees
CANDIDATES = 4           # related pages retrieved per source page
MIN_SIMILARITY = 0.45    # ignore weak matches
DELAY_SECONDS = 4        # pause between calls to stay inside free-tier limits

PROMPT = """You are an SEO internal-linking assistant.

SOURCE PAGE
Title: {title}
URL: {url}
Text:
{text}

CANDIDATE PAGES TO LINK TO
{candidates}

Task: suggest internal links FROM the source page TO the candidate pages.
Rules:
- Only suggest a link if it would genuinely help a reader.
- "anchor_text" MUST be an exact phrase copied from the source text above (2 to 6 words).
- Use each candidate at most once.
- "reason" is one short sentence.
- If no link makes sense, return an empty list.
- Prefer descriptive, specific anchors. Avoid generic phrases like "web page", "website", "click here" or "search engine".
- Suggest at most 3 links, and only the strongest ones.
- Anchor text must describe the target page’s specific topic.
Return ONLY JSON in this shape:
[{{"target_url": "...", "anchor_text": "...", "reason": "..."}}]
"""


def format_candidates(related, pages_by_url):
    lines = []
    for title, url, score in related:
        snippet = pages_by_url[url]["text"][:SNIPPET_CHARS].replace("\n", " ")
        lines.append(f"- Title: {title}\n  URL: {url}\n  Summary: {snippet}")
    return "\n".join(lines)


def ask_gemini(client, page, related, pages_by_url, retries=3):
    source_text = page["text"][:SOURCE_CHARS]
    prompt = PROMPT.format(
        title=page["title"],
        url=page["url"],
        text=source_text,
        candidates=format_candidates(related, pages_by_url),
    )
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            data = json.loads(response.text)
            return data if isinstance(data, list) else [], source_text
        except Exception as e:  # network errors, rate limits, bad JSON
            print(f"   attempt {attempt} failed: {e}")
            time.sleep(DELAY_SECONDS * attempt * 2)
    return [], source_text


def validate(suggestions, source_text, allowed_urls):
    """Keep only suggestions whose anchor text really appears in the page
    and whose target is one of the retrieved candidates."""
    good = []
    used = set()
    lowered = source_text.lower()
    for s in suggestions:
        if not isinstance(s, dict):
            continue
        url = s.get("target_url")
        anchor = (s.get("anchor_text") or "").strip()
        if url not in allowed_urls or url in used or not anchor:
            continue
        if anchor.lower() not in lowered:
            continue  # model invented a phrase that is not on the page
        used.add(url)
        good.append({"target_url": url, "anchor_text": anchor, "reason": s.get("reason", "")})
    return good


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--out", default="suggestions.json")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not found. Create a .env file with GEMINI_API_KEY=your_key")
    client = genai.Client(api_key=api_key)

    pages = load_pages()
    pages_by_url = {p["url"]: p for p in pages}
    print(f"Loaded {len(pages)} pages. Building index...")
    index, emb = build_index(pages)

    results = []
    todo = range(len(pages)) if args.max_pages is None else range(min(args.max_pages, len(pages)))
    for i in todo:
        page = pages[i]
        print(f"\n[{i + 1}/{len(pages)}] {page['title']}")
        related = [r for r in related_pages(index, emb, pages, i, k=CANDIDATES) if r[2] >= MIN_SIMILARITY]
        if not related:
            print("   no strong matches, skipping")
            continue
        raw, source_text = ask_gemini(client, page, related, pages_by_url)
        good = validate(raw, source_text, {r[1] for r in related})
        for s in good:
            print(f"   link '{s['anchor_text']}' -> {s['target_url']}\n      {s['reason']}")
        if not good:
            print("   no valid suggestions")
        results.append({"source_url": page["url"], "source_title": page["title"], "suggestions": good})
        time.sleep(DELAY_SECONDS)

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(r["suggestions"]) for r in results)
    print(f"\nDone. {total} link suggestions saved to {args.out}")


if __name__ == "__main__":
    main()