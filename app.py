"""Interlinq web app (Streamlit).

Run locally:
    pip install streamlit
    streamlit run app.py
"""
import os
import time

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from sentence_transformers import SentenceTransformer

from crawler import fetch_page
from indexer import MODEL_NAME, TEXT_CHARS, related_pages
from suggest_links import CANDIDATES, DELAY_SECONDS, MIN_SIMILARITY, ask_gemini, validate

MAX_PAGES = 15          # free-tier limit, change later for paid plans
CRAWL_DELAY = 1.0       # seconds between page downloads

st.set_page_config(page_title="Interlinq", page_icon="🔗", layout="wide")


@st.cache_resource(show_spinner="Loading the embedding model (first run only)...")
def get_model():
    return SentenceTransformer(MODEL_NAME)


def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]  # used when deployed on Streamlit Cloud
    except Exception:
        return os.getenv("GEMINI_API_KEY")   # used locally via .env


def parse_urls(raw):
    urls = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith(("http://", "https://")) and line not in urls:
            urls.append(line)
    return urls[:MAX_PAGES]


def build_index(pages):
    model = get_model()
    docs = [f"{p['title']}. {p['text'][:TEXT_CHARS]}" for p in pages]
    emb = model.encode(docs, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index, emb


def run_pipeline(urls, api_key):
    # Step 1: crawl
    pages = []
    bar = st.progress(0.0, text="Reading pages...")
    for i, url in enumerate(urls, 1):
        page = fetch_page(url)
        if page:
            pages.append(page)
        bar.progress(i / len(urls), text=f"Reading pages... {i}/{len(urls)}")
        time.sleep(CRAWL_DELAY)
    if len(pages) < 2:
        bar.empty()
        st.error("Could not read enough pages. Need at least 2 readable pages. "
                 "Some sites block bots or load content with JavaScript.")
        return None, len(pages)

    # Step 2: embeddings and FAISS
    bar.progress(0.0, text="Finding related pages...")
    index, emb = build_index(pages)
    pages_by_url = {p["url"]: p for p in pages}

    # Step 3: Gemini anchor text
    client = genai.Client(api_key=api_key)
    rows = []
    for i, page in enumerate(pages):
        bar.progress(i / len(pages), text=f"Suggesting links... {i + 1}/{len(pages)}")
        related = [r for r in related_pages(index, emb, pages, i, k=CANDIDATES) if r[2] >= MIN_SIMILARITY]
        if not related:
            continue
        raw, source_text = ask_gemini(client, page, related, pages_by_url)
        for s in validate(raw, source_text, {r[1] for r in related}):
            rows.append({
                "Source page": page["title"],
                "Source URL": page["url"],
                "Anchor text": s["anchor_text"],
                "Link to": pages_by_url[s["target_url"]]["title"],
                "Target URL": s["target_url"],
                "Why": s["reason"],
            })
        time.sleep(DELAY_SECONDS)
    bar.empty()
    return pd.DataFrame(rows), len(pages)


# ---------------------------------------------------------------- UI
st.title("🔗 Interlinq")
st.caption("Find internal linking opportunities with AI. Paste your page URLs, get link suggestions with anchor text.")

api_key = get_api_key()
if not api_key:
    st.error("GEMINI_API_KEY is missing. Add it to your .env file (local) or Streamlit secrets (deployed).")
    st.stop()

raw_urls = st.text_area(
    f"Page URLs (one per line, up to {MAX_PAGES})",
    height=220,
    placeholder="https://yoursite.com/blog/post-1\nhttps://yoursite.com/blog/post-2",
)
st.caption("Only analyse sites you own or have permission to analyse. Interlinq respects robots.txt.")

if st.button("Find link opportunities", type="primary"):
    urls = parse_urls(raw_urls)
    if len(urls) < 2:
        st.warning("Enter at least 2 URLs (each starting with http:// or https://).")
    else:
        df, n_pages = run_pipeline(urls, api_key)
        if df is not None:
            st.session_state["results"] = df
            st.session_state["n_pages"] = n_pages

if "results" in st.session_state:
    df = st.session_state["results"]
    st.success(f"Analysed {st.session_state['n_pages']} pages and found {len(df)} link suggestions.")
    if df.empty:
        st.info("No strong link opportunities found. Try adding more related pages.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="interlinq_suggestions.csv",
            mime="text/csv",
        )