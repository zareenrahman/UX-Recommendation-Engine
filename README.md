# UX Recommendation Engine

> An AI-powered assistant that helps designers discover UX ideas and patterns instantly.

Live demo: https://ux-recommendation-engine.streamlit.app/

## What it does
Type a UX/design topic (e.g., *form validation*, *error message UX*, *dark mode*) and get relevant
patterns, components, and snippets ranked by semantic similarity.

## Highlights
- TF-IDF + cosine similarity over a curated UX catalog
- Filters (type, tags) and keyword boosting
- One-click retrain from CSV
- Clean, demo-ready Streamlit UI

## Quick start (local)
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python train.py
python -m streamlit run app.py
