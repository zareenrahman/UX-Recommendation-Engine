import streamlit as st
import pandas as pd
import joblib, uuid, time, base64, os
from pathlib import Path
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="UX Recommendation Engine", page_icon="🎨", layout="wide")

def set_bg(image_path: str | None = None):
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url("data:image/png;base64,{b64}") no-repeat center center fixed;
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp { background: radial-gradient(1200px 600px at 10% 0%, #f7f8fc, #eef2ff) fixed; }
            </style>
            """,
            unsafe_allow_html=True,
        )

set_bg(None)

st.markdown("""
<style>
.hero { background: linear-gradient(135deg,#111827,#1f2937); color:#fff; padding:22px 24px; border-radius:18px; box-shadow:0 10px 24px rgba(0,0,0,0.15); margin-bottom:18px; }
.hero h1 { font-size:28px; line-height:1.2; margin:0 0 6px 0; }
.hero p { margin:0; opacity:.9; }
[data-testid="stSidebar"] > div:first-child { background: rgba(255,255,255,.72); backdrop-filter: blur(6px); border-right:1px solid rgba(0,0,0,.06); }
.stButton>button { border-radius:10px; }
code, pre { border-radius:10px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <h1>🎨 UX Recommendation Engine</h1>
      <p>An AI-powered assistant that helps designers discover UX ideas and patterns instantly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("Type a UX/design topic and get relevant patterns, components, and snippets.")

MODEL_PATH = Path("tfidf_model.joblib")
MATRIX_PATH = Path("items_matrix.npz")
DATA_PATH  = Path("items_with_bag.csv")
RAW_DATA   = Path("data/items.csv")
LOG_PATH   = Path("interactions.csv")

missing = [p.name for p in [MODEL_PATH, MATRIX_PATH, DATA_PATH] if not p.exists()]
if missing:
    st.error(f"Missing files: {', '.join(missing)}. Run `python train.py` first.")
    st.stop()

@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load(MODEL_PATH)
    X = sparse.load_npz(MATRIX_PATH)
    df = pd.read_csv(DATA_PATH)
    return vectorizer, X, df

vectorizer, X, df = load_artifacts()

with st.sidebar:
    st.header("Filters & Settings")
    k = st.slider("Number of recommendations", 3, 15, 5)
    types = ["(any)"] + sorted(df["type"].dropna().unique().tolist())
    type_filter = st.selectbox("Type", types)
    tag_filter = st.text_input("Must include tag (optional)")
    boost_keyword = st.text_input("Boost keyword (optional)")
    st.caption("Tip: Try queries like *form accessibility*, *checkout UX*, *error messaging*")
    st.markdown("---")
    st.subheader("About")
    st.write("This demo uses TF-IDF + cosine similarity to recommend UX patterns.")

query = st.text_input("Enter a topic:", placeholder="e.g., form validation accessibility")
go = st.button("Recommend")

def rank(query_text: str):
    q = vectorizer.transform([query_text])
    sims = cosine_similarity(q, X).ravel()
    if boost_keyword:
        boost = df["bag"].fillna("").str.contains(boost_keyword, case=False, regex=False).astype(float)
        sims = sims * (1.15 ** boost.values)
    mask = pd.Series([True] * len(df))
    if type_filter != "(any)":
        mask &= (df["type"].fillna("") == type_filter)
    if tag_filter:
        mask &= df["tags"].fillna("").str.contains(tag_filter, case=False, regex=False)
    idx_all = sims.argsort()[::-1]
    idx = [i for i in idx_all if mask.iloc[i]][:k]
    out = df.iloc[idx][["item_id", "title", "type", "tags", "text"]].copy()
    out.insert(1, "score", [round(float(sims[i]), 4) for i in idx])
    return out

def log_event(event: str, item_id: str | None, query_text: str, score: float | None = None):
    LOG_PATH.touch(exist_ok=True)
    header_needed = LOG_PATH.stat().st_size == 0
    row = pd.DataFrame([{
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session": st.session_state.get("sid", ""),
        "event": event,
        "item_id": item_id or "",
        "query": query_text,
        "score": score if score is not None else "",
    }])
    row.to_csv(LOG_PATH, mode="a", header=header_needed, index=False)

if "sid" not in st.session_state:
    st.session_state["sid"] = str(uuid.uuid4())[:8]

if go and query.strip():
    results = rank(query)
    if results.empty:
        st.warning("No matches. Try changing the query or filters.")
        log_event("no_results", None, query)
    else:
        log_event("search", None, query)
        for _, r in results.iterrows():
            with st.container(border=True):
                icon = {"pattern": "💡", "component": "🧩", "snippet": "🔧"}.get(str(r.get("type", "")).lower(), "✨")
                st.subheader(f"{icon} {r['title']}")
                st.caption(f"Score: {r['score']} • Type: {r.get('type','')} • Tags: {r.get('tags','')}")
                st.code(str(r["text"]).strip()[:1500])

                c1, c2 = st.columns(2)
                if c1.button("Copy snippet", key=f"copy_{r['item_id']}"):
                    st.session_state["copied"] = r["text"]
                    st.toast("Copied to session (paste from here).")
                    log_event("copy", r["item_id"], query, r["score"])
                if c2.button("Helpful 👍", key=f"helpful_{r['item_id']}"):
                    log_event("helpful", r["item_id"], query, r["score"])
else:
    st.info("Enter a topic and click **Recommend**.")

st.divider()
cA, cB = st.columns([2, 1])
with cA:
    st.caption("Data utilities")
    if RAW_DATA.exists():
        st.download_button("Download current dataset (items.csv)", data=RAW_DATA.read_bytes(), file_name="items.csv", mime="text/csv")
with cB:
    if st.button("Retrain model from items.csv"):
        import subprocess, sys
        st.info("Retraining…")
        res = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True)
        if res.returncode == 0:
            st.success("Retrained. Click Recommend again.")
        else:
            st.error(res.stderr or "Retrain failed—check train.py output.")