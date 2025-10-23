from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import joblib

DATA = Path("data/items.csv")

df = pd.read_csv(DATA)
# build a simple text field the model reads
df["bag"] = (df["title"].fillna("") + " "
             + df["type"].fillna("") + " "
             + df["tags"].fillna("") + " "
             + df["text"].fillna(""))

# train TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["bag"].values)

# save artifacts
joblib.dump(vectorizer, "tfidf_model.joblib")
sparse.save_npz("items_matrix.npz", X)
df.to_csv("items_with_bag.csv", index=False)

print("Model trained. Files created: tfidf_model.joblib, items_matrix.npz, items_with_bag.csv")