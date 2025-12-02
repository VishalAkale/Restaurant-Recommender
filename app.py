from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re
import json

app = Flask(__name__)

# ---------- CONFIG ----------
DATA_PATH = r"C:\Users\Vishal\Desktop\Intership\Task_2\Dataset.csv"
TOP_K = 10

# Hybrid weights (tweak these if you want)
W_CUISINE = 0.45
W_RATING = 0.20
W_VOTES = 0.15
W_COST = 0.10
W_CITY = 0.10

THEME = {
    "electric_blue": "#3D5AFE",
    "cyan": "#18FFFF",
    "rich_black": "#02040F",
    "white_smoke": "#F5F5F5"
}

# ---------- TEXT CLEAN ----------
def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9, ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------- LOAD CLEAN SUGGESTION FILE ----------
with open("suggestions.json", "r", encoding="utf-8") as f:
    sug = json.load(f)

cities_list = sug["cities"]
cuisines_list = sug["cuisines"]

# ---------- LOAD DATA ----------
print("\nLoading dataset from", DATA_PATH)
df = pd.read_csv(DATA_PATH, encoding="latin1")
df = df.rename(columns=lambda x: x.strip())

# Keep required columns only
cols_needed = [
    'Restaurant Name','City','Cuisines',
    'Average Cost for two','Aggregate rating','Votes',
    'Longitude','Latitude','Locality'
]

available_cols = [c for c in cols_needed if c in df.columns]
df = df[available_cols].copy()

# Clean cuisine & city for matching (internal use only)
df['cuisines_clean'] = df['Cuisines'].fillna("").apply(clean_text)
df['city_clean'] = df['City'].astype(str).str.lower().str.strip()

df['cost_for_two'] = pd.to_numeric(df.get('Average Cost for two', 0), errors='coerce').fillna(0).astype(int)
df['rating'] = pd.to_numeric(df.get('Aggregate rating', 0), errors='coerce').fillna(0.0)
df['Votes'] = pd.to_numeric(df.get('Votes', 0), errors='coerce').fillna(0).astype(int)

# ---------- TF-IDF ----------
tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
tfidf_matrix = tfidf.fit_transform(df['cuisines_clean'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Use DataFrame index as Restaurant ID
indices = pd.Series(df.index, index=df.index)

print(f"Loaded {len(df)} restaurants.")

# ---------- HYBRID SCORING HELPERS ----------
def normalize_rating(r):
    """Normalize rating (0-5) to 0-1"""
    try:
        r = float(r)
    except:
        r = 0.0
    return max(0.0, min(1.0, r / 5.0))

def normalize_votes(v, max_votes):
    """Log-normalize votes to 0-1"""
    try:
        v = float(v)
    except:
        v = 0.0
    # use log1p to handle zero votes
    return np.log1p(v) / np.log1p(max_votes) if max_votes > 0 else 0.0

def cost_similarity(user_max_cost, item_cost):
    """Return similarity in [0,1] between user budget and item cost"""
    if user_max_cost is None:
        return 1.0
    try:
        user_max_cost = float(user_max_cost)
    except:
        return 1.0
    item_cost = float(item_cost) if not pd.isna(item_cost) else 0.0
    # if item cost is zero, treat as full similarity
    if user_max_cost <= 0:
        return 1.0
    diff = abs(item_cost - user_max_cost)
    denom = max(user_max_cost, item_cost, 1.0)
    sim = max(0.0, 1.0 - (diff / denom))
    return sim

def city_match_score(user_city, item_city):
    """Exact match 1.0, else 0.0. Keeps city preference strong."""
    if not user_city:
        return 1.0
    try:
        return 1.0 if str(user_city).lower().strip() == str(item_city).lower().strip() else 0.0
    except:
        return 0.0

# Precompute max votes for normalization
MAX_VOTES = int(df['Votes'].max()) if 'Votes' in df.columns else 0
if MAX_VOTES < 1:
    MAX_VOTES = 1

# ---------- RECOMMENDATION ENGINE (HYBRID MODEL) ----------
def recommend_restaurants(city=None, cuisine_input=None, max_cost=None, min_rating=0.0, top_k=TOP_K):

    # Filter by city if provided
    filtered_idx = df.index
    if city:
        city_q = city.lower().strip()
        filtered_idx = df[df['city_clean'] == city_q].index

    # fallback to whole dataset if city filter returns none
    if len(filtered_idx) == 0:
        filtered_idx = df.index

    # cuisine similarity scores (array aligned with df index)
    if cuisine_input and str(cuisine_input).strip():
        q = clean_text(cuisine_input)
        q_vec = tfidf.transform([q])
        sim_scores = linear_kernel(q_vec, tfidf_matrix).flatten()
    else:
        sim_scores = np.ones(len(df))  # neutral similarity

    candidates = []
    for idx in filtered_idx:
        r = df.loc[idx]

        # apply rating filter early
        if float(r.get('rating', 0.0)) < float(min_rating):
            continue

        # compute component scores
        cuisine_sim = float(sim_scores[idx]) if idx < len(sim_scores) else 0.0
        rating_score = normalize_rating(r.get('rating', 0.0))
        votes_score = normalize_votes(r.get('Votes', 0), MAX_VOTES)
        cost_score = cost_similarity(max_cost, r.get('cost_for_two', 0))
        city_score = city_match_score(city, r.get('City', ''))

        # combine weighted
        final_score = (
            W_CUISINE * cuisine_sim +
            W_RATING * rating_score +
            W_VOTES * votes_score +
            W_COST * cost_score +
            W_CITY * city_score
        )

        candidates.append((idx, final_score, cuisine_sim, rating_score, votes_score))

    # sort by final score, then by rating as tie-breaker, then votes
    candidates_sorted = sorted(candidates, key=lambda x: (x[1], df.loc[x[0]].get('rating', 0), df.loc[x[0]].get('Votes', 0)), reverse=True)

    top = candidates_sorted[:top_k]

    results = []
    for idx, score, cuisine_sim, rating_score, votes_score in top:
        r = df.loc[idx]
        results.append({
            "restaurant_id": int(idx),
            "name": r.get('Restaurant Name','-'),
            "city": r.get('City','-'),
            "locality": r.get('Locality','-'),
            "cuisines": r.get('Cuisines','-'),
            "cost_for_two": int(r.get('cost_for_two',0)),
            "rating": float(r.get('rating',0)),
            "votes": int(r.get('Votes',0)),
            "score": float(score)
        })

    return results

# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html", theme=THEME, cities=cities_list)


@app.route("/suggest_cuisines")
def suggest_cuisines():
    q = request.args.get("q", "").strip().lower()
    if not q:
        return jsonify(cuisines_list[:20])
    return jsonify([c for c in cuisines_list if q in c.lower()][:20])


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.form or request.json or {}

    city = data.get("city", "")
    cuisine = data.get("cuisine", "")
    max_cost = data.get("max_cost") or None
    min_rating = float(data.get("min_rating") or 0)
    top_k = int(data.get("top_k") or TOP_K)

    recs = recommend_restaurants(
        city=city,
        cuisine_input=cuisine,
        max_cost=max_cost,
        min_rating=min_rating,
        top_k=top_k
    )

    if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify(recs)

    return render_template("result.html", results=recs, theme=THEME, query={
        "city": city,
        "cuisine": cuisine,
        "max_cost": max_cost,
        "min_rating": min_rating
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
