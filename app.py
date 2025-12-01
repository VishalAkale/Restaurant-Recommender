from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re

app = Flask(__name__)

# ---------- CONFIG ----------
DATA_PATH = r"C:\Users\Vishal\Desktop\Intership\Task_2\Dataset.csv"  # <-- dataset path (your uploaded file)
TOP_K = 10
# Theme-safe defaults (used only for returning CSS vars if needed)
THEME = {
    "electric_blue": "#3D5AFE",
    "cyan": "#18FFFF",
    "rich_black": "#02040F",
    "white_smoke": "#F5F5F5"
}

# ---------- LOAD & PREPROCESS ----------
def clean_text(s):
    if pd.isna(s):
        return ""
    # lower, remove strange chars but keep commas as separators
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9, ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

print("Loading dataset from", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Keep relevant columns and drop duplicates (if any)
df = df.rename(columns=str.strip)
cols_needed = ['Restaurant ID','Restaurant Name','City','Cuisines',
               'Average Cost for two','Aggregate rating','Votes','Longitude','Latitude','Locality']
available_cols = [c for c in cols_needed if c in df.columns]
df = df[available_cols].copy()

# Fill missing cuisines
df['Cuisines'] = df.get('Cuisines', "").fillna("")
# Normalize text fields
df['cuisines_clean'] = df['Cuisines'].apply(clean_text)
df['city_clean'] = df['City'].astype(str).str.lower().str.strip()

# Create cost buckets (optional)
df['cost_for_two'] = pd.to_numeric(df.get('Average Cost for two', 0), errors='coerce').fillna(0).astype(int)
df['rating'] = pd.to_numeric(df.get('Aggregate rating', 0), errors='coerce').fillna(0.0)

# Precompute TF-IDF on cuisines
tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")  # simple tokenizer
tfidf_matrix = tfidf.fit_transform(df['cuisines_clean'])

# Precompute cosine similarity matrix (can be memory heavy for large datasets)
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)  # dense-ish but okay for ~9.5k rows

# Map index
indices = pd.Series(df.index, index=df['Restaurant ID']).drop_duplicates()

# Prepare unique cuisine suggestions (for autosuggest)
all_cuisines = set()
for row in df['Cuisines'].dropna().astype(str):
    # cuisines may be comma separated
    parts = [p.strip() for p in row.split(',') if p.strip()]
    for p in parts:
        all_cuisines.add(p)
cuisines_list = sorted(all_cuisines, key=lambda x: x.lower())

# Prepare unique cities
cities_list = sorted(df['City'].dropna().unique(), key=lambda x: x.lower())

print(f"Loaded {len(df)} restaurants, {len(cuisines_list)} cuisines, {len(cities_list)} cities.")


# ---------- RECOMMENDATION LOGIC ----------
def recommend_restaurants(city=None, cuisine_input=None, max_cost=None, min_rating=0.0, top_k=TOP_K):
    # Start with filtering by city (if provided)
    filtered_idx = df.index
    if city:
        city_q = str(city).lower().strip()
        filtered_idx = df[df['city_clean'] == city_q].index

    # If no results in city filter, fallback to worldwide (but we will keep as fallback)
    if len(filtered_idx) == 0 and city:
        filtered_idx = df.index

    # If cuisine_input provided, find similarity scores
    if cuisine_input and cuisine_input.strip():
        q = clean_text(cuisine_input)
        # compute tfidf vector for query (using same vectorizer)
        q_vec = tfidf.transform([q])
        sim_scores = linear_kernel(q_vec, tfidf_matrix).flatten()

        # consider only filtered index
        sim_scores_masked = [(i, sim_scores[i]) for i in filtered_idx]
        # apply cost and rating filters next
    else:
        # if no cuisine, take baseline scores = 1 for all filtered items
        sim_scores_masked = [(i, 1.0) for i in filtered_idx]

    # Apply cost and rating filters and collect candidates
    candidates = []
    for idx, score in sim_scores_masked:
        row = df.loc[idx]
        # cost filter
        if max_cost is not None:
            try:
                if row['cost_for_two'] > int(max_cost):
                    continue
            except:
                pass
        # rating filter
        try:
            if float(row['rating']) < float(min_rating):
                continue
        except:
            pass
        candidates.append((idx, score, row['rating']))

    # sort by: score (desc) then rating (desc) then votes (desc)
    candidates_sorted = sorted(candidates, key=lambda x: (x[1], x[2]), reverse=True)

    # top k
    top = candidates_sorted[:top_k]
    results = []
    for idx, score, rating in top:
        r = df.loc[idx]
        results.append({
            "restaurant_id": int(r.get('Restaurant ID', idx)),
            "name": r.get('Restaurant Name','-'),
            "city": r.get('City','-'),
            "locality": r.get('Locality','-'),
            "cuisines": r.get('Cuisines','-'),
            "cost_for_two": int(r.get('cost_for_two',0)),
            "rating": float(r.get('rating',0.0)),
            "longitude": float(r.get('Longitude', np.nan)) if 'Longitude' in r else None,
            "latitude": float(r.get('Latitude', np.nan)) if 'Latitude' in r else None,
            "score": float(score)
        })
    return results

# ---------- FLASK ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html", theme=THEME, cities=cities_list)

@app.route("/suggest_cuisines")
def suggest_cuisines():
    q = request.args.get('q', '').strip().lower()
    if not q:
        # return first 20
        matches = cuisines_list[:20]
    else:
        matches = [c for c in cuisines_list if q in c.lower()][:20]
    return jsonify(matches)

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.form or request.json or {}
    city = data.get('city') or ""
    cuisine = data.get('cuisine') or ""
    max_cost = data.get('max_cost') or None
    min_rating = float(data.get('min_rating') or 0.0)
    top_k = int(data.get('top_k') or TOP_K)

    recs = recommend_restaurants(city=city, cuisine_input=cuisine,
                                 max_cost=max_cost, min_rating=min_rating, top_k=top_k)
    # If AJAX request -> return JSON
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
        return jsonify(recs)
    # Else render result page
    return render_template("result.html", results=recs, theme=THEME, query={
        "city": city, "cuisine": cuisine, "max_cost": max_cost, "min_rating": min_rating
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
