"""
Synthetic data generator for the Konnect POI Intelligence & Ranking assignment.

Produces three CSVs in data/:
  - pois.csv          (POI catalog for one destination: Lisbon)
  - travelers.csv      (traveler / trip context, one row per traveler)
  - interactions.csv  (historical traveler-POI interactions)

The generator deliberately injects realistic messiness (missing fields,
duplicate POIs, inconsistent category casing, popularity skew) so that the
data-preparation step in the pipeline has real problems to solve.
"""
import random
import numpy as np
import pandas as pd
from pathlib import Path

random.seed(42)
np.random.seed(42)

OUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# 1. POI catalog
# ---------------------------------------------------------------------------
# Lisbon-ish bounding box, used only to generate plausible lat/lon.
CENTER_LAT, CENTER_LON = 38.7223, -9.1393

CATEGORIES = {
    "Restaurant": ["Local Cuisine", "Fine Dining", "Street Food", "Cafe", "Bakery"],
    "Historical Site": ["Monument", "Castle", "Church", "Ruins"],
    "Museum": ["Art Museum", "History Museum", "Science Museum"],
    "Neighborhood Experience": ["Local Market", "Walking Area", "Artisan Street"],
    "Activity": ["Tour", "Workshop", "Boat Trip"],
    "Nature/Outdoor": ["Park", "Viewpoint", "Garden"],
    "Shopping": ["Boutique", "Market", "Mall"],
    "Event": ["Live Music", "Festival", "Exhibition"],
}
# messy variants that real-world category data tends to have
CATEGORY_CASING_NOISE = {"Restaurant": ["restaurant ", "RESTAURANT", "Restaurant"]}

TAG_POOL = [
    "local", "touristy", "family-friendly", "romantic", "budget", "luxury",
    "outdoor", "indoor", "historic", "modern", "quiet", "lively", "scenic",
    "authentic", "hidden-gem", "instagrammable", "accessible",
]

N_POIS = 320


def sample_price_level(category):
    # Fine dining / luxury shopping skew pricier
    if category in ("Restaurant", "Shopping"):
        return int(np.clip(np.random.choice([1, 2, 3, 4], p=[0.25, 0.35, 0.25, 0.15]), 1, 4))
    return int(np.clip(np.random.choice([1, 2, 3, 4], p=[0.4, 0.35, 0.2, 0.05]), 1, 4))


def make_poi(i):
    category = random.choice(list(CATEGORIES.keys()))
    subcategory = random.choice(CATEGORIES[category])

    # Popularity is intentionally power-law distributed: a small number of
    # "famous landmark" POIs dominate raw popularity, and a long tail of
    # small local spots have very few reviews. This is the popularity-bias
    # problem the assignment explicitly asks us to be aware of.
    review_count = int(np.random.pareto(1.3) * 40) + np.random.randint(0, 5)
    rating = float(np.clip(np.random.normal(4.2, 0.5), 2.0, 5.0))
    # famous landmarks: high popularity AND high "touristy" tag prevalence
    is_famous = review_count > 300
    tags = set(random.sample(TAG_POOL, k=random.randint(2, 5)))
    if is_famous:
        tags.add("touristy")
    else:
        # local/small POIs are more likely tagged local/hidden-gem
        if random.random() < 0.5:
            tags.add("local")
        if random.random() < 0.25:
            tags.add("hidden-gem")

    lat = CENTER_LAT + np.random.normal(0, 0.03)
    lon = CENTER_LON + np.random.normal(0, 0.04)

    price_level = sample_price_level(category)
    duration = random.choice([30, 45, 60, 90, 120, 150, 180])

    open_hour = random.choice([7, 8, 9, 10])
    close_hour = random.choice([17, 18, 20, 22, 23])

    row = {
        "poi_id": f"P{i:04d}",
        "name": f"{subcategory} #{i}",
        "category": category,
        "subcategory": subcategory,
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        # sorted(): `tags` is a set, and bare set iteration order varies per
        # process (salted string hashing), which would make the generated
        # descriptions - and therefore the CSVs - non-reproducible.
        "description": f"A {subcategory.lower()} known for {', '.join(sorted(tags)[:2])} vibes.",
        "tags": "|".join(sorted(tags)),
        "price_level": price_level,
        "rating": round(rating, 1),
        "review_count": review_count,
        # Raw platform-style popularity indicator (0-100), as listed in the
        # assignment's POI schema. Correlated with review_count but noisy, the
        # way a vendor-supplied popularity score usually is. The pipeline
        # deliberately does NOT rank on this directly - see data_prep.
        "popularity": int(np.clip(np.log1p(review_count) / np.log1p(600) * 100
                                  + np.random.normal(0, 4), 0, 100)),
        "expected_duration": duration,
        "opening_hour": open_hour,
        "closing_hour": close_hour,
        "reservation_required": random.random() < (0.4 if category == "Restaurant" else 0.05),
        "accessibility": random.random() < 0.7,
    }
    return row


def inject_messiness(df):
    df = df.copy()
    # 1. Missing values
    for col in ["rating", "review_count", "description", "opening_hour", "closing_hour"]:
        idx = df.sample(frac=0.04, random_state=1).index
        df.loc[idx, col] = np.nan

    # 2. Inconsistent category casing for a subset of restaurants
    idx = df[df.category == "Restaurant"].sample(frac=0.15, random_state=2).index
    df.loc[idx, "category"] = df.loc[idx, "category"].apply(
        lambda c: random.choice(CATEGORY_CASING_NOISE["Restaurant"])
    )

    # 3. Duplicate POIs (same place, slightly different id/name casing) -
    #    simulates duplicate ingestion from multiple data sources.
    dupes = df.sample(n=8, random_state=3).copy()
    dupes["poi_id"] = dupes["poi_id"] + "_DUP"
    dupes["name"] = dupes["name"].str.upper()
    df = pd.concat([df, dupes], ignore_index=True)

    return df


pois = pd.DataFrame([make_poi(i) for i in range(N_POIS)])
pois = inject_messiness(pois)
pois.to_csv(OUT_DIR / "pois.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Travelers / trips
# ---------------------------------------------------------------------------
INTEREST_POOL = [
    "local food", "history", "architecture", "museums", "nightlife", "nature",
    "shopping", "art", "family activities", "photography", "beaches",
    "local experiences", "neighborhoods", "adventure", "relaxation",
]
BUDGETS = ["low", "medium", "high"]
PARTY_TYPES = ["solo", "couple", "family", "friends"]
MOBILITY = ["walking", "public_transport", "car"]

N_TRAVELERS = 60


def make_traveler(i):
    n_interests = random.randint(2, 4)
    interests = random.sample(INTEREST_POOL, k=n_interests)

    # Preference is conditioned on party type. Drawing them independently
    # produced incoherent travelers (a solo traveler stating "child-friendly
    # only"), which makes the demo output look careless and muddies the
    # party-constraint logic in context_scoring.
    party_type = random.choice(PARTY_TYPES)
    if party_type == "family":
        explicit_pref = random.choice(
            ["child-friendly only", "child-friendly only",
             "famous landmarks are acceptable", "no preference"]
        )
    else:
        explicit_pref = random.choice(
            ["prefer less touristy experiences", "famous landmarks are acceptable",
             "no preference", "quiet and scenic places"]
        )
    # Trip length varies (2-10 nights). This matters: a traveler with a
    # 2-day trip cannot absorb the same set of long-duration POIs as one
    # with 10 days, so `trip_duration` is a real ranking signal rather
    # than a constant column.
    trip_duration = int(np.random.choice([2, 3, 4, 5, 6, 7, 10],
                                         p=[0.12, 0.18, 0.20, 0.22, 0.12, 0.10, 0.06]))
    start = pd.Timestamp("2026-09-15") + pd.Timedelta(days=random.randint(0, 60))
    end = start + pd.Timedelta(days=trip_duration)
    return {
        "traveler_id": f"U{i:03d}",
        "destination": "Lisbon",
        "travel_start": start.strftime("%Y-%m-%d"),
        "travel_end": end.strftime("%Y-%m-%d"),
        "trip_duration": trip_duration,
        "interests": "|".join(interests),
        "budget": random.choice(BUDGETS),
        "party_type": party_type,
        "mobility": random.choice(MOBILITY),
        "explicit_preferences": explicit_pref,
    }


travelers = pd.DataFrame([make_traveler(i) for i in range(N_TRAVELERS)])
travelers.to_csv(OUT_DIR / "travelers.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Historical interactions
# ---------------------------------------------------------------------------
INTERACTION_TYPES = ["view", "click", "save", "share", "navigate", "visit", "booking", "dismiss"]
# rough probability of each type occurring (views dominate, bookings are rare)
INTERACTION_PROBS = [0.35, 0.20, 0.15, 0.03, 0.10, 0.09, 0.03, 0.05]

real_pois = pois[~pois.poi_id.str.endswith("_DUP")].reset_index(drop=True)


def traveler_affinity_score(traveler_row, poi_row):
    """Ground-truth affinity used only to bias *synthetic* interaction
    sampling so the dataset has learnable signal. The model never sees this
    function directly - it only sees the resulting interactions."""
    interests = set(traveler_row.interests.split("|"))
    tags = set(str(poi_row.tags).split("|"))
    cat_words = {poi_row.category.lower(), poi_row.subcategory.lower()}

    score = 0.0
    interest_map = {
        "local food": {"restaurant", "local"},
        "history": {"historical site", "monument", "castle", "historic"},
        "architecture": {"historical site", "church", "historic"},
        "museums": {"museum"},
        "neighborhoods": {"neighborhood experience", "local"},
        "local experiences": {"neighborhood experience", "local"},
        "family activities": {"family-friendly", "activity"},
        "nature": {"nature/outdoor", "outdoor", "scenic"},
        "shopping": {"shopping"},
        "art": {"museum", "art museum"},
    }
    for interest in interests:
        keywords = interest_map.get(interest, set())
        if keywords & (tags | cat_words | {poi_row.category.lower()}):
            score += 1.0

    if "prefer less touristy" in traveler_row.explicit_preferences and "local" in tags:
        score += 0.8
    if "prefer less touristy" in traveler_row.explicit_preferences and "touristy" in tags:
        score -= 0.8
    if "famous landmarks" in traveler_row.explicit_preferences and "touristy" in tags:
        score += 0.8
    if "child-friendly" in traveler_row.explicit_preferences and "family-friendly" in tags:
        score += 1.0

    budget_map = {"low": 1, "medium": 2, "high": 4}
    score -= abs(budget_map[traveler_row.budget] - poi_row.price_level) * 0.15

    return score


rows = []
for _, trav in travelers.iterrows():
    scores = real_pois.apply(lambda p: traveler_affinity_score(trav, p), axis=1)
    probs = np.exp(scores) / np.exp(scores).sum()  # softmax over affinity
    n_interactions = random.randint(8, 25)
    chosen_idx = np.random.choice(real_pois.index, size=n_interactions, replace=False, p=probs)
    for idx in chosen_idx:
        poi = real_pois.loc[idx]
        aff = scores[idx]
        # higher affinity -> more likely to have a "deep" interaction
        if aff > 1.5:
            itype = np.random.choice(["visit", "booking", "save", "share"], p=[0.4, 0.2, 0.3, 0.1])
        elif aff > 0.5:
            itype = np.random.choice(["save", "visit", "click", "view"], p=[0.3, 0.2, 0.25, 0.25])
        elif aff < -0.5:
            itype = np.random.choice(["dismiss", "view"], p=[0.6, 0.4])
        else:
            itype = np.random.choice(INTERACTION_TYPES, p=INTERACTION_PROBS)
        rows.append({
            "traveler_id": trav.traveler_id,
            "poi_id": poi.poi_id,
            "interaction": itype,
            "timestamp": pd.Timestamp("2026-06-01") + pd.Timedelta(days=random.randint(0, 90)),
        })

interactions = pd.DataFrame(rows)
interactions.to_csv(OUT_DIR / "interactions.csv", index=False)

print(f"pois.csv: {len(pois)} rows ({len(real_pois)} unique + {len(pois)-len(real_pois)} injected dupes)")
print(f"travelers.csv: {len(travelers)} rows")
print(f"interactions.csv: {len(interactions)} rows")
