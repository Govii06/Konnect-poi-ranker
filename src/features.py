"""
Pairwise (traveler, POI) Feature Engineering
==============================================
This is the shared feature computation used both when constructing
training examples and at inference time - critical for train/serve
consistency.

Feature groups:
  Content-based    : interest_overlap, explicit_tag_score
  Text              : text_similarity (TF-IDF cosine, see src/text_features.py)
  Behavioral        : hist_cat_affinity, hist_tag_affinity, confidence
  Geographic        : distance_km (already computed in candidate_gen), distance_score
  Popularity/quality: rating_norm, popularity_pctile, is_local, is_touristy
  Practical/context : budget_compat, mobility_compat, duration_fit
"""
import numpy as np
import pandas as pd
from src.candidate_gen import INTEREST_TO_TAGS


FEATURE_COLUMNS = [
    "interest_overlap", "explicit_tag_score", "text_similarity",
    "hist_cat_affinity", "hist_tag_affinity",
    "distance_score", "rating_norm", "popularity_pctile",
    "is_local", "is_touristy", "is_family_friendly",
    "budget_compat", "mobility_compat", "duration_fit", "confidence",
]

# Assumed hours per day a traveler actually spends visiting POIs. Used to
# turn `trip_duration` into a time budget.
ACTIVE_MINUTES_PER_DAY = 480

# How aggressively a POI's share of the total trip time is penalized.
DURATION_PENALTY = 4.0


def _explicit_tag_score(poi_row, interests: set) -> float:
    wanted_tags = set()
    for interest in interests:
        wanted_tags.update(INTEREST_TO_TAGS.get(interest, {}).get("tags", []))
    if not wanted_tags:
        return 0.0
    overlap = len(wanted_tags & poi_row.tag_set)
    return overlap / len(wanted_tags)


def _interest_overlap(poi_row, interests: set) -> float:
    wanted_categories = set()
    for interest in interests:
        wanted_categories.update(INTEREST_TO_TAGS.get(interest, {}).get("category", []))
    return 1.0 if poi_row.category in wanted_categories else 0.0


def budget_compatibility(poi_price_level: float, budget_num: int) -> float:
    """Budget as a CEILING, not a target.

    Full compatibility when a POI costs at or below what the traveler is
    willing to spend; the score decays only as the POI goes *over* budget.

    This was previously symmetric (`abs(price - budget)`), which meant a
    high-budget traveler scored 0.00 on the cheapest POIs and the explainer
    reported them as "priced outside budget" - false, and directly opposed
    to the long-tail / local-discovery goal, since small local gems are
    exactly the cheap POIs the retrieval channels work to surface. Someone
    who can afford anything is not badly served by a EUR 8 neighbourhood
    cafe; someone on a low budget genuinely is blocked by a EUR 90 tasting
    menu. Only the second direction is a constraint.
    """
    over = max(0.0, float(poi_price_level) - budget_num)
    return float(np.clip(1.0 - over / 3.0, 0.0, 1.0))


def mobility_compatibility(distance_km: float, mobility: str) -> float:
    caps = {"walking": 6.0, "public_transport": 15.0, "car": 40.0}
    cap = caps.get(mobility, 12.0)
    return float(np.clip(1.0 - distance_km / cap, 0.0, 1.0))


def duration_fit(expected_duration_min: float, trip_duration_days: int) -> float:
    """How comfortably a POI's visit length fits the trip's total time budget.

    A 3-hour boat trip is a reasonable ask on a 10-day trip and an
    expensive one on a 2-day trip. Expressed as the share of total active
    trip time the POI would consume, penalized linearly:

        share = expected_duration / (trip_days * ACTIVE_MINUTES_PER_DAY)
        fit   = clip(1 - share * DURATION_PENALTY, 0, 1)

    This is the only place `trip_duration` enters the model, and it is
    genuinely discriminative because trip length varies across travelers
    in the generated dataset (2-10 nights).
    """
    if not trip_duration_days or trip_duration_days <= 0:
        return 1.0
    if pd.isna(expected_duration_min):
        return 1.0  # unknown duration -> do not penalize
    budget = trip_duration_days * ACTIVE_MINUTES_PER_DAY
    share = float(expected_duration_min) / budget
    return float(np.clip(1.0 - share * DURATION_PENALTY, 0.0, 1.0))


def build_features(candidates: pd.DataFrame, profile: dict) -> pd.DataFrame:
    """candidates must already have `distance_km` (see candidate_gen)."""
    df = candidates.copy()

    df["interest_overlap"] = df.apply(lambda r: _interest_overlap(r, profile["explicit_interests"]), axis=1)
    df["explicit_tag_score"] = df.apply(lambda r: _explicit_tag_score(r, profile["explicit_interests"]), axis=1)

    df["hist_cat_affinity"] = df["category"].map(profile["cat_affinity"]).fillna(0.0)
    # sorted() keeps the summation order stable across processes: Python
    # salts string hashing, so bare set iteration order varies per run.
    df["hist_tag_affinity"] = df["tag_set"].apply(
        lambda tags: np.mean([profile["tag_affinity"].get(t, 0.0) for t in sorted(tags)])
        if tags else 0.0
    )

    # Text similarity is attached upstream (pipeline / trainer) because it
    # needs the catalog-wide TF-IDF model. Absent -> feature abstains at 0.
    if "text_similarity" not in df.columns:
        df["text_similarity"] = 0.0
    df["text_similarity"] = df["text_similarity"].astype(float).fillna(0.0)

    df["distance_score"] = 1.0 / (1.0 + df["distance_km"])  # closer -> higher

    df["is_local"] = df["is_local"].astype(float)
    df["is_touristy"] = df["is_touristy"].astype(float)
    df["is_family_friendly"] = df["is_family_friendly"].astype(float)

    df["budget_compat"] = df["price_level"].apply(lambda p: budget_compatibility(p, profile["budget_num"]))
    df["mobility_compat"] = df["distance_km"].apply(lambda d: mobility_compatibility(d, profile["mobility"]))
    df["duration_fit"] = df["expected_duration"].apply(
        lambda d: duration_fit(d, profile.get("trip_duration", 5))
    )

    # explicit free-text preference nudges (kept simple / rule-based)
    pref = profile["explicit_preferences"]
    if "prefer less touristy" in pref:
        df["is_local"] = df["is_local"] * 1.0 + 0.1
        df.loc[df["is_touristy"] == 1.0, "explicit_tag_score"] -= 0.3
    if "famous landmarks" in pref:
        df.loc[df["is_touristy"] == 1.0, "explicit_tag_score"] += 0.3
    # NOTE: "child-friendly only" is deliberately NOT nudged here. It is a hard
    # party constraint, not a taste, and is applied in the context gate
    # (`context_scoring.party_compatibility`). Applying it in both places would
    # double-count it, and as a preference nudge alone it was too weak - a
    # non-child-friendly POI still reached rank 3 of the Family scenario.

    df["confidence"] = profile["confidence"]

    return df
