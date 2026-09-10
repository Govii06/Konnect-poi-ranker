"""
Personalized Ranking Model
============================
Approach: pointwise learning-to-rank via gradient-boosted regression.

Why this approach (see docs/TECHNICAL_DOC.md section 5 for the full
rationale): the assignment explicitly favors "a simpler model with
thoughtful features ... over unnecessary model complexity". A pairwise or
listwise ranker (e.g. LambdaMART) would be the natural next step in
production, but a pointwise regressor over well-designed features is:
  - simple to train, debug, and explain to a downstream consumer
  - naturally produces a *calibrated-ish* score in a bounded range that
    can be interpreted as "relevance", not just a rank order
  - cheap to retrain frequently as new interaction data arrives

Learning target
----------------
For every (traveler, poi) pair with at least one interaction, we convert
the interaction into a regression label using `interaction_label()`
(see traveler_profile.py), then min-max squash the *sum* of that
traveler-POI's interaction weights into [0, 1]. Pairs the traveler never
interacted with are held out for negative sampling (see below).

Training data construction
----------------------------
1. Positive/negative-labeled pairs come from historical interactions
   (label = squashed interaction weight, in [0, 1], can be < 0.5 for
   dismiss-heavy pairs).
2. For every traveler we also sample random *non-interacted* POIs as
   implicit negatives (label = 0), at a 1:2 positive:negative ratio, so
   the model learns what a traveler does *not* prefer, not just what
   they've seen.
3. Features for every pair are computed with the same `build_features()`
   used at inference time (train/serve consistency).
4. **Leave-one-out** feature construction for positive pairs: a pair's
   behavioral features are computed from the traveler's history with
   that POI removed (`profile_excluding_poi`). Otherwise the features
   are derived from the same interactions as the label, and the model
   simply reads the answer off `hist_cat_affinity` instead of learning
   preference. Fixing this raised held-out Precision@10 substantially
   while dropping those features from ~75% of importance to a sane
   share - see docs/TECHNICAL_DOC.md section 6.

Inference
----------
For a new traveler/candidate-set, we compute the same features and call
`model.predict()`, then clip to [0, 1] to get a `preference_score`.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.features import build_features, FEATURE_COLUMNS
from src.candidate_gen import haversine_km
from src.traveler_profile import interaction_label, profile_excluding_poi


def _pair_label(weights: list) -> float:
    total = sum(weights)
    # squash roughly into [0,1]; a single "booking" (6) maps near 1.0,
    # a single "dismiss" (-3) maps near 0.0, mixed/no signal sits mid-range
    return float(np.clip((total + 6) / 18.0, 0.0, 1.0))


def _anchor(profile: dict, center_lat: float, center_lon: float):
    """Trip anchor with an explicit None check.

    `profile["trip_anchor_lat"] or center_lat` would also fall back on a
    legitimate 0.0 coordinate (the equator / prime meridian).
    """
    lat = profile["trip_anchor_lat"]
    lon = profile["trip_anchor_lon"]
    return (center_lat if lat is None else lat,
            center_lon if lon is None else lon)


def _pair_features(poi_row, profile: dict, center_lat: float, center_lon: float,
                    text_model=None) -> pd.DataFrame:
    """Build the one-row feature frame for a single (traveler, POI) pair,
    using exactly the same `build_features()` the serving path uses."""
    anchor_lat, anchor_lon = _anchor(profile, center_lat, center_lon)
    cand_df = pd.DataFrame([poi_row])
    # `poi_lookup` is indexed by poi_id, so it arrives as the Series name
    # rather than a column - restore it, the text model looks POIs up by id.
    cand_df["poi_id"] = poi_row.name
    cand_df["distance_km"] = haversine_km(anchor_lat, anchor_lon,
                                          poi_row.latitude, poi_row.longitude)
    if text_model is not None:
        cand_df["text_similarity"] = text_model.similarity(cand_df["poi_id"], profile)
    return build_features(cand_df, profile)


def build_training_set(pois: pd.DataFrame, profiles: dict, interactions: pd.DataFrame,
                        center_lat: float, center_lon: float,
                        neg_per_pos: int = 2, seed: int = 42,
                        text_model=None) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    poi_lookup = pois.set_index("poi_id")
    rows = []

    grouped = interactions.groupby(["traveler_id", "poi_id"])["interaction"].apply(list)

    for (tid, poi_id), interaction_list in grouped.items():
        if tid not in profiles or poi_id not in poi_lookup.index:
            continue
        profile = profiles[tid]
        weights = [interaction_label(i) for i in interaction_list]
        label = _pair_label(weights)

        # --- LEAVE-ONE-OUT (target leakage fix) -----------------------
        # The label for this pair is derived from this traveler's
        # interactions with this POI. Those same interactions also feed
        # `hist_cat_affinity` / `hist_tag_affinity` and the geographic
        # anchor. Training on that combination teaches the model to read
        # the answer off the features, so it scores high importance in
        # training and generalizes badly at serving time, where a
        # candidate POI has contributed nothing to the profile.
        train_profile = profile_excluding_poi(profile, poi_id)

        poi_row = poi_lookup.loc[poi_id]
        feat = _pair_features(poi_row, train_profile, center_lat, center_lon, text_model)
        feat["label"] = label
        feat["traveler_id"] = tid
        feat["poi_id"] = poi_id
        rows.append(feat)

    pos_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # --- implicit negative sampling: random non-interacted POIs per traveler
    neg_rows = []
    interacted_pairs = set(zip(interactions.traveler_id, interactions.poi_id))
    all_poi_ids = pois.poi_id.values
    for tid, profile in profiles.items():
        n_pos = (pos_df.traveler_id == tid).sum() if len(pos_df) else 0
        n_neg = max(3, n_pos * neg_per_pos)
        candidates = rng.choice(all_poi_ids, size=min(n_neg * 3, len(all_poi_ids)), replace=False)
        picked = 0
        for poi_id in candidates:
            if picked >= n_neg:
                break
            if (tid, poi_id) in interacted_pairs or poi_id not in poi_lookup.index:
                continue
            # No leave-one-out needed here: by construction the traveler
            # never interacted with this POI, so it contributed nothing to
            # the profile in the first place.
            poi_row = poi_lookup.loc[poi_id]
            feat = _pair_features(poi_row, profile, center_lat, center_lon, text_model)
            feat["label"] = 0.0
            feat["traveler_id"] = tid
            feat["poi_id"] = poi_id
            neg_rows.append(feat)
            picked += 1

    neg_df = pd.concat(neg_rows, ignore_index=True) if neg_rows else pd.DataFrame()
    training_set = pd.concat([pos_df, neg_df], ignore_index=True)
    return training_set


class POIRanker:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42
        )
        self.is_fitted = False

    def fit(self, training_set: pd.DataFrame):
        X = training_set[FEATURE_COLUMNS]
        y = training_set["label"]
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, feature_df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Ranker must be fit before predict()")
        X = feature_df[FEATURE_COLUMNS]
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, 1.0)

    def feature_importances(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
