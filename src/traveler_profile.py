"""
Traveler Representation
========================
Builds a per-traveler profile that fuses:
  - Explicit signals: stated interests, budget, mobility, party type,
    free-text preference string, and trip context (destination, travel
    dates, trip duration).
  - Implicit signals: derived from historical interactions (which
    categories/tags they actually engaged with, and how strongly).

Explicit and implicit signals are combined with a confidence-weighted
blend: the more historical interaction data we have for a traveler, the
more weight implicit behavior gets relative to their stated interests.
This mirrors how real preference tends to work - people say what they
*think* they want, but behavior is often a stronger, corrective signal
(and the two usually agree, which increases confidence).

Leave-one-out support
----------------------
Implicit affinities are stored in BOTH raw (unnormalized sums) and
normalized form, alongside a per-POI record of exactly what each
interacted POI contributed. That makes `profile_excluding_poi()` cheap
and exact, which is what the trainer uses to avoid target leakage: when
building the training row for (traveler T, POI P), T's affinity features
must not be derived from T's interactions with P, because those are the
very interactions the label encodes. See `src/ranker.py`.
"""
import numpy as np
import pandas as pd

INTERACTION_WEIGHTS = {
    "view": 1.0,
    "click": 2.0,
    "save": 3.0,
    "share": 3.0,
    "navigate": 4.0,
    "visit": 5.0,
    "booking": 6.0,
    "dismiss": -3.0,
}

BUDGET_TO_NUM = {"low": 1, "medium": 2, "high": 4}

# Interactions implying the traveler actually committed to a place; used
# to derive the geographic anchor of the trip.
ANCHOR_INTERACTIONS = ("visit", "save", "booking")

# Interaction count at which we fully trust the implicit signal.
CONFIDENCE_SATURATION = 20.0


def interaction_label(interaction_type: str) -> float:
    """Convert a raw interaction type into a learning-signal weight.
    Positive weights = attraction toward the POI, negative = repulsion.
    """
    return INTERACTION_WEIGHTS.get(interaction_type, 0.0)


def _normalize(d: dict) -> dict:
    """Scale a raw affinity dict into roughly [-1, 1]."""
    if not d:
        return {}
    m = max(abs(v) for v in d.values()) or 1.0
    return {k: v / m for k, v in d.items()}


def _confidence(n_interactions: int) -> float:
    return float(np.clip(n_interactions / CONFIDENCE_SATURATION, 0.0, 1.0))


def _mean_anchor(points: list):
    """points: list of (poi_id, lat, lon) -> (lat, lon) or (None, None)."""
    if not points:
        return None, None
    return (float(np.mean([p[1] for p in points])),
            float(np.mean([p[2] for p in points])))


def build_traveler_profiles(travelers: pd.DataFrame, pois: pd.DataFrame,
                             interactions: pd.DataFrame) -> dict:
    """Returns {traveler_id: profile_dict} for every traveler in `travelers`,
    including travelers with zero interaction history (cold start)."""
    poi_lookup = pois.set_index("poi_id")
    profiles = {}

    inter_by_traveler = {tid: g for tid, g in interactions.groupby("traveler_id")}

    for _, trav in travelers.iterrows():
        tid = trav.traveler_id
        explicit_interests = set(trav.interests.split("|")) if pd.notna(trav.interests) else set()

        hist = inter_by_traveler.get(tid, pd.DataFrame(columns=interactions.columns))
        n_interactions = len(hist)

        # --- implicit signal: weighted category/tag affinity from history ---
        # `poi_contributions` records each POI's exact contribution so it
        # can be subtracted again for leave-one-out feature construction.
        cat_raw, tag_raw = {}, {}
        poi_contributions = {}
        anchor_points = []

        for _, row in hist.iterrows():
            poi_id = row.poi_id
            if poi_id not in poi_lookup.index:
                continue
            poi = poi_lookup.loc[poi_id]
            w = interaction_label(row.interaction)
            tags = tuple(sorted(poi.tag_set))

            cat_raw[poi.category] = cat_raw.get(poi.category, 0.0) + w
            for tag in tags:
                tag_raw[tag] = tag_raw.get(tag, 0.0) + w

            contrib = poi_contributions.setdefault(
                poi_id,
                {"weight": 0.0, "count": 0, "category": poi.category, "tags": tags},
            )
            contrib["weight"] += w
            contrib["count"] += 1

            if row.interaction in ANCHOR_INTERACTIONS:
                anchor_points.append((poi_id, float(poi.latitude), float(poi.longitude)))

        anchor_lat, anchor_lon = _mean_anchor(anchor_points)
        trip_duration = int(trav.trip_duration) if pd.notna(trav.trip_duration) else 5

        profiles[tid] = {
            "traveler_id": tid,

            # --- explicit: stated preferences ---
            "explicit_interests": explicit_interests,
            "budget_num": BUDGET_TO_NUM.get(trav.budget, 2),
            "party_type": trav.party_type,
            "mobility": trav.mobility,
            "explicit_preferences": trav.explicit_preferences,

            # --- explicit: trip context ---
            "destination": trav.destination,
            "travel_start": trav.travel_start,
            "travel_end": trav.travel_end,
            "trip_duration": trip_duration,

            # --- implicit: normalized (read by the feature builder) ---
            "cat_affinity": _normalize(cat_raw),
            "tag_affinity": _normalize(tag_raw),
            # --- implicit: raw (read by leave-one-out arithmetic) ---
            "cat_affinity_raw": cat_raw,
            "tag_affinity_raw": tag_raw,
            "poi_contributions": poi_contributions,
            "anchor_points": anchor_points,

            "n_interactions": n_interactions,
            "confidence": _confidence(n_interactions),
            "trip_anchor_lat": anchor_lat,      # geographic anchor from history
            "trip_anchor_lon": anchor_lon,
            "is_cold_start": n_interactions == 0,
        }

    return profiles


def profile_excluding_poi(profile: dict, poi_id: str) -> dict:
    """Return a copy of `profile` with `poi_id`'s contribution removed from
    the POI-specific implicit signals: category affinity, tag affinity, and
    the geographic trip anchor.

    This is the leave-one-out view used when constructing the training row
    for (traveler, poi_id). Without it, `hist_cat_affinity` /
    `hist_tag_affinity` / `distance_score` are computed from the same
    interactions that produced the label - textbook target leakage. It
    measurably cost accuracy here; see docs/TECHNICAL_DOC.md section 6.

    Deliberately NOT adjusted: `n_interactions`, `confidence`,
    `is_cold_start`. These are traveler-level properties that say how much
    history this person has overall - they encode nothing about this
    particular POI, so removing one POI from the count leaks nothing.
    Decrementing them would actively harm the model: positive pairs (which
    go through this function) would carry systematically lower confidence
    than sampled negatives (which do not), handing the trainer a spurious
    "low confidence => high label" shortcut that has nothing to do with
    preference. An earlier version of this fix did exactly that and drove
    `confidence` to >50% of feature importance.
    """
    contrib = profile["poi_contributions"].get(poi_id)
    if contrib is None:
        # traveler never interacted with this POI: nothing to remove
        return profile

    cat_raw = dict(profile["cat_affinity_raw"])
    tag_raw = dict(profile["tag_affinity_raw"])
    w = contrib["weight"]

    category = contrib["category"]
    cat_raw[category] = cat_raw.get(category, 0.0) - w
    if abs(cat_raw[category]) < 1e-12:
        cat_raw.pop(category, None)

    for tag in contrib["tags"]:
        tag_raw[tag] = tag_raw.get(tag, 0.0) - w
        if abs(tag_raw[tag]) < 1e-12:
            tag_raw.pop(tag, None)

    anchor_points = [p for p in profile["anchor_points"] if p[0] != poi_id]
    anchor_lat, anchor_lon = _mean_anchor(anchor_points)

    loo = dict(profile)
    loo.update({
        "cat_affinity": _normalize(cat_raw),
        "tag_affinity": _normalize(tag_raw),
        "cat_affinity_raw": cat_raw,
        "tag_affinity_raw": tag_raw,
        "poi_contributions": {k: v for k, v in profile["poi_contributions"].items()
                              if k != poi_id},
        "anchor_points": anchor_points,
        "trip_anchor_lat": anchor_lat,
        "trip_anchor_lon": anchor_lon,
        # n_interactions / confidence / is_cold_start intentionally carried
        # over unchanged - see the docstring above.
    })
    return loo
