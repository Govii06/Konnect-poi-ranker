"""
Context & Practical Compatibility Scoring
============================================
Separates two conceptually distinct questions (per assignment section 11):

  preference_score  -> "How much would this traveler like this POI?"
                        (output of the ML ranker)
  context_compat     -> "Does this POI actually work for this trip?"
                        (deterministic, rule-based: budget, mobility,
                        availability)

Combination: we use a *gated* combination rather than a pure weighted
sum. A POI that is a 0.95 preference match but has 0.0 availability
(closed / fully booked) should NOT rank near the top just because the
other terms are large - an additive formula lets a high preference score
compensate for a hard-blocking constraint, which produces useless
recommendations for the downstream itinerary planner. A multiplicative
gate makes hard constraints actually hard:

    final_utility = preference_score * (floor + (1 - floor) * context_compat)

`floor` (default 0.15) keeps the gate from *fully* zeroing out a POI on
soft mismatches (e.g. budget slightly off) - it should suppress, not
delete, unless availability is truly 0.
"""
import numpy as np
import pandas as pd


def availability_score(poi_row) -> float:
    # Simplified: treat POIs outside a sane opening window as unavailable
    # for demo purposes; in production this would check real hours against
    # the traveler's planned visit slot.
    if pd.isna(poi_row.opening_hour) or pd.isna(poi_row.closing_hour):
        return 0.7  # unknown hours -> mild uncertainty penalty, not a hard block
    return 1.0 if poi_row.closing_hour > poi_row.opening_hour else 0.0


def party_compatibility(df: pd.DataFrame, profile: dict) -> pd.Series:
    """Does this POI work for who the traveler is travelling with?

    A stated "child-friendly only" is a *filter*, not a taste. It was
    previously handled as a small `explicit_tag_score` nudge inside
    `build_features`, which meant a well-located, popular, non-child-friendly
    POI could still outrank child-friendly ones on the strength of the other
    features - the traveler asked for "only" and got a suggestion they cannot
    use. That is precisely the relevant-but-unsuitable failure this module
    exists to prevent, so the constraint belongs here, in the gate, alongside
    budget / mobility / availability.

    Returns 1.0 when there is no party constraint or the POI satisfies it, and
    a small value when it does not. Non-zero on purpose: combined with the
    gate's `floor`, this suppresses an unsuitable POI far down the ranking
    without deleting it outright, which keeps the list populated when a
    destination has few tagged-family-friendly options.
    """
    prefs = str(profile.get("explicit_preferences") or "")
    if "child-friendly" not in prefs:
        return pd.Series(1.0, index=df.index)
    return df["is_family_friendly"].apply(lambda ok: 1.0 if ok == 1.0 else 0.1)


def compute_context_compatibility(df: pd.DataFrame, profile: dict = None) -> pd.Series:
    avail = df.apply(availability_score, axis=1)
    # weighted blend of the practical dimensions
    context = 0.4 * df["budget_compat"] + 0.4 * df["mobility_compat"] + 0.2 * avail

    # Party fit multiplies rather than adds: a hard "only" requirement must not
    # be outvoted by a strong budget/mobility score.
    if profile is not None:
        context = context * party_compatibility(df, profile)

    return context.clip(0, 1)


def compute_final_utility(df: pd.DataFrame, floor: float = 0.15,
                           profile: dict = None) -> pd.DataFrame:
    df = df.copy()
    df["context_compatibility"] = compute_context_compatibility(df, profile)
    gate = floor + (1 - floor) * df["context_compatibility"]
    df["score"] = (df["preference_score"] * gate).clip(0, 1)

    # confidence: blends traveler history depth with candidate's own data
    # sparsity (cold-start POIs get a slightly lower confidence)
    poi_data_confidence = np.where(df.get("is_cold_start_poi", False), 0.7, 1.0)
    df["confidence"] = (0.5 + 0.5 * df["confidence"]) * poi_data_confidence
    df["confidence"] = df["confidence"].clip(0, 1)

    return df
