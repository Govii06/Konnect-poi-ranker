"""
Explainability
===============
Simple, template-based reason generation from the same features the
model and scorer already computed - no separate explanation model
needed. This keeps explanations guaranteed-consistent with the actual
score (a common failure mode of post-hoc explainers is disagreeing with
the model).
"""
import pandas as pd


def explain_row(row: pd.Series, profile: dict) -> list:
    reasons = []

    if row.interest_overlap > 0 or row.explicit_tag_score > 0.3:
        reasons.append("Strong match with traveler's stated interests")
    if getattr(row, "text_similarity", 0.0) > 0.15:
        reasons.append("Description language matches the traveler's stated interests")
    if row.hist_cat_affinity > 0.3 or row.hist_tag_affinity > 0.3:
        reasons.append("Similar to previously visited / saved experiences")
    # budget_compat is one-sided (see features.budget_compatibility): it only
    # drops when a POI is OVER budget, so a low score means exactly that.
    if row.budget_compat >= 0.85:
        reasons.append("Within budget")
    elif row.budget_compat < 0.5:
        reasons.append("More expensive than the traveler's stated budget")
    if row.mobility_compat >= 0.8:
        reasons.append(f"Easily reachable via {profile['mobility'].replace('_', ' ')}")
    elif row.mobility_compat < 0.4:
        reasons.append(f"May be far via {profile['mobility'].replace('_', ' ')}")
    if row.is_local == 1.0 and "prefer less touristy" in profile["explicit_preferences"]:
        reasons.append("Lower tourist concentration than comparable POIs")
    if row.is_touristy == 1.0 and "famous landmarks" in profile["explicit_preferences"]:
        reasons.append("A well-known landmark, matching traveler's openness to popular sites")
    if row.is_family_friendly == 1.0 and profile["party_type"] == "family":
        reasons.append("Family-friendly, matching travel party")
    if row.is_family_friendly == 0.0 and "child-friendly" in profile["explicit_preferences"]:
        reasons.append("Not marked child-friendly - suppressed by the traveler's "
                       "child-friendly-only requirement")
    if getattr(row, "duration_fit", 1.0) < 0.5:
        reasons.append(
            f"Long visit relative to a {profile.get('trip_duration', 5)}-day trip"
        )
    if not reasons:
        reasons.append("Reasonable general fit based on category and location")

    return reasons
