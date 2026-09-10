"""
End-to-end demo: trains the pipeline on the synthetic dataset, evaluates
it against random and popularity baselines, and runs 5 example traveler
scenarios:
  - 3 from the assignment brief (local-experience / history / family)
  - 1 true cold-start traveler with zero history and no stated interests
  - 1 returning traveler with real interaction history, so the
    behavioral half of the model is actually exercised

Fully deterministic: every random draw is explicitly seeded, so two runs
produce byte-identical `docs/example_results.md`.

Run with:  python -m scripts.run_demo
"""
import os

# Pin BLAS/OpenMP thread counts BEFORE numpy, scipy or sklearn are imported -
# once they are loaded, these are read-only.
#
# Why: threaded BLAS sums floating-point values in whatever order threads
# finish, so a dot product gives answers that differ in the last bit or two
# depending on how many cores the machine has. That is normally harmless, but
# here it perturbs `text_similarity` at ~1e-16, which is occasionally enough to
# flip a split decision in the gradient-boosted ranker and visibly change the
# reported metrics. Measured on this project: 1 and 2 threads agreed, 8 threads
# did not. Since the assignment requires another engineer to *reproduce the
# reported results*, the demo pins the thread count so the numbers in
# docs/example_results.md are machine-independent rather than merely
# machine-consistent. Costs a fraction of a second at this data size.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.pipeline import POIIntelligencePipeline
from src.traveler_profile import build_traveler_profiles
from src.evaluate import (
    evaluate_ranking, evaluate_personalization, evaluate_long_tail_coverage,
    evaluate_baselines, evaluate_candidate_recall_ceiling,
    evaluate_constraint_compatibility, evaluate_category_diversity,
)

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_MD = Path(__file__).parent.parent / "docs" / "example_results.md"

SEED = 7
np.random.seed(SEED)


def load_data():
    pois = pd.read_csv(DATA_DIR / "pois.csv")
    travelers = pd.read_csv(DATA_DIR / "travelers.csv")
    interactions = pd.read_csv(DATA_DIR / "interactions.csv")
    return pois, travelers, interactions


def split_interactions(interactions: pd.DataFrame, test_frac: float = 0.3, seed: int = SEED):
    """Per-traveler random split so every traveler has some train and some
    held-out interactions (where possible)."""
    train_parts, test_parts = [], []
    for tid, group in interactions.groupby("traveler_id"):
        group = group.sample(frac=1.0, random_state=seed)
        n_test = max(2, int(len(group) * test_frac)) if len(group) >= 5 else 0
        test_parts.append(group.iloc[:n_test])
        train_parts.append(group.iloc[n_test:])
    return pd.concat(train_parts).reset_index(drop=True), pd.concat(test_parts).reset_index(drop=True)


def make_scenario_profile(pipeline, traveler_id, destination, interests, budget, party_type,
                           mobility, explicit_preferences, trip_duration=5):
    """Builds a fresh traveler profile dict (no historical interactions) -
    used for the 3 brief scenarios and to demonstrate cold-start behavior."""
    start = pd.Timestamp("2026-10-01")
    fake_traveler = pd.DataFrame([{
        "traveler_id": traveler_id, "destination": destination,
        "travel_start": start.strftime("%Y-%m-%d"),
        "travel_end": (start + pd.Timedelta(days=trip_duration)).strftime("%Y-%m-%d"),
        "trip_duration": trip_duration, "interests": "|".join(interests),
        "budget": budget, "party_type": party_type, "mobility": mobility,
        "explicit_preferences": explicit_preferences,
    }])
    empty_interactions = pd.DataFrame(columns=["traveler_id", "poi_id", "interaction", "timestamp"])
    profiles = build_traveler_profiles(fake_traveler, pipeline.pois, empty_interactions)
    return profiles[traveler_id]


def _fmt_signals(signals: list) -> str:
    if not signals:
        return ""
    return "<br>".join(f"`{s['feature']}`={s['value']} (contrib {s['contribution']})"
                       for s in signals)


BUDGET_LABEL = {1: "low", 2: "medium", 4: "high"}

SCENARIO_TITLES = {
    "SCEN_LOCAL": "1 - Local Experience",
    "SCEN_HISTORY": "2 - History",
    "SCEN_FAMILY": "3 - Family",
    "SCEN_COLDSTART": "4 - Cold start (no history, no stated interests)",
}


def format_results_md(scenario_name, profile, recs):
    budget = BUDGET_LABEL.get(profile["budget_num"], str(profile["budget_num"]))
    lines = [f"## Scenario: {scenario_name}", ""]
    lines.append(f"- Interests: {', '.join(sorted(profile['explicit_interests'])) or '(none stated)'}")
    lines.append(f"- Budget: {budget} | Party: {profile['party_type']} "
                 f"| Mobility: {profile['mobility'].replace('_', ' ')}")
    lines.append(f"- Trip: {profile['destination']}, {profile['trip_duration']} days "
                 f"({profile['travel_start']} to {profile['travel_end']})")
    lines.append(f"- Explicit preference: {profile['explicit_preferences']}")
    lines.append(f"- History: {profile['n_interactions']} interactions "
                 f"| profile confidence {profile['confidence']:.2f} "
                 f"| cold start: {profile['is_cold_start']}")
    lines.append("")
    lines.append("| Rank | POI | Category | Score | Preference | Context | Confidence | "
                 "Top signals (feature = value) | Explanation |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(recs, 1):
        reasons = "; ".join(r.get("reasons", []))
        lines.append(
            f"| {i} | {r['name']} | {r['category']} | {r['score']} | {r['preference_weight']} | "
            f"{r['context_compatibility']} | {r['confidence']} | "
            f"{_fmt_signals(r.get('top_signals'))} | {reasons} |"
        )
    lines.append("")
    return "\n".join(lines)


def pick_returning_traveler(pipeline, test_interactions):
    """The traveler with the richest history that also has held-out
    ground truth - the most informative case to display."""
    eligible = set(test_interactions.traveler_id)
    scored = [(p["n_interactions"], tid) for tid, p in pipeline.profiles.items()
              if tid in eligible]
    if not scored:
        scored = [(p["n_interactions"], tid) for tid, p in pipeline.profiles.items()]
    scored.sort(reverse=True)
    return scored[0][1]


def main():
    pois_raw, travelers, interactions = load_data()
    train_interactions, test_interactions = split_interactions(interactions)

    pipeline = POIIntelligencePipeline()
    pipeline.fit(pois_raw, travelers, train_interactions)
    print(f"\nTrained on {len(pipeline._training_set)} (pos+neg) pairs, "
          f"{len(pipeline.pois)} clean POIs, {len(pipeline.profiles)} traveler profiles.\n")

    # ---- Evaluation -------------------------------------------------
    summary, per_traveler = evaluate_ranking(pipeline, test_interactions, k=10)
    baselines = evaluate_baselines(pipeline, test_interactions, k=10)
    funnel = evaluate_candidate_recall_ceiling(pipeline, test_interactions)

    print("=== Ranking metrics vs. baselines (held-out interactions, K=10) ===")
    print(f"  {'':<14}{'P@10':>9}{'R@10':>9}{'NDCG@10':>10}")
    print(f"  {'random':<14}{baselines['random_precision@k']:>9.4f}"
          f"{baselines['random_recall@k']:>9.4f}{baselines['random_ndcg@k']:>10.4f}")
    print(f"  {'popularity':<14}{baselines['popularity_precision@k']:>9.4f}"
          f"{baselines['popularity_recall@k']:>9.4f}{baselines['popularity_ndcg@k']:>10.4f}")
    print(f"  {'MODEL':<14}{summary['precision@k']:>9.4f}"
          f"{summary['recall@k']:>9.4f}{summary['ndcg@k']:>10.4f}")
    lift_rand = summary['precision@k'] / baselines['random_precision@k'] if baselines['random_precision@k'] else float('nan')
    lift_pop = summary['precision@k'] / baselines['popularity_precision@k'] if baselines['popularity_precision@k'] else float('nan')
    print(f"  -> lift over random: {lift_rand:.2f}x | lift over popularity: {lift_pop:.2f}x")

    print("\n=== Candidate-generation funnel (recall ceiling) ===")
    print(f"  catalog size                : {funnel['catalog_size']}")
    print(f"  avg candidate set size      : {funnel['avg_candidate_set_size']:.1f}")
    print(f"  avg relevant per traveler   : {funnel['avg_relevant_per_traveler']:.2f}")
    print(f"  CANDIDATE RECALL CEILING    : {funnel['candidate_recall_ceiling']:.3f}"
          f"   <- hard upper bound on Recall@K")

    all_ids = sorted(pipeline.profiles.keys())
    sample_ids = list(np.random.RandomState(1).choice(all_ids, size=20, replace=False))
    personalization = evaluate_personalization(pipeline, sample_ids, k=10)
    long_tail = evaluate_long_tail_coverage(pipeline, sample_ids, k=10)
    constraint_ok = evaluate_constraint_compatibility(pipeline, sample_ids, k=10)
    diversity = evaluate_category_diversity(pipeline, sample_ids, k=10)

    print("\n=== Beyond-accuracy metrics (20 sampled travelers, K=10) ===")
    print(f"  personalization (1 - Jaccard) : {personalization:.3f}")
    print(f"  long-tail coverage            : {long_tail:.3f}")
    print(f"  constraint compatibility      : {constraint_ok:.3f}")
    print(f"  category diversity            : {diversity:.3f}")

    importances = pipeline.ranker.feature_importances()
    print("\n=== Feature importances ===")
    print(importances.to_string())

    # ---- Scenarios ----------------------------------------------------
    scenarios = [
        dict(traveler_id="SCEN_LOCAL", destination="Lisbon",
             interests=["local food", "neighborhoods", "local experiences"],
             budget="medium", party_type="couple", mobility="public_transport",
             explicit_preferences="prefer less touristy experiences", trip_duration=4),
        dict(traveler_id="SCEN_HISTORY", destination="Lisbon",
             interests=["history", "architecture", "museums"],
             budget="high", party_type="couple", mobility="public_transport",
             explicit_preferences="famous landmarks are acceptable", trip_duration=5),
        # Interests worded to match assignment section 17, Scenario 3 exactly:
        # "Activities, parks, interactive experiences".
        dict(traveler_id="SCEN_FAMILY", destination="Lisbon",
             interests=["activities", "parks", "interactive experiences"],
             budget="medium", party_type="family", mobility="car",
             explicit_preferences="child-friendly only", trip_duration=7),
        dict(traveler_id="SCEN_COLDSTART", destination="Lisbon",
             interests=[],  # no stated interests either - worst-case cold start
             budget="medium", party_type="solo", mobility="walking",
             explicit_preferences="no preference", trip_duration=2),
    ]

    md = ["# Example Results", "",
          "Generated by `python -m scripts.run_demo`. Deterministic: rerunning "
          "reproduces this file exactly (seeded throughout, with BLAS thread "
          "counts pinned so results do not depend on the host's core count).",
          "",
          "## How to read this",
          "",
          "Each scenario below is one traveler. The table is what the system "
          "hands the downstream itinerary planner, ranked best-first.",
          "",
          "| Column | Meaning | Range |",
          "|---|---|---|",
          "| **Score** | Final utility — what to rank on. `Preference x context gate`. | 0-1, higher is better |",
          "| **Preference** | How much this traveler is predicted to *want* this POI, from the ML ranker alone, ignoring practicality. | 0-1 |",
          "| **Context** | How well the POI *works* for this trip: budget, mobility, opening hours, party fit. | 0-1 |",
          "| **Confidence** | How much evidence stands behind the estimate — grows with the traveler's interaction history, reduced for POIs with almost no reviews. | 0-1 |",
          "| **Top signals** | The features that most drove this POI's preference score, as `feature=value (contribution)`, where contribution is `value x the feature's global importance`. A first-order attribution, not SHAP. | — |",
          "| **Explanation** | Plain-language reasons, generated from the same feature values that produced the score, so they cannot contradict it. | — |",
          "",
          "Preference and Context are deliberately **separate**: a POI can be a "
          "great match and still be unusable (closed, unaffordable, unreachable). "
          "They are combined multiplicatively so a hard practical failure cannot "
          "be outvoted by a high preference score. See `TECHNICAL_DOC.md` section 7.",
          ""]

    scenario_sets = {}   # name -> set of recommended poi_ids, for the overlap check

    for s in scenarios:
        profile = make_scenario_profile(pipeline, **s)
        recs = pipeline.recommend_for_profile(profile, top_k=8, explain=True)
        name = SCENARIO_TITLES.get(s["traveler_id"], s["traveler_id"])
        scenario_sets[name] = {r["poi_id"] for r in recs}
        print(f"\n=== Scenario: {name} ===")
        for i, r in enumerate(recs[:5], 1):
            print(f"  {i}. {r['name']} ({r['category']}) score={r['score']} "
                  f"[pref={r['preference_weight']} ctx={r['context_compatibility']}]")
            print(f"     signals: {_fmt_signals(r['top_signals']).replace('<br>', ' | ')}")
            print(f"     reasons: {r['reasons']}")
        md.append(format_results_md(name, profile, recs))

    # --- 5th scenario: a REAL traveler with interaction history --------
    returning_id = pick_returning_traveler(pipeline, test_interactions)
    returning_profile = pipeline.profiles[returning_id]
    recs = pipeline.recommend(returning_id, top_k=8, explain=True)
    print(f"\n=== Scenario: Returning traveler ({returning_id}, "
          f"{returning_profile['n_interactions']} interactions) ===")
    for i, r in enumerate(recs[:5], 1):
        print(f"  {i}. {r['name']} ({r['category']}) score={r['score']} "
              f"[pref={r['preference_weight']} ctx={r['context_compatibility']}]")
        print(f"     signals: {_fmt_signals(r['top_signals']).replace('<br>', ' | ')}")
        print(f"     reasons: {r['reasons']}")
    returning_title = (f"5 - Returning traveler {returning_id} "
                       f"({returning_profile['n_interactions']} past interactions)")
    scenario_sets[returning_title] = {r["poi_id"] for r in recs}
    md.append(format_results_md(returning_title, returning_profile, recs))

    # --- Do the scenarios actually differ? (assignment section 17) -------
    md.append("## Do these rankings actually differ?\n")
    md.append("The point of the scenarios is to show the system produces "
              "**personalized rankings rather than one universal popularity "
              "ranking**. Overlap between each pair of scenario top-8 lists:\n")
    names = list(scenario_sets)
    md.append("| | " + " | ".join(n.split(" - ")[0] for n in names) + " |")
    md.append("|---" * (len(names) + 1) + "|")
    for a in names:
        row = [a]
        for b in names:
            if a == b:
                row.append("—")
            else:
                shared = len(scenario_sets[a] & scenario_sets[b])
                row.append(f"{shared}/8")
        md.append("| " + " | ".join(row) + " |")
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    overlaps = [len(scenario_sets[a] & scenario_sets[b]) for a, b in pairs]
    md.append("")
    md.append(f"Average overlap across all {len(pairs)} pairs: "
              f"**{sum(overlaps) / len(overlaps):.1f} of 8 POIs**. "
              "A popularity-ranked system would score 8/8 everywhere.\n")

    # --- Output schema, so a consumer knows what it receives -------------
    md.append("## Output schema\n")
    md.append("The tables above are a rendering of what `pipeline.recommend()` "
              "returns. One entry looks like this:\n")
    sample = dict(recs[0])
    sample["reasons"] = sample["reasons"][:2] + ["..."]
    sample["top_signals"] = sample["top_signals"][:1] + ["..."]
    md.append("```json")
    md.append(json.dumps(sample, indent=2)[:1200])
    md.append("```\n")

    # --- Evaluation summary written into the doc ------------------------
    md.append("## Evaluation Summary\n")
    md.append("### Ranking quality vs. baselines (K=10, held-out interactions)\n")
    md.append("| Ranker | Precision@10 | Recall@10 | NDCG@10 |")
    md.append("|---|---|---|---|")
    md.append(f"| Random (same candidate set) | {baselines['random_precision@k']:.4f} | "
              f"{baselines['random_recall@k']:.4f} | {baselines['random_ndcg@k']:.4f} |")
    md.append(f"| Popularity-only | {baselines['popularity_precision@k']:.4f} | "
              f"{baselines['popularity_recall@k']:.4f} | {baselines['popularity_ndcg@k']:.4f} |")
    md.append(f"| **This model** | **{summary['precision@k']:.4f}** | "
              f"**{summary['recall@k']:.4f}** | **{summary['ndcg@k']:.4f}** |")
    md.append("")
    md.append(f"Lift over random: **{lift_rand:.2f}x** | "
              f"lift over popularity-only: **{lift_pop:.2f}x** (Precision@10)")
    md.append("")

    # Report any metric where a baseline actually wins, rather than quoting
    # only the favourable one.
    losses = [name for name, mine, base in (
        ("Precision@10", summary['precision@k'], baselines['popularity_precision@k']),
        ("Recall@10", summary['recall@k'], baselines['popularity_recall@k']),
        ("NDCG@10", summary['ndcg@k'], baselines['popularity_ndcg@k']),
    ) if mine < base]
    if losses:
        md.append(f"Honest caveat: the popularity baseline still edges the model on "
                  f"**{', '.join(losses)}**. The model wins on the rank-sensitive metric "
                  "(NDCG@10), i.e. it places relevant POIs higher when it finds them, but "
                  "popularity retrieves a slightly wider set of them. With ~2.8 relevant "
                  "POIs per traveler these differences rest on a handful of events; the "
                  "fix is more interaction volume and a wider candidate cap, not more "
                  "model capacity.")
    else:
        md.append("The model beats both baselines on all three ranking metrics.")
    md.append("")
    md.append("Absolute Precision@10 is necessarily small here: each traveler has "
              f"~{funnel['avg_relevant_per_traveler']:.1f} held-out relevant POIs, so even a "
              "perfect ranker tops out near "
              f"{funnel['avg_relevant_per_traveler'] / 10:.2f}. The lift over the two baselines "
              "is the meaningful figure, and beating the popularity ranker is the "
              "assignment's stated objective.\n")

    md.append("### Candidate-generation funnel\n")
    md.append(f"- Catalog size: {funnel['catalog_size']} POIs")
    md.append(f"- Average candidate set: {funnel['avg_candidate_set_size']:.1f} POIs")
    md.append(f"- **Candidate recall ceiling: {funnel['candidate_recall_ceiling']:.3f}** "
              "— the hard upper bound on Recall@10; anything retrieval drops is "
              "unreachable by the ranker.")
    md.append(f"- Recall@10 of {summary['recall@k']:.3f} against a ceiling of "
              f"{funnel['candidate_recall_ceiling']:.3f} means the ranker captures "
              f"{summary['recall@k'] / funnel['candidate_recall_ceiling'] * 100:.0f}% "
              "of what retrieval makes reachable.\n")

    md.append("### Beyond-accuracy metrics (20 sampled travelers, K=10)\n")
    md.append(f"- Personalization (1 - Jaccard, avg pairwise): {personalization:.3f}")
    md.append(f"- Long-tail coverage (fraction below median popularity): {long_tail:.3f}")
    md.append(f"- Constraint compatibility (context >= 0.6): {constraint_ok:.3f}")
    md.append(f"- Category diversity (distinct categories / 10): {diversity:.3f}\n")

    md.append("### Model feature importances\n")
    md.append("| Feature | Importance |")
    md.append("|---|---|")
    for feat, val in importances.items():
        md.append(f"| `{feat}` | {val:.4f} |")
    md.append("")

    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
