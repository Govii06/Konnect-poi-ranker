"""
Evaluation
===========
1. Ranking quality: Precision@K, Recall@K, NDCG@K, evaluated by holding
   out a slice of each traveler's positive interactions ("visit",
   "booking", "save") as ground-truth relevant POIs, then checking how
   many appear in the top-K recommendations generated *without* that
   held-out interaction being visible to the model at inference time
   (we simply check whether the held-out POI appears in top-K; the
   model was trained on the pre-split interaction set).

2. Personalization: average pairwise dissimilarity between different
   travelers' top-K recommendation sets (1 - Jaccard). Near 0 means the
   system is returning the same list to everyone (bad); higher means
   rankings are meaningfully personalized.

3. Long-tail / local discovery: fraction of top-K POIs that sit below
   the catalog's median popularity percentile.

4. Baselines: Precision/Recall/NDCG@K for (a) a random ranker over the
   same candidate set and (b) a pure popularity ranker. Absolute
   Precision@K on this dataset is small (each traveler has ~2-3 held-out
   relevant POIs out of a 100-POI candidate set), so a bare "0.02" is
   uninterpretable. What matters is the lift over these two references -
   and beating the popularity baseline is precisely the assignment's
   stated goal. Both baselines rank the *same* candidate sets the model
   sees, so the comparison isolates ranking quality from retrieval.

5. Candidate recall ceiling: the fraction of held-out relevant POIs that
   survive candidate generation at all. This is the hard upper bound on
   Recall@K - anything the retrieval stage drops is unreachable no
   matter how good the ranker is. Reporting it separates "the ranker is
   weak" from "the funnel is narrow", which are different problems with
   different fixes.
"""
import numpy as np
import pandas as pd

from src.candidate_gen import generate_candidates

RELEVANT_TYPES = {"visit", "booking", "save"}


def _relevant_by_traveler(held_out_interactions: pd.DataFrame):
    """{traveler_id: set(poi_id)} of ground-truth relevant held-out POIs."""
    held_out = held_out_interactions[held_out_interactions.interaction.isin(RELEVANT_TYPES)]
    return held_out.groupby("traveler_id")["poi_id"].apply(set)


def precision_recall_ndcg_at_k(recommended_ids: list, relevant_ids: set, k: int):
    top_k = recommended_ids[:k]
    if not top_k:
        return 0.0, 0.0, 0.0
    hits = [1 if poi_id in relevant_ids else 0 for poi_id in top_k]
    precision = sum(hits) / len(top_k)
    recall = sum(hits) / len(relevant_ids) if relevant_ids else 0.0

    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    ideal_hits = sorted(hits, reverse=True)
    idcg = sum(h / np.log2(i + 2) for i, h in enumerate(ideal_hits))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return precision, recall, ndcg


def evaluate_ranking(pipeline, held_out_interactions: pd.DataFrame, k: int = 10):
    relevant_by_traveler = _relevant_by_traveler(held_out_interactions)

    rows = []
    for tid, relevant_ids in relevant_by_traveler.items():
        if tid not in pipeline.profiles:
            continue
        recs = pipeline.recommend(tid, top_k=k, explain=False)
        rec_ids = [r["poi_id"] for r in recs]
        p, r_, n = precision_recall_ndcg_at_k(rec_ids, relevant_ids, k)
        rows.append({"traveler_id": tid, "precision@k": p, "recall@k": r_, "ndcg@k": n})

    result_df = pd.DataFrame(rows)
    summary = result_df[["precision@k", "recall@k", "ndcg@k"]].mean().to_dict()
    return summary, result_df


def evaluate_personalization(pipeline, traveler_ids: list, k: int = 10):
    rec_sets = {}
    for tid in traveler_ids:
        recs = pipeline.recommend(tid, top_k=k, explain=False)
        rec_sets[tid] = set(r["poi_id"] for r in recs)

    dissimilarities = []
    ids = list(rec_sets.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = rec_sets[ids[i]], rec_sets[ids[j]]
            union = a | b
            jaccard = len(a & b) / len(union) if union else 0.0
            dissimilarities.append(1 - jaccard)

    return float(np.mean(dissimilarities)) if dissimilarities else 0.0


def evaluate_long_tail_coverage(pipeline, traveler_ids: list, k: int = 10):
    median_pop = pipeline.pois.popularity_pctile.median()
    pop_lookup = pipeline.pois.set_index("poi_id").popularity_pctile

    fractions = []
    for tid in traveler_ids:
        recs = pipeline.recommend(tid, top_k=k, explain=False)
        if not recs:
            continue
        long_tail_count = sum(
            1 for r in recs if pop_lookup.get(r["poi_id"], 1.0) < median_pop
        )
        fractions.append(long_tail_count / len(recs))

    return float(np.mean(fractions)) if fractions else 0.0


def evaluate_baselines(pipeline, held_out_interactions: pd.DataFrame, k: int = 10,
                        n_trials: int = 20, seed: int = 0) -> dict:
    """Random and popularity baselines over the SAME candidate sets the
    model ranks, so the comparison measures ranking, not retrieval.

    - random: average over `n_trials` shuffles, to keep the number stable.
    - popularity: top-K by `popularity_pctile`, i.e. the naive
      "just show the famous places" system this assignment exists to beat.
    """
    relevant_by_traveler = _relevant_by_traveler(held_out_interactions)
    rng = np.random.RandomState(seed)

    rand_rows, pop_rows = [], []
    for tid, relevant_ids in relevant_by_traveler.items():
        profile = pipeline.profiles.get(tid)
        if profile is None:
            continue
        candidates = generate_candidates(pipeline.pois, profile,
                                         pipeline.center_lat, pipeline.center_lon)
        if candidates.empty:
            continue
        cand_ids = list(candidates.poi_id)

        trial_scores = []
        for _ in range(n_trials):
            shuffled = list(rng.permutation(cand_ids))
            trial_scores.append(precision_recall_ndcg_at_k(shuffled, relevant_ids, k))
        rand_rows.append(np.mean(trial_scores, axis=0))

        pop_ids = list(candidates.sort_values("popularity_pctile", ascending=False).poi_id)
        pop_rows.append(precision_recall_ndcg_at_k(pop_ids, relevant_ids, k))

    def _summarize(rows, prefix):
        if not rows:
            return {f"{prefix}_precision@k": 0.0, f"{prefix}_recall@k": 0.0,
                    f"{prefix}_ndcg@k": 0.0}
        m = np.mean(rows, axis=0)
        return {f"{prefix}_precision@k": float(m[0]),
                f"{prefix}_recall@k": float(m[1]),
                f"{prefix}_ndcg@k": float(m[2])}

    out = {}
    out.update(_summarize(rand_rows, "random"))
    out.update(_summarize(pop_rows, "popularity"))
    return out


def evaluate_candidate_recall_ceiling(pipeline, held_out_interactions: pd.DataFrame) -> dict:
    """Upper bound on achievable recall, imposed by candidate generation.

    If this reads 0.30, then 70% of the ground-truth POIs never reach the
    ranker and Recall@K can never exceed 0.30 regardless of model quality.
    It is the single most useful diagnostic for deciding whether to invest
    in the retrieval stage or the ranking stage.
    """
    relevant_by_traveler = _relevant_by_traveler(held_out_interactions)

    ceilings, sizes, rel_counts = [], [], []
    for tid, relevant_ids in relevant_by_traveler.items():
        profile = pipeline.profiles.get(tid)
        if profile is None:
            continue
        candidates = generate_candidates(pipeline.pois, profile,
                                         pipeline.center_lat, pipeline.center_lon)
        cand_ids = set(candidates.poi_id)
        sizes.append(len(cand_ids))
        rel_counts.append(len(relevant_ids))
        ceilings.append(len(relevant_ids & cand_ids) / len(relevant_ids) if relevant_ids else 0.0)

    return {
        "candidate_recall_ceiling": float(np.mean(ceilings)) if ceilings else 0.0,
        "avg_candidate_set_size": float(np.mean(sizes)) if sizes else 0.0,
        "avg_relevant_per_traveler": float(np.mean(rel_counts)) if rel_counts else 0.0,
        "catalog_size": int(len(pipeline.pois)),
        "travelers_evaluated": len(ceilings),
    }


def evaluate_constraint_compatibility(pipeline, traveler_ids: list, k: int = 10,
                                       threshold: float = 0.6) -> float:
    """Fraction of top-K recommendations that are practically usable
    (`context_compatibility` >= threshold).

    Guards the failure mode the multiplicative gate exists to prevent:
    highly-preferred POIs that the traveler cannot actually afford or
    reach still floating to the top.
    """
    fractions = []
    for tid in traveler_ids:
        recs = pipeline.recommend(tid, top_k=k, explain=False)
        if not recs:
            continue
        ok = sum(1 for r in recs if r["context_compatibility"] >= threshold)
        fractions.append(ok / len(recs))
    return float(np.mean(fractions)) if fractions else 0.0


def evaluate_category_diversity(pipeline, traveler_ids: list, k: int = 10) -> float:
    """Average number of distinct categories per top-K list, normalized by
    the number of categories in the catalog. A list of ten restaurants is
    a poor brief for a downstream itinerary planner even if every entry
    scores well."""
    cat_lookup = pipeline.pois.set_index("poi_id").category
    n_categories = pipeline.pois.category.nunique()
    if not n_categories:
        return 0.0

    scores = []
    for tid in traveler_ids:
        recs = pipeline.recommend(tid, top_k=k, explain=False)
        if not recs:
            continue
        cats = {cat_lookup.get(r["poi_id"]) for r in recs}
        scores.append(len(cats) / min(k, n_categories))
    return float(np.mean(scores)) if scores else 0.0
