"""
POI Intelligence Pipeline (end-to-end orchestrator)
=====================================================
    Raw POI Data + Traveler/Trip Context + Historical Interactions
        -> data_prep.clean_pois
        -> traveler_profile.build_traveler_profiles
        -> candidate_gen.generate_candidates          (per traveler)
        -> features.build_features                     (per traveler)
        -> ranker.POIRanker.predict                     -> preference_score
        -> context_scoring.compute_final_utility        -> final score
        -> explain.explain_row                           -> reasons
        -> ranked, weighted POI list                     (JSON-serializable)
"""
import pandas as pd

from src.data_prep import clean_pois
from src.traveler_profile import build_traveler_profiles
from src.candidate_gen import generate_candidates
from src.features import build_features, FEATURE_COLUMNS
from src.ranker import POIRanker, build_training_set
from src.context_scoring import compute_final_utility
from src.explain import explain_row
from src.text_features import POITextModel

# How many per-POI contributing signals to surface in the output.
N_TOP_SIGNALS = 3


class POIIntelligencePipeline:
    def __init__(self):
        self.pois = None
        self.profiles = None
        self.ranker = POIRanker()
        self.text_model = POITextModel()
        self.center_lat = None
        self.center_lon = None

    def fit(self, raw_pois: pd.DataFrame, travelers: pd.DataFrame, interactions: pd.DataFrame):
        self.pois = clean_pois(raw_pois)
        self.center_lat = self.pois.latitude.mean()
        self.center_lon = self.pois.longitude.mean()
        self.profiles = build_traveler_profiles(travelers, self.pois, interactions)
        self.text_model.fit(self.pois)

        training_set = build_training_set(
            self.pois, self.profiles, interactions, self.center_lat, self.center_lon,
            text_model=self.text_model,
        )
        self.ranker.fit(training_set)
        self._training_set = training_set  # kept for evaluation splits
        return self

    def _top_signals(self, row: pd.Series) -> list:
        """Which features actually drove this POI's preference score.

        Attribution is `feature_value x global_feature_importance`, ranked
        by magnitude. This is a first-order approximation, not SHAP: it
        answers "which of the features the model cares about are strongly
        present for this POI", which is what a downstream consumer needs
        to display. Named explicitly so the approximation is not mistaken
        for an exact per-prediction decomposition.
        """
        importances = self.ranker.feature_importances()
        contributions = {f: float(row[f]) * float(importances.get(f, 0.0))
                         for f in FEATURE_COLUMNS}
        ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return [{"feature": f, "value": round(float(row[f]), 3),
                 "contribution": round(c, 4)}
                for f, c in ranked[:N_TOP_SIGNALS] if abs(c) > 1e-9]

    def recommend(self, traveler_id: str, top_k: int = 10, explain: bool = True) -> list:
        profile = self.profiles.get(traveler_id)
        if profile is None:
            raise KeyError(f"Unknown traveler_id: {traveler_id}")
        return self.recommend_for_profile(profile, top_k=top_k, explain=explain)

    def recommend_for_profile(self, profile: dict, top_k: int = 10, explain: bool = True) -> list:
        candidates = generate_candidates(self.pois, profile, self.center_lat, self.center_lon)
        if candidates.empty:
            return []
        candidates = candidates.copy()
        candidates["text_similarity"] = self.text_model.similarity(candidates.poi_id, profile)
        feat = build_features(candidates, profile)
        feat["preference_score"] = self.ranker.predict(feat)
        scored = compute_final_utility(feat, profile=profile)
        scored = scored.sort_values("score", ascending=False).head(top_k)

        results = []
        for _, row in scored.iterrows():
            entry = {
                "poi_id": row.poi_id,
                "name": row["name"],
                "category": row.category,
                "score": round(float(row.score), 3),
                "preference_weight": round(float(row.preference_score), 3),
                "context_compatibility": round(float(row.context_compatibility), 3),
                "confidence": round(float(row.confidence), 3),
            }
            if explain:
                entry["reasons"] = explain_row(row, profile)
                entry["top_signals"] = self._top_signals(row)
            results.append(entry)
        return results
