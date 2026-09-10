"""
Text Representation of POIs
=============================
The POI catalog carries free-text (`description`) plus semi-structured
text (`tags`, `subcategory`) that the rest of the feature set only uses
categorically. This module turns that text into a single ranking
feature: the cosine similarity between a POI's text and a query built
from the traveler's stated interests and free-text preference.

Why TF-IDF rather than embeddings
-----------------------------------
The catalog is a few hundred POIs with short descriptions. A TF-IDF +
cosine baseline is transparent (you can read off which terms matched),
has no model to host, and needs no training data. Sentence embeddings
would be the upgrade once descriptions are real prose and the catalog is
large enough that vocabulary mismatch ("kid-friendly" vs "family
friendly") starts to hurt - at which point the same interface here
(`fit` / `similarity`) can be backed by an ANN index instead.

Honest caveat on the synthetic data
-------------------------------------
`data/generate_data.py` writes formulaic descriptions of the form
"A {subcategory} known for {tag}, {tag} vibes." So on THIS dataset the
text signal is largely a restatement of `subcategory` + `tags`, and the
feature earns only modest importance. It is wired end-to-end so the
mechanism is real and would carry independent signal on genuine POI
copy; it is not presented as a strong contributor here.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _poi_text(row) -> str:
    """Concatenate every textual surface a POI exposes."""
    parts = [
        str(row.get("description", "") or ""),
        str(row.get("subcategory", "") or ""),
        str(row.get("category", "") or ""),
        " ".join(sorted(row.get("tag_set", set()))).replace("-", " "),
    ]
    return " ".join(p for p in parts if p).lower()


def _profile_query(profile: dict) -> str:
    """Build the traveler-side query string from explicit signals only.

    Deliberately explicit-only: the implicit/behavioral side already has
    dedicated features (`hist_cat_affinity`, `hist_tag_affinity`), and
    mixing behavior in here would make this feature partly redundant with
    them - and partly leak-prone at training time.
    """
    parts = list(profile.get("explicit_interests") or [])
    pref = profile.get("explicit_preferences") or ""
    if pref and pref != "no preference":
        parts.append(str(pref))
    parts.append(str(profile.get("party_type") or ""))
    return " ".join(parts).replace("-", " ").lower()


class POITextModel:
    """Fits once over the POI catalog, then scores (traveler, POI) pairs."""

    def __init__(self):
        self.vectorizer = None
        self.matrix = None
        self.row_of = {}     # poi_id -> row index in `matrix`
        self._query_cache = {}

    def fit(self, pois: pd.DataFrame):
        texts = [_poi_text(r) for _, r in pois.iterrows()]
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=2, sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform(texts)
        self.row_of = {pid: i for i, pid in enumerate(pois.poi_id)}
        self._query_cache = {}
        return self

    def _query_vector(self, profile: dict):
        """Cached per traveler - the query only depends on explicit signals,
        which do not change across candidates."""
        key = profile["traveler_id"]
        cached = self._query_cache.get(key)
        if cached is not None:
            return cached
        query = _profile_query(profile)
        vec = self.vectorizer.transform([query]) if query.strip() else None
        self._query_cache[key] = vec
        return vec

    def similarity(self, poi_ids, profile: dict) -> np.ndarray:
        """Cosine similarity in [0, 1], one value per POI id.

        Returns zeros for a traveler with no stated interests or preference
        (a true cold-start traveler) - correct behavior: there is no text
        query to match against, so the feature abstains rather than guessing.
        """
        poi_ids = list(poi_ids)
        if self.matrix is None or not poi_ids:
            return np.zeros(len(poi_ids))

        qvec = self._query_vector(profile)
        if qvec is None or qvec.nnz == 0:
            return np.zeros(len(poi_ids))

        rows = [self.row_of.get(pid) for pid in poi_ids]
        known = [i for i, r in enumerate(rows) if r is not None]
        out = np.zeros(len(poi_ids))
        if not known:
            return out

        sub = self.matrix[[rows[i] for i in known]]
        sims = cosine_similarity(sub, qvec).ravel()
        # Round before the value reaches the ranker. Threaded BLAS reduces in
        # non-deterministic order, so this cosine can wobble in its last bits
        # between machines - enough, occasionally, to flip a tree split and
        # change the reported metrics. Ten decimals is far finer than any
        # meaningful similarity difference and makes the feature stable
        # regardless of how the host chooses to thread. (run_demo also pins
        # thread counts; this belt-and-braces protects library users too.)
        out[known] = np.round(np.clip(sims, 0.0, 1.0), 10)
        return out
