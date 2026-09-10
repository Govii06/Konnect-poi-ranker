# Technical Documentation — Konnect POI Intelligence & Ranking System

## Requirements map

The assignment (Deliverable C) asks for eleven specific topics. This document
covers all of them, but adds two sections of its own — Candidate Generation and
Explainability — because the assignment body requires them separately (§9 and
§13). That shifts the numbering, so here is the mapping:

| # | Required topic | Covered in |
|---|---|---|
| 1 | Problem formulation | §1 |
| 2 | Architecture | §2 |
| 3 | Data assumptions | §3 |
| 4 | Feature engineering | §4 (POI prep, POI/traveler representation) |
| 5 | Model selection | §6.1 *Model selection* |
| 6 | Training methodology | §6.2 *Training methodology* |
| 7 | Ranking methodology | §5 (retrieval) + §6 (ranking) |
| 8 | Scoring methodology | §7 (preference × context gate) |
| 9 | Evaluation | §9 |
| 10 | Cold-start strategy | §10 |
| 11 | Production considerations | §11 |

Additional sections: **§5 Candidate Generation** (assignment §9, incl. the
long-tail question) and **§8 Explainability** (assignment §13).

---

## 1. Problem Formulation

Given a traveler's trip context, a POI catalog, and historical
traveler-POI interactions, produce a **personalized, ranked list of
POIs with meaningful relevance scores** for a specific traveler — not a
popularity leaderboard. The output feeds a downstream itinerary
optimizer (out of scope here), so the contract is: for each candidate
POI, provide a relevance/utility score, a preference weight, a context
compatibility score, and a confidence value, plus a short human-readable
explanation for the top results.

Two questions are treated as **distinct**, per the assignment's section
11: "would this traveler *want* this POI?" (preference) vs. "does this
POI *work* for this trip?" (context/practical compatibility). Conflating
them is the main way naive popularity-based systems fail — a beautiful,
highly-rated restaurant that's closed during the trip or triple the
traveler's budget is not a good recommendation no matter how well it
matches interests.

## 2. Architecture

```
Raw POI Data + Traveler/Trip Context + Historical Interactions
    -> Data Preparation        (dedup, impute, normalize, popularity-bias correction)
    -> POI Representation       (numerical/categorical/geographic/behavioral features)
    -> Traveler Representation (explicit interests + implicit behavioral affinity, confidence-weighted)
    -> Candidate Generation     (multi-channel retrieval, long-tail protected)
    -> Personalized Ranking     (gradient-boosted regressor, pointwise learning-to-rank)
    -> Context/Constraint Scoring (deterministic budget/mobility/availability gate)
    -> Explainability            (template-based, derived from the same features)
    -> Ranked & Weighted POIs   (JSON-serializable output for the itinerary planner)
```

Each stage is a separate module (`src/*.py`) with a narrow interface, so
any stage (e.g. swapping the GBM ranker for a two-tower neural model
later) can be replaced without touching the others.

## 3. Data Assumptions

Three synthetic datasets are generated (`data/generate_data.py`), all
for a single destination ("Lisbon") to keep the demo runnable locally:

- **320 POIs** across 8 categories, with an intentionally power-law
  (Pareto) `review_count` distribution — a handful of "famous landmark"
  POIs, a long tail of small local spots with < 5 reviews. 8 near-duplicate
  rows are injected (same place, different casing/id) to exercise
  dedup logic. ~4% of rows have missing rating/review_count/hours to
  exercise imputation.
- **60 travelers**, each with 2-4 explicit interests, a budget tier, party
  type, mobility mode, a free-text explicit preference string, and a
  trip of 2-10 nights on varying dates. Trip length is deliberately
  varied so `trip_duration` is a discriminative feature rather than a
  constant column.
- **~1,000 interactions** sampled from an internal ground-truth affinity
  function (traveler interests × POI tags/category/price), so the
  dataset has real, learnable signal rather than pure noise, while the
  ranking model itself never sees that ground-truth function directly —
  only the resulting interaction events.

Assumption: a single destination is enough to demonstrate the ranking
logic end-to-end; multi-destination catalogs and cross-destination
transfer are discussed conceptually in §10 (cold start) but not
implemented.

## 4. Feature Engineering

### 4.1 POI Data Preparation (`src/data_prep.py`)
- Category casing/whitespace normalized to a canonical form.
- Near-duplicate POIs collapsed on (normalized name, rounded lat/lon).
- Missing rating/hours imputed with the category-level median (falls
  back to global median); missing review_count treated as 0.
- **Popularity bias correction**: raw `review_count` is log-transformed
  then converted to a **percentile rank** (`popularity_pctile`),
  bounding it to [0,1] instead of letting a handful of landmarks with
  1000+ reviews dominate every distance/similarity calculation.
- `is_cold_start_poi` flag for POIs with < 5 reviews (see §10).
- Derived boolean flags (`is_touristy`, `is_local`, `is_family_friendly`)
  from the tag set, used heavily downstream.

### 4.2 POI Representation (`src/features.py`, `src/candidate_gen.py`, `src/text_features.py`)
A **hybrid representation**: category/tag membership (categorical),
`rating_norm` / `popularity_pctile` / `price_norm` (numerical),
lat/lon-derived `distance_km` (geographic), boolean semantic flags
(`is_local`, `is_touristy`, `is_family_friendly`), and a **text**
channel (`text_similarity`).

The text channel is a TF-IDF vector over each POI's
`description + subcategory + category + tags`, scored by cosine
similarity against a query built from the traveler's stated interests
and free-text preference (`src/text_features.py`). TF-IDF rather than
sentence embeddings because the catalog is a few hundred short
descriptions: it is transparent (you can read which terms matched),
needs no training data, and has no model to host. The same
`fit`/`similarity` interface can be backed by an embedding + ANN index
once vocabulary mismatch starts to matter at scale (§11).

*Honest caveat*: the synthetic generator writes formulaic descriptions
("A {subcategory} known for {tag}, {tag} vibes"), so on this dataset the
text signal partly restates `subcategory` and `tags`. It still earns
~0.14 importance, but it is wired end-to-end to demonstrate the
mechanism, not presented as proof that free text carries independent
signal here.

No learned embeddings were used for the POI representation as a whole —
with ~15 hand-designed features and a few hundred POIs, a GBM over them
already separates "popular landmark" from "small local spot" (they
differ sharply on `popularity_pctile`, `is_local`, and `is_touristy`
even within the same category), which was the explicit bar set by the
assignment. Embeddings/learned representations are the natural upgrade
at catalog scale (§11).

### 4.3 Traveler Representation (`src/traveler_profile.py`)
Explicit and implicit signals are computed **separately, then fused at
scoring time** rather than merged into one opaque vector:
- **Explicit**: interests, budget, mobility, party type, free-text
  preference — used directly as rule-based nudges and as features.
- **Explicit trip context**: destination, travel dates, and
  `trip_duration`. Trip length enters the model as `duration_fit`: a
  POI's `expected_duration` is expressed as a share of the trip's total
  active time (`trip_days × 8h`) and penalized linearly, so a 3-hour
  boat trip is a reasonable ask on a 10-day trip and an expensive one on
  a 2-day city break. Trip length is generated across 2–10 nights
  precisely so this is a discriminative feature rather than a constant
  column; it earns ~0.05 importance. Travel dates are carried on the
  profile and used for the availability window, but the synthetic
  catalog has no seasonal or per-date closure data, so they do not yet
  drive a learned feature — flagged here rather than papered over.
- **Implicit**: category/tag affinity computed from historical
  interactions, weighted by interaction type (see `interaction_label()`
  — view=1 ... booking=6, dismiss=-3) and normalized to roughly [-1, 1].
- **Fusion**: both explicit-interest and implicit-affinity features are
  passed to the ranking model, which learns their relative weight from
  data, rather than a hand-tuned linear blend. A separate `confidence`
  score (interaction count / 20, capped at 1.0) captures *how much* to
  trust the implicit signal — a traveler with 20+ interactions has a
  confidence of 1.0; a brand-new traveler has confidence 0, so their
  score leans entirely on explicit interests. This confidence also
  propagates into the final output's `confidence` field.

## 5. Candidate Generation (`src/candidate_gen.py`)

**Problem addressed explicitly by the assignment**: a naive
"popularity-sorted top-N" candidate cut would systematically remove
relevant long-tail POIs before the ranker ever sees them — no amount of
downstream ranking sophistication can recover from that.

**Approach**: five independent retrieval **channels**, unioned and
deduplicated:
1. Interest/category match (uncapped by popularity)
2. Geographic proximity (nearest 40 to the traveler's trip anchor)
3. Budget-compatible pool (±1 price tier)
4. **Long-tail guarantee** — a dedicated slice sampled from the bottom
   popularity quartile that still matches interests. This exists
   specifically so a 4-review local gem isn't crowded out by 300-review
   landmarks in the other channels.
5. Popularity backstop (top 15 by popularity) — mainly relevant for
   cold-start travelers with no stated interests at all, so they still
   get a reasonable candidate set to fall back on.

A hard mobility filter (walking/public-transport/car distance caps)
is applied after the union.

**The cap is where multi-channel retrieval usually dies.** When the union
exceeds `max_candidates` (100), something has to be dropped. An earlier
version protected the long-tail slice and then trimmed the remainder *by
popularity* — which quietly re-imposed the exact bias the five channels
exist to prevent. The effect was measurable: widening the budget channel
enlarged the union, the cap bit harder, and long-tail coverage collapsed
to **0.115** while the candidate recall ceiling fell to **0.262**.

The cap is now a **round-robin across channels**: take one POI from each
channel in turn until full, long-tail first. Every channel is then
represented in proportion to how many channels there are, not to how
popular its members happen to be. Same cap, same budget, but long-tail
coverage rose to **0.560** and the recall ceiling recovered to **0.316**.

This is the concrete answer to the assignment's question ("how do you
prevent candidate generation from eliminating relevant but less popular
POIs?"): it is not enough to *have* a long-tail channel — the truncation
policy has to be bias-free too, and it is worth measuring whether it is.

## 6. Personalized Ranking (`src/ranker.py`)

### 6.1 Model selection

1. **Why this approach**: the assignment explicitly favors "a simpler
   model with thoughtful features ... over unnecessary model
   complexity." A pointwise regressor (`GradientBoostingRegressor`,
   150 trees, depth 3) over ~13 engineered features is easy to train,
   debug, and explain — and it naturally outputs a bounded score usable
   directly as `preference_weight`, unlike a pure ranking loss that only
   guarantees relative order. A pairwise/listwise learning-to-rank model
   (e.g. LambdaMART) is the natural production upgrade (§11) once there
   is enough interaction volume to support it.
2. **Features** (15, see `FEATURE_COLUMNS` in `src/features.py`):
   `interest_overlap`, `explicit_tag_score` (content / explicit);
   `text_similarity` (text); `hist_cat_affinity`, `hist_tag_affinity`,
   `confidence` (behavioral / implicit); `distance_score` (geographic);
   `rating_norm`, `popularity_pctile`, `is_local`, `is_touristy`,
   `is_family_friendly` (quality / semantic); `budget_compat`,
   `mobility_compat`, `duration_fit` (practical / trip context).
### 6.2 Training methodology

3. **Learning target**: for every (traveler, POI) pair with ≥1
   historical interaction, sum the interaction-type weights
   (`interaction_label()`) and squash into [0, 1] via
   `(sum + 6) / 18`. A single "booking" lands near 1.0; a single
   "dismiss" lands near 0.0; mixed or weak signals sit mid-range.
4. **Training data construction**: positive-labeled pairs from real
   interaction history, plus **implicit negative sampling** — for every
   traveler, randomly sampled non-interacted POIs are added with
   label 0.0 at a 1:2 positive:negative ratio, so the model learns what
   a traveler *doesn't* want, not just what they've seen.

   **Leave-one-out feature construction (target-leakage fix).** A pair's
   label is derived from traveler T's interactions with POI P. Those
   same interactions also feed `hist_cat_affinity`, `hist_tag_affinity`,
   and the geographic anchor behind `distance_score`. Computing both
   from the same events lets the model read the answer off the features
   instead of learning preference. `profile_excluding_poi()`
   (`src/traveler_profile.py`) subtracts P's exact contribution before
   the training row is built.

   The effect was large and measurable. Before the fix, the two
   behavioral features carried **~75% of total feature importance**
   (0.41 + 0.34) while held-out Precision@10 sat at 0.012 — and
   dropping those two features outright *improved* test precision by
   71%, the signature of leakage. After the fix, importance is spread
   sanely across the feature set (top feature ~0.16, the two behavioral
   features ~0.21 combined) and the model beats both baselines.

   One subtlety worth recording: leave-one-out must apply **only to
   POI-specific signals**. An earlier iteration also decremented
   `n_interactions`/`confidence`, which are traveler-level. Because
   positives pass through the LOO path and sampled negatives do not,
   that handed the trainer a spurious "lower confidence ⇒ higher label"
   shortcut and drove `confidence` to **52% of importance**. Traveler-
   level fields are now carried through unchanged.
5. **Inference**: for a candidate set, compute the same feature vector
   used in training (train/serve consistency is enforced by sharing
   `build_features()` in both paths), call `model.predict()`, clip to
   [0, 1] → `preference_score`.

## 7. Context & Practical Compatibility (`src/context_scoring.py`)

`context_compatibility = (0.4·budget_compat + 0.4·mobility_compat + 0.2·availability) × party_compat`

`budget_compat` is **one-sided**: it is 1.0 whenever a POI costs at or
below the traveler's budget and decays only as the POI goes over. A
symmetric version penalizes a high-budget traveler for cheap POIs, which
is both wrong (someone who can afford anything is not badly served by a
€8 neighbourhood cafe) and self-defeating, since cheap local spots are
exactly what the long-tail channel works to surface.

`party_compat` multiplies rather than adds, so a stated hard requirement
("child-friendly only") cannot be outvoted by a strong budget or
mobility score. It drops to 0.1 — not 0 — for a POI that fails the
requirement, so combined with the gate's floor it suppresses without
deleting, keeping the list populated when few POIs carry the tag.

Combination with preference is **multiplicative with a floor**, not
additive:

```
final_score = preference_score × (floor + (1 − floor) × context_compatibility)     # floor = 0.15
```

Rationale: an additive formula lets a very high preference score
compensate for a hard-blocking constraint (e.g. a 0.95-preference POI
that's completely unaffordable would still rank near the top if you
just add the terms). The multiplicative gate makes practical
constraints actually constrain the outcome, while the `floor` keeps
minor mismatches from *fully* zeroing a POI out — only near-total
incompatibility drives the score close to zero.

## 8. Explainability (`src/explain.py`)

Template-based, generated from the **same feature values** the scorer
already computed (`interest_overlap`, `hist_cat_affinity`,
`budget_compat`, `mobility_compat`, `is_local`/`is_touristy` vs. stated
preference, `is_family_friendly` vs. party type). Deliberately avoids a
separate post-hoc explanation model — reusing the scoring features
guarantees the explanation can never contradict the score.

## 9. Evaluation (`src/evaluate.py`)

- **Precision@10 / Recall@10 / NDCG@10**: a per-traveler 70/30 split
  of historical interactions; `visit`/`booking`/`save` events in the
  held-out 30% are treated as ground truth. Recommendations are
  generated using only the training-split interactions (the held-out
  events are invisible to both training and traveler-profile
  construction).
- **Personalization**: average pairwise `1 - Jaccard` similarity between
  20 travelers' top-10 sets. A well-personalized system should sit well
  above 0 (near-0 would mean everyone gets the same list).
- **Long-tail / local discovery**: fraction of top-10 recommendations
  below the catalog's median popularity percentile.
- **Constraint compatibility**: fraction of top-10 with
  `context_compatibility ≥ 0.6` — guards the exact failure mode the
  multiplicative gate exists to prevent (highly-preferred POIs the
  traveler cannot afford or reach still floating to the top).
- **Category diversity**: distinct categories per top-10 list. Ten
  restaurants is a poor brief for an itinerary planner even if each
  entry scores well.
- **Baselines**: a random ranker and a popularity-only ranker, both over
  the *same candidate sets* the model ranks, so the comparison isolates
  ranking quality from retrieval.
- **Candidate recall ceiling**: the fraction of held-out relevant POIs
  that survive candidate generation at all.

### Why baselines and a recall ceiling are reported

Absolute Precision@10 here is structurally tiny: each traveler has ~2.8
held-out relevant POIs, so a *perfect* ranker caps at ~0.28, and the
interesting question is never "is 0.013 good" but "good relative to
what". Two references answer that, and the second is the naive system
this assignment exists to beat:

| Ranker | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Random (same candidate set) | 0.0080 | 0.0291 | 0.0340 |
| Popularity-only | 0.0037 | 0.0108 | 0.0151 |
| **This model** | **0.0130** | **0.0451** | **0.0614** |

The model beats both baselines on **all three** metrics: **1.63×
random** and **3.50× popularity-only** on Precision@10, and **4.1×
popularity-only** on NDCG@10. Beating a popularity ranker is the
assignment's stated objective, and it is beaten decisively rather than
marginally.

Note the popularity baseline is *weak* here (0.0037) — worse than
random. That is not an artifact: once retrieval stops being
popularity-ordered (see §5), the most-reviewed POIs in a candidate set
are no longer the ones a given traveler engaged with, which is precisely
the premise the assignment is testing.

**The dominant bottleneck is retrieval, not ranking.** The measured
candidate recall ceiling is **0.316**: candidate generation narrows 320
POIs to 100, and ~68% of held-out relevant POIs never reach the ranker.
Recall@10 of 0.045 against that 0.316 ceiling means the ranker captures
~14% of what retrieval makes reachable. That single number reframes the
whole evaluation — it says the next engineering investment belongs in
candidate generation (wider cap, better interest→tag mapping, an
embedding retrieval channel), not in swapping the GBM for something
fancier. Measuring it is why it is here.

### What counts as success

For this prototype: **personalization well above 0** (0.914 — travelers
demonstrably do not receive the same list), **long-tail coverage well
above 0** (0.560 — over half of every top-10 comes from below-median-
popularity POIs, which is the local-discovery goal met head-on),
**constraint compatibility high** (1.000 — the gate is doing its job;
highly-ranked POIs are almost always practically usable), **category
diversity healthy** (0.681 — lists are not ten restaurants), and
**ranking metrics above both baselines on all three measures**. All are
computed by `scripts/run_demo.py` and written into
`docs/example_results.md` on every run.

### Two bugs this evaluation caught

Worth recording, because both were found by *reading the generated
results*, not by a test:

- **Budget compatibility was symmetric** (`abs(price − budget)`), so a
  high-budget traveler scored 0.00 on the cheapest POIs and the
  explainer printed "priced outside traveler's usual budget" for an €€
  cafe — a false statement in a required deliverable, and directly
  opposed to the long-tail goal. Budget is now a **ceiling**: full score
  at or under budget, decaying only above it (`src/features.py`).
- **"Child-friendly only" was a soft nudge**, so a non-child-friendly
  POI reached rank 3 of the Family scenario. "Only" is a filter, not a
  taste; it now runs through the context gate as a real constraint
  (`context_scoring.party_compatibility`) rather than as a feature
  penalty that stronger features could outvote. Constraint compatibility
  moved 0.865 → 1.000 as a result.

### Reproducibility

Every random draw is explicitly seeded and the demo is deterministic:
two runs produce byte-identical `docs/example_results.md`, including
across different `PYTHONHASHSEED` values. This required fixing a real
bug — candidate generation seeded its long-tail sample with the builtin
`hash()` of the traveler id, and Python salts string hashing per
process, so every invocation silently produced a different candidate
set, a different ranking, and different reported metrics.

## 10. Cold-Start Strategy

- **New traveler** (no interaction history): `confidence = 0`, so the
  ranker relies entirely on `interest_overlap` / `explicit_tag_score`
  and the deterministic context-compatibility gate; `hist_cat_affinity`
  / `hist_tag_affinity` default to 0 rather than being imputed from
  other travelers. The candidate-generation popularity backstop channel
  ensures they still get a reasonable candidate pool even with zero
  stated interests (demonstrated by the `SCEN_COLDSTART` scenario).
- **New POI** (no interaction history, `is_cold_start_poi = True`):
  ranking falls back to its content features (category, tags, price,
  rating) since `hist_*` affinity features are traveler-side, not
  POI-side, so a new POI is not structurally disadvantaged there — but
  its output `confidence` is discounted (×0.7) to signal lower certainty
  to the downstream consumer.
- **New destination** (little/no interaction data at all): the whole
  implicit-behavioral half of the system has no signal. The
  recommended approach (not implemented, since a single destination is
  used for the demo) is to fall back to a purely content-based +
  context-compatibility score, seeded with global (cross-destination)
  category/tag affinity priors learned from *other* destinations where
  the traveler has history, if any exist — otherwise pure explicit
  interest + context matching, same as the new-traveler path.

## 11. Production Considerations

- **Scale**: candidate generation must move from an in-memory pandas
  scan to a proper retrieval index — geo queries via PostGIS/S2/H3
  cells, interest/tag matching via an inverted index or a vector store
  (embeddings) for approximate nearest-neighbor retrieval — so
  candidate generation stays sub-second against a catalog of 100k+ POIs.
- **Feature computation**: POI-side features (rating_norm,
  popularity_pctile, tag sets, category) are static-ish and computed
  **offline** in a nightly batch job. Traveler implicit-affinity
  features are also computed **offline** on a schedule (e.g. hourly),
  since they only need to reflect "recent" behavior, not the literal
  latest click. Only `distance_km`/`mobility_compat` (depends on live
  trip anchor) and the final model inference are computed **online**,
  at request time.
- **Model serving**: candidate generation + feature lookup + a single
  batched `model.predict()` call over ~100 candidates is cheap enough
  to run synchronously per request; the trained model would be
  versioned and loaded once per serving process, not per request.
- **Data freshness**: POI info/hours/availability refreshed via
  scheduled ingestion jobs with a freshness SLA (e.g. hours changed
  same-day); popularity/review_count refreshed daily (it moves slowly);
  user behavior features refreshed on a shorter cycle (near-real-time
  event stream feeding the offline affinity job) so a traveler's
  in-trip behavior can influence recommendations within the same trip.
- **Retraining**: the ranking model itself changes slowly (feature
  distributions and constants are stable) — weekly/biweekly retraining
  on the accumulated interaction log is reasonable to start, moving to
  daily as volume grows.
- **Feedback loop**: new interactions flow into the same interaction
  log used for training, so `booking`/`visit`/`dismiss` events
  automatically become future positive/negative training examples on
  the next retrain — no separate labeling step needed since interaction
  type already encodes the learning signal (`interaction_label()`).
- **Monitoring**: candidate-generation funnel size (catalog → candidates
  → top-K) per request; score distribution drift; per-traveler
  personalization metric over time (regression toward "everyone gets
  the same list" is a real production failure mode); long-tail coverage
  trend; latency; and, once online feedback is available, calibration
  (do POIs with `score` ≈ 0.8 actually convert to visits ~80% of the
  time).
