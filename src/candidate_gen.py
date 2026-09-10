"""
Candidate Generation
=====================
Reduces the full POI catalog (100s-1000s) down to a manageable candidate
set (tens) before the more expensive personalized ranking step runs.

Design goal called out explicitly in the assignment:
    "How do you prevent candidate generation from eliminating relevant
    but less popular / long-tail POIs?"

Approach: a *union of retrieval channels* rather than a single popularity-
sorted cut. Each channel retrieves independently, then results are merged
and deduplicated. This is the same idea production recsys use (multi-
channel / multi-source candidate generation) to avoid one signal
(popularity) starving out everything else.

Channels:
  1. Interest/category match  - POIs whose category or tags overlap the
     traveler's explicit interests or implicit affinities (uncapped by
     popularity).
  2. Geographic proximity      - nearest POIs to the traveler's trip
     anchor (or destination center for cold-start).
  3. Budget/mobility compatible - filtered pool respecting hard
     constraints traveler explicitly cares about.
  4. Long-tail guarantee        - a dedicated slice reserved for POIs in
     the bottom popularity quartile that still match interests. This
     channel exists specifically so a great local spot with 4 reviews
     isn't drowned out by 300-review landmarks in the other channels.
  5. Popularity backstop         - a small top-popularity slice, mainly
     useful for cold-start travelers with no stated interests at all.

The final candidate set is the union (deduplicated), capped at
`max_candidates`.
"""
import zlib

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


INTEREST_TO_TAGS = {
    "local food": {"category": ["Restaurant"], "tags": ["local"]},
    "history": {"category": ["Historical Site"], "tags": ["historic"]},
    "architecture": {"category": ["Historical Site"], "tags": ["historic"]},
    "museums": {"category": ["Museum"], "tags": []},
    "neighborhoods": {"category": ["Neighborhood Experience"], "tags": ["local"]},
    "local experiences": {"category": ["Neighborhood Experience"], "tags": ["local", "authentic"]},
    "nature": {"category": ["Nature/Outdoor"], "tags": ["outdoor", "scenic"]},
    "shopping": {"category": ["Shopping"], "tags": []},
    "art": {"category": ["Museum"], "tags": []},
    "family activities": {"category": ["Activity"], "tags": ["family-friendly"]},
    # Wording used verbatim by the assignment's Scenario 3 (section 17).
    "activities": {"category": ["Activity"], "tags": []},
    "parks": {"category": ["Nature/Outdoor"], "tags": ["outdoor", "scenic"]},
    "interactive experiences": {"category": ["Activity"], "tags": ["family-friendly"]},
    "nightlife": {"category": ["Event"], "tags": ["lively"]},
    "photography": {"category": [], "tags": ["scenic", "instagrammable"]},
    "adventure": {"category": ["Activity"], "tags": ["outdoor"]},
    "relaxation": {"category": ["Nature/Outdoor"], "tags": ["quiet", "scenic"]},
    "beaches": {"category": ["Nature/Outdoor"], "tags": ["scenic"]},
}


def _interest_match_mask(pois: pd.DataFrame, interests: set, tag_affinity: dict) -> pd.Series:
    if not interests and not tag_affinity:
        return pd.Series(False, index=pois.index)
    wanted_categories, wanted_tags = set(), set()
    for interest in interests:
        m = INTEREST_TO_TAGS.get(interest, {})
        wanted_categories.update(m.get("category", []))
        wanted_tags.update(m.get("tags", []))
    wanted_tags.update({t for t, v in tag_affinity.items() if v > 0.1})

    return pois.apply(
        lambda r: bool(wanted_categories & {r.category}) or bool(wanted_tags & r.tag_set),
        axis=1,
    )


def generate_candidates(pois: pd.DataFrame, profile: dict,
                         center_lat: float, center_lon: float,
                         max_candidates: int = 100) -> pd.DataFrame:
    df = pois.copy()

    anchor_lat = profile["trip_anchor_lat"] or center_lat
    anchor_lon = profile["trip_anchor_lon"] or center_lon
    df["distance_km"] = haversine_km(anchor_lat, anchor_lon, df.latitude, df.longitude)

    # Channel 1: interest / affinity match
    interest_mask = _interest_match_mask(df, profile["explicit_interests"], profile["tag_affinity"])
    channel_interest = df[interest_mask]

    # Channel 2: geographic proximity (nearest 40)
    channel_geo = df.nsmallest(40, "distance_km")

    # Channel 3: budget-compatible pool - anything at or below the traveler's
    # budget, plus one tier of stretch. Deliberately NOT a symmetric window:
    # `abs(price - budget) <= 1` excluded cheap POIs for high-budget travelers,
    # which starved this channel of exactly the small local spots the long-tail
    # channel exists to protect.
    budget_num = profile["budget_num"]
    channel_budget = df[df.price_level <= budget_num + 1]

    # Channel 4: long-tail guarantee - bottom popularity quartile, still
    # interest-relevant, mobility- and budget-compatible.
    pop_cutoff = df.popularity_pctile.quantile(0.25)
    long_tail_pool = df[(df.popularity_pctile <= pop_cutoff) & interest_mask]
    # NOTE: seeded with crc32, NOT the builtin hash(). Python salts string
    # hashing per process (PYTHONHASHSEED), so hash() here made the whole
    # pipeline non-reproducible across runs - different long-tail samples
    # meant different candidate sets, different rankings, and different
    # reported metrics on every invocation.
    channel_long_tail = long_tail_pool.sample(
        n=min(10, len(long_tail_pool)),
        random_state=zlib.crc32(str(profile["traveler_id"]).encode()) % (2**32),
    ) if len(long_tail_pool) else long_tail_pool

    # Channel 5: popularity backstop (top 15 by popularity) - safety net,
    # dominant only when a traveler has no stated interests (cold start).
    channel_popularity = df.nlargest(15, "popularity_pctile")

    # Ordered channels: long-tail first so it wins ties during the round-robin.
    channels = [
        ("long_tail", channel_long_tail),
        ("interest", channel_interest),
        ("geo", channel_geo),
        ("budget", channel_budget),
        ("popularity", channel_popularity),
    ]

    candidates = pd.concat([c for _, c in channels]).drop_duplicates(subset="poi_id")

    # Hard filters: mobility feasibility (walking travelers shouldn't see
    # POIs 15km away at all, regardless of channel)
    mobility_max_km = {"walking": 6.0, "public_transport": 15.0, "car": 40.0}
    max_km = mobility_max_km.get(profile["mobility"], 12.0)
    candidates = candidates[candidates.distance_km <= max_km]

    if len(candidates) > max_candidates:
        # ROUND-ROBIN across channels, not a popularity cut.
        #
        # The cap is where a multi-channel retriever quietly loses its
        # multi-channel-ness: an earlier version sorted the survivors by
        # popularity, which re-introduced exactly the bias the five channels
        # were designed to avoid - measurably so, since widening the budget
        # channel then pushed long-tail coverage down and lowered the
        # candidate recall ceiling. Taking one POI from each channel in turn
        # keeps every channel represented in proportion to how many channels
        # there are, not how popular its members happen to be.
        alive = set(candidates.poi_id)
        queues = [[p for p in ch.poi_id if p in alive] for _, ch in channels]
        seen, ordered = set(), []
        while len(ordered) < max_candidates and any(queues):
            for q in queues:
                while q:
                    pid = q.pop(0)
                    if pid not in seen:
                        seen.add(pid)
                        ordered.append(pid)
                        break
                if len(ordered) >= max_candidates:
                    break
        rank = {pid: i for i, pid in enumerate(ordered)}
        candidates = (candidates[candidates.poi_id.isin(rank)]
                      .assign(_o=lambda d: d.poi_id.map(rank))
                      .sort_values("_o")
                      .drop(columns="_o"))

    return candidates.reset_index(drop=True)
