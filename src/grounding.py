"""Entity grounding: resolve and verify hotels, restaurants, attractions and transport against the seed/cache and search providers."""
import copy
import csv
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from src.common import compact_json_for_prompt, estimate_room_count, safe_float, safe_int, safe_text
from src.config import DATA_ROOT, RAW_SEARCH_CACHE_DIR, STRUCTURED_GROUNDING_CACHE_DIR
from src.llm_client import call_generation_json
from src.prompts import (
    build_hotel_grounding_prompt,
    build_online_grounding_prompt,
    build_transport_grounding_prompt,
)

ONLINE_GROUNDING_ENABLED = os.getenv("ENABLE_ONLINE_GROUNDING", "false").lower() in [
    "1",
    "true",
    "yes",
    "y",
]
REQUIRE_ONLINE_GROUNDING = os.getenv("REQUIRE_ONLINE_GROUNDING", "false").lower() in [
    "1",
    "true",
    "yes",
    "y",
]
ONLINE_SEARCH_PROVIDER = os.getenv("ONLINE_SEARCH_PROVIDER", "tavily").lower()
ONLINE_SEARCH_MAX_RESULTS = safe_int(os.getenv("ONLINE_SEARCH_MAX_RESULTS", "3"), 3)
ONLINE_SEARCH_TIMEOUT_SECONDS = safe_int(os.getenv("ONLINE_SEARCH_TIMEOUT_SECONDS", "20"), 20)
ONLINE_GROUNDING_MODEL_NAME = os.getenv(
    "ONLINE_GROUNDING_MODEL_NAME",
    os.getenv(
        "OPENROUTER_GENERATION_MODEL",
        os.getenv(
            "OPENROUTER_MODEL_NAME", os.getenv("GENERATION_MODEL_NAME", "openai/gpt-4o-mini")
        ),
    ),
)
ONLINE_GROUNDING_MAX_TOKENS = safe_int(os.getenv("ONLINE_GROUNDING_MAX_TOKENS", "5000"), 5000)

ONLINE_GROUNDING_CACHE_ENABLED = os.getenv("ONLINE_GROUNDING_CACHE_ENABLED", "true").lower() in [
    "1",
    "true",
    "yes",
    "y",
]
# Structured-snapshot hygiene. The structured cache is meant to be a curated, reproducible
# snapshot of real, sourced places. Without a gate the pipeline auto-writes its own Tavily/
# fallback output (empty names, no source_urls, area-as-name junk) back into it and pollutes
# the snapshot. STRUCTURED_CACHE_AUTOWRITE=false freezes the snapshot (recommended for the main
# experiment after you seed it). STRUCTURED_CACHE_REQUIRE_SOURCE keeps only entries that have a
# concrete name AND an http(s) source_url, so auto-writes can never re-introduce vague rows.
STRUCTURED_CACHE_AUTOWRITE = os.getenv("STRUCTURED_CACHE_AUTOWRITE", "true").lower() in [
    "1",
    "true",
    "yes",
    "y",
]
STRUCTURED_CACHE_REQUIRE_SOURCE = os.getenv("STRUCTURED_CACHE_REQUIRE_SOURCE", "true").lower() in [
    "1",
    "true",
    "yes",
    "y",
]
# Quality gate for auto-cached candidates: drop low-rated Google Places results and low-tier /
# blacklisted hotel names so only reputable, well-rated places sediment into the snapshot.
GOOGLE_PLACES_MIN_RATING = safe_float(os.getenv("GOOGLE_PLACES_MIN_RATING", "4.0"), 4.0)
STRUCTURED_CACHE_MIN_RATING = safe_float(os.getenv("STRUCTURED_CACHE_MIN_RATING", "0"), 0.0)
STRUCTURED_CACHE_HOTEL_NAME_BLACKLIST = [
    t.strip()
    for t in safe_text(
        os.getenv("STRUCTURED_CACHE_HOTEL_NAME_BLACKLIST", "旅馆,招待所,迷你,旅社,青年旅舍,青旅,胶囊,民宿,客栈")
    ).split(",")
    if t.strip()
]
STRUCTURED_CACHE_HOTEL_BRAND_WHITELIST = [
    "万豪",
    "希尔顿",
    "洲际",
    "喜来登",
    "香格里拉",
    "凯宾斯基",
    "丽思卡尔顿",
    "瑞吉",
    "君悦",
    "柏悦",
    "文华东方",
    "半岛",
    "四季",
    "威斯汀",
    "假日",
    "皇冠假日",
    "铂尔曼",
    "索菲特",
    "诺富特",
    "万达",
    "凯悦",
    "安缦",
    "华尔道夫",
    "丽笙",
    "费尔蒙",
    "日航",
    "康莱德",
    "马迭尔",
    "白天鹅",
    "和平饭店",
]
# v53: Tavily/SerpAPI/Bing request failures (quota, HTTP 429/432, timeout)
# should not invalidate a generated candidate. Missing search evidence is
# treated as a soft grounding issue by Module 6/7. Configuration errors still
# raise normally so users know their API key/provider is wrong.
ONLINE_SEARCH_SOFT_FAIL = os.getenv("ONLINE_SEARCH_SOFT_FAIL", "true").lower() in [
    "1",
    "true",
    "yes",
    "y",
]
# v54: Keep Tavily transport grounding day-by-day.  Long multi-stop, multi-day
# queries are a common cause of provider-side rejections such as HTTP 432.
# We still build a full day chain for output, but each Tavily query is limited
# to the current day's key stops.
TRANSPORT_GROUNDING_DAY_BY_DAY = os.getenv("TRANSPORT_GROUNDING_DAY_BY_DAY", "true").lower() in [
    "1",
    "true",
    "yes",
    "y",
]
TRANSPORT_QUERY_MAX_STOPS_PER_DAY = safe_int(os.getenv("TRANSPORT_QUERY_MAX_STOPS_PER_DAY", "3"), 3)
TRANSPORT_QUERY_MAX_CHARS = safe_int(os.getenv("TRANSPORT_QUERY_MAX_CHARS", "180"), 180)
# v55: restaurant grounding is also day-by-day / meal-by-meal.  This prevents
# the system from using one generic city-level restaurant query and then filling
# many meals with "附近中等价位餐厅" placeholders.
RESTAURANT_GROUNDING_DAY_BY_DAY = os.getenv("RESTAURANT_GROUNDING_DAY_BY_DAY", "true").lower() in [
    "1",
    "true",
    "yes",
    "y",
]
RESTAURANT_QUERY_MAX_MEALS = safe_int(os.getenv("RESTAURANT_QUERY_MAX_MEALS", "10"), 10)
RESTAURANT_QUERY_MAX_CHARS = safe_int(os.getenv("RESTAURANT_QUERY_MAX_CHARS", "150"), 150)
CACHE_MAX_RELEVANT_ITEMS = safe_int(os.getenv("ONLINE_GROUNDING_CACHE_MAX_ITEMS", "3"), 3)

# v22: Module 4 stays at rough intent level; Module 5 can pre-ground rough
# day themes/intents into concrete POI candidates with Tavily before asking the
# generation model to write the final itinerary.
MODULE5_ACTIVITY_GROUNDING_BEFORE_GENERATION = os.getenv(
    "MODULE5_ACTIVITY_GROUNDING_BEFORE_GENERATION", "true"
).lower() in ["1", "true", "yes", "y"]
MODULE5_ACTIVITY_GROUNDING_MAX_QUERIES = safe_int(
    os.getenv("MODULE5_ACTIVITY_GROUNDING_MAX_QUERIES", "8"), 8
)
MODULE5_ACTIVITY_GROUNDING_RESULTS_PER_QUERY = safe_int(
    os.getenv("MODULE5_ACTIVITY_GROUNDING_RESULTS_PER_QUERY", "3"), 3
)

# v60: Seed-first grounding. From Module 5 on, every concrete place (attraction /
# restaurant / hotel / transport) is taken from the durable grounding_seed first;
# the online provider (Google Places / Tavily) is only called for a category when
# the seed has too few suitable entries for the destination city. A fresh API
# candidate is evaluated against the quality gate before use, and a passing
# attraction is linked with a restaurant + hotel + transport route into one
# reusable "complete trip unit" that is written back to the seed so coverage grows.
SEED_FIRST_GROUNDING = os.getenv("SEED_FIRST_GROUNDING", "true").lower() in [
    "1",
    "true",
    "yes",
    "y",
]
SEED_FIRST_MIN_CANDIDATES = safe_int(os.getenv("SEED_FIRST_MIN_CANDIDATES", "3"), 3)
SEED_FIRST_MIN_HOTELS = safe_int(os.getenv("SEED_FIRST_MIN_HOTELS", "1"), 1)
SEED_FIRST_MIN_RESTAURANTS = safe_int(os.getenv("SEED_FIRST_MIN_RESTAURANTS", "2"), 2)
SEED_FIRST_MIN_ROUTES = safe_int(os.getenv("SEED_FIRST_MIN_ROUTES", "1"), 1)
# Honesty cap: at most this many meals in one itinerary may stay without a concrete
# restaurant name.  Beyond it, remaining slots are filled by reusing real DB
# restaurants for the city (never fabricated).
MAX_UNGROUNDED_MEALS = safe_int(os.getenv("MAX_UNGROUNDED_MEALS", "2"), 2)
# Budget realism. Family size drives cost: hotel rooms scale with members, and
# meals/tickets/intercity are per-person.  Intercity (round-trip) must be counted.
HOTEL_PEOPLE_PER_ROOM = max(1, safe_int(os.getenv("HOTEL_PEOPLE_PER_ROOM", "3"), 3))
INTERCITY_ROUNDTRIP_DEFAULT_PER_PERSON = safe_float(
    os.getenv("INTERCITY_ROUNDTRIP_DEFAULT_PER_PERSON", "600"), 600.0
)
# When the honest total exceeds budget, a cheaper blacklisted (旅馆/招待所/民宿…)
# hotel may be used as relief, but it carries a quality penalty for Module 6.
BUDGET_ALLOW_LOW_TIER_HOTEL_RELIEF = os.getenv(
    "BUDGET_ALLOW_LOW_TIER_HOTEL_RELIEF", "true"
).lower() in ["1", "true", "yes", "y"]
# When over budget and even the cheapest DB hotel is unaffordable, cap the nightly
# hotel spend to this fraction of the budget (spread over nights and rooms) instead
# of letting one expensive hotel blow the whole budget several times over.
HOTEL_BUDGET_FRACTION = safe_float(os.getenv("HOTEL_BUDGET_FRACTION", "0.4"), 0.4)
# Floor for a per-room nightly budget estimate, so the hotel is never priced absurdly low.
HOTEL_MIN_NIGHTLY = safe_float(os.getenv("HOTEL_MIN_NIGHTLY", "120"), 120.0)
# Target budget utilisation: spend up to ~85% and keep ~15% headroom (the family
# should use most of the budget but not spend to the very limit).  This is both the
# ceiling for downward relief (trim to here when over) and the cap for the upward
# fill below (never push past here, so >=15% always remains).
BUDGET_TARGET_FRACTION = safe_float(os.getenv("BUDGET_TARGET_FRACTION", "0.85"), 0.85)
# Upward budget fill: when a plan leaves too much unused (utilisation below this),
# upgrade it (pricier real hotel, then richer meals) toward BUDGET_TARGET_FRACTION.
BUDGET_FILL_ENABLED = os.getenv("BUDGET_FILL_ENABLED", "true").lower() in ["1", "true", "yes", "y"]
BUDGET_FILL_MIN_UTILIZATION = safe_float(os.getenv("BUDGET_FILL_MIN_UTILIZATION", "0.80"), 0.80)
# When still over budget after sizing the hotel, swap the most expensive paid
# attractions that are NOT a member's must-include for free real attractions.
ATTRACTION_BUDGET_RELIEF = os.getenv("ATTRACTION_BUDGET_RELIEF", "true").lower() in [
    "1",
    "true",
    "yes",
    "y",
]
# 替换餐厅: when over budget, the family eats more economically.  Per-meal per-person
# spend is reduced toward an economical floor so the meal budget fits — the cheaper
# spend is shown on each meal (no hidden numbers).
BUDGET_MEAL_RELIEF = os.getenv("BUDGET_MEAL_RELIEF", "true").lower() in ["1", "true", "yes", "y"]
ECONOMICAL_MEAL_PER_PERSON = safe_float(os.getenv("ECONOMICAL_MEAL_PER_PERSON", "30"), 30.0)
# Nightly-price estimate (CNY) by hotel price level, used only when a hotel record
# carries no concrete nightly price — so the hotel is always counted in the budget.
_HOTEL_PRICE_BY_LEVEL = {
    "budget": safe_float(os.getenv("HOTEL_PRICE_BUDGET", "350"), 350.0),
    "low": safe_float(os.getenv("HOTEL_PRICE_BUDGET", "350"), 350.0),
    "economy": safe_float(os.getenv("HOTEL_PRICE_BUDGET", "350"), 350.0),
    "medium": safe_float(os.getenv("HOTEL_PRICE_MEDIUM", "600"), 600.0),
    "mid": safe_float(os.getenv("HOTEL_PRICE_MEDIUM", "600"), 600.0),
    "high": safe_float(os.getenv("HOTEL_PRICE_HIGH", "1000"), 1000.0),
    "luxury": safe_float(os.getenv("HOTEL_PRICE_LUXURY", "1800"), 1800.0),
}

# Structured cache files live under data/cache/online_grounding/structured/.
# They store reusable online-derived facts instead of free-form logs:
# hotels.json, restaurants.json, attractions.json, transport_routes.json.
STRUCTURED_CACHE_FILES = {
    "hotels": STRUCTURED_GROUNDING_CACHE_DIR / "hotels.json",
    "restaurants": STRUCTURED_GROUNDING_CACHE_DIR / "restaurants.json",
    "attractions": STRUCTURED_GROUNDING_CACHE_DIR / "attractions.json",
    "transport_routes": STRUCTURED_GROUNDING_CACHE_DIR / "transport_routes.json",
}


def _ensure_cache_dirs() -> None:
    if not ONLINE_GROUNDING_CACHE_ENABLED:
        return
    RAW_SEARCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    STRUCTURED_GROUNDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _stable_cache_key(value: Any) -> str:
    text = (
        json.dumps(value, ensure_ascii=False, sort_keys=True)
        if not isinstance(value, str)
        else value
    )
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _read_json_file(path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default


def _write_json_file(path, data) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # Cache failure should never break itinerary generation.
        pass


# ---------------------------------------------------------------------------
# Unified grounding_seed CSV store (v61).  grounding_seed/ is now the single
# durable database: hotels.csv / restaurants.csv / attractions.csv /
# transportation.csv, each row carrying a unique source_id.  The pipeline reads
# this store first and only calls the online provider when it lacks a match;
# qualified online results are upserted back here (deduplicated by entity), so no
# duplicate entity is ever written for the same city+name.
# ---------------------------------------------------------------------------
_SEED_CSV_META = {
    "hotels": {
        "file": "hotels.csv",
        "prefix": "hotel_",
        "header": [
            "source_id",
            "name",
            "city",
            "address",
            "latitude",
            "longitude",
            "price_per_night",
            "rating",
            "check_in",
            "check_out",
            "facilities",
            "family_friendly",
            "nearby_metro_station",
            "price_level",
            "place_id",
            "source_url",
            "seed_cache_key",
        ],
    },
    "restaurants": {
        "file": "restaurants.csv",
        "prefix": "rest_",
        "header": [
            "source_id",
            "name",
            "city",
            "address",
            "latitude",
            "longitude",
            "avg_price",
            "price_range_per_person",
            "open_time",
            "close_time",
            "cuisine",
            "dietary_tags",
            "child_friendly",
            "meal_hint",
            "nearby_metro_station",
            "place_id",
            "source_url",
            "seed_cache_key",
        ],
    },
    "attractions": {
        "file": "attractions.csv",
        "prefix": "attr_",
        "header": [
            "source_id",
            "name",
            "city",
            "address",
            "latitude",
            "longitude",
            "ticket_price",
            "open_time",
            "close_time",
            "opening_hours_raw",
            "visit_duration_min",
            "tags",
            "walking_intensity",
            "place_id",
            "source_url",
            "seed_cache_key",
        ],
    },
    "transport_routes": {
        "file": "transportation.csv",
        "prefix": "trans_",
        "header": [
            "source_id",
            "city",
            "origin",
            "destination",
            "origin_city",
            "route",
            "mode",
            "duration_min",
            "cost_total",
            "cost_per_person",
            "day_hint",
            "source_url",
            "seed_cache_key",
        ],
    },
}
_SEED_CSV_CACHE: Dict[str, Any] = {}


def _seed_csv_path(category: str):
    meta = _SEED_CSV_META.get(category)
    return (DATA_ROOT / "grounding_seed" / meta["file"]) if meta else None


def _maps_url_from_place_id(place_id: str) -> str:
    place_id = safe_text(place_id)
    return f"https://www.google.com/maps/place/?q=place_id:{place_id}" if place_id else ""


def _csv_row_to_record(category: str, row: Dict[str, Any]) -> Dict[str, Any]:
    """Map a CSV seed row back to the internal dict the pipeline already expects."""
    g = lambda k: safe_text(row.get(k))
    src = g("source_url") or _maps_url_from_place_id(g("place_id"))
    source_urls = [src] if src else []
    common = {
        "destination_city": g("city"),
        "category": category,
        "source_urls": source_urls,
        "place_id": g("place_id"),
        "latitude": g("latitude"),
        "longitude": g("longitude"),
        "source_id": g("source_id"),
        "cache_key": g("seed_cache_key"),
    }
    if category == "hotels":
        common.update(
            {
                "hotel_name": g("name"),
                "area": g("address"),
                "address_or_location": g("address"),
                "estimated_price_per_night": safe_float(row.get("price_per_night"), 0.0),
                "rating": safe_float(row.get("rating"), 0.0) or None,
                "nearby_metro_station": g("nearby_metro_station"),
                "price_level": g("price_level"),
                "available_lines": [x for x in g("facilities").split(";") if x],
                "family_suitability": g("family_friendly"),
                "check_in": g("check_in"),
                "check_out": g("check_out"),
            }
        )
    elif category == "restaurants":
        common.update(
            {
                "restaurant_name": g("name"),
                "area": g("address"),
                "restaurant_type": g("cuisine"),
                "estimated_price_per_person": safe_float(row.get("avg_price"), 0.0),
                "price_range_per_person": g("price_range_per_person"),
                "nearby_metro_station": g("nearby_metro_station"),
                "meal_hint": g("meal_hint"),
                "dietary_tags": g("dietary_tags"),
                "child_friendly": g("child_friendly"),
                "open_time": g("open_time"),
                "close_time": g("close_time"),
            }
        )
    elif category == "attractions":
        common.update(
            {
                "poi_name": g("name"),
                "location_or_area": g("address"),
                "ticket_price_per_person": safe_float(row.get("ticket_price"), 0.0),
                "opening_hours": g("opening_hours_raw") or g("open_time"),
                "open_time": g("open_time"),
                "close_time": g("close_time"),
                "visit_duration_min": g("visit_duration_min"),
                "tags": g("tags"),
                "walking_intensity": g("walking_intensity"),
            }
        )
    elif category == "transport_routes":
        common.update(
            {
                "origin": g("origin"),
                "destination": g("destination"),
                "origin_city": g("origin_city"),
                "route": g("route"),
                "mode": g("mode"),
                "estimated_duration_minutes": safe_float(row.get("duration_min"), 0.0),
                "estimated_cost_total": safe_float(row.get("cost_total"), 0.0),
                "estimated_cost_per_person": safe_float(row.get("cost_per_person"), 0.0),
                "day_hint": safe_int(row.get("day_hint"), 0),
            }
        )
    return common


def _record_to_csv_row(category: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """Map an internal record (LLM/online-derived) to a CSV seed row."""
    srcs = (
        record.get("source_urls")
        if isinstance(record.get("source_urls"), list)
        else ([record.get("source_urls")] if record.get("source_urls") else [])
    )
    http_src = next(
        (safe_text(u) for u in srcs if safe_text(u).startswith(("http://", "https://"))), ""
    )
    pid = safe_text(record.get("place_id"))
    if not pid:
        m = re.search(r"place_id:([A-Za-z0-9_\-]+)", http_src)
        pid = m.group(1) if m else ""

    def num(v):
        f = safe_float(v, -1.0)
        if f < 0:
            return ""
        return str(int(f)) if f == int(f) else str(round(f, 2))

    base = {
        "city": safe_text(record.get("destination_city")),
        "place_id": pid,
        "source_url": http_src,
        "latitude": safe_text(record.get("latitude")),
        "longitude": safe_text(record.get("longitude")),
        "seed_cache_key": safe_text(record.get("cache_key")),
    }
    if category == "hotels":
        base.update(
            {
                "name": safe_text(record.get("hotel_name")),
                "address": safe_text(record.get("address_or_location"))
                or safe_text(record.get("area")),
                "price_per_night": num(record.get("estimated_price_per_night")),
                "rating": num(record.get("rating")),
                "facilities": ";".join(record.get("available_lines"))
                if isinstance(record.get("available_lines"), list)
                else safe_text(record.get("available_lines")),
                "family_friendly": safe_text(record.get("family_suitability")),
                "nearby_metro_station": safe_text(record.get("nearby_metro_station")),
                "price_level": safe_text(record.get("price_level")),
                "check_in": safe_text(record.get("check_in")),
                "check_out": safe_text(record.get("check_out")),
            }
        )
    elif category == "restaurants":
        base.update(
            {
                "name": safe_text(record.get("restaurant_name")),
                "address": safe_text(record.get("area")),
                "avg_price": num(record.get("estimated_price_per_person")),
                "price_range_per_person": safe_text(record.get("price_range_per_person")),
                "open_time": safe_text(record.get("open_time")),
                "close_time": safe_text(record.get("close_time")),
                "cuisine": safe_text(record.get("restaurant_type")),
                "dietary_tags": safe_text(record.get("dietary_tags")),
                "child_friendly": safe_text(record.get("child_friendly"))
                or safe_text(record.get("family_suitability")),
                "meal_hint": safe_text(record.get("meal_hint")),
                "nearby_metro_station": safe_text(record.get("nearby_metro_station")),
            }
        )
    elif category == "attractions":
        base.update(
            {
                "name": safe_text(record.get("poi_name")),
                "address": safe_text(record.get("location_or_area")),
                "ticket_price": num(record.get("ticket_price_per_person")),
                "open_time": safe_text(record.get("open_time")),
                "close_time": safe_text(record.get("close_time")),
                "opening_hours_raw": safe_text(record.get("opening_hours")),
                "visit_duration_min": safe_text(record.get("visit_duration_min")),
                "tags": safe_text(record.get("tags")),
                "walking_intensity": safe_text(record.get("walking_intensity")),
            }
        )
    elif category == "transport_routes":
        base.update(
            {
                "origin": safe_text(record.get("origin")),
                "destination": safe_text(record.get("destination")),
                "origin_city": safe_text(record.get("origin_city")),
                "route": safe_text(record.get("route")),
                "mode": safe_text(record.get("mode")),
                "duration_min": num(record.get("estimated_duration_minutes")),
                "cost_total": num(record.get("estimated_cost_total")),
                "cost_per_person": num(record.get("estimated_cost_per_person")),
                "day_hint": num(record.get("day_hint")),
            }
        )
    return base


def _read_seed_csv_rows(category: str) -> List[Dict[str, Any]]:
    """Raw CSV rows for a category, memoized by file mtime."""
    path = _seed_csv_path(category)
    if not path or not path.exists():
        return []
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0
    cached = _SEED_CSV_CACHE.get(category)
    if cached and cached[0] == mtime:
        return cached[1]
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = [dict(r) for r in csv.DictReader(f)]
    except Exception:
        rows = []
    _SEED_CSV_CACHE[category] = (mtime, rows)
    return rows


def _load_category_cache(category: str) -> List[Dict[str, Any]]:
    """Durable records for a category.

    Seed CSV (grounding_seed/<table>.csv) is the primary store.  The legacy
    structured JSON cache is consulted only when the CSV is missing/empty, so the
    transition is non-breaking if a CSV has not been generated yet.
    """
    rows = _read_seed_csv_rows(category)
    if rows:
        return [_csv_row_to_record(category, r) for r in rows]
    path = STRUCTURED_CACHE_FILES.get(category)
    if not path:
        return []
    data = _read_json_file(path, [])
    return data if isinstance(data, list) else []


def _record_key(category: str, record: Dict[str, Any]) -> str:
    if category == "hotels":
        parts = [record.get("destination_city"), record.get("area"), record.get("hotel_name")]
    elif category == "restaurants":
        parts = [
            record.get("destination_city"),
            record.get("area"),
            record.get("restaurant_name") or record.get("restaurant_type"),
            record.get("price_range_per_person"),
        ]
    elif category == "attractions":
        parts = [record.get("destination_city"), record.get("poi_name")]
    elif category == "transport_routes":
        parts = [
            record.get("destination_city"),
            record.get("origin"),
            record.get("destination"),
            record.get("route"),
        ]
    else:
        parts = [record]
    return _stable_cache_key("|".join([safe_text(x).lower() for x in parts]))


def _record_is_snapshot_worthy(category: str, record: Dict[str, Any]) -> bool:
    """Only concrete, (optionally) sourced records belong in the curated snapshot.

    This stops the auto-writer from re-introducing empty-name / no-source / area-as-name junk
    that previously polluted the structured cache.
    """
    if not isinstance(record, dict):
        return False
    if category == "restaurants":
        name = safe_text(record.get("restaurant_name"))
    elif category == "hotels":
        name = safe_text(record.get("hotel_name"))
    elif category == "attractions":
        name = safe_text(record.get("poi_name"))
    elif category == "transport_routes":
        name = safe_text(record.get("route")) or (
            safe_text(record.get("origin"))
            and safe_text(record.get("destination"))
            and f"{record.get('origin')}->{record.get('destination')}"
        )
    else:
        name = ""
    if not name:
        return False
    # Name blacklist: drop low-tier hotel names (旅馆/招待所/迷你/民宿…) unless a known brand is present.
    if (
        category == "hotels"
        and any(b in name for b in STRUCTURED_CACHE_HOTEL_NAME_BLACKLIST)
        and not any(br in name for br in STRUCTURED_CACHE_HOTEL_BRAND_WHITELIST)
    ):
        return False
    # Rating gate: when the record carries a numeric Google rating, require it to clear the bar.
    rating = record.get("rating")
    if (
        STRUCTURED_CACHE_MIN_RATING > 0
        and isinstance(rating, (int, float))
        and float(rating) < STRUCTURED_CACHE_MIN_RATING
    ):
        return False
    if STRUCTURED_CACHE_REQUIRE_SOURCE and category != "transport_routes":
        sources = record.get("source_urls")
        sources = sources if isinstance(sources, list) else ([sources] if sources else [])
        if not any(safe_text(u).startswith(("http://", "https://")) for u in sources):
            return False
    return True


def _upsert_structured_cache(category: str, record: Dict[str, Any]) -> None:
    if not ONLINE_GROUNDING_CACHE_ENABLED or not isinstance(record, dict):
        return
    if not STRUCTURED_CACHE_AUTOWRITE:
        return
    if not _record_is_snapshot_worthy(category, record):
        return
    path = STRUCTURED_CACHE_FILES.get(category)
    if not path:
        return
    _ensure_cache_dirs()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    record = dict(record)
    record["category"] = category
    record["cache_updated_at"] = now
    key = _record_key(category, record)
    record["cache_key"] = key
    items = _load_category_cache(category)
    by_key = {safe_text(item.get("cache_key")): item for item in items if isinstance(item, dict)}
    old = by_key.get(key, {}) if isinstance(by_key.get(key), dict) else {}
    merged = {**old, **record}
    merged["cache_hit_count"] = safe_int(old.get("cache_hit_count"), 0)
    by_key[key] = merged
    _write_json_file(path, list(by_key.values()))
    _sync_record_to_seed(category, merged, key)


def _sync_record_to_seed(category: str, record: Dict[str, Any], key: str) -> None:
    """Upsert a quality-gated record into the grounding_seed CSV store.

    Deduplicated by entity (city+name via _record_key): an existing row for the
    same entity is updated in place (non-empty new fields win, others preserved)
    and keeps its source_id; a new entity gets the next source_id in the table.
    This guarantees no duplicate entity is ever written to the same file.
    """
    meta = _SEED_CSV_META.get(category)
    if not meta:
        return
    path = _seed_csv_path(category)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = list(_read_seed_csv_rows(category))  # copy of memoized rows
        # Locate an existing row for this entity and the current max numeric id.
        match_idx = None
        max_idx = 0
        for i, row in enumerate(rows):
            sid = safe_text(row.get("source_id"))
            m = re.search(r"(\d+)$", sid)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
            if _record_key(category, _csv_row_to_record(category, row)) == key:
                match_idx = i
        new_row = _record_to_csv_row(category, record)
        if not safe_text(new_row.get("seed_cache_key")):
            new_row["seed_cache_key"] = key
        if match_idx is not None:
            old_row = rows[match_idx]
            merged = dict(old_row)
            for col in meta["header"]:
                val = safe_text(new_row.get(col))
                if val:
                    merged[col] = val
            merged["source_id"] = (
                safe_text(old_row.get("source_id")) or f"{meta['prefix']}{match_idx + 1:04d}"
            )
            rows[match_idx] = merged
        else:
            new_row["source_id"] = f"{meta['prefix']}{max_idx + 1:04d}"
            rows.append(new_row)
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=meta["header"], extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in meta["header"]})
        _SEED_CSV_CACHE.pop(category, None)  # force reload on next read
    except Exception:
        # Seed growth must never break itinerary generation.
        pass


_CITY_ALIAS = {
    "香港": "hong kong",
    "上海": "shanghai",
    "北京": "beijing",
    "深圳": "shenzhen",
    "广州": "guangzhou",
    "哈尔滨": "harbin",
    "南京": "nanjing",
    "杭州": "hangzhou",
    "成都": "chengdu",
    "重庆": "chongqing",
    "武汉": "wuhan",
    "西安": "xi'an",
    "厦门": "xiamen",
    "长沙": "changsha",
    "青岛": "qingdao",
    "大连": "dalian",
    "天津": "tianjin",
    "珠海": "zhuhai",
    "海口": "haikou",
    "桂林": "guilin",
    "南昌": "nanchang",
    "南宁": "nanning",
    "贵阳": "guiyang",
    "福州": "fuzhou",
    "丽江": "lijiang",
    "宁波": "ningbo",
    "沈阳": "shenyang",
    "太原": "taiyuan",
    "威海": "weihai",
    "开封": "kaifeng",
    "郑州": "zhengzhou",
}


def _cache_record_matches_destination(record: Dict[str, Any], destination: str) -> bool:
    dest = safe_text(destination).lower()
    if not dest:
        return True
    record_dest = safe_text(record.get("destination_city")).lower()
    if not record_dest:
        return True
    if dest in record_dest or record_dest in dest:
        return True
    dest_en = _CITY_ALIAS.get(dest, "")
    if dest_en and (dest_en in record_dest or record_dest in dest_en):
        return True
    record_en = record_dest
    for cn, en in _CITY_ALIAS.items():
        if en == record_en or record_en in en:
            if cn in dest or dest in cn:
                return True
    return False


def _destination_cn_tokens(destination: Any) -> List[str]:
    """Chinese city tokens for a destination (which may be given in English)."""
    d = safe_text(destination).strip()
    dl = d.lower()
    out = set()
    if re.search(r"[一-鿿]", d):
        out.add(d)
    for cn, en in _CITY_ALIAS.items():
        if cn == d or cn in d or en == dl or (en and (en in dl or dl in en)):
            out.add(cn)
    return [t for t in out if t]


def _online_result_in_city(content: Any, destination: Any) -> bool:
    """True if a Google Places result's address/content places it in the target city.

    Filters cross-city / cross-country false matches (e.g. '重庆 海洋馆' returning San
    Diego SeaWorld) so they are neither used nor written into the database."""
    tokens = _destination_cn_tokens(destination)
    if not tokens:
        return True  # unknown city -> cannot validate, do not block
    c = safe_text(content)
    return any(tok in c for tok in tokens)


# ---------------------------------------------------------------------------
# Seed-first grounding (v60): prefer durable grounding_seed entries; evaluate a
# fresh online candidate before use; link a passing attraction with a restaurant,
# hotel and transport route into one reusable "complete trip unit".
# ---------------------------------------------------------------------------


def _candidate_display_name(category: str, record: Dict[str, Any]) -> str:
    if not isinstance(record, dict):
        return ""
    if category == "restaurants":
        return safe_text(record.get("restaurant_name"))
    if category == "hotels":
        return safe_text(record.get("hotel_name"))
    if category == "attractions":
        return safe_text(record.get("poi_name"))
    if category == "transport_routes":
        return safe_text(record.get("route")) or (
            f"{safe_text(record.get('origin'))}->{safe_text(record.get('destination'))}"
            if safe_text(record.get("origin")) and safe_text(record.get("destination"))
            else ""
        )
    return ""


def _seed_candidates_for_city(category: str, destination: str) -> List[Dict[str, Any]]:
    """All durable seed/snapshot records of a category that match the city.

    Reads through _load_category_cache, which already falls back to grounding_seed
    when the (wipeable) structured cache is empty.  Only records with a concrete
    name are returned, so vague placeholder rows never count toward seed coverage.
    """
    out: List[Dict[str, Any]] = []
    for r in _load_category_cache(category):
        if not isinstance(r, dict) or not _cache_record_matches_destination(r, destination):
            continue
        if not _candidate_display_name(category, r):
            continue
        out.append(r)
    return out


def evaluate_grounding_candidate(category: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """Quality-gate a fresh online candidate before it is linked / saved.

    Checks: concrete name, Google rating bar (only when a numeric rating exists),
    hotel name blacklist, and an http(s) source.  Returns {passed, reasons, name}
    so the run log can show exactly why a candidate was accepted or rejected.
    """
    reasons: List[str] = []
    name = _candidate_display_name(category, record)
    if not name:
        reasons.append("missing_concrete_name")
    rating = record.get("rating")
    if (
        isinstance(rating, (int, float))
        and float(rating) > 0
        and GOOGLE_PLACES_MIN_RATING > 0
        and float(rating) < GOOGLE_PLACES_MIN_RATING
    ):
        reasons.append(f"rating_below_{GOOGLE_PLACES_MIN_RATING}:{rating}")
    if (
        category == "hotels"
        and name
        and any(b in name for b in STRUCTURED_CACHE_HOTEL_NAME_BLACKLIST)
        and not any(br in name for br in STRUCTURED_CACHE_HOTEL_BRAND_WHITELIST)
    ):
        reasons.append("hotel_name_blacklisted")
    if STRUCTURED_CACHE_REQUIRE_SOURCE and category != "transport_routes":
        srcs = record.get("source_urls")
        srcs = srcs if isinstance(srcs, list) else ([srcs] if srcs else [])
        if not any(safe_text(u).startswith(("http://", "https://")) for u in srcs):
            reasons.append("missing_http_source")
    passed = (not reasons) and _record_is_snapshot_worthy(category, record)
    return {"passed": bool(passed), "reasons": reasons, "name": name}


def _pick_seed_component(
    category: str, destination: str, budget_limit: float = 0.0, days: int = 1, members: int = 3
) -> Optional[Dict[str, Any]]:
    """Best single durable-seed record of a category for a city.

    Used to attach a restaurant / hotel / transport route to a freshly grounded
    attraction.  Prefers budget-fitting, sourced, previously-hit records.
    """
    best: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    for r in _seed_candidates_for_city(category, destination):
        if not _record_is_snapshot_worthy(category, r):
            continue
        score = 0.0
        price = _price_value_from_cache(r, category)
        if price is not None:
            score += 5
            if budget_limit > 0 and _price_fits_budget(
                price, category, budget_limit, days, members
            ):
                score += 8
        srcs = r.get("source_urls")
        if isinstance(srcs, list) and any(
            safe_text(u).startswith(("http://", "https://")) for u in srcs
        ):
            score += 4
        score += safe_int(r.get("cache_hit_count"), 0)
        if best_score is None or score > best_score:
            best, best_score = r, score
    return best


def _save_trip_unit_to_seed(unit: Dict[str, Any]) -> None:
    """Persist one complete trip unit into grounding_seed/trip_units.json (city+attraction keyed)."""
    seed_path = DATA_ROOT / "grounding_seed" / "trip_units.json"
    try:
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        existing = _read_json_file(seed_path, [])
        if not isinstance(existing, list):
            existing = []

        def _unit_key(u: Dict[str, Any]) -> str:
            return _stable_cache_key(
                "|".join(
                    [
                        safe_text(u.get("destination_city")).lower(),
                        _candidate_display_name(
                            "attractions",
                            u.get("attraction", {})
                            if isinstance(u.get("attraction"), dict)
                            else {},
                        ).lower(),
                    ]
                )
            )

        by_key: Dict[str, Dict[str, Any]] = {}
        for it in existing:
            if isinstance(it, dict):
                by_key[safe_text(it.get("unit_key")) or _unit_key(it)] = it
        key = _unit_key(unit)
        unit = dict(unit)
        unit["unit_key"] = key
        by_key[key] = {**(by_key.get(key) or {}), **unit}
        _write_json_file(seed_path, list(by_key.values()))
    except Exception:
        # Seed growth must never break itinerary generation.
        pass


def assemble_complete_trip_unit(
    destination: str,
    attraction: Dict[str, Any],
    family_agents: List[Dict[str, Any]],
    requirements: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Link a gate-passing attraction with a restaurant, hotel and transport route.

    Components are taken seed-first so bundling stays cheap and deterministic; the
    assembled unit (and the attraction itself, via _upsert_structured_cache at the
    call site) are written back to grounding_seed so the seed grows into reusable
    complete trips instead of disconnected POIs.
    """
    requirements = requirements or {}
    budget = safe_float(requirements.get("budget_limit") or requirements.get("budget"), 0.0)
    days = max(1, safe_int(requirements.get("days") or requirements.get("day"), 1))
    members = max(1, safe_int(requirements.get("member_count"), len(family_agents or []) or 3))
    if not _candidate_display_name("attractions", attraction):
        return None
    restaurant = _pick_seed_component("restaurants", destination, budget, days, members)
    hotel = _pick_seed_component("hotels", destination, budget, days, members)
    transport = _pick_seed_component("transport_routes", destination, budget, days, members)
    unit = {
        "destination_city": safe_text(attraction.get("destination_city")) or safe_text(destination),
        "attraction": attraction,
        "restaurant": restaurant,
        "hotel": hotel,
        "transport_route": transport,
        "complete": bool(restaurant and hotel and transport),
        "missing_components": [
            name
            for name, val in [
                ("restaurant", restaurant),
                ("hotel", hotel),
                ("transport_route", transport),
            ]
            if not val
        ],
        "assembled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_trip_unit_to_seed(unit)
    return unit


def _text_tokens(value: Any) -> set:
    text = safe_text(value).lower()
    if not text:
        return set()
    tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{2,}|[\u4e00-\u9fff]{2,}", text))
    stop = {
        "hotel",
        "restaurant",
        "family",
        "friendly",
        "near",
        "area",
        "price",
        "route",
        "travel",
        "景点",
        "酒店",
        "餐厅",
        "交通",
        "附近",
        "家庭",
        "价格",
    }
    return {t for t in tokens if t not in stop and len(t) >= 2}


def _price_value_from_cache(item: Dict[str, Any], category: str) -> Optional[float]:
    keys = [
        "estimated_price_per_person",
        "price_per_person",
        "estimated_price_per_night",
        "ticket_price_per_person",
        "estimated_cost_total",
        "estimated_cost",
    ]
    for k in keys:
        if k in item:
            val = _money(item.get(k), default=-1)
            if val is not None and val >= 0:
                return float(val)
    text = safe_text(item.get("price_range_per_person") or item.get("price_range") or "")
    if text:
        val = _money(text, default=-1)
        if val >= 0:
            return float(val)
    return None


def _price_fits_budget(
    price: Optional[float], category: str, budget_limit: float, days: int, member_count: int
) -> bool:
    if price is None:
        return False
    if price < 0:
        return False
    # Conservative category ranges.  They are not exact quotes, only suitability
    # filters so a luxury hotel/meal does not dominate a strict budget plan.
    if category == "restaurants":
        return price <= max(220, budget_limit / max(1, days * max(1, member_count) * 2) * 1.6)
    if category == "hotels":
        return price <= max(1200, budget_limit / max(1, days - 1) * 0.7)
    if category == "attractions":
        return price <= max(450, budget_limit / max(1, member_count * days) * 0.8)
    if category == "transport_routes":
        return price <= max(300, budget_limit / max(1, days) * 0.35)
    return True


def score_grounding_cache_item(
    item: Dict[str, Any], category: str, requirements: Dict[str, Any]
) -> float:
    """Score structured online cache by city/type/area/price/completeness.\n\n    v49 change: prior versions mostly loaded the first same-city records.  This
    score ensures Module 5 receives the most useful cached hotel/restaurant/POI/
    route facts and does not waste prompt space on vague or mismatched rows.
    """
    if not isinstance(item, dict):
        return 0.0
    score = 0.0
    destination = safe_text(requirements.get("destination_city")).lower()
    record_dest = safe_text(item.get("destination_city")).lower()
    if destination and record_dest and (destination in record_dest or record_dest in destination):
        score += 30
    elif destination and record_dest:
        score -= 40

    if safe_text(item.get("category")) == category or category in STRUCTURED_CACHE_FILES:
        score += 20

    ticket_req = (
        requirements.get("ticket_requirements", {})
        if isinstance(requirements.get("ticket_requirements"), dict)
        else {}
    )
    pois = (
        ticket_req.get("pois_to_check", [])
        if isinstance(ticket_req.get("pois_to_check"), list)
        else []
    )
    itinerary_text = json.dumps(
        requirements.get("itinerary") or requirements.get("per_day_itinerary") or requirements,
        ensure_ascii=False,
        default=str,
    )
    area_context = " ".join([safe_text(x) for x in pois]) + " " + itinerary_text
    area_tokens = _text_tokens(area_context)
    item_area_text = " ".join(
        [
            safe_text(item.get("area")),
            safe_text(item.get("location_or_area")),
            safe_text(item.get("address_or_location")),
            safe_text(item.get("nearby_metro_station")),
            safe_text(item.get("origin")),
            safe_text(item.get("destination")),
            safe_text(item.get("route")),
            safe_text(item.get("poi_name")),
            safe_text(item.get("restaurant_name")),
            safe_text(item.get("hotel_name")),
        ]
    )
    item_tokens = _text_tokens(item_area_text)
    if area_tokens and item_tokens:
        overlap = len(area_tokens & item_tokens)
        score += min(20, overlap * 4)

    # Attraction cache should match at least one requested POI when POIs exist.
    if category == "attractions" and pois:
        name = safe_text(item.get("poi_name")).lower()
        if name and any(
            (safe_text(p).lower() in name or name in safe_text(p).lower()) for p in pois
        ):
            score += 25
        else:
            score -= 25

    # Transport route should connect names that appear in the current itinerary.
    if category == "transport_routes":
        route_text = (
            safe_text(item.get("route") or "")
            + " "
            + safe_text(item.get("origin"))
            + " "
            + safe_text(item.get("destination"))
        )
        route_tokens = _text_tokens(route_text)
        if area_tokens and route_tokens:
            score += min(18, len(area_tokens & route_tokens) * 3)

    budget_limit = safe_float(requirements.get("budget_limit") or requirements.get("budget"), 0.0)
    days = max(1, safe_int(requirements.get("days") or requirements.get("day"), 1))
    member_count = max(1, safe_int(requirements.get("member_count"), 3))
    price = _price_value_from_cache(item, category)
    if price is not None:
        score += 10
        if budget_limit > 0 and _price_fits_budget(
            price, category, budget_limit, days, member_count
        ):
            score += 12
        elif budget_limit > 0:
            score -= 16
    else:
        score -= 10

    useful_by_category = {
        "hotels": [
            "hotel_name",
            "area",
            "estimated_price_per_night",
            "nearby_metro_station",
            "source_urls",
        ],
        "restaurants": [
            "restaurant_name",
            "restaurant_type",
            "area",
            "price_range_per_person",
            "estimated_price_per_person",
            "source_urls",
        ],
        "attractions": ["poi_name", "location_or_area", "ticket_price_per_person", "source_urls"],
        "transport_routes": [
            "origin",
            "destination",
            "route",
            "mode",
            "estimated_duration_minutes",
            "estimated_cost_total",
        ],
    }
    filled = sum(1 for k in useful_by_category.get(category, []) if safe_text(item.get(k)))
    score += filled * 3

    urls = item.get("source_urls")
    if isinstance(urls, list) and urls:
        score += 6
    elif safe_text(urls):
        score += 4

    text = json.dumps(item, ensure_ascii=False, default=str).lower()
    if "未知" in text or "unknown" in text:
        score -= 12
    if category == "restaurants" and any(
        x in text for x in ["家庭友好 dinner", "其他餐厅", 'restaurant_name": ""']
    ):
        score -= 15
    if category == "hotels" and any(x in text for x in ["酒店名称", "真实酒店名称", "未注明酒店"]):
        score -= 18

    # Prefer reusable records that were actually hit before, but not too much.
    score += min(5, safe_int(item.get("cache_hit_count"), 0))
    return round(float(score), 4)


def load_relevant_grounding_cache(requirements: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Load and score reusable structured grounding facts.

    v49 change: use relevance scoring instead of taking the first same-city cache
    rows.  Criteria: city, category, area/POI overlap, price suitability, field
    completeness, source URLs, and penalties for unknown/vague rows.
    """
    if not ONLINE_GROUNDING_CACHE_ENABLED:
        return {"hotels": [], "restaurants": [], "attractions": [], "transport_routes": []}
    destination = safe_text(requirements.get("destination_city"))
    output: Dict[str, List[Dict[str, Any]]] = {k: [] for k in STRUCTURED_CACHE_FILES}
    for category in output:
        scored: List[Dict[str, Any]] = []
        for item in _load_category_cache(category):
            if not isinstance(item, dict) or not _cache_record_matches_destination(
                item, destination
            ):
                continue
            score = score_grounding_cache_item(
                item=item, category=category, requirements=requirements
            )
            if score <= 0:
                continue
            enriched = dict(item)
            enriched["cache_relevance_score"] = score
            scored.append(enriched)
        scored.sort(key=lambda x: x.get("cache_relevance_score", 0.0), reverse=True)
        output[category] = scored[:CACHE_MAX_RELEVANT_ITEMS]
    return output


def _cache_items_to_search_results(
    items: List[Dict[str, Any]], category: str
) -> List[Dict[str, Any]]:
    results = []
    for item in items[:CACHE_MAX_RELEVANT_ITEMS]:
        title_parts = [f"本地缓存:{category}"]
        for k in ["hotel_name", "restaurant_name", "restaurant_type", "poi_name", "route"]:
            if safe_text(item.get(k)):
                title_parts.append(safe_text(item.get(k)))
                break
        # Surface the snapshot record's real http source URL so plan-level grounding counts a
        # verifiable web source, instead of the synthetic "local:" path the evaluator ignores.
        # Without this the curated snapshot never registers as a real-world source.
        raw_sources = item.get("source_urls") if isinstance(item.get("source_urls"), list) else []
        http_sources = [
            safe_text(u) for u in raw_sources if safe_text(u).startswith(("http://", "https://"))
        ]
        results.append(
            {
                "title": " - ".join(title_parts),
                "url": http_sources[0]
                if http_sources
                else f"local:data/cache/online_grounding/structured/{category}.json",
                "content": json.dumps(item, ensure_ascii=False, sort_keys=True),
                "source_urls": http_sources,
                "provider": "local_structured_cache",
                "cache_hit": True,
            }
        )
    return results


def structured_cache_as_search_results(
    cache: Dict[str, List[Dict[str, Any]]], category: str
) -> List[Dict[str, Any]]:
    return _cache_items_to_search_results(cache.get(category, []), category)


GENERIC_HOTEL_TERMS = [
    "family-friendly hotel",
    "family friendly hotel",
    "hotel",
    "convenient area",
    "near main route",
    "住宿未注明",
    "家庭友好型酒店",
    "交通便利区域",
    "合适酒店",
    "真实酒店名称",
    "酒店所在商圈或地铁站附近区域",
    "未注明酒店",
]
GENERIC_TRANSPORT_TERMS = [
    "metro/taxi",
    "metro/taxi as needed",
    "taxi as needed",
    "public transport",
    "地铁/出租车",
    "地铁或出租车",
    "公共交通",
    "交通未注明",
    "短途打车",
    "具体地铁路线",
    "酒店附近地铁站",
    "某号线",
    "某站",
]


def _money(value: Any, default: float = 0.0) -> float:
    """Convert money-like values from LLM/grounding JSON into a float.

    Online grounding responses may contain strings such as "¥350", "HK$1,200",
    "100-200", "about 80 yuan", or already-numeric values.  This helper keeps
    cost calculation robust and prevents grounding from failing only because the
    model returned a formatted price.
    """
    if value is None:
        return float(default)
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return float(default)

    text = safe_text(value)
    if not text:
        return float(default)

    # Normalize common currency/unit text and thousands separators.
    normalized = (
        text.replace(",", "")
        .replace("￥", "")
        .replace("¥", "")
        .replace("元", "")
        .replace("人民币", "")
        .replace("RMB", "")
        .replace("CNY", "")
        .replace("HKD", "")
        .replace("HK$", "")
        .replace("$", "")
        .replace("约", "")
        .replace("大约", "")
        .replace("左右", "")
        .strip()
    )

    # If the model outputs a range like "100-200", use the average.
    numbers = []
    for match in __import__("re").findall(r"\d+(?:\.\d+)?", normalized):
        try:
            numbers.append(float(match))
        except Exception:
            pass
    if not numbers:
        return float(default)
    if len(numbers) >= 2 and ("-" in normalized or "至" in normalized or "到" in normalized):
        return float(sum(numbers[:2]) / 2)
    return float(numbers[0])


def _is_api_configuration_error(exc: Exception) -> bool:
    """Return True when an online grounding failure is really about missing API setup."""
    text = safe_text(exc).lower()
    keywords = [
        "tavily_api_key",
        "serpapi_api_key",
        "bing_search_api_key",
        "google_places_api_key",
        "request_denied",
        "online_search_provider",
        "api key",
        "unauthorized",
        "401",
        "403",
    ]
    return any(k in text for k in keywords)


_ONLINE_UNAVAILABLE_WARNED = False


def _warn_online_unavailable_once(detail: str = "") -> None:
    """Print a single warning when the online provider is unusable but online
    grounding is optional, so we fall back to the seed/offline data instead of
    crashing every family (e.g. Google Places billing disabled -> REQUEST_DENIED)."""
    global _ONLINE_UNAVAILABLE_WARNED
    if _ONLINE_UNAVAILABLE_WARNED:
        return
    _ONLINE_UNAVAILABLE_WARNED = True
    print(
        "[Grounding] 在线搜索不可用，已降级为仅使用本地 grounding_seed 数据继续生成"
        "（REQUIRE_ONLINE_GROUNDING=false）。原因: " + safe_text(detail)[:160]
    )


def is_generic_hotel_text(value: Any) -> bool:
    text = safe_text(value).lower()
    if not text:
        return True
    return any((term.lower() == text or term.lower() in text for term in GENERIC_HOTEL_TERMS))


def is_generic_transport_text(value: Any) -> bool:
    text = safe_text(value).lower()
    if not text:
        return True
    return any((term.lower() == text or term.lower() in text for term in GENERIC_TRANSPORT_TERMS))


def _json_http_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    headers = headers or {}
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {**headers, "Content-Type": "application/json"}
    req = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=ONLINE_SEARCH_TIMEOUT_SECONDS) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    return json.loads(raw)


def online_search(query: str, max_results: int = ONLINE_SEARCH_MAX_RESULTS) -> List[Dict[str, Any]]:
    """Search the web with a small raw-result cache.

    Raw cache files are stored in ``data/cache/online_grounding/raw_search``.
    This avoids repeated API calls for exactly the same query while keeping the
    structured hotel/restaurant/ticket/route cache separate.
    """
    if not ONLINE_GROUNDING_ENABLED:
        return []
    query = safe_text(query)
    if not query:
        return []
    provider = ONLINE_SEARCH_PROVIDER
    cache_payload = {"provider": provider, "query": query, "max_results": int(max_results)}
    cache_path = RAW_SEARCH_CACHE_DIR / f"{_stable_cache_key(cache_payload)}.json"
    if ONLINE_GROUNDING_CACHE_ENABLED:
        _ensure_cache_dirs()
        cached = _read_json_file(cache_path, None)
        if isinstance(cached, dict) and isinstance(cached.get("results"), list):
            output = cached.get("results", [])[:max_results]
            for item in output:
                if isinstance(item, dict):
                    item["cache_hit"] = True
            return output

    try:
        if provider == "tavily":
            api_key = os.getenv("TAVILY_API_KEY", "")
            if not api_key:
                raise RuntimeError("需要配置 TAVILY_API_KEY，或把 ONLINE_SEARCH_PROVIDER 改为 serpapi/bing。")
            payload = {
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
                "include_answer": False,
                "include_raw_content": False,
            }
            data = _json_http_request(
                "https://api.tavily.com/search", method="POST", payload=payload
            )
            results = [
                {
                    "title": safe_text(item.get("title")),
                    "url": safe_text(item.get("url")),
                    "content": safe_text(item.get("content")),
                    "score": item.get("score"),
                    "provider": "tavily",
                }
                for item in data.get("results", [])[:max_results]
            ]
        elif provider == "serpapi":
            api_key = os.getenv("SERPAPI_API_KEY", "")
            if not api_key:
                raise RuntimeError("需要配置 SERPAPI_API_KEY，或把 ONLINE_SEARCH_PROVIDER 改为 tavily/bing。")
            params = urllib.parse.urlencode(
                {
                    "engine": "google",
                    "q": query,
                    "api_key": api_key,
                    "num": max_results,
                    "hl": "zh-cn",
                }
            )
            data = _json_http_request(f"https://serpapi.com/search.json?{params}")
            items = data.get("organic_results", [])[:max_results]
            results = [
                {
                    "title": safe_text(item.get("title")),
                    "url": safe_text(item.get("link")),
                    "content": safe_text(item.get("snippet")),
                    "provider": "serpapi",
                }
                for item in items
            ]
        elif provider == "bing":
            api_key = os.getenv("BING_SEARCH_API_KEY", "")
            if not api_key:
                raise RuntimeError(
                    "需要配置 BING_SEARCH_API_KEY，或把 ONLINE_SEARCH_PROVIDER 改为 tavily/serpapi。"
                )
            params = urllib.parse.urlencode({"q": query, "count": max_results, "mkt": "zh-CN"})
            headers = {"Ocp-Apim-Subscription-Key": api_key}
            data = _json_http_request(
                f"https://api.bing.microsoft.com/v7.0/search?{params}", headers=headers
            )
            items = data.get("webPages", {}).get("value", [])[:max_results]
            results = [
                {
                    "title": safe_text(item.get("name")),
                    "url": safe_text(item.get("url")),
                    "content": safe_text(item.get("snippet")),
                    "provider": "bing",
                }
                for item in items
            ]
        elif provider == "google_places":
            # Google Places Text Search returns real, named places (restaurants near a POI /
            # area, or concrete attractions for a vague theme) with an address and a Maps URL
            # that counts as a verifiable source. This is far stronger than web search for the
            # "restaurant near attraction" and "vague activity -> concrete place" cases.
            api_key = os.getenv("GOOGLE_PLACES_API_KEY", "") or os.getenv("GOOGLE_MAPS_API_KEY", "")
            if not api_key:
                raise RuntimeError(
                    "需要配置 GOOGLE_PLACES_API_KEY，或把 ONLINE_SEARCH_PROVIDER 改为 tavily/serpapi/bing。"
                )
            params = urllib.parse.urlencode({"query": query, "key": api_key, "language": "zh-CN"})
            data = _json_http_request(
                f"https://maps.googleapis.com/maps/api/place/textsearch/json?{params}"
            )
            status = safe_text(data.get("status"))
            if status not in ("OK", "ZERO_RESULTS"):
                # REQUEST_DENIED / INVALID_REQUEST usually mean a key/billing/API-not-enabled issue.
                raise RuntimeError(
                    f'GOOGLE_PLACES_API_KEY 或请求有误: {status} {safe_text(data.get("error_message"))}'
                )
            results = []
            for item in data.get("results", [])[:max_results]:
                if not isinstance(item, dict):
                    continue
                name = safe_text(item.get("name"))
                place_id = safe_text(item.get("place_id"))
                addr = safe_text(item.get("formatted_address"))
                rating = item.get("rating")
                # Quality gate: drop low-rated places (keep ones with no rating, e.g. new/obscure).
                if (
                    GOOGLE_PLACES_MIN_RATING > 0
                    and isinstance(rating, (int, float))
                    and float(rating) < GOOGLE_PLACES_MIN_RATING
                ):
                    continue
                types = item.get("types", []) if isinstance(item.get("types", []), list) else []
                url = (
                    f"https://www.google.com/maps/place/?q=place_id:{place_id}"
                    if place_id
                    else "https://maps.google.com/"
                )
                content = "；".join(
                    [
                        x
                        for x in [
                            addr,
                            (f"评分:{rating}" if rating else ""),
                            (" ".join(types[:4]) if types else ""),
                        ]
                        if x
                    ]
                )
                results.append(
                    {
                        "title": name,
                        "url": url,
                        "content": content,
                        "rating": rating,
                        "place_id": place_id,
                        "provider": "google_places",
                    }
                )
        else:
            raise RuntimeError(
                f"不支持的 ONLINE_SEARCH_PROVIDER: {provider}。请使用 tavily、serpapi、bing 或 google_places。"
            )
    except Exception as exc:
        # Missing API keys or unsupported providers should still be hard errors.
        # HTTP 432/429/timeouts/quota-like errors are search availability issues;
        # they should not make Module 5/7 candidate generation fail.
        # An unsupported provider name is always a hard config error.  A missing key
        # / disabled billing (REQUEST_DENIED) / quota is only fatal when online
        # grounding is REQUIRED; otherwise degrade gracefully to seed/offline so one
        # broken provider does not crash every family.
        if "不支持的 ONLINE_SEARCH_PROVIDER" in safe_text(exc):
            raise
        if _is_api_configuration_error(exc) and REQUIRE_ONLINE_GROUNDING:
            raise
        if (not ONLINE_SEARCH_SOFT_FAIL) and not _is_api_configuration_error(exc):
            raise
        if _is_api_configuration_error(exc):
            _warn_online_unavailable_once(safe_text(exc))
        error_text = safe_text(exc) or exc.__class__.__name__
        results = []
        if ONLINE_GROUNDING_CACHE_ENABLED:
            _write_json_file(
                cache_path,
                {
                    "provider": provider,
                    "query": query,
                    "max_results": max_results,
                    "cached_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "results": [],
                    "search_error": error_text,
                    "soft_failed": True,
                },
            )
        return results

    if ONLINE_GROUNDING_CACHE_ENABLED:
        _write_json_file(
            cache_path,
            {
                "provider": provider,
                "query": query,
                "max_results": max_results,
                "cached_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": results,
            },
        )
    return results


def compact_search_results(
    results: List[Dict[str, Any]],
    max_chars: int = 4500,
    max_items: int = 8,
    content_chars: int = 260,
) -> str:
    """Compact raw/cache search results before sending them to an LLM.

    v48 change: online cache is useful, but passing too many raw snippets into
    grounding prompts made Module 5 very long.  This helper keeps only a small
    number of high-signal fields, with short snippets, so cache does not bloat
    downstream prompts.
    """
    lines = []
    seen_urls = set()
    for i, item in enumerate(results[:max_items], 1):
        title = safe_text(item.get("title"))[:120]
        url = safe_text(item.get("url"))[:260]
        content = safe_text(item.get("content"))[:content_chars]
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        if title or url or content:
            lines.append(f"[{i}] {title}\nURL: {url}\n摘要: {content}")
    text = "\n\n".join(lines)
    return text[:max_chars]


def collect_search_source_urls(results: List[Dict[str, Any]], max_urls: int = 8) -> List[str]:
    urls = []
    for item in results:
        if isinstance(item, dict):
            url = safe_text(item.get("url"))
            if url and url not in urls:
                urls.append(url)
            if len(urls) >= max_urls:
                break
    return urls


def build_rule_based_restaurant_ticket_grounding(
    plan: Dict[str, Any],
    requirements: Dict[str, Any],
    search_results: List[Dict[str, Any]],
    family_agents: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create restaurant and ticket grounding without an extra LLM call.

    The previous version asked the LLM to extract a full restaurant/ticket JSON
    from many search snippets.  Logs showed that this sub-call often hit 5000
    tokens and failed.  For this project, deterministic construction is safer:
    use online/cache URLs as evidence, produce diverse area/type restaurant
    placeholders with explicit price ranges, and estimate ticket prices with the
    existing ticket rules.
    """
    destination = (
        safe_text(requirements.get("destination_city"))
        or safe_text(plan.get("destination_city"))
        or "当地"
    )
    restaurant_req = (
        requirements.get("restaurant_requirements", {})
        if isinstance(requirements.get("restaurant_requirements"), dict)
        else {}
    )
    price_level = (
        safe_text(restaurant_req.get("price_level"))
        or safe_text(plan.get("restaurant_price_level"))
        or "medium"
    )
    food_prefs = (
        restaurant_req.get("food_preferences", [])
        if isinstance(restaurant_req.get("food_preferences", []), list)
        else []
    )
    member_count = max(len(family_agents), 1)
    days = max(safe_int(plan.get("day"), 1), 1)
    urls = collect_search_source_urls(search_results, 8)

    restaurants = []
    per_day = (
        plan.get("per_day_itinerary", [])
        if isinstance(plan.get("per_day_itinerary", []), list)
        else []
    )
    for day in range(1, days + 1):
        day_item = (
            per_day[day - 1]
            if day - 1 < len(per_day) and isinstance(per_day[day - 1], dict)
            else {}
        )
        # Prefer the first concrete activity/area of the day as the meal area.
        area = ""
        for block in ["morning", "afternoon", "evening"]:
            values = day_item.get(block, []) if isinstance(day_item, dict) else []
            if not isinstance(values, list):
                values = [values] if safe_text(values) else []
            for activity in values:
                if isinstance(activity, dict):
                    area = (
                        safe_text(activity.get("area"))
                        or safe_text(activity.get("location"))
                        or safe_text(activity.get("poi_name"))
                        or safe_text(activity.get("activity"))
                    )
                else:
                    area = safe_text(activity)
                if area:
                    break
            if area:
                break
        area = area or destination
        for meal in ["lunch", "dinner"]:
            area_for_meal = _specific_restaurant_area_for_day_meal(plan, day, meal, destination)
            if _is_generic_restaurant_area(area_for_meal, destination):
                area_for_meal = area
            desc = build_restaurant_area_fallback_type(destination, price_level, food_prefs)
            price = clamp_restaurant_price(0, price_level)
            restaurants.append(
                {
                    "day": day,
                    "meal": meal,
                    "restaurant_name": "",
                    "name_or_type": "",
                    "restaurant_type": desc,
                    "area": area_for_meal,
                    "address_or_location": area_for_meal,
                    "specific_restaurant_found": False,
                    "restaurant_grounding_status": "area_fallback_no_specific_name",
                    "estimated_price_per_person": price,
                    "estimated_total_cost": round(price * member_count, 2),
                    "price_level": price_level,
                    "price_range_per_person": restaurant_price_range_text(price_level),
                    "source_urls": urls[:3],
                    "comfort_note": "未取得可靠具体餐厅名；仅给出当天景点或酒店站点附近的用餐范围，不编造店名；Module 6 会按缺少具体餐厅软扣分。",
                }
            )

    ticket_plan = []
    for poi in collect_daily_poi_names(plan):
        name = safe_text(poi.get("poi_name"))
        price = estimate_default_ticket_price(name)
        if price <= 0:
            continue
        ticket_plan.append(
            {
                "day": safe_int(poi.get("day"), 0),
                "poi_name": name,
                "ticket_price_per_person": round(price, 2),
                "participant_count": member_count,
                "estimated_total_cost": round(price * member_count, 2),
                "source_urls": urls[:3],
                "price_source": "rule_estimate_with_online_cache_reference",
            }
        )

    return {
        "restaurants": restaurants,
        "ticket_plan": ticket_plan,
        "source_urls": urls,
        "grounding_issues": [],
    }


def collect_plan_poi_names(plan: Dict[str, Any], max_items: int = 8) -> List[str]:
    names = []
    for day_item in plan.get("per_day_itinerary", []):
        if not isinstance(day_item, dict):
            continue
        for block_name in ["morning", "afternoon", "optional_activities"]:
            block = day_item.get(block_name, [])
            if not isinstance(block, list):
                continue
            for item in block:
                if isinstance(item, dict):
                    name = safe_text(item.get("poi_name")) or safe_text(item.get("activity"))
                else:
                    name = safe_text(item)
                if name and name not in names:
                    names.append(name)
                if len(names) >= max_items:
                    return names
    return names


def build_grounding_requirements(
    plan: Dict[str, Any], family_agents: List[Dict[str, Any]]
) -> Dict[str, Any]:
    existing = plan.get("grounding_requirements", {})
    if not isinstance(existing, dict):
        existing = {}
    first_agent = family_agents[0] if family_agents else {}
    destination = safe_text(plan.get("destination_city")) or safe_text(
        first_agent.get("destination_city")
    )
    departure = safe_text(plan.get("departure_city")) or safe_text(
        first_agent.get("departure_city")
    )
    days = safe_int(plan.get("day"), safe_int(first_agent.get("day"), 1))
    poi_names = collect_plan_poi_names(plan)
    strict_budget = False
    comfort_needs = []
    food_tags = set()
    for agent in family_agents:
        profile = agent.get("profile", {})
        constraints = agent.get("constraints", {})
        role_text = (
            safe_text(profile.get("role", "")).lower()
            + " "
            + safe_text(profile.get("age_stage", "")).lower()
        )
        if any(
            (
                k in role_text
                for k in ["child", "kid", "elder", "older", "grandfather", "grandmother"]
            )
        ):
            comfort_needs.append("适合儿童或老人，步行少，周边交通便利")
        if safe_text(profile.get("physical_constraint")).lower() == "low_intensity":
            comfort_needs.append("低强度出行，靠近地铁站或核心景点")
        if safe_text(profile.get("budget_sensitivity")).lower() == "strict_budget":
            strict_budget = True
        for tag in (
            profile.get("dietary_preference", [])
            if isinstance(profile.get("dietary_preference", []), list)
            else []
        ):
            if safe_text(tag):
                food_tags.add(safe_text(tag))
        latest = safe_text(constraints.get("latest_return_time"))
        if latest:
            comfort_needs.append(f"每天尽量在 {latest} 前回酒店")
    price_level = safe_text(plan.get("restaurant_price_level")) or "medium"
    hotel_plan = plan.get("hotel_plan", {}) if isinstance(plan.get("hotel_plan"), dict) else {}
    budget_limit = safe_float(plan.get("budget_limit"), 0)
    daily_route_requirements = []
    per_day = (
        plan.get("per_day_itinerary", [])
        if isinstance(plan.get("per_day_itinerary", []), list)
        else []
    )
    hotel_name = safe_text(hotel_plan.get("hotel_name"))
    hotel_area = safe_text(hotel_plan.get("hotel_area")) or safe_text(hotel_plan.get("area"))
    hotel_station = (
        safe_text(hotel_plan.get("nearby_metro_station"))
        or safe_text(hotel_plan.get("nearby_station"))
        or safe_text(hotel_plan.get("nearest_metro_station"))
    )
    preferred_area = hotel_area or "靠近多数主景点或地铁站的安全便利区域"
    itinerary_anchor_pois: List[str] = []
    for day_item in per_day:
        if not isinstance(day_item, dict):
            continue
        for block_name in ["morning", "afternoon"]:
            block = day_item.get(block_name, [])
            if not isinstance(block, list):
                continue
            for activity in block:
                if isinstance(activity, dict):
                    name = safe_text(activity.get("poi_name")) or safe_text(
                        activity.get("activity")
                    )
                else:
                    name = safe_text(activity)
                if name and name not in itinerary_anchor_pois:
                    itinerary_anchor_pois.append(name)
    for idx, day_item in enumerate(per_day):
        if not isinstance(day_item, dict):
            continue
        day_num = safe_int(day_item.get("day"), idx + 1)
        destinations = []
        for block_name in ["morning", "afternoon", "optional_activities"]:
            block = day_item.get(block_name, [])
            if not isinstance(block, list):
                continue
            for activity in block:
                if isinstance(activity, dict):
                    name = safe_text(activity.get("poi_name")) or safe_text(
                        activity.get("activity")
                    )
                else:
                    name = safe_text(activity)
                if name and name not in destinations:
                    destinations.append(name)
        # v50: after restaurants are grounded, include meal stops in the route chain.
        for meal_key in ["lunch", "dinner"]:
            meal = day_item.get(meal_key, {})
            if isinstance(meal, dict):
                meal_stop = (
                    safe_text(meal.get("restaurant_name"))
                    or safe_text(meal.get("restaurant_area"))
                    or safe_text(meal.get("location"))
                    or safe_text(meal.get("area"))
                )
                if meal_stop and meal_stop not in destinations:
                    destinations.append(meal_stop)
        grounded_restaurants = (
            plan.get("grounded_restaurants", [])
            if isinstance(plan.get("grounded_restaurants", []), list)
            else []
        )
        for restaurant in grounded_restaurants:
            if not isinstance(restaurant, dict):
                continue
            if safe_int(restaurant.get("day"), 0) != day_num:
                continue
            restaurant_stop = (
                safe_text(restaurant.get("restaurant_name"))
                or safe_text(restaurant.get("area"))
                or safe_text(restaurant.get("restaurant_type"))
            )
            if restaurant_stop and restaurant_stop not in destinations:
                destinations.append(restaurant_stop)
        daily_route_requirements.append(
            {
                "day": day_num,
                "origin_requirement": f"从酒店所在区域或酒店附近地铁站出发，当前期望区域：{preferred_area}",
                "destination_requirements": destinations[:3],
                "latest_return_time": safe_text(day_item.get("estimated_return_time"))
                or safe_text(plan.get("return_time"))
                or "21:00 前",
                "max_walking_distance_km": min(
                    [
                        safe_float(a.get("constraints", {}).get("max_walking_distance_km"), 999)
                        for a in family_agents
                    ]
                    or [999]
                ),
                "comfort_requirements": list(dict.fromkeys(comfort_needs)) or ["少换乘，减少步行，适合家庭"],
                "required_output": "在线查询时必须返回当天从酒店到主要目的地、再返回酒店的路线、时间和费用",
            }
        )
    requirements = {
        "departure_city": departure,
        "destination_city": destination,
        "days": days,
        "budget_limit": budget_limit,
        "hotel_requirements": existing.get("hotel_requirements")
        or {
            "price_level": safe_text(hotel_plan.get("price_level"))
            or ("low" if strict_budget else "medium"),
            "preferred_area": preferred_area,
            "selected_hotel_name": hotel_name,
            "selected_hotel_area": hotel_area,
            "selected_hotel_nearby_station": hotel_station,
            "itinerary_anchor_pois": itinerary_anchor_pois[:8],
            "location_priority": "选择距离多数主景点/每日路线链最近且符合预算、舒适度和家庭出行要求的酒店",
            "max_price_per_night": safe_float(hotel_plan.get("max_price_per_night"), 0),
            "nights": max(days - 1, 1),
            "nearby_transport_requirement": "优先选择步行可到地铁站或主要公共交通站点、且靠近主行程区域的酒店",
            "location_requirements": ["距离主景点更近", "靠近地铁站", "减少每天折返", "方便前往多数主景点", "周边安全便利"],
            "comfort_requirements": list(dict.fromkeys(comfort_needs)) or ["交通便利，适合家庭入住"],
            "search_keywords": [
                destination,
                preferred_area,
                "最近",
                "主景点附近",
                "家庭酒店",
                "地铁站附近",
                safe_text(hotel_plan.get("price_level")) or ("low" if strict_budget else "medium"),
            ],
            "why_needed_for_itinerary": "酒店位置会决定每天交通路线，因此必须优先选择距离完整行程主景点最近且符合预算/舒适/交通要求的真实酒店",
        },
        "restaurant_requirements": existing.get("restaurant_requirements")
        or {
            "price_level": price_level,
            "food_preferences": sorted(food_tags) or ["本地餐饮", "家庭友好"],
            "comfort_requirements": ["适合家庭", "排队时间不要过长", "价格清楚"],
        },
        "transportation_requirements": existing.get("transportation_requirements")
        or {
            "intercity_origin": departure,
            "intercity_destination": destination,
            "selected_hotel_name": hotel_name,
            "hotel_area_dependency": "市内路线必须以最终酒店区域或酒店附近站点为起点",
            "hotel_base_station": hotel_station or preferred_area,
            "post_generation_route_stitching": True,
            "within_city": "优先地铁或公共交通，老人儿童疲劳时可短途打车",
            "daily_return_time": safe_text(plan.get("return_time")) or "21:00 前",
            "daily_route_requirements": daily_route_requirements,
            "route_requirements": [
                "全体行程生成后再查询路线",
                "按酒店->上午景点->午餐->下午景点->晚餐->酒店串联",
                "包含站点/线路/预计费用",
                "避免长距离步行",
                "标注出发地、目的地、出行时间和舒适度要求",
            ],
        },
        "ticket_requirements": existing.get("ticket_requirements")
        or {"pois_to_check": poi_names, "need_ticket_prices": True},
    }
    plan["grounding_requirements"] = requirements
    return requirements


def build_hotel_online_queries(requirements: Dict[str, Any]) -> List[str]:
    destination = safe_text(requirements.get("destination_city"))
    hotel_req = (
        requirements.get("hotel_requirements", {})
        if isinstance(requirements.get("hotel_requirements"), dict)
        else {}
    )
    area = safe_text(hotel_req.get("preferred_area"))
    price = safe_text(hotel_req.get("price_level"))
    anchors = (
        hotel_req.get("itinerary_anchor_pois", [])
        if isinstance(hotel_req.get("itinerary_anchor_pois", []), list)
        else []
    )
    anchor_text = " ".join([safe_text(x) for x in anchors[:4] if safe_text(x)])
    comfort = (
        " ".join([safe_text(x) for x in hotel_req.get("comfort_requirements", [])])
        if isinstance(hotel_req.get("comfort_requirements", []), list)
        else safe_text(hotel_req.get("comfort_requirements"))
    )
    queries = [
        f"{destination} {anchor_text} 附近 最近 符合预算 家庭 酒店 {price} 每晚 价格 地铁站",
        f"{destination} {area} 主景点附近 最近 家庭酒店 {price} 价格 地铁站 交通便利",
        f"{destination} near {anchor_text} closest qualified family friendly hotel {price} metro station price",
        f"{destination} {area} 亲子 住宿 舒适 交通便利 酒店 价格 {comfort}",
    ]
    return [q for q in queries if safe_text(q)]


def build_transport_online_queries(requirements: Dict[str, Any]) -> List[str]:
    destination = safe_text(requirements.get("destination_city"))
    departure = safe_text(requirements.get("departure_city"))
    hotel_req = (
        requirements.get("hotel_requirements", {})
        if isinstance(requirements.get("hotel_requirements"), dict)
        else {}
    )
    transport_req = (
        requirements.get("transportation_requirements", {})
        if isinstance(requirements.get("transportation_requirements"), dict)
        else {}
    )
    daily_routes = (
        transport_req.get("daily_route_requirements", [])
        if isinstance(transport_req.get("daily_route_requirements", []), list)
        else []
    )
    hotel_name = safe_text(transport_req.get("selected_hotel_name")) or safe_text(
        hotel_req.get("selected_hotel_name")
    )
    base = (
        safe_text(transport_req.get("hotel_base_station"))
        or safe_text(hotel_req.get("selected_hotel_nearby_station"))
        or safe_text(hotel_req.get("selected_hotel_area"))
        or safe_text(hotel_req.get("preferred_area"))
        or "酒店"
    )
    queries = []
    if departure and destination:
        queries.append(f"{departure} 到 {destination} 高铁 飞机 交通 时间 票价")
    if hotel_name:
        queries.append(f"{destination} {hotel_name} 附近 地铁站 公共交通 出行")
    queries.append(f"{destination} {base} 地铁 出租车 票价 公共交通 家庭 出行 路线")
    for item in daily_routes[:6]:
        if not isinstance(item, dict):
            continue
        day = safe_int(item.get("day"), 0)
        destinations = (
            item.get("destination_requirements", [])
            if isinstance(item.get("destination_requirements", []), list)
            else []
        )
        clean_dests = [safe_text(x) for x in destinations if safe_text(x)]
        if clean_dests:
            chain_text = " 到 ".join(clean_dests[:4])
            queries.append(f"{destination} 第{day}天 从 {base} 到 {chain_text} 再回酒店 地铁 出租车 路线 时间 费用")
        for dest in clean_dests[:3]:
            queries.append(f"{destination} 从 {base} 到 {dest} 地铁 出租车 路线 时间 票价")
    return [q for q in queries if safe_text(q)]


def build_restaurant_ticket_online_queries(requirements: Dict[str, Any]) -> List[str]:
    """Build restaurant and ticket search queries.

    v45 improvement:
    - Search restaurants around different day-level areas/POIs, not only one
      generic city-level query. This gives the grounding LLM more choices and
      reduces repeated restaurant names in the final plan.
    - Search ticket prices for every important paid attraction candidate so
      scenic-area admission fees can be included in the budget.
    """
    destination = safe_text(requirements.get("destination_city"))
    restaurant_req = (
        requirements.get("restaurant_requirements", {})
        if isinstance(requirements.get("restaurant_requirements"), dict)
        else {}
    )
    transport_req = (
        requirements.get("transportation_requirements", {})
        if isinstance(requirements.get("transportation_requirements"), dict)
        else {}
    )
    ticket_req = (
        requirements.get("ticket_requirements", {})
        if isinstance(requirements.get("ticket_requirements"), dict)
        else {}
    )
    pois = (
        ticket_req.get("pois_to_check", [])
        if isinstance(ticket_req.get("pois_to_check", []), list)
        else []
    )
    food_preferences = (
        restaurant_req.get("food_preferences", [])
        if isinstance(restaurant_req.get("food_preferences", []), list)
        else []
    )
    food_text = " ".join([safe_text(x) for x in food_preferences[:4]]) or "本地餐饮 家庭友好"
    price_level = safe_text(restaurant_req.get("price_level")) or "medium"

    queries = [
        f"{destination} 适合家庭 餐厅 {price_level} 人均价格 位置 推荐 {food_text}",
        f"{destination} family restaurant price per person location recommended {price_level}",
    ]

    # Add area-specific restaurant queries from daily destinations.
    daily_routes = (
        transport_req.get("daily_route_requirements", [])
        if isinstance(transport_req.get("daily_route_requirements", []), list)
        else []
    )
    for item in daily_routes[:6]:
        if not isinstance(item, dict):
            continue
        dests = (
            item.get("destination_requirements", [])
            if isinstance(item.get("destination_requirements", []), list)
            else []
        )
        for dest in dests[:2]:
            area = safe_text(dest)
            if area:
                queries.append(f"{destination} {area} 附近 家庭 餐厅 人均价格 推荐 {food_text}")

    # Ticket queries. Use more POIs because missing ticket prices were causing
    # under-estimated total costs.
    for poi in pois[:10]:
        poi_text = safe_text(poi)
        if poi_text:
            queries.append(f"{destination} {poi_text} 门票 票价 开放时间 官方 2026")
            queries.append(f"{destination} {poi_text} ticket price official opening hours")

    # De-duplicate while preserving order.
    seen = set()
    output = []
    for q in queries:
        q = safe_text(q)
        key = q.lower()
        if q and key not in seen:
            seen.add(key)
            output.append(q)
    return output


def run_search_queries(queries: List[str]) -> List[Dict[str, Any]]:
    all_results: List[Dict[str, Any]] = []
    seen_urls = set()
    for query in queries:
        try:
            results = online_search(query, max_results=ONLINE_SEARCH_MAX_RESULTS)
        except Exception as exc:
            # Keep configuration errors visible; otherwise allow grounding to
            # continue with cache/LLM/rule estimates. This prevents one Tavily
            # HTTP 432 from killing a Module 7 candidate.
            if _is_api_configuration_error(exc) or not ONLINE_SEARCH_SOFT_FAIL:
                raise
            results = []
        for item in results:
            item["query"] = query
            url = safe_text(item.get("url"))
            key = url or json.dumps(item, ensure_ascii=False, sort_keys=True)
            if key in seen_urls:
                continue
            seen_urls.add(key)
            all_results.append(item)
    return all_results


# ---------------------------------------------------------------------------
# v45 grounding cleanup helpers
# ---------------------------------------------------------------------------

PRICE_LEVEL_RANGES = {
    "low": (30.0, 80.0),
    "medium": (50.0, 180.0),
    "high": (100.0, 350.0),
    "luxury": (180.0, 600.0),
}

PRICE_LEVEL_CN = {
    "low": "低价位",
    "medium": "中等价位",
    "high": "中高价位",
    "luxury": "高端价位",
}


# -----------------------------------------------------------------------------
# v55: explicit Tavily day-by-day restaurant grounding
# -----------------------------------------------------------------------------
RESTAURANT_RESULT_GENERIC_TERMS = [
    "推荐",
    "攻略",
    "排名",
    "榜单",
    "十大",
    "10大",
    "top ",
    "best ",
    "餐厅推荐",
    "美食推荐",
    "restaurants in",
    "restaurant guide",
    "tripadvisor",
    "openrice",
    "大众点评",
    "booking",
    "附近餐厅",
    "家庭餐厅",
    "中等价位",
    "当地美食",
    "本地餐厅",
    "家庭友好",
    "restaurant near",
]
RESTAURANT_NAME_HINTS = [
    "餐厅",
    "餐廳",
    "饭店",
    "飯店",
    "酒楼",
    "酒樓",
    "茶餐厅",
    "茶餐廳",
    "火锅",
    "火鍋",
    "料理",
    "食堂",
    "小馆",
    "小館",
    "菜馆",
    "菜館",
    "自助",
    "咖啡",
    "Cafe",
    "Café",
    "Restaurant",
    "Bistro",
    "Kitchen",
    "Grill",
    "Diner",
    "Dining",
    "Bar & Grill",
]


def _meal_label_cn(meal: str) -> str:
    return "午餐" if safe_text(meal).lower() == "lunch" else "晚餐"


def _collect_text_from_day_block(value: Any) -> List[str]:
    out: List[str] = []
    if isinstance(value, dict):
        for key in [
            "restaurant_name",
            "name_or_type",
            "restaurant_type",
            "poi_name",
            "activity",
            "area",
            "location",
        ]:
            text = safe_text(value.get(key))
            if text:
                out.append(text)
    elif isinstance(value, list):
        for item in value:
            out.extend(_collect_text_from_day_block(item))
    else:
        text = safe_text(value)
        if text:
            out.append(text)
    return out


def _is_non_route_stop_for_transport(value: Any) -> bool:
    text = safe_text(value).strip()
    if not text:
        return True
    low = text.lower()
    non_stop_terms = [
        "早餐",
        "午餐",
        "晚餐",
        "用餐",
        "餐厅",
        "美食",
        "简餐",
        "自由活动",
        "自由安排",
        "准备返程",
        "返回",
        "返程",
        "整理行李",
        "入住",
        "退房",
        "休息",
        "无",
        "none",
        "null",
        "hotel",
        "meal",
    ]
    if any(term.lower() in low for term in non_stop_terms):
        # Keep concrete restaurant/place names even when they include a food word.
        if any(x in text for x in ["路", "街", "区", "站", "广场", "城", "园", "馆", "大学", "中心", "港"]):
            return False
        return True
    return False


GENERIC_RESTAURANT_AREA_TERMS = [
    "当地",
    "本地",
    "市中心",
    "中心城区",
    "中央区域",
    "同区域",
    "酒店区域",
    "酒店附近",
    "餐厅",
    "午餐",
    "晚餐",
    "美食",
    "当地美食",
    "本地餐厅",
    "家庭餐厅",
    "家庭友好餐厅",
    "简餐",
    "中档",
    "中等价位",
    "自选餐饮",
    "无",
    "none",
    "null",
    "city center",
    "短路线需求",
    "短途出租车",
    "第1天主要活动区域",
    "第2天主要活动区域",
    "第3天主要活动区域",
    "第4天主要活动区域",
    "第5天主要活动区域",
    "高档餐饮+高档",
    "自助餐+高",
    "儿童友好餐饮+中等价格",
]


def _is_generic_restaurant_area(value: Any, destination: Any = "") -> bool:
    """Return True when a meal fallback location is only city/type-level.

    v56: Avoid final output like "Tianjin附近" or "市中心".  A fallback meal
    location should be anchored to a concrete day stop, hotel area, or metro
    station whenever possible.
    """
    text = safe_text(value).strip()
    if not text:
        return True
    low = text.lower()
    dest = safe_text(destination).strip()
    dest_low = dest.lower()
    if dest and (low == dest_low or low == dest_low.replace(" ", "")):
        return True
    if len(text) <= 1:
        return True
    if any(term.lower() in low for term in GENERIC_RESTAURANT_AREA_TERMS):
        # Keep specific district/station names even if they contain a broad word.
        if any(
            x in text
            for x in ["站", "路", "街", "区", "园", "馆", "港", "城", "广场", "寺", "塔", "大学", "Disney", "迪士尼"]
        ):
            return False
        return True
    # Abstract activity names are not good fallback addresses.
    if any(x in text for x in ["文化地标", "文化景点", "独特建筑", "拍照机会", "最后的文化体验", "轻松的下午", "自由活动", "准备返程"]):
        return True
    return False


def _clean_restaurant_area_anchor(value: Any, destination: Any = "") -> str:
    text = safe_text(value).strip("。；;，, ")
    if not text:
        return ""
    text = re.sub(r"^(参观|游览|探索|体验|享受|品尝|观看|欣赏|在|于)", "", text).strip()
    text = re.sub(r"(用餐|午餐|晚餐|轻食)$", "", text).strip(" 的附近周边，；。")
    if _is_generic_restaurant_area(text, destination):
        return ""
    return text


def _hotel_anchor_for_restaurant(plan: Dict[str, Any]) -> str:
    hotel = plan.get("hotel_plan", {}) if isinstance(plan.get("hotel_plan"), dict) else {}
    station = (
        safe_text(hotel.get("nearby_metro_station"))
        or safe_text(hotel.get("nearby_station"))
        or safe_text(hotel.get("nearest_metro_station"))
    )
    area = safe_text(hotel.get("hotel_area")) or safe_text(hotel.get("area"))
    name = safe_text(hotel.get("hotel_name")) or safe_text(hotel.get("hotel_name_or_type"))
    if area and station:
        return f"{area}、{station}"
    if station:
        return station
    if area:
        return area
    return name


def _specific_restaurant_area_for_day_meal(
    plan: Dict[str, Any], day: int, meal: str, destination: str
) -> str:
    """Choose the most concrete fallback location for one meal.

    Priority: existing meal location -> day activity anchor -> hotel area/station -> city.
    This keeps fallback addresses specific enough for final TXT export while not
    inventing a restaurant name.
    """
    per_day = (
        plan.get("per_day_itinerary", [])
        if isinstance(plan.get("per_day_itinerary", []), list)
        else []
    )
    day_item = (
        per_day[day - 1] if 0 < day <= len(per_day) and isinstance(per_day[day - 1], dict) else {}
    )

    candidates: List[str] = []
    meal_block = day_item.get(meal) if isinstance(day_item, dict) else None
    for text in _collect_text_from_day_block(meal_block):
        candidates.append(text)

    blocks = (
        ["morning", "afternoon"]
        if safe_text(meal).lower() == "lunch"
        else ["afternoon", "evening", "morning"]
    )
    for block in blocks:
        for text in _collect_text_from_day_block(day_item.get(block)):
            if text and not _is_non_route_stop_for_transport(text):
                candidates.append(text)

    # Route chains sometimes already contain the hotel/station and activity names.
    transportation = day_item.get("transportation") if isinstance(day_item, dict) else None
    for text in _collect_text_from_day_block(transportation):
        for part in re.split(r"[-→>—–]+", text):
            part = safe_text(part)
            if part and not _is_non_route_stop_for_transport(part):
                candidates.append(part)

    for cand in candidates:
        cand = _clean_restaurant_area_anchor(cand, destination)
        if cand and not _is_placeholder_restaurant_name(cand):
            return cand

    hotel_anchor = _clean_restaurant_area_anchor(_hotel_anchor_for_restaurant(plan), destination)
    if hotel_anchor:
        return hotel_anchor
    return safe_text(destination) or "当地"


def _restaurant_food_query_text(
    restaurant_req: Dict[str, Any], family_agents: List[Dict[str, Any]], meal: str
) -> str:
    prefs: List[str] = []
    raw = restaurant_req.get("food_preferences", []) if isinstance(restaurant_req, dict) else []
    if isinstance(raw, list):
        prefs.extend([safe_text(x) for x in raw if safe_text(x)])
    for agent in family_agents:
        profile = agent.get("profile", {}) if isinstance(agent, dict) else {}
        dp = profile.get("dietary_preference", []) if isinstance(profile, dict) else []
        if isinstance(dp, list):
            prefs.extend([safe_text(x) for x in dp if safe_text(x)])
    # Translate the most useful benchmark tokens into query-friendly Chinese.
    joined = " ".join(prefs).lower()
    mapped: List[str] = []
    if "korean" in joined:
        mapped.append("韩国料理")
    if "buffet" in joined:
        mapped.append("自助餐")
    if "western" in joined:
        mapped.append("西餐")
    if "seafood" in joined:
        mapped.append("海鲜")
    if "local" in joined:
        mapped.append("本地菜")
    if "child" in joined or "family" in joined:
        mapped.append("适合家庭")
    if not mapped:
        mapped.append("适合家庭")
    return " ".join(list(dict.fromkeys(mapped))[:4])


def _build_day_meal_restaurant_query(
    destination: str, area: str, meal: str, food_text: str, price_level: str
) -> str:
    # Compact query: area + meal + cuisine; avoid dumping the full itinerary.
    q = f"{destination} {area} 附近 {_meal_label_cn(meal)} {food_text} 具体餐厅 名称 人均 推荐"
    if price_level:
        q += f" {price_level}"
    return _compact_query_text(q, RESTAURANT_QUERY_MAX_CHARS)


def _clean_restaurant_candidate_name(text: Any) -> str:
    text = safe_text(text)
    if not text:
        return ""
    # Search titles often append site names and list titles.  Keep the most
    # name-like leading fragment, but reject list/guide pages.
    text = re.sub(r"<[^>]+>", "", text)
    parts = re.split(r"\s*[-|｜_–—:]\s*", text)
    candidates = parts[:2] if parts else [text]
    for cand in candidates:
        cand = re.sub(r"^(【.*?】|\[.*?\])", "", cand).strip(" \t\n\r，,。")
        if not cand:
            continue
        low = cand.lower()
        if any(term.lower() in low for term in RESTAURANT_RESULT_GENERIC_TERMS):
            continue
        if _is_placeholder_restaurant_name(cand):
            continue
        if len(cand) < 2 or len(cand) > 36:
            continue
        # Prefer names containing restaurant-like hints.  Allow short Chinese
        # brand names from titles if the surrounding result is restaurant-related.
        if any(h.lower() in low for h in [x.lower() for x in RESTAURANT_NAME_HINTS]):
            return cand
    return ""


def _extract_specific_restaurant_from_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    for r in results:
        if not isinstance(r, dict):
            continue
        title = safe_text(r.get("title"))
        content = safe_text(r.get("content"))
        url = safe_text(r.get("url"))
        # Try title first; titles usually contain the restaurant/entity name.
        name = _clean_restaurant_candidate_name(title)
        if not name:
            # Then try short restaurant-name patterns from snippets.
            text = f"{title}。{content}"
            for m in re.finditer(
                r"([\u4e00-\u9fffA-Za-z0-9·&’\'（）() ]{2,30}(?:餐厅|餐廳|饭店|飯店|酒楼|酒樓|茶餐厅|茶餐廳|火锅|火鍋|料理|自助|Cafe|Café|Restaurant|Bistro|Kitchen))",
                text,
                flags=re.IGNORECASE,
            ):
                cand = _clean_restaurant_candidate_name(m.group(1))
                if cand:
                    name = cand
                    break
        if name:
            prices = _extract_price_candidates_from_text(f"{title} {content}")
            return {
                "restaurant_name": name,
                "source_urls": [url] if url else [],
                "price_candidates": prices,
                "raw_title": title,
            }
    return {}


def tavily_restaurant_grounding(
    plan: Dict[str, Any], requirements: Dict[str, Any], family_agents: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Search concrete restaurants meal-by-meal.

    The older path used one city-level restaurant query and allowed generic
    placeholders.  This function queries each day's lunch/dinner near the actual
    day anchor.  If no supported restaurant name is found, it returns no exact
    name for that meal and records a soft issue; Module 6 can deduct points but
    the candidate continues.
    """
    if not RESTAURANT_GROUNDING_DAY_BY_DAY:
        return {"restaurants": [], "grounding_issues": ["restaurant_grounding_day_by_day_disabled"]}
    destination = safe_text(requirements.get("destination_city")) or safe_text(
        plan.get("destination_city")
    )
    restaurant_req = (
        requirements.get("restaurant_requirements", {})
        if isinstance(requirements.get("restaurant_requirements"), dict)
        else {}
    )
    price_level = (
        safe_text(restaurant_req.get("price_level"))
        or safe_text(plan.get("restaurant_price_level"))
        or "medium"
    )
    member_count = max(len(family_agents), 1)
    per_day = (
        plan.get("per_day_itinerary", [])
        if isinstance(plan.get("per_day_itinerary", []), list)
        else []
    )
    restaurants: List[Dict[str, Any]] = []
    issues: List[str] = []
    searched = 0
    for idx, day_item in enumerate(per_day):
        if not isinstance(day_item, dict):
            continue
        day_num = safe_int(day_item.get("day"), idx + 1)
        for meal in ["lunch", "dinner"]:
            if searched >= RESTAURANT_QUERY_MAX_MEALS:
                issues.append("restaurant_query_limit_reached")
                break
            area = _specific_restaurant_area_for_day_meal(plan, day_num, meal, destination)
            food_text = _restaurant_food_query_text(restaurant_req, family_agents, meal)
            query = _build_day_meal_restaurant_query(
                destination, area, meal, food_text, price_level
            )
            searched += 1
            results: List[Dict[str, Any]] = []
            try:
                results = online_search(
                    query, max_results=max(1, min(ONLINE_SEARCH_MAX_RESULTS, 2))
                )
            except Exception as exc:
                issues.append(
                    f"tavily_restaurant_search_failed:day{day_num}_{meal}:{safe_text(exc)}"
                )
                results = []
            extracted = _extract_specific_restaurant_from_results(results)
            if not extracted:
                issues.append(f"restaurant_name_pending:day{day_num}_{meal}:{area}")
                price = clamp_restaurant_price(0, price_level)
                fallback_type = build_restaurant_area_fallback_type(
                    destination, price_level, [food_text]
                )
                restaurants.append(
                    {
                        "day": day_num,
                        "meal": meal,
                        "restaurant_name": "",
                        "name_or_type": "",
                        "restaurant_type": fallback_type,
                        "area": area,
                        "address_or_location": area,
                        "specific_restaurant_found": False,
                        "restaurant_grounding_status": "area_fallback_no_specific_name",
                        "estimated_price_per_person": price,
                        "estimated_total_cost": round(price * member_count, 2),
                        "price_level": price_level,
                        "source_urls": [],
                        "search_query": query,
                        "comfort_note": "未返回可靠具体餐厅名；仅给出当天景点/酒店站点附近的用餐范围，不编造店名；Module 6 会按缺少具体餐厅软扣分。",
                    }
                )
                continue
            price = clamp_restaurant_price(
                extracted.get("price_candidates", [0])[0]
                if extracted.get("price_candidates")
                else 0,
                price_level,
            )
            restaurants.append(
                {
                    "day": day_num,
                    "meal": meal,
                    "restaurant_name": extracted.get("restaurant_name"),
                    "name_or_type": extracted.get("restaurant_name"),
                    "restaurant_type": food_text or "家庭友好餐厅",
                    "area": area,
                    "address_or_location": area,
                    "specific_restaurant_found": True,
                    "restaurant_grounding_status": "specific_name_found",
                    "estimated_price_per_person": price,
                    "estimated_total_cost": round(price * member_count, 2),
                    "price_level": price_level,
                    "source_urls": extracted.get("source_urls", []),
                    "search_query": query,
                    "comfort_note": "按天按餐别调用Tavily查询到的具体餐厅；营业时间、排队和人均价格需出行前核验。",
                }
            )
    return {"restaurants": restaurants, "grounding_issues": issues}


# -----------------------------------------------------------------------------
# v52: explicit Tavily/API-style ticket and transport cost grounding
# -----------------------------------------------------------------------------
NON_TICKET_ACTIVITY_KEYWORDS = [
    "自由活动",
    "自由安排",
    "准备返程",
    "返回",
    "返程",
    "离开",
    "抵达",
    "早餐",
    "午餐",
    "晚餐",
    "用餐",
    "餐厅",
    "美食",
    "小吃",
    "购物",
    "逛街",
    "整理行李",
    "入住",
    "退房",
    "酒店",
    "散步",
    "休息",
    "海边散步",
    "公园散步",
    "market",
    "shopping",
    "restaurant",
    "meal",
    "breakfast",
    "lunch",
    "dinner",
    "return",
    "departure",
    "arrival",
    "hotel",
    "free time",
]
PAID_ATTRACTION_HINTS = [
    "迪士尼",
    "Disney",
    "海洋公园",
    "欢乐谷",
    "长隆",
    "野生动物园",
    "虎园",
    "博物馆",
    "museum",
    "塔",
    "观景台",
    "山顶",
    "缆车",
    "寺",
    "园博园",
    "动物园",
    "主题公园",
    "theme park",
    "景区",
    "景点",
    "展览",
    "show",
    "表演",
    "马戏",
]
PRICE_REGEX_PATTERNS = [
    r"(?:门票|票价|成人票|入场费|admission|ticket(?: price)?)\D{0,18}(?:¥|￥|RMB|CNY|HK\$|HKD)?\s*(\d{1,4}(?:\.\d+)?)",
    r"(?:¥|￥|RMB|CNY|HK\$|HKD)\s*(\d{1,4}(?:\.\d+)?)\D{0,12}(?:门票|票价|成人票|admission|ticket)",
    r"(\d{1,4}(?:\.\d+)?)\s*(?:元|港币|港元)\D{0,12}(?:门票|票价|成人票|入场|入園|入园|admission|ticket)",
]


def _is_non_ticket_activity(name: Any) -> bool:
    text = safe_text(name).lower()
    if not text:
        return True
    return any(k.lower() in text for k in NON_TICKET_ACTIVITY_KEYWORDS)


def _looks_like_paid_attraction(name: Any) -> bool:
    text = safe_text(name)
    if not text or _is_non_ticket_activity(text):
        return False
    if any(k.lower() in text.lower() for k in FREE_ATTRACTION_KEYWORDS):
        return False
    return any(k.lower() in text.lower() for k in PAID_ATTRACTION_HINTS)


def _extract_price_candidates_from_text(text: str) -> List[float]:
    text = safe_text(text)
    candidates: List[float] = []
    for pattern in PRICE_REGEX_PATTERNS:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            try:
                value = float(m.group(1))
            except Exception:
                continue
            if 0 < value <= 2000:
                candidates.append(value)
    return candidates


def _choose_reasonable_ticket_price(values: List[float]) -> float:
    values = sorted([v for v in values if 0 < v <= 2000])
    if not values:
        return 0.0
    # Use the lower-middle value to avoid accidentally taking package/tour prices.
    idx = min(len(values) - 1, max(0, len(values) // 2))
    return round(values[idx], 2)


def tavily_ticket_price_grounding(
    plan: Dict[str, Any], requirements: Dict[str, Any], family_agents: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Use Tavily/raw online search specifically for attraction ticket prices.

    Tavily is still a search API, not a ticketing API, so the output is marked as
    online_search_estimate. If no price is found, we do NOT fall back to 40 CNY;
    the item is reported in grounding_issues and left for soft evaluation.
    """
    destination = safe_text(requirements.get("destination_city"))
    member_count = max(len(family_agents), 1)
    pois = collect_daily_poi_names(plan)
    ticket_plan: List[Dict[str, Any]] = []
    issues: List[str] = []
    seen = set()
    for poi in pois[:12]:
        name = safe_text(poi.get("poi_name"))
        key = _norm_name(name)
        if not name or key in seen:
            continue
        seen.add(key)
        if _is_non_ticket_activity(name):
            continue
        rule_price = estimate_default_ticket_price(name)
        should_check = _looks_like_paid_attraction(name) or rule_price > 0
        if not should_check:
            continue
        queries = [
            f"{destination} {name} 官方 门票 票价 成人票 2026",
            f"{destination} {name} official ticket price adult admission 2026",
        ]
        results: List[Dict[str, Any]] = []
        for query in queries:
            try:
                results.extend(online_search(query, max_results=max(2, ONLINE_SEARCH_MAX_RESULTS)))
            except Exception as exc:
                issues.append(f"tavily_ticket_search_failed:{name}:{safe_text(exc)}")
                break
        prices: List[float] = []
        urls: List[str] = []
        for r in results:
            if not isinstance(r, dict):
                continue
            urls.append(safe_text(r.get("url")))
            prices.extend(
                _extract_price_candidates_from_text(
                    " ".join([safe_text(r.get("title")), safe_text(r.get("content"))])
                )
            )
        price = _choose_reasonable_ticket_price(prices)
        source = "tavily_price_search"
        if price <= 0 and rule_price > 0:
            # Keep known paid-attraction rules such as Disney as fallback, but
            # mark them clearly as rule estimates.
            price = rule_price
            source = "rule_estimate_after_tavily_no_price"
        if price <= 0:
            issues.append(f"ticket_price_pending:{name}")
            continue
        ticket_plan.append(
            {
                "day": safe_int(poi.get("day"), 0),
                "poi_name": name,
                "ticket_price_per_person": round(price, 2),
                "participant_count": member_count,
                "estimated_total_cost": round(price * member_count, 2),
                "source_urls": list(dict.fromkeys([u for u in urls if u]))[:4],
                "price_source": source,
            }
        )
    return {"ticket_plan": ticket_plan, "grounding_issues": issues}


ROUTE_NON_STOP_KEYWORDS = [
    "准备返程",
    "返回",
    "返程",
    "离开",
    "抵达",
    "整理行李",
    "入住",
    "退房",
    "酒店享用早餐",
    "早餐后",
    "自由活动",
    "自由安排",
    "休息",
    "总结",
    "return",
    "departure",
    "arrival",
    "check in",
    "check-in",
    "check out",
    "check-out",
    "free time",
]


def _compact_query_text(text: Any, max_chars: Optional[int] = None) -> str:
    """Normalize and truncate search queries so Tavily receives stable requests."""
    text = re.sub(r"\s+", " ", safe_text(text)).strip()
    text = re.sub(r"[-–—]{2,}", "-", text)
    limit = max_chars or TRANSPORT_QUERY_MAX_CHARS
    if limit > 0 and len(text) > limit:
        return text[:limit].rstrip(" ，,;；、-")
    return text


def _is_placeholder_route_stop(value: Any) -> bool:
    text = safe_text(value).strip()
    low = text.lower()
    if not text:
        return True
    if len(text) <= 1:
        return True
    if any(k.lower() in low for k in ROUTE_NON_STOP_KEYWORDS):
        return True
    if any(k in text for k in ["短路线需求", "短途出租车", "路线需求", "具体路线", "当地餐厅", "家庭餐厅"]):
        return True
    # A pure meal label is not a real stop; a specific restaurant/area is kept.
    if text in ["早餐", "午餐", "晚餐", "用餐", "餐饮", "酒店", "市中心酒店"]:
        return True
    return False


def _route_stop_from_day_block(item: Any) -> List[str]:
    """Extract routeable stops from one day block.

    This is intentionally different from ticket filtering: restaurants and
    shopping areas may be route stops even though they should never get tickets.
    """
    stops: List[str] = []
    if isinstance(item, list):
        for act in item[:3]:
            stops.extend(_route_stop_from_day_block(act))
        return stops
    if isinstance(item, dict):
        for key in [
            "restaurant_name",
            "restaurant_area",
            "location",
            "area",
            "poi_name",
            "activity",
            "name",
            "name_or_type",
            "destination",
            "place",
        ]:
            value = safe_text(item.get(key))
            if value:
                stops.append(value)
                break
        return stops
    value = safe_text(item)
    if value:
        stops.append(value)
    return stops


def _dedupe_daily_stops(stops: List[str]) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for stop in stops:
        stop = safe_text(stop).strip()
        if _is_placeholder_route_stop(stop):
            continue
        key = _norm_name(stop)
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(stop)
    return cleaned


def _build_day_transport_query(destination: str, base: str, day_num: int, stops: List[str]) -> str:
    """Build a short per-day Tavily query instead of one long route-chain query."""
    key_stops = _dedupe_daily_stops(stops)[: max(1, TRANSPORT_QUERY_MAX_STOPS_PER_DAY)]
    stop_text = "、".join(key_stops) if key_stops else "主要景点"
    if TRANSPORT_GROUNDING_DAY_BY_DAY:
        query = f"{destination} 第{day_num}天 市内交通费用 地铁 出租车 从{base} 到 {stop_text} 路线 时间"
    else:
        query = f"{destination} 从 {base} 到 {stop_text} 再回酒店 地铁 出租车 路线 费用 时间"
    return _compact_query_text(query, TRANSPORT_QUERY_MAX_CHARS)


def _extract_transport_cost_from_results(
    results: List[Dict[str, Any]], member_count: int, mode_hint: str = ""
) -> float:
    text = " ".join(
        [
            safe_text(r.get("title")) + " " + safe_text(r.get("content"))
            for r in results
            if isinstance(r, dict)
        ]
    )
    values = _extract_price_candidates_from_text(text)
    values = [v for v in values if 0 < v <= 800]
    if values:
        # If the snippet likely mentions per-person metro fares, multiply by family.
        base = _choose_reasonable_ticket_price(values)
        if any(k in safe_text(mode_hint).lower() for k in ["metro", "地铁", "mtr", "bus", "公交"]):
            return round(max(base * member_count * 2, 10 * member_count), 2)
        return round(max(base, 10 * member_count), 2)
    # Conservative fallback from city fare knowledge; marked by source as estimate.
    if any(k in safe_text(mode_hint).lower() for k in ["metro", "地铁", "mtr", "bus", "公交"]):
        return round(12.0 * member_count, 2)
    return round(35.0 * member_count, 2)


def tavily_transport_cost_grounding(
    plan: Dict[str, Any], requirements: Dict[str, Any], family_agents: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Run day-by-day Tavily/raw online search for city transport costs.

    v54 change: route search is now explicitly day-by-day.  Earlier versions
    built long route-chain queries such as "hotel -> stop1 -> lunch -> stop2 ->
    dinner -> hotel".  Those are useful for the final output but too long and
    noisy for Tavily, and can trigger provider-side HTTP 432.  This function now:
    1) searches the city fare context once;
    2) extracts routeable stops for each day;
    3) sends one compact Tavily query per day using only the day's key stops;
    4) still outputs a complete daily chain for the itinerary;
    5) soft-fails each day independently if search returns no result/error.
    """
    destination = safe_text(requirements.get("destination_city"))
    member_count = max(len(family_agents), 1)
    transport_req = (
        requirements.get("transportation_requirements", {})
        if isinstance(requirements.get("transportation_requirements"), dict)
        else {}
    )
    hotel_req = (
        requirements.get("hotel_requirements", {})
        if isinstance(requirements.get("hotel_requirements"), dict)
        else {}
    )
    hotel = plan.get("hotel_plan", {}) if isinstance(plan.get("hotel_plan"), dict) else {}
    hotel_name = (
        safe_text(hotel.get("hotel_name"))
        or safe_text(transport_req.get("selected_hotel_name"))
        or safe_text(hotel_req.get("selected_hotel_name"))
    )
    base = (
        safe_text(hotel.get("nearby_metro_station"))
        or safe_text(transport_req.get("hotel_base_station"))
        or safe_text(hotel.get("hotel_area"))
        or safe_text(hotel_req.get("preferred_area"))
        or "酒店"
    )
    base_display = hotel_name or base
    daily_routes_req = (
        transport_req.get("daily_route_requirements", [])
        if isinstance(transport_req.get("daily_route_requirements", []), list)
        else []
    )
    per_day = (
        plan.get("per_day_itinerary", [])
        if isinstance(plan.get("per_day_itinerary", []), list)
        else []
    )
    daily_transport: List[Dict[str, Any]] = []
    issues: List[str] = []
    all_urls: List[str] = []

    fare_context_results: List[Dict[str, Any]] = []
    fare_query = _compact_query_text(f"{destination} 市内交通 地铁票价 出租车起步价 公共交通票价 2026", 120)
    try:
        fare_context_results = online_search(
            fare_query, max_results=max(1, min(ONLINE_SEARCH_MAX_RESULTS, 2))
        )
    except Exception as exc:
        issues.append(f"tavily_transport_fare_search_failed:{safe_text(exc)}")
    all_urls.extend([safe_text(r.get("url")) for r in fare_context_results if isinstance(r, dict)])

    req_by_day = {safe_int(x.get("day"), 0): x for x in daily_routes_req if isinstance(x, dict)}
    for idx, day_item in enumerate(per_day):
        if not isinstance(day_item, dict):
            continue
        day_num = safe_int(day_item.get("day"), idx + 1)
        stops: List[str] = []
        # Keep route stops day-local.  Restaurants/shopping areas are valid route
        # stops, but generic non-stops such as "准备返程" are removed later.
        for block in ["morning", "lunch", "afternoon", "dinner", "evening"]:
            stops.extend(_route_stop_from_day_block(day_item.get(block)))
        if not stops and day_num in req_by_day:
            stops = [
                safe_text(x)
                for x in req_by_day[day_num].get("destination_requirements", [])
                if safe_text(x)
            ]
        stops = _dedupe_daily_stops(stops)
        if not stops:
            issues.append(f"day{day_num}_transport_stops_missing")
            continue

        full_chain_stops = stops[:5]
        chain = [base_display] + full_chain_stops + [base_display]
        query = _build_day_transport_query(destination, base, day_num, full_chain_stops)
        route_results: List[Dict[str, Any]] = []
        try:
            # One compact query per day.  Keep max_results small to reduce Tavily
            # pressure and avoid repeated 432/429 errors during Module 7.
            route_results = online_search(
                query, max_results=max(1, min(ONLINE_SEARCH_MAX_RESULTS, 2))
            )
        except Exception as exc:
            issues.append(f"tavily_transport_route_search_failed:day{day_num}:{safe_text(exc)}")
        if not route_results:
            issues.append(f"day{day_num}_transport_search_no_result_or_soft_failed")

        urls = list(
            dict.fromkeys(
                [
                    safe_text(r.get("url"))
                    for r in route_results + fare_context_results
                    if isinstance(r, dict) and safe_text(r.get("url"))
                ]
            )
        )[:4]
        cost = _extract_transport_cost_from_results(
            route_results + fare_context_results,
            member_count,
            safe_text(day_item.get("transport_mode")) or "metro/taxi",
        )
        source = (
            "tavily_day_by_day_transport_search_estimate"
            if route_results or fare_context_results
            else "fallback_city_fare_estimate_after_tavily_soft_fail"
        )
        daily_transport.append(
            {
                "day": day_num,
                "origin": base_display,
                "stops": full_chain_stops,
                "destination": base_display,
                "route": "->".join(chain),
                "search_query": query,
                "estimated_travel_time_minutes": safe_int(
                    day_item.get("estimated_transport_minutes"), 0
                ),
                "estimated_cost_total": round(cost, 2),
                "cost_source": source,
                "comfort_note": "按天调用Tavily查询交通费用；若当天搜索失败，则用城市地铁/出租车基础票价软估算，具体路线和实时价格需出行前核验。",
                "source_urls": urls,
            }
        )
        all_urls.extend(urls)

    return {
        "transportation_plan": {
            "main_transport_mode": safe_text(
                plan.get("transportation_plan", {}).get("main_transport_mode")
            )
            if isinstance(plan.get("transportation_plan"), dict)
            else "地铁或出租车",
            "hotel_base_station": base,
            "daily_local_transport": daily_transport,
            "local_transport_total_cost": round(
                sum(_money(x.get("estimated_cost_total"), 0) for x in daily_transport), 2
            ),
            "transportation_realism_note": "交通费用按天调用Tavily搜索地铁/出租车/路线费用信息后估算；单日查询失败不会导致整个行程失败。",
            "transport_grounding_mode": "day_by_day",
            "source_urls": list(dict.fromkeys([u for u in all_urls if u]))[:8],
        },
        "transport_grounding_issues": issues,
    }


def restaurant_price_range_text(price_level: str = "medium") -> str:
    level = safe_text(price_level).lower() or "medium"
    lo, hi = PRICE_LEVEL_RANGES.get(level, PRICE_LEVEL_RANGES["medium"])
    return f"{int(lo)}-{int(hi)}元/人"


def restaurant_price_level_cn(price_level: str = "medium") -> str:
    return PRICE_LEVEL_CN.get(safe_text(price_level).lower(), PRICE_LEVEL_CN["medium"])


def infer_restaurant_type_for_destination(
    destination: Any, food_preferences: Optional[List[Any]] = None
) -> str:
    """Return a non-invented restaurant type for placeholder/repeated meals."""
    dest = safe_text(destination).lower()
    food_text = " ".join([safe_text(x).lower() for x in (food_preferences or [])])
    if "harbin" in dest or "哈尔滨" in dest:
        return "东北菜或饺子家庭餐厅"
    if "hong kong" in dest or "香港" in dest:
        if "buffet" in food_text or "自助" in food_text:
            return "自助餐或粤菜家庭餐厅"
        return "茶餐厅、粤菜或点心家庭餐厅"
    if "shanghai" in dest or "上海" in dest:
        if "western" in food_text or "西餐" in food_text:
            return "西餐或本帮菜家庭餐厅"
        return "本帮菜、江浙菜或西餐家庭餐厅"
    if "shenzhen" in dest or "深圳" in dest:
        return "粤菜、海鲜或商场家庭餐厅"
    return "当地家庭友好餐厅"


def build_generic_restaurant_description(
    destination: Any,
    area: Any,
    price_level: str = "medium",
    food_preferences: Optional[List[Any]] = None,
) -> str:
    area_text = safe_text(area)
    area_prefix = f"{area_text}附近" if area_text else f'{safe_text(destination) or "当地"}'
    return f"{area_prefix}{restaurant_price_level_cn(price_level)}{infer_restaurant_type_for_destination(destination, food_preferences)}（人均约{restaurant_price_range_text(price_level)}）"


def build_restaurant_area_fallback_type(
    destination: Any, price_level: str = "medium", food_preferences: Optional[List[Any]] = None
) -> str:
    """Return cuisine/type phrase without pretending it is a restaurant name.

    v57: when no reliable restaurant name is found, keep restaurant_name empty,
    output only a concrete nearby area plus cuisine/price range, and let Module 6
    softly penalize the missing specific restaurant.
    """
    return f"{restaurant_price_level_cn(price_level)}{infer_restaurant_type_for_destination(destination, food_preferences)}"


PLACEHOLDER_RESTAURANT_TERMS = [
    "family-friendly dinner",
    "family friendly dinner",
    "family restaurant",
    "local restaurant",
    "restaurant",
    "家庭友好 dinner",
    "家庭友好餐厅",
    "晚餐或中档",
    "早餐或低档",
    "午餐或中档",
    "普通餐饮",
    "未知",
    "当地美食餐厅",
    "本地餐厅",
    "本地晚餐",
    "中档餐厅",
    "高档餐厅",
    "休闲餐厅",
    "家庭餐厅",
    "儿童友好餐厅",
    "国际美食餐厅",
    "多样化餐饮",
    "多样化美食餐厅",
    "市中心中档",
    "市中心附近",
    "附近中等价位",
    "中等价位餐厅",
    "当地家庭友好餐厅",
    "粤菜家庭餐厅",
    "茶餐厅、粤菜",
    "东北地方美食餐厅",
    "家庭友好餐饮",
    "乐园餐厅",
    "公园餐厅",
    "景区餐厅",
    "园区餐厅",
    "主题公园餐厅",
    "动物园餐厅",
    "海洋公园餐厅",
    "上海野生动物园餐厅",
    "海昌极地海洋公园餐厅",
    "极地馆餐厅",
    "海洋王国餐厅",
    "欢乐谷餐厅",
    "黄鹤楼餐厅",
    "博物馆餐厅",
    "艺术馆餐厅",
    "米其林餐厅",
    "自助餐厅",
    "西餐厅",
    "中餐厅",
    "海鲜火锅",
    "潮汕牛肉火锅",
]

FREE_ATTRACTION_KEYWORDS = [
    "park",
    "公园",
    "avenue",
    "street",
    "road",
    "步行街",
    "海滨",
    "长廊",
    "promenade",
    "harbour",
    "harbor",
    "central",
    "market",
    "mall",
    "购物",
    "外滩",
    "南京路",
    "中央大街",
    "星光大道",
    "海港城",
    "九龙公园",
    "维多利亚公园",
]

DEFAULT_TICKET_PRICE_RULES = [
    (["disney", "迪士尼"], 550.0),
    (["ocean park", "海洋公园"], 420.0),
    (["wild animal", "野生动物园", "safari"], 165.0),
    (["shanghai tower", "上海中心"], 180.0),
    (["oriental pearl", "东方明珠"], 199.0),
    (["peak tram", "山顶缆车", "太平山顶"], 120.0),
    (["m+", "museum", "博物馆", "艺术馆"], 60.0),
    (["garden", "豫园"], 40.0),
    (["tiger", "东北虎"], 110.0),
]


def _norm_name(value: Any) -> str:
    text = safe_text(value).lower()
    text = re.sub(r"[\s\-_/（）()·•,，。:：]+", "", text)
    return text


def _is_placeholder_restaurant_name(value: Any) -> bool:
    raw = safe_text(value).strip()
    text = raw.lower()
    if not text:
        return True
    if any(term.lower() in text for term in PLACEHOLDER_RESTAURANT_TERMS):
        return True
    if re.search(r"(乐园|公园|景区|动物园|博物馆|艺术馆|极地馆|海洋王国|欢乐谷|黄鹤楼|野生动物园).{0,4}餐厅$", raw):
        return True
    if re.fullmatch(r".{0,6}(自助餐厅|西餐厅|中餐厅|家庭餐厅|休闲餐厅|高档餐厅|中档餐厅|茶餐厅|饭店|餐厅)", raw) and len(raw) <= 8:
        if not any(brand in raw for brand in ["翠华", "大家乐", "海底捞", "鼎泰丰", "小南国", "外婆家"]):
            return True
    return False


def clamp_restaurant_price(value: Any, price_level: str = "medium", name: Any = "") -> float:
    """Clamp restaurant per-person prices to a reasonable range.

    Search snippets and LLM extraction sometimes return total bill, hotel price,
    or currency-mixed values as restaurant price.  We keep the number realistic
    instead of letting one bad extraction distort the whole budget.

    When no reliable price is available we spread deterministically across the
    price-level range using the restaurant name, so different restaurants get
    different (but stable) per-person prices instead of all collapsing to the
    same midpoint ("每个都一样").
    """
    level = safe_text(price_level).lower() or "medium"
    lo, hi = PRICE_LEVEL_RANGES.get(level, PRICE_LEVEL_RANGES["medium"])
    price = _money(value, 0)
    # No reliable price (missing, zero, or implausibly small extraction).
    if price <= 0 or price < lo * 0.45:
        nm = safe_text(name)
        if nm:
            frac = (sum(ord(c) for c in nm) % 97) / 96.0
            return round(lo + frac * (hi - lo), 2)
        return round((lo + hi) / 2, 2)
    if price < lo:
        return round(lo, 2)
    if price > hi * 2.5:
        return round(hi, 2)
    if price > hi:
        return round(hi, 2)
    return round(price, 2)


def estimate_default_ticket_price(poi_name: Any) -> float:
    """Return a conservative ticket estimate for common paid attractions.

    Free/public places return 0. Paid attractions return an approximate per-person
    price only when online extraction did not provide one.  The estimate is also
    recorded with source='rule_estimate' so it is not confused with official data.
    """
    text = safe_text(poi_name).lower()
    if not text:
        return 0.0
    if any(k in text for k in FREE_ATTRACTION_KEYWORDS):
        return 0.0
    for keywords, price in DEFAULT_TICKET_PRICE_RULES:
        if any(k.lower() in text for k in keywords):
            return float(price)
    # v52: do not invent a generic 40 CNY ticket. Unknown prices are left as
    # 0 / pending verification, and Module 6 only applies a soft penalty.
    # This prevents meals, free time, return trips, and vague activities from
    # being printed as "门票约 40".
    return 0.0


def collect_daily_poi_names(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect POIs from the structured itinerary with day information."""
    output: List[Dict[str, Any]] = []
    seen = set()
    per_day = (
        plan.get("per_day_itinerary", [])
        if isinstance(plan.get("per_day_itinerary", []), list)
        else []
    )
    for idx, day in enumerate(per_day):
        if not isinstance(day, dict):
            continue
        day_num = safe_int(day.get("day"), idx + 1)
        for block in ["morning", "afternoon", "evening", "optional_activities"]:
            values = day.get(block, [])
            if not isinstance(values, list):
                values = [values] if safe_text(values) else []
            for item in values:
                if isinstance(item, dict):
                    name = (
                        safe_text(item.get("poi_name"))
                        or safe_text(item.get("activity"))
                        or safe_text(item.get("name"))
                    )
                else:
                    name = safe_text(item)
                key = (day_num, _norm_name(name))
                if name and key not in seen:
                    seen.add(key)
                    output.append({"day": day_num, "poi_name": name})
    return output


def normalize_ticket_plan(
    grounding: Dict[str, Any], plan: Dict[str, Any], family_agents: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Ensure paid attraction ticket prices are present and included in cost."""
    member_count = max(len(family_agents), 1)
    raw_tickets = (
        grounding.get("ticket_plan", [])
        if isinstance(grounding.get("ticket_plan", []), list)
        else []
    )
    by_name: Dict[str, Dict[str, Any]] = {}
    for item in raw_tickets:
        if not isinstance(item, dict):
            continue
        name = (
            safe_text(item.get("poi_name"))
            or safe_text(item.get("attraction"))
            or safe_text(item.get("name"))
        )
        if not name:
            continue
        key = _norm_name(name)
        if _is_non_ticket_activity(name):
            continue
        ticket_price = (
            _money(item.get("estimated_ticket_price_per_person"), 0)
            or _money(item.get("ticket_price_per_person"), 0)
            or _money(item.get("ticket_price"), 0)
        )
        if ticket_price <= 0:
            ticket_price = estimate_default_ticket_price(name)
        # If the only value is an unsupported generic 40-like estimate, remove it.
        if (
            ticket_price in [40, 40.0]
            and not item.get("source_urls")
            and not _looks_like_paid_attraction(name)
        ):
            continue
        participants = safe_int(item.get("participant_count"), member_count)
        if participants <= 0:
            participants = member_count
        normalized = {
            "poi_name": name,
            "ticket_price_per_person": round(ticket_price, 2),
            "participant_count": participants,
            "estimated_total_cost": round(ticket_price * participants, 2),
            "source_urls": item.get("source_urls", [])
            if isinstance(item.get("source_urls", []), list)
            else [],
            "price_source": "online_or_llm" if item.get("source_urls") else "rule_estimate",
        }
        by_name[key] = normalized

    # Add missing ticket entries for paid POIs in the itinerary.
    for poi in collect_daily_poi_names(plan):
        name = poi["poi_name"]
        key = _norm_name(name)
        if key in by_name:
            continue
        price = estimate_default_ticket_price(name)
        if price <= 0:
            continue
        by_name[key] = {
            "day": poi.get("day"),
            "poi_name": name,
            "ticket_price_per_person": round(price, 2),
            "participant_count": member_count,
            "estimated_total_cost": round(price * member_count, 2),
            "source_urls": [],
            "price_source": "rule_estimate",
        }

    tickets = list(by_name.values())
    grounding["ticket_plan"] = tickets
    return tickets


def attach_ticket_prices_to_per_day(
    plan: Dict[str, Any], tickets: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Copy ticket prices into per_day_itinerary activities for TXT export."""
    lookup = {_norm_name(item.get("poi_name")): item for item in tickets if isinstance(item, dict)}
    per_day = (
        plan.get("per_day_itinerary", [])
        if isinstance(plan.get("per_day_itinerary", []), list)
        else []
    )
    for day in per_day:
        if not isinstance(day, dict):
            continue
        for block in ["morning", "afternoon", "evening", "optional_activities"]:
            values = day.get(block, [])
            if not isinstance(values, list):
                continue
            for activity in values:
                if not isinstance(activity, dict):
                    continue
                name = safe_text(activity.get("poi_name")) or safe_text(activity.get("activity"))
                ticket = lookup.get(_norm_name(name))
                if ticket and _money(activity.get("ticket_price"), 0) <= 0:
                    activity["ticket_price"] = ticket.get("ticket_price_per_person", 0)
                    activity["ticket_price_source"] = ticket.get("price_source", "")
                    activity["ticket_participant_count"] = ticket.get("participant_count", 0)
    plan["per_day_itinerary"] = per_day
    return plan


def _free_city_attractions(destination: str) -> List[str]:
    """Distinct FREE real attractions for a city (ticket price 0), for budget swaps."""
    out: List[str] = []
    seen = set()
    for r in _load_category_cache("attractions"):
        if not isinstance(r, dict) or not _cache_record_matches_destination(r, destination):
            continue
        if _money(r.get("ticket_price_per_person"), 0) > 0:
            continue
        nm = safe_text(r.get("poi_name"))
        k = nm.lower()
        if nm and k not in seen:
            seen.add(k)
            out.append(nm)
    return out


def _family_must_include(family_agents: List[Dict[str, Any]]) -> set:
    must = set()
    for a in family_agents or []:
        if not isinstance(a, dict):
            continue
        c = a.get("constraints", {}) if isinstance(a.get("constraints"), dict) else {}
        mi = c.get("must_include") or a.get("must_include")
        if isinstance(mi, list):
            for x in mi:
                if safe_text(x):
                    must.add(_norm_name(x))
        elif safe_text(mi):
            must.add(_norm_name(mi))
    return must


def _swap_paid_attractions_for_budget(
    plan: Dict[str, Any], family_agents: List[Dict[str, Any]], member_count: int, target_drop: float
) -> float:
    """Replace the most expensive non-must-include paid attractions with free real
    ones until ~target_drop of ticket cost is removed.  Returns the amount dropped.
    Modifies plan['per_day_itinerary'] in place so the displayed plan matches."""
    if target_drop <= 0 or not ATTRACTION_BUDGET_RELIEF:
        return 0.0
    destination = safe_text(plan.get("destination_city"))
    must = _family_must_include(family_agents)
    free_pool = _free_city_attractions(destination)
    per_day = (
        plan.get("per_day_itinerary", []) if isinstance(plan.get("per_day_itinerary"), list) else []
    )
    used = set()
    candidates = []  # (price, day_idx, block, item_idx, name)
    for di, day_item in enumerate(per_day):
        if not isinstance(day_item, dict):
            continue
        for block in ("morning", "afternoon"):
            acts = day_item.get(block) if isinstance(day_item.get(block), list) else []
            for ii, act in enumerate(acts):
                if not isinstance(act, dict):
                    continue
                nm = safe_text(act.get("poi_name") or act.get("activity"))
                used.add(_norm_name(nm))
                price = _money(act.get("ticket_price"), 0) or _money(
                    act.get("ticket_price_per_person"), 0
                )
                if price > 0 and _norm_name(nm) not in must and not act.get("is_transit"):
                    candidates.append((price, di, block, ii, nm))
    candidates.sort(reverse=True)  # most expensive first
    dropped = 0.0
    fi = 0
    for price, di, block, ii, nm in candidates:
        if dropped >= target_drop:
            break
        # find next unused free attraction
        replacement = ""
        while fi < len(free_pool):
            cand = free_pool[fi]
            fi += 1
            if _norm_name(cand) not in used:
                replacement = cand
                used.add(_norm_name(cand))
                break
        if not replacement:
            break
        per_day[di][block][ii] = {
            "activity": replacement,
            "poi_name": replacement,
            "poi_type": "general",
            "ticket_price": 0,
            "budget_swapped_from_paid": nm,
        }
        dropped += price * member_count
    return dropped


_PLACE_DETAILS_CACHE_FILE = STRUCTURED_CACHE_FILES["attractions"].parent / "place_details.json"
_PLACE_DETAILS_CACHE: Optional[Dict[str, Any]] = None
_PLACE_DETAILS_DISABLED = False


def lookup_place_details(place_id: Any) -> Dict[str, Any]:
    """Legacy Google Place Details for a place_id → {name, address, lat, lng, types,
    opening_hours, price_level, rating}.  Cached on disk; trips a per-run breaker on
    auth errors; the API key is read from env and never logged.  Returns {} on failure
    (the New Places API and Routes API are NOT enabled on this project — legacy Details
    and Text Search are)."""
    global _PLACE_DETAILS_CACHE, _PLACE_DETAILS_DISABLED
    pid = safe_text(place_id)
    if not pid or _PLACE_DETAILS_DISABLED:
        return {}
    if _PLACE_DETAILS_CACHE is None:
        loaded = _read_json_file(_PLACE_DETAILS_CACHE_FILE, {})
        _PLACE_DETAILS_CACHE = loaded if isinstance(loaded, dict) else {}
    if pid in _PLACE_DETAILS_CACHE:
        return _PLACE_DETAILS_CACHE[pid]
    key = os.getenv("GOOGLE_PLACES_API_KEY", "") or os.getenv("GOOGLE_MAPS_API_KEY", "")
    if not key:
        _PLACE_DETAILS_DISABLED = True
        return {}
    params = urllib.parse.urlencode(
        {
            "place_id": pid,
            "language": "zh-CN",
            "key": key,
            "fields": "name,formatted_address,geometry,type,opening_hours,price_level,rating",
        }
    )
    try:
        data = _json_http_request(
            "https://maps.googleapis.com/maps/api/place/details/json?" + params
        )
    except Exception:
        return {}
    if safe_text(data.get("status")) == "REQUEST_DENIED":
        _PLACE_DETAILS_DISABLED = True
        return {}
    res = data.get("result", {}) if isinstance(data.get("result"), dict) else {}
    if not res:
        return {}
    geo = (
        res.get("geometry", {}).get("location", {}) if isinstance(res.get("geometry"), dict) else {}
    )
    oh = res.get("opening_hours", {}) if isinstance(res.get("opening_hours"), dict) else {}
    out = {
        "name": safe_text(res.get("name")),
        "address": safe_text(res.get("formatted_address")),
        "lat": geo.get("lat"),
        "lng": geo.get("lng"),
        "types": res.get("types", []) if isinstance(res.get("types"), list) else [],
        "opening_hours": oh.get("weekday_text", [])
        if isinstance(oh.get("weekday_text"), list)
        else [],
        "opening_periods": oh.get("periods", []) if isinstance(oh.get("periods"), list) else [],
        "price_level": res.get("price_level"),
        "rating": res.get("rating"),
    }
    _PLACE_DETAILS_CACHE[pid] = out
    _write_json_file(_PLACE_DETAILS_CACHE_FILE, _PLACE_DETAILS_CACHE)
    return out


def resolve_place_id(name: Any, city: Any, require_city: bool = True) -> str:
    """Best-effort place_id for a named entity via legacy Text Search.

    require_city=True (grounding use): only accept a result whose address is in the
    city.  require_city=False (validation use): accept the top name-matching result
    regardless of city, so the caller can DETECT a city mismatch instead of just
    missing the entity."""
    nm = safe_text(name)
    if not nm or is_generic_place_label(nm):
        return ""
    try:
        results = online_search(f"{nm} {safe_text(city)}".strip(), max_results=3)
    except Exception:
        return ""
    for r in results:
        if not isinstance(r, dict) or not safe_text(r.get("place_id")):
            continue
        if require_city and not _online_result_in_city(r.get("content"), city):
            continue
        return safe_text(r.get("place_id"))
    return ""


def city_attraction_pool(destination: str) -> List[str]:
    """Distinct real attraction names for a city (from the grounding seed/DB).

    Used to replace a duplicate POI in an itinerary with a different real
    attraction, so no attraction is visited more than once per family.
    """
    names: List[str] = []
    seen = set()
    for r in _load_category_cache("attractions"):
        if not isinstance(r, dict) or not _cache_record_matches_destination(r, destination):
            continue
        nm = safe_text(r.get("poi_name"))
        k = nm.lower()
        if nm and k not in seen:
            seen.add(k)
            names.append(nm)
    return names


# A label is "generic" when it is only a category word (optionally with a vague
# prefix), e.g. '自然公园' / '购物街' — but NOT a named place like '中山公园'.
_GENERIC_PLACE_RE = re.compile(
    r"^(自然|城市|当地|本地|附近|周边|周围|大型|知名|某个?|一[处个]|海洋)?\s*"
    r"(公园|购物街|商业街|步行街|购物中心|购物广场|商场|博物馆|美术馆|艺术馆|科技馆|水族馆|海洋馆|海滩|沙滩|海滨|海边|"
    r"夜市|美食街|小吃街|古镇|老街|古街|古城|动物园|植物园|广场|购物|游览|景点)$"
)
# English category labels, with optional generic adjective(s): 'history museum', 'local food street'.
_GENERIC_PLACE_RE_EN = re.compile(
    r"^(?:(?:local|natural|city|central|main|public|nearby|history|historical|art|science|modern|"
    r"national|contemporary|folk|children\'?s?|theme|water|amusement|maritime|ocean|wildlife|"
    r"botanical|famous|popular|traditional|old|cultural)\s+)*"
    r"(shopping(\s*(street|mall|center))?|food\s*street|night\s*market|museum|park|beach|zoo|square|"
    r"garden|aquarium|gallery|street|mall|market|temple)$",
    re.I,
)

# Per category: (key, generic terms that trigger it, seed POI-name keywords,
# online query suffix, regex to pull a concrete place name out of online snippets).
_CATEGORY_SPECS = [
    (
        "park",
        ("自然公园", "城市公园", "森林公园", "公园", "绿地", "park"),
        ("公园", "植物园", "森林公园", "湿地公园", "园博园"),
        "公园 绿地 自然 推荐",
        r"[一-鿿]{2,8}(?:公园|植物园|森林公园|湿地公园)",
    ),
    (
        "shopping_street",
        ("购物街", "商业街", "步行街", "购物", "shopping street", "shopping"),
        ("步行街", "商业街", "商业步行街", "老街", "美食街"),
        "步行街 商业街 购物 推荐",
        r"[一-鿿]{2,8}(?:步行街|商业街|商业步行街|老街|美食街)",
    ),
    (
        "mall",
        ("购物中心", "商场", "购物广场", "shopping mall", "mall"),
        ("购物中心", "商场", "购物广场", "万象城"),
        "购物中心 商场 推荐",
        r"[一-鿿]{2,10}(?:购物中心|商场|万象城)",
    ),
    (
        "museum",
        ("博物馆", "美术馆", "艺术馆", "museum"),
        ("博物馆", "纪念馆", "美术馆", "艺术馆", "科技馆"),
        "博物馆 美术馆 推荐",
        r"[一-鿿]{2,10}(?:博物馆|纪念馆|美术馆|艺术馆|科技馆)",
    ),
    (
        "aquarium",
        ("水族馆", "海洋馆", "aquarium"),
        ("水族馆", "海洋馆", "极地", "海洋世界", "海洋公园"),
        "水族馆 海洋馆 推荐",
        r"[一-鿿]{2,12}(?:水族馆|海洋馆|海洋世界|极地海洋)",
    ),
    (
        "beach",
        ("海滩", "沙滩", "海滨", "海边", "beach"),
        ("沙滩", "海滩", "海滨浴场"),
        "海滩 沙滩 推荐",
        r"[一-鿿]{2,8}(?:沙滩|海滩|海滨浴场)",
    ),
    (
        "night_market",
        ("夜市", "美食街", "小吃街", "night market", "food street"),
        ("夜市", "美食街", "小吃街", "小吃一条街", "美食一条街", "一条街"),
        "美食街 小吃街",
        r"[一-鿿]{2,10}(?:夜市|美食街|小吃街|小吃一条街|美食一条街|一条街)",
    ),
    ("zoo", ("动物园", "zoo"), ("动物园", "野生动物园"), "动物园 推荐", r"[一-鿿]{2,10}(?:动物园|野生动物园)"),
    (
        "oldtown",
        ("古镇", "古街", "老街", "古城"),
        ("古镇", "老街", "古街", "古城"),
        "古镇 古街 推荐",
        r"[一-鿿]{2,8}(?:古镇|古街|老街|古城)",
    ),
]
# Descriptive fragments that a category regex may capture from prose but are not real names.
_PLACE_NAME_NOISE = (
    "周边",
    "附近",
    "周围",
    "推荐",
    "一些",
    "这些",
    "各大",
    "各个",
    "比如",
    "以及",
    "还有",
    "当地",
    "本地",
    "大型",
    "知名",
    "热门",
    "著名",
)


def is_generic_place_label(text: Any) -> bool:
    """True for bare category labels (公园/购物街/history museum...), False for named places (中山公园/上海博物馆)."""
    t = safe_text(text).strip()
    return bool(t) and bool(_GENERIC_PLACE_RE.match(t) or _GENERIC_PLACE_RE_EN.match(t))


def _category_spec_for_label(label: Any):
    t = safe_text(label).lower()
    for spec in _CATEGORY_SPECS:
        if any(g.lower() in t for g in spec[1]):
            return spec
    return None


def _real_poi_by_spec(spec, pool: List[str], used: set, destination: str = "") -> str:
    """Concrete same-category real POI for a category spec: seed pool -> Google Places
    -> ''.  The single engine behind both generic-label grounding and the itinerary
    de-dup/diversity fetch.  An online hit is upserted into the attraction cache so
    the pool grows incrementally (only when a category is actually needed).  The
    online step is gated by ``ONLINE_GROUNDING_ENABLED``; the API key is read inside
    ``online_search`` and never logged."""
    if not spec:
        return ""
    _key, _generics, seed_kw, query_suffix, name_re = spec
    for poi in pool:  # 1) durable seed/DB attractions for this city
        if safe_text(poi).lower() not in used and any(kw in poi for kw in seed_kw):
            return poi
    if not (destination and ONLINE_GROUNDING_ENABLED):
        return ""
    try:  # 2) on-demand Google Places lookup for this exact category
        results = online_search(
            _compact_query_text(f"{destination} {query_suffix}", 100),
            max_results=max(4, ONLINE_SEARCH_MAX_RESULTS),
        )
    except Exception:
        results = []
    # google_places returns structured place results whose title IS the place name;
    # prefer a category-fitting title directly (avoids regex truncating a long name).
    for x in results:
        if not isinstance(x, dict):
            continue
        title = safe_text(x.get("title"))
        if (
            title
            and title.lower() not in used
            and not is_generic_place_label(title)
            and not any(w in title for w in _PLACE_NAME_NOISE)
            and _online_result_in_city(x.get("content"), destination)
            and (any(kw in title for kw in seed_kw) or re.search(name_re, title))
        ):
            url = safe_text(x.get("url"))
            _upsert_structured_cache(
                "attractions",
                {
                    "destination_city": safe_text(destination),
                    "poi_name": title,
                    "location_or_area": "",
                    "source_urls": [url] if url else [],
                    "price_source": "google_places_category_lookup",
                    "verify_note": "景点名来自地图搜索，开放时间/门票需出行前核验。",
                },
            )
            return title
    # fallback: pull a place name out of prose snippets (Tavily-style text).
    text = " ".join(
        safe_text(x.get("title")) + " " + safe_text(x.get("content"))
        for x in results
        if isinstance(x, dict)
    )
    cands = [m.group(0) for m in re.finditer(name_re, text)]
    cands = [
        nm
        for nm in cands
        if nm.lower() not in used
        and not is_generic_place_label(nm)
        and not any(w in nm for w in _PLACE_NAME_NOISE)
    ]
    if cands:
        return max(cands, key=len)  # a longer name is the more specific place
    return ""


def _specific_poi_for_label(label: Any, pool: List[str], used: set, destination: str = "") -> str:
    """Concrete same-category real POI for a generic label (公园→中山公园…)."""
    return _real_poi_by_spec(_category_spec_for_label(label), pool, used, destination)


def find_real_poi_by_category(
    destination: Any,
    category_key: str,
    used: Optional[set] = None,
    pool: Optional[List[str]] = None,
) -> str:
    """A real POI of a given category (`_CATEGORY_SPECS` key) for a city — seed pool
    first, then an on-demand Google Places lookup.  Returns '' when none is found."""
    used = used if isinstance(used, set) else set()
    spec = next((s for s in _CATEGORY_SPECS if s[0] == category_key), None)
    if not spec:
        return ""
    if pool is None:
        try:
            pool = city_attraction_pool(destination)
        except Exception:
            pool = []
    return _real_poi_by_spec(spec, pool, used, safe_text(destination))


# When the itinerary de-dup needs to replace an over-represented category, try these
# varied "things to do" categories (museum last = last resort), skipping the ones that
# map to the over-cap category being avoided.
_DEDUP_FETCH_ORDER = (
    "park",
    "oldtown",
    "shopping_street",
    "zoo",
    "aquarium",
    "beach",
    "night_market",
    "mall",
    "museum",
)
_DEDUP_AVOID_SPEC_KEYS = {
    "museum": {"museum"},
    "park": {"park"},
    "street": {"shopping_street", "mall", "night_market"},
    "temple": {"oldtown"},
    "themepark": {"zoo"},
}


def fetch_alternative_real_poi(
    destination: Any, used: Optional[set] = None, avoid_category: str = ""
) -> str:
    """On-demand real POI of a DIFFERENT, under-represented category — used by the
    itinerary de-dup/diversity when the local pool has no fresh alternative.  Looks
    up by the specific missing category via Google Places (gated by online grounding)."""
    used = used if isinstance(used, set) else set()
    dest = safe_text(destination)
    if not dest:
        return ""
    try:
        pool = city_attraction_pool(dest)
    except Exception:
        pool = []
    avoid = _DEDUP_AVOID_SPEC_KEYS.get(safe_text(avoid_category), set())
    for key in _DEDUP_FETCH_ORDER:
        if key in avoid:
            continue
        poi = find_real_poi_by_category(dest, key, used, pool)
        if poi and safe_text(poi).lower() not in used:
            return poi
    return ""


def ground_generic_and_split_activities(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Replace bare category labels (公园/购物街/...) in day activities AND split
    activities with a concrete same-category real POI from the city pool, so '自然公园'
    becomes e.g. '中山公园' and split branches name real places.  No-ops when the pool
    lacks a matching place (the label is left unchanged, never worsened)."""
    # Gated by the grounding ablation: when grounding is off, leave generic labels
    # as-is so the no-grounding condition genuinely degrades (and the evaluator
    # penalizes the remaining generic labels).
    if not ONLINE_GROUNDING_ENABLED:
        return plan
    destination = safe_text(plan.get("destination_city"))
    if not destination:
        return plan
    pool = city_attraction_pool(destination)
    days = (
        plan.get("per_day_itinerary", []) if isinstance(plan.get("per_day_itinerary"), list) else []
    )
    used = set()
    for day in days:
        if not isinstance(day, dict):
            continue
        for block in ("morning", "afternoon", "evening"):
            for it in day.get(block) or []:
                nm = (
                    safe_text(it.get("poi_name") or it.get("activity"))
                    if isinstance(it, dict)
                    else safe_text(it)
                )
                if nm and not is_generic_place_label(nm):
                    used.add(nm.lower())

    def _fix(value: Any) -> str:
        nm = safe_text(value)
        if nm and is_generic_place_label(nm):
            repl = _specific_poi_for_label(nm, pool, used, destination)
            if repl:
                used.add(repl.lower())
                return repl
        return nm

    split_blocks = list(plan.get("split_activities") or [])
    for day in days:
        if not isinstance(day, dict):
            continue
        for block in ("morning", "afternoon", "evening"):
            for it in day.get(block) or []:
                if isinstance(it, dict):
                    new = _fix(it.get("poi_name") or it.get("activity"))
                    if new and new != safe_text(it.get("poi_name") or it.get("activity")):
                        it["poi_name"] = new
                        it["activity"] = new
        split_blocks.extend(day.get("split_activities") or [])
    for sp in split_blocks:
        if not isinstance(sp, dict):
            continue
        # role_assignment is the composed sentence the TXT renders (e.g.
        # 'father 参加“history museum”；...'); keep it in sync with the grounded fields.
        role_text = safe_text(sp.get("role_assignment"))
        for fld in ("activity", "alternative_for_non_participants"):
            old = safe_text(sp.get(fld))
            new = _fix(old)
            if new and new != old:
                if fld == "activity":
                    # Preserve the original conflict term (e.g. 'history museum') so the
                    # split<->conflict matcher (split_activity_resolves_conflict_term,
                    # which scans the split's text) still links this grounded split to
                    # the conflict it resolves — otherwise the conflict is wrongly left
                    # 'unresolved' and the plan fails on unresolved_conflict_penalty.
                    sp["conflict_term"] = old
                sp[fld] = new
                if old and old in role_text:
                    role_text = role_text.replace(old, new)
        # Ground any bare category still quoted inside the sentence (参加“X”).
        for m in re.finditer(r'[“"]([^”"]{2,40})[”"]', role_text):
            inner = m.group(1)
            new_inner = _fix(inner)
            if new_inner and new_inner != inner:
                role_text = role_text.replace(inner, new_inner)
        if role_text and role_text != safe_text(sp.get("role_assignment")):
            sp["role_assignment"] = role_text
    return plan


def _city_restaurant_pool(destination: str) -> List[Dict[str, Any]]:
    """All http-sourced seed restaurants for a city, in a stable order.

    Used as a last-resort rotation pool: when a city has fewer distinct restaurants
    than the trip has meals (e.g. 3 Harbin restaurants for 6 meals), we reuse real
    DB names round-robin instead of printing '不指定具体餐厅'.
    """
    pool: List[Dict[str, Any]] = []
    for r in _load_category_cache("restaurants"):
        if not isinstance(r, dict) or not _cache_record_matches_destination(r, destination):
            continue
        if not safe_text(r.get("restaurant_name")):
            continue
        srcs = r.get("source_urls") if isinstance(r.get("source_urls"), list) else []
        if not any(safe_text(u).startswith(("http://", "https://")) for u in srcs):
            continue
        pool.append(r)
    pool.sort(key=lambda r: _norm_name(r.get("restaurant_name")))
    return pool


def _augment_restaurant_pool(
    destination: str, target_count: int, have_norms: set
) -> List[Dict[str, Any]]:
    """Fetch additional DISTINCT real restaurants for a city from the online provider.

    Called when the seed pool cannot give every meal its own restaurant.  Each
    fetched place is quality-gated (rating / name / source) and persisted to the
    grounding_seed so the city's coverage grows and next time needs no API call.
    Returns new pool entries (deduplicated against ``have_norms``).
    """
    if not ONLINE_GROUNDING_ENABLED or target_count <= 0:
        return []
    queries = [
        f"{destination} 餐厅 推荐 家庭 当地特色",
        f"{destination} 美食 特色餐厅",
        f"{destination} 家庭聚餐 餐厅",
    ]
    extra: List[Dict[str, Any]] = []
    seen = set(have_norms)
    for q in queries:
        if len(extra) >= target_count:
            break
        try:
            results = online_search(q, max_results=max(target_count + 4, 10))
        except Exception as exc:
            if _is_api_configuration_error(exc) and REQUIRE_ONLINE_GROUNDING:
                raise
            continue
        for r in results or []:
            if not isinstance(r, dict):
                continue
            prov = safe_text(r.get("provider"))
            name = (
                safe_text(r.get("title"))
                if prov == "google_places"
                else _clean_search_title_to_place_name(r.get("title"))
            )
            url = safe_text(r.get("url"))
            norm = _norm_name(name)
            if not norm or norm in seen:
                continue
            rec = {
                "destination_city": destination,
                "restaurant_name": name,
                "restaurant_type": "家庭友好餐厅",
                "area": "",
                "source_urls": [url] if url else [],
                "rating": r.get("rating"),
                "estimated_price_per_person": 0,
                "category": "restaurants",
            }
            if not evaluate_grounding_candidate("restaurants", rec)["passed"]:
                continue
            seen.add(norm)
            _upsert_structured_cache("restaurants", rec)  # quality-gated -> grows seed
            extra.append(rec)
            if len(extra) >= target_count:
                break
    return extra


def normalize_restaurants(
    grounding: Dict[str, Any], plan: Dict[str, Any], family_agents: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Make restaurant assignments diverse, specific when possible, and honest when not.

    v57: If no reliable concrete restaurant name is available, do not place a
    generic restaurant description in restaurant_name/name_or_type.  Keep
    restaurant_name empty and export an area-level fallback such as
    "五大道/天津大学站附近 + 中等价位韩国料理 + price range".  Module 6 then counts the
    missing concrete name and applies a soft penalty.
    """
    member_count = max(len(family_agents), 1)
    price_level = safe_text(plan.get("restaurant_price_level")) or "medium"
    destination = safe_text(plan.get("destination_city")) or "当地"
    food_preferences = []
    for agent in family_agents:
        profile = agent.get("profile", {}) if isinstance(agent, dict) else {}
        prefs = profile.get("dietary_preference", [])
        if isinstance(prefs, list):
            food_preferences.extend(prefs)
    restaurants = (
        grounding.get("restaurants", [])
        if isinstance(grounding.get("restaurants", []), list)
        else []
    )

    def _norm_meal(raw: str) -> str:
        m = safe_text(raw).lower()
        if m in ("lunch", "dinner"):
            return m
        return "dinner" if "晚" in safe_text(raw) else "lunch"

    # Index whatever the online/LLM grounding produced by (day, meal).
    days = max(safe_int(plan.get("day"), 1), 1)
    online_by_slot: Dict[tuple, Dict[str, Any]] = {}
    for item in restaurants:
        if isinstance(item, dict):
            online_by_slot[(safe_int(item.get("day"), 0), _norm_meal(item.get("meal")))] = item

    # Dedup-optimal fill pool: distinct, http-sourced DB restaurants for the city.
    pool = _city_restaurant_pool(destination)
    # Goal: give EVERY meal its own distinct restaurant.  The number of distinct
    # names available must therefore reach the number of meals (days * 2).  Count
    # the distinct concrete names the online grounding already supplied, and top up
    # the pool from the online provider so no meal is left without a restaurant.
    needed_meals = days * 2
    distinct_online = {
        _norm_name(it.get("restaurant_name") or it.get("name"))
        for it in online_by_slot.values()
        if _norm_name(it.get("restaurant_name") or it.get("name"))
        and not _is_placeholder_restaurant_name(it.get("restaurant_name") or it.get("name"))
    }
    have_norms = set(distinct_online) | {_norm_name(r.get("restaurant_name")) for r in pool}
    shortfall = needed_meals - len(have_norms)
    if shortfall > 0:
        try:
            pool = pool + _augment_restaurant_pool(destination, shortfall, have_norms)
        except Exception:
            pass
    name_use_count: Dict[str, int] = {}

    def _choose_from_pool(area: str) -> Dict[str, Any]:
        """Pick an UNUSED DB restaurant (no name may repeat within a family).

        Returns {} once every distinct DB restaurant for the city is taken; the
        remaining meals then stay honestly ungrounded rather than repeating a name.
        """
        area_l = safe_text(area).lower()
        unused = [
            r for r in pool if name_use_count.get(_norm_name(r.get("restaurant_name")), 0) == 0
        ]
        if not unused:
            return {}

        def score(r: Dict[str, Any]):
            r_area = safe_text(r.get("area")).lower()
            area_match = 1 if (area_l and r_area and (area_l in r_area or r_area in area_l)) else 0
            return (-area_match, _norm_name(r.get("restaurant_name")))

        return sorted(unused, key=score)[0]

    normalized: List[Dict[str, Any]] = []
    for day in range(1, days + 1):
        for meal in ("lunch", "dinner"):
            item = online_by_slot.get((day, meal), {})
            area = (
                safe_text(item.get("area"))
                or safe_text(item.get("restaurant_area"))
                or safe_text(item.get("address_or_location"))
            )
            area = _clean_restaurant_area_anchor(
                area, destination
            ) or _specific_restaurant_area_for_day_meal(plan, day, meal, destination)
            online_name = safe_text(item.get("restaurant_name")) or safe_text(item.get("name"))
            online_norm = _norm_name(online_name)
            online_ok = (
                bool(online_norm)
                and not _is_placeholder_restaurant_name(online_name)
                and name_use_count.get(online_norm, 0) == 0
            )

            if online_ok:
                name = online_name
                rtype = safe_text(item.get("restaurant_type")) or "家庭友好餐厅"
                status = "specific_name_found"
                srcs = (
                    item.get("source_urls", []) if isinstance(item.get("source_urls"), list) else []
                )
                price = clamp_restaurant_price(
                    item.get("estimated_price_per_person") or item.get("price_per_person"),
                    price_level,
                    name=name,
                )
                note = "在线查询到具体餐厅名，价格需核验。"
            else:
                pick = _choose_from_pool(area)
                pick_name = safe_text(pick.get("restaurant_name")) if pick else ""
                if pick_name:
                    name = pick_name
                    rtype = safe_text(pick.get("restaurant_type")) or "家庭友好餐厅"
                    area = safe_text(pick.get("area")) or area
                    srcs = (
                        pick.get("source_urls", [])
                        if isinstance(pick.get("source_urls"), list)
                        else []
                    )
                    pk_price = safe_float(pick.get("estimated_price_per_person"), 0)
                    price = clamp_restaurant_price(
                        pk_price if pk_price > 0 else item.get("estimated_price_per_person"),
                        price_level,
                        name=name,
                    )
                    status = "specific_name_found"
                    note = "取自本地数据库（含来源链接），价格与营业时间需出行前核验。"
                else:
                    name = ""
                    rtype = build_restaurant_area_fallback_type(
                        destination, price_level, food_preferences
                    )
                    status = "area_fallback_no_specific_name"
                    srcs = []
                    price = clamp_restaurant_price(
                        item.get("estimated_price_per_person"), price_level
                    )
                    note = "未取得可靠具体餐厅名；仅给出当天景点/酒店站点附近的用餐范围，不编造店名；Module 6 会按缺少具体餐厅软扣分。"

            if name:
                name_use_count[_norm_name(name)] = name_use_count.get(_norm_name(name), 0) + 1
            normalized.append(
                {
                    "day": day,
                    "meal": meal,
                    "restaurant_name": name,
                    "name_or_type": name,
                    "restaurant_type": rtype,
                    "area": area,
                    "address_or_location": area,
                    "specific_restaurant_found": bool(name),
                    "restaurant_grounding_failure": not bool(name),
                    "restaurant_grounding_status": status,
                    "estimated_price_per_person": price,
                    "estimated_total_cost": round(price * member_count, 2),
                    "price_level": price_level,
                    "source_urls": srcs,
                    "comfort_note": note,
                }
            )

    # No-repeat policy: a restaurant name is never reused within one family.  When
    # the city's distinct DB restaurants run out, the remaining meals stay honestly
    # ungrounded (area-level), which Module 6 counts as a soft penalty.  Whether
    # this stays within MAX_UNGROUNDED_MEALS depends on how many distinct restaurants
    # the city has — the meal-count-aware online gating fetches more when the seed
    # is short, so well-covered cities still end up with ≤ MAX_UNGROUNDED_MEALS gaps.
    grounding["restaurants"] = normalized
    return normalized


def _fallback_hotel_grounding(
    plan: Dict[str, Any],
    requirements: Dict[str, Any],
    search_results: List[Dict[str, Any]],
    issue: str = "",
) -> Dict[str, Any]:
    """Build a non-crashing hotel grounding fallback.

    This fallback is deliberately conservative: it reuses an existing hotel from
    the generated plan when present, otherwise it returns an empty/area-level
    hotel object and lets Module 6 apply grounding penalties.  It never invents a
    concrete hotel name from unsupported snippets.
    """
    hotel_req = (
        requirements.get("hotel_requirements", {})
        if isinstance(requirements.get("hotel_requirements"), dict)
        else {}
    )
    plan_hotel = plan.get("hotel_plan", {}) if isinstance(plan.get("hotel_plan"), dict) else {}
    destination = (
        safe_text(requirements.get("destination_city"))
        or safe_text(plan.get("destination_city"))
        or "目的地"
    )
    days = max(safe_int(requirements.get("days") or plan.get("day"), 1), 1)
    nights = max(days - 1, 1)
    price_level = (
        safe_text(plan_hotel.get("price_level"))
        or safe_text(hotel_req.get("price_level"))
        or "medium"
    )
    hotel_name = safe_text(plan_hotel.get("hotel_name")) or safe_text(plan_hotel.get("name"))
    if is_generic_hotel_text(hotel_name):
        hotel_name = ""
    area = (
        safe_text(plan_hotel.get("hotel_area"))
        or safe_text(plan_hotel.get("area"))
        or safe_text(hotel_req.get("selected_hotel_area"))
        or safe_text(hotel_req.get("preferred_area"))
        or destination
    )
    station = (
        safe_text(plan_hotel.get("nearby_metro_station"))
        or safe_text(plan_hotel.get("nearby_station"))
        or safe_text(hotel_req.get("selected_hotel_nearby_station"))
    )
    price = _money(
        plan_hotel.get("estimated_price_per_night")
        or plan_hotel.get("price_per_night")
        or hotel_req.get("max_price_per_night"),
        0,
    )
    source_urls: List[str] = []
    for item in search_results[:5]:
        if isinstance(item, dict) and safe_text(item.get("url")):
            source_urls.append(safe_text(item.get("url")))
    # The itinerary must always name a concrete hotel; when grounding produced none,
    # fall back to a real hotel from the local seed DB (budget-fit, else cheapest).
    if not hotel_name:
        members = max(1, safe_int(requirements.get("member_count"), 0))
        budget = safe_float(requirements.get("budget_limit") or requirements.get("budget"), 0.0)
        seed_hotel = _pick_seed_component(
            "hotels", destination, budget, days, members
        ) or _cheapest_city_hotel(destination)
        cand = safe_text(seed_hotel.get("hotel_name")) if isinstance(seed_hotel, dict) else ""
        if cand and not is_generic_hotel_text(cand):
            hotel_name = cand
            area = safe_text(seed_hotel.get("area")) or safe_text(seed_hotel.get("address")) or area
            station = station or safe_text(seed_hotel.get("nearby_metro_station"))
            seed_price = _money(seed_hotel.get("estimated_price_per_night"), 0)
            if price <= 0 and seed_price > 0:
                price = seed_price
            for u in seed_hotel.get("source_urls") or []:
                if safe_text(u):
                    source_urls.append(safe_text(u))
    hotel_plan = {
        "hotel_name": hotel_name,
        "hotel_area": area,
        "area": area,
        "address_or_landmark": safe_text(plan_hotel.get("address_or_landmark"))
        or safe_text(plan_hotel.get("address"))
        or area,
        "nearby_metro_station": station,
        "metro_lines_near_hotel": plan_hotel.get("metro_lines_near_hotel", [])
        if isinstance(plan_hotel.get("metro_lines_near_hotel", []), list)
        else [],
        "nearest_itinerary_anchor": safe_text(hotel_req.get("itinerary_anchor_pois", [""])[0])
        if isinstance(hotel_req.get("itinerary_anchor_pois", []), list)
        and hotel_req.get("itinerary_anchor_pois")
        else "",
        "estimated_distance_to_anchor": "未核验",
        "price_per_night": round(price, 2),
        "estimated_price_per_night": round(price, 2),
        "nights": nights,
        "hotel_total_cost": round(price * nights, 2) if price > 0 else 0,
        "price_level": price_level,
        "comfort_features": plan_hotel.get("comfort_features", [])
        if isinstance(plan_hotel.get("comfort_features", []), list)
        else [],
        "family_friendly_features": plan_hotel.get("family_friendly_features", [])
        if isinstance(plan_hotel.get("family_friendly_features", []), list)
        else [],
        "accessibility_features": plan_hotel.get("accessibility_features", [])
        if isinstance(plan_hotel.get("accessibility_features", []), list)
        else [],
        "hotel_selection_reason": safe_text(plan_hotel.get("hotel_selection_reason"))
        or safe_text(plan_hotel.get("reason"))
        or "未取得可靠具体酒店核验结果；保留行程草案中的酒店/区域信息，Module 6 按 grounding 质量软扣分。",
        "source_urls": list(dict.fromkeys(source_urls)),
        "verification_note": "酒店价格、房态、距离和附近站点需预订前再次核验。",
    }
    issues = []
    if issue:
        issues.append(issue)
    if not hotel_name:
        issues.append("hotel_specific_name_missing_area_fallback_used")
    return {"hotel_plan": hotel_plan, "hotel_grounding_issues": issues}


def extract_hotel_grounding_with_llm(
    plan: Dict[str, Any], requirements: Dict[str, Any], search_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Extract nearest-qualified hotel grounding from online snippets.

    v59/v15 fix: v14 called this function but did not define it, causing every
    Module 5 plan to fail with `name 'extract_hotel_grounding_with_llm' is not
    defined`.  Keep this function soft-failing: LLM extraction/search problems
    return a hotel fallback plus issues, not an exception that kills the whole
    candidate.
    """
    hotel_req = (
        requirements.get("hotel_requirements", {})
        if isinstance(requirements.get("hotel_requirements"), dict)
        else {}
    )
    plan_text = compact_json_for_prompt(plan, max_chars=3600)
    hotel_requirements_text = compact_json_for_prompt(
        {
            "destination_city": requirements.get("destination_city"),
            "days": requirements.get("days"),
            "budget_limit": requirements.get("budget_limit"),
            "hotel_requirements": hotel_req,
        },
        max_chars=2400,
    )
    search_results_text = compact_search_results(
        search_results, max_chars=4200, max_items=8, content_chars=260
    )
    prompt = build_hotel_grounding_prompt(
        plan_text=plan_text,
        hotel_requirements_text=hotel_requirements_text,
        search_results_text=search_results_text,
    )
    try:
        response = call_generation_json(
            messages=[{"role": "user", "content": prompt}],
            model=ONLINE_GROUNDING_MODEL_NAME,
            max_tokens=min(ONLINE_GROUNDING_MAX_TOKENS, 2600),
            temperature=0.0,
        )
        data = response.get("data", {}) if isinstance(response, dict) else {}
        if not isinstance(data, dict):
            return _fallback_hotel_grounding(
                plan, requirements, search_results, "hotel_grounding_llm_returned_non_dict"
            )
        hotel = data.get("hotel_plan", {}) if isinstance(data.get("hotel_plan"), dict) else {}
        if not hotel:
            return _fallback_hotel_grounding(
                plan, requirements, search_results, "hotel_grounding_llm_returned_no_hotel_plan"
            )
        # Normalize keys so downstream Module 5/6/TXT export all read the same fields.
        hotel_name = safe_text(hotel.get("hotel_name")) or safe_text(hotel.get("name"))
        hotel_area = (
            safe_text(hotel.get("hotel_area"))
            or safe_text(hotel.get("area"))
            or safe_text(hotel.get("location"))
        )
        price = _money(hotel.get("estimated_price_per_night") or hotel.get("price_per_night"), 0)
        nights = safe_int(
            hotel.get("nights"),
            max(safe_int(requirements.get("days") or plan.get("day"), 1) - 1, 1),
        )
        if nights <= 0:
            nights = max(safe_int(requirements.get("days") or plan.get("day"), 1) - 1, 1)
        source_urls = (
            hotel.get("source_urls", []) if isinstance(hotel.get("source_urls", []), list) else []
        )
        if not source_urls:
            for item in search_results[:5]:
                if isinstance(item, dict) and safe_text(item.get("url")):
                    source_urls.append(safe_text(item.get("url")))
        hotel["hotel_name"] = hotel_name
        hotel["hotel_area"] = hotel_area
        hotel["area"] = hotel_area
        hotel["nearby_metro_station"] = safe_text(hotel.get("nearby_metro_station")) or safe_text(
            hotel.get("nearest_metro_station")
        )
        hotel["price_per_night"] = round(price, 2)
        hotel["estimated_price_per_night"] = round(price, 2)
        hotel["nights"] = nights
        hotel["hotel_total_cost"] = (
            round(_money(hotel.get("hotel_total_cost"), 0) or price * nights, 2)
            if price > 0
            else _money(hotel.get("hotel_total_cost"), 0)
        )
        hotel["source_urls"] = list(
            dict.fromkeys([safe_text(x) for x in source_urls if safe_text(x)])
        )[:8]
        hotel["verification_note"] = (
            safe_text(hotel.get("verification_note")) or "价格、房态、距离和附近站点需预订前再次核验。"
        )
        issues = (
            data.get("hotel_grounding_issues", [])
            if isinstance(data.get("hotel_grounding_issues", []), list)
            else []
        )
        if not hotel_name or is_generic_hotel_text(hotel_name):
            fallback = _fallback_hotel_grounding(
                plan,
                requirements,
                search_results,
                "hotel_grounding_llm_missing_specific_hotel_name",
            )
            # Preserve any LLM issues and search urls, but do not invent a name.
            fallback["hotel_grounding_issues"] = list(
                dict.fromkeys(fallback.get("hotel_grounding_issues", []) + issues)
            )
            return fallback
        return {"hotel_plan": hotel, "hotel_grounding_issues": issues}
    except Exception as exc:
        return _fallback_hotel_grounding(
            plan, requirements, search_results, f"hotel_grounding_llm_failed: {safe_text(exc)}"
        )


def extract_restaurant_ticket_grounding_with_llm(
    plan: Dict[str, Any], requirements: Dict[str, Any], search_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Extract restaurants and ticket prices from a compact set of search snippets.

    This replaces the old v48 deterministic restaurant fallback as the primary path.
    The deterministic builder is kept only as a last-resort fallback when the LLM
    call fails or returns no usable meal/ticket entries.
    """
    plan_text = compact_json_for_prompt(plan, max_chars=3000)
    requirements_text = compact_json_for_prompt(
        {
            "destination_city": requirements.get("destination_city"),
            "days": requirements.get("days"),
            "restaurant_requirements": requirements.get("restaurant_requirements", {}),
            "ticket_requirements": requirements.get("ticket_requirements", {}),
        },
        max_chars=2200,
    )
    search_results_text = compact_search_results(
        search_results, max_chars=3600, max_items=8, content_chars=220
    )
    prompt = build_online_grounding_prompt(
        plan_text=plan_text,
        requirements_text=requirements_text,
        search_results_text=search_results_text,
    )
    try:
        response = call_generation_json(
            messages=[{"role": "user", "content": prompt}],
            model=ONLINE_GROUNDING_MODEL_NAME,
            max_tokens=min(ONLINE_GROUNDING_MAX_TOKENS, 3000),
            temperature=0.0,
        )
        data = response.get("data", {})
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        return {"grounding_issues": [f"restaurant_ticket_grounding_llm_failed: {safe_text(exc)}"]}


def build_plan_context_for_post_generation_transport(
    plan: Dict[str, Any], hotel_grounding: Dict[str, Any], other_grounding: Dict[str, Any]
) -> Dict[str, Any]:
    """Build a lightweight full-itinerary context for transport stitching.

    Transport grounding should happen after the hotel and day-level POI/restaurant
    choices are available.  This function does not finalize costs; it only enriches
    a copy of the plan so the transport LLM/API can connect real stops.
    """
    context = copy.deepcopy(plan)
    hotel_plan = (
        context.get("hotel_plan", {}) if isinstance(context.get("hotel_plan"), dict) else {}
    )
    grounded_hotel = (
        hotel_grounding.get("hotel_plan", {})
        if isinstance(hotel_grounding.get("hotel_plan"), dict)
        else {}
    )
    if grounded_hotel:
        hotel_plan.update(grounded_hotel)
        context["hotel_plan"] = hotel_plan
    restaurants = (
        other_grounding.get("restaurants", [])
        if isinstance(other_grounding.get("restaurants", []), list)
        else []
    )
    if restaurants:
        context["grounded_restaurants"] = restaurants
    restaurant_lookup = {}
    for item in restaurants:
        if isinstance(item, dict):
            restaurant_lookup[
                (safe_int(item.get("day"), 0), safe_text(item.get("meal")).lower())
            ] = item
    per_day = (
        context.get("per_day_itinerary", [])
        if isinstance(context.get("per_day_itinerary", []), list)
        else []
    )
    for idx, day_item in enumerate(per_day):
        if not isinstance(day_item, dict):
            continue
        day_num = safe_int(day_item.get("day"), idx + 1)
        for meal_key in ["lunch", "dinner"]:
            source = restaurant_lookup.get((day_num, meal_key))
            if not source:
                continue
            meal = day_item.get(meal_key, {}) if isinstance(day_item.get(meal_key), dict) else {}
            meal["restaurant_name"] = safe_text(source.get("restaurant_name")) or safe_text(
                source.get("name_or_type")
            )
            meal["restaurant_area"] = safe_text(source.get("area"))
            meal["restaurant_type"] = safe_text(source.get("restaurant_type"))
            day_item[meal_key] = meal
        per_day[idx] = day_item
    context["per_day_itinerary"] = per_day
    if isinstance(other_grounding.get("ticket_plan"), list):
        context["ticket_plan"] = other_grounding.get("ticket_plan")
    context["post_generation_transport_stage"] = True
    return context


# Raw 'cuisine+price' meal labels (e.g. '地方特色+中等') that must NEVER appear as a
# concrete route stop — only a grounded restaurant NAME may.
_RAW_MEAL_LABEL_RE = re.compile(r"[+＋](中等|中档|低档|高档|经济|实惠|medium|low|high)$|^(地方特色|当地美食|特色美食|家庭友好)")


def _itinerary_day_stops(day_item: Dict[str, Any]) -> List[str]:
    """Concrete, de-duplicated route stops for one day.  A meal (lunch/dinner) becomes
    a route stop ONLY when grounded to a real restaurant name — never a raw '菜系+价位'
    label such as '地方特色+中等'; arrival/departure markers are skipped."""
    stops: List[str] = []
    for block in ["morning", "lunch", "afternoon", "dinner"]:
        value = day_item.get(block) if isinstance(day_item, dict) else None
        is_meal = block in ("lunch", "dinner")
        if isinstance(value, dict):
            if is_meal:
                # Only a grounded real restaurant name; never the raw meal label.
                text = safe_text(value.get("restaurant_name"))
                if is_generic_place_label(text):
                    text = ""
            else:
                text = (
                    safe_text(value.get("poi_name"))
                    or safe_text(value.get("activity"))
                    or safe_text(value.get("name"))
                    or safe_text(value.get("area"))
                )
            if text:
                stops.append(text)
        elif isinstance(value, list):
            for x in value[:2]:
                if isinstance(x, dict) and (
                    x.get("is_transit") or x.get("poi_type") in ("arrival", "departure")
                ):
                    continue  # arrival/departure are not route stops
                text = (
                    (
                        safe_text(x.get("poi_name"))
                        or safe_text(x.get("activity"))
                        or safe_text(x.get("name"))
                    )
                    if isinstance(x, dict)
                    else safe_text(x)
                )
                if text:
                    stops.append(text)
        elif safe_text(value) and not is_meal:
            # A bare string in a meal slot is a raw '菜系+价位' label, not a real place.
            stops.append(safe_text(value))
    clean: List[str] = []
    for s in stops:
        if is_generic_transport_text(s) or _RAW_MEAL_LABEL_RE.search(s) or s in clean:
            continue
        clean.append(s)
    return clean


_BASE_PLACEHOLDER_TOKENS = (
    "未核验",
    "未提供",
    "未注明",
    "未指定",
    "待定",
    "暂无",
    "不详",
    "未知",
    "unknown",
    "n/a",
    "na",
    "none",
    "tbd",
)


def _is_base_placeholder(text: Any) -> bool:
    t = safe_text(text).strip().lower()
    return (not t) or any(p in t for p in _BASE_PLACEHOLDER_TOKENS)


def _hotel_base_station(plan: Dict[str, Any]) -> str:
    """The route's home base is the HOTEL — use the real hotel name, else the literal
    '酒店'.  NEVER a bare district/area ('渝中区') or a placeholder ('未核验'): the route
    starts and ends at the lodging, and the specific area is shown in the 酒店安排
    section rather than repeated on every route line."""
    hotel = plan.get("hotel_plan", {}) if isinstance(plan.get("hotel_plan"), dict) else {}
    name = safe_text(hotel.get("hotel_name"))
    if name and not _is_base_placeholder(name) and not is_generic_hotel_text(name):
        return name
    return "酒店"


def _build_day_route(plan: Dict[str, Any], day_num: int):
    """Build a concrete daily route (酒店->景点->餐厅->景点->酒店) from the itinerary.

    Falls back to the day-area placeholder only when the day genuinely has no
    concrete stop.  Returns (route_text, destination_text).
    """
    base = _hotel_base_station(plan)
    per_day = (
        plan.get("per_day_itinerary", []) if isinstance(plan.get("per_day_itinerary"), list) else []
    )
    day_item = next(
        (
            d
            for i, d in enumerate(per_day)
            if isinstance(d, dict) and safe_int(d.get("day"), i + 1) == day_num
        ),
        None,
    )
    stops = _itinerary_day_stops(day_item) if day_item else []
    if stops:
        return f"{base}->" + "->".join(stops[:4]) + f"->{base}", stops[-1]
    return f"{base}->第{day_num}天主要活动区域->{base}", f"第{day_num}天主要活动区域"


def _fallback_transport_grounding(
    plan: Dict[str, Any],
    requirements: Dict[str, Any],
    search_results: List[Dict[str, Any]],
    issue: str = "",
) -> Dict[str, Any]:
    """Build a safe transportation fallback from the already generated itinerary.

    It keeps Module 5 runnable when transport grounding LLM/search extraction is
    unavailable.  Exact costs remain conservative estimates and are marked as
    soft grounding issues for Module 6.
    """
    transport_req = (
        requirements.get("transportation_requirements", {})
        if isinstance(requirements.get("transportation_requirements"), dict)
        else {}
    )
    hotel = plan.get("hotel_plan", {}) if isinstance(plan.get("hotel_plan"), dict) else {}
    hotel_name = (
        safe_text(hotel.get("hotel_name"))
        or safe_text(transport_req.get("selected_hotel_name"))
        or "酒店"
    )
    base_station = (
        safe_text(hotel.get("nearby_metro_station"))
        or safe_text(transport_req.get("hotel_base_station"))
        or safe_text(hotel.get("hotel_area"))
        or safe_text(hotel.get("area"))
        or hotel_name
    )
    per_day = (
        plan.get("per_day_itinerary", [])
        if isinstance(plan.get("per_day_itinerary", []), list)
        else []
    )
    daily_items: List[Dict[str, Any]] = []
    for idx, day_item in enumerate(per_day):
        if not isinstance(day_item, dict):
            continue
        day_num = safe_int(day_item.get("day"), idx + 1)
        clean_stops = _itinerary_day_stops(day_item)
        route = (
            f"{base_station}->" + "->".join(clean_stops[:4]) + f"->{base_station}"
            if clean_stops
            else f"{base_station}->当天主要活动区域->{base_station}"
        )
        daily_items.append(
            {
                "day": day_num,
                "origin": base_station,
                "destination": clean_stops[-1] if clean_stops else "当天主要活动区域",
                "route": route,
                "mode": safe_text(plan.get("transportation_plan", {}).get("main_transport_mode"))
                if isinstance(plan.get("transportation_plan"), dict)
                else "地铁或短途出租车",
                "estimated_duration_minutes": 0,
                "estimated_cost_total": 50.0,
                "currency": "CNY",
                "cost_source": "rule_estimate_transport_fallback",
                "source_urls": [],
                "comfort_note": "未取得可靠逐日交通费用；使用保守估算，Module 6 按 grounding 质量软扣分。",
            }
        )
    source_urls: List[str] = []
    for item in search_results[:5]:
        if isinstance(item, dict) and safe_text(item.get("url")):
            source_urls.append(safe_text(item.get("url")))
    plan_obj = {
        "main_transport_mode": safe_text(
            plan.get("transportation_plan", {}).get("main_transport_mode")
        )
        if isinstance(plan.get("transportation_plan"), dict)
        else "地铁或短途出租车",
        "hotel_base_station": base_station,
        "metro_lines_near_hotel": hotel.get("metro_lines_near_hotel", [])
        if isinstance(hotel.get("metro_lines_near_hotel", []), list)
        else [],
        "intercity_transport": [],
        "daily_local_transport": daily_items,
        "local_transport_total_cost": round(
            sum(_money(x.get("estimated_cost_total"), 0) for x in daily_items), 2
        ),
        "source_urls": list(dict.fromkeys(source_urls)),
    }
    issues = []
    if issue:
        issues.append(issue)
    if daily_items:
        issues.append("transport_cost_area_fallback_used")
    return {
        "transportation_plan": plan_obj,
        "transport_grounding_issues": list(dict.fromkeys(issues)),
    }


def extract_transport_grounding_with_llm(
    plan: Dict[str, Any], requirements: Dict[str, Any], search_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Extract post-generation transport route/cost grounding from snippets.

    v59/v15 fix: v14 called this function but did not define it.  Missing or
    malformed transport extraction is now downgraded to a day-level fallback so
    Module 5 can finish and Module 6 can score the missing detail.
    """
    transport_req = (
        requirements.get("transportation_requirements", {})
        if isinstance(requirements.get("transportation_requirements"), dict)
        else {}
    )
    plan_text = compact_json_for_prompt(plan, max_chars=4300)
    transportation_requirements_text = compact_json_for_prompt(
        {
            "destination_city": requirements.get("destination_city"),
            "days": requirements.get("days"),
            "transportation_requirements": transport_req,
        },
        max_chars=2600,
    )
    search_results_text = compact_search_results(
        search_results, max_chars=4200, max_items=8, content_chars=260
    )
    prompt = build_transport_grounding_prompt(
        plan_text=plan_text,
        transportation_requirements_text=transportation_requirements_text,
        search_results_text=search_results_text,
    )
    try:
        response = call_generation_json(
            messages=[{"role": "user", "content": prompt}],
            model=ONLINE_GROUNDING_MODEL_NAME,
            max_tokens=min(ONLINE_GROUNDING_MAX_TOKENS, 2800),
            temperature=0.0,
        )
        data = response.get("data", {}) if isinstance(response, dict) else {}
        if not isinstance(data, dict):
            return _fallback_transport_grounding(
                plan, requirements, search_results, "transport_grounding_llm_returned_non_dict"
            )
        t_plan = (
            data.get("transportation_plan", {})
            if isinstance(data.get("transportation_plan"), dict)
            else {}
        )
        if not t_plan:
            return _fallback_transport_grounding(
                plan,
                requirements,
                search_results,
                "transport_grounding_llm_returned_no_transportation_plan",
            )
        daily = (
            t_plan.get("daily_local_transport", [])
            if isinstance(t_plan.get("daily_local_transport", []), list)
            else []
        )
        normalized_daily: List[Dict[str, Any]] = []
        for idx, item in enumerate(daily):
            if not isinstance(item, dict):
                continue
            day = safe_int(item.get("day"), idx + 1)
            route = safe_text(item.get("route"))
            cost = _money(item.get("estimated_cost_total") or item.get("cost_total"), 0)
            urls = (
                item.get("source_urls", []) if isinstance(item.get("source_urls", []), list) else []
            )
            normalized_daily.append(
                {
                    **item,
                    "day": day,
                    "route": route,
                    "estimated_cost_total": round(cost, 2),
                    "currency": safe_text(item.get("currency")) or "CNY",
                    "cost_source": safe_text(item.get("cost_source"))
                    or ("online_or_llm" if urls else "llm_or_estimate"),
                    "source_urls": [safe_text(x) for x in urls if safe_text(x)],
                }
            )
        if normalized_daily:
            t_plan["daily_local_transport"] = normalized_daily
            t_plan["local_transport_total_cost"] = round(
                sum(_money(x.get("estimated_cost_total"), 0) for x in normalized_daily), 2
            )
        issues = (
            data.get("transport_grounding_issues", [])
            if isinstance(data.get("transport_grounding_issues", []), list)
            else []
        )
        if not normalized_daily:
            fallback = _fallback_transport_grounding(
                plan, requirements, search_results, "transport_grounding_llm_missing_daily_routes"
            )
            fallback["transport_grounding_issues"] = list(
                dict.fromkeys(fallback.get("transport_grounding_issues", []) + issues)
            )
            return fallback
        return {"transportation_plan": t_plan, "transport_grounding_issues": issues}
    except Exception as exc:
        return _fallback_transport_grounding(
            plan, requirements, search_results, f"transport_grounding_llm_failed: {safe_text(exc)}"
        )


def persist_structured_grounding_cache(
    plan: Dict[str, Any], grounding: Dict[str, Any], family_agents: List[Dict[str, Any]]
) -> None:
    """Persist reusable online-derived facts to data/cache.

    The cache is intentionally structured and practical.  Each record contains
    clear parameters such as destination, location/area, cost, route, source URLs,
    and notes so future plans to the same city can reuse these references.
    """
    if not ONLINE_GROUNDING_CACHE_ENABLED:
        return
    destination = safe_text(plan.get("destination_city")) or safe_text(
        (family_agents[0] if family_agents else {}).get("destination_city")
    )
    departure = safe_text(plan.get("departure_city")) or safe_text(
        (family_agents[0] if family_agents else {}).get("departure_city")
    )
    currency = (
        safe_text((plan.get("cost_breakdown") or {}).get("currency"))
        if isinstance(plan.get("cost_breakdown"), dict)
        else "CNY"
    )
    currency = currency or "CNY"

    hotel = plan.get("hotel_plan", {}) if isinstance(plan.get("hotel_plan"), dict) else {}
    hotel_name = safe_text(hotel.get("hotel_name"))
    if hotel_name and not is_generic_hotel_text(hotel_name):
        _upsert_structured_cache(
            "hotels",
            {
                "destination_city": destination,
                "hotel_name": hotel_name,
                "area": safe_text(hotel.get("hotel_area")) or safe_text(hotel.get("area")),
                "address_or_location": safe_text(hotel.get("address"))
                or safe_text(hotel.get("location")),
                "price_level": safe_text(hotel.get("price_level")) or "medium",
                "estimated_price_per_night": _money(
                    hotel.get("estimated_price_per_night") or hotel.get("price_per_night"), 0
                ),
                "currency": currency,
                "nearby_metro_station": safe_text(hotel.get("nearby_metro_station")),
                "available_lines": hotel.get("available_lines", [])
                if isinstance(hotel.get("available_lines", []), list)
                else [],
                "family_suitability": safe_text(hotel.get("family_suitability"))
                or safe_text(hotel.get("reason")),
                "source_urls": hotel.get("source_urls", [])
                if isinstance(hotel.get("source_urls", []), list)
                else [],
                "verify_note": "价格和房态需预订前再次核验。",
            },
        )

    for item in (
        grounding.get("restaurants", [])
        if isinstance(grounding.get("restaurants", []), list)
        else []
    ):
        if not isinstance(item, dict):
            continue
        name = safe_text(item.get("restaurant_name"))
        rtype = safe_text(item.get("restaurant_type")) or safe_text(item.get("name_or_type"))
        area = safe_text(item.get("area")) or safe_text(item.get("restaurant_area")) or destination
        price_level = (
            safe_text(item.get("price_level"))
            or safe_text(plan.get("restaurant_price_level"))
            or "medium"
        )
        price = clamp_restaurant_price(
            item.get("estimated_price_per_person") or item.get("price_per_person"),
            price_level,
            name=name,
        )
        if not name and not rtype:
            continue
        _upsert_structured_cache(
            "restaurants",
            {
                "destination_city": destination,
                "restaurant_name": name,
                "restaurant_type": rtype
                or build_generic_restaurant_description(destination, area, price_level),
                "area": area,
                "meal_hint": safe_text(item.get("meal")),
                "price_level": price_level,
                "price_range_per_person": restaurant_price_range_text(price_level),
                "estimated_price_per_person": price,
                "currency": currency,
                "family_suitability": safe_text(item.get("comfort_note")) or "适合家庭；尽量避免重复餐厅。",
                "source_urls": item.get("source_urls", [])
                if isinstance(item.get("source_urls", []), list)
                else [],
                "verify_note": "餐厅营业时间、排队情况和人均价格需出行前再次核验。",
            },
        )

    for item in (
        grounding.get("ticket_plan", [])
        if isinstance(grounding.get("ticket_plan", []), list)
        else []
    ):
        if not isinstance(item, dict):
            continue
        poi = safe_text(item.get("poi_name")) or safe_text(item.get("attraction"))
        if not poi:
            continue
        price = _money(
            item.get("ticket_price_per_person")
            or item.get("estimated_ticket_price_per_person")
            or item.get("ticket_price"),
            0,
        )
        _upsert_structured_cache(
            "attractions",
            {
                "destination_city": destination,
                "poi_name": poi,
                "location_or_area": safe_text(item.get("location")) or safe_text(item.get("area")),
                "ticket_price_per_person": price,
                "currency": currency,
                "opening_hours": safe_text(item.get("opening_hours")),
                "participant_count_default": safe_int(
                    item.get("participant_count"), max(len(family_agents), 1)
                ),
                "estimated_total_cost_default": _money(
                    item.get("estimated_total_cost"), price * max(len(family_agents), 1)
                ),
                "price_source": safe_text(item.get("price_source")) or "online_or_rule_estimate",
                "source_urls": item.get("source_urls", [])
                if isinstance(item.get("source_urls", []), list)
                else [],
                "verify_note": "门票、开放时间和优惠政策需出行前再次核验。",
            },
        )

    transport = (
        plan.get("transportation_plan", {})
        if isinstance(plan.get("transportation_plan"), dict)
        else {}
    )
    for item in (
        transport.get("daily_local_transport", [])
        if isinstance(transport.get("daily_local_transport", []), list)
        else []
    ):
        if not isinstance(item, dict):
            continue
        route = safe_text(item.get("route"))
        if not route:
            continue
        _upsert_structured_cache(
            "transport_routes",
            {
                "destination_city": destination,
                "origin_city": departure,
                "origin": safe_text(item.get("origin")) or "酒店区域",
                "destination": safe_text(item.get("destination"))
                or safe_text(item.get("destination_name")),
                "route": route,
                "mode": safe_text(item.get("mode")) or "metro/walk/taxi",
                "estimated_duration_minutes": safe_float(item.get("estimated_duration_minutes"), 0),
                "estimated_cost_total": _money(item.get("estimated_cost_total"), 0),
                "currency": currency,
                "day_hint": safe_int(item.get("day"), 0),
                "comfort_note": safe_text(item.get("comfort_note")) or "优先少换乘、少步行。",
                "source_urls": item.get("source_urls", [])
                if isinstance(item.get("source_urls", []), list)
                else [],
                "verify_note": "交通票价和线路需出行前再次核验。",
            },
        )
    for item in (
        transport.get("intercity_transport", [])
        if isinstance(transport.get("intercity_transport", []), list)
        else []
    ):
        if not isinstance(item, dict):
            continue
        mode = safe_text(item.get("mode")) or safe_text(item.get("transport_mode"))
        _upsert_structured_cache(
            "transport_routes",
            {
                "destination_city": destination,
                "origin_city": departure,
                "origin": safe_text(item.get("origin")) or departure,
                "destination": safe_text(item.get("destination")) or destination,
                "route": safe_text(item.get("route")) or f"{departure} 到 {destination} {mode}",
                "mode": mode,
                "estimated_duration_minutes": safe_float(item.get("estimated_duration_minutes"), 0),
                "estimated_cost_per_person": _money(
                    item.get("estimated_price_per_person") or item.get("price_per_person"), 0
                ),
                "currency": currency,
                "source_urls": item.get("source_urls", [])
                if isinstance(item.get("source_urls", []), list)
                else [],
                "verify_note": "城际交通班次和票价需购票前再次核验。",
            },
        )


def _is_low_tier_hotel_name(name: str) -> bool:
    name = safe_text(name)
    if not name:
        return False
    return any(b in name for b in STRUCTURED_CACHE_HOTEL_NAME_BLACKLIST) and not any(
        br in name for br in STRUCTURED_CACHE_HOTEL_BRAND_WHITELIST
    )


def _cheapest_city_hotel(destination: str) -> Dict[str, Any]:
    """Cheapest priced hotel for the city from the DB, INCLUDING low-tier ones.

    Used only as a budget-relief fallback; the caller records a penalty when the
    chosen hotel is a low-tier (旅馆/招待所/民宿…) name.
    """
    best: Dict[str, Any] = {}
    best_price = None
    for r in _load_category_cache("hotels"):
        if not isinstance(r, dict) or not _cache_record_matches_destination(r, destination):
            continue
        if not safe_text(r.get("hotel_name")):
            continue
        price = _money(r.get("estimated_price_per_night"), 0)
        if price <= 0:
            continue
        if best_price is None or price < best_price:
            best_price, best = price, r
    return best


def _best_city_hotel_under(destination: str, max_nightly: float) -> Dict[str, Any]:
    """Priciest real (brand / non-low-tier) seed hotel with nightly <= max_nightly.

    Used by the upward budget fill to upgrade an under-budget plan to a nicer real
    hotel without crossing the headroom cap.
    """
    best: Dict[str, Any] = {}
    best_price = -1.0
    for r in _load_category_cache("hotels"):
        if not isinstance(r, dict) or not _cache_record_matches_destination(r, destination):
            continue
        name = safe_text(r.get("hotel_name"))
        if not name or _is_low_tier_hotel_name(name):
            continue
        price = _money(r.get("estimated_price_per_night"), 0)
        if price <= 0 or price > max_nightly:
            continue
        if price > best_price:
            best_price, best = price, r
    return best


def _intercity_roundtrip_cost(
    plan: Dict[str, Any], grounding: Dict[str, Any], member_count: int
) -> float:
    """Round-trip intercity (departure<->destination) ticket cost for the whole family.

    Uses the grounded intercity legs when present (doubling a one-way-only set for
    the return trip); otherwise estimates from a seeded same-route fare, and finally
    from a per-person default.  Always covers BOTH directions and every member.
    """
    transport = (
        grounding.get("transportation_plan", {})
        if isinstance(grounding.get("transportation_plan"), dict)
        else {}
    )
    legs = (
        transport.get("intercity_transport", [])
        if isinstance(transport.get("intercity_transport", []), list)
        else []
    )
    fares = []
    for leg in legs:
        if not isinstance(leg, dict):
            continue
        pp = _money(leg.get("estimated_price_per_person") or leg.get("price_per_person"), 0)
        if pp <= 0:
            tot = _money(leg.get("estimated_cost_total"), 0)
            pp = tot / max(member_count, 1) if tot > 0 else 0
        if pp > 0:
            fares.append(pp)
    if fares:
        # The legs are alternative one-way options (高铁/飞机), not two directions;
        # budget a round trip by the cheapest mode for every family member.
        return round(min(fares) * 2 * max(member_count, 1), 2)
    # No grounded legs: estimate a one-way fare from a seeded same-route record.
    departure = safe_text(plan.get("departure_city"))
    destination = safe_text(plan.get("destination_city"))
    one_way = 0.0
    for r in _load_category_cache("transport_routes"):
        if not isinstance(r, dict) or not _cache_record_matches_destination(r, destination):
            continue
        oc = safe_text(r.get("origin_city")).lower()
        if departure and oc and not (departure.lower() in oc or oc in departure.lower()):
            continue
        pp = _money(r.get("estimated_cost_per_person"), 0)
        if pp > one_way:
            one_way = pp
    if one_way <= 0:
        one_way = INTERCITY_ROUNDTRIP_DEFAULT_PER_PERSON / 2.0
    return round(one_way * 2 * max(member_count, 1), 2)


def normalize_cost_breakdown(
    grounding: Dict[str, Any], plan: Dict[str, Any], family_agents: List[Dict[str, Any]]
) -> Dict[str, Any]:
    breakdown = (
        grounding.get("cost_breakdown", {})
        if isinstance(grounding.get("cost_breakdown"), dict)
        else {}
    )
    member_count = max(len(family_agents), 1)
    day = max(safe_int(plan.get("day"), 1), 1)
    nights = max(day - 1, 1)
    # Rooms by family composition: parents+kids share, each teenager/青年 own room,
    # grandparents share one room.
    rooms = estimate_room_count(family_agents, people_per_room=HOTEL_PEOPLE_PER_ROOM)
    destination = safe_text(plan.get("destination_city"))
    hotel_plan = (
        grounding.get("hotel_plan", {}) if isinstance(grounding.get("hotel_plan"), dict) else {}
    )
    price_per_night = _money(hotel_plan.get("estimated_price_per_night"), 0) or _money(
        hotel_plan.get("price_per_night"), 0
    )
    # The final chosen hotel + its nightly price live in plan['hotel_plan'];
    # grounding['hotel_plan'] may lack the price, which previously left hotel_cost
    # off the total entirely.  Fall back to the plan's hotel so the budget matches
    # the displayed nightly price.
    plan_hotel = plan.get("hotel_plan", {}) if isinstance(plan.get("hotel_plan"), dict) else {}
    if price_per_night <= 0:
        price_per_night = _money(plan_hotel.get("estimated_price_per_night"), 0) or _money(
            plan_hotel.get("price_per_night"), 0
        )
    # Many DB hotels have no recorded nightly price.  Never drop the hotel from the
    # budget: fall back to a price-level estimate so hotel cost is always counted.
    if price_per_night <= 0:
        level = (
            safe_text(hotel_plan.get("price_level")).lower()
            or safe_text(plan_hotel.get("price_level")).lower()
        )
        price_per_night = _HOTEL_PRICE_BY_LEVEL.get(level, _HOTEL_PRICE_BY_LEVEL["medium"])
        hotel_plan["estimated_price_per_night"] = price_per_night  # keep TXT display consistent
        hotel_plan["hotel_price_estimated"] = True
        grounding["hotel_plan"] = hotel_plan
    # Hotel cost = nightly price * nights * rooms (rooms scale with family size).
    hotel_cost = round(price_per_night * nights * rooms, 2)
    # The five real components (hotel, intercity, local, meals, tickets) are
    # comprehensive; the LLM's free-form other_cost is unreliable and has inflated
    # totals (e.g. a spurious +3000), so it is not trusted.
    other_cost = 0.0

    per_day = (
        plan.get("per_day_itinerary", []) if isinstance(plan.get("per_day_itinerary"), list) else []
    )

    # Local daily transport: ALWAYS the sum of the real per-day route costs that are
    # actually shown in the itinerary (not a generic estimate).
    transport = (
        grounding.get("transportation_plan", {})
        if isinstance(grounding.get("transportation_plan"), dict)
        else {}
    )
    daily = (
        transport.get("daily_local_transport", [])
        if isinstance(transport.get("daily_local_transport", []), list)
        else []
    )
    local_cost = sum(
        (_money(item.get("estimated_cost_total"), 0) for item in daily if isinstance(item, dict))
    )
    if local_cost <= 0:
        local_cost = _money(breakdown.get("local_transport_cost"), 0)
    # Intercity = round-trip (departure<->destination) for every family member.
    intercity_cost = _intercity_roundtrip_cost(plan, grounding, member_count)
    # Meals: ALWAYS sum the real per-person restaurant prices over every meal * members.
    restaurants = (
        grounding.get("restaurants", [])
        if isinstance(grounding.get("restaurants", []), list)
        else []
    )
    meal_cost = sum(
        (
            (
                _money(item.get("estimated_price_per_person"), 0)
                or _money(item.get("price_per_person"), 0)
            )
            * member_count
            for item in restaurants
            if isinstance(item, dict)
        )
    )
    if meal_cost <= 0:
        meal_cost = _money(breakdown.get("meal_cost"), 0)
    # Tickets: ALWAYS sum the real ticket price of each DISTINCT attraction actually
    # in the itinerary (the prices shown as 门票约 …) * participants.  Free POIs add 0,
    # and a POI repeated across days is paid for once.
    ticket_cost = 0.0
    seen_poi = set()
    for day_item in per_day:
        if not isinstance(day_item, dict):
            continue
        for block in ("morning", "afternoon"):
            acts = day_item.get(block) if isinstance(day_item.get(block), list) else []
            for act in acts:
                if not isinstance(act, dict):
                    continue
                poi_key = _norm_name(act.get("poi_name") or act.get("activity"))
                if not poi_key or poi_key in seen_poi:
                    continue
                seen_poi.add(poi_key)
                price = (
                    _money(act.get("ticket_price"), 0)
                    or _money(act.get("ticket_price_per_person"), 0)
                    or _money(act.get("estimated_ticket_price_per_person"), 0)
                )
                participants = (
                    safe_int(act.get("ticket_participant_count"), member_count) or member_count
                )
                ticket_cost += price * participants
    if ticket_cost <= 0:
        # Fallback: sum the ticket plan (deduped) when the itinerary has no attached prices.
        tickets = (
            grounding.get("ticket_plan", [])
            if isinstance(grounding.get("ticket_plan", []), list)
            else []
        )
        seen_tp = set()
        for item in tickets:
            if not isinstance(item, dict):
                continue
            k = _norm_name(item.get("poi_name"))
            if k and k in seen_tp:
                continue
            if k:
                seen_tp.add(k)
            participants = safe_int(item.get("participant_count"), member_count) or member_count
            ticket_cost += (
                _money(item.get("estimated_ticket_price_per_person"), 0)
                or _money(item.get("ticket_price_per_person"), 0)
            ) * participants

    # Honest total: always the real sum of the components (they add up exactly).
    total = hotel_cost + intercity_cost + local_cost + meal_cost + ticket_cost + other_cost
    cost_note = (
        safe_text(breakdown.get("cost_note"))
        or "总费用包含酒店、往返城际交通、市内交通、餐饮和景点门票（按家庭人数计）；价格为在线查询或保守估算，预订前需核验。"
    )

    budget_limit = safe_float(plan.get("budget_limit"), 0)
    meals_reduced = False
    # Step 0 (替换餐厅): if over budget, switch to economical dining — lower the
    # per-person meal spend toward a floor so the meal budget fits.  The reduced
    # per-person price is written back to each restaurant so the displayed plan matches.
    if budget_limit > 0 and BUDGET_MEAL_RELIEF and meal_cost > 0:
        floor_hotel = HOTEL_MIN_NIGHTLY * nights * rooms
        room_for_meals = budget_limit * BUDGET_TARGET_FRACTION - (
            intercity_cost + local_cost + ticket_cost + other_cost + floor_hotel
        )
        if 0 < room_for_meals < meal_cost:
            restaurants = (
                grounding.get("restaurants", [])
                if isinstance(grounding.get("restaurants"), list)
                else []
            )
            n_meals = sum(1 for it in restaurants if isinstance(it, dict))
            if n_meals > 0:
                target_pp = max(ECONOMICAL_MEAL_PER_PERSON, room_for_meals / member_count / n_meals)
                new_meal = 0.0
                for it in restaurants:
                    if not isinstance(it, dict):
                        continue
                    old_pp = _money(it.get("estimated_price_per_person"), 0) or _money(
                        it.get("price_per_person"), 0
                    )
                    new_pp = round(min(old_pp, target_pp), 2) if old_pp > 0 else round(target_pp, 2)
                    it["estimated_price_per_person"] = new_pp
                    it["estimated_total_cost"] = round(new_pp * member_count, 2)
                    if new_pp < old_pp:
                        it["comfort_note"] = "预算偏紧，已选经济餐饮/套餐，价格需核验。"
                    new_meal += new_pp * member_count
                if new_meal < meal_cost:
                    meals_reduced = True
                    meal_cost = round(new_meal, 2)
                    total = (
                        hotel_cost
                        + intercity_cost
                        + local_cost
                        + meal_cost
                        + ticket_cost
                        + other_cost
                    )
    if meals_reduced:
        cost_note += "；预算偏紧，餐饮已按经济档估算以适配预算。"

    attractions_swapped = False
    # Step 1 (替换收费景点): if even a floor-priced hotel cannot fit, swap the most
    # expensive non-must-include paid attractions for free real ones, keeping members'
    # must-include attractions.  Recompute the real ticket cost from the new itinerary.
    if budget_limit > 0 and ATTRACTION_BUDGET_RELIEF:
        floor_hotel = HOTEL_MIN_NIGHTLY * nights * rooms
        room_for_tickets = budget_limit * BUDGET_TARGET_FRACTION - (
            intercity_cost + local_cost + meal_cost + other_cost + floor_hotel
        )
        target_drop = ticket_cost - max(0.0, room_for_tickets)
        if target_drop > 0:
            dropped = _swap_paid_attractions_for_budget(
                plan, family_agents, member_count, target_drop
            )
            if dropped > 0:
                attractions_swapped = True
                ticket_cost = max(0.0, ticket_cost - dropped)
                total = (
                    hotel_cost + intercity_cost + local_cost + meal_cost + ticket_cost + other_cost
                )
    if attractions_swapped:
        cost_note += "；预算偏紧，已将部分收费景点替换为同城免费景点以压低门票（保留成员必去项）。"

    # Budget relief: if over budget, size the hotel to the remaining budget.
    # Fixed real costs (tickets, round-trip transport) are NEVER scaled.
    low_tier_penalty = False
    budget_estimated_hotel = False
    if budget_limit > 0 and total > budget_limit and BUDGET_ALLOW_LOW_TIER_HOTEL_RELIEF:
        cheap = _cheapest_city_hotel(destination) if destination else {}
        cheap_price = _money(cheap.get("estimated_price_per_night"), 0)
        # Let the hotel absorb whatever budget is left after the other real costs
        # (meals + tickets + round-trip + local), so the plan fits the budget when the
        # non-hotel costs leave room.  Bounded below by a sane nightly floor; if even
        # the floor doesn't fit, the overage is reported honestly.
        budget_for_hotel = max(
            0.0,
            budget_limit * BUDGET_TARGET_FRACTION
            - (intercity_cost + local_cost + meal_cost + ticket_cost + other_cost),
        )
        affordable_room_night = (budget_for_hotel / nights / max(1, rooms)) if nights > 0 else 0.0
        budget_room_night = round(max(HOTEL_MIN_NIGHTLY, affordable_room_night), 2)
        if cheap_price > 0 and (budget_room_night <= 0 or cheap_price <= budget_room_night):
            # An affordable real hotel exists — use it.
            target_price, target_name = cheap_price, safe_text(cheap.get("hotel_name"))
            target_area = safe_text(cheap.get("area")) or safe_text(hotel_plan.get("area"))
            target_station = safe_text(cheap.get("nearby_metro_station")) or safe_text(
                hotel_plan.get("nearby_metro_station")
            )
            target_sources = (
                cheap.get("source_urls", []) if isinstance(cheap.get("source_urls"), list) else []
            )
            low_tier_penalty = _is_low_tier_hotel_name(cheap.get("hotel_name"))
        elif budget_room_night > 0:
            # No affordable real hotel — cap to a clearly-labelled budget estimate
            # rather than charging an unaffordable hotel several times the budget.
            target_price = budget_room_night
            target_name = "经济型酒店（按预算估算，需自行确认具体酒店）"
            target_area = safe_text(hotel_plan.get("area")) or destination
            target_station = safe_text(hotel_plan.get("nearby_metro_station"))
            target_sources = []
            budget_estimated_hotel = True
        else:
            target_price = 0
        if target_price > 0:
            new_hotel_cost = round(target_price * nights * rooms, 2)
            if new_hotel_cost < hotel_cost or hotel_cost <= 0:
                relief_fields = {
                    "hotel_name": target_name,
                    "estimated_price_per_night": target_price,
                    "area": target_area,
                    "nearby_metro_station": target_station,
                    "source_urls": target_sources,
                    "price_level": "budget",
                    "budget_relief_low_tier": low_tier_penalty,
                    "hotel_price_estimated": budget_estimated_hotel,
                }
                hotel_plan = {**dict(hotel_plan), **relief_fields}
                grounding["hotel_plan"] = hotel_plan
                # Sync the DISPLAYED hotel too, otherwise a later recompute from
                # plan['hotel_plan'] would revert to the expensive hotel and undo the relief.
                plan["hotel_plan"] = {
                    **(
                        plan.get("hotel_plan", {})
                        if isinstance(plan.get("hotel_plan"), dict)
                        else {}
                    ),
                    **relief_fields,
                }
                hotel_cost = new_hotel_cost
                total = (
                    hotel_cost + intercity_cost + local_cost + meal_cost + ticket_cost + other_cost
                )

    # Budget fill: when the plan leaves too much unused, upgrade toward the target
    # utilisation (~85%) so it spends most of the budget while keeping ~15% headroom.
    # Levers (most realistic first): a pricier real city hotel, then richer meals;
    # both are capped so the total never crosses budget * BUDGET_TARGET_FRACTION.
    budget_filled = False
    if (
        budget_limit > 0
        and BUDGET_FILL_ENABLED
        and not low_tier_penalty
        and not budget_estimated_hotel
        and total < budget_limit * BUDGET_FILL_MIN_UTILIZATION
    ):
        cap = budget_limit * BUDGET_TARGET_FRACTION
        # Lever 1 — upgrade the hotel to the priciest real city hotel that fits the cap.
        room_for_hotel = cap - (intercity_cost + local_cost + meal_cost + ticket_cost + other_cost)
        max_nightly = (room_for_hotel / nights / max(1, rooms)) if nights > 0 else 0.0
        if destination and max_nightly > price_per_night:
            better = _best_city_hotel_under(destination, max_nightly)
            better_price = _money(better.get("estimated_price_per_night"), 0)
            if better_price > price_per_night:
                up_fields = {
                    "hotel_name": safe_text(better.get("hotel_name")),
                    "estimated_price_per_night": better_price,
                    "area": safe_text(better.get("area")) or safe_text(hotel_plan.get("area")),
                    "nearby_metro_station": safe_text(better.get("nearby_metro_station"))
                    or safe_text(hotel_plan.get("nearby_metro_station")),
                    "source_urls": better.get("source_urls", [])
                    if isinstance(better.get("source_urls"), list)
                    else [],
                    "price_level": "comfort",
                }
                hotel_plan = {**dict(hotel_plan), **up_fields}
                grounding["hotel_plan"] = hotel_plan
                plan["hotel_plan"] = {
                    **(
                        plan.get("hotel_plan", {})
                        if isinstance(plan.get("hotel_plan"), dict)
                        else {}
                    ),
                    **up_fields,
                }
                price_per_night = better_price
                hotel_cost = round(better_price * nights * rooms, 2)
                total = (
                    hotel_cost + intercity_cost + local_cost + meal_cost + ticket_cost + other_cost
                )
                budget_filled = True
        # Lever 2 — if still under, scale meals up proportionally (keeps each meal's
        # relative price; per-meal capped at the 'high' tier so they stay realistic).
        if total < budget_limit * BUDGET_FILL_MIN_UTILIZATION and meal_cost > 0:
            room_for_meals = cap - (
                hotel_cost + intercity_cost + local_cost + ticket_cost + other_cost
            )
            if room_for_meals > meal_cost:
                hi_cap = PRICE_LEVEL_RANGES.get("high", (0, 300))[1]
                scale = min(room_for_meals / meal_cost, 2.0)
                restaurants = (
                    grounding.get("restaurants", [])
                    if isinstance(grounding.get("restaurants"), list)
                    else []
                )
                new_meal = 0.0
                for it in restaurants:
                    if not isinstance(it, dict):
                        continue
                    old_pp = _money(it.get("estimated_price_per_person"), 0) or _money(
                        it.get("price_per_person"), 0
                    )
                    new_pp = round(min(old_pp * scale, hi_cap), 2) if old_pp > 0 else 0.0
                    if new_pp > old_pp:
                        it["estimated_price_per_person"] = new_pp
                        it["estimated_total_cost"] = round(new_pp * member_count, 2)
                    new_meal += (new_pp if new_pp > old_pp else old_pp) * member_count
                if new_meal > meal_cost:
                    meal_cost = round(new_meal, 2)
                    total = (
                        hotel_cost
                        + intercity_cost
                        + local_cost
                        + meal_cost
                        + ticket_cost
                        + other_cost
                    )
                    budget_filled = True
    if budget_filled:
        cost_note += "；预算较充裕，已上调住宿/餐饮档次以更充分利用预算（仍保留约15%余量）。"

    if budget_estimated_hotel:
        cost_note += "；该城市数据库无符合预算的真实酒店，住宿费用已按预算上限估算（约定每晚每间预算），请自行确认具体经济住宿。"

    if low_tier_penalty:
        cost_note += "；预算不足，已改用更经济的低星/无品牌住宿（旅馆/招待所/民宿类）以压低费用，按住宿档次下降计入质量扣分，预订前需核验。"
        issues = (
            grounding.get("grounding_issues", [])
            if isinstance(grounding.get("grounding_issues"), list)
            else []
        )
        if "budget_relief_low_tier_hotel_used" not in issues:
            issues.append("budget_relief_low_tier_hotel_used")
        grounding["grounding_issues"] = issues
    if budget_limit > 0 and total > budget_limit:
        cost_note += f"；按真实组件估算总费用 {round(total, 2)} 元仍超出预算 {round(budget_limit, 2)} 元，已如实呈现（未缩放门票/往返交通等固定成本）。"

    return {
        "currency": safe_text(breakdown.get("currency")) or "CNY",
        "hotel_cost": round(hotel_cost, 2),
        "hotel_rooms": rooms,
        "hotel_nights": nights,
        "intercity_transport_cost": round(intercity_cost, 2),
        "local_transport_cost": round(local_cost, 2),
        "meal_cost": round(meal_cost, 2),
        "attraction_ticket_cost": round(ticket_cost, 2),
        "other_cost": round(other_cost, 2),
        "total_cost": round(total, 2),
        "budget_relief_low_tier_hotel": bool(low_tier_penalty),
        "member_count": member_count,
        "cost_note": cost_note,
    }


_DURATION_HM_RE = re.compile(
    r"(?:约|大约|历时|耗时)?\s*(\d+(?:\.\d+)?)\s*(?:个)?\s*(?:小时|小時|hours?|hrs?|h|时)\s*(?:(\d+)\s*(?:分钟|分鐘|分|min))?",
    re.I,
)
_DURATION_MIN_RE = re.compile(r"(\d+)\s*(?:分钟|分鐘|min)", re.I)
# Permissive fare matcher for intercity snippets ('机票680元起', '二等座553元', '¥780').
_INTERCITY_FARE_RE = re.compile(r"(?:¥|￥|rmb)?\s*(\d{2,5})\s*(?:元|块|rmb|¥|￥)", re.I)


def _extract_intercity_fares(text: Any) -> List[float]:
    out: List[float] = []
    for m in _INTERCITY_FARE_RE.finditer(safe_text(text)):
        try:
            out.append(float(m.group(1)))
        except (TypeError, ValueError):
            continue
    return out


def _parse_duration_minutes_from_text(text: Any) -> float:
    """Best-effort travel duration (minutes) from a Chinese/English snippet.

    Matches '约3小时15分', '3小时', '2h30m', '150分钟'.  Bounded to a sane intercity
    range (20 min .. 24 h) so a stray number does not become the duration.
    """
    t = safe_text(text)
    if not t:
        return 0.0
    m = _DURATION_HM_RE.search(t)
    if m:
        total = safe_float(m.group(1), 0) * 60.0 + (
            safe_float(m.group(2), 0) if m.group(2) else 0.0
        )
        if 20.0 <= total <= 24 * 60.0:
            return round(total, 0)
    m2 = _DURATION_MIN_RE.search(t)
    if m2:
        total = safe_float(m2.group(1), 0)
        if 20.0 <= total <= 24 * 60.0:
            return round(total, 0)
    return 0.0


# (mode label, search keywords, price_lo, price_hi per person CNY, default duration minutes)
_INTERCITY_MODE_SPECS = [
    ("高铁", "高铁 动车 二等座 票价 历时 时间", 50.0, 1600.0, 300.0),
    ("飞机", "飞机 机票 经济舱 票价 飞行时间", 200.0, 3000.0, 150.0),
]
_INTERCITY_MODE_HINTS = (
    "高铁",
    "动车",
    "火车",
    "列车",
    "飞机",
    "航班",
    "客机",
    "高速",
    "城际",
    "train",
    "flight",
    "rail",
    "air",
    "plane",
    "intercity",
    "highspeed",
)
_LOCAL_MODE_HINTS = (
    "地铁",
    "公交",
    "出租",
    "打车",
    "步行",
    "轮渡",
    "巴士",
    "metro",
    "bus",
    "taxi",
    "walk",
    "subway",
    "mtr",
    "ferry",
)


def _is_intercity_transport_mode(mode_text: Any) -> bool:
    """True only for genuine intercity modes (高铁/飞机...), not local transit (地铁/公交...).

    transport_routes records carry the TRIP's origin_city/destination_city even on
    local legs, so mode is the only reliable intercity signal.
    """
    m = safe_text(mode_text).lower()
    if not m or any(k in m for k in _LOCAL_MODE_HINTS):
        return False
    return any(k in m for k in _INTERCITY_MODE_HINTS)


def lookup_intercity_options(
    departure: Any, destination: Any, member_count: int
) -> List[Dict[str, Any]]:
    """Look up intercity (departure<->destination) flight/train options — real
    price-per-person and approximate duration.

    Seed-first (transport_routes.csv) -> online search (price + 大概时间) -> estimate.
    Returns one-way legs (the budget round-trips them); each leg carries mode, route,
    estimated_price_per_person, estimated_duration_minutes, currency, is_estimated, source.
    Online failures degrade to seed/estimate (never raises).
    """
    departure = safe_text(departure)
    destination = safe_text(destination)
    member_count = max(safe_int(member_count, 1), 1)
    if not departure or not destination:
        return []

    # 1) seed-first: a stored same-route fare/duration.
    seed_legs: List[Dict[str, Any]] = []
    seen_modes = set()
    for r in _load_category_cache("transport_routes"):
        if not isinstance(r, dict) or not _cache_record_matches_destination(r, destination):
            continue
        oc = safe_text(r.get("origin_city")).lower()
        if oc and not (departure.lower() in oc or oc in departure.lower()):
            continue
        mode = safe_text(r.get("mode"))
        if not _is_intercity_transport_mode(mode):  # skip local metro/bus legs
            continue
        pp = _money(r.get("estimated_cost_per_person"), 0)
        if pp <= 0:
            tot = _money(r.get("estimated_cost_total"), 0)
            pp = tot / member_count if tot > 0 else 0
        if pp <= 0:
            continue
        if mode in seen_modes:
            continue
        seen_modes.add(mode)
        seed_legs.append(
            {
                "mode": mode,
                "origin": departure,
                "destination": destination,
                "route": safe_text(r.get("route")) or f"{departure}→{destination}",
                "estimated_price_per_person": round(pp, 2),
                "estimated_duration_minutes": safe_float(r.get("estimated_duration_minutes"), 0),
                "currency": "CNY",
                "is_estimated": False,
                "source": "seed:transport_routes.csv",
            }
        )
        if len(seed_legs) >= 2:
            break
    if seed_legs:
        return seed_legs

    # 2) online: look up flight + train price and approximate time.
    online_legs: List[Dict[str, Any]] = []
    for mode_label, kw, lo, hi, default_minutes in _INTERCITY_MODE_SPECS:
        query = _compact_query_text(f"{departure} 到 {destination} {kw} 2026", 120)
        try:
            results = online_search(query, max_results=max(1, min(ONLINE_SEARCH_MAX_RESULTS, 2)))
        except Exception:
            results = []
        if not results:
            continue
        text = " ".join(
            safe_text(x.get("title")) + " " + safe_text(x.get("content"))
            for x in results
            if isinstance(x, dict)
        )
        prices = sorted(v for v in _extract_intercity_fares(text) if lo <= v <= hi)
        if not prices:
            continue
        # Median-ish to avoid package/business-class outliers.
        price = prices[len(prices) // 2]
        online_legs.append(
            {
                "mode": mode_label,
                "origin": departure,
                "destination": destination,
                "route": f"{departure}→{destination}",
                "estimated_price_per_person": round(price, 2),
                "estimated_duration_minutes": _parse_duration_minutes_from_text(text)
                or default_minutes,
                "currency": "CNY",
                "is_estimated": False,
                "source": "online_search",
            }
        )
    if online_legs:
        return online_legs

    # 3) estimate: a single train leg at the default one-way fare.
    return [
        {
            "mode": "高铁",
            "origin": departure,
            "destination": destination,
            "route": f"{departure}→{destination}",
            "estimated_price_per_person": round(INTERCITY_ROUNDTRIP_DEFAULT_PER_PERSON / 2.0, 2),
            "estimated_duration_minutes": 300.0,
            "currency": "CNY",
            "is_estimated": True,
            "source": "estimate",
        }
    ]


def normalize_transport_costs(
    grounding: Dict[str, Any], plan: Dict[str, Any], family_agents: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Normalize daily and intercity transportation costs.

    v59/v15 fix: v14 called normalize_transport_costs but never defined it.
    This function keeps transport entries structurally consistent and prevents
    absurd extracted values from distorting the budget.
    """
    transport = (
        grounding.get("transportation_plan", {})
        if isinstance(grounding.get("transportation_plan"), dict)
        else {}
    )
    if not isinstance(transport, dict):
        transport = {}
    member_count = max(len(family_agents), 1)
    days = max(safe_int(plan.get("day"), 1), 1)

    raw_daily = (
        transport.get("daily_local_transport", [])
        if isinstance(transport.get("daily_local_transport", []), list)
        else []
    )
    normalized_daily: List[Dict[str, Any]] = []
    seen_days = set()
    for idx, item in enumerate(raw_daily):
        if not isinstance(item, dict):
            continue
        day = safe_int(item.get("day"), idx + 1)
        if day <= 0:
            day = idx + 1
        route = safe_text(item.get("route")) or safe_text(item.get("route_summary"))
        mode = (
            safe_text(item.get("mode"))
            or safe_text(item.get("transport_mode"))
            or safe_text(transport.get("main_transport_mode"))
            or "地铁或短途出租车"
        )
        cost = _money(
            item.get("estimated_cost_total")
            or item.get("cost_total")
            or item.get("estimated_price_total"),
            0,
        )
        # Conservative family-level daily city transport estimate when missing.
        if cost <= 0:
            low_mode = mode.lower()
            if "taxi" in low_mode or "出租" in mode:
                cost = 60.0
            elif "metro" in low_mode or "地铁" in mode or "mtr" in low_mode:
                cost = max(20.0, 10.0 * member_count)
            else:
                cost = 50.0
        # Search snippets can return package/flight/hotel prices by mistake.
        # Local daily transport for a family should rarely exceed this cap.
        daily_cap = max(120.0, 80.0 * member_count)
        if cost > daily_cap:
            cost = daily_cap
        urls = item.get("source_urls", []) if isinstance(item.get("source_urls", []), list) else []
        # Always rebuild the route from THIS day's final itinerary stops, so the
        # transport never lists a POI the day does not actually visit (the LLM/grounded
        # route often includes extra attractions that were dropped from the itinerary).
        # The grounded mode/cost above are kept; only the route text is made consistent.
        route = _build_day_route(plan, day)[0]
        normalized = dict(item)
        normalized.update(
            {
                "day": day,
                "route": route,
                "mode": mode,
                "estimated_cost_total": round(cost, 2),
                "currency": safe_text(item.get("currency")) or "CNY",
                "cost_source": safe_text(item.get("cost_source"))
                or ("online_or_llm" if urls else "rule_estimate_transport_fallback"),
                "source_urls": [safe_text(x) for x in urls if safe_text(x)],
                "comfort_note": safe_text(item.get("comfort_note")) or "优先少换乘、少步行；费用需出行前核验。",
            }
        )
        normalized_daily.append(normalized)
        seen_days.add(day)

    # Fill missing days with day-level area fallback instead of leaving blanks.
    existing_transport = (
        plan.get("transportation_plan", {})
        if isinstance(plan.get("transportation_plan"), dict)
        else {}
    )
    main_mode = (
        safe_text(transport.get("main_transport_mode"))
        or safe_text(existing_transport.get("main_transport_mode"))
        or "地铁或短途出租车"
    )
    for day in range(1, days + 1):
        if day in seen_days:
            continue
        fallback_cost = 60.0 if "出租" in main_mode else max(20.0, 10.0 * member_count)
        normalized_daily.append(
            {
                "day": day,
                "route": _build_day_route(plan, day)[0],
                "mode": main_mode,
                "estimated_cost_total": round(fallback_cost, 2),
                "currency": "CNY",
                "cost_source": "rule_estimate_transport_fallback",
                "source_urls": [],
                "comfort_note": "未取得可靠逐日交通费用；使用保守估算，Module 6 按 grounding 质量软扣分。",
            }
        )
    normalized_daily = sorted(normalized_daily, key=lambda x: safe_int(x.get("day"), 0))
    transport["daily_local_transport"] = normalized_daily
    transport["local_transport_total_cost"] = round(
        sum(_money(x.get("estimated_cost_total"), 0) for x in normalized_daily), 2
    )

    raw_intercity = (
        transport.get("intercity_transport", [])
        if isinstance(transport.get("intercity_transport", []), list)
        else []
    )
    normalized_intercity: List[Dict[str, Any]] = []
    for item in raw_intercity:
        if not isinstance(item, dict):
            continue
        price = _money(item.get("estimated_price_per_person") or item.get("price_per_person"), 0)
        if price > 5000:
            price = 5000.0
        normalized = dict(item)
        normalized["estimated_price_per_person"] = round(price, 2)
        normalized["estimated_duration_minutes"] = safe_float(
            item.get("estimated_duration_minutes"), 0
        )
        normalized["currency"] = safe_text(item.get("currency")) or "CNY"
        normalized_intercity.append(normalized)
    # Add concrete flight/train legs (price + 大概时间) when the model gave none usable:
    # seed -> online search -> estimate.  Never raises (online failures degrade).
    if not any(_money(x.get("estimated_price_per_person"), 0) > 0 for x in normalized_intercity):
        looked_up = lookup_intercity_options(
            plan.get("departure_city"), plan.get("destination_city"), max(len(family_agents), 1)
        )
        if looked_up:
            normalized_intercity = looked_up
    # Backfill an approximate duration for any leg still missing one (flight≈2.5h, train≈5h).
    for leg in normalized_intercity:
        if safe_float(leg.get("estimated_duration_minutes"), 0) <= 0:
            leg["estimated_duration_minutes"] = (
                150.0
                if any(k in safe_text(leg.get("mode")) for k in ("飞", "机", "flight", "air"))
                else 300.0
            )
    transport["intercity_transport"] = normalized_intercity
    grounding["transportation_plan"] = transport
    return transport


def merge_online_grounding_into_plan(
    plan: Dict[str, Any], grounding: Dict[str, Any], family_agents: List[Dict[str, Any]]
) -> Dict[str, Any]:
    if not isinstance(grounding, dict):
        grounding = {}

    # v45: clean API/LLM extracted details before merging into the plan.
    # This keeps restaurants diverse, prices reasonable, local transport costs
    # realistic, and attraction tickets included in the budget.
    try:
        normalize_restaurants(grounding, plan, family_agents)
        normalize_transport_costs(grounding, plan, family_agents)
        normalize_ticket_plan(grounding, plan, family_agents)
    except Exception as cleanup_exc:
        issues = (
            grounding.get("grounding_issues", [])
            if isinstance(grounding.get("grounding_issues", []), list)
            else []
        )
        issues.append(f"grounding_cleanup_warning: {safe_text(cleanup_exc)}")
        grounding["grounding_issues"] = issues

    if isinstance(grounding.get("hotel_plan"), dict):
        hotel_plan = plan.get("hotel_plan", {}) if isinstance(plan.get("hotel_plan"), dict) else {}
        hotel_plan.update(grounding.get("hotel_plan", {}))
        hotel_plan["online_verified"] = bool(hotel_plan.get("source_urls"))
        hotel_plan["real_hotel_reference"] = bool(hotel_plan.get("source_urls"))
        plan["hotel_plan"] = hotel_plan
    if isinstance(grounding.get("transportation_plan"), dict):
        transport_plan = (
            plan.get("transportation_plan", {})
            if isinstance(plan.get("transportation_plan"), dict)
            else {}
        )
        transport_plan.update(grounding.get("transportation_plan", {}))
        transport_plan["online_verified"] = bool(
            transport_plan.get("metro_routes") or transport_plan.get("source_urls")
        )
        plan["transportation_plan"] = transport_plan
    restaurants = (
        grounding.get("restaurants", [])
        if isinstance(grounding.get("restaurants", []), list)
        else []
    )
    restaurant_lookup = {}
    for item in restaurants:
        if isinstance(item, dict):
            key = (safe_int(item.get("day"), 0), safe_text(item.get("meal")).lower())
            # Prefer the first normalized item for each day/meal.
            restaurant_lookup.setdefault(key, item)
    hotel_name = (
        safe_text(plan.get("hotel_plan", {}).get("hotel_name"))
        if isinstance(plan.get("hotel_plan"), dict)
        else ""
    )
    transport = (
        plan.get("transportation_plan", {})
        if isinstance(plan.get("transportation_plan"), dict)
        else {}
    )
    daily_routes = (
        transport.get("daily_local_transport", [])
        if isinstance(transport.get("daily_local_transport", []), list)
        else []
    )
    daily_route_lookup = {
        safe_int(item.get("day"), 0): item for item in daily_routes if isinstance(item, dict)
    }
    per_day = (
        plan.get("per_day_itinerary", [])
        if isinstance(plan.get("per_day_itinerary", []), list)
        else []
    )
    metro_routes = (
        transport.get("metro_routes", [])
        if isinstance(transport.get("metro_routes", []), list)
        else []
    )
    for idx, day_item in enumerate(per_day):
        if not isinstance(day_item, dict):
            continue
        day_num = safe_int(day_item.get("day"), idx + 1)
        if hotel_name:
            day_item["hotel"] = hotel_name
        for meal_key in ["lunch", "dinner"]:
            meal = day_item.get(meal_key, {}) if isinstance(day_item.get(meal_key), dict) else {}
            source = restaurant_lookup.get((day_num, meal_key))
            if source:
                meal["restaurant_name"] = safe_text(source.get("restaurant_name"))
                meal["restaurant_type"] = (
                    safe_text(source.get("restaurant_type"))
                    or safe_text(source.get("name_or_type"))
                    or safe_text(meal.get("restaurant_type"))
                )
                meal["restaurant_area"] = (
                    safe_text(source.get("area"))
                    or safe_text(source.get("restaurant_area"))
                    or safe_text(source.get("address_or_location"))
                )
                meal["address_or_location"] = (
                    safe_text(source.get("address_or_location")) or meal["restaurant_area"]
                )
                meal["specific_restaurant_found"] = bool(source.get("specific_restaurant_found"))
                meal["restaurant_grounding_status"] = safe_text(
                    source.get("restaurant_grounding_status")
                ) or (
                    "specific_name_found"
                    if meal["restaurant_name"]
                    else "area_fallback_no_specific_name"
                )
                meal["estimated_price_per_person"] = clamp_restaurant_price(
                    source.get("estimated_price_per_person") or source.get("price_per_person"),
                    safe_text(source.get("price_level"))
                    or safe_text(plan.get("restaurant_price_level"))
                    or "medium",
                )
                meal["estimated_total_cost"] = round(
                    meal["estimated_price_per_person"] * max(len(family_agents), 1), 2
                )
                meal["price_level"] = (
                    safe_text(source.get("price_level"))
                    or safe_text(plan.get("restaurant_price_level"))
                    or "medium"
                )
                meal["source_urls"] = (
                    source.get("source_urls", [])
                    if isinstance(source.get("source_urls", []), list)
                    else []
                )
                meal["comfort_note"] = safe_text(source.get("comfort_note"))
                day_item[meal_key] = meal
        route_item = daily_route_lookup.get(day_num)
        if route_item:
            day_item["transportation"] = [safe_text(route_item.get("route"))]
            day_item["estimated_local_transport_cost"] = _money(
                route_item.get("estimated_cost_total"), 0
            )
        elif metro_routes:
            day_item["transportation"] = [safe_text(metro_routes[idx % len(metro_routes)])]
        per_day[idx] = day_item
    plan["per_day_itinerary"] = per_day
    # Re-stitch each day's local route NOW that real restaurant names are merged into
    # per_day, so the route reflects the grounded restaurant instead of the raw
    # '菜系+价位' label captured by the earlier normalize_transport_costs pass.
    for idx, day_item in enumerate(per_day):
        if not isinstance(day_item, dict):
            continue
        day_num = safe_int(day_item.get("day"), idx + 1)
        new_route = _build_day_route(plan, day_num)[0]
        if not new_route:
            continue
        day_item["transportation"] = [new_route]
        ritem = daily_route_lookup.get(day_num)
        if isinstance(ritem, dict):
            ritem["route"] = new_route
    plan["per_day_itinerary"] = per_day
    tickets = (
        grounding.get("ticket_plan", [])
        if isinstance(grounding.get("ticket_plan", []), list)
        else []
    )
    plan = attach_ticket_prices_to_per_day(plan, tickets)
    breakdown = normalize_cost_breakdown(grounding, plan, family_agents)
    plan["cost_breakdown"] = breakdown
    if breakdown.get("total_cost", 0) > 0:
        plan["total_cost"] = breakdown["total_cost"]
    plan["budget_note"] = breakdown.get("cost_note", plan.get("budget_note", ""))
    source_urls = (
        grounding.get("source_urls", [])
        if isinstance(grounding.get("source_urls", []), list)
        else []
    )
    if not source_urls:
        for key in ["hotel_plan", "restaurants", "transportation_plan", "ticket_plan"]:
            value = grounding.get(key)
            if isinstance(value, dict):
                source_urls.extend(
                    value.get("source_urls", [])
                    if isinstance(value.get("source_urls", []), list)
                    else []
                )
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and isinstance(item.get("source_urls", []), list):
                        source_urls.extend(item.get("source_urls", []))
    plan["online_grounding"] = {
        "enabled": ONLINE_GROUNDING_ENABLED,
        "provider": ONLINE_SEARCH_PROVIDER,
        "search_status": "success" if source_urls else "partial_or_unverified",
        "source_urls": list(dict.fromkeys([safe_text(x) for x in source_urls if safe_text(x)])),
        "grounding_issues": grounding.get("grounding_issues", [])
        if isinstance(grounding.get("grounding_issues", []), list)
        else [],
        "note": "具体酒店、餐厅、路线和费用来自在线搜索结果；价格和房态需预订前再次核验。",
    }
    unique_restaurant_keys = set()
    for item in restaurants:
        if isinstance(item, dict):
            name = (
                safe_text(item.get("restaurant_name"))
                or safe_text(item.get("name_or_type"))
                or safe_text(item.get("restaurant_type"))
            )
            if name:
                unique_restaurant_keys.add(_norm_name(name))
    ticket_total = safe_float(breakdown.get("attraction_ticket_cost"), 0)
    plan["real_world_grounding"] = {
        "hotel_realism_pass": bool(plan.get("hotel_plan", {}).get("hotel_name"))
        and (not is_generic_hotel_text(plan.get("hotel_plan", {}).get("hotel_name"))),
        "metro_route_realism_pass": bool(plan.get("transportation_plan", {}).get("metro_routes"))
        or bool(plan.get("transportation_plan", {}).get("daily_local_transport")),
        "restaurant_diversity_pass": len(unique_restaurant_keys)
        >= max(2, min(safe_int(plan.get("day"), 1), 4)),
        "ticket_cost_included": ticket_total > 0 or not collect_daily_poi_names(plan),
        "online_grounding_used": True,
        "structured_cache_saved": ONLINE_GROUNDING_CACHE_ENABLED,
        "issues": plan["online_grounding"].get("grounding_issues", []),
    }
    persist_structured_grounding_cache(plan, grounding, family_agents)
    return plan


def _clean_search_title_to_place_name(title: Any) -> str:
    text = safe_text(title)
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    # Search titles often append source names after separators. Keep the left
    # meaningful segment unless it is too short.
    parts = re.split(r"\s*[-_|｜]\s*", text)
    if parts:
        candidate = safe_text(parts[0])
        if len(candidate) >= 2:
            text = candidate
    text = re.sub(
        r"(官方网站|官网|门票|票价|攻略|旅游攻略|推荐|排名|大众点评|Tripadvisor|携程|马蜂窝|百度百科)$", "", text, flags=re.I
    ).strip()
    # Keep short, name-like title; snippets remain separately available.
    return text[:40]


def _extract_rough_day_intents_from_negotiation(
    negotiation_info: Dict[str, Any], day_count: int
) -> List[Dict[str, Any]]:
    final_plan = (
        negotiation_info.get("round4_final_plan", {}) if isinstance(negotiation_info, dict) else {}
    )
    plan_items = []
    if isinstance(final_plan, dict):
        plan_items = final_plan.get("core_consensus_plan", [])
    if not isinstance(plan_items, list):
        plan_items = []
    output: List[Dict[str, Any]] = []
    for idx in range(max(day_count, len(plan_items))):
        item = (
            plan_items[idx] if idx < len(plan_items) and isinstance(plan_items[idx], dict) else {}
        )
        day_num = safe_int(item.get("day"), idx + 1)
        intents = []
        for key in [
            "core_theme",
            "day_theme",
            "morning_intent",
            "afternoon_intent",
            "evening_intent",
            "policy",
        ]:
            val = safe_text(item.get(key))
            if val:
                intents.append(val)
        acts = item.get("core_activities") or item.get("activity_type_targets") or []
        if isinstance(acts, list):
            intents.extend([safe_text(x) for x in acts if safe_text(x)])
        elif safe_text(acts):
            intents.append(safe_text(acts))
        output.append(
            {"day": day_num, "raw_intent": "；".join(dict.fromkeys([x for x in intents if x]))}
        )
    return output[: max(day_count, 1)]


def _proposal_place_terms(
    individual_proposals: List[Dict[str, Any]], max_terms: int = 12
) -> List[str]:
    terms: List[str] = []
    for item in individual_proposals or []:
        proposal = item.get("proposal", {}) if isinstance(item, dict) else {}
        if not isinstance(proposal, dict):
            continue
        for key in ["preferred_attractions", "proposal_priority"]:
            values = proposal.get(key, [])
            if not isinstance(values, list):
                continue
            for v in values:
                if isinstance(v, dict):
                    name = safe_text(v.get("name_or_type") or v.get("need") or v.get("activity"))
                else:
                    name = safe_text(v)
                if name and len(name) <= 60:
                    terms.append(name)
    return list(dict.fromkeys(terms))[:max_terms]


def build_grounded_activity_candidates(
    destination_city: str,
    day_count: int,
    family_agents: List[Dict[str, Any]],
    individual_proposals: List[Dict[str, Any]],
    negotiation_info: Dict[str, Any],
    conflict_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Pre-ground Module 4 rough intents into concrete POI candidates.

    This does not finalize the itinerary. It gives Module 5 search-backed
    options so the generation model can replace broad intents like "culture",
    "shopping", "theme park", or "photo spot" with concrete named places.
    Failures are soft: an empty candidate set should not break Module 5.
    """
    destination = safe_text(destination_city)
    if not MODULE5_ACTIVITY_GROUNDING_BEFORE_GENERATION or not destination:
        return {
            "enabled": MODULE5_ACTIVITY_GROUNDING_BEFORE_GENERATION,
            "used_online_search": False,
            "daily_candidates": [],
            "source_urls": [],
            "issues": [],
        }

    # Seeded/cached attractions for this city become offline activity candidates, so vague
    # themes ("主题公园乐趣"/"购物与美食"/"休闲") can be filled from the snapshot even when
    # Tavily is disabled or returns 0 usable results.
    snapshot_candidates: List[Dict[str, Any]] = []
    for r in _load_category_cache("attractions"):
        if not isinstance(r, dict) or not _cache_record_matches_destination(r, destination):
            continue
        poi = safe_text(r.get("poi_name"))
        if not poi:
            continue
        srcs = r.get("source_urls") if isinstance(r.get("source_urls"), list) else []
        snapshot_candidates.append(
            {
                "poi_name": poi,
                "area": safe_text(r.get("location_or_area")),
                "source_url": safe_text(srcs[0]) if srcs else "",
                "confidence": "local_snapshot_candidate",
            }
        )

    if not ONLINE_GROUNDING_ENABLED:
        # Offline / snapshot-only mode: still hand the seeded POIs to Module 5.
        if not snapshot_candidates:
            return {
                "enabled": True,
                "used_online_search": False,
                "daily_candidates": [],
                "source_urls": [],
                "issues": ["online_disabled_and_no_snapshot_candidates"],
            }
        offline_intents = _extract_rough_day_intents_from_negotiation(
            negotiation_info, max(1, safe_int(day_count, 1))
        )
        offline_daily = [
            {
                "day": safe_int(d.get("day"), i + 1),
                "rough_intent": safe_text(d.get("raw_intent")),
                "query": "",
                "candidates": snapshot_candidates,
            }
            for i, d in enumerate(offline_intents)
        ]
        return {
            "enabled": True,
            "used_online_search": False,
            "provider": "local_snapshot",
            "daily_candidates": offline_daily,
            "source_urls": [c["source_url"] for c in snapshot_candidates if c["source_url"]][:20],
            "issues": ["online_disabled_used_snapshot_candidates"],
        }

    issues: List[str] = []
    source_urls: List[str] = []
    proposal_terms = _proposal_place_terms(individual_proposals)
    # Keep explicit must_include terms visible, but use broad terms as query
    # context rather than accepting them blindly as final POIs.
    must_include_terms: List[str] = []
    for agent in family_agents or []:
        raw = agent.get("must_include") or (
            agent.get("constraints", {}) if isinstance(agent.get("constraints"), dict) else {}
        ).get("must_include")
        if isinstance(raw, list):
            must_include_terms.extend([safe_text(x) for x in raw if safe_text(x)])
        elif safe_text(raw):
            must_include_terms.append(safe_text(raw))
    context_terms = list(dict.fromkeys(proposal_terms + must_include_terms))[:16]
    daily_intents = _extract_rough_day_intents_from_negotiation(
        negotiation_info, max(1, safe_int(day_count, 1))
    )

    daily_candidates: List[Dict[str, Any]] = []
    query_count = 0
    # Seed-first: when the durable seed already covers this city with enough
    # concrete attraction candidates, use them and skip the Google Places / Tavily
    # call for that day.  The provider is only consulted when the seed is too thin.
    enough_seed = SEED_FIRST_GROUNDING and len(snapshot_candidates) >= SEED_FIRST_MIN_CANDIDATES
    unit_requirements = {
        "destination_city": destination,
        "days": day_count,
        "member_count": len(family_agents or []),
    }
    for day_item in daily_intents:
        day_num = safe_int(day_item.get("day"), len(daily_candidates) + 1)
        intent = safe_text(day_item.get("raw_intent"))
        if enough_seed:
            daily_candidates.append(
                {
                    "day": day_num,
                    "rough_intent": intent,
                    "query": "",
                    "candidates": snapshot_candidates,
                    "source": "grounding_seed",
                }
            )
            continue
        if query_count >= MODULE5_ACTIVITY_GROUNDING_MAX_QUERIES:
            # Out of query budget: still hand the seeded candidates to Module 5.
            daily_candidates.append(
                {
                    "day": day_num,
                    "rough_intent": intent,
                    "query": "",
                    "candidates": snapshot_candidates,
                    "source": "grounding_seed_quota_reached",
                }
            )
            continue
        # Add a little proposal context, but keep query short to avoid Tavily 432.
        context = " ".join(context_terms[:5])
        query = f"{destination} {intent} {context} 具体景点 推荐 家庭 交通便利"[:180]
        try:
            results = online_search(query, max_results=MODULE5_ACTIVITY_GROUNDING_RESULTS_PER_QUERY)
        except Exception as exc:
            # Config errors only abort when online grounding is required; otherwise
            # fall back to seed candidates (online_search already soft-fails them).
            if _is_api_configuration_error(exc) and REQUIRE_ONLINE_GROUNDING:
                raise
            issues.append(f"activity_grounding_search_failed:day{day_num}:{safe_text(exc)}")
            results = []
        query_count += 1
        candidates = []
        for r in results or []:
            if not isinstance(r, dict):
                continue
            name = _clean_search_title_to_place_name(r.get("title"))
            url = safe_text(r.get("url"))
            snippet = safe_text(r.get("content"))[:180]
            if url:
                source_urls.append(url)
            if not name:
                continue
            candidates.append(
                {
                    "poi_name": name,
                    "intent": intent[:120],
                    "source_title": safe_text(r.get("title"))[:100],
                    "source_url": url,
                    "source_snippet": snippet,
                    "confidence": "search_result_candidate",
                }
            )
            # Evaluate-then-bundle: a fresh Google Places candidate is gated, and a
            # passing attraction is saved and linked into a complete trip unit so it
            # is available seed-first next time.
            attraction_record = {
                "destination_city": destination,
                "poi_name": name,
                "location_or_area": "",
                "source_urls": [url] if url else [],
                "rating": r.get("rating"),
                "category": "attractions",
            }
            verdict = evaluate_grounding_candidate("attractions", attraction_record)
            if verdict["passed"]:
                _upsert_structured_cache("attractions", attraction_record)
                assemble_complete_trip_unit(
                    destination, attraction_record, family_agents, unit_requirements
                )
            else:
                issues.append(
                    f"candidate_rejected:day{day_num}:{name[:40]}:{','.join(verdict['reasons'])[:60]}"
                )
        merged = candidates[:MODULE5_ACTIVITY_GROUNDING_RESULTS_PER_QUERY]
        seen_names = {safe_text(c.get("poi_name")) for c in merged}
        for sc in snapshot_candidates:
            if safe_text(sc.get("poi_name")) and safe_text(sc.get("poi_name")) not in seen_names:
                merged.append(sc)
                seen_names.add(safe_text(sc.get("poi_name")))
        daily_candidates.append(
            {
                "day": day_num,
                "rough_intent": intent,
                "query": query,
                "candidates": merged,
                "source": "google_places_topup" if query_count else "grounding_seed",
            }
        )

    return {
        "enabled": True,
        "used_online_search": query_count > 0,
        "seed_first": bool(SEED_FIRST_GROUNDING),
        "seed_candidate_count": len(snapshot_candidates),
        "provider": "grounding_seed" if query_count == 0 else ONLINE_SEARCH_PROVIDER,
        "daily_candidates": daily_candidates,
        "source_urls": list(dict.fromkeys(source_urls))[:20],
        "issues": issues,
        "note": "Module 5 pre-grounding candidates: seed-first, Google Places only when the seed is too thin. Passing online candidates are bundled into complete trip units and written back to the seed.",
    }


def ensure_online_grounding(
    plan: Dict[str, Any], family_agents: List[Dict[str, Any]]
) -> Dict[str, Any]:
    if isinstance(plan.get("online_grounding"), dict) and plan["online_grounding"].get(
        "search_status"
    ) in ["success", "partial_or_unverified", "failed", "disabled"]:
        return plan
    requirements = build_grounding_requirements(plan, family_agents)
    plan["grounding_requirements"] = requirements
    if not ONLINE_GROUNDING_ENABLED:
        plan["online_grounding"] = {
            "enabled": False,
            "provider": ONLINE_SEARCH_PROVIDER,
            "search_status": "disabled",
            "source_urls": [],
            "grounding_issues": ["online_grounding_disabled"],
            "note": "在线具体化已关闭，未使用内置参考表。",
        }
        return plan
    try:
        # First load prior online-derived structured facts from data/cache.
        # They are used as local evidence for the same destination/POIs, then
        # fresh web results are appended when online grounding is enabled.
        relevant_cache = load_relevant_grounding_cache(requirements)
        destination_city = safe_text(requirements.get("destination_city"))
        # Seed-first coverage check per category: only call the online provider for
        # a category when the durable seed has too few suitable entries for the city.
        # Seed-first is meal/day-count aware, not all-or-nothing: only skip the
        # online provider for a category when the seed can cover the WHOLE trip
        # (e.g. 2 meals/day, ~2 attractions/day, 1 route/day).  If the seed is
        # short — e.g. 3 Harbin restaurants for a 3-day, 6-meal trip — the API is
        # still called to fill the gap, while seed entries remain in the candidate
        # pool.  This implements "seed first, API for the remainder".
        _days = max(
            1,
            safe_int(
                requirements.get("days")
                or requirements.get("day")
                or requirements.get("day_count"),
                1,
            ),
        )
        _need_meals = _days * 2
        _need_attractions = _days * 2
        _need_routes = _days
        seed_first_hotels = (
            SEED_FIRST_GROUNDING
            and len(_seed_candidates_for_city("hotels", destination_city)) >= SEED_FIRST_MIN_HOTELS
        )
        seed_first_restaurants = SEED_FIRST_GROUNDING and len(
            _seed_candidates_for_city("restaurants", destination_city)
        ) >= max(SEED_FIRST_MIN_RESTAURANTS, _need_meals)
        seed_first_attractions = SEED_FIRST_GROUNDING and len(
            _seed_candidates_for_city("attractions", destination_city)
        ) >= max(SEED_FIRST_MIN_CANDIDATES, _need_attractions)
        seed_first_transport = SEED_FIRST_GROUNDING and len(
            _seed_candidates_for_city("transport_routes", destination_city)
        ) >= max(SEED_FIRST_MIN_ROUTES, _need_routes)

        hotel_cache_results = structured_cache_as_search_results(relevant_cache, "hotels")
        if seed_first_hotels and hotel_cache_results:
            hotel_results = hotel_cache_results[:8]
        else:
            hotel_results = (
                hotel_cache_results + run_search_queries(build_hotel_online_queries(requirements))
            )[:8]
        restaurant_cache_results = structured_cache_as_search_results(relevant_cache, "restaurants")
        attraction_cache_results = structured_cache_as_search_results(relevant_cache, "attractions")
        if seed_first_restaurants and restaurant_cache_results:
            other_results = (restaurant_cache_results[:6] + attraction_cache_results[:4])[:10]
        else:
            other_results = (
                restaurant_cache_results[:4]
                + attraction_cache_results[:4]
                + run_search_queries(build_restaurant_ticket_online_queries(requirements))[:8]
            )

        # Stage 1: ground hotel and meals/tickets from the complete itinerary draft.
        # The hotel prompt now chooses the nearest qualified hotel to the itinerary
        # anchors, instead of any generic family hotel.
        hotel_grounding = extract_hotel_grounding_with_llm(
            plan=plan, requirements=requirements, search_results=hotel_results
        )
        other_grounding = extract_restaurant_ticket_grounding_with_llm(
            plan=plan, requirements=requirements, search_results=other_results
        )
        # v55: explicit Tavily/raw-search restaurant pass.  It searches concrete
        # restaurants day-by-day and meal-by-meal, then replaces generic LLM
        # entries for the same day/meal when a supported restaurant name exists.
        # v58/v14: avoid shadowing the function name.  The old code used
        # `tavily_restaurant_grounding = tavily_restaurant_grounding(...)`, which
        # raises UnboundLocalError at runtime even though compileall passes.
        # v60: seed-first — skip the live restaurant search when the seed already
        # has enough restaurants for this city.
        restaurant_api_grounding = (
            {}
            if seed_first_restaurants
            else tavily_restaurant_grounding(plan, requirements, family_agents)
        )
        if isinstance(restaurant_api_grounding, dict):
            tavily_restaurants = (
                restaurant_api_grounding.get("restaurants", [])
                if isinstance(restaurant_api_grounding.get("restaurants", []), list)
                else []
            )
            if tavily_restaurants:
                existing_restaurants = (
                    other_grounding.get("restaurants", [])
                    if isinstance(other_grounding.get("restaurants", []), list)
                    else []
                )
                by_key = {}
                for item in existing_restaurants:
                    if isinstance(item, dict):
                        by_key[
                            (
                                safe_int(item.get("day"), 0),
                                safe_text(item.get("meal")).lower() or "lunch",
                            )
                        ] = item
                for item in tavily_restaurants:
                    if not isinstance(item, dict):
                        continue
                    key = (
                        safe_int(item.get("day"), 0),
                        safe_text(item.get("meal")).lower() or "lunch",
                    )
                    new_name = safe_text(item.get("restaurant_name"))
                    old_item = by_key.get(key)
                    old_name = (
                        safe_text(old_item.get("restaurant_name"))
                        if isinstance(old_item, dict)
                        else ""
                    )
                    new_specific = bool(new_name) and not _is_placeholder_restaurant_name(new_name)
                    old_specific = bool(old_name) and not _is_placeholder_restaurant_name(old_name)
                    # Keep exact Tavily names when found.  When no exact name is
                    # found, still keep the day/meal area fallback so TXT export can
                    # say e.g. "天津大学站附近" instead of a city-level placeholder.
                    if new_specific or (not old_specific):
                        by_key[key] = item
                other_grounding["restaurants"] = list(by_key.values())
            restaurant_issues = (
                restaurant_api_grounding.get("grounding_issues", [])
                if isinstance(restaurant_api_grounding.get("grounding_issues", []), list)
                else []
            )
            if restaurant_issues:
                other_grounding["grounding_issues"] = (
                    other_grounding.get("grounding_issues", [])
                    if isinstance(other_grounding.get("grounding_issues", []), list)
                    else []
                ) + restaurant_issues
        # v52: explicit Tavily/raw-search ticket price pass. This overrides the
        # old generic 40 CNY fallback and records missing prices as soft issues.
        # v60: seed-first — skip when the seed already covers this city's attractions.
        tavily_ticket_grounding = (
            {}
            if seed_first_attractions
            else tavily_ticket_price_grounding(plan, requirements, family_agents)
        )
        if isinstance(tavily_ticket_grounding, dict):
            if tavily_ticket_grounding.get("ticket_plan"):
                existing_tickets = (
                    other_grounding.get("ticket_plan", [])
                    if isinstance(other_grounding.get("ticket_plan", []), list)
                    else []
                )
                by_key = {
                    _norm_name(x.get("poi_name")): x
                    for x in existing_tickets
                    if isinstance(x, dict)
                }
                for item in tavily_ticket_grounding.get("ticket_plan", []):
                    if isinstance(item, dict):
                        by_key[_norm_name(item.get("poi_name"))] = item
                other_grounding["ticket_plan"] = list(by_key.values())
            ticket_issues = (
                tavily_ticket_grounding.get("grounding_issues", [])
                if isinstance(tavily_ticket_grounding.get("grounding_issues", []), list)
                else []
            )
            if ticket_issues:
                other_grounding["grounding_issues"] = (
                    other_grounding.get("grounding_issues", [])
                    if isinstance(other_grounding.get("grounding_issues", []), list)
                    else []
                ) + ticket_issues
        has_meal_or_ticket = bool(other_grounding.get("restaurants")) or bool(
            other_grounding.get("ticket_plan")
        )
        if not has_meal_or_ticket:
            fallback_grounding = build_rule_based_restaurant_ticket_grounding(
                plan=plan,
                requirements=requirements,
                search_results=other_results,
                family_agents=family_agents,
            )
            issues = (
                other_grounding.get("grounding_issues", [])
                if isinstance(other_grounding.get("grounding_issues", []), list)
                else []
            )
            issues.append("restaurant_ticket_llm_returned_no_entries_rule_fallback_used")
            fallback_grounding["grounding_issues"] = issues + (
                fallback_grounding.get("grounding_issues", [])
                if isinstance(fallback_grounding.get("grounding_issues", []), list)
                else []
            )
            other_grounding = fallback_grounding

        # Stage 2: after hotel + itinerary stops are known, run the transport
        # search/API and stitch the whole daily route chain.
        transport_context_plan = build_plan_context_for_post_generation_transport(
            plan, hotel_grounding, other_grounding
        )
        transport_requirements = build_grounding_requirements(transport_context_plan, family_agents)
        plan["grounding_requirements"] = transport_requirements
        # v60: seed-first — reuse seeded transport routes for this city and skip the
        # live route search when the seed already has enough routes.
        transport_cache_results = structured_cache_as_search_results(
            relevant_cache, "transport_routes"
        )
        if seed_first_transport and transport_cache_results:
            transport_results = transport_cache_results[:10]
        else:
            transport_results = (
                transport_cache_results
                + run_search_queries(build_transport_online_queries(transport_requirements))
            )[:10]
        transport_grounding = extract_transport_grounding_with_llm(
            plan=transport_context_plan,
            requirements=transport_requirements,
            search_results=transport_results,
        )
        tavily_transport_grounding = (
            {}
            if seed_first_transport
            else tavily_transport_cost_grounding(
                transport_context_plan, transport_requirements, family_agents
            )
        )
        # v52: fill missing or vague route-cost entries with Tavily transport estimates.
        if isinstance(tavily_transport_grounding, dict):
            t_plan = (
                transport_grounding.get("transportation_plan", {})
                if isinstance(transport_grounding.get("transportation_plan"), dict)
                else {}
            )
            api_plan = (
                tavily_transport_grounding.get("transportation_plan", {})
                if isinstance(tavily_transport_grounding.get("transportation_plan"), dict)
                else {}
            )
            llm_daily = (
                t_plan.get("daily_local_transport", [])
                if isinstance(t_plan.get("daily_local_transport", []), list)
                else []
            )
            api_daily = (
                api_plan.get("daily_local_transport", [])
                if isinstance(api_plan.get("daily_local_transport", []), list)
                else []
            )
            llm_by_day = {safe_int(x.get("day"), 0): x for x in llm_daily if isinstance(x, dict)}
            for item in api_daily:
                if not isinstance(item, dict):
                    continue
                day = safe_int(item.get("day"), 0)
                old_item = llm_by_day.get(day)
                old_route = safe_text(old_item.get("route")) if isinstance(old_item, dict) else ""
                old_cost = (
                    _money(old_item.get("estimated_cost_total"), 0)
                    if isinstance(old_item, dict)
                    else 0
                )
                if (not old_item) or is_generic_transport_text(old_route) or old_cost <= 0:
                    llm_by_day[day] = item
                else:
                    # Preserve the more detailed route but attach Tavily cost evidence.
                    old_item.setdefault("source_urls", [])
                    if isinstance(old_item.get("source_urls"), list):
                        old_item["source_urls"] = list(
                            dict.fromkeys(
                                old_item.get("source_urls", []) + item.get("source_urls", [])
                            )
                        )[:6]
                    old_item.setdefault(
                        "cost_source", item.get("cost_source", "tavily_transport_search_estimate")
                    )
            if llm_by_day:
                t_plan["daily_local_transport"] = [
                    llm_by_day[k] for k in sorted(llm_by_day.keys()) if k
                ]
                t_plan["local_transport_total_cost"] = round(
                    sum(
                        _money(x.get("estimated_cost_total"), 0)
                        for x in t_plan["daily_local_transport"]
                    ),
                    2,
                )
            for k, v in api_plan.items():
                if k not in t_plan or not t_plan.get(k):
                    t_plan[k] = v
            transport_grounding["transportation_plan"] = t_plan
            api_issues = (
                tavily_transport_grounding.get("transport_grounding_issues", [])
                if isinstance(
                    tavily_transport_grounding.get("transport_grounding_issues", []), list
                )
                else []
            )
            if api_issues:
                transport_grounding["transport_grounding_issues"] = (
                    transport_grounding.get("transport_grounding_issues", [])
                    if isinstance(transport_grounding.get("transport_grounding_issues", []), list)
                    else []
                ) + api_issues

        grounding = {}
        for partial in [other_grounding, hotel_grounding, transport_grounding]:
            if isinstance(partial, dict):
                grounding.update(partial)
        grounding["source_urls"] = []
        for result_group in [hotel_results, other_results, transport_results]:
            for item in result_group:
                url = safe_text(item.get("url"))
                if url:
                    grounding["source_urls"].append(url)
        grounding["grounding_issues"] = []
        for key in ["grounding_issues", "hotel_grounding_issues", "transport_grounding_issues"]:
            value = grounding.get(key)
            if isinstance(value, list):
                grounding["grounding_issues"].extend(value)
        plan = merge_online_grounding_into_plan(plan, grounding, family_agents)
        return plan
    except Exception as exc:
        if REQUIRE_ONLINE_GROUNDING:
            if _is_api_configuration_error(exc):
                raise RuntimeError(
                    f"在线具体化失败：{exc}\n"
                    "请在 code/.env 中配置 ONLINE_SEARCH_PROVIDER=tavily/serpapi/bing 以及对应 API Key，例如 TAVILY_API_KEY=...。"
                )
            raise RuntimeError(f"在线具体化失败：{exc}\n" "这不是 .env 配置问题，而是在线具体化过程中的代码或模型输出处理错误。")
        plan["online_grounding"] = {
            "enabled": ONLINE_GROUNDING_ENABLED,
            "provider": ONLINE_SEARCH_PROVIDER,
            "search_status": "failed",
            "source_urls": [],
            "grounding_issues": [safe_text(exc)],
            "note": "在线查询失败，且 REQUIRE_ONLINE_GROUNDING=false，所以保留 LLM 需求层结果；未使用内置参考表。",
        }
        plan["real_world_grounding"] = {
            "hotel_realism_pass": False,
            "metro_route_realism_pass": False,
            "online_grounding_used": False,
            "issues": [safe_text(exc)],
        }
        return plan
