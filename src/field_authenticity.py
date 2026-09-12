"""Field-authenticity validation against Google Maps (legacy Place Details + Text Search).

Validates the *entity* fields of a generated plan — hotels, restaurants, attractions —
for real-world authenticity, and returns a graded status per the project spec:

    verified | partially_verified | unverified | contradicted | not_applicable

For each entity it checks (only with signals the project's Google quota actually
provides — the New Places API and the Routes API are NOT enabled, legacy Text Search
and Place Details ARE):

  A. existence      place_id resolvable          -> entity_score
  B. city match     formatted_address / lat-lng in destination city
  C. route realism  haversine between same-day stops (Routes API unavailable, so this
                    is a straight-line LOWER BOUND, not a real road duration)
  D. opening hours  Place Details opening_hours when present, else unknown
  E. price          price_level (0-4) -> price_grounded; an exact amount is only an
                    estimate, never "fully verified"

Aggregates into three reportable rates: Entity Grounding Rate, Spatiotemporal
Consistency Rate, Route Feasibility Rate.
"""
from __future__ import annotations

import difflib
import math
from typing import Any, Dict, List, Optional

from src.common import safe_int, safe_text
from src.grounding import (
    _online_result_in_city,
    is_generic_place_label,
    lookup_place_details,
    resolve_place_id,
)

# Five-class verification status.
VERIFIED = "verified"
PARTIALLY_VERIFIED = "partially_verified"
UNVERIFIED = "unverified"
CONTRADICTED = "contradicted"
NOT_APPLICABLE = "not_applicable"

# Expected Google place `types` per entity category (intersection => category match).
_EXPECTED_TYPES = {
    "hotel": {"lodging", "hotel", "resort_hotel", "motel"},
    "restaurant": {"restaurant", "food", "cafe", "meal_takeaway", "meal_delivery", "bakery", "bar"},
    "attraction": {
        "tourist_attraction",
        "park",
        "national_park",
        "museum",
        "art_gallery",
        "zoo",
        "aquarium",
        "amusement_park",
        "place_of_worship",
        "hindu_temple",
        "church",
        "natural_feature",
        "point_of_interest",
        "landmark",
        "establishment",
        "shopping_mall",
        "stadium",
    },
}

# Vague, non-groundable entity phrasings.
_VAGUE_HINTS = (
    "附近",
    "周边",
    "当地",
    "本地",
    "a local",
    "nearby",
    "local restaurant",
    "某",
    "合适",
    "推荐的",
    "一家",
    "某个",
)

# Same-day consecutive stops farther apart than this (straight line) are physically
# implausible for a single intra-city day -> route flagged unrealistic.
MAX_INTRACITY_KM = 60.0
# Per-dimension weights for the authenticity score (re-normalised over applicable ones).
_WEIGHTS = {
    "entity": 0.30,
    "city": 0.20,
    "category": 0.15,
    "time": 0.15,
    "route": 0.10,
    "source": 0.10,
}


def _haversine_km(a_lat: float, a_lng: float, b_lat: float, b_lng: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dphi = math.radians(b_lat - a_lat)
    dlmb = math.radians(b_lng - a_lng)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return round(2 * r * math.asin(min(1.0, math.sqrt(h))), 2)


def _is_vague(name: str) -> bool:
    n = safe_text(name)
    if not n or is_generic_place_label(n):
        return True
    low = n.lower()
    return any(h in low for h in _VAGUE_HINTS)


def _category_match(types: List[Any], category: str) -> Optional[bool]:
    expected = _EXPECTED_TYPES.get(category)
    if not expected:
        return None
    if not types:
        return None  # no type info -> not_applicable for this check
    return any(safe_text(t).lower() in expected for t in types)


def _authenticity_score(checks: Dict[str, Optional[bool]]) -> float:
    num = den = 0.0
    for dim, weight in _WEIGHTS.items():
        v = checks.get(dim)
        if v is None:
            continue  # not_applicable -> excluded from the (re-normalised) score
        den += weight
        num += weight if v else 0.0
    return round(num / den, 3) if den > 0 else 0.0


def validate_entity(
    value: Any, destination_city: Any, category: str, place_id: Any = "", source_url: Any = ""
) -> Dict[str, Any]:
    """Validate one named entity field against Google Maps. See module docstring."""
    name = safe_text(value)
    res: Dict[str, Any] = {
        "field": category,
        "generated": name,
        "matched_place_id": "",
        "checks": {},
        "details": {},
    }
    if _is_vague(name):
        res.update(
            status=UNVERIFIED, error_type="vague_entity", confidence=0.0, authenticity_score=0.0
        )
        return res

    # Resolve regardless of city so a real place in the WRONG city is detected as a
    # mismatch (contradicted) rather than simply "not found".
    pid = safe_text(place_id) or resolve_place_id(name, destination_city, require_city=False)
    if not pid:
        res.update(
            status=UNVERIFIED, error_type="entity_not_found", confidence=0.0, authenticity_score=0.0
        )
        return res
    res["matched_place_id"] = pid
    det = lookup_place_details(pid)
    res["details"] = {
        k: det.get(k) for k in ("name", "address", "lat", "lng", "price_level", "rating")
    }

    # Name match: the resolved place should actually BE the named entity.
    nm_score = (
        round(
            difflib.SequenceMatcher(None, name.lower(), safe_text(det.get("name")).lower()).ratio(),
            3,
        )
        if det.get("name")
        else 1.0
    )
    res["name_match_score"] = nm_score
    if (
        det.get("name")
        and nm_score < 0.45
        and name not in safe_text(det.get("name"))
        and safe_text(det.get("name")) not in name
    ):
        res.update(
            status=UNVERIFIED, error_type="entity_not_found", confidence=0.0, authenticity_score=0.0
        )
        return res

    checks: Dict[str, Optional[bool]] = {}
    checks["entity"] = True  # a concrete place_id was resolved
    checks["source"] = bool(pid or safe_text(source_url))
    # City match: address (or, fallback, the resolved name) carries the destination city.
    addr = safe_text(det.get("address"))
    checks["city"] = _online_result_in_city(addr, destination_city) if addr else None
    # Category match against Google place types.
    checks["category"] = _category_match(det.get("types", []), category)
    # Opening hours present? (true presence; activity-time fit is checked at plan level.)
    checks["time"] = None if not det.get("opening_hours") else True
    # route dimension is judged at plan level (needs the neighbouring stop); leave N/A here.
    checks["route"] = None

    # Price grounding (E): a price_level grounds the tier; an exact amount stays estimate.
    res["price_grounded"] = det.get("price_level") is not None
    res["price_note"] = (
        "price_level=%s (档位, 非精确金额)" % det.get("price_level")
        if det.get("price_level") is not None
        else "price_level 缺失, 金额仅为估算"
    )

    # Status decision.
    if checks["city"] is False:
        status, err = CONTRADICTED, "entity_exists_but_city_mismatch"
    elif checks["category"] is False:
        status, err = CONTRADICTED, "category_mismatch"
    elif checks["city"] and (checks["category"] is not False):
        status = VERIFIED if checks["category"] else PARTIALLY_VERIFIED
        err = "" if status == VERIFIED else "category_unknown"
    else:
        status, err = PARTIALLY_VERIFIED, "city_unknown"

    res["checks"] = checks
    res["status"] = status
    res["error_type"] = err
    res["authenticity_score"] = _authenticity_score(checks)
    res["confidence"] = res["authenticity_score"]
    return res


def _plan_entities(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten a plan's hotel / restaurants / attractions into validatable entities."""
    out: List[Dict[str, Any]] = []
    dest = safe_text(plan.get("destination_city"))
    hotel = plan.get("hotel_plan", {}) if isinstance(plan.get("hotel_plan"), dict) else {}
    hname = safe_text(hotel.get("hotel_name"))
    if hname:
        out.append(
            {
                "category": "hotel",
                "value": hname,
                "place_id": safe_text(hotel.get("place_id")),
                "source_url": (hotel.get("source_urls") or [""])[0]
                if isinstance(hotel.get("source_urls"), list)
                else "",
                "day": 0,
                "destination_city": dest,
            }
        )
    per_day = (
        plan.get("per_day_itinerary", []) if isinstance(plan.get("per_day_itinerary"), list) else []
    )
    for idx, d in enumerate(per_day):
        if not isinstance(d, dict):
            continue
        day = safe_int(d.get("day"), idx + 1)
        for blk in ("morning", "afternoon"):
            for x in d.get(blk) or []:
                if isinstance(x, dict):
                    if x.get("is_transit") or x.get("poi_type") in ("arrival", "departure"):
                        continue
                    nm = safe_text(x.get("poi_name")) or safe_text(x.get("activity"))
                    pid = safe_text(x.get("place_id"))
                elif isinstance(x, str):
                    nm, pid = safe_text(x), ""
                else:
                    continue
                if nm:
                    out.append(
                        {
                            "category": "attraction",
                            "value": nm,
                            "place_id": pid,
                            "source_url": "",
                            "day": day,
                            "destination_city": dest,
                        }
                    )
        for meal in ("lunch", "dinner"):
            v = d.get(meal)
            if isinstance(v, dict) and safe_text(v.get("restaurant_name")):
                out.append(
                    {
                        "category": "restaurant",
                        "value": safe_text(v.get("restaurant_name")),
                        "place_id": safe_text(v.get("place_id")),
                        "source_url": (v.get("source_urls") or [""])[0]
                        if isinstance(v.get("source_urls"), list)
                        else "",
                        "day": day,
                        "destination_city": dest,
                    }
                )
    return out


def evaluate_plan_authenticity(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Validate every entity in a plan and aggregate the three reportable rates."""
    entities = _plan_entities(plan)
    results = [
        validate_entity(
            e["value"], e["destination_city"], e["category"], e.get("place_id"), e.get("source_url")
        )
        for e in entities
    ]
    for e, r in zip(entities, results):
        r["day"] = e["day"]

    total = len(results) or 1
    grounded = sum(1 for r in results if r.get("matched_place_id"))
    spatiotemporal_ok = sum(
        1 for r in results if r.get("checks", {}).get("city") and r.get("status") != CONTRADICTED
    )

    # Route feasibility: straight-line distance between SAME-DAY consecutive stops that
    # both have coordinates; a transfer is feasible when within MAX_INTRACITY_KM.
    by_day: Dict[int, List[Dict[str, Any]]] = {}
    for r in results:
        d = lookup_place_details(r.get("matched_place_id")) if r.get("matched_place_id") else {}
        if d.get("lat") is not None and d.get("lng") is not None:
            by_day.setdefault(r.get("day", 0), []).append(
                {"name": r.get("generated"), "lat": d["lat"], "lng": d["lng"]}
            )
    transfers = feasible = 0
    bad_transfers: List[str] = []
    for day, stops in by_day.items():
        if day == 0:
            continue  # hotel bucket, not a day route
        for a, b in zip(stops, stops[1:]):
            transfers += 1
            km = _haversine_km(a["lat"], a["lng"], b["lat"], b["lng"])
            if km <= MAX_INTRACITY_KM:
                feasible += 1
            else:
                bad_transfers.append(f"D{day} {a['name']}->{b['name']} 直线{km}km")

    scores = [r.get("authenticity_score", 0.0) for r in results]
    return {
        "entities": results,
        "entity_count": total,
        "entity_grounding_rate": round(grounded / total, 4),
        "spatiotemporal_consistency_rate": round(spatiotemporal_ok / total, 4),
        "route_feasibility_rate": round(feasible / transfers, 4) if transfers else None,
        "route_transfers_checked": transfers,
        "unrealistic_transfers": bad_transfers,
        "mean_authenticity_score": round(sum(scores) / total, 4),
        "status_breakdown": {
            s: sum(1 for r in results if r.get("status") == s)
            for s in (VERIFIED, PARTIALLY_VERIFIED, UNVERIFIED, CONTRADICTED, NOT_APPLICABLE)
        },
    }


def run_authenticity_audit(
    plans_csv: str = "results/module_7/generated_plans_feedback_final.csv", limit: int = 0
) -> Dict[str, Any]:
    """Audit every family's plan in a results CSV and report the corpus-level rates.

    Set ONLINE_SEARCH_PROVIDER=google_places (+ key) so entities can be resolved.
    """
    import csv
    import json as _json

    rows = list(csv.DictReader(open(plans_csv, encoding="utf-8")))
    if limit:
        rows = rows[:limit]
    per_family: List[Dict[str, Any]] = []
    for row in rows:
        try:
            plan = {
                "destination_city": row.get("destination_city"),
                "hotel_plan": _json.loads(row.get("hotel_plan_json", "{}") or "{}"),
                "per_day_itinerary": _json.loads(row.get("per_day_itinerary_json", "[]") or "[]"),
            }
        except Exception:
            continue
        rep = evaluate_plan_authenticity(plan)
        per_family.append(
            {
                "family_id": row.get("family_id"),
                "entity_grounding_rate": rep["entity_grounding_rate"],
                "spatiotemporal_consistency_rate": rep["spatiotemporal_consistency_rate"],
                "route_feasibility_rate": rep["route_feasibility_rate"],
                "mean_authenticity_score": rep["mean_authenticity_score"],
            }
        )

    def _avg(key):
        vals = [f[key] for f in per_family if f.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "families_audited": len(per_family),
        "corpus_entity_grounding_rate": _avg("entity_grounding_rate"),
        "corpus_spatiotemporal_consistency_rate": _avg("spatiotemporal_consistency_rate"),
        "corpus_route_feasibility_rate": _avg("route_feasibility_rate"),
        "corpus_mean_authenticity_score": _avg("mean_authenticity_score"),
        "per_family": per_family,
    }


if __name__ == "__main__":
    import json as _json
    import sys as _sys

    csv_path = (
        _sys.argv[1]
        if len(_sys.argv) > 1
        else "results/module_7/generated_plans_feedback_final.csv"
    )
    lim = int(_sys.argv[2]) if len(_sys.argv) > 2 else 0
    out = run_authenticity_audit(csv_path, lim)
    out.pop("per_family", None)
    print(_json.dumps(out, ensure_ascii=False, indent=2))
