"""Build and snapshot the grounding CSV database (hotels/restaurants/attractions/transport) used to ground itineraries."""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"
# The unified CSV database lives INSIDE grounding_seed (single durable folder).
SEED_DIR = DATA_ROOT / "grounding_seed"
OUT_DIR = SEED_DIR
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

PLACE_ID_RE = re.compile(r"place_id:([A-Za-z0-9_\-]+)")
HHMM_RANGE_RE = re.compile(r"(\d{1,2}:\d{2})\s*[-~–至到]\s*(\d{1,2}:\d{2})")


# ============================ build (JSON seed -> CSV) ============================
def _read_seed(name: str) -> list:
    path = SEED_DIR / f"{name}.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []


def _txt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return ";".join(_txt(x) for x in v if _txt(x))
    return str(v).strip()


def _num(v) -> str:
    """Return a clean numeric string, or '' when not a usable number."""
    if v is None or v == "":
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        m = re.search(r"\d+(?:\.\d+)?", str(v))
        if not m:
            return ""
        f = float(m.group(0))
    if f < 0:
        return ""
    return str(int(f)) if f == int(f) else str(round(f, 2))


def _http_source(rec: dict) -> str:
    for u in rec.get("source_urls") or []:
        u = _txt(u)
        if u.startswith(("http://", "https://")):
            return u
    return ""


def _place_id(rec: dict) -> str:
    for u in rec.get("source_urls") or []:
        m = PLACE_ID_RE.search(_txt(u))
        if m:
            return m.group(1)
    return ""


def _parse_hours(raw: str):
    m = HHMM_RANGE_RE.search(_txt(raw))
    if not m:
        return "", ""
    return m.group(1), m.group(2)


def _family_flag(rec: dict) -> str:
    fs = _txt(rec.get("family_suitability"))
    if not fs:
        return ""
    neg = any(w in fs for w in ["不适合", "不推荐", "不宜", "no", "not "])
    return "0" if neg else "1"


def _write_csv(path: Path, header: list, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_hotels(seed: list) -> list:
    rows = []
    for rec in seed:
        name = _txt(rec.get("hotel_name"))
        if not name:
            continue
        rows.append(
            {
                "name": name,
                "city": _txt(rec.get("destination_city")),
                "address": _txt(rec.get("address_or_location")) or _txt(rec.get("area")),
                "latitude": "",
                "longitude": "",
                "price_per_night": _num(rec.get("estimated_price_per_night")),
                "rating": _num(rec.get("rating")),
                "check_in": "",
                "check_out": "",
                "facilities": _txt(rec.get("available_lines")),
                "family_friendly": _family_flag(rec),
                "nearby_metro_station": _txt(rec.get("nearby_metro_station")),
                "price_level": _txt(rec.get("price_level")),
                "place_id": _place_id(rec),
                "source_url": _http_source(rec),
                "seed_cache_key": _txt(rec.get("cache_key")),
            }
        )
    rows.sort(key=lambda r: (r["city"], r["name"]))
    for i, r in enumerate(rows, 1):
        r["source_id"] = f"hotel_{i:04d}"
    return rows


def build_restaurants(seed: list) -> list:
    rows = []
    for rec in seed:
        name = _txt(rec.get("restaurant_name"))
        if not name:
            continue
        rows.append(
            {
                "name": name,
                "city": _txt(rec.get("destination_city")),
                "address": _txt(rec.get("area")),
                "latitude": "",
                "longitude": "",
                "avg_price": _num(rec.get("estimated_price_per_person")),
                "price_range_per_person": _txt(rec.get("price_range_per_person")),
                "open_time": "",
                "close_time": "",
                "cuisine": _txt(rec.get("restaurant_type")),
                "dietary_tags": "",
                "child_friendly": _family_flag(rec),
                "meal_hint": _txt(rec.get("meal_hint")),
                "nearby_metro_station": _txt(rec.get("nearby_metro_station")),
                "place_id": _place_id(rec),
                "source_url": _http_source(rec),
                "seed_cache_key": _txt(rec.get("cache_key")),
            }
        )
    rows.sort(key=lambda r: (r["city"], r["name"]))
    for i, r in enumerate(rows, 1):
        r["source_id"] = f"rest_{i:04d}"
    return rows


def build_attractions(seed: list) -> list:
    rows = []
    for rec in seed:
        name = _txt(rec.get("poi_name"))
        if not name:
            continue
        open_t, close_t = _parse_hours(rec.get("opening_hours"))
        rows.append(
            {
                "name": name,
                "city": _txt(rec.get("destination_city")),
                "address": _txt(rec.get("location_or_area")),
                "latitude": "",
                "longitude": "",
                "ticket_price": _num(rec.get("ticket_price_per_person")),
                "open_time": open_t,
                "close_time": close_t,
                "opening_hours_raw": _txt(rec.get("opening_hours")),
                "visit_duration_min": "",
                "tags": "",
                "walking_intensity": "",
                "place_id": _place_id(rec),
                "source_url": _http_source(rec),
                "seed_cache_key": _txt(rec.get("cache_key")),
            }
        )
    rows.sort(key=lambda r: (r["city"], r["name"]))
    for i, r in enumerate(rows, 1):
        r["source_id"] = f"attr_{i:04d}"
    return rows


def build_transportation(seed: list) -> list:
    rows = []
    for rec in seed:
        route = _txt(rec.get("route"))
        origin = _txt(rec.get("origin"))
        dest = _txt(rec.get("destination"))
        if not (route or (origin and dest)):
            continue
        rows.append(
            {
                "city": _txt(rec.get("destination_city")),
                "origin": origin,
                "destination": dest,
                "origin_city": _txt(rec.get("origin_city")),
                "route": route,
                "mode": _txt(rec.get("mode")),
                "duration_min": _num(rec.get("estimated_duration_minutes")),
                "cost_total": _num(rec.get("estimated_cost_total")),
                "cost_per_person": _num(rec.get("estimated_cost_per_person")),
                "day_hint": _num(rec.get("day_hint")),
                "source_url": _http_source(rec),
                "seed_cache_key": _txt(rec.get("cache_key")),
            }
        )
    rows.sort(key=lambda r: (r["city"], r["origin"], r["destination"], r["route"]))
    for i, r in enumerate(rows, 1):
        r["source_id"] = f"trans_{i:04d}"
    return rows


HEADERS = {
    "hotels": [
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
    "restaurants": [
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
    "attractions": [
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
    "transportation": [
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
}

# Required minimum fields per the spec, used to compute backfill coverage.
REQUIRED = {
    "hotels": ["name", "city", "latitude", "longitude", "price_per_night", "rating", "facilities"],
    "restaurants": [
        "name",
        "latitude",
        "longitude",
        "avg_price",
        "open_time",
        "close_time",
        "cuisine",
        "dietary_tags",
    ],
    "attractions": [
        "name",
        "latitude",
        "longitude",
        "ticket_price",
        "open_time",
        "close_time",
        "visit_duration_min",
        "tags",
        "walking_intensity",
    ],
    "transportation": ["origin", "destination", "mode", "duration_min", "cost_total"],
}


def _coverage(rows: list, fields: list) -> dict:
    n = len(rows) or 1
    return {f: round(sum(1 for r in rows if _txt(r.get(f))) / n, 3) for f in fields}


def cmd_build(args) -> int:
    builders = {
        "hotels": build_hotels,
        "restaurants": build_restaurants,
        "attractions": build_attractions,
        "transportation": build_transportation,
    }
    seed_names = {
        "hotels": "hotels",
        "restaurants": "restaurants",
        "attractions": "attractions",
        "transportation": "transport_routes",
    }
    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "data/grounding_seed/*.json",
        "id_scheme": {
            "hotels": "hotel_NNNN",
            "restaurants": "rest_NNNN",
            "attractions": "attr_NNNN",
            "transportation": "trans_NNNN",
        },
        "source_id_reference_format": "<table>.csv#<source_id>",
        "tables": {},
        "needs_backfill": {},
        "notes": [
            "latitude/longitude are empty everywhere — backfill via Google Places Details using place_id.",
            "open_time/close_time/visit_duration_min/tags/walking_intensity/dietary_tags/facilities/rating need enrichment before strict scheduling (steps 三/四/五).",
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for table, build in builders.items():
        seed = _read_seed(seed_names[table])
        rows = build(seed)
        _write_csv(OUT_DIR / f"{table}.csv", HEADERS[table], rows)
        cov = _coverage(rows, REQUIRED[table])
        meta["tables"][table] = {
            "file": f"{table}.csv",
            "row_count": len(rows),
            "with_place_id": sum(1 for r in rows if r.get("place_id")),
            "with_source_url": sum(1 for r in rows if r.get("source_url")),
            "required_field_coverage": cov,
        }
        meta["needs_backfill"][table] = sorted([f for f, c in cov.items() if c < 0.5])
        print(
            f"[travel_database] {table}.csv  rows={len(rows):4d}  "
            f"place_id={meta['tables'][table]['with_place_id']:4d}  "
            f"backfill={meta['needs_backfill'][table]}"
        )
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[travel_database] wrote metadata.json -> {OUT_DIR / 'metadata.json'}")
    return 0


# ====================== snapshot (audit / clean / restore / save) ======================
def _dedupe_key(rec):
    name = (
        rec.get("restaurant_name")
        or rec.get("hotel_name")
        or rec.get("poi_name")
        or rec.get("route")
        or ""
    )
    return (str(rec.get("destination_city", "")).strip().lower(), str(name).strip().lower())


def cmd_snapshot(args) -> int:
    from src.grounding import STRUCTURED_CACHE_FILES, _record_is_snapshot_worthy

    if args.save:
        SEED_DIR.mkdir(parents=True, exist_ok=True)
        for category, path in STRUCTURED_CACHE_FILES.items():
            if not path.exists():
                continue
            try:
                cache_recs = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                cache_recs = []
            cache_recs = [
                r
                for r in cache_recs
                if isinstance(r, dict) and _record_is_snapshot_worthy(category, r)
            ]
            seed_file = SEED_DIR / f"{category}.json"
            seed_recs = []
            if seed_file.exists():
                try:
                    seed_recs = json.loads(seed_file.read_text(encoding="utf-8"))
                except Exception:
                    seed_recs = []
            seed_recs = seed_recs if isinstance(seed_recs, list) else []
            merged = {}
            for r in seed_recs + cache_recs:  # cache (incl. Tavily finds) overrides on key clash
                if isinstance(r, dict):
                    merged[_dedupe_key(r)] = r
            out = list(merged.values())
            seed_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"[save] {category}: seed now {len(out)} (+{len(out) - len(seed_recs)} new from cache)"
            )
        print("Save done.")
        return 0

    if args.restore:
        for category, path in STRUCTURED_CACHE_FILES.items():
            seed_file = SEED_DIR / f"{category}.json"
            if seed_file.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(seed_file, path)
                print(f"[restore] {category}: seed -> {path.name}")
            else:
                print(f"[restore] {category}: no seed at {seed_file} (skipped)")
        print("Restore done.")

    for category, path in STRUCTURED_CACHE_FILES.items():
        if not path.exists():
            print(f"[{category}] (missing) {path}")
            continue
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[{category}] ERROR reading {path}: {exc}")
            continue
        records = records if isinstance(records, list) else []
        worthy = [
            r for r in records if isinstance(r, dict) and _record_is_snapshot_worthy(category, r)
        ]
        junk = len(records) - len(worthy)
        by_city = Counter(str(r.get("destination_city", "")) for r in worthy)
        print(
            f"[{category}] total={len(records)}  worthy={len(worthy)}  junk={junk}  cities={dict(by_city)}"
        )
        if args.city:
            c = [
                r for r in worthy if str(r.get("destination_city", "")).lower() == args.city.lower()
            ]
            print(f"    {args.city}: {len(c)} worthy records")
        if args.apply:
            path.write_text(json.dumps(worthy, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"    -> rewrote {path} with {len(worthy)} records (removed {junk})")

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to rewrite the files.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Grounding data tools (seed CSV build + cache snapshot maintenance)."
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser(
        "build", help="grounding_seed/*.json -> unified *.csv tables + metadata.json"
    )
    p_build.set_defaults(func=cmd_build)

    p_snap = sub.add_parser(
        "snapshot", help="audit / clean / restore / save the structured grounding cache"
    )
    p_snap.add_argument(
        "--apply", action="store_true", help="rewrite cache files keeping only worthy records"
    )
    p_snap.add_argument("--city", default="", help="also print a per-city breakdown for this city")
    p_snap.add_argument(
        "--restore",
        action="store_true",
        help="copy the durable seed back into the structured cache (before a run)",
    )
    p_snap.add_argument(
        "--save",
        action="store_true",
        help="merge the cache's qualified records back into the durable seed (after a run)",
    )
    p_snap.set_defaults(func=cmd_snapshot)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
