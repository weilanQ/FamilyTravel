"""Filter the raw test set and diverse-select small-family travel queries into the working dataset."""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from budget_feasibility import is_feasible, parse_budget_from_query  # noqa: E402
from dataset_utils import safe_float, safe_int, safe_text  # noqa: E402

INPUT_JSON = PROJECT_ROOT / "data" / "raw" / "triptailor" / "data" / "test.json"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "selected_travel_queries.csv"

TARGET_MIN = 100
TARGET_MAX = 200

# 可行性闸门用的最小家庭规模：连 3 口之家都装不下就丢弃。
MIN_FAMILY_SIZE = 3

# 预算合理上限（人均每天 CNY）：剔除奢华离群值，避免全家预算被乘到过高。
MAX_PER_PERSON_PER_DAY = 2500
# 每个目的地城市最多保留多少条 query：提升城市多样性，避免热门城市霸榜。
MAX_PER_CITY = 10
# 价格分层（人均每天 CNY 的档位边界）：让 低/中/高 预算均匀分配。
BUDGET_TIER_EDGES = [0, 1000, 1600, MAX_PER_PERSON_PER_DAY]
BUDGET_TIER_LABELS = ["low", "mid", "high"]


ROUTE_KEYWORDS = [
    "route",
    "itinerary",
    "schedule",
    "plan",
    "travel plan",
    "transportation",
    "transport",
    "drive",
    "driving",
    "car",
    "train",
    "bus",
    "subway",
    "metro",
    "airport",
    "station",
    "start from",
    "depart from",
    "return",
    "round trip",
    "one way",
    "not too rushed",
    "relaxed pace",
    "avoid backtracking",
]

POI_KEYWORDS = [
    "attraction",
    "attractions",
    "poi",
    "sightseeing",
    "landmark",
    "landmarks",
    "museum",
    "museums",
    "park",
    "parks",
    "zoo",
    "aquarium",
    "theme park",
    "amusement park",
    "beach",
    "lake",
    "mountain",
    "temple",
    "castle",
    "palace",
    "old town",
    "downtown",
    "shopping street",
    "night market",
    "restaurant",
    "food street",
    "local food",
]


def contains_any(text, keywords):
    text = text.lower()
    return any(k.lower() in text for k in keywords)


def count_poi_score(text):
    """用 query 文本粗估 POI 丰富程度。"""
    text = text.lower()
    keyword_count = sum(1 for kw in POI_KEYWORDS if kw.lower() in text)
    # 逗号/分号/and 越多，越可能列举了多个地点或约束。
    sep_count = sum(text.count(sep) for sep in [",", ";", " and "])
    return keyword_count + min(sep_count, 5)


def load_test_queries(path):
    """读取 test.json，抽出下游需要的结构化字段。"""
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for it in items:
        # budget 优先用结构化字段，缺失再退回从 query 文本解析。
        budget = safe_float(it.get("budget"), 0.0) or parse_budget_from_query(it.get("query"))
        rows.append(
            {
                "pid": it.get("pid", ""),
                "departure_city": safe_text(it.get("departure_city")),
                "destination_city": safe_text(it.get("destination_city")),
                "day": safe_int(it.get("day"), 0),
                "query": safe_text(it.get("query")),
                "budget": float(budget or 0.0),
            }
        )
    return pd.DataFrame(rows)


def score_query(row):
    full_text = (
        f"{safe_text(row['query'])} {safe_text(row['departure_city'])} "
        f"{safe_text(row['destination_city'])}"
    ).lower()
    day_int = safe_int(row.get("day"), 0)
    budget = safe_float(row.get("budget"), 0.0)

    score = 0
    is_good_day = False

    # 1. 行程天数：本数据集要求 3–5 天（is_good_day 同时充当天数硬闸门）
    if 3 <= day_int <= 5:
        score += 4
        is_good_day = True

    # 2. 预算约束（test.json 自带 budget，基本都有）
    has_budget = budget > 0
    if has_budget:
        score += 3

    # 3. 路线 / 交通 / 行程约束
    has_route = contains_any(full_text, ROUTE_KEYWORDS)
    if has_route:
        score += 3

    # 4. POI 丰富程度
    poi_score = count_poi_score(full_text)
    if poi_score >= 5:
        score += 3
    elif poi_score >= 3:
        score += 2
    elif poi_score >= 1:
        score += 1

    # 5. 预算可行性：budget 是人均，全家总预算 = 人均 × 人数。
    #    能装下 3 口之家（人均×3）+1 分，连 3 口都装不下记为不可行。
    budget_feasible = (
        day_int > 0
        and budget > 0
        and is_feasible(budget * MIN_FAMILY_SIZE, day_int, MIN_FAMILY_SIZE)
    )
    if budget_feasible:
        score += 1

    # 6. 预算上限：人均每天过高的奢华离群值剔除（解决"有些预算太高"）。
    per_person_per_day = budget / day_int if day_int > 0 else 0.0
    budget_reasonable = 0 < per_person_per_day <= MAX_PER_PERSON_PER_DAY

    return pd.Series(
        {
            "score": score,
            "is_good_day": is_good_day,
            "has_budget": has_budget,
            "has_route": has_route,
            "poi_score": poi_score,
            "budget_feasible": bool(budget_feasible),
            "budget_reasonable": bool(budget_reasonable),
        }
    )


def diverse_select(pool, target, max_per_city, global_city_counts=None):
    """按目的地城市轮转挑选，提升城市多样性（避免热门城市霸榜）。

    每城市内部按分数 / POI 降序；逐轮从各城市取下一条，直到达到 target 或候选耗尽。
    global_city_counts：跨多次调用共享的「城市已选计数」，用于在价格分层时仍保持
    全局城市上限（单城市跨所有档位合计不超过 max_per_city）。
    """
    if pool.empty:
        return []
    groups = {
        city: g.sort_values(by=["score", "poi_score"], ascending=False).to_dict("records")
        for city, g in pool.groupby("destination_city")
    }
    cities = sorted(groups, key=lambda c: -groups[c][0]["score"])
    ptr = {c: 0 for c in cities}
    gcounts = global_city_counts if global_city_counts is not None else {}
    selected = []
    while len(selected) < target:
        progressed = False
        for c in cities:
            if ptr[c] >= len(groups[c]) or gcounts.get(c, 0) >= max_per_city:
                continue
            selected.append(groups[c][ptr[c]])
            ptr[c] += 1
            gcounts[c] = gcounts.get(c, 0) + 1
            progressed = True
            if len(selected) >= target:
                break
        if not progressed:
            break
    return selected


def main():
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"找不到输入文件: {INPUT_JSON}")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = load_test_queries(INPUT_JSON)
    print(f"原始 query 数量: {len(df)}")

    # 去重：同一 query 文本只保留一条
    before = len(df)
    df = df.drop_duplicates(subset=["query"]).reset_index(drop=True)
    if len(df) < before:
        print(f"去重后剩余: {len(df)}（移除 {before - len(df)} 条重复 query）")

    df = pd.concat([df, df.apply(score_query, axis=1)], axis=1)

    # 预算双闸门：下限（可行性，装得下 3 口之家）+ 上限（人均每天不超 MAX_PER_PERSON_PER_DAY）。
    infeasible_count = int((~df["budget_feasible"]).sum())
    too_high_count = int((df["budget_feasible"] & ~df["budget_reasonable"]).sum())
    pool0 = df[df["budget_feasible"] & df["budget_reasonable"]].copy()
    print(
        f"预算闸门：丢弃过低 {infeasible_count} 条（装不下 {MIN_FAMILY_SIZE} 口）、"
        f"过高 {too_high_count} 条（人均每天 > {MAX_PER_PERSON_PER_DAY}），剩余 {len(pool0)} 条"
    )

    # 第一轮：严格筛选（好天数 + 预算 + 路线 + POI 丰富）
    strict_df = pool0[
        (pool0["is_good_day"])
        & (pool0["has_budget"])
        & (pool0["has_route"])
        & (pool0["poi_score"] >= 3)
    ]
    if len(strict_df) < TARGET_MIN:
        print("严格筛选不足 100 条，放宽筛选条件（好天数 + 总分≥7）。")
        pool = pool0[(pool0["is_good_day"]) & (pool0["score"] >= 7)]
    else:
        pool = strict_df

    # 价格分层 + 城市轮转：先把 query 按人均每天分到 低/中/高 三档，每档均匀取
    #   ~TARGET_MAX/3 条（档内再按城市轮转），让价格高低均匀分配且城市多样。
    pool = pool.copy()
    ppd = pool["budget"] / pool["day"].clip(lower=1)
    pool["_tier"] = pd.cut(ppd, BUDGET_TIER_EDGES, labels=BUDGET_TIER_LABELS)
    gcounts, parts = {}, []
    n_tiers = len(BUDGET_TIER_LABELS)
    base = TARGET_MAX // n_tiers
    for i, tier in enumerate(BUDGET_TIER_LABELS):
        quota = base + (TARGET_MAX - base * n_tiers if i == n_tiers - 1 else 0)
        got = diverse_select(pool[pool["_tier"] == tier], quota, MAX_PER_CITY, gcounts)
        print(f"  价格档 {tier}: 取 {len(got)}/{quota} 条")
        parts.extend(got)
    selected_df = pd.DataFrame(parts)

    # 下游构造 persona 需要的字段 + 人均预算（列名 budget_per_person 以示口径）。
    output_columns = ["departure_city", "destination_city", "day", "query", "budget"]
    selected_df = selected_df[output_columns].rename(columns={"budget": "budget_per_person"})
    selected_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"最终筛选数量: {len(selected_df)}")
    print(f"已保存到: {OUTPUT_CSV}")
    if len(selected_df) >= 1:
        b = selected_df["budget_per_person"]
        ppd = (b / selected_df["day"]).round(0)
        print(
            f"人均预算区间: {b.min():.0f}–{b.max():.0f}，中位数 {b.median():.0f} 元"
            f"（人均每天 {ppd.min():.0f}–{ppd.max():.0f}，全家总预算在 generate 阶段 = 人均 × 人数）"
        )
        vc = selected_df["destination_city"].value_counts()
        print(f"覆盖目的地城市数: {len(vc)}，单城市最多 {vc.max()} 条")
        print(f"天数分布: {selected_df['day'].value_counts().sort_index().to_dict()}")
    if len(selected_df) < TARGET_MIN:
        print("提醒：最终结果少于 100 条，符合条件的数据可能不够。")


if __name__ == "__main__":
    main()
