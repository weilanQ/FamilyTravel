from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_utils import safe_float, safe_text  # noqa: E402


def _env_float(name: str, default: float) -> float:
    return safe_float(os.getenv(name, str(default)), default)


# 与主程序成本口径对齐（src/grounding.py / src/module6 的同名常量）。
HOTEL_MIN_NIGHTLY = _env_float("HOTEL_MIN_NIGHTLY", 120.0)
ECONOMICAL_MEAL_PER_PERSON = _env_float("ECONOMICAL_MEAL_PER_PERSON", 30.0)
HOTEL_PEOPLE_PER_ROOM = max(1, int(_env_float("HOTEL_PEOPLE_PER_ROOM", 3.0)))
INTERCITY_ROUNDTRIP_DEFAULT_PER_PERSON = _env_float("INTERCITY_ROUNDTRIP_DEFAULT_PER_PERSON", 600.0)
BUDGET_PASS_TOLERANCE = _env_float("BUDGET_PASS_TOLERANCE", 1.10)
# 市内交通：构造阶段无真实线路，按家庭每天一个保守估值（城际占主导，这里只是补足）。
LOCAL_TRANSPORT_PER_DAY = _env_float("DATA_SELECT_LOCAL_TRANSPORT_PER_DAY", 50.0)


def parse_budget_from_query(query) -> float:
    """从 query 文本里抽取家庭总预算（与 baseline/baseline_common 同一套正则）。"""
    q = safe_text(query)
    m = re.search(r"(?:budget[^\d]{0,12}|预算[^\d]{0,6}|¥|￥)\s*([\d,]+(?:\.\d+)?)", q, re.IGNORECASE,)
    if m:
        return safe_float(m.group(1).replace(",", ""), 0.0)
    m = re.search(r"([\d,]{3,})\s*(?:元|cny|rmb|yuan)", q, re.IGNORECASE)
    return safe_float(m.group(1).replace(",", ""), 0.0) if m else 0.0


_BUDGET_REWRITE_RE = re.compile(
    r"((?:budget|预算)[^\d¥￥$]{0,12}[¥￥$]?\s*)([\d,]+(?:\.\d+)?)", re.IGNORECASE,
)


def rewrite_query_budget(query, new_total):
    """把 query 里的（人均）预算金额改写为给定的全家总预算。

    只替换带 budget/预算 上下文的那个金额（避免误改 query 里的餐标 ¥50–¥100 等）。
    返回 (新 query, 是否成功替换)。主程序从 query 文本解析预算，故改写后口径一致。
    """
    q = safe_text(query)
    new_str = str(int(round(safe_float(new_total, 0.0))))
    new_q, n = _BUDGET_REWRITE_RE.subn(lambda m: m.group(1) + new_str, q, count=1)
    return new_q, n > 0


def estimate_min_cost(days: int, members: int) -> float:
    """最低可达成本估计，口径同主程序 infeasible_family_ids。"""
    days = max(1, int(days))
    members = max(1, int(members))
    nights = max(1, days - 1)
    meals = max(1, 2 * days - 1)  # 到达日不算午餐、离开日不算晚餐
    rooms = math.ceil(members / HOTEL_PEOPLE_PER_ROOM)

    transport = INTERCITY_ROUNDTRIP_DEFAULT_PER_PERSON * members + LOCAL_TRANSPORT_PER_DAY * days
    hotel = HOTEL_MIN_NIGHTLY * nights * rooms
    food = ECONOMICAL_MEAL_PER_PERSON * members * meals
    return transport + hotel + food


def is_feasible(budget: float, days: int, members: int, tolerance: float = None) -> bool:
    """预算（×容差）能否装下该规模家庭的最低成本。"""
    budget = safe_float(budget, 0.0)
    if budget <= 0:
        return False
    tol = BUDGET_PASS_TOLERANCE if tolerance is None else tolerance
    return estimate_min_cost(days, members) <= budget * tol


def largest_feasible_family_size(
    budget: float, days: int, candidate_sizes=(4, 3), tolerance: float = 1.0,
) -> int:
    """返回预算装得下的最大家庭规模；都装不下返回 0。

    默认 tolerance=1.0（不吃容差），给主程序留出 10% 余量，避免真实城际价
    略高于估计时被判为不可行。
    """
    for size in sorted(candidate_sizes, reverse=True):
        if is_feasible(budget, days, size, tolerance=tolerance):
            return size
    return 0
