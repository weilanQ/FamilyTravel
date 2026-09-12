"""Shared helpers for pipeline scripts: family-id loading, family sizes, ablation env, and budget/feasibility lookups."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import pandas as pd  # noqa: E402

from src.config import PERSONA_AGENTS_JSON  # noqa: E402

DEFAULT_EVAL_CSV = "results/module_7/evaluation_plan_level_feedback.csv"
DEFAULT_PLANS_CSV = "results/module_7/generated_plans_feedback_final.csv"
ABLATION_KEYS = ["ENABLE_QUALITY_CONTROL", "ENABLE_NEGOTIATION", "ENABLE_SPLIT", "ENABLE_GROUNDING"]


def num(x, default=0.0):
    """float() with a NaN/None guard."""
    try:
        v = float(x)
        return v if v == v else default
    except (TypeError, ValueError):
        return default


def _persona_agents() -> list:
    data = json.loads(Path(PERSONA_AGENTS_JSON).read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("agents", [])


def load_family_ids() -> list:
    """Sorted unique family ids present in the shared persona file."""
    ids = [
        int(a["family_id"])
        for a in _persona_agents()
        if isinstance(a, dict) and a.get("family_id") is not None
    ]
    return sorted(set(ids))


def family_sizes() -> dict:
    """family_id -> member count."""
    out: dict = {}
    try:
        for a in _persona_agents():
            fid = a.get("family_id") if isinstance(a, dict) else None
            if fid is None:
                continue
            out[int(fid)] = out.get(int(fid), 0) + 1
    except Exception:
        pass
    return out


def ablation_env() -> dict:
    """Forward the 5 ablation switches if the caller set them in the environment."""
    return {k: os.environ[k] for k in ABLATION_KEYS if k in os.environ}


def budget_pass_map(plans_csv: str, tolerance: float) -> dict:
    """family_id -> budget_pass recomputed consistently at the current tolerance.

    The stored budget_pass column can mix tolerances across a partially re-run
    batch; recomputing from total_cost/budget_limit guarantees one consistent rule.
    """
    out: dict = {}
    if not plans_csv or not os.path.exists(plans_csv):
        return out
    df = pd.read_csv(plans_csv)
    for _, r in df.iterrows():
        try:
            fid = int(r["family_id"])
        except (TypeError, ValueError, KeyError):
            continue
        bl = num(r.get("budget_limit"))
        tc = num(r.get("total_cost"))
        if bl > 0:
            out[fid] = tc <= bl * tolerance
    return out


def infeasible_family_ids(plans_csv: str) -> set:
    """真无解实例：最低可达成本仍超预算上限的家庭。

    最低成本 = 真实城际+市内交通 + 经济酒店底价×间数×晚数 + 经济餐标×人数×餐数。
    即使用最便宜的住宿和餐饮、仅保留不可避免的交通，总价仍 > 预算×容差，则任何方案
    都无法落入预算 —— 这是数据集预算本身不合理（大家庭给极低预算），应记为无解而非
    模型失败，从'可行口径'剔除、也不必重跑。
    """
    out: set = set()
    if not plans_csv or not os.path.exists(plans_csv):
        return out
    try:
        from src.grounding import ECONOMICAL_MEAL_PER_PERSON, HOTEL_MIN_NIGHTLY
        from src.module6_evaluation_and_explanation import BUDGET_PASS_TOLERANCE
    except Exception:
        HOTEL_MIN_NIGHTLY, ECONOMICAL_MEAL_PER_PERSON, BUDGET_PASS_TOLERANCE = 120.0, 30.0, 1.10
    sizes = family_sizes()
    df = pd.read_csv(plans_csv)
    for _, r in df.iterrows():
        try:
            fid = int(r["family_id"])
        except (TypeError, ValueError, KeyError):
            continue
        bl = num(r.get("budget_limit"))
        if bl <= 0:
            continue
        try:
            d = json.loads(r.get("cost_breakdown_json", "") or "{}")
        except Exception:
            d = {}
        transport = num(d.get("intercity_transport_cost")) + num(d.get("local_transport_cost"))
        days = int(num(r.get("day"), 1)) or 1
        nights = max(1, days - 1)
        rooms = int(num(d.get("hotel_rooms"), 1)) or 1
        members = sizes.get(fid, 3)
        meals = max(1, 2 * days - 1)  # arrival day skips lunch, departure day skips dinner
        min_cost = (
            transport
            + HOTEL_MIN_NIGHTLY * nights * rooms
            + ECONOMICAL_MEAL_PER_PERSON * members * meals
        )
        if min_cost > bl * BUDGET_PASS_TOLERANCE:
            out.add(fid)
    return out


def failed_family_ids(
    eval_csv: str = DEFAULT_EVAL_CSV,
    plans_csv: str = DEFAULT_PLANS_CSV,
    exclude_infeasible: bool = True,
) -> list:
    """Families whose latest evaluated plan fails is_plan_passed_requirements.

    budget_pass is recomputed consistently at the current tolerance.  When
    exclude_infeasible is True, genuinely-unsolvable families (min cost > budget)
    are dropped because re-running them cannot make them pass.
    """
    from src.module6_evaluation_and_explanation import (
        BUDGET_PASS_TOLERANCE,
        is_plan_passed_requirements,
    )

    if not eval_csv or not os.path.exists(eval_csv):
        return []
    df = pd.read_csv(eval_csv)
    if "family_id" not in df.columns:
        return []
    bp = budget_pass_map(plans_csv, BUDGET_PASS_TOLERANCE)
    if bp:
        df["budget_pass"] = df.apply(
            lambda r: bp.get(int(r["family_id"]), r.get("budget_pass"))
            if pd.notna(r.get("family_id"))
            else r.get("budget_pass"),
            axis=1,
        )
    infeasible = infeasible_family_ids(plans_csv) if exclude_infeasible else set()
    failed = []
    for _, r in df.iterrows():
        try:
            fid = int(r["family_id"])
        except (TypeError, ValueError, KeyError):
            continue
        if fid in infeasible:
            continue
        if not is_plan_passed_requirements(r):
            failed.append(fid)
    return sorted(set(failed))
