"""Recompute strict and budget-tolerant pass rates from evaluation reports at the current budget tolerance."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import pandas as pd  # noqa: E402
from pipeline_common import budget_pass_map, infeasible_family_ids  # noqa: E402

from src.module6_evaluation_and_explanation import is_plan_passed_requirements  # noqa: E402

_CORE = [
    ("Success", None),  # 由通过率单独填充
    ("Hard", "hard_constraint_pass_rate"),
    ("Feasibility", "feasibility_pass_rate"),
    ("AvgSat", "average_group_satisfaction"),
    ("MinSat", "minimum_member_satisfaction"),
    ("Fairness", "max_min_fairness"),
]
_SUPP = [
    ("Grounding", "restaurant_specificity_rate"),
    ("Token", "total_tokens"),
    ("Runtime", "runtime_seconds"),
]


def _metrics(df: pd.DataFrame, passed: pd.Series) -> dict:
    out = {"N": len(df), "Success": round(100.0 * passed.mean(), 1) if len(df) else 0.0}
    for label, col in _CORE:
        if col is None:
            continue
        out[label] = (
            round(pd.to_numeric(df[col], errors="coerce").mean(), 3) if col in df.columns else None
        )
    for label, col in _SUPP:
        out[label] = (
            round(pd.to_numeric(df[col], errors="coerce").mean(), 1) if col in df.columns else None
        )
    return out


def _print_block(title: str, m: dict) -> None:
    print(f"\n[{title}]  家庭数={m['N']}  通过率(Success)={m['Success']}%")
    core = "  ".join(f"{k}={m.get(k)}" for k, _ in _CORE)
    supp = "  ".join(f"{k}={m.get(k)}" for k, _ in _SUPP)
    print("  核心指标: " + core)
    print("  补充指标: " + supp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="results/module_7/evaluation_plan_level_feedback.csv")
    ap.add_argument("--plans", default="results/module_7/generated_plans_feedback_final.csv")
    a = ap.parse_args()

    eval_path = a.eval
    if not os.path.exists(eval_path):
        eval_path = "results/module_6/evaluation_plan_level.csv"
    df = pd.read_csv(eval_path)

    # Recompute budget_pass consistently at the current tolerance (the stored column can
    # mix tolerances across a partially re-run batch).
    from src.module6_evaluation_and_explanation import BUDGET_PASS_TOLERANCE

    bp_map = budget_pass_map(a.plans, BUDGET_PASS_TOLERANCE)

    def _budget_pass(r):
        try:
            return bp_map.get(int(r["family_id"]), r.get("budget_pass"))
        except (TypeError, ValueError, KeyError):
            return r.get("budget_pass")

    if bp_map and "family_id" in df.columns:
        df["budget_pass"] = df.apply(_budget_pass, axis=1)

    passed_all = df.apply(is_plan_passed_requirements, axis=1)
    df = df.assign(_passed=passed_all.values)

    infeasible = infeasible_family_ids(a.plans)

    def _is_feasible(fid) -> bool:
        try:
            return int(fid) not in infeasible
        except (TypeError, ValueError):
            return True

    if "family_id" in df.columns:
        feas_df = df[df["family_id"].apply(_is_feasible)]
    else:
        feas_df = df

    print("=" * 64)
    print(f"评估文件: {eval_path}")
    print(f"预算不可行家庭 ({len(infeasible)}个，已从可行口径剔除): {sorted(infeasible)}")
    _print_block("总体口径 (全部家庭)", _metrics(df, df["_passed"]))
    _print_block("可行口径 (剔除预算不可行家庭)", _metrics(feas_df, feas_df["_passed"]))
    print("=" * 64)


if __name__ == "__main__":
    main()
