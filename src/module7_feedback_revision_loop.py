"""Module 7 - Feedback/revision loop: re-run failing families with targeted feedback or backtracking until they pass or the budget is exhausted."""
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.common import (
    DATA_DIR,
    RESULTS_DIR,
    RoleMemoryStore,
    compact_feedback_for_llm,
    compact_json_for_prompt,
    load_json_file,
    safe_float,
    safe_int,
    safe_text,
    save_json_file,
)
from src.config import MEMORY_DIR, MODULE7_TEMP_DIR
from src.module5_consensus_itinerary_generation import (
    build_family_agents,
    flatten_plan_for_csv,
    generate_consensus_itinerary_for_family,
    load_conflict_results,
    load_individual_proposals,
    load_negotiation_results,
)
from src.module6_evaluation_and_explanation import (
    build_module6_feedback_prompt,
    compute_plan_quality_score,
    evaluate_and_explain,
    find_failed_families,
    get_failed_reasons,
    get_plan_eval_row_by_family,
    is_plan_passed_requirements,
)

PERSONA_CSV = DATA_DIR / "FamilyTravel_Mini.csv"
AGENTS_JSON = RESULTS_DIR / "module_1" / "persona_agents.json"
PROPOSALS_CSV = RESULTS_DIR / "module_2" / "individual_agent_proposals.csv"
CONFLICTS_JSON = RESULTS_DIR / "module_3" / "conflict_detection_results.json"
NEGOTIATION_JSON = RESULTS_DIR / "module_4" / "negotiation_protocol_results.json"
ORIGINAL_PLANS_CSV = RESULTS_DIR / "module_5" / "generated_plans.csv"
PLAN_EVAL_CSV = RESULTS_DIR / "module_6" / "evaluation_plan_level.csv"
MEMBER_EVAL_CSV = RESULTS_DIR / "module_6" / "evaluation_member_level.csv"
FINAL_REVISED_PLANS_CSV = RESULTS_DIR / "module_7" / "generated_plans_feedback_final.csv"
REVISION_RESULTS_JSON = RESULTS_DIR / "module_7" / "feedback_revision_results.json"
REVISED_MEMBER_EVAL_CSV = RESULTS_DIR / "module_7" / "evaluation_member_level_feedback.csv"
REVISED_PLAN_EVAL_CSV = RESULTS_DIR / "module_7" / "evaluation_plan_level_feedback.csv"
REVISED_REPORT_JSON = RESULTS_DIR / "module_7" / "evaluation_report_feedback.json"
BACKTRACKING_FEEDBACK_JSON = RESULTS_DIR / "module_7" / "backtracking_to_module2_feedback.json"
# Module 2.5 as a standalone stage was removed; the proposal quality score is now embedded
# per-agent in Module 2's proposal CSV (written inline by the Module 2 quality gate, present
# even when the gate is ablated off). Read the score from there for backtracking prioritization.
PROPOSAL_EVAL_CSV = RESULTS_DIR / "module_2" / "individual_agent_proposals.csv"
TEMP_DIR = MODULE7_TEMP_DIR
NO_IMPROVEMENT_PATIENCE_ROUNDS = 1
MIN_IMPROVEMENT_DELTA = 0.1
MAX_REGENERATION_ATTEMPTS_PER_FAMILY = 4
NUM_REVISION_CANDIDATES = 2
BACKTRACK_AFTER_REVISION_ROUNDS = 2
FAST_EVAL_USE_LLM_JUDGE = False
FINAL_EVAL_USE_LLM_JUDGE = True


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in ["1", "true", "yes", "y", "on"]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


# Efficiency controls: keep Module 7 from spending many expensive generations
# on families that should be sent back to targeted backtracking instead.
MODULE7_INCREMENTAL_ROUND_EVAL = _env_bool("MODULE7_INCREMENTAL_ROUND_EVAL", True)
MODULE7_EARLY_BACKTRACK_ON_ALL_CANDIDATES_FAILED = _env_bool(
    "MODULE7_EARLY_BACKTRACK_ON_ALL_CANDIDATES_FAILED", True
)
MODULE7_MAX_ALL_FAILED_ROUNDS_PER_FAMILY = _env_int("MODULE7_MAX_ALL_FAILED_ROUNDS_PER_FAMILY", 1)
MODULE7_EARLY_BACKTRACK_ON_NO_IMPROVEMENT = _env_bool(
    "MODULE7_EARLY_BACKTRACK_ON_NO_IMPROVEMENT", True
)
MODULE7_MAX_NO_IMPROVEMENT_ROUNDS_PER_FAMILY = _env_int(
    "MODULE7_MAX_NO_IMPROVEMENT_ROUNDS_PER_FAMILY", 1
)


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(key): make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(item) for item in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(item) for item in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        if np.isnan(value):
            return None
        return value
    if isinstance(obj, np.bool_):
        return bool(obj)
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")
    return pd.read_csv(path)


def save_plans(plans_df: pd.DataFrame, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "family_id" in plans_df.columns:
        plans_df = plans_df.sort_values(by=["family_id"])
    plans_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def get_current_plan_row(plans_df: pd.DataFrame, family_id: int) -> Dict[str, Any]:
    if plans_df.empty:
        return {"family_id": family_id, "status": "missing"}
    if "family_id" not in plans_df.columns:
        return {"family_id": family_id, "status": "missing"}
    family_df = plans_df[plans_df["family_id"].astype(int) == int(family_id)].copy()
    if family_df.empty:
        return {"family_id": family_id, "status": "missing"}
    return family_df.iloc[0].to_dict()


def replace_family_plan(plans_df: pd.DataFrame, new_plan_row: Dict[str, Any]) -> pd.DataFrame:
    family_id = safe_int(new_plan_row.get("family_id"), 0)
    if "family_id" not in plans_df.columns:
        raise ValueError("plans_df 缺少 family_id 字段，无法替换 plan。")
    plans_df = plans_df[plans_df["family_id"].astype(int) != int(family_id)].copy()
    new_df = pd.DataFrame([new_plan_row])
    merged_df = pd.concat([plans_df, new_df], ignore_index=True, sort=False)
    if "family_id" in merged_df.columns:
        merged_df = merged_df.sort_values(by=["family_id"])
    return merged_df


def count_failed_reasons_from_eval(eval_row: Dict[str, Any]) -> int:
    if not eval_row:
        return 999
    try:
        reasons = get_failed_reasons(eval_row)
    except Exception:
        return 999
    if not isinstance(reasons, list):
        return 999
    return len(reasons)


def get_tiebreak_values(eval_row: Dict[str, Any]) -> Dict[str, float]:
    if not eval_row:
        return {
            "failed_count": 999,
            "hard_constraint_pass_rate": 0.0,
            "minimum_member_satisfaction": 0.0,
            "acceptance_vote_ratio": 0.0,
            "feasibility_pass_rate": 0.0,
            "rationality_pass_rate": 0.0,
        }
    return {
        "failed_count": count_failed_reasons_from_eval(eval_row),
        "hard_constraint_pass_rate": safe_float(eval_row.get("hard_constraint_pass_rate"), 0.0),
        "minimum_member_satisfaction": safe_float(eval_row.get("minimum_member_satisfaction"), 0.0),
        "acceptance_vote_ratio": safe_float(eval_row.get("acceptance_vote_ratio"), 0.0),
        "feasibility_pass_rate": safe_float(eval_row.get("feasibility_pass_rate"), 0.0),
        "rationality_pass_rate": safe_float(eval_row.get("rationality_pass_rate"), 0.0),
    }


def is_tiebreak_improved(
    old_eval_row: Dict[str, Any], new_eval_row: Dict[str, Any], min_improvement_delta: float = 0.01
) -> bool:
    old_values = get_tiebreak_values(old_eval_row)
    new_values = get_tiebreak_values(new_eval_row)
    if new_values["failed_count"] < old_values["failed_count"]:
        return True
    if (
        new_values["hard_constraint_pass_rate"]
        > old_values["hard_constraint_pass_rate"] + min_improvement_delta
    ):
        return True
    if (
        new_values["minimum_member_satisfaction"]
        > old_values["minimum_member_satisfaction"] + min_improvement_delta
    ):
        return True
    if (
        new_values["acceptance_vote_ratio"]
        > old_values["acceptance_vote_ratio"] + min_improvement_delta
    ):
        return True
    if (
        new_values["feasibility_pass_rate"]
        > old_values["feasibility_pass_rate"] + min_improvement_delta
    ):
        return True
    if (
        new_values["rationality_pass_rate"]
        > old_values["rationality_pass_rate"] + min_improvement_delta
    ):
        return True
    return False


def make_feedback_debug_info(
    feedback_prompt: str,
    family_id: int,
    revision_round: int,
    candidate_index: int,
    failure_reasons: List[str],
) -> Dict[str, Any]:
    text = safe_text(feedback_prompt)
    return {
        "family_id": int(family_id),
        "revision_round": int(revision_round),
        "candidate_index": int(candidate_index),
        "feedback_length": len(text),
        "feedback_hash": hashlib.md5(text.encode("utf-8")).hexdigest()[:12],
        "failure_reasons_used": failure_reasons,
        "feedback_preview": text[:600],
    }


def compact_feedback_for_module5(
    feedback_prompt: str, failure_reasons: List[str], max_chars: int = 2400
) -> str:
    """Compact Module 6 feedback without losing unresolved conflict evidence.

    The normal feedback compressor may keep generic failure lines and drop the exact
    unresolved_conflicts / split_activity evidence.  Module 5 revision needs those
    exact terms to repair the next candidate, so we preserve matching lines plus a
    small local context window before final compaction.
    """
    reason_text = "\n".join([safe_text(x) for x in (failure_reasons or [])[:8] if safe_text(x)])
    raw_text = safe_text(feedback_prompt)
    priority_keywords = [
        "unresolved_conflicts",
        "resolved_conflicts",
        "constraint_satisfiability",
        "unresolved_conflict_penalty",
        "unresolved_direct_conflict_count",
        "direct_must_include_must_avoid_conflict",
        "must_include_vs_must_avoid_term_conflict",
        "preference_vs_must_avoid_term_conflict",
        "hard_constraint_feasibility_conflict",
        "Unsatisfiable Constraints",
        "Split-Satisfiable Constraints",
        "split_activity_cost",
        "weak_or_incomplete_split_activities",
        "Current unresolved conflict feedback",
        "未解决冲突",
        "直接冲突",
        "分流活动成本",
        "must_include",
        "must_avoid",
        "split_activities",
        "participating_roles",
        "non_participating_roles",
    ]
    lines = [safe_text(line) for line in raw_text.splitlines() if safe_text(line)]
    preserved_lines = []
    for index, line in enumerate(lines):
        lower_line = line.lower()
        if any(keyword.lower() in lower_line for keyword in priority_keywords):
            for context_line in lines[max(0, index - 1) : min(len(lines), index + 3)]:
                if context_line:
                    preserved_lines.append(context_line)
    preserved_lines = list(dict.fromkeys(preserved_lines))
    preserved_text = "\n".join(preserved_lines)
    feedback_text = compact_feedback_for_llm(raw_text, max_chars=max_chars)
    parts = []
    if reason_text:
        parts.append("Current failed reasons:\n" + reason_text)
    if preserved_text:
        parts.append(
            "Preserved unresolved/split conflict feedback for Module 5 revision:\n" + preserved_text
        )
    parts.append(feedback_text)
    return compact_feedback_for_llm("\n\n".join(parts), max_chars=max_chars)


# -----------------------------------------------------------------------------
# Search-inspired candidate diversification for better backtracking repairs
# -----------------------------------------------------------------------------
# Instead of asking Module 5 to generate two nearly identical revisions, each
# candidate follows a distinct repair policy.  This is a lightweight version of
# Tree-of-Thought / LATS style search: expand several different repair paths,
# evaluate them with Module 6, and keep the best one.  The strategies are pure
# prompt/control logic, so they do not require any new dependency.

REVISION_STRATEGY_LIBRARY = [
    {
        "strategy_id": "split_conflict_first",
        "strategy_name": "Role-explicit split conflict repair",
        "when_to_use": "unresolved conflicts, must_include vs must_avoid conflicts, invalid split activities",
        "instruction": "First resolve each direct hard conflict with one role-explicit split activity. Name participating_roles, non_participating_roles, concrete alternative, rejoin point, and reason. If a conflict cannot be resolved, keep it in unresolved_conflicts and downgrade consensus_status.",
    },
    {
        "strategy_id": "hard_constraint_first",
        "strategy_name": "Hard-constraint feasibility repair",
        "when_to_use": "walking distance, return time, budget, must_include, must_avoid, hotel/transport completeness",
        "instruction": "Lock the strictest max walking distance, earliest latest_return_time, must_include/must_avoid terms, and budget first. Reduce daily POIs, move tiring activities earlier, add rest blocks, and keep route/hotel/transport concrete.",
    },
    {
        "strategy_id": "lowest_member_first",
        "strategy_name": "Lowest-satisfaction member repair",
        "when_to_use": "low minimum satisfaction, rejected members, obvious unfairness, low acceptance vote ratio",
        "instruction": "Start from the previous lowest-satisfaction/rejected member. Add at least one concrete activity or comfort adjustment for that member without breaking any hard constraint. Explain why the member should now accept.",
    },
    {
        "strategy_id": "minimal_change_repair",
        "strategy_name": "Minimal-change repair",
        "when_to_use": "high-scoring plan with only one or two remaining issues",
        "instruction": "Keep all already-good itinerary parts. Only change the failing day, failing activity, or failing constraint. Avoid large rewrites that can introduce new errors.",
    },
    {
        "strategy_id": "grounded_feasibility_repair",
        "strategy_name": "Grounded feasibility repair",
        "when_to_use": "hotel, restaurant, route, ticket, cost, online grounding, route zero anomaly",
        "instruction": "Improve grounding_requirements, cost_breakdown, hotel area, restaurant variety, transportation routes, ticket requirements, and budget note. Make daily route origins/destinations searchable and keep costs within budget.",
    },
]


def _text_blob(*values: Any) -> str:
    return " ".join([safe_text(v).lower() for v in values if safe_text(v)])


def choose_primary_strategy(
    failure_reasons: List[str], old_eval_row: Dict[str, Any], original_plan: Dict[str, Any]
) -> str:
    blob = _text_blob(" ".join(failure_reasons or []), old_eval_row, original_plan)
    if any(
        k in blob
        for k in [
            "unresolved",
            "must_include_vs_must_avoid",
            "direct_must_include",
            "split_activity",
            "分流",
            "未解决冲突",
        ]
    ):
        return "split_conflict_first"
    if any(
        k in blob
        for k in [
            "hard_constraint",
            "walking",
            "return_time",
            "budget",
            "must_include",
            "must_avoid",
            "硬约束",
            "步行",
            "返回",
        ]
    ):
        return "hard_constraint_first"
    if any(
        k in blob
        for k in [
            "minimum_member_satisfaction",
            "lowest",
            "reject",
            "acceptance",
            "unfair",
            "最低",
            "拒绝",
            "不公平",
        ]
    ):
        return "lowest_member_first"
    if any(
        k in blob
        for k in [
            "grounding",
            "hotel",
            "restaurant",
            "transport",
            "route",
            "ticket",
            "cost",
            "路线",
            "酒店",
            "餐厅",
        ]
    ):
        return "grounded_feasibility_repair"
    return "minimal_change_repair"


def get_strategy_by_id(strategy_id: str) -> Dict[str, Any]:
    for item in REVISION_STRATEGY_LIBRARY:
        if item["strategy_id"] == strategy_id:
            return dict(item)
    return dict(REVISION_STRATEGY_LIBRARY[0])


def select_revision_strategy(
    candidate_index: int,
    revision_round: int,
    failure_reasons: List[str],
    old_eval_row: Dict[str, Any],
    original_plan: Dict[str, Any],
) -> Dict[str, Any]:
    primary_id = choose_primary_strategy(
        failure_reasons=failure_reasons, old_eval_row=old_eval_row, original_plan=original_plan
    )
    ids = [item["strategy_id"] for item in REVISION_STRATEGY_LIBRARY]
    ordered_ids = [primary_id] + [x for x in ids if x != primary_id]
    # Rotate by candidate index and revision round so repeated rounds explore new paths.
    offset = (max(candidate_index, 1) - 1 + max(revision_round, 1) - 1) % len(ordered_ids)
    strategy = get_strategy_by_id(ordered_ids[offset])
    strategy["candidate_index"] = int(candidate_index)
    strategy["revision_round"] = int(revision_round)
    strategy["primary_strategy_id"] = primary_id
    strategy["strategy_rank_in_round"] = offset + 1
    return strategy


TARGETED_LOW_SATISFACTION_THRESHOLD = 82.0


def build_targeted_repair_directives(eval_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Map detected failure types to local repair directives (targeted backtracking).

    Each directive repairs only the affected day/meal/activity and never globally regenerates
    or deletes a member's core/must_include preference. Only directives whose failure is actually
    present on this plan are returned, so Module 5 repairs exactly what is broken. Reads the
    failure signals produced by Module 6 (hard constraints, member satisfaction, restaurant/
    activity specificity failures, transport chain coverage, budget).
    """

    def _b(value: Any) -> bool:
        return safe_text(value).lower() in ["true", "1", "yes", "y"]

    directives: List[Dict[str, Any]] = []

    hard_pass = safe_float(eval_row.get("hard_constraint_pass_rate"), 1.0)
    if hard_pass < 1.0:
        directives.append(
            {
                "failure_type": "hard_constraint_violation",
                "evidence": f"hard_constraint_pass_rate={hard_pass}",
                "action": "Rewrite only the affected day(s)/activity that cause the missing must_include, the violated must_avoid, or the too-late return time. Do not regenerate the whole plan.",
            }
        )

    min_sat = safe_float(eval_row.get("minimum_member_satisfaction"), 100.0)
    low_cnt = safe_int(eval_row.get("low_satisfaction_member_count"), 0)
    if min_sat < TARGETED_LOW_SATISFACTION_THRESHOLD or low_cnt > 0:
        directives.append(
            {
                "failure_type": "low_member_satisfaction",
                "evidence": f"minimum_member_satisfaction={min_sat}, low_satisfaction_member_count={low_cnt}",
                "action": "Add the lowest-satisfaction member's core preference, or lower the weight of other members' high-conflict activities. Keep all hard constraints satisfied.",
            }
        )

    rest_fail = safe_int(eval_row.get("restaurant_failure_count"), 0)
    rest_rate = safe_float(eval_row.get("restaurant_specificity_rate"), 1.0)
    if rest_fail > 0 or rest_rate < 0.7:
        directives.append(
            {
                "failure_type": "restaurant_missing",
                "evidence": f"restaurant_failure_count={rest_fail}, restaurant_specificity_rate={rest_rate}",
                "action": "Search restaurant candidates only for the failing meals and replace them with concrete named restaurants (+source). Keep meals that are already specific.",
            }
        )

    act_fail = safe_int(eval_row.get("activity_specificity_failure_count"), 0)
    if act_fail > 0:
        directives.append(
            {
                "failure_type": "vague_activity",
                "evidence": f"activity_specificity_failure_count={act_fail}",
                "action": "Replace each vague label (自由活动/当地体验/general) with a concrete POI or actionable activity and attach a source id. Keep already-concrete activities unchanged.",
            }
        )

    chain_pass = _b(eval_row.get("post_transport_chain_pass"))
    chain_cov = safe_float(eval_row.get("daily_route_chain_coverage"), 1.0)
    if (not chain_pass) or chain_cov < 0.8:
        directives.append(
            {
                "failure_type": "incomplete_transport_chain",
                "evidence": f"post_transport_chain_pass={chain_pass}, daily_route_chain_coverage={chain_cov}",
                "action": "For the incomplete days only, fill the hotel->POI, POI->restaurant, and restaurant->hotel legs with concrete transport. Do not touch complete days.",
            }
        )

    if not _b(eval_row.get("budget_pass")):
        directives.append(
            {
                "failure_type": "budget_overrun",
                "evidence": "budget_pass=False",
                "action": "Replace hotel, restaurants, and high-price activities first to fit the budget. Do NOT delete any core / must_include preference.",
            }
        )

    return directives


def build_candidate_strategy_prompt(
    strategy: Dict[str, Any],
    family_id: int,
    old_eval_row: Dict[str, Any],
    original_plan: Dict[str, Any],
) -> str:
    metrics = {
        "old_quality_score": compute_plan_quality_score(old_eval_row),
        "hard_constraint_pass_rate": safe_float(
            old_eval_row.get("hard_constraint_pass_rate"), None
        ),
        "feasibility_pass_rate": safe_float(old_eval_row.get("feasibility_pass_rate"), None),
        "rationality_pass_rate": safe_float(old_eval_row.get("rationality_pass_rate"), None),
        "minimum_member_satisfaction": safe_float(
            old_eval_row.get("minimum_member_satisfaction"), None
        ),
        "acceptance_vote_ratio": safe_float(old_eval_row.get("acceptance_vote_ratio"), None),
        "unresolved_conflict_penalty": safe_float(
            old_eval_row.get("unresolved_conflict_penalty"), 0
        ),
        "split_activity_cost": safe_float(old_eval_row.get("split_activity_cost"), 0),
    }
    payload = {
        "family_id": int(family_id),
        "CANDIDATE_REPAIR_STRATEGY": strategy,
        "TARGETED_REPAIR_DIRECTIVES": build_targeted_repair_directives(old_eval_row),
        "current_metrics_before_revision": metrics,
        "must_differ_from_other_candidates": True,
        "selection_goal": "Generate a candidate that Module 6 can score higher than the current plan by fixing the stated failure reasons, not by hiding them.",
        "do_not_do": [
            "Do not remove unresolved_conflicts unless a concrete resolution exists.",
            "Do not claim Full Consensus if direct hard conflicts remain.",
            "Do not add vague split activity wording such as 部分成员/其他成员/有人.",
            "Do not increase walking distance, return time, or cost to improve only average satisfaction.",
        ],
    }
    return "CANDIDATE_REPAIR_STRATEGY_BLOCK:\n" + compact_json_for_prompt(payload, max_chars=1400)


def build_backtracking_failure_profile(
    plan_eval_row: Dict[str, Any],
    lowest_member: Dict[str, Any],
    rejected_members: List[Dict[str, Any]],
    hard_failed_members: List[Dict[str, Any]],
    reasons: List[str],
) -> Dict[str, Any]:
    blob = _text_blob(" ".join(reasons or []), plan_eval_row)
    root_causes = []
    if any(
        k in blob
        for k in [
            "unresolved",
            "must_include_vs_must_avoid",
            "direct_must_include",
            "split_activity",
            "未解决",
            "分流",
        ]
    ):
        root_causes.append("unresolved_or_weak_split_conflicts")
    if any(
        k in blob
        for k in [
            "hard_constraint",
            "must_include",
            "must_avoid",
            "walking",
            "return_time",
            "budget",
            "硬约束",
            "步行",
            "返回",
        ]
    ):
        root_causes.append("hard_constraint_or_feasibility_failure")
    if rejected_members or any(
        k in blob
        for k in [
            "acceptance",
            "reject",
            "minimum_member_satisfaction",
            "lowest",
            "unfair",
            "拒绝",
            "最低",
        ]
    ):
        root_causes.append("low_acceptance_or_unfairness")
    if any(
        k in blob
        for k in [
            "grounding",
            "hotel",
            "restaurant",
            "transport",
            "route",
            "ticket",
            "cost",
            "路线",
            "酒店",
            "餐厅",
        ]
    ):
        root_causes.append("weak_real_world_grounding")
    if not root_causes:
        root_causes.append("general_quality_gap")
    if "unresolved_or_weak_split_conflicts" in root_causes:
        recommended = "split_conflict_first"
    elif "hard_constraint_or_feasibility_failure" in root_causes:
        recommended = "hard_constraint_first"
    elif "low_acceptance_or_unfairness" in root_causes:
        recommended = "lowest_member_first"
    elif "weak_real_world_grounding" in root_causes:
        recommended = "grounded_feasibility_repair"
    else:
        recommended = "minimal_change_repair"
    return {
        "root_causes": root_causes,
        "recommended_regeneration_strategy": recommended,
        "quality_score": compute_plan_quality_score(plan_eval_row),
        "hard_constraint_pass_rate": safe_float(
            plan_eval_row.get("hard_constraint_pass_rate"), None
        ),
        "feasibility_pass_rate": safe_float(plan_eval_row.get("feasibility_pass_rate"), None),
        "rationality_pass_rate": safe_float(plan_eval_row.get("rationality_pass_rate"), None),
        "average_group_satisfaction": safe_float(
            plan_eval_row.get("average_group_satisfaction"), None
        ),
        "minimum_member_satisfaction": safe_float(
            plan_eval_row.get("minimum_member_satisfaction"), None
        ),
        "acceptance_vote_ratio": safe_float(plan_eval_row.get("acceptance_vote_ratio"), None),
        "unresolved_conflict_count": safe_float(plan_eval_row.get("unresolved_conflict_count"), 0),
        "unresolved_direct_conflict_count": safe_float(
            plan_eval_row.get("unresolved_direct_conflict_count"), 0
        ),
        "unresolved_conflict_penalty": safe_float(
            plan_eval_row.get("unresolved_conflict_penalty"), 0
        ),
        "split_activity_cost": safe_float(plan_eval_row.get("split_activity_cost"), 0),
        "lowest_satisfaction_member": lowest_member,
        "rejected_member_count": len(rejected_members),
        "hard_failed_member_count": len(hard_failed_members),
        "failed_reasons": (reasons or [])[:8],
    }


def evaluate_and_explain_fast(
    persona_csv_path: Path,
    agents_json_path: Path,
    generated_plans_csv_path: Path,
    member_output_csv_path: Path,
    plan_output_csv_path: Path,
    report_output_json_path: Path,
    use_llm_judge: bool,
) -> Dict[str, Any]:
    kwargs = {
        "persona_csv_path": persona_csv_path,
        "agents_json_path": agents_json_path,
        "generated_plans_csv_path": generated_plans_csv_path,
        "member_output_csv_path": member_output_csv_path,
        "plan_output_csv_path": plan_output_csv_path,
        "report_output_json_path": report_output_json_path,
    }
    signature = inspect.signature(evaluate_and_explain)
    if "use_llm_judge" in signature.parameters:
        kwargs["use_llm_judge"] = use_llm_judge
    return evaluate_and_explain(**kwargs)


def evaluate_candidate_row(
    candidate_row: Dict[str, Any],
    persona_csv_path: Path,
    agents_json_path: Path,
    family_id: int,
    revision_round: int,
    candidate_index: int,
) -> Dict[str, Any]:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_plan_csv = (
        TEMP_DIR / f"candidate_plan_r{revision_round}_f{family_id}_c{candidate_index}.csv"
    )
    temp_member_eval_csv = (
        TEMP_DIR / f"candidate_member_eval_r{revision_round}_f{family_id}_c{candidate_index}.csv"
    )
    temp_plan_eval_csv = (
        TEMP_DIR / f"candidate_plan_eval_r{revision_round}_f{family_id}_c{candidate_index}.csv"
    )
    temp_report_json = (
        TEMP_DIR / f"candidate_report_r{revision_round}_f{family_id}_c{candidate_index}.json"
    )
    pd.DataFrame([candidate_row]).to_csv(temp_plan_csv, index=False, encoding="utf-8-sig")
    eval_result = evaluate_and_explain_fast(
        persona_csv_path=persona_csv_path,
        agents_json_path=agents_json_path,
        generated_plans_csv_path=temp_plan_csv,
        member_output_csv_path=temp_member_eval_csv,
        plan_output_csv_path=temp_plan_eval_csv,
        report_output_json_path=temp_report_json,
        use_llm_judge=FAST_EVAL_USE_LLM_JUDGE,
    )
    plan_eval_df = eval_result["plan_eval_df"]
    candidate_eval_row = get_plan_eval_row_by_family(plan_eval_df=plan_eval_df, family_id=family_id)
    return candidate_eval_row


def _normalize_family_id_set(family_ids: List[int]) -> set:
    output = set()
    for value in family_ids or []:
        try:
            fid = int(value)
            if fid > 0:
                output.add(fid)
        except Exception:
            continue
    return output


def merge_eval_rows_by_family_id(
    current_df: pd.DataFrame, updated_df: pd.DataFrame, family_ids: List[int]
) -> pd.DataFrame:
    """Replace only target family rows in an existing evaluation dataframe.

    Module 7 usually changes only a few failed families. Re-evaluating all 50/200
    plans after every round is expensive and was the main repair-efficiency bottleneck.
    This function keeps old evaluation rows for unchanged families and merges in the
    newly evaluated target rows.
    """
    target_ids = _normalize_family_id_set(family_ids)
    if not target_ids:
        return current_df.copy()
    if updated_df is None or updated_df.empty:
        return current_df.copy()
    if current_df is None or current_df.empty or "family_id" not in current_df.columns:
        merged = updated_df.copy()
    elif "family_id" not in updated_df.columns:
        merged = current_df.copy()
    else:
        base = current_df[~current_df["family_id"].astype(int).isin(target_ids)].copy()
        patch = updated_df[updated_df["family_id"].astype(int).isin(target_ids)].copy()
        merged = pd.concat([base, patch], ignore_index=True, sort=False)
    if "family_id" in merged.columns:
        merged = merged.sort_values(by=["family_id"]).reset_index(drop=True)
    return merged


def save_eval_dfs(
    plan_eval_df: pd.DataFrame,
    member_eval_df: pd.DataFrame,
    plan_output_csv_path: Path,
    member_output_csv_path: Path,
):
    plan_output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    member_output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if plan_eval_df is not None:
        plan_eval_df.to_csv(plan_output_csv_path, index=False, encoding="utf-8-sig")
    if member_eval_df is not None:
        member_eval_df.to_csv(member_output_csv_path, index=False, encoding="utf-8-sig")


def evaluate_changed_families_and_merge(
    current_plans_df: pd.DataFrame,
    changed_family_ids: List[int],
    current_plan_eval_df: pd.DataFrame,
    current_member_eval_df: pd.DataFrame,
    persona_csv_path: Path,
    agents_json_path: Path,
    member_output_csv_path: Path,
    plan_output_csv_path: Path,
    report_output_json_path: Path,
    revision_round: int,
) -> Dict[str, Any]:
    """Evaluate only changed families, then merge into the full evaluation tables."""
    target_ids = sorted(_normalize_family_id_set(changed_family_ids))
    if not target_ids:
        save_eval_dfs(
            current_plan_eval_df,
            current_member_eval_df,
            plan_output_csv_path,
            member_output_csv_path,
        )
        save_json_file(
            data={
                "status": "incremental_eval_skipped_no_changed_family",
                "revision_round": revision_round,
            },
            path=report_output_json_path,
        )
        return {
            "plan_eval_df": current_plan_eval_df,
            "member_eval_df": current_member_eval_df,
            "changed_family_ids": [],
        }
    if current_plans_df.empty or "family_id" not in current_plans_df.columns:
        raise ValueError("current_plans_df 缺少 family_id，无法进行增量评价。")
    subset_plans_df = current_plans_df[
        current_plans_df["family_id"].astype(int).isin(target_ids)
    ].copy()
    if subset_plans_df.empty:
        save_eval_dfs(
            current_plan_eval_df,
            current_member_eval_df,
            plan_output_csv_path,
            member_output_csv_path,
        )
        save_json_file(
            data={
                "status": "incremental_eval_skipped_missing_changed_plan",
                "revision_round": revision_round,
                "changed_family_ids": target_ids,
            },
            path=report_output_json_path,
        )
        return {
            "plan_eval_df": current_plan_eval_df,
            "member_eval_df": current_member_eval_df,
            "changed_family_ids": target_ids,
        }
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_plan_csv = TEMP_DIR / f"round_{revision_round}_changed_plans.csv"
    temp_member_eval_csv = TEMP_DIR / f"round_{revision_round}_changed_member_eval.csv"
    temp_plan_eval_csv = TEMP_DIR / f"round_{revision_round}_changed_plan_eval.csv"
    temp_report_json = TEMP_DIR / f"round_{revision_round}_changed_report.json"
    subset_plans_df.to_csv(temp_plan_csv, index=False, encoding="utf-8-sig")
    print(f"[Module 7] 增量评价：本轮只重新评价 changed family_id={target_ids}，避免全量重评。")
    eval_result = evaluate_and_explain_fast(
        persona_csv_path=persona_csv_path,
        agents_json_path=agents_json_path,
        generated_plans_csv_path=temp_plan_csv,
        member_output_csv_path=temp_member_eval_csv,
        plan_output_csv_path=temp_plan_eval_csv,
        report_output_json_path=temp_report_json,
        use_llm_judge=FAST_EVAL_USE_LLM_JUDGE,
    )
    updated_plan_eval_df = eval_result.get("plan_eval_df", pd.DataFrame())
    updated_member_eval_df = eval_result.get("member_eval_df", pd.DataFrame())
    merged_plan_eval_df = merge_eval_rows_by_family_id(
        current_plan_eval_df, updated_plan_eval_df, target_ids
    )
    merged_member_eval_df = merge_eval_rows_by_family_id(
        current_member_eval_df, updated_member_eval_df, target_ids
    )
    save_eval_dfs(
        merged_plan_eval_df, merged_member_eval_df, plan_output_csv_path, member_output_csv_path
    )
    save_json_file(
        data={
            "status": "incremental_eval_merged",
            "revision_round": revision_round,
            "changed_family_ids": target_ids,
            "full_plan_eval_rows_after_merge": 0
            if merged_plan_eval_df is None
            else len(merged_plan_eval_df),
            "full_member_eval_rows_after_merge": 0
            if merged_member_eval_df is None
            else len(merged_member_eval_df),
            "temp_plan_eval_csv": str(temp_plan_eval_csv),
            "temp_member_eval_csv": str(temp_member_eval_csv),
        },
        path=report_output_json_path,
    )
    return {
        "plan_eval_df": merged_plan_eval_df,
        "member_eval_df": merged_member_eval_df,
        "changed_family_ids": target_ids,
    }


def build_best_plan_issue_report(
    plan_eval_df: pd.DataFrame,
    member_eval_df: pd.DataFrame,
    remaining_failed_families: Dict[int, List[str]],
) -> Dict[str, Any]:
    issue_report = {
        "failed_family_count": len(remaining_failed_families),
        "families": [],
        "summary": "",
    }
    if not remaining_failed_families:
        issue_report["summary"] = "所有 family 的最终方案均已达到通过标准。"
        return issue_report
    for family_id_raw, reasons in remaining_failed_families.items():
        family_id = safe_int(family_id_raw, 0)
        plan_row = get_plan_eval_row_by_family(plan_eval_df=plan_eval_df, family_id=family_id)
        family_member_df = pd.DataFrame()
        if not member_eval_df.empty and "family_id" in member_eval_df.columns:
            family_member_df = member_eval_df[
                member_eval_df["family_id"].astype(int) == int(family_id)
            ].copy()
        lowest_member = {}
        rejected_members = []
        hard_failed_members = []
        if not family_member_df.empty:
            family_member_df["_score"] = family_member_df["individual_satisfaction_score"].apply(
                lambda x: safe_float(x, 0)
            )
            lowest_row = family_member_df.sort_values(by="_score", ascending=True).iloc[0]
            lowest_member = {
                "member_id": safe_int(lowest_row.get("member_id"), 0),
                "agent_id": safe_text(lowest_row.get("agent_id", "")),
                "role": safe_text(lowest_row.get("role", "")),
                "individual_satisfaction_score": safe_float(
                    lowest_row.get("individual_satisfaction_score"), 0
                ),
                "accepted_vote": safe_text(lowest_row.get("accepted_vote", "")),
                "hard_constraint_pass": safe_text(lowest_row.get("hard_constraint_pass", "")),
                "hard_constraint_failed_reasons": safe_text(
                    lowest_row.get("hard_constraint_failed_reasons", "")
                ),
                "member_explanation": safe_text(lowest_row.get("member_explanation", "")),
            }
            for _, member_row in family_member_df.iterrows():
                accepted_text = safe_text(member_row.get("accepted_vote", "")).lower()
                hard_text = safe_text(member_row.get("hard_constraint_pass", "")).lower()
                if accepted_text not in ["true", "1", "yes"]:
                    rejected_members.append(
                        {
                            "member_id": safe_int(member_row.get("member_id"), 0),
                            "agent_id": safe_text(member_row.get("agent_id", "")),
                            "role": safe_text(member_row.get("role", "")),
                            "individual_satisfaction_score": safe_float(
                                member_row.get("individual_satisfaction_score"), 0
                            ),
                            "member_explanation": safe_text(
                                member_row.get("member_explanation", "")
                            ),
                        }
                    )
                if hard_text not in ["true", "1", "yes"]:
                    hard_failed_members.append(
                        {
                            "member_id": safe_int(member_row.get("member_id"), 0),
                            "agent_id": safe_text(member_row.get("agent_id", "")),
                            "role": safe_text(member_row.get("role", "")),
                            "hard_constraint_failed_reasons": safe_text(
                                member_row.get("hard_constraint_failed_reasons", "")
                            ),
                        }
                    )
        issue_report["families"].append(
            {
                "family_id": family_id,
                "best_plan_score": compute_plan_quality_score(plan_row),
                "failed_reasons": reasons,
                "hard_constraint_pass_rate": safe_float(
                    plan_row.get("hard_constraint_pass_rate"), 0
                ),
                "feasibility_pass_rate": safe_float(plan_row.get("feasibility_pass_rate"), 0),
                "rationality_pass_rate": safe_float(plan_row.get("rationality_pass_rate"), 0),
                "average_group_satisfaction": safe_float(
                    plan_row.get("average_group_satisfaction"), 0
                ),
                "minimum_member_satisfaction": safe_float(
                    plan_row.get("minimum_member_satisfaction"), 0
                ),
                "acceptance_vote_ratio": safe_float(plan_row.get("acceptance_vote_ratio"), 0),
                "obvious_unfairness": safe_text(plan_row.get("obvious_unfairness", "")),
                "lowest_member": lowest_member,
                "rejected_members": rejected_members,
                "hard_failed_members": hard_failed_members,
            }
        )
    family_ids = [item["family_id"] for item in issue_report["families"]]
    issue_report[
        "summary"
    ] = f"仍有 {len(family_ids)} 个 family 未完全达标，family_id={family_ids}。系统已保留每个 family 当前综合得分最高的方案，并在 families 字段中列出未通过指标、最低满意度成员、拒绝方案成员和硬约束失败成员。"
    return issue_report


def format_issue_diagnosis_text(family_issue: Dict[str, Any]) -> str:
    family_id = family_issue.get("family_id")
    lines = []
    lines.append("【问题诊断】")
    lines.append(f"family_id={family_id} 当前保留的是综合得分最高的方案，但仍存在以下未完全达标问题。")
    best_score = family_issue.get("best_plan_score")
    if best_score is not None:
        lines.append(f"综合质量分：{best_score}")
    failed_reasons = family_issue.get("failed_reasons", [])
    if failed_reasons:
        lines.append("未通过指标：")
        for reason in failed_reasons:
            lines.append(f"- {reason}")
    metric_items = [
        ("hard_constraint_pass_rate", "硬约束通过率"),
        ("feasibility_pass_rate", "可行性通过率"),
        ("rationality_pass_rate", "行程合理性通过率"),
        ("average_group_satisfaction", "群体平均满意度"),
        ("minimum_member_satisfaction", "最低成员满意度"),
        ("acceptance_vote_ratio", "成员接受比例"),
        ("obvious_unfairness", "是否明显不公平"),
    ]
    lines.append("关键评分：")
    for key, name in metric_items:
        if key in family_issue:
            lines.append(f"- {name}: {family_issue.get(key)}")
    lowest_member = family_issue.get("lowest_member", {})
    if lowest_member:
        lines.append("最低满意度成员：")
        lines.append(
            f"- member_id={lowest_member.get('member_id')}, role={lowest_member.get('role')}, score={lowest_member.get('individual_satisfaction_score')}"
        )
        explanation = safe_text(lowest_member.get("member_explanation", ""))
        if explanation:
            lines.append(f"- 原因说明：{explanation}")
    rejected_members = family_issue.get("rejected_members", [])
    if rejected_members:
        lines.append("未接受方案成员：")
        for member in rejected_members:
            lines.append(
                f"- member_id={member.get('member_id')}, role={member.get('role')}, score={member.get('individual_satisfaction_score')}, reason={safe_text(member.get('member_explanation', ''))}"
            )
    hard_failed_members = family_issue.get("hard_failed_members", [])
    if hard_failed_members:
        lines.append("硬约束未满足成员：")
        for member in hard_failed_members:
            lines.append(
                f"- member_id={member.get('member_id')}, role={member.get('role')}, failed_reasons={safe_text(member.get('hard_constraint_failed_reasons', ''))}"
            )
    lines.append(
        "处理策略：系统未使用 fallback conservative planner，而是保留当前搜索过程中综合得分最高的方案，并把上述问题作为后续人工修改或下一轮优化依据。"
    )
    return "\n".join(lines)


def append_issue_diagnosis_to_plans(
    plans_df: pd.DataFrame, best_plan_issue_report: Dict[str, Any]
) -> pd.DataFrame:
    if plans_df.empty or "family_id" not in plans_df.columns:
        return plans_df
    output_df = plans_df.copy()
    issue_items = best_plan_issue_report.get("families", [])
    issue_lookup = {safe_int(item.get("family_id"), 0): item for item in issue_items}
    diagnosis_texts = []
    diagnosis_jsons = []
    for index, row in output_df.iterrows():
        family_id = safe_int(row.get("family_id"), 0)
        issue_item = issue_lookup.get(family_id)
        if issue_item:
            diagnosis_text = format_issue_diagnosis_text(issue_item)
            diagnosis_json = make_json_safe(issue_item)
        else:
            diagnosis_text = "【问题诊断】当前方案已达到所有通过标准，暂无未达标问题。"
            diagnosis_json = {
                "family_id": family_id,
                "status": "passed_all_requirements",
                "diagnosis": "当前方案已达到所有通过标准，暂无未达标问题。",
            }
        diagnosis_texts.append(diagnosis_text)
        diagnosis_jsons.append(diagnosis_json)
        old_itinerary = safe_text(row.get("itinerary_text", ""))
        old_summary = safe_text(row.get("coordinator_summary", ""))
        if diagnosis_text not in old_itinerary:
            output_df.at[index, "itinerary_text"] = (
                old_itinerary + "\n\n" + diagnosis_text
            ).strip()
        if "coordinator_summary" in output_df.columns:
            if diagnosis_text not in old_summary:
                output_df.at[index, "coordinator_summary"] = (
                    old_summary + "\n\n" + diagnosis_text
                ).strip()
    output_df["issue_diagnosis_text"] = diagnosis_texts
    output_df["issue_diagnosis_json"] = [
        __import__("json").dumps(item, ensure_ascii=False) for item in diagnosis_jsons
    ]
    return output_df


def _is_budget_infeasible_row(plan_row: Dict[str, Any]) -> bool:
    """True when no LLM revision can make the plan fit the budget — don't burn rounds.

    The plan has already passed through every deterministic budget reduction in
    Module 5 (economical meals, paid-attraction swaps, hotel sized to the remaining
    budget).  So if (a) the unavoidable round-trip + local transport alone exceeds
    the budget, or (b) the already-relieved total is still well over budget, the LLM
    cannot bridge the gap and the revision rounds are wasted.
    """
    budget = safe_float(plan_row.get("budget_limit"), 0)
    if budget <= 0:
        return False
    try:
        cb = json.loads(plan_row.get("cost_breakdown_json", "{}") or "{}")
    except Exception:
        cb = {}
    if isinstance(cb, dict):
        fixed = safe_float(cb.get("intercity_transport_cost"), 0) + safe_float(
            cb.get("local_transport_cost"), 0
        )
        if fixed > budget * 1.10:
            return True
    total = safe_float(plan_row.get("total_cost"), 0)
    # Relieved total still far over budget => not LLM-fixable.
    return total > 0 and total > budget * 1.20


def _only_budget_failure(failure_reasons: List[str]) -> bool:
    """True when budget is the only failing dimension — the rest (satisfaction,
    acceptance, hard constraints, grounding) are things LLM revision CAN fix."""
    blob = " ".join(safe_text(r).lower() for r in (failure_reasons or []))
    if not blob:
        return False
    has_budget = "budget" in blob or "预算" in blob
    has_other = any(
        k in blob
        for k in [
            "satisfaction",
            "acceptance",
            "hard_constraint",
            "unresolved",
            "vague",
            "grounding",
            "walking",
            "return_time",
            "rationality",
            "objective",
            "consensus",
            "unfair",
            "满意",
            "接受",
            "冲突",
        ]
    )
    return has_budget and not has_other


def revise_one_family(
    family_id: int,
    failure_reasons: List[str],
    revision_round: int,
    current_plans_df: pd.DataFrame,
    current_plan_eval_df: pd.DataFrame,
    current_member_eval_df: pd.DataFrame,
    family_agents_map: Dict[int, List[Dict[str, Any]]],
    family_proposals_map: Dict[int, List[Dict[str, Any]]],
    conflict_lookup: Dict[int, Dict[str, Any]],
    negotiation_lookup: Dict[int, Dict[str, Any]],
    persona_csv_path: Path,
    agents_json_path: Path,
    num_candidates: int,
) -> Dict[str, Any]:
    if family_id not in family_agents_map:
        raise ValueError(f"family_id={family_id} 不在 persona_agents.json 中。")
    family_agents = family_agents_map[family_id]
    individual_proposals = family_proposals_map.get(family_id, [])
    conflict_info = conflict_lookup.get(family_id, {"family_id": family_id, "conflicts": []})
    negotiation_info = negotiation_lookup.get(family_id, {"family_id": family_id})
    original_plan = get_current_plan_row(plans_df=current_plans_df, family_id=family_id)
    old_eval_row = get_plan_eval_row_by_family(
        plan_eval_df=current_plan_eval_df, family_id=family_id
    )
    old_score = compute_plan_quality_score(old_eval_row)
    # Efficiency: if budget is the only failure AND even the unavoidable costs already
    # exceed the budget, no LLM revision can fix it — skip the slow regeneration rounds
    # and keep the current best plan (it will be parked for at most one backtrack).
    if _only_budget_failure(failure_reasons) and _is_budget_infeasible_row(original_plan):
        print(f"[Module 7] family_id={family_id} 预算不可行（城际+市内交通已超预算），跳过 LLM 修复轮次，保留现状。")
        return {
            "family_id": family_id,
            "revision_round": revision_round,
            "old_score": old_score,
            "best_score": old_score,
            "best_candidate_row": None,
            "best_candidate_eval": old_eval_row,
            "best_candidate_passed": False,
            "candidate_logs": [],
            "generated_candidate_count": 0,
            "skipped_budget_infeasible": True,
        }
    best_candidate_row = None
    best_candidate_eval = None
    best_candidate_passed = False
    best_score = -999999.0
    candidate_logs = []
    for candidate_index in range(1, num_candidates + 1):
        print(
            f"[Module 7] family_id={family_id}, revision_round={revision_round}, candidate={candidate_index}/{num_candidates}"
        )
        try:
            raw_feedback_prompt = build_module6_feedback_prompt(
                family_id=family_id,
                plan_eval_df=current_plan_eval_df,
                member_eval_df=current_member_eval_df,
                original_plan=original_plan,
                failure_reasons=failure_reasons,
                revision_round=revision_round,
                candidate_index=candidate_index,
            )
            strategy = select_revision_strategy(
                candidate_index=candidate_index,
                revision_round=revision_round,
                failure_reasons=failure_reasons,
                old_eval_row=old_eval_row,
                original_plan=original_plan,
            )
            strategy_prompt = build_candidate_strategy_prompt(
                strategy=strategy,
                family_id=family_id,
                old_eval_row=old_eval_row,
                original_plan=original_plan,
            )
            feedback_prompt = compact_feedback_for_module5(
                raw_feedback_prompt, failure_reasons=failure_reasons, max_chars=2400
            )
            feedback_prompt = compact_feedback_for_llm(
                strategy_prompt + "\n\n" + feedback_prompt, max_chars=2800
            )
            feedback_debug = make_feedback_debug_info(
                feedback_prompt=feedback_prompt,
                family_id=family_id,
                revision_round=revision_round,
                candidate_index=candidate_index,
                failure_reasons=failure_reasons,
            )
            feedback_debug["candidate_strategy"] = strategy
            feedback_debug["raw_feedback_length"] = len(safe_text(raw_feedback_prompt))
            feedback_debug["contains_unresolved_conflicts"] = "unresolved_conflicts" in safe_text(
                feedback_prompt
            )
            feedback_debug["contains_split_activity_feedback"] = (
                "split_activity" in safe_text(feedback_prompt).lower()
                or "split_activities" in safe_text(feedback_prompt).lower()
            )
            print(
                f"[Module 7] feedback updated check: family_id={family_id}, round={revision_round}, candidate={candidate_index}, length={feedback_debug['feedback_length']}, hash={feedback_debug['feedback_hash']}, unresolved={feedback_debug['contains_unresolved_conflicts']}"
            )
            plan = generate_consensus_itinerary_for_family(
                family_id=family_id,
                family_agents=family_agents,
                individual_proposals=individual_proposals,
                conflict_info=conflict_info,
                negotiation_info=negotiation_info,
                extra_feedback_prompt=feedback_prompt,
            )
            plan["revision_round"] = revision_round
            plan["candidate_index"] = candidate_index
            plan["method"] = "feedback_revision_loop"
            plan["plan_id"] = f"consensus_plan_revised_r{revision_round}_c{candidate_index}"
            plan["status"] = "success"
            candidate_row = flatten_plan_for_csv(plan)
            candidate_row["revision_round"] = revision_round
            candidate_row["candidate_index"] = candidate_index
            candidate_eval_row = evaluate_candidate_row(
                candidate_row=candidate_row,
                persona_csv_path=persona_csv_path,
                agents_json_path=agents_json_path,
                family_id=family_id,
                revision_round=revision_round,
                candidate_index=candidate_index,
            )
            candidate_score = compute_plan_quality_score(candidate_eval_row)
            candidate_passed = is_plan_passed_requirements(candidate_eval_row)
            candidate_logs.append(
                {
                    "candidate_index": candidate_index,
                    "status": "success",
                    "candidate_score": candidate_score,
                    "candidate_passed": candidate_passed,
                    "candidate_eval": candidate_eval_row,
                    "candidate_strategy": strategy,
                    "feedback_debug": feedback_debug,
                }
            )
            print(
                f"[Module 7] family_id={family_id}, candidate={candidate_index}, old_score={old_score}, candidate_score={candidate_score}, passed={candidate_passed}"
            )
            if candidate_score > best_score:
                best_score = candidate_score
                best_candidate_row = candidate_row
                best_candidate_eval = candidate_eval_row
                best_candidate_passed = candidate_passed
            if candidate_passed:
                print(f"[Module 7] family_id={family_id}, candidate={candidate_index} 已达标，提前停止。")
                break
        except Exception as e:
            print(f"[Module 7] family_id={family_id}, candidate={candidate_index} 失败: {e}")
            candidate_logs.append(
                {
                    "candidate_index": candidate_index,
                    "status": "failed",
                    "error": str(e),
                    "candidate_strategy": locals().get("strategy", {}),
                    "feedback_debug": locals().get("feedback_debug", {}),
                }
            )
    return {
        "family_id": family_id,
        "revision_round": revision_round,
        "old_score": old_score,
        "best_score": best_score,
        "best_candidate_row": best_candidate_row,
        "best_candidate_eval": best_candidate_eval,
        "best_candidate_passed": best_candidate_passed,
        "candidate_logs": candidate_logs,
        "generated_candidate_count": len(candidate_logs),
    }


def build_agent_lookup_from_list(agents: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup = {}
    for agent in agents:
        agent_id = safe_text(agent.get("agent_id", ""))
        if agent_id:
            lookup[agent_id] = agent
    return lookup


def load_proposal_eval_lookup(path: Path = PROPOSAL_EVAL_CSV) -> Dict[str, Dict[str, Any]]:
    if path is None or not Path(path).exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    lookup = {}
    for _, row in df.iterrows():
        agent_id = safe_text(row.get("agent_id", ""))
        if agent_id:
            lookup[agent_id] = row.to_dict()
    return lookup


def get_family_member_rows(member_eval_df: pd.DataFrame, family_id: int) -> pd.DataFrame:
    if member_eval_df.empty or "family_id" not in member_eval_df.columns:
        return pd.DataFrame()
    return member_eval_df[member_eval_df["family_id"].astype(int) == int(family_id)].copy()


def summarize_member_for_backtracking(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "member_id": safe_int(row.get("member_id"), 0),
        "agent_id": safe_text(row.get("agent_id", "")),
        "role": safe_text(row.get("role", "")),
        "individual_satisfaction_score": safe_float(row.get("individual_satisfaction_score"), 0),
        "accepted_vote": str(row.get("accepted_vote", "")).lower() in ["true", "1", "yes"],
        "hard_constraint_pass": str(row.get("hard_constraint_pass", "")).lower()
        in ["true", "1", "yes"],
        "hard_constraint_failed_reasons": safe_text(row.get("hard_constraint_failed_reasons", "")),
        "member_explanation": safe_text(row.get("member_explanation", ""))[:800],
    }


def build_backtracking_feedback(
    failed_families: Dict[int, List[str]],
    current_plan_eval_df: pd.DataFrame,
    current_member_eval_df: pd.DataFrame,
    agents: List[Dict[str, Any]],
    proposal_eval_csv_path: Path = PROPOSAL_EVAL_CSV,
    revision_rounds_completed: int = 3,
) -> Dict[str, Any]:
    agent_lookup = build_agent_lookup_from_list(agents)
    proposal_eval_lookup = load_proposal_eval_lookup(proposal_eval_csv_path)
    failed_items = []
    for family_id, reasons in failed_families.items():
        family_id = int(family_id)
        plan_eval_row = get_plan_eval_row_by_family(current_plan_eval_df, family_id)
        member_rows_df = get_family_member_rows(current_member_eval_df, family_id)
        lowest_member = {}
        rejected_members = []
        hard_failed_members = []
        target_agent_ids = set()
        if not member_rows_df.empty:
            member_rows_df["score_numeric"] = member_rows_df["individual_satisfaction_score"].apply(
                lambda x: safe_float(x, 0)
            )
            lowest_row = (
                member_rows_df.sort_values(by="score_numeric", ascending=True).iloc[0].to_dict()
            )
            lowest_member = summarize_member_for_backtracking(lowest_row)
            if lowest_member.get("agent_id"):
                target_agent_ids.add(lowest_member["agent_id"])
            for _, mrow in member_rows_df.iterrows():
                m = summarize_member_for_backtracking(mrow.to_dict())
                if not m.get("accepted_vote", True):
                    rejected_members.append(m)
                    if m.get("agent_id"):
                        target_agent_ids.add(m["agent_id"])
                if not m.get("hard_constraint_pass", True):
                    hard_failed_members.append(m)
                    if m.get("agent_id"):
                        target_agent_ids.add(m["agent_id"])
        for agent_id, peval in proposal_eval_lookup.items():
            if safe_int(peval.get("family_id"), -1) == family_id:
                proposal_score = safe_float(peval.get("individual_proposal_score"), 100)
                if proposal_score < 70:
                    target_agent_ids.add(agent_id)
        bad_individual_proposals = []
        for agent_id in sorted(target_agent_ids):
            peval = proposal_eval_lookup.get(agent_id, {})
            agent = agent_lookup.get(agent_id, {})
            profile = agent.get("profile", {})
            issues_text = safe_text(peval.get("main_issues_json", "")) or safe_text(
                peval.get("main_issues", "")
            )
            instructions_text = safe_text(peval.get("future_instructions_json", "")) or safe_text(
                peval.get("future_instructions", "")
            )
            bad_individual_proposals.append(
                {
                    "agent_id": agent_id,
                    "member_id": safe_int(
                        agent.get("member_id"), safe_int(peval.get("member_id"), 0)
                    ),
                    "role": safe_text(profile.get("role", peval.get("role", ""))),
                    "individual_proposal_score": safe_float(
                        peval.get("individual_proposal_score"), None
                    ),
                    "quality_level": safe_text(
                        peval.get("quality_level", peval.get("proposal_quality_level", ""))
                    ),
                    "why_bad": issues_text[:800],
                    "future_instruction": instructions_text[:800],
                }
            )
        failure_profile = build_backtracking_failure_profile(
            plan_eval_row=plan_eval_row,
            lowest_member=lowest_member,
            rejected_members=rejected_members,
            hard_failed_members=hard_failed_members,
            reasons=reasons,
        )
        recommended_strategy = failure_profile.get(
            "recommended_regeneration_strategy", "hard_constraint_first"
        )
        feedback_for_module2 = [
            "Backtracking retry: regenerate proposals with the previous failure profile in mind, not as a normal first attempt.",
            "Each persona must restate exact hard constraints, rejection triggers, negotiable parts, and acceptance criteria.",
            "If the previous failure involved split activities, state which activities can be split and which alternatives are acceptable.",
            f"Recommended family-level regeneration strategy: {recommended_strategy}.",
        ]
        feedback_for_module4 = [
            "Previous final plan failed after Module 7 revision rounds.",
            "During negotiation, explicitly protect previous rejected members and the previous lowest-satisfaction member.",
            "Do not sacrifice hard constraints or rejected-member acceptance for average group satisfaction.",
            f"Use recommended strategy: {recommended_strategy}.",
            "Round 4 final consensus direction must list: shared activities, split activities with exact roles, alternatives, return time, walking limit, and budget control.",
            "If a hard conflict cannot be solved, mark it honestly instead of pretending Full Consensus.",
        ]
        if lowest_member:
            feedback_for_module4.append(
                f"Previous lowest member: member_id={lowest_member.get('member_id')}, role={lowest_member.get('role')}, score={lowest_member.get('individual_satisfaction_score')}."
            )
        failed_items.append(
            {
                "family_id": family_id,
                "trigger_rule": "failed_after_3_revision_rounds",
                "revision_rounds_completed": revision_rounds_completed,
                "final_plan_quality_score": compute_plan_quality_score(plan_eval_row),
                "final_failed_reasons": reasons,
                "backtracking_failure_profile": failure_profile,
                "recommended_regeneration_strategy": recommended_strategy,
                "strategy_templates_for_next_cycle": REVISION_STRATEGY_LIBRARY,
                "lowest_satisfaction_member": lowest_member,
                "rejected_members": rejected_members,
                "hard_failed_members": hard_failed_members,
                "bad_individual_proposals": bad_individual_proposals,
                "feedback_for_module2": feedback_for_module2,
                "feedback_for_module4_negotiation": feedback_for_module4,
                "feedback_for_module5_generation": [
                    "When the family returns to Module 5, do not generate from scratch blindly; follow the recommended strategy and previous failure profile.",
                    "Create role-explicit split activities only for real hard conflicts.",
                    "Preserve any previously successful constraints and only change failing parts when possible.",
                ],
            }
        )
    return {
        "backtrack_required": len(failed_items) > 0,
        "trigger_rule": "failed_after_3_revision_rounds",
        "failed_families": failed_items,
    }


def write_backtracking_negative_memories(
    backtracking_feedback: Dict[str, Any],
    agents: List[Dict[str, Any]],
    memory_dir: Path = MEMORY_DIR,
):
    if not backtracking_feedback.get("backtrack_required"):
        return
    agent_lookup = build_agent_lookup_from_list(agents)
    memory_store = RoleMemoryStore(memory_dir)
    for family in backtracking_feedback.get("failed_families", []):
        for item in family.get("bad_individual_proposals", []):
            agent_id = safe_text(item.get("agent_id", ""))
            agent = agent_lookup.get(agent_id)
            if not agent:
                continue
            memory_store.append_memory(
                agent=agent,
                memory={
                    "memory_type": "negative_proposal_feedback",
                    "source": "module7_backtracking_after_3_rounds",
                    "proposal_score": item.get("individual_proposal_score"),
                    "quality_level": item.get("quality_level"),
                    "why_bad": [safe_text(item.get("why_bad", ""))[:220]],
                    "final_plan_failure_evidence": family.get("final_failed_reasons", [])[:5],
                    "future_instruction": [
                        safe_text(item.get("future_instruction", ""))[:220]
                        or "Future proposal must clearly state hard constraints, rejection triggers, negotiable parts, and acceptance criteria."
                    ],
                },
            )


def run_feedback_revision_loop(
    persona_csv_path: Path = PERSONA_CSV,
    agents_json_path: Path = AGENTS_JSON,
    proposals_csv_path: Path = PROPOSALS_CSV,
    conflicts_json_path: Path = CONFLICTS_JSON,
    negotiation_json_path: Path = NEGOTIATION_JSON,
    original_plans_csv_path: Path = ORIGINAL_PLANS_CSV,
    plan_eval_csv_path: Path = PLAN_EVAL_CSV,
    member_eval_csv_path: Path = MEMBER_EVAL_CSV,
    final_revised_plans_csv_path: Path = FINAL_REVISED_PLANS_CSV,
    revision_results_json_path: Path = REVISION_RESULTS_JSON,
    revised_member_eval_csv_path: Path = REVISED_MEMBER_EVAL_CSV,
    revised_plan_eval_csv_path: Path = REVISED_PLAN_EVAL_CSV,
    revised_report_json_path: Path = REVISED_REPORT_JSON,
    max_revision_rounds: Optional[int] = None,
    test_first_n_bad_families: Optional[int] = None,
    max_workers_families: int = 1,
    num_candidates: int = NUM_REVISION_CANDIDATES,
    no_improvement_patience_rounds: int = NO_IMPROVEMENT_PATIENCE_ROUNDS,
    min_improvement_delta: float = MIN_IMPROVEMENT_DELTA,
    max_regeneration_attempts_per_family: int = MAX_REGENERATION_ATTEMPTS_PER_FAMILY,
    backtrack_after_revision_rounds: int = BACKTRACK_AFTER_REVISION_ROUNDS,
    backtracking_feedback_json_path: Path = BACKTRACKING_FEEDBACK_JSON,
    proposal_eval_csv_path: Path = PROPOSAL_EVAL_CSV,
    memory_dir: Path = MEMORY_DIR,
    enable_backtracking_memory: bool = True,
    enable_revision: bool = True,
) -> Dict[str, Any]:
    print("[Module 7] Feedback Revision Loop 开始")
    print("[Module 7] 停止策略: 最多修复 3 轮；若仍不达标则生成 backtracking feedback 回到模块 2。")
    print(f"[Module 7] no_improvement_patience_rounds: {no_improvement_patience_rounds}")
    print(f"[Module 7] min_improvement_delta: {min_improvement_delta}")
    if max_revision_rounds is not None:
        print("[Module 7] 提示: max_revision_rounds 参数已保留兼容，但当前版本不再作为停止条件。")
    print(f"[Module 7] num_candidates: {num_candidates}")
    print(f"[Module 7] test_first_n_bad_families: {test_first_n_bad_families}")
    print(f"[Module 7] 每个家庭最多重新生成次数: {max_regeneration_attempts_per_family}")
    print(f"[Module 7] backtrack_after_revision_rounds: {backtrack_after_revision_rounds}")
    print(f"[Module 7] incremental_round_eval: {MODULE7_INCREMENTAL_ROUND_EVAL}")
    print(
        f"[Module 7] early_backtrack_all_candidates_failed: {MODULE7_EARLY_BACKTRACK_ON_ALL_CANDIDATES_FAILED}, limit={MODULE7_MAX_ALL_FAILED_ROUNDS_PER_FAMILY}"
    )
    print(
        f"[Module 7] early_backtrack_no_improvement: {MODULE7_EARLY_BACKTRACK_ON_NO_IMPROVEMENT}, limit={MODULE7_MAX_NO_IMPROVEMENT_ROUNDS_PER_FAMILY}"
    )
    required_files = [
        persona_csv_path,
        agents_json_path,
        proposals_csv_path,
        conflicts_json_path,
        negotiation_json_path,
        original_plans_csv_path,
        plan_eval_csv_path,
        member_eval_csv_path,
    ]
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"[Module 7] 找不到必要文件: {path}")
    agents = load_json_file(agents_json_path)
    family_agents_map = build_family_agents(agents)
    family_proposals_map = load_individual_proposals(proposals_csv_path)
    conflict_lookup = load_conflict_results(conflicts_json_path)
    negotiation_lookup = load_negotiation_results(negotiation_json_path)
    current_plans_df = load_csv(original_plans_csv_path)
    current_plan_eval_df = load_csv(plan_eval_csv_path)
    current_member_eval_df = load_csv(member_eval_csv_path)
    all_family_ids = sorted(current_plan_eval_df["family_id"].astype(int).unique().tolist())
    regeneration_counter = {int(family_id): 0 for family_id in all_family_ids}
    all_round_logs = []
    failed_families = find_failed_families(
        plan_eval_df=current_plan_eval_df, test_first_n_failed_families=test_first_n_bad_families
    )
    print(f"[Module 7] 初始不达标家庭数量: {len(failed_families)}")
    if not enable_revision:
        # Repair disabled (ENABLE_QUALITY_CONTROL=False, i.e. the no_quality_control ablation):
        # do NO revision at all — the Module 5 first-attempt plans are final.  We only re-run the
        # final evaluation so post-repair == pre and compute_repair_effect shows 0 gain
        # (instead of the loop silently improving plans).
        print("[Module 7] Repair 已关闭(质量管控消融)：不修订，直接以 Module 5 原方案定稿并评估。")
        save_plans(plans_df=current_plans_df, output_path=final_revised_plans_csv_path)
        eval_result = evaluate_and_explain_fast(
            persona_csv_path=persona_csv_path,
            agents_json_path=agents_json_path,
            generated_plans_csv_path=final_revised_plans_csv_path,
            member_output_csv_path=revised_member_eval_csv_path,
            plan_output_csv_path=revised_plan_eval_csv_path,
            report_output_json_path=revised_report_json_path,
            use_llm_judge=FINAL_EVAL_USE_LLM_JUDGE,
        )
        still_failed = find_failed_families(
            plan_eval_df=eval_result["plan_eval_df"],
            test_first_n_failed_families=test_first_n_bad_families,
        )
        final_report = {
            "status": "repair_disabled_no_revision",
            "all_requirements_met": not still_failed,
            "total_revision_rounds": 0,
            "remaining_failed_families": {int(k): v for k, v in still_failed.items()},
            "round_logs": [],
            "final_revised_plans_csv": str(final_revised_plans_csv_path),
            "revised_plan_eval_csv": str(revised_plan_eval_csv_path),
            "revised_member_eval_csv": str(revised_member_eval_csv_path),
            "revised_report_json": str(revised_report_json_path),
        }
        save_json_file(data=make_json_safe(final_report), path=revision_results_json_path)
        save_json_file(
            data={"backtrack_required": False, "failed_families": []},
            path=backtracking_feedback_json_path,
        )
        return final_report
    if not failed_families:
        print("[Module 7] 快速规则评价没有不达标家庭，先保存原计划为 feedback final。")
        save_plans(plans_df=current_plans_df, output_path=final_revised_plans_csv_path)
        eval_result = evaluate_and_explain_fast(
            persona_csv_path=persona_csv_path,
            agents_json_path=agents_json_path,
            generated_plans_csv_path=final_revised_plans_csv_path,
            member_output_csv_path=revised_member_eval_csv_path,
            plan_output_csv_path=revised_plan_eval_csv_path,
            report_output_json_path=revised_report_json_path,
            use_llm_judge=FINAL_EVAL_USE_LLM_JUDGE,
        )
        current_plan_eval_df = eval_result["plan_eval_df"]
        current_member_eval_df = eval_result["member_eval_df"]
        failed_families = find_failed_families(
            plan_eval_df=current_plan_eval_df,
            test_first_n_failed_families=test_first_n_bad_families,
        )
        if not failed_families:
            final_report = {
                "status": "completed_all_requirements_met",
                "all_requirements_met": True,
                "stop_strategy": "per_family_three_round_limit_then_targeted_module2_backtrack",
                "stopped_reason": "initial_all_requirements_met_after_final_llm_judge",
                "no_improvement_rounds": 0,
                "no_improvement_patience_rounds": no_improvement_patience_rounds,
                "min_improvement_delta": min_improvement_delta,
                "max_regeneration_attempts_per_family": max_regeneration_attempts_per_family,
                "total_revision_rounds": 0,
                "remaining_failed_families": {},
                "regeneration_counter": regeneration_counter,
                "total_regeneration_count": 0,
                "round_logs": [],
                "final_revised_plans_csv": str(final_revised_plans_csv_path),
                "revised_plan_eval_csv": str(revised_plan_eval_csv_path),
                "revised_member_eval_csv": str(revised_member_eval_csv_path),
                "revised_report_json": str(revised_report_json_path),
            }
            save_json_file(data=make_json_safe(final_report), path=revision_results_json_path)
            save_json_file(
                data={"backtrack_required": False, "failed_families": []},
                path=backtracking_feedback_json_path,
            )
            return final_report
        print(
            f"[Module 7] 最终 LLM Judge 复查后发现仍不达标家庭，进入 Module 7 修复循环，family_id={sorted([int(x) for x in failed_families.keys()])}"
        )
    revision_round = 0
    no_improvement_rounds = 0
    stopped_reason = "all_requirements_met"
    family_revision_rounds = {int(family_id): 0 for family_id in failed_families.keys()}
    family_all_candidate_failed_rounds: Dict[int, int] = {}
    family_no_improvement_rounds: Dict[int, int] = {}
    backtrack_waiting_family_ids = set()
    backtrack_waiting_reasons: Dict[int, List[str]] = {}
    while failed_families:
        revision_round += 1
        print("\n" + "=" * 100)
        print(f"[Module 7] revision_round={revision_round}")
        print(f"[Module 7] 当前不达标家庭数量: {len(failed_families)}")
        print("=" * 100)
        round_improved = False
        changed_family_ids_this_round = set()
        round_improvement_details = []
        round_log = {
            "revision_round": revision_round,
            "no_improvement_rounds_before_round": no_improvement_rounds,
            "family_logs": [],
        }
        active_failed_families = {}
        for family_id, reasons in failed_families.items():
            fid = int(family_id)
            family_revision_rounds.setdefault(fid, 0)
            if fid in backtrack_waiting_family_ids:
                continue
            if family_revision_rounds.get(fid, 0) >= backtrack_after_revision_rounds:
                backtrack_waiting_family_ids.add(fid)
                backtrack_waiting_reasons[fid] = reasons
                print(
                    f"[Module 7] family_id={fid} 已完成 {family_revision_rounds.get(fid, 0)} 轮修复仍不达标，停止继续生成，等待与其他失败家庭一起回退 Module 2。"
                )
                continue
            current_count = regeneration_counter.get(fid, 0)
            if current_count >= max_regeneration_attempts_per_family:
                backtrack_waiting_family_ids.add(fid)
                backtrack_waiting_reasons[fid] = reasons
                print(
                    f"[Module 7] family_id={fid} 已达到最大重新生成次数 {max_regeneration_attempts_per_family}，停止继续生成，等待统一回退 Module 2。"
                )
                continue
            active_failed_families[fid] = reasons
        if not active_failed_families:
            if backtrack_waiting_family_ids:
                print(
                    f"[Module 7] 当前没有可继续修复的 active failed family。等待回退 Module 2 的 family_id={sorted(backtrack_waiting_family_ids)}"
                )
                stopped_reason = "all_remaining_failed_families_waiting_for_module2_backtrack"
            else:
                print("[Module 7] 没有可继续修复的 failed family，停止。")
                stopped_reason = "no_active_failed_families"
            break
        for family_id, reasons in active_failed_families.items():
            remaining_attempts = max_regeneration_attempts_per_family - regeneration_counter.get(
                int(family_id), 0
            )
            if remaining_attempts <= 0:
                print(f"[Module 7] family_id={family_id} 没有剩余重生成次数，跳过。")
                continue
            effective_num_candidates = min(num_candidates, remaining_attempts)
            result = revise_one_family(
                family_id=family_id,
                failure_reasons=reasons,
                revision_round=revision_round,
                current_plans_df=current_plans_df,
                current_plan_eval_df=current_plan_eval_df,
                current_member_eval_df=current_member_eval_df,
                family_agents_map=family_agents_map,
                family_proposals_map=family_proposals_map,
                conflict_lookup=conflict_lookup,
                negotiation_lookup=negotiation_lookup,
                persona_csv_path=persona_csv_path,
                agents_json_path=agents_json_path,
                num_candidates=effective_num_candidates,
            )
            candidate_count = len(result.get("candidate_logs", []))
            regeneration_counter[int(family_id)] = (
                regeneration_counter.get(int(family_id), 0) + candidate_count
            )
            family_revision_rounds[int(family_id)] = (
                family_revision_rounds.get(int(family_id), 0) + 1
            )
            result["family_revision_rounds_completed"] = family_revision_rounds[int(family_id)]
            best_candidate_row = result.get("best_candidate_row")
            best_candidate_passed = bool(result.get("best_candidate_passed", False))
            old_score = safe_float(result.get("old_score"), -999999.0)
            best_score = safe_float(result.get("best_score"), -999999.0)
            old_eval_row = get_plan_eval_row_by_family(
                plan_eval_df=current_plan_eval_df, family_id=family_id
            )
            new_eval_row = result.get("best_candidate_eval", {})
            score_improved = (
                best_candidate_row is not None and best_score > old_score + min_improvement_delta
            )
            same_score = (
                best_candidate_row is not None
                and abs(best_score - old_score) <= min_improvement_delta
            )
            tiebreak_improved = same_score and is_tiebreak_improved(
                old_eval_row=old_eval_row,
                new_eval_row=new_eval_row,
                min_improvement_delta=min_improvement_delta,
            )
            should_replace = best_candidate_row is not None and (
                best_candidate_passed or score_improved or tiebreak_improved
            )
            old_tiebreak = get_tiebreak_values(old_eval_row)
            new_tiebreak = get_tiebreak_values(new_eval_row)
            result["score_improved"] = score_improved
            result["same_score"] = same_score
            result["tiebreak_improved"] = tiebreak_improved
            result["old_tiebreak"] = old_tiebreak
            result["new_tiebreak"] = new_tiebreak
            result["score_improvement_delta"] = round(best_score - old_score, 4)
            result["min_improvement_delta"] = min_improvement_delta
            all_candidates_failed = best_candidate_row is None and candidate_count > 0
            if all_candidates_failed:
                family_all_candidate_failed_rounds[int(family_id)] = (
                    family_all_candidate_failed_rounds.get(int(family_id), 0) + 1
                )
                result["all_candidates_failed_rounds"] = family_all_candidate_failed_rounds[
                    int(family_id)
                ]
                if (
                    MODULE7_EARLY_BACKTRACK_ON_ALL_CANDIDATES_FAILED
                    and family_all_candidate_failed_rounds[int(family_id)]
                    >= MODULE7_MAX_ALL_FAILED_ROUNDS_PER_FAMILY
                ):
                    backtrack_waiting_family_ids.add(int(family_id))
                    backtrack_waiting_reasons[int(family_id)] = reasons + [
                        f"Module 7 early backtrack: all candidates failed for {family_all_candidate_failed_rounds[int(family_id)]} round(s)."
                    ]
                    print(
                        f"[Module 7] family_id={family_id} 本轮所有 candidate 都生成/评价失败，直接等待回退 Module 2，避免继续低效重试。"
                    )
            if should_replace:
                family_all_candidate_failed_rounds[int(family_id)] = 0
                family_no_improvement_rounds[int(family_id)] = 0
                changed_family_ids_this_round.add(int(family_id))
                current_plans_df = replace_family_plan(
                    plans_df=current_plans_df, new_plan_row=best_candidate_row
                )
                if score_improved or tiebreak_improved or best_candidate_passed:
                    round_improved = True
                    round_improvement_details.append(
                        {
                            "family_id": int(family_id),
                            "old_score": old_score,
                            "best_score": best_score,
                            "delta": round(best_score - old_score, 4),
                            "passed": best_candidate_passed,
                            "score_improved": score_improved,
                            "tiebreak_improved": tiebreak_improved,
                            "old_tiebreak": old_tiebreak,
                            "new_tiebreak": new_tiebreak,
                        }
                    )
                if score_improved:
                    replace_reason = "score_improved"
                elif best_candidate_passed:
                    replace_reason = "candidate_passed"
                elif tiebreak_improved:
                    replace_reason = "same_score_but_tiebreak_improved"
                else:
                    replace_reason = "unknown"
                print(
                    f"[Module 7] family_id={family_id} 已替换为更优 candidate，replace_reason={replace_reason}, old_score={old_score}, best_score={best_score}, delta={round(best_score - old_score, 4)}, passed={best_candidate_passed}, old_failed_count={old_tiebreak.get('failed_count')}, new_failed_count={new_tiebreak.get('failed_count')}, 累计重新生成次数={regeneration_counter[int(family_id)]}"
                )
            else:
                if not all_candidates_failed:
                    family_no_improvement_rounds[int(family_id)] = (
                        family_no_improvement_rounds.get(int(family_id), 0) + 1
                    )
                    result["no_improvement_rounds_for_family"] = family_no_improvement_rounds[
                        int(family_id)
                    ]
                    if (
                        MODULE7_EARLY_BACKTRACK_ON_NO_IMPROVEMENT
                        and family_no_improvement_rounds[int(family_id)]
                        >= MODULE7_MAX_NO_IMPROVEMENT_ROUNDS_PER_FAMILY
                    ):
                        backtrack_waiting_family_ids.add(int(family_id))
                        backtrack_waiting_reasons[int(family_id)] = reasons + [
                            f"Module 7 early backtrack: no improvement for {family_no_improvement_rounds[int(family_id)]} round(s)."
                        ]
                        print(
                            f"[Module 7] family_id={family_id} 本轮没有有效提升，直接等待回退 Module 2，避免继续生成相似低效 candidate。"
                        )
                print(
                    f"[Module 7] family_id={family_id} 本轮 candidate 没有超过当前方案，也没有减少未达标问题，保留当前最高分方案。old_score={old_score}, best_score={best_score}, delta={round(best_score - old_score, 4)}, old_failed_count={old_tiebreak.get('failed_count')}, new_failed_count={new_tiebreak.get('failed_count')}"
                )
            round_log["family_logs"].append(result)
        round_log["round_improved"] = round_improved
        round_log["round_improvement_details"] = round_improvement_details
        all_round_logs.append(round_log)
        save_plans(plans_df=current_plans_df, output_path=final_revised_plans_csv_path)
        if MODULE7_INCREMENTAL_ROUND_EVAL:
            if changed_family_ids_this_round:
                print("[Module 7] 本轮结束，进行增量 Module 6 评价。")
                eval_result = evaluate_changed_families_and_merge(
                    current_plans_df=current_plans_df,
                    changed_family_ids=sorted(changed_family_ids_this_round),
                    current_plan_eval_df=current_plan_eval_df,
                    current_member_eval_df=current_member_eval_df,
                    persona_csv_path=persona_csv_path,
                    agents_json_path=agents_json_path,
                    member_output_csv_path=revised_member_eval_csv_path,
                    plan_output_csv_path=revised_plan_eval_csv_path,
                    report_output_json_path=revised_report_json_path,
                    revision_round=revision_round,
                )
                current_plan_eval_df = eval_result["plan_eval_df"]
                current_member_eval_df = eval_result["member_eval_df"]
            else:
                print("[Module 7] 本轮没有替换任何方案，跳过全量 Module 6 重评，沿用上一轮评价结果。")
                save_eval_dfs(
                    current_plan_eval_df,
                    current_member_eval_df,
                    revised_plan_eval_csv_path,
                    revised_member_eval_csv_path,
                )
                save_json_file(
                    data={
                        "status": "round_eval_skipped_no_plan_replaced",
                        "revision_round": revision_round,
                    },
                    path=revised_report_json_path,
                )
        else:
            print("[Module 7] 本轮结束，重新运行全量模块 6 评价。")
            eval_result = evaluate_and_explain_fast(
                persona_csv_path=persona_csv_path,
                agents_json_path=agents_json_path,
                generated_plans_csv_path=final_revised_plans_csv_path,
                member_output_csv_path=revised_member_eval_csv_path,
                plan_output_csv_path=revised_plan_eval_csv_path,
                report_output_json_path=revised_report_json_path,
                use_llm_judge=FAST_EVAL_USE_LLM_JUDGE,
            )
            current_plan_eval_df = eval_result["plan_eval_df"]
            current_member_eval_df = eval_result["member_eval_df"]
        failed_families = find_failed_families(
            plan_eval_df=current_plan_eval_df,
            test_first_n_failed_families=test_first_n_bad_families,
        )
        print(f"[Module 7] revision_round={revision_round} 后 仍不达标家庭数量: {len(failed_families)}")
        print("[Module 7] 当前重新生成次数：")
        for family_id, count in regeneration_counter.items():
            print(f"family_id={family_id}, regeneration_count={count}")
        newly_waiting = []
        for family_id, reasons in failed_families.items():
            fid = int(family_id)
            family_revision_rounds.setdefault(fid, 0)
            if family_revision_rounds.get(fid, 0) >= backtrack_after_revision_rounds:
                if fid not in backtrack_waiting_family_ids:
                    newly_waiting.append(fid)
                backtrack_waiting_family_ids.add(fid)
                backtrack_waiting_reasons[fid] = reasons
        if newly_waiting:
            print(
                f"[Module 7] 以下 family 已修满三轮仍不达标，后续不再在 Module 7 生成，等待统一回退 Module 2: {sorted(newly_waiting)}"
            )
        active_remaining_after_round = {
            int(fid): reasons
            for fid, reasons in failed_families.items()
            if int(fid) not in backtrack_waiting_family_ids
            and family_revision_rounds.get(int(fid), 0) < backtrack_after_revision_rounds
            and (regeneration_counter.get(int(fid), 0) < max_regeneration_attempts_per_family)
        }
        if failed_families and (not active_remaining_after_round):
            print("[Module 7] 所有仍不达标 family 都已经通过、修满三轮或达到重生成上限；停止最终方案层修复，准备统一回退 Module 2。")
            stopped_reason = "all_remaining_failed_families_ready_for_module2_backtrack"
            break
        if round_improved:
            no_improvement_rounds = 0
            print("[Module 7] 本轮发现更高分方案，no_improvement_rounds 重置为 0。")
        else:
            no_improvement_rounds += 1
            print(
                f"[Module 7] 本轮没有发现更高分方案，no_improvement_rounds={no_improvement_rounds}/{no_improvement_patience_rounds}"
            )
        running_report = {
            "status": "running",
            "revision_round": revision_round,
            "stop_strategy": "per_family_three_round_limit_then_targeted_module2_backtrack",
            "no_improvement_rounds": no_improvement_rounds,
            "no_improvement_patience_rounds": no_improvement_patience_rounds,
            "min_improvement_delta": min_improvement_delta,
            "max_regeneration_attempts_per_family": max_regeneration_attempts_per_family,
            "round_improved": round_improved,
            "round_improvement_details": round_improvement_details,
            "remaining_failed_families": failed_families,
            "active_remaining_after_round": active_remaining_after_round
            if "active_remaining_after_round" in locals()
            else {},
            "backtrack_waiting_family_ids": sorted(backtrack_waiting_family_ids),
            "backtrack_waiting_reasons": backtrack_waiting_reasons,
            "family_revision_rounds": family_revision_rounds,
            "feedback_update_check": "candidate_logs contain feedback_debug with length/hash/preview for each generated candidate",
            "regeneration_counter": regeneration_counter,
            "total_regeneration_count": sum(regeneration_counter.values()),
            "round_logs": all_round_logs,
            "final_revised_plans_csv": str(final_revised_plans_csv_path),
            "revised_plan_eval_csv": str(revised_plan_eval_csv_path),
            "revised_member_eval_csv": str(revised_member_eval_csv_path),
            "revised_report_json": str(revised_report_json_path),
        }
        save_json_file(data=make_json_safe(running_report), path=revision_results_json_path)
        if not failed_families:
            print("[Module 7] 所有方案已经达标。")
            stopped_reason = "all_requirements_met"
            break
        if no_improvement_rounds >= no_improvement_patience_rounds:
            print(
                f"[Module 7] 连续 {no_improvement_rounds} 轮没有发现更高分方案。当前版本不再因为其他 family 的分数变化或全局无提升而无限继续/提前终止；每个 family 会按自己的三轮上限决定是否进入 Module 2 回退。"
            )
    if FINAL_EVAL_USE_LLM_JUDGE and final_revised_plans_csv_path.exists():
        print("[Module 7] 循环结束，开始对最终方案运行一次完整 LLM Judge 评价。")
        final_eval_result = evaluate_and_explain_fast(
            persona_csv_path=persona_csv_path,
            agents_json_path=agents_json_path,
            generated_plans_csv_path=final_revised_plans_csv_path,
            member_output_csv_path=revised_member_eval_csv_path,
            plan_output_csv_path=revised_plan_eval_csv_path,
            report_output_json_path=revised_report_json_path,
            use_llm_judge=True,
        )
        current_plan_eval_df = final_eval_result["plan_eval_df"]
        current_member_eval_df = final_eval_result["member_eval_df"]
        failed_families = find_failed_families(
            plan_eval_df=current_plan_eval_df,
            test_first_n_failed_families=test_first_n_bad_families,
        )
        print(f"[Module 7] 完整 LLM Judge 评价后 仍不达标家庭数量: {len(failed_families)}")
        if failed_families:
            stopped_reason = "final_llm_judge_failed_requirements"
            for fid, reasons in failed_families.items():
                print(f"[Module 7] final LLM Judge failed: family_id={fid}, reasons={reasons}")
    all_requirements_met = len(failed_families) == 0
    best_plan_issue_report = build_best_plan_issue_report(
        plan_eval_df=current_plan_eval_df,
        member_eval_df=current_member_eval_df,
        remaining_failed_families=failed_families,
    )
    if final_revised_plans_csv_path.exists():
        final_plans_df = load_csv(final_revised_plans_csv_path)
        final_plans_df = append_issue_diagnosis_to_plans(
            plans_df=final_plans_df, best_plan_issue_report=best_plan_issue_report
        )
        save_plans(plans_df=final_plans_df, output_path=final_revised_plans_csv_path)
    if all_requirements_met:
        final_status = "completed_all_requirements_met"
        backtracking_feedback = {"backtrack_required": False, "failed_families": []}
        save_json_file(data=backtracking_feedback, path=backtracking_feedback_json_path)
    else:
        final_status = "completed_best_score_plan_selected_with_issues"
        # Don't backtrack budget-infeasible families through the whole M2→M7 pipeline:
        # re-running cannot change a fixed budget/route, so it only wastes a full cycle.
        backtrack_failed_families = {}
        budget_infeasible_skipped = []
        for fid, reasons in failed_families.items():
            fid = int(fid)
            if _only_budget_failure(reasons) and _is_budget_infeasible_row(
                get_current_plan_row(plans_df=current_plans_df, family_id=fid)
            ):
                budget_infeasible_skipped.append(fid)
                continue
            backtrack_failed_families[fid] = reasons
        if budget_infeasible_skipped:
            print(
                f"[Module 7] 预算不可行家庭 family_id={sorted(budget_infeasible_skipped)} 不回退 Module 2（重跑无法满足预算），如实标记为不可行。"
            )
        if backtrack_failed_families:
            backtracking_feedback = build_backtracking_feedback(
                failed_families=backtrack_failed_families,
                current_plan_eval_df=current_plan_eval_df,
                current_member_eval_df=current_member_eval_df,
                agents=agents,
                proposal_eval_csv_path=proposal_eval_csv_path,
                revision_rounds_completed=revision_round,
            )
            backtracking_feedback["trigger_rule"] = stopped_reason
            for item in backtracking_feedback.get("failed_families", []):
                item["trigger_rule"] = stopped_reason
            backtracking_feedback["feedback_update_check"] = {
                "uses_latest_module6_evaluation": True,
                "source_plan_eval_csv": str(revised_plan_eval_csv_path),
                "source_member_eval_csv": str(revised_member_eval_csv_path),
                "final_llm_judge_failures_are_included": True,
                "backtrack_waiting_family_ids": sorted(backtrack_waiting_family_ids),
                "final_failed_family_ids": sorted([int(x) for x in failed_families.keys()]),
                "backtrack_target_family_ids": sorted(
                    [int(x) for x in backtrack_failed_families.keys()]
                ),
            }
            save_json_file(
                data=make_json_safe(backtracking_feedback), path=backtracking_feedback_json_path
            )
            if enable_backtracking_memory:
                write_backtracking_negative_memories(
                    backtracking_feedback=backtracking_feedback,
                    agents=agents,
                    memory_dir=memory_dir,
                )
            else:
                print("[Module 7] 奖惩记忆已关闭：只保存 backtracking feedback，不写入 negative memory。")
            print(f"[Module 7] 已生成回退反馈: {backtracking_feedback_json_path}")
            print(f"[Module 7] 回退目标 family_id={sorted(backtrack_failed_families.keys())}")
        else:
            backtracking_feedback = {"backtrack_required": False, "failed_families": []}
            save_json_file(data=backtracking_feedback, path=backtracking_feedback_json_path)
    final_report = {
        "status": final_status,
        "all_requirements_met": all_requirements_met,
        "selection_policy": "select_highest_scoring_plan_without_fallback",
        "framework_name": "QG-BMP",
        "framework_full_name": "Quality-Gated Backtracking Multi-Agent Planning",
        "quality_gated_backtracking_summary": {
            "proposal_quality_gate": "handled inline in Module 2 proposal generation (best-of-N quality gate)",
            "final_plan_quality_gate": "handled in Module 6 before Module 7 revision",
            "targeted_backtracking": "only failed family_id values are returned to Module 2",
            "reward_penalty_memory": "positive/negative role-level memories are updated when enabled",
        },
        "stop_strategy": "per_family_three_round_limit_then_targeted_module2_backtrack",
        "stopped_reason": stopped_reason,
        "no_improvement_rounds": no_improvement_rounds,
        "no_improvement_patience_rounds": no_improvement_patience_rounds,
        "min_improvement_delta": min_improvement_delta,
        "max_regeneration_attempts_per_family": max_regeneration_attempts_per_family,
        "note": "Module 7 uses a per-family revision limit. If the final LLM Judge still finds failed families, those families are written into backtracking feedback and returned to Module 2 instead of silently ending.",
        "total_revision_rounds": revision_round,
        "family_revision_rounds": family_revision_rounds,
        "family_all_candidate_failed_rounds": family_all_candidate_failed_rounds,
        "family_no_improvement_rounds": family_no_improvement_rounds,
        "backtrack_waiting_family_ids": sorted(backtrack_waiting_family_ids),
        "backtrack_waiting_reasons": backtrack_waiting_reasons,
        "feedback_update_check": "Each candidate log contains feedback_debug including feedback_length, feedback_hash, failure_reasons_used, and feedback_preview. Backtracking feedback uses the latest revised Module 6 evaluation files.",
        "remaining_failed_families": failed_families,
        "best_plan_issue_report": best_plan_issue_report,
        "backtracking_feedback_json": str(backtracking_feedback_json_path),
        "backtrack_required": backtracking_feedback.get("backtrack_required", False),
        "regeneration_counter": regeneration_counter,
        "total_regeneration_count": sum(regeneration_counter.values()),
        "round_logs": all_round_logs,
        "final_revised_plans_csv": str(final_revised_plans_csv_path),
        "revised_plan_eval_csv": str(revised_plan_eval_csv_path),
        "revised_member_eval_csv": str(revised_member_eval_csv_path),
        "revised_report_json": str(revised_report_json_path),
    }
    save_json_file(data=make_json_safe(final_report), path=revision_results_json_path)
    print("\n" + "=" * 100)
    print("[Module 7] Feedback Revision Loop 完成")
    print(f"[Module 7] final_status: {final_status}")
    print(f"[Module 7] stopped_reason: {stopped_reason}")
    print(f"[Module 7] all_requirements_met: {all_requirements_met}")
    if not all_requirements_met:
        print("[Module 7] 未完全达标，已输出当前最高分方案的问题诊断：")
        print(best_plan_issue_report.get("summary", ""))
    print(f"[Module 7] total_revision_rounds: {revision_round}")
    print(f"[Module 7] total_regeneration_count: {sum(regeneration_counter.values())}")
    print("[Module 7] 每个 family_id 的重新生成次数：")
    for family_id, count in regeneration_counter.items():
        print(f"family_id={family_id}, regeneration_count={count}")
    print(f"[Module 7] final revised plans: {final_revised_plans_csv_path}")
    print(f"[Module 7] revised plan-level evaluation: {revised_plan_eval_csv_path}")
    print(f"[Module 7] revised member-level evaluation: {revised_member_eval_csv_path}")
    print(f"[Module 7] revised report: {revised_report_json_path}")
    print(f"[Module 7] revision results json: {revision_results_json_path}")
    print("=" * 100)
    return final_report
