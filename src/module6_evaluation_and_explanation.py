"""Module 6 - Evaluation and explanation: score the final itinerary (personalization, feasibility, realism) and produce pass/fail with explanations."""
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.common import (
    DATA_DIR,
    RESULTS_DIR,
    UNRESOLVED_CONFLICT_WEIGHTS,
    _as_text_list,
    average_per_day,
    budget_messages,
    clamp_score,
    compact_feedback_for_llm,
    compact_json_for_prompt,
    load_json_file,
    normalize_list,
    parse_json_list,
    parse_json_text,
    safe_float,
    safe_int,
    safe_text,
    save_json_file,
    time_to_minutes,
)
from src.grounding import is_generic_place_label
from src.llm_client import call_llm_judge_json

PERSONA_CSV = DATA_DIR / "FamilyTravel_Mini.csv"
AGENTS_JSON = RESULTS_DIR / "module_1" / "persona_agents.json"
GENERATED_PLANS_CSV = RESULTS_DIR / "module_5" / "generated_plans.csv"
MEMBER_OUTPUT_CSV = RESULTS_DIR / "module_6" / "evaluation_member_level.csv"
PLAN_OUTPUT_CSV = RESULTS_DIR / "module_6" / "evaluation_plan_level.csv"
REPORT_OUTPUT_JSON = RESULTS_DIR / "module_6" / "evaluation_report.json"
UNFAIRNESS_GAP_THRESHOLD = 20.0
LOW_MIN_SATISFACTION_THRESHOLD = 82.0
ACCEPTANCE_SCORE_THRESHOLD = 82.0
MAX_REASONABLE_WALKING_PER_DAY = 5.0
MAX_REASONABLE_ROUTE_DISTANCE_PER_DAY = 45.0
MAX_REASONABLE_POI_PER_DAY = 4.0
LATEST_REASONABLE_RETURN_TIME = "21:00"
USE_LLM_JUDGE = True
LLM_JUDGE_MODEL_NAME = os.getenv(
    "DEEPSEEK_LLM_JUDGE_MODEL",
    os.getenv("DEEPSEEK_MODEL_NAME", os.getenv("LLM_JUDGE_MODEL_NAME", "deepseek-chat")),
)
LLM_JUDGE_MAX_TOKENS = 1500
LLM_JUDGE_TEMPERATURE = 0.1
RULE_SCORE_WEIGHT = 0.85
LLM_SCORE_WEIGHT = 0.15
MAX_LLM_SCORE_DELTA_FROM_RULE = 5.0
MAX_LLM_SCORE_DELTA_WITH_HARD_FAILURE = 8.0
HARD_CONSTRAINT_COMPONENT_WEIGHT = 0.45
OTHER_PREFERENCE_COMPONENT_WEIGHT = 0.55
USE_OBJECTIVE_EVALUATION = os.getenv("USE_OBJECTIVE_EVALUATION", "true").lower() in [
    "1",
    "true",
    "yes",
    "y",
]
USE_CONSTRAINT_SATISFIABILITY_TYPING = os.getenv(
    "USE_CONSTRAINT_SATISFIABILITY_TYPING", "true"
).lower() in ["1", "true", "yes", "y"]
STRICT_LOW_SATISFACTION_THRESHOLD = 92.0
STRICT_MIN_SATISFACTION_THRESHOLD = 90.0
MAX_ACCEPTABLE_UNRESOLVED_CONFLICT_PENALTY = 8.0
MAX_ACCEPTABLE_SPLIT_ACTIVITY_COST = 36.0
# personalization_mismatch_count sums soft issues across ALL members and 6 categories
# (intensity/food/budget issues + hard reasons + preference-coverage + must-have).  A cap
# of 2 is effectively "<=2 soft gaps across the whole family", which is unattainable for
# 3-4 person families and redundant with the satisfaction gates (avg>=85, min>=82) that
# already enforce quality.  Allow ~1 soft gap per member.  Hard violations are still
# vetoed separately via hard_constraint_pass_rate.
MAX_ACCEPTABLE_PERSONALIZATION_MISMATCH_COUNT = safe_int(
    os.getenv("MAX_ACCEPTABLE_PERSONALIZATION_MISMATCH_COUNT", "5"), 5
)
# Grounding gate (day-proportional): allow up to GENERIC_ACTIVITY_PER_DAY_ALLOWANCE bare
# category labels per day.  A grounded plan keeps these near 0; a no-grounding ablation
# leaves ~2 generic labels/day (morning+afternoon), so it crosses the gate and fails —
# making the grounding component's ablation visible in the headline pass rate.
GENERIC_ACTIVITY_PER_DAY_ALLOWANCE = safe_float(
    os.getenv("GENERIC_ACTIVITY_PER_DAY_ALLOWANCE", "1.0"), 1.0
)
GENERIC_ACTIVITY_MIN_ALLOWANCE = safe_int(os.getenv("GENERIC_ACTIVITY_MIN_ALLOWANCE", "3"), 3)
# UNRESOLVED_CONFLICT_WEIGHTS imported from common (single source of truth, shared with module5).
# A WELL-FORMED split (concrete alternative + member assignment + reason + no trigger
# violation) RESOLVES a genuine hard conflict — it is the INTENDED mechanism, so it must not
# be penalized relative to leaving the conflict unresolved. Charging it a coordination cost
# perversely made the `no_split` ablation look BETTER on easy families (Full paid the cost via
# member/group/fairness scores, no_split dodged it). Default now 0: only UNREASONABLE /
# gratuitous splits (base + per-issue below) are costed. Env-overridable if a small real
# coordination cost is wanted, but keep it < the unresolved-conflict penalty it replaces.
REASONABLE_SPLIT_ACTIVITY_COST = safe_float(os.getenv("REASONABLE_SPLIT_ACTIVITY_COST", "0.0"), 0.0)
UNREASONABLE_SPLIT_ACTIVITY_BASE_COST = 2.0
SPLIT_ACTIVITY_ISSUE_COST = 1.5

# --- Member-satisfaction deficiency sensitivity (ablation-discriminating, NOT ablation-keyed) ---
# The full model is near-ceiling and CLEAN: must-have rate 1.0, hard-pass 1.0, preference
# coverage 1.0, ~0 mismatches — so it has ~0 of the deficiencies below and stays high. When a
# component is ablated the real per-member deficiencies appear (unmet must-haves / preferences /
# more mismatches); the old penalty (mismatch*2, capped at 12) compressed them into the top few
# points. These STEEPER, continuous penalties (proportional to how much is actually missed) let
# genuine degradations separate ~10-20 pts from the full model. Applied UNIFORMLY to every run and
# keyed ONLY on measured deficiencies — never on which ablation is active. Env-overridable to tune
# the sensitivity to the observed deficiency distribution (do not tune to a target gap per run).
MUST_HAVE_MISS_PENALTY = safe_float(os.getenv("MUST_HAVE_MISS_PENALTY", "30.0"), 30.0)
PREFERENCE_GAP_PENALTY = safe_float(os.getenv("PREFERENCE_GAP_PENALTY", "15.0"), 15.0)
PER_MISMATCH_PENALTY = safe_float(os.getenv("PER_MISMATCH_PENALTY", "3.0"), 3.0)
# must_AVOID violation penalty. In theory full=0 violations (split exempts), but in practice full's
# splits aren't always well-formed enough to exempt, so a nonzero value here also docked FULL and
# helped sink its pass rate — DEFAULT REVERTED TO 0 (2026-07) to restore full. The must_avoid
# violation still costs the hard-severity hit (−35 → hard_score) + a mismatch tick; this dedicated
# term is extra. Re-enable (e.g. 22) via env MUST_AVOID_VIOLATION_PENALTY to sharpen no_split /
# no_negotiation ONLY after confirming full's splits reliably exempt (else it double-hits full).
MUST_AVOID_VIOLATION_PENALTY = safe_float(os.getenv("MUST_AVOID_VIOLATION_PENALTY", "0.0"), 0.0)
MEMBER_DEFICIENCY_PENALTY_CAP = safe_float(os.getenv("MEMBER_DEFICIENCY_PENALTY_CAP", "45.0"), 45.0)
# Concession dissatisfaction (Module 4's non-split resolution, see module5.apply_concession_resolution).
# When a must_avoid member CONCEDES a contested must-have (split unavailable → no_split), the term is
# not a hard veto (no −35, no unresolved penalty), but the conceder is NOT indifferent — they accept
# something they'd rather avoid. We charge a MODERATE hard-severity hit (< the 35 of a raw violation)
# so the outcome lands strictly between split (0, full) and an unresolved violation (35+penalty,
# no_negotiation). Keyed only on the presence of a recorded concession — never on which ablation runs.
CONCESSION_HARD_PENALTY = safe_float(os.getenv("CONCESSION_HARD_PENALTY", "20.0"), 20.0)
# Split-activity formal trigger predicate (item 3): a split is a core mechanism only when it
# resolves a real conflict, stays time/space-bounded, keeps transport+safety cost in check, and
# the family reconvenes the same day. Thresholds are env-tunable.
SPLIT_TRIGGER_MAX_BRANCH_DISTANCE_KM = safe_float(
    os.getenv("SPLIT_TRIGGER_MAX_BRANCH_DISTANCE_KM", "5.0"), 5.0
)
SPLIT_TRIGGER_MAX_BRANCH_DURATION_MINUTES = safe_float(
    os.getenv("SPLIT_TRIGGER_MAX_BRANCH_DURATION_MINUTES", "180"), 180.0
)
SPLIT_TRIGGER_MAX_EXTRA_TRANSPORT_COST = safe_float(
    os.getenv("SPLIT_TRIGGER_MAX_EXTRA_TRANSPORT_COST", "150"), 150.0
)
SPLIT_TRIGGER_PREF_MUST_CONFLICT_TYPES = {
    "direct_must_include_must_avoid_conflict",
    "must_include_vs_must_avoid_term_conflict",
    "preference_vs_must_avoid_term_conflict",
    "hard_constraint_feasibility_conflict",
    "direct_preference_conflict",
    "preference_conflict",
    "split_activity",
}
ROUTE_ZERO_ANOMALY_PENALTY = 10.0
MAX_ACCEPTABLE_REAL_WORLD_GROUNDING_PENALTY = 24.0
MAX_ACCEPTABLE_OBJECTIVE_EVAL_PENALTY = 24.0
# V7: post-grounding quality (nearest hotel / stitched transport / real restaurant names) is a soft diagnostic.
# Missing details should reduce scores and guide Module 7, but should not by itself veto a plan.
MAX_SOFT_GROUNDING_QUALITY_PENALTY_FOR_WARNING = 10.0
MAX_SOFT_PLACEHOLDER_ARTIFACTS_FOR_WARNING = 6
MIN_OBJECTIVE_FEASIBILITY_MICRO = 0.65
MIN_OBJECTIVE_RATIONALITY_MICRO = 0.75
# Tightened (env-overridable) so more first-attempt plans fail on REPAIRABLE dimensions
# (feasibility/rationality/satisfaction), giving the Repair (M7 backtracking) ablation
# visible headroom — repair regenerates with targeted feedback and recovers them.
MIN_FEASIBILITY_PASS_RATE = safe_float(os.getenv("MIN_FEASIBILITY_PASS_RATE", "0.90"), 0.90)
MIN_RATIONALITY_PASS_RATE = safe_float(os.getenv("MIN_RATIONALITY_PASS_RATE", "0.90"), 0.90)
# 82/72 (2026-07), down from the old 88/85 (which were set for the ~93-band easy dataset).
# Per-family diagnosis on the harder set: PASSING families sit ~90 group / ~89 min, so 82/72 is
# safely below them and does not let weak plans through; the low FULL pass rate was NOT a
# satisfaction-threshold problem (the earlier aggregate 83/66 was dragged down by the FAILING
# families) — it was conflict-resolution CAPACITY (M5 under-splitting + injected clashes >2),
# fixed in module5 (split-each) + the reverted must_avoid density. These two gates barely
# discriminate the ablations, so 82/72 costs no discrimination; the ablation gap lives on the
# unresolved-conflict / grounding / hard-constraint gates, which stay strict. Env-overridable.
STRICT_MIN_GROUP_SATISFACTION = safe_float(os.getenv("STRICT_MIN_GROUP_SATISFACTION", "82"), 82.0)
STRICT_MIN_MEMBER_SATISFACTION = safe_float(os.getenv("STRICT_MIN_MEMBER_SATISFACTION", "72"), 72.0)
# The Module 4 consensus label can read 'Partial'/'No Consensus' even when the final
# plan is unanimously accepted with good satisfaction.  Don't veto purely on the
# label when every member ultimately accepts and the least-happy member is fine —
# the real outcome signals (acceptance, min satisfaction) already gate this.
ACCEPT_NONFULL_CONSENSUS_WHEN_UNANIMOUS = os.getenv(
    "ACCEPT_NONFULL_CONSENSUS_WHEN_UNANIMOUS", "true"
).lower() in ["1", "true", "yes", "y"]
# Small tolerance on the budget pass (travel costs are estimates, not exact quotes).
BUDGET_PASS_TOLERANCE = safe_float(os.getenv("BUDGET_PASS_TOLERANCE", "1.10"), 1.10)
MIN_GROUP_CONSENSUS_SCORE = 0.0  # Group-consensus score is noisy when split activity is used; report it but do not hard-fail by default.
GENERIC_HOTEL_TERMS_FOR_EVAL = [
    "family-friendly hotel",
    "family friendly hotel",
    "hotel",
    "convenient area",
    "near main route",
    "家庭友好型酒店",
    "交通便利区域",
    "合适酒店",
    "住宿未注明",
]

# v56 (eval honesty): low real-world specificity must bound how high a plan can score.
# These are score *ceilings*, not subtractive penalties: a vague-but-tidy itinerary
# cannot report excellent satisfaction/feasibility. Defaults sit below the success
# thresholds in is_plan_passed_requirements (avg>=85, min>=82, feasibility_pass>=0.80),
# so a specificity-low plan is also excluded from the evaluated-only result set.
# All values are env-tunable and pre-cap numbers are reported alongside capped ones.
ENABLE_GROUNDING_SPECIFICITY_CAP = os.getenv(
    "ENABLE_GROUNDING_SPECIFICITY_CAP", "true"
).lower() in ["1", "true", "yes", "y"]
GROUNDING_CAP_RESTAURANT_SPECIFICITY_MIN = safe_float(
    os.getenv("GROUNDING_CAP_RESTAURANT_SPECIFICITY_MIN", "0.7"), 0.7
)
GROUNDING_FAIL_SATISFACTION_CAP = safe_float(
    os.getenv("GROUNDING_FAIL_SATISFACTION_CAP", "80.0"), 80.0
)
GROUNDING_FAIL_FEASIBILITY_CAP = safe_float(
    os.getenv("GROUNDING_FAIL_FEASIBILITY_CAP", "70.0"), 70.0
)
# Specificity->failure (items 4/5): every lunch/dinner must be a concrete restaurant or
# it is an explicit grounding failure that bounds the plan score (not a silent fallback).
GROUNDING_CAP_REQUIRE_ALL_MEALS_SPECIFIC = os.getenv(
    "GROUNDING_CAP_REQUIRE_ALL_MEALS_SPECIFIC", "true"
).lower() in ["1", "true", "yes", "y"]
# Tolerate the same number of ungrounded meals the generator is allowed to leave
# (MAX_UNGROUNDED_MEALS).  This keeps the full model from being falsely score-capped
# (and re-revised) for the 1–2 meals a thin city legitimately can't fill, while
# no_grounding (many ungrounded meals) is still capped — preserving the ablation gap.
ALLOWED_UNGROUNDED_MEALS = safe_int(os.getenv("MAX_UNGROUNDED_MEALS", "2"), 2)
# 'graded' scales the ceiling by how specific the plan actually is (honest + differentiates
# good-but-imperfect plans); 'binary' hard-caps to GROUNDING_FAIL_* whenever any issue exists.
GROUNDING_CAP_MODE = os.getenv("GROUNDING_CAP_MODE", "graded").lower()
GROUNDING_CAP_FLOOR = safe_float(os.getenv("GROUNDING_CAP_FLOOR", "60.0"), 60.0)


def combine_plan_text(plan_row) -> str:
    parts = [
        safe_text(plan_row.get("itinerary_text", "")),
        safe_text(plan_row.get("per_day_itinerary_json", "")),
        safe_text(plan_row.get("poi_tags", "")),
        safe_text(plan_row.get("meal_tags", "")),
        safe_text(plan_row.get("high_intensity_tags", "")),
        safe_text(plan_row.get("hotel_plan_json", "")),
        safe_text(plan_row.get("transportation_plan_json", "")),
        safe_text(plan_row.get("split_activities_json", "")),
        safe_text(plan_row.get("coordinator_summary", "")),
    ]
    return " ".join(parts).lower()


def normalize_hard_term_for_eval(term: str) -> str:
    text = safe_text(term).lower()
    text = text.replace("_", " ").replace("-", " ")
    text = text.replace("theme park", "theme parks")
    text = text.replace("museum", "museums")
    text = " ".join(text.split())
    return text


def hard_terms_overlap_for_eval(a: str, b: str) -> bool:
    ta = normalize_hard_term_for_eval(a)
    tb = normalize_hard_term_for_eval(b)
    if not ta or not tb:
        return False
    if ta == tb or ta in tb or tb in ta:
        return True
    groups = [
        {"museums", "art museums", "history museums", "museum"},
        {"theme parks", "amusement parks", "disneyland", "ocean park"},
        {"shopping", "shopping malls", "street markets", "markets"},
        {"crowded", "crowds", "busy places"},
        {"long queues", "queues", "waiting lines", "long waiting"},
        {"hiking", "climbing", "steep climbing", "high intensity", "long walking"},
        {"nightlife", "late night", "night market", "late return"},
    ]
    for group in groups:
        if any((x in ta for x in group)) and any((x in tb for x in group)):
            return True
    return False


def parse_unresolved_conflicts_from_plan_row(plan_row) -> List[Dict[str, Any]]:
    unresolved = parse_json_text(plan_row.get("unresolved_conflicts_json", "[]"), default=[])
    if isinstance(unresolved, list):
        return [x for x in unresolved if isinstance(x, dict)]
    return []


def compute_unresolved_conflict_penalty(plan_row) -> Dict[str, Any]:
    unresolved = parse_unresolved_conflicts_from_plan_row(plan_row)
    counts: Dict[str, int] = {}
    penalty = 0.0
    direct_count = 0
    for item in unresolved:
        conflict_type = safe_text(item.get("conflict_type", "unknown")) or "unknown"
        counts[conflict_type] = counts.get(conflict_type, 0) + 1
        weight = UNRESOLVED_CONFLICT_WEIGHTS.get(conflict_type, 5.0)
        penalty += weight
        if "direct" in conflict_type or "must_include_vs_must_avoid" in conflict_type:
            direct_count += 1
    return {
        "unresolved_conflict_count": len(unresolved),
        "unresolved_direct_conflict_count": direct_count,
        "unresolved_conflict_penalty": round(penalty, 4),
        "unresolved_conflict_type_counts_json": json.dumps(counts, ensure_ascii=False),
    }


def adjust_consensus_status_by_evaluation(
    original_status: Any,
    hard_constraint_pass_rate: float,
    acceptance_vote_ratio: float,
    unresolved_penalty: float,
    unresolved_direct_count: int,
    split_cost: float,
    mismatch_count: int,
    obvious_unfairness: bool = False,
) -> Dict[str, Any]:
    """Repair consensus status using core agreement signals.

    A plan should not be downgraded to Partial Consensus only because it uses
    role-explicit split activities.  Split cost is reported as a diagnostic, but
    Full Consensus is allowed when all members accept, no direct hard conflict
    remains, hard constraints pass, and there is no obvious unfairness.
    """
    original = safe_text(original_status) or "Unknown"
    reasons: List[str] = []
    diagnostic_reasons: List[str] = []

    if hard_constraint_pass_rate < 1.0:
        reasons.append("hard_constraint_not_fully_passed")
    if acceptance_vote_ratio < 1.0:
        reasons.append("not_all_members_accept")
    if obvious_unfairness:
        reasons.append("obvious_unfairness")

    if reasons:
        adjusted = "No Consensus"
    else:
        if unresolved_direct_count > 0:
            reasons.append("direct_unresolved_hard_conflict_exists")
        if unresolved_penalty > MAX_ACCEPTABLE_UNRESOLVED_CONFLICT_PENALTY:
            reasons.append("unresolved_conflict_penalty_high")
        if mismatch_count > MAX_ACCEPTABLE_PERSONALIZATION_MISMATCH_COUNT:
            reasons.append("personalization_mismatch_high")
        # Split activity cost is a quality warning, not a consensus veto when
        # everyone accepts and the split is role-explicit.
        if split_cost > MAX_ACCEPTABLE_SPLIT_ACTIVITY_COST:
            diagnostic_reasons.append("split_activity_cost_high_but_not_consensus_veto")
        adjusted = "Partial Consensus" if reasons else "Full Consensus"

    all_reasons = reasons + diagnostic_reasons
    return {
        "consensus_status_original": original,
        "consensus_status": adjusted,
        "consensus_forced_downgrade": original.lower() == "full consensus"
        and adjusted != "Full Consensus",
        "consensus_adjustment_reasons": json.dumps(all_reasons, ensure_ascii=False),
    }


def _split_activity_has_member_assignment(item: Dict[str, Any]) -> bool:
    participants = _as_text_list(item.get("participating_agents")) + _as_text_list(
        item.get("participating_roles")
    )
    non_participants = _as_text_list(item.get("non_participating_agents")) + _as_text_list(
        item.get("non_participating_roles")
    )
    return bool(participants and non_participants)


def _split_activity_has_reason(item: Dict[str, Any]) -> bool:
    reason_fields = [
        "reason",
        "rationale",
        "conflict_type",
        "resolution",
        "evaluation_note",
        "activity_or_term",
    ]
    return any((safe_text(item.get(field)) for field in reason_fields))


def evaluate_split_activity_trigger(item: Dict[str, Any]) -> Dict[str, Any]:
    """Formal trigger predicate for a split activity (item 3).

    A split is *justified* only when ALL of these hold:
      1. conflict_basis      - it resolves a direct preference or must-include/must-avoid conflict.
      2. time_compatible     - the two branches run in compatible time windows and neither is over-long.
      3. distance_ok         - branch locations are within SPLIT_TRIGGER_MAX_BRANCH_DISTANCE_KM.
      4. cost_safety_ok      - extra transport cost is bounded and there is no explicit safety risk
                               (e.g. an unsupervised minor).
      5. continuity_preserved- the family reconvenes the same day (regroup point + time given), so the
                               overall itinerary stays continuous.

    Each condition reports an ``ok`` and a ``known`` flag. Absent telemetry makes a condition unknown
    (recorded under ``split_trigger_missing_info``, not penalised), while a present-and-failing
    condition is an explicit ``violation``. This gates genuinely bad splits without nuking splits that
    merely lack the newer fields, and documents the field contract the generator should emit.
    """
    if not isinstance(item, dict):
        item = {}
    blob = json.dumps(item, ensure_ascii=False).lower()

    conflict_type = safe_text(
        item.get("conflict_type") or item.get("resolution_type") or item.get("conflict_id")
    )
    activity_or_term = safe_text(item.get("activity_or_term") or item.get("conflicting_term"))
    reason_text = " ".join(
        safe_text(item.get(f)) for f in ["reason", "rationale", "resolution", "evaluation_note"]
    ).lower()
    conflict_basis_known = bool(conflict_type or activity_or_term or reason_text)
    conflict_basis_ok = bool(
        conflict_type.lower() in SPLIT_TRIGGER_PREF_MUST_CONFLICT_TYPES
        or activity_or_term
        or any(
            k in reason_text
            for k in ["must_include", "must_avoid", "preference", "偏好", "冲突", "conflict"]
        )
    )

    tc = item.get("time_compatible")
    duration = safe_float(
        item.get("duration_minutes") or item.get("branch_duration_minutes"), np.nan
    )
    main_win = safe_text(
        item.get("main_time_window") or item.get("time_window") or item.get("participating_time")
    )
    alt_win = safe_text(item.get("alternative_time_window") or item.get("non_participating_time"))
    time_known = tc is not None or bool(main_win and alt_win) or (not pd.isna(duration))
    time_ok = bool_from_any(tc) if tc is not None else bool(main_win and alt_win)
    if (not pd.isna(duration)) and duration > SPLIT_TRIGGER_MAX_BRANCH_DURATION_MINUTES:
        time_ok = False
        time_known = True

    distance = safe_float(
        item.get("branch_distance_km") or item.get("distance_km") or item.get("split_distance_km"),
        np.nan,
    )
    distance_known = not pd.isna(distance)
    distance_ok = bool(distance_known and distance <= SPLIT_TRIGGER_MAX_BRANCH_DISTANCE_KM)

    extra_cost = safe_float(item.get("extra_transport_cost") or item.get("transport_cost"), np.nan)
    safety_risk = safe_text(item.get("safety_risk") or item.get("safety_level")).lower()
    unsupervised_minor = bool_from_any(item.get("unsupervised_minor")) or (
        "unsupervised" in blob and any(m in blob for m in ["child", "minor", "儿童", "孩子"])
    )
    cost_safety_known = (
        (not pd.isna(extra_cost))
        or bool(safety_risk)
        or ("safety" in blob)
        or ("unsupervised" in blob)
    )
    cost_violation = (
        not pd.isna(extra_cost)
    ) and extra_cost > SPLIT_TRIGGER_MAX_EXTRA_TRANSPORT_COST
    safety_violation = safety_risk in ["high", "unsafe", "high_risk"] or unsupervised_minor
    cost_safety_ok = not (cost_violation or safety_violation)

    regroup_point = safe_text(
        item.get("regroup_point")
        or item.get("meeting_point")
        or item.get("reconvene_point")
        or item.get("rejoin_point")
    )
    reconvene_time = safe_text(
        item.get("reconvene_time") or item.get("rejoin_time") or item.get("regroup_time")
    )
    same_day = bool_from_any(item.get("same_day")) or bool_from_any(item.get("reconvene_same_day"))
    continuity_known = (
        bool(regroup_point or reconvene_time)
        or (item.get("same_day") is not None)
        or (item.get("reconvene_same_day") is not None)
    )
    continuity_ok = bool(regroup_point) and (bool(reconvene_time) or same_day)

    conditions = {
        "conflict_basis": (conflict_basis_ok, conflict_basis_known),
        "time_compatible": (time_ok, time_known),
        "distance_ok": (distance_ok, distance_known),
        "cost_safety_ok": (cost_safety_ok, cost_safety_known),
        "continuity_preserved": (continuity_ok, continuity_known),
    }
    required = ["conflict_basis", "time_compatible", "distance_ok", "continuity_preserved"]
    violations = [k for k, (ok, known) in conditions.items() if known and not ok]
    missing = [k for k, (ok, known) in conditions.items() if not known]
    justified = (len(violations) == 0) and all(conditions[k][0] for k in required)
    return {
        "split_trigger_conflict_basis_ok": bool(conflict_basis_ok),
        "split_trigger_time_compatible_ok": bool(time_ok),
        "split_trigger_distance_ok": bool(distance_ok),
        "split_trigger_cost_safety_ok": bool(cost_safety_ok),
        "split_trigger_continuity_ok": bool(continuity_ok),
        "split_trigger_justified": bool(justified),
        "split_trigger_violations": violations,
        "split_trigger_missing_info": missing,
    }


def compute_split_activity_cost(plan_row) -> Dict[str, Any]:
    splits = parse_split_activities_from_plan_row(plan_row)
    weak_count = 0
    long_or_vague_count = 0
    missing_assignment_count = 0
    missing_reason_count = 0
    reasonable_count = 0
    unreasonable_count = 0
    trigger_justified_count = 0
    trigger_violation_count = 0
    trigger_missing_info_count = 0
    cost = 0.0
    weak_alternatives = {"rest", "free time", "nearby cafe", "hotel rest", "自由活动"}
    for item in splits:
        alt = safe_text(
            item.get("alternative_for_non_participants", "")
            or item.get("alternative_activity", "")
            or item.get("alternative", "")
        ).lower()
        duration = safe_float(item.get("duration_minutes", np.nan))
        has_assignment = _split_activity_has_member_assignment(item)
        has_reason = _split_activity_has_reason(item)
        trigger = evaluate_split_activity_trigger(item)
        if trigger["split_trigger_justified"]:
            trigger_justified_count += 1
        if trigger["split_trigger_violations"]:
            trigger_violation_count += 1
        if trigger["split_trigger_missing_info"]:
            trigger_missing_info_count += 1
        # “休息” is acceptable if it is made concrete, for example 公园/咖啡馆休息 + 集合点.
        weak_alt = (not alt) or (alt in weak_alternatives) or (alt == "休息")
        # NOTE: the length floor is for Chinese POI names — a concrete place can be 2–3 chars
        # ("外滩"/"颐和园"), so the old `len(alt) < 6` falsely flagged grounded parks like
        # "黔灵山公园"(5) as vague and pushed them into the penalized "unreasonable" bucket
        # (emptiness is already caught by weak_alt above). Floor is now 2 (only ~empty alts).
        long_or_vague = (
            ((not pd.isna(duration)) and duration >= 240)
            or len(alt) < 2
            or any(
                word in json.dumps(item, ensure_ascii=False)
                for word in ["部分成员", "其他成员", "有人", "其余成员"]
            )
        )
        issue_count = 0
        if weak_alt:
            weak_count += 1
            issue_count += 1
        if long_or_vague:
            long_or_vague_count += 1
            issue_count += 1
        if not has_assignment:
            missing_assignment_count += 1
            issue_count += 1
        if not has_reason:
            missing_reason_count += 1
            issue_count += 1
        if trigger["split_trigger_violations"]:
            # An explicit trigger violation (e.g. branches too far apart, unsafe, or no real
            # conflict) makes the split unjustified as a core mechanism, not just stylistically weak.
            issue_count += 1
        if issue_count == 0:
            reasonable_count += 1
            cost += REASONABLE_SPLIT_ACTIVITY_COST
        else:
            unreasonable_count += 1
            cost += UNREASONABLE_SPLIT_ACTIVITY_BASE_COST + SPLIT_ACTIVITY_ISSUE_COST * issue_count
    if not splits:
        assessment = "no_split_activity"
    elif unreasonable_count == 0:
        assessment = "reasonable_split_activity"
    elif reasonable_count > 0:
        assessment = "mixed_split_activity_quality"
    else:
        assessment = "unreasonable_split_activity"
    split_trigger_justified_rate = trigger_justified_count / len(splits) if splits else 0.0
    return {
        "split_activity_count": len(splits),
        "reasonable_split_activity_count": reasonable_count,
        "unreasonable_split_activity_count": unreasonable_count,
        "weak_split_alternative_count": weak_count,
        "long_or_vague_split_count": long_or_vague_count,
        "missing_split_assignment_count": missing_assignment_count,
        "missing_split_reason_count": missing_reason_count,
        "split_trigger_justified_count": int(trigger_justified_count),
        "split_trigger_violation_count": int(trigger_violation_count),
        "split_trigger_missing_info_count": int(trigger_missing_info_count),
        "split_trigger_justified_rate": round(split_trigger_justified_rate, 4),
        "split_activity_cost": round(cost, 4),
        "split_activity_assessment": assessment,
    }


def compute_route_realism_penalty(plan_row) -> Dict[str, Any]:
    day = safe_float(plan_row.get("day", 1), 1)
    route_distance = safe_float(plan_row.get("route_distance_km", np.nan))
    walking_distance = safe_float(plan_row.get("walking_distance_km", np.nan))
    daily_poi_count = safe_float(plan_row.get("daily_poi_count", np.nan))
    penalty = 0.0
    issues = []
    if (
        not pd.isna(route_distance)
        and route_distance <= 0.01
        and (not pd.isna(daily_poi_count))
        and (daily_poi_count > 0)
    ):
        penalty += ROUTE_ZERO_ANOMALY_PENALTY
        issues.append("route_distance_zero_with_pois")
    if (
        not pd.isna(route_distance)
        and (not pd.isna(walking_distance))
        and (route_distance > 0)
        and (walking_distance > route_distance * 1.5)
    ):
        penalty += 6.0
        issues.append("walking_distance_much_larger_than_route_distance")
    avg_route = average_per_day(route_distance, day)
    if not pd.isna(avg_route) and avg_route > MAX_REASONABLE_ROUTE_DISTANCE_PER_DAY:
        penalty += 8.0
        issues.append("avg_route_distance_too_high")
    return {
        "route_realism_penalty": round(penalty, 4),
        "route_realism_issues_json": json.dumps(issues, ensure_ascii=False),
    }


def _text_looks_generic_for_eval(value: Any, generic_terms: List[str]) -> bool:
    text = safe_text(value).lower().strip()
    if not text:
        return True
    # Avoid treating every real hotel name containing the word "hotel/酒店" as generic.
    exact_only_terms = {"hotel", "酒店"}
    for term in generic_terms:
        t = term.lower().strip()
        if not t:
            continue
        if t in exact_only_terms:
            if text == t:
                return True
            continue
        if text == t or t in text:
            return True
    return False


def _contains_concrete_metro_route(text: str) -> bool:
    lowered = safe_text(text).lower()
    if not lowered:
        return False
    has_metro_word = any((k in lowered for k in ["地铁", "港铁", "metro", "mtr", "subway"]))
    has_station_or_line = any((k in lowered for k in ["站", "号线", "线", "line"]))
    return has_metro_word and has_station_or_line


# -----------------------------------------------------------------------------
# V6: stricter grounding evaluation for nearest-qualified hotel and
# post-generation route stitching.
# -----------------------------------------------------------------------------


def _json_as_dict(value: Any) -> Dict[str, Any]:
    data = parse_json_text(value, default={})
    return data if isinstance(data, dict) else {}


def _json_as_list(value: Any) -> List[Any]:
    data = parse_json_text(value, default=[])
    return data if isinstance(data, list) else []


def _hotel_station_for_eval(
    hotel_plan: Dict[str, Any], transportation_plan: Optional[Dict[str, Any]] = None
) -> str:
    if not isinstance(hotel_plan, dict):
        hotel_plan = {}
    if not isinstance(transportation_plan, dict):
        transportation_plan = {}
    candidates = [
        hotel_plan.get("nearby_metro_station"),
        hotel_plan.get("nearby_station"),
        hotel_plan.get("nearest_metro_station"),
        hotel_plan.get("hotel_nearby_station"),
        transportation_plan.get("hotel_base_station"),
        transportation_plan.get("origin_station"),
    ]
    for value in candidates:
        text = safe_text(value)
        if text:
            return text
    return ""


def _transport_text_is_placeholder_for_eval(text: Any) -> bool:
    lowered = safe_text(text).lower().strip()
    if not lowered:
        return True
    # v20 structural route check: route text should describe a movement chain, not just a transport mode or template fragment.
    if re.search(
        r"(第\s*\d+\s*天.*活动区域|core\s+attraction|local\s+family\s+activity|nearest\s+(metro|mtr)|短路线|路线需求)",
        lowered,
    ):
        return True
    generic_mode_only = re.fullmatch(
        r"(短途)?(出租车|地铁|公交|步行|taxi|metro|mtr|subway|walk)(或|/)?((短途)?(出租车|地铁|公交|步行|taxi|metro|mtr|subway|walk))?",
        lowered,
    )
    if generic_mode_only:
        return True
    # A route should normally contain an origin/destination chain or station/line evidence.
    has_chain_marker = any(
        marker in lowered for marker in ["->", "→", "到", "至", "经", "换乘", "line", "号线", "站"]
    )
    has_place_pair = lowered.count("酒店") >= 1 and any(
        x in lowered for x in ["景点", "餐厅", "机场", "车站", "火车站", "高铁站"]
    )
    if not has_chain_marker and not has_place_pair and len(lowered) < 20:
        return True
    return False


def _extract_activity_name_for_route_eval(item: Any) -> str:
    if isinstance(item, dict):
        return (
            safe_text(item.get("poi_name"))
            or safe_text(item.get("activity"))
            or safe_text(item.get("name"))
            or safe_text(item.get("attraction"))
            or safe_text(item.get("place"))
        )
    return safe_text(item)


def _collect_day_route_stops_for_eval(day_item: Dict[str, Any]) -> List[str]:
    stops: List[str] = []
    if not isinstance(day_item, dict):
        return stops
    for section in ["morning", "afternoon", "evening", "optional_activities"]:
        raw = day_item.get(section, [])
        if isinstance(raw, dict):
            raw = [raw]
        if isinstance(raw, list):
            for item in raw:
                name = _extract_activity_name_for_route_eval(item)
                if name and name not in stops and not _is_generic_entity_name(name):
                    stops.append(name)
    for meal_key in ["breakfast", "lunch", "dinner"]:
        meal = day_item.get(meal_key, {})
        if isinstance(meal, dict):
            name = (
                safe_text(meal.get("restaurant_name"))
                or safe_text(meal.get("restaurant_area"))
                or safe_text(meal.get("location"))
                or safe_text(meal.get("area"))
            )
            if name and name not in stops:
                stops.append(name)
    return stops


def _route_contains_any_stop(route_text: str, stops: List[str]) -> bool:
    lowered = safe_text(route_text).lower()
    if not lowered or not stops:
        return False
    for stop in stops:
        norm = safe_text(stop).lower()
        if len(norm) >= 2 and (norm in lowered or lowered in norm):
            return True
    return False


def _route_item_complete_for_eval(
    route_item: Any, day_item: Dict[str, Any], hotel_station: str
) -> Dict[str, Any]:
    if not isinstance(route_item, dict):
        route_text = safe_text(route_item)
        origin = ""
        destination = ""
        stops = []
        cost = 0.0
        duration = 0.0
    else:
        route_text = safe_text(route_item.get("route")) or json.dumps(
            route_item, ensure_ascii=False
        )
        origin = safe_text(route_item.get("origin"))
        destination = safe_text(route_item.get("destination")) or safe_text(
            route_item.get("final_destination")
        )
        stops = route_item.get("stops", []) if isinstance(route_item.get("stops", []), list) else []
        cost = safe_float(
            route_item.get("estimated_cost_total") or route_item.get("estimated_cost"), 0
        )
        duration = safe_float(
            route_item.get("estimated_travel_time_minutes")
            or route_item.get("estimated_duration_minutes"),
            0,
        )
    expected_stops = _collect_day_route_stops_for_eval(day_item)
    stops_text = " ".join([safe_text(x) for x in stops])
    route_all_text = " ".join([route_text, origin, destination, stops_text])
    placeholder = _transport_text_is_placeholder_for_eval(route_all_text)
    has_origin = (
        bool(origin)
        or bool(hotel_station and hotel_station.lower() in route_all_text.lower())
        or any(k in route_all_text for k in ["酒店", "hotel"])
    )
    has_destination = bool(destination) or any(
        marker in route_text for marker in ["->", "→", "回酒店", "返回酒店"]
    )
    has_stop_match = _route_contains_any_stop(route_all_text, expected_stops)
    has_cost_or_time = (
        cost > 0 or duration > 0 or any(k in route_all_text for k in ["分钟", "min", "约", "元"])
    )
    complete = (
        (not placeholder) and has_origin and has_destination and has_stop_match and has_cost_or_time
    )
    stop_coverage = (
        1.0
        if not expected_stops
        else sum(1 for s in expected_stops if safe_text(s).lower() in route_all_text.lower())
        / max(len(expected_stops), 1)
    )
    return {
        "complete": bool(complete),
        "placeholder": bool(placeholder),
        "has_origin": bool(has_origin),
        "has_destination": bool(has_destination),
        "has_stop_match": bool(has_stop_match),
        "has_cost_or_time": bool(has_cost_or_time),
        "stop_coverage": round(stop_coverage, 4),
    }


def _looks_like_structural_placeholder(text: Any) -> bool:
    """Detect malformed template fragments using structural patterns, not long fixed word lists."""
    value = safe_text(text).strip()
    if not value:
        return True
    low = value.lower()
    if re.search(
        r"(第\s*\d+\s*天.*活动区域|core\s+attraction|local\s+family\s+activity|nearest\s+(metro|mtr)|短路线|路线需求)",
        low,
    ):
        return True
    if re.fullmatch(r"(未注明|未知|none|null|nan|n/a|na)", low):
        return True
    return False


def _strip_activity_action_words(text: Any) -> str:
    value = safe_text(text).strip()
    value = re.sub(r"^(上午|下午|晚上|早上|中午)[:：\s]*", "", value)
    value = re.sub(r"^(参观|游览|探索|体验|享受|品尝|观看|欣赏|前往|抵达|返回|在|于)", "", value).strip()
    value = re.sub(r"(观光|参观|游览|游玩|体验|打卡|拍照|散步|放松|校园游览|短时游览)$", "", value).strip()
    value = re.sub(r"(附近|周边|一带)$", "", value).strip(" 的，；。")
    return value


def _has_place_name_signal(text: Any) -> bool:
    """Approximate whether a text looks like a named place/POI using structural signals."""
    value = _strip_activity_action_words(text)
    if not value or _looks_like_structural_placeholder(value):
        return False
    if re.search(r"(早餐|午餐|晚餐|用餐|轻食|美食|小吃|火锅|餐厅|饭店|餐馆|料理|自助餐|餐饮|返程|退房|休息|整理行李)$", value):
        return False
    # Concrete place + action labels, such as 海港城购物, are acceptable because the place anchor is specific.
    if re.search(
        r".{2,}(城|街|路|商场|市场|夜市|广场|港|湾|小镇|古城|公园|乐园|馆|楼|塔|台|观景台|寺|庙|宫|教堂|大教堂|佛|大佛|中心|大学|岛|山|湖|江|桥|站|王国|景区)(购物|打卡|体验|表演|美食|散步|游览)?$",
        value,
    ):
        return True
    # Broad category labels with abstract modifiers are not concrete POIs even if they look noun-like.
    if re.fullmatch(
        r"(文化|历史|主题|当地|本地|互动|观景|美食|餐饮|自由|轻松|放松|现场|游乐).{0,4}(遗址|地标|公园|乐园|活动|体验|游览|美食|购物|表演|打卡|晚餐|午餐)?",
        value,
    ):
        return False
    if re.fullmatch(r".{0,6}(活动|体验|游览|探索|美食|小吃|购物|表演|打卡|观景|休闲|放松)$", value):
        return False
    if re.search(r"[A-Za-z0-9＋+&·•（）()]", value):
        return len(value) >= 2
    if re.search(
        r"(滩|园|乐园|公园|馆|博物馆|美术馆|艺术馆|楼|塔|台|观景台|寺|庙|宫|教堂|大教堂|佛|大佛|城|古城|小镇|广场|中心|大学|街|路|巷|港|湾|岛|山|湖|江|桥|站|市场|夜市|商场|王国|景区)$",
        value,
    ):
        return len(value) >= 2 and not re.fullmatch(
            r".{0,4}(活动|体验|游览|探索|美食|小吃|购物|表演|打卡|观景)$", value
        )
    if re.fullmatch(r"[一-鿿]{2,10}", value) and not re.search(
        r"(活动|体验|游览|探索|美食|小吃|购物|表演|打卡|观景|餐饮|餐厅|火锅|自助餐|返程|退房|休息)$", value
    ):
        return True
    return False


def _has_restaurant_name_signal(text: Any) -> bool:
    value = safe_text(text).strip()
    if not value or _looks_like_structural_placeholder(value):
        return False
    if re.fullmatch(r"(餐饮|用餐|午餐|晚餐|早餐|美食|小吃|自助|火锅|西餐|中餐|茶餐厅|餐厅|饭店|餐馆|料理)$", value):
        return False
    if re.search(r"[A-Za-z0-9＋+&·•（）()]", value):
        return len(value) >= 2
    # Short Chinese brand-like names without dining suffix can still be specific, e.g. 鼎泰丰 / 小南国.
    if re.fullmatch(r"[一-鿿]{2,8}", value) and not re.search(
        r"(餐饮|用餐|午餐|晚餐|早餐|美食|小吃|自助|火锅|西餐|中餐|茶餐厅|餐厅|饭店|餐馆|料理)$", value
    ):
        return True
    if re.search(r"(餐厅|饭店|餐馆|酒楼|酒家|食府|茶餐厅|小馆|厨房|料理|烤肉|火锅|小吃|面馆|粉店|烧烤|海鲜|咖啡|茶楼)$", value):
        prefix = re.sub(r"(餐厅|饭店|餐馆|酒楼|酒家|食府|茶餐厅|小馆|厨房|料理|烤肉|火锅|小吃|面馆|粉店|烧烤|海鲜|咖啡|茶楼)$", "", value)
        return len(prefix.strip()) >= 2 and not re.fullmatch(
            r"(当地|本地|家庭|友好|中档|高档|附近|景区|园区|公园|乐园|主题|自助|西餐|中餐|海鲜|火锅|餐饮)+", prefix.strip()
        )
    return False


def _web_source_count_for_eval(value: Any) -> int:
    if isinstance(value, str):
        return 1 if value.startswith(("http://", "https://")) else 0
    if isinstance(value, list):
        return sum(_web_source_count_for_eval(v) for v in value)
    if isinstance(value, dict):
        return sum(_web_source_count_for_eval(v) for v in value.values())
    return 0


def _restaurant_specificity_score_for_eval(meal_or_name: Any) -> float:
    """0-1 restaurant specificity score based on evidence and field structure, not long blacklists."""
    if isinstance(meal_or_name, dict):
        meal = meal_or_name
        name = safe_text(meal.get("restaurant_name")) or safe_text(meal.get("name"))
        area = (
            safe_text(meal.get("restaurant_area"))
            or safe_text(meal.get("location"))
            or safe_text(meal.get("area"))
        )
        rtype = safe_text(meal.get("restaurant_type")) or safe_text(meal.get("type"))
        status = safe_text(meal.get("restaurant_grounding_status"))
        explicit_flag = (
            bool_from_any(meal.get("specific_restaurant_found")) or status == "specific_name_found"
        )
        source_count = _web_source_count_for_eval(
            meal.get("source_urls") or meal.get("sources") or meal.get("source_url")
        )
    else:
        name = safe_text(meal_or_name)
        area = ""
        rtype = ""
        explicit_flag = False
        source_count = 0
    if not name:
        return 0.0
    if re.search(r"(返回|返程|退房|入住|抵达.*酒店|离开|整理行李|早餐后退房)", safe_text(name)):
        return 0.0
    if re.fullmatch(r"前往[\u4e00-\u9fff]{2,4}", safe_text(name)) and not re.search(
        r"(滩|园|乐园|公园|馆|博物馆|美术馆|艺术馆|楼|塔|台|观景台|寺|庙|宫|教堂|大教堂|佛|大佛|城|古城|小镇|广场|中心|大学|街|路|巷|港|湾|岛|山|湖|江|桥|站|市场|夜市|商场|王国|景区)$",
        re.sub(r"^前往", "", safe_text(name)),
    ):
        return 0.0
    score = 0.20
    if explicit_flag:
        score += 0.25
    if source_count > 0:
        score += 0.25
    name_signal = _has_restaurant_name_signal(name)
    if name_signal:
        score += 0.35
    if area and safe_text(name).lower() == safe_text(area).lower():
        score -= 0.25
    if rtype and safe_text(name).lower() == safe_text(rtype).lower():
        score -= 0.25
    if _looks_like_structural_placeholder(name):
        score -= 0.40
    # Source evidence can support a borderline name, but it should not upgrade a pure category label.
    if not name_signal:
        score = min(score, 0.45)
    return max(0.0, min(1.0, score))


def _restaurant_grounding_status_for_eval(meal_or_name: Any) -> str:
    """Three-way grounding status for one meal slot (items 2 + 4).

    - 'sourced_specific': concrete restaurant name AND a verifiable web source.
    - 'named_unsourced': concrete restaurant name but no verifiable source.
    - 'failure': no concrete restaurant (area/type fallback, placeholder, or empty).

    This separates "specific but unverifiable" from "specific and verifiable", and turns
    every non-concrete meal into an explicit failure instead of a silent area fallback.
    """
    if isinstance(meal_or_name, dict):
        source_count = _web_source_count_for_eval(
            meal_or_name.get("source_urls")
            or meal_or_name.get("sources")
            or meal_or_name.get("source_url")
        )
    else:
        source_count = 0
    if _restaurant_specificity_score_for_eval(meal_or_name) < 0.55:
        return "failure"
    return "sourced_specific" if source_count > 0 else "named_unsourced"


def _restaurant_specificity_stats_for_eval(per_day: List[Any]) -> Dict[str, Any]:
    total_meals = 0
    specific_count = 0
    area_fallback_count = 0
    sourced_specific_count = 0
    named_unsourced_count = 0
    failure_count = 0
    scores: List[float] = []
    for day_item in per_day if isinstance(per_day, list) else []:
        if not isinstance(day_item, dict):
            continue
        for meal_key in ["lunch", "dinner"]:
            meal = day_item.get(meal_key, {})
            if not isinstance(meal, dict):
                continue
            total_meals += 1
            score = _restaurant_specificity_score_for_eval(meal)
            scores.append(score)
            status = _restaurant_grounding_status_for_eval(meal)
            if status == "failure":
                area_fallback_count += 1
                failure_count += 1
            else:
                specific_count += 1
                if status == "sourced_specific":
                    sourced_specific_count += 1
                else:
                    named_unsourced_count += 1
    rate = specific_count / total_meals if total_meals else 0.0
    sourced_rate = sourced_specific_count / total_meals if total_meals else 0.0
    failure_rate = failure_count / total_meals if total_meals else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0
    return {
        "restaurant_total_meal_count": int(total_meals),
        "specific_restaurant_count": int(specific_count),
        "restaurant_sourced_specific_count": int(sourced_specific_count),
        "restaurant_named_unsourced_count": int(named_unsourced_count),
        "restaurant_area_fallback_count": int(area_fallback_count),
        "restaurant_failure_count": int(failure_count),
        "restaurant_specificity_rate": round(rate, 4),
        "restaurant_sourced_specificity_rate": round(sourced_rate, 4),
        "restaurant_failure_rate": round(failure_rate, 4),
        "restaurant_meal_grounding_pass": bool(total_meals > 0 and failure_count == 0),
        "restaurant_specificity_avg_score": round(avg_score, 4),
    }


def _activity_name_for_specificity_eval(item: Any) -> str:
    if isinstance(item, dict):
        return (
            safe_text(item.get("poi_name"))
            or safe_text(item.get("activity"))
            or safe_text(item.get("name"))
            or safe_text(item.get("attraction"))
            or safe_text(item.get("place"))
        )
    return safe_text(item)


def _activity_specificity_score_for_eval(item: Any) -> float:
    if isinstance(item, dict):
        name = _activity_name_for_specificity_eval(item)
        source_count = _web_source_count_for_eval(
            item.get("source_urls") or item.get("sources") or item.get("source_url")
        )
        has_geo = bool(
            safe_text(item.get("address"))
            or safe_text(item.get("area"))
            or safe_text(item.get("nearest_station"))
        )
        has_ticket = (
            safe_float(item.get("ticket_price") or item.get("ticket_price_per_person"), 0) > 0
        )
    else:
        name = safe_text(item)
        source_count = 0
        has_geo = False
        has_ticket = False
    if not name:
        return 0.0
    if re.search(r"(返回|返程|退房|入住|抵达.*酒店|离开|整理行李|早餐后退房)", safe_text(name)):
        return 0.0
    if re.fullmatch(r"前往[\u4e00-\u9fff]{2,4}", safe_text(name)) and not re.search(
        r"(滩|园|乐园|公园|馆|博物馆|美术馆|艺术馆|楼|塔|台|观景台|寺|庙|宫|教堂|大教堂|佛|大佛|城|古城|小镇|广场|中心|大学|街|路|巷|港|湾|岛|山|湖|江|桥|站|市场|夜市|商场|王国|景区)$",
        re.sub(r"^前往", "", safe_text(name)),
    ):
        return 0.0
    score = 0.20
    if _has_place_name_signal(name):
        score += 0.45
    if source_count > 0:
        score += 0.20
    if has_geo:
        score += 0.10
    if has_ticket:
        score += 0.10
    if _looks_like_structural_placeholder(name):
        score -= 0.40
    norm = _strip_activity_action_words(name)
    if re.search(r"(早餐|午餐|晚餐|用餐|轻食|返程|退房|整理行李|休息)$", norm):
        score -= 0.35
    if re.search(
        r".{2,}(城|街|路|商场|市场|夜市|广场|港|湾|小镇|古城|公园|乐园|馆|楼|塔|台|观景台|寺|庙|宫|教堂|大教堂|佛|大佛|中心|大学|岛|山|湖|江|桥|站|王国|景区)(购物|打卡|体验|表演|美食|散步|游览)?$",
        norm,
    ):
        score += 0.20
    elif re.fullmatch(
        r"(文化|历史|主题|当地|本地|互动|观景|美食|餐饮|自由|轻松|放松|现场|游乐).{0,4}(遗址|地标|公园|乐园|活动|体验|游览|美食|购物|表演|打卡|晚餐|午餐)?",
        norm,
    ):
        score -= 0.45
    elif re.fullmatch(r".{0,6}(活动|体验|游览|探索|美食|小吃|购物|表演|打卡|观景|休闲|放松)$", norm):
        score -= 0.25
    return max(0.0, min(1.0, score))


def _generic_activity_count_for_eval(per_day: List[Any]) -> int:
    count = 0
    if not isinstance(per_day, list):
        return 0
    for day_item in per_day:
        if not isinstance(day_item, dict):
            continue
        for section in ["morning", "afternoon", "evening"]:
            raw = day_item.get(section, [])
            if isinstance(raw, dict):
                raw = [raw]
            if not isinstance(raw, list):
                raw = [raw] if safe_text(raw) else []
            for item in raw:
                # Generic if the heuristic score is low OR it is a bare category
                # label (公园/购物街/自然公园) that grounding should have made specific.
                name = _activity_name_for_specificity_eval(item)
                if _activity_specificity_score_for_eval(item) < 0.45 or is_generic_place_label(
                    name
                ):
                    count += 1
        # Split-activity branches must also name a concrete place, not a bare category.
        splits = day_item.get("split_activities", [])
        if isinstance(splits, dict):
            splits = [splits]
        for sp in splits if isinstance(splits, list) else []:
            if isinstance(sp, dict) and any(
                is_generic_place_label(sp.get(fld))
                for fld in ("activity", "alternative_for_non_participants")
            ):
                count += 1
    return count


def _quality_marker_metrics_for_eval(markers: Dict[str, Any]) -> Dict[str, Any]:
    """Read Module-5 quality markers instead of re-inferring all specificity from text.

    The markers are generated after Module 5 grounding and contain explicit flags such as
    activity:specific, restaurant:area_or_type_fallback, transport:complete_route_chain.
    Module 6 uses them as structured evidence and only falls back to text heuristics when absent.
    """
    if not isinstance(markers, dict) or not markers.get("marker_schema_version"):
        return {"module5_quality_markers_present": False}
    summary = markers.get("summary", {}) if isinstance(markers.get("summary"), dict) else {}
    issue_flags = (
        summary.get("issue_flags", []) if isinstance(summary.get("issue_flags", []), list) else []
    )
    activity_markers = (
        markers.get("activity_markers", [])
        if isinstance(markers.get("activity_markers", []), list)
        else []
    )
    restaurant_markers = (
        markers.get("restaurant_markers", [])
        if isinstance(markers.get("restaurant_markers", []), list)
        else []
    )
    transport_markers = (
        markers.get("transport_markers", [])
        if isinstance(markers.get("transport_markers", []), list)
        else []
    )
    day_markers = (
        markers.get("day_markers", []) if isinstance(markers.get("day_markers", []), list) else []
    )
    vague_activity_count = sum(
        1
        for x in activity_markers
        if isinstance(x, dict) and not bool(x.get("is_concrete_activity"))
    )
    source_missing_activity_count = sum(
        1
        for x in activity_markers
        if isinstance(x, dict) and not bool(x.get("has_grounding_source"))
    )
    restaurant_fallback_count = sum(
        1 for x in restaurant_markers if isinstance(x, dict) and bool(x.get("area_fallback_used"))
    )
    incomplete_transport_count = sum(
        1
        for x in transport_markers
        if isinstance(x, dict) and not bool(x.get("is_complete_route_chain"))
    )
    incomplete_day_count = sum(
        1
        for x in day_markers
        if isinstance(x, dict) and not bool(x.get("has_at_least_one_concrete_core_activity"))
    )
    score = safe_float(summary.get("module5_truthfulness_score"), np.nan)
    marker_pass_raw = summary.get("module5_quality_markers_pass")
    marker_pass = (
        bool(marker_pass_raw)
        if marker_pass_raw not in [None, ""]
        else (not pd.isna(score) and score >= 75.0)
    )
    return {
        "module5_quality_markers_present": True,
        "module5_truthfulness_score": round(score, 4) if not pd.isna(score) else np.nan,
        "module5_quality_markers_pass": bool(marker_pass),
        "module5_marker_issue_count": int(len(issue_flags)),
        "module5_marker_issue_flags_json": json.dumps(issue_flags, ensure_ascii=False),
        "module5_marker_activity_concreteness_rate": safe_float(
            summary.get("activity_concreteness_rate"), 0.0
        ),
        "module5_marker_source_backed_activity_rate": safe_float(
            summary.get("source_backed_activity_rate"), 0.0
        ),
        "module5_marker_restaurant_specificity_rate": safe_float(
            summary.get("restaurant_specificity_rate"), 0.0
        ),
        "module5_marker_transport_chain_coverage": safe_float(
            summary.get("transport_chain_coverage"), 0.0
        ),
        "module5_marker_complete_day_rate": safe_float(summary.get("complete_day_rate"), 0.0),
        "module5_marker_vague_activity_count": int(vague_activity_count),
        "module5_marker_source_missing_activity_count": int(source_missing_activity_count),
        "module5_marker_restaurant_fallback_count": int(restaurant_fallback_count),
        "module5_marker_incomplete_transport_count": int(incomplete_transport_count),
        "module5_marker_incomplete_day_count": int(incomplete_day_count),
    }


def compute_post_generation_grounding_metrics(plan_row) -> Dict[str, Any]:
    """Evaluate whether grounding happened after the full itinerary, not before it.

    This catches common false positives: generic hotels not selected around route
    anchors, daily transport that is only "short taxi", and restaurant fallback
    descriptions without real names.
    """
    hotel_plan = _json_as_dict(plan_row.get("hotel_plan_json", "{}"))
    transportation_plan = _json_as_dict(plan_row.get("transportation_plan_json", "{}"))
    per_day = _json_as_list(plan_row.get("per_day_itinerary_json", "[]"))
    issues: List[str] = []
    penalty = 0.0
    itinerary_quality_markers = _json_as_dict(plan_row.get("itinerary_quality_markers_json", "{}"))
    marker_metrics = _quality_marker_metrics_for_eval(itinerary_quality_markers)

    hotel_name = safe_text(hotel_plan.get("hotel_name") or hotel_plan.get("hotel_name_or_type"))
    hotel_station = _hotel_station_for_eval(hotel_plan, transportation_plan)
    hotel_reason_text = " ".join(
        [
            safe_text(hotel_plan.get("hotel_selection_reason")),
            safe_text(hotel_plan.get("selection_reason")),
            safe_text(hotel_plan.get("reason")),
            safe_text(hotel_plan.get("location_reason")),
            safe_text(hotel_plan.get("distance_note")),
            safe_text(hotel_plan.get("route_fit_note")),
            safe_text(hotel_plan.get("verification_note")),
        ]
    ).lower()
    hotel_distance_value = safe_float(
        hotel_plan.get("distance_to_itinerary_core_km")
        or hotel_plan.get("distance_to_main_route_km")
        or hotel_plan.get("avg_distance_to_anchor_pois_km"),
        np.nan,
    )
    hotel_has_nearest_logic = (
        not pd.isna(hotel_distance_value) and hotel_distance_value <= 8.0
    ) or any(
        k in hotel_reason_text
        for k in ["最近", "距离", "主景点", "主行程", "路线", "近", "near", "closest", "itinerary", "anchor"]
    )
    hotel_nearest_requirement_pass = bool(hotel_name and hotel_station and hotel_has_nearest_logic)
    if not hotel_nearest_requirement_pass:
        penalty += 2.0
        issues.append("soft_hotel_not_verified_as_nearest_qualified_to_itinerary")

    daily_transport = (
        transportation_plan.get("daily_local_transport", [])
        if isinstance(transportation_plan.get("daily_local_transport", []), list)
        else []
    )
    route_lookup: Dict[int, Any] = {}
    for idx, item in enumerate(daily_transport):
        if isinstance(item, dict):
            day_num = safe_int(item.get("day"), idx + 1)
            if day_num:
                route_lookup[day_num] = item
    complete_route_count = 0
    route_placeholder_count = 0
    stop_coverages: List[float] = []
    day_count = len(per_day) if isinstance(per_day, list) else 0
    for idx, day_item in enumerate(per_day):
        if not isinstance(day_item, dict):
            continue
        day_num = safe_int(day_item.get("day"), idx + 1)
        route_item = route_lookup.get(day_num)
        if route_item is None:
            # Fall back to day-level transport field only for diagnosing old outputs.
            route_item = {
                "day": day_num,
                "route": "；".join([safe_text(x) for x in day_item.get("transportation", [])])
                if isinstance(day_item.get("transportation", []), list)
                else safe_text(day_item.get("transportation")),
                "origin": hotel_station or "酒店",
                "destination": "酒店",
            }
        route_eval = _route_item_complete_for_eval(route_item, day_item, hotel_station)
        if route_eval["complete"]:
            complete_route_count += 1
        if route_eval["placeholder"]:
            route_placeholder_count += 1
        stop_coverages.append(safe_float(route_eval.get("stop_coverage"), 0))
    daily_route_chain_coverage = complete_route_count / day_count if day_count > 0 else 0.0
    daily_route_stop_coverage = sum(stop_coverages) / len(stop_coverages) if stop_coverages else 0.0
    post_transport_chain_pass = (
        day_count > 0
        and daily_route_chain_coverage >= 0.8
        and daily_route_stop_coverage >= 0.5
        and route_placeholder_count == 0
    )
    if not post_transport_chain_pass:
        penalty += 3.0
        issues.append("soft_post_generation_transport_route_chain_incomplete")
    if route_placeholder_count > 0:
        penalty += min(3.0, route_placeholder_count * 1.0)
        issues.append(f"soft_transport_placeholder_route_count={route_placeholder_count}")

    restaurant_stats = _restaurant_specificity_stats_for_eval(per_day)
    restaurant_specificity_rate = safe_float(
        restaurant_stats.get("restaurant_specificity_rate"), 0.0
    )
    restaurant_area_fallback_count = safe_int(
        restaurant_stats.get("restaurant_area_fallback_count"), 0
    )
    restaurant_specificity_pass = restaurant_specificity_rate >= 0.7
    if restaurant_area_fallback_count > 0:
        penalty += min(4.0, restaurant_area_fallback_count * 0.5)
        issues.append(f"soft_restaurant_area_fallback_count={restaurant_area_fallback_count}")
    if not restaurant_specificity_pass:
        penalty += 2.0
        issues.append("soft_restaurants_not_specific_enough_after_grounding")

    whole_text = " ".join(
        [
            safe_text(plan_row.get("itinerary_text")),
            safe_text(plan_row.get("per_day_itinerary_json")),
            safe_text(plan_row.get("hotel_plan_json")),
            safe_text(plan_row.get("transportation_plan_json")),
            safe_text(plan_row.get("online_grounding_json")),
        ]
    ).lower()
    generic_activity_count = _generic_activity_count_for_eval(per_day)
    if generic_activity_count > 0:
        penalty += min(5.0, generic_activity_count * 1.0)
        issues.append(f"soft_generic_activity_label_count={generic_activity_count}")

    placeholder_artifact_count = 0
    if re.search(
        r"(core\s+attraction|local\s+family\s+activity|第\s*\d+\s*天.*活动区域|nearest\s+(metro|mtr)|短路线|路线需求)",
        whole_text,
    ):
        placeholder_artifact_count += 1
    if placeholder_artifact_count > 0:
        penalty += min(3.0, placeholder_artifact_count * 0.75)
        issues.append(f"soft_placeholder_artifact_count={placeholder_artifact_count}")

    if marker_metrics.get("module5_quality_markers_present"):
        marker_score = safe_float(marker_metrics.get("module5_truthfulness_score"), 100.0)
        if not bool(marker_metrics.get("module5_quality_markers_pass", True)):
            issues.append("soft_module5_quality_markers_not_pass")
            penalty += max(
                1.0, min(6.0, (75.0 - marker_score) / 10.0 if marker_score < 75.0 else 1.0)
            )
        marker_vague = safe_int(marker_metrics.get("module5_marker_vague_activity_count"), 0)
        if marker_vague > 0:
            issues.append(f"soft_module5_marker_vague_activity_count={marker_vague}")
        marker_rest_fallback = safe_int(
            marker_metrics.get("module5_marker_restaurant_fallback_count"), 0
        )
        if marker_rest_fallback > 0 and marker_rest_fallback > restaurant_area_fallback_count:
            issues.append(f"soft_module5_marker_restaurant_fallback_count={marker_rest_fallback}")
        marker_incomplete_transport = safe_int(
            marker_metrics.get("module5_marker_incomplete_transport_count"), 0
        )
        if marker_incomplete_transport > 0:
            issues.append(
                f"soft_module5_marker_incomplete_transport_count={marker_incomplete_transport}"
            )

    marker_pass_for_quality = (
        bool(marker_metrics.get("module5_quality_markers_pass", True))
        if marker_metrics.get("module5_quality_markers_present")
        else True
    )
    itinerary_grounding_quality_pass = (
        hotel_nearest_requirement_pass
        and post_transport_chain_pass
        and restaurant_specificity_pass
        and placeholder_artifact_count == 0
        and generic_activity_count == 0
        and marker_pass_for_quality
    )
    grounding_quality_soft_score = clamp_score(100.0 - penalty * 5.0)
    return {
        "itinerary_grounding_quality_pass": bool(itinerary_grounding_quality_pass),
        "hotel_nearest_requirement_pass": bool(hotel_nearest_requirement_pass),
        "post_transport_chain_pass": bool(post_transport_chain_pass),
        "daily_route_chain_coverage": round(daily_route_chain_coverage, 4),
        "daily_route_stop_coverage": round(daily_route_stop_coverage, 4),
        "transport_placeholder_route_count": int(route_placeholder_count),
        "restaurant_specificity_rate": round(restaurant_specificity_rate, 4),
        "restaurant_specificity_pass": bool(restaurant_specificity_pass),
        "specific_restaurant_count": int(restaurant_stats.get("specific_restaurant_count", 0)),
        "restaurant_sourced_specific_count": int(
            restaurant_stats.get("restaurant_sourced_specific_count", 0)
        ),
        "restaurant_named_unsourced_count": int(
            restaurant_stats.get("restaurant_named_unsourced_count", 0)
        ),
        "restaurant_sourced_specificity_rate": safe_float(
            restaurant_stats.get("restaurant_sourced_specificity_rate"), 0.0
        ),
        "restaurant_failure_count": int(restaurant_stats.get("restaurant_failure_count", 0)),
        "restaurant_failure_rate": safe_float(restaurant_stats.get("restaurant_failure_rate"), 0.0),
        "restaurant_meal_grounding_pass": bool(
            restaurant_stats.get("restaurant_meal_grounding_pass")
        ),
        "restaurant_total_meal_count": int(restaurant_stats.get("restaurant_total_meal_count", 0)),
        "restaurant_area_fallback_count": int(restaurant_area_fallback_count),
        "restaurant_specificity_avg_score": safe_float(
            restaurant_stats.get("restaurant_specificity_avg_score"), 0.0
        ),
        "placeholder_artifact_count": int(placeholder_artifact_count),
        "generic_activity_label_count": int(generic_activity_count),
        "activity_specificity_failure_count": int(generic_activity_count),
        "activity_grounding_pass": bool(generic_activity_count == 0),
        "grounding_quality_penalty": round(penalty, 4),
        "grounding_quality_soft_score": round(grounding_quality_soft_score, 4),
        "grounding_quality_issues_json": json.dumps(
            list(dict.fromkeys(issues)), ensure_ascii=False
        ),
        **marker_metrics,
    }


def compute_real_world_grounding_metrics(plan_row) -> Dict[str, Any]:
    hotel_plan = parse_json_text(plan_row.get("hotel_plan_json", "{}"), default={})
    transportation_plan = parse_json_text(
        plan_row.get("transportation_plan_json", "{}"), default={}
    )
    per_day = parse_json_text(plan_row.get("per_day_itinerary_json", "[]"), default=[])
    grounding = parse_json_text(plan_row.get("real_world_grounding_json", "{}"), default={})
    online_grounding = parse_json_text(plan_row.get("online_grounding_json", "{}"), default={})
    cost_breakdown = parse_json_text(plan_row.get("cost_breakdown_json", "{}"), default={})
    issues = []
    penalty = 0.0
    hotel_name = ""
    hotel_price = 0.0
    hotel_sources = []
    if isinstance(hotel_plan, dict):
        hotel_name = safe_text(hotel_plan.get("hotel_name")) or safe_text(
            hotel_plan.get("hotel_name_or_type")
        )
        hotel_price = safe_float(hotel_plan.get("estimated_price_per_night"), 0)
        hotel_sources = (
            hotel_plan.get("source_urls", [])
            if isinstance(hotel_plan.get("source_urls", []), list)
            else []
        )
    hotel_price_estimated = (
        bool(hotel_plan.get("hotel_price_estimated")) if isinstance(hotel_plan, dict) else False
    )
    hotel_realism_pass = bool(hotel_name) and (
        not _text_looks_generic_for_eval(hotel_name, GENERIC_HOTEL_TERMS_FOR_EVAL)
    )
    if not hotel_realism_pass:
        if hotel_price_estimated:
            # Budget-driven estimate (city DB had no affordable real hotel) — honest
            # behavior, so a small soft note instead of the hallucination penalty.
            penalty += 2.0
            issues.append("budget_estimated_hotel_used")
        else:
            penalty += 8.0
            issues.append("missing_specific_real_hotel_name")
    if hotel_price <= 0:
        penalty += 3.0
        issues.append("missing_hotel_price_per_night")
    hotel_station = ""
    if isinstance(hotel_plan, dict):
        hotel_station = safe_text(hotel_plan.get("nearby_metro_station"))
    if not hotel_station:
        penalty += 3.0
        issues.append("missing_hotel_nearby_transport_station")
    transport_text = ""
    if isinstance(transportation_plan, dict):
        transport_text = json.dumps(transportation_plan, ensure_ascii=False)
    metro_route_realism_pass = _contains_concrete_metro_route(transport_text) or any(
        (k in transport_text for k in ["公交", "巴士", "出租车", "高铁", "火车", "flight", "train"])
    )
    if not metro_route_realism_pass:
        penalty += 8.0
        issues.append("missing_concrete_public_transport_route")
    daily_route_count = 0
    day_count = 0
    if isinstance(per_day, list):
        for day_item in per_day:
            if not isinstance(day_item, dict):
                continue
            day_count += 1
            daily_transport_text = json.dumps(
                day_item.get("transportation", []), ensure_ascii=False
            )
            daily_transport_text += " " + safe_text(day_item.get("transportation_note", ""))
            if _contains_concrete_metro_route(daily_transport_text) or any(
                (k in daily_transport_text for k in ["公交", "巴士", "出租车", "高铁", "火车"])
            ):
                daily_route_count += 1
    daily_route_coverage = daily_route_count / day_count if day_count > 0 else 0.0
    if day_count > 0 and daily_route_coverage < 0.8:
        penalty += 4.0
        issues.append("daily_transport_route_coverage_too_low")
    source_urls = []
    if isinstance(grounding, dict):
        raw_sources = grounding.get("source_urls", [])
        if isinstance(raw_sources, list):
            source_urls.extend([safe_text(x) for x in raw_sources if safe_text(x)])
    structured_grounding_complete = bool(
        isinstance(grounding, dict) and grounding.get("structured_grounding_complete")
    )
    online_status = "missing"
    if isinstance(online_grounding, dict):
        online_status = safe_text(online_grounding.get("search_status")) or "missing"
        raw_sources = online_grounding.get("source_urls", [])
        if isinstance(raw_sources, list):
            source_urls.extend([safe_text(x) for x in raw_sources if safe_text(x)])
        raw_issues = online_grounding.get("grounding_issues", [])
        if isinstance(raw_issues, list) and raw_issues:
            issues.extend([safe_text(x) for x in raw_issues if safe_text(x)])
            penalty += min(4.0, 1.5 * len(raw_issues))
    source_urls.extend([safe_text(x) for x in hotel_sources if safe_text(x)])
    source_urls = list(dict.fromkeys(source_urls))
    online_source_count = len(source_urls)
    online_grounding_used = (
        online_status in ["success", "partial_or_unverified"] and online_source_count > 0
    ) or structured_grounding_complete
    if not online_grounding_used:
        penalty += 4.0
        issues.append("missing_online_grounding_sources")
    cost_total = 0.0
    cost_positive_categories = 0
    if isinstance(cost_breakdown, dict):
        cost_total = safe_float(cost_breakdown.get("total_cost"), 0)
        for key in [
            "hotel_cost",
            "intercity_transport_cost",
            "local_transport_cost",
            "meal_cost",
            "attraction_ticket_cost",
        ]:
            if safe_float(cost_breakdown.get(key), 0) > 0:
                cost_positive_categories += 1
    cost_breakdown_complete = cost_total > 0 and cost_positive_categories >= 3
    if not cost_breakdown_complete:
        penalty += 5.0
        issues.append("cost_breakdown_incomplete")
    # Budget relief used a low-tier (旅馆/招待所/民宿…) hotel to fit the budget:
    # keep the plan feasible but deduct for the lowered accommodation quality.
    if isinstance(cost_breakdown, dict) and bool_from_any(
        cost_breakdown.get("budget_relief_low_tier_hotel")
    ):
        penalty += 6.0
        issues.append("budget_relief_low_tier_hotel_used")
    if isinstance(grounding, dict):
        raw_issues = grounding.get("issues", [])
        if isinstance(raw_issues, list) and raw_issues:
            issues.extend([safe_text(x) for x in raw_issues if safe_text(x)])
            penalty += min(4.0, 1.5 * len(raw_issues))
    post_grounding_quality = compute_post_generation_grounding_metrics(plan_row)
    try:
        post_issues = json.loads(post_grounding_quality.get("grounding_quality_issues_json", "[]"))
        if isinstance(post_issues, list):
            issues.extend([safe_text(x) for x in post_issues if safe_text(x)])
    except Exception:
        pass
    penalty += safe_float(post_grounding_quality.get("grounding_quality_penalty"), 0.0)
    # V7: pass/fail only checks the basic presence of grounded evidence.
    # Nearest-hotel proof, stitched daily routes, and restaurant specificity are soft diagnostics: they reduce scores
    # through grounding_quality_penalty instead of vetoing the whole plan.
    real_world_grounding_pass = (
        hotel_realism_pass
        and metro_route_realism_pass
        and (daily_route_coverage >= 0.6)
        and (online_grounding_used or structured_grounding_complete)
        and cost_breakdown_complete
    )
    return {
        "hotel_realism_pass": hotel_realism_pass,
        "hotel_price_estimated": bool(hotel_price_estimated),
        "metro_route_realism_pass": metro_route_realism_pass,
        "daily_metro_route_coverage": round(daily_route_coverage, 4),
        **post_grounding_quality,
        "real_world_grounding_pass": real_world_grounding_pass,
        "real_world_grounding_penalty": round(penalty, 4),
        "real_world_grounding_issues_json": json.dumps(
            list(dict.fromkeys(issues)), ensure_ascii=False
        ),
        "online_grounding_used": bool(online_grounding_used),
        "online_grounding_status": online_status,
        "online_source_count": int(online_source_count),
        "cost_breakdown_complete": bool(cost_breakdown_complete),
    }


def _is_nonempty_json_list(value: Any) -> bool:
    return isinstance(value, list) and any(
        (safe_text(x) or isinstance(x, (dict, list)) for x in value)
    )


def _normal_text_key(value: Any) -> str:
    text = safe_text(value).lower()
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def _is_generic_entity_name(value: Any) -> bool:
    text = _normal_text_key(value)
    if not text:
        return True
    generic_terms = [
        "poi",
        "attraction",
        "sightseeing",
        "museum",
        "park",
        "restaurant",
        "local restaurant",
        "family friendly restaurant",
        "hotel",
        "shopping",
        "scenic spot",
        "tourist attraction",
        "景点",
        "餐厅",
        "本地餐厅",
        "家庭友好餐厅",
        "酒店",
        "公园",
        "博物馆",
        "商场",
        "观景台",
        "合适餐厅",
    ]
    if text in generic_terms:
        return True
    return len(text) < 3


def _collect_day_activities(day_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    activities = []
    for section in ["morning", "afternoon", "evening", "optional_activities"]:
        raw = day_item.get(section, [])
        if isinstance(raw, dict):
            raw = [raw]
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    activities.append(item)
    return activities


def _collect_day_meals(day_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    meals = []
    for key in ["breakfast", "lunch", "dinner"]:
        meal = day_item.get(key)
        if isinstance(meal, dict):
            meals.append(meal)
    return meals


def _extract_activity_name(activity: Dict[str, Any]) -> str:
    return (
        safe_text(activity.get("poi_name"))
        or safe_text(activity.get("activity"))
        or safe_text(activity.get("poi_type"))
    )


def _extract_meal_name_or_type(meal: Dict[str, Any]) -> str:
    return (
        safe_text(meal.get("restaurant_name"))
        or safe_text(meal.get("restaurant_type"))
        or safe_text(meal.get("name"))
        or safe_text(meal.get("type"))
    )


def _price_level_to_numeric_for_eval(value: Any) -> int:
    text = _normal_text_key(value)
    if any((k in text for k in ["low", "cheap", "budget", "低", "便宜", "经济"])):
        return 1
    if any((k in text for k in ["high", "expensive", "luxury", "premium", "高", "贵", "奢"])):
        return 3
    return 2


def _activity_duration_minutes(activity: Dict[str, Any]) -> Optional[float]:
    for key in ["estimated_duration_minutes", "duration_minutes", "duration"]:
        value = activity.get(key)
        number = safe_float(value, np.nan)
        if not pd.isna(number) and number > 0:
            return number
    text = safe_text(activity.get("time")) or safe_text(activity.get("time_range"))
    if text:
        m = re.search("(\\d{1,2}:\\d{2})\\s*[-–]\\s*(\\d{1,2}:\\d{2})", text)
        if m:
            start = time_to_minutes(m.group(1))
            end = time_to_minutes(m.group(2))
            if start is not None and end is not None and (end > start):
                return float(end - start)
    return None


def _is_activity_duration_reasonable(activity: Dict[str, Any]) -> bool:
    duration = _activity_duration_minutes(activity)
    if duration is None:
        return True
    name = _normal_text_key(_extract_activity_name(activity))
    if any(
        (
            k in name
            for k in ["disney", "迪士尼", "theme park", "wild animal", "动物园", "polarland", "乐园"]
        )
    ):
        return 120 <= duration <= 540
    return 30 <= duration <= 240


def _day_has_specific_transport(day_item: Dict[str, Any]) -> bool:
    text = json.dumps(day_item.get("transportation", []), ensure_ascii=False)
    text += " " + safe_text(day_item.get("transportation_note", ""))
    return _contains_concrete_metro_route(text)


def compute_objective_style_metrics(
    plan_row, real_world_grounding: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if not USE_OBJECTIVE_EVALUATION:
        return {
            "objective_within_reference": True,
            "objective_complete_information": True,
            "objective_complete_information_rate": 1.0,
            "objective_per_day_complete_rate": 1.0,
            "objective_daily_transport_coverage": 1.0,
            "objective_hotel_complete": True,
            "objective_transport_complete": True,
            "objective_attraction_concrete_rate": 1.0,
            "objective_restaurant_concrete_rate": 1.0,
            "objective_feasibility_micro": 1.0,
            "objective_feasibility_macro": True,
            "objective_defined_budget_limit": True,
            "objective_diverse_restaurants": True,
            "objective_diverse_attractions": True,
            "objective_reasonable_meal_prices": True,
            "objective_appropriate_visit_duration": True,
            "objective_route_efficiency_pass": True,
            "objective_rationality_micro": 1.0,
            "objective_rationality_macro": True,
            "objective_final_pass": True,
            "objective_average_route_distance": np.nan,
            "objective_average_walking_distance": np.nan,
            "objective_route_distance_to_walking_ratio": np.nan,
            "objective_duplicate_attraction_count": 0,
            "objective_duplicate_restaurant_count": 0,
            "objective_duration_fail_count": 0,
            "objective_daily_activity_overload_count": 0,
            "objective_evaluation_penalty": 0.0,
            "objective_feasibility_penalty": 0.0,
            "objective_rationality_penalty": 0.0,
            "objective_issues_json": json.dumps(
                ["objective_evaluation_disabled"], ensure_ascii=False
            ),
        }
    hotel_plan = parse_json_text(plan_row.get("hotel_plan_json", "{}"), default={})
    transportation_plan = parse_json_text(
        plan_row.get("transportation_plan_json", "{}"), default={}
    )
    per_day = parse_json_text(plan_row.get("per_day_itinerary_json", "[]"), default=[])
    day_count_expected = safe_int(plan_row.get("day", 0), 0)
    total_cost = safe_float(plan_row.get("total_cost", np.nan))
    budget_limit = safe_float(plan_row.get("budget_limit", np.nan))
    walking_distance = safe_float(plan_row.get("walking_distance_km", np.nan))
    route_distance = safe_float(plan_row.get("route_distance_km", np.nan))
    if real_world_grounding is None:
        real_world_grounding = compute_real_world_grounding_metrics(plan_row)
    issues = []
    penalty = 0.0
    hotel_has_name = isinstance(hotel_plan, dict) and bool(
        safe_text(hotel_plan.get("hotel_name") or hotel_plan.get("hotel_name_or_type"))
    )
    hotel_has_area = isinstance(hotel_plan, dict) and bool(safe_text(hotel_plan.get("hotel_area")))
    hotel_has_station = isinstance(hotel_plan, dict) and bool(
        _hotel_station_for_eval(hotel_plan, transportation_plan)
    )
    hotel_has_price = isinstance(hotel_plan, dict) and bool(
        safe_text(hotel_plan.get("price_level"))
    )
    hotel_complete = hotel_has_name and hotel_has_area and hotel_has_station and hotel_has_price
    transport_has_mode = isinstance(transportation_plan, dict) and bool(
        safe_text(transportation_plan.get("main_transport_mode"))
    )
    transport_has_strategy = isinstance(transportation_plan, dict) and bool(
        safe_text(transportation_plan.get("route_strategy"))
    )
    transport_has_routes = isinstance(transportation_plan, dict) and (
        _is_nonempty_json_list(transportation_plan.get("metro_routes", []))
        or _is_nonempty_json_list(transportation_plan.get("daily_local_transport", []))
    )
    transport_complete = transport_has_mode and transport_has_strategy and transport_has_routes
    days_complete_count = 0
    daily_transport_count = 0
    attraction_names = []
    restaurant_names = []
    duration_fail_count = 0
    activity_count = 0
    generic_activity_count = 0
    generic_meal_count = 0
    daily_activity_overload_count = 0
    if isinstance(per_day, list):
        for day_item in per_day:
            if not isinstance(day_item, dict):
                continue
            activities = _collect_day_activities(day_item)
            meals = _collect_day_meals(day_item)
            has_core_activity = len(activities) > 0
            has_meal = len(meals) >= 1
            has_transport = _day_has_specific_transport(day_item)
            has_return_time = bool(safe_text(day_item.get("estimated_return_time")))
            if has_core_activity and has_meal and has_transport and has_return_time:
                days_complete_count += 1
            if has_transport:
                daily_transport_count += 1
            if len(activities) > 3:
                daily_activity_overload_count += 1
            for activity in activities:
                activity_count += 1
                name = _extract_activity_name(activity)
                if name:
                    attraction_names.append(_normal_text_key(name))
                if _is_generic_entity_name(name):
                    generic_activity_count += 1
                if not _is_activity_duration_reasonable(activity):
                    duration_fail_count += 1
            for meal in meals:
                meal_name = _extract_meal_name_or_type(meal)
                if meal_name:
                    restaurant_names.append(_normal_text_key(meal_name))
                if _is_generic_entity_name(meal_name):
                    generic_meal_count += 1
    actual_day_count = len(per_day) if isinstance(per_day, list) else 0
    day_count_match = True
    if day_count_expected > 0:
        day_count_match = actual_day_count == day_count_expected
    per_day_complete_rate = days_complete_count / actual_day_count if actual_day_count > 0 else 0.0
    daily_transport_coverage = (
        daily_transport_count / actual_day_count if actual_day_count > 0 else 0.0
    )
    cost_complete = not pd.isna(total_cost) and (not pd.isna(budget_limit)) and (budget_limit > 0)
    budget_pass = bool(cost_complete and total_cost <= budget_limit)
    complete_checks = [
        hotel_complete,
        transport_complete,
        actual_day_count > 0,
        day_count_match,
        per_day_complete_rate >= 0.8,
        cost_complete,
    ]
    objective_complete_information_rate = sum(complete_checks) / len(complete_checks)
    objective_complete_information = objective_complete_information_rate >= 0.85
    if not hotel_complete:
        issues.append("tp_hotel_information_incomplete")
        penalty += 5.0
    if not transport_complete:
        issues.append("tp_transportation_information_incomplete")
        penalty += 5.0
    if not day_count_match:
        issues.append("tp_day_count_mismatch")
        penalty += 3.0
    if per_day_complete_rate < 0.8:
        issues.append("tp_per_day_information_incomplete")
        penalty += 4.0
    if not cost_complete:
        issues.append("tp_cost_or_budget_missing")
        penalty += 4.0
    # V7: these reference checks are coarse pass/fail gates. Fine-grained grounding quality is scored separately.
    hotel_reference_pass = bool(real_world_grounding.get("hotel_realism_pass")) and hotel_has_name
    transport_reference_pass = (
        bool(real_world_grounding.get("metro_route_realism_pass"))
        and daily_transport_coverage >= 0.6
    )
    attraction_concrete_rate = (
        1.0 if activity_count == 0 else 1.0 - generic_activity_count / max(activity_count, 1)
    )
    meal_concrete_rate = (
        1.0 if not restaurant_names else 1.0 - generic_meal_count / max(len(restaurant_names), 1)
    )
    attraction_reference_pass = attraction_concrete_rate >= 0.6
    restaurant_reference_pass = meal_concrete_rate >= 0.45
    objective_within_reference = (
        hotel_reference_pass
        and transport_reference_pass
        and attraction_reference_pass
        and restaurant_reference_pass
    )
    if not hotel_reference_pass:
        issues.append("tp_hotel_not_grounded")
        penalty += 2.0
    if not attraction_reference_pass:
        issues.append("tp_attractions_too_generic")
        penalty += 2.0
    if not restaurant_reference_pass:
        issues.append("tp_restaurants_too_generic_or_rule_fallback")
        penalty += 2.0
    if not transport_reference_pass:
        issues.append("tp_daily_transport_routes_not_specific_or_not_stitched")
        penalty += 2.0
    unique_attractions = {x for x in attraction_names if x}
    unique_restaurants = {x for x in restaurant_names if x}
    duplicate_attraction_count = max(0, len(attraction_names) - len(unique_attractions))
    duplicate_restaurant_count = max(0, len(restaurant_names) - len(unique_restaurants))
    diverse_attractions = (
        len(unique_attractions) >= min(2, len(attraction_names)) and duplicate_attraction_count == 0
    )
    diverse_restaurants = len(unique_restaurants) >= min(
        2, len(restaurant_names)
    ) and duplicate_restaurant_count <= max(1, actual_day_count // 2)
    meal_price_values = []
    if isinstance(per_day, list):
        for day_item in per_day:
            if isinstance(day_item, dict):
                for meal in _collect_day_meals(day_item):
                    meal_price_values.append(
                        _price_level_to_numeric_for_eval(
                            meal.get(
                                "price_level", plan_row.get("restaurant_price_level", "medium")
                            )
                        )
                    )
    if not meal_price_values:
        meal_price_values = [
            _price_level_to_numeric_for_eval(plan_row.get("restaurant_price_level", "medium"))
        ]
    avg_meal_price_level = sum(meal_price_values) / len(meal_price_values)
    reasonable_meal_prices = avg_meal_price_level <= 2.6 or budget_pass
    appropriate_visit_duration = duration_fail_count == 0 and daily_activity_overload_count == 0
    avg_route_distance = average_per_day(
        route_distance, day_count_expected or actual_day_count or 1
    )
    avg_walking_distance = average_per_day(
        walking_distance, day_count_expected or actual_day_count or 1
    )
    route_efficiency_pass = True
    route_distance_to_walking_ratio = np.nan
    if (
        not pd.isna(avg_route_distance)
        and (not pd.isna(avg_walking_distance))
        and (avg_walking_distance > 0)
    ):
        route_distance_to_walking_ratio = avg_route_distance / avg_walking_distance
        route_efficiency_pass = 1.0 <= route_distance_to_walking_ratio <= 18.0
    if (
        not pd.isna(avg_route_distance)
        and avg_route_distance > MAX_REASONABLE_ROUTE_DISTANCE_PER_DAY
    ):
        route_efficiency_pass = False
    if not diverse_attractions:
        issues.append("tp_attractions_not_diverse")
        penalty += 3.0
    if not diverse_restaurants:
        issues.append("tp_restaurants_not_diverse")
        penalty += 3.0
    if not reasonable_meal_prices:
        issues.append("tp_meal_prices_not_reasonable")
        penalty += 3.0
    if not appropriate_visit_duration:
        issues.append("tp_visit_duration_or_daily_load_not_reasonable")
        penalty += 4.0
    if not route_efficiency_pass:
        issues.append("tp_route_distance_not_efficient")
        penalty += 4.0
    # V41: use a granular objective feasibility score instead of an all-or-nothing gate.
    # The previous version collapsed to 0.0 whenever either reference grounding or
    # complete-information failed, which made Module 7 repeatedly regenerate even
    # when most concrete evidence was present.
    feasibility_checks = [
        hotel_reference_pass,
        transport_reference_pass,
        attraction_reference_pass,
        restaurant_reference_pass,
        objective_complete_information,
        cost_complete,
    ]
    objective_feasibility_micro = sum((int(x) for x in feasibility_checks)) / len(
        feasibility_checks
    )
    objective_feasibility_macro = objective_feasibility_micro >= MIN_OBJECTIVE_FEASIBILITY_MICRO
    rationality_checks = [
        budget_pass,
        diverse_restaurants,
        diverse_attractions,
        reasonable_meal_prices,
        appropriate_visit_duration,
        route_efficiency_pass,
    ]
    objective_rationality_micro = sum((int(x) for x in rationality_checks)) / len(
        rationality_checks
    )
    objective_rationality_macro = objective_rationality_micro >= MIN_OBJECTIVE_RATIONALITY_MICRO
    objective_final_pass = objective_feasibility_macro and objective_rationality_macro
    feasibility_penalty = 0.0
    if not objective_feasibility_macro:
        feasibility_penalty += 6.0
    rationality_penalty = 0.0
    if not objective_rationality_macro:
        rationality_penalty += 5.0
    return {
        "objective_within_reference": bool(objective_within_reference),
        "objective_complete_information": bool(objective_complete_information),
        "objective_complete_information_rate": round(objective_complete_information_rate, 4),
        "objective_per_day_complete_rate": round(per_day_complete_rate, 4),
        "objective_daily_transport_coverage": round(daily_transport_coverage, 4),
        "objective_hotel_complete": bool(hotel_complete),
        "objective_transport_complete": bool(transport_complete),
        "objective_attraction_concrete_rate": round(attraction_concrete_rate, 4),
        "objective_restaurant_concrete_rate": round(meal_concrete_rate, 4),
        "objective_feasibility_micro": round(objective_feasibility_micro, 4),
        "objective_feasibility_macro": bool(objective_feasibility_macro),
        "objective_defined_budget_limit": bool(budget_pass),
        "objective_diverse_restaurants": bool(diverse_restaurants),
        "objective_diverse_attractions": bool(diverse_attractions),
        "objective_reasonable_meal_prices": bool(reasonable_meal_prices),
        "objective_appropriate_visit_duration": bool(appropriate_visit_duration),
        "objective_route_efficiency_pass": bool(route_efficiency_pass),
        "objective_rationality_micro": round(objective_rationality_micro, 4),
        "objective_rationality_macro": bool(objective_rationality_macro),
        "objective_final_pass": bool(objective_final_pass),
        "objective_average_route_distance": round(avg_route_distance, 4)
        if not pd.isna(avg_route_distance)
        else np.nan,
        "objective_average_walking_distance": round(avg_walking_distance, 4)
        if not pd.isna(avg_walking_distance)
        else np.nan,
        "objective_route_distance_to_walking_ratio": round(route_distance_to_walking_ratio, 4)
        if not pd.isna(route_distance_to_walking_ratio)
        else np.nan,
        "objective_duplicate_attraction_count": int(duplicate_attraction_count),
        "objective_duplicate_restaurant_count": int(duplicate_restaurant_count),
        "objective_duration_fail_count": int(duration_fail_count),
        "objective_daily_activity_overload_count": int(daily_activity_overload_count),
        "objective_evaluation_penalty": round(penalty, 4),
        "objective_feasibility_penalty": round(feasibility_penalty, 4),
        "objective_rationality_penalty": round(rationality_penalty, 4),
        "objective_issues_json": json.dumps(list(dict.fromkeys(issues)), ensure_ascii=False),
    }


def compute_constraint_satisfiability_typing(plan_row) -> Dict[str, Any]:
    if not USE_CONSTRAINT_SATISFIABILITY_TYPING:
        return {
            "constraint_satisfiability_overall": "disabled",
            "fully_satisfiable_constraint_count": 0,
            "split_satisfiable_constraint_count": 0,
            "unsatisfiable_constraint_count": 0,
            "split_satisfiable_constraint_rate": 0.0,
            "unsatisfiable_constraint_rate": 0.0,
            "constraint_satisfiability_pass": True,
            "constraint_satisfiability_typing_json": json.dumps(
                {"disabled": True}, ensure_ascii=False
            ),
        }
    data = parse_json_text(plan_row.get("constraint_satisfiability_json", "{}"), default={})
    if not isinstance(data, dict):
        data = {}
    fully_count = safe_int(data.get("fully_satisfiable_count"), 0)
    split_count = safe_int(data.get("split_satisfiable_count"), 0)
    unsat_count = safe_int(data.get("unsatisfiable_count"), 0)
    if not data:
        splits = parse_split_activities_from_plan_row(plan_row)
        unresolved = parse_unresolved_conflicts_from_plan_row(plan_row)
        split_count = len(splits)
        unsat_count = len(unresolved)
        fully_count = 1 if unsat_count == 0 and split_count == 0 else 0
        if unsat_count > 0:
            overall = "has_unsatisfiable_constraints"
        elif split_count > 0:
            overall = "split_satisfiable_plan"
        else:
            overall = "fully_satisfiable_plan"
        data = {
            "framework_component": "Constraint Satisfiability Typing",
            "overall_satisfiability": overall,
            "fully_satisfiable_count": fully_count,
            "split_satisfiable_count": split_count,
            "unsatisfiable_count": unsat_count,
            "typed_constraints": [],
        }
    total = max(fully_count + split_count + unsat_count, 1)
    return {
        "constraint_satisfiability_overall": safe_text(
            data.get("overall_satisfiability", "unknown")
        )
        or "unknown",
        "fully_satisfiable_constraint_count": fully_count,
        "split_satisfiable_constraint_count": split_count,
        "unsatisfiable_constraint_count": unsat_count,
        "split_satisfiable_constraint_rate": round(split_count / total, 4),
        "unsatisfiable_constraint_rate": round(unsat_count / total, 4),
        "constraint_satisfiability_pass": unsat_count == 0,
        "constraint_satisfiability_typing_json": json.dumps(data, ensure_ascii=False),
    }


def compute_fairness_aware_group_objective(
    avg_score: float,
    min_score: float,
    hard_constraint_severity_score: float,
    accepted_vote_ratio: float,
    max_min_fairness: float,
    unresolved_penalty: float,
    split_cost: float,
    route_penalty: float,
    mismatch_count: int,
    obvious_unfairness: bool,
) -> Dict[str, Any]:
    avg_component = safe_float(avg_score, 0)
    min_component = safe_float(min_score, 0)
    hard_component = safe_float(hard_constraint_severity_score, 0)
    acceptance_component = safe_float(accepted_vote_ratio, 0) * 100
    fairness_component = safe_float(max_min_fairness, 0) * 100
    unfairness_penalty = 10.0 if obvious_unfairness else 0.0
    hard_violation_penalty = max(0.0, 100.0 - hard_component) * 0.4
    penalty = (
        1.1 * safe_float(unresolved_penalty, 0)
        + 0.5 * safe_float(split_cost, 0)
        + 0.9 * safe_float(route_penalty, 0)
        + 1.3 * safe_int(mismatch_count, 0)
        + unfairness_penalty
        + hard_violation_penalty
    )
    objective = (
        0.35 * avg_component
        + 0.3 * min_component
        + 0.15 * acceptance_component
        + 0.1 * hard_component
        + 0.1 * fairness_component
        - penalty
    )
    components = {
        "average_satisfaction_component": round(avg_component, 4),
        "minimum_satisfaction_protection_component": round(min_component, 4),
        "acceptance_ratio_component": round(acceptance_component, 4),
        "hard_constraint_component": round(hard_component, 4),
        "fairness_component": round(fairness_component, 4),
        "unresolved_conflict_penalty": round(safe_float(unresolved_penalty, 0), 4),
        "split_coordination_cost": round(safe_float(split_cost, 0), 4),
        "route_and_grounding_penalty": round(safe_float(route_penalty, 0), 4),
        "personalization_mismatch_count": safe_int(mismatch_count, 0),
        "obvious_unfairness_penalty": round(unfairness_penalty, 4),
        "hard_violation_penalty": round(hard_violation_penalty, 4),
    }
    return {
        "fairness_aware_group_objective": round(clamp_score(objective), 4),
        "fairness_aware_group_objective_components_json": json.dumps(
            components, ensure_ascii=False
        ),
        "fairness_aware_objective_formula": "0.35*avg_satisfaction + 0.30*min_satisfaction + 0.15*acceptance + 0.10*hard_constraint + 0.10*fairness - penalties",
    }


def compute_member_level_mismatch_metrics(member_eval_df: pd.DataFrame) -> Dict[str, Any]:
    mismatch_count = 0
    low_satisfaction_count = 0
    strict_low_satisfaction_count = 0
    rejected_member_count = 0
    hard_violation_count = 0
    for _, row in member_eval_df.iterrows():
        score = safe_float(row.get("individual_satisfaction_score"), 100)
        if score < STRICT_LOW_SATISFACTION_THRESHOLD:
            low_satisfaction_count += 1
        if score < STRICT_MIN_SATISFACTION_THRESHOLD:
            strict_low_satisfaction_count += 1
        if not bool_from_any(row.get("accepted_vote")):
            rejected_member_count += 1
        if not bool_from_any(row.get("hard_constraint_pass")):
            hard_violation_count += 1
        for json_col in [
            "intensity_issues",
            "food_issues",
            "budget_issues",
            "hard_constraint_failed_reasons",
        ]:
            parsed = parse_json_list(row.get(json_col, "[]"))
            if parsed:
                mismatch_count += len(parsed)
        if safe_float(row.get("preference_coverage_rate"), 1.0) < 1.0:
            mismatch_count += 1
        if safe_float(row.get("must_have_satisfaction_rate"), 1.0) < 1.0:
            mismatch_count += 1
    return {
        "personalization_mismatch_count": int(mismatch_count),
        "low_satisfaction_member_count": int(low_satisfaction_count),
        "strict_low_satisfaction_member_count": int(strict_low_satisfaction_count),
        "rejected_member_count": int(rejected_member_count),
        "hard_violation_member_count": int(hard_violation_count),
    }


def compute_adjusted_group_score(
    avg_score: float,
    min_score: float,
    hard_severity: float,
    feasibility_score: float,
    rationality_score: float,
    acceptance_vote_ratio: float,
    max_min_fairness: float,
    unresolved_penalty: float,
    split_cost: float,
    route_penalty: float,
    mismatch_count: int,
) -> float:
    if pd.isna(avg_score):
        return np.nan
    base = (
        0.12 * safe_float(avg_score, 0)
        + 0.28 * safe_float(min_score, 0)
        + 0.22 * safe_float(hard_severity, 0)
        + 0.1 * safe_float(feasibility_score, 0)
        + 0.08 * safe_float(rationality_score, 0)
        + 0.12 * safe_float(acceptance_vote_ratio, 0) * 100
        + 0.08 * safe_float(max_min_fairness, 0) * 100
    )
    penalty = (
        1.2 * safe_float(unresolved_penalty, 0)
        + 0.65 * safe_float(split_cost, 0)
        + 1.0 * safe_float(route_penalty, 0)
        + 1.5 * safe_float(mismatch_count, 0)
    )
    return round(clamp_score(base - penalty), 4)


def parse_split_activities_from_plan_row(plan_row) -> List[Dict[str, Any]]:
    split_activities = parse_json_text(plan_row.get("split_activities_json", "[]"), default=[])
    if isinstance(split_activities, list):
        return [x for x in split_activities if isinstance(x, dict)]
    return []


def is_avoid_exempted_by_split_activity(persona_row, plan_row, avoid_term: str) -> bool:
    split_activities = parse_split_activities_from_plan_row(plan_row)
    if not split_activities:
        return False
    family_id = safe_int(persona_row.get("family_id"), 0)
    member_id = safe_int(persona_row.get("member_id"), 0)
    agent_id = (
        safe_text(persona_row.get("agent_id", "")) or f"family_{family_id}_member_{member_id}"
    )
    role = safe_text(persona_row.get("role", ""))
    age_stage = safe_text(persona_row.get("age_stage", ""))
    identifiers = {
        agent_id.lower(),
        f"family_{family_id}_member_{member_id}".lower(),
        f"member_{member_id}".lower(),
        role.lower(),
        age_stage.lower(),
    }
    for activity in split_activities:
        activity_text = json.dumps(activity, ensure_ascii=False).lower()
        if not hard_terms_overlap_for_eval(avoid_term, activity_text):
            continue
        non_participants = []
        for key in ["non_participating_agents", "non_participating_roles", "avoid_agents"]:
            vals = activity.get(key, [])
            if isinstance(vals, list):
                non_participants.extend(vals)
            else:
                non_participants.append(vals)
        non_participants = {safe_text(x).lower() for x in non_participants if safe_text(x)}
        if identifiers & non_participants:
            return True
    return False


def parse_concession_from_plan_row(plan_row) -> List[Dict[str, Any]]:
    concessions = parse_json_text(plan_row.get("concession_resolutions_json", "[]"), default=[])
    if isinstance(concessions, list):
        return [x for x in concessions if isinstance(x, dict)]
    return []


def is_avoid_conceded_by_negotiation(persona_row, plan_row, avoid_term: str) -> bool:
    """True when THIS member conceded ``avoid_term`` via Module 4's non-split resolution.

    Mirrors ``is_avoid_exempted_by_split_activity`` but reads ``concession_resolutions`` and matches
    the member against ``conceding_agents``/``conceding_roles``. A conceded avoid is not a hard veto,
    but the caller still charges ``CONCESSION_HARD_PENALTY`` (unlike a split, which is cost-free).
    """
    concessions = parse_concession_from_plan_row(plan_row)
    if not concessions:
        return False
    family_id = safe_int(persona_row.get("family_id"), 0)
    member_id = safe_int(persona_row.get("member_id"), 0)
    agent_id = (
        safe_text(persona_row.get("agent_id", "")) or f"family_{family_id}_member_{member_id}"
    )
    identifiers = {
        agent_id.lower(),
        f"family_{family_id}_member_{member_id}".lower(),
        f"member_{member_id}".lower(),
        safe_text(persona_row.get("role", "")).lower(),
        safe_text(persona_row.get("age_stage", "")).lower(),
    }
    for item in concessions:
        term = safe_text(item.get("activity_or_term", ""))
        if term and not hard_terms_overlap_for_eval(avoid_term, term):
            continue
        conceders = []
        for key in ["conceding_agents", "conceding_roles"]:
            vals = item.get(key, [])
            conceders.extend(vals if isinstance(vals, list) else [vals])
        conceders = {safe_text(x).lower() for x in conceders if safe_text(x)}
        if identifiers & conceders:
            return True
    return False


def build_agent_weight_lookup(agents_json_path: Path) -> Dict[str, Dict[str, float]]:
    if not agents_json_path.exists():
        print(f"[Module 6] 没有找到 persona_agents.json，将使用默认权重: {agents_json_path}")
        return {}
    agents = load_json_file(agents_json_path)
    lookup = {}
    for agent in agents:
        agent_id = safe_text(agent.get("agent_id", ""))
        if not agent_id:
            continue
        weights = agent.get("preference_weights", {})
        if isinstance(weights, dict):
            lookup[agent_id] = weights
    return lookup


def get_member_agent_id(persona_row) -> str:
    family_id = safe_int(persona_row.get("family_id"), 0)
    member_id = safe_int(persona_row.get("member_id"), 0)
    return f"family_{family_id}_member_{member_id}"


def get_default_weights() -> Dict[str, float]:
    return {
        "interest_weight": 0.2,
        "physical_weight": 0.2,
        "budget_weight": 0.2,
        "food_weight": 0.15,
        "pace_weight": 0.15,
        "hard_constraint_weight": 0.1,
    }


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    default_weights = get_default_weights()
    merged = default_weights.copy()
    if isinstance(weights, dict):
        for key, value in weights.items():
            try:
                merged[key] = float(value)
            except Exception:
                pass
    total = sum(merged.values())
    if total <= 0:
        return default_weights
    return {key: value / total for key, value in merged.items()}


def get_valid_family_ids(persona_df: pd.DataFrame, plans_df: pd.DataFrame) -> List[int]:
    valid_family_ids = []
    skipped_info = []
    persona_family_ids = set(persona_df["family_id"].astype(int).unique())
    plan_family_ids = set(plans_df["family_id"].astype(int).unique())
    for family_id in sorted(plan_family_ids):
        reasons = []
        member_count = 0
        if family_id not in persona_family_ids:
            reasons.append("missing_persona")
        else:
            family_personas = persona_df[persona_df["family_id"].astype(int) == family_id]
            member_count = len(family_personas)
            if member_count < 3 or member_count > 4:
                reasons.append(f"invalid_member_count:{member_count}")
        family_plans = plans_df[plans_df["family_id"].astype(int) == family_id]
        if len(family_plans) == 0:
            reasons.append("empty_generated_plan")
        if "status" in family_plans.columns:
            status_values = [safe_text(x).lower() for x in family_plans["status"].tolist()]
            if "success" not in status_values:
                reasons.append("generated_plan_not_success")
        if reasons:
            skipped_info.append(
                {"family_id": family_id, "member_count": member_count, "reasons": reasons}
            )
        else:
            valid_family_ids.append(family_id)
    print("\n[Module 6] 完整可用 family_id 检查完成")
    print(f"[Module 6] generated_plans 中家庭数量: {len(plan_family_ids)}")
    print(f"[Module 6] 可用家庭数量: {len(valid_family_ids)}")
    print(f"[Module 6] 跳过家庭数量: {len(skipped_info)}")
    if skipped_info:
        print("[Module 6] 被跳过的 family_id：")
        for item in skipped_info[:10]:
            print(
                f"family_id={item['family_id']}, member_count={item['member_count']}, reasons={item['reasons']}"
            )
        if len(skipped_info) > 10:
            print(f"... 有 {len(skipped_info) - 10} 个家庭被跳过。")
    return valid_family_ids


def build_llm_judge_messages(
    family_id: int,
    plan_row: Dict[str, Any],
    family_personas: pd.DataFrame,
    member_rule_evaluations: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    plan_summary = {
        "family_id": family_id,
        "plan_id": plan_row.get("plan_id"),
        "destination_city": plan_row.get("destination_city"),
        "day": plan_row.get("day"),
        "itinerary_text": safe_text(plan_row.get("itinerary_text", ""))[:1200],
        "poi_tags": plan_row.get("poi_tags"),
        "meal_tags": plan_row.get("meal_tags"),
        "restaurant_price_level": plan_row.get("restaurant_price_level"),
        "daily_poi_count": plan_row.get("daily_poi_count"),
        "walking_distance_km": plan_row.get("walking_distance_km"),
        "return_time": plan_row.get("return_time"),
        "total_cost": plan_row.get("total_cost"),
        "budget_limit": plan_row.get("budget_limit"),
        "per_day_itinerary_json": safe_text(plan_row.get("per_day_itinerary_json", ""))[:1600],
        "split_activities_json": safe_text(plan_row.get("split_activities_json", ""))[:900],
        "unresolved_conflicts_json": safe_text(plan_row.get("unresolved_conflicts_json", ""))[:700],
    }
    personas = []
    for _, row in family_personas.iterrows():
        agent_id = safe_text(row.get("agent_id", ""))
        if not agent_id:
            agent_id = get_member_agent_id(row)
        personas.append(
            {
                "agent_id": agent_id,
                "family_id": safe_int(row.get("family_id"), family_id),
                "member_id": safe_int(row.get("member_id"), 0),
                "role": safe_text(row.get("role", "")),
                "age_stage": safe_text(row.get("age_stage", "")),
                "interest_preferences": safe_text(row.get("interest_preferences", "")),
                "physical_constraint": safe_text(row.get("physical_constraint", "")),
                "budget_sensitivity": safe_text(row.get("budget_sensitivity", "")),
                "dietary_preference": safe_text(row.get("dietary_preference", "")),
                "travel_pace": safe_text(row.get("travel_pace", "")),
                "must_include": safe_text(row.get("must_include", "")),
                "must_avoid": safe_text(row.get("must_avoid", "")),
            }
        )
    output_format = {
        "family_id": family_id,
        "member_llm_judgements": [
            {
                "agent_id": "family_1_member_1",
                "member_id": 1,
                "role": "father",
                "matched_evidence": ["<=2 short evidence"],
                "missing_or_weak_evidence": ["<=2 short issues"],
                "deduction_reasons": ["<=2 short reasons"],
                "llm_personalization_score": 0,
                "llm_reasonableness_score": 0,
                "llm_fairness_score": 0,
                "llm_judge_score": 0,
                "llm_judge_explanation": "<=25 words",
            }
        ],
        "family_level_llm_comment": "<=25 words",
    }
    system_prompt = "\nYou are an LLM-as-a-Judge for a multi-agent family travel planning evaluation system.\n\nYour task is to score subjective persona satisfaction.\nYou must not override rule-based hard constraints.\nBudget, walking distance, return time, duplicate POIs, and hard constraints are already evaluated by deterministic rules.\nYou only judge personalization, preference fit, fairness perception, and subjective satisfaction.\n\nOutput valid JSON only.\nDo not output markdown.\nOutput token limit: stay under 1000 output tokens.\n".strip()
    user_prompt = f"""\nEvaluate the following family travel plan.\n\nFamily ID:\n{family_id}\n\nPlan summary:\n{json.dumps(plan_summary, ensure_ascii=False, indent=2)}\n\nPersonas:\n{json.dumps(personas, ensure_ascii=False, indent=2)}\n\nRule-based member evaluations:\n{json.dumps(member_rule_evaluations, ensure_ascii=False, indent=2)}\n\nScoring requirements:\nOutput token limit: stay under 1000 output tokens. Use concise evidence lists.\nFor each member, first identify evidence, then assign scores.\n1. matched_evidence:\n   List concrete plan elements that satisfy this member's interests, must_include, pace, dietary, budget, or physical needs.\n2. missing_or_weak_evidence:\n   List concrete missing or weak matches only if they are supported by the input.\n3. fairness_evidence:\n   Explain whether this member is treated fairly compared with the others.\n4. deductions:\n   Use small, evidence-based deductions from 100:\n   - 0-2: no real issue or only wording is imperfect.\n   - 3-5: minor preference weakness, but the member would still likely accept.\n   - 6-10: clear personalization, fairness, or reasonableness weakness.\n   - more than 10: only for explicit hard-constraint failure, ignored vulnerable member, or serious fairness problem.\n5. llm_personalization_score, llm_reasonableness_score, llm_fairness_score, llm_judge_score: 0-100.\n6. llm_judge_explanation:\n   Short evidence-based explanation.\n\nScore anchors:\n- 95-100: core preferences and bottom-line needs are satisfied; only tiny weaknesses.\n- 90-94: most needs are satisfied; there are minor weak matches.\n- 85-89: basically acceptable, but there is a visible personalization or fairness weakness.\n- 80-84: likely reluctant acceptance with clear dissatisfaction points.\n- below 80: likely rejection or important needs are ignored.\n\nImportant:\n- You must return one judgement for every persona.\n- Use exactly the same agent_id as shown in Personas.\n- Do not give everyone the same score, but do not create artificial differences without evidence.\n- Every deduction must be grounded in one of these inputs: persona preferences, hard constraints, rule-based member evaluation, per-day itinerary, split activities, unresolved conflicts, walking distance, return time, or cost.\n- Do not deduct points for vague reasons such as "could be better", "not ideal", or "may prefer more" unless a concrete missing item is shown in the input.\n- Reasonable split activities are acceptable when participants, non-participants, alternatives, and reasons are clear.\n- Do not penalize split activity just because it exists.\n- If hard_constraint_pass is false, lower the score proportionally according to hard_constraint_score and the exact failed reasons.\n- If hard_constraint_failed_reasons is not empty, mention the exact reason in llm_judge_explanation.\n- If a child, older member, low-intensity member, strict-budget member, teenager with low satisfaction, or lowest-score member is ignored, lower the fairness score.\n- If one member has much lower rule score than others, explicitly reflect this in fairness scoring.\n- LLM score is auxiliary. Do not invent new objective constraints.\n\nReturn JSON exactly in this format:\n{json.dumps(output_format, ensure_ascii=False, indent=2)}\n""".strip()
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def run_llm_judge_for_family(
    family_id: int,
    plan_row: Dict[str, Any],
    family_personas: pd.DataFrame,
    member_rule_evaluations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not USE_LLM_JUDGE:
        return {
            "family_id": family_id,
            "member_llm_judgements": [],
            "family_level_llm_comment": "LLM Judge disabled.",
        }
    try:
        messages = build_llm_judge_messages(
            family_id=family_id,
            plan_row=plan_row,
            family_personas=family_personas,
            member_rule_evaluations=member_rule_evaluations,
        )
        messages = budget_messages(messages, max_input_tokens=4200)
        api_result = call_llm_judge_json(
            messages=messages,
            model=LLM_JUDGE_MODEL_NAME,
            max_tokens=LLM_JUDGE_MAX_TOKENS,
            temperature=LLM_JUDGE_TEMPERATURE,
        )
        data = api_result["data"]
        if "member_llm_judgements" not in data:
            data["member_llm_judgements"] = []
        return data
    except Exception as e:
        print(f"[Module 6] family_id={family_id} LLM Judge 失败: {e}")
        return {
            "family_id": family_id,
            "member_llm_judgements": [],
            "family_level_llm_comment": f"LLM Judge failed: {e}",
        }


def build_llm_judge_lookup(llm_judge_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lookup = {}
    judgements = llm_judge_result.get("member_llm_judgements", [])
    if not isinstance(judgements, list):
        return lookup
    for item in judgements:
        agent_id = safe_text(item.get("agent_id", ""))
        if agent_id:
            lookup[agent_id] = item
    return lookup


def clamp_llm_score_near_rule_score(
    rule_score: float,
    llm_score: float,
    hard_constraint_pass: bool = True,
    max_delta: float = MAX_LLM_SCORE_DELTA_FROM_RULE,
) -> float:
    rule_score = safe_float(rule_score, 0)
    llm_score = safe_float(llm_score, rule_score)
    if pd.isna(llm_score):
        llm_score = rule_score
    effective_delta = max_delta
    if not hard_constraint_pass:
        effective_delta = max(max_delta, MAX_LLM_SCORE_DELTA_WITH_HARD_FAILURE)
    lower = rule_score - effective_delta
    upper = rule_score + effective_delta
    bounded_score = min(max(llm_score, lower), upper)
    return round(clamp_score(bounded_score), 4)


def compute_weighted_satisfaction_score(
    rule_score: float, llm_score: float, hard_constraint_pass: bool = True
) -> float:
    rule_score = safe_float(rule_score, 0)
    if llm_score is None:
        llm_score = rule_score
    bounded_llm_score = clamp_llm_score_near_rule_score(
        rule_score=rule_score, llm_score=llm_score, hard_constraint_pass=hard_constraint_pass
    )
    final_score = RULE_SCORE_WEIGHT * rule_score + LLM_SCORE_WEIGHT * bounded_llm_score
    return round(clamp_score(final_score), 4)


def evaluate_hard_constraints(persona_row, plan_row) -> Dict[str, Any]:
    plan_text = combine_plan_text(plan_row)
    must_include = normalize_list(parse_json_list(persona_row.get("must_include", "[]")))
    must_avoid = normalize_list(parse_json_list(persona_row.get("must_avoid", "[]")))
    max_walking = safe_float(persona_row.get("max_walking_distance_km", np.nan))
    latest_return_time = persona_row.get("latest_return_time", "")
    day = safe_float(plan_row.get("day", 1), 1)
    avg_walking = average_per_day(plan_row.get("walking_distance_km", np.nan), day)
    return_time = plan_row.get("return_time", "")
    failed_reasons = []
    passed_checks = 0
    total_checks = 0
    severity_penalty = 0.0
    must_include_missing_count = 0
    must_avoid_violation_count = 0
    concession_count = 0
    walking_excess_km = 0.0
    return_delay_minutes = 0
    for item in must_include:
        total_checks += 1
        if item in plan_text:
            passed_checks += 1
        else:
            must_include_missing_count += 1
            severity_penalty += 28.0
            failed_reasons.append(f"must_include_not_satisfied:{item}")
    for item in must_avoid:
        total_checks += 1
        if item not in plan_text:
            passed_checks += 1
        elif is_avoid_exempted_by_split_activity(
            persona_row=persona_row, plan_row=plan_row, avoid_term=item
        ):
            passed_checks += 1
        elif is_avoid_conceded_by_negotiation(
            persona_row=persona_row, plan_row=plan_row, avoid_term=item
        ):
            # Negotiated concession (no_split): not a hard veto, so it does NOT fail the hard gate,
            # but the conceder still eats a moderate dissatisfaction — strictly worse than a split
            # (cost-free) and strictly better than an unresolved violation (35 + unresolved penalty).
            passed_checks += 1
            concession_count += 1
            severity_penalty += CONCESSION_HARD_PENALTY
        else:
            must_avoid_violation_count += 1
            severity_penalty += 35.0
            failed_reasons.append(f"must_avoid_violated:{item}")
    if not pd.isna(max_walking) and (not pd.isna(avg_walking)):
        total_checks += 1
        if avg_walking <= max_walking:
            passed_checks += 1
        else:
            walking_excess_km = max(0.0, avg_walking - max_walking)
            severity_penalty += min(35.0, 14.0 + 5.0 * walking_excess_km)
            failed_reasons.append(
                f"walking_distance_exceeded:avg_per_day={round(avg_walking, 2)},limit={max_walking}"
            )
    latest_minutes = time_to_minutes(latest_return_time)
    return_minutes = time_to_minutes(return_time)
    if latest_minutes is not None and return_minutes is not None:
        total_checks += 1
        if return_minutes <= latest_minutes:
            passed_checks += 1
        else:
            return_delay_minutes = max(0, return_minutes - latest_minutes)
            severity_penalty += min(35.0, 12.0 + 0.12 * return_delay_minutes)
            failed_reasons.append(
                f"return_time_too_late:return={return_time},limit={latest_return_time}"
            )
    if total_checks == 0:
        pass_ratio_score = 100.0
        hard_constraint_pass = True
    else:
        pass_ratio_score = 100.0 * passed_checks / total_checks
        hard_constraint_pass = len(failed_reasons) == 0
    severity_score = clamp_score(100.0 - severity_penalty)
    hard_constraint_score = severity_score
    return {
        "hard_constraint_pass": hard_constraint_pass,
        "hard_constraint_score": round(hard_constraint_score, 4),
        "hard_constraint_pass_ratio_score": round(pass_ratio_score, 4),
        "hard_constraint_severity_score": round(severity_score, 4),
        "hard_constraint_violation_count": len(failed_reasons),
        "must_include_missing_count": must_include_missing_count,
        "must_avoid_violation_count": must_avoid_violation_count,
        "concession_count": concession_count,
        "walking_excess_km": round(walking_excess_km, 4),
        "return_delay_minutes": int(return_delay_minutes),
        "hard_constraint_failed_reasons": json.dumps(failed_reasons, ensure_ascii=False),
    }


def evaluate_preference_coverage(persona_row, plan_row) -> Dict[str, Any]:
    interest_preferences = normalize_list(
        parse_json_list(persona_row.get("interest_preferences", "[]"))
    )
    must_include = normalize_list(parse_json_list(persona_row.get("must_include", "[]")))
    poi_tags = normalize_list(parse_json_list(plan_row.get("poi_tags", "[]")))
    meal_tags = normalize_list(parse_json_list(plan_row.get("meal_tags", "[]")))
    plan_text = combine_plan_text(plan_row)
    all_plan_features = set(poi_tags + meal_tags)
    if interest_preferences:
        matched_preferences = []
        for pref in interest_preferences:
            if pref in all_plan_features or pref in plan_text:
                matched_preferences.append(pref)
        preference_coverage_rate = len(matched_preferences) / len(interest_preferences)
    else:
        matched_preferences = []
        preference_coverage_rate = 1.0
    if must_include:
        matched_must_have = []
        for item in must_include:
            if item in all_plan_features or item in plan_text:
                matched_must_have.append(item)
        must_have_satisfaction_rate = len(matched_must_have) / len(must_include)
    else:
        matched_must_have = []
        must_have_satisfaction_rate = 1.0
    return {
        "preference_coverage_rate": round(preference_coverage_rate, 4),
        "matched_preferences": json.dumps(matched_preferences, ensure_ascii=False),
        "must_have_satisfaction_rate": round(must_have_satisfaction_rate, 4),
        "matched_must_have": json.dumps(matched_must_have, ensure_ascii=False),
    }


def evaluate_food_match(persona_row, plan_row) -> Dict[str, Any]:
    dietary_preference = normalize_list(
        parse_json_list(persona_row.get("dietary_preference", "[]"))
    )
    meal_tags = normalize_list(parse_json_list(plan_row.get("meal_tags", "[]")))
    restaurant_price_level = safe_text(plan_row.get("restaurant_price_level", "")).lower()
    score = 100.0
    issues = []
    if dietary_preference:
        matched = [item for item in dietary_preference if item in meal_tags]
        coverage = len(matched) / len(dietary_preference)
        score = 50 + 50 * coverage
        if "price_sensitive" in dietary_preference:
            if restaurant_price_level in ["high", "expensive", "luxury"]:
                score -= 30
                issues.append("price_sensitive_but_high_price_meals")
        if "child_friendly" in dietary_preference:
            if "child_friendly" not in meal_tags and "kids_menu" not in meal_tags:
                score -= 20
                issues.append("child_friendly_food_not_satisfied")
        if "light_food" in dietary_preference:
            heavy_tags = ["spicy", "fried", "barbecue", "heavy"]
            if any((tag in meal_tags for tag in heavy_tags)):
                score -= 20
                issues.append("light_food_preference_conflict")
    return {
        "food_preference_score": round(clamp_score(score), 4),
        "food_issues": json.dumps(issues, ensure_ascii=False),
    }


def evaluate_intensity_match(persona_row, plan_row) -> Dict[str, Any]:
    physical_constraint = safe_text(persona_row.get("physical_constraint", "")).lower()
    travel_pace = safe_text(persona_row.get("travel_pace", "")).lower()
    day = safe_float(plan_row.get("day", 1), 1)
    avg_walking = average_per_day(plan_row.get("walking_distance_km", np.nan), day)
    daily_poi_count = safe_float(plan_row.get("daily_poi_count", np.nan))
    high_intensity_tags = normalize_list(parse_json_list(plan_row.get("high_intensity_tags", "[]")))
    score = 100.0
    issues = []
    if not pd.isna(avg_walking):
        if physical_constraint == "low_intensity":
            if avg_walking > 5:
                score -= 50
                issues.append("too_much_walking_for_low_intensity")
            elif avg_walking > 3:
                score -= 25
                issues.append("walking_slightly_high_for_low_intensity")
        elif physical_constraint == "medium_intensity":
            if avg_walking > 7:
                score -= 35
                issues.append("too_much_walking_for_medium_intensity")
        elif physical_constraint == "high_intensity":
            if avg_walking < 2:
                score -= 10
                issues.append("may_be_too_light_for_high_intensity")
    if not pd.isna(daily_poi_count):
        if travel_pace == "relaxed" and daily_poi_count > 4:
            score -= 25
            issues.append("too_many_pois_for_relaxed_pace")
        if travel_pace == "compact" and daily_poi_count < 3:
            score -= 10
            issues.append("too_few_pois_for_compact_pace")
    if physical_constraint == "low_intensity" and len(high_intensity_tags) > 0:
        score -= 30
        issues.append("high_intensity_activity_conflict")
    return {
        "intensity_match_score": round(clamp_score(score), 4),
        "intensity_issues": json.dumps(issues, ensure_ascii=False),
    }


def evaluate_budget_match(persona_row, plan_row) -> Dict[str, Any]:
    budget_sensitivity = safe_text(persona_row.get("budget_sensitivity", "")).lower()
    total_cost = safe_float(plan_row.get("total_cost", np.nan))
    budget_limit = safe_float(plan_row.get("budget_limit", np.nan))
    restaurant_price_level = safe_text(plan_row.get("restaurant_price_level", "")).lower()
    hotel_plan = parse_json_text(plan_row.get("hotel_plan_json", "{}"), default={})
    hotel_price_level = safe_text(hotel_plan.get("price_level", "")).lower()
    score = 100.0
    issues = []
    if not pd.isna(total_cost) and (not pd.isna(budget_limit)) and (budget_limit > 0):
        if total_cost > budget_limit:
            over_ratio = (total_cost - budget_limit) / budget_limit
            penalty = min(50, over_ratio * 100)
            score -= penalty
            issues.append("total_cost_exceeds_budget_limit")
    if budget_sensitivity == "strict_budget":
        if restaurant_price_level in ["high", "expensive", "luxury"]:
            score -= 20
            issues.append("strict_budget_but_high_price_restaurants")
        if hotel_price_level in ["high", "expensive", "luxury"]:
            score -= 20
            issues.append("strict_budget_but_high_price_hotel")
    if budget_sensitivity == "experience_first":
        if restaurant_price_level == "low" and hotel_price_level == "low":
            score -= 5
            issues.append("experience_first_but_plan_too_budget_oriented")
    return {
        "budget_match_score": round(clamp_score(score), 4),
        "budget_issues": json.dumps(issues, ensure_ascii=False),
    }


def compute_member_satisfaction(
    persona_row, plan_row, agent_weight_lookup: Dict[str, Dict[str, float]]
) -> Dict[str, Any]:
    hard = evaluate_hard_constraints(persona_row, plan_row)
    pref = evaluate_preference_coverage(persona_row, plan_row)
    food = evaluate_food_match(persona_row, plan_row)
    intensity = evaluate_intensity_match(persona_row, plan_row)
    budget = evaluate_budget_match(persona_row, plan_row)
    agent_id = get_member_agent_id(persona_row)
    weights = normalize_weights(agent_weight_lookup.get(agent_id, {}))
    interest_score = pref["preference_coverage_rate"] * 100
    must_score = pref["must_have_satisfaction_rate"] * 100
    food_score = food["food_preference_score"]
    physical_score = intensity["intensity_match_score"]
    budget_score = budget["budget_match_score"]
    hard_score = safe_float(
        hard.get("hard_constraint_severity_score"), hard.get("hard_constraint_score", 100.0)
    )
    preference_component_score = (
        interest_score * weights["interest_weight"]
        + physical_score * weights["physical_weight"]
        + budget_score * weights["budget_weight"]
        + food_score * weights["food_weight"]
        + physical_score * weights["pace_weight"]
        + must_score * weights["hard_constraint_weight"]
    )
    rule_score = (
        hard_score * HARD_CONSTRAINT_COMPONENT_WEIGHT
        + preference_component_score * OTHER_PREFERENCE_COMPONENT_WEIGHT
    )
    mismatch_count = 0
    for json_value in [
        hard.get("hard_constraint_failed_reasons"),
        food.get("food_issues"),
        intensity.get("intensity_issues"),
        budget.get("budget_issues"),
    ]:
        parsed = parse_json_list(json_value)
        if parsed:
            mismatch_count += len(parsed)
    if pref["preference_coverage_rate"] < 1.0:
        mismatch_count += 1
    if pref["must_have_satisfaction_rate"] < 1.0:
        mismatch_count += 1
    # Deficiency-sensitive penalty (see constants above): continuous in how much is actually
    # missed, so ablations that leave must-haves/preferences unmet separate from the clean full
    # model. must_gap / pref_gap are 0 for the full model (rates=1.0) → no penalty there.
    must_gap = max(0.0, 1.0 - safe_float(pref["must_have_satisfaction_rate"], 1.0))
    pref_gap = max(0.0, 1.0 - safe_float(pref["preference_coverage_rate"], 1.0))
    # must_avoid_violation_count already honors the split exemption → 0 for the full model.
    avoid_violations = safe_int(hard.get("must_avoid_violation_count"), 0)
    member_penalty_score = min(
        MEMBER_DEFICIENCY_PENALTY_CAP,
        must_gap * MUST_HAVE_MISS_PENALTY
        + pref_gap * PREFERENCE_GAP_PENALTY
        + avoid_violations * MUST_AVOID_VIOLATION_PENALTY
        + mismatch_count * PER_MISMATCH_PENALTY,
    )
    rule_score = clamp_score(rule_score - member_penalty_score)
    accepted_vote = rule_score >= ACCEPTANCE_SCORE_THRESHOLD
    explanation_parts = []
    explanation_parts.append("硬约束满足" if hard["hard_constraint_pass"] else "硬约束未完全满足")
    explanation_parts.append(f"硬约束严重度分={hard_score:.1f}")
    explanation_parts.append(f"兴趣覆盖率={pref['preference_coverage_rate']:.2f}")
    explanation_parts.append(f"must-have满足率={pref['must_have_satisfaction_rate']:.2f}")
    explanation_parts.append(f"体力/节奏匹配分={physical_score:.1f}")
    explanation_parts.append(f"餐饮匹配分={food_score:.1f}")
    explanation_parts.append(f"预算匹配分={budget_score:.1f}")
    explanation_parts.append(f"个性化不匹配数={mismatch_count}")
    if member_penalty_score > 0:
        explanation_parts.append(f"个性化惩罚={member_penalty_score:.1f}")
    return {
        "agent_id": agent_id,
        **hard,
        **pref,
        **food,
        **intensity,
        **budget,
        "personalization_mismatch_count": int(mismatch_count),
        "member_penalty_score": round(member_penalty_score, 4),
        "individual_rule_score": round(rule_score, 4),
        "llm_judge_score": None,
        "llm_personalization_score": None,
        "llm_reasonableness_score": None,
        "llm_fairness_score": None,
        "llm_judge_explanation": "",
        "individual_satisfaction_score": round(rule_score, 4),
        "accepted_vote": accepted_vote,
        "member_explanation": "；".join(explanation_parts),
        "used_preference_weights": json.dumps(weights, ensure_ascii=False),
    }


def compute_grounding_specificity_cap(grounding_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Eval-honesty score ceilings implied by real-world specificity.

    When restaurants/activities/hotels are not concretely grounded, a plan must not be
    able to report near-perfect satisfaction or feasibility. We return *ceilings* (applied
    later with min()), the triggering reasons, and a low-specificity flag, so the report
    can show both pre-cap and capped numbers and the honesty adjustment is auditable.

    ``grounding_metrics`` is the dict produced by compute_real_world_grounding_metrics
    (which embeds the post-generation grounding fields), so this reads the same flags the
    grounding evaluator already computes rather than re-inferring specificity from text.
    """
    restaurant_specificity_rate = safe_float(
        grounding_metrics.get("restaurant_specificity_rate"), 1.0
    )
    restaurant_failure_count = safe_int(grounding_metrics.get("restaurant_failure_count"), 0)
    generic_activity_count = safe_int(grounding_metrics.get("generic_activity_label_count"), 0)
    placeholder_artifact_count = safe_int(grounding_metrics.get("placeholder_artifact_count"), 0)
    hotel_specific = bool_from_any(grounding_metrics.get("hotel_realism_pass")) and bool_from_any(
        grounding_metrics.get("hotel_nearest_requirement_pass")
    )
    hotel_estimated = bool_from_any(grounding_metrics.get("hotel_price_estimated"))
    # A city's distinct restaurants may run out, so up to ALLOWED_UNGROUNDED_MEALS
    # area-fallback meals are acceptable and must not cap the score.
    restaurant_over_allowance = restaurant_failure_count > ALLOWED_UNGROUNDED_MEALS
    reasons = []
    if (
        restaurant_over_allowance
        and restaurant_specificity_rate < GROUNDING_CAP_RESTAURANT_SPECIFICITY_MIN
    ):
        reasons.append(
            f"restaurant_specificity_rate<{GROUNDING_CAP_RESTAURANT_SPECIFICITY_MIN}:{round(restaurant_specificity_rate, 4)}"
        )
    if GROUNDING_CAP_REQUIRE_ALL_MEALS_SPECIFIC and restaurant_over_allowance:
        reasons.append(
            f"restaurant_failure_count={restaurant_failure_count}>{ALLOWED_UNGROUNDED_MEALS}"
        )
    if generic_activity_count > 0:
        reasons.append(f"generic_activity_label_count={generic_activity_count}")
    if placeholder_artifact_count > 0:
        reasons.append(f"placeholder_artifact_count={placeholder_artifact_count}")
    # A budget-driven estimate hotel is honest behavior (no affordable real hotel in
    # the DB), so it is not treated as a vague/ungrounded hotel here.
    if not hotel_specific and not hotel_estimated:
        reasons.append("hotel_not_specifically_grounded")
    grounding_specificity_low = bool(reasons) and ENABLE_GROUNDING_SPECIFICITY_CAP
    if not grounding_specificity_low:
        satisfaction_cap = feasibility_cap = 100.0
    elif GROUNDING_CAP_MODE == "graded":
        # Penalty scales with how non-specific the plan is, so a mostly-grounded plan keeps a
        # high (but <100) ceiling while a vague plan is pushed toward the floor.
        penalty = (
            max(0.0, 1.0 - restaurant_specificity_rate) * 15.0
            + min(10.0, generic_activity_count * 2.5)
            + min(6.0, placeholder_artifact_count * 2.0)
            + (0.0 if hotel_specific else 6.0)
        )
        satisfaction_cap = max(GROUNDING_CAP_FLOOR, 100.0 - penalty)
        feasibility_cap = max(GROUNDING_CAP_FLOOR, 100.0 - penalty * 0.9)
    else:
        satisfaction_cap = GROUNDING_FAIL_SATISFACTION_CAP
        feasibility_cap = GROUNDING_FAIL_FEASIBILITY_CAP
    return {
        "grounding_specificity_low": bool(grounding_specificity_low),
        "grounding_specificity_cap_reasons_json": json.dumps(reasons, ensure_ascii=False),
        "satisfaction_grounding_cap": round(satisfaction_cap, 4),
        "feasibility_grounding_cap": round(feasibility_cap, 4),
    }


def evaluate_plan_feasibility(plan_row) -> Dict[str, Any]:
    total_cost = safe_float(plan_row.get("total_cost", np.nan))
    budget_limit = safe_float(plan_row.get("budget_limit", np.nan))
    day = safe_float(plan_row.get("day", 1), 1)
    walking_distance = safe_float(plan_row.get("walking_distance_km", np.nan))
    route_distance = safe_float(plan_row.get("route_distance_km", np.nan))
    daily_poi_count = safe_float(plan_row.get("daily_poi_count", np.nan))
    return_time = safe_text(plan_row.get("return_time", ""))
    hotel_plan = parse_json_text(plan_row.get("hotel_plan_json", "{}"), default={})
    transportation_plan = parse_json_text(
        plan_row.get("transportation_plan_json", "{}"), default={}
    )
    budget_pass = True
    budget_overrun_ratio = 0.0
    if not pd.isna(total_cost) and (not pd.isna(budget_limit)) and (budget_limit > 0):
        budget_pass = total_cost <= budget_limit * BUDGET_PASS_TOLERANCE
        budget_overrun_ratio = max(0.0, (total_cost - budget_limit) / budget_limit)
    required_fields = [
        "itinerary_text",
        "per_day_itinerary_json",
        "hotel_plan_json",
        "transportation_plan_json",
        "meal_tags",
        "poi_tags",
    ]
    complete_count = sum((1 for field in required_fields if safe_text(plan_row.get(field, ""))))
    complete_information_rate = complete_count / len(required_fields)
    accommodation_completeness = 1.0 if hotel_plan else 0.0
    transportation_feasibility = 1.0 if transportation_plan else 0.0
    avg_walking_per_day = average_per_day(walking_distance, day)
    avg_route_per_day = average_per_day(route_distance, day)
    walking_reasonable = (
        pd.isna(avg_walking_per_day) or avg_walking_per_day <= MAX_REASONABLE_WALKING_PER_DAY
    )
    route_reasonable = (
        pd.isna(avg_route_per_day) or avg_route_per_day <= MAX_REASONABLE_ROUTE_DISTANCE_PER_DAY
    )
    poi_reasonable = pd.isna(daily_poi_count) or daily_poi_count <= MAX_REASONABLE_POI_PER_DAY
    return_minutes = time_to_minutes(return_time)
    latest_reasonable_minutes = time_to_minutes(LATEST_REASONABLE_RETURN_TIME)
    return_time_reasonable = True
    if return_minutes is not None and latest_reasonable_minutes is not None:
        return_time_reasonable = return_minutes <= latest_reasonable_minutes
    real_world_grounding = compute_real_world_grounding_metrics(plan_row)
    real_world_grounding_pass = bool(real_world_grounding.get("real_world_grounding_pass"))
    objective_metrics = compute_objective_style_metrics(
        plan_row=plan_row, real_world_grounding=real_world_grounding
    )
    feasibility_checks = [
        budget_pass,
        complete_information_rate >= 0.8,
        accommodation_completeness >= 1.0,
        transportation_feasibility >= 1.0,
        real_world_grounding_pass,
        bool(objective_metrics.get("objective_feasibility_macro")),
        walking_reasonable,
        route_reasonable,
        poi_reasonable,
        return_time_reasonable,
    ]
    feasibility_pass_rate = sum(feasibility_checks) / len(feasibility_checks)
    no_hallucinated_poi_rate = 1.0
    score = 100.0
    if not budget_pass:
        score -= min(35.0, 12.0 + 120.0 * budget_overrun_ratio)
    score -= max(0.0, 0.8 - complete_information_rate) * 40.0
    if accommodation_completeness < 1.0:
        score -= 10.0
    if transportation_feasibility < 1.0:
        score -= 10.0
    if not walking_reasonable and (not pd.isna(avg_walking_per_day)):
        score -= min(
            20.0, 5.0 + 2.5 * max(0.0, avg_walking_per_day - MAX_REASONABLE_WALKING_PER_DAY)
        )
    if not route_reasonable and (not pd.isna(avg_route_per_day)):
        score -= min(
            12.0, 4.0 + 0.1 * max(0.0, avg_route_per_day - MAX_REASONABLE_ROUTE_DISTANCE_PER_DAY)
        )
    if not poi_reasonable and (not pd.isna(daily_poi_count)):
        score -= min(12.0, 4.0 + 3.0 * max(0.0, daily_poi_count - MAX_REASONABLE_POI_PER_DAY))
    if not return_time_reasonable:
        score -= 10.0
    route_realism = compute_route_realism_penalty(plan_row)
    score -= safe_float(route_realism.get("route_realism_penalty"), 0.0)
    score -= safe_float(real_world_grounding.get("real_world_grounding_penalty"), 0.0)
    score -= safe_float(objective_metrics.get("objective_feasibility_penalty"), 0.0)
    grounding_cap = compute_grounding_specificity_cap(real_world_grounding)
    pre_cap_feasibility_score = clamp_score(score)
    feasibility_score = min(
        pre_cap_feasibility_score, safe_float(grounding_cap.get("feasibility_grounding_cap"), 100.0)
    )
    return {
        "budget_pass": budget_pass,
        "complete_information_rate": round(complete_information_rate, 4),
        "no_hallucinated_poi_rate": no_hallucinated_poi_rate,
        "accommodation_completeness": accommodation_completeness,
        "transportation_feasibility": transportation_feasibility,
        **real_world_grounding,
        **objective_metrics,
        **grounding_cap,
        "avg_walking_per_day": round(avg_walking_per_day, 4)
        if not pd.isna(avg_walking_per_day)
        else np.nan,
        "avg_route_distance_per_day": round(avg_route_per_day, 4)
        if not pd.isna(avg_route_per_day)
        else np.nan,
        "walking_reasonable": walking_reasonable,
        "route_reasonable": route_reasonable,
        "poi_count_reasonable": poi_reasonable,
        "return_time_reasonable": return_time_reasonable,
        "feasibility_pass_rate": round(feasibility_pass_rate, 4),
        "pre_cap_feasibility_score": round(pre_cap_feasibility_score, 4),
        "feasibility_score": round(feasibility_score, 4),
        **route_realism,
    }


def normalize_price_level(value) -> str:
    text = safe_text(value).lower().strip()
    if text in ["", "nan", "none", "null"]:
        return "medium"
    if text in ["low", "cheap", "budget", "affordable", "低", "便宜", "经济", "实惠"]:
        return "low"
    if text in ["medium", "moderate", "mid", "mid-range", "normal", "中等", "适中", "普通", "正常"]:
        return "medium"
    if text in ["high", "expensive", "luxury", "premium", "高", "昂贵", "豪华", "高端"]:
        return "high"
    if "low" in text and "medium" in text and ("high" in text):
        return "medium"
    if "budget" in text or "affordable" in text or "cheap" in text:
        return "low"
    if "moderate" in text or "mid" in text or "medium" in text:
        return "medium"
    if "expensive" in text or "luxury" in text or "premium" in text:
        return "high"
    return "medium"


def evaluate_plan_rationality(plan_row) -> Dict[str, Any]:
    poi_tags = normalize_list(parse_json_list(plan_row.get("poi_tags", "[]")))
    meal_tags = normalize_list(parse_json_list(plan_row.get("meal_tags", "[]")))
    duplicate_poi_count = safe_int(plan_row.get("duplicate_poi_count", 0), 0)
    high_intensity_tags = normalize_list(parse_json_list(plan_row.get("high_intensity_tags", "[]")))
    daily_poi_count = safe_float(plan_row.get("daily_poi_count", np.nan))
    restaurant_price_level = normalize_price_level(plan_row.get("restaurant_price_level", ""))
    total_cost = safe_float(plan_row.get("total_cost", np.nan))
    budget_limit = safe_float(plan_row.get("budget_limit", np.nan))
    budget_pass = True
    if not pd.isna(total_cost) and (not pd.isna(budget_limit)) and (budget_limit > 0):
        budget_pass = total_cost <= budget_limit
    objective_metrics = compute_objective_style_metrics(plan_row)
    diverse_attractions = len(set(poi_tags)) >= 2
    diverse_restaurants = len(set(meal_tags)) >= 2
    no_duplicate_poi = duplicate_poi_count == 0
    appropriate_visit_duration = True
    if not pd.isna(daily_poi_count):
        appropriate_visit_duration = 1 <= daily_poi_count <= MAX_REASONABLE_POI_PER_DAY
    if restaurant_price_level in ["low", "medium"]:
        reasonable_meal_prices = True
    elif restaurant_price_level == "high":
        reasonable_meal_prices = budget_pass
    else:
        reasonable_meal_prices = budget_pass
    travel_intensity_balance = len(high_intensity_tags) <= 1
    checks = [
        diverse_attractions,
        diverse_restaurants,
        no_duplicate_poi,
        appropriate_visit_duration,
        reasonable_meal_prices,
        travel_intensity_balance,
        bool(objective_metrics.get("objective_rationality_macro")),
    ]
    rationality_pass_rate = sum(checks) / len(checks)
    score = 100.0
    if not diverse_attractions:
        score -= 10.0
    if not diverse_restaurants:
        score -= 8.0
    if not no_duplicate_poi:
        score -= min(18.0, duplicate_poi_count * 6.0)
    if not appropriate_visit_duration:
        score -= 10.0
    if not reasonable_meal_prices:
        score -= 8.0
    if not travel_intensity_balance:
        score -= min(15.0, len(high_intensity_tags) * 5.0)
    score -= safe_float(objective_metrics.get("objective_rationality_penalty"), 0.0)
    return {
        "diverse_attractions": diverse_attractions,
        "diverse_restaurants": diverse_restaurants,
        "no_duplicate_poi": no_duplicate_poi,
        "appropriate_visit_duration": appropriate_visit_duration,
        "reasonable_meal_prices": reasonable_meal_prices,
        "travel_intensity_balance": travel_intensity_balance,
        "rationality_pass_rate": round(rationality_pass_rate, 4),
        "rationality_score": round(clamp_score(score), 4),
    }


def compute_plan_level_metrics(member_eval_df: pd.DataFrame, plan_row) -> Dict[str, Any]:
    scores = pd.to_numeric(
        member_eval_df["individual_satisfaction_score"], errors="coerce"
    ).dropna()
    if scores.empty:
        avg_score = np.nan
        min_score = np.nan
        max_score = np.nan
        variance = np.nan
        std = np.nan
    else:
        avg_score = scores.mean()
        min_score = scores.min()
        max_score = scores.max()
        variance = scores.var(ddof=0)
        std = scores.std(ddof=0)
    hard_constraint_pass_rate = member_eval_df["hard_constraint_pass"].mean()
    accepted_vote_ratio = member_eval_df["accepted_vote"].mean()
    max_min_gap = max_score - min_score if not pd.isna(max_score) else np.nan
    hard_severity_values = pd.to_numeric(
        member_eval_df.get("hard_constraint_severity_score", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    hard_constraint_severity_score = (
        hard_severity_values.mean()
        if not hard_severity_values.empty
        else hard_constraint_pass_rate * 100
    )
    if not pd.isna(min_score) and (not pd.isna(max_score)) and (max_score > 0):
        max_min_fairness = min_score / max_score
    else:
        max_min_fairness = np.nan
    feasibility = evaluate_plan_feasibility(plan_row)
    rationality = evaluate_plan_rationality(plan_row)
    unresolved_metrics = compute_unresolved_conflict_penalty(plan_row)
    split_metrics = compute_split_activity_cost(plan_row)
    constraint_typing_metrics = compute_constraint_satisfiability_typing(plan_row)
    mismatch_metrics = compute_member_level_mismatch_metrics(member_eval_df)
    # Concession count (Module 4's non-split resolution): how many must_avoid holders yielded so the
    # family could reach consensus without split. >0 marks the no_split regime; 0 for full (split) and
    # no_negotiation (nothing conceded). Surfaced at plan level so the ablation report can separate them.
    concession_count = int(
        pd.to_numeric(
            member_eval_df.get("concession_count", pd.Series(dtype=float)), errors="coerce"
        )
        .fillna(0)
        .sum()
    )
    unresolved_penalty = safe_float(unresolved_metrics.get("unresolved_conflict_penalty"), 0)
    split_cost = safe_float(split_metrics.get("split_activity_cost"), 0)
    route_penalty = (
        safe_float(feasibility.get("route_realism_penalty"), 0)
        + safe_float(feasibility.get("real_world_grounding_penalty"), 0)
        + safe_float(feasibility.get("objective_evaluation_penalty"), 0)
    )
    mismatch_count = safe_int(mismatch_metrics.get("personalization_mismatch_count"), 0)
    raw_group_consensus_score = (
        avg_score - 0.35 * std if not pd.isna(avg_score) and (not pd.isna(std)) else np.nan
    )
    adjusted_group_consensus_score = compute_adjusted_group_score(
        avg_score=avg_score,
        min_score=min_score,
        hard_severity=hard_constraint_severity_score,
        feasibility_score=safe_float(
            feasibility.get("feasibility_score"), feasibility.get("feasibility_pass_rate", 0) * 100
        ),
        rationality_score=safe_float(
            rationality.get("rationality_score"), rationality.get("rationality_pass_rate", 0) * 100
        ),
        acceptance_vote_ratio=accepted_vote_ratio,
        max_min_fairness=max_min_fairness,
        unresolved_penalty=unresolved_penalty,
        split_cost=split_cost,
        route_penalty=route_penalty,
        mismatch_count=mismatch_count,
    )
    obvious_unfairness = False
    unfairness_reasons = []
    if not pd.isna(max_min_gap) and max_min_gap >= UNFAIRNESS_GAP_THRESHOLD:
        obvious_unfairness = True
        unfairness_reasons.append("max_min_satisfaction_gap_too_large")
    if not pd.isna(min_score) and min_score < LOW_MIN_SATISFACTION_THRESHOLD:
        obvious_unfairness = True
        unfairness_reasons.append("minimum_member_satisfaction_too_low")
    if not pd.isna(hard_constraint_pass_rate) and hard_constraint_pass_rate < 1.0:
        obvious_unfairness = True
        unfairness_reasons.append("some_members_hard_constraints_failed")
    if mismatch_metrics["strict_low_satisfaction_member_count"] > 0:
        obvious_unfairness = True
        unfairness_reasons.append("strict_low_satisfaction_member_exists")
    if unresolved_penalty > MAX_ACCEPTABLE_UNRESOLVED_CONFLICT_PENALTY:
        unfairness_reasons.append("unresolved_conflict_penalty_high")
    if split_cost > MAX_ACCEPTABLE_SPLIT_ACTIVITY_COST:
        unfairness_reasons.append("split_activity_cost_high")
    consensus_metrics = adjust_consensus_status_by_evaluation(
        original_status=plan_row.get("consensus_status", ""),
        hard_constraint_pass_rate=hard_constraint_pass_rate,
        acceptance_vote_ratio=accepted_vote_ratio,
        unresolved_penalty=unresolved_penalty,
        unresolved_direct_count=safe_int(
            unresolved_metrics.get("unresolved_direct_conflict_count"), 0
        ),
        split_cost=split_cost,
        mismatch_count=mismatch_count,
        obvious_unfairness=obvious_unfairness,
    )
    fairness_objective_metrics = compute_fairness_aware_group_objective(
        avg_score=avg_score,
        min_score=min_score,
        hard_constraint_severity_score=hard_constraint_severity_score,
        accepted_vote_ratio=accepted_vote_ratio,
        max_min_fairness=max_min_fairness,
        unresolved_penalty=unresolved_penalty,
        split_cost=split_cost,
        route_penalty=route_penalty,
        mismatch_count=mismatch_count,
        obvious_unfairness=obvious_unfairness,
    )
    # Eval-honesty: bound reported group/min satisfaction by real-world specificity.
    # Composite scores (group_consensus_score, fairness objective) already absorb grounding
    # via route_penalty, so we cap only the headline satisfaction numbers here and keep the
    # uncapped values under pre_cap_* for transparency.
    satisfaction_grounding_cap = safe_float(feasibility.get("satisfaction_grounding_cap"), 100.0)
    grounding_specificity_low = bool_from_any(feasibility.get("grounding_specificity_low"))
    capped_avg_score = (
        min(avg_score, satisfaction_grounding_cap) if not pd.isna(avg_score) else avg_score
    )
    capped_min_score = (
        min(min_score, satisfaction_grounding_cap) if not pd.isna(min_score) else min_score
    )
    satisfaction_cap_applied = grounding_specificity_low and (
        ((not pd.isna(avg_score)) and avg_score > satisfaction_grounding_cap)
        or ((not pd.isna(min_score)) and min_score > satisfaction_grounding_cap)
    )
    return {
        "family_id": plan_row.get("family_id"),
        "plan_id": plan_row.get("plan_id", "consensus_plan"),
        "method": plan_row.get("method", ""),
        "departure_city": plan_row.get("departure_city", ""),
        "destination_city": plan_row.get("destination_city", ""),
        "day": plan_row.get("day", ""),
        **feasibility,
        **rationality,
        **unresolved_metrics,
        **split_metrics,
        **constraint_typing_metrics,
        **mismatch_metrics,
        **fairness_objective_metrics,
        "hard_constraint_pass_rate": round(hard_constraint_pass_rate, 4),
        "hard_constraint_severity_score": round(hard_constraint_severity_score, 4),
        "concession_count": concession_count,
        "average_group_satisfaction": round(capped_avg_score, 4),
        "minimum_member_satisfaction": round(capped_min_score, 4),
        "pre_cap_average_group_satisfaction": round(avg_score, 4),
        "pre_cap_minimum_member_satisfaction": round(min_score, 4),
        "satisfaction_cap_applied": bool(satisfaction_cap_applied),
        "maximum_member_satisfaction": round(max_score, 4),
        "satisfaction_variance": round(variance, 4),
        "satisfaction_std": round(std, 4),
        "max_min_satisfaction_gap": round(max_min_gap, 4),
        "max_min_fairness": round(max_min_fairness, 4) if not pd.isna(max_min_fairness) else np.nan,
        "raw_group_consensus_score": round(raw_group_consensus_score, 4)
        if not pd.isna(raw_group_consensus_score)
        else np.nan,
        "group_consensus_score": round(adjusted_group_consensus_score, 4)
        if not pd.isna(adjusted_group_consensus_score)
        else np.nan,
        "obvious_unfairness": obvious_unfairness,
        "unfairness_reasons": json.dumps(unfairness_reasons, ensure_ascii=False),
        "acceptance_vote_ratio": round(accepted_vote_ratio, 4),
        "negotiation_rounds": plan_row.get("negotiation_rounds", ""),
        "number_of_objections_resolved": plan_row.get("number_of_objections_resolved", ""),
        **consensus_metrics,
        "prompt_tokens": plan_row.get("prompt_tokens", ""),
        "completion_tokens": plan_row.get("completion_tokens", ""),
        "total_tokens": plan_row.get("total_tokens", ""),
        "runtime_seconds": plan_row.get("runtime_seconds", ""),
        "itinerary_text": plan_row.get("itinerary_text", ""),
        "coordinator_summary": plan_row.get("coordinator_summary", ""),
    }


def identify_success_and_failure_cases(plan_eval_df: pd.DataFrame) -> Dict[str, Any]:
    if plan_eval_df.empty:
        return {"success_cases": [], "failure_cases": []}
    df = plan_eval_df.copy()
    df["overall_rank_score"] = (
        _numeric_column(df, "group_consensus_score")
        + _numeric_column(df, "hard_constraint_severity_score") * 0.5
        + _numeric_column(df, "feasibility_score") * 0.2
        + _numeric_column(df, "rationality_score") * 0.2
        - _numeric_column(df, "unresolved_conflict_penalty") * 1.5
        - _numeric_column(df, "split_activity_cost") * 0.8
        - _numeric_column(df, "personalization_mismatch_count") * 1.0
        - _numeric_column(df, "satisfaction_std") * 0.2
    )
    success_df = df.sort_values(by=["overall_rank_score"], ascending=False).head(3)
    failure_df = df.sort_values(by=["overall_rank_score"], ascending=True).head(3)
    success_cases = []
    for _, row in success_df.iterrows():
        success_cases.append(
            {
                "family_id": row.get("family_id"),
                "plan_id": row.get("plan_id"),
                "why_successful": [
                    f"平均群体满意度为 {row.get('average_group_satisfaction')}",
                    f"最低成员满意度为 {row.get('minimum_member_satisfaction')}",
                    f"硬约束通过率为 {row.get('hard_constraint_pass_rate')}",
                    f"可行性通过率为 {row.get('feasibility_pass_rate')}",
                    f"行程合理性通过率为 {row.get('rationality_pass_rate')}",
                    f"群体共识分为 {row.get('group_consensus_score')}",
                ],
                "summary": safe_text(row.get("coordinator_summary", "")),
            }
        )
    failure_cases = []
    for _, row in failure_df.iterrows():
        reasons = []
        if row.get("obvious_unfairness") is True:
            reasons.append("存在明显不公平")
        if safe_float(row.get("hard_constraint_pass_rate"), 1) < 1:
            reasons.append("部分成员硬约束未满足")
        if safe_float(row.get("minimum_member_satisfaction"), 100) < LOW_MIN_SATISFACTION_THRESHOLD:
            reasons.append("最低成员满意度偏低")
        if safe_float(row.get("acceptance_vote_ratio"), 1) < 0.7:
            reasons.append("接受投票比例偏低")
        if safe_float(row.get("feasibility_pass_rate"), 1) < 0.8:
            reasons.append("基础可行性较弱")
        if safe_float(row.get("rationality_pass_rate"), 1) < 0.8:
            reasons.append("行程合理性较弱")
        failure_cases.append(
            {
                "family_id": row.get("family_id"),
                "plan_id": row.get("plan_id"),
                "why_failed": reasons,
                "average_group_satisfaction": row.get("average_group_satisfaction"),
                "minimum_member_satisfaction": row.get("minimum_member_satisfaction"),
                "hard_constraint_pass_rate": row.get("hard_constraint_pass_rate"),
                "feasibility_pass_rate": row.get("feasibility_pass_rate"),
                "rationality_pass_rate": row.get("rationality_pass_rate"),
                "summary": safe_text(row.get("coordinator_summary", "")),
            }
        )
    return {"success_cases": success_cases, "failure_cases": failure_cases}


def build_rationality_diagnostics(plan_eval_df: pd.DataFrame) -> Dict[str, Any]:
    if plan_eval_df.empty:
        return {"summary": "没有可诊断的行程合理性结果。", "details": {}}
    rationality_columns = {
        "diverse_attractions": "景点类型多样性",
        "diverse_restaurants": "餐饮类型多样性",
        "no_duplicate_poi": "无重复景点",
        "appropriate_visit_duration": "每日游览数量/时长合理",
        "reasonable_meal_prices": "餐饮价格合理",
        "travel_intensity_balance": "行程强度平衡",
    }
    details = {}
    issue_sentences = []
    for col, chinese_name in rationality_columns.items():
        if col not in plan_eval_df.columns:
            continue
        values = plan_eval_df[col].apply(lambda x: str(x).lower() in ["true", "1", "yes"])
        pass_rate = values.mean()
        failed_df = plan_eval_df[~values].copy()
        failed_family_ids = []
        if "family_id" in failed_df.columns:
            failed_family_ids = failed_df["family_id"].astype(int).tolist()
        details[col] = {
            "metric_name_cn": chinese_name,
            "pass_rate": round(pass_rate, 4),
            "failed_count": len(failed_family_ids),
            "failed_family_ids": failed_family_ids,
        }
        if pass_rate < 1.0:
            issue_sentences.append(
                f"{chinese_name}通过率为 {pass_rate:.2%}，失败家庭数量为 {len(failed_family_ids)}，失败 family_id={failed_family_ids}"
            )
    if not issue_sentences:
        summary = "所有行程合理性子指标均通过，景点、餐饮、强度和价格安排整体合理。"
    else:
        summary = "行程合理性问题主要包括：" + "；".join(issue_sentences) + "。"
    return {"summary": summary, "details": details}


def _numeric_column(plan_eval_df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in plan_eval_df.columns:
        return pd.Series([default] * len(plan_eval_df), index=plan_eval_df.index)
    return pd.to_numeric(plan_eval_df[column], errors="coerce").fillna(default)


def build_split_activity_diagnostics(plan_eval_df: pd.DataFrame) -> Dict[str, Any]:
    diagnostics = {
        "split_activity_families": [],
        "high_cost_split_activity_families": [],
        "summary": "",
    }
    if plan_eval_df.empty:
        diagnostics["summary"] = "没有可诊断的 split activity 结果。"
        return diagnostics
    split_count_series = _numeric_column(plan_eval_df, "split_activity_count")
    reasonable_series = _numeric_column(plan_eval_df, "reasonable_split_activity_count")
    unreasonable_series = _numeric_column(plan_eval_df, "unreasonable_split_activity_count")
    split_cost_series = _numeric_column(plan_eval_df, "split_activity_cost")
    split_df = plan_eval_df[split_count_series > 0].copy()
    high_cost_df = plan_eval_df[split_cost_series > MAX_ACCEPTABLE_SPLIT_ACTIVITY_COST].copy()
    for _, row in split_df.iterrows():
        diagnostics["split_activity_families"].append(
            {
                "family_id": safe_int(row.get("family_id"), 0),
                "split_activity_count": safe_int(row.get("split_activity_count"), 0),
                "reasonable_split_activity_count": safe_int(
                    row.get("reasonable_split_activity_count"), 0
                ),
                "unreasonable_split_activity_count": safe_int(
                    row.get("unreasonable_split_activity_count"), 0
                ),
                "split_activity_cost": safe_float(row.get("split_activity_cost"), 0),
                "split_activity_assessment": safe_text(row.get("split_activity_assessment", "")),
            }
        )
    for _, row in high_cost_df.iterrows():
        diagnostics["high_cost_split_activity_families"].append(
            {
                "family_id": safe_int(row.get("family_id"), 0),
                "split_activity_count": safe_int(row.get("split_activity_count"), 0),
                "unreasonable_split_activity_count": safe_int(
                    row.get("unreasonable_split_activity_count"), 0
                ),
                "split_activity_cost": safe_float(row.get("split_activity_cost"), 0),
                "weak_split_alternative_count": safe_int(
                    row.get("weak_split_alternative_count"), 0
                ),
                "long_or_vague_split_count": safe_int(row.get("long_or_vague_split_count"), 0),
                "missing_split_assignment_count": safe_int(
                    row.get("missing_split_assignment_count"), 0
                ),
                "missing_split_reason_count": safe_int(row.get("missing_split_reason_count"), 0),
            }
        )
    split_family_count = len(diagnostics["split_activity_families"])
    high_cost_count = len(diagnostics["high_cost_split_activity_families"])
    avg_cost = round(split_cost_series.mean(), 4) if len(split_cost_series) else 0.0
    avg_unreasonable = round(unreasonable_series.mean(), 4) if len(unreasonable_series) else 0.0
    avg_reasonable = round(reasonable_series.mean(), 4) if len(reasonable_series) else 0.0
    if split_family_count == 0:
        summary = "未检测到 split activity。"
    elif high_cost_count == 0 and avg_unreasonable == 0:
        summary = f"检测到 {split_family_count} 个家庭使用 split activity，平均合理分组数为 {avg_reasonable}，平均协调成本为 {avg_cost}。这些分组活动不视为失败，只在共识和合理性相关指标中轻微扣分。"
    else:
        high_cost_ids = [
            item["family_id"] for item in diagnostics["high_cost_split_activity_families"]
        ]
        summary = f"检测到 {split_family_count} 个家庭使用 split activity，其中高成本或不合理分组家庭数量为 {high_cost_count}，family_id={high_cost_ids}。合理分组活动可以通过，但替代活动过弱、成员分配不清或理由不足时会增加扣分。"
    diagnostics["summary"] = summary
    return diagnostics


def build_fairness_and_acceptance_diagnostics(
    plan_eval_df: pd.DataFrame, member_eval_df: pd.DataFrame
) -> Dict[str, Any]:
    diagnostics = {"unfair_families": [], "low_acceptance_families": [], "summary": ""}
    if plan_eval_df.empty:
        diagnostics["summary"] = "没有可诊断的 plan-level 评价结果。"
        return diagnostics
    for _, plan_row in plan_eval_df.iterrows():
        family_id = safe_int(plan_row.get("family_id"), 0)
        obvious_unfairness = bool_from_any(plan_row.get("obvious_unfairness"))
        acceptance_vote_ratio = safe_float(plan_row.get("acceptance_vote_ratio"), 0)
        minimum_member_satisfaction = safe_float(plan_row.get("minimum_member_satisfaction"), 0)
        average_group_satisfaction = safe_float(plan_row.get("average_group_satisfaction"), 0)
        family_member_df = member_eval_df[
            member_eval_df["family_id"].astype(int) == int(family_id)
        ].copy()
        lowest_member_info = {}
        rejected_members = []
        if not family_member_df.empty:
            family_member_df["individual_satisfaction_score_numeric"] = family_member_df[
                "individual_satisfaction_score"
            ].apply(lambda x: safe_float(x, 0))
            lowest_row = family_member_df.sort_values(
                by="individual_satisfaction_score_numeric", ascending=True
            ).iloc[0]
            lowest_member_info = {
                "member_id": safe_int(lowest_row.get("member_id"), 0),
                "agent_id": safe_text(lowest_row.get("agent_id", "")),
                "role": safe_text(lowest_row.get("role", "")),
                "individual_satisfaction_score": safe_float(
                    lowest_row.get("individual_satisfaction_score"), 0
                ),
                "accepted_vote": bool_from_any(lowest_row.get("accepted_vote")),
                "hard_constraint_pass": bool_from_any(lowest_row.get("hard_constraint_pass")),
                "hard_constraint_failed_reasons": safe_text(
                    lowest_row.get("hard_constraint_failed_reasons", "")
                ),
                "preference_coverage_rate": safe_float(
                    lowest_row.get("preference_coverage_rate"), 0
                ),
                "must_have_satisfaction_rate": safe_float(
                    lowest_row.get("must_have_satisfaction_rate"), 0
                ),
                "food_preference_score": safe_float(lowest_row.get("food_preference_score"), 0),
                "intensity_match_score": safe_float(lowest_row.get("intensity_match_score"), 0),
                "budget_match_score": safe_float(lowest_row.get("budget_match_score"), 0),
                "member_explanation": safe_text(lowest_row.get("member_explanation", "")),
            }
            for _, member_row in family_member_df.iterrows():
                accepted_vote = bool_from_any(member_row.get("accepted_vote"))
                if not accepted_vote:
                    rejected_members.append(
                        {
                            "member_id": safe_int(member_row.get("member_id"), 0),
                            "agent_id": safe_text(member_row.get("agent_id", "")),
                            "role": safe_text(member_row.get("role", "")),
                            "individual_satisfaction_score": safe_float(
                                member_row.get("individual_satisfaction_score"), 0
                            ),
                            "hard_constraint_pass": bool_from_any(
                                member_row.get("hard_constraint_pass")
                            ),
                            "main_reason": safe_text(member_row.get("member_explanation", "")),
                        }
                    )
        if obvious_unfairness:
            diagnostics["unfair_families"].append(
                {
                    "family_id": family_id,
                    "average_group_satisfaction": average_group_satisfaction,
                    "minimum_member_satisfaction": minimum_member_satisfaction,
                    "lowest_member": lowest_member_info,
                    "unfairness_reasons": safe_text(plan_row.get("unfairness_reasons", "")),
                }
            )
        if acceptance_vote_ratio < 1.0:
            diagnostics["low_acceptance_families"].append(
                {
                    "family_id": family_id,
                    "acceptance_vote_ratio": acceptance_vote_ratio,
                    "rejected_member_count": len(rejected_members),
                    "rejected_members": rejected_members,
                    "lowest_member": lowest_member_info,
                }
            )
    unfair_count = len(diagnostics["unfair_families"])
    low_acceptance_count = len(diagnostics["low_acceptance_families"])
    summary_parts = []
    if unfair_count > 0:
        unfair_family_ids = [item["family_id"] for item in diagnostics["unfair_families"]]
        first_unfair = diagnostics["unfair_families"][0]
        lowest = first_unfair.get("lowest_member", {})
        summary_parts.append(
            f"明显不公平案例数量为 {unfair_count}，family_id={unfair_family_ids}。其中 family_id={first_unfair.get('family_id')} 的最低满意度成员为 member_id={lowest.get('member_id')}, role={lowest.get('role')}, score={lowest.get('individual_satisfaction_score')}。主要问题：{lowest.get('member_explanation')}"
        )
    else:
        summary_parts.append("未发现明显不公平案例。")
    if low_acceptance_count > 0:
        low_acceptance_family_ids = [
            item["family_id"] for item in diagnostics["low_acceptance_families"]
        ]
        first_low = diagnostics["low_acceptance_families"][0]
        rejected_roles = [item.get("role") for item in first_low.get("rejected_members", [])]
        summary_parts.append(
            f"成员接受比例不足的家庭数量为 {low_acceptance_count}，family_id={low_acceptance_family_ids}。例如 family_id={first_low.get('family_id')} 的接受比例为 {first_low.get('acceptance_vote_ratio'):.2f}，未接受方案的成员角色包括 {rejected_roles}。"
        )
    else:
        summary_parts.append("所有家庭的成员接受比例均达到要求。")
    diagnostics["summary"] = " ".join(summary_parts)
    return diagnostics


def clean_summary_comment(text: Any) -> str:
    return safe_text(text).rstrip("。；; ")


def build_qgbmp_framework_summary(
    plan_eval_df: pd.DataFrame, member_eval_df: pd.DataFrame, metrics: Dict[str, Any]
) -> Dict[str, Any]:
    member_count = len(member_eval_df) if member_eval_df is not None else 0
    low_member_count = 0
    if (
        member_eval_df is not None
        and (not member_eval_df.empty)
        and ("individual_satisfaction_score" in member_eval_df.columns)
    ):
        low_member_count = int(
            (
                pd.to_numeric(member_eval_df["individual_satisfaction_score"], errors="coerce")
                < STRICT_LOW_SATISFACTION_THRESHOLD
            ).sum()
        )
    return {
        "framework_name": "QG-BMP",
        "framework_full_name": "Quality-Gated Backtracking Multi-Agent Planning",
        "components": {
            "proposal_quality_gate": {
                "description": "Evaluate each member proposal before negotiation using constraint awareness, preference specificity, role consistency, negotiation usefulness, feasibility, and conflict-risk control.",
                "formula": "0.30*hard_constraint_awareness + 0.20*preference_specificity + 0.15*role_consistency + 0.15*negotiation_usefulness + 0.10*feasibility + 0.10*conflict_risk_control",
            },
            "constraint_satisfiability_typing": {
                "description": "Classify family constraints into fully satisfiable, split-satisfiable, and unsatisfiable constraints.",
                "pass_rate": metrics.get("constraint_satisfiability_pass_rate"),
                "avg_split_satisfiable_constraint_count": metrics.get(
                    "avg_split_satisfiable_constraint_count"
                ),
                "avg_unsatisfiable_constraint_count": metrics.get(
                    "avg_unsatisfiable_constraint_count"
                ),
            },
            "fairness_aware_group_objective": {
                "description": "Optimize group satisfaction while protecting the least satisfied member and penalizing unfairness and hard-constraint violations.",
                "avg_objective": metrics.get("avg_fairness_aware_group_objective"),
                "low_satisfaction_member_count": low_member_count,
                "member_count": member_count,
            },
            "targeted_backtracking_with_reward_penalty_memory": {
                "description": "Revise only failed families and write positive or negative role-level memories rather than rerunning all successful families.",
                "expected_runtime_benefit": "reduces repeated LLM calls and preserves stable successful plans",
            },
        },
        "theoretical_contribution_summary": "The framework turns a linear LLM planning process into an evaluable, repairable, and partially backtrackable finite-state workflow with explicit constraint typing and fairness-aware group optimization.",
    }


def compute_repair_effect(
    pre_plan_eval_df: pd.DataFrame, post_plan_eval_df: pd.DataFrame
) -> Dict[str, Any]:
    """Repair-ablation diagnostic: how many families that failed BEFORE repair pass
    AFTER repair.  With Repair on this is positive; with Repair off it is 0 — a clear
    ablation signal even when the overall pass rates are close."""
    out: Dict[str, Any] = {}
    try:
        if (
            pre_plan_eval_df is None
            or post_plan_eval_df is None
            or pre_plan_eval_df.empty
            or post_plan_eval_df.empty
        ):
            return out
        pre = pre_plan_eval_df.drop_duplicates("family_id").set_index("family_id")
        post = post_plan_eval_df.drop_duplicates("family_id").set_index("family_id")
        pre_pass = pre.apply(is_plan_passed_requirements, axis=1)
        post_pass = post.apply(is_plan_passed_requirements, axis=1)
        failed_ids = [fid for fid in pre.index if not bool(pre_pass.get(fid, False))]
        fixed = sum(
            1 for fid in failed_ids if fid in post.index and bool(post_pass.get(fid, False))
        )
        out["pre_repair_pass_rate"] = round(float(pre_pass.mean()), 4)
        out["post_repair_pass_rate"] = round(float(post_pass.mean()), 4)
        out["repair_pass_rate_gain"] = round(
            out["post_repair_pass_rate"] - out["pre_repair_pass_rate"], 4
        )
        out["repair_failed_family_count"] = len(failed_ids)
        out["repair_fixed_family_count"] = int(fixed)
        out["repair_success_rate_on_failed"] = (
            round(fixed / len(failed_ids), 4) if failed_ids else 0.0
        )
    except Exception:
        pass
    return out


def build_overall_report(
    plan_eval_df: pd.DataFrame, member_eval_df: pd.DataFrame
) -> Dict[str, Any]:
    if plan_eval_df.empty:
        return {
            "overall_summary": "没有可评价的 plan。",
            "metrics": {},
            "success_cases": [],
            "failure_cases": [],
        }
    metrics = {
        "plan_count": len(plan_eval_df),
        "family_count": plan_eval_df["family_id"].nunique(),
        "avg_hard_constraint_pass_rate": round(
            pd.to_numeric(plan_eval_df["hard_constraint_pass_rate"], errors="coerce").mean(), 4
        ),
        "avg_budget_pass_rate": round(plan_eval_df["budget_pass"].mean(), 4),
        "avg_complete_information_rate": round(
            pd.to_numeric(plan_eval_df["complete_information_rate"], errors="coerce").mean(), 4
        ),
        "avg_no_hallucinated_poi_rate": round(
            pd.to_numeric(plan_eval_df["no_hallucinated_poi_rate"], errors="coerce").mean(), 4
        ),
        "avg_accommodation_completeness": round(
            pd.to_numeric(plan_eval_df["accommodation_completeness"], errors="coerce").mean(), 4
        ),
        "avg_transportation_feasibility": round(
            pd.to_numeric(plan_eval_df["transportation_feasibility"], errors="coerce").mean(), 4
        ),
        "hotel_nearest_requirement_pass_rate": round(
            plan_eval_df.get("hotel_nearest_requirement_pass", pd.Series(dtype=bool))
            .apply(bool_from_any)
            .mean(),
            4,
        ),
        "post_transport_chain_pass_rate": round(
            plan_eval_df.get("post_transport_chain_pass", pd.Series(dtype=bool))
            .apply(bool_from_any)
            .mean(),
            4,
        ),
        "avg_restaurant_specificity_rate": round(
            pd.to_numeric(
                plan_eval_df.get("restaurant_specificity_rate", pd.Series(dtype=float)),
                errors="coerce",
            ).mean(),
            4,
        ),
        "avg_objective_feasibility_micro": round(
            pd.to_numeric(
                plan_eval_df.get("objective_feasibility_micro", pd.Series(dtype=float)),
                errors="coerce",
            ).mean(),
            4,
        ),
        "avg_objective_rationality_micro": round(
            pd.to_numeric(
                plan_eval_df.get("objective_rationality_micro", pd.Series(dtype=float)),
                errors="coerce",
            ).mean(),
            4,
        ),
        "objective_final_pass_rate": round(
            plan_eval_df.get("objective_final_pass", pd.Series(dtype=bool))
            .apply(bool_from_any)
            .mean(),
            4,
        ),
        "avg_feasibility_pass_rate": round(
            pd.to_numeric(plan_eval_df["feasibility_pass_rate"], errors="coerce").mean(), 4
        ),
        "avg_rationality_pass_rate": round(
            pd.to_numeric(plan_eval_df["rationality_pass_rate"], errors="coerce").mean(), 4
        ),
        "avg_group_satisfaction": round(
            pd.to_numeric(plan_eval_df["average_group_satisfaction"], errors="coerce").mean(), 4
        ),
        "avg_minimum_member_satisfaction": round(
            pd.to_numeric(plan_eval_df["minimum_member_satisfaction"], errors="coerce").mean(), 4
        ),
        "avg_satisfaction_variance": round(
            pd.to_numeric(plan_eval_df["satisfaction_variance"], errors="coerce").mean(), 4
        ),
        "avg_max_min_fairness": round(
            pd.to_numeric(plan_eval_df["max_min_fairness"], errors="coerce").mean(), 4
        ),
        "avg_group_consensus_score": round(
            pd.to_numeric(plan_eval_df["group_consensus_score"], errors="coerce").mean(), 4
        ),
        "avg_fairness_aware_group_objective": round(
            pd.to_numeric(
                plan_eval_df.get("fairness_aware_group_objective", pd.Series(dtype=float)),
                errors="coerce",
            ).mean(),
            4,
        ),
        "constraint_satisfiability_pass_rate": round(
            plan_eval_df.get("constraint_satisfiability_pass", pd.Series(dtype=bool))
            .apply(bool_from_any)
            .mean(),
            4,
        ),
        "avg_split_satisfiable_constraint_count": round(
            pd.to_numeric(
                plan_eval_df.get("split_satisfiable_constraint_count", pd.Series(dtype=float)),
                errors="coerce",
            ).mean(),
            4,
        ),
        "avg_unsatisfiable_constraint_count": round(
            pd.to_numeric(
                plan_eval_df.get("unsatisfiable_constraint_count", pd.Series(dtype=float)),
                errors="coerce",
            ).mean(),
            4,
        ),
        "obvious_unfairness_rate": round(plan_eval_df["obvious_unfairness"].mean(), 4),
        "avg_acceptance_vote_ratio": round(
            pd.to_numeric(plan_eval_df["acceptance_vote_ratio"], errors="coerce").mean(), 4
        ),
        "avg_negotiation_rounds": round(
            pd.to_numeric(plan_eval_df["negotiation_rounds"], errors="coerce").mean(), 4
        ),
        "avg_total_tokens": round(
            pd.to_numeric(plan_eval_df["total_tokens"], errors="coerce").mean(), 4
        ),
        "avg_runtime_seconds": round(
            pd.to_numeric(plan_eval_df["runtime_seconds"], errors="coerce").mean(), 4
        ),
    }
    # Headline qualification rate (the number ablations compare; M7−M6 gap = repair value).
    try:
        metrics["pass_rate"] = round(
            plan_eval_df.apply(is_plan_passed_requirements, axis=1).mean(), 4
        )
    except Exception:
        pass
    # Single composite Grounding score = mean of the real-world grounding signals,
    # so reports/baselines can show one "Grounding" column.  (The near-impossible
    # "perfect grounding" gate itinerary_grounding_quality_pass_rate — always ~0 because
    # it demands ZERO generic/under-detailed activities — is excluded so it doesn't drag
    # the score to 0; it stays only as a per-plan soft diagnostic.)
    try:
        _g = [
            metrics.get(k)
            for k in ("avg_no_hallucinated_poi_rate", "avg_restaurant_specificity_rate")
        ]
        _g = [float(x) for x in _g if isinstance(x, (int, float))]
        if _g:
            metrics["grounding_score"] = round(sum(_g) / len(_g), 4)
    except Exception:
        pass
    # Memory-sensitive diagnostic: reward memory lets agents learn from prior
    # role-failures, lowering the average proposal attempts and raising the
    # first-attempt pass rate.  Read from the Module 2 proposals when available.
    try:
        from src.config import INDIVIDUAL_PROPOSALS_CSV

        if INDIVIDUAL_PROPOSALS_CSV.exists():
            _pdf = pd.read_csv(INDIVIDUAL_PROPOSALS_CSV)
            if "proposal_attempts" in _pdf.columns:
                _att = pd.to_numeric(_pdf["proposal_attempts"], errors="coerce")
                metrics["avg_proposal_attempts"] = round(_att.mean(), 4)
                metrics["proposal_first_attempt_pass_rate"] = round((_att <= 1).mean(), 4)
    except Exception:
        pass
    split_count_series = _numeric_column(plan_eval_df, "split_activity_count")
    split_cost_series = _numeric_column(plan_eval_df, "split_activity_cost")
    reasonable_split_series = _numeric_column(plan_eval_df, "reasonable_split_activity_count")
    unreasonable_split_series = _numeric_column(plan_eval_df, "unreasonable_split_activity_count")
    metrics.update(
        {
            "split_activity_usage_rate": round((split_count_series > 0).mean(), 4),
            "avg_split_activity_count": round(split_count_series.mean(), 4),
            "avg_reasonable_split_activity_count": round(reasonable_split_series.mean(), 4),
            "avg_unreasonable_split_activity_count": round(unreasonable_split_series.mean(), 4),
            "avg_split_activity_cost": round(split_cost_series.mean(), 4),
        }
    )
    metrics.update(
        {
            "avg_split_trigger_justified_count": round(
                _numeric_column(plan_eval_df, "split_trigger_justified_count").mean(), 4
            ),
            "avg_split_trigger_violation_count": round(
                _numeric_column(plan_eval_df, "split_trigger_violation_count").mean(), 4
            ),
            "split_trigger_justified_rate": round(
                _numeric_column(plan_eval_df, "split_trigger_justified_rate").mean(), 4
            ),
        }
    )
    # Specificity->failure reporting (items 2/4/5): expose meal/activity grounding honesty.
    metrics.update(
        {
            "avg_restaurant_failure_rate": round(
                _numeric_column(plan_eval_df, "restaurant_failure_rate").mean(), 4
            ),
            "avg_restaurant_sourced_specificity_rate": round(
                _numeric_column(plan_eval_df, "restaurant_sourced_specificity_rate").mean(), 4
            ),
            "restaurant_meal_grounding_pass_rate": round(
                plan_eval_df.get("restaurant_meal_grounding_pass", pd.Series(dtype=bool))
                .apply(bool_from_any)
                .mean(),
                4,
            ),
            "activity_grounding_pass_rate": round(
                plan_eval_df.get("activity_grounding_pass", pd.Series(dtype=bool))
                .apply(bool_from_any)
                .mean(),
                4,
            ),
            "avg_activity_specificity_failure_count": round(
                _numeric_column(plan_eval_df, "activity_specificity_failure_count").mean(), 4
            ),
        }
    )
    split_activity_diagnostics = build_split_activity_diagnostics(plan_eval_df)
    qgbmp_framework = build_qgbmp_framework_summary(
        plan_eval_df=plan_eval_df, member_eval_df=member_eval_df, metrics=metrics
    )
    comments = []
    if metrics["avg_hard_constraint_pass_rate"] >= 0.9:
        comments.append("大多数方案能够满足成员硬约束")
    else:
        comments.append("仍有较多方案未完全满足成员硬约束")
    if metrics["avg_feasibility_pass_rate"] >= 0.85:
        comments.append("整体可行性较好")
    else:
        comments.append("部分方案存在预算、交通、住宿或强度方面的可行性问题")
    if metrics.get("objective_final_pass_rate", 0) >= 0.9:
        comments.append("客观约束检查整体通过；最近合格酒店、后置交通串联和具体餐厅名作为软诊断项计入扣分，不再单独一票否决")
    else:
        comments.append("部分方案在客观约束检查中仍需改进；最近合格酒店选择、全日路线串联和具体餐厅名会作为软扣分项反馈给后续修复")
    rationality_diagnostics = build_rationality_diagnostics(plan_eval_df)
    if metrics["avg_rationality_pass_rate"] >= 0.85:
        comments.append("行程安排整体较合理")
    else:
        comments.append(rationality_diagnostics["summary"])
    if metrics.get("split_activity_usage_rate", 0) > 0:
        comments.append(split_activity_diagnostics["summary"])
    if metrics.get("avg_split_satisfiable_constraint_count", 0) > 0:
        comments.append("检测到可通过分流满足的成员约束冲突，系统已将部分硬约束冲突转化为局部参与规划问题")
    if metrics["avg_group_satisfaction"] >= 75:
        comments.append("群体平均满意度较高")
    elif metrics["avg_group_satisfaction"] >= 60:
        comments.append("群体平均满意度中等")
    else:
        comments.append("群体平均满意度偏低")
    fairness_acceptance_diagnostics = build_fairness_and_acceptance_diagnostics(
        plan_eval_df=plan_eval_df, member_eval_df=member_eval_df
    )
    if (
        metrics["obvious_unfairness_rate"] > 0
        or metrics["avg_acceptance_vote_ratio"] < 1.0
        or metrics["avg_minimum_member_satisfaction"] < LOW_MIN_SATISFACTION_THRESHOLD
    ):
        comments.append(fairness_acceptance_diagnostics["summary"])
    else:
        comments.append("明显不公平案例较少，且大部分成员倾向接受最终行程")
    cases = identify_success_and_failure_cases(plan_eval_df)
    return {
        "overall_summary": "；".join(
            (
                clean_summary_comment(comment)
                for comment in comments
                if clean_summary_comment(comment)
            )
        )
        + "。",
        "metrics": metrics,
        "rationality_diagnostics": rationality_diagnostics,
        "split_activity_diagnostics": split_activity_diagnostics,
        "fairness_acceptance_diagnostics": fairness_acceptance_diagnostics,
        "qgbmp_framework": qgbmp_framework,
        "success_cases": cases["success_cases"],
        "failure_cases": cases["failure_cases"],
    }


def bool_from_any(value) -> bool:
    if isinstance(value, bool):
        return value
    text = safe_text(value).lower()
    return text in ["true", "1", "yes", "y"]


def is_plan_passed_requirements(row) -> bool:
    hard_constraint_pass_rate = safe_float(row.get("hard_constraint_pass_rate"), 0)
    hard_constraint_severity_score = safe_float(
        row.get("hard_constraint_severity_score"), hard_constraint_pass_rate * 100
    )
    feasibility_pass_rate = safe_float(row.get("feasibility_pass_rate"), 0)
    rationality_pass_rate = safe_float(row.get("rationality_pass_rate"), 0)
    average_group_satisfaction = safe_float(row.get("average_group_satisfaction"), 0)
    minimum_member_satisfaction = safe_float(row.get("minimum_member_satisfaction"), 0)
    acceptance_vote_ratio = safe_float(row.get("acceptance_vote_ratio"), 0)
    group_consensus_score = safe_float(row.get("group_consensus_score"), 0)
    unresolved_penalty = safe_float(row.get("unresolved_conflict_penalty"), 0)
    mismatch_count = safe_int(row.get("personalization_mismatch_count"), 0)
    budget_pass = bool_from_any(row.get("budget_pass"))
    obvious_unfairness = bool_from_any(row.get("obvious_unfairness"))
    consensus_status = safe_text(row.get("consensus_status", ""))
    if consensus_status and consensus_status != "Full Consensus":
        unanimous_ok = (
            ACCEPT_NONFULL_CONSENSUS_WHEN_UNANIMOUS
            and acceptance_vote_ratio >= 1.0
            and minimum_member_satisfaction >= STRICT_MIN_MEMBER_SATISFACTION
            and hard_constraint_pass_rate >= 1.0
        )
        if not unanimous_ok:
            return False
    if hard_constraint_pass_rate < 1.0:
        return False
    if hard_constraint_severity_score < 95:
        return False
    if not budget_pass:
        return False
    if feasibility_pass_rate < MIN_FEASIBILITY_PASS_RATE:
        return False
    if rationality_pass_rate < MIN_RATIONALITY_PASS_RATE:
        return False
    if average_group_satisfaction < STRICT_MIN_GROUP_SATISFACTION:
        return False
    if minimum_member_satisfaction < STRICT_MIN_MEMBER_SATISFACTION:
        return False
    if acceptance_vote_ratio < 1.0:
        return False
    if group_consensus_score < MIN_GROUP_CONSENSUS_SCORE:
        return False
    if unresolved_penalty > MAX_ACCEPTABLE_UNRESOLVED_CONFLICT_PENALTY:
        return False
    if not bool_from_any(row.get("constraint_satisfiability_pass", True)):
        return False
    if safe_int(row.get("unsatisfiable_constraint_count"), 0) > 0:
        return False
    # High split cost is diagnostic. It should not by itself reject a plan when
    # hard constraints, budget, satisfaction, and acceptance are all acceptable.
    if mismatch_count > MAX_ACCEPTABLE_PERSONALIZATION_MISMATCH_COUNT:
        return False
    # Grounding gate (day-proportional, tolerant): a grounded plan names concrete
    # POIs (generic count ~0); a no-grounding plan leaves bare category labels on
    # most days and crosses this allowance — so the grounding ablation degrades the
    # headline pass rate instead of being masked as a soft penalty.
    generic_allowance = max(
        GENERIC_ACTIVITY_MIN_ALLOWANCE,
        int(GENERIC_ACTIVITY_PER_DAY_ALLOWANCE * max(safe_int(row.get("day"), 1), 1)),
    )
    if safe_int(row.get("generic_activity_label_count"), 0) > generic_allowance:
        return False
    if obvious_unfairness:
        return False
    return True


def get_failed_reasons(row) -> List[str]:
    reasons = []
    hard_constraint_pass_rate = safe_float(row.get("hard_constraint_pass_rate"), 0)
    hard_constraint_severity_score = safe_float(
        row.get("hard_constraint_severity_score"), hard_constraint_pass_rate * 100
    )
    feasibility_pass_rate = safe_float(row.get("feasibility_pass_rate"), 0)
    rationality_pass_rate = safe_float(row.get("rationality_pass_rate"), 0)
    average_group_satisfaction = safe_float(row.get("average_group_satisfaction"), 0)
    minimum_member_satisfaction = safe_float(row.get("minimum_member_satisfaction"), 0)
    acceptance_vote_ratio = safe_float(row.get("acceptance_vote_ratio"), 0)
    group_consensus_score = safe_float(row.get("group_consensus_score"), 0)
    unresolved_penalty = safe_float(row.get("unresolved_conflict_penalty"), 0)
    split_cost = safe_float(row.get("split_activity_cost"), 0)
    mismatch_count = safe_int(row.get("personalization_mismatch_count"), 0)
    budget_pass = bool_from_any(row.get("budget_pass"))
    obvious_unfairness = bool_from_any(row.get("obvious_unfairness"))
    consensus_status = safe_text(row.get("consensus_status", ""))
    if consensus_status and consensus_status != "Full Consensus":
        # Keep this consistent with is_plan_passed_requirements: a non-Full label is
        # NOT a failure (and not a repair target) when every member ultimately accepts
        # and the least-happy member is fine — otherwise the repair chases a noisy
        # label it cannot change instead of the real issue.
        unanimous_ok = (
            ACCEPT_NONFULL_CONSENSUS_WHEN_UNANIMOUS
            and acceptance_vote_ratio >= 1.0
            and minimum_member_satisfaction >= STRICT_MIN_MEMBER_SATISFACTION
            and hard_constraint_pass_rate >= 1.0
        )
        if not unanimous_ok:
            reasons.append(f"consensus_status is not Full Consensus: {consensus_status}")
    if hard_constraint_pass_rate < 1.0:
        reasons.append(f"hard_constraint_pass_rate too low: {hard_constraint_pass_rate}")
    if hard_constraint_severity_score < 95:
        reasons.append(f"hard_constraint_severity_score too low: {hard_constraint_severity_score}")
    if not budget_pass:
        reasons.append("budget_pass is False")
    if feasibility_pass_rate < MIN_FEASIBILITY_PASS_RATE:
        reasons.append(f"feasibility_pass_rate too low: {feasibility_pass_rate}")
    if rationality_pass_rate < MIN_RATIONALITY_PASS_RATE:
        reasons.append(f"rationality_pass_rate too low: {rationality_pass_rate}")
    if average_group_satisfaction < STRICT_MIN_GROUP_SATISFACTION:
        reasons.append(f"average_group_satisfaction too low: {average_group_satisfaction}")
    if minimum_member_satisfaction < STRICT_MIN_MEMBER_SATISFACTION:
        reasons.append(f"minimum_member_satisfaction too low: {minimum_member_satisfaction}")
    if acceptance_vote_ratio < 1.0:
        reasons.append(f"acceptance_vote_ratio too low: {acceptance_vote_ratio}")
    if group_consensus_score < MIN_GROUP_CONSENSUS_SCORE:
        reasons.append(f"group_consensus_score too low: {group_consensus_score}")
    if not bool_from_any(row.get("real_world_grounding_pass")):
        reasons.append("real_world_grounding_pass is False")
    if (
        safe_float(row.get("real_world_grounding_penalty"), 0)
        > MAX_ACCEPTABLE_REAL_WORLD_GROUNDING_PENALTY
    ):
        reasons.append(
            f"real_world_grounding_penalty too high: {safe_float(row.get('real_world_grounding_penalty'), 0)}"
        )
    if not bool_from_any(row.get("objective_feasibility_macro")):
        reasons.append("objective_feasibility_macro is False")
    if safe_float(row.get("objective_feasibility_micro"), 0) < MIN_OBJECTIVE_FEASIBILITY_MICRO:
        reasons.append(
            f"objective_feasibility_micro too low: {safe_float(row.get('objective_feasibility_micro'), 0)}"
        )
    if not bool_from_any(row.get("objective_rationality_macro")):
        reasons.append("objective_rationality_macro is False")
    if safe_float(row.get("objective_rationality_micro"), 0) < MIN_OBJECTIVE_RATIONALITY_MICRO:
        reasons.append(
            f"objective_rationality_micro too low: {safe_float(row.get('objective_rationality_micro'), 0)}"
        )
    if (
        safe_float(row.get("objective_evaluation_penalty"), 0)
        > MAX_ACCEPTABLE_OBJECTIVE_EVAL_PENALTY
    ):
        reasons.append(
            f"objective_evaluation_penalty too high: {safe_float(row.get('objective_evaluation_penalty'), 0)}"
        )
    soft_grounding_penalty = safe_float(row.get("grounding_quality_penalty"), 0)
    soft_placeholder_count = safe_int(row.get("placeholder_artifact_count"), 0)
    if (
        soft_grounding_penalty > MAX_SOFT_GROUNDING_QUALITY_PENALTY_FOR_WARNING
        or soft_placeholder_count > MAX_SOFT_PLACEHOLDER_ARTIFACTS_FOR_WARNING
    ):
        reasons.append(
            f"soft_grounding_quality_penalty high: {soft_grounding_penalty}; placeholder_artifact_count={soft_placeholder_count}"
        )
    if not bool_from_any(row.get("constraint_satisfiability_pass", True)):
        reasons.append("constraint_satisfiability_pass is False")
    if safe_int(row.get("unsatisfiable_constraint_count"), 0) > 0:
        reasons.append(
            f"unsatisfiable_constraint_count > 0: {safe_int(row.get('unsatisfiable_constraint_count'), 0)}"
        )
    _generic_allow = max(
        GENERIC_ACTIVITY_MIN_ALLOWANCE,
        int(GENERIC_ACTIVITY_PER_DAY_ALLOWANCE * max(safe_int(row.get("day"), 1), 1)),
    )
    if safe_int(row.get("generic_activity_label_count"), 0) > _generic_allow:
        reasons.append(
            f"generic_activity_label_count too high (未落地的通用地点过多): {safe_int(row.get('generic_activity_label_count'), 0)} > {_generic_allow}"
        )
    if unresolved_penalty > MAX_ACCEPTABLE_UNRESOLVED_CONFLICT_PENALTY:
        reasons.append(f"unresolved_conflict_penalty too high: {unresolved_penalty}")
    if split_cost > MAX_ACCEPTABLE_SPLIT_ACTIVITY_COST:
        reasons.append(f"split_activity_cost too high: {split_cost}")
    if mismatch_count > MAX_ACCEPTABLE_PERSONALIZATION_MISMATCH_COUNT:
        reasons.append(f"personalization_mismatch_count too high: {mismatch_count}")
    if obvious_unfairness:
        reasons.append("obvious_unfairness is True")
    return reasons


def find_failed_families(
    plan_eval_df: pd.DataFrame, test_first_n_failed_families: Optional[int] = None
) -> Dict[int, List[str]]:
    failed = {}
    for _, row in plan_eval_df.iterrows():
        family_id = safe_int(row.get("family_id"), 0)
        if not is_plan_passed_requirements(row):
            failed[family_id] = get_failed_reasons(row)
    failed_items = list(failed.items())
    if test_first_n_failed_families is not None:
        failed_items = failed_items[:test_first_n_failed_families]
    return dict(failed_items)


def compute_plan_quality_score(row) -> float:
    if row is None or len(row) == 0:
        return -999999.0
    hard_constraint_severity_score = safe_float(
        row.get("hard_constraint_severity_score"),
        safe_float(row.get("hard_constraint_pass_rate"), 0) * 100,
    )
    feasibility_score = safe_float(
        row.get("feasibility_score"), safe_float(row.get("feasibility_pass_rate"), 0) * 100
    )
    rationality_score = safe_float(
        row.get("rationality_score"), safe_float(row.get("rationality_pass_rate"), 0) * 100
    )
    average_group_satisfaction = safe_float(row.get("average_group_satisfaction"), 0)
    minimum_member_satisfaction = safe_float(row.get("minimum_member_satisfaction"), 0)
    max_min_fairness = safe_float(row.get("max_min_fairness"), 0)
    acceptance_vote_ratio = safe_float(row.get("acceptance_vote_ratio"), 0)
    group_consensus_score = safe_float(row.get("group_consensus_score"), 0)
    unresolved_penalty = safe_float(row.get("unresolved_conflict_penalty"), 0)
    split_cost = safe_float(row.get("split_activity_cost"), 0)
    route_penalty = safe_float(row.get("route_realism_penalty"), 0)
    real_world_penalty = safe_float(row.get("real_world_grounding_penalty"), 0)
    grounding_quality_penalty = safe_float(row.get("grounding_quality_penalty"), 0)
    placeholder_artifact_count = safe_int(row.get("placeholder_artifact_count"), 0)
    mismatch_count = safe_int(row.get("personalization_mismatch_count"), 0)
    budget_pass = bool_from_any(row.get("budget_pass"))
    obvious_unfairness = bool_from_any(row.get("obvious_unfairness"))
    score = 0.0
    score += group_consensus_score * 3.5
    score += hard_constraint_severity_score * 3.2
    score += minimum_member_satisfaction * 3.0
    score += average_group_satisfaction * 0.5
    score += feasibility_score * 0.8
    score += rationality_score * 0.7
    score += max_min_fairness * 100
    score += acceptance_vote_ratio * 120
    if budget_pass:
        score += 40
    else:
        score -= 100
    score -= unresolved_penalty * 5.0
    score -= split_cost * 3.0
    score -= route_penalty * 4.0
    score -= real_world_penalty * 2.0
    score -= grounding_quality_penalty * 2.0
    score -= placeholder_artifact_count * 5.0
    score -= mismatch_count * 8.0
    if obvious_unfairness:
        score -= 220
    return round(score, 4)


def get_plan_eval_row_by_family(plan_eval_df: pd.DataFrame, family_id: int) -> Dict[str, Any]:
    if plan_eval_df.empty:
        return {}
    family_df = plan_eval_df[plan_eval_df["family_id"].astype(int) == int(family_id)].copy()
    if family_df.empty:
        return {}
    return family_df.iloc[0].to_dict()


def get_member_eval_rows_by_family(
    member_eval_df: pd.DataFrame, family_id: int
) -> List[Dict[str, Any]]:
    if member_eval_df.empty:
        return []
    family_df = member_eval_df[member_eval_df["family_id"].astype(int) == int(family_id)].copy()
    return family_df.to_dict(orient="records")


def parse_hard_constraint_failed_reasons(value) -> List[str]:
    if isinstance(value, list):
        return [safe_text(item) for item in value if safe_text(item)]
    text = safe_text(value).strip()
    if not text or text in ["[]", "null", "None", "nan"]:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [safe_text(item) for item in parsed if safe_text(item)]
    except Exception:
        pass
    parsed_list = parse_json_list(text)
    if parsed_list:
        return [safe_text(item) for item in parsed_list if safe_text(item)]
    return [text]


def build_hard_constraint_fix_targets(
    member_id: int, role: str, failed_reasons: List[str]
) -> List[str]:
    targets = []
    for reason in failed_reasons:
        reason_text = safe_text(reason)
        reason_lower = reason_text.lower()
        if "must_include_not_satisfied" in reason_lower:
            keyword = reason_text.split(":", 1)[1] if ":" in reason_text else reason_text
            targets.append(
                f"针对 member_id={member_id}, role={role}: 必须把 must_include 关键词 '{keyword}' 原样写入 itinerary_text、per_day_itinerary 或 poi_tags。"
            )
        elif "must_avoid_violated" in reason_lower:
            keyword = reason_text.split(":", 1)[1] if ":" in reason_text else reason_text
            targets.append(
                f"针对 member_id={member_id}, role={role}: 必须从 itinerary_text、per_day_itinerary、poi_tags、meal_tags、hotel_plan 和 transportation_plan 中完全移除 must_avoid 关键词 '{keyword}'。"
            )
        elif "walking_distance_exceeded" in reason_lower:
            targets.append(
                f"针对 member_id={member_id}, role={role}: 必须降低 walking_distance_km，确保 walking_distance_km / day 不超过该成员 max_walking_distance_km。优先减少景点、增加地铁/出租车、加入休息。"
            )
        elif "return_time_too_late" in reason_lower:
            targets.append(
                f"针对 member_id={member_id}, role={role}: 必须提前 return_time，确保不晚于该成员 latest_return_time。必要时删除晚间活动。"
            )
        else:
            targets.append(f"针对 member_id={member_id}, role={role}: 必须修复硬约束失败原因：{reason_text}。")
    return targets


def build_current_hard_constraint_feedback(
    member_eval_rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    failed_members = []
    revision_targets = []
    for row in member_eval_rows:
        member_id = safe_int(row.get("member_id"), 0)
        agent_id = safe_text(row.get("agent_id", ""))
        role = safe_text(row.get("role", ""))
        hard_pass = bool_from_any(row.get("hard_constraint_pass"))
        failed_reasons = parse_hard_constraint_failed_reasons(
            row.get("hard_constraint_failed_reasons", "[]")
        )
        if not hard_pass or failed_reasons:
            failed_members.append(
                {
                    "member_id": member_id,
                    "agent_id": agent_id,
                    "role": role,
                    "hard_constraint_pass": hard_pass,
                    "hard_constraint_failed_reasons": failed_reasons,
                    "individual_satisfaction_score": safe_float(
                        row.get("individual_satisfaction_score"), 0
                    ),
                    "individual_rule_score": safe_float(
                        row.get("individual_rule_score"),
                        row.get("individual_satisfaction_score", 0),
                    ),
                    "llm_judge_score": safe_float(
                        row.get("llm_judge_score"), row.get("individual_satisfaction_score", 0)
                    ),
                }
            )
            revision_targets.extend(
                build_hard_constraint_fix_targets(
                    member_id=member_id, role=role, failed_reasons=failed_reasons
                )
            )
    revision_targets = list(dict.fromkeys(revision_targets))
    return {
        "has_hard_constraint_failure": len(failed_members) > 0,
        "failed_member_count": len(failed_members),
        "failed_members": failed_members,
        "hard_constraint_revision_targets": revision_targets,
        "hard_constraint_priority_note": "The next plan will fail automatically unless every listed hard constraint is fixed. Do not optimize satisfaction, diversity, or richness before these are satisfied.",
    }


def build_current_fairness_acceptance_feedback(
    plan_eval_row: Dict[str, Any], member_eval_rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    fairness_feedback = {
        "has_obvious_unfairness": False,
        "acceptance_vote_ratio": 1.0,
        "lowest_satisfaction_member": {},
        "rejected_members": [],
        "specific_fairness_problems": [],
        "specific_revision_targets": [],
    }
    if not plan_eval_row:
        return fairness_feedback
    obvious_unfairness = bool_from_any(plan_eval_row.get("obvious_unfairness"))
    acceptance_vote_ratio = safe_float(plan_eval_row.get("acceptance_vote_ratio"), 1.0)
    fairness_feedback["has_obvious_unfairness"] = obvious_unfairness
    fairness_feedback["acceptance_vote_ratio"] = acceptance_vote_ratio
    if not member_eval_rows:
        return fairness_feedback
    sorted_members = sorted(
        member_eval_rows, key=lambda row: safe_float(row.get("individual_satisfaction_score"), 0)
    )
    lowest = sorted_members[0]
    lowest_member = {
        "member_id": safe_int(lowest.get("member_id"), 0),
        "agent_id": safe_text(lowest.get("agent_id", "")),
        "role": safe_text(lowest.get("role", "")),
        "individual_satisfaction_score": safe_float(lowest.get("individual_satisfaction_score"), 0),
        "accepted_vote": bool_from_any(lowest.get("accepted_vote")),
        "hard_constraint_pass": bool_from_any(lowest.get("hard_constraint_pass")),
        "hard_constraint_failed_reasons": safe_text(
            lowest.get("hard_constraint_failed_reasons", "")
        ),
        "preference_coverage_rate": safe_float(lowest.get("preference_coverage_rate"), 0),
        "must_have_satisfaction_rate": safe_float(lowest.get("must_have_satisfaction_rate"), 0),
        "food_preference_score": safe_float(lowest.get("food_preference_score"), 0),
        "intensity_match_score": safe_float(lowest.get("intensity_match_score"), 0),
        "budget_match_score": safe_float(lowest.get("budget_match_score"), 0),
        "member_explanation": safe_text(lowest.get("member_explanation", "")),
    }
    fairness_feedback["lowest_satisfaction_member"] = lowest_member
    rejected_members = []
    for row in member_eval_rows:
        accepted_vote = bool_from_any(row.get("accepted_vote"))
        if not accepted_vote:
            rejected_members.append(
                {
                    "member_id": safe_int(row.get("member_id"), 0),
                    "agent_id": safe_text(row.get("agent_id", "")),
                    "role": safe_text(row.get("role", "")),
                    "individual_satisfaction_score": safe_float(
                        row.get("individual_satisfaction_score"), 0
                    ),
                    "hard_constraint_pass": bool_from_any(row.get("hard_constraint_pass")),
                    "member_explanation": safe_text(row.get("member_explanation", "")),
                }
            )
    fairness_feedback["rejected_members"] = rejected_members
    problems = []
    if obvious_unfairness:
        problems.append("当前方案存在明显不公平，需要优先保护最低满意度成员。")
    if acceptance_vote_ratio < 1.0:
        problems.append(f"当前成员接受比例不足，acceptance_vote_ratio={acceptance_vote_ratio}。")
    if lowest_member:
        problems.append(
            f"最低满意度成员是 member_id={lowest_member['member_id']}, role={lowest_member['role']}, score={lowest_member['individual_satisfaction_score']}。"
        )
        if not lowest_member["hard_constraint_pass"]:
            problems.append("该最低满意度成员硬约束未满足：" + lowest_member["hard_constraint_failed_reasons"])
        if lowest_member["intensity_match_score"] < 70:
            problems.append("该最低满意度成员对行程强度不满意，需要降低强度、减少步行、增加休息。")
        if lowest_member["food_preference_score"] < 70:
            problems.append("该最低满意度成员对餐饮不满意，需要调整餐厅类型、meal_tags 或价格等级。")
        if lowest_member["budget_match_score"] < 70:
            problems.append("该最低满意度成员对预算不满意，需要降低餐饮、酒店或交通成本。")
        if lowest_member["preference_coverage_rate"] < 0.7:
            problems.append("该最低满意度成员偏好覆盖不足，需要加入其核心偏好。")
        if lowest_member["must_have_satisfaction_rate"] < 1.0:
            problems.append("该最低满意度成员 must-have 没有完全满足，需要原样写入对应关键词。")
    fairness_feedback["specific_fairness_problems"] = problems
    targets = []
    if lowest_member:
        member_id = lowest_member["member_id"]
        role = lowest_member["role"]
        targets.append(f"优先保护最低满意度成员 member_id={member_id}, role={role}，不要只提高平均满意度。")
        if lowest_member["intensity_match_score"] < 70:
            targets.append(
                f"针对 member_id={member_id}, role={role}: 每天核心 POI 控制在 2-3 个以内，增加休息块，降低 walking_distance_km。"
            )
        if lowest_member["food_preference_score"] < 70:
            targets.append(
                f"针对 member_id={member_id}, role={role}: 调整餐饮安排，使 meal_tags、restaurant_price_level 和餐厅类型更符合该成员。"
            )
        if lowest_member["budget_match_score"] < 70:
            targets.append(
                f"针对 member_id={member_id}, role={role}: 优先选择 low 或 medium 餐饮和酒店，避免高价安排。"
            )
        if not lowest_member["hard_constraint_pass"]:
            targets.append(f"针对 member_id={member_id}, role={role}: 必须先修复硬约束失败原因。")
    for member in rejected_members:
        targets.append(
            f"未接受方案成员 member_id={member['member_id']}, role={member['role']}，需要根据其 member_explanation 修改方案。"
        )
    fairness_feedback["specific_revision_targets"] = list(dict.fromkeys(targets))
    return fairness_feedback


def build_current_unresolved_conflict_feedback(
    plan_eval_row: Dict[str, Any], original_plan: Dict[str, Any]
) -> Dict[str, Any]:
    """Build compact feedback for unresolved conflicts that must be carried into Module 5 revision."""
    unresolved = parse_json_text(original_plan.get("unresolved_conflicts_json", "[]"), default=[])
    if not isinstance(unresolved, list):
        unresolved = []
    resolved = parse_json_text(original_plan.get("resolved_conflicts_json", "[]"), default=[])
    if not isinstance(resolved, list):
        resolved = []
    split_activities = parse_json_text(original_plan.get("split_activities_json", "[]"), default=[])
    if not isinstance(split_activities, list):
        split_activities = []
    constraint_satisfiability = parse_json_text(
        original_plan.get("constraint_satisfiability_json", "{}"), default={}
    )
    if not isinstance(constraint_satisfiability, dict):
        constraint_satisfiability = {}
    conflict_type_counts = {}
    if plan_eval_row:
        conflict_type_counts = parse_json_text(
            plan_eval_row.get("unresolved_conflict_type_counts_json", "{}"), default={}
        )
        if not isinstance(conflict_type_counts, dict):
            conflict_type_counts = {}
    unresolved_items = []
    for item in unresolved[:6]:
        if isinstance(item, dict):
            unresolved_items.append(
                {
                    "conflict_type": safe_text(item.get("conflict_type", "unknown")) or "unknown",
                    "reason": safe_text(item.get("reason", ""))[:220],
                    "conflicting_terms": item.get(
                        "conflicting_terms", item.get("terms", item.get("term", []))
                    ),
                    "must_include_role": item.get("must_include_role") or item.get("include_role"),
                    "must_avoid_role": item.get("must_avoid_role") or item.get("avoid_role"),
                }
            )
        else:
            unresolved_items.append({"conflict_type": "unknown", "reason": safe_text(item)[:220]})
    weak_split_items = []
    for item in split_activities[:5]:
        if not isinstance(item, dict):
            continue
        participants = item.get("participating_roles") or item.get("participating_agents") or []
        non_participants = (
            item.get("non_participating_roles") or item.get("non_participating_agents") or []
        )
        alternative = safe_text(
            item.get("alternative_for_non_participants", "") or item.get("alternative_activity", "")
        )
        role_assignment = safe_text(item.get("role_assignment", ""))
        if (
            (not participants)
            or (not non_participants)
            or len(alternative) < 10
            or ("部分成员" in role_assignment)
            or ("其他成员" in role_assignment)
        ):
            weak_split_items.append(
                {
                    "activity": safe_text(item.get("activity") or item.get("activity_or_term")),
                    "conflict_type": safe_text(item.get("conflict_type")),
                    "participating_roles": participants,
                    "non_participating_roles": non_participants,
                    "alternative_for_non_participants": alternative,
                    "role_assignment": role_assignment[:220],
                }
            )
    feedback = {
        "has_unresolved_conflicts": len(unresolved_items) > 0,
        "unresolved_conflict_count": safe_int(
            plan_eval_row.get("unresolved_conflict_count"), len(unresolved_items)
        )
        if plan_eval_row
        else len(unresolved_items),
        "unresolved_direct_conflict_count": safe_int(
            plan_eval_row.get("unresolved_direct_conflict_count"), 0
        )
        if plan_eval_row
        else 0,
        "unresolved_conflict_penalty": safe_float(
            plan_eval_row.get("unresolved_conflict_penalty"), 0
        )
        if plan_eval_row
        else 0,
        "unresolved_conflict_type_counts": conflict_type_counts,
        "unresolved_conflicts": unresolved_items,
        "resolved_conflicts_preview": resolved[:3],
        "split_activity_cost": safe_float(plan_eval_row.get("split_activity_cost"), 0)
        if plan_eval_row
        else 0,
        "split_activity_count": safe_int(
            plan_eval_row.get("split_activity_count"), len(split_activities)
        )
        if plan_eval_row
        else len(split_activities),
        "reasonable_split_activity_count": safe_int(
            plan_eval_row.get("reasonable_split_activity_count"), 0
        )
        if plan_eval_row
        else 0,
        "unreasonable_split_activity_count": safe_int(
            plan_eval_row.get("unreasonable_split_activity_count"), 0
        )
        if plan_eval_row
        else 0,
        "weak_or_incomplete_split_activities": weak_split_items,
        "constraint_satisfiability_overall": constraint_satisfiability.get(
            "overall_satisfiability"
        ),
        "unsatisfiable_constraint_count": safe_int(
            plan_eval_row.get("unsatisfiable_constraint_count"),
            constraint_satisfiability.get("unsatisfiable_count", 0),
        )
        if plan_eval_row
        else safe_int(constraint_satisfiability.get("unsatisfiable_count"), 0),
        "split_satisfiable_constraint_count": safe_int(
            plan_eval_row.get("split_satisfiable_constraint_count"),
            constraint_satisfiability.get("split_satisfiable_count", 0),
        )
        if plan_eval_row
        else safe_int(constraint_satisfiability.get("split_satisfiable_count"), 0),
        "revision_targets": [],
    }
    targets = []
    if feedback["has_unresolved_conflicts"]:
        targets.append(
            "必须优先处理 unresolved_conflicts：每个直接 must_include vs must_avoid 冲突必须变成完整 split_activities，或保留在 unresolved_conflicts 且不得标记 Full Consensus。"
        )
    for item in unresolved_items[:5]:
        ctype = safe_text(item.get("conflict_type", "unknown"))
        terms = item.get("conflicting_terms")
        targets.append(
            f"处理未解决冲突 {ctype}: conflicting_terms={terms}。如果可分流，写清 participating_roles、non_participating_roles、alternative_for_non_participants、role_assignment 和 reason。"
        )
    if weak_split_items:
        targets.append("修复不完整 split activity：必须补齐参与角色、不参与角色、替代活动、集合/重合流说明，并去掉“部分成员/其他成员”等模糊表达。")
    if feedback["split_activity_cost"] > MAX_ACCEPTABLE_SPLIT_ACTIVITY_COST:
        targets.append("split_activity_cost 过高：只保留真正解决硬冲突的分流活动，删除普通偏好差异导致的分流活动。")
    if feedback["unsatisfiable_constraint_count"] > 0:
        targets.append(
            "存在 Unsatisfiable Constraints：不要假装完全解决；若无法通过分流解决，需要在 unresolved_conflicts 和 coordinator_summary 中说明。"
        )
    feedback["revision_targets"] = list(dict.fromkeys(targets))
    return feedback


def build_critical_hard_constraint_plain_text(hard_constraint_feedback: Dict[str, Any]) -> str:
    failed_members = hard_constraint_feedback.get("failed_members", [])
    targets = hard_constraint_feedback.get("hard_constraint_revision_targets", [])
    if not failed_members and (not targets):
        return "CRITICAL_HARD_CONSTRAINT_FIXES: No hard constraint failures detected."
    lines = [
        "CRITICAL_HARD_CONSTRAINT_FIXES:",
        "The next candidate must fix the exact failed reasons below. If these are not fixed, the member score may be capped and the plan will remain rejected.",
    ]
    for member in failed_members:
        member_id = member.get("member_id")
        role = member.get("role")
        agent_id = member.get("agent_id")
        reasons = member.get("hard_constraint_failed_reasons", [])
        if not isinstance(reasons, list):
            reasons = parse_hard_constraint_failed_reasons(reasons)
        lines.append(
            f"- member_id={member_id}, role={role}, agent_id={agent_id}, hard_constraint_failed_reasons={json.dumps(reasons, ensure_ascii=False)}"
        )
        for reason in reasons:
            reason_text = safe_text(reason)
            reason_lower = reason_text.lower()
            if "return_time_too_late" in reason_lower:
                lines.append(
                    "  Required fix: set return_time and every per-day estimated_return_time no later than this member's latest_return_time."
                )
            elif "walking_distance_exceeded" in reason_lower:
                lines.append(
                    "  Required fix: lower walking_distance_km and per-day estimated_walking_distance_km below this member's walking limit."
                )
            elif "must_avoid_violated" in reason_lower:
                lines.append(
                    "  Required fix: remove the exact must_avoid keyword from itinerary_text, per_day_itinerary, poi_tags, meal_tags, hotel_plan, and transportation_plan."
                )
            elif "must_include_not_satisfied" in reason_lower:
                lines.append(
                    "  Required fix: copy the exact must_include keyword into itinerary_text, per_day_itinerary, or poi_tags."
                )
            else:
                lines.append(f"  Required fix: fix this exact reason: {reason_text}")
    if targets:
        lines.append("Concrete hard-constraint revision targets:")
        for target in targets[:8]:
            lines.append(f"- {safe_text(target)}")
    return "\n".join(lines)


def build_module6_feedback_prompt(
    family_id: int,
    plan_eval_df: pd.DataFrame,
    member_eval_df: pd.DataFrame,
    original_plan: Dict[str, Any],
    failure_reasons: List[str],
    revision_round: int,
    candidate_index: int,
) -> str:
    plan_eval_row = get_plan_eval_row_by_family(plan_eval_df=plan_eval_df, family_id=family_id)
    member_eval_rows = get_member_eval_rows_by_family(
        member_eval_df=member_eval_df, family_id=family_id
    )
    hard_constraint_feedback = build_current_hard_constraint_feedback(
        member_eval_rows=member_eval_rows
    )
    critical_hard_constraint_text = build_critical_hard_constraint_plain_text(
        hard_constraint_feedback
    )
    fairness_acceptance_feedback = build_current_fairness_acceptance_feedback(
        plan_eval_row=plan_eval_row, member_eval_rows=member_eval_rows
    )
    unresolved_conflict_feedback = build_current_unresolved_conflict_feedback(
        plan_eval_row=plan_eval_row, original_plan=original_plan
    )
    last_plan_summary = {
        "family_id": original_plan.get("family_id"),
        "plan_id": original_plan.get("plan_id"),
        "method": original_plan.get("method"),
        "total_cost": original_plan.get("total_cost"),
        "budget_limit": original_plan.get("budget_limit"),
        "walking_distance_km": original_plan.get("walking_distance_km"),
        "route_distance_km": original_plan.get("route_distance_km"),
        "daily_poi_count": original_plan.get("daily_poi_count"),
        "restaurant_price_level": original_plan.get("restaurant_price_level"),
        "return_time": original_plan.get("return_time"),
        "poi_tags": original_plan.get("poi_tags"),
        "meal_tags": original_plan.get("meal_tags"),
        "high_intensity_tags": original_plan.get("high_intensity_tags"),
        "split_activities_json": safe_text(original_plan.get("split_activities_json", ""))[:1200],
        "unresolved_conflicts_json": safe_text(original_plan.get("unresolved_conflicts_json", ""))[
            :1200
        ],
        "constraint_satisfiability_json": safe_text(
            original_plan.get("constraint_satisfiability_json", "")
        )[:900],
        "hard_constraint_checklist_json": original_plan.get("hard_constraint_checklist_json", ""),
        "hotel_plan_json": safe_text(original_plan.get("hotel_plan_json", ""))[:1000],
        "transportation_plan_json": safe_text(original_plan.get("transportation_plan_json", ""))[
            :1200
        ],
        "per_day_itinerary_json": safe_text(original_plan.get("per_day_itinerary_json", ""))[:1400],
        "itinerary_text": safe_text(original_plan.get("itinerary_text", ""))[:2200],
    }
    plan_level_issues = []
    if plan_eval_row:
        checks = [
            (
                "hard_constraint_pass_rate",
                safe_float(plan_eval_row.get("hard_constraint_pass_rate"), 0),
                1.0,
                "硬约束通过率不足",
            ),
            (
                "feasibility_pass_rate",
                safe_float(plan_eval_row.get("feasibility_pass_rate"), 0),
                0.85,
                "可行性不足",
            ),
            (
                "rationality_pass_rate",
                safe_float(plan_eval_row.get("rationality_pass_rate"), 0),
                0.85,
                "行程合理性不足",
            ),
            (
                "average_group_satisfaction",
                safe_float(plan_eval_row.get("average_group_satisfaction"), 0),
                75,
                "群体平均满意度不足",
            ),
            (
                "minimum_member_satisfaction",
                safe_float(plan_eval_row.get("minimum_member_satisfaction"), 0),
                65,
                "最低成员满意度不足",
            ),
            (
                "acceptance_vote_ratio",
                safe_float(plan_eval_row.get("acceptance_vote_ratio"), 0),
                0.75,
                "成员接受比例不足",
            ),
        ]
        for metric_name, value, threshold, reason_cn in checks:
            if value < threshold:
                plan_level_issues.append(
                    {
                        "metric": metric_name,
                        "value": value,
                        "threshold": threshold,
                        "problem": reason_cn,
                    }
                )
        if not bool_from_any(plan_eval_row.get("budget_pass")):
            plan_level_issues.append(
                {"metric": "budget_pass", "value": False, "threshold": True, "problem": "预算不通过"}
            )
        if bool_from_any(plan_eval_row.get("obvious_unfairness")):
            plan_level_issues.append(
                {
                    "metric": "obvious_unfairness",
                    "value": True,
                    "threshold": False,
                    "problem": "存在明显不公平",
                }
            )
        unresolved_penalty = safe_float(plan_eval_row.get("unresolved_conflict_penalty"), 0)
        unresolved_direct_count = safe_int(plan_eval_row.get("unresolved_direct_conflict_count"), 0)
        if (
            unresolved_direct_count > 0
            or unresolved_penalty > MAX_ACCEPTABLE_UNRESOLVED_CONFLICT_PENALTY
        ):
            plan_level_issues.append(
                {
                    "metric": "unresolved_conflicts",
                    "value": {
                        "unresolved_direct_conflict_count": unresolved_direct_count,
                        "unresolved_conflict_penalty": unresolved_penalty,
                    },
                    "threshold": {
                        "unresolved_direct_conflict_count": 0,
                        "unresolved_conflict_penalty": MAX_ACCEPTABLE_UNRESOLVED_CONFLICT_PENALTY,
                    },
                    "problem": "存在未解决硬冲突或未解决冲突惩罚过高",
                }
            )
        split_cost = safe_float(plan_eval_row.get("split_activity_cost"), 0)
        if split_cost > MAX_ACCEPTABLE_SPLIT_ACTIVITY_COST:
            plan_level_issues.append(
                {
                    "metric": "split_activity_cost",
                    "value": split_cost,
                    "threshold": MAX_ACCEPTABLE_SPLIT_ACTIVITY_COST,
                    "problem": "分流活动协调成本过高，需要减少或补全分流活动",
                }
            )
        if safe_int(plan_eval_row.get("unsatisfiable_constraint_count"), 0) > 0:
            plan_level_issues.append(
                {
                    "metric": "unsatisfiable_constraint_count",
                    "value": safe_int(plan_eval_row.get("unsatisfiable_constraint_count"), 0),
                    "threshold": 0,
                    "problem": "存在不可满足约束，不能假装完全共识",
                }
            )
        if not bool_from_any(plan_eval_row.get("itinerary_grounding_quality_pass", True)):
            plan_level_issues.append(
                {
                    "metric": "itinerary_grounding_quality_pass",
                    "value": False,
                    "threshold": True,
                    "problem": "真实行程具体化不足：作为软扣分项，不直接判失败",
                }
            )
        if not bool_from_any(plan_eval_row.get("hotel_nearest_requirement_pass", True)):
            plan_level_issues.append(
                {
                    "metric": "hotel_nearest_requirement_pass",
                    "value": False,
                    "threshold": True,
                    "problem": "酒店最近性说明不足：作为软扣分项",
                }
            )
        if not bool_from_any(plan_eval_row.get("post_transport_chain_pass", True)):
            plan_level_issues.append(
                {
                    "metric": "post_transport_chain_pass",
                    "value": False,
                    "threshold": True,
                    "problem": "交通串联不完整：作为软扣分项",
                }
            )
        if safe_float(plan_eval_row.get("restaurant_specificity_rate"), 1.0) < 0.7:
            plan_level_issues.append(
                {
                    "metric": "restaurant_specificity_rate",
                    "value": safe_float(plan_eval_row.get("restaurant_specificity_rate"), 0),
                    "threshold": 0.7,
                    "problem": "具体餐厅名不足：允许输出范围附近，但作为软扣分项",
                }
            )
        if safe_int(plan_eval_row.get("restaurant_area_fallback_count"), 0) > 0:
            plan_level_issues.append(
                {
                    "metric": "restaurant_area_fallback_count",
                    "value": safe_int(plan_eval_row.get("restaurant_area_fallback_count"), 0),
                    "threshold": 0,
                    "problem": "部分餐次未查到具体餐厅名，仅输出范围附近，作为软扣分项",
                }
            )
        if safe_int(plan_eval_row.get("placeholder_artifact_count"), 0) > 0:
            plan_level_issues.append(
                {
                    "metric": "placeholder_artifact_count",
                    "value": safe_int(plan_eval_row.get("placeholder_artifact_count"), 0),
                    "threshold": 0,
                    "problem": "输出中仍有占位词或规则兜底文本：作为软扣分项",
                }
            )
        if safe_text(plan_eval_row.get("grounding_quality_issues_json")) not in [
            "",
            "[]",
            "null",
            "None",
        ]:
            plan_level_issues.append(
                {
                    "metric": "grounding_quality_issues_json",
                    "value": safe_text(plan_eval_row.get("grounding_quality_issues_json"))[:500],
                    "threshold": "[]",
                    "problem": "真实行程具体化诊断发现软扣分项",
                }
            )
        rationality_sub_metrics = [
            ("diverse_attractions", "景点类型多样性不足"),
            ("diverse_restaurants", "餐饮类型多样性不足"),
            ("no_duplicate_poi", "存在重复景点"),
            ("appropriate_visit_duration", "每日游览数量或时长不合理"),
            ("reasonable_meal_prices", "餐饮价格不合理"),
            ("travel_intensity_balance", "行程强度不平衡"),
        ]
        for col, problem_cn in rationality_sub_metrics:
            if col in plan_eval_row and (not bool_from_any(plan_eval_row.get(col))):
                plan_level_issues.append(
                    {
                        "metric": col,
                        "value": plan_eval_row.get(col),
                        "threshold": True,
                        "problem": problem_cn,
                    }
                )
    dissatisfied_members = []
    for row in member_eval_rows:
        satisfaction = safe_float(row.get("individual_satisfaction_score"), 0)
        accepted_vote = bool_from_any(row.get("accepted_vote"))
        hard_constraint_pass = bool_from_any(row.get("hard_constraint_pass"))
        hard_failed_reasons = parse_hard_constraint_failed_reasons(
            row.get("hard_constraint_failed_reasons", "[]")
        )
        must_have_rate = safe_float(row.get("must_have_satisfaction_rate"), 0)
        preference_coverage = safe_float(row.get("preference_coverage_rate"), 0)
        food_score = safe_float(row.get("food_preference_score"), 0)
        intensity_score = safe_float(row.get("intensity_match_score"), 0)
        budget_score = safe_float(row.get("budget_match_score"), 0)
        member_problems = []
        if not hard_constraint_pass:
            member_problems.append(
                "硬约束未通过，具体失败原因 hard_constraint_failed_reasons="
                + json.dumps(hard_failed_reasons, ensure_ascii=False)
            )
        if satisfaction < 70:
            member_problems.append(f"个人满意度低，score={satisfaction}")
        if not accepted_vote:
            member_problems.append("该成员没有接受当前方案")
        if must_have_rate < 1.0:
            member_problems.append(f"must-have 满足率不足，rate={must_have_rate}")
        if preference_coverage < 0.7:
            member_problems.append(f"偏好覆盖率不足，rate={preference_coverage}")
        if food_score < 70:
            member_problems.append(f"餐饮偏好分数低，score={food_score}")
        if intensity_score < 70:
            member_problems.append(f"行程强度匹配分数低，score={intensity_score}")
        if budget_score < 70:
            member_problems.append(f"预算匹配分数低，score={budget_score}")
        if member_problems:
            dissatisfied_members.append(
                {
                    "member_id": row.get("member_id"),
                    "agent_id": row.get("agent_id"),
                    "role": row.get("role"),
                    "individual_satisfaction_score": satisfaction,
                    "individual_rule_score": row.get("individual_rule_score"),
                    "llm_judge_score": row.get("llm_judge_score"),
                    "accepted_vote": accepted_vote,
                    "hard_constraint_pass": hard_constraint_pass,
                    "hard_constraint_failed_reasons": hard_failed_reasons,
                    "preference_coverage_rate": preference_coverage,
                    "must_have_satisfaction_rate": must_have_rate,
                    "food_preference_score": food_score,
                    "intensity_match_score": intensity_score,
                    "budget_match_score": budget_score,
                    "problems": member_problems,
                    "member_explanation": safe_text(row.get("member_explanation", ""))[:900],
                }
            )
    concrete_revision_targets = []
    concrete_revision_targets.extend(
        hard_constraint_feedback.get("hard_constraint_revision_targets", [])
    )
    concrete_revision_targets.extend(unresolved_conflict_feedback.get("revision_targets", []))
    if plan_eval_row:
        if not bool_from_any(plan_eval_row.get("hotel_nearest_requirement_pass", True)):
            concrete_revision_targets.append(
                "重新选择酒店：必须围绕完整行程主景点/每日路线链选择最近的合格真实酒店，写清酒店名、区域、附近站点、为何离主行程最近、预算是否可行。"
            )
        if not bool_from_any(plan_eval_row.get("post_transport_chain_pass", True)):
            concrete_revision_targets.append(
                "重新生成交通：必须在酒店、景点、餐厅确定后，按每天“酒店->上午景点->午餐->下午景点->晚餐->酒店”串联路线，写清起点、站点/线路或打车段、时间和费用。"
            )
        if safe_float(plan_eval_row.get("restaurant_specificity_rate"), 1.0) < 0.7:
            concrete_revision_targets.append(
                "重新具体化餐厅：尽量使用真实餐厅名；如果搜索不到，不要编造店名，保持 restaurant_name 为空，只给出当天景点/酒店站点附近的用餐范围、菜系、人均和来源限制。"
            )
        if safe_int(plan_eval_row.get("placeholder_artifact_count"), 0) > 0:
            concrete_revision_targets.append(
                "删除交通/景点占位词，例如“短路线需求”“短途出租车”“core attraction”；餐厅查不到具体名时允许范围兜底，但必须标记为无具体餐厅名并接受软扣分。"
            )
    for member in dissatisfied_members:
        role = member.get("role")
        member_id = member.get("member_id")
        for problem in member.get("problems", []):
            problem_text = safe_text(problem).lower()
            if "walking" in problem_text or "步行" in problem_text:
                concrete_revision_targets.append(
                    f"针对 member_id={member_id}, role={role}: 降低每日步行距离，减少长距离步行，增加出租车/地铁/短途交通。"
                )
            elif "return" in problem_text or "返回" in problem_text or "latest" in problem_text:
                concrete_revision_targets.append(
                    f"针对 member_id={member_id}, role={role}: 提前每天返回酒店时间，避免晚归。"
                )
            elif "强度" in problem_text or "intensity" in problem_text:
                concrete_revision_targets.append(
                    f"针对 member_id={member_id}, role={role}: 降低行程强度，每天核心 POI 不超过 2-3 个，加入休息时间。"
                )
            elif "餐饮" in problem_text or "food" in problem_text or "meal" in problem_text:
                concrete_revision_targets.append(
                    f"针对 member_id={member_id}, role={role}: 调整餐饮安排，使 meal_tags 和餐厅类型更符合该成员饮食偏好。"
                )
            elif "预算" in problem_text or "budget" in problem_text:
                concrete_revision_targets.append(
                    f"针对 member_id={member_id}, role={role}: 降低餐饮/酒店/交通成本，restaurant_price_level 优先使用 low 或 medium。"
                )
            elif "must-have" in problem_text or "must_include" in problem_text:
                concrete_revision_targets.append(
                    f"针对 member_id={member_id}, role={role}: 必须把该成员 must_include 的关键词原样写入 itinerary_text、per_day_itinerary 或 poi_tags。"
                )
            elif "hard" in problem_text or "硬约束" in problem_text:
                concrete_revision_targets.append(
                    f"针对 member_id={member_id}, role={role}: 优先修复该成员硬约束，不要只提高平均满意度。"
                )
            else:
                concrete_revision_targets.append(
                    f"针对 member_id={member_id}, role={role}: 根据该成员不满意原因调整方案。问题：{problem}"
                )
    concrete_revision_targets = list(dict.fromkeys(concrete_revision_targets))
    no_fallback_instruction = "\nNo fallback conservative planner is used.\nThe system will keep the highest-scoring candidate found during revision.\nTherefore, the next candidate should directly reduce the current failed reasons and improve the quality score, rather than generating a generic conservative plan.\n".strip()
    last_plan_summary_text = compact_json_for_prompt(last_plan_summary, max_chars=900)
    hard_constraint_feedback_text = compact_json_for_prompt(
        hard_constraint_feedback, max_chars=1000
    )
    plan_level_issues_text = compact_json_for_prompt(plan_level_issues, max_chars=700)
    dissatisfied_members_text = compact_json_for_prompt(dissatisfied_members, max_chars=1000)
    fairness_acceptance_feedback_text = compact_json_for_prompt(
        fairness_acceptance_feedback, max_chars=700
    )
    unresolved_conflict_feedback_text = compact_json_for_prompt(
        unresolved_conflict_feedback, max_chars=1200
    )
    concrete_revision_targets_text = compact_json_for_prompt(
        concrete_revision_targets, max_chars=1100
    )
    failure_reasons_text = compact_json_for_prompt(failure_reasons, max_chars=500)
    feedback_prompt = f"""
Additional Module 6 Feedback for Current Revision Only:
Output token limit for the next revision prompt: keep this feedback under 1800 tokens when used.

Important:
This prompt only describes the problems of the immediately previous plan.
Do not consider earlier revision rounds.
Do not accumulate old feedback.
Fix the current plan based only on the current evaluation below.

Automatic evaluation warning:
The revised plan will be rejected unless hard_constraint_pass_rate equals 1.0.
Therefore, hard constraints must be fixed before improving satisfaction, diversity, richness, or style.

Revision metadata:
- family_id: {family_id}
- revision_round: {revision_round}
- candidate_index: {candidate_index}

Highest priority exact hard-constraint fixes:
{critical_hard_constraint_text}

Previous plan summary:
{last_plan_summary_text}

Highest-priority hard constraint failures:
{hard_constraint_feedback_text}

Current failed reasons:
{failure_reasons_text}

Current unresolved conflict feedback:
{unresolved_conflict_feedback_text}

Current plan-level problems:
{plan_level_issues_text}

Dissatisfied members and their specific problems:
{dissatisfied_members_text}

Current fairness and acceptance problems:
{fairness_acceptance_feedback_text}

Concrete revision targets:
{concrete_revision_targets_text}

Selection strategy note:
{no_fallback_instruction}

Revision rules:
1. Only fix the problems from the immediately previous plan.
2. Do not include historical feedback from earlier rounds.

Hard constraint priority:
3. Hard constraints have absolute priority over satisfaction, diversity, richness, and nice-to-have preferences.
4. First fix all hard constraint failures before improving average satisfaction.
5. If any member's hard_constraint_pass is false, the revised plan must directly fix that member first.
6. If must_include is not satisfied, copy the exact must_include keyword into itinerary_text, per_day_itinerary, or poi_tags.
7. If must_avoid is violated, remove that exact keyword from itinerary_text, per_day_itinerary, poi_tags, meal_tags, hotel_plan, and transportation_plan.
8. If walking distance fails, reduce walking_distance_km so that walking_distance_km / day is below the strictest member limit.
9. If return time fails, set return_time no later than the earliest latest_return_time among all members.
10. If budget fails, reduce total_cost below budget_limit.
11. If low_intensity, child, teenager with low tolerance, or older members are present, avoid high_intensity_tags and keep daily_poi_count conservative.
12. Before outputting JSON, internally verify that every hard_constraint_failed_reason listed above has been fixed.
13. If Current unresolved conflict feedback contains unresolved_conflicts, do not drop them silently. A conflict can be removed only if the revised plan includes a complete role-explicit split_activities item or a concrete shared-itinerary resolution.
14. If a direct must_include vs must_avoid conflict cannot be resolved, keep it in unresolved_conflicts and do not mark consensus_status as Full Consensus.
15. Split activities must be used only for real hard conflicts; ordinary preference differences should be solved by shared low-cost/low-intensity alternatives.

Fairness and satisfaction:
16. After hard constraints are satisfied, protect the lowest-satisfaction member.
17. Do not only improve group average satisfaction.
18. If obvious unfairness exists, prioritize the lowest-satisfaction member over the average score.
19. If acceptance_vote_ratio is low, identify who rejected the plan and adjust their specific pain points.
20. If a specific member is dissatisfied, explain in coordinator_summary how the new plan fixes that member's issue.

Real-world grounding and objective evaluation:
21. The revised plan must not contain transport/POI placeholders such as short route requirement, 短路线需求, or core attraction. For restaurants, do not invent fake names; area-level fallback is allowed only when marked as no specific restaurant.
22. Hotel selection must explain why the hotel is the nearest qualified option to the itinerary route anchors, not just why it is in the city. Include hotel name, area, nearby station, route-fit reason, and price.
23. Transportation must be post-generation route stitching: for each day connect hotel -> morning POI -> lunch -> afternoon POI -> dinner -> hotel with origin, stops, destination, mode, time, and cost.
24. Restaurants should be concrete grounded names whenever possible. If a specific name is unavailable, leave restaurant_name empty and give only area + cuisine + price + source limitation; this will be softly penalized instead of failing the whole plan.

Generation style:
25. Do not make the itinerary richer if richness increases walking distance, late return, cost, or unfairness.
26. If the previous plan is already mostly good, make targeted edits instead of rewriting everything.
27. Keep the same JSON output structure required by Module 5.
28. Output valid JSON only.
29. Do not output markdown.
30. Output token limit: keep the next revision JSON compact and under the model max_tokens.
""".strip()
    return compact_feedback_for_llm(feedback_prompt, max_chars=2400)


def validate_input_columns(persona_df: pd.DataFrame, plans_df: pd.DataFrame):
    required_persona_columns = [
        "family_id",
        "member_id",
        "role",
        "interest_preferences",
        "physical_constraint",
        "budget_sensitivity",
        "dietary_preference",
        "travel_pace",
        "must_include",
        "must_avoid",
        "max_walking_distance_km",
        "latest_return_time",
    ]
    required_plan_columns = [
        "family_id",
        "plan_id",
        "method",
        "itinerary_text",
        "per_day_itinerary_json",
        "total_cost",
        "budget_limit",
        "walking_distance_km",
        "daily_poi_count",
        "poi_tags",
        "meal_tags",
        "restaurant_price_level",
        "route_distance_km",
        "return_time",
        "duplicate_poi_count",
        "high_intensity_tags",
        "hotel_plan_json",
        "transportation_plan_json",
        "negotiation_rounds",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ]
    for col in required_persona_columns:
        if col not in persona_df.columns:
            raise ValueError(f"persona 表缺少字段: {col}")
    for col in required_plan_columns:
        if col not in plans_df.columns:
            raise ValueError(f"generated_plans 表缺少字段: {col}")


def evaluate_and_explain(
    persona_csv_path: Path = PERSONA_CSV,
    agents_json_path: Path = AGENTS_JSON,
    generated_plans_csv_path: Path = GENERATED_PLANS_CSV,
    member_output_csv_path: Path = MEMBER_OUTPUT_CSV,
    plan_output_csv_path: Path = PLAN_OUTPUT_CSV,
    report_output_json_path: Path = REPORT_OUTPUT_JSON,
    use_llm_judge: bool = USE_LLM_JUDGE,
) -> Dict[str, Any]:
    if not persona_csv_path.exists():
        raise FileNotFoundError(f"找不到 persona 表，请先生成 persona: {persona_csv_path}")
    if not generated_plans_csv_path.exists():
        raise FileNotFoundError(f"找不到 generated_plans.csv，请先运行模块 5: {generated_plans_csv_path}")
    print("[Module 6] Evaluation and Explanation 开始")
    print(f"[Module 6] persona_csv_path: {persona_csv_path}")
    print(f"[Module 6] agents_json_path: {agents_json_path}")
    print(f"[Module 6] generated_plans_csv_path: {generated_plans_csv_path}")
    print(f"[Module 6] use_llm_judge: {use_llm_judge}")
    persona_df = pd.read_csv(persona_csv_path)
    plans_df = pd.read_csv(generated_plans_csv_path)
    validate_input_columns(persona_df=persona_df, plans_df=plans_df)
    persona_df["family_id"] = persona_df["family_id"].astype(int)
    plans_df["family_id"] = plans_df["family_id"].astype(int)
    valid_family_ids = get_valid_family_ids(persona_df=persona_df, plans_df=plans_df)
    persona_df = persona_df[persona_df["family_id"].isin(valid_family_ids)].copy()
    plans_df = plans_df[plans_df["family_id"].isin(valid_family_ids)].copy()
    if "status" in plans_df.columns:
        success_df = plans_df[plans_df["status"].astype(str).str.lower() == "success"].copy()
        if not success_df.empty:
            plans_df = success_df
    plans_df = (
        plans_df.sort_values(by=["family_id"])
        .drop_duplicates(subset=["family_id"], keep="first")
        .copy()
    )
    agent_weight_lookup = build_agent_weight_lookup(agents_json_path)
    member_rows = []
    plan_rows = []
    for _, plan_row in plans_df.iterrows():
        family_id = safe_int(plan_row.get("family_id"), 0)
        family_personas = persona_df[persona_df["family_id"] == family_id].copy()
        family_member_eval_rows = []
        for _, persona_row in family_personas.iterrows():
            member_eval = compute_member_satisfaction(
                persona_row=persona_row, plan_row=plan_row, agent_weight_lookup=agent_weight_lookup
            )
            output_member_row = {
                "family_id": family_id,
                "plan_id": plan_row.get("plan_id", "consensus_plan"),
                "member_id": persona_row.get("member_id"),
                "agent_id": member_eval.get("agent_id"),
                "role": persona_row.get("role"),
                "conflict_level": persona_row.get("conflict_level", ""),
                "family_structure": persona_row.get("family_structure", ""),
                "departure_city": persona_row.get("departure_city", ""),
                "destination_city": persona_row.get("destination_city", ""),
                "day": persona_row.get("day", ""),
                **member_eval,
            }
            family_member_eval_rows.append(output_member_row)
        if use_llm_judge:
            llm_judge_result = run_llm_judge_for_family(
                family_id=family_id,
                plan_row=plan_row,
                family_personas=family_personas,
                member_rule_evaluations=family_member_eval_rows,
            )
        else:
            llm_judge_result = {
                "family_id": family_id,
                "member_llm_judgements": [],
                "family_level_llm_comment": "LLM Judge skipped for fast rule-based evaluation.",
            }
        llm_judge_lookup = build_llm_judge_lookup(llm_judge_result)
        for member_eval in family_member_eval_rows:
            agent_id = safe_text(member_eval.get("agent_id", ""))
            rule_score = safe_float(
                member_eval.get("individual_rule_score"),
                member_eval.get("individual_satisfaction_score", 0),
            )
            llm_item = llm_judge_lookup.get(agent_id, {})
            llm_score = safe_float(llm_item.get("llm_judge_score"), rule_score)
            final_score = compute_weighted_satisfaction_score(
                rule_score=rule_score,
                llm_score=llm_score,
                hard_constraint_pass=bool(member_eval.get("hard_constraint_pass", True)),
            )
            bounded_llm_score = clamp_llm_score_near_rule_score(
                rule_score=rule_score,
                llm_score=llm_score,
                hard_constraint_pass=bool(member_eval.get("hard_constraint_pass", True)),
            )
            member_eval["llm_judge_score"] = round(llm_score, 4)
            member_eval["bounded_llm_judge_score"] = round(bounded_llm_score, 4)
            member_eval["llm_rule_score_delta"] = round(llm_score - rule_score, 4)
            member_eval["bounded_llm_rule_score_delta"] = round(bounded_llm_score - rule_score, 4)
            member_eval["llm_matched_evidence_json"] = json.dumps(
                llm_item.get("matched_evidence", []), ensure_ascii=False
            )
            member_eval["llm_missing_or_weak_evidence_json"] = json.dumps(
                llm_item.get("missing_or_weak_evidence", []), ensure_ascii=False
            )
            member_eval["llm_deduction_reasons_json"] = json.dumps(
                llm_item.get("deduction_reasons", []), ensure_ascii=False
            )
            member_eval["llm_deductions_json"] = json.dumps(
                llm_item.get("deductions", {}), ensure_ascii=False
            )
            member_eval["llm_fairness_evidence"] = safe_text(llm_item.get("fairness_evidence", ""))
            member_eval["llm_personalization_score"] = safe_float(
                llm_item.get("llm_personalization_score"), llm_score
            )
            member_eval["llm_reasonableness_score"] = safe_float(
                llm_item.get("llm_reasonableness_score"), llm_score
            )
            member_eval["llm_fairness_score"] = safe_float(
                llm_item.get("llm_fairness_score"), llm_score
            )
            member_eval["llm_judge_explanation"] = safe_text(
                llm_item.get(
                    "llm_judge_explanation", "LLM Judge unavailable; fallback to rule score."
                )
            )
            member_eval["individual_satisfaction_score"] = final_score
            member_eval["accepted_vote"] = final_score >= ACCEPTANCE_SCORE_THRESHOLD
            member_eval[
                "member_explanation"
            ] += f"；规则分={rule_score:.1f}；LLM Judge原始分={llm_score:.1f}；LLM Judge限制后分={bounded_llm_score:.1f}；最终加权分={final_score:.1f}"
            member_rows.append(member_eval)
        family_member_eval_df = pd.DataFrame(family_member_eval_rows)
        plan_eval = compute_plan_level_metrics(
            member_eval_df=family_member_eval_df, plan_row=plan_row
        )
        plan_rows.append(plan_eval)
    member_eval_df = pd.DataFrame(member_rows)
    plan_eval_df = pd.DataFrame(plan_rows)
    member_output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    plan_output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_output_json_path.parent.mkdir(parents=True, exist_ok=True)
    member_eval_df.to_csv(member_output_csv_path, index=False, encoding="utf-8-sig")
    plan_eval_df.to_csv(plan_output_csv_path, index=False, encoding="utf-8-sig")
    report = build_overall_report(plan_eval_df=plan_eval_df, member_eval_df=member_eval_df)
    # Eval-honesty: report BOTH all-task results (every family, failures included) and
    # evaluated-only results (families whose plan passes is_plan_passed_requirements).
    # The gap between the two exposes how optimistic the success-filtered numbers are.
    if not plan_eval_df.empty:
        passed_mask = plan_eval_df.apply(is_plan_passed_requirements, axis=1)
        evaluated_df = plan_eval_df[passed_mask].copy()
        total_task_count = int(len(plan_eval_df))
        evaluated_pass_count = int(passed_mask.sum())
        report["result_partition"] = {
            "total_task_count": total_task_count,
            "evaluated_pass_count": evaluated_pass_count,
            "failed_count": total_task_count - evaluated_pass_count,
            "overall_pass_rate": round(evaluated_pass_count / total_task_count, 4)
            if total_task_count
            else 0.0,
        }
        report["metrics_all_task"] = report.get("metrics", {})
        report["metrics_evaluated_only"] = (
            build_overall_report(plan_eval_df=evaluated_df, member_eval_df=member_eval_df).get(
                "metrics", {}
            )
            if not evaluated_df.empty
            else {}
        )
    save_json_file(data=report, path=report_output_json_path)
    print("[Module 6] Evaluation and Explanation 完成")
    print(f"[Module 6] 成员级评价结果: {member_output_csv_path}")
    print(f"[Module 6] 计划级评价结果: {plan_output_csv_path}")
    print(f"[Module 6] 总体解释报告: {report_output_json_path}")
    print("[Module 6] 计划级评价前 5 行预览：")
    preview_cols = [
        "family_id",
        "plan_id",
        "hard_constraint_pass_rate",
        "budget_pass",
        "feasibility_pass_rate",
        "rationality_pass_rate",
        "average_group_satisfaction",
        "minimum_member_satisfaction",
        "satisfaction_variance",
        "max_min_fairness",
        "group_consensus_score",
        "objective_feasibility_micro",
        "objective_rationality_micro",
        "objective_final_pass",
        "obvious_unfairness",
        "acceptance_vote_ratio",
        "negotiation_rounds",
        "total_tokens",
    ]
    available_cols = [col for col in preview_cols if col in plan_eval_df.columns]
    if available_cols:
        print(plan_eval_df[available_cols].head(5).to_string(index=False))
    print("[Module 6] 总体总结：")
    print(report["overall_summary"])
    return {"member_eval_df": member_eval_df, "plan_eval_df": plan_eval_df, "report": report}
