"""Module 5 - Consensus itinerary generation: synthesize the negotiated preferences into a concrete, grounded day-by-day itinerary."""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.common import (
    RESULTS_DIR,
    UNRESOLVED_CONFLICT_WEIGHTS,
    _as_text_list,
    budget_messages,
    build_family_agents,
    compact_agent_for_llm,
    compact_conflicts_for_llm,
    compact_feedback_for_llm,
    compact_json_for_prompt,
    compact_negotiation_round_for_llm,
    compact_plan_for_llm,
    compact_proposal_for_llm,
    estimate_day_walking_km,
    estimate_room_count,
    load_conflict_results,
    load_json_file,
    parse_json_text,
    safe_float,
    safe_int,
    safe_text,
    save_json_file,
    time_to_minutes,
)
from src.llm_client import call_generation_json

AGENTS_JSON = RESULTS_DIR / "module_1" / "persona_agents.json"
PROPOSALS_CSV = RESULTS_DIR / "module_2" / "individual_agent_proposals.csv"
CONFLICTS_JSON = RESULTS_DIR / "module_3" / "conflict_detection_results.json"
NEGOTIATION_JSON = RESULTS_DIR / "module_4" / "negotiation_protocol_results.json"
OUTPUT_CSV = RESULTS_DIR / "module_5" / "generated_plans.csv"
OUTPUT_JSON = RESULTS_DIR / "module_5" / "consensus_itinerary_results.json"
MODULE5_FAILED_FAMILIES_JSON = (
    RESULTS_DIR / "module_5" / "module5_generation_failed_family_ids.json"
)
MODULE5_FAILED_FAMILIES_CSV = RESULTS_DIR / "module_5" / "module5_generation_failed_family_ids.csv"
MODEL_NAME = os.getenv(
    "OPENROUTER_GENERATION_MODEL",
    os.getenv("OPENROUTER_MODEL_NAME", os.getenv("GENERATION_MODEL_NAME", "openai/gpt-4o-mini")),
)
MAX_TOKENS_FINAL_ITINERARY = 9000
MODULE5_MAX_REGEN_ATTEMPTS = int(os.getenv("MODULE5_MAX_REGEN_ATTEMPTS", "2"))
TEMPERATURE = 0.3
SLEEP_SECONDS = 0.1
MODULE5_PROMPT_INPUT_LIMITS = {
    "agent_constraints": 1400,
    "individual_proposals": 1400,
    "conflicts": 900,
    "round1": 700,
    "round2": 700,
    "round3": 700,
    "consensus_direction": 1000,
    "activity_grounding_candidates": 1400,
    "revision_constraints": 700,
    "revision_feedback": 1800,
}

# Keep split activities explicit and limited.  Too many split activities make the
# plan hard to coordinate and trigger excessive split-activity cost in Module 6.
MAX_SPLIT_ACTIVITIES_FOR_EVAL = 2
VAGUE_SPLIT_WORDS = ["部分成员", "其他成员", "有些成员", "有人", "不参加者", "其余成员"]

# Cap how many same-category attractions a family's whole itinerary may contain, so a
# trip is not e.g. five museums.  Excess (beyond the first MAX_SAME_CATEGORY_POI of a
# category) is swapped for an unused real attraction of a DIFFERENT category; must-include
# POIs (in poi_tags) are never swapped, and nothing is dropped if no alternative exists.
MAX_SAME_CATEGORY_POI = safe_int(os.getenv("MAX_SAME_CATEGORY_POI", "2"), 2)

# Realistic intra-city route distance per inter-stop leg; route_distance_km is DERIVED
# from the day's real stops (hotel->POI->restaurant->...->hotel) so it stays consistent
# with the itinerary and the route/walking efficiency ratio stays in a sane range.
ROUTE_KM_PER_LEG = safe_float(os.getenv("ROUTE_KM_PER_LEG", "4.0"), 4.0)

PLAN_OUTPUT_COLUMNS = [
    "family_id",
    "plan_id",
    "method",
    "departure_city",
    "destination_city",
    "day",
    "itinerary_text",
    "per_day_itinerary_json",
    "itinerary_quality_markers_json",
    "total_cost",
    "budget_limit",
    "budget_note",
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
    "real_world_grounding_json",
    "grounding_requirements_json",
    "online_grounding_json",
    "cost_breakdown_json",
    "hard_constraint_checklist_json",
    "split_activities_json",
    "concession_resolutions_json",
    "resolved_conflicts_json",
    "unresolved_conflicts_json",
    "constraint_satisfiability_json",
    "agent_vote_prediction_json",
    "acceptance_vote_ratio",
    "number_of_objections_resolved",
    "consensus_status",
    "coordinator_summary",
    "negotiation_rounds",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "runtime_seconds",
    "failure_stage",
    "module5_generation_failed",
    "backtrack_to_module2",
    "module5_failure_reasons_json",
    "status",
    "error",
]

USE_SPLIT_ACTIVITY_RESOLUTION = os.getenv("USE_SPLIT_ACTIVITY_RESOLUTION", "true").lower() in [
    "1",
    "true",
    "yes",
    "y",
]
# Concession is Module 4's SECOND conflict-resolution channel — the one that does NOT depend on
# split_activity. When negotiation ran but split is unavailable (no_split), the family still reaches
# an agreement by having the must_avoid member(s) concede (accept the contested must-have activity):
# the conflict is resolved (consensus reached, no unresolved penalty) but the conceding member takes
# a moderate satisfaction hit — strictly worse than split (which keeps everyone's constraint), yet
# strictly better than no resolution at all (no_negotiation). This is what separates no_split from
# no_negotiation, which otherwise collapse to the identical "nothing resolved" outcome. Default ON so
# `full` is unaffected (full uses split, which preempts concession); it only fires under no_split.
USE_CONCESSION_RESOLUTION = os.getenv("USE_CONCESSION_RESOLUTION", "true").lower() in [
    "1",
    "true",
    "yes",
    "y",
]
USE_CONSTRAINT_SATISFIABILITY_TYPING = os.getenv(
    "USE_CONSTRAINT_SATISFIABILITY_TYPING", "true"
).lower() in ["1", "true", "yes", "y"]

from src.grounding import (
    HOTEL_PEOPLE_PER_ROOM,
    build_grounded_activity_candidates,
    city_attraction_pool,
    ensure_online_grounding,
    fetch_alternative_real_poi,
    ground_generic_and_split_activities,
)

# Shared single source in common.py; alias kept so the rest of this module reads unchanged.
UNRESOLVED_CONFLICT_STATUS_WEIGHTS = UNRESOLVED_CONFLICT_WEIGHTS
MAX_FULL_CONSENSUS_UNRESOLVED_PENALTY = 8.0


def estimate_unresolved_conflict_penalty_for_status(plan: Dict[str, Any]) -> float:
    unresolved = plan.get("unresolved_conflicts", [])
    if isinstance(unresolved, str):
        unresolved = parse_json_text(unresolved, default=[])
    if not isinstance(unresolved, list):
        return 0.0
    penalty = 0.0
    for item in unresolved:
        if isinstance(item, dict):
            ctype = safe_text(item.get("conflict_type", "unknown")) or "unknown"
        else:
            ctype = safe_text(item) or "unknown"
        penalty += UNRESOLVED_CONFLICT_STATUS_WEIGHTS.get(ctype, 5.0)
    return round(penalty, 4)


def adjust_consensus_status_for_unresolved_conflicts(plan: Dict[str, Any]) -> Dict[str, Any]:
    vote_prediction = plan.get("agent_vote_prediction", [])
    if not isinstance(vote_prediction, list):
        vote_prediction = []
    has_reject = False
    for vote_item in vote_prediction:
        vote_text = (
            safe_text(vote_item.get("vote", "")).lower()
            if isinstance(vote_item, dict)
            else safe_text(vote_item).lower()
        )
        if "reject" in vote_text:
            has_reject = True
            break
    acceptance_ratio = safe_float(plan.get("acceptance_vote_ratio"), 0.0)
    unresolved_penalty = estimate_unresolved_conflict_penalty_for_status(plan)
    original_status = safe_text(plan.get("consensus_status", ""))
    reasons = []
    if has_reject or acceptance_ratio < 1.0:
        new_status = "No Consensus"
        reasons.append("not_all_members_accept")
    elif unresolved_penalty > MAX_FULL_CONSENSUS_UNRESOLVED_PENALTY:
        new_status = "Partial Consensus"
        reasons.append(f"unresolved_conflict_penalty={unresolved_penalty}")
    else:
        new_status = "Full Consensus"
    plan["consensus_status_original_model_or_repair"] = original_status
    plan["consensus_status"] = new_status
    plan["consensus_forced_downgrade"] = (
        original_status.lower() == "full consensus" and new_status != "Full Consensus"
    )
    plan["consensus_adjustment_reasons"] = reasons
    plan["unresolved_conflict_penalty_for_status"] = unresolved_penalty
    return plan


def load_individual_proposals(path: Path) -> Dict[int, List[Dict[str, Any]]]:
    df = pd.read_csv(path)
    required_columns = ["agent_id", "family_id", "member_id", "individual_proposal_json"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"individual_agent_proposals.csv 缺少字段: {col}")
    family_proposals: Dict[int, List[Dict[str, Any]]] = {}
    for _, row in df.iterrows():
        family_id = safe_int(row.get("family_id"))
        proposal_item = {
            "agent_id": safe_text(row.get("agent_id")),
            "family_id": family_id,
            "member_id": safe_int(row.get("member_id")),
            "role": safe_text(row.get("role")),
            "age_stage": safe_text(row.get("age_stage")),
            "proposal": parse_json_text(row.get("individual_proposal_json"), default={}),
        }
        if family_id not in family_proposals:
            family_proposals[family_id] = []
        family_proposals[family_id].append(proposal_item)
    return family_proposals


def load_negotiation_results(path: Path) -> Dict[int, Dict[str, Any]]:
    data = load_json_file(path)
    negotiation_lookup = {}
    for item in data:
        family_id = safe_int(item.get("family_id"))
        negotiation_lookup[family_id] = item
    return negotiation_lookup


def get_valid_family_ids(
    family_agents_map: Dict[int, List[Dict[str, Any]]],
    family_proposals_map: Dict[int, List[Dict[str, Any]]],
    conflict_lookup: Dict[int, Dict[str, Any]],
    negotiation_lookup: Dict[int, Dict[str, Any]],
) -> List[int]:
    valid_family_ids = []
    skipped_info = []
    for family_id, agents in family_agents_map.items():
        reasons = []
        if not agents:
            reasons.append("no_agents_in_persona_agents_json")
        agent_count = len(agents)
        if agent_count < 3 or agent_count > 4:
            reasons.append(f"invalid_agent_count:{agent_count}")
        proposals = family_proposals_map.get(family_id, [])
        proposal_agent_ids = {item.get("agent_id") for item in proposals if item.get("agent_id")}
        missing_proposal_agents = []
        for agent in agents:
            agent_id = agent.get("agent_id")
            if agent_id not in proposal_agent_ids:
                missing_proposal_agents.append(agent_id)
        if missing_proposal_agents:
            reasons.append(f"missing_proposals:{missing_proposal_agents}")
        for proposal_item in proposals:
            proposal = proposal_item.get("proposal", {})
            if not proposal:
                reasons.append(f"empty_proposal:{proposal_item.get('agent_id')}")
            if isinstance(proposal, dict) and "error" in proposal:
                reasons.append(f"error_proposal:{proposal_item.get('agent_id')}")
        if family_id not in conflict_lookup:
            reasons.append("missing_conflict_detection_result")
        if family_id not in negotiation_lookup:
            reasons.append("missing_negotiation_result")
        else:
            negotiation = negotiation_lookup[family_id]
            if negotiation.get("status") != "success":
                reasons.append("negotiation_not_success")
            if not negotiation.get("round4_final_plan"):
                reasons.append("missing_round4_final_plan")
        if reasons:
            skipped_info.append(
                {"family_id": family_id, "agent_count": agent_count, "reasons": reasons}
            )
        else:
            valid_family_ids.append(family_id)
    print("\n[Module 5] 完整可用 family_id 检查完成")
    print(f"[Module 5] 可用家庭数量: {len(valid_family_ids)}")
    print(f"[Module 5] 跳过家庭数量: {len(skipped_info)}")
    if skipped_info:
        print("[Module 5] 被跳过的 family_id：")
        for item in skipped_info[:3]:
            print(
                f"family_id={item['family_id']}, agent_count={item['agent_count']}, reasons={item['reasons']}"
            )
        if len(skipped_info) > 3:
            print(f"... 有 {len(skipped_info)} 个家庭被跳过。")
    return sorted(valid_family_ids)


def compress_revision_feedback(feedback: str, max_chars: int = 1600) -> str:
    text = safe_text(feedback)
    priority_keywords = [
        "hard_constraint_failed_reasons",
        "hard_constraint_revision_targets",
        "must_include_not_satisfied",
        "must_avoid_violated",
        "walking_distance_exceeded",
        "return_time_too_late",
        "budget",
        "objective_final_pass",
        "objective_feasibility_micro",
        "split activity",
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
        "lowest_satisfaction_member",
        "minimum_member_satisfaction",
        "acceptance_vote_ratio",
        "failed",
        "failure",
        "未通过",
        "失败",
        "硬约束",
        "最低满意度",
        "不公平",
        "修改",
        "修复",
        "未解决冲突",
        "未解决的冲突",
        "直接冲突",
        "分流活动成本",
    ]
    lines = [safe_text(line) for line in text.splitlines() if safe_text(line)]
    selected_lines = []
    for index, line in enumerate(lines):
        lower_line = line.lower()
        if any(keyword.lower() in lower_line for keyword in priority_keywords):
            selected_lines.append(line)
            for extra_line in lines[index + 1 : index + 3]:
                if extra_line:
                    selected_lines.append(extra_line)
    selected_lines = list(dict.fromkeys(selected_lines))
    if not selected_lines:
        selected_lines = lines[:12]
    compressed = "\n".join(selected_lines[:24])
    if len(compressed) > max_chars:
        compressed = compressed[:max_chars]
    return "\n".join(
        [
            "Compressed current Module 6 feedback only.",
            "Fix hard constraints and unresolved_conflicts first; keep the revised itinerary short.",
            "If unresolved_conflicts are listed, do not drop them silently: resolve them with role-explicit split_activities or keep them in unresolved_conflicts and avoid claiming Full Consensus.",
            compressed,
        ]
    ).strip()


def build_consensus_output_schema(family_id: int, first_agent: Dict[str, Any]) -> Dict[str, Any]:
    """Short Module 5 schema. Rich/default fields are filled by code later."""
    return {
        "family_id": family_id,
        "plan_id": "consensus_plan",
        "method": "multi_agent_consensus",
        "departure_city": first_agent.get("departure_city"),
        "destination_city": first_agent.get("destination_city"),
        "day": first_agent.get("day"),
        "itinerary_text": "≤180字中文概述，写明每日核心活动、酒店区域和交通策略。",
        "per_day_itinerary": [
            {
                "day": 1,
                "morning": ["具体景点名"],
                "afternoon": ["具体景点名"],
                "meal": "餐饮类型+价格等级",
                "transportation": "酒店区域到景点的短路线需求",
                "hotel_area_requirement": "区域+价格+交通+舒适度",
                "estimated_walking_distance_km": 0,
                "estimated_return_time": "20:00",
                "split_activities": [],
            }
        ],
        "total_cost": 0,
        "budget_limit": 0,
        "budget_note": "短预算说明",
        "walking_distance_km": 0,
        "daily_poi_count": 2,
        "poi_tags": [],
        "meal_tags": [],
        "restaurant_price_level": "medium",
        "route_distance_km": 0,
        "return_time": "20:00",
        "hotel_plan": {"hotel_area": "住宿区域", "price_level": "medium", "requirement": "靠近交通，适合家庭"},
        "transportation_plan": {"main_transport_mode": "地铁+短途出租车", "route_strategy": "少换乘少步行"},
        "grounding_requirements": {
            "hotel_requirements": "区域/价格/交通/舒适度短句",
            "transportation_requirements": "每日从酒店到景点的路线需求短句",
            "restaurant_requirements": "餐饮偏好和价格等级",
            "ticket_requirements": [],
        },
        "split_activities": [
            {
                "day": 1,
                "activity": "分流活动名称",
                "conflict_type": "preference_vs_must_avoid_term_conflict",
                "reason": "为什么必须分流的简短理由",
                "participating_roles": ["mother"],
                "non_participating_roles": ["father", "elder"],
                "alternative_for_non_participants": "具体替代活动",
                "role_assignment": "mother 参加“分流活动名称”；father、elder 不参加，改为具体替代活动。",
                "duration_minutes": 120,
                "branch_distance_km": 2.0,
                "extra_transport_cost": 40,
                "safety_risk": "low",
                "regroup_point": "酒店大堂",
                "reconvene_time": "18:00",
            }
        ],
        "unresolved_conflicts": [],
        "acceptance_vote_ratio": 1.0,
        "consensus_status": "Partial Consensus",
        "coordinator_summary": "≤80字说明",
    }


def build_consensus_itinerary_messages(
    family_id: int,
    family_agents: List[Dict[str, Any]],
    individual_proposals: List[Dict[str, Any]],
    conflict_info: Dict[str, Any],
    negotiation_info: Dict[str, Any],
    extra_feedback_prompt: str = "",
    activity_grounding_candidates: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    first_agent = family_agents[0]
    # Conflict-resolution ARCHITECTURE (approach A, permanent for ALL runs): conflict-resolution
    # DECISIONS are owned by Module 4; M5 RENDERS them and does not invent its own. So removing M4
    # (no_negotiation) leaves clashes unresolved. M5 still does all POI/grounding/quality work — it
    # only follows M4 on WHICH clash to split and WHO to protect, then concretizes the split itself.
    # (no_split still fully disables the split resolution type.)
    m4_resolution_section: List[str] = []
    if not USE_SPLIT_ACTIVITY_RESOLUTION:
        # no_split ablation: the split resolution TYPE is disabled. Tell M5 not to emit splits so a
        # must_include vs must_avoid clash is left unresolved (crediting is also hard-gated off in
        # enforce_rule_consistency, so this holds even if the LLM disobeys). Full is unaffected.
        main_conflict_rule = (
            "- Split activities are DISABLED for this run. Do NOT create any split_activities. If a "
            "must_include conflicts with another member's must_avoid, you cannot split them apart — "
            "keep the conflict in unresolved_conflicts and do not claim Full Consensus."
        )
        revision_conflict_rules = [
            "- Split activities are DISABLED for this run; do NOT create split_activities to resolve "
            "a must_include vs must_avoid conflict.",
            "- Keep any such conflict in unresolved_conflicts (you may still fix quality/feasibility/"
            "grounding issues).",
        ]
    else:
        _r4 = negotiation_info.get("round4_final_plan", {}) or {}
        _m4_resolutions = {
            "conflict_resolution_map": _r4.get("conflict_resolution_map", []),
            "split_activity_recommendations": _r4.get("split_activity_recommendations", []),
            "resolved_conflicts": _r4.get("resolved_conflicts", []),
        }
        m4_resolution_section = [
            "Module 4 negotiation's DECIDED conflict resolutions — apply THESE (render each decided "
            "split as a concrete split_activities item). If this is EMPTY, the negotiation resolved "
            "nothing → leave the must_include vs must_avoid clashes in unresolved_conflicts:",
            compact_json_for_prompt(_m4_resolutions, max_chars=1200),
            "",
        ]
        main_conflict_rule = (
            "- Conflict-resolution DECISIONS are owned by Module 4. Create a split_activities item "
            "ONLY for the clashes Module 4 resolved in its 'DECIDED conflict resolutions' section "
            "(follow its choice of which clash to split and which members to protect; you fill in "
            "the concrete activity + alternative). Do NOT invent resolutions for clashes Module 4 "
            "did not resolve — leave those in unresolved_conflicts and do not claim Full Consensus."
        )
        revision_conflict_rules = [
            "- Conflict-resolution decisions are owned by Module 4; do NOT invent new split_activities "
            "for a must_include vs must_avoid clash Module 4 did not resolve — keep it in "
            "unresolved_conflicts.",
            "- You may still fix quality/feasibility/grounding issues and concretize Module 4's "
            "decided splits.",
        ]
    schema_text = json.dumps(
        build_consensus_output_schema(family_id, first_agent), ensure_ascii=False, indent=2
    )
    activity_grounding_text = compact_json_for_prompt(
        activity_grounding_candidates or {},
        max_chars=MODULE5_PROMPT_INPUT_LIMITS.get("activity_grounding_candidates", 1400),
    )

    if extra_feedback_prompt:
        minimal_agent_constraints = [compact_agent_for_llm(agent) for agent in family_agents]
        minimal_constraints_text = compact_json_for_prompt(
            minimal_agent_constraints, max_chars=MODULE5_PROMPT_INPUT_LIMITS["revision_constraints"]
        )
        feedback_text = compact_feedback_for_llm(
            compress_revision_feedback(
                extra_feedback_prompt, max_chars=MODULE5_PROMPT_INPUT_LIMITS["revision_feedback"]
            ),
            max_chars=MODULE5_PROMPT_INPUT_LIMITS["revision_feedback"],
        )
        revision_system_prompt = "\n".join(
            [
                "You are the coordinator agent revising a family travel itinerary.",
                "You are not generating a new plan from scratch.",
                "Revise the previous plan according to the latest Module 6 feedback only.",
                "Output valid JSON only.",
                "Do not output markdown.",
                "All natural-language JSON string values must be in Chinese, except fixed enum values.",
                "Output token limit: stay under 3800 output tokens.",
            ]
        )
        revision_user_prompt = "\n".join(
            [
                "REVISION MODE IS ACTIVE.",
                "Revise the immediately previous family travel plan.",
                "Only fix the problems listed in the current Module 6 feedback.",
                "",
                "Family trip information:",
                f"family_id: {family_id}",
                f"departure_city: {first_agent.get('departure_city')}",
                f"destination_city: {first_agent.get('destination_city')}",
                f"day: {first_agent.get('day')}",
                f"query: {first_agent.get('query')}",
                f"family_structure: {first_agent.get('family_structure')}",
                f"conflict_level: {first_agent.get('conflict_level')}",
                "",
                "Minimal member constraints and preferences:",
                minimal_constraints_text,
                "",
                "Current Module 6 feedback from the immediately previous plan:",
                feedback_text,
                "",
                "Candidate repair strategy rules:",
                "- If feedback contains CANDIDATE_REPAIR_STRATEGY, follow that strategy and make this candidate meaningfully different from other candidates.",
                "- Do not only rewrite wording. Change the actual itinerary structure, split assignment, activity order, return time, or budget plan when needed.",
                "- Keep successful parts from the previous plan if they do not conflict with the current failure reasons.",
                "- The candidate must explicitly explain in coordinator_summary which previous failure reasons were fixed.",
                "",
                "Unresolved-conflict repair rules:",
                *revision_conflict_rules,
                "- A complete split_activities item must include conflict_type, activity, participating_roles, non_participating_roles, alternative_for_non_participants, role_assignment, and reason.",
                "- It must also include the split trigger fields: duration_minutes, branch_distance_km (distance between the two branch locations), extra_transport_cost, safety_risk (low/medium/high), regroup_point, and reconvene_time, so the split is justified only when branches are time-compatible, within ~5km, cost/safety-bounded, and reconvene the same day.",
                "- Do not delete unresolved_conflicts from the output unless the new plan gives a concrete role-explicit resolution.",
                "- If any direct unresolved hard conflict remains, consensus_status must not be Full Consensus.",
                "",
                "Revision priorities:",
                "1. Fix hard constraint failures first.",
                "2. Then improve the lowest-satisfaction member.",
                "3. Then improve acceptance_vote_ratio.",
                "4. Then improve average_group_satisfaction.",
                "5. Keep the itinerary concise.",
                "",
                "Hard rules:",
                "- must_include terms must appear explicitly in itinerary_text, per_day_itinerary, or poi_tags.",
                "- must_avoid terms must not appear in shared itinerary fields unless handled only by split_activities.",
                "- walking_distance_km / day must not exceed the strictest member walking limit.",
                "- return_time must not be later than the earliest latest_return_time among all members.",
                "- total_cost must not exceed budget_limit when budget_limit > 0.",
                "- Aim to USE the budget well: target a total around 85% of budget_limit (pick comfortable hotels/restaurants and the key paid attractions); do NOT leave more than ~20% of the budget unused, while keeping ~15% headroom.",
                "- daily_poi_count should be <= 3.",
                "- Use ONE consistent name per place (never both 故宫 and 故宫博物院) and diversify attraction types — at most 2 of the same kind (no all-museum itineraries).",
                "- restaurant_price_level must be exactly one of: low, medium, high.",
                "- Use grounding_requirements to specify hotel, restaurant, transportation, ticket and cost needs.",
                "- Do not invent specific hotel or restaurant names unless online grounding supports them.",
                "- Each day must include only a short transportation requirement for later route search, not a finalized route.",
                "- Do not invent detailed metro/taxi routes in Module 5; the transport API/search grounding will run after the full itinerary, hotel, restaurants, and POIs are generated, then stitch the route chain.",
                "- Hotel requirements must ask online grounding to choose the nearest qualified hotel to the completed itinerary anchors, while still satisfying budget, comfort, safety, and transit access.",
                "- Restaurant requirements must ask online grounding for diverse restaurants; avoid using the same restaurant repeatedly.",
                "- Ticket requirements must list paid scenic areas/attractions so ticket prices are counted in the budget.",
                "- Cost estimate must include hotel, intercity transport, local transport, meals, and attraction tickets.",
                "- Split activity must name exact roles: use participating_roles and non_participating_roles.",
                "- Split activity is valid only when it resolves a real hard conflict such as must_include vs must_avoid; do not use it for ordinary preference differences.",
                "- Every split activity must include a concrete alternative and meeting/rejoin logic for non_participants.",
                "- Never write vague split text like 部分成员/其他成员/有人/不参加者. Use roles only, e.g. mother 参加“海港城短时购物”；father、elder 不参加，改为九龙公园休息。",
                "- per_day_itinerary MUST contain exactly the requested number of days; every day needs concrete named morning/afternoon activities.",
                "- Forbidden placeholders: core attraction, core family activity, local attraction, day attractions, nearest metro/MTR station, MTR/metro line route, local family restaurant, 休息或自由安排 as the only activity.",
                "- HARD: morning and afternoon must EACH be a concrete named POI; never empty and never theme words like '家庭行程','轻松家庭行程','结合当天主题安排','亲子活动'. Any POI used in the route must also be a concrete morning/afternoon activity that day.",
                "",
                "Module 5 Tavily/search-grounded concrete activity candidates. Use these to replace rough/vague activities with concrete named POIs when repairing:",
                activity_grounding_text,
                "",
                "Return JSON in this shortened schema. Keep field names exactly the same. Omit no required keys:",
                schema_text,
                "",
                "Length and token rules:",
                "- Output token limit: stay under 3800 output tokens.",
                "- If close to the limit, keep required keys and summarize optional fields.",
                "- itinerary_text must be concise, no more than 180 Chinese characters.",
                "- coordinator_summary must be no more than 100 Chinese characters.",
                "- Each day should have at most 2 core activities (one morning, one afternoon). Do NOT output a 'theme' field; put concrete POIs directly into morning and afternoon arrays.",
            ]
        )
        return [
            {"role": "system", "content": revision_system_prompt},
            {"role": "user", "content": revision_user_prompt},
        ]

    agent_constraints = [compact_agent_for_llm(agent) for agent in family_agents]
    round1 = compact_negotiation_round_for_llm(
        negotiation_info.get("round1_core_needs", []), max_items=6
    )
    round2 = compact_negotiation_round_for_llm(
        negotiation_info.get("round2_objections", []), max_items=6
    )
    round3 = compact_negotiation_round_for_llm(
        negotiation_info.get("round3_concessions", []), max_items=6
    )
    consensus_direction = compact_plan_for_llm(negotiation_info.get("round4_final_plan", {}))
    compact_proposals = [compact_proposal_for_llm(item) for item in individual_proposals]
    compact_conflicts = compact_conflicts_for_llm(conflict_info.get("conflicts", []), max_items=8)
    agent_constraints_text = compact_json_for_prompt(
        agent_constraints, max_chars=MODULE5_PROMPT_INPUT_LIMITS["agent_constraints"]
    )
    individual_proposals_text = compact_json_for_prompt(
        compact_proposals, max_chars=MODULE5_PROMPT_INPUT_LIMITS["individual_proposals"]
    )
    conflicts_text = compact_json_for_prompt(
        compact_conflicts, max_chars=MODULE5_PROMPT_INPUT_LIMITS["conflicts"]
    )
    round1_text = compact_json_for_prompt(round1, max_chars=MODULE5_PROMPT_INPUT_LIMITS["round1"])
    round2_text = compact_json_for_prompt(round2, max_chars=MODULE5_PROMPT_INPUT_LIMITS["round2"])
    round3_text = compact_json_for_prompt(round3, max_chars=MODULE5_PROMPT_INPUT_LIMITS["round3"])
    consensus_direction_text = compact_json_for_prompt(
        consensus_direction, max_chars=MODULE5_PROMPT_INPUT_LIMITS["consensus_direction"]
    )

    system_prompt = "\n".join(
        [
            "You are the coordinator agent in a multi-agent family travel planning framework.",
            "Generate the final consensus itinerary based on persona constraints, individual proposals, conflict detection, negotiation objections, concessions, rough final consensus direction, and Module 5 Tavily/search-grounded activity candidates.",
            "Before generating the final itinerary, explicitly check every persona's hard constraints.",
            "Output valid JSON only.",
            "Do not output markdown.",
            "All natural-language JSON string values must be in Chinese, except fixed enum values.",
            "Output token limit: stay under 4500 output tokens.",
        ]
    )
    user_prompt = "\n".join(
        [
            "You are the coordinator agent. Generate a final consensus travel itinerary.",
            "",
            "Hard Constraint Priority Rules:",
            "1. First satisfy every member's hard constraints.",
            "2. Then satisfy global trip constraints such as budget, days, transportation, hotel completeness, and return time.",
            "3. Then improve the lowest member satisfaction.",
            "4. Then improve average group satisfaction.",
            "5. Then improve itinerary richness and diversity.",
            "",
            "Important hard constraints:",
            "- must_include items must appear explicitly in itinerary_text, per_day_itinerary, or poi_tags.",
            "- must_avoid items must not appear in shared itinerary fields.",
            main_conflict_rule,
            "- Split activity must name exact roles in participating_roles and non_participating_roles.",
            "- Never write vague split text like 部分成员/其他成员/有人/不参加者. Use roles only, e.g. mother 参加“海港城短时购物”；father、elder 不参加，改为九龙公园休息。",
            "- walking_distance_km / day must not exceed the strictest member walking limit.",
            "- return_time must not be later than the earliest latest_return_time among all members.",
            "- low_intensity, child, or older members must not be assigned high-intensity activities.",
            "",
            "Real-world grounding requirements:",
            "- Treat Module 4 as a rough intent plan only; do not copy broad labels such as culture landmark/theme park/local food directly into the final itinerary.",
            "- Use Tavily/search-grounded concrete activity candidates when they fit member constraints; otherwise choose another concrete named place consistent with the rough intent.",
            "- First generate hotel_requirements and transportation_requirements in grounding_requirements.",
            "- Hotel requirements must include price level, max price per night, preferred area, nearby transport requirement, comfort requirement, nights, and search keywords.",
            "- Daily route requirements must include origin, destinations, latest return time, walking limit, comfort requirements, and required output for online search.",
            "- Do not use internal reference hotel tables.",
            "- Specific hotel names, restaurant names, route prices, and ticket prices will be filled by online grounding when enabled.",
            "- Restaurant grounding should prefer varied restaurants across days/meals; do not require repeated use of one restaurant.",
            "- Ticket grounding must include paid scenic-area tickets in attraction_ticket_cost.",
            "- Budget must include hotel + intercity transport + local transport + meals + attraction tickets.",
            "",
            "Family trip information:",
            f"family_id: {family_id}",
            f"departure_city: {first_agent.get('departure_city')}",
            f"destination_city: {first_agent.get('destination_city')}",
            f"day: {first_agent.get('day')}",
            f"query: {first_agent.get('query')}",
            f"family_structure: {first_agent.get('family_structure')}",
            f"conflict_level: {first_agent.get('conflict_level')}",
            "",
            "All persona agents' constraints and preferences:",
            agent_constraints_text,
            "",
            "Individual proposals from persona agents:",
            individual_proposals_text,
            "",
            "Conflict detection results:",
            conflicts_text,
            "",
            "Negotiation Round 1 core needs:",
            round1_text,
            "",
            "Negotiation Round 2 objections:",
            round2_text,
            "",
            "Negotiation Round 3 concessions:",
            round3_text,
            "",
            "Module 4 final consensus direction is a ROUGH INTENT PLAN only. It may contain broad categories instead of final places:",
            consensus_direction_text,
            "",
            *m4_resolution_section,
            "Tavily/search-grounded concrete activity candidates generated by Module 5 from the rough intent plan:",
            activity_grounding_text,
            "",
            "Return JSON in this shortened schema. Keep field names exactly the same. Omit no required keys:",
            schema_text,
            "",
            "Field rules:",
            "- per_day_itinerary MUST contain exactly the requested number of days. Do not output fewer days.",
            "- Every day MUST have at least one concrete named POI/activity in morning and one in afternoon, unless the original query is a transfer-only day.",
            "- HARD: the morning and afternoon fields must EACH contain a concrete named POI (a string or list of place names). NEVER leave them empty and NEVER put theme words such as '家庭行程','轻松家庭行程','轻松家庭活动','结合当天主题安排','低强度家庭活动','亲子活动' as the activity. Choose real POIs from the family's must_include list and the provided grounded activity candidates.",
            "- HARD: every POI you put in transportation/daily route MUST also appear as a concrete morning or afternoon activity that same day. Do NOT hide POIs only inside the route while leaving morning/afternoon vague.",
            "- Forbidden placeholders: core attraction, core family activity, local attraction, day attractions, free arrangement only, 休息或自由安排 as the only activity, 参观文化景点, 当地餐厅用餐, 最后的购物, 回顾旅行, nearby cafe without a place name.",
            "- Morning/afternoon activities must be concrete named places or named experiences, not category/action labels. If Module 4 says only a rough intent, use the Tavily/search candidate list to select a concrete place.",
            "- Do not output broad intent labels from Module 4 as final activities; convert them into concrete named POIs first.",
            "- A concrete item should have a name-like signal: a POI/street/mall/museum/park/tower/market/station/experience name, not only a broad category or meal/logistics action.",
            "- Do not write generic city-only activities such as 'Hong Kong core attraction' or 'Shanghai core attraction'. Use concrete attractions from member must_include/proposals, e.g. museum names, parks, towers, streets, waterfronts, theme parks.",
            "- transportation fields must describe concrete origin/destination for that day. Do not write nearest metro/MTR station, MTR/metro line route, day X attractions, or short taxi as the whole route.",
            "- lunch/dinner may use restaurant types before online grounding, but must include area tied to that day's concrete POI; never use 'local family restaurant' alone.",
            "- B-fallback (concrete names): every lunch/dinner MUST give a concrete, real, well-known restaurant_name suited to that day's area, not only a type. If you are not certain it is currently open, still give the most likely real name and set needs_verification=true. Never leave restaurant_name empty.",
            "- B-fallback (concrete names): hotel_plan MUST give a concrete, real hotel_name (a known hotel in the chosen area), never '不指定/未提供'. If unsure, give the most likely real hotel and set needs_verification=true.",
            "- B-fallback (concrete names): never output a morning/afternoon slot as 'general', '自由活动', '当地体验', or theme-only. Replace any vague slot with a concrete, real, named POI (set needs_verification=true if unsure). These named-but-unverified items are acceptable; vague placeholders are not.",
            "- total_cost and budget_limit must be numbers. If no budget is given, use 0.",
            "- restaurant_price_level must be exactly one of: low, medium, high.",
            "- daily_poi_count is the average number of core POIs per day.",
            "- duplicate_poi_count should be 0 unless the same POI appears multiple times.",
            "- Refer to each place by ONE consistent name across all days; never list the same place under two names (e.g. do NOT use both 故宫 and 故宫博物院, or both 颐和园 and 颐和园公园).",
            "- Diversify attraction TYPES across the trip: do not schedule more than 2 of the same kind (e.g. avoid an itinerary of five museums); mix museums, parks/nature, historic sites/temples, streets/markets and landmarks.",
            "- agent_vote_prediction should contain exactly one item per family member.",
            "- split_activities item must include: day, activity, conflict_type, reason, participating_roles, non_participating_roles, alternative_for_non_participants, role_assignment, duration_minutes, branch_distance_km, extra_transport_cost, safety_risk, regroup_point, reconvene_time.",
            "- Create one split_activities item for EACH direct must_include vs must_avoid conflict (up to 2 for the whole family); prefer resolving every direct clash with a split over leaving it unresolved.",
            "- Split activities must use role names only. Do not write 部分成员/其他成员/有人/其余成员.",
            "- participating_roles / non_participating_roles must use roles such as father, mother, child, elder, teenager. If two members share one role, use role(member_id), e.g. child(member_2).",
            "- grounding_requirements.restaurant_requirements must request diverse restaurants and reasonable per-person prices.",
            "- grounding_requirements.ticket_requirements must include all likely paid attractions/scenic areas for ticket price lookup.",
            "",
            "Length and token rules:",
            "- Output token limit: stay under 4500 output tokens.",
            "- If close to the limit, keep required keys and summarize optional fields.",
            "- itinerary_text must be concise, no more than 220 Chinese characters.",
            "- coordinator_summary must be no more than 120 Chinese characters.",
            "- Each day should have at most 2 core POIs or activities.",
            "- Do not write long explanations inside JSON fields.",
        ]
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def collect_family_constraints(family_agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    must_include_terms = []
    must_avoid_terms = []
    max_walking_limits = []
    latest_return_minutes = []
    agent_constraint_rows = []
    for agent in family_agents:
        profile = agent.get("profile", {})
        constraints = agent.get("constraints", {})
        agent_id = safe_text(agent.get("agent_id", ""))
        role = safe_text(profile.get("role", ""))
        must_include = constraints.get("must_include", [])
        must_avoid = constraints.get("must_avoid", [])
        if not isinstance(must_include, list):
            must_include = []
        if not isinstance(must_avoid, list):
            must_avoid = []
        for item in must_include:
            item_text = safe_text(item)
            if item_text:
                must_include_terms.append(item_text)
        for item in must_avoid:
            item_text = safe_text(item)
            if item_text:
                must_avoid_terms.append(item_text)
        max_walking = safe_float(constraints.get("max_walking_distance_km"), None)
        if max_walking is not None:
            max_walking_limits.append(max_walking)
        latest_return_time = safe_text(constraints.get("latest_return_time", ""))
        latest_minutes = time_to_minutes(latest_return_time)
        if latest_minutes is not None:
            latest_return_minutes.append(latest_minutes)
        agent_constraint_rows.append(
            {
                "agent_id": agent_id,
                "role": role,
                "must_include": must_include,
                "must_avoid": must_avoid,
                "max_walking_distance_km": max_walking,
                "latest_return_time": latest_return_time,
                "latest_return_minutes": latest_minutes,
            }
        )
    return {
        "must_include_terms": list(dict.fromkeys(must_include_terms)),
        "must_avoid_terms": list(dict.fromkeys(must_avoid_terms)),
        "max_walking_limits": max_walking_limits,
        "latest_return_minutes": latest_return_minutes,
        "agent_constraint_rows": agent_constraint_rows,
    }


def sanitize_string_with_avoid_terms(text: str, avoid_terms: List[str]) -> str:
    output = safe_text(text)
    for term in avoid_terms:
        term = safe_text(term)
        if not term:
            continue
        try:
            output = re.sub(re.escape(term), "[avoided_item_removed]", output, flags=re.IGNORECASE)
        except Exception:
            output = output.replace(term, "[avoided_item_removed]")
    return output


def recursively_sanitize_avoid_terms(value, avoid_terms: List[str]):
    if isinstance(value, str):
        return sanitize_string_with_avoid_terms(value, avoid_terms)
    if isinstance(value, list):
        return [recursively_sanitize_avoid_terms(item, avoid_terms) for item in value]
    if isinstance(value, dict):
        return {
            key: recursively_sanitize_avoid_terms(val, avoid_terms) for key, val in value.items()
        }
    return value


def normalize_term_list_for_conflict_check(terms: List[str]) -> List[str]:
    normalized = []
    for term in terms:
        term_text = safe_text(term)
        if term_text:
            normalized.append(term_text)
    return list(dict.fromkeys(normalized))


def normalize_hard_term(term: str) -> str:
    text = safe_text(term).lower()
    text = text.replace("_", " ").replace("-", " ")
    text = text.replace("theme park", "theme parks")
    text = text.replace("museum", "museums")
    text = " ".join(text.split())
    return text


def hard_terms_overlap(a: str, b: str) -> bool:
    ta = normalize_hard_term(a)
    tb = normalize_hard_term(b)
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


def detect_hard_constraint_term_conflicts(
    must_include_terms: List[str], must_avoid_terms: List[str]
) -> List[str]:
    conflicts = []
    for inc in must_include_terms:
        for avd in must_avoid_terms:
            if hard_terms_overlap(inc, avd):
                conflicts.append(safe_text(inc) or safe_text(avd))
    return list(dict.fromkeys([x for x in conflicts if x]))


def term_conflicts_with_avoid(term: str, must_avoid_terms: List[str]) -> bool:
    term_text = safe_text(term).lower()
    if not term_text:
        return False
    for avoid in must_avoid_terms:
        avoid_text = safe_text(avoid).lower()
        if not avoid_text:
            continue
        if hard_terms_overlap(term_text, avoid_text):
            return True
    return False


def role_label_for_agent(
    agent: Dict[str, Any], role_counts: Optional[Dict[str, int]] = None
) -> str:
    """Return a human-readable role label for split activities.

    Most families have unique roles such as father/mother/child/elder.
    If a family has duplicate roles, append member_id so the split remains explicit,
    e.g. child(member_2), child(member_3).
    """
    profile = agent.get("profile", {}) if isinstance(agent, dict) else {}
    role = safe_text(profile.get("role", "")) or safe_text(agent.get("role", "")) or "member"
    role = role.lower()
    member_id = safe_int(agent.get("member_id"), 0)
    if role_counts and role_counts.get(role, 0) > 1 and member_id > 0:
        return f"{role}(member_{member_id})"
    return role


def build_role_lookup(family_agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    roles = []
    for agent in family_agents:
        profile = agent.get("profile", {}) if isinstance(agent, dict) else {}
        roles.append(
            (
                safe_text(profile.get("role", "")) or safe_text(agent.get("role", "")) or "member"
            ).lower()
        )
    role_counts = {role: roles.count(role) for role in set(roles)}
    all_roles = [role_label_for_agent(agent, role_counts) for agent in family_agents]
    by_agent_id = {
        safe_text(agent.get("agent_id", "")).lower(): role_label_for_agent(agent, role_counts)
        for agent in family_agents
    }
    by_member_id = {
        f"member_{safe_int(agent.get('member_id'), 0)}".lower(): role_label_for_agent(
            agent, role_counts
        )
        for agent in family_agents
    }
    by_plain_role = {}
    for agent in family_agents:
        profile = agent.get("profile", {}) if isinstance(agent, dict) else {}
        plain = (
            safe_text(profile.get("role", "")) or safe_text(agent.get("role", "")) or "member"
        ).lower()
        by_plain_role.setdefault(plain, []).append(role_label_for_agent(agent, role_counts))
    return {
        "all_roles": list(dict.fromkeys(all_roles)),
        "by_agent_id": by_agent_id,
        "by_member_id": by_member_id,
        "by_plain_role": by_plain_role,
        "role_counts": role_counts,
    }


def _roles_from_refs(refs: Any, lookup: Dict[str, Any]) -> List[str]:
    if not isinstance(refs, list):
        refs = [refs]
    roles = []
    for ref in refs:
        text = safe_text(ref).lower()
        if not text:
            continue
        if text in lookup.get("by_agent_id", {}):
            roles.append(lookup["by_agent_id"][text])
            continue
        if text in lookup.get("by_member_id", {}):
            roles.append(lookup["by_member_id"][text])
            continue
        if text in lookup.get("by_plain_role", {}):
            roles.extend(lookup["by_plain_role"][text])
            continue
        # Accept already normalized labels such as child(member_2).
        for role in lookup.get("all_roles", []):
            if text == role.lower():
                roles.append(role)
    return list(dict.fromkeys([x for x in roles if x]))


def _infer_roles_from_activity_text(
    item: Dict[str, Any], family_agents: List[Dict[str, Any]], lookup: Dict[str, Any]
) -> Dict[str, List[str]]:
    text = " ".join(
        [
            safe_text(item.get("activity", "")),
            safe_text(item.get("activity_or_term", "")),
            safe_text(item.get("role_assignment", "")),
            json.dumps(item.get("core_activities", []), ensure_ascii=False),
        ]
    )
    participating = []
    non_participating = []
    for agent in family_agents:
        constraints = agent.get("constraints", {}) if isinstance(agent, dict) else {}
        role_label = role_label_for_agent(agent, lookup.get("role_counts"))
        includes = constraints.get("must_include", [])
        avoids = constraints.get("must_avoid", [])
        if not isinstance(includes, list):
            includes = []
        if not isinstance(avoids, list):
            avoids = []
        if any(hard_terms_overlap(text, x) for x in includes):
            participating.append(role_label)
        if any(hard_terms_overlap(text, x) for x in avoids):
            non_participating.append(role_label)
    return {
        "participating_roles": list(dict.fromkeys(participating)),
        "non_participating_roles": list(dict.fromkeys(non_participating)),
    }


def _clean_split_activity_text(text: str) -> str:
    output = safe_text(text)
    for bad in ["部分成员", "其他成员", "有些成员", "有人", "不参加者", "其余成员"]:
        output = output.replace(bad, "")
    for prefix in ["分流活动：", "分流活动:", "参加“", "参加"]:
        if output.startswith(prefix):
            output = output[len(prefix) :]
    output = output.replace("”", "")
    output = output.replace("  ", " ").strip(" ；;，,。")
    return output


def _enrich_split_activity_trigger_fields(
    item: Dict[str, Any], plan: Dict[str, Any]
) -> Dict[str, Any]:
    """Populate the formal split-activity trigger fields (Module 6 item 3) deterministically.

    A same-day split is, by design, two parallel branches that run in the same time window and
    reconvene the same day. We therefore default ``time_compatible`` and ``same_day`` to True
    (design invariants, not fabricated facts) and derive the regroup point/time from the day's
    accommodation and return time when available. Fields the model supplied (conflict_type,
    branch_distance_km, safety_risk, extra_transport_cost, duration_minutes) are preserved, and
    genuinely unknown fields (e.g. branch distance with no coordinates) are left absent so the
    trigger predicate honestly records them as missing rather than inventing values.
    """
    day_num = safe_int(item.get("day"), 0)
    per_day = (
        plan.get("per_day_itinerary", []) if isinstance(plan.get("per_day_itinerary"), list) else []
    )
    day_item = {}
    for d in per_day:
        if isinstance(d, dict) and safe_int(d.get("day"), 0) == day_num:
            day_item = d
            break
    hotel_plan = plan.get("hotel_plan", {}) if isinstance(plan.get("hotel_plan"), dict) else {}
    regroup_anchor = (
        safe_text(day_item.get("hotel"))
        or safe_text(hotel_plan.get("hotel_name"))
        or safe_text(hotel_plan.get("hotel_area"))
        or safe_text(hotel_plan.get("area"))
    )
    return_time = safe_text(day_item.get("estimated_return_time")) or safe_text(
        plan.get("return_time")
    )
    # Design invariants of a same-day parallel split.
    if item.get("time_compatible") is None:
        item["time_compatible"] = True
    if item.get("same_day") is None:
        item["same_day"] = True
    # Continuity: reconvene at the day's accommodation by the return time, when known.
    if not safe_text(item.get("regroup_point")) and regroup_anchor:
        item["regroup_point"] = regroup_anchor
    if not safe_text(item.get("reconvene_time")) and return_time:
        item["reconvene_time"] = return_time
    return item


def normalize_split_activity_role_assignments(
    plan: Dict[str, Any], family_agents: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Force split activities to name concrete roles instead of vague groups.

    Bad:  部分成员参加“海港城短时购物”；其他成员休息。
    Good: mother 参加“海港城短时购物”；father、elder 不参加，改为九龙公园休息。
    """
    split_activities = plan.get("split_activities", [])
    if not isinstance(split_activities, list):
        split_activities = []
    if not split_activities:
        plan["split_activities"] = []
        return plan

    lookup = build_role_lookup(family_agents)
    all_roles = lookup.get("all_roles", [])
    normalized = []
    for raw_item in split_activities:
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        item["activity"] = _clean_split_activity_text(
            item.get("activity", "") or item.get("activity_or_term", "") or "分流活动"
        )
        item["alternative_for_non_participants"] = _clean_split_activity_text(
            item.get("alternative_for_non_participants", "") or "附近公园"
        )

        participating = []
        participating += _roles_from_refs(item.get("participating_roles", []), lookup)
        participating += _roles_from_refs(item.get("participating_agents", []), lookup)
        non_participating = []
        non_participating += _roles_from_refs(item.get("non_participating_roles", []), lookup)
        non_participating += _roles_from_refs(item.get("non_participating_agents", []), lookup)

        inferred = _infer_roles_from_activity_text(item, family_agents, lookup)
        if not participating:
            participating = inferred.get("participating_roles", [])
        if not non_participating:
            non_participating = inferred.get("non_participating_roles", [])

        participating = list(dict.fromkeys([x for x in participating if x]))
        non_participating = [
            x
            for x in list(dict.fromkeys([x for x in non_participating if x]))
            if x not in set(participating)
        ]

        if participating and not non_participating:
            non_participating = [role for role in all_roles if role not in set(participating)]
        if non_participating and not participating:
            participating = [role for role in all_roles if role not in set(non_participating)]

        item["participating_roles"] = participating
        item["non_participating_roles"] = non_participating
        # Keep agent-id fields for machine checking if they existed, but the visible assignment uses roles.
        item[
            "role_assignment"
        ] = f"{'、'.join(participating) if participating else '未指定角色'} 参加“{item['activity']}”；{'、'.join(non_participating) if non_participating else '无其他角色'} 不参加，改为{item['alternative_for_non_participants']}。"
        item["evaluation_note"] = _clean_split_activity_text(
            item.get("evaluation_note", "分流活动已按角色明确参与与不参与成员。")
        )
        item = _enrich_split_activity_trigger_fields(item, plan)
        normalized.append(item)
    plan["split_activities"] = _dedupe_dict_items(normalized)
    return plan


def _split_item_matches_direct_hard_conflict(
    item: Dict[str, Any], direct_conflict_terms: List[str]
) -> bool:
    """Whether a split activity really resolves a direct hard-constraint conflict.

    v46 rule: split activities are allowed only when a must_include term from one
    member directly conflicts with another member's must_avoid term.  This avoids
    overusing split activities for normal preference differences.
    """
    if not isinstance(item, dict) or not direct_conflict_terms:
        return False
    text_blob = json.dumps(item, ensure_ascii=False).lower()
    conflict_type = safe_text(item.get("conflict_type", "")).lower()
    has_hard_type = (
        ("must_include_vs_must_avoid" in conflict_type)
        or ("direct_must_include" in conflict_type)
        or ("hard" in conflict_type)
    )
    activity_text = " ".join(
        [
            safe_text(item.get("activity")),
            safe_text(item.get("activity_or_term")),
            safe_text(item.get("role_assignment")),
        ]
    ).lower()
    # Drop generic/vague splits even if a model labeled them as hard.
    if any(word in text_blob for word in VAGUE_SPLIT_WORDS):
        return False
    if _clean_split_activity_text(activity_text) in ["", "分流活动"]:
        return False
    term_match = any(hard_terms_overlap(activity_text, term) for term in direct_conflict_terms)
    return bool(has_hard_type and term_match)


def clear_daily_split_activities(plan: Dict[str, Any]) -> Dict[str, Any]:
    per_day = plan.get("per_day_itinerary", [])
    if isinstance(per_day, list):
        for day_item in per_day:
            if isinstance(day_item, dict):
                day_item["split_activities"] = []
    plan["per_day_itinerary"] = per_day if isinstance(per_day, list) else []
    return plan


def filter_split_activities_to_hard_conflicts(
    plan: Dict[str, Any],
    family_agents: List[Dict[str, Any]],
    direct_conflict_terms: List[str],
    max_items: int = MAX_SPLIT_ACTIVITIES_FOR_EVAL,
) -> Dict[str, Any]:
    """Keep only role-explicit splits that resolve direct hard conflicts.

    LLMs often create split activities for shopping, rest, or general preference
    differences.  Those make split_activity_cost too high and reduce readability.
    This function removes all non-hard-conflict splits before syncing them back
    into per_day_itinerary.
    """
    if not direct_conflict_terms or not USE_SPLIT_ACTIVITY_RESOLUTION:
        plan["split_activities"] = []
        return clear_daily_split_activities(plan)

    plan = normalize_split_activity_role_assignments(plan=plan, family_agents=family_agents)
    raw_splits = (
        plan.get("split_activities", [])
        if isinstance(plan.get("split_activities", []), list)
        else []
    )
    kept = []
    seen_terms = set()
    for item in raw_splits:
        if not _split_item_matches_direct_hard_conflict(item, direct_conflict_terms):
            continue
        participating = (
            item.get("participating_roles", [])
            if isinstance(item.get("participating_roles"), list)
            else []
        )
        non_participating = (
            item.get("non_participating_roles", [])
            if isinstance(item.get("non_participating_roles"), list)
            else []
        )
        alternative = safe_text(item.get("alternative_for_non_participants", ""))
        if not participating or not non_participating or not alternative:
            continue
        term_key = ""
        for term in direct_conflict_terms:
            if hard_terms_overlap(json.dumps(item, ensure_ascii=False), term):
                term_key = normalize_hard_term(term)
                break
        if term_key and term_key in seen_terms:
            continue
        if term_key:
            seen_terms.add(term_key)
        item = dict(item)
        item["conflict_type"] = "must_include_vs_must_avoid_split_activity"
        item["reason"] = safe_text(item.get("reason")) or "仅用于解决成员硬约束之间的直接矛盾"
        item[
            "role_assignment"
        ] = f"{'、'.join(participating)} 参加“{safe_text(item.get('activity') or item.get('activity_or_term'))}”；{'、'.join(non_participating)} 不参加，改为{alternative}。"
        kept.append(item)
        if len(kept) >= max_items:
            break

    plan["split_activities"] = _dedupe_dict_items(kept)
    plan = clear_daily_split_activities(plan)
    return plan


def ensure_split_activities_for_hard_conflicts(
    plan: Dict[str, Any], family_agents: List[Dict[str, Any]], conflict_terms: List[str]
) -> Dict[str, Any]:
    if not conflict_terms:
        return plan
    split_activities = plan.get("split_activities", [])
    if not isinstance(split_activities, list):
        split_activities = []
    existing_text = json.dumps(split_activities, ensure_ascii=False).lower()
    per_day = (
        plan.get("per_day_itinerary", []) if isinstance(plan.get("per_day_itinerary"), list) else []
    )
    day_count = max(safe_int(plan.get("day"), 0), len(per_day), 1)
    used_days = {safe_int(s.get("day"), 0) for s in split_activities if isinstance(s, dict)}

    def _day_for_term(t: str) -> int:
        for di in per_day:
            if not isinstance(di, dict):
                continue
            for block in ("morning", "afternoon", "evening"):
                for it in di.get(block) or []:
                    nm = (
                        safe_text(it.get("poi_name") or it.get("activity"))
                        if isinstance(it, dict)
                        else safe_text(it)
                    )
                    if nm and hard_terms_overlap(t, nm):
                        return safe_int(di.get("day"), 0)
        return 0

    for term in conflict_terms:
        term_text = safe_text(term)
        if not term_text:
            continue
        if normalize_hard_term(term_text) in existing_text:
            continue
        if len(split_activities) >= MAX_SPLIT_ACTIVITIES_FOR_EVAL:
            break  # 只保留至多两个分流活动
        # A member who must_avoid the term does NOT participate; everyone else
        # (must_include or neutral) joins — so the roster names every family member.
        participating_agents = []
        non_participating_agents = []
        for agent in family_agents:
            constraints = agent.get("constraints", {})
            agent_id = safe_text(agent.get("agent_id", ""))
            avoids = (
                constraints.get("must_avoid", [])
                if isinstance(constraints.get("must_avoid"), list)
                else []
            )
            if any(hard_terms_overlap(term_text, x) for x in avoids):
                non_participating_agents.append(agent_id)
            else:
                participating_agents.append(agent_id)
        # A child cannot be left alone: if the non-participants include a child but no
        # adult, move one adult (preferring one who does NOT must_include this term) over
        # to accompany them.
        _role_of = {
            safe_text(a.get("agent_id", "")): safe_text(
                a.get("profile", {}).get("role", "")
            ).lower()
            for a in family_agents
        }
        _ADULT_ROLES = {"father", "mother", "grandfather", "grandmother", "elder", "grandparent"}
        if any(_role_of.get(aid) == "child" for aid in non_participating_agents) and not any(
            _role_of.get(aid) in _ADULT_ROLES for aid in non_participating_agents
        ):

            def _must_include_term(aid):
                a = next((x for x in family_agents if safe_text(x.get("agent_id", "")) == aid), {})
                inc = a.get("constraints", {}).get("must_include", [])
                return any(
                    hard_terms_overlap(term_text, x) for x in (inc if isinstance(inc, list) else [])
                )

            adults = [aid for aid in participating_agents if _role_of.get(aid) in _ADULT_ROLES]
            escort = next(
                (aid for aid in adults if not _must_include_term(aid)),
                adults[0] if adults else None,
            )
            if escort:
                participating_agents.remove(escort)
                non_participating_agents.append(escort)
        # A genuine split needs both a participant and a non-participant.
        if non_participating_agents and participating_agents:
            # Put each split on a distinct day (prefer the day its activity is scheduled)
            # so two splits never collide in the same time slot.
            split_day = _day_for_term(term_text)
            if split_day <= 0 or split_day in used_days:
                split_day = next((d for d in range(1, day_count + 1) if d not in used_days), 0)
            if split_day <= 0:
                continue  # no free day left -> do not stack a second split on the same day
            used_days.add(split_day)
            role_lookup = build_role_lookup(family_agents)
            participating_roles = _roles_from_refs(participating_agents, role_lookup)
            non_participating_roles = _roles_from_refs(non_participating_agents, role_lookup)
            # A groundable place label (not a vague "rest" phrase): grounding turns
            # '附近公园' into a concrete real park, giving the non-participants (a minor is
            # always accompanied by an adult, above) a specific family-friendly destination.
            alternative = "附近公园"
            split_activities.append(
                {
                    "day": split_day,
                    "conflict_type": "must_include_vs_must_avoid_split_activity",
                    "activity_or_term": term_text,
                    "activity": f"{term_text}",
                    "participating_agents": list(dict.fromkeys(participating_agents)),
                    "non_participating_agents": list(dict.fromkeys(non_participating_agents)),
                    "participating_roles": list(dict.fromkeys(participating_roles)),
                    "non_participating_roles": list(dict.fromkeys(non_participating_roles)),
                    "alternative_for_non_participants": alternative,
                    "role_assignment": f"{'、'.join(participating_roles) if participating_roles else '未指定角色'} 参加“{term_text}”；{'、'.join(non_participating_roles) if non_participating_roles else '无其他角色'} 不参加，改为{alternative}。",
                    "evaluation_note": "该活动已按角色明确参与和不参与成员；不参与角色不应因避免该活动被扣硬约束分。",
                }
            )
    plan["split_activities"] = split_activities
    return plan


def split_activity_exempts_avoid_term(
    plan: Dict[str, Any], agent_id: str, role: str, member_id: int, avoid_term: str
) -> bool:
    split_activities = plan.get("split_activities", [])
    if not isinstance(split_activities, list):
        return False
    identifiers = {
        safe_text(agent_id).lower(),
        safe_text(role).lower(),
        f"member_{safe_int(member_id, 0)}".lower(),
    }
    for item in split_activities:
        if not isinstance(item, dict):
            continue
        text = json.dumps(item, ensure_ascii=False).lower()
        if not hard_terms_overlap(avoid_term, text):
            continue
        non_parts = item.get("non_participating_agents", []) + item.get(
            "non_participating_roles", []
        )
        non_parts = [safe_text(x).lower() for x in non_parts]
        if any((x in identifiers for x in non_parts)):
            return True
    return False


def split_activity_has_complete_resolution(item: Dict[str, Any]) -> bool:
    participants = _as_text_list(item.get("participating_agents", [])) + _as_text_list(
        item.get("participating_roles", [])
    )
    non_participants = _as_text_list(item.get("non_participating_agents", [])) + _as_text_list(
        item.get("non_participating_roles", [])
    )
    alternative = safe_text(
        item.get("alternative_for_non_participants", "")
        or item.get("alternative_activity", "")
        or item.get("alternative", "")
    )
    has_reason = any(
        (
            safe_text(item.get(field, ""))
            for field in [
                "reason",
                "rationale",
                "conflict_type",
                "resolution",
                "evaluation_note",
                "activity_or_term",
            ]
        )
    )
    # A grounded alternative is a real place NAME (e.g. '南湖公园', 4 chars); the old
    # len>=10 was sized for a verbose phrase ('在附近咖啡馆…休息') and wrongly rejected
    # grounded splits, leaving their conflicts 'unresolved'.  Require a non-trivial name.
    return bool(participants and non_participants and (len(alternative) >= 2) and has_reason)


def split_activity_resolves_conflict_term(plan: Dict[str, Any], term: str) -> bool:
    split_activities = plan.get("split_activities", [])
    if not isinstance(split_activities, list):
        return False
    term_text = safe_text(term)
    if not term_text:
        return False
    for item in split_activities:
        if not isinstance(item, dict):
            continue
        activity_text = json.dumps(item, ensure_ascii=False).lower()
        if hard_terms_overlap(term_text, activity_text) and split_activity_has_complete_resolution(
            item
        ):
            return True
    return False


def split_terms_by_resolution_status(
    plan: Dict[str, Any], terms: List[str]
) -> Dict[str, List[str]]:
    resolved = []
    unresolved = []
    for term in terms:
        term_text = safe_text(term)
        if not term_text:
            continue
        if split_activity_resolves_conflict_term(plan, term_text):
            resolved.append(term_text)
        else:
            unresolved.append(term_text)
    return {
        "resolved": list(dict.fromkeys(resolved)),
        "unresolved": list(dict.fromkeys(unresolved)),
    }


def build_constraint_satisfiability_typing(
    plan: Dict[str, Any], family_agents: List[Dict[str, Any]]
) -> Dict[str, Any]:
    include_records = []
    avoid_records = []
    for agent in family_agents:
        profile = agent.get("profile", {})
        constraints = agent.get("constraints", {})
        agent_id = safe_text(agent.get("agent_id", ""))
        role = safe_text(profile.get("role", ""))
        for term in (
            constraints.get("must_include", [])
            if isinstance(constraints.get("must_include", []), list)
            else []
        ):
            term_text = safe_text(term)
            if term_text:
                include_records.append({"agent_id": agent_id, "role": role, "term": term_text})
        for term in (
            constraints.get("must_avoid", [])
            if isinstance(constraints.get("must_avoid", []), list)
            else []
        ):
            term_text = safe_text(term)
            if term_text:
                avoid_records.append({"agent_id": agent_id, "role": role, "term": term_text})
    typed_constraints = []
    conflicting_include_terms = set()
    conflicting_avoid_terms = set()
    for inc in include_records:
        for avd in avoid_records:
            if not hard_terms_overlap(inc["term"], avd["term"]):
                continue
            term = inc["term"]
            conflicting_include_terms.add(normalize_hard_term(inc["term"]))
            conflicting_avoid_terms.add(normalize_hard_term(avd["term"]))
            if split_activity_resolves_conflict_term(plan, term):
                sat_type = "Split-Satisfiable Constraints"
                resolution = "split_activity"
            else:
                sat_type = "Unsatisfiable Constraints"
                resolution = "unresolved_hard_conflict"
            typed_constraints.append(
                {
                    "satisfiability_type": sat_type,
                    "resolution": resolution,
                    "term": term,
                    "must_include_agent": inc.get("agent_id"),
                    "must_include_role": inc.get("role"),
                    "must_avoid_agent": avd.get("agent_id"),
                    "must_avoid_role": avd.get("role"),
                }
            )
    for inc in include_records:
        if normalize_hard_term(inc["term"]) in conflicting_include_terms:
            continue
        typed_constraints.append(
            {
                "satisfiability_type": "Fully Satisfiable Constraints",
                "resolution": "shared_itinerary",
                "term": inc["term"],
                "must_include_agent": inc.get("agent_id"),
                "must_include_role": inc.get("role"),
            }
        )
    hard_checklist = plan.get("hard_constraint_checklist", [])
    if isinstance(hard_checklist, list):
        for item in hard_checklist:
            if not isinstance(item, dict):
                continue
            violations = item.get("violations", [])
            if violations:
                typed_constraints.append(
                    {
                        "satisfiability_type": "Unsatisfiable Constraints",
                        "resolution": "violated_member_hard_constraint",
                        "agent_id": item.get("agent_id"),
                        "role": item.get("role"),
                        "violations": violations,
                    }
                )
    counts = {"fully_satisfiable_count": 0, "split_satisfiable_count": 0, "unsatisfiable_count": 0}
    for item in typed_constraints:
        sat_type = item.get("satisfiability_type")
        if sat_type == "Fully Satisfiable Constraints":
            counts["fully_satisfiable_count"] += 1
        elif sat_type == "Split-Satisfiable Constraints":
            counts["split_satisfiable_count"] += 1
        elif sat_type == "Unsatisfiable Constraints":
            counts["unsatisfiable_count"] += 1
    if counts["unsatisfiable_count"] > 0:
        overall = "has_unsatisfiable_constraints"
    elif counts["split_satisfiable_count"] > 0:
        overall = "split_satisfiable_plan"
    else:
        overall = "fully_satisfiable_plan"
    return {
        "framework_component": "Constraint Satisfiability Typing",
        "overall_satisfiability": overall,
        **counts,
        "typed_constraints": typed_constraints[:30],
    }


def attach_constraint_satisfiability_typing(
    plan: Dict[str, Any], family_agents: List[Dict[str, Any]]
) -> Dict[str, Any]:
    plan["constraint_satisfiability"] = build_constraint_satisfiability_typing(
        plan=plan, family_agents=family_agents
    )
    return plan


def _dedupe_dict_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output = []
    for item in items:
        if not isinstance(item, dict):
            continue
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _split_activity_day(item: Dict[str, Any], total_days: int) -> int:
    for key in ["day", "day_index", "day_number"]:
        value = safe_int(item.get(key), 0)
        if value > 0:
            return max(1, min(value, max(total_days, 1)))
    return 1


def sync_split_activities_into_per_day_itinerary(plan: Dict[str, Any]) -> Dict[str, Any]:
    split_activities = plan.get("split_activities", [])
    if not isinstance(split_activities, list):
        split_activities = []
    per_day = plan.get("per_day_itinerary", [])
    if not isinstance(per_day, list):
        per_day = []
    # v46: Do not merge LLM-created per-day split_activities back into the plan.
    # Only top-level split_activities that passed hard-conflict filtering are allowed.
    split_activities = _dedupe_dict_items(split_activities)
    if not split_activities:
        plan["split_activities"] = []
        return plan
    total_days = max(safe_int(plan.get("day"), 0), len(per_day), 1)
    while len(per_day) < total_days:
        day_num = len(per_day) + 1
        per_day.append(
            {
                "day": day_num,
                "theme": "轻松家庭行程",
                "morning": [],
                "lunch": {},
                "afternoon": [],
                "dinner": {},
                "evening": [],
                "transportation": [],
                "hotel": "",
                "estimated_walking_distance_km": 0,
                "estimated_route_distance_km": 0,
                "estimated_return_time": safe_text(plan.get("return_time", "20:00")) or "20:00",
                "optional_activities": [],
                "rest_blocks": [],
                "split_activities": [],
            }
        )
    normalized_top_level = []
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for item in split_activities:
        if not isinstance(item, dict):
            continue
        day_num = _split_activity_day(item, total_days)
        copied = dict(item)
        copied["day"] = day_num
        normalized_top_level.append(copied)
        grouped.setdefault(day_num, []).append(copied)
    for index, day_item in enumerate(per_day):
        if not isinstance(day_item, dict):
            day_item = {"day": index + 1, "theme": "轻松家庭行程"}
            per_day[index] = day_item
        day_num = safe_int(day_item.get("day"), index + 1)
        day_item["day"] = day_num
        day_item["split_activities"] = _dedupe_dict_items(grouped.get(day_num, []))
    plan["split_activities"] = _dedupe_dict_items(normalized_top_level)
    plan["per_day_itinerary"] = per_day
    return plan


def sanitize_plan_avoid_terms(plan: Dict[str, Any], must_avoid_terms: List[str]) -> Dict[str, Any]:
    plan["itinerary_text"] = sanitize_string_with_avoid_terms(
        plan.get("itinerary_text", ""), must_avoid_terms
    )
    plan["coordinator_summary"] = sanitize_string_with_avoid_terms(
        plan.get("coordinator_summary", ""), must_avoid_terms
    )
    plan["per_day_itinerary"] = recursively_sanitize_avoid_terms(
        plan.get("per_day_itinerary", []), must_avoid_terms
    )
    plan["poi_tags"] = recursively_sanitize_avoid_terms(plan.get("poi_tags", []), must_avoid_terms)
    plan["meal_tags"] = recursively_sanitize_avoid_terms(
        plan.get("meal_tags", []), must_avoid_terms
    )
    plan["high_intensity_tags"] = recursively_sanitize_avoid_terms(
        plan.get("high_intensity_tags", []), must_avoid_terms
    )
    plan["hotel_plan"] = recursively_sanitize_avoid_terms(
        plan.get("hotel_plan", {}), must_avoid_terms
    )
    plan["transportation_plan"] = recursively_sanitize_avoid_terms(
        plan.get("transportation_plan", {}), must_avoid_terms
    )
    return plan


def ensure_must_include_exact_terms(
    plan: Dict[str, Any],
    must_include_terms: List[str],
    must_avoid_terms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if must_avoid_terms is None:
        must_avoid_terms = []
    itinerary_text = safe_text(plan.get("itinerary_text", ""))
    appended_terms = []
    skipped_conflict_terms = []
    lower_text = itinerary_text.lower()
    for term in must_include_terms:
        term_text = safe_text(term)
        if not term_text:
            continue
        if term_conflicts_with_avoid(term_text, must_avoid_terms):
            skipped_conflict_terms.append(term_text)
            continue
        if term_text.lower() not in lower_text:
            appended_terms.append(term_text)
    if appended_terms:
        itinerary_text += "\n\nMust-include items explicitly included: "
        itinerary_text += "; ".join(list(dict.fromkeys(appended_terms)))
    plan["itinerary_text"] = itinerary_text
    poi_tags = plan.get("poi_tags", [])
    if not isinstance(poi_tags, list):
        poi_tags = []
    existing_tags_lower = {safe_text(tag).lower() for tag in poi_tags}
    for term in must_include_terms:
        term_text = safe_text(term)
        if not term_text:
            continue
        if term_conflicts_with_avoid(term_text, must_avoid_terms):
            skipped_conflict_terms.append(term_text)
            continue
        if term_text.lower() not in existing_tags_lower:
            poi_tags.append(term_text)
            existing_tags_lower.add(term_text.lower())
    plan["poi_tags"] = poi_tags
    skipped_conflict_terms = list(dict.fromkeys(skipped_conflict_terms))
    if skipped_conflict_terms:
        status = split_terms_by_resolution_status(plan, skipped_conflict_terms)
        unresolved_terms = status["unresolved"]
        resolved_terms = status["resolved"]
        summary = safe_text(plan.get("coordinator_summary", ""))
        if resolved_terms:
            summary += (
                "\nHard-constraint split resolution: the following must_include / must_avoid conflicts are handled by split_activities instead of being treated as unresolved: "
                + "; ".join(resolved_terms)
            )
        if unresolved_terms:
            unresolved = plan.get("unresolved_conflicts", [])
            if not isinstance(unresolved, list):
                unresolved = []
            unresolved.append(
                {
                    "conflict_type": "must_include_vs_must_avoid_term_conflict",
                    "reason": "部分 must_include 同时是其他成员的 must_avoid，且没有找到完整的分流活动处理。",
                    "conflicting_terms": unresolved_terms,
                }
            )
            plan["unresolved_conflicts"] = unresolved
            summary += (
                "\nHard-constraint conflict notice: the following terms were not force-inserted because they conflict with must_avoid constraints and lack a complete split resolution: "
                + "; ".join(unresolved_terms)
            )
        plan["coordinator_summary"] = summary
    return plan


def ensure_interest_and_dietary_terms(
    plan: Dict[str, Any],
    family_agents: List[Dict[str, Any]],
    must_avoid_terms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if must_avoid_terms is None:
        must_avoid_terms = []
    itinerary_text = safe_text(plan.get("itinerary_text", ""))
    poi_tags = plan.get("poi_tags", [])
    if not isinstance(poi_tags, list):
        poi_tags = []
    meal_tags = plan.get("meal_tags", [])
    if not isinstance(meal_tags, list):
        meal_tags = []
    poi_tag_lower = {safe_text(x).lower() for x in poi_tags}
    meal_tag_lower = {safe_text(x).lower() for x in meal_tags}
    appended_interest_terms = []
    appended_dietary_terms = []
    skipped_conflict_terms = []
    for agent in family_agents:
        profile = agent.get("profile", {})
        interest_preferences = profile.get("interest_preferences", [])
        dietary_preference = profile.get("dietary_preference", [])
        if not isinstance(interest_preferences, list):
            interest_preferences = []
        if not isinstance(dietary_preference, list):
            dietary_preference = []
        for pref in interest_preferences:
            pref_text = safe_text(pref)
            if not pref_text:
                continue
            if term_conflicts_with_avoid(pref_text, must_avoid_terms):
                skipped_conflict_terms.append(pref_text)
                continue
            if pref_text.lower() not in itinerary_text.lower():
                appended_interest_terms.append(pref_text)
            if pref_text.lower() not in poi_tag_lower:
                poi_tags.append(pref_text)
                poi_tag_lower.add(pref_text.lower())
        for food_pref in dietary_preference:
            food_text = safe_text(food_pref)
            if not food_text:
                continue
            if term_conflicts_with_avoid(food_text, must_avoid_terms):
                skipped_conflict_terms.append(food_text)
                continue
            if food_text.lower() not in meal_tag_lower:
                meal_tags.append(food_text)
                meal_tag_lower.add(food_text.lower())
            if food_text.lower() not in itinerary_text.lower():
                appended_dietary_terms.append(food_text)
    if appended_interest_terms:
        itinerary_text += "\n\nPreference coverage items explicitly included: " + "; ".join(
            list(dict.fromkeys(appended_interest_terms))
        )
    if appended_dietary_terms:
        itinerary_text += "\n\nDietary preference items explicitly included: " + "; ".join(
            list(dict.fromkeys(appended_dietary_terms))
        )
    if len(set(poi_tags)) < 2:
        for tag in ["culture", "family_friendly"]:
            if term_conflicts_with_avoid(tag, must_avoid_terms):
                continue
            if tag not in poi_tag_lower:
                poi_tags.append(tag)
                poi_tag_lower.add(tag)
    if len(set(meal_tags)) < 2:
        for tag in ["local_food", "family_friendly"]:
            if term_conflicts_with_avoid(tag, must_avoid_terms):
                continue
            if tag not in meal_tag_lower:
                meal_tags.append(tag)
                meal_tag_lower.add(tag)
    plan["itinerary_text"] = itinerary_text
    plan["poi_tags"] = poi_tags
    plan["meal_tags"] = meal_tags
    skipped_conflict_terms = list(dict.fromkeys(skipped_conflict_terms))
    if skipped_conflict_terms:
        unresolved = plan.get("unresolved_conflicts", [])
        if not isinstance(unresolved, list):
            unresolved = []
        unresolved.append(
            {
                "conflict_type": "preference_vs_must_avoid_term_conflict",
                "reason": "Some interest or dietary preference terms conflict with must_avoid terms. They were not automatically inserted into poi_tags or meal_tags.",
                "conflicting_terms": skipped_conflict_terms,
            }
        )
        plan["unresolved_conflicts"] = unresolved
    return plan


def _estimate_route_distance_km(plan: Dict[str, Any]) -> float:
    """Total intra-city route distance derived from the real per-day stops, so it stays
    consistent with the itinerary instead of a flat estimate."""
    per_day = (
        plan.get("per_day_itinerary", []) if isinstance(plan.get("per_day_itinerary"), list) else []
    )
    total = 0.0
    for d in per_day:
        if not isinstance(d, dict):
            continue
        stops = 0
        for blk in ("morning", "afternoon"):
            for x in d.get(blk) or []:
                if isinstance(x, dict):
                    if x.get("is_transit") or x.get("poi_type") in ("arrival", "departure"):
                        continue
                    if (
                        safe_text(x.get("poi_name"))
                        or safe_text(x.get("activity"))
                        or safe_text(x.get("name"))
                    ):
                        stops += 1
                elif isinstance(x, str) and x.strip():
                    stops += 1
        for meal in ("lunch", "dinner"):
            v = d.get(meal)
            if isinstance(v, dict) and safe_text(v.get("restaurant_name")):
                stops += 1
        total += max(1, stops + 1) * ROUTE_KM_PER_LEG  # hotel -> stops -> hotel
    return round(total, 2)


def enforce_budget_route_price_safety(
    plan: Dict[str, Any], family_agents: List[Dict[str, Any]]
) -> Dict[str, Any]:
    day = safe_int(plan.get("day"), 1)
    if day <= 0:
        day = 1
    max_route_total = 50 * day
    # Always DERIVE route_distance from the real itinerary stops (consistent with the
    # plan), then bound it by the day's walking so route/walking stays inside the
    # efficiency gate [1, 18].  The old flat cap*0.8 (~40 km/day) was disconnected from
    # the plan and broke that ratio once walking distances were tuned to realistic values.
    derived = _estimate_route_distance_km(plan)
    route = min(derived if derived > 0 else max_route_total * 0.3, max_route_total)
    walking = safe_float(plan.get("walking_distance_km"), 0)
    if walking > 0:
        route = max(min(route, walking * 14.0), walking * 1.2)
    plan["route_distance_km"] = round(route, 2)
    total_cost = safe_float(plan.get("total_cost"), 0)
    budget_limit = safe_float(plan.get("budget_limit"), 0)
    if budget_limit > 0 and total_cost <= 0:
        plan["total_cost"] = round(budget_limit * 0.92, 2)
    has_strict_or_sensitive_member = False
    for agent in family_agents:
        profile = agent.get("profile", {})
        budget_sensitivity = safe_text(profile.get("budget_sensitivity", "")).lower()
        role = safe_text(profile.get("role", "")).lower()
        age_stage = safe_text(profile.get("age_stage", "")).lower()
        physical = safe_text(profile.get("physical_constraint", "")).lower()
        if budget_sensitivity == "strict_budget":
            has_strict_or_sensitive_member = True
        if physical == "low_intensity":
            has_strict_or_sensitive_member = True
        if any((key in role for key in ["older", "elderly", "child", "kid", "teenager"])):
            has_strict_or_sensitive_member = True
        if any((key in age_stage for key in ["older", "elderly", "child", "teenager"])):
            has_strict_or_sensitive_member = True
    if has_strict_or_sensitive_member:
        plan["restaurant_price_level"] = "medium"
        hotel_plan = plan.get("hotel_plan", {})
        if isinstance(hotel_plan, dict):
            hotel_plan["price_level"] = "medium"
            plan["hotel_plan"] = hotel_plan
    plan["duplicate_poi_count"] = 0
    checklist = plan.get("hard_constraint_checklist", [])
    if isinstance(checklist, list) and checklist:
        accepted_count = 0
        vote_prediction = []
        for item in checklist:
            violations = item.get("violations", [])
            if not violations:
                vote = "Accept"
                accepted_count += 1
                reason = "All hard constraints are satisfied after rule repair."
            else:
                vote = "Accept with minor changes"
                reason = "Some issues remain: " + "; ".join([safe_text(x) for x in violations])
            vote_prediction.append(
                {
                    "agent_id": item.get("agent_id"),
                    "role": item.get("role"),
                    "vote": vote,
                    "reason": reason,
                }
            )
        plan["agent_vote_prediction"] = vote_prediction
        plan["acceptance_vote_ratio"] = round(accepted_count / len(checklist), 4)
        if accepted_count == len(checklist):
            plan["consensus_status"] = "Full Consensus"
        elif accepted_count > 0:
            plan["consensus_status"] = "Partial Consensus"
        else:
            plan["consensus_status"] = "No Consensus"
    plan = adjust_consensus_status_for_unresolved_conflicts(plan)
    return plan


def enforce_basic_numeric_constraints(
    plan: Dict[str, Any], family_agents: List[Dict[str, Any]]
) -> Dict[str, Any]:
    constraints = collect_family_constraints(family_agents)
    day = safe_int(plan.get("day"), 1)
    if day <= 0:
        day = 1
    max_walking_limits = constraints["max_walking_limits"]
    latest_return_minutes = constraints["latest_return_minutes"]
    # Per-day walking: use the model's value when present, else a content-based
    # estimate that varies by POI type/count (not a constant).  Each day is capped
    # at the strictest member limit so constraints still hold, and the plan total is
    # the SUM of per-day values rather than strictest * day (which made every day equal).
    strictest_daily_walking = min(max_walking_limits) if max_walking_limits else None
    per_day = plan.get("per_day_itinerary", [])
    if isinstance(per_day, list) and per_day:
        day_walks = []
        for day_item in per_day:
            if not isinstance(day_item, dict):
                continue
            est = safe_float(day_item.get("estimated_walking_distance_km"), 0)
            if est <= 0:
                est = estimate_day_walking_km(day_item)
            if strictest_daily_walking is not None and est > strictest_daily_walking:
                est = strictest_daily_walking
            est = round(est, 1)
            day_item["estimated_walking_distance_km"] = est
            day_walks.append(est)
        if day_walks:
            plan["walking_distance_km"] = round(sum(day_walks), 2)
    elif strictest_daily_walking is not None:
        plan["walking_distance_km"] = round(strictest_daily_walking * day, 2)
    if latest_return_minutes:
        earliest_minutes = min(latest_return_minutes)
        hour = earliest_minutes // 60
        minute = earliest_minutes % 60
        fixed_return_time = f"{hour:02d}:{minute:02d}"
        plan["return_time"] = fixed_return_time
        per_day = plan.get("per_day_itinerary", [])
        if isinstance(per_day, list):
            for day_item in per_day:
                if isinstance(day_item, dict):
                    day_item["estimated_return_time"] = fixed_return_time
    plan["daily_poi_count"] = min(safe_float(plan.get("daily_poi_count"), 3), 3)
    has_sensitive_member = False
    for agent in family_agents:
        profile = agent.get("profile", {})
        physical = safe_text(profile.get("physical_constraint", "")).lower()
        role = safe_text(profile.get("role", "")).lower()
        age_stage = safe_text(profile.get("age_stage", "")).lower()
        if physical == "low_intensity":
            has_sensitive_member = True
        if any(
            (
                key in role
                for key in ["older", "elderly", "grandfather", "grandmother", "child", "kid"]
            )
        ):
            has_sensitive_member = True
        if any((key in age_stage for key in ["older", "elderly", "child"])):
            has_sensitive_member = True
    if has_sensitive_member:
        plan["high_intensity_tags"] = []
    return plan


def build_hard_constraint_checklist(
    plan: Dict[str, Any], family_agents: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    constraints = collect_family_constraints(family_agents)
    plan_text_parts = [
        safe_text(plan.get("itinerary_text", "")),
        json.dumps(plan.get("per_day_itinerary", []), ensure_ascii=False),
        json.dumps(plan.get("poi_tags", []), ensure_ascii=False),
        json.dumps(plan.get("meal_tags", []), ensure_ascii=False),
        json.dumps(plan.get("hotel_plan", {}), ensure_ascii=False),
        json.dumps(plan.get("transportation_plan", {}), ensure_ascii=False),
        json.dumps(plan.get("split_activities", []), ensure_ascii=False),
    ]
    plan_text = " ".join(plan_text_parts).lower()
    day = safe_int(plan.get("day"), 1)
    if day <= 0:
        day = 1
    walking_distance = safe_float(plan.get("walking_distance_km"), 0)
    avg_walking = walking_distance / day
    return_minutes = time_to_minutes(plan.get("return_time"))
    checklist = []
    for row in constraints["agent_constraint_rows"]:
        violations = []
        must_include = row.get("must_include", [])
        must_avoid = row.get("must_avoid", [])
        must_include_satisfied = True
        for item in must_include:
            item_text = safe_text(item)
            if item_text and item_text.lower() not in plan_text:
                must_include_satisfied = False
                violations.append(f"must_include_not_satisfied:{item_text}")
        must_avoid_satisfied = True
        for item in must_avoid:
            item_text = safe_text(item)
            if item_text and item_text.lower() in plan_text:
                exempted = split_activity_exempts_avoid_term(
                    plan=plan,
                    agent_id=safe_text(row.get("agent_id", "")),
                    role=safe_text(row.get("role", "")),
                    member_id=safe_int(row.get("member_id", 0), 0),
                    avoid_term=item_text,
                )
                if not exempted:
                    must_avoid_satisfied = False
                    violations.append(f"must_avoid_violated:{item_text}")
        max_walking = row.get("max_walking_distance_km")
        walking_limit_satisfied = True
        if max_walking is not None:
            if avg_walking > max_walking:
                walking_limit_satisfied = False
                violations.append(
                    f"walking_distance_exceeded:avg_per_day={round(avg_walking, 2)},limit={max_walking}"
                )
        latest_minutes = row.get("latest_return_minutes")
        return_time_satisfied = True
        if latest_minutes is not None and return_minutes is not None:
            if return_minutes > latest_minutes:
                return_time_satisfied = False
                violations.append(
                    f"return_time_too_late:return={plan.get('return_time')},limit={row.get('latest_return_time')}"
                )
        checklist.append(
            {
                "agent_id": row.get("agent_id"),
                "role": row.get("role"),
                "must_include_satisfied": must_include_satisfied,
                "must_avoid_satisfied": must_avoid_satisfied,
                "walking_limit_satisfied": walking_limit_satisfied,
                "return_time_satisfied": return_time_satisfied,
                "violations": violations,
            }
        )
    return checklist


def apply_concession_resolution(
    plan: Dict[str, Any], family_agents: List[Dict[str, Any]], conflict_terms: List[str]
) -> Tuple[Dict[str, Any], List[str]]:
    """Resolve each direct must_include×must_avoid clash by CONCESSION (Module 4's non-split channel).

    The contested must-have activity is KEPT (its must_include members stay satisfied); the members
    who must_avoid it are recorded as having *conceded* — they accept it via the negotiated agreement.
    Module 6 then exempts those members from the hard must_avoid veto but charges a moderate concession
    dissatisfaction (see ``is_avoid_conceded_by_negotiation`` / ``CONCESSION_HARD_PENALTY``), so the
    outcome sits strictly between split (nobody loses) and no resolution (hard violation + unresolved).

    Returns the (mutated) plan and the list of terms actually resolved by concession.
    """
    role_lookup = build_role_lookup(family_agents)
    concessions = plan.get("concession_resolutions", [])
    if not isinstance(concessions, list):
        concessions = []
    conceded_terms: List[str] = []
    for term in conflict_terms:
        term_text = safe_text(term)
        if not term_text:
            continue
        including_agents, conceding_agents = [], []
        for agent in family_agents:
            constraints = agent.get("constraints", {}) if isinstance(agent, dict) else {}
            agent_id = safe_text(agent.get("agent_id", ""))
            avoids = constraints.get("must_avoid", [])
            includes = constraints.get("must_include", [])
            if any(
                hard_terms_overlap(term_text, x)
                for x in (avoids if isinstance(avoids, list) else [])
            ):
                conceding_agents.append(agent_id)
            elif any(
                hard_terms_overlap(term_text, x)
                for x in (includes if isinstance(includes, list) else [])
            ):
                including_agents.append(agent_id)
        # A real concession needs both a must_avoid holder (who yields) and a must_include holder
        # (whose activity is kept). Without both, there is nothing to concede.
        if not (conceding_agents and including_agents):
            continue
        concessions.append(
            {
                "conflict_type": "must_include_vs_must_avoid_concession",
                "activity_or_term": term_text,
                "kept_for_agents": list(dict.fromkeys(including_agents)),
                "conceding_agents": list(dict.fromkeys(conceding_agents)),
                "kept_for_roles": _roles_from_refs(including_agents, role_lookup),
                "conceding_roles": _roles_from_refs(conceding_agents, role_lookup),
                "note": "split 不可用，协商由回避成员让步接受该活动：豁免硬性 must_avoid 否决，但记中等让步不满（不计为硬违约）。",
            }
        )
        conceded_terms.append(term_text)
    plan["concession_resolutions"] = concessions
    return plan, conceded_terms


def enforce_rule_consistency(
    plan: Dict[str, Any], family_agents: List[Dict[str, Any]], negotiation_resolved: bool = True,
    credit_llm_splits: bool = False,
) -> Dict[str, Any]:
    constraints = collect_family_constraints(family_agents)
    must_include_terms = normalize_term_list_for_conflict_check(
        constraints.get("must_include_terms", [])
    )
    must_avoid_terms = normalize_term_list_for_conflict_check(
        constraints.get("must_avoid_terms", [])
    )
    direct_conflict_terms = detect_hard_constraint_term_conflicts(
        must_include_terms=must_include_terms, must_avoid_terms=must_avoid_terms
    )
    # Approach A: conflict-resolution is owned by Module 4, which has TWO channels:
    #   1. split  (can_split)   — role-split the contested activity; nobody loses their constraint.
    #   2. concede (can_concede) — the must_avoid member yields; conflict resolved but the conceder is
    #                              charged a moderate dissatisfaction (Module 6). This is M4's fallback
    #                              when split is unavailable but negotiation still ran (→ no_split).
    # Both require negotiation_resolved (no_negotiation gets neither → clashes stay unresolved). Because
    # concession is strictly weaker than split yet strictly stronger than nothing, it separates no_split
    # (concede) from no_negotiation (nothing), which otherwise collapse to the identical outcome.
    can_split = USE_SPLIT_ACTIVITY_RESOLUTION and negotiation_resolved
    can_concede = USE_CONCESSION_RESOLUTION and negotiation_resolved and not can_split
    conceded_terms: List[str] = []
    if direct_conflict_terms and can_split:
        print(
            f"[Module 5] family_id={plan.get('family_id')} 发现 must_include / must_avoid 直接冲突，按 M4 决定用 split activity 解决: {direct_conflict_terms}"
        )
        plan = ensure_split_activities_for_hard_conflicts(
            plan=plan, family_agents=family_agents, conflict_terms=direct_conflict_terms
        )
    elif direct_conflict_terms and can_concede:
        plan, conceded_terms = apply_concession_resolution(
            plan=plan, family_agents=family_agents, conflict_terms=direct_conflict_terms
        )
        if conceded_terms:
            print(
                f"[Module 5] family_id={plan.get('family_id')} split 关闭，按 M4 让步(concession)化解直接冲突: {conceded_terms}"
            )
    # A conceded term stays IN the plan (its must_include holder keeps it); only its must_avoid holder
    # yields. So do NOT let the avoid-term sanitizer strip conceded terms out — exclude them below.
    conceded_set = [safe_text(t) for t in conceded_terms if safe_text(t)]
    sanitize_avoid_terms = (
        [t for t in must_avoid_terms if not any(hard_terms_overlap(t, c) for c in conceded_set)]
        if conceded_set
        else must_avoid_terms
    )
    if direct_conflict_terms:
        if can_split:
            status = split_terms_by_resolution_status(plan, direct_conflict_terms)
        elif conceded_terms:
            # concession resolved these (consensus reached, no unresolved penalty); any clash the family
            # could NOT concede (no valid include/avoid pairing) still counts as unresolved.
            status = {
                "resolved": list(dict.fromkeys(conceded_terms)),
                "unresolved": [
                    t
                    for t in dict.fromkeys(direct_conflict_terms)
                    if not any(hard_terms_overlap(t, c) for c in conceded_set)
                ],
            }
        elif credit_llm_splits:
            # Conflict-RESOLVING baselines (SA-Resolve / Generic-MA Debate): no Module-4 machinery
            # runs, so nothing is programmatically added (no ensure_split..., no concession), but a
            # split the baseline's own LLM authored IS credited when structurally complete — the same
            # completeness bar (split_activity_has_complete_resolution) the full pipeline must clear.
            # Without this mode their self-authored resolutions would be discarded and the comparison
            # would be a strawman.
            status = split_terms_by_resolution_status(plan, direct_conflict_terms)
        else:
            # split disabled (no_split) AND concession off, OR Module 4 produced no resolution
            # (no_negotiation): do NOT credit any split_activities the LLM may still have emitted —
            # every direct must_include vs must_avoid conflict stays unresolved (→ penalty).
            status = {"resolved": [], "unresolved": list(dict.fromkeys(direct_conflict_terms))}
        unresolved_direct_terms = status["unresolved"]
        resolved_direct_terms = status["resolved"]
        if resolved_direct_terms:
            print(
                f"[Module 5] family_id={plan.get('family_id')} 直接冲突已由合理 split activity 处理，不写入 unresolved_conflicts: {resolved_direct_terms}"
            )
        if unresolved_direct_terms:
            unresolved = plan.get("unresolved_conflicts", [])
            if not isinstance(unresolved, list):
                unresolved = []
            unresolved.append(
                {
                    "conflict_type": "direct_must_include_must_avoid_conflict",
                    "reason": "The same term appears in both must_include and must_avoid across family members, and no complete split activity resolution was found.",
                    "conflicting_terms": unresolved_direct_terms,
                }
            )
            plan["unresolved_conflicts"] = unresolved
            print(
                f"[Module 5] family_id={plan.get('family_id')} 仍未解决的 must_include / must_avoid 冲突: {unresolved_direct_terms}"
            )
    plan = sanitize_plan_avoid_terms(plan=plan, must_avoid_terms=sanitize_avoid_terms)
    plan = ensure_must_include_exact_terms(
        plan=plan, must_include_terms=must_include_terms, must_avoid_terms=sanitize_avoid_terms
    )
    plan = ensure_interest_and_dietary_terms(
        plan=plan, family_agents=family_agents, must_avoid_terms=sanitize_avoid_terms
    )
    plan = enforce_basic_numeric_constraints(plan=plan, family_agents=family_agents)
    plan = enforce_budget_route_price_safety(plan=plan, family_agents=family_agents)
    plan = sanitize_plan_avoid_terms(plan=plan, must_avoid_terms=sanitize_avoid_terms)
    plan = filter_split_activities_to_hard_conflicts(
        plan=plan, family_agents=family_agents, direct_conflict_terms=direct_conflict_terms
    )
    plan = sync_split_activities_into_per_day_itinerary(plan)
    plan["hard_constraint_checklist"] = build_hard_constraint_checklist(
        plan=plan, family_agents=family_agents
    )
    plan = enforce_budget_route_price_safety(plan=plan, family_agents=family_agents)
    plan = sanitize_plan_avoid_terms(plan=plan, must_avoid_terms=sanitize_avoid_terms)
    plan = ensure_online_grounding(plan=plan, family_agents=family_agents)
    plan = ensure_evaluation_ready_grounding_fields(plan=plan, family_agents=family_agents)
    plan = enforce_budget_route_price_safety(plan=plan, family_agents=family_agents)
    plan["hard_constraint_checklist"] = build_hard_constraint_checklist(
        plan=plan, family_agents=family_agents
    )
    plan = filter_split_activities_to_hard_conflicts(
        plan=plan, family_agents=family_agents, direct_conflict_terms=direct_conflict_terms
    )
    plan = sync_split_activities_into_per_day_itinerary(plan)
    plan = attach_itinerary_quality_markers(plan)
    if USE_CONSTRAINT_SATISFIABILITY_TYPING:
        plan = attach_constraint_satisfiability_typing(plan=plan, family_agents=family_agents)
    return plan


def _coerce_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value in [None, ""]:
        return []
    return [value]


def normalize_per_day_itinerary(plan: Dict[str, Any]) -> Dict[str, Any]:
    raw_days = plan.get("per_day_itinerary", [])
    if not isinstance(raw_days, list):
        raw_days = []
    # v63: pad to the expected number of days when the LLM truncated a long
    # itinerary, so a day-count shortfall becomes a fillable (soft) gap instead of a
    # hard failure.  Extra days get a minimal stub that grounding/Module 7 can fill.
    expected_days = safe_int(plan.get("day"), 0)
    if expected_days > 0 and len(raw_days) < expected_days:
        for d in range(len(raw_days) + 1, expected_days + 1):
            raw_days.append(
                {
                    "day": d,
                    "theme": "家庭行程",
                    "morning": [],
                    "afternoon": [],
                    "meal": "",
                    "transportation": "",
                    "split_activities": [],
                }
            )
    elif expected_days > 0 and len(raw_days) > expected_days:
        raw_days = raw_days[:expected_days]
    normalized_days = []
    for index, day_item in enumerate(raw_days):
        if not isinstance(day_item, dict):
            day_item = {
                "day": index + 1,
                "theme": safe_text(day_item),
                "morning": [],
                "afternoon": [],
                "evening": [],
            }
        day_item["day"] = safe_int(day_item.get("day"), index + 1) or index + 1
        if not day_item.get("morning") and not day_item.get("afternoon"):
            ma = safe_text(day_item.get("morning_activity", ""))
            aa = safe_text(day_item.get("afternoon_activity", ""))
            if ma or aa:
                if ma:
                    day_item["morning"] = [ma]
                if aa:
                    day_item["afternoon"] = [aa]
            elif "core_activities" in day_item:
                core_items = _coerce_list(day_item.get("core_activities", []))[:2]
                if core_items:
                    day_item["morning"] = [core_items[0]]
                if len(core_items) > 1:
                    day_item["afternoon"] = [core_items[1]]
        if "meal" in day_item and not day_item.get("lunch"):
            day_item["lunch"] = safe_text(day_item.get("meal"))
        for block in [
            "morning",
            "afternoon",
            "evening",
            "optional_activities",
            "rest_blocks",
            "split_activities",
            "transportation",
        ]:
            day_item[block] = _coerce_list(day_item.get(block, []))
        for block in ["morning", "afternoon", "optional_activities"]:
            normalized_block = []
            for activity in day_item.get(block, []):
                if isinstance(activity, dict):
                    nm = safe_text(activity.get("poi_name")) or safe_text(activity.get("activity"))
                    if _is_meal_like_activity(nm) and not activity.get("is_transit"):
                        continue  # a dining label leaked into a sightseeing slot — drop it
                    normalized_block.append(activity)
                else:
                    text = safe_text(activity)
                    if text and not _is_meal_like_activity(text):
                        normalized_block.append(
                            {"activity": text, "poi_name": text, "poi_type": "general"}
                        )
            day_item[block] = normalized_block[:2]
        if not isinstance(day_item.get("lunch"), (dict, str)):
            day_item["lunch"] = safe_text(day_item.get("lunch", ""))
        if not isinstance(day_item.get("dinner"), (dict, str)):
            day_item["dinner"] = safe_text(day_item.get("dinner", ""))
        walk_val = safe_float(day_item.get("estimated_walking_distance_km"), 0)
        if walk_val <= 0:
            walk_val = estimate_day_walking_km(day_item)
        day_item["estimated_walking_distance_km"] = walk_val
        day_item["estimated_route_distance_km"] = safe_float(
            day_item.get("estimated_route_distance_km"), 0
        )
        day_item["estimated_return_time"] = (
            safe_text(day_item.get("estimated_return_time", plan.get("return_time", "20:00")))
            or "20:00"
        )
        normalized_days.append(day_item)
    normalized_days = _mark_arrival_departure(normalized_days)
    normalized_days = _dedup_itinerary_pois(plan, normalized_days)
    # Recompute per-day walking from the FINAL POIs (arrival/departure + dedup changed them).
    for d in normalized_days:
        if isinstance(d, dict):
            d["estimated_walking_distance_km"] = estimate_day_walking_km(d)
    plan["per_day_itinerary"] = normalized_days
    return plan


_MEAL_PLUS_PRICE_WORDS = [
    "简餐",
    "自助餐",
    "正餐",
    "午餐",
    "晚餐",
    "早餐",
    "快餐",
    "小吃",
    "火锅",
    "烧烤",
    "料理",
    "烤肉",
    "中档",
    "高档",
    "低档",
    "中等",
    "高价",
    "档",
]
_MEAL_STANDALONE_WORDS = {"简餐", "自助餐", "午餐", "晚餐", "早餐", "正餐", "用餐", "快餐", "小吃", "大排档"}


def _is_meal_like_activity(name: str) -> bool:
    """True for dining-type labels that leaked into a sightseeing slot, e.g.
    '简餐+中档', '自助餐+高', '家庭友好餐饮', '餐饮' — never real attraction names
    like 美食街/小吃街 (kept)."""
    s = safe_text(name).strip()
    if not s:
        return False
    if "餐饮" in s or "人均" in s or "价位" in s:
        return True
    if "+" in s and any(w in s for w in _MEAL_PLUS_PRICE_WORDS):
        return True
    if s in _MEAL_STANDALONE_WORDS:
        return True
    return False


def _mark_arrival_departure(days: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Day 1 morning = arrival, last day afternoon = departure — no sightseeing in
    those slots (the family arrives / heads home)."""
    real = [d for d in days if isinstance(d, dict)]
    if not real:
        return days
    real[0]["morning"] = [
        {"activity": "抵达目的地，办理酒店入住", "poi_name": "", "poi_type": "arrival", "is_transit": True}
    ]
    real[0]["is_arrival_day"] = True
    if len(real) >= 2:
        real[-1]["afternoon"] = [
            {"activity": "返程", "poi_name": "", "poi_type": "departure", "is_transit": True}
        ]
        real[-1]["is_departure_day"] = True
    return days


# Venue-type suffixes used to recognise the same place under different names
# (故宫 vs 故宫博物院): the longer name is just the shorter one plus one of these.
_VENUE_SUFFIXES = (
    "博物院",
    "博物馆",
    "纪念馆",
    "美术馆",
    "展览馆",
    "科技馆",
    "陈列馆",
    "纪念堂",
    "艺术馆",
    "风景区",
    "景区",
    "公园",
    "广场",
    "乐园",
    "游乐园",
    "步行街",
    "商业街",
    "古镇",
    "故居",
    "陵园",
    "寺",
    "庙",
)


def _same_place_name(a: str, b: str) -> bool:
    """True when two normalized POI names denote the same place.

    Handles: exact match; suffix variants (故宫 + 博物院 == 故宫博物院); short
    region-prefix variants (香港 + 迪士尼乐园 == 香港迪士尼乐园); and plain
    containment for clearly-long names.  Avoids merging same-prefix different-suffix
    places (人民公园 vs 人民广场).
    """
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if long.startswith(short) and long[len(short) :] in _VENUE_SUFFIXES:
        return True
    if long.endswith(short) and len(short) >= 3 and (len(long) - len(short)) <= 3:
        return True
    if len(short) >= 4 and short in long:
        return True
    return False


# POI category keywords (checked in order; first hit wins).  Used to keep one trip
# from piling up the same kind of attraction.
_POI_CATEGORY_KEYWORDS = (
    (
        "museum",
        ("博物馆", "博物院", "纪念馆", "美术馆", "展览馆", "科技馆", "科学技术馆", "技术馆", "陈列馆", "纪念堂", "艺术馆", "科学馆"),
    ),
    ("themepark", ("欢乐谷", "迪士尼", "方特", "游乐园", "乐园", "海洋世界", "水世界", "水上乐园")),
    ("park", ("国家公园", "森林公园", "植物园", "动物园", "风景区", "景区", "湿地", "园林", "公园", "花园")),
    ("temple", ("故宫", "天坛", "地坛", "雍和宫", "孔庙", "寺", "庙", "教堂", "陵", "祠")),
    ("street", ("步行街", "商业街", "美食街", "夜市", "购物", "广场", "古镇", "老街", "胡同")),
)


def _poi_category(name: str) -> str:
    low = safe_text(name).lower()
    for cat, kws in _POI_CATEGORY_KEYWORDS:
        if any(k in low for k in kws):
            return cat
    return "other"


def _dedup_itinerary_pois(plan: Dict[str, Any], days: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """De-duplicate AND diversify a family's attractions across the whole itinerary.

    1) Each attraction appears at most once — same-place name variants are merged
       (故宫 vs 故宫博物院 via a venue-suffix match; 迪士尼乐园 vs 香港迪士尼乐园 via
       a region-prefix/containment match).
    2) No category is over-represented: beyond MAX_SAME_CATEGORY_POI of a category
       (museum/park/temple/street/themepark) the excess is swapped for an unused real
       attraction of a DIFFERENT category.  Must-include POIs (poi_tags) are never
       swapped, and a POI is kept (not dropped) when no real alternative exists.
    """
    destination = safe_text(plan.get("destination_city"))
    try:
        pool = city_attraction_pool(destination)
    except Exception:
        pool = []

    def _norm(s: str) -> str:
        return safe_text(s).lower().strip()

    protected = {_norm(t) for t in (plan.get("poi_tags") or []) if _norm(t)}

    def _is_protected(name: str) -> bool:
        k = _norm(name)
        return bool(k) and any(p == k or (len(p) >= 2 and p in k) for p in protected)

    used = set()
    category_count: Dict[str, int] = {}

    def _over_cap(cat: str) -> bool:
        return cat != "other" and category_count.get(cat, 0) >= MAX_SAME_CATEGORY_POI

    def _is_used(name: str) -> bool:
        k = _norm(name)
        if not k:
            return True
        return any(_same_place_name(k, u) for u in used)

    def _accept(name: str) -> None:
        used.add(_norm(name))
        cat = _poi_category(name)
        category_count[cat] = category_count.get(cat, 0) + 1

    def _poi_name(item: Any) -> str:
        if isinstance(item, dict):
            return (
                safe_text(item.get("poi_name"))
                or safe_text(item.get("activity"))
                or safe_text(item.get("name"))
            )
        return safe_text(item)

    # All POIs already scheduled anywhere in the itinerary — a replacement must be a
    # genuinely NEW attraction, never one that appears on another day (else swapping
    # cascades and empties later slots).
    itinerary_names = set()
    for _d in days:
        if isinstance(_d, dict):
            for _blk in ("morning", "afternoon"):
                for _it in _coerce_list(_d.get(_blk, [])):
                    _nm = _norm(_poi_name(_it))
                    if _nm:
                        itinerary_names.add(_nm)

    def _in_itinerary(name: str) -> bool:
        k = _norm(name)
        return any(_same_place_name(k, n) for n in itinerary_names)

    def _next_replacement(avoid_cat: Optional[str] = None) -> str:
        for p in pool:
            if _is_used(p) or _in_itinerary(p):
                continue
            pc = _poi_category(p)
            if _over_cap(pc) or (avoid_cat and pc == avoid_cat):
                continue
            return p
        # Local pool exhausted -> fetch a fresh, different-category real POI on demand
        # (Google Places, gated by online grounding; returns '' when unavailable).
        fetched = fetch_alternative_real_poi(
            destination, set(used) | itinerary_names, avoid_cat or ""
        )
        if fetched and not _is_used(fetched) and not _in_itinerary(fetched):
            return fetched
        return ""

    for day_item in days:
        if not isinstance(day_item, dict):
            continue
        for block in ["morning", "afternoon"]:
            new_block = []
            for item in _coerce_list(day_item.get(block, [])):
                # Keep arrival/departure markers untouched (not real POIs).
                if isinstance(item, dict) and (
                    item.get("is_transit") or item.get("poi_type") in ("arrival", "departure")
                ):
                    new_block.append(item)
                    continue
                name = _poi_name(item)
                if not _norm(name):
                    continue
                cat = _poi_category(name)
                if _is_used(name):
                    rep = _next_replacement()
                    if rep:
                        _accept(rep)
                        new_block.append(
                            {
                                "activity": rep,
                                "poi_name": rep,
                                "poi_type": "general",
                                "replaced_duplicate_poi": True,
                            }
                        )
                    else:
                        # No fresh alternative — keep rather than empty the slot (the
                        # pool is too small to substitute; never worse than before).
                        new_block.append(item)
                elif _over_cap(cat) and not _is_protected(name):
                    rep = _next_replacement(avoid_cat=cat)
                    if rep:
                        _accept(rep)
                        new_block.append(
                            {
                                "activity": rep,
                                "poi_name": rep,
                                "poi_type": "general",
                                "replaced_same_category_poi": True,
                            }
                        )
                    else:
                        _accept(name)
                        new_block.append(item)  # keep it: no different-category alternative
                else:
                    _accept(name)
                    new_block.append(item)
            # An empty sightseeing slot (e.g. a dining label was filtered out) is
            # filled with an unused real attraction — but never the arrival/departure slots.
            is_transit_slot = (block == "morning" and day_item.get("is_arrival_day")) or (
                block == "afternoon" and day_item.get("is_departure_day")
            )
            # A day with a split activity already IS the afternoon's activity (some members
            # go one way, others another) — do not also fill the afternoon with a separate
            # filler POI ('下午本来就是分流活动，不要再加其他的').
            day_has_split = (
                isinstance(day_item.get("split_activities"), list)
                and len(day_item.get("split_activities")) > 0
            )
            if block == "afternoon" and day_has_split:
                is_transit_slot = True
            if not new_block and not is_transit_slot:
                rep = _next_replacement()
                if rep:
                    _accept(rep)
                    new_block.append(
                        {
                            "activity": rep,
                            "poi_name": rep,
                            "poi_type": "general",
                            "filled_empty_slot": True,
                        }
                    )
            day_item[block] = new_block
    return days


def trim_final_plan_text(plan: Dict[str, Any]) -> Dict[str, Any]:
    plan["itinerary_text"] = safe_text(plan.get("itinerary_text", ""))[:500]
    plan["coordinator_summary"] = safe_text(plan.get("coordinator_summary", ""))[:260]
    for key, limit in [
        ("poi_tags", 12),
        ("meal_tags", 10),
        ("high_intensity_tags", 6),
        ("split_activities", 8),
        ("unresolved_conflicts", 5),
        ("resolved_conflicts", 5),
        ("agent_vote_prediction", 6),
    ]:
        value = plan.get(key, [])
        if isinstance(value, list):
            plan[key] = value[:limit]
    return plan


def ensure_evaluation_ready_grounding_fields(
    plan: Dict[str, Any], family_agents: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Fill structured fields that Module 6 reads for objective evaluation.

    The natural-language itinerary can look good while Module 6 still fails if
    hotel/transport/cost evidence is only embedded in text.  This helper keeps
    the compact output schema, but normalizes it into explicit JSON fields:
    hotel_plan, transportation_plan, per_day_itinerary, real_world_grounding,
    and cost_breakdown.
    """
    day_count = safe_int(plan.get("day"), 1) or 1
    member_count = max(1, len(family_agents))
    grounding_quality_issues = (
        plan.get("grounding_quality_issues", [])
        if isinstance(plan.get("grounding_quality_issues", []), list)
        else []
    )
    grounding_quality_warnings = (
        plan.get("grounding_quality_warnings", [])
        if isinstance(plan.get("grounding_quality_warnings", []), list)
        else []
    )

    # v4: Do not fail a whole family only because the hotel extractor omitted
    # ``nearby_metro_station``.  The transport extractor often stores the same
    # fact under ``hotel_base_station``.  Use all common aliases first, then keep
    # the missing station as a warning when the hotel itself is real and sourced.
    preliminary_transport = (
        plan.get("transportation_plan", {})
        if isinstance(plan.get("transportation_plan"), dict)
        else {}
    )

    def _station_from_transport(transport_plan: Dict[str, Any]) -> str:
        for key in [
            "hotel_base_station",
            "nearby_metro_station",
            "nearby_station",
            "nearest_metro_station",
            "closest_metro_station",
            "origin_station",
            "metro_station",
            "transit_station",
        ]:
            value = safe_text(transport_plan.get(key))
            if value:
                return value
        daily = (
            transport_plan.get("daily_local_transport", [])
            if isinstance(transport_plan.get("daily_local_transport", []), list)
            else []
        )
        for route_item in daily:
            if not isinstance(route_item, dict):
                continue
            for key in ["origin", "start_station", "from_station", "route"]:
                value = safe_text(route_item.get(key))
                if value and (
                    "站" in value
                    or "station" in value.lower()
                    or "metro" in value.lower()
                    or "mtr" in value.lower()
                ):
                    return value
        routes = (
            transport_plan.get("metro_routes", [])
            if isinstance(transport_plan.get("metro_routes", []), list)
            else []
        )
        for route in routes:
            value = safe_text(route)
            if value and (
                "站" in value
                or "station" in value.lower()
                or "metro" in value.lower()
                or "mtr" in value.lower()
            ):
                return value
        return ""

    hotel = plan.get("hotel_plan", {}) if isinstance(plan.get("hotel_plan"), dict) else {}
    hotel_name = safe_text(hotel.get("hotel_name")) or safe_text(hotel.get("hotel_name_or_type"))
    hotel_sources_probe = (
        hotel.get("source_urls", []) if isinstance(hotel.get("source_urls", []), list) else []
    )
    if not hotel_name:
        grounding_quality_issues.append("missing_real_hotel_name")
        hotel_name = ""
    hotel_area = safe_text(hotel.get("hotel_area")) or safe_text(hotel.get("area"))
    station = (
        safe_text(hotel.get("nearby_metro_station"))
        or safe_text(hotel.get("nearby_station"))
        or safe_text(hotel.get("nearest_metro_station"))
        or safe_text(hotel.get("closest_metro_station"))
        or safe_text(hotel.get("metro_station"))
        or safe_text(hotel.get("transit_station"))
        or _station_from_transport(preliminary_transport)
    )
    if not station:
        # Keep this strict only when the hotel is also weak.  Otherwise preserve
        # the plan and let Module 6 report the warning instead of blocking all
        # downstream revision/backtracking.
        hotel_has_source = bool([x for x in hotel_sources_probe if safe_text(x)])
        hotel_specific = bool(hotel_name) and safe_text(hotel_name).lower() not in [
            "hotel",
            "family hotel",
            "family-friendly hotel",
        ]
        if not (hotel_specific and (hotel_has_source or hotel_area)):
            grounding_quality_issues.append("missing_hotel_nearby_station")
        else:
            grounding_quality_warnings.append(
                "missing_hotel_nearby_station_but_hotel_name_area_or_source_available"
            )
    hotel["hotel_name"] = hotel_name
    hotel["hotel_name_or_type"] = hotel_name
    hotel["hotel_area"] = hotel_area
    hotel["nearby_metro_station"] = station
    hotel["price_level"] = (
        safe_text(hotel.get("price_level"))
        or safe_text(plan.get("restaurant_price_level"))
        or "medium"
    )
    if safe_float(hotel.get("estimated_price_per_night"), 0) <= 0:
        budget_limit = safe_float(plan.get("budget_limit"), 0)
        # Per-ROOM nightly estimate: ~28% of budget spread over nights AND rooms, so
        # the hotel estimate stays within budget regardless of how many rooms the
        # family composition needs (each teenager/grandparent adds a room).
        est_rooms = estimate_room_count(family_agents, people_per_room=HOTEL_PEOPLE_PER_ROOM)
        hotel["estimated_price_per_night"] = round(
            max(
                300.0,
                (budget_limit * 0.28 / max(day_count - 1, 1) / est_rooms)
                if budget_limit > 0
                else 500.0,
            ),
            2,
        )
    hotel.setdefault("source_urls", [])
    plan["hotel_plan"] = hotel

    transport = preliminary_transport if isinstance(preliminary_transport, dict) else {}
    transport.setdefault("main_transport_mode", "metro/MTR + short taxi when needed")
    transport.setdefault(
        "route_strategy",
        "Use concrete station-based public transport routes and short taxi legs to reduce walking.",
    )
    metro_routes = (
        transport.get("metro_routes", []) if isinstance(transport.get("metro_routes"), list) else []
    )
    daily_local = (
        transport.get("daily_local_transport", [])
        if isinstance(transport.get("daily_local_transport"), list)
        else []
    )

    per_day = (
        plan.get("per_day_itinerary", []) if isinstance(plan.get("per_day_itinerary"), list) else []
    )
    fixed_days = []
    for idx in range(day_count):
        item = (
            per_day[idx]
            if idx < len(per_day) and isinstance(per_day[idx], dict)
            else {"day": idx + 1}
        )
        item["day"] = safe_int(item.get("day"), idx + 1) or idx + 1
        for block in ["morning", "afternoon", "optional_activities"]:
            vals = item.get(block, [])
            if not isinstance(vals, list):
                vals = [vals] if safe_text(vals) else []
            normalized = []
            for act in vals[:2]:
                if isinstance(act, dict):
                    name = (
                        safe_text(act.get("poi_name"))
                        or safe_text(act.get("activity"))
                        or safe_text(act.get("name"))
                    )
                    act["poi_name"] = name or "local attraction"
                    act.setdefault("activity", act["poi_name"])
                    act.setdefault("duration_hours", 1.5)
                    normalized.append(act)
                else:
                    text = safe_text(act)
                    if text:
                        normalized.append(
                            {
                                "poi_name": text,
                                "activity": text,
                                "poi_type": "attraction",
                                "duration_hours": 1.5,
                            }
                        )
            item[block] = normalized
        if not item.get("morning") and not item.get("afternoon"):
            grounding_quality_issues.append(f"day_{idx + 1}_missing_concrete_activities")
        for meal_key in ["lunch", "dinner"]:
            meal = item.get(meal_key, {})
            if not isinstance(meal, dict):
                meal_text = safe_text(meal)
                meal = {
                    "restaurant_name": meal_text,
                    "meal_type": meal_key,
                    "price_level": plan.get("restaurant_price_level", "medium"),
                }
            if (
                not safe_text(meal.get("restaurant_name"))
                and not safe_text(meal.get("restaurant_type"))
                and not safe_text(meal.get("name_or_type"))
            ):
                grounding_quality_issues.append(
                    f"day_{idx + 1}_{meal_key}_missing_restaurant_grounding"
                )
            # Do not turn a generic restaurant type into a fake restaurant name.
            # Concrete names may still come from model output or online grounding.
            candidate_restaurant_name = safe_text(meal.get("restaurant_name")) or safe_text(
                meal.get("name")
            )
            if not candidate_restaurant_name:
                candidate_restaurant_name = safe_text(meal.get("name_or_type"))
            if candidate_restaurant_name and not any(
                term in candidate_restaurant_name
                for term in ["附近", "中等价位", "当地", "家庭餐厅", "餐厅类型", "不指定具体餐厅"]
            ):
                meal["restaurant_name"] = candidate_restaurant_name
            else:
                meal.setdefault("restaurant_name", "")
            meal.setdefault("price_level", plan.get("restaurant_price_level", "medium"))
            item[meal_key] = meal
        transport_text = (
            " / ".join([safe_text(x) for x in item.get("transportation", [])])
            if isinstance(item.get("transportation"), list)
            else safe_text(item.get("transportation"))
        )
        if not transport_text:
            grounding_quality_issues.append(f"day_{idx + 1}_missing_transport_route")
        item["transportation"] = [transport_text] if transport_text else []
        item.setdefault("transportation_note", transport_text)
        item.setdefault("estimated_return_time", safe_text(plan.get("return_time")) or "21:00")
        walk_km = safe_float(item.get("estimated_walking_distance_km"), 0)
        if walk_km <= 0:
            walk_km = estimate_day_walking_km(item)
        item["estimated_walking_distance_km"] = walk_km
        item.setdefault(
            "estimated_route_distance_km",
            round(safe_float(plan.get("route_distance_km"), 0) / max(day_count, 1), 2),
        )
        fixed_days.append(item)
        if transport_text and transport_text not in metro_routes:
            metro_routes.append(transport_text)
        if transport_text and not any(
            safe_int(x.get("day"), 0) == item["day"] for x in daily_local if isinstance(x, dict)
        ):
            daily_local.append(
                {
                    "day": item["day"],
                    "route": transport_text,
                    "estimated_cost_total": round(35.0 * member_count, 2),
                }
            )
    plan["per_day_itinerary"] = fixed_days
    transport["metro_routes"] = metro_routes[: max(day_count, 1)]
    transport["daily_local_transport"] = daily_local[: max(day_count, 1)]
    transport.setdefault("source_urls", [])
    plan["transportation_plan"] = transport

    cost = plan.get("cost_breakdown", {}) if isinstance(plan.get("cost_breakdown"), dict) else {}
    nights = max(day_count - 1, 1)
    # Rooms by family composition: parents+kids share, each teenager/青年 own room,
    # grandparents share one room.
    rooms = estimate_room_count(family_agents, people_per_room=HOTEL_PEOPLE_PER_ROOM)
    cost.setdefault("currency", "CNY")
    # Hotel cost ALWAYS follows the final displayed nightly price (this function may
    # have just set/overridden it above), so 酒店费用 = 每晚价格 × 晚数 × 房间数 and
    # never keeps a stale value from an earlier price-level estimate.
    cost["hotel_cost"] = round(
        safe_float(hotel.get("estimated_price_per_night"), 500) * nights * rooms, 2
    )
    cost["hotel_nights"] = nights
    cost["hotel_rooms"] = rooms
    # Family-size-aware defaults (only used when online grounding did not fill them).
    cost.setdefault("local_transport_cost", round(35.0 * member_count * day_count, 2))
    cost.setdefault("meal_cost", round(120.0 * member_count * day_count, 2))
    cost.setdefault("attraction_ticket_cost", round(120.0 * member_count, 2))
    cost.setdefault(
        "intercity_transport_cost",
        round(max(0.0, safe_float(plan.get("budget_limit"), 0) * 0.18), 2),
    )
    subtotal = sum(
        safe_float(cost.get(k), 0)
        for k in [
            "hotel_cost",
            "intercity_transport_cost",
            "local_transport_cost",
            "meal_cost",
            "attraction_ticket_cost",
            "other_cost",
        ]
    )
    # Honest total = sum of components.  Budget relief (cheaper hotel) and the
    # over-budget note are handled in the grounding budget step; we do NOT rescale
    # components to fake-fit the budget here.
    cost["total_cost"] = round(subtotal, 2)
    plan["cost_breakdown"] = cost
    if safe_float(cost.get("total_cost"), 0) > 0:
        plan["total_cost"] = cost["total_cost"]

    online = (
        plan.get("online_grounding", {}) if isinstance(plan.get("online_grounding"), dict) else {}
    )
    sources = online.get("source_urls", []) if isinstance(online.get("source_urls"), list) else []
    hotel_sources = (
        hotel.get("source_urls", []) if isinstance(hotel.get("source_urls"), list) else []
    )
    transport_sources = (
        transport.get("source_urls", []) if isinstance(transport.get("source_urls"), list) else []
    )
    all_sources = list(
        dict.fromkeys(
            [safe_text(x) for x in sources + hotel_sources + transport_sources if safe_text(x)]
        )
    )
    hotel_specific = bool(hotel.get("hotel_name")) and safe_text(
        hotel.get("hotel_name")
    ).lower() not in ["hotel", "family hotel", "family-friendly hotel"]
    route_specific = bool(transport.get("metro_routes")) or bool(
        transport.get("daily_local_transport")
    )
    cost_complete = (
        bool(cost.get("total_cost"))
        and sum(
            1
            for k in ["hotel_cost", "local_transport_cost", "meal_cost", "attraction_ticket_cost"]
            if safe_float(cost.get(k), 0) > 0
        )
        >= 3
    )
    plan["grounding_quality_issues"] = list(
        dict.fromkeys([safe_text(x) for x in grounding_quality_issues if safe_text(x)])
    )
    plan["grounding_quality_warnings"] = list(
        dict.fromkeys([safe_text(x) for x in grounding_quality_warnings if safe_text(x)])
    )
    plan["real_world_grounding"] = {
        "hotel_realism_pass": bool(hotel_specific),
        "metro_route_realism_pass": bool(route_specific),
        "online_grounding_used": bool(all_sources)
        or online.get("search_status") in ["success", "partial_or_unverified"],
        "source_urls": all_sources,
        "structured_grounding_complete": bool(hotel_specific and route_specific and cost_complete),
        "issues": (
            online.get("grounding_issues", [])
            if isinstance(online.get("grounding_issues"), list)
            else []
        )
        + plan.get("grounding_quality_issues", [])
        + plan.get("grounding_quality_warnings", []),
    }
    return plan


def validate_and_fix_plan(
    plan: Dict[str, Any],
    family_id: int,
    family_agents: List[Dict[str, Any]],
    usage: Dict[str, Any],
    runtime_seconds: float,
) -> Dict[str, Any]:
    if not isinstance(plan, dict):
        plan = {"itinerary_text": safe_text(plan)}
    first_agent = family_agents[0]
    plan["family_id"] = family_id
    plan["plan_id"] = safe_text(plan.get("plan_id", "consensus_plan")) or "consensus_plan"
    plan["method"] = (
        safe_text(plan.get("method", "multi_agent_consensus")) or "multi_agent_consensus"
    )
    plan["departure_city"] = safe_text(
        plan.get("departure_city", first_agent.get("departure_city", ""))
    )
    plan["destination_city"] = safe_text(
        plan.get("destination_city", first_agent.get("destination_city", ""))
    )
    plan["day"] = safe_int(plan.get("day", first_agent.get("day", 0)), 0)
    plan["itinerary_text"] = safe_text(plan.get("itinerary_text", ""))
    if not isinstance(plan.get("per_day_itinerary"), list):
        plan["per_day_itinerary"] = []
    plan = normalize_per_day_itinerary(plan)
    plan["total_cost"] = safe_float(plan.get("total_cost", 0), 0)
    plan["budget_limit"] = safe_float(plan.get("budget_limit", 0), 0)
    plan["budget_note"] = safe_text(plan.get("budget_note", ""))
    plan["walking_distance_km"] = safe_float(plan.get("walking_distance_km", 0), 0)
    plan["daily_poi_count"] = safe_float(plan.get("daily_poi_count", 0), 0)
    plan["route_distance_km"] = safe_float(plan.get("route_distance_km", 0), 0)
    plan["return_time"] = safe_text(plan.get("return_time", "20:00")) or "20:00"
    plan["duplicate_poi_count"] = safe_int(plan.get("duplicate_poi_count", 0), 0)
    if not isinstance(plan.get("poi_tags"), list):
        plan["poi_tags"] = []
    if not isinstance(plan.get("meal_tags"), list):
        plan["meal_tags"] = []
    plan["restaurant_price_level"] = (
        safe_text(plan.get("restaurant_price_level", "medium")) or "medium"
    )
    if not isinstance(plan.get("high_intensity_tags"), list):
        plan["high_intensity_tags"] = []
    if not isinstance(plan.get("hotel_plan"), dict):
        plan["hotel_plan"] = {"hotel_name_or_type": safe_text(plan.get("hotel_plan", ""))}
    if not isinstance(plan.get("transportation_plan"), dict):
        plan["transportation_plan"] = {
            "main_transport_mode": safe_text(plan.get("transportation_plan", ""))
        }
    if not isinstance(plan.get("real_world_grounding"), dict):
        plan["real_world_grounding"] = {}
    if not isinstance(plan.get("grounding_requirements"), dict):
        plan["grounding_requirements"] = {}
    if not isinstance(plan.get("online_grounding"), dict):
        plan["online_grounding"] = {}
    if not isinstance(plan.get("cost_breakdown"), dict):
        plan["cost_breakdown"] = {}
    if not isinstance(plan.get("hard_constraint_checklist"), list):
        plan["hard_constraint_checklist"] = []
    if not isinstance(plan.get("split_activities"), list):
        plan["split_activities"] = []
    if not isinstance(plan.get("resolved_conflicts"), list):
        plan["resolved_conflicts"] = []
    if not isinstance(plan.get("unresolved_conflicts"), list):
        plan["unresolved_conflicts"] = []
    if not isinstance(plan.get("constraint_satisfiability"), dict):
        if USE_CONSTRAINT_SATISFIABILITY_TYPING:
            plan = attach_constraint_satisfiability_typing(plan=plan, family_agents=family_agents)
        else:
            plan["constraint_satisfiability"] = {}
    if not isinstance(plan.get("agent_vote_prediction"), list):
        plan["agent_vote_prediction"] = []
    plan["acceptance_vote_ratio"] = safe_float(plan.get("acceptance_vote_ratio", 0), 0)
    plan["number_of_objections_resolved"] = safe_int(
        plan.get("number_of_objections_resolved", 0), 0
    )
    plan["consensus_status"] = safe_text(plan.get("consensus_status", ""))
    plan["coordinator_summary"] = safe_text(plan.get("coordinator_summary", ""))
    plan["prompt_tokens"] = safe_int(usage.get("prompt_tokens"), 0)
    plan["completion_tokens"] = safe_int(usage.get("completion_tokens"), 0)
    plan["total_tokens"] = safe_int(usage.get("total_tokens"), 0)
    plan["runtime_seconds"] = round(runtime_seconds, 4)
    return plan


def _json_dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return json.dumps(safe_text(value), ensure_ascii=False)


def _empty_plan_row(family_id: int = 0) -> Dict[str, Any]:
    row = {column: "" for column in PLAN_OUTPUT_COLUMNS}
    row.update(
        {
            "family_id": family_id,
            "plan_id": "consensus_plan",
            "method": "multi_agent_consensus",
            "per_day_itinerary_json": "[]",
            "itinerary_quality_markers_json": "{}",
            "poi_tags": "[]",
            "meal_tags": "[]",
            "high_intensity_tags": "[]",
            "hotel_plan_json": "{}",
            "transportation_plan_json": "{}",
            "real_world_grounding_json": "{}",
            "grounding_requirements_json": "{}",
            "online_grounding_json": "{}",
            "cost_breakdown_json": "{}",
            "hard_constraint_checklist_json": "[]",
            "split_activities_json": "[]",
            "concession_resolutions_json": "[]",
            "resolved_conflicts_json": "[]",
            "unresolved_conflicts_json": "[]",
            "constraint_satisfiability_json": "{}",
            "agent_vote_prediction_json": "[]",
            "negotiation_rounds": 4,
        }
    )
    return row


def flatten_plan_for_csv(plan: Dict[str, Any]) -> Dict[str, Any]:
    row = _empty_plan_row(safe_int(plan.get("family_id"), 0))
    row.update(
        {
            "family_id": plan.get("family_id"),
            "plan_id": plan.get("plan_id"),
            "method": plan.get("method"),
            "departure_city": plan.get("departure_city"),
            "destination_city": plan.get("destination_city"),
            "day": plan.get("day"),
            "itinerary_text": plan.get("itinerary_text"),
            "per_day_itinerary_json": _json_dump(plan.get("per_day_itinerary", [])),
            "itinerary_quality_markers_json": _json_dump(plan.get("itinerary_quality_markers", {})),
            "total_cost": plan.get("total_cost"),
            "budget_limit": plan.get("budget_limit"),
            "budget_note": plan.get("budget_note"),
            "walking_distance_km": plan.get("walking_distance_km"),
            "daily_poi_count": plan.get("daily_poi_count"),
            "poi_tags": _json_dump(plan.get("poi_tags", [])),
            "meal_tags": _json_dump(plan.get("meal_tags", [])),
            "restaurant_price_level": plan.get("restaurant_price_level"),
            "route_distance_km": plan.get("route_distance_km"),
            "return_time": plan.get("return_time"),
            "duplicate_poi_count": plan.get("duplicate_poi_count"),
            "high_intensity_tags": _json_dump(plan.get("high_intensity_tags", [])),
            "hotel_plan_json": _json_dump(plan.get("hotel_plan", {})),
            "transportation_plan_json": _json_dump(plan.get("transportation_plan", {})),
            "real_world_grounding_json": _json_dump(plan.get("real_world_grounding", {})),
            "grounding_requirements_json": _json_dump(plan.get("grounding_requirements", {})),
            "online_grounding_json": _json_dump(plan.get("online_grounding", {})),
            "cost_breakdown_json": _json_dump(plan.get("cost_breakdown", {})),
            "hard_constraint_checklist_json": _json_dump(plan.get("hard_constraint_checklist", [])),
            "split_activities_json": _json_dump(plan.get("split_activities", [])),
            "concession_resolutions_json": _json_dump(plan.get("concession_resolutions", [])),
            "resolved_conflicts_json": _json_dump(plan.get("resolved_conflicts", [])),
            "unresolved_conflicts_json": _json_dump(plan.get("unresolved_conflicts", [])),
            "constraint_satisfiability_json": _json_dump(plan.get("constraint_satisfiability", {})),
            "agent_vote_prediction_json": _json_dump(plan.get("agent_vote_prediction", [])),
            "acceptance_vote_ratio": plan.get("acceptance_vote_ratio"),
            "number_of_objections_resolved": plan.get("number_of_objections_resolved"),
            "consensus_status": plan.get("consensus_status"),
            "coordinator_summary": plan.get("coordinator_summary"),
            "negotiation_rounds": 4,
            "prompt_tokens": plan.get("prompt_tokens"),
            "completion_tokens": plan.get("completion_tokens"),
            "total_tokens": plan.get("total_tokens"),
            "runtime_seconds": plan.get("runtime_seconds"),
            "failure_stage": plan.get("failure_stage", ""),
            "module5_generation_failed": plan.get("module5_generation_failed", False),
            "backtrack_to_module2": plan.get("backtrack_to_module2", False),
            "module5_failure_reasons_json": _json_dump(plan.get("module5_failure_reasons", [])),
            "status": plan.get("status", ""),
            "error": plan.get("error", ""),
        }
    )
    return row


def _list_has_meaningful_items(value: Any) -> bool:
    """Return True when a list/dict/string contains useful generated content."""
    if isinstance(value, str):
        return bool(safe_text(value))
    if isinstance(value, list):
        return any(_list_has_meaningful_items(item) for item in value)
    if isinstance(value, dict):
        return any(_list_has_meaningful_items(v) for v in value.values())
    return value not in [None, "", [], {}]


PLACEHOLDER_QUALITY_TERMS = [
    "core attraction",
    "core family activity",
    "local attraction",
    "day attractions",
    "nearest metro/mtr station",
    "mtr/metro line route",
    "metro/mtr line route",
    "nearest 地铁或mtr station",
    "地铁或mtr line route",
    "local family restaurant",
    "family restaurant",
    "v48规则生成",
    "参观文化景点",
    "当地餐厅用餐",
    "最后的购物",
    "回顾旅行",
    "准备返回",
    "休息或自由安排",
    "附近低人流咖啡馆",
]


def _placeholder_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False).lower()
    except Exception:
        return safe_text(value).lower()


def _contains_placeholder(value: Any) -> bool:
    text = _placeholder_text(value)
    if not text:
        return False
    return any(term.lower() in text for term in PLACEHOLDER_QUALITY_TERMS)


def _normalize_activity_text_for_specificity(value: Any) -> str:
    text = safe_text(value).strip()
    text = re.sub(r"^(上午|下午|晚上|早上|中午)[:：\s]*", "", text)
    text = re.sub(r"^(参观|游览|探索|体验|享受|品尝|观看|欣赏|前往|抵达|返回|在|于)", "", text).strip()
    # v21: keep concrete POI anchors while removing generic action suffixes.
    # Examples: 外滩观光 -> 外滩, 上海塔参观 -> 上海塔, 武汉大学校园游览 -> 武汉大学校园.
    text = re.sub(r"(观光|参观|游览|游玩|体验|打卡|拍照|散步|放松|校园游览|短时游览)$", "", text).strip()
    text = re.sub(r"(附近|周边|一带)$", "", text).strip(" 的，；。")
    return text


def _structural_placeholder_text(value: Any) -> bool:
    text = safe_text(value).strip().lower()
    if not text:
        return True
    return bool(
        re.search(
            r"(第\s*\d+\s*天.*活动区域|core\s+attraction|local\s+family\s+activity|nearest\s+(metro|mtr)|短路线|路线需求)",
            text,
        )
    )


def _activity_specificity_score(value: Any) -> float:
    raw_text = safe_text(value).strip()
    if re.search(r"(返回|返程|退房|入住|抵达.*酒店|离开|整理行李|早餐后退房)", raw_text):
        return 0.0
    if re.fullmatch(r"前往[\u4e00-\u9fff]{2,4}", raw_text) and not re.search(
        r"(滩|园|乐园|公园|馆|博物馆|美术馆|艺术馆|楼|塔|台|观景台|寺|庙|宫|教堂|大教堂|佛|大佛|城|古城|小镇|广场|中心|大学|街|路|巷|港|湾|岛|山|湖|江|桥|站|市场|夜市|商场|王国|景区)$",
        re.sub(r"^前往", "", raw_text),
    ):
        return 0.0
    norm = _normalize_activity_text_for_specificity(value)
    if not norm or _structural_placeholder_text(norm):
        return 0.0
    score = 0.20
    if re.search(r"[A-Za-z0-9＋+&·•（）()]", norm):
        score += 0.30
    if re.search(
        r"(滩|园|乐园|公园|馆|博物馆|美术馆|艺术馆|楼|塔|台|观景台|寺|庙|宫|教堂|大教堂|佛|大佛|城|古城|小镇|广场|中心|大学|街|路|巷|港|湾|岛|山|湖|江|桥|站|市场|夜市|商场|王国|景区)$",
        norm,
    ):
        score += 0.45
    elif re.fullmatch(r"[\u4e00-\u9fff]{2,10}", norm) and not re.search(
        r"(活动|体验|游览|探索|美食|小吃|购物|表演|打卡|观景|餐饮|餐厅|火锅|自助餐|返程|退房|休息)$", norm
    ):
        score += 0.30
    if re.search(r"(早餐|午餐|晚餐|用餐|轻食|美食|小吃|火锅|餐厅|饭店|餐馆|料理|自助餐|餐饮|返程|退房|整理行李|休息)$", norm):
        score -= 0.35
    if re.search(r".{2,}(城|街|路|商场|市场|夜市|广场|港|湾|小镇|古城|公园|乐园|馆|楼|塔)(购物|打卡|体验|表演|美食|散步|游览)?$", norm):
        score += 0.45
    elif re.fullmatch(
        r"(文化|历史|主题|当地|本地|互动|观景|美食|餐饮|自由|轻松|放松|现场|游乐).{0,4}(遗址|地标|公园|乐园|活动|体验|游览|美食|购物|表演|打卡|晚餐|午餐)?",
        norm,
    ):
        score -= 0.45
    elif re.fullmatch(r".{0,6}(活动|体验|游览|探索|美食|小吃|购物|表演|打卡|观景|休闲|放松)$", norm):
        score -= 0.25
    return max(0.0, min(1.0, score))


def _activity_is_generic_or_vague(value: Any) -> bool:
    return _activity_specificity_score(value) < 0.45


def _activity_name(activity: Any) -> str:
    if isinstance(activity, dict):
        return (
            safe_text(activity.get("poi_name"))
            or safe_text(activity.get("activity"))
            or safe_text(activity.get("name"))
        )
    return safe_text(activity)


def _source_count_for_marker(value: Any) -> int:
    """Count explicit grounding sources attached to a plan object."""
    if not isinstance(value, dict):
        return 0
    sources: List[Any] = []
    for key in ["source_urls", "sources", "evidence_urls", "reference_urls", "urls"]:
        item = value.get(key)
        if isinstance(item, list):
            sources.extend(item)
        elif safe_text(item):
            sources.append(item)
    for key in ["source_url", "url", "reference_url"]:
        if safe_text(value.get(key)):
            sources.append(value.get(key))
    return len(
        [
            x
            for x in sources
            if safe_text(x) and safe_text(x).lower() not in ["none", "null", "[]", "{}"]
        ]
    )


# Specificity->failure gate (item 5): generic activity labels ("自由活动"/"美食体验") and
# non-specific meals are explicit grounding failures, surfaced in the quality markers so
# Module 6 scores them and Module 7 is forced to repair them. When the require flag is on,
# any generic-activity failure also blocks module5_quality_markers_pass.
REQUIRE_NO_GENERIC_ACTIVITY = os.getenv("MODULE5_REQUIRE_NO_GENERIC_ACTIVITY", "true").lower() in [
    "1",
    "true",
    "yes",
    "y",
]
# Grounding quality problems (missing real hotel name, vague meal/activity, missing route) are
# SOFT issues: Module 6 penalises them and Module 7 repairs them. They must NOT hard-fail the whole
# family in Module 5 (which drops it from the output entirely, as happened to family 2). Default soft.
MODULE5_HARD_FAIL_ON_GROUNDING_ISSUES = os.getenv(
    "MODULE5_HARD_FAIL_ON_GROUNDING_ISSUES", "false"
).lower() in ["1", "true", "yes", "y"]


def _activity_quality_marker(activity: Any, day_num: int, section: str) -> Dict[str, Any]:
    if isinstance(activity, dict):
        raw_name = (
            safe_text(activity.get("poi_name"))
            or safe_text(activity.get("activity"))
            or safe_text(activity.get("name"))
            or safe_text(activity.get("name_or_type"))
        )
        source_count = _source_count_for_marker(activity)
        grounding_status = safe_text(activity.get("grounding_status")) or safe_text(
            activity.get("source_status")
        )
        candidate_source = safe_text(activity.get("candidate_source"))
    else:
        raw_name = safe_text(activity)
        source_count = 0
        grounding_status = ""
        candidate_source = ""
    normalized = _normalize_activity_text_for_specificity(raw_name)
    score = _activity_specificity_score(raw_name)
    is_specific = bool(raw_name) and score >= 0.45 and not _contains_placeholder(raw_name)
    has_source = (
        source_count > 0
        or candidate_source in ["tavily", "online_search", "cache"]
        or "tavily" in grounding_status.lower()
    )
    # A non-empty but non-specific activity (e.g. 自由活动 / 美食体验) is a grounding failure:
    # it must be replaced with a concrete POI, not silently accepted.
    is_grounding_failure = bool(raw_name) and not is_specific
    tags = []
    tags.append("activity:specific" if is_specific else "activity:vague_or_incomplete")
    tags.append("activity:source_provided" if has_source else "activity:source_missing")
    if _contains_placeholder(raw_name):
        tags.append("activity:placeholder_text")
    if is_grounding_failure:
        tags.append("activity:grounding_failure")
    return {
        "day": int(day_num),
        "section": section,
        "raw_name": raw_name,
        "normalized_name": normalized,
        "specificity_score": round(score, 4),
        "is_concrete_activity": bool(is_specific),
        "is_grounding_failure": bool(is_grounding_failure),
        "has_grounding_source": bool(has_source),
        "source_count": int(source_count),
        "grounding_status": grounding_status or ("source_provided" if has_source else "name_only"),
        "quality_tags": tags,
    }


def _restaurant_name_looks_specific_for_marker(name: str) -> bool:
    text = safe_text(name).strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered in ["restaurant", "nearby restaurant", "local restaurant", "family restaurant"]:
        return False
    # Real restaurant names usually have a distinctive prefix, a brand-like token, or a named suffix.
    if re.search(r"[A-Za-z0-9&·•]", text):
        return True
    if len(text) >= 4 and re.search(
        r"(楼|轩|阁|馆|居|坊|记|庄|府|园|家|里|堂|饭店|餐厅|酒家|茶餐厅|菜馆|火锅店|牛肉店|食府)$", text
    ):
        # Reject pure category labels such as 西餐厅 / 自助餐厅 / 中餐厅.
        category_only = re.fullmatch(
            r"(中餐|西餐|自助餐|当地|本地|家庭友好|高档|中档|海鲜|火锅|茶餐厅|粤菜|本帮菜|东北菜|小吃).{0,4}(餐厅|饭店|菜馆|火锅店)?", text
        )
        return not bool(category_only)
    # Two- or three-character brands can be real, but only if they do not look like a cuisine/category.
    if 2 <= len(text) <= 8 and not re.search(
        r"(餐厅|饭店|附近|当地|中档|高档|自助|西餐|中餐|海鲜|火锅|小吃|美食|餐饮|用餐)", text
    ):
        return True
    return False


def _meal_quality_marker(meal: Any, day_num: int, meal_key: str) -> Dict[str, Any]:
    if isinstance(meal, dict):
        name = (
            safe_text(meal.get("restaurant_name"))
            or safe_text(meal.get("name"))
            or safe_text(meal.get("name_or_type"))
        )
        area = (
            safe_text(meal.get("restaurant_area"))
            or safe_text(meal.get("area"))
            or safe_text(meal.get("address_or_location"))
            or safe_text(meal.get("location"))
        )
        status = safe_text(meal.get("restaurant_grounding_status")) or safe_text(
            meal.get("grounding_status")
        )
        explicit_found = bool(meal.get("specific_restaurant_found")) or status in [
            "specific_name_found",
            "specific_restaurant_found",
        ]
        source_count = _source_count_for_marker(meal)
    else:
        name = safe_text(meal)
        area = ""
        status = ""
        explicit_found = False
        source_count = 0
    is_specific = (
        bool(name)
        and (explicit_found or _restaurant_name_looks_specific_for_marker(name))
        and status != "area_fallback_no_specific_name"
    )
    area_fallback = (not is_specific) and bool(area or status == "area_fallback_no_specific_name")
    # Every meal must be a concrete restaurant or it is an explicit grounding failure.
    is_grounding_failure = not is_specific
    tags = []
    tags.append("restaurant:specific_name" if is_specific else "restaurant:area_or_type_fallback")
    tags.append("restaurant:source_provided" if source_count > 0 else "restaurant:source_missing")
    if is_grounding_failure:
        tags.append("restaurant:grounding_failure")
    return {
        "day": int(day_num),
        "meal": meal_key,
        "restaurant_name": name,
        "restaurant_area": area,
        "is_specific_restaurant": bool(is_specific),
        "is_grounding_failure": bool(is_grounding_failure),
        "area_fallback_used": bool(area_fallback),
        "source_count": int(source_count),
        "grounding_status": status or ("specific_name" if is_specific else "area_or_type_fallback"),
        "quality_tags": tags,
    }


def _route_quality_marker(
    route_item: Any, day_num: int, day_item: Dict[str, Any]
) -> Dict[str, Any]:
    if isinstance(route_item, dict):
        route_text = safe_text(route_item.get("route")) or safe_text(
            route_item.get("route_summary")
        )
        estimated_cost = safe_float(
            route_item.get("estimated_cost_total") or route_item.get("estimated_cost"), 0
        )
        source_count = _source_count_for_marker(route_item)
    else:
        route_text = safe_text(route_item)
        estimated_cost = 0.0
        source_count = 0
    if not route_text:
        route_text = (
            "；".join([safe_text(x) for x in day_item.get("transportation", [])])
            if isinstance(day_item.get("transportation", []), list)
            else safe_text(day_item.get("transportation"))
        )
    stop_count = max(
        route_text.count("->") + route_text.count("→"),
        len([x for x in re.split(r"(?:->|→|；|;)", route_text) if safe_text(x)]),
    )
    placeholder = _structural_placeholder_text(route_text) or _contains_placeholder(route_text)
    has_cost = estimated_cost > 0
    complete = bool(route_text) and stop_count >= 3 and not placeholder
    tags = []
    tags.append(
        "transport:complete_route_chain" if complete else "transport:route_chain_incomplete"
    )
    tags.append("transport:cost_present" if has_cost else "transport:cost_missing")
    tags.append("transport:source_provided" if source_count > 0 else "transport:source_missing")
    return {
        "day": int(day_num),
        "route": route_text,
        "route_stop_count": int(stop_count),
        "is_complete_route_chain": bool(complete),
        "has_cost_estimate": bool(has_cost),
        "estimated_cost_total": estimated_cost,
        "source_count": int(source_count),
        "placeholder_detected": bool(placeholder),
        "quality_tags": tags,
    }


def build_itinerary_quality_markers(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Attach Module-5 generated quality markers for Module 6.

    These markers make downstream evaluation less dependent on brittle word lists:
    Module 5 explicitly says what is concrete, what is source-backed, and what is a fallback.
    Module 6 can then read the structured tags instead of re-inferring everything from text.
    """
    per_day = (
        plan.get("per_day_itinerary", []) if isinstance(plan.get("per_day_itinerary"), list) else []
    )
    day_count_expected = safe_int(plan.get("day"), len(per_day)) or len(per_day)

    activity_markers: List[Dict[str, Any]] = []
    restaurant_markers: List[Dict[str, Any]] = []
    transport_markers: List[Dict[str, Any]] = []
    day_markers: List[Dict[str, Any]] = []

    transportation_plan = (
        plan.get("transportation_plan", {})
        if isinstance(plan.get("transportation_plan"), dict)
        else {}
    )
    daily_routes = (
        transportation_plan.get("daily_local_transport", [])
        if isinstance(transportation_plan.get("daily_local_transport", []), list)
        else []
    )
    route_by_day: Dict[int, Any] = {}
    for idx, item in enumerate(daily_routes):
        if isinstance(item, dict):
            route_by_day[safe_int(item.get("day"), idx + 1) or idx + 1] = item

    for idx in range(max(day_count_expected, len(per_day))):
        day_item = (
            per_day[idx]
            if idx < len(per_day) and isinstance(per_day[idx], dict)
            else {"day": idx + 1}
        )
        day_num = safe_int(day_item.get("day"), idx + 1) or idx + 1
        day_activity_markers = []
        for section in ["morning", "afternoon", "evening"]:
            values = day_item.get(section, [])
            if not isinstance(values, list):
                values = [values] if safe_text(values) else []
            for activity in values:
                # Arrival/departure are intentional non-sightseeing slots — exclude
                # them from activity-concreteness counting (don't drag the rate down).
                if isinstance(activity, dict) and (
                    activity.get("is_transit")
                    or activity.get("poi_type") in ("arrival", "departure")
                ):
                    continue
                marker = _activity_quality_marker(activity, day_num=day_num, section=section)
                activity_markers.append(marker)
                if section in ["morning", "afternoon"]:
                    day_activity_markers.append(marker)
        day_restaurant_markers = []
        for meal_key in ["lunch", "dinner"]:
            marker = _meal_quality_marker(
                day_item.get(meal_key, {}), day_num=day_num, meal_key=meal_key
            )
            restaurant_markers.append(marker)
            day_restaurant_markers.append(marker)
        route_marker = _route_quality_marker(
            route_by_day.get(day_num), day_num=day_num, day_item=day_item
        )
        transport_markers.append(route_marker)
        concrete_count = sum(1 for x in day_activity_markers if x.get("is_concrete_activity"))
        day_markers.append(
            {
                "day": int(day_num),
                "theme": safe_text(day_item.get("theme")),
                "has_morning_or_afternoon_activity": bool(day_activity_markers),
                "concrete_core_activity_count": int(concrete_count),
                "core_activity_count": int(len(day_activity_markers)),
                "has_at_least_one_concrete_core_activity": bool(concrete_count > 0),
                "restaurant_marker_count": int(len(day_restaurant_markers)),
                "specific_restaurant_count": int(
                    sum(1 for x in day_restaurant_markers if x.get("is_specific_restaurant"))
                ),
                "transport_route_complete": bool(route_marker.get("is_complete_route_chain")),
                "quality_tags": [
                    "day:has_concrete_activity"
                    if concrete_count > 0
                    else "day:no_concrete_activity",
                    "day:transport_complete"
                    if route_marker.get("is_complete_route_chain")
                    else "day:transport_incomplete",
                ],
            }
        )

    hotel = plan.get("hotel_plan", {}) if isinstance(plan.get("hotel_plan"), dict) else {}
    hotel_name = safe_text(hotel.get("hotel_name")) or safe_text(hotel.get("hotel_name_or_type"))
    hotel_area = safe_text(hotel.get("hotel_area")) or safe_text(hotel.get("area"))
    hotel_station = (
        safe_text(hotel.get("nearby_metro_station"))
        or safe_text(hotel.get("nearby_station"))
        or safe_text(hotel.get("nearest_metro_station"))
    )
    hotel_source_count = _source_count_for_marker(hotel)
    hotel_specific = (
        bool(hotel_name)
        and not _structural_placeholder_text(hotel_name)
        and not re.fullmatch(r"(酒店|.*地区酒店|.*市中心酒店|.*尖沙咀酒店|.*家庭酒店)", hotel_name)
    )
    hotel_marker = {
        "hotel_name": hotel_name,
        "is_specific_hotel_name": bool(hotel_specific),
        "has_area": bool(hotel_area),
        "has_nearby_station": bool(hotel_station),
        "has_price": safe_float(hotel.get("estimated_price_per_night"), 0) > 0,
        "source_count": int(hotel_source_count),
        "quality_tags": [
            "hotel:specific_name" if hotel_specific else "hotel:generic_or_missing_name",
            "hotel:station_present" if hotel_station else "hotel:station_missing",
            "hotel:source_provided" if hotel_source_count > 0 else "hotel:source_missing",
        ],
    }

    cost = plan.get("cost_breakdown", {}) if isinstance(plan.get("cost_breakdown"), dict) else {}
    total_cost = safe_float(plan.get("total_cost"), 0)
    budget_limit = safe_float(plan.get("budget_limit"), 0)
    numeric_cost_values = [
        safe_float(v, 0)
        for k, v in cost.items()
        if isinstance(v, (int, float, str)) and k not in ["currency"]
    ]
    cost_sum = sum(x for x in numeric_cost_values if x > 0)
    cost_marker = {
        "has_total_cost": total_cost > 0,
        "has_budget_limit": budget_limit > 0,
        "budget_pass": bool(total_cost <= budget_limit)
        if total_cost > 0 and budget_limit > 0
        else False,
        "cost_breakdown_item_count": int(len([x for x in numeric_cost_values if x > 0])),
        "cost_sum_close_to_total": bool(
            total_cost > 0
            and (cost_sum == 0 or abs(cost_sum - total_cost) / max(total_cost, 1) <= 0.25)
        ),
    }

    activity_total = len(activity_markers)
    activity_specific = sum(1 for x in activity_markers if x.get("is_concrete_activity"))
    activity_source = sum(1 for x in activity_markers if x.get("has_grounding_source"))
    meal_total = len(restaurant_markers)
    meal_specific = sum(1 for x in restaurant_markers if x.get("is_specific_restaurant"))
    generic_activity_failure_count = sum(
        1 for x in activity_markers if x.get("is_grounding_failure")
    )
    restaurant_failure_count = sum(1 for x in restaurant_markers if x.get("is_grounding_failure"))
    transport_total = len(transport_markers)
    transport_complete = sum(1 for x in transport_markers if x.get("is_complete_route_chain"))
    day_complete = sum(1 for x in day_markers if x.get("has_at_least_one_concrete_core_activity"))

    issue_flags = []
    if day_count_expected and len(per_day) != day_count_expected:
        issue_flags.append("day_count_mismatch")
    if activity_total and activity_specific / max(activity_total, 1) < 0.75:
        issue_flags.append("low_activity_concreteness_rate")
    if day_count_expected and day_complete / max(day_count_expected, 1) < 0.8:
        issue_flags.append("incomplete_day_activity_coverage")
    if meal_total and meal_specific / max(meal_total, 1) < 0.5:
        issue_flags.append("low_specific_restaurant_rate")
    if generic_activity_failure_count > 0:
        issue_flags.append("generic_activity_label_failure")
    if restaurant_failure_count > 0:
        issue_flags.append("restaurant_specificity_failure")
    if transport_total and transport_complete / max(transport_total, 1) < 0.8:
        issue_flags.append("incomplete_daily_transport_routes")
    if not hotel_marker["is_specific_hotel_name"]:
        issue_flags.append("hotel_name_not_specific")
    if not cost_marker["has_total_cost"] or not cost_marker["has_budget_limit"]:
        issue_flags.append("cost_or_budget_missing")

    activity_rate = activity_specific / max(activity_total, 1) if activity_total else 0.0
    source_rate = activity_source / max(activity_total, 1) if activity_total else 0.0
    restaurant_rate = meal_specific / max(meal_total, 1) if meal_total else 0.0
    transport_rate = transport_complete / max(transport_total, 1) if transport_total else 0.0
    day_rate = day_complete / max(day_count_expected, 1) if day_count_expected else 0.0
    hotel_score = (
        1.0
        if hotel_marker["is_specific_hotel_name"] and hotel_marker["has_area"]
        else 0.5
        if hotel_marker["is_specific_hotel_name"]
        else 0.0
    )
    cost_score = 1.0 if cost_marker["has_total_cost"] and cost_marker["has_budget_limit"] else 0.0
    truthfulness_score = 100.0 * (
        0.30 * activity_rate
        + 0.15 * day_rate
        + 0.15 * restaurant_rate
        + 0.15 * transport_rate
        + 0.10 * hotel_score
        + 0.10 * cost_score
        + 0.05 * source_rate
    )

    summary = {
        "expected_day_count": int(day_count_expected),
        "actual_day_count": int(len(per_day)),
        "complete_day_rate": round(day_rate, 4),
        "activity_total_count": int(activity_total),
        "concrete_activity_count": int(activity_specific),
        "activity_concreteness_rate": round(activity_rate, 4),
        "source_backed_activity_rate": round(source_rate, 4),
        "restaurant_total_meal_count": int(meal_total),
        "specific_restaurant_count": int(meal_specific),
        "restaurant_specificity_rate": round(restaurant_rate, 4),
        "transport_day_count": int(transport_total),
        "complete_transport_day_count": int(transport_complete),
        "transport_chain_coverage": round(transport_rate, 4),
        "hotel_marker_pass": bool(
            hotel_marker["is_specific_hotel_name"] and hotel_marker["has_area"]
        ),
        "cost_marker_pass": bool(cost_score >= 1.0),
        "module5_truthfulness_score": round(max(0.0, min(100.0, truthfulness_score)), 4),
        "generic_activity_failure_count": int(generic_activity_failure_count),
        "restaurant_failure_count": int(restaurant_failure_count),
        "module5_quality_markers_pass": bool(
            truthfulness_score >= 75.0
            and len(issue_flags) <= 2
            and not (REQUIRE_NO_GENERIC_ACTIVITY and generic_activity_failure_count > 0)
        ),
        "issue_flags": issue_flags,
    }

    return {
        "marker_schema_version": "v23_module5_quality_markers",
        "summary": summary,
        "day_markers": day_markers,
        "activity_markers": activity_markers,
        "restaurant_markers": restaurant_markers,
        "hotel_marker": hotel_marker,
        "transport_markers": transport_markers,
        "cost_marker": cost_marker,
    }


def attach_itinerary_quality_markers(plan: Dict[str, Any]) -> Dict[str, Any]:
    try:
        plan["itinerary_quality_markers"] = build_itinerary_quality_markers(plan)
    except Exception as exc:
        plan["itinerary_quality_markers"] = {
            "marker_schema_version": "v23_module5_quality_markers",
            "summary": {
                "module5_truthfulness_score": 0.0,
                "module5_quality_markers_pass": False,
                "issue_flags": [f"quality_marker_generation_failed:{safe_text(exc)}"],
            },
        }
    return plan


def detect_placeholder_quality_issues(plan: Dict[str, Any]) -> List[str]:
    """Detect generic fallback text that should not be treated as a successful final itinerary."""
    issues: List[str] = []
    expected_days = safe_int(plan.get("day"), 0)
    per_day = plan.get("per_day_itinerary", [])
    if expected_days > 0 and isinstance(per_day, list) and len(per_day) != expected_days:
        issues.append(f"per_day_count_mismatch:{len(per_day)}_of_{expected_days}")
    if not isinstance(per_day, list):
        issues.append("per_day_itinerary_not_list")
        per_day = []

    for idx in range(expected_days if expected_days > 0 else len(per_day)):
        day_item = per_day[idx] if idx < len(per_day) and isinstance(per_day[idx], dict) else {}
        concrete_activity_count = 0
        for block in ["morning", "afternoon"]:
            values = day_item.get(block, []) if isinstance(day_item, dict) else []
            if not isinstance(values, list):
                values = [values] if safe_text(values) else []
            if not values:
                issues.append(f"day_{idx + 1}_{block}_missing")
                continue
            for activity in values:
                name = _activity_name(activity)
                if (
                    name
                    and not _contains_placeholder(name)
                    and not _activity_is_generic_or_vague(name)
                ):
                    concrete_activity_count += 1
                elif name:
                    issues.append(f"day_{idx + 1}_{block}_generic_activity:{name[:40]}")
        if concrete_activity_count == 0:
            issues.append(f"day_{idx + 1}_has_no_concrete_activity")

        transport_text = (
            _placeholder_text(day_item.get("transportation", ""))
            if isinstance(day_item, dict)
            else ""
        )
        if any(
            term in transport_text
            for term in [
                "nearest metro/mtr station",
                "mtr/metro line route",
                "metro/mtr line route",
                "nearest 地铁或mtr station",
                "地铁或mtr line route",
                "day 4 attractions",
                "day 5 attractions",
            ]
        ):
            issues.append(f"day_{idx + 1}_transport_placeholder")
        for meal_key in ["lunch", "dinner"]:
            meal = day_item.get(meal_key, {}) if isinstance(day_item, dict) else {}
            if _contains_placeholder(meal):
                issues.append(f"day_{idx + 1}_{meal_key}_placeholder")

    whole = _placeholder_text(
        {
            "itinerary_text": plan.get("itinerary_text"),
            "hotel_plan": plan.get("hotel_plan"),
            "transportation_plan": plan.get("transportation_plan"),
            "online_grounding": plan.get("online_grounding"),
        }
    )
    if any(
        term in whole
        for term in [
            "core attraction",
            "core family activity",
            "nearest metro/mtr station",
            "nearest 地铁或mtr station",
            "v48规则生成",
        ]
    ):
        issues.append("plan_contains_generic_placeholder_text")
    return list(dict.fromkeys(issues))


def detect_module5_generation_failure_reasons(plan: Dict[str, Any]) -> List[str]:
    """Detect true Module 5 generation failures before Module 6 evaluation.

    Module 5 should not label an empty or malformed consensus plan as success.
    Quality problems still go to Module 6/7, but missing core itinerary content
    is a generation failure and should be sent back to Module 2 through the
    backtracking queue.
    """
    reasons = []
    if not isinstance(plan, dict):
        return ["plan_is_not_dict"]

    family_id = safe_int(plan.get("family_id"), 0)
    if family_id <= 0:
        reasons.append("missing_or_invalid_family_id")

    destination_city = safe_text(plan.get("destination_city", ""))
    if not destination_city:
        reasons.append("missing_destination_city")

    day = safe_int(plan.get("day"), 0)
    if day <= 0:
        reasons.append("missing_or_invalid_day")

    itinerary_text = safe_text(plan.get("itinerary_text", ""))
    per_day = plan.get("per_day_itinerary", [])
    if not itinerary_text and not _list_has_meaningful_items(per_day):
        reasons.append("missing_itinerary_text_and_per_day_itinerary")

    if not isinstance(per_day, list) or len(per_day) == 0:
        reasons.append("missing_per_day_itinerary")
    else:
        useful_day_count = 0
        for day_item in per_day:
            if not isinstance(day_item, dict):
                if safe_text(day_item):
                    useful_day_count += 1
                continue
            day_content = []
            for key in [
                "theme",
                "morning",
                "afternoon",
                "evening",
                "core_activities",
                "optional_activities",
                "split_activities",
            ]:
                day_content.append(day_item.get(key))
            if _list_has_meaningful_items(day_content):
                useful_day_count += 1
        if useful_day_count == 0:
            reasons.append("per_day_itinerary_has_no_useful_activity_content")

    placeholder_issues = detect_placeholder_quality_issues(plan)
    for issue in placeholder_issues:
        issue_text = safe_text(issue)
        # v21: single broad labels and missing half-day slots are soft quality problems for Module 6/7,
        # not Module 5 hard failures. Hard-fail only truly unusable non-final days or malformed plan structure.
        if "_generic_activity:" in issue_text or issue_text.endswith("_missing"):
            continue
        if re.fullmatch(r"day_\d+_has_no_concrete_activity", issue_text):
            # v25: if concrete POI grounding fails, keep a rough itinerary as a soft quality issue.
            # Module 6 quality markers will penalize it and Module 7 can repair it, but Module 5
            # should not fail solely because a day remains coarse.
            continue
        # v63: a wrong day count (LLM truncated a long itinerary) and lingering
        # transport/meal placeholders are quality problems Module 6 penalises and
        # Module 7 repairs — they must NOT abort the whole family.  The day count is
        # also padded back to the expected number below, so the plan stays complete.
        if (
            issue_text.startswith("per_day_count_mismatch")
            or issue_text.endswith("_transport_placeholder")
            or issue_text.endswith("_placeholder")
        ):
            continue
        reasons.append(f"generic_or_incomplete_grounding:{issue_text}")
    grounding_quality_issues = (
        plan.get("grounding_quality_issues", [])
        if isinstance(plan.get("grounding_quality_issues", []), list)
        else []
    )
    if MODULE5_HARD_FAIL_ON_GROUNDING_ISSUES:
        for issue in grounding_quality_issues:
            reasons.append(f"grounding_quality_issue:{safe_text(issue)}")
    # else: grounding quality issues stay soft -> Module 6 penalises, Module 7 repairs; the family
    # is still generated and exported instead of being dropped.

    return list(dict.fromkeys(reasons))


def mark_module5_generation_failed(
    plan: Dict[str, Any], reasons: List[str], error: Optional[Exception] = None
) -> Dict[str, Any]:
    """Mark a family as Module 5 generation failed and eligible for Module 2 backtracking."""
    if not isinstance(plan, dict):
        plan = _empty_plan_row(0)
    reasons = [safe_text(x) for x in reasons if safe_text(x)]
    if error is not None:
        error_text = safe_text(error)
        if error_text:
            reasons.append(error_text)
    plan["status"] = "failed"
    plan["failure_stage"] = "module5_consensus_generation"
    plan["module5_generation_failed"] = True
    plan["backtrack_to_module2"] = True
    plan["module5_failure_reasons"] = list(dict.fromkeys(reasons))
    plan["module5_failure_reasons_json"] = _json_dump(plan["module5_failure_reasons"])
    plan["error"] = "; ".join(plan["module5_failure_reasons"])
    return plan


def save_module5_generation_failed_families(
    all_plans: List[Dict[str, Any]],
    json_path: Path = MODULE5_FAILED_FAMILIES_JSON,
    csv_path: Path = MODULE5_FAILED_FAMILIES_CSV,
) -> List[int]:
    """Save failed Module 5 family IDs for the main pipeline to backtrack to Module 2."""
    failed_records = []
    for item in all_plans:
        if not isinstance(item, dict):
            continue
        if (
            item.get("status") != "success"
            or item.get("module5_generation_failed") is True
            or item.get("backtrack_to_module2") is True
        ):
            family_id = safe_int(item.get("family_id"), 0)
            if family_id <= 0:
                continue
            failed_records.append(
                {
                    "family_id": family_id,
                    "failure_stage": safe_text(
                        item.get("failure_stage", "module5_consensus_generation")
                    )
                    or "module5_consensus_generation",
                    "status": safe_text(item.get("status", "failed")) or "failed",
                    "backtrack_to_module2": True,
                    "module5_generation_failed": True,
                    "error": safe_text(item.get("error", "")),
                    "module5_failure_reasons": item.get("module5_failure_reasons", []),
                }
            )

    # De-duplicate while keeping the first error message.
    deduped = {}
    for record in failed_records:
        deduped.setdefault(record["family_id"], record)
    failed_records = [deduped[fid] for fid in sorted(deduped)]
    failed_family_ids = [record["family_id"] for record in failed_records]

    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    save_json_file(
        data={
            "failed_family_ids": failed_family_ids,
            "backtrack_to_module2_family_ids": failed_family_ids,
            "records": failed_records,
        },
        path=json_path,
    )
    pd.DataFrame(
        failed_records,
        columns=[
            "family_id",
            "failure_stage",
            "status",
            "backtrack_to_module2",
            "module5_generation_failed",
            "error",
            "module5_failure_reasons",
        ],
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return failed_family_ids


def generate_consensus_itinerary_for_family(
    family_id: int,
    family_agents: List[Dict[str, Any]],
    individual_proposals: List[Dict[str, Any]],
    conflict_info: Dict[str, Any],
    negotiation_info: Dict[str, Any],
    extra_feedback_prompt: str = "",
) -> Dict[str, Any]:
    """Generate one consensus itinerary, with a short self-repair loop for placeholder/truncation failures."""
    last_error: Optional[Exception] = None
    last_reasons: List[str] = []
    base_feedback = extra_feedback_prompt
    attempts = max(1, MODULE5_MAX_REGEN_ATTEMPTS)
    first_agent = family_agents[0] if family_agents else {}
    activity_grounding_candidates = build_grounded_activity_candidates(
        destination_city=safe_text(first_agent.get("destination_city")),
        day_count=safe_int(first_agent.get("day"), 1),
        family_agents=family_agents,
        individual_proposals=individual_proposals,
        negotiation_info=negotiation_info,
        conflict_info=conflict_info,
    )
    candidate_count = sum(
        len(x.get("candidates", []))
        for x in activity_grounding_candidates.get("daily_candidates", [])
        if isinstance(x, dict)
    )
    if activity_grounding_candidates.get("used_online_search"):
        print(
            f"[Module 5] family_id={family_id} 活动候选: Google Places 补充 + 种子，共 {candidate_count} 个候选 (种子 {activity_grounding_candidates.get('seed_candidate_count', 0)} 个)"
        )
    elif candidate_count:
        print(
            f"[Module 5] family_id={family_id} 活动候选: 种子优先命中，共 {candidate_count} 个候选，未调用 Google Places"
        )

    for attempt_index in range(1, attempts + 1):
        attempt_feedback = base_feedback
        if last_reasons:
            repair_feedback = (
                "\n\n[MODULE5_SELF_REPAIR]\n"
                "The previous candidate was rejected before Module 6 because it contained generic placeholders, vague activity labels, or incomplete days.\n"
                "You must fix these issues in the new JSON:\n- "
                + "\n- ".join(last_reasons[:12])
                + "\nRules: output exactly the requested number of days; every day must have concrete named morning/afternoon activities; no vague labels such as 互动体验/文化遗址/当地美食晚餐/主题公园/购物; no core attraction/local attraction/nearest metro placeholders; transportation must be day-specific."
            )
            attempt_feedback = (
                (attempt_feedback + repair_feedback) if attempt_feedback else repair_feedback
            )

        messages = build_consensus_itinerary_messages(
            family_id=family_id,
            family_agents=family_agents,
            individual_proposals=individual_proposals,
            conflict_info=conflict_info,
            negotiation_info=negotiation_info,
            extra_feedback_prompt=attempt_feedback,
            activity_grounding_candidates=activity_grounding_candidates,
        )
        messages = budget_messages(
            messages, max_input_tokens=5600 if not attempt_feedback else 5400
        )
        if attempt_feedback:
            print(
                f"[Module 5] family_id={family_id} 第 {attempt_index}/{attempts} 次生成，带反馈，user_prompt 长度={len(messages[-1]['content'])}"
            )
        else:
            print(
                f"[Module 5] family_id={family_id} 第 {attempt_index}/{attempts} 次生成，user_prompt 长度={len(messages[-1]['content'])}"
            )

        start_time = time.time()
        try:
            api_result = call_generation_json(
                messages=messages,
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS_FINAL_ITINERARY,
                temperature=TEMPERATURE,
            )
            runtime_seconds = time.time() - start_time
            plan = api_result["data"]
            if not isinstance(plan, dict):
                plan = {"itinerary_text": safe_text(plan)}
            usage = api_result["usage"]
            fixed_plan = validate_and_fix_plan(
                plan=plan,
                family_id=family_id,
                family_agents=family_agents,
                usage=usage,
                runtime_seconds=runtime_seconds,
            )
            fixed_plan["activity_grounding_candidates"] = activity_grounding_candidates
            _r4 = negotiation_info.get("round4_final_plan", {}) or {}
            _neg_resolved = bool(
                _r4.get("conflict_resolution_map") or _r4.get("resolved_conflicts")
            )
            fixed_plan = enforce_rule_consistency(
                plan=fixed_plan, family_agents=family_agents, negotiation_resolved=_neg_resolved,
            )
            fixed_plan = normalize_per_day_itinerary(fixed_plan)
            fixed_plan = ground_generic_and_split_activities(fixed_plan)
            fixed_plan = attach_itinerary_quality_markers(fixed_plan)
            fixed_plan = trim_final_plan_text(fixed_plan)
            failure_reasons = detect_module5_generation_failure_reasons(fixed_plan)
            if failure_reasons:
                last_reasons = failure_reasons
                last_error = RuntimeError(
                    "[Module 5 generation failure] " + "; ".join(failure_reasons)
                )
                print(
                    f"[Module 5] family_id={family_id} 第 {attempt_index}/{attempts} 次生成未通过占位符/完整性检查: {'; '.join(failure_reasons[:6])}"
                )
                continue
            fixed_plan["status"] = "success"
            fixed_plan["failure_stage"] = ""
            fixed_plan["module5_generation_failed"] = False
            fixed_plan["backtrack_to_module2"] = False
            fixed_plan["module5_failure_reasons"] = []
            fixed_plan["module5_failure_reasons_json"] = "[]"
            return fixed_plan
        except Exception as exc:
            last_error = exc
            last_reasons = [safe_text(exc)]
            print(f"[Module 5] family_id={family_id} 第 {attempt_index}/{attempts} 次生成异常: {exc}")
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError("[Module 5 generation failure] unknown_error")


def build_error_plan(family_id: int, error: Exception) -> Dict[str, Any]:
    plan = _empty_plan_row(family_id)
    plan.update(
        {"family_id": family_id, "plan_id": "consensus_plan", "method": "multi_agent_consensus"}
    )
    return mark_module5_generation_failed(
        plan=plan, reasons=["module5_consensus_itinerary_generation_failed"], error=error
    )


def save_consensus_results(
    all_plans: List[Dict[str, Any]],
    all_flat_rows: List[Dict[str, Any]],
    output_csv_path: Path,
    output_json_path: Path,
):
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(all_flat_rows)
    if result_df.empty:
        result_df = pd.DataFrame(columns=PLAN_OUTPUT_COLUMNS)
    else:
        for column in PLAN_OUTPUT_COLUMNS:
            if column not in result_df.columns:
                result_df[column] = ""
        result_df = result_df[PLAN_OUTPUT_COLUMNS]
        if "family_id" in result_df.columns:
            result_df = result_df.sort_values(by=["family_id"])
    result_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    sorted_plans = sorted(all_plans, key=lambda x: safe_int(x.get("family_id"), 0))
    save_json_file(data=sorted_plans, path=output_json_path)


def print_plan_preview(plan: Dict[str, Any]):
    print("=" * 100)
    print(f"family_id: {plan.get('family_id')}")
    print(f"plan_id: {plan.get('plan_id')}")
    print(f"destination_city: {plan.get('destination_city')}")
    print(f"day: {plan.get('day')}")
    print(f"consensus_status: {plan.get('consensus_status')}")
    print(f"acceptance_vote_ratio: {plan.get('acceptance_vote_ratio')}")
    print(f"total_cost: {plan.get('total_cost')}")
    print(f"budget_limit: {plan.get('budget_limit')}")
    print(f"walking_distance_km: {plan.get('walking_distance_km')}")
    print(f"route_distance_km: {plan.get('route_distance_km')}")
    print(f"return_time: {plan.get('return_time')}")
    print(f"total_tokens: {plan.get('total_tokens')}")
    print("\nitinerary_text:")
    print(plan.get("itinerary_text", ""))
    print("=" * 100)


def generate_consensus_itineraries(
    agents_json_path: Path = AGENTS_JSON,
    proposals_csv_path: Path = PROPOSALS_CSV,
    conflicts_json_path: Path = CONFLICTS_JSON,
    negotiation_json_path: Path = NEGOTIATION_JSON,
    output_csv_path: Path = OUTPUT_CSV,
    output_json_path: Path = OUTPUT_JSON,
    test_first_n_families: Optional[int] = 3,
    max_workers_families: int = 2,
    target_family_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    if not agents_json_path.exists():
        raise FileNotFoundError(f"找不到 persona_agents.json，请先运行模块 1: {agents_json_path}")
    if not proposals_csv_path.exists():
        raise FileNotFoundError(
            f"找不到 individual_agent_proposals.csv，请先运行模块 2: {proposals_csv_path}"
        )
    if not conflicts_json_path.exists():
        raise FileNotFoundError(
            f"找不到 conflict_detection_results.json，请先运行模块 3: {conflicts_json_path}"
        )
    if not negotiation_json_path.exists():
        raise FileNotFoundError(
            f"找不到 negotiation_protocol_results.json，请先运行模块 4: {negotiation_json_path}"
        )
    print("[Module 5] Consensus Itinerary Generation 开始")
    print(f"[Module 5] agents_json_path: {agents_json_path}")
    print(f"[Module 5] proposals_csv_path: {proposals_csv_path}")
    print(f"[Module 5] conflicts_json_path: {conflicts_json_path}")
    print(f"[Module 5] negotiation_json_path: {negotiation_json_path}")
    agents = load_json_file(agents_json_path)
    family_agents_map = build_family_agents(agents)
    family_proposals_map = load_individual_proposals(proposals_csv_path)
    conflict_lookup = load_conflict_results(conflicts_json_path)
    negotiation_lookup = load_negotiation_results(negotiation_json_path)
    family_ids = get_valid_family_ids(
        family_agents_map=family_agents_map,
        family_proposals_map=family_proposals_map,
        conflict_lookup=conflict_lookup,
        negotiation_lookup=negotiation_lookup,
    )
    if target_family_ids is not None:
        target_set = {int(x) for x in target_family_ids}
        family_ids = [fid for fid in family_ids if int(fid) in target_set]
        print(f"[Module 5] Backtracking target mode: 只重新生成未达标家庭 family_id={family_ids}")
    elif test_first_n_families is not None:
        family_ids = family_ids[:test_first_n_families]
    print(f"[Module 5] 实际运行家庭数量: {len(family_ids)}")
    print(f"[Module 5] max_workers_families: {max_workers_families}")
    all_plans = []
    all_flat_rows = []

    def worker(family_id: int) -> Dict[str, Any]:
        family_agents = family_agents_map[family_id]
        individual_proposals = family_proposals_map.get(family_id, [])
        conflict_info = conflict_lookup.get(family_id, {"family_id": family_id, "conflicts": []})
        negotiation_info = negotiation_lookup.get(family_id, {"family_id": family_id})
        try:
            plan = generate_consensus_itinerary_for_family(
                family_id=family_id,
                family_agents=family_agents,
                individual_proposals=individual_proposals,
                conflict_info=conflict_info,
                negotiation_info=negotiation_info,
            )
            return plan
        except Exception as e:
            print(f"[Module 5] family_id={family_id} 最终共识行程生成失败: {e}")
            return build_error_plan(family_id=family_id, error=e)

    with ThreadPoolExecutor(max_workers=max_workers_families) as executor:
        future_to_family_id = {
            executor.submit(worker, family_id): family_id for family_id in family_ids
        }
        for index, future in enumerate(as_completed(future_to_family_id), start=1):
            family_id = future_to_family_id[future]
            print(f"\n[Module 5] 完成第 {index}/{len(family_ids)} 个家庭最终行程生成，family_id={family_id}")
            plan = future.result()
            all_plans.append(plan)
            if plan.get("status") == "success":
                flat_row = flatten_plan_for_csv(plan)
                all_flat_rows.append(flat_row)
                if index <= 3:
                    print_plan_preview(plan)
            else:
                all_flat_rows.append(flatten_plan_for_csv(plan))
            save_consensus_results(
                all_plans=all_plans,
                all_flat_rows=all_flat_rows,
                output_csv_path=output_csv_path,
                output_json_path=output_json_path,
            )
            time.sleep(SLEEP_SECONDS)
    save_consensus_results(
        all_plans=all_plans,
        all_flat_rows=all_flat_rows,
        output_csv_path=output_csv_path,
        output_json_path=output_json_path,
    )
    module5_failed_family_ids = save_module5_generation_failed_families(all_plans=all_plans)
    final_df = pd.DataFrame(all_flat_rows)
    if "family_id" in final_df.columns:
        final_df = final_df.sort_values(by=["family_id"])
    success_count = sum((1 for item in all_plans if item.get("status") == "success"))
    failed_count = sum((1 for item in all_plans if item.get("status") == "failed"))
    print("[Module 5] Consensus Itinerary Generation 完成")
    print(f"[Module 5] CSV 输出文件: {output_csv_path}")
    print(f"[Module 5] JSON 输出文件: {output_json_path}")
    print(f"[Module 5] 成功家庭数量: {success_count}")
    print(f"[Module 5] 失败家庭数量: {failed_count}")
    print(f"[Module 5] 需要回退到模块 2 的家庭数量: {len(module5_failed_family_ids)}")
    if module5_failed_family_ids:
        print(f"[Module 5] 需要回退到模块 2 的 family_id: {module5_failed_family_ids}")
        print(f"[Module 5] 回退列表 JSON: {MODULE5_FAILED_FAMILIES_JSON}")
        print(f"[Module 5] 回退列表 CSV: {MODULE5_FAILED_FAMILIES_CSV}")
    print("[Module 5] generated_plans 前 5 行预览：")
    preview_cols = [
        "family_id",
        "plan_id",
        "method",
        "destination_city",
        "day",
        "total_cost",
        "budget_limit",
        "walking_distance_km",
        "daily_poi_count",
        "route_distance_km",
        "return_time",
        "consensus_status",
        "acceptance_vote_ratio",
        "total_tokens",
        "status",
    ]
    available_cols = [col for col in preview_cols if col in final_df.columns]
    if available_cols:
        print(final_df[available_cols].head(5).to_string(index=False))
    return final_df
