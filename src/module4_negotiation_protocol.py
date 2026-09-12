"""Module 4 - Negotiation protocol: run multi-round negotiation across member agents to reach a consensus direction."""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.common import (
    RESULTS_DIR,
    RoleMemoryStore,
    budget_messages,
    build_family_agents,
    compact_json_for_prompt,
    compact_memory_items,
    load_backtracking_feedback_lookup,
    load_conflict_results,
    load_json_file,
    parse_json_text,
    safe_int,
    safe_text,
    save_json_file,
)
from src.config import MODULE4_MEMORY_DIR
from src.llm_client import call_generation_json

AGENTS_JSON = RESULTS_DIR / "module_1" / "persona_agents.json"
PROPOSALS_CSV = RESULTS_DIR / "module_2" / "individual_agent_proposals.csv"
CONFLICTS_JSON = RESULTS_DIR / "module_3" / "conflict_detection_results.json"
MEMORY_DIR = MODULE4_MEMORY_DIR
OUTPUT_CSV = RESULTS_DIR / "module_4" / "negotiation_protocol_results.csv"
OUTPUT_JSON = RESULTS_DIR / "module_4" / "negotiation_protocol_results.json"
BACKTRACKING_FEEDBACK_JSON = RESULTS_DIR / "module_7" / "backtracking_to_module2_feedback.json"
MODEL_NAME = os.getenv(
    "OPENROUTER_GENERATION_MODEL",
    os.getenv("OPENROUTER_MODEL_NAME", os.getenv("GENERATION_MODEL_NAME", "openai/gpt-4o-mini")),
)
MAX_TOKENS_ROUND_AGENT = 2000
MAX_TOKENS_INITIAL_PLAN = 3500
MAX_TOKENS_FINAL_PLAN = 3500
TEMPERATURE_AGENT = 0.2
TEMPERATURE_COORDINATOR = 0.25
SLEEP_SECONDS = 0.1


def compact_backtracking_feedback(family_feedback: Dict[str, Any], max_chars: int = 1200) -> str:
    if not family_feedback:
        return ""
    payload = {
        "trigger_rule": family_feedback.get("trigger_rule"),
        "revision_rounds_completed": family_feedback.get("revision_rounds_completed"),
        "final_failed_reasons": family_feedback.get("final_failed_reasons", [])[:5],
        "backtracking_failure_profile": family_feedback.get("backtracking_failure_profile", {}),
        "recommended_regeneration_strategy": family_feedback.get(
            "recommended_regeneration_strategy", ""
        ),
        "lowest_satisfaction_member": family_feedback.get("lowest_satisfaction_member", {}),
        "rejected_members": family_feedback.get("rejected_members", [])[:3],
        "hard_failed_members": family_feedback.get("hard_failed_members", [])[:3],
        "bad_individual_proposals": family_feedback.get("bad_individual_proposals", [])[:3],
        "feedback_for_module4_negotiation": family_feedback.get(
            "feedback_for_module4_negotiation", []
        )[:6],
        "strategy_templates_for_next_cycle": family_feedback.get(
            "strategy_templates_for_next_cycle", []
        )[:3],
    }
    return compact_json_for_prompt(payload, max_chars=max_chars)


def append_backtracking_feedback_to_messages(
    messages: List[Dict[str, str]], family_feedback: Dict[str, Any], stage_name: str
) -> List[Dict[str, str]]:
    feedback_text = compact_backtracking_feedback(family_feedback)
    if not feedback_text:
        return messages
    messages[-1][
        "content"
    ] += f"\n\nPrior failure feedback for {stage_name}, keep fixes concise:\n{feedback_text}\nRules: protect rejected/lowest-satisfaction members; do not sacrifice hard constraints.\n"
    return messages


def load_individual_proposals(path: Path) -> Dict[str, Dict[str, Any]]:
    df = pd.read_csv(path)
    required_columns = ["agent_id", "family_id", "member_id", "individual_proposal_json"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"individual_agent_proposals.csv 缺少字段: {col}")
    proposal_lookup = {}
    for _, row in df.iterrows():
        agent_id = safe_text(row.get("agent_id"))
        proposal_lookup[agent_id] = {
            "agent_id": agent_id,
            "family_id": safe_int(row.get("family_id")),
            "member_id": safe_int(row.get("member_id")),
            "role": safe_text(row.get("role")),
            "age_stage": safe_text(row.get("age_stage")),
            "individual_proposal": parse_json_text(row.get("individual_proposal_json"), default={}),
        }
    return proposal_lookup


def truncate_words(text: Any, max_words: int = 12) -> str:
    words = safe_text(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def limit_text_list(value: Any, max_items: int, max_words: int = 10) -> List[str]:
    output = []
    for item in _as_list(value):
        if isinstance(item, dict):
            item = (
                item.get("item")
                or item.get("need")
                or item.get("objection")
                or item.get("concession")
                or item.get("activity")
                or json.dumps(item, ensure_ascii=False)
            )
        text = truncate_words(item, max_words=max_words)
        if text:
            output.append(text)
        if len(output) >= max_items:
            break
    return output


def compact_agent_profile_for_negotiation(agent: Dict[str, Any]) -> Dict[str, Any]:
    profile = agent.get("profile", {}) or {}
    constraints = agent.get("constraints", {}) or {}
    return {
        "agent_id": agent.get("agent_id"),
        "role": profile.get("role"),
        "age_stage": profile.get("age_stage"),
        "interests": _as_list(profile.get("interest_preferences", []))[:3],
        "physical": profile.get("physical_constraint"),
        "budget": profile.get("budget_sensitivity"),
        "diet": _as_list(profile.get("dietary_preference", []))[:3],
        "pace": profile.get("travel_pace"),
        "must_include": _as_list(constraints.get("must_include", []))[:3],
        "must_avoid": _as_list(constraints.get("must_avoid", []))[:3],
        "max_walk_km": constraints.get("max_walking_distance_km"),
        "latest_return": constraints.get("latest_return_time"),
    }


def compact_individual_proposal_for_round1(proposal: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(proposal, dict):
        return {}
    return {
        "summary": truncate_words(proposal.get("short_self_summary", ""), 18),
        "priority": _as_list(proposal.get("proposal_priority", []))[:4],
        "preferred": _as_list(proposal.get("preferred_attractions", []))[:3],
        "rejected": _as_list(proposal.get("rejected_arrangements", []))[:2],
        "negotiable": limit_text_list(proposal.get("negotiable_parts", []), 2, 8),
        "non_negotiable": limit_text_list(proposal.get("non_negotiable_parts", []), 2, 8),
    }


def normalize_round1_output(data: Any, agent: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        data = {}
    profile = agent.get("profile", {}) or {}
    constraints = agent.get("constraints", {}) or {}
    out = {
        "round": 1,
        "agent_id": safe_text(agent.get("agent_id")),
        "agent_name": safe_text(profile.get("role")) or safe_text(data.get("agent_name")),
        "must_have_constraints": limit_text_list(
            data.get("must_have_constraints") or constraints.get("must_include"), 3, 10
        ),
        "soft_preferences": limit_text_list(
            data.get("soft_preferences") or profile.get("interest_preferences"), 2, 8
        ),
        "negotiable_items": limit_text_list(data.get("negotiable_items"), 2, 8),
        "non_negotiable_items": limit_text_list(
            data.get("non_negotiable_items") or constraints.get("must_avoid"), 2, 10
        ),
        "possible_conflicts": limit_text_list(data.get("possible_conflicts"), 1, 10),
        "minimum_acceptance_standard": limit_text_list(
            data.get("minimum_acceptance_standard"), 2, 10
        ),
        "objections_to_current_plan": [],
        "acceptable_concessions": [],
        "consensus_vote": "Reject",
        "reason": "No draft plan yet.",
    }
    if not out["must_have_constraints"]:
        items = []
        for x in _as_list(constraints.get("must_include", []))[:2]:
            items.append(f"include {safe_text(x)}")
        for x in _as_list(constraints.get("must_avoid", []))[:1]:
            items.append(f"avoid {safe_text(x)}")
        if constraints.get("max_walking_distance_km"):
            items.append(f"walk <= {constraints.get('max_walking_distance_km')}km/day")
        if constraints.get("latest_return_time"):
            items.append(f"return by {constraints.get('latest_return_time')}")
        out["must_have_constraints"] = limit_text_list(items, 3, 10)
    return out


def normalize_round2_output(data: Any, agent: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        data = {}
    profile = agent.get("profile", {}) or {}
    objections = []
    for obj in _as_list(data.get("objections_to_current_plan"))[:2]:
        if not isinstance(obj, dict):
            obj = {"objection": safe_text(obj)}
        objections.append(
            {
                "objection": truncate_words(obj.get("objection", ""), 10),
                "severity": safe_text(obj.get("severity", "medium")) or "medium",
                "violation_type": safe_text(obj.get("violation_type", "soft_preference_mismatch"))
                or "soft_preference_mismatch",
                "required_fix": truncate_words(obj.get("required_fix", ""), 10),
                "will_reject_if_unfixed": bool(obj.get("will_reject_if_unfixed", False)),
            }
        )
    return {
        "round": 2,
        "agent_id": safe_text(agent.get("agent_id")),
        "agent_name": safe_text(profile.get("role")),
        "objections_to_current_plan": objections,
        "acceptable_concessions": [],
        "consensus_vote": normalize_vote(data.get("consensus_vote")),
        "reason": truncate_words(data.get("reason", ""), 12),
    }


def normalize_round3_output(data: Any, agent: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        data = {}
    profile = agent.get("profile", {}) or {}
    concessions = []
    for con in _as_list(data.get("acceptable_concessions"))[:2]:
        if not isinstance(con, dict):
            con = {"concession": safe_text(con)}
        concessions.append(
            {
                "concession": truncate_words(con.get("concession", ""), 10),
                "concession_type": safe_text(con.get("concession_type", "soft_preference"))
                or "soft_preference",
                "condition": truncate_words(con.get("condition", ""), 10),
                "priority_loss": safe_text(con.get("priority_loss", "low")) or "low",
                "still_reject_if": truncate_words(con.get("still_reject_if", ""), 10),
            }
        )
    return {
        "round": 3,
        "agent_id": safe_text(agent.get("agent_id")),
        "agent_name": safe_text(profile.get("role")),
        "acceptable_concessions": concessions,
        "non_negotiable_items": limit_text_list(data.get("non_negotiable_items"), 2, 10),
        "consensus_vote": normalize_vote(data.get("consensus_vote")),
        "reason": truncate_words(data.get("reason", ""), 12),
    }


def get_valid_family_ids(
    family_agents_map: Dict[int, List[Dict[str, Any]]],
    proposal_lookup: Dict[str, Dict[str, Any]],
    conflict_lookup: Dict[int, Dict[str, Any]],
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
        missing_proposal_agents = []
        for agent in agents:
            agent_id = agent.get("agent_id")
            if agent_id not in proposal_lookup:
                missing_proposal_agents.append(agent_id)
                continue
            proposal_data = proposal_lookup.get(agent_id, {})
            proposal = proposal_data.get("individual_proposal", {})
            if not proposal:
                missing_proposal_agents.append(agent_id)
                continue
            if isinstance(proposal, dict) and "error" in proposal:
                missing_proposal_agents.append(agent_id)
        if missing_proposal_agents:
            reasons.append(f"missing_or_invalid_proposals:{missing_proposal_agents}")
        if family_id not in conflict_lookup:
            reasons.append("missing_conflict_detection_result")
        if reasons:
            skipped_info.append(
                {"family_id": family_id, "agent_count": agent_count, "reasons": reasons}
            )
        else:
            valid_family_ids.append(family_id)
    print("\n[Module 4] 完整可用 family_id 检查完成")
    print(f"[Module 4] 可用家庭数量: {len(valid_family_ids)}")
    print(f"[Module 4] 跳过家庭数量: {len(skipped_info)}")
    if skipped_info:
        print("[Module 4] 被跳过的 family_id：")
        for item in skipped_info[:3]:
            print(
                f"family_id={item['family_id']}, agent_count={item['agent_count']}, reasons={item['reasons']}"
            )
        if len(skipped_info) > 3:
            print(f"... 有 {len(skipped_info)} 个家庭被跳过。")
    return sorted(valid_family_ids)


NEGOTIATION_HIGH_PRIORITY_TYPES = {
    "hard_constraint_feasibility_conflict",
    "crowd_queue_feasibility_conflict",
    "intensity_feasibility_conflict",
    "child_older_vs_high_intensity_conflict",
    "child_older_vs_late_return_conflict",
    "rest_requirement_conflict",
}
NEGOTIATION_SPLIT_REQUIRED_TYPES = {
    "hard_constraint_feasibility_conflict",
    "crowd_queue_feasibility_conflict",
    "attraction_type_conflict",
    "preference_avoidance_conflict",
}
NEGOTIATION_HARD_OBJECTION_KEYWORDS = [
    "must",
    "must_have",
    "must-have",
    "must avoid",
    "must_avoid",
    "must include",
    "must_include",
    "hard constraint",
    "non-negotiable",
    "walking",
    "return time",
    "late return",
    "budget",
    "avoid",
]


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False).lower()
    except Exception:
        return safe_text(value).lower()


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return [value]
    text = safe_text(value)
    return [text] if text else []


def normalize_vote(value: Any) -> str:
    text = safe_text(value).lower()
    if "reject" in text:
        return "Reject"
    if "minor" in text or "change" in text:
        return "Accept with minor changes"
    if "accept" in text:
        return "Accept"
    return "Reject"


def conflict_priority_score(conflict: Dict[str, Any]) -> int:
    ctype = safe_text(conflict.get("conflict_type"))
    severity = safe_text(conflict.get("severity", "medium")).lower()
    score = 0
    if severity == "high":
        score += 3
    elif severity == "medium":
        score += 2
    else:
        score += 1
    if ctype in NEGOTIATION_HIGH_PRIORITY_TYPES:
        score += 3
    if ctype in NEGOTIATION_SPLIT_REQUIRED_TYPES:
        score += 2
    return score


def classify_constraint_satisfiability(conflict: Dict[str, Any]) -> str:
    ctype = safe_text(conflict.get("conflict_type"))
    evidence_text = _json_text(conflict.get("evidence", {}))
    suggested_text = safe_text(conflict.get("suggested_resolution", "")).lower()
    conflict_text = " ".join([ctype.lower(), evidence_text, suggested_text])
    if ctype in NEGOTIATION_SPLIT_REQUIRED_TYPES:
        return "split_satisfiable"
    if any(
        (
            k in conflict_text
            for k in [
                "split_activity",
                "optional_activity",
                "non_participating",
                "must_include",
                "must_avoid",
            ]
        )
    ):
        return "split_satisfiable"
    if ctype in [
        "budget_conflict",
        "meal_budget_conflict",
        "hotel_budget_conflict",
        "time_conflict",
        "pace_conflict",
        "route_distance_conflict",
        "daily_poi_count_conflict",
        "travel_intensity_conflict",
    ]:
        return "fully_satisfiable"
    if ctype in ["hard_constraint_feasibility_conflict"]:
        return "split_satisfiable"
    return "fully_satisfiable"


def recommended_strategy_for_conflict(conflict: Dict[str, Any], satisfiability_type: str) -> str:
    ctype = safe_text(conflict.get("conflict_type"))
    if satisfiability_type == "split_satisfiable":
        return "split_activity"
    if satisfiability_type == "unsatisfiable":
        return "remove_or_mark_unresolved"
    if ctype in ["budget_conflict", "meal_budget_conflict", "hotel_budget_conflict"]:
        return "budget_compromise"
    if ctype in [
        "route_distance_conflict",
        "travel_intensity_conflict",
        "child_older_vs_high_intensity_conflict",
    ]:
        return "lower_intensity_shared_plan"
    if ctype in ["time_conflict", "child_older_vs_late_return_conflict"]:
        return "earlier_return_policy"
    return "shared_compromise"


def build_negotiation_agenda(
    conflict_info: Dict[str, Any], family_agents: List[Dict[str, Any]]
) -> Dict[str, Any]:
    conflicts = _as_list(conflict_info.get("conflicts", []))
    items = []
    for idx, conflict in enumerate(conflicts, start=1):
        if not isinstance(conflict, dict):
            continue
        satisfiability_type = classify_constraint_satisfiability(conflict)
        priority_score = conflict_priority_score(conflict)
        if priority_score >= 6:
            priority = "high"
        elif priority_score >= 4:
            priority = "medium"
        else:
            priority = "low"
        involved_agents = _as_list(conflict.get("involved_agents", []))
        items.append(
            {
                "conflict_id": f"C{idx}",
                "priority": priority,
                "priority_score": priority_score,
                "conflict_type": safe_text(conflict.get("conflict_type", "unknown")),
                "severity": safe_text(conflict.get("severity", "medium")),
                "satisfiability_type": satisfiability_type,
                "recommended_strategy": recommended_strategy_for_conflict(
                    conflict, satisfiability_type
                ),
                "involved_agents": involved_agents,
                "description": safe_text(conflict.get("description", "")),
                "evidence": conflict.get("evidence", {}),
                "suggested_resolution": safe_text(conflict.get("suggested_resolution", "")),
                "must_be_addressed_in_round4": priority in ["high", "medium"],
            }
        )
    items = sorted(items, key=lambda x: x.get("priority_score", 0), reverse=True)
    return {
        "agenda_version": "structured_negotiation_agenda_v1",
        "family_id": safe_int(conflict_info.get("family_id"), 0),
        "total_conflicts": len(items),
        "high_priority_conflict_count": sum((1 for x in items if x.get("priority") == "high")),
        "split_required_conflict_count": sum(
            (1 for x in items if x.get("recommended_strategy") == "split_activity")
        ),
        "agenda_items": items,
    }


def count_round2_objections(round2_outputs: List[Dict[str, Any]]) -> Dict[str, int]:
    total = 0
    high = 0
    hard = 0
    reject_members = 0
    for item in _as_list(round2_outputs):
        if normalize_vote(item.get("consensus_vote")) == "Reject":
            reject_members += 1
        for obj in _as_list(item.get("objections_to_current_plan", [])):
            if not obj:
                continue
            total += 1
            obj_text = _json_text(obj)
            severity = safe_text(obj.get("severity") if isinstance(obj, dict) else "").lower()
            if severity == "high":
                high += 1
            if any((k in obj_text for k in NEGOTIATION_HARD_OBJECTION_KEYWORDS)):
                hard += 1
    return {
        "total_objection_count": total,
        "high_severity_objection_count": high,
        "hard_constraint_objection_count": hard,
        "rejected_member_count": reject_members,
    }


def count_quality_concessions(round3_outputs: List[Dict[str, Any]]) -> Dict[str, int]:
    total = 0
    useful = 0
    bad_hard_concession = 0
    for item in _as_list(round3_outputs):
        for concession in _as_list(item.get("acceptable_concessions", [])):
            if not concession:
                continue
            total += 1
            text = _json_text(concession)
            ctype = safe_text(
                concession.get("concession_type", "") if isinstance(concession, dict) else ""
            ).lower()
            has_condition = (
                "condition" in concession and safe_text(concession.get("condition"))
                if isinstance(concession, dict)
                else False
            )
            has_still_reject = (
                "still_reject_if" in concession and safe_text(concession.get("still_reject_if"))
                if isinstance(concession, dict)
                else False
            )
            if (
                has_condition
                or has_still_reject
                or ctype in ["soft_preference", "schedule_adjustment", "optional_activity"]
            ):
                useful += 1
            if "must" in text and any(
                (k in text for k in ["give up", "remove", "cancel", "ignore"])
            ):
                bad_hard_concession += 1
            if ctype in ["hard_constraint", "must_have"]:
                bad_hard_concession += 1
    return {
        "total_concession_count": total,
        "useful_concession_count": useful,
        "bad_hard_concession_count": bad_hard_concession,
    }


def count_unresolved_conflicts(final_plan: Dict[str, Any]) -> Dict[str, int]:
    unresolved = _as_list(final_plan.get("unresolved_conflicts", []))
    high = 0
    for item in unresolved:
        text = _json_text(item)
        severity = safe_text(item.get("severity") if isinstance(item, dict) else "").lower()
        if severity == "high" or any(
            (k in text for k in ["must_include", "must_avoid", "hard", "non-negotiable", "reject"])
        ):
            high += 1
    return {"unresolved_conflict_count": len(unresolved), "high_unresolved_conflict_count": high}


def compute_acceptance_ratio_from_votes(
    final_plan: Dict[str, Any], family_agents: List[Dict[str, Any]]
) -> float:
    votes = _as_list(final_plan.get("agent_votes", []))
    if not votes:
        return 0.0
    accept_like = 0
    for vote_item in votes:
        vote = normalize_vote(vote_item.get("vote") if isinstance(vote_item, dict) else vote_item)
        if vote in ["Accept", "Accept with minor changes"]:
            accept_like += 1
    denom = max(len(votes), len(family_agents), 1)
    return round(accept_like / denom, 4)


def compute_negotiation_quality_score(
    round2_outputs: List[Dict[str, Any]],
    round3_outputs: List[Dict[str, Any]],
    final_plan: Dict[str, Any],
    negotiation_agenda: Dict[str, Any],
    family_agents: List[Dict[str, Any]],
) -> Dict[str, Any]:
    obj_counts = count_round2_objections(round2_outputs)
    concession_counts = count_quality_concessions(round3_outputs)
    unresolved_counts = count_unresolved_conflicts(final_plan)
    agenda_items = _as_list(negotiation_agenda.get("agenda_items", []))
    high_agenda_count = sum(
        (1 for x in agenda_items if isinstance(x, dict) and x.get("priority") == "high")
    )
    split_required_count = sum(
        (
            1
            for x in agenda_items
            if isinstance(x, dict) and x.get("recommended_strategy") == "split_activity"
        )
    )
    split_recommendations = _as_list(final_plan.get("split_activity_recommendations", []))
    resolved_conflicts = _as_list(final_plan.get("resolved_conflicts", [])) + _as_list(
        final_plan.get("conflict_resolution_map", [])
    )
    unresolved_count = unresolved_counts["unresolved_conflict_count"]
    high_unresolved_count = unresolved_counts["high_unresolved_conflict_count"]
    acceptance_ratio = compute_acceptance_ratio_from_votes(final_plan, family_agents)
    hard_conflict_resolution_score = 100.0
    if high_agenda_count > 0:
        hard_conflict_resolution_score = max(
            0.0,
            100.0
            - 28.0 * high_unresolved_count
            - 8.0
            * max(0, high_agenda_count - len(resolved_conflicts) - len(split_recommendations)),
        )
    if obj_counts["high_severity_objection_count"] > 0:
        resolved_num = safe_int(final_plan.get("number_of_objections_resolved"), 0)
        objection_resolution_score = min(
            100.0, 100.0 * resolved_num / max(obj_counts["high_severity_objection_count"], 1)
        )
    else:
        objection_resolution_score = 100.0
    if split_required_count > 0:
        split_coverage_score = min(100.0, 100.0 * len(split_recommendations) / split_required_count)
    else:
        split_coverage_score = 100.0
    total_concessions = concession_counts["total_concession_count"]
    if total_concessions > 0:
        concession_quality_score = (
            100.0 * concession_counts["useful_concession_count"] / total_concessions
        )
    else:
        concession_quality_score = 70.0
    concession_quality_score = max(
        0.0, concession_quality_score - 20.0 * concession_counts["bad_hard_concession_count"]
    )
    rejected_protection_score = 100.0
    if obj_counts["rejected_member_count"] > 0:
        protection_items = _as_list(final_plan.get("rejected_member_protection", []))
        protected = len(protection_items)
        rejected_protection_score = min(
            100.0, 100.0 * protected / obj_counts["rejected_member_count"]
        )
    voting_score = acceptance_ratio * 100.0
    score = (
        0.25 * hard_conflict_resolution_score
        + 0.2 * objection_resolution_score
        + 0.2 * split_coverage_score
        + 0.15 * concession_quality_score
        + 0.1 * rejected_protection_score
        + 0.1 * voting_score
    )
    score -= 10.0 * high_unresolved_count
    score -= 4.0 * max(0, unresolved_count - high_unresolved_count)
    score = max(0.0, min(100.0, round(score, 2)))
    return {
        "negotiation_quality_score": score,
        "hard_conflict_resolution_score": round(hard_conflict_resolution_score, 2),
        "high_severity_objection_resolution_score": round(objection_resolution_score, 2),
        "split_required_conflict_coverage_score": round(split_coverage_score, 2),
        "concession_quality_score": round(concession_quality_score, 2),
        "rejected_member_protection_score": round(rejected_protection_score, 2),
        "voting_score": round(voting_score, 2),
        "acceptance_vote_ratio_computed": acceptance_ratio,
        "high_priority_conflict_count": high_agenda_count,
        "split_required_conflict_count": split_required_count,
        **obj_counts,
        **concession_counts,
        **unresolved_counts,
    }


def enforce_consensus_status(final_plan: Dict[str, Any], quality: Dict[str, Any]) -> Dict[str, Any]:
    original_status = safe_text(final_plan.get("consensus_status", ""))
    acceptance_ratio = float(quality.get("acceptance_vote_ratio_computed", 0.0))
    high_unresolved = safe_int(quality.get("high_unresolved_conflict_count"), 0)
    unresolved = safe_int(quality.get("unresolved_conflict_count"), 0)
    quality_score = float(quality.get("negotiation_quality_score", 0.0))
    final_votes = _as_list(final_plan.get("agent_votes", []))
    final_reject_count = 0
    for vote_item in final_votes:
        vote = normalize_vote(vote_item.get("vote") if isinstance(vote_item, dict) else vote_item)
        if vote == "Reject":
            final_reject_count += 1
    reasons = []
    if final_reject_count > 0:
        reasons.append("final_agent_vote_reject")
    if acceptance_ratio < 0.67:
        reasons.append("final_acceptance_ratio_too_low")
    if high_unresolved > 0:
        reasons.append("high_unresolved_conflict_exists")
    elif unresolved > 0:
        reasons.append("unresolved_conflict_exists")
    if quality_score < 78.0:
        reasons.append("negotiation_quality_score_low")
    if final_reject_count > 0 or acceptance_ratio < 0.67:
        new_status = "No Consensus"
    elif high_unresolved > 0 or unresolved > 0 or acceptance_ratio < 1.0 or (quality_score < 78.0):
        new_status = "Partial Consensus"
    else:
        new_status = "Full Consensus"
    final_plan["consensus_status_original_model"] = original_status
    final_plan["consensus_status"] = new_status
    final_plan["acceptance_vote_ratio"] = round(acceptance_ratio, 4)
    final_plan["forced_consensus_downgrade"] = bool(
        original_status and original_status != new_status
    )
    final_plan["consensus_adjustment_reasons"] = reasons
    final_plan["final_reject_count"] = final_reject_count
    return final_plan


def build_round1_messages(
    agent: Dict[str, Any], proposal: Dict[str, Any], memories: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    profile = compact_agent_profile_for_negotiation(agent)
    proposal_text = compact_json_for_prompt(
        compact_individual_proposal_for_round1(proposal.get("individual_proposal", {})),
        max_chars=650,
    )
    memory_text = compact_memory_items(memories, max_items=1, max_chars=220)
    system_prompt = "You are one persona agent. Output compact valid JSON only. Output token limit: under 900 tokens."
    user_prompt = f"""
Round 1: declare compact negotiation needs.
Persona: {json.dumps(profile, ensure_ascii=False)}
Proposal brief: {proposal_text}
Memory brief: {memory_text}

Return JSON only:
{{
  "round": 1,
  "agent_id": "{agent.get('agent_id')}",
  "agent_name": "{profile.get('role')}",
  "must_have_constraints": ["max 3 short items"],
  "soft_preferences": ["max 2 short items"],
  "negotiable_items": ["max 2 short items"],
  "non_negotiable_items": ["max 2 short items"],
  "possible_conflicts": ["max 1 short risk"],
  "minimum_acceptance_standard": ["max 2 short checks"],
  "objections_to_current_plan": [],
  "acceptable_concessions": [],
  "consensus_vote": "Reject",
  "reason": "No draft plan yet."
}}
Output token limit: total output under 450 tokens. Each item under 10 words. No paragraphs.
""".strip()
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def run_round1_core_needs(
    family_agents: List[Dict[str, Any]],
    proposal_lookup: Dict[str, Dict[str, Any]],
    memory_store: RoleMemoryStore,
    family_feedback: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    outputs = []
    for agent in family_agents:
        agent_id = agent["agent_id"]
        proposal = proposal_lookup.get(agent_id, {})
        memories = memory_store.load_memories(agent=agent, max_items=3)
        messages = build_round1_messages(agent=agent, proposal=proposal, memories=memories)
        messages = append_backtracking_feedback_to_messages(
            messages=messages,
            family_feedback=family_feedback or {},
            stage_name="Round 1 core needs",
        )
        messages = budget_messages(messages, max_input_tokens=2300)
        try:
            api_result = call_generation_json(
                messages=messages,
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS_ROUND_AGENT,
                temperature=TEMPERATURE_AGENT,
            )
            data = normalize_round1_output(api_result.get("data", {}), agent)
            usage = api_result.get("usage", {})
        except Exception as exc:
            print(f"[Module 4] agent_id={agent_id} Round 1 LLM 失败，使用本地压缩兜底: {exc}")
            data = normalize_round1_output({}, agent)
            usage = {}
        data["agent_id"] = agent_id
        data["usage"] = usage
        outputs.append(data)
        memory_store.save_negotiation_memory(
            agent=agent, round_name="negotiation_round_1_core_needs", content=data
        )
        time.sleep(SLEEP_SECONDS)
    return outputs


def build_initial_plan_messages(
    family_id: int,
    family_agents: List[Dict[str, Any]],
    round1_outputs: List[Dict[str, Any]],
    conflict_info: Dict[str, Any],
    negotiation_agenda: Dict[str, Any],
) -> List[Dict[str, str]]:
    first_agent = family_agents[0]
    round1_text = compact_json_for_prompt(round1_outputs, max_chars=1500)
    conflicts_text = compact_json_for_prompt(
        _as_list(conflict_info.get("conflicts", []))[:8], max_chars=900
    )
    compact_agenda = {
        "family_id": negotiation_agenda.get("family_id"),
        "agenda_items": _as_list(negotiation_agenda.get("agenda_items", []))[:8],
    }
    agenda_text = compact_json_for_prompt(compact_agenda, max_chars=1200)
    system_prompt = "\nYou are the coordinator agent in a structured multi-agent family travel planning system.\nYour task is to create a compact initial draft plan for negotiation.\nOutput token limit: under 1200 tokens.\nYou must follow the negotiation agenda conflict by conflict.\nYou must output valid JSON only.\nDo not output markdown.\nDo not output explanations outside JSON.\n".strip()
    user_prompt = f"""\nCreate a compact initial draft itinerary for negotiation.\nOutput token limit: keep JSON under 1200 tokens.\nFamily trip:\nfamily_id: {family_id}\ndeparture_city: {first_agent.get('departure_city')}\ndestination_city: {first_agent.get('destination_city')}\nday: {first_agent.get('day')}\nquery: {first_agent.get('query')}\nfamily_structure: {first_agent.get('family_structure')}\nconflict_level: {first_agent.get('conflict_level')}\nRound 1 core needs and negotiation boundaries:\n{round1_text}\nDetected conflicts:\n{conflicts_text}\nStructured negotiation agenda:\n{agenda_text}\nImportant:\nThis is only an initial plan for negotiation.\nDo NOT generate a very long detailed itinerary.\nKeep each day concise.\nReturn JSON exactly in this structure:\n{{\n  "family_id": {family_id},\n  "draft_plan_id": "round2_initial_plan",\n  "itinerary_summary": "...",\n  "per_day_plan": [\n    {{\n      "day": 1,\n      "core_plan": "...",\n      "meal_strategy": "...",\n      "hotel_strategy": "...",\n      "transportation_strategy": "...",\n      "estimated_walking_distance_km": 0,\n      "estimated_return_time": "20:00",\n      "optional_activity": "...",\n      "rest_block": "..."\n    }}\n  ],\n  "budget_strategy": "...",\n  "main_tradeoffs": ["..."],\n  "conflict_handling_strategy": [\n    {{\n      "conflict_id": "C1",\n      "strategy": "shared_compromise / split_activity / optional_activity / remove_or_mark_unresolved",\n      "reason": "...",\n      "protected_agents": ["..."]\n    }}\n  ]\n}}\nRules:\n- Use the structured negotiation agenda as the main checklist.\n- Every high-priority agenda item must be addressed in conflict_handling_strategy.\n- Use split activities for split_satisfiable conflicts instead of forcing all members into the same activity.\n- For direct must_include vs must_avoid conflicts, include the activity only for the member who requires it and mark avoid members as non-participating.\n- Do not ask members to compromise on non_negotiable_items or must-have constraints.\n- For children and older members, keep walking distance and return time conservative.\n- Output token limit: keep the whole JSON under 1200 tokens.\n- Keep output short.\n""".strip()
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def coordinator_generate_initial_plan(
    family_id: int,
    family_agents: List[Dict[str, Any]],
    round1_outputs: List[Dict[str, Any]],
    conflict_info: Dict[str, Any],
    negotiation_agenda: Dict[str, Any],
    family_feedback: Dict[str, Any] = None,
) -> Dict[str, Any]:
    messages = build_initial_plan_messages(
        family_id=family_id,
        family_agents=family_agents,
        round1_outputs=round1_outputs,
        conflict_info=conflict_info,
        negotiation_agenda=negotiation_agenda,
    )
    messages = append_backtracking_feedback_to_messages(
        messages=messages,
        family_feedback=family_feedback or {},
        stage_name="Coordinator initial plan",
    )
    messages = budget_messages(messages, max_input_tokens=3600)
    try:
        api_result = call_generation_json(
            messages=messages,
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS_INITIAL_PLAN,
            temperature=TEMPERATURE_COORDINATOR,
        )
        data = api_result.get("data", {})
        usage = api_result.get("usage", {})
    except Exception as exc:
        print(f"[Module 4] family_id={family_id} 初版行程 LLM 失败，使用本地压缩兜底: {exc}")
        first_agent = family_agents[0]
        days = max(1, safe_int(first_agent.get("day"), 1))
        data = {
            "family_id": family_id,
            "draft_plan_id": "round2_initial_plan_fallback",
            "itinerary_summary": "Compact fallback draft for negotiation.",
            "per_day_plan": [
                {
                    "day": i,
                    "day_theme": "rough shared family day",
                    "morning_intent": "low-intensity cultural or scenic activity",
                    "afternoon_intent": "easy family activity or split activity if needed",
                    "evening_intent": "local food and early return",
                    "estimated_walking_distance_km": 3,
                    "estimated_return_time": "20:00",
                }
                for i in range(1, min(days, 5) + 1)
            ],
            "budget_strategy": "keep medium budget",
            "main_tradeoffs": [],
            "conflict_handling_strategy": [],
        }
        usage = {}
    data["usage"] = usage
    return data


def build_round2_messages(
    agent: Dict[str, Any],
    round1_output: Dict[str, Any],
    initial_plan: Dict[str, Any],
    memories: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    profile = compact_agent_profile_for_negotiation(agent)
    round1_text = compact_json_for_prompt(round1_output, max_chars=550)
    plan_text = compact_json_for_prompt(initial_plan, max_chars=900)
    system_prompt = "You are one persona agent evaluating a draft plan. Output compact valid JSON only. Output token limit: under 800 tokens."
    user_prompt = f"""
Round 2: evaluate draft plan from your own perspective.
Persona: {json.dumps(profile, ensure_ascii=False)}
Round1: {round1_text}
Draft plan: {plan_text}

Return JSON only:
{{
  "round": 2,
  "agent_id": "{agent.get('agent_id')}",
  "agent_name": "{profile.get('role')}",
  "objections_to_current_plan": [{{"objection": "short", "severity": "low/medium/high", "violation_type": "hard_constraint_violation/soft_preference_mismatch/feasibility_risk", "required_fix": "short", "will_reject_if_unfixed": false}}],
  "acceptable_concessions": [],
  "consensus_vote": "Accept / Reject / Accept with minor changes",
  "reason": "brief"
}}
Limits: max 2 objections. Each string under 10 words. Output token limit: whole output under 450 tokens.
""".strip()
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def run_round2_objections(
    family_agents: List[Dict[str, Any]],
    round1_outputs: List[Dict[str, Any]],
    initial_plan: Dict[str, Any],
    memory_store: RoleMemoryStore,
) -> List[Dict[str, Any]]:
    round1_lookup = {item["agent_id"]: item for item in round1_outputs}
    outputs = []
    for agent in family_agents:
        agent_id = agent["agent_id"]
        round1_output = round1_lookup.get(agent_id, {})
        memories = memory_store.load_memories(agent=agent, max_items=3)
        messages = build_round2_messages(
            agent=agent, round1_output=round1_output, initial_plan=initial_plan, memories=memories
        )
        try:
            api_result = call_generation_json(
                messages=messages,
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS_ROUND_AGENT,
                temperature=TEMPERATURE_AGENT,
            )
            data = normalize_round2_output(api_result.get("data", {}), agent)
            usage = api_result.get("usage", {})
        except Exception as exc:
            print(f"[Module 4] agent_id={agent_id} Round 2 LLM 失败，使用本地压缩兜底: {exc}")
            data = normalize_round2_output(
                {
                    "consensus_vote": "Accept with minor changes",
                    "reason": "Fallback compact evaluation.",
                },
                agent,
            )
            usage = {}
        data["agent_id"] = agent_id
        data["usage"] = usage
        outputs.append(data)
        memory_store.save_negotiation_memory(
            agent=agent, round_name="negotiation_round_2_objections", content=data
        )
        time.sleep(SLEEP_SECONDS)
    return outputs


def build_round3_messages(
    agent: Dict[str, Any],
    round1_output: Dict[str, Any],
    round2_output: Dict[str, Any],
    initial_plan: Dict[str, Any],
    memories: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    profile = compact_agent_profile_for_negotiation(agent)
    round1_text = compact_json_for_prompt(round1_output, max_chars=450)
    round2_text = compact_json_for_prompt(round2_output, max_chars=650)
    plan_text = compact_json_for_prompt(initial_plan, max_chars=800)
    system_prompt = "You are one persona agent giving compact concessions. Output valid JSON only. Output token limit: under 800 tokens."
    user_prompt = f"""
Round 3: give compact conditional concessions.
Persona: {json.dumps(profile, ensure_ascii=False)}
Round1: {round1_text}
Round2: {round2_text}
Draft plan: {plan_text}

Return JSON only:
{{
  "round": 3,
  "agent_id": "{agent.get('agent_id')}",
  "agent_name": "{profile.get('role')}",
  "acceptable_concessions": [{{"concession": "short", "concession_type": "soft_preference/schedule_adjustment/split_activity", "condition": "short", "priority_loss": "low/medium/high", "still_reject_if": "short"}}],
  "non_negotiable_items": ["max 2 bottom lines"],
  "consensus_vote": "Accept / Reject / Accept with minor changes",
  "reason": "brief"
}}
Limits: max 2 concessions. Each string under 10 words. Output token limit: whole output under 450 tokens.
""".strip()
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def run_round3_concessions(
    family_agents: List[Dict[str, Any]],
    round1_outputs: List[Dict[str, Any]],
    round2_outputs: List[Dict[str, Any]],
    initial_plan: Dict[str, Any],
    memory_store: RoleMemoryStore,
) -> List[Dict[str, Any]]:
    round1_lookup = {item["agent_id"]: item for item in round1_outputs}
    round2_lookup = {item["agent_id"]: item for item in round2_outputs}
    outputs = []
    for agent in family_agents:
        agent_id = agent["agent_id"]
        round1_output = round1_lookup.get(agent_id, {})
        round2_output = round2_lookup.get(agent_id, {})
        memories = memory_store.load_memories(agent=agent, max_items=3)
        messages = build_round3_messages(
            agent=agent,
            round1_output=round1_output,
            round2_output=round2_output,
            initial_plan=initial_plan,
            memories=memories,
        )
        try:
            api_result = call_generation_json(
                messages=messages,
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS_ROUND_AGENT,
                temperature=TEMPERATURE_AGENT,
            )
            data = normalize_round3_output(api_result.get("data", {}), agent)
            usage = api_result.get("usage", {})
        except Exception as exc:
            print(f"[Module 4] agent_id={agent_id} Round 3 LLM 失败，使用本地压缩兜底: {exc}")
            data = normalize_round3_output(
                {
                    "consensus_vote": "Accept with minor changes",
                    "reason": "Fallback compact concessions.",
                },
                agent,
            )
            usage = {}
        data["agent_id"] = agent_id
        data["usage"] = usage
        outputs.append(data)
        memory_store.save_negotiation_memory(
            agent=agent, round_name="negotiation_round_3_concessions", content=data
        )
        time.sleep(SLEEP_SECONDS)
    return outputs


def build_round4_messages(
    family_id: int,
    family_agents: List[Dict[str, Any]],
    initial_plan: Dict[str, Any],
    round1_outputs: List[Dict[str, Any]],
    round2_outputs: List[Dict[str, Any]],
    round3_outputs: List[Dict[str, Any]],
    conflict_info: Dict[str, Any],
    negotiation_agenda: Dict[str, Any],
) -> List[Dict[str, str]]:
    first_agent = family_agents[0]
    compact_agenda = {
        "family_id": negotiation_agenda.get("family_id"),
        "high_priority_conflict_count": negotiation_agenda.get("high_priority_conflict_count"),
        "split_required_conflict_count": negotiation_agenda.get("split_required_conflict_count"),
        "agenda_items": _as_list(negotiation_agenda.get("agenda_items", []))[:6],
    }
    agenda_text = compact_json_for_prompt(compact_agenda, max_chars=1800)
    round1_brief = []
    for item in _as_list(round1_outputs):
        if not isinstance(item, dict):
            continue
        round1_brief.append(
            {
                "agent_id": item.get("agent_id"),
                "agent_name": item.get("agent_name"),
                "non_negotiable_items": _as_list(item.get("non_negotiable_items", []))[:3],
                "negotiable_items": _as_list(item.get("negotiable_items", []))[:2],
                "minimum_acceptance_standard": safe_text(
                    item.get("minimum_acceptance_standard", "")
                )[:180],
            }
        )
    round2_brief = []
    for item in _as_list(round2_outputs):
        if not isinstance(item, dict):
            continue
        round2_brief.append(
            {
                "agent_id": item.get("agent_id"),
                "agent_name": item.get("agent_name"),
                "consensus_vote": item.get("consensus_vote"),
                "top_objections": _as_list(item.get("objections_to_current_plan", []))[:2],
            }
        )
    round3_brief = []
    for item in _as_list(round3_outputs):
        if not isinstance(item, dict):
            continue
        round3_brief.append(
            {
                "agent_id": item.get("agent_id"),
                "agent_name": item.get("agent_name"),
                "acceptable_concessions": _as_list(item.get("acceptable_concessions", []))[:2],
                "still_reject_if": _as_list(item.get("still_reject_if", []))[:2],
            }
        )
    initial_plan_brief = {
        "summary": safe_text(
            initial_plan.get("initial_itinerary_summary", initial_plan.get("summary", ""))
        )[:500],
        "potential_risks": _as_list(initial_plan.get("potential_risks", []))[:4],
        "split_activity_candidates": _as_list(initial_plan.get("split_activity_candidates", []))[
            :4
        ],
    }
    system_prompt = "\nYou are the coordinator agent in a structured multi-agent family travel planning system.\nOutput a compact final consensus direction only.\nOutput token limit: under 1500 tokens.\nDo not generate a full detailed itinerary here; Module 5 will generate it.\nYou must output valid JSON only.\nNo markdown. No text outside JSON.\n".strip()
    user_prompt = f"""\nRound 4: Compact Final Consensus Direction.\nOutput token limit: keep JSON under 1500 tokens.\n\nFamily trip:\nfamily_id: {family_id}\ndeparture_city: {first_agent.get('departure_city')}\ndestination_city: {first_agent.get('destination_city')}\nday_count: {first_agent.get('day')}\nquery: {first_agent.get('query')}\nfamily_structure: {first_agent.get('family_structure')}\nconflict_level: {first_agent.get('conflict_level')}\n\nStructured negotiation agenda, top items only:\n{agenda_text}\n\nInitial draft plan brief:\n{compact_json_for_prompt(initial_plan_brief, max_chars=900)}\n\nRound 1 brief:\n{compact_json_for_prompt(round1_brief, max_chars=1200)}\n\nRound 2 objections brief:\n{compact_json_for_prompt(round2_brief, max_chars=1400)}\n\nRound 3 concessions brief:\n{compact_json_for_prompt(round3_brief, max_chars=1200)}\n\nOutput requirements:\n- Output token limit: keep the whole JSON under 1500 tokens.\n- Use short phrases, not paragraphs.\n- Do not repeat all member constraints.\n- core_consensus_plan must contain at most {min(safe_int(first_agent.get('day'), 3), 5)} day items.\n- Each day must give a morning_activity and an afternoon_activity: two CONCRETE real POI names of the destination city (real attraction/park/museum/street/landmark from members' must_include or well-known places). NEVER vague themes or categories.\n- conflict_resolution_map has at most 4 items.\n- split_activity_recommendations has at most 3 items.\n- unresolved_conflicts has at most 3 items.\n- agent_votes must include exactly one vote per family member.\n\nReturn JSON exactly in this compact structure:\n{{\n  "round": 4,\n  "family_id": {family_id},\n  "final_plan_id": "final_consensus_direction",\n  "final_itinerary_summary": "one short sentence",\n  "core_consensus_plan": [\n    {{\n      "day": 1,\n      "morning_activity": "concrete POI name for morning",\n      "afternoon_activity": "concrete POI name for afternoon",\n      "policy": "walking/return-time policy"\n    }}\n  ],\n  "budget_strategy": "short budget strategy",\n  "hotel_strategy": "short hotel strategy",\n  "transportation_strategy": "short transportation strategy",\n  "conflict_resolution_map": [\n    {{\n      "conflict_id": "C1",\n      "resolution_type": "shared_activity / split_activity / optional_activity / removed / left_unresolved",\n      "protected_agents": ["agent_id"],\n      "remaining_risk": "low/medium/high"\n    }}\n  ],\n  "resolved_conflicts": [\n    {{"conflict_id": "C1", "resolution_type": "split_activity", "resolution": "short phrase"}}\n  ],\n  "unresolved_conflicts": [\n    {{"conflict_id": "C2", "severity": "low/medium/high", "reason": "short reason"}}\n  ],\n  "split_activity_recommendations": [\n    {{\n      "conflict_id": "C1",\n      "activity_or_term": "short term",\n      "participating_agents": ["agent_id"],\n      "non_participating_agents": ["agent_id"],\n      "alternative_for_non_participants": "specific nearby alternative"\n    }}\n  ],\n  "rejected_member_protection": [\n    {{"agent_id": "agent_id", "main_objection": "short", "fix": "short", "final_vote_expected": "Accept"}}\n  ],\n  "agent_votes": [\n    {{"agent_id": "agent_id", "agent_name": "role", "vote": "Accept / Reject / Accept with minor changes", "reason": "short"}}\n  ],\n  "consensus_status": "Full Consensus / Partial Consensus / No Consensus",\n  "number_of_objections_resolved": 0,\n  "acceptance_vote_ratio": 0.0,\n  "coordinator_reasoning_summary": "one short sentence"\n}}\n\nConsensus rules:\n- Full Consensus: all final votes are Accept or Accept with minor changes, and no unresolved conflicts.\n- Partial Consensus: all/most members can accept, but some unresolved conflicts or risks remain.\n- No Consensus: at least one final vote is Reject, or acceptance ratio is below 0.67.\n- Do not mark No Consensus only because someone rejected the initial draft in Round 2; judge the final direction.\n""".strip()
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def build_round4_ultra_compact_messages(
    family_id: int, family_agents: List[Dict[str, Any]], negotiation_agenda: Dict[str, Any]
) -> List[Dict[str, str]]:
    first_agent = family_agents[0]
    agents_brief = []
    for agent in family_agents:
        profile = agent.get("profile", {}) if isinstance(agent, dict) else {}
        constraints = agent.get("constraints", {}) if isinstance(agent, dict) else {}
        agents_brief.append(
            {
                "agent_id": agent.get("agent_id") if isinstance(agent, dict) else "",
                "role": profile.get("role"),
                "must_include": _as_list(constraints.get("must_include", []))[:3],
                "must_avoid": _as_list(constraints.get("must_avoid", []))[:3],
                "max_walking_distance_km": constraints.get("max_walking_distance_km"),
                "latest_return_time": constraints.get("latest_return_time"),
            }
        )
    agenda_brief = (
        _as_list(negotiation_agenda.get("agenda_items", []))[:5]
        if isinstance(negotiation_agenda, dict)
        else []
    )
    user_prompt = f"""
Return valid JSON only. Create an ultra-compact final consensus direction.
Output token limit: under 1200 tokens.
Do not generate a detailed itinerary; Module 5 will do that.

family_id: {family_id}
trip: {first_agent.get('departure_city')} to {first_agent.get('destination_city')}, {first_agent.get('day')} days
agents: {compact_json_for_prompt(agents_brief, max_chars=1200)}
conflicts: {compact_json_for_prompt(agenda_brief, max_chars=900)}

Return JSON only:
{{
  "round": 4,
  "family_id": {family_id},
  "final_plan_id": "final_consensus_direction",
  "final_itinerary_summary": "short summary",
  "core_consensus_plan": [{{"day": 1, "morning_activity": "concrete POI", "afternoon_activity": "concrete POI", "policy": "short"}}],
  "budget_strategy": "short",
  "hotel_strategy": "short",
  "transportation_strategy": "short",
  "conflict_resolution_map": [{{"conflict_id": "C1", "resolution_type": "split_activity", "protected_agents": ["agent_id"], "remaining_risk": "low"}}],
  "split_activity_recommendations": [],
  "unresolved_conflicts": [],
  "agent_votes": [{{"agent_id": "agent_id", "agent_name": "role", "vote": "Accept", "reason": "short"}}],
  "consensus_status": "Full Consensus / Partial Consensus / No Consensus",
  "number_of_objections_resolved": 0,
  "acceptance_vote_ratio": 1.0,
  "coordinator_reasoning_summary": "short"
}}
""".strip()
    messages = [
        {
            "role": "system",
            "content": "You output compact valid JSON only. Output token limit: under 1200 tokens. No markdown.",
        },
        {"role": "user", "content": user_prompt},
    ]
    return budget_messages(messages, max_input_tokens=2200)


def coordinator_generate_final_plan(
    family_id: int,
    family_agents: List[Dict[str, Any]],
    initial_plan: Dict[str, Any],
    round1_outputs: List[Dict[str, Any]],
    round2_outputs: List[Dict[str, Any]],
    round3_outputs: List[Dict[str, Any]],
    conflict_info: Dict[str, Any],
    negotiation_agenda: Dict[str, Any],
    family_feedback: Dict[str, Any] = None,
) -> Dict[str, Any]:
    messages = build_round4_messages(
        family_id=family_id,
        family_agents=family_agents,
        initial_plan=initial_plan,
        round1_outputs=round1_outputs,
        round2_outputs=round2_outputs,
        round3_outputs=round3_outputs,
        conflict_info=conflict_info,
        negotiation_agenda=negotiation_agenda,
    )
    messages = append_backtracking_feedback_to_messages(
        messages=messages,
        family_feedback=family_feedback or {},
        stage_name="Round 4 final consensus direction",
    )
    messages = budget_messages(messages, max_input_tokens=4200)
    try:
        api_result = call_generation_json(
            messages=messages,
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS_FINAL_PLAN,
            temperature=TEMPERATURE_COORDINATOR,
        )
        data = api_result.get("data", {})
        usage = api_result.get("usage", {})
    except Exception as exc:
        print(
            f"[Module 4] family_id={family_id} Round 4 压缩版失败，尝试 ultra-compact LLM fallback。错误: {exc}"
        )
        try:
            fallback_messages = build_round4_ultra_compact_messages(
                family_id=family_id,
                family_agents=family_agents,
                negotiation_agenda=negotiation_agenda,
            )
            api_result = call_generation_json(
                messages=fallback_messages,
                model=MODEL_NAME,
                max_tokens=1400,
                temperature=0.15,
                max_retries=1,
            )
            data = api_result.get("data", {})
            usage = api_result.get("usage", {})
            if not isinstance(data, dict):
                raise ValueError("ultra-compact fallback did not return a JSON object")
            data["fallback_used"] = True
            data["fallback_type"] = "ultra_compact_llm"
        except Exception as fallback_exc:
            print(
                f"[Module 4] family_id={family_id} ultra-compact LLM fallback 仍失败，使用本地短兜底。错误: {fallback_exc}"
            )
            first_agent = family_agents[0]
            days = max(1, safe_int(first_agent.get("day"), 1))
            # Seed the fallback consensus with real must_include POIs (cycled across days) so the plan
            # names concrete places instead of '轻松家庭行程'; Module 5 + snapshot candidates fill the rest.
            fb_must_pois = []
            for a in family_agents:
                cons = a.get("constraints", {}) if isinstance(a, dict) else {}
                for x in _as_list(cons.get("must_include", [])):
                    xs = safe_text(x)
                    if xs and xs not in fb_must_pois:
                        fb_must_pois.append(xs)
            data = {
                "round": 4,
                "family_id": family_id,
                "final_plan_id": "final_consensus_direction_fallback",
                "final_itinerary_summary": "Local compact consensus direction.",
                "core_consensus_plan": [
                    {
                        "day": i,
                        "morning_activity": fb_must_pois[((i - 1) * 2) % len(fb_must_pois)]
                        if fb_must_pois
                        else "当地核心景点",
                        "afternoon_activity": fb_must_pois[((i - 1) * 2 + 1) % len(fb_must_pois)]
                        if fb_must_pois
                        else "当地知名公园",
                        "policy": "respect walking and return limits",
                    }
                    for i in range(1, min(days, 5) + 1)
                ],
                "budget_strategy": "medium budget",
                "hotel_strategy": "near transit",
                "transportation_strategy": "short transfers",
                "conflict_resolution_map": [],
                "split_activity_recommendations": [],
                "unresolved_conflicts": [],
                "agent_votes": [
                    {
                        "agent_id": a.get("agent_id"),
                        "agent_name": (a.get("profile") or {}).get("role"),
                        "vote": "Accept with minor changes",
                        "reason": "local fallback",
                    }
                    for a in family_agents
                ],
                "consensus_status": "Partial Consensus",
                "number_of_objections_resolved": 0,
                "acceptance_vote_ratio": 1.0,
                "coordinator_reasoning_summary": "LLM failed; local short fallback used.",
                "fallback_used": True,
                "fallback_type": "local_deterministic",
            }
            usage = {}
    if not isinstance(data, dict):
        data = {}
    data["usage"] = usage
    return data


def build_no_negotiation_result(
    family_id: int,
    family_agents: List[Dict[str, Any]],
    proposal_lookup: Dict[str, Dict[str, Any]],
    conflict_info: Dict[str, Any],
) -> Dict[str, Any]:
    """消融 no_negotiation（真·无协商）：不做任何结构化协商——无议程、无多轮异议/让步、
    无 coordinator 共识合成。把各成员 must_include 的朴素并集作为"未消解方向"，连同 M2
    原始提案 + M3 冲突一起交给 M5，由 M5 自行合成统一行程（朴素聚合，无协商修正）。所有
    检测到的冲突原样留在 unresolved_conflicts（协商没有消解它们），consensus_status=No Negotiation。"""
    first = family_agents[0]
    days = max(1, safe_int(first.get("day"), 1))
    must_pois: List[str] = []
    for a in family_agents:
        cons = a.get("constraints", {}) if isinstance(a, dict) else {}
        for x in _as_list(cons.get("must_include", [])):
            xs = safe_text(x)
            if xs and xs not in must_pois:
                must_pois.append(xs)
    core_plan = [
        {
            "day": i,
            "morning_activity": must_pois[((i - 1) * 2) % len(must_pois)]
            if must_pois
            else "当地核心景点",
            "afternoon_activity": must_pois[((i - 1) * 2 + 1) % len(must_pois)]
            if must_pois
            else "当地知名公园",
            "policy": "raw aggregation of member demands; no negotiation / no conflict resolution",
        }
        for i in range(1, min(days, 5) + 1)
    ]
    conflicts = conflict_info.get("conflicts", []) if isinstance(conflict_info, dict) else []
    final_plan = {
        "round": 4,
        "family_id": family_id,
        "final_plan_id": "no_negotiation_raw_aggregation",
        "final_itinerary_summary": "无协商：各成员提案直接聚合，未经议程/异议/让步/共识合成。",
        "core_consensus_plan": core_plan,
        "budget_strategy": "",
        "hotel_strategy": "",
        "transportation_strategy": "",
        "conflict_resolution_map": [],
        "split_activity_recommendations": [],
        "unresolved_conflicts": conflicts,
        "agent_votes": [],
        "consensus_status": "No Negotiation",
        "number_of_objections_resolved": 0,
        "acceptance_vote_ratio": "",
        "forced_consensus_downgrade": False,
        "coordinator_reasoning_summary": "Negotiation ablated (no_negotiation): raw member proposals passed to Module 5.",
        "negotiation_ablated": True,
    }
    return {
        "family_id": family_id,
        "departure_city": first.get("departure_city", ""),
        "destination_city": first.get("destination_city", ""),
        "day": first.get("day", ""),
        "query": first.get("query", ""),
        "family_structure": first.get("family_structure", ""),
        "conflict_level": first.get("conflict_level", ""),
        "negotiation_rounds": 0,
        "negotiation_ablated": True,
        "negotiation_agenda": {},
        "negotiation_quality": {},
        "round1_core_needs": [],
        "round2_initial_plan": {},
        "round2_objections": [],
        "round3_concessions": [],
        "round4_final_plan": final_plan,
        "consensus_status": "No Negotiation",
        "acceptance_vote_ratio": "",
        "number_of_objections_resolved": 0,
        "negotiation_quality_score": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "status": "success",
    }


def run_negotiation_for_family(
    family_id: int,
    family_agents: List[Dict[str, Any]],
    proposal_lookup: Dict[str, Dict[str, Any]],
    conflict_lookup: Dict[int, Dict[str, Any]],
    memory_store: RoleMemoryStore,
    backtracking_feedback_lookup: Dict[int, Dict[str, Any]] = None,
    enable_negotiation: bool = True,
) -> Dict[str, Any]:
    conflict_info = conflict_lookup.get(family_id, {"family_id": family_id, "conflicts": []})
    if not enable_negotiation:
        # 消融 no_negotiation（真·无协商）：完全跳过 M4 结构化协商——无议程、无 coordinator
        # 共识合成、无多轮异议/让步。把各成员原始提案直接交给 M5 自行合成（朴素聚合）。
        # 保留 M5 的分流机制（那是 no_split 的维度），但没有协商引导，分流会自然减少。
        print(f"\n[Module 4] family_id={family_id} [消融 no_negotiation] 真·无协商：跳过议程/轮次/共识合成，提案直传 M5")
        return build_no_negotiation_result(
            family_id=family_id,
            family_agents=family_agents,
            proposal_lookup=proposal_lookup,
            conflict_info=conflict_info,
        )
    print(f"\n[Module 4] 开始 family_id={family_id} 的 4 轮协商")
    family_feedback = (backtracking_feedback_lookup or {}).get(int(family_id), {})
    if family_feedback:
        print(f"[Module 4] family_id={family_id} 检测到上一轮失败反馈，将加入协商 prompt。")
    negotiation_agenda = build_negotiation_agenda(
        conflict_info=conflict_info, family_agents=family_agents
    )
    print(
        f"[Module 4] family_id={family_id} 结构化协商议程生成完成: total_conflicts={negotiation_agenda.get('total_conflicts')}, high_priority={negotiation_agenda.get('high_priority_conflict_count')}, split_required={negotiation_agenda.get('split_required_conflict_count')}"
    )
    round1_outputs = run_round1_core_needs(
        family_agents=family_agents,
        proposal_lookup=proposal_lookup,
        memory_store=memory_store,
        family_feedback=family_feedback,
    )
    print(f"[Module 4] family_id={family_id} Round 1 完成")
    initial_plan = coordinator_generate_initial_plan(
        family_id=family_id,
        family_agents=family_agents,
        round1_outputs=round1_outputs,
        conflict_info=conflict_info,
        negotiation_agenda=negotiation_agenda,
        family_feedback=family_feedback,
    )
    print(f"[Module 4] family_id={family_id} 初版行程生成完成")
    round2_outputs = run_round2_objections(
        family_agents=family_agents,
        round1_outputs=round1_outputs,
        initial_plan=initial_plan,
        memory_store=memory_store,
    )
    print(f"[Module 4] family_id={family_id} Round 2 完成")
    round3_outputs = run_round3_concessions(
        family_agents=family_agents,
        round1_outputs=round1_outputs,
        round2_outputs=round2_outputs,
        initial_plan=initial_plan,
        memory_store=memory_store,
    )
    print(f"[Module 4] family_id={family_id} Round 3 完成")
    final_plan = coordinator_generate_final_plan(
        family_id=family_id,
        family_agents=family_agents,
        initial_plan=initial_plan,
        round1_outputs=round1_outputs,
        round2_outputs=round2_outputs,
        round3_outputs=round3_outputs,
        conflict_info=conflict_info,
        negotiation_agenda=negotiation_agenda,
        family_feedback=family_feedback,
    )
    negotiation_quality = compute_negotiation_quality_score(
        round2_outputs=round2_outputs,
        round3_outputs=round3_outputs,
        final_plan=final_plan,
        negotiation_agenda=negotiation_agenda,
        family_agents=family_agents,
    )
    final_plan = enforce_consensus_status(final_plan=final_plan, quality=negotiation_quality)
    final_plan["negotiation_quality"] = negotiation_quality
    print(
        f"[Module 4] family_id={family_id} Round 4 最终共识方向完成，negotiation_quality_score={negotiation_quality.get('negotiation_quality_score')}, consensus_status={final_plan.get('consensus_status')}"
    )
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    usage_sources = round1_outputs + [initial_plan] + round2_outputs + round3_outputs + [final_plan]
    for item in usage_sources:
        usage = item.get("usage", {}) or {}
        total_prompt_tokens += safe_int(usage.get("prompt_tokens"), 0)
        total_completion_tokens += safe_int(usage.get("completion_tokens"), 0)
        total_tokens += safe_int(usage.get("total_tokens"), 0)
    result = {
        "family_id": family_id,
        "departure_city": family_agents[0].get("departure_city", ""),
        "destination_city": family_agents[0].get("destination_city", ""),
        "day": family_agents[0].get("day", ""),
        "query": family_agents[0].get("query", ""),
        "family_structure": family_agents[0].get("family_structure", ""),
        "conflict_level": family_agents[0].get("conflict_level", ""),
        "negotiation_rounds": 4,
        "negotiation_ablated": False,
        "negotiation_agenda": negotiation_agenda,
        "negotiation_quality": negotiation_quality,
        "round1_core_needs": round1_outputs,
        "round2_initial_plan": initial_plan,
        "round2_objections": round2_outputs,
        "round3_concessions": round3_outputs,
        "round4_final_plan": final_plan,
        "consensus_status": final_plan.get("consensus_status", ""),
        "acceptance_vote_ratio": final_plan.get("acceptance_vote_ratio", ""),
        "number_of_objections_resolved": final_plan.get("number_of_objections_resolved", ""),
        "negotiation_quality_score": negotiation_quality.get("negotiation_quality_score"),
        "high_severity_objection_count": negotiation_quality.get("high_severity_objection_count"),
        "hard_constraint_objection_count": negotiation_quality.get(
            "hard_constraint_objection_count"
        ),
        "unresolved_conflict_count": negotiation_quality.get("unresolved_conflict_count"),
        "high_unresolved_conflict_count": negotiation_quality.get("high_unresolved_conflict_count"),
        "split_required_conflict_count": negotiation_quality.get("split_required_conflict_count"),
        "split_required_conflict_coverage_score": negotiation_quality.get(
            "split_required_conflict_coverage_score"
        ),
        "forced_consensus_downgrade": final_plan.get("forced_consensus_downgrade", False),
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "status": "success",
    }
    return result


def flatten_negotiation_result(result: Dict[str, Any]) -> Dict[str, Any]:
    final_plan = result.get("round4_final_plan", {})
    negotiation_quality = (
        result.get("negotiation_quality", {}) or final_plan.get("negotiation_quality", {}) or {}
    )
    return {
        "family_id": result.get("family_id"),
        "departure_city": result.get("departure_city"),
        "destination_city": result.get("destination_city"),
        "day": result.get("day"),
        "query": result.get("query"),
        "family_structure": result.get("family_structure"),
        "conflict_level": result.get("conflict_level"),
        "negotiation_rounds": result.get("negotiation_rounds"),
        "consensus_status": result.get("consensus_status"),
        "acceptance_vote_ratio": result.get("acceptance_vote_ratio"),
        "number_of_objections_resolved": result.get("number_of_objections_resolved"),
        "negotiation_quality_score": negotiation_quality.get("negotiation_quality_score"),
        "hard_conflict_resolution_score": negotiation_quality.get("hard_conflict_resolution_score"),
        "high_severity_objection_resolution_score": negotiation_quality.get(
            "high_severity_objection_resolution_score"
        ),
        "split_required_conflict_coverage_score": negotiation_quality.get(
            "split_required_conflict_coverage_score"
        ),
        "concession_quality_score": negotiation_quality.get("concession_quality_score"),
        "rejected_member_protection_score": negotiation_quality.get(
            "rejected_member_protection_score"
        ),
        "high_severity_objection_count": negotiation_quality.get("high_severity_objection_count"),
        "hard_constraint_objection_count": negotiation_quality.get(
            "hard_constraint_objection_count"
        ),
        "unresolved_conflict_count": negotiation_quality.get("unresolved_conflict_count"),
        "high_unresolved_conflict_count": negotiation_quality.get("high_unresolved_conflict_count"),
        "split_required_conflict_count": negotiation_quality.get("split_required_conflict_count"),
        "rejected_member_count": negotiation_quality.get("rejected_member_count"),
        "forced_consensus_downgrade": final_plan.get("forced_consensus_downgrade", False),
        "negotiation_agenda_json": json.dumps(
            result.get("negotiation_agenda", {}), ensure_ascii=False
        ),
        "final_itinerary_summary": final_plan.get("final_itinerary_summary", ""),
        "core_consensus_plan_json": json.dumps(
            final_plan.get("core_consensus_plan", []), ensure_ascii=False
        ),
        "budget_plan_json": json.dumps(final_plan.get("budget_plan", {}), ensure_ascii=False),
        "hotel_plan_json": json.dumps(final_plan.get("hotel_plan", {}), ensure_ascii=False),
        "transportation_plan_json": json.dumps(
            final_plan.get("transportation_plan", {}), ensure_ascii=False
        ),
        "resolved_conflicts_json": json.dumps(
            final_plan.get("resolved_conflicts", []), ensure_ascii=False
        ),
        "unresolved_conflicts_json": json.dumps(
            final_plan.get("unresolved_conflicts", []), ensure_ascii=False
        ),
        "conflict_resolution_map_json": json.dumps(
            final_plan.get("conflict_resolution_map", []), ensure_ascii=False
        ),
        "split_activity_recommendations_json": json.dumps(
            final_plan.get("split_activity_recommendations", []), ensure_ascii=False
        ),
        "agent_votes_json": json.dumps(final_plan.get("agent_votes", []), ensure_ascii=False),
        "rejected_member_protection_json": json.dumps(
            final_plan.get("rejected_member_protection", []), ensure_ascii=False
        ),
        "coordinator_reasoning_summary": final_plan.get("coordinator_reasoning_summary", ""),
        "prompt_tokens": result.get("prompt_tokens"),
        "completion_tokens": result.get("completion_tokens"),
        "total_tokens": result.get("total_tokens"),
        "status": result.get("status", ""),
    }


def build_error_result(family_id: int, error: Exception) -> Dict[str, Any]:
    return {"family_id": family_id, "status": "failed", "error": str(error)}


def save_negotiation_results(
    all_results: List[Dict[str, Any]],
    all_flat_rows: List[Dict[str, Any]],
    output_csv_path: Path,
    output_json_path: Path,
):
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(all_flat_rows)
    if "family_id" in result_df.columns:
        result_df = result_df.sort_values(by=["family_id"])
    result_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    sorted_results = sorted(all_results, key=lambda x: safe_int(x.get("family_id"), 0))
    save_json_file(data=sorted_results, path=output_json_path)


def print_negotiation_preview(result: Dict[str, Any]):
    print("=" * 100)
    print(f"family_id: {result.get('family_id')}")
    print(f"destination_city: {result.get('destination_city')}")
    print(f"consensus_status: {result.get('consensus_status')}")
    print(f"acceptance_vote_ratio: {result.get('acceptance_vote_ratio')}")
    print(f"number_of_objections_resolved: {result.get('number_of_objections_resolved')}")
    print(f"negotiation_quality_score: {result.get('negotiation_quality_score')}")
    print(f"high_severity_objection_count: {result.get('high_severity_objection_count')}")
    print(f"unresolved_conflict_count: {result.get('unresolved_conflict_count')}")
    print(f"total_tokens: {result.get('total_tokens')}")
    final_plan = result.get("round4_final_plan", {})
    print("\nfinal_itinerary_summary:")
    print(final_plan.get("final_itinerary_summary", ""))
    print("\nagent_votes:")
    print(json.dumps(final_plan.get("agent_votes", []), ensure_ascii=False, indent=2))
    print("=" * 100)


def run_negotiation_protocol(
    agents_json_path: Path = AGENTS_JSON,
    proposals_csv_path: Path = PROPOSALS_CSV,
    conflicts_json_path: Path = CONFLICTS_JSON,
    output_csv_path: Path = OUTPUT_CSV,
    output_json_path: Path = OUTPUT_JSON,
    memory_dir: Path = MEMORY_DIR,
    test_first_n_families: Optional[int] = 3,
    max_workers_families: int = 2,
    backtracking_feedback_json_path: Path = BACKTRACKING_FEEDBACK_JSON,
    target_family_ids: Optional[List[int]] = None,
    enable_negotiation: bool = True,
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
    print("[Module 4] Negotiation Protocol 开始")
    print(
        f"[Module 4] enable_negotiation（多轮协商）: {enable_negotiation}"
        + (
            ""
            if enable_negotiation
            else "  →  消融 no_negotiation：真·无协商（无议程/无轮次/无 coordinator 共识合成，提案直传 M5）"
        )
    )
    print(f"[Module 4] agents_json_path: {agents_json_path}")
    print(f"[Module 4] proposals_csv_path: {proposals_csv_path}")
    print(f"[Module 4] conflicts_json_path: {conflicts_json_path}")
    print(f"[Module 4] memory_dir: {memory_dir}")
    agents = load_json_file(agents_json_path)
    family_agents_map = build_family_agents(agents)
    proposal_lookup = load_individual_proposals(proposals_csv_path)
    conflict_lookup = load_conflict_results(conflicts_json_path)
    backtracking_feedback_lookup = load_backtracking_feedback_lookup(
        backtracking_feedback_json_path
    )
    if backtracking_feedback_lookup:
        print(f"[Module 4] 已加载 backtracking feedback 家庭数量: {len(backtracking_feedback_lookup)}")
    memory_store = RoleMemoryStore(memory_dir)
    family_ids = get_valid_family_ids(
        family_agents_map=family_agents_map,
        proposal_lookup=proposal_lookup,
        conflict_lookup=conflict_lookup,
    )
    if target_family_ids is not None:
        target_set = {int(x) for x in target_family_ids}
        family_ids = [fid for fid in family_ids if int(fid) in target_set]
        print(f"[Module 4] Backtracking target mode: 只重新协商未达标家庭 family_id={family_ids}")
    elif test_first_n_families is not None:
        family_ids = family_ids[:test_first_n_families]
    print(f"[Module 4] 实际运行家庭数量: {len(family_ids)}")
    print(f"[Module 4] max_workers_families: {max_workers_families}")
    all_results = []
    all_flat_rows = []

    def worker(family_id: int) -> Dict[str, Any]:
        family_agents = family_agents_map[family_id]
        try:
            result = run_negotiation_for_family(
                family_id=family_id,
                family_agents=family_agents,
                proposal_lookup=proposal_lookup,
                conflict_lookup=conflict_lookup,
                memory_store=memory_store,
                backtracking_feedback_lookup=backtracking_feedback_lookup,
                enable_negotiation=enable_negotiation,
            )
            return result
        except Exception as e:
            print(f"[Module 4] family_id={family_id} 协商失败: {e}")
            return build_error_result(family_id=family_id, error=e)

    with ThreadPoolExecutor(max_workers=max_workers_families) as executor:
        future_to_family_id = {
            executor.submit(worker, family_id): family_id for family_id in family_ids
        }
        for index, future in enumerate(as_completed(future_to_family_id), start=1):
            family_id = future_to_family_id[future]
            print(f"\n[Module 4] 完成第 {index}/{len(family_ids)} 个家庭协商，family_id={family_id}")
            result = future.result()
            all_results.append(result)
            if result.get("status") == "success":
                flat_row = flatten_negotiation_result(result)
                all_flat_rows.append(flat_row)
                if index <= 3:
                    print_negotiation_preview(result)
            else:
                all_flat_rows.append(
                    {"family_id": family_id, "status": "failed", "error": result.get("error", "")}
                )
            save_negotiation_results(
                all_results=all_results,
                all_flat_rows=all_flat_rows,
                output_csv_path=output_csv_path,
                output_json_path=output_json_path,
            )
            time.sleep(SLEEP_SECONDS)
    save_negotiation_results(
        all_results=all_results,
        all_flat_rows=all_flat_rows,
        output_csv_path=output_csv_path,
        output_json_path=output_json_path,
    )
    result_df = pd.DataFrame(all_flat_rows)
    if "family_id" in result_df.columns:
        result_df = result_df.sort_values(by=["family_id"])
    success_count = sum((1 for item in all_results if item.get("status") == "success"))
    failed_count = sum((1 for item in all_results if item.get("status") == "failed"))
    print("[Module 4] Negotiation Protocol 完成")
    print(f"[Module 4] CSV 输出文件: {output_csv_path}")
    print(f"[Module 4] JSON 输出文件: {output_json_path}")
    print(f"[Module 4] 成功家庭数量: {success_count}")
    print(f"[Module 4] 失败家庭数量: {failed_count}")
    return result_df
