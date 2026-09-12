"""Module 2 - Individual proposal generation: each persona agent proposes its own itinerary preferences (optionally guided by backtracking feedback)."""
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
    compact_feedback_for_llm,
    compact_json_for_prompt,
    compact_memory_items,
    load_backtracking_feedback_lookup,
    load_json_file,
    safe_int,
    safe_text,
    save_json_file,
)
from src.config import MEMORY_DIR
from src.llm_client import call_generation_json
from src.module2_5_individual_proposal_evaluation import (
    PROPOSAL_PASS_THRESHOLD,
    PROPOSAL_STRONG_THRESHOLD,
    evaluate_individual_proposal,
    memory_payload_from_evaluation,
)
from src.prompts import build_individual_proposal_messages_template

AGENTS_JSON = RESULTS_DIR / "module_1" / "persona_agents.json"
OUTPUT_CSV = RESULTS_DIR / "module_2" / "individual_agent_proposals.csv"
OUTPUT_JSON = RESULTS_DIR / "module_2" / "individual_agent_proposals.json"
BACKTRACKING_FEEDBACK_JSON = RESULTS_DIR / "module_7" / "backtracking_to_module2_feedback.json"
MODEL_NAME = os.getenv(
    "OPENROUTER_GENERATION_MODEL",
    os.getenv("OPENROUTER_MODEL_NAME", os.getenv("GENERATION_MODEL_NAME", "openai/gpt-4o-mini")),
)
MAX_TOKENS_PROPOSAL = 2000
TEMPERATURE = 0.25
SLEEP_SECONDS = 0.1
ENABLE_PROPOSAL_QUALITY_GATE = True
PROPOSAL_MAX_REGEN_ATTEMPTS = 2
MAX_MEMORY_ITEMS_FOR_PROMPT = 2


# -----------------------------------------------------------------------------
# Backtracking-aware proposal regeneration
# -----------------------------------------------------------------------------
# Inspired by Reflexion / Self-Refine style verbal feedback and Tree-of-Thought
# candidate diversification: when a family returns from Module 7, Module 2 should
# not simply regenerate the same personal proposals.  It should read the latest
# structured failure profile and force each persona to make clearer hard
# constraints, concessions, and acceptance conditions for the next negotiation.


def compact_backtracking_feedback_for_agent(
    agent: Dict[str, Any], family_feedback: Dict[str, Any], max_chars: int = 1000
) -> str:
    if not isinstance(family_feedback, dict) or not family_feedback:
        return ""
    agent_id = safe_text(agent.get("agent_id"))
    member_id = safe_int(agent.get("member_id"), 0)
    profile = agent.get("profile", {}) if isinstance(agent, dict) else {}
    role = safe_text(profile.get("role"))
    related_bad_proposals = []
    for item in family_feedback.get("bad_individual_proposals", []) or []:
        if (
            safe_text(item.get("agent_id")) == agent_id
            or safe_int(item.get("member_id"), -1) == member_id
            or safe_text(item.get("role")) == role
        ):
            related_bad_proposals.append(item)
    payload = {
        "trigger_rule": family_feedback.get("trigger_rule"),
        "revision_rounds_completed": family_feedback.get("revision_rounds_completed"),
        "final_failed_reasons": (family_feedback.get("final_failed_reasons") or [])[:5],
        "failure_profile": family_feedback.get("backtracking_failure_profile")
        or family_feedback.get("failure_profile")
        or {},
        "lowest_satisfaction_member": family_feedback.get("lowest_satisfaction_member", {}),
        "rejected_members": (family_feedback.get("rejected_members") or [])[:3],
        "hard_failed_members": (family_feedback.get("hard_failed_members") or [])[:3],
        "this_agent_previous_proposal_issues": related_bad_proposals[:2],
        "feedback_for_module2": (family_feedback.get("feedback_for_module2") or [])[:5],
        "recommended_regeneration_strategy": family_feedback.get(
            "recommended_regeneration_strategy"
        )
        or family_feedback.get("suggested_module2_strategy")
        or "",
    }
    text = compact_json_for_prompt(payload, max_chars=max_chars)
    return compact_feedback_for_llm(text, max_chars=max_chars)


def append_backtracking_feedback_to_proposal_messages(
    messages: List[Dict[str, str]], agent: Dict[str, Any], family_feedback: Dict[str, Any]
) -> List[Dict[str, str]]:
    feedback_text = compact_backtracking_feedback_for_agent(
        agent=agent, family_feedback=family_feedback
    )
    if not feedback_text:
        return messages
    guidance = (
        "\n\nBACKTRACKING MODE: this family failed in a previous full planning cycle.\n"
        + "Use the prior failure profile to produce a more useful personal proposal for negotiation.\n"
        + "Rules for this proposal:\n"
        + "- State non-negotiable hard constraints exactly, especially must_include, must_avoid, max walking, latest return time.\n"
        + "- Add specific objection_triggers that would make you reject the final plan.\n"
        + "- Add concrete negotiable_parts and acceptable split_activity alternatives.\n"
        + "- Protect lowest-satisfaction / rejected / hard-failed members from the previous cycle.\n"
        + "- Do not repeat the previous failure pattern; make the proposal shorter but more diagnostic.\n"
        + "Prior failure profile for this family and this agent:\n"
        + feedback_text
        + "\n"
    )
    messages[-1]["content"] += guidance
    return budget_messages(messages, max_input_tokens=3000)


def compact_retry_feedback(quality_feedback: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(quality_feedback, dict) or not quality_feedback:
        return {}
    return {
        "score": quality_feedback.get("individual_proposal_score"),
        "quality_level": quality_feedback.get("quality_level"),
        "main_issues": (quality_feedback.get("main_issues") or [])[:2],
        "future_instructions": (quality_feedback.get("future_instructions") or [])[:2],
    }


def build_individual_proposal_messages(
    agent: Dict[str, Any],
    memories: List[Dict[str, Any]],
    quality_feedback: Optional[Dict[str, Any]] = None,
    attempt_index: int = 1,
    backtracking_feedback: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    profile = agent.get("profile", {})
    constraints = agent.get("constraints", {})
    preference_weights = agent.get("preference_weights", {})
    memory_text = compact_memory_items(
        memories, max_items=MAX_MEMORY_ITEMS_FOR_PROMPT, max_chars=360
    )
    feedback_text = compact_feedback_for_llm(
        compact_json_for_prompt(compact_retry_feedback(quality_feedback), max_chars=420),
        max_chars=360,
    )
    messages = build_individual_proposal_messages_template(
        agent=agent,
        profile=profile,
        constraints=constraints,
        preference_weights=preference_weights,
        memory_text=memory_text,
        feedback_text=feedback_text,
        attempt_index=attempt_index,
    )
    if backtracking_feedback:
        messages = append_backtracking_feedback_to_proposal_messages(
            messages=messages, agent=agent, family_feedback=backtracking_feedback
        )
    return budget_messages(messages, max_input_tokens=3000 if backtracking_feedback else 2600)


def _words(text: Any) -> List[str]:
    return str(text).split()


def truncate_words(text: Any, max_words: int) -> str:
    words = _words(text)
    if len(words) <= max_words:
        return str(text) if text is not None else ""
    return " ".join(words[:max_words])


def limit_list(value: Any, max_items: int) -> List[Any]:
    if isinstance(value, list):
        return value[:max_items]
    if value is None:
        return []
    return [value][:max_items]


def short_reason_fields(value: Any, max_words: int = 10) -> Any:
    if isinstance(value, dict):
        output = {}
        for key, val in value.items():
            if key in {"reason", "rationale", "why", "note"} and isinstance(val, str):
                output[key] = truncate_words(val, max_words)
            elif isinstance(val, (dict, list)):
                output[key] = short_reason_fields(val, max_words=max_words)
            else:
                output[key] = val
        return output
    if isinstance(value, list):
        return [short_reason_fields(item, max_words=max_words) for item in value]
    return value


def make_priority_item(rank: int, need: Any, item_type: str, flexibility: str) -> Dict[str, Any]:
    return {
        "rank": int(rank),
        "need": truncate_words(need, 8),
        "type": safe_priority_type(item_type),
        "flexibility": safe_flexibility(flexibility),
    }


def safe_priority_type(value: Any) -> str:
    text = str(value or "preference").strip()
    allowed = {"must_include", "must_avoid", "preference", "comfort", "budget"}
    return text if text in allowed else "preference"


def safe_flexibility(value: Any) -> str:
    text = str(value or "negotiable").strip()
    allowed = {"non_negotiable", "split_possible", "negotiable"}
    return text if text in allowed else "negotiable"


def normalize_priority_item(item: Any, rank: int) -> Dict[str, Any]:
    if isinstance(item, dict):
        item_type = safe_priority_type(item.get("type") or "preference")
        default_flex = (
            "split_possible"
            if item_type == "must_include"
            else "non_negotiable"
            if item_type == "must_avoid"
            else "negotiable"
        )
        return make_priority_item(
            rank=rank,
            need=item.get("need")
            or item.get("name_or_type")
            or item.get("arrangement")
            or "preference",
            item_type=item_type,
            flexibility=item.get("flexibility") or default_flex,
        )
    return make_priority_item(rank, item, "preference", "negotiable")


def strip_priority_extra_fields(proposal: Dict[str, Any]) -> Dict[str, Any]:
    priorities = proposal.get("proposal_priority")
    if not isinstance(priorities, list):
        return proposal
    cleaned = []
    for index, item in enumerate(priorities[:5], start=1):
        cleaned.append(normalize_priority_item(item, index))
    proposal["proposal_priority"] = cleaned
    return proposal


RISKY_TERM_REPLACEMENTS = {
    "long walking": "short-walk limit",
    "high intensity": "tiring plan",
    "theme park marathon": "packed park day",
    "hiking": "steep outdoor route",
    "climbing": "many stairs",
    "高强度": "较累安排",
    "爬山": "陡坡路线",
    "长距离": "步行过多",
}

EXPENSIVE_TERM_REPLACEMENTS = {
    "luxury": "comfortable",
    "high-end": "comfortable",
    "expensive": "costly",
    "high price": "costly",
    "豪华": "舒适",
    "高端": "舒适",
    "昂贵": "费用较高",
}


def replace_terms_in_text(text: Any, replacements: Dict[str, str]) -> Any:
    if not isinstance(text, str):
        return text
    output = text
    for old, new in replacements.items():
        output = output.replace(old, new)
        output = output.replace(old.title(), new)
        output = output.replace(old.upper(), new)
    return output


def recursively_replace_terms(value: Any, replacements: Dict[str, str]) -> Any:
    if isinstance(value, str):
        return replace_terms_in_text(value, replacements)
    if isinstance(value, list):
        return [recursively_replace_terms(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: recursively_replace_terms(val, replacements) for key, val in value.items()}
    return value


def replace_terms_in_selected_fields(
    proposal: Dict[str, Any], fields: List[str], replacements: Dict[str, str]
) -> Dict[str, Any]:
    for field in fields:
        if field in proposal:
            proposal[field] = recursively_replace_terms(proposal.get(field), replacements)
    return proposal


def sanitize_role_sensitive_terms(
    proposal: Dict[str, Any], agent: Dict[str, Any]
) -> Dict[str, Any]:
    profile = agent.get("profile", {})
    physical = str(profile.get("physical_constraint", "")).lower()
    budget = str(profile.get("budget_sensitivity", "")).lower()
    role = str(profile.get("role", "")).lower()
    age_stage = str(profile.get("age_stage", "")).lower()
    role_age = role + " " + age_stage
    if physical == "low_intensity" or any(key in role_age for key in ["elder", "older", "child"]):
        # Only clean positive recommendation fields. Keep rejected_arrangements and objection_triggers unchanged
        # so must_avoid terms can still be matched by the evaluator.
        proposal = replace_terms_in_selected_fields(
            proposal,
            ["preferred_attractions", "daily_intensity_bottom_line"],
            RISKY_TERM_REPLACEMENTS,
        )
    if budget == "strict_budget":
        proposal = replace_terms_in_selected_fields(
            proposal,
            [
                "short_self_summary",
                "preferred_attractions",
                "dining_preferences",
                "hotel_preferences",
            ],
            EXPENSIVE_TERM_REPLACEMENTS,
        )
        hotel = proposal.get("hotel_preferences")
        if isinstance(hotel, dict):
            hotel["price_level"] = "medium"
            proposal["hotel_preferences"] = hotel
    if physical == "low_intensity" or any(key in role_age for key in ["elder", "older"]):
        proposal = replace_terms_in_selected_fields(
            proposal,
            ["short_self_summary", "preferred_attractions", "daily_intensity_bottom_line"],
            {
                "compact pace": "relaxed pace",
                "packed schedule": "relaxed schedule",
                "many activities": "few activities",
            },
        )
    proposal = strip_priority_extra_fields(proposal)
    return proposal


def normalize_short_proposal(proposal: Dict[str, Any], agent: Dict[str, Any]) -> Dict[str, Any]:
    profile = agent.get("profile", {})
    constraints = agent.get("constraints", {})
    role = profile.get("role") or "member"
    interests = profile.get("interest_preferences") or []
    must_include = constraints.get("must_include") or []
    must_avoid = constraints.get("must_avoid") or []
    max_walking = constraints.get("max_walking_distance_km")
    latest_return = constraints.get("latest_return_time")

    proposal["short_self_summary"] = truncate_words(
        proposal.get("short_self_summary")
        or f"{role} protects hard constraints and accepts split activities.",
        20,
    )

    priorities = (
        proposal.get("proposal_priority")
        if isinstance(proposal.get("proposal_priority"), list)
        else []
    )
    normalized_priorities = [
        normalize_priority_item(item, index) for index, item in enumerate(priorities[:5], start=1)
    ]

    existing_needs = " ".join(
        str(item.get("need", item)) for item in normalized_priorities if isinstance(item, dict)
    ).lower()
    rank = len(normalized_priorities) + 1
    for item in list(must_include)[:3]:
        if rank > 5:
            break
        if str(item).lower() not in existing_needs:
            normalized_priorities.append(
                make_priority_item(rank, item, "must_include", "split_possible")
            )
            rank += 1
    for item in list(must_avoid)[:2]:
        if rank > 5:
            break
        if str(item).lower() not in existing_needs:
            normalized_priorities.append(
                make_priority_item(rank, item, "must_avoid", "non_negotiable")
            )
            rank += 1
    for item in list(interests)[:3]:
        if rank > 5:
            break
        normalized_priorities.append(make_priority_item(rank, item, "preference", "negotiable"))
        rank += 1
    while len(normalized_priorities) < 3:
        rank = len(normalized_priorities) + 1
        normalized_priorities.append(
            make_priority_item(rank, "balanced family plan", "comfort", "negotiable")
        )
    proposal["proposal_priority"] = normalized_priorities[:5]

    proposal["preferred_attractions"] = limit_list(
        short_reason_fields(proposal.get("preferred_attractions", []), 6), 3
    )
    proposal["rejected_arrangements"] = limit_list(
        short_reason_fields(proposal.get("rejected_arrangements", []), 6), 2
    )
    proposal["negotiable_parts"] = limit_list(
        short_reason_fields(proposal.get("negotiable_parts", []), 6), 3
    )
    proposal["non_negotiable_parts"] = limit_list(
        short_reason_fields(proposal.get("non_negotiable_parts", []), 6), 3
    )
    proposal["initial_acceptance_criteria"] = limit_list(
        short_reason_fields(proposal.get("initial_acceptance_criteria", []), 6), 3
    )
    proposal["objection_triggers"] = limit_list(
        short_reason_fields(proposal.get("objection_triggers", []), 6), 3
    )

    if not proposal["preferred_attractions"] and must_include:
        proposal["preferred_attractions"] = [
            {"name_or_type": str(item), "reason": "Must include.", "priority": "high"}
            for item in list(must_include)[:3]
        ]
    if not proposal["rejected_arrangements"] and must_avoid:
        proposal["rejected_arrangements"] = [
            {"arrangement": str(item), "reason": "Must avoid.",} for item in list(must_avoid)[:2]
        ]
    if not proposal["non_negotiable_parts"]:
        base = list(must_avoid)[:2] + list(must_include)[:1]
        proposal["non_negotiable_parts"] = [str(item) for item in base[:3]] or [
            "respect hard constraints"
        ]
    if not proposal["negotiable_parts"]:
        proposal["negotiable_parts"] = ["exact timing", "specific restaurant", "optional activity"][
            :3
        ]
    if not proposal["initial_acceptance_criteria"]:
        proposal["initial_acceptance_criteria"] = [
            "hard constraints respected",
            "walking limit respected",
            "return time respected",
        ][:3]
    if not proposal["objection_triggers"]:
        triggers = [f"includes {x}" for x in list(must_avoid)[:2]]
        proposal["objection_triggers"] = triggers[:3] or ["violates hard constraints"]

    daily = proposal.get("daily_intensity_bottom_line")
    if not isinstance(daily, dict):
        daily = {}
    daily.setdefault("max_walking_distance_km_per_day", max_walking or 4)
    daily.setdefault("max_poi_count_per_day", 3)
    daily.setdefault("latest_return_time", latest_return or "20:00")
    daily.setdefault("preferred_pace", profile.get("travel_pace") or "moderate")
    daily.setdefault("required_rest", True)
    proposal["daily_intensity_bottom_line"] = daily

    dining = proposal.get("dining_preferences")
    if not isinstance(dining, dict):
        dining = {}
    dining = {
        "preferred_food_types": limit_list(
            dining.get("preferred_food_types") or profile.get("dietary_preference") or [], 3
        ),
        "must_avoid_food": limit_list(
            dining.get("must_avoid_food") or dining.get("avoid_food_types") or [], 2
        ),
        "meal_price_preference": dining.get("meal_price_preference")
        or dining.get("price_level")
        or "medium",
    }
    proposal["dining_preferences"] = short_reason_fields(dining, 6)

    hotel = proposal.get("hotel_preferences")
    if not isinstance(hotel, dict):
        hotel = {}
    hotel = {
        "preferred_area": truncate_words(hotel.get("preferred_area") or "near transit", 6),
        "price_level": hotel.get("price_level") or "medium",
        "comfort_requirement": truncate_words(
            hotel.get("comfort_requirement") or hotel.get("hotel_type") or "family-friendly", 6
        ),
        "accessibility_requirement": truncate_words(
            hotel.get("accessibility_requirement") or "low walking", 6
        ),
    }
    proposal["hotel_preferences"] = short_reason_fields(hotel, 6)

    transport = proposal.get("transportation_preferences")
    if not isinstance(transport, dict):
        transport = {}
    transport = {
        "preferred_mode": truncate_words(transport.get("preferred_mode") or "metro/taxi", 4),
        "route_strategy": truncate_words(transport.get("route_strategy") or "short transfers", 6),
    }
    proposal["transportation_preferences"] = transport
    proposal = sanitize_role_sensitive_terms(proposal, agent)
    allowed_keys = {
        "short_self_summary",
        "proposal_priority",
        "preferred_attractions",
        "rejected_arrangements",
        "daily_intensity_bottom_line",
        "dining_preferences",
        "hotel_preferences",
        "transportation_preferences",
        "negotiable_parts",
        "non_negotiable_parts",
        "initial_acceptance_criteria",
        "objection_triggers",
        "agent_id",
        "family_id",
        "member_id",
        "role",
    }
    proposal = {k: v for k, v in proposal.items() if k in allowed_keys}
    return proposal


def ensure_required_proposal_fields(
    proposal: Dict[str, Any], agent: Dict[str, Any]
) -> Dict[str, Any]:
    if not isinstance(proposal, dict):
        proposal = {}
    return normalize_short_proposal(proposal, agent)


def call_model_for_agent(
    agent: Dict[str, Any],
    memory_store: RoleMemoryStore,
    quality_feedback: Optional[Dict[str, Any]] = None,
    attempt_index: int = 1,
    backtracking_feedback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    agent_id = agent.get("agent_id")
    profile = agent.get("profile", {})
    memories = memory_store.load_memories(agent=agent, max_items=MAX_MEMORY_ITEMS_FOR_PROMPT)
    messages = build_individual_proposal_messages(
        agent=agent,
        memories=memories,
        quality_feedback=quality_feedback,
        attempt_index=attempt_index,
        backtracking_feedback=backtracking_feedback,
    )
    api_result = call_generation_json(
        messages=messages, model=MODEL_NAME, max_tokens=MAX_TOKENS_PROPOSAL, temperature=TEMPERATURE
    )
    proposal_data = api_result.get("data", {}) or {}
    usage = api_result.get("usage", {}) or {}
    if isinstance(proposal_data, dict) and isinstance(
        proposal_data.get("individual_proposal"), dict
    ):
        proposal = proposal_data.get("individual_proposal", {})
    elif isinstance(proposal_data, dict):
        proposal = proposal_data
    else:
        proposal = {}
    proposal.setdefault("agent_id", agent_id)
    proposal.setdefault("family_id", agent.get("family_id"))
    proposal.setdefault("member_id", agent.get("member_id"))
    proposal.setdefault("role", profile.get("role"))
    proposal = ensure_required_proposal_fields(proposal, agent)
    return {"proposal": proposal, "usage": usage}


def build_output_row(
    agent: Dict[str, Any],
    proposal: Dict[str, Any],
    usage: Dict[str, Any],
    evaluation: Dict[str, Any],
    attempts: int,
) -> Dict[str, Any]:
    profile = agent.get("profile", {})
    return {
        "agent_id": agent.get("agent_id"),
        "family_id": agent.get("family_id"),
        "member_id": agent.get("member_id"),
        "role": profile.get("role"),
        "age_stage": profile.get("age_stage"),
        "conflict_level": agent.get("conflict_level"),
        "family_structure": agent.get("family_structure"),
        "departure_city": agent.get("departure_city"),
        "destination_city": agent.get("destination_city"),
        "day": agent.get("day"),
        "query": agent.get("query"),
        "individual_proposal_json": json.dumps(proposal, ensure_ascii=False),
        "individual_proposal_score": evaluation.get("individual_proposal_score"),
        "individual_proposal_raw_score": evaluation.get("individual_proposal_raw_score"),
        "proposal_quality_level": evaluation.get("quality_level"),
        "proposal_pass": evaluation.get("proposal_pass"),
        # Per-component proposal-quality sub-scores (the QG-ablation discriminators). Formerly
        # persisted by the standalone Module 2.5 stage; that stage is gone, so embed them here
        # from the same evaluate_individual_proposal() call the quality gate already runs.
        "hard_constraint_awareness_score": evaluation.get("hard_constraint_awareness_score"),
        "preference_specificity_score": evaluation.get("preference_specificity_score"),
        "role_consistency_score": evaluation.get("role_consistency_score"),
        "negotiation_usefulness_score": evaluation.get("negotiation_usefulness_score"),
        "feasibility_score": evaluation.get("feasibility_score"),
        "conflict_risk_control_score": evaluation.get("conflict_risk_control_score"),
        "proposal_main_issues_json": json.dumps(
            evaluation.get("main_issues", []), ensure_ascii=False
        ),
        "proposal_future_instructions_json": json.dumps(
            evaluation.get("future_instructions", []), ensure_ascii=False
        ),
        "proposal_attempts": attempts,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "status": "success",
    }


def generate_proposal_for_agent(
    agent: Dict[str, Any],
    memory_store: RoleMemoryStore,
    enable_quality_gate: bool = ENABLE_PROPOSAL_QUALITY_GATE,
    enable_reward_memory: bool = True,
    proposal_pass_threshold: float = PROPOSAL_PASS_THRESHOLD,
    proposal_strong_threshold: float = PROPOSAL_STRONG_THRESHOLD,
    proposal_max_regen_attempts: int = PROPOSAL_MAX_REGEN_ATTEMPTS,
    backtracking_feedback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    agent_id = agent.get("agent_id")
    if not agent_id:
        raise ValueError("agent 缺少 agent_id")
    memory_store.initialize_profile_memory(agent)
    best = None
    best_eval = None
    best_usage = {}
    best_attempt = 0
    quality_feedback = None
    max_attempts = max(1, int(proposal_max_regen_attempts) + 1) if enable_quality_gate else 1
    for attempt in range(1, max_attempts + 1):
        generated = call_model_for_agent(
            agent=agent,
            memory_store=memory_store,
            quality_feedback=quality_feedback,
            attempt_index=attempt,
            backtracking_feedback=backtracking_feedback,
        )
        proposal = generated["proposal"]
        usage = generated["usage"]
        evaluation = evaluate_individual_proposal(agent, proposal)
        evaluation["proposal_pass"] = (
            evaluation["individual_proposal_score"] >= proposal_pass_threshold
        )
        evaluation["proposal_strong"] = (
            evaluation["individual_proposal_score"] >= proposal_strong_threshold
        )
        print(
            f"[Module 2] agent_id={agent_id}, attempt={attempt}, proposal_score={evaluation['individual_proposal_score']}, quality={evaluation['quality_level']}"
        )
        if (
            best_eval is None
            or evaluation["individual_proposal_score"] > best_eval["individual_proposal_score"]
        ):
            best = proposal
            best_eval = evaluation
            best_usage = usage
            best_attempt = attempt
        if (
            enable_quality_gate
            and evaluation["individual_proposal_score"] < proposal_pass_threshold
        ):
            if enable_reward_memory:
                memory_store.append_memory(
                    agent=agent,
                    memory=memory_payload_from_evaluation(
                        evaluation=evaluation,
                        memory_type="negative_proposal_feedback",
                        source="module2_quality_gate",
                    ),
                )
            quality_feedback = evaluation
            if attempt < max_attempts:
                if enable_reward_memory:
                    print(f"[Module 2] agent_id={agent_id} proposal 低分，写入 negative memory 后重新生成。")
                else:
                    print(f"[Module 2] agent_id={agent_id} proposal 低分，重新生成；奖惩记忆已关闭。")
                time.sleep(SLEEP_SECONDS)
                continue
        if (
            enable_quality_gate
            and enable_reward_memory
            and (evaluation["individual_proposal_score"] >= proposal_strong_threshold)
        ):
            memory_store.append_memory(
                agent=agent,
                memory=memory_payload_from_evaluation(
                    evaluation=evaluation,
                    memory_type="positive_proposal_feedback",
                    source="module2_quality_gate",
                ),
            )
        break
    memory_store.save_individual_proposal_memory(agent=agent, proposal=best or {}, usage=best_usage)
    return build_output_row(
        agent=agent,
        proposal=best or {},
        usage=best_usage,
        evaluation=best_eval or {},
        attempts=best_attempt,
    )


def build_error_result(agent: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    profile = agent.get("profile", {})
    return {
        "agent_id": agent.get("agent_id"),
        "family_id": agent.get("family_id"),
        "member_id": agent.get("member_id"),
        "role": profile.get("role"),
        "age_stage": profile.get("age_stage"),
        "conflict_level": agent.get("conflict_level"),
        "family_structure": agent.get("family_structure"),
        "departure_city": agent.get("departure_city"),
        "destination_city": agent.get("destination_city"),
        "day": agent.get("day"),
        "query": agent.get("query"),
        "individual_proposal_json": json.dumps({"error": str(error)}, ensure_ascii=False),
        "individual_proposal_score": 0,
        "proposal_quality_level": "failed",
        "proposal_pass": False,
        "proposal_main_issues_json": json.dumps([str(error)], ensure_ascii=False),
        "proposal_future_instructions_json": json.dumps([], ensure_ascii=False),
        "proposal_attempts": 0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "status": "failed",
    }


def save_proposal_results(
    results: List[Dict[str, Any]], output_csv_path: Path, output_json_path: Path
):
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(results)
    if "family_id" in result_df.columns and "member_id" in result_df.columns:
        result_df = result_df.sort_values(by=["family_id", "member_id"])
    result_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    save_json_file(result_df.to_dict(orient="records"), output_json_path)


def print_proposal_preview(result: Dict[str, Any]):
    print("=" * 100)
    print(f"agent_id: {result.get('agent_id')}")
    print(f"family_id: {result.get('family_id')}")
    print(f"member_id: {result.get('member_id')}")
    print(f"role: {result.get('role')}")
    print(f"status: {result.get('status')}")
    print(f"proposal_score: {result.get('individual_proposal_score')}")
    print(f"proposal_quality_level: {result.get('proposal_quality_level')}")
    print("individual_proposal_json:")
    try:
        proposal = json.loads(result.get("individual_proposal_json", "{}"))
        print(json.dumps(proposal, ensure_ascii=False, indent=2)[:1500])
    except Exception:
        print(result.get("individual_proposal_json"))
    print("=" * 100)


def build_family_agents_map(agents: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    family_agents_map = {}
    for agent in agents:
        family_id = int(agent["family_id"])
        family_agents_map.setdefault(family_id, []).append(agent)
    return family_agents_map


def select_agents_by_family(
    agents: List[Dict[str, Any]],
    test_first_n_families: Optional[int] = None,
    target_family_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    family_agents_map = build_family_agents_map(agents)
    all_family_ids = sorted(family_agents_map.keys())
    if target_family_ids is not None:
        target_set = {int(x) for x in target_family_ids}
        family_ids = [fid for fid in all_family_ids if int(fid) in target_set]
        print(f"[Module 2] Backtracking target mode: 只重新生成未达标家庭 family_id={family_ids}")
    else:
        family_ids = all_family_ids
        if test_first_n_families is not None:
            family_ids = family_ids[:test_first_n_families]
    selected_agents = []
    for family_id in family_ids:
        selected_agents.extend(family_agents_map[family_id])
    print(f"[Module 2] 实际运行家庭数量: {len(family_ids)}")
    print(f"[Module 2] 实际运行 agent 数量: {len(selected_agents)}")
    print(f"[Module 2] family_id 列表: {family_ids}")
    return selected_agents


def generate_individual_proposals(
    agents_json_path: Path = AGENTS_JSON,
    output_csv_path: Path = OUTPUT_CSV,
    output_json_path: Path = OUTPUT_JSON,
    memory_dir: Path = MEMORY_DIR,
    test_first_n: Optional[int] = None,
    test_first_n_families: Optional[int] = None,
    max_workers: int = 3,
    max_workers_agents: Optional[int] = None,
    max_workers_families: Optional[int] = None,
    enable_quality_gate: bool = ENABLE_PROPOSAL_QUALITY_GATE,
    enable_reward_memory: bool = True,
    proposal_pass_threshold: float = PROPOSAL_PASS_THRESHOLD,
    proposal_strong_threshold: float = PROPOSAL_STRONG_THRESHOLD,
    proposal_max_regen_attempts: int = PROPOSAL_MAX_REGEN_ATTEMPTS,
    target_family_ids: Optional[List[int]] = None,
    backtracking_feedback_json_path: Path = BACKTRACKING_FEEDBACK_JSON,
) -> pd.DataFrame:
    if not agents_json_path.exists():
        raise FileNotFoundError(f"找不到 persona_agents.json，请先运行模块 1: {agents_json_path}")
    agents = load_json_file(agents_json_path)
    if not isinstance(agents, list):
        raise ValueError("persona_agents.json 格式错误，应该是 list。")
    if test_first_n_families is None:
        test_first_n_families = test_first_n
    if max_workers_agents is not None:
        max_workers = max_workers_agents
    elif max_workers_families is not None:
        max_workers = max_workers_families
    agents = select_agents_by_family(
        agents=agents,
        test_first_n_families=test_first_n_families,
        target_family_ids=target_family_ids,
    )
    print("[Module 2] Individual Proposal Generation + Quality Gate 开始")
    print(f"[Module 2] agent 数量: {len(agents)}")
    print(f"[Module 2] max_workers: {max_workers}")
    print(f"[Module 2] role 共享记忆库: {memory_dir}")
    print(f"[Module 2] enable_quality_gate: {enable_quality_gate}")
    print(f"[Module 2] proposal_pass_threshold: {proposal_pass_threshold}")
    print(f"[Module 2] proposal_max_regen_attempts: {proposal_max_regen_attempts}")
    memory_store = RoleMemoryStore(memory_dir)
    backtracking_feedback_lookup = load_backtracking_feedback_lookup(
        backtracking_feedback_json_path
    )
    if target_family_ids is not None and backtracking_feedback_lookup:
        print(f"[Module 2] 已加载 backtracking feedback 家庭数量: {len(backtracking_feedback_lookup)}")
    results = []

    def worker(agent: Dict[str, Any]) -> Dict[str, Any]:
        agent_id = agent.get("agent_id")
        try:
            family_feedback = backtracking_feedback_lookup.get(
                safe_int(agent.get("family_id"), 0), {}
            )
            return generate_proposal_for_agent(
                agent=agent,
                memory_store=memory_store,
                enable_quality_gate=enable_quality_gate,
                enable_reward_memory=enable_reward_memory,
                proposal_pass_threshold=proposal_pass_threshold,
                proposal_strong_threshold=proposal_strong_threshold,
                proposal_max_regen_attempts=proposal_max_regen_attempts,
                backtracking_feedback=family_feedback,
            )
        except Exception as e:
            print(f"[Module 2] agent_id={agent_id} 生成失败: {e}")
            return build_error_result(agent, e)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_agent = {executor.submit(worker, agent): agent for agent in agents}
        for index, future in enumerate(as_completed(future_to_agent), start=1):
            agent = future_to_agent[future]
            agent_id = agent.get("agent_id")
            print(f"[Module 2] 完成第 {index}/{len(agents)} 个 agent: {agent_id}")
            result = future.result()
            results.append(result)
            save_proposal_results(results, output_csv_path, output_json_path)
            if index <= 2:
                print_proposal_preview(result)
            time.sleep(SLEEP_SECONDS)
    save_proposal_results(results, output_csv_path, output_json_path)
    final_df = pd.DataFrame(results)
    if "family_id" in final_df.columns and "member_id" in final_df.columns:
        final_df = final_df.sort_values(by=["family_id", "member_id"])
    success_count = sum((1 for item in results if item.get("status") == "success"))
    failed_count = sum((1 for item in results if item.get("status") == "failed"))
    low_count = sum(
        (
            1
            for item in results
            if item.get("status") == "success"
            and float(item.get("individual_proposal_score") or 0) < proposal_pass_threshold
        )
    )
    print("[Module 2] Individual Proposal Generation + Quality Gate 完成")
    print(f"[Module 2] CSV 输出文件: {output_csv_path}")
    print(f"[Module 2] JSON 输出文件: {output_json_path}")
    print(f"[Module 2] 成功 agent 数量: {success_count}")
    print(f"[Module 2] 失败 agent 数量: {failed_count}")
    print(f"[Module 2] 最终仍低于阈值的 proposal 数量: {low_count}")
    return final_df


run_individual_proposal_generation = generate_individual_proposals
