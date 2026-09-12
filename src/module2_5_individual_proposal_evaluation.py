"""Proposal-quality scoring library (formerly the standalone Module 2.5 stage).

The batch evaluation stage and its CSV/JSON outputs were removed; this module now only
provides the reusable per-proposal scorer `evaluate_individual_proposal` (plus thresholds and
`memory_payload_from_evaluation`) that the Module 2 quality gate and Module 7 repair import."""
import json
from typing import Any, Dict, List, Tuple

from src.common import (
    PROPOSAL_QUALITY_FORMULA,
    PROPOSAL_QUALITY_WEIGHTS,
    clamp_score,
    normalize_list,
    normalize_role_key,
    parse_json_text,
    proposal_quality_theory,
    safe_float,
    safe_int,
    safe_text,
    time_to_minutes,
)

PROPOSAL_PASS_THRESHOLD = 80.0
PROPOSAL_STRONG_THRESHOLD = 90.0
CRITICAL_FIELD_CAPS = {
    "preferred_attractions": 72.0,
    "rejected_arrangements": 72.0,
    "daily_intensity_bottom_line": 72.0,
    "negotiable_parts": 78.0,
    "non_negotiable_parts": 76.0,
    "initial_acceptance_criteria": 76.0,
    "objection_triggers": 76.0,
}


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return normalize_list(value)
    if isinstance(value, tuple):
        return normalize_list(list(value))
    text = safe_text(value)
    if not text:
        return []
    parsed = parse_json_text(text, default=None)
    if isinstance(parsed, list):
        return normalize_list(parsed)
    return normalize_list([text])


def _flatten_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False).lower()
    except Exception:
        return safe_text(value).lower()


def _field_text(proposal: Dict[str, Any], fields: List[str]) -> str:
    parts = []
    for field in fields:
        parts.append(_flatten_text(proposal.get(field, "")))
    return " ".join(parts).lower()


def _list_len(value: Any) -> int:
    if isinstance(value, list):
        count = 0
        for item in value:
            if isinstance(item, dict):
                if _dict_nonempty_count(item) > 0:
                    count += 1
            elif safe_text(item):
                count += 1
        return count
    if isinstance(value, dict):
        return _dict_nonempty_count(value)
    return 1 if safe_text(value) else 0


def _dict_nonempty_count(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    count = 0
    for _, v in value.items():
        if isinstance(v, list) and _list_len(v) > 0:
            count += 1
        elif isinstance(v, dict) and _dict_nonempty_count(v) > 0:
            count += 1
        elif safe_text(v):
            count += 1
    return count


def _item_detail_score(item: Any) -> float:
    text = _flatten_text(item)
    if not text:
        return 0.0
    score = 35.0
    if len(text) >= 20:
        score += 20
    if any((k in text for k in ["reason", "because", "why", "原因", "因为"])):
        score += 15
    if any((k in text for k in ["priority", "high", "medium", "low", "优先级", "重要"])):
        score += 15
    if any(
        (
            k in text
            for k in [
                "must",
                "avoid",
                "limit",
                "latest",
                "walking",
                "budget",
                "不可",
                "必须",
                "避免",
                "限制",
            ]
        )
    ):
        score += 15
    return clamp_score(score)


def _list_detail_average(value: Any) -> float:
    if isinstance(value, list):
        items = [item for item in value if safe_text(item) or isinstance(item, dict)]
    elif isinstance(value, dict):
        items = list(value.values())
    else:
        items = [value] if safe_text(value) else []
    if not items:
        return 0.0
    return sum((_item_detail_score(item) for item in items)) / len(items)


def _field_missing(proposal: Dict[str, Any], field: str) -> bool:
    return _list_len(proposal.get(field, [])) == 0


def score_hard_constraint_awareness(
    agent: Dict[str, Any], proposal: Dict[str, Any]
) -> Tuple[float, List[str], List[str]]:
    constraints = agent.get("constraints", {})
    must_include = _as_list(constraints.get("must_include", []))
    must_avoid = _as_list(constraints.get("must_avoid", []))
    persona_max_walking = safe_float(constraints.get("max_walking_distance_km"), None)
    persona_latest_return = safe_text(constraints.get("latest_return_time", ""))
    all_text = _flatten_text(proposal)
    avoid_text = _field_text(
        proposal, ["rejected_arrangements", "non_negotiable_parts", "objection_triggers"]
    )
    include_text = _field_text(
        proposal,
        [
            "preferred_attractions",
            "non_negotiable_parts",
            "proposal_priority",
            "initial_acceptance_criteria",
        ],
    )
    score = 100.0
    issues, instructions = ([], [])
    for item in must_include:
        item_text = safe_text(item).lower()
        if not item_text:
            continue
        if item_text not in include_text and item_text not in all_text:
            score -= 15
            issues.append(f"must_include 未在个人方案中明确体现: {item}")
            instructions.append(
                f"下次必须把 must_include '{item}' 写入 preferred_attractions 或 non_negotiable_parts，并说明优先级。"
            )
    for item in must_avoid:
        item_text = safe_text(item).lower()
        if not item_text:
            continue
        if item_text not in avoid_text:
            score -= 18
            issues.append(f"must_avoid 没有进入 rejected_arrangements / objection_triggers: {item}")
            instructions.append(
                f"下次必须把 must_avoid '{item}' 写入 rejected_arrangements 或 objection_triggers，并说明是否绝对避免。"
            )
    daily = proposal.get("daily_intensity_bottom_line", {})
    if not isinstance(daily, dict) or _dict_nonempty_count(daily) < 3:
        score -= 15
        issues.append("daily_intensity_bottom_line 不完整，缺少步行/返回时间/POI 数量/休息安排。")
        instructions.append(
            "下次必须完整写出 max_walking_distance、max_poi_count、latest_return_time、pace 和 rest。"
        )
    else:
        if persona_max_walking is not None:
            proposal_walking = safe_float(daily.get("max_walking_distance_km_per_day"), None)
            if proposal_walking is None or proposal_walking > persona_max_walking + 0.01:
                score -= 16
                issues.append("个人方案没有清楚表达或遵守 max_walking_distance_km。")
                instructions.append("下次必须写清楚每日最大步行距离，并且不超过 persona 限制。")
        latest_minutes = time_to_minutes(persona_latest_return)
        if latest_minutes is not None:
            proposal_return = safe_text(daily.get("latest_return_time", ""))
            proposal_minutes = time_to_minutes(proposal_return)
            if proposal_minutes is None or proposal_minutes > latest_minutes:
                score -= 16
                issues.append("个人方案没有清楚表达或遵守 latest_return_time。")
                instructions.append("下次必须写清楚最晚返回时间，并且不能晚于 persona 限制。")
    preferred_detail = _list_detail_average(proposal.get("preferred_attractions", []))
    rejected_detail = _list_detail_average(proposal.get("rejected_arrangements", []))
    if preferred_detail < 55:
        score -= 10
        issues.append("preferred_attractions 过于笼统，缺少具体类型、理由或优先级。")
    if rejected_detail < 55:
        score -= 10
        issues.append("rejected_arrangements 过于笼统，缺少拒绝原因或触发条件。")
    return (clamp_score(score), issues, instructions)


def score_preference_specificity(
    agent: Dict[str, Any], proposal: Dict[str, Any]
) -> Tuple[float, List[str], List[str]]:
    issues, instructions = ([], [])
    score_parts = []
    fields = [
        (
            "preferred_attractions",
            0.25,
            "preferred_attractions 太空或不具体",
            "下次给出具体 POI 类型、活动类型、理由和优先级。",
        ),
        ("proposal_priority", 0.18, "proposal_priority 太空，没有需求排序", "下次按重要程度列出 3 条以上优先级。"),
        (
            "initial_acceptance_criteria",
            0.17,
            "initial_acceptance_criteria 太空",
            "下次明确什么条件下会接受群体方案。",
        ),
        ("objection_triggers", 0.17, "objection_triggers 太空", "下次明确什么安排会导致反对。"),
    ]
    for field, weight, issue, instruction in fields:
        detail = _list_detail_average(proposal.get(field, []))
        score_parts.append(detail * weight)
        if detail < 55:
            issues.append(issue)
            instructions.append(instruction)
    dining = proposal.get("dining_preferences", {})
    hotel = proposal.get("hotel_preferences", {})
    dining_score = (
        100.0
        if _dict_nonempty_count(dining) >= 3
        else 65.0
        if _dict_nonempty_count(dining) >= 2
        else 30.0
    )
    hotel_score = (
        100.0
        if _dict_nonempty_count(hotel) >= 3
        else 65.0
        if _dict_nonempty_count(hotel) >= 2
        else 30.0
    )
    score_parts.append(dining_score * 0.12)
    score_parts.append(hotel_score * 0.11)
    if dining_score < 70:
        issues.append("dining_preferences 不够具体。")
        instructions.append("下次写清楚食物类型、价格偏好、特殊饮食要求。")
    if hotel_score < 70:
        issues.append("hotel_preferences 不够具体。")
        instructions.append("下次写清楚酒店区域、价格等级、舒适度或无障碍需求。")
    summary = safe_text(proposal.get("short_self_summary", ""))
    if len(summary) < 15:
        issues.append("short_self_summary 太短，无法帮助 coordinator 理解该成员。")
        instructions.append("下次用一句话清楚概括该成员旅行态度、底线和可妥协方向。")
        score_parts.append(45.0 * 0.0)
    return (clamp_score(sum(score_parts)), issues, instructions)


def score_role_consistency(
    agent: Dict[str, Any], proposal: Dict[str, Any]
) -> Tuple[float, List[str], List[str]]:
    profile = agent.get("profile", {})
    role_key = normalize_role_key(profile.get("role", ""), profile.get("age_stage", ""))
    physical = safe_text(profile.get("physical_constraint", "")).lower()
    budget = safe_text(profile.get("budget_sensitivity", "")).lower()
    text = _flatten_text(proposal)
    positive_activity_text = _field_text(
        proposal, ["preferred_attractions", "daily_intensity_bottom_line"]
    )
    positive_budget_text = _field_text(
        proposal,
        ["short_self_summary", "preferred_attractions", "dining_preferences", "hotel_preferences"],
    )
    score = 100.0
    issues, instructions = ([], [])
    role_keywords = {
        "teenager": ["shopping", "photo", "instagram", "social", "entertainment", "自由", "拍照", "购物"],
        "child": ["child", "kid", "family", "safe", "interactive", "亲子", "儿童", "安全"],
        "older": [
            "rest",
            "comfort",
            "accessible",
            "low",
            "relaxed",
            "early",
            "休息",
            "舒适",
            "低强度",
            "早返回",
        ],
    }
    if role_key in role_keywords and (not any((k in text for k in role_keywords[role_key]))):
        score -= 22
        issues.append(f"{role_key} 方案没有体现该角色的典型需求。")
        instructions.append(f"{role_key} 下次需要明确符合角色的活动、节奏和底线。")
    if physical == "low_intensity":
        risky = ["hiking", "climbing", "theme park marathon", "long walking", "高强度", "爬山", "长距离"]
        if any((k in positive_activity_text for k in risky)):
            score -= 30
            issues.append("low_intensity 成员方案中出现高强度活动倾向。")
            instructions.append("low_intensity 成员下次必须避免高强度活动和长距离步行。")
    if budget == "strict_budget" and any(
        (
            k in positive_budget_text
            for k in ["luxury", "high-end", "expensive", "high price", "豪华", "高端", "昂贵"]
        )
    ):
        score -= 25
        issues.append("strict_budget 成员方案中出现高价倾向。")
        instructions.append("strict_budget 成员下次必须强调 low/medium 价格等级和预算控制。")
    return (clamp_score(score), issues, instructions)


def score_negotiation_usefulness(
    agent: Dict[str, Any], proposal: Dict[str, Any]
) -> Tuple[float, List[str], List[str]]:
    fields = [
        ("negotiable_parts", "negotiable_parts 为空，协商时不知道可以让步什么。", "下次必须列出可妥协项，并说明让步范围。"),
        ("non_negotiable_parts", "non_negotiable_parts 为空，协商时不知道底线。", "下次必须列出不可妥协项。"),
        (
            "initial_acceptance_criteria",
            "initial_acceptance_criteria 为空，协商时不知道接受条件。",
            "下次必须列出接受最终方案的条件。",
        ),
        ("objection_triggers", "objection_triggers 为空，协商时不知道反对触发点。", "下次必须列出会反对的安排。"),
    ]
    scores, issues, instructions = ([], [], [])
    for field, issue, instruction in fields:
        detail = _list_detail_average(proposal.get(field, []))
        scores.append(detail)
        if detail < 55:
            issues.append(issue)
            instructions.append(instruction)
    return (clamp_score(sum(scores) / len(scores)), issues, instructions)


def score_feasibility(
    agent: Dict[str, Any], proposal: Dict[str, Any]
) -> Tuple[float, List[str], List[str]]:
    daily = proposal.get("daily_intensity_bottom_line", {})
    score = 100.0
    issues, instructions = ([], [])
    if isinstance(daily, dict):
        poi_count = safe_float(daily.get("max_poi_count_per_day"), None)
        walking = safe_float(daily.get("max_walking_distance_km_per_day"), None)
        if poi_count is None:
            score -= 15
            issues.append("个人方案没有写每日 POI 数量上限。")
        elif poi_count > 5:
            score -= 30
            issues.append("个人方案每天 POI 数量上限过高，不够现实。")
            instructions.append("下次每日核心 POI 控制在 2-4 个更现实。")
        if walking is None:
            score -= 15
            issues.append("个人方案没有写每日步行距离上限。")
        elif walking > 8:
            score -= 30
            issues.append("个人方案每日步行距离过高，不够现实。")
            instructions.append("下次降低每日步行距离，必要时加入交通和休息。")
    else:
        score -= 35
        issues.append("daily_intensity_bottom_line 缺失，无法判断方案可行性。")
    dining = proposal.get("dining_preferences", {})
    hotel = proposal.get("hotel_preferences", {})
    price_text = (_flatten_text(dining) + " " + _flatten_text(hotel)).lower()
    budget = safe_text(agent.get("profile", {}).get("budget_sensitivity", "")).lower()
    if budget == "strict_budget" and any(
        (
            k in price_text
            for k in ["luxury", "high-end", "expensive", "high price", "豪华", "高端", "昂贵"]
        )
    ):
        score -= 30
        issues.append("strict_budget 成员的餐饮或酒店价格偏好不现实。")
        instructions.append("strict_budget 成员下次应选择 low 或 medium 价格等级。")
    return (clamp_score(score), issues, instructions)


def score_conflict_risk_control(
    agent: Dict[str, Any], proposal: Dict[str, Any]
) -> Tuple[float, List[str], List[str]]:
    score = 100.0
    issues, instructions = ([], [])
    if _list_len(proposal.get("negotiable_parts", [])) == 0:
        score -= 30
        issues.append("没有 negotiable_parts，容易增加协商冲突。")
        instructions.append("下次必须写出哪些偏好可以作为 optional 或 split activity。")
    if _list_len(proposal.get("objection_triggers", [])) == 0:
        score -= 30
        issues.append("没有 objection_triggers，后续系统难以预防拒绝。")
        instructions.append("下次必须写清楚哪些安排会导致反对。")
    text = _flatten_text(proposal)
    if any((k in text for k in ["must all", "everyone must", "全家必须", "所有人必须"])):
        score -= 25
        issues.append("方案里有强制全家接受的表达，容易导致群体冲突。")
        instructions.append("下次将个人强偏好设计为 optional/split activity，而不是强制全家参加。")
    strong_need_text = _field_text(proposal, ["preferred_attractions", "non_negotiable_parts"])
    flexible_text = _field_text(
        proposal, ["negotiable_parts", "proposal_priority", "initial_acceptance_criteria"]
    )
    potential_conflict_keywords = [
        "museums",
        "theme parks",
        "shopping",
        "crowded",
        "long queues",
        "night",
        "high intensity",
    ]
    has_risky_need = any((k in strong_need_text for k in potential_conflict_keywords))
    has_mitigation = any(
        (
            k in flexible_text
            for k in ["optional", "split", "alternative", "can skip", "可选", "分流", "替代", "可以不参加"]
        )
    )
    if has_risky_need and (not has_mitigation):
        score -= 20
        issues.append("方案包含潜在高冲突需求，但没有说明 optional/split/alternative 处理方式。")
        instructions.append("下次对可能冲突的活动写明是否可分流或可替代。")
    return (clamp_score(score), issues, instructions)


def has_severe_role_positive_conflict(agent: Dict[str, Any], proposal: Dict[str, Any]) -> bool:
    profile = agent.get("profile", {})
    role_key = normalize_role_key(profile.get("role", ""), profile.get("age_stage", ""))
    physical = safe_text(profile.get("physical_constraint", "")).lower()
    budget = safe_text(profile.get("budget_sensitivity", "")).lower()
    positive_activity_text = _field_text(
        proposal, ["preferred_attractions", "daily_intensity_bottom_line"]
    )
    positive_budget_text = _field_text(proposal, ["hotel_preferences", "dining_preferences"])
    severe_activity_terms = [
        "hiking",
        "climbing",
        "theme park marathon",
        "long walking",
        "high intensity",
        "高强度",
        "爬山",
        "长距离",
    ]
    sensitive_role = physical == "low_intensity" or role_key in ["older", "elder", "child"]
    if sensitive_role and any((k in positive_activity_text for k in severe_activity_terms)):
        return True
    severe_price_terms = ["luxury", "high-end", "expensive", "high price", "豪华", "高端", "昂贵"]
    if budget == "strict_budget" and any((k in positive_budget_text for k in severe_price_terms)):
        return True
    return False


def _score_caps(agent: Dict[str, Any], proposal: Dict[str, Any]) -> Tuple[float, List[str]]:
    caps = []
    reasons = []
    for field, cap in CRITICAL_FIELD_CAPS.items():
        if _field_missing(proposal, field):
            caps.append(cap)
            reasons.append(f"缺少关键字段 {field}，最高分限制为 {cap}")
    if _list_detail_average(proposal.get("preferred_attractions", [])) < 45:
        caps.append(72.0)
        reasons.append("preferred_attractions 过于笼统，最高分限制为 72")
    if _list_detail_average(proposal.get("objection_triggers", [])) < 45:
        caps.append(76.0)
        reasons.append("objection_triggers 不够具体，最高分限制为 76")
    role_score, _, _ = score_role_consistency(agent, proposal)
    if role_score < 60 and has_severe_role_positive_conflict(agent, proposal):
        caps.append(74.0)
        reasons.append("正向推荐中存在严重角色冲突，最高分限制为 74")
    return (min(caps) if caps else 100.0, reasons)


def quality_level(score: float) -> str:
    if score >= PROPOSAL_STRONG_THRESHOLD:
        return "strong"
    if score >= PROPOSAL_PASS_THRESHOLD:
        return "acceptable"
    if score >= 60:
        return "weak"
    return "bad"


def evaluate_individual_proposal(agent: Dict[str, Any], proposal: Dict[str, Any]) -> Dict[str, Any]:
    scorers = [
        (
            "hard_constraint_awareness_score",
            PROPOSAL_QUALITY_WEIGHTS["hard_constraint_awareness_score"],
            score_hard_constraint_awareness,
        ),
        (
            "preference_specificity_score",
            PROPOSAL_QUALITY_WEIGHTS["preference_specificity_score"],
            score_preference_specificity,
        ),
        (
            "role_consistency_score",
            PROPOSAL_QUALITY_WEIGHTS["role_consistency_score"],
            score_role_consistency,
        ),
        (
            "negotiation_usefulness_score",
            PROPOSAL_QUALITY_WEIGHTS["negotiation_usefulness_score"],
            score_negotiation_usefulness,
        ),
        ("feasibility_score", PROPOSAL_QUALITY_WEIGHTS["feasibility_score"], score_feasibility),
        (
            "conflict_risk_control_score",
            PROPOSAL_QUALITY_WEIGHTS["conflict_risk_control_score"],
            score_conflict_risk_control,
        ),
    ]
    total = 0.0
    output: Dict[str, Any] = {}
    component_scores: Dict[str, float] = {}
    component_weights: Dict[str, float] = {}
    issues: List[str] = []
    instructions: List[str] = []
    for name, weight, scorer in scorers:
        score, sub_issues, sub_instructions = scorer(agent, proposal)
        output[name] = round(score, 4)
        component_scores[name] = round(score, 4)
        component_weights[name] = weight
        total += score * weight
        issues.extend(sub_issues)
        instructions.extend(sub_instructions)
    raw_score = clamp_score(total)
    cap_score, cap_reasons = _score_caps(agent, proposal)
    final_score = round(clamp_score(min(raw_score, cap_score)), 4)
    issues.extend(cap_reasons)
    if cap_reasons:
        instructions.append("下次先补齐关键字段，再增加细节；不要只写笼统偏好。")
    issues = list(dict.fromkeys([safe_text(x) for x in issues if safe_text(x)]))[:10]
    instructions = list(dict.fromkeys([safe_text(x) for x in instructions if safe_text(x)]))[:10]
    critical_missing_fields = [
        field for field in CRITICAL_FIELD_CAPS if _field_missing(proposal, field)
    ]
    output.update(
        {
            "agent_id": agent.get("agent_id"),
            "family_id": safe_int(agent.get("family_id"), 0),
            "member_id": safe_int(agent.get("member_id"), 0),
            "role": agent.get("profile", {}).get("role", ""),
            "age_stage": agent.get("profile", {}).get("age_stage", ""),
            "individual_proposal_raw_score": round(raw_score, 4),
            "score_cap": round(cap_score, 4),
            "score_cap_reasons": cap_reasons,
            "critical_missing_fields": critical_missing_fields,
            "low_quality_signal_count": len(issues),
            "individual_proposal_score": final_score,
            "quality_level": quality_level(final_score),
            "proposal_pass": final_score >= PROPOSAL_PASS_THRESHOLD,
            "proposal_strong": final_score >= PROPOSAL_STRONG_THRESHOLD,
            "proposal_quality_formula": PROPOSAL_QUALITY_FORMULA,
            "proposal_quality_theory_json": json.dumps(
                proposal_quality_theory(component_scores), ensure_ascii=False
            ),
            "main_issues": issues,
            "future_instructions": instructions,
        }
    )
    return output


def memory_payload_from_evaluation(
    evaluation: Dict[str, Any], memory_type: str, source: str
) -> Dict[str, Any]:
    if memory_type == "positive_proposal_feedback":
        return {
            "memory_type": memory_type,
            "source": source,
            "proposal_score": evaluation.get("individual_proposal_score"),
            "quality_level": evaluation.get("quality_level"),
            "why_good": [
                "This proposal clearly expressed hard constraints, negotiation boundaries, acceptance criteria, and objection triggers.",
                "It is suitable as a reusable positive example for future agents with the same role.",
            ],
            "future_instruction": evaluation.get("future_instructions", [])[:3]
            or [
                "Continue producing concrete preferences, clear non-negotiable parts, explicit acceptance criteria, and conflict-aware alternatives."
            ],
        }
    return {
        "memory_type": memory_type,
        "source": source,
        "proposal_score": evaluation.get("individual_proposal_score"),
        "quality_level": evaluation.get("quality_level"),
        "why_bad": evaluation.get("main_issues", [])[:5],
        "future_instruction": evaluation.get("future_instructions", [])[:5],
        "critical_missing_fields": evaluation.get("critical_missing_fields", []),
    }
