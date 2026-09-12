"""Shared utilities and project paths: safe parsers, JSON I/O, role/room/walking estimation, and the role-memory store."""
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Re-export common project paths so older module imports stay simple.
from src.config import BACKTRACKING_FEEDBACK_JSON, DATA_DIR, MEMORY_DIR, RESULTS_DIR


def now_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Hotel room allocation by family composition (not just headcount):
#   父母 + 孩子  → 同住一间（按容量上限拆分）
#   青年/青少年  → 各自单独一间（隐私需求）
#   爷爷奶奶     → 单独一间（夫妻同住，>2 人再加）
_ELDER_ROLE_KEYS = (
    "elder",
    "elderly",
    "grandfather",
    "grandmother",
    "grandparent",
    "grandpa",
    "grandma",
    "grandparents",
    "爷爷",
    "奶奶",
    "外公",
    "外婆",
    "祖",
)
_YOUTH_ROLE_KEYS = (
    "teenager",
    "teen",
    "youth",
    "young_adult",
    "young-adult",
    "adolescent",
    "youngster",
    "青年",
    "青少年",
    "少年",
)


def _agent_role(agent: Dict[str, Any]) -> str:
    if not isinstance(agent, dict):
        return ""
    role = agent.get("role")
    if not role and isinstance(agent.get("profile"), dict):
        role = agent["profile"].get("role")
    return str(role or "")


def classify_member_role(role: str) -> str:
    """Map a member role to a rooming group: 'elder' | 'youth' | 'parent_child'."""
    r = str(role or "").lower()
    if any(k in r for k in _ELDER_ROLE_KEYS):
        return "elder"
    if any(k in r for k in _YOUTH_ROLE_KEYS):
        return "youth"
    return "parent_child"


def estimate_room_count(family_agents: List[Dict[str, Any]], people_per_room: int = 3) -> int:
    """Rooms needed for a family, by composition.

    父母+孩子 share rooms (capacity ``people_per_room``); each 青年/teenager takes a
    separate room; 爷爷奶奶 (grandparents) share one room (a couple), adding more
    only beyond two.  Always at least one room.
    """
    import math

    parent_child = youth = elder = 0
    for agent in family_agents or []:
        cls = classify_member_role(_agent_role(agent))
        if cls == "elder":
            elder += 1
        elif cls == "youth":
            youth += 1
        else:
            parent_child += 1
    cap = max(1, people_per_room)
    # 爷爷奶奶成对同住一间；落单的一位老人可与一位成年人同住一间，因此并入父母/成人房间池。
    elder_couple_rooms = elder // 2
    lone_elder = elder % 2
    pool = parent_child + lone_elder
    rooms = 0
    if pool > 0:
        rooms += math.ceil(pool / cap)  # 父母+孩子（+落单老人）同间
    rooms += youth  # 青年 各自单独一间
    rooms += elder_couple_rooms  # 爷爷奶奶成对一间
    return max(rooms, 1)


# Per-day walking estimation.  A fixed n_pois * 1.5 makes every day identical
# (e.g. 3.0 km when each day has 2 POIs).  This varies the estimate by the kind of
# POIs: parks/streets/scenic areas walk a lot; museums/indoor venues walk little.
# Per-POI in-venue walking by type, GRADUATED into four tiers so different POIs and
# days are more clearly differentiated (instead of most POIs collapsing into one
# value).  NB: avoid bare single-char geographic chars like 江/山/湖 — they match
# province and city names (e.g. 黑龙江省博物馆) and would mislabel indoor venues.
# Tier 1 — large nature/scenic areas (walk the most, but tuned DOWN from 2.4).
_WALK_NATURE_KEYWORDS = [
    "风景区",
    "景区",
    "古镇",
    "老街",
    "森林公园",
    "国家公园",
    "湿地",
    "登山",
    "动物园",
    "植物园",
    "海滩",
    "沙滩",
    "湖区",
    "步道",
    "外滩",
    "江畔",
    "峡谷",
    "瀑布",
    "溶洞",
    "陵园",
    "园林",
    "bund",
]
# Tier 2 — streets / parks / squares (moderate strolling).
_WALK_STREET_KEYWORDS = [
    "公园",
    "步行街",
    "商业街",
    "大街",
    "广场",
    "花园",
    "美食街",
    "夜市",
    "park",
    "street",
    "square",
    "avenue",
]
# Tier 4 — indoor venues (walk the least).
_LOW_WALK_KEYWORDS = [
    "博物馆",
    "美术馆",
    "纪念馆",
    "纪念堂",
    "展览",
    "教堂",
    "寺",
    "庙",
    "塔",
    "中心",
    "水族",
    "极地",
    "海洋馆",
    "剧院",
    "剧场",
    "商场",
    "购物",
    "mall",
    "museum",
    "church",
    "temple",
    "tower",
    "aquarium",
    "center",
]


def _poi_names_from_day(day_item: Dict[str, Any], keys=("morning", "afternoon")) -> List[str]:
    names: List[str] = []
    for key in keys:
        v = day_item.get(key, []) if isinstance(day_item, dict) else []
        if isinstance(v, list):
            for x in v:
                if isinstance(x, str) and x.strip():
                    names.append(x)
                elif isinstance(x, dict):
                    if x.get("is_transit") or x.get("poi_type") in ("arrival", "departure"):
                        continue  # arrival/departure are not sightseeing walking
                    nm = (
                        x.get("poi_name")
                        or x.get("name")
                        or x.get("poi")
                        or x.get("activity")
                        or ""
                    )
                    if str(nm).strip():
                        names.append(str(nm))
        elif isinstance(v, str) and v.strip():
            names.append(v)
    return names


def _poi_in_venue_walk_km(name: str, default_per_poi: float = 1.0) -> float:
    """Estimated walking WITHIN a single POI by type, graduated for discrimination:
    large nature/scenic 1.8, street/park/square 1.3, indoor venue 0.7, else default 1.0.
    Nature is checked before street so 森林公园/动物园 score above a plain 公园."""
    low = str(name).lower()
    if any(k in low for k in _WALK_NATURE_KEYWORDS):
        return 1.8
    if any(k in low for k in _WALK_STREET_KEYWORDS):
        return 1.3
    if any(k in low for k in _LOW_WALK_KEYWORDS):
        return 0.7
    return default_per_poi


def estimate_day_walking_km(day_item: Dict[str, Any], default_per_poi: float = 1.0) -> float:
    """Content-based per-day walking estimate (km), varying with POI type and count.

    Per POI: large nature/scenic ~1.8 km, street/park/square ~1.3 km, indoor/museum
    ~0.7 km, others ``default_per_poi`` (1.0); plus inter-POI transfer walking and
    split-activity coordination.  Returns 0.0 when the day has no concrete POIs.
    """
    pois = _poi_names_from_day(day_item)
    if not pois:
        return 0.0
    total = sum(_poi_in_venue_walk_km(name, default_per_poi) for name in pois)
    total += max(0, len(pois) - 1) * 0.4  # walking between stops
    split = day_item.get("split_activities", []) if isinstance(day_item, dict) else []
    if isinstance(split, list):
        total += len(split) * 0.5
    return round(total, 1)


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(data: Any, path: Path):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def build_agent_lookup(agents: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup = {}
    for agent in agents:
        agent_id = safe_text(agent.get("agent_id"))
        if agent_id:
            lookup[agent_id] = agent
    return lookup


def build_family_agents(agents: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    family_map: Dict[int, List[Dict[str, Any]]] = {}
    for agent in agents:
        family_id = safe_int(agent.get("family_id"), 0)
        if family_id not in family_map:
            family_map[family_id] = []
        family_map[family_id].append(agent)
    return family_map


def load_conflict_results(path: Path) -> Dict[int, Dict[str, Any]]:
    data = load_json_file(path)
    conflict_lookup = {}
    for item in data:
        family_id = safe_int(item.get("family_id"))
        conflict_lookup[family_id] = item
    return conflict_lookup


# Single source of truth for unresolved-conflict severity weights (was duplicated verbatim in
# module5's consensus-status self-check and module6's evaluation penalty). Both stages weigh the
# same conflict types identically; keep them here so tuning one place can't drift from the other.
UNRESOLVED_CONFLICT_WEIGHTS = {
    "direct_must_include_must_avoid_conflict": 24.0,
    "must_include_vs_must_avoid_term_conflict": 16.0,
    "preference_vs_must_avoid_term_conflict": 8.0,
    "hard_constraint_feasibility_conflict": 16.0,
}


def load_backtracking_feedback_lookup(
    path: Path = BACKTRACKING_FEEDBACK_JSON,
) -> Dict[int, Dict[str, Any]]:
    """Load Module 7's targeted-backtracking feedback as {family_id: item}. Shared by Module 2
    and Module 4 (was defined identically in both)."""
    if path is None or not Path(path).exists():
        return {}
    try:
        data = load_json_file(Path(path))
    except Exception:
        return {}
    lookup: Dict[int, Dict[str, Any]] = {}
    if isinstance(data, dict):
        items = data.get("failed_families", [])
    elif isinstance(data, list):
        items = data
    else:
        items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        family_id = safe_int(item.get("family_id"), 0)
        if family_id:
            lookup[family_id] = item
    return lookup


def _as_text_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [safe_text(x) for x in value if safe_text(x)]
    text = safe_text(value)
    return [text] if text else []


def safe_float(value: Any, default=np.nan):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def clamp_score(score: float) -> float:
    return max(0.0, min(100.0, float(score)))


def parse_json_text(value: Any, default=None):
    if default is None:
        default = {}
    text = safe_text(value)
    if text == "":
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def parse_json_list(value: Any) -> List[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if text == "":
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        return [str(data).strip()]
    except Exception:
        parts = re.split("[,;/|]", text)
        return [p.strip() for p in parts if p.strip()]


def normalize_list(items: Any) -> List[str]:
    if not isinstance(items, list):
        items = parse_json_list(items)
    return [str(x).strip().lower() for x in items if str(x).strip()]


def extract_json(text: Any) -> Dict[str, Any]:
    if text is None:
        raise ValueError("模型返回内容为空")
    text = str(text).strip()
    if text == "":
        raise ValueError("模型返回内容是空字符串")
    text = re.sub("^```json\\s*", "", text)
    text = re.sub("^```\\s*", "", text)
    text = re.sub("\\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"没有找到 JSON 内容: {repr(text[:300])}")
    return json.loads(text[start : end + 1])


def compact_json_for_prompt(data: Any, max_chars: int = 5000) -> str:
    try:
        return compact_json(data, max_chars=max_chars)
    except Exception:
        text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n... TRUNCATED FOR PROMPT ..."


def time_to_minutes(time_text: Any) -> Optional[int]:
    text = safe_text(time_text)
    if text == "":
        return None
    match = re.search("(\\d{1,2}):(\\d{2})", text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    return hour * 60 + minute


def average_per_day(total_value, day) -> float:
    value = safe_float(total_value, np.nan)
    day_value = safe_float(day, 1)
    if pd.isna(value):
        return np.nan
    if pd.isna(day_value) or day_value <= 0:
        day_value = 1
    return value / day_value


def normalize_role_key(role: str, age_stage: str = "") -> str:
    role_lower = safe_text(role).lower()
    age_lower = safe_text(age_stage).lower()
    if "father" in role_lower or "dad" in role_lower or "爸爸" in role_lower or ("父亲" in role_lower):
        return "father"
    if (
        "mother" in role_lower
        or "mom" in role_lower
        or "mum" in role_lower
        or ("妈妈" in role_lower)
        or ("母亲" in role_lower)
    ):
        return "mother"
    if (
        "teenager" in role_lower
        or "teen" in role_lower
        or "adolescent" in role_lower
        or ("teenager" in age_lower)
        or ("teen" in age_lower)
        or ("青少年" in role_lower)
    ):
        return "teenager"
    if (
        "child" in role_lower
        or "kid" in role_lower
        or "younger" in role_lower
        or ("child" in age_lower)
        or ("kid" in age_lower)
        or ("儿童" in role_lower)
        or ("孩子" in role_lower)
    ):
        return "child"
    if (
        "older" in role_lower
        or "elderly" in role_lower
        or "elder" in role_lower
        or ("grandfather" in role_lower)
        or ("grandmother" in role_lower)
        or ("grandparent" in role_lower)
        or ("older" in age_lower)
        or ("elderly" in age_lower)
        or ("老人" in role_lower)
        or ("爷爷" in role_lower)
        or ("奶奶" in role_lower)
        or ("外公" in role_lower)
        or ("外婆" in role_lower)
    ):
        return "older"
    return "other"


def price_level_to_score(price_level: str) -> int:
    value = safe_text(price_level).lower()
    if value in ["low", "cheap", "budget", "affordable"]:
        return 1
    if value in ["high", "expensive", "luxury"]:
        return 3
    return 2


def pace_to_score(pace: str) -> int:
    value = safe_text(pace).lower()
    if value == "relaxed":
        return 1
    if value == "compact":
        return 3
    return 2


# ---------------------------------------------------------------------
# Integrated helper sections below reduce extra helper .py files.
# ---------------------------------------------------------------------

# Integrated: prompt budgeting and compact LLM inputs.


def approx_token_count(value: Any) -> int:
    text = safe_text(value)
    if not text:
        return 0
    # Conservative mixed Chinese/English approximation.
    return max(1, len(text) // 3)


def truncate_text(text: Any, max_chars: int = 1200, keep_tail_chars: int = 0) -> str:
    text = safe_text(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if keep_tail_chars and keep_tail_chars < max_chars:
        head_chars = max_chars - keep_tail_chars - 80
        if head_chars < 120:
            head_chars = max_chars // 2
            keep_tail_chars = max_chars - head_chars - 80
        return (
            text[:head_chars].rstrip()
            + "\n... TRUNCATED BY PROMPT_BUDGET ...\n"
            + text[-keep_tail_chars:].lstrip()
        )
    return text[:max_chars].rstrip() + "\n... TRUNCATED BY PROMPT_BUDGET ..."


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _short_list(value: Any, max_items: int = 3, max_chars_each: int = 120) -> List[Any]:
    output = []
    for item in _as_list(value)[:max_items]:
        if isinstance(item, dict):
            output.append(
                compact_dict(item, max_keys=6, max_list_items=2, max_string_chars=max_chars_each)
            )
        elif isinstance(item, list):
            output.append([truncate_text(x, max_chars_each) for x in item[:2]])
        else:
            text = truncate_text(item, max_chars_each)
            if text:
                output.append(text)
    return output


def compact_dict(
    data: Dict[str, Any], max_keys: int = 12, max_list_items: int = 3, max_string_chars: int = 180
) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    output = {}
    for index, (key, value) in enumerate(data.items()):
        if index >= max_keys:
            output["_truncated_keys"] = True
            break
        if isinstance(value, str):
            output[key] = truncate_text(value, max_string_chars)
        elif isinstance(value, list):
            output[key] = _short_list(
                value, max_items=max_list_items, max_chars_each=max_string_chars
            )
        elif isinstance(value, dict):
            output[key] = compact_dict(
                value,
                max_keys=max(4, max_keys // 2),
                max_list_items=max(1, max_list_items - 1),
                max_string_chars=max_string_chars,
            )
        else:
            output[key] = value
    return output


def compact_json(data: Any, max_chars: int = 2000) -> str:
    try:
        compacted = compact_object(data, max_chars=max_chars)
        text = json.dumps(compacted, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        text = safe_text(data)
    if len(text) <= max_chars:
        return text
    return truncate_text(text, max_chars=max_chars, keep_tail_chars=min(400, max_chars // 3))


def compact_object(data: Any, max_chars: int = 2000) -> Any:
    if isinstance(data, dict):
        # Preserve important fields first.
        priority_keys = [
            "family_id",
            "agent_id",
            "member_id",
            "role",
            "age_stage",
            "status",
            "must_include",
            "must_avoid",
            "max_walking_distance_km",
            "latest_return_time",
            "short_self_summary",
            "proposal_priority",
            "non_negotiable_parts",
            "negotiable_parts",
            "hard_constraint_failed_reasons",
            "failed_reasons",
            "lowest_satisfaction_member",
            "conflict_type",
            "severity",
            "resolution",
            "consensus_status",
            "acceptance_vote_ratio",
            "final_itinerary_summary",
            "core_consensus_plan",
            "split_activities",
        ]
        ordered = {}
        for key in priority_keys:
            if key in data:
                ordered[key] = data[key]
        for key, value in data.items():
            if key not in ordered:
                ordered[key] = value
        return compact_dict(ordered, max_keys=16, max_list_items=3, max_string_chars=220)
    if isinstance(data, list):
        return [compact_object(item, max_chars=max_chars // 2) for item in data[:6]]
    if isinstance(data, str):
        return truncate_text(data, max_chars=max(200, min(max_chars, 1200)))
    return data


def compact_agent_for_llm(agent: Dict[str, Any]) -> Dict[str, Any]:
    profile = agent.get("profile", {}) if isinstance(agent, dict) else {}
    constraints = agent.get("constraints", {}) if isinstance(agent, dict) else {}
    return {
        "agent_id": agent.get("agent_id"),
        "family_id": agent.get("family_id"),
        "member_id": agent.get("member_id"),
        "role": profile.get("role"),
        "age_stage": profile.get("age_stage"),
        "interests": _short_list(profile.get("interest_preferences", []), 3, 80),
        "physical": profile.get("physical_constraint"),
        "budget": profile.get("budget_sensitivity"),
        "diet": _short_list(profile.get("dietary_preference", []), 3, 80),
        "pace": profile.get("travel_pace"),
        "must_include": _short_list(constraints.get("must_include", []), 4, 90),
        "must_avoid": _short_list(constraints.get("must_avoid", []), 4, 90),
        "max_walk_km": constraints.get("max_walking_distance_km"),
        "latest_return": constraints.get("latest_return_time"),
    }


def compact_proposal_for_llm(item: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    proposal = (
        item.get("proposal")
        or item.get("individual_proposal")
        or item.get("individual_proposal_json")
        or item
    )
    if isinstance(proposal, str):
        try:
            proposal = json.loads(proposal)
        except Exception:
            proposal = {"summary": proposal}
    if not isinstance(proposal, dict):
        proposal = {}
    return {
        "agent_id": item.get("agent_id") or proposal.get("agent_id"),
        "role": item.get("role") or proposal.get("role"),
        "summary": truncate_text(proposal.get("short_self_summary", ""), 160),
        "priority": _short_list(proposal.get("proposal_priority", []), 4, 130),
        "preferred": _short_list(proposal.get("preferred_attractions", []), 3, 120),
        "rejected": _short_list(proposal.get("rejected_arrangements", []), 2, 120),
        "non_negotiable": _short_list(proposal.get("non_negotiable_parts", []), 3, 120),
        "negotiable": _short_list(proposal.get("negotiable_parts", []), 3, 120),
    }


def compact_conflicts_for_llm(conflicts: Any, max_items: int = 8) -> List[Dict[str, Any]]:
    output = []
    for item in _as_list(conflicts)[:max_items]:
        if isinstance(item, dict):
            output.append(
                {
                    "conflict_type": item.get("conflict_type") or item.get("type"),
                    "severity": item.get("severity") or item.get("level"),
                    "agents": _short_list(
                        item.get("affected_agents", item.get("agents", [])), 4, 80
                    ),
                    "terms": _short_list(
                        item.get("conflicting_terms", item.get("terms", [])), 4, 80
                    ),
                    "resolution_hint": truncate_text(
                        item.get("resolution_hint")
                        or item.get("suggested_resolution")
                        or item.get("reason", ""),
                        160,
                    ),
                }
            )
        else:
            output.append({"conflict": truncate_text(item, 160)})
    return output


def compact_negotiation_round_for_llm(items: Any, max_items: int = 6) -> List[Dict[str, Any]]:
    output = []
    for item in _as_list(items)[:max_items]:
        if isinstance(item, dict):
            output.append(compact_dict(item, max_keys=8, max_list_items=3, max_string_chars=140))
        else:
            output.append({"item": truncate_text(item, 140)})
    return output


def compact_plan_for_llm(plan: Any) -> Dict[str, Any]:
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except Exception:
            return {"summary": truncate_text(plan, 500)}
    if not isinstance(plan, dict):
        return {}
    return {
        "family_id": plan.get("family_id"),
        "summary": truncate_text(
            plan.get("final_itinerary_summary")
            or plan.get("itinerary_text")
            or plan.get("coordinator_summary", ""),
            450,
        ),
        "status": plan.get("consensus_status") or plan.get("status"),
        "acceptance_vote_ratio": plan.get("acceptance_vote_ratio"),
        "core_plan": _short_list(
            plan.get("core_consensus_plan") or plan.get("per_day_itinerary") or [], 5, 160
        ),
        "split_activities": _short_list(plan.get("split_activities", []), 4, 140),
        "unresolved": _short_list(plan.get("unresolved_conflicts", []), 4, 140),
    }


def compact_feedback_for_llm(feedback: Any, max_chars: int = 1400) -> str:
    text = safe_text(feedback)
    if not text:
        return ""
    priority_keywords = [
        "hard_constraint",
        "must_include",
        "must_avoid",
        "walking",
        "return_time",
        "budget",
        "lowest",
        "minimum_member",
        "reject",
        "acceptance",
        "objective",
        "split",
        "unresolved",
        "resolved_conflicts",
        "constraint_satisfiability",
        "direct_must_include_must_avoid_conflict",
        "must_include_vs_must_avoid_term_conflict",
        "preference_vs_must_avoid_term_conflict",
        "hard_constraint_feasibility_conflict",
        "unsatisfiable",
        "split_activity_cost",
        "unfair",
        "failed",
        "failure",
        "修复",
        "失败",
        "硬约束",
        "最低",
        "不公平",
        "未解决冲突",
        "直接冲突",
        "分流活动成本",
    ]
    selected = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if any(k.lower() in low for k in priority_keywords):
            selected.append(truncate_text(line, 220))
        if len("\n".join(selected)) >= max_chars:
            break
    if not selected:
        return truncate_text(text, max_chars=max_chars, keep_tail_chars=max_chars // 3)
    return truncate_text(
        "\n".join(dict.fromkeys(selected)), max_chars=max_chars, keep_tail_chars=max_chars // 3
    )


def compact_memory_items(memories: Any, max_items: int = 2, max_chars: int = 500) -> str:
    """Compact selected agent memories for prompts.

    v49 change: when memories include ``memory_relevance_score``, keep the
    highest-scoring items instead of blindly taking the last chronological rows.
    This prevents old or unrelated role-level memories from occupying prompt
    budget and misleading the current agent.
    """
    all_items = _as_list(memories)
    if any(isinstance(x, dict) and "memory_relevance_score" in x for x in all_items):
        items = sorted(
            all_items,
            key=lambda x: safe_float(x.get("memory_relevance_score"), 0.0)
            if isinstance(x, dict)
            else 0.0,
            reverse=True,
        )[:max_items]
    else:
        items = all_items[-max_items:]
    compacted = []
    for item in items:
        if isinstance(item, dict):
            compacted.append(
                compact_dict(item, max_keys=10, max_list_items=2, max_string_chars=120)
            )
        else:
            compacted.append(truncate_text(item, 120))
    return compact_json(compacted, max_chars=max_chars)


def budget_messages(
    messages: List[Dict[str, str]], max_input_tokens: int = 6500
) -> List[Dict[str, str]]:
    if not messages:
        return messages
    max_chars_total = max(1200, int(max_input_tokens) * 3)
    copied = [dict(m) for m in messages]
    total_chars = sum(len(safe_text(m.get("content", ""))) for m in copied)
    if total_chars <= max_chars_total:
        return copied
    # Preserve system messages but cap them; aggressively head-tail user messages.
    user_count = max(1, sum(1 for m in copied if m.get("role") == "user"))
    system_budget = min(1800, max_chars_total // 5)
    remaining_budget = max_chars_total - system_budget
    user_budget_each = max(1200, remaining_budget // user_count)
    for m in copied:
        content = safe_text(m.get("content", ""))
        if m.get("role") == "system":
            m["content"] = truncate_text(content, max_chars=system_budget, keep_tail_chars=300)
        else:
            m["content"] = truncate_text(
                content, max_chars=user_budget_each, keep_tail_chars=max(500, user_budget_each // 3)
            )
    notice = f"\n\n[PromptBudget] Input was automatically compressed to stay under about {max_input_tokens} input tokens. Use the compressed facts only."
    for index in range(len(copied) - 1, -1, -1):
        if copied[index].get("role") == "user":
            copied[index]["content"] = copied[index].get("content", "") + notice
            break
    return copied


# Integrated: role-level reward and penalty memory store.
import threading

"\n所有 role 共享记忆文件默认保存在：\ncode/results/module_2/agent_memories/\n"
DEFAULT_MEMORY_DIR = MEMORY_DIR


MEMORY_RELEVANCE_FILTER_ENABLED = os.getenv(
    "AGENT_MEMORY_RELEVANCE_FILTER_ENABLED", "true"
).lower() in ["1", "true", "yes", "y"]
MEMORY_SCAN_MAX_ITEMS = safe_int(os.getenv("AGENT_MEMORY_SCAN_MAX_ITEMS", "80"), 80)
MEMORY_MIN_RELEVANCE_SCORE = safe_float(os.getenv("AGENT_MEMORY_MIN_RELEVANCE_SCORE", "12"), 12.0)


def _memory_keyword_tokens(value: Any) -> set:
    """Extract coarse bilingual/English keyword tokens for memory relevance."""
    text = safe_text(value).lower()
    if not text:
        return set()
    # Keep words, Chinese chunks, and mixed travel terms.  This is intentionally
    # lightweight; it avoids adding vector libraries and stays deterministic.
    tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{2,}|[\u4e00-\u9fff]{2,}", text))
    stop = {
        "include",
        "avoid",
        "must",
        "preference",
        "family",
        "member",
        "agent",
        "travel",
        "trip",
        "plan",
        "day",
        "with",
        "and",
        "the",
        "for",
        "this",
        "that",
        "activity",
        "restaurant",
        "hotel",
        "transport",
        "budget",
        "安排",
        "成员",
        "家庭",
        "行程",
        "旅游",
        "活动",
        "偏好",
        "约束",
        "方案",
    }
    return {t for t in tokens if t not in stop and len(t) >= 2}


def _agent_memory_context_text(agent: Dict[str, Any]) -> str:
    profile = agent.get("profile", {}) if isinstance(agent.get("profile"), dict) else {}
    constraints = agent.get("constraints", {}) if isinstance(agent.get("constraints"), dict) else {}
    parts = [
        agent.get("agent_id"),
        agent.get("family_id"),
        agent.get("member_id"),
        agent.get("destination_city"),
        agent.get("departure_city"),
        profile.get("role"),
        profile.get("age_stage"),
        profile.get("interest_preferences"),
        profile.get("dietary_preference"),
        profile.get("physical_constraint"),
        profile.get("budget_sensitivity"),
        profile.get("travel_pace"),
        constraints.get("must_include"),
        constraints.get("must_avoid"),
        constraints.get("max_walking_distance_km"),
        constraints.get("latest_return_time"),
        agent.get("preference_weights"),
    ]
    return json.dumps(parts, ensure_ascii=False, default=str)


def score_agent_memory_item(agent: Dict[str, Any], memory: Dict[str, Any], index: int = 0) -> float:
    """Score role-level agent memory for the current agent.\n\n    The score prefers same-agent/same-family recent quality feedback and memories\n    that mention the current constraints, interests, destination, or conflict\n    terms.  It penalizes unrelated init_profile records and vague/empty content.\n    """
    if not isinstance(memory, dict):
        return 0.0

    score = 0.0
    agent_id = safe_text(agent.get("agent_id"))
    family_id = safe_text(agent.get("family_id"))
    member_id = safe_text(agent.get("member_id"))
    role_key = normalize_role_key(
        role=(agent.get("profile", {}) or {}).get("role", ""),
        age_stage=(agent.get("profile", {}) or {}).get("age_stage", ""),
    )

    if safe_text(memory.get("agent_id")) == agent_id:
        score += 30
    if safe_text(memory.get("family_id")) == family_id:
        score += 18
    if safe_text(memory.get("member_id")) == member_id:
        score += 10
    if safe_text(memory.get("role_memory_key")) == role_key:
        score += 8

    memory_type = safe_text(memory.get("memory_type")).lower()
    if "negative_proposal_feedback" in memory_type:
        score += 34
    elif "positive_proposal_feedback" in memory_type:
        score += 26
    elif "proposal_quality" in memory_type:
        score += 24
    elif "negotiation_round_2" in memory_type or "objection" in memory_type:
        score += 20
    elif "negotiation_round_3" in memory_type or "concession" in memory_type:
        score += 18
    elif "negotiation_round_1" in memory_type:
        score += 14
    elif "individual_proposal" in memory_type:
        score += 10
    elif "init_profile" in memory_type:
        # Init profiles are useful only for same agent; old same-role init profiles
        # are usually noise.
        score += 6 if safe_text(memory.get("agent_id")) == agent_id else -18

    content_text = json.dumps(memory.get("content", memory), ensure_ascii=False, default=str)
    agent_tokens = _memory_keyword_tokens(_agent_memory_context_text(agent))
    memory_tokens = _memory_keyword_tokens(content_text)
    overlap = 0
    if agent_tokens and memory_tokens:
        overlap = len(agent_tokens & memory_tokens)
        score += min(20, overlap * 3)

    same_agent = safe_text(memory.get("agent_id")) == agent_id
    same_family = safe_text(memory.get("family_id")) == family_id
    if (
        "individual_proposal" in memory_type
        and (not same_agent)
        and (not same_family)
        and overlap == 0
    ):
        score -= 18
    if "init_profile" in memory_type and (not same_agent):
        score -= 10

    # Quality gate memories with explicit future instructions are valuable.
    if memory.get("future_instruction") or memory.get("why_bad") or memory.get("why_good"):
        score += 10

    # Prefer more recent rows when scores tie, without making recency dominate.
    score += min(5.0, max(0, index) * 0.02)

    text = content_text.lower()
    if not text or text in ["{}", "[]"]:
        score -= 20
    if len(text) > 6000:
        # Very long raw proposal memories are expensive; only keep if strongly relevant.
        score -= 8
    if any(bad in text for bad in ["unknown", "未知", "无关", "irrelevant"]):
        score -= 4

    return round(float(score), 4)


def select_relevant_agent_memories(
    agent: Dict[str, Any], memories: List[Dict[str, Any]], max_items: int = 20
) -> List[Dict[str, Any]]:
    """Return the most relevant memories for an agent, with scores attached."""
    if not MEMORY_RELEVANCE_FILTER_ENABLED or max_items >= 100:
        return memories[-max_items:]
    scan_items = memories[-max(1, MEMORY_SCAN_MAX_ITEMS) :]
    scored = []
    for i, memory in enumerate(scan_items):
        if not isinstance(memory, dict):
            continue
        score = score_agent_memory_item(agent, memory, index=i)
        if score >= MEMORY_MIN_RELEVANCE_SCORE:
            item = dict(memory)
            item["memory_relevance_score"] = score
            scored.append(item)
    scored.sort(key=lambda x: x.get("memory_relevance_score", 0.0), reverse=True)
    # Keep chronological order inside the selected subset only when printed by
    # compact_memory_items with scores; it will sort by score again for prompts.
    return scored[:max_items]


class RoleMemoryStore:
    def __init__(self, memory_dir: Path = DEFAULT_MEMORY_DIR):
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.locks = {}
        self.global_lock = threading.Lock()

    def get_role_key(self, agent: Dict[str, Any]) -> str:
        profile = agent.get("profile", {})
        role = profile.get("role", "")
        age_stage = profile.get("age_stage", "")
        return normalize_role_key(role=role, age_stage=age_stage)

    def get_memory_path(self, agent: Dict[str, Any]) -> Path:
        # Memory is scoped PER FAMILY: backtracking writes a family's role-level errors and
        # only that family's regenerated proposals may read them. A shared role-level file
        # would let later test families absorb earlier families' failure feedback
        # (online test-set adaptation / order dependence). Verified on the 200-family full
        # run: only families 198/200 ever wrote memories, so results are unaffected, but
        # the scope is now enforced by construction rather than by luck.
        role_key = self.get_role_key(agent)
        family_id = safe_int(agent.get("family_id"), 0)
        if family_id > 0:
            return self.memory_dir / f"family_{family_id}_role_{role_key}.jsonl"
        return self.memory_dir / f"role_{role_key}.jsonl"

    def get_lock(self, agent: Dict[str, Any]):
        role_key = self.get_role_key(agent)
        with self.global_lock:
            if role_key not in self.locks:
                self.locks[role_key] = threading.Lock()
            return self.locks[role_key]

    def load_memories(self, agent: Dict[str, Any], max_items: int = 20) -> List[Dict[str, Any]]:
        memory_path = self.get_memory_path(agent)
        lock = self.get_lock(agent)
        with lock:
            if not memory_path.exists():
                return []
            memories = []
            with open(memory_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        memories.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return select_relevant_agent_memories(
                agent=agent, memories=memories, max_items=max_items
            )

    def append_memory(self, agent: Dict[str, Any], memory: Dict[str, Any]):
        memory_path = self.get_memory_path(agent)
        lock = self.get_lock(agent)
        memory["timestamp"] = now_time()
        memory["agent_id"] = agent.get("agent_id")
        memory["family_id"] = agent.get("family_id")
        memory["member_id"] = agent.get("member_id")
        memory["role_memory_key"] = self.get_role_key(agent)
        with lock:
            with open(memory_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(memory, ensure_ascii=False) + "\n")

    def initialize_profile_memory(self, agent: Dict[str, Any]):
        agent_id = agent.get("agent_id")
        memories = self.load_memories(agent=agent, max_items=300)
        has_init_profile = any(
            (
                memory.get("memory_type") == "init_profile" and memory.get("agent_id") == agent_id
                for memory in memories
            )
        )
        if has_init_profile:
            return
        self.append_memory(
            agent=agent,
            memory={
                "memory_type": "init_profile",
                "content": {
                    "family_id": agent.get("family_id"),
                    "member_id": agent.get("member_id"),
                    "agent_id": agent.get("agent_id"),
                    "profile": agent.get("profile", {}),
                    "constraints": agent.get("constraints", {}),
                    "preference_weights": agent.get("preference_weights", {}),
                },
            },
        )

    def save_individual_proposal_memory(
        self, agent: Dict[str, Any], proposal: Dict[str, Any], usage: Dict[str, Any] = None
    ):
        if usage is None:
            usage = {}
        self.append_memory(
            agent=agent,
            memory={
                "memory_type": "individual_proposal",
                "content": {"proposal": proposal, "usage": usage},
            },
        )

    def save_negotiation_memory(
        self, agent: Dict[str, Any], round_name: str, content: Dict[str, Any]
    ):
        self.append_memory(agent=agent, memory={"memory_type": round_name, "content": content})

    def save_proposal_quality_memory(
        self, agent: Dict[str, Any], evaluation: Dict[str, Any], memory_type: str, source: str
    ):
        payload = {
            "memory_type": memory_type,
            "source": source,
            "proposal_score": evaluation.get("individual_proposal_score"),
            "quality_level": evaluation.get("quality_level"),
            "why_bad": evaluation.get("main_issues", [])[:3],
            "why_good": evaluation.get("why_good", [])[:3],
            "future_instruction": evaluation.get("future_instructions", [])[:3],
        }
        self.append_memory(agent=agent, memory=payload)

    def clear_role_memory(self, role_key: str):
        memory_path = self.memory_dir / f"role_{role_key}.jsonl"
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(memory_path, "w", encoding="utf-8") as f:
            f.write("")

    def clear_all_memories(self):
        role_keys = ["father", "mother", "child", "older", "teenager", "other"]
        for role_key in role_keys:
            self.clear_role_memory(role_key)


# Integrated: QG-BMP theory definitions and formulas.
PROPOSAL_QUALITY_WEIGHTS = {
    "hard_constraint_awareness_score": 0.3,
    "preference_specificity_score": 0.2,
    "role_consistency_score": 0.15,
    "negotiation_usefulness_score": 0.15,
    "feasibility_score": 0.1,
    "conflict_risk_control_score": 0.1,
}
PROPOSAL_QUALITY_FORMULA = "0.30*hard_constraint_awareness + 0.20*preference_specificity + 0.15*role_consistency + 0.15*negotiation_usefulness + 0.10*feasibility + 0.10*conflict_risk_control"


def proposal_quality_theory(component_scores: Dict[str, float]) -> Dict[str, Any]:
    return {
        "framework_component": "Proposal Quality Gate",
        "component_scores": component_scores,
        "component_weights": PROPOSAL_QUALITY_WEIGHTS,
        "formula": PROPOSAL_QUALITY_FORMULA,
    }
