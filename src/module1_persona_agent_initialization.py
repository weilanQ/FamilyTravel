"""Module 1 - Persona agent initialization: build per-member persona agents (profile, constraints, preference weights, system prompt) from the family CSV."""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.common import (
    DATA_DIR,
    RESULTS_DIR,
    normalize_role_key,
    parse_json_list,
    safe_float,
    safe_int,
    safe_text,
    save_json_file,
)

PERSONA_CSV = DATA_DIR / "FamilyTravel_Mini.csv"
OUTPUT_JSON = RESULTS_DIR / "module_1" / "persona_agents.json"


@dataclass
class PersonaProfile:
    age_stage: str
    role: str
    interest_preferences: List[str]
    physical_constraint: str
    budget_sensitivity: str
    dietary_preference: List[str]
    travel_pace: str


@dataclass
class PersonaConstraints:
    must_include: List[str]
    must_avoid: List[str]
    max_walking_distance_km: Optional[float]
    latest_return_time: str


@dataclass
class PersonaAgent:
    agent_id: str
    family_id: int
    member_id: int
    conflict_level: str
    family_structure: str
    departure_city: str
    destination_city: str
    day: int
    query: str
    profile: PersonaProfile
    constraints: PersonaConstraints
    preference_weights: Dict[str, float]
    system_prompt: str

    def to_dict(self):
        return asdict(self)


def infer_age_stage(role: str) -> str:
    role_lower = safe_text(role).lower()
    if "teen" in role_lower or "teenager" in role_lower:
        return "teenager"
    if "child" in role_lower or "kid" in role_lower or "younger" in role_lower:
        return "child"
    if (
        "grandfather" in role_lower
        or "grandmother" in role_lower
        or "grandparent" in role_lower
        or ("elderly" in role_lower)
        or ("elder" in role_lower)
        or ("older" in role_lower)
    ):
        return "elderly"
    return "adult"


def normalize_physical_constraint(value: str) -> str:
    value = safe_text(value).lower()
    if value in ["low_intensity", "medium_intensity", "high_intensity"]:
        return value
    if "low" in value:
        return "low_intensity"
    if "high" in value:
        return "high_intensity"
    return "medium_intensity"


def normalize_budget_sensitivity(value: str) -> str:
    value = safe_text(value).lower()
    if value in ["strict_budget", "moderate_budget", "experience_first"]:
        return value
    if "strict" in value or "price" in value or "cheap" in value:
        return "strict_budget"
    if "experience" in value:
        return "experience_first"
    return "moderate_budget"


def normalize_travel_pace(value: str) -> str:
    value = safe_text(value).lower()
    if value in ["compact", "moderate", "relaxed"]:
        return value
    if "compact" in value or "fast" in value or "tight" in value:
        return "compact"
    if "relax" in value or "slow" in value:
        return "relaxed"
    return "moderate"


def build_preference_weights(
    role: str, physical_constraint: str, budget_sensitivity: str, travel_pace: str
) -> Dict[str, float]:
    role_key = normalize_role_key(role)
    weights = {
        "interest_weight": 0.2,
        "physical_weight": 0.2,
        "budget_weight": 0.2,
        "food_weight": 0.15,
        "pace_weight": 0.15,
        "hard_constraint_weight": 0.1,
    }
    if role_key == "father":
        weights = {
            "interest_weight": 0.15,
            "physical_weight": 0.15,
            "budget_weight": 0.3,
            "food_weight": 0.1,
            "pace_weight": 0.15,
            "hard_constraint_weight": 0.15,
        }
    elif role_key == "mother":
        weights = {
            "interest_weight": 0.18,
            "physical_weight": 0.18,
            "budget_weight": 0.18,
            "food_weight": 0.22,
            "pace_weight": 0.14,
            "hard_constraint_weight": 0.1,
        }
    elif role_key == "child":
        weights = {
            "interest_weight": 0.25,
            "physical_weight": 0.25,
            "budget_weight": 0.05,
            "food_weight": 0.2,
            "pace_weight": 0.15,
            "hard_constraint_weight": 0.1,
        }
    elif role_key == "teenager":
        weights = {
            "interest_weight": 0.35,
            "physical_weight": 0.1,
            "budget_weight": 0.1,
            "food_weight": 0.15,
            "pace_weight": 0.2,
            "hard_constraint_weight": 0.1,
        }
    elif role_key == "older":
        weights = {
            "interest_weight": 0.1,
            "physical_weight": 0.35,
            "budget_weight": 0.1,
            "food_weight": 0.15,
            "pace_weight": 0.2,
            "hard_constraint_weight": 0.1,
        }
    if budget_sensitivity == "strict_budget":
        weights["budget_weight"] += 0.1
        weights["interest_weight"] -= 0.05
        weights["food_weight"] -= 0.05
    if budget_sensitivity == "experience_first":
        weights["interest_weight"] += 0.1
        weights["budget_weight"] -= 0.1
    if physical_constraint == "low_intensity":
        weights["physical_weight"] += 0.1
        weights["pace_weight"] += 0.05
        weights["interest_weight"] -= 0.05
        weights["budget_weight"] -= 0.05
    if travel_pace == "relaxed":
        weights["pace_weight"] += 0.1
        weights["interest_weight"] -= 0.05
        weights["budget_weight"] -= 0.05
    if travel_pace == "compact":
        weights["interest_weight"] += 0.05
        weights["pace_weight"] += 0.05
        weights["physical_weight"] -= 0.05
        weights["food_weight"] -= 0.05
    for key in weights:
        weights[key] = max(weights[key], 0.01)
    total = sum(weights.values())
    if total <= 0:
        total = 1
    weights = {key: round(value / total, 4) for key, value in weights.items()}
    return weights


def build_system_prompt(
    agent_id: str,
    family_id: int,
    member_id: int,
    profile: PersonaProfile,
    constraints: PersonaConstraints,
) -> str:
    prompt = f"\nYou are an independent persona agent in a multi-agent family travel planning framework.\nYour agent_id is {agent_id}.\nYou represent one family member, not the entire family.\nFamily information:\n- family_id: {family_id}\n- member_id: {member_id}\n- role: {profile.role}\n- age_stage: {profile.age_stage}\nYour personal preferences:\n- interest_preferences: {profile.interest_preferences}\n- physical_constraint: {profile.physical_constraint}\n- budget_sensitivity: {profile.budget_sensitivity}\n- dietary_preference: {profile.dietary_preference}\n- travel_pace: {profile.travel_pace}\nYour hard constraints:\n- must_include: {constraints.must_include}\n- must_avoid: {constraints.must_avoid}\n- max_walking_distance_km: {constraints.max_walking_distance_km}\n- latest_return_time: {constraints.latest_return_time}\nBehavior rules:\n1. Evaluate travel plans only from your own perspective.\n2. Always protect your hard constraints.\n3. If a plan violates your hard constraints, raise an objection.\n4. If a plan partially satisfies your preferences, explain what is missing.\n5. Be concise, realistic, and consistent with your persona.\n6. Do not represent other family members.\n7. Do not invent new constraints that are not in your profile.\n8. Output token limit: if asked to respond, keep output under 300 tokens.\n"
    return prompt.strip()


def create_persona_agent_from_row(row: pd.Series) -> PersonaAgent:
    family_id = safe_int(row.get("family_id"), 0)
    member_id = safe_int(row.get("member_id"), 0)
    role = safe_text(row.get("role", ""))
    if "age_stage" in row.index and safe_text(row.get("age_stage", "")):
        age_stage = safe_text(row.get("age_stage", "")).lower()
    else:
        age_stage = infer_age_stage(role)
    interest_preferences = parse_json_list(row.get("interest_preferences", "[]"))
    physical_constraint = normalize_physical_constraint(
        row.get("physical_constraint", "medium_intensity")
    )
    budget_sensitivity = normalize_budget_sensitivity(
        row.get("budget_sensitivity", "moderate_budget")
    )
    dietary_preference = parse_json_list(row.get("dietary_preference", "[]"))
    travel_pace = normalize_travel_pace(row.get("travel_pace", "moderate"))
    must_include = parse_json_list(row.get("must_include", "[]"))
    must_avoid = parse_json_list(row.get("must_avoid", "[]"))
    max_walking_distance_km = safe_float(row.get("max_walking_distance_km", None), None)
    latest_return_time = safe_text(row.get("latest_return_time", ""))
    conflict_level = safe_text(row.get("conflict_level", ""))
    family_structure = safe_text(row.get("family_structure", ""))
    departure_city = safe_text(row.get("departure_city", ""))
    destination_city = safe_text(row.get("destination_city", ""))
    day = safe_int(row.get("day", 0), 0)
    query = safe_text(row.get("query", ""))
    agent_id = f"family_{family_id}_member_{member_id}"
    profile = PersonaProfile(
        age_stage=age_stage,
        role=role,
        interest_preferences=interest_preferences,
        physical_constraint=physical_constraint,
        budget_sensitivity=budget_sensitivity,
        dietary_preference=dietary_preference,
        travel_pace=travel_pace,
    )
    constraints = PersonaConstraints(
        must_include=must_include,
        must_avoid=must_avoid,
        max_walking_distance_km=max_walking_distance_km,
        latest_return_time=latest_return_time,
    )
    preference_weights = build_preference_weights(
        role=role,
        physical_constraint=physical_constraint,
        budget_sensitivity=budget_sensitivity,
        travel_pace=travel_pace,
    )
    system_prompt = build_system_prompt(
        agent_id=agent_id,
        family_id=family_id,
        member_id=member_id,
        profile=profile,
        constraints=constraints,
    )
    agent = PersonaAgent(
        agent_id=agent_id,
        family_id=family_id,
        member_id=member_id,
        conflict_level=conflict_level,
        family_structure=family_structure,
        departure_city=departure_city,
        destination_city=destination_city,
        day=day,
        query=query,
        profile=profile,
        constraints=constraints,
        preference_weights=preference_weights,
        system_prompt=system_prompt,
    )
    return agent


def validate_persona_csv_columns(df: pd.DataFrame):
    required_columns = [
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
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"persona CSV 缺少必要字段: {col}")


def initialize_persona_agents(
    persona_csv_path: Path = PERSONA_CSV, output_json_path: Path = OUTPUT_JSON
) -> List[Dict]:
    if not persona_csv_path.exists():
        raise FileNotFoundError(f"找不到 persona CSV 文件: {persona_csv_path}")
    df = pd.read_csv(persona_csv_path)
    validate_persona_csv_columns(df)
    agents = []
    for _, row in df.iterrows():
        agent = create_persona_agent_from_row(row)
        agents.append(agent.to_dict())
    save_json_file(data=agents, path=output_json_path)
    print("[Module 1] Persona Agent Initialization 完成")
    print(f"[Module 1] 输入文件: {persona_csv_path}")
    print(f"[Module 1] 输出文件: {output_json_path}")
    print(f"[Module 1] 初始化 agent 数量: {len(agents)}")
    print("[Module 1] 前 5 个 agent 预览：")
    for agent in agents[:5]:
        profile = agent.get("profile", {})
        constraints = agent.get("constraints", {})
        print("-" * 80)
        print(f"agent_id: {agent.get('agent_id')}")
        print(f"family_id: {agent.get('family_id')}")
        print(f"member_id: {agent.get('member_id')}")
        print(f"role: {profile.get('role')}")
        print(f"age_stage: {profile.get('age_stage')}")
        print(f"interest_preferences: {profile.get('interest_preferences')}")
        print(f"physical_constraint: {profile.get('physical_constraint')}")
        print(f"budget_sensitivity: {profile.get('budget_sensitivity')}")
        print(f"dietary_preference: {profile.get('dietary_preference')}")
        print(f"travel_pace: {profile.get('travel_pace')}")
        print(f"must_include: {constraints.get('must_include')}")
        print(f"must_avoid: {constraints.get('must_avoid')}")
        print(f"max_walking_distance_km: {constraints.get('max_walking_distance_km')}")
        print(f"latest_return_time: {constraints.get('latest_return_time')}")
        print(f"preference_weights: {agent.get('preference_weights')}")
    return agents
