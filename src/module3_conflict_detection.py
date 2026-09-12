"""Module 3 - Conflict detection: detect budget/time/intensity/dining and other conflicts across member proposals."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.common import (
    RESULTS_DIR,
    build_agent_lookup,
    load_json_file,
    normalize_list,
    normalize_role_key,
    pace_to_score,
    parse_json_text,
    price_level_to_score,
    safe_float,
    safe_int,
    safe_text,
    save_json_file,
    time_to_minutes,
)

AGENTS_JSON = RESULTS_DIR / "module_1" / "persona_agents.json"
PROPOSALS_CSV = RESULTS_DIR / "module_2" / "individual_agent_proposals.csv"
OUTPUT_CSV = RESULTS_DIR / "module_3" / "conflict_detection_results.csv"
OUTPUT_JSON = RESULTS_DIR / "module_3" / "conflict_detection_results.json"
WALKING_DISTANCE_CONFLICT_THRESHOLD = 2.0
POI_COUNT_CONFLICT_THRESHOLD = 2
RETURN_TIME_CONFLICT_THRESHOLD_MINUTES = 90
LOW_WALKING_LIMIT = 3.0
HIGH_POI_COUNT_LIMIT = 5
LATE_RETURN_TIME = "21:00"


def normalize_constraint_term(term: str) -> str:
    text = safe_text(term).lower()
    text = text.replace("_", " ").replace("-", " ")
    text = text.replace("theme park", "theme parks")
    text = text.replace("museum", "museums")
    text = " ".join(text.split())
    return text


def terms_semantically_overlap(a: str, b: str) -> bool:
    ta = normalize_constraint_term(a)
    tb = normalize_constraint_term(b)
    if not ta or not tb:
        return False
    if ta == tb or ta in tb or tb in ta:
        return True
    groups = [
        {"museums", "art museums", "history museums", "museum"},
        {"theme parks", "amusement parks", "disneyland", "ocean park"},
        {"shopping", "shopping malls", "markets", "street markets"},
        {"crowded", "crowds", "busy places", "tourist crowds"},
        {"long queues", "queues", "waiting lines", "long waiting"},
        {"hiking", "climbing", "steep climbing", "high intensity", "long walking"},
        {"nightlife", "late night", "night market", "late return"},
    ]
    for group in groups:
        if any((x in ta for x in group)) and any((x in tb for x in group)):
            return True
    return False


def term_related_to_any(term: str, keywords: List[str]) -> bool:
    text = normalize_constraint_term(term)
    return any((k in text for k in keywords))


def is_child_or_teenager(proposal: Dict[str, Any]) -> bool:
    role_key = normalize_role_key(proposal.get("role", ""), proposal.get("age_stage", ""))
    return role_key in ["child", "teenager"]


def is_older(proposal: Dict[str, Any]) -> bool:
    role_key = normalize_role_key(proposal.get("role", ""), proposal.get("age_stage", ""))
    return role_key == "older"


def standardize_proposal_row(
    row: pd.Series, agent_lookup: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    agent_id = safe_text(row.get("agent_id", ""))
    agent = agent_lookup.get(agent_id, {})
    profile = agent.get("profile", {})
    constraints = agent.get("constraints", {})
    proposal = parse_json_text(row.get("individual_proposal_json", "{}"), default={})
    daily_bottom = proposal.get("daily_intensity_bottom_line", {})
    dining = proposal.get("dining_preferences", {})
    hotel = proposal.get("hotel_preferences", {})
    preferred_attractions_raw = proposal.get("preferred_attractions", [])
    preferred_attractions = []
    for item in preferred_attractions_raw:
        if isinstance(item, dict):
            preferred_attractions.append(safe_text(item.get("name_or_type", "")))
        else:
            preferred_attractions.append(safe_text(item))
    rejected_raw = proposal.get("rejected_arrangements", [])
    rejected_arrangements = []
    for item in rejected_raw:
        if isinstance(item, dict):
            rejected_arrangements.append(safe_text(item.get("arrangement", "")))
        else:
            rejected_arrangements.append(safe_text(item))
    standardized = {
        "agent_id": agent_id,
        "family_id": safe_int(row.get("family_id"), 0),
        "member_id": safe_int(row.get("member_id"), 0),
        "role": safe_text(row.get("role", profile.get("role", ""))),
        "age_stage": safe_text(row.get("age_stage", profile.get("age_stage", ""))),
        "conflict_level": safe_text(row.get("conflict_level", agent.get("conflict_level", ""))),
        "family_structure": safe_text(
            row.get("family_structure", agent.get("family_structure", ""))
        ),
        "departure_city": safe_text(row.get("departure_city", agent.get("departure_city", ""))),
        "destination_city": safe_text(
            row.get("destination_city", agent.get("destination_city", ""))
        ),
        "day": safe_int(row.get("day", agent.get("day", 0)), 0),
        "query": safe_text(row.get("query", agent.get("query", ""))),
        "interest_preferences": normalize_list(profile.get("interest_preferences", [])),
        "physical_constraint": safe_text(profile.get("physical_constraint", "")),
        "budget_sensitivity": safe_text(profile.get("budget_sensitivity", "")),
        "dietary_preference": normalize_list(profile.get("dietary_preference", [])),
        "travel_pace": safe_text(profile.get("travel_pace", "")),
        "must_include": normalize_list(constraints.get("must_include", [])),
        "must_avoid": normalize_list(constraints.get("must_avoid", [])),
        "persona_max_walking_distance_km": safe_float(
            constraints.get("max_walking_distance_km", None), None
        ),
        "persona_latest_return_time": safe_text(constraints.get("latest_return_time", "")),
        "preferred_attractions": normalize_list(preferred_attractions),
        "rejected_arrangements": normalize_list(rejected_arrangements),
        "proposal_max_walking_distance_km_per_day": safe_float(
            daily_bottom.get("max_walking_distance_km_per_day", None), None
        ),
        "proposal_max_poi_count_per_day": safe_int(
            daily_bottom.get("max_poi_count_per_day", None), None
        ),
        "proposal_latest_return_time": safe_text(daily_bottom.get("latest_return_time", "")),
        "proposal_preferred_pace": safe_text(daily_bottom.get("preferred_pace", "")),
        "proposal_required_rest": bool(daily_bottom.get("required_rest", False)),
        "preferred_food_types": normalize_list(dining.get("preferred_food_types", [])),
        "avoid_food_types": normalize_list(dining.get("avoid_food_types", [])),
        "meal_price_preference": safe_text(dining.get("meal_price_preference", "")),
        "special_dining_requirements": normalize_list(dining.get("special_requirements", [])),
        "hotel_preferred_area": safe_text(hotel.get("preferred_area", "")),
        "hotel_price_level": safe_text(hotel.get("price_level", "")),
        "hotel_comfort_requirement": safe_text(hotel.get("comfort_requirement", "")),
        "hotel_accessibility_requirement": safe_text(hotel.get("accessibility_requirement", "")),
        "negotiable_parts": normalize_list(proposal.get("negotiable_parts", [])),
        "non_negotiable_parts": normalize_list(proposal.get("non_negotiable_parts", [])),
        "proposal_priority": normalize_list(proposal.get("proposal_priority", [])),
        "initial_acceptance_criteria": normalize_list(
            proposal.get("initial_acceptance_criteria", [])
        ),
        "objection_triggers": normalize_list(proposal.get("objection_triggers", [])),
        "short_self_summary": safe_text(proposal.get("short_self_summary", "")),
    }
    return standardized


class ConflictDetector:
    def __init__(self, family_proposals: List[Dict[str, Any]]):
        self.family_proposals = family_proposals
        self.conflicts = []

    def add_conflict(
        self,
        conflict_type: str,
        severity: str,
        involved_agents: List[str],
        description: str,
        evidence: Dict[str, Any],
        suggested_resolution: str,
    ):
        self.conflicts.append(
            {
                "conflict_type": conflict_type,
                "severity": severity,
                "involved_agents": involved_agents,
                "description": description,
                "evidence": evidence,
                "suggested_resolution": suggested_resolution,
            }
        )

    def detect_hard_constraint_feasibility_conflicts(self):
        include_items = []
        avoid_items = []
        walking_limits = {}
        latest_returns = {}
        high_intensity_agents = []
        for p in self.family_proposals:
            agent_id = p["agent_id"]
            role = p.get("role", "")
            for item in p.get("must_include", []):
                if safe_text(item):
                    include_items.append(
                        {"agent_id": agent_id, "role": role, "term": safe_text(item)}
                    )
            for item in p.get("must_avoid", []):
                if safe_text(item):
                    avoid_items.append(
                        {"agent_id": agent_id, "role": role, "term": safe_text(item)}
                    )
            walking_limit = p.get("persona_max_walking_distance_km")
            if walking_limit is not None:
                walking_limits[agent_id] = walking_limit
            latest_time = p.get("persona_latest_return_time")
            minutes = time_to_minutes(latest_time)
            if minutes is not None:
                latest_returns[agent_id] = minutes
            text_for_intensity = " ".join(
                [
                    safe_text(p.get("physical_constraint", "")),
                    safe_text(p.get("travel_pace", "")),
                    " ".join(p.get("interest_preferences", [])),
                    " ".join(p.get("preferred_attractions", [])),
                    " ".join(p.get("must_include", [])),
                ]
            ).lower()
            if any(
                (
                    k in text_for_intensity
                    for k in ["high_intensity", "compact", "theme parks", "hiking", "climbing"]
                )
            ):
                high_intensity_agents.append(agent_id)
        for inc in include_items:
            for avd in avoid_items:
                if inc["agent_id"] == avd["agent_id"]:
                    continue
                if terms_semantically_overlap(inc["term"], avd["term"]):
                    self.add_conflict(
                        conflict_type="hard_constraint_feasibility_conflict",
                        severity="high",
                        involved_agents=list({inc["agent_id"], avd["agent_id"]}),
                        description="A member's must_include conflicts with another member's must_avoid. A fully shared itinerary may be impossible.",
                        evidence={
                            "include_agent": inc,
                            "avoid_agent": avd,
                            "conflict_pattern": "must_include_vs_must_avoid",
                        },
                        suggested_resolution="Use split_activity or optional_activity: include the activity only for the member who requires it, and explicitly mark the avoid member as non_participating with an alternative nearby low-conflict activity.",
                    )
        includes_theme_or_shopping = [
            item
            for item in include_items
            if term_related_to_any(
                item["term"],
                ["theme parks", "amusement", "disneyland", "ocean park", "shopping", "markets"],
            )
        ]
        avoids_crowds_or_queues = [
            item
            for item in avoid_items
            if term_related_to_any(
                item["term"], ["crowded", "crowds", "long queues", "queues", "waiting"]
            )
        ]
        if includes_theme_or_shopping and avoids_crowds_or_queues:
            self.add_conflict(
                conflict_type="crowd_queue_feasibility_conflict",
                severity="high",
                involved_agents=list(
                    {x["agent_id"] for x in includes_theme_or_shopping + avoids_crowds_or_queues}
                ),
                description="Some requested activities are likely to involve crowds or queues, while other members must avoid crowds or long queues.",
                evidence={
                    "crowd_prone_required_items": includes_theme_or_shopping,
                    "crowd_or_queue_avoid_items": avoids_crowds_or_queues,
                },
                suggested_resolution="Schedule such activities only as optional/split activities, choose off-peak hours, or replace them with low-crowd alternatives.",
            )
        if high_intensity_agents:
            low_walking_agents = [
                aid for aid, limit in walking_limits.items() if limit is not None and limit <= 3
            ]
            early_return_agents = [
                aid
                for aid, minutes in latest_returns.items()
                if minutes <= time_to_minutes("19:00")
            ]
            if low_walking_agents or early_return_agents:
                self.add_conflict(
                    conflict_type="intensity_feasibility_conflict",
                    severity="high",
                    involved_agents=list(
                        set(high_intensity_agents + low_walking_agents + early_return_agents)
                    ),
                    description="High-intensity or compact preferences conflict with low walking limits or early return requirements.",
                    evidence={
                        "high_intensity_agents": high_intensity_agents,
                        "low_walking_agents": low_walking_agents,
                        "early_return_agents": early_return_agents,
                    },
                    suggested_resolution="Use a low-intensity shared core plan and make high-intensity activities optional for high-energy members only.",
                )

    def detect_budget_conflicts(self):
        strict_budget_agents = []
        high_cost_agents = []
        meal_price_scores = {}
        hotel_price_scores = {}
        for p in self.family_proposals:
            agent_id = p["agent_id"]
            budget_sensitivity = p["budget_sensitivity"].lower()
            meal_score = price_level_to_score(p["meal_price_preference"])
            hotel_score = price_level_to_score(p["hotel_price_level"])
            meal_price_scores[agent_id] = meal_score
            hotel_price_scores[agent_id] = hotel_score
            if budget_sensitivity == "strict_budget":
                strict_budget_agents.append(agent_id)
            if budget_sensitivity == "experience_first" or meal_score == 3 or hotel_score == 3:
                high_cost_agents.append(agent_id)
        if strict_budget_agents and high_cost_agents:
            self.add_conflict(
                conflict_type="budget_conflict",
                severity="high",
                involved_agents=list(set(strict_budget_agents + high_cost_agents)),
                description="Some members are strict-budget oriented, while others prefer high-cost or experience-first arrangements.",
                evidence={
                    "strict_budget_agents": strict_budget_agents,
                    "high_cost_or_experience_first_agents": high_cost_agents,
                    "meal_price_scores": meal_price_scores,
                    "hotel_price_scores": hotel_price_scores,
                },
                suggested_resolution="Set a family-level budget cap, choose medium-priced hotels/restaurants, and make premium experiences optional.",
            )
        if meal_price_scores:
            if max(meal_price_scores.values()) - min(meal_price_scores.values()) >= 2:
                self.add_conflict(
                    conflict_type="meal_budget_conflict",
                    severity="medium",
                    involved_agents=list(meal_price_scores.keys()),
                    description="Members have very different meal price expectations.",
                    evidence={"meal_price_scores": meal_price_scores},
                    suggested_resolution="Use medium-priced family restaurants by default and provide optional premium dining.",
                )
        if hotel_price_scores:
            if max(hotel_price_scores.values()) - min(hotel_price_scores.values()) >= 2:
                self.add_conflict(
                    conflict_type="hotel_budget_conflict",
                    severity="medium",
                    involved_agents=list(hotel_price_scores.keys()),
                    description="Members have very different hotel price expectations.",
                    evidence={"hotel_price_scores": hotel_price_scores},
                    suggested_resolution="Choose a mid-range family-friendly hotel with good location and accessibility.",
                )

    def detect_time_conflicts(self):
        return_times = {}
        pace_scores = {}
        for p in self.family_proposals:
            agent_id = p["agent_id"]
            latest_time = p["proposal_latest_return_time"] or p["persona_latest_return_time"]
            minutes = time_to_minutes(latest_time)
            if minutes is not None:
                return_times[agent_id] = minutes
            pace = p["proposal_preferred_pace"] or p["travel_pace"]
            pace_scores[agent_id] = pace_to_score(pace)
        if len(return_times) >= 2:
            max_time = max(return_times.values())
            min_time = min(return_times.values())
            if max_time - min_time >= RETURN_TIME_CONFLICT_THRESHOLD_MINUTES:
                self.add_conflict(
                    conflict_type="time_conflict",
                    severity="medium",
                    involved_agents=list(return_times.keys()),
                    description="Members have different acceptable latest return times.",
                    evidence={
                        "latest_return_time_minutes": return_times,
                        "difference_minutes": max_time - min_time,
                    },
                    suggested_resolution="Use the earliest acceptable return time as the default and keep late-night activities optional.",
                )
        if len(pace_scores) >= 2:
            if max(pace_scores.values()) - min(pace_scores.values()) >= 2:
                self.add_conflict(
                    conflict_type="pace_conflict",
                    severity="medium",
                    involved_agents=list(pace_scores.keys()),
                    description="Some members prefer a compact schedule while others prefer a relaxed pace.",
                    evidence={"pace_scores": pace_scores},
                    suggested_resolution="Use a moderate core itinerary and mark extra attractions as optional.",
                )

    def detect_route_distance_conflicts(self):
        walking_limits = {}
        poi_limits = {}
        for p in self.family_proposals:
            agent_id = p["agent_id"]
            walking_limit = (
                p["proposal_max_walking_distance_km_per_day"]
                if p["proposal_max_walking_distance_km_per_day"] is not None
                else p["persona_max_walking_distance_km"]
            )
            if walking_limit is not None:
                walking_limits[agent_id] = walking_limit
            if p["proposal_max_poi_count_per_day"] is not None:
                poi_limits[agent_id] = p["proposal_max_poi_count_per_day"]
        if len(walking_limits) >= 2:
            max_limit = max(walking_limits.values())
            min_limit = min(walking_limits.values())
            if max_limit - min_limit >= WALKING_DISTANCE_CONFLICT_THRESHOLD:
                severity = "high" if min_limit <= LOW_WALKING_LIMIT else "medium"
                self.add_conflict(
                    conflict_type="route_distance_conflict",
                    severity=severity,
                    involved_agents=list(walking_limits.keys()),
                    description="Members have conflicting tolerance for daily walking distance.",
                    evidence={
                        "walking_limits_km_per_day": walking_limits,
                        "difference_km": round(max_limit - min_limit, 2),
                    },
                    suggested_resolution="Design a low-walking core route, add optional extensions for high-energy members, and use taxis or public transport between distant POIs.",
                )
        if len(poi_limits) >= 2:
            max_poi = max(poi_limits.values())
            min_poi = min(poi_limits.values())
            if max_poi - min_poi >= POI_COUNT_CONFLICT_THRESHOLD:
                self.add_conflict(
                    conflict_type="daily_poi_count_conflict",
                    severity="medium",
                    involved_agents=list(poi_limits.keys()),
                    description="Members disagree on how many POIs should be visited per day.",
                    evidence={"max_poi_count_per_day": poi_limits, "difference": max_poi - min_poi},
                    suggested_resolution="Keep 2-4 core POIs per day and provide optional add-ons.",
                )

    def detect_attraction_type_conflicts(self):
        interest_map = {}
        preferred_map = {}
        rejected_map = {}
        must_avoid_map = {}
        for p in self.family_proposals:
            agent_id = p["agent_id"]
            interest_map[agent_id] = set(p["interest_preferences"])
            preferred_map[agent_id] = set(p["preferred_attractions"])
            rejected_map[agent_id] = set(p["rejected_arrangements"])
            must_avoid_map[agent_id] = set(p["must_avoid"])
        all_agents = list(interest_map.keys())
        interest_sets = [item for item in interest_map.values() if item]
        if len(interest_sets) >= 2:
            common_interests = set.intersection(*interest_sets)
            union_interests = set.union(*interest_sets)
            if union_interests:
                overlap_ratio = len(common_interests) / len(union_interests)
                if overlap_ratio < 0.25:
                    self.add_conflict(
                        conflict_type="attraction_type_conflict",
                        severity="medium",
                        involved_agents=all_agents,
                        description="Family members have very different attraction type preferences.",
                        evidence={
                            "agent_interest_map": {k: list(v) for k, v in interest_map.items()},
                            "common_interests": list(common_interests),
                            "union_interests": list(union_interests),
                            "overlap_ratio": round(overlap_ratio, 4),
                        },
                        suggested_resolution="Build a mixed itinerary containing at least one POI type for each member and use optional half-day activities.",
                    )
        for agent_a in all_agents:
            avoided_by_a = rejected_map[agent_a].union(must_avoid_map[agent_a])
            if not avoided_by_a:
                continue
            for agent_b in all_agents:
                if agent_a == agent_b:
                    continue
                preferred_by_b = preferred_map[agent_b].union(interest_map[agent_b])
                conflict_items = []
                for avoid_item in avoided_by_a:
                    for prefer_item in preferred_by_b:
                        if avoid_item and prefer_item:
                            if avoid_item in prefer_item or prefer_item in avoid_item:
                                conflict_items.append(
                                    {"avoid_item": avoid_item, "preferred_item": prefer_item}
                                )
                if conflict_items:
                    self.add_conflict(
                        conflict_type="preference_avoidance_conflict",
                        severity="high",
                        involved_agents=[agent_a, agent_b],
                        description="One member wants to avoid arrangements that another member prefers.",
                        evidence={
                            "avoiding_agent": agent_a,
                            "preferring_agent": agent_b,
                            "conflict_items": conflict_items,
                        },
                        suggested_resolution="Do not make the disputed POI mandatory. Make it optional or replace it with a shared-interest substitute.",
                    )

    def detect_intensity_conflicts(self):
        low_intensity_agents = []
        compact_agents = []
        rest_required_agents = []
        for p in self.family_proposals:
            agent_id = p["agent_id"]
            physical = p["physical_constraint"].lower()
            pace = p["proposal_preferred_pace"] or p["travel_pace"]
            max_poi = p["proposal_max_poi_count_per_day"]
            if physical == "low_intensity":
                low_intensity_agents.append(agent_id)
            if pace == "compact":
                compact_agents.append(agent_id)
            if max_poi is not None and max_poi >= HIGH_POI_COUNT_LIMIT:
                compact_agents.append(agent_id)
            if p["proposal_required_rest"]:
                rest_required_agents.append(agent_id)
        compact_agents = list(set(compact_agents))
        if low_intensity_agents and compact_agents:
            self.add_conflict(
                conflict_type="travel_intensity_conflict",
                severity="high",
                involved_agents=list(set(low_intensity_agents + compact_agents)),
                description="Low-intensity members may conflict with members who prefer compact or high-POI schedules.",
                evidence={
                    "low_intensity_agents": low_intensity_agents,
                    "compact_or_high_poi_agents": compact_agents,
                },
                suggested_resolution="Use a relaxed core schedule, include rest blocks, and make extra POIs optional for high-energy members.",
            )
        if rest_required_agents and compact_agents:
            self.add_conflict(
                conflict_type="rest_vs_compact_schedule_conflict",
                severity="medium",
                involved_agents=list(set(rest_required_agents + compact_agents)),
                description="Some members require rest while others prefer compact schedules.",
                evidence={
                    "rest_required_agents": rest_required_agents,
                    "compact_agents": compact_agents,
                },
                suggested_resolution="Insert fixed rest periods after lunch or before dinner.",
            )

    def detect_dining_conflicts(self):
        preferred_food_map = {}
        avoid_food_map = {}
        dietary_map = {}
        meal_price_scores = {}
        for p in self.family_proposals:
            agent_id = p["agent_id"]
            preferred_food_map[agent_id] = set(p["preferred_food_types"])
            avoid_food_map[agent_id] = set(p["avoid_food_types"])
            dietary_map[agent_id] = set(p["dietary_preference"])
            meal_price_scores[agent_id] = price_level_to_score(p["meal_price_preference"])
        all_agents = list(preferred_food_map.keys())
        for agent_a in all_agents:
            avoid_foods = avoid_food_map[agent_a]
            if not avoid_foods:
                continue
            for agent_b in all_agents:
                if agent_a == agent_b:
                    continue
                preferred_foods = preferred_food_map[agent_b]
                conflict_foods = []
                for avoid_item in avoid_foods:
                    for prefer_item in preferred_foods:
                        if avoid_item and prefer_item:
                            if avoid_item in prefer_item or prefer_item in avoid_item:
                                conflict_foods.append(
                                    {"avoid_food": avoid_item, "preferred_food": prefer_item}
                                )
                if conflict_foods:
                    self.add_conflict(
                        conflict_type="dining_preference_conflict",
                        severity="medium",
                        involved_agents=[agent_a, agent_b],
                        description="One member wants to avoid food types that another member prefers.",
                        evidence={
                            "avoiding_agent": agent_a,
                            "preferring_agent": agent_b,
                            "conflict_foods": conflict_foods,
                        },
                        suggested_resolution="Choose restaurants with diverse menu options and avoid making the disputed food type the only choice.",
                    )
        sensitive_dining_agents = []
        for agent_id, dietary in dietary_map.items():
            if (
                "child_friendly" in dietary
                or "light_food" in dietary
                or "price_sensitive" in dietary
            ):
                sensitive_dining_agents.append(agent_id)
        high_price_agents = [
            agent_id for agent_id, score in meal_price_scores.items() if score == 3
        ]
        if sensitive_dining_agents and high_price_agents:
            self.add_conflict(
                conflict_type="meal_price_and_accessibility_conflict",
                severity="medium",
                involved_agents=list(set(sensitive_dining_agents + high_price_agents)),
                description="Some members need child-friendly, light, or price-sensitive meals, while others prefer high-price dining.",
                evidence={
                    "sensitive_dining_agents": sensitive_dining_agents,
                    "high_price_dining_agents": high_price_agents,
                    "meal_price_scores": meal_price_scores,
                },
                suggested_resolution="Use affordable family-friendly restaurants as default and make premium dining optional.",
            )

    def detect_child_older_friendliness_conflicts(self):
        child_or_teen_agents = []
        older_agents = []
        high_intensity_agents = []
        late_return_agents = []
        no_rest_agents = []
        for p in self.family_proposals:
            agent_id = p["agent_id"]
            if is_child_or_teenager(p):
                child_or_teen_agents.append(agent_id)
            if is_older(p):
                older_agents.append(agent_id)
            pace = p["proposal_preferred_pace"] or p["travel_pace"]
            max_poi = p["proposal_max_poi_count_per_day"]
            walking_limit = p["proposal_max_walking_distance_km_per_day"]
            if pace == "compact":
                high_intensity_agents.append(agent_id)
            if max_poi is not None and max_poi >= HIGH_POI_COUNT_LIMIT:
                high_intensity_agents.append(agent_id)
            if walking_limit is not None and walking_limit >= 6:
                high_intensity_agents.append(agent_id)
            latest_time = p["proposal_latest_return_time"] or p["persona_latest_return_time"]
            latest_minutes = time_to_minutes(latest_time)
            late_limit_minutes = time_to_minutes(LATE_RETURN_TIME)
            if (
                latest_minutes is not None
                and late_limit_minutes is not None
                and (latest_minutes > late_limit_minutes)
            ):
                late_return_agents.append(agent_id)
            if not p["proposal_required_rest"]:
                no_rest_agents.append(agent_id)
        vulnerable_agents = list(set(child_or_teen_agents + older_agents))
        if vulnerable_agents and high_intensity_agents:
            self.add_conflict(
                conflict_type="child_older_vs_high_intensity_conflict",
                severity="high",
                involved_agents=list(set(vulnerable_agents + high_intensity_agents)),
                description="Children, teenagers, or older members may be affected by high-intensity schedules proposed by other members.",
                evidence={
                    "child_or_teen_agents": child_or_teen_agents,
                    "older_agents": older_agents,
                    "high_intensity_agents": list(set(high_intensity_agents)),
                },
                suggested_resolution="Use a child/older-friendly core itinerary and add optional high-intensity activities for others.",
            )
        if vulnerable_agents and late_return_agents:
            self.add_conflict(
                conflict_type="child_older_vs_late_return_conflict",
                severity="medium",
                involved_agents=list(set(vulnerable_agents + late_return_agents)),
                description="Children, teenagers, or older members may not tolerate late return schedules.",
                evidence={
                    "child_or_teen_agents": child_or_teen_agents,
                    "older_agents": older_agents,
                    "late_return_agents": late_return_agents,
                },
                suggested_resolution="Set an earlier default return time and keep evening activities optional.",
            )
        if vulnerable_agents and no_rest_agents:
            self.add_conflict(
                conflict_type="rest_requirement_conflict",
                severity="medium",
                involved_agents=list(set(vulnerable_agents + no_rest_agents)),
                description="Children or older members usually need rest, while some proposals do not include rest blocks.",
                evidence={
                    "child_or_teen_agents": child_or_teen_agents,
                    "older_agents": older_agents,
                    "agents_without_required_rest": no_rest_agents,
                },
                suggested_resolution="Add daily rest blocks, especially after lunch or before dinner.",
            )

    def run_all_detectors(self) -> List[Dict[str, Any]]:
        self.detect_hard_constraint_feasibility_conflicts()
        self.detect_budget_conflicts()
        self.detect_time_conflicts()
        self.detect_route_distance_conflicts()
        self.detect_attraction_type_conflicts()
        self.detect_intensity_conflicts()
        self.detect_dining_conflicts()
        self.detect_child_older_friendliness_conflicts()
        return self.conflicts


def summarize_conflicts(conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(conflicts)
    severity_count = {"low": 0, "medium": 0, "high": 0}
    type_count = {}
    for conflict in conflicts:
        severity = conflict.get("severity", "medium")
        conflict_type = conflict.get("conflict_type", "unknown")
        if severity not in severity_count:
            severity_count[severity] = 0
        severity_count[severity] += 1
        if conflict_type not in type_count:
            type_count[conflict_type] = 0
        type_count[conflict_type] += 1
    if total == 0:
        conflict_level_detected = "No Conflict"
    elif severity_count.get("high", 0) >= 2:
        conflict_level_detected = "High Conflict"
    elif severity_count.get("high", 0) == 1 or severity_count.get("medium", 0) >= 2:
        conflict_level_detected = "Medium Conflict"
    else:
        conflict_level_detected = "Low Conflict"
    return {
        "conflict_count": total,
        "high_conflict_count": severity_count.get("high", 0),
        "medium_conflict_count": severity_count.get("medium", 0),
        "low_conflict_count": severity_count.get("low", 0),
        "conflict_type_count_json": json.dumps(type_count, ensure_ascii=False),
        "conflict_level_detected": conflict_level_detected,
    }


def validate_proposal_csv_columns(df: pd.DataFrame):
    required_columns = ["agent_id", "family_id", "member_id", "individual_proposal_json"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"proposal CSV 缺少字段: {col}")


def detect_conflicts(
    agents_json_path: Path = AGENTS_JSON,
    proposals_csv_path: Path = PROPOSALS_CSV,
    output_csv_path: Path = OUTPUT_CSV,
    output_json_path: Path = OUTPUT_JSON,
    target_family_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    if not agents_json_path.exists():
        raise FileNotFoundError(f"找不到 persona_agents.json，请先运行模块 1: {agents_json_path}")
    if not proposals_csv_path.exists():
        raise FileNotFoundError(
            f"找不到 individual_agent_proposals.csv，请先运行模块 2: {proposals_csv_path}"
        )
    print("[Module 3] Conflict Detection 开始")
    print(f"[Module 3] agents_json_path: {agents_json_path}")
    print(f"[Module 3] proposals_csv_path: {proposals_csv_path}")
    agents = load_json_file(agents_json_path)
    agent_lookup = build_agent_lookup(agents)
    proposal_df = pd.read_csv(proposals_csv_path)
    validate_proposal_csv_columns(proposal_df)
    if target_family_ids is not None:
        target_set = {int(x) for x in target_family_ids}
        proposal_df = proposal_df[proposal_df["family_id"].astype(int).isin(target_set)].copy()
        print(f"[Module 3] Backtracking target mode: 只重新检测未达标家庭 family_id={sorted(target_set)}")
        if proposal_df.empty:
            raise ValueError(
                f"[Module 3] target_family_ids={sorted(target_set)} 在 proposals_csv 中没有对应记录。"
            )
    standardized_proposals = []
    for _, row in proposal_df.iterrows():
        standardized = standardize_proposal_row(row=row, agent_lookup=agent_lookup)
        standardized_proposals.append(standardized)
    std_df = pd.DataFrame(standardized_proposals)
    all_family_outputs = []
    all_json_outputs = []
    for family_id, family_group in std_df.groupby("family_id"):
        family_proposals = family_group.to_dict(orient="records")
        detector = ConflictDetector(family_proposals)
        conflicts = detector.run_all_detectors()
        summary = summarize_conflicts(conflicts)
        family_info = {
            "family_id": int(family_id),
            "member_count": len(family_proposals),
            "conflict_level_original": safe_text(
                family_group["conflict_level"].iloc[0]
                if "conflict_level" in family_group.columns
                else ""
            ),
            "family_structure": safe_text(
                family_group["family_structure"].iloc[0]
                if "family_structure" in family_group.columns
                else ""
            ),
            "departure_city": safe_text(
                family_group["departure_city"].iloc[0]
                if "departure_city" in family_group.columns
                else ""
            ),
            "destination_city": safe_text(
                family_group["destination_city"].iloc[0]
                if "destination_city" in family_group.columns
                else ""
            ),
            "day": safe_int(family_group["day"].iloc[0] if "day" in family_group.columns else 0, 0),
            "query": safe_text(
                family_group["query"].iloc[0] if "query" in family_group.columns else ""
            ),
        }
        output_row = {
            **family_info,
            **summary,
            "conflicts_json": json.dumps(conflicts, ensure_ascii=False),
        }
        all_family_outputs.append(output_row)
        all_json_outputs.append({**family_info, **summary, "conflicts": conflicts})
    result_df = pd.DataFrame(all_family_outputs)
    result_df = result_df.sort_values(by=["family_id"])
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    save_json_file(data=all_json_outputs, path=output_json_path)
    print("[Module 3] Conflict Detection 完成")
    print(f"[Module 3] CSV 输出文件: {output_csv_path}")
    print(f"[Module 3] JSON 输出文件: {output_json_path}")
    print("[Module 3] 前 5 个家庭冲突检测结果预览：")
    preview_cols = [
        "family_id",
        "member_count",
        "conflict_level_original",
        "conflict_level_detected",
        "conflict_count",
        "high_conflict_count",
        "medium_conflict_count",
        "low_conflict_count",
        "conflict_type_count_json",
    ]
    available_cols = [col for col in preview_cols if col in result_df.columns]
    print(result_df[available_cols].head(5).to_string(index=False))
    return result_df
