"""Prompt templates for individual-proposal generation and online grounding (hotel/transport/attraction)."""
import json
from typing import Any, Dict, List


def build_individual_proposal_messages_template(
    agent: Dict[str, Any],
    profile: Dict[str, Any],
    constraints: Dict[str, Any],
    preference_weights: Dict[str, Any],
    memory_text: str,
    feedback_text: str,
    attempt_index: int,
) -> List[Dict[str, str]]:
    system_prompt = """
You are an independent persona agent in a multi-agent family travel planning framework.
You only represent yourself, not the whole family.
Your task is to generate a SHORT individual travel proposal before group negotiation.
You must output valid JSON only.
Do not output markdown.
Do not output explanations outside JSON.
Keep every field concise.
Output token limit: stay under 650 output tokens.
""".strip()
    user_prompt = f"""
Generate a SHORT individual travel proposal in JSON format.

Identity:
agent_id: {agent.get('agent_id')}
family_id: {agent.get('family_id')}
member_id: {agent.get('member_id')}
role: {profile.get('role')}
age_stage: {profile.get('age_stage')}
departure_city: {agent.get('departure_city')}
destination_city: {agent.get('destination_city')}
day: {agent.get('day')}
query: {agent.get('query')}
conflict_level: {agent.get('conflict_level')}
family_structure: {agent.get('family_structure')}

Profile:
interest_preferences: {profile.get('interest_preferences')}
physical_constraint: {profile.get('physical_constraint')}
budget_sensitivity: {profile.get('budget_sensitivity')}
dietary_preference: {profile.get('dietary_preference')}
travel_pace: {profile.get('travel_pace')}

Hard constraints:
must_include: {constraints.get('must_include')}
must_avoid: {constraints.get('must_avoid')}
max_walking_distance_km: {constraints.get('max_walking_distance_km')}
latest_return_time: {constraints.get('latest_return_time')}

Preference weights:
{json.dumps(preference_weights, ensure_ascii=False)}

Short role-shared memory:
{memory_text}

Current regeneration attempt: {attempt_index}
Short feedback to fix, if any:
{feedback_text}

Required output rules:
- Output token limit: stay under 650 output tokens.
- If close to the limit, keep required keys and summarize optional details.
- short_self_summary: no more than 20 English words.
- preferred_attractions: at most 3 items.
- rejected_arrangements: at most 2 items.
- proposal_priority: at least 3 and at most 5 items.
- initial_acceptance_criteria: at most 3 items.
- objection_triggers: at most 3 items.
- negotiable_parts: at most 3 items.
- non_negotiable_parts: at most 3 items.
- proposal_priority items must contain only rank, need, type, flexibility. Do not include importance or reason.
- Every reason field outside proposal_priority: no more than 6 English words.
- Use short factual phrases. Do not write paragraphs.
- Include must_include and must_avoid needs in proposal_priority when possible.
- If a need may conflict with others, mark flexibility as split_possible or negotiable.
- For child, elder, or low_intensity members, never output these exact risky phrases anywhere: long walking, high intensity, hiking, climbing, theme park marathon. Use 'short-walk limit' or 'tiring plan' instead.
- For strict_budget members, never output luxury, high-end, expensive, or high price. Use 'comfortable' or 'medium price' instead.
- Budget scope (PER PERSON): this is your INDIVIDUAL proposal, so size all costs against your personal budget share = the total trip budget in the query divided by the number of family members in family_structure. Keep your proposal within roughly one person's share; do NOT plan against the whole-family total (the family-wide budget is reconciled later when the full itinerary is assembled).

Return this compact JSON only. Do not add extra keys:
{{
  "short_self_summary": "<=20 words",
  "proposal_priority": [{{"rank": 1, "need": "...", "type": "must_include/must_avoid/preference/comfort/budget", "flexibility": "non_negotiable/split_possible/negotiable"}}],
  "preferred_attractions": [{{"name_or_type": "...", "reason": "<=6 words", "priority": "high/medium/low"}}],
  "rejected_arrangements": [{{"arrangement": "...", "reason": "<=6 words"}}],
  "daily_intensity_bottom_line": {{"max_walking_distance_km_per_day": 0, "max_poi_count_per_day": 0, "latest_return_time": "HH:MM", "preferred_pace": "compact/moderate/relaxed", "required_rest": true}},
  "dining_preferences": {{"preferred_food_types": [], "must_avoid_food": [], "meal_price_preference": "low/medium/high"}},
  "hotel_preferences": {{"preferred_area": "...", "price_level": "low/medium/high", "comfort_requirement": "...", "accessibility_requirement": "..."}},
  "transportation_preferences": {{"preferred_mode": "...", "route_strategy": "..."}},
  "negotiable_parts": [],
  "non_negotiable_parts": [],
  "initial_acceptance_criteria": [],
  "objection_triggers": []
}}
""".strip()
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def build_online_grounding_prompt(
    plan_text: str, requirements_text: str, search_results_text: str
) -> str:
    return f"""
You are grounding a family travel itinerary using online search snippets.
Do not invent information that is not supported by the search snippets.
Use Chinese for natural-language fields.
Return valid JSON only.
Output token limit: stay under 900 output tokens.
Use compact arrays and short Chinese strings.

Important v55 restaurant rules:
- Restaurants are grounded meal-by-meal. Prefer concrete restaurant names supported by snippets for each day lunch/dinner.
- Put a value in restaurant_name ONLY when it is an exact supported restaurant/brand name. Never put generic labels such as 当地美食餐厅、中档餐厅、家庭餐厅、休闲餐厅、附近中等价位餐厅 in restaurant_name.
- If no exact restaurant name is supported, leave restaurant_name empty and use name_or_type as an area + cuisine + price-range fallback. This is a soft grounding issue, not a hard failure.
- Restaurants should be diverse: do not repeat the same restaurant name across multiple meals unless no supported alternative exists. If an exact restaurant would repeat, use an area + price range + restaurant type description, e.g. 尖沙咀附近中等价位茶餐厅或粤菜家庭餐厅（人均约50-180元/人）, not a vague 'other restaurant'.
- For every day, provide lunch and dinner if possible.
- Restaurant price_per_person must be realistic for the stated price level; do not use hotel prices or total-trip prices as meal prices.
- Ticket_plan must include paid attractions/scenic areas from the itinerary, with per-person ticket prices and participant counts. Do NOT use a generic 40 CNY fallback. If the ticket price is not supported by search snippets, omit the item or set ticket_price_per_person=0 and add grounding_issues. Include 0 for clearly free public places.
- Cost_breakdown total_cost must include hotel, intercity transport, local transport, meals, and attraction_ticket_cost.
- Split activities should NOT be proposed here unless the itinerary has a direct hard-constraint contradiction where one role must include an activity and another role must avoid it.

Current plan draft:
{plan_text}

Grounding requirements:
{requirements_text}

Online search results:
{search_results_text}

Return JSON with this structure:
{{
  "restaurants": [
    {{"day": 1, "meal": "lunch", "restaurant_name": "supported name or empty", "name_or_type": "餐厅名称或餐饮类型", "area": "位置", "price_per_person": 0, "estimated_total_cost": 0, "source_urls": []}}
  ],
  "ticket_plan": [
    {{"poi_name": "景点", "ticket_price_per_person": 0, "participant_count": 0, "estimated_total_cost": 0, "source_urls": []}}
  ],
  "cost_breakdown": {{
    "currency": "CNY",
    "meal_cost": 0,
    "attraction_ticket_cost": 0,
    "other_cost": 0,
    "total_cost": 0,
    "cost_note": "费用来源和估算说明"
  }},
  "source_urls": [],
  "grounding_issues": []
}}
""".strip()


def build_hotel_grounding_prompt(
    plan_text: str, hotel_requirements_text: str, search_results_text: str
) -> str:
    return f"""
You are grounding the hotel part of a family travel itinerary using online search snippets.
Do not invent hotel names, prices, metro stations, or comfort features that are not supported by the search snippets.
Use Chinese for natural-language fields.
Return valid JSON only.
Output token limit: stay under 750 output tokens.
Use compact arrays and short Chinese strings.

Hotel selection rule v50:
- Choose the nearest qualified hotel, not just any cheap or famous hotel.
- "Qualified" means it satisfies the itinerary requirements: destination city, budget/price level, family comfort, safety, accessibility, and convenient public transport.
- "Nearest" means closest to the majority of the finalized daily POIs, or closest to the best hub for the daily route chain when exact distance is unavailable.
- Prefer hotels within walking distance of a metro/MTR/subway station and close to the first/most frequent sightseeing area.
- If snippets do not give exact distance, infer only from supported area/station evidence and explain the uncertainty in verification_note.
- Do not select a distant hotel only because it is cheaper unless the budget makes closer hotels infeasible.

Current full itinerary draft:
{plan_text}

Hotel requirements generated by the itinerary planner:
{hotel_requirements_text}

Online hotel search results:
{search_results_text}

Return JSON exactly in this structure:
{{
  "hotel_plan": {{
    "hotel_name": "真实酒店名称",
    "hotel_area": "酒店所在区域",
    "address_or_landmark": "地址或附近地标",
    "nearby_metro_station": "附近地铁站",
    "metro_lines_near_hotel": [],
    "nearest_itinerary_anchor": "最靠近的主景点或行程区域",
    "estimated_distance_to_anchor": "如片段支持则填写距离或步行/车程时间，否则写未核验",
    "price_per_night": 0,
    "nights": 0,
    "hotel_total_cost": 0,
    "price_level": "low/medium/high",
    "comfort_features": [],
    "family_friendly_features": [],
    "accessibility_features": [],
    "hotel_selection_reason": "说明为什么这是距离主行程最近且符合预算、舒适度和交通要求的酒店",
    "source_urls": [],
    "verification_note": "价格、房态、距离和附近站点需预订前再次核验"
  }},
  "hotel_grounding_issues": []
}}
""".strip()


def build_transport_grounding_prompt(
    plan_text: str, transportation_requirements_text: str, search_results_text: str
) -> str:
    return f"""
You are grounding the transportation part AFTER the full family itinerary has already been generated and hotel/POI/restaurant candidates have been grounded.
Your job is route stitching: connect the finalized hotel, daily attractions, meals, and return leg into a coherent route chain.
Do not change attractions, restaurants, hotel, dates, or member constraints.
Do not invent metro lines, station names, ticket prices, travel times, or intercity transport prices that are not supported by the search snippets.
Use Chinese for natural-language fields.
Return valid JSON only.
Output token limit: stay under 900 output tokens.
Use compact arrays and short Chinese strings.

Post-generation transport rules v50:
- Build routes only after reading the finalized full itinerary below.
- Each day should connect: hotel/nearby station -> morning POI -> lunch area/restaurant -> afternoon POI -> dinner area/restaurant -> hotel.
- If a route chain is too long, summarize the route but keep exact origin, main stops, destination, mode, time, and cost.
- Prefer metro/public transport for predictable city routes; use short taxi only when it reduces walking or transfers for children/elders.
- The origin must use the selected hotel name, hotel area, or nearby metro station when available.
- Daily local transport prices must be realistic for a family using metro/bus/short taxi; do not output absurdly high extracted numbers.
- Prefer Tavily/search-supported fare evidence. If exact route price is not in snippets, use a conservative clearly marked estimate and set cost_source/comfort_note accordingly.

Finalized itinerary and grounded hotel/restaurant context:
{plan_text}

Transportation stitching requirements:
{transportation_requirements_text}

Online transportation search/API results:
{search_results_text}

Return JSON exactly in this structure:
{{
  "transportation_plan": {{
    "main_transport_mode": "主要交通方式",
    "hotel_base_station": "酒店附近出发站点",
    "metro_lines_near_hotel": [],
    "intercity_transport": [
      {{"from": "出发城市或站点", "to": "目的城市或站点", "mode": "train/flight/bus", "estimated_time": "时间", "estimated_price_per_person": 0, "source_urls": []}}
    ],
    "daily_local_transport": [
      {{"day": 1, "origin": "酒店或酒店附近站点", "stops": [], "destination": "当天最终返回点", "route": "酒店->上午景点->午餐->下午景点->晚餐->酒店的具体串联路线", "estimated_travel_time_minutes": 0, "estimated_cost_total": 0, "comfort_note": "少换乘、少步行或适合老人儿童的说明", "source_urls": []}}
    ],
    "metro_routes": [],
    "local_transport_total_cost": 0,
    "intercity_transport_total_cost": 0,
    "transportation_realism_note": "说明路线如何把全体行程串联起来，并如何满足出行时间、舒适度和预算要求",
    "source_urls": []
  }},
  "transport_grounding_issues": []
}}
""".strip()
