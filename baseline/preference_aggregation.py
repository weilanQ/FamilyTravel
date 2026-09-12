"""Baseline 3 — Preference Aggregation.

Step 1: merge ALL members' preferences/constraints into ONE unified "family
profile" (a single aggregated set of interests, dietary needs, pace, budget, hard
constraints).  Step 2: generate the itinerary from that aggregated profile only —
the individual members are no longer visible to the generator.

Hypothesis it probes: is explicit per-member negotiation better than simply
averaging/merging everyone's preferences up front?
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from baseline_common import OUTPUT_SCHEMA_HINT, family_header, format_member, llm_json, run_baseline

from src.common import safe_text

METHOD = "preference_aggregation"


def _aggregate_profile(hdr: Dict[str, Any], members: str) -> Dict[str, Any]:
    user = (
        f"将以下家庭所有成员的偏好与约束**合并成一个统一的家庭画像**（不要分成员列出，"
        f"取并集/折中，冲突项给出折中倾向）。\n成员：\n{members}\n\n"
        '只输出 JSON：{"family_interests": [...], "dietary": [...], "pace": "...", '
        '"budget_sensitivity": "...", "must_include": [...], "must_avoid": [...], '
        '"max_walking_km_per_day": 0, "latest_return_time": "HH:MM"}'
    )
    data, _, _ = llm_json(
        [
            {"role": "system", "content": "你负责把多名家庭成员的偏好聚合成统一家庭画像。"},
            {"role": "user", "content": user},
        ],
        max_tokens=1200,
    )
    return data if isinstance(data, dict) else {}


def generate_for_family(family_id: int, family_agents: List[Dict[str, Any]]):
    hdr = family_header(family_agents)
    members = "\n".join(format_member(a) for a in family_agents)
    profile = _aggregate_profile(hdr, members)
    user = (
        f"根据下面**统一家庭画像**（已合并全家偏好），规划 {hdr['day']} 天 "
        f"{hdr['departure_city']} 到 {hdr['destination_city']} 的行程。\n"
        f"预算上限：{hdr['budget']} 元。\n"
        f"家庭画像：{json.dumps(profile, ensure_ascii=False)}\n\n"
        f"{OUTPUT_SCHEMA_HINT}"
    )
    raw, usage, rt = llm_json(
        [
            {"role": "system", "content": "你根据统一家庭画像直接生成 JSON 行程。"},
            {"role": "user", "content": user},
        ]
    )
    raw["aggregated_family_profile_json"] = safe_text(json.dumps(profile, ensure_ascii=False))
    return raw, usage, rt


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--first-n", type=int, default=0)
    ap.add_argument("--families", type=str, default="")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()
    ids = [int(x) for x in a.families.split(",") if x.strip().isdigit()] or None
    run_baseline(
        METHOD,
        generate_for_family,
        first_n=a.first_n,
        only_ids=ids,
        max_workers=a.max_workers,
        evaluate=not a.no_eval,
    )
