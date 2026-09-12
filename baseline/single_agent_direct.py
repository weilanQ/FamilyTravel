"""Baseline 1 — Single-Agent Direct.

All family members' needs are dropped into one prompt and a single agent directly
writes the itinerary, with no explicit analysis, aggregation, or multi-agent step.

Hypothesis it probes: is the multi-agent system better than the simplest possible
single-agent generation?
"""
from __future__ import annotations

from typing import Any, Dict, List

from baseline_common import OUTPUT_SCHEMA_HINT, family_header, format_member, llm_json, run_baseline

METHOD = "single_agent_direct"


def generate_for_family(family_id: int, family_agents: List[Dict[str, Any]]):
    hdr = family_header(family_agents)
    members = "\n".join(format_member(a) for a in family_agents)
    user = (
        f"为一个家庭规划 {hdr['day']} 天 {hdr['departure_city']} 到 {hdr['destination_city']} 的行程。\n"
        f"预算上限：{hdr['budget']} 元（全家总预算）。\n"
        f"家庭成员需求：\n{members}\n\n"
        f"请直接给出能兼顾全家的完整行程。\n{OUTPUT_SCHEMA_HINT}"
    )
    messages = [
        {"role": "system", "content": "你是一个家庭旅行行程规划助手，直接输出 JSON 行程。"},
        {"role": "user", "content": user},
    ]
    return llm_json(messages)


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
