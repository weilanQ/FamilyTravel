"""Baseline 6 — Generic Multi-Agent Debate: negotiation WITHOUT the typed conflict graph.

Every family member is a persona agent that speaks in ``DEBATE_ROUNDS`` (default 4,
matching the full system's R=4 negotiation) open discussion rounds; a coordinator
summarizes each round into an updated draft and finally synthesizes the itinerary.
Crucially the agents get NO conflict detector output, NO SHARED/SPLIT/UNSAT typing,
and NO typed conflict graph — conflicts must emerge and be coordinated in free-form
debate, exactly the generic multi-agent-debate recipe (AutoGen/CAMEL style).

After the debate the SAME rule verifier the full pipeline's repair loop uses runs
``DEBATE_REPAIR_ROUNDS`` (default 2, mirroring the full system's M7 repair budget)
feedback/revision rounds on the coordinator, so quality control and token budget are
comparable to the full system.

Question it answers: is the explicit conflict graph + satisfiability dispatch better
than generic multi-agent discussion with the same personas, rounds, grounding, QC
and budget?

Self-authored split activities are credited when structurally complete
(``credit_llm_splits=True``) — same bar as the full pipeline; nothing is resolved
programmatically for it.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from baseline_common import (
    CONCRETE_SCHEMA_HINT,
    family_header,
    format_member,
    llm_json,
    merge_usages,
    refine_with_verifier,
    run_baseline,
)

from src.common import safe_int, safe_text

METHOD = "generic_ma_debate"
DEBATE_ROUNDS = safe_int(os.environ.get("DEBATE_ROUNDS", ""), 4)
REPAIR_ROUNDS = safe_int(os.environ.get("DEBATE_REPAIR_ROUNDS", ""), 2)
# Full's negotiation runs at temperature 0.15; keep the debate at the same setting.
DEBATE_TEMPERATURE = 0.15

SPLIT_HINT = (
    "如成员们在讨论中同意分组行动，最终 JSON 需包含顶层 split_activities，每条含 "
    '{"day", "conflict_type": "must_include_vs_must_avoid_split_activity", '
    '"activity_or_term", "participating_roles", "non_participating_roles", '
    '"alternative_for_non_participants": "具体真实地点名", "rejoin_point", "reason"}；'
    "activity_or_term 使用成员画像里必去/必避条目的**英文原文**（如 \"crowded theme park\"），"
    "否则该分组不被评估器承认；未能达成一致的分歧如实写入顶层 unresolved_conflicts。"
)


def _member_statement(
    hdr: Dict[str, Any], agent: Dict[str, Any], draft: str, last_round: List[Dict[str, str]]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    others = "\n".join(
        f"- {s['role']}: {safe_text(s['statement'])[:160]}" for s in last_round
    ) or "（第一轮，还没有其他成员发言）"
    user = (
        f"家庭正在讨论 {hdr['day']} 天 {hdr['destination_city']} 的行程（总预算 {hdr['budget']} 元）。\n"
        f"你的身份与需求：\n{format_member(agent)}\n\n"
        f"当前草案：\n{safe_text(draft)[:1200] or '（尚无草案）'}\n\n"
        f"其他成员上一轮发言：\n{others}\n\n"
        "请以这位成员的身份发言：哪些安排你接受、哪些你反对（说明理由）、你提出什么修改或分组建议。\n"
        '只输出 JSON：{"stance": "accept|object|conditional", "objections": ["≤40字/条"], '
        '"requests": ["≤40字/条"], "split_suggestion": "如建议分组则≤60字，否则空"}'
    )
    data, usage, _ = llm_json(
        [
            {"role": "system", "content": "你是家庭中的一位成员，只代表自己的利益发言，但要参与集体协商。"},
            {"role": "user", "content": user},
        ],
        max_tokens=650,  # 500 occasionally truncated verbose members' JSON mid-string
        temperature=DEBATE_TEMPERATURE,
    )
    return (data if isinstance(data, dict) else {}), usage


def _coordinator_update(
    hdr: Dict[str, Any], draft: str, statements: List[Dict[str, str]], round_no: int
) -> Tuple[str, Dict[str, Any]]:
    stmts = "\n".join(f"- {s['role']}: {json.dumps(s['data'], ensure_ascii=False)[:400]}" for s in statements)
    user = (
        f"你是协调者。第 {round_no} 轮讨论结束，请更新 {hdr['day']} 天 {hdr['destination_city']} 行程草案。\n"
        f"当前草案：\n{safe_text(draft)[:1500] or '（尚无草案）'}\n\n"
        f"本轮各成员发言：\n{stmts}\n\n"
        '只输出 JSON：{"draft": "更新后的草案：每天一行列出上午/下午/晚上活动与用餐，≤350字", '
        '"open_issues": ["尚存分歧点，≤30字/条"]}'
    )
    data, usage, _ = llm_json(
        [
            {"role": "system", "content": "你是家庭讨论的协调者：汇总意见、更新草案、记录分歧。不做任何冲突类型分类。"},
            {"role": "user", "content": user},
        ],
        max_tokens=850,  # 700 occasionally truncated long multi-day drafts
        temperature=DEBATE_TEMPERATURE,
    )
    if not isinstance(data, dict):
        data = {}
    text = safe_text(data.get("draft")) or safe_text(data.get("itinerary_text"))
    issues = [safe_text(x) for x in (data.get("open_issues") or []) if safe_text(x)]
    if issues:
        text += "\n尚存分歧: " + "；".join(issues)
    return (text or safe_text(json.dumps(data, ensure_ascii=False))), usage


def generate_for_family(family_id: int, family_agents: List[Dict[str, Any]]):
    hdr = family_header(family_agents)
    roles = [safe_text((a.get("profile") or {}).get("role")) or f"member_{i}" for i, a in enumerate(family_agents)]

    usages: List[Dict[str, Any]] = []
    total_rt = 0.0
    draft = ""
    last_round: List[Dict[str, str]] = []
    import time as _time

    t0 = _time.time()
    for rnd in range(1, DEBATE_ROUNDS + 1):
        statements = []
        for role, agent in zip(roles, family_agents):
            data, u = _member_statement(hdr, agent, draft, last_round)
            usages.append(u)
            statements.append(
                {"role": role, "data": data, "statement": json.dumps(data, ensure_ascii=False)}
            )
        draft, u = _coordinator_update(hdr, draft, statements, rnd)
        usages.append(u)
        last_round = statements

    # Final synthesis: coordinator turns the debated draft into the strict plan JSON.
    members = "\n".join(format_member(a) for a in family_agents)
    final_user = (
        f"讨论结束。请把最终草案落成一份 {hdr['day']} 天 {hdr['departure_city']} 到 "
        f"{hdr['destination_city']} 的行程（预算 {hdr['budget']} 元）。\n"
        f"家庭成员：\n{members}\n\n"
        f"经 {DEBATE_ROUNDS} 轮讨论后的草案（含分歧记录）：\n{draft}\n\n"
        f"{SPLIT_HINT}\n{CONCRETE_SCHEMA_HINT}"
    )
    raw, u, rt = llm_json(
        [
            {"role": "system", "content": "你是协调者：把讨论结果落成 JSON 行程，忠实反映已达成的一致与分组安排。"},
            {"role": "user", "content": final_user},
        ]
    )
    usages.append(u)
    total_rt = round(_time.time() - t0, 2)

    def revise(prev_raw: Dict[str, Any], feedback: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        user = (
            f"你产出的 {hdr['day']} 天 {hdr['destination_city']} 家庭行程未通过验证器。\n"
            f"家庭成员：\n{members}\n\n"
            f"讨论草案（保持已达成的一致）：\n{safe_text(draft)[:1200]}\n\n"
            f"验证器反馈（逐条修复）：\n{feedback}\n\n"
            f"上一版行程 JSON：\n{json.dumps(prev_raw, ensure_ascii=False)[:12000]}\n\n"
            f"{SPLIT_HINT}\n{CONCRETE_SCHEMA_HINT}"
        )
        new_raw, uu, _ = llm_json(
            [
                {"role": "system", "content": "你是协调者：按验证器反馈修订 JSON 行程，不推翻成员已达成的一致。"},
                {"role": "user", "content": user},
            ]
        )
        return new_raw, uu

    best_raw, revision_usages, info = refine_with_verifier(
        raw, family_id, family_agents, METHOD, revise, max_rounds=REPAIR_ROUNDS
    )
    best_raw["debate_info_json"] = json.dumps(
        {"debate_rounds": DEBATE_ROUNDS, **info}, ensure_ascii=False
    )
    return best_raw, merge_usages(*usages, *revision_usages), total_rt


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
        credit_llm_splits=True,
    )
