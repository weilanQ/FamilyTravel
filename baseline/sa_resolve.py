"""Baseline 5 — SA-Resolve: budget-matched SINGLE-AGENT conflict resolution.

Unlike the four generation-only baselines, SA-Resolve IS allowed to resolve conflicts:
one central LLM receives the structured member table plus the EXPLICIT conflict list
(the same rule detector the benchmark uses, run on the member personas) and may apply
merge / substitute / split / relax. Each round the SAME rule verifier the full
pipeline's repair loop uses (Module 6 gates, no LLM judge) scores the candidate and
its compacted feedback drives a revision — up to ``SA_RESOLVE_ROUNDS`` (default 4)
rounds, bringing the token budget close to the full system's ~24k instead of the
~1k single-shot baselines.

Question it answers: is the multi-agent negotiation + typed conflict graph better
than a budget-matched CENTRAL single-agent resolver with the same model, same
grounding, same verifier, and the same number of repair rounds?

Its self-authored split activities are credited by the shared post-processing when
structurally complete (``credit_llm_splits=True``) — the same completeness bar the
full pipeline must clear — but nothing is programmatically resolved for it.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd

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

METHOD = "sa_resolve"
MAX_ROUNDS = safe_int(os.environ.get("SA_RESOLVE_ROUNDS", ""), 4)

# The four operators the central resolver may use. `split` must emit entries that pass
# module5.split_activity_has_complete_resolution (participants + non-participants +
# named alternative + reason) or it earns NO credit — identical bar to the full system.
OPERATOR_GUIDE = (
    "你可以且应当显式化解冲突，允许四类操作（在 resolution_log 中逐条记录）：\n"
    "1) merge —— 选择能同时满足冲突双方的共同活动；\n"
    "2) substitute —— 用满足所有相关约束的替代活动替换冲突活动；\n"
    "3) split —— 并行分组：冲突双方在同一时段去不同活动，之后汇合。必须写入顶层 "
    'split_activities，每条包含 {"day": 天数, '
    '"conflict_type": "must_include_vs_must_avoid_split_activity", '
    '"activity_or_term": "冲突活动（含冲突术语英文原文）", '
    '"participating_roles": ["参加的成员角色"], "non_participating_roles": ["不参加的角色"], '
    '"alternative_for_non_participants": "不参加成员去的**具体真实地点名**", '
    '"rejoin_point": "汇合地点", "reason": "为何这样分组"}；儿童/老人必须有成人陪同。'
    '填写示例：{"day": 2, "conflict_type": "must_include_vs_must_avoid_split_activity", '
    '"activity_or_term": "crowded theme park（长影世纪城）", "participating_roles": ["child", "father"], '
    '"non_participating_roles": ["mother"], "alternative_for_non_participants": "市博物馆", '
    '"rejoin_point": "酒店附近晚餐餐厅", "reason": "母亲必避人多排队场所，孩子必去乐园，父亲陪同"}；\n'
    "4) relax —— 放宽某位成员的**软偏好**并说明补偿（不得放宽 must_avoid、最晚返回时间、"
    "步行上限等硬约束）。\n"
    "【硬性要求】冲突清单非空时，resolution_log 必须为清单中的**每一条**冲突记录一个 operator 决策，"
    "不得输出空的 resolution_log；must_include×must_avoid 类直接冲突必须要么给出一条完整的 "
    "split_activities 记录、要么写入 unresolved_conflicts——不要静默折中。"
    "split_activities 的 activity_or_term 必须**原样包含冲突清单“冲突术语”里的英文原文**"
    "（如 \"crowded theme park\"，可在 reason 中用中文解释）——否则该 split 不被评估器承认。"
    "确实无法化解的冲突，如实写入顶层 unresolved_conflicts（不要假装已解决）。\n"
    "【质量要求】冲突处理不得以牺牲行程具体性为代价：每天 transportation 写出具体路线"
    "（如“地铁2号线 人民广场站→静安寺站，约20分钟”，不要写“酒店→景点”这类占位符）；"
    "meal 给出具体餐厅名或“街区+菜系+价位”；hotel_plan 给出具体住宿区域；"
    "每天上午/下午安排该城市真实、可搜索的景点名。"
)

EXTRA_SCHEMA = (
    "在上面 schema 基础上额外包含顶层字段：\n"
    '"split_activities": [ ...见 split 操作说明... ],\n'
    '"unresolved_conflicts": [ {"conflict_type": "...", "reason": "...", "conflicting_terms": [...]} ],\n'
    '"resolution_log": [ {"conflict": "≤30字", "operator": "merge|substitute|split|relax", "decision": "≤50字"} ]'
)


def detect_persona_conflicts(
    family_id: int, family_agents: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Run the benchmark's OWN rule detector (module3) on the member personas — the explicit,
    member-attributable conflict list handed to the resolver. Proposal-level fields are empty,
    so this is exactly the persona-level subset of the benchmark's conflict typology."""
    from src.module3_conflict_detection import ConflictDetector, standardize_proposal_row

    lookup = {safe_text(a.get("agent_id")): a for a in family_agents}
    std_rows = []
    for a in family_agents:
        row = pd.Series(
            {
                "agent_id": safe_text(a.get("agent_id")),
                "family_id": family_id,
                "member_id": safe_int(a.get("member_id"), 0),
                "individual_proposal_json": "{}",
            }
        )
        std_rows.append(standardize_proposal_row(row=row, agent_lookup=lookup))
    return ConflictDetector(std_rows).run_all_detectors()


def _role_of(agent_id: str, family_agents: List[Dict[str, Any]]) -> str:
    for a in family_agents:
        if safe_text(a.get("agent_id")) == safe_text(agent_id):
            return safe_text((a.get("profile") or {}).get("role")) or safe_text(agent_id)
    return safe_text(agent_id)


def _evidence_terms(evidence: Any) -> List[str]:
    """Pull every exact 'term' string out of a (nested) detector evidence blob — these are the
    literal constraint terms the credit check matches against, so the resolver must see them."""
    terms: List[str] = []
    if isinstance(evidence, dict):
        for k, v in evidence.items():
            if k == "term" and safe_text(v):
                terms.append(safe_text(v))
            else:
                terms.extend(_evidence_terms(v))
    elif isinstance(evidence, list):
        for item in evidence:
            terms.extend(_evidence_terms(item))
    return list(dict.fromkeys(terms))


def format_conflicts(
    conflicts: List[Dict[str, Any]], family_agents: List[Dict[str, Any]], max_items: int = 24
) -> str:
    lines = []
    for c in conflicts[:max_items]:
        roles = "、".join(_role_of(x, family_agents) for x in c.get("involved_agents") or [])
        terms = _evidence_terms(c.get("evidence"))
        lines.append(
            f"- [{safe_text(c.get('conflict_type'))}|{safe_text(c.get('severity'))}] "
            f"涉及: {roles or '全家'} | {safe_text(c.get('description'))[:120]}"
            + (f" | 冲突术语: {'; '.join(terms[:6])}" if terms else "")
            + (
                f" | 建议方向: {safe_text(c.get('suggested_resolution'))[:80]}"
                if safe_text(c.get("suggested_resolution"))
                else ""
            )
        )
    return "\n".join(lines) if lines else "（检测器未发现成员间冲突）"


SYSTEM_PROMPT = (
    "你是中央家庭旅行规划器：独自一人负责发现并化解全体成员的冲突，再产出行程。"
    "先逐条处理冲突清单（merge/substitute/split/relax），再输出 JSON 行程。"
)


def _initial_messages(hdr, members: str, conflict_text: str) -> List[Dict[str, str]]:
    user = (
        f"为一个家庭规划 {hdr['day']} 天 {hdr['departure_city']} 到 {hdr['destination_city']} 的行程。\n"
        f"预算上限：{hdr['budget']} 元（全家总预算）。\n"
        f"家庭成员结构化画像：\n{members}\n\n"
        f"检测到的成员间冲突清单（必须逐条处理）：\n{conflict_text}\n\n"
        f"{OPERATOR_GUIDE}\n\n{EXTRA_SCHEMA}\n{CONCRETE_SCHEMA_HINT}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def generate_for_family(family_id: int, family_agents: List[Dict[str, Any]]):
    hdr = family_header(family_agents)
    members = "\n".join(format_member(a) for a in family_agents)
    conflicts = detect_persona_conflicts(family_id, family_agents)
    conflict_text = format_conflicts(conflicts, family_agents)

    raw, usage0, rt0 = llm_json(_initial_messages(hdr, members, conflict_text))
    total_rt = rt0

    def revise(prev_raw: Dict[str, Any], feedback: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        nonlocal total_rt
        user = (
            f"你上一版 {hdr['day']} 天 {hdr['destination_city']} 家庭行程未通过验证器。\n"
            f"家庭成员结构化画像：\n{members}\n\n"
            f"成员间冲突清单：\n{conflict_text}\n\n"
            f"验证器反馈（逐条修复，不要引入新问题）：\n{feedback}\n\n"
            f"你上一版行程 JSON（在其基础上修订）：\n"
            f"{json.dumps(prev_raw, ensure_ascii=False)[:12000]}\n\n"
            f"{OPERATOR_GUIDE}\n\n{EXTRA_SCHEMA}\n{CONCRETE_SCHEMA_HINT}"
        )
        new_raw, u, rt = llm_json(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        )
        total_rt += rt
        return new_raw, u

    best_raw, revision_usages, info = refine_with_verifier(
        raw, family_id, family_agents, METHOD, revise, max_rounds=MAX_ROUNDS
    )
    best_raw["sa_resolve_info_json"] = json.dumps(info, ensure_ascii=False)
    best_raw["detected_conflicts_count"] = len(conflicts)
    return best_raw, merge_usages(usage0, *revision_usages), total_rt


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
