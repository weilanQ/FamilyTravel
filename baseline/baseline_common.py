"""Shared harness for the comparison baselines.

Every baseline differs ONLY in how the itinerary is generated.  Everything
downstream — plan validation, real-world grounding, cost computation, dedup,
walking, CSV flattening, and Module 6 evaluation — is the SAME code the QG-BMP
pipeline uses, so a difference in the metrics reflects the generation method, not
the post-processing.  Each baseline implements ``generate_for_family`` and calls
``run_baseline``.
"""
from __future__ import annotations

import copy
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.common import safe_float, safe_int, safe_text  # noqa: E402
from src.config import PERSONA_AGENTS_JSON, PERSONA_CSV, RESULTS_DIR  # noqa: E402
from src.llm_client import call_generation_json  # noqa: E402
from src.module5_consensus_itinerary_generation import (  # noqa: E402
    MAX_TOKENS_FINAL_ITINERARY,
    MODEL_NAME,
    TEMPERATURE,
    attach_itinerary_quality_markers,
    build_family_agents,
    enforce_rule_consistency,
    flatten_plan_for_csv,
    normalize_per_day_itinerary,
    save_consensus_results,
    trim_final_plan_text,
    validate_and_fix_plan,
)
from src.module6_evaluation_and_explanation import (  # noqa: E402
    build_module6_feedback_prompt,
    evaluate_and_explain,
    get_failed_reasons,
    get_plan_eval_row_by_family,
)

BASELINE_ROOT = RESULTS_DIR / "baseline"
VERIFIER_TEMP_DIR = BASELINE_ROOT / "_verifier_temp"


# --------------------------------------------------------------------------- #
# Family loading + prompt building
# --------------------------------------------------------------------------- #
def load_families(first_n: int = 0, only_ids: List[int] = None) -> Dict[int, List[Dict[str, Any]]]:
    data = json.loads(Path(PERSONA_AGENTS_JSON).read_text(encoding="utf-8"))
    agents = data if isinstance(data, list) else data.get("agents", [])
    fam_map = build_family_agents(agents)
    ids = sorted(fam_map.keys())
    if only_ids:
        ids = [i for i in ids if i in set(only_ids)]
    if first_n and first_n > 0:
        ids = ids[:first_n]
    return {i: fam_map[i] for i in ids}


def parse_budget_from_query(query: str) -> float:
    """Pull the total family budget out of the query text (e.g. 'budget of ¥10200')."""
    q = safe_text(query)
    m = re.search(r"(?:budget[^\d]{0,12}|预算[^\d]{0,6}|¥|￥)\s*([\d,]+(?:\.\d+)?)", q, re.IGNORECASE)
    if m:
        return safe_float(m.group(1).replace(",", ""), 0.0)
    m = re.search(r"([\d,]{3,})\s*(?:元|cny|rmb|yuan)", q, re.IGNORECASE)
    return safe_float(m.group(1).replace(",", ""), 0.0) if m else 0.0


def format_member(agent: Dict[str, Any]) -> str:
    p = agent.get("profile", {}) if isinstance(agent.get("profile"), dict) else {}
    c = agent.get("constraints", {}) if isinstance(agent.get("constraints"), dict) else {}
    return (
        f"- {safe_text(p.get('role'))}（{safe_text(p.get('age_stage'))}）: "
        f"兴趣={p.get('interest_preferences')}; 体力={safe_text(p.get('physical_constraint'))}; "
        f"预算敏感={safe_text(p.get('budget_sensitivity'))}; 饮食={p.get('dietary_preference')}; "
        f"节奏={safe_text(p.get('travel_pace'))}; 必去={c.get('must_include')}; 必避={c.get('must_avoid')}; "
        f"每日步行≤{c.get('max_walking_distance_km')}km; 最晚返回={safe_text(c.get('latest_return_time'))}"
    )


def family_header(family_agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    a0 = family_agents[0] if family_agents else {}
    return {
        "departure_city": safe_text(a0.get("departure_city")),
        "destination_city": safe_text(a0.get("destination_city")),
        "day": safe_int(a0.get("day"), 1),
        "query": safe_text(a0.get("query")),
        "budget": parse_budget_from_query(a0.get("query")),
        "member_count": len(family_agents),
    }


# Output JSON contract (must match what validate_and_fix_plan expects). Trimmed
# from the Module 5 schema; code fills the rich/default fields afterwards.
OUTPUT_SCHEMA_HINT = (
    "严格只输出一个 JSON 对象，结构如下（day 数量必须等于给定天数）：\n"
    "{\n"
    '  "destination_city": "...", "day": 3, "budget_limit": 0,\n'
    '  "itinerary_text": "≤180字中文概述",\n'
    '  "per_day_itinerary": [\n'
    '    {"day": 1, "morning": ["具体景点名"], "afternoon": ["具体景点名"],\n'
    '     "meal": "餐饮类型+价格等级", "transportation": "酒店区域到景点的短路线",\n'
    '     "estimated_return_time": "20:00", "split_activities": []}\n'
    "  ],\n"
    '  "restaurant_price_level": "medium",\n'
    '  "hotel_plan": {"hotel_area": "住宿区域", "price_level": "medium", "requirement": "靠近交通，适合家庭"},\n'
    '  "transportation_plan": {"main_transport_mode": "地铁+短途出租车", "route_strategy": "少换乘少步行"}\n'
    "}\n"
    "要求：每天上午/下午都是该城市真实具体的景点名；不要输出 JSON 以外的任何文字。"
)


# The shared hint's sample values ("酒店区域到景点的短路线" / "餐饮类型+价格等级") are placeholders
# that models tend to echo verbatim, which then scores as ungrounded transport/unspecific meals.
# The resolving baselines use this variant whose sample values instruct concreteness instead.
CONCRETE_SCHEMA_HINT = OUTPUT_SCHEMA_HINT.replace(
    '"transportation": "酒店区域到景点的短路线"',
    '"transportation": "具体公交/地铁线路+起讫站名+时长（不要写`酒店到景点的短路线`这类占位文字）"',
).replace(
    '"meal": "餐饮类型+价格等级"',
    '"meal": "具体餐厅名或街区+菜系+价位"',
)


# --------------------------------------------------------------------------- #
# LLM call + shared finalization (identical to QG-BMP post-processing)
# --------------------------------------------------------------------------- #
def llm_json(
    messages: List[Dict[str, str]],
    max_tokens: int = MAX_TOKENS_FINAL_ITINERARY,
    temperature: float = TEMPERATURE,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    t0 = time.time()
    res = call_generation_json(
        messages=messages, model=MODEL_NAME, max_tokens=max_tokens, temperature=temperature
    )
    data = res.get("data") if isinstance(res, dict) else None
    if not isinstance(data, dict):
        data = {"itinerary_text": safe_text(data)}
    return data, (res.get("usage", {}) if isinstance(res, dict) else {}), round(time.time() - t0, 2)


def _avg_daily_poi_count(plan: Dict[str, Any]) -> float:
    """Average number of core POIs per day from the grounded per_day itinerary (morning/afternoon/
    evening). Used to give baselines a real daily_poi_count instead of the 0 that defaults because
    the baseline generators, unlike QG-BMP's M5, don't self-report it."""
    per_day = plan.get("per_day_itinerary", [])
    if not isinstance(per_day, list) or not per_day:
        return 0.0
    counts = []
    for d in per_day:
        if not isinstance(d, dict):
            continue
        c = 0
        for blk in ("morning", "afternoon", "evening"):
            for it in d.get(blk) or []:
                name = (it.get("poi_name") or it.get("activity")) if isinstance(it, dict) else it
                if safe_text(name):
                    c += 1
        counts.append(c)
    return round(sum(counts) / len(counts), 2) if counts else 0.0


def merge_usages(*usages: Dict[str, Any]) -> Dict[str, Any]:
    """Sum token usage across every LLM call a baseline makes, so multi-call baselines report
    their REAL cumulative cost (the same accounting standard as the full pipeline's M2+M4+M5+M7)."""
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for u in usages:
        if not isinstance(u, dict):
            continue
        for k in total:
            total[k] += safe_int(u.get(k), 0)
    if not total["total_tokens"]:
        total["total_tokens"] = total["prompt_tokens"] + total["completion_tokens"]
    return total


def finalize_plan(
    raw_plan: Dict[str, Any],
    family_id: int,
    family_agents: List[Dict[str, Any]],
    usage: Dict[str, Any],
    runtime: float,
    method: str,
    credit_llm_splits: bool = False,
) -> Dict[str, Any]:
    """Run the EXACT QG-BMP post-processing so baselines are evaluated identically."""
    hdr = family_header(family_agents)
    raw_plan.setdefault("destination_city", hdr["destination_city"])
    raw_plan.setdefault("departure_city", hdr["departure_city"])
    raw_plan.setdefault("day", hdr["day"])
    # Inject the real budget from the query so budget evaluation is accurate.
    if safe_float(raw_plan.get("budget_limit"), 0) <= 0 and hdr["budget"] > 0:
        raw_plan["budget_limit"] = hdr["budget"]
    plan = validate_and_fix_plan(
        plan=raw_plan,
        family_id=family_id,
        family_agents=family_agents,
        usage=usage,
        runtime_seconds=runtime,
    )
    # negotiation_resolved=False: a baseline has NO Module-4 machinery, so do NOT let the shared
    # programmatic split-resolution (ensure_split_activities_for_hard_conflicts) auto-fix its
    # must_include×must_avoid clashes. Grounding/cost/format post-processing still runs identically.
    # credit_llm_splits (conflict-RESOLVING baselines only, e.g. SA-Resolve / Generic-MA Debate):
    # still nothing is programmatically added, but a structurally complete split the baseline's own
    # LLM authored IS credited — same completeness bar as the full pipeline. Non-resolving baselines
    # keep False, so their unresolved clashes surface as real degradation.
    plan = enforce_rule_consistency(
        plan=plan, family_agents=family_agents, negotiation_resolved=False,
        credit_llm_splits=credit_llm_splits,
    )
    plan = normalize_per_day_itinerary(plan)
    # Fairness fix: QG-BMP's Module 5 self-reports `daily_poi_count` in its output schema; the baseline
    # generators do NOT, so it defaulted to 0 and the rationality check `1 <= daily_poi_count <= 4`
    # failed for EVERY baseline family — sinking rationality_pass_rate to 6/7 even on no-conflict (Low)
    # families that have nothing to resolve. That is a harness artifact, not a real quality gap, and it
    # unfairly handicaps the baselines. Compute the count from the actual grounded itinerary and cap it
    # at 3 exactly like the full pipeline (module5 line ~2009), so the check measures the same thing.
    plan["daily_poi_count"] = min(_avg_daily_poi_count(plan), 3.0)
    plan = attach_itinerary_quality_markers(plan)
    plan = trim_final_plan_text(plan)
    plan["method"] = method
    plan["plan_id"] = method
    plan["status"] = "success"
    plan.setdefault("acceptance_vote_ratio", 1.0)
    plan.setdefault("consensus_status", "Full Consensus")
    return plan


# --------------------------------------------------------------------------- #
# Shared verifier loop for conflict-RESOLVING baselines (SA-Resolve, Generic-MA)
# --------------------------------------------------------------------------- #
def verifier_evaluate(
    plan: Dict[str, Any], family_id: int, method: str, round_index: int
) -> Tuple[Dict[str, Any], List[str], str]:
    """Run the SAME rule verifier the full pipeline's repair loop uses (Module 6 gates, no LLM
    judge — mirrors module7.evaluate_candidate_row + build_module6_feedback_prompt +
    compact_feedback_for_module5). Returns (plan_eval_row, failed_reasons, compact_feedback)."""
    import pandas as pd

    from src.module7_feedback_revision_loop import compact_feedback_for_module5

    VERIFIER_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{method}_f{family_id}_r{round_index}"
    flat_row = flatten_plan_for_csv(plan)
    plan_csv = VERIFIER_TEMP_DIR / f"plan_{tag}.csv"
    pd.DataFrame([flat_row]).to_csv(plan_csv, index=False, encoding="utf-8-sig")
    res = evaluate_and_explain(
        persona_csv_path=PERSONA_CSV,
        agents_json_path=PERSONA_AGENTS_JSON,
        generated_plans_csv_path=plan_csv,
        member_output_csv_path=VERIFIER_TEMP_DIR / f"member_{tag}.csv",
        plan_output_csv_path=VERIFIER_TEMP_DIR / f"planeval_{tag}.csv",
        report_output_json_path=VERIFIER_TEMP_DIR / f"report_{tag}.json",
        use_llm_judge=False,
    )
    plan_eval_df = res["plan_eval_df"]
    member_eval_df = res["member_eval_df"]
    eval_row = get_plan_eval_row_by_family(plan_eval_df=plan_eval_df, family_id=family_id)
    reasons = get_failed_reasons(eval_row)
    feedback = build_module6_feedback_prompt(
        family_id=family_id,
        plan_eval_df=plan_eval_df,
        member_eval_df=member_eval_df,
        original_plan=flat_row,
        failure_reasons=reasons,
        revision_round=round_index,
        candidate_index=0,
    )
    return eval_row, reasons, compact_feedback_for_module5(feedback, reasons)


def refine_with_verifier(
    raw_plan: Dict[str, Any],
    family_id: int,
    family_agents: List[Dict[str, Any]],
    method: str,
    revise_fn: Callable[[Dict[str, Any], str], Tuple[Dict[str, Any], Dict[str, Any]]],
    max_rounds: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Iterative verifier-feedback refinement: finalize a candidate copy (identical
    post-processing incl. grounding), run the shared rule verifier, stop when every gate passes,
    otherwise hand the compacted feedback to ``revise_fn(prev_raw, feedback) -> (new_raw, usage)``
    for the next candidate. Keeps the raw plan with the fewest failed gates. Returns
    (best_raw_plan, revision_usages, loop_info)."""
    usages: List[Dict[str, Any]] = []
    current_raw = raw_plan
    best_raw, best_key, best_failed = raw_plan, None, None
    rounds_used = 0
    for r in range(max_rounds + 1):  # r=0 evaluates the initial plan
        candidate = finalize_plan(
            copy.deepcopy(current_raw), family_id, family_agents, {}, 0.0, method,
            credit_llm_splits=True,
        )
        try:
            eval_row, reasons, feedback = verifier_evaluate(candidate, family_id, method, r)
        except Exception as exc:
            print(f"[baseline:{method}] family_{family_id} verifier round {r} 失败: {exc}")
            break
        failed = len(reasons or [])
        # Rank candidates by fewest failed gates, then higher minimum member satisfaction,
        # then higher average — so a revision that fixes nothing but degrades content never wins.
        key = (
            failed,
            -safe_float(eval_row.get("minimum_member_satisfaction"), 0.0),
            -safe_float(eval_row.get("average_group_satisfaction"), 0.0),
        )
        if best_key is None or key < best_key:
            best_raw, best_key, best_failed = current_raw, key, failed
        if failed == 0 or r >= max_rounds:
            break
        try:
            new_raw, usage = revise_fn(current_raw, feedback)
        except Exception as exc:
            print(f"[baseline:{method}] family_{family_id} revision round {r+1} 失败: {exc}")
            break
        usages.append(usage or {})
        rounds_used = r + 1
        if isinstance(new_raw, dict) and new_raw.get("per_day_itinerary"):
            current_raw = new_raw
    info = {"verifier_rounds_used": rounds_used, "verifier_best_failed_gates": best_failed}
    return best_raw, usages, info


# --------------------------------------------------------------------------- #
# Runner: generate -> save -> evaluate with Module 6
# --------------------------------------------------------------------------- #
def run_baseline(
    method: str,
    generate_for_family: Callable[
        [int, List[Dict[str, Any]]], Tuple[Dict[str, Any], Dict[str, Any], float]
    ],
    first_n: int = 0,
    only_ids: List[int] = None,
    max_workers: int = 4,
    evaluate: bool = True,
    use_llm_judge: bool = True,
    output_root: Path = None,
    credit_llm_splits: bool = False,
) -> Path:
    families = load_families(first_n=first_n, only_ids=only_ids)
    out_dir = (output_root or BASELINE_ROOT) / method
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "generated_plans.csv"
    json_path = out_dir / "generated_plans.json"
    print(f"[baseline:{method}] 家庭数={len(families)} 并发={max_workers} -> {out_dir}")

    plans: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    def _one(fid: int) -> Dict[str, Any]:
        try:
            raw, usage, rt = generate_for_family(fid, families[fid])
            return finalize_plan(
                raw, fid, families[fid], usage, rt, method, credit_llm_splits=credit_llm_splits
            )
        except Exception as exc:
            print(f"[baseline:{method}] family_{fid} 失败: {exc}")
            from src.module5_consensus_itinerary_generation import build_error_plan

            return build_error_plan(fid, exc)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_one, fid): fid for fid in families}
        done = 0
        for fut in as_completed(futures):
            plan = fut.result()
            plans.append(plan)
            rows.append(flatten_plan_for_csv(plan))
            done += 1
            print(f"[baseline:{method}] ({done}/{len(families)}) family_{plan.get('family_id')} ok")

    save_consensus_results(plans, rows, csv_path, json_path)
    print(f"[baseline:{method}] 生成完成，用时 {round(time.time()-t0,1)}s -> {csv_path}")

    if evaluate:
        print(f"[baseline:{method}] 运行 Module 6 评估 ...")
        evaluate_and_explain(
            persona_csv_path=PERSONA_CSV,
            agents_json_path=PERSONA_AGENTS_JSON,
            generated_plans_csv_path=csv_path,
            member_output_csv_path=out_dir / "evaluation_member_level.csv",
            plan_output_csv_path=out_dir / "evaluation_plan_level.csv",
            report_output_json_path=out_dir / "evaluation_report.json",
            use_llm_judge=use_llm_judge,
        )
        print(f"[baseline:{method}] 评估完成 -> {out_dir / 'evaluation_report.json'}")
    return out_dir
