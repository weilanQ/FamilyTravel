"""Run all comparison baselines and build a side-by-side metric table.

Each baseline writes results/baseline/<method>/ with generated_plans.csv +
evaluation_report.json (same format as QG-BMP).  This script then collects the
headline metrics from each report — plus the QG-BMP result when available — into
results/baseline/baseline_comparison.csv so the multi-agent contribution is
directly readable.

Usage:
    python baseline/run_baselines.py                 # all 4 baselines, all families
    python baseline/run_baselines.py --first-n 10
    python baseline/run_baselines.py --only sa_resolve,generic_ma_debate
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import generic_ma_debate
import preference_aggregation
import sa_resolve
import single_agent_direct  # noqa: E402
from baseline_common import BASELINE_ROOT  # noqa: E402

from src.config import PERSONA_AGENTS_JSON, PERSONA_CSV, RESULTS_DIR  # noqa: E402
from src.module1_persona_agent_initialization import initialize_persona_agents  # noqa: E402

# Lineup (2×2 + full): {single, aggregated} × non-resolving anchors, then the two
# budget-matched RESOLVING baselines. single_agent_cot and independent_multi_agent were
# removed 2026-07: each converged with its sibling (33% vs 34%, min_sat within 0.5) and is
# strictly subsumed by a stronger new baseline (CoT → sa_resolve, indep-MA → generic_ma_debate).
BASELINES = {
    "single_agent_direct": single_agent_direct.generate_for_family,
    "preference_aggregation": preference_aggregation.generate_for_family,
    "sa_resolve": sa_resolve.generate_for_family,
    "generic_ma_debate": generic_ma_debate.generate_for_family,
}
# Conflict-RESOLVING baselines: their self-authored splits are credited by post-processing
# (credit_llm_splits=True); nothing is programmatically resolved for them. The four
# generation-only baselines above stay non-resolving.
RESOLVING_BASELINES = {"sa_resolve", "generic_ma_debate"}

# Reported metrics: (column label, report metric key, is_core).
# Core 必报: Success, Hard, Feasibility, Avg Sat, Min Sat, Fairness
# Supplementary 补充: Grounding, Token, Runtime
METRIC_COLUMNS = [
    ("Success", "pass_rate", True),
    ("Hard", "avg_hard_constraint_pass_rate", True),
    ("Feasibility", "avg_feasibility_pass_rate", True),
    ("AvgSat", "avg_group_satisfaction", True),
    ("MinSat", "avg_minimum_member_satisfaction", True),
    ("Fairness", "avg_max_min_fairness", True),
    ("Grounding", "grounding_score", False),
    ("Token", "avg_total_tokens", False),
    ("Runtime", "avg_runtime_seconds", False),
]


def _load_metrics(report_path: Path) -> dict:
    try:
        return (json.loads(report_path.read_text(encoding="utf-8")) or {}).get("metrics", {})
    except Exception:
        return {}


def _find_qgbmp_report(explicit: str = "") -> Path | None:
    """Locate the QG-BMP full-system report. Checks --qgbmp-dir first, then the default results
    dir, then results/ablation/full/ (where run_all_ablations writes the full run)."""
    roots = []
    if explicit:
        roots.append(Path(explicit))
    roots += [RESULTS_DIR, RESULTS_DIR / "ablation" / "full"]
    names = [
        Path("module_7") / "evaluation_report_feedback.json",
        Path("module_6") / "evaluation_report.json",
    ]
    for root in roots:
        # allow --qgbmp-dir to point straight at a report file
        if root.suffix == ".json" and root.exists():
            return root
        for n in names:
            c = root / n
            if c.exists():
                return c
    return None


def build_comparison(qgbmp_dir: str = "", output_root: Path = None) -> Path:
    root = output_root or BASELINE_ROOT
    labels = [lab for lab, _, _ in METRIC_COLUMNS]
    rows = []
    # QG-BMP (the full system) — include its report when present for direct comparison.
    qgbmp = _find_qgbmp_report(qgbmp_dir)
    if qgbmp is None:
        print(
            "[baselines] 提示：未找到 QG-BMP 报告（results/module_7 或 results/ablation/full/module_7）；对比表将只含 baseline。用 --qgbmp-dir 指定。"
        )
        qgbmp = RESULTS_DIR / "module_7" / "evaluation_report_feedback.json"
    else:
        print(f"[baselines] QG-BMP 报告: {qgbmp}")
    sources = [("QG-BMP(full)", qgbmp)] + [
        (m, root / m / "evaluation_report.json") for m in BASELINES
    ]
    for name, path in sources:
        if not path.exists():
            continue
        m = _load_metrics(path)
        rows.append({"method": name, **{lab: m.get(key, "") for lab, key, _ in METRIC_COLUMNS}})
    out = root / "baseline_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method"] + labels)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    core = [lab for lab, _, c in METRIC_COLUMNS if c]
    supp = [lab for lab, _, c in METRIC_COLUMNS if not c]
    print(f"[baselines] 对比表 -> {out}")
    print("  核心指标: " + " | ".join(core) + "   ‖ 补充: " + " | ".join(supp))
    hdr = "method".ljust(26) + "".join(lab.rjust(11) for lab in labels)
    print("  " + hdr)
    for r in rows:
        line = r["method"].ljust(26) + "".join(str(r.get(lab, "")).rjust(11) for lab in labels)
        print("  " + line)
    return out


def main() -> int:
    try:  # zh-CN Windows console is gbk; force UTF-8 so the 中文 table doesn't crash
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-n", type=int, default=0)
    ap.add_argument("--families", type=str, default="")
    ap.add_argument("--only", type=str, default="", help="逗号分隔的 baseline 名，只跑这些")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--no-eval", action="store_true", help="只生成不评估")
    ap.add_argument("--compare-only", action="store_true", help="只汇总已有结果，不重新生成")
    ap.add_argument(
        "--qgbmp-dir",
        type=str,
        default="",
        help="QG-BMP 结果目录或报告文件（默认自动找 results/ 与 results/ablation/full/）。",
    )
    ap.add_argument(
        "--refresh-personas",
        action="store_true",
        help="强制用当前数据集重跑 Module 1 生成 persona（重生成数据集后务必加上，否则用旧 persona）。",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="把所有 baseline 结果和对比表存到这个文件夹（默认 results/baseline/）。",
    )
    args = ap.parse_args()
    output_root = Path(args.output_dir) if args.output_dir else None

    if not args.compare_only:
        if args.refresh_personas or not Path(PERSONA_AGENTS_JSON).exists():
            print("[baselines] 运行 Module 1 生成 persona（当前数据集）...")
            initialize_persona_agents(
                persona_csv_path=PERSONA_CSV, output_json_path=PERSONA_AGENTS_JSON
            )
        ids = [int(x) for x in args.families.split(",") if x.strip().isdigit()] or None
        wanted = [x.strip() for x in args.only.split(",") if x.strip()] or list(BASELINES)
        for name in wanted:
            if name not in BASELINES:
                print(f"[baselines] 未知 baseline: {name}（可选: {list(BASELINES)}）")
                continue
            from baseline_common import run_baseline

            run_baseline(
                name,
                BASELINES[name],
                first_n=args.first_n,
                only_ids=ids,
                max_workers=args.max_workers,
                evaluate=not args.no_eval,
                output_root=output_root,
                credit_llm_splits=name in RESOLVING_BASELINES,
            )

    build_comparison(qgbmp_dir=args.qgbmp_dir, output_root=output_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
