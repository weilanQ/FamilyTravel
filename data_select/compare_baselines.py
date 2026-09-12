"""QG-BMP (full) vs external baselines — rigorous comparison, same standard as compare_ablations.

The old `run_baselines.build_comparison` read each method's `evaluation_report.json` (the dataset
AGGREGATE, which has a known upsert/aggregation bug — see the ablation memory) and emitted a flat
metric table with no per-conflict-level split and no significance. This script instead:
  * reads the PER-FAMILY `evaluation_plan_level*.csv` (never the report JSON),
  * replicates the module6 pass gate via compare_ablations.plan_passed (identical to the pipeline),
  * reports overall + per-conflict-level pass rate, discriminating metrics vs full, the failure-gate
    tally, and PAIRED bootstrap significance vs full (same families) — proving QG-BMP > each baseline.

Baselines share QG-BMP's post-processing/grounding/eval; they differ ONLY in generation and get NO
conflict-resolution (baseline_common.finalize_plan → enforce_rule_consistency(negotiation_resolved=
False)), so a gap here reflects the multi-agent method, not the harness.

Usage: python scripts/compare_baselines.py [--baseline-root results/baseline] [--full-dir results/ablation/full]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_ablations import (  # noqa: E402  (reuse the exact same gate + stats as the ablation report)
    TH,
    _stars,
    cliffs_delta,
    conflict_level_map,
    mcnemar_exact,
    paired_bootstrap,
    plan_passed,
)

CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = CODE_DIR.parent

# Method order: full first (the reference), then the 2 generation-only anchors, then the
# 2 conflict-RESOLVING budget-matched baselines. (single_agent_cot / independent_multi_agent
# removed 2026-07: converged with their siblings and are subsumed by sa_resolve /
# generic_ma_debate; their old results dirs are simply no longer read.)
BASELINES = [
    "single_agent_direct",
    "preference_aggregation",
    "sa_resolve",
    "generic_ma_debate",
]
# These two RESOLVE conflicts (verifier-refined, credit_llm_splits=True) at a token budget
# near the full system — they answer "is the typed graph + negotiation better than a
# budget-matched central resolver / generic debate", not "is resolution better than none".
RESOLVING_BASELINES = {"sa_resolve", "generic_ma_debate"}
METRICS = [
    "minimum_member_satisfaction",
    "average_group_satisfaction",
    "max_min_fairness",
    "hard_constraint_pass_rate",
    "feasibility_pass_rate",
    "unresolved_conflict_penalty",
    "restaurant_specificity_rate",
]


def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


def _eval_csv(method: str, baseline_root: Path, full_dir: Path) -> Path | None:
    """Baselines write evaluation_plan_level.csv directly in <root>/<method>/; QG-BMP full lives
    under module_7 (feedback) or module_6."""
    if method == "full":
        for c in [
            full_dir / "module_7" / "evaluation_plan_level_feedback.csv",
            full_dir / "module_6" / "evaluation_plan_level.csv",
        ]:
            if c.exists():
                return c
        return None
    c = baseline_root / method / "evaluation_plan_level.csv"
    return c if c.exists() else None


OUTCOME_TH = dict(TH)
OUTCOME_TH["SKIP_CONSENSUS"] = True
OUTCOME_TH["SKIP_ACCEPTANCE"] = True


def load_method(csv: Path, clv: dict) -> pd.DataFrame:
    df = pd.read_csv(csv)
    passed, gate, outcome = [], [], []
    for _, r in df.iterrows():
        ok, g = plan_passed(r)
        passed.append(ok)
        gate.append(g)
        # PRIMARY metric: outcome-only pass (process gates consensus+acceptance skipped).
        # Verified 2026-07-17 to coincide family-by-family with the process-aware pass on
        # every preset and baseline; this column keeps that identity auditable per run.
        outcome.append(plan_passed(r, OUTCOME_TH)[0])
    df["_passed"] = passed
    df["_fail_gate"] = gate
    df["_passed_outcome"] = outcome
    fid = df["family_id"] if "family_id" in df.columns else pd.Series(dtype="int64", index=df.index)
    df["_conflict_level"] = fid.map(clv).fillna("?")
    return df


def _mean(d: pd.DataFrame, col: str):
    return pd.to_numeric(d[col], errors="coerce").mean() if col in d.columns else None


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="QG-BMP vs 外部基线：严格对比(对齐消融报告标准)。")
    ap.add_argument("--baseline-root", default=str(PROJECT_DIR / "results" / "baseline"))
    ap.add_argument("--full-dir", default=str(PROJECT_DIR / "results" / "ablation" / "full"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    baseline_root = Path(args.baseline_root)
    full_dir = Path(args.full_dir)
    clv = conflict_level_map()

    methods = ["full"] + BASELINES
    data: dict[str, pd.DataFrame] = {}
    for m in methods:
        csv = _eval_csv(m, baseline_root, full_dir)
        if csv is not None:
            data[m] = load_method(csv, clv)
    if "full" not in data:
        print("[compare_baselines] 未找到 QG-BMP(full) 评估 CSV，无法对比。")
        return 1
    full = data["full"]
    # Same-subset full: the PRIMARY comparison row. Every baseline runs on the stratified
    # subset; restricting full to the union of the baselines' family_ids puts ALL methods on
    # one identical family set (the 200-family full row is kept as reference only).
    subset_ids: set = set()
    for m in BASELINES:
        if m in data and "family_id" in data[m].columns:
            subset_ids |= set(data[m]["family_id"].astype(int))
    if subset_ids and "family_id" in full.columns:
        full_sub = full[full["family_id"].astype(int).isin(subset_ids)].copy()
    else:
        full_sub = full.copy()
    data["full_sub"] = full_sub
    methods = ["full_sub", "full"] + BASELINES
    present = [m for m in methods if m in data]
    label = {"full_sub": "QG-BMP(full,同子集)", "full": "QG-BMP(full,200家参考)"}
    counts = {label.get(m, m): len(data[m]) for m in present}

    # Outcome-only identity check: if it ever diverges from the process-aware pass for any
    # method, report it loudly (the paper's primary-metric claim depends on this identity).
    outcome_notes = []
    for m in present:
        d = data[m]
        diffs = int((d["_passed"] != d["_passed_outcome"]).sum())
        if diffs:
            outcome_notes.append(f"{label.get(m, m)}: {diffs} 家庭 outcome-only ≠ process-aware")
    levels = ["Low Conflict", "Medium Conflict", "High Conflict"]
    out = [
        f"# QG-BMP vs 外部基线对比  (thresholds group≥{TH['MIN_GROUP']:.0f}/min≥{TH['MIN_MEMBER']:.0f})",
        "",
        f"> **样本量**: {counts}。所有方法与 QG-BMP 共享 grounding/评估/后处理。**主对比行是 QG-BMP(full,同子集)**"
        "——与所有基线在同一批家庭上比较；200 家行仅作参考。single_agent_direct / preference_aggregation 是 "
        "generation-only 锚点，不获得任何冲突化解(`negotiation_resolved=False`)；**sa_resolve / generic_ma_debate "
        "是预算匹配的冲突解决基线**，其 LLM 自产的结构完整 split 会被承认(`credit_llm_splits=True`，与 full 同一"
        "完整性标准)，但不获得任何程序化补齐。指标读 per-family `evaluation_plan_level*.csv`(非有聚合 bug 的 "
        "report.json)，门判定与 module6 一致。",
        "",
        (
            "> ⚠️ **outcome-only ≠ process-aware**: " + "；".join(outcome_notes)
            if outcome_notes
            else "> ✅ **outcome-only 校验**: 全部方法 outcome-only pass(去 consensus/acceptance 门)与 "
            "process-aware pass 逐家庭一致——主表数字即 outcome-only 口径。"
        ),
        "",
        "## 1) 通过率（overall + 按冲突等级）",
        "",
        "| method | overall | " + " | ".join(l.split()[0] for l in levels) + " | n |",
        "|" + "---|" * (len(levels) + 3),
    ]
    for m in present:
        d = data[m]
        row = f"| {label.get(m, m)} | {d['_passed'].mean()*100:.0f}% "
        for lv in levels:
            sub = d[d["_conflict_level"] == lv]
            row += f"| {('%.0f%% (n=%d)' % (sub['_passed'].mean()*100, len(sub))) if len(sub) else '—'} "
        row += f"| {len(d)} |"
        out.append(row)
    out += [
        "",
        "> **读法**: 对比矩阵是 2×2——{单智能体 direct, 画像聚合 prefAgg} 为**不化解**锚点(~33% 天花板 = Low 占比，"
        "证明'不化解冲突→质量不动'；已验证 CoT / independent-MA 与其完全趋同，故从阵容中移除)；**sa_resolve / "
        "generic_ma_debate 是方法性对照**：两者以接近 QG-BMP 的预算显式化解冲突(中央单智能体 4 轮验证器修复 / "
        "同 persona 4 轮自由辩论)，它们与 QG-BMP(full,同子集)的差距衡量的是 **typed conflict graph + "
        "SHARED/SPLIT/UNSAT 分派 + 五算子的机制增益**，而不再是'化解 vs 不化解'。",
    ]

    # 2) discriminating metrics vs full on the SAME subset (Δ in parentheses)
    out += ["", "## 2) 判别指标（均值，括号为 Δ vs QG-BMP(full,同子集)）", ""]
    out += [
        "| metric | " + " | ".join(label.get(m, m) for m in present) + " |",
        "|" + "---|" * (len(present) + 1),
    ]
    for metric in METRICS:
        fmean = _mean(full_sub, metric)
        cells = []
        for m in present:
            v = _mean(data[m], metric)
            if v is None:
                cells.append("—")
            elif m in ("full_sub", "full") or fmean is None:
                cells.append(fmt(v))
            else:
                cells.append(f"{fmt(v)} ({v - fmean:+.2f})")
        out.append(f"| {metric} | " + " | ".join(cells) + " |")

    # 3) failure-gate tally
    out += ["", "## 3) 失败家庭卡在哪道门（每方法计数）", ""]
    for m in present:
        d = data[m]
        fails = d[~d["_passed"]]
        if fails.empty:
            out.append(f"- **{label.get(m, m)}**: 全通过 ✅")
            continue
        tally = fails["_fail_gate"].value_counts()
        out.append(
            f"- **{label.get(m, m)}** ({len(fails)}/{len(d)} 失败): "
            + ", ".join(f"{g}×{c}" for g, c in tally.items())
        )

    # 3b) cost/efficiency — the 4 baselines differ mainly here (quality is capped identically by the
    # shared harness). QG-BMP's per-family token is its CUMULATIVE pipeline cost (M2+M4+M5+M7), not the
    # final-plan CSV alone (which hides negotiation/repair), so the "compute ladder" is honest.
    try:
        from compare_ablations import cumulative_cost

        full_cum = cumulative_cost(full_dir).get("total")
    except Exception:
        full_cum = None
    out += [
        "",
        "## 3b) 成本效率（每家庭均值 token / 秒 + 每 1 万 token 换多少通过率）",
        "",
        "| method | 每家庭 token | runtime(s) | pass% | pass% / 10k tok |",
        "|---|---|---|---|---|",
    ]
    for m in present:
        d = data[m]
        if m in ("full_sub", "full") and full_cum:
            tok_val = float(full_cum)
            tok = f"{full_cum:,.0f}（累计 M2+M4+M5+M7）"
        else:
            tok_val = _mean(d, "total_tokens") or 0.0
            tok = fmt(tok_val, 0)
        pass_pct = d["_passed"].mean() * 100
        eff = (pass_pct / (tok_val / 10000.0)) if tok_val else None
        out.append(
            f"| {label.get(m, m)} | {tok} | {fmt(_mean(d,'runtime_seconds'), 1)} "
            f"| {pass_pct:.0f}% | {fmt(eff, 1)} |"
        )
    out += [
        "",
        "> generation-only 基线是单次生成(~1k token)；**sa_resolve / generic_ma_debate 是预算匹配的"
        "冲突解决基线**(与 QG-BMP 同模型、同 grounding、同验证器、同轮数量级)。因此本表同时呈现两条信息："
        "绝对质量(pass%)与单位算力质量(pass%/10k tok)。若 QG-BMP 在**同预算**下 pass% 仍显著高于两个"
        "resolving 基线，则优势来自 typed graph + 分派机制而非算力；反之若 resolving 基线追平，"
        "则说明化解本身(而非本文机制)是关键——两种结论都写得进论文,但只有跑了才知道。",
    ]

    # 4) paired bootstrap significance vs full (same families)
    out += [
        "",
        "## 4) 配对 bootstrap 95% CI + 显著性（Δ = 基线 − QG-BMP，同一批家庭）",
        "",
        "| metric | " + " | ".join(label.get(m, m) for m in BASELINES if m in data) + " |",
        "|" + "---|" * (sum(1 for m in BASELINES if m in data) + 1),
    ]
    stat_metrics = [
        ("_passed", "pass_rate", 3),
        ("minimum_member_satisfaction", "min_sat", 2),
        ("max_min_fairness", "fairness", 3),
        ("average_group_satisfaction", "avg_sat", 2),
    ]
    for col, lbl, nd in stat_metrics:
        cells = []
        for m in BASELINES:
            if m not in data:
                continue
            res = paired_bootstrap(full, data[m], col)
            if res is None:
                cells.append("—")
            else:
                dlt, lo, hi, p, n = res
                cells.append(f"{dlt:+.{nd}f} [{lo:+.{nd}f},{hi:+.{nd}f}]{_stars(p)}")
        out.append(f"| {lbl} | " + " | ".join(cells) + " |")
    # 4b) exact McNemar for the binary pass difference + Cliff's delta effect size for min_sat
    out += [
        "",
        "### 4b) 二元 pass 的 exact McNemar + 效应量（同一批家庭）",
        "",
        "| method | McNemar n01/n10 (full胜/基线胜) | exact p | Cliff's δ (min_sat) |",
        "|---|---|---|---|",
    ]
    for m in BASELINES:
        if m not in data:
            continue
        mc = mcnemar_exact(full, data[m])
        cd = cliffs_delta(full, data[m], "minimum_member_satisfaction")
        if mc is None:
            out.append(f"| {label.get(m, m)} | — | — | {fmt(cd,3)} |")
        else:
            n01, n10, p = mc
            out.append(f"| {label.get(m, m)} | {n01}/{n10} | {p:.2e}{_stars(p)} | {fmt(cd,3)} |")
    out += [
        "",
        "> 主假设族(进 Holm 校正): 每方法 vs full 的 {pass_rate, min_sat, fairness}；avg_sat 及其余连续指标为"
        "探索性,不进同一校正族。|Cliff's δ|≥0.474 为大效应。",
    ]
    out += [
        "",
        "> \\*p<.05, \\*\\*p<.01, \\*\\*\\*p<.001（bootstrap 双侧，n_boot=5000）。CI 不含 0 即显著。所有基线相对 QG-BMP 的"
        " min_sat/fairness/pass_rate 若均为显著负，则 QG-BMP 的多智能体化解在满意度、公平、通过率上**一致且显著**优于外部基线。",
        "",
        "## 5) 为何 generation-only 基线在质量上趋同（诚实说明，非 bug；不适用于 2 个 resolving 基线）",
        "",
        "> generation-only 基线(现保留 single-agent direct、preference-aggregation；已移除的 CoT、"
        "independent-multi-agent 亦然)在**质量指标上几乎相同**(相同 consensus_not_full 失败数、相同 "
        "unresolved_conflict_penalty、must_have_rate 同为 ~0.92、min_sat 差<0.5)，但在 **§3b 成本上迥异**"
        "(token 最省与最贵相差约 1/3)。原因有二,都不是 bug:(1) **质量天花板由共享"
        "后处理决定**——must-have 注入 / 偏好·grounding 强制匹配把质量拉到同一水平,故生成方法主要影响'花多少算力',而非"
        "结果质量;(2) 冲突家庭上**四者都不做协商化解**,留着同一批注入冲突→撞同一堵墙。**判别性结论**:在家庭协同规划上"
        "真正起作用的是**协商化解机制**(QG-BMP 独有),**堆智能体 / 加 CoT 只是更贵、并不能在无化解时提升质量**——这正是"
        "4 基线一起要证明的点(共同划出'无化解'的质量天花板,且更复杂的方法未必更省)。若要让基线在**质量/通过率**上也分层,"
        "需撤掉基线的 must-have 注入(让裸生成覆盖决定结果),那是另一种公平口径,需重跑。",
        "",
    ]

    report = "\n".join(out)
    print(report)
    dest = Path(args.out) if args.out else baseline_root / "baseline_comparison.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(report + "\n", encoding="utf-8")
    print(f"\n[compare_baselines] 报告已写入: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
