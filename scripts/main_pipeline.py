"""Main QG-BMP pipeline runner: execute modules 1-7 (optionally per-family, with ablation presets and staged control)."""
import argparse
import importlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = CODE_DIR.parent
SELF_ENTRY = Path(__file__).resolve()  # this file, re-invoked per family as the --worker
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
from pipeline_common import (
    DEFAULT_EVAL_CSV,
    DEFAULT_PLANS_CSV,
    ablation_env,
    failed_family_ids,
    load_family_ids,
)

from src.config import *


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


# 运行开关（支持环境变量覆盖，供 per-family 子进程编排器分阶段控制）
RUN_MODULE_1 = _env_bool("RUN_MODULE_1", True)
RUN_MODULE_2 = _env_bool("RUN_MODULE_2", True)
RUN_MODULE_3 = _env_bool("RUN_MODULE_3", True)
RUN_MODULE_4 = _env_bool("RUN_MODULE_4", True)
RUN_MODULE_5 = _env_bool("RUN_MODULE_5", True)
RUN_MODULE_6 = _env_bool("RUN_MODULE_6", True)
RUN_MODULE_7 = _env_bool("RUN_MODULE_7", True)
RUN_EXPORT_FINAL_ITINERARY_TXT = _env_bool("RUN_EXPORT_FINAL_ITINERARY_TXT", True)


# ============== 编排器运行方式（直接改这里即可，无需命令行参数）==============
# 跑哪些家庭：
#   "all"         — 全部家庭
#   "failed_only" — 只重跑上一轮未通过的家庭（按 Module 7 评估自动识别）
#   "families"    — 只跑下面 ORCH_FAMILIES 里列出的 family_id
#   "first_n"     — 只跑前 ORCH_FIRST_N 个家庭
ORCH_RUN_MODE = os.getenv("ORCH_RUN_MODE", "first_n")
ORCH_FAMILIES = [15]  # 仅当 ORCH_RUN_MODE = "families" 时生效
ORCH_FIRST_N = 200  # 仅当 ORCH_RUN_MODE = "first_n" 时生效
ORCH_START_FROM = 0  # >0：任何模式下都只保留 family_id >= 此值
ORCH_INCLUDE_INFEASIBLE = False  # failed_only：是否也重跑预算真无解家庭（默认跳过）
# 并发与重试：
ORCH_MAX_PARALLEL = 4  # 并发运行的家庭数
ORCH_PER_FAMILY_CONCURRENCY = 4  # 每个家庭子进程的 LLM_MAX_CONCURRENCY
ORCH_MAX_RETRIES = 2  # 单家庭失败(rc!=0)重试次数；-1 = 一直重试到成功
ORCH_SKIP_MERGE = False  # True：跑完不合并到 results/ 主目录
ORCH_KEEP_SCRATCH = False  # True：保留每个家庭完整中间产物（默认只留一份 txt）


# ===================== 消融实验配置 =====================
# 【在主程序直接跑单个消融】把 ABLATION_PRESET 改成下面之一即可（默认 "manual"=用下方手动开关=full）：
#   "manual" /  / "no_negotiation" / "no_split" / "no_grounding" / "no_quality_control"
# 环境变量 ABLATION_PRESET 若已设置则优先（run_all_ablations.py 逐个消融就是靠它传入，不受此行影响）。
ABLATION_PRESET = "manual"

# ---- 组件开关（仅 "manual" 模式下这些手动值生效；选了上面的 preset 会覆盖对应开关）----
# Quality Control — 质量管控：
ENABLE_QUALITY_CONTROL = True
# Negotiation   — 多智能体多轮协商（M4 Round2 异议 / Round3 让步 / 复议投票）；
ENABLE_NEGOTIATION = True
# Split         — 分流活动解决（家庭内活动冲突时拆分子行程）【单因子消融维度 → no_split】
ENABLE_SPLIT = True
# Grounding     — 在线搜索真实性落地（Google Places ）【单因子消融维度 → no_grounding】
ENABLE_GROUNDING = True

# 非消融维度：奖惩记忆恒开（曾作消融维度，但判别力弱——仅影响 proposal_attempts/首次通过率，
# 最终满意度被下游补回，且不在 QG-BMP 命名支柱内，故固定开启、不再作为消融维度）
ENABLE_MEMORY = True

# 向后兼容映射（管线内部仍读旧变量名；门控与修复均由 ENABLE_QUALITY_CONTROL 统一驱动）
ENABLE_PROPOSAL_QUALITY_GATE = ENABLE_QUALITY_CONTROL  # M2 提案质量门控
ENABLE_REWARD_MEMORY = ENABLE_MEMORY
ENABLE_SPLIT_ACTIVITY_RESOLUTION = ENABLE_SPLIT
ENABLE_ONLINE_GROUNDING = ENABLE_GROUNDING
ENABLE_BACKTRACK_TO_MODULE2 = ENABLE_QUALITY_CONTROL  # M7 回退到 M2
ENABLE_NEGOTIATION_BACKTRACK_FEEDBACK = ENABLE_QUALITY_CONTROL
ENABLE_TARGETED_BACKTRACKING = ENABLE_QUALITY_CONTROL
ENABLE_CONSTRAINT_SATISFIABILITY_TYPING = True  # M6 评估特性：恒开
ENABLE_OBJECTIVE_EVALUATION = True  # M6 评估特性：恒开
ENABLE_NATURAL_TXT_EXPORT = True  # 始终开启，非消融维度

# 测试规模（支持环境变量覆盖，方便消融批量跑器传入）
_env_families = os.getenv("TEST_FIRST_N_FAMILIES", "")
TEST_FIRST_N_FAMILIES: Optional[int] = int(_env_families) if _env_families.isdigit() else 200
TEST_FIRST_N_BAD_FAMILIES: Optional[int] = None

# per-family 编排：只运行指定 family_id（逗号分隔）。设置后从第一轮起即把模块范围限定到这些家庭，
# 配合每家庭独立的 FAMILY_MATRIP_RESULTS_DIR 实现"每个家庭独立 M2→M7 流水线"。
_env_only = os.getenv("FAMILY_MATRIP_ONLY_FAMILY_IDS", "").strip()
ONLY_FAMILY_IDS: Optional[List[int]] = (
    [int(x) for x in _env_only.split(",") if x.strip().lstrip("-").isdigit()] if _env_only else None
)

# 并行参数
MODULE2_MAX_WORKERS_AGENTS = 5
MODULE3_MAX_WORKERS_FAMILIES = 5
MODULE4_MAX_WORKERS_FAMILIES = 5
MODULE5_MAX_WORKERS_FAMILIES = 5
MODULE7_MAX_WORKERS_FAMILIES = 5

# 质量门控参数
PROPOSAL_PASS_THRESHOLD = 80
PROPOSAL_STRONG_THRESHOLD = 90.0
PROPOSAL_MAX_REGEN_ATTEMPTS = 2

# 评价与回退参数
MODULE6_USE_LLM_JUDGE = True
NUM_REVISION_CANDIDATES = 2
BACKTRACK_AFTER_REVISION_ROUNDS = 2
MAX_REGENERATION_ATTEMPTS_PER_FAMILY = 2
NO_IMPROVEMENT_PATIENCE_ROUNDS = 2
MIN_IMPROVEMENT_DELTA = 0.1
MODULE7_FAST_EVAL_USE_LLM_JUDGE = False
MODULE7_FINAL_EVAL_USE_LLM_JUDGE = True
MAX_BACKTRACK_CYCLES = 1

ABLATION_PRESETS = {
    # ---- 完整模型（四维度：质量管控[门控+修复] + 协商 + 分流 + 落地）----
    "full": {
        "ENABLE_QUALITY_CONTROL": True,
        "ENABLE_NEGOTIATION": True,
        "ENABLE_SPLIT": True,
        "ENABLE_GROUNDING": True,
    },
    # ---- 单因子消融（逐一关闭）。注：no_quality_gate / no_repair 两个单因子消融已按决定移除
    #      （质量门控在本基准与修复冗余、单独看不出家庭级效果）。门控与修复已合并为单一
    #      ENABLE_QUALITY_CONTROL 开关，仅通过下方 no_quality_control 一并关闭考察。----
    "no_negotiation": {"ENABLE_NEGOTIATION": False},
    "no_split": {"ENABLE_SPLIT": False},
    "no_grounding": {"ENABLE_GROUNDING": False},
    # ---- 质量管控（门控+修复）消融：一次关掉两道"发现质量问题→纠正"工序
    #      （提案层门控 + 成稿层修复），考察去掉全部质量兜底后弱提案对家庭级结果的影响。----
    "no_quality_control": {"ENABLE_QUALITY_CONTROL": False},
}


def apply_main_ablation_preset():
    # ENABLE_QUALITY_CONTROL/NEGOTIATION/SPLIT/GROUNDING are set via globals().update()
    # below, so they need no `global` here; only the derived legacy flags do.
    global ENABLE_PROPOSAL_QUALITY_GATE, ENABLE_REWARD_MEMORY, ENABLE_SPLIT_ACTIVITY_RESOLUTION
    global ENABLE_ONLINE_GROUNDING, ENABLE_BACKTRACK_TO_MODULE2, ENABLE_NEGOTIATION_BACKTRACK_FEEDBACK
    global ENABLE_TARGETED_BACKTRACKING, ENABLE_CONSTRAINT_SATISFIABILITY_TYPING, ENABLE_OBJECTIVE_EVALUATION

    # 环境变量优先（run_all_ablations 逐个传入）；否则用文件顶部的 ABLATION_PRESET 选择器。
    preset = os.getenv("ABLATION_PRESET") or ABLATION_PRESET
    if preset in ("", "manual", "none"):
        active = "manual"
    else:
        if preset not in ABLATION_PRESETS:
            raise ValueError(f"未知消融设置: {preset}. 可选: {sorted(ABLATION_PRESETS)}")
        globals().update(ABLATION_PRESETS[preset])
        active = preset

    # 向后兼容：从组件开关派生旧变量（门控与修复均由 ENABLE_QUALITY_CONTROL 驱动）
    ENABLE_PROPOSAL_QUALITY_GATE = ENABLE_QUALITY_CONTROL
    # 约束可满足性分型 + objective 评估是 M6 的【评估】特性，不是 M2.5 的提案质量门——
    # 它们必须在所有消融下都开着，否则关掉质量管控时(no_quality_control)会把 split_satisfiable
    # 清零、并让 objective_final 因"未评估"而默认判过(=1.0)，从而污染消融对比。
    ENABLE_CONSTRAINT_SATISFIABILITY_TYPING = True
    ENABLE_OBJECTIVE_EVALUATION = True
    ENABLE_REWARD_MEMORY = ENABLE_MEMORY
    ENABLE_SPLIT_ACTIVITY_RESOLUTION = ENABLE_SPLIT
    ENABLE_ONLINE_GROUNDING = ENABLE_GROUNDING
    ENABLE_BACKTRACK_TO_MODULE2 = ENABLE_QUALITY_CONTROL
    ENABLE_NEGOTIATION_BACKTRACK_FEEDBACK = ENABLE_QUALITY_CONTROL
    ENABLE_TARGETED_BACKTRACKING = ENABLE_QUALITY_CONTROL

    # 同步到环境变量供子模块读取
    os.environ["ENABLE_ONLINE_GROUNDING"] = "true" if ENABLE_GROUNDING else "false"
    os.environ["REQUIRE_ONLINE_GROUNDING"] = "true" if ENABLE_GROUNDING else "false"
    os.environ["USE_OBJECTIVE_EVALUATION"] = "true"  # 评估特性：所有消融下恒开
    os.environ["USE_CONSTRAINT_SATISFIABILITY_TYPING"] = "true"
    os.environ["USE_SPLIT_ACTIVITY_RESOLUTION"] = "true" if ENABLE_SPLIT else "false"
    os.environ["USE_TARGETED_BACKTRACKING"] = "true" if ENABLE_QUALITY_CONTROL else "false"
    return active


ACTIVE_ABLATION_PRESET = apply_main_ablation_preset()
CURRENT_TARGET_FAMILY_IDS: Optional[List[int]] = None

PROPOSAL_REQUIRED_COLUMNS = ["agent_id", "family_id", "individual_proposal_json", "status"]
PLAN_REQUIRED_COLUMNS = ["family_id", "itinerary_text", "per_day_itinerary_json", "status"]


def safe_mkdir(path: Path):
    if path.exists() and (not path.is_dir()):
        raise RuntimeError(f"路径已存在但不是文件夹，请删除或改名：{path}")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        print("\n[Main] 创建文件夹失败，Windows 拒绝访问：")
        print(path)
        print("\n请检查编辑器/终端权限、OneDrive/杀毒软件占用、文件夹只读等问题。")
        raise e


def ensure_directories():
    for directory in RESULT_DIRECTORIES:
        safe_mkdir(directory)


def load_function(module_name: str, candidate_function_names: list[str]) -> Callable:
    module = importlib.import_module(module_name)
    for name in candidate_function_names:
        func = getattr(module, name, None)
        if callable(func):
            print(f"[Main] 已加载函数: {module_name}.{name}")
            return func
    raise ImportError(f"在模块 {module_name} 中没有找到函数: {candidate_function_names}")


def call_with_supported_kwargs(func: Callable, **kwargs) -> Any:
    signature = inspect.signature(func)
    params = signature.parameters
    has_kwargs = any((p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()))
    if has_kwargs:
        return func(**kwargs)
    supported = {k: v for k, v in kwargs.items() if k in params}
    ignored = sorted(set(kwargs) - set(supported))
    if ignored:
        print(f"[Main] 提示：{func.__name__} 不支持这些参数，已忽略：{ignored}")
    return func(**supported)


def require_file(path: Path, description: str):
    if not path.exists():
        raise FileNotFoundError(f"缺少{description}: {path}")


def run_step(step_name: str, func: Callable):
    print("\n" + "=" * 120)
    print(f"开始：{step_name}")
    print("=" * 120)
    start = time.time()
    try:
        result = func()
        print("\n" + "-" * 120)
        print(f"{step_name} 完成，用时 {time.time() - start:.2f} 秒。")
        print("-" * 120)
        return result
    except Exception as e:
        print("\n" + "!" * 120)
        print(f"{step_name} 失败，用时 {time.time() - start:.2f} 秒。")
        print(f"错误信息: {e}")
        traceback.print_exc()
        print("!" * 120)
        raise


def print_run_config():
    print("\n[Main] 运行配置：")
    print("[Main] 模块运行开关：")
    print(f"  RUN_MODULE_1 = {RUN_MODULE_1}")
    print(f"  RUN_MODULE_2 = {RUN_MODULE_2}")
    print(f"  RUN_MODULE_3 = {RUN_MODULE_3}")
    print(f"  RUN_MODULE_4 = {RUN_MODULE_4}")
    print(f"  RUN_MODULE_5 = {RUN_MODULE_5}")
    print(f"  RUN_MODULE_6 = {RUN_MODULE_6}")
    print(f"  RUN_MODULE_7 = {RUN_MODULE_7}")
    print(f"  RUN_EXPORT_FINAL_ITINERARY_TXT = {RUN_EXPORT_FINAL_ITINERARY_TXT}")
    print("[Main] 消融开关（5 维度）：")
    print(f"  ACTIVE_ABLATION_PRESET = {ACTIVE_ABLATION_PRESET}")
    print(f"  ENABLE_QUALITY_CONTROL = {ENABLE_QUALITY_CONTROL}  # 门控+修复")
    print(f"  ENABLE_NEGOTIATION   = {ENABLE_NEGOTIATION}")
    print(f"  ENABLE_SPLIT         = {ENABLE_SPLIT}")
    print(f"  ENABLE_GROUNDING     = {ENABLE_GROUNDING}")
    print(f"  （非消融）ENABLE_MEMORY = {ENABLE_MEMORY}  # 恒开")
    print("[Main] 规模与轮次参数：")
    print(f"  TEST_FIRST_N_FAMILIES = {TEST_FIRST_N_FAMILIES}")
    print(f"  TEST_FIRST_N_BAD_FAMILIES = {TEST_FIRST_N_BAD_FAMILIES}")
    print(f"  PROPOSAL_PASS_THRESHOLD = {PROPOSAL_PASS_THRESHOLD}")
    print(f"  PROPOSAL_MAX_REGEN_ATTEMPTS = {PROPOSAL_MAX_REGEN_ATTEMPTS}")
    print(f"  BACKTRACK_AFTER_REVISION_ROUNDS = {BACKTRACK_AFTER_REVISION_ROUNDS}")
    print(f"  MAX_BACKTRACK_CYCLES = {MAX_BACKTRACK_CYCLES}")
    print(f"  NUM_REVISION_CANDIDATES = {NUM_REVISION_CANDIDATES}")
    print(f"  MODULE6_USE_LLM_JUDGE = {MODULE6_USE_LLM_JUDGE}")
    print(f"  MODULE7_FAST_EVAL_USE_LLM_JUDGE = {MODULE7_FAST_EVAL_USE_LLM_JUDGE}")
    print(f"  MODULE7_FINAL_EVAL_USE_LLM_JUDGE = {MODULE7_FINAL_EVAL_USE_LLM_JUDGE}")


def read_csv_or_raise(path: Path, description: str) -> pd.DataFrame:
    require_file(path, description)
    if path.stat().st_size == 0:
        raise RuntimeError(
            f"[Safety Gateway] {description} 是空文件，停止运行。\n请检查上一模块是否没有生成任何可用结果: {path}"
        )
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise RuntimeError(
            f"[Safety Gateway] {description} 没有表头或没有任何列，停止运行。\n请检查上一模块是否没有生成任何可用结果: {path}"
        )


def validate_individual_proposals_before_downstream():
    df = read_csv_or_raise(INDIVIDUAL_PROPOSALS_CSV, "模块 3/4/5 输入 individual_agent_proposals.csv")
    missing = [c for c in PROPOSAL_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"[Safety Gateway] 模块 2 输出字段不完整，停止进入后续模块。\n缺少字段: {missing}\n请检查: {INDIVIDUAL_PROPOSALS_CSV}"
        )
    success_count = (df["status"].astype(str).str.lower() == "success").sum()
    if success_count == 0:
        raise RuntimeError(
            f"[Safety Gateway] 模块 2 没有任何成功 proposal，停止进入模块 3。\n这通常说明 LLM 调用函数、API Key 或模型调用失败。\n请先检查 Module 2 日志和文件: {INDIVIDUAL_PROPOSALS_CSV}"
        )
    print(f"[Safety Gateway] 模块 2 输出通过基本检查，成功 proposal 数量: {success_count}。")


def validate_generated_plans_before_module6():
    df = read_csv_or_raise(GENERATED_PLANS_CSV, "模块 6 输入 generated_plans.csv")
    missing = [c for c in PLAN_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"[Safety Gateway] 模块 5 输出不完整，停止进入模块 6。\n缺少字段: {missing}\n请先检查模块 5 是否全部 failed: {GENERATED_PLANS_CSV}"
        )
    success_count = (df["status"].astype(str).str.lower() == "success").sum()
    if success_count == 0:
        raise RuntimeError(
            f"[Safety Gateway] 模块 5 没有任何 success 方案，停止进入模块 6。\n请检查模块 2/4/5 是否产生有效结果: {GENERATED_PLANS_CSV}"
        )
    print("[Safety Gateway] 模块 5 输出通过基本检查，可以进入模块 6。")


# Module 5 generation failures are different from Module 7 evaluation failures.
# Module 5 writes this file when a family cannot produce a valid consensus plan.
# Main pipeline must include these families in the next Module 2 backtracking cycle;
# otherwise a pure Module-5 failure such as family_id=44 will be skipped by Module 7
# because Module 6 cannot evaluate a non-success plan.
MODULE5_FAILED_FAMILIES_JSON = (
    RESULTS_DIR / "module_5" / "module5_generation_failed_family_ids.json"
)


def _read_json_dict_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_module5_backtrack_target_family_ids() -> List[int]:
    """Read family_ids that failed during Module 5 consensus itinerary generation."""
    data = _read_json_dict_if_exists(MODULE5_FAILED_FAMILIES_JSON)
    ids = data.get("backtrack_to_module2_family_ids", data.get("failed_family_ids", []))
    if not isinstance(ids, list):
        ids = []
    output = []
    for value in ids:
        try:
            fid = int(value)
            if fid > 0:
                output.append(fid)
        except Exception:
            continue
    return sorted(set(output))


def load_module7_backtrack_target_family_ids() -> List[int]:
    """Read family_ids that failed Module 7 feedback revision / final evaluation."""
    data = _read_json_dict_if_exists(BACKTRACKING_FEEDBACK_JSON)
    if not data.get("backtrack_required"):
        return []
    family_ids = []
    for item in data.get("failed_families", []):
        try:
            fid = int(item.get("family_id"))
            if fid > 0:
                family_ids.append(fid)
        except Exception:
            continue
    return sorted(set(family_ids))


def load_backtrack_required() -> bool:
    # Backtracking is required if either Module 7 has failed families OR
    # Module 5 has generation-failed families.
    module7_required = bool(
        _read_json_dict_if_exists(BACKTRACKING_FEEDBACK_JSON).get("backtrack_required")
    )
    module5_failed_ids = load_module5_backtrack_target_family_ids()
    if module5_failed_ids:
        print(f"[Main] 检测到 Module 5 生成失败家庭，也将加入回退: family_id={module5_failed_ids}")
    return bool(module7_required or module5_failed_ids)


def reset_backtracking_feedback():
    BACKTRACKING_FEEDBACK_JSON.parent.mkdir(parents=True, exist_ok=True)
    BACKTRACKING_FEEDBACK_JSON.write_text(
        json.dumps(
            {"backtrack_required": False, "failed_families": []}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    # Avoid stale Module 5 failure IDs from an older run being reused when Module 5 is skipped.
    if MODULE5_FAILED_FAMILIES_JSON.exists():
        try:
            MODULE5_FAILED_FAMILIES_JSON.write_text(
                json.dumps(
                    {
                        "failed_family_ids": [],
                        "backtrack_to_module2_family_ids": [],
                        "records": [],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass


def load_backtrack_target_family_ids() -> List[int]:
    # Union of Module 7 evaluation failures and Module 5 generation failures.
    # This is the key fix: families that fail in Module 5 do not appear in
    # Module 7 feedback because Module 6 skips non-success plans.
    module7_ids = load_module7_backtrack_target_family_ids()
    module5_ids = load_module5_backtrack_target_family_ids()
    merged_ids = sorted(set(module7_ids + module5_ids))
    if module5_ids:
        print(f"[Main] Module 5 生成失败 family_id 已合并进回退目标: {module5_ids}")
    if module7_ids:
        print(f"[Main] Module 7 评价/修复失败 family_id: {module7_ids}")
    return merged_ids


def load_json_list_if_exists(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def merge_target_csv_output(
    previous_df: Optional[pd.DataFrame],
    current_csv_path: Path,
    target_family_ids: Optional[List[int]],
):
    if not target_family_ids or previous_df is None or (not current_csv_path.exists()):
        return
    current_df = pd.read_csv(current_csv_path)
    if current_df.empty or "family_id" not in current_df.columns:
        return
    target_set = {int(x) for x in target_family_ids}
    previous_keep = previous_df[~previous_df["family_id"].astype(int).isin(target_set)].copy()
    merged = pd.concat([previous_keep, current_df], ignore_index=True, sort=False)
    sort_cols = [c for c in ["family_id", "member_id", "agent_id"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(by=sort_cols)
    merged.to_csv(current_csv_path, index=False, encoding="utf-8-sig")
    print(
        f"[Main] 已合并 target-only CSV 输出: {current_csv_path.name}，保留非目标家庭，替换目标 family_id={sorted(target_set)}"
    )


def merge_target_json_output(
    previous_items: List[Dict[str, Any]],
    current_json_path: Path,
    target_family_ids: Optional[List[int]],
):
    if not target_family_ids or not previous_items or (not current_json_path.exists()):
        return
    current_items = load_json_list_if_exists(current_json_path)
    if not current_items:
        return
    target_set = {int(x) for x in target_family_ids}

    def is_target(item: Dict[str, Any]) -> bool:
        try:
            return int(item.get("family_id")) in target_set
        except Exception:
            return False

    merged = [item for item in previous_items if not is_target(item)]
    merged.extend(current_items)

    def sort_key(item: Dict[str, Any]):
        try:
            fid = int(item.get("family_id", 0))
        except Exception:
            fid = 0
        try:
            mid = int(item.get("member_id", 0))
        except Exception:
            mid = 0
        return (fid, mid, str(item.get("agent_id", "")))

    merged = sorted(merged, key=sort_key)
    current_json_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[Main] 已合并 target-only JSON 输出: {current_json_path.name}，保留非目标家庭，替换目标 family_id={sorted(target_set)}"
    )


def read_csv_optional(path: Path, description: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    if path.stat().st_size == 0:
        print(f"[Main] 提示：{description} 是空文件，snapshot 时按不存在处理: {path}")
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"[Main] 提示：{description} 没有表头，snapshot 时按不存在处理: {path}")
        return None


def snapshot_outputs(csv_path: Path, json_path: Path):
    previous_df = read_csv_optional(csv_path, csv_path.name)
    previous_json = load_json_list_if_exists(json_path) if json_path.exists() else []
    return (previous_df, previous_json)


def merge_outputs_after_target_run(
    previous_df: Optional[pd.DataFrame],
    previous_json: List[Dict[str, Any]],
    csv_path: Path,
    json_path: Path,
    target_family_ids: Optional[List[int]],
):
    merge_target_csv_output(previous_df, csv_path, target_family_ids)
    merge_target_json_output(previous_json, json_path, target_family_ids)


initialize_persona_agents = load_function(
    "src.module1_persona_agent_initialization",
    ["initialize_persona_agents", "run_persona_agent_initialization", "run_persona_initialization"],
)
generate_individual_proposals = load_function(
    "src.module2_individual_proposal_generation",
    ["generate_individual_proposals", "run_individual_proposal_generation"],
)
run_conflict_detection = load_function(
    "src.module3_conflict_detection",
    ["run_conflict_detection", "detect_conflicts", "run_conflict_detector"],
)
run_negotiation_protocol = load_function(
    "src.module4_negotiation_protocol", ["run_negotiation_protocol"]
)
generate_consensus_itineraries = load_function(
    "src.module5_consensus_itinerary_generation", ["generate_consensus_itineraries"]
)
evaluate_and_explain = load_function(
    "src.module6_evaluation_and_explanation", ["evaluate_and_explain"]
)
run_feedback_revision_loop = load_function(
    "src.module7_feedback_revision_loop", ["run_feedback_revision_loop"]
)
export_final_itinerary_txt = load_function(
    "src.final_itinerary_txt_export", ["export_final_itinerary_txt"]
)


def run_module_1():
    require_file(PERSONA_CSV, "模块 1 输入 persona CSV")
    return call_with_supported_kwargs(
        initialize_persona_agents,
        persona_csv_path=PERSONA_CSV,
        input_csv_path=PERSONA_CSV,
        output_json_path=PERSONA_AGENTS_JSON,
        test_first_n_families=TEST_FIRST_N_FAMILIES,
    )


def run_module_2():
    require_file(PERSONA_AGENTS_JSON, "模块 2 输入 persona_agents.json")
    return call_with_supported_kwargs(
        generate_individual_proposals,
        agents_json_path=PERSONA_AGENTS_JSON,
        output_csv_path=INDIVIDUAL_PROPOSALS_CSV,
        output_json_path=INDIVIDUAL_PROPOSALS_JSON,
        memory_dir=MEMORY_DIR,
        test_first_n_families=TEST_FIRST_N_FAMILIES,
        max_workers_agents=MODULE2_MAX_WORKERS_AGENTS,
        enable_quality_gate=ENABLE_PROPOSAL_QUALITY_GATE,
        enable_reward_memory=ENABLE_REWARD_MEMORY,
        proposal_pass_threshold=PROPOSAL_PASS_THRESHOLD,
        proposal_strong_threshold=PROPOSAL_STRONG_THRESHOLD,
        proposal_max_regen_attempts=PROPOSAL_MAX_REGEN_ATTEMPTS,
        target_family_ids=CURRENT_TARGET_FAMILY_IDS,
        backtracking_feedback_json_path=BACKTRACKING_FEEDBACK_JSON
        if ENABLE_BACKTRACK_TO_MODULE2
        else None,
    )


def run_module_3():
    require_file(PERSONA_AGENTS_JSON, "模块 3 输入 persona_agents.json")
    require_file(INDIVIDUAL_PROPOSALS_CSV, "模块 3 输入 individual_agent_proposals.csv")
    return call_with_supported_kwargs(
        run_conflict_detection,
        agents_json_path=PERSONA_AGENTS_JSON,
        proposals_csv_path=INDIVIDUAL_PROPOSALS_CSV,
        output_csv_path=CONFLICT_RESULTS_CSV,
        output_json_path=CONFLICT_RESULTS_JSON,
        test_first_n_families=TEST_FIRST_N_FAMILIES,
        max_workers_families=MODULE3_MAX_WORKERS_FAMILIES,
        target_family_ids=CURRENT_TARGET_FAMILY_IDS,
    )


def run_module_4():
    require_file(PERSONA_AGENTS_JSON, "模块 4 输入 persona_agents.json")
    require_file(INDIVIDUAL_PROPOSALS_CSV, "模块 4 输入 individual_agent_proposals.csv")
    require_file(CONFLICT_RESULTS_JSON, "模块 4 输入 conflict_detection_results.json")
    return call_with_supported_kwargs(
        run_negotiation_protocol,
        agents_json_path=PERSONA_AGENTS_JSON,
        proposals_csv_path=INDIVIDUAL_PROPOSALS_CSV,
        conflicts_json_path=CONFLICT_RESULTS_JSON,
        output_csv_path=NEGOTIATION_RESULTS_CSV,
        output_json_path=NEGOTIATION_RESULTS_JSON,
        memory_dir=MODULE4_MEMORY_DIR,
        test_first_n_families=TEST_FIRST_N_FAMILIES,
        max_workers_families=MODULE4_MAX_WORKERS_FAMILIES,
        backtracking_feedback_json_path=BACKTRACKING_FEEDBACK_JSON
        if ENABLE_NEGOTIATION_BACKTRACK_FEEDBACK
        else None,
        target_family_ids=CURRENT_TARGET_FAMILY_IDS,
        enable_negotiation=ENABLE_NEGOTIATION,
    )


def run_module_5():
    require_file(PERSONA_AGENTS_JSON, "模块 5 输入 persona_agents.json")
    require_file(INDIVIDUAL_PROPOSALS_CSV, "模块 5 输入 individual_agent_proposals.csv")
    require_file(CONFLICT_RESULTS_JSON, "模块 5 输入 conflict_detection_results.json")
    require_file(NEGOTIATION_RESULTS_JSON, "模块 5 输入 negotiation_protocol_results.json")
    return call_with_supported_kwargs(
        generate_consensus_itineraries,
        agents_json_path=PERSONA_AGENTS_JSON,
        proposals_csv_path=INDIVIDUAL_PROPOSALS_CSV,
        conflicts_json_path=CONFLICT_RESULTS_JSON,
        negotiation_json_path=NEGOTIATION_RESULTS_JSON,
        output_csv_path=GENERATED_PLANS_CSV,
        output_json_path=CONSENSUS_ITINERARY_JSON,
        test_first_n_families=TEST_FIRST_N_FAMILIES,
        max_workers_families=MODULE5_MAX_WORKERS_FAMILIES,
        target_family_ids=CURRENT_TARGET_FAMILY_IDS,
    )


def run_module_6():
    require_file(PERSONA_CSV, "模块 6 输入 persona CSV")
    require_file(PERSONA_AGENTS_JSON, "模块 6 输入 persona_agents.json")
    validate_generated_plans_before_module6()
    return call_with_supported_kwargs(
        evaluate_and_explain,
        persona_csv_path=PERSONA_CSV,
        agents_json_path=PERSONA_AGENTS_JSON,
        generated_plans_csv_path=GENERATED_PLANS_CSV,
        member_output_csv_path=EVALUATION_MEMBER_CSV,
        plan_output_csv_path=EVALUATION_PLAN_CSV,
        report_output_json_path=EVALUATION_REPORT_JSON,
        use_llm_judge=MODULE6_USE_LLM_JUDGE,
    )


def run_module_7():
    require_file(PERSONA_CSV, "模块 7 输入 persona CSV")
    require_file(PERSONA_AGENTS_JSON, "模块 7 输入 persona_agents.json")
    require_file(INDIVIDUAL_PROPOSALS_CSV, "模块 7 输入 individual_agent_proposals.csv")
    require_file(CONFLICT_RESULTS_JSON, "模块 7 输入 conflict_detection_results.json")
    require_file(NEGOTIATION_RESULTS_JSON, "模块 7 输入 negotiation_protocol_results.json")
    require_file(GENERATED_PLANS_CSV, "模块 7 输入 generated_plans.csv")
    require_file(EVALUATION_PLAN_CSV, "模块 7 输入 evaluation_plan_level.csv")
    require_file(EVALUATION_MEMBER_CSV, "模块 7 输入 evaluation_member_level.csv")
    return call_with_supported_kwargs(
        run_feedback_revision_loop,
        persona_csv_path=PERSONA_CSV,
        agents_json_path=PERSONA_AGENTS_JSON,
        proposals_csv_path=INDIVIDUAL_PROPOSALS_CSV,
        conflicts_json_path=CONFLICT_RESULTS_JSON,
        negotiation_json_path=NEGOTIATION_RESULTS_JSON,
        original_plans_csv_path=GENERATED_PLANS_CSV,
        plan_eval_csv_path=EVALUATION_PLAN_CSV,
        member_eval_csv_path=EVALUATION_MEMBER_CSV,
        final_revised_plans_csv_path=FINAL_REVISED_PLANS_CSV,
        revision_results_json_path=REVISION_RESULTS_JSON,
        revised_member_eval_csv_path=REVISED_EVALUATION_MEMBER_CSV,
        revised_plan_eval_csv_path=REVISED_EVALUATION_PLAN_CSV,
        revised_report_json_path=REVISED_EVALUATION_REPORT_JSON,
        test_first_n_bad_families=TEST_FIRST_N_BAD_FAMILIES,
        max_workers_families=MODULE7_MAX_WORKERS_FAMILIES,
        num_candidates=NUM_REVISION_CANDIDATES,
        no_improvement_patience_rounds=NO_IMPROVEMENT_PATIENCE_ROUNDS,
        min_improvement_delta=MIN_IMPROVEMENT_DELTA,
        max_regeneration_attempts_per_family=MAX_REGENERATION_ATTEMPTS_PER_FAMILY,
        backtrack_after_revision_rounds=BACKTRACK_AFTER_REVISION_ROUNDS,
        backtracking_feedback_json_path=BACKTRACKING_FEEDBACK_JSON,
        proposal_eval_csv_path=INDIVIDUAL_PROPOSALS_CSV,
        memory_dir=MEMORY_DIR,
        enable_backtracking_memory=ENABLE_REWARD_MEMORY,
        enable_revision=ENABLE_QUALITY_CONTROL,
    )


def run_export_final_itinerary_txt():
    require_file(FINAL_REVISED_PLANS_CSV, "TXT 导出输入 Module 7 final plans")
    require_file(REVISED_EVALUATION_MEMBER_CSV, "TXT 导出输入 Module 7 member evaluation")
    return call_with_supported_kwargs(
        export_final_itinerary_txt,
        output_dir=FINAL_ITINERARY_TXT_DIR,
        combined_output_txt=FINAL_ITINERARY_ALL_TXT,
    )


def any_module_2_to_7_enabled() -> bool:
    return any(
        [RUN_MODULE_2, RUN_MODULE_3, RUN_MODULE_4, RUN_MODULE_5, RUN_MODULE_6, RUN_MODULE_7,]
    )


def run_modules_2_to_7_cycle(cycle_index: int, target_family_ids: Optional[List[int]] = None):
    global CURRENT_TARGET_FAMILY_IDS
    CURRENT_TARGET_FAMILY_IDS = target_family_ids
    print("\n" + "#" * 120)
    print(f"[Main] 开始 Module 2→7 Cycle {cycle_index}")
    if target_family_ids:
        print(f"[Main] Backtracking target mode: 只处理未达标家庭 family_id={target_family_ids}")
    else:
        print("[Main] 正常模式：运行测试集/全量家庭。")
    print("#" * 120)
    if RUN_MODULE_2:
        prev_m2_csv, prev_m2_json = snapshot_outputs(
            INDIVIDUAL_PROPOSALS_CSV, INDIVIDUAL_PROPOSALS_JSON
        )
        run_step(f"Cycle {cycle_index} - 模块 2：Individual Proposal Generation", run_module_2)
        merge_outputs_after_target_run(
            prev_m2_csv,
            prev_m2_json,
            INDIVIDUAL_PROPOSALS_CSV,
            INDIVIDUAL_PROPOSALS_JSON,
            target_family_ids,
        )
        validate_individual_proposals_before_downstream()
    else:
        print(f"[Main] 跳过 Cycle {cycle_index} - 模块 2。将复用已有个人方案文件。")
        validate_individual_proposals_before_downstream()
    if RUN_MODULE_3:
        prev_m3_csv, prev_m3_json = snapshot_outputs(CONFLICT_RESULTS_CSV, CONFLICT_RESULTS_JSON)
        run_step(f"Cycle {cycle_index} - 模块 3：Conflict Detection", run_module_3)
        merge_outputs_after_target_run(
            prev_m3_csv,
            prev_m3_json,
            CONFLICT_RESULTS_CSV,
            CONFLICT_RESULTS_JSON,
            target_family_ids,
        )
    else:
        print(f"[Main] 跳过 Cycle {cycle_index} - 模块 3。将复用已有冲突检测文件。")
    if RUN_MODULE_4:
        prev_m4_csv, prev_m4_json = snapshot_outputs(
            NEGOTIATION_RESULTS_CSV, NEGOTIATION_RESULTS_JSON
        )
        run_step(f"Cycle {cycle_index} - 模块 4：Negotiation Protocol", run_module_4)
        merge_outputs_after_target_run(
            prev_m4_csv,
            prev_m4_json,
            NEGOTIATION_RESULTS_CSV,
            NEGOTIATION_RESULTS_JSON,
            target_family_ids,
        )
    else:
        print(f"[Main] 跳过 Cycle {cycle_index} - 模块 4。将复用已有协商结果。")
    if RUN_MODULE_5:
        prev_m5_csv, prev_m5_json = snapshot_outputs(GENERATED_PLANS_CSV, CONSENSUS_ITINERARY_JSON)
        run_step(f"Cycle {cycle_index} - 模块 5：Consensus Itinerary Generation", run_module_5)
        merge_outputs_after_target_run(
            prev_m5_csv,
            prev_m5_json,
            GENERATED_PLANS_CSV,
            CONSENSUS_ITINERARY_JSON,
            target_family_ids,
        )
    else:
        print(f"[Main] 跳过 Cycle {cycle_index} - 模块 5。将复用已有最终方案。")
    if RUN_MODULE_6:
        run_step(f"Cycle {cycle_index} - 模块 6：Evaluation and Explanation", run_module_6)
    else:
        print(f"[Main] 跳过 Cycle {cycle_index} - 模块 6。将复用已有评价结果。")
    if RUN_MODULE_7:
        run_step(f"Cycle {cycle_index} - 模块 7：Feedback Revision Loop", run_module_7)
    else:
        print(f"[Main] 跳过 Cycle {cycle_index} - 模块 7。反馈修复消融关闭。")
    CURRENT_TARGET_FAMILY_IDS = None


def worker_main():
    """Single-family (or batch) worker: runs Module 2->7 + txt export for the
    families selected via env vars (RUN_MODULE_*, FAMILY_MATRIP_ONLY_FAMILY_IDS).
    Invoked as a subprocess by the orchestrator with the --worker flag."""
    ensure_directories()
    print_run_config()
    if RUN_MODULE_1:
        run_step("模块 1：Persona Agent Initialization", run_module_1)
    else:
        print("跳过模块 1。")
    if any_module_2_to_7_enabled():
        if ENABLE_NEGOTIATION_BACKTRACK_FEEDBACK or ENABLE_BACKTRACK_TO_MODULE2:
            reset_backtracking_feedback()
        # per-family mode: restrict to the single target family from the first cycle.
        target_family_ids = ONLY_FAMILY_IDS
        if ONLY_FAMILY_IDS:
            print(f"[Main] per-family 模式：仅运行 family_id={ONLY_FAMILY_IDS}")
        for cycle in range(MAX_BACKTRACK_CYCLES + 1):
            run_modules_2_to_7_cycle(cycle, target_family_ids=target_family_ids)
            backtrack_required = load_backtrack_required() if RUN_MODULE_7 else False
            if backtrack_required and (not ENABLE_BACKTRACK_TO_MODULE2):
                print(
                    "\n[Main] Module 7 已发现三轮后仍未达标家庭，但 ENABLE_BACKTRACK_TO_MODULE2=False，回退机制已消融关闭。"
                )
                print(f"[Main] 可查看失败反馈: {BACKTRACKING_FEEDBACK_JSON}")
                break
            if (
                backtrack_required
                and ENABLE_BACKTRACK_TO_MODULE2
                and (cycle < MAX_BACKTRACK_CYCLES)
            ):
                target_family_ids = load_backtrack_target_family_ids()
                if not target_family_ids:
                    print("\n[Main] backtrack_required=True，但没有读取到 failed family_id，停止回退。")
                    break
                print("\n[Main] 模块 7 三轮后仍不合格，已写入回退反馈。")
                print(f"[Main] 只回退未达标家庭，不重跑已达标家庭: family_id={target_family_ids}")
                print("[Main] 将回到模块 2，重新生成这些家庭的个人方案；若启用协商反馈，则模块 4 会读取失败 feedback。")
                continue
            if backtrack_required and cycle >= MAX_BACKTRACK_CYCLES:
                target_family_ids = load_backtrack_target_family_ids()
                print("\n[Main] 仍需要 backtrack，但已达到 MAX_BACKTRACK_CYCLES，停止继续回退。")
                print(f"[Main] 仍未达标 family_id={target_family_ids}")
            break
    else:
        print("跳过模块 2→7。")
    if RUN_EXPORT_FINAL_ITINERARY_TXT and ENABLE_NATURAL_TXT_EXPORT:
        run_step("导出最终行程 TXT：Final Itinerary Input TXT", run_export_final_itinerary_txt)
    else:
        print("跳过最终行程 TXT 导出。")
    print("\n" + "=" * 120)
    print("[Main] 全部流程结束。")
    print("=" * 120)
    print(f"模块 2 个人方案: {INDIVIDUAL_PROPOSALS_CSV}")
    print(f"模块 5 原始最终行程: {GENERATED_PLANS_CSV}")
    print(f"模块 6 原始评价报告: {EVALUATION_REPORT_JSON}")
    print(f"模块 7 修正后最终行程: {FINAL_REVISED_PLANS_CSV}")
    print(f"模块 7 修正后评价报告: {REVISED_EVALUATION_REPORT_JSON}")
    print(f"模块 7 回退反馈: {BACKTRACKING_FEEDBACK_JSON}")
    print(f"最终行程 TXT 文件夹: {FINAL_ITINERARY_TXT_DIR}")
    print(f"最终行程 TXT 合并文件: {FINAL_ITINERARY_ALL_TXT}")


# ============================================================================
#  ORCHESTRATOR — per-family streaming (default mode). Spawns this same file as
#  a --worker subprocess per family so module-global path constants stay isolated.
# ============================================================================
PER_FAMILY_ROOT = RESULTS_DIR / "per_family"


def run_one_family(family_id: int, per_family_concurrency: int, max_retries: int = 2) -> dict:
    fam_dir = PER_FAMILY_ROOT / f"family_{family_id}"
    fam_dir.mkdir(parents=True, exist_ok=True)
    # Seed the family's Module 1 dir with the shared persona file so Module 2 can
    # read it from this subprocess's own RESULTS_DIR.
    (fam_dir / "module_1").mkdir(parents=True, exist_ok=True)
    shutil.copy2(PERSONA_AGENTS_JSON, fam_dir / "module_1" / "persona_agents.json")

    env = dict(os.environ)
    env.update(
        {
            "FAMILY_MATRIP_RESULTS_DIR": str(fam_dir),
            "FAMILY_MATRIP_ONLY_FAMILY_IDS": str(family_id),
            "RUN_MODULE_1": "false",  # persona already provided
            "RUN_MODULE_2": "true",
            "RUN_MODULE_3": "true",
            "RUN_MODULE_4": "true",
            "RUN_MODULE_5": "true",
            "RUN_MODULE_6": "true",
            "RUN_MODULE_7": "true",
            "RUN_EXPORT_FINAL_ITINERARY_TXT": "true",
            "TEST_FIRST_N_FAMILIES": "100000",  # ONLY_FAMILY_IDS takes precedence
            "LLM_MAX_CONCURRENCY": str(per_family_concurrency),
            "PYTHONIOENCODING": "utf-8",
        }
    )
    env.update(ablation_env())

    # Module 5 generation can flakily fail (LLM produces the wrong day count / a
    # placeholder), which aborts the whole subprocess (rc=1).  A fresh attempt
    # usually succeeds, so re-run the family up to max_retries times until rc==0.
    t0 = time.time()
    log_path = fam_dir / "run.log"
    rc = 1
    attempts = 0
    # max_retries < 0 => retry until success (bounded high); else 1 + max_retries attempts.
    _max_attempts = 10 ** 6 if max_retries < 0 else max(1, max_retries + 1)
    for attempt in range(1, _max_attempts + 1):
        attempts = attempt
        mode = "w" if attempt == 1 else "a"
        with log_path.open(mode, encoding="utf-8") as log:
            if attempt > 1:
                log.write(f"\n\n===== retry attempt {attempt} (previous rc != 0) =====\n")
            proc = subprocess.run(
                [sys.executable, str(SELF_ENTRY), "--worker"],
                cwd=str(CODE_DIR),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        rc = proc.returncode
        if rc == 0:
            break
        if attempt < _max_attempts:
            print(f"[per-family] family_{family_id} 第 {attempt} 次失败(rc={rc})，重新生成 ...")
    return {
        "family_id": family_id,
        "returncode": rc,
        "attempts": attempts,
        "elapsed": round(time.time() - t0, 1),
        "dir": fam_dir,
        "log": log_path,
    }


# --------------------------- merge per-family -> main ----------------------------
def _upsert_csv_into_results(fam_csv: Path, out_csv: Path) -> None:
    """Append one family's CSV rows into the shared results CSV (dedup by family_id)."""
    try:
        fam_df = pd.read_csv(fam_csv)
    except Exception:
        return
    frames = []
    if Path(out_csv).exists():
        try:
            frames.append(pd.read_csv(out_csv))
        except Exception:
            pass
    frames.append(fam_df)
    df = pd.concat(frames, ignore_index=True)
    if "family_id" in df.columns:
        df = df.drop_duplicates(subset=["family_id"], keep="last")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")


def _family_id_key(item):
    """Normalized family_id key (int when possible), or None when absent."""
    if not isinstance(item, dict):
        return None
    fid = item.get("family_id")
    if fid is None:
        return None
    try:
        return int(fid)
    except (TypeError, ValueError):
        return fid


def _upsert_json_into_results(fam_json: Path, out_json: Path) -> None:
    """Merge one family's JSON into the shared results JSON.

    Lists are deduplicated by family_id (last occurrence wins, matching the CSV
    merge), so a re-run family never leaves a stale duplicate entry; list items
    without a family_id are kept as-is.  Dicts are shallow-updated.
    """
    try:
        fam_data = json.loads(Path(fam_json).read_text(encoding="utf-8"))
    except Exception:
        return
    cur = None
    if Path(out_json).exists():
        try:
            cur = json.loads(Path(out_json).read_text(encoding="utf-8"))
        except Exception:
            cur = None
    if isinstance(fam_data, list):
        combined = (cur if isinstance(cur, list) else []) + fam_data
        by_fid = {}  # family_id -> latest item (last wins)
        others = []  # items without a family_id, kept in order
        for item in combined:
            key = _family_id_key(item)
            if key is None:
                others.append(item)
            else:
                by_fid[key] = item
        merged = [by_fid[k] for k in sorted(by_fid, key=lambda x: (str(type(x)), x))] + others
    elif isinstance(fam_data, dict):
        merged = cur if isinstance(cur, dict) else {}
        merged.update(fam_data)
    else:
        merged = fam_data
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")


# Dataset-level aggregate reports (dicts with a "metrics" block).  A SINGLE family's
# copy is meaningless: shallow dict-merging them per-family leaves the LAST family's
# single-family report at the dataset path (symptom: avg_group==avg_min because that
# one family's members are near-uniform, objective_final_pass_rate=0/1 for that one
# plan).  These are rebuilt from the MERGED CSVs by recompute_overall_reports(), which
# is their ONLY legitimate writer — so the per-family merge must skip them.
_AGGREGATE_REPORT_JSON_NAMES = {
    "evaluation_report.json",  # Module 6 dataset report
    "evaluation_report_feedback.json",  # Module 7 (post-repair) dataset report
}


def finalize_one_family(fid: int, keep_scratch: bool) -> None:
    """Merge one family's structured outputs into results/, then reduce its scratch
    folder to a single txt (the final itinerary)."""
    fam_dir = PER_FAMILY_ROOT / f"family_{fid}"
    if not fam_dir.exists():
        return
    # 1) merge structured outputs (skip persona/module_1, logs, and per-family txt —
    #    txt is regenerated in the normal place at the end).
    #    Also skip any loose file at the family-folder root (len(parts)==1): that is
    #    only ever the reduced single-txt artifact from a previous finalize or run.log,
    #    NOT a structured output — copying it would litter the results/ root.
    for path in fam_dir.rglob("*"):
        rel = path.relative_to(fam_dir)
        parts = rel.parts
        if not parts or len(parts) == 1 or parts[0] in ("module_1", "final_itinerary_txt"):
            continue
        if path.is_dir():
            if "agent_memories" in parts:
                dst = RESULTS_DIR / rel
                dst.mkdir(parents=True, exist_ok=True)
                for f in path.glob("*"):
                    if f.is_file():
                        try:
                            shutil.copy2(f, dst / f.name)
                        except Exception:
                            pass
            continue
        out = RESULTS_DIR / rel
        if rel.suffix == ".csv":
            _upsert_csv_into_results(path, out)
        elif rel.name in _AGGREGATE_REPORT_JSON_NAMES:
            # Dataset aggregate — do NOT per-family merge; recompute_overall_reports
            # rebuilds it from the merged CSVs. (See _AGGREGATE_REPORT_JSON_NAMES.)
            continue
        elif rel.suffix == ".json":
            _upsert_json_into_results(path, out)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(path, out)
            except Exception:
                pass
    if keep_scratch:
        return
    # 2) preserve the run log outside the family folder for debugging.
    log = fam_dir / "run.log"
    if log.exists():
        logs_dir = RESULTS_DIR / "per_family_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(log), str(logs_dir / f"family_{fid}.log"))
        except Exception:
            pass
    # 3) reduce the family folder to exactly one txt = the final itinerary.
    txt_name = f"family_{fid}_final_itinerary_input.txt"
    src_txt = fam_dir / "final_itinerary_txt" / txt_name
    content = None
    if src_txt.exists():
        try:
            content = src_txt.read_text(encoding="utf-8")
        except Exception:
            content = None
    shutil.rmtree(fam_dir, ignore_errors=True)
    fam_dir.mkdir(parents=True, exist_ok=True)
    if content is not None:
        (fam_dir / txt_name).write_text(content, encoding="utf-8")


def recompute_overall_reports() -> None:
    """Rebuild the dataset-level summary from the MERGED per-family CSVs, so the
    summary reflects the whole dataset exactly like a normal batch run."""
    from src.config import (
        EVALUATION_MEMBER_CSV,
        EVALUATION_PLAN_CSV,
        EVALUATION_REPORT_JSON,
        REVISED_EVALUATION_MEMBER_CSV,
        REVISED_EVALUATION_PLAN_CSV,
        REVISED_EVALUATION_REPORT_JSON,
    )
    from src.module6_evaluation_and_explanation import build_overall_report, compute_repair_effect

    def _defensive_metrics(plan_df: pd.DataFrame) -> dict:
        """Minimal-but-correct headline metrics straight from the merged CSV, used when
        build_overall_report raises (e.g. a schema-drifted / missing column). Guarantees
        the written report still matches the per-family data instead of leaving garbage."""

        def _m(col):
            if col not in plan_df.columns:
                return None
            s = pd.to_numeric(plan_df[col], errors="coerce").dropna()
            return round(float(s.mean()), 4) if len(s) else None

        def _r(col):
            if col not in plan_df.columns:
                return None

            def _b(v):
                t = str(v).strip().lower()
                return (
                    1.0
                    if t in ("true", "1", "1.0", "yes")
                    else (0.0 if t in ("false", "0", "0.0", "no", "nan", "") else None)
                )

            s = pd.Series([_b(v) for v in plan_df[col]]).dropna()
            return round(float(s.mean()), 4) if len(s) else None

        m = {
            "plan_count": int(len(plan_df)),
            "family_count": int(plan_df["family_id"].nunique())
            if "family_id" in plan_df.columns
            else int(len(plan_df)),
            "avg_group_satisfaction": _m("average_group_satisfaction"),
            "avg_minimum_member_satisfaction": _m("minimum_member_satisfaction"),
            "avg_max_min_fairness": _m("max_min_fairness"),
            "avg_satisfaction_variance": _m("satisfaction_variance"),
            "avg_hard_constraint_pass_rate": _m("hard_constraint_pass_rate"),
            "avg_restaurant_specificity_rate": _m("restaurant_specificity_rate"),
            "objective_final_pass_rate": _r("objective_final_pass"),
            "constraint_satisfiability_pass_rate": _r("constraint_satisfiability_pass"),
            "avg_acceptance_vote_ratio": _m("acceptance_vote_ratio"),
        }
        return {k: v for k, v in m.items() if v is not None}

    def _rebuild(plan_csv: Path, member_csv: Path, out_json: Path, label: str) -> None:
        if not (Path(plan_csv).exists() and Path(member_csv).exists()):
            return
        plan_df = None
        try:
            plan_df = pd.read_csv(plan_csv)
            member_df = pd.read_csv(member_csv)
            report = build_overall_report(plan_eval_df=plan_df, member_eval_df=member_df)
            Path(out_json).write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            m = report.get("metrics", {})
            print(
                f"[per-family] 重算{label}汇总: 家庭={m.get('family_count')} "
                f"群体满意度={m.get('avg_group_satisfaction')} "
                f"最低成员满意度={m.get('avg_minimum_member_satisfaction')} "
                f"硬约束通过率={m.get('avg_hard_constraint_pass_rate')}"
            )
        except Exception as exc:
            # LOUD failure (was silently swallowed → left the garbage single-family
            # report in place). Still write a defensive CSV-derived report so the
            # dataset report can never contradict the per-family data.
            import traceback

            print(f"[per-family] 重算{label}汇总失败（改用防御式聚合）: {exc}")
            traceback.print_exc()
            if plan_df is not None and not plan_df.empty:
                try:
                    dm = _defensive_metrics(plan_df)
                    Path(out_json).write_text(
                        json.dumps(
                            {
                                "overall_summary": f"（防御式聚合：build_overall_report 抛错，直接用 {plan_csv.name} 逐家庭均值）",
                                "metrics": dm,
                                "_fallback_aggregate": True,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    print(
                        f"[per-family] {label} 防御式汇总: 家庭={dm.get('family_count')} "
                        f"群体满意度={dm.get('avg_group_satisfaction')} "
                        f"最低成员满意度={dm.get('avg_minimum_member_satisfaction')} "
                        f"objective通过率={dm.get('objective_final_pass_rate')}"
                    )
                except Exception as exc2:
                    print(f"[per-family] {label} 防御式汇总也失败: {exc2}")

    _rebuild(EVALUATION_PLAN_CSV, EVALUATION_MEMBER_CSV, EVALUATION_REPORT_JSON, "Module 6")
    _rebuild(
        REVISED_EVALUATION_PLAN_CSV,
        REVISED_EVALUATION_MEMBER_CSV,
        REVISED_EVALUATION_REPORT_JSON,
        "Module 7",
    )

    # Repair-ablation diagnostic: compare pre-repair (M6) vs post-repair (M7).
    try:
        if Path(EVALUATION_PLAN_CSV).exists() and Path(REVISED_EVALUATION_PLAN_CSV).exists():
            pre_df = pd.read_csv(EVALUATION_PLAN_CSV)
            post_df = pd.read_csv(REVISED_EVALUATION_PLAN_CSV)
            repair = compute_repair_effect(pre_df, post_df)
            if repair and Path(REVISED_EVALUATION_REPORT_JSON).exists():
                rep = json.loads(Path(REVISED_EVALUATION_REPORT_JSON).read_text(encoding="utf-8"))
                rep.setdefault("metrics", {}).update(repair)
                Path(REVISED_EVALUATION_REPORT_JSON).write_text(
                    json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(
                    f"[per-family] 修复效果: 失败{repair.get('repair_failed_family_count')} "
                    f"修复好{repair.get('repair_fixed_family_count')} "
                    f"修复成功率={repair.get('repair_success_rate_on_failed')} "
                    f"通过率增益={repair.get('repair_pass_rate_gain')}"
                )
    except Exception as exc:
        print(f"[per-family] 修复效果计算失败: {exc}")


def reexport_combined_txt() -> None:
    """Re-run only the final TXT export over the merged main results."""
    env = dict(os.environ)
    env.update(
        {
            "RUN_MODULE_1": "false",
            "RUN_MODULE_2": "false",
            "RUN_MODULE_3": "false",
            "RUN_MODULE_4": "false",
            "RUN_MODULE_5": "false",
            "RUN_MODULE_6": "false",
            "RUN_MODULE_7": "false",
            "RUN_EXPORT_FINAL_ITINERARY_TXT": "true",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    env.pop("FAMILY_MATRIP_ONLY_FAMILY_IDS", None)
    # Keep FAMILY_MATRIP_RESULTS_DIR: the combined TXT must be exported over THIS run's results
    # dir. Normal runs don't set it (→ default results/); ablation runs set it to the preset dir
    # (results/ablation/<preset>/), which is where module_7 outputs actually live. Popping it made
    # the reexport read the default dir and crash on a missing generated_plans_feedback_final.csv.
    subprocess.run([sys.executable, str(SELF_ENTRY), "--worker"], cwd=str(CODE_DIR), env=env)


def run_orchestrator(
    mode: str = ORCH_RUN_MODE,
    families=None,
    first_n: int = ORCH_FIRST_N,
    start_from: int = ORCH_START_FROM,
    include_infeasible: bool = ORCH_INCLUDE_INFEASIBLE,
    max_parallel: int = ORCH_MAX_PARALLEL,
    per_family_concurrency: int = ORCH_PER_FAMILY_CONCURRENCY,
    max_retries: int = ORCH_MAX_RETRIES,
    skip_merge: bool = ORCH_SKIP_MERGE,
    keep_scratch: bool = ORCH_KEEP_SCRATCH,
    eval_csv: str = DEFAULT_EVAL_CSV,
    plans_csv: str = DEFAULT_PLANS_CSV,
) -> int:
    """Per-family streaming run into the current RESULTS_DIR: Module 1 is shared, then
    each family runs its own M2→M7 (with backtracking) in an isolated subprocess,
    concurrently — a family that finishes a module keeps going without waiting for the
    others.  Callable directly (e.g. by the ablation runner) as well as from the CLI."""
    families = [int(x) for x in (families or []) if str(x).strip().lstrip("-").isdigit()]
    print("=" * 90)
    print(
        f"[per-family] 运行方式 mode={mode}  并发家庭={max_parallel}  "
        f"每家庭并发={per_family_concurrency}  重试={max_retries}"
    )
    print("[per-family] 阶段 1：运行 Module 1（共享 persona 初始化，确定性、不调用 LLM）")
    initialize_persona_agents(persona_csv_path=PERSONA_CSV, output_json_path=PERSONA_AGENTS_JSON)
    family_ids = load_family_ids()
    if start_from > 0:
        family_ids = [fid for fid in family_ids if fid >= start_from]
    if mode == "failed_only":
        failed = set(
            failed_family_ids(eval_csv, plans_csv, exclude_infeasible=not include_infeasible)
        )
        family_ids = [f for f in family_ids if f in failed]
        skipped = "" if include_infeasible else "（已跳过预算真无解家庭，可设 include_infeasible=True 一并重跑）"
        print(f"[per-family] failed_only：自动识别未通过家庭 {len(family_ids)} 个{skipped}")
    elif mode == "families":
        wanted = set(families)
        family_ids = [f for f in family_ids if f in wanted]
    elif mode == "first_n":
        family_ids = family_ids[: max(0, first_n)]
    # else "all"：全部家庭
    if not family_ids:
        print("[per-family] 没有家庭可运行。")
        return 1
    print(
        f"[per-family] 家庭数={len(family_ids)}  "
        f"≈总并发 {max_parallel * per_family_concurrency} 个 LLM 请求"
    )
    print(f"[per-family] family_ids={family_ids}")

    print("=" * 90)
    print("[per-family] 阶段 2：每个家庭独立 M2→M7（含各自回退），子进程并发；")
    print("[per-family]          每个家庭一跑完即归并到 results/ 并把其文件夹瘦身成单个 txt")
    do_finalize = not skip_merge
    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futures = {
            ex.submit(run_one_family, fid, per_family_concurrency, max_retries): fid
            for fid in family_ids
        }
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            tries = f" (尝试{r.get('attempts', 1)}次)" if r.get("attempts", 1) > 1 else ""
            status = "OK" if r["returncode"] == 0 else f"FAIL(rc={r['returncode']})"
            if do_finalize:
                finalize_one_family(r["family_id"], keep_scratch)
            print(
                f"[per-family] ({done}/{len(family_ids)}) family_{r['family_id']} "
                f"{status}{tries} 用时 {r['elapsed']}s"
            )

    failed = [r["family_id"] for r in results if r["returncode"] != 0]
    print("=" * 90)
    print(f"[per-family] 全部家庭完成，用时 {round(time.time() - t0, 1)}s，失败 {len(failed)} 个: {failed}")

    if not skip_merge:
        print("[per-family] 阶段 3：重算数据集级汇总 + 主目录重导出 txt")
        recompute_overall_reports()
        reexport_combined_txt()
        print("[per-family] 完成。results/ 与正常批量完全一致；" "per_family/family_<id>/ 每个家庭仅保留一个最终行程 txt。")
    return 0 if not failed else 2


def orchestrator_main() -> int:
    # 默认全部取自文件顶部的 ORCH_* 配置；命令行参数可临时覆盖（不改文件也能跑）。
    ap = argparse.ArgumentParser(
        description="主流程：每家庭流式 M2→M7（子进程隔离并发）。" "默认读取文件顶部 ORCH_* 配置；命令行参数可临时覆盖。"
    )
    ap.add_argument(
        "--mode",
        default=ORCH_RUN_MODE,
        choices=["all", "failed_only", "families", "first_n"],
        help="跑哪些家庭（默认取顶部 ORCH_RUN_MODE）",
    )
    ap.add_argument(
        "--families",
        type=str,
        default=",".join(str(x) for x in ORCH_FAMILIES),
        help="mode=families 时的 family_id 列表（逗号分隔）",
    )
    ap.add_argument("--first-n", type=int, default=ORCH_FIRST_N, help="mode=first_n 时取前 N 个")
    ap.add_argument("--start-from", type=int, default=ORCH_START_FROM, help=">0：只保留 family_id>=此值")
    ap.add_argument(
        "--include-infeasible",
        action="store_true",
        default=ORCH_INCLUDE_INFEASIBLE,
        help="mode=failed_only 时也重跑预算真无解家庭",
    )
    ap.add_argument("--max-parallel", type=int, default=ORCH_MAX_PARALLEL, help="并发运行的家庭数")
    ap.add_argument(
        "--per-family-concurrency",
        type=int,
        default=ORCH_PER_FAMILY_CONCURRENCY,
        help="每个家庭子进程的 LLM_MAX_CONCURRENCY",
    )
    ap.add_argument(
        "--max-retries", type=int, default=ORCH_MAX_RETRIES, help="单家庭失败(rc!=0)重试次数；-1=一直到成功"
    )
    ap.add_argument("--skip-merge", action="store_true", default=ORCH_SKIP_MERGE, help="跑完不合并到主目录")
    ap.add_argument(
        "--keep-scratch",
        action="store_true",
        default=ORCH_KEEP_SCRATCH,
        help="保留每个家庭完整中间产物（默认只留一份 txt）",
    )
    ap.add_argument("--eval-csv", type=str, default=DEFAULT_EVAL_CSV, help="failed_only 读取的评估 CSV")
    ap.add_argument(
        "--plans-csv", type=str, default=DEFAULT_PLANS_CSV, help="failed_only 读取的方案 CSV"
    )
    args = ap.parse_args()

    return run_orchestrator(
        mode=args.mode,
        families=[x for x in args.families.split(",") if x.strip()],
        first_n=args.first_n,
        start_from=args.start_from,
        include_infeasible=args.include_infeasible,
        max_parallel=args.max_parallel,
        per_family_concurrency=args.per_family_concurrency,
        max_retries=args.max_retries,
        skip_merge=args.skip_merge,
        keep_scratch=args.keep_scratch,
        eval_csv=args.eval_csv,
        plans_csv=args.plans_csv,
    )


if __name__ == "__main__":
    # Default = orchestrator (per-family streaming). The orchestrator re-invokes this
    # same file as a single-family worker via the --worker flag (subprocess isolation).
    if "--worker" in sys.argv:
        worker_main()
    else:
        sys.exit(orchestrator_main())
