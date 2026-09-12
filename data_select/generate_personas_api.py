"""Generate per-member persona profiles for each selected family (LLM-backed, with a deterministic offline fallback)."""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from budget_feasibility import (  # noqa: E402
    BUDGET_PASS_TOLERANCE,
    is_feasible,
    parse_budget_from_query,
    rewrite_query_budget,
)
from dataset_utils import extract_json, safe_float, safe_text, time_to_minutes  # noqa: E402

INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "selected_travel_queries.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "FamilyTravel_Mini.csv"

# 调用 LLM 生成 persona；置 False（或环境变量 DATA_SELECT_USE_API=0）则全部走
# 确定性规则生成，无需网络/密钥，便于离线自检。
USE_API = os.getenv("DATA_SELECT_USE_API", "1").lower() not in ("0", "false", "no")

MAX_WORKERS = 3  # 同时生成的 query 数
SAVE_EVERY = 5  # 每完成 5 个 query 临时保存一次
# 质量门控：家庭不合格就重生成，直到合格。
#   >0：最多重试这么多次，仍不合格则用规则兜底（确定性、必定合格）；
#   <=0：无限重试直到合格（真正"一直生成"，不兜底——注意可能持续烧 API）。
MAX_QUALITY_ATTEMPTS = 10

# 硬约束合理区间
WALK_MIN_KM, WALK_MAX_KM = 1.5, 8.0
RETURN_MIN_MIN, RETURN_MAX_MIN = 18 * 60 + 30, 22 * 60  # 18:30 ~ 22:00


CONFLICT_LEVELS = ["Low Conflict", "Medium Conflict", "High Conflict"]


_PATTERN_ALIAS = {"L": "Low Conflict", "M": "Medium Conflict", "H": "High Conflict"}
CONFLICT_LEVEL_PATTERN = [
    _PATTERN_ALIAS[tok.strip().upper()[:1]]
    for tok in os.getenv("CONFLICT_LEVEL_PATTERN", "L,M,H").split(",")
    if tok.strip() and tok.strip().upper()[:1] in _PATTERN_ALIAS
] or list(CONFLICT_LEVELS)

# 按规模拆分的家庭结构池：分配规模由预算决定，再在该规模内循环取结构。
FAMILY_STRUCTURES_BY_SIZE = {
    3: [
        "father + mother + child",
        "father + mother + teenager",
        "father + mother + elder",
        "father + 2 children",
        "father + child + teenager",
        "mother + 2 children",
        "mother + child + teenager",
        "father + elder + child",
        "father + elder + teenager",
        "mother + elder + child",
        "mother + elder + teenager",
    ],
    4: [
        "father + mother + 2 children",
        "father + mother + child + teenager",
        "father + mother + 2 teenagers",
        "father + mother + elder + child",
        "father + mother + elder + teenager",
        "father + elder + 2 children",
        "father + elder + child + teenager",
        "mother + elder + 2 children",
        "mother + elder + child + teenager",
        "mother + elder + 2 teenagers",
    ],
}

OUTPUT_COLUMNS = [
    "family_id",
    "member_id",
    "conflict_level",
    "family_structure",
    "departure_city",
    "destination_city",
    "day",
    "family_budget",
    "query",
    "role",
    "interest_preferences",
    "physical_constraint",
    "budget_sensitivity",
    "dietary_preference",
    "travel_pace",
    "must_include",
    "must_avoid",
    "max_walking_distance_km",
    "latest_return_time",
]


load_dotenv()

# 生成 persona 用 OpenRouter（与主程序生成端一致，OpenAI 兼容接口）。
_API_KEY = os.getenv("OPENROUTER_API_KEY")
_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv(
    "OPENROUTER_GENERATION_MODEL", os.getenv("OPENROUTER_MODEL_NAME", "openai/gpt-4o-mini"),
)

client = None
if USE_API:
    if not _API_KEY:
        raise ValueError("USE_API=True 但 .env 里没有 OPENROUTER_API_KEY。")
    client = OpenAI(api_key=_API_KEY, base_url=_BASE_URL)


# --------------------------------------------------------------------------- #
# 分配：冲突强度 / 家庭规模 / 家庭结构
# --------------------------------------------------------------------------- #
def choose_conflict_level(family_id):
    """按 family_id 循环取冲突档位；默认 1:1:1 均衡（见 CONFLICT_LEVEL_PATTERN）。"""
    return CONFLICT_LEVEL_PATTERN[(family_id - 1) % len(CONFLICT_LEVEL_PATTERN)]


def choose_family_size(per_person, day, preferred):
    """按人均预算选家庭规模（全家预算 = 人均 × 规模）。

    尊重 preferred（3/4 交替以保持规模多样），装不下再降一档；都装不下返回 0。
    4 口要求 10% 安全余量，3 口按主程序同一容差（BUDGET_PASS_TOLERANCE）。"""
    for size in [4, 3] if preferred == 4 else [3, 4]:
        tol = 1.0 if size == 4 else BUDGET_PASS_TOLERANCE
        if is_feasible(per_person * size, day, size, tolerance=tol):
            return size
    return 0


def _structure_pool(size, conflict_level):
    """High Conflict 优先选含 elder 的结构（代际差异最大）。"""
    pool = FAMILY_STRUCTURES_BY_SIZE[size]
    if conflict_level == "High Conflict":
        elder_pool = [s for s in pool if "elder" in s]
        if elder_pool:
            return elder_pool
    return pool


def choose_family_structure(size, conflict_level, counter):
    pool = _structure_pool(size, conflict_level)
    return pool[counter % len(pool)]


def parse_structure_roles(structure):
    """'father + mother + 2 children' -> ['father','mother','child','child']。"""
    roles = []
    for token in safe_text(structure).split("+"):
        token = token.strip().lower()
        if not token:
            continue
        count = 1
        parts = token.split()
        if parts and parts[0].isdigit():
            count = int(parts[0])
            token = " ".join(parts[1:])
        role = {
            "children": "child",
            "child": "child",
            "teenagers": "teenager",
            "teenager": "teenager",
            "elders": "elder",
            "elder": "elder",
            "father": "father",
            "mother": "mother",
        }.get(token, token)
        roles.extend([role] * count)
    return roles


# --------------------------------------------------------------------------- #
# 硬约束清洗
# --------------------------------------------------------------------------- #
def clamp_walking(value, physical_constraint):
    v = safe_float(value, None)
    if v is None or v <= 0:
        v = {"low_intensity": 2.5, "high_intensity": 6.0}.get(
            safe_text(physical_constraint).lower(), 4.0
        )
    return round(min(max(v, WALK_MIN_KM), WALK_MAX_KM), 1)


def clamp_return_time(value, age_stage):
    minutes = time_to_minutes(value)
    if minutes is None:
        minutes = 20 * 60 if age_stage in ("child", "elderly") else 21 * 60 + 30
    minutes = min(max(minutes, RETURN_MIN_MIN), RETURN_MAX_MIN)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _age_stage(role):
    r = safe_text(role).lower()
    if "teen" in r:
        return "teenager"
    if "child" in r or "kid" in r:
        return "child"
    if "elder" in r or "grand" in r:
        return "elderly"
    return "adult"


# --------------------------------------------------------------------------- #
# 质量门控：判断生成的家庭是否合格
# --------------------------------------------------------------------------- #
ALLOWED_INTERESTS = {
    "culture",
    "nature",
    "parent-child",
    "shopping",
    "food",
    "entertainment",
    "social-media-check-in",
}
ALLOWED_PHYSICAL = {"low_intensity", "medium_intensity", "high_intensity"}
ALLOWED_BUDGET = {"strict_budget", "moderate_budget", "experience_first"}
ALLOWED_DIET = {"local_food", "light_food", "child_friendly", "price_sensitive"}
ALLOWED_PACE = {"compact", "moderate", "relaxed"}

# 冲突强度合格区间（基于成员差异度 conflict_index ∈ [0,1]）
# 基于结构化主导的 conflict_index 标定（带重叠，容忍模型偶尔越界，降低重试率）。
LOW_CONFLICT_MAX = 0.22
HIGH_CONFLICT_MIN = 0.48
MEDIUM_CONFLICT_RANGE = (0.15, 0.60)

# 每个成员的硬约束条数由家庭冲突程度决定：冲突越高，非妥协项越多。
# include = must_include 目标条数（上限）；avoid = must_avoid 目标条数（上限，下限恒为 1）。
# must_avoid 条数保持稀疏（Medium 1 / High 2）。曾试过加密到 2/3 以扩大冲突面，但那会在每人
# 多出的 must_avoid 槽上制造额外“有机”must_include×must_avoid 冲突，叠加注入的 clash 后，一个
# 家庭的直接硬冲突数超过了现实性上限（≤2 个 split）能化解的量 → 连 FULL 也化解不完、被判 No
# Consensus。判别力靠“注入的 clash（High=2，正好 2 个 split 可解）”承担即可，不需要靠加密度。
CONSTRAINT_COUNTS = {
    "Low Conflict": {"include": 1, "avoid": 1},
    "Medium Conflict": {"include": 2, "avoid": 1},
    "High Conflict": {"include": 3, "avoid": 2},
}


def normalize_match_role(role):
    """把模型可能用的同义角色名归一，用于和 family_structure 比对。"""
    r = safe_text(role).lower()
    if "teen" in r:
        return "teenager"
    if "grand" in r or "elder" in r or "old" in r:
        return "elder"
    if "child" in r or "kid" in r or "son" in r or "daughter" in r or "younger" in r:
        return "child"
    if "father" in r or "dad" in r:
        return "father"
    if "mother" in r or "mom" in r:
        return "mother"
    return r


def _valid_subset(items, allowed):
    return {safe_text(x).lower() for x in (items or [])} & allowed


def conflict_index(personas):
    """成员差异度 ∈ [0,1]。

    以**结构化维度**（体力/节奏/预算的取值分散度）为主（权重 0.8），兴趣不重合度为辅
    （权重 0.2）。原因：家庭成员的兴趣天然各异、几乎总是接近满分，会让指标饱和在 0.9+
    而无法区分 Medium / High；真正决定冲突档位的是体力/节奏/预算/底线这些结构化分歧。
    """
    n = len(personas)
    if n <= 1:
        return 0.0

    def spread(values):
        distinct = len({v for v in values if v})
        return (distinct - 1) / (min(n, 3) - 1) if min(n, 3) > 1 else 0.0

    phys = spread([safe_text(p.get("physical_constraint")).lower() for p in personas])
    pace = spread([safe_text(p.get("travel_pace")).lower() for p in personas])
    budget = spread([safe_text(p.get("budget_sensitivity")).lower() for p in personas])
    structural = (phys + pace + budget) / 3

    interest_sets = [
        {safe_text(x).lower() for x in (p.get("interest_preferences") or [])} for p in personas
    ]
    pairs, overlap_sum = 0, 0.0
    for a in range(n):
        for b in range(a + 1, n):
            union = interest_sets[a] | interest_sets[b]
            inter = interest_sets[a] & interest_sets[b]
            overlap_sum += (len(inter) / len(union)) if union else 1.0
            pairs += 1
    interest_conflict = 1.0 - (overlap_sum / pairs) if pairs else 0.0

    return round(0.8 * structural + 0.2 * interest_conflict, 4)


def _mode(values):
    """返回出现最多的非空值。"""
    from collections import Counter

    vals = [v for v in values if v]
    return Counter(vals).most_common(1)[0][0] if vals else None


def repair_conflict_level(personas, conflict_level):
    """**确定性地把结构化维度（体力/节奏/预算）规整到目标冲突档位**。

    gpt-4o-mini 难以可靠产出"中等"结构分歧（Medium 常被它做成接近 High），靠重试代价高。
    改为：LLM 负责生成兴趣/底线/硬约束等内容，本函数只规整体力/节奏/预算这几维，
    保证冲突强度恰好落在分配档位，从而消除冲突导致的反复重试。
    """
    if not personas:
        return
    phys = [safe_text(p.get("physical_constraint")).lower() for p in personas]
    pace = [safe_text(p.get("travel_pace")).lower() for p in personas]
    budg = [safe_text(p.get("budget_sensitivity")).lower() for p in personas]

    def force_uniform(key, values, default):
        v = _mode(values) or default
        for p in personas:
            p[key] = v

    def ensure_diverse(key, options):
        if len({safe_text(p.get(key)).lower() for p in personas if safe_text(p.get(key))}) < 2:
            cur = safe_text(personas[0].get(key)).lower() or options[0]
            personas[-1][key] = next(x for x in options if x != cur)

    if conflict_level == "Low Conflict":
        # 全部统一：体力/节奏/预算各取众数
        force_uniform("physical_constraint", phys, "medium_intensity")
        force_uniform("travel_pace", pace, "moderate")
        force_uniform("budget_sensitivity", budg, "moderate_budget")
    elif conflict_level == "Medium Conflict":
        # 同体力、同预算；只在节奏上保留/制造分歧
        force_uniform("physical_constraint", phys, "medium_intensity")
        force_uniform("budget_sensitivity", budg, "moderate_budget")
        ensure_diverse("travel_pace", ["moderate", "relaxed", "compact"])
    else:  # High Conflict：体力/节奏/预算各保证至少两种取值
        ensure_diverse(
            "physical_constraint", ["low_intensity", "high_intensity", "medium_intensity"]
        )
        ensure_diverse("travel_pace", ["compact", "relaxed", "moderate"])
        ensure_diverse(
            "budget_sensitivity", ["strict_budget", "experience_first", "moderate_budget"]
        )


def evaluate_family_quality(data, family_structure, conflict_level):
    """返回 (是否合格, 不合格原因列表)。前提：已通过结构校验 validate_personas。"""
    reasons = []
    personas = data.get("personas", [])

    # 1. 角色与家庭结构匹配（数量 + 角色多重集合一致）
    expected = sorted(parse_structure_roles(family_structure))
    actual = sorted(normalize_match_role(p.get("role")) for p in personas)
    if expected != actual:
        reasons.append(f"role_mismatch: expected {expected}, got {actual}")

    # 2. 枚举字段合法、偏好列表非空
    for p in personas:
        if safe_text(p.get("physical_constraint")).lower() not in ALLOWED_PHYSICAL:
            reasons.append(f"bad physical_constraint={p.get('physical_constraint')}")
        if safe_text(p.get("budget_sensitivity")).lower() not in ALLOWED_BUDGET:
            reasons.append(f"bad budget_sensitivity={p.get('budget_sensitivity')}")
        if safe_text(p.get("travel_pace")).lower() not in ALLOWED_PACE:
            reasons.append(f"bad travel_pace={p.get('travel_pace')}")
        if not _valid_subset(p.get("interest_preferences"), ALLOWED_INTERESTS):
            reasons.append("empty/invalid interest_preferences")
        if not _valid_subset(p.get("dietary_preference"), ALLOWED_DIET):
            reasons.append("empty/invalid dietary_preference")
        hc = p.get("hard_constraints", {}) or {}
        if safe_float(hc.get("max_walking_distance_km"), None) is None:
            reasons.append("bad max_walking_distance_km")
        if time_to_minutes(hc.get("latest_return_time")) is None:
            reasons.append("bad latest_return_time")
        inc_max = CONSTRAINT_COUNTS.get(conflict_level, {}).get("include", 3)
        av_max = CONSTRAINT_COUNTS.get(conflict_level, {}).get("avoid", 2)
        inc_items = [x for x in (hc.get("must_include") or []) if safe_text(x)]
        av_items = [x for x in (hc.get("must_avoid") or []) if safe_text(x)]
        if len(inc_items) > inc_max:
            reasons.append(f"too many must_include ({len(inc_items)}>{inc_max})")
        if not (1 <= len(av_items) <= av_max):
            reasons.append(f"must_avoid count {len(av_items)} not in 1..{av_max}")

    # 3. 实际冲突强度落在分配档位
    ci = conflict_index(personas)
    if conflict_level == "Low Conflict" and ci > LOW_CONFLICT_MAX:
        reasons.append(f"conflict too high for Low (index={ci})")
    elif conflict_level == "High Conflict" and ci < HIGH_CONFLICT_MIN:
        reasons.append(f"conflict too low for High (index={ci})")
    elif conflict_level == "Medium Conflict" and not (
        MEDIUM_CONFLICT_RANGE[0] <= ci <= MEDIUM_CONFLICT_RANGE[1]
    ):
        reasons.append(f"conflict out of Medium band (index={ci})")

    return (len(reasons) == 0), reasons


# --------------------------------------------------------------------------- #
# LLM 生成
# --------------------------------------------------------------------------- #
def conflict_guidance(conflict_level):
    """给模型按档位的硬性差异度规则，确保实际冲突落在分配档位（减少重试）。"""
    if conflict_level == "Low Conflict":
        return (
            "LOW CONFLICT — members must be SIMILAR:\n"
            "- ALL members share the SAME travel_pace.\n"
            "- ALL members share the SAME physical_constraint.\n"
            "- ALL members share the SAME budget_sensitivity.\n"
            "- Every member shares at least ONE common interest with the others.\n"
        )
    if conflict_level == "High Conflict":
        return (
            "HIGH CONFLICT — members must clearly CLASH:\n"
            "- Use at least 2 different travel_pace values across members.\n"
            "- Use at least 2 different physical_constraint values (mix low_intensity and high_intensity).\n"
            "- Use at least 2 different budget_sensitivity values (e.g. strict_budget vs experience_first).\n"
            "- Interests should barely overlap (elder=culture/nature, child=entertainment,\n"
            "  teenager=social-media-check-in/shopping, parent=budget/route focus).\n"
            "- DIRECT clashes (REQUIRED): make at least TWO cases where one member's must_include\n"
            "  category appears VERBATIM in another member's must_avoid — e.g. a child/teenager\n"
            "  must_include 'crowded theme park' while an elder must_avoid 'crowded theme park'.\n"
            "  At least one clash must be a physically demanding attraction a low_intensity/elder\n"
            "  member avoids. A single shared itinerary CANNOT satisfy both sides — it forces a\n"
            "  split activity or a negotiated concession.\n"
        )
    return (
        "MEDIUM CONFLICT — partial clash, still manageable:\n"
        "- ALL members share the SAME physical_constraint (no intensity clash).\n"
        "- ALL members share the SAME budget_sensitivity (budgets are compatible).\n"
        "- Members DIFFER in travel_pace (use at least 2 different values) and in interests.\n"
        "  This is the ONLY structural divergence allowed — do NOT also vary physical or budget.\n"
        "- DIRECT clash (REQUIRED): make at least ONE case where one member's must_include category\n"
        "  appears VERBATIM in another member's must_avoid (e.g. teenager must_include 'late-night\n"
        "  entertainment district', child/parent must_avoid 'late-night entertainment district').\n"
        "  This direct conflict cannot be satisfied by one shared plan — it needs a split or a concession.\n"
    )


# --------------------------------------------------------------------------- #
# 硬冲突注入：让 Medium/High 家庭出现"必去×必避"直接冲突，逼出分流/协商
# --------------------------------------------------------------------------- #
# 原生成的"冲突"仅是结构化分歧（体力/节奏/预算不同）+ 兴趣各异——彼此可调和，一个并集
# 行程就能满足所有人，于是分流/协商并非必需、消融也测不出退化。这里确定性地注入"某成员的
# must_include 恰是另一成员的 must_avoid"（同一通用类目、字符串完全一致，保证 Module 3 判为
# must_include_vs_must_avoid_term_conflict = split-required）——一个共享方案无法同时满足，
# 必须靠分流(一人陪去、其余做替代)或协商让步才能化解。用环境变量可关闭/调数量。
INJECT_HARD_CONFLICTS = os.getenv("INJECT_HARD_CONFLICTS", "true").lower() in (
    "1",
    "true",
    "yes",
    "y",
)
HARD_CLASH_COUNTS = {
    "Low Conflict": int(os.getenv("HARD_CLASH_LOW", "0")),
    "Medium Conflict": int(os.getenv("HARD_CLASH_MEDIUM", "1")),
    "High Conflict": int(os.getenv("HARD_CLASH_HIGH", "1")),
}
# 每条冲突 = 同一通用类目，放进"活跃方"的 must_include 与"谨慎方"的 must_avoid。真实家庭常见冲突。
HARD_CLASH_TERMS = [
    "crowded theme park",
    "late-night entertainment district",
    "high-intensity mountain climb",
    "busy shopping mall",
    "loud night market",
]
_ACTIVE_CLASH_ROLES = {"child", "teenager", "father"}  # 想去高强度/热闹项目的一方
_CAUTIOUS_CLASH_ROLES = {"elder", "mother"}  # 想避开的一方（老人/顾家/低强度）

# 可选后手（默认关闭）：把每条注入冲突「多维化」——给**同一对**(活跃/谨慎)成员再注入一个
# **不同**的争议词（活跃方 must_include[1] = 谨慎方 must_avoid[1]），使一条冲突变成两条
# must_include×must_avoid 直接冲突。未化解时谨慎方 must_avoid 违规数翻倍 → module6 惩罚更重、
# no_split/no_negotiation 差距更大；分流/协商化解时两词都走 split 豁免 → **full 模型仍为 0 违规、
# 基线不动**（与 walk/return 这类行程级指标不同，不会误伤 full）。只替换已存在的 slot1，不改
# 每人 include/avoid 条数（上一步已把 Medium avoid→2、High avoid→3，槽位够）。CLASH_MULTIDIM=1 开启。
CLASH_MULTIDIM = os.getenv("CLASH_MULTIDIM", "false").lower() in ("1", "true", "yes", "y")


def _set_slot(items, idx, value):
    """把 items[idx] 设为 value；不足则补到该长度（不超过已校验的条数上限时安全）。"""
    while len(items) <= idx:
        items.append(value)
    items[idx] = value


def _clash_ensure_hc(persona):
    hc = persona.get("hard_constraints")
    if not isinstance(hc, dict):
        hc = {}
        persona["hard_constraints"] = hc
    if not isinstance(hc.get("must_include"), list):
        hc["must_include"] = []
    if not isinstance(hc.get("must_avoid"), list):
        hc["must_avoid"] = []
    return hc


def inject_cross_member_conflicts(personas, conflict_level):
    """确定性注入 must_include×must_avoid 直接冲突。替换首项以保持每人条数不变（不破坏
    质量门控已校验的 include/avoid 计数）。仅对 Medium/High 生效。"""
    if not INJECT_HARD_CONFLICTS:
        return 0
    n_clashes = HARD_CLASH_COUNTS.get(conflict_level, 0)
    if n_clashes <= 0 or len(personas) < 2:
        return 0
    roles = [normalize_match_role(p.get("role", "")) for p in personas]
    active_idx = [i for i, r in enumerate(roles) if r in _ACTIVE_CLASH_ROLES]
    cautious_idx = [i for i, r in enumerate(roles) if r in _CAUTIOUS_CLASH_ROLES]
    if not active_idx:  # 兜底：用高强度体质当"活跃方"
        active_idx = [
            i
            for i, p in enumerate(personas)
            if safe_text(p.get("physical_constraint")).lower() == "high_intensity"
        ] or [0]
    if not cautious_idx:  # 兜底：用低强度体质当"谨慎方"
        cautious_idx = [
            i
            for i, p in enumerate(personas)
            if safe_text(p.get("physical_constraint")).lower() == "low_intensity"
        ] or [len(personas) - 1]
    used, made = set(), 0
    for k in range(n_clashes):
        term = HARD_CLASH_TERMS[k % len(HARD_CLASH_TERMS)]
        a = next((i for i in active_idx if i not in used), active_idx[0])
        b = next(
            (i for i in cautious_idx if i not in used and i != a),
            next((i for i in range(len(personas)) if i != a), None),
        )
        if b is None:
            break
        used.add(a)
        used.add(b)
        hca = _clash_ensure_hc(personas[a])
        hcb = _clash_ensure_hc(personas[b])
        if hca["must_include"]:
            hca["must_include"][0] = term
        else:
            hca["must_include"].append(term)
        if hcb["must_avoid"]:
            hcb["must_avoid"][0] = term
        else:
            hcb["must_avoid"].append(term)
        if CLASH_MULTIDIM:
            # 同一对再加一个不同的争议词（活跃 must_include[1] = 谨慎 must_avoid[1]），
            # 让这条冲突变两条；仅当两侧都有 slot1（Medium/High 的 include/avoid≥2）时生效。
            term2 = HARD_CLASH_TERMS[(k + 2) % len(HARD_CLASH_TERMS)]
            if term2 != term and len(hca["must_include"]) >= 2 and len(hcb["must_avoid"]) >= 2:
                _set_slot(hca["must_include"], 1, term2)
                _set_slot(hcb["must_avoid"], 1, term2)
        made += 1
    return made


def build_prompt(row, family_id, conflict_level, family_structure):
    roles = parse_structure_roles(family_structure)
    counts = CONSTRAINT_COUNTS.get(conflict_level, {"include": 2, "avoid": 1})
    inc, av = counts["include"], counts["avoid"]
    budget = (
        safe_float(row.get("family_budget"), 0.0)
        or safe_float(row.get("budget"), 0.0)
        or parse_budget_from_query(row.get("query"))
    )
    return f"""
You are generating synthetic family member personas for a family travel planning dataset.

Each input query represents ONE family trip. Create exactly {len(roles)} personas,
one per family member, matching this family_structure.

Input:
family_id: {family_id}
conflict_level: {conflict_level}
family_structure: {family_structure}
required roles (in order): {roles}
departure_city: {safe_text(row.get("departure_city"))}
destination_city: {safe_text(row.get("destination_city"))}
day: {safe_text(row.get("day"))}
whole-family total budget (CNY, for ALL members combined, NOT per person): {int(budget) if budget else "unspecified"}
query: {safe_text(row.get("query"))}

Conflict level requirement (MUST follow exactly):
{conflict_guidance(conflict_level)}
The generated personas MUST reflect conflict_level={conflict_level} and follow the required roles.

Each persona must include:
- role: one of the required roles above.
- interest_preferences: subset of [culture, nature, parent-child, shopping, food, entertainment, social-media-check-in]
- physical_constraint: exactly one of [low_intensity, medium_intensity, high_intensity]
- budget_sensitivity: exactly one of [strict_budget, moderate_budget, experience_first]
- dietary_preference: subset of [local_food, light_food, child_friendly, price_sensitive]
- travel_pace: exactly one of [compact, moderate, relaxed]
- hard_constraints:
    - must_include: exactly {inc} GENERAL place/experience type(s) this member wants — use
      generic categories, NEVER specific named attractions. Good: "history museum",
      "nature park", "local food street", "kid-friendly attraction", "shopping street",
      "scenic viewpoint". Bad: "West Lake", "Hong Kong Disneyland" (count fixed at {inc}).
    - must_avoid: exactly {av} EXPERIENCE-ATTRIBUTE thing(s) this member wants to avoid — describe a
      DISCOMFORT (intensity / pace / cost / noise / food), NOT an attraction TYPE. Never reuse an
      attraction category that another member might want to include (do NOT write "history museum",
      "shopping mall", "nature park" as an avoid — those collide with others' must_include and are
      handled separately). Examples (count fixed at {av}):
        low_intensity / elder -> "long-distance hiking", "steep stair climbs"
        strict_budget -> "overpriced tourist traps", "expensive premium tickets"
        child / parent-child -> "late-night activities", "loud crowded venues"
        light_food -> "very spicy food", "greasy barbecue"
        teenager / social-media-check-in -> "slow-paced relaxed sightseeing"
    - max_walking_distance_km: number between {WALK_MIN_KM} and {WALK_MAX_KM}
    - latest_return_time: "HH:MM" between 18:30 and 22:00

Rules:
- Return exactly {len(roles)} personas; make members differ from each other.
- Personas must fit the destination, trip length and budget in the query.
- Each persona lists exactly {inc} must_include and {av} must_avoid item(s) — no more, no fewer.
- Output valid JSON only. No markdown, no explanation.

Output format:
{{
  "family_id": {family_id},
  "conflict_level": "{conflict_level}",
  "family_structure": "{family_structure}",
  "personas": [
    {{
      "member_id": 1,
      "role": "...",
      "interest_preferences": ["..."],
      "physical_constraint": "...",
      "budget_sensitivity": "...",
      "dietary_preference": ["..."],
      "travel_pace": "...",
      "hard_constraints": {{
        "must_include": ["..."],
        "must_avoid": ["..."],
        "max_walking_distance_km": 3.0,
        "latest_return_time": "20:00"
      }}
    }}
  ]
}}
""".strip()


def validate_personas(data, expected_count):
    if not isinstance(data, dict):
        raise ValueError("返回结果不是 dict")
    personas = data.get("personas")
    if not isinstance(personas, list):
        raise ValueError("personas 不是 list")
    if not 3 <= len(personas) <= 4:
        raise ValueError(f"persona 数量应为 3–4，实际 {len(personas)}")

    required = [
        "role",
        "interest_preferences",
        "physical_constraint",
        "budget_sensitivity",
        "dietary_preference",
        "travel_pace",
        "hard_constraints",
    ]
    for p in personas:
        for f in required:
            if f not in p:
                raise ValueError(f"persona 缺少字段: {f}")
        hc = p["hard_constraints"]
        if not isinstance(hc, dict):
            raise ValueError("hard_constraints 不是 dict")
        for f in ["must_include", "must_avoid", "max_walking_distance_km", "latest_return_time"]:
            if f not in hc:
                raise ValueError(f"hard_constraints 缺少字段: {f}")
    return True


def generate_once_via_api(row, family_id, conflict_level, family_structure):
    """单次调用 OpenRouter，返回结构合法的 dict；调用/解析失败返回 None。"""
    prompt = build_prompt(row, family_id, conflict_level, family_structure)
    expected = len(parse_structure_roles(family_structure))
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a data generation assistant. Output valid JSON only. "
                        "No markdown, no explanations."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,  # 稍高温度让每次重生成有差异
            max_tokens=1800,
            response_format={"type": "json_object"},
        )
        data = extract_json(response.choices[0].message.content)
        data["family_id"] = family_id
        data["conflict_level"] = conflict_level
        data["family_structure"] = family_structure
        validate_personas(data, expected)
        return data
    except Exception as e:  # noqa: BLE001
        print(f"family_id={family_id} 生成调用失败: {e}")
        return None


# --------------------------------------------------------------------------- #
# 确定性规则生成（API 兜底 / 离线自检）
# --------------------------------------------------------------------------- #
_ROLE_TEMPLATE = {
    "father": (
        ["culture", "food"],
        "medium_intensity",
        "strict_budget",
        ["local_food"],
        "moderate",
    ),
    "mother": (
        ["food", "parent-child"],
        "medium_intensity",
        "moderate_budget",
        ["light_food", "child_friendly"],
        "moderate",
    ),
    "child": (
        ["parent-child", "entertainment"],
        "low_intensity",
        "moderate_budget",
        ["child_friendly"],
        "relaxed",
    ),
    "teenager": (
        ["social-media-check-in", "shopping"],
        "high_intensity",
        "experience_first",
        ["local_food"],
        "compact",
    ),
    "elder": (["culture", "nature"], "low_intensity", "strict_budget", ["light_food"], "relaxed"),
}


def _apply_conflict(interests, physical, budget, pace, conflict_level):
    """按冲突档位调节单个成员属性，使整族差异度落入对应区间。

    Low：体力/节奏/预算全部收敛 + 兴趣并入共同项 food（差异度趋近 0）；
    Medium：仅弱化极端体力/节奏，保留预算与兴趣差异；
    High：保留角色模板的强差异（High 家庭必含 elder，差异天然拉满）。
    """
    interests = list(interests)
    if conflict_level == "Low Conflict":
        physical, pace, budget = "medium_intensity", "moderate", "moderate_budget"
        interests = list(dict.fromkeys(["food"] + interests))[:2]
    elif conflict_level == "Medium Conflict":
        # 同体力、同预算，只在节奏与兴趣上分歧（保留模板的 pace 差异）。
        physical, budget = "medium_intensity", "moderate_budget"
    # High Conflict：保留角色模板的强差异（High 家庭必含 elder）。
    return interests, physical, budget, pace


_INTEREST_TO_CATEGORY = {
    "culture": "cultural landmark",
    "nature": "nature park",
    "parent-child": "kid-friendly attraction",
    "shopping": "shopping street",
    "food": "local food street",
    "entertainment": "entertainment venue",
    "social-media-check-in": "scenic viewpoint",
}


def _fallback_must_include(interests, inc_count):
    """按成员兴趣映射为**通用类别**（非具体 POI），取前 inc_count 条。"""
    cats = [_INTEREST_TO_CATEGORY[i] for i in interests if i in _INTEREST_TO_CATEGORY]
    if not cats:
        cats = ["local food street"]
    return list(dict.fromkeys(cats))[: max(inc_count, 1)]


def _fallback_must_avoid(role, physical, budget, diet):
    """按角色/约束确定性地推出 1-2 条 must_avoid，保证兜底族也有底线约束。"""
    avoid = []
    if physical == "low_intensity":
        avoid.append("long-distance hiking")
    if role == "elder":
        avoid.append("stair-heavy climbs")
    if role == "child":
        avoid.append("late-night nightlife")
    if role == "teenager":
        avoid.append("slow-paced relaxed sightseeing")
    if budget == "strict_budget":
        avoid.append("overpriced tourist traps")
    if "light_food" in diet:
        avoid.append("very spicy food")
    if not avoid:
        avoid.append("overcrowded tourist traps")
    return list(dict.fromkeys(avoid))[:2]


def build_fallback_personas(row, family_id, conflict_level, family_structure):
    roles = parse_structure_roles(family_structure)
    counts = CONSTRAINT_COUNTS.get(conflict_level, {"include": 2, "avoid": 1})
    inc_count, av_count = counts["include"], counts["avoid"]
    personas = []
    for i, role in enumerate(roles, start=1):
        interests, physical, budget, diet, pace = _ROLE_TEMPLATE.get(role, _ROLE_TEMPLATE["father"])
        interests, physical, budget, pace = _apply_conflict(
            interests, physical, budget, pace, conflict_level
        )
        personas.append(
            {
                "member_id": i,
                "role": role,
                "interest_preferences": interests,
                "physical_constraint": physical,
                "budget_sensitivity": budget,
                "dietary_preference": list(diet),
                "travel_pace": pace,
                "hard_constraints": {
                    "must_include": _fallback_must_include(interests, inc_count),
                    "must_avoid": _fallback_must_avoid(role, physical, budget, list(diet))[
                        :av_count
                    ],
                    "max_walking_distance_km": {"low_intensity": 2.5, "high_intensity": 6.0}.get(
                        physical, 4.0
                    ),
                    "latest_return_time": "20:00" if role in ("child", "elder") else "21:30",
                },
            }
        )
    return {
        "family_id": family_id,
        "conflict_level": conflict_level,
        "family_structure": family_structure,
        "personas": personas,
    }


# --------------------------------------------------------------------------- #
# 展开为「一行一个成员」
# --------------------------------------------------------------------------- #
def flatten_family_personas(data, row):
    rows = []
    for i, persona in enumerate(data["personas"], start=1):
        hc = persona.get("hard_constraints", {}) or {}
        role = safe_text(persona.get("role", ""))
        physical = safe_text(persona.get("physical_constraint", "medium_intensity"))
        rows.append(
            {
                "family_id": data["family_id"],
                "member_id": persona.get("member_id", i),
                "conflict_level": data["conflict_level"],
                "family_structure": data["family_structure"],
                "departure_city": row.get("departure_city", ""),
                "destination_city": row.get("destination_city", ""),
                "day": row.get("day", ""),
                "family_budget": row.get("family_budget", ""),
                "query": row.get("query", ""),
                "role": role,
                "interest_preferences": json.dumps(
                    persona.get("interest_preferences", []), ensure_ascii=False
                ),
                "physical_constraint": physical,
                "budget_sensitivity": safe_text(persona.get("budget_sensitivity", "")),
                "dietary_preference": json.dumps(
                    persona.get("dietary_preference", []), ensure_ascii=False
                ),
                "travel_pace": safe_text(persona.get("travel_pace", "")),
                "must_include": json.dumps(hc.get("must_include", []), ensure_ascii=False),
                "must_avoid": json.dumps(hc.get("must_avoid", []), ensure_ascii=False),
                "max_walking_distance_km": clamp_walking(
                    hc.get("max_walking_distance_km"), physical
                ),
                "latest_return_time": clamp_return_time(
                    hc.get("latest_return_time"), _age_stage(role)
                ),
            }
        )
    return rows


def process_one_query(assignment):
    """assignment: (row_dict, family_id, conflict_level, family_structure)。

    生成 + 质量门控：不合格就重生成，直到合格或到达 MAX_QUALITY_ATTEMPTS；
    到上限仍不合格则用规则兜底（确定性、必定合格）。
    """
    row, family_id, conflict_level, family_structure = assignment
    print(f"生成 family_id={family_id}, conflict={conflict_level}, " f"structure={family_structure}")

    data = None
    if USE_API:
        attempt = 0
        while True:
            attempt += 1
            candidate = generate_once_via_api(row, family_id, conflict_level, family_structure)
            if candidate is not None:
                # 先把结构化维度规整到目标档位，再判定（消除冲突导致的重试）。
                repair_conflict_level(candidate.get("personas", []), conflict_level)
                ok, reasons = evaluate_family_quality(candidate, family_structure, conflict_level)
                if ok:
                    if attempt > 1:
                        print(f"family_id={family_id} 第 {attempt} 次生成合格。")
                    data = candidate
                    break
                print(f"family_id={family_id} 第 {attempt} 次不合格: {reasons}")
            if 0 < MAX_QUALITY_ATTEMPTS <= attempt:
                print(f"family_id={family_id} 连续 {attempt} 次未达标，改用规则兜底。")
                break
            time.sleep(1)

    if data is None:
        data = build_fallback_personas(row, family_id, conflict_level, family_structure)
    # 质量门控通过后再注入硬冲突（替换首项、不改条数），保证 Medium/High 家庭有真正需要
    # 分流/协商才能化解的 must_include×must_avoid 直接冲突。
    made = inject_cross_member_conflicts(data.get("personas", []), conflict_level)
    if made:
        print(f"family_id={family_id} 注入 {made} 条 must_include×must_avoid 硬冲突（{conflict_level}）")
    return family_id, flatten_family_personas(data, row)


def build_assignments(df):
    """顺序、确定性地为每个 query 分配冲突强度 / 家庭规模 / 结构。"""
    assignments = []
    size_counter = {3: 0, 4: 0}
    dropped = 0
    rewrite_fail = 0
    for i, row in df.iterrows():
        family_id = i + 1
        # query 预算是**人均**；全家总预算 = 人均 × 家庭规模。
        per_person = (
            safe_float(row.get("budget_per_person"), 0.0)
            or safe_float(row.get("budget"), 0.0)
            or parse_budget_from_query(row.get("query"))
        )
        day = row.get("day", 1)
        preferred = 4 if family_id % 2 == 0 else 3  # 3/4 口交替，保持规模多样
        size = choose_family_size(per_person, day, preferred)
        if size == 0:
            # filter 已挡掉这类，这里再兜一层底，避免生成不可行家庭。
            dropped += 1
            continue
        family_budget = int(round(per_person * size))
        conflict_level = choose_conflict_level(family_id)
        family_structure = choose_family_structure(size, conflict_level, size_counter[size])
        size_counter[size] += 1

        rd = row.to_dict()
        rd["family_budget"] = family_budget
        # 把 query 文本里的预算金额改写成全家总预算，让主程序解析到同一口径。
        new_query, ok = rewrite_query_budget(rd.get("query"), family_budget)
        rd["query"] = new_query
        if not ok:
            rewrite_fail += 1
        assignments.append((rd, family_id, conflict_level, family_structure))
    if dropped:
        print(f"跳过 {dropped} 条预算装不下 3 口之家的 query。")
    if rewrite_fail:
        print(f"提醒：{rewrite_fail} 条 query 未定位到预算金额、未改写。")
    print(f"家庭规模分布：3 口 {size_counter[3]} 个，4 口 {size_counter[4]} 个。")
    return assignments


def save_rows(rows, output_path):
    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if not out_df.empty:
        out_df = out_df.sort_values(by=["family_id", "member_id"])
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return out_df


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"找不到输入文件: {INPUT_CSV}")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    rename = {}
    if "departure" in df.columns and "departure_city" not in df.columns:
        rename["departure"] = "departure_city"
    if "destination" in df.columns and "destination_city" not in df.columns:
        rename["destination"] = "destination_city"
    if rename:
        df = df.rename(columns=rename)

    for col in ["departure_city", "destination_city", "day", "query"]:
        if col not in df.columns:
            raise ValueError(f"输入 CSV 缺少字段: {col}")

    print(f"读取 query 数量: {len(df)}")
    assignments = build_assignments(df)
    print(f"实际生成家庭数: {len(assignments)}；并发 MAX_WORKERS={MAX_WORKERS}，USE_API={USE_API}")

    all_rows = []
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_one_query, a) for a in assignments]
        for future in as_completed(futures):
            try:
                family_id, rows = future.result()
                all_rows.extend(rows)
                completed += 1
                print(f"完成 family_id={family_id}，进度 {completed}/{len(assignments)}")
                if completed % SAVE_EVERY == 0:
                    save_rows(all_rows, OUTPUT_CSV)
                    print(f"已临时保存: {OUTPUT_CSV}")
            except Exception as e:  # noqa: BLE001
                print(f"某个 query 生成失败: {e}")

    final_df = save_rows(all_rows, OUTPUT_CSV)
    print("全部生成完成！")
    print(f"输出文件: {OUTPUT_CSV}")
    print(
        f"家庭数: {final_df['family_id'].nunique() if not final_df.empty else 0}，"
        f"persona 总数: {len(final_df)}"
    )


if __name__ == "__main__":
    main()
