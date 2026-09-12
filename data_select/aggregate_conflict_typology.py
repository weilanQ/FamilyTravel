# -*- coding: utf-8 -*-
"""
聚合 200 家庭冲突类型学统计 v2
支持两种存储形态: (a) 每家庭一个 json; (b) 单个汇总 json 内含 200 条家庭记录(list 或 dict)。
若无法识别字段名, 会打印第一条记录的结构预览, 把预览发回来即可精确适配。

用法: python aggregate_conflict_typology.py
输出: conflict_typology_200.md
"""
import json
import statistics
from collections import Counter
from pathlib import Path

# ======== 按你的实际目录/文件修改 ========
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "ablation" / "full"
FILE_GLOB = "**/conflict_detection_results*.json"
OUT_MD = Path("conflict_typology_200.md")
# ========================================

CONFLICT_KEYS = ["conflicts", "detected_conflicts", "conflict_list", "edges", "conflict_edges"]
TYPE_KEYS = ["conflict_type", "type", "category"]
LEVEL_KEYS = ["conflict_level", "detected_level", "overall_conflict_level", "level", "conflict_intensity"]
FAMILY_ID_KEYS = ["family_id", "familyId", "fid", "family"]

ZH_NAME = {
    "hard_constraint_feasibility_conflict": "硬约束可行性冲突",
    "attraction_type_conflict": "景点类型冲突",
    "time_conflict": "时间冲突",
    "route_distance_conflict": "路线距离冲突",
    "crowd_queue_feasibility_conflict": "拥挤排队冲突",
    "intensity_feasibility_conflict": "强度可行性冲突",
    "preference_avoidance_conflict": "偏好-规避冲突",
    "child_older_vs_late_return_conflict": "儿童/老人-晚归冲突",
    "child_older_vs_high_intensity_conflict": "儿童/老人-高强度冲突",
    "budget_conflict": "预算冲突",
    "pace_conflict": "节奏冲突",
    "rest_vs_compact_schedule_conflict": "休息-紧凑日程冲突",
    "travel_intensity_conflict": "整体旅行强度冲突",
}


def first_key(d, keys):
    if isinstance(d, dict):
        for k in keys:
            if k in d:
                return d[k]
    return None


def deep_find(d, keys, max_depth=3):
    """在嵌套 dict 中(最多 max_depth 层)找第一个匹配键。"""
    v = first_key(d, keys)
    if v is not None:
        return v
    if max_depth > 0 and isinstance(d, dict):
        for sub in d.values():
            if isinstance(sub, dict):
                v = deep_find(sub, keys, max_depth - 1)
                if v is not None:
                    return v
    return None


def preview(obj, depth=0, max_depth=3):
    pad = "  " * depth
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:12]:
            print(f"{pad}{k}: {type(v).__name__}")
            if depth < max_depth and isinstance(v, (dict, list)):
                preview(v, depth + 1, max_depth)
    elif isinstance(obj, list):
        print(f"{pad}[list, len={len(obj)}] 首元素:")
        if obj and depth < max_depth:
            preview(obj[0], depth + 1, max_depth)


def iter_family_records(files):
    """产出 (family_key, record_dict)。自动展开单文件多家庭结构。"""
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(data, list):
            # list 元素若像家庭记录(含冲突列表或 family_id) → 逐条产出
            if data and isinstance(data[0], dict) and (
                first_key(data[0], CONFLICT_KEYS) is not None
                or first_key(data[0], FAMILY_ID_KEYS) is not None
            ):
                for i, rec in enumerate(data):
                    fid = first_key(rec, FAMILY_ID_KEYS) or f"{f.name}#{i}"
                    yield fid, rec
            else:  # list 本身就是单家庭的冲突列表
                yield f.stem, {"conflicts": data}
        elif isinstance(data, dict):
            # dict 若以 family_id 为键 → 逐条产出
            vals = list(data.values())
            if vals and all(isinstance(v, dict) for v in vals) and any(
                deep_find(v, CONFLICT_KEYS) is not None for v in vals[:3]
            ) and first_key(data, CONFLICT_KEYS) is None:
                for fid, rec in data.items():
                    yield fid, rec
            else:  # 单家庭一个文件
                yield first_key(data, FAMILY_ID_KEYS) or f.stem, data


def main():
    files = sorted(RESULTS_DIR.glob(FILE_GLOB))
    if not files:
        raise SystemExit(f"未找到文件: {RESULTS_DIR / FILE_GLOB}")

    type_counter, level_counter = Counter(), Counter()
    per_family_counts = []
    unknown_sample = None

    for fid, rec in iter_family_records(files):
        conflicts = deep_find(rec, CONFLICT_KEYS) or []
        types = []
        for c in conflicts:
            t = first_key(c, TYPE_KEYS) if isinstance(c, dict) else None
            if t is None and unknown_sample is None:
                unknown_sample = c
            types.append(t if t else "<unknown_type>")
        type_counter.update(types)
        per_family_counts.append(len(types))
        level = deep_find(rec, LEVEL_KEYS)
        if level:
            level_counter[str(level)] += 1

    n_fam = len(per_family_counts)
    total = sum(per_family_counts)
    if type_counter and set(type_counter) == {"<unknown_type>"}:
        print("[!] 冲突条目里找不到类型字段。单条冲突的结构预览如下, 请把它发回来:")
        preview(unknown_sample)
        return
    if n_fam <= 2:
        print("[!] 只识别出", n_fam, "个家庭。首文件结构预览如下, 请把它发回来:")
        preview(json.loads(files[0].read_text(encoding="utf-8")))
        return

    mean = total / n_fam
    median = statistics.median(per_family_counts)
    lines = [f"# 冲突类型学统计（{n_fam} 个家庭）\n"]
    lines.append(f"- 冲突总数: {total}；平均每家庭 {mean:.2f} 个；中位数 {median:g}")
    lines.append("- 检测等级分布: " + (", ".join(f"{k} {v}" for k, v in level_counter.most_common()) or "（未找到等级字段）") + "\n")
    lines.append("| 冲突类型 | 中文名称 | 次数 |")
    lines.append("| --- | --- | --- |")
    for t, c in type_counter.most_common():
        lines.append(f"| {t} | {ZH_NAME.get(t, '（请补充中文名）')} | {c} |")
        if t not in ZH_NAME and t != "<unknown_type>":
            print(f"[note] 新冲突类型未在映射中: {t}")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"完成: {n_fam} 家庭，{total} 个冲突 -> {OUT_MD}")
    if n_fam != 200:
        print(f"[warn] 家庭数为 {n_fam}，不是 200")


if __name__ == "__main__":
    main()
