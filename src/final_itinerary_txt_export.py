"""Render the final consensus itineraries into human-readable .txt files with label cleaning and specificity checks."""
import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.common import RESULTS_DIR, estimate_day_walking_km, safe_float, safe_int, safe_text

FINAL_PLANS_CSV = RESULTS_DIR / "module_7" / "generated_plans_feedback_final.csv"
MEMBER_EVAL_CSV = RESULTS_DIR / "module_7" / "evaluation_member_level_feedback.csv"
PLAN_EVAL_CSV = RESULTS_DIR / "module_7" / "evaluation_plan_level_feedback.csv"
REPORT_JSON = RESULTS_DIR / "module_7" / "evaluation_report_feedback.json"
OUTPUT_DIR = RESULTS_DIR / "final_itinerary_txt"
COMBINED_OUTPUT_TXT = OUTPUT_DIR / "all_families_final_itinerary_input.txt"

EXPORT_ONLY_SUCCESS_PLANS = os.getenv("EXPORT_ONLY_SUCCESS_PLANS", "true").lower() not in [
    "0",
    "false",
    "no",
]
EXPORT_REQUIRE_EVALUATION_ROW = os.getenv("EXPORT_REQUIRE_EVALUATION_ROW", "true").lower() not in [
    "0",
    "false",
    "no",
]


def cn_text(value: Any) -> str:
    # Output content in its original language (English stays English). The previous
    # English->Chinese glossary used naive substring .replace(), which corrupted words such
    # as "placeholder"->"placeh老人", "following"->"fol低ing", and "too low"->"too 低", and
    # mangled English diagnostic / LLM-judge text. It is therefore disabled.
    text = safe_text(value)
    if not text:
        return ""
    return text.replace("_", " ").strip()


BAD_PLACEHOLDER_TERMS = [
    "家庭友好 dinner",
    "家庭友好 晚餐",
    "family friendly dinner",
    "family-friendly dinner",
    "dinner",
    "lunch",
    "午餐或",
    "晚餐或",
    "早餐或",
    "或中档",
    "或低档",
    "未知",
    "未注明",
    "nan",
    "none",
    "null",
]


def clean_output_text(text: Any) -> str:
    """Clean placeholder and awkward mixed-language text before writing txt output."""
    out = cn_text(text)
    if not out:
        return ""
    replacements = {
        "家庭友好 dinner": "家庭友好晚餐",
        "家庭友好 lunch": "家庭友好午餐",
        "family friendly dinner": "家庭友好晚餐",
        "family-friendly dinner": "家庭友好晚餐",
        "晚餐或中档": "",
        "午餐或中档": "",
        "早餐或低档": "",
        "晚餐或低档": "",
        "午餐或低档": "",
        "，低档": "",
        ", low": "",
    }
    for old, new in replacements.items():
        out = out.replace(old, new)
    out = re.sub(r"\s+", " ", out).strip(" ，；。")
    return out


def is_placeholder_text(value: Any) -> bool:
    text = clean_output_text(value).lower()
    if not text:
        return True
    # Standalone placeholders or broken model fragments should not be exported.
    if text in ["晚餐", "午餐", "早餐", "中档", "低档", "普通餐饮", "合适餐厅", "家庭友好晚餐"]:
        return True
    return any(term.lower() in text for term in BAD_PLACEHOLDER_TERMS)


def structural_placeholder_text(value: Any) -> bool:
    text = clean_output_text(value).lower()
    if not text or is_placeholder_text(text):
        return True
    return bool(
        re.search(
            r"(第\s*\d+\s*天.*活动区域|core\s+attraction|local\s+family\s+activity|nearest\s+(metro|mtr)|短路线|路线需求)",
            text,
        )
    )


def is_bad_area_text(value: Any) -> bool:
    text = clean_output_text(value)
    if not text or structural_placeholder_text(text):
        return True
    if re.search(r"(站|区|路|街|广场|商圈|酒店|城|园|馆|楼|塔|港|湾|市场|夜市)$", text):
        return False
    if re.fullmatch(r".{0,6}(活动|体验|游览|探索|美食|小吃|购物|表演|打卡|观景|用餐|午餐|晚餐)$", text):
        return True
    # Meal-type / price fragments (e.g. "中餐，价格高", "自助餐+中等", "享用广东菜") are not
    # real dining areas; they leaked in from the activity text and must not become "用餐范围".
    if re.search(r"(价格|人均|预算|档次)", text):
        return True
    if re.search(r"(自助餐|快餐|中餐|西餐|早餐|午餐|晚餐|夜宵|轻食|享用|品尝|游玩)", text) and not re.search(
        r"(站|区|路|街|广场|城|园|馆|楼|塔|港|湾|市场|夜市|公园|乐园|景区|大道|步行街|商场)$", text
    ):
        return True
    return False


def clean_area_label(value: Any) -> str:
    text = clean_output_text(value)
    if not text:
        return ""
    text = re.sub(r"^(参观|游览|探索|体验|享受|品尝|观看|欣赏|前往|抵达|返回|在|于)", "", text).strip()
    text = re.sub(r"(用餐|午餐|晚餐|轻食)$", "", text).strip(" 的附近周边，；。")
    text = text.replace("的美景", "").replace("短时游览", "").strip(" ，；。")
    if is_bad_area_text(text):
        return ""
    return text


def normalize_activity_label_for_export(value: Any) -> str:
    text = clean_output_text(value)
    if not text:
        return ""
    text = re.sub(r"^购物于(.+)$", r"\1购物", text).strip()
    text = re.sub(r"^(参观|游览|探索|体验|享受|品尝|观看|欣赏|前往|抵达|返回|在|于)", "", text).strip()
    text = re.sub(r"(观光|参观|游览|游玩|体验|打卡|拍照|散步|放松|校园游览|短时游览)$", "", text).strip()
    text = text.replace("的美景", "").replace("短时游览", "").strip(" ，；。")
    return text


def activity_specificity_score_for_export(value: Any) -> float:
    raw_text = clean_output_text(value)
    if re.search(r"(返回|返程|退房|入住|抵达.*酒店|离开|整理行李|早餐后退房)", raw_text):
        return 0.0
    if re.fullmatch(r"前往[\u4e00-\u9fff]{2,4}", raw_text) and not re.search(
        r"(滩|园|乐园|公园|馆|博物馆|美术馆|艺术馆|楼|塔|台|观景台|寺|庙|宫|教堂|大教堂|佛|大佛|城|古城|小镇|广场|中心|大学|街|路|巷|港|湾|岛|山|湖|江|桥|站|市场|夜市|商场|王国|景区)$",
        re.sub(r"^前往", "", raw_text),
    ):
        return 0.0
    text = normalize_activity_label_for_export(value)
    if not text or structural_placeholder_text(text):
        return 0.0
    score = 0.20
    if re.search(r"[A-Za-z0-9＋+&·•（）()]", text):
        score += 0.30
    if re.search(
        r"(滩|园|乐园|公园|馆|博物馆|美术馆|艺术馆|楼|塔|台|观景台|寺|庙|宫|教堂|大教堂|佛|大佛|城|古城|小镇|广场|中心|大学|街|路|巷|港|湾|岛|山|湖|江|桥|站|市场|夜市|商场|王国|景区)$",
        text,
    ):
        score += 0.45
    elif re.fullmatch(r"[\u4e00-\u9fff]{2,10}", text) and not re.search(
        r"(活动|体验|游览|探索|美食|小吃|购物|表演|打卡|观景|餐饮|餐厅|火锅|自助餐|返程|退房|休息)$", text
    ):
        score += 0.30
    if re.search(r"(早餐|午餐|晚餐|用餐|轻食|美食|小吃|火锅|餐厅|饭店|餐馆|料理|自助餐|餐饮|返程|退房|整理行李|休息)$", text):
        score -= 0.35
    if re.search(r".{2,}(城|街|路|商场|市场|夜市|广场|港|湾|小镇|古城|公园|乐园|馆|楼|塔)(购物|打卡|体验|表演|美食|散步|游览)?$", text):
        score += 0.45
    elif re.fullmatch(
        r"(文化|历史|主题|当地|本地|互动|观景|美食|餐饮|自由|轻松|放松|现场|游乐).{0,4}(遗址|地标|公园|乐园|活动|体验|游览|美食|购物|表演|打卡|晚餐|午餐)?",
        text,
    ):
        score -= 0.45
    elif re.fullmatch(r".{0,6}(活动|体验|游览|探索|美食|小吃|购物|表演|打卡|观景|休闲|放松)$", text):
        score -= 0.25
    return max(0.0, min(1.0, score))


def is_generic_activity_for_export(value: Any) -> bool:
    return activity_specificity_score_for_export(value) < 0.45


def is_specific_restaurant_name_for_export(value: Any) -> bool:
    name = clean_output_text(value)
    if not name or structural_placeholder_text(name):
        return False
    if re.fullmatch(r"(餐饮|用餐|午餐|晚餐|早餐|美食|小吃|自助|火锅|西餐|中餐|茶餐厅|餐厅|饭店|餐馆|料理)$", name):
        return False
    if re.search(r"[A-Za-z0-9＋+&·•（）()]", name):
        return len(name) >= 2
    if re.fullmatch(r"[一-鿿]{2,8}", name) and not re.search(
        r"(餐饮|用餐|午餐|晚餐|早餐|美食|小吃|自助|火锅|西餐|中餐|茶餐厅|餐厅|饭店|餐馆|料理)$", name
    ):
        return True
    if re.search(r"(餐厅|饭店|餐馆|酒楼|酒家|食府|茶餐厅|小馆|厨房|料理|烤肉|火锅|小吃|面馆|粉店|烧烤|海鲜|咖啡|茶楼)$", name):
        prefix = re.sub(r"(餐厅|饭店|餐馆|酒楼|酒家|食府|茶餐厅|小馆|厨房|料理|烤肉|火锅|小吃|面馆|粉店|烧烤|海鲜|咖啡|茶楼)$", "", name)
        return len(prefix.strip()) >= 2 and not re.fullmatch(
            r"(当地|本地|家庭|友好|中档|高档|附近|景区|园区|公园|乐园|主题|自助|西餐|中餐|海鲜|火锅|餐饮)+", prefix.strip()
        )
    return False


_HOTEL_GENERIC_QUALIFIERS = (
    "市中心",
    "市区",
    "城区",
    "商业区",
    "商圈",
    "火车站",
    "高铁站",
    "机场",
    "景区",
    "市内",
    "周边",
    "经济型",
    "经济",
    "快捷",
    "商务",
    "连锁",
    "高档",
    "豪华",
    "舒适",
    "家庭友好",
    "家庭",
    "星级",
    "三星",
    "四星",
    "五星",
    "精品",
    "主题",
    "度假",
    "青年",
    "中档",
    "高端",
    "平价",
    "当地",
    "本地",
    "附近",
    "优质",
    "品牌",
    "亲子",
    "温馨",
)
_HOTEL_NAME_SUFFIX_RE = re.compile(r"(国际大酒店|大酒店|度假村|酒店式公寓|公寓|酒店|宾馆|饭店|旅馆|客栈|旅店)$")


def is_generic_hotel_name_for_export(value: Any, area: Any = "", destination: Any = "") -> bool:
    name = clean_output_text(value)
    if not name or is_placeholder_text(name):
        return True
    if name in ["市中心酒店", "尖沙咀酒店", "香港尖沙咀酒店", "珠海假日酒店", "武昌区酒店", "汉阳区酒店", "江汉区酒店"]:
        return True
    area_text = clean_output_text(area)
    dest_text = clean_output_text(destination)
    for loc in (area_text, dest_text):
        if loc and name in [f"{loc}酒店", f"{loc}附近酒店", f"{loc}宾馆", f"{loc}饭店", f"{loc}市中心酒店"]:
            return True
    # Strip the hotel-type suffix and any leading area/city + generic qualifier
    # tokens; if no distinctive (brand / proper-noun) token remains, it is a
    # generic type label, not a real hotel name.  Keeps e.g. 成都帝盛酒店 / 亚朵酒店.
    core = _HOTEL_NAME_SUFFIX_RE.sub("", name)
    for loc in (area_text, dest_text):
        if loc:
            core = core.replace(loc, "")
    for q in _HOTEL_GENERIC_QUALIFIERS:
        core = core.replace(q, "")
    core = re.sub(r"新区|开发区|自治区|[省市区县]", "", core)
    return len(core.strip()) < 2


def should_show_ticket_for_activity(item: Dict[str, Any], price: float) -> bool:
    if price <= 0:
        return False
    name = (
        clean_output_text(item.get("poi_name"))
        or clean_output_text(item.get("activity"))
        or clean_output_text(item.get("poi_type"))
    )
    source = clean_output_text(item.get("ticket_price_source")) or clean_output_text(
        item.get("price_source")
    )
    urls = item.get("source_urls", [])
    has_url = isinstance(urls, list) and any(str(u).startswith("http") for u in urls)
    # The old generic 40 CNY fallback should not be displayed without evidence.
    if abs(price - 40.0) < 0.01 and not source and not has_url:
        return False
    if any(x in name for x in ["自由活动", "返回", "返程", "退房", "早餐", "午餐", "晚餐", "购物", "散步", "休息"]):
        return False
    return True


def normalize_for_dedupe(text: str) -> str:
    text = clean_output_text(text)
    text = re.sub(r"[\s，。；;、,.（）()]+", "", text)
    text = text.replace("预估", "").replace("元", "")
    return text.lower()


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        item = clean_output_text(item)
        if not item:
            continue
        key = normalize_for_dedupe(item)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def infer_city_family_restaurant(plan_row: Dict[str, Any], meal_name: str, area: str = "") -> str:
    """Create a clean generic restaurant description when the LLM output is a placeholder.

    This avoids exporting bad strings such as "家庭友好 dinner" or "早餐或低档".
    It does not invent a precise restaurant name; it uses a safe area/type description.
    """
    destination = clean_output_text(plan_row.get("destination_city"))
    area = clean_output_text(area) or clean_output_text(plan_row.get("hotel_area"))
    area_prefix = f"{area}附近" if area else ""
    if "哈尔滨" in destination or "Harbin" in destination:
        return f"{area_prefix}东北菜家庭餐厅"
    if "香港" in destination or "Hong Kong" in destination:
        return f"{area_prefix}茶餐厅或粤菜家庭餐厅"
    if "上海" in destination or "Shanghai" in destination:
        return f"{area_prefix}本帮菜或西餐家庭餐厅"
    if "深圳" in destination or "Shenzhen" in destination:
        return f"{area_prefix}粤菜或海鲜家庭餐厅"
    return f"{area_prefix}家庭友好餐厅" if area_prefix else f"家庭友好{meal_name}餐厅"


def meal_price_range_text(price_level: str = "中等") -> str:
    text = clean_output_text(price_level).lower()
    if text in ["low", "低", "低档", "低价位"]:
        return "30-80元/人"
    if text in ["high", "高", "中高", "中高端", "中高价位"]:
        return "100-350元/人"
    if text in ["luxury", "高端", "豪华", "高端价位"]:
        return "180-600元/人"
    return "50-180元/人"


_GENERIC_CUISINE_LABELS = {
    "家庭友好餐厅",
    "家庭友好餐饮",
    "当地家庭友好餐厅",
    "家庭餐厅",
    "当地餐厅",
    "当地美食餐厅",
    "普通餐饮",
    "合适餐厅",
    "餐厅",
    "餐饮",
    "家庭友好",
    "中档餐厅",
    "舒适餐厅",
    "休闲餐厅",
}


def is_generic_cuisine_label(value: Any) -> bool:
    """A non-informative cuisine/type label that should not be shown (家庭友好餐厅 等)."""
    t = clean_output_text(value)
    if not t or t in _GENERIC_CUISINE_LABELS:
        return True
    if "家庭友好" in t:
        return True
    return bool(re.fullmatch(r"(当地|本地|附近|中档|高档|经济|普通|舒适|休闲|家庭)*(餐厅|餐馆|餐饮|美食)", t))


def infer_restaurant_type_text(plan_row: Dict[str, Any], area: str = "") -> str:
    destination = clean_output_text(plan_row.get("destination_city"))
    if "哈尔滨" in destination or "Harbin" in destination:
        return "东北菜或饺子家庭餐厅"
    if "香港" in destination or "Hong Kong" in destination:
        return "茶餐厅、粤菜或点心家庭餐厅"
    if "上海" in destination or "Shanghai" in destination:
        return "本帮菜、西餐或江浙菜家庭餐厅"
    if "深圳" in destination or "Shenzhen" in destination:
        return "粤菜、海鲜或商场家庭餐厅"
    return "当地家庭友好餐厅"


def clean_split_item_for_export(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Drop vague/non-hard split activities before TXT export.

    Only direct hard-constraint split activities should be shown.  This prevents
    output like 部分成员/其他成员 and generic 分流活动 from appearing in the final TXT.
    """
    if not isinstance(item, dict):
        return None
    blob = json.dumps(item, ensure_ascii=False)
    if any(word in blob for word in ["部分成员", "其他成员", "有人", "其余成员"]):
        return None
    activity = cn_text(item.get("activity")) or cn_text(item.get("activity_or_term"))
    if not activity or activity in ["分流活动", "split activity"]:
        return None
    conflict_type = safe_text(item.get("conflict_type")).lower()
    if not any(
        key in conflict_type
        for key in ["must_include_vs_must_avoid", "direct_must_include", "hard"]
    ):
        return None
    roles_in = (
        item.get("participating_roles", [])
        if isinstance(item.get("participating_roles"), list)
        else []
    )
    roles_out = (
        item.get("non_participating_roles", [])
        if isinstance(item.get("non_participating_roles"), list)
        else []
    )
    if not roles_in or not roles_out:
        return None
    return item


def clean_route_text(value: Any) -> str:
    text = clean_output_text(value)
    if not text:
        return ""
    # Fix common generated awkward transfer descriptions.
    text = text.replace("转港岛线至金钟站（实际同一站）", "抵达金钟站")
    text = text.replace("转港岛线至金钟站", "抵达金钟站")
    text = text.replace("乘荃湾线至金钟站，抵达金钟站", "乘荃湾线至金钟站")
    # Collapse repeated stops in an "A->B->A" chain (e.g. 博物馆->午餐->博物馆), keeping the
    # first occurrence of each stop and always keeping the final stop so the hotel round-trip
    # (酒店->...->酒店) is preserved.
    if "->" in text or "→" in text:
        stops = [s.strip() for s in re.split(r"\s*(?:->|→)\s*", text) if s.strip()]
        result: List[str] = []
        seen = set()
        for i, stop in enumerate(stops):
            key = normalize_for_dedupe(stop)
            is_last = i == len(stops) - 1
            if key and key in seen and not is_last:
                continue
            if not result or normalize_for_dedupe(result[-1]) != key:
                result.append(stop)
            if key:
                seen.add(key)
        text = "->".join(result)
    # Remove exact repeated fragments inside a single route: "步行；步行".
    parts = re.split(r"[；;]+", text)
    parts = dedupe_keep_order(parts)
    text = "；".join(parts)
    return text.strip("；。 ")


def route_dedupe_key(route: str) -> str:
    text = clean_output_text(route)
    text = re.sub(r"（预估\s*[0-9.]+\s*元）", "", text)
    text = re.sub(r"[；;。\s]+", "", text)
    return text.lower()


def dedupe_routes_keep_cost(routes: List[str]) -> List[str]:
    seen = set()
    output = []
    for route in routes:
        route = clean_route_text(route)
        if not route:
            continue
        key = route_dedupe_key(route)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(route)
    return output


def format_duration_cn(minutes: Any) -> str:
    """Minutes -> '4小时30分' / '2小时' / '55分钟' (empty when unknown)."""
    m = int(round(safe_float(minutes, 0)))
    if m <= 0:
        return ""
    h, mm = divmod(m, 60)
    if h and mm:
        return f"{h}小时{mm}分"
    return f"{h}小时" if h else f"{mm}分钟"


def parse_json_like(value: Any, default: Any = None) -> Any:
    if default is None:
        default = []
    if isinstance(value, (dict, list)):
        return value
    text = safe_text(value)
    if not text or text.lower() in ["nan", "none", "null"]:
        return default
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return default


def read_optional_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def require_file(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"找不到 {name}: {path}\n请先确认 Module 7 已经运行完成。")


def join_text_items(value: Any, empty_text: str = "无") -> str:
    parsed = parse_json_like(value, default=[])
    if isinstance(parsed, list):
        items = []
        for item in parsed:
            if isinstance(item, dict):
                text = dict_to_phrase(item)
            else:
                text = cn_text(item)
            if text:
                items.append(text)
        return "、".join(items) if items else empty_text
    if isinstance(parsed, dict):
        text = dict_to_phrase(parsed)
        return text if text else empty_text
    text = cn_text(parsed)
    return text if text else empty_text


def dict_to_phrase(data: Dict[str, Any]) -> str:
    if not isinstance(data, dict) or not data:
        return ""
    keys = [
        "time",
        "activity",
        "activity_or_term",
        "poi_name",
        "poi_type",
        "restaurant_name",
        "restaurant_type",
        "hotel_name",
        "hotel_name_or_type",
        "main_transport_mode",
        "route",
        "route_strategy",
        "alternative_for_non_participants",
        "reason",
        "resolution",
        "price_level",
        "estimated_price_per_person",
    ]
    parts = []
    for key in keys:
        if key not in data:
            continue
        value = data.get(key)
        if isinstance(value, list):
            value_text = "、".join(cn_text(x) for x in value if cn_text(x))
        elif isinstance(value, dict):
            value_text = dict_to_phrase(value)
        else:
            value_text = cn_text(value)
        if value_text:
            parts.append(value_text)
    if parts:
        return "，".join(parts)
    fallback = []
    for _, value in data.items():
        if isinstance(value, (list, dict)):
            text = join_text_items(value, empty_text="")
        else:
            text = cn_text(value)
        if text:
            fallback.append(text)
    return "，".join(fallback)


def bool_to_cn(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    text = safe_text(value).lower()
    if text in ["true", "1", "yes", "y"]:
        return "是"
    if text in ["false", "0", "no", "n"]:
        return "否"
    return cn_text(value) or "未知"


def load_data():
    require_file(FINAL_PLANS_CSV, "Module 7 最终方案 generated_plans_feedback_final.csv")
    require_file(MEMBER_EVAL_CSV, "Module 7 成员评价 evaluation_member_level_feedback.csv")
    plans_df = pd.read_csv(FINAL_PLANS_CSV)
    member_df = pd.read_csv(MEMBER_EVAL_CSV)
    plan_eval_df = pd.read_csv(PLAN_EVAL_CSV) if PLAN_EVAL_CSV.exists() else pd.DataFrame()
    report = read_optional_json(REPORT_JSON)
    if "family_id" not in plans_df.columns:
        raise ValueError("generated_plans_feedback_final.csv 缺少 family_id 字段。")
    if "family_id" not in member_df.columns:
        raise ValueError("evaluation_member_level_feedback.csv 缺少 family_id 字段。")
    plans_df["family_id"] = plans_df["family_id"].astype(int)
    member_df["family_id"] = member_df["family_id"].astype(int)
    if not plan_eval_df.empty and "family_id" in plan_eval_df.columns:
        plan_eval_df["family_id"] = plan_eval_df["family_id"].astype(int)
    return plans_df, member_df, plan_eval_df, report


def describe_activity(item: Any) -> str:
    if isinstance(item, dict):
        time_text = cn_text(item.get("time"))
        activity = cn_text(item.get("activity"))
        poi_name = cn_text(item.get("poi_name"))
        poi_type = cn_text(item.get("poi_type"))
        duration = safe_int(item.get("estimated_duration_minutes"), 0)
        price = safe_float(item.get("ticket_price"), 0)
        raw_subject = poi_name or activity or poi_type or dict_to_phrase(item)
        if is_generic_activity_for_export(raw_subject):
            return ""
        subject = normalize_activity_label_for_export(raw_subject)
        details = []
        if time_text:
            details.append(time_text)
        if activity and activity != raw_subject and not is_generic_activity_for_export(activity):
            details.append(normalize_activity_label_for_export(activity))
        if duration > 0:
            details.append(f"约 {duration} 分钟")
        if should_show_ticket_for_activity(item, price):
            details.append(f"门票约 {price:.1f}")
        return subject + ("（" + "，".join(details) + "）" if details else "")
    if is_generic_activity_for_export(item):
        return ""
    return normalize_activity_label_for_export(item)


def flatten_activity_text_for_rough(value: Any) -> str:
    """Return a readable rough activity label when concrete POI generation failed.

    This is intentionally used only for final TXT display. Quality evaluation still
    reads Module 5 quality markers and can penalize or repair non-concrete items.
    """
    if isinstance(value, list):
        parts = [flatten_activity_text_for_rough(x) for x in value]
        return "、".join([p for p in parts if p])
    if isinstance(value, dict):
        for key in [
            "rough_intent",
            "intent",
            "activity_intent",
            "activity_type",
            "poi_type",
            "activity",
            "poi_name",
            "name",
            "name_or_type",
            "theme",
        ]:
            txt = clean_output_text(value.get(key))
            if txt:
                return normalize_activity_label_for_export(txt) or txt
        return ""
    txt = clean_output_text(value)
    return normalize_activity_label_for_export(txt) or txt


def rough_intent_from_day(day_item: Optional[Dict[str, Any]], section: str = "") -> str:
    """Infer a coarse itinerary description from Module 4/5 day-level intent fields."""
    if not isinstance(day_item, dict):
        return ""
    candidates: List[Any] = []
    if section:
        candidates.extend(
            [
                day_item.get(f"{section}_intent"),
                day_item.get(f"rough_{section}_intent"),
                day_item.get(f"{section}_activity_type"),
            ]
        )
    candidates.extend(
        [
            day_item.get("day_theme"),
            day_item.get("theme"),
            day_item.get("core_theme"),
            day_item.get("activity_type_targets"),
            day_item.get("meal_strategy") if section == "evening" else "",
        ]
    )
    for value in candidates:
        if isinstance(value, list):
            txt = "、".join([cn_text(x) for x in value if cn_text(x)])
        else:
            txt = cn_text(value)
        txt = clean_output_text(txt)
        if txt and not is_placeholder_text(txt):
            return txt
    return ""


def rough_activity_fallback(
    value: Any, day_item: Optional[Dict[str, Any]] = None, section: str = ""
) -> str:
    """Display a coarse itinerary instead of exposing internal 'to be filled' text."""
    raw = flatten_activity_text_for_rough(value)
    intent = rough_intent_from_day(day_item, section)
    # Departure / return / leisure days genuinely have no fixed POI; render them cleanly instead of
    # exposing a "未具体化" placeholder that reads like a failure.
    theme_blob = f"{raw} {intent} {clean_output_text((day_item or {}).get('theme'))}".lower()
    if re.search(r"(返回|返程|回程|离开|出发|退房|打包|送机|前往机场|机场|结束|总结)", theme_blob):
        return "退房与返程准备：酒店附近自由活动、就近用餐，按车次或航班前往车站/机场"
    if re.search(r"(休息|休闲|放松|自由活动|自由安排|缓冲)", theme_blob):
        return "自由活动：酒店或周边商圈休闲、就近用餐，不安排固定景点"
    chosen = raw or intent
    if raw and intent and raw != intent and len(raw) < 30:
        chosen = f"{intent}，方向包括：{raw}"
    if not chosen:
        if section == "morning":
            chosen = "结合当天主题安排低强度家庭活动"
        elif section == "afternoon":
            chosen = "结合当天主题安排轻松游览或亲子活动"
        else:
            chosen = "根据当天状态安排休息或本地餐饮"
    return f"大概安排：{chosen}（未具体化到单个真实地点）"


def describe_activity_list(
    value: Any, day_item: Optional[Dict[str, Any]] = None, section: str = ""
) -> str:
    if isinstance(value, list):
        raw_count = len(value)
        items = [describe_activity(item) for item in value]
        items = [item for item in items if item]
        if items:
            return "、".join(items)
        if raw_count > 0:
            return rough_activity_fallback(value, day_item=day_item, section=section)
        return rough_activity_fallback(value, day_item=day_item, section=section)
    if is_generic_activity_for_export(value):
        return (
            rough_activity_fallback(value, day_item=day_item, section=section)
            if safe_text(value) or day_item
            else "休息或自由安排"
        )
    return normalize_activity_label_for_export(value) or rough_activity_fallback(
        value, day_item=day_item, section=section
    )


def describe_meal(
    value: Any, meal_name: str, plan_row: Optional[Dict[str, Any]] = None, duplicate_index: int = 1,
) -> str:
    """Format one meal while filtering placeholders and repeated restaurant names."""
    plan_row = plan_row or {}
    meal = value if isinstance(value, dict) else parse_json_like(value, default={})
    if not isinstance(meal, dict) or not meal:
        subject = infer_city_family_restaurant(plan_row, meal_name)
        return f"{meal_name}：{subject}"

    restaurant_name = clean_output_text(meal.get("restaurant_name"))
    restaurant_type = clean_output_text(meal.get("restaurant_type"))
    area = (
        clean_area_label(meal.get("restaurant_area"))
        or clean_area_label(meal.get("address_or_location"))
        or clean_area_label(meal.get("area"))
    )
    if is_bad_area_text(area):
        area = ""
    if not area:
        hotel = (
            parse_json_like(plan_row.get("hotel_plan_json", "{}"), default={})
            if isinstance(plan_row, dict)
            else {}
        )
        if isinstance(hotel, dict):
            station = clean_output_text(hotel.get("nearby_metro_station"))
            hotel_area = clean_output_text(hotel.get("hotel_area")) or clean_output_text(
                hotel.get("area")
            )
            if hotel_area and station:
                area = f"{hotel_area}、{station}"
            else:
                area = station or hotel_area
            area = clean_area_label(area) or area
    price_level = clean_output_text(meal.get("price_level")) or "中等"
    price_person = safe_float(meal.get("estimated_price_per_person"), 0)
    comfort = clean_output_text(meal.get("comfort_note"))

    if is_placeholder_text(restaurant_name) or not is_specific_restaurant_name_for_export(
        restaurant_name
    ):
        restaurant_name = ""
        meal["specific_restaurant_found"] = False
        meal["restaurant_grounding_status"] = "area_fallback_no_specific_name"
    if is_placeholder_text(restaurant_type):
        restaurant_type = ""
    if is_placeholder_text(price_level):
        price_level = "中等"

    status_text = clean_output_text(meal.get("restaurant_grounding_status"))
    specific_flag_text = clean_output_text(meal.get("specific_restaurant_found")).lower()
    specific_flag = meal.get("specific_restaurant_found") is True or specific_flag_text in [
        "true",
        "1",
        "yes",
        "y",
        "是",
    ]
    specific_found = bool(restaurant_name) and (
        specific_flag
        or status_text == "specific_name_found"
        or status_text != "area_fallback_no_specific_name"
    )

    # v57: If no reliable specific restaurant exists, do not print a generic type
    # as if it were the restaurant's name.  Export only the dining range/area,
    # cuisine/type and price range; Module 6 will apply a soft penalty.
    if not restaurant_name or not specific_found:
        restaurant_type_text = (
            restaurant_type
            if restaurant_type and not is_placeholder_text(restaurant_type)
            else infer_restaurant_type_text(plan_row, area)
        )
        range_text = area or clean_output_text(plan_row.get("destination_city")) or "当地合适区域"
        range_display = range_text if range_text.endswith("附近") else f"{range_text}附近"
        parts = [f"{meal_name}：不指定具体餐厅"]
        parts.append(f"用餐范围：{range_display}")
        if not is_generic_cuisine_label(restaurant_type_text):
            parts.append(f"菜系：{restaurant_type_text}")
        parts.append(f"参考价格：{meal_price_range_text(price_level)}")
        note = (
            comfort
            if comfort and not is_placeholder_text(comfort)
            else "未取得可靠具体餐厅名，仅给出范围附近；作为软扣分项，不编造店名。"
        )
        parts.append(f"说明：{note}")
        return "；".join(parts)

    subject = restaurant_name

    # Avoid showing the same restaurant name across several days as if it were required every meal.
    if duplicate_index > 1 and restaurant_name:
        price_range = meal_price_range_text(price_level)
        restaurant_type_text = (
            restaurant_type
            if restaurant_type and not is_placeholder_text(restaurant_type)
            else infer_restaurant_type_text(plan_row, area)
        )
        subject = "不指定具体餐厅"
        duplicate_range_text = area or clean_output_text(plan_row.get("destination_city")) or "同区域"
        duplicate_range_display = (
            duplicate_range_text
            if duplicate_range_text.endswith("附近")
            else f"{duplicate_range_text}附近"
        )
        parts = [f"{meal_name}：{subject}"]
        parts.append(f"用餐范围：{duplicate_range_display}")
        parts.append(f"类型：同价位{restaurant_type_text}")
        parts.append(f"参考价格：人均约{price_range}")
        parts.append("说明：避免重复使用同一家餐厅；此餐改为范围附近备选，作为软扣分项。")
        return "；".join(parts)

    parts = [f"{meal_name}：{subject}"]
    if area and not is_placeholder_text(area):
        parts.append(f"位置：{area}")
    # Show a real cuisine only; drop the generic "家庭友好餐饮" type and the redundant
    # "价格等级：medium" — the concrete per-person price below carries the real signal.
    cuisine = restaurant_type if not is_generic_cuisine_label(restaurant_type) else ""
    if cuisine:
        parts.append(f"菜系：{cuisine}")
    if price_person > 0:
        parts.append(f"预估人均：{price_person:.0f}元")
    if comfort and not is_placeholder_text(comfort):
        parts.append(f"说明：{comfort}")
    return "；".join(parts)


def get_split_activities_from_plan(plan_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = plan_row.get("split_activities_json", "")
    if not safe_text(raw):
        raw = plan_row.get("split_activities", "")
    items = parse_json_like(raw, default=[])
    if not isinstance(items, list):
        return []
    cleaned = []
    for item in items:
        if isinstance(item, dict):
            item = clean_split_item_for_export(item)
            if item:
                cleaned.append(item)
    return cleaned


def split_activity_day(item: Dict[str, Any], default_day: int = 1) -> int:
    for key in ["day", "day_index", "day_number"]:
        value = safe_int(item.get(key), 0)
        if value > 0:
            return value
    return default_day


def describe_split_activity(item: Dict[str, Any]) -> str:
    item = clean_split_item_for_export(item)
    if not item:
        return ""
    role_assignment = cn_text(item.get("role_assignment"))
    if role_assignment and "部分成员" not in role_assignment and "其他成员" not in role_assignment:
        return role_assignment.rstrip("。")
    activity = cn_text(item.get("activity")) or cn_text(item.get("activity_or_term"))
    participants = join_text_items(item.get("participating_roles", []), empty_text="")
    non_participants = join_text_items(item.get("non_participating_roles", []), empty_text="")
    alternative = cn_text(item.get("alternative_for_non_participants")) or "附近公园或咖啡馆轻松休息"
    if not participants or not non_participants:
        return ""
    return f"{participants} 参加“{activity}”；{non_participants} 不参加，改为{alternative}"


def day_split_sentences(
    day_item: Dict[str, Any], top_level_splits: List[Dict[str, Any]], day_num: int
) -> List[str]:
    splits = []
    daily = day_item.get("split_activities", []) if isinstance(day_item, dict) else []
    if isinstance(daily, list):
        splits.extend([item for item in daily if isinstance(item, dict)])
    splits.extend([item for item in top_level_splits if split_activity_day(item, 1) == day_num])
    seen = set()
    output = []
    for item in splits:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        sentence = describe_split_activity(item)
        if sentence:
            output.append(sentence)
    return output


def get_per_day_items(plan_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    days = parse_json_like(plan_row.get("per_day_itinerary_json", "[]"), default=[])
    return [item for item in days if isinstance(item, dict)] if isinstance(days, list) else []


_CITY_ALIASES = {
    "hong kong": {"hong kong", "香港"},
    "香港": {"hong kong", "香港"},
    "harbin": {"harbin", "哈尔滨"},
    "哈尔滨": {"harbin", "哈尔滨"},
    "shanghai": {"shanghai", "上海"},
    "上海": {"shanghai", "上海"},
    "shenzhen": {"shenzhen", "深圳"},
    "深圳": {"shenzhen", "深圳"},
    "beijing": {"beijing", "北京"},
    "北京": {"beijing", "北京"},
    "guangzhou": {"guangzhou", "广州"},
    "广州": {"guangzhou", "广州"},
}


def _city_tokens(city: Any) -> set:
    c = safe_text(city).strip().lower()
    return (_CITY_ALIASES.get(c, {c}) | {c}) if c else set()


def _is_placeholder_poi(name: Any, city_tokens: set) -> bool:
    """A route stop that is not a real visitable POI: empty, the city name itself, a transit node
    (station/airport/pier), or a generic label."""
    n = safe_text(name).strip()
    if not n or n.lower() in city_tokens:
        return True
    if n.endswith("站") or n.endswith("码头") or re.search(r"(地铁站|火车站|高铁站|汽车站|机场|码头)", n):
        return True
    return is_generic_activity_for_export(n)


def _is_vague_export_activity(text: Any, city_tokens: Optional[set] = None) -> bool:
    t = safe_text(text).strip()
    if (not t) or t.startswith("大概安排") or "未具体化" in t:
        return True
    return bool(city_tokens) and t.lower() in city_tokens


def _extract_route_section_pois(route_text: Any, city_tokens: set) -> tuple:
    """Split a 'A->B->午餐->C->晚餐->酒店' route into clean (morning_pois, afternoon_pois): drop the
    repeated hotel/station hub at both ends, meals, transit nodes and the city name; dedupe."""
    text = clean_route_text(route_text)
    if not text:
        return [], []
    stops = [s.strip() for s in re.split(r"\s*->\s*", re.sub(r"（预估.*?）", "", text)) if s.strip()]
    if len(stops) >= 2 and stops[0] == stops[-1]:  # first==last is the hotel/station hub
        stops = stops[1:-1]
    exclude = re.compile(r"(酒店|宾馆|旅馆|招待所|^返回|返程|^前往|出租车|打车|步行|公交|未提供|附近|^游览|^参观|^游玩)")
    meal = re.compile(
        r"(早餐|午餐|中餐|晚餐|用餐|美食|餐厅|餐饮|火锅|自助|小吃|大排档|Cafe|cafe|Café|Kitchen|Restaurant|Bar)"
    )
    morning, afternoon = [], []
    seen_meal = False
    for s in stops:
        if meal.search(s):
            seen_meal = True
            continue
        poi = re.sub(r"^(游览|参观|游玩|前往|抵达|在|于)", "", s).strip()
        if exclude.search(poi) or _is_placeholder_poi(poi, city_tokens):
            continue
        (afternoon if seen_meal else morning).append(poi)
    morning = list(dict.fromkeys(morning))
    afternoon = [
        p for p in dict.fromkeys(afternoon) if p not in morning
    ]  # don't repeat morning hubs
    return morning, afternoon


def _day_route_section_pois(
    plan_row: Dict[str, Any], day_num: int, day_item: Dict[str, Any], city_tokens: set
) -> tuple:
    transport = parse_json_like(plan_row.get("transportation_plan_json", "{}"), default={})
    route_text = ""
    if isinstance(transport, dict):
        for it in transport.get("daily_local_transport") or []:
            if isinstance(it, dict) and safe_int(it.get("day"), 0) == day_num:
                route_text = safe_text(it.get("route"))
                if route_text:
                    break
    if not route_text:
        route_text = join_text_items(day_item.get("transportation", []), empty_text="")
    return _extract_route_section_pois(route_text, city_tokens)


def _load_snapshot_attractions_for_city(dest: str) -> List[str]:
    """Return a list of POI names from the snapshot attractions cache for a destination city."""
    try:
        from src.grounding import _cache_record_matches_destination, _load_category_cache

        recs = _load_category_cache("attractions")
    except Exception:
        return []
    names = []
    for r in recs:
        if not isinstance(r, dict):
            continue
        if not _cache_record_matches_destination(r, dest):
            continue
        n = safe_text(r.get("poi_name", ""))
        if n and n not in names:
            names.append(n)
    return names


def _resolve_section_text(
    day_item: Dict[str, Any],
    section: str,
    route_pois: List[str],
    city_tokens: set,
    snapshot_pois: Optional[List[str]] = None,
    used_snapshot: Optional[set] = None,
) -> str:
    """Render a day's morning/afternoon: keep concrete plan activities, else backfill from clean route
    POIs, else pick an unused snapshot attraction — never 家庭自由活动 or 未具体化."""
    # Arrival / departure markers render directly and never get backfilled with a POI.
    sec_items = day_item.get(section, [])
    if isinstance(sec_items, list):
        for it in sec_items:
            if isinstance(it, dict) and (
                it.get("is_transit") or it.get("poi_type") in ("arrival", "departure")
            ):
                return safe_text(it.get("activity")) or (
                    "抵达" if it.get("poi_type") == "arrival" else "返程"
                )
    txt = describe_activity_list(day_item.get(section, []), day_item=day_item, section=section)
    if not _is_vague_export_activity(txt, city_tokens):
        return txt
    if route_pois:
        return "、".join(route_pois)
    if snapshot_pois and used_snapshot is not None:
        for poi in snapshot_pois:
            if poi not in used_snapshot:
                used_snapshot.add(poi)
                return poi
    if (
        txt.startswith("大概安排")
        or "未具体化" in txt
        or (city_tokens and txt.strip().lower() in city_tokens)
    ):
        return "周边景点游览"
    return txt


def section_daily_itinerary(plan_row: Dict[str, Any]) -> str:
    days = get_per_day_items(plan_row)
    if not days:
        return "• 每日行程：\n每日行程没有提供结构化安排。"
    splits = get_split_activities_from_plan(plan_row)
    dest = safe_text(plan_row.get("destination_city", ""))
    ct = _city_tokens(dest)
    snapshot_pois = _load_snapshot_attractions_for_city(dest)
    used_snapshot: set = set()
    blocks = ["• 每日行程："]
    for index, item in enumerate(days):
        day_num = safe_int(item.get("day"), index + 1)
        # Only show a theme when it's meaningful — the generic "家庭行程" repeats on every
        # day and adds nothing, so drop it and just write "第 N 天".
        theme = cn_text(item.get("theme"))
        if theme in ("家庭行程", "轻松家庭行程", "家庭旅行", "家庭游", "家庭", "行程", "亲子行程"):
            theme = ""
        walking = safe_float(item.get("estimated_walking_distance_km"), 0)
        if walking <= 0:
            walking = estimate_day_walking_km(item) or 2.0
        return_time = cn_text(item.get("estimated_return_time")) or "未注明"
        lines = [f"第 {day_num} 天：{theme}" if theme else f"第 {day_num} 天"]
        route_morning, route_afternoon = _day_route_section_pois(plan_row, day_num, item, ct)
        morning_txt = _resolve_section_text(
            item, "morning", route_morning, ct, snapshot_pois, used_snapshot
        )
        afternoon_txt = _resolve_section_text(
            item, "afternoon", route_afternoon, ct, snapshot_pois, used_snapshot
        )
        day_sentences = list(day_split_sentences(item, splits, day_num))
        # When the afternoon IS the split (no real POI of its own — just an empty slot or a
        # filler), point to the split instead of showing a separate filler activity
        # ('下午本来就是分流活动，不要再加其他的').
        aft_items = item.get("afternoon") if isinstance(item.get("afternoon"), list) else []
        aft_is_filler = (not aft_items) or all(
            isinstance(x, dict) and x.get("filled_empty_slot") for x in aft_items
        )
        if day_sentences and aft_is_filler:
            afternoon_txt = "分流活动（详见下方）"
        lines.append(f"上午：{morning_txt}")
        lines.append(f"下午：{afternoon_txt}")
        evening = join_text_items(item.get("evening", []), empty_text="")
        if evening:
            lines.append(f"晚上：{evening}")
        rest_blocks = join_text_items(item.get("rest_blocks", []), empty_text="")
        if rest_blocks:
            lines.append(f"休息：{rest_blocks}")
        for sentence in day_sentences:
            lines.append(f"分流活动：{sentence}")
        lines.append(f"预计步行约 {walking:.1f} 公里，返回时间 {return_time}。")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def section_daily_transport(plan_row: Dict[str, Any]) -> str:
    """Export transportation once per day, deduplicating plan-level and day-level routes."""
    transport = parse_json_like(plan_row.get("transportation_plan_json", "{}"), default={})
    days = get_per_day_items(plan_row)
    blocks = ["• 每日交通安排："]
    daily_routes: Dict[int, List[str]] = {}
    plan_transport_days = set()

    if isinstance(transport, dict) and transport:
        mode = clean_output_text(transport.get("main_transport_mode")) or "地铁、步行和必要时短途打车"
        strategy = clean_output_text(transport.get("route_strategy")) or "减少换乘和步行，优先直达"
        blocks.append(f"主要方式：{mode}。路线策略：{strategy}。")
        intercity = (
            transport.get("intercity_transport", [])
            if isinstance(transport.get("intercity_transport", []), list)
            else []
        )
        for item in intercity:
            if not isinstance(item, dict):
                continue
            route = clean_route_text(item.get("route"))
            mode = clean_output_text(item.get("mode"))
            label = f"{mode} {route}".strip() if mode else route
            if not label:
                continue
            price = safe_float(item.get("estimated_price_per_person"), 0)
            duration = format_duration_cn(item.get("estimated_duration_minutes"))
            parts = [f"城际交通：{label}"]
            if duration:
                parts.append(f"约 {duration}")
            if price > 0:
                parts.append(f"预估每人 {price:.0f} 元")
            blocks.append("，".join(parts) + "。")
        daily_local = (
            transport.get("daily_local_transport", [])
            if isinstance(transport.get("daily_local_transport", []), list)
            else []
        )
        for item in daily_local:
            if isinstance(item, dict):
                day = safe_int(item.get("day"), 0)
                route = clean_route_text(item.get("route"))
                cost = safe_float(item.get("estimated_cost_total"), 0)
                if day > 0 and route:
                    if cost > 0:
                        route = f"{route}（预估 {cost:.1f} 元）"
                    daily_routes.setdefault(day, []).append(route)
                    plan_transport_days.add(day)

    # Add per-day transportation only when it is not already covered by transportation_plan_json.
    for index, item in enumerate(days):
        day_num = safe_int(item.get("day"), index + 1)
        transportation = join_text_items(item.get("transportation", []), empty_text="")
        note = clean_output_text(item.get("transportation_note"))
        route = clean_route_text(transportation)
        if note:
            route = f"{route}；{note}" if route else note
        if route and day_num not in plan_transport_days:
            daily_routes.setdefault(day_num, []).append(route)

    for day_num in sorted(daily_routes):
        routes = dedupe_routes_keep_cost(daily_routes[day_num])
        if routes:
            blocks.append(f"第 {day_num} 天：" + "；".join(routes) + "。")

    if len(blocks) == 1:
        blocks.append("交通安排未注明。")
    return "\n".join(blocks)


def section_meals(plan_row: Dict[str, Any]) -> str:
    """Export meals with placeholder filtering and repeated-restaurant cleanup."""
    days = get_per_day_items(plan_row)
    blocks = ["• 餐饮安排："]
    if not days:
        blocks.append("餐饮安排未注明。")
        return "\n".join(blocks)

    restaurant_seen: Dict[str, int] = {}
    for index, item in enumerate(days):
        day_num = safe_int(item.get("day"), index + 1)
        blocks.append(f"第 {day_num} 天：")
        for meal_key, meal_name in [("lunch", "午餐"), ("dinner", "晚餐")]:
            meal = item.get(meal_key, {})
            parsed_meal = meal if isinstance(meal, dict) else parse_json_like(meal, default={})
            restaurant_name = ""
            if isinstance(parsed_meal, dict):
                restaurant_name = clean_output_text(parsed_meal.get("restaurant_name"))
            norm_name = (
                normalize_for_dedupe(restaurant_name)
                if restaurant_name and not is_placeholder_text(restaurant_name)
                else ""
            )
            duplicate_index = 1
            if norm_name:
                duplicate_index = restaurant_seen.get(norm_name, 0) + 1
                restaurant_seen[norm_name] = duplicate_index
            blocks.append(
                describe_meal(meal, meal_name, plan_row=plan_row, duplicate_index=duplicate_index)
            )
    return "\n".join(blocks)


def section_hotel(plan_row: Dict[str, Any]) -> str:
    hotel = parse_json_like(plan_row.get("hotel_plan_json", "{}"), default={})
    blocks = ["• 酒店安排："]
    if not isinstance(hotel, dict) or not hotel:
        blocks.append("住宿未注明。")
        return "\n".join(blocks)
    hotel_name = (
        cn_text(hotel.get("hotel_name")) or cn_text(hotel.get("hotel_name_or_type")) or "未注明酒店"
    )
    area = cn_text(hotel.get("hotel_area")) or "未注明区域"
    destination = cn_text(plan_row.get("destination_city"))
    generic_hotel_name = is_generic_hotel_name_for_export(hotel_name, area, destination)
    station = cn_text(hotel.get("nearby_metro_station"))
    lines = join_text_items(hotel.get("metro_lines_near_hotel", []), empty_text="")
    nightly = safe_float(hotel.get("estimated_price_per_night"), 0)
    reason = cn_text(hotel.get("hotel_selection_reason"))
    note = cn_text(hotel.get("verification_note"))
    if generic_hotel_name:
        scope = area
        if station:
            scope = f"{area}、{station}" if area and area != "未注明区域" else station
        blocks.append("酒店：不指定具体酒店。")
        blocks.append(f"住宿范围：{scope}附近。")
    else:
        blocks.append(f"酒店：{hotel_name}。")
        blocks.append(f"区域：{area}。")
    if nightly > 0:
        blocks.append(f"预估每晚价格：{nightly:.1f}。")
    if station:
        blocks.append(f"附近地铁站：{station}。")
    if lines:
        blocks.append(f"可用线路：{lines}。")
    if reason:
        blocks.append(f"选择理由：{reason}")
    if note:
        blocks.append(f"核验说明：{note}")
    return "\n".join(blocks)


def section_budget(plan_row: Dict[str, Any]) -> str:
    data = parse_json_like(plan_row.get("cost_breakdown_json", "{}"), default={})
    total_cost = safe_float(plan_row.get("total_cost"), 0)
    budget_limit = safe_float(plan_row.get("budget_limit"), 0)
    blocks = ["• 预算估算："]
    currency = "CNY"
    if isinstance(data, dict) and data:
        currency = cn_text(data.get("currency")) or currency
        items = [
            ("酒店费用", safe_float(data.get("hotel_cost"), 0)),
            ("城际交通费用", safe_float(data.get("intercity_transport_cost"), 0)),
            ("市内交通费用", safe_float(data.get("local_transport_cost"), 0)),
            ("餐饮费用", safe_float(data.get("meal_cost"), 0)),
            ("景点门票费用", safe_float(data.get("attraction_ticket_cost"), 0)),
            ("其他费用", safe_float(data.get("other_cost"), 0)),
        ]
        computed_total = safe_float(data.get("total_cost"), total_cost)
        if computed_total <= 0:
            computed_total = sum(value for _, value in items)
        # v60: Some budget-protection paths cap total_cost but leave stale other_cost,
        # which makes the displayed item sum exceed the total.  Reconcile for TXT.
        known_without_other = sum(value for label, value in items if label != "其他费用")
        other = max(0.0, computed_total - known_without_other) if computed_total > 0 else 0.0
        if abs(sum(value for _, value in items) - computed_total) > 1.0:
            items = [
                (label, value) if label != "其他费用" else (label, other) for label, value in items
            ]
        blocks.append(f"总费用：{computed_total:.1f} {currency}。")
        if budget_limit > 0:
            blocks.append(f"预算上限：{budget_limit:.1f} {currency}。")
        for label, amount in items:
            if amount > 0:
                blocks.append(f"{label}：{amount:.1f} {currency}。")
        note = cn_text(data.get("cost_note"))
        if note:
            blocks.append(f"费用说明：{note}")
    else:
        blocks.append(f"总费用：{total_cost:.1f} {currency}。")
        if budget_limit > 0:
            blocks.append(f"预算上限：{budget_limit:.1f} {currency}。")
        blocks.append("费用明细未注明。")
    online = parse_json_like(plan_row.get("online_grounding_json", "{}"), default={})
    if isinstance(online, dict) and online:
        status = cn_text(online.get("search_status")) or "未知"
        provider = cn_text(online.get("provider")) or "未知"
        blocks.append(f"在线查询状态：{status}；来源：{provider}。")
        urls = (
            online.get("source_urls", []) if isinstance(online.get("source_urls", []), list) else []
        )
        urls = [
            safe_text(url) for url in urls if safe_text(url).startswith(("http://", "https://"))
        ]
        if urls:
            blocks.append("参考链接：" + "；".join(urls[:5]))
    return "\n".join(blocks)


def section_member_preferences(member_rows: pd.DataFrame) -> str:
    blocks = ["• 每个成员偏好满足情况说明："]
    if member_rows.empty:
        blocks.append("无成员评价信息。")
        return "\n".join(blocks)
    if "member_id" in member_rows.columns:
        member_rows = member_rows.sort_values(by=["member_id"])
    for _, row in member_rows.iterrows():
        member_id = safe_int(row.get("member_id"), 0)
        role = cn_text(row.get("role")) or "成员"
        score = safe_float(row.get("individual_satisfaction_score"), 0)
        accepted = bool_to_cn(row.get("accepted_vote"))
        hard_pass = bool_to_cn(row.get("hard_constraint_pass"))
        pref_rate = safe_float(row.get("preference_coverage_rate"), -1)
        must_rate = safe_float(row.get("must_have_satisfaction_rate"), -1)
        matched_pref = join_text_items(row.get("matched_preferences", []), empty_text="无明确匹配")
        matched_must = join_text_items(row.get("matched_must_have", []), empty_text="无明确匹配")
        explanation = cn_text(row.get("member_explanation"))
        llm_explanation = cn_text(row.get("llm_judge_explanation"))
        parts = [f"成员 {member_id}（{role}）：满意度 {score:.1f}，接受：{accepted}，硬约束通过：{hard_pass}"]
        if pref_rate >= 0:
            parts.append(f"兴趣覆盖率 {pref_rate:.2f}")
        if must_rate >= 0:
            parts.append(f"底线满足率 {must_rate:.2f}")
        parts.append(f"匹配偏好：{matched_pref}")
        parts.append(f"匹配必需项：{matched_must}")
        if explanation:
            parts.append(f"说明：{explanation}")
        if llm_explanation:
            parts.append(f"LLM Judge：{llm_explanation}")
        blocks.append("；".join(parts) + "。")
    return "\n".join(blocks)


def section_consensus(plan_row: Dict[str, Any], plan_eval_row: Optional[Dict[str, Any]]) -> str:
    departure = cn_text(plan_row.get("departure_city")) or "未注明"
    destination = cn_text(plan_row.get("destination_city")) or "未注明"
    day = safe_int(plan_row.get("day"), 0)
    acceptance = safe_float(plan_row.get("acceptance_vote_ratio"), 0)
    if plan_eval_row:
        consensus_status = cn_text(plan_eval_row.get("consensus_status"))
        hard_rate = safe_float(plan_eval_row.get("hard_constraint_pass_rate"), 0)
        avg_sat = safe_float(plan_eval_row.get("average_group_satisfaction"), 0)
        min_sat = safe_float(plan_eval_row.get("minimum_member_satisfaction"), 0)
        group_score = safe_float(plan_eval_row.get("group_consensus_score"), 0)
        fairness_obj = safe_float(plan_eval_row.get("fairness_aware_group_objective"), 0)
        split_count = safe_int(plan_eval_row.get("split_satisfiable_constraint_count"), 0)
        unsat_count = safe_int(plan_eval_row.get("unsatisfiable_constraint_count"), 0)
    else:
        consensus_status = ""
        hard_rate = 0
        avg_sat = 0
        min_sat = 0
        group_score = 0
        fairness_obj = 0
        split_count = 0
        unsat_count = 0
    consensus_status = consensus_status or cn_text(plan_row.get("consensus_status")) or "未注明"
    coordinator_summary = cn_text(plan_row.get("coordinator_summary"))
    typing = parse_json_like(plan_row.get("constraint_satisfiability_json", "{}"), default={})
    blocks = ["• 最终共识说明："]
    blocks.append(f"本方案为 {departure} 到 {destination} 的 {day} 天家庭行程，当前状态为{consensus_status}。")
    blocks.append(f"成员接受比例为 {acceptance:.2f}，硬约束通过率为 {hard_rate:.2f}。")
    if avg_sat > 0:
        blocks.append(f"平均满意度为 {avg_sat:.1f}，最低成员满意度为 {min_sat:.1f}，群体共识分为 {group_score:.1f}。")
    if fairness_obj > 0:
        blocks.append(f"公平感知群体目标分为 {fairness_obj:.1f}。")
    if split_count > 0 or unsat_count > 0:
        blocks.append(f"约束分类结果：分流可满足约束 {split_count} 个，不可满足约束 {unsat_count} 个。")
    elif isinstance(typing, dict) and typing:
        blocks.append("约束分类结果已记录，未发现需要额外说明的不可满足约束。")
    if coordinator_summary:
        blocks.append(f"共识说明：{coordinator_summary}")
    return "\n".join(blocks)


def build_family_txt(
    family_id: int,
    plan_row: Dict[str, Any],
    member_rows: pd.DataFrame,
    plan_eval_row: Optional[Dict[str, Any]] = None,
    report: Optional[Dict[str, Any]] = None,
) -> str:
    parts = [
        f"家庭 {family_id} 最终行程",
        "=" * 40,
        section_daily_itinerary(plan_row),
        "",
        section_daily_transport(plan_row),
        "",
        section_meals(plan_row),
        "",
        section_hotel(plan_row),
        "",
        section_budget(plan_row),
        "",
        section_member_preferences(member_rows),
        "",
        section_consensus(plan_row, plan_eval_row),
    ]
    return "\n".join(parts)


def export_final_itinerary_txt(
    output_dir: Path = OUTPUT_DIR, combined_output_txt: Path = COMBINED_OUTPUT_TXT
):
    plans_df, member_df, plan_eval_df, report = load_data()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_texts = []
    if EXPORT_ONLY_SUCCESS_PLANS and "status" in plans_df.columns:
        before_count = len(plans_df)
        plans_df = plans_df[plans_df["status"].astype(str).str.lower() == "success"].copy()
        print(f"[Export TXT] 仅导出 success 方案: {len(plans_df)}/{before_count}")
    family_ids = sorted(plans_df["family_id"].astype(int).unique().tolist())
    print(f"[Export TXT] 发现最终方案 family 数量: {len(family_ids)}")
    print("[Export TXT] 当前导出格式: 每日行程、每日交通、餐饮、酒店、预算、成员偏好满足、最终共识。")
    for family_id in family_ids:
        plan_rows = plans_df[plans_df["family_id"].astype(int) == int(family_id)]
        if plan_rows.empty:
            continue
        plan_row = plan_rows.iloc[0].to_dict()
        member_rows = member_df[member_df["family_id"].astype(int) == int(family_id)].copy()
        plan_eval_row = None
        if not plan_eval_df.empty and "family_id" in plan_eval_df.columns:
            eval_rows = plan_eval_df[plan_eval_df["family_id"].astype(int) == int(family_id)]
            if not eval_rows.empty:
                plan_eval_row = eval_rows.iloc[0].to_dict()
        if EXPORT_REQUIRE_EVALUATION_ROW and plan_eval_row is None:
            print(f"[Export TXT] 跳过 family_id={family_id}: 缺少 Module 6/7 评价结果。")
            continue
        txt = build_family_txt(
            family_id=family_id,
            plan_row=plan_row,
            member_rows=member_rows,
            plan_eval_row=plan_eval_row,
            report=report,
        )
        family_output_path = output_dir / f"family_{family_id}_final_itinerary_input.txt"
        with open(family_output_path, "w", encoding="utf-8") as f:
            f.write(txt)
        all_texts.append(txt)
        print(f"[Export TXT] 已输出: {family_output_path}")
    with open(combined_output_txt, "w", encoding="utf-8") as f:
        f.write("\n\n\n".join(all_texts))
    print(f"[Export TXT] 合并文件已输出: {combined_output_txt}")
    return {
        "output_dir": str(output_dir),
        "combined_output_txt": str(combined_output_txt),
        "family_count": len(family_ids),
    }


if __name__ == "__main__":
    export_final_itinerary_txt()
