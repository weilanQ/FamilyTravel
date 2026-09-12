from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import pandas as pd


def safe_text(value: Any) -> str:
    """转成去空白的字符串；None / NaN 返回空串。"""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value: Any, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def extract_json(text: Any) -> Dict[str, Any]:
    """从模型返回里解析出 JSON 对象，容忍 ```json 围栏和前后多余文本。"""
    if text is None:
        raise ValueError("模型返回内容为空")
    text = str(text).strip()
    if text == "":
        raise ValueError("模型返回内容是空字符串")
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"没有找到 JSON 内容: {repr(text[:300])}")
    return json.loads(text[start : end + 1])


def time_to_minutes(time_text: Any) -> Optional[int]:
    """'20:30' -> 1230；无法解析返回 None。"""
    text = safe_text(time_text)
    if text == "":
        return None
    match = re.search(r"(\d{1,2}):(\d{2})", text)
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))
