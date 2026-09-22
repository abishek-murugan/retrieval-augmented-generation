import json
import re


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from model output: {text[:300]}")


def get_bool(data: dict, keys: list[str]) -> bool:
    for key in keys:
        if key in data:
            val = data[key]
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.strip().lower() in {"true", "1", "yes", "supported", "useful"}
    return False


def get_str(data: dict, keys: list[str], default: str = "") -> str:
    for key in keys:
        if key in data and data[key]:
            return str(data[key])
    return default


def get_int_list(data: dict, keys: list[str]) -> list[int]:
    for key in keys:
        if key in data:
            val = data[key]
            if isinstance(val, list):
                out = []
                for item in val:
                    if isinstance(item, (int, float)):
                        out.append(int(item))
                    if isinstance(item, str) and item.isdigit():
                        out.append(int(item))
                return out
    return []