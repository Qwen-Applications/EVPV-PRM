# -*- coding: utf-8 -*-
"""
evpv_prm/utils.py
-----------------
Shared utilities used across the EVPV-PRM pipeline.

Sections
--------
1. JSON / text parsing  — robust parsing of model outputs (JSON objects, arrays, int labels)
2. Image helpers        — path resolution and PIL image loading
3. IO helpers           — atomic JSONL append (thread-safe and single-threaded variants)
"""

import os
import re
import ast
import json
import threading
from typing import Any, Dict, List, Optional

from PIL import Image


# =============================================================================
# 1. JSON / text parsing
# =============================================================================

_float_re = re.compile(r"[-+]?(?:\d+\.\d+|\d+\.|\.\d+|\d+)(?:[eE][-+]?\d+)?")
_int_re   = re.compile(r"[-+]?\d+")
_step_line_re = re.compile(r"^\s*Step\s*(\d+)\s*[:：.\-]\s*(.*)\s*$", re.IGNORECASE)
_JSON_BLOCK_RE  = re.compile(r"\{.*\}", re.S)
_JSON_ARRAY_RE  = re.compile(r"\[[\s\S]*\]")


def strip_code_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences from model output."""
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def extract_first_balanced_json(text: str, want_array: bool) -> Optional[str]:
    """Return the first balanced JSON array or object substring from *text*."""
    if not text:
        return None
    t = strip_code_fences(text)

    open_ch  = "[" if want_array else "{"
    close_ch = "]" if want_array else "}"

    start = t.find(open_ch)
    if start == -1:
        return None

    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return t[start:i + 1]
    return None


def json_loads_lenient(s: str) -> Optional[Any]:
    """
    Parse JSON with progressive fallback:
      1. json.loads
      2. trailing-comma fix + json.loads
      3. ast.literal_eval
      4. single→double quote replacement
    """
    if not s:
        return None
    s0 = s.strip()
    if not s0:
        return None
    try:
        return json.loads(s0)
    except Exception:
        pass
    s1 = re.sub(r",\s*([}\]])", r"\1", s0)
    try:
        return json.loads(s1)
    except Exception:
        pass
    try:
        return ast.literal_eval(s1)
    except Exception:
        pass
    if "'" in s1 and '"' not in s1:
        try:
            return json.loads(s1.replace("'", '"'))
        except Exception:
            pass
    return None


def parse_json_object(text: str) -> Dict[str, Any]:
    """
    Extract and parse the first JSON *object* from model output.
    Returns an empty dict on failure.
    """
    if not text:
        return {}
    t = (text or "").strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = _JSON_BLOCK_RE.search(t)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def parse_vision_json(text: str) -> Dict[str, Any]:
    """
    Parse structured vision JSON.  Returns a placeholder dict on failure so
    downstream code always receives a dict with `caption`, `key_facts`, and
    `uncertain` keys.
    """
    result = parse_json_object(text)
    if result:
        return result
    return {"caption": "", "key_facts": [], "uncertain": ["parse_failed"]}


def parse_json_array(raw_text: str) -> Optional[List[Any]]:
    """
    Try to parse a JSON array from raw model output.
    Also checks for dicts containing common list-valued keys.
    """
    if not raw_text:
        return None
    t = strip_code_fences(raw_text)

    cand = extract_first_balanced_json(t, want_array=True)
    if cand:
        parsed = json_loads_lenient(cand)
        if isinstance(parsed, list):
            return parsed

    cand_obj = extract_first_balanced_json(t, want_array=False)
    if cand_obj:
        obj = json_loads_lenient(cand_obj)
        if isinstance(obj, dict):
            for k in ("steps", "result", "output", "step_list"):
                if k in obj and isinstance(obj[k], list):
                    return obj[k]
    return None


def parse_int_array_1_or_minus1(text: str) -> Optional[List[int]]:
    """
    Parse a JSON array whose elements are all 1 or -1.
    Returns None if parsing fails or any element is not 1 / -1.
    """
    if text is None:
        return None
    t = text.strip()

    def _coerce(arr):
        out = []
        for x in arr:
            if x in (1, -1):
                out.append(int(x))
            elif isinstance(x, str) and x.strip() in ("1", "-1"):
                out.append(int(x.strip()))
            else:
                return None
        return out

    try:
        arr = json.loads(t)
        if isinstance(arr, list):
            return _coerce(arr)
    except Exception:
        pass

    m = _JSON_ARRAY_RE.search(t)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
        if isinstance(arr, list):
            return _coerce(arr)
    except Exception:
        pass
    return None


def parse_step_score(raw_out: str) -> int:
    """
    Parse model output into 1 (correct) or -1 (incorrect).

    Priority: exact match → strict JSON int → regex for -1 → regex for 1
              → first integer → conservative default -1.
    """
    if not raw_out:
        return -1
    t = strip_code_fences(raw_out).strip()
    if t in ("1", "-1"):
        return int(t)

    parsed = json_loads_lenient(t)
    if not isinstance(parsed, bool) and isinstance(parsed, (int, float)):
        v = int(parsed)
        if v in (1, -1):
            return v

    if re.search(r"(?<!\d)-1(?!\d)", t):
        return -1
    if re.search(r"(?<!\d)1(?!\d)", t):
        return 1

    m = _int_re.search(t)
    if m:
        try:
            x = int(m.group(0))
            if x >= 1:
                return 1
            if x <= -1:
                return -1
        except Exception:
            pass

    return -1


# Alias used by several scripts
parse_label_1_or_minus1 = parse_step_score


# =============================================================================
# 2. Image helpers
# =============================================================================

def abs_image_paths(image_field: Any, img_root: str) -> List[str]:
    """
    Resolve one or more image paths (string or list) relative to *img_root*.
    Strips a leading ``images/`` prefix if present.
    """
    if image_field is None:
        return []
    imgs = [image_field] if isinstance(image_field, str) else \
           image_field    if isinstance(image_field, list) else \
           [str(image_field)]
    out = []
    for p in imgs:
        if os.path.isabs(p):
            out.append(p)
        else:
            rp = p[len("images/"):] if p.startswith("images/") else p
            out.append(os.path.normpath(os.path.join(img_root, rp)))
    return out


def safe_load_rgb(image_path: str) -> Image.Image:
    """Load an image and convert it to RGB mode."""
    return Image.open(image_path).convert("RGB")


# =============================================================================
# 3. IO helpers
# =============================================================================

_write_lock = threading.Lock()


def atomic_append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    """Append *obj* as a JSON line to *path*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def atomic_append_jsonl_threadsafe(path: str, obj: Dict[str, Any]) -> None:
    """Thread-safe version of :func:`atomic_append_jsonl`."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
