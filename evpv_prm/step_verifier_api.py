# -*- coding: utf-8 -*-
"""
Step-level verification on VisualProcessBench via remote API.

This script evaluates each solution step in VisualProcessBench by calling a
remote vision-language model API. For each sample it runs two evaluation
pipelines:

  (A) Caption-then-judge: first generates a structured visual description of
      the image (caption stage), then uses that description as additional
      context when judging each step.
  (B) Direct-judge: judges each step using only the image and question,
      without a prior caption stage.

Key features:
  - Multi-threaded execution (8 workers by default) for high API-IO throughput.
  - API-level retry with exponential backoff + jitter for 429 rate-limiting.
  - Resume support: completed samples are identified by a unique UID and
    skipped on restart.
  - Thread-safe, real-time JSONL output (one line per completed sample).

Input:
  - VisualProcessBench JSONL file (each record contains question, image path,
    and response.steps).

Output:
  - JSONL file with two added fields per record:
      "step_predict_caption_then_judge": List[int]  (pipeline A predictions)
      "step_predict_direct":            List[int]  (pipeline B predictions)
      "structured_vision_api":          dict        (caption-stage output)
"""

import os
import json
import time
import base64
import hashlib
import random
import threading
from typing import Any, Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from .prompts import (
    STRUCTURED_VISION_SYSTEM,
    JUDGE_BATCH_SYSTEM,
    structured_vision_user,
)
from .utils import (
    abs_image_paths,
    parse_vision_json,
    parse_int_array_1_or_minus1,
    atomic_append_jsonl_threadsafe,
)


# ----------------------- API Configuration -----------------------
# Replace the placeholder values below with your actual API endpoint and
# authentication token before running.

API_URL = "https://your-api-endpoint/v1/api/chat"
API_HEADERS = {
    "Content-Type": "application/json",
    "token": "YOUR_API_TOKEN_HERE",
}

API_DEFAULT_PAYLOAD_FIELDS = {
    "tag": "evpv_prm_evaluation",
    "app": "evpv_prm",
    "user_id": "YOUR_USER_ID",
    "access_key": "YOUR_ACCESS_KEY",
    "quota_id": "YOUR_QUOTA_ID",
}

DEFAULT_MODEL = "qwen2.5-vl-72b-instruct"

API_RETRY       = 3
API_TIMEOUT     = 120
API_BACKOFF_BASE = 1.0   # seconds
API_BACKOFF_MAX  = 20.0  # seconds

_thread_local = threading.local()


# ----------------------- API Client -----------------------

def get_session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        _thread_local.session = s
    return s


def _extract_standard_code(resp_dict: Dict[str, Any]) -> Optional[int]:
    """Extract standard_code from various response structures (e.g. 429)."""
    if not isinstance(resp_dict, dict):
        return None
    sc = resp_dict.get("standard_code")
    if isinstance(sc, int):
        return sc
    data = resp_dict.get("data")
    if isinstance(data, dict) and isinstance(data.get("standard_code"), int):
        return data["standard_code"]
    return None


def requestgpt_once(
    prompt: List[Dict[str, Any]],
    params: Dict[str, Any],
    model: str = DEFAULT_MODEL,
    timeout: int = API_TIMEOUT,
) -> Tuple[Optional[str], Dict[str, Any]]:
    payload = {
        "model": model,
        "prompt": prompt,
        "params": params,
        **API_DEFAULT_PAYLOAD_FIELDS,
    }
    try:
        sess = get_session()
        resp = sess.post(API_URL, headers=API_HEADERS, data=json.dumps(payload), timeout=timeout)
        rd = resp.json()
    except Exception as e:
        return None, {"_client_exception": repr(e)}

    if "data" in rd and isinstance(rd["data"], dict) and "message" in rd["data"]:
        return rd["data"]["message"], rd
    return None, rd


def requestgpt_with_retry(
    prompt: List[Dict[str, Any]],
    params: Dict[str, Any],
    model: str = DEFAULT_MODEL,
    timeout: int = API_TIMEOUT,
    retry: int = API_RETRY,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Retry on: network exceptions, unexpected response structure, 429 rate-limits.
    Uses exponential backoff with jitter for rate-limit errors.
    """
    last_rd: Dict[str, Any] = {}
    for attempt in range(retry + 1):
        text, rd = requestgpt_once(prompt=prompt, params=params, model=model, timeout=timeout)
        last_rd = rd

        if text is not None:
            return text, rd

        sc = _extract_standard_code(rd)
        is_rate_limited = sc in (429, 429005) or "429" in json.dumps(rd, ensure_ascii=False)

        if attempt >= retry:
            print("[API] Request failed (giving up):", rd)
            return None, rd

        sleep_s = (
            min(API_BACKOFF_MAX, API_BACKOFF_BASE * (2 ** attempt)) * (0.7 + 0.6 * random.random())
            if is_rate_limited
            else min(5.0, 0.5 * (attempt + 1))
        )
        print(f"[API] Retry {attempt+1}/{retry}, sleep={sleep_s:.2f}s, resp={rd}")
        time.sleep(sleep_s)

    return None, last_rd


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ----------------------- Prompt builder -----------------------

def build_judge_batch_user_content(
    question: str,
    steps_all: List[str],
    batch_start0: int,
    batch_steps: List[str],
    vision_json: Optional[Dict[str, Any]] = None,
    history_max_chars: int = 8000,
) -> str:
    history_steps = steps_all[:batch_start0]
    hist_lines = [f"{i+1}. {s}" for i, s in enumerate(history_steps)]

    if history_max_chars and history_max_chars > 0:
        acc, kept = 0, []
        for line in reversed(hist_lines):
            acc += len(line) + 1
            if acc > history_max_chars:
                break
            kept.append(line)
        hist_lines = list(reversed(kept))

    start_1 = batch_start0 + 1
    end_1   = batch_start0 + len(batch_steps)

    parts: List[str] = []
    if vision_json is not None:
        parts.append("Structured image description (JSON): " + json.dumps(vision_json, ensure_ascii=False))

    parts.append("Question:\n" + question)

    if hist_lines:
        parts.append("Previous solution steps (for context; assume they are part of the student's full solution):")
        parts.extend(hist_lines)
    else:
        parts.append("Previous solution steps: (none)")

    parts.append(
        f"You will evaluate EXACTLY {len(batch_steps)} steps: steps {start_1} to {end_1}.\n"
        "Important:\n"
        "- Use the image + question + previous steps as context.\n"
        "- Judge each CURRENT step as correct/incorrect relative to the full solution so far.\n"
        "- If a step contradicts the image/question/previous steps, mark it -1.\n"
    )
    parts.append(
        "Return format requirements:\n"
        "1) Output ONLY a JSON array.\n"
        f"2) The array length MUST be {len(batch_steps)}.\n"
        "3) Each element MUST be exactly 1 or -1.\n"
        "4) Do NOT output any other text, keys, markdown, or explanation.\n"
        "If you cannot comply, output an array of the required length anyway."
    )
    parts.append("Current steps to evaluate:")
    for j, s in enumerate(batch_steps):
        parts.append(f"{start_1 + j}. {s}")

    return "\n".join(parts)


# ----------------------- IO / resume helpers -----------------------

def sample_uid(obj: Dict[str, Any], fallback_idx: int) -> str:
    for k in ("id", "uid", "sample_id", "qid"):
        if k in obj and obj[k] is not None:
            return f"{k}:{obj[k]}"
    base = {
        "image":    obj.get("image", None),
        "question": obj.get("question", ""),
        "steps":    (obj.get("response") or {}).get("steps", None),
    }
    h = hashlib.md5(json.dumps(base, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return f"md5:{h}:{fallback_idx}"


def load_done_uids(out_path: str) -> Set[str]:
    done: Set[str] = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            uid = obj.get("_uid")
            if uid:
                done.add(uid)
            else:
                for k in ("id", "uid", "sample_id", "qid"):
                    if k in obj and obj[k] is not None:
                        done.add(f"{k}:{obj[k]}")
                        break
    return done


# ----------------------- API pipeline calls -----------------------

def call_structured_vision_api(
    image_path: str,
    question: str,
    model: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    b64 = image_to_base64(image_path)
    msg = [
        {"role": "system", "content": STRUCTURED_VISION_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text",      "text": structured_vision_user(question)},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        },
    ]
    text, _ = requestgpt_with_retry(prompt=msg, params=params, model=model)
    return parse_vision_json(text)


def batch_iter(lst: List[str], batch_size: int):
    for i in range(0, len(lst), batch_size):
        yield i, lst[i:i + batch_size]


def call_judge_batch_api(
    image_path: str,
    question: str,
    steps_all: List[str],
    batch_start0: int,
    batch_steps: List[str],
    model: str,
    params: Dict[str, Any],
    vision_json: Optional[Dict[str, Any]] = None,
    retry: int = 2,
    sleep_between: float = 0.2,
    history_max_chars: int = 8000,
) -> List[int]:
    """
    Judge one batch of steps.
    The underlying requestgpt_with_retry handles API errors and 429s (>=3 retries).
    This function adds an extra retry layer for malformed output (wrong array length).
    """
    b64 = image_to_base64(image_path)

    for attempt in range(retry + 1):
        user_text = build_judge_batch_user_content(
            question=question,
            steps_all=steps_all,
            batch_start0=batch_start0,
            batch_steps=batch_steps,
            vision_json=vision_json,
            history_max_chars=history_max_chars,
        )
        msg = [
            {"role": "system", "content": JUDGE_BATCH_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text",      "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            },
        ]

        text, _ = requestgpt_with_retry(prompt=msg, params=params, model=model)
        arr = parse_int_array_1_or_minus1(text)
        if arr is not None and len(arr) == len(batch_steps):
            return arr

        if attempt < retry:
            time.sleep(sleep_between)
            params = dict(params)
            params["temperature"] = min(params.get("temperature", 0.2), 0.2)
            params["topp"]        = min(params.get("topp", 0.9),        0.9)

    return [-1] * len(batch_steps)


# ----------------------- Worker -----------------------

def process_one_sample(
    idx: int,
    line: str,
    img_root: str,
    out_path: str,
    model: str,
    vision_params: Dict[str, Any],
    judge_params: Dict[str, Any],
    batch_size: int,
    judge_retry: int,
    history_max_chars: int,
) -> None:
    line = line.strip()
    if not line:
        return

    obj = json.loads(line)
    uid = sample_uid(obj, fallback_idx=idx)
    obj["_uid"] = uid

    img_paths = abs_image_paths(obj.get("image"), img_root=img_root)
    if not img_paths or not os.path.exists(img_paths[0]):
        obj.update({"step_predict_caption_then_judge": [], "step_predict_direct": [],
                    "step_predict_error": "image_missing"})
        atomic_append_jsonl_threadsafe(out_path, obj)
        return
    image_path = img_paths[0]

    question = obj.get("question", "")
    steps = (obj.get("response") or {}).get("steps", [])
    if not isinstance(steps, list) or len(steps) == 0:
        obj.update({"step_predict_caption_then_judge": [], "step_predict_direct": [],
                    "step_predict_error": "no_steps"})
        atomic_append_jsonl_threadsafe(out_path, obj)
        return
    steps = [str(s) for s in steps]

    # Pipeline A: caption → judge
    try:
        vision_json = call_structured_vision_api(image_path, question, model, vision_params)
    except Exception as e:
        vision_json = {"caption": "", "key_facts": [], "uncertain": [f"vision_call_failed:{type(e).__name__}"]}

    preds_a: List[int] = []
    for start0, batch in batch_iter(steps, batch_size):
        preds_a.extend(call_judge_batch_api(
            image_path, question, steps, start0, batch, model, judge_params,
            vision_json=vision_json, retry=judge_retry, history_max_chars=history_max_chars,
        ))

    # Pipeline B: direct judge (no caption)
    preds_b: List[int] = []
    for start0, batch in batch_iter(steps, batch_size):
        preds_b.extend(call_judge_batch_api(
            image_path, question, steps, start0, batch, model, judge_params,
            vision_json=None, retry=judge_retry, history_max_chars=history_max_chars,
        ))

    obj["structured_vision_api"]          = vision_json
    obj["step_predict_caption_then_judge"] = preds_a
    obj["step_predict_direct"]             = preds_b

    atomic_append_jsonl_threadsafe(out_path, obj)
    print(
        f"[done idx={idx}] steps={len(steps)} | "
        f"A(1/-1)={preds_a.count(1)}/{preds_a.count(-1)} | "
        f"B(1/-1)={preds_b.count(1)}/{preds_b.count(-1)}"
    )


# ----------------------- Main -----------------------

def main():
    # ---- Path configuration ----
    # Update these paths to point to your local data and output directories.
    in_path  = "data/visualprocessbench/visualprocessbench.jsonl"
    img_root = "data/visualprocessbench/images"
    out_path = "output/step_predictions_api.jsonl"

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")
    if not os.path.exists(img_root):
        raise FileNotFoundError(f"Image root directory not found: {img_root}")

    MODEL             = DEFAULT_MODEL
    BATCH_SIZE        = 5
    JUDGE_RETRY       = 2
    HISTORY_MAX_CHARS = 8000
    WORKERS           = 8

    vision_params = {"topp": 0.9, "temperature": 0.4, "maxtokens": 512}
    judge_params  = {"topp": 0.9, "temperature": 0.0, "maxtokens": 256}

    done_uids = load_done_uids(out_path)
    if done_uids:
        print(f"[Resume] loaded {len(done_uids)} completed UIDs from {out_path}")

    futures: List = []
    submitted, skipped = 0, 0

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        with open(in_path, "r", encoding="utf-8") as fin:
            for idx, line in enumerate(fin):
                line_strip = line.strip()
                if not line_strip:
                    continue
                try:
                    tmp_obj = json.loads(line_strip)
                except Exception:
                    continue

                uid = sample_uid(tmp_obj, fallback_idx=idx)
                if uid in done_uids:
                    skipped += 1
                    continue

                fut = ex.submit(
                    process_one_sample,
                    idx, line_strip, img_root, out_path,
                    MODEL, vision_params, judge_params,
                    BATCH_SIZE, JUDGE_RETRY, HISTORY_MAX_CHARS,
                )
                futures.append(fut)
                submitted += 1

        print(
            f"[Submit] submitted={submitted}, skipped={skipped}, workers={WORKERS}, "
            f"API_RETRY={API_RETRY}, JUDGE_RETRY={JUDGE_RETRY}"
        )

        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print("[Worker Exception]", repr(e))

    print("All done. Output:", out_path)


if __name__ == "__main__":
    main()
