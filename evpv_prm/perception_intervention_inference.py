# -*- coding: utf-8 -*-
"""
Perception intervention experiments: causal analysis of visual evidence quality.

This script implements four controlled perception conditions to measure how
strongly multimodal reasoning accuracy depends on visual perception quality:

  Condition 1 - Normal (image + question):
      The policy model receives the original image and question with no
      extra context. This is the standard baseline.

  Condition 2 - Oracle perception (image + question + structured description):
      A strong API model generates an accurate structured description of the
      image. The description is appended to the prompt alongside the image,
      providing the policy with high-quality grounded visual facts.

  Condition 3 - Noisy perception (question + corrupted description, NO image):
      The structured description from condition 2 is deliberately corrupted
      by an API model (injecting subtle but misleading errors). The policy
      receives only the corrupted description—no image—to isolate the effect
      of incorrect visual premises.

  Condition 4 - Text-only (question only, NO image):
      The policy receives no visual input at all, establishing a lower bound
      on performance without any perception.

Results are written to a JSONL file in real time (one record per question).
Resume support: previously processed lines are detected by line count and
skipped on restart.

Requirements:
  pip install lmdeploy requests pillow

Input:
  - Benchmark JSONL file (each record: `question`, `image`, `answer`).

Output:
  - JSONL file, each input record augmented with:
      "perception_intervention_exps": {
          "cond1": {"reasoning": str, "final": str},   # normal
          "cond2": {"reasoning": str, "final": str},   # oracle perception
          "cond3": {"reasoning": str, "final": str},   # noisy perception
          "cond4": {"reasoning": str, "final": str},   # text-only
      }
      "perception_intervention_meta": {
          "errors": [...],
          "api_raw": { "structured": ..., "noisy": ... }
      }
"""

import os
import re
import json
import time
import base64
import traceback
import multiprocessing
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image


# =============================================================================
# 1) Remote API client
#    Replace placeholder values with your actual API credentials before running.
# =============================================================================

def call_remote_api(
    prompt: List,
    params: Dict,
    model: str = "qwen3-vl-235b-a22b-instruct",
) -> Tuple[Optional[str], Dict]:
    """Call a remote VLM API and return (response_text, raw_response_dict)."""
    url = "https://your-api-endpoint/v1/api/chat"
    headers = {
        "Content-Type": "application/json",
        "token": "YOUR_API_TOKEN_HERE",
    }
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "tag": "evpv_perception_intervention",
        "app": "evpv_prm",
        "user_id": "YOUR_USER_ID",
        "access_key": "YOUR_ACCESS_KEY",
        "quota_id": "YOUR_QUOTA_ID",
        "params": params,
    })

    try:
        response     = requests.post(url, headers=headers, data=payload, timeout=180)
        responsedict = response.json()
    except Exception as e:
        print(f"[API] Request error: {e}")
        responsedict = {}

    if "data" in responsedict and "message" in responsedict["data"]:
        return responsedict["data"]["message"], responsedict
    else:
        print("[API] Unexpected response:", responsedict)
        return None, responsedict


def image_to_base64_data_url(image_path: str) -> str:
    """Convert a local image file to a base64 data URL."""
    ext  = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "bmp": "bmp"}.get(ext, "png")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


from .prompts import (
    STRUCTURED_VISION_SYSTEM,
    NOISE_INJECT_SYSTEM,
    structured_vision_user,
    noise_inject_user,
    local_solve_prompt,
)


# =============================================================================
# 3) Local policy model (InternVL via lmdeploy)
# =============================================================================

@dataclass
class LocalModelConfig:
    # Update model_dir to point to your local InternVL model.
    model_dir:            str   = "/path/to/InternVL-model"
    tensor_parallel_size: int   = 1
    session_len:          int   = 8192
    temperature:          float = 0.2
    top_p:                float = 0.9
    max_tokens:           int   = 2048


class LocalPolicyModel:
    """Thin wrapper around lmdeploy pipeline for InternVL inference."""

    def __init__(self, cfg: LocalModelConfig):
        from lmdeploy import pipeline, TurbomindEngineConfig

        if not os.path.exists(cfg.model_dir):
            raise FileNotFoundError(f"Model directory not found: {cfg.model_dir}")

        self.cfg = cfg
        print(f"Loading policy model from: {cfg.model_dir}")
        print(f"TP={cfg.tensor_parallel_size}, session_len={cfg.session_len}")

        backend_config = TurbomindEngineConfig(
            session_len=cfg.session_len,
            tp=cfg.tensor_parallel_size,
        )
        self.pipe = pipeline(cfg.model_dir, backend_config=backend_config)

    def generate(self, prompt_text: str, image: Optional[Image.Image]) -> str:
        model_input = (prompt_text, image) if image is not None else prompt_text
        response    = self.pipe(
            model_input,
            max_new_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
        )
        return response.text


# =============================================================================
# 4) Utilities
# =============================================================================

def load_json_from_text(text: str) -> Optional[dict]:
    """Extract and parse the first JSON object from model output."""
    if text is None:
        return None
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        try:
            return json.loads(t)
        except Exception:
            pass
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def pil_load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


# =============================================================================
# 5) API call wrappers
# =============================================================================

def call_api_structured_vision(
    question: str, image_path: str, api_model_name: str
) -> Tuple[Optional[dict], Dict]:
    """Generate oracle structured description via remote API."""
    data_url      = image_to_base64_data_url(image_path)
    message_prompt = [
        {"role": "system", "content": STRUCTURED_VISION_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": structured_vision_user(question)},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]
    params       = {"topp": 0.95, "temperature": 0.2, "maxtokens": 2048}
    response_text, raw = call_remote_api(prompt=message_prompt, params=params, model=api_model_name)
    if response_text is None:
        return None, raw
    return load_json_from_text(response_text), raw


def call_api_noise_inject(
    structured: dict, api_model_name: str
) -> Tuple[Optional[dict], Dict]:
    """Corrupt the structured description via remote API (condition 3)."""
    message_prompt = [
        {"role": "system", "content": NOISE_INJECT_SYSTEM},
        {"role": "user",   "content": noise_inject_user(structured)},
    ]
    params       = {"topp": 0.95, "temperature": 0.8, "maxtokens": 2048}
    response_text, raw = call_remote_api(prompt=message_prompt, params=params, model=api_model_name)
    if response_text is None:
        return None, raw
    return load_json_from_text(response_text), raw


# =============================================================================
# 6) Main experiment runner
# =============================================================================

def run(
    input_jsonl:        str,
    images_root:        str,
    output_jsonl:       str,
    api_model_name:     str   = "qwen3-vl-235b-a22b-instruct",
    resume:             bool  = True,
    sleep_between_api:  float = 0.0,
):
    """
    Run the four perception-intervention conditions for every question in
    `input_jsonl` and write results to `output_jsonl`.
    """
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)

    # Resume: count already-written lines
    done = 0
    if resume and os.path.exists(output_jsonl):
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for _ in f:
                done += 1
        print(f"[Resume] {done} lines already written; skipping them.")

    print("Initializing local policy model ...")
    local = LocalPolicyModel(LocalModelConfig())
    print("Policy model ready.")

    with open(input_jsonl, "r", encoding="utf-8") as fin, \
         open(output_jsonl, "a", encoding="utf-8") as fout:

        for idx, line in enumerate(fin):
            if idx < done:
                continue
            line = line.strip()
            if not line:
                continue

            meta: Dict[str, Any] = {"errors": [], "api_raw": {}}
            exp:  Dict[str, Any] = {"cond1": None, "cond2": None, "cond3": None, "cond4": None}

            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[Skip] Line {idx}: invalid JSON: {e}")
                continue

            question = obj.get("question", "")
            rel_img  = obj.get("image", "")
            img_path = os.path.join(images_root, rel_img) if rel_img else ""
            has_image = bool(img_path) and os.path.exists(img_path)
            raw_image = pil_load_image(img_path) if has_image else None

            # ---- Condition 1: Normal (image + question) ----
            try:
                p1 = local_solve_prompt(question, extra_desc=None)
                t1 = local.generate(p1, image=raw_image)
                exp["cond1"] = load_json_from_text(t1) or {"reasoning": "", "final": "", "_raw": t1}
            except Exception as e:
                meta["errors"].append({"cond": 1, "err": str(e), "trace": traceback.format_exc()})

            # ---- Condition 2: Oracle perception (image + question + structured desc) ----
            structured = None
            if has_image:
                try:
                    structured, raw = call_api_structured_vision(question, img_path, api_model_name)
                    meta["api_raw"]["structured"] = raw
                    if sleep_between_api > 0:
                        time.sleep(sleep_between_api)
                except Exception as e:
                    meta["errors"].append({"cond": 2, "stage": "api_structured",
                                           "err": str(e), "trace": traceback.format_exc()})

            try:
                p2 = local_solve_prompt(question, extra_desc=structured)
                t2 = local.generate(p2, image=raw_image)
                exp["cond2"] = load_json_from_text(t2) or {"reasoning": "", "final": "", "_raw": t2}
            except Exception as e:
                meta["errors"].append({"cond": 2, "stage": "local",
                                       "err": str(e), "trace": traceback.format_exc()})

            # ---- Condition 3: Noisy perception (no image, corrupted desc) ----
            noisy = None
            if structured is not None:
                try:
                    noisy, raw = call_api_noise_inject(structured, api_model_name)
                    meta["api_raw"]["noisy"] = raw
                    if sleep_between_api > 0:
                        time.sleep(sleep_between_api)
                except Exception as e:
                    meta["errors"].append({"cond": 3, "stage": "api_noise",
                                           "err": str(e), "trace": traceback.format_exc()})

            try:
                p3 = local_solve_prompt(question, extra_desc=noisy)
                t3 = local.generate(p3, image=None)   # no image for condition 3
                exp["cond3"] = load_json_from_text(t3) or {"reasoning": "", "final": "", "_raw": t3}
            except Exception as e:
                meta["errors"].append({"cond": 3, "stage": "local",
                                       "err": str(e), "trace": traceback.format_exc()})

            # ---- Condition 4: Text-only (no image, no description) ----
            try:
                p4 = local_solve_prompt(question, extra_desc=None)
                t4 = local.generate(p4, image=None)   # no image for condition 4
                exp["cond4"] = load_json_from_text(t4) or {"reasoning": "", "final": "", "_raw": t4}
            except Exception as e:
                meta["errors"].append({"cond": 4, "err": str(e), "trace": traceback.format_exc()})

            obj["perception_intervention_exps"]  = exp
            obj["perception_intervention_meta"]  = meta

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fout.flush()
            os.fsync(fout.fileno())

            if (idx + 1) % 10 == 0:
                print(f"[Progress] Processed {idx + 1} samples.")

    print(f"[Done] Results saved to: {output_jsonl}")


# =============================================================================
# 7) Entry point
# =============================================================================

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # ---- Path configuration ----
    # Update these paths to point to your local data and output directories.
    INPUT_JSONL  = "data/benchmark/benchmark_questions.jsonl"
    IMAGES_ROOT  = "data/benchmark/images"
    OUTPUT_JSONL = "output/perception_intervention_results.jsonl"

    run(
        input_jsonl=INPUT_JSONL,
        images_root=IMAGES_ROOT,
        output_jsonl=OUTPUT_JSONL,
        api_model_name="qwen3-vl-235b-a22b-instruct",
        resume=True,
        sleep_between_api=0.0,
    )
