# -*- coding: utf-8 -*-
"""
Step-level verification on VisualProcessBench using a locally deployed model (vLLM).

This script evaluates each solution step in VisualProcessBench by calling a
locally deployed vision-language model via vLLM. For each sample it:

  1. Generates a structured visual description (caption) of the image.
  2. Uses the caption + image + question + step history as context to judge
     each reasoning step (output: 1 = correct, -1 = incorrect).
  3. Saves step-level predictions in real time (one JSONL line per sample).

Resume support: previously processed samples are identified by their index in
the input file and skipped on restart.

Requirements:
  pip install vllm transformers pillow

Input:
  - VisualProcessBench JSONL file
  - Corresponding images directory

Output:
  - JSONL file, each record augmented with:
      "step_predict":     List[int]   (1 or -1 per step)
      "structured_vision": dict        (caption-stage output)
"""

import os
import json
import multiprocessing
from typing import Any, Dict, List

from .prompts import STRUCTURED_VISION_SYSTEM, JUDGE_PREFIX, structured_vision_user
from .utils import (
    abs_image_paths,
    safe_load_rgb,
    parse_vision_json,
    parse_label_1_or_minus1,
    atomic_append_jsonl,
)


# ----------------------- Prompt builder -----------------------

def build_judge_user_content(
    question: str,
    vision_json: Dict[str, Any],
    prev_steps: List[str],
    cur_step: str,
) -> str:
    vision_str = json.dumps(vision_json, ensure_ascii=False)
    parts = [
        JUDGE_PREFIX,
        f"Structured image description (JSON): {vision_str}",
        f"Question: {question}",
    ]
    if prev_steps:
        parts.append("Previous solution steps:")
        parts.extend([f"{i+1}. {s}" for i, s in enumerate(prev_steps)])
    parts.append("Step to evaluate:")
    parts.append(cur_step)
    return "\n".join(parts)


# ----------------------- vLLM Calls -----------------------

def vllm_generate_one(llm, processor, raw_image, messages, sampling_params) -> str:
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = llm.generate(
        prompts=[{"prompt": text_prompt, "multi_modal_data": {"image": raw_image}}],
        sampling_params=sampling_params,
    )
    return outputs[0].outputs[0].text


def get_structured_vision(llm, processor, raw_image, question: str, sampling_params) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": STRUCTURED_VISION_SYSTEM}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": structured_vision_user(question)}]},
    ]
    out = vllm_generate_one(llm, processor, raw_image, messages, sampling_params)
    return parse_vision_json(out)


def judge_step(
    llm,
    processor,
    raw_image,
    question: str,
    vision_json: Dict[str, Any],
    prev_steps: List[str],
    cur_step: str,
    sampling_params,
) -> int:
    user_text = build_judge_user_content(question, vision_json, prev_steps, cur_step)
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}
    ]
    out = vllm_generate_one(llm, processor, raw_image, messages, sampling_params)
    label = parse_label_1_or_minus1(out)
    return label if label in (1, -1) else -1


# ----------------------- Main -----------------------

def main():
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    import torch  # noqa: F401
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    # ---- Path configuration ----
    # Update these paths to point to your local data and model directories.
    in_path   = "data/visualprocessbench/visualprocessbench.jsonl"
    img_root  = "data/visualprocessbench/images"
    out_path  = "output/step_predictions_local.jsonl"
    model_dir = "EVPV-PRM"

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")
    if not os.path.exists(img_root):
        raise FileNotFoundError(f"Image root directory not found: {img_root}")

    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=8096,
    )
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    vision_params = SamplingParams(max_tokens=512, temperature=0.2, top_p=0.9)
    judge_params  = SamplingParams(max_tokens=8,   temperature=0.0)

    done = 0
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            done = sum(1 for _ in f if _.strip())
        print(f"[Resume] {done} lines already in {out_path}, will skip them.")

    with open(in_path, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            if idx < done:
                continue
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            img_paths = abs_image_paths(obj.get("image"), img_root=img_root)
            if not img_paths or not os.path.exists(img_paths[0]):
                obj["step_predict"] = []
                obj["step_predict_error"] = "image_missing"
                atomic_append_jsonl(out_path, obj)
                continue

            raw_image = safe_load_rgb(img_paths[0])
            question  = obj.get("question", "")
            steps     = (obj.get("response") or {}).get("steps", [])

            if not isinstance(steps, list) or len(steps) == 0:
                obj["step_predict"] = []
                obj["step_predict_error"] = "no_steps"
                atomic_append_jsonl(out_path, obj)
                continue

            try:
                vision_json = get_structured_vision(llm, processor, raw_image, question, vision_params)
            except Exception as e:
                vision_json = {"caption": "", "key_facts": [], "uncertain": [f"vision_call_failed:{type(e).__name__}"]}

            preds: List[int] = []
            prev:  List[str] = []
            for s in steps:
                cur = str(s)
                try:
                    lab = judge_step(llm, processor, raw_image, question, vision_json, prev, cur, judge_params)
                except Exception:
                    lab = -1
                preds.append(int(lab))
                prev.append(cur)

            obj["step_predict"]      = preds
            obj["structured_vision"] = vision_json

            atomic_append_jsonl(out_path, obj)
            print(f"[{idx}] saved, steps={len(steps)}, preds(1/-1)={preds.count(1)}/{preds.count(-1)}")

    print("All done. Output:", out_path)


if __name__ == "__main__":
    main()
