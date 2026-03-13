# -*- coding: utf-8 -*-
"""
EVPV-PRM step-level evaluation on perception-intervention policy outputs.

This script applies the EVPV-PRM step-level verifier to the policy outputs
generated under the four perception-intervention conditions (see
`perception_intervention_inference.py`).

For every sample the pipeline:
  1. Splits each condition's reasoning text into individual steps (via the
     PRM model in JSON-output mode; falls back to rule-based parsing on
     failure).
  2. Scores each step as 1 (correct) or -1 (incorrect) using the PRM model,
     conditioned on the structured image description that was already stored
     in the input file.

All four conditions (cond1–cond4) are processed in a single batched vLLM
pass per sample to maximise GPU utilisation.

Features:
  - Batched multimodal vLLM inference (InternVL-compatible).
  - Robust JSON parsing with rule-based fallback.
  - Resume support: records already written to the output file are skipped
    automatically.
  - Real-time JSONL output (append mode, flushed after every sample).

Input:
  - JSONL file produced by `perception_intervention_inference.py`.
    Each record must contain:
      * "question": str
      * "image":    relative path to the image file
      * "perception_intervention_exps": dict with keys cond1–cond4, each
        having a "reasoning" field
      * "perception_intervention_meta.api_raw.structured.message": optional
        structured image description text (used as image_description in step
        scoring prompt)

Output:
  - JSONL file with each input record augmented:
      "perception_intervention_exps"[condX]["prm_eval"] = {
          "raw_split_output": str,
          "steps":            [str, ...],
          "raw_judge_outputs":[str, ...],
          "step_scores":      [1 or -1, ...],
      }

Requirements:
  pip install vllm transformers pillow tqdm
"""

import os
import re
import json
import multiprocessing
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from .prompts import PROMPT_SPLIT_REASONING_TO_STEPS, PROMPT_STEP_JUDGE
from .utils import strip_code_fences, parse_json_array, parse_step_score


# =============================================================================
# Configuration
# =============================================================================

MODEL_DIR   = "EVPV-PRM"   # path to the local EVPV-PRM model checkpoint

INPUT_FILE_PATH  = "output/perception_intervention_results.jsonl"
IMAGE_ROOT       = "data/benchmark/images"
OUTPUT_DIR       = "output"
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, "perception_intervention_prm_eval.jsonl")

INFERENCE_BATCH_SIZE          = 8
TEMPERATURE                   = 0.1
MAX_TOKENS_SPLIT              = 1024
MAX_TOKENS_JUDGE              = 128
HISTORY_MAX_STEPS_FOR_JUDGE   = 12
CONDITION_KEYS                = ["cond1", "cond2", "cond3", "cond4"]


# =============================================================================
# JSON / text parsing utilities (delegated to utils.py)
# =============================================================================

_step_line_re = re.compile(r"^\s*Step\s*(\d+)\s*[:：.\-]\s*(.*)\s*$", re.IGNORECASE)


# =============================================================================
# Rule-based step splitter (fallback when JSON parsing fails)
# =============================================================================

def rule_split_steps(raw: str, fallback_reasoning: str) -> List[str]:
    """
    Extract individual steps from `raw` text using heuristic rules:
      1. Look for "Step k: …" pattern.
      2. Split by lines.
      3. Split by sentence-ending punctuation.
      4. Treat the whole text as one step.
    """
    text = strip_code_fences(raw).strip()
    if not text:
        text = (fallback_reasoning or "").strip()

    pattern = re.compile(r"(Step\s*\d+\s*[:：.\-]\s*)", re.IGNORECASE)
    if pattern.search(text):
        parts = pattern.split(text)
        steps = []
        i = 1
        while i + 1 < len(parts):
            head = parts[i].strip()
            body = parts[i + 1].strip()
            steps.append((head + " " + body).strip())
            i += 2
        steps = [s for s in steps if s]
        if steps:
            return normalize_steps(steps, fallback_reasoning)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2:
        return normalize_steps(lines, fallback_reasoning)

    chunks = re.split(r"[.。;；]\s*", text)
    chunks = [c.strip() for c in chunks if c.strip()]
    if len(chunks) >= 2:
        merged, buf = [], ""
        for c in chunks:
            if not buf:
                buf = c
            elif len(buf) < 40:
                buf = buf + "; " + c
            else:
                merged.append(buf)
                buf = c
        if buf:
            merged.append(buf)
        return normalize_steps(merged, fallback_reasoning)

    fb = (fallback_reasoning or text or "").strip()
    return [f"Step 1: {fb}" if fb else "Step 1:"]


def normalize_steps(step_list: List[Any], fallback_reasoning: str) -> List[str]:
    """Normalize a list of step texts to 'Step k: …' format with sequential numbering."""
    cleaned = [str(s).strip() for s in step_list if s is not None and str(s).strip()]
    if not cleaned:
        fb = (fallback_reasoning or "").strip()
        return [f"Step 1: {fb}" if fb else "Step 1:"]

    if any(_step_line_re.match(x) for x in cleaned):
        bodies = []
        for x in cleaned:
            m = _step_line_re.match(x)
            body = m.group(2).strip() if m else x.strip()
            if body:
                bodies.append(body)
        if not bodies:
            fb = (fallback_reasoning or "").strip()
            return [f"Step 1: {fb}" if fb else "Step 1:"]
        return [f"Step {i + 1}: {b}" for i, b in enumerate(bodies)]

    return [f"Step {i + 1}: {x}" for i, x in enumerate(cleaned)]


def truncate_history(history: List[str], max_steps: int) -> List[str]:
    if not max_steps or max_steps <= 0 or len(history) <= max_steps:
        return history
    return history[-max_steps:]


# =============================================================================
# vLLM multimodal inference helpers
# =============================================================================

def prepare_vllm_request(
    processor, prompt_text: str, image_path: str
) -> Dict[str, Any]:
    messages    = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt": text_prompt, "image_path": image_path}


def execute_vllm_batch(
    llm: LLM, requests: List[Dict[str, Any]], sampling_params: SamplingParams
) -> List[str]:
    prompts_for_vllm = []
    for req in requests:
        try:
            image_pil = Image.open(req["image_path"]).convert("RGB")
            prompts_for_vllm.append({"prompt": req["prompt"], "multi_modal_data": {"image": image_pil}})
        except FileNotFoundError:
            prompts_for_vllm.append({"prompt": "Error: image not found.", "multi_modal_data": None})
        except Exception:
            prompts_for_vllm.append({"prompt": "Error: image read error.", "multi_modal_data": None})

    if not prompts_for_vllm:
        return []
    outputs = llm.generate(prompts_for_vllm, sampling_params)
    return [o.outputs[0].text if o.outputs else "" for o in outputs]


# =============================================================================
# Resume helper
# =============================================================================

def load_processed_ids(out_path: str) -> set:
    """Return the set of sample IDs already written to `out_path`."""
    processed = set()
    if not os.path.exists(out_path):
        return processed
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                uid = obj.get("pid") or obj.get("image")
                if uid is not None:
                    processed.add(str(uid))
            except Exception:
                continue
    return processed


# =============================================================================
# Main
# =============================================================================

def main():
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    llm       = LLM(
        model=MODEL_DIR,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=20480,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)

    sampling_split = SamplingParams(max_tokens=MAX_TOKENS_SPLIT, temperature=TEMPERATURE)
    sampling_judge = SamplingParams(max_tokens=MAX_TOKENS_JUDGE, temperature=TEMPERATURE)

    if not os.path.exists(INPUT_FILE_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE_PATH}")

    with open(INPUT_FILE_PATH, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    processed_ids  = load_processed_ids(OUTPUT_FILE_PATH)
    data_to_process = []
    for obj in all_data:
        uid = str(obj.get("pid") or obj.get("image") or "")
        if uid not in processed_ids:
            data_to_process.append(obj)

    if not data_to_process:
        print("All samples already processed. Nothing to do.")
        return

    print(f"Total={len(all_data)}, to_process={len(data_to_process)}, already_done={len(processed_ids)}")

    with open(OUTPUT_FILE_PATH, "a", encoding="utf-8") as f_out:
        for sample in tqdm(data_to_process, desc="Evaluating samples"):
            try:
                question    = str(sample.get("question", "") or "")
                image_rel   = str(sample.get("image",    "") or "")
                image_path  = os.path.join(IMAGE_ROOT, image_rel)

                # Retrieve the structured image description stored by the
                # perception_intervention_inference.py pipeline.
                structured_msg = (
                    sample.get("perception_intervention_meta", {})
                          .get("api_raw", {})
                          .get("structured", {})
                          .get("message", "")
                )
                image_description_text = str(structured_msg or "").strip()
                if not image_description_text:
                    image_description_text = "(no structured description available)"

                exps = sample.get("perception_intervention_exps", {})
                if not isinstance(exps, dict):
                    exps = {}
                    sample["perception_intervention_exps"] = exps

                # -----------------------------------------------------------------
                # Stage A: split each condition's reasoning into steps (batch)
                # -----------------------------------------------------------------
                split_requests: List[Dict[str, Any]] = []
                split_map:      List[Tuple[str, str]] = []

                for cond_key in CONDITION_KEYS:
                    cond_obj = exps.get(cond_key)
                    if not isinstance(cond_obj, dict):
                        continue
                    reasoning = str(cond_obj.get("reasoning", "") or "").strip()
                    if not reasoning:
                        continue
                    prompt = PROMPT_SPLIT_REASONING_TO_STEPS.format(
                        question_text=question,
                        reasoning_text=reasoning,
                    )
                    split_requests.append(prepare_vllm_request(processor, prompt, image_path))
                    split_map.append((cond_key, reasoning))

                split_outputs: List[str] = []
                for i in range(0, len(split_requests), INFERENCE_BATCH_SIZE):
                    split_outputs.extend(
                        execute_vllm_batch(llm, split_requests[i:i + INFERENCE_BATCH_SIZE], sampling_split)
                    )

                for raw_out, (cond_key, reasoning) in zip(split_outputs, split_map):
                    cond_obj = exps.setdefault(cond_key, {})
                    cond_obj.setdefault("prm_eval", {})
                    cond_obj["prm_eval"]["raw_split_output"] = raw_out

                    parsed = parse_json_array(raw_out)
                    steps  = normalize_steps(parsed, reasoning) if parsed is not None \
                             else rule_split_steps(raw_out, reasoning)

                    cond_obj["prm_eval"]["steps"]            = steps
                    cond_obj["prm_eval"]["raw_judge_outputs"] = [""] * len(steps)
                    cond_obj["prm_eval"]["step_scores"]       = [-1] * len(steps)
                    exps[cond_key] = cond_obj

                # -----------------------------------------------------------------
                # Stage B: judge each step (batch across all conditions)
                # -----------------------------------------------------------------
                judge_requests: List[Dict[str, Any]] = []
                judge_map:      List[Tuple[str, int]] = []

                for cond_key in CONDITION_KEYS:
                    cond_obj = exps.get(cond_key)
                    if not isinstance(cond_obj, dict):
                        continue
                    prm_eval = cond_obj.get("prm_eval", {})
                    steps    = prm_eval.get("steps", [])
                    if not isinstance(steps, list) or not steps:
                        continue

                    for s_idx, cur_step in enumerate(steps):
                        history = truncate_history(steps[:s_idx], HISTORY_MAX_STEPS_FOR_JUDGE)
                        prompt  = PROMPT_STEP_JUDGE.format(
                            question_text=question,
                            image_description_text=image_description_text,
                            history_steps_text=json.dumps(history, ensure_ascii=False),
                            current_step_text=str(cur_step),
                        )
                        judge_requests.append(prepare_vllm_request(processor, prompt, image_path))
                        judge_map.append((cond_key, s_idx))

                judge_outputs: List[str] = []
                for i in range(0, len(judge_requests), INFERENCE_BATCH_SIZE):
                    judge_outputs.extend(
                        execute_vllm_batch(llm, judge_requests[i:i + INFERENCE_BATCH_SIZE], sampling_judge)
                    )

                for raw_out, (cond_key, s_idx) in zip(judge_outputs, judge_map):
                    cond_obj = exps.get(cond_key, {})
                    prm_eval = cond_obj.get("prm_eval", {})
                    steps    = prm_eval.get("steps", [])
                    if not isinstance(steps, list) or not (0 <= s_idx < len(steps)):
                        continue
                    prm_eval["raw_judge_outputs"][s_idx] = raw_out
                    prm_eval["step_scores"][s_idx]       = parse_step_score(raw_out)
                    cond_obj["prm_eval"] = prm_eval
                    exps[cond_key]       = cond_obj

                sample["perception_intervention_exps"] = exps

                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                f_out.flush()

            except Exception as e:
                sample.setdefault("__error__", {})["exception"] = repr(e)
                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                f_out.flush()
                continue

    print(f"[Done] Results saved to: {OUTPUT_FILE_PATH}")


if __name__ == "__main__":
    main()
