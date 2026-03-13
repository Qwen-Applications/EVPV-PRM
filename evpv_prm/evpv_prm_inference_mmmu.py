# -*- coding: utf-8 -*-
"""
EVPV-PRM inference pipeline for MMMU (multi-image benchmark).

This script is the MMMU-adapted variant of evpv_prm_inference.py. The primary
difference is support for multi-image inputs: each MMMU sample may contain
multiple images (stored under `image`, `images`, or `image_paths`), and all
images are passed together to the model.

Pipeline (same three stages as evpv_prm_inference.py):
  Stage 1 - Image description (multi-image aware).
  Stage 2 - Visual checklist evaluation per response.
  Stage 3 - Step-level reward per response.

Resume support: completed samples are identified by pid and skipped.

Requirements:
  pip install vllm transformers pillow tqdm

Input:
  - JSONL file produced by policy_inference_mmmu.py (multi-image records).

Output:
  - JSONL file with evaluation results appended (same schema as
    evpv_prm_inference.py output).
"""

import os
import json
import multiprocessing
from typing import List, Dict, Any, Optional, Tuple, Union

from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from .prompts import (
    PROMPT_GENERATE_IMAGE_DESCRIPTION,
    PROMPT_VISUAL_CHECKLIST_EVALUATION,
    PROMPT_STEP_REWARD,
)
from .utils import (
    strip_code_fences,
    extract_first_balanced_json,
    json_loads_lenient,
    parse_step_score as parse_single_judge_int,
)


# ---- Path and parameter configuration ----
# Update these paths to point to your local data and model directories.
MODEL_DIR        = "EVPV-PRM"
INPUT_FILE_PATH  = "data/mmmu/policy_outputs_mmmu.jsonl"
IMAGE_ROOT       = "data/benchmark/images"
OUTPUT_FILE_PATH = "output/evpv_prm_results_mmmu.jsonl"

INFERENCE_BATCH_SIZE                 = 8
TEMPERATURE                          = 0.1
MAX_TOKENS_STAGE12                   = 2048
MAX_TOKENS_STAGE3                    = 256
HISTORY_MAX_STEPS_FOR_STAGE3         = 12
REUSE_IMAGE_DESCRIPTION_PER_SAMPLE   = True


import re
_float_re = re.compile(r"[-+]?(?:\d+\.\d+|\d+\.|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def clean_and_parse_json(raw_text: str, is_array: bool = False) -> Optional[Any]:
    if not raw_text:
        return None
    candidate = extract_first_balanced_json(raw_text, want_array=is_array)
    if candidate:
        parsed = json_loads_lenient(candidate)
        if parsed is not None:
            return parsed
    if is_array:
        cand_obj = extract_first_balanced_json(raw_text, want_array=False)
        if cand_obj:
            obj = json_loads_lenient(cand_obj)
            if isinstance(obj, dict):
                for k in ("scores", "step_scores", "raw_step_scores", "result", "results", "output"):
                    if k in obj and isinstance(obj[k], list):
                        return obj[k]
    return None


def safe_float(x: Any, default: float = -1.0) -> float:
    if x is None:
        return default
    if isinstance(x, (int, float)):
        try:
            return float(x)
        except Exception:
            return default
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return default
        try:
            return float(s)
        except Exception:
            m = _float_re.search(s)
            if m:
                try:
                    return float(m.group(0))
                except Exception:
                    return default
            return default
    try:
        return float(x)
    except Exception:
        return default


def norm_visual_dep(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() == "null":
            return None
        return s
    s = str(v).strip()
    if not s or s.lower() == "null":
        return None
    return s


# ----------------------- Multi-image Path Resolution -----------------------

def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def get_sample_image_paths(sample: Dict[str, Any], image_root: str) -> List[str]:
    """
    Resolve image paths from a sample supporting three field names:
    - sample["image"]: str or list[str]
    - sample["images"]: list[str]
    - sample["image_paths"]: list[str]
    Returns absolute paths joined with image_root.
    """
    if "image_paths" in sample:
        candidates = _as_list(sample.get("image_paths"))
    elif "images" in sample:
        candidates = _as_list(sample.get("images"))
    else:
        candidates = _as_list(sample.get("image"))

    paths: List[str] = []
    for rel in candidates:
        if rel is None:
            continue
        rel = str(rel).strip()
        if not rel:
            continue
        p = rel if os.path.isabs(rel) else os.path.join(image_root, rel)
        paths.append(p)

    seen  = set()
    dedup = []
    for p in paths:
        if p not in seen:
            dedup.append(p)
            seen.add(p)
    return dedup


# ----------------------- vLLM Multi-image Batch Inference -----------------------

def prepare_vllm_request_multi(
    processor, prompt_text: str, image_paths: Union[str, List[str]]
) -> Dict[str, Any]:
    if isinstance(image_paths, str):
        image_paths_list = [image_paths]
    else:
        image_paths_list = list(image_paths)

    content = [{"type": "image"} for _ in image_paths_list]
    content.append({"type": "text", "text": prompt_text})
    messages    = [{"role": "user", "content": content}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt": text_prompt, "image_paths": image_paths_list}


def execute_vllm_batch(llm: LLM, requests: List[Dict[str, Any]], sampling_params: SamplingParams) -> List[str]:
    prompts_for_vllm = []
    for req in requests:
        image_paths = req.get("image_paths") or []
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        images_pil = []
        ok = True
        for p in image_paths:
            try:
                images_pil.append(Image.open(p).convert("RGB"))
            except Exception:
                ok = False
                break
        if not image_paths or not ok or not images_pil:
            prompts_for_vllm.append({"prompt": "Error: Image not found/read error.", "multi_modal_data": None})
        else:
            prompts_for_vllm.append({"prompt": req["prompt"], "multi_modal_data": {"image": images_pil}})
    if not prompts_for_vllm:
        return []
    outputs = llm.generate(prompts_for_vllm, sampling_params)
    return [o.outputs[0].text if o.outputs else "" for o in outputs]


# ----------------------- Resume Support -----------------------

def load_processed_pids(out_path: str) -> set:
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
                pid = obj.get("pid", obj.get("image"))
                if pid is not None:
                    processed.add(str(pid))
            except Exception:
                continue
    return processed


# ----------------------- Visual Checklist Utilities -----------------------

def build_visual_checklist_from_reasoning(reasoning_steps: List[Dict[str, Any]]) -> Tuple[str, List[int]]:
    deps           = []
    visual_indices = []
    for idx, st in enumerate(reasoning_steps):
        dep = norm_visual_dep(st.get("visualdependency"))
        if dep is not None:
            deps.append(dep)
            visual_indices.append(idx)
    seen  = set()
    dedup = []
    for d in deps:
        if d not in seen:
            dedup.append(d)
            seen.add(d)
    checklist_text = "(empty)" if not dedup else "\n".join([f"- {d}" for d in dedup])
    return checklist_text, visual_indices


def truncate_history(history_steps: List[str], max_steps: int) -> List[str]:
    if max_steps is None or max_steps <= 0:
        return history_steps
    if len(history_steps) <= max_steps:
        return history_steps
    return history_steps[-max_steps:]


# ----------------------- Main -----------------------

def main():
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    llm = LLM(
        model=MODEL_DIR,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=20480,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)

    sampling_stage12 = SamplingParams(max_tokens=MAX_TOKENS_STAGE12, temperature=TEMPERATURE)
    sampling_stage3  = SamplingParams(max_tokens=MAX_TOKENS_STAGE3,  temperature=TEMPERATURE)

    if not os.path.exists(INPUT_FILE_PATH):
        raise FileNotFoundError(INPUT_FILE_PATH)
    with open(INPUT_FILE_PATH, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    processed_pids  = load_processed_pids(OUTPUT_FILE_PATH)
    data_to_process = [obj for obj in all_data if str(obj.get("pid", obj.get("image"))) not in processed_pids]

    if not data_to_process:
        print("All done. Nothing to process.")
        return

    print(f"Total={len(all_data)}, to_process={len(data_to_process)}, already={len(processed_pids)}")
    out_dir = os.path.dirname(OUTPUT_FILE_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(OUTPUT_FILE_PATH, "a", encoding="utf-8") as f_out:
        for sample in tqdm(data_to_process, desc="Processing samples"):
            try:
                query        = sample.get("question", "")
                image_paths  = get_sample_image_paths(sample, IMAGE_ROOT)
                if not image_paths:
                    raise ValueError("No image paths found in sample.")

                # ---- Stage 1: Image description ----
                image_description = None
                if REUSE_IMAGE_DESCRIPTION_PER_SAMPLE:
                    prompt1 = PROMPT_GENERATE_IMAGE_DESCRIPTION.format(question_text=query)
                    raw1    = execute_vllm_batch(
                        llm, [prepare_vllm_request_multi(processor, prompt1, image_paths)], sampling_stage12
                    )[0]
                    parsed1 = clean_and_parse_json(raw1, is_array=False)
                    image_description = (
                        parsed1["image_description"]
                        if parsed1 and isinstance(parsed1, dict) and "image_description" in parsed1
                        else "Error: Failed to generate image description."
                    )
                    sample["__golden_image_description__"]     = image_description
                    sample["__golden_image_description_raw__"] = raw1

                # ---- Stage 2: Visual checklist evaluation ----
                stage2_requests = []
                stage2_map      = []

                for r in range(1, 9):
                    resp_key = f"vlmresponse{r}"
                    resp     = sample.get(resp_key)
                    if not isinstance(resp, dict):
                        continue
                    steps = resp.get("reasoningprocess", [])
                    if not isinstance(steps, list) or not steps:
                        continue
                    checklist_text, visual_indices = build_visual_checklist_from_reasoning(steps)
                    prompt2 = PROMPT_VISUAL_CHECKLIST_EVALUATION.format(
                        golden_standard_text=image_description,
                        checklist_to_review_text=checklist_text,
                    )
                    stage2_requests.append(prepare_vllm_request_multi(processor, prompt2, image_paths))
                    stage2_map.append((resp_key, checklist_text, visual_indices))

                stage2_outputs = []
                for i in range(0, len(stage2_requests), INFERENCE_BATCH_SIZE):
                    stage2_outputs.extend(
                        execute_vllm_batch(llm, stage2_requests[i: i + INFERENCE_BATCH_SIZE], sampling_stage12)
                    )

                for raw_out, (resp_key, checklist_text, visual_indices) in zip(stage2_outputs, stage2_map):
                    resp    = sample.get(resp_key, {})
                    if not isinstance(resp, dict):
                        continue
                    parsed2 = clean_and_parse_json(raw_out, is_array=False)
                    if parsed2 and isinstance(parsed2, dict) and "p_score" in parsed2:
                        p_score = safe_float(parsed2.get("p_score"), default=0.0)
                        if not (0.0 <= p_score <= 1.0):
                            p_score = 0.0
                    else:
                        p_score = 0.0
                        parsed2 = {"error": "JSON parsing failed", "raw_output": raw_out, "p_score": p_score}

                    resp.setdefault("eval", {})
                    resp["eval"]["image_description"]           = image_description
                    resp["eval"]["visual_checklist"]            = checklist_text
                    resp["eval"]["visual_step_indices"]         = visual_indices
                    resp["eval"]["visual_checklist_evaluation"] = parsed2
                    resp["eval"]["local_p_score"]               = p_score
                    sample[resp_key] = resp

                # ---- Stage 3: Step-level rewards ----
                stage3_requests = []
                stage3_map      = []

                for r in range(1, 9):
                    resp_key = f"vlmresponse{r}"
                    resp     = sample.get(resp_key)
                    if not isinstance(resp, dict):
                        continue
                    steps = resp.get("reasoningprocess", [])
                    if not isinstance(steps, list) or not steps:
                        continue
                    if "eval" not in resp or "image_description" not in resp["eval"]:
                        continue

                    img_desc   = resp["eval"]["image_description"]
                    step_texts = [str(st.get("steptext", "")).strip() for st in steps]

                    for s_idx, cur_step in enumerate(step_texts):
                        history = truncate_history(step_texts[:s_idx], HISTORY_MAX_STEPS_FOR_STAGE3)
                        prompt3 = PROMPT_STEP_REWARD.format(
                            question_text=query,
                            image_description_text=img_desc,
                            history_steps_text=json.dumps(history, ensure_ascii=False, indent=2),
                            current_step_text=cur_step,
                        )
                        stage3_requests.append(prepare_vllm_request_multi(processor, prompt3, image_paths))
                        stage3_map.append((resp_key, s_idx))

                stage3_outputs = []
                for i in range(0, len(stage3_requests), INFERENCE_BATCH_SIZE):
                    stage3_outputs.extend(
                        execute_vllm_batch(llm, stage3_requests[i: i + INFERENCE_BATCH_SIZE], sampling_stage3)
                    )

                resp_to_scores:  Dict[str, List[float]] = {}
                resp_to_nsteps:  Dict[str, int]         = {}
                for r in range(1, 9):
                    resp_key = f"vlmresponse{r}"
                    resp     = sample.get(resp_key)
                    if isinstance(resp, dict) and isinstance(resp.get("reasoningprocess"), list):
                        resp_to_nsteps[resp_key] = len(resp["reasoningprocess"])
                for resp_key, n in resp_to_nsteps.items():
                    resp_to_scores[resp_key] = [-1.0] * n

                for raw_out, (resp_key, s_idx) in zip(stage3_outputs, stage3_map):
                    v = parse_single_judge_int(raw_out, default=-1)
                    if resp_key in resp_to_scores and 0 <= s_idx < len(resp_to_scores[resp_key]):
                        resp_to_scores[resp_key][s_idx] = float(v)

                # ---- Aggregate and write output ----
                for r in range(1, 9):
                    resp_key = f"vlmresponse{r}"
                    resp     = sample.get(resp_key)
                    if not isinstance(resp, dict):
                        continue
                    steps = resp.get("reasoningprocess", [])
                    if not isinstance(steps, list) or not steps:
                        continue
                    if resp_key not in resp_to_scores:
                        continue

                    raw_scores = resp_to_scores[resp_key]
                    if len(raw_scores) != len(steps):
                        if len(raw_scores) > len(steps):
                            raw_scores = raw_scores[: len(steps)]
                        else:
                            raw_scores = raw_scores + [-1.0] * (len(steps) - len(raw_scores))

                    p_score        = safe_float(resp.get("eval", {}).get("local_p_score", 0.0), default=0.0)
                    if not (0.0 <= p_score <= 1.0):
                        p_score = 0.0
                    visual_indices = resp.get("eval", {}).get("visual_step_indices", [])
                    if not isinstance(visual_indices, list):
                        visual_indices = []

                    final_scores = list(raw_scores)
                    for idx in visual_indices:
                        if 0 <= idx < len(final_scores) and final_scores[idx] != -1.0:
                            final_scores[idx] = final_scores[idx] * p_score

                    resp.setdefault("eval", {})
                    resp["eval"]["raw_step_scores"]   = raw_scores
                    resp["eval"]["final_step_scores"] = final_scores
                    sample[resp_key] = resp

                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                f_out.flush()

            except Exception as e:
                sample.setdefault("__error__", {})
                sample["__error__"]["exception"] = repr(e)
                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                f_out.flush()
                continue

    print(f"Done. Saved to: {OUTPUT_FILE_PATH}")


if __name__ == "__main__":
    main()
