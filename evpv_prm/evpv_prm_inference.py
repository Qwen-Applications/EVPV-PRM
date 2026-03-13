# -*- coding: utf-8 -*-
"""
EVPV-PRM inference pipeline for general multimodal reasoning benchmarks.

This script runs the full EVPV-PRM pipeline on candidate solutions generated
by a policy model (e.g., InternVL2.5) on benchmarks such as MathVista,
MathVision, MathVerse-VO, WeMath, and LogicVista.

Pipeline (three stages per sample):
  Stage 1 - Image description:
      Generate a golden structured image description once per sample.
  Stage 2 - Visual checklist evaluation (per response):
      Extract each response's visual-dependency checklist (from
      `visualdependency` fields), then score the checklist against the golden
      image description to produce a local_p_score in [0, 1].
  Stage 3 - Step-level reward (per response, per step):
      Judge each reasoning step individually (1 = correct, -1 = incorrect).
      For visually dependent steps, multiply the raw score by local_p_score.

Checkpoint design:
  - A JSONL checkpoint file stores intermediate state per sample (pid).
  - A companion index file maps each pid to its byte offset in the checkpoint,
    enabling O(1) in-place updates without rewriting the whole file.
  - On restart, processing resumes from the exact stage/response/step where
    it was interrupted.

Input:
  - JSONL file produced by policy_inference.py (each record contains
    vlmresponse1..8, each with `reasoningprocess` and `finalanswer`).

Output:
  - JSONL file with evaluation results appended to each record:
      vlmresponseN.eval.raw_step_scores:     List[float]
      vlmresponseN.eval.final_step_scores:   List[float]  (reliability-gated)
      vlmresponseN.eval.local_p_score:       float
      vlmresponseN.eval.visual_checklist:    str
      vlmresponseN.eval.visual_step_indices: List[int]
"""

import os
import json
import time
import multiprocessing
from typing import List, Dict, Any, Optional, Tuple

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
MODEL_DIR          = "EVPV-PRM"
INPUT_FILE_PATH    = "data/benchmark/policy_outputs.jsonl"
IMAGE_ROOT         = "data/benchmark/images"
OUTPUT_FILE_PATH   = "output/evpv_prm_results.jsonl"

CHECKPOINT_PATH       = OUTPUT_FILE_PATH + ".ckpt.jsonl"
CHECKPOINT_INDEX_PATH = OUTPUT_FILE_PATH + ".ckpt.index.json"

INFERENCE_BATCH_SIZE             = 8
TEMPERATURE                      = 0.1
MAX_TOKENS_STAGE12               = 2048
MAX_TOKENS_STAGE3                = 256
HISTORY_MAX_STEPS_FOR_STAGE3     = 12
REUSE_IMAGE_DESCRIPTION_PER_SAMPLE = True

# Fixed line size for checkpoint in-place updates (2 MB per snapshot).
# Increase if samples are very large; the code falls back to appending if exceeded.
CKPT_FIXED_LINE_BYTES = 2_000_000


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


# ----------------------- vLLM Batch Inference -----------------------

def prepare_vllm_request(processor, prompt_text: str, image_path: str) -> Dict[str, Any]:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt": text_prompt, "image_path": image_path}


def execute_vllm_batch(llm: LLM, requests: List[Dict[str, Any]], sampling_params: SamplingParams) -> List[str]:
    prompts_for_vllm = []
    for req in requests:
        try:
            image_pil = Image.open(req["image_path"]).convert("RGB")
            prompts_for_vllm.append({"prompt": req["prompt"], "multi_modal_data": {"image": image_pil}})
        except FileNotFoundError:
            prompts_for_vllm.append({"prompt": "Error: Image not found.", "multi_modal_data": None})
        except Exception:
            prompts_for_vllm.append({"prompt": "Error: Image read error.", "multi_modal_data": None})
    if not prompts_for_vllm:
        return []
    outputs = llm.generate(prompts_for_vllm, sampling_params)
    return [o.outputs[0].text if o.outputs else "" for o in outputs]


# ----------------------- Checkpoint Read/Write -----------------------

def _load_json_file(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def load_checkpoint_index(index_path: str) -> Dict[str, Dict[str, Any]]:
    return _load_json_file(index_path, {})


def save_checkpoint_index(index_path: str, index: Dict[str, Dict[str, Any]]) -> None:
    tmp = index_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    os.replace(tmp, index_path)


def ckpt_make_initial(sample: Dict[str, Any]) -> Dict[str, Any]:
    pid = str(sample.get("pid", sample.get("image")))
    return {
        "pid":        pid,
        "image":      sample.get("image"),
        "question":   sample.get("question", ""),
        "updated_at": time.time(),
        "done":       False,
        "stage1": {"done": False, "image_description": None, "raw": None},
        "stage2": {"done": False, "responses": {}},
        "stage3": {"done": False, "responses": {}},
    }


def ckpt_read_one(ckpt_path: str, ckpt_index: Dict[str, Dict[str, Any]], pid: str) -> Optional[Dict[str, Any]]:
    meta = ckpt_index.get(pid)
    if not meta:
        return None
    try:
        with open(ckpt_path, "rb") as f:
            f.seek(int(meta["offset"]))
            b = f.read(int(meta["length"]))
        line = b.decode("utf-8", errors="ignore").rstrip(" \n")
        if not line.strip():
            return None
        return json.loads(line)
    except Exception:
        return None


def ckpt_write_one(
    ckpt_path: str,
    ckpt_index: Dict[str, Dict[str, Any]],
    pid: str,
    obj: Dict[str, Any],
    fixed_line_bytes: int = CKPT_FIXED_LINE_BYTES,
) -> None:
    """In-place update: overwrite existing slot if it fits; otherwise append."""
    obj["updated_at"] = time.time()
    s = json.dumps(obj, ensure_ascii=False)
    b = s.encode("utf-8")

    target_len = fixed_line_bytes
    if len(b) + 1 > target_len:
        target_len = len(b) + 1

    payload = b + b"\n"
    if len(payload) < target_len:
        payload = payload + (b" " * (target_len - len(payload)))

    if not os.path.exists(ckpt_path):
        os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
        open(ckpt_path, "wb").close()

    meta = ckpt_index.get(pid)
    if meta and int(meta.get("length", 0)) >= len(payload):
        with open(ckpt_path, "r+b") as f:
            f.seek(int(meta["offset"]))
            f.write(payload)
        meta["updated_at"] = obj["updated_at"]
        ckpt_index[pid] = meta
        return

    with open(ckpt_path, "ab") as f:
        f.seek(0, os.SEEK_END)
        offset = f.tell()
        f.write(payload)

    ckpt_index[pid] = {"offset": offset, "length": len(payload), "updated_at": obj["updated_at"]}


def load_processed_pids_from_output(out_path: str) -> set:
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
    """Collect unique visualdependency strings and their step indices."""
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

    done_pids   = load_processed_pids_from_output(OUTPUT_FILE_PATH)
    ckpt_index  = load_checkpoint_index(CHECKPOINT_INDEX_PATH)

    out_dir = os.path.dirname(OUTPUT_FILE_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    to_process = [obj for obj in all_data if str(obj.get("pid", obj.get("image"))) not in done_pids]

    if not to_process:
        print("All done. Nothing to process.")
        return

    print(f"Total={len(all_data)}, to_process={len(to_process)}, already_done={len(done_pids)}")

    with open(OUTPUT_FILE_PATH, "a", encoding="utf-8") as f_out:
        for sample in tqdm(to_process, desc="Processing samples"):
            pid = str(sample.get("pid", sample.get("image")))
            if pid in done_pids:
                continue

            image_rel  = sample.get("image", "")
            image_path = os.path.join(IMAGE_ROOT, image_rel)
            query      = sample.get("question", "")

            ckpt_obj = ckpt_read_one(CHECKPOINT_PATH, ckpt_index, pid)
            if ckpt_obj is None:
                ckpt_obj = ckpt_make_initial(sample)
                ckpt_write_one(CHECKPOINT_PATH, ckpt_index, pid, ckpt_obj)
                save_checkpoint_index(CHECKPOINT_INDEX_PATH, ckpt_index)

            try:
                # ---- Stage 1: Image description ----
                if REUSE_IMAGE_DESCRIPTION_PER_SAMPLE and not ckpt_obj["stage1"]["done"]:
                    prompt1  = PROMPT_GENERATE_IMAGE_DESCRIPTION.format(question_text=query)
                    raw1     = execute_vllm_batch(
                        llm, [prepare_vllm_request(processor, prompt1, image_path)], sampling_stage12
                    )[0]
                    parsed1  = clean_and_parse_json(raw1, is_array=False)
                    image_description = (
                        parsed1["image_description"]
                        if parsed1 and isinstance(parsed1, dict) and "image_description" in parsed1
                        else "Error: Failed to generate image description."
                    )
                    ckpt_obj["stage1"].update({"image_description": image_description, "raw": raw1, "done": True})
                    ckpt_write_one(CHECKPOINT_PATH, ckpt_index, pid, ckpt_obj)
                    save_checkpoint_index(CHECKPOINT_INDEX_PATH, ckpt_index)

                image_description = ckpt_obj["stage1"]["image_description"] or "Error: Missing image description."

                # ---- Stage 2: Visual checklist evaluation ----
                ckpt_obj["stage2"].setdefault("responses", {})
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
                    if ckpt_obj["stage2"]["responses"].get(resp_key, {}).get("done", False):
                        continue

                    checklist_text, visual_indices = build_visual_checklist_from_reasoning(steps)
                    prompt2 = PROMPT_VISUAL_CHECKLIST_EVALUATION.format(
                        golden_standard_text=image_description,
                        checklist_to_review_text=checklist_text,
                    )
                    stage2_requests.append(prepare_vllm_request(processor, prompt2, image_path))
                    stage2_map.append((resp_key, checklist_text, visual_indices))

                stage2_outputs = []
                for i in range(0, len(stage2_requests), INFERENCE_BATCH_SIZE):
                    stage2_outputs.extend(
                        execute_vllm_batch(llm, stage2_requests[i: i + INFERENCE_BATCH_SIZE], sampling_stage12)
                    )

                for raw_out, meta in zip(stage2_outputs, stage2_map):
                    resp_key, checklist_text, visual_indices = meta
                    parsed2  = clean_and_parse_json(raw_out, is_array=False)
                    if parsed2 and isinstance(parsed2, dict) and "p_score" in parsed2:
                        p_score = safe_float(parsed2.get("p_score"), default=0.0)
                        if not (0.0 <= p_score <= 1.0):
                            p_score = 0.0
                    else:
                        p_score  = 0.0
                        parsed2  = {"error": "JSON parsing failed", "raw_output": raw_out, "p_score": p_score}

                    ckpt_obj["stage2"]["responses"][resp_key] = {
                        "done": True, "checklist_text": checklist_text,
                        "visual_indices": visual_indices, "parsed": parsed2,
                        "raw": raw_out, "p_score": p_score,
                    }
                    ckpt_write_one(CHECKPOINT_PATH, ckpt_index, pid, ckpt_obj)

                if stage2_map:
                    save_checkpoint_index(CHECKPOINT_INDEX_PATH, ckpt_index)

                all_resp_done = all(
                    ckpt_obj["stage2"]["responses"].get(f"vlmresponse{r}", {}).get("done", False)
                    for r in range(1, 9)
                    if isinstance(sample.get(f"vlmresponse{r}"), dict)
                    and isinstance(sample[f"vlmresponse{r}"].get("reasoningprocess"), list)
                    and sample[f"vlmresponse{r}"]["reasoningprocess"]
                )
                ckpt_obj["stage2"]["done"] = all_resp_done
                ckpt_write_one(CHECKPOINT_PATH, ckpt_index, pid, ckpt_obj)
                save_checkpoint_index(CHECKPOINT_INDEX_PATH, ckpt_index)

                # ---- Stage 3: Step-level rewards ----
                ckpt_obj["stage3"].setdefault("responses", {})
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

                    step_texts = [str(st.get("steptext", "")).strip() for st in steps]
                    n_steps    = len(step_texts)
                    resp_state = ckpt_obj["stage3"]["responses"].setdefault(
                        resp_key, {"n_steps": n_steps, "steps": {}}
                    )
                    resp_state["n_steps"] = n_steps
                    resp_state.setdefault("steps", {})

                    for s_idx, cur_step in enumerate(step_texts):
                        if resp_state["steps"].get(str(s_idx), {}).get("done", False):
                            continue
                        history = truncate_history(step_texts[:s_idx], HISTORY_MAX_STEPS_FOR_STAGE3)
                        prompt3 = PROMPT_STEP_REWARD.format(
                            question_text=query,
                            image_description_text=image_description,
                            history_steps_text=json.dumps(history, ensure_ascii=False, indent=2),
                            current_step_text=cur_step,
                        )
                        stage3_requests.append(prepare_vllm_request(processor, prompt3, image_path))
                        stage3_map.append((resp_key, s_idx))

                stage3_outputs = []
                for i in range(0, len(stage3_requests), INFERENCE_BATCH_SIZE):
                    stage3_outputs.extend(
                        execute_vllm_batch(llm, stage3_requests[i: i + INFERENCE_BATCH_SIZE], sampling_stage12)
                    )

                for raw_out, (resp_key, s_idx) in zip(stage3_outputs, stage3_map):
                    judge      = parse_single_judge_int(raw_out, default=-1)
                    resp_state = ckpt_obj["stage3"]["responses"].setdefault(resp_key, {"n_steps": 0, "steps": {}})
                    resp_state.setdefault("steps", {})[str(s_idx)] = {"done": True, "judge": int(judge), "raw": raw_out}
                    ckpt_write_one(CHECKPOINT_PATH, ckpt_index, pid, ckpt_obj)

                if stage3_map:
                    save_checkpoint_index(CHECKPOINT_INDEX_PATH, ckpt_index)

                ckpt_obj["stage3"]["done"] = True
                ckpt_write_one(CHECKPOINT_PATH, ckpt_index, pid, ckpt_obj)
                save_checkpoint_index(CHECKPOINT_INDEX_PATH, ckpt_index)

                # ---- Stage 4: Write final output ----
                if ckpt_obj.get("done", False):
                    continue
                if not all(ckpt_obj[s]["done"] for s in ("stage1", "stage2", "stage3")):
                    continue

                sample["__golden_image_description__"]     = ckpt_obj["stage1"]["image_description"]
                sample["__golden_image_description_raw__"] = ckpt_obj["stage1"]["raw"]

                for r in range(1, 9):
                    resp_key = f"vlmresponse{r}"
                    resp     = sample.get(resp_key)
                    if not isinstance(resp, dict):
                        continue
                    steps = resp.get("reasoningprocess", [])
                    if not isinstance(steps, list) or not steps:
                        continue

                    s2 = ckpt_obj["stage2"]["responses"].get(resp_key)
                    if not s2:
                        continue

                    resp.setdefault("eval", {})
                    resp["eval"]["image_description"]         = ckpt_obj["stage1"]["image_description"]
                    resp["eval"]["visual_checklist"]          = s2.get("checklist_text")
                    resp["eval"]["visual_step_indices"]       = s2.get("visual_indices", [])
                    resp["eval"]["visual_checklist_evaluation"] = s2.get("parsed")
                    resp["eval"]["local_p_score"]             = s2.get("p_score", 0.0)
                    sample[resp_key] = resp

                resp_to_scores: Dict[str, List[float]] = {}
                for r in range(1, 9):
                    resp_key   = f"vlmresponse{r}"
                    resp       = sample.get(resp_key)
                    if not isinstance(resp, dict):
                        continue
                    steps = resp.get("reasoningprocess", [])
                    if not isinstance(steps, list) or not steps:
                        continue
                    n_steps    = len(steps)
                    resp_state = ckpt_obj["stage3"]["responses"].get(resp_key, {})
                    step_state = resp_state.get("steps", {})
                    scores     = [-1.0] * n_steps
                    for s_idx in range(n_steps):
                        st = step_state.get(str(s_idx), {})
                        if st.get("done", False):
                            scores[s_idx] = float(int(st.get("judge", -1)))
                    resp_to_scores[resp_key] = scores

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

                    raw_scores     = resp_to_scores[resp_key]
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

                ckpt_obj["done"] = True
                ckpt_write_one(CHECKPOINT_PATH, ckpt_index, pid, ckpt_obj)
                save_checkpoint_index(CHECKPOINT_INDEX_PATH, ckpt_index)
                done_pids.add(pid)

            except Exception as e:
                ckpt_obj["error"] = {"exception": repr(e), "time": time.time()}
                ckpt_write_one(CHECKPOINT_PATH, ckpt_index, pid, ckpt_obj)
                save_checkpoint_index(CHECKPOINT_INDEX_PATH, ckpt_index)
                continue

    print(f"Done. Saved to: {OUTPUT_FILE_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
