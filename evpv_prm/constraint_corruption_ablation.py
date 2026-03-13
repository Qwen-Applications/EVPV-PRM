# -*- coding: utf-8 -*-
"""
Constraint quality ablation: causal impact of constraint fidelity on step verification.

This script provides causal evidence that EVPV-PRM's verification gains stem
from constraint fidelity rather than incidental prompt effects. It does so by
injecting controlled noise into the structured visual constraints extracted
from each image, then measuring how step-verification accuracy degrades as
noise increases.

Two noise types are evaluated across six noise ratios (0.1 to 1.0):

  DROP – randomly remove a fraction of constraint items
         (caption and individual key_facts).
  FLIP – randomly corrupt a fraction of constraint items by altering one
         numeric token (e.g., 5 → 7.5) or one relation token (e.g., = → <).

For each sample the script also produces a noise-free baseline prediction.
All conditions are stored in a single output record per sample under
`constraint_experiments`, enabling downstream analysis of the
constraint-quality vs. performance curve.

Requirements:
  pip install vllm transformers pillow

Input:
  - VisualProcessBench JSONL file with step-level ground-truth labels.

Output:
  - JSONL file, each record augmented with:
      "structured_vision":    dict   (noise-free constraints)
      "constraint_experiments": dict  (baseline + drop/flip results per ratio)
      "step_predict":         List[int] (baseline step predictions)
"""

import os
import re
import json
import random
import multiprocessing
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

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


# ----------------------- Constraint Corruption -----------------------

_NUM_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")
_REL_RE = re.compile(r"(<=|>=|!=|=|<|>)")


def _flip_number_token(s: str, rng: random.Random) -> str:
    """Replace one numeric token in `s` with a perturbed value."""
    nums = list(_NUM_RE.finditer(s))
    if not nums:
        return s
    m = rng.choice(nums)
    token = m.group(0)
    try:
        val = float(token)
    except Exception:
        return s

    if abs(val) < 1e-9:
        new_val = rng.choice([1.0, 2.0, 3.0, 5.0, 10.0])
    else:
        factor  = rng.choice([0.5, 0.8, 1.2, 1.5, 2.0])
        new_val = val * factor + rng.choice([-2, -1, 1, 2])

    new_token = str(int(round(new_val))) if ("." not in token and float(int(val)) == val) \
                else f"{new_val:.3g}"
    return s[:m.start()] + new_token + s[m.end():]


def _flip_relation_token(s: str, rng: random.Random) -> str:
    """Replace one relation symbol in `s` with a different relation."""
    rels = list(_REL_RE.finditer(s))
    if not rels:
        return s
    m          = rng.choice(rels)
    old        = m.group(0)
    candidates = [c for c in ["<", ">", "<=", ">=", "=", "!="] if c != old]
    new        = rng.choice(candidates)
    return s[:m.start()] + new + s[m.end():]


def flip_constraint_text(s: str, rng: random.Random) -> str:
    """Flip either a numeric or a relation token in `s` (prefer numeric)."""
    if _NUM_RE.search(s):
        return _flip_number_token(s, rng)
    if _REL_RE.search(s):
        return _flip_relation_token(s, rng)
    return s


def drop_constraints(
    vision_json: Dict[str, Any],
    drop_ratio: float,
    rng: random.Random,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Randomly drop constraint items:
    - caption: dropped with probability `drop_ratio`.
    - key_facts[i]: each item dropped independently with probability `drop_ratio`.
    Returns (corrupted_json, metadata).
    """
    v   = deepcopy(vision_json)
    meta: Dict[str, Any] = {"type": "drop", "ratio": drop_ratio}

    cap = str(v.get("caption", "") or "")
    cap_dropped = cap and rng.random() < drop_ratio
    v["caption"] = "" if cap_dropped else cap

    kf = v.get("key_facts", [])
    if not isinstance(kf, list):
        kf = [str(kf)]
    new_kf, dropped_idx = [], []
    for i, item in enumerate(kf):
        if item is None:
            continue
        it = str(item)
        if it and rng.random() < drop_ratio:
            dropped_idx.append(i)
        else:
            new_kf.append(it)
    v["key_facts"] = new_kf

    u = v.get("uncertain", [])
    v["uncertain"] = [str(x) for x in (u if isinstance(u, list) else [u]) if x is not None]

    meta.update({
        "caption_dropped": cap_dropped,
        "dropped_key_facts_idx": dropped_idx,
        "key_facts_before": len(kf),
        "key_facts_after": len(new_kf),
    })
    return v, meta


def flip_constraints(
    vision_json: Dict[str, Any],
    flip_ratio: float,
    rng: random.Random,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Randomly flip constraint items:
    - caption: flipped with probability `flip_ratio`.
    - key_facts[i]: each item flipped independently with probability `flip_ratio`.
    Flipping alters one numeric or relation token in the text.
    Returns (corrupted_json, metadata).
    """
    v   = deepcopy(vision_json)
    meta: Dict[str, Any] = {"type": "flip", "ratio": flip_ratio}

    cap = str(v.get("caption", "") or "")
    cap_flipped = False
    if cap and rng.random() < flip_ratio:
        new_cap     = flip_constraint_text(cap, rng)
        cap_flipped = new_cap != cap
        v["caption"] = new_cap
    else:
        v["caption"] = cap

    kf = v.get("key_facts", [])
    if not isinstance(kf, list):
        kf = [str(kf)]
    new_kf, flipped_idx = [], []
    for i, item in enumerate(kf):
        if item is None:
            continue
        it = str(item)
        if it and rng.random() < flip_ratio:
            new_it = flip_constraint_text(it, rng)
            if new_it != it:
                flipped_idx.append(i)
            new_kf.append(new_it)
        else:
            new_kf.append(it)
    v["key_facts"] = new_kf

    u = v.get("uncertain", [])
    v["uncertain"] = [str(x) for x in (u if isinstance(u, list) else [u]) if x is not None]

    meta.update({"caption_flipped": cap_flipped, "flipped_key_facts_idx": flipped_idx,
                 "key_facts_count": len(kf)})
    return v, meta


# ----------------------- vLLM Calls -----------------------

def vllm_generate_one(llm, processor, raw_image: Image.Image, messages, sampling_params) -> str:
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = llm.generate(
        prompts=[{"prompt": text_prompt, "multi_modal_data": {"image": raw_image}}],
        sampling_params=sampling_params,
    )
    return outputs[0].outputs[0].text


def get_structured_vision(
    llm, processor, raw_image: Image.Image, question: str, sampling_params
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": STRUCTURED_VISION_SYSTEM}]},
        {"role": "user",   "content": [{"type": "image"}, {"type": "text", "text": structured_vision_user(question)}]},
    ]
    return parse_vision_json(vllm_generate_one(llm, processor, raw_image, messages, sampling_params))


def judge_step(
    llm, processor, raw_image: Image.Image, question: str,
    vision_json: Dict[str, Any], prev_steps: List[str],
    cur_step: str, sampling_params,
) -> int:
    user_text = build_judge_user_content(question, vision_json, prev_steps, cur_step)
    messages  = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}]
    out       = vllm_generate_one(llm, processor, raw_image, messages, sampling_params)
    label     = parse_label_1_or_minus1(out)
    return label if label in (1, -1) else -1


# ----------------------- Experiment Runner -----------------------

def judge_all_steps(
    llm, processor, raw_image: Image.Image, question: str,
    steps: List[str], vision_json: Dict[str, Any], judge_params,
) -> List[int]:
    """Run step-by-step judgment for a full solution trace."""
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
    return preds


# ----------------------- Main -----------------------

def main():
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    # ---- Path configuration ----
    in_path   = "data/visualprocessbench/visualprocessbench.jsonl"
    img_root  = "data/visualprocessbench/images"
    out_path  = "output/constraint_corruption_results.jsonl"
    model_dir = "EVPV-PRM"

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")
    if not os.path.exists(img_root):
        raise FileNotFoundError(f"Image root directory not found: {img_root}")

    # Noise ratios for both DROP and FLIP experiments
    DROP_RATIOS = [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
    FLIP_RATIOS = [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]

    # Fixed seed for reproducibility across runs
    BASE_SEED = 20260130

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
                obj["structured_vision"]       = {"caption": "", "key_facts": [], "uncertain": ["image_missing"]}
                obj["constraint_experiments"]  = {"error": "image_missing"}
                atomic_append_jsonl(out_path, obj)
                continue

            raw_image = safe_load_rgb(img_paths[0])
            question  = obj.get("question", "")
            steps     = (obj.get("response") or {}).get("steps", [])
            if not isinstance(steps, list) or len(steps) == 0:
                obj["structured_vision"]       = {"caption": "", "key_facts": [], "uncertain": ["no_steps"]}
                obj["constraint_experiments"]  = {"error": "no_steps"}
                atomic_append_jsonl(out_path, obj)
                continue

            # Step 1: extract clean constraints
            try:
                vision_json = get_structured_vision(llm, processor, raw_image, question, vision_params)
            except Exception as e:
                vision_json = {"caption": "", "key_facts": [], "uncertain": [f"vision_call_failed:{type(e).__name__}"]}

            # Step 2: baseline + DROP / FLIP ablations
            exp: Dict[str, Any] = {
                "baseline": {"vision_json": vision_json, "step_predict": []},
                "drop":     {},
                "flip":     {},
            }

            sample_id = obj.get("id", idx)
            rng = random.Random(
                (BASE_SEED * 1_000_003) ^ (idx * 9176) ^ (hash(str(sample_id)) & 0xFFFFFFFF)
            )

            exp["baseline"]["step_predict"] = judge_all_steps(
                llm, processor, raw_image, question, steps, vision_json, judge_params
            )

            for r in DROP_RATIOS:
                v_corrupt, meta = drop_constraints(vision_json, r, rng)
                preds = judge_all_steps(llm, processor, raw_image, question, steps, v_corrupt, judge_params)
                exp["drop"][str(r)] = {"meta": meta, "step_predict": preds, "vision_json": v_corrupt}

            for r in FLIP_RATIOS:
                v_corrupt, meta = flip_constraints(vision_json, r, rng)
                preds = judge_all_steps(llm, processor, raw_image, question, steps, v_corrupt, judge_params)
                exp["flip"][str(r)] = {"meta": meta, "step_predict": preds, "vision_json": v_corrupt}

            # Step 3: save
            obj["structured_vision"]      = vision_json
            obj["constraint_experiments"] = exp
            obj["step_predict"]           = exp["baseline"]["step_predict"]

            atomic_append_jsonl(out_path, obj)

            base = exp["baseline"]["step_predict"]
            print(f"[{idx}] saved. steps={len(steps)}, baseline: +1={base.count(1)}, -1={base.count(-1)}")

    print("All done. Output:", out_path)


if __name__ == "__main__":
    main()
