# -*- coding: utf-8 -*-
"""
Comprehensive ablation studies on VisualProcessBench.

This script runs a comprehensive suite of ablation experiments on
VisualProcessBench to identify which components of EVPV-PRM are responsible
for its step-verification and reranking gains.

Each ablation varies one aspect of the pipeline while holding all others fixed:
  - Evidence type: structured constraints, caption-only, empty, shuffled, noisy.
  - Modality: image + JSON, text-only + JSON, text-only only.
  - Vision prompt: base (detailed) vs. short (concise).
  - Judge prompt: strict, lenient, or no-vision.
  - Step history: none, last-k, or full.
  - Sampling temperature for vision and judge stages.
  - Parse-failure policy: default to -1, to +1, or random.

Architecture note:
  Each ablation runs in its own spawned subprocess so that the vLLM engine
  is fully released after each run. This prevents resource contention when
  many configurations are run sequentially.

Output:
  One JSONL file per ablation configuration, saved to OUTPUT_DIR.
  Each record contains the original sample plus:
    `response.process_correctness`: List[int]  (predicted step labels)
    `ablation_name`:                str
    `ablation_meta`:                dict        (full ablation config)
    `structured_vision`:            dict        (extracted constraints)

Requirements:
  pip install vllm transformers pillow
"""

import os
import re
import json
import multiprocessing
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from PIL import Image

from .prompts import (
    STRUCTURED_VISION_SYSTEM as STRUCTURED_VISION_SYSTEM_BASE,
    JUDGE_PREFIX_STRICT,
    JUDGE_PREFIX_LENIENT,
    JUDGE_PREFIX_NO_VISION,
    structured_vision_user as structured_vision_user_base,
    structured_vision_user_short,
)
from .utils import (
    abs_image_paths,
    safe_load_rgb,
    parse_vision_json,
    parse_label_1_or_minus1,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---- Global path and model configuration ----
# Update these paths to point to your local data and model directories.
IN_PATH    = "data/visualprocessbench/visualprocessbench.jsonl"
IMG_ROOT   = "data/visualprocessbench/images"
OUTPUT_DIR = "output/ablation_results"
MODEL_DIR  = "EVPV-PRM"
MAX_MODEL_LEN = 8096

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------- Prompt builder -----------------------

def build_judge_user_content(
    question: str,
    vision_json: Optional[Dict[str, Any]],
    prev_steps: List[str],
    cur_step: str,
    prefix: str,
    include_vision: bool,
) -> str:
    parts = [prefix]
    if include_vision and vision_json is not None:
        parts.append("Structured image description (JSON): " + json.dumps(vision_json, ensure_ascii=False))
    parts.append(f"Question: {question}")
    if prev_steps:
        parts.append("Previous solution steps:")
        parts.extend([f"{i+1}. {s}" for i, s in enumerate(prev_steps)])
    parts.append("Step to evaluate:")
    parts.append(cur_step)
    return "\n".join(parts)


def atomic_append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def normalize_steps(steps: Any) -> List[str]:
    if not isinstance(steps, list):
        return []
    return [str(s) for s in steps]


def corrupt_vision_json(vision_json: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Apply a deterministic corruption to the constraint JSON for ablation."""
    v = dict(vision_json) if isinstance(vision_json, dict) else {"caption": "", "key_facts": [], "uncertain": []}
    v.setdefault("caption", "")
    v.setdefault("key_facts", [])
    v.setdefault("uncertain", [])

    if mode == "empty":
        return {"caption": "", "key_facts": [], "uncertain": ["ablated_empty"]}
    if mode == "drop_facts":
        return {**v, "key_facts": [], "uncertain": v["uncertain"] + ["ablated_drop_facts"]}
    if mode == "shuffle_facts":
        import random
        kf = list(v.get("key_facts", []))
        random.Random(12345).shuffle(kf)
        return {**v, "key_facts": kf, "uncertain": v["uncertain"] + ["ablated_shuffle_facts"]}
    if mode == "keep_caption_only":
        return {"caption": v.get("caption", ""), "key_facts": [], "uncertain": ["ablated_caption_only"]}
    if mode == "noise_caption":
        return {**v, "caption": "The image contains some geometric and numerical information.",
                "uncertain": v["uncertain"] + ["ablated_noise_caption"]}
    return v


def maybe_truncate_history(prev: List[str], mode: str) -> List[str]:
    if mode == "none":
        return []
    if mode == "full":
        return prev
    if mode.startswith("last"):
        try:
            k = int(mode.replace("last", ""))
            return prev[-k:] if k > 0 else []
        except Exception:
            return prev
    return prev


# ----------------------- vLLM Wrappers -----------------------

def vllm_generate_one(llm, processor, raw_image: Optional[Image.Image], messages, sampling_params) -> str:
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if raw_image is None:
        outputs = llm.generate(prompts=[{"prompt": text_prompt}], sampling_params=sampling_params)
    else:
        outputs = llm.generate(
            prompts=[{"prompt": text_prompt, "multi_modal_data": {"image": raw_image}}],
            sampling_params=sampling_params,
        )
    return outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""


def get_structured_vision(
    llm, processor, raw_image: Image.Image, question: str,
    sampling_params, sys_prompt: str, user_prompt_variant: str, vision_disabled: bool,
) -> Dict[str, Any]:
    if vision_disabled:
        return {"caption": "", "key_facts": [], "uncertain": ["vision_disabled"]}
    user_text = (
        structured_vision_user_base(question)
        if user_prompt_variant == "base"
        else structured_vision_user_short(question)
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {"role": "user",   "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
    ]
    return parse_vision_json(vllm_generate_one(llm, processor, raw_image, messages, sampling_params))


def judge_step(
    llm, processor, raw_image: Optional[Image.Image], question: str,
    vision_json: Optional[Dict[str, Any]], prev_steps: List[str], cur_step: str,
    sampling_params, prefix: str, include_vision: bool, parse_fail_policy: str,
) -> int:
    user_text = build_judge_user_content(
        question=question, vision_json=vision_json, prev_steps=prev_steps,
        cur_step=cur_step, prefix=prefix, include_vision=include_vision,
    )
    if raw_image is None:
        messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
    else:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}]

    out   = vllm_generate_one(llm, processor, raw_image, messages, sampling_params)
    label = parse_label_1_or_minus1(out)
    if label in (1, -1):
        return int(label)
    if parse_fail_policy == "to_1":
        return 1
    if parse_fail_policy == "random":
        import random
        return 1 if random.random() < 0.5 else -1
    return -1


# ----------------------- Ablation Configuration -----------------------

@dataclass
class Ablation:
    name: str

    vision_disabled:      bool  = False
    vision_sys_prompt:    str   = STRUCTURED_VISION_SYSTEM_BASE
    vision_user_variant:  str   = "base"              # "base" | "short"
    vision_corrupt_mode:  str   = "none"              # "none" | "empty" | "drop_facts" |
                                                       # "shuffle_facts" | "keep_caption_only" |
                                                       # "noise_caption"

    judge_prefix_mode:       str  = "strict"          # "strict" | "lenient" | "novision"
    include_vision_in_judge: bool = True
    include_image_in_judge:  bool = True

    history_mode:         str   = "full"              # "none" | "last1/2/4/8" | "full"

    vision_temp:          float = 0.2
    vision_top_p:         float = 0.9
    judge_temp:           float = 0.0
    judge_top_p:          float = 1.0

    parse_fail_policy:    str   = "to_-1"             # "to_-1" | "to_1" | "random"
    multi_image_mode:     str   = "first"             # "first" | "none"


def build_ablation_suite() -> List[Ablation]:
    suite: List[Ablation] = [Ablation(name="baseline")]

    suite += [
        # Evidence type ablations
        Ablation(name="no_structured_vision",           vision_disabled=True, include_vision_in_judge=False),
        Ablation(name="vision_prompt_short",            vision_user_variant="short"),
        Ablation(name="vision_corrupt_empty",           vision_corrupt_mode="empty"),
        Ablation(name="vision_corrupt_drop_facts",      vision_corrupt_mode="drop_facts"),
        Ablation(name="vision_corrupt_shuffle_facts",   vision_corrupt_mode="shuffle_facts"),
        Ablation(name="vision_corrupt_caption_only",    vision_corrupt_mode="keep_caption_only"),
        Ablation(name="vision_corrupt_noise_caption",   vision_corrupt_mode="noise_caption"),

        # Vision sampling temperature
        Ablation(name="vision_temp_0_0",   vision_temp=0.0, vision_top_p=1.0),
        Ablation(name="vision_temp_0_5",   vision_temp=0.5, vision_top_p=0.9),
        Ablation(name="vision_top_p_0_7",  vision_temp=0.2, vision_top_p=0.7),

        # Modality ablations
        Ablation(name="judge_text_only_keep_visionjson",
                 include_image_in_judge=False, include_vision_in_judge=True),
        Ablation(name="judge_text_only_no_visionjson",
                 include_image_in_judge=False, include_vision_in_judge=False, judge_prefix_mode="novision"),
        Ablation(name="judge_no_visionjson_with_image",
                 include_image_in_judge=True,  include_vision_in_judge=False, judge_prefix_mode="strict"),

        # Judge prompt ablations
        Ablation(name="judge_prefix_lenient",  judge_prefix_mode="lenient"),
        Ablation(name="judge_prefix_novision", judge_prefix_mode="novision", include_vision_in_judge=False),
        Ablation(name="judge_temp_0_2",        judge_temp=0.2, judge_top_p=1.0),
        Ablation(name="judge_temp_0_5",        judge_temp=0.5, judge_top_p=0.9),

        # History length ablations
        Ablation(name="history_none",   history_mode="none"),
        Ablation(name="history_last1",  history_mode="last1"),
        Ablation(name="history_last2",  history_mode="last2"),
        Ablation(name="history_last4",  history_mode="last4"),
        Ablation(name="history_last8",  history_mode="last8"),

        # Parse-failure policy
        Ablation(name="parse_fail_to_1",   parse_fail_policy="to_1"),
        Ablation(name="parse_fail_random", parse_fail_policy="random"),

        # Multi-image mode
        Ablation(name="multi_image_none", multi_image_mode="none"),

        # Compound ablations
        Ablation(
            name="combo_no_visionjson_text_only",
            vision_disabled=True, include_image_in_judge=False,
            include_vision_in_judge=False, judge_prefix_mode="novision",
        ),
        Ablation(
            name="combo_vision_corrupt_dropfacts_text_only",
            vision_corrupt_mode="drop_facts", include_image_in_judge=False,
            include_vision_in_judge=True,
        ),
        Ablation(
            name="combo_history_none_no_visionjson",
            include_vision_in_judge=False, history_mode="none",
        ),
        Ablation(
            name="combo_judge_temp_0_5_history_last2",
            judge_temp=0.5, judge_top_p=0.9, history_mode="last2",
        ),
    ]
    return suite


def judge_prefix_from_mode(mode: str) -> str:
    if mode == "lenient":
        return JUDGE_PREFIX_LENIENT
    if mode == "novision":
        return JUDGE_PREFIX_NO_VISION
    return JUDGE_PREFIX_STRICT


# ----------------------- Worker: one ablation per subprocess -----------------------

def _run_one_ablation_worker(ab_dict: Dict[str, Any]) -> None:
    """
    Runs a single ablation in a spawned subprocess.
    vLLM is imported and initialized here so that GPU resources are
    fully released when the subprocess exits.
    """
    ab       = Ablation(**ab_dict)
    out_path = os.path.join(OUTPUT_DIR, f"{ab.name}.jsonl")
    print(f"\n=== Ablation: {ab.name} ===")
    print(f"Output: {out_path}")

    done = count_lines(out_path)
    if done > 0:
        print(f"[Resume] {done} lines already written; skipping them.")

    import torch  # noqa
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_DIR,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        disable_log_stats=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)

    vision_params = SamplingParams(max_tokens=512, temperature=ab.vision_temp, top_p=ab.vision_top_p)
    judge_params  = SamplingParams(max_tokens=8,   temperature=ab.judge_temp,  top_p=ab.judge_top_p)

    with open(IN_PATH, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            if idx < done:
                continue
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            img_paths = abs_image_paths(obj.get("image"), img_root=IMG_ROOT)
            raw_image: Optional[Image.Image] = None
            if ab.multi_image_mode != "none":
                if img_paths and os.path.exists(img_paths[0]):
                    raw_image = safe_load_rgb(img_paths[0])

            question = obj.get("question", "")
            steps    = normalize_steps((obj.get("response") or {}).get("steps", []))

            obj.setdefault("response", {})
            obj["response"].setdefault("steps", steps)

            if not steps:
                obj["response"]["process_correctness"]       = []
                obj["response"]["process_correctness_error"] = "no_steps"
                obj["ablation_name"] = ab.name
                atomic_append_jsonl(out_path, obj)
                continue

            if raw_image is None:
                vision_json = {"caption": "", "key_facts": [], "uncertain": ["no_image_available"]}
            else:
                try:
                    vision_json = get_structured_vision(
                        llm, processor, raw_image, question, vision_params,
                        sys_prompt=ab.vision_sys_prompt,
                        user_prompt_variant=ab.vision_user_variant,
                        vision_disabled=ab.vision_disabled,
                    )
                except Exception as e:
                    vision_json = {"caption": "", "key_facts": [], "uncertain": [f"vision_call_failed:{type(e).__name__}"]}

            if ab.vision_corrupt_mode != "none":
                vision_json = corrupt_vision_json(vision_json, ab.vision_corrupt_mode)

            preds:  List[int] = []
            prev:   List[str] = []
            prefix = judge_prefix_from_mode(ab.judge_prefix_mode)

            for s in steps:
                cur       = str(s)
                prev_used = maybe_truncate_history(prev, ab.history_mode)
                include_vision = ab.include_vision_in_judge and (ab.judge_prefix_mode != "novision")
                judge_image    = raw_image if ab.include_image_in_judge else None

                try:
                    lab = judge_step(
                        llm, processor, judge_image, question,
                        vision_json if include_vision else None,
                        prev_used, cur, judge_params,
                        prefix=prefix,
                        include_vision=include_vision,
                        parse_fail_policy=ab.parse_fail_policy,
                    )
                except Exception:
                    lab = 1 if ab.parse_fail_policy == "to_1" else -1

                preds.append(int(lab))
                prev.append(cur)

            obj["response"]["process_correctness"] = preds
            obj["ablation_name"] = ab.name
            obj["ablation_meta"] = asdict(ab)
            obj["structured_vision"] = vision_json

            atomic_append_jsonl(out_path, obj)

            if (idx % 50) == 0:
                print(f"[{ab.name}] idx={idx}, steps={len(steps)}, 1={preds.count(1)}, -1={preds.count(-1)}")

    print(f"Ablation done: {ab.name} -> {out_path}")


# ----------------------- Main -----------------------

def run_ablation_in_subprocess(ab: Ablation) -> None:
    """Run a single ablation in an isolated subprocess to free GPU memory after each run."""
    p = multiprocessing.Process(target=_run_one_ablation_worker, args=(asdict(ab),), daemon=False)
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"Ablation '{ab.name}' failed (exit code {p.exitcode}).")


def main():
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"Input file not found: {IN_PATH}")
    if not os.path.exists(IMG_ROOT):
        raise FileNotFoundError(f"Image root directory not found: {IMG_ROOT}")

    suite = build_ablation_suite()
    print(f"Total ablation configurations: {len(suite)}")
    print(f"Output directory: {OUTPUT_DIR}")

    for ab in suite:
        run_ablation_in_subprocess(ab)

    print("All ablations complete.")


if __name__ == "__main__":
    main()
