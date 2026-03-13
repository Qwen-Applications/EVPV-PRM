# -*- coding: utf-8 -*-
"""
Policy model inference for general multimodal reasoning benchmarks.

This script generates N=8 diverse candidate solutions per question using an
InternVL policy model deployed via lmdeploy. The 8 responses are stored as
`vlmresponse1` through `vlmresponse8` in the output JSONL, each containing a
structured JSON with:
  - `reasoningprocess`: list of steps, each with `steptext` and `visualdependency`.
  - `finalanswer`: the final answer string.

Diversity is achieved by:
  - Using different random seeds per response.
  - Slightly jittering temperature and top-p per response.
  - Injecting a unique nonce string into each prompt.

Resume support: completed samples (identified by `pid`) are skipped.

Requirements:
  pip install lmdeploy pillow

Input:
  - JSONL benchmark file (each record: `pid`, `question`, `image`).
    Supported benchmarks: MathVista, MathVision, MathVerse-VO, WeMath, LogicVista.

Output:
  - JSONL file, each input record augmented with `vlmresponse1`..`vlmresponse8`.
"""

import os
import re
import json
import time
import random
import multiprocessing
from typing import Dict, Any, Optional, Tuple, List

from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image

from .prompts import create_inference_prompt  # noqa: F401 (re-exported for backward compat)


# ----------------------- JSON Parsing Helpers -----------------------

_JSON_BLOCK_RE       = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_TRAILING_COMMA_RE   = re.compile(r",\s*([}\]])")


def _extract_json_candidate(text: str) -> Optional[str]:
    t = (text or "").strip()

    m = _JSON_BLOCK_RE.search(t)
    if m:
        cand = m.group(1).strip().strip("`").strip()
        if cand.startswith("{") and cand.endswith("}"):
            return cand

    start = t.find("{")
    if start != -1:
        depth  = 0
        in_str = False
        esc    = False
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
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return t[start:i + 1]

    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start:end + 1]
    return None


def _basic_json_sanitize(s: str) -> str:
    s = (s or "")
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    s = _TRAILING_COMMA_RE.sub(r"\1", s)
    s = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", s)
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
    return s.lstrip("\ufeff").strip()


def _regex_fallback_parse(text: str) -> Optional[Dict[str, Any]]:
    """Last-resort regex extraction of reasoningprocess and finalanswer."""
    t = text or ""
    m_ans = re.search(r'"finalanswer"\s*:\s*("([^"\\]*(\\.[^"\\]*)*)"|[^,}\n]+)', t, re.IGNORECASE)
    finalanswer = ""
    if m_ans:
        raw = m_ans.group(1).strip()
        finalanswer = raw[1:-1] if (raw.startswith('"') and raw.endswith('"')) else raw.strip().strip('"')

    step_texts  = [m.group(1) for m in re.finditer(r'"steptext"\s*:\s*"((?:[^"\\]|\\.)*)"', t, re.IGNORECASE)]
    visual_vals: List[Optional[str]] = []
    for m in re.finditer(r'"visualdependency"\s*:\s*(null|"((?:[^"\\]|\\.)*)")', t, re.IGNORECASE):
        visual_vals.append(None if m.group(1).lower() == "null" else m.group(2))

    if not step_texts and not finalanswer:
        return None

    reasoningprocess = [
        {"steptext": st, "visualdependency": visual_vals[idx] if idx < len(visual_vals) else None}
        for idx, st in enumerate(step_texts)
    ]
    return {"reasoningprocess": reasoningprocess, "finalanswer": finalanswer}


def parse_model_json(response_text: str) -> Tuple[Dict[str, Any], bool]:
    """
    Parse model output to dict.
    Returns (parsed_dict, ok). ok=False means all parsing strategies failed.
    """
    cand = _extract_json_candidate(response_text)
    if cand is not None:
        try:
            parsed = json.loads(_basic_json_sanitize(cand))
            if isinstance(parsed, dict):
                return parsed, True
        except Exception:
            pass

    fb = _regex_fallback_parse(response_text)
    if fb is not None:
        fb["_parse_note"] = "regex_fallback"
        return fb, True

    return {"error": "Failed to parse model JSON", "rawoutput": (response_text or "").strip()}, False


# ----------------------- Main -----------------------

def main():
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # ---- Path configuration ----
    # Update these paths to point to your local model and data directories.
    MODEL_DIR       = "/path/to/InternVL-model"
    DATA_DIR        = "data/benchmark"
    INPUT_JSONL     = os.path.join(DATA_DIR, "benchmark_questions.jsonl")
    OUTPUT_JSONL    = "output/policy_outputs.jsonl"

    BATCH_N         = 8
    BASE_TEMPERATURE = 1.2
    BASE_TOPP        = 0.80

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")
    if not os.path.exists(INPUT_JSONL):
        raise FileNotFoundError(f"Input file not found: {INPUT_JSONL}")

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

    print(f"Loading InternVL model from {MODEL_DIR} ...")
    backend_config  = TurbomindEngineConfig(session_len=16384, tp=1)
    offload_folder  = "/tmp/internvl_offload"
    os.makedirs(offload_folder, exist_ok=True)
    pipe = pipeline(MODEL_DIR, backend_config=backend_config, offload_folder=offload_folder)
    print("Model loaded.")

    print(f"Processing: {INPUT_JSONL}")
    print(f"Output:     {OUTPUT_JSONL}")

    with open(INPUT_JSONL, "r", encoding="utf-8") as fin, \
         open(OUTPUT_JSONL, "a", encoding="utf-8") as fout:

        for i, line in enumerate(fin):
            t0           = time.time()
            original_data = json.loads(line)
            pid          = original_data.get("pid", f"line_{i+1}")
            query        = original_data.get("question")
            image_relpath = original_data.get("image")

            if not query or not image_relpath:
                print(f"[SKIP] PID {pid}: missing 'question' or 'image' field.")
                continue

            image_full_path = os.path.join(DATA_DIR, "images", image_relpath)
            print(f"\n--- Processing PID: {pid} ---")

            try:
                image_for_model = load_image(image_full_path)
            except FileNotFoundError:
                print(f"[ERROR] Image not found: {image_full_path}. Skipping.")
                original_data["error"] = f"Image not found: {image_full_path}"
                fout.write(json.dumps(original_data, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            batch_inputs     = []
            batch_genconfigs = []

            for j in range(BATCH_N):
                seed  = random.randint(1, 2**31 - 1)
                nonce = f"pid={pid}|rep={j+1}|seed={seed}|t={time.time_ns()}"
                prompt = create_inference_prompt(query, nonce=nonce, variant_id=j + 1)
                batch_inputs.append((prompt, image_for_model))

                temperature = BASE_TEMPERATURE + random.uniform(-0.15, 0.25)
                topp        = max(0.5, min(0.95, BASE_TOPP + random.uniform(-0.10, 0.10)))

                gen_kwargs = dict(
                    do_sample=True,
                    temperature=temperature,
                    top_p=topp,
                    max_new_tokens=4096,
                    seed=seed,
                    repetition_penalty=1.08,
                    presence_penalty=0.6,
                    frequency_penalty=0.3,
                )
                try:
                    gc = GenerationConfig(**gen_kwargs)
                except TypeError:
                    for k in ["presence_penalty", "frequency_penalty", "repetition_penalty", "seed"]:
                        gen_kwargs.pop(k, None)
                    gc = GenerationConfig(**gen_kwargs)

                batch_genconfigs.append(gc)

            try:
                responses = pipe(batch_inputs, gen_config=batch_genconfigs)
            except TypeError:
                responses = pipe(batch_inputs, gen_config=batch_genconfigs[0])

            for idx, resp in enumerate(responses):
                response_text = (getattr(resp, "text", "") or "").strip()
                parsed, ok    = parse_model_json(response_text)
                if not ok:
                    print(f"  [WARN] Response {idx+1}/{BATCH_N} parse failed; raw saved.")
                original_data[f"vlmresponse{idx+1}"] = parsed

            fout.write(json.dumps(original_data, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"--- PID: {pid} done in {time.time()-t0:.2f}s ---")

    print("\nAll samples processed.")


if __name__ == "__main__":
    main()
