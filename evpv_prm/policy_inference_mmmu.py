# -*- coding: utf-8 -*-
"""
Policy model inference for MMMU (multi-image benchmark).

This script is the MMMU-adapted variant of policy_inference.py. The primary
difference is support for multi-image inputs and a per-item inference timeout
with automatic worker restart to prevent hangs on long generation.

For each question, 8 diverse candidate solutions are generated and stored as
`vlmresponse1`..`vlmresponse8`. Each response contains:
  - `reasoningprocess`: list of steps with `steptext` and `visualdependency`.
  - `finalanswer`: option label (multiple-choice) or numerical result.

Architecture:
  - A dedicated worker subprocess hosts the model; the main process feeds it
    tasks via a queue and collects results with a per-item timeout.
  - If a timeout fires, the worker is killed and restarted for the next item.

Resume support: completed samples (identified by `id`) are skipped.

Requirements:
  pip install lmdeploy pillow

Input:
  - MMMU validation JSONL file (each record: `id`, `question`, `image` or
    list of image paths).

Output:
  - JSONL file, each input record augmented with `vlmresponse1`..`vlmresponse8`.
"""

import os
import re
import json
import time
import random
import signal
import traceback
import multiprocessing as mp
from typing import Dict, Any, Optional, Tuple, List, Union

from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image

from .prompts import create_inference_prompt  # noqa: F401


# ----------------------- JSON Parsing Helpers -----------------------

_JSON_BLOCK_RE     = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


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


# ----------------------- Multi-image Loading -----------------------

def load_images_from_relpaths(
    img_root_dir: str,
    relpaths: Union[str, List[str]],
    max_images: int = 6,
) -> Tuple[Optional[List[Any]], List[str]]:
    """
    Load PIL images from relative paths. Returns (images, missing_paths).
    `missing_paths` is non-empty if any image failed to load.
    """
    if not relpaths:
        return None, []

    if isinstance(relpaths, str):
        relpaths = [relpaths]
    elif not isinstance(relpaths, list):
        relpaths = [str(relpaths)]

    relpaths = [str(rp).strip() for rp in relpaths if str(rp).strip()]
    if max_images and len(relpaths) > max_images:
        relpaths = relpaths[:max_images]

    images  = []
    missing = []
    for rp in relpaths:
        full = os.path.join(img_root_dir, rp)
        if not os.path.exists(full):
            missing.append(full)
            continue
        try:
            images.append(load_image(full))
        except Exception:
            missing.append(full)

    if not images:
        return None, missing
    return images, missing


# ----------------------- Worker Process -----------------------

def worker_loop(task_q: mp.Queue, result_q: mp.Queue, cfg: Dict[str, Any]):
    """Worker subprocess: loads the model once, then processes tasks from the queue."""
    try:
        backend_config = TurbomindEngineConfig(session_len=cfg["SESSION_LEN"], tp=cfg["TP"])
        os.makedirs(cfg["OFFLOAD"], exist_ok=True)
        pipe = pipeline(cfg["MODEL_DIR"], backend_config=backend_config, offload_folder=cfg["OFFLOAD"])
    except Exception as e:
        result_q.put({"_fatal": True, "error": f"Model load failed: {e}", "trace": traceback.format_exc()})
        return

    while True:
        item = task_q.get()
        if item is None:
            return

        pid = item["pid"]
        try:
            responses  = pipe(item["batch_inputs"], gen_config=item["batch_genconfigs"])
            out_texts  = [(getattr(r, "text", "") or "").strip() for r in responses]
            result_q.put({"pid": pid, "ok": True, "texts": out_texts})
        except TypeError:
            try:
                responses = pipe(item["batch_inputs"], gen_config=item["batch_genconfigs"][0])
                out_texts = [(getattr(r, "text", "") or "").strip() for r in responses]
                result_q.put({"pid": pid, "ok": True, "texts": out_texts})
            except Exception as e:
                result_q.put({"pid": pid, "ok": False, "error": str(e), "trace": traceback.format_exc()})
        except Exception as e:
            result_q.put({"pid": pid, "ok": False, "error": str(e), "trace": traceback.format_exc()})


def kill_process(p: mp.Process):
    if p is None or not p.is_alive():
        return
    try:
        os.kill(p.pid, signal.SIGKILL)
    except Exception:
        pass


# ----------------------- Resume Support -----------------------

def load_done_pids(output_path: str) -> set:
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pid = obj.get("id") or obj.get("pid")
                if pid:
                    done.add(pid)
            except Exception:
                continue
    return done


# ----------------------- Main -----------------------

def main():
    mp.set_start_method("spawn", force=True)

    # ---- Path and hardware configuration ----
    # Update these paths to point to your local model and data directories.
    MODEL_DIR       = "/path/to/InternVL-model"
    INPUT_JSONL     = "data/mmmu/mmmu_validation.jsonl"
    OUTPUT_JSONL    = "output/policy_outputs_mmmu.jsonl"
    IMG_ROOT_DIR    = "data/benchmark/images"

    BATCH_N              = 8
    MAX_NEW_TOKENS       = 1024
    PER_ITEM_TIMEOUT     = 600   # seconds; worker is restarted if exceeded
    MAX_IMAGES_PER_SAMPLE = 6
    SESSION_LEN          = 16384
    TP                   = 4     # tensor parallel degree

    BASE_TEMPERATURE = 1.2
    BASE_TOPP        = 0.80

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")
    if not os.path.exists(INPUT_JSONL):
        raise FileNotFoundError(f"Input file not found: {INPUT_JSONL}")

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

    done_pids = load_done_pids(OUTPUT_JSONL)
    if done_pids:
        print(f"[Resume] Found {len(done_pids)} completed records, skipping them.")

    cfg = dict(
        MODEL_DIR=MODEL_DIR,
        SESSION_LEN=SESSION_LEN,
        TP=TP,
        OFFLOAD="/tmp/internvl_offload",
    )

    task_q   = mp.Queue(maxsize=8)
    result_q = mp.Queue(maxsize=8)

    worker = mp.Process(target=worker_loop, args=(task_q, result_q, cfg), daemon=True)
    worker.start()
    print("Worker started (first model load may take a while).")

    with open(INPUT_JSONL, "r", encoding="utf-8") as fin, \
         open(OUTPUT_JSONL, "a", encoding="utf-8") as fout:

        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue

            original_data = json.loads(line)
            pid           = original_data.get("id", f"line_{i+1}")
            if pid in done_pids:
                continue

            query        = original_data.get("question")
            image_relpaths = original_data.get("image")

            if not query:
                original_data["error"] = "Missing question"
                fout.write(json.dumps(original_data, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            print(f"\n--- Processing ID: {pid} ---")
            t0 = time.time()

            image_list_for_model, missing = load_images_from_relpaths(
                IMG_ROOT_DIR, image_relpaths, max_images=MAX_IMAGES_PER_SAMPLE
            )
            if missing:
                print(f"[WARN] {pid}: {len(missing)} image(s) missing/failed (ignored).")
                original_data["_missing_images"] = missing

            if image_list_for_model is None:
                original_data["error"] = f"No valid images. Missing/failed: {missing}"
                fout.write(json.dumps(original_data, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            batch_inputs     = []
            batch_genconfigs = []
            for j in range(BATCH_N):
                seed  = random.randint(1, 2**31 - 1)
                nonce = f"pid={pid}|rep={j+1}|seed={seed}|t={time.time_ns()}"
                prompt = create_inference_prompt(query, nonce=nonce, variant_id=j + 1)
                batch_inputs.append((prompt, image_list_for_model))

                temperature = BASE_TEMPERATURE + random.uniform(-0.15, 0.25)
                topp        = max(0.5, min(0.95, BASE_TOPP + random.uniform(-0.10, 0.10)))
                gen_kwargs  = dict(
                    do_sample=True, temperature=temperature, top_p=topp,
                    max_new_tokens=MAX_NEW_TOKENS, seed=seed,
                    repetition_penalty=1.08, presence_penalty=0.6, frequency_penalty=0.3,
                )
                try:
                    gc = GenerationConfig(**gen_kwargs)
                except TypeError:
                    for k in ["presence_penalty", "frequency_penalty", "repetition_penalty", "seed"]:
                        gen_kwargs.pop(k, None)
                    gc = GenerationConfig(**gen_kwargs)
                batch_genconfigs.append(gc)

            task_q.put({"pid": pid, "batch_inputs": batch_inputs, "batch_genconfigs": batch_genconfigs})

            try:
                res = result_q.get(timeout=PER_ITEM_TIMEOUT)
            except Exception:
                print(f"[TIMEOUT] {pid} exceeded {PER_ITEM_TIMEOUT}s, restarting worker.")
                original_data["error"] = f"Timeout after {PER_ITEM_TIMEOUT}s"
                fout.write(json.dumps(original_data, ensure_ascii=False) + "\n")
                fout.flush()
                kill_process(worker)
                worker.join(timeout=3)
                worker = mp.Process(target=worker_loop, args=(task_q, result_q, cfg), daemon=True)
                worker.start()
                continue

            if res.get("_fatal"):
                raise RuntimeError(f"Worker fatal: {res.get('error')}\n{res.get('trace')}")

            if not res.get("ok"):
                print(f"[ERROR] {pid} inference failed: {res.get('error')}")
                original_data["error"]  = res.get("error")
                original_data["_trace"] = res.get("trace")
                fout.write(json.dumps(original_data, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            for idx, response_text in enumerate(res["texts"]):
                parsed, ok = parse_model_json(response_text)
                if not ok:
                    print(f"  [WARN] Response {idx+1}/{BATCH_N} parse failed; raw saved.")
                original_data[f"vlmresponse{idx+1}"] = parsed

            fout.write(json.dumps(original_data, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"--- ID: {pid} done in {time.time()-t0:.2f}s ---")

    task_q.put(None)
    worker.join(timeout=5)
    kill_process(worker)
    print("\nAll samples processed.")


if __name__ == "__main__":
    main()
