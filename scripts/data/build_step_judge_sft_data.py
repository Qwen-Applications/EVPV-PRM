"""
Build step-discrimination SFT data from swiftdata + image-description JSONL.
Calls LLM API to get per-step correctness and error category; outputs SFT JSONL.
Supports resume via progress file. Configure paths and API via environment variables.
"""

import base64
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, IO, List, Optional, Set, Tuple

import requests
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuration (override via env)
# -----------------------------------------------------------------------------
MAX_WORKERS = int(os.environ.get("STEP_JUDGE_NUM_WORKERS", "16"))
MAX_RETRIES = int(os.environ.get("STEP_JUDGE_MAX_RETRIES", "3"))
RETRY_DELAY = int(os.environ.get("STEP_JUDGE_RETRY_DELAY", "2"))

OUTPUT_BASE_DIR = os.environ.get("OUTPUT_BASE_DIR", "/path/to/evpv_data")
DATA_SUB_DIRS_STR = os.environ.get(
    "DATA_SUB_DIRS",
    "Geo170K,GeometryData,GeomVerse,GEOS,MAVIS-Geometry,TabMWP,UniGeo",
)
DIRECTORIES_TO_PROCESS = [
    os.path.join(OUTPUT_BASE_DIR, d.strip())
    for d in DATA_SUB_DIRS_STR.split(",")
    if d.strip()
]
INPUT_FILENAME = os.environ.get("STEP_JUDGE_INPUT_FILENAME", "swiftdata_image_describe.jsonl")
OUTPUT_FILENAME = os.environ.get(
    "STEP_JUDGE_OUTPUT_FILENAME",
    os.path.join(OUTPUT_BASE_DIR, "sft_processed_data_multithread.jsonl"),
)
PROGRESS_FILENAME = os.environ.get(
    "STEP_JUDGE_PROGRESS_FILE",
    os.path.join(OUTPUT_BASE_DIR, "processing_progress.json"),
)

API_URL = os.environ.get("LLM_API_URL", "")
API_TOKEN = os.environ.get("LLM_API_TOKEN", "")
API_MODEL = os.environ.get("LLM_API_MODEL", "qwen3-vl-235b-a22b-instruct")

# Logging
for h in list(logging.root.handlers):
    logging.root.removeHandler(h)
log_formatter = logging.Formatter(
    "%(asctime)s - %(threadName)s - %(levelname)s - %(message)s"
)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)
log_file = os.environ.get(
    "STEP_JUDGE_LOG_FILE",
    os.path.join(OUTPUT_BASE_DIR, "processing_multithread.log"),
)
os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

file_write_lock = threading.Lock()
progress_lock = threading.Lock()


def generate_line_id(line: str) -> str:
    """Generate unique id for a line (MD5 hash)."""
    return hashlib.md5(line.encode("utf-8")).hexdigest()


def load_processed_ids() -> Set[str]:
    """Load set of already processed line IDs from progress file."""
    if os.path.exists(PROGRESS_FILENAME):
        try:
            with open(PROGRESS_FILENAME, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data.get("processed_ids", []))
        except Exception as e:
            logging.warning("Could not load progress file: %s. Starting fresh.", e)
    return set()


def save_processed_id(line_id: str, processed_ids: Set[str]) -> None:
    """Append line_id to processed set and write progress file."""
    with progress_lock:
        processed_ids.add(line_id)
        try:
            with open(PROGRESS_FILENAME, "w", encoding="utf-8") as f:
                json.dump({"processed_ids": list(processed_ids)}, f)
        except Exception as e:
            logging.error("Failed to save progress file: %s", e)


def request_llm(
    prompt: List,
    params: Dict,
    model: str = API_MODEL,
    max_retries: int = MAX_RETRIES,
) -> Tuple[Optional[str], Dict]:
    """Call LLM API with retries. Returns (message_text, full_response_dict)."""
    url = API_URL
    headers = {"Content-Type": "application/json", "token": API_TOKEN}
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "params": params,
    })
    extra = os.environ.get("LLM_API_EXTRA_JSON", "")
    if extra:
        try:
            payload_dict = json.loads(payload)
            payload_dict.update(json.loads(extra))
            payload = json.dumps(payload_dict)
        except json.JSONDecodeError:
            pass

    for attempt in range(max_retries):
        try:
            response = requests.post(
                url, headers=headers, data=payload, timeout=180
            )
            response.raise_for_status()
            response_dict = response.json()
            if "data" in response_dict and "message" in response_dict["data"]:
                return response_dict["data"]["message"], response_dict
            logging.warning(
                "API returned unexpected response (attempt %s/%s): %s",
                attempt + 1,
                max_retries,
                response_dict,
            )
        except requests.exceptions.RequestException as e:
            logging.error(
                "Request error (attempt %s/%s): %s",
                attempt + 1,
                max_retries,
                e,
            )
        if attempt < max_retries - 1:
            time.sleep(RETRY_DELAY * (attempt + 1))
    return None, {}


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """Encode image file to Base64 string. image_path can be relative to CWD or absolute."""
    if not os.path.isabs(image_path):
        # Try under each DATA dir
        for base in DIRECTORIES_TO_PROCESS:
            full = os.path.join(base, image_path)
            if os.path.exists(full):
                image_path = full
                break
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        logging.error("Image not found: %s", image_path)
        return None
    except Exception as e:
        logging.error("Error encoding image %s: %s", image_path, e)
        return None


def create_llm_prompt(
    question_and_solution: str, gpt_judgement: str, image_describe: str
) -> str:
    """Build the prompt for step-level correctness and error category."""
    judgement_instruction = (
        "The overall solution is marked as correct ('+'). Therefore, you must evaluate every step as correct (score: 1, category: null)."
        if gpt_judgement == "+"
        else "The overall solution is marked as incorrect ('-'). You must carefully analyze each step to identify the specific error(s). Assign a score of 0 and the appropriate error category to the faulty step(s). All other steps should be marked as correct (score: 1)."
    )
    return f"""
You are an expert in process supervision for problem-solving. Your task is to analyze a given problem, its proposed solution, and an accompanying image. You must break down the solution into logical steps and evaluate each step's correctness.

**Input Provided:**
1.  **Question and Solution:** A problem description followed by a step-by-step solution process.
2.  **Overall Judgement:** A simple '+' (correct) or '-' (incorrect) label for the entire solution.
3.  **Image Description:** A text description of the visual elements in the image.
4.  **Image:** The actual image associated with the problem.

**Your Task:**
1.  Parse the 'Solution Process' section from the provided text into a list of individual steps.
2.  Analyze each step for correctness based on the question, the image, and logical reasoning.
3.  Use the provided 'Overall Judgement' to guide your analysis: {judgement_instruction}
4.  For each step that is incorrect, you must assign one of the following error categories.

**Error Categories (for incorrect steps only):**
1.  **Visual Misinterpretation**: The step contradicts visual facts in the image. (Highest Priority)
    -   *1.1: Object Misidentification* (e.g., calling a square a circle).
    -   *1.2: Value Misreading* (e.g., reading a label '50' as '5').
    -   *1.3: Structural Misunderstanding* (e.g., misinterpreting the spatial relationship between objects).
2.  **Logical Error**: The reasoning is flawed, even if the premises are correct.
3.  **Calculation Error**: A mistake in mathematical computation.
4.  **Knowledge Error**: An incorrect formula or theorem is used.
5.  **Incompleteness**: The step is not wrong but omits a critical piece of information.

**Output Format:**
You MUST respond with a single, valid JSON object and nothing else. Do not add explanations or markdown. The JSON object must have the following structure:
{{
    "question": "The original question text, extracted from the input.",
    "Solution_steps": [
        "Step 1 text here.",
        "Step 2 text here.",
        "..."
    ],
    "step_judge_result": [
        {{ "step": "Step 1 text here", "score": 1, "category": "" }},
        {{ "step": "Step 2 text here", "score": -1, "category": "Error Categories" }},
        "..."
    ]
}}

---
**Overall Judgement:** {gpt_judgement}

**Image Description:** {image_describe}

**Question and Solution:**
{question_and_solution}
"""


def clean_llm_response(
    response: str, max_attempts: int = 4
) -> Optional[Dict]:
    """Parse JSON from LLM response with several fallback strategies."""
    if not response:
        return None
    # Try 1: strip markdown code block
    try:
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        out = json.loads(cleaned.strip())
        if all(k in out for k in ("question", "Solution_steps", "step_judge_result")):
            return out
    except (json.JSONDecodeError, AttributeError):
        pass
    # Try 2: regex for JSON object
    try:
        json_pattern = r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}"
        for match in re.findall(json_pattern, response, re.DOTALL):
            try:
                parsed = json.loads(match)
                if all(
                    k in parsed
                    for k in ("question", "Solution_steps", "step_judge_result")
                ):
                    return parsed
            except json.JSONDecodeError:
                continue
    except Exception:
        pass
    # Try 3: first { to last }
    try:
        start_idx = response.find("{")
        end_idx = response.rfind("}")
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            return json.loads(response[start_idx : end_idx + 1])
    except (json.JSONDecodeError, AttributeError):
        pass
    logging.error(
        "All JSON parse attempts failed. Response prefix: %s",
        response[:200],
    )
    return None


def process_line_task(line: str, line_id: str) -> Optional[Tuple[str, str]]:
    """Process one line: call API, build SFT record. Returns (line_id, json_str) or None."""
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        logging.warning("Skip line (JSON decode error): %s...", line[:100])
        return None

    image_path = data.get("image")
    image_describe = data.get("image_describe", "")
    conversations = data.get("conversations", [])
    human_convo = next((c for c in conversations if c.get("from") == "human"), None)
    gpt_convo = next((c for c in conversations if c.get("from") == "gpt"), None)
    if not all([image_path, human_convo, gpt_convo]):
        logging.warning("Skip line (missing fields). id=%s", data.get("id"))
        return None

    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return None

    question_and_solution = human_convo.get("value")
    gpt_judgement = gpt_convo.get("value")
    image_format = os.path.splitext(image_path)[1][1:].lower() or "jpeg"
    llm_prompt_text = create_llm_prompt(
        question_and_solution, gpt_judgement, image_describe
    )
    message_prompt = [
        {
            "role": "system",
            "content": "You are a helpful and meticulous expert assistant following instructions precisely.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": llm_prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_format};base64,{base64_image}"
                    },
                },
            ],
        },
    ]
    params = {"top_p": 0.9, "temperature": 0.1, "max_tokens": 8192}

    response_text, _ = request_llm(prompt=message_prompt, params=params)
    if not response_text:
        logging.error("LLM API call failed for image: %s", image_path)
        return None

    llm_output_dict = clean_llm_response(response_text)
    if not llm_output_dict:
        logging.error("Could not parse LLM response for image: %s", image_path)
        return None

    try:
        user_content = (
            f"Question: {llm_output_dict['question']}\n\n"
            f"Image Describe: {image_describe}\n\n"
            f"Solution Steps:\n"
            + "\n".join(
                f"{i+1}. {step}"
                for i, step in enumerate(llm_output_dict["Solution_steps"])
            )
        )
        assistant_content = json.dumps(
            llm_output_dict["step_judge_result"], ensure_ascii=False
        )
        sft_record = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "images": [image_path],
        }
        return (line_id, json.dumps(sft_record, ensure_ascii=False))
    except (KeyError, TypeError) as e:
        logging.error("LLM response format mismatch: %s. Output: %s", e, llm_output_dict)
        return None


def worker(
    line: str,
    line_id: str,
    output_file: IO,
    pbar: tqdm,
    success_counter: List[int],
    failure_counter: List[int],
    processed_ids: Set[str],
) -> None:
    """One worker: process line and write result under lock."""
    result = process_line_task(line, line_id)
    with file_write_lock:
        if result:
            _, result_json_string = result
            output_file.write(result_json_string + "\n")
            output_file.flush()
            success_counter[0] += 1
            save_processed_id(line_id, processed_ids)
        else:
            failure_counter[0] += 1
        pbar.update(1)
        pbar.set_postfix(
            Success=success_counter[0], Failed=failure_counter[0]
        )


def main() -> None:
    if not API_URL or not API_TOKEN:
        raise ValueError(
            "Set LLM_API_URL and LLM_API_TOKEN (e.g. from config.env)."
        )

    processed_ids = load_processed_ids()
    logging.info("Loaded %s processed record IDs.", len(processed_ids))

    all_lines = []
    line_ids = []
    for directory in DIRECTORIES_TO_PROCESS:
        input_file_path = os.path.join(directory, INPUT_FILENAME)
        if not os.path.exists(input_file_path):
            continue
        try:
            with open(input_file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                lid = generate_line_id(line)
                if lid not in processed_ids:
                    all_lines.append(line)
                    line_ids.append(lid)
            logging.info(
                "From %s: %s lines read, %s new.",
                input_file_path,
                len(lines),
                sum(1 for ln in lines if generate_line_id(ln) not in processed_ids),
            )
        except Exception as e:
            logging.error("Could not read %s: %s", input_file_path, e)

    total = len(all_lines)
    if total == 0:
        logging.warning("No lines to process. Exit.")
        return

    logging.info("Total lines to process: %s", total)
    success_counter = [0]
    failure_counter = [0]
    file_mode = "a" if os.path.exists(OUTPUT_FILENAME) else "w"
    os.makedirs(os.path.dirname(OUTPUT_FILENAME) or ".", exist_ok=True)

    with open(OUTPUT_FILENAME, file_mode, encoding="utf-8") as output_file:
        with tqdm(total=total, desc="Progress") as pbar:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [
                    executor.submit(
                        worker,
                        line,
                        lid,
                        output_file,
                        pbar,
                        success_counter,
                        failure_counter,
                        processed_ids,
                    )
                    for line, lid in zip(all_lines, line_ids)
                ]
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        logging.error("Task exception: %s", e)
                        failure_counter[0] += 1

    logging.info("Done. Success: %s, Failed: %s.", success_counter[0], failure_counter[0])
    logging.info("Output: %s. Progress: %s.", OUTPUT_FILENAME, PROGRESS_FILENAME)


if __name__ == "__main__":
    main()
