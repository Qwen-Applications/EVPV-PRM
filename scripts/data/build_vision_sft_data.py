"""
Build vision-understanding SFT data from swiftdata.jsonl in multiple subset dirs.
Uses an optional LLM API to generate image descriptions; outputs sft_vision_data.jsonl.
Configure API and paths via environment variables (see scripts/config.example.env).
"""

import base64
import os
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1. Configuration (override via env)
# -----------------------------------------------------------------------------
NUM_WORKERS = int(os.environ.get("VISION_SFT_NUM_WORKERS", "8"))
API_URL = os.environ.get("LLM_API_URL", "")
# Comma-separated tokens for round-robin across workers
API_TOKENS_STR = os.environ.get("LLM_API_TOKEN", "")
API_TOKENS = [t.strip() for t in API_TOKENS_STR.split(",") if t.strip()]
API_MODEL = os.environ.get("LLM_API_MODEL", "qwen3-vl-235b-a22b-instruct")
API_PARAMS = {"top_p": 0.8, "temperature": 0.7, "max_tokens": 4096}

OUTPUT_BASE_DIR = os.environ.get("OUTPUT_BASE_DIR", "/path/to/evpv_data")
# Subdirs under OUTPUT_BASE_DIR that contain swiftdata.jsonl and image/
DATA_SUB_DIRS = os.environ.get(
    "DATA_SUB_DIRS",
    "Geo170K,GeometryData,TabMWP,UniGeo,GeomVerse,GEOS,MAVIS-Geometry",
).split(",")
DATA_SUB_DIRS = [d.strip() for d in DATA_SUB_DIRS if d.strip()]
OUTPUT_SFT_FILE = os.path.join(OUTPUT_BASE_DIR, "sft_vision_data.jsonl")

# -----------------------------------------------------------------------------
# Prompts (English; no sensitive content)
# -----------------------------------------------------------------------------
API_SYSTEM_PROMPT = """You are a senior mathematics teacher and problem-solving expert. Your task is to create a clear and accurate core information summary for a composite problem that includes both a **textual question** and a **mathematical image**.

This summary will be used as high-quality training data, so you must strictly separate and completely cover both the **textual information** and the **visual information**.

Please strictly follow the two-part structure below for your output, enclosed in a single JSON object.

---

### **Part 1: Question Text**

-   **Task**: Please repeat, **verbatim and in full**, all textual content from the problem, including the question stem, prompts, and all options. The goal here is to ensure the absolute fidelity of the textual information.

### **Part 2: Image Description**

-   **Task**: Now, focus on the image itself. Using **concise, precise, and key-element-focused** natural language, explain the crucial points of the image as if you were explaining it to a student. Your description must cover the following core points:
    1.  **[What is this?]** First, summarize the **overall type and theme** of the image in one sentence.
    2.  **[What are the key elements and data?]** Identify the **main mathematical objects** in the image and clearly list all **directly provided numbers, labels, and symbols**.
    3.  **[What are their important relationships?]** Describe the **key spatial layout and geometric relationships** between these elements, which are critical for understanding the image's logic.

**[Output Format Requirement]**
You MUST output a single valid JSON object with the following structure:
```json
{
  "question_text": "...",
  "image_description": "..."
}
```
Do not add any text before or after the JSON object.
"""

SFT_USER_PROMPT = """You are an expert in mathematical diagrams. Your task is to provide a detailed, single-paragraph description of the provided image, focusing on its core mathematical content. Your description should be structured to answer the following key questions in a natural, flowing manner:

1.  **Overall Identification:** What type of mathematical diagram is this (e.g., a geometric figure, a graph, a Venn diagram)?
2.  **Key Elements and Labels:** What are the primary geometric shapes, lines, points, or other components? List all given labels, numbers, symbols (like angle markers, right-angle symbols), and explicit values.
3.  **Crucial Relationships:** Describe the essential spatial and logical relationships between these elements. For example, mention parallelism, perpendicularity, congruence, tangency, intersections, or how points are situated on lines or curves.

Synthesize these points into a single, comprehensive paragraph that clearly and accurately explains the image for problem-solving purposes.
"""


def image_to_base64_uri(file_path: str) -> Optional[str]:
    """Encode local image file to Base64 data URI."""
    try:
        with open(file_path, "rb") as image_file:
            binary_data = image_file.read()
            base64_string = base64.b64encode(binary_data).decode("utf-8")
        ext = os.path.splitext(file_path)[1].lower()
        mime_type = f"image/{ext[1:]}" if ext else "image/png"
        return f"data:{mime_type};base64,{base64_string}"
    except Exception as e:
        print(f"Error converting image to Base64 for {file_path}: {e}")
        return None


def request_llm_api(
    image_uri: str, question_text: str, api_token: str
) -> Tuple[Optional[str], Dict]:
    """Call multimodal LLM API with the given token."""
    if not image_uri:
        return None, {"error": "Invalid image_uri"}

    headers = {"Content-Type": "application/json", "token": api_token}
    prompt = [
        {"role": "system", "content": API_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Here is the problem text:\n\n{question_text}"},
                {"type": "image_url", "image_url": {"url": image_uri}},
            ],
        },
    ]
    payload = json.dumps({
        "model": API_MODEL,
        "prompt": prompt,
        "params": API_PARAMS,
    })
    # Add optional extra fields from env if your API requires them
    extra = os.environ.get("LLM_API_EXTRA_JSON", "")
    if extra:
        try:
            payload_dict = json.loads(payload)
            payload_dict.update(json.loads(extra))
            payload = json.dumps(payload_dict)
        except json.JSONDecodeError:
            pass

    try:
        response = requests.post(API_URL, headers=headers, data=payload, timeout=180)
        response.raise_for_status()
        response_dict = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request API error: {e}")
        return None, {"error": str(e)}

    if "data" in response_dict and "message" in response_dict["data"]:
        return response_dict["data"]["message"], response_dict
    print(f"API returned unexpected format: {response_dict}")
    return None, response_dict


def parse_model_response(response_text: str) -> Optional[str]:
    """Parse image_description from model response (JSON block or raw JSON)."""
    try:
        match = re.search(r"```json\s*([\s\S]+?)\s*```", response_text)
        if match:
            json_str = match.group(1)
        else:
            start_index = response_text.find("{")
            end_index = response_text.rfind("}")
            if start_index != -1 and end_index != -1 and start_index < end_index:
                json_str = response_text[start_index : end_index + 1]
            else:
                return None
        data = json.loads(json_str)
        if "image_description" in data and isinstance(data["image_description"], str):
            return data["image_description"].strip()
        return None
    except (json.JSONDecodeError, KeyError):
        return None


def process_single_item(
    task: Dict, worker_id: int
) -> Optional[Tuple[str, Dict]]:
    """Process one item: call API and build SFT record. Returns (unique_id, sft_data) or None."""
    full_image_path = task["full_image_path"]
    human_question = task["human_question"]
    unique_image_identifier = task["unique_image_identifier"]

    api_token = API_TOKENS[worker_id % len(API_TOKENS)] if API_TOKENS else ""

    image_uri = image_to_base64_uri(full_image_path)
    if not image_uri:
        return None
    response_text, _ = request_llm_api(image_uri, human_question, api_token)
    if not response_text:
        return None
    image_description = parse_model_response(response_text)
    if not image_description:
        return None

    sft_data = {
        "messages": [
            {"role": "user", "content": SFT_USER_PROMPT},
            {"role": "assistant", "content": image_description},
        ],
        "images": [unique_image_identifier],
    }
    return unique_image_identifier, sft_data


def main():
    if not API_URL or not API_TOKENS:
        raise ValueError(
            "Set LLM_API_URL and LLM_API_TOKEN (or copy config.env from scripts/config.example.env)."
        )

    processed_images = set()
    if os.path.exists(OUTPUT_SFT_FILE):
        print("Loading already processed IDs from existing output...")
        with open(OUTPUT_SFT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("images"):
                        processed_images.add(data["images"][0])
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(processed_images)} processed records.")

    tasks_to_process = []
    print("Collecting tasks...")
    for dir_name in DATA_SUB_DIRS:
        source_dir = os.path.join(OUTPUT_BASE_DIR, dir_name)
        swiftdata_path = os.path.join(source_dir, "swiftdata.jsonl")
        if not os.path.exists(swiftdata_path):
            print(f"Warning: not found, skip dir '{dir_name}'")
            continue
        with open(swiftdata_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                try:
                    data = json.loads(line)
                    if data["conversations"][-1].get("value") != "+":
                        continue
                    original_image_path = data.get("image")
                    if not original_image_path:
                        continue
                    unique_image_identifier = os.path.join(
                        dir_name, original_image_path
                    )
                    if unique_image_identifier in processed_images:
                        continue
                    full_image_path = os.path.join(
                        OUTPUT_BASE_DIR, unique_image_identifier
                    )
                    if not os.path.exists(full_image_path):
                        continue
                    tasks_to_process.append({
                        "full_image_path": full_image_path,
                        "human_question": data["conversations"][1].get(
                            "value", ""
                        ),
                        "unique_image_identifier": unique_image_identifier,
                    })
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    if not tasks_to_process:
        print("No new tasks. Exit.")
        return

    print(f"Collected {len(tasks_to_process)} new tasks. Processing with {NUM_WORKERS} workers...")
    os.makedirs(os.path.dirname(OUTPUT_SFT_FILE) or ".", exist_ok=True)
    with open(OUTPUT_SFT_FILE, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            future_to_task = {
                executor.submit(
                    process_single_item, task, i % NUM_WORKERS
                ): task
                for i, task in enumerate(tasks_to_process)
            }
            for future in tqdm(
                as_completed(future_to_task),
                total=len(tasks_to_process),
                desc="Processing",
            ):
                try:
                    result = future.result()
                    if result:
                        _, sft_data = result
                        f_out.write(
                            json.dumps(sft_data, ensure_ascii=False) + "\n"
                        )
                        f_out.flush()
                except Exception as exc:
                    task_info = future_to_task[future]
                    print(f"Error for {task_info['unique_image_identifier']}: {exc}")

    print(f"Done. SFT data written to: {OUTPUT_SFT_FILE}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Elapsed: {time.time() - start_time:.2f}s")
