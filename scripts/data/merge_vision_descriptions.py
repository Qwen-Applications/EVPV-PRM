"""
Merge image descriptions from sft_vision_data.jsonl into swiftdata.jsonl per subset dir.
Produces swiftdata_image_describe.jsonl in each dir for use by build_step_judge_sft_data.py.
Configure OUTPUT_BASE_DIR and DATA_SUB_DIRS via environment (see config.example.env).
"""

import json
import os
from tqdm import tqdm

OUTPUT_BASE_DIR = os.environ.get("OUTPUT_BASE_DIR", "/path/to/evpv_data")
DATA_SUB_DIRS_STR = os.environ.get(
    "DATA_SUB_DIRS",
    "Geo170K,GeometryData,TabMWP,UniGeo,GeomVerse,GEOS,MAVIS-Geometry",
)
DATA_SUB_DIRS = [d.strip() for d in DATA_SUB_DIRS_STR.split(",") if d.strip()]
SFT_VISION_FILE = os.path.join(OUTPUT_BASE_DIR, "sft_vision_data.jsonl")
SWIFTDATA_FILENAME = "swiftdata.jsonl"
OUTPUT_FILENAME = "swiftdata_image_describe.jsonl"


def main():
    if not os.path.exists(SFT_VISION_FILE):
        raise FileNotFoundError(
            f"Run build_vision_sft_data.py first to create {SFT_VISION_FILE}"
        )

    # Build mapping: unique_image_identifier -> image_description
    id_to_desc = {}
    with open(SFT_VISION_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading vision SFT data"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                images = data.get("images") or []
                messages = data.get("messages") or []
                if not images or len(messages) < 2:
                    continue
                key = images[0]
                desc = messages[1].get("content", "")
                if isinstance(desc, str):
                    id_to_desc[key] = desc
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(id_to_desc)} image descriptions.")

    for dir_name in DATA_SUB_DIRS:
        source_dir = os.path.join(OUTPUT_BASE_DIR, dir_name)
        swiftdata_path = os.path.join(source_dir, SWIFTDATA_FILENAME)
        out_path = os.path.join(source_dir, OUTPUT_FILENAME)
        if not os.path.exists(swiftdata_path):
            print(f"Skip {dir_name}: {swiftdata_path} not found.")
            continue
        merged_count = 0
        with open(swiftdata_path, "r", encoding="utf-8") as f_in:
            lines = f_in.readlines()
        with open(out_path, "w", encoding="utf-8") as f_out:
            for line in tqdm(lines, desc=f"Merge {dir_name}"):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    img_rel = rec.get("image") or ""
                    unique_id = os.path.join(dir_name, img_rel)
                    rec["image_describe"] = id_to_desc.get(unique_id, "")
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    merged_count += 1
                except json.JSONDecodeError:
                    continue
        print(f"  {dir_name}: wrote {merged_count} lines to {OUTPUT_FILENAME}.")

    print("Done.")


if __name__ == "__main__":
    main()
